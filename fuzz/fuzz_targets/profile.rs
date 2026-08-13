//! A coverage-guided campaign against the profile writers.
//!
//! ```sh
//! cargo +nightly fuzz run profile -- -max_len=8192
//! ```
//!
//! `tests/profile_fuzz.rs` checks the same contract on every `cargo test`, with
//! generators that cannot see inside the writers. This one is driven by
//! coverage, so it finds the inputs that reach a branch nothing else reaches: a
//! module path whose last byte is the first byte of a three-byte sequence, a
//! symbol that renders to exactly the length where a buffer is reused, a
//! program point list that folds to one entry, a counter at `u64::MAX`.
//!
//! # The contract
//!
//! A profile is mostly made of text this crate did not write — a symbol from a
//! symbol table that may be truncated or from another build, a path from the
//! loader, a command line from `argv`, a thread name from the platform. **No
//! text of any shape** may make a writer produce a document a parser refuses, a
//! page that stops parsing halfway, or a line a terminal acts on. Panicking is
//! a finding, and so is hanging or exhausting memory, which libFuzzer reports
//! on its own.
//!
//! # Two oracles, one contract
//!
//! Validity is checked here with `serde_json`, and in `tests/profile_fuzz.rs`
//! with the strict parser in `tests/support/json.rs`. That is deliberate and
//! not redundant. `serde_json` is the parser almost every reader of a profile
//! actually has, so its verdict is the one that decides whether a file opens;
//! the strict parser refuses several things `serde_json` accepts — duplicate
//! keys, leading zeros — because a profile that only parses in a lenient parser
//! is a profile that will not open in the bundled viewer. Neither answers the
//! other's question.
//!
//! The screen is stated below rather than imported, for the reason
//! `tests/support/display.rs` gives about its own copy: a rule checked against
//! its own implementation agrees with any answer that implementation gives,
//! including a future one that quietly stops escaping something.

#![no_main]

use arbitrary::Arbitrary;
use heapscope::output::{
    Counters, FrameFormat, PointKind, ProgramPoint, RegionStats, Snapshot, TallyStats, ThreadStats,
};
use heapscope::symbol::modules::Module;
use libfuzzer_sys::fuzz_target;
use serde_json::Value;

/// Every string a profile carries that came from outside this crate, plus
/// enough shape to drive the emitters through their folding and trimming.
#[derive(Debug, Arbitrary)]
struct Input {
    command: String,
    module_path: String,
    build_id: Option<String>,
    thread_names: Vec<Option<String>>,
    region_names: Vec<Option<String>>,
    /// What the symbol table said this run's frames are called.
    symbol: String,
    /// Return addresses per program point, innermost first. An empty stack is
    /// legal and is the point with no frames to render.
    stacks: Vec<Vec<usize>>,
    counters: Vec<[u64; 9]>,
    /// How many program points the text summary is asked for.
    top: u8,
    trim_frames: bool,
}

/// A frame rendering that returns whatever the symbol table said.
///
/// `FrameFormat` is public and the shipped implementation reads a symbol table,
/// so this is the shape of every real renderer: text of unknown provenance,
/// arriving at an emitter that has to screen it rather than trust it.
struct FromSymbolTable<'a>(&'a str);

impl FrameFormat for FromSymbolTable<'_> {
    fn format(&self, address: usize, out: &mut String) {
        out.push_str(&format!("{address:#x}: {}", self.0));
    }
}

/// Whether a character means something other than itself to a terminal or to a
/// bidirectional renderer, and so must not reach output unescaped.
fn is_escaped(character: char) -> bool {
    let code = character as u32;
    // C0, DEL, C1.
    let control = code < 0x20 || (0x7F..=0x9F).contains(&code);
    // Bidirectional marks, embeddings, overrides and isolates, and the two line
    // separators that are line terminators in JavaScript but not in JSON.
    let reordering = matches!(
        code,
        0x061C | 0x200E | 0x200F | 0x202A..=0x202E | 0x2066..=0x2069 | 0x2028 | 0x2029
    );
    control || reordering
}

/// Parses `document`, and asserts every string in it is one a reader can be
/// shown.
///
/// Parsing first is the point. A JSON `\u2028` escape is invisible in the raw
/// bytes and arrives at the viewer as a line separator all the same, so what
/// gets checked is the decoded string rather than the file.
fn parse_and_screen(document: &str, what: &str) -> Value {
    let value: Value = serde_json::from_str(document)
        .unwrap_or_else(|error| panic!("the {what} is not valid JSON: {error}\n{document}"));
    walk(&value, what);
    value
}

fn walk(value: &Value, what: &str) {
    match value {
        Value::String(text) => {
            let offenders: Vec<char> = text.chars().filter(|&c| is_escaped(c)).collect();
            assert!(
                offenders.is_empty(),
                "the {what} carries {offenders:?} in {text:?}, which a reader's \
                 terminal will act on"
            );
        }
        Value::Array(items) => items.iter().for_each(|item| walk(item, what)),
        Value::Object(members) => {
            for (key, item) in members {
                assert!(
                    !key.chars().any(is_escaped),
                    "the {what} has a key a reader's terminal will act on: {key:?}"
                );
                walk(item, what);
            }
        }
        _ => {}
    }
}

/// The text of the `<script>` block opened by `tag`, exactly as the page carries
/// it.
///
/// Not unescaped afterwards. `\u003c` is valid JSON for `<`, so the block parses
/// as it stands, and undoing the escape by text substitution would corrupt a
/// profile that legitimately contains the six characters `\u003c` — which a
/// symbol or a path really can.
fn block<'a>(page: &'a str, tag: &str) -> &'a str {
    let start = page
        .find(tag)
        .unwrap_or_else(|| panic!("the page has no {tag} block"))
        + tag.len();
    let end = start
        + page[start..]
            .find("</script>")
            .expect("the block is never closed");
    &page[start..end]
}

const PROFILE_BLOCK: &str = r#"<script type="application/json" id="heapscope-profile">"#;
const DISPLAY_BLOCK: &str = r#"<script type="application/json" id="heapscope-display">"#;

fn snapshot(input: &Input) -> Snapshot {
    let mut snapshot = Snapshot::default();

    snapshot.points = input
        .stacks
        .iter()
        .enumerate()
        .map(|(at, frames)| {
            let c = input.counters.get(at).copied().unwrap_or([1024; 9]);
            ProgramPoint {
                kind: PointKind::Recorded,
                frames: frames.clone(),
                counters: Counters {
                    total_bytes: c[0],
                    total_blocks: c[1],
                    total_lifetime: c[2],
                    curr_bytes: c[3],
                    curr_blocks: c[4],
                    max_bytes: c[5],
                    max_blocks: c[6],
                    at_gmax_bytes: c[7],
                    at_gmax_blocks: c[8],
                },
                unretired_lifetime: c[2],
            }
        })
        .collect();

    snapshot.command = input.command.clone();
    snapshot.pid = 4242;
    snapshot.time_at_end = u64::MAX;
    snapshot.settings.trim_frames = input.trim_frames;
    snapshot.modules = vec![Module {
        path: input.module_path.clone(),
        start: 0,
        size: usize::MAX,
        bias: 0,
        image_base: 0,
        build_id: input.build_id.clone(),
    }];
    snapshot.threads = input
        .thread_names
        .iter()
        .enumerate()
        .map(|(at, name)| ThreadStats {
            id: at as u16,
            overflow: false,
            name: name.clone(),
            first_seen: at as u64,
            counts: TallyStats::default(),
        })
        .collect();
    snapshot.regions = input
        .region_names
        .iter()
        .enumerate()
        .map(|(at, name)| RegionStats {
            id: at as u16,
            overflow: false,
            name: name.clone(),
            first_seen: at as u64,
            entries: 1,
            active: 0,
            counts: TallyStats::default(),
        })
        .collect();
    snapshot
}

fuzz_target!(|input: Input| {
    // A snapshot the size of the whole input is the interesting one; a corpus
    // entry that expands into millions of program points only measures how long
    // formatting takes.
    if input.stacks.len() > 256 || input.stacks.iter().any(|stack| stack.len() > 64) {
        return;
    }

    let snapshot = snapshot(&input);
    let format = FromSymbolTable(&input.symbol);

    let mut native = Vec::new();
    snapshot
        .write_native(&mut native)
        .expect("writing to a Vec cannot fail");
    let native = String::from_utf8(native).expect("the profile is UTF-8");
    parse_and_screen(&native, "native profile");

    let mut dhat = Vec::new();
    snapshot
        .write_dhat_v2_with(&mut dhat, &format)
        .expect("writing to a Vec cannot fail");
    let dhat = String::from_utf8(dhat).expect("the profile is UTF-8");
    parse_and_screen(&dhat, "DHAT profile");

    let mut page = Vec::new();
    snapshot
        .write_html_with(&mut page, &format)
        .expect("writing to a Vec cannot fail");
    let page = String::from_utf8(page).expect("the page is UTF-8");

    // Three script elements: the profile, the sidecar, and the viewer. Any more
    // means text became markup, and the failure that follows is not a mangled
    // name but a page that stops parsing and displays nothing — on a machine
    // chosen precisely because it has no other viewer.
    assert_eq!(page.matches("<script").count(), 3, "text became markup");
    assert_eq!(page.matches("</script>").count(), 3, "text became markup");

    let embedded = block(&page, PROFILE_BLOCK);
    let sidecar = block(&page, DISPLAY_BLOCK);
    assert!(!embedded.contains('<'), "a raw < survived into the profile");
    assert!(!sidecar.contains('<'), "a raw < survived into the sidecar");
    parse_and_screen(embedded, "embedded profile");
    parse_and_screen(sidecar, "display sidecar");

    // And it is the same profile, not a second rendering of one. Compared in
    // the escaping direction, which is total; the other direction is not a
    // function, because `\u003c` in the text is written `\\u003c` and undoing
    // the escape by substitution would corrupt it.
    assert_eq!(
        embedded,
        native.replace('<', r"\u003c"),
        "the page carries something other than the native profile"
    );

    let mut summary = Vec::new();
    snapshot
        .write_text_summary_with(&mut summary, &format, usize::from(input.top))
        .expect("writing to a Vec cannot fail");
    let summary = String::from_utf8(summary).expect("the summary is UTF-8");
    // `\n` is the summary's own layout and is the one control character that
    // means itself here. Everything else in the set is an instruction.
    let offenders: Vec<char> = summary
        .chars()
        .filter(|&c| c != '\n' && is_escaped(c))
        .collect();
    assert!(
        offenders.is_empty(),
        "the summary carries {offenders:?}, which a terminal will act on:\n{summary}"
    );
});
