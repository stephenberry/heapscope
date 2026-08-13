//! Any text at all, through every writer.
//!
//! A profile is mostly made of text this crate did not write. A symbol comes
//! from a symbol table that may be truncated, stripped, or from a different
//! build; a path comes from the loader; a command line comes from `argv`; a
//! thread name comes from the platform; a region name comes from the program.
//! None of it is validated anywhere, and all of it ends up inside a JSON
//! string, a terminal line, and a `<script>` element a browser parses.
//!
//! So the contract is not "correct for profiles a workload produces". It is
//! that **no text of any shape** may make a writer produce a document a strict
//! parser refuses, a page that stops parsing halfway, or a line a terminal acts
//! on. `fuzz/fuzz_targets/profile.rs` runs a coverage-guided campaign against
//! the same contract; this file is what holds it on every `cargo test`.
//!
//! `tests/dhat_output.rs` and `tests/native_output.rs` already generate
//! profiles, and this is not that test again. Those two generate the command
//! line and one image path, and render frames with a fixed name. The text a
//! *symbol table* supplies has never been generated anywhere, and it is both
//! the largest string in a profile and the one with the weakest provenance.
//! Thread and region names have never been generated either. Neither the page
//! nor the text summary has had a property test at all.
//!
//! Generation is not uniform random text. Uniform text is overwhelmingly
//! characters that mean only themselves, and never produces `</script>` or a
//! bidirectional override, so the cases below are built from fragments that end
//! a document, end a string, end a script element, or reorder a line — mixed
//! with arbitrary code points so the generator is not limited to the failures
//! somebody already thought of.
//!
//! What is deliberately *not* here is `support::native::problems` and
//! `support::dhat::problems`. Those check that a profile's counters add up,
//! which is a fact about arithmetic and not about text, and they are already
//! run over generated profiles by the two files named above — with snapshots
//! built to be coherent, which takes a hundred lines that would say nothing
//! about escaping. The oracle here is a strict parser and the independent
//! statement of the screen in `support::display`.

mod support;

use heapscope::output::{FrameFormat, RegionStats, Snapshot, TallyStats, ThreadStats};
use heapscope::symbol::modules::Module;
use proptest::prelude::*;
use support::display;
use support::json::{self, Value};
use support::page::{block, DISPLAY_BLOCK, PROFILE_BLOCK};
use support::snapshot::{hand_built, point};

/// Text that has ended something, somewhere.
///
/// Each of these is a real terminator rather than an awkward-looking character:
/// `</script>` and `<!--` end or re-scope a script element, `"` and `\` end or
/// escape a JSON string, U+2028 ends a line in JavaScript but not in JSON, ESC
/// begins a terminal control sequence, and U+202E reverses everything after it.
const TERMINATORS: &[&str] = &[
    "</script>",
    "<!--",
    "-->",
    "<script>",
    "<",
    ">",
    "\u{2028}",
    "\u{2029}",
    "\"",
    "\\",
    "\\u003c",
    "\\u{1b}",
    "\u{0}",
    "\u{1b}[2J",
    "\u{202e}",
    "\u{2066}",
    "\u{7f}",
    "\u{85}",
    "{",
    "}",
    "[",
    "]",
    ",",
    ":",
    "\r\n",
    "\t",
    // Not hostile, and here so that a screen written as an allowlist of ASCII
    // fails: a profile legitimately carries these. `µs` is the unit label in
    // `Monotonic` mode.
    "µs",
    "\u{1f600}",
    "C:\\Program Files\\a.exe",
];

/// One string, mostly built from things that terminate something.
fn text() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            3 => prop::sample::select(TERMINATORS).prop_map(str::to_owned),
            1 => r"\p{Any}".prop_map(String::from),
        ],
        0..6,
    )
    .prop_map(|pieces| pieces.concat())
}

/// Every string a profile carries that came from outside this crate.
///
/// One generated value per field rather than one shared between them, so that a
/// shrunk failure names the field that broke rather than all five.
#[derive(Clone, Debug)]
struct Borrowed {
    command: String,
    module_path: String,
    build_id: String,
    thread_name: String,
    region_name: String,
    symbol: String,
}

fn borrowed() -> impl Strategy<Value = Borrowed> {
    (text(), text(), text(), text(), text(), text()).prop_map(
        |(command, module_path, build_id, thread_name, region_name, symbol)| Borrowed {
            command,
            module_path,
            build_id,
            thread_name,
            region_name,
            symbol,
        },
    )
}

/// A frame rendering that returns whatever the symbol table said.
///
/// This is the injection point no other test has: `FrameFormat` is public, the
/// shipped implementation reads a symbol table, and the emitters screen what
/// comes back rather than trusting it. The address is included so that distinct
/// frames stay distinct, because a rendering that collapses every call site is
/// already covered by `OneName` in `tests/dhat_output.rs`.
struct FromSymbolTable(String);

impl FrameFormat for FromSymbolTable {
    fn format(&self, address: usize, out: &mut String) {
        out.push_str(&format!("{address:#x}: {}", self.0));
    }
}

/// A snapshot with generated text in every field that can hold any.
///
/// The counters come from `hand_built`, which sums them the way the validators
/// read them: a run's peak is what the points held *at the peak* — their
/// `at_gmax` columns — and not the sum of their own peaks, which happened at
/// different instants. Written out by hand, this fixture had disagreed with that
/// rule on four columns.
///
/// Nothing here asserts on any of it, and this suite deliberately runs no
/// validator (see above). The counters are coherent anyway, so that a profile
/// built to carry hostile text is not also a profile no run could have produced:
/// two failures that arrived looking alike would be two failures to tell apart.
/// The one thing that would still fail `support::native::problems` is the shape
/// histogram: these totals account for twelve blocks and nothing ever called
/// `Shapes::record`, which is the only way to fill one. Filling it would say
/// nothing about escaping, which is what this file is for.
fn snapshot(borrowed: &Borrowed) -> Snapshot {
    let mut snapshot = hand_built(vec![
        point(&[0x1000, 0x2000, 0x3000], 4096, 4),
        point(&[0x1500, 0x2000, 0x3000], 2048, 4),
        // A capture that found nothing. It still allocated, so it still has to
        // be written, and it is the point with no frames to render.
        point(&[], 512, 4),
    ]);
    snapshot.command = borrowed.command.clone();
    snapshot.modules = vec![Module {
        path: borrowed.module_path.clone(),
        start: 0x1000,
        size: 0x3000,
        bias: 0x400,
        image_base: 0x1000,
        // A build identity is bytes out of an image's note section rendered as
        // hex, so this crate's own producer cannot make an unsafe one. The
        // field is public on a public struct all the same, and the rule
        // `push_display` states is that a string is screened where it becomes
        // output rather than where it was produced — precisely so that being
        // safe does not depend on knowing every producer.
        build_id: Some(borrowed.build_id.clone()),
    }];
    // One thread, so its row *is* the run rather than a copy of it. Read from
    // the totals rather than written out, because a row that has to sum to them
    // is not somewhere the numbers should be able to drift.
    let stats = snapshot.stats;
    snapshot.threads = vec![ThreadStats {
        id: 0,
        overflow: false,
        name: Some(borrowed.thread_name.clone()),
        first_seen: 1,
        counts: TallyStats {
            total_bytes: stats.total_bytes,
            total_blocks: stats.total_blocks,
            curr_bytes: stats.curr_bytes,
            curr_blocks: stats.curr_blocks,
            max_bytes: stats.max_bytes,
            max_blocks: stats.max_blocks,
        },
    }];
    // One region, holding the first point's share and nothing else. Unlike the
    // thread row this need not sum to the totals — an allocation made outside
    // every region belongs to no row — and the validator asks only that it not
    // exceed them. Its peak is what that point held at the run's peak, for the
    // same reason the totals' is, which is also what keeps it under them for any
    // point list rather than just this one.
    let first = snapshot.points[0].counters;
    snapshot.regions = vec![RegionStats {
        id: 0,
        overflow: false,
        name: Some(borrowed.region_name.clone()),
        first_seen: 1,
        entries: 2,
        active: 0,
        counts: TallyStats {
            total_bytes: first.total_bytes,
            total_blocks: first.total_blocks,
            curr_bytes: first.curr_bytes,
            curr_blocks: first.curr_blocks,
            max_bytes: first.at_gmax_bytes,
            max_blocks: first.at_gmax_blocks,
        },
    }];
    snapshot
}

fn native_profile(snapshot: &Snapshot) -> String {
    let mut out = Vec::new();
    snapshot.write_native(&mut out).expect("a Vec cannot fail");
    String::from_utf8(out).expect("the profile is UTF-8")
}

fn dhat_profile(snapshot: &Snapshot, format: &dyn FrameFormat) -> String {
    let mut out = Vec::new();
    snapshot
        .write_dhat_v2_with(&mut out, format)
        .expect("a Vec cannot fail");
    String::from_utf8(out).expect("the profile is UTF-8")
}

fn page(snapshot: &Snapshot, format: &dyn FrameFormat) -> String {
    let mut out = Vec::new();
    snapshot
        .write_html_with(&mut out, format)
        .expect("a Vec cannot fail");
    String::from_utf8(out).expect("the page is UTF-8")
}

fn summary(snapshot: &Snapshot, format: &dyn FrameFormat, top: usize) -> String {
    let mut out = Vec::new();
    snapshot
        .write_text_summary_with(&mut out, format, top)
        .expect("a Vec cannot fail");
    String::from_utf8(out).expect("the summary is UTF-8")
}

/// Whether any string anywhere in `value` contains `wanted`.
///
/// Key-name independent on purpose: the assertion is that the text arrived and
/// arrived screened, which is a fact about the document rather than about which
/// field happens to hold it, and stating it this way survives a field being
/// renamed without weakening.
fn some_string_contains(value: &Value, wanted: &str) -> bool {
    match value {
        Value::String(text) => text.contains(wanted),
        Value::Array(items) => items.iter().any(|item| some_string_contains(item, wanted)),
        Value::Object(members) => members
            .values()
            .any(|item| some_string_contains(item, wanted)),
        _ => false,
    }
}

/// Every character the summary carries that a terminal or a bidirectional
/// renderer would act on.
///
/// `\n` is the summary's own layout and is the one control character that means
/// itself here. Everything else in `display::is_escaped` is an instruction.
fn summary_offenders(summary: &str) -> Vec<char> {
    summary
        .chars()
        .filter(|&c| c != '\n' && display::is_escaped(c))
        .collect()
}

proptest! {
    // Proptest saves failing seeds to a file next to the test, which means
    // resolving the current directory -- and Miri's filesystem isolation makes
    // that a hard abort, which takes the whole test binary with it. Persistence
    // is worth keeping natively, where it turns a rare failure into a
    // permanently reproducible one, so it is dropped only under Miri. Same
    // shape as `tests/native_output.rs` and `tests/dhat_output.rs`.
    #![proptest_config(ProptestConfig {
        // The default rather than a number of this file's own, so that
        // `PROPTEST_CASES=200000 cargo test --release --test profile_fuzz` is a
        // campaign anyone can run without editing anything.
        cases: if cfg!(miri) { 2 } else { ProptestConfig::default().cases },
        failure_persistence: if cfg!(miri) {
            None
        } else {
            Some(Box::new(
                proptest::test_runner::FileFailurePersistence::default(),
            ))
        },
        ..ProptestConfig::default()
    })]

    /// The native profile parses, says what it was given, and carries nothing a
    /// reader's terminal will act on.
    #[test]
    fn the_native_profile_survives_any_text(borrowed in borrowed()) {
        let snapshot = snapshot(&borrowed);
        let profile = native_profile(&snapshot);

        let parsed = json::parse(&profile)
            .unwrap_or_else(|error| panic!("not valid JSON: {error}\n{profile}"));
        let offenders = display::offenders(&profile);
        prop_assert!(
            offenders.is_empty(),
            "the profile carries {offenders:?}, which a reader's terminal will \
             act on:\n{profile}"
        );

        // Screened, not dropped. A writer that deleted the field it could not
        // render safely would pass every check above.
        for original in [
            &borrowed.command,
            &borrowed.module_path,
            &borrowed.build_id,
            &borrowed.thread_name,
            &borrowed.region_name,
        ] {
            let screened = display::screen(original);
            prop_assert!(
                some_string_contains(&parsed, &screened),
                "{screened:?} reached no field of the profile"
            );
        }
    }

    /// The same for the DHAT file, where the generated text also reaches the
    /// frame table.
    #[test]
    fn the_dhat_profile_survives_any_text(borrowed in borrowed()) {
        let snapshot = snapshot(&borrowed);
        let format = FromSymbolTable(borrowed.symbol.clone());
        let profile = dhat_profile(&snapshot, &format);

        let parsed = json::parse(&profile)
            .unwrap_or_else(|error| panic!("not valid JSON: {error}\n{profile}"));
        let offenders = display::offenders(&profile);
        prop_assert!(
            offenders.is_empty(),
            "the profile carries {offenders:?}, which a reader's terminal will \
             act on:\n{profile}"
        );

        for original in [&borrowed.command, &borrowed.symbol] {
            let screened = display::screen(original);
            prop_assert!(
                some_string_contains(&parsed, &screened),
                "{screened:?} reached no field of the profile"
            );
        }
    }

    /// The page keeps parsing, whatever the text was.
    ///
    /// The failure this exists for is not a mangled name. It is a page that
    /// stops parsing at a `</script>` inside a path and displays nothing, on a
    /// machine chosen precisely because it has no other viewer.
    #[test]
    fn the_page_survives_any_text(borrowed in borrowed()) {
        let snapshot = snapshot(&borrowed);
        let format = FromSymbolTable(borrowed.symbol.clone());
        let page = page(&snapshot, &format);

        // Three script elements: the profile, the sidecar, and the viewer. Any
        // more means text became markup.
        prop_assert_eq!(page.matches("<script").count(), 3, "{}", &page);
        prop_assert_eq!(page.matches("</script>").count(), 3, "{}", &page);

        let profile = block(&page, PROFILE_BLOCK);
        let display_data = block(&page, DISPLAY_BLOCK);
        prop_assert!(!profile.contains('<'), "a raw < survived into the profile");
        prop_assert!(!display_data.contains('<'), "a raw < survived into the sidecar");

        // Escaping `<` is a change of spelling and not of content: `\u003c` is
        // valid JSON for `<`, so what the page carries parses as it stands and
        // means what the profile meant.
        let parsed = json::parse(profile)
            .unwrap_or_else(|error| panic!("the embedded profile is not valid JSON: {error}"));
        let sidecar = json::parse(display_data)
            .unwrap_or_else(|error| panic!("the sidecar is not valid JSON: {error}"));

        // And it is the same profile, not a second rendering of one. Compared
        // in the escaping direction, which is total; the other direction is not
        // a function, because `\u003c` in the text is written `\\u003c` and
        // undoing the escape by substitution would corrupt it.
        prop_assert_eq!(profile, native_profile(&snapshot).replace('<', r"\u003c"));

        let names = sidecar.get("names").expect("the sidecar names the frames");
        prop_assert!(
            some_string_contains(names, &display::screen(&borrowed.symbol)),
            "the symbol reached no frame name"
        );
        prop_assert!(some_string_contains(&parsed, &display::screen(&borrowed.command)));
    }

    /// The summary prints, rather than being executed by the terminal printing
    /// it.
    #[test]
    fn the_text_summary_survives_any_text(
        borrowed in borrowed(),
        top in prop_oneof![Just(0usize), Just(1), Just(5), Just(1000)],
    ) {
        let snapshot = snapshot(&borrowed);
        let format = FromSymbolTable(borrowed.symbol.clone());
        let summary = summary(&snapshot, &format, top);

        let offenders = summary_offenders(&summary);
        prop_assert!(
            offenders.is_empty(),
            "the summary carries {offenders:?}, which a terminal will act on:\n{summary}"
        );
        prop_assert!(
            summary.contains(&display::screen(&borrowed.command)),
            "the command line reached no line of the summary:\n{summary}"
        );
    }
}
