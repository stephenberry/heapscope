//! Asking a real symbolizer, in batches, and reading three different answers.
//!
//! The profile carries what `symbol::modules` recorded: a path, a load address,
//! and for each frame the address **as it appears in the file**. That is exactly
//! what the three tools below consume, which is the whole point of recording it
//! — so this module is thin by design. What is not thin is reading the answers,
//! because the three formats have nothing in common:
//!
//! ```text
//! llvm-symbolizer   name              addr2line   0x<address>   atos   name (in image) (file:line)
//!                   file:line:column              name
//!                   <blank>                       file:line
//! ```
//!
//! `llvm-symbolizer` ends each address's answer with a blank line, and
//! `addr2line -a` begins each with the address it is about. `atos` is
//! positional, exactly one line per address, so a parser that loses count
//! attributes every remaining frame to the wrong function — silently, and in a
//! way that looks like a profile rather than like a bug. Each parser below
//! therefore returns a slot per address asked about, and the caller checks the
//! length.
//!
//! With inlined frames, `llvm-symbolizer` and `addr2line` repeat the name and
//! location pair once per function, innermost first, so how many lines one
//! address produces is not known in advance. That is why both are asked for a
//! delimiter, and why neither may be read by counting lines.
//!
//! # Mangled on purpose
//!
//! `llvm-symbolizer` and `addr2line` are asked **not** to demangle, and the
//! names they return go through [`heapscope::demangle`](fn@heapscope::demangle)
//! instead. `addr2line -C`
//! is the C++ demangler, which does not understand Rust's v0 scheme at all — it
//! returns `_RNvCs...` unchanged — and a Rust heap profile is mostly v0 names on
//! a current toolchain. Using this crate's demangler also means a symbolized
//! profile and a text summary from the same run spell a name the same way.
//!
//! `atos` has no such flag and always demangles. Its answers are passed through
//! the same call, which refuses what it does not recognise and leaves the name
//! alone — the behaviour `Symbolized` already relies on for every C and C++
//! frame in a process.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// One resolved location, and the inlined frames above it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub function: String,
    pub file: Option<String>,
    pub line: Option<u32>,
}

/// What a symbolizer said about one address.
///
/// `frames` is innermost first and never empty: `frames[0]` is the function the
/// address is in, and anything after it is the chain of callers that inlined it.
/// An address the tool could not place produces `None` rather than a `Resolution`
/// with an empty list, so "no answer" and "an answer of nothing" cannot be
/// confused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolution {
    pub frames: Vec<Frame>,
}

/// Which symbolizer to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Atos,
    LlvmSymbolizer,
    Addr2Line,
}

impl Tool {
    pub fn name(self) -> &'static str {
        match self {
            Tool::Atos => "atos",
            Tool::LlvmSymbolizer => "llvm-symbolizer",
            Tool::Addr2Line => "addr2line",
        }
    }

    pub fn parse(name: &str) -> Option<Tool> {
        match name {
            "atos" => Some(Tool::Atos),
            "llvm-symbolizer" => Some(Tool::LlvmSymbolizer),
            "addr2line" => Some(Tool::Addr2Line),
            _ => None,
        }
    }

    /// Whether this tool takes the address the image was **loaded at** rather
    /// than the address it has in the file.
    ///
    /// True only for `atos`, and it is the difference that makes a wrong choice
    /// here produce confident nonsense instead of nothing: the two numbers are
    /// equal exactly when the image's bias is zero, so a mix-up resolves
    /// correctly on some images and names the wrong function on the rest.
    pub fn wants_runtime_addresses(self) -> bool {
        matches!(self, Tool::Atos)
    }

    /// The tools that could serve this platform, best first.
    ///
    /// `atos` leads on Apple because it is the only one that resolves an image
    /// mapped from the dyld shared cache — almost everything under `/usr/lib` —
    /// where the recorded offset is an address in the cache rather than in the
    /// file on disk, so `llvm-symbolizer` resolves it to nothing even though the
    /// file exists and its UUID matches. Elsewhere `llvm-symbolizer` leads, as
    /// the reader maintained alongside the backend rustc compiles with, and
    /// `addr2line` is the fallback binutils provides on a Linux machine with no
    /// LLVM installed. That is the ordinary case rather than the rare one: it
    /// is what both GitHub Ubuntu runners have **\[measured\]**, so `addr2line`
    /// is held to the same standard, inlined frames included.
    ///
    /// On Windows `addr2line` is not among them. `Module::bias` there is the
    /// image base, so a recorded file address is a relative virtual address;
    /// `llvm-symbolizer` takes one with `--relative-address`, and `addr2line`
    /// resolves against section VMAs, which include the image base, and has no
    /// equivalent option. Asking it anyway returns `??` for every address —
    /// confidently, and in a file that looks finished.
    pub fn preference() -> &'static [Tool] {
        if cfg!(target_vendor = "apple") {
            &[Tool::Atos, Tool::LlvmSymbolizer, Tool::Addr2Line]
        } else if cfg!(windows) {
            &[Tool::LlvmSymbolizer]
        } else {
            &[Tool::LlvmSymbolizer, Tool::Addr2Line, Tool::Atos]
        }
    }

    /// Whether this tool can be started at all.
    ///
    /// Asks the operating system by trying, because that is the only question
    /// that matters and `PATH` is not the only way a program is found. `stdin`
    /// is closed so that a tool which would otherwise wait for addresses cannot
    /// hang this probe.
    pub fn available(self) -> bool {
        Command::new(self.name())
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    }

    fn arguments(self, image: &str, load: u64) -> Vec<String> {
        match self {
            Tool::Atos => vec![
                String::from("-o"),
                String::from(image),
                String::from("-l"),
                format!("{load:#x}"),
            ],
            Tool::LlvmSymbolizer => {
                let mut arguments = vec![
                    format!("--obj={image}"),
                    String::from("--no-demangle"),
                    // Every function the address is in, innermost first: the
                    // inlined one and each caller it was inlined into. Without
                    // it the answer is the innermost alone, and the function
                    // the code physically lies in — the name every in-process
                    // lookup gives the frame, and the one a folded stack shows
                    // — is not reported at all. It is this tool's default, and
                    // is passed so that the shape the parser below depends on
                    // is asked for rather than assumed.
                    String::from("--inlines"),
                ];
                // What `Module::bias` records on Windows is the image base, so
                // the file address in a profile is relative to it. Without this
                // every address resolves to `??:0:0` — measured on a real
                // Windows run, where the tool reported 21 of 23 addresses
                // resolved and had named none of them.
                if cfg!(windows) {
                    arguments.push(String::from("--relative-address"));
                }
                arguments
            }
            Tool::Addr2Line => vec![
                String::from("-f"),
                // Inlined callers, as `--inlines` asks of `llvm-symbolizer`, and
                // for the same reason. Without it the answer is the innermost
                // inlined function alone, which is the wrong one both to trim
                // by and to show: `RawVec::with_capacity_in` inlined into
                // `Vec::with_capacity` is the program's frame, and reads as the
                // allocation path.
                String::from("-i"),
                // The address before each answer. `-i` makes an answer any
                // number of lines long, so this is what says where the next one
                // begins; see `parse_addr2line`.
                String::from("-a"),
                String::from("-e"),
                String::from(image),
            ],
        }
    }

    fn parse_output(self, text: &str, asked: &[u64]) -> Vec<Option<Resolution>> {
        match self {
            Tool::Atos => parse_atos(text, asked.len()),
            Tool::LlvmSymbolizer => parse_llvm(text, asked.len()),
            Tool::Addr2Line => parse_addr2line(text, asked),
        }
    }
}

/// Resolves `addresses` in `image` by running `tool` once.
///
/// One process per image rather than per address. A profile's frames repeat
/// heavily — every stack shares its outermost frames — and the caller has
/// already deduplicated, but even so a thousand distinct addresses across four
/// images is four processes here and would be a thousand the other way.
///
/// The returned vector has one slot per address, in the order asked.
pub fn resolve(
    tool: Tool,
    image: &str,
    load: u64,
    addresses: &[u64],
) -> Result<Vec<Option<Resolution>>, String> {
    if addresses.is_empty() {
        return Ok(Vec::new());
    }

    let mut child = Command::new(tool.name())
        .args(tool.arguments(image, load))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot start {}: {error}", tool.name()))?;

    // Written from another thread while this one reads. All three tools stream:
    // they answer each address as it arrives, so a parent that writes the whole
    // batch before reading any of it deadlocks as soon as the answers fill the
    // pipe — which a few hundred addresses do.
    let mut stdin = child.stdin.take().expect("stdin was piped");
    let batch: Vec<u8> = addresses
        .iter()
        .flat_map(|address| format!("{address:#x}\n").into_bytes())
        .collect();
    let writer = std::thread::spawn(move || {
        // A tool that exits early — a missing file, a bad flag — closes the pipe
        // under us, and the resulting `BrokenPipe` is not the error worth
        // reporting. What it said on stderr is, and that is read below.
        let _ = stdin.write_all(&batch);
        drop(stdin);
    });

    let mut output = String::new();
    let mut reader = BufReader::new(child.stdout.take().expect("stdout was piped"));
    let mut line = Vec::new();
    while reader
        .read_until(b'\n', &mut line)
        .map_err(|error| format!("reading from {}: {error}", tool.name()))?
        > 0
    {
        // Lossy: a symbol table can hold any bytes at all, and a name that is
        // not UTF-8 is still worth reporting with replacement characters.
        output.push_str(&String::from_utf8_lossy(&line));
        line.clear();
    }

    let finished = child
        .wait_with_output()
        .map_err(|error| format!("waiting for {}: {error}", tool.name()))?;
    let _ = writer.join();

    if !finished.status.success() {
        let complaint = String::from_utf8_lossy(&finished.stderr);
        let complaint = complaint.trim();
        return Err(if complaint.is_empty() {
            format!("{} exited with {}", tool.name(), finished.status)
        } else {
            format!("{}: {complaint}", tool.name())
        });
    }

    let resolved = tool.parse_output(&output, addresses);
    if resolved.len() != addresses.len() {
        return Err(format!(
            "{} answered about {} addresses out of {}, so nothing it said can be \
             matched to an address with confidence",
            tool.name(),
            resolved.len(),
            addresses.len()
        ));
    }
    Ok(resolved)
}

/// Whether a tool is saying "I could not place this".
///
/// All three spell it `??`. An empty name counts too: it is what a truncated
/// answer looks like, and attributing a frame to the empty string would put a
/// nameless band in the middle of every flame graph built from the result.
fn is_unknown(name: &str) -> bool {
    name.is_empty() || name == "??" || name == "?"
}

/// Splits `file:line` or `file:line:column` into its parts.
///
/// Read from the right, and **numerically** rather than by counting colons. A
/// path legitimately contains one — `C:\src\main.rs` on Windows, and any
/// directory somebody named that way anywhere else — so a parser that treats the
/// last two colons as the line and column turns `C:\src\main.rs:129` into the
/// file `C` at line `\src\main.rs`. The rule that survives both is: a trailing
/// segment is a number or it is part of the path, and a column exists only when
/// the segment before it is *also* a number.
///
/// binutils `addr2line` adds two spellings of its own, and with `-i` they turn
/// up at every inlined level rather than occasionally: `file:12
/// (discriminator 3)`, where the annotation distinguishes basic blocks on one
/// line and says nothing a reader of a profile needs, and `file:?`, a known file
/// with no line. Both keep the file.
fn split_location(text: &str) -> (Option<String>, Option<u32>) {
    let text = text.trim();
    let text = match text.rsplit_once(" (discriminator ") {
        Some((location, annotation)) if annotation.ends_with(')') => location,
        _ => text,
    };
    if text.is_empty() || text.starts_with("??") {
        return (None, None);
    }
    let Some((head, tail)) = text.rsplit_once(':') else {
        return (None, None);
    };
    if tail == "?" {
        return ((!head.is_empty()).then(|| String::from(head)), None);
    }
    if tail.parse::<u32>().is_err() {
        return (None, None);
    }
    // `head:tail` is either `file:line` or `file:line:column`. It is the second
    // exactly when `head` itself ends in a number.
    let (file, line) = match head.rsplit_once(':') {
        Some((before, middle)) if middle.parse::<u32>().is_ok() => (before, middle),
        _ => (head, tail),
    };
    (
        (!file.is_empty() && file != "??").then(|| String::from(file)),
        // Zero is `addr2line`'s "no line", not line zero.
        line.parse::<u32>().ok().filter(|&line| line != 0),
    )
}

/// Names go through this crate's demangler, which refuses what it does not
/// understand and leaves the linker's own spelling in place.
fn readable(name: &str) -> String {
    let mut out = String::new();
    if heapscope::demangle(name, &mut out) {
        return out;
    }
    String::from(name)
}

/// One frame, or nothing where the tool placed neither a name nor a file.
///
/// A location with no name is kept: `addr2line` against a binary that has line
/// tables and no symbol table answers exactly that way, and a file and line is
/// more than the reader had.
fn frame(name: &str, file: Option<String>, line: Option<u32>) -> Option<Frame> {
    match (is_unknown(name), &file) {
        (true, None) => None,
        // The same three characters every unnamed frame in this crate uses.
        (true, Some(_)) => Some(Frame {
            function: String::from("???"),
            file,
            line,
        }),
        (false, _) => Some(Frame {
            function: readable(name),
            file,
            line,
        }),
    }
}

/// An answer built from lines that alternate name, location, name, location.
///
/// Shared by `llvm-symbolizer` and `addr2line`, which differ in how one
/// address's lines are delimited and not in what they contain.
fn alternating(lines: &[&str]) -> Option<Resolution> {
    let mut frames = Vec::new();
    for pair in lines.chunks(2) {
        let (file, line) = pair.get(1).map_or((None, None), |at| split_location(at));
        if let Some(frame) = frame(pair[0].trim(), file, line) {
            frames.push(frame);
        }
    }
    (!frames.is_empty()).then_some(Resolution { frames })
}

/// `llvm-symbolizer`: name and location per frame, a blank line per address.
///
/// The blank line is the only self-describing separator any of the three
/// provide, and with `--inlines` it is doing real work: one address can produce
/// any number of name/location pairs, so nothing else in the output says where
/// the next address begins.
fn parse_llvm(text: &str, asked: usize) -> Vec<Option<Resolution>> {
    let mut answers = Vec::with_capacity(asked);
    let mut pending: Vec<&str> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            // A blank line before anything at all is not an answer of nothing;
            // it is a tool that started with a newline.
            if !pending.is_empty() {
                answers.push(alternating(&pending));
                pending.clear();
            }
            continue;
        }
        pending.push(line);
    }
    if !pending.is_empty() {
        answers.push(alternating(&pending));
    }
    answers
}

/// `addr2line -f -i -a`: the address, then a name and location per frame.
///
/// Each answer is matched against the address it was asked about rather than
/// merely counted, and reading stops at the first that disagrees. The echoed
/// address is the only thing in the output that says which address an answer
/// belongs to, so an answer that claims to be about some other address — or a
/// line that is not one at all where a header should be — is the moment the
/// output stopped meaning what this reads it as. Stopping leaves the caller a
/// short list, which it refuses; carrying on would attribute every frame after
/// it to the wrong function.
///
/// A header cannot be confused with a name. It is `0x` and hex digits only, and
/// a name is a mangled symbol, `??`, or a C identifier, none of which can begin
/// with a digit.
fn parse_addr2line(text: &str, asked: &[u64]) -> Vec<Option<Resolution>> {
    let mut answers = Vec::with_capacity(asked.len());
    let mut lines = text.lines().peekable();
    for &address in asked {
        let Some(header) = lines.next() else {
            break;
        };
        if address_header(header) != Some(address) {
            break;
        }
        let mut body = Vec::new();
        while let Some(&line) = lines.peek() {
            if address_header(line).is_some() {
                break;
            }
            body.push(line);
            lines.next();
        }
        // An odd count is half an answer: a name whose location never came.
        // Dropped rather than completed, as `alternating` would otherwise do.
        if body.is_empty() || body.len() % 2 != 0 {
            break;
        }
        answers.push(alternating(&body));
    }
    answers
}

/// The address `addr2line -a` writes before an answer, or `None` for any other
/// line.
fn address_header(line: &str) -> Option<u64> {
    let digits = line.trim().strip_prefix("0x")?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(digits, 16).ok()
}

/// `atos`: one line per address, and the address itself where it failed.
///
/// ```text
/// profile_a_program::churn (in profile_a_program) (profile_a_program.rs:129)
/// _malloc (in libsystem_malloc.dylib) + 32
/// 0x1044c81f0
/// ```
fn parse_atos(text: &str, asked: usize) -> Vec<Option<Resolution>> {
    let mut answers = Vec::with_capacity(asked);
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // An address echoed back is the failure spelling, and it is checked
        // first because it is also a syntactically valid symbol name.
        if line.starts_with("0x") && !line.contains(' ') {
            answers.push(None);
            continue;
        }
        // ` (in ` is the marker between the symbol and the image. Split from the
        // left: an image name cannot contain it, and a Rust generic can contain
        // very nearly anything.
        let (name, rest) = line
            .split_once(" (in ")
            .map_or((line, ""), |(name, rest)| (name.trim(), rest));
        // What follows the image is either ` (file:line)` or ` + offset`, and
        // only the first is a location.
        let (file, at) = rest
            .split_once(") (")
            .and_then(|(_, tail)| tail.strip_suffix(')'))
            .map_or((None, None), split_location);
        answers.push(frame(name, file, at).map(|frame| Resolution {
            frames: vec![frame],
        }));
    }
    answers
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real output, captured from `llvm-symbolizer --inlines --no-demangle`.
    #[test]
    fn llvm_output_is_read_one_answer_per_blank_line() {
        let text = "\
_ZN4core3fmt5write17hb1f9a4a7f2f1a0c9E
/rustc/lib/core/src/fmt/mod.rs:1234:9

??
??:0

_ZN17profile_a_program5churn17h0123456789abcdefE
/src/main.rs:129:5

";
        let answers = parse_llvm(text, 3);
        assert_eq!(answers.len(), 3);

        let first = answers[0].as_ref().expect("a named frame");
        assert_eq!(first.frames[0].function, "core::fmt::write");
        assert_eq!(
            first.frames[0].file.as_deref(),
            Some("/rustc/lib/core/src/fmt/mod.rs")
        );
        assert_eq!(first.frames[0].line, Some(1234));

        assert_eq!(answers[1], None, "`??` is not a name");
        assert_eq!(
            answers[2].as_ref().expect("a named frame").frames[0].function,
            "profile_a_program::churn"
        );
    }

    /// The reason `--inlines` is passed: an address inside an inlined function
    /// otherwise reports the function it was inlined into, which the reader had.
    #[test]
    fn inlined_callers_arrive_as_extra_frames_innermost_first() {
        let text = "\
_ZN4core6option6unwrap17h00112233445566aaE
/rustc/lib/core/src/option.rs:900:21
_ZN17profile_a_program5parse17hfedcba9876543210E
/src/main.rs:42:13

";
        let answers = parse_llvm(text, 1);
        let frames = &answers[0].as_ref().expect("frames").frames;
        assert_eq!(frames.len(), 2, "{frames:#?}");
        assert_eq!(frames[0].function, "core::option::unwrap");
        assert_eq!(frames[1].function, "profile_a_program::parse");
        assert_eq!(frames[1].line, Some(42));
    }

    /// The shape of `addr2line -f -i -a`: an address, then one name and
    /// location per frame, so one answer here is three frames long and the
    /// next is one.
    #[test]
    fn addr2line_output_is_read_one_answer_per_address() {
        let text = "\
0x0000000000012340
_ZN5alloc5alloc7realloc17h0123456789abcdefE
/rustc/library/alloc/src/alloc.rs:140
_ZN5alloc7raw_vec11finish_grow17h0123456789abcdefE
/rustc/library/alloc/src/raw_vec/mod.rs:780
_ZN17profile_a_program4grow17h0123456789abcdefE
/src/main.rs:12
0x0000000000012350
_ZN17profile_a_program5churn17h0123456789abcdefE
/src/main.rs:129
0x0000000000012360
??
??:0
";
        let answers = parse_addr2line(text, &[0x12340, 0x12350, 0x12360]);
        assert_eq!(answers.len(), 3, "{answers:#?}");

        let inlined = &answers[0].as_ref().expect("frames").frames;
        let names: Vec<&str> = inlined
            .iter()
            .map(|frame| frame.function.as_str())
            .collect();
        assert_eq!(
            names,
            [
                "alloc::alloc::realloc",
                "alloc::raw_vec::finish_grow",
                "profile_a_program::grow"
            ],
            "innermost first"
        );
        assert_eq!(inlined[2].line, Some(12));

        assert_eq!(
            answers[1].as_ref().expect("a name").frames[0].function,
            "profile_a_program::churn"
        );
        assert_eq!(answers[2], None);
    }

    /// A truncated final answer is dropped rather than paired with whatever
    /// comes next, because the caller checks the count and a wrong-but-complete
    /// list is the one thing it cannot detect.
    #[test]
    fn a_half_answer_from_addr2line_is_not_completed_by_guessing() {
        let answers = parse_addr2line(
            "0x10\n_ZN1a1bE\n/src/a.rs:1\n0x20\n_ZN1c1dE\n",
            &[0x10, 0x20],
        );
        assert_eq!(answers.len(), 1, "{answers:#?}");
    }

    /// `llvm-addr2line` echoes the address unpadded where binutils pads it to
    /// sixteen digits. Read as a number, the two are the same header.
    #[test]
    fn an_unpadded_addr2line_header_is_the_same_address() {
        let padded = "0x0000000000012340\n_ZN1a1bE\n/src/a.rs:1\n";
        let unpadded = "0x12340\n_ZN1a1bE\n/src/a.rs:1\n";
        let expected = parse_addr2line(padded, &[0x12340]);
        assert_eq!(expected.len(), 1, "{expected:#?}");
        assert_eq!(parse_addr2line(unpadded, &[0x12340]), expected);
    }

    /// The echoed address is what ties an answer to a question, so an answer
    /// about some other address ends the reading rather than being taken as the
    /// answer to this one.
    #[test]
    fn an_addr2line_answer_about_another_address_is_not_believed() {
        let text = "0x10\n_ZN1a1bE\n/src/a.rs:1\n0x30\n_ZN1c1dE\n/src/c.rs:3\n";
        assert_eq!(parse_addr2line(text, &[0x10, 0x20]).len(), 1);

        // And output with no headers at all — `addr2line` run without `-a` — is
        // not read positionally as though it had them.
        assert!(parse_addr2line("_ZN1a1bE\n/src/a.rs:1\n", &[0x10]).is_empty());
    }

    #[test]
    fn atos_output_is_read_one_line_at_a_time() {
        let text = "\
profile_a_program::churn (in profile_a_program) (main.rs:129)
_malloc (in libsystem_malloc.dylib) + 32
0x1044c81f0
";
        let answers = parse_atos(text, 3);
        assert_eq!(answers.len(), 3);

        let first = answers[0].as_ref().expect("a name");
        assert_eq!(first.frames[0].function, "profile_a_program::churn");
        assert_eq!(first.frames[0].file.as_deref(), Some("main.rs"));
        assert_eq!(first.frames[0].line, Some(129));

        // An offset is not a location, and reading it as one would put every
        // system frame at a line number that is really a byte count.
        let second = answers[1].as_ref().expect("a name");
        assert_eq!(second.frames[0].function, "_malloc");
        assert_eq!(second.frames[0].file, None);
        assert_eq!(second.frames[0].line, None);

        assert_eq!(answers[2], None, "an echoed address is a failure");
    }

    /// A path with a colon in it, which is every Windows path and any directory
    /// somebody named that way.
    #[test]
    fn a_colon_in_a_path_is_not_read_as_a_line_number() {
        assert_eq!(
            split_location(r"C:\src\main.rs:129"),
            (Some(String::from(r"C:\src\main.rs")), Some(129))
        );
        assert_eq!(
            split_location(r"C:\src\main.rs:129:5"),
            (Some(String::from(r"C:\src\main.rs")), Some(129))
        );
        assert_eq!(
            split_location("/src/main.rs:129:5"),
            (Some(String::from("/src/main.rs")), Some(129))
        );
        assert_eq!(
            split_location("/src/main.rs:129"),
            (Some(String::from("/src/main.rs")), Some(129))
        );
    }

    /// binutils' own spellings, which `-i` makes routine: a discriminator after
    /// the line, and a file whose line is unknown. Both keep the file.
    #[test]
    fn binutils_annotations_do_not_cost_the_location() {
        assert_eq!(
            split_location("/src/main.rs:12 (discriminator 3)"),
            (Some(String::from("/src/main.rs")), Some(12))
        );
        assert_eq!(
            split_location(r"C:\src\main.rs:12 (discriminator 1)"),
            (Some(String::from(r"C:\src\main.rs")), Some(12))
        );
        assert_eq!(
            split_location("/rustc/library/std/src/process.rs:?"),
            (
                Some(String::from("/rustc/library/std/src/process.rs")),
                None
            )
        );
        assert_eq!(split_location("??:?"), (None, None));
    }

    #[test]
    fn the_not_found_spellings_resolve_to_nothing() {
        assert_eq!(split_location("??:0"), (None, None));
        assert_eq!(split_location(""), (None, None));
        assert_eq!(split_location("no-colons-here"), (None, None));
        // Line zero is `addr2line`'s "no line", not the first line of a file.
        assert_eq!(
            split_location("/src/main.rs:0"),
            (Some(String::from("/src/main.rs")), None)
        );
    }

    /// A binary with line tables and no symbol table. The location is worth
    /// keeping even though the name is not.
    #[test]
    fn a_location_with_no_name_is_still_an_answer() {
        let answers = parse_addr2line("0x4d\n??\n/src/main.rs:77\n", &[0x4d]);
        let frame = &answers[0].as_ref().expect("a location").frames[0];
        assert_eq!(frame.function, "???");
        assert_eq!(frame.line, Some(77));
    }

    /// Only `atos` takes the runtime address. The two numbers are equal exactly
    /// when an image's bias is zero, so getting this wrong resolves correctly on
    /// some images and names the wrong function on the rest.
    #[test]
    fn only_atos_wants_the_address_the_image_was_loaded_at() {
        assert!(Tool::Atos.wants_runtime_addresses());
        assert!(!Tool::LlvmSymbolizer.wants_runtime_addresses());
        assert!(!Tool::Addr2Line.wants_runtime_addresses());
    }

    #[test]
    fn every_tool_name_round_trips() {
        for tool in [Tool::Atos, Tool::LlvmSymbolizer, Tool::Addr2Line] {
            assert_eq!(Tool::parse(tool.name()), Some(tool));
        }
        assert_eq!(Tool::parse("nm"), None);
    }

    /// The preference list has to name every tool that can work here, or
    /// `--tool` would offer one automatic selection can never reach.
    ///
    /// Windows names one, and that is a decision rather than an omission, so it
    /// is pinned rather than described: `addr2line` resolves against section
    /// VMAs and cannot take the relative virtual address a Windows profile
    /// records, and `atos` does not exist there. Listing either would put a tool
    /// in front of automatic selection that answers `??` for every address.
    #[test]
    fn the_preference_order_covers_every_tool_that_can_work_here() {
        let preferred = Tool::preference();
        if cfg!(windows) {
            assert_eq!(
                preferred,
                [Tool::LlvmSymbolizer],
                "only llvm-symbolizer can resolve a relative virtual address"
            );
        } else {
            for tool in [Tool::Atos, Tool::LlvmSymbolizer, Tool::Addr2Line] {
                assert!(preferred.contains(&tool), "{tool:?} is unreachable");
            }
            assert_eq!(preferred.len(), 3, "a tool is listed twice");
        }
    }
}
