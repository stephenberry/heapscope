//! The bundled viewer: one HTML file with the profile inside it.
//!
//! PLAN.md section 6.12 makes this load-bearing rather than cosmetic. Valgrind
//! does not exist on Windows and does not support Apple Silicon, which are two
//! of the four platforms this crate supports and the one primary development
//! happens on. Without this, those users are handed a format whose only viewer
//! comes from a tool they cannot install.
//!
//! # What goes in the file
//!
//! The page is `viewer.html`, verbatim, with one marker replaced by two
//! `application/json` blocks:
//!
//! - **the native profile**, byte for byte what [`Snapshot::save_native`] would
//!   write. So the page is not only a viewer but the profile itself: a reader
//!   with no browser can lift the JSON back out of it, and there is no second
//!   schema to keep in step with the first.
//! - **a display sidecar**: one rendered name per frame in that profile, the
//!   trimmed range of each point's stack, and the two labels a frameless point
//!   is shown under.
//!
//! # Why the names are rendered here rather than in the page
//!
//! A native profile stores the linker's own name, still mangled, because
//! demangling is a rendering decision and a reader that wants the raw name must
//! be able to get it. The page is a reader, and the rendering it needs is the
//! one every other emitter already makes: a [`FrameFormat`], which knows how to
//! symbolize an address and which frames of a stack are worth showing.
//!
//! Doing it in JavaScript instead would mean a second demangler for both Rust
//! manglings — the v0 one is the largest module in this crate — living in a file
//! that may not acquire a build step. That is exactly the metastasis section
//! 6.12 sets out to prevent, and it would be a second implementation of
//! something already tested against `rustc-demangle` over a corpus of 201,457
//! real symbols.
//!
//! So the seam stays where it is. The sidecar's `keep` ranges are the renderer's
//! own answer put through the same [`clamp_frames`](super::dhat_v2::clamp_frames)
//! the DHAT emitter puts it through, over the same screened text, which is what
//! makes the two files trim identically rather than similarly.

use std::io::{self, Write};

use super::json::{JsonWriter, Layout};
use super::{FrameFormat, Snapshot};

/// The viewer, hand-written and complete.
const VIEWER: &str = include_str!("viewer.html");

/// What the profile is written in place of.
const MARKER: &str = "<!--HEAPSCOPE-PROFILE-->";

/// Where [`MARKER`] is, decided while compiling.
///
/// A runtime `expect` would be a panic reachable from `Profiler::drop`, where a
/// panic during unwinding aborts the process — so a template that has lost its
/// marker, or grown a second one, fails the build instead. There is no case in
/// which shipping is the better answer: both mean the page cannot be assembled.
const SPLIT: usize = marker_offset();

/// The offset of the single occurrence of [`MARKER`] in [`VIEWER`].
const fn marker_offset() -> usize {
    let haystack = VIEWER.as_bytes();
    let needle = MARKER.as_bytes();
    let mut found = usize::MAX;
    let mut at = 0;
    while at + needle.len() <= haystack.len() {
        let mut matched = 0;
        while matched < needle.len() && haystack[at + matched] == needle[matched] {
            matched += 1;
        }
        if matched == needle.len() {
            assert!(
                found == usize::MAX,
                "the viewer template contains more than one profile marker"
            );
            found = at;
        }
        at += 1;
    }
    assert!(
        found != usize::MAX,
        "the viewer template does not contain the profile marker"
    );
    found
}

/// Writes `snapshot` as a self-contained HTML page.
pub(super) fn write<W: Write>(
    snapshot: &Snapshot,
    format: &dyn FrameFormat,
    mut out: W,
) -> io::Result<()> {
    out.write_all(&VIEWER.as_bytes()[..SPLIT])?;

    // No newline of our own on either side of either block. Both writers end
    // their output with one, so the text between the tags is exactly the file
    // `Snapshot::save_native` would have written — which is what lets the page
    // be lifted back apart into the profile it displays.
    out.write_all(b"<script type=\"application/json\" id=\"heapscope-profile\">")?;
    super::native::write(snapshot, ScriptSafe(&mut out))?;
    out.write_all(b"</script>\n\n")?;

    out.write_all(b"<script type=\"application/json\" id=\"heapscope-display\">")?;
    write_display(snapshot, format, ScriptSafe(&mut out))?;
    out.write_all(b"</script>\n")?;

    out.write_all(&VIEWER.as_bytes()[SPLIT + MARKER.len()..])?;

    // `out` is taken by value, so nobody else can flush it: a `BufWriter` handed
    // here is dropped at the end of this function, and `BufWriter::drop`
    // discards the error from its final write. Without this line a failure on
    // the last chunk of the page is lost, `Snapshot::save_html` returns `Ok`,
    // and `save_with` renames a truncated page into place as though it were
    // complete. The other two emitters already flush, inside `JsonWriter::finish`.
    out.flush()
}

/// Writes the sidecar: what to call each frame, and how much of each stack to
/// show.
fn write_display<W: Write>(
    snapshot: &Snapshot,
    format: &dyn FrameFormat,
    out: W,
) -> io::Result<()> {
    // The same table the native profile writes, from the same function, so the
    // indices in `names` address the entries of its `frames` array by
    // construction rather than because both happen to deduplicate alike.
    let (addresses, per_point) = super::native::frame_table(snapshot);

    let mut raw = String::new();
    let mut names: Vec<String> = Vec::with_capacity(addresses.len());
    for &address in &addresses {
        raw.clear();
        format.format(address, &mut raw);
        // Screened for the same reason every borrowed string in this layer is:
        // these bytes came out of a symbol table, and the page is a place a
        // reordering character would be believed. See `push_display`.
        let mut screened = String::new();
        super::push_display(&mut screened, &raw);
        names.push(screened);
    }

    let mut json = JsonWriter::new(out);
    json.begin_object(Layout::Wrap)?;

    json.key("names")?;
    json.begin_array(Layout::Wrap)?;
    for name in &names {
        json.string(name)?;
    }
    json.end_array()?;

    // Which frames of each stack the renderer chose to show, as a half-open
    // range over the point's own frame list. A range rather than a filtered
    // list because the page offers to show the trimmed frames too, and that
    // costs nothing when the full stack is already in the profile beside it.
    json.key("keep")?;
    json.begin_array(Layout::Wrap)?;
    let mut stack: Vec<String> = Vec::new();
    for frames in &per_point {
        stack.clear();
        stack.extend(frames.iter().map(|&frame| names[frame as usize].clone()));
        let keep = super::dhat_v2::clamp_frames(format.keep(&stack), stack.len());
        json.begin_array(Layout::Inline)?;
        json.u64(keep.start as u64)?;
        json.u64(keep.end as u64)?;
        json.end_array()?;
    }
    json.end_array()?;

    // Defined once, in Rust, so the page and the DHAT file cannot come to
    // describe the same two conditions differently.
    json.key("labels")?;
    json.begin_object(Layout::Wrap)?;
    json.field_str("overflow", super::dhat_v2::OVERFLOW_FRAME)?;
    json.field_str("unwalkable", super::dhat_v2::UNWALKABLE_FRAME)?;
    json.end_object()?;

    json.end_object()?;
    json.finish()?;
    Ok(())
}

/// Passes JSON through, leaving nothing in it that can end a `<script>`.
///
/// # Why a byte substitution is enough, and why it is `<` rather than `</script`
///
/// The content of a `script` element is raw text: entities are not decoded, and
/// the only thing the HTML parser looks for is a run beginning with `<`. Three
/// of the strings in a profile are written by somebody else — a symbol read out
/// of a symbol table, a path from the filesystem, and `argv` — and a directory
/// really can be named so that a path contains `</script>`. Escaping the whole
/// class rather than the one spelling also covers `<!--`, which puts the parser
/// into a state where a later `<script` nests rather than closes.
///
/// `\u003c` is a valid JSON escape for `<`, so the replacement changes no
/// string's value. It is safe to do a byte at a time because JSON has no `<`
/// outside a string literal — no structural character, number, or keyword
/// contains one — and because `<` is ASCII and so cannot be a byte of some
/// longer UTF-8 sequence.
struct ScriptSafe<W>(W);

impl<W: Write> Write for ScriptSafe<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut written = 0;
        for (at, &byte) in buf.iter().enumerate() {
            if byte == b'<' {
                self.0.write_all(&buf[written..at])?;
                self.0.write_all(br"\u003c")?;
                written = at + 1;
            }
        }
        self.0.write_all(&buf[written..])?;
        // Everything handed over was consumed, whatever it expanded to. A
        // count of the bytes actually emitted would be a lie to the caller,
        // which asked how much of *its* buffer was taken.
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::RawAddresses;

    fn escape(input: &[u8]) -> String {
        let mut out = Vec::new();
        ScriptSafe(&mut out).write_all(input).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn a_path_that_would_end_the_script_element_does_not() {
        // A directory may be named `a<`, which puts `</script>` in a path
        // without anybody having to be hostile about it.
        let escaped = escape(br#"{"path":"/tmp/a</script><script>alert(1)</script>/x"}"#);
        assert!(!escaped.contains('<'), "{escaped}");
        assert!(escaped.contains(r"\u003c/script"), "{escaped}");
    }

    #[test]
    fn a_comment_opener_is_escaped_too() {
        // Not the same failure: `<!--` in script content starts the escaped
        // state, where a later `</script>` is data rather than the end tag.
        let escaped = escape(br#"{"symbol":"<!--"}"#);
        assert!(!escaped.contains('<'), "{escaped}");
    }

    #[test]
    fn nothing_else_is_touched() {
        let plain = br#"{"a":1,"b":"x > y","c":[2,3]}"#;
        assert_eq!(escape(plain), String::from_utf8(plain.to_vec()).unwrap());
    }

    #[test]
    fn escaping_survives_a_split_across_writes() {
        // `write` is called with whatever a `BufWriter` happens to hand over,
        // so the guarantee has to hold per call rather than per document.
        let mut out = Vec::new();
        {
            let mut safe = ScriptSafe(&mut out);
            safe.write_all(b"\"</scr").unwrap();
            safe.write_all(b"ipt><b>\"").unwrap();
        }
        let escaped = String::from_utf8(out).unwrap();
        assert!(!escaped.contains('<'), "{escaped}");
    }

    #[test]
    fn the_template_carries_exactly_one_marker() {
        // `SPLIT` is a compile-time assertion, so this cannot fail while the
        // build succeeds. It is here to say out loud what the build is
        // checking, and to fail with a readable message if the const is ever
        // relaxed into something computed at run time.
        assert_eq!(VIEWER.matches(MARKER).count(), 1);
        assert_eq!(&VIEWER[SPLIT..SPLIT + MARKER.len()], MARKER);
    }

    #[test]
    fn the_page_is_written_around_the_marker() {
        let snapshot = Snapshot::default();
        let mut page = Vec::new();
        write(&snapshot, &RawAddresses, &mut page).unwrap();
        let page = String::from_utf8(page).unwrap();

        assert!(page.starts_with("<!doctype html>"), "{}", &page[..40]);
        assert!(page.trim_end().ends_with("</html>"));
        assert!(!page.contains(MARKER), "the marker itself must not survive");
        assert_eq!(page.matches(r#"id="heapscope-profile""#).count(), 1);
        assert_eq!(page.matches(r#"id="heapscope-display""#).count(), 1);
    }

    #[test]
    fn a_sink_that_fails_only_on_flush_still_reports_it() {
        // The failure this rules out is silent. `out` is taken by value, so a
        // `BufWriter` handed here is dropped inside this function -- and
        // `BufWriter::drop` flushes and *discards* the error. Without an
        // explicit flush, a page whose last chunk could not be written would
        // return `Ok`, and `Snapshot::save_with` would rename the truncated
        // file into place as though it were the finished page.
        //
        // Accepting every `write` and failing only on `flush` is the shape of a
        // real full disk behind a buffer: the bytes are taken, and the error
        // arrives when somebody tries to make them durable.
        //
        // Failing on the *last* flush specifically, and not on any earlier one.
        // A sink that failed on every flush passes this test whether or not the
        // page flushes at the end, because the sidecar's `JsonWriter::finish`
        // flushes too and its error propagates from the middle of the function
        // -- which is what the first version of this test was accidentally
        // measuring. Mutation caught it: removing the final flush left this
        // green. The condition below is the page being complete, so only a
        // flush after the last byte can trip it.
        struct FailsOnceComplete {
            seen: Vec<u8>,
        }
        impl Write for FailsOnceComplete {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.seen.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                if self.seen.ends_with(b"</html>\n") {
                    return Err(io::Error::new(io::ErrorKind::StorageFull, "no space"));
                }
                Ok(())
            }
        }

        let sink = FailsOnceComplete { seen: Vec::new() };
        let result = write(&Snapshot::default(), &RawAddresses, sink);
        assert!(
            result.is_err(),
            "a page that could not be made durable must say so, not return Ok"
        );
    }
}
