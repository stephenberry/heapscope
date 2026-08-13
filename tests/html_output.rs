//! The bundled viewer of PLAN.md section 6.12, as a file rather than as an idea.
//!
//! The promise is narrow and checkable: **one file, opened by double-clicking
//! it, on a machine with no Valgrind, no network, and no tooling.** Each half of
//! that is a test here.
//!
//! Self-containment is the half that rots quietly. A page that grows a font
//! link, a CDN script, or an icon by URL still looks right on the machine that
//! wrote it and on every machine with a network, and fails silently on exactly
//! the readers this exists for. So the check is structural and runs on every
//! build, rather than being a thing someone remembers to look at.
//!
//! The other half is that the profile survives the trip into the page. A symbol
//! or a path can contain `</script>` — a Rust generic name contains `<` several
//! times over, and a directory really can be named so that a path ends a script
//! element — and the failure mode is not a mangled name but a page that stops
//! parsing halfway and displays nothing.
//!
//! What is *not* here is whether the page looks right. That is a person's job,
//! and `ci/check-bundled-viewer.mjs` covers the arithmetic underneath it.

mod support;

use heapscope::output::{FrameFormat, PointKind, ProgramPoint, RawAddresses, Snapshot};
use heapscope::symbol::modules::Module;
use support::page::{block, parsed, DISPLAY_BLOCK, PROFILE_BLOCK};
use support::snapshot::{hand_built, point};

/// A snapshot with two stacks that share their outermost frame, so the tree the
/// page builds has something to branch.
fn snapshot() -> Snapshot {
    hand_built(vec![
        point(&[0x1000, 0x2000, 0x3000], 4096, 4),
        point(&[0x1500, 0x2000, 0x3000], 2048, 4),
    ])
}

fn page_of(snapshot: &Snapshot) -> String {
    let mut out = Vec::new();
    snapshot.write_html(&mut out).expect("the page is written");
    String::from_utf8(out).expect("the page is UTF-8")
}

/// **The promise.** A page that reaches for anything outside itself works on the
/// machine that wrote it and fails on the machine it was written for.
///
/// Checked by shape rather than by network, because a check that needs the
/// network to prove independence from the network proves nothing when it is
/// skipped.
#[test]
fn the_page_refers_to_nothing_outside_itself() {
    let page = page_of(&snapshot());

    for reference in [
        "http://",
        "https://",
        "//cdn",
        "<link",
        "<img",
        "<iframe",
        "@import",
        "url(",
        "fetch(",
        "XMLHttpRequest",
        "importScripts",
        "WebSocket",
        "navigator.sendBeacon",
        "src=",
    ] {
        assert!(
            !page.contains(reference),
            "the page reaches outside itself with {reference:?}"
        );
    }
}

/// The page carries the profile, not a rendering of it: the bytes between the
/// tags are what `save_native` writes, so a reader with no browser can lift the
/// data back out and a reader with one is looking at the same numbers.
#[test]
fn the_embedded_profile_is_what_save_native_writes() {
    let snapshot = snapshot();

    let mut native = Vec::new();
    snapshot.write_native(&mut native).unwrap();
    let native = String::from_utf8(native).unwrap();

    let page = page_of(&snapshot);
    let embedded = block(&page, PROFILE_BLOCK).replace(r"\u003c", "<");

    assert_eq!(embedded, native);
}

/// A path can end a script element without anybody being hostile: a directory
/// may be named `a<`, and then a path contains `</script>`.
///
/// The failure this prevents is not a mangled name. It is a page that stops
/// parsing at the injected tag and renders nothing at all, on a profile that
/// looked fine when it was written.
#[test]
fn a_path_that_would_end_the_script_element_does_not() {
    let mut snapshot = snapshot();
    snapshot.modules = vec![Module {
        path: String::from("/tmp/a</script><script>document.title='owned'</script>/libx.so"),
        start: 0x1000,
        size: 0x4000,
        bias: 0,
        image_base: 0x1000,
        build_id: None,
    }];
    snapshot.command = String::from("./x </script><script>1</script>");

    let page = page_of(&snapshot);

    // Three script elements and no more: the profile, the sidecar, and the
    // viewer. A fourth means something in the profile opened one.
    assert_eq!(page.matches("</script>").count(), 3, "{page}");
    assert_eq!(page.matches("<script").count(), 3);

    // The text is still there, still readable, and still parses.
    let profile = parsed(&page, PROFILE_BLOCK);
    let modules = profile.get("modules").unwrap().as_array().unwrap();
    assert!(
        modules[0]
            .get("path")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("</script>"),
        "the path survives the escape unchanged"
    );
    assert!(profile
        .get("run")
        .unwrap()
        .get("command")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("</script>"));
}

/// One name per frame, indexed the way the profile's own `frames` array is.
///
/// The two tables come from one function for this reason; if they ever stopped
/// agreeing, every frame in the page would be labelled with a different frame's
/// name and nothing would look broken.
#[test]
fn the_sidecar_names_every_frame_the_profile_carries() {
    let page = page_of(&snapshot());
    let profile = parsed(&page, PROFILE_BLOCK);
    let display = parsed(&page, DISPLAY_BLOCK);

    let frames = profile.get("frames").unwrap().as_array().unwrap();
    let names = display.get("names").unwrap().as_array().unwrap();
    let keep = display.get("keep").unwrap().as_array().unwrap();
    let points = profile.get("points").unwrap().as_array().unwrap();

    assert_eq!(names.len(), frames.len());
    assert_eq!(keep.len(), points.len());

    // Four distinct addresses across the two stacks, which share two frames.
    assert_eq!(frames.len(), 4);
    for (name, frame) in names.iter().zip(frames) {
        let address = frame.get("addr").unwrap().as_str().unwrap();
        assert!(
            name.as_str().unwrap().starts_with(address),
            "{name:?} does not name {address}"
        );
    }
}

/// Renders every frame and keeps only the middle one, so that a `keep` which
/// ignored the renderer would show a different stack.
struct KeepTheMiddle;

impl FrameFormat for KeepTheMiddle {
    fn format(&self, address: usize, out: &mut String) {
        out.push_str(&format!("frame at {address:#x}"));
    }

    fn keep(&self, frames: &[String]) -> std::ops::Range<usize> {
        1..frames.len().max(2) - 1
    }
}

/// The page trims where the renderer says to, and carries the frames it trimmed
/// so that showing them costs nothing.
///
/// This is the thing DHAT v2 cannot do: its `fs` holds only the frames that
/// survived trimming, so a reader who wants the rest has to re-record. Here the
/// full stack is in the profile and the range is advice.
#[test]
fn the_page_keeps_what_the_renderer_kept_and_carries_the_rest() {
    let snapshot = snapshot();
    let mut out = Vec::new();
    snapshot.write_html_with(&mut out, &KeepTheMiddle).unwrap();
    let page = String::from_utf8(out).unwrap();

    let profile = parsed(&page, PROFILE_BLOCK);
    let display = parsed(&page, DISPLAY_BLOCK);

    let points = profile.get("points").unwrap().as_array().unwrap();
    let keep = display.get("keep").unwrap().as_array().unwrap();

    for (point, range) in points.iter().zip(keep) {
        let frames = point.get("frames").unwrap().as_array().unwrap();
        let range = range.as_array().unwrap();
        let (start, end) = (range[0].as_u64().unwrap(), range[1].as_u64().unwrap());

        // Every frame is still in the profile...
        assert_eq!(frames.len(), 3, "the full stack is carried");
        // ...and the renderer's narrower answer is what the page shows.
        assert_eq!(
            (start, end),
            (1, 2),
            "the renderer's range reached the page"
        );
    }

    let names = display.get("names").unwrap().as_array().unwrap();
    assert!(names[0].as_str().unwrap().starts_with("frame at 0x"));
}

/// A range that would empty a stack, or index past it, is corrected rather than
/// believed — the same correction the DHAT emitter makes, because the page runs
/// in `Profiler::drop` where a panic aborts the process.
struct KeepNothing;

impl FrameFormat for KeepNothing {
    fn format(&self, _address: usize, out: &mut String) {
        out.push_str("frame");
    }

    // The nonsense is the input under test: `FrameFormat` is a public trait, so
    // a range that starts past the end and runs backwards is something an
    // implementation this crate never sees can return, and the emitter runs
    // where a panic aborts the process.
    #[allow(clippy::reversed_empty_ranges)]
    fn keep(&self, _frames: &[String]) -> std::ops::Range<usize> {
        9..3
    }
}

#[test]
fn a_renderer_cannot_make_a_walked_stack_look_unwalkable() {
    let snapshot = snapshot();
    let mut out = Vec::new();
    snapshot.write_html_with(&mut out, &KeepNothing).unwrap();
    let page = String::from_utf8(out).unwrap();

    let display = parsed(&page, DISPLAY_BLOCK);
    for range in display.get("keep").unwrap().as_array().unwrap() {
        let range = range.as_array().unwrap();
        let (start, end) = (range[0].as_u64().unwrap(), range[1].as_u64().unwrap());
        assert!(
            start < end,
            "a stack that was walked shows at least one frame"
        );
        assert!(end <= 3, "the range stays inside the stack");
    }
}

/// The two ways a point has no frames have opposite remedies, so the page is
/// given both labels rather than inventing its own wording for either.
#[test]
fn the_labels_for_a_frameless_point_come_from_the_crate() {
    let mut snapshot = snapshot();
    snapshot.points.push(ProgramPoint {
        kind: PointKind::Overflow,
        frames: Vec::new(),
        ..point(&[], 64, 4)
    });

    let page = page_of(&snapshot);
    let display = parsed(&page, DISPLAY_BLOCK);
    let labels = display.get("labels").unwrap();

    let overflow = labels.get("overflow").unwrap().as_str().unwrap();
    let unwalkable = labels.get("unwalkable").unwrap().as_str().unwrap();
    assert!(overflow.contains("overflow"), "{overflow}");
    assert!(unwalkable.contains("unwalkable"), "{unwalkable}");
    assert_ne!(overflow, unwalkable);
}

/// A run that recorded nothing still produces a page that opens, because the
/// alternative is a reader who cannot tell an empty profile from a broken one.
#[test]
fn a_run_with_no_allocations_still_produces_a_page() {
    let page = page_of(&Snapshot::default());

    assert!(page.starts_with("<!doctype html>"));
    assert!(page.trim_end().ends_with("</html>"));
    let profile = parsed(&page, PROFILE_BLOCK);
    assert_eq!(profile.get("points").unwrap().as_array().unwrap().len(), 0);
    let display = parsed(&page, DISPLAY_BLOCK);
    assert_eq!(display.get("names").unwrap().as_array().unwrap().len(), 0);
}

/// The rendering the page gets is the one every other emitter gets, so a profile
/// asked for as HTML and as DHAT names its frames identically.
#[test]
fn the_page_and_the_dhat_file_name_frames_the_same_way() {
    let snapshot = snapshot();

    let mut dhat = Vec::new();
    snapshot
        .write_dhat_v2_with(&mut dhat, &RawAddresses)
        .unwrap();
    let dhat = String::from_utf8(dhat).unwrap();

    let mut out = Vec::new();
    snapshot.write_html_with(&mut out, &RawAddresses).unwrap();
    let page = String::from_utf8(out).unwrap();
    let display = parsed(&page, DISPLAY_BLOCK);

    for name in display.get("names").unwrap().as_array().unwrap() {
        let name = name.as_str().unwrap();
        assert!(
            dhat.contains(name),
            "the DHAT file does not carry the frame the page calls {name:?}"
        );
    }
}
