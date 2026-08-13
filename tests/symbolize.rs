//! `heapscope-symbolize`, run as a program against a profile of this test.
//!
//! The unit tests inside the binary check each half against text they supply:
//! what three symbolizers' output looks like, and what a rewrite does to a
//! profile. Neither can check the thing the tool exists for, which is that the
//! numbers a profile records really do resolve to the function that allocated —
//! against a real binary, through a real symbolizer, with the file addresses the
//! module map computed.
//!
//! So this records a profile of itself, runs the tool over it, and requires a
//! function defined below to come back named. That closes the loop the README
//! opens: **52 of 52 frames named in-process on macOS aarch64, 0 of 70 on Linux
//! aarch64.** On the second of those the profile is a wall of addresses until
//! something does this, and a test that only ran where in-process symbolization
//! already worked would be checking the platform that does not need the tool.
//!
//! It has to be an integration test because it installs a `#[global_allocator]`,
//! and a single recording test because there is one engine per process — the
//! same constraint `tests/end_to_end.rs` documents.

mod support;

use std::hint::black_box;
use std::path::Path;
use std::process::Command;

use support::json::{self, Value};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// The binary under test, as Cargo built it.
const SYMBOLIZE: &str = env!("CARGO_BIN_EXE_heapscope-symbolize");

/// The function whose name has to come back. `#[inline(never)]` so that it is a
/// frame at all, and named distinctly enough that finding it in the output is
/// not an accident.
#[inline(never)]
fn allocate_from_a_function_with_a_findable_name(count: usize) -> Vec<Vec<u8>> {
    let mut kept = Vec::with_capacity(count);
    for _ in 0..count {
        let mut block: Vec<u8> = Vec::with_capacity(4096);
        block.resize(4096, 0x5A);
        kept.push(black_box(block));
    }
    kept
}

fn run(arguments: &[&str]) -> std::process::Output {
    Command::new(SYMBOLIZE)
        .args(arguments)
        .output()
        .unwrap_or_else(|error| panic!("cannot run {SYMBOLIZE}: {error}"))
}

/// Whether this environment can spawn a symbolizer at all.
///
/// `ci/windows-under-wine.sh` runs the suite as Windows binaries under Wine, so
/// every process spawned must be a Windows one and no symbolizer installed in
/// the Linux container can be reached. That is a property of the harness, which
/// is why the harness says so itself rather than this test trying to detect it.
fn can_spawn_tools() -> bool {
    std::env::var_os("HEAPSCOPE_NO_SUBPROCESS_TOOLS").is_none()
}

fn a_symbolizer_is_installed() -> bool {
    ["llvm-symbolizer", "addr2line", "atos"].iter().any(|tool| {
        Command::new(tool)
            .arg("--version")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
    })
}

/// One test, doing everything, because the process has one engine.
#[test]
#[cfg_attr(
    miri,
    ignore = "needs a real backtrace, a real filesystem, and a subprocess"
)]
fn a_recorded_profile_resolves_to_the_function_that_allocated() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let recorded = directory.path().join("profile.native.json");

    // ---- record ----
    {
        let profiler = heapscope::Profiler::builder()
            .output(heapscope::Output::native(recorded.clone()))
            // Untrimmed, so the profile carries every captured frame. Trimming
            // reads names, and on Linux there are none at record time — which is
            // the whole reason this tool exists — so a trimmed profile would be
            // trimmed on one platform and not the other and this test would be
            // asking a different question on each.
            .trim_frames(false)
            .build()
            .expect("the profiler starts");
        let kept = allocate_from_a_function_with_a_findable_name(64);
        assert_eq!(kept.len(), 64);
        drop(profiler);
    }

    let profile = std::fs::read_to_string(&recorded).expect("the profile was written");
    let parsed = json::parse(&profile).expect("the profile is JSON");
    let recorded_frames = frames(&parsed).len();
    assert!(recorded_frames > 0, "the profile has no frames: {profile}");

    // ---- the shapes that need no symbolizer ----
    a_dhat_file_is_refused_by_name(directory.path());
    an_unknown_version_is_refused(directory.path());

    if !can_spawn_tools() {
        eprintln!("skipping the resolution checks: the harness cannot spawn a symbolizer");
        return;
    }
    // Hard failure rather than a skip, on the same terms as `tests/end_to_end.rs`:
    // "offline symbolization works end to end" is an exit criterion, and a
    // criterion that quietly stops being checked has stopped being one.
    assert!(
        a_symbolizer_is_installed(),
        "no symbolizer is installed, so nothing here was verified"
    );

    // ---- resolve ----
    let resolved_path = directory.path().join("resolved.json");
    let outcome = run(&[
        &recorded.to_string_lossy(),
        "-o",
        &resolved_path.to_string_lossy(),
    ]);
    assert!(
        outcome.status.success(),
        "the tool failed:\n{}",
        String::from_utf8_lossy(&outcome.stderr)
    );

    let resolved_text = std::fs::read_to_string(&resolved_path).expect("the resolved profile");
    let resolved = json::parse(&resolved_text).expect("the resolved profile is JSON");

    // **The property.** The function that made the allocations is named.
    let named: Vec<&str> = frames(&resolved)
        .iter()
        .filter_map(|frame| frame.get("function").and_then(Value::as_str))
        .collect();
    assert!(
        named
            .iter()
            .any(|name| name.contains("allocate_from_a_function_with_a_findable_name")),
        "the function that allocated is not among the {} names resolved:\n{named:#?}\n{}",
        named.len(),
        String::from_utf8_lossy(&outcome.stderr)
    );

    // ---- what was already there is still there ----
    assert_eq!(
        frames(&resolved).len(),
        recorded_frames,
        "the frame table changed size, so the indices in `points` no longer mean what they did"
    );
    assert_eq!(
        resolved.get("points"),
        parsed.get("points"),
        "a rewrite changed the counters"
    );
    assert_eq!(
        resolved.get("totals"),
        parsed.get("totals"),
        "a rewrite changed the totals"
    );
    assert_eq!(
        resolved.get("modules"),
        parsed.get("modules"),
        "a rewrite changed the module map"
    );

    // ---- running it again is idempotent ----
    let again_path = directory.path().join("again.json");
    let again = run(&[
        &resolved_path.to_string_lossy(),
        "-o",
        &again_path.to_string_lossy(),
    ]);
    assert!(again.status.success(), "the second run failed");
    assert_eq!(
        std::fs::read_to_string(&again_path).expect("the second rendering"),
        resolved_text,
        "symbolizing an already-symbolized profile changed it"
    );

    // ---- folded output ----
    a_folded_rendering_carries_the_resolved_names(&recorded);
    let_a_missing_binary_be_pointed_somewhere_else(&recorded, directory.path());
}

/// Folded output, with the names the tool just resolved and the trimming those
/// names make possible.
fn a_folded_rendering_carries_the_resolved_names(recorded: &Path) {
    let outcome = run(&[&recorded.to_string_lossy(), "-f", "folded"]);
    assert!(
        outcome.status.success(),
        "folded output failed:\n{}",
        String::from_utf8_lossy(&outcome.stderr)
    );
    let folded = String::from_utf8(outcome.stdout).expect("folded output is UTF-8");
    assert!(!folded.is_empty(), "nothing was written");

    for line in folded.lines() {
        let (stack, count) = line
            .rsplit_once(' ')
            .unwrap_or_else(|| panic!("a folded line with no count: {line:?}"));
        count
            .parse::<u64>()
            .unwrap_or_else(|_| panic!("`{count}` is not a count: {line:?}"));
        assert!(!stack.is_empty(), "a count with no stack: {line:?}");
        for frame in stack.split(';') {
            assert!(!frame.is_empty(), "an empty frame: {line:?}");
        }
    }
    assert!(
        folded.contains("allocate_from_a_function_with_a_findable_name"),
        "the folded output does not name the function that allocated:\n{folded}"
    );

    // Trimming reads names, so it can only happen once they exist. That is the
    // gain this tool buys on a platform where nothing is named at record time,
    // and it is checked two ways because a single one would pass vacuously.
    //
    // First: no stack still *ends* in the allocation path. `Vec::with_capacity`
    // is where a program decided to allocate and stays; the `RawVec`,
    // `alloc::alloc` and `__rust_alloc` frames beneath it are the same on every
    // stack in the process and go.
    for line in folded.lines() {
        let innermost = line
            .rsplit_once(' ')
            .expect("a count")
            .0
            .rsplit(';')
            .next()
            .expect("a frame");
        for machinery in ["__rust_alloc", "alloc::alloc::", "alloc::raw_vec"] {
            assert!(
                !innermost.contains(machinery),
                "a stack still ends in the allocation path: {innermost}"
            );
        }
    }

    // Second: something was actually removed. Compared against the profile's own
    // frame counts rather than a constant, because how deep these stacks run is
    // a property of the build.
    let recorded = json::parse(&std::fs::read_to_string(recorded).expect("the profile"))
        .expect("the profile is JSON");
    let deepest_recorded = recorded
        .get("points")
        .and_then(Value::as_array)
        .expect("points")
        .iter()
        .filter_map(|point| point.get("frames").and_then(Value::as_array))
        .map(<[Value]>::len)
        .max()
        .expect("at least one point");
    let deepest_folded = folded
        .lines()
        .map(|line| line.rsplit_once(' ').expect("a count").0.split(';').count())
        .max()
        .expect("at least one stack");
    assert!(
        deepest_folded < deepest_recorded,
        "the deepest stack is {deepest_folded} frames folded and {deepest_recorded} \
         recorded, so nothing was trimmed:\n{folded}"
    );

    // The name is added to the image and offset rather than replacing them, so
    // a symbolized profile stays resolvable all over again.
    assert!(
        folded.contains("+0x"),
        "the file attribution was dropped once a name was found:\n{folded}"
    );
}

/// A profile recorded on another machine names images this one does not have.
/// `--binary OLD=NEW` is what makes that case work, and it is the case the
/// recorded build identity exists for.
fn let_a_missing_binary_be_pointed_somewhere_else(recorded: &Path, directory: &Path) {
    let text = std::fs::read_to_string(recorded).expect("the profile");
    let parsed = json::parse(&text).expect("JSON");
    let executable = std::env::current_exe().expect("this test binary");
    let executable = executable.to_string_lossy().into_owned();

    // Rewrite the module map to claim the binary lives somewhere it does not,
    // which is what a profile from another machine looks like.
    let elsewhere = "/nowhere/that/exists/test-binary";
    let moved = text.replace(&json_escape(&executable), &json_escape(elsewhere));
    assert_ne!(moved, text, "this binary is not in its own module map");
    let moved_path = directory.join("moved.native.json");
    std::fs::write(&moved_path, &moved).expect("the moved profile");
    assert!(json::parse(&moved).is_ok(), "the rewrite broke the JSON");
    let _ = parsed;

    // Without the mapping the image cannot be read, and the tool says so rather
    // than reporting success over a profile it changed nothing in.
    let blind = run(&[&moved_path.to_string_lossy(), "-o", "/dev/null"]);
    assert!(
        !blind.status.success(),
        "resolving nothing at all reported success"
    );
    let complaint = String::from_utf8_lossy(&blind.stderr);
    assert!(
        complaint.contains("no such file here") || complaint.contains("resolved none"),
        "the failure does not say the image was unreadable: {complaint}"
    );

    // With it, the same profile resolves.
    let pointed = directory.join("pointed.json");
    let outcome = run(&[
        &moved_path.to_string_lossy(),
        "--binary",
        &format!("{elsewhere}={executable}"),
        "-o",
        &pointed.to_string_lossy(),
    ]);
    assert!(
        outcome.status.success(),
        "`--binary` did not rescue the profile:\n{}",
        String::from_utf8_lossy(&outcome.stderr)
    );
    let resolved = json::parse(&std::fs::read_to_string(&pointed).expect("the profile"))
        .expect("the profile is JSON");
    assert!(
        frames(&resolved)
            .iter()
            .filter_map(|frame| frame.get("function").and_then(Value::as_str))
            .any(|name| name.contains("allocate_from_a_function_with_a_findable_name")),
        "`--binary` resolved nothing useful"
    );
}

/// A DHAT file has no addresses left to resolve, and it is by far the likeliest
/// wrong file to be handed — it is the one this crate writes by default.
fn a_dhat_file_is_refused_by_name(directory: &Path) {
    let path = directory.join("dhat-heap.json");
    std::fs::write(&path, r#"{"dhatFileVersion":2,"mode":"heap","pps":[]}"#).expect("a DHAT file");
    let outcome = run(&[&path.to_string_lossy()]);
    assert!(!outcome.status.success(), "a DHAT file was accepted");
    let complaint = String::from_utf8_lossy(&outcome.stderr);
    assert!(complaint.contains("DHAT"), "{complaint}");
    assert!(complaint.contains("native"), "{complaint}");
}

/// The other half of the rule every profile states about itself: *refuse a
/// `formatVersion` you do not know*. A tool that tried anyway would be writing
/// into a frame table that may mean something else.
fn an_unknown_version_is_refused(directory: &Path) {
    let path = directory.join("future.native.json");
    std::fs::write(
        &path,
        r#"{"format":"heapscope-profile","formatVersion":99,"frames":[],"points":[]}"#,
    )
    .expect("a future profile");
    let outcome = run(&[&path.to_string_lossy()]);
    assert!(!outcome.status.success(), "a future version was accepted");
    assert!(
        String::from_utf8_lossy(&outcome.stderr).contains("99"),
        "{}",
        String::from_utf8_lossy(&outcome.stderr)
    );
}

fn frames(profile: &Value) -> &[Value] {
    profile
        .get("frames")
        .and_then(Value::as_array)
        .expect("the profile has a frame table")
}

/// A path as it appears inside a JSON string, so a textual substitution lands on
/// the same bytes the file has.
fn json_escape(text: &str) -> String {
    text.replace('\\', r"\\").replace('"', r#"\""#)
}

/// `--help` and `--version` answer without needing a profile, which is what
/// makes the tool usable from a shell that has not been told anything yet.
#[test]
#[cfg_attr(miri, ignore = "spawns a subprocess")]
fn the_tool_describes_itself() {
    let help = run(&["--help"]);
    assert!(help.status.success());
    let text = String::from_utf8_lossy(&help.stdout);
    assert!(text.contains("--binary"), "{text}");
    assert!(text.contains("inferno-flamegraph"), "{text}");

    let version = run(&["--version"]);
    assert!(version.status.success());
    assert!(
        String::from_utf8_lossy(&version.stdout).contains(env!("CARGO_PKG_VERSION")),
        "the tool and the crate report different versions"
    );

    // A run with no arguments is a usage error rather than a silent success.
    let nothing = run(&[]);
    assert!(!nothing.status.success());

    // An option that does not exist is named, rather than ignored.
    let wrong = run(&["--tool", "nm", "profile.json"]);
    assert!(!wrong.status.success());
    assert!(
        String::from_utf8_lossy(&wrong.stderr).contains("llvm-symbolizer"),
        "the complaint does not list the tools it knows"
    );
}
