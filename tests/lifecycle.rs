//! One test per row of PLAN.md section 4.6.
//!
//! The rows are the profiler's contract with the surrounding process: what
//! happens to a profile when the program exits abruptly, forks, panics, or is
//! aborted. They are the part of a heap profiler people discover by being
//! disappointed, so each row gets a test that produces the real condition in a
//! real process rather than a unit test of the code that is *supposed* to handle
//! it.
//!
//! # Not run under Miri
//!
//! Every test here spawns `examples/lifecycle_probe` and inspects what the
//! process left behind. Miri interprets a program rather than executing one, so
//! there is nothing for it to spawn — the first test to try reports `lstat` (or
//! `mkdir`, or the spawn itself) as unavailable under its filesystem isolation.
//!
//! Gated at the file rather than per test because no subset of this file is
//! runnable: producing a real dying process is the whole method. That also
//! matters more than it looks — an unsupported operation aborts the entire test
//! binary, so one un-gated test here would hide the other fourteen and every
//! test in whichever suite ran after it.

#![cfg(not(miri))]

//! Rows that are about the engine's counters rather than the process are tested
//! where they live — the reference-tracker differential suite for `realloc`
//! attribution, `core::engine` for table exhaustion and poisoning — and are
//! checked here only for their effect on a real profile.
//!
//! The fixture is `examples/lifecycle_probe.rs`. Everything below runs it.
//!
//! # Run these with plain `cargo test`
//!
//! `cargo test --test lifecycle` rebuilds this file but **not** the example, so
//! it would run the previous build of the library. These tests refuse to run
//! against a stale fixture rather than reporting on one — see [`assert_fresh`],
//! which exists because a deliberately broken `fork` implementation once passed
//! this whole file for exactly that reason.

mod support;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use support::json;

/// How long a probe may run before it is treated as wedged.
///
/// Generous — the slowest mode forks two dozen times — but finite. The probe
/// bounds the wait for its own children; nothing bounded the wait for the probe
/// itself, so a mutation that wedged the *parent* left `cargo test` running for
/// ten minutes with no output. This file's own documentation says a test that
/// detects a deadlock by never finishing is not a test; that reasoning applies
/// one level up too.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Runs the probe in `mode`, writing its profile to a fresh path.
///
/// Returns the process result and the path, which may or may not exist: several
/// rows are specifically about a profile *not* being written.
fn run(mode: &str) -> (Output, PathBuf) {
    let directory = temporary_directory(mode);
    let output_path = directory.join("dhat-heap.json");

    let mut child = Command::new(probe_binary())
        .arg(mode)
        .arg(&output_path)
        // Every path handed to the probe is absolute, so the working directory
        // matters for exactly one thing: a profile written to a *relative*
        // default, which is what an exit handler nobody disarmed does. Pointing
        // it at the temporary directory makes that visible instead of dropping a
        // `dhat-heap.json` into the repository.
        .current_dir(&directory)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("could not run the lifecycle probe: {error}"));

    let deadline = std::time::Instant::now() + PROBE_TIMEOUT;
    loop {
        match child
            .try_wait()
            .expect("could not poll the lifecycle probe")
        {
            Some(_) => break,
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "the lifecycle probe in mode {mode:?} did not finish within \
                     {PROBE_TIMEOUT:?}, which is what a wedged profiler looks like"
                );
            }
            None => std::thread::sleep(std::time::Duration::from_millis(5)),
        }
    }

    let result = child
        .wait_with_output()
        .expect("could not collect the probe's output");
    (result, output_path)
}

/// Runs the probe and requires a clean exit, reporting its output if not.
fn run_expecting_success(mode: &str) -> (Output, PathBuf) {
    let (result, path) = run(mode);
    assert!(
        result.status.success(),
        "the probe failed in mode {mode:?}: {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        result.status,
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr),
    );
    (result, path)
}

/// The `heapscope` extension object from a written profile.
///
/// Also validates the native profile written beside it. Every probe run writes
/// both, from one reading of the engine, so a row that produces a DHAT file this
/// harness is willing to read produces a native one too — and putting the check
/// here rather than in a test of its own is what makes it cover every row rather
/// than the one row someone remembered.
fn extension(path: &Path) -> json::Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("no profile at {}: {error}", path.display()));
    let profile = json::parse(&text).expect("the profile is valid JSON");
    support::dhat::assert_valid(&text);
    support::native::assert_valid(&native_text(path));
    profile
        .get("heapscope")
        .expect("the profile has a heapscope section")
        .clone()
}

/// The native profile written beside `path`.
fn native_text(path: &Path) -> String {
    let path = path.with_extension("native.json");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("no native profile at {}: {error}", path.display()))
}

/// The native profile written beside `path`, parsed.
fn native(path: &Path) -> json::Value {
    let text = native_text(path);
    support::native::assert_valid(&text);
    json::parse(&text).expect("the native profile is valid JSON")
}

/// Checks the two program points a non-heap probe run produced against the shape
/// the two reports had.
///
/// Reads the **untrimmed** companion profile, which is the frames as captured.
/// The default rendering removes the frames every stack shares, so a capture
/// that wrongly began inside `heapscope` would have that frame trimmed away and
/// look correct here.
///
/// The two reports go through one function one call apart, so the recorded
/// stacks must differ by exactly one frame with the innermost unchanged. That is
/// `unwind::calibrate`'s argument and it needs no assumption about code layout.
/// The address check that follows is the other direction — an extra frame inside
/// the profiler shifts both stacks equally and is invisible to the comparison —
/// and it does need one, so it is the weaker of the two.
///
/// This used to live inside the probe, reading `profiler.snapshot()` directly. A
/// check performed only by the thing being checked is covered by nothing:
/// widening its tolerance to `usize::MAX`, or returning from it before it did
/// anything, left the entire suite green.
fn assert_two_reports_one_call_apart(profile: &json::Value, stdout: &str) {
    /// How far past its entry point a return address may land and still be
    /// inside the reporting function, whose body is a single call.
    const SPAN: u64 = 8_192;

    let points = profile
        .get("pps")
        .and_then(json::Value::as_array)
        .expect("the profile has program points");
    let mut stacks: Vec<&[json::Value]> = points
        .iter()
        .filter_map(|point| point.get("fs").and_then(json::Value::as_array))
        .collect();
    stacks.sort_by_key(|frames| frames.len());
    let [shallow, deep] = stacks[..] else {
        panic!(
            "two call sites reported one event each and produced {} program \
             points",
            stacks.len()
        );
    };
    assert_eq!(
        deep.len(),
        shallow.len() + 1,
        "one more call has to mean exactly one more frame: {deep:?} against \
         {shallow:?}"
    );
    assert_eq!(
        deep.first().and_then(json::Value::as_u64),
        shallow.first().and_then(json::Value::as_u64),
        "the two reports came from the same line of the same function, so the \
         innermost frame must be the same one"
    );

    // And that shared innermost frame is inside the function that reported,
    // rather than one frame in either direction from it. The probe prints the
    // address it must be inside; the frame table holds the address recorded.
    let site = stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix("report-site 0x"))
        .and_then(|hex| u64::from_str_radix(hex.trim(), 16).ok())
        .unwrap_or_else(|| panic!("the probe did not report its call site:\n{stdout}"));
    let table = profile
        .get("ftbl")
        .and_then(json::Value::as_array)
        .expect("the profile has a frame table");
    let index = shallow.first().and_then(json::Value::as_u64).unwrap() as usize;
    let frame = table[index].as_str().expect("a frame is a string");
    let recorded = frame
        .strip_prefix("0x")
        .and_then(|rest| rest.split(':').next())
        .and_then(|hex| u64::from_str_radix(hex, 16).ok())
        .unwrap_or_else(|| panic!("the frame does not begin with an address: {frame}"));
    assert!(
        recorded >= site && recorded - site < SPAN,
        "an event was attributed to {recorded:#x}, which is not inside the \
         function that reported it ({site:#x}); every program point would begin \
         with heapscope's own frames"
    );
}

/// The companion profile written beside `path` with nothing trimmed.
fn untrimmed(path: &Path) -> json::Value {
    let path = path.with_extension("untrimmed.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("no untrimmed profile at {}: {error}", path.display()));
    support::dhat::assert_valid(&text);
    json::parse(&text).expect("the profile is valid JSON")
}

/// Which path produced the profile at `path`.
fn shutdown_path(path: &Path) -> String {
    extension(path)
        .get("shutdown")
        .and_then(json::Value::as_str)
        .expect("the profile records how recording ended")
        .to_owned()
}

// --- The rows -------------------------------------------------------------

/// Rows: allocation before start (ignored, no underflow), allocation live at
/// stop (counted in `eb`/`ebk`), and the ordinary `Profiler::drop` ending.
#[test]
fn dropping_the_profiler_writes_a_profile_taken_before_teardown() {
    let (_, path) = run_expecting_success("drop");
    assert_eq!(shutdown_path(&path), "drop");

    let extension = extension(&path);
    let totals = extension.get("totals").expect("totals");
    let live_bytes = totals
        .get("currBytes")
        .and_then(json::Value::as_u64)
        .unwrap();
    let total_bytes = totals
        .get("totalBytes")
        .and_then(json::Value::as_u64)
        .unwrap();

    // The probe holds 33 blocks at the end and frees far more than it holds.
    assert!(
        live_bytes > 0,
        "blocks live at stop must be counted in `eb`"
    );
    assert!(
        total_bytes > live_bytes,
        "the probe frees more than it holds, so the cumulative total must exceed \
         the live total: total={total_bytes}, live={live_bytes}"
    );

    // The pre-start blocks are freed while recording. Their frees find no entry.
    // If those frees were counted, live bytes would have gone negative and
    // wrapped — the exact failure a saturating subtraction hides.
    assert!(
        live_bytes < 1 << 40,
        "live bytes look like an underflow that wrapped: {live_bytes}"
    );

    // Instrumentation left in a program profiled the ordinary way. This is the
    // likeliest way to hold the feature wrong, and the failure it guards is not
    // a missing number but a wrong one: a heap profile whose `tb` mixed bytes
    // with dimensionless ad hoc weights would look entirely normal.
    //
    // Must match `MISDIRECTED_CALLS` in `examples/lifecycle_probe.rs`, doubled:
    // the probe calls both reporting functions.
    assert_eq!(
        extension.get("refusedEvents").and_then(json::Value::as_u64),
        Some(6),
        "a heap run did not refuse the events reported to it"
    );
    assert!(
        total_bytes < 6_000_000,
        "a heap run recorded the reported weights into its own byte total: \
         {total_bytes}"
    );
}

/// What only a real run can show about the native format: the numbers that come
/// from measurement rather than from a hand-built snapshot.
///
/// Every hand-built snapshot in `tests/native_output.rs` sets its own shapes and
/// its own self-metrics, so all of those tests would pass with the engine
/// wired to none of it. This one runs a program that allocates, reallocates, and
/// asks for zeroed memory, and requires the file to say so.
#[test]
fn a_real_run_writes_a_native_profile_of_what_it_actually_did() {
    let (_, path) = run_expecting_success("drop");
    let native = native(&path);

    let at = |path: &str| -> u64 {
        let mut current = native.clone();
        for step in path.split('.') {
            current = current
                .get(step)
                .unwrap_or_else(|| panic!("the native profile has no `{path}`"))
                .clone();
        }
        current
            .as_u64()
            .unwrap_or_else(|| panic!("`{path}` is not an integer"))
    };

    // The probe's workload allocates zeroed blocks (`vec![0u8; n]`) and grows a
    // vector, so both counters describe something that happened. Without the
    // shim passing the shape through, both would be zero and every hand-built
    // test would still pass.
    assert!(
        at("shapes.zeroed.blocks") > 0,
        "the probe allocates zeroed blocks and the profile counted none, so the \
         shim is not telling the engine which method the program called"
    );
    assert!(
        at("shapes.reallocs.count") > 0,
        "the probe grows a vector and the profile counted no reallocations"
    );
    // `count` alone proves almost nothing: a shim handing `record_realloc` the
    // *new* address as `old_address` reports every reallocation as a resize in
    // place, and the whole suite stayed green when it did. Worse, a debug build
    // of a doubling `Vec` genuinely produces `moved: 0`, so zero here is not
    // even suspicious to a reader. The probe now forces one reallocation the
    // allocator cannot satisfy in place.
    assert!(
        at("shapes.reallocs.moved") > 0,
        "no reallocation moved, so nothing distinguishes the shim passing the \
         old address from it passing the new one"
    );
    assert!(
        at("shapes.reallocs.bytesCopied") > 0,
        "a reallocation moved and copied nothing, which is not what moving means"
    );
    assert!(
        at("shapes.observedBlocks") > 0,
        "a heap run observed no allocation requests at all"
    );

    // Self-metrics that only a running engine has. An arena that reserved
    // nothing is a profiler that recorded nothing.
    assert!(
        at("selfMetrics.arena.bytesReserved") > 0,
        "the profile reports an arena that never reserved anything"
    );
    assert!(
        at("selfMetrics.programPoints.entries") > 0,
        "the profile reports a program-point table with nothing in it"
    );
    assert!(
        at("selfMetrics.captureCost.captures") > 0,
        "no capture cost was measured, so the profile's overhead figures are \
         missing the one number that has to be timed"
    );

    // The two files describe one reading of the engine, so their totals are the
    // same number rather than two numbers that ought to agree.
    let dhat = extension(&path);
    let dhat_at = |field: &str| -> u64 {
        dhat.get("totals")
            .and_then(|totals| totals.get(field))
            .and_then(json::Value::as_u64)
            .unwrap_or_else(|| panic!("the DHAT file has no `totals.{field}`"))
    };
    assert_eq!(dhat_at("totalBytes"), at("totals.totalBytes"));
    assert_eq!(dhat_at("maxBytes"), at("totals.maxBytes"));
    assert_eq!(dhat_at("currBytes"), at("totals.currBytes"));
}

/// Which thread, and which phase — the two questions DHAT v2 has no field for,
/// against a real run rather than a hand-built snapshot.
///
/// The worker thread has exited by the time the profile is written, which is
/// the whole reason its name is captured when it first allocates. A profile
/// that named it "main", or did not name it, would be the same profile a shim
/// asking the platform at output time produces.
#[test]
fn a_real_run_names_the_threads_and_phases_that_allocated() {
    let (_, path) = run_expecting_success("drop");
    let native = native(&path);

    let rows = |key: &str| -> Vec<json::Value> {
        native
            .get(key)
            .and_then(|value| value.as_array().map(<[json::Value]>::to_vec))
            .unwrap_or_else(|| panic!("the native profile has no array `{key}`"))
    };
    let named = |row: &json::Value| -> String {
        row.get("name")
            .and_then(json::Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let number = |row: &json::Value, field: &str| -> u64 {
        row.get(field)
            .and_then(json::Value::as_u64)
            .unwrap_or_else(|| panic!("a row has no `{field}`"))
    };

    let threads = rows("threads");
    assert_eq!(
        threads.len(),
        2,
        "the probe allocates on its main thread and on one worker; the profile \
         has {} rows",
        threads.len()
    );
    let worker = threads
        .iter()
        .find(|row| named(row).starts_with("hs-worker"))
        .unwrap_or_else(|| {
            panic!(
                "no row is named after the worker thread; got {:?}",
                threads.iter().map(named).collect::<Vec<_>>()
            )
        });
    assert!(
        number(worker, "currBytes") > 0,
        "the worker's blocks outlive it and are still live at the end, so its \
         row must still hold them after the thread has gone"
    );

    // Every recorded allocation belongs to exactly one thread. The validator
    // checks this too; it is here as well because this is the run where the
    // rows come from a real engine rather than a literal.
    let summed: u64 = threads.iter().map(|row| number(row, "totalBytes")).sum();
    let total = native
        .get("totals")
        .and_then(|totals| totals.get("totalBytes"))
        .and_then(json::Value::as_u64)
        .expect("the profile has totals");
    assert_eq!(summed, total, "the thread rows do not sum to the run");

    let regions = rows("regions");
    let names: Vec<String> = regions.iter().map(named).collect();
    for expected in ["parsing", "parsing/lexing", "worker"] {
        assert!(
            names.iter().any(|name| name == expected),
            "the probe enters a region named {expected:?} and the profile has \
             no row for it; got {names:?}"
        );
    }

    // Nesting is exclusive: an allocation belongs to the innermost open region
    // and to that one only. The probe's inner region allocates more bytes than
    // its outer one, so an emitter that folded the inner into the outer would
    // report the outer as the larger of the two.
    let bytes = |name: &str| -> u64 {
        let row = regions
            .iter()
            .find(|row| named(row) == name)
            .expect("the region row exists");
        number(row, "totalBytes")
    };
    assert!(
        bytes("parsing/lexing") > bytes("parsing"),
        "the inner region's bytes were counted in the outer one as well: \
         parsing has {} and parsing/lexing has {}",
        bytes("parsing"),
        bytes("parsing/lexing")
    );
    for row in &regions {
        assert_eq!(
            number(row, "active"),
            0,
            "a region guard outlived the profiler in a run where every one is \
             scoped"
        );
        assert!(number(row, "entries") > 0);
    }
    assert!(
        bytes("worker") > 0,
        "a region entered on a worker thread recorded nothing, so regions are \
         not per-thread state"
    );
}

/// The native file keeps the addresses, and keeps them resolvable.
///
/// A DHAT frame is a rendered string, so a tool wanting the address has to parse
/// it back out of text meant to be read. Here it is an address, an image, and a
/// file address as separate answers — and the file address has to be the number
/// `addr2line` takes, not an offset from the load address, or offline
/// resolution silently names the wrong function.
#[test]
fn a_native_profile_keeps_addresses_resolvable_against_its_module_map() {
    let (_, path) = run_expecting_success("drop");
    let native = native(&path);

    let modules = native
        .get("modules")
        .and_then(json::Value::as_array)
        .expect("the profile carries a module map");
    let frames = native
        .get("frames")
        .and_then(json::Value::as_array)
        .expect("the profile carries a frame table");
    assert!(!frames.is_empty(), "a real run captured no frames");

    let mut attributed = 0;
    for frame in frames {
        let address = frame
            .get("addr")
            .and_then(json::Value::as_str)
            .and_then(|text| usize::from_str_radix(text.trim_start_matches("0x"), 16).ok())
            .expect("every frame has a hexadecimal address");
        let Some(index) = frame.get("module").and_then(json::Value::as_u64) else {
            continue;
        };
        attributed += 1;

        let module = &modules[index as usize];
        // Strictly a hexadecimal string, not "either representation". Accepting
        // both is how this helper was written first, and it would have passed
        // just as happily against the emitter writing the JSON numbers the
        // format exists to avoid.
        let module_address = |field: &str| -> usize {
            let text = module
                .get(field)
                .and_then(json::Value::as_str)
                .unwrap_or_else(|| panic!("`{field}` is not a string"));
            usize::from_str_radix(
                text.strip_prefix("0x")
                    .unwrap_or_else(|| panic!("`{field}` is {text:?}, not `0x` hexadecimal")),
                16,
            )
            .expect("a hexadecimal module address")
        };
        let size = module
            .get("size")
            .and_then(json::Value::as_u64)
            .expect("a module's size is a count, and stays a number") as usize;
        let (start, bias) = (module_address("start"), module_address("bias"));
        assert!(
            address >= start && address - start < size,
            "frame {address:#x} was attributed to an image covering \
             {start:#x}..{:#x}",
            start + size
        );

        let file_address = frame
            .get("fileAddr")
            .and_then(json::Value::as_str)
            .and_then(|text| usize::from_str_radix(text.trim_start_matches("0x"), 16).ok())
            .expect("an attributed frame has a file address");
        assert_eq!(
            file_address,
            address.wrapping_sub(bias),
            "the file address is not the runtime address minus the bias, so \
             `addr2line` would resolve it to the wrong function"
        );
    }

    assert!(
        attributed > 0,
        "no frame was attributed to any image, so the module map proves nothing"
    );
}

/// Whether `std::process::exit` reaches the C `atexit` list on this platform.
///
/// On unix it does: `std::process::exit` calls `libc::exit`, which runs the
/// handler list and then terminates. On Windows it calls `ExitProcess`
/// directly (`library/std/src/sys/exit.rs`), which terminates the process
/// without going through the CRT's `exit` — so no `atexit` handler runs, and
/// there is no documented hook that would let one. Returning from `main` is
/// unaffected on both, because there the CRT calls `exit` itself.
///
/// This is a capability difference, not a defect to work around, and the point
/// of expressing it here is that both halves are *checked*: on Windows the test
/// asserts the profile is absent, so if the platform ever gains a hook the
/// suite says so instead of quietly continuing to claim less than it delivers.
const PROCESS_EXIT_RUNS_HANDLERS: bool = cfg!(unix);

/// Row: `std::process::exit`, which runs no destructor on any thread.
#[test]
fn process_exit_produces_a_profile_wherever_the_platform_allows_one() {
    let (_, path) = run_expecting_success("process-exit");
    if PROCESS_EXIT_RUNS_HANDLERS {
        assert!(
            path.exists(),
            "a program that ends in `process::exit` produced no profile at all, \
             which is the single most common way a heap profiler disappoints"
        );
        assert_eq!(shutdown_path(&path), "atexit");
    } else {
        assert!(
            !path.exists(),
            "`ExitProcess` does not run `atexit` handlers, so a profile here \
             means the platform changed and the documentation is now wrong"
        );
    }
}

/// Row: `process::exit` from a thread that is not `main`.
#[test]
fn exiting_from_another_thread_behaves_like_exiting_from_main() {
    let (_, path) = run_expecting_success("exit-from-thread");
    if PROCESS_EXIT_RUNS_HANDLERS {
        assert_eq!(shutdown_path(&path), "atexit");
    } else {
        assert!(!path.exists());
    }
}

/// A profiler that outlives `main` — the shape a `static` profiler has — and
/// the row about what that costs.
///
/// Section 4.6's `atexit` ordering row says the snapshot this path produces is
/// taken **partway through teardown** and will differ from the one `Drop`
/// produces. That is the whole content of the row, and until this test the suite
/// only checked the label: a handler that ran before teardown rather than during
/// it would have written `"atexit"` just as happily and nothing would have
/// noticed.
///
/// The difference is in the live totals, and it is not subtle. `main` has
/// already returned by the time the handler runs, so everything `main` held has
/// been freed and the profile reports a heap emptier than the one the drop path
/// sees. Both runs do identical work, so the cumulative totals are the reading
/// that stays comparable and the live totals are the reading that moves.
#[test]
fn a_forgotten_profiler_is_still_written_at_exit() {
    let (_, path) = run_expecting_success("forget");
    assert_eq!(shutdown_path(&path), "atexit");

    let at_exit = live_bytes(&extension(&path));
    let (_, dropped_path) = run_expecting_success("drop");
    let at_drop = live_bytes(&extension(&dropped_path));

    assert!(
        at_drop > 0,
        "the drop path reports nothing live, so the comparison below is between \
         two zeroes and proves nothing"
    );
    assert!(
        at_exit < at_drop,
        "the exit handler reported {at_exit} bytes live where the drop path \
         reported {at_drop}. The row says this snapshot is taken partway through \
         teardown, and a reading that matches the drop path means it is being \
         taken somewhere else"
    );
}

/// Bytes a profile says were still live when it was taken.
fn live_bytes(extension: &json::Value) -> u64 {
    extension
        .get("totals")
        .and_then(|totals| totals.get("currBytes"))
        .and_then(json::Value::as_u64)
        .expect("a profile records what was live when it was taken")
}

/// Row: panic with unwinding. `Profiler::drop` runs on the way out.
#[test]
fn a_panicking_program_still_writes_its_profile() {
    let (result, path) = run("panic");
    assert!(
        !result.status.success(),
        "the probe was supposed to panic in this mode"
    );
    assert!(
        path.exists(),
        "the profile of a program that panicked is the profile most worth having"
    );
    // Not `atexit`: unwinding reaches `Profiler::drop` first, and the exit
    // handler must not then write a second one over the top.
    assert_eq!(shutdown_path(&path), "drop");
}

/// Row: `abort` bypasses the `atexit` list, so there is no profile.
///
/// The value of this test is that the limitation is *stated and checked*, not
/// discovered by someone whose profile is mysteriously missing.
#[test]
fn abort_writes_no_profile_and_that_is_documented() {
    let (result, path) = run("abort");
    assert!(!result.status.success());
    assert!(
        !path.exists(),
        "`abort` bypasses `atexit`; a profile here would mean something else \
         wrote it, which is not a mechanism this crate has"
    );
}

/// Row: a fatal signal. `SIGKILL` cannot be caught, so this is a statement
/// about what the crate does *not* claim, checked rather than assumed.
#[test]
#[cfg(unix)]
fn a_fatal_signal_writes_no_profile_and_that_is_documented() {
    let (result, path) = run("fatal-signal");
    assert!(!result.status.success());
    assert!(
        !path.exists(),
        "no mechanism in this crate survives SIGKILL; a file here would mean \
         the probe did not actually die"
    );
}

/// Row: allocation after the profiler has stopped. The shim is a pass-through.
#[test]
fn allocations_after_the_profiler_stops_are_not_recorded() {
    // The probe compares the engine's counters either side of two thousand
    // allocations made after the profiler stopped, and fails itself if they
    // moved. Here we only need the profile it wrote to still be valid.
    let (_, path) = run_expecting_success("alloc-after-stop");
    assert_eq!(shutdown_path(&path), "drop");
}

/// Row: concurrent shutdown. Eight threads are inside the shim when the
/// profiler is dropped.
#[test]
fn shutting_down_under_load_neither_hangs_nor_corrupts_the_profile() {
    let (_, path) = run_expecting_success("concurrent-shutdown");
    // `assert_valid` is the substance here: it checks that the per-point columns
    // sum to the engine's own totals, which is the invariant a shutdown racing
    // live events is most likely to break.
    assert_eq!(shutdown_path(&path), "drop");
}

/// Row: an internal invariant violation. Recording stops, the program keeps
/// running, and the profile says it happened.
#[test]
fn a_poisoned_profiler_still_writes_a_profile_that_admits_it() {
    let (result, path) = run_expecting_success("poison");
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("deliberate poison"),
        "a poisoned profiler must say so on stderr: {:?}",
        String::from_utf8_lossy(&result.stderr)
    );

    let extension = extension(&path);
    assert_eq!(
        extension.get("poisoned").and_then(json::Value::as_bool),
        Some(true),
        "a reader deciding whether to trust a surprising figure needs to know \
         the profiler reported its own corruption"
    );
}

/// Row: table capacity exhausted, for the one table a program can fill through
/// the public API.
///
/// The live-block table is bounded because section 4.5 requires it to be, and
/// this is the first run in the suite that reaches the bound. Everything that
/// knows what a full table does — the engine's own tests, the validator's rule
/// that a run with dropped blocks cannot be held to the sum, the emitters'
/// `notRecorded` block — has until now been exercised against hand-built
/// snapshots that asserted a count into existence.
///
/// What must hold is that a block the table could not take is left out
/// *entirely* rather than half-counted. Both are one line apart in
/// `Engine::record_alloc`: recording the allocation and then failing to insert
/// would leave a block that raises the totals, raises the peak, and can never be
/// freed, because its free finds no entry. The observed-request count is what
/// makes that visible — it is taken before the table is consulted, so a run
/// where recorded and dropped do not add up to it has lost or double-counted a
/// block somewhere in between.
///
/// The section 4.6 row also covers the program-point table, which no test can
/// reach this way: `max_live_blocks` is on the builder and the program-point
/// ceiling is not, so filling it means a program with more than a million
/// distinct call sites. That half stays covered by `internals::pp`'s own tests
/// and by the emitter tests that render an overflow point.
#[test]
fn a_run_that_filled_its_live_block_table_says_so_and_stays_coherent() {
    /// Must match `FULL_TABLE_LIVE_BLOCKS` in `examples/lifecycle_probe.rs`. The
    /// per-shard rounding leaves this one unchanged, unlike the `configured`
    /// mode's.
    const LIVE_BLOCKS: u64 = 128;

    let (result, path) = run_expecting_success("full-table");
    let extension = extension(&path);

    assert_eq!(
        extension
            .get("settings")
            .and_then(|settings| settings.get("maxLiveBlocks"))
            .and_then(json::Value::as_u64),
        Some(LIVE_BLOCKS),
        "the ceiling this run was given did not reach the profile"
    );

    let dropped = extension
        .get("droppedBlocks")
        .and_then(json::Value::as_u64)
        .expect("a profile says how many blocks it could not track");
    assert!(
        dropped > 0,
        "the probe held far more blocks than the table can hold and the profile \
         reports none dropped, so either the ceiling was ignored or the count is \
         not wired to it"
    );

    // Blocks were still recorded: a table that dropped *everything* would satisfy
    // the sum below trivially and say nothing about the boundary.
    let native = native(&path);
    let at = |path: &str| -> u64 {
        let mut current = native.clone();
        for key in path.split('.') {
            current = current.get(key).unwrap_or_else(|| panic!("{path}")).clone();
        }
        current
            .as_u64()
            .unwrap_or_else(|| panic!("{path} is not a number"))
    };
    let recorded = at("totals.totalBlocks");
    assert!(
        recorded > 0,
        "no block at all was recorded, so this run says nothing about a table \
         that fills partway through"
    );

    // The whole point of the row, in one equation. `observedBlocks` counts
    // requests before the table is consulted; `totalBlocks` counts what was
    // recorded; `notRecorded.blocks` counts what was turned away.
    let observed = at("shapes.observedBlocks");
    let turned_away = at("notRecorded.blocks");
    assert_eq!(
        recorded + turned_away,
        observed,
        "the run asked for {observed} blocks, recorded {recorded} and turned away \
         {turned_away}, which do not add up: a block was lost or counted twice \
         where the table filled"
    );
    assert_eq!(
        turned_away, dropped,
        "the two profiles disagree about how many blocks this run could not track"
    );

    // Frees of blocks the table never held, in bulk. The probe drops all of them
    // while the table is full, and every one of those frees must find no entry
    // and stop there — the same path a pre-start block's free takes, reached
    // here eight thousand times in a row instead of sixty-four.
    //
    // A free that instead subtracted from whichever point happened to be
    // freeing drives that point's live bytes below zero, and this crate does not
    // hide that: `site::grow_or_shrink` poisons on underflow. So the profile
    // saying it is *not* poisoned is the assertion, and it is a far better one
    // than a live total of zero, which a saturating subtraction would also
    // produce.
    assert_ne!(
        extension.get("poisoned").and_then(json::Value::as_bool),
        Some(true),
        "a run whose table filled poisoned itself, which means a free of a block \
         the table never held reached a counter:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

/// Row: `_exit`, the other way to bypass the handler list.
#[test]
fn raw_exit_writes_no_profile_and_that_is_documented() {
    let (result, path) = run("raw-exit");
    assert!(result.status.success(), "`_exit(0)` is a successful exit");
    assert!(!path.exists(), "`_exit` bypasses `atexit`");
}

/// The remedy for every row above that produces nothing.
///
/// Three rows of section 4.6 — `_exit`, `abort`, and `std::process::exit` on
/// Windows — end a process without walking the `atexit` list, and the README
/// tells a program facing one of them to write the profile itself first. That
/// is a documented remedy, and a documented remedy nobody runs is the failure
/// this repository has already had once, in `ci/dhat-viewer-check.mjs`.
///
/// So the probe calls `Profiler::save_dhat_v2`, `save_native` and `save_html`
/// and then takes the exit. What is being checked is not that the call returns
/// `Ok` — the probe already fails itself on an error — but that the files are
/// **complete** afterwards. Nothing runs after `_exit`: no destructor, no
/// `atexit` handler, no stdio teardown. A writer still holding bytes in a
/// buffer, or one that had not yet renamed its temporary file into place, would
/// leave a missing or truncated profile with no later stage to repair it, and
/// every check below would fail rather than pass quietly.
#[test]
fn a_profile_saved_before_a_bypassing_exit_survives_it() {
    for mode in ["save-then-raw-exit", "save-then-abort"] {
        let (result, path) = run(mode);
        if mode == "save-then-raw-exit" {
            assert!(result.status.success(), "`_exit(0)` is a successful exit");
        } else {
            assert!(!result.status.success(), "`abort` is not a clean exit");
        }

        // `extension` validates both the DHAT file and the native profile
        // beside it, which is what "complete" means here: a truncated file is
        // not valid JSON, and a file whose rename never happened is not there.
        // `running`, not `drop` and not `explicit`: this reading was taken with
        // recording still going, and the field says so. That distinction is the
        // point rather than a detail — the two remedies the README gives differ
        // in exactly this way. Dropping the profiler stops recording and writes
        // an end-of-run profile; saving by hand leaves the run going and yields
        // a point-in-time one, which for a program about to `_exit` is all
        // there was ever going to be. A reader has to be able to tell which of
        // the two they are holding.
        assert_eq!(
            shutdown_path(&path),
            "running",
            "a profile saved by hand mid-run must say it was taken mid-run, so \
             that it is not read as a reading of the finished program ({mode})"
        );

        // And the page, which is the format the advice most often points at:
        // the readers who need this remedy are on the platforms with no
        // `dh_view.html` to open the DHAT file with.
        let page_path = path.with_extension("html");
        let page = std::fs::read_to_string(&page_path)
            .unwrap_or_else(|error| panic!("no page at {}: {error}", page_path.display()));
        assert!(
            page.trim_end().ends_with("</html>"),
            "the page written before {mode} is truncated, which is what an \
             unflushed writer leaves behind"
        );
        // Validated as the page carries it, with the escape left in place. The
        // emitter spells `<` as a JSON unicode escape, which is valid JSON for
        // the same character, so the block parses as it stands -- and undoing
        // the escape by text substitution would corrupt a profile that
        // legitimately contained that spelling.
        let embedded = page
            .split_once(r#"<script type="application/json" id="heapscope-profile">"#)
            .and_then(|(_, rest)| rest.split_once("</script>"))
            .map(|(profile, _)| profile)
            .expect("the page carries a profile");
        support::native::assert_valid(embedded);
    }
}

/// Row: `fork` from a process with other threads inside the allocator shim.
///
/// Twenty-four forks against four threads allocating continuously. Without the
/// `pthread_atfork` handlers a child inherits a lock held by a thread that does
/// not exist in it, and the child's first allocation blocks forever; the probe
/// gives up on such a child rather than hanging the suite.
#[test]
#[cfg(unix)]
fn forking_under_allocation_pressure_leaves_the_child_able_to_allocate() {
    let (_, path) = run_expecting_success("fork");
    assert_eq!(
        shutdown_path(&path),
        "drop",
        "the profile belongs to the parent, which dropped its profiler normally"
    );
}

/// Row: `fork`, from the child's side. The child inherits the profiler value
/// along with the parent's counters, and dropping it must write nothing.
#[test]
#[cfg(unix)]
fn a_forked_child_does_not_write_the_parents_profile() {
    let (_, path) = run_expecting_success("fork-child-drop");
    assert_eq!(shutdown_path(&path), "drop");
}

/// Row: two profilers at once. Checked inside the probe, where a second
/// `Profiler::new` can actually be attempted against a running first one.
#[test]
fn the_probe_refuses_a_second_profiler() {
    // Every successful mode asserts this internally before doing anything else,
    // so any passing mode is also this row's evidence. Naming it as its own test
    // keeps the row from looking uncovered.
    run_expecting_success("drop");
}

/// Everything `ProfilerBuilder` configures, read back out of a real profile.
///
/// Not a section 4.6 row. It is here for the same reason the unwinder test is:
/// a setting is only meaningful against real stacks and a real allocation load,
/// and this file is the only place that runs one. The builder's own bookkeeping
/// — which field a method writes, what `output` replaces — is unit-tested in
/// `src/profiler.rs`; what cannot be tested there is whether any of it reaches
/// the engine, because there is one engine per process.
/// A run that counts what the program reports, not what the allocator sees.
///
/// The probe does the whole allocating workload under this profiler, so the
/// four numbers below are the load-bearing part: if the shim recorded anything
/// at all, `tbk` is in the thousands rather than four. That is the one property
/// a heap-mode test structurally cannot check.
#[test]
fn an_ad_hoc_run_counts_only_what_the_program_reported() {
    /// Must match `AD_HOC_WEIGHTS` in `examples/lifecycle_probe.rs`: one event
    /// from each of two call depths.
    const WEIGHT: u64 = 7 + 7_000;
    const EVENTS: u64 = 2;
    /// Must match `MISDIRECTED_CALLS`.
    const REFUSED: u64 = 3;

    let (result, path) = run_expecting_success("ad-hoc");
    let text = std::fs::read_to_string(&path).expect("the profile is readable");
    let profile = json::parse(&text).expect("valid JSON");

    // The file's own shape. `assert_valid` inside `extension` already refuses a
    // profile whose `verb` or `bklt` contradicts its mode, or that carries a
    // lifetime field it has no lifetimes for; these say what it *is*.
    assert_eq!(
        profile.get("mode").and_then(json::Value::as_str),
        Some("ad-hoc")
    );
    assert_eq!(
        profile.get("bksu").and_then(json::Value::as_str),
        Some("events"),
        "the viewer would label dimensionless weights as bytes"
    );
    let extension = extension(&path);

    let points = profile.get("pps").and_then(json::Value::as_array).unwrap();
    let sum = |field: &str| -> u64 {
        points
            .iter()
            .filter_map(|point| point.get(field).and_then(json::Value::as_u64))
            .sum()
    };
    assert_eq!(
        sum("tb"),
        WEIGHT,
        "the profile does not hold the weights the program reported"
    );
    assert_eq!(
        sum("tbk"),
        EVENTS,
        "the allocator shim recorded {} events of its own in a mode that \
         records none",
        sum("tbk").saturating_sub(EVENTS)
    );

    assert_eq!(
        extension.get("refusedEvents").and_then(json::Value::as_u64),
        Some(REFUSED),
        "calls to the other reporting function were not counted, so a run in \
         the wrong mode looks like a program that reported nothing"
    );

    assert_two_reports_one_call_apart(&untrimmed(&path), &String::from_utf8_lossy(&result.stdout));

    // And the summary, which reads the mode separately from the emitter.
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("mode       ad-hoc") && stderr.contains("occurred   7,007 units"),
        "the summary does not describe an ad hoc run:\n{stderr}"
    );
    assert!(
        !stderr.contains("at t-gmax"),
        "the summary reported a heap peak for a run that has no live blocks:\n{stderr}"
    );
    // The whole sentence, not just that there was one. Rotating the three arms
    // of the clause that says *why* the calls were refused left every assertion
    // in this file green while the summary named the wrong mode.
    assert!(
        stderr.contains(
            "3 calls to heapscope::event or heapscope::copied were refused, \
             because this run counts ad hoc events"
        ),
        "the summary did not say why the calls were refused:\n{stderr}"
    );
    // The time base names itself in prose here and nowhere else. It said
    // "allocation events" until this chunk, in a mode that records none.
    assert!(
        stderr.contains("2 observed events at end"),
        "the summary does not name the time base:\n{stderr}"
    );
}

/// The same machinery under the other verb.
///
/// Separate from the ad hoc test because the mode decides which of the two
/// reporting functions records and which is refused. With one mode tested, a
/// `copied` wired to `Mode::AdHoc` would pass.
#[test]
fn a_copy_run_counts_the_bytes_the_program_said_it_copied() {
    /// Must match `COPIED_BYTES` in `examples/lifecycle_probe.rs`.
    const BYTES: u64 = 1_111 + 333_333;
    const CALLS: u64 = 2;

    let (result, path) = run_expecting_success("copy");
    let text = std::fs::read_to_string(&path).expect("the profile is readable");
    let profile = json::parse(&text).expect("valid JSON");
    let extension = extension(&path);

    assert_eq!(
        profile.get("mode").and_then(json::Value::as_str),
        Some("copy")
    );
    assert_eq!(
        profile.get("verb").and_then(json::Value::as_str),
        Some("Copied")
    );
    // Copy mode really is counting bytes, so it keeps the viewer's defaults —
    // unlike ad hoc, which has to rename them.
    assert!(
        profile.get("bksu").is_none(),
        "copy mode renamed units it has no reason to rename"
    );

    let points = profile.get("pps").and_then(json::Value::as_array).unwrap();
    let copied: u64 = points
        .iter()
        .filter_map(|point| point.get("tb").and_then(json::Value::as_u64))
        .sum();
    let calls: u64 = points
        .iter()
        .filter_map(|point| point.get("tbk").and_then(json::Value::as_u64))
        .sum();
    assert_eq!(copied, BYTES);
    assert_eq!(
        calls, CALLS,
        "the allocator shim recorded blocks of its own"
    );

    assert_eq!(
        extension.get("refusedEvents").and_then(json::Value::as_u64),
        Some(3),
        "heapscope::event was recorded by a run that counts copies"
    );

    let stdout = String::from_utf8_lossy(&result.stdout);
    assert_two_reports_one_call_apart(&untrimmed(&path), &stdout);

    // Copy mode is the one place where "counts bytes" and "has block lifetimes"
    // disagree, so it is the only run that can tell the summary's two mode
    // predicates apart. Rendered in binary units, because these really are
    // bytes; ad hoc weights are not and are rendered as a plain count.
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("mode       copy") && stderr.contains("copied     326.6 KiB in 2"),
        "the summary does not render copied bytes as bytes:\n{stderr}"
    );
    assert!(
        stderr.contains(
            "3 calls to heapscope::event or heapscope::copied were refused, \
             because this run counts copied bytes"
        ),
        "the summary did not say why the calls were refused:\n{stderr}"
    );
}

#[test]
fn a_configured_profiler_records_what_it_was_configured_to() {
    /// Must match `CONFIGURED_DEPTH` in `examples/lifecycle_probe.rs`.
    const DEPTH: usize = 3;
    /// `CONFIGURED_LIVE_BLOCKS` is 5,000; the table rounds each of its 64
    /// shards' share up to a power of two, and the profile reports the ceiling
    /// that rounding produced rather than the request.
    const LIVE_BLOCKS: u64 = 8_192;
    /// Must match `CONFIGURED_TOP`.
    const TOP: usize = 3;

    let (result, path) = run_expecting_success("configured");
    let extension = extension(&path);

    let settings = extension
        .get("settings")
        .expect("the profile records the settings its run had");
    assert_eq!(
        settings.get("maxDepth").and_then(json::Value::as_u64),
        Some(DEPTH as u64),
        "the depth limit did not reach the profile"
    );
    assert_eq!(
        settings.get("maxLiveBlocks").and_then(json::Value::as_u64),
        Some(LIVE_BLOCKS),
        "the live-block ceiling did not reach the profile"
    );
    // `trimFrames` is deliberately not in this block. It is a rendering setting,
    // and this file was written by one particular rendering; what it did is
    // `trimmedFrames`, asserted below.
    assert!(
        settings.get("trimFrames").is_none(),
        "a rendering setting is being reported as though it described this file"
    );

    // The depth limit, in the frames it actually recorded. Read from the file
    // rather than from the setting, because a limit the shim never consulted
    // would still be reported correctly by the setting alone.
    let text = std::fs::read_to_string(&path).expect("the profile is readable");
    let profile = json::parse(&text).expect("valid JSON");
    let points = profile.get("pps").and_then(json::Value::as_array).unwrap();
    let deepest = points
        .iter()
        .filter_map(|point| point.get("fs").and_then(json::Value::as_array))
        .map(|frames| frames.len())
        .max()
        .unwrap_or(0);
    assert!(
        deepest <= DEPTH,
        "a stack of {deepest} frames was recorded under a limit of {DEPTH}"
    );
    assert!(
        deepest > 1,
        "no stack survived the depth limit, so this proves nothing about it"
    );

    // A stack cut short by the limit is the same event as one cut short by the
    // shim's buffer, and the profile has to say so either way. Without this,
    // truncating after the walk rather than before it would pass everything
    // above while quietly claiming every capture reached the outermost frame.
    let truncated = extension
        .get("selfMetrics")
        .and_then(|metrics| metrics.get("captures"))
        .and_then(|captures| captures.get("truncated"))
        .and_then(json::Value::as_u64)
        .expect("the profile counts truncated captures");
    assert!(
        truncated > 0,
        "every capture claimed to be complete under a {DEPTH}-frame limit"
    );

    // `trim_frames(false)`, in the file rather than in the setting.
    assert_eq!(
        extension.get("trimmedFrames").and_then(json::Value::as_u64),
        Some(0),
        "frames were trimmed from a profile configured not to trim"
    );

    // And the second output. `also` is the only way a run produces two
    // destinations from one reading; a builder that dropped it would leave this
    // silent.
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("heapscope profile") && stderr.contains("at t-gmax"),
        "the text summary asked for with `also` was never written:\n{stderr}"
    );
    // A heap run is the only one whose summary names the peak, and the clause
    // that does is on a line no test read: deleting it left the whole suite
    // green while every heap profile lost the instant it peaked at.
    assert!(
        stderr.contains("mode       heap") && stderr.contains("at end, peak at "),
        "the summary does not say when the heap peaked:\n{stderr}"
    );

    // `trim_frames(false)` again, this time in the summary. The two emitters
    // read the setting separately, and deleting the branch in the summary alone
    // left every other assertion in this file green. A summary that trimmed
    // says so in a line of its own; one that did not says nothing.
    assert!(
        !stderr.contains("frames are not shown"),
        "the text summary trimmed frames for a run configured not to:\n{stderr}"
    );

    // `top`. Nothing else in the suite can see it: the builder compares
    // destinations, and every `text_summary_to_stderr` describes itself the same
    // way whatever number it holds.
    let listed = stderr
        .lines()
        .filter(|line| {
            let line = line.trim_start();
            line.starts_with(|c: char| c.is_ascii_digit()) && line.contains(". ")
        })
        .count();
    assert_eq!(
        listed, TOP,
        "the summary listed {listed} program points for a `top` of {TOP}:\n{stderr}"
    );

    // One reading, however many destinations. Writing a profile allocates, so a
    // second `Snapshot::capture` cannot produce the same totals as the first —
    // which is what makes this assertion able to see the difference at all.
    let file_total = extension
        .get("totals")
        .and_then(|totals| totals.get("totalBlocks"))
        .and_then(json::Value::as_u64)
        .expect("the profile records its totals");
    let summary_total = stderr
        .lines()
        .find_map(|line| line.trim().strip_prefix("allocated "))
        .and_then(|line| line.split(" in ").nth(1))
        .and_then(|blocks| blocks.split_whitespace().next())
        .map(|blocks| blocks.replace(',', ""))
        .and_then(|blocks| blocks.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("no `allocated` line in the summary:\n{stderr}"));
    assert_eq!(
        summary_total, file_total,
        "the summary and the profile disagree about the same run, so they were \
         written from two readings rather than one:\n{stderr}"
    );
}

/// The depth limit against the platform's own unwinder.
///
/// Separate from the test above because the two backends spend the caller's
/// buffer differently: unix `backtrace(3)` takes no skip parameter, so the
/// frames it is about to discard come out of the same buffer. A limit at or
/// below the calibrated skip made every capture return nothing, and the profile
/// then blamed a missing frame-pointer chain for a limit the user had set.
#[test]
fn a_depth_limit_survives_the_platform_unwinder() {
    /// Must match `CONFIGURED_DEPTH` in `examples/lifecycle_probe.rs`.
    const DEPTH: usize = 3;

    let (_, path) = run_expecting_success("configured-system");
    let extension = extension(&path);

    let no_frames = extension
        .get("selfMetrics")
        .and_then(|metrics| metrics.get("captures"))
        .and_then(|captures| captures.get("noFrames"))
        .and_then(json::Value::as_u64)
        .expect("the profile counts frameless captures");
    assert_eq!(
        no_frames, 0,
        "a {DEPTH}-frame limit made the platform unwinder return nothing at all"
    );

    let text = std::fs::read_to_string(&path).expect("the profile is readable");
    let profile = json::parse(&text).expect("valid JSON");
    let unwalkable = profile
        .get("ftbl")
        .and_then(json::Value::as_array)
        .expect("a frame table")
        .iter()
        .filter_map(json::Value::as_str)
        .any(|frame| frame.contains("[unwalkable]"));
    assert!(
        !unwalkable,
        "the profile blames a missing frame-pointer chain for a depth limit"
    );

    let points = profile.get("pps").and_then(json::Value::as_array).unwrap();
    let deepest = points
        .iter()
        .filter_map(|point| point.get("fs").and_then(json::Value::as_array))
        .map(|frames| frames.len())
        .max()
        .unwrap_or(0);
    assert!(
        deepest > 1 && deepest <= DEPTH,
        "the deepest stack was {deepest} frames under a limit of {DEPTH}"
    );
}

/// `no_output` has to disarm the process-exit handler, not only the drop path.
///
/// The probe forgets its profiler and exits through the handler, in a working
/// directory of its own. A handler still armed with the default destination
/// writes `dhat-heap.json` there — which is exactly what it did in the
/// repository root when this was tested only against the builder's in-memory
/// list.
#[test]
fn no_output_writes_nothing_even_when_the_profiler_is_never_dropped() {
    let (_, path) = run_expecting_success("no-output");
    assert!(
        !path.exists(),
        "a profiler built with `no_output` wrote {}",
        path.display()
    );

    let directory = path.parent().expect("the probe ran somewhere");
    let stray: Vec<_> = std::fs::read_dir(directory)
        .expect("the probe's directory is readable")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name())
        .collect();
    assert!(
        stray.is_empty(),
        "`no_output` left the exit handler armed: {stray:?}"
    );
}

/// The platform's own unwinder, end to end.
///
/// Not a section 4.6 row. It is here because this file already knows how to run
/// a real program and validate what it wrote, and because the interesting
/// failure — the platform returning nothing, so that every allocation lands on
/// one empty program point — only shows up in a profile.
#[test]
fn the_system_unwinder_produces_a_usable_profile() {
    let (_, path) = run_expecting_success("system-unwinder");
    let extension = extension(&path);

    assert_eq!(
        extension.get("unwinder").and_then(json::Value::as_str),
        Some("system"),
        "the profile must record which unwinder captured its frames"
    );

    // The point of the exercise: real stacks, not one empty program point that
    // absorbed the whole run.
    let text = std::fs::read_to_string(&path).expect("the profile is readable");
    let profile = json::parse(&text).expect("valid JSON");
    let points = profile.get("pps").and_then(json::Value::as_array).unwrap();
    assert!(
        points.len() > 1,
        "the platform unwinder produced {} program point(s); a profiler that \
         cannot tell call sites apart is not profiling",
        points.len()
    );
    // Depth is asked of the untrimmed profile the probe writes beside this one,
    // because the frames a default profile *shows* are not the frames it
    // captured. The default rendering removes the allocation path and the
    // runtime entry, and in a release build — where every helper in the probe
    // is inlined into `main` — what legitimately survives is one frame per call
    // site, `lifecycle_probe::main+0x318`, distinguished by the offset alone
    // **[measured]**. Asserting on the shown depth made this test fail for a
    // profile that was entirely correct, which is what a proxy does when the
    // thing it stands in for moves.
    //
    // The obvious repair — captured depth is what is shown plus `trimmedFrames`
    // — is not one. Trimming can make two program points identical, and the
    // frames of the point that then folds into the other leave the file without
    // being counted anywhere, so the sum understates what was walked and the
    // assertion can get *harder* on a profile that is entirely correct. Reading
    // a file where nothing was removed avoids needing the identity at all, and
    // it holds on platforms that name no frames, where trimming does nothing.
    let untrimmed_path = path.with_extension("untrimmed.json");
    let untrimmed_text = std::fs::read_to_string(&untrimmed_path).unwrap_or_else(|error| {
        panic!(
            "no untrimmed profile at {}: {error}",
            untrimmed_path.display()
        )
    });
    support::dhat::assert_valid(&untrimmed_text);
    let untrimmed = json::parse(&untrimmed_text).expect("valid JSON");
    // The depth below is only worth anything if this file really is the
    // untrimmed rendering. In a debug build a trimmed profile is deep enough to
    // pass the assertion by accident; only a release build would notice.
    assert_eq!(
        untrimmed
            .get("heapscope")
            .and_then(|section| section.get("trimmedFrames"))
            .and_then(json::Value::as_u64),
        Some(0),
        "the companion profile was rendered with trimming, so its depth says \
         nothing about what the platform walked"
    );
    let untrimmed_points = untrimmed
        .get("pps")
        .and_then(json::Value::as_array)
        .expect("the untrimmed profile has program points");
    let deepest = untrimmed_points
        .iter()
        .filter_map(|point| point.get("fs").and_then(json::Value::as_array))
        .map(|frames| frames.len())
        .max()
        .unwrap_or(0);
    assert!(
        deepest >= 3,
        "the deepest untrimmed trace has {deepest} frames across {} program \
         points; a platform unwinder that walks no stack is not usable",
        untrimmed_points.len()
    );
}

// --- Locating the fixture -------------------------------------------------

/// Path to the compiled `lifecycle_probe` example.
///
/// Searched for rather than computed; `support::fixture` says why.
fn probe_binary() -> PathBuf {
    let path = support::fixture::example_binary(
        "lifecycle_probe",
        "plain `cargo test` builds examples, but `cargo test --all-targets` does\n\
         not -- it compiles them as test harnesses instead, which is a different\n\
         file under a different name. See tests/support/fixture.rs. Either way:\n\
         \x20   cargo build --example lifecycle_probe",
    );
    assert_fresh(&path);
    path
}

/// Refuses to run against a fixture built from older source than this checkout.
///
/// `cargo test --test lifecycle` rebuilds the *test* but not the examples, so a
/// change to the library leaves the probe behind and every test here silently
/// reports on the previous build. That is not a hypothetical: it made a
/// deliberately broken `fork` implementation pass this whole file, because the
/// binary under test predated the breakage.
///
/// The cheapest correct check is the one the build system itself uses. If any
/// source that goes into the probe is newer than the probe, the probe is stale.
///
/// "The one the build system itself uses" is meant literally, and it is why the
/// list below is `.rs` files and nothing else. Cargo decides a source file is
/// stale by its modification time, so comparing modification times agrees with
/// Cargo exactly. Cargo decides nothing that way about `Cargo.toml` or
/// `Cargo.lock`: it fingerprints the build-relevant content of one and the
/// resolution recorded in the other. Both used to be in this list, and both were
/// wrong in the same way — an edit Cargo provably ignores (a comment, a
/// `[[bin]]` section, a `cargo update` of a dependency this fixture does not
/// link) moves the file's clock without moving anything Cargo will rebuild.
///
/// That is not merely a stricter check. It is an unsatisfiable one, which is
/// worse than none: the guard fails, prints the remedy below, and Cargo answers
/// the remedy with `Finished` and leaves the probe's clock exactly where it was,
/// because nothing it tracks changed. There is no command that turns the suite
/// green again. Measured, not reasoned: `touch Cargo.lock` followed by
/// `cargo build --example lifecycle_probe` reproduces it in two commands.
///
/// What that gives up is a manifest or resolution change that really does alter
/// the probe and touches no `.rs` file. For this fixture there is no such
/// change: the probe links `heapscope`, `heapscope` has no dependencies, and a
/// manifest edit that alters the build (a `[profile.dev]` key, an edition bump)
/// makes Cargo rebuild the probe the next time anything builds examples — which
/// is what the remedy below says to do, and what CI now does before it tests.
#[track_caller]
fn assert_fresh(probe: &Path) {
    let built = modified(probe).expect("the probe has a modification time");

    let mut newest_source = std::time::SystemTime::UNIX_EPOCH;
    let mut newest_path = PathBuf::new();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for source in sources(&manifest.join("src"))
        .into_iter()
        .chain([manifest.join("examples/lifecycle_probe.rs")])
    {
        if let Some(time) = modified(&source) {
            if time > newest_source {
                newest_source = time;
                newest_path = source;
            }
        }
    }

    assert!(
        built >= newest_source,
        "the lifecycle probe at {} is older than {}, so these tests would report \
         on the previous build of the library.\n\
         Run `cargo build --example lifecycle_probe`, or plain `cargo test`, \
         which builds examples -- `cargo test --all-targets` does not.",
        probe.display(),
        newest_path.display(),
    );
}

fn modified(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

/// Every `.rs` file under `directory`, recursively.
fn sources(directory: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(directory) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(sources(&path));
        } else if path.extension().is_some_and(|kind| kind == "rs") {
            found.push(path);
        }
    }
    found
}

/// A fresh directory for one run of the probe.
///
/// Under the repository rather than the system temporary directory, and removed
/// and recreated on each run so that a stale profile from a previous run can
/// never be mistaken for one this run produced — which would turn every "no
/// profile is written" test into a false pass.
///
/// The counter is not decoration. Keying only on the mode gave two tests that
/// run the same mode the same directory, and since each wipes it on entry, the
/// one that got there second deleted the other's profile mid-test. That was an
/// intermittent failure in the suite for as long as it took to write this
/// sentence, and the fix is for no two runs to share a directory at all.
fn temporary_directory(mode: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static RUN: AtomicUsize = AtomicUsize::new(0);

    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tmp")
        .join("lifecycle")
        .join(format!("{mode}-{}", RUN.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).expect("could not create the output directory");
    directory
}
