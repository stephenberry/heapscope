//! A fixture for `tests/lifecycle.rs`, not a demonstration of the API.
//!
//! PLAN.md section 4.6 fixes the behaviour of the profiler around a dozen ways a
//! process can end or fork. Most of them cannot be produced from inside a test
//! harness: `process::exit` takes the harness with it, `abort` leaves no result
//! to report, and `fork` in a process running a thread pool is not something to
//! do to a test runner. So the harness runs *this* program instead, once per
//! row, and inspects what it left behind.
//!
//! Every mode does the same work first, so the profiles are comparable, and
//! takes its output path from `argv[2]`. A mode that fails its own internal
//! check exits with a distinct code rather than a panic, because several modes
//! run in states where panicking is itself the thing under test.
//!
//! Usage: `lifecycle_probe <mode> <output-path>`

use std::hint::black_box;
use std::path::PathBuf;

use heapscope::Profiler;

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// Exit code for "the probe's own check failed", distinct from a panic's 101.
const CHECK_FAILED: i32 = 3;
/// Exit code for "the harness asked for a mode this program does not have".
const UNKNOWN_MODE: i32 = 4;

/// Frames per capture in the `configured` mode. Small enough that real stacks
/// are cut by it, so that a limit doing nothing would show up.
const CONFIGURED_DEPTH: usize = 3;

/// Live-block ceiling in the `configured` mode. Not round, so a profile
/// reporting the default instead would be obvious.
const CONFIGURED_LIVE_BLOCKS: usize = 5_000;

/// Live-block ceiling in the `full-table` mode: small enough that an ordinary
/// workload runs past it in the first few hundred allocations.
///
/// The table gives each of its 64 shards an equal share and rounds that share up
/// to a power of two, so this is both the request and the ceiling the profile
/// reports — and it is distinct from `CONFIGURED_LIVE_BLOCKS`' rounded 8,192 and
/// from the default, so a profile reporting a table other than this one is
/// visibly not this run.
const FULL_TABLE_LIVE_BLOCKS: usize = 128;

/// Blocks the `full-table` mode holds live at once. Two orders of magnitude past
/// the ceiling, so the table fills whatever order the shards happen to fill in.
const FULL_TABLE_BLOCKS: usize = 8_192;

/// Program points the `configured` mode's text summary is asked for. Small
/// enough that the run has more points than this, so a `top` nothing applied
/// would print more.
const CONFIGURED_TOP: usize = 3;

/// Weights the `ad-hoc` mode reports: one from each of two call depths. Distinct
/// and unround, so a total that came from anywhere but these two calls is
/// visibly not this.
const AD_HOC_WEIGHTS: [u64; 2] = [7, 7_000];

/// Byte counts the `copy` mode reports, chosen for the same reason.
const COPIED_BYTES: [usize; 2] = [1_111, 333_333];

/// Calls each non-heap mode makes to the *other* reporting function, which the
/// run must refuse and count.
const MISDIRECTED_CALLS: usize = 3;

/// Blocks allocated before the profiler starts, and freed while it is running.
///
/// Their frees find no entry in the live-block table. The row this covers says
/// they must be ignored: no underflow, no phantom negative, no second data
/// structure to remember them in.
const PRE_START_BLOCKS: usize = 64;

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_default();
    let output = PathBuf::from(args.next().unwrap_or_default());

    // Row: allocation *before* the profiler starts.
    let before_start: Vec<Vec<u8>> = (0..PRE_START_BLOCKS)
        .map(|size| vec![0u8; 1024 + size])
        .collect();

    // Settings are fixed for the life of a run, so every mode that varies one
    // has to say so here, before the profiler starts.
    // Both formats from one reading of the engine, everywhere a profile is
    // written at all. `native` costs one line here and puts every row of section
    // 4.6 — a fork child, a poisoned engine, a run whose live-block table filled
    // — through the native emitter as well, which no hand-built snapshot reaches.
    let mut builder = Profiler::builder()
        .output(heapscope::Output::dhat_v2(output.clone()))
        .also(heapscope::Output::native(
            output.with_extension("native.json"),
        ));
    if mode == "system-unwinder" {
        builder = builder.unwinder(heapscope::unwind::Strategy::System);
    }
    if mode == "configured" {
        builder = builder
            .max_depth(CONFIGURED_DEPTH)
            .max_live_blocks(CONFIGURED_LIVE_BLOCKS)
            .trim_frames(false)
            .also(heapscope::Output::text_summary_to_stderr(CONFIGURED_TOP));
    }
    if mode == "configured-system" {
        // The depth limit against the *platform* unwinder, which on unix spends
        // the caller's buffer on the frames it is about to discard. A limit at
        // or below that skip once emptied every capture.
        builder = builder
            .unwinder(heapscope::unwind::Strategy::System)
            .max_depth(CONFIGURED_DEPTH)
            // So that the depth the harness reads is the depth that was
            // captured, rather than what trimming left of it.
            .trim_frames(false);
    }
    if mode == "full-table" {
        builder = builder.max_live_blocks(FULL_TABLE_LIVE_BLOCKS);
    }
    if mode == "no-output" {
        builder = builder.no_output();
    }
    if mode == "ad-hoc" {
        builder = builder
            .mode(heapscope::Mode::AdHoc)
            .also(heapscope::Output::text_summary_to_stderr(CONFIGURED_TOP));
    }
    if mode == "copy" {
        builder = builder
            .mode(heapscope::Mode::Copy)
            .also(heapscope::Output::text_summary_to_stderr(CONFIGURED_TOP));
    }
    let started = builder.build();

    let profiler = match started {
        Ok(profiler) => profiler,
        Err(error) => {
            eprintln!("lifecycle_probe: could not start the profiler: {error}");
            std::process::exit(CHECK_FAILED);
        }
    };

    // Row: a second profiler while the first is running.
    let unwinder_before = heapscope::unwind::strategy();
    if !matches!(Profiler::new(), Err(heapscope::StartError::AlreadyRunning)) {
        eprintln!("lifecycle_probe: a second profiler was allowed to start");
        std::process::exit(CHECK_FAILED);
    }
    // ...and refusing it must change nothing. Asking whether a profiler is
    // already running is the documented way to find out, and the first version
    // of this answered by resetting the running profiler's unwinder to the
    // platform default.
    if heapscope::unwind::strategy() != unwinder_before {
        eprintln!(
            "lifecycle_probe: a refused profiler changed the unwinder from \
             {unwinder_before} to {}",
            heapscope::unwind::strategy()
        );
        std::process::exit(CHECK_FAILED);
    }

    let held = workload();

    // Freed while recording, allocated before it started. Explicit rather than
    // left to drop order, because drop order is exactly what this must not
    // depend on.
    drop(before_start);

    match mode.as_str() {
        // The ordinary ending: the profiler is dropped, then `main` returns.
        //
        // Also the only mode that reports events during a *heap* run. That is
        // the likeliest way to hold this feature wrong — instrumentation left
        // in a program profiled the ordinary way — and both calls must do
        // nothing but be counted, or a heap profile's `tb` would mix bytes with
        // dimensionless weights.
        "drop" => {
            for _ in 0..MISDIRECTED_CALLS {
                heapscope::event(1_000_000);
                heapscope::copied(1_000_000);
            }
            drop(profiler);
            drop(held);
        }

        // The profiler outlives `main`, as it would in a `static`. Only the
        // exit handler can produce a profile here.
        "forget" => {
            std::mem::forget(profiler);
            drop(held);
        }

        // Row: `std::process::exit`. No destructor runs, on any thread.
        "process-exit" => {
            std::mem::forget(held);
            std::process::exit(0);
        }

        // The same, from a thread that is not `main`. `main`'s stack — and the
        // profiler on it — is never unwound.
        "exit-from-thread" => {
            std::mem::forget(held);
            let joined = std::thread::spawn(|| std::process::exit(0)).join();
            eprintln!("lifecycle_probe: the exiting thread returned: {joined:?}");
            std::process::exit(CHECK_FAILED);
        }

        // Row: panic with unwinding. `Profiler::drop` runs as the stack unwinds.
        "panic" => {
            drop(held);
            panic!("lifecycle_probe: deliberate panic");
        }

        // Row: `abort`. No `atexit` handler runs, so there is no profile. The
        // point of the test is that this is *stated*, not that it works.
        "abort" => {
            std::mem::forget(held);
            std::process::abort();
        }

        // Row: a fatal signal. `SIGKILL` cannot be caught, blocked, or handled,
        // so nothing this crate could ever do would produce a profile here.
        #[cfg(unix)]
        "fatal-signal" => {
            std::mem::forget(held);
            // SAFETY: sending a signal to this process. It does not return.
            unsafe { kill(getpid(), 9) };
            eprintln!("lifecycle_probe: SIGKILL did not kill this process");
            std::process::exit(CHECK_FAILED);
        }

        // Row: allocation after the profiler has stopped. The shim becomes a
        // straight pass-through, so the numbers in the file cannot move after
        // it has been written.
        "alloc-after-stop" => {
            drop(held);
            drop(profiler);

            let before = heapscope::engine().stats();
            for size in 0..2048 {
                black_box(vec![5u8; 64 + size]);
            }
            let after = heapscope::engine().stats();
            if before.total_blocks != after.total_blocks || before.total_bytes != after.total_bytes
            {
                eprintln!(
                    "lifecycle_probe: {} blocks were recorded after the profiler stopped",
                    after.total_blocks - before.total_blocks
                );
                std::process::exit(CHECK_FAILED);
            }
        }

        // Row: concurrent shutdown. The profiler is dropped while other threads
        // are still inside the shim; the state flips first and the drain is
        // bounded, so the process must not hang and the profile must be
        // internally consistent.
        "concurrent-shutdown" => {
            concurrent_shutdown_mode(profiler, held);
        }

        // Row: an internal invariant violation. Recording stops, one line goes
        // to stderr, the profile says so, and the program carries on.
        "poison" => {
            heapscope::internals::diagnostic::poison("lifecycle_probe: deliberate poison");
            for size in 0..512 {
                black_box(vec![6u8; 64 + size]);
            }
            drop(held);
            drop(profiler);
        }

        // Row: table capacity exhausted. The live-block table is the one table a
        // program can fill through the public API, and what it must not do when
        // full is half-count: a block it cannot track is left out of the totals
        // entirely and counted apart, so that every column still describes the
        // same set of blocks. `droppedBlocks` is how the profile says how many
        // were left out.
        "full-table" => {
            let mut kept: Vec<Vec<u8>> = Vec::with_capacity(FULL_TABLE_BLOCKS);
            for size in 0..FULL_TABLE_BLOCKS {
                kept.push(vec![9u8; 32 + (size % 64)]);
            }

            let stats = heapscope::engine().stats();
            if stats.dropped_blocks == 0 {
                eprintln!(
                    "lifecycle_probe: {FULL_TABLE_BLOCKS} live blocks did not fill a table of \
                     {FULL_TABLE_LIVE_BLOCKS}, so this mode proves nothing"
                );
                std::process::exit(CHECK_FAILED);
            }

            // Freed while the table is still full. Every one of these frees that
            // belongs to a block the table never held has to be ignored — the
            // same path a pre-start block's free takes, reached here in bulk.
            drop(kept);
            drop(held);
            drop(profiler);
        }

        // Not a section 4.6 row: the opt-in platform unwinder, end to end. It
        // is the default on Windows and an escape hatch elsewhere, and either
        // way a profile it produced has to be a valid profile.
        //
        // This mode writes a second profile beside the first, rendered without
        // trimming. What the harness wants to know is how deep the stacks the
        // platform *walked* were, and the default rendering does not answer
        // that: it removes the allocation path and the runtime entry, and can
        // merge two program points that become identical once it has. A file
        // where nothing was removed answers it directly.
        "system-unwinder" => {
            write_untrimmed(&profiler, &output);
            drop(held);
            drop(profiler);
        }

        // Not a section 4.6 row: everything `ProfilerBuilder` sets, in a real
        // process. The settings are only meaningful against real stacks and a
        // real allocation load, so the harness reads them back out of the file
        // this writes rather than out of the builder that set them.
        "configured" => {
            drop(held);
            drop(profiler);
        }

        // The same settings against the platform unwinder. Separate from
        // `configured` because the two backends spend the caller's buffer
        // differently, and a limit that works on one said nothing about the
        // other.
        "configured-system" => {
            drop(held);
            drop(profiler);
        }

        // Not a section 4.6 row: a run that counts what the program reports
        // rather than what the shim sees. The whole `workload` above ran under
        // this profiler and must appear nowhere in the profile, which is the
        // part a heap-mode test cannot check.
        "ad-hoc" => {
            let site = report_event(AD_HOC_WEIGHTS[0]);
            report_event_deeper(AD_HOC_WEIGHTS[1]);
            println!("report-site {site:#x}");
            for _ in 0..MISDIRECTED_CALLS {
                heapscope::copied(1);
            }
            write_untrimmed(&profiler, &output);
            drop(held);
            drop(profiler);
        }

        // The same machinery, the other verb. Separate because the mode decides
        // which of the two functions records and which is refused, and a single
        // mode could not tell a working pair from one wired to itself.
        "copy" => {
            let site = report_copy(COPIED_BYTES[0]);
            report_copy_deeper(COPIED_BYTES[1]);
            println!("report-site {site:#x}");
            for _ in 0..MISDIRECTED_CALLS {
                heapscope::event(1);
            }
            write_untrimmed(&profiler, &output);
            drop(held);
            drop(profiler);
        }

        // Not a section 4.6 row: `no_output` has to disarm the exit handler as
        // well as the drop path. This mode forgets the profiler *and* exits
        // through the handler, which is the only arrangement where an armed
        // handler shows itself — by writing `dhat-heap.json` into the working
        // directory, which is what this whole mode exists to catch.
        "no-output" => {
            std::mem::forget(held);
            std::mem::forget(profiler);
            std::process::exit(0);
        }

        // Row: `_exit`. Same, by the other route.
        "raw-exit" => {
            std::mem::forget(held);
            // SAFETY: `_exit` takes an integer status and does not return.
            unsafe { raw_exit(0) }
        }

        // Not a section 4.6 row on its own: the *remedy* for the three rows
        // that produce nothing. The README tells a program facing `_exit`,
        // `abort`, or a Windows `process::exit` to write the profile itself
        // first, so these two modes do exactly that and then take the exit that
        // bypasses everything.
        //
        // All three formats, because the advice names all three and because
        // `_exit` is the sharpest available test of whether a `save_*` call has
        // really finished with the file when it returns. Nothing flushes after
        // this line: no destructor, no `atexit`, no stdio teardown. A writer
        // holding bytes in a buffer, or one that had not yet renamed its
        // temporary file into place, leaves a missing or truncated profile that
        // no later stage will repair.
        "save-then-raw-exit" | "save-then-abort" => {
            save_everything(&profiler, &output);
            std::mem::forget(held);
            std::mem::forget(profiler);
            if mode == "save-then-abort" {
                std::process::abort();
            }
            // SAFETY: `_exit` takes an integer status and does not return.
            unsafe { raw_exit(0) }
        }

        // Row: `fork` from a process whose other threads are inside the shim.
        #[cfg(unix)]
        "fork" => {
            fork_mode(&profiler, &output, held);
            drop(profiler);
        }

        // The same, but the child drops the inherited profiler rather than
        // exiting. `Profiler::drop` in a child must not write the parent's
        // numbers to the parent's file.
        #[cfg(unix)]
        "fork-child-drop" => {
            drop(held);
            fork_child_drop_mode(profiler, &output);
        }

        other => {
            eprintln!("lifecycle_probe: unknown mode {other:?}");
            std::process::exit(UNKNOWN_MODE);
        }
    }
}

/// Reports one ad hoc event, and returns the address it must be attributed to.
///
/// `#[inline(never)]` so that the address of this function is the address of the
/// code that made the call. That is the same reason the shim's methods are, and
/// the property being pinned is the same one: the calibrated skip has to leave
/// the *caller* of `heapscope::event` as the innermost frame, not
/// `heapscope::event` itself.
#[inline(never)]
fn report_event(weight: u64) -> usize {
    heapscope::event(weight);
    (report_event as fn(u64) -> usize) as usize
}

/// The same call, one frame further out.
///
/// The extra frame is the only difference between the two captures, which is
/// what lets [`check_only_these_events`] check the skip without guessing how
/// long a function is. See `unwind::calibrate`, which uses the same argument for
/// the same reason.
#[inline(never)]
fn report_event_deeper(weight: u64) {
    black_box(report_event(weight));
}

/// The copy-mode counterparts, spelled out rather than shared through a function
/// pointer: a `fn(u64)` built from a closure is a real function, and calling
/// through it would insert exactly the extra frame this is measuring.
#[inline(never)]
fn report_copy(bytes: usize) -> usize {
    heapscope::copied(bytes);
    (report_copy as fn(usize) -> usize) as usize
}

#[inline(never)]
fn report_copy_deeper(bytes: usize) {
    black_box(report_copy(bytes));
}

/// Writes a second profile beside the first, rendered without trimming.
///
/// Used by every mode whose harness needs to see the frames as *captured*. The
/// default rendering does not answer that: it removes the allocation path and
/// the runtime entry, and can merge two program points that become identical
/// once it has. A file where nothing was removed answers it directly.
///
/// This exists because the checks it enables used to live *inside* this program,
/// reading `profiler.snapshot()` — and a check that only the thing being checked
/// performs is not covered by anything. Deleting it outright left the whole
/// suite green. Here the harness does the checking and the probe only produces
/// evidence.
fn write_untrimmed(profiler: &Profiler, output: &std::path::Path) {
    let snapshot = profiler.snapshot();
    let untrimmed = heapscope::symbol::Symbolized::new(&snapshot.modules);
    let path = output.with_extension("untrimmed.json");
    let written =
        std::fs::File::create(&path).and_then(|file| snapshot.write_dhat_v2_with(file, &untrimmed));
    if let Err(error) = written {
        eprintln!(
            "lifecycle_probe: could not write {}: {error}",
            path.display()
        );
        std::process::exit(CHECK_FAILED);
    }
}

/// Writes all three formats by hand, as a program facing an exit that bypasses
/// the handler list has to.
///
/// A failure here exits the probe rather than being ignored. These modes exist
/// to show that this call is the whole difference between a profile and none,
/// so a silent failure would leave the harness unable to tell "the remedy does
/// not work" from "the remedy was never attempted".
fn save_everything(profiler: &Profiler, output: &std::path::Path) {
    let attempts = [
        ("DHAT", profiler.save_dhat_v2(output)),
        (
            "native",
            profiler.save_native(output.with_extension("native.json")),
        ),
        ("page", profiler.save_html(output.with_extension("html"))),
    ];
    for (what, attempt) in attempts {
        if let Err(error) = attempt {
            eprintln!("lifecycle_probe: could not save the {what} profile: {error}");
            std::process::exit(CHECK_FAILED);
        }
    }
}

/// The work every mode records, chosen to exercise the rows that are about
/// *what* gets counted rather than about how the process ends.
fn workload() -> Vec<Vec<u8>> {
    // Freed while recording: counted in the totals, not in `eb`/`ebk`.
    for size in 0..256 {
        black_box(vec![7u8; 64 + size]);
    }

    // Through `alloc_zeroed` rather than `alloc`: `vec![0u8; n]` reaches
    // `RawVec::with_capacity_zeroed`, which is the one shim entry point nothing
    // else here exercises. It is a separate mode gate, and a review found it
    // unreachable: a non-heap run recorded nothing in the whole suite because no
    // profiled workload ever asked for zeroed memory.
    for size in 0..64 {
        black_box(vec![0u8; 128 + size]);
    }

    // Row: `realloc`. A `Vec` that grows past its capacity reallocates, and the
    // resize must be attributed to the point that first allocated it. The
    // one-at-a-time push is the whole point here — `vec![1; 4096]`, which is
    // what clippy would rather see, performs a single allocation and no
    // reallocation at all.
    #[allow(clippy::same_item_push)]
    let grown = {
        let mut grown: Vec<u8> = Vec::new();
        for _ in 0..4096 {
            grown.push(1);
        }
        grown
    };

    // A reallocation the allocator cannot possibly satisfy in place, so that the
    // profile has a *moved* one to count and the bytes it copied are not zero.
    //
    // The doubling above is not enough on its own: in a debug build every one of
    // its 4,096 pushes was satisfied in place, and the probe reported
    // `"moved": 0, "bytesCopied": 0` — which is also exactly what a shim that
    // passed the wrong address for `old_address` would report, from correct
    // code. Two live blocks pinned either side of one that then grows by three
    // orders of magnitude leaves the allocator nowhere to put it.
    let moved = {
        let pin_before = black_box(vec![3u8; 64]);
        let mut small: Vec<u8> = Vec::with_capacity(64);
        small.resize(64, 5);
        let pin_after = black_box(vec![4u8; 64]);
        small.reserve_exact(1 << 20);
        black_box((pin_before, pin_after));
        small
    };
    black_box(&moved);

    // Row: a named phase, with a nested one inside it. The profile must show
    // both, must not fold the inner one into the outer, and must bring the
    // region back down when the blocks allocated inside it are freed -- which
    // happens here after the guards have gone.
    let phased = {
        let _outer = heapscope::region("parsing");
        let mut kept = Vec::new();
        for size in 0..48 {
            kept.push(vec![9u8; 512 + size]);
        }
        {
            let _inner = heapscope::region("parsing/lexing");
            for size in 0..16 {
                black_box(vec![8u8; 2048 + size]);
            }
        }
        kept
    };
    drop(phased);

    // Row: a second thread, named, allocating blocks that outlive it. The
    // profile has to name the thread and keep its live bytes attributed to it
    // after it has exited -- both of which need the name captured at record
    // time, because by the time the file is written the thread is gone.
    let from_worker = std::thread::Builder::new()
        .name(String::from("hs-worker"))
        .spawn(|| {
            let _region = heapscope::region("worker");
            let mut made = Vec::new();
            for size in 0..24 {
                made.push(vec![6u8; 1024 + size]);
            }
            made
        })
        .expect("spawning the worker thread")
        .join()
        .expect("the worker thread panicked");

    // Row: still live when the profiler stops. These are the `eb`/`ebk` blocks.
    let mut held: Vec<Vec<u8>> = Vec::with_capacity(32);
    for size in 0..32 {
        held.push(vec![3u8; 4096 + size]);
    }
    held.push(grown);
    held.extend(from_worker);
    held
}

/// Drops the profiler while other threads are still allocating.
///
/// The shutdown path flips the state to `Finished` *before* it waits, so the
/// threads still in flight stop being recorded immediately and the wait is for
/// events already inside the gate. The wait is bounded, which is what keeps a
/// process with a busy thread pool from hanging at exit instead of writing a
/// slightly incomplete profile.
fn concurrent_shutdown_mode(profiler: Profiler, held: Vec<Vec<u8>>) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    const WORKERS: usize = 8;

    let stop = Arc::new(AtomicBool::new(false));
    let workers: Vec<_> = (0..WORKERS)
        .map(|worker| {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut size = 16 + worker;
                while !stop.load(Ordering::Relaxed) {
                    black_box(vec![0u8; size]);
                    size = 16 + (size * 13 + 7) % 8192;
                }
            })
        })
        .collect();

    // Let every worker get properly into the shim before the rug is pulled.
    std::thread::sleep(std::time::Duration::from_millis(50));
    drop(profiler);

    stop.store(true, Ordering::Relaxed);
    for worker in workers {
        let _ = worker.join();
    }
    drop(held);
}

/// Forks repeatedly while another thread is holding the profiler's locks.
///
/// # Why this does not just fork under load
///
/// It did, first, and the test was worthless: with the `fork` handlers removed
/// it still passed. Allocation pressure alone leaves a lock held for a few tens
/// of nanoseconds at a time, spread across sixty-four shards, so a child almost
/// never touches the one shard that was busy. A test whose failure depends on
/// winning that race reports "safe" for a profiler that is not.
///
/// So one thread holds a lock the child is *guaranteed* to need. Every recorded
/// allocation acquires the peak gate, and `flush_and_visit` holds that gate
/// across its visitor — so a visitor that spins holds the one lock the child
/// must pass through. Without `pthread_atfork` the child inherits it held by a
/// thread that does not exist; with the handlers, `prepare` waits for the holder
/// to finish before the fork happens at all.
///
/// The holder alternates holding and releasing, so `prepare` always has a gap to
/// acquire in and the protected build makes progress.
///
/// # Why the child takes a snapshot rather than just allocating
///
/// Because allocating in a child proves nothing. The child is `ForkedChild`, so
/// `alloc.rs` skips the engine entirely and never touches a lock — which is why
/// deleting the entire bodies of `fork_prepare` and `fork_parent` left every
/// test in this repository passing. `Snapshot::capture` is public API a child
/// may legitimately call, and it acquires the gate and every program-point
/// shard, so it is the thing that hangs when the locks were inherited held.
#[cfg(unix)]
fn fork_mode(profiler: &Profiler, output: &std::path::Path, held: Vec<Vec<u8>>) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    const FORKS: usize = 24;
    const WORKERS: usize = 4;
    /// Long enough that a fork reliably lands inside it, short enough that
    /// twenty-four of them do not dominate the test suite.
    const HOLD: Duration = Duration::from_micros(400);

    let stop = Arc::new(AtomicBool::new(false));

    let gate_holder = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let mut first = true;
                let flush = heapscope::engine().flush_and_visit(
                    Duration::from_secs(5),
                    |_, _, _| {
                        if std::mem::take(&mut first) {
                            // Spun rather than slept: `flush_and_visit` holds
                            // the gate across this, and parking a thread that
                            // every allocating thread is waiting on is a worse
                            // citizen than burning 400 microseconds.
                            let until = Instant::now() + HOLD;
                            while Instant::now() < until {
                                std::hint::spin_loop();
                            }
                            GATE_HOLDS.fetch_add(1, Ordering::Relaxed);
                        }
                    },
                    |_| {},
                    |_| {},
                );
                if !flush.exclusive {
                    GATE_NOT_EXCLUSIVE.fetch_add(1, Ordering::Relaxed);
                }
                // The gap `prepare` acquires in.
                std::thread::sleep(HOLD);
            }
        })
    };

    let workers: Vec<_> = (0..WORKERS)
        .map(|worker| {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut size = 32 + worker;
                while !stop.load(Ordering::Relaxed) {
                    black_box(vec![0u8; size]);
                    size = 32 + (size * 7 + 1) % 4096;
                }
            })
        })
        .collect();

    for round in 0..FORKS {
        // SAFETY: `fork` has no preconditions. What the child may do afterwards
        // is the constrained part, and the child branch below does only what
        // this crate's `fork` handlers are supposed to make safe.
        let pid = unsafe { fork() };
        if pid == 0 {
            child_after_fork(round);
        }
        if pid < 0 {
            eprintln!("lifecycle_probe: fork failed");
            std::process::exit(CHECK_FAILED);
        }
        if let Err(problem) = wait_for(pid) {
            eprintln!("lifecycle_probe: child {pid} in round {round}: {problem}");
            std::process::exit(CHECK_FAILED);
        }
    }

    stop.store(true, Ordering::Relaxed);
    for worker in workers {
        let _ = worker.join();
    }
    let _ = gate_holder.join();
    drop(held);

    // The gate hold is the entire reason this test can fail, and it is possible
    // for it to stop happening silently: the visitor only runs for points that
    // recorded something, and `flush_and_visit` runs it with no gate held at all
    // if it times out. Either would quietly turn this back into the worthless
    // version that passed with the handlers removed.
    let holds = GATE_HOLDS.load(Ordering::Relaxed);
    let lost = GATE_NOT_EXCLUSIVE.load(Ordering::Relaxed);
    if holds == 0 {
        eprintln!("lifecycle_probe: the gate was never actually held, so the forks raced nothing");
        std::process::exit(CHECK_FAILED);
    }
    if lost > 0 {
        eprintln!("lifecycle_probe: {lost} of {holds} gate acquisitions were not exclusive");
        std::process::exit(CHECK_FAILED);
    }

    // No child may have produced a profile: the recording they inherited is
    // this process's, and the file is this process's to write.
    if output.exists() {
        eprintln!("lifecycle_probe: a forked child wrote the parent's profile");
        std::process::exit(CHECK_FAILED);
    }
    let _ = profiler;
}

/// Counts of gate acquisitions that actually held it, and of those that did not.
#[cfg(unix)]
static GATE_HOLDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(unix)]
static GATE_NOT_EXCLUSIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// What a forked child does before exiting: use the engine, then leave.
///
/// Taking a snapshot is the part that matters. It acquires the peak gate and
/// every program-point shard, so if this child inherited them held by a thread
/// `fork` did not copy, this call never returns and the parent's `wait_for`
/// reports the child as wedged.
#[cfg(unix)]
fn child_after_fork(round: usize) -> ! {
    use std::sync::atomic::Ordering;

    let snapshot = heapscope::Snapshot::capture();
    // Read something out of it, so the work cannot be optimised away, and check
    // the child did not somehow keep recording.
    black_box(snapshot.points.len());

    // `exact`, not merely "it returned". Every wait in this crate is bounded, so
    // a child that inherited the peak gate held by a thread `fork` did not copy
    // does not hang here — it spends the flush timeout, gives up, and reports an
    // inexact snapshot. Asserting liveness alone would pass for a child whose
    // locks were never reset.
    if !snapshot.exact {
        eprintln!(
            "lifecycle_probe: a forked child could not acquire the peak gate, \
             so it inherited it held by a thread that no longer exists"
        );
        // SAFETY: `_exit` takes a status and does not return.
        unsafe { raw_exit(CHECK_FAILED) }
    }

    if snapshot.shutdown != heapscope::output::Shutdown::ForkedChild {
        eprintln!(
            "lifecycle_probe: a forked child reports shutdown {:?}",
            snapshot.shutdown
        );
        // SAFETY: `_exit` takes a status and does not return.
        unsafe { raw_exit(CHECK_FAILED) }
    }

    // Allocating too, which must stay a pass-through.
    for size in 0..256 {
        black_box(vec![9u8; 128 + size]);
    }
    let _ = GATE_HOLDS.load(Ordering::Relaxed);

    // Half the children exit through the `atexit` list and half bypass it,
    // because the child must not write a profile either way.
    if round.is_multiple_of(2) {
        std::process::exit(0);
    }
    // SAFETY: `_exit` takes a status and does not return.
    unsafe { raw_exit(0) }
}

/// Forks once and has the child drop the inherited profiler.
#[cfg(unix)]
fn fork_child_drop_mode(profiler: Profiler, output: &std::path::Path) -> ! {
    // SAFETY: as in `fork_mode`.
    let pid = unsafe { fork() };
    if pid == 0 {
        // The child owns a `Profiler` value that refers to a recording it does
        // not own. Dropping it must write nothing.
        drop(profiler);
        // SAFETY: `_exit` takes a status and does not return.
        unsafe { raw_exit(if output.exists() { CHECK_FAILED } else { 0 }) }
    }
    if pid < 0 {
        eprintln!("lifecycle_probe: fork failed");
        std::process::exit(CHECK_FAILED);
    }
    if let Err(problem) = wait_for(pid) {
        eprintln!("lifecycle_probe: child {pid}: {problem}");
        std::process::exit(CHECK_FAILED);
    }

    // Now the parent's own drop writes the profile, as usual.
    drop(profiler);
    std::process::exit(0);
}

/// Waits for `pid`, giving up rather than hanging if the child wedged.
///
/// A deadlocked child is the failure this whole mode exists to detect, and a
/// test that detects it by never finishing is not a test.
#[cfg(unix)]
fn wait_for(pid: std::ffi::c_int) -> Result<(), String> {
    use std::time::{Duration, Instant};

    const WNOHANG: std::ffi::c_int = 1;
    let deadline = Instant::now() + Duration::from_secs(20);

    loop {
        let mut status: std::ffi::c_int = 0;
        // SAFETY: `status` is a live, correctly typed local for the duration of
        // the call.
        let reaped = unsafe { waitpid(pid, &mut status, WNOHANG) };
        if reaped == pid {
            // The low seven bits hold the terminating signal; a zero there and
            // a zero in the next eight bits is a clean exit with status 0.
            return if status == 0 {
                Ok(())
            } else {
                Err(format!("exited with raw status {status:#x}"))
            };
        }
        if reaped < 0 {
            return Err(String::from("waitpid failed"));
        }
        if Instant::now() >= deadline {
            // SAFETY: sending SIGKILL to a child of this process.
            unsafe { kill(pid, 9) };
            return Err(String::from(
                "did not exit within 20 seconds, which is what a child that \
                 inherited a held lock looks like",
            ));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[cfg(unix)]
extern "C" {
    fn fork() -> std::ffi::c_int;
    fn getpid() -> std::ffi::c_int;
    fn waitpid(
        pid: std::ffi::c_int,
        status: *mut std::ffi::c_int,
        options: std::ffi::c_int,
    ) -> std::ffi::c_int;
    fn kill(pid: std::ffi::c_int, signal: std::ffi::c_int) -> std::ffi::c_int;
}

extern "C" {
    /// `_exit` on unix, and in the Universal CRT on Windows. Terminates the
    /// process without running `atexit` handlers, flushing stdio, or unwinding.
    #[link_name = "_exit"]
    fn raw_exit(status: std::ffi::c_int) -> !;
}
