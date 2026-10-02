//! `Profiler::reset` through a real allocator, from the warm-up to the files.
//!
//! The engine's unit tests restart an engine they drive by hand, and the
//! differential model compares every counter after a restart against an eager
//! one. Neither goes through the `#[global_allocator]`, the public method, the
//! assertions a test writes after a reset, or the profiles a restarted run
//! saves. This does.
//!
//! # Why one `#[test]`
//!
//! There is one engine per process and `cargo test` runs tests concurrently: a
//! second test allocating during the window would be counted in it, and a
//! second test starting a profiler would be refused. The same arrangement
//! `tests/testing_api.rs` explains, for the same reason.

mod support;

use std::hint::black_box;
use std::panic::{self, AssertUnwindSafe};

use heapscope::{HeapStats, ResetError};
use support::{dhat, native};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

const MIB: usize = 1 << 20;

/// What a warm-up looks like: a cache that stays, and a transient peak several
/// times its size that the steady state never comes near again.
#[inline(never)]
fn warm_up() -> Vec<u8> {
    let scratch = black_box(vec![0x5Au8; 4 * MIB]);
    let cache = black_box(vec![0xC5u8; MIB]);
    drop(scratch);
    cache
}

/// The part worth measuring: small, balanced, and repeated.
#[inline(never)]
fn steady_state(rounds: usize) {
    for round in 0..rounds {
        let request = black_box(vec![round as u8; 4096]);
        black_box(&request);
    }
}

/// Runs `body`, and returns the panic message if it panicked.
///
/// The hook is replaced because these panics are the subject rather than a
/// failure, and a green run should not print them.
fn failure_message(body: impl FnOnce()) -> String {
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let outcome = panic::catch_unwind(AssertUnwindSafe(body));
    panic::set_hook(previous);

    let payload = outcome.expect_err("the assertion passed where it had to fail");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic payload that is a message")
}

#[test]
#[cfg_attr(
    miri,
    ignore = "needs a real backtrace, and Miri cannot execute inline assembly"
)]
fn a_reset_leaves_the_warm_up_out_of_everything_a_run_reports() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    // A failing assertion below is deliberate, and its profile is not wanted.
    // Safe as a process-wide write only because this binary runs one test.
    std::env::set_var(heapscope::stats::DUMP_VARIABLE, "off");

    let profiler = heapscope::Profiler::builder()
        .no_output()
        .build()
        .expect("the profiler should start");

    // ---- the warm-up, counted like anything else ----
    let cache = warm_up();
    let warm = HeapStats::get().expect("a running heap run has counters");
    assert_eq!(warm.resets, 0);
    assert!(
        warm.max_bytes >= (5 * MIB) as u64,
        "the warm-up's transient peak was not counted"
    );

    // ---- the restart ----
    profiler
        .reset()
        .expect("a running profiler restarts its counts");
    let restarted = HeapStats::get().expect("a running heap run has counters");
    assert_eq!(restarted.resets, 1);
    assert!(
        restarted.total_bytes < MIB as u64,
        "the warm-up's {} bytes are still in the totals",
        restarted.total_bytes
    );
    assert!(
        restarted.curr_bytes >= MIB as u64,
        "the cache is live and is no longer counted as live"
    );
    assert!(
        restarted.max_bytes >= restarted.curr_bytes && restarted.max_bytes < (4 * MIB) as u64,
        "the peak should start again from what is live, not keep the warm-up's: {}",
        restarted.max_bytes
    );

    // ---- the window ----
    steady_state(64);
    let window = HeapStats::get().expect("a running heap run has counters");
    assert!(
        window.total_blocks >= restarted.total_blocks + 64,
        "the steady state's allocations are not in the window"
    );
    // The assertion a budget test writes after a reset needs no mark: the peak
    // it measures is the window's. Before the reset this would have failed by
    // the four mebibytes the warm-up's scratch buffer held.
    heapscope::assert_max_bytes!(window.max_bytes);
    heapscope::assert_max_bytes!(restarted.curr_bytes + (MIB as u64));

    // ---- a mark from before the reset ----
    // Its totals are another window's, and the count of resets says so to
    // code that subtracts them.
    assert_ne!(window.resets, warm.resets);
    // Its live figures carry across the reset unchanged, so a leak check from
    // it asks whether the warm-up and the work together left anything behind:
    // nothing beyond the cache, which was live at the mark too.
    heapscope::assert_no_leaks!(since: warm);
    let leaked = black_box(vec![0u8; 64]);
    let message = failure_message(|| heapscope::assert_no_leaks!(since: warm));
    assert!(
        message.contains("more blocks are live than at the mark"),
        "a block allocated after a pre-reset mark went unseen: {message}"
    );
    drop(leaked);
    let mark = HeapStats::get().expect("a running heap run has counters");
    steady_state(8);
    heapscope::assert_no_leaks!(since: mark);

    // ---- a call from inside the profiler is refused, and changes nothing ----
    {
        let _inside = heapscope::internals::guard::enter()
            .expect("the test thread is not inside the profiler");
        assert_eq!(profiler.reset(), Err(ResetError::CannotEnter));
    }
    assert_eq!(HeapStats::get().unwrap().resets, 1);

    // ---- every profile says so ----
    let snapshot = profiler.snapshot();
    let declared = snapshot.reset.expect("the snapshot declares the restart");
    assert_eq!(declared.count, 1);
    assert!(
        declared.carried_bytes >= MIB as u64,
        "the cache was live at the restart: {declared:?}"
    );

    let native_path = directory.path().join("profile.json");
    snapshot
        .save_native(&native_path)
        .expect("the native profile");
    let native_text = std::fs::read_to_string(&native_path).expect("readable");
    native::assert_valid(&native_text);
    assert!(native_text.contains(r#""formatVersion":2"#));
    assert!(native_text.contains(r#""reset":{"count":1,"#));

    let dhat_path = directory.path().join("dhat.json");
    snapshot.save_dhat_v2(&dhat_path).expect("the DHAT profile");
    let dhat_text = std::fs::read_to_string(&dhat_path).expect("readable");
    dhat::assert_valid(&dhat_text);
    assert!(dhat_text.contains("counts restarted by Profiler::reset"));

    let html_path = directory.path().join("profile.html");
    snapshot.save_html(&html_path).expect("the HTML profile");
    assert!(std::fs::metadata(&html_path).expect("written").len() > 0);

    let mut summary = Vec::new();
    snapshot
        .write_text_summary(&mut summary, 5)
        .expect("writing to a Vec cannot fail");
    let summary = String::from_utf8(summary).expect("UTF-8");
    assert!(summary.contains("  restarted  at "), "{summary}");
    assert!(summary.contains("  carried    "), "{summary}");

    // ---- a stopped run is not restarted ----
    heapscope::engine().stop(heapscope::output::Shutdown::Explicit);
    assert_eq!(profiler.reset(), Err(ResetError::Stopped));

    drop(cache);
    drop(profiler);
}
