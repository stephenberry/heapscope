//! Starting a heap profiler in a program that never installed the shim.
//!
//! **This file has no `#[global_allocator]`, and that omission is the fixture.**
//! Every other test binary here installs `heapscope::Alloc`, which is why none
//! of them could ever have caught this: the defect is only reachable from a
//! program that forgot, and a test suite full of programs that remembered says
//! nothing about it.
//!
//! What the run produced before this was refused: `build()` returned `Ok`, the
//! engine recorded nothing, every figure was zero, and
//! `assert_max_bytes!(64 * 1024)` passed after allocating 10 MiB. That is the
//! "assertion that cannot fail" shape the `stats` module documentation names as
//! the thing its refusals exist to prevent, reached by the one route its table
//! did not list.

use heapscope::{Mode, Profiler, StartError};

/// The refusal itself.
///
/// This never claims the engine — the check runs before the claim — so it does
/// not race the ad hoc test below, whichever order they run in.
///
/// Ignored under Miri, which cannot execute the inline assembly a capture
/// needs: the unwinder probe runs first and fails, so `build()` returns
/// `NoBacktraces` and this would be asserting about the wrong refusal.
#[test]
#[cfg_attr(miri, ignore = "starting a profiler captures a real backtrace")]
fn a_heap_run_is_refused_when_the_shim_is_not_installed() {
    let error = Profiler::builder()
        .no_output()
        .build()
        .expect_err("a heap run with no shim installed must not start");

    assert_eq!(error, StartError::NotInstalled);
}

/// A refusal that does not say what to do about it is a worse error than none,
/// because the remedy is one line the reader has no way to guess.
#[test]
fn the_refusal_names_the_remedy() {
    let text = StartError::NotInstalled.to_string();

    assert!(
        text.contains("#[global_allocator]"),
        "the refusal must name the attribute, got: {text}"
    );
    assert!(
        text.contains("heapscope::Alloc::system()"),
        "the refusal must name the constructor, got: {text}"
    );
    assert!(
        text.contains("zero"),
        "the refusal must say what the run would otherwise have reported, \
         got: {text}"
    );
}

/// The other half of the check, and the one that would make it a defect of its
/// own if it were wrong.
///
/// `AdHoc` and `Copy` turn the allocator shim off and count what the program
/// reports instead, so for them an uninstalled shim is the configuration rather
/// than a mistake. A check that refused these would break every ad hoc run in
/// a program that does not profile the heap at all.
#[test]
#[cfg_attr(miri, ignore = "starting a profiler captures a real backtrace")]
fn an_ad_hoc_run_is_not_refused() {
    let profiler = Profiler::builder()
        .mode(Mode::AdHoc)
        .no_output()
        .build()
        .expect("an ad hoc run does not need the shim");

    heapscope::event(7);
    drop(profiler);
}
