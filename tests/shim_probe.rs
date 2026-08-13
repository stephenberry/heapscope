//! The startup probe, in a program that *did* install the shim.
//!
//! `tests/not_installed.rs` covers the refusal. This covers the two ways the
//! check could be a defect of its own, and both are silent failures rather than
//! loud ones, so nothing else would report them.
//!
//! # Why one `#[test]`
//!
//! One engine per process, and `cargo test` runs tests concurrently: a second
//! test allocating during the profiled window would be counted into these
//! totals. The same arrangement, and the same reason, as `tests/testing_api.rs`.

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

use std::hint::black_box;

use heapscope::{HeapStats, Profiler};

/// Two properties, because they need the same fixture and the fixture is the
/// whole process.
///
/// **The probe is not recorded.** It is an allocation the profiler makes on its
/// own behalf, and the engine is deliberately still idle when it happens. If it
/// were made a moment later, while the engine was running, it would land in
/// every profile as a program point belonging to `Profiler::start` and add one
/// to the `total_blocks` that `assert_alloc_count!` compares against — turning
/// a correct `assert_alloc_count!(3)` into a failure in every program that used
/// it. The first assertion below is what says the ordering still holds.
///
/// **The check does not refuse a program that did install the shim.** That is
/// the failure direction that matters more than the defect being fixed, because
/// it would break working programs rather than misreport broken ones, and it is
/// what `build().expect(..)` below is for. Worth running in release as well as
/// debug: the probe is an allocation whose result is unused, which is the shape
/// LLVM is entitled to delete along with its free. It does not — see
/// `Profiler::shim_is_installed` for why the answer is the same either way —
/// but "does not today" is a thing to keep a release run pointed at.
///
/// [`StartError::NotInstalled`]: heapscope::StartError::NotInstalled
#[test]
#[cfg_attr(miri, ignore = "starting a profiler captures a real backtrace")]
fn the_startup_probe_is_invisible_to_the_profile_and_does_not_refuse_a_correct_program() {
    // Reaching `unwrap` proves the probe ran: it is the only thing that sets
    // the flag the check reads.
    let _profiler = Profiler::builder()
        .no_output()
        .build()
        .expect("the shim is installed in this binary, so the run must start");

    let at_start = HeapStats::get().expect("a running heap profiler has stats");
    assert_eq!(
        at_start.total_blocks, 0,
        "the startup probe was recorded into the profile it is checking for"
    );

    // Exactly three allocations, and nothing about a `Box<u8>` is subject to a
    // growth strategy or a platform difference.
    let boxes = [Box::new(1u8), Box::new(2u8), Box::new(3u8)];
    black_box(&boxes);

    let after = HeapStats::get().expect("a running heap profiler has stats");
    assert_eq!(
        after.total_blocks, 3,
        "the run recorded something other than the three blocks it made"
    );
}
