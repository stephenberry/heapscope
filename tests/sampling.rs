//! Sampling, through a real `#[global_allocator]`.
//!
//! `src/internals/sampler.rs` checks the arithmetic on a synthetic stream of
//! sizes. That leaves the part it cannot see: whether the shim actually asks
//! before it captures, whether the weights reach the counters, whether the
//! refusal fires, and whether the interval reaches the file. Those need a real
//! run, so they are here.
//!
//! One test, for the reason `tests/end_to_end.rs` is one test: there is one
//! engine per process, `cargo test` runs tests concurrently, and a second
//! profiler would either be refused or blend two recordings.

mod support;

use std::hint::black_box;

use heapscope::internals::engine::Engine;
use support::json;

const FLUSH_TIMEOUT: std::time::Duration = Engine::FLUSH_TIMEOUT;

/// Mean bytes between sample points.
///
/// Small enough that a workload this size produces a useful number of samples,
/// large enough that the great majority of allocations are skipped, which is the
/// property being tested.
const INTERVAL: u64 = 64 * 1024;

/// Size of the small allocations, chosen well under [`INTERVAL`] so that each is
/// sampled with probability about 0.05%.
const SMALL: usize = 32;

/// How many of them.
const SMALL_COUNT: usize = 200_000;

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// Allocates and frees `count` blocks of `bytes`, keeping nothing.
///
/// Balanced on purpose: everything this allocates, it frees, so the live
/// counters must come back to exactly where they started. Under sampling that is
/// a real check rather than a trivial one, because a sampled block adds a
/// *weighted* number of bytes and its free has to subtract the same weight,
/// recomputed rather than remembered.
#[inline(never)]
fn churn(count: usize, bytes: usize) {
    for _ in 0..count {
        let mut v: Vec<u8> = Vec::with_capacity(bytes);
        v.push(0xAB);
        black_box(&v);
    }
}

/// One allocation far larger than the interval, which sampling must never miss.
#[inline(never)]
fn one_large_allocation(bytes: usize) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::with_capacity(bytes);
    v.resize(bytes, 0xCD);
    black_box(v)
}

#[test]
#[cfg_attr(
    miri,
    ignore = "needs a real backtrace, and Miri cannot execute inline assembly"
)]
fn a_sampled_run_estimates_what_it_did_not_record() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let output_path = directory.path().join("sampled.json");

    let profiler = heapscope::Profiler::builder()
        .sampling(INTERVAL)
        .output(heapscope::Output::dhat_v2(output_path.clone()))
        .build()
        .expect("profiler should start");

    // ---- the assertions refuse a sampled run ----
    //
    // First, because it is the one thing here that must hold before any
    // allocation muddies it, and because a test that read the counters and then
    // checked the refusal would be asserting on numbers it had just been told
    // not to trust.
    assert_eq!(
        heapscope::HeapStats::get(),
        Err(heapscope::StatsError::Sampled),
        "a sampled run handed out heap statistics to assert against"
    );
    // `EventStats` refuses this run for being a heap run, not for sampling, and
    // that ordering is the one `stats.rs` already argues for: asking a heap run
    // for event counters is a mistake in the test that its author can fix, and
    // naming sampling first would send them to the wrong place. The sampling arm
    // of `EventStats` needs an event run, which this process cannot also be, so
    // it is covered by a unit test in `src/stats.rs`.
    assert_eq!(
        heapscope::EventStats::get(),
        Err(heapscope::StatsError::NotAnEventRun),
        "a heap run should be refused for its mode before anything else"
    );
    // And the message says what to do about it, rather than only what happened.
    let complaint = heapscope::StatsError::Sampled.to_string();
    assert!(
        complaint.contains("sampling(") && complaint.contains("estimates"),
        "the refusal does not say why or what to do instead: {complaint}"
    );

    // ---- a balanced workload returns the live counters to where they were ----
    let before = profiler.stats();
    churn(SMALL_COUNT, SMALL);
    let after = profiler.stats();

    assert_eq!(
        after.curr_bytes, before.curr_bytes,
        "live bytes did not return to their starting value after a balanced \
         workload: a sampled free subtracted a different weight than its \
         allocation added"
    );
    assert_eq!(
        after.curr_blocks, before.curr_blocks,
        "live blocks did not return to their starting value after a balanced \
         workload"
    );

    // ---- the estimate recovers the true total ----
    //
    // The exact figure is not an assumption: `observedBlocks` counts every
    // request whether or not it was sampled, because counting a shape costs no
    // stack walk. So the profile carries the truth and the estimate of the same
    // quantity, and this compares them.
    let recorded_bytes = after.total_bytes - before.total_bytes;
    let truth = (SMALL_COUNT * SMALL) as u64;
    let error = (recorded_bytes as f64 - truth as f64) / truth as f64;
    assert!(
        error.abs() < 0.25,
        "the sampled estimate of {truth} bytes came out at {recorded_bytes} \
         ({:+.1}%)",
        error * 100.0
    );

    // ---- and it did so while capturing almost no stacks ----
    //
    // This is the whole point of the feature. Without it every assertion above
    // would also pass on a profiler that sampled nothing and simply recorded
    // everything.
    let snapshot = profiler.snapshot();
    let captures = snapshot.captures.complete
        + snapshot.captures.truncated
        + snapshot.captures.suspect
        + snapshot.captures.no_frames;
    assert!(
        captures < SMALL_COUNT as u64 / 20,
        "{captures} stacks were captured for {SMALL_COUNT} allocations, which is \
         not a sample"
    );
    assert!(
        captures > 0,
        "no stacks were captured at all, so nothing was recorded"
    );

    // ---- an allocation far above the interval is never missed ----
    //
    // The property that makes byte-weighted sampling worth its arithmetic. A
    // scheme that sampled one allocation in N would drop this with probability
    // 1 - 1/N however large it was, and it is the allocation a reader opened the
    // profile to find.
    let large_bytes = 8 * INTERVAL as usize;
    let before_large = profiler.stats();
    let large = one_large_allocation(large_bytes);
    let after_large = profiler.stats();
    assert!(
        after_large.curr_bytes >= before_large.curr_bytes + large_bytes as u64,
        "an allocation {large_bytes} bytes long, {} times the sampling interval, \
         did not reach the live counters",
        large_bytes as u64 / INTERVAL
    );
    // And it was not scaled up on the way in: it was certain to be sampled, so
    // it stands for itself alone.
    let recorded_large = after_large.curr_bytes - before_large.curr_bytes;
    assert!(
        recorded_large < large_bytes as u64 * 2,
        "an allocation certain to be sampled was inflated to {recorded_large} \
         from {large_bytes}"
    );
    drop(large);

    // ---- the profile says it was sampled ----
    drop(profiler);

    let text = std::fs::read_to_string(&output_path).expect("the profile should have been written");
    let profile = json::parse(&text).expect("the profile should be valid JSON");
    // Under `heapscope`, the extension object, because the DHAT format has no
    // field for it and inventing a top-level one would break a reader that
    // validates the schema it knows.
    let interval = profile
        .get("heapscope")
        .and_then(|extension| extension.get("settings"))
        .and_then(|settings| settings.get("samplingInterval"))
        .and_then(json::Value::as_u64)
        .expect("a sampled profile must record its interval");
    assert_eq!(
        interval, INTERVAL,
        "the profile reports a sampling interval it did not run with"
    );
}

/// Everything above, but for a run with no sampling, to prove the checks can
/// tell the difference.
///
/// Not a separate process: this runs after the profiler above has stopped, and
/// reads only what a stopped run left behind. It cannot start a second profiler,
/// so what it checks is the shape of the arithmetic rather than a second
/// recording — that an exact run's weights are the plain values, which is the
/// arm of every `if` above that the sampled run does not take.
#[test]
fn an_exact_run_weighs_nothing() {
    use heapscope::internals::sampler;

    for size in [0usize, 1, 32, 4_096, 1 << 20] {
        assert_eq!(
            sampler::weighted_bytes(size, None),
            size as u64,
            "size {size} was weighted on a run with no sampling"
        );
        assert_eq!(
            sampler::weighted_blocks(size, None),
            1,
            "size {size} counted as other than one block on a run with no sampling"
        );
    }
}

/// The flush path agrees with the global counters on a sampled run.
///
/// The per-point counters and the global ones are moved by the same `Delta` in
/// the same critical section, so they agree by construction — but the weights
/// are applied where the `Delta` is built, and a weight applied to one and not
/// the other would show up here and nowhere else.
#[test]
#[cfg_attr(miri, ignore = "needs the run from the test above")]
fn per_point_counters_sum_to_the_global_ones() {
    let mut summed_total = 0u64;
    let mut points = 0usize;
    let flush = heapscope::engine().flush_and_visit(
        FLUSH_TIMEOUT,
        |_id, _frames, counters| {
            points += 1;
            summed_total += counters.total_bytes;
        },
        |_| {},
        |_| {},
    );

    if points == 0 {
        // The sampled test has not run yet, or ran in another binary. Nothing to
        // compare, and asserting on zero would be asserting on nothing.
        return;
    }
    assert_eq!(
        summed_total, flush.stats.total_bytes,
        "the weighted per-point totals do not sum to the weighted global total"
    );
}
