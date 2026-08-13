//! Poisson sampling over the stream of allocated bytes.
//!
//! Recording every allocation costs a stack capture, and the capture is most of
//! what a profiler costs: `benches/overhead.rs` puts a recorded allocation at
//! about 129 ns against 32 ns unprofiled, and `benches/unwind.rs` puts the walk
//! at most of the difference. Sampling buys that back by capturing a stack for
//! some allocations and inferring the rest, which measures 51 ns on the same
//! workload.
//!
//! What it does not buy back is the rest of the per-allocation path: entering
//! the guard, counting the request in the size histograms, and advancing the
//! countdown below. That floors the overhead at about 18 ns per allocation
//! however large the interval gets, and `benches/overhead.rs` has the table that
//! shows it flat from 128 KiB to 16 MiB while the accuracy falls away.
//!
//! # What is sampled is bytes, not allocations
//!
//! One in every `N` allocations is the wrong rule. A program that allocates a
//! million 16-byte nodes and one 100 MiB buffer would report the buffer as a
//! rounding error or miss it entirely, and the buffer is the answer. So the
//! sample points fall on a Poisson process over the *byte* stream with mean
//! spacing `R`: an allocation of `s` bytes is sampled when at least one point
//! lands inside it, which happens with probability
//!
//! ```text
//! p(s) = 1 - exp(-s / R)
//! ```
//!
//! A 16-byte node at `R = 1 MiB` is sampled about once in 65,000 times; a
//! 100 MiB buffer is sampled with probability indistinguishable from one. Big
//! allocations are never missed, and small ones cost nothing to ignore.
//!
//! # Every sampled event stands for several
//!
//! A sampled allocation is scaled up by [`scale`], which is `1 / p(s)`,
//! **computed from its own size**. PLAN.md section 6.3 records the revision that
//! got this wrong: a single global multiplier of `R / s̄` inflates every
//! allocation larger than the mean by exactly the factor by which it was already
//! certain to be sampled. An allocation with `s >> R` has `p ≈ 1` and must be
//! scaled by 1, not by anything.
//!
//! The same scale applies to bytes, to block counts, and to the lifetime sums
//! behind `tl` and the short-lived counts. A sampled block that lived 40 µs is
//! evidence of `scale` blocks that lived about that long, not of one.
//!
//! # The state is per thread, and it is not in thread-local storage
//!
//! The countdown has to be per thread or it is one more contended atomic on the
//! path this exists to make cheaper. It cannot be a `thread_local!`, for the
//! reason [`guard`](super::guard) exists: a thread-local's first touch can
//! allocate, and this code runs inside the allocator. So the two words live in
//! the guard slot, which is already written on the way in, and this module is
//! the arithmetic over them.
//!
//! # What is reproducible and what is not
//!
//! [`TimeSource::Events`](crate::TimeSource) exists so that two runs of one
//! workload produce the same profile. Sampling can only preserve that if the
//! draws are the same, so the generator is seeded from a counter that starts at
//! zero in every process rather than from an address or a clock. A
//! single-threaded program therefore samples identically on every run.
//!
//! A multi-threaded one does not, and no seeding could make it: the threads race
//! for the allocator, so the byte stream each one sees differs between runs
//! before sampling looks at it. What is guaranteed is that the *n*th thread to
//! sample draws the *n*th seed, not that a particular thread is the *n*th.

use std::sync::atomic::{AtomicU64, Ordering};

use super::table::mix;

/// Seeds handed out to threads, in claim order.
///
/// A process-wide counter rather than a per-thread address so that a
/// single-threaded run draws the same sequence every time. See the module
/// documentation for the limit of that promise.
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Fixed odd constant, so that sequence 0 does not seed the generator with zero.
///
/// SplitMix64 maps zero to zero at its first step, which would give the first
/// thread a degenerate stream.
const SEED: u64 = 0x2545_F491_4F6C_DD1D;

/// Resets the seed sequence, so a fresh run in the same process starts over.
///
/// A process runs at most one profiler, but a *test* binary runs many, and
/// reproducibility is exactly what the tests check.
pub fn reset_sequence() {
    SEQUENCE.store(0, Ordering::Relaxed);
}

/// The per-thread state, as it is held in a guard slot.
///
/// Two words. `countdown` is bytes remaining until the next sample point, and
/// zero means "not yet seeded" — a real countdown is always at least 1, which
/// [`draw`] guarantees.
#[derive(Debug)]
pub struct State {
    countdown: AtomicU64,
    generator: AtomicU64,
}

impl State {
    pub const fn new() -> Self {
        Self {
            countdown: AtomicU64::new(0),
            generator: AtomicU64::new(0),
        }
    }

    /// Returns the slot to the state a thread that has never used it finds.
    ///
    /// Called when a slot changes hands. A thread that inherited the previous
    /// owner's countdown would not be wrong on average, but it would make the
    /// sequence depend on slot reuse, which is the reproducibility this module
    /// promises.
    pub fn clear(&self) {
        self.countdown.store(0, Ordering::Relaxed);
        self.generator.store(0, Ordering::Relaxed);
    }

    /// Whether an allocation of `size` bytes carries a sample point, advancing
    /// the countdown past it either way.
    ///
    /// `interval` is the mean spacing `R` and must be non-zero; the caller has
    /// already decided that sampling is on.
    ///
    /// Relaxed throughout, and single-threaded despite the atomics: only the
    /// owning thread touches these words, exactly as with the depth counter
    /// beside them. The atomics are for the shared `static`, not for sharing.
    #[inline]
    pub fn admits(&self, size: usize, interval: u64) -> bool {
        let mut countdown = self.countdown.load(Ordering::Relaxed);
        if countdown == 0 {
            countdown = self.initialize(interval);
        }

        let size = size as u64;
        if size < countdown {
            // The common case by construction: no sample point in this
            // allocation. One load, one compare, one store, no floating point,
            // and the generator is not touched at all.
            self.countdown.store(countdown - size, Ordering::Relaxed);
            return false;
        }

        // At least one point falls inside. Walk past however many do, so that a
        // single allocation much larger than `R` does not leave a backlog that
        // samples the next several allocations regardless of their size.
        let mut generator = self.generator.load(Ordering::Relaxed);
        let mut remaining = size - countdown;
        loop {
            let next = draw(&mut generator, interval);
            if remaining < next {
                countdown = next - remaining;
                break;
            }
            remaining -= next;
        }

        self.countdown.store(countdown, Ordering::Relaxed);
        self.generator.store(generator, Ordering::Relaxed);
        true
    }

    /// Seeds this thread's generator and draws its first countdown.
    ///
    /// Both words are stored before returning, and that is the point of this
    /// being a function rather than three lines in [`State::admits`]. Storing the
    /// generator only on the sampled path loses the seed on every thread whose
    /// first allocation is not sampled — which is almost all of them, since not
    /// sampling is the common case — leaving every such thread drawing from the
    /// same zero-valued generator. That was the first version, and the
    /// reproducibility test could not see it because a single thread that keeps
    /// redrawing from zero is perfectly reproducible.
    #[cold]
    #[inline(never)]
    fn initialize(&self, interval: u64) -> u64 {
        let mut generator = seed();
        let countdown = draw(&mut generator, interval);
        self.generator.store(generator, Ordering::Relaxed);
        self.countdown.store(countdown, Ordering::Relaxed);
        countdown
    }

    /// A state seeded explicitly, for tests that must not depend on the order
    /// the process happened to hand out seeds.
    #[cfg(test)]
    fn seeded(seed: u64, interval: u64) -> Self {
        let state = Self::new();
        let mut generator = mix(seed);
        let countdown = draw(&mut generator, interval);
        state.generator.store(generator, Ordering::Relaxed);
        state.countdown.store(countdown, Ordering::Relaxed);
        state
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

/// The next seed in the process-wide sequence.
///
/// `#[cold]`, because it runs once per thread and its only caller is otherwise
/// three instructions.
#[cold]
#[inline(never)]
fn seed() -> u64 {
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    mix(SEED ^ sequence)
}

/// Draws the distance to the next sample point, in bytes.
///
/// The gaps in a Poisson process with mean `interval` are exponentially
/// distributed, and `-interval * ln(u)` for uniform `u` is the standard way to
/// get one. The result is at least 1, so a caller looping on it always makes
/// progress.
#[inline]
fn draw(generator: &mut u64, interval: u64) -> u64 {
    *generator = mix(generator.wrapping_add(SEED));

    // 53 bits, the mantissa, mapped into (0, 1]. Excluding zero matters: `ln(0)`
    // is negative infinity, which would make the countdown `u64::MAX` and stop
    // sampling for the rest of the run.
    let uniform = (((*generator >> 11) + 1) as f64) * (1.0 / ((1u64 << 53) as f64 + 1.0));

    let gap = -(interval as f64) * uniform.ln();
    // `as u64` saturates at zero and at `u64::MAX` on a NaN or an out-of-range
    // float, which is the behaviour wanted here rather than something to guard
    // against: both ends are a valid, if unlikely, distance.
    (gap as u64).max(1)
}

/// How many allocations one sampled allocation of `size` bytes stands for.
///
/// This is `1 / p(size)`, the reciprocal of the probability that sampling would
/// have caught it. Returns 1 when sampling is off, and never less than 1.
///
/// # Why this is not stored on the block
///
/// It is a pure function of the size and the interval, and
/// [`Engine::record_free`](super::engine::Engine::record_free) is handed the
/// size by the shim. Recomputing it costs one `exp` on a path that runs only for
/// sampled blocks; storing it would add a word to
/// [`LiveBlock`](super::live::LiveBlock), which is asserted at 16 bytes and
/// multiplied by a live-block ceiling of four million.
///
/// The recomputation is exact rather than approximate, so the bytes a free
/// subtracts are the same integer the allocation added, and live bytes return to
/// zero on a balanced run.
#[inline]
pub fn scale(size: usize, interval: Option<u64>) -> f64 {
    let Some(interval) = interval else {
        return 1.0;
    };
    if size == 0 {
        // Never sampled: a zero-byte request covers no part of the byte stream.
        // Reached only if a caller asks anyway, and 1.0 keeps it finite.
        return 1.0;
    }

    let ratio = size as f64 / interval as f64;
    // `-expm1(-x)` rather than `1 - exp(-x)`. For the small ratios that dominate
    // — a 16-byte node against a 1 MiB interval is 1.5e-5 — `exp(-x)` rounds to
    // a value within an ulp of 1 and the subtraction loses most of the
    // significant digits, which is precisely the case whose scale is largest and
    // therefore matters most.
    let probability = -((-ratio).exp_m1());
    if probability <= 0.0 {
        // Unreachable for a non-zero size, because `expm1` is exact near zero.
        // Present so that this function has no path returning infinity.
        return 1.0;
    }
    (1.0 / probability).max(1.0)
}

/// The integer weight an event of `size` bytes contributes to a byte total.
///
/// Rounded once, here, so that the allocation and the free of one block agree to
/// the byte. See [`scale`].
#[inline]
pub fn weighted_bytes(size: usize, interval: Option<u64>) -> u64 {
    if interval.is_none() {
        return size as u64;
    }
    round(size as f64 * scale(size, interval))
}

/// The integer weight an event of `size` bytes contributes to a block count.
#[inline]
pub fn weighted_blocks(size: usize, interval: Option<u64>) -> u64 {
    if interval.is_none() {
        return 1;
    }
    round(scale(size, interval))
}

/// Nearest integer, clamped into `u64`.
///
/// `f64::round` then `as u64`, which saturates rather than wrapping. A scale
/// large enough to reach the ceiling means an interval so large that nothing is
/// being sampled, and saturating is the honest answer there.
#[inline]
fn round(value: f64) -> u64 {
    value.round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The estimator is unbiased, which is the whole claim.
    ///
    /// Averaged over eight seeds, because one seed is one draw from a random
    /// process and the first version of this test failed on a sample count whose
    /// own standard error was three times the tolerance. The interval is set
    /// relative to the size so that every row gets a comparable number of
    /// samples: at `size * 64` the sampling probability is about 1.55%, so
    /// 100,000 allocations give roughly 1,550 samples per seed and 12,400 across
    /// the eight, for a standard error under 1%.
    ///
    /// Deterministic despite being statistical: the seeds are fixed, so this
    /// either passes forever or fails on the first run after a change.
    #[test]
    fn weighted_totals_recover_the_true_total() {
        for size in [16usize, 100, 4_096, 1 << 20] {
            let interval = size as u64 * 64;
            let count = 100_000u64;

            let mut estimated: u64 = 0;
            let mut sampled = 0u64;
            for seed in 0..8u64 {
                let state = State::seeded(seed, interval);
                for _ in 0..count {
                    if state.admits(size, interval) {
                        sampled += 1;
                        estimated += weighted_bytes(size, Some(interval));
                    }
                }
            }

            let truth = size as u64 * count * 8;
            let error = (estimated as f64 - truth as f64) / truth as f64;
            assert!(
                error.abs() < 0.03,
                "size {size}: estimated {estimated} against {truth} ({sampled} samples, \
                 {:+.1}% error)",
                error * 100.0
            );
        }
    }

    /// The size-blind version of this scheme fails here, which is why the scale
    /// is computed per allocation.
    ///
    /// One workload, two populations: a great many small blocks and a few large
    /// ones that hold most of the bytes. A single global multiplier gets one of
    /// the two right and inflates or erases the other. Both totals have to come
    /// back, separately, or the profile points at the wrong program.
    #[test]
    fn a_mixed_workload_recovers_both_populations() {
        let interval = 1 << 16;
        let small = 32usize;
        let large = 1 << 22;
        let smalls = 500_000u64;
        let larges = 200u64;

        let mut small_bytes = 0u64;
        let mut large_bytes = 0u64;
        for seed in 0..4u64 {
            let state = State::seeded(seed, interval);
            for round in 0..smalls {
                if state.admits(small, interval) {
                    small_bytes += weighted_bytes(small, Some(interval));
                }
                // The large ones are spread through the run rather than bunched.
                if round % (smalls / larges) == 0 && state.admits(large, interval) {
                    large_bytes += weighted_bytes(large, Some(interval));
                }
            }
        }

        let small_truth = small as u64 * smalls * 4;
        let large_truth = large as u64 * larges * 4;
        let small_error = (small_bytes as f64 - small_truth as f64) / small_truth as f64;
        let large_error = (large_bytes as f64 - large_truth as f64) / large_truth as f64;

        assert!(
            small_error.abs() < 0.05,
            "small: {small_bytes} against {small_truth} ({:+.1}%)",
            small_error * 100.0
        );
        // The large ones are individually certain to be sampled, so this is not
        // an estimate at all and the tolerance is for the arithmetic only.
        assert!(
            large_error.abs() < 0.001,
            "large: {large_bytes} against {large_truth} ({:+.1}%)",
            large_error * 100.0
        );
    }

    /// An allocation far larger than the interval is always caught.
    ///
    /// This is the property that makes byte-weighted sampling worth the
    /// arithmetic: the 100 MiB buffer is the answer, and a scheme that could miss
    /// it would be worthless however cheap.
    #[test]
    fn allocations_far_above_the_interval_are_never_missed() {
        let interval = 1 << 16;
        let state = State::seeded(11, interval);
        for _ in 0..1_000 {
            assert!(
                state.admits(64 << 20, interval),
                "a 64 MiB allocation escaped a 64 KiB sampling interval"
            );
        }
    }

    /// And is not scaled up when it is caught.
    #[test]
    fn certain_allocations_are_not_inflated() {
        let interval = 1 << 16;
        assert_eq!(weighted_blocks(64 << 20, Some(interval)), 1);
        assert_eq!(weighted_bytes(64 << 20, Some(interval)), 64 << 20);
    }

    /// Small allocations are scaled by about the ratio, which is what makes
    /// their totals come out right.
    #[test]
    fn small_allocations_carry_the_interval_ratio() {
        let interval = 1 << 20;
        let scale = scale(16, Some(interval));
        let expected = interval as f64 / 16.0;
        assert!(
            (scale - expected).abs() / expected < 0.001,
            "scale {scale} against expected {expected}"
        );
    }

    /// Off means off: no scaling, and the weights are the plain values.
    #[test]
    fn no_interval_weighs_nothing() {
        assert_eq!(scale(16, None), 1.0);
        assert_eq!(weighted_bytes(4_096, None), 4_096);
        assert_eq!(weighted_blocks(4_096, None), 1);
    }

    /// A free must subtract exactly what the allocation added, or live bytes
    /// drift away from zero over a balanced run.
    #[test]
    fn the_weight_of_a_size_is_stable() {
        let interval = Some(1 << 18);
        for size in [1usize, 15, 16, 17, 1_000, 65_536, 1 << 22] {
            let first = weighted_bytes(size, interval);
            for _ in 0..100 {
                assert_eq!(weighted_bytes(size, interval), first, "size {size} drifted");
            }
        }
    }

    /// A single allocation far larger than the interval must not leave a backlog
    /// that samples the next several allocations whatever their size.
    #[test]
    fn a_huge_allocation_does_not_arm_the_next_ones() {
        let interval = 1 << 16;
        let state = State::seeded(3, interval);
        assert!(state.admits(1 << 30, interval));

        let mut sampled = 0;
        for _ in 0..10_000 {
            if state.admits(8, interval) {
                sampled += 1;
            }
        }
        // 10,000 eight-byte allocations against a 64 KiB interval is 80,000
        // bytes, so one or two samples is expected and a dozen would mean the
        // backlog was not walked off.
        assert!(sampled < 12, "{sampled} samples after a 1 GiB allocation");
    }

    /// One seed, one sequence of decisions, which is what `TimeSource::Events`
    /// reproducibility rests on.
    ///
    /// This checks the arithmetic only. That a *process* draws the same seeds in
    /// the same order across runs is a property of
    /// [`reset_sequence`] and the global counter, and it cannot be tested here:
    /// cargo runs these tests on several threads at once, so the counter is
    /// being consumed by other tests while this one reads it. `tests/sampling.rs`
    /// checks it where it can be checked, by running a workload twice.
    #[test]
    fn one_seed_gives_one_sequence_of_decisions() {
        let interval = 1 << 14;
        let sizes = [24usize, 100, 8, 4_096, 17, 64, 1_000];

        let record = |seed| {
            let state = State::seeded(seed, interval);
            let mut decisions = Vec::new();
            for round in 0..2_000 {
                decisions.push(state.admits(sizes[round % sizes.len()], interval));
            }
            decisions
        };

        assert_eq!(record(1), record(1));
        // And a different seed does not merely reproduce the same answer, which
        // is the way the assertion above passes without the seed being used.
        assert_ne!(record(1), record(2));
    }

    /// A thread whose first allocation is not sampled must still keep its seed.
    ///
    /// The bug this exists for: storing the generator only on the sampled path
    /// left every such thread drawing from a zero generator, so all of them
    /// shared one sequence. Not sampling is the common case, so that was almost
    /// every thread, and no single-threaded test could see it.
    #[test]
    fn a_seed_survives_an_unsampled_first_allocation() {
        let interval = 1 << 20;
        let state = State::new();

        // Far below the interval, so this is not sampled with near-certainty.
        assert!(!state.admits(8, interval));
        let generator = state.generator.load(Ordering::Relaxed);
        assert_ne!(generator, 0, "the seed was discarded");

        // And the countdown that was drawn with it is still in force.
        let countdown = state.countdown.load(Ordering::Relaxed);
        assert!(countdown > 0);
        state.admits(8, interval);
        assert_eq!(state.countdown.load(Ordering::Relaxed), countdown - 8);
    }

    /// The draw always moves the countdown forward, or `admits` spins.
    #[test]
    fn a_drawn_gap_is_never_zero() {
        let mut generator = 1;
        for _ in 0..100_000 {
            assert!(draw(&mut generator, 1) >= 1);
        }
        // A tiny interval is where a truncating conversion would produce zero.
        let mut generator = mix(7);
        for _ in 0..100_000 {
            assert!(draw(&mut generator, 2) >= 1);
        }
    }

    /// Clearing a slot must not leave the next thread mid-countdown.
    #[test]
    fn clearing_reseeds() {
        let interval = 1 << 12;
        let state = State::new();
        for _ in 0..100 {
            state.admits(64, interval);
        }
        state.clear();
        assert_eq!(state.countdown.load(Ordering::Relaxed), 0);
        assert_eq!(state.generator.load(Ordering::Relaxed), 0);
    }
}
