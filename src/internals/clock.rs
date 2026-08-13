//! The profiler's notion of time.
//!
//! DHAT's `tu` and `Mtu` fields are free-text unit labels, so the time base is
//! ours to choose. Two are offered, and which is the default matters more than
//! it looks.
//!
//! # `Events` is the default
//!
//! A monotonically increasing count of allocation events. It costs nothing — the
//! counter is bumped anyway — and it is what lets two runs of one workload
//! record **the same numbers**, which turns most regressions into a diff.
//!
//! Every lifetime in the profile is a difference of two of these counts, so with
//! a wall clock they would differ in every run and every digit. The rest of what
//! reproducibility takes is elsewhere: the profile's *order* comes from
//! `Snapshot::points`, and what a profile is still allowed to differ in between
//! runs — its pid, where the loader mapped it, and the profiler's measurements
//! of itself — is listed in `ci/check-reproducible.sh`, which checks the claim
//! against real processes.
//!
//! `dhat-rs` and Valgrind both use wall-clock time, and PLAN.md revision 1
//! followed them. That was wrong on performance grounds as much as on
//! reproducibility grounds: `Instant::now()` was measured at **17.7 ns**, about
//! the same as an entire frame-pointer walk (PLAN.md section 4.4). Choosing
//! wall-clock time roughly doubles the cost of the hot path, to produce a number
//! that is worse for testing.
//!
//! # `Monotonic` is available, and honest about its cost
//!
//! Microseconds from a raw monotonic clock, for users who genuinely want to
//! correlate a profile with wall-clock behaviour.
//!
//! It does not use `std::time::Instant`. `Instant` cannot be stored in a
//! `static` without lazy initialization, and lazy initialization is not
//! reachable from a shim that is live before `main`. Reading the platform clock
//! directly keeps the whole engine const-initializable.

use std::sync::atomic::{AtomicU64, Ordering};

/// Which time base a profile uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimeSource {
    /// A count of the events the profiler observed, which in a heap run means
    /// allocations. Free, and deterministic.
    ///
    /// "observed" rather than "recorded": `record_alloc` ticks before it inserts
    /// the live-block entry, so an allocation dropped because that table was
    /// full still advances the clock. A free does not tick at all — a block
    /// allocated and freed with nothing in between has a lifetime of zero, which
    /// is what "no events elapsed" means.
    #[default]
    Events,
    /// Microseconds from process start, from the platform's monotonic clock.
    Monotonic,
}

impl TimeSource {
    /// The unit label for the DHAT `tu` field.
    pub fn unit(self) -> &'static str {
        match self {
            TimeSource::Events => "events",
            TimeSource::Monotonic => "µs",
        }
    }

    /// The unit label spelled out, for prose.
    ///
    /// "observed events" rather than "allocation events", because only a heap
    /// run observes allocations — an ad hoc profile whose clock counts ad hoc
    /// events used to report its `te` in allocation events, of which it has
    /// none. And "observed" rather than "recorded", because the clock ticks
    /// before the live-block table is consulted, so an allocation that table had
    /// no room for still advanced it.
    pub fn unit_long(self) -> &'static str {
        match self {
            TimeSource::Events => "observed events",
            TimeSource::Monotonic => "microseconds",
        }
    }

    /// The label for a *million* of the unit, which is the DHAT `Mtu` field.
    ///
    /// The viewer uses it as the denominator of a rate — Valgrind's `Minstr`
    /// renders as "3.5/Minstr", meaning 3.5 per million instructions — so it
    /// names a quantity, not a duration.
    pub fn unit_million(self) -> &'static str {
        match self {
            TimeSource::Events => "Mevent",
            TimeSource::Monotonic => "Mµs",
        }
    }
}

/// A monotonically increasing time source.
///
/// Const-initializable, so it can live in the same `static` as the rest of the
/// engine.
#[derive(Debug)]
pub struct Clock {
    /// Allocation events observed. Also the value reported in `Events` mode.
    events: AtomicU64,
    /// Raw platform time at which the clock was started, in nanoseconds.
    /// Zero means "not started", which reads as time zero.
    origin_nanos: AtomicU64,
}

impl Clock {
    /// Creates a stopped clock reading zero.
    pub const fn new() -> Self {
        Self {
            events: AtomicU64::new(0),
            origin_nanos: AtomicU64::new(0),
        }
    }

    /// Marks the start of profiling, resetting both bases.
    pub fn start(&self) {
        self.events.store(0, Ordering::Relaxed);
        // A zero reading would be indistinguishable from "not started"; one
        // nanosecond of skew is not worth a second flag.
        self.origin_nanos
            .store(monotonic_nanos().max(1), Ordering::Relaxed);
    }

    /// Advances the event counter and returns the new time in `source`'s unit.
    ///
    /// This is the hot-path entry point: it must be called exactly once per
    /// recorded event, because in `Events` mode its return value *is* the time.
    #[inline]
    pub fn tick(&self, source: TimeSource) -> u64 {
        let events = self.events.fetch_add(1, Ordering::Relaxed) + 1;
        match source {
            TimeSource::Events => events,
            TimeSource::Monotonic => self.elapsed_micros(),
        }
    }

    /// Reads the current time in `source`'s unit without advancing anything.
    #[inline]
    pub fn now(&self, source: TimeSource) -> u64 {
        match source {
            TimeSource::Events => self.events.load(Ordering::Relaxed),
            TimeSource::Monotonic => self.elapsed_micros(),
        }
    }

    /// Number of events counted so far.
    pub fn events(&self) -> u64 {
        self.events.load(Ordering::Relaxed)
    }

    fn elapsed_micros(&self) -> u64 {
        let origin = self.origin_nanos.load(Ordering::Relaxed);
        if origin == 0 {
            return 0;
        }
        monotonic_nanos().saturating_sub(origin) / 1_000
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

/// Reads the platform's monotonic clock, in nanoseconds from an arbitrary
/// origin.
///
/// Never goes backwards and is unaffected by wall-clock adjustments.
pub fn monotonic_nanos() -> u64 {
    imp::monotonic_nanos()
}

#[cfg(unix)]
mod imp {
    use std::ffi::c_int;
    #[cfg(target_vendor = "apple")]
    use std::ffi::c_uint;

    // `struct timespec` on 64-bit Linux and Darwin: `time_t` and the nanosecond
    // field are both `long`. This crate supports 64-bit targets only, so both
    // are `i64`; a 32-bit port would have to revisit this.
    #[repr(C)]
    struct Timespec {
        seconds: i64,
        nanoseconds: i64,
    }

    // `clockid_t` is not the same type everywhere. glibc defines it as
    // `__S32_TYPE`, an `int`; Darwin defines it as an unnamed enum whose values
    // are all non-negative, which clang gives an *unsigned* underlying type.
    // Declaring one as the other is an ABI mismatch — technically undefined even
    // though both land in the same register, and Miri rejects it outright.
    #[cfg(target_os = "linux")]
    type ClockId = c_int;
    #[cfg(target_vendor = "apple")]
    type ClockId = c_uint;

    // `CLOCK_MONOTONIC` is not the same number everywhere either: 1 on Linux and
    // 6 on Darwin. Getting it wrong would silently select a different clock
    // rather than fail, so both are spelled out.
    #[cfg(target_os = "linux")]
    const CLOCK_MONOTONIC: ClockId = 1;
    #[cfg(target_vendor = "apple")]
    const CLOCK_MONOTONIC: ClockId = 6;

    extern "C" {
        fn clock_gettime(clock_id: ClockId, timespec: *mut Timespec) -> c_int;
    }

    pub(super) fn monotonic_nanos() -> u64 {
        let mut timespec = Timespec {
            seconds: 0,
            nanoseconds: 0,
        };
        // SAFETY: `timespec` is a valid, writable, correctly laid out
        // `struct timespec`. `clock_gettime` allocates nothing and is
        // async-signal-safe.
        let rc = unsafe { clock_gettime(CLOCK_MONOTONIC, &mut timespec) };
        if rc != 0 {
            return 0;
        }
        (timespec.seconds as u64)
            .wrapping_mul(1_000_000_000)
            .wrapping_add(timespec.nanoseconds as u64)
    }
}

#[cfg(windows)]
mod imp {
    use std::sync::atomic::{AtomicU64, Ordering};

    #[link(name = "kernel32", kind = "raw-dylib")]
    extern "system" {
        fn QueryPerformanceCounter(count: *mut i64) -> i32;
        fn QueryPerformanceFrequency(frequency: *mut i64) -> i32;
    }

    /// Ticks per second. Fixed for the life of the process, so it is read once
    /// and cached; zero means "not yet read".
    static FREQUENCY: AtomicU64 = AtomicU64::new(0);

    fn frequency() -> u64 {
        let cached = FREQUENCY.load(Ordering::Relaxed);
        if cached != 0 {
            return cached;
        }
        let mut raw = 0i64;
        // SAFETY: a valid out-pointer to an initialized local. The call cannot
        // fail on any Windows version this crate supports.
        unsafe { QueryPerformanceFrequency(&mut raw) };
        let value = if raw > 0 { raw as u64 } else { 1 };
        // A benign race: every thread computes the same value.
        FREQUENCY.store(value, Ordering::Relaxed);
        value
    }

    pub(super) fn monotonic_nanos() -> u64 {
        let mut raw = 0i64;
        // SAFETY: a valid out-pointer to an initialized local.
        unsafe { QueryPerformanceCounter(&mut raw) };
        if raw <= 0 {
            return 0;
        }
        // Scale before dividing, in 128-bit, so a high-frequency counter does
        // not lose resolution and a long-running process does not overflow.
        ((raw as u128 * 1_000_000_000) / frequency() as u128) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_mode_is_a_dense_counter() {
        let clock = Clock::new();
        clock.start();
        for expected in 1..=1000u64 {
            assert_eq!(clock.tick(TimeSource::Events), expected);
        }
        assert_eq!(clock.events(), 1000);
    }

    #[test]
    fn events_mode_is_reproducible() {
        // The property that makes `Events` the default: the same sequence of
        // operations produces the same timestamps, every run.
        let run = || {
            let clock = Clock::new();
            clock.start();
            (0..100)
                .map(|_| clock.tick(TimeSource::Events))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn now_does_not_advance_the_event_counter() {
        let clock = Clock::new();
        clock.start();
        clock.tick(TimeSource::Events);
        let before = clock.events();
        for _ in 0..10 {
            clock.now(TimeSource::Events);
        }
        assert_eq!(clock.events(), before);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "Miri's clock is virtual and does not advance with work"
    )]
    fn the_monotonic_clock_advances_and_never_goes_backwards() {
        let mut previous = monotonic_nanos();
        assert!(previous > 0, "the platform monotonic clock returned zero");
        for _ in 0..1000 {
            let now = monotonic_nanos();
            assert!(now >= previous, "the monotonic clock went backwards");
            previous = now;
        }
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "Miri's clock is virtual and does not advance with work"
    )]
    fn monotonic_mode_measures_elapsed_time() {
        let clock = Clock::new();
        clock.start();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let elapsed = clock.now(TimeSource::Monotonic);
        assert!(
            (15_000..500_000).contains(&elapsed),
            "20 ms read as {elapsed} µs"
        );
    }

    #[test]
    fn an_unstarted_clock_reads_zero_rather_than_a_huge_number() {
        // Without the origin check this would report nanoseconds since boot,
        // divided by a thousand — a plausible-looking but meaningless number.
        let clock = Clock::new();
        assert_eq!(clock.now(TimeSource::Monotonic), 0);
    }

    #[test]
    fn unit_labels_match_the_time_source() {
        assert_eq!(TimeSource::Events.unit(), "events");
        assert_eq!(TimeSource::Monotonic.unit(), "µs");
        assert_eq!(TimeSource::default(), TimeSource::Events);
    }

    #[test]
    fn ticks_are_unique_across_threads() {
        // Every recorded event needs its own timestamp in `Events` mode; two
        // events sharing one would make block lifetimes wrong.
        #[cfg(miri)]
        const PER_THREAD: usize = 20;
        #[cfg(not(miri))]
        const PER_THREAD: usize = 2_000;
        const THREADS: usize = 8;

        let clock = Clock::new();
        clock.start();

        let all: Vec<u64> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let clock = &clock;
                    s.spawn(move || {
                        (0..PER_THREAD)
                            .map(|_| clock.tick(TimeSource::Events))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });

        let mut sorted = all.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            THREADS * PER_THREAD,
            "two events were given the same timestamp"
        );
    }
}
