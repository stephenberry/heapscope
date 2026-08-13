//! The workload the three overhead fixtures run, and the protocol they report
//! it in.
//!
//! Not an example. `benches/overhead.rs` is the driver, and it measures nothing
//! itself, because the three configurations it compares cannot coexist in one
//! process: a `#[global_allocator]` is chosen at compile time and there is one
//! per binary, so "this program with heapscope, with `dhat-rs`, and with
//! neither" is three programs. This file is what makes them the same program in
//! every other respect — identical work, in identical order, at identical
//! sizes — and the checksum below is what holds them to it. The driver requires
//! all three to report the same one, so a fixture that quietly did different
//! work fails the run rather than publishing a faster number for it.
//!
//! # What the workload is made of, and why
//!
//! A profiler is paid per allocation, so the workload is allocation-dominated.
//! But a loop around one `Vec::with_capacity` would measure one program point,
//! one size class, and one live-block occupancy, and every profiler design
//! choice this benchmark exists to compare is a choice about what happens when
//! there are many. So the budget splits three ways:
//!
//! - **Churn**, half of it: allocated and freed immediately, reached through
//!   three call chains of different depths. Three program points, not one, and
//!   the deepest is what proves a captured stack reaches past the allocator's
//!   own frames — `benches/overhead.rs` requires the deepest one's name to
//!   appear in every profile written.
//! - **Steady state**, most of the rest: a live set of a fixed size with one
//!   block replaced per iteration. This is the shape a program that is neither
//!   growing nor shrinking presents to a live-block table, and it is the only
//!   phase where a free has to *find* something.
//! - **Growth**, an eighth: a buffer reallocated to a larger capacity each
//!   iteration. `realloc` is neither an allocation nor a free and both tools
//!   account for it their own way, which is a difference worth having in the
//!   number rather than outside it.
//!
//! Sizes cycle through a fixed list rather than being drawn at random. A random
//! workload would need a seeded generator in all three binaries and would buy
//! nothing: what matters is that the sizes are not all one size class, and a
//! fixed list is both sufficient for that and reproducible without argument.
//!
//! # What the number does and does not include
//!
//! The timer covers the workload and nothing else. Process startup, the
//! profiler's own construction — which for heapscope includes the frame-pointer
//! capability probe — and writing the profile at the end are all outside it,
//! and the last of those is reported separately because a user waits for it too.
//!
//! Fixed per-run costs that *are* inside it, chiefly first touch of whatever
//! the profiler allocates up front, are divided by [`ALLOCATIONS_PER_THREAD`]
//! along with everything else. That is the honest treatment for a number
//! labelled "per allocation", and it is why the count is published beside it:
//! at a tenth of the allocations the same fixed cost would read ten times
//! larger.

use std::hint::black_box;
use std::time::{Duration, Instant};

/// Allocations each thread makes, in every configuration.
///
/// Large enough that a run's one-off costs are spread thin over the
/// per-allocation figure, and small enough that the slowest configuration
/// finishes in seconds rather than minutes.
pub const ALLOCATIONS_PER_THREAD: usize = 250_000;

/// Blocks held at once during the steady-state phase.
///
/// Small next to heapscope's four-million-block default ceiling, on purpose:
/// this benchmark is about the per-event cost of recording, not about what
/// either tool does when its table is full. That is a separate experiment and
/// it needs a separate workload.
const LIVE_SET: usize = 4_096;

/// Allocations in the churn phase.
const CHURN: usize = ALLOCATIONS_PER_THREAD / 2;

/// Reallocations in the growth phase.
const GROWTH: usize = ALLOCATIONS_PER_THREAD / 8;

/// Replacements in the steady-state phase, after the live set is filled.
const STEADY: usize = ALLOCATIONS_PER_THREAD - CHURN - GROWTH - LIVE_SET;

// The phases have to add up, or the count published beside the timing is not
// the count that was made. A const assertion rather than a runtime one: this is
// arithmetic over constants and it can be settled at compile time.
const _: () = assert!(CHURN + GROWTH + LIVE_SET + STEADY == ALLOCATIONS_PER_THREAD);
const _: () = assert!(
    STEADY > 0,
    "the allocation budget is too small for the live set"
);

/// Allocation sizes, cycled through.
///
/// Deliberately not all powers of two and not all small: an allocator sorts
/// requests into size classes and both profilers bucket them for reporting, so
/// a workload that only ever asks for 64 bytes measures one bucket of each.
const SIZES: [usize; 8] = [24, 56, 120, 248, 33, 1_000, 4_096, 17];

/// Capacity steps the growth phase reallocates through before starting over.
///
/// Bounded at 2 KiB because every step copies the previous capacity forward.
/// Left unbounded, the memcpy would grow until it, rather than the profiler,
/// was what the phase measured.
const GROWTH_STEPS: usize = 6;

/// What one run of the workload did.
pub struct Run {
    /// Threads it ran on.
    pub threads: usize,
    /// Allocations made, across all threads, excluding the handful the
    /// workload's own bookkeeping makes.
    pub allocations: usize,
    /// How long the work took, measured from the moment every thread was ready.
    pub elapsed: Duration,
    /// A value derived from every block the workload touched.
    ///
    /// The driver requires all three configurations to report the same one at a
    /// given thread count. It is the only thing standing between this benchmark
    /// and a fixture that was accidentally edited into doing less work than the
    /// others, which would show up as an improvement.
    pub checksum: u64,
}

impl Run {
    /// Writes the run out in the driver's `key=value` protocol.
    ///
    /// A fixture prints this and then whatever else it alone can report —
    /// `shutdown-ns`, `profile-bytes`, `total-blocks`. The driver requires every
    /// key here and refuses a run that omits one, so a fixture that failed
    /// before reporting cannot be read as a fixture that reported a zero.
    pub fn report(&self, configuration: &str) {
        println!("configuration={configuration}");
        println!("threads={}", self.threads);
        println!("allocations={}", self.allocations);
        println!("checksum={}", self.checksum);
        println!("workload-ns={}", self.elapsed.as_nanos());
        if let Some(bytes) = max_rss_bytes() {
            println!("max-rss-bytes={bytes}");
        }
    }
}

/// Runs the workload on `threads` threads.
///
/// Every thread does the same work, distinguished only by a tag byte that
/// reaches the checksum, so that the threads contend for the same program
/// points and the same shards — which is the case the two designs differ on.
pub fn run(threads: usize) -> Run {
    assert!(threads > 0, "the workload needs at least one thread");

    let start_line = std::sync::Barrier::new(threads);
    let started = std::sync::Mutex::new(None::<Instant>);
    let mut checksums = Vec::with_capacity(threads);

    std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(threads);
        for index in 0..threads {
            let (start_line, started) = (&start_line, &started);
            workers.push(scope.spawn(move || {
                // The timer starts when the last thread arrives, so that thread
                // creation is not counted and the threads actually overlap.
                // Taken from `benches/contention.rs`, which needs it for the
                // same reason.
                if start_line.wait().is_leader() {
                    *started.lock().expect("the start instant is not poisoned") =
                        Some(Instant::now());
                }
                one_thread(index)
            }));
        }
        for worker in workers {
            checksums.push(worker.join().expect("a workload thread panicked"));
        }
    });

    let elapsed = started
        .lock()
        .expect("the start instant is not poisoned")
        .expect("the barrier leader recorded a start instant")
        .elapsed();

    // Folded in thread order rather than summed, so that the count of threads
    // is part of what the checksum attests to.
    let checksum = checksums.into_iter().fold(0, mix);

    Run {
        threads,
        allocations: threads * ALLOCATIONS_PER_THREAD,
        elapsed,
        checksum,
    }
}

/// One thread's share, in the three phases the module documentation describes.
fn one_thread(index: usize) -> u64 {
    let tag = index as u8;
    let mut checksum = mix(0, index as u64);

    for i in 0..CHURN {
        let size = SIZES[i % SIZES.len()];
        // Three call chains rather than one, so the profile has program points
        // to distinguish and the deepest reaches four frames past the shim.
        let touched = match i % 3 {
            0 => churn(size, tag),
            1 => churn_one_deeper(size, tag),
            _ => churn_two_deeper(size, tag),
        };
        checksum = mix(checksum, touched);
    }

    let mut live: Vec<Vec<u8>> = Vec::with_capacity(LIVE_SET);
    for i in 0..LIVE_SET {
        live.push(block(SIZES[i % SIZES.len()], tag));
    }
    for i in 0..STEADY {
        let slot = i % LIVE_SET;
        // The new block is allocated before the old one is dropped, so the live
        // set momentarily holds one more than its nominal size. That is what a
        // program replacing a cache entry does, and it is the case where a free
        // has to find an entry that is genuinely there.
        live[slot] = block(SIZES[(i + 3) % SIZES.len()], tag);
        checksum = mix(checksum, live[slot].capacity() as u64);
    }
    checksum = mix(checksum, live.len() as u64);
    drop(live);

    let mut growing: Vec<u8> = Vec::new();
    for i in 0..GROWTH {
        let step = i % GROWTH_STEPS;
        if step == 0 {
            // Frees the previous buffer and returns the capacity to zero, so
            // the next `reserve_exact` starts the sequence again rather than
            // finding the room already there and allocating nothing.
            growing = Vec::new();
        }
        growing.reserve_exact(64 << step);
        checksum = mix(checksum, growing.capacity() as u64);
    }
    drop(growing);

    checksum
}

/// Allocates a block and writes to it.
///
/// The write matters: an allocation never touched costs the operating system
/// nothing, and a workload made entirely of them would be measuring a profiler
/// against an allocator that had not yet done its half of the work.
#[inline(never)]
fn block(size: usize, tag: u8) -> Vec<u8> {
    let mut block = Vec::with_capacity(size);
    block.push(tag);
    block
}

/// The shallowest of the three churn call chains.
#[inline(never)]
fn churn(size: usize, tag: u8) -> u64 {
    let block = block(size, tag);
    u64::from(block[0]) + block.capacity() as u64
}

/// One frame deeper than [`churn`].
///
/// The `black_box` is what keeps the frame: without it the call is in tail
/// position, and a tail call leaves the two chains indistinguishable in a
/// captured stack — which would quietly turn three program points into two.
#[inline(never)]
fn churn_one_deeper(size: usize, tag: u8) -> u64 {
    let touched = churn(size, tag);
    black_box(touched)
}

/// Two frames deeper than [`churn`], and the name `benches/overhead.rs` looks
/// for in every profile the run writes.
#[inline(never)]
fn churn_two_deeper(size: usize, tag: u8) -> u64 {
    let touched = churn_one_deeper(size, tag);
    black_box(touched)
}

/// Folds `value` into `accumulator`.
///
/// Order-dependent on purpose, so that the same values arriving in a different
/// order do not agree.
fn mix(accumulator: u64, value: u64) -> u64 {
    accumulator
        .rotate_left(7)
        .wrapping_add(value)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Peak resident set size in bytes, or `None` where this build cannot ask.
///
/// Peak rather than current, because the question is what the profiler cost the
/// machine at its worst, and because a reading taken at exit has already missed
/// it. Resident rather than anything the process counts for itself: a profiler
/// that keeps its state in an arena it maps directly is invisible to a counting
/// allocator, and the arena is exactly what is being compared.
#[cfg(all(unix, target_pointer_width = "64"))]
pub fn max_rss_bytes() -> Option<u64> {
    /// As much of `struct rusage` as this needs, plus room.
    ///
    /// The two leading `struct timeval`s occupy sixteen bytes each on every
    /// 64-bit unix this crate builds for, though macOS and Linux disagree about
    /// the width of the second member — `suseconds_t` is `i32` on Darwin and
    /// `i64` on glibc — so their space is reserved rather than named. `ru_maxrss`
    /// follows at offset 32 on both.
    ///
    /// The trailing array is longer than the thirteen `long`s the real struct
    /// ends with, and that is deliberate. `getrusage` writes the whole
    /// structure, so a definition short by one field is a stack overwrite that
    /// would appear as unrelated corruption; a definition that is too long
    /// wastes 150 bytes of stack and cannot be wrong in that direction.
    #[repr(C)]
    struct Rusage {
        times: [u64; 4],
        max_rss: i64,
        rest: [i64; 32],
    }

    extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }

    /// `RUSAGE_SELF`. The same value on Darwin and on Linux.
    const SELF: i32 = 0;

    let mut usage = Rusage {
        times: [0; 4],
        max_rss: 0,
        rest: [0; 32],
    };
    // SAFETY: `getrusage` writes a `struct rusage` through the pointer and
    // reads nothing through it. The pointee is a live, uniquely borrowed local
    // that is at least as large as the structure the platform will write.
    if unsafe { getrusage(SELF, &mut usage) } != 0 {
        return None;
    }

    let max_rss = u64::try_from(usage.max_rss).ok()?;
    // Darwin reports bytes here and every other unix reports kilobytes. Getting
    // this backwards is a factor of 1,024 in a published number, so it is
    // checked rather than assumed: `benches/overhead.rs` documents the
    // cross-check against `/usr/bin/time`.
    Some(if cfg!(target_vendor = "apple") {
        max_rss
    } else {
        max_rss * 1024
    })
}

/// Peak resident set size, where the platform is not one this knows how to ask.
#[cfg(not(all(unix, target_pointer_width = "64")))]
pub fn max_rss_bytes() -> Option<u64> {
    None
}

/// How a fixture is asked to capture stacks.
pub enum Unwinder {
    /// Whatever the tool chooses for the platform.
    Default,
    /// The platform unwinder, where the tool offers a choice.
    System,
}

/// The thread count and profiler settings a fixture was invoked with.
pub struct Arguments {
    /// Threads to run the workload on.
    pub threads: usize,
    /// Frames to capture per allocation, or `None` for the tool's own default.
    pub frames: Option<usize>,
    /// How stacks are captured. Only heapscope offers the choice; the other two
    /// fixtures take the argument and ignore it.
    pub unwinder: Unwinder,
    /// Mean bytes between sample points, or `None` to record everything. Only
    /// heapscope offers this; the other two fixtures take the argument and
    /// ignore it, as they do the unwinder.
    pub sampling: Option<u64>,
    /// Where the profile goes. Ignored by the unprofiled fixture, which is
    /// given one anyway so that all three take the same command line.
    pub output: String,
}

/// Usage, quoted by every parse failure below.
const USAGE: &str = "<threads> <frames|default> <default|system> <sampling|none> <output-path>";

/// Parses the fixture command line.
///
/// Positional and mandatory, all four. A fixture is only ever run by the
/// driver, and an absent argument quietly defaulted is how a benchmark ends up
/// publishing a configuration nobody chose.
pub fn arguments() -> Arguments {
    let mut argv = std::env::args().skip(1);
    let mut next = |name: &str| {
        argv.next()
            .unwrap_or_else(|| panic!("missing argument <{name}>; usage: {USAGE}"))
    };

    let threads = next("threads");
    let frames = next("frames");
    let unwinder = next("unwinder");
    let sampling = next("sampling");
    let output = next("output-path");

    Arguments {
        threads: threads
            .parse()
            .unwrap_or_else(|error| panic!("<threads> is not a count: {threads:?}: {error}")),
        frames: match frames.as_str() {
            "default" => None,
            other => Some(
                other
                    .parse()
                    .unwrap_or_else(|error| panic!("<frames> is not a count: {other:?}: {error}")),
            ),
        },
        unwinder: match unwinder.as_str() {
            "default" => Unwinder::Default,
            "system" => Unwinder::System,
            other => panic!("<unwinder> is `default` or `system`, not {other:?}"),
        },
        sampling: match sampling.as_str() {
            "none" => None,
            other => Some(other.parse().unwrap_or_else(|error| {
                panic!("<sampling> is `none` or a byte count: {other:?}: {error}")
            })),
        },
        output,
    }
}
