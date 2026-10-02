//! The recording engine: global counters, the peak, and how they stay
//! consistent with the per-program-point ones.
//!
//! # The peak problem
//!
//! DHAT reports, per program point, the live bytes at the instant the whole heap
//! peaked. Getting that right under concurrency is the hardest correctness
//! problem in this crate. PLAN.md section 4.3 modelled the naive approach over
//! 400,000 two-thread traces: 0.6% violated the profiler's own invariant
//! (`sum(pp.gb) > gmax`) and 8.3% silently under-attributed.
//!
//! [`super::gate::Gate`] supplies the missing linearization point, but *how* an
//! event uses it is what makes the result exact, and the obvious reading of the
//! plan does not work.
//!
//! # Why "upgrade to exclusive on a peak" is wrong
//!
//! The natural design — take the gate shared, apply the update, then take it
//! exclusive if the result looks like a peak — loses peaks. Thread A allocates,
//! bringing live bytes to 100, a new maximum. Before A can re-acquire
//! exclusively, thread B frees down to 50. A now takes the gate, reads 50, sees
//! it is below the recorded maximum, and records nothing. The peak of 100
//! happened and went unreported.
//!
//! # What this does instead
//!
//! An event that *could* be a peak takes the gate exclusively from the start,
//! and the decision is made from values read under that exclusion:
//!
//! - **Shared path.** A compare-exchange commits the new total only if it is
//!   still strictly below the recorded maximum at the moment of commit. Such an
//!   event provably is not a peak, so no epoch bump is needed and many threads
//!   proceed at once.
//! - **Exclusive path.** Anything that would reach or exceed the maximum. Under
//!   exclusion nothing else is in flight, so the totals, the per-point counters,
//!   and the epoch move as one atomic step.
//! - **Frees** always take the shared path: reducing live bytes cannot create a
//!   peak.
//!
//! The comparison is `>=`, not `>`. Valgrind is explicit (`dh_main.c:373-379`):
//! *"The use of `>=` rather than `>` means that if there are multiple equal
//! peaks we record the latest one."* A model check over 200,000 traces found
//! 12,110 mismatches with `>` and none with `>=`.
//!
//! # The cost, and where it is worst
//!
//! Steady state is one shared acquire per event. During a **monotonically
//! growing** phase, though, every allocation is a new peak and every one takes
//! the exclusive path — so warmup, not steady state, is the worst case, and it
//! is the specific thing the benchmark measures.

use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{fence, AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::arena::Arena;
use super::clock::{Clock, TimeSource};
use super::gate::Gate;
use super::guard::Guard;
use super::live::{LiveBlock, LiveBlocks};
use super::pp::{Counters, PpId, PpTable};
use super::shape::{Realloc, Shape, ShapeStats, Shapes};
use super::site::{Name, RegionId, Regions, Site, ThreadId, Threads, MAX_NAME};

/// What a run counts.
///
/// DHAT has three, and the choice is a property of the whole run rather than of
/// an individual event: the file carries one `mode`, and the viewer labels every
/// column from it. Summing heap blocks and ad hoc weights into one `tb` would
/// produce a number with no unit.
///
/// The two non-heap modes are driven entirely by the program calling
/// [`event`](fn@crate::event) or [`copied`](crate::copied). The allocator shim
/// records nothing in either: an allocation costs the reentrancy guard and two
/// relaxed atomic loads and then goes straight to the inner allocator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Mode {
    /// Every allocation the shim sees, with block lifetimes and the heap peak.
    ///
    /// The default, and the only mode in which a profile carries `tg`, `tuth`,
    /// and the per-point live and at-peak figures.
    #[default]
    Heap = 0,
    /// Weighted events the program reports itself with [`event`](fn@crate::event).
    ///
    /// The weight means whatever the program says it means: cache misses, rows
    /// parsed, retries. What the profile gives back is the call sites they
    /// happened at, ranked by summed weight.
    AdHoc = 1,
    /// Bytes the program reports having copied with [`copied`](crate::copied).
    ///
    /// A `GlobalAlloc` shim cannot see a `memcpy`, so unlike Valgrind's copy
    /// mode this counts what the program says it copied and nothing else.
    Copy = 2,
}

impl Mode {
    fn from_u8(raw: u8) -> Mode {
        match raw {
            1 => Mode::AdHoc,
            2 => Mode::Copy,
            _ => Mode::Heap,
        }
    }

    /// DHAT's `mode` field.
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Heap => "heap",
            Mode::AdHoc => "ad-hoc",
            Mode::Copy => "copy",
        }
    }

    /// DHAT's `verb`: what the viewer writes before a stack trace.
    pub fn verb(self) -> &'static str {
        match self {
            Mode::Heap => "Allocated",
            Mode::AdHoc => "Occurred",
            Mode::Copy => "Copied",
        }
    }

    /// Whether block lifetimes are recorded. DHAT's `bklt`.
    ///
    /// False for both non-heap modes, and that decides the shape of the file:
    /// `tg`, `tuth`, and the per-point `tl`, `mb`, `mbk`, `gb`, `gbk`, `eb`, and
    /// `ebk` are **omitted** rather than zeroed. An event has no lifetime and is
    /// never live, so a zero there would be a measurement rather than the
    /// absence of one.
    pub fn block_lifetimes(self) -> bool {
        matches!(self, Mode::Heap)
    }

    /// Whether the allocator shim records what it sees.
    #[inline(always)]
    pub fn records_allocations(self) -> bool {
        matches!(self, Mode::Heap)
    }

    /// What one unit, many units, and a count of them are called. DHAT's `bu`,
    /// `bsu`, and `bksu`.
    ///
    /// Copy mode really does count bytes, so it keeps the defaults; ad hoc
    /// weights are dimensionless, and calling them bytes would invite reading a
    /// total of 5,000 retries as five kilobytes.
    ///
    /// Total rather than optional so that the file and the text summary cannot
    /// name the same numbers differently. The emitter omits the fields when they
    /// match [`Mode::DEFAULT_UNITS`], which is what "omitted" means to the
    /// viewer; the summary always has words to print.
    pub fn units(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Mode::Heap | Mode::Copy => Mode::DEFAULT_UNITS,
            Mode::AdHoc => ("unit", "units", "events"),
        }
    }

    /// What the viewer calls these things when a file says nothing.
    pub const DEFAULT_UNITS: (&'static str, &'static str, &'static str) =
        ("byte", "bytes", "blocks");

    /// Whether an amount is a count of bytes, and so should be rendered in
    /// binary units.
    ///
    /// An ad hoc weight is dimensionless: printing 1,024 retries as `1.0 KiB`
    /// would be a unit error in the one place a reader cannot check it.
    pub fn counts_bytes(self) -> bool {
        !matches!(self, Mode::AdHoc)
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Lifecycle of the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum State {
    /// No profiler is attached; the shim is a pass-through.
    Idle = 0,
    /// Recording.
    Running = 1,
    /// Recording has stopped and the profile has been, or is being, written.
    Finished = 2,
    /// This process is a `fork` child of a profiled parent. Recording is off,
    /// and the inherited profile belongs to the parent.
    ForkedChild = 3,
    /// A profiler has claimed the engine and is still configuring it.
    ///
    /// Recording is off during this window, which is the point of having it.
    /// Everything a run's settings decide — the time base, the clock's zero, the
    /// capture strategy — used to be applied *after* the state became `Running`,
    /// so allocations on other threads in that window were recorded against
    /// settings that had not been chosen yet. With `Strategy::System` that meant
    /// capturing with the previous unwinder while the profile said otherwise.
    Starting = 4,
}

impl State {
    fn from_u8(raw: u8) -> State {
        match raw {
            1 => State::Running,
            2 => State::Finished,
            3 => State::ForkedChild,
            4 => State::Starting,
            _ => State::Idle,
        }
    }
}

/// How recording ended.
///
/// PLAN.md section 4.6 requires this to be in the output. The two automatic
/// paths do not produce the same profile and cannot be made to: `atexit`
/// handlers run last-in-first-out and share their list with C++ static
/// destructors through `__cxa_atexit`, so a profile written from one is taken
/// *partway through teardown*, with whatever other handlers were registered
/// after ours having already run. A profile written from `Profiler::drop` is
/// taken before teardown starts. Numbers from the two are both correct and
/// differ; a reader comparing them needs to know which is which.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Shutdown {
    /// Recording had not stopped when this reading was taken.
    #[default]
    Running = 0,
    /// The [`Profiler`](crate::Profiler) was dropped.
    Dropped = 1,
    /// The process exited without dropping the profiler, and the `atexit`
    /// handler wrote the profile.
    AtExit = 2,
    /// Recording was stopped through the engine directly.
    Explicit = 3,
    /// This process is a `fork` child; recording stopped because the profile
    /// belongs to the parent.
    ForkedChild = 4,
}

impl Shutdown {
    fn from_u8(raw: u8) -> Shutdown {
        match raw {
            1 => Shutdown::Dropped,
            2 => Shutdown::AtExit,
            3 => Shutdown::Explicit,
            4 => Shutdown::ForkedChild,
            _ => Shutdown::Running,
        }
    }

    /// The name this appears under in a profile.
    pub fn as_str(self) -> &'static str {
        match self {
            Shutdown::Running => "running",
            Shutdown::Dropped => "drop",
            Shutdown::AtExit => "atexit",
            Shutdown::Explicit => "explicit",
            Shutdown::ForkedChild => "forked-child",
        }
    }
}

impl fmt::Display for Shutdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A change to apply to one program point, to the global totals, and to the
/// rows for the thread and region it happened on.
///
/// Bundling them is what keeps the three consistent: they are applied together,
/// under one acquisition of the gate.
///
/// **Deliberately not `Default`.** Every construction site spreads a base, and
/// with a default one, a new path that forgot `site` would move the counters
/// and no row — silently, caught only by an aggregate sum a validator reads.
/// [`Delta::at`] makes the attribution the thing you cannot leave out, so
/// forgetting it is a compile error rather than a missing row.
#[derive(Clone, Copy, Debug)]
struct Delta {
    /// Change in live bytes. Negative for frees and shrinking reallocations.
    curr_bytes: i64,
    /// Change in live blocks.
    curr_blocks: i64,
    /// Bytes to add to the cumulative total, which never decreases.
    total_bytes: u64,
    /// Blocks to add to the cumulative total.
    total_blocks: u64,
    /// Lifetime of a block that just died, in clock units.
    lifetime: u64,
    /// Who did it, and what for.
    ///
    /// On the delta rather than passed alongside it because the attribution has
    /// to be applied in the *same* critical section as the counters — see
    /// [`Engine::attribute`] — and every gated path already carries the delta
    /// through. A separate parameter would be one more thing five private
    /// functions could forget to pass on.
    site: Site,
}

impl Delta {
    /// A change that moves nothing, attributed to `site`.
    ///
    /// The base every construction site spreads. See the type documentation for
    /// why there is no `Default`.
    const fn at(site: Site) -> Delta {
        Delta {
            curr_bytes: 0,
            curr_blocks: 0,
            total_bytes: 0,
            total_blocks: 0,
            lifetime: 0,
            site,
        }
    }
}

/// The result of an end-of-run flush.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flush {
    /// The global counters, read in the same window as the per-point ones.
    pub stats: GlobalStats,
    /// What the program asked for, read in the same window.
    ///
    /// In the window rather than after it because a profile cross-checks the
    /// two: every observed request either counted toward `total_blocks` or was
    /// dropped. Read afterwards, in a second unsynchronised acquisition, the
    /// histograms would also contain every request that landed *between* the
    /// two reads, and the check would fail on a correct profiler — which is
    /// what it did, on the probe's concurrent-shutdown row.
    ///
    /// This narrows the window rather than closing it. A shape is counted at
    /// the top of [`Engine::record_alloc`] and the block counters move at the
    /// bottom, under the gate, so a thread caught between the two has its shape
    /// counted and its block not. Closing that would mean counting the shape
    /// *inside* the gated region, which is putting work in the one place this
    /// crate spends its effort keeping empty in order to make exact an equality
    /// only a validator reads.
    pub shapes: ShapeStats,
    /// Whether exclusive access was obtained.
    ///
    /// When `false`, an event may have landed between the per-point snapshot
    /// and the global one, so the two need not agree. The profile records this
    /// rather than presenting a possibly inconsistent snapshot as exact.
    pub exclusive: bool,
    /// The most recent restart of the counts, if there has been one, read in
    /// the same window: it says what the totals beside it cover.
    pub reset: Option<Reset>,
}

/// The most recent restart of a run's counts, and what the run held then.
///
/// A run whose counts were restarted with
/// [`Profiler::reset`](crate::Profiler::reset) reports totals, peaks and
/// lifetimes for the window since, while what was live at the restart stays
/// live and stays counted as live. A profile carries this so that a reader can
/// tell that window from the whole run, which nothing in the numbers
/// themselves would say. See [`Engine::reset`] for every counter's rule.
///
/// `#[non_exhaustive]`, like the other readings a profile is written from.
/// Built outside this crate as `Reset::default()` and then assigned to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Reset {
    /// Restarts so far. At least one wherever a `Reset` is reported.
    pub count: u64,
    /// The clock reading at the most recent restart, in the run's time source.
    ///
    /// On the same axis as every other instant in the profile: a restart does
    /// not move the clock's zero, so the window a profile describes runs from
    /// here to its time at end.
    pub at: u64,
    /// Bytes live at the restart, and so carried into the window: allocated
    /// before it, and counted as live and toward its peaks but not toward its
    /// totals.
    pub carried_bytes: u64,
    /// Blocks live at the restart. See [`Reset::carried_bytes`].
    pub carried_blocks: u64,
    /// Blocks the live-block table had turned away by the restart.
    ///
    /// [`GlobalStats::dropped_blocks`] is not restarted, because a block the
    /// table could not track may still be live, and the live figures carry
    /// across a restart with that block missing from them. This is the part of
    /// that count that predates the window, which is what a reader needs in
    /// order to hold the window's own requests against its own totals.
    pub dropped_blocks: u64,
    /// The peak epoch the restart left the run at.
    ///
    /// A restart is a peak: live bytes at that instant are the most the window
    /// has held, so it moves the epoch like any other. [`GlobalStats::epoch`]
    /// less this is how many times the peak has moved since.
    pub epoch: u64,
}

/// Why a run's counts could not be restarted.
///
/// Every variant leaves the counts exactly as they were. A restart is applied
/// whole or not at all: half a restart would be a profile whose totals and
/// peaks describe two different windows, with nothing in it to say so.
///
/// `#[non_exhaustive]`: the reasons a run cannot be restarted are not a closed
/// set, and [`StatsError`](crate::StatsError) gained one after it shipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResetError {
    /// Recording has stopped.
    ///
    /// A run stops when it is stopped explicitly or when the process starts to
    /// exit, and its profile has then been, or is being, written from its final
    /// counts, which [`HeapStats::get`](crate::HeapStats::get) keeps answering
    /// with. Restarting them would change numbers a reader has already been
    /// given.
    ///
    /// An engine that never started answers this too, having nothing to
    /// restart. No [`Profiler`](crate::Profiler) can see that case, because one
    /// exists only once recording has begun, so through it this always means
    /// stopped. It is not [`StatsError::NotRecording`](crate::StatsError::NotRecording),
    /// which is the never-started case and treats a stopped run as readable.
    Stopped,
    /// This process is a `fork` child of a profiled parent.
    ///
    /// The counters came across the `fork` and describe the parent's run, as
    /// [`StatsError::ForkedChild`](crate::StatsError::ForkedChild) says.
    ForkedChild,
    /// The profiler reported an internal failure and stopped recording.
    ///
    /// The failure was in what the counters hold, and a restart keeps what is
    /// live: it would carry the damage into a window that looked clean.
    Poisoned,
    /// The calling thread could not enter the profiler.
    ///
    /// Either it is already inside it (this was called from a signal handler
    /// that interrupted an allocation, or from a `Drop` running under the
    /// allocator) or the profiler's table of threads is full, so this thread
    /// has no slot to enter with. A restart takes locks an interrupted
    /// allocation may hold, and on Apple platforms taking one twice kills the
    /// process rather than deadlocking it.
    ///
    /// One variant for both because the profiler's entry point answers one
    /// question, whether this thread may enter now, on the allocator's hot
    /// path; telling the two apart would add a branch there for the sake of an
    /// error message here. What the two have in common is the remedy: call
    /// again from ordinary code on a thread the profiler already knows, which
    /// any thread that has allocated is.
    CannotEnter,
    /// Other threads kept the profiler busy past the deadline.
    ///
    /// A restart needs a quiet point, when no thread is midway through moving
    /// a counter or a live-block entry, and waits for one for as long as a
    /// profile written at shutdown would. A thread stopped inside the
    /// profiler, under a debugger say, can hold it off for longer than that.
    Busy,
}

impl fmt::Display for ResetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResetError::Stopped => write!(
                f,
                "the heapscope profiler has stopped recording, explicitly or \
                 because the process is exiting, so its counts are final and \
                 there is nothing to restart"
            ),
            ResetError::ForkedChild => write!(
                f,
                "this process is a fork child, and the counters it inherited \
                 belong to the parent's run"
            ),
            ResetError::Poisoned => write!(
                f,
                "the profiler reported an internal failure and stopped \
                 recording; restarting would carry its live counts forward"
            ),
            ResetError::CannotEnter => write!(
                f,
                "this thread could not enter the profiler to restart its counts: \
                 it is already inside it (a signal handler, or a Drop running \
                 under the allocator), or its table of threads is full"
            ),
            ResetError::Busy => write!(
                f,
                "other threads kept the profiler busy for longer than {} seconds; \
                 nothing was restarted",
                Engine::FLUSH_TIMEOUT.as_secs()
            ),
        }
    }
}

impl std::error::Error for ResetError {}

/// What a run was configured to do, as it was actually applied.
///
/// Read back from the engine rather than from the builder that set it, so a
/// profile reports the settings the run had rather than the ones it was asked
/// for. The two differ wherever a request is clamped.
///
/// `#[non_exhaustive]`: sampling metadata joins this in M6 and the native format
/// adds more after that. Adding a field to a struct anyone can build as a
/// literal is a breaking change, and this diff is its own evidence — adding
/// `settings` to [`Snapshot`](crate::Snapshot) is what forced two other files to
/// change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Settings {
    /// What the run counts.
    pub mode: Mode,
    /// Frames kept per capture.
    pub max_depth: usize,
    /// Ceiling on simultaneously live tracked blocks.
    pub max_live_blocks: usize,
    /// Whether a rendering nobody chose explicitly drops the shared frames.
    pub trim_frames: bool,
    /// Mean bytes between sample points, or `None` for an exact run.
    ///
    /// See [`sampler`](super::sampler). A profile carries this because every
    /// number in a sampled one is an estimate, and a reader cannot tell that
    /// from the numbers themselves.
    pub sampling: Option<NonZeroU64>,
}

impl Default for Settings {
    /// What a profiler started with no configuration runs with.
    fn default() -> Self {
        Self {
            mode: Mode::Heap,
            max_depth: crate::CAPTURE_DEPTH,
            max_live_blocks: super::live::DEFAULT_MAX_LIVE_BLOCKS,
            trim_frames: true,
            sampling: None,
        }
    }
}

/// Snapshot of the engine's global state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlobalStats {
    /// Live bytes now.
    pub curr_bytes: u64,
    /// Live blocks now.
    pub curr_blocks: u64,
    /// Greatest live bytes reached since the run started or its counts were
    /// last restarted. DHAT's `gmax`.
    pub max_bytes: u64,
    /// Live blocks at the moment of that peak.
    pub max_blocks: u64,
    /// Bytes allocated since the run started or its counts were last
    /// restarted.
    pub total_bytes: u64,
    /// Blocks allocated since the run started or its counts were last
    /// restarted.
    pub total_blocks: u64,
    /// Clock reading at the peak. DHAT's `tg`.
    pub time_at_max: u64,
    /// The current peak epoch, which moves on every new peak and on every
    /// restart of the counts.
    ///
    /// The number of peaks recorded, in a run that was never restarted. In one
    /// that was, the peaks since the most recent restart are this less
    /// [`Reset::epoch`].
    pub epoch: u64,
    /// Allocations not recorded because the live-block table was full.
    ///
    /// Over the whole run, and not restarted with the other counts: a block the
    /// table turned away may still be live, so the live figures, which carry
    /// across a restart, may be short of it. See [`Reset::dropped_blocks`].
    pub dropped_blocks: u64,
    /// Events reported to a run that does not count them.
    ///
    /// [`event`](fn@crate::event) during a heap run, or an allocation-free mode
    /// asked for the other kind. Counted rather than ignored, because the
    /// symptom otherwise is an empty profile with nothing in it to say the calls
    /// were made and refused.
    pub refused_events: u64,
}

/// The recording engine.
///
/// Const-initializable so it can be a plain `static`: the shim is live before
/// `main`, so nothing it reaches may require lazy initialization.
#[derive(Debug)]
pub struct Engine {
    state: AtomicU8,
    time_source: AtomicU8,

    gate: Gate,
    curr_bytes: AtomicU64,
    curr_blocks: AtomicU64,
    max_bytes: AtomicU64,
    max_blocks: AtomicU64,
    total_bytes: AtomicU64,
    total_blocks: AtomicU64,
    time_at_max: AtomicU64,
    /// Bumped on every new peak. See the module documentation.
    epoch: AtomicU64,
    dropped_blocks: AtomicU64,
    refused_events: AtomicU64,
    /// Which path stopped recording. See [`Shutdown`].
    shutdown: AtomicU8,
    /// Whether `fork_prepare` acquired the locks, so `fork_parent` releases
    /// exactly what was taken. Touched only from the `fork` handlers, which run
    /// serialised around a `fork` on one thread.
    fork_locks_held: AtomicBool,

    clock: Clock,
    arena: Arena,
    pps: PpTable,
    live: LiveBlocks,

    /// What the program asked for, beyond a number of bytes.
    ///
    /// Everything here is a relaxed increment on a word chosen by the request
    /// itself, so a program allocating a spread of sizes spreads the traffic
    /// rather than concentrating it the way `total_bytes` does. Nothing on this
    /// path is read back until a profile is written.
    shapes: Shapes,

    /// Who allocated, and what for.
    ///
    /// Both tables are read on the allocator path and written there only when a
    /// thread records its first event; a region is interned by the program at a
    /// phase boundary, never by the shim. See [`super::site`].
    ///
    /// # One engine per process
    ///
    /// A thread's row id is cached in its guard slot, and the guard's slot table
    /// is process-wide while these tables are per-engine. So an id means "row *n*
    /// of the engine this process has" — which is exact for the shipped crate,
    /// where [`crate::engine`] is the only one, and ambiguous for a test that
    /// builds several and records into more than one from the same thread. Such
    /// a test sees attribution silently go missing (the row is unpublished in
    /// the second table) rather than land on a wrong thread, but neither is a
    /// state to reason from: a test that cares about attribution should drive
    /// one engine.
    threads: Threads,
    regions: Regions,

    /// Frames to keep per capture, never more than the shim's buffer holds.
    ///
    /// Read once per recorded allocation: a relaxed load of a value written at
    /// most once per process, so the line sits in every core's cache in shared
    /// state and generates no coherence traffic. The same shape as
    /// [`Engine::serialized`] and [`crate::unwind::strategy`]. Unmeasured —
    /// no benchmark covers the shim end to end — so that is a reasoned bound
    /// rather than a number.
    max_depth: AtomicUsize,

    /// Whether the renderings nobody chose explicitly drop the frames every
    /// stack shares. Read when a profile is written, never on the hot path.
    trim_frames: AtomicBool,

    /// What this run counts. See [`Mode`].
    ///
    /// Read on the allocator path, immediately after the state, and paid for by
    /// the same argument as [`Engine::max_depth`]: written at most once per
    /// process, so the line sits in every core's cache in shared state and
    /// generates no coherence traffic. Folding it into `state` would save the
    /// load at the cost of a lifecycle enum that also encodes a setting, which
    /// is a trade the hot path does not need made for it.
    mode: AtomicU8,

    /// Forces every event through the exclusive path. See
    /// [`Engine::serialize_for_testing`].
    ///
    /// Not `cfg`-gated, because integration tests link the library compiled
    /// *without* `cfg(test)`, and a Cargo feature would mean the test only runs
    /// when someone remembers to pass a flag — which is close to not having it.
    /// The cost is one relaxed load of a field that is written at most once per
    /// process and never again, so it sits in every core's cache in shared state
    /// and generates no coherence traffic.
    serialized: AtomicBool,

    /// Mean bytes between sample points, or zero for an exact run.
    ///
    /// Read on the allocator path *before* the capture, which is the whole point
    /// of it: a run that skips an allocation skips the stack walk that dominates
    /// what recording costs. Same shape and same argument as
    /// [`Engine::max_depth`] — written at most once per process, so the load
    /// generates no coherence traffic.
    sampling: AtomicU64,

    /// Twice the number of restarts of the counts, plus one while a restart is
    /// being applied: a sequence lock around [`Engine::reset`].
    ///
    /// A reading of the counters outside the gate, which is what
    /// [`HeapStats::get`](crate::HeapStats::get) is, cannot otherwise say
    /// which window it read. [`Engine::read_window`] is the reader's half, and
    /// the comments there and in [`Engine::reset`] give the orderings.
    reset_sequence: AtomicU64,
    /// The most recent restart of the counts, field by field. See [`Reset`].
    ///
    /// Written only by [`Engine::reset`], with the peak gate held exclusively,
    /// and read together only in a flush window, which holds it too: so a
    /// profile never pairs half of one restart with half of another. Nothing on
    /// the allocator path reads them.
    reset_at: AtomicU64,
    reset_epoch: AtomicU64,
    carried_bytes: AtomicU64,
    carried_blocks: AtomicU64,
    dropped_at_reset: AtomicU64,
}

impl Engine {
    /// Creates an idle engine with default limits.
    pub const fn new() -> Self {
        Self::with_limits(
            super::pp::DEFAULT_MAX_PROGRAM_POINTS,
            super::live::DEFAULT_MAX_LIVE_BLOCKS,
        )
    }

    /// Creates an idle engine with explicit table ceilings.
    pub const fn with_limits(max_program_points: usize, max_live_blocks: usize) -> Self {
        Self {
            state: AtomicU8::new(State::Idle as u8),
            time_source: AtomicU8::new(TimeSource::Events as u8),
            gate: Gate::new(),
            curr_bytes: AtomicU64::new(0),
            curr_blocks: AtomicU64::new(0),
            max_bytes: AtomicU64::new(0),
            max_blocks: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            total_blocks: AtomicU64::new(0),
            time_at_max: AtomicU64::new(0),
            epoch: AtomicU64::new(0),
            dropped_blocks: AtomicU64::new(0),
            refused_events: AtomicU64::new(0),
            shutdown: AtomicU8::new(Shutdown::Running as u8),
            fork_locks_held: AtomicBool::new(false),
            clock: Clock::new(),
            arena: Arena::new(),
            pps: PpTable::with_capacity(max_program_points),
            live: LiveBlocks::with_capacity(max_live_blocks),
            shapes: Shapes::new(),
            threads: Threads::new(),
            regions: Regions::new(),
            max_depth: AtomicUsize::new(crate::CAPTURE_DEPTH),
            trim_frames: AtomicBool::new(true),
            mode: AtomicU8::new(Mode::Heap as u8),
            serialized: AtomicBool::new(false),
            sampling: AtomicU64::new(0),
            reset_sequence: AtomicU64::new(0),
            reset_at: AtomicU64::new(0),
            reset_epoch: AtomicU64::new(0),
            carried_bytes: AtomicU64::new(0),
            carried_blocks: AtomicU64::new(0),
            dropped_at_reset: AtomicU64::new(0),
        }
    }

    /// Frames kept per capture, or `usize::MAX` for as many as the buffer holds.
    #[inline(always)]
    pub fn max_depth(&self) -> usize {
        self.max_depth.load(Ordering::Relaxed)
    }

    /// Whether a rendering nobody chose explicitly drops the shared frames.
    pub fn trim_frames(&self) -> bool {
        self.trim_frames.load(Ordering::Relaxed)
    }

    /// The ceiling on simultaneously live tracked blocks.
    pub fn max_live_blocks(&self) -> usize {
        self.live.max_blocks()
    }

    /// What this run counts.
    #[inline(always)]
    pub fn mode(&self) -> Mode {
        Mode::from_u8(self.mode.load(Ordering::Relaxed))
    }

    /// Whether the allocator shim should record what it sees.
    ///
    /// The shim's whole condition, in one call, because the two halves are only
    /// ever meaningful together: a run recording ad hoc events is `Running`, and
    /// an allocation during it must reach the inner allocator and nothing else.
    #[inline(always)]
    pub fn records_allocations(&self) -> bool {
        self.is_running() && self.mode().records_allocations()
    }

    /// The mean bytes between sample points, or `None` for an exact run.
    ///
    /// One relaxed load, on the allocator path ahead of the capture.
    #[inline(always)]
    pub fn sampling(&self) -> Option<u64> {
        match self.sampling.load(Ordering::Relaxed) {
            0 => None,
            interval => Some(interval),
        }
    }

    /// Whether this run records estimates rather than exact counts.
    ///
    /// The question every reader of a number from this engine has to ask, given
    /// its own name so that no caller has to remember that "sampling is on" and
    /// "these figures are estimates" are the same thing.
    #[inline]
    pub fn is_sampled(&self) -> bool {
        self.sampling.load(Ordering::Relaxed) != 0
    }

    /// Everything [`Engine::configure`] was given, as it stands.
    pub fn settings(&self) -> Settings {
        Settings {
            mode: self.mode(),
            max_depth: self.max_depth(),
            max_live_blocks: self.max_live_blocks(),
            trim_frames: self.trim_frames(),
            sampling: NonZeroU64::new(self.sampling.load(Ordering::Relaxed)),
        }
    }

    /// Applies one run's settings.
    ///
    /// `pub(crate)`, and that is load-bearing rather than tidiness: a depth or a
    /// ceiling changing mid-run would make one profile describe two
    /// configurations, with nothing in the file to say where the change fell.
    /// While this was reachable from outside, `heapscope::engine().configure(..)`
    /// from an integration test did exactly that and took effect
    /// **\[measured\]**. The `debug_assert` states the same thing to anyone
    /// adding a caller inside the crate.
    ///
    /// Called from inside [`Engine::start`]'s configuration window, where the
    /// engine is `Starting` and the shim refuses every event. What that window
    /// buys is that no *new* recording can begin against half-applied settings;
    /// a thread already inside the shim can still finish a capture under the
    /// previous ones, exactly as it can for the capture strategy.
    pub(crate) fn configure(&self, requested: Settings) {
        debug_assert_eq!(
            self.state(),
            State::Starting,
            "settings are fixed for the life of a run"
        );
        let Settings {
            mode,
            max_depth,
            max_live_blocks,
            trim_frames,
            sampling,
        } = requested;
        // Clamped rather than rejected, because both ends of the range mean
        // "as much as there is" rather than a contradiction the caller has to
        // resolve: the shim's buffer is a fixed array, and a program point with
        // no frames is not a program point. What the run ended up with is read
        // back out of here into every profile, so the clamp is never silent.
        let max_depth = max_depth.clamp(1, crate::CAPTURE_DEPTH);
        self.mode.store(mode as u8, Ordering::Relaxed);
        self.max_depth.store(max_depth, Ordering::Relaxed);
        self.trim_frames.store(trim_frames, Ordering::Relaxed);
        self.live.set_max_blocks(max_live_blocks);
        self.sampling
            .store(sampling.map_or(0, NonZeroU64::get), Ordering::Relaxed);
        // Each run starts the seed sequence over, so that two runs of one
        // single-threaded workload sample the same allocations. A process runs
        // one profiler, but a test binary runs many, and reproducibility across
        // runs is exactly what `TimeSource::Events` promises.
        super::sampler::reset_sequence();
    }

    /// Forces every event through the exclusive path, giving the engine a single
    /// total order over its events.
    ///
    /// # What this is for
    ///
    /// Differential testing against a serial model is straightforward for a
    /// single-threaded trace and impossible for a concurrent one: the engine's
    /// linearization point sits *inside* the gate, so a reference tracker
    /// wrapped around `record_alloc`/`record_free` takes its own, different
    /// linearization point. Two threads doing `alloc(100)` and `free(100)` can
    /// be ordered A-then-B by the gate and B-then-A by the reference, producing
    /// legitimately different peaks. No amount of test effort fixes that; it is
    /// the same non-linearizability the gate exists to remove, reappearing one
    /// layer up.
    ///
    /// Serializing gives the two a shared order. Every event then takes the gate
    /// exclusively, so the order in which threads acquire it *is* the order the
    /// reference replays, and every counter can be compared exactly under real
    /// concurrency — real threads, real interleaving, real contention on the
    /// program-point and live-block shards.
    ///
    /// # What it does not cover
    ///
    /// The shared path's compare-exchange, which is the thing being skipped.
    /// That is covered separately by the summation invariants, which hold under
    /// any interleaving.
    ///
    /// Nothing in the public API reaches this; a profiler never enables it.
    #[doc(hidden)]
    pub fn serialize_for_testing(&self) {
        self.serialized.store(true, Ordering::Release);
    }

    #[inline(always)]
    fn is_serialized(&self) -> bool {
        self.serialized.load(Ordering::Relaxed)
    }

    /// Begins recording.
    ///
    /// Returns `false` if the engine is not idle, which is how a second
    /// concurrent profiler is refused.
    pub fn start(&self, time_source: TimeSource, configure: impl FnOnce()) -> bool {
        // Claimed in two steps. The first takes the engine out of `Idle` so no
        // second profiler can claim it; the second publishes `Running`, which is
        // what the shim checks. Everything between happens with recording off,
        // so a setting can never be observed before it has been applied.
        if self
            .state
            .compare_exchange(
                State::Idle as u8,
                State::Starting as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }
        self.time_source.store(time_source as u8, Ordering::Relaxed);
        self.clock.start();
        configure();
        self.state.store(State::Running as u8, Ordering::Release);
        true
    }

    /// Stops recording, then waits briefly for events already in flight.
    ///
    /// The state flips **first**, so the writer's own allocations — buffers,
    /// formatted strings — stay out of the profile it is writing. Only then does
    /// it wait, and only for a bounded time: PLAN.md section 4.6 requires the
    /// shutdown path to degrade to partial output rather than hang the process
    /// at `exit`, because other threads are still live at that point.
    ///
    /// Acquiring the gate exclusively *is* the drain — it cannot be granted
    /// until every shared holder has finished.
    ///
    /// `cause` is recorded and emitted in the profile. The **first** call wins:
    /// a program that exits after dropping its profiler runs the `atexit`
    /// handler too, and the profile it produced was written by the drop.
    pub fn stop(&self, cause: Shutdown) {
        let _ = self.shutdown.compare_exchange(
            Shutdown::Running as u8,
            cause as u8,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );
        self.state.store(State::Finished as u8, Ordering::Release);
        if self.gate.write_for(Self::FLUSH_TIMEOUT).is_none() {
            super::diagnostic::report(
                "some allocation events were still in flight at shutdown; \
                 the profile may be missing them",
            );
        }
    }

    /// Acquires every lock the engine owns, in the order [`super::order`] fixes.
    ///
    /// This is the `pthread_atfork` **prepare** handler's work. `fork` copies
    /// only the calling thread, so a lock held at the instant of the call is
    /// held in the child by a thread that does not exist — and the data it was
    /// midway through updating is copied in that half-updated state. Acquiring
    /// everything first means the child inherits structures no one was touching.
    ///
    /// Acquiring in the global lock order is what keeps this from deadlocking
    /// against the recording paths: no path holds a lock at a deeper level while
    /// waiting for a shallower one, so every holder this handler waits on is
    /// making progress toward release.
    ///
    /// # Giving up
    ///
    /// The gate is acquired with a deadline, and if it is not granted this
    /// handler releases what it took and returns having acquired nothing. A
    /// prepare handler runs *inside* `fork()`, so one that waits forever hangs
    /// `fork` — and `Command::spawn` — with no diagnostic and no timeout the
    /// caller can set. What is lost by giving up is the guarantee that the child
    /// inherits tables nobody was midway through updating; the child resets the
    /// locks either way, and does not read the tables.
    ///
    /// # Safety
    ///
    /// A matching [`Engine::fork_parent`] must run on the same thread, or
    /// [`Engine::fork_child`] must reset the locks.
    pub unsafe fn fork_prepare(&self) {
        // SAFETY: each call is paired by `fork_parent` or discharged by
        // `fork_child`, per this function's own contract.
        unsafe { self.live.lock_all_for_fork() };

        // SAFETY: as above.
        if !unsafe { self.gate.lock_for_fork(Self::FLUSH_TIMEOUT) } {
            // SAFETY: acquired immediately above and released here, on the same
            // thread, having taken nothing deeper.
            unsafe { self.live.unlock_all_for_fork() };
            self.fork_locks_held.store(false, Ordering::Release);
            super::diagnostic::report(
                "could not quiesce the profiler before a fork; the child is \
                 safe, but its inherited tables may be mid-update",
            );
            return;
        }

        // SAFETY: as above. These are leaf locks with no drain loop, held for a
        // bounded number of instructions, so unlike the gate they cannot be kept
        // busy by a stream of arrivals.
        unsafe {
            self.pps.lock_all_for_fork();
            self.regions.lock_for_fork();
            self.arena.lock_for_fork();
        }
        self.fork_locks_held.store(true, Ordering::Release);
    }

    /// Releases what [`Engine::fork_prepare`] acquired, in the parent.
    ///
    /// Runs even when `fork` **failed**: both glibc and Darwin call the parent
    /// handlers on the error path, which is what keeps a failed `fork` from
    /// leaving this process holding every lock forever **[measured on both]**.
    ///
    /// # Safety
    ///
    /// Must run on the thread that ran [`Engine::fork_prepare`].
    pub unsafe fn fork_parent(&self) {
        if !self.fork_locks_held.swap(false, Ordering::AcqRel) {
            // `fork_prepare` gave up and already released what it had.
            return;
        }
        // SAFETY: delegated to the caller's obligation. Released in the reverse
        // of the acquisition order.
        unsafe {
            self.arena.unlock_for_fork();
            self.regions.unlock_for_fork();
            self.pps.unlock_all_for_fork();
            self.gate.unlock_for_fork();
            self.live.unlock_all_for_fork();
        }
    }

    /// Resets every lock and stops recording, in the child.
    ///
    /// The locks are overwritten rather than released, because releasing them
    /// is not always correct. [`Engine::fork_prepare`] gives up if it cannot
    /// quiesce the profiler in time, in which case a lock may be held by a
    /// thread that `fork` did not copy and that this process cannot release on
    /// its behalf. Overwriting is right in both cases; `raw_unlock` is right
    /// only in one, and would be unsound in the other.
    ///
    /// This is also the load-bearing half of the arrangement. A child recovers
    /// from an inherited lock *here*, not in `prepare` — which is why deleting
    /// this function's body wedges a child that takes a snapshot, while deleting
    /// `prepare`'s does not.
    ///
    /// Recording stops because the profile belongs to the parent. The child
    /// inherits the parent's counters, its live blocks, and its output path;
    /// letting it carry on would produce a second profile that double-counts
    /// everything before the `fork` and would race the parent for the file.
    ///
    /// # Safety
    ///
    /// Call only from a `pthread_atfork` child handler, where the process is
    /// single-threaded by definition.
    pub unsafe fn fork_child(&self) {
        // SAFETY: delegated to the caller's single-threadedness obligation.
        unsafe {
            self.arena.reinit_after_fork();
            self.regions.reinit_after_fork();
            self.pps.reinit_after_fork();
            self.gate.reinit_after_fork();
            self.live.reinit_after_fork();
            super::guard::reinit_after_fork();
            super::order::reinit_after_fork();
        }

        // Only a *running* engine becomes a forked child. An idle one has
        // recorded nothing worth disowning, and a child of an unprofiled process
        // should be as free to start a profiler as any other process is.
        if self
            .state
            .compare_exchange(
                State::Running as u8,
                State::ForkedChild as u8,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            self.shutdown
                .store(Shutdown::ForkedChild as u8, Ordering::Relaxed);
        }
    }

    /// Which path stopped recording, or [`Shutdown::Running`] if none has.
    pub fn shutdown(&self) -> Shutdown {
        Shutdown::from_u8(self.shutdown.load(Ordering::Relaxed))
    }

    /// Current lifecycle state.
    #[inline(always)]
    pub fn state(&self) -> State {
        State::from_u8(self.state.load(Ordering::Relaxed))
    }

    /// Whether events should be recorded right now.
    ///
    /// Poisoning stops recording, which is the "stop recording" half of
    /// PLAN.md section 4.6's poison-and-degrade rule. Without this check the
    /// flag was set and reported but nothing acted on it, so a profiler that
    /// had detected its own corruption carried on producing numbers from it.
    ///
    /// # Why the load is `Acquire`
    ///
    /// It pairs with the `Release` store in [`Engine::start`], and that pairing
    /// is what makes every setting applied in the configuration window visible
    /// to a thread that observes `Running`. It was `Relaxed`, which gave no
    /// happens-before edge at all — benign while the settings were a depth and
    /// a ceiling, where a stale read costs one event captured under the previous
    /// value, and no longer benign now that [`Engine::mode`] decides *whether*
    /// the shim records: a thread already inside the shim could observe
    /// `Running` alongside a stale `Heap` and record an allocation into a run
    /// configured to count something else. The documented guarantee on
    /// [`Engine::configure`] is that no new recording begins against
    /// half-applied settings, and this is the load that makes it one.
    ///
    /// The cost is one acquire load per allocator entry, on a word written at
    /// most twice per process.
    #[inline(always)]
    pub fn is_running(&self) -> bool {
        self.state.load(Ordering::Acquire) == State::Running as u8
            && !super::diagnostic::is_poisoned()
    }

    /// The active time base.
    #[inline(always)]
    pub fn time_source(&self) -> TimeSource {
        match self.time_source.load(Ordering::Relaxed) {
            1 => TimeSource::Monotonic,
            _ => TimeSource::Events,
        }
    }

    /// The arena backing all profiler state.
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// The program-point table.
    pub fn program_points(&self) -> &PpTable {
        &self.pps
    }

    /// The live-block table.
    pub fn live_blocks(&self) -> &LiveBlocks {
        &self.live
    }

    /// The clock.
    pub fn clock(&self) -> &Clock {
        &self.clock
    }

    /// Records an allocation of `shape` at `address`, attributed to `frames`.
    ///
    /// The live-block entry is created *before* the counters move. If the table
    /// is full the event is dropped entirely rather than half-recorded: counting
    /// an allocation whose free can never be attributed would leave live bytes
    /// permanently inflated, which is a wrong number rather than a missing one.
    ///
    /// The shape has already been counted, by [`Engine::observe`], which is what
    /// the caller asked in order to learn whether to capture a stack at all. It
    /// is counted there rather than here for the reason it was always counted
    /// first: a request this profiler did not record still happened, and the
    /// histograms describe what the program asked for. Under sampling that
    /// becomes load-bearing rather than incidental —
    /// [`ShapeStats::observed_blocks`](super::shape::ShapeStats::observed_blocks)
    /// stays an exact count of requests while `total_blocks` becomes an estimate,
    /// so a profile carries both the truth and the estimate of the same quantity
    /// and a reader can see how well sampling did.
    pub fn record_alloc(&self, guard: &Guard, address: usize, shape: Shape, frames: &[usize]) {
        let site = self.site(guard);
        let pp = self.pps.intern(&self.arena, frames).id();
        let now = self.clock.tick(self.time_source());

        if !self.live.insert(
            &self.arena,
            address,
            LiveBlock {
                birth: now,
                pp,
                site,
            },
        ) {
            self.dropped_blocks.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let (bytes, blocks) = self.weights(shape.size);
        self.apply(
            pp,
            Delta {
                curr_bytes: bytes as i64,
                curr_blocks: blocks as i64,
                total_bytes: bytes,
                total_blocks: blocks,
                ..Delta::at(site)
            },
        );
    }

    /// Counts what the program asked for, and answers whether this request is
    /// one the run will record.
    ///
    /// Asked by the shim *before* it captures a stack, because the capture is
    /// what sampling exists to skip: a recorded allocation costs about 123 ns
    /// against 33 ns unprofiled, and `benches/unwind.rs` puts the walk at most of
    /// the difference.
    ///
    /// Two things happen here and they are deliberately not separable. Counting
    /// the shape has to happen for every request whether or not it is sampled, or
    /// the histograms stop describing the program; asking the sampler has to
    /// happen exactly once per request, or the countdown advances twice for one
    /// allocation. A caller given two functions can get either wrong, and one of
    /// the two mistakes is silent.
    ///
    /// Returns `true` for every request when sampling is off, which is the
    /// default.
    #[inline]
    pub fn observe(&self, guard: &Guard, shape: Shape) -> bool {
        self.shapes.record(shape);
        match self.sampling() {
            None => true,
            Some(interval) => guard.admits(shape.size, interval),
        }
    }

    /// Counts a reallocation, and answers whether it wants a stack.
    ///
    /// The counterpart to [`Engine::observe`] for the resize path, which counts
    /// two things: the copy the allocator had to do, and the shape of the block
    /// that came out of it.
    ///
    /// `tracked` says whether the old block was one this run had recorded. A
    /// reallocation of a tracked block is not a new sampling decision — the block
    /// is already in the profile, and dropping it here would leave its live bytes
    /// standing until the run ended. Only a resize of a block this run never saw
    /// is a fresh allocation, and only that asks the sampler.
    #[inline]
    pub fn observe_realloc(&self, guard: &Guard, realloc: &Realloc, tracked: bool) -> bool {
        self.shapes.record_realloc(realloc);
        self.shapes.record(realloc.new);
        if tracked {
            return true;
        }
        match self.sampling() {
            None => true,
            Some(interval) => guard.admits(realloc.new.size, interval),
        }
    }

    /// What one recorded allocation of `size` bytes stands for, as `(bytes,
    /// blocks)`.
    ///
    /// `(size, 1)` on an exact run. Under sampling both are scaled by the
    /// reciprocal of the probability that this size would have been sampled, and
    /// the pair is rounded here so that the allocation and the free of one block
    /// agree to the byte. See [`sampler::scale`](super::sampler::scale).
    #[inline]
    fn weights(&self, size: usize) -> (u64, u64) {
        let interval = self.sampling();
        (
            super::sampler::weighted_bytes(size, interval),
            super::sampler::weighted_blocks(size, interval),
        )
    }

    /// A lifetime as it should be counted, given the blocks it stands for.
    ///
    /// A sampled block that lived 40 µs is evidence of `blocks` blocks that lived
    /// about that long, not of one. Without this the average-lifetime column —
    /// `tl` divided by the block count — would be deflated by exactly the
    /// sampling scale, which is the column a reader uses to find short-lived
    /// churn.
    #[inline]
    fn weighted_lifetime(lifetime: u64, blocks: u64) -> u64 {
        lifetime.saturating_mul(blocks)
    }

    /// Who is allocating, and what for, resolving the thread's row on first use.
    ///
    /// Two relaxed loads in the common case. The claim below happens once per
    /// thread per run, and its result — including the answer "there was no room
    /// for you" — is cached in the guard slot, so a thread never asks twice.
    #[inline]
    fn site(&self, guard: &Guard) -> Site {
        let site = guard.site();
        if site.thread.is_unclaimed() {
            return Site {
                thread: self.claim_thread(guard),
                region: site.region,
            };
        }
        // A thread whose name the platform did not have yet asks again, a few
        // times. One relaxed load and a compare in the common case; see
        // `site::NAME_ATTEMPTS` for the case it exists for, which is a thread
        // whose *first* recorded allocation is std's own name buffer.
        if self.threads.wants_name(site.thread) {
            self.name_thread(site.thread);
        }
        site
    }

    /// Gives the calling thread a row in the thread table, naming it from the
    /// platform.
    ///
    /// Cold, and `inline(never)`, because it runs once per thread and its
    /// buffer would otherwise widen every recorded allocation's stack frame.
    #[cold]
    #[inline(never)]
    fn claim_thread(&self, guard: &Guard) -> ThreadId {
        let now = self.clock.now(self.time_source());
        let id = self.threads.claim(&self.arena, current_name(), now);
        guard.set_thread(id);
        id
    }

    /// Asks the platform again for a thread's name, on a row that has none.
    ///
    /// Cold and out of line for the same reason [`Engine::claim_thread`] is: it
    /// runs a bounded number of times per thread and its buffer would otherwise
    /// widen every recorded allocation's stack frame.
    #[cold]
    #[inline(never)]
    fn name_thread(&self, id: ThreadId) {
        self.threads.name(id, current_name());
    }

    /// The per-thread rows.
    pub fn threads(&self) -> &Threads {
        &self.threads
    }

    /// The per-region rows.
    pub fn regions(&self) -> &Regions {
        &self.regions
    }

    /// Returns the row for a region named `name`, creating it if it is new.
    pub fn intern_region(&self, name: &str) -> RegionId {
        self.regions
            .intern(&self.arena, name, self.clock.now(self.time_source()))
    }

    /// Records one event of `weight` at `frames`.
    ///
    /// The counterpart to [`Engine::record_alloc`] for the modes where the
    /// program says what happened instead of the shim observing it. Nothing
    /// becomes live and nothing has a lifetime, so the event contributes to the
    /// cumulative totals only — which is exactly the subset of the per-point
    /// counters a `bklt: false` profile carries.
    ///
    /// The clock ticks, because in [`TimeSource::Events`] the clock counts
    /// recorded events and these are all of them; a run whose `te` stayed at
    /// zero would report every event as having happened at the same instant.
    ///
    /// # Why this takes a [`Guard`](super::guard::Guard)
    ///
    /// Not to use it — to require it. This reaches [`Gate::read`], and
    /// [`Gate`](super::gate::Gate) is writer-preferring: a thread that already
    /// holds a read guard and enters `read` again while a flush is waiting
    /// deadlocks against itself, because the writer waits for that thread's
    /// outer guard to drain while the thread waits on the writer's lock. The
    /// reentrancy guard is what makes that unreachable — a signal handler that
    /// interrupted this, or a `Drop` running under the shim, finds
    /// [`enter`](super::guard::enter) returning `None` — and the guard has to be
    /// held *across* this call, not merely taken before it.
    ///
    /// A review inserted `drop(guard)` between the capture and this call and no
    /// test failed, because the deadlock needs a signal to land in a window of a
    /// few instructions. Taking the proof as an argument makes the early drop a
    /// borrow-check error instead of a test nobody can write.
    ///
    /// [`Engine::record_alloc`] has the same requirement and states it the same
    /// way, which it did not always: taking the proof became unavoidable when
    /// the guard slot became where a thread's attribution row is cached.
    pub fn record_event(&self, guard: &Guard, weight: u64, frames: &[usize]) {
        let site = self.site(guard);
        let pp = self.pps.intern(&self.arena, frames).id();
        self.clock.tick(self.time_source());

        self.apply_without_peak(
            pp,
            Delta {
                total_bytes: weight,
                total_blocks: 1,
                ..Delta::at(site)
            },
        );
    }

    /// Counts an event reported to a run that does not record its kind.
    pub fn refuse_event(&self) {
        self.refused_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a free of `size` bytes at `address`.
    ///
    /// A pointer with no entry is ignored. That is the normal case for blocks
    /// allocated before profiling started, and it is why no separate pre-start
    /// set is needed.
    pub fn record_free(&self, address: usize, size: usize) {
        let Some(block) = self.live.remove(address) else {
            return;
        };
        // Reads the clock without advancing it. In `Events` mode the clock
        // counts *allocations*, so a block allocated and freed with nothing in
        // between has a lifetime of zero — which is what "zero allocation events
        // elapsed" means. Ticking here would double the clock's rate and make
        // the unit label ("events") a lie.
        let now = self.clock.now(self.time_source());

        // The block's own attribution, not the freeing thread's. A block
        // allocated on a worker and freed on the main thread must bring the
        // worker's live bytes down, or every program that hands ownership
        // across threads reports its workers as leaking.
        // The same weights the allocation used, recomputed rather than stored.
        // They are a pure function of the size and the run's interval, and the
        // shim hands the size back on this path, so a word per live block would
        // buy nothing — see `sampler::scale`. Recomputing is exact, so a balanced
        // run returns to zero live bytes rather than drifting by a rounding error
        // per block.
        let (bytes, blocks) = self.weights(size);
        self.apply(
            block.pp,
            Delta {
                curr_bytes: -(bytes as i64),
                curr_blocks: -(blocks as i64),
                // Zero for a block carried across a restart of the counts. See
                // `LiveBlock::CARRIED` for why, and for why it costs nothing.
                lifetime: Self::weighted_lifetime(block.lifetime_at(now), blocks),
                ..Delta::at(block.site)
            },
        );
    }

    /// Records a reallocation whose old live-block entry the caller has already
    /// removed.
    ///
    /// The resize is attributed to the program point that made the **original**
    /// allocation (PLAN.md decision 10.5), which is how people reason about
    /// `Vec` growth: the cost belongs to whoever created the vector, not to
    /// whichever `push` happened to trigger a resize. The block counts as newly
    /// allocated for the cumulative totals but does not change the live block
    /// count, and its birth instant is reset, matching `dhat-rs`.
    ///
    /// # Why there is no one-shot `record_realloc`
    ///
    /// There was, and it could not be used correctly. PLAN.md section 4.1
    /// requires the live-block entry to be removed *before* the inner free — but
    /// a caller cannot learn the new address without first calling the inner
    /// allocator, by which point the old address has been released and another
    /// thread may already have received it and recorded its own block there.
    /// A one-shot call is only invocable after that window has closed, so it
    /// would delete the new owner's record, leak its block from the accounting
    /// forever, and attribute the resize to the wrong program point.
    ///
    /// Splitting the operation makes the correct sequence the only expressible
    /// one: remove, call the inner allocator, then land the result here.
    ///
    /// `taken` of `None` means the block was not tracked, and the event is
    /// treated as a fresh allocation attributed to `frames` — the only
    /// attribution available.
    ///
    /// # What the reallocation itself cost
    ///
    /// Counted before anything else and regardless of whether the block was
    /// tracked, because a resize is an event with a cost of its own: the bytes
    /// the allocator had to copy are real work the program paid for, and they
    /// appear nowhere in the sizes it asked for. See
    /// [`Shapes::record_realloc`](super::shape::Shapes::record_realloc).
    ///
    /// The resulting block is *also* counted as an allocation of its new shape,
    /// once, on whichever of the two paths below it takes — matching
    /// `total_blocks`, which likewise counts a reallocation as a new block.
    pub fn record_realloc_taken(
        &self,
        guard: &Guard,
        taken: Option<LiveBlock>,
        realloc: Realloc,
        frames: &[usize],
    ) {
        let Realloc {
            old_size,
            new_address,
            new,
            ..
        } = realloc;
        let new_size = new.size;

        let Some(old) = taken else {
            self.record_alloc(guard, new_address, new, frames);
            return;
        };

        let now = self.clock.tick(self.time_source());
        // The old block's life ends here. Recording its lifetime matters because
        // the reallocation also increments the block total: counting a block
        // without its lifetime deflates the average-lifetime column at every
        // realloc-heavy site, which is exactly where a reader looks first.
        //
        // A block carried across a restart of the counts has none to record:
        // it is in none of the window's allocation counts. The block it becomes is
        // born here, so it is in them, and its lifetime will count.
        let old_lifetime = old.lifetime_at(now);

        // The block stood for `old_blocks` blocks of `old_bytes` and now stands
        // for `new_blocks` of `new_bytes`. On an exact run both block figures are
        // 1 and the difference below is the zero the counters have always seen.
        let (old_bytes, old_blocks) = self.weights(old_size);
        let (new_bytes, new_blocks) = self.weights(new_size);
        let old_lifetime = Self::weighted_lifetime(old_lifetime, old_blocks);

        // The resulting block keeps the *original* block's attribution, for
        // the reason it keeps the original program point: a `Vec` grown by
        // whichever thread happened to push belongs to whoever created it.
        if !self.live.insert(
            &self.arena,
            new_address,
            LiveBlock {
                birth: now,
                pp: old.pp,
                site: old.site,
            },
        ) {
            // The old entry is already gone, so the block is now untracked. Undo
            // its contribution to the live counters, or they would never come
            // back down.
            self.dropped_blocks.fetch_add(1, Ordering::Relaxed);
            self.apply(
                old.pp,
                Delta {
                    curr_bytes: -(old_bytes as i64),
                    curr_blocks: -(old_blocks as i64),
                    lifetime: old_lifetime,
                    ..Delta::at(old.site)
                },
            );
            return;
        }

        self.apply(
            old.pp,
            Delta {
                curr_bytes: new_bytes as i64 - old_bytes as i64,
                curr_blocks: new_blocks as i64 - old_blocks as i64,
                total_bytes: new_bytes,
                total_blocks: new_blocks,
                lifetime: old_lifetime,
                site: old.site,
            },
        );
    }

    /// Applies `delta` to the global totals, to `pp`, and to the rows for the
    /// thread and region it names, under the gate.
    #[inline]
    fn apply(&self, pp: PpId, delta: Delta) {
        // `>= 0`, not `> 0`. A same-size reallocation has a delta of exactly
        // zero, and when live bytes already sit at the maximum that event is a
        // new equal peak — which Valgrind records and which the `>=` rule says
        // to prefer over the earlier one. Routing zero-delta events down the
        // shrink path skipped the check entirely and left `tg` pointing at the
        // wrong moment.
        if self.is_serialized() {
            // One total order over every event, for differential testing.
            let _order = super::order::enter(super::order::Level::PeakGate);
            let _guard = self.gate.write();
            self.apply_locked(pp, delta);
            return;
        }

        if delta.curr_bytes >= 0 {
            // Growth may set a new peak. The shared path commits only if the
            // result is provably not one.
            //
            // The pre-check is racy on purpose. During a monotonically growing
            // phase every event peaks, so without it every event acquires the
            // gate shared, fails, and acquires it again exclusively — two
            // acquisitions where one would do, on the hottest path in the
            // profiler. Reading stale values here is harmless: a false negative
            // costs one wasted shared attempt, and a false positive costs
            // nothing at all, because `apply_exclusive` re-reads everything
            // under exclusion. The shared path is purely an optimisation, so
            // skipping it can never be wrong.
            let growth = delta.curr_bytes as u64;
            let might_peak = self.curr_bytes.load(Ordering::Relaxed).wrapping_add(growth)
                >= self.max_bytes.load(Ordering::Relaxed);

            if !might_peak && self.try_apply_shared(pp, delta) {
                return;
            }
            self.apply_exclusive(pp, delta);
        } else {
            // Shrinking cannot create a peak, so the shared path is always safe.
            let _read = super::order::enter(super::order::Level::PeakGate);
            let _guard = self.gate.read();
            self.commit(pp, delta, self.epoch.load(Ordering::Relaxed));
        }
    }

    /// Commits a change that moves no live bytes and no live blocks, so it
    /// cannot create a peak.
    ///
    /// Separate from [`Engine::apply`] by intent rather than by shape, and the
    /// distinction is load-bearing in both directions. A same-size reallocation
    /// also has a zero live-byte delta, and it *must* take the exclusive path,
    /// because a block really was allocated at a moment when live bytes may
    /// already sit at the maximum — Valgrind's `>=` rule records that as the
    /// latest equal peak. An ad hoc event allocated nothing, so treating it as a
    /// peak would move `tg` to an instant at which the heap did not change.
    ///
    /// Reading it off the delta instead would therefore be wrong for reallocs,
    /// and routing events through `apply` would be wrong for `tg` — and, since
    /// `curr_bytes + 0 >= max_bytes` holds whenever the heap is at its peak,
    /// would also send every event to the exclusive gate.
    fn apply_without_peak(&self, pp: PpId, delta: Delta) {
        debug_assert_eq!(delta.curr_bytes, 0, "this path cannot move live bytes");
        debug_assert_eq!(delta.curr_blocks, 0, "this path cannot move live blocks");

        if self.is_serialized() {
            // One total order over every event, because
            // [`Engine::serialize_for_testing`] promises one over *every* event
            // and an exception here would make that promise conditional on which
            // kind an event was.
            //
            // No test distinguishes this branch from the shared one below, and
            // that is a fact about the harness rather than about the branch:
            // `tests/differential.rs` holds one mutex across both the engine
            // call and the model call, so its order is already total before the
            // gate sees anything. Removing it would leave a future harness that
            // drove events without that mutex comparing against a model whose
            // order the engine no longer follows.
            //
            // Exclusive, and still not `apply_locked`: that is the path for
            // something that may be a peak, and it records an equal one when
            // live bytes sit at the maximum and the delta moves them by zero,
            // which is every event. It did, until the differential model
            // started counting peaks.
            let _order = super::order::enter(super::order::Level::PeakGate);
            let _guard = self.gate.write();
            self.commit(pp, delta, self.epoch.load(Ordering::Relaxed));
            return;
        }

        let _read = super::order::enter(super::order::Level::PeakGate);
        let _guard = self.gate.read();
        self.commit(pp, delta, self.epoch.load(Ordering::Relaxed));
    }

    /// Adds `delta` to the rows for the thread and region it happened on.
    ///
    /// Either may name no row — a thread that could not be given one, or an
    /// allocation outside every region, which is where most allocations in most
    /// programs happen. Both are a load and a compare, not a branch mispredict
    /// waiting to happen: a program that uses no regions takes the same side of
    /// the region test every time.
    ///
    /// # Why this runs inside the gate
    ///
    /// Because otherwise the rows describe a different instant from the totals,
    /// and by more than "threads in flight" covers. Applied *before* the gate
    /// was acquired, a thread that has moved its row then blocks waiting for the
    /// flush to release — so under a shutdown that holds the gate while threads
    /// queue behind it, the rows run ahead of the totals by however many are
    /// queued. **Measured**: 15,534 bytes of 175,191 on the probe's
    /// concurrent-shutdown row, which is 9% and a wrong profile rather than a
    /// failed check.
    ///
    /// So it costs up to six relaxed read-modify-writes inside the critical
    /// section — on a row written almost exclusively by one thread, next to a
    /// shard lock and nine counters that were already there. That is the price
    /// of the rows summing to the totals in a file anyone can check, and a
    /// profile whose own numbers disagree by 9% is not worth the cycles it
    /// saved.
    #[inline]
    fn attribute(&self, delta: Delta) {
        if let Some(tally) = self.threads.tally(delta.site.thread) {
            tally.apply(
                delta.curr_bytes,
                delta.curr_blocks,
                delta.total_bytes,
                delta.total_blocks,
            );
        }
        if let Some(tally) = self.regions.tally(delta.site.region) {
            tally.apply(
                delta.curr_bytes,
                delta.curr_blocks,
                delta.total_bytes,
                delta.total_blocks,
            );
        }
    }

    /// Commits `delta` if it can be shown, at the moment of commit, not to
    /// reach the recorded maximum.
    ///
    /// Returns `false` if the event has to be handled exclusively.
    #[inline]
    fn try_apply_shared(&self, pp: PpId, delta: Delta) -> bool {
        let growth = delta.curr_bytes as u64;

        // Checked *before* acquiring, and racy on purpose. During a
        // monotonically growing phase every event is a new peak, so without this
        // every event would acquire the gate shared, discover it cannot commit,
        // release, and acquire again exclusively — two acquisitions of the most
        // contended lock in the profiler where one would do.
        //
        // Stale reads are harmless in both directions: a false negative costs
        // one wasted shared attempt, and a false positive costs nothing at all,
        // because `apply_exclusive` re-reads everything under exclusion. The
        // shared path is purely an optimisation, so declining to take it can
        // never produce a wrong answer.
        if self.curr_bytes.load(Ordering::Relaxed).wrapping_add(growth)
            >= self.max_bytes.load(Ordering::Relaxed)
        {
            return false;
        }

        let _read = super::order::enter(super::order::Level::PeakGate);
        let _guard = self.gate.read();

        loop {
            let curr = self.curr_bytes.load(Ordering::Relaxed);
            let next = curr.wrapping_add(growth);
            if next >= self.max_bytes.load(Ordering::Relaxed) {
                // This would reach the peak. Bail out *without committing*, so
                // that the exclusive path can re-read a state nobody is midway
                // through changing.
                return false;
            }
            if self
                .curr_bytes
                .compare_exchange_weak(curr, next, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // The epoch cannot change while this shared guard is held, so
                // the value read here is the one the update must use.
                let epoch = self.epoch.load(Ordering::Relaxed);
                self.commit_after_bytes(pp, delta, epoch);
                return true;
            }
        }
    }

    /// Applies `delta` with the gate held exclusively, recording a new peak if
    /// this event produces one.
    #[cold]
    fn apply_exclusive(&self, pp: PpId, delta: Delta) {
        let _write = super::order::enter(super::order::Level::PeakGate);
        let _guard = self.gate.write();
        self.apply_locked(pp, delta);
    }

    /// The body of the exclusive path, with the gate already held.
    fn apply_locked(&self, pp: PpId, delta: Delta) {
        if delta.curr_bytes < 0 {
            let amount = delta.curr_bytes.unsigned_abs();
            let previous = self.curr_bytes.load(Ordering::Relaxed);
            match previous.checked_sub(amount) {
                Some(next) => self.curr_bytes.store(next, Ordering::Relaxed),
                None => {
                    self.curr_bytes.store(0, Ordering::Relaxed);
                    super::diagnostic::poison("live bytes went negative");
                }
            }
            let epoch = self.epoch.load(Ordering::Relaxed);
            self.commit_after_bytes(pp, delta, epoch);
            return;
        }

        let growth = delta.curr_bytes as u64;
        let next = self.curr_bytes.load(Ordering::Relaxed).wrapping_add(growth);
        self.curr_bytes.store(next, Ordering::Relaxed);

        let epoch = self.epoch.load(Ordering::Relaxed);
        let next_blocks = self.commit_after_bytes(pp, delta, epoch);

        // `>=`, not `>`: with several equal peaks, the latest is the one
        // reported. Checked *after* the counters move, so the snapshot the epoch
        // implies includes this event.
        if next >= self.max_bytes.load(Ordering::Relaxed) {
            // For readers outside the gate, and nothing else: the peak is the
            // one counter of a window written with a plain store, which,
            // unlike the totals' read-modify-writes, carries no release
            // sequence back to the restart that opened the window. With this
            // fence the chain is formal: that restart's odd sequence store
            // happens before this thread took the gate, which happens before
            // this fence, which synchronizes with the acquire fence of any
            // reader that loads the stores below, so that reader's second load
            // of the sequence sees the restart and retries. See
            // `Engine::read_window`. On the cold path only, with the gate held
            // exclusively, where one fence is nothing beside the peak itself.
            fence(Ordering::Release);
            self.max_bytes.store(next, Ordering::Relaxed);
            self.max_blocks.store(next_blocks, Ordering::Relaxed);
            self.time_at_max
                .store(self.clock.now(self.time_source()), Ordering::Relaxed);
            // Released last, so a reader that observes the new epoch also
            // observes the counters that justify it.
            self.epoch.store(epoch + 1, Ordering::Release);
        }
    }

    /// Applies everything in `delta` except the live-byte total, which the
    /// caller has already committed in the way its path requires.
    ///
    /// Returns the resulting live block count.
    #[inline]
    fn commit_after_bytes(&self, pp: PpId, delta: Delta, epoch: u64) -> u64 {
        let blocks = if delta.curr_blocks == 0 {
            // Guarded for the same reason the two totals below are, and it
            // matters most on the path that has no blocks at all: an ad hoc
            // event moves nothing here, and an unguarded `fetch_add(0)` is
            // still a read-modify-write on the most contended word in the
            // profiler. The whole justification for `apply_without_peak` is
            // that an event allocated nothing; paying a `lock xadd` per event
            // to add zero contradicts it.
            self.curr_blocks.load(Ordering::Relaxed)
        } else if delta.curr_blocks > 0 {
            self.curr_blocks
                .fetch_add(delta.curr_blocks as u64, Ordering::Relaxed)
                .wrapping_add(delta.curr_blocks as u64)
        } else {
            // Checked, for the same reason as `curr_bytes`, and by the same
            // cheap after-the-fact test rather than a compare-exchange loop.
            let amount = delta.curr_blocks.unsigned_abs();
            let previous = self.curr_blocks.fetch_sub(amount, Ordering::Relaxed);
            match previous.checked_sub(amount) {
                Some(remaining) => remaining,
                None => {
                    self.curr_blocks.store(0, Ordering::Relaxed);
                    super::diagnostic::poison("live blocks went negative");
                    0
                }
            }
        };

        if delta.total_bytes != 0 {
            self.total_bytes
                .fetch_add(delta.total_bytes, Ordering::Relaxed);
        }
        if delta.total_blocks != 0 {
            self.total_blocks
                .fetch_add(delta.total_blocks, Ordering::Relaxed);
        }

        self.pps.update(pp, epoch, |counters| {
            apply_to_counters(counters, delta);
        });

        // Here, and not before the gate was taken: every gated path funnels
        // through this function, so this is the one place the rows can move in
        // the same critical section as the counters they are checked against.
        self.attribute(delta);

        blocks
    }

    /// The shrink path: commits the live-byte change and everything else.
    ///
    /// The subtraction is checked. An unchecked `fetch_sub` past zero wraps to
    /// roughly 2^64, and the damage does not stop there: the very next
    /// allocation sees `next >= max_bytes` trivially, takes the exclusive path,
    /// and **stores the wrapped value as the peak**. Every at-peak number in the
    /// profile is then meaningless, permanently, with nothing to indicate it.
    ///
    /// PLAN.md section 4.6 requires poison-and-degrade on an internal invariant
    /// violation, and live bytes going negative is exactly that.
    #[inline]
    fn commit(&self, pp: PpId, delta: Delta, epoch: u64) {
        if delta.curr_bytes < 0 {
            let amount = delta.curr_bytes.unsigned_abs();
            // `fetch_sub` and check the *returned* previous value, rather than a
            // `fetch_update` compare-exchange loop. Detection is equally exact
            // and costs one comparison instead of a CAS retry storm on the
            // hottest contended word in the profiler — measured at roughly
            // double the per-free cost at 8 threads.
            //
            // The wrapped value is observable for the few instructions before
            // the repair below, and that is harmless: `max_bytes` only changes
            // under *exclusive* access, which cannot run while this shared guard
            // is held. A concurrent shared reader that sees the wrapped value
            // fails its "is this a peak" test, bails to the exclusive path, and
            // waits — by which time the value has been repaired to zero.
            let previous = self.curr_bytes.fetch_sub(amount, Ordering::Relaxed);
            if previous < amount {
                self.curr_bytes.store(0, Ordering::Relaxed);
                super::diagnostic::poison("live bytes went negative");
            }
        }
        self.commit_after_bytes(pp, delta, epoch);
    }

    /// Reads the global counters.
    pub fn stats(&self) -> GlobalStats {
        GlobalStats {
            curr_bytes: self.curr_bytes.load(Ordering::Relaxed),
            curr_blocks: self.curr_blocks.load(Ordering::Relaxed),
            max_bytes: self.max_bytes.load(Ordering::Relaxed),
            max_blocks: self.max_blocks.load(Ordering::Relaxed),
            total_bytes: self.total_bytes.load(Ordering::Relaxed),
            total_blocks: self.total_blocks.load(Ordering::Relaxed),
            time_at_max: self.time_at_max.load(Ordering::Relaxed),
            epoch: self.epoch.load(Ordering::Relaxed),
            dropped_blocks: self.dropped_blocks.load(Ordering::Relaxed),
            refused_events: self.refused_events.load(Ordering::Relaxed),
        }
    }

    /// Reads what the program asked for, beyond a number of bytes.
    ///
    /// A separate call from [`Engine::stats`] because the arrays are a kilobyte
    /// and most callers of `stats` want six numbers. A profile reads both
    /// through [`Engine::flush_and_visit`], which takes them in one window —
    /// see [`Flush::shapes`] for why that matters and for what it still does
    /// not guarantee.
    pub fn shapes(&self) -> ShapeStats {
        self.shapes.snapshot()
    }

    /// Brings every program point's at-peak snapshot up to date, visits them,
    /// and reports the global counters **from the same exclusive window**.
    ///
    /// Returning the stats is not a convenience. An earlier version returned
    /// `()` and left the caller to call `stats()` afterwards, in a second
    /// unsynchronised acquisition — so any event landing in between changed
    /// `max_bytes` and the emitted per-point values no longer summed to it, with
    /// no bug anywhere. A profile that fails its own invariant for timing
    /// reasons is worse than one that fails for a reason you can find.
    ///
    /// Waits at most `timeout` for exclusive access. On expiry the flush
    /// proceeds without it and [`Flush::exclusive`] is `false`: PLAN.md section
    /// 4.6 requires the shutdown path to degrade to partial output rather than
    /// hang the process at `exit`, and a wedged reader must not be able to
    /// prevent a profile from being written at all.
    ///
    /// # Why the thread and region rows are visited here too
    ///
    /// Because every number a profile cross-checks against another has to be
    /// read in one window, and the rows are cross-checked: they sum to the
    /// totals. Three visitors rather than one is the price of making that
    /// structural — a caller cannot read a table outside the window without
    /// writing the code to do so, which is exactly the mistake this signature
    /// exists to prevent.
    ///
    /// **No visitor may allocate.** They run with the gate held, and an
    /// allocation from here re-enters the engine, which blocks on the gate this
    /// thread is holding. Callers reserve their storage before the call; a row
    /// view is plain `Copy` data, so turning its name into an owned string
    /// happens afterwards.
    pub fn flush_and_visit(
        &self,
        timeout: Duration,
        visit: impl FnMut(PpId, &[usize], &Counters),
        visit_thread: impl FnMut(super::site::ThreadView),
        visit_region: impl FnMut(super::site::RegionView),
    ) -> Flush {
        let _order = super::order::enter(super::order::Level::PeakGate);
        let guard = self.gate.write_for(timeout);
        let exclusive = guard.is_some();
        if !exclusive {
            super::diagnostic::report(
                "could not reach a quiet point before writing the profile; \
                 the at-peak columns may not sum to the peak",
            );
        }

        let epoch = self.epoch.load(Ordering::Relaxed);
        self.pps.flush_and_visit(epoch, visit);
        let stats = self.stats();
        let shapes = self.shapes();
        // In the window, for the same reason `Flush::shapes` is. Read
        // afterwards, these would describe an instant several milliseconds
        // later than the totals do — after the frames have been copied and the
        // live-block table swept — so anything still recording in that interval
        // lands in one number and not the other.
        //
        // The shutdown path cannot demonstrate that, and a comment here once
        // claimed it could: `Profiler::drop` stops the engine and drains the
        // gate *before* a snapshot is taken, so nothing is left in flight, and
        // moving these two lines below `drop(guard)` changes no number on that
        // path in twenty runs. The case this is for is
        // `Snapshot::capture` on a **running** engine, which
        // `a_snapshot_of_a_running_engine_still_sums` covers.
        self.threads.visit(visit_thread);
        self.regions.visit(visit_region);
        let reset = self.last_reset();
        drop(guard);

        Flush {
            stats,
            shapes,
            exclusive,
            reset,
        }
    }

    /// Restarts the run's counts, keeping what is live.
    ///
    /// This is [`Profiler::reset`](crate::Profiler::reset), and what follows is
    /// the rule it applies to every counter the engine keeps. One principle
    /// decides all of them: what *happened* before the restart is forgotten,
    /// and what *is* at the restart is kept. A warm-up's allocations happened;
    /// the caches it left behind are.
    ///
    /// | Counters | After a restart |
    /// |---|---|
    /// | Bytes and blocks allocated, globally, per program point, per thread and per region | zero |
    /// | Events refused for being the wrong kind | zero |
    /// | Live bytes and blocks, at every one of those levels | unchanged |
    /// | The peak, and each point's and row's own running maximum | what is live now |
    /// | Each point's bytes and blocks at the peak | what it holds now |
    /// | The instant of the peak | now |
    /// | Summed lifetimes per point | zero, with every block live now counting toward none |
    /// | Requested sizes, alignments, zeroed blocks, reallocation costs | zero |
    /// | Blocks the live-block table turned away | unchanged; see [`GlobalStats::dropped_blocks`] |
    /// | The clock, capture counts, a region's entries, the profiler's own costs | unchanged |
    ///
    /// # Why live state survives
    ///
    /// Because it is still true. A block allocated during the warm-up and
    /// freed afterwards must bring live bytes down when it goes, or every
    /// figure that depends on them goes negative, which this crate treats as
    /// corruption. Its program point, thread and region must still hold it, or
    /// the free lands on a counter that never knew it. So the live-block table
    /// is kept entry for entry, and so are the live counters it balances.
    ///
    /// # The peak
    ///
    /// A restart is a peak, at the level that is live when it happens: that is
    /// the most the new window has held. So it moves the peak to the current
    /// level, moves its instant to now, and moves the epoch like any other new
    /// peak, so the per-point at-peak figures describe this instant and sum to
    /// it. They are set eagerly, by the walk of the program-point table this
    /// already makes, rather than left to the lazy refresh; the epoch moves so
    /// that the scheme's invariant, one epoch per peak, still holds.
    ///
    /// # Lifetimes
    ///
    /// DHAT's average lifetime is the summed lifetime over the blocks counted,
    /// and the blocks counted are now the window's. A block live across the
    /// restart is not one of them, so its lifetime is counted nowhere, not even
    /// the part of it that falls inside the window: in the sum and not in the
    /// count, a warm-up's long-lived caches would inflate the average at
    /// exactly the sites a reader scans for churn. Every live block is marked
    /// [`LiveBlock::CARRIED`] to say so, which costs the free path nothing.
    /// A reallocation of one is a new block, born in the window, and counts.
    ///
    /// # Locks
    ///
    /// Every live-block shard, then the peak gate exclusively: the order
    /// [`super::order`] fixes, and the one [`Engine::fork_prepare`] takes
    /// them in. Holding both means no block is inserted or removed and no
    /// counter moves while the restart is applied, so it is one step at one
    /// instant.
    ///
    /// All of it is waited for against one deadline, the one a profile written
    /// at shutdown has, and [`ResetError::Busy`] is the result of missing it,
    /// with every lock taken so far released and nothing changed. The shards
    /// are polled for rather than blocked on, unlike `fork_prepare`'s, because
    /// that handler has no way to give up and this does: a thread stopped while
    /// holding one shard, under a debugger say, would otherwise hold the
    /// restart forever, and with it every thread whose blocks hash to a shard
    /// the restart already holds. Bounded, the worst a wedged shard costs is
    /// those threads stalling for the deadline, once.
    ///
    /// # What is not exact while other threads are recording
    ///
    /// Every counter the gate covers is: an event's counters land wholly before
    /// the restart or wholly after it. Two things straddle it, both because
    /// they happen outside the gate. A shape is counted before an allocation
    /// reaches the gate, as [`Flush::shapes`] documents. And a block's live
    /// entry is written before its counters move, so a block whose entry was
    /// marked carried and whose counters land after the restart is in the
    /// window's totals with no lifetime, while a block removed just before the
    /// restart and counted just after it brings a lifetime from before. Both
    /// are bounded by the threads in flight at that instant.
    ///
    /// # Readers outside the gate
    ///
    /// [`Engine::reset_sequence`] is made odd before the first counter moves
    /// and even again after the last, so a reader that does not take the gate
    /// can still tell which window it read. See [`Engine::read_window`].
    ///
    /// # Why this takes a [`Guard`]
    ///
    /// For the reason [`Engine::record_event`] does, and with more at stake:
    /// this takes every live-block shard, and a thread already inside the
    /// profiler may hold one. On Apple platforms a second acquisition kills the
    /// process. Proof of the guard makes that a borrow-check error rather than
    /// a crash.
    pub fn reset(&self, _entered: &Guard) -> Result<(), ResetError> {
        // Refused before any lock is taken, so that a stopped run is not made
        // to wait two seconds to be told it is stopped.
        self.resettable()?;

        // `checked_add` for the reason `RawLock::try_lock_for` gives: a panic
        // here would be inside the profiler, and two seconds from now is
        // representable on every platform this runs on, so the fallback is
        // never taken in practice.
        let deadline = Instant::now().checked_add(Self::FLUSH_TIMEOUT);
        let Some(mut live) = self.live.lock_all_until(deadline) else {
            return Err(ResetError::Busy);
        };
        let _order = super::order::enter(super::order::Level::PeakGate);
        let remaining = deadline.map_or(Self::FLUSH_TIMEOUT, |deadline| {
            deadline.saturating_duration_since(Instant::now())
        });
        let Some(_gate) = self.gate.write_for(remaining) else {
            return Err(ResetError::Busy);
        };
        // Asked again now that nothing can move: a shutdown or a poison may have
        // arrived while this waited, and a stopping run's profile is about to be
        // read from these counters.
        self.resettable()?;

        let now = self.clock.now(self.time_source());
        let curr_bytes = self.curr_bytes.load(Ordering::Relaxed);
        let curr_blocks = self.curr_blocks.load(Ordering::Relaxed);
        let epoch = self.epoch.load(Ordering::Relaxed) + 1;

        // The writer's half of the sequence lock. Relaxed load: the gate makes
        // this the only writer, and the value it last wrote is its own. Odd
        // before anything moves, then a release fence, so that a reader which
        // observes any store below has, through its own acquire fence, also
        // observed this one. See `read_window` for that side.
        let sequence = self.reset_sequence.load(Ordering::Relaxed);
        debug_assert!(sequence.is_multiple_of(2), "a restart began inside another");
        self.reset_sequence.store(sequence + 1, Ordering::Relaxed);
        fence(Ordering::Release);
        // Even again when this drops, on every way out of here. Declared after
        // the gate, so dropped before it: the even store is made while the
        // gate is still held, as the odd one was.
        let restarting = Restarting {
            sequence: &self.reset_sequence,
            even: sequence + 2,
        };

        self.total_bytes.store(0, Ordering::Relaxed);
        self.total_blocks.store(0, Ordering::Relaxed);
        self.refused_events.store(0, Ordering::Relaxed);
        self.max_bytes.store(curr_bytes, Ordering::Relaxed);
        self.max_blocks.store(curr_blocks, Ordering::Relaxed);
        self.time_at_max.store(now, Ordering::Relaxed);

        self.pps.restart(epoch);
        self.threads.restart();
        self.regions.restart();
        self.shapes.clear();
        live.for_each_mut(|_, block| block.birth = LiveBlock::CARRIED);

        self.reset_at.store(now, Ordering::Relaxed);
        self.reset_epoch.store(epoch, Ordering::Relaxed);
        self.carried_bytes.store(curr_bytes, Ordering::Relaxed);
        self.carried_blocks.store(curr_blocks, Ordering::Relaxed);
        self.dropped_at_reset.store(
            self.dropped_blocks.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        // Released last, as `apply_locked` releases it, so that a reader that
        // observes the new epoch also observes the counters that justify it.
        self.epoch.store(epoch, Ordering::Release);
        drop(restarting);
        Ok(())
    }

    /// Whether this engine has counts a restart may touch, or why not.
    fn resettable(&self) -> Result<(), ResetError> {
        match self.state() {
            State::Running => {}
            State::ForkedChild => return Err(ResetError::ForkedChild),
            // Idle and Starting are unreachable through a `Profiler`, which
            // exists only once the run is `Running`; see `ResetError::Stopped`.
            State::Idle | State::Starting | State::Finished => return Err(ResetError::Stopped),
        }
        if super::diagnostic::is_poisoned() {
            return Err(ResetError::Poisoned);
        }
        Ok(())
    }

    /// The most recent restart of the counts, or `None` if there has been none.
    ///
    /// Read field by field, so exact only with the peak gate held, which is how
    /// [`Engine::flush_and_visit`] reads it. A caller that needs to know only
    /// *whether* the counts moved under it wants [`Engine::resets`].
    pub fn last_reset(&self) -> Option<Reset> {
        // Even, and the count it implies complete, wherever the gate is held,
        // because the restart holds the gate across the whole odd interval.
        let count = self.reset_sequence.load(Ordering::Relaxed) / 2;
        if count == 0 {
            return None;
        }
        Some(Reset {
            count,
            at: self.reset_at.load(Ordering::Relaxed),
            carried_bytes: self.carried_bytes.load(Ordering::Relaxed),
            carried_blocks: self.carried_blocks.load(Ordering::Relaxed),
            dropped_blocks: self.dropped_at_reset.load(Ordering::Relaxed),
            epoch: self.reset_epoch.load(Ordering::Relaxed),
        })
    }

    /// How many restarts of the counts have completed.
    ///
    /// Exact as a count, and not a statement about any counter read beside
    /// it: a restart may land between the two reads. A reader that needs both
    /// to describe one window wants [`Engine::read_window`].
    pub fn resets(&self) -> u64 {
        self.reset_sequence.load(Ordering::Acquire) / 2
    }

    /// Runs `read` and returns what it read with the number of restarts that
    /// had completed, guaranteeing that no restart landed in between.
    ///
    /// The reader's half of a sequence lock around [`Engine::reset`], and how a
    /// reading outside the gate pairs its counters with its window without
    /// taking a lock. `read` may run more than once, so it must only load.
    ///
    /// # Orderings
    ///
    /// The first load is `Acquire`: if it sees the even value a restart
    /// released at its end, every store that restart made is visible to
    /// `read`, which then reads the new window whole. An odd value means a
    /// restart is under way, and the reading waits it out rather than read
    /// half of one.
    ///
    /// `read` loads with `Relaxed`, then an `Acquire` fence, then the second
    /// load. If `read` saw any store a restart made after its release fence,
    /// the two fences synchronize, the restart's odd store happens before the
    /// second load, and the second load cannot return the first's value: the
    /// reading is retried. That covers every counter a window owns:
    ///
    /// - the totals, because after a restart's store to one every later write
    ///   to it is a read-modify-write, which extends that store's release
    ///   sequence, so a total read from the new window synchronizes however
    ///   many allocations have added to it since;
    /// - the peak, which a new peak writes with a plain store, because
    ///   `apply_locked` puts a release fence before it, and the restart that
    ///   opened the window happens before that thread took the gate.
    ///
    /// The live figures are not a window's: a restart carries them across
    /// unchanged, so a reading cannot pair them with the wrong one.
    ///
    /// # When a restart is under way
    ///
    /// The reading waits it out, which is short: a sweep of the tables. Except
    /// on a thread that cannot enter the profiler, which is how a thread inside
    /// it looks, the restarting thread among them: a signal handler that
    /// interrupted a restart and read the counters would otherwise wait for
    /// the very restart it stopped, forever. That reading returns `None`. The
    /// check is made only when the sequence is odd, so an ordinary reading
    /// never pays for it.
    pub fn read_window<T>(&self, mut read: impl FnMut() -> T) -> Option<(u64, T)> {
        loop {
            let before = self.reset_sequence.load(Ordering::Acquire);
            if !before.is_multiple_of(2) {
                // `None` from `enter` is a thread already inside the profiler,
                // or one with no slot to enter with; either way it cannot be
                // told apart from the thread the restart is running on. The
                // guard is only a probe: `let _` releases it at once, so this
                // thread does not wait while holding it.
                let _ = super::guard::enter()?;
                // A restart holds the gate and every shard for a sweep of the
                // tables, which is long enough to be worth giving the core up.
                std::thread::yield_now();
                continue;
            }
            let value = read();
            fence(Ordering::Acquire);
            if self.reset_sequence.load(Ordering::Relaxed) == before {
                return Some((before / 2, value));
            }
        }
    }

    /// How long the shutdown flush waits for in-flight events to finish.
    ///
    /// Long enough that any real critical section completes many times over,
    /// short enough that a wedged thread costs a noticeable pause rather than a
    /// hang.
    pub const FLUSH_TIMEOUT: Duration = Duration::from_secs(2);
}

/// The writer's half of the sequence lock around [`Engine::reset`], from the
/// odd store to the even one.
///
/// A guard rather than a store at the end, because a reset that unwound
/// between the two, and was caught, would leave the sequence odd for good, and
/// every reading after it waiting for a restart that will never finish. The
/// even store is made however the restart ends. An unwinding restart has left
/// the counters half restarted, so it also poisons the run: what readers then
/// get is a refusal that says the profiler failed, not a window that is half
/// of one and half of another.
struct Restarting<'a> {
    sequence: &'a AtomicU64,
    even: u64,
}

impl Drop for Restarting<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            super::diagnostic::poison("a restart of the counts was interrupted partway");
        }
        // Release: a reader that loads this value with acquire sees every
        // store the restart made, so it reads the new window whole.
        self.sequence.store(self.even, Ordering::Release);
    }
}

/// [`Engine::record_alloc`] and [`Engine::record_realloc_taken`] with the
/// reentrancy guard taken here rather than by the caller.
///
/// A unit test that drives the engine directly is standing in for the allocator
/// shim, which holds the guard across every recording call — so a test must
/// too, or it is exercising a sequence the shim never produces. These exist so
/// that the obligation is met in one place instead of by a `let guard = ...`
/// line in forty tests whose subject is something else entirely. Tests that are
/// *about* the guard take their own and call the real methods.
#[cfg(test)]
impl Engine {
    /// Standing in for the shim means asking [`Engine::observe`] first, not only
    /// holding the guard: that call is where the shape is counted and where the
    /// sampler is advanced, so a helper that skipped it would leave every test
    /// driving a sequence with no histograms and a sampler that never moves.
    pub(crate) fn record_alloc_guarded(&self, address: usize, shape: Shape, frames: &[usize]) {
        let guard = super::guard::enter().expect("this thread is already inside the profiler");
        if self.observe(&guard, shape) {
            self.record_alloc(&guard, address, shape, frames);
        }
    }

    pub(crate) fn record_realloc_guarded(
        &self,
        taken: Option<LiveBlock>,
        realloc: Realloc,
        frames: &[usize],
    ) {
        let guard = super::guard::enter().expect("this thread is already inside the profiler");
        if self.observe_realloc(&guard, &realloc, taken.is_some()) {
            self.record_realloc_taken(&guard, taken, realloc, frames);
        }
    }

    /// [`Engine::reset`], from a thread standing in for the program, which
    /// is not inside the profiler.
    pub(crate) fn reset_guarded(&self) -> Result<(), ResetError> {
        let guard = super::guard::enter().expect("this thread is already inside the profiler");
        self.reset(&guard)
    }

    /// Puts this engine in the state a `fork` child inherits.
    ///
    /// [`Engine::fork_child`] is the real transition and cannot stand in for
    /// this: it reinitialises the guard and lock-order tables, which are
    /// **process-wide** rather than per-engine, so calling it from one test
    /// would reset state the rest of the test binary is using. What a caller
    /// wants here is the state alone.
    pub(crate) fn disown_for_testing(&self) {
        self.state
            .store(State::ForkedChild as u8, Ordering::Release);
        self.shutdown
            .store(Shutdown::ForkedChild as u8, Ordering::Relaxed);
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

/// The calling thread's name, as the platform has it right now.
fn current_name() -> Name {
    let mut buffer = [0u8; MAX_NAME];
    let len = super::site::current_thread_name(&mut buffer);
    Name::of_bytes(&buffer[..len])
}

fn apply_to_counters(counters: &mut Counters, delta: Delta) {
    // The per-point counters use the same checked discipline as the global ones.
    // An earlier version used `saturating_sub` here while the global counter
    // wrapped, so on an underflow the two diverged in *opposite* directions --
    // one clamped at zero, the other at 2^64 -- and nothing reported it.
    // Saturation was not a safety net; it was what hid the fault.
    if delta.curr_bytes >= 0 {
        counters.curr_bytes = counters.curr_bytes.saturating_add(delta.curr_bytes as u64);
    } else {
        match counters
            .curr_bytes
            .checked_sub(delta.curr_bytes.unsigned_abs())
        {
            Some(value) => counters.curr_bytes = value,
            None => {
                counters.curr_bytes = 0;
                super::diagnostic::poison("per-point live bytes went negative");
            }
        }
    }
    if delta.curr_blocks >= 0 {
        counters.curr_blocks = counters
            .curr_blocks
            .saturating_add(delta.curr_blocks as u64);
    } else {
        match counters
            .curr_blocks
            .checked_sub(delta.curr_blocks.unsigned_abs())
        {
            Some(value) => counters.curr_blocks = value,
            None => {
                counters.curr_blocks = 0;
                super::diagnostic::poison("per-point live blocks went negative");
            }
        }
    }
    // `saturating_add`, not `+=`: a plain add panics on overflow in debug
    // builds, and this runs inside the allocator shim where a panic allocates
    // its own message and re-enters. `total_lifetime` is the realistic one --
    // it accumulates every block's lifetime in event units.
    counters.total_bytes = counters.total_bytes.saturating_add(delta.total_bytes);
    counters.total_blocks = counters.total_blocks.saturating_add(delta.total_blocks);
    counters.total_lifetime = counters.total_lifetime.saturating_add(delta.lifetime);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Engine {
        let engine = Engine::with_limits(1 << 12, 1 << 14);
        assert!(engine.start(TimeSource::Events, || {}));
        engine
    }

    /// An engine sampling at `interval` mean bytes.
    fn sampled(interval: u64) -> Engine {
        let engine = Engine::with_limits(1 << 12, 1 << 14);
        assert!(
            engine.start(TimeSource::Events, || engine.configure(Settings {
                sampling: NonZeroU64::new(interval),
                ..Settings::default()
            }))
        );
        engine
    }

    /// Records allocations of `size` at `frames` until one is sampled, and
    /// returns its address.
    ///
    /// Sampling is a random process, so a test that recorded once and expected a
    /// sample would fail a fraction of the time. Looping is what makes the tests
    /// below deterministic without pinning the generator.
    fn record_until_sampled(engine: &Engine, from: usize, size: usize, frames: &[usize]) -> usize {
        let before = engine.stats().total_blocks;
        let mut address = from;
        loop {
            engine.record_alloc_guarded(address, Shape::of(size), frames);
            if engine.stats().total_blocks != before {
                return address;
            }
            address += 0x40;
            assert!(
                address < from + 0x0400_0000,
                "nothing was sampled in a million allocations of {size} bytes"
            );
        }
    }

    /// A reallocation, named at the call site so that neither old/new pair can
    /// be transposed without the test reading wrong.
    fn grew(old_address: usize, old_size: usize, new_address: usize, new_size: usize) -> Realloc {
        Realloc {
            old_address,
            old_size,
            new_address,
            new: Shape::of(new_size),
        }
    }

    /// Sums every attribution row's counters, which is the invariant the
    /// profile's own validator checks: every recorded allocation belongs to
    /// exactly one thread.
    fn thread_totals(engine: &Engine) -> (u64, u64, u64) {
        let mut totals = (0, 0, 0);
        engine.threads().visit(|row| {
            totals.0 += row.counts.total_bytes;
            totals.1 += row.counts.total_blocks;
            totals.2 += row.counts.curr_bytes;
        });
        totals
    }

    /// The rows and the global counters take the same delta from the same call
    /// site, so they cannot disagree in a quiet run. This is the equality the
    /// native format's validator applies, checked here where it can be exact.
    #[test]
    fn every_recorded_allocation_belongs_to_exactly_one_thread() {
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(4_096), &[0xAA]);
        engine.record_alloc_guarded(0x2000, Shape::of(1_024), &[0xBB]);
        engine.record_free(0x1000, 4_096);
        engine.record_realloc_guarded(
            engine.live_blocks().remove(0x2000),
            grew(0x2000, 1_024, 0x3000, 8_192),
            &[0xBB],
        );

        let stats = engine.stats();
        let (bytes, blocks, live) = thread_totals(&engine);
        assert_eq!(bytes, stats.total_bytes);
        assert_eq!(blocks, stats.total_blocks);
        assert_eq!(live, stats.curr_bytes);
    }

    /// A row whose name the platform did not have when it was claimed must be
    /// asked again by the *engine*, on a later recorded event.
    ///
    /// The table's own retry has a unit test; this pins the wiring that calls
    /// it, which is the part that is dead code on unix — there the first attempt
    /// always succeeds, so only Windows exercises it end to end. Driven here
    /// from a row deliberately claimed with no name, so the check holds on every
    /// platform: either the engine finds a name and the row settles, or it runs
    /// the attempts out and the row settles anyway. What it must not do is never
    /// ask, which is what leaves every Windows worker unnamed.
    #[test]
    fn a_row_claimed_without_a_name_is_asked_again_by_the_engine() {
        use crate::internals::site::NAME_ATTEMPTS;

        let engine = engine();
        let id = engine.threads().claim(engine.arena(), Name::EMPTY, 0);
        assert!(
            engine.threads().wants_name(id),
            "a row claimed with no name settled on the spot"
        );

        let guard = super::super::guard::enter().expect("this thread is not inside the profiler");
        guard.set_thread(id);
        for event in 0..u64::from(NAME_ATTEMPTS) {
            engine.record_alloc(&guard, 0x1000 + event as usize * 64, Shape::of(64), &[0xAA]);
        }

        assert!(
            !engine.threads().wants_name(id),
            "the engine never asked the platform again, so a thread named after \
             its first allocation — every worker on Windows — stays unnamed"
        );
    }

    /// Releases the workers below when it drops.
    ///
    /// Not a convenience. `std::thread::scope` **joins** its threads when the
    /// closure unwinds, and a worker looping until a flag is never joinable if the
    /// panic happened before the line that sets it — so an assertion failure inside
    /// the scope hangs the test binary instead of failing it. Measured the hard
    /// way: a mutation run left this suite spinning four threads for 134 CPU
    /// minutes and reported nothing, because a test that hangs produces no result
    /// at all. A guard drops during the unwind, before the join.
    struct StopOnDrop<'a>(&'a std::sync::atomic::AtomicBool);

    impl Drop for StopOnDrop<'_> {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The rows are read inside the flush window because a snapshot of a
    /// **running** engine is the case where reading them afterwards is wrong.
    ///
    /// The shutdown path cannot show this: `Profiler::drop` stops the engine and
    /// drains the gate before a snapshot is taken, so nothing is in flight and
    /// the rows cannot move in the interval. Here threads keep allocating
    /// throughout, and the flush's exclusive window is the only thing that makes
    /// the rows and the totals describe one instant.
    #[test]
    fn a_snapshot_of_a_running_engine_still_sums() {
        // Bounded on both sides under Miri, which interprets every instruction:
        // an unbounded spin there is not slow, it is a run that does not end.
        // The workers stop on their own count as well as on the flag, so the
        // live-block table cannot grow while the interpreter works through it.
        #[cfg(miri)]
        const SNAPSHOTS: usize = 3;
        #[cfg(not(miri))]
        const SNAPSHOTS: usize = 20;
        #[cfg(miri)]
        const ROUNDS: usize = 32;
        #[cfg(not(miri))]
        const ROUNDS: usize = usize::MAX;

        let engine = engine();
        let stop = std::sync::atomic::AtomicBool::new(false);
        let threads = 4;

        std::thread::scope(|scope| {
            for worker in 0..threads {
                let engine = &engine;
                let stop = &stop;
                scope.spawn(move || {
                    let base = 0x1_0000_0000usize + worker * 0x1000_0000;
                    let mut round = 0usize;
                    while !stop.load(Ordering::Relaxed) && round < ROUNDS {
                        let address = base + (round % 512) * 128;
                        engine.record_alloc_guarded(
                            address,
                            Shape::of(64 + round % 256),
                            &[0xA0 + worker],
                        );
                        if round.is_multiple_of(3) {
                            engine.record_free(address, 64 + round % 256);
                        }
                        round += 1;
                    }
                });
            }

            // Releases the workers however this scope is left, including by a
            // failing assertion below. See `StopOnDrop`.
            let _release = StopOnDrop(&stop);

            // Snapshots taken while all of that is in flight. Each one has to be
            // internally consistent on its own terms.
            for _ in 0..SNAPSHOTS {
                let (mut rows, mut blocks) = (0u64, 0u64);
                let flush = engine.flush_and_visit(
                    Engine::FLUSH_TIMEOUT,
                    |_, _, _| {},
                    |row| {
                        rows += row.counts.total_bytes;
                        blocks += row.counts.total_blocks;
                    },
                    |_| {},
                );
                if !flush.exclusive {
                    // The file says so, and claims nothing simultaneous.
                    continue;
                }
                assert_eq!(
                    rows, flush.stats.total_bytes,
                    "the thread rows and the totals were read in one exclusive \
                     window and disagree"
                );
                assert_eq!(blocks, flush.stats.total_blocks);
            }
        });
    }

    /// A block allocated on one thread and freed on another must bring the
    /// *allocating* thread's live bytes down. Attributing the free to whoever
    /// happened to call `dealloc` would report every producer thread in a
    /// producer/consumer program as leaking everything it ever made.
    #[test]
    fn a_block_freed_on_another_thread_still_belongs_to_the_one_that_made_it() {
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(4_096), &[0xAA]);
        let owner = engine
            .live_blocks()
            .get(0x1000)
            .expect("the block was just recorded")
            .site
            .thread;

        std::thread::scope(|scope| {
            scope.spawn(|| {
                // A second thread, which claims a row of its own the moment it
                // records anything -- and must not be the one charged for the
                // free below.
                engine.record_alloc_guarded(0x9000, Shape::of(64), &[0xCC]);
                engine.record_free(0x1000, 4_096);
            });
        });

        let mut rows = Vec::new();
        engine
            .threads()
            .visit(|row| rows.push((row.id, row.counts)));
        assert_eq!(rows.len(), 2, "the spawned thread did not get a row");

        let (_, owner_counts) = rows
            .iter()
            .find(|(id, _)| *id == owner)
            .expect("the allocating thread has a row");
        assert_eq!(
            owner_counts.total_bytes, 4_096,
            "the allocating thread was not charged for its own block"
        );
        assert_eq!(
            owner_counts.curr_bytes, 0,
            "a free on another thread left the allocating thread holding bytes \
             it no longer has"
        );

        let (_, other_counts) = rows
            .iter()
            .find(|(id, _)| *id != owner)
            .expect("the freeing thread has a row");
        assert_eq!(
            other_counts.total_bytes, 64,
            "the freeing thread was charged for a block it did not allocate"
        );
        assert_eq!(other_counts.curr_bytes, 64);
    }

    /// A thread asks the platform for its name once. The row is cached in the
    /// guard slot, so a second allocation must not produce a second row.
    #[test]
    fn a_thread_claims_one_row_however_much_it_allocates() {
        let engine = engine();
        for i in 0..64 {
            engine.record_alloc_guarded(0x1000 + i * 64, Shape::of(64), &[0xAA]);
        }
        assert_eq!(engine.threads().len(), 1);
    }

    /// Allocations made inside a region are attributed to it, and the free
    /// brings the region back down even though it happens outside.
    #[test]
    fn a_region_holds_what_was_allocated_inside_it() {
        let engine = engine();
        let parsing = engine.intern_region("parsing");

        // Outside every region.
        engine.record_alloc_guarded(0x1000, Shape::of(1_000), &[0xAA]);

        let held = super::super::guard::enter().expect("not inside the profiler");
        let previous = super::super::guard::enter_region(&held, parsing);
        drop(held);
        engine.regions().enter(parsing);
        engine.record_alloc_guarded(0x2000, Shape::of(4_096), &[0xBB]);
        engine.record_alloc_guarded(0x3000, Shape::of(2_048), &[0xBB]);
        super::super::guard::leave_region(previous);
        engine.regions().leave(parsing);

        // Outside the region again: the free must still find the region row.
        engine.record_free(0x2000, 4_096);

        let mut rows = Vec::new();
        engine.regions().visit(|row| {
            rows.push((
                row.name.as_bytes().to_vec(),
                row.entries,
                row.active,
                row.counts,
            ))
        });
        assert_eq!(rows.len(), 1);
        let (name, entries, active, counts) = &rows[0];
        assert_eq!(name, b"parsing");
        assert_eq!(*entries, 1);
        assert_eq!(*active, 0);
        assert_eq!(
            counts.total_bytes, 6_144,
            "the region was charged for an allocation made outside it, or \
             missed one made inside"
        );
        assert_eq!(counts.total_blocks, 2);
        assert_eq!(
            counts.curr_bytes, 2_048,
            "a free outside the region did not bring the region back down"
        );
        assert_eq!(counts.max_bytes, 6_144);
        assert_eq!(
            engine.stats().total_bytes,
            7_144,
            "the run's own totals must count the allocation made outside too"
        );
    }

    /// A region is scoped to the thread that entered it. Without this, a
    /// process-wide "current phase" would attribute whatever a background
    /// thread happened to be doing to whichever phase some other thread was in.
    #[test]
    fn a_region_open_on_one_thread_does_not_capture_another() {
        let engine = engine();
        let parsing = engine.intern_region("parsing");

        let held = super::super::guard::enter().expect("not inside the profiler");
        let previous = super::super::guard::enter_region(&held, parsing);
        drop(held);
        engine.regions().enter(parsing);
        engine.record_alloc_guarded(0x1000, Shape::of(4_096), &[0xAA]);

        // A second thread, allocating while this one is inside the region.
        std::thread::scope(|scope| {
            scope.spawn(|| {
                engine.record_alloc_guarded(0x9000, Shape::of(2_048), &[0xCC]);
            });
        });

        super::super::guard::leave_region(previous);
        engine.regions().leave(parsing);

        let mut rows = Vec::new();
        engine.regions().visit(|row| rows.push(row.counts));
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].total_bytes, 4_096,
            "the other thread's allocation landed in a region it was never in"
        );
        assert_eq!(
            engine.stats().total_bytes,
            6_144,
            "the run recorded both allocations even though only one is in a region"
        );
    }

    /// A block allocated inside a region and freed anywhere, on any thread,
    /// brings that region's live bytes down. The region row is on the block, not
    /// on whoever frees it.
    #[test]
    fn a_region_comes_back_down_when_another_thread_frees_its_blocks() {
        let engine = engine();
        let parsing = engine.intern_region("parsing");

        let held = super::super::guard::enter().expect("not inside the profiler");
        let previous = super::super::guard::enter_region(&held, parsing);
        drop(held);
        engine.regions().enter(parsing);
        engine.record_alloc_guarded(0x1000, Shape::of(4_096), &[0xAA]);
        super::super::guard::leave_region(previous);
        engine.regions().leave(parsing);

        std::thread::scope(|scope| {
            scope.spawn(|| engine.record_free(0x1000, 4_096));
        });

        let mut rows = Vec::new();
        engine.regions().visit(|row| rows.push(row.counts));
        assert_eq!(rows[0].total_bytes, 4_096);
        assert_eq!(
            rows[0].curr_bytes, 0,
            "a free on a thread that was never in the region left the region \
             holding bytes the program no longer has"
        );
        assert_eq!(rows[0].max_bytes, 4_096);
    }

    /// An ad hoc event has a thread and a region like anything else, and
    /// contributes to the cumulative counters only -- an event was never live.
    #[test]
    fn an_event_is_attributed_without_becoming_live() {
        let engine = Engine::with_limits(1 << 12, 1 << 14);
        assert!(
            engine.start(TimeSource::Events, || engine.configure(Settings {
                mode: Mode::AdHoc,
                ..Settings::default()
            }))
        );

        let guard = super::super::guard::enter().expect("not inside the profiler");
        engine.record_event(&guard, 700, &[0xAA]);
        engine.record_event(&guard, 70, &[0xAA]);
        drop(guard);

        let (bytes, blocks, live) = thread_totals(&engine);
        assert_eq!(bytes, 770);
        assert_eq!(blocks, 2);
        assert_eq!(live, 0, "an event became live on its thread's row");
    }

    /// A profile reports the settings the run *had*, which is why they are read
    /// back from here rather than from the builder that asked for them. Both
    /// ends of the depth range are clamps rather than refusals, so both have to
    /// come back as the value that took effect.
    #[test]
    fn the_settings_reported_are_the_ones_that_took_effect() {
        // Configured through `start`, because that is the only window in which
        // a run's settings may be applied and the only one this will accept.
        fn configured(max_depth: usize, max_live_blocks: usize, trim: bool) -> Settings {
            let engine = Engine::with_limits(1 << 12, 1 << 14);
            assert!(
                engine.start(TimeSource::Events, || engine.configure(Settings {
                    mode: Mode::Heap,
                    max_depth,
                    max_live_blocks,
                    trim_frames: trim,
                    sampling: None,
                }))
            );
            engine.settings()
        }

        let past_the_buffer = configured(crate::CAPTURE_DEPTH + 100, 1 << 15, false);
        assert_eq!(
            past_the_buffer.max_depth,
            crate::CAPTURE_DEPTH,
            "a depth past the shim's buffer was reported as though it fit"
        );
        assert!(!past_the_buffer.trim_frames);
        assert_eq!(past_the_buffer.max_live_blocks, 1 << 15);

        let none_at_all = configured(0, 1 << 14, true);
        assert_eq!(
            none_at_all.max_depth, 1,
            "zero frames per point is not a program point"
        );
        assert!(none_at_all.trim_frames);
    }

    /// An event is not a block. It moves the cumulative totals and nothing
    /// else, and in particular it must not move the peak: `tg` names the instant
    /// the *heap* was largest, and an ad hoc run has no heap.
    ///
    /// The failure this pins is not hypothetical arithmetic. Routing an event
    /// through the ordinary path would take the exclusive gate on every call,
    /// because `curr_bytes + 0 >= max_bytes` holds whenever live bytes sit at
    /// the maximum — and would then record the event's instant as a new equal
    /// peak, which is Valgrind's `>=` rule applied to something that allocated
    /// nothing.
    #[test]
    fn an_event_moves_the_totals_and_nothing_else() {
        let engine = engine();
        // Live bytes are left sitting *at* the maximum, which is the only
        // interesting moment for an event. There `curr + 0 >= max` holds, so the
        // ordinary path would take the exclusive gate and record this instant as
        // a new equal peak — Valgrind's `>=` rule applied to an operation that
        // allocated nothing.
        engine.record_alloc_guarded(0x1000, Shape::of(4_096), &[0xAA]);
        let after_alloc = engine.stats();
        assert_eq!(after_alloc.curr_bytes, after_alloc.max_bytes);

        let guard =
            crate::internals::guard::enter().expect("this thread is not inside the profiler");
        engine.record_event(&guard, 700, &[0xBB]);
        engine.record_event(&guard, 70, &[0xBB]);
        let stats = engine.stats();

        assert_eq!(stats.total_bytes, after_alloc.total_bytes + 770);
        assert_eq!(stats.total_blocks, after_alloc.total_blocks + 2);
        assert_eq!(
            stats.curr_bytes, after_alloc.curr_bytes,
            "an event became live"
        );
        assert_eq!(stats.curr_blocks, after_alloc.curr_blocks);
        assert_eq!(
            stats.max_bytes, after_alloc.max_bytes,
            "an event that allocated nothing raised the heap peak"
        );
        assert_eq!(
            stats.time_at_max, after_alloc.time_at_max,
            "t-gmax moved to an instant at which the heap did not change"
        );
        assert_eq!(
            stats.epoch, after_alloc.epoch,
            "an event bumped the peak epoch, which would re-snapshot every \
             program point's at-peak columns against a peak that did not happen"
        );
    }

    /// An event must not re-run the lazy-epoch refresh, which would replace a
    /// program point's record of what it held at the peak with what it holds
    /// now.
    ///
    /// Only visible when the two differ, so the block is freed first: the point
    /// then has `at_gmax` of 4,096 and `curr` of 0, and an event committed
    /// against a stale epoch copies the zero over the 4,096. With the two equal
    /// the refresh is idempotent and the epoch cannot be observed at all —
    /// which is why the test above, where live bytes sit at the maximum, cannot
    /// also cover this.
    #[test]
    fn an_event_does_not_resnapshot_what_a_point_held_at_the_peak() {
        let engine = engine();
        let stack = [0xAAusize, 0xBB];
        engine.record_alloc_guarded(0x1000, Shape::of(4_096), &stack);
        engine.record_free(0x1000, 4_096);

        let guard =
            crate::internals::guard::enter().expect("this thread is not inside the profiler");
        engine.record_event(&guard, 700, &stack);
        engine.record_event(&guard, 70, &stack);

        let mut point = Counters::default();
        engine.flush_and_visit(
            Engine::FLUSH_TIMEOUT,
            |_, _, counters| {
                point = *counters;
            },
            |_| {},
            |_| {},
        );
        assert_eq!(
            point.at_gmax_bytes, 4_096,
            "an event replaced the bytes this point held at the peak with the \
             zero it holds now"
        );
        assert_eq!(point.curr_bytes, 0);
        assert_eq!(point.total_bytes, 4_096 + 770);
    }

    /// The clock counts recorded events, and these are recorded events. A run
    /// whose `te` stayed at zero would report every event as simultaneous, and
    /// would make `TimeSource::Events` mean "allocations" in a mode that has
    /// none.
    #[test]
    fn an_event_advances_the_clock() {
        let engine = engine();
        let guard =
            crate::internals::guard::enter().expect("this thread is not inside the profiler");
        let before = engine.clock().events();
        engine.record_event(&guard, 1, &[0xCC]);
        engine.record_event(&guard, 1, &[0xCC]);
        assert_eq!(engine.clock().events(), before + 2);
    }

    /// A mode reaches the shim through `records_allocations`, which is the only
    /// thing standing between a non-heap run and a profile full of the
    /// allocations it was supposed to ignore.
    #[test]
    fn only_a_heap_run_tells_the_shim_to_record() {
        for (mode, records) in [
            (Mode::Heap, true),
            (Mode::AdHoc, false),
            (Mode::Copy, false),
        ] {
            let engine = Engine::with_limits(1 << 12, 1 << 14);
            assert!(
                engine.start(TimeSource::Events, || engine.configure(Settings {
                    mode,
                    ..Settings::default()
                }))
            );
            assert_eq!(engine.mode(), mode);
            assert_eq!(
                engine.records_allocations(),
                records,
                "the shim was told the wrong thing about {mode}"
            );
        }

        // And an engine nobody has started records nothing at all, whatever its
        // mode says.
        let idle = Engine::with_limits(1 << 12, 1 << 14);
        assert!(!idle.records_allocations());
    }

    /// The depth limit reaches the shim through this one relaxed load, and
    /// nothing else reads it. A default that did not match the buffer would cut
    /// every stack short in a profiler nobody configured.
    #[test]
    fn an_unconfigured_engine_records_as_deep_as_the_buffer_allows() {
        assert_eq!(Engine::new().max_depth(), crate::CAPTURE_DEPTH);
        assert!(Engine::new().trim_frames());
    }

    #[test]
    fn a_single_allocation_is_counted_everywhere() {
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(128), &[0xAA, 0xBB]);

        let stats = engine.stats();
        assert_eq!(stats.curr_bytes, 128);
        assert_eq!(stats.curr_blocks, 1);
        assert_eq!(stats.total_bytes, 128);
        assert_eq!(stats.total_blocks, 1);
        assert_eq!(stats.max_bytes, 128);
        assert_eq!(stats.max_blocks, 1);
    }

    #[test]
    fn a_free_returns_live_bytes_but_not_totals() {
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(128), &[0xAA]);
        engine.record_free(0x1000, 128);

        let stats = engine.stats();
        assert_eq!(stats.curr_bytes, 0);
        assert_eq!(stats.curr_blocks, 0);
        assert_eq!(
            stats.total_bytes, 128,
            "cumulative totals must not decrease"
        );
        assert_eq!(stats.max_bytes, 128, "the peak must survive the free");
    }

    #[test]
    fn freeing_an_untracked_pointer_is_ignored() {
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(64), &[0xAA]);
        // A block that predates profiling, or one sampling skipped.
        engine.record_free(0x9999, 4096);

        let stats = engine.stats();
        assert_eq!(
            stats.curr_bytes, 64,
            "an unknown free must not drive live bytes negative"
        );
        assert_eq!(stats.curr_blocks, 1);
    }

    #[test]
    fn the_peak_is_the_high_water_mark() {
        let engine = engine();
        for i in 0..10u64 {
            engine.record_alloc_guarded(0x1000 + i as usize * 64, Shape::of(100), &[0xAA]);
        }
        assert_eq!(engine.stats().max_bytes, 1000);

        for i in 0..10u64 {
            engine.record_free(0x1000 + i as usize * 64, 100);
        }
        assert_eq!(engine.stats().curr_bytes, 0);
        assert_eq!(engine.stats().max_bytes, 1000);

        // A smaller second wave must not lower the recorded peak.
        engine.record_alloc_guarded(0x5000, Shape::of(10), &[0xBB]);
        assert_eq!(engine.stats().max_bytes, 1000);
    }

    /// The `>=` rule: among equal peaks, the latest is recorded, so the epoch
    /// advances even when the maximum does not change.
    #[test]
    fn equal_peaks_advance_the_epoch() {
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(100), &[0xAA]);
        let first = engine.stats().epoch;

        engine.record_free(0x1000, 100);
        engine.record_alloc_guarded(0x2000, Shape::of(100), &[0xAA]);
        let second = engine.stats().epoch;

        assert!(
            second > first,
            "returning to an equal peak must record the later one \
             (epoch {first} -> {second})"
        );
        assert_eq!(engine.stats().max_bytes, 100);
    }

    #[test]
    fn per_point_totals_agree_with_the_global_ones() {
        let engine = engine();
        for i in 0..100usize {
            engine.record_alloc_guarded(0x1000 + i * 64, Shape::of(32), &[i % 7]);
        }

        let mut summed_total = 0u64;
        let mut summed_curr = 0u64;
        engine.flush_and_visit(
            Engine::FLUSH_TIMEOUT,
            |_id, _frames, counters| {
                summed_total += counters.total_bytes;
                summed_curr += counters.curr_bytes;
            },
            |_| {},
            |_| {},
        );

        let stats = engine.stats();
        assert_eq!(summed_total, stats.total_bytes);
        assert_eq!(summed_curr, stats.curr_bytes);
    }

    /// The invariant the peak gate exists to guarantee.
    #[test]
    fn at_peak_values_sum_to_the_peak() {
        let engine = engine();
        // Grow, shrink, and grow again so that the peak sits in the middle and
        // several points are stale by the end.
        for i in 0..50usize {
            engine.record_alloc_guarded(0x1000 + i * 64, Shape::of(100), &[i % 5]);
        }
        for i in 0..40usize {
            engine.record_free(0x1000 + i * 64, 100);
        }
        for i in 0..20usize {
            engine.record_alloc_guarded(0x9000 + i * 64, Shape::of(50), &[i % 3]);
        }

        let mut summed = 0u64;
        engine.flush_and_visit(
            Engine::FLUSH_TIMEOUT,
            |_id, _frames, counters| {
                summed += counters.at_gmax_bytes;
            },
            |_| {},
            |_| {},
        );

        let stats = engine.stats();
        assert_eq!(
            summed, stats.max_bytes,
            "per-point at-peak bytes must sum to the global peak exactly"
        );
    }

    #[test]
    fn realloc_is_attributed_to_the_original_program_point() {
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(100), &[0xAAAA]);
        // Grown from a different call site, as a `Vec::push` would be.
        let taken = engine.live_blocks().remove(0x1000);
        engine.record_realloc_guarded(taken, grew(0x1000, 100, 0x2000, 400), &[0xBBBB]);

        let mut by_point: Vec<(Vec<usize>, u64)> = Vec::new();
        engine.flush_and_visit(
            Engine::FLUSH_TIMEOUT,
            |_id, frames, counters| {
                by_point.push((frames.to_vec(), counters.curr_bytes));
            },
            |_| {},
            |_| {},
        );

        let original = by_point.iter().find(|(f, _)| f == &[0xAAAA]).unwrap();
        assert_eq!(
            original.1, 400,
            "the resize should belong to the point that made the original allocation"
        );
        assert!(
            !by_point.iter().any(|(f, _)| f == &[0xBBBB]),
            "the resizing call site should not have acquired the bytes"
        );
    }

    #[test]
    fn realloc_counts_as_a_new_block_for_totals_but_not_for_live_blocks() {
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(100), &[0xAA]);
        let taken = engine.live_blocks().remove(0x1000);
        engine.record_realloc_guarded(taken, grew(0x1000, 100, 0x2000, 400), &[0xAA]);

        let stats = engine.stats();
        assert_eq!(stats.curr_blocks, 1, "a realloc does not add a live block");
        assert_eq!(stats.total_blocks, 2, "a realloc counts toward the totals");
        assert_eq!(stats.curr_bytes, 400);
        assert_eq!(stats.total_bytes, 500);
    }

    #[test]
    fn a_shrinking_realloc_reduces_live_bytes() {
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(1000), &[0xAA]);
        let taken = engine.live_blocks().remove(0x1000);
        engine.record_realloc_guarded(taken, grew(0x1000, 1000, 0x1000, 100), &[0xAA]);

        let stats = engine.stats();
        assert_eq!(stats.curr_bytes, 100);
        assert_eq!(stats.max_bytes, 1000, "the pre-shrink peak must be kept");
    }

    #[test]
    fn realloc_of_an_untracked_pointer_becomes_a_plain_allocation() {
        let engine = engine();
        let taken = engine.live_blocks().remove(0xDEAD);
        engine.record_realloc_guarded(taken, grew(0xDEAD, 100, 0x2000, 400), &[0xCC]);

        let stats = engine.stats();
        assert_eq!(stats.curr_bytes, 400);
        assert_eq!(stats.curr_blocks, 1);
        assert!(engine.live_blocks().get(0x2000).is_some());
    }

    // ---- what the program asked for ----

    /// The invariant the native format's validator checks, and the reason the
    /// shape is counted before the live-block table is consulted rather than
    /// after: a request the table had no room for still happened.
    #[test]
    fn every_observed_request_is_either_recorded_or_dropped() {
        // One shard's worth of room, so the table fills and the rest are
        // dropped while every request is still observed.
        let engine = Engine::with_limits(1 << 12, 64);
        assert!(engine.start(TimeSource::Events, || {}));
        for i in 0..crate::internals::miri_scale(4_000) {
            engine.record_alloc_guarded(0x8000_0000 + i * 4096, Shape::of(64), &[0xAA]);
        }

        let stats = engine.stats();
        let shapes = engine.shapes();
        assert!(
            stats.dropped_blocks > 0,
            "the table never filled, so this test proves nothing"
        );
        assert_eq!(
            shapes.observed_blocks,
            stats.total_blocks + stats.dropped_blocks,
            "an allocation the table had no room for was left out of the \
             histograms, which describe what the program asked for rather than \
             what this profiler managed to track"
        );
        assert_eq!(shapes.sizes.iter().sum::<u64>(), shapes.observed_blocks);
        assert_eq!(
            shapes.alignments.iter().sum::<u64>(),
            shapes.observed_blocks
        );
    }

    #[test]
    fn the_shape_of_a_request_reaches_the_histograms() {
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(24).aligned(8), &[0xAA]);
        engine.record_alloc_guarded(0x2000, Shape::of(4096).aligned(64).zeroed(), &[0xBB]);

        let shapes = engine.shapes();
        assert_eq!(shapes.observed_blocks, 2);
        assert_eq!(
            shapes.size_classes().collect::<Vec<_>>(),
            [(16, 31, 1), (4096, 8191, 1)]
        );
        assert_eq!(
            shapes.alignments_used().collect::<Vec<_>>(),
            [(8, 1), (64, 1)]
        );
        assert_eq!(shapes.zeroed_blocks, 1);
        assert_eq!(
            shapes.zeroed_bytes, 4096,
            "a zeroed block's bytes are what a reader compares against the \
             process's resident size"
        );
    }

    /// A reallocation is two facts, and the engine records both: the cost of the
    /// move, and a block of the new shape. `total_blocks` counts it as a block,
    /// so the histograms have to as well or they stop summing to it.
    #[test]
    fn a_realloc_counts_its_copy_and_its_resulting_block() {
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(100).aligned(8), &[0xAA]);
        let taken = engine.live_blocks().remove(0x1000);
        engine.record_realloc_guarded(
            taken,
            Realloc {
                old_address: 0x1000,
                old_size: 100,
                new_address: 0x2000,
                new: Shape::of(400).aligned(8),
            },
            &[0xAA],
        );

        let shapes = engine.shapes();
        assert_eq!(shapes.reallocs, 1);
        assert_eq!(shapes.reallocs_moved, 1);
        assert_eq!(shapes.bytes_copied, 100, "the move copied what was there");
        assert_eq!(shapes.bytes_grown, 300);
        assert_eq!(
            shapes.observed_blocks,
            engine.stats().total_blocks,
            "a reallocation adds one block to the totals, so it must add \
             exactly one to the histograms"
        );
    }

    /// The one path that counts a shape and drops a block: a reallocation whose
    /// new entry the live-block table has no room for.
    ///
    /// `every_observed_request_is_either_recorded_or_dropped` fills the table
    /// through `record_alloc` only, so this path was outside the invariant it
    /// checks — and moving the shape below the failure return left both green.
    #[test]
    fn a_realloc_the_table_cannot_track_is_observed_and_dropped() {
        // One shard's worth of room, filled before the reallocation, so the
        // insert of the new address is refused.
        let engine = Engine::with_limits(1 << 12, 64);
        assert!(engine.start(TimeSource::Events, || {}));
        for i in 0..crate::internals::miri_scale(4_000) {
            engine.record_alloc_guarded(0x8000_0000 + i * 4096, Shape::of(64), &[0xAA]);
        }
        let before = engine.shapes();
        let stats_before = engine.stats();
        assert!(
            stats_before.dropped_blocks > 0,
            "the table never filled, so this test proves nothing"
        );

        // A tracked block, reallocated to an address the full table refuses.
        let tracked = engine
            .live_blocks()
            .remove(0x8000_0000)
            .expect("the first block was tracked");
        engine.record_realloc_guarded(
            Some(tracked),
            grew(0x8000_0000, 64, 0xDEAD_0000, 128),
            &[0xAA],
        );

        let shapes = engine.shapes();
        let stats = engine.stats();
        assert_eq!(
            shapes.observed_blocks,
            before.observed_blocks + 1,
            "the reallocation's resulting block was not observed"
        );
        assert_eq!(
            stats.dropped_blocks,
            stats_before.dropped_blocks + 1,
            "the refused insert was not counted as a drop"
        );
        assert_eq!(
            stats.total_blocks, stats_before.total_blocks,
            "a reallocation the table could not track must not count toward the \
             totals, or the block would never come back down"
        );
        assert_eq!(
            shapes.observed_blocks,
            stats.total_blocks + stats.dropped_blocks,
            "the invariant the native format's validator checks"
        );
    }

    /// The `taken == None` path forwards to `record_alloc`, which counts the
    /// shape itself. Counting it here as well would double every untracked
    /// reallocation.
    #[test]
    fn a_realloc_of_an_untracked_block_is_counted_once() {
        let engine = engine();
        let taken = engine.live_blocks().remove(0xDEAD);
        engine.record_realloc_guarded(taken, grew(0xDEAD, 100, 0x2000, 400), &[0xCC]);

        let shapes = engine.shapes();
        assert_eq!(shapes.observed_blocks, 1);
        assert_eq!(shapes.sizes.iter().sum::<u64>(), 1);
        assert_eq!(
            shapes.reallocs, 1,
            "an untracked block was still reallocated, and the copy still \
             happened"
        );
    }

    /// Nothing outside a heap run reaches the shim, so an ad hoc run's
    /// histograms have to stay empty rather than describing events as though
    /// they were blocks.
    #[test]
    fn an_event_has_no_shape() {
        let engine = Engine::with_limits(1 << 12, 1 << 14);
        assert!(
            engine.start(TimeSource::Events, || engine.configure(Settings {
                mode: Mode::AdHoc,
                ..Settings::default()
            }))
        );
        let guard =
            crate::internals::guard::enter().expect("this thread is not inside the profiler");
        engine.record_event(&guard, 5_000, &[0xAA]);

        assert_eq!(engine.stats().total_blocks, 1);
        assert_eq!(
            engine.shapes(),
            ShapeStats::default(),
            "an ad hoc event has no size, no alignment and nothing to zero"
        );
    }

    #[test]
    fn block_lifetimes_are_accumulated() {
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(10), &[0xAA]);
        for i in 0..9usize {
            engine.record_alloc_guarded(0x2000 + i * 8, Shape::of(10), &[0xBB]);
        }
        engine.record_free(0x1000, 10);

        let mut lifetime = 0u64;
        engine.flush_and_visit(
            Engine::FLUSH_TIMEOUT,
            |_id, frames, counters| {
                if frames == [0xAA] {
                    lifetime = counters.total_lifetime;
                }
            },
            |_| {},
            |_| {},
        );
        assert!(
            lifetime >= 9,
            "a block that survived nine later events should record a lifetime of \
             at least nine, got {lifetime}"
        );
    }

    /// A sampled block's lifetime counts for all the blocks it stands for.
    ///
    /// `tl` divided by the block count is the average-lifetime column, which is
    /// how a reader finds short-lived churn. Sampling scales both halves, so the
    /// column survives; scaling only the count would deflate it by exactly the
    /// sampling ratio, and *that* is the error that would read as a finding about
    /// the program rather than about the profiler.
    #[test]
    fn a_sampled_lifetime_counts_for_the_blocks_it_stands_for() {
        const SIZE: usize = 64;
        const INTERVAL: u64 = 1 << 16;

        let engine = sampled(INTERVAL);
        let scale = super::super::sampler::weighted_blocks(SIZE, Some(INTERVAL));
        assert!(
            scale > 100,
            "this interval should make one sample stand for many blocks, not {scale}"
        );

        // One sampled block at a program point of its own, then a second
        // somewhere else so that the clock advances past the first before it
        // dies. In `Events` mode the clock counts recorded allocations, so the
        // first block's lifetime is at least one.
        let first = record_until_sampled(&engine, 0x1_0000, SIZE, &[0xAA]);
        record_until_sampled(&engine, 0x400_0000, SIZE, &[0xBB]);
        engine.record_free(first, SIZE);

        let mut lifetime = 0u64;
        engine.flush_and_visit(
            Engine::FLUSH_TIMEOUT,
            |_id, frames, counters| {
                if frames == [0xAA] {
                    lifetime = counters.total_lifetime;
                }
            },
            |_| {},
            |_| {},
        );

        assert!(
            lifetime >= scale,
            "a sampled block standing for {scale} blocks recorded a lifetime of \
             {lifetime}, which is what one unscaled block would record"
        );
    }

    #[test]
    fn a_second_start_is_refused() {
        let engine = Engine::with_limits(64, 64);
        assert!(engine.start(TimeSource::Events, || {}));
        assert!(
            !engine.start(TimeSource::Events, || {}),
            "two profilers must not attach to one engine"
        );
    }

    #[test]
    fn stopping_flips_state_before_output() {
        let engine = engine();
        assert!(engine.is_running());
        engine.stop(Shutdown::Explicit);
        assert!(!engine.is_running());
        assert_eq!(engine.state(), State::Finished);
    }

    #[test]
    fn a_full_live_table_drops_events_rather_than_inflating_live_bytes() {
        let engine = Engine::with_limits(1 << 10, super::super::live::SHARDS * 8);
        assert!(engine.start(TimeSource::Events, || {}));

        #[cfg(miri)]
        const ATTEMPTS: usize = 2_000;
        #[cfg(not(miri))]
        const ATTEMPTS: usize = 100_000;
        for i in 0..ATTEMPTS {
            engine.record_alloc_guarded(0x10_0000 + i * 64, Shape::of(16), &[0xAA]);
        }
        let stats = engine.stats();
        assert!(stats.dropped_blocks > 0, "the ceiling was never reached");
        assert_eq!(
            stats.curr_bytes,
            stats.curr_blocks * 16,
            "live bytes and live blocks disagree, so a partly-recorded event slipped through"
        );
    }

    /// The whole point of the gate. Every thread allocates and frees; at the
    /// end, the per-point at-peak values must sum to exactly the global peak.
    #[test]
    fn concurrent_traffic_keeps_the_peak_exact() {
        #[cfg(miri)]
        const ROUNDS: usize = 20;
        #[cfg(not(miri))]
        const ROUNDS: usize = 2_000;
        const THREADS: usize = 8;

        let engine = Engine::with_limits(1 << 12, 1 << 16);
        assert!(engine.start(TimeSource::Events, || {}));

        std::thread::scope(|s| {
            for t in 0..THREADS {
                let engine = &engine;
                s.spawn(move || {
                    let base = 0x1_0000_0000usize + t * 0x1000_0000;
                    for i in 0..ROUNDS {
                        let address = base + i * 64;
                        engine.record_alloc_guarded(address, Shape::of(64), &[t, i % 4]);
                        if i % 3 == 0 {
                            engine.record_free(address, 64);
                        }
                    }
                });
            }
        });

        let mut summed_at_peak = 0u64;
        let mut summed_curr = 0u64;
        let mut summed_total = 0u64;
        engine.flush_and_visit(
            Engine::FLUSH_TIMEOUT,
            |_id, _frames, counters| {
                summed_at_peak += counters.at_gmax_bytes;
                summed_curr += counters.curr_bytes;
                summed_total += counters.total_bytes;
            },
            |_| {},
            |_| {},
        );

        let stats = engine.stats();
        assert_eq!(summed_curr, stats.curr_bytes, "live bytes drifted");
        assert_eq!(summed_total, stats.total_bytes, "cumulative bytes drifted");
        assert_eq!(
            summed_at_peak, stats.max_bytes,
            "per-point at-peak bytes did not sum to the global peak; \
             this is exactly the failure the peak gate exists to prevent"
        );
    }

    /// Growth with no frees at all: every allocation is a new peak, so every one
    /// takes the exclusive path. The worst case for the gate, and the one most
    /// likely to expose an ordering bug.
    #[test]
    fn monotonic_growth_stays_exact_under_contention() {
        #[cfg(miri)]
        const ROUNDS: usize = 20;
        #[cfg(not(miri))]
        const ROUNDS: usize = 1_000;
        const THREADS: usize = 8;

        let engine = Engine::with_limits(1 << 12, 1 << 16);
        assert!(engine.start(TimeSource::Events, || {}));

        std::thread::scope(|s| {
            for t in 0..THREADS {
                let engine = &engine;
                s.spawn(move || {
                    let base = 0x2_0000_0000usize + t * 0x1000_0000;
                    for i in 0..ROUNDS {
                        engine.record_alloc_guarded(base + i * 64, Shape::of(128), &[t]);
                    }
                });
            }
        });

        let stats = engine.stats();
        assert_eq!(stats.curr_bytes, (THREADS * ROUNDS * 128) as u64);
        assert_eq!(
            stats.max_bytes, stats.curr_bytes,
            "with no frees the peak must equal the final live bytes"
        );

        let mut summed = 0u64;
        engine.flush_and_visit(
            Engine::FLUSH_TIMEOUT,
            |_id, _frames, counters| summed += counters.at_gmax_bytes,
            |_| {},
            |_| {},
        );
        assert_eq!(summed, stats.max_bytes);
    }

    /// The `fork` prepare/parent contract, tested without forking.
    ///
    /// This exists because the fork *tests* cannot see it. A child resets every
    /// lock unconditionally, so it recovers whether or not `prepare` ran, and
    /// deleting the bodies of both handlers left the entire lifecycle suite
    /// green. What `prepare` actually buys is that the child inherits tables
    /// nobody was midway through updating — a `Snapshot::capture` in a child
    /// reads an `ArenaVec` length and indexes with it — and that is a property
    /// no observation of the child can distinguish from luck.
    ///
    /// So the contract is checked directly: while `prepare` is in force, a
    /// recording thread must be unable to make progress, and after `parent` it
    /// must.
    ///
    /// Unix only, because that is where the thing under test exists: these
    /// handlers serve `pthread_atfork`, `core::fork` compiles its handlers under
    /// `cfg(unix)`, and its Windows `register` is a no-op — so `fork_prepare` is
    /// never called on Windows and there is no behaviour there to pin.
    ///
    /// It also hangs the Windows test binary under Wine when the harness runs
    /// tests in parallel, while passing serially and passing in parallel once
    /// this test is skipped. That is not root-caused: a thread parked on an
    /// `SRWLOCK` while the harness schedules other tests is Wine's business, and
    /// chasing it would be chasing the behaviour of an unreachable code path on
    /// an emulator. Recorded rather than quietly worked around.
    #[test]
    #[cfg(unix)]
    #[cfg_attr(miri, ignore = "spawns a thread that blocks on a real lock")]
    fn fork_prepare_stops_recording_threads_and_fork_parent_releases_them() {
        use std::sync::mpsc;
        use std::time::Duration;

        static ENGINE: Engine = Engine::new();
        ENGINE.start(TimeSource::Events, || {});
        // A point already interned, so the blocked thread below is not the one
        // paying for the table's first growth.
        ENGINE.record_alloc_guarded(0x1000, Shape::of(64), &[0xAA]);

        // SAFETY: paired with `fork_parent` on this thread below. No `fork`
        // happens in between, which the handlers do not require.
        unsafe { ENGINE.fork_prepare() };

        let (sender, receiver) = mpsc::channel();
        let recorder = std::thread::spawn(move || {
            ENGINE.record_alloc_guarded(0x2000, Shape::of(128), &[0xBB]);
            let _ = sender.send(());
        });

        assert!(
            receiver.recv_timeout(Duration::from_millis(500)).is_err(),
            "a thread recorded an allocation while the fork prepare handler \
             held every lock, so the child of a real fork would inherit a table \
             mid-update"
        );

        // SAFETY: this thread ran `fork_prepare` immediately above.
        unsafe { ENGINE.fork_parent() };

        assert!(
            receiver.recv_timeout(Duration::from_secs(10)).is_ok(),
            "the recording thread never woke after the parent handler released \
             the locks"
        );
        recorder.join().expect("the recording thread panicked");
        ENGINE.stop(Shutdown::Explicit);
    }

    // ---- restarting the counts ------------------------------------------

    /// The lock every test that poisons holds, for a test to hold from its
    /// first line to its last.
    ///
    /// A restart refuses a poisoned profiler, and the poison flag is
    /// process-wide: a test elsewhere in this binary that poisons on purpose
    /// would otherwise make a restart here fail, or a later `is_poisoned`
    /// assertion here fail, for a reason the test has nothing to do with.
    /// Holding it around the restart alone was not enough, because these tests
    /// go on to assert the flag is clear after the restart has released it.
    fn quiet() -> super::super::lock::RawGuard<'static> {
        super::super::diagnostic::POISON_TESTS.lock()
    }

    /// Restarts `engine`'s counts as the program would. The caller holds
    /// [`quiet`].
    fn reset(engine: &Engine) {
        engine
            .reset_guarded()
            .expect("a running engine restarts its counts");
    }

    /// Every program point the engine would put in a profile, keyed by frames.
    fn points(engine: &Engine) -> std::collections::BTreeMap<Vec<usize>, Counters> {
        let mut points = std::collections::BTreeMap::new();
        let flush = engine.flush_and_visit(
            Engine::FLUSH_TIMEOUT,
            |_id, frames, counters| {
                points.insert(frames.to_vec(), *counters);
            },
            |_| {},
            |_| {},
        );
        assert!(flush.exclusive, "a quiet engine could not be flushed");
        points
    }

    /// The whole rule, on one run: what happened before the restart is
    /// forgotten, and what is live at it is kept.
    ///
    /// The run is shaped so that every class of counter has a different answer
    /// before and after: a block held across the restart, one freed before it,
    /// a transient peak above the level the restart finds, and a site that
    /// allocates only afterwards.
    #[test]
    fn a_restart_forgets_what_happened_and_keeps_what_is() {
        let _quiet = quiet();
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(4_096), &[0xAA]);
        engine.record_alloc_guarded(0x2000, Shape::of(1_024), &[0xBB]);
        engine.record_free(0x2000, 1_024);
        engine.record_alloc_guarded(0x3000, Shape::of(512), &[0xCC]);
        let before = engine.stats();
        assert_eq!(before.max_bytes, 5_120, "the warm-up's transient peak");
        assert_eq!(before.curr_bytes, 4_608);

        reset(&engine);

        let after = engine.stats();
        assert_eq!((after.total_bytes, after.total_blocks), (0, 0));
        assert_eq!(
            (after.curr_bytes, after.curr_blocks),
            (4_608, 2),
            "a restart moved what is live"
        );
        assert_eq!(
            (after.max_bytes, after.max_blocks),
            (4_608, 2),
            "the peak did not start again from what is live"
        );
        assert_eq!(after.time_at_max, engine.clock().now(TimeSource::Events));
        assert_eq!(after.epoch, before.epoch + 1, "a restart is a peak");

        let reset_record = engine.last_reset().expect("the restart is recorded");
        assert_eq!(reset_record.count, 1);
        assert_eq!(reset_record.at, after.time_at_max);
        assert_eq!(
            (reset_record.carried_bytes, reset_record.carried_blocks),
            (4_608, 2)
        );
        assert_eq!(reset_record.epoch, after.epoch);

        let held = points(&engine);
        assert_eq!(
            held.keys().cloned().collect::<Vec<_>>(),
            vec![vec![0xAA], vec![0xCC]],
            "the point whose block was freed before the restart is still \
             reported, or one holding a carried block is not"
        );
        let aa = held[&vec![0xAA]];
        assert_eq!(
            aa,
            Counters {
                total_bytes: 0,
                total_blocks: 0,
                total_lifetime: 0,
                curr_bytes: 4_096,
                curr_blocks: 1,
                max_bytes: 4_096,
                max_blocks: 1,
                at_gmax_bytes: 4_096,
                at_gmax_blocks: 1,
            }
        );

        // The window's own events. A carried block freed: live bytes come down
        // by exactly its size, and it brings no lifetime with it, because it is
        // in none of the window's allocation counts.
        engine.record_alloc_guarded(0x4000, Shape::of(8_192), &[0xDD]);
        engine.record_alloc_guarded(0x5000, Shape::of(16), &[0xEE]);
        engine.record_free(0x1000, 4_096);
        engine.record_free(0x4000, 8_192);

        let end = engine.stats();
        assert_eq!(end.curr_bytes, 512 + 16);
        assert_eq!((end.total_bytes, end.total_blocks), (8_192 + 16, 2));
        assert_eq!(end.max_bytes, 4_608 + 8_192 + 16, "the window's own peak");

        let window = points(&engine);
        let aa = window[&vec![0xAA]];
        assert_eq!(
            aa.curr_bytes, 0,
            "the carried block's free missed its point"
        );
        assert_eq!(
            aa.total_lifetime, 0,
            "a carried block brought a lifetime into a point that allocated \
             nothing in the window, so its average lifetime is a division of \
             something by zero"
        );
        assert_eq!(aa.at_gmax_bytes, 4_096, "it was live at the window's peak");
        let dd = window[&vec![0xDD]];
        assert_eq!(
            dd.total_lifetime, 1,
            "a block of the window's own lived one event and recorded otherwise"
        );
        let at_peak: u64 = window.values().map(|counters| counters.at_gmax_bytes).sum();
        assert_eq!(at_peak, end.max_bytes, "the at-peak columns do not sum");
        assert!(!super::super::diagnostic::is_poisoned());
    }

    /// The at-peak columns have to describe the restart for a point the window
    /// never touches, which only the epoch can arrange: the lazy refresh that
    /// would otherwise copy a stale snapshot happens on a touch that never
    /// comes. And the `>=` rule has to apply to the level the restart set, so
    /// returning to it is an equal peak that moves the snapshot to now.
    #[test]
    fn a_restart_is_a_peak_the_lazy_scheme_honours() {
        let _quiet = quiet();
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(1_000), &[0xAA]);
        engine.record_alloc_guarded(0x2000, Shape::of(3_000), &[0xBB]);
        engine.record_free(0x2000, 3_000);
        engine.record_alloc_guarded(0x3000, Shape::of(500), &[0xCC]);
        // AA was last touched at an epoch whose snapshot is the 4,000-byte peak.
        reset(&engine);

        let restarted = points(&engine);
        assert_eq!(restarted[&vec![0xAA]].at_gmax_bytes, 1_000);
        assert_eq!(restarted[&vec![0xCC]].at_gmax_bytes, 500);

        // Back to exactly the restart's level, held by different points.
        engine.record_free(0x3000, 500);
        engine.record_alloc_guarded(0x4000, Shape::of(500), &[0xDD]);
        let equal = points(&engine);
        assert_eq!(engine.stats().max_bytes, 1_500);
        assert_eq!(
            equal
                .get(&vec![0xCC])
                .map_or(0, |counters| counters.at_gmax_bytes),
            0,
            "an equal peak after a restart must record the latest instant"
        );
        assert_eq!(equal[&vec![0xDD]].at_gmax_bytes, 500);
    }

    /// A reallocation of a carried block ends a block the window never counted
    /// and begins one it does: no lifetime from the old one, a block and its
    /// bytes in the totals, and a lifetime for the new one when it goes.
    #[test]
    fn a_carried_block_reallocated_after_a_restart_is_a_new_block_of_the_window() {
        let _quiet = quiet();
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(100), &[0xAA]);
        for i in 0..5usize {
            engine.record_alloc_guarded(0x8000 + i * 64, Shape::of(8), &[0xBB]);
        }
        reset(&engine);

        let taken = engine.live_blocks().remove(0x1000);
        assert!(taken.is_some_and(|block| block.is_carried()));
        engine.record_realloc_guarded(taken, grew(0x1000, 100, 0x2000, 300), &[0xCC]);

        let point = points(&engine)[&vec![0xAA]];
        assert_eq!((point.total_bytes, point.total_blocks), (300, 1));
        assert_eq!(
            point.total_lifetime, 0,
            "the carried half brought a lifetime"
        );
        assert_eq!(point.curr_bytes, 300);
        assert!(
            engine
                .live_blocks()
                .get(0x2000)
                .is_some_and(|block| !block.is_carried()),
            "the resized block is the window's, and must be born in it"
        );

        engine.record_alloc_guarded(0x9000, Shape::of(8), &[0xBB]);
        engine.record_free(0x2000, 300);
        assert_eq!(points(&engine)[&vec![0xAA]].total_lifetime, 1);
    }

    /// The rows and the histograms restart with the totals they are checked
    /// against, so the window's profile still sums: the rows to the totals
    /// and the live figures, and the requests to the blocks recorded.
    #[test]
    fn a_restart_restarts_the_rows_and_the_shapes_with_the_totals() {
        let _quiet = quiet();
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(4_096), &[0xAA]);
        engine.record_alloc_guarded(0x2000, Shape::of(64), &[0xBB]);
        engine.record_free(0x2000, 64);
        reset(&engine);

        let (bytes, blocks, live) = thread_totals(&engine);
        assert_eq!((bytes, blocks), (0, 0));
        assert_eq!(live, 4_096, "the row lost its carried block");
        assert_eq!(engine.shapes(), ShapeStats::default());

        engine.record_alloc_guarded(0x3000, Shape::of(32), &[0xCC]);
        engine.record_free(0x1000, 4_096);
        let stats = engine.stats();
        let (bytes, blocks, live) = thread_totals(&engine);
        assert_eq!((bytes, blocks, live), (32, 1, 32));
        assert_eq!(
            (stats.total_bytes, stats.total_blocks, stats.curr_bytes),
            (bytes, blocks, live)
        );
        assert_eq!(
            engine.shapes().observed_blocks,
            stats.total_blocks + stats.dropped_blocks
                - engine.last_reset().expect("restarted").dropped_blocks,
            "the window's requests and its recorded blocks disagree"
        );
    }

    /// An event run has totals and nothing live, and restarts the totals.
    /// Refused events restart too: they qualify the totals, not the run.
    #[test]
    fn an_event_run_restarts_its_totals() {
        let _quiet = quiet();
        let engine = Engine::with_limits(1 << 10, 1 << 12);
        assert!(
            engine.start(TimeSource::Events, || engine.configure(Settings {
                mode: Mode::AdHoc,
                ..Settings::default()
            }))
        );
        let guard = super::super::guard::enter().expect("not inside the profiler");
        engine.record_event(&guard, 40, &[0xAA]);
        engine.refuse_event();
        drop(guard);

        reset(&engine);
        let stats = engine.stats();
        assert_eq!((stats.total_bytes, stats.total_blocks), (0, 0));
        assert_eq!(stats.refused_events, 0);
        assert!(
            points(&engine).is_empty(),
            "an emptied event point was reported"
        );

        let guard = super::super::guard::enter().expect("not inside the profiler");
        engine.record_event(&guard, 7, &[0xAA]);
        drop(guard);
        assert_eq!(points(&engine)[&vec![0xAA]].total_bytes, 7);
    }

    /// A sampled block carried across a restart is freed with the weight it
    /// was recorded with, so live bytes come back to exactly zero rather than
    /// drifting by a rounding error, and its estimate brings no lifetime.
    #[test]
    fn a_sampled_run_restarts_and_still_balances() {
        let _quiet = quiet();
        const SIZE: usize = 256;
        let engine = sampled(4_096);
        let address = record_until_sampled(&engine, 0x10_0000, SIZE, &[0xAA]);
        let held = engine.stats().curr_bytes;
        assert!(held > 0);

        reset(&engine);
        assert_eq!(engine.stats().curr_bytes, held);
        engine.record_free(address, SIZE);

        let stats = engine.stats();
        assert_eq!((stats.curr_bytes, stats.curr_blocks), (0, 0));
        let point = points(&engine)[&vec![0xAA]];
        assert_eq!(point.total_lifetime, 0);
        assert_eq!(point.max_bytes, held);
        assert!(!super::super::diagnostic::is_poisoned());
    }

    /// A restart is refused, with nothing changed, wherever the counts are not
    /// this process's to restart.
    #[test]
    fn a_restart_is_refused_where_the_counts_are_not_ours() {
        let _quiet = super::super::diagnostic::POISON_TESTS.lock();

        let idle = Engine::with_limits(1 << 10, 1 << 12);
        assert_eq!(idle.reset_guarded(), Err(ResetError::Stopped));

        let stopped = engine();
        stopped.record_alloc_guarded(0x1000, Shape::of(64), &[0xAA]);
        stopped.stop(Shutdown::Explicit);
        let before = stopped.stats();
        assert_eq!(stopped.reset_guarded(), Err(ResetError::Stopped));
        assert_eq!(
            stopped.stats(),
            before,
            "a refused restart changed the counts"
        );
        assert_eq!(stopped.last_reset(), None);

        let child = engine();
        child.record_alloc_guarded(0x1000, Shape::of(64), &[0xAA]);
        child.disown_for_testing();
        let before = child.stats();
        assert_eq!(child.reset_guarded(), Err(ResetError::ForkedChild));
        assert_eq!(child.stats(), before);
    }

    /// A poisoned profiler is not restarted: what is live is what it keeps,
    /// and the failure was in what it holds.
    #[test]
    fn a_poisoned_engine_is_not_restarted() {
        let _quiet = super::super::diagnostic::POISON_TESTS.lock();
        super::super::diagnostic::reset();
        super::super::diagnostic::set_quiet(true);

        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(64), &[0xAA]);
        let before = engine.stats();
        super::super::diagnostic::poison("test: a restart must refuse this");
        let refused = engine.reset_guarded();
        super::super::diagnostic::reset();

        assert_eq!(refused, Err(ResetError::Poisoned));
        assert_eq!(engine.stats(), before);
    }

    /// A reading on a thread inside the profiler refuses instead of waiting
    /// out a restart, because the restart may be running on that very thread,
    /// interrupted by the signal handler that is reading. Elsewhere, and once
    /// the restart is done, it reads as ever.
    #[test]
    fn a_reading_inside_the_profiler_refuses_rather_than_wait_for_a_restart() {
        let _quiet = quiet();
        let engine = engine();
        let sequence = engine.reset_sequence.load(Ordering::Relaxed);
        assert_eq!(engine.read_window(|| ()), Some((sequence / 2, ())));

        // What a restart on this thread looks like to a handler that
        // interrupts it: the sequence odd, and the guard held.
        engine.reset_sequence.store(sequence + 1, Ordering::Relaxed);
        let guard = super::super::guard::enter().expect("the test thread can enter");
        assert_eq!(
            engine.read_window(|| ()),
            None,
            "a reading inside the profiler waited for a restart"
        );
        drop(guard);
        engine.reset_sequence.store(sequence + 2, Ordering::Release);

        assert_eq!(engine.read_window(|| ()), Some((sequence / 2 + 1, ())));
    }

    /// A restart that unwinds partway, and is caught, still closes its window:
    /// the sequence is even again, so readings return instead of waiting for
    /// ever, and the run is poisoned, so what they return is a refusal and
    /// not half a restart.
    #[test]
    fn an_interrupted_restart_does_not_leave_readings_waiting() {
        let _quiet = quiet();
        super::super::diagnostic::reset();
        super::super::diagnostic::set_quiet(true);
        let engine = engine();
        let sequence = engine.reset_sequence.load(Ordering::Relaxed);

        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.reset_sequence.store(sequence + 1, Ordering::Relaxed);
            let _restarting = Restarting {
                sequence: &engine.reset_sequence,
                even: sequence + 2,
            };
            // Not `panic!`, which would print through the hook.
            std::panic::resume_unwind(Box::new("a restart failed partway"));
        }));
        let poisoned = super::super::diagnostic::is_poisoned();
        super::super::diagnostic::reset();

        assert!(unwound.is_err());
        assert!(poisoned, "a half-done restart left the run unpoisoned");
        assert_eq!(engine.read_window(|| ()), Some((sequence / 2 + 1, ())));
    }

    /// Restarting twice is restarting from the second one: the record counts
    /// both and describes the last, and the peaks the profile reports count
    /// from it.
    #[test]
    fn a_second_restart_describes_the_window_since_it() {
        let _quiet = quiet();
        let engine = engine();
        engine.record_alloc_guarded(0x1000, Shape::of(100), &[0xAA]);
        reset(&engine);
        engine.record_alloc_guarded(0x2000, Shape::of(200), &[0xBB]);
        let peaks_in_first_window = engine.stats().epoch - engine.last_reset().unwrap().epoch;
        assert_eq!(peaks_in_first_window, 1);
        reset(&engine);

        let record = engine.last_reset().unwrap();
        assert_eq!(record.count, 2);
        assert_eq!((record.carried_bytes, record.carried_blocks), (300, 2));
        assert_eq!(engine.stats().epoch, record.epoch);
        assert_eq!(engine.resets(), 2);
        let window = points(&engine);
        assert!(window.values().all(|counters| counters.total_blocks == 0));
    }
}
