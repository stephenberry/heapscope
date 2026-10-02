//! Reading the counters from a test, and failing a test on them.
//!
//! A profile is something a person reads. This module is for the other case: a
//! number a *program* reads, so that "this parser allocates at most 64 KiB" can
//! be a check that runs on every commit rather than a thing someone measured
//! once and wrote in a comment.
//!
//! ```no_run
//! # fn parse(_: &str) {}
//! #[global_allocator]
//! static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
//!
//! #[test]
//! fn parsing_stays_inside_its_budget() {
//!     let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
//!     parse("...");
//!     heapscope::assert_max_bytes!(64 * 1024);
//! }
//! ```
//!
//! # Every reading can refuse, and that is the design
//!
//! [`HeapStats::get`] returns a [`Result`], and the assertion macros fail rather
//! than pass whenever the answer would be a guess. The alternative — a getter
//! that returns zeros when nothing is recording — turns every assertion built on
//! it into one that **cannot fail**: a test whose profiler was never started, or
//! was started in the wrong mode, would report a peak of zero and pass every
//! budget in the file. This crate has met that shape of defect repeatedly (see
//! PLAN.md section 9.1), and a testing API is the worst place to meet it again,
//! because the whole point of the thing is to notice.
//!
//! So the refusals are:
//!
//! | Condition | Why the numbers would be a guess |
//! |---|---|
//! | Nothing is recording | There is no run, so there is nothing to assert about |
//! | The run counts something else | An ad hoc run has no heap peak, and a heap run has no event weights |
//! | The profiler was poisoned | It stopped recording partway through and says so |
//! | This process is a `fork` child | The counters were inherited and describe the parent's run |
//! | The run samples | Its counters are estimates, which move between runs of the same program |
//! | The run dropped blocks | The live-block table filled; the totals are missing however many it turned away |
//!
//! The last one is only a refusal for the *assertions*, not for [`HeapStats::get`]:
//! the count is a field on the reading, so a caller who wants the numbers with
//! their caveat can have them. An assertion cannot carry a caveat — it passes or
//! it fails — so it declines to draw a confident conclusion from an incomplete
//! measurement. Raise the ceiling with
//! [`max_live_blocks`](crate::ProfilerBuilder::max_live_blocks) and the numbers
//! become assertable again.
//!
//! [`RegionBreakdown::get`] refuses on the same terms, with two differences
//! that follow from what it reads. It answers in every mode, because a region
//! row means the same thing in each — what was recorded while that region was
//! innermost — and the reading says which mode it came from. And it has one
//! refusal of its own, [`StatsError::NoQuietPoint`]: its row outside every
//! region is a difference between the totals and the region rows, which is
//! only a measurement when both are read at one instant.
//!
//! # The condition this table did not list
//!
//! A program whose `#[global_allocator]` is not [`Alloc`](crate::Alloc) records
//! nothing, so every figure is zero — and a zero peak passes every budget in the
//! file. That is the cannot-fail shape above, reached by a route none of the
//! rows covers, and for a while it was reachable: `assert_max_bytes!(64 * 1024)`
//! passed in a program that had just allocated 10 MiB.
//!
//! It is not another row, because a reading is the wrong place to catch it. By
//! then the run is over and the answer is still zero. It is refused where the
//! mistake is, at startup, by
//! [`StartError::NotInstalled`](crate::StartError::NotInstalled) — so the only
//! programs that reach these readings are the ones being measured.
//!
//! # One engine per process, and what that means for `cargo test`
//!
//! There is one profiler per process, and it measures **the whole process** for
//! as long as it is alive. `cargo test` runs the tests in a binary
//! concurrently, so a second test allocating while the first holds the profiler
//! is counted into the first one's totals — and a second test *starting* a
//! profiler is refused outright.
//!
//! So a budget belongs in an integration test of its own, containing one
//! `#[test]`, the way `tests/testing_api.rs` in this repository is arranged.
//! `--test-threads=1` also works, and is weaker: it stops other tests running
//! *during* the profiled window, which is enough for
//! [`assert_max_bytes!`](crate::assert_max_bytes) but leaves whatever the
//! harness itself does on the profiled thread inside the counts.
//!
//! This is why [`assert_alloc_count!`](crate::assert_alloc_count) is usually
//! written against a mark rather than from the start of the run: read
//! [`HeapStats::get`] immediately before the code under test, and assert
//! `since: mark`.
//!
//! ## A finished run keeps answering
//!
//! There is one engine per process and it does not restart, so after a
//! profiler is dropped its counters stay readable and frozen. That is
//! deliberate — asserting after an explicit `drop` is a legitimate shape — and
//! it has a sharp edge: a **second** test in the same binary reads the *first*
//! test's numbers. Its own `Profiler::builder().build()` returns
//! [`StartError::AlreadyRecorded`](crate::StartError::AlreadyRecorded), so a
//! test that unwraps it fails loudly; a test that ignores it will assert
//! against a run it never made, and `assert_no_leaks!()` will pass while
//! measuring nothing. One `#[test]` per binary is what avoids it.
//!
//! # Two things worth knowing about the failure report
//!
//! **The summary is not separately switchable.** [`DUMP_VARIABLE`] turns off
//! the profile *and* the program-point summary together, because they are one
//! diagnostic. It is written to file descriptor 2 directly, which no panic hook
//! and no test harness intercepts, so a run with deliberate failures in it will
//! print summaries whatever else it does.
//!
//! **A dump that cannot be written says so** rather than failing silently: the
//! panic message carries the error, so a path in a directory that does not
//! exist is distinguishable from dumping being turned off. And the dump is
//! independent of whatever the profiler was configured to write —
//! [`no_output`](crate::ProfilerBuilder::no_output) suppresses the profile
//! written when the profiler stops, not this one.
//!
//! Both environment variables are read on **every** call rather than cached, so
//! a test can change its mind between assertions. `HEAPSCOPE_SYMBOLIZE` caches;
//! these do not.
//!
//! # A failing assertion writes a profile
//!
//! "The budget was 64 KiB and the peak was 400 KiB" says a test failed. It does
//! not say *which call site* spent the difference, and that is the only thing
//! anyone wants to know next. So a failing assertion prints the heaviest program
//! points to stderr and writes a DHAT profile beside the test, and the panic
//! message names the file. See [`DUMP_VARIABLE`] for where it goes.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::internals::engine::{Engine, Mode, State};
use crate::output::{count, OutsideRegions, Ranking, RegionStats, Snapshot};

/// What a heap run has recorded, as of now.
///
/// A point-in-time reading of the engine's global counters, which is what makes
/// it cheap: no program points are visited, no live blocks are swept, no lock is
/// taken. Compare [`Snapshot::capture`], which reads everything and is what a
/// profile is written from.
///
/// Each field is read on its own, so a reading taken while *other* threads are
/// still recording describes an instant rather than a consistent state: two
/// fields may come from either side of one event. A run that has stopped, or a
/// single-threaded one, is exact.
///
/// The assertions here do **not** each read a single field — every one of them
/// reads [`dropped_blocks`](HeapStats::dropped_blocks) as well, and
/// [`assert_baseline!`](crate::assert_baseline) compares all six. What they do
/// instead is never *arithmetic* across two of them: the one place that
/// subtracted two live counters produced "1 blocks totalling 0 bytes were never
/// freed", which is a sentence that cannot be true, and it now reports each
/// reading as the absolute figure it is.
///
/// The `since: mark` forms do subtract, and what they subtract is the reason
/// they may: one counter from *the same counter* read earlier, never one
/// counter from another. For [`total_blocks`](HeapStats::total_blocks), which
/// only grows, that difference is a count of what happened between the two
/// readings. For `curr_blocks` it is only a net change, which is why
/// [`assert_no_leaks!`](crate::assert_no_leaks) reports blocks *beyond* the
/// mark and gives the bytes as two absolute figures rather than a third.
///
/// `#[non_exhaustive]`: sampling metadata joins this in M6.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct HeapStats {
    /// Bytes allocated and not yet freed.
    pub curr_bytes: u64,
    /// Blocks allocated and not yet freed.
    pub curr_blocks: u64,
    /// The greatest [`curr_bytes`](HeapStats::curr_bytes) ever reached. DHAT's
    /// `gmax`, and what [`assert_max_bytes!`](crate::assert_max_bytes) is about.
    pub max_bytes: u64,
    /// Blocks live at the moment of that peak.
    pub max_blocks: u64,
    /// Bytes ever allocated, freed or not.
    pub total_bytes: u64,
    /// Allocations ever made.
    ///
    /// A reallocation counts as one, in addition to the block it grew, because
    /// that is what DHAT's `tbk` counts and a resize really is a new block. A
    /// `Vec` pushed to a thousand times is not one allocation.
    pub total_blocks: u64,
    /// Allocations the live-block table had no room to track.
    ///
    /// Non-zero means every other figure here is missing this many blocks, so
    /// the assertions refuse rather than compare against an incomplete
    /// measurement. Zero for any run that stayed under
    /// [`max_live_blocks`](crate::ProfilerBuilder::max_live_blocks), which is
    /// almost all of them.
    pub dropped_blocks: u64,
}

/// What a run counting [`event`](fn@crate::event)s or [`copied`](crate::copied)
/// bytes has recorded, as of now.
///
/// The counterpart of [`HeapStats`] for the two modes where the allocator shim
/// records nothing and the program reports its own events. There is no live
/// figure and no peak here, because an event is never live and never dies: a
/// zero in those columns would be a measurement of something that did not
/// happen, which is the same reason the DHAT emitter omits the fields rather
/// than zeroing them.
///
/// The plan (PLAN.md section 4) calls this `AdHocStats`. It is spelled
/// `EventStats` because [`Mode::Copy`] is an event mode too — both go through
/// the same recording path — and a copy run made to read its byte total out of
/// a type named after ad hoc mode would be reading a name that is false about
/// its own units.
///
/// `#[non_exhaustive]`: sampling metadata joins this in M6.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EventStats {
    /// Which of the two modes produced these, and therefore what
    /// [`total_weight`](EventStats::total_weight) is counted in.
    pub mode: Mode,
    /// Summed weight of every event recorded.
    ///
    /// Bytes under [`Mode::Copy`]. Under [`Mode::AdHoc`] it means whatever the
    /// program said it means when it called [`event`](fn@crate::event): retries,
    /// rows, cache misses.
    pub total_weight: u64,
    /// Events recorded.
    pub total_events: u64,
    /// Calls to the reporting function this run does *not* count.
    ///
    /// [`copied`](crate::copied) during an ad hoc run, or
    /// [`event`](fn@crate::event) during a copy one. Non-zero means instrumentation
    /// is being reported into a run that discards it, so a test asserting on a
    /// weight is asserting on a number that is missing those calls.
    pub refused_events: u64,
}

/// Every region's row, and what was recorded outside all of them, as of now.
///
/// The cheap way to report a phase-structured program by phase. Compare
/// [`Snapshot::capture`], which carries the same rows and also copies out
/// every program point and every thread the run has; this reads the region
/// table, a few hundred rows at most, and nothing else.
///
/// ```no_run
/// # fn parse() {}
/// # fn emit() {}
/// let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// {
///     let _region = heapscope::region("parsing");
///     parse();
/// }
/// {
///     let _region = heapscope::region("emitting");
///     emit();
/// }
/// let breakdown = heapscope::RegionBreakdown::get().unwrap();
/// for region in &breakdown.regions {
///     println!("{:?}: {} allocations", region.name, region.counts.total_blocks);
/// }
/// println!("(no region): {} allocations", breakdown.outside.total_blocks);
/// ```
///
/// # What adds up
///
/// For each of `total_bytes`, `total_blocks`, `curr_bytes` and `curr_blocks`,
/// the rows in [`regions`](RegionBreakdown::regions) — the shared overflow row
/// among them — plus [`outside`](RegionBreakdown::outside) equal the run's own
/// total at the instant of the reading, exactly. The peaks do not add up and
/// are not meant to: each region's is its own, reached at its own moment.
///
/// Unlike [`HeapStats`], this is one consistent instant even while other
/// threads are recording. The row outside every region is a subtraction, and a
/// subtraction across two instants can report something that never happened,
/// so the region rows and the totals are read together with the profiler's peak
/// gate held: every recording thread waits for as long as it takes to copy a
/// few hundred rows. In a run that has stopped there is nothing to wait for.
///
/// # Reading it does not change it
///
/// The reading allocates, for the rows and their names, and none of that is
/// recorded: the calling thread is inside the profiler for the duration, so a
/// reading taken inside a region is not charged to that region.
///
/// `#[non_exhaustive]`, as the other readings are.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RegionBreakdown {
    /// Which mode produced these, and therefore what the columns count.
    ///
    /// In [`Mode::Heap`], bytes and blocks. In an event mode, `total_bytes` is
    /// summed weight and `total_blocks` is events, and the live and peak
    /// columns are zero because an event is never live — read them only in a
    /// heap run.
    pub mode: Mode,
    /// One row per region name the program entered, in the order it first
    /// entered them, with the shared row for names past the table's capacity
    /// last if anything reached it. Empty if the program has entered no region.
    pub regions: Vec<RegionStats>,
    /// What was recorded while no region was open on the recording thread.
    ///
    /// In a run that has entered no region, the whole run.
    pub outside: OutsideRegions,
}

impl RegionBreakdown {
    /// The region rows of the run recording in this process, and what was
    /// recorded outside them.
    ///
    /// # Errors
    ///
    /// [`StatsError::NotRecording`], [`StatsError::ForkedChild`],
    /// [`StatsError::Poisoned`] and [`StatsError::Sampled`], for the reasons
    /// [`HeapStats::get`] gives: each is a run whose numbers would be zeros, a
    /// parent's, incomplete, or estimates. Not the two mode errors, because
    /// region rows exist in every mode and [`RegionBreakdown::mode`] says which
    /// one these came from.
    ///
    /// [`StatsError::NoQuietPoint`] if the region rows and the totals could not
    /// be read at one instant, which takes a thread stuck inside the profiler —
    /// stopped by a debugger, or this one, calling from a signal handler that
    /// interrupted it. Asking again from ordinary code is the remedy.
    pub fn get() -> Result<RegionBreakdown, StatsError> {
        Self::of(crate::engine())
    }

    /// Reads a specific engine. Testing hook.
    pub(crate) fn of(engine: &Engine) -> Result<RegionBreakdown, StatsError> {
        recording(engine)?;
        // Before the read, as in `HeapStats::of`: sampling is fixed for the
        // life of a run, so there is nothing to wait for.
        if engine.is_sampled() {
            return Err(StatsError::Sampled);
        }

        // Taken before anything allocates, so that this reading's own rows
        // and names are not recorded into the run it is reading — and, for a
        // caller inside a region, not charged to that region.
        let _quiet = crate::internals::guard::enter();

        // Room for every row the table can ever hold, reserved before the gate
        // is taken because the visitor may not allocate under it. The table
        // is bounded, so this is a few dozen kilobytes at worst — and it means
        // no row can arrive during the read and fail to fit, which would leave
        // the rows short of the totals with nothing here to say so.
        let mut rows = Vec::with_capacity(engine.regions().capacity() + 1);
        let outside = engine
            .visit_regions(Engine::FLUSH_TIMEOUT, |row| {
                // Checked anyway, because a push past capacity would allocate
                // under the gate, and that is a deadlock rather than a wrong
                // number. Not an assertion, for the same reason: a panic
                // allocates too.
                if rows.len() < rows.capacity() {
                    rows.push(row);
                }
            })
            .ok_or(StatsError::NoQuietPoint)?;

        // After the read, as in `HeapStats::of`, and for more than the usual
        // reason: a region table that accounts for more than the run recorded
        // poisons the engine during the read, so this is where that refusal
        // surfaces rather than as a row that was quietly clamped to zero.
        unpoisoned()?;
        Ok(RegionBreakdown {
            mode: engine.mode(),
            regions: rows.iter().map(RegionStats::of_view).collect(),
            outside,
        })
    }

    /// The row for the region named `name`, if the program has entered it.
    ///
    /// `name` is cut to the length the profiler keeps, exactly as
    /// [`region`](fn@crate::region) cuts it, so the name a program entered
    /// finds its row however long it was. `None` means the program never
    /// entered that name, or entered it only after the table was full — its
    /// figures are then in the shared overflow row, which this never returns
    /// because it is not that region alone.
    pub fn region(&self, name: &str) -> Option<&RegionStats> {
        let kept = crate::internals::site::Name::of(name);
        self.regions.iter().find(|row| {
            // An empty name is kept as `None` on the row, because a row's name
            // is optional for the overflow row's sake.
            !row.overflow && row.name.as_deref().unwrap_or("").as_bytes() == kept.as_bytes()
        })
    }
}

/// Why there are no statistics to read.
///
/// Every variant is a case where returning numbers would mean returning zeros,
/// and zeros are indistinguishable from a program that allocated nothing. Which
/// one it is decides what to do about it, which is why they are distinguished
/// rather than folded into one "unavailable".
///
/// `#[non_exhaustive]` because the reasons a number can be unavailable are not a
/// closed set: [`Sampled`](StatsError::Sampled) was added in M6, after the three
/// public types this reaches had already shipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StatsError {
    /// No profiler has recorded anything in this process.
    NotRecording,
    /// This run counts events the program reports, not allocations.
    NotAHeapRun(Mode),
    /// This run counts allocations, not events the program reports.
    NotAnEventRun,
    /// The profiler reported an internal failure and stopped recording.
    Poisoned,
    /// This process is a `fork` child of a profiled parent.
    ///
    /// The counters came across the `fork` and describe what the *parent* had
    /// recorded by then, which is why the child does not write a profile of them
    /// either.
    ForkedChild,
    /// This run samples, so its counters are estimates rather than counts.
    ///
    /// Every assertion this crate offers compares a number against a budget, and
    /// a sampled number is a draw from a distribution: it moves between runs of
    /// the same program, by more than most budgets allow. An assertion against
    /// one does not fail *less* often than an exact one — it fails and passes for
    /// reasons that have nothing to do with the program under test, which is
    /// worse than not having it.
    ///
    /// PLAN.md section 6.3 originally put this refusal on the builder, as a
    /// rejection of `sampling` combined with a `testing` flag. That can only
    /// refuse a program which *declared* that it meant to assert; a program that
    /// did not declare it would go on asserting against estimates in silence. The
    /// refusal is where the number is read because there it needs no declaration
    /// and cannot be bypassed.
    Sampled,
    /// The reading needed every recording thread held still for an instant,
    /// and they could not be.
    ///
    /// Only [`RegionBreakdown::get`] returns this. Its row outside every region
    /// is the totals less the region rows, so it reads both with the
    /// profiler's peak gate held, and it waits a bounded time for that rather
    /// than forever. A thread holding the gate that long is stuck — stopped by
    /// a debugger, or the calling thread itself, interrupted by the signal
    /// handler that is asking. Not a fault in the program under test, and
    /// asking again from ordinary code is the remedy.
    NoQuietPoint,
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatsError::NotRecording => write!(
                f,
                "no heapscope profiler is recording in this process; \
                 start one before reading its counters"
            ),
            // Spelled out per mode rather than interpolated, because
            // "this run counts copy events" names a unit this crate does not
            // have: copy mode counts the bytes a program says it copied.
            StatsError::NotAHeapRun(Mode::Copy) => write!(
                f,
                "this run counts the bytes it copied rather than allocations, \
                 so it has no heap statistics; read EventStats::get() instead"
            ),
            StatsError::NotAHeapRun(mode) => write!(
                f,
                "this run counts {mode} events rather than allocations, \
                 so it has no heap statistics; read EventStats::get() instead"
            ),
            StatsError::NotAnEventRun => write!(
                f,
                "this run counts allocations rather than reported events, \
                 so it has no event statistics; read HeapStats::get() instead"
            ),
            StatsError::Poisoned => write!(
                f,
                "the profiler reported an internal failure and stopped \
                 recording; its counters are incomplete"
            ),
            StatsError::ForkedChild => write!(
                f,
                "these counters were inherited from a profiled parent by fork \
                 and describe the parent's run, not this one"
            ),
            StatsError::Sampled => write!(
                f,
                "this run samples allocations, so its counters are estimates \
                 and not a budget worth asserting against; build the profiler \
                 without sampling(..) for a test that asserts"
            ),
            StatsError::NoQuietPoint => write!(
                f,
                "the profiler could not hold every recording thread still long \
                 enough to read the regions and the totals at one instant; a \
                 thread is stuck inside the profiler, so ask again from \
                 ordinary code"
            ),
        }
    }
}

impl std::error::Error for StatsError {}

impl HeapStats {
    /// The counters of the run recording in this process.
    ///
    /// # Errors
    ///
    /// Every case in [`StatsError`]: no run, an event run, a poisoned engine, a
    /// `fork` child, a sampled run. None of them is an internal failure — each is
    /// a question this run cannot answer, and returning zeros for them is what
    /// would make an assertion built on this unable to fail.
    ///
    /// A sampled `gmax` is an estimate with variance rather than a bound, so a
    /// budget assertion against it would be confident nonsense. PLAN.md section
    /// 6.3 put that refusal on the builder — reject `sampling` combined with
    /// `testing` — and it is here instead, because a builder can only refuse a
    /// program that *declared* it intended to assert, and a program that did not
    /// declare it would go on asserting against estimates in silence. Refusing
    /// where the number is read needs no declaration and cannot be bypassed.
    pub fn get() -> Result<HeapStats, StatsError> {
        Self::of(crate::engine())
    }

    /// Reads a specific engine. Testing hook.
    pub(crate) fn of(engine: &Engine) -> Result<HeapStats, StatsError> {
        recording(engine)?;
        let mode = engine.mode();
        if mode != Mode::Heap {
            return Err(StatsError::NotAHeapRun(mode));
        }
        // Before the counters are read rather than after, unlike the poison
        // check below: sampling is fixed for the life of a run, so there is no
        // window for it to arrive mid-read, and refusing before doing the work
        // is what a caller would expect of a question with a static answer.
        if engine.is_sampled() {
            return Err(StatsError::Sampled);
        }
        let stats = engine.stats();
        // Checked *after* the counters are read, not before: a poison raised
        // while they were being read would otherwise be missed, and the whole
        // point of this module is to refuse rather than to guess.
        unpoisoned()?;
        Ok(HeapStats {
            curr_bytes: stats.curr_bytes,
            curr_blocks: stats.curr_blocks,
            max_bytes: stats.max_bytes,
            max_blocks: stats.max_blocks,
            total_bytes: stats.total_bytes,
            total_blocks: stats.total_blocks,
            dropped_blocks: stats.dropped_blocks,
        })
    }
}

impl EventStats {
    /// The counters of the ad hoc or copy run recording in this process.
    ///
    /// # Errors
    ///
    /// As [`HeapStats::get`], except that the mode this one refuses is
    /// [`Mode::Heap`].
    pub fn get() -> Result<EventStats, StatsError> {
        Self::of(crate::engine())
    }

    /// Reads a specific engine. Testing hook.
    pub(crate) fn of(engine: &Engine) -> Result<EventStats, StatsError> {
        recording(engine)?;
        let mode = engine.mode();
        if mode == Mode::Heap {
            return Err(StatsError::NotAnEventRun);
        }
        // Refused for the same reason as in `HeapStats::of`, even though an
        // event run's own weights are never sampled: `sampling` is a property of
        // the run, and a program that set it and then read event counters has
        // asked for a number this run does not have. Silently answering the
        // question it did not ask is what this module exists not to do.
        if engine.is_sampled() {
            return Err(StatsError::Sampled);
        }
        let stats = engine.stats();
        unpoisoned()?;
        Ok(EventStats {
            mode,
            total_weight: stats.total_bytes,
            total_events: stats.total_blocks,
            refused_events: stats.refused_events,
        })
    }
}

/// Whether `engine` holds counters that describe a run of this process.
///
/// `Finished` qualifies deliberately: a run that has stopped has final numbers,
/// and asserting on them after the profiler is dropped is a legitimate shape for
/// a test. See the module documentation for the trap that admits — a *later*
/// test in the same binary reads the earlier run's numbers.
///
/// `Starting` does not qualify. It **is** reachable from outside this crate,
/// contrary to what this said first: the state word is process-wide, so any
/// other thread calling [`HeapStats::get`] while `Profiler::builder().build()`
/// runs on the main thread observes it. Nothing has been recorded in that
/// window, so "no run has recorded anything" is the true answer, which is why
/// the wrong justification did not produce a wrong result.
fn recording(engine: &Engine) -> Result<(), StatsError> {
    match engine.state() {
        State::Idle | State::Starting => Err(StatsError::NotRecording),
        State::ForkedChild => Err(StatsError::ForkedChild),
        State::Running | State::Finished => Ok(()),
    }
}

/// Whether the profiler has reported an internal failure.
///
/// A poisoned engine stops recording, so its counters stopped moving somewhere
/// nobody chose. They may still be entirely usable, which is why a profile
/// carries the flag and prints the numbers anyway — but an assertion cannot
/// carry a caveat, and a reader who has to decide whether to trust a surprising
/// figure is exactly who this refuses on behalf of.
///
/// Checked *after* the mode, and that ordering is deliberate: asking a copy run
/// for a heap peak is a mistake in the test that its author can fix, and
/// reporting the poison first would send them looking for a fault in the
/// profiler instead. A poisoned engine still knows what it was counting.
fn unpoisoned() -> Result<(), StatsError> {
    if crate::internals::diagnostic::is_poisoned() {
        return Err(StatsError::Poisoned);
    }
    Ok(())
}

/// Whether this engine has a run worth writing a profile of.
fn has_a_profile(engine: &Engine) -> bool {
    matches!(engine.state(), State::Running | State::Finished)
}

/// What a measurement covers: the whole run, or only what followed a mark.
///
/// Carried by the count complaints so that a failure says which of the two
/// questions it answered, and read by [`report`] through [`AssertionFailure::scope`] so
/// that it can say the profile it writes answers the wider one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    /// Everything since the profiler started.
    WholeRun,
    /// Everything since a [`HeapStats`] mark was read.
    SinceMark,
}

impl Scope {
    /// The scope of an assertion given its optional mark.
    fn of(since: Option<HeapStats>) -> Scope {
        match since {
            None => Scope::WholeRun,
            Some(_) => Scope::SinceMark,
        }
    }

    /// The words a complaint inserts after "made". Nothing for the whole run,
    /// because that is what the unqualified sentence already means.
    fn phrase(self) -> &'static str {
        match self {
            Scope::WholeRun => "",
            Scope::SinceMark => " since the mark",
        }
    }
}

/// What an allocation count is checked against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Expected {
    /// `assert_alloc_count!(n)`: exactly this many.
    Exactly(u64),
    /// `assert_alloc_count!(<= n)`: this many or fewer, zero included.
    AtMost(u64),
}

/// What [`report`] needs to know about a failure beyond its sentence.
///
/// A trait rather than parameters to `report`, because both answers belong to
/// the failure and not to the call site. A `since: mark` assertion that was
/// refused for a sampled run measured nothing since the mark, and a note
/// qualifying its profile as broader than the stage would be qualifying a
/// measurement that was never made. Asked of the failure, the question cannot
/// be answered for the wrong one.
pub(crate) trait AssertionFailure: fmt::Display {
    /// What the failing measurement covered. A refusal measured nothing, which
    /// is the default: there is no narrower interval to warn about.
    fn scope(&self) -> Scope {
        Scope::WholeRun
    }

    /// Which figure the program points printed with the failure are ranked by.
    fn ranking(&self) -> Ranking {
        Ranking::Bytes
    }
}

/// Why an assertion did not pass.
///
/// Separated from the panic so that the decision is a value a test can examine.
/// Every one of these is a `Display` line; the panic wraps them with the caller's
/// context and the profile it wrote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Complaint {
    /// There were no numbers to check. Not a failure of the program under test.
    Unavailable(StatsError),
    /// The numbers are missing however many blocks the table turned away.
    Incomplete { dropped_blocks: u64 },
    /// The peak was above the budget.
    OverBudget { peak: u64, limit: u64 },
    /// The allocation count was not the expected one.
    WrongCount {
        counted: u64,
        expected: u64,
        scope: Scope,
    },
    /// More allocations were made than the ceiling allows.
    OverCeiling {
        counted: u64,
        ceiling: u64,
        scope: Scope,
    },
    /// The mark records more allocations than the run it is compared against.
    ///
    /// [`total_blocks`](HeapStats::total_blocks) only grows, so a mark read from
    /// these counters and passed on as read can never be ahead of a later
    /// reading. One that is was read elsewhere or changed since, and either way
    /// the count since it is not zero but unknown.
    MarkAhead { mark: u64, now: u64 },
    /// Blocks were still live.
    ///
    /// The byte figures are absolute readings rather than a difference, and
    /// that is a correction. `curr_blocks` and `curr_bytes` are separate
    /// counters, so a mark taken before a large block was freed and a small one
    /// allocated gives one *more* live block and *fewer* live bytes —
    /// subtracting both produced "1 blocks totalling 0 bytes were never freed",
    /// a sentence that fails a test and cannot be true.
    Leaked {
        /// Blocks live beyond the mark, or live at all when there was no mark.
        blocks: u64,
        /// Live bytes when the assertion ran.
        live_bytes: u64,
        /// Live bytes at the mark, if there was one.
        mark_bytes: Option<u64>,
    },
}

impl AssertionFailure for Complaint {
    fn scope(&self) -> Scope {
        match self {
            Complaint::WrongCount { scope, .. } | Complaint::OverCeiling { scope, .. } => *scope,
            Complaint::Leaked {
                mark_bytes: Some(_),
                ..
            } => Scope::SinceMark,
            // `MarkAhead` included: it is a complaint about the mark, not a
            // measurement of anything after it.
            Complaint::Unavailable(_)
            | Complaint::Incomplete { .. }
            | Complaint::OverBudget { .. }
            | Complaint::MarkAhead { .. }
            | Complaint::Leaked {
                mark_bytes: None, ..
            } => Scope::WholeRun,
        }
    }

    /// Blocks for a count that failed, because a count is about how many
    /// allocations were made, and the sites that made the most are the ones
    /// that answer it. Bytes for everything else, including a leak: what a leak
    /// costs is the memory it holds.
    fn ranking(&self) -> Ranking {
        match self {
            Complaint::WrongCount { .. } | Complaint::OverCeiling { .. } => Ranking::Blocks,
            Complaint::Unavailable(_)
            | Complaint::Incomplete { .. }
            | Complaint::OverBudget { .. }
            | Complaint::MarkAhead { .. }
            | Complaint::Leaked { .. } => Ranking::Bytes,
        }
    }
}

/// `one` when `value` is exactly one, `many` otherwise.
///
/// For the nouns and verbs a complaint puts beside a number. A message is the
/// whole of what a failing CI job shows, and "1 allocations were made" reads as
/// a message nobody looked at, which invites the reader to doubt the number
/// beside it as well.
fn agreeing<'a>(value: u64, one: &'a str, many: &'a str) -> &'a str {
    if value == 1 {
        one
    } else {
        many
    }
}

/// `value` grouped, followed by `one` or `many` to agree with it.
fn counted(value: u64, one: &str, many: &str) -> String {
    format!("{} {}", count(value), agreeing(value, one, many))
}

impl fmt::Display for Complaint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Complaint::Unavailable(error) => write!(f, "{error}"),
            Complaint::Incomplete { dropped_blocks } => write!(
                f,
                "the live-block table had no room for {} of this run's \
                 allocations, so its totals are incomplete and cannot be \
                 asserted on; raise the ceiling with \
                 Profiler::builder().max_live_blocks(..)",
                count(*dropped_blocks)
            ),
            Complaint::OverBudget { peak, limit } => write!(
                f,
                "peak live bytes reached {}, above the limit of {}",
                count(*peak),
                count(*limit)
            ),
            Complaint::WrongCount {
                counted: made,
                expected,
                scope,
            } => write!(
                f,
                "{} {} made{}, not {}",
                counted(*made, "allocation", "allocations"),
                agreeing(*made, "was", "were"),
                scope.phrase(),
                count(*expected)
            ),
            Complaint::OverCeiling {
                counted: made,
                ceiling,
                scope,
            } => write!(
                f,
                "{} {} made{}, above the ceiling of {}",
                counted(*made, "allocation", "allocations"),
                agreeing(*made, "was", "were"),
                scope.phrase(),
                count(*ceiling)
            ),
            // Names the remedy for the reason `Incomplete` does, and says what
            // the numbers imply rather than only what they are. It does not say
            // *where* the mark came from, because the numbers cannot tell: a
            // mark read from another run and one whose fields were changed after
            // reading look identical here, and a message that guessed would send
            // half its readers to look for the wrong mistake.
            Complaint::MarkAhead { mark, now } => write!(
                f,
                "the mark records {} but this run has made {}, which a mark \
                 read from this run and passed on unchanged cannot do, so the \
                 number made since it cannot be known; read the mark with \
                 HeapStats::get() during the run being asserted on, and pass it \
                 as read",
                counted(*mark, "allocation", "allocations"),
                count(*now)
            ),
            // The remedy is named for the reason `Incomplete` names one: the
            // likeliest cause of this failing is not a leak but an assertion
            // written without a mark, on a program that legitimately holds
            // memory of its own — and the bare form fails on any real test
            // binary, which this crate's own suite asserts.
            Complaint::Leaked {
                blocks,
                live_bytes,
                mark_bytes: None,
            } => write!(
                f,
                "{} totalling {} {} never freed; if the program holds memory of \
                 its own, take a mark with HeapStats::get() first and assert \
                 `since: mark`",
                counted(*blocks, "block", "blocks"),
                counted(*live_bytes, "byte", "bytes"),
                agreeing(*blocks, "was", "were")
            ),
            Complaint::Leaked {
                blocks,
                live_bytes,
                mark_bytes: Some(mark),
            } => write!(
                f,
                "{} more {} {} live than at the mark, where live bytes went \
                 from {} to {}",
                count(*blocks),
                agreeing(*blocks, "block", "blocks"),
                agreeing(*blocks, "is", "are"),
                count(*mark),
                count(*live_bytes)
            ),
        }
    }
}

/// The reading the assertions work from, or the reason there is not one.
///
/// The completeness check lives here rather than in each assertion because it
/// applies to all of them for one reason: a dropped block is an allocation this
/// profiler saw and could not track, so it is missing from `total_blocks`, from
/// `curr_blocks`, and from every peak the run reached while it was live.
pub(crate) fn assertable(engine: &Engine) -> Result<HeapStats, Complaint> {
    let stats = HeapStats::of(engine).map_err(Complaint::Unavailable)?;
    if stats.dropped_blocks > 0 {
        return Err(Complaint::Incomplete {
            dropped_blocks: stats.dropped_blocks,
        });
    }
    Ok(stats)
}

pub(crate) fn check_max_bytes(engine: &Engine, limit: u64) -> Result<(), Complaint> {
    let stats = assertable(engine)?;
    if stats.max_bytes > limit {
        return Err(Complaint::OverBudget {
            peak: stats.max_bytes,
            limit,
        });
    }
    Ok(())
}

pub(crate) fn check_alloc_count(
    engine: &Engine,
    since: Option<HeapStats>,
    expected: Expected,
) -> Result<(), Complaint> {
    let stats = assertable(engine)?;
    let counted = allocations_since(&stats, since)?;
    let scope = Scope::of(since);
    match expected {
        Expected::Exactly(expected) if counted != expected => Err(Complaint::WrongCount {
            counted,
            expected,
            scope,
        }),
        Expected::AtMost(ceiling) if counted > ceiling => Err(Complaint::OverCeiling {
            counted,
            ceiling,
            scope,
        }),
        Expected::Exactly(_) | Expected::AtMost(_) => Ok(()),
    }
}

/// Allocations made since `since`, or over the whole run without one.
///
/// A difference of one counter between two of its own readings, which is not
/// the arithmetic [`HeapStats`] warns against: `total_blocks` only grows, so
/// what lies between two readings of it is exactly the allocations made in
/// between.
///
/// # A mark ahead of the reading is refused rather than saturated
///
/// `check_no_leaks` saturates its difference, and the reason does not transfer.
/// There, a reading below the mark has an innocent cause — blocks live at the
/// mark were freed — and its true answer, no leak, is the one saturating gives.
/// Here there is no innocent cause: a counter that only grows cannot read below
/// an earlier reading of itself, so a mark ahead of `stats` was not passed on as
/// it was read from these counters, and the count since it is unknown.
/// Saturating would report zero, which passes every ceiling and
/// `since: mark, 0` — an answer invented in the direction of passing, which is
/// the one this module exists to refuse.
///
/// It is reachable. [`HeapStats`] is `#[non_exhaustive]`, which stops a caller
/// *building* one but not changing one: its fields are public, so a mark read
/// properly and then adjusted (`mark.total_blocks += 1_000`) arrives here
/// ahead of the run. Totals that could restart would reach it a second way, from
/// a mark read before the restart.
///
/// This check is not the whole of what a restart would need, and saying so
/// matters: a mark from before one that happens to be *behind* the new run's
/// total gives a plausible difference that measures nothing. Telling those apart
/// needs the reading to carry which run it came from, which is what
/// `#[non_exhaustive]` leaves room for.
fn allocations_since(stats: &HeapStats, since: Option<HeapStats>) -> Result<u64, Complaint> {
    let Some(mark) = since else {
        return Ok(stats.total_blocks);
    };
    stats
        .total_blocks
        .checked_sub(mark.total_blocks)
        .ok_or(Complaint::MarkAhead {
            mark: mark.total_blocks,
            now: stats.total_blocks,
        })
}

pub(crate) fn check_no_leaks(engine: &Engine, since: Option<HeapStats>) -> Result<(), Complaint> {
    let stats = assertable(engine)?;
    // Blocks, not bytes, decide whether anything leaked: a live zero-sized
    // allocation is a block that was never freed and contributes no bytes, and
    // gating on bytes would report it as clean.
    //
    // Saturating rather than wrapping, because a reading taken while another
    // thread frees can legitimately come back smaller than the mark. That is not
    // a leak of a negative number of blocks; it is no leak.
    let before = since.map_or(0, |mark| mark.curr_blocks);
    let blocks = stats.curr_blocks.saturating_sub(before);
    if blocks > 0 {
        return Err(Complaint::Leaked {
            blocks,
            live_bytes: stats.curr_bytes,
            mark_bytes: since.map(|mark| mark.curr_bytes),
        });
    }
    Ok(())
}

/// Where a failing assertion writes its profile.
///
/// Set it to a path and every dump goes there. Set it to `0`, `off`, `no`, or
/// `false` and nothing is written — the panic message still names the numbers.
/// Unset, a dump goes to `heapscope-assert-<thread>.json` in the working
/// directory, which for a `cargo test` binary names the test that failed.
///
/// A second dump in the same process never overwrites the first: it takes the
/// same path with `.2` inserted before the extension, then `.3`, and so on. That
/// is not tidiness — a panic message naming a file that a *different* test has
/// since overwritten sends the reader to the wrong profile, and two tests
/// failing in the same run is the ordinary case rather than an unlucky one.
pub const DUMP_VARIABLE: &str = "HEAPSCOPE_ASSERT_PROFILE";

/// Program points printed to stderr when an assertion fails.
///
/// Enough to name the site that spent the budget, few enough that a failing
/// assertion does not bury its own message.
const TOP_ON_FAILURE: usize = 5;

/// Dumps written by this process so far, so that the second one does not land on
/// the first.
static DUMPS: AtomicU64 = AtomicU64::new(0);

/// Writes a profile of the run as it stands, and returns the line describing
/// where it went for the panic message to carry.
///
/// The caller has already decided *where* — see [`dump_target`] for the two
/// reasons there may be nowhere.
///
/// # The `&Guard` is required rather than used
///
/// Capturing a snapshot takes the peak gate exclusively and then walks the
/// live-block shard locks. A thread that is already inside the profiler — an
/// assertion reached from a `Drop` running under the allocator shim, or from a
/// signal handler that interrupted one — may be holding either.
///
/// The two misbehave differently, and saying so matters because only one of
/// them is fatal. The gate is deadline-bounded (`Gate::write_for`), so a
/// reentrant acquisition there is a two-second stall of every allocating thread
/// followed by a "could not reach a quiet point" diagnostic and an inexact
/// profile. The **shard locks are not**: `LiveBlocks::for_each` takes them
/// blocking, and on Apple platforms `os_unfair_lock` kills the process outright
/// on reentrant acquisition rather than deadlocking.
///
/// Holding the reentrancy guard is what makes both unreachable, so this takes
/// the proof as an argument: dumping from a thread that could not enter is a
/// borrow-check error rather than a test nobody can write. The same move
/// [`Engine::record_event`](crate::internals::engine::Engine::record_event) and
/// `guard::enter_region` each made.
fn dump(
    engine: &Engine,
    _entered: &crate::internals::guard::Guard,
    path: &Path,
    ranking: Ranking,
    summary: &mut dyn io::Write,
) -> String {
    // One reading, both destinations, for the reason `write_outputs` takes one:
    // a summary and a file that disagree about the same failure are two
    // readings of a program that kept running in between.
    let snapshot = Snapshot::of(engine);
    let _ = snapshot.write_text_summary_ranked(summary, TOP_ON_FAILURE, ranking);

    let named = screened(&path.display().to_string());
    match snapshot.save_dhat_v2(path) {
        Ok(()) => format!("profile written to {named}"),
        Err(error) => format!("could not write a profile to {named}: {error}"),
    }
}

/// Where this failure's profile goes, or `None` if it should not write one.
///
/// The two reasons not to are separate and both need to be reachable by a test:
/// there is no run to describe, or the reader turned dumping off. Composed from
/// [`has_a_profile`] and [`dump_base`] rather than written out here, because a
/// function that reads the environment and bumps a counter is one no test can
/// drive twice with the same answer.
fn dump_target(engine: &Engine) -> Option<PathBuf> {
    if !has_a_profile(engine) {
        return None;
    }
    let base = dump_base(std::env::var_os(DUMP_VARIABLE).as_deref())?;
    let ordinal = DUMPS.fetch_add(1, Ordering::Relaxed);
    // The other half of the same question, asked of the filesystem rather than
    // of the string, and asked here rather than inside `distinguish` because it
    // is the one place a real dump happens. A base that already names a
    // directory keeps its name: the write then fails with "Is a directory",
    // which is the true answer, where distinguishing it would quietly succeed
    // at writing a *sibling* of the directory the caller named.
    if base.is_dir() {
        return Some(base);
    }
    Some(distinguish(base, ordinal))
}

/// The name a dump takes before the ordinal is applied, given the setting.
///
/// Pure, so the three cases can be checked without a test mutating the
/// environment out from under every other test in the binary.
fn dump_base(setting: Option<&std::ffi::OsStr>) -> Option<PathBuf> {
    match setting {
        Some(setting) if is_off(setting) => None,
        Some(setting) => Some(PathBuf::from(setting)),
        None => Some(PathBuf::from(format!(
            "heapscope-assert{}.json",
            thread_suffix()
        ))),
    }
}

/// `-<thread name>`, or nothing for a thread the platform has no name for.
///
/// Read through `std::thread`, not through the platform call the engine uses,
/// because this runs from a failing assertion rather than from the allocator
/// path: there is no `Drop`-during-teardown hazard here, and `std`'s name is the
/// full Rust one rather than the 15 bytes Linux keeps.
fn thread_suffix() -> String {
    let current = std::thread::current();
    let Some(name) = current.name() else {
        return String::new();
    };
    let mut suffix = String::with_capacity(name.len() + 1);
    suffix.push('-');
    // A test name is a path (`tests::budgets::parsing`), and a profile is a file
    // rather than a directory tree. Everything outside this set becomes an
    // underscore, so the name survives as something a person recognises without
    // ever naming a directory that does not exist.
    for character in name.chars().take(MAX_THREAD_SUFFIX) {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
            suffix.push(character);
        } else {
            suffix.push('_');
        }
    }
    if suffix.len() == 1 {
        return String::new();
    }
    suffix
}

/// Characters of a thread name that reach the file name.
const MAX_THREAD_SUFFIX: usize = 64;

/// `path` for dump zero, and `path` with `.n+1` before the extension after that.
fn distinguish(path: PathBuf, dump: u64) -> PathBuf {
    if dump == 0 {
        return path;
    }
    // A path naming a directory has no file name to distinguish, and
    // `file_stem` will not say so: for `dumps/` it answers `dumps`, so
    // `with_file_name` would produce `dumps.2` — a *sibling of* the directory
    // the caller named, written outside it. `HEAPSCOPE_ASSERT_PROFILE=/tmp/`
    // put a file at the filesystem root. The first dump at such a path fails to
    // open and says so, which is the right answer; the second must not succeed
    // somewhere else instead.
    if ends_in_separator(&path) {
        return path;
    }
    let ordinal = dump + 1;
    let name = match (path.file_stem(), path.extension()) {
        (Some(stem), Some(extension)) => {
            let mut name = stem.to_os_string();
            name.push(format!(".{ordinal}."));
            name.push(extension);
            name
        }
        (Some(stem), None) => {
            let mut name = stem.to_os_string();
            name.push(format!(".{ordinal}"));
            name
        }
        // A path with no file name at all — `/` or `..`. Nothing sensible can be
        // derived from it, and it will fail to open either way.
        (None, _) => return path,
    };
    path.with_file_name(name)
}

/// Whether `path` ends in a separator, and so names a directory.
///
/// The case `Path` will not answer: a trailing separator is normalised away by
/// `components`, and `file_stem` reports the last directory name as though it
/// were a file. Testing the string is the only way to see it — and testing the
/// string rather than the filesystem is what keeps [`distinguish`] a pure
/// function, which is what keeps it checkable under Miri.
fn ends_in_separator(path: &Path) -> bool {
    path.as_os_str()
        .to_string_lossy()
        .ends_with(std::path::is_separator)
}

/// A path or a file fragment on its way to a terminal, with anything that would
/// drive one removed.
///
/// Shared with [`crate::baseline`], which screens the same two kinds of string
/// for the same reason: both come from the caller rather than from us.
pub(crate) fn screened(text: &str) -> String {
    let mut screened = String::new();
    crate::output::push_display(&mut screened, text);
    screened
}

/// Whether an environment setting reads as "off".
///
/// The same four spellings `HEAPSCOPE_SYMBOLIZE` accepts, **and in the same
/// letter case**, which is a fix rather than a flourish: this folded no case
/// while `symbol::dynamic` folded to lowercase, so `HEAPSCOPE_UPDATE_BASELINE=FALSE`
/// read as *on* and silently rewrote every baseline it was supposed to check
/// against. A variable spelled two ways in one crate is a variable nobody can
/// remember, and the failure it produced here was a gate reporting success
/// forever.
pub(crate) fn is_off(setting: &std::ffi::OsStr) -> bool {
    setting
        .to_str()
        .map(|text| {
            matches!(
                text.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "no" | "false"
            )
        })
        .unwrap_or(false)
}

/// Fails the test, having first written down what the program was doing.
///
/// `#[track_caller]` so the panic names the assertion in the test rather than
/// this function.
///
/// # Why this holds the reentrancy guard
///
/// Because building the message allocates, and the profile written two lines
/// later would otherwise contain it. That is the failure `write_text_summary`
/// and `write_native` each had in turn — a profiler that changes what it
/// measures — and it is worse here than in either of them: the numbers in the
/// panic message have already been read, so the profile the reader is sent to
/// would disagree with the message that sent them.
///
/// # A failure since a mark says what its profile covers
///
/// A measurement made `since: mark` is about an interval, and the profile
/// written for it is not: a mark holds the run's totals and nothing per program
/// point, so there is nothing to subtract the earlier sites from. Left unsaid,
/// the heaviest sites printed for a stage's budget include whatever the warm-up
/// before the mark allocated — the first place a reader looks for the stage's
/// cost. The profile stays whole because it is still the evidence there is;
/// the note is what keeps it from being read as something narrower.
#[track_caller]
pub(crate) fn report<E: AssertionFailure>(
    outcome: Result<(), E>,
    context: Option<fmt::Arguments<'_>>,
) {
    let Err(complaint) = outcome else {
        // The passing path allocates nothing and takes nothing, which is what
        // lets an assertion sit inside a loop.
        return;
    };
    let quiet = crate::internals::guard::enter();
    let engine = crate::engine();
    let mut message = format!("heapscope: {complaint}");
    if let Some(context) = context {
        message.push_str("\n  ");
        message.push_str(&context.to_string());
    }
    // `None` from `enter` means this thread could not be entered — it is
    // already inside the profiler, or the guard table had no slot for it —
    // where taking a snapshot could deadlock against its own outer acquisition.
    // The assertion still fails and still says what it measured; what it cannot
    // do from there is write a profile. See [`dump`].
    let dumped = match (&quiet, dump_target(engine)) {
        (Some(entered), Some(path)) => {
            let mut stderr = io::stderr().lock();
            Some(dump(
                engine,
                entered,
                &path,
                complaint.ranking(),
                &mut stderr,
            ))
        }
        _ => None,
    };
    if let Some(line) = dumped {
        message.push_str("\n  ");
        message.push_str(&line);
        if complaint.scope() == Scope::SinceMark {
            message.push_str("\n  ");
            message.push_str(WHOLE_RUN_NOTE);
        }
    }
    panic!("{message}");
}

/// Appended to a failure measured since a mark, wherever a dump was attempted.
///
/// Attempted rather than written, because the summary reaches stderr whether or
/// not the file could be. Worded so that it is true either way: the line above
/// it says which happened to the file.
const WHOLE_RUN_NOTE: &str = "the program points printed to stderr, and any profile written, \
                              cover the whole run, not only what followed the mark";

/// The types a budget or a count can be given as: the integer primitives, and
/// nothing else.
///
/// The bound was `TryInto<u64>`, which `bool` and `char` both satisfy. That
/// made a comparison written where a count belongs compile, and run as a count
/// of zero or one: `assert_alloc_count!(since: mark, used <= 4)` asserted that
/// the stage made exactly one allocation whenever `used <= 4` held. A budget
/// that type-checks as something it is not is the cannot-fail shape again, so
/// the bound is this instead, and its impls are the twelve integer types and
/// their `NonZero` forms. The `NonZero` impls are not a nicety: `NonZeroU64`
/// satisfied the old bound, so a budget held in one compiled before the bound
/// narrowed and has to compile after it.
///
/// Public only so that it can appear in the hidden entry points' signatures;
/// it lives in a private module, so no caller can name it, let alone implement
/// it for a type of their own.
///
/// ```compile_fail,E0277
/// # #[global_allocator]
/// # static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
/// # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// let mark = heapscope::HeapStats::get().unwrap();
/// let used = 3usize;
/// heapscope::assert_alloc_count!(since: mark, used <= 4);
/// ```
///
/// ```compile_fail,E0277
/// # #[global_allocator]
/// # static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
/// # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// heapscope::assert_max_bytes!('k');
/// ```
mod integer {
    use core::num::NonZero;

    #[diagnostic::on_unimplemented(
        message = "`{Self}` is not an integer count or byte budget",
        label = "expected an integer here",
        note = "a heapscope count or budget is an integer type, or the `NonZero` form of one; \
                a `bool` here is usually a comparison written where the number belongs"
    )]
    pub trait Integer: Copy {
        /// The value as a `u64`, or `None` when it is negative or too large.
        fn to_u64(self) -> Option<u64>;
    }

    macro_rules! integers {
        ($($integer:ty)*) => {
            $(
                impl Integer for $integer {
                    fn to_u64(self) -> Option<u64> {
                        u64::try_from(self).ok()
                    }
                }
            )*
        };
    }

    integers!(u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize);

    // A `NonZero` is a number that is already known not to be zero, so it
    // converts through the integer it wraps: a negative `NonZero<i32>` is
    // refused exactly as a negative `i32` is.
    macro_rules! non_zero {
        ($($integer:ty)*) => {
            $(
                impl Integer for NonZero<$integer> {
                    fn to_u64(self) -> Option<u64> {
                        self.get().to_u64()
                    }
                }
            )*
        };
    }

    non_zero!(u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize);
}

/// A macro argument as the `u64` the engine keeps its counters in.
///
/// Generic rather than a plain `u64` parameter because **every size and count
/// in Rust is a `usize`**: `assert_alloc_count!(items.len())` and a budget held
/// in a `usize` are the ordinary call sites, and both are a type error against
/// a `u64`. An integer literal still infers, because the fallback type, `i32`,
/// is one of the integers. There is no `From<usize> for u64`, so the
/// conversion is a fallible one.
///
/// A value that does not fit — a negative one — panics rather than saturating.
/// A budget of `-1` silently becoming `u64::MAX` is an assertion that cannot
/// fail, which is the one outcome this module exists to prevent.
#[track_caller]
fn as_count<N: integer::Integer>(value: N) -> u64 {
    value.to_u64().unwrap_or_else(|| {
        panic!(
            "heapscope: a negative or oversized number is not a byte count or an allocation count"
        )
    })
}

/// The body of [`assert_max_bytes!`](crate::assert_max_bytes). Not a supported
/// entry point.
#[doc(hidden)]
#[track_caller]
pub fn __assert_max_bytes<N: integer::Integer>(limit: N, context: Option<fmt::Arguments<'_>>) {
    report(check_max_bytes(crate::engine(), as_count(limit)), context);
}

/// The body of [`assert_alloc_count!`](crate::assert_alloc_count) for a bare
/// count. Not a supported entry point.
///
/// A `since:` with the count forgotten is a compile error that names the two
/// shapes it could have been, rather than "no rules expected the token":
///
/// ```compile_fail
/// # #[global_allocator]
/// # static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
/// # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// let mark = heapscope::HeapStats::get().unwrap();
/// heapscope::assert_alloc_count!(since: mark);
/// ```
#[doc(hidden)]
#[track_caller]
pub fn __assert_alloc_count<N: integer::Integer>(
    since: Option<HeapStats>,
    expected: N,
    context: Option<fmt::Arguments<'_>>,
) {
    let expected = Expected::Exactly(as_count(expected));
    report(check_alloc_count(crate::engine(), since, expected), context);
}

/// The body of [`assert_alloc_count!`](crate::assert_alloc_count) for a
/// `<= ceiling`. Not a supported entry point.
///
/// A function of its own rather than a flag on [`__assert_alloc_count`], so
/// that what the macro expands to says which comparison it is.
#[doc(hidden)]
#[track_caller]
pub fn __assert_alloc_count_at_most<N: integer::Integer>(
    since: Option<HeapStats>,
    ceiling: N,
    context: Option<fmt::Arguments<'_>>,
) {
    let expected = Expected::AtMost(as_count(ceiling));
    report(check_alloc_count(crate::engine(), since, expected), context);
}

/// The body of [`assert_no_leaks!`](crate::assert_no_leaks). Not a supported
/// entry point.
#[doc(hidden)]
#[track_caller]
pub fn __assert_no_leaks(since: Option<HeapStats>, context: Option<fmt::Arguments<'_>>) {
    report(check_no_leaks(crate::engine(), since), context);
}

/// Fails unless the run's peak live bytes stayed at or below `limit`.
///
/// The peak is DHAT's `gmax`: the greatest number of bytes that were
/// simultaneously live at any point since the profiler started. It is the figure
/// a memory budget is actually about — a program that allocates a gigabyte one
/// kilobyte at a time, freeing as it goes, has a peak of a kilobyte.
///
/// Takes any integer that fits a `u64`, or the `NonZero` form of one, so a
/// budget held in a `usize` works without a cast. Only integers: the bound was
/// once `TryInto<u64>`, which let a `bool` or a `char` through as a budget of
/// zero, one, or a code point. A trailing message is formatted as
/// [`format_args!`] and printed with the failure, which is worth using when
/// the same assertion runs over several fixtures.
///
/// ```
/// # #[global_allocator]
/// # static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
/// # let fixture = "big.json";
/// # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// heapscope::assert_max_bytes!(64 * 1024);
/// heapscope::assert_max_bytes!(64 * 1024, "while parsing {fixture}");
/// ```
///
/// # There is no `since:` form
///
/// [`assert_no_leaks!`](crate::assert_no_leaks) and
/// [`assert_alloc_count!`](crate::assert_alloc_count) take a mark and this does
/// not, because a peak since a mark cannot always be read off two readings of a
/// running maximum. Often it can. A run whose peak is still within the limit
/// passes, whatever the stage did, and a peak that rose after the mark was set
/// by the stage. What two readings cannot settle is a peak above the limit that
/// was reached before the mark and has not moved since: the stage's own peak is
/// then anywhere from what was live at the mark up to that figure, and neither
/// reading says which. A form that passed, failed or refused depending on when
/// the warm-up happened to peak would be harder to trust than no form at all.
///
/// # Panics
///
/// When the peak exceeded `limit`, and when there are no numbers to check — see
/// the [module documentation](crate::stats) for that list. It does **not** pass
/// quietly in either case. And when `limit` is negative, because a budget of
/// `-1` read as `u64::MAX` could not fail, and when it is larger than
/// `u64::MAX`.
#[macro_export]
macro_rules! assert_max_bytes {
    ($limit:expr $(,)?) => {
        $crate::__assert_max_bytes($limit, ::core::option::Option::None)
    };
    ($limit:expr, $($arg:tt)+) => {
        $crate::__assert_max_bytes(
            $limit,
            ::core::option::Option::Some(::core::format_args!($($arg)+)),
        )
    };
}

/// Fails unless the run made exactly `expected` allocations, or, written
/// `<= ceiling`, at most that many.
///
/// ```
/// # #[global_allocator]
/// # static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
/// # let fixture = "big.json";
/// # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// let rows = [Box::new(1u8), Box::new(2u8), Box::new(3u8)];   // three allocations
/// # std::hint::black_box(&rows);
/// heapscope::assert_alloc_count!(3);
/// heapscope::assert_alloc_count!(<= 4, "while parsing {fixture}");
/// ```
///
/// A reallocation counts as an allocation, so a `Vec` that grows four times made
/// five. See [`HeapStats::total_blocks`].
///
/// Takes any integer that fits a `u64`, or the `NonZero` form of one, so
/// `items.len()` works without a cast. Only integers: the bound was once
/// `TryInto<u64>`, which let a `bool` or a `char` through, so a comparison
/// written where the count belongs, `assert_alloc_count!(used <= 4)`, ran as a
/// count of zero or one. It is a type error now. Every form takes a trailing message, formatted as
/// [`format_args!`] and printed with the failure, which is worth using when
/// the same assertion runs over several fixtures.
///
/// # A bare count is an equality
///
/// `assert_alloc_count!(3)` means exactly three, not at most three. A ceiling
/// read into a bare number has a failure mode this crate refuses elsewhere: it
/// passes a run that allocated nothing, which is precisely how a broken test
/// goes green, and nothing at the call site says that it would. Spelled
/// `<= 3`, the ceiling is the reading written down, and a reviewer can see that
/// zero passes it. The objection was never to ceilings; it was to a permission
/// nobody could see.
///
/// # Since a mark
///
/// The bare forms count from when the profiler started, which in a test binary
/// includes whatever the harness and any warm-up allocated. A budget for one
/// stage counts from a [`HeapStats`] read immediately before it instead:
///
/// ```
/// # #[global_allocator]
/// # static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
/// # fn warm_up() -> Vec<u8> { Vec::with_capacity(4_096) }
/// # let name = "parser";
/// # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// let cache = warm_up();
/// # std::hint::black_box(&cache);
/// let mark = heapscope::HeapStats::get().unwrap();
/// let rows = [Box::new(1u8), Box::new(2u8), Box::new(3u8)];   // three allocations
/// # std::hint::black_box(&rows);
/// heapscope::assert_alloc_count!(since: mark, 3);
/// heapscope::assert_alloc_count!(since: mark, <= 4, "while compiling {name}");
/// ```
///
/// [`total_blocks`](HeapStats::total_blocks) only grows, so the count since a
/// mark is the difference between two readings of that one counter: every
/// allocation made in between, and nothing else. A mark *ahead* of the reading
/// therefore did not come unchanged from the counters being read, and the count
/// since it is not zero but unknown, so the assertion fails rather than guess.
/// A zero would pass every ceiling.
///
/// It is a difference over the whole process, as
/// [`assert_no_leaks!`](crate::assert_no_leaks)`(since: ..)` is, so an
/// allocation made by another thread during the stage is counted into it. So is
/// one made by the assertion's own arguments: the count, the mark and the
/// message's arguments are all evaluated before the counters are read, whether
/// or not the assertion fails, so `"{}", path.display().to_string()` adds an
/// allocation to the stage it is describing. A name captured by the format
/// string, as `{name}` is above, is only formatted on failure and costs a
/// passing assertion nothing.
///
/// The profile a failure writes still covers the whole run, because a mark
/// holds the run's totals and nothing per program point; the failure says so,
/// so that a site that allocated during the warm-up is not read as the stage's
/// cost. What it can do is rank the sites by how many allocations they made
/// rather than by bytes, which it does for every failing count, so that a
/// stage of many small allocations is not listed under one large buffer.
///
/// # Panics
///
/// When the count differs or exceeds the ceiling, when the mark is ahead of the
/// run, and when there are no numbers to check. And when the count or ceiling
/// is negative or larger than `u64::MAX`, because a ceiling of `-1` read as
/// `u64::MAX` could not fail.
#[macro_export]
macro_rules! assert_alloc_count {
    // The `since:` and `<=` arms come first, and neither can capture a call
    // site that meant a bare count. `<=` cannot begin an expression, so a bare
    // count never matches a `<=` arm; `since` can, but only followed by `:`,
    // which no expression is — `since::N` is one `::` token, not two `:` — so
    // a bare count that happens to be a variable named `since` falls through
    // to the arms below.
    (since: $mark:expr, <= $ceiling:expr $(,)?) => {
        $crate::__assert_alloc_count_at_most(
            ::core::option::Option::Some($mark),
            $ceiling,
            ::core::option::Option::None,
        )
    };
    (since: $mark:expr, <= $ceiling:expr, $($arg:tt)+) => {
        $crate::__assert_alloc_count_at_most(
            ::core::option::Option::Some($mark),
            $ceiling,
            ::core::option::Option::Some(::core::format_args!($($arg)+)),
        )
    };
    (since: $mark:expr, $expected:expr $(,)?) => {
        $crate::__assert_alloc_count(
            ::core::option::Option::Some($mark),
            $expected,
            ::core::option::Option::None,
        )
    };
    (since: $mark:expr, $expected:expr, $($arg:tt)+) => {
        $crate::__assert_alloc_count(
            ::core::option::Option::Some($mark),
            $expected,
            ::core::option::Option::Some(::core::format_args!($($arg)+)),
        )
    };
    // Anything else after `since:` is a mistake, the likeliest being a count
    // left out. Without this it falls through to the bare arms and is reported
    // as "no rules expected `:`", which names neither the problem nor the fix.
    (since: $($rest:tt)*) => {
        ::core::compile_error!(
            "assert_alloc_count! takes a count after the mark: \
             `since: mark, n` for exactly n, or `since: mark, <= n` for at most n"
        )
    };
    (<= $ceiling:expr $(,)?) => {
        $crate::__assert_alloc_count_at_most(
            ::core::option::Option::None,
            $ceiling,
            ::core::option::Option::None,
        )
    };
    (<= $ceiling:expr, $($arg:tt)+) => {
        $crate::__assert_alloc_count_at_most(
            ::core::option::Option::None,
            $ceiling,
            ::core::option::Option::Some(::core::format_args!($($arg)+)),
        )
    };
    ($expected:expr $(,)?) => {
        $crate::__assert_alloc_count(
            ::core::option::Option::None,
            $expected,
            ::core::option::Option::None,
        )
    };
    ($expected:expr, $($arg:tt)+) => {
        $crate::__assert_alloc_count(
            ::core::option::Option::None,
            $expected,
            ::core::option::Option::Some(::core::format_args!($($arg)+)),
        )
    };
}

/// Fails if anything allocated since the profiler started is still live.
///
/// ```
/// # #[global_allocator]
/// # static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
/// # fn work() {}
/// # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// work();
/// heapscope::assert_no_leaks!();
/// ```
///
/// # In a program that is already holding memory
///
/// The bare form asks whether the *whole run* is clean, which is the right
/// question for a test that starts its profiler, does one thing, and asserts.
/// It is the wrong question anywhere the program legitimately holds state —
/// caches, lazily initialized statics, a test harness's own buffers — because
/// all of it is live and none of it leaked.
///
/// So there is a second form, taking a [`HeapStats`] read earlier, which asks
/// whether anything is live now that was not live then:
///
/// ```
/// # #[global_allocator]
/// # static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
/// # fn work() {}
/// # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// let before = heapscope::HeapStats::get().unwrap();
/// work();
/// heapscope::assert_no_leaks!(since: before);
/// ```
///
/// That form is a *difference*, so it cannot distinguish a block leaked by
/// `work` from one leaked by a background thread during the same interval. It is
/// still the honest question in a program with a heap that was not empty to
/// begin with.
///
/// Either form takes a trailing [`format_args!`] message —
/// `assert_no_leaks!("after {fixture}")`, or
/// `assert_no_leaks!(since: mark, "after {fixture}")`.
///
/// # Panics
///
/// When blocks are still live, and when there are no numbers to check.
#[macro_export]
macro_rules! assert_no_leaks {
    () => {
        $crate::__assert_no_leaks(::core::option::Option::None, ::core::option::Option::None)
    };
    ($(,)? since: $mark:expr $(,)?) => {
        $crate::__assert_no_leaks(
            ::core::option::Option::Some($mark),
            ::core::option::Option::None,
        )
    };
    (since: $mark:expr, $($arg:tt)+) => {
        $crate::__assert_no_leaks(
            ::core::option::Option::Some($mark),
            ::core::option::Option::Some(::core::format_args!($($arg)+)),
        )
    };
    ($($arg:tt)+) => {
        $crate::__assert_no_leaks(
            ::core::option::Option::None,
            ::core::option::Option::Some(::core::format_args!($($arg)+)),
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internals::engine::Settings;
    use std::num::NonZeroU64;

    /// Held for the whole of any test that reads a counter.
    ///
    /// **Poison is not local.** The engines below are per-test; the poison flag
    /// is process-wide, and a reading refuses a poisoned run — so the one test
    /// here that poisons deliberately is visible to every other test that reads
    /// anything. Found by mutating `HeapStats::of` and watching the mutation
    /// get killed by two tests that have nothing to do with poisoning, which is
    /// how a race announces itself.
    ///
    /// One acquisition per test rather than one per engine, because a test that
    /// compares two engines needs both at once and `RawLock` is not reentrant —
    /// on Apple platforms that is a `SIGKILL` rather than a hang, which is how
    /// the first version of this announced itself.
    fn serialized() -> crate::internals::lock::RawGuard<'static> {
        crate::internals::diagnostic::POISON_TESTS.lock()
    }

    /// An engine nobody has started.
    ///
    /// A local one, not the process-wide singleton: there is one of those and
    /// one test may claim it, so everything here builds its own.
    fn idle() -> Engine {
        // A forgotten `serialized()` is otherwise a flake that appears only
        // when the poisoning test happens to run alongside. Held by us it reads
        // as locked, and under `--test-threads=1` that is exact.
        assert!(
            crate::internals::diagnostic::POISON_TESTS
                .try_lock()
                .is_none(),
            "a test that reads a counter must hold `serialized()` first"
        );
        Engine::with_limits(1 << 10, 1 << 12)
    }

    fn engine(mode: Mode) -> Engine {
        configured(Settings {
            mode,
            ..Settings::default()
        })
    }

    fn configured(settings: Settings) -> Engine {
        let engine = idle();
        assert!(
            engine.start(crate::TimeSource::Events, || engine.configure(settings)),
            "a fresh engine refused to start"
        );
        engine
    }

    fn record(engine: &Engine, address: usize, size: usize) {
        engine.record_alloc_guarded(address, crate::internals::shape::Shape::of(size), &[0x1000]);
    }

    /// A heap run in which **no two of the six figures are equal**.
    ///
    /// This exists because of the single worst defect an adversarial review
    /// found in this module, and it is worth stating plainly: every test here
    /// used to allocate monotonically and free nothing, so at every assertion
    /// point `max_bytes == total_bytes` and `curr_blocks == total_blocks`, and
    /// **the suite could not tell which counter any assertion read**. A
    /// mutation making `assert_max_bytes!` compare `total_bytes`, and one
    /// making `assert_alloc_count!` compare `curr_blocks`, each passed the
    /// entire suite. The two most-used macros in the crate were pinned by their
    /// names and nothing else.
    ///
    /// Freeing *before* the reading is the whole trick. The figures it leaves:
    ///
    /// | | bytes | blocks |
    /// |---|---|---|
    /// | live now | 80 | 2 |
    /// | at the peak | 448 | 3 |
    /// | ever | 464 | 4 |
    fn distinct_figures() -> Engine {
        let engine = engine(Mode::Heap);
        record(&engine, 0x100, 64);
        record(&engine, 0x200, 128);
        record(&engine, 0x300, 256);
        engine.record_free(0x300, 256);
        engine.record_free(0x200, 128);
        record(&engine, 0x400, 16);
        engine
    }

    /// The fixture is only useful while its six figures stay distinct, and a
    /// later edit to it would not otherwise say so.
    #[test]
    fn the_fixture_really_does_separate_every_figure() {
        let _serial = serialized();
        let stats = HeapStats::of(&distinct_figures()).unwrap();
        let figures = [
            stats.curr_bytes,
            stats.curr_blocks,
            stats.max_bytes,
            stats.max_blocks,
            stats.total_bytes,
            stats.total_blocks,
        ];
        for (at, one) in figures.iter().enumerate() {
            for other in &figures[at + 1..] {
                assert_ne!(
                    one, other,
                    "two figures are equal, so a test using this fixture cannot \
                     tell which of them an assertion read: {figures:?}"
                );
            }
        }
    }

    #[test]
    fn a_heap_run_reports_what_it_recorded() {
        let _serial = serialized();
        let stats = HeapStats::of(&distinct_figures()).expect("a running heap engine has stats");

        // Every field, against a run where no two of them agree. Skipping one
        // is how `max_blocks` came to be readable from the live-block counter
        // with the whole suite green.
        assert_eq!(stats.curr_bytes, 80);
        assert_eq!(stats.curr_blocks, 2);
        assert_eq!(stats.max_bytes, 448);
        assert_eq!(stats.max_blocks, 3);
        assert_eq!(stats.total_bytes, 464);
        assert_eq!(stats.total_blocks, 4);
        assert_eq!(stats.dropped_blocks, 0);
    }

    /// The refusal that matters most: an idle engine must not report zeros, or
    /// every assertion in a test that forgot to start a profiler passes.
    #[test]
    fn an_idle_engine_refuses_rather_than_reporting_zeros() {
        let _serial = serialized();
        let idle = idle();
        assert_eq!(HeapStats::of(&idle), Err(StatsError::NotRecording));
        assert_eq!(EventStats::of(&idle), Err(StatsError::NotRecording));
    }

    #[test]
    fn a_finished_run_still_has_final_numbers() {
        let _serial = serialized();
        let engine = engine(Mode::Heap);
        record(&engine, 0x100, 64);
        engine.stop(crate::output::Shutdown::Explicit);

        let stats = HeapStats::of(&engine).expect("a stopped run has final counters");
        assert_eq!(stats.total_bytes, 64);
    }

    /// Each mode answers one of the two questions and refuses the other, so a
    /// budget asserted against the wrong kind of run fails rather than reading a
    /// column that was never measured.
    #[test]
    fn each_mode_refuses_the_other_kind_of_reading() {
        let _serial = serialized();
        let heap = engine(Mode::Heap);
        assert_eq!(EventStats::of(&heap), Err(StatsError::NotAnEventRun));
        assert!(HeapStats::of(&heap).is_ok());

        for mode in [Mode::AdHoc, Mode::Copy] {
            let events = engine(mode);
            assert_eq!(HeapStats::of(&events), Err(StatsError::NotAHeapRun(mode)));
            let stats = EventStats::of(&events).expect("an event run has event statistics");
            assert_eq!(stats.mode, mode);
        }
    }

    #[test]
    fn an_event_run_reports_weight_and_count() {
        let _serial = serialized();
        let engine = engine(Mode::AdHoc);
        let guard = crate::internals::guard::enter().expect("not inside the profiler");
        engine.record_event(&guard, 700, &[0x1000]);
        engine.record_event(&guard, 300, &[0x1000]);
        drop(guard);
        // The refused counter has to move for this to be a reading of anything.
        // Asserted against zero on a run that refused nothing, it was checked
        // only against the value a hardcoded zero would return.
        engine.refuse_event();

        let stats = EventStats::of(&engine).expect("an ad hoc run has event statistics");
        assert_eq!(stats.total_weight, 1_000);
        assert_eq!(stats.total_events, 2);
        assert_eq!(stats.refused_events, 1);
    }

    #[test]
    fn a_forked_child_refuses_the_parents_counters() {
        let _serial = serialized();
        let engine = engine(Mode::Heap);
        record(&engine, 0x100, 64);
        assert!(HeapStats::of(&engine).is_ok());

        engine.disown_for_testing();
        assert_eq!(HeapStats::of(&engine), Err(StatsError::ForkedChild));
    }

    /// A profiler that detected its own corruption stopped recording, so its
    /// counters stopped moving somewhere nobody chose. A profile says so and
    /// prints them anyway; an assertion cannot say so, so it refuses.
    ///
    /// **Both** readings refuse. The event side is a copy of the heap side, and
    /// a copy of a checked path is not itself a checked path: deleting the
    /// poison check from `EventStats::of` passed the whole suite.
    #[test]
    fn a_poisoned_profiler_has_nothing_assertable() {
        // Declared before the flag is set and dropped before the lock is
        // released, so the poison this test raises is invisible outside it
        // whether the test passes or panics.
        let _serial = serialized();
        let _clear = ClearPoison;

        let heap = engine(Mode::Heap);
        let events = engine(Mode::AdHoc);
        crate::internals::diagnostic::set_quiet(true);
        record(&heap, 0x100, 64);
        assert!(HeapStats::of(&heap).is_ok());
        assert!(EventStats::of(&events).is_ok());

        crate::internals::diagnostic::poison("test: the assertions must refuse this");
        assert_eq!(HeapStats::of(&heap), Err(StatsError::Poisoned));
        assert_eq!(EventStats::of(&events), Err(StatsError::Poisoned));
        assert_eq!(
            check_max_bytes(&heap, u64::MAX),
            Err(Complaint::Unavailable(StatsError::Poisoned))
        );
    }

    /// The mode is reported before the poison, and the reason is in
    /// `unpoisoned`'s documentation: asking a copy run for a heap peak is a
    /// mistake in the test that its author can fix, and naming the poison first
    /// would send them looking for a fault in the profiler instead. Only a run
    /// that is both can tell the two orderings apart.
    #[test]
    fn a_wrong_mode_is_reported_before_a_poison() {
        let _serial = serialized();
        let _clear = ClearPoison;

        let events = engine(Mode::AdHoc);
        crate::internals::diagnostic::set_quiet(true);
        crate::internals::diagnostic::poison("test: both wrong at once");

        assert_eq!(
            HeapStats::of(&events),
            Err(StatsError::NotAHeapRun(Mode::AdHoc))
        );
    }

    /// Clears the poison flag however the test that raised it ends.
    ///
    /// A failing assertion unwinds before any line after it, so a test that
    /// cleared the flag on its last line would leave it set on the run where it
    /// broke — and every later test that reads a counter would fail too,
    /// reporting a poisoned profiler when what happened is that one test broke.
    /// Found by mutation: deleting the poison check from `HeapStats::of` was
    /// killed by this test *and* by two with nothing to do with poisoning,
    /// which is a cascade rather than coverage.
    struct ClearPoison;

    impl Drop for ClearPoison {
        fn drop(&mut self) {
            crate::internals::diagnostic::reset();
        }
    }

    #[test]
    fn the_budget_passes_at_the_limit_and_fails_above_it() {
        let _serial = serialized();
        let engine = distinct_figures();

        assert_eq!(check_max_bytes(&engine, 448), Ok(()));
        assert_eq!(
            check_max_bytes(&engine, 447),
            Err(Complaint::OverBudget {
                peak: 448,
                limit: 447
            })
        );
    }

    /// The budget is about the peak, and the peak is not any of the other five
    /// figures. `total_bytes` is the one that matters here: it is 464 on this
    /// run against a peak of 448, so a budget of 448 passes only if the peak is
    /// what is being compared.
    #[test]
    fn the_budget_is_the_peak_and_not_the_live_or_cumulative_figure() {
        let _serial = serialized();
        let engine = distinct_figures();
        let stats = HeapStats::of(&engine).unwrap();
        assert!(stats.total_bytes > stats.max_bytes);
        assert!(stats.curr_bytes < stats.max_bytes);

        // Passes against the peak; would fail against `total_bytes` and pass
        // vacuously against `curr_bytes`.
        assert_eq!(check_max_bytes(&engine, stats.max_bytes), Ok(()));
        assert!(check_max_bytes(&engine, stats.curr_bytes).is_err());
    }

    /// A program that frees everything still had a peak, and that is what a
    /// memory budget is about.
    #[test]
    fn freeing_everything_does_not_lower_the_budget() {
        let _serial = serialized();
        let engine = engine(Mode::Heap);
        record(&engine, 0x100, 4_096);
        engine.record_free(0x100, 4_096);

        assert_eq!(HeapStats::of(&engine).unwrap().curr_bytes, 0);
        assert_eq!(
            check_max_bytes(&engine, 1_024),
            Err(Complaint::OverBudget {
                peak: 4_096,
                limit: 1_024
            })
        );
    }

    #[test]
    fn the_count_is_an_equality_in_both_directions() {
        let _serial = serialized();
        let engine = distinct_figures();

        assert_eq!(
            check_alloc_count(&engine, None, Expected::Exactly(4)),
            Ok(())
        );
        for wrong in [5, 3, 0] {
            assert_eq!(
                check_alloc_count(&engine, None, Expected::Exactly(wrong)),
                Err(Complaint::WrongCount {
                    counted: 4,
                    expected: wrong,
                    scope: Scope::WholeRun
                })
            );
        }
    }

    /// Allocations ever made, not blocks still live. On this run those are 4
    /// and 2, so a check reading the live figure would pass `2` — which is the
    /// "passes a run that allocated nothing" failure the equality exists to
    /// prevent, one column over. The ceiling reads the same column, and a
    /// ceiling of `2` is where reading the wrong one would pass.
    #[test]
    fn the_count_is_of_allocations_rather_than_of_live_blocks() {
        let _serial = serialized();
        let engine = distinct_figures();
        let stats = HeapStats::of(&engine).unwrap();
        assert!(stats.total_blocks > stats.curr_blocks);

        assert_eq!(
            check_alloc_count(&engine, None, Expected::Exactly(stats.total_blocks)),
            Ok(())
        );
        assert!(check_alloc_count(&engine, None, Expected::Exactly(stats.curr_blocks)).is_err());
        assert!(check_alloc_count(&engine, None, Expected::AtMost(stats.curr_blocks)).is_err());
    }

    /// A ceiling admits its own value and everything under it, and nothing
    /// over. Pinned at the edge, because `<` for `<=` is the mutation a
    /// ceiling invites and only the edge can see it.
    #[test]
    fn the_ceiling_passes_at_and_under_it_and_fails_above_it() {
        let _serial = serialized();
        let engine = distinct_figures();

        for ceiling in [4, 5, u64::MAX] {
            assert_eq!(
                check_alloc_count(&engine, None, Expected::AtMost(ceiling)),
                Ok(()),
                "a ceiling of {ceiling} refused 4 allocations"
            );
        }
        for ceiling in [3, 0] {
            assert_eq!(
                check_alloc_count(&engine, None, Expected::AtMost(ceiling)),
                Err(Complaint::OverCeiling {
                    counted: 4,
                    ceiling,
                    scope: Scope::WholeRun
                })
            );
        }
    }

    /// The count since a mark is what the run made after it, in both forms, and
    /// a complaint about it says that it was counted from the mark.
    #[test]
    fn a_mark_counts_only_what_followed_it() {
        let _serial = serialized();
        let engine = engine(Mode::Heap);
        record(&engine, 0x100, 64);
        record(&engine, 0x200, 64);
        let mark = HeapStats::of(&engine).unwrap();

        // Nothing made since: the reading equals the mark, which is a count of
        // zero rather than a mark ahead of the run.
        assert_eq!(
            check_alloc_count(&engine, Some(mark), Expected::Exactly(0)),
            Ok(())
        );
        assert_eq!(
            check_alloc_count(&engine, Some(mark), Expected::AtMost(0)),
            Ok(())
        );

        record(&engine, 0x300, 64);
        record(&engine, 0x400, 64);
        record(&engine, 0x500, 64);
        assert_eq!(
            check_alloc_count(&engine, Some(mark), Expected::Exactly(3)),
            Ok(())
        );
        assert_eq!(
            check_alloc_count(&engine, Some(mark), Expected::AtMost(3)),
            Ok(())
        );
        // Five is the whole run's count, so passing it would mean the mark was
        // ignored.
        assert_eq!(
            check_alloc_count(&engine, Some(mark), Expected::Exactly(5)),
            Err(Complaint::WrongCount {
                counted: 3,
                expected: 5,
                scope: Scope::SinceMark
            })
        );
        assert_eq!(
            check_alloc_count(&engine, Some(mark), Expected::AtMost(2)),
            Err(Complaint::OverCeiling {
                counted: 3,
                ceiling: 2,
                scope: Scope::SinceMark
            })
        );
        assert_eq!(
            check_alloc_count(&engine, None, Expected::Exactly(5)),
            Ok(())
        );
    }

    /// Allocations made since the mark, not the change in live blocks. Each
    /// block made after the mark here is freed before the reading, so the live
    /// figure is back where it started and a check reading it would find
    /// nothing — passing `since: mark, 0` on a stage that allocated twice.
    #[test]
    fn a_mark_counts_allocations_rather_than_the_change_in_live_blocks() {
        let _serial = serialized();
        let engine = engine(Mode::Heap);
        record(&engine, 0x100, 64);
        let mark = HeapStats::of(&engine).unwrap();
        record(&engine, 0x200, 32);
        engine.record_free(0x200, 32);
        record(&engine, 0x300, 32);
        engine.record_free(0x300, 32);
        assert_eq!(
            HeapStats::of(&engine).unwrap().curr_blocks,
            mark.curr_blocks
        );

        assert_eq!(
            check_alloc_count(&engine, Some(mark), Expected::Exactly(2)),
            Ok(())
        );
        assert!(check_alloc_count(&engine, Some(mark), Expected::Exactly(0)).is_err());
        assert_eq!(
            check_alloc_count(&engine, Some(mark), Expected::AtMost(1)),
            Err(Complaint::OverCeiling {
                counted: 2,
                ceiling: 1,
                scope: Scope::SinceMark
            })
        );
    }

    /// A mark ahead of the reading did not come from the counters being read,
    /// so the count since it is unknown. Refused, not saturated to zero — zero
    /// would pass `since: mark, 0` and every ceiling — and refused before any
    /// comparison is made, which the widest ceiling there is shows.
    ///
    /// Two engines stand in for two runs, which one process cannot otherwise
    /// produce: that is the only way to reach this, and why it is checked here
    /// rather than through the macro.
    #[test]
    fn a_mark_ahead_of_the_run_is_refused_rather_than_saturated() {
        let _serial = serialized();
        let earlier = engine(Mode::Heap);
        for address in [0x100, 0x200, 0x300, 0x400, 0x500] {
            record(&earlier, address, 16);
        }
        let mark = HeapStats::of(&earlier).unwrap();
        let later = engine(Mode::Heap);
        record(&later, 0x100, 16);
        record(&later, 0x200, 16);

        for expected in [
            Expected::Exactly(0),
            Expected::Exactly(3),
            Expected::AtMost(0),
            Expected::AtMost(u64::MAX),
        ] {
            assert_eq!(
                check_alloc_count(&later, Some(mark), expected),
                Err(Complaint::MarkAhead { mark: 5, now: 2 }),
                "{expected:?}"
            );
        }
    }

    #[test]
    fn a_live_block_is_a_leak_and_a_freed_one_is_not() {
        let _serial = serialized();
        let engine = engine(Mode::Heap);
        record(&engine, 0x100, 64);
        assert_eq!(
            check_no_leaks(&engine, None),
            Err(Complaint::Leaked {
                blocks: 1,
                live_bytes: 64,
                mark_bytes: None
            })
        );

        engine.record_free(0x100, 64);
        assert_eq!(check_no_leaks(&engine, None), Ok(()));
    }

    /// Blocks decide, not bytes. A live zero-sized allocation is a block that
    /// was never freed and contributes nothing to `curr_bytes`, so a check
    /// gated on bytes would report it clean.
    #[test]
    fn a_zero_sized_block_is_still_a_leak() {
        let _serial = serialized();
        let engine = engine(Mode::Heap);
        record(&engine, 0x100, 0);

        assert_eq!(
            check_no_leaks(&engine, None),
            Err(Complaint::Leaked {
                blocks: 1,
                live_bytes: 0,
                mark_bytes: None
            })
        );
    }

    /// The `since` form asks what changed, so memory that was already held when
    /// the mark was taken is not reported as this interval's leak.
    #[test]
    fn a_mark_excludes_what_was_already_live() {
        let _serial = serialized();
        let engine = engine(Mode::Heap);
        record(&engine, 0x100, 64);
        let mark = HeapStats::of(&engine).unwrap();

        assert_eq!(check_no_leaks(&engine, Some(mark)), Ok(()));
        assert!(check_no_leaks(&engine, None).is_err());

        record(&engine, 0x200, 32);
        assert_eq!(
            check_no_leaks(&engine, Some(mark)),
            Err(Complaint::Leaked {
                blocks: 1,
                live_bytes: 96,
                mark_bytes: Some(64)
            })
        );
    }

    /// Freeing something that was live at the mark is not a leak of a negative
    /// number of blocks.
    #[test]
    fn a_mark_taken_before_a_free_reports_nothing() {
        let _serial = serialized();
        let engine = engine(Mode::Heap);
        record(&engine, 0x100, 64);
        let mark = HeapStats::of(&engine).unwrap();
        engine.record_free(0x100, 64);

        assert_eq!(check_no_leaks(&engine, Some(mark)), Ok(()));
    }

    /// The two live counters are not a pair that can be subtracted, and the
    /// first version of this subtracted both. A large block freed and a small
    /// one allocated across the mark leaves one more live block and *fewer*
    /// live bytes, which reported "1 blocks totalling 0 bytes were never
    /// freed" — a sentence that fails a test and cannot be true.
    #[test]
    fn a_leak_across_a_shrinking_heap_still_reads_as_a_sentence() {
        let _serial = serialized();
        let engine = engine(Mode::Heap);
        record(&engine, 0x100, 65_536);
        let mark = HeapStats::of(&engine).unwrap();
        engine.record_free(0x100, 65_536);
        record(&engine, 0x200, 8);
        record(&engine, 0x300, 8);

        let complaint = check_no_leaks(&engine, Some(mark)).expect_err("a block was leaked");
        assert_eq!(
            complaint,
            Complaint::Leaked {
                blocks: 1,
                live_bytes: 16,
                mark_bytes: Some(65_536)
            }
        );
        let message = complaint.to_string();
        assert!(message.contains("1 more block is live"), "{message}");
        assert!(message.contains("65,536"), "{message}");
        assert!(message.contains("16"), "{message}");
        assert!(
            !message.contains("totalling 0 bytes"),
            "the byte figures are not a difference: {message}"
        );
    }

    /// Every assertion refuses an incomplete measurement rather than comparing
    /// against it. A run that dropped blocks is missing them from the peak, the
    /// count, and the live figure alike, so a passing budget would mean nothing.
    #[test]
    fn a_run_that_dropped_blocks_is_not_assertable() {
        let _serial = serialized();
        let engine = configured(Settings {
            max_live_blocks: 1,
            ..Settings::default()
        });
        // The ceiling rounds up to whatever the shards can express, so fill it
        // by recording until the engine says it turned one away.
        let mut address = 0x1000;
        while HeapStats::of(&engine).unwrap().dropped_blocks == 0 {
            record(&engine, address, 16);
            address += 0x10;
            assert!(address < 0x1000_0000, "the ceiling was never reached");
        }
        let dropped = HeapStats::of(&engine).unwrap().dropped_blocks;

        let incomplete = Err(Complaint::Incomplete {
            dropped_blocks: dropped,
        });
        assert_eq!(check_max_bytes(&engine, u64::MAX), incomplete);
        // Every form of the count, and the widest ceiling from a mark equal to
        // the reading in particular: that one passes unless the gate comes
        // before the arithmetic.
        let mark = HeapStats::of(&engine).unwrap();
        for since in [None, Some(mark)] {
            for expected in [Expected::Exactly(0), Expected::AtMost(u64::MAX)] {
                assert_eq!(check_alloc_count(&engine, since, expected), incomplete);
            }
        }
        assert_eq!(check_no_leaks(&engine, None), incomplete);
        // `assert_baseline!` goes through the same gate, and used to not: it
        // called `HeapStats::of` directly, so the one assertion aimed at CI
        // passed on the run where the measurement was incomplete.
        assert_eq!(assertable(&engine).map(|_| ()), incomplete);
    }

    /// Sampled counters are estimates, so nothing that asserts against a budget
    /// may read them.
    ///
    /// Every assertion in this crate goes through one of the two readings, so
    /// refusing there is what makes the whole family refuse. `tests/sampling.rs`
    /// checks the heap arm against a real run; this checks both arms and every
    /// assertion built on them, which one process cannot do because it can only
    /// be one mode at a time.
    #[test]
    fn a_sampled_run_is_not_assertable() {
        let _serial = serialized();

        let heap = configured(Settings {
            sampling: NonZeroU64::new(4_096),
            ..Settings::default()
        });
        record(&heap, 0x1000, 64);
        assert_eq!(HeapStats::of(&heap), Err(StatsError::Sampled));

        // Refused before the counters are read, so a poisoned *and* sampled run
        // still names sampling: the run is unassertable for a reason its author
        // chose, and reporting the poison would send them to look for a fault in
        // the profiler.
        assert_eq!(
            check_max_bytes(&heap, u64::MAX),
            Err(Complaint::Unavailable(StatsError::Sampled))
        );
        assert_eq!(
            check_alloc_count(&heap, None, Expected::Exactly(0)),
            Err(Complaint::Unavailable(StatsError::Sampled))
        );
        assert_eq!(
            check_alloc_count(&heap, None, Expected::AtMost(u64::MAX)),
            Err(Complaint::Unavailable(StatsError::Sampled))
        );
        assert_eq!(
            check_no_leaks(&heap, None),
            Err(Complaint::Unavailable(StatsError::Sampled))
        );
        assert_eq!(
            assertable(&heap).map(|_| ()),
            Err(Complaint::Unavailable(StatsError::Sampled))
        );

        // And the event arm, which `tests/sampling.rs` cannot reach: there the
        // mode check fires first and correctly, because a heap run asked for
        // event counters is a mistake with a nearer cause than sampling.
        let events = configured(Settings {
            mode: Mode::AdHoc,
            sampling: NonZeroU64::new(4_096),
            ..Settings::default()
        });
        assert_eq!(EventStats::of(&events), Err(StatsError::Sampled));
        assert_eq!(
            EventStats::of(&engine(Mode::Heap)),
            Err(StatsError::NotAnEventRun),
            "the mode check must still win on a run that does not sample"
        );
    }

    /// Records `size` bytes at `address` with `name` the innermost region on
    /// this thread, the way `crate::region` would arrange it.
    fn record_in(engine: &Engine, name: &str, address: usize, size: usize) {
        let id = engine.intern_region(name);
        let held = crate::internals::guard::enter().expect("not inside the profiler");
        let previous = crate::internals::guard::enter_region(&held, id);
        drop(held);
        engine.regions().enter(id);
        record(engine, address, size);
        crate::internals::guard::leave_region(previous);
        engine.regions().leave(id);
    }

    /// The breakdown is the region rows a snapshot would carry, plus the
    /// remainder, and the two together are the run.
    ///
    /// Built on `distinct_figures`, so the remainder's live and total columns
    /// differ from each other and from the region's: an accessor that read the
    /// wrong counter into the wrong field cannot land on the right answer.
    #[test]
    fn a_region_breakdown_adds_up_to_the_run() {
        let _serial = serialized();
        let engine = distinct_figures();
        record_in(&engine, "parsing", 0x500, 1_000);
        record_in(&engine, "parsing", 0x600, 24);
        engine.record_free(0x600, 24);

        let breakdown = RegionBreakdown::of(&engine).expect("a running heap engine has regions");
        assert_eq!(breakdown.mode, Mode::Heap);
        assert_eq!(breakdown.regions.len(), 1);
        let parsing = breakdown.region("parsing").expect("the program entered it");
        assert_eq!(parsing.entries, 2);
        assert_eq!(parsing.counts.total_bytes, 1_024);
        assert_eq!(parsing.counts.curr_bytes, 1_000);

        // Everything `distinct_figures` recorded happened outside every region.
        assert_eq!(breakdown.outside.total_bytes, 464);
        assert_eq!(breakdown.outside.total_blocks, 4);
        assert_eq!(breakdown.outside.curr_bytes, 80);
        assert_eq!(breakdown.outside.curr_blocks, 2);

        let stats = HeapStats::of(&engine).expect("heap stats");
        assert_eq!(
            parsing.counts.total_bytes + breakdown.outside.total_bytes,
            stats.total_bytes
        );
        assert_eq!(
            parsing.counts.curr_blocks + breakdown.outside.curr_blocks,
            stats.curr_blocks
        );
    }

    /// The same rows a snapshot carries, not a parallel reading of them: a field
    /// copied differently in one place is a disagreement between two answers to
    /// one question.
    #[test]
    fn a_region_breakdown_agrees_with_a_snapshot() {
        let _serial = serialized();
        let engine = distinct_figures();
        record_in(&engine, "parsing", 0x500, 1_000);
        record_in(&engine, "emitting", 0x600, 24);

        let breakdown = RegionBreakdown::of(&engine).expect("regions");
        let snapshot = Snapshot::of(&engine);
        assert_eq!(breakdown.regions, snapshot.regions);
        assert_eq!(breakdown.outside, snapshot.outside_regions);
    }

    /// A name is found the way the program entered it, however long it was:
    /// `region` keeps 64 bytes, and a lookup that did not cut the same way would
    /// report a region the program plainly entered as never entered.
    #[test]
    fn a_region_is_found_by_the_name_the_program_gave_it() {
        let _serial = serialized();
        let engine = engine(Mode::Heap);
        let long = "a phase name long enough to be cut by the profiler, and then some more";
        assert!(long.len() > crate::internals::site::MAX_NAME);
        record_in(&engine, long, 0x100, 64);
        record_in(&engine, "", 0x200, 32);

        let breakdown = RegionBreakdown::of(&engine).expect("regions");
        assert_eq!(
            breakdown.region(long).map(|row| row.counts.total_bytes),
            Some(64)
        );
        assert_eq!(
            breakdown.region("").map(|row| row.counts.total_bytes),
            Some(32),
            "a region named with the empty string is still a region"
        );
        assert!(breakdown.region("never entered").is_none());
    }

    /// With no region entered, the remainder is the whole run, which is the
    /// true answer rather than a missing one.
    #[test]
    fn a_run_with_no_regions_is_entirely_outside_them() {
        let _serial = serialized();
        let engine = distinct_figures();
        let breakdown = RegionBreakdown::of(&engine).expect("regions");
        assert!(breakdown.regions.is_empty());
        let stats = HeapStats::of(&engine).expect("heap stats");
        assert_eq!(breakdown.outside.total_bytes, stats.total_bytes);
        assert_eq!(breakdown.outside.curr_bytes, stats.curr_bytes);
    }

    /// Region rows mean the same thing in every mode, so an event run has a
    /// breakdown too, and says it is one.
    #[test]
    fn an_event_run_has_a_region_breakdown_in_its_own_units() {
        let _serial = serialized();
        let engine = engine(Mode::AdHoc);
        let id = engine.intern_region("retrying");
        let held = crate::internals::guard::enter().expect("not inside the profiler");
        engine.record_event(&held, 700, &[0x1000]);
        let previous = crate::internals::guard::enter_region(&held, id);
        engine.regions().enter(id);
        engine.record_event(&held, 30, &[0x1000]);
        crate::internals::guard::leave_region(previous);
        engine.regions().leave(id);
        drop(held);

        let breakdown = RegionBreakdown::of(&engine).expect("an event run has regions");
        assert_eq!(breakdown.mode, Mode::AdHoc);
        assert_eq!(breakdown.outside.total_bytes, 700);
        assert_eq!(breakdown.outside.total_blocks, 1);
        assert_eq!(
            breakdown
                .region("retrying")
                .map(|row| row.counts.total_bytes),
            Some(30)
        );
    }

    /// Every refusal that is about the run rather than about the reading's
    /// mode, on the same terms as `HeapStats::of`. A breakdown of zeros from a
    /// profiler nobody started would say every phase allocated nothing.
    #[test]
    fn a_region_breakdown_refuses_what_the_other_readings_refuse() {
        let _serial = serialized();
        assert_eq!(RegionBreakdown::of(&idle()), Err(StatsError::NotRecording));

        let sampled = configured(Settings {
            sampling: NonZeroU64::new(4_096),
            ..Settings::default()
        });
        assert_eq!(RegionBreakdown::of(&sampled), Err(StatsError::Sampled));

        let forked = engine(Mode::Heap);
        record_in(&forked, "parsing", 0x100, 64);
        forked.disown_for_testing();
        assert_eq!(RegionBreakdown::of(&forked), Err(StatsError::ForkedChild));

        let finished = engine(Mode::Heap);
        record_in(&finished, "parsing", 0x100, 64);
        finished.stop(crate::output::Shutdown::Explicit);
        assert!(
            RegionBreakdown::of(&finished).is_ok(),
            "a stopped run still has final numbers"
        );
    }

    /// As for the other readings, a poisoned run is refused rather than read.
    /// A copy of a checked path is not itself a checked path, which is why this
    /// is its own test rather than an assumption.
    #[test]
    fn a_poisoned_profiler_has_no_region_breakdown() {
        let _serial = serialized();
        let _clear = ClearPoison;

        let heap = engine(Mode::Heap);
        record_in(&heap, "parsing", 0x100, 64);
        crate::internals::diagnostic::set_quiet(true);
        assert!(RegionBreakdown::of(&heap).is_ok());

        crate::internals::diagnostic::poison("test: the breakdown must refuse this");
        assert_eq!(RegionBreakdown::of(&heap), Err(StatsError::Poisoned));
    }

    /// A run with nothing to describe writes no profile, and a finished one
    /// still has something to describe — the documented shape of a test that
    /// asserts after its profiler is dropped.
    #[test]
    fn only_a_run_that_happened_is_worth_dumping() {
        let _serial = serialized();
        assert!(!has_a_profile(&idle()));

        let engine = engine(Mode::Heap);
        assert!(has_a_profile(&engine));
        engine.stop(crate::output::Shutdown::Explicit);
        assert!(
            has_a_profile(&engine),
            "a finished run still has numbers a reader would want the sites for"
        );

        engine.disown_for_testing();
        assert!(!has_a_profile(&engine), "the profile belongs to the parent");
    }

    #[test]
    fn the_dump_setting_chooses_a_path_or_refuses_one() {
        let named = dump_base(Some(std::ffi::OsStr::new("/tmp/p.json")));
        assert_eq!(named, Some(PathBuf::from("/tmp/p.json")));

        for off in ["0", "off", "OFF", "No", "FALSE"] {
            assert_eq!(
                dump_base(Some(std::ffi::OsStr::new(off))),
                None,
                "{off} did not read as off"
            );
        }

        let default = dump_base(None).expect("an unset variable dumps by default");
        let name = default.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("heapscope-assert"), "{name}");
        assert!(name.ends_with(".json"), "{name}");
    }

    /// The failure report is the whole of what a reader gets, and until an
    /// adversarial review pointed it out nothing read a byte of it: deleting
    /// the summary, or asking for zero program points, left every test green.
    #[test]
    #[cfg_attr(miri, ignore = "writes a profile, and Miri has no filesystem")]
    fn a_dump_writes_a_summary_and_a_profile_and_names_it() {
        let _serial = serialized();
        let engine = distinct_figures();
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("failure.json");
        let guard = crate::internals::guard::enter().expect("not inside the profiler");

        let mut summary = Vec::new();
        let line = dump(&engine, &guard, &path, Ranking::Bytes, &mut summary);
        drop(guard);

        let summary = String::from_utf8(summary).expect("the summary is text");
        assert!(
            summary.contains("heapscope"),
            "no summary was written: {summary:?}"
        );
        assert!(
            summary.contains("  1."),
            "the summary listed no program points, so `TOP_ON_FAILURE` reached \
             the reader as zero: {summary}"
        );

        assert!(line.contains("profile written to"), "{line}");
        assert!(line.contains(&path.display().to_string()), "{line}");
        let profile = std::fs::read_to_string(&path).expect("the profile it named");
        assert!(profile.contains("\"dhatFileVersion\""), "{profile:.200}");
    }

    /// A path that cannot be written must say so. Silence there is
    /// indistinguishable from dumping being switched off, and sends a reader
    /// looking for a file that was never written.
    #[test]
    #[cfg_attr(miri, ignore = "writes a profile, and Miri has no filesystem")]
    fn a_dump_that_cannot_be_written_says_so() {
        let _serial = serialized();
        let engine = distinct_figures();
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("no-such-directory").join("p.json");
        let guard = crate::internals::guard::enter().expect("not inside the profiler");

        let line = dump(&engine, &guard, &path, Ranking::Bytes, &mut Vec::new());
        drop(guard);

        assert!(line.contains("could not write a profile"), "{line}");
        assert!(line.contains(&path.display().to_string()), "{line}");
    }

    /// The dump path comes from the caller, through an environment variable,
    /// and ends up in a panic message on its way to a terminal.
    #[test]
    #[cfg_attr(miri, ignore = "writes a profile, and Miri has no filesystem")]
    fn a_dump_path_cannot_drive_the_terminal() {
        let _serial = serialized();
        let engine = distinct_figures();
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("run\u{1b}[2Kmasked.json");
        let guard = crate::internals::guard::enter().expect("not inside the profiler");

        let line = dump(&engine, &guard, &path, Ranking::Bytes, &mut Vec::new());
        drop(guard);

        assert!(!line.contains('\u{1b}'), "{line}");
    }

    /// A second dump must not land on the first, or a panic message names a file
    /// another test has since replaced.
    #[test]
    fn dumps_after_the_first_take_a_name_of_their_own() {
        let path = PathBuf::from("/tmp/heapscope-assert.json");
        assert_eq!(distinguish(path.clone(), 0), path);
        assert_eq!(
            distinguish(path.clone(), 1),
            PathBuf::from("/tmp/heapscope-assert.2.json")
        );
        assert_eq!(
            distinguish(path, 2),
            PathBuf::from("/tmp/heapscope-assert.3.json")
        );

        let bare = PathBuf::from("profile");
        assert_eq!(distinguish(bare.clone(), 1), PathBuf::from("profile.2"));
        assert_eq!(distinguish(bare, 0), PathBuf::from("profile"));

        // Nothing sensible can be derived from a path with no file name, and
        // inventing one would write somewhere the caller did not name.
        let root = PathBuf::from("/");
        assert_eq!(distinguish(root.clone(), 1), root);
    }

    /// A path naming a directory has no file name to distinguish, and
    /// `file_stem` answers as though it did — so the second dump was written as
    /// a *sibling of* the directory the caller named. With the variable set to
    /// `/tmp/`, that put a file at the filesystem root.
    #[test]
    fn a_directory_is_never_turned_into_a_file_beside_it() {
        for directory in ["/tmp/dumps/", "/tmp/"] {
            let path = PathBuf::from(directory);
            assert_eq!(
                distinguish(path.clone(), 1),
                path,
                "{directory} was distinguished into a sibling"
            );
        }

        // A directory named without a trailing separator is indistinguishable
        // from a file by its name alone, so `distinguish` cannot see it and
        // does not try. `dump_target` asks the filesystem instead.
        assert_eq!(
            distinguish(PathBuf::from("/tmp/dumps"), 1),
            PathBuf::from("/tmp/dumps.2")
        );
    }

    /// A test name is a path, and a profile is a file.
    #[test]
    fn a_thread_name_becomes_something_a_file_system_accepts() {
        let named = std::thread::Builder::new()
            .name("stats::tests::budgets/one".to_string())
            .spawn(thread_suffix)
            .expect("a named thread")
            .join()
            .expect("the thread panicked");
        assert_eq!(named, "-stats__tests__budgets_one");

        // The characters a file name may keep are kept. Replacing these too
        // would leave `tokio-runtime-worker` as `tokio_runtime_worker`, which
        // is no longer the name the reader is looking for.
        let kept = std::thread::Builder::new()
            .name("tokio-runtime.worker_3".to_string())
            .spawn(thread_suffix)
            .expect("a named thread")
            .join()
            .expect("the thread panicked");
        assert_eq!(kept, "-tokio-runtime.worker_3");

        let long = "x".repeat(MAX_THREAD_SUFFIX * 2);
        let cut = std::thread::Builder::new()
            .name(long)
            .spawn(thread_suffix)
            .expect("a named thread")
            .join()
            .expect("the thread panicked");
        assert_eq!(cut.len(), MAX_THREAD_SUFFIX + 1);
    }

    #[test]
    fn dumping_can_be_turned_off_the_way_symbolization_can() {
        for off in ["0", "off", "no", "false", " off ", "OFF", "False", "NO"] {
            assert!(is_off(std::ffi::OsStr::new(off)), "{off:?}");
        }
        for on in ["1", "", "yes", "/tmp/profile.json", "offer"] {
            assert!(!is_off(std::ffi::OsStr::new(on)), "{on:?}");
        }
    }

    /// The two variables this crate reads for the same purpose must read the
    /// same spellings. They did not: this one folded no case while
    /// `symbol::dynamic` folded to lowercase, so `HEAPSCOPE_UPDATE_BASELINE=FALSE`
    /// read as *on* and rewrote every baseline it should have checked.
    #[test]
    fn off_means_the_same_thing_here_as_it_does_for_symbolization() {
        for spelling in [
            "0", "off", "OFF", "Off", "no", "NO", "false", "FALSE", "False", " off ",
        ] {
            assert_eq!(
                is_off(std::ffi::OsStr::new(spelling)),
                crate::symbol::dynamic::reads_as_off(spelling),
                "the two readings of {spelling:?} disagree"
            );
        }
        for spelling in ["1", "on", "yes", "", "offer", "/tmp/p.json"] {
            assert_eq!(
                is_off(std::ffi::OsStr::new(spelling)),
                crate::symbol::dynamic::reads_as_off(spelling),
                "the two readings of {spelling:?} disagree"
            );
        }
    }

    /// A budget of `-1` saturating to `u64::MAX` would be an assertion that
    /// cannot fail, which is the one outcome this module exists to prevent.
    #[test]
    #[should_panic(expected = "not a byte count")]
    fn a_negative_limit_is_refused_rather_than_saturated() {
        as_count(-1i64);
    }

    /// The other way a value does not fit, which only the 128-bit types reach.
    #[test]
    #[should_panic(expected = "not a byte count")]
    fn an_oversized_limit_is_refused_rather_than_truncated() {
        as_count(u128::from(u64::MAX) + 1);
    }

    #[test]
    fn a_limit_can_be_any_of_the_integer_types_a_call_site_has() {
        assert_eq!(as_count(64u8), 64);
        assert_eq!(as_count(64u16), 64);
        assert_eq!(as_count(64u32), 64);
        assert_eq!(as_count(64u64), 64);
        assert_eq!(as_count(64u128), 64);
        assert_eq!(as_count(64usize), 64);
        assert_eq!(as_count(64i8), 64);
        assert_eq!(as_count(64i16), 64);
        assert_eq!(as_count(64i32), 64);
        assert_eq!(as_count(64i64), 64);
        assert_eq!(as_count(64i128), 64);
        assert_eq!(as_count(64isize), 64);
        assert_eq!(as_count(u64::MAX), u64::MAX);
    }

    /// `NonZeroU64` satisfied the old `TryInto<u64>` bound, so a budget held
    /// in one has to keep compiling; the other widths come with it, and a
    /// negative one is refused like the integer it wraps.
    #[test]
    fn a_limit_can_be_a_non_zero_integer() {
        use std::num::NonZero;
        assert_eq!(as_count(NonZero::<u64>::MAX), u64::MAX);
        assert_eq!(as_count(NonZero::new(64usize).unwrap()), 64);
        assert_eq!(as_count(NonZero::new(64i8).unwrap()), 64);
        assert_eq!(as_count(NonZero::new(64u128).unwrap()), 64);
    }

    #[test]
    #[should_panic(expected = "not a byte count")]
    fn a_negative_non_zero_limit_is_refused() {
        as_count(std::num::NonZero::new(-1i32).unwrap());
    }

    /// Every complaint has to read as a sentence naming both numbers, because it
    /// is the whole of what a failing CI job shows.
    #[test]
    fn every_complaint_names_the_numbers_behind_it() {
        // Each figure is checked with the words around it, not on its own: two
        // numbers in a sentence are two numbers whichever way round they are
        // printed, and printing the budget as the peak is a message that reads
        // perfectly and says the opposite of what happened.
        let over = Complaint::OverBudget {
            peak: 1_234_567,
            limit: 1_048_576,
        }
        .to_string();
        assert!(over.contains("reached 1,234,567"), "{over}");
        assert!(over.contains("limit of 1,048,576"), "{over}");

        let wrong = Complaint::WrongCount {
            counted: 5,
            expected: 3,
            scope: Scope::WholeRun,
        }
        .to_string();
        assert!(wrong.contains("5 allocations were made, not 3"), "{wrong}");

        // A count since a mark has to say so, or a stage's count reads as the
        // whole run's and the reader goes looking for allocations that were
        // never in it.
        let staged = Complaint::WrongCount {
            counted: 5,
            expected: 3,
            scope: Scope::SinceMark,
        }
        .to_string();
        assert!(
            staged.contains("5 allocations were made since the mark, not 3"),
            "{staged}"
        );

        let ceiling = Complaint::OverCeiling {
            counted: 7,
            ceiling: 4,
            scope: Scope::WholeRun,
        }
        .to_string();
        assert!(
            ceiling.contains("7 allocations were made, above the ceiling of 4"),
            "{ceiling}"
        );
        let staged_ceiling = Complaint::OverCeiling {
            counted: 1_234,
            ceiling: 4,
            scope: Scope::SinceMark,
        }
        .to_string();
        assert!(
            staged_ceiling
                .contains("1,234 allocations were made since the mark, above the ceiling of 4"),
            "{staged_ceiling}"
        );

        let ahead = Complaint::MarkAhead {
            mark: 1_000,
            now: 900,
        }
        .to_string();
        assert!(ahead.contains("records 1,000 allocations"), "{ahead}");
        assert!(ahead.contains("this run has made 900"), "{ahead}");
        assert!(
            ahead.contains("HeapStats::get()"),
            "a refusal has to name the remedy: {ahead}"
        );
        // The numbers cannot say whether the mark came from another run or
        // was edited after it was read, so the message must not pick one.
        assert!(!ahead.contains("was not taken from"), "{ahead}");

        let leaked = Complaint::Leaked {
            blocks: 2,
            live_bytes: 96,
            mark_bytes: None,
        }
        .to_string();
        assert!(
            leaked.contains("2 blocks totalling 96 bytes were never freed"),
            "{leaked}"
        );
        assert!(
            leaked.contains("since: mark"),
            "the likeliest cause of this is an assertion written without a \
             mark, so it has to name that: {leaked}"
        );
        let staged_leak = Complaint::Leaked {
            blocks: 2,
            live_bytes: 96,
            mark_bytes: Some(64),
        }
        .to_string();
        assert!(
            staged_leak.contains("2 more blocks are live than at the mark"),
            "{staged_leak}"
        );

        let incomplete = Complaint::Incomplete { dropped_blocks: 7 }.to_string();
        assert!(incomplete.contains('7'), "{incomplete}");
        assert!(
            incomplete.contains("max_live_blocks"),
            "a refusal has to name the remedy: {incomplete}"
        );

        let unavailable = Complaint::Unavailable(StatsError::NotRecording).to_string();
        assert!(unavailable.contains("start one"), "{unavailable}");
    }

    /// One of anything is singular, and so is its verb. Checked for every
    /// complaint that puts a noun beside a number that can be one, because
    /// each builds its sentence separately and fixing one fixes none of the
    /// others.
    #[test]
    fn a_count_of_one_reads_as_one() {
        let sentences = [
            (
                Complaint::WrongCount {
                    counted: 1,
                    expected: 0,
                    scope: Scope::SinceMark,
                },
                "1 allocation was made since the mark, not 0",
            ),
            (
                Complaint::WrongCount {
                    counted: 1,
                    expected: 2,
                    scope: Scope::WholeRun,
                },
                "1 allocation was made, not 2",
            ),
            (
                Complaint::OverCeiling {
                    counted: 1,
                    ceiling: 0,
                    scope: Scope::WholeRun,
                },
                "1 allocation was made, above the ceiling of 0",
            ),
            (
                Complaint::MarkAhead { mark: 1, now: 0 },
                "the mark records 1 allocation but this run has made 0",
            ),
            (
                Complaint::Leaked {
                    blocks: 1,
                    live_bytes: 1,
                    mark_bytes: None,
                },
                "1 block totalling 1 byte was never freed",
            ),
            (
                Complaint::Leaked {
                    blocks: 1,
                    live_bytes: 16,
                    mark_bytes: Some(8),
                },
                "1 more block is live than at the mark",
            ),
        ];
        for (complaint, sentence) in sentences {
            let message = complaint.to_string();
            assert!(message.contains(sentence), "{message}");
        }

        // Zero is plural in English, which is the one case a `> 1` test gets
        // wrong.
        let none = Complaint::OverCeiling {
            counted: 0,
            ceiling: 0,
            scope: Scope::WholeRun,
        }
        .to_string();
        assert!(none.contains("0 allocations were made"), "{none}");
    }

    /// The note that a profile covers the whole run belongs to a measurement
    /// made since a mark, and to nothing else: not to a whole-run failure,
    /// where it would teach readers to skip it, and not to a refusal, which
    /// measured nothing. `MarkAhead` is the case that matters, because it
    /// comes only from a `since:` assertion and is about the mark rather than
    /// about anything after it.
    #[test]
    fn only_a_measurement_since_a_mark_qualifies_its_profile() {
        let since_a_mark = [
            Complaint::WrongCount {
                counted: 2,
                expected: 1,
                scope: Scope::SinceMark,
            },
            Complaint::OverCeiling {
                counted: 2,
                ceiling: 1,
                scope: Scope::SinceMark,
            },
            Complaint::Leaked {
                blocks: 1,
                live_bytes: 8,
                mark_bytes: Some(0),
            },
        ];
        for complaint in since_a_mark {
            assert_eq!(complaint.scope(), Scope::SinceMark, "{complaint:?}");
        }

        let whole_run = [
            Complaint::MarkAhead { mark: 5, now: 2 },
            Complaint::Unavailable(StatsError::Sampled),
            Complaint::Incomplete { dropped_blocks: 1 },
            Complaint::OverBudget { peak: 2, limit: 1 },
            Complaint::WrongCount {
                counted: 2,
                expected: 1,
                scope: Scope::WholeRun,
            },
            Complaint::Leaked {
                blocks: 1,
                live_bytes: 8,
                mark_bytes: None,
            },
        ];
        for complaint in whole_run {
            assert_eq!(complaint.scope(), Scope::WholeRun, "{complaint:?}");
        }
    }

    /// A failing count lists the sites that made the most allocations; every
    /// other failure lists the sites that allocated the most bytes.
    #[test]
    fn a_failing_count_ranks_its_sites_by_allocations() {
        for scope in [Scope::WholeRun, Scope::SinceMark] {
            let wrong = Complaint::WrongCount {
                counted: 2,
                expected: 1,
                scope,
            };
            let over = Complaint::OverCeiling {
                counted: 2,
                ceiling: 1,
                scope,
            };
            assert_eq!(wrong.ranking(), Ranking::Blocks);
            assert_eq!(over.ranking(), Ranking::Blocks);
        }
        for complaint in [
            Complaint::OverBudget { peak: 2, limit: 1 },
            Complaint::MarkAhead { mark: 5, now: 2 },
            Complaint::Leaked {
                blocks: 1,
                live_bytes: 8,
                mark_bytes: Some(0),
            },
        ] {
            assert_eq!(complaint.ranking(), Ranking::Bytes, "{complaint:?}");
        }
    }

    /// A ranking chosen by the complaint has to reach the summary a failure
    /// prints, or choosing it changes nothing a reader sees.
    #[test]
    #[cfg_attr(miri, ignore = "writes a profile, and Miri has no filesystem")]
    fn a_dump_ranks_its_summary_as_asked() {
        let _serial = serialized();
        let engine = distinct_figures();
        let directory = tempfile::tempdir().expect("a temporary directory");
        let guard = crate::internals::guard::enter().expect("not inside the profiler");

        let mut by_blocks = Vec::new();
        dump(
            &engine,
            &guard,
            &directory.path().join("blocks.json"),
            Ranking::Blocks,
            &mut by_blocks,
        );
        let mut by_bytes = Vec::new();
        dump(
            &engine,
            &guard,
            &directory.path().join("bytes.json"),
            Ranking::Bytes,
            &mut by_bytes,
        );
        drop(guard);

        let by_blocks = String::from_utf8(by_blocks).expect("the summary is text");
        let by_bytes = String::from_utf8(by_bytes).expect("the summary is text");
        assert!(by_blocks.contains("by blocks allocated"), "{by_blocks}");
        assert!(by_bytes.contains("by bytes allocated"), "{by_bytes}");
    }

    /// Asking a heap run for event statistics is a mistake in the test, and the
    /// message has to send its author to the other function rather than to us.
    #[test]
    fn a_mode_refusal_names_the_reading_that_would_work() {
        let heap = StatsError::NotAHeapRun(Mode::AdHoc).to_string();
        assert!(heap.contains("EventStats::get()"), "{heap}");
        assert!(heap.contains("ad-hoc"), "{heap}");

        // Copy mode counts bytes copied. Describing it as "copy events" reads
        // as a unit this crate does not have.
        let copy = StatsError::NotAHeapRun(Mode::Copy).to_string();
        assert!(copy.contains("bytes it copied"), "{copy}");
        assert!(!copy.contains("copy events"), "{copy}");

        let event = StatsError::NotAnEventRun.to_string();
        assert!(event.contains("HeapStats::get()"), "{event}");
    }
}
