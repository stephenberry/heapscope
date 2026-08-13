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
//! # The condition this table did not list
//!
//! A program whose `#[global_allocator]` is not [`Alloc`](crate::Alloc) records
//! nothing, so every figure is zero — and a zero peak passes every budget in the
//! file. That is the cannot-fail shape above, reached by a route none of the
//! rows covers, and for a while it was reachable: `assert_max_bytes!(64 * 1024)`
//! passed in a program that had just allocated 10 MiB.
//!
//! It is not a sixth row, because a reading is the wrong place to catch it. By
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
//! written against a mark rather than against zero: read [`HeapStats::get`]
//! immediately before the code under test, and assert `mark.total_blocks + n`.
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
use crate::output::{count, Snapshot};

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
    WrongCount { counted: u64, expected: u64 },
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
            Complaint::WrongCount { counted, expected } => write!(
                f,
                "{} allocations were made, not {}",
                count(*counted),
                count(*expected)
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
                "{} blocks totalling {} bytes were never freed; if the program \
                 holds memory of its own, take a mark with HeapStats::get() \
                 first and assert `since: mark`",
                count(*blocks),
                count(*live_bytes)
            ),
            Complaint::Leaked {
                blocks,
                live_bytes,
                mark_bytes: Some(mark),
            } => write!(
                f,
                "{} more blocks are live than at the mark, where live bytes \
                 went from {} to {}",
                count(*blocks),
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

pub(crate) fn check_alloc_count(engine: &Engine, expected: u64) -> Result<(), Complaint> {
    let stats = assertable(engine)?;
    if stats.total_blocks != expected {
        return Err(Complaint::WrongCount {
            counted: stats.total_blocks,
            expected,
        });
    }
    Ok(())
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
    summary: &mut dyn io::Write,
) -> String {
    // One reading, both destinations, for the reason `write_outputs` takes one:
    // a summary and a file that disagree about the same failure are two
    // readings of a program that kept running in between.
    let snapshot = Snapshot::of(engine);
    let _ = snapshot.write_text_summary(summary, TOP_ON_FAILURE);

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
#[track_caller]
pub(crate) fn report<E: fmt::Display>(outcome: Result<(), E>, context: Option<fmt::Arguments<'_>>) {
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
            Some(dump(engine, entered, &path, &mut stderr))
        }
        _ => None,
    };
    if let Some(line) = dumped {
        message.push_str("\n  ");
        message.push_str(&line);
    }
    panic!("{message}");
}

/// A macro argument as the `u64` the engine keeps its counters in.
///
/// Generic rather than a plain `u64` parameter because **every size and count
/// in Rust is a `usize`**: `assert_alloc_count!(items.len())` and a budget held
/// in a `usize` are the ordinary call sites, and both are a type error against
/// a `u64`. An integer literal still infers, because the fallback type
/// satisfies the bound. There is no `From<usize> for u64`, so this is the
/// conversion that exists.
///
/// A value that does not fit — a negative one — panics rather than saturating.
/// A budget of `-1` silently becoming `u64::MAX` is an assertion that cannot
/// fail, which is the one outcome this module exists to prevent.
#[track_caller]
fn as_count<N: TryInto<u64>>(value: N) -> u64 {
    value.try_into().unwrap_or_else(|_| {
        panic!(
            "heapscope: a negative or oversized number is not a byte count or an allocation count"
        )
    })
}

/// The body of [`assert_max_bytes!`](crate::assert_max_bytes). Not a supported
/// entry point.
#[doc(hidden)]
#[track_caller]
pub fn __assert_max_bytes<N: TryInto<u64>>(limit: N, context: Option<fmt::Arguments<'_>>) {
    report(check_max_bytes(crate::engine(), as_count(limit)), context);
}

/// The body of [`assert_alloc_count!`](crate::assert_alloc_count). Not a
/// supported entry point.
#[doc(hidden)]
#[track_caller]
pub fn __assert_alloc_count<N: TryInto<u64>>(expected: N, context: Option<fmt::Arguments<'_>>) {
    report(
        check_alloc_count(crate::engine(), as_count(expected)),
        context,
    );
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
/// Takes any integer that fits a `u64`, so a budget held in a `usize` works
/// without a cast. A trailing message is formatted as [`format_args!`] and
/// printed with the failure, which is worth using when the same assertion runs
/// over several fixtures.
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
/// # Panics
///
/// When the peak exceeded `limit`, and when there are no numbers to check — see
/// the [module documentation](crate::stats) for that list. It does **not** pass
/// quietly in either case.
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

/// Fails unless the run made exactly `expected` allocations.
///
/// An equality rather than a ceiling, because the name says count and because
/// the other reading has a failure mode this crate refuses elsewhere: a budget
/// spelled `assert_alloc_count!(3)` would pass a run that allocated nothing,
/// which is precisely how a broken test goes green. Write the ceiling out where
/// that is what you mean:
///
/// ```
/// # #[global_allocator]
/// # static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
/// # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// assert!(heapscope::HeapStats::get().unwrap().total_blocks <= 3);
/// ```
///
/// A reallocation counts as an allocation, so a `Vec` that grows four times made
/// five. See [`HeapStats::total_blocks`].
///
/// Takes any integer that fits a `u64`, so `items.len()` works without a cast.
/// A trailing message is formatted as [`format_args!`] and printed with the
/// failure:
///
/// ```
/// # #[global_allocator]
/// # static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
/// # let fixture = "big.json";
/// # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// let rows = [Box::new(1u8), Box::new(2u8), Box::new(3u8)];   // three allocations
/// # std::hint::black_box(&rows);
/// heapscope::assert_alloc_count!(3, "while parsing {fixture}");
/// ```
///
/// # Panics
///
/// When the count differs, and when there are no numbers to check.
#[macro_export]
macro_rules! assert_alloc_count {
    ($expected:expr $(,)?) => {
        $crate::__assert_alloc_count($expected, ::core::option::Option::None)
    };
    ($expected:expr, $($arg:tt)+) => {
        $crate::__assert_alloc_count(
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

        assert_eq!(check_alloc_count(&engine, 4), Ok(()));
        assert_eq!(
            check_alloc_count(&engine, 5),
            Err(Complaint::WrongCount {
                counted: 4,
                expected: 5
            })
        );
        assert_eq!(
            check_alloc_count(&engine, 3),
            Err(Complaint::WrongCount {
                counted: 4,
                expected: 3
            })
        );
    }

    /// Allocations ever made, not blocks still live. On this run those are 4
    /// and 2, so a check reading the live figure would pass `2` — which is the
    /// "passes a run that allocated nothing" failure the equality exists to
    /// prevent, one column over.
    #[test]
    fn the_count_is_of_allocations_rather_than_of_live_blocks() {
        let _serial = serialized();
        let engine = distinct_figures();
        let stats = HeapStats::of(&engine).unwrap();
        assert!(stats.total_blocks > stats.curr_blocks);

        assert_eq!(check_alloc_count(&engine, stats.total_blocks), Ok(()));
        assert!(check_alloc_count(&engine, stats.curr_blocks).is_err());
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
        assert!(message.contains("1 more blocks are live"), "{message}");
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
        assert_eq!(check_alloc_count(&engine, 0), incomplete);
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
            check_alloc_count(&heap, 0),
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
        let line = dump(&engine, &guard, &path, &mut summary);
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

        let line = dump(&engine, &guard, &path, &mut Vec::new());
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

        let line = dump(&engine, &guard, &path, &mut Vec::new());
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

    #[test]
    fn a_limit_can_be_any_of_the_integer_types_a_call_site_has() {
        assert_eq!(as_count(64usize), 64);
        assert_eq!(as_count(64u32), 64);
        assert_eq!(as_count(64u64), 64);
        assert_eq!(as_count(64i32), 64);
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
        }
        .to_string();
        assert!(wrong.contains("5 allocations were made, not 3"), "{wrong}");

        let leaked = Complaint::Leaked {
            blocks: 2,
            live_bytes: 96,
            mark_bytes: None,
        }
        .to_string();
        assert!(leaked.contains("2 blocks"), "{leaked}");
        assert!(leaked.contains("96 bytes"), "{leaked}");
        assert!(
            leaked.contains("since: mark"),
            "the likeliest cause of this is an assertion written without a \
             mark, so it has to name that: {leaked}"
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
