//! Starting and stopping profiling.
//!
//! The profiler starts the engine, verifies that the build can actually capture
//! backtraces, and stops on drop — writing whatever outputs it was configured
//! with as it goes. Sampling and regions arrive in later milestones; the
//! lifecycle contract here is the part that has to be right first, because
//! everything else hangs off it.

use std::alloc::Layout;
use std::fmt;
use std::io;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::internals::clock::TimeSource;
use crate::internals::engine::{GlobalStats, Mode, Settings, Shutdown, State};
use crate::internals::{diagnostic, fork};
use crate::output::{FoldedMetric, Snapshot};
use crate::unwind::{self, ProbeFailure, Strategy};

/// Where a profile goes when nobody says otherwise.
///
/// The same name `dhat-rs` and Valgrind use, so that existing tooling and
/// muscle memory keep working.
pub const DEFAULT_OUTPUT_PATH: &str = "dhat-heap.json";

/// Why a profiler could not start.
///
/// `#[non_exhaustive]`: this is the only channel through which a builder can
/// refuse a configuration, and a later milestone may need one it does not have.
///
/// The refusal PLAN.md section 6.3 asks for here is **not** among them, and that
/// was settled in M5. "Sampling combined with the testing API" cannot be a
/// builder error, because a builder only ever sees the programs that *declared*
/// they intended to assert, and one that did not declare it would go on
/// asserting against estimates in silence. That refusal belongs to
/// [`StatsError`](crate::StatsError), which is returned from the reading, where
/// it needs no declaration and cannot be bypassed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StartError {
    /// A profiler is already running in this process.
    ///
    /// There is one engine per process, so a second profiler would silently
    /// blend two recordings into one profile.
    AlreadyRunning,
    /// This process has already recorded a profile.
    ///
    /// The engine records once per process and does not restart: a second run
    /// would either continue the first one's counters or silently discard them,
    /// and neither is a profile anyone asked for. Run the program again.
    AlreadyRecorded,
    /// Backtraces cannot be captured in this build.
    ///
    /// With the frame-pointer walk on x86_64 this usually means the program was
    /// compiled without `-C force-frame-pointers=yes`. It is a hard error rather
    /// than an automatic fallback on purpose: the platform unwinder costs about
    /// a hundred times as much per capture on x86_64 glibc, and a profiler that
    /// is silently that much slower is one the user concludes is broken.
    ///
    /// With [`Strategy::System`] it means the platform's own unwinder did not
    /// work, which has a different remedy or none at all. The contained
    /// [`ProbeFailure`] says which case it is and what to do about it.
    NoBacktraces(ProbeFailure),
    /// [`Alloc`](crate::Alloc) is not this program's `#[global_allocator]`.
    ///
    /// Nothing would reach the engine, so the run would record no allocations
    /// and every figure in the profile would be zero. That is refused rather
    /// than reported because zero is the one answer a reader cannot tell from a
    /// real one: an empty profile looks exactly like a program that behaved
    /// itself, and [`assert_max_bytes!`](crate::assert_max_bytes) and its
    /// siblings would pass every budget in the file. The [`stats`](crate::stats)
    /// module documents that shape of defect as the thing its refusals exist to
    /// prevent; this is the condition that reaches it before any reading does.
    ///
    /// Only [`Mode::Heap`](crate::Mode) can raise it. The other two modes count
    /// what the program reports through [`event`](fn@crate::event) and
    /// [`copied`](crate::copied) and turn the shim off, so for them an
    /// uninstalled shim is the expected configuration rather than a mistake.
    NotInstalled,
}

impl fmt::Display for StartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StartError::AlreadyRunning => {
                write!(f, "a heapscope profiler is already running in this process")
            }
            StartError::AlreadyRecorded => write!(
                f,
                "this process has already recorded a heapscope profile; \
                 the engine does not restart"
            ),
            StartError::NoBacktraces(failure) => write!(f, "{failure}"),
            StartError::NotInstalled => write!(
                f,
                "heapscope is not installed as this program's global allocator, \
                 so nothing would be recorded and every figure would be zero; \
                 add to the crate root:\n\n    \
                 #[global_allocator]\n    \
                 static ALLOC: heapscope::Alloc = heapscope::Alloc::system();"
            ),
        }
    }
}

impl std::error::Error for StartError {}

/// A running profiler. Recording stops when this is dropped, and the profile is
/// written.
///
/// # Example
///
/// Compiled but not executed, and the reason is the `drop` on the last line:
/// this example writes a profile where it stands, so running it would drop
/// `dhat-heap.json` into whatever directory the test happened to start in and
/// race every other doctest doing the same. The examples that assert rather
/// than write do run — see [`assert_max_bytes!`](crate::assert_max_bytes) — and
/// this one is exercised for real by `tests/end_to_end.rs`.
///
/// It is *not* `no_run` because of Miri, which is what this said until the
/// doctest exclusion moved into the Miri job itself. Miri cannot execute the
/// inline assembly a capture needs, and scoping that to the one job that has
/// the problem is what let every other example start running.
///
/// ```no_run
/// #[global_allocator]
/// static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let profiler = heapscope::Profiler::new()?;
/// // ... work ...
/// drop(profiler);   // writes dhat-heap.json
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Profiler {
    /// Written on drop, in order. Empty disables the automatic write.
    outputs: Vec<Output>,
    /// Not `Send`: a profiler is a scope marker, and moving one between threads
    /// makes the scope it marks ambiguous.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Profiler {
    /// Starts profiling with the default settings.
    ///
    /// The time base is [`TimeSource::Events`], which is deterministic and free,
    /// and a DHAT profile is written to [`DEFAULT_OUTPUT_PATH`] on drop. See
    /// [`Profiler::builder`] to change any of that.
    pub fn new() -> Result<Self, StartError> {
        Self::builder().build()
    }

    /// Configures a profiler before starting it.
    pub fn builder() -> ProfilerBuilder {
        ProfilerBuilder::new()
    }

    /// Whether [`Alloc`](crate::Alloc) is this program's `#[global_allocator]`.
    ///
    /// Answered by allocating, because nothing else in the process can be asked:
    /// a `#[global_allocator]` leaves no record of itself that a library can
    /// read. The shim notes that it ran, and this allocates on purpose to make
    /// it run. A *passive* read of that flag would be wrong in the one direction
    /// that matters — a program reaching here having not yet allocated would
    /// look uninstalled, and refusing to start a correctly configured program is
    /// a worse defect than the one this prevents.
    ///
    /// # `black_box`, and why the answer does not depend on it
    ///
    /// LLVM may delete an allocation whose result is unused, together with its
    /// free. `black_box` says not to. That was put here expecting it to be
    /// load-bearing, and a mutation that removes it **survives in both debug and
    /// release** — because the check is correct either way, for a reason worth
    /// stating rather than rediscovering:
    ///
    /// - Shim installed: `Alloc::alloc` is `#[inline(never)]` and stores to
    ///   `INSTALLED`, so the call has a side effect LLVM cannot discard, and the
    ///   probe cannot be elided.
    /// - Shim not installed: elided or not, nothing sets the flag, which is the
    ///   answer being asked for.
    ///
    /// So this stays as belt and braces against a future toolchain that sees
    /// further, not as the thing making it work. See PLAN.md section 9.1.
    fn shim_is_installed() -> bool {
        // Nothing depends on the size or alignment; only that the shim sees it.
        let layout = Layout::from_size_align(64, 8).expect("a valid probe layout");

        // SAFETY: `layout` has non-zero size, and the block is released below
        // with that same layout. Nothing reads the memory in between.
        let ptr = std::hint::black_box(unsafe { std::alloc::alloc(layout) });
        if ptr.is_null() {
            // The allocator failed, which answers a different question than the
            // one asked. Reporting an uninstalled shim here would name the wrong
            // cause; let the next allocation fail on its own terms.
            return true;
        }
        // SAFETY: `ptr` came from the matching `alloc` directly above, with this
        // same layout, and has not been freed.
        unsafe { std::alloc::dealloc(ptr, layout) };

        crate::alloc::installed()
    }

    fn start(builder: ProfilerBuilder) -> Result<Self, StartError> {
        let ProfilerBuilder {
            time_source,
            strategy,
            settings,
            outputs,
        } = builder;
        // Checked before the engine is claimed, so a build that cannot capture
        // backtraces leaves the engine idle and the error is repeatable. The
        // probe uses the strategy that will actually be used, which is the whole
        // point: proving frame pointers work says nothing about a run that is
        // about to ask the platform instead.
        unwind::probe(strategy).map_err(StartError::NoBacktraces)?;

        // Also before the claim, for that reason and for a second one specific
        // to this check: an allocation made while the engine is *running* gets
        // recorded, so probing after the claim would put a program point
        // belonging to this function in every profile and add one to the
        // `total_blocks` that `assert_alloc_count!` compares against. Asked
        // while the engine is idle, the probe block is invisible.
        //
        // Only for the modes that record allocations. `AdHoc` and `Copy` turn
        // the shim off, so for them an uninstalled shim is the configuration,
        // not a mistake.
        if settings.mode.records_allocations() && !Self::shim_is_installed() {
            return Err(StartError::NotInstalled);
        }

        // PLAN.md sections 5.3 and 10.2 both require the opt-in unwinder to
        // announce its cost, and neither had been implemented. It is a real
        // difference in kind, not a tuning knob: on x86_64 glibc a capture costs
        // 5,613 ns against 51 ns, so a profiled program runs visibly slower and
        // the reason should not have to be guessed. Not warned about where it is
        // the platform default, because there it is not a choice anyone made.
        if strategy == Strategy::System && Strategy::platform_default() != Strategy::System {
            diagnostic::report(
                "using the platform's stack unwinder: captures cost roughly 110x \
                 a frame-pointer walk on x86_64 glibc and 5x on aarch64 macOS, \
                 so expect the profiled program to run noticeably slower",
            );
        }

        // Before the engine starts recording, because both hooks exist to make
        // a process that forks or exits abruptly *while recording* behave, and
        // the window between starting and installing them would be exactly the
        // gap they are supposed to close.
        if !fork::install() {
            diagnostic::report(
                "could not register fork handlers; a fork from this process \
                 while profiling may wedge the child",
            );
        }
        install_exit_handler();

        // The strategy is applied *inside* the claim, while the engine is
        // `Starting` and nothing is being recorded. Doing it before the claim
        // let a failed `Profiler::new` reset a running profiler's unwinder;
        // doing it after left a window in which allocations were recorded into
        // this run while still being captured by the previous strategy.
        if !crate::engine().start(time_source, || {
            unwind::select(strategy);
            crate::engine().configure(settings);
        }) {
            // Which of the three it is decides what the user should do about it,
            // so the error distinguishes them rather than guessing.
            return Err(match crate::engine().state() {
                State::Running | State::Starting => StartError::AlreadyRunning,
                _ => StartError::AlreadyRecorded,
            });
        }

        // Recording is on from here, so the profiler's own bookkeeping has to be
        // invisible to it. Two `PathBuf`s is a small lie to tell in a profile,
        // but it is a lie told by the profiler about itself, and the first two
        // program points a user sees would be this function.
        let _quiet = crate::internals::guard::enter();

        set_exit_outputs(&outputs);
        Ok(Self {
            outputs,
            _not_send: std::marker::PhantomData,
        })
    }

    /// The global counters as they stand.
    pub fn stats(&self) -> GlobalStats {
        crate::engine().stats()
    }

    /// Reads everything recorded so far into a [`Snapshot`].
    pub fn snapshot(&self) -> Snapshot {
        Snapshot::capture()
    }

    /// Writes a DHAT format version 2 profile of everything recorded so far.
    ///
    /// Usable while profiling is still running; the result is a point-in-time
    /// reading. The allocations this call makes are kept out of the profile it
    /// is writing.
    pub fn save_dhat_v2(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.snapshot().save_dhat_v2(path)
    }

    /// Writes a native profile of everything recorded so far.
    ///
    /// See [`Snapshot::write_native`] for how it differs from the DHAT file.
    pub fn save_native(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.snapshot().save_native(path)
    }

    /// Writes a self-contained HTML page of everything recorded so far: the
    /// profile, and a viewer for it.
    ///
    /// This is the one of the three to reach for when the profile has to be
    /// readable where `_exit`, `abort`, or a Windows `std::process::exit` is
    /// about to end the process. Those paths bypass the exit handler, so what a
    /// program writes by hand beforehand is all there is — and a reader on
    /// Windows or Apple Silicon has no `dh_view.html` to open a DHAT file with.
    /// See [`Snapshot::write_html`](crate::Snapshot::write_html).
    pub fn save_html(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.snapshot().save_html(path)
    }

    /// Writes a summary of the `top` heaviest program points to stderr.
    pub fn print_summary(&self, top: usize) -> io::Result<()> {
        self.snapshot().write_text_summary(io::stderr().lock(), top)
    }
}

/// A place a profiler writes when it stops.
///
/// Opaque on purpose. A profile can go to more kinds of destination over time —
/// PLAN.md section 6.12 adds a self-contained HTML page in M7 — and every one of
/// them is a new variant. An enum callers could match on would make each of
/// those a breaking change for no benefit: nothing outside this crate can act on
/// the distinction, because nothing outside this crate writes the files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    kind: Kind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    DhatV2(PathBuf),
    Native(PathBuf),
    Html(PathBuf),
    Folded(PathBuf, FoldedMetric),
    TextSummaryToStderr(usize),
}

impl Output {
    /// A DHAT format version 2 profile at `path`.
    ///
    /// The file Valgrind's `dh_view.html` opens, and what this crate writes when
    /// nobody says otherwise.
    pub fn dhat_v2(path: impl Into<PathBuf>) -> Self {
        Self {
            kind: Kind::DhatV2(path.into()),
        }
    }

    /// A native profile at `path`: everything recorded, in the shape it was
    /// recorded in.
    ///
    /// Not the default, and that is deliberate. The DHAT file opens in a viewer
    /// that already exists on the reader's machine; this one carries everything
    /// DHAT v2 has no field for and needs a tool that reads it. Ask for both —
    /// they come from one reading of the engine, so they cannot disagree.
    ///
    /// See [`Snapshot::write_native`](crate::Snapshot::write_native) for what
    /// the difference amounts to.
    pub fn native(path: impl Into<PathBuf>) -> Self {
        Self {
            kind: Kind::Native(path.into()),
        }
    }

    /// A self-contained HTML page at `path`: the profile, and a viewer for it.
    ///
    /// One file, no build step, nothing fetched when it opens. Ask for this
    /// where the reader cannot be assumed to have Valgrind — which is every
    /// Windows machine and every Apple Silicon one, because `dh_view.html`
    /// comes from a tool that does not run on either.
    ///
    /// A complement to [`Output::dhat_v2`] rather than a replacement: the page
    /// carries the native profile verbatim, so asking for both is asking for
    /// one reading of the engine written twice, and they cannot disagree.
    ///
    /// See [`Snapshot::write_html`](crate::Snapshot::write_html) for what the
    /// page shows that DHAT v2 has no field for.
    pub fn html(path: impl Into<PathBuf>) -> Self {
        Self {
            kind: Kind::Html(path.into()),
        }
    }

    /// Folded stacks at `path`, counted by `metric`.
    ///
    /// The line-oriented format `inferno`, `flamegraph.pl`, `speedscope`, and
    /// the Firefox Profiler read. Ask for this where the reader already has a
    /// flame graph tool and would rather use it than learn a viewer.
    ///
    /// A folded file carries **one** number per stack, so `metric` decides what
    /// the picture is of. Asking for several is asking for several files, and
    /// they come from one reading of the engine, so they cannot disagree:
    ///
    /// ```no_run
    /// use heapscope::{FoldedMetric, Output, Profiler};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let profiler = Profiler::builder()
    ///     .output(Output::folded("target/allocated.folded", FoldedMetric::TotalBytes))
    ///     .also(Output::folded("target/leaked.folded", FoldedMetric::LiveBytes))
    ///     .build()?;
    /// # drop(profiler);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// See [`Snapshot::write_folded`](crate::Snapshot::write_folded) for what
    /// each metric counts, and for the one case this refuses to write:
    /// [`FoldedMetric::PeakBytes`] and [`FoldedMetric::LiveBytes`] are not
    /// measurements a run without block lifetimes took.
    pub fn folded(path: impl Into<PathBuf>, metric: FoldedMetric) -> Self {
        Self {
            kind: Kind::Folded(path.into(), metric),
        }
    }

    /// The `top` heaviest program points, printed to stderr.
    ///
    /// Stderr rather than stdout, because a profiler that writes to stdout
    /// corrupts the output of every program whose output is data.
    pub fn text_summary_to_stderr(top: usize) -> Self {
        Self {
            kind: Kind::TextSummaryToStderr(top),
        }
    }

    fn write(&self, snapshot: &Snapshot) -> io::Result<()> {
        match &self.kind {
            Kind::DhatV2(path) => snapshot.save_dhat_v2(path),
            Kind::Native(path) => snapshot.save_native(path),
            Kind::Html(path) => snapshot.save_html(path),
            Kind::Folded(path, metric) => snapshot.save_folded(path, *metric),
            Kind::TextSummaryToStderr(top) => {
                snapshot.write_text_summary(io::stderr().lock(), *top)
            }
        }
    }

    /// What to call this destination in a diagnostic.
    fn describe(&self) -> String {
        match &self.kind {
            Kind::DhatV2(path) | Kind::Native(path) | Kind::Html(path) => {
                path.display().to_string()
            }
            // The metric as well as the path, because two folded destinations
            // differ only by it and a run can legitimately have several. A
            // message naming just the file would not say which one was refused.
            Kind::Folded(path, metric) => format!("{} ({})", path.display(), metric.as_str()),
            Kind::TextSummaryToStderr(_) => "stderr".to_string(),
        }
    }
}

/// Settings for a profiler, applied when it starts.
///
/// Every setting here is fixed for the life of a run. That is not an omission:
/// a depth limit or a block ceiling that changed mid-run would make one profile
/// describe two configurations, with nothing in the file to say where the change
/// fell. What a run was configured with is written into every profile it
/// produces, read back from the engine rather than from here, so a clamped
/// request is visible as the value it became.
///
/// A request is clamped only where both ends of the range mean "as much as there
/// is" — a depth past the shim's fixed buffer, a live-block ceiling finer than
/// 64 shards can express. A request that is not a saturation but a contradiction
/// is [`build`](ProfilerBuilder::build)'s to refuse, which is why it returns a
/// [`Result`] rather than a [`Profiler`].
///
/// ```no_run
/// use heapscope::{Output, Profiler, TimeSource};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let profiler = Profiler::builder()
///     .time_source(TimeSource::Events)
///     .max_depth(24)
///     .output(Output::dhat_v2("target/dhat-heap.json"))
///     .also(Output::text_summary_to_stderr(20))
///     .build()?;
/// # drop(profiler);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct ProfilerBuilder {
    time_source: TimeSource,
    strategy: Strategy,
    /// Everything the engine holds for the life of a run, in the shape it is
    /// handed over in. Kept whole rather than spread across fields here so that
    /// adding a setting is one change rather than four, and so that the
    /// defaults have exactly one definition.
    settings: Settings,
    outputs: Vec<Output>,
}

impl Default for ProfilerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfilerBuilder {
    /// A builder holding the defaults [`Profiler::new`] uses.
    ///
    /// Everything the engine holds comes from
    /// [`Settings::default`](crate::output::Settings), not restated here. Two
    /// lists of defaults drift: with them written out twice, changing this one
    /// alone left every test green while a profiler built by hand and one built
    /// by the builder ran with different ceilings.
    pub fn new() -> Self {
        Self {
            time_source: TimeSource::Events,
            strategy: Strategy::platform_default(),
            settings: Settings::default(),
            outputs: vec![Output::dhat_v2(DEFAULT_OUTPUT_PATH)],
        }
    }

    /// The time base recorded lifetimes are measured in.
    ///
    /// [`TimeSource::Events`] is the default: deterministic, free, and what lets
    /// two runs of one workload record the same numbers rather than the same
    /// numbers plus a clock's worth of noise.
    ///
    /// A profile still names the process that produced it — its pid, its command
    /// line, where the loader mapped each image, and the profiler's own
    /// measurements of itself — so two runs are not byte-identical files.
    /// Everything the *program* did is identical, down to the order the program
    /// points are written in.
    pub fn time_source(mut self, time_source: TimeSource) -> Self {
        self.time_source = time_source;
        self
    }

    /// What the run counts.
    ///
    /// [`Mode::Heap`] is the default and is the only mode in which the allocator
    /// shim records anything. The other two count what the program reports with
    /// [`event`](fn@crate::event) and [`copied`](crate::copied), which a shim
    /// cannot see, and turn allocation recording **off** — leaving an allocation
    /// the cost of the reentrancy guard and two atomic loads.
    ///
    /// A mode is a property of the whole run because a DHAT file carries one
    /// `mode` and the viewer labels every column from it. Calls to the wrong
    /// one of the two do nothing and are counted; see [`crate::event`](fn@crate::event).
    pub fn mode(mut self, mode: Mode) -> Self {
        self.settings.mode = mode;
        self
    }

    /// How stacks are captured.
    ///
    /// [`Strategy::System`] is the escape hatch for a build that cannot supply
    /// frame pointers — a large C++ dependency someone else compiled, most
    /// likely. It is never selected for you on unix, because it costs three
    /// orders of magnitude more per capture and a profiler that is silently that
    /// much slower is one you conclude is broken. On Windows it is already the
    /// default, because there is no frame-pointer chain there to walk.
    pub fn unwinder(mut self, strategy: Strategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Frames to record per allocation, counted from the allocating code
    /// outwards.
    ///
    /// Clamped to `1..=`[`CAPTURE_DEPTH`](crate::CAPTURE_DEPTH), which is the
    /// shim's buffer. A stack cut short by this limit is reported as truncated
    /// in the profile's capture counts, exactly as one cut short by the buffer
    /// is, because it is the same thing happening for a reason the reader still
    /// needs to know. That holds under either capture strategy, which took a
    /// fix: unix's `backtrace(3)` takes no skip parameter, so a limit at or
    /// below the calibrated skip once emptied every capture and left the profile
    /// blaming a missing frame-pointer chain. See `unwind::depth_room`.
    pub fn max_depth(mut self, frames: usize) -> Self {
        self.settings.max_depth = frames;
        self
    }

    /// Ceiling on simultaneously tracked live blocks.
    ///
    /// A memory-analysis tool with unbounded memory growth is a contradiction,
    /// so the live-block table has a ceiling; blocks beyond it are counted as
    /// dropped and reported rather than silently mis-attributed.
    ///
    /// Rounded up, never down: the table is 64 shards and each rounds its share
    /// to a power of two, so a request for 5,000 becomes 8,192. What the profile
    /// reports is the ceiling that rounding produced, because that is the number
    /// its dropped-block count has to be read against.
    pub fn max_live_blocks(mut self, blocks: usize) -> Self {
        self.settings.max_live_blocks = blocks;
        self
    }

    /// Whether the outputs drop the frames that every stack has.
    ///
    /// The default is `true`: the allocation path above the program and the
    /// runtime entry below it are most of every stack and are the same on all of
    /// them. `false` keeps every frame. Either way the count that went missing
    /// is in the profile, as `trimmedFrames` — which is what a file reports
    /// about itself, rather than this setting, because a snapshot can be
    /// rendered more than one way. Also
    /// [`Snapshot::write_dhat_v2_with`](crate::Snapshot::write_dhat_v2_with)
    /// overrides this for one call.
    pub fn trim_frames(mut self, trim: bool) -> Self {
        self.settings.trim_frames = trim;
        self
    }

    /// Records a sample of allocations rather than all of them, at a mean of one
    /// sample point per `bytes` bytes allocated.
    ///
    /// What this buys is the stack capture, which is most of what profiling
    /// costs: `benches/overhead.rs` measures 129 ns per recorded allocation
    /// against 51 sampled and 32 unprofiled. What it costs is that **every figure
    /// in the profile becomes an estimate**, including the peak.
    ///
    /// # Choosing the interval
    ///
    /// The number that matters is how many sample points a run takes, which is
    /// the bytes it allocates in total divided by this. **Aim for a thousand or
    /// more.** Below a few hundred the estimates degrade quickly and program
    /// points start vanishing from the profile altogether, and there is nothing
    /// to be gained by going further: the overhead floors at about 18 ns per
    /// allocation, because entering the guard and counting the request in the
    /// size histograms happen whether or not a stack is captured.
    ///
    /// # What sampling does not lose
    ///
    /// Sample points fall on the stream of allocated *bytes*, not on the sequence
    /// of allocations, so an allocation of `s` bytes is sampled with probability
    /// `1 - exp(-s / bytes)`. A 100 MiB buffer is therefore caught with
    /// probability indistinguishable from one however large the interval is, and
    /// a sampled allocation is scaled by the reciprocal of its own probability
    /// rather than by a global factor. The big allocations that a profile exists
    /// to find are not the ones sampling drops.
    ///
    /// The size histograms stay exact, because counting a request costs no stack
    /// walk. A sampled profile therefore carries both the true number of
    /// allocations and the estimate of it, and `observedBlocks` against
    /// `totalBlocks` is how well the sampling did on that run.
    ///
    /// # What it does lose
    ///
    /// [`HeapStats::get`](crate::HeapStats::get) refuses a sampled run with
    /// [`StatsError::Sampled`](crate::StatsError::Sampled), so
    /// [`assert_max_bytes!`](crate::assert_max_bytes) and the other assertions
    /// fail rather than compare a budget against an estimate. A ceiling asserted
    /// against a sampled peak is not a weaker check than an exact one; it is a
    /// check that passes and fails for reasons unrelated to the program.
    ///
    /// Zero turns sampling off, which is the default.
    pub fn sampling(mut self, bytes: u64) -> Self {
        self.settings.sampling = NonZeroU64::new(bytes);
        self
    }

    /// Writes `output` when the profiler stops, and nothing else.
    ///
    /// Replaces the whole list, including the default DHAT file **and anything
    /// [`also`](ProfilerBuilder::also) added before it**. Use `also` to add a
    /// destination rather than choose one.
    pub fn output(mut self, output: Output) -> Self {
        self.outputs.clear();
        self.outputs.push(output);
        self
    }

    /// Adds `output` to what is written when the profiler stops.
    pub fn also(mut self, output: Output) -> Self {
        self.outputs.push(output);
        self
    }

    /// Writes nothing automatically.
    ///
    /// For programs that would rather call [`Profiler::save_dhat_v2`] themselves
    /// — or write nothing at all and read [`Profiler::snapshot`] instead. It also
    /// disarms the process-exit handler, which otherwise writes the same
    /// destinations for a program that never drops its profiler.
    pub fn no_output(mut self) -> Self {
        self.outputs.clear();
        self
    }

    /// Starts profiling.
    pub fn build(self) -> Result<Profiler, StartError> {
        Profiler::start(self)
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        let engine = crate::engine();

        // This process is a `fork` child that inherited both the profiler value
        // and the parent's counters. Writing here would hand the parent's
        // numbers to a file named after the parent's run, and race the parent
        // for it. The profile belongs to the parent.
        //
        // Load-bearing beyond the profile: returning here is also what keeps
        // this from touching `EXIT_OUTPUTS`, a `std::sync::Mutex` that the fork
        // handlers cannot reinitialise. A child that inherited it locked would
        // block here forever.
        if engine.state() == State::ForkedChild {
            return;
        }

        if engine.state() == State::Running {
            // Flips the state first, so that anything the output path allocates
            // stays out of the profile it is writing.
            engine.stop(Shutdown::Dropped);
        }

        let outputs = std::mem::take(&mut self.outputs);
        if outputs.is_empty() {
            // Asked to write nothing. Claim the write anyway, so the exit
            // handler does not decide otherwise on this profiler's behalf.
            let _ = claim_automatic_write();
            set_exit_outputs(&[]);
            return;
        }

        if !claim_automatic_write() {
            // The exit handler got there first, on another thread. It is
            // writing these same files; a second writer would interleave with
            // it.
            return;
        }
        set_exit_outputs(&[]);
        write_outputs(&outputs);
    }
}

/// Writes a profile, reporting a failure rather than propagating it.
///
/// Shared by `Profiler::drop` and the process-exit handler, both of which run
/// where a panic is fatal: a panic during unwinding aborts the process, and one
/// inside an `atexit` handler unwinds out of C, which is undefined. The report
/// goes out through `writeln!` rather than `eprintln!` for the same reason —
/// `eprintln!` panics if stderr cannot be written, which would be a panic in
/// the very handler written to avoid panicking.
fn write_outputs(outputs: &[Output]) {
    if outputs.is_empty() {
        return;
    }
    // One reading, however many destinations. Capturing per output would let a
    // DHAT file and the summary printed beside it disagree about the same run,
    // and the disagreement would be the profiler's own writing showing up in
    // the second reading.
    let snapshot = Snapshot::capture();
    for output in outputs {
        if let Err(error) = output.write(&snapshot) {
            use std::io::Write;
            // The destination is whatever the caller built the `Output` from,
            // so it is the same class of string as an image path or a symbol
            // name: not this crate's, and on its way to a terminal.
            // `push_display` is what stands between the two everywhere else in
            // the output layer; a diagnostic is no less a place for a terminal
            // to be told to do something.
            let mut screened = String::new();
            crate::output::push_display(&mut screened, &output.describe());
            let _ = writeln!(
                io::stderr(),
                "heapscope: could not write the profile to {screened}: {error}"
            );
        }
    }
}

/// What the process-exit handler writes, or empty if it should not write.
///
/// A global rather than something reachable from the [`Profiler`], because an
/// `atexit` handler is a bare `extern "C" fn()` with nowhere to put a context
/// pointer. The mutex is only ever taken by profiler setup and by the handler
/// itself, never from the allocation path.
static EXIT_OUTPUTS: Mutex<Vec<Output>> = Mutex::new(Vec::new());

/// Whether [`install_exit_handler`] has already run.
static EXIT_HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

/// Claimed by whichever automatic path writes the profile first.
///
/// The two automatic writers both used to check `state == Running` and then
/// stop the engine, which is a check and a store that are not atomic with
/// respect to each other. One thread dropping the `Profiler` while another
/// called `std::process::exit` could get through both checks, and both would
/// write the same destination — interleaved, unparseable, and with a `shutdown`
/// field naming whichever writer lost. One claim decides it.
///
/// This governs only the automatic write. [`Profiler::save_dhat_v2`] is the
/// caller's to invoke as often as they like.
static AUTOMATIC_WRITE_CLAIMED: AtomicBool = AtomicBool::new(false);

/// Whether this call owns the automatic write of the profile.
fn claim_automatic_write() -> bool {
    !AUTOMATIC_WRITE_CLAIMED.swap(true, Ordering::AcqRel)
}

/// Takes a borrowed slice so that the copy it stores is made *here*, under the
/// reentrancy guard, rather than by the caller where it would be recorded.
fn set_exit_outputs(outputs: &[Output]) {
    let _quiet = crate::internals::guard::enter();
    // A poisoned mutex means a previous holder panicked while holding it. The
    // value behind it is a list that is either fully written or not written at
    // all, so there is no torn state to protect anyone from, and refusing to
    // write a profile because of it would be the wrong trade.
    let mut current = EXIT_OUTPUTS
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    *current = outputs.to_vec();
}

/// Registers the process-exit handler, once per process.
///
/// # What this buys, and what it does not
///
/// It covers the programs that never drop their profiler: anything ending in
/// `std::process::exit`, including from a thread that is not `main`, and any
/// program that keeps the profiler alive in a `static`. Without it those
/// produce no profile at all, which is the single most common way a heap
/// profiler disappoints.
///
/// It does **not** cover `_exit`, `abort`, or a fatal signal. Those bypass the
/// `atexit` list by design, and no handler registered through it can see them.
/// PLAN.md section 4.6 records that as a stated limitation rather than
/// something to be discovered. The remedy is for the program to write what it
/// wants before it goes — [`Profiler::save_dhat_v2`], [`Profiler::save_native`]
/// or [`Profiler::save_html`] — and `tests/lifecycle.rs` runs that remedy in
/// front of a real `_exit` and a real `abort` rather than leaving it as advice.
///
/// # One platform difference, and it is not a small one
///
/// **On Windows, `std::process::exit` does not reach this handler.** It calls
/// `ExitProcess` directly (`library/std/src/sys/exit.rs`), which terminates the
/// process without going through the CRT's `exit`, so the `atexit` list is
/// never walked. Windows offers no documented hook for it: `DLL_PROCESS_DETACH`
/// would catch it, but only for code loaded as a DLL, not for an executable.
///
/// Returning from `main` is unaffected on every platform, because there the C
/// runtime calls `exit` itself. So on Windows a profiler kept in a `static`
/// still produces a profile; a program that ends in `std::process::exit` does
/// not, and must drop its profiler or call one of the `save_*` methods before
/// exiting. [`Profiler::save_html`] is usually the one to want here, because a
/// reader on Windows has no `dh_view.html` to open the DHAT file with.
fn install_exit_handler() {
    if EXIT_HANDLER_INSTALLED.swap(true, Ordering::AcqRel) {
        return;
    }

    extern "C" {
        /// C89. Present in libc on unix and in the Universal CRT on Windows,
        /// both of which the standard library already links. Returns 0 on
        /// success; a non-zero return means the list is full.
        fn atexit(handler: extern "C" fn()) -> std::ffi::c_int;
    }

    // SAFETY: `on_process_exit` is an `extern "C"` function of the required
    // signature with `'static` lifetime. Registration has no preconditions.
    if unsafe { atexit(on_process_exit) } != 0 {
        diagnostic::report(
            "could not register the exit handler; a profile will only be \
             written if the profiler is dropped",
        );
    }
}

/// Writes the profile for a process that is exiting without dropping its
/// profiler.
extern "C" fn on_process_exit() {
    let engine = crate::engine();

    // Anything other than `Running` means this profile has already been dealt
    // with — dropped, stopped by hand, or disowned by a `fork`. In particular
    // the ordinary path through `Profiler::drop` lands here a moment later, and
    // must not overwrite what it just wrote.
    //
    // The `ForkedChild` case is also what keeps a child from touching
    // `EXIT_OUTPUTS`, which the fork handlers cannot reinitialise.
    if engine.state() != State::Running {
        return;
    }
    engine.stop(Shutdown::AtExit);

    // The state check above is a read, and `Profiler::drop` on another thread
    // may have passed the same read a moment ago. Only one of the two writes.
    if !claim_automatic_write() {
        return;
    }

    let outputs = {
        let outputs = EXIT_OUTPUTS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        outputs.clone()
    };
    if outputs.is_empty() {
        return;
    }

    // Unwinding out of an `atexit` handler crosses a C frame, which is
    // undefined behaviour. Everything below is written not to panic; this is
    // the backstop for the case where something does anyway. Under
    // `panic = "abort"` it compiles to a direct call.
    let _ = std::panic::catch_unwind(|| write_outputs(&outputs));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_error_for_a_missing_frame_pointer_names_the_remedy() {
        let error = StartError::NoBacktraces(ProbeFailure::ChainTooShort {
            found: 0,
            expected: 3,
        });
        let message = error.to_string();
        assert!(message.contains("force-frame-pointers=yes"), "{message}");
    }

    /// What a builder holds before anything starts. The engine is a process-wide
    /// singleton, so these are the only builder assertions a unit test can make;
    /// what the settings *do* is asserted end to end in `tests/lifecycle.rs`.
    #[test]
    fn a_builder_writes_one_dhat_profile_unless_told_otherwise() {
        let default = ProfilerBuilder::new();
        assert_eq!(default.outputs.len(), 1);
        assert_eq!(default.outputs[0].describe(), DEFAULT_OUTPUT_PATH);
        assert_eq!(
            default.settings,
            Settings::default(),
            "a builder nobody configured must hand the engine exactly the \
             defaults an unconfigured engine already holds"
        );
    }

    /// `output` chooses and `also` adds. The distinction is the whole reason
    /// there are two: a builder where naming a destination silently kept the
    /// default one would drop a `dhat-heap.json` in the working directory of
    /// every program that asked for a file somewhere else.
    #[test]
    fn choosing_an_output_replaces_the_default_and_adding_one_does_not() {
        let chosen = ProfilerBuilder::new().output(Output::dhat_v2("/tmp/one.json"));
        assert_eq!(
            chosen
                .outputs
                .iter()
                .map(Output::describe)
                .collect::<Vec<_>>(),
            ["/tmp/one.json"]
        );

        let added = chosen.also(Output::text_summary_to_stderr(5));
        assert_eq!(
            added
                .outputs
                .iter()
                .map(Output::describe)
                .collect::<Vec<_>>(),
            ["/tmp/one.json", "stderr"]
        );

        // And choosing again replaces everything, including what `also` added.
        let rechosen = added.output(Output::dhat_v2("/tmp/two.json"));
        assert_eq!(
            rechosen
                .outputs
                .iter()
                .map(Output::describe)
                .collect::<Vec<_>>(),
            ["/tmp/two.json"]
        );

        assert!(rechosen.no_output().outputs.is_empty());
    }

    #[test]
    fn already_running_reads_clearly() {
        assert!(StartError::AlreadyRunning
            .to_string()
            .contains("already running"));
    }
}
