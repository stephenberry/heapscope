//! Stack unwinding: the performance crux of the whole profiler.
//!
//! Capturing a backtrace is by far the most expensive thing the allocator shim
//! does, and the available strategies differ by nearly three orders of
//! magnitude. Measured on aarch64-apple-darwin with `benches/unwind.rs`, against
//! a baseline `malloc`/`free` of a 64-byte block at **16.7 ns**:
//!
//! | Strategy | Cost |
//! |---|---|
//! | Frame-pointer walk, 12 frames | **~21 ns** |
//! | `std::backtrace::Backtrace::force_capture` | **~18,800 ns** |
//!
//! PLAN.md section 5.1 records the same shape from an independent probe, adding
//! libc `backtrace()` at 157 ns and `_Unwind_Backtrace` at 8,335 ns for a
//! 32-frame capture.
//!
//! A frame-pointer capture therefore costs about the same as the allocation it
//! is recording, while the standard library's unwinder costs roughly **900
//! times** as much. That is not "slower"; it is a different tool. So on unix the
//! frame-pointer walk is the only strategy ever selected automatically, and a
//! build that cannot support it is a **configuration error reported at
//! startup**, not a reason to fall back to something three orders of magnitude
//! slower. The governing rule is *never silently slow*: the failure mode to
//! avoid is a user profiling for ten minutes and concluding the tool is broken.
//!
//! # Windows is the exception, and it is not a fallback
//!
//! PLAN.md section 10.2 resolved "require frame pointers on x86_64, with the
//! system unwinder as an explicit opt-in that is never selected automatically".
//! That is right on Linux and macOS and impossible on Windows, where there is no
//! frame-pointer chain to require: the Microsoft x64 ABI mandates unwind tables
//! for every function, so it never needed one, and `-C force-frame-pointers=yes`
//! does not produce one that can be walked. Measured — see [`system`].
//!
//! So [`Strategy::System`] is the **default on Windows**. Applying the rule as
//! written would have meant refusing to start there at all: the startup probe
//! fires correctly and the tool is unusable. And the reasoning behind the rule
//! does not carry across, because `RtlCaptureStackBackTrace` is not a slow
//! fallback to a fast path that exists — it is the platform's own supported
//! mechanism, and unlike the unix system unwinder it needs no build flags from
//! the user at all.

pub mod frame_pointer;
pub mod system;

use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};

pub use frame_pointer::{Capture, Outcome};

// There is deliberately no depth constant here. Both the buffer and the default
// are [`crate::CAPTURE_DEPTH`], declared next to the shim that owns the buffer,
// and the two that used to live here — a 256 that bounded nothing and a 24 that
// was nobody's default — were read as though they did. A second name for a
// number this file does not decide is a way to be wrong about it.

/// How backtraces are captured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Strategy {
    /// Walk the frame-pointer chain. Around twenty nanoseconds, and the default
    /// everywhere it works.
    FramePointer = 0,
    /// Ask the platform. `RtlCaptureStackBackTrace` on Windows, libc
    /// `backtrace` elsewhere. See [`system`] for what this costs and when it is
    /// the right answer.
    System = 1,
}

impl Strategy {
    /// The strategy used when nobody chooses one.
    ///
    /// **x86_64** Windows has no walkable frame-pointer chain, so there is
    /// nothing to choose there; see the module documentation.
    ///
    /// The architecture matters and the first version of this omitted it. Every
    /// justification for the Windows default — here, in [`system`], in the
    /// README and in PLAN.md section 10.2 — says "the Microsoft **x64** ABI",
    /// and the measurement behind it is `x86_64-pc-windows-gnu`. ARM64 Windows
    /// maintains the `x29` frame chain for non-leaf frames, which PLAN.md
    /// section 5.3 already measured on the two aarch64 targets, so a bare
    /// `cfg!(windows)` threw away a working ~50 ns walk for a table walk on a
    /// platform the argument never covered. Neither Wine on x86_64 nor CI's
    /// x86_64 `windows-latest` runner could have shown it.
    pub const fn platform_default() -> Strategy {
        if cfg!(all(windows, target_arch = "x86_64")) {
            Strategy::System
        } else {
            Strategy::FramePointer
        }
    }

    fn from_u8(raw: u8) -> Strategy {
        match raw {
            1 => Strategy::System,
            _ => Strategy::FramePointer,
        }
    }

    /// The name this appears under in a profile.
    pub fn as_str(self) -> &'static str {
        match self {
            Strategy::FramePointer => "frame-pointer",
            Strategy::System => "system",
        }
    }
}

impl Default for Strategy {
    fn default() -> Self {
        Strategy::platform_default()
    }
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The strategy every capture uses.
///
/// Written once, at profiler startup, and read on every allocation. A relaxed
/// load of a value that never changes after startup sits in every core's cache
/// in shared state and generates no coherence traffic — the same argument the
/// engine's `serialized` flag makes.
static SELECTED: AtomicU8 = AtomicU8::new(Strategy::platform_default() as u8);

/// The strategy captures are currently using.
#[inline(always)]
pub fn strategy() -> Strategy {
    Strategy::from_u8(SELECTED.load(Ordering::Relaxed))
}

/// Chooses the capture strategy. Called from profiler startup, before recording
/// begins.
pub fn select(strategy: Strategy) {
    SELECTED.store(strategy as u8, Ordering::Relaxed);
    INTERNAL_FRAMES.store(calibrate(strategy), Ordering::Relaxed);
}

/// How many innermost frames of a capture belong to the capture machinery.
///
/// Measured rather than assumed, because the answer differs by strategy and by
/// optimisation level and there is no constant that is right for both. The
/// frame-pointer walk begins at the frame `frame_pointer()` was inlined into;
/// the platform unwinder begins at the caller of the platform's own function,
/// several `heapscope` frames further in. A single constant covering both is
/// wrong for at least one of them, and it was: with `Strategy::System` selected,
/// every program point in every profile began with two or three `heapscope`
/// frames — measured at 3 in a debug build and 2 in a release build — which is
/// precisely what the constant existed to prevent.
static INTERNAL_FRAMES: AtomicUsize = AtomicUsize::new(0);

/// Frames the capture machinery contributes, for the current strategy.
#[inline(always)]
pub fn internal_frames() -> usize {
    INTERNAL_FRAMES.load(Ordering::Relaxed)
}

/// Buffer slots a capture of `wanted` frames needs under `strategy`, given that
/// `skip` of them are to be discarded.
///
/// Not simply `wanted`, and the difference is the whole reason this exists.
/// Unix's `backtrace(3)` takes no skip parameter, so the skipped frames are
/// written into the caller's buffer and shifted away afterwards: a buffer of
/// `wanted` slots yields `wanted - skip` usable frames there, and none at all
/// once `wanted <= skip`. The frame-pointer walk and Windows'
/// `RtlCaptureStackBackTrace` both skip before they write, and need exactly
/// `wanted`.
///
/// The failure this prevents is quiet and total. A depth limit at or below the
/// calibrated skip made every capture come back empty, so a profile had one
/// program point reading `[unwalkable]: no frame pointer chain at this
/// allocation` — a sentence that was false, since frame pointers were fine and
/// the user's own setting had done it **\[measured\]**. The skip is calibrated at
/// startup and differs by optimisation level, so the same limit worked in one
/// build and emptied the profile in another.
pub fn depth_room(strategy: Strategy, wanted: usize, skip: usize) -> usize {
    match strategy {
        Strategy::FramePointer => wanted,
        // Windows skips natively; every unix `backtrace(3)` does not.
        Strategy::System if cfg!(windows) => wanted,
        Strategy::System => wanted.saturating_add(skip),
    }
}

/// Counts the frames a capture puts *below* the caller of the capturing
/// function.
///
/// The shim wants its recorded frames to begin at the code that allocated. The
/// capturing function there is `<Alloc as GlobalAlloc>::alloc`, so the target is
/// the frame of *its* caller — and the number of frames below that is what the
/// shim must skip.
///
/// # Why this compares two captures instead of looking for an address
///
/// The obvious method is to note the address of a function standing in for the
/// caller and find it in the capture. There is no portable way to ask how long
/// a function is, so "find it" has to mean "within some window of its entry
/// point", and that window is a guess about code layout rather than a fact
/// about it.
///
/// The guess was 8 KB, and it broke: adding an unrelated module to this crate
/// moved another function into the window, `calibrate` matched that frame
/// instead, and the platform unwinder went back to putting two `heapscope`
/// frames at the start of every program point — silently, at MSRV only, with
/// the whole suite green apart from the one test that checks the result.
///
/// So this compares two captures taken from the same place at different depths.
/// [`calibration_shallow`] reaches [`calibration_site`] directly, and
/// [`calibration_deep`] reaches it through [`calibration_shallow`]. Every frame
/// from the capture machinery up to and including `calibration_shallow`'s own
/// is the same code returning to the same instruction in both, so the two
/// captures agree byte for byte up to that point and differ at the frame just
/// above it:
///
/// ```text
///   shallow:  [ machinery.. , site , shallow , calibrate       , .. ]
///   deep:     [ machinery.. , site , shallow , calibration_deep, .. ]
///                                              ^ first difference
/// ```
///
/// `calibration_shallow` stands in for the code that allocated, so the answer
/// is the index *before* the first difference. No address windows, no
/// assumption about how the linker laid anything out.
fn calibrate(strategy: Strategy) -> usize {
    let mut shallow = [0usize; 32];
    let mut deep = [0usize; 32];
    let shallow_len = calibration_shallow(strategy, &mut shallow);
    let deep_len = calibration_deep(strategy, &mut deep);

    // One extra call has to produce exactly one extra frame, or these are not
    // the pair this reasoning is about — a call was inlined away despite
    // `#[inline(never)]`.
    //
    // Except when the walk ran out of buffer, where it cannot: both captures
    // stop at the same capacity and the extra frame pushes one off the far end
    // rather than lengthening the list. Requiring the arithmetic there refused
    // to calibrate on any stack deeper than these arrays, which is not an exotic
    // condition — it is `Profiler::build()` called from far enough inside a
    // program, and a TSan build reaches it in the crate's own unit tests
    // **\[measured\]**, at 32 frames against a 32-frame buffer.
    //
    // Reading the comparison there is still sound, and for the reason the
    // diagram above makes visible: a truncated walk loses its *outermost*
    // frames, while everything this looks at — the agreeing prefix and the
    // first difference — sits at the innermost end, which is captured first and
    // is what survives. What the length check also bought, catching a
    // `calibration_deep` whose call was inlined away, is not lost with it: an
    // inlined call makes the two captures identical, so the search below finds
    // no difference and this still declines.
    let truncated = shallow_len == shallow.len() && deep_len == deep.len();
    let usable = shallow_len.min(deep_len);
    let found = (truncated || deep_len == shallow_len + 1)
        .then(|| (0..usable).find(|&index| shallow[index] != deep[index]))
        .flatten()
        // A difference at index 0 would mean the machinery itself differed
        // between two identical calls, which is not a stack this can reason
        // about.
        .and_then(|index| index.checked_sub(1));

    match found {
        Some(index) => index,
        None => {
            // No number to use. Zero is the safe direction: it keeps
            // recognisable `heapscope` frames rather than silently discarding
            // real ones, and the diagnostic says the profile will show them.
            super::internals::diagnostic::report(
                "could not calibrate how many frames the capture machinery \
                 contributes; program points may begin with heapscope's own frames",
            );
            0
        }
    }
}

/// Stands in for the code that allocated: calls the capturing function
/// directly.
#[inline(never)]
fn calibration_shallow(strategy: Strategy, out: &mut [usize]) -> usize {
    std::hint::black_box(calibration_site(strategy, out))
}

/// The same, one frame further away, which is the only difference between the
/// two captures [`calibrate`] compares.
#[inline(never)]
fn calibration_deep(strategy: Strategy, out: &mut [usize]) -> usize {
    std::hint::black_box(calibration_shallow(strategy, out))
}

/// Stands in for the shim method that captures, which is `#[inline(never)]` for
/// exactly this reason: the frame layout above the machinery has to be the same
/// at every optimisation level.
#[inline(never)]
fn calibration_site(strategy: Strategy, out: &mut [usize]) -> usize {
    let capture = capture_with(strategy, crate::internals::stack::current_bounds(), 0, out);
    std::hint::black_box(capture.len)
}

/// Why the frame-pointer capability probe failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeFailure {
    /// The architecture has no frame-pointer convention this crate knows.
    UnsupportedArchitecture,
    /// A frame pointer was read but the walk produced fewer frames than the
    /// probe's own known call depth, which means frame pointers are omitted.
    ChainTooShort {
        /// Frames the probe recovered.
        found: usize,
        /// Frames the probe knows it called through.
        expected: usize,
    },
    /// The walk did not reach the probe's own return addresses.
    ChainInvalid,
    /// The platform's own unwinder returned nothing from a known call stack.
    SystemUnwinderEmpty,
    /// The platform's own unwinder returned frames that were not the probe's.
    SystemUnwinderWrong {
        /// Frames it returned.
        found: usize,
    },
}

impl fmt::Display for ProbeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProbeFailure::UnsupportedArchitecture => write!(
                f,
                "heapscope does not know the frame-pointer convention for {}",
                std::env::consts::ARCH
            ),
            ProbeFailure::ChainTooShort { found, expected } => write!(
                f,
                "frame pointers are not available: a known {expected}-deep call \
                 stack yielded only {found} frames.\n\
                 Rebuild with:  RUSTFLAGS=\"-C force-frame-pointers=yes\"\n\
                 For C/C++ dependencies built via `cc`, also set:\n\
                 \x20              CFLAGS=\"-fno-omit-frame-pointer\""
            ),
            ProbeFailure::ChainInvalid => write!(
                f,
                "the frame-pointer chain did not contain the probe's own frames.\n\
                 Rebuild with:  RUSTFLAGS=\"-C force-frame-pointers=yes\""
            ),
            ProbeFailure::SystemUnwinderEmpty => write!(
                f,
                "the platform's stack unwinder returned no frames at all. Every \n\
                 allocation would be attributed to one program point with no \n\
                 stack, so profiling is refused rather than producing that.\n\
                 {}",
                // PLAN.md section 5.2, from this project's own measurement:
                // `_Unwind_Backtrace` under `-C panic=abort` returns *success*
                // having captured nothing, and glibc's `backtrace` is built on
                // it. That is the failure a Linux user of this escape hatch
                // actually hits, and it has a remedy.
                if cfg!(all(unix, not(target_vendor = "apple"))) {
                    "This is what a build without unwind tables looks like.\n\
                     Rebuild with:  RUSTFLAGS=\"-C force-unwind-tables=yes\"\n\
                     A `panic = \"abort\"` profile omits them by default."
                } else {
                    "The platform unwinder takes no configuration here, so there \n\
                     is nothing to rebuild with."
                }
            ),
            ProbeFailure::SystemUnwinderWrong { found } => write!(
                f,
                "the platform's stack unwinder returned {found} frames, none of \n\
                 which were the probe's own, so its output cannot be trusted."
            ),
        }
    }
}

/// Running counts of capture quality, reported in a profile's self-metrics.
///
/// PLAN.md section 5.4: the startup probe walks *our* frames, which under
/// uniform `RUSTFLAGS` says nothing about `cc`-built C/C++ dependencies,
/// hand-written assembly, JIT frames, or threads created by a C library. These
/// counters turn that false confidence into a number the user can read.
#[derive(Debug, Default)]
pub struct Counters {
    complete: AtomicU64,
    truncated: AtomicU64,
    suspect: AtomicU64,
    no_frames: AtomicU64,
}

/// A snapshot of [`Counters`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CounterSnapshot {
    /// Captures that reached the outermost frame.
    pub complete: u64,
    /// Captures cut off by the depth limit.
    pub truncated: u64,
    /// Captures stopped by a failed validation check.
    pub suspect: u64,
    /// Captures that found no frame pointer at all.
    pub no_frames: u64,
}

impl Counters {
    /// Creates zeroed counters.
    pub const fn new() -> Self {
        Self {
            complete: AtomicU64::new(0),
            truncated: AtomicU64::new(0),
            suspect: AtomicU64::new(0),
            no_frames: AtomicU64::new(0),
        }
    }

    /// Records one capture outcome.
    #[inline]
    pub fn record(&self, outcome: Outcome) {
        let counter = match outcome {
            Outcome::Complete => &self.complete,
            Outcome::TruncatedByDepth => &self.truncated,
            Outcome::Suspect => &self.suspect,
            Outcome::NoFrames => &self.no_frames,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Reads the current counts.
    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            complete: self.complete.load(Ordering::Relaxed),
            truncated: self.truncated.load(Ordering::Relaxed),
            suspect: self.suspect.load(Ordering::Relaxed),
            no_frames: self.no_frames.load(Ordering::Relaxed),
        }
    }
}

/// Running capture-quality counts for this process.
///
/// These were built in M1 and wired to nothing until M3: `alloc::capture`
/// discarded the outcome, `Counters::record` had no callers, and four separate
/// comments — including PLAN.md section 5.4 — described a profile field that did
/// not exist. A counter nobody increments is worse than no counter, because the
/// documentation around it reads as a guarantee.
static COUNTERS: Counters = Counters::new();

/// The process-wide capture-quality counters.
#[inline(always)]
pub fn counters() -> &'static Counters {
    &COUNTERS
}

/// Verifies that `strategy` actually captures backtraces in this build.
///
/// Not "verifies that frame pointers are available", which is what this said
/// before it took a strategy: on Windows the default strategy does not use them
/// and proving they work would establish nothing about the run about to start.
///
/// The probe calls through a chain of functions that cannot be inlined or
/// tail-called away, then checks that the capture recovers at least that many
/// frames *and* that they are the probe's own return addresses. Counting frames
/// alone is not enough: a walk that wanders into unrelated stack memory can
/// produce plenty of plausible-looking addresses.
///
/// Called once at profiler startup. A failure is fatal to starting the
/// profiler and names the exact remedy.
pub fn probe(strategy: Strategy) -> Result<(), ProbeFailure> {
    if strategy == Strategy::FramePointer && frame_pointer::frame_pointer().is_none() {
        return Err(ProbeFailure::UnsupportedArchitecture);
    }

    let mut out = [0usize; 32];
    let (len, markers) = probe_depth_4(strategy, &mut out);

    // Deliberately allocation-free. The probe runs from `Profiler::new`, by
    // which point the shim is already installed, so a `Vec` here would be a
    // profiled allocation inside the code deciding whether profiling can work.
    let captured = &out[..len];
    let recovered = markers
        .iter()
        .filter(|&&marker| {
            // A return address lands just after the call instruction, so it is
            // inside the calling function rather than at its entry. Accept any
            // captured address within a generous window after the function's
            // start.
            captured
                .iter()
                .any(|&addr| addr >= marker && addr - marker < 4096)
        })
        .count();

    if recovered >= markers.len() {
        // Only after the strategy is known to work. Timing a walk that returns
        // nothing would report a capture cost of a few nanoseconds and be
        // perfectly true about a capture that captured nothing.
        measure_cost(strategy);
    }

    if recovered < markers.len() {
        return Err(match strategy {
            // The frame-pointer failures name a build flag, because that is
            // almost always the cause and always the remedy.
            Strategy::FramePointer if len < markers.len() => ProbeFailure::ChainTooShort {
                found: len,
                expected: markers.len(),
            },
            Strategy::FramePointer => ProbeFailure::ChainInvalid,
            // The platform unwinder takes no configuration, so there is no flag
            // to name. Zero frames and wrong frames are different faults and the
            // message says which.
            Strategy::System if len == 0 => ProbeFailure::SystemUnwinderEmpty,
            Strategy::System => ProbeFailure::SystemUnwinderWrong { found: len },
        });
    }

    Ok(())
}

/// Captures with `strategy`, for the shim and the probe.
///
/// `stack` is the calling thread's bounds, which only the frame-pointer walk
/// uses; the platform unwinder finds its own way.
#[inline]
pub fn capture_with(
    strategy: Strategy,
    stack: Option<std::ops::Range<usize>>,
    skip: usize,
    out: &mut [usize],
) -> Capture {
    match strategy {
        Strategy::FramePointer => {
            let source = frame_pointer::RealStack::new(stack);
            frame_pointer::capture(&source, skip, out)
        }
        Strategy::System => system::capture(skip, out),
    }
}

/// What one stack capture cost, measured on this machine in this build.
///
/// # Why this is measured rather than timed on the hot path
///
/// The number a reader wants is how much of their program's runtime went into
/// the profiler, and the honest way to get it looks like timing every capture.
/// That does not work: reading the clock costs about as much as a
/// frame-pointer walk (17.7 ns against 21 ns, PLAN.md section 4.4), so timing
/// each capture would roughly triple what it is measuring and report the
/// tripled figure. Sampling instead — timing one capture in every N — needs a
/// counter shared by every thread, which is another contended word on the one
/// path this crate spends its effort keeping uncontended.
///
/// So the cost is measured once, at startup, over a batch large enough that the
/// clock's own resolution does not matter, and the profile carries the raw
/// numbers rather than a rate: `nanos` for `captures` captures. Multiplied by
/// the capture counts, which are exact, that is the profiler's stack-walking
/// time for the run.
///
/// # What it does not include, and what that means
///
/// It times [`capture_with`] and nothing else: not the reentrancy guard, not
/// interning, not the counters, not the peak gate. It is the cost of the stack
/// walk, which is the part that varies by three orders of magnitude between
/// strategies and is the reason [`Strategy::System`] carries a warning.
///
/// The measured stack is the calibration's own, so a program whose stacks are
/// deeper pays more per capture than this says, roughly in proportion to the
/// frame count. [`Cost::frames`] is how deep the measured stack was, so a reader
/// with a profile in front of them can scale it — and the profile's own program
/// points say how deep theirs are. PLAN.md section 5.1's "~5 ns fixed +
/// ~1.3 ns/frame" is the shape of that scaling but not a substitute for this
/// number: it benchmarks the walk alone, where this times a whole
/// [`capture_with`] call including the strategy dispatch and the bounds check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cost {
    /// Nanoseconds the fastest measured batch took.
    ///
    /// The fastest rather than the mean: a machine doing something else while
    /// the profiled program starts adds time and never removes it, so the
    /// minimum over several batches is the one estimate that noise cannot
    /// inflate.
    pub nanos: u64,
    /// Captures in that batch. Zero when nothing was measured.
    pub captures: u64,
    /// How deep the measured stack was.
    pub frames: usize,
    /// Which strategy was measured.
    ///
    /// Carried rather than assumed to match the run's: the two are the same in
    /// any process that started a profiler, because [`probe`] measures the
    /// strategy [`select`] is about to install, but a profile that asserted it
    /// would be asserting the call order rather than reporting a measurement.
    pub strategy: Strategy,
}

impl Cost {
    /// Whether a measurement was taken at all.
    pub fn measured(&self) -> bool {
        self.captures > 0
    }

    /// Picoseconds per capture, or `None` if nothing was measured.
    ///
    /// Picoseconds because a frame-pointer walk costs about 21 ns and rounding
    /// that to whole nanoseconds throws away a twentieth of it.
    pub fn picos_per_capture(&self) -> Option<u64> {
        (self.captures > 0).then(|| self.nanos.saturating_mul(1_000) / self.captures)
    }
}

/// Captures in the first batch, before the size is adjusted to the clock.
const COST_FIRST_BATCH: usize = 64;

/// The largest batch the calibration will grow to.
///
/// 65,536 frame-pointer walks are about 750 µs, at the 11.5 ns per capture this
/// crate measures for itself on aarch64-apple-darwin. Past that the calibration
/// would be spending more of the program's startup than the number is worth.
const COST_MAX_BATCH: usize = 1 << 16;

/// Clock ticks a batch must span before its timing means anything.
///
/// Fifty, so that the per-capture figure carries two significant digits: a
/// batch measured to within one tick of 50 is measured to within 2%.
///
/// **On every platform this crate supports, the floor below dominates this**:
/// Darwin's microsecond granularity gives exactly 50 µs, Linux's nanosecond
/// clock 50 ns, and Windows' performance counter about 5 µs. So the granularity
/// term is a guard against a clock coarser than a microsecond rather than a live
/// input, and a mutation deleting it cannot be caught on any machine this is
/// tested on. It stays because the alternative is a fixed target that reports
/// the clock's own resolution as a measurement on a platform we have not met.
const COST_TICKS_PER_BATCH: u64 = 50;

/// The shortest batch worth timing, whatever the clock says its resolution is.
///
/// A clock can report a fine granularity and still be noisy at that scale, so
/// this is a floor and not a fallback: on Linux the granularity term alone
/// would ask for a 50 ns batch, which is two frame-pointer walks.
const COST_MIN_BATCH_NANOS: u64 = 50_000;

/// Batches to time, of which the fastest is kept.
const COST_BATCHES: usize = 5;

/// How long the whole calibration may take.
///
/// The frame-pointer walk finishes in well under a tenth of this. The budget is
/// here for [`Strategy::System`], which costs 5,613 ns per capture on x86_64
/// glibc: there the very first batch already spans long enough to time, and
/// four more of them would be spent refining a number whose first two digits
/// were settled by the first.
const COST_BUDGET_NANOS: u64 = 2_000_000;

static COST_NANOS: AtomicU64 = AtomicU64::new(0);
static COST_CAPTURES: AtomicU64 = AtomicU64::new(0);
static COST_FRAMES: AtomicUsize = AtomicUsize::new(0);
static COST_STRATEGY: AtomicU8 = AtomicU8::new(Strategy::platform_default() as u8);

/// What one capture cost, or a zeroed [`Cost`] if none was measured.
///
/// Zero for any process that never started a profiler, and for one whose
/// capture strategy failed its probe.
pub fn capture_cost() -> Cost {
    Cost {
        nanos: COST_NANOS.load(Ordering::Relaxed),
        captures: COST_CAPTURES.load(Ordering::Relaxed),
        frames: COST_FRAMES.load(Ordering::Relaxed),
        strategy: Strategy::from_u8(COST_STRATEGY.load(Ordering::Relaxed)),
    }
}

/// Times `strategy`'s captures and records the result.
///
/// Deliberately does **not** go through [`counters`]. Those count the profiled
/// program's captures, and a thousand of the profiler's own at startup would
/// swamp a short run's — a profile of a program making fifty allocations would
/// report a capture-quality figure that was 95% calibration.
fn measure_cost(strategy: Strategy) {
    let mut buffer = [0usize; 32];

    // The batch is sized to the clock rather than fixed, because the clocks
    // this crate is built on differ by three orders of magnitude in what they
    // can resolve. **Measured: `clock_gettime(CLOCK_MONOTONIC)` on Darwin
    // advances in whole microseconds**, so a fixed batch of 64 frame-pointer
    // walks — about 730 ns in a release build, at the 11.5 ns per capture this
    // crate measures for itself — timed as zero on every batch, and the
    // calibration reported no measurement at all rather than a wrong one. Linux
    // resolves nanoseconds and Windows' performance counter typically 100 ns,
    // so no single batch size is right for the three.
    let target = clock_granularity()
        .saturating_mul(COST_TICKS_PER_BATCH)
        .max(COST_MIN_BATCH_NANOS);

    // Grow until one batch spans `target`. The last iteration is both the
    // sizing run and the first timed one, so nothing is measured twice, and it
    // is warm by then: the first capture of all faults in the buffer and trains
    // the branch predictor, and neither is a cost a profiled program pays per
    // allocation.
    let mut captures = COST_FIRST_BATCH;
    let mut spent = 0u64;
    let (mut frames, mut best);
    loop {
        let start = crate::internals::clock::monotonic_nanos();
        frames = cost_batch(strategy, captures, &mut buffer);
        best = crate::internals::clock::monotonic_nanos().saturating_sub(start);
        spent = spent.saturating_add(best);
        if best >= target || captures >= COST_MAX_BATCH || spent >= COST_BUDGET_NANOS {
            break;
        }
        captures *= 2;
    }

    if best < target {
        // The clock does not move, or the largest batch this is willing to run
        // still cannot be timed on it. Reporting nothing is better than
        // reporting a figure that is really the clock's resolution.
        return;
    }

    let mut batches = 1;
    while batches < COST_BATCHES && spent < COST_BUDGET_NANOS {
        let start = crate::internals::clock::monotonic_nanos();
        cost_batch(strategy, captures, &mut buffer);
        let elapsed = crate::internals::clock::monotonic_nanos().saturating_sub(start);
        best = best.min(elapsed);
        spent = spent.saturating_add(elapsed);
        batches += 1;
    }

    COST_NANOS.store(best, Ordering::Relaxed);
    COST_CAPTURES.store(captures as u64, Ordering::Relaxed);
    COST_FRAMES.store(frames, Ordering::Relaxed);
    COST_STRATEGY.store(strategy as u8, Ordering::Relaxed);
}

/// The smallest change this platform's monotonic clock can show.
///
/// Spins until the reading changes, which is the only way to learn it: a clock
/// that reports nanoseconds may still advance in microsecond steps, and nothing
/// in its interface says which. Bounded, so a clock that never advances returns
/// zero rather than hanging the program that asked.
fn clock_granularity() -> u64 {
    /// Reads before giving up. Generous by four orders of magnitude: on Darwin
    /// the answer arrives after about forty, a microsecond tick being roughly
    /// forty `clock_gettime` calls apart **[measured]**.
    const SPINS: u32 = 1 << 22;

    let mut smallest = u64::MAX;
    for _ in 0..3 {
        let start = crate::internals::clock::monotonic_nanos();
        let mut spins = 0;
        loop {
            let now = crate::internals::clock::monotonic_nanos();
            if now > start {
                smallest = smallest.min(now - start);
                break;
            }
            spins += 1;
            if spins == SPINS {
                return 0;
            }
        }
    }
    smallest
}

/// One timed batch of `captures` captures, returning how deep the stack was.
///
/// `#[inline(never)]` so the measured stack is a stack — inlined into
/// [`measure_cost`] it would be one frame shallower than the loop it stands for.
#[inline(never)]
fn cost_batch(strategy: Strategy, captures: usize, buffer: &mut [usize; 32]) -> usize {
    let bounds = crate::internals::stack::current_bounds();
    let mut frames = 0;
    for _ in 0..captures {
        // `black_box` on the buffer, because a capture whose result is never
        // read is a capture the optimiser may delete, and a calibration that
        // measured an empty loop would report a capture cost near zero.
        let capture = capture_with(strategy, bounds.clone(), 0, std::hint::black_box(buffer));
        frames = std::hint::black_box(capture.len);
    }
    frames
}

/// A probe level's own entry address.
///
/// Cast through a function *pointer* rather than straight from the function
/// item: a direct `fn_item as usize` is a lint-worthy cast whose result is not
/// guaranteed to be the address the caller means.
type ProbeFn = fn(Strategy, &mut [usize]) -> (usize, [usize; 3]);

/// Captures from a call stack of known depth, returning the capture length and
/// the start addresses of the functions that were called through.
///
/// Each level is `#[inline(never)]` and passes its result through `black_box`,
/// which prevents both inlining and tail-call elimination from collapsing the
/// chain the probe is trying to measure.
#[inline(never)]
fn probe_depth_4(strategy: Strategy, out: &mut [usize]) -> (usize, [usize; 3]) {
    let (len, mut markers) = std::hint::black_box(probe_depth_3(strategy, out));
    markers[2] = (probe_depth_4 as ProbeFn) as usize;
    std::hint::black_box((len, markers))
}

#[inline(never)]
fn probe_depth_3(strategy: Strategy, out: &mut [usize]) -> (usize, [usize; 3]) {
    let (len, mut markers) = std::hint::black_box(probe_depth_2(strategy, out));
    markers[1] = (probe_depth_3 as ProbeFn) as usize;
    std::hint::black_box((len, markers))
}

#[inline(never)]
fn probe_depth_2(strategy: Strategy, out: &mut [usize]) -> (usize, [usize; 3]) {
    let capture = capture_with(strategy, crate::internals::stack::current_bounds(), 0, out);
    let markers = [(probe_depth_2 as ProbeFn) as usize, 0, 0];
    std::hint::black_box((capture.len, markers))
}

#[cfg(test)]
mod tests {

    /// A depth limit means "keep this many frames", and it has to mean that on
    /// every backend. Unix `backtrace(3)` has no skip parameter, so the frames it
    /// is about to discard come out of the caller's buffer: a limit at or below
    /// the calibrated skip returned nothing at all, and the profile then blamed a
    /// missing frame-pointer chain for a setting the user had chosen
    /// **\[measured\]**.
    #[test]
    fn a_depth_limit_leaves_the_platform_unwinder_room_for_its_own_skip() {
        use super::{depth_room, Strategy};

        // The walker skips before it writes, so it needs exactly what is wanted.
        assert_eq!(depth_room(Strategy::FramePointer, 3, 4), 3);
        assert_eq!(depth_room(Strategy::FramePointer, 64, 0), 64);

        let system = depth_room(Strategy::System, 3, 4);
        if cfg!(windows) {
            assert_eq!(system, 3, "RtlCaptureStackBackTrace skips natively");
        } else {
            assert_eq!(system, 7, "backtrace(3) spends the buffer on the skip");
            assert!(
                depth_room(Strategy::System, 1, 4) > 4,
                "a limit below the skip must still leave a frame to record"
            );
        }

        // A skip large enough to overflow is a skip nothing can honour; the
        // buffer length clamps the result either way, so saturating here keeps
        // the arithmetic from being the thing that fails.
        assert_eq!(
            depth_room(Strategy::System, usize::MAX, usize::MAX),
            usize::MAX
        );
    }

    /// Serialises the tests that touch the process-wide capture strategy.
    ///
    /// `select` writes a global. One test mutates it and restores it; another
    /// asserts what it holds before anything has. Those cannot both be true at
    /// once, and which one loses depends on how the harness happens to schedule
    /// them — it passed on macOS and failed on Linux, on the same commit.
    static SELECTION: crate::internals::lock::RawLock = crate::internals::lock::RawLock::new();
    use super::*;

    /// Real-stack tests are excluded under Miri: reading a frame record is an
    /// integer-to-pointer access with no provenance, which Miri cannot model.
    /// The walk *policy* is covered under Miri by the synthetic-stack tests in
    /// `frame_pointer`.
    /// The strategy this platform picks for itself must work on this platform.
    ///
    /// Stated that way rather than as "frame pointers work", because on Windows
    /// they do not and the platform default is the system unwinder instead.
    #[test]
    #[cfg_attr(miri, ignore = "reads real stack memory, which Miri cannot model")]
    fn the_platform_default_strategy_works_on_this_build() {
        let strategy = Strategy::platform_default();
        match probe(strategy) {
            Ok(()) => {}
            Err(failure) => {
                if strategy == Strategy::FramePointer && cfg!(target_arch = "x86_64") {
                    panic!(
                        "the frame-pointer probe failed on x86_64. CI sets \
                         -C force-frame-pointers=yes; is it set here?\n{failure}"
                    );
                }
                panic!("the {strategy} probe failed: {failure}");
            }
        }
    }

    /// The claim in section 12 of PLAN.md is "honestly measured overhead", and
    /// this is the measurement. A calibration that reported zero, or that
    /// reported a number without having walked a stack, would put a figure in
    /// every profile that nobody could check.
    #[test]
    #[cfg_attr(miri, ignore = "reads real stack memory, which Miri cannot model")]
    fn a_successful_probe_measures_what_a_capture_costs() {
        let strategy = Strategy::platform_default();
        probe(strategy).expect("the platform default works here");

        let cost = capture_cost();
        assert!(cost.measured(), "a successful probe measured nothing");
        assert_eq!(cost.strategy, strategy);
        assert!(
            (COST_FIRST_BATCH..=COST_MAX_BATCH).contains(&(cost.captures as usize)),
            "a batch of {} captures is outside the range the sizing loop can \
             produce",
            cost.captures
        );
        assert!(
            cost.frames >= 2,
            "the calibration measured a walk of {} frames, which is not a stack \
             — so the number it produced is the cost of failing to capture",
            cost.frames
        );

        let picos = cost.picos_per_capture().expect("a measured cost");
        // Three orders of magnitude either side of the measured 21 ns, which is
        // wide enough for a loaded machine, a debug build, an emulator, and the
        // platform unwinder at 5,613 ns — and still excludes both a capture
        // that took no time and one that took a millisecond.
        assert!(
            (20..1_000_000_000).contains(&picos),
            "a capture measured at {picos} picoseconds is not a measurement of \
             a stack walk"
        );
    }

    /// The figure every profile publishes must be a measurement of *this*
    /// machine, which the range above cannot tell.
    ///
    /// Three orders of magnitude is what an absolute band needs in order to
    /// cover a loaded machine, a debug build, an emulator, and the platform
    /// unwinder at 5,613 ns — and a fabricated constant of 21 ns per capture
    /// sits comfortably inside it. **Measured, M7 chunk K:** storing
    /// `captures * 21` in place of the timed value passes the entire suite, and
    /// so does storing the real value multiplied by ten. Every profile then
    /// carries a number the README invites the reader to check, and nothing
    /// checks it.
    ///
    /// A comparison against *another timing of the same work* needs no wide
    /// band, because the two share the machine and the build. So this does what
    /// a reader with a stopwatch would do: time the same number of captures the
    /// same way and require the two to agree.
    ///
    /// # Why both sides are minima over several rounds
    ///
    /// Because otherwise the two timings share the machine but not the *moment*,
    /// and that is enough to fail. **Measured** under twenty busy loops on ten
    /// cores: a calibration taken while the process was time-slicing reports 515
    /// ps per capture and an independent timing taken moments later, when it had
    /// a core to itself, reports 109 — a genuine five-fold change in what the
    /// machine could do, with nothing wrong on either side. Both quantities are
    /// "how long this batch took", so the minimum of several rounds converges to
    /// the uncontended cost from above, and comparing two minima removes the
    /// scheduler from the question.
    ///
    /// It times whichever strategy the published figure *names*, rather than
    /// the platform default, and that is not pedantry — the two differ by two
    /// orders of magnitude, so a profile whose `unwinder` field and whose
    /// number came from different strategies is wrong in a way no absolute band
    /// would show.
    #[test]
    #[cfg_attr(miri, ignore = "times real stack walks, which Miri cannot model")]
    fn the_published_capture_cost_agrees_with_an_independent_timing() {
        /// How far apart two minima of the same work on the same machine may
        /// land. The systematic part is that the calibration includes a colder
        /// batch than a re-timing does; measured at 1.17 to 1.22 across ten idle
        /// runs, so this sits about three times clear of the spread while
        /// staying tight enough to reject a constant.
        const TOLERANCE: u64 = 4;
        /// Rounds of (calibrate, re-time). Each costs about 400 µs.
        const ROUNDS: usize = 5;

        let _serialised = SELECTION.lock();

        let bounds = crate::internals::stack::current_bounds();
        let mut buffer = [0usize; 32];
        let mut published = u64::MAX;
        let mut independent = u64::MAX;
        let mut strategy = Strategy::platform_default();
        let mut batch = 0;

        for _ in 0..ROUNDS {
            probe(Strategy::platform_default()).expect("the platform default works here");
            let cost = capture_cost();
            published = published.min(cost.picos_per_capture().expect("a measured cost"));
            strategy = cost.strategy;
            batch = cost.captures as usize;

            // Written out here rather than calling `cost_batch`, because a
            // stopwatch that is the code under test is not a second opinion.
            let start = crate::internals::clock::monotonic_nanos();
            for _ in 0..batch {
                let capture = capture_with(
                    strategy,
                    bounds.clone(),
                    0,
                    std::hint::black_box(&mut buffer),
                );
                std::hint::black_box(capture.len);
            }
            let elapsed = crate::internals::clock::monotonic_nanos().saturating_sub(start);
            independent = independent.min(elapsed.saturating_mul(1_000) / batch as u64);
        }

        assert!(
            published <= independent.saturating_mul(TOLERANCE)
                && independent <= published.saturating_mul(TOLERANCE),
            "every profile from this build will say a {:?} capture costs {published} \
             picoseconds, and timing the same {batch} captures here says \
             {independent} — so the number a reader is invited to check is not a \
             measurement of the work it names",
            strategy
        );
    }

    /// The calibration must not spend the profile's capture-quality counters on
    /// itself: a program making fifty allocations would otherwise get a profile
    /// whose capture counts were almost entirely startup.
    #[test]
    #[cfg_attr(miri, ignore = "reads real stack memory, which Miri cannot model")]
    fn the_calibration_does_not_count_as_captures_the_program_made() {
        let before = counters().snapshot();
        probe(Strategy::platform_default()).expect("the platform default works here");
        let after = counters().snapshot();

        assert_eq!(
            (
                after.complete - before.complete,
                after.truncated - before.truncated,
                after.suspect - before.suspect,
                after.no_frames - before.no_frames
            ),
            (0, 0, 0, 0),
            "the probe and its calibration added captures to the counts a \
             profile reports about the program"
        );
    }

    /// Calibrating from a stack deeper than the buffer must give the same answer
    /// as calibrating from a shallow one.
    ///
    /// The two captures are taken wherever the caller happens to be, so their
    /// depth is the program's business and not this crate's. Once it passes the
    /// arrays' capacity both walks truncate, and `calibrate` used to require
    /// `deep_len == shallow_len + 1`, which truncation makes impossible — so it
    /// declined, warned, and every program point in that profile began with
    /// `heapscope`'s own frames. A caller thirty-odd frames inside their own
    /// program was enough.
    ///
    /// Found by a ThreadSanitizer run, where an instrumented standard library
    /// pushed the crate's own unit tests past the limit. Nothing about the
    /// finding is specific to sanitizers, which is why the fix is here and not
    /// in a `cfg`.
    ///
    /// The depth is far past the capacity rather than just over it, so that the
    /// test keeps testing truncation if the buffer is ever enlarged.
    #[test]
    #[cfg_attr(miri, ignore = "reads real stack memory and calls the platform")]
    fn calibration_survives_a_stack_deeper_than_its_buffer() {
        #[inline(never)]
        fn deeper(remaining: usize, strategy: Strategy) -> usize {
            match remaining {
                0 => std::hint::black_box(calibrate(strategy)),
                _ => std::hint::black_box(deeper(remaining - 1, strategy)),
            }
        }

        for strategy in [Strategy::platform_default(), Strategy::System] {
            let shallow = calibrate(strategy);
            let deep = deeper(200, strategy);
            assert_eq!(
                deep, shallow,
                "{strategy}: calibrating 200 frames inside the program answered \
                 {deep}, and calibrating at the top answered {shallow}. The two \
                 captures are of the same machinery either way, so a difference \
                 means the depth of the caller decides how many frames the shim \
                 skips."
            );
        }
    }

    /// The system unwinder is available on every supported platform, whether or
    /// not it is the default there. A user reaching for the escape hatch must
    /// find it working.
    #[test]
    #[cfg_attr(miri, ignore = "calls the platform's unwinder")]
    fn the_system_unwinder_works_on_this_build() {
        probe(Strategy::System).expect("the platform's own unwinder should work");
    }

    /// `capture_with` must route each strategy to the backend that names it.
    ///
    /// Swapping the two arms of that match used to pass every test in the
    /// repository on unix. Nothing noticed, because the `unwinder` field a
    /// profile records comes from the global selection rather than from
    /// whichever code actually ran, and every other assertion held for either
    /// backend.
    ///
    /// The discriminator is the innermost frame. The system backend's own frame
    /// sits at the bottom of what it returns, and `system::capture` calls the
    /// platform from exactly one instruction, so **every** capture that went
    /// through it starts at that same address — whatever route it took to get
    /// there. A frame-pointer walk starts somewhere else entirely.
    ///
    /// That is an equality, not a proximity. An earlier version asked whether
    /// the two addresses were within 8 KB of each other, which is a guess about
    /// code layout rather than a fact about it, and the guess was wrong under
    /// `rustc` 1.96 with optimisations on. The same guess in `calibrate` broke
    /// the same way; both are gone.
    #[test]
    #[cfg_attr(miri, ignore = "reads real stack memory and calls the platform")]
    #[cfg_attr(
        windows,
        ignore = "the frame-pointer arm captures nothing here, so the two arms \
                  are not distinguishable"
    )]
    fn each_strategy_reaches_the_backend_that_names_it() {
        #[inline(never)]
        fn dispatched(strategy: Strategy, out: &mut [usize]) -> Capture {
            std::hint::black_box(capture_with(
                strategy,
                crate::internals::stack::current_bounds(),
                0,
                out,
            ))
        }

        #[inline(never)]
        fn straight_to_system(out: &mut [usize]) -> Capture {
            std::hint::black_box(system::capture(0, out))
        }

        let mut reference = [0usize; 32];
        assert!(straight_to_system(&mut reference).len > 0);

        let mut through_system = [0usize; 32];
        assert!(dispatched(Strategy::System, &mut through_system).len > 0);
        assert_eq!(
            through_system[0], reference[0],
            "`capture_with(System, ..)` did not reach the system backend: its \
             innermost frame is {:#x}, and a direct call's is {:#x}",
            through_system[0], reference[0]
        );

        let mut through_frame_pointer = [0usize; 32];
        assert!(dispatched(Strategy::FramePointer, &mut through_frame_pointer).len > 0);
        assert_ne!(
            through_frame_pointer[0], reference[0],
            "`capture_with(FramePointer, ..)` reached the system backend: its \
             innermost frame is {:#x}, which is where a direct system call \
             starts",
            through_frame_pointer[0]
        );
    }

    /// The calibrated skip must land the innermost frame on the code that
    /// asked for the allocation, for every strategy.
    ///
    /// This replaced a bare `SKIP_FRAMES = 1` covering both strategies, which
    /// was right only for a debug-build frame-pointer walk: the platform
    /// unwinder starts several `heapscope` frames further in, and an optimised
    /// frame-pointer walk starts one later. Nothing caught it, because nothing
    /// tested the constant — setting it to 6, stripping five real user frames
    /// from every trace, passed the whole suite.
    ///
    /// The shape below mirrors the shim: `shim_like` stands in for
    /// `<Alloc as GlobalAlloc>::alloc`, which is `#[inline(never)]` for exactly
    /// this reason, and `user_code` for whatever allocated.
    #[test]
    #[cfg_attr(miri, ignore = "reads real stack memory and calls the platform")]
    fn the_calibrated_skip_lands_on_the_code_that_allocated() {
        let _serialised = SELECTION.lock();

        #[inline(never)]
        fn shim_like(strategy: Strategy, out: &mut [usize]) -> usize {
            let capture = capture_with(
                strategy,
                crate::internals::stack::current_bounds(),
                internal_frames(),
                out,
            );
            std::hint::black_box(capture.len)
        }

        #[inline(never)]
        fn user_code(strategy: Strategy, out: &mut [usize]) -> (usize, usize) {
            let len = std::hint::black_box(shim_like(strategy, out));
            let marker = (user_code as fn(Strategy, &mut [usize]) -> (usize, usize)) as usize;
            std::hint::black_box((len, marker))
        }

        let strategies = if Strategy::platform_default() == Strategy::FramePointer {
            &[Strategy::FramePointer, Strategy::System][..]
        } else {
            &[Strategy::System][..]
        };

        let mut innermost: Option<(Strategy, usize)> = None;
        for &strategy in strategies {
            select(strategy);
            let mut out = [0usize; 32];
            let (len, marker) = user_code(strategy, &mut out);
            assert!(len > 0, "{strategy}: nothing captured");
            assert!(
                out[0] >= marker && out[0] - marker < 8192,
                "{strategy}: after skipping {} machinery frames the innermost \
                 frame is {:#x}, which is not in the code that allocated \
                 ({marker:#x}). Every program point would start with heapscope's \
                 own frames.",
                internal_frames(),
                out[0],
            );

            // The check above locates the frame by a window around a function
            // pointer, which is a guess about code layout — the same guess that
            // `calibrate` used to make, and that broke when an unrelated module
            // moved another function into the window. This is the exact
            // version: `user_code` calls `shim_like` from one instruction, so
            // there is exactly one right answer for the innermost frame, and
            // two mechanisms with nothing in common must both produce it.
            match innermost {
                None => innermost = Some((strategy, out[0])),
                Some((first, address)) => assert_eq!(
                    out[0], address,
                    "{first} and {strategy} disagree about which instruction \
                     allocated; at most one of them can be right"
                ),
            }
        }

        // Leave the process as it was found.
        select(Strategy::platform_default());
    }

    #[test]
    fn the_selected_strategy_starts_at_the_platform_default() {
        let _serialised = SELECTION.lock();
        assert_eq!(strategy(), Strategy::platform_default());
    }

    /// The platform unwinder's failure must name the remedy where there is one.
    ///
    /// This test previously asserted the opposite — that the message names no
    /// build flag — on the reasoning that the platform unwinder takes no
    /// configuration. True on Windows and macOS, false on glibc, which is the
    /// one platform where this strategy is a real answer: PLAN.md section 5.2
    /// measured `_Unwind_Backtrace` returning *success* with zero frames under
    /// `-C panic=abort`, and glibc's `backtrace` is built on it. Asserting the
    /// absence locked in a hard refusal with no route out.
    #[test]
    fn a_probe_failure_for_the_system_unwinder_names_the_remedy_where_there_is_one() {
        let message = ProbeFailure::SystemUnwinderEmpty.to_string();
        assert!(message.contains("no frames"), "{message}");
        // Never the frame-pointer flag: it would send the user to fix something
        // that has no bearing on the platform unwinder.
        assert!(!message.contains("force-frame-pointers"), "{message}");

        if cfg!(all(unix, not(target_vendor = "apple"))) {
            assert!(
                message.contains("force-unwind-tables"),
                "on glibc a zero-frame capture is what a `panic = \"abort\"` build \
                 looks like, and that has a remedy: {message}"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "reads real stack memory, which Miri cannot model")]
    #[cfg_attr(
        windows,
        ignore = "the Microsoft x64 ABI has no walkable frame-pointer chain; \
                  `system` covers what Windows actually uses"
    )]
    fn capture_from_a_real_stack_returns_plausible_addresses() {
        let source = frame_pointer::RealStack::new(crate::internals::stack::current_bounds());
        let mut out = [0usize; 32];
        let capture = frame_pointer::capture(&source, 0, &mut out);

        assert!(capture.len > 0, "no frames captured: {capture:?}");
        assert!(
            out[..capture.len].iter().all(|&addr| addr != 0),
            "a captured return address was null"
        );
        // Return addresses point into executable memory, which on every
        // supported platform is nowhere near the low addresses.
        assert!(
            out[..capture.len].iter().all(|&addr| addr > 0x1000),
            "a captured return address is implausibly low: {:x?}",
            &out[..capture.len]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "reads real stack memory, which Miri cannot model")]
    #[cfg_attr(
        windows,
        ignore = "the Microsoft x64 ABI has no walkable frame-pointer chain; \
                  `system` covers what Windows actually uses"
    )]
    fn deeper_call_stacks_capture_more_frames() {
        #[inline(never)]
        fn recurse(depth: usize, out: &mut [usize]) -> usize {
            if depth == 0 {
                let source =
                    frame_pointer::RealStack::new(crate::internals::stack::current_bounds());
                return std::hint::black_box(frame_pointer::capture(&source, 0, out)).len;
            }
            std::hint::black_box(recurse(depth - 1, out))
        }

        let mut shallow = [0usize; 64];
        let mut deep = [0usize; 64];
        let shallow_len = recurse(1, &mut shallow);
        let deep_len = recurse(9, &mut deep);

        assert!(
            deep_len > shallow_len,
            "8 extra call levels did not produce more frames \
             (shallow={shallow_len}, deep={deep_len}); the walk is not following the chain"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "reads real stack memory, which Miri cannot model")]
    #[cfg_attr(
        windows,
        ignore = "the Microsoft x64 ABI has no walkable frame-pointer chain; \
                  `system` covers what Windows actually uses"
    )]
    fn capture_respects_the_depth_limit() {
        #[inline(never)]
        fn recurse(depth: usize, out: &mut [usize]) -> Capture {
            if depth == 0 {
                let source =
                    frame_pointer::RealStack::new(crate::internals::stack::current_bounds());
                return std::hint::black_box(frame_pointer::capture(&source, 0, out));
            }
            std::hint::black_box(recurse(depth - 1, out))
        }

        let mut out = [0usize; 4];
        let capture = recurse(20, &mut out);
        assert_eq!(capture.len, 4);
        assert_eq!(capture.outcome, Outcome::TruncatedByDepth);
    }

    #[test]
    #[cfg_attr(miri, ignore = "reads real stack memory, which Miri cannot model")]
    #[cfg_attr(
        windows,
        ignore = "the Microsoft x64 ABI has no walkable frame-pointer chain; \
                  `system` covers what Windows actually uses"
    )]
    fn capture_works_on_a_spawned_thread() {
        // A spawned thread has a different stack from the main thread's, and on
        // some platforms a differently shaped outermost frame.
        let capture = std::thread::spawn(|| {
            let source = frame_pointer::RealStack::new(crate::internals::stack::current_bounds());
            let mut out = [0usize; 32];
            let capture = frame_pointer::capture(&source, 0, &mut out);
            (capture, out)
        })
        .join()
        .unwrap();

        assert!(capture.0.len > 0, "no frames captured on a spawned thread");
    }

    #[test]
    fn counters_classify_every_outcome() {
        let counters = Counters::new();
        counters.record(Outcome::Complete);
        counters.record(Outcome::Complete);
        counters.record(Outcome::TruncatedByDepth);
        counters.record(Outcome::Suspect);
        counters.record(Outcome::NoFrames);

        assert_eq!(
            counters.snapshot(),
            CounterSnapshot {
                complete: 2,
                truncated: 1,
                suspect: 1,
                no_frames: 1,
            }
        );
    }

    #[test]
    fn probe_failure_messages_name_the_remedy() {
        let message = ProbeFailure::ChainTooShort {
            found: 1,
            expected: 3,
        }
        .to_string();
        assert!(
            message.contains("force-frame-pointers=yes"),
            "the error must tell the user exactly what to do: {message}"
        );
        assert!(
            message.contains("fno-omit-frame-pointer"),
            "cc-built dependencies need their own flag, and no RUSTFLAGS setting reaches them: {message}"
        );
    }
}
