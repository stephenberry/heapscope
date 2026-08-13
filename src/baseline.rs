//! A recorded reading a later run is expected to reproduce.
//!
//! [`assert_max_bytes!`](crate::assert_max_bytes) needs a number somebody chose.
//! A baseline is the other way round: record what the program does today, commit
//! the file, and let the next run fail if it does more. That is the shape a CI
//! gate actually wants, because nobody knows what the budget should be until
//! they have measured it once, and because the file then shows up in a diff —
//! "peak went from 1.2 MB to 4.8 MB" is a line a reviewer reads, where the same
//! change buried in a threshold constant is not.
//!
//! ```no_run
//! # fn work() {}
//! # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
//! work();
//! heapscope::assert_baseline!("tests/baselines/parsing.txt");
//! ```
//!
//! Run with `HEAPSCOPE_UPDATE_BASELINE=1` to write the file rather than check
//! against it. See [`UPDATE_VARIABLE`].
//!
//! # Why this is not JSON
//!
//! Two reasons, and the second is the load-bearing one.
//!
//! A baseline is a handful of integers, and it is **committed to a repository**
//! and read by people in review. One `key value` per line diffs to exactly the
//! figures that moved; a JSON object diffs to the same thing with punctuation,
//! and a pretty-printer disagreement rewrites the whole file.
//!
//! And this crate ships a JSON *writer*, not a reader. Adding a parser to the
//! shipped library so that a baseline could be spelled in JSON would be a real
//! amount of new attack surface — reached, by construction, with a file that
//! someone edited by hand — for a format nothing else here reads back.
//!
//! # A truncated baseline
//!
//! Every known key must be present, so a file cut short by a full disk fails to
//! load rather than comparing against the figures that survived. That rule
//! catches a missing *key* and **not a lost digit**: `totalBlocks 1024` cut to
//! `totalBlocks 1` has every key and a wrong number, and nothing downstream can
//! tell. So this is not a property the reader can enforce, and the writer
//! enforces it instead — [`Baseline::save`] writes beside its destination and
//! renames into place, exactly as
//! [`Snapshot::save_dhat_v2`](crate::Snapshot::save_dhat_v2) does, so a failed
//! write leaves the previous baseline intact rather than replacing a good file
//! with half of one.

use std::fmt;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::internals::engine::Mode;
use crate::stats::{Complaint, HeapStats};

/// Set this and [`assert_baseline!`](crate::assert_baseline) writes the file
/// instead of checking against it.
///
/// `1`, or any value other than `0`, `off`, `no`, and `false` — the same
/// spellings [`crate::stats::DUMP_VARIABLE`] reads, so that knowing one is
/// knowing the other.
///
/// This is the only thing that writes a baseline, and it is deliberately not a
/// "write it if it is missing" rule: a gate whose baseline file has gone missing
/// must fail, not quietly record whatever the run happened to do that day and
/// pass. Recording a baseline is a thing a person decides.
pub const UPDATE_VARIABLE: &str = "HEAPSCOPE_UPDATE_BASELINE";

/// The word the first line starts with. The version follows it on the same
/// line, and together they decide whether this build can read the rest.
const MAGIC: &str = "heapscope-baseline";

/// The format version.
///
/// A reader ignores keys it does not know and refuses a version it does not
/// know, which is the rule the native format states in every file it writes: a
/// new figure is additive, and the version moves only when the meaning of an
/// existing one changes — the case a reader cannot detect for itself.
const VERSION: u32 = 1;

/// What a run recorded, kept for a later run to be measured against.
///
/// `#[non_exhaustive]`: a baseline gains figures whenever there is a new one
/// worth gating on, and a caller building one field by field would break.
/// [`Baseline::of`] is how one is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Baseline {
    /// What the recorded run counted. Always [`Mode::Heap`] today; the field is
    /// written and checked so that a baseline can never be silently compared
    /// against a run measuring something else.
    pub mode: Mode,
    /// Bytes live when the baseline was taken.
    pub curr_bytes: u64,
    /// Blocks live when the baseline was taken.
    pub curr_blocks: u64,
    /// The peak. DHAT's `gmax`.
    pub max_bytes: u64,
    /// Blocks live at that peak.
    pub max_blocks: u64,
    /// Bytes ever allocated.
    pub total_bytes: u64,
    /// Allocations ever made.
    pub total_blocks: u64,
}

/// One figure and its key in the file, so that reading and writing cannot
/// disagree about the set and a failure can name the line to edit.
type Figure = (&'static str, fn(&Baseline) -> u64, fn(&HeapStats) -> u64);

const FIGURES: [Figure; 6] = [
    ("currBytes", |b| b.curr_bytes, |s| s.curr_bytes),
    ("currBlocks", |b| b.curr_blocks, |s| s.curr_blocks),
    ("maxBytes", |b| b.max_bytes, |s| s.max_bytes),
    ("maxBlocks", |b| b.max_blocks, |s| s.max_blocks),
    ("totalBytes", |b| b.total_bytes, |s| s.total_bytes),
    ("totalBlocks", |b| b.total_blocks, |s| s.total_blocks),
];

impl Baseline {
    /// The baseline a run with these counters would record.
    pub fn of(stats: &HeapStats) -> Baseline {
        Baseline {
            mode: Mode::Heap,
            curr_bytes: stats.curr_bytes,
            curr_blocks: stats.curr_blocks,
            max_bytes: stats.max_bytes,
            max_blocks: stats.max_blocks,
            total_bytes: stats.total_bytes,
            total_blocks: stats.total_blocks,
        }
    }

    /// Reads a baseline from `path`.
    ///
    /// # Errors
    ///
    /// Whatever opening the file produced, or [`io::ErrorKind::InvalidData`]
    /// with a message naming the line at fault. See [`Baseline::parse`] for what
    /// counts as at fault.
    pub fn read(path: impl AsRef<Path>) -> io::Result<Baseline> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        Baseline::parse(&text)
    }

    /// Reads a baseline from the text of one.
    ///
    /// Separate from [`Baseline::read`] because the format is worth testing
    /// without a filesystem — under Miri there is not one, and a malformed
    /// baseline is much easier to write as a string than as a file.
    ///
    /// # Errors
    ///
    /// A first line that is not `heapscope-baseline <version>`, a version this
    /// build does not know, a line that is not `key value`, a value that is not
    /// a number, a key given twice, or any known key missing. Keys it does not
    /// know are ignored, which is what makes adding a figure a compatible
    /// change.
    pub fn parse(text: &str) -> io::Result<Baseline> {
        let mut lines = text.lines().enumerate();
        let (_, first) = lines
            .next()
            .ok_or_else(|| invalid("the file is empty; a baseline starts with a version line"))?;
        parse_version(first)?;

        let mut mode = None;
        let mut values = [None; FIGURES.len()];
        for (at, line) in lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line.split_once(char::is_whitespace).ok_or_else(|| {
                invalid(&format!(
                    "line {}: expected `key value`, found `{}`",
                    at + 1,
                    fragment(line)
                ))
            })?;
            let value = value.trim();

            if key == "mode" {
                if mode.is_some() {
                    return Err(invalid(&format!("line {}: `mode` given twice", at + 1)));
                }
                mode = Some(parse_mode(value).ok_or_else(|| {
                    invalid(&format!(
                        "line {}: `{}` is not a mode",
                        at + 1,
                        fragment(value)
                    ))
                })?);
                continue;
            }

            let Some(figure) = FIGURES.iter().position(|(name, _, _)| *name == key) else {
                // Deliberately not an error. A baseline written by a later
                // version of this crate is readable by this one, minus whatever
                // it does not understand.
                continue;
            };
            if values[figure].is_some() {
                return Err(invalid(&format!("line {}: `{key}` given twice", at + 1)));
            }
            values[figure] = Some(value.parse::<u64>().map_err(|_| {
                invalid(&format!(
                    "line {}: `{}` is not a number that fits a u64",
                    at + 1,
                    fragment(value)
                ))
            })?);
        }

        // Absent rather than defaulted. A missing `maxBytes` defaulted to zero
        // fails every run, and defaulted to `u64::MAX` passes every run; a
        // baseline is a contract, and a partial one is not one.
        let mode = mode.ok_or_else(|| invalid("`mode` is missing"))?;
        if mode != Mode::Heap {
            return Err(invalid(&format!(
                "this baseline describes a run counting {mode} events; \
                 only heap baselines can be compared"
            )));
        }
        let mut figures = [0u64; FIGURES.len()];
        for (at, value) in values.iter().enumerate() {
            figures[at] =
                value.ok_or_else(|| invalid(&format!("`{}` is missing", FIGURES[at].0)))?;
        }

        Ok(Baseline {
            mode,
            curr_bytes: figures[0],
            curr_blocks: figures[1],
            max_bytes: figures[2],
            max_blocks: figures[3],
            total_bytes: figures[4],
            total_blocks: figures[5],
        })
    }

    /// Writes this baseline to `path`, replacing whatever is there.
    ///
    /// Written beside its destination and renamed into place, so a failed write
    /// leaves the previous baseline intact rather than replacing a good one
    /// with half a file. The same treatment
    /// [`Snapshot::save_dhat_v2`](crate::Snapshot::save_dhat_v2) gives a
    /// profile, and it is needed here for a sharper reason: the completeness
    /// rule catches a file missing a *key*, and a file cut inside the last
    /// value — `totalBlocks 1024` truncated to `totalBlocks 1` — has every key
    /// and a wrong number. Nothing downstream can detect that, so it must not
    /// be possible to write.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let mut partial = path.as_os_str().to_os_string();
        partial.push(".partial");
        let partial = std::path::PathBuf::from(partial);

        let mut file = std::fs::File::create(&partial)?;
        if let Err(error) = self.write(&mut file).and_then(|()| file.flush()) {
            let _ = std::fs::remove_file(&partial);
            return Err(error);
        }
        drop(file);
        std::fs::rename(&partial, path).inspect_err(|_| {
            let _ = std::fs::remove_file(&partial);
        })
    }

    /// Writes this baseline in the format [`Baseline::parse`] reads.
    pub fn write<W: Write>(&self, mut out: W) -> io::Result<()> {
        writeln!(out, "{MAGIC} {VERSION}")?;
        // The instruction travels in the file rather than in documentation the
        // file will be read without, which is the same reason the native format
        // carries its own compatibility rule.
        writeln!(
            out,
            "# Written by heapscope {}. Rewrite with {UPDATE_VARIABLE}=1.",
            crate::VERSION
        )?;
        writeln!(out, "mode {}", self.mode)?;
        for (name, of_baseline, _) in FIGURES {
            writeln!(out, "{name} {}", of_baseline(self))?;
        }
        Ok(())
    }

    /// Every figure that grew past what `tolerance` allows.
    ///
    /// Empty is a pass. One-sided on purpose: a run that allocates *less* than
    /// its baseline is the outcome the gate exists to encourage, and failing it
    /// would be a gate people turn off. Rerecord with
    /// [`UPDATE_VARIABLE`] when an improvement should become the new floor.
    pub fn compare(&self, stats: &HeapStats, tolerance: Tolerance) -> Vec<Regression> {
        let mut regressions = Vec::new();
        for (name, of_baseline, of_stats) in FIGURES {
            let recorded = of_baseline(self);
            let measured = of_stats(stats);
            let allowed = tolerance.allowance(recorded);
            if measured > allowed {
                regressions.push(Regression {
                    figure: name,
                    baseline: recorded,
                    measured,
                    allowed,
                });
            }
        }
        regressions
    }
}

/// How much growth a comparison lets through.
///
/// A percentage of the recorded figure, and nothing else. An absolute term was
/// considered and left out: one number cannot be both a byte allowance and a
/// block allowance, and a `Tolerance` carrying two would have to be told which
/// figure it was being asked about at every use.
///
/// The arithmetic is integer, so a percentage of a small figure rounds **down**
/// — 5% of 3 blocks allows nothing. That is the right way round for a gate: a
/// small count is exactly the case where an increase of one is a real change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tolerance {
    percent: u32,
}

impl Tolerance {
    /// No growth at all.
    ///
    /// The default, and usually the right one: none of the six figures depends
    /// on a clock, so two runs of the same workload record the same numbers
    /// under either [`TimeSource`](crate::TimeSource). (§6.9 ties this to
    /// `Events`; that is true of a profile's lifetimes and not of anything a
    /// baseline holds.) Loosen it where the workload itself is not
    /// deterministic — a thread pool that sizes itself to the machine, an input
    /// read from the network.
    pub const fn exact() -> Tolerance {
        Tolerance { percent: 0 }
    }

    /// Growth of up to `percent` of each recorded figure.
    pub const fn percent(percent: u32) -> Tolerance {
        Tolerance { percent }
    }

    /// The largest value `baseline` may grow to.
    ///
    /// Saturating: a tolerance wide enough to overflow means everything is
    /// allowed, which is what `u64::MAX` says.
    pub fn allowance(&self, baseline: u64) -> u64 {
        let headroom = u128::from(baseline) * u128::from(self.percent) / 100;
        u64::try_from(u128::from(baseline) + headroom).unwrap_or(u64::MAX)
    }
}

/// One figure that grew past its allowance.
///
/// `#[non_exhaustive]`: this is a report, and reports gain detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Regression {
    /// The key naming this figure in the baseline file, so a reader knows which
    /// line moved.
    pub figure: &'static str,
    /// What the baseline recorded.
    pub baseline: u64,
    /// What this run measured.
    pub measured: u64,
    /// The largest value the tolerance allowed.
    pub allowed: u64,
}

impl fmt::Display for Regression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = crate::output::count;
        write!(
            f,
            "{}: {} against a baseline of {}",
            self.figure,
            count(self.measured),
            count(self.baseline)
        )?;
        if self.allowed != self.baseline {
            write!(f, ", above the {} allowed", count(self.allowed))?;
        }
        Ok(())
    }
}

/// Why a baseline check did not pass.
#[derive(Debug)]
enum Failure {
    /// There were no numbers to compare. Not a failure of the program.
    Stats(Complaint),
    /// No file to compare against.
    Missing(PathBuf),
    /// A file that could not be read, or that is not a baseline.
    Unreadable { path: PathBuf, error: io::Error },
    /// The file could not be written during an update run.
    NotWritten { path: PathBuf, error: io::Error },
    /// The run did more than the baseline records.
    Regressed {
        path: PathBuf,
        regressions: Vec<Regression>,
    },
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Failure::Stats(complaint) => write!(f, "{complaint}"),
            Failure::Missing(path) => write!(
                f,
                "there is no baseline at {}; record one by running with \
                 {UPDATE_VARIABLE}=1",
                screened(path)
            ),
            // The error is screened as well as the path, because screening at
            // the point a message is *built* is not the boundary that matters:
            // `invalid` screens the file fragments it interpolates, and an
            // `io::Error` from anywhere else arrives here unexamined. The
            // boundary is the terminal, so the screening belongs where the text
            // is rendered for one.
            Failure::Unreadable { path, error } => write!(
                f,
                "the baseline at {} is unusable: {}; record a fresh one by \
                 running with {UPDATE_VARIABLE}=1",
                screened(path),
                crate::stats::screened(&error.to_string())
            ),
            Failure::NotWritten { path, error } => write!(
                f,
                "the baseline at {} could not be written: {}",
                screened(path),
                crate::stats::screened(&error.to_string())
            ),
            Failure::Regressed { path, regressions } => {
                write!(
                    f,
                    "this run is above the baseline recorded in {}:",
                    screened(path)
                )?;
                for regression in regressions {
                    write!(f, "\n    {regression}")?;
                }
                write!(
                    f,
                    "\n  if this run is the new correct answer, rerecord with \
                     {UPDATE_VARIABLE}=1"
                )
            }
        }
    }
}

/// A path on its way to a terminal, with anything that would drive one removed.
///
/// The same treatment image paths and symbol names get in the output layer. A
/// baseline path comes from the caller rather than from us, and a panic message
/// is no less a place for a terminal to be told to do something.
fn screened(path: &Path) -> String {
    crate::stats::screened(&path.display().to_string())
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

/// Characters of a baseline's own text that reach a message.
const MAX_FRAGMENT: usize = 64;

/// A fragment of the file, on its way into a panic message.
///
/// Screened for the reason a path is — this is the one input the module
/// documents as hand-edited, so it is the likeliest of the two to carry an
/// escape sequence — and **capped**, which the path is not. A line of a file
/// someone edited can be as long as their editor allowed, and a megabyte of it
/// interpolated into a panic message is a denial of the message: the numbers
/// that say what actually failed scroll away above it.
fn fragment(text: &str) -> String {
    let mut cut: String = text.chars().take(MAX_FRAGMENT).collect();
    let elided = cut.chars().count() < text.chars().count();
    cut = crate::stats::screened(&cut);
    if elided {
        cut.push('\u{2026}');
    }
    cut
}

fn parse_version(line: &str) -> io::Result<u32> {
    let Some(version) = line.trim().strip_prefix(MAGIC) else {
        return Err(invalid(&format!(
            "the first line is `{}`, not `{MAGIC} <version>`",
            fragment(line.trim())
        )));
    };
    let version: u32 = version.trim().parse().map_err(|_| {
        invalid(&format!(
            "`{}` is not a version number",
            fragment(version.trim())
        ))
    })?;
    if version != VERSION {
        return Err(invalid(&format!(
            "this is a version {version} baseline and this build of heapscope \
             reads version {VERSION}"
        )));
    }
    Ok(version)
}

fn parse_mode(value: &str) -> Option<Mode> {
    [Mode::Heap, Mode::AdHoc, Mode::Copy]
        .into_iter()
        .find(|mode| mode.as_str() == value)
}

/// Whether this run should record the baseline rather than check against it.
fn updating() -> bool {
    wants_update(std::env::var_os(UPDATE_VARIABLE).as_deref())
}

/// Whether a setting of [`UPDATE_VARIABLE`] asks for the file to be rewritten.
///
/// Pure, so that the off spellings can be enumerated without a test mutating
/// the environment out from under every other test in the binary — which is
/// why nothing checked them, and why `HEAPSCOPE_UPDATE_BASELINE=0` turning a
/// gate into a recorder went unnoticed.
fn wants_update(setting: Option<&std::ffi::OsStr>) -> bool {
    match setting {
        Some(setting) => !crate::stats::is_off(setting),
        None => false,
    }
}

/// The body of [`assert_baseline!`](crate::assert_baseline). Not a supported
/// entry point.
#[doc(hidden)]
#[track_caller]
pub fn __assert_baseline(
    path: impl AsRef<Path>,
    tolerance: Tolerance,
    context: Option<fmt::Arguments<'_>>,
) {
    let path = path.as_ref();
    // The guard has to start *before* the environment is read, not inside
    // `check`. Asking whether a variable is set allocates on Windows whatever
    // the answer is — `var_os` converts the name to UTF-16 first — and that
    // read happens before `check` snapshots the counters, so unguarded it lands
    // inside the very numbers the check is about to compare. It is 52 bytes,
    // one block, and it is charged to the program under test on every call, so
    // a gate asserted in a loop drifts upwards a variable name at a time.
    // Nothing on unix shows it: `getenv` and `unsetenv` build their C string in
    // a stack buffer for names this short, and `var_os` allocates only for a
    // value that exists. `check` takes the guard again and gets `None`, which
    // is the documented answer for a thread already inside.
    let outcome = {
        let _quiet = crate::internals::guard::enter();
        check(crate::engine(), path, tolerance, updating())
    };
    if matches!(outcome, Ok(Checked::Recorded)) {
        // Not a diagnostic: a run that rewrote the file it was supposed to be
        // checked against has to say so, or a green CI job that had
        // `HEAPSCOPE_UPDATE_BASELINE` set in its environment reads as a gate
        // that passed.
        let _quiet = crate::internals::guard::enter();
        let _ = writeln!(
            io::stderr(),
            "heapscope: recorded the baseline in {} rather than checking against it",
            screened(path)
        );
    }
    crate::stats::report(outcome.map(|_| ()), context);
}

/// What a check that did not fail did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Checked {
    /// The run was compared against the file and stayed within it.
    Passed,
    /// The file was rewritten from this run instead of being checked against.
    Recorded,
}

/// The engine, the environment, and the filesystem all reach this through
/// arguments rather than through globals, and that is what makes the failure
/// arms testable at all.
///
/// A mutation replacing the `Unreadable` arm with `Ok(())` — a corrupt baseline
/// silently passing the gate — survived the entire suite, because nothing could
/// construct the state that reaches it. Same for `NotWritten`. Both are the
/// case where a CI gate reports success on the run where it mattered.
fn check(
    engine: &crate::internals::engine::Engine,
    path: &Path,
    tolerance: Tolerance,
    updating: bool,
) -> Result<Checked, Failure> {
    // Held for the whole check, because reading the baseline allocates and this
    // is the one assertion whose *passing* path does. What it protects is the
    // *next* reading rather than this one — the counters are snapshotted before
    // the file is opened — which is what the record-then-check pair in
    // `tests/testing_api.rs` exercises: without the guard, recording a baseline
    // and immediately checking against it disagrees by the size of the read.
    //
    // `None` here when `__assert_baseline` is the caller, which takes the guard
    // one step earlier so that reading the environment is inside it too. The
    // unit tests below call this directly and are what this line still serves.
    let _quiet = crate::internals::guard::enter();
    let stats = crate::stats::assertable(engine).map_err(Failure::Stats)?;

    if updating {
        let baseline = Baseline::of(&stats);
        return match baseline.save(path) {
            Ok(()) => Ok(Checked::Recorded),
            Err(error) => Err(Failure::NotWritten {
                path: path.to_path_buf(),
                error,
            }),
        };
    }

    let baseline = match Baseline::read(path) {
        Ok(baseline) => baseline,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err(Failure::Missing(path.to_path_buf()))
        }
        Err(error) => {
            return Err(Failure::Unreadable {
                path: path.to_path_buf(),
                error,
            })
        }
    };

    let regressions = baseline.compare(&stats, tolerance);
    if regressions.is_empty() {
        return Ok(Checked::Passed);
    }
    Err(Failure::Regressed {
        path: path.to_path_buf(),
        regressions,
    })
}

/// Fails unless this run stays within the baseline recorded at `path`.
///
/// The baseline is a committed file recording what the program did when someone
/// last looked. Every figure in it is compared against this run, and any that
/// grew past the tolerance fails the test, naming which one and by how much.
///
/// ```no_run
/// # fn work() {}
/// # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// use heapscope::Tolerance;
///
/// work();
/// heapscope::assert_baseline!("tests/baselines/work.txt");
/// heapscope::assert_baseline!("tests/baselines/work.txt", Tolerance::percent(5));
/// ```
///
/// The default tolerance is [`Tolerance::exact`]: none of these figures depends
/// on a clock, so two runs of the same workload record the same numbers and a
/// gate can be exact rather than approximately reassuring.
///
/// # The trailing message needs a tolerance in front of it
///
/// Unlike the other three assertions, this one's second argument is a
/// [`Tolerance`], so a message has to come third:
///
/// ```no_run
/// # let fixture = "big.json";
/// # let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
/// use heapscope::Tolerance;
/// heapscope::assert_baseline!("b.txt", Tolerance::exact(), "while parsing {fixture}");
/// ```
///
/// Writing it in the two-argument form gives a type error naming `Tolerance`
/// rather than a silent misparse, which is the right way round, but it is worth
/// knowing before the compiler tells you.
///
/// # Recording one
///
/// Run with `HEAPSCOPE_UPDATE_BASELINE=1` and the file is written instead of
/// checked. That is the only thing that writes it — a missing baseline **fails**
/// rather than recording itself, because a gate that silently records whatever
/// it found the first time it could not find a file is a gate that passes
/// forever.
///
/// # Panics
///
/// When any figure regressed, when the baseline is missing or unreadable, and
/// when there are no numbers to check — see [`crate::stats`] for that list.
#[macro_export]
macro_rules! assert_baseline {
    ($path:expr $(,)?) => {
        $crate::__assert_baseline(
            $path,
            $crate::Tolerance::exact(),
            ::core::option::Option::None,
        )
    };
    ($path:expr, $tolerance:expr $(,)?) => {
        $crate::__assert_baseline($path, $tolerance, ::core::option::Option::None)
    };
    ($path:expr, $tolerance:expr, $($arg:tt)+) => {
        $crate::__assert_baseline(
            $path,
            $tolerance,
            ::core::option::Option::Some(::core::format_args!($($arg)+)),
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reading in which **no two figures are equal**, so a comparison test
    /// can tell which row moved.
    ///
    /// The first version of this hardcoded `curr_bytes: 0` and
    /// `curr_blocks: 0`, so the two live rows could be made to measure nothing
    /// at all and every test stayed green — the figure a leak-over-time gate is
    /// entirely about.
    fn stats(max_bytes: u64, total_blocks: u64) -> HeapStats {
        HeapStats {
            curr_bytes: max_bytes / 4,
            curr_blocks: 2,
            max_bytes,
            max_blocks: 3,
            total_bytes: max_bytes * 2,
            total_blocks,
            dropped_blocks: 0,
        }
    }

    /// A reading that differs from `stats(..)` in exactly one row.
    fn grown_by(base: &HeapStats, figure: &str, by: u64) -> HeapStats {
        let mut grown = *base;
        match figure {
            "currBytes" => grown.curr_bytes += by,
            "currBlocks" => grown.curr_blocks += by,
            "maxBytes" => grown.max_bytes += by,
            "maxBlocks" => grown.max_blocks += by,
            "totalBytes" => grown.total_bytes += by,
            "totalBlocks" => grown.total_blocks += by,
            other => panic!("no such figure: {other}"),
        }
        grown
    }

    fn serialized() -> crate::internals::lock::RawGuard<'static> {
        crate::internals::diagnostic::POISON_TESTS.lock()
    }

    /// A started engine of this crate's own, for the `check` arms.
    fn engine() -> crate::internals::engine::Engine {
        let engine = crate::internals::engine::Engine::with_limits(1 << 10, 1 << 12);
        assert!(engine.start(crate::TimeSource::Events, || {}));
        engine
    }

    fn written(baseline: &Baseline) -> String {
        let mut text = Vec::new();
        baseline.write(&mut text).expect("writing to a Vec");
        String::from_utf8(text).expect("the format is ASCII")
    }

    #[test]
    fn a_baseline_survives_being_written_and_read_back() {
        let recorded = Baseline::of(&stats(65_536, 1_024));
        let text = written(&recorded);
        assert_eq!(Baseline::parse(&text).expect("its own output"), recorded);
    }

    /// The file tells its own reader how to rewrite it, because the file is
    /// what someone opens when the gate fails.
    #[test]
    fn the_file_says_what_it_is_and_how_to_refresh_it() {
        let text = written(&Baseline::of(&stats(1, 1)));
        assert!(text.starts_with("heapscope-baseline 1\n"), "{text}");
        assert!(text.contains(UPDATE_VARIABLE), "{text}");
        assert!(text.contains("mode heap"), "{text}");
    }

    #[test]
    fn comments_and_blank_lines_are_not_data() {
        let text = "heapscope-baseline 1\n\n# a note\nmode heap\ncurrBytes 1\n\
                    currBlocks 2\nmaxBytes 3\nmaxBlocks 4\ntotalBytes 5\ntotalBlocks 6\n";
        let baseline = Baseline::parse(text).expect("a readable baseline");
        assert_eq!(baseline.curr_bytes, 1);
        assert_eq!(baseline.total_blocks, 6);
    }

    /// The rule the native format states in every file it writes, applied here:
    /// a key from a later version is ignored, a version from a later one is not.
    #[test]
    fn an_unknown_key_is_ignored_and_an_unknown_version_is_not() {
        let mut text = written(&Baseline::of(&stats(8, 2)));
        text.push_str("sampledBytes 99\n");
        assert!(Baseline::parse(&text).is_ok());

        let future = text.replace("heapscope-baseline 1", "heapscope-baseline 2");
        let error = Baseline::parse(&future).expect_err("a version this build cannot read");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("version 2"), "{error}");
    }

    /// Each of these would otherwise produce a baseline that compares against
    /// numbers nobody recorded.
    #[test]
    fn a_baseline_missing_anything_is_refused() {
        let complete = written(&Baseline::of(&stats(8, 2)));

        for key in ["mode", "currBytes", "maxBytes", "totalBlocks"] {
            let damaged: String = complete
                .lines()
                .filter(|line| !line.starts_with(&format!("{key} ")))
                .collect::<Vec<_>>()
                .join("\n");
            assert_ne!(damaged, complete, "the `{key}` line was already absent");
            let error = Baseline::parse(&damaged).expect_err("a baseline missing `{key}`");
            assert!(error.to_string().contains(key), "{error}");
        }
    }

    #[test]
    fn a_key_given_twice_is_refused() {
        let mut text = written(&Baseline::of(&stats(8, 2)));
        text.push_str("maxBytes 999\n");
        let error = Baseline::parse(&text).expect_err("an ambiguous baseline");
        assert!(error.to_string().contains("twice"), "{error}");
        // The duplicate-key sites carry a line number too, and are the two a
        // line-number mutation reached when the others had been reworded.
        assert!(error.to_string().contains("line 10"), "{error}");

        let mut text = written(&Baseline::of(&stats(8, 2)));
        text.push_str("mode heap\n");
        assert!(Baseline::parse(&text).is_err());
    }

    #[test]
    fn a_file_that_is_not_a_baseline_is_refused() {
        for text in ["", "{}\n", "hello\nmode heap\n", "heapscope-baseline x\n"] {
            let error = Baseline::parse(text).expect_err("not a baseline: {text:?}");
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        }

        let malformed = "heapscope-baseline 1\nmaxBytes\n";
        assert!(Baseline::parse(malformed).is_err());

        let unparseable = "heapscope-baseline 1\nmaxBytes twelve\n";
        assert!(Baseline::parse(unparseable).is_err());
    }

    /// A heap baseline compared against an ad hoc run would be comparing bytes
    /// against dimensionless weights.
    #[test]
    fn a_baseline_of_another_mode_is_refused() {
        let text = written(&Baseline::of(&stats(8, 2))).replace("mode heap", "mode ad-hoc");
        let error = Baseline::parse(&text).expect_err("an event-mode baseline");
        assert!(error.to_string().contains("ad-hoc"), "{error}");
    }

    /// Every row is compared, and each one on its own. Two of the six were
    /// unreachable by any test until this existed: a mutation making the
    /// `currBytes` row measure a constant zero passed the entire suite.
    #[test]
    fn each_figure_is_compared_and_names_itself() {
        let base = stats(1_000, 10);
        let baseline = Baseline::of(&base);
        for figure in [
            "currBytes",
            "currBlocks",
            "maxBytes",
            "maxBlocks",
            "totalBytes",
            "totalBlocks",
        ] {
            let grown = grown_by(&base, figure, 1);
            let regressions = baseline.compare(&grown, Tolerance::exact());
            assert_eq!(
                regressions.iter().map(|r| r.figure).collect::<Vec<_>>(),
                [figure],
                "growing {figure} did not report {figure} and nothing else"
            );
            assert_eq!(regressions[0].measured, regressions[0].baseline + 1);
        }
    }

    #[test]
    fn an_exact_comparison_passes_on_equality_and_fails_one_above_it() {
        let baseline = Baseline::of(&stats(1_000, 10));
        assert!(baseline
            .compare(&stats(1_000, 10), Tolerance::exact())
            .is_empty());

        let regressions = baseline.compare(&stats(1_001, 10), Tolerance::exact());
        assert_eq!(regressions.len(), 2, "{regressions:?}");
        assert_eq!(regressions[0].figure, "maxBytes");
        assert_eq!(regressions[0].measured, 1_001);
        assert_eq!(regressions[0].allowed, 1_000);
        assert_eq!(regressions[1].figure, "totalBytes");
    }

    /// A gate that failed on an improvement is a gate people turn off.
    #[test]
    fn using_less_than_the_baseline_is_not_a_regression() {
        let baseline = Baseline::of(&stats(1_000, 10));
        assert!(baseline
            .compare(&stats(1, 1), Tolerance::exact())
            .is_empty());
    }

    #[test]
    fn a_tolerance_is_a_percentage_that_rounds_down() {
        let ten = Tolerance::percent(10);
        assert_eq!(ten.allowance(1_000), 1_100);
        // 10% of 3 is 0.3, and a gate on three blocks should not let a fourth
        // through.
        assert_eq!(ten.allowance(3), 3);
        assert_eq!(Tolerance::exact().allowance(1_000), 1_000);

        // Wide enough to overflow means everything is allowed, and says so
        // rather than wrapping to a tiny allowance.
        assert_eq!(Tolerance::percent(1_000).allowance(u64::MAX), u64::MAX);
        assert_eq!(Tolerance::percent(0).allowance(u64::MAX), u64::MAX);
    }

    #[test]
    fn a_tolerance_lets_growth_through_up_to_its_limit() {
        let baseline = Baseline::of(&stats(1_000, 10));
        let five = Tolerance::percent(5);
        assert!(baseline.compare(&stats(1_050, 10), five).is_empty());
        assert_eq!(baseline.compare(&stats(1_051, 10), five).len(), 2);
    }

    /// The message is the whole of what a failing CI job shows.
    #[test]
    fn a_regression_names_the_line_to_look_at() {
        let regression = Regression {
            figure: "maxBytes",
            baseline: 1_048_576,
            measured: 4_194_304,
            allowed: 1_101_004,
        };
        let message = regression.to_string();
        assert!(message.contains("maxBytes"), "{message}");
        assert!(message.contains("4,194,304"), "{message}");
        assert!(message.contains("1,048,576"), "{message}");
        assert!(message.contains("1,101,004"), "{message}");

        // With no tolerance there is no third number, and printing the baseline
        // twice reads as a bug in the profiler rather than a budget.
        let exact = Regression {
            allowed: 1_048_576,
            ..regression
        };
        assert_eq!(
            exact.to_string().matches("1,048,576").count(),
            1,
            "{exact:?}"
        );
    }

    /// A missing baseline is the case a gate most needs to fail on, and the
    /// message has to say how to make one.
    #[test]
    fn a_missing_baseline_names_the_variable_that_records_one() {
        let message = Failure::Missing(PathBuf::from("baselines/work.txt")).to_string();
        assert!(message.contains("baselines/work.txt"), "{message}");
        assert!(message.contains(UPDATE_VARIABLE), "{message}");
    }

    #[test]
    fn a_regression_failure_lists_every_figure() {
        let message = Failure::Regressed {
            path: PathBuf::from("b.txt"),
            regressions: vec![
                Regression {
                    figure: "maxBytes",
                    baseline: 1,
                    measured: 2,
                    allowed: 1,
                },
                Regression {
                    figure: "totalBlocks",
                    baseline: 3,
                    measured: 4,
                    allowed: 3,
                },
            ],
        }
        .to_string();
        assert!(message.contains("maxBytes"), "{message}");
        assert!(message.contains("totalBlocks"), "{message}");
    }

    /// A path in a failure message comes from the caller and is on its way to a
    /// terminal — and so does the *file*, which is the one this module
    /// documents as hand-edited. Only one of the four failures was checked, and
    /// the file body was not screened at all.
    #[test]
    fn nothing_in_a_failure_can_drive_the_terminal() {
        let hostile = PathBuf::from("work\u{1b}[2Kmasked.txt");
        let failures = [
            Failure::Missing(hostile.clone()),
            Failure::Unreadable {
                path: hostile.clone(),
                error: invalid("line 2: `\u{1b}[2K` is not a number that fits a u64"),
            },
            Failure::NotWritten {
                path: hostile.clone(),
                error: invalid("\u{1b}[2K"),
            },
            Failure::Regressed {
                path: hostile,
                regressions: vec![Regression {
                    figure: "maxBytes",
                    baseline: 1,
                    measured: 2,
                    allowed: 1,
                }],
            },
        ];
        for failure in failures {
            let message = failure.to_string();
            assert!(!message.contains('\u{1b}'), "{failure:?}: {message}");
        }
    }

    /// The file is hand-edited, so one of its lines can be as long as somebody's
    /// editor allowed. A megabyte of it in a panic message is a denial of the
    /// message: the numbers saying what actually failed scroll away above it.
    #[test]
    fn a_fragment_of_the_file_is_screened_and_capped() {
        let escape = fragment("run\u{1b}[2Kmasked");
        assert!(!escape.contains('\u{1b}'), "{escape}");

        let long = "x".repeat(MAX_FRAGMENT * 100);
        let cut = fragment(&long);
        assert!(cut.chars().count() <= MAX_FRAGMENT + 1, "{}", cut.len());
        assert!(cut.ends_with('\u{2026}'), "an elision has to say it elided");

        // Short enough to keep whole, and kept whole.
        assert_eq!(fragment("maxBytes"), "maxBytes");
    }

    /// A parse failure names the line to edit, and named the one before it.
    #[test]
    fn a_parse_failure_names_the_line_it_is_about() {
        let text = "heapscope-baseline 1\nmode heap\nmaxBytes twelve\n";
        let error = Baseline::parse(text).expect_err("`twelve` is not a number");
        assert!(error.to_string().contains("line 3"), "{error}");

        let text = "heapscope-baseline 1\n# a note\n\nmaxBytes\n";
        let error = Baseline::parse(text).expect_err("a line that is not `key value`");
        assert!(error.to_string().contains("line 4"), "{error}");
    }

    /// The format is one people align by hand in a review. A value with padding
    /// around it is the same value.
    #[test]
    fn a_value_is_read_without_the_whitespace_around_it() {
        let text = "heapscope-baseline 1\nmode   heap\ncurrBytes    1\n\
                    currBlocks 2\nmaxBytes   3   \nmaxBlocks 4\n\
                    totalBytes 5\ntotalBlocks 6\n";
        let baseline = Baseline::parse(text).expect("an aligned baseline is a baseline");
        assert_eq!(baseline.max_bytes, 3);
        assert_eq!(baseline.mode, Mode::Heap);
    }

    /// The one spelling nobody checked, and it turns a gate into a recorder.
    #[test]
    fn the_update_variable_reads_the_same_off_spellings_as_the_others() {
        for off in ["0", "off", "OFF", "no", "NO", "false", "FALSE", " off "] {
            assert!(
                !wants_update(Some(std::ffi::OsStr::new(off))),
                "{off:?} asked for a rewrite"
            );
        }
        for on in ["1", "yes", "true", "on", ""] {
            assert!(
                wants_update(Some(std::ffi::OsStr::new(on))),
                "{on:?} did not ask for a rewrite"
            );
        }
        assert!(!wants_update(None), "an unset variable rewrites nothing");
    }

    /// Every arm of `check`, including the two that no test could reach until
    /// it took its engine and its environment as arguments. Both of those arms
    /// could be replaced with `Ok(())` — a corrupt baseline and a failed
    /// recording each passing the gate green — with the whole suite passing.
    #[test]
    #[cfg_attr(miri, ignore = "reads and writes files, and Miri has no filesystem")]
    fn every_way_a_check_can_end() {
        let _serial = serialized();
        let engine = engine();
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("work.txt");
        let exact = Tolerance::exact();

        // Missing: the gate must fail rather than record itself.
        let missing = check(&engine, &path, exact, false).expect_err("there is no baseline");
        assert!(matches!(missing, Failure::Missing(_)), "{missing:?}");
        assert!(
            !path.exists(),
            "a check wrote the file it was meant to read"
        );

        // Recorded, then passing against what was just recorded.
        assert_eq!(
            check(&engine, &path, exact, true).expect("recording a baseline"),
            Checked::Recorded
        );
        assert_eq!(
            check(&engine, &path, exact, false).expect("checking against what was just recorded"),
            Checked::Passed
        );

        // Unreadable: a file that is not a baseline must not pass.
        std::fs::write(&path, "not a baseline at all\n").expect("writing the file");
        let unusable = check(&engine, &path, exact, false).expect_err("a corrupt baseline");
        assert!(
            matches!(unusable, Failure::Unreadable { .. }),
            "{unusable:?}"
        );
        assert!(
            unusable.to_string().contains(UPDATE_VARIABLE),
            "a reader has to be told how to replace it: {unusable}"
        );

        // NotWritten: a recording that could not happen must not pass either.
        let unwritable = directory.path().join("no-such-directory").join("work.txt");
        let failed = check(&engine, &unwritable, exact, true).expect_err("an unwritable path");
        assert!(matches!(failed, Failure::NotWritten { .. }), "{failed:?}");
    }

    /// A run whose live-block table filled is missing however many blocks it
    /// turned away, so the gate aimed at CI must refuse it rather than pass on
    /// the one run where the measurement was incomplete. This called
    /// `HeapStats::of` directly and skipped the shared gate entirely.
    #[test]
    #[cfg_attr(miri, ignore = "reads and writes files, and Miri has no filesystem")]
    fn a_run_that_dropped_blocks_is_not_compared_against_a_baseline() {
        let _serial = serialized();
        let engine = crate::internals::engine::Engine::with_limits(1 << 10, 1 << 12);
        assert!(engine.start(crate::TimeSource::Events, || engine.configure(
            crate::internals::engine::Settings {
                max_live_blocks: 1,
                ..crate::internals::engine::Settings::default()
            }
        )));

        let mut address = 0x1000;
        while crate::stats::HeapStats::of(&engine).unwrap().dropped_blocks == 0 {
            engine.record_alloc_guarded(address, crate::internals::shape::Shape::of(16), &[0x1000]);
            address += 0x10;
            assert!(address < 0x1000_0000, "the ceiling was never reached");
        }

        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("work.txt");
        let refused = check(&engine, &path, Tolerance::exact(), true)
            .expect_err("an incomplete measurement is not a baseline");
        assert!(matches!(refused, Failure::Stats(_)), "{refused:?}");
        assert!(
            !path.exists(),
            "an incomplete run recorded itself as the baseline"
        );
    }
}
