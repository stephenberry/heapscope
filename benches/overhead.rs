//! What profiling costs, against `dhat-rs` and against not profiling at all.
//!
//! PLAN.md section 9's M6 exit criterion is "published overhead vs. dhat-rs".
//! This is the publication, and it is a driver rather than a benchmark: the
//! three configurations it compares cannot share a process, because a
//! `#[global_allocator]` is chosen at compile time and there is exactly one per
//! binary. So the work lives in `examples/overhead/workload.rs`, three fixtures
//! wrap it in three allocators, and this runs them.
//!
//! ```text
//! cargo build --release --examples && cargo bench --bench overhead
//! ```
//!
//! On x86_64 both fixtures need `RUSTFLAGS="-C force-frame-pointers=yes"` on
//! that build — heapscope refuses to start without it (PLAN.md section 5.3),
//! and giving it to one fixture and not the others would be measuring the flag.
//!
//! # What this is careful about
//!
//! A benchmark between two tools is mostly an argument about whether they were
//! asked to do the same thing, so the setup is matched everywhere it can be and
//! the places it cannot be are rows rather than footnotes. Both fixtures run in
//! heap mode, write one DHAT version 2 profile at shutdown, and trim frames.
//! Capture depth is the one setting that materially drives cost and the two
//! tools default differently — 64 frames here, 10 in `dhat-rs` — so the table
//! carries heapscope twice, once at its own default and once at `dhat-rs`'s, and
//! the matched comparison is the second of those against the `dhat-rs` row.
//!
//! Three things are checked rather than assumed, because each is a way this
//! could publish a flattering number without anyone noticing:
//!
//! - **All three fixtures did the same work.** Each reports a checksum derived
//!   from every block it touched, and a thread count where the three disagree
//!   fails the run. A fixture edited into doing less work would otherwise read
//!   as an improvement.
//! - **The captured stacks reached the workload.** The deepest of the three
//!   churn call chains must appear by name in every profile written. A profiler
//!   whose traces stopped inside its own shim would be fast and useless, and the
//!   table cannot tell those apart.
//! - **The fixtures are not stale.** Their sources and the library's are checked
//!   against the binaries' timestamps. `cargo bench --bench overhead` rebuilds
//!   this driver and not the fixtures, so without this a run reports on whatever
//!   the examples were last built from. That is not hypothetical: PLAN.md
//!   section 9.1 records the same trap catching `tests/lifecycle.rs`, where a
//!   deliberately broken `fork` passed the whole file.
//!
//! Peak resident set size is measured because it is the axis where this crate
//! has no reason to expect to win: heapscope keeps its state in an arena sized
//! for a four-million-block ceiling, and publishing only the axis it wins on
//! would be choosing the result. The reading comes from `getrusage`, whose
//! `ru_maxrss` is bytes on Darwin and kilobytes everywhere else; the conversion
//! was cross-checked once against `/usr/bin/time -l` on aarch64-apple-darwin.
//!
//! # Result, aarch64-apple-darwin, 10 cores
//!
//! Nanoseconds per allocation, 250,000 allocations per thread:
//!
//! | configuration | 1t | 4t |
//! |---|---|---|
//! | no profiler | 31.9 | 15.0 |
//! | heapscope, 10 frames | **129.4** | 300.1 |
//! | heapscope, default (64) | 125.1 | 325.1 |
//! | heapscope, platform unwinder | 248.1 | 345.3 |
//! | dhat-rs, default (10) | **8,353.6** | 8,758.2 |
//! | dhat-rs, 5 frames | 5,882.9 | 6,220.5 |
//! | heapscope, sampled (128 KiB) | **51.0** | 69.8 |
//!
//! | configuration | peak RSS | shutdown | profile | blocks counted |
//! |---|---|---|---|---|
//! | no profiler | 4.8 MiB | | | |
//! | heapscope, 10 frames | 7.1 MiB | 1.0 ms | 16.4 KiB | 250,010 |
//! | dhat-rs, default (10) | 11.3 MiB | 5.8 ms | 17.9 KiB | 250,010 |
//! | heapscope, sampled (128 KiB) | 7.0 MiB | 1.2 ms | 12.4 KiB | 248,568 |
//!
//! **At one thread and a matched ten-frame capture, heapscope adds about 97 ns
//! per allocation against `dhat-rs`'s 8,300: a factor of roughly 85.** The
//! table is one run; across eleven adequately sampled ones the heapscope row
//! fell between 123.1 and 130.6 ns, the `dhat-rs` row between 7,547 and 8,354,
//! and the ratio between 78 and 85. All of them were taken on a machine carrying
//! a load average of about ten on ten cores, so the absolutes are pessimistic
//! and the ratio is the durable part.
//!
//! **Most of that difference is the unwinder, and the `5 frames` row is what
//! says so rather than an argument that it must be.** Five frames of capture
//! cost `dhat-rs` 1,829 ns, about 370 a frame, which is the order PLAN.md
//! section 5.1 measures for `_Unwind_Backtrace` in isolation. `backtrace-rs`
//! also allocates a `Vec` of frames per capture and hashes it, and the whole of
//! it happens under one global lock.
//!
//! **heapscope's own default depth costs nothing over ten frames here, because
//! the workload's stacks are about ten frames deep**: the walk runs out of stack
//! before it runs out of buffer. Checked rather than assumed — the emitted
//! profile stops changing above a depth of 5, and the count of trimmed frames
//! saturates at 81 from a depth of 10 upward. Shallow stacks are the case that
//! *favours* `dhat-rs`, whose cost is per frame; a program with deeper ones
//! widens this gap rather than narrowing it.
//!
//! **Four threads is where heapscope looks worst, and that is the useful part
//! of this table.** The unprofiled row falls from 33.3 to 12.0 ns as the work
//! spreads out, so the workload itself scales. heapscope's rises from 123.5 to
//! 323.2, which means aggregate throughput *falls* from 8.1 to 3.1 million
//! allocations a second as three more cores are added. That is
//! `benches/contention.rs`'s finding — five globally contended atomics per
//! event — arriving through the shim end to end, and it is the measurement M6's
//! remaining work is against. `dhat-rs` barely moves (7,683 to 8,028) because
//! its per-allocation cost is dominated by a capture that does not contend, so
//! its global lock never gets the chance to be the bottleneck.
//!
//! **Memory and shutdown go the same way as the timing**, which was not a
//! foregone conclusion: heapscope's arena is sized for a four-million-block
//! ceiling. It costs 2.4 MiB over the unprofiled run against `dhat-rs`'s 6.3,
//! and writes its profile in 1.2 ms against 6.2. The live set here is 4,096
//! blocks, though, and a program holding millions is a different experiment
//! than this one.
//!
//! **Both tools counted 250,010 blocks**, independently, on every unsampled
//! configuration and both thread counts. That is the only check in this project
//! where something other than heapscope audits heapscope's accounting.
//!
//! # Sampling, and where its cost floor is
//!
//! **Sampling cuts what profiling costs by about five times and does not go
//! below a floor of roughly 18 ns per allocation.** At 128 KiB the recorded cost
//! is 51.0 ns against 129.4 unsampled and 31.9 unprofiled, so the overhead falls
//! from 97.5 ns to 19.1. It scales better as well: from one thread to four the
//! unsampled row rises by 2.3x and the sampled one by 1.4x, because an
//! allocation that is not sampled never takes the contended lines that
//! `benches/contention.rs` measures.
//!
//! The floor is not the sampled captures, and this is what says so — the same
//! fixture at one thread, minimum of eight runs, against a true 174,059,696
//! bytes in 250,010 blocks:
//!
//! | interval | ns/alloc | bytes | blocks | program points |
//! |---|---|---|---|---|
//! | none | 128.8 | exact | exact | 7 |
//! | 16 KiB | 56.1 | +9.1% | +9.3% | 7 |
//! | 128 KiB | 51.6 | +6.0% | -0.6% | 7 |
//! | 1 MiB | 51.5 | -14.3% | -31.3% | 7 |
//! | 16 MiB | 48.5 | -42.2% | -90.2% | **2** |
//!
//! **Raising the interval by 128x buys 3 ns and costs most of the accuracy.**
//! What remains at every interval is the per-allocation work that sampling does
//! not skip: entering the guard, counting the request in the size histograms,
//! and advancing the countdown. Only the stack capture is behind the sampling
//! decision, and by 128 KiB it has already been removed.
//!
//! So the useful control is the number of sample points, not the interval:
//! accuracy tracks bytes-allocated divided by interval, and this workload
//! allocates 174 MiB, which is about 1,330 points at 128 KiB and 10 at 16 MiB.
//! **Aim for a thousand or more**; past that the estimate stops improving and
//! below a few hundred it degrades fast. The last row is the shape of getting it
//! wrong: five of the seven program points are missing entirely, so the profile
//! has stopped naming the program rather than merely counting it imprecisely.
//!
//! **PLAN.md section 9 asks for "sampling overhead in low single digits" and
//! that is not what this measures**, on any reading that makes the phrase mean
//! percent. 19 ns on a 31.9 ns baseline is 60%. Reaching single digits would
//! mean sampling the size histograms too, which is what makes a sampled profile
//! able to state its own accuracy — `observedBlocks` against `totalBlocks` — and
//! that trade is not made here.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime};

// Shared with the integration tests, which locate their fixtures the same way
// and got the same thing wrong. Included as a file rather than through
// `tests/support/mod.rs`: a bench has no business compiling the profile
// validators, and this is the only piece of that module it wants.
#[path = "../tests/support/fixture.rs"]
mod fixture;

// Why a benchmark has to recognise `cargo test` at all is in the file.
#[path = "support/harness.rs"]
mod harness;
use harness::run_as_a_test;

/// Runs per cell, at least. The fastest is kept.
///
/// The minimum rather than the mean, because every source of noise on a
/// developer's machine makes a run slower and none makes it faster: the fastest
/// run is the one least contaminated by what this is not measuring.
///
/// How many runs it takes for that minimum to settle is not the same for every
/// row, and a fixed count gets it wrong in both directions. The cheap rows here
/// take about thirty milliseconds, which is short enough that one scheduling
/// decision by the operating system moves the figure by a factor of three — a
/// run under a load average of ten produced 388, 149 and 126 ns for the same
/// cell on three attempts. The `dhat-rs` rows take eight seconds and vary by
/// about three percent, because at that length the same interference is spread
/// too thin to matter. So a cell is sampled until it has spent [`BUDGET`]
/// rather than until it has been run a set number of times: the fast rows get
/// dozens of samples for the price of one slow row's five.
const MIN_RUNS: usize = 5;

/// Runs per cell, at most.
///
/// High, because the cell that needs the most samples is the cheapest one: the
/// unprofiled row is eight milliseconds of work, it is the noisiest thing in the
/// table, and it is subtracted from every other row — so its noise is in every
/// overhead figure this publishes. At twenty-five samples it moved between 32
/// and 46 ns on consecutive runs.
const MAX_RUNS: usize = 100;

/// Wall-clock a cell may spend before it stops starting new runs.
///
/// Forty-five seconds, and the cell that sets it is the one-thread `dhat-rs`
/// row at about two seconds a run. Sampling is what that row's figure turns on:
/// across nine runs of this benchmark it read between 7,589 and 10,438 ns, and
/// the two readings above ten thousand are exactly the two where a loaded
/// machine let the cell finish only seven runs and three. Twenty seconds was
/// not enough to make that stop happening. This is the whole reason a run takes
/// minutes, and it buys the difference between a number and a number that moves
/// by a third depending on what else the machine was doing.
const BUDGET: Duration = Duration::from_secs(45);

/// Thread counts measured.
///
/// Fixed rather than derived from the machine, so that two runs on two machines
/// compare. One thread is the per-event cost; four is where `dhat-rs`'s global
/// lock and heapscope's shards are asked different questions.
const THREAD_COUNTS: [usize; 2] = [1, 4];

/// The name that must appear in every profile written.
///
/// The deepest of the workload's three churn call chains. See
/// `examples/overhead/workload.rs`.
const DEEPEST_CALL_SITE: &str = "churn_two_deeper";

/// One row of the table: a fixture and the settings it is run with.
struct Configuration {
    /// How the row is labelled.
    label: &'static str,
    /// The example binary to run.
    fixture: &'static str,
    /// The `<frames>` argument, `"default"` or a count.
    frames: &'static str,
    /// The `<unwinder>` argument, `"default"` or `"system"`.
    unwinder: &'static str,
    /// The `<sampling>` argument, `"none"` or a mean interval in bytes.
    sampling: &'static str,
    /// Whether this configuration writes a profile.
    profiles: bool,
}

const CONFIGURATIONS: [Configuration; 7] = [
    Configuration {
        label: "no profiler",
        fixture: "overhead_none",
        frames: "default",
        unwinder: "default",
        sampling: "none",
        profiles: false,
    },
    Configuration {
        label: "heapscope, 10 frames",
        fixture: "overhead_heapscope",
        frames: "10",
        unwinder: "default",
        sampling: "none",
        profiles: true,
    },
    Configuration {
        label: "heapscope, default (64)",
        fixture: "overhead_heapscope",
        frames: "default",
        unwinder: "default",
        sampling: "none",
        profiles: true,
    },
    // heapscope's escape hatch for a build without frame pointers. What it
    // isolates depends on the platform, and the difference matters: on
    // x86_64-linux it is `_Unwind_Backtrace`, the same mechanism `backtrace-rs`
    // reaches for `dhat-rs`, so the gap to the row above is the unwinder's
    // share of the cost and the gap to the `dhat-rs` row is the engine's. On
    // macOS it is libSystem's `backtrace`, which is itself a frame-pointer walk
    // (`src/unwind/system.rs` measures 5x, not 110x), so there it separates
    // nothing and is simply a third heapscope configuration.
    Configuration {
        label: "heapscope, platform unwinder",
        fixture: "overhead_heapscope",
        frames: "10",
        unwinder: "system",
        sampling: "none",
        profiles: true,
    },
    Configuration {
        label: "dhat-rs, default (10)",
        fixture: "overhead_dhat",
        frames: "default",
        unwinder: "default",
        sampling: "none",
        profiles: true,
    },
    // A second depth, so the per-frame slope is measured rather than inferred:
    // that is what says whether the row above is dominated by its unwinder or
    // by everything else it does per allocation.
    //
    // Five and not four, which is the shallowest `trim_backtraces` accepts,
    // because at four the captured stacks no longer reach the workload — the
    // call-site check below refuses that profile, and it is right to. Comparing
    // the cost of a profile that names the allocating code against one that
    // does not would be measuring the wrong difference.
    Configuration {
        label: "dhat-rs, 5 frames",
        fixture: "overhead_dhat",
        frames: "5",
        unwinder: "default",
        sampling: "none",
        profiles: true,
    },
    // The second half of M6's exit criterion. Same engine, same depth, same
    // profile written, and the only difference is that most allocations never
    // reach the stack walk -- so the gap to the `10 frames` row is what sampling
    // buys, measured rather than reasoned about.
    //
    // 128 KiB rather than the megabyte a production default would use, because
    // the checks have to stay real: the count check below holds this row to
    // twenty per cent, and the call-site check still requires the workload's
    // deepest chain to appear by name in the profile. At this interval the run
    // takes roughly 1,300 samples, which is enough for both; at a megabyte it
    // would be about 160 and the call-site check would start turning on which
    // allocations happened to be drawn.
    Configuration {
        label: "heapscope, sampled (128 KiB)",
        fixture: "overhead_heapscope",
        frames: "10",
        unwinder: "default",
        sampling: "131072",
        profiles: true,
    },
];

/// What one fixture run reported.
struct Record {
    /// Runs this one was the fastest of. Filled in by [`fastest_of`]; a single
    /// run reports zero, because it was not the fastest of anything.
    runs: usize,
    /// How far the slowest of those runs was above this one, as a percentage.
    ///
    /// Reported per cell rather than kept to a caveat, because the cells do not
    /// deserve equal trust and nothing else in the table says so. A row that
    /// ran a hundred times and varied by three percent has been measured; one
    /// that ran five times and varied by forty has been sampled, and a reader
    /// deciding what to conclude needs to see which is which.
    spread: f64,
    /// What the fixture called itself. Checked against the row's label.
    configuration: String,
    allocations: usize,
    checksum: u64,
    workload_ns: u128,
    shutdown_ns: u128,
    max_rss_bytes: Option<u64>,
    total_blocks: Option<u64>,
    profile_bytes: Option<u64>,
}

impl Record {
    /// Nanoseconds per allocation, across all threads.
    fn per_allocation(&self) -> f64 {
        self.workload_ns as f64 / self.allocations as f64
    }
}

fn main() {
    if run_as_a_test() {
        return;
    }

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let scratch = tempfile::tempdir().expect("a scratch directory for the profiles");

    println!(
        "overhead against dhat-rs {}, {} cores available",
        dhat_version(),
        cores
    );
    println!(
        "{} allocations per thread, each cell the fastest of {MIN_RUNS} to {MAX_RUNS} runs\n",
        allocations_per_thread(&scratch),
    );

    // Indexed by configuration, then by thread count.
    let mut table: Vec<Vec<Record>> = Vec::new();
    for configuration in &CONFIGURATIONS {
        let mut row = Vec::new();
        for &threads in &THREAD_COUNTS {
            row.push(fastest_of(configuration, threads, scratch.path()));
        }
        table.push(row);
    }

    require_identical_work(&table);
    report(&table);
}

/// Runs one configuration until the budget is spent and keeps the fastest.
///
/// Progress goes to stderr so that a run of this can be piped somewhere without
/// the report acquiring lines that are not part of it.
fn fastest_of(configuration: &Configuration, threads: usize, scratch: &Path) -> Record {
    let profile = scratch.join(format!("{}-{threads}t.json", configuration.fixture));
    let mut fastest: Option<Record> = None;
    let mut slowest_ns = 0;
    let mut runs = 0;

    let started = Instant::now();
    while runs < MAX_RUNS && (runs < MIN_RUNS || started.elapsed() < BUDGET) {
        let record = run_fixture(configuration, threads, &profile);
        if configuration.profiles {
            require_the_call_sites_reached_the_profile(configuration, &profile);
        }
        require_the_count_is_the_workload_s(configuration, threads, &record);
        slowest_ns = slowest_ns.max(record.workload_ns);
        if fastest
            .as_ref()
            .is_none_or(|best| record.workload_ns < best.workload_ns)
        {
            fastest = Some(record);
        }
        runs += 1;
    }

    let mut fastest = fastest.expect("MIN_RUNS is not zero");
    fastest.runs = runs;
    fastest.spread = (slowest_ns - fastest.workload_ns) as f64 / fastest.workload_ns as f64 * 100.0;
    eprintln!(
        "  {:<30} {threads}t  {:>8.1} ns/alloc  ({runs} runs, spread {:.0}%)",
        configuration.label,
        fastest.per_allocation(),
        fastest.spread,
    );
    fastest
}

/// Runs a fixture once and parses what it printed.
fn run_fixture(configuration: &Configuration, threads: usize, profile: &Path) -> Record {
    let binary = fixture_binary(configuration.fixture);
    let output = Command::new(&binary)
        .arg(threads.to_string())
        .arg(configuration.frames)
        .arg(configuration.unwinder)
        .arg(configuration.sampling)
        .arg(profile)
        .output()
        .unwrap_or_else(|error| panic!("could not run {}: {error}", binary.display()));

    assert!(
        output.status.success(),
        "{} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        binary.display(),
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let record = parse(&String::from_utf8_lossy(&output.stdout), &binary);

    // Every fixture reports the same keys, so a row pointed at the wrong binary
    // produces an entirely plausible row under a label that is now a lie.
    assert_eq!(
        record.configuration,
        tool_claimed_by(configuration.label),
        "the row labelled `{}` ran {}, which reports itself as `{}`",
        configuration.label,
        binary.display(),
        record.configuration,
    );

    record
}

/// The tool a row's label claims it is measuring.
///
/// Read out of the label, and that is the entire point of it. The first version
/// took the expectation from `Configuration::fixture` — the same field the
/// defect lives in — so a mutation that pointed a heapscope row at the
/// `dhat-rs` fixture moved the expectation along with it and survived. A check
/// is only a check against something it does not derive from.
fn tool_claimed_by(label: &str) -> &'static str {
    match label.split(',').next().unwrap_or(label).trim() {
        "no profiler" => "none",
        "dhat-rs" => "dhat",
        "heapscope" => "heapscope",
        other => panic!("the row label `{other}` does not begin with a tool's name"),
    }
}

/// Reads the fixture's `key=value` report.
///
/// Every key the driver needs is required. A missing one is a fixture that
/// stopped early, and defaulting it to zero would turn that into the fastest
/// run in the table.
fn parse(stdout: &str, binary: &Path) -> Record {
    let mut fields: Vec<(&str, &str)> = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = line.split_once('=').unwrap_or_else(|| {
            panic!(
                "{} printed a line that is not `key=value`: {line:?}",
                binary.display()
            )
        });
        fields.push((key, value));
    }

    let find = |key: &str| fields.iter().find(|(k, _)| *k == key).map(|(_, v)| *v);
    let required = |key: &str| -> &str {
        find(key).unwrap_or_else(|| {
            panic!(
                "{} did not report `{key}`.\nIt printed:\n{stdout}",
                binary.display()
            )
        })
    };
    let number = |key: &str, text: &str| -> u128 {
        text.parse().unwrap_or_else(|error| {
            panic!(
                "{} reported `{key}={text}`, which is not a number: {error}",
                binary.display()
            )
        })
    };

    Record {
        runs: 0,
        spread: 0.0,
        configuration: required("configuration").to_string(),
        allocations: number("allocations", required("allocations")) as usize,
        checksum: number("checksum", required("checksum")) as u64,
        workload_ns: number("workload-ns", required("workload-ns")),
        shutdown_ns: number("shutdown-ns", required("shutdown-ns")),
        max_rss_bytes: find("max-rss-bytes").map(|text| number("max-rss-bytes", text) as u64),
        total_blocks: find("total-blocks").map(|text| number("total-blocks", text) as u64),
        profile_bytes: find("profile-bytes").map(|text| number("profile-bytes", text) as u64),
    }
}

/// Refuses a run where the three fixtures did not do the same work.
///
/// The checksum covers every block the workload touched, in order, and the
/// thread count is folded into it. Configurations that disagree are not
/// comparable, and the failure mode this catches — an edit that makes one
/// fixture do less — presents as a faster row rather than as an error.
fn require_identical_work(table: &[Vec<Record>]) {
    for (column, threads) in THREAD_COUNTS.iter().enumerate() {
        let baseline = &table[0][column];
        for (row, configuration) in CONFIGURATIONS.iter().enumerate().skip(1) {
            let record = &table[row][column];
            assert_eq!(
                record.checksum, baseline.checksum,
                "at {threads} thread(s), `{}` did different work from `{}`: \
                 checksum {} against {}. The fixtures share \
                 `examples/overhead/workload.rs`, so they cannot disagree unless \
                 one of them was changed.",
                configuration.label, CONFIGURATIONS[0].label, record.checksum, baseline.checksum,
            );
            assert_eq!(
                record.allocations,
                baseline.allocations,
                "at {threads} thread(s), `{}` made {} allocations against `{}`'s {}",
                configuration.label,
                record.allocations,
                CONFIGURATIONS[0].label,
                baseline.allocations,
            );
        }
    }
}

/// Refuses a run whose profiler did not count the allocations the workload made.
///
/// Both tools count the run for themselves, and both counts have to land on the
/// workload's. This is the one place in the project where an independent
/// implementation checks heapscope's accounting rather than heapscope checking
/// its own, which is worth more than it costs: a shim that dropped one
/// allocation in a thousand would produce a plausible profile and a plausible
/// per-allocation figure, and nothing else here would notice.
///
/// The allowance is for what a fixture allocates around the workload -- its
/// bookkeeping vectors, the runtime's own startup -- and not for accounting
/// error, which is why it is a flat hundred rather than a percentage.
///
/// A sampled row cannot be held to that, because its count is an estimate by
/// construction. It is held to a band instead, which is a weaker check and still
/// a real one: an estimator that had lost its scaling would be out by the
/// sampling ratio, which is three orders of magnitude, not twenty per cent.
fn require_the_count_is_the_workload_s(
    configuration: &Configuration,
    threads: usize,
    record: &Record,
) {
    let Some(blocks) = record.total_blocks else {
        return;
    };
    let intended = record.allocations as u64;
    if configuration.sampling != "none" {
        let error = (blocks as f64 - intended as f64) / intended as f64;
        assert!(
            error.abs() < 0.20,
            "at {threads} thread(s), `{}` estimated {blocks} blocks where the \
             workload made {intended} ({:+.1}%)",
            configuration.label,
            error * 100.0,
        );
        return;
    }
    assert!(
        (intended..intended + 100).contains(&blocks),
        "at {threads} thread(s), `{}` counted {blocks} blocks where the workload \
         made {intended}",
        configuration.label,
    );
}

/// Refuses a profile that does not name the workload's deepest call site.
///
/// A profiler whose captured stacks stop before they reach the code that
/// allocated produces a fast, small, useless profile, and every number in the
/// table above would still look reasonable. This is the check that tells the
/// two apart.
fn require_the_call_sites_reached_the_profile(configuration: &Configuration, profile: &Path) {
    let written = std::fs::read_to_string(profile).unwrap_or_else(|error| {
        panic!(
            "`{}` reported a profile at {} that cannot be read: {error}",
            configuration.label,
            profile.display()
        )
    });
    assert!(
        written.contains(DEEPEST_CALL_SITE),
        "`{}` wrote a profile with no frame naming `{DEEPEST_CALL_SITE}`, so its \
         captured stacks did not reach the code that allocated. The profile is at \
         {} ({} bytes).",
        configuration.label,
        profile.display(),
        written.len(),
    );
}

// --- Reporting ------------------------------------------------------------

/// Width of the label column, wide enough for the longest label.
const LABEL: usize = 30;

/// Width of every other column.
const COLUMN: usize = 12;

fn report(table: &[Vec<Record>]) {
    println!("nanoseconds per allocation, lower is better");
    print!("{:<LABEL$}", "configuration");
    for threads in THREAD_COUNTS {
        print!("{:>COLUMN$}", format!("{threads}t"));
    }
    println!();
    println!("{}", "-".repeat(LABEL + COLUMN * THREAD_COUNTS.len()));
    for (row, configuration) in CONFIGURATIONS.iter().enumerate() {
        print!("{:<LABEL$}", configuration.label);
        for record in &table[row] {
            print!("{:>COLUMN$.1}", record.per_allocation());
        }
        println!();
    }
    let runs: Vec<String> = table
        .iter()
        .flatten()
        .map(|record| format!("{}/{:.0}%", record.runs, record.spread))
        .collect();
    println!(
        "\nSamples per cell and the spread across them, row by row: {}.\n\
         A cell with few samples and a wide spread is one to read as an order of\n\
         magnitude rather than as a figure.\n\
         \n\
         Across threads these are aggregate figures -- wall clock over every thread's\n\
         allocations -- so the unprofiled row falls from one thread to four as the work\n\
         spreads out, and a profiled row that rises instead is one whose recording does\n\
         not scale.",
        runs.join(" ")
    );

    println!(
        "\nwhat a run cost besides time, at {} thread",
        THREAD_COUNTS[0]
    );
    print!("{:<LABEL$}", "configuration");
    for heading in ["peak RSS", "shutdown", "profile", "blocks"] {
        print!("{heading:>COLUMN$}");
    }
    println!();
    println!("{}", "-".repeat(LABEL + COLUMN * 4));
    for (row, configuration) in CONFIGURATIONS.iter().enumerate() {
        let record = &table[row][0];
        print!("{:<LABEL$}", configuration.label);
        print!(
            "{:>COLUMN$}",
            record.max_rss_bytes.map_or("-".into(), bytes)
        );
        print!("{:>COLUMN$}", milliseconds(record.shutdown_ns));
        print!(
            "{:>COLUMN$}",
            record.profile_bytes.map_or("-".into(), bytes)
        );
        print!(
            "{:>COLUMN$}",
            record
                .total_blocks
                .map_or("-".to_string(), |blocks| blocks.to_string())
        );
        println!();
    }

    summarise(table);
}

/// The sentence M6's exit criterion asks for, computed from the table above.
///
/// Computed rather than written down, because a headline figure maintained by
/// hand is one that stops matching the rows underneath it and says nothing when
/// it does.
fn summarise(table: &[Vec<Record>]) {
    let baseline = table[0][0].per_allocation();
    let find = |label: &str| {
        let row = CONFIGURATIONS
            .iter()
            .position(|configuration| configuration.label == label)
            .unwrap_or_else(|| panic!("no configuration is labelled `{label}`"));
        table[row][0].per_allocation() - baseline
    };

    let heapscope = find("heapscope, 10 frames");
    let dhat = find("dhat-rs, default (10)");
    println!(
        "\nAt one thread and a ten-frame capture, heapscope adds {heapscope:.0} ns per allocation\n\
         and dhat-rs adds {dhat:.0} ns, a factor of {:.0}. The `dhat-rs, 5 frames` row is what\n\
         says where that goes: {:.0} ns of it is five frames of capture.",
        dhat / heapscope,
        dhat - find("dhat-rs, 5 frames"),
    );
}

/// Bytes, in whichever unit keeps the number readable.
fn bytes(count: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    let count = count as f64;
    if count >= MIB {
        format!("{:.1} MiB", count / MIB)
    } else {
        format!("{:.1} KiB", count / KIB)
    }
}

fn milliseconds(nanoseconds: u128) -> String {
    if nanoseconds == 0 {
        return "-".to_string();
    }
    format!("{:.1} ms", nanoseconds as f64 / 1_000_000.0)
}

/// The `dhat-rs` version the comparison was run against.
///
/// Read from `Cargo.lock` rather than written down here, because a version in a
/// comment is a version that stops being true at the next `cargo update` and
/// says nothing when it does.
fn dhat_version() -> String {
    let lock = std::fs::read_to_string(manifest_directory().join("Cargo.lock"))
        .expect("Cargo.lock is beside Cargo.toml");
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line.trim() == "name = \"dhat\"" {
            if let Some(version) = lines
                .next()
                .and_then(|l| l.trim().strip_prefix("version = "))
            {
                return version.trim_matches('"').to_string();
            }
        }
    }
    panic!("Cargo.lock does not name the `dhat` package, but the fixture depends on it");
}

/// The workload's per-thread allocation count, reported by the cheapest fixture.
///
/// Asked of the fixture rather than duplicated here: the constant lives in
/// `examples/overhead/workload.rs`, and a second copy is a second thing to get
/// wrong.
fn allocations_per_thread(scratch: &tempfile::TempDir) -> usize {
    let record = run_fixture(&CONFIGURATIONS[0], 1, &scratch.path().join("probe.json"));
    record.allocations
}

// --- Locating the fixtures ------------------------------------------------

fn manifest_directory() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Path to a compiled fixture, checked for existence and for freshness.
///
/// Searched for rather than computed; `tests/support/fixture.rs` says why.
fn fixture_binary(name: &str) -> PathBuf {
    let path = fixture::example_binary(
        name,
        "`cargo bench --bench overhead` builds this driver and not the examples. Run\n\
         \x20   cargo build --release --examples\n\
         first (on x86_64, with RUSTFLAGS=\"-C force-frame-pointers=yes\").",
    );
    assert_fresh(name, &path);
    path
}

/// Refuses a fixture built from older source than this checkout.
///
/// `tests/lifecycle.rs` carries the long version of why, having been made to
/// pass by a stale binary once already. The short version: nothing rebuilds the
/// examples when this driver is rebuilt, so a published number would describe
/// whichever library the fixtures were last linked against.
///
/// Neither `Cargo.toml` nor `Cargo.lock` is among the inputs, for the reason
/// that file spells out at length: Cargo tracks `.rs` files by modification time
/// and tracks those two by content, so their clocks moving is not evidence that
/// anything needs rebuilding — and a guard that demands a rebuild Cargo will
/// decline to perform cannot be satisfied by any command.
fn assert_fresh(name: &str, fixture: &Path) {
    let built = modified(fixture).expect("the fixture has a modification time");

    let manifest = manifest_directory();
    let mut newest = SystemTime::UNIX_EPOCH;
    let mut newest_path = PathBuf::new();
    // One resolution change this cannot see: `overhead_dhat` links `dhat`, so a
    // new version of it changes that fixture without changing a `.rs` file here.
    // Cargo rebuilds the fixture for that, which is why the remedy below is the
    // whole answer, and why watching `Cargo.lock` for it would only have added
    // false refusals — a `cargo update` of `proptest` moves that clock too.
    for source in sources(&manifest.join("src")).into_iter().chain([
        manifest.join("examples").join(format!("{name}.rs")),
        manifest.join("examples/overhead/workload.rs"),
    ]) {
        if let Some(time) = modified(&source) {
            if time > newest {
                newest = time;
                newest_path = source;
            }
        }
    }

    assert!(
        built >= newest,
        "the fixture {} is older than {}, so this run would report on the previous \
         build.\nRun `cargo build --release --examples`.",
        fixture.display(),
        newest_path.display(),
    );
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

/// Every `.rs` file under `directory`, recursively.
fn sources(directory: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", directory.display()));
    for entry in entries {
        let path = entry.expect("a readable directory entry").path();
        if path.is_dir() {
            found.extend(sources(&path));
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            found.push(path);
        }
    }
    found
}
