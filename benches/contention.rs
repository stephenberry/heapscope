//! Peak-gate contention: the measurement PLAN.md's risk table asks for first.
//!
//! The gate is where the design deliberately gives up sharding for correctness
//! (PLAN.md section 4.3). Every allocation event acquires it, so if it does not
//! scale, nothing else about the engine matters. The risk table names the exact
//! shape to measure:
//!
//! > Peak-gate shared acquire becomes the bottleneck under high thread counts.
//! > Measured in M1, before the architecture sets. Warmup (all-peaks) is the
//! > worst case and is the specific thing benchmarked.
//!
//! That worst case is easy to overlook. During a **monotonically growing** phase
//! every allocation sets a new peak, so every allocation takes the gate
//! *exclusively* — and the gate degenerates to a global mutex. Steady state,
//! where the heap oscillates below its high-water mark, takes the shared path
//! instead. Benchmarking only the latter would publish a number that does not
//! describe program startup, which is when heap profilers are most often run.
//!
//! # Why this is not a criterion benchmark
//!
//! Criterion measures the latency of a single-threaded iteration. The quantity
//! here is aggregate throughput across threads under contention, which is a
//! different experiment: threads must start together, run for a fixed count,
//! and be measured as a whole. It also allocates on its own measurement path,
//! which would flow through the engine under test.
//!
//! # Result, aarch64-apple-darwin, 10 cores
//!
//! Nanoseconds per recorded event:
//!
//! | pattern | 1t | 2t | 4t | 8t | 16t |
//! |---|---|---|---|---|---|
//! | monotonic growth (all exclusive) | 74 | 265 | 300 | 250 | 275 |
//! | steady state (all shared) | 27 | 125 | 270 | 512 | 553 |
//! | free-heavy (all shared) | 34 | 88 | 222 | 467 | 500 |
//!
//! Measured on a machine carrying a load average of ~9 on 10 cores, so the
//! absolute values are pessimistic; three consecutive runs agreed to within a
//! few percent. An earlier measurement on an idle machine put steady state at
//! 30/91/191/209/276 — better in absolute terms, with the same shape. The
//! **scaling** is the durable finding and it does not depend on machine load.
//!
//! **Those figures were taken one sample per cell, which this benchmark no
//! longer does, and they stand only as the shape.** The scaling they show is
//! real and reproduces; the individual numbers are not reproducible to better
//! than a factor of two on a busy machine, and finding that out is what
//! prompted the sampling below. They will be replaced by a run of the sampled
//! instrument taken on a quiet machine, and the M6 change described below needs
//! that run before it means anything.
//!
//! **The gate does not scale, and the shared path does not either.** Per-event
//! cost rises about 4x on the exclusive path and **20x** on the shared one
//! between one thread and sixteen — aggregate throughput therefore *falls* as
//! cores are added. The risk PLAN.md section 11 lists has materialised, and M1
//! is exactly when it was supposed to be found.
//!
//! The cause is not the gate's exclusion semantics. Even on the shared path an
//! event writes several globally shared words, and the cost is in the cache
//! lines carrying them rather than in the number of atomic operations.
//!
//! **This paragraph used to say something more specific and it was wrong.** It
//! counted "five globally contended atomics — the gate word, `curr_bytes`,
//! `curr_blocks`, `total_bytes`, `total_blocks` — each of which is a single
//! cache line", and proposed removing two of them as an M6 change with the
//! measurement already in hand. Neither half survived M6 chunk B.
//!
//! Five words are not five lines. Measured on aarch64-apple-darwin, where
//! `sysctl hw.cachelinesize` reports **128** bytes, `std::mem::offset_of!` puts
//! them on two:
//!
//! | word | offset | 128-byte line |
//! |---|---|---|
//! | `gate` | 16,640 | 130 |
//! | `curr_bytes` | 16,752 | 130 |
//! | `curr_blocks` | 16,760 | 130 |
//! | `max_bytes` | 16,768 | 131 |
//! | `total_bytes` | 16,784 | 131 |
//! | `total_blocks` | 16,792 | 131 |
//! | `epoch` | 16,808 | 131 |
//!
//! So an event takes two lines, and taking `total_bytes` and `total_blocks` off
//! the second one cannot stop it being taken: `max_bytes` is read there on every
//! event and `epoch` on every peak. Deleting both counters outright — which
//! bounds any sharding win from above, sharding being strictly more work than
//! deleting — moved nothing beyond the noise. See PLAN.md section 9.1.
//!
//! The remaining contention is not a defect but a price, and it is line 130's.
//! Detecting an *exact* global peak requires a globally consistent running
//! total, so one exclusively-held cache line per allocation is inherent to
//! PLAN.md decision 10.1 — correctness over throughput. What this benchmark
//! supplies is the number that decision costs, rather than an assurance that it
//! is small.
//!
//! Run with: `cargo bench --bench contention`

use std::sync::Barrier;
use std::time::Instant;

use heapscope::internals::clock::TimeSource;
use heapscope::internals::engine::Engine;
use heapscope::internals::shape::Shape;

// Why a benchmark has to recognise `cargo test` at all is in the file.
#[path = "support/harness.rs"]
mod harness;
use harness::run_as_a_test;

/// Events per thread. Large enough that thread startup is negligible.
const EVENTS: usize = 200_000;

/// Which access pattern a run exercises.
#[derive(Clone, Copy)]
enum Pattern {
    /// Every allocation sets a new peak, so every one takes the exclusive path.
    /// The worst case, and what a growing program's startup looks like.
    MonotonicGrowth,
    /// The heap oscillates well below its high-water mark, so every event takes
    /// the shared path. What steady-state operation looks like.
    SteadyState,
    /// Frees only, which always take the shared path.
    FreeHeavy,
}

impl Pattern {
    fn name(self) -> &'static str {
        match self {
            Pattern::MonotonicGrowth => "monotonic growth (all exclusive)",
            Pattern::SteadyState => "steady state (all shared)",
            Pattern::FreeHeavy => "free-heavy (all shared)",
        }
    }
}

/// Runs `threads` threads through `EVENTS` events each and returns nanoseconds
/// per event.
///
/// Setup — establishing a high-water mark, or pre-allocating blocks to free —
/// happens before the timer starts, so the number describes the access pattern
/// rather than the scaffolding around it.
fn measure(pattern: Pattern, threads: usize, engine: &Engine) -> f64 {
    // `record_alloc` requires proof that the caller holds the reentrancy guard,
    // because it reaches the peak gate. Taken once per thread rather than per
    // call: the shim takes it per call, but that is a hash and a compare on a
    // line the thread owns, and putting it inside the loop would fold a
    // measurement of the guard into a measurement of the gate.
    let guard =
        heapscope::internals::guard::enter().expect("this thread is not inside the profiler");
    if matches!(pattern, Pattern::SteadyState | Pattern::FreeHeavy) {
        let base = 0xF000_0000_0000usize;
        for i in 0..4096 {
            engine.record_alloc(&guard, base + i * 64, Shape::of(1 << 20), &[0xDEAD]);
        }
        for i in 0..4096 {
            engine.record_free(base + i * 64, 1 << 20);
        }
    }
    if matches!(pattern, Pattern::FreeHeavy) {
        for t in 0..threads {
            let base = 0x1_0000_0000usize + t * 0x1_0000_0000;
            for i in 0..EVENTS {
                engine.record_alloc(&guard, base + i * 64, Shape::of(64), &[t]);
            }
        }
    }

    let start_line = Barrier::new(threads);
    let started = std::sync::Mutex::new(None::<Instant>);

    std::thread::scope(|scope| {
        for t in 0..threads {
            let (start_line, started) = (&start_line, &started);
            scope.spawn(move || {
                // `Guard` is `!Send`, so each worker takes its own.
                let guard = heapscope::internals::guard::enter()
                    .expect("a worker thread is not inside the profiler");
                let base = 0x1_0000_0000usize + t * 0x1_0000_0000;
                // Every thread arrives, then one records the start instant. The
                // barrier is what makes the threads actually contend rather
                // than run one after another.
                if start_line.wait().is_leader() {
                    *started.lock().unwrap() = Some(Instant::now());
                }
                match pattern {
                    Pattern::MonotonicGrowth => {
                        for i in 0..EVENTS {
                            engine.record_alloc(&guard, base + i * 64, Shape::of(64), &[t]);
                        }
                    }
                    Pattern::SteadyState => {
                        for i in 0..EVENTS / 2 {
                            let address = base + i * 64;
                            engine.record_alloc(&guard, address, Shape::of(64), &[t]);
                            engine.record_free(address, 64);
                        }
                    }
                    Pattern::FreeHeavy => {
                        for i in 0..EVENTS {
                            engine.record_free(base + i * 64, 64);
                        }
                    }
                }
            });
        }
    });

    let elapsed = started
        .lock()
        .unwrap()
        .expect("the barrier leader should have recorded a start instant")
        .elapsed();

    // Steady state performs an allocation and a free per iteration, over half
    // as many iterations, so every pattern records exactly `EVENTS` events per
    // thread and the numbers are directly comparable.
    let events = threads * EVENTS;
    elapsed.as_nanos() as f64 / events as f64
}

/// Runs one cell repeatedly and returns the fastest, with the spread across
/// samples.
///
/// One sample per cell is what this benchmark used to take, and on a loaded
/// machine that is not enough to answer the question it exists for. Removing
/// the two monotonic counters from the allocation path as an experiment moved
/// the **free-heavy** row by up to 77% — a row that path cannot touch, because
/// a free contributes nothing to either counter. The control moving that far is
/// the measurement saying it cannot tell the difference.
///
/// The minimum rather than the mean, for the reason every sampled benchmark
/// takes the minimum: interference makes a run slower and nothing makes it
/// faster. The spread comes back alongside so a cell that was not sampled
/// enough says so rather than looking like a figure.
fn fastest_of(pattern: Pattern, threads: usize) -> (f64, f64) {
    /// Samples per cell, at least.
    const MIN_RUNS: usize = 5;
    /// Samples per cell, at most.
    const MAX_RUNS: usize = 60;
    /// Wall clock a cell may spend before it stops starting new samples. The
    /// sixteen-thread cells cost about a second each and set this.
    const BUDGET: std::time::Duration = std::time::Duration::from_secs(6);

    let mut fastest = f64::INFINITY;
    let mut slowest: f64 = 0.0;
    let mut runs = 0;

    let started = Instant::now();
    while runs < MAX_RUNS && (runs < MIN_RUNS || started.elapsed() < BUDGET) {
        // A fresh engine per sample: a reused one carries a high-water mark and
        // a populated live table into the next measurement.
        let engine = Engine::with_limits(1 << 12, 1 << 22);
        assert!(engine.start(TimeSource::Events, || {}));
        let sample = measure(pattern, threads, &engine);
        fastest = fastest.min(sample);
        slowest = slowest.max(sample);
        runs += 1;
    }

    (fastest, (slowest - fastest) / fastest * 100.0)
}

fn main() {
    if run_as_a_test() {
        return;
    }

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("peak-gate contention, {cores} cores available");
    println!("nanoseconds per recorded event, lower is better\n");

    let thread_counts: Vec<usize> = [1usize, 2, 4, 8, 16]
        .into_iter()
        .filter(|&n| n <= cores.max(1) * 2)
        .collect();

    print!("{:<34}", "pattern");
    for threads in &thread_counts {
        print!("{:>10}", format!("{threads}t"));
    }
    println!();
    println!("{}", "-".repeat(34 + 10 * thread_counts.len()));

    let mut spreads = Vec::new();
    for pattern in [
        Pattern::MonotonicGrowth,
        Pattern::SteadyState,
        Pattern::FreeHeavy,
    ] {
        print!("{:<34}", pattern.name());
        for &threads in &thread_counts {
            let (fastest, spread) = fastest_of(pattern, threads);
            print!("{fastest:>10.1}");
            spreads.push(format!("{spread:.0}%"));
        }
        println!();
    }

    println!(
        "\nSpread across samples, row by row: {}.",
        spreads.join(" ")
    );
    println!(
        "\nA cost that stays flat as threads are added means the gate scales.\n\
         One that grows in proportion to the thread count means it has become a\n\
         global mutex, which is the risk PLAN.md section 11 asks to measure here.\n\
         \n\
         The free-heavy row is also a control. A free contributes nothing to the\n\
         two monotonic counters, so a change confined to the allocation path must\n\
         leave that row where it was; a run where it moves is a run that measured\n\
         the machine rather than the change."
    );
}
