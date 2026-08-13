//! The `dhat-rs` fixture for `benches/overhead.rs`.
//!
//! The comparison this crate's design has to answer for. `dhat-rs` is the
//! reference implementation of the format heapscope emits and the tool a reader
//! would otherwise reach for, so an overhead claim that does not name it is not
//! a claim about anything.
//!
//! It is a dev-dependency, and running it is the point: PLAN.md section 1.2's
//! rule is that a comparison asserted is worth less than a comparison run. The
//! same rule put `rustc-demangle` in the dev-dependencies for `tests/demangle.rs`.
//!
//! Configured to match `examples/overhead_heapscope.rs`: heap mode, one DHAT
//! version 2 profile at shutdown, backtrace trimming on, and the driver's
//! capture depth. `trim_backtraces(None)` is *not* used — it is `dhat-rs`'s own
//! documented "much slower" setting, and benchmarking a tool in a mode its
//! documentation warns against would be picking the number rather than
//! measuring it.
//!
//! Usage: `overhead_dhat <threads> <frames|default> <output-path>`.

use std::time::Instant;

#[allow(dead_code)]
#[path = "overhead/workload.rs"]
mod workload;

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = workload::arguments();

    let mut builder = dhat::Profiler::builder().file_name(&arguments.output);
    if let Some(frames) = arguments.frames {
        builder = builder.trim_backtraces(Some(frames));
    }
    let profiler = builder.build();

    let run = workload::run(arguments.threads);

    // Takes `dhat-rs`'s global lock, so it is read after the timer for the same
    // reason heapscope's counters are.
    let blocks = dhat::HeapStats::get().total_blocks;

    let shutdown = Instant::now();
    drop(profiler);
    let shutdown = shutdown.elapsed();

    run.report("dhat");
    println!("shutdown-ns={}", shutdown.as_nanos());
    println!("total-blocks={blocks}");
    println!(
        "profile-bytes={}",
        std::fs::metadata(&arguments.output)?.len()
    );
    Ok(())
}
