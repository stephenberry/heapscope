//! The heapscope fixture for `benches/overhead.rs`.
//!
//! Configured to match `examples/overhead_dhat.rs` everywhere the two tools can
//! be made to agree: heap mode, one DHAT version 2 profile written at
//! shutdown, top-and-bottom frame trimming on, and a capture depth the driver
//! chooses. Where they cannot agree the difference belongs in the number, not
//! in the setup.
//!
//! Usage: `overhead_heapscope <threads> <frames|default> <output-path>`.

use std::time::Instant;

#[allow(dead_code)]
#[path = "overhead/workload.rs"]
mod workload;

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = workload::arguments();

    let mut builder = heapscope::Profiler::builder()
        .mode(heapscope::Mode::Heap)
        .output(heapscope::Output::dhat_v2(&arguments.output));
    if let Some(frames) = arguments.frames {
        builder = builder.max_depth(frames);
    }
    if let workload::Unwinder::System = arguments.unwinder {
        // The escape hatch for a build that cannot supply frame pointers, and
        // the same `_Unwind_Backtrace` that `dhat-rs` reaches through
        // `backtrace-rs`. Measured so that the table can separate what the
        // unwinder costs from what the engine behind it costs, which is the
        // only way to read the `dhat-rs` row as a statement about either.
        builder = builder.unwinder(heapscope::unwind::Strategy::System);
    }
    if let Some(interval) = arguments.sampling {
        builder = builder.sampling(interval);
    }
    // Built before the timer starts: this is where the frame-pointer capability
    // probe runs, and a one-off startup cost charged to the per-allocation
    // figure would be a cost the figure does not describe.
    let profiler = builder.build()?;

    let run = workload::run(arguments.threads);

    // Read before the profiler is dropped, and after the timer: two atomic
    // loads, but they are two the measurement should not carry.
    let blocks = profiler.stats().total_blocks;

    let shutdown = Instant::now();
    drop(profiler);
    let shutdown = shutdown.elapsed();

    run.report("heapscope");
    println!("shutdown-ns={}", shutdown.as_nanos());
    println!("total-blocks={blocks}");
    println!(
        "profile-bytes={}",
        std::fs::metadata(&arguments.output)?.len()
    );
    Ok(())
}
