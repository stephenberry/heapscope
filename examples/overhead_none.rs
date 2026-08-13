//! The unprofiled fixture: the baseline `benches/overhead.rs` subtracts.
//!
//! Not an example of anything. It exists so that the cost of the workload
//! itself appears in the table as a row rather than as an assumption, because
//! "profiling costs 60 ns per allocation" is only meaningful next to what an
//! allocation costs when nobody is watching.
//!
//! `System` is named explicitly rather than left to the default. It is the same
//! allocator either way, and writing it down is what makes it visible that all
//! three fixtures bottom out in the same one — both profilers wrap `System`, so
//! the rows differ by the recording and by nothing underneath it.
//!
//! Usage: `overhead_none <threads> <frames|default> <output-path>`. The last two
//! are ignored and taken anyway, so that the driver invokes all three fixtures
//! identically.

use std::alloc::System;

// Each fixture reports what it alone can report, so each uses a different part
// of the shared module.
#[allow(dead_code)]
#[path = "overhead/workload.rs"]
mod workload;

#[global_allocator]
static ALLOC: System = System;

fn main() {
    let arguments = workload::arguments();
    let run = workload::run(arguments.threads);
    run.report("none");
    // No profiler, so nothing is written and nothing is torn down. Reported as
    // zero rather than omitted: the driver requires the key, and a fixture that
    // crashed before shutdown must not be indistinguishable from this one.
    println!("shutdown-ns=0");
}
