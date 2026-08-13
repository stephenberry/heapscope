//! Micro-benchmarks for the primitives on the allocator hot path.
//!
//! # What belongs here, and what does not
//!
//! This file benchmarks **pure functions** — things that can be called without
//! a `#[global_allocator]` installed. Criterion is a good fit for those: it
//! handles warmup, outlier rejection, and per-iteration statistics.
//!
//! Two kinds of measurement deliberately live elsewhere, in a hand-written
//! std-only harness:
//!
//! - **Anything measured with the profiler's own shim installed.** Criterion
//!   allocates on its own measurement path — sample vectors, formatting,
//!   analysis — so those allocations would flow through the very shim under
//!   test and inflate the result. A benchmark that measures itself is not a
//!   benchmark.
//! - **Multi-threaded contention**, such as the peak gate under a monotonically
//!   growing heap. Criterion measures the latency of a single-threaded
//!   iteration; the quantity of interest there is aggregate throughput across
//!   threads, which is a different experiment with a different harness.
//!
//! Keeping the split explicit stops a misleading number from being published
//! later on the strength of "well, criterion said so".

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use heapscope::internals::lock::RawLock;

/// Uncontended acquire/release.
///
/// This is the figure that matters for the shard locks: under sharding, the
/// overwhelmingly common case is a lock nobody else wants. It sets the floor
/// for what a shard update can cost.
fn uncontended_lock(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_lock");
    group.measurement_time(Duration::from_secs(3));

    let lock = RawLock::new();
    // Warm up outside the measurement: the first acquire on some platforms
    // touches a page that later acquires do not.
    drop(lock.lock());

    group.bench_function("lock_unlock_uncontended", |b| {
        b.iter(|| {
            let guard = lock.lock();
            black_box(&guard);
        });
    });

    group.bench_function("try_lock_uncontended", |b| {
        b.iter(|| {
            let guard = lock.try_lock();
            black_box(&guard);
        });
    });

    group.finish();
}

criterion_group!(benches, uncontended_lock);
criterion_main!(benches);
