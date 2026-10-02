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
use heapscope::internals::arena::Arena;
use heapscope::internals::live::{LiveBlock, LiveBlocks};
use heapscope::internals::lock::RawLock;
use heapscope::internals::pp::PpId;

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

/// One free and one allocation against the live-block table, at a steady
/// population: the pair a program that churns pays for every allocation.
///
/// Addresses are recycled from a ring twice the live population, as an
/// allocator recycles freed blocks, so the table settles at a fixed size and a
/// run of any length measures the same thing. With addresses that never
/// repeat, a table that leaked capacity on removal would reach its ceiling
/// partway through and start refusing inserts, which is cheaper than
/// accepting them and would read as a speed-up.
fn live_table_churn(c: &mut Criterion) {
    const LIVE: usize = 4_096;
    const RING: usize = 2 * LIVE;
    let address = |i: usize| 0x6000_0000_0000 + (i % RING) * 48;
    let block = LiveBlock::unattributed(0, PpId::OVERFLOW);

    let mut group = c.benchmark_group("live_table");
    group.measurement_time(Duration::from_secs(3));

    let arena = Arena::new();
    let table = LiveBlocks::new();
    for i in 0..LIVE {
        assert!(table.insert(&arena, address(i), block));
    }
    let mut next = LIVE;
    group.bench_function("remove_then_insert_steady", |b| {
        b.iter(|| {
            black_box(table.remove(address(next - LIVE)));
            black_box(table.insert(&arena, address(next), block));
            next += 1;
        });
    });
    group.finish();
}

criterion_group!(benches, uncontended_lock, live_table_churn);
criterion_main!(benches);
