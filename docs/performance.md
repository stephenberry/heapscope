# Performance

What profiling costs, measured rather than asserted, and what to do when it costs too much.

## What the profiler cost

The development plan promises honestly measured overhead. Every profile carries the measurement behind the per-capture half of that, so it is checkable by the reader rather than by us — the per-allocation figures further down come from the benchmark, which you can run:

```text
heapscope overhead
  memory     1.9 MiB held, 1.6 MiB in use
  tables     8 of 1,048,576 program points, 4,004 of 4,194,304 live blocks
  captures   26,018 walks at 10.99 ns each = 285.8 µs of stack walking
```

The per-capture figure is timed at startup, on the machine and in the build that is about to run, and the capture count is exact — so their product is this run's stack-walking time rather than an estimate. It covers the stack walk and nothing else: not interning, not the peak gate. It is timed once rather than per capture because reading the clock costs about as much as a frame-pointer walk, so timing each one would triple what it was measuring and then report the tripled figure.

## Paying less, on purpose

`.sampling(bytes)` records a sample of allocations instead of all of them, which is worth about a five-fold cut in overhead:

```rust
let profiler = heapscope::Profiler::builder()
    .sampling(128 * 1024)   // mean bytes between sample points
    .build()?;
```

Sample points fall on the stream of allocated *bytes*, not on the sequence of allocations, so an allocation of `s` bytes is caught with probability `1 - exp(-s / interval)`. A 100 MiB buffer is therefore caught however large the interval is, and a sampled allocation is scaled by the reciprocal of its own probability rather than by one global factor — the large allocations a profile exists to find are not the ones sampling drops. The same scale applies to block lifetimes, so the average-lifetime column still means what it did.

Two things to know before turning it on. **Every figure becomes an estimate, including the peak**, so [`HeapStats::get`](testing.md) refuses a sampled run rather than let an assertion compare a budget against a draw from a distribution. And **the interval to pick is the one that gives you enough sample points**, not a number of bytes: divide what your program allocates in total by the interval, and aim for a thousand or more.

## Measured so far

### End to end, against `dhat-rs`

`dhat-rs` is the reference implementation of the format this crate emits, so it is the comparison that counts. Both tools capturing ten frames, in heap mode, writing one DHAT file, over the same workload at 250,000 allocations per thread. Nanoseconds per allocation:

| | 1 thread | 4 threads | peak RSS | writing the profile |
|---|---|---|---|---|
| no profiler | 31.9 | 15.0 | 4.8 MiB | |
| **heapscope** | **129.4** | 300.1 | 7.1 MiB | 1.0 ms |
| heapscope, sampled at 128 KiB | 51.0 | 69.8 | 7.0 MiB | 1.2 ms |
| **`dhat-rs` 0.3.3** | **8,353.6** | 8,758.2 | 11.3 MiB | 5.8 ms |

Profiling adds about 97 ns per allocation here against `dhat-rs`'s 8,300, a factor of roughly 85, and 2.3 MiB of resident memory against 6.5. Most of the difference is the unwinder rather than the engine: `dhat-rs` captures through `backtrace-rs`, which on unix is `_Unwind_Backtrace`, and measuring it again at five frames puts that at about 490 ns per frame. Taken on aarch64-apple-darwin under a load average of about ten, so treat the ratio as the finding and the absolutes as pessimistic.

**Four threads is where heapscope is weakest, and the table is arranged so you can see it.** The unprofiled row falls from 31.9 to 15.0 ns as the work spreads over four threads, so the workload itself scales. heapscope's rises to 300.1, which means aggregate throughput *falls* as cores are added. That is the price of an exact global peak: detecting one needs a globally consistent running total, so there is a contended atomic on every allocation. It is a known cost with a measurement behind it rather than a surprise. Sharding those counters was tried and measured and did not pay; sampling is what actually reduces it, because an allocation that is not sampled never touches them.

**Sampling costs accuracy in a way worth choosing deliberately.** At 128 KiB it took 0.6% off the block count and 6% off the byte total on this workload, and the profile says so itself: the size histograms stay exact whether or not an allocation was sampled, so `observedBlocks` is the true count sitting next to `totalBlocks`, the estimate. What matters is the number of sample points rather than the interval — this workload allocates 174 MiB, which is about 1,330 points at 128 KiB. Below a few hundred the estimate degrades quickly and program points start disappearing from the profile altogether, and raising the interval past that point buys almost nothing: overhead floors at about 18 ns per allocation, which is the guard, the histograms, and the countdown, none of which the stack capture was hiding.

Run it yourself with `cargo build --release --examples && cargo bench --bench overhead`. It refuses to report anything if the three programs did not do identical work, if a captured stack did not reach the code that allocated, or if the fixtures are older than the sources they were built from.

### What the pieces cost

On aarch64-apple-darwin, against a baseline `malloc`/`free` of a 64-byte block at **16.7 ns**:

| | |
|---|---|
| Frame-pointer capture, fixed cost | ~5 ns |
| Frame-pointer capture, per frame | ~1.3 ns |
| Frame-pointer capture, 12 frames | **~21 ns** |
| `std::backtrace::Backtrace::force_capture` | ~18,800 ns |

Capturing a stack costs about as much as the allocation it records. The standard library's unwinder costs roughly 900 times as much, which is why it is never selected automatically on Linux or macOS.

The platform's own unwinder, which `Strategy::System` selects, sits between the two and differs by an order of magnitude across platforms (12 frames):

| | frame-pointer walk | platform unwinder |
|---|---|---|
| x86_64-unknown-linux-gnu | 51 ns | **5,613 ns** |
| aarch64-apple-darwin | 47 ns | **246 ns** |

The macOS figures were taken on a loaded machine — the baseline `malloc`/`free` measured 29 ns there against 16.7 ns idle — so read them as a ratio.

### Under contention

Recording an event — capture, intern, attribute, and update the peak — costs **27 ns single-threaded** and rises to ~550 ns at 16 threads, so aggregate throughput falls as cores are added. That is a known limitation, measured deliberately early rather than discovered late; its cause and the part of it that is removable are documented in `benches/contention.rs`.
