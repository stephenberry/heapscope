# heapscope

**Dynamic heap analysis for Rust.**

A heap profiler that tracks every allocation, attributes it to the call site that made it, and writes a profile you can open in a bundled single-file viewer or in [Valgrind's DHAT viewer](https://valgrind.org/docs/manual/dh-manual.html).

- **No dependencies outside the standard library.** `[dependencies]` is empty and stays empty, so what reaches your build is this crate and `std`. [Why, and what it costs](docs/design.md#no-dependencies).
- **Every allocation, by default.** [Sampling](docs/performance.md#paying-less-on-purpose) trades exactness for most of the cost, and the profile says which one it is.
- **About 97 ns per allocation**, against `dhat-rs`'s 8,300 on the same workload. [The measurements](docs/performance.md#measured-so-far).
- **Four outputs from one reading**, so a summary and a file can never disagree about the same run. [Formats](docs/output-formats.md).
- **Numbers a test can read**, so "this parser allocates at most 64 KiB" runs on every commit instead of being measured once and written in a comment. [Budgets and baselines](docs/testing.md).

## Install

```toml
[dependencies]
heapscope = "0.1"
```

Two requirements, both checked at startup rather than left to produce a confusing profile:

1. `heapscope::Alloc` must be the program's `#[global_allocator]` — the line in the example below.
2. **On x86_64, build with frame pointers.** aarch64 (Apple and Linux) and Windows need no flag.

```sh
RUSTFLAGS="-C force-frame-pointers=yes" cargo run --release
```

A run missing either one refuses to start and says which, rather than producing an empty or 500×-slower profile. [The requirements in full](docs/platforms.md#requirements).

## Quickstart

```rust
#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profiler = heapscope::Profiler::new()?;   // writes dhat-heap.json on drop
    // ... work ...
    profiler.print_summary(10)?;
    Ok(())
}
```

To see it on a real workload without writing one first:

```sh
cargo run --release --example profile_a_program
```

### Looking at the profile

`dhat-heap.json` is the file Valgrind's `dh_view.html` opens. Valgrind does not exist on Windows and does not support Apple Silicon, so ask for the bundled page instead — one file, double-click to open, nothing fetched:

```rust
let profiler = heapscope::Profiler::builder()
    .output(heapscope::Output::html("target/profile.html"))
    .build()?;
```

There are four formats and asking for several is nearly free: they all come from a single reading of the engine, so they cannot disagree about the same run. See [output formats](docs/output-formats.md).

## Supported platforms

| Target | Frame pointers | Stack capture |
|---|---|---|
| `aarch64-apple-darwin` | on by default | frame-pointer walk |
| `aarch64-unknown-linux-gnu` | on by default | frame-pointer walk |
| `x86_64-unknown-linux-gnu` | `-C force-frame-pointers=yes` | frame-pointer walk |
| `x86_64-pc-windows-msvc` | not needed | `RtlCaptureStackBackTrace` |

musl / Alpine is not supported and never will be. Everything documented here is verified by execution on the first three; Windows is built and run under Wine, which is not Windows. [What that leaves unproven](docs/platforms.md).

## Documentation

- [Configuration](docs/configuration.md) — the builder, and counting something other than allocations
- [Output formats](docs/output-formats.md) — DHAT, native, the bundled viewer, flame graphs
- [Reading a profile](docs/reading-a-profile.md) — sizes, reallocation, which thread and which phase
- [Failing a test on the numbers](docs/testing.md) — budgets and baselines that break a build
- [Symbolization](docs/symbolization.md) — resolving offline, and the `heapscope-symbolize` tool
- [Performance](docs/performance.md) — what profiling costs, measured, and how to pay less
- [Stack capture](docs/stack-capture.md) — which unwinder, why, and how much to trust the frames
- [When the profile gets written](docs/lifecycle.md) — exits, `fork`, signals
- [Platforms and requirements](docs/platforms.md) — where this is verified, and the MSRV
- [Design decisions](docs/design.md) — the zero-dependency rule, non-goals, and where we diverge from Valgrind

## Project status

Pre-1.0, and the version number is the promise: the public API may still change. The engine, the three emitters, the bundled viewer, sampling, and the test-time budgets are built and tested.

What stands between this and 1.0 is mostly evidence rather than features. Continuous integration has not yet completed a run, so the four-platform matrix is verified locally — natively on macOS aarch64, under Docker on both Linux targets, and under Wine for Windows — rather than on a runner. A native Windows run is the other gap.

`heapscope::internals` and `heapscope::unwind` are public but `#[doc(hidden)]`, and carry no stability promise: they exist because the benchmarks, the reference tracker and the property tests need to observe the engine. The supported surface is what the documentation shows, and it is written down and enforced by a test, so growing it is a decision rather than an accident.

[dev/PLAN.md](https://github.com/stephenberry/heapscope/blob/main/dev/PLAN.md) carries the design: the architecture, the platform facts it is shaped around, the rules the code holds itself to and what each cost to learn, and what remains unproven.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
