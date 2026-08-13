# heapscope documentation

The [README](../README.md) gets you a first profile. These pages are the detail behind it.

## Using it

- [Configuration](configuration.md) — the builder, and counting something other than allocations
- [Output formats](output-formats.md) — DHAT, native, the bundled viewer, flame graphs
- [Reading a profile](reading-a-profile.md) — sizes, reallocation, which thread and which phase
- [Failing a test on the numbers](testing.md) — budgets and baselines that break a build
- [Symbolization](symbolization.md) — resolving offline, and the `heapscope-symbolize` tool

## How it behaves

- [Performance](performance.md) — what profiling costs, measured, and how to pay less
- [Stack capture](stack-capture.md) — which unwinder, why, and how much to trust the frames
- [When the profile gets written](lifecycle.md) — exits, `fork`, signals
- [Platforms and requirements](platforms.md) — where this is verified, and the MSRV

## Why it is built this way

- [Design decisions](design.md) — the zero-dependency rule, non-goals, and where we diverge from Valgrind
- [dev/PLAN.md](https://github.com/stephenberry/heapscope/blob/main/dev/PLAN.md) — the architecture, the platform facts it is shaped around, and the rules the code holds itself to
