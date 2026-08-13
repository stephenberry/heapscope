# How stacks are captured

| Platform | Default | Why |
|---|---|---|
| Linux, macOS | frame-pointer walk | ~21 ns. Requires frame pointers on x86_64, and says so at startup rather than degrading silently. |
| Windows | `RtlCaptureStackBackTrace` | There is no frame-pointer chain to walk. |

Windows is not a fallback to something worse; it is the platform's own mechanism. The Microsoft x64 ABI mandates unwind tables (`.pdata`/`.xdata`) for every function, so Windows never needed a linked `rbp` chain, and `-C force-frame-pointers=yes` does not produce one you can follow. Measured under Wine on `x86_64-pc-windows-gnu`, from the same stack at the same instant: hand-walking `rbp` yielded 2 entries with the flag and 1 without, the second being a stack address rather than a return address; `RtlCaptureStackBackTrace` returned 9 plausible frames either way. The upshot is that Windows users need no build flags at all. **What this costs on Windows is unmeasured** — Wine timings say nothing about the real platform, and a table walk is certainly far more than a chain walk.

On Linux and macOS the platform unwinder is available as an explicit opt-in, `Profiler::builder().unwinder(Strategy::System)`, for a build that genuinely cannot supply frame pointers. It is never chosen for you there.

On macOS the platform unwinder is *also* a frame-pointer walk, in libSystem — so it is not an escape hatch from missing frame pointers there, and the startup probe refuses rather than pretending. It is a real answer on x86_64 Linux, where it reads the unwind tables instead.

[What each of these costs](performance.md#what-the-pieces-cost) is measured, and the gap between them is an order of magnitude.

## How much to trust the frames

Every profile records which unwinder produced its frames, in `heapscope.unwinder`, because the two disagree about how deep a trace goes and where it stops. Alongside it, `heapscope.captures` counts how many stack walks came back whole:

```json
"captures": {"complete": 26011, "truncated": 0, "suspect": 0, "noFrames": 0}
```

That matters because the startup probe walks *heapscope's* frames, which says nothing about C or C++ dependencies someone else compiled, hand-written assembly, or threads a C library created. A profile where `complete` does not dominate is one whose call sites should be read with suspicion.
