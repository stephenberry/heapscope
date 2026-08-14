# Platforms and requirements

## Supported platforms

| Target | Frame pointers | Stack capture | Verification |
|---|---|---|---|
| `aarch64-apple-darwin` | on by default | frame-pointer walk | by execution |
| `aarch64-unknown-linux-gnu` | on by default | frame-pointer walk | by execution |
| `x86_64-unknown-linux-gnu` | `-C force-frame-pointers=yes` | frame-pointer walk | by execution |
| `x86_64-pc-windows-msvc` | not needed | `RtlCaptureStackBackTrace` | by execution |
| musl / Alpine | — | — | not supported, and never will be ([why](#musl--alpine-will-never-be-supported)) |

All four run the suite on every push, Windows natively rather than under Wine — which is what settles in-process symbolization there, since `SymFromAddr` cannot execute under an emulator. Two sanitizers and Miri run the suite as well, each sanitizer behind a positive control that fails the job if it cannot see a planted defect through this crate's `#[global_allocator]`.

What is still unproven is narrower and named: the cost of `RtlCaptureStackBackTrace` on Windows is unmeasured, no Windows browser has opened the bundled page, and `bias` on Windows is a relative virtual address rather than the address the file records — see [symbolization](symbolization.md#what-is-verified-and-what-is-not).

Why each platform captures stacks the way it does is in [stack capture](stack-capture.md).

## Requirements

`heapscope::Alloc` must be the program's `#[global_allocator]`, or a heap run refuses to start. Nothing reaches the engine without it, so the alternative is a profile of zeros that looks exactly like a well-behaved program. Ad hoc and copy runs do not need it: they count what the program reports and turn the shim off.

Frame pointers are required on x86_64 targets:

```
RUSTFLAGS="-C force-frame-pointers=yes"
CFLAGS="-fno-omit-frame-pointer"   # for C/C++ dependencies built via `cc`
```

They are enabled by default on aarch64 (Apple and Linux), where no flag is needed. The profiler fails at startup with this message rather than silently producing empty or 500×-slower profiles.

The minimum supported Rust version is **1.96**. It is checked against the shipped library alone: `ci/msrv-check.sh` builds a copy of the crate with every dev-only section removed, so a dev-dependency that raises its own floor cannot quietly raise the one promised to you. Raising it is a minor version bump.

## musl / Alpine will never be supported

Under `crt-static` — the musl default — three capabilities this profiler depends on fail simultaneously:

- `dladdr` is a non-functional stub, so in-process symbolization yields nothing.
- There is no libc `backtrace()`.
- The unwinder is statically bundled, so under `panic=abort` an `_Unwind_Backtrace` declaration can fail at *link* time rather than degrading at runtime.

Supporting musl would mean a separate symbolization path, a separate unwinding path, and a permanent CI matrix entry. **This library does not support musl targets and does not intend to add support in the future.** It does not detect musl or attempt to work around it. If you need heap profiling on Alpine, use a different tool.
