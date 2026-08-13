# Symbolization

The hot path stores return addresses and nothing else. Every profile carries a **module map** — the path, load address, extent, and build identity (Mach-O `LC_UUID`, ELF `NT_GNU_BUILD_ID`) of each loaded image — and renders frames as `image + offset`, with the name the running process knows the address by where there is one:

```text
0x1044c81f0: core::fmt::write+0x1c (/path/to/program+0x2c1f0)
0x1044c9330: ??? (/path/to/program+0x2d330)
```

The name is added in front of the image and offset rather than replacing them, so nothing is lost by having it. How many frames get one is a property of the platform, not of this library: Apple's loader reads an image's whole symbol table, while `dladdr` on ELF sees only what an image *exports*, which for an executable is almost nothing. Measured on the same program: 52 of 52 frames named on macOS aarch64, 0 of 70 on Linux aarch64. `-C link-args=-rdynamic` does not close that gap — it brings back the `std` and `alloc` generic instantiations and leaves your own crate's functions unnamed. On Linux, [resolve offline](#heapscope-symbolize). Rust names are demangled here, both the legacy scheme and v0.

Set `HEAPSCOPE_SYMBOLIZE=0` to leave the names out. Frames then render as address, image, and offset only. The reason it is an environment variable is that the reasons to want it tend to arrive at an inconvenient moment: on Windows, `dbghelp` honours `_NT_SYMBOL_PATH`, so if that names a symbol server, writing a profile at process exit can block on the network.

## Frames that are the same on every stack

Most of a captured stack is not about your program. Above it is the standard allocation path — `Vec` reaching `RawVec` reaching `__rust_alloc` — and below it is the runtime that started the thread. On the example program in this repository those are 93 of 144 frames, and a spawned thread's entry sequence alone is nine.

Those are left out by default. What survives is the call chain that decided to allocate:

```text
0x104ad41d0: <alloc::vec::Vec<u8>>::with_capacity+0x24 (…/profile_a_program+0x10002c1d0)
0x104aac134: profile_a_program::churn+0x90 (…/profile_a_program+0x100004134)
0x104aabed0: profile_a_program::main+0x174 (…/profile_a_program+0x100003ed0)
0x104aaca08: <fn() -> core::result::Result<(), alloc::boxed::Box<dyn core::error::Error>> as core::ops::function::FnOnce<()>>::call_once+0x14 (…/profile_a_program+0x100004a08)
```

The cut below is the one `std` makes for its own backtraces, at `__rust_begin_short_backtrace` — which is why the `FnOnce::call_once` shim just inside it stays, here as in a panic backtrace. The cut above stops at the first frame that is not on the allocation path, so a frame of yours is never removed because of what sits above it. Nothing is hidden silently: the count is in the profile as `heapscope.trimmedFrames` and in the text summary as a line of its own. To keep every frame, render with `Symbolized` explicitly:

```rust
snapshot.write_dhat_v2_with(file, &heapscope::symbol::Symbolized::new(&snapshot.modules))?;
```

Trimming reads frame *names*, so where symbolization finds none — a stripped build, or Linux — nothing is trimmed and the stack is left exactly as it was.

## `heapscope-symbolize`

The crate ships a binary that does the whole of it in one pass:

```sh
cargo install heapscope        # or: cargo build --bin heapscope-symbolize

heapscope-symbolize profile.native.json -o resolved.json
# heapscope-symbolize: atos resolved 51 of 51 addresses; 51 of 51 frames now named
```

It reads the **native** profile — a DHAT frame is a string, so a DHAT file has no addresses left to resolve — batches each image's file addresses through `atos`, `llvm-symbolizer`, or `addr2line`, whichever is installed, and writes the profile back with `function`, `file`, `line`, and `inlinedBy` added per frame.

Added, not substituted. `symbol` stays exactly as the running process reported it, so a reader can see when the two disagree, which is the symptom of resolving against the wrong binary. Everything else in the file — every counter, every histogram, the module map, and any field a future version adds — comes back byte for byte, because the format's own rule is that a reader ignores what it does not know and a rewriter has to preserve what it ignored.

Straight to a flame graph, with the names it just found:

```sh
heapscope-symbolize profile.native.json -f folded | inferno-flamegraph > heap.svg
```

That composes better than it looks. Frame trimming reads frame *names*, so on Linux, where nothing is named at record time, nothing is trimmed either and every stack keeps the nine frames of runtime entry and the allocation path above it. Resolving first is what lets the same cut happen: **measured on the example program, 12–17 recorded frames per stack become 4–8.**

For a profile recorded somewhere else, point it at the build:

```sh
heapscope-symbolize profile.native.json --binary /build/app=./archive/app-v1.2.3
```

which is what the recorded build identity is for. If it resolves nothing at all it says so and exits non-zero, rather than handing back a file that looks fine and names nothing.

It does not rewrite the bundled HTML page: that page renders from display names chosen when it was written, so resolved names show up in the JSON and in the folded output instead.

## Resolving offline is the primary path

In-process symbolization does not work on the binaries people ship: on a stripped image `dladdr` returns *success* with a null symbol name, and `strip = true` is common in release profiles. Resolving offline also means a profile recorded on one machine can be symbolized on another, against an archived build, a year later — which is why the build identity is recorded alongside the path.

Whether a frame is named or not, it stays resolvable afterwards by a tool that was not running when the profile was recorded. In a rendered frame, the second number is the address **as it appears in the file**, not an offset from where the image was mapped — those are different numbers on macOS, where file addresses start at 0x1_0000_0000, and on a non-PIE executable, where they start at 0x400000. It is what the ELF tools take directly:

```sh
llvm-symbolizer --obj=/path/to/program 0x10002c1f0
addr2line -f -C -e /path/to/program 0x10002c1f0
```

`atos` works from the runtime address instead, given the image's load address, which the profile's module map records alongside the path:

```sh
atos -o /path/to/program -l 0x1044a0000 0x1044c81f0
```

**On macOS, use `atos` for system libraries.** Almost everything under `/usr/lib` is mapped from the dyld shared cache rather than from the file at that path, and a cache image's segments are laid out at cache addresses. The recorded offset is therefore an address in the cache, not in the file — for `/usr/lib/dyld`, `0x1801344e4` where the file wants `0x204e4`, a difference of the in-cache `__TEXT` address — so `llvm-symbolizer` and `addr2line` resolve it to nothing even though the file exists and its UUID matches. `atos` works, because it takes the load address this crate records as `image_base` and asks the running system. Your own binaries and anything you built are unaffected.

## What is verified, and what is not

The module map is verified by execution on macOS (aarch64), Linux x86_64, Linux aarch64, and — under Wine, which is not the same as Windows — `x86_64-pc-windows-gnu`. The Windows module map is enumerated from `K32EnumProcessModules` and does not read the PDB signature, so images there carry no build identity, and it records each image's whole span rather than its executable sections.

**In-process symbolization on Windows has never been executed anywhere.** `SymFromAddr` cannot run under Wine — see `ci/windows-under-wine.sh` — so it is compiled, reviewed against Microsoft's documentation, and unproven until a native Windows run. Everything else in this documentation is verified by execution on the platforms listed above.
