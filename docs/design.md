# Design decisions

The rules this crate holds itself to, what they cost, and the things it will never do.

## No dependencies

`[dependencies]` is empty, and `tests/no_dependencies.rs` fails if it stops being empty. Dev-dependencies are permitted and Cargo never builds them for you, so what reaches your build is this crate and `std`.

The reason is what a `#[global_allocator]` is. It links into every binary that uses it, it is live before `main`, and it runs on the one path in the process that must not allocate through itself. Every crate in its dependency tree inherits those properties whether or not its author knew about them. A dependency also brings version resolution into a build that currently has none, and it puts this crate's MSRV floor in somebody else's hands. None of that buys a profiler anything a profiler needs.

The rule is cheap almost everywhere. Platform capabilities are reached with `extern "C"` declarations against libraries the process already links, which is how `std` reaches them too.

### Where it is expensive: the demangler

`src/symbol/demangle` reimplements both Rust manglings, a spec this project does not own and that moves with the compiler. `rustc-demangle` exists, is itself dependency-free, and is maintained by the Rust project. Not depending on it is the single largest cost the zero-dependency rule imposes here, so it needs a reason better than tidiness.

The reason is that **a demangler in a profiler runs on adversarial input as a matter of routine, and this one needs a refusal policy that `rustc-demangle` does not offer.** Symbol tables get stripped, mismatched against the binary that loaded them, and corrupted. What is needed on top of "produce the right name for a valid symbol":

- **A work budget.** Backreferences can make `n` bytes describe a tree with `2^n` nodes. Bounded output size is not enough: punycode decoding inserts each character into the middle of what it has decoded, so its cost is quadratic in identifier length, and an identifier was admissible at a size where decoding takes about a second — inside a profiler's shutdown. It is charged quadratically now.
- **Bytes refused rather than passed through.** v0 copies identifier bytes into the name verbatim, so one stray byte in a stripped symbol table became a control character in whatever rendered the report. Neither mangling can legitimately carry a non-ASCII byte, so those are refused before parsing, and a finished name holding a control character is refused rather than returned. That is only the demangler's half, deliberately: a right-to-left override can still arrive through punycode, which is the mangling's own way of carrying a non-ASCII identifier, and image paths and `argv` never go near a demangler at all. The guarantee that nothing displayed can repaint a terminal or reverse its own display order therefore belongs to `push_display`, at the point a string becomes output. It is also the one item in this list that depending on `rustc-demangle` would not have cost.
- **A refusal that is visible.** A suffix that cannot be shown is a refusal rather than a silent omission, because omitting it renders two different pieces of code under one name — exactly the collision the suffix exists to prevent.

Fuzzing found six defects, every one needing an input no compiler emits, which is why 26,323 real symbols had missed them. That is the evidence for the argument: the input this runs on is not the input a compiler produces.

**What makes the reimplementation defensible is that the reference is a dev-dependency and gets run.** Reimplementing something to avoid depending on it is a claim about agreement, and agreement is checkable. `tests/demangle.rs` checks it against `rustc-demangle` on symbols that occurred — over 200,000 of them, taken with `nm` from this crate's own test binaries and from the `rustlib` archives of two toolchains — and `tests/demangle_fuzz.rs` checks it on symbols that could. There are zero divergences where the reference produces a name, and 52 where this produces one and the reference refuses, all Mach-O thread-local initialisers; a test asserts that this is the *only* divergence, so it stays deliberate rather than becoming a drift nobody noticed.

The module contains no `unsafe` at all, which is also checked rather than assumed — it is what lets the Miri job skip its corpus walks and finish in 15 minutes instead of 37.

## Non-goals

These are permanent decisions, not gaps waiting to be filled.

### musl / Alpine

Never supported, never detected, never worked around. The three capabilities that fail simultaneously under `crt-static` are listed in [platforms](platforms.md#musl--alpine-will-never-be-supported).

### Memory access counting

Valgrind's DHAT reports per-block read and write counts (`rb`, `wb`, `acc`). Producing those requires instrumenting every load and store, which needs a dynamic binary translator. This library emits `bkacc: false` and the viewer hides those columns. We would rather report nothing than report an approximation.

### Copy profiling

Attributing `memcpy`/`strcpy` costs automatically has the same requirement and the same answer. `Mode::Copy` and `heapscope::copied(bytes)` are the explicit opt-in instrumentation provided instead: they count what the program says it copied, which is a narrower measurement than Valgrind's and a very much cheaper one.

## Deliberate divergence from Valgrind

The per-program-point `mb`/`mbk` ("Max") values differ from Valgrind's. Valgrind samples a program point's maximum only at moments when the *whole heap* is at its peak, so a site that peaked at 4 MB while the global heap was small can record a maximum of zero. This library computes a true per-program-point running maximum. See [dev/PLAN.md](https://github.com/stephenberry/heapscope/blob/main/dev/PLAN.md) §3.3.
