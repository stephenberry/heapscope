# Development Plan: `heapscope`

A production-grade dynamic heap analysis library for Rust, published as **`heapscope`**. The shipped library has zero Cargo dependencies. MSRV 1.96.

Claims are tagged **[measured]** where a probe was run on this machine, **[source]** where verified against upstream source, and **[unverified]** where reasoned but untested. The last category is deliberately small and always called out.

Sections 1–8 are the design. §9.1 is the implementation record: what was built, what reviews found, and what it cost. Section numbers are cited from ~130 places in the source and CI, so they are stable.

---

## 1. Goals and non-goals

### Goals

1. **Correctness first.** Never corrupt the observed program, never deadlock, never report a number we cannot substantiate. Where a measurement is impossible, say so in the output rather than approximating silently.
2. **Robustness.** The explicit opposite of dhat-rs's self-description ("it may work fine for you, or it may crash, hang, or otherwise do the wrong thing"). No unbounded growth, no panics from the allocator, no reliance on TLS destructor ordering, defined behavior across `fork`, `exit`, panics, and unwinding.
3. **Performance.** Overhead dominated by stack unwinding, and unwinding is the fastest *correct* option on the platform.
4. **Simplicity.** A small core — allocator shim, live-block table, program-point table, serializer — with every subsystem behind a narrow trait.
5. **Fully featured.** Everything dhat-rs offers, plus deterministic time, sampling, phase markers, thread attribution, a built-in text report, and offline symbolization.
6. **Zero dependencies in the shipped library.** `std` only — see §1.2. Platform capabilities are reached via `extern "C"` declarations against libraries the process **already links** (the system unwinder, libdl, libpthread, kernel32/dbghelp). This adds no supply-chain surface, no version resolution, and no build-time cost. It is how `std` itself reaches the platform.

### Non-goals

- **Memory access counting** (DHAT's `rb`/`wb`/`acc`). Requires instrumenting every load and store, which needs a dynamic binary translator. We emit `bkacc: false`; the viewer hides those columns. Honesty is the feature.
- **Copy profiling** (`memcpy`/`strcpy` costs). Same reason. Explicit opt-in instrumentation instead (§6.10).
- **musl / Alpine.** Permanently out of scope — see §1.1. Stated in the README so nobody plans around it.
- Replacing Valgrind's DHAT for C/C++. This is a Rust-native tool that speaks DHAT's file format.

### 1.1 Why musl is a permanent non-goal

Under `crt-static` — the musl default — three capabilities fail at once:

- `dladdr` is a non-functional stub, so in-process symbolization yields nothing **[source: `library/unwind/src/lib.rs` links `dl` only for the non-`crt-static` case]**.
- There is no libc `backtrace()`.
- The unwinder is statically bundled, so under `panic=abort` an `extern "C" _Unwind_Backtrace` declaration can fail at **link** time rather than degrading at runtime.

Supporting this would mean a separate symbolization path, a separate unwinding path, and a CI matrix entry, to serve a target we do not deploy on. **Decision: not supported, not planned, ever.** The crate does not attempt to detect or work around it.

### 1.2 What "zero dependencies" means precisely

**`[dependencies]` is empty and stays empty.** Nothing reaches a downstream consumer's build.

**`[dev-dependencies]` are permitted** for tests and benchmarks. Cargo does not build these for downstream consumers, so they cost users nothing.

The one real hazard is **MSRV drift**: a dev-dependency raising its own MSRV would break CI through no action of ours. The mitigation is structural — the MSRV job runs on the **library and bins only** (`ci/msrv-check.sh` strips every dev-only section), the full test and bench jobs run on stable, and `Cargo.lock` is committed. A dev-dependency can therefore never silently raise the floor promised to users.

README wording must be precise: "zero dependencies" unqualified invites a well-actually issue in the first week, so it reads "no dependencies outside the standard library."

---

## 2. What dhat-rs does, and where we diverge

| Aspect | dhat-rs | This library |
|---|---|---|
| Dependencies | `backtrace`, `serde`, `serde_json`, `thousands`, `rustc-hash`, `mintex`, `lazy_static` | none shipped |
| Locking | one global `mintex::Mutex` around all state | sharded platform locks; hot path touches one shard |
| Backtraces | `backtrace` → `_Unwind_Backtrace`, **~10 µs/capture [measured]** | frame-pointer walk, **~18 ns/capture [measured]** |
| Symbolization | eager-ish, in-process only | deferred to output time; offline supported |
| Reentrancy guard | thread-local `Cell<bool>` | static slot table keyed by thread id, no TLS at all |
| Time base | wall-clock µs | deterministic event counter (default) or wall-clock µs |
| gmax snapshot | O(#PPs) sweep on every peak | O(1) amortized lazy epoch (§4.3) |
| Coverage | "only measures allocations inside `main`" | process lifetime |
| `fork` | undefined | `pthread_atfork` handlers |
| `alloc_zeroed` | not overridden — destroys `calloc`'s lazy-zero-page path | forwarded to the inner allocator |
| Failure mode | may crash or hang | poison-and-degrade; never panics from the hot path |
| Bounds | unbounded PP/frame growth | caps on PPs, frames, **and live blocks**, with overflow accounting |

The ~10 µs/capture figure is the headline: `std::backtrace` and the `backtrace` crate both route to `backtrace_rs::backtrace::libunwind::trace`, confirmed by capturing a stack from inside an allocator hook **[measured]**. Cutting that to ~18 ns is the single largest opportunity in this project, which is why the unwinder is an M1 concern (§9), not an M3 detail.

---

## 3. The output format (contract we must satisfy)

Target: **DHAT file format version 2**, consumed by Valgrind's `dh_view.html`. `kExpectedFileVersion` has been 2 since 2019 and there is no v3 **[source]**.

**Mandatory top level** **[source: `dh_view.js` `checkFields`]**:

```js
["dhatFileVersion", "mode", "verb", "bklt", "bkacc",
 "tu", "Mtu", "cmd", "pid", "te", "pps", "ftbl"]
```

**Additionally mandatory when `bklt` is true** — not optional: `if (gData.bklt) { checkFields(gData, ["tg", "tuth"]); }`

**Per program point:** `checkPP` validates `["tb","tbk","fs"]`, plus `["mb","mbk","gb","gbk","eb","ebk"]` when `bklt`, plus `["rb","wb"]` when `bkacc` (never, for us).

### 3.1 Three traps the viewer will not warn you about

1. **`tl` must be emitted but is never validated.** It appears in neither `checkPP` list, yet `dh_view.js` reads `aPP.tl` unguarded. Omit it and the file loads cleanly while every average-lifetime cell renders `NaN`. **Our validator must therefore be *stricter* than the viewer, not a reimplementation of it.**
2. **`fs` sequences must be unique across program points** — the viewer throws `data file contains a repeated location`. This collides with `max_depth` truncation (§5) and frame trimming (§6): both can collapse two distinct interned PPs onto the same emitted frame list, producing a file that **fails to open**. Requires the emit-time fold of §3.2.
3. **`ftbl[0]` must be `"[root]"`**, and `fs` arrays are innermost-first and must exclude index 0.

A fourth trap belongs with them, discovered when the native format arrived: **an address must be a hexadecimal *string*, never a JSON number.** A JSON number is a double in JavaScript, exact only to 2⁵³, so `JSON.parse` silently rounds a 64-bit address — and an address wrong in its low bits names the wrong line of the wrong function with nothing about it looking wrong.

### 3.2 Emit-time PP folding (required)

PPs are interned on the hot path by *raw* frame array, but emitted with trimmed and truncated frames. Between the two, re-key every PP by its **final** `fs` vector and fold collisions: sum `tb`, `tbk`, `tl`, `eb`, `ebk`, `gb`, `gbk`. Without it, trimming produces unopenable files.

**The merge of `mb` is not the max of the two maxima.** That can produce a program point with more bytes live at the global peak than it ever had live at all — a file whose own numbers contradict each other. t-gmax is a single instant, so *both* folded points held their `gb` bytes simultaneously and the merged point demonstrably reached `gb₁ + gb₂` at once. The correct merge is the largest of the three provable bounds:

```text
mb = max(mb₁, mb₂, gb₁ + gb₂, eb₁ + eb₂)
```

Still a lower bound on the truth — the real joint maximum is unknowable once two points are indistinguishable — but never a number that contradicts the rest of the record. Found by a property test asserting that every emitted profile passes the validator, not by reading the rule.

### 3.3 `mb`/`mbk` semantics — a deliberate divergence from Valgrind

Valgrind assigns `ppi->max_bytes` in exactly two places, **both inside `if (g_curr_bytes >= g_max_bytes)`** (`dh_main.c:382`, `:566`) **[source]**. A program point's "max" is therefore only ever sampled at moments when the *whole heap* is at its peak, despite the struct comment claiming otherwise. A site that peaked at 4 MB while the global heap was small can record a max of zero.

**Decision (§10.3): a true per-PP running max** — `pp.max = max(pp.max, pp.curr)` on every touch, O(1). Our numbers legitimately differ from Valgrind's for the same program; documented in the README and the output header. The reference tracker (§8, item 2) encodes *our* definition.

Low-risk in the viewer: `Max:` renders only for leaf nodes, carries no percentages, and has no sort metric — the viewer's own comment says *"not interesting, and unclear how to sort"* — precisely because max values are not summable across a tree.

### 3.4 Native format

A versioned superset (§6.8) is the source of truth. The DHAT v2 emitter is one lossy projection of it. This decouples us from upstream format churn and carries everything DHAT v2 has no field for.

---

## 4. Architecture

```
src/
  lib.rs              public API + crate docs
  profiler.rs         Profiler, ProfilerBuilder, lifecycle state machine
  alloc.rs            Alloc<A>: GlobalAlloc shim
  stats.rs            HeapStats / EventStats, testing assertions
  baseline.rs         recorded readings, and comparing a run against one
  event.rs            ad hoc and copy event counting
  region.rs           phase markers

  internals/          #[doc(hidden)]: public so tests can observe it, promised to nobody
    engine.rs         the recording engine: mode, state machine, counters
    gate.rs           the peak gate; the linearization point t-gmax needs (§4.3)
    lock.rs           RawLock over os_unfair_lock / futex / SRWLOCK; shard arrays
    order.rs          debug-build lock-order enforcement (§4.2)
    fork.rs           pthread_atfork handlers
    arena.rs          bump arena over std::alloc::System
    table.rs          open-addressing map living in the arena
    live.rs           the live-block table
    pp.rs             program points: interned call stacks and their counters
    site.rs           thread and region attribution rows (§6.5, §6.6)
    shape.rs          what the program asked for, beyond a number of bytes (§6.8)
    sampler.rs        Poisson sampling over the byte stream (§6.3)
    clock.rs          TimeSource: Events (default) | Monotonic
    guard.rs          reentrancy guard (no TLS; slot table keyed by thread id)
    diagnostic.rs     reporting failures from inside the allocator; the poison flag
    stack.rs          stack bounds for the calling thread

  unwind/             Strategy trait, capability probe, skip calibration, counters
  symbol/             frame renderers, module map, dladdr/SymFromAddr, demangle/, trim.rs
  output/             json.rs, dhat_v2.rs, native.rs, folded.rs, text.rs, html.rs, viewer.html
  bin/heapscope-symbolize/    offline resolver (§6.1)
```

The type this plan calls `Unwinder` is spelled `Strategy` in the code. The module this plan called `core` is spelled `internals`, because `pub mod core` shadowed the `core` crate at the crate root.

### 4.1 The hot path

1. **Acquire the reentrancy guard first.** Calling the inner allocator first infinitely recurses if the wrapped allocator itself allocates — exactly what `Alloc<A: GlobalAlloc>` (§7) invites with pool and arena allocators.
2. Load the global phase (one acquire atomic). If not `Running`, call inner and return.
3. Call the inner allocator.
4. Sampling decision if enabled (§6.3). Slot-local counter, no atomics.
5. Capture a backtrace into a fixed-size stack array of `usize`. No heap allocation.
6. Hash the frame array → intern to a `PpId`.
7. Update the PP counters, global totals, and peak state **under the peak gate** (§4.3).
8. Insert the live-block entry.

`dealloc` inverts this with one hard constraint: **the live-block entry must be removed before, or while holding the pointer's shard lock across, the inner free.** Once `inner.dealloc(p)` returns, `p` is available to any thread; another thread can receive it and insert its own entry before the freeing thread removes it, destroying the new owner's record. dhat-rs is accidentally safe here only because it holds one global mutex across both. A probe on macOS scored zero cross-thread reuse hits (Apple's per-thread magazines make immediate reuse rare) **[measured]**, but glibc's shared arenas make it reachable.

`alloc_zeroed` **must be overridden** to forward to `inner.alloc_zeroed`. The default `GlobalAlloc::alloc_zeroed` calls `self.alloc` then `write_bytes` **[source: `core/src/alloc/global.rs`]** — which removes `calloc`'s lazy-zero-page fast path from the profiled program, changing its RSS and timing. dhat-rs has this bug. Forwarding also avoids double-counting through `self.alloc`.

**No allocation ever occurs on this path.** All storage comes from the bump arena. This is the structural fix for the deadlock class that forced dhat-rs onto `mintex`.

### 4.2 Locking

**Not `std::sync::Mutex`.** On Apple and other non-futex unix targets it routes through `pthread.rs`, which holds a `OnceBox<pal::Mutex>` whose own doc comment reads *"used to implement synchronization primitives that need allocation"* **[source]**. Measured: a fresh `Mutex`'s first `lock()` performs **1 allocation of 64 bytes** **[measured]**. Inside a `GlobalAlloc` shim that is precisely the recursion being eliminated.

**Not a hand-rolled spinlock either:** priority inversion (Apple deprecated `OSSpinLock` for exactly this, and `sched_yield()` does not donate priority on macOS), unbounded spin inside the `atexit` handler where other threads are still live, and no fairness under a hot shard.

**A thin `RawLock` over the platform primitive** — all allocation-free, all statically initializable:

| Platform | Primitive | Measured |
|---|---|---|
| Apple | `os_unfair_lock` (4 bytes) | 0 allocations, first and 100k'th lock **[measured]** |
| Linux | futex / `PTHREAD_MUTEX_INITIALIZER` | 0 allocations **[measured]** |
| Windows | `SRWLOCK` | — |

`RawLock` also exposes `try_lock_for(bounded)` so the shutdown path degrades to partial output rather than hanging. Under Miri a `cfg(miri)` pure-Rust atomic backend replaces all three, so the race detector stays on (§8, item 7).

**Sharding.** The live-block table is sharded by *pointer* rather than by allocating thread, which makes cross-thread free patterns contend no worse than same-thread ones; PP counters by `PpId`. **Shard count is a compile-time constant (64)** — a runtime count cannot be const-initialized and therefore forces lazy initialization, with a race, into the allocator hot path before `main`.

**Lock order is global and documented: live-block shard → peak gate → program-point shard → arena.** The gate must be held *across* the per-point update, so it is acquired before the point shard. The two shard families never nest in practice — `alloc` releases the gate before inserting, `dealloc` removes before taking it — so the order is deliberately permissive about them. Enforced by a debug-build order checker (`internals/order.rs`), which is the authority.

**All global state must be const-initializable.** No `OnceLock`-guarded allocation is reachable from the shim, because the shim is live before `main`.

### 4.3 t-gmax: the lazy epoch algorithm, and the exactness gate

DHAT reports, per program point, live bytes/blocks *at the instant the whole process hit its peak*. Valgrind and dhat-rs both do an O(#PPs) sweep on every new peak. We do it in O(1) amortized.

**The algorithm.** A global `gmax_epoch`, incremented whenever a new global max is set. Each PP stores `snapshot_epoch`, `curr_*`, `at_gmax_*`. On any touch of a PP: if `pp.snapshot_epoch != global_epoch`, first copy `curr_* → at_gmax_*` (the PP was untouched since the peak, so its current values *were* its values at the peak), then update `snapshot_epoch`, then apply the change. At end of run, flush every PP with a stale epoch identically.

**The epoch must bump on `>=`, not `>`.** Valgrind is explicit (`dh_main.c:373-379`): *"The use of `>=` rather than `>` means that if there are multiple equal peaks we record the latest one"* **[source]**. A model checker comparing this against Valgrind's eager sweep over 200,000 random traces **[measured]**:

```
epoch-on-'>=':  at_gmax mismatches = 0 / 199999      tg mismatches = 0
epoch-on-'>' :  at_gmax mismatches = 12110 / 199999  tg mismatches = 12929
```

**The concurrency problem, and the gate.** Because the PP update and the global peak check are not one atomic action, another thread can bump the epoch between them. Modelled with two threads over 400,000 traces **[measured]**: `sum(pp.gb) > gmax` in 2,274 cases (0.6%), `sum(pp.gb) < gmax` in 33,247 (8.3%). Not a defect in the epoch trick — it is inherent to decoupling PP updates from peak detection, so "the values at t-gmax" is not well-defined.

**Decision (§10.1): make the peak linearizable.** A global readers-writer **peak gate**: every allocation event takes it *shared*; an event that could reach the maximum takes it *exclusively from the start*. Peaks are rare after warmup, so the exclusive path is amortized-cheap.

**Not an upgrade from shared to exclusive, which loses peaks.** Thread A allocates to a new maximum of 100; before it re-acquires exclusively, thread B frees to 50; A then reads 50, sees it below the maximum, and records nothing. The peak happened and went unreported. The shared path instead uses a compare-exchange that commits only if the result stays strictly below the maximum, and escalates otherwise.

During a monotonically growing phase every allocation is a new peak, so the exclusive path is *not* rare during warmup. See §11 for what this costs.

### 4.4 Time

`tu`/`Mtu` are format-level strings, so the time base is ours to choose.

- **`TimeSource::Events` (default)** — a monotonically increasing counter of recorded events. Free, and it is what lets two runs of one workload record the same numbers (§8, item 1). Described as "recorded events" rather than "allocation events", because an ad hoc run has no allocations to count.
- **`TimeSource::Monotonic`** — `Instant::now()` deltas in µs. Matches dhat-rs and Valgrind's intent.

`Monotonic` is not the default, on performance grounds as much as reproducibility: `Instant::now()` costs **17.7 ns [measured]**, about the same as an entire frame-pointer walk (18.4 ns), so choosing wall-clock time roughly doubles hot-path cost.

### 4.5 The live-block table: what it is for, and its budget

`GlobalAlloc::dealloc` receives the `Layout`, so **we never store the size** — and using `layout.size()` at free time is strictly more robust than a stored copy, which can desync. What the table is still required for:

1. **PP attribution of the free.** The layout says how big, not *who allocated*. Without `ptr → PpId`, frees decrement the freeing site, driving per-PP `curr_bytes` negative and destroying `eb`/`gb`.
2. **Block lifetime** — `tl` and the `tuth` short-lived counts need a per-block allocation timestamp.
3. **Membership.** Distinguishing "we never recorded this block" (pre-start, or skipped by sampling) from "we did". Essential under sampling.

The value is 16 bytes: a `PpId`, a `Site` (§6.5–6.6, which fits in padding that was already there), and a **64-bit** birth timestamp. Not the 8-byte form with a 32-bit event counter — that wraps after 4.29 billion allocations, under a minute for a busy process, and every block outliving a wrap reports a wrong, smaller lifetime. Those are exactly the long-lived startup blocks a profile is read to understand.

The table *is* the membership set: a free of an unknown pointer is ignored, which is the desired behavior. **A cap is required** — 10 M live blocks is ~160–320 MB of profiler state, and a memory-analysis tool with unbounded memory growth contradicts §1.2. `max_live_blocks` with explicit overflow accounting, mirroring the `[overflow]` PP.

### 4.6 Lifecycle and edge cases

Three states (`Idle → Starting → Running → Finished`) behind an atomic. The `Starting` phase exists because publishing `Running` and *then* configuring the run records allocations on other threads against settings not yet chosen.

| Situation | Behavior |
|---|---|
| Allocation before profiler start | Not in the table; a later free finds no entry and is ignored. No underflow. |
| Allocation live at profiler stop | Counted in `eb`/`ebk`. |
| Allocation after stop | Shim is a straight passthrough. |
| `realloc` | Resize attributed to the **original** PP (§10.5). Resets the allocation instant (matching dhat-rs); counts toward `tb`/`tbk` as a new block while leaving `curr_blocks` unchanged; **participates in the epoch discipline** — a shrinking realloc is a descent from the peak. |
| `std::process::exit` | `atexit` hook. Verified on normal return, `process::exit`, panic-with-unwind, and `process::exit` from a non-main thread **[measured, unix]**. **Not on Windows**, where Rust's `process::exit` is a direct `ExitProcess` call that never walks the CRT's handler list (`library/std/src/sys/exit.rs`); returning from `main` is unaffected. The suite asserts the absence there rather than skipping the row. |
| `atexit` ordering | Handlers run LIFO, sharing the list with C++ static destructors via `__cxa_atexit`. The at-t-end snapshot is therefore taken **partway through teardown** and differs from the `Drop` path. **Measured**: one workload, two endings — the drop path reports 162,052 bytes in 58 blocks still live where the exit handler reports none, because `main` has returned. Cumulative totals identical, live totals not. |
| `_exit`, `abort`, fatal signals | **No output at all.** `atexit` is bypassed. The remedy is `Profiler::save_dhat_v2`/`save_native`/`save_html` first, which is run in front of a real `_exit` and a real `abort`. Such a file records `shutdown: running` — a point-in-time reading, not a reading of the finished program. Nothing can be done about a fatal signal. |
| Panic / unwind | `Profiler::drop` runs; output written. **Under `panic = "abort"` there is no output**, for the same reason `abort` produces none. `ci/check-panic-abort.sh` covers both halves, and the first is the one users depend on: a `panic = "abort"` program that ends normally is profiled exactly as any other. |
| Concurrent shutdown | The handler flips state to `Finished` **first**, then bounded-waits for in-flight events. Flipping first also keeps the writer's own allocations out of the profile. One claim decides which writer runs, and temporary paths are unique per write. |
| `fork()` | `pthread_atfork(prepare, parent, child)`. `prepare` takes every lock in the §4.2 order **and holds the reentrancy guard across the whole fork window**; `parent` releases; `child` force-reinitializes and enters `ForkedChild`. |
| Allocation from a signal handler | The reentrancy guard covers this by design: a signal arriving while the interrupted thread is inside the shim sees the guard set and skips. Requires the guard to span the **entire** critical section including arena refills. |
| Internal invariant violation | Poison, stop recording, one diagnostic line to stderr, program continues. **Never panic, never abort.** |
| Two profilers at once | Second `Profiler::new()` returns `Err`. |
| Table capacity exhausted | Stop interning; accumulate into a synthetic `[overflow]` PP visible in the output. **Exercised by a real run**: a `full-table` probe mode holds 8,192 blocks against a ceiling of 128, records 416, turns away 8,251, and reports 8,667 observed requests — taken *before* the table is consulted, so 416 + 8,251 = 8,667 is the assertion. What the row forbids is half-counting, and the two sides of that are one line apart in `Engine::record_alloc`. |

**Why `pthread_atfork` and not pid detection.** A probe running a spinlock design with a background thread hammering it, then forking **[measured]**: `child: LOCK IS STUCK HELD -> would deadlock forever (6/6 runs)`. The lock is held by a thread that does not exist in the child; detection cannot help, because the state is already corrupt and any later acquisition hangs forever. It would also put a `getpid()` on the hot path.

Two cases remain unhandled and documented: a second thread forking while the first is inside our own prepare handler, and a `fork` issued from a signal handler that interrupted a thread inside the shim.

### 4.7 TLS safety

A `const {}`-initialized, **destructor-free** thread-local is safe during teardown **[measured]**. But that is not what this crate uses, and the reason is a hazard the plan originally described wrongly.

**Reproduced in M3.** `tests/cdylib_tls.rs` loads a `cdylib` fixture with `dlopen` and calls into it on a thread the *test* created, so the first thing that image does on that thread is allocate. Two guard builds were put through it:

| Guard consults | Result |
|---|---|
| a `const`-initialized, destructor-free thread-local (the `dhat-rs` shape) | **passes** — no recursion, no leak |
| a thread-local whose initializer allocates | **stack overflow, `SIGABRT`** |

So the mechanism is not dyld's `tlv_get_addr`, as first written: dyld's TLV allocation calls the C `malloc` in libsystem, and a Rust `#[global_allocator]` does not sit in front of that. What recurses is a thread-local whose *initialization allocates through the Rust global allocator*, reached before the guard is established. `try_with` cannot see it, because the slot is not unavailable — it is mid-initialization.

**So the guard uses no thread-local storage at all.** The depth lives in a static table keyed by `pthread_self`/`GetCurrentThreadId`. The plan's "lock-free fallback path" was never built: a fallback implies a primary that can fail *and be noticed failing*, and this one cannot be.

The slot also carries the thread's attribution row, its stack bounds, and the sampling countdown — 48 bytes, a 192 KiB static table, guarded by a `const` size assertion so a future field stops the build rather than doubling a table quietly.

Getting the slot table's fork reset wrong is a live hazard: releasing dead threads' slots punches gaps into an open-addressed probe sequence, silently migrating a surviving thread to a *second* slot, whose thread-local destructor then zeroes the depth of the slot a live `Guard` still refers to. Dead slots keep their owners and lose only their contents.

---

## 5. Stack unwinding

The performance crux, and an M1 concern.

`std::backtrace::Backtrace` is unusable: `Backtrace::frames()` is unstable on 1.96, 1.97, **and current nightly** (`E0658`, tracking issue #79676) **[measured]**, so the only std-pure route is parsing `Display` output — which symbolizes eagerly and allocates.

### 5.1 Measured costs (aarch64-apple-darwin)

Baseline: 1 M allocations through a trivial counting shim = **15.8 ns/alloc**.

| Strategy | Frames | ns/call |
|---|---|---|
| Frame-pointer walk | 13 | **18.4** |
| libc `backtrace()` (libSystem) | 15 | 157 |
| `_Unwind_Backtrace`, cap 1 | 1 | 1,404 |
| `_Unwind_Backtrace`, cap 4 | 4 | 2,938 |
| `_Unwind_Backtrace`, cap 32 | 14 | **8,335** |
| `std::backtrace::Backtrace::force_capture` | — | 9,937 |

`_Unwind_Backtrace` costs ~1.4 µs fixed plus ~520 ns/frame. At `max_depth: 32` that is ~500× the baseline allocation. This is not "slower"; it is a different tool.

The libc `backtrace()` row is macOS-specific — libSystem implements it as a frame-pointer walk. glibc implements it on top of `_Unwind_Backtrace`, so this middle tier likely does not exist on Linux **[unverified]**, and given §5.3 it is not load-bearing.

### 5.2 The `panic=abort` trap

`_Unwind_Backtrace` under `-Cpanic=abort` returns `_URC_END_OF_STACK` — *success* — having captured **nothing** **[measured]**:

```
=== panic=abort ===                            returned 5, frames=0
=== panic=abort + force-unwind-tables=yes ===  returned 5, frames=7
```

`-Cpanic=abort` disables unwind tables, so libunwind cannot take the first step. The result is a profile where every allocation is attributed to an empty stack, with no error anywhere. Any path that can reach the system unwinder must treat a zero-frame capture as a **hard configuration error** naming `-Cforce-unwind-tables=yes`, never as an empty stack.

### 5.3 Frame pointers by target, and the flag requirement

Verified via `rustc -Zunstable-options --print target-spec-json` plus disassembly of a non-leaf function per target **[measured]**:

| Target | `frame-pointer` spec | User-crate non-leaf prologue | Flag needed? |
|---|---|---|---|
| aarch64-apple-darwin | `"non-leaf"` | `stp x29,x30` + `add x29,sp` | **No** |
| aarch64-unknown-linux-gnu | `"non-leaf"` | `stp x29,x30,[sp,#-32]!` | **No** |
| x86_64-unknown-linux-gnu | *(absent)* | `pushq %r14; pushq %rbx` | **Yes** |
| x86_64-pc-windows-msvc | *(absent)* | no FP | n/a — see §10.2 |

`frame-pointer: "non-leaf"` is a codegen policy applied to user crates too, so FP walking works out of the box on both aarch64 targets **[measured]**. On x86_64-linux the shipped `libstd` *is* built with frame pointers (940/1237 functions start `push %rbp` = 76% **[measured]**) but user crates are not, so a walk starting in our shim is invalid on the first hop.

**Decision (§10.2): require the flag on x86_64 unix.** At startup the capability probe walks a known-depth stack and verifies it, and on failure returns a hard error naming the remedy:

```
error: heapscope requires frame pointers on x86_64-unknown-linux-gnu.
       Rebuild with:  RUSTFLAGS="-C force-frame-pointers=yes"
       For C/C++ dependencies built via `cc`, also set:
                      CFLAGS="-fno-omit-frame-pointer"
```

The C/C++ line matters: `cc`-built dependencies default to `-fomit-frame-pointer` at `-O2`, and no `RUSTFLAGS` setting reaches them.

`Strategy::System` remains an **explicit opt-in** on unix, never selected automatically. The governing rule is **never silently slow** — the failure mode to avoid is a user profiling for ten minutes and concluding the tool is broken.

### 5.4 Validation, and honest reporting of trace quality

Per-frame stack-bounds and FP-monotonicity checks prevent *crashes*. They do not prevent *wrong or silently truncated* traces, and the startup probe can produce false confidence: it walks our own frames, which under uniform `RUSTFLAGS` says nothing about `cc`-built dependencies, hand-written asm, JIT frames, or threads created by a C library.

So captures are counted by outcome and surfaced as `heapscope.captures` with four values — `complete`, `truncated`, `suspect`, `noFrames`. The validator requires a profile with program points to have recorded captures, so a counter nobody increments fails a test rather than reading as a guarantee.

**Where a capture starts is calibrated at startup, not assumed.** One `SKIP_FRAMES` constant cannot cover both strategies — they begin at different depths, measured at 3 extra frames in a debug build and 2 in release. The calibration takes two captures one frame apart from the same call site and reads the index of the first difference: everything below the extra frame is the same code returning to the same instruction, so the two agree byte for byte. No address windows, no layout assumptions. The shim methods are `#[inline(never)]` so the frame layout is the same at every optimisation level.

The hot path stores **return addresses only** — no symbol lookup, no allocation, no string work.

---

## 6. Symbolization and features

### 6.1 Tier 2 (module map + offline) is the primary path

`dladdr` returns **success with a NULL symbol name** on a stripped binary **[measured]**, and `strip = true` is common in release profiles:

```
pub_fn   rc=1 sym=_ZN2dl6pub_fn17ha505993b13ec8b66E   off=0
--- after strip -x ---
pub_fn   rc=1 sym=<null>                              off=4364011916
```

- **Tier 2 — module map + offline symbolization — is a required deliverable.** Emit raw addresses plus per-image load address, path, and build-id/UUID. Resolve later via `heapscope-symbolize`, or `atos`/`addr2line`/`llvm-symbolizer`. This also enables symbolizing a profile on a different machine from the one that produced it, which dhat-rs cannot do.
- **Tier 1 — `dladdr` / `SymFromAddr` — is the convenience layer on top.** It renders the name *in front of* the module and offset rather than instead of them, so a profile written on a machine with symbols stays exactly as resolvable offline as one without. Where nothing is found the two renderings are byte-identical, which is what makes tier 1 safe as a default. `sname == NULL` is not the only way the platform reports an answer it does not have — see §9.1.
- **Tier 3 — built-in DWARF/PDB — is post-1.0** (M8), behind a cargo feature.

### 6.2 Demangling

**Both manglings ship, and the entry point picks by prefix rather than by configuration.** v0 is not a minority case: the default changed between this crate's MSRV and current stable, same source, no flags **[measured]**:

```
rustc 1.96.0   _ZN1p14probe_function17he140cc384555f8bfE   legacy
rustc 1.97.0   _RNvCs785SGTk9yHm_1p14probe_function        v0
```

Legacy remains just as necessary, because MSRV builds are entirely legacy and any prebuilt artefact can be either. Both also appear in one binary.

Budgeted at ~2,000 lines against `rustc-demangle`'s 1,619 for v0 alone; **built at 1,656**. Fuzzing was pulled forward from M7 to M4, because malformed input from stripped or partial symbol tables is a live recursion-bomb risk — and it earned the move, finding six defects a 201,457-symbol differential could not (§9.1).

### 6.3 Sampling

With mean interval `R`, an allocation of size `s` is sampled with probability `1 - exp(-s/R)`, and the unbiased weight is `s / (1 - exp(-s/R))` — **computed per sample from its own size**, not a global multiplier. Allocations with `s >> R` are sampled with probability ≈1 and must not be scaled up.

Also required:

- `tl` and the `tuth` short-lived counts need the same per-sample weighting.
- The bytes-until-next-sample counter must be **per-thread**, or it is a contended atomic that defeats the purpose. It cannot be a `thread_local!` either (§4.7), so it lives in the guard slot.
- The PRNG must be per-thread and fixed-seeded, or sampling breaks `TimeSource::Events` reproducibility.
- Sampled `gmax` is an estimate with variance, not a bound, so **a reading must refuse a sampled run**.

That last refusal belongs at the **reading**, not on the builder. A builder can only refuse a program that *declared* it intended to assert; a program that did not declare it goes on asserting against estimates in silence. So the refusal is `StatsError::Sampled`, returned from `HeapStats::get`, and there is no `testing` flag on the builder.

### 6.4–6.11 Feature set

- **6.4 Ad hoc profiling** — `heapscope::event(weight)`, same PP machinery, `bklt: false`.
- **6.5 Phases / regions** — `let _r = heapscope::region("parsing");` with per-region breakdowns in the native format.
- **6.6 Thread attribution** — which thread allocated each block, plus thread names. Valgrind's DHAT structurally cannot do this. **Names must be captured at record time** (they die with the thread), so this is a hot-path design constraint.
- **6.7 Self-metrics** — every profile carries arena bytes, live-table bytes, PP and frame counts, capture strategy, ns/capture, capture outcomes, dropped events, sampling rate. §12 promises "honestly measured overhead"; this is what makes that claim checkable by the user rather than by us.
- **6.8 Native format** — versioned superset: module map, unsymbolized addresses, realloc/zeroed/alignment histograms, region and thread attribution, sampling metadata, size-class histograms, truncation accounting.
- **6.9 Testing API** — `HeapStats::get()`, plus `assert_max_bytes!`, `assert_no_leaks!`, `assert_alloc_count!` that dump a profile on failure, and `assert_baseline!` for CI gates.
- **6.10 Explicit copy instrumentation** — `heapscope::copied(n)`.
- **6.11 Text report** — top-N program points by total bytes, peak, block count, and lifetime, to stderr at shutdown.

### 6.12 The bundled viewer

`Output::html("profile.html")` writes a single self-contained file with the profile inlined. A **complement** to DHAT v2 output, not a replacement: DHAT v2 stays the primary interchange format so profiles remain shareable.

Three reasons this is load-bearing rather than cosmetic:

1. **Valgrind does not exist on Windows and does not support Apple Silicon.** On both, we would otherwise ship a format whose only viewer comes from a tool that cannot be installed.
2. **The version-mismatch failure is incomprehensible.** dhat-rs issue #10 was a user on Valgrind 3.16's v1 viewer opening a v2 file; because `checkFields` runs *before* the version comparison, the error names a missing **v1** field (`mi`) that has not existed since 2021 **[source]**. Distros ship old Valgrind for years.
3. **DHAT v2 has no field for most of what we collect.** Thread attribution, regions, sampling metadata, self-metrics, and size-class histograms are all invisible in `dh_view.html`.

**Constraints, because they are what keep this from metastasizing:** hand-written HTML/CSS/JS in one file, **no build step, ever** — the `[dependencies]` discipline of §1.2 applies to the frontend too. Scope is a sortable program-point tree plus the thread and region views DHAT cannot show; we are not reproducing `dh_view.html` feature-for-feature, because a half-reproduction is worse than sending people to the original.

**Checked in three places, because one file is two programs.** `ci/check-bundled-viewer.sh` runs the page's arithmetic against closed forms with no DOM; `ci/check-viewer-interaction.sh` drives its controls in a headless browser and checks the DOM against that arithmetic; `tests/html_output.rs` holds it to being one self-contained file that a hostile string cannot end early. The no-build-step constraint survives all three — the browser is spoken to over a pipe, so nothing is installed to run any of them.

---

## 7. Public API sketch

```rust
#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

fn main() {
    let _profiler = heapscope::Profiler::new().unwrap(); // writes dhat-heap.json on drop
}
```

```rust
use heapscope::{Output, Profiler, TimeSource};

let profiler = Profiler::builder()
    .time_source(TimeSource::Events)          // default; deterministic and cheaper
    .max_depth(24)
    .max_live_blocks(4_000_000)
    .sampling(512 * 1024)                     // mean bytes between sample points
    .trim_frames(true)                        // the default; `false` keeps every frame
    .output(Output::dhat_v2("target/dhat-heap.json"))
    .also(Output::html("target/profile.html"))
    .build()?;   // Err if frame pointers are unavailable, or the shim is not installed
```

**`Alloc<A: GlobalAlloc>` carries a documented contract:** `A` must not allocate through the global allocator. Pool and arena allocators that do will recurse. This is stated on the type, not buried in prose. The arena requests memory from `std::alloc::System` rather than from `A` for the same reason: routing refills through `A` would make the profiler's own correctness depend on a user upholding that contract, and the failure mode is unbounded recursion inside an allocator.

**A setting is fixed for the life of a run.** A depth limit or a block ceiling that changed halfway through would make one profile describe two configurations, with nothing in the file to say where the change fell. Settings are applied inside `Engine::start`'s configuration window, where the engine is `Starting` and the shim refuses every event.

---

## 8. Correctness strategy

1. **Deterministic output.** With `TimeSource::Events` and a fixed unwinder, two runs of the same workload record the same profile: the same program points, counters, frames, **and order**. Not byte-identical files — a profile also carries the pid, the command line, module load addresses, the runtime address on every frame, and the profiler's measurements of itself. `ci/check-reproducible.sh` enumerates exactly what is excluded and why each item is a fact about the execution rather than about the program. In a program whose threads race to reach the same sites, which of two threads interned a point first is not something a profiler can make repeatable and not something worth trying to.
2. **Differential testing.** A deliberately slow, obviously-correct `ReferenceTracker` (one `BTreeMap`, one lock, eager gmax sweep) runs alongside the real engine in test builds, covering multi-threaded traces. One seam is stated rather than papered over: comparing against a serial model needs a shared order, and the way the multi-threaded comparison gets one is by **serializing the engine** — so it is exact about everything except the shared path it turns off. That path has two checks of its own: the summation invariants, which hold under any interleaving, and a workload whose peak is fixed by its shape rather than by the schedule.
3. **Model checking.** The single-threaded epoch-vs-eager equivalence and the multi-threaded gate are checked by standalone models over hundreds of thousands of generated traces (§4.3). These found two real bugs before any implementation existed and stay in the repo as tests.
4. **Property tests** via `proptest`, whose shrinking is what makes a failing 10,000-operation trace debuggable. Invariants: `curr_bytes` never negative; `sum(pp.eb) == curr_bytes` at end; `sum(pp.gb) == gmax` exactly; `total_bytes` monotonic; every live block has exactly one owner.
5. **Schema validation stricter than the viewer** (§3.1) on every integration test's output, in both directions: a file carrying a field its mode has no measurement for is refused too.
6. **Stress and concurrency.** Alloc storms, allocation during TLS destruction, allocation from `Drop` impls of statics, allocation inside panic handlers, threads spawned/joined during profiling, `fork` in a test child, the cdylib TLS path (§4.7), signal-handler reentrancy (§4.6).
7. **Sanitizers and Miri.** ASan/TSan on nightly via `ci/sanitizers.sh`, each suite preceded by a **positive control** that fails the job if the sanitizer cannot see a planted defect through this crate's `#[global_allocator]` — without it a broken composition is silent and the job reports success for having watched nothing. Miri runs the whole suite with the race detector on, using `RawLock`'s `cfg(miri)` pure-Rust backend, because Miri's own `os_unfair_lock` shim performs a non-atomic read of the lock word and reports a false race. Iteration counts scale under `cfg(miri)`.
8. **Platform matrix.** macOS aarch64, Linux x86_64, Linux aarch64, Windows x86_64 (MSVC). **No musl** (§1.1).

---

## 9. Milestones

The unwinder sits in M1, because §5.1 shows the capture strategy determines the architecture, and M1's reference tracker and property tests need real program points to differentiate on.

| # | Milestone | Contents | Exit criteria |
|---|---|---|---|
| **M0** | Scaffolding | crate layout, split CI, lints, empty-`[dependencies]` enforcement, Miri feasibility spike | CI green; a dev-dep MSRV bump provably cannot break the MSRV job |
| **M1** | Core engine + unwinding | `RawLock`, arena, tables, PP records, epoch + peak gate, state machine, `Alloc` shim, FP walker + capability probe, `ReferenceTracker` | multi-threaded differential tests pass; published ns/capture and gate-contention numbers |
| **M2** | Output + module map | JSON writer, DHAT v2 emitter with §3.2 folding, module map, text summary, strict validator | loads in real `dh_view.html`; trimmed traces never collide; offline symbolization works end to end |
| **M3** | Platform completion | Windows `RtlCaptureStackBackTrace`, `pthread_atfork`, `atexit`, cdylib TLS path, signal-handler property, opt-in `Strategy::System` | all §4.6 rows tested on all platforms |
| **M4** | Symbolization T1 | `dladdr`/`SymFromAddr`, legacy + v0 demanglers, frame trimming, demangler fuzzing | readable reports; corpus passes; fuzzer clean |
| **M5** | API completeness | `ProfilerBuilder`, ad hoc mode, regions, thread attribution, testing API, native format, self-metrics | full documented API; doc examples run in CI |
| **M6** | Performance | shard tuning, sampling with per-sample weights, benchmark suite | published overhead vs. dhat-rs; sampling overhead in low single digits |
| **M7** | Hardening + viewer | bundled HTML viewer, JSON writer fuzzing, `_exit`/`abort` paths, docs + `forbid(missing_docs)`, sanitizers | release candidate — which is not testable on its own, so it was run as an audit of §12's six bullets |
| **M8** | *(post-1.0)* DWARF | ELF/Mach-O/PE readers, `.debug_line`, inline frames | file:line without external tools |

---

## 9.1 Implementation status

**M0 through M7 are complete.** Work after M7 — folded-stack output, the offline symbolizer of §6.1, and the CI step that builds the fixtures the suite runs as programs — is not on the milestone table.

This section is not a change log; git history is. What it records is the part of the implementation that is **still load-bearing**: where the built engine departs from this plan, the platform facts the design is shaped around, the rules the code follows and what each cost to learn, and what remains unproven.

Two exit criteria are not met as written.

**M6's "sampling overhead in low single digits"** is not met on any reading where that means percent: 19 ns against a 31.9 ns unprofiled baseline is 60%. Getting to single digits would mean sampling the size histograms too, and those are what let a sampled profile state its own accuracy. That trade is available and is not made, because a profile that cannot be checked against itself is worth less than 10 ns an allocation.

**M7's "release candidate"** is not testable on its own, so it was run as an audit of §12's six bullets. Four were false or half-true; §12 records what each audit found.

### Divergences from this plan

| Plan says | Built instead | Why |
|---|---|---|
| §4.3: an event that reaches a new peak "upgrades to exclusive" | It takes the gate exclusively **from the start**; the shared path uses a compare-exchange that commits only if the result stays strictly below the maximum | The upgrade loses peaks. A allocates to a new maximum of 100; before it re-acquires exclusively, B frees to 50; A reads 50, sees it below the maximum, records nothing. |
| §4.7: a thread-local flag with a "lock-free fallback" | **No thread-local storage at all**; a static table keyed by thread id | A fallback implies a primary that can fail *and be noticed failing*. `try_with` cannot see a slot that is mid-initialization. |
| §4.5: live-block value is 8 bytes with a 32-bit event counter | 16 bytes with a 64-bit birth timestamp | A 32-bit counter wraps in under a minute for a busy process, and every block outliving a wrap reports a wrong, smaller lifetime — exactly the long-lived startup blocks a profile is read to understand. |
| §4.1: the arena requests memory "from the inner allocator" | From `std::alloc::System` | Otherwise the profiler's own correctness depends on a user upholding §7's contract, and the failure mode is unbounded recursion inside an allocator. |
| §4.2: lock order "pointer shard → PP shard → peak gate" | live-block shard → peak gate → PP shard → arena | The gate must be held *across* the per-point update. |

Two names differ from the plan: `Unwinder` is spelled `Strategy`, and the module called `core` is `internals`, because `pub mod core` shadowed the `core` crate at the crate root.

### Measured

On aarch64-apple-darwin, baseline `malloc`/`free` of 64 bytes = 16.7 ns:

- Frame-pointer capture: ~5 ns fixed + ~1.3 ns/frame; ~21 ns at 12 frames. `force_capture`: ~18,800 ns, roughly **900×**.
- Uncontended `RawLock` acquire+release: 7.95 ns.
- Full event recording: **27 ns single-threaded, 553 ns at 16 threads** — see §11.
- Platform unwinder at 12 frames: **5,613 ns** on x86_64 glibc against 51 ns for the FP walk; **246 ns** on aarch64-darwin against 47 ns, because Darwin's `backtrace` *is* a frame-pointer walk.
- End to end against `dhat-rs` at ten frames: **129.4 ns/alloc against 8,353.6**, a factor of ~85; 2.3 MiB extra RSS against 6.5; profile written in 1.0 ms against 5.8. At four threads heapscope rises to 300.1 while the unprofiled workload *falls* from 31.9 to 15.0 — aggregate throughput falls as cores are added.
- Sampled at 128 KiB: 51.0 ns at one thread, 69.8 at four. Profiling's cost falls from 97.5 ns to 19.1, a factor of 5.1, and scales better: one thread to four costs the unsampled row 2.3× and the sampled one 1.4×.
- Frame trimming on `examples/profile_a_program`: 144 frames become 51, and the frame table falls from 55 entries to 37. A spawned thread's entry sequence alone is nine frames.
- **Sampling has a floor.** From 128 KiB to 16 MiB — a factor of 128 — cost moves only from 51.6 ns to 48.5, while the byte estimate goes from +6.0% to −42.2% and the program points in the profile from 7 to 2. Raising the interval past about a thousand sample points buys nothing and costs the profile: what remains at every interval is the guard, the histograms and the countdown, and only the capture is behind the decision.

**Sharding the global counters was measured and rejected.** §11 said an allocation touches "five globally contended atomics" and that two could become per-shard accumulators. Both halves were wrong. `sysctl hw.cachelinesize` reports 128 bytes and `offset_of!` puts the five words on **two** lines, not five. Deleting both counters outright — which bounds any sharding win from above, sharding being strictly more work — measured *slower* at one thread, and the free-heavy control moved 10–14% on a path that touches neither counter, so the experiment carries a code-layout confound as large as the effect. **The lever is which words share a line, not how many atomic operations there are**, and the first line already holds the minimum decision 10.1 permits: an exact global peak needs a globally consistent running total and the gate that linearises it. The one untried idea is packing `curr_bytes` and `curr_blocks` into a single word.

### Platform facts the design is shaped around

These are properties of the world rather than of this crate. Each one decides something structural, and none would have been found by reading the code.

**Symbolization**

- **`dladdr((void *)-1)` returns *success*.** Measured on macOS 15 arm64 from a Rust binary and again from a C one: it attributes the address to the main executable and names whichever symbol is last in it, at an offset of 18 quintillion. Exactly that one value, because dyld uses it as a sentinel. `0xFFFF_FFFF_FFFF_FFFF` is what a truncated stack walk or a poisoned slot produces, so **the single address the platform answers wrongly for is the one a profiler is most likely to ask about**, and the answer is a confident real name with nothing marking it doubtful. `Symbolized` therefore asks the module map whether an address is in a real image *before* asking what it is called — which also costs no lookup at all for a garbage address, saving a lock and a dbghelp call per bad frame on Windows.
- **How many frames get a name is a platform fact, not a design choice**: **52 of 52 on macOS aarch64, 0 of 70 on Linux aarch64** **[measured, both]**. `dladdr` on ELF sees `.dynsym`, and an executable exports almost nothing; `-C link-args=-rdynamic` does not close it, bringing back the generic `std` and `alloc` instantiations and leaving every function in the user's own crate unnamed **[measured]**. This is why §6.1 puts tier 2 first, why trimming does nothing on Linux (the rules read frame *names*), and why a test may only *report* how many frames were named — a test demanding a name would fail on a configuration where the profiler works exactly as designed.
- **glibc reports the main executable with an empty name.** Always. The one image whose frames matter most rendered as `(+0xe928)` with nothing to resolve against. Filled in from `/proc/self/exe`.
- **The kernel places `[vdso]` inside the gap between `ld.so`'s two `PT_LOAD` segments**, so recording an image's whole virtual span makes spans overlap and the profiler's own validator rejects the profile it just wrote. On x86_64 the vdso happened to land outside and the test passed; on aarch64 it did not. The map records the **executable** region, not the whole span.
- **`_dyld_image_count` does not enumerate dyld**, which holds the outermost frame of every stack. It is found through `task_info(TASK_DYLD_INFO)`, which is where a debugger gets it. And the slide for such an image cannot come from "the segment that maps file offset zero" — the textbook rule, wrong for both images it can meet: dyld is mapped from the shared cache, where file offsets are offsets into the cache, and a main executable's `__PAGEZERO` maps file offset zero at link-time address zero. It comes from the lowest-addressed executable segment.
- **For an image in the dyld shared cache, `bias` is not the number `llvm-symbolizer` wants.** A cache image's segments sit at cache addresses, so the profile emits an address in the cache rather than in the file — for `/usr/lib/dyld`, `0x1801344e4` where the file wants `0x204e4` — **with the UUIDs matching, so nothing warns the reader**. `atos` resolves it because it takes `image_base`. 41 of the 46 images in a sample profile have paths that do not exist on disk at all. Closing this needs the link-time address out of the file, which is tier 3; until then it is documented on `Module::bias` and macOS users are told to reach for `atos` for system libraries.
- **A failed second `SymInitialize` is the normal case on Windows, not evidence dbghelp is absent.** `std`'s backtrace support opens a session with the same handle and never calls `SymCleanup`, so in any program that has printed a panic backtrace the second call fails — and treating that as "unavailable" would name nothing for the rest of the process, silently, in exactly the case where dbghelp is *most* available. `SymFromAddr` asks only that the handle was passed to `SymInitialize` by someone. Relatedly, `SymSetOptions` replaces the process-wide mask rather than OR-ing into it, so it must not be used carelessly: doing so changes how `std` resolves backtraces as a side effect.

**Process and platform**

- **POSIX runs `pthread_atfork` prepare handlers in *reverse* registration order.** Ours registers late, so it runs **first**, and every handler registered by a library initialised before `main` then runs with all 131 of our locks held. One of those allocating enters the shim, finds the guard free, and reacquires a lock on the same thread — a `SIGKILL` shared with other libraries. The handlers hold the reentrancy guard across the whole fork window.
- **`std::thread::current()` panics once a thread's local data has been destroyed** (`library/std/src/thread/current.rs`), and a late allocation during thread teardown is precisely what reaches it — a panic inside a `GlobalAlloc` method being undefined behaviour rather than a test failure. Thread names come from `pthread_getname_np` and `GetThreadDescription`, which cost one call per thread, allocate nothing, and cannot panic. They also give a better answer: the name in the profile is the string `top -H` and a debugger show, because it is the same string.
- **Rust's `process::exit` on Windows is a direct `ExitProcess` call** that never walks the CRT's handler list, and no hook exists for an executable to notice. Returning from `main` is unaffected everywhere.
- **`-Cpanic=abort` makes `_Unwind_Backtrace` return `_URC_END_OF_STACK` — success — having captured nothing** (§5.2). Any path reaching the system unwinder must treat zero frames as a configuration error, never as an empty stack.
- **Windows has no walkable `rbp` chain, with or without the flag.** Measured under Wine from the same stack at the same instant: hand-walking yielded 2 entries with `-C force-frame-pointers=yes` and 1 without, the second a stack address rather than a return address, while `RtlCaptureStackBackTrace` returned 9 plausible frames either way. See §10.2.
- **`std::sync::Mutex` allocates.** On Apple targets a fresh mutex's first `lock()` performs **1 allocation of 64 bytes** **[measured]**, through a `OnceBox` whose own doc comment says it exists "to implement synchronization primitives that need allocation".
- **`Instant::now()` costs 17.7 ns [measured]** — about a whole frame-pointer walk — which is why `TimeSource::Events` is the default and why the per-capture cost is calibrated once rather than sampled per capture.
- **Darwin's `CLOCK_MONOTONIC` advances in whole microseconds [measured]**, so a fixed calibration batch of 64 walks times as zero on every batch. The calibration probes the clock's granularity and grows a batch until it spans fifty ticks.
- **rustc's default mangling changed from legacy to v0 between this crate's MSRV and current stable** — same source, no flags **[measured]**. Both ship, and the entry point picks by prefix rather than by configuration; both also appear in one binary.
- **ASan's `detect_stack_use_after_return` moves locals into a heap "fake stack"**, so a local is no longer inside the thread's real stack and the frame-pointer walk rejects frames outside the bounds `internals::stack` reports. It defaults on for Linux and off for macOS, so it is set explicitly rather than left alone — two runs that check different things must not report the same word.
- **Cargo's `--all-targets` does not build every target; it *selects* examples explicitly, and an explicit selection compiles each one as a test target.** The artifact is never uplifted to the plain name, and a `crate-type = ["cdylib"]` example comes out as an executable. `test = false` does not stop it either, because it sets a default an explicit selection overrides. And Cargo passes `--bench` under `cargo bench` and **nothing at all** under `cargo test --all-targets` **[measured, both ways]**, so a hand-written benchmark must recognise the *presence* of the flag as its licence to measure.

### Rules the code follows

Each of these is a rule the implementation now holds itself to, and each was arrived at by getting it wrong.

**Take the proof as an argument.** `Engine::record_event`, `record_alloc`, `guard::enter_region` and `stats::dump` all take a `&Guard` — not to use it, to require it. Each reaches the writer-preferring peak gate, where a thread that already holds a read guard and enters `read` again while a flush waits deadlocks against itself, and the reentrancy guard is what makes that unreachable. Inserting `drop(guard)` between the capture and the call failed no test, because the deadlock needs a signal to land in a window of a few instructions. **Taking the proof as an argument makes the early drop a borrow-check error instead of a test nobody can write.**

**Screen where a string becomes output, not at each producer.** `output::push_display` escapes Unicode category `Cc` and the bidirectional formatting characters, applied to the finished frame string rather than inside this crate's own `FrameFormat`s — so it covers one written by somebody else. Applying it per producer was tried and leaked twice: a property test asserting that *no string anywhere in a parsed profile* carries such a character failed on the module map's `path`, and fuzzing later found `Module::build_id` written straight through by both emitters, two lines below a path that is screened. That second one is the argument in miniature — the field is a note section rendered as hexadecimal and this crate's own producer cannot make an unsafe one, so **the reasoning that would have caught it is a chain through a producer, and the rule exists so that nobody has to make that chain.** The same rule is why `push_display` is `#[doc(hidden)] pub` rather than copied into `heapscope-symbolize`: a second copy of a screening rule drifts, and one of the two then stops guarding anything.

The escaped set deliberately excludes the zero-width formatting characters: U+200D carries legitimate meaning in filenames and emoji sequences, and escaping it would damage real paths to guard against two frames looking alike.

**Every emitter takes the reentrancy guard, or it records itself.** Found three times before it became a rule: `write_text_summary` recorded its own `format!` calls at **76 allocations per call, measured**, in the function most likely to be called mid-run; `write_native` is the most allocation-heavy emitter in the crate and every path reaching it in the suite did so after `stop()`; and `assert_baseline!` reads its file on the path that *passes*, so an unguarded check pushes the totals past the numbers it had just compared them against.

**A `write_*` that takes its sink by value must flush it, because the caller cannot — it was moved.** `write_dhat_v2` and `write_native` end in `JsonWriter::finish`, which flushes and propagates; `write_html` ended in a bare `write_all` whose `BufWriter` was flushed by `Drop`, which **discards** the error, so `save_html` returned `Ok` over a failed final write and a page could be renamed into place truncated. `_exit` is the sharpest available test of this, because nothing runs after it to repair a buffer that was never emptied or a rename that never happened.

**Omit what was not measured; never zero it.** `bklt: false` means those fields are absent, not zero — an event was never live and never died, so a zero would be a measurement of something that did not happen. The reader must honour the same rule: the viewer read `totals.maxBytes || 0` and **`|| 0` turns a deliberate absence back into a measurement**, printing `peak 0 B` for ad hoc profiles. Of three renderings of one snapshot, two agreed and the third did not. Deriving presence from the *absence* rather than from `mode === "heap"` is deliberate — the writer's omission is already the statement.

**Refuse where the number is read; refuse a misconfiguration where it is made.** The two halves are not the same. §6.3 said the builder should reject sampling combined with testing, but a builder can only refuse a program that *declared* it intended to assert, so the refusal is `StatsError::Sampled` returned from `HeapStats::get` — which needs no declaration and cannot be bypassed. A **missing shim** goes the other way: a heap run without `#[global_allocator] static ALLOC: heapscope::Alloc` recorded nothing and `assert_max_bytes!(64 * 1024)` passed in a program that had just allocated 10 MiB, and a reading is the wrong place to catch that, because by then the run is over and the answer is still zero. It refuses at startup, naming the missing line.

That probe runs before the engine is claimed for two reasons, and the second is easy to miss: **the check *is* an allocation, and an allocation made once the engine is running is recorded** — so probing a moment later would put a program point belonging to `Profiler::start` into every profile and add one to the `total_blocks` that `assert_alloc_count!` compares against.

**A reading can refuse, and that is what makes an assertion able to fail.** `HeapStats::get` returns a `Result` and refuses on six conditions: no run in this process, a run counting something else, a poisoned engine, a `fork` child holding its parent's counters, a sampled run, and a run whose live-block table filled. **A getter that returned zeros for any of them would turn every budget built on it into an assertion that cannot fail**, and the failing case — a test whose profiler was never started — is silent forever. The table-filled case is a refusal for the *assertions* only: `dropped_blocks` is a field on the reading, because a caller who wants the numbers with their caveat can have them and an assertion cannot carry a caveat.

**Report what took effect, not what was asked for.** `heapscope.settings` is read back from the engine rather than from the builder, so a depth past the shim's buffer and a live-block ceiling rounded up by 64 shards come back as the values in force. A run that asked for 5,000 and tracked 8,192 would otherwise leave `droppedBlocks` reading as a contradiction. `settings.trimFrames` was removed outright for the same reason: trimming is a *rendering* setting and the emitter renders with whatever it is handed, so a default profiler written through `write_dhat_v2_with(&Symbolized::new(..))` reported `"trimFrames":true` beside `"trimmedFrames":0`. What a file did is `trimmedFrames`.

**A setting is fixed for the life of a run**, because a depth limit or a block ceiling that changed halfway through would make one profile describe two configurations with nothing in the file to say where the change fell. Settings are applied inside `Engine::start`'s configuration window, where the engine is `Starting` and the shim refuses every event. `Engine::configure` is `pub(crate)`, which it was not at first: a review reached it through the public `heapscope::engine()` and changed a running profiler's settings mid-process.

**Names are added in front of the image and offset, never substituted for them.** In-process, that is what makes tier 1 safe as a default — where nothing is found the rendering is byte-identical to `ModuleOffsets`, so a profile written on a machine with symbols stays exactly as resolvable offline as one without. Offline, `heapscope-symbolize` keeps `symbol` as the running process reported it and *adds* `function`/`file`/`line`/`inlinedBy`, which is what makes resolving against the **wrong binary** visible instead of silent.

**Addresses are hexadecimal strings.** A JSON number is a double in JavaScript, exact only to 2⁵³, so `JSON.parse` silently rounds a 64-bit address — and an address wrong in its low bits names the wrong line of the wrong function with nothing about it looking wrong. The validator has one `address` rule and every address goes through it.

**Count the request before consulting the table**, so `observedBlocks == totalBlocks + droppedBlocks` and a reader can tell what the program asked for from what the profiler managed to track. Under sampling this is what lets a profile **contain the evidence for its own accuracy**: counting a shape costs no stack walk, so the histograms stay exact and the true count sits beside the estimate — 248,568 estimated against 250,010 real on the benchmark workload, an error of 0.6%.

**Fold by provable bound, not by maximum** (§3.2). Taking the larger of two maxima can produce a program point with more bytes live at the global peak than it ever had live at all.

**Order by first arrival, not by shard iteration.** `PpTable::intern` chooses a shard by hashing return addresses, so **the order the engine offers its points in is a reading of where the loader mapped the program**, and ASLR permutes the profile on every execution — six distinct orderings across six runs of one deterministic workload, with every number agreeing. A program point carries the position at which the program first reached it, which is the only slide-independent identity a record has, and `Snapshot::of` sorts by it so every emitter inherits it. The counter is claimed once per *new* point, and the field occupies padding `frames_len` already left behind.

**Never upgrade the peak gate from shared to exclusive** — see the divergences table.

**Apply a depth limit by shortening the unwinder's buffer, not by cutting the result.** Both produce the same frames; only the first makes the profile say what happened, because a walk that stops because the buffer is full reports itself as truncated. Every other assertion in the end-to-end test passes under the wrong implementation. `unwind::depth_room` asks the strategy how much buffer a capture of *n* frames needs, because `backtrace(3)` takes no skip parameter and the discarded frames come out of the caller's buffer.

**Identify frames by comparison, never by an address window.** `calibrate` finds where a capture starts by taking two captures one frame apart from the same call site and reading the index of the first difference: everything below the extra frame is the same code returning to the same instruction, so the two agree byte for byte. The shim methods are `#[inline(never)]` so the layout is the same at every optimisation level. **The rejected alternative — "within 8 KB of this function pointer" — is a guess about code layout rather than a fact about it**, and it broke three separate tests when an unrelated module moved code.

**Trim on position, not on name.** `std::thread::lifecycle::spawn_unchecked` appears *below* `__rust_begin_short_backtrace` on a spawned thread, where it is what started the thread, and *above* it on the parent, where it is the frame that boxed the closure and is exactly where those bytes came from. Any rule matching it by name is wrong in one of the two places. The same argument rejected a per-platform list of thread entry points (`thread_start`, `clone3`, `RtlUserThreadStart`) in a test oracle: asking whether a point names any of *this binary's* functions needs no such list.

**A profile is a partition for sums and not for maxima.** Summing bytes over a subtree is arithmetic over a partition; two points that each peaked at 4 MB at different moments did not jointly peak at 8 MB. The viewer takes the largest of the three provable lower bounds — the biggest single peak, the sum at t-gmax, the sum at t-end — which is the identical rule `Point::merge` applies. The same fact rules out a fifth `FoldedMetric`: a point's own `maxBytes` is a real measurement that sums to nothing, so `PeakBytes` is `atGmaxBytes`, and the two are one field apart with the wrong one drawing a flame graph wider than the peak it claims to show.

**A rewriter has to preserve what it ignores.** Every profile states its own rule — ignore unknown fields, refuse an unknown `formatVersion` — and a reader that also writes cannot honour the first half with a `BTreeMap`, which reorders every object and merges repeated keys, nor by parsing numbers to `f64`, which rounds a counter past 2⁵³ and turns `1e3` into `1000`. `heapscope-symbolize`'s JSON layer keeps member order and a number's own text, and is deliberately a **third** parser of this format: `tests/support/json.rs` is the oracle the emitters are checked against, and an oracle sharing code with what it checks agrees with any answer that code gives.

**The bundled page carries the native profile verbatim**, and that is the decision everything else followed from. The bytes between the two script tags are exactly what `save_native` writes, so there is no second schema to keep in step and a reader with no browser can lift the JSON out with a text editor. Everything beyond the tree came free from it. What the page cannot do is demangle — that would mean a second implementation of both manglings in a file that may not acquire a build step — so names are rendered in Rust by the same `FrameFormat` the other emitters use and travel beside the profile.

**`</script>` in a profile is not hypothetical.** Three of its strings are written by somebody else: a symbol, a path, and `argv`. A directory may be named `a<`, which puts `</script>` in a path without anybody being hostile, and the failure is not a mangled name — it is a page that stops parsing at the injected tag and displays nothing, on a profile that looked fine when written. The emitter escapes `<` as `\u003c` on the byte stream: valid JSON, changes no string's value, safe a byte at a time because JSON has no `<` outside a string literal, and escaping the class rather than the spelling also covers `<!--`. **This runs on essentially every frame of every Rust profile**, since a demangled generic contains `<` several times over.

**A demangler in a profiler runs on adversarial input as a matter of routine**, which is what justifies reimplementing a spec this project does not own. Symbol tables get stripped, mismatched, and corrupted, and what that demands beyond producing the right name is a work budget charged **quadratically** for punycode (decoding inserts each character into the middle of what it has decoded, and an identifier was admissible at a size where decoding takes about a second, inside a profiler's shutdown), non-ASCII bytes **rejected before parsing** (neither mangling can legitimately carry one — that is what punycode and `$u..$` exist for), and a suffix that cannot be shown treated as a **refusal rather than a silent omission**, because omitting it renders two different pieces of code under one name, verbatim the collision the suffix exists to prevent. Fuzzing found six defects that 26,323 real symbols missed, every one needing an input no compiler emits.

Termination is proved by construction rather than tested: backreference targets must lie strictly before the `B` naming them, making progress monotonic, and `MAX_DEPTH` bounds nesting. Neither catches backreferences that *share* subtrees, so `n` bytes describe a tree with `2^n` nodes — finite, shallow, and still not worth waiting for; the work budget is what catches that.

**What makes the reimplementation defensible is that the reference is a dev-dependency and gets run.** Reimplementing something to avoid depending on it is a claim about agreement, and agreement is checkable: **201,457 real symbols, zero divergences** — 130,723 v0 and 70,734 legacy from this crate's test binaries and two toolchains' `rustlib` archives — plus 52 where this produces a name and `rustc-demangle` refuses, all Mach-O thread-local initialisers, with a test asserting that this is the *only* divergence. `src/symbol/demangle` also contains **no `unsafe` at all**, checked rather than assumed, which is what lets the Miri job skip its corpus walks and finish in 15 minutes instead of 37 — a justification that would expire silently the day somebody adds an `unsafe` block.

**Two v0 productions are deliberately unimplemented**: `W` (pattern types) and `w` (splat), both behind unstable language features, so no toolchain emits them and there is no way to generate a real symbol to check against. A symbol carrying one is refused and the caller shows the raw text. Implementing them against a specification with no test input would be writing code whose correctness rests on having read the grammar right, which this milestone twice demonstrated is not reliable.

**Both attribution rows move under the peak gate, and that was not the first design.** They started outside every acquisition, on the reasoning that work inside the gate is the cost this crate spends effort avoiding — right for the size histograms and wrong here, because a thread that has moved its row then **blocks** waiting for the flush, so under a shutdown that holds the gate the rows run ahead of the totals by however many are queued: **measured at 15,534 bytes of 175,191, 9%**. That is a wrong profile, not a failed check. Attribution happens in `commit_after_bytes`, the one funnel every gated path uses, and the site travels on a `Delta` that has no `Default`, so a path that forgot it would not compile.

The validator's row-sum rule is an **equality** wherever the file says the counters were read under exclusion. It was written as one part in a thousand first, copying the histogram rule, and **it passed while the rows were 9% adrift, because the tolerance had been sized to accommodate the defect rather than the profiler fixed to make it unnecessary.** The histogram rule stays a bound, for a reason that does not transfer: closing its last part-in-30,000 would mean counting a shape *inside* the gated region, putting work in the one place this crate keeps empty to make exact an equality only a validator reads.

**Regions are per thread and the innermost one wins.** A process-wide "current phase" would attribute whatever a background thread happens to be doing to whichever phase some other thread is in. Attribution is exclusive — an outer region does not include its inner ones — because a name can be entered under different parents at different times, and a tree built from that would be a shape the run never had.

**`assert_alloc_count!` is an equality, and that is the safe direction to be wrong in.** A ceiling reads better for a CI gate and fails the wrong way: `assert_alloc_count!(3)` meaning "at most three" passes a run that allocated nothing, which is how a broken test goes green and stays green. `assert_max_bytes!` is a ceiling because "max" already names the peak.

**Baselines are line-oriented rather than JSON**, and the second reason is load-bearing. One `key value` per line diffs to exactly the figures that moved. And this crate ships a JSON *writer*, not a reader: adding a parser to the shipped library so a baseline could be spelled in JSON is real new attack surface, reached by construction with a file somebody edited by hand. A baseline holds the six run-level figures and **not** the program points, because gating per point needs an identity that survives the builds people ship — and addresses move, module and offset are stable only within one build, and names need symbolization, so a per-point gate would degrade to "no points matched, everything passed" on exactly the configuration it was written to protect.

**A rule with no spellings has no spelling to get wrong.** `HEAPSCOPE_UPDATE_BASELINE=FALSE` was read as *on*, because `is_off` folded no case while `symbol::dynamic` folded to lowercase — a gate turned into a recorder by a spelling, silently, forever. The public-surface snapshot is updated by an ignored test rather than an environment variable for this reason.

**One engine per process is a constraint on the user's test suite, not only ours.** `cargo test` runs a binary's tests concurrently, so a second test allocating during the profiled window is counted into the first one's totals. Budgets belong in an integration test of their own containing one `#[test]`, and the module documentation says so rather than leaving users to discover it from a count that is occasionally wrong.

**The supported public surface is a committed file**, `tests/data/public_surface.txt`, 630 lines. An `internals` Cargo feature was proposed instead and costs more than it buys: `required-features` on test targets makes a plain `cargo test` *silently skip* the reference tracker and the property tests, and a self dev-dependency makes the library under test a different compilation from the one that ships. Two properties of the scanner are decisions: it **follows re-exports into private and hidden modules**, because `Profiler` is `pub` inside a private module and its methods are supported surface all the same — a scan stopping at module visibility reports 288 items where there are 630; and it **panics on anything it cannot parse**, because a scanner that skips an unfamiliar form reports a smaller surface, the snapshot agrees with it, and the test passes for a reason nobody wrote down. It found its own first version that way, swallowing every file and producing a surface of *nothing*.

### What verification here has learned

Findings about the apparatus rather than the crate, kept because each one silently invalidated results, and because several are still live constraints on how this repository is tested.

**A check performed only by the thing being checked is not a check.** Met four times: a validator rule guarded on the very field it validates, so a heap profile whose shim passed no shapes skipped the check entirely — the exact case its own comment named; a probe's frame checks living inside the probe; a DOM shim in node, which implements what the code under test happens to call and therefore agrees with any implementation; and an oracle sharing code with the emitter it checks. Sanitizer jobs carry the same requirement from the other side: `ci/sanitizers.sh` builds a probe with a deliberate use-after-free, overflow and data race, all in memory that went through the shim, and **fails the job if the sanitizer does not report them** — ASan works by replacing `malloc` and this crate installs a `#[global_allocator]` in front of it, so a broken composition is otherwise silent and the job reports success for having watched nothing.

**A platform that is only compiled for is not a platform that works.** Linux gave up six module-map defects the day it was first executed in a container. Windows, run under Wine, gave up four more and one measurement that changed a decision (§10.2). Both were "supported" beforehand.

**A verification tool nothing executes will be wrong within a month and silent about it.** `ci/dhat-viewer-check.mjs` enforced M2's exit criterion and had never run. The `_exit` remedy, the offline symbolization path of §6.1, and the `panic = "abort"` row were each documented and unexecuted — the last having *changed its mind* by reading what `abort` is defined to do rather than by running a program. Each is now wired into CI. The fuzz targets are compiled by CI for the same reason: running a campaign needs nightly and `cargo-fuzz`, but *compiling* needs neither, and compiling is what rots.

**A concurrency test that waits for a race to happen is not a test of anything.** The first fork test forked twenty-four times under continuous allocation pressure and passed with the handlers removed: a lock is held for tens of nanoseconds and spread across sixty-four shards, so a child almost never touches the shard that was busy. Holding the **peak gate** — the one lock every recorded allocation must pass through — kills the child on the second fork, deterministically. The same shape governs the equal-peak test that covers the gate's shared path: its first workload only allocated, and a coverage guard caught that immediately at one qualifying allocation in a whole run against 1,114 for the shaped version.

**Summation is self-consistency, not correctness.** Every concurrent assertion in the repository once had the form *the parts sum to the whole*, and a lost peak leaves all of them true, because the per-point at-peak counters are refreshed from the same epoch the global maximum was recorded at and so agree with it whatever instant that epoch names — including the wrong one. Deleting the gate's escalation check passed all 673 tests. What covers it now is a workload whose peak is fixed by its shape rather than by the schedule: every round frees one block per thread, waits at a barrier, then allocates one back, so the round ends on an **equal peak**, which the `>=` rule says is the one to record — so every point's at-peak counters must equal its current ones. No reference tracker, no serialization, no dependence on interleaving.

**A test can exercise the right scenario and still observe the wrong thing.** The fork test produced a genuine inherited-lock child and asked it to do something the child does not do. Deleting the entire bodies of `fork_prepare` and `fork_parent` left all 269 tests green, because a `ForkedChild` never records and so never touches an engine lock.

**A passing test can prove nothing, and only mutation testing reveals it.** Three separate times a test passed while exercising none of the path it named, each for a different structural reason: uniform random sizes mean live bytes essentially never returns to a value it held before, so an *equal* peak — the only place the epoch's two rules differ — never occurred; a multi-threaded test performed no reallocations; and the at-peak snapshot reflects only the most recent peak, so a trace ending on a strict increase records identical values either way. Tests now assert their own coverage.

**Where a thing has two arms, the second is untested.** The pattern held across a whole module: poison checked in `HeapStats::of` and not `EventStats::of`, an environment rule gating one variable and not its neighbour, path screening tested at one site of four, a macro rule tested for one macro of three. In each case a mutation on the covered arm died and the identical mutation one function over survived.

**A fixture that no test can distinguish from the answer proves nothing.** Every workload in the testing-API suite allocated monotonically and freed nothing before its assertion, so `max_bytes == total_bytes` and `curr_blocks == total_blocks` at every assertion point and **the suite could not tell which counter any assertion read** — `assert_max_bytes!` comparing `total_bytes` passed the entire suite, which inverts the documented meaning of the crate's most-used macro. There is now a `distinct_figures` fixture in which no two of the six figures are equal, and a test asserting they stay unequal.

**A tolerance sized to accommodate a defect hides it**, and widening one until a flake stops can pass a fabricated value straight through. The published per-capture figure was checked against a band three orders of magnitude wide, and storing `captures * 21` in place of the timed value passed all 674 tests. The repair is a second stopwatch rather than a wider band: two timings of the same operation in the same build agree to within 1.17–1.22 across ten idle runs. That test was flaky at first and **what it was catching was the scheduler** — the calibration ran while time-slicing and reported 515 ps, the re-timing a moment later reported 109, and both were honest — so each side is now the minimum over five rounds.

**A test that cannot fail cleanly is worse than a missing test.** One asserted inside `std::thread::scope` with workers looping until a flag set *after* the assertions, so a failing assertion unwound before that line and `scope` joined threads that would never stop: a mutation run spun four threads for 134 CPU minutes and reported nothing. The same lesson arrived from the opposite direction when a poison test cleared its flag on its last line — **a failing assertion never reaches its last line** — so one broken test made four other results lie. Both now clean up from a `Drop`.

**A check is only a check against something it does not derive from.** The benchmark's wrong-binary check took its expectation from the field the defect was in, so a row pointed at the wrong fixture moved the expectation along with it.

**A stale fixture reports on something other than what you think.** `cargo test --test lifecycle` rebuilds the test but not the examples, so a mutation run executed against a probe compiled from the *unmutated* library. But **a freshness guard no command can satisfy is worse than no guard**: comparing against `Cargo.toml` and `Cargo.lock`, which Cargo tracks by content and by resolution, made every lifecycle test refuse to run after an unrelated manifest edit while printing a remedy that leaves the mtime where it was. Modification time is Cargo's own criterion for a `.rs` file, so that half agrees with Cargo exactly and the other half is dropped. Related: a mutation runner that restores with `mv` or `shutil.copy2` preserves the backup's mtime, so Cargo considers the restored source fresh and the next run tests the previous mutation — hit three times before it was understood.

**A mutation that partially applies is not a result**, and a mutation runner must pass `--no-fail-fast`, or a mutation killed by a unit test stops the run before any integration test executes and the wiring is reported as covered when it was never run.

**Benchmarks need sampling, not repetition, and need a control that must not move.** `benches/contention.rs` took one sample per cell, and three consecutive runs of unmodified code put one figure at 118.4, 107.5 and 69.6 ns. Cells in `benches/overhead.rs` differ in cost by three orders of magnitude, so a fixed repeat count serves neither end. Both now sample against a wall-clock budget and print the spread. **The free-heavy row is a control**: a change to the allocation path cannot move it, so when it moves the run measured the machine rather than the change — and without it, two spurious results would have read as findings. The overhead harness additionally requires a checksum every fixture agrees on, and that **every profile names the workload's deepest call site**, because a profiler whose stacks stopped inside its own shim would be fast, small, useless, and invisible in a table of nanoseconds.

**Miri only checks what it executes.** The `cfg(miri)` lock backend replaces all three platform backends and runs the Darwin paths, so the glibc FFI declarations are never exercised — a hand audit found `pthread_t` is a pointer on Darwin and `unsigned long` on glibc, which Miri could not have caught. **A green Miri run does not mean the Linux FFI is verified.** Three operational constraints keep the job honest: filesystem isolation makes `open` a hard **abort** rather than an `io::Error`, and an unsupported operation aborts the whole test binary, so one blocked test hides every suite scheduled after it — `tests/dhat_output.rs` reported no result at all rather than 32 passes; the job runs `--test-threads=1`, because Miri's deadlock detector is process-wide and under the parallel harness a test that legitimately blocks one thread is attributed to whichever test happened to be running; and locally Miri means `--lib --tests`, because nightly stopped allowing a `cdylib` crate type on `aarch64-apple-darwin`.

**TSan needs `-Zbuild-std`.** An uninstrumented standard library hides its own synchronisation from the race detector and produces reports that are not real, and the first response to a red TSan job is to disbelieve it. With it: 669 tests including the multi-threaded differential suite, no race — the first run against the real `os_unfair_lock`, the real allocator and real threads, where Miri had only ever covered those interleavings with the pure-Rust backend substituted.

**A harness that renders but cannot click covers nothing it draws.** Both bundled-viewer checks ran the page's script under node, and its rendering half opens with `if (typeof document !== "undefined")` — so 460 lines were skipped by construction on every run. Seven deliberately broken controls passed 675 tests and both existing viewer checks. Chrome is now driven headless over `--remote-debugging-pipe`, which speaks the DevTools protocol as NUL-terminated JSON over file descriptors and therefore needs no WebSocket, no port and no npm. **Clicks are dispatched at coordinates, not on elements**, so the browser's hit testing decides what was clicked: `pointer-events: none`, a full-page overlay, and `display: none` are each caught and each names what was under the pointer, which a synthetic `element.click()` cannot see. The two harnesses meet at the pure half — one knows the numbers are right and cannot see the page, the other knows the page shows what the numbers say and takes the numbers on trust.

**A long job's most useful output is the last line before it stops.** Piping a run through `tail` cost fifteen minutes on a process that never exited.

### Still unproven

- **`SymFromAddr` has never been executed anywhere.** It cannot run under Wine — the first call kills the test process with `rosetta error: invalid gdt selector index 5`, a message from Apple's translator that says nothing about whether the code is right. `HEAPSCOPE_SYMBOLIZE=0` exists partly because of this, and is justified independently: dbghelp honours `_NT_SYMBOL_PATH`, so a profile written at exit can block on a network symbol server; two machines produce different frame text for one program, which makes profiles awkward to diff; and an emulator may simply be unable to run it. A native Windows run is the only thing that can close this.
- **dbghelp requires all calls for a process to be serialized, and this crate and `std` hold different locks.** Closing it needs a lock they share.
- **No Windows browser has opened the bundled page**, and the `dh_view.html` half of the viewer check remains node-only against a stub DOM.
- **There is no exact field-by-field differential under genuinely overlapping events, and there cannot be one.** Comparing against a serial model needs a shared order, which the multi-threaded test gets by serializing the engine — so it is exact about everything except the shared path it turns off. Under a serial model the question has no answer, which is the same non-linearizability the gate exists to remove, reappearing one layer up. What the racing path has instead is two independent checks: invariants that hold under any interleaving, and a peak whose value is known before the run starts.
- **The cost of `RtlCaptureStackBackTrace` on real Windows is unmeasured.** Wine timings say nothing about it, and a table walk is certainly far more than a chain walk.
- **The macOS shared-cache bias is wrong for `llvm-symbolizer` and `addr2line`**, and fixing it needs tier 3.
- **CI has never completed a run.** Three runs, all failed before starting, all with the same annotation about account payments. The four-platform matrix is verified locally — natively on macOS aarch64, under Docker on both Linux targets, and under Wine for Windows.
- One flake is recorded rather than fixed: `event::tests::events_are_inert_when_no_profiler_is_running` reads global engine stats around an event and fails under the parallel harness when another test holds the engine. `ci/sanitizers.sh` runs `--test-threads=1`, which is why the sanitizer job does not see it.

---

## 10. Resolved decisions

**10.1 gmax exactness under concurrency — RESOLVED: exact.** Global readers-writer peak gate (§4.3). Shared acquire on every allocation event, exclusive on anything that could peak. `gb`/`gbk` exact under concurrency; differential testing keeps full multi-threaded strength. Correctness over throughput, per goal #1. The gate's shared path was itself unchecked until the M7 audit; the decision stands, and what was missing was a test that could tell whether the mechanism it bought was still there.

**10.2 x86_64 unwinding — RESOLVED: require frame pointers on unix; amended for Windows.** Hard startup error naming `-C force-frame-pointers=yes` and `CFLAGS=-fno-omit-frame-pointer` (§5.3). `Strategy::System` available as explicit opt-in with a cost warning, never selected automatically on unix. Never silently slow.

**Amendment [measured]:** the "never selected automatically" half does not hold on Windows, because there is nothing there to select. The Microsoft x64 ABI mandates unwind tables for every function, so the platform never needed a linked `rbp` chain, and `-C force-frame-pointers=yes` does not produce one that can be walked. Measured under Wine on `x86_64-pc-windows-gnu`, from the same stack at the same instant: hand-walking `rbp` yielded 2 entries with the flag and 1 without — the second a stack address, not a return address — while `RtlCaptureStackBackTrace` returned 9 plausible frames either way. Applying the decision as written meant the startup probe fired correctly and heapscope was unusable on Windows.

`Strategy::System` is therefore the **default on Windows**. The original reasoning does not carry across: `RtlCaptureStackBackTrace` is not a slow fallback to a fast path that exists, it is the platform's own mechanism, and unlike the unix system unwinder it needs no build flags at all. Its cost on real Windows is **unmeasured** — Wine timings say nothing about it.

The unix half is unchanged and has its own numbers (§9.1). On aarch64-apple-darwin the platform unwinder is *also* a frame-pointer walk, so it is **not** an escape hatch from missing frame pointers there. It is a real answer only where the problem actually arises: x86_64 Linux with frame pointers omitted.

**10.3 `mb`/`mbk` semantics — RESOLVED: true per-PP running max.** Diverges from Valgrind's global-peak-sampled value (§3.3); documented in the README and output header. The reference tracker encodes our definition.

**10.4 musl — RESOLVED: permanent non-goal.** Documented as never intended (§1.1). Not detected, not worked around, not in CI.

**10.5 `realloc` attribution — RESOLVED: resize attributed to the original PP.** Matches dhat-rs and how people reason about `Vec` growth. Semantics pinned in §4.6, including the instant reset and epoch participation.

**10.6 Dev-dependencies — RESOLVED: permitted; only the shipped library is std-only.** `[dependencies]` stays empty. MSRV drift is contained by splitting CI: the MSRV job checks the library and bins alone and never resolves dev-dependencies (§1.2).

**10.7 Crate name — RESOLVED: `heapscope`.** Verified unclaimed on crates.io **[measured]**. Chosen over `dha` (unsearchable — "DHA" is a fatty acid) and `dhat2` (version-numbered names age badly, and it implies a succession relationship to nnethercote's crate that does not exist).

**10.8 Bundled viewer — RESOLVED: yes.** Rationale and scope constraints in §6.12. DHAT v2 remains the primary interchange format.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Peak-gate shared acquire becomes the bottleneck under high thread counts | **Measured; the risk is real.** `benches/contention.rs` on 10 cores: per-event cost rises from 27 ns to 553 ns (1→16 threads), so aggregate throughput *falls* as cores are added. The cause is contended cache lines rather than the gate's exclusion, and the two lines are the minimum decision 10.1 permits — sharding was measured and does not pay (§9.1). Sampling is what actually reduces it. |
| Frame pointers absent in `cc`-built C/C++ dependencies despite correct `RUSTFLAGS` | `heapscope.captures` makes it visible rather than silent (§5.4). |
| Users cannot set the flag in their build system | `Strategy::System` opt-in, with the cost stated up front. |
| `dladdr` gives poor or NULL symbols | Tier 2 is the primary path (§6.1), not the fallback, and `heapscope-symbolize` makes it usable rather than merely possible. |
| Windows dbghelp is single-threaded and awkward | Symbolization is output-time only, behind one lock. **Open**: this crate and `std` hold different locks, and closing that needs a lock they share. |
| v0 demangler is 4× the estimated size | Budgeted at ~2,000 lines (§6.2), built at 1,656; fuzzing pulled forward to M4 and found six defects. |
| Overhead still too high for large workloads | **Built.** `sampling(bytes)` puts sample points on a Poisson process over the byte stream, so the decision is made before the stack walk that dominates the cost. It reduces the contention finding as a side effect sharding could not: an allocation that is not sampled takes neither contended line at all. |
| Users on pre-3.17 Valgrind viewers get a confusing error | **Closed.** The bundled viewer refuses a `formatVersion` it does not know and says so. |
| A dev-dependency raises its MSRV and breaks CI | MSRV job checks the library and bins alone (§1.2). |
| The viewer acquires a build step and quietly becomes a `node_modules` problem | Hand-written single file, no build step, enforced by `tests/no_dependencies.rs`, which fails the build if the repository grows a `package.json` or a `node_modules`. |
| Viewer scope creeps toward reproducing `dh_view.html` | Scope fixed in §6.12. **Held**: no flame graph, no diffing, no search. |
| A string a profile borrowed repaints a reader's terminal, reverses its display order, or ends the page | `push_display` screens at the point a string becomes output rather than at each producer. **Fuzzed**, which found the one field that had slipped the rule. |
| Scope creep into a debugger toolchain | M8 is explicitly optional and post-1.0. |

*Not a risk: "DHAT format v3 lands upstream." `kExpectedFileVersion` has been 2 since 2019 and `dh_view.js` has had no format change since.*

---

## 12. What "done" looks like for v1.0

Each bullet was audited at M7; what the audit found is in §9.1.

- `[dependencies]` empty, builds on MSRV 1.96 with the library-only check, clean on the four-platform matrix. **Audited**: the command all four `Test (…)` jobs run could not exit zero on any platform, and CI has never completed a run to notice. Fixed. **Still open**: no CI run has completed, so the matrix is verified locally only — natively on macOS aarch64, under Docker on both Linux targets, and under Wine for Windows.
- Profiles load in Valgrind's `dh_view.html` **and** in our bundled viewer, which requires no Valgrind install on any supported platform; trimmed traces never produce a colliding `fs`. **Audited**: the page was rendered in a real browser for the first time and found reporting two figures the profile deliberately omits; clicking it was then audited and seven broken controls had passed the entire repository. Both fixed. **Still open**: no Windows browser has opened the page, and the `dh_view.html` half remains node-only against a stub DOM.
- Multi-threaded differential tests prove the fast engine agrees exactly with the reference tracker, including `gb`/`gbk`. **Audited**: the exact comparison holds, with a qualification the tests already stated — under real threads it runs the engine *serialized*. What no test drove was the shared path's own decision, and deleting that check passed all 673 tests. Fixed with a peak whose value is known before the run starts. **Still open**: there is no exact field-by-field comparison under genuinely overlapping events, and there cannot be one — under a serial model the question has no answer.
- Deterministic mode records the same profile across runs — every program point, counter, frame, and their order — differing only in what names the process rather than the program. **Audited**: false. Three of four emitters wrote a different order every run. Fixed at the root and checked by `ci/check-reproducible.sh` against real processes.
- Every edge case in §4.6 has defined, tested behavior — no "may crash, hang, or otherwise do the wrong thing." **Audited** row by row, asking of each not whether a test exists but whether it asserts what the row says. Eleven of fourteen held; the three that did not are fixed.
- Per-capture and per-allocation overhead published from real measurements, with self-metrics in every profile so users can verify the **per-capture** claim themselves. **Audited**: every profile does carry the block, and the capture count is exact — but the timed figure was never compared to a stack walk, and a plausible constant passed the entire suite. Fixed with a second stopwatch. The wording is corrected too: the per-**allocation** figure is not derivable from a profile, because self-metrics time the stack walk and nothing else, so `benches/overhead.rs` backs that half and the profile backs the other.

**Remaining before 1.0** is evidence rather than features: a completed CI run on the four-platform matrix, a native Windows run (which is the only thing that can execute `SymFromAddr`), and a Windows browser opening the bundled page.
