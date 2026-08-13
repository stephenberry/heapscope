# Reading a profile

Bytes and blocks per call site are what DHAT was built to carry. Everything on this page is something the shim already knows that the format has no field for.

## What the profile says beyond bytes and blocks

A DHAT file has one number per program point per quantity, and that is the whole of what it can say about an allocation. Three things the shim already knows have nowhere to go in it, so they go in the native format instead — and, being a handful of scalars, into a few lines of the DHAT file's own extension block.

```text
  allocated  4.3 MiB in 26,018 blocks
  commonest  128 B to 255 B (51.2% of 26,018 blocks)
  zeroed     16 B in 1 blocks
  reallocs   2,014, of which 11 moved and copied 63.9 KiB
```

That is a real reading of `profile_a_program`, which is why the zeroed line is one small block: the example barely uses `calloc`. In a program that does, the same line is the first thing to look at when a profile and `ps` disagree, because `calloc` may hand back pages that are never faulted in, and a run whose bytes are mostly zeroed has a resident size unrelated to its allocated size.

The distribution rather than a mean, because a program making a million 24-byte allocations and one 24 MB allocation has the same mean as one making two million 24-byte allocations, and they are not the same program. And what growth copied, because those bytes are real work the program paid for and they appear in none of the sizes it asked for.

## Which thread, and which phase

A stack trace says *where* an allocation happened. Two questions it cannot answer come up constantly, and neither has a field in DHAT v2.

**Which thread.** The same call site reached from four workers is one program point, so a profile as DHAT can express it cannot say that one of those threads is the one holding the memory. Every block carries the thread that allocated it, and a free brings that thread's live bytes down even when another thread performs it — otherwise every program that hands ownership across threads reports its producers as leaking everything they ever made.

Names come from the platform, so they are the strings `top -H`, `perf`, and a debugger show. `std::thread::Builder::name` pushes the name to the OS on every supported platform, so Rust's own names arrive; on Linux the kernel caps them at 15 bytes, and the profile reports what the kernel kept.

**Which phase.** Name one and the profile breaks the run down by it:

```rust
{
    let _region = heapscope::region("parsing");
    parse(&bytes)?;
    {
        let _region = heapscope::region("parsing/lexing");
        lex(&bytes)?;
    }
}
```

A real reading of `examples/lifecycle_probe`, which does that on its main thread and opens a third region on a worker it names `hs-worker`:

```text
heapscope threads
  #0 main       1.2 MiB in 444 blocks (98.0%), 0 B live, peak 1.1 MiB
  #1 hs-worker  25.6 KiB in 29 blocks (2.0%), 0 B live, peak 25.0 KiB

heapscope regions
  parsing/lexing  32.1 KiB in 16 blocks (2.5%), 0 B live, peak 2.0 KiB
  parsing         28.0 KiB in 53 blocks (2.1%), 0 B live, peak 26.6 KiB
  worker          25.6 KiB in 28 blocks (2.0%), 0 B live, peak 25.0 KiB
```

`parsing/lexing` holds more than `parsing` despite being inside it, and that is the design rather than a mistake: an allocation belongs to the innermost region open and to that one only, so the rows partition the run instead of double-counting it. An outer region does **not** include what its inner ones recorded, because a name can be entered under different parents at different times and a tree built from that would be a shape the run never had.

A region is scoped to the calling thread and nests to any depth. A process-wide "current phase" would be worse than either: it attributes whatever a background thread happens to be doing to whichever phase some other thread is in.

Names are interned, so entering `"parsing"` a thousand times is one row that says it was entered a thousand times. Each row's peak is its own — the most that thread or region ever held at once, which may well have been at an instant when the whole heap was nowhere near its maximum. `region` costs two atomic loads and a branch when nothing is profiling, so instrumentation can be left in place.

## What else is in there

- [`heapscope.overhead`](performance.md#what-the-profiler-cost) — this run's own memory and stack-walking cost, measured rather than estimated.
- [`heapscope.captures`](stack-capture.md) — how many stack walks came back whole, which is how much to trust the call sites.
- [`heapscope.shutdown`](lifecycle.md) — which path wrote the file, which two profiles of the same program can legitimately differ by.
- `heapscope.settings` — the settings that were actually in force, after clamping.
