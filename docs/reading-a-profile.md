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
  #0 main       1.2 MiB in 445 blocks (98.0%), 0 B live, peak 1.1 MiB
  #1 hs-worker  25.6 KiB in 29 blocks (2.0%), 0 B live, peak 25.0 KiB

heapscope regions
  parsing/lexing  32.1 KiB in 16 blocks (2.5%), 0 B live, peak 2.0 KiB
  parsing         28.0 KiB in 53 blocks (2.1%), 0 B live, peak 26.6 KiB
  worker          25.6 KiB in 28 blocks (2.0%), 0 B live, peak 25.0 KiB
```

`parsing/lexing` holds more than `parsing` despite being inside it, and that is the design rather than a mistake: an allocation belongs to the innermost region open and to that one only, so the rows partition the run instead of double-counting it. An outer region does **not** include what its inner ones recorded, because a name can be entered under different parents at different times and a tree built from that would be a shape the run never had.

A region is scoped to the calling thread and nests to any depth. A process-wide "current phase" would be worse than either: it attributes whatever a background thread happens to be doing to whichever phase some other thread is in.

Names are interned, so entering `"parsing"` a thousand times is one row that says it was entered a thousand times. Each row's peak is its own — the most that thread or region ever held at once, which may well have been at an instant when the whole heap was nowhere near its maximum. `region` costs two atomic loads and a branch when nothing is profiling, so instrumentation can be left in place.

### What no region covered

The last row of the regions section is what was allocated while no region was open on the allocating thread. The same probe again:

```text
heapscope regions
  parsing/lexing  32.1 KiB in 16 blocks (2.5%), 0 B live, peak 2.0 KiB
  parsing         28.0 KiB in 53 blocks (2.1%), 0 B live, peak 26.6 KiB
  worker          25.6 KiB in 28 blocks (2.0%), 0 B live, peak 25.0 KiB
  (no region)     1.1 MiB in 377 blocks (93.4%), 0 B live
```

It follows the same rule as every region row: a block belongs to whatever was innermost on its thread when it was allocated, and its free and every reallocation of it come back to that place, whichever thread performs them and whatever region is open by then. So with it, the rows partition the run, and four columns add up exactly to the run's own totals: bytes and blocks allocated, and bytes and blocks still live. The shared row for names past the region table's capacity is one of the rows being added.

The peaks do not add up and are not meant to. Each region's is its own, reached at its own moment, and the `(no region)` row has none at all: it is the run's totals less the region rows, read at one instant, rather than a row the profiler keeps. Keeping one would put several more atomic operations on a single process-wide counter into nearly every allocation of every program, regions or not, and a peak is the only figure the subtraction cannot give.

In the native format it is `outsideRegions`, a key of its own beside `regions` rather than an entry in it, so it can never be mistaken for a region the program happened to name `(no region)`, and a reader that predates it simply ignores it. The bundled viewer shows it as the last row of the region table, set apart.

### Reading the regions from inside the program

`heapscope::RegionBreakdown::get()` returns the same rows and remainder without taking a snapshot, which copies out every program point and every thread as well:

```rust
let breakdown = heapscope::RegionBreakdown::get()?;
for region in &breakdown.regions {
    println!("{:?}: {} allocations", region.name, region.counts.total_blocks);
}
println!("(no region): {} allocations", breakdown.outside_regions.total_blocks);
let lexing = breakdown.region("parsing/lexing");
```

It refuses rather than returning zeros in the cases `HeapStats::get()` does: nothing recording, a poisoned profiler, a `fork` child, a sampled run (whose rows are estimates, like everything else in it). It answers in every mode, saying which, because a region row means the same thing in each, and it carries `dropped_blocks` and `refused_events` from the same reading: the rows still add up when the live-block table overflows, but they undercount, and those say by how much.

Because the remainder is a subtraction, it reads the region rows and the totals at one instant, holding every recording thread still for as long as copying a few hundred rows takes. If something else holds the profiler that long, it refuses with `StatsError::NoQuietPoint` rather than subtracting across two moments. It allocates, so it is not for a signal handler, and a thread already inside the profiler is refused at once with `StatsError::InsideTheProfiler` rather than left to stall every other thread while it waits.

## A profile whose counts were restarted

A program that called `Profiler::reset` says so before any figure, because the restart changes what every figure means:

```text
  restarted  at 6,001 observed events; totals, peaks and lifetimes cover what followed
  carried    258.5 KiB in 4,001 blocks live at the restart, counted as live and not as allocated
```

That is `profile_a_program heap restart`, which restarts once its index is built. The totals, the peak and the lifetimes are the window's. The live figures are not: the index is still live, still counted at the call site that built it, and part of the window's peak. So a call site can hold more than it allocated, and one that allocated nothing since the restart still appears, with zero allocations and the bytes it carried. Its average lifetime counts only blocks it allocated in the window, because a block carried across has no lifetime the window saw begin.

Run-wide counters that qualify the live figures are not restarted: blocks the live-block table turned away may still be live, so that count stays over the whole run, and the profile records how much of it came before the window.

## What else is in there

- [`heapscope.overhead`](performance.md#what-the-profiler-cost) — this run's own memory and stack-walking cost, measured rather than estimated.
- [`heapscope.captures`](stack-capture.md) — how many stack walks came back whole, which is how much to trust the call sites.
- [`heapscope.shutdown`](lifecycle.md) — which path wrote the file, which two profiles of the same program can legitimately differ by.
- [`heapscope.reset`](lifecycle.md#leaving-a-warm-up-out) — present only when the counts were restarted: how many times, when the last one happened, and what was live then.
- `heapscope.settings` — the settings that were actually in force, after clamping.
