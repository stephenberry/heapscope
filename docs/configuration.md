# Configuration

Everything a run does differently is chosen before it starts, because a limit that changed halfway through would make one profile describe two configurations.

```rust
use heapscope::{Mode, Output, Profiler, TimeSource};

let profiler = Profiler::builder()
    .time_source(TimeSource::Events)   // the default: deterministic and free
    .max_depth(24)                     // frames per allocation
    .max_live_blocks(4_000_000)        // ceiling on tracked live blocks
    .trim_frames(true)                 // the default; `false` keeps every frame
    .output(Output::dhat_v2("target/dhat-heap.json"))
    .also(Output::text_summary_to_stderr(20))
    .build()?;
```

`output` chooses where a profile goes and `also` adds a second destination; `no_output` writes nothing at all, for a program that would rather call `save_dhat_v2` itself. Every destination is written from one reading, so a summary and a file taken together never disagree about the same run. See [output formats](output-formats.md) for what each one carries.

Settings are adjusted to what is achievable rather than refused: a depth past the capture buffer is clamped to it, and a live-block ceiling is rounded up to what 64 shards can hold, so `4_000_000` becomes 4,194,304. Every profile records the settings that were actually in force, in `heapscope.settings`, so a file never has to be read against a guess about how it was made.

Two more settings have pages of their own: [`sampling`](performance.md#paying-less-on-purpose) trades exactness for most of the cost, and [`unwinder`](stack-capture.md) selects the platform's own unwinder over the frame-pointer walk.

## Counting something other than allocations

A `GlobalAlloc` shim sees allocations and nothing else. Two other modes let a program report what it did, and get back the same thing a heap profile gives — the call sites, ranked, with the machinery frames left out.

```rust
let profiler = Profiler::builder().mode(Mode::AdHoc).build()?;

for row in rows {
    heapscope::event(parse(row).cost());   // any weight you like
}
```

| Mode | Counts | Allocator shim |
|---|---|---|
| `Mode::Heap` | every allocation | on |
| `Mode::AdHoc` | `heapscope::event(weight)` | off |
| `Mode::Copy` | `heapscope::copied(bytes)` | off |

The weight means whatever the program says it means: cache misses, retries, rows parsed, bytes sent. The viewer labels ad hoc totals `units` and `events` rather than bytes and blocks, because those numbers have no unit of ours to give them.

A mode is a property of the whole run, not of a call, because a DHAT file carries one `mode` and the viewer labels every column from it. In either non-heap mode the shim records nothing: an allocation costs the reentrancy guard and two atomic loads, and then goes straight to the inner allocator. Neither `event` nor `copied` allocates. Calling the wrong one of the two does nothing and is counted in the profile as `heapscope.refusedEvents` — a silent no-op would turn a wrong mode into an empty profile with no explanation.

Neither function does anything when no profiler is running, so instrumentation can be left in place.
