# When the profile gets written

A profile is written when the `Profiler` is dropped. If the program never drops it — because it ends in `std::process::exit`, or because the profiler lives in a `static` — an `atexit` handler writes one anyway. That covers the common cases where a heap profiler otherwise produces nothing at all: a panic (the profile of a program that crashed being the one most worth having), an exit from a worker thread, and a profiler that outlives `main`.

Every profile records which path produced it, in `heapscope.shutdown`:

| Value | Written by |
|---|---|
| `drop` | `Profiler::drop`, before process teardown begins |
| `atexit` | the exit handler, partway through teardown |
| `explicit` | a direct call to `Snapshot::save_dhat_v2` after stopping |
| `forked-child` | never automatically; a `fork` child disowns the parent's recording, and only an explicit `save_dhat_v2` in the child emits this |

The distinction is not bookkeeping. `atexit` handlers run last-in-first-out and share their list with C++ static destructors through `__cxa_atexit`, so a profile written from one is taken *after* whatever was registered later has already torn down. Two profiles of the same program taken by the two paths can legitimately differ, and the field is how you tell which you are holding.

## Leaving a warm-up out

A process records one run, and a stopped run does not start again: a second `Profiler::builder().build()` is refused with `StartError::AlreadyRecorded`. What a program can do instead is keep the one run going and restart its counts with `Profiler::reset` when the warm-up is done. Everything written afterwards describes the window since the last reset.

| What | After a reset |
|---|---|
| Totals: bytes and blocks allocated, globally, per call site, per thread, per region; size and alignment histograms; reallocation copies; events refused | Start again from zero |
| Live state: blocks and bytes live, globally and everywhere they are attributed | Kept. A block allocated before the reset and freed after it brings every figure down as it would have; a leak check still sees it |
| Peaks: the global maximum and when it happened, each call site's maximum, and the bytes each held at the global peak | Start again from what is live, as though the peak had just happened |
| Lifetimes | Count only blocks allocated after the reset. A block live across it is in none of the window's allocation counts (`totalBlocks`, DHAT's `tbk`), which an average lifetime is taken over, so it contributes no lifetime either. It is in the live and at-peak counts, because it is live |
| Time | Not reset. The profile records when the reset happened, on the same axis as everything else |
| Blocks the live-block table could not track, region entries, capture and overhead counters | Kept, over the whole run. A block the table turned away may still be live, and the overhead is the profiler's own |

The outputs written afterwards say so:

- `run.reset` in the native format, which is then version 2 (see [output formats](output-formats.md));
- `heapscope.reset` and a note in `cmd` in the DHAT file;
- a warning in the HTML page;
- a `restarted` line in the text summary.

Folded stacks cannot. The format is a stack and a number per line, with nowhere to put anything else, so a flame graph of a reset run is a flame graph of the window with nothing on it to say so. Ask for another output beside it where that matters.

A reading taken before a reset is from another window, and `HeapStats::resets` is how code subtracting its totals can tell. Its live figures carry across, so a [leak check](testing.md) from it still answers.

A reset is refused, and changes nothing, on a run that is not recording, in a `fork` child, on a poisoned profiler, from inside the profiler's own bookkeeping, and when other threads keep it from reaching a quiet point for as long as a shutdown waits. `Profiler` is neither `Send` nor `Sync`, so the reset is called from the thread that owns it, and it is safe while other threads allocate: it waits for a moment at which no counter is midway through moving and applies itself there in one step. An allocation in flight at that moment can still land on either side of it: for its lifetime, and for whether its size and alignment are in the window's histograms. A reset where the program is quiet gives an exact profile.

## The exits that write nothing

**Nothing is written for `_exit`, `abort`, or a fatal signal.** Those bypass the `atexit` list by definition, and no handler can see them. This is a stated limitation with a test for each case rather than something to discover when a profile is missing.

What a program can do about it is write the profile itself, before it goes: `Profiler::save_dhat_v2`, `save_native` or `save_html`. Those are usable with recording still going, so the file records `shutdown: running` — a point-in-time reading rather than a reading of the finished program, which for a process about to `_exit` is all there was ever going to be. Dropping the profiler first is the other remedy and gives an end-of-run profile instead; the field is how a reader tells the two apart.

That is a documented remedy, so the suite runs it: a probe saves all three formats and then calls `_exit`, another aborts, and every file has to be complete and valid afterwards. Nothing runs after `_exit` to flush a buffer or finish a rename, which makes it the sharpest available check that a `save_*` call is really done with the file when it returns. A remedy nobody executes is a remedy that stops working quietly.

**On Windows, `std::process::exit` is in that category too.** Rust implements it as a direct `ExitProcess` call, which terminates the process without walking the CRT's `atexit` list, and Windows provides no hook that would let an executable notice. Returning from `main` is unaffected on every platform, so a profiler kept in a `static` still works; a Windows program that ends in `std::process::exit` must drop its profiler or save first. `save_html` is usually the one to want there, since a Windows reader has no `dh_view.html` to open a DHAT file with. The test suite asserts the absence on Windows and the presence everywhere else, so this stops being true the moment the platform changes.

## `fork`

Forking a profiled process is safe. `pthread_atfork` handlers take every lock before the fork and reset them in the child, which then stops recording: the inherited counters belong to the parent, and so does the output file. A child that exits, or that drops the inherited `Profiler`, writes nothing.

Without this the failure is not a wrong number. The child inherits a lock held by a thread `fork` did not copy, so it can never be released, and the child's next allocation blocks forever — or, on Apple platforms, the process dies of `SIGKILL` with no message. Two cases remain unhandled and are documented rather than defended against: a second thread forking while the first is inside our own prepare handler, and a `fork` issued from a signal handler that interrupted a thread inside the shim.

## Signal handlers

A signal handler that allocates while it has interrupted a thread inside the allocator shim is safe: the reentrancy guard is already held, so the handler's allocation is forwarded to the inner allocator and not recorded. This is a designed property with a test that raises the signal from inside the shim deterministically, not a race it happens to win.
