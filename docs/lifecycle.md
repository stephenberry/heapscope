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
