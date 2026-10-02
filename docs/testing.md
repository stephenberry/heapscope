# Failing a test on the numbers

A profile is something a person reads. The other case is a number a *program* reads, so that "this parser allocates at most 64 KiB" runs on every commit instead of being something someone measured once and wrote in a comment.

```rust
#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

#[test]
fn parsing_stays_inside_its_budget() {
    let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();

    let mark = heapscope::HeapStats::get().unwrap();
    parse(FIXTURE);

    heapscope::assert_max_bytes!(64 * 1024);
    heapscope::assert_no_leaks!(since: mark);
}
```

**Every reading can refuse, and that is the design.** `HeapStats::get()` returns a `Result`, and the assertions fail rather than pass whenever the answer would be a guess: no profiler running, a run counting something other than allocations, a poisoned engine, a `fork` child holding its parent's counters, or a run whose live-block table filled up and whose totals are therefore missing however many blocks it turned away. A getter that returned zeros for any of those would turn every budget built on it into an assertion that *cannot fail* — the test whose profiler was never started passes silently, forever.

There is a sixth way to reach zeros, and it is not on that list because it is refused earlier. A program that never installed `heapscope::Alloc` as its `#[global_allocator]` records nothing, so `assert_max_bytes!(64 * 1024)` passed in a program that had just allocated 10 MiB. A reading is the wrong place to catch that — by then the run is over and the answer is still zero — so a heap run now refuses to **start** without the shim, naming the missing line.

**A failing assertion writes a profile.** "The budget was 64 KiB and the peak was 400 KiB" says a test failed; it does not say which call site spent the difference, which is the only thing anyone wants to know next. So a failure prints the heaviest program points to stderr and writes a DHAT file, and the panic message names it. A second failure in the same run gets a file of its own, because a message pointing at a profile another test has since overwritten is worse than no profile.

**Counting allocations, from the start or from a mark.** `assert_alloc_count!(3)` means exactly three. Read as a ceiling, a bare number would pass a run that allocated nothing, and nothing at the call site would say so. A ceiling is spelled out instead, as `<= 3`, where a reader can see that zero passes. Either form counts from when the profiler started, or from a mark read just before the code under test:

```rust
warm_up();
let mark = heapscope::HeapStats::get().unwrap();
compile(FIXTURE);

heapscope::assert_alloc_count!(since: mark, <= 4, "while compiling {name}");
```

A mark recording more allocations than the run has made was not taken from that run, and the count since it is unknown, so the assertion fails instead of counting zero. A failure since a mark still writes a profile of the whole run, because a mark holds totals and no program points, and its message says so. `assert_max_bytes!` takes no mark: when the run's peak was set before the mark, two readings of it say nothing about the stage's own peak.

**Baselines, for the gate you cannot write a number for.** Nobody knows what the budget should be until they have measured it once:

```rust
heapscope::assert_baseline!("tests/baselines/parsing.txt");
```

The file is a handful of `key value` lines, recorded by running with `HEAPSCOPE_UPDATE_BASELINE=1` and committed alongside the test. Every figure in it is compared, and the ones that grew are named — so the number that moved shows up in the pull request as a line a reviewer reads, rather than as a threshold constant nobody looks at. The default tolerance is exact, which is the useful one under `TimeSource::Events`: none of these figures depends on a clock, so two runs of the same workload record the same numbers. A missing baseline **fails** rather than recording itself.

One constraint worth knowing before you write the second such test: there is one profiler per process, and it measures the whole process for as long as it is alive. `cargo test` runs a binary's tests concurrently, so budgets belong in an integration test of their own containing one `#[test]`.

Sampled runs are refused here rather than accommodated. [Every figure a sampled run produces is an estimate](performance.md#paying-less-on-purpose), including the peak, and comparing a budget against a draw from a distribution is a flaky test wearing a threshold.
