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

**Baselines, for the gate you cannot write a number for.** Nobody knows what the budget should be until they have measured it once:

```rust
heapscope::assert_baseline!("tests/baselines/parsing.txt");
```

The file is a handful of `key value` lines, recorded by running with `HEAPSCOPE_UPDATE_BASELINE=1` and committed alongside the test. Every figure in it is compared, and the ones that grew are named — so the number that moved shows up in the pull request as a line a reviewer reads, rather than as a threshold constant nobody looks at. The default tolerance is exact, which is the useful one under `TimeSource::Events`: none of these figures depends on a clock, so two runs of the same workload record the same numbers. A missing baseline **fails** rather than recording itself.

One constraint worth knowing before you write the second such test: there is one profiler per process, and it measures the whole process for as long as it is alive. `cargo test` runs a binary's tests concurrently, so budgets belong in an integration test of their own containing one `#[test]`.

Sampled runs are refused here rather than accommodated. [Every figure a sampled run produces is an estimate](performance.md#paying-less-on-purpose), including the peak, and comparing a budget against a draw from a distribution is a flaky test wearing a threshold.
