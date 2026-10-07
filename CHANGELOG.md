# Changelog

## Unreleased

### Added

- `HeapStats::allocations_since(&mark)`: the count `assert_alloc_count!(since: mark)` checks, with `MarkError` for a mark that cannot be compared.
- `RegionBreakdown::totals`: the run's counters from the same instant as the rows.

### Changed

- Text summaries, including the one a failing assertion prints, render frames as function names (`FunctionNames`). Use `write_text_summary_with` and `Symbolized` for addresses.
- Text summaries leave out of the ranking any program point that allocated nothing since a `Profiler::reset`, and say how many were left out.
- A failing assertion's profile goes in `target/<profile>/heapscope/` under `cargo test`, not the working directory.

## 0.2.0 (2026-10-06)

### Added

- `Profiler::reset` restarts the counts to leave a warm-up out. Readings carry `resets`; profiles record the restart.
- `assert_alloc_count!`: `<= n` ceiling and `since: mark` forms.
- `RegionBreakdown::get` reads region rows without a snapshot.
- An `outside_regions` row for what no region covered, in snapshots, profiles and summaries.
- `symbol::FunctionNames` frame renderer.

### Changed

- Folded output renders frames as function names by default.
- Assertion macros accept only integer types, so a `bool` or `char` no longer compiles as a count.
- The native profile of a restarted run is `formatVersion` 2.

## 0.1.0 (2026-08-13)

Initial release.
