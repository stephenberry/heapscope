# Output formats

Four formats, one reading. Ask for as many as you want: they come from a single reading of the engine, so they cannot disagree about the same run.

`Output::dhat_v2` writes the file Valgrind's `dh_view.html` opens, and is the default: the reader almost certainly has a viewer for it already.

`Output::native` writes a versioned JSON superset that is the source of truth, of which the DHAT file is one lossy projection.

`Output::html` writes one self-contained page: the native profile, with a viewer for it wrapped around it.

`Output::folded` writes folded stacks, for whatever flame graph tool you already have.

| | DHAT v2 | native |
|---|---|---|
| Frames | one rendered string per frame | address, image, file address, symbol, apart |
| Trimming and folding | applied, because the viewer needs them | never; neither is a fact about the run |
| Block lifetimes | one `tl`, the two summed | freed and still-alive kept apart |
| Sizes, alignments, zeroed, realloc cost | in the extension block | yes |
| Arena and table occupancy, capture cost | in the extension block | yes |
| Thread and region attribution | no field for it | one row each, with names |

Addresses are hexadecimal *strings* there. A JSON number is a double in JavaScript, exact only to 2^53, so `JSON.parse` would silently round a 64-bit address — and an address wrong in its low bits names the wrong line of the wrong function with nothing about it looking wrong.

```rust
let profiler = Profiler::builder()
    .output(Output::dhat_v2("target/dhat-heap.json"))
    .also(Output::native("target/profile.native.json"))
    .also(Output::html("target/profile.html"))
    .also(Output::folded("target/profile.folded", FoldedMetric::TotalBytes))
    .build()?;
```

## The bundled viewer

Valgrind does not exist on Windows and does not support Apple Silicon, so on two of the four supported platforms `dh_view.html` is not something you can be assumed to have. `Output::html` is the answer to that: one file, double-click to open, nothing fetched, no build step anywhere in its making.

It is a complement to the DHAT file rather than a replacement, and `dh_view.html` is better at the tree than it is. What it shows that DHAT structurally cannot is everything around the tree — which thread allocated what, which region, the distribution of sizes and alignments, what reallocation copied, and what the profiler itself cost — plus two things the format cannot express: the frames trimming left out, because the full stacks travel in the page, and how accurate a sampled run is, because the profile carries an exact count of requests beside the estimate of the same quantity.

The page carries the native profile verbatim, so it is also the data: the bytes between its two script tags are exactly the file `Output::native` writes.

DHAT v2 output remains the primary interchange format, so profiles stay shareable with anyone. Note that Valgrind releases before 3.17 (March 2021) ship a v1 viewer, which reports a v2 file as `data file is missing a field: mi` rather than as a version mismatch — which is one of the reasons the bundled viewer exists.

## Flame graphs

`Output::folded` writes the line-oriented format every flame graph tool reads: one line per distinct stack, outermost frame first, separated by `;`, with a count at the end.

```text
main;run;parse;Vec::with_capacity 1048576
```

Nothing downstream needs to know anything about this crate:

```sh
inferno-flamegraph < target/profile.folded > profile.svg
```

`speedscope`, `flamegraph.pl`, and the Firefox Profiler read the same file.

A folded file carries **one** number per stack, so which one is a parameter rather than a silent choice. Each of the four is a counter that sums to a figure the profile reports globally, so the flame graph's total width is checkable against the summary:

| `FoldedMetric` | Per stack | Sums to | The question |
|---|---|---|---|
| `TotalBytes` | `totalBytes` | `totals.totalBytes` | where allocation volume went |
| `TotalBlocks` | `totalBlocks` | `totals.totalBlocks` | where the *number* of allocations went |
| `PeakBytes` | `atGmaxBytes` | `totals.maxBytes` | what the peak was made of |
| `LiveBytes` | `atEndBytes` | `totals.currBytes` | what was still held at the end |

Asking for several is asking for several files, and they still come from one reading:

```rust
let profiler = Profiler::builder()
    .output(Output::folded("target/allocated.folded", FoldedMetric::TotalBytes))
    .also(Output::folded("target/leaked.folded", FoldedMetric::LiveBytes))
    .build()?;
```

`PeakBytes` is `atGmaxBytes` — what each site held *at the instant the whole heap was largest* — and not each site's own maximum, which is a real measurement that sums to nothing because the sites peaked at different moments. The two are one field apart, and the wrong one draws a flame graph wider than the peak it claims to show.

The last two are not measurements an ad hoc or copy run took: an event is never live and never dies. Asking for one is refused rather than written as a file of zeroes, which would read as a program that allocated nothing. `FoldedMetric::needs_block_lifetimes` is the check that predicts it.

Frames are trimmed and symbolized like everywhere else, so on Linux — where in-process symbolization names almost nothing — [resolve offline](symbolization.md) first or the flame graph is a tower of addresses.
