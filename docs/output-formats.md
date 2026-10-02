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

### Frames are function names

A frame in a folded file is the name of the function and nothing else: `core::fmt::write`, not `0x1044c81f0: core::fmt::write+0x1c (/path/to/program+0x2c1f0)`. A flame graph merges frames by their text, so an address in the frame would split a function into one node per return address it was reached through, and an image path would make every label a path with a name somewhere at the end of it. Two stacks that render alike are written as one line with their counts summed.

Names are found and demangled exactly as in every other output, so a frame in the flame graph can be searched for in the text summary or the DHAT file of the same run. The hash a legacy-mangled name ends in is dropped; generic arguments are kept, because `Vec<u8>::push` and `Vec<String>::push` are different code and merging them is a choice a viewer's search can make and a file cannot undo.

A frame with no name is `[program+0x2c1f0]`: the image's file name and the frame's address as it appears in that file. That is the return address the stack walk recorded, the same number the native profile and `Symbolized` carry, so a symbolizer asked about it by hand should be asked about one byte earlier, the call itself; see [symbolization](symbolization.md). Unnamed frames stay apart from each other, rather than collapsing into one node per image. Where two loaded images share a file name, both are written with their whole path, so code from two different `libfoo.so` files is never merged. An address in no image is written as itself, `[0x1044c81f0]`.

Frames are trimmed as in every other output: the allocation path above a stack and the runtime entry below it are left out. Trimming reads names, and the names come from the running process, so on Linux, where in-process symbolization names almost nothing, a folded file written at record time is untrimmed and mostly `[program+0x…]` frames. `heapscope-symbolize profile.native.json -f folded` writes the same file from names an offline symbolizer found, trimmed by the same rules; see [symbolization](symbolization.md).

What a name alone gives up is the offset from the symbol, which is how a reader spots a name the platform matched from too far away, and resolvability: a folded file is a picture of the profile, and the native profile is the record. Folded output up to 0.1.0 used the addressed rendering, and it is one argument away for anyone who wants it:

```rust
use heapscope::symbol::Symbolized;

let file = std::fs::File::create("target/addressed.folded")?;
snapshot.write_folded_with(file, &Symbolized::new(&snapshot.modules), FoldedMetric::TotalBytes)?;
```
