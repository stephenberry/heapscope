//! `heapscope` — dynamic heap analysis for Rust.
//!
//! A heap profiler that records every allocation, attributes it to the call site
//! that made it, and writes a profile readable by Valgrind's DHAT viewer or by a
//! bundled single-file viewer. Every allocation is the default and not the only
//! setting: [`ProfilerBuilder::sampling`] trades exactness for most of the cost,
//! and says so in the profile rather than quietly.
//!
//! No *allocation* is recorded unless [`Alloc`] is the program's
//! `#[global_allocator]` — nothing else in the process can see one — so that
//! line comes first, and a heap run refuses to start without it rather than
//! reporting a profile of zeros:
//!
//! ```
//! #[global_allocator]
//! static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
//! ```
//!
//! It can also count something other than allocations. A run built with
//! [`Mode::AdHoc`] or [`Mode::Copy`] turns the allocator shim off entirely and
//! profiles what the program reports through [`event`](fn@crate::event) or
//! [`copied`] — the same call-site attribution, applied to whatever the program
//! says is worth counting.
//!
//! The shipped library has no dependencies outside the standard library.
//! Platform capabilities are reached through direct `extern "C"` declarations
//! against libraries the process already links.
//!
//! # Four formats, one reading
//!
//! [`Output::dhat_v2`] writes the file Valgrind's `dh_view.html` opens, and is
//! what a profiler writes when nobody says otherwise: the reader almost
//! certainly has a viewer for it already.
//!
//! [`Output::native`] writes everything DHAT v2 has no field for — frames as
//! addresses rather than rendered text, the distribution of sizes and
//! alignments the program asked for, what reallocation copied, and what the
//! profiler itself cost. It is the source of truth, and the DHAT file is one
//! lossy projection of it.
//!
//! [`Output::html`] writes one self-contained page: the native profile, and a
//! viewer for it. No build step, nothing fetched when it opens, so it works
//! from a `file://` URL on a machine with no network and no tooling — which
//! includes every Windows machine and every Apple Silicon one, because
//! `dh_view.html` comes from a tool that runs on neither.
//!
//! [`Output::folded`] writes folded stacks — one line per distinct stack,
//! outermost frame first, separated by `;`, with a count at the end. That is
//! what `inferno`, `flamegraph.pl`, `speedscope`, and the Firefox Profiler read,
//! and none of them has to know anything about this crate. A folded file carries
//! one number per stack, so [`FoldedMetric`] says which: each of the four sums
//! to a figure the profile reports globally, which makes a flame graph's total
//! width checkable against the summary.
//!
//! Ask for as many as you want: they come from a single reading of the engine,
//! so they cannot disagree.
//!
//! # Which thread, and which phase
//!
//! A stack trace says *where* an allocation happened. Every block also records
//! **which thread** made it — with the thread's name, read from the platform
//! while the thread is still alive — and **which phase**, where the program
//! named one with [`region`](fn@crate::region). Neither has a field in DHAT v2;
//! both are in the native format and the text summary.
//!
//! ```no_run
//! # fn parse() {}
//! # #[global_allocator]
//! # static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
//! let _profiler = heapscope::Profiler::builder().build().unwrap();
//! let _region = heapscope::region("parsing");
//! parse();
//! ```
//!
//! # A number a test can fail on
//!
//! The other thing a profile is for is a budget that holds. [`HeapStats::get`]
//! reads the counters mid-run, [`assert_max_bytes!`] and its siblings fail a
//! test on them, and a [`Baseline`] is the same gate without anyone having to
//! choose the number first: record what the program does today, commit the
//! file, and let the next run fail if it does more.
//!
//! Every one of those readings can **refuse**, which is the design rather than a
//! caveat — see the [`stats`] module. A reading that returned zeros when it did
//! not know would turn every assertion built on it into one that cannot fail.
//!
//! ```
//! # #[global_allocator]
//! # static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
//! # fn parse() {}
//! let _profiler = heapscope::Profiler::builder().no_output().build().unwrap();
//! parse();
//! heapscope::assert_max_bytes!(64 * 1024);
//! ```
//!
//! # Status
//!
//! Under construction. See `PLAN.md` in the repository for the development plan
//! and the milestone this crate has reached.
//!
//! # Requirements
//!
//! Frame pointers are required on x86_64 targets:
//!
//! ```text
//! RUSTFLAGS="-C force-frame-pointers=yes"
//! CFLAGS="-fno-omit-frame-pointer"   # for C/C++ dependencies built via `cc`
//! ```
//!
//! They are enabled by default on aarch64 (Apple and Linux). The profiler fails
//! at startup with a message naming the remedy rather than silently producing
//! empty or dramatically slower profiles.

// Unsafe code is pervasive and deliberate in this crate; every block must carry
// its own justification and be explicitly scoped.
#![deny(unsafe_op_in_unsafe_fn)]
// `forbid` rather than `deny`, and the difference is the escape hatch: `deny`
// can be lifted by an `#[allow(missing_docs)]` on the item, `forbid` makes that
// allow itself a hard error (E0453). The reason this is the right level *here*
// and not for the `deny`s around it is that a local exception to this lint has
// no legitimate use. "Not part of the supported surface" is already spelled
// `#[doc(hidden)]`, which does not trip the lint at all, so the only thing a
// local allow can express is an undocumented public item — the exact outcome
// the lint exists to prevent. Lifting it is now a visible one-line change to
// this attribute rather than a local one that reads as noise in a diff.
#![forbid(missing_docs)]
#![deny(missing_debug_implementations)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![warn(rust_2018_idioms)]
#![warn(unreachable_pub)]

// `#[doc(hidden)]` rather than private: integration tests, benchmarks, and the
// reference tracker all need these types, but none of them is part of the
// supported surface and none carries a stability promise. What *is* the
// supported surface is written down and enforced by `tests/public_surface.rs`,
// so growing it is a decision rather than an accident.
//
// Both carry `allow(rustdoc::private_intra_doc_links)`, and the reason is the
// `doc(hidden)` above. The lint is about a *rendered* page showing a link that
// goes nowhere; these pages are never rendered, and under
// `--document-private-items` — which is how anyone reads them — every one of
// those links resolves. The alternative is unlinking a working reference to
// satisfy a page nobody sees. The lint stays on for the supported surface,
// where it is checking the thing it is for.
#[doc(hidden)]
#[allow(rustdoc::private_intra_doc_links)]
pub mod internals;
#[doc(hidden)]
#[allow(rustdoc::private_intra_doc_links)]
pub mod unwind;

mod alloc;
pub mod baseline;
mod event;
pub mod output;
mod profiler;
mod region;
pub mod stats;
pub mod symbol;

pub use alloc::{engine, Alloc, CAPTURE_DEPTH};
pub use baseline::{Baseline, Regression, Tolerance};
pub use event::{copied, event};
pub use internals::clock::TimeSource;
pub use internals::engine::Mode;
pub use output::{FoldedMetric, Snapshot};
pub use profiler::{Output, Profiler, ProfilerBuilder, StartError, DEFAULT_OUTPUT_PATH};
pub use region::{region, Region};
pub use stats::{EventStats, HeapStats, StatsError};
pub use symbol::demangle;

// The bodies of the assertion macros. `#[macro_export]` puts a macro at the
// crate root whatever module it is written in, so what it expands to has to be
// reachable from there too.
#[doc(hidden)]
pub use baseline::__assert_baseline;
#[doc(hidden)]
pub use stats::{__assert_alloc_count, __assert_max_bytes, __assert_no_leaks};

/// The version of this crate, as reported in profile headers.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
