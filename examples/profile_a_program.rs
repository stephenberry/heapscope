//! Records a small workload and writes a profile.
//!
//! ```text
//! cargo run --release --example profile_a_program [output.json] [heap|ad-hoc|copy]
//! ```
//!
//! The result opens in Valgrind's `dh_view.html`. Frames carry the name the
//! running process knows an address by where it knows one, and always carry
//! `image + offset` for `atos`, `addr2line`, or `llvm-symbolizer` to resolve
//! afterwards — including on a machine that never ran this program.
//!
//! Three more files are written beside it, with `.native.json`, `.html` and
//! `.folded` in place of `.json`: the same reading of the engine, carrying
//! everything DHAT v2 has no field for; a self-contained page that opens on a
//! machine with no Valgrind; and folded stacks for whatever flame graph tool the
//! reader already has. All four come from one snapshot, so they cannot disagree.
//!
//! ```text
//! inferno-flamegraph < dhat-heap.folded > dhat-heap.svg
//! ```
//!
//! The mode argument exists so `ci/check-dhat-viewer.sh` can put a file of each
//! shape through the real viewer. A `bklt: false` profile omits seven per-point
//! fields and two top-level ones, so it is a different file for the viewer to
//! accept, not the same file with different numbers.

use std::collections::HashMap;
use std::hint::black_box;

// One line is the whole installation. Everything the program allocates from
// here on is recorded, including allocations made before `main` starts.
#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = heapscope::Profiler::builder();
    if let Some(path) = std::env::args().nth(1) {
        // `output` chooses and `also` adds, so this replaces the default
        // `dhat-heap.json` and then asks for the native file beside it.
        builder = builder
            .output(heapscope::Output::dhat_v2(&path))
            .also(heapscope::Output::native(sibling(&path, ".native.json")))
            .also(heapscope::Output::html(sibling(&path, ".html")))
            // `TotalBytes` rather than one of the live-block metrics, because
            // this example runs in all three modes and an ad hoc run has no
            // live blocks to report — `Output::folded` would refuse, which is
            // the intended behaviour and not what an example should demonstrate.
            .also(heapscope::Output::folded(
                sibling(&path, ".folded"),
                heapscope::FoldedMetric::TotalBytes,
            ));
    }
    let mode = match std::env::args().nth(2).as_deref() {
        None | Some("heap") => heapscope::Mode::Heap,
        Some("ad-hoc") => heapscope::Mode::AdHoc,
        Some("copy") => heapscope::Mode::Copy,
        Some(other) => return Err(format!("unknown mode {other:?}").into()),
    };
    let profiler = builder.mode(mode).build()?;

    // Three shapes of allocation behaviour, so the profile has something to
    // distinguish: one site that holds, one that churns, and one that grows.
    // In a non-heap mode none of it is recorded — the shim is a pass-through —
    // and the profile is made of the reports below instead.
    let held = build_index(2_000);
    churn(20_000);
    let grown = grow_by_pushing(50_000);

    // One helper reached from two places, which is the ordinary thing a call
    // tree exists to show and is also a fixture `ci/check-viewer-interaction.mjs`
    // depends on: closing a twisty can only distinguish a per-path open set
    // from a per-name one at a node whose name is a branch somewhere else, and
    // without this every node here has a unique name. The identity is the
    // return address rather than the name, so this holds on a platform that
    // symbolizes nothing as well as on one that names every frame.
    let shared = collect_from_first(mode, 300) + collect_from_second(mode, 300);
    black_box(shared);

    report(mode, held.len(), grown.len());

    println!(
        "indexed {} keys, grew a vector to {} bytes",
        held.len(),
        grown.len()
    );

    profiler.print_summary(10)?;
    Ok(())
}

/// `path` with `.native.json` in place of its `.json`.
///
/// A sibling rather than a subdirectory, so that the two files a run produces
/// stay together and neither can be picked up without the other being obvious.
/// `path` with `.json` replaced by `suffix`, so the three files of one run sit
/// beside each other under one name.
fn sibling(path: &str, suffix: &str) -> String {
    match path.strip_suffix(".json") {
        Some(stem) => format!("{stem}{suffix}"),
        None => format!("{path}{suffix}"),
    }
}

/// Reports what the run did, in whichever way the mode counts.
///
/// Two call sites rather than one, so a non-heap profile has a tree to render
/// rather than a single row — which is what the viewer is being asked to accept.
#[inline(never)]
fn report(mode: heapscope::Mode, keys: usize, grown: usize) {
    match mode {
        heapscope::Mode::Heap => {}
        heapscope::Mode::AdHoc => {
            for n in 0..keys {
                heapscope::event(1 + (n % 7) as u64);
            }
            report_more(mode, grown);
        }
        heapscope::Mode::Copy => {
            for n in 0..keys {
                heapscope::copied(64 + n % 192);
            }
            report_more(mode, grown);
        }
    }
}

/// A second call site, one frame deeper.
#[inline(never)]
fn report_more(mode: heapscope::Mode, count: usize) {
    for n in 0..count / 100 {
        match mode {
            heapscope::Mode::AdHoc => heapscope::event(100 + n as u64),
            heapscope::Mode::Copy => heapscope::copied(4_096 + n),
            heapscope::Mode::Heap => {}
        }
    }
}

/// Allocates and keeps: a map whose contents live to the end of the run.
#[inline(never)]
fn build_index(count: usize) -> HashMap<String, Vec<u32>> {
    let mut index = HashMap::with_capacity(count);
    for n in 0..count {
        index.insert(format!("key-{n:06}"), vec![n as u32; 4]);
    }
    index
}

/// Allocates and immediately frees: churn that must not move the peak.
#[inline(never)]
fn churn(rounds: usize) {
    for n in 0..rounds {
        let scratch: Vec<u8> = Vec::with_capacity(64 + (n % 192));
        black_box(&scratch);
    }
}

/// The two call sites that reach [`collect_into`] through [`collect_tail`].
///
/// Two frames of shared tail rather than one, deliberately. With a single
/// shared frame the repeated node is a *leaf*, which has no twisty and no
/// descendants to hide, so a per-name open set changes nothing on screen and
/// the check that depends on this is vacuous. Two makes the repeated node a
/// branch. They take different arguments so nothing folds them together.
/// The arithmetic on the way out of each of these is load bearing, and deleting
/// it as dead weight is the way this fixture breaks. `#[inline(never)]` stops
/// the compiler inlining a function; it does not stop it turning a call in tail
/// position into a jump, which leaves no frame at all. Measured: written as a
/// bare `collect_tail(count, 3)`, every one of these frames was gone from the
/// recorded stacks and the tree was `main` straight to `collect_into`. Using
/// the returned value keeps each call out of tail position, and the frame with
/// it.
#[inline(never)]
fn collect_from_first(mode: heapscope::Mode, count: usize) -> usize {
    collect_tail(mode, count, 3).wrapping_add(1)
}

#[inline(never)]
fn collect_from_second(mode: heapscope::Mode, count: usize) -> usize {
    collect_tail(mode, count, 5).wrapping_add(2)
}

#[inline(never)]
fn collect_tail(mode: heapscope::Mode, count: usize, width: usize) -> usize {
    collect_into(mode, count, width).wrapping_add(width)
}

/// Records in whichever way the mode counts, so all three pages carry this
/// shape. A heap-only helper here left the ad hoc and copy profiles with no
/// repeated branch at all, because their shim is a pass-through.
#[inline(never)]
fn collect_into(mode: heapscope::Mode, count: usize, width: usize) -> usize {
    let mut total = 0;
    for n in 0..count {
        match mode {
            heapscope::Mode::Heap => {
                let row: Vec<u32> = vec![n as u32; width];
                total += row.len();
                black_box(&row);
            }
            heapscope::Mode::AdHoc => {
                heapscope::event((width + n) as u64);
                total += width;
            }
            heapscope::Mode::Copy => {
                heapscope::copied(width * 8 + n);
                total += width;
            }
        }
    }
    total
}

/// Grows one allocation repeatedly, which the profiler attributes to the site
/// that created the vector rather than to whichever push happened to resize it.
#[inline(never)]
fn grow_by_pushing(count: usize) -> Vec<u64> {
    let mut growing = Vec::new();
    for n in 0..count {
        growing.push(n as u64);
    }
    growing
}
