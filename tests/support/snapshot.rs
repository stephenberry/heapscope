//! Hand-built snapshots, for the four suites that never touch the engine.
//!
//! `tests/dhat_output.rs` and `tests/native_output.rs` build their profiles by
//! hand — a point whose capture found no frames, an address above 2^53, counters
//! at `u64::MAX` — because a recorded workload produces none of those on demand.
//! `tests/html_output.rs` and `tests/profile_fuzz.rs` do the same for the page.
//! All four had their own copy of some of this, and the copies had drifted:
//!
//! * one multiplied a lifetime out and another saturated it, which is the
//!   difference between a fixture that survives `u64::MAX` blocks and one that
//!   panics on them;
//! * one zeroed the shape histogram for a non-heap run and another did not,
//!   harmless only because the file that skipped it happens never to build a
//!   snapshot with shapes in it;
//! * the two page suites shared a third shape between them, with their own flat
//!   constants for a point's lifetime and peak block count.
//!
//! What a plausible program point looks like, what a run's totals have to add up
//! to, and what a mode means are not things four files should each hold an
//! opinion about.
//!
//! [`totals`] is the load-bearing part of that. A run's peak is the sum of what
//! the points held **at the peak** — their `at_gmax` columns — and not the sum
//! of their own peaks, which happened at different instants; both validators
//! read it that way. `tests/profile_fuzz.rs` had written the other rule out by
//! hand and `tests/html_output.rs` had left those columns at zero; both were
//! wrong on four of them. Deriving the totals from the points is what makes that
//! unwritable rather than merely written down.
//!
//! What stays local to each suite is what that suite is *about*: the module map
//! the DHAT frame renderer resolves against, the thread and region rows the
//! native validator sums, and the generated text the page has to survive.

#![allow(dead_code)]

use heapscope::output::{Counters, GlobalStats, PointKind, ProgramPoint, Shutdown, Snapshot};
use heapscope::{Mode, TimeSource};

/// One program point, with counters that are coherent with each other.
///
/// Every figure is a fraction of `total_bytes` or of `blocks` rather than a
/// constant, so that an emitter reading the wrong field writes a visibly wrong
/// number rather than one that happens to match its neighbour. Two numbers that
/// are equal by construction cannot do that: swapping them is a no-op on the
/// data, so the profile is byte-identical and no validator has anything to
/// compare.
///
/// The bytes columns are therefore three distinct fractions —
/// `curr_bytes` an eighth, `at_gmax_bytes` a quarter, `max_bytes` a half — and
/// the order is not free. It has to descend from the peak: the live and at-peak
/// figures are held against each other per row and per run, so raising the
/// at-peak figure rather than lowering the live one would put a row's live bytes
/// above its own peak. An emitter that swapped the two now draws four
/// complaints, two from each validator.
///
/// **The blocks columns are equal and have to be.** `curr_blocks` and
/// `at_gmax_blocks` are both `blocks.min(1)`, and `total_blocks` and
/// `max_blocks` are both `blocks`, so a swap within either pair is invisible
/// here and to both validators. That is a real gap in what these fixtures can
/// catch, and it is load-bearing rather than an oversight:
/// `tests/native_output.rs::shares` splits a total as
/// `(t - t/4 - t/8, t/4, t/8)`, whose major component is **not monotone** —
/// `shares(7)` gives it 6 and `shares(8)` gives it 5, and so on at every
/// multiple of eight. Give the two block totals different values and the major
/// thread row can be handed a live count above its own peak, which the row rules
/// reject. Measured, not assumed: `at_gmax_blocks: blocks.min(2)` fails
/// `every_emitted_profile_is_valid` with "`threads[]` holds 6 blocks, more than
/// its own peak of 5".
///
/// The bytes columns escape this because block counts are small integers where
/// those division jumps dominate, while byte counts are large and separated by a
/// factor of two. Separating the blocks means making `shares` monotone first,
/// which is a change to how attribution rows are built rather than to a
/// fixture.
///
/// Saturating rather than wrapping, because `blocks` reaches `u64::MAX` here —
/// which is what a long-running process's summed counters look like — and
/// because a summed lifetime saturates in the engine too.
pub fn point(frames: &[usize], total_bytes: u64, blocks: u64) -> ProgramPoint {
    ProgramPoint {
        kind: PointKind::Recorded,
        frames: frames.to_vec(),
        counters: Counters {
            total_bytes,
            total_blocks: blocks,
            total_lifetime: blocks.saturating_mul(17),
            curr_bytes: total_bytes / 8,
            curr_blocks: blocks.min(1),
            max_bytes: total_bytes / 2,
            max_blocks: blocks,
            at_gmax_bytes: total_bytes / 4,
            at_gmax_blocks: blocks.min(1),
        },
        unretired_lifetime: 3,
    }
}

/// The global counters a run that recorded exactly these points would report.
///
/// Summed rather than chosen, because that is the invariant both validators
/// check: the points account for the whole run. A hand-built snapshot is held to
/// it the same way a recorded one is.
///
/// `time_at_max` and `epoch` are non-zero so that an emitter writing a zero
/// where one of them belongs is visible rather than indistinguishable from an
/// unset field.
///
/// Private, and not because nobody outside would want it: `#![allow(dead_code)]`
/// is in scope for this whole module, so a `pub` helper that lost its last
/// caller would sit here unnoticed. This one has exactly one, below.
fn totals(points: &[ProgramPoint]) -> GlobalStats {
    let mut stats = GlobalStats {
        time_at_max: 42,
        epoch: 1,
        ..GlobalStats::default()
    };
    for point in points {
        stats.total_bytes += point.counters.total_bytes;
        stats.total_blocks += point.counters.total_blocks;
        stats.max_bytes += point.counters.at_gmax_bytes;
        stats.max_blocks += point.counters.at_gmax_blocks;
        stats.curr_bytes += point.counters.curr_bytes;
        stats.curr_blocks += point.counters.curr_blocks;
    }
    stats
}

/// A snapshot of a run that recorded exactly `points`, and nothing else.
///
/// `Snapshot` is `#[non_exhaustive]`, so this starts from an empty one and fills
/// in what a run has to have. That is the point of the attribute: the next field
/// the profiler learns to record does not break a single one of these suites.
///
/// Callers add what their own emitter is about — a module map, a shape
/// histogram, thread and region rows — on top of what comes back.
pub fn hand_built(points: Vec<ProgramPoint>) -> Snapshot {
    let mut snapshot = Snapshot::default();
    snapshot.stats = totals(&points);
    snapshot.shutdown = Shutdown::Dropped;
    // `Strategy::default()` is whatever the platform capture strategy is, so a
    // profile that left it alone would record a different unwinder on Windows
    // and the suites that assert on the name would follow it there. Both
    // callers write that name into their file. `TimeSource::default()` is
    // `Events` everywhere and is written for the reader rather than the
    // platform: it says which unit the times below are in.
    snapshot.unwinder = heapscope::unwind::Strategy::FramePointer;
    snapshot.time_source = TimeSource::Events;
    // A profile with program points recorded captures; the validators reject one
    // that claims points and no stack walks.
    snapshot.captures = heapscope::unwind::CounterSnapshot {
        complete: points.len() as u64,
        ..Default::default()
    };
    snapshot.time_at_end = 100;
    snapshot.points = points;
    snapshot.command = String::from("target/debug/example --flag");
    snapshot.pid = 4242;
    snapshot
}

/// The same snapshot as a run in `mode` would have produced it.
///
/// A non-heap run records events, and an event is never live and never dies, so
/// every live, peak, and lifetime counter is zero — in the global stats, in
/// every point, and in the shape histogram, which is built by the allocator shim
/// a non-heap run turns off entirely.
///
/// Zeroing them here rather than leaving them is what makes these tests about
/// the emitter: the file omits those columns, and a validator cross-checking it
/// against the totals would otherwise be comparing an absent column against a
/// number no event could have produced.
pub fn as_mode(mut snapshot: Snapshot, mode: Mode) -> Snapshot {
    snapshot.settings.mode = mode;
    if mode.block_lifetimes() {
        return snapshot;
    }
    snapshot.stats.curr_bytes = 0;
    snapshot.stats.curr_blocks = 0;
    snapshot.stats.max_bytes = 0;
    snapshot.stats.max_blocks = 0;
    snapshot.stats.time_at_max = 0;
    snapshot.shapes = Default::default();
    for point in &mut snapshot.points {
        point.counters.total_lifetime = 0;
        point.counters.curr_bytes = 0;
        point.counters.curr_blocks = 0;
        point.counters.max_bytes = 0;
        point.counters.max_blocks = 0;
        point.counters.at_gmax_bytes = 0;
        point.counters.at_gmax_blocks = 0;
        point.unretired_lifetime = 0;
    }
    snapshot
}
