//! The native profile format: everything recorded, in the shape it was
//! recorded in.
//!
//! PLAN.md section 3.4 makes this the source of truth and the DHAT v2 emitter
//! one lossy projection of it. That is not a preference about formats. DHAT v2
//! is fixed by a viewer this project does not own and has had no version 3 since
//! 2019, so every field this profiler learns to record either finds an existing
//! DHAT field that means something close enough, or has nowhere to go. This is
//! the somewhere.
//!
//! # What DHAT v2 cannot carry, and this does
//!
//! - **Addresses.** A DHAT frame is a *string*, so a profile's addresses survive
//!   only as text a tool has to parse back out of a rendering that was designed
//!   to be read. Here a frame is an address, an image, a file address, and a
//!   symbol, as four separate answers.
//! - **What the program asked for.** Sizes and alignments as distributions
//!   rather than a mean, the blocks it wanted zeroed, and what reallocation
//!   copied. See [`shape`](crate::internals::shape).
//! - **What the profiler cost.** Arena and table occupancy, and a measured
//!   per-capture time. PLAN.md section 12 promises honestly measured overhead;
//!   these are the numbers that make it checkable by the reader rather than by
//!   us.
//! - **Both lifetime totals separately.** DHAT's `tl` is one number, so the
//!   lifetime of blocks that were freed and the lifetime so far of blocks that
//!   were not have to be added together before they are written. Both are here.
//!
//! # Nothing is trimmed and nothing is folded
//!
//! Frame trimming and the emit-time program-point fold are both there to satisfy
//! `dh_view.html`: it refuses a file with two points on one frame list, and it
//! is unreadable when every stack begins with nine frames of runtime entry.
//! Neither is a fact about the run, so neither happens here. A tool reading this
//! can apply either — [`Trimmed`](crate::symbol::Trimmed) is public — and a tool
//! that needs the distinction between two call sites that render alike still has
//! it.
//!
//! # Versioning
//!
//! `formatVersion` is a single integer, and the compatibility rule is stated in
//! the file rather than in a document that travels separately from it: **a
//! reader must ignore fields it does not know, and must refuse a
//! `formatVersion` it does not know.** New fields are added without a bump. The
//! version moves only when the meaning of an existing field changes, which is
//! the case a reader cannot detect for itself.
//!
//! # Numbers, and the one place they are strings
//!
//! Counts and sizes are JSON numbers. **Addresses are strings**, in `0x` hex,
//! and that is the third trap in this crate's list of traps a viewer will not
//! warn you about: a JSON number is a double in JavaScript, which is exact only
//! to 2^53, and a 64-bit address is not. `JSON.parse` would round it silently,
//! and the bundled viewer of PLAN.md section 6.12 parses this file. An address
//! that is wrong in its low bits names the wrong line of the wrong function,
//! with nothing about it looking wrong.

use std::io::{self, Write};

use super::json::{JsonWriter, Layout};
use super::{PointKind, Snapshot};
use crate::symbol::{modules::Module, Resolved};

/// The format this writer produces.
///
/// A name in the file, so that a tool handed an unknown JSON document can tell
/// this from a DHAT file without guessing from which keys are present.
const FORMAT: &str = "heapscope-profile";

/// The version this writer produces. See the module documentation for what a
/// bump means.
const FORMAT_VERSION: u64 = 1;

/// Writes `snapshot` as a native profile.
pub(super) fn write<W: Write>(snapshot: &Snapshot, out: W) -> io::Result<()> {
    let mut json = JsonWriter::new(out);
    let mut hex = String::new();

    json.begin_object(Layout::Wrap)?;
    json.field_str("format", FORMAT)?;
    json.field_u64("formatVersion", FORMAT_VERSION)?;
    // Stated in the file, because a reader that has this file may not have this
    // documentation and the rule is what makes adding a field safe.
    json.field_str(
        "compatibility",
        "ignore unknown fields; refuse an unknown formatVersion",
    )?;
    json.field_str("producer", concat!("heapscope ", env!("CARGO_PKG_VERSION")))?;

    write_run(&mut json, snapshot)?;
    write_settings(&mut json, snapshot)?;
    write_totals(&mut json, snapshot)?;
    write_shapes(&mut json, snapshot)?;
    write_self_metrics(&mut json, snapshot)?;
    write_threads(&mut json, snapshot)?;
    write_regions(&mut json, snapshot)?;
    let frames = write_frames(&mut json, snapshot, &mut hex)?;
    write_points(&mut json, snapshot, &frames)?;
    write_modules(&mut json, snapshot, &mut hex)?;

    json.end_object()?;
    json.finish()?;
    Ok(())
}

/// What was recorded, and under what conditions.
fn write_run<W: Write>(json: &mut JsonWriter<W>, snapshot: &Snapshot) -> io::Result<()> {
    json.key("run")?;
    json.begin_object(Layout::Wrap)?;
    json.field_str("mode", snapshot.settings.mode.as_str())?;
    // `argv` is chosen by whoever started the process, so it gets the same
    // screening a symbol name gets. See `output::push_display`.
    let mut command = String::new();
    super::push_display(&mut command, &snapshot.command);
    json.field_str("command", &command)?;
    json.field_u64("pid", u64::from(snapshot.pid))?;
    json.field_str("shutdown", snapshot.shutdown.as_str())?;
    // Whether the per-point counters and the global ones were read in one
    // exclusive window. False means an event may have landed between them, so a
    // reader checking that the at-peak columns sum to the peak should expect
    // them not to.
    json.field_bool("exact", snapshot.exact)?;
    json.field_bool("poisoned", snapshot.poisoned)?;
    json.field_str("timeSource", snapshot.time_source.unit())?;
    json.field_str("timeUnit", snapshot.time_source.unit_long())?;
    json.field_u64("timeAtEnd", snapshot.time_at_end)?;
    // Omitted rather than zeroed in a mode that has no peak, for the reason
    // `bklt: false` omits DHAT's `tg`: an event was never live, so the instant
    // at which live bytes were greatest is not a measurement that exists.
    if snapshot.settings.mode.block_lifetimes() {
        json.field_u64("timeAtMax", snapshot.stats.time_at_max)?;
    }
    json.field_str("unwinder", snapshot.unwinder.as_str())?;
    json.end_object()
}

/// The settings the run had, as they took effect.
fn write_settings<W: Write>(json: &mut JsonWriter<W>, snapshot: &Snapshot) -> io::Result<()> {
    json.key("settings")?;
    json.begin_object(Layout::Inline)?;
    json.field_u64("maxDepth", snapshot.settings.max_depth as u64)?;
    json.field_u64("maxLiveBlocks", snapshot.settings.max_live_blocks as u64)?;
    // Present here and absent from the DHAT file, and the difference is not an
    // oversight. There it would describe a rendering the emitter may have been
    // overridden out of; here nothing is rendered, so it is what it says it is:
    // what the run's *default* output would do, which is a fact about the run.
    json.field_bool("trimFrames", snapshot.settings.trim_frames)?;
    // Present only on a run that sampled, so a reader that does not know the key
    // is looking at an exact profile rather than one whose rate it failed to
    // notice. A zero would have been the other way round: indistinguishable from
    // a reader that ignored the field.
    if let Some(interval) = snapshot.settings.sampling {
        json.field_u64("samplingInterval", interval.get())?;
    }
    json.end_object()
}

/// The global counters, and everything that did not fit.
fn write_totals<W: Write>(json: &mut JsonWriter<W>, snapshot: &Snapshot) -> io::Result<()> {
    let stats = &snapshot.stats;
    json.key("totals")?;
    json.begin_object(Layout::Wrap)?;
    json.field_u64("totalBytes", stats.total_bytes)?;
    json.field_u64("totalBlocks", stats.total_blocks)?;
    // Everything below describes blocks that were *live*, and an event never
    // was. Omitted rather than zeroed in a mode that has none, exactly as the
    // per-point `atEndBytes` is one level down — the rule was applied there and
    // not here at first, so an ad hoc profile reported "0 bytes live at the end"
    // and "the peak moved 0 times", which are the non-measurements the rule
    // exists to keep out.
    if snapshot.settings.mode.block_lifetimes() {
        json.field_u64("currBytes", stats.curr_bytes)?;
        json.field_u64("currBlocks", stats.curr_blocks)?;
        json.field_u64("maxBytes", stats.max_bytes)?;
        json.field_u64("maxBlocks", stats.max_blocks)?;
        // How many times the peak moved, which is also the epoch the lazy
        // at-peak algorithm is on. A reader comparing two runs of the same
        // workload can see from this whether they took the same path to the
        // same maximum.
        json.field_u64("peaks", stats.epoch)?;
    }
    json.end_object()?;

    // Everything the profiler could not record, in one place. A profile with a
    // number here is a profile whose other numbers are incomplete by exactly
    // that much, and grouping them is what makes that readable rather than
    // something to be assembled from five fields spread across the file.
    json.key("notRecorded")?;
    json.begin_object(Layout::Wrap)?;
    json.field_u64("blocks", stats.dropped_blocks)?;
    json.field_u64("programPoints", snapshot.points_dropped)?;
    // Attribution rows that appeared during the flush and did not fit. Non-zero
    // means the thread rows no longer sum to `totals`, which is the one thing a
    // reader checks them against, so it has to be in the file rather than
    // inferred from a sum that does not add up.
    json.field_u64("attributionRows", snapshot.rows_dropped)?;
    json.field_u64("unattributedBlocks", snapshot.unattributed_blocks)?;
    json.field_u64("refusedEvents", stats.refused_events)?;
    json.end_object()
}

/// What the program asked for, beyond a number of bytes.
pub(super) fn write_shapes<W: Write>(
    json: &mut JsonWriter<W>,
    snapshot: &Snapshot,
) -> io::Result<()> {
    let shapes = &snapshot.shapes;
    json.key("shapes")?;
    json.begin_object(Layout::Wrap)?;
    // Every request, including the ones the live-block table had no room to
    // track: `observedBlocks == totals.totalBlocks + notRecorded.blocks` in an
    // unsampled heap run. The histograms below each sum to it.
    //
    // On a sampled run the equality does not hold, and its two sides become
    // worth comparing rather than worth checking: this is an exact count of
    // requests, because counting a shape needs no stack walk, while
    // `totalBlocks` is an estimate from the fraction that were sampled. The
    // profile therefore carries both the truth and the estimate of the same
    // quantity, which is the only self-check a sampled profile can offer.
    json.field_u64("observedBlocks", shapes.observed_blocks)?;

    json.key("sizeClasses")?;
    json.begin_array(Layout::Wrap)?;
    for (floor, ceiling, blocks) in shapes.size_classes() {
        json.begin_object(Layout::Inline)?;
        json.field_u64("atLeast", floor as u64)?;
        json.field_u64("atMost", ceiling as u64)?;
        json.field_u64("blocks", blocks)?;
        json.end_object()?;
    }
    json.end_array()?;

    json.key("alignments")?;
    json.begin_array(Layout::Wrap)?;
    for (bytes, blocks) in shapes.alignments_used() {
        json.begin_object(Layout::Inline)?;
        json.field_u64("bytes", bytes as u64)?;
        json.field_u64("blocks", blocks)?;
        json.end_object()?;
    }
    json.end_array()?;

    // Blocks the program asked for through `alloc_zeroed`. Worth its own line
    // because `calloc` may return pages that are never faulted in, so a run
    // whose bytes are mostly zeroed has a resident size unrelated to its
    // allocated size — the first thing a reader gets wrong when a profile and
    // `ps` disagree.
    json.key("zeroed")?;
    json.begin_object(Layout::Inline)?;
    json.field_u64("blocks", shapes.zeroed_blocks)?;
    json.field_u64("bytes", shapes.zeroed_bytes)?;
    json.end_object()?;

    // What growth cost. `bytesCopied` is work the allocator did that appears
    // nowhere in the sizes the program asked for.
    json.key("reallocs")?;
    json.begin_object(Layout::Inline)?;
    json.field_u64("count", shapes.reallocs)?;
    json.field_u64("moved", shapes.reallocs_moved)?;
    json.field_u64("bytesCopied", shapes.bytes_copied)?;
    json.field_u64("bytesGrown", shapes.bytes_grown)?;
    json.field_u64("bytesShrunk", shapes.bytes_shrunk)?;
    json.end_object()?;

    json.end_object()
}

/// What the profiler cost the program it was measuring.
pub(super) fn write_self_metrics<W: Write>(
    json: &mut JsonWriter<W>,
    snapshot: &Snapshot,
) -> io::Result<()> {
    let metrics = &snapshot.metrics;
    json.key("selfMetrics")?;
    json.begin_object(Layout::Wrap)?;

    json.key("arena")?;
    json.begin_object(Layout::Inline)?;
    json.field_u64("bytesReserved", metrics.arena.bytes_reserved as u64)?;
    json.field_u64("bytesUsed", metrics.arena.bytes_used as u64)?;
    json.field_u64("chunks", metrics.arena.chunks as u64)?;
    json.field_u64("refused", metrics.arena.refused as u64)?;
    json.field_u64("limit", metrics.arena.limit as u64)?;
    json.end_object()?;

    // `bytesUsed` above minus the four `bytes` below is what the arena handed
    // out that its tables no longer point at: blocks abandoned by growth, plus
    // alignment padding. That subtraction is why no table reports a waste of
    // its own — one of them is built on a map that does not count it — and it
    // is only meaningful because each table reports *all* of its storage, which
    // is also why the thread and region rows have to appear here even though
    // they are two of the smallest numbers in the file.
    // `PpTable::bytes` left its frame lists out at first, and the difference
    // then reported live frame storage as abandoned: measured at 592 bytes of
    // "waste" for a nine-point run whose 74 frame slots are exactly 592 bytes.
    write_table(json, "programPoints", &metrics.program_points)?;
    write_table(json, "liveBlocks", &metrics.live_blocks)?;
    write_table(json, "threads", &metrics.threads)?;
    write_table(json, "regions", &metrics.regions)?;

    // Capture quality. `complete` is the only outcome that means the trace
    // reached the outermost frame; the rest are the ways a stack walk gives up.
    let captures = &snapshot.captures;
    json.key("captures")?;
    json.begin_object(Layout::Inline)?;
    json.field_u64("complete", captures.complete)?;
    json.field_u64("truncated", captures.truncated)?;
    json.field_u64("suspect", captures.suspect)?;
    json.field_u64("noFrames", captures.no_frames)?;
    json.end_object()?;

    // Raw numbers rather than a rate, because a rate would be a division this
    // writer did and a reader could not check. Multiplied by the capture counts
    // above, which are exact, this is the run's stack-walking time.
    //
    // Omitted entirely when nothing was measured — a process that never started
    // a profiler, or one whose clock could not resolve a batch. A zero here
    // would read as a free capture.
    let cost = &metrics.capture_cost;
    if cost.measured() {
        json.key("captureCost")?;
        json.begin_object(Layout::Inline)?;
        json.field_u64("nanos", cost.nanos)?;
        json.field_u64("captures", cost.captures)?;
        // How deep the measured stack was. A frame-pointer walk costs more per
        // capture the deeper the stack is, so a program whose stacks run deeper
        // than this pays more than the figure says. The scaling is roughly
        // linear in the frame count; PLAN.md section 5.1 puts it at about
        // 1.3 ns per frame on a 5 ns fixed cost, which is a benchmark of the
        // walk alone and reads low against what this calibration measures for a
        // whole `capture_with` call.
        json.field_u64("frames", cost.frames as u64)?;
        json.field_str("unwinder", cost.strategy.as_str())?;
        json.end_object()?;
    }

    json.end_object()
}

/// Who allocated. One row per thread that recorded anything.
///
/// The rows sum to `totals` — every recorded allocation belongs to exactly one
/// thread — **exactly**, whenever `run.exact` is true: the rows move under the
/// peak gate and are read in the same window the totals are, so there is no
/// interval for them to differ in. Where `run.exact` is false the flush could
/// not reach a quiet point, and nothing in the file is claimed to be
/// simultaneous. See `Engine::attribute`.
fn write_threads<W: Write>(json: &mut JsonWriter<W>, snapshot: &Snapshot) -> io::Result<()> {
    json.key("threads")?;
    json.begin_array(Layout::Wrap)?;
    for thread in &snapshot.threads {
        json.begin_object(Layout::Inline)?;
        json.field_u64("id", u64::from(thread.id))?;
        if thread.overflow {
            // Not one thread but every thread past the table's capacity. Named
            // rather than left to be inferred from an id, because a reader who
            // does not know the sentinel would read it as a thread.
            json.field_bool("overflow", true)?;
        } else {
            // Omitted on the shared row, not zeroed: it stands for many threads
            // with many first instants, so any single one would be a
            // measurement of something that does not exist. Same rule the
            // lifetime fields follow in a mode that has none.
            json.field_u64("firstSeen", thread.first_seen)?;
        }
        if let Some(name) = &thread.name {
            // A thread name comes from the platform, which promises nothing
            // about its contents, so it gets the same screening a symbol name
            // gets. See `output::push_display`.
            let mut safe = String::new();
            super::push_display(&mut safe, name);
            json.field_str("name", &safe)?;
        }
        write_tally(json, &thread.counts, snapshot)?;
        json.end_object()?;
    }
    json.end_array()
}

/// What for. One row per region name the program entered.
///
/// Empty for a run that used no regions. Unlike the thread rows these do
/// **not** sum to `totals`: an allocation made outside every region belongs to
/// no row, which is where most allocations in most programs happen.
fn write_regions<W: Write>(json: &mut JsonWriter<W>, snapshot: &Snapshot) -> io::Result<()> {
    json.key("regions")?;
    json.begin_array(Layout::Wrap)?;
    for region in &snapshot.regions {
        json.begin_object(Layout::Inline)?;
        json.field_u64("id", u64::from(region.id))?;
        if region.overflow {
            json.field_bool("overflow", true)?;
        } else {
            json.field_u64("firstSeen", region.first_seen)?;
        }
        if let Some(name) = &region.name {
            let mut safe = String::new();
            super::push_display(&mut safe, name);
            json.field_str("name", &safe)?;
        }
        json.field_u64("entries", region.entries)?;
        // Regions still open when the profile was written. Zero is the ordinary
        // answer; anything else says a guard outlived the profiler.
        json.field_u64("active", region.active)?;
        write_tally(json, &region.counts, snapshot)?;
        json.end_object()?;
    }
    json.end_array()
}

/// One row's counters, on the same terms as `totals`.
///
/// The live and peak figures are omitted rather than zeroed in a mode with no
/// live blocks, for the reason `totals` omits its own: an event was never live,
/// so "0 bytes live" would be a measurement of something that does not exist.
fn write_tally<W: Write>(
    json: &mut JsonWriter<W>,
    counts: &super::TallyStats,
    snapshot: &Snapshot,
) -> io::Result<()> {
    json.field_u64("totalBytes", counts.total_bytes)?;
    json.field_u64("totalBlocks", counts.total_blocks)?;
    if snapshot.settings.mode.block_lifetimes() {
        json.field_u64("currBytes", counts.curr_bytes)?;
        json.field_u64("currBlocks", counts.curr_blocks)?;
        // This row's own peak, not its share of the global one. The two are
        // different questions and only the first is answerable without putting
        // every row under the peak gate: `maxBytes` here is the most this
        // thread or region ever held at once, which may well have been at an
        // instant when the whole heap was nowhere near its maximum.
        json.field_u64("maxBytes", counts.max_bytes)?;
        json.field_u64("maxBlocks", counts.max_blocks)?;
    }
    Ok(())
}

fn write_table<W: Write>(
    json: &mut JsonWriter<W>,
    key: &str,
    usage: &super::TableUsage,
) -> io::Result<()> {
    json.key(key)?;
    json.begin_object(Layout::Inline)?;
    json.field_u64("entries", usage.entries as u64)?;
    json.field_u64("capacity", usage.capacity as u64)?;
    json.field_u64("bytes", usage.bytes as u64)?;
    json.end_object()
}

/// The distinct addresses in the profile, and where each program point's frames
/// are in that table — innermost first, the order they were captured in.
///
/// Deduplicated by address, because a profile's stacks share their outermost
/// frames with each other: the whole point of a table is that a program with a
/// thousand call sites still resolves each address once. On Windows that is also
/// a lock and a dbghelp call saved per repeat.
///
/// Shared with the HTML emitter, which writes one display name per entry of the
/// table this returns. Sharing the function rather than the convention is what
/// makes its indices address the same frames the profile's do; two
/// deduplications that merely agree today would be free to stop agreeing.
pub(super) fn frame_table(snapshot: &Snapshot) -> (Vec<usize>, Vec<Vec<u32>>) {
    use std::collections::HashMap;

    let mut index: HashMap<usize, u32> = HashMap::new();
    let mut order: Vec<usize> = Vec::new();
    let mut per_point: Vec<Vec<u32>> = Vec::with_capacity(snapshot.points.len());
    for point in &snapshot.points {
        let mut frames = Vec::with_capacity(point.frames.len());
        for &address in &point.frames {
            let id = *index.entry(address).or_insert_with(|| {
                order.push(address);
                (order.len() - 1) as u32
            });
            frames.push(id);
        }
        per_point.push(frames);
    }
    (order, per_point)
}

/// Writes the frame table, resolved as far as this process could, and returns
/// what [`frame_table`] said about where each point's frames are in it.
fn write_frames<W: Write>(
    json: &mut JsonWriter<W>,
    snapshot: &Snapshot,
    hex: &mut String,
) -> io::Result<Vec<Vec<u32>>> {
    let (order, per_point) = frame_table(snapshot);

    json.key("frames")?;
    json.begin_array(Layout::Wrap)?;
    for &address in &order {
        let resolved = crate::symbol::resolve(&snapshot.modules, address);
        write_frame(json, address, &resolved, hex)?;
    }
    json.end_array()?;
    Ok(per_point)
}

fn write_frame<W: Write>(
    json: &mut JsonWriter<W>,
    address: usize,
    resolved: &Resolved,
    hex: &mut String,
) -> io::Result<()> {
    json.begin_object(Layout::Inline)?;
    json.field_str("addr", push_hex_into(hex, address))?;
    // Absent rather than null where the address is in no known image. The two
    // are different claims and only the second is true: a profile that says
    // "unknown" has looked and failed, and one that says nothing has an address
    // that belongs to no image this process had mapped.
    if let Some(module) = resolved.module {
        json.field_u64("module", module as u64)?;
    }
    if let Some(file_address) = resolved.file_address {
        json.field_str("fileAddr", push_hex_into(hex, file_address))?;
    }
    if let Some(symbol) = &resolved.symbol {
        // The linker's own name, still mangled: demangling is a rendering
        // decision, `heapscope::demangle` is public, and a reader wanting the
        // raw name would otherwise have no way back to it. Screened, because
        // these bytes came out of a symbol table that may be truncated or
        // mismatched — the same reason every other borrowed string in the
        // output layer is.
        let mut name = String::new();
        super::push_display(&mut name, &symbol.name);
        json.field_str("symbol", &name)?;
        json.field_u64("symbolOffset", symbol.offset as u64)?;
    }
    json.end_object()
}

/// One entry per program point, with the counters exactly as recorded.
fn write_points<W: Write>(
    json: &mut JsonWriter<W>,
    snapshot: &Snapshot,
    frames: &[Vec<u32>],
) -> io::Result<()> {
    let lifetimes = snapshot.settings.mode.block_lifetimes();

    json.key("points")?;
    json.begin_array(Layout::Wrap)?;
    for (point, frames) in snapshot.points.iter().zip(frames) {
        let counters = &point.counters;
        json.begin_object(Layout::Inline)?;
        // What the point is. The synthetic overflow point has no frames, and
        // "no frames" otherwise means a stack that could not be walked — two
        // conditions with opposite remedies, so the file names which one.
        json.field_str(
            "kind",
            match point.kind {
                PointKind::Recorded => "recorded",
                PointKind::Overflow => "overflow",
            },
        )?;
        json.field_u64("totalBytes", counters.total_bytes)?;
        json.field_u64("totalBlocks", counters.total_blocks)?;
        if lifetimes {
            // Two numbers rather than DHAT's one. `retiredLifetime` counts only
            // blocks that were freed, which is what the engine accumulates;
            // `unretiredLifetime` is the life so far of blocks that were not,
            // measured to the snapshot. Their sum is DHAT's `tl`. Kept apart
            // because a site that allocates and holds is not the same as one
            // whose blocks were short-lived, and adding them is what makes the
            // two indistinguishable.
            json.field_u64("retiredLifetime", counters.total_lifetime)?;
            json.field_u64("unretiredLifetime", point.unretired_lifetime)?;
            json.field_u64("maxBytes", counters.max_bytes)?;
            json.field_u64("maxBlocks", counters.max_blocks)?;
            json.field_u64("atGmaxBytes", counters.at_gmax_bytes)?;
            json.field_u64("atGmaxBlocks", counters.at_gmax_blocks)?;
            json.field_u64("atEndBytes", counters.curr_bytes)?;
            json.field_u64("atEndBlocks", counters.curr_blocks)?;
        }
        json.key("frames")?;
        json.begin_array(Layout::Inline)?;
        for &frame in frames {
            json.u64(u64::from(frame))?;
        }
        json.end_array()?;
        json.end_object()?;
    }
    json.end_array()
}

/// Every image mapped into the process.
fn write_modules<W: Write>(
    json: &mut JsonWriter<W>,
    snapshot: &Snapshot,
    hex: &mut String,
) -> io::Result<()> {
    json.key("modules")?;
    json.begin_array(Layout::Wrap)?;
    for module in &snapshot.modules {
        write_module(json, module, hex)?;
    }
    json.end_array()
}

fn write_module<W: Write>(
    json: &mut JsonWriter<W>,
    module: &Module,
    hex: &mut String,
) -> io::Result<()> {
    json.begin_object(Layout::Inline)?;
    // Whatever the loader read out of the filesystem, screened for the same
    // reason a symbol name is.
    let mut path = String::new();
    super::push_display(&mut path, &module.path);
    json.field_str("path", &path)?;
    // `load` is what `atos -l` wants; `start` and `size` bound the executable
    // region a return address can be in; `bias` converts a runtime address to
    // an address in the file. See `symbol::modules` for why these are four
    // numbers and not one.
    json.field_str("load", push_hex_into(hex, module.image_base))?;
    json.field_str("start", push_hex_into(hex, module.start))?;
    json.field_u64("size", module.size as u64)?;
    json.field_str("bias", push_hex_into(hex, module.bias))?;
    if let Some(build_id) = &module.build_id {
        // Screened like the path above it. This crate's own producer renders
        // note bytes as hexadecimal and cannot make an unsafe one, but the
        // field is public on a public struct, and the rule `push_display`
        // states is that a string is screened where it becomes output rather
        // than where it was produced -- so that being safe never depends on
        // knowing every producer.
        let mut identity = String::new();
        super::push_display(&mut identity, build_id);
        json.field_str("buildId", &identity)?;
    }
    json.end_object()
}

/// Renders `value` into `scratch` and returns it, so that one buffer serves
/// every address in the file.
fn push_hex_into(scratch: &mut String, value: usize) -> &str {
    scratch.clear();
    super::push_hex(scratch, value);
    scratch
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internals::clock::TimeSource;
    use crate::internals::engine::{GlobalStats, Mode};
    use crate::internals::shape::{Shape, Shapes};
    use crate::output::{Counters, ProgramPoint, Shutdown};

    fn point(frames: &[usize], total_bytes: u64) -> ProgramPoint {
        ProgramPoint {
            kind: PointKind::Recorded,
            frames: frames.to_vec(),
            counters: Counters {
                total_bytes,
                total_blocks: 1,
                total_lifetime: 7,
                curr_bytes: 1,
                curr_blocks: 1,
                max_bytes: total_bytes,
                max_blocks: 1,
                at_gmax_bytes: 2,
                at_gmax_blocks: 1,
            },
            unretired_lifetime: 3,
        }
    }

    fn snapshot(points: Vec<ProgramPoint>) -> Snapshot {
        Snapshot {
            stats: GlobalStats {
                total_bytes: points.iter().map(|p| p.counters.total_bytes).sum(),
                total_blocks: points.len() as u64,
                ..GlobalStats::default()
            },
            points,
            command: String::from("test"),
            pid: 1,
            time_at_end: 100,
            time_source: TimeSource::Events,
            shutdown: Shutdown::Dropped,
            unwinder: crate::unwind::Strategy::FramePointer,
            ..Snapshot::default()
        }
    }

    fn emit(snapshot: &Snapshot) -> String {
        let mut buffer = Vec::new();
        write(snapshot, &mut buffer).expect("writing to a Vec cannot fail");
        String::from_utf8(buffer).expect("valid UTF-8")
    }

    #[test]
    fn a_profile_names_its_format_and_its_version() {
        let text = emit(&snapshot(vec![point(&[0x10], 100)]));
        assert!(text.contains(r#""format":"heapscope-profile""#), "{text}");
        assert!(text.contains(r#""formatVersion":1"#), "{text}");
        assert!(text.contains(r#""producer":"heapscope "#), "{text}");
    }

    /// The one place addresses are strings, and the reason is that
    /// `JSON.parse` would round them. A profile with an address above 2^53 has
    /// to come back out of the file bit for bit.
    #[test]
    fn addresses_are_hexadecimal_strings_rather_than_numbers() {
        let high = 0x7fff_dead_beef_1234usize;
        let text = emit(&snapshot(vec![point(&[high], 100)]));
        assert!(
            text.contains(r#""addr":"0x7fffdeadbeef1234""#),
            "an address was not written as an exact hexadecimal string: {text}"
        );
        assert!(
            !text.contains("9223090566172032000"),
            "an address was written as a JSON number, which is a double: {text}"
        );
    }

    /// Distinct call sites sharing outer frames must share table entries and
    /// keep distinct frame lists — the opposite of the DHAT emitter's fold,
    /// which is allowed to merge them because the viewer cannot tell them apart.
    #[test]
    fn frames_are_shared_and_points_are_not_folded() {
        let text = emit(&snapshot(vec![
            point(&[0x10, 0x20], 100),
            point(&[0x11, 0x20], 50),
        ]));
        assert_eq!(text.matches(r#""addr":"#).count(), 3, "{text}");
        assert!(text.contains(r#""frames":[0,1]"#), "{text}");
        assert!(text.contains(r#""frames":[2,1]"#), "{text}");
        assert_eq!(text.matches(r#""kind":"#).count(), 2);
    }

    /// Two points that a rendering would collapse onto one frame list stay two
    /// points here, because nothing renders.
    #[test]
    fn two_points_that_would_render_alike_stay_two_points() {
        let text = emit(&snapshot(vec![point(&[0x10], 100), point(&[0x20], 50)]));
        assert_eq!(text.matches(r#""kind":"recorded""#).count(), 2, "{text}");
    }

    #[test]
    fn the_overflow_point_says_which_kind_of_frameless_it_is() {
        let mut overflow = point(&[], 100);
        overflow.kind = PointKind::Overflow;
        let text = emit(&snapshot(vec![overflow, point(&[], 50)]));
        assert!(text.contains(r#""kind":"overflow","#), "{text}");
        assert!(text.contains(r#""kind":"recorded","#), "{text}");
        assert!(text.contains(r#""frames":[]"#), "{text}");
    }

    /// DHAT folds the two lifetimes together because it has one field. Losing
    /// the distinction is exactly the kind of loss this format exists to avoid.
    #[test]
    fn both_lifetime_totals_are_kept_apart() {
        let text = emit(&snapshot(vec![point(&[0x10], 100)]));
        assert!(text.contains(r#""retiredLifetime":7"#), "{text}");
        assert!(text.contains(r#""unretiredLifetime":3"#), "{text}");
    }

    /// A mode with no block lifetimes omits them here for the same reason
    /// `bklt: false` omits them from a DHAT file: an event was never live, and
    /// a zero would be a measurement of something that did not happen.
    #[test]
    fn a_run_without_block_lifetimes_omits_them() {
        let mut snapshot = snapshot(vec![point(&[0x10], 100)]);
        snapshot.settings.mode = Mode::AdHoc;
        let text = emit(&snapshot);

        for absent in [
            "retiredLifetime",
            "unretiredLifetime",
            "maxBytes",
            "maxBlocks",
            "atGmaxBytes",
            "atGmaxBlocks",
            "atEndBytes",
            "atEndBlocks",
            "timeAtMax",
        ] {
            assert!(
                !text.contains(absent),
                "an ad hoc profile carries `{absent}`, which it has no measurement for: {text}"
            );
        }
        assert!(text.contains(r#""totalBytes":100"#), "{text}");
    }

    #[test]
    fn what_the_program_asked_for_reaches_the_file() {
        let shapes = Shapes::new();
        shapes.record(Shape::of(24).aligned(8));
        shapes.record(Shape::of(24).aligned(8));
        shapes.record(Shape::of(4096).aligned(64).zeroed());

        let mut snapshot = snapshot(vec![point(&[0x10], 100)]);
        snapshot.shapes = shapes.snapshot();
        let text = emit(&snapshot);

        assert!(text.contains(r#""observedBlocks":3"#), "{text}");
        assert!(
            text.contains(r#"{"atLeast":16,"atMost":31,"blocks":2}"#),
            "{text}"
        );
        assert!(text.contains(r#"{"bytes":8,"blocks":2}"#), "{text}");
        assert!(
            text.contains(r#""zeroed":{"blocks":1,"bytes":4096}"#),
            "{text}"
        );
    }

    /// An empty class is a class the program never used, and a 64-bit process
    /// has sixty of them. Writing them out would put sixty lines of zero in
    /// every profile to say nothing.
    #[test]
    fn empty_size_classes_are_left_out() {
        let shapes = Shapes::new();
        shapes.record(Shape::of(24));
        let mut snapshot = snapshot(vec![point(&[0x10], 100)]);
        snapshot.shapes = shapes.snapshot();
        let text = emit(&snapshot);

        assert_eq!(text.matches(r#""atLeast":"#).count(), 1, "{text}");
    }

    /// A capture cost of zero would read as a free capture, which is why it is
    /// left out rather than written as one.
    #[test]
    fn an_unmeasured_capture_cost_is_absent_rather_than_zero() {
        let text = emit(&snapshot(vec![point(&[0x10], 100)]));
        assert!(!text.contains("captureCost"), "{text}");

        let mut measured = snapshot(vec![point(&[0x10], 100)]);
        measured.metrics.capture_cost = crate::unwind::Cost {
            nanos: 1_344,
            captures: 64,
            frames: 11,
            strategy: crate::unwind::Strategy::FramePointer,
        };
        let text = emit(&measured);
        assert!(
            text.contains(r#""captureCost":{"nanos":1344,"captures":64,"frames":11"#),
            "{text}"
        );
    }

    /// The rule a reader has to follow to be forward compatible travels with the
    /// file, because a reader may not have this crate's documentation.
    #[test]
    fn the_compatibility_rule_is_in_the_file() {
        let text = emit(&snapshot(vec![point(&[0x10], 100)]));
        assert!(text.contains("ignore unknown fields"), "{text}");
        assert!(text.contains("refuse an unknown formatVersion"), "{text}");
    }

    /// Strings that came from somewhere else — a symbol table, the filesystem,
    /// `argv` — reach a terminal through this file too.
    #[test]
    fn borrowed_strings_are_screened() {
        let mut snapshot = snapshot(vec![point(&[0x10], 100)]);
        snapshot.command = String::from("prog \u{1b}[2J \u{202e}gnp.eslaf");
        snapshot.modules = vec![Module {
            path: String::from("/tmp/\u{202e}exe"),
            image_base: 0x1000,
            start: 0x1000,
            size: 0x1000,
            bias: 0,
            build_id: None,
        }];
        let text = emit(&snapshot);

        assert!(
            !text.contains('\u{202e}'),
            "a bidi override survived: {text}"
        );
        assert!(!text.contains('\u{1b}'), "an escape survived: {text}");
        assert!(text.contains("\\\\u{202e}"), "{text}");
    }
}
