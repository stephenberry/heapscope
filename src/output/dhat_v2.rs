//! The DHAT file format version 2, as consumed by Valgrind's `dh_view.html`.
//!
//! The format has been at version 2 since 2019 and there is no version 3. What
//! follows is written against `dhat/dh_view.js` and `dhat/dh_main.c` from the
//! Valgrind source tree.
//!
//! # Three things the viewer will not warn you about
//!
//! 1. **`tl` is mandatory but unvalidated.** `checkPP` never looks for it, yet
//!    the tree builder reads `aPP.tl` unguarded. Omit it and the file loads
//!    perfectly while every average-lifetime cell renders `NaN`.
//! 2. **Frame sequences must be unique across program points.** Two points with
//!    the same `fs` make the viewer throw `data file contains a repeated
//!    location` and refuse the file. Interning happens on the hot path keyed by
//!    *raw addresses*, but emission renders those addresses to strings — and any
//!    rendering that is not injective (symbolization, frame trimming, depth
//!    truncation) can collapse two distinct points onto one frame list. Hence
//!    [`Folded`], which re-keys every point by its **final** frame list before
//!    anything is written.
//! 3. **`ftbl[0]` must be `"[root]"`** and no `fs` may refer to index 0. The
//!    viewer seeds its tree root with frame 0 and then appends the first point's
//!    frames to it.
//!
//! # One deliberate divergence: `mb` and `mbk`
//!
//! Valgrind assigns a program point's maximum only inside
//! `if (g_curr_bytes >= g_max_bytes)`, so a site that peaked at 4 MB while the
//! whole heap was small records a maximum of zero. We record a true per-point
//! running maximum instead, so our numbers legitimately differ from Valgrind's
//! for the same program. The profile says so in its own `heapscope` section
//! rather than leaving it to be discovered.

use std::collections::HashMap;
use std::io::{self, Write};

use super::json::{JsonWriter, Layout};
use super::{PointKind, Snapshot};
use crate::internals::engine::Mode;

/// The version this emitter writes, and the only version `dh_view.js` accepts.
const FILE_VERSION: u64 = 2;

/// The viewer's "short-lived" cutoff, in whatever unit `tu` names.
///
/// A program point whose average block lifetime is at or below this is offered
/// as a candidate for the "Total (blocks), short-lived" sort. Valgrind uses 500
/// instructions; we use 500 of our own time unit, which is the same statement
/// about scale rather than a calibrated number.
///
/// Deliberately not a [`ProfilerBuilder`](crate::ProfilerBuilder) setting. It
/// changes nothing about what is recorded — only which points `dh_view.html`
/// offers under one of its sorts — so a knob for it would be a knob on someone
/// else's user interface. The per-point lifetime figures a reader would use to
/// draw their own line are already in the file.
const SHORT_LIVED_THRESHOLD: u64 = 500;

/// Stands in for the frames of the overflow program point, which has none.
///
/// Shaped like a Valgrind frame string (`0xADDR: symbol (file:line)`) so the
/// viewer's columns line up, with a bracketed name in place of the address for
/// the same reason `[root]` uses one.
pub(super) const OVERFLOW_FRAME: &str =
    "[overflow]: allocations recorded after the program-point table filled up";

/// Stands in for the frames of a point whose stack could not be walked.
pub(super) const UNWALKABLE_FRAME: &str = "[unwalkable]: no frame pointer chain at this allocation";

/// How a stack becomes the text of a profile.
///
/// This is the seam between the profile and whatever knows how to name code:
/// nothing else in the emitter knows what an address means. The default,
/// [`RawAddresses`], names nothing at all and leaves symbolization to be done
/// later — possibly on another machine.
///
/// There are two questions, and they are here together because the second can
/// only be answered by whoever answered the first: deciding that a frame is
/// uninteresting means reading its name, and the name is whatever this trait
/// produced.
pub trait FrameFormat {
    /// Appends the rendering of `address` to `out`.
    ///
    /// `out` may already contain text; append, never clear.
    fn format(&self, address: usize, out: &mut String);

    /// Which of one stack's frames are worth showing, given every frame of it
    /// already rendered by [`format`](FrameFormat::format), innermost first.
    ///
    /// The default keeps all of them, so a renderer that has no opinion behaves
    /// as though this did not exist.
    /// [`Trimmed`](crate::symbol::Trimmed) is the implementation that has one.
    ///
    /// The emitter forces the answer into range and never lets it empty a stack
    /// that had frames, so an implementation cannot crash a profile or make a
    /// walked stack claim it was unwalkable. What it *can* do is hide frames,
    /// and the number hidden is written into the profile.
    fn keep(&self, frames: &[String]) -> std::ops::Range<usize> {
        0..frames.len()
    }
}

/// Renders frames as bare hexadecimal return addresses.
///
/// The rendering is `0x1044c81f0: ???`, which mirrors the shape Valgrind uses
/// (`0xADDR: symbol (file:line)`) so the viewer's columns line up, with `???`
/// standing in for the name.
#[derive(Clone, Copy, Debug, Default)]
pub struct RawAddresses;

impl FrameFormat for RawAddresses {
    fn format(&self, address: usize, out: &mut String) {
        push_hex(out, address);
        out.push_str(": ???");
    }
}

/// Appends `0x` and the lower-case hexadecimal form of `value`.
pub(crate) fn push_hex(out: &mut String, value: usize) {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    out.push_str("0x");
    let mut leading = true;
    for shift in (0..usize::BITS).step_by(4).rev() {
        let digit = (value >> shift) & 0xF;
        if digit == 0 && leading && shift != 0 {
            continue;
        }
        leading = false;
        out.push(DIGITS[digit] as char);
    }
}

/// A program point after its frames have been rendered and collisions merged.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Point {
    /// Indices into the frame table, innermost first, never containing 0.
    fs: Vec<u32>,
    total_bytes: u64,
    total_blocks: u64,
    total_lifetime: u64,
    max_bytes: u64,
    max_blocks: u64,
    at_gmax_bytes: u64,
    at_gmax_blocks: u64,
    at_end_bytes: u64,
    at_end_blocks: u64,
}

impl Point {
    /// Merges `other` into this point, which the two sharing a frame list makes
    /// indistinguishable in the output.
    ///
    /// Cumulative and point-in-time quantities add. Maxima do not: two points
    /// that each peaked at 4 MB, at different moments, did not jointly peak at
    /// 8 MB, so their sum is not a number anything justifies writing down.
    ///
    /// Taking the larger of the two is not enough either, and this is the part
    /// the plan got wrong. At t-gmax *both* points had their bytes live at the
    /// same instant, so the merged point demonstrably held `gb + gb` bytes at
    /// once — which can exceed either point's own maximum. The same argument
    /// applies at t-end. The merged maximum is therefore the largest of the
    /// three lower bounds we can actually prove:
    ///
    /// ```text
    /// mb = max(mb₁, mb₂, gb₁ + gb₂, eb₁ + eb₂)
    /// ```
    ///
    /// Still a lower bound on the truth — the real joint maximum is unknowable
    /// once the two points are indistinguishable — but never a number that
    /// contradicts the rest of the record.
    ///
    /// # Saturation
    ///
    /// Every sum saturates. `total_lifetime` is already a saturating sum by the
    /// time it arrives here — a long-running process genuinely can drive it to
    /// `u64::MAX`, and `Snapshot` is a public type whose fields anyone may set —
    /// so an unchecked `+=` would panic in a debug build and wrap in a release
    /// one. Either would happen inside `Profiler::drop`, where a panic during
    /// unwinding aborts the process, and the wrap is the worse of the two: it
    /// turns an implausibly large number into a plausibly small one.
    fn merge(&mut self, other: &Point) {
        self.total_bytes = self.total_bytes.saturating_add(other.total_bytes);
        self.total_blocks = self.total_blocks.saturating_add(other.total_blocks);
        self.total_lifetime = self.total_lifetime.saturating_add(other.total_lifetime);
        self.at_gmax_bytes = self.at_gmax_bytes.saturating_add(other.at_gmax_bytes);
        self.at_gmax_blocks = self.at_gmax_blocks.saturating_add(other.at_gmax_blocks);
        self.at_end_bytes = self.at_end_bytes.saturating_add(other.at_end_bytes);
        self.at_end_blocks = self.at_end_blocks.saturating_add(other.at_end_blocks);
        self.max_bytes = self
            .max_bytes
            .max(other.max_bytes)
            .max(self.at_gmax_bytes)
            .max(self.at_end_bytes);
        self.max_blocks = self
            .max_blocks
            .max(other.max_blocks)
            .max(self.at_gmax_blocks)
            .max(self.at_end_blocks);
    }
}

/// The whole file, resolved: a frame table plus points that index into it.
#[derive(Debug)]
struct Folded {
    /// Frame strings. Index 0 is always `"[root]"`.
    ftbl: Vec<String>,
    points: Vec<Point>,
    /// How many snapshot points were merged away by the fold.
    collisions: u64,
    /// How many captured frames the renderer chose not to show.
    trimmed: u64,
}

impl Folded {
    fn build(snapshot: &Snapshot, format: &dyn FrameFormat) -> Self {
        let mut ftbl = vec![String::from("[root]")];
        // Deliberately not seeded with `[root]`: a frame that renders to that
        // exact string must get its own index, because index 0 is reserved and
        // an `fs` containing it would confuse the viewer's tree root. Two equal
        // strings in the table are harmless — they are only ever displayed.
        let mut frame_ids: HashMap<String, u32> = HashMap::new();
        // Two buffers, because what a `FrameFormat` produces is not yet fit to
        // put in a file: it may contain a symbol read out of a corrupt table or
        // a path from the filesystem. `push_display` is what stands between
        // those and a reader; see its documentation for what it is guarding
        // against. Screening here rather than inside this crate's own formats
        // means the guarantee also covers a `FrameFormat` written by someone
        // else, and that the screened form is what gets interned, so two frames
        // differing only in an escaped character stay two frames.
        let mut raw = String::new();
        // Held across points so the strings keep their capacity; only the first
        // `point.frames.len()` entries are ever live.
        let mut rendered: Vec<String> = Vec::new();

        let mut points: Vec<Point> = Vec::with_capacity(snapshot.points.len());
        let mut by_frames: HashMap<Vec<u32>, usize> = HashMap::new();
        let mut collisions = 0;
        let mut trimmed = 0u64;

        for point in &snapshot.points {
            let shown = shown_frames(&point.frames, format, &mut raw, &mut rendered);
            trimmed += (point.frames.len() - shown.len()) as u64;

            let mut fs = Vec::with_capacity(shown.len());
            for frame in shown {
                let id = match frame_ids.get(frame) {
                    Some(&id) => id,
                    None => {
                        let id = u32::try_from(ftbl.len()).unwrap_or(u32::MAX);
                        ftbl.push(frame.clone());
                        frame_ids.insert(frame.clone(), id);
                        id
                    }
                };
                fs.push(id);
            }

            // A point with no frames would be written as `"fs": []`, which is a
            // row the viewer renders as nothing at all — indistinguishable from
            // a bug in the emitter, and hiding two conditions a reader needs to
            // know about. Both get a frame that says which one it is.
            if fs.is_empty() {
                let label = match point.kind {
                    PointKind::Overflow => OVERFLOW_FRAME,
                    PointKind::Recorded => UNWALKABLE_FRAME,
                };
                let id = match frame_ids.get(label) {
                    Some(&id) => id,
                    None => {
                        let id = u32::try_from(ftbl.len()).unwrap_or(u32::MAX);
                        ftbl.push(String::from(label));
                        frame_ids.insert(String::from(label), id);
                        id
                    }
                };
                fs.push(id);
            }

            let counters = &point.counters;
            let folded = Point {
                fs,
                total_bytes: counters.total_bytes,
                total_blocks: counters.total_blocks,
                total_lifetime: point.total_lifetime(),
                max_bytes: counters.max_bytes,
                max_blocks: counters.max_blocks,
                at_gmax_bytes: counters.at_gmax_bytes,
                at_gmax_blocks: counters.at_gmax_blocks,
                at_end_bytes: counters.curr_bytes,
                at_end_blocks: counters.curr_blocks,
            };

            match by_frames.get(&folded.fs) {
                Some(&at) => {
                    points[at].merge(&folded);
                    collisions += 1;
                }
                None => {
                    by_frames.insert(folded.fs.clone(), points.len());
                    points.push(folded);
                }
            }
        }

        let mut file = Folded {
            ftbl,
            points,
            collisions,
            trimmed,
        };
        file.canonicalize();
        file
    }

    /// Puts the file in a canonical order: heaviest point first, and frame table
    /// indices assigned in the order the points use them.
    ///
    /// Heaviest-first is a presentation choice — it is the column `dh_view.js`
    /// opens on. Reproducibility comes from the tiebreak, which is each point's
    /// position in [`Snapshot::points`](super::Snapshot::points): that order is
    /// a reading of what the program did rather than of where it was mapped.
    ///
    /// The tiebreak used to compare the points' rendered frame text, which was
    /// wrong in a way that stayed hidden because it was usually right. Every
    /// rendered frame begins with a runtime address, and two addresses in one
    /// image keep their relative order under any load bias — so the comparison
    /// only changes answer when the tied points come from *different* images, or
    /// when the bias pushes one address to a different number of hex digits.
    /// Both are ordinary, and neither would have been noticed as anything but a
    /// file that stopped diffing cleanly.
    fn canonicalize(&mut self) {
        // Decorated with the position each point arrived in, so the comparison
        // is a total order in its own right. Leaving the tiebreak implicit in
        // `sort_by`'s stability would put the guarantee one `sort_unstable_by`
        // away from being lost with nothing to notice it.
        let mut decorated: Vec<(usize, Point)> = std::mem::take(&mut self.points)
            .into_iter()
            .enumerate()
            .collect();
        decorated.sort_unstable_by(|(left_at, left), (right_at, right)| {
            right
                .total_bytes
                .cmp(&left.total_bytes)
                .then_with(|| right.total_blocks.cmp(&left.total_blocks))
                .then_with(|| left_at.cmp(right_at))
        });
        self.points = decorated.into_iter().map(|(_, point)| point).collect();

        let mut remap = vec![u32::MAX; self.ftbl.len()];
        let mut ordered = Vec::with_capacity(self.ftbl.len());
        ordered.push(std::mem::take(&mut self.ftbl[0]));
        for point in &mut self.points {
            for slot in &mut point.fs {
                let old = *slot as usize;
                if remap[old] == u32::MAX {
                    remap[old] = ordered.len() as u32;
                    ordered.push(std::mem::take(&mut self.ftbl[old]));
                }
                *slot = remap[old];
            }
        }
        self.ftbl = ordered;
    }
}

/// The frames of one stack, rendered, screened, and narrowed to what `format`
/// wants shown.
///
/// Every emitter needs the same four steps in the same order, and getting the
/// order wrong is silent: `keep` is a judgement about a *whole* stack, so every
/// frame has to be rendered before any of it is used, and the answer has to be
/// clamped before it indexes anything. Two emitters had this inline and a third
/// (`Output::html`, PLAN.md section 6.12) is coming, so it lives here.
///
/// `raw` and `rendered` are scratch owned by the caller, reused across program
/// points so the strings keep their capacity. Only the first `addresses.len()`
/// entries of `rendered` are ever live; nothing reads past that, so a longer
/// stack seen earlier cannot leak into a shorter one.
///
/// What comes back is the screened text, which is what gets interned and what a
/// reader sees. Screening before `keep` rather than after means the trimming
/// rules read the same bytes the file will carry, and that a `FrameFormat` this
/// crate did not write is covered by both.
pub(super) fn shown_frames<'a>(
    addresses: &[usize],
    format: &dyn FrameFormat,
    raw: &mut String,
    rendered: &'a mut Vec<String>,
) -> &'a [String] {
    for (at, &address) in addresses.iter().enumerate() {
        if rendered.len() == at {
            rendered.push(String::new());
        }
        raw.clear();
        format.format(address, raw);
        rendered[at].clear();
        super::push_display(&mut rendered[at], raw);
    }
    let stack = &rendered[..addresses.len()];
    let keep = clamp_frames(format.keep(stack), stack.len());
    &stack[keep]
}

/// `keep`, forced to be a usable range over `len` frames.
///
/// [`FrameFormat::keep`] is answered by whoever supplied the renderer, and the
/// emitter runs inside `Profiler::drop`, where a panic while unwinding aborts
/// the process — so an out-of-range answer is corrected rather than indexed
/// with. Two corrections, and both are about honesty rather than safety:
///
/// - A range that keeps **nothing** from a stack that had frames is widened to
///   keep the innermost. The emitter labels a frameless point `[unwalkable]`,
///   which says the stack could not be walked; a stack that was walked must not
///   be made to say that.
/// - A range that starts past the end keeps the outermost frame rather than
///   silently keeping nothing.
pub(super) fn clamp_frames(keep: std::ops::Range<usize>, len: usize) -> std::ops::Range<usize> {
    if len == 0 {
        return 0..0;
    }
    let start = keep.start.min(len - 1);
    let end = keep.end.clamp(start + 1, len);
    start..end
}

/// Writes `snapshot` as a DHAT version 2 file.
pub(super) fn write<W: Write>(
    snapshot: &Snapshot,
    format: &dyn FrameFormat,
    out: W,
) -> io::Result<()> {
    let file = Folded::build(snapshot, format);
    let mut json = JsonWriter::new(out);

    // What the run counted decides the shape of the whole file. The viewer
    // reads `bklt` and then *requires* one set of fields and ignores another, so
    // this is not a label: emitting a lifetime for an event that never lived
    // would be a measurement where there is none, and omitting one for a heap
    // block makes the file fail to open.
    let mode = snapshot.settings.mode;
    let lifetimes = mode.block_lifetimes();

    json.begin_object(Layout::Wrap)?;
    json.field_u64("dhatFileVersion", FILE_VERSION)?;
    json.field_str("mode", mode.as_str())?;
    json.field_str("verb", mode.verb())?;
    // Block accesses: no, in every mode, and there is no way for a `GlobalAlloc`
    // shim to know them without instrumenting every load and store the way
    // Valgrind does.
    json.field_bool("bklt", lifetimes)?;
    json.field_bool("bkacc", false)?;
    // Ad hoc weights are dimensionless, so the viewer is told not to call them
    // bytes. Omitted where they match what the viewer already assumes, which
    // includes copy mode: it really is counting bytes.
    let units = mode.units();
    if units != Mode::DEFAULT_UNITS {
        json.field_str("bu", units.0)?;
        json.field_str("bsu", units.1)?;
        json.field_str("bksu", units.2)?;
    }
    json.field_str("tu", snapshot.time_source.unit())?;
    json.field_str("Mtu", snapshot.time_source.unit_million())?;
    if lifetimes {
        json.field_u64("tuth", SHORT_LIVED_THRESHOLD)?;
    }
    // `argv` is chosen by whoever started the process, so it gets the same
    // screening the frame names get.
    let mut command = String::new();
    super::push_display(&mut command, &snapshot.command);
    json.field_str("cmd", &command)?;
    json.field_u64("pid", u64::from(snapshot.pid))?;
    json.field_u64("te", snapshot.time_at_end)?;
    if lifetimes {
        json.field_u64("tg", snapshot.stats.time_at_max)?;
    }

    json.key("pps")?;
    json.begin_array(Layout::Wrap)?;
    for point in &file.points {
        json.begin_object(Layout::Inline)?;
        json.field_u64("tb", point.total_bytes)?;
        json.field_u64("tbk", point.total_blocks)?;
        if lifetimes {
            json.field_u64("tl", point.total_lifetime)?;
            json.field_u64("mb", point.max_bytes)?;
            json.field_u64("mbk", point.max_blocks)?;
            json.field_u64("gb", point.at_gmax_bytes)?;
            json.field_u64("gbk", point.at_gmax_blocks)?;
            json.field_u64("eb", point.at_end_bytes)?;
            json.field_u64("ebk", point.at_end_blocks)?;
        }
        json.key("fs")?;
        json.begin_array(Layout::Inline)?;
        for &frame in &point.fs {
            json.u64(u64::from(frame))?;
        }
        json.end_array()?;
        json.end_object()?;
    }
    json.end_array()?;

    json.key("ftbl")?;
    json.begin_array(Layout::Wrap)?;
    for frame in &file.ftbl {
        json.string(frame)?;
    }
    json.end_array()?;

    // Everything DHAT v2 has no field for. The viewer checks only that the
    // fields it knows are present, so an extra object is ignored there and
    // available to our own tooling.
    json.key("heapscope")?;
    json.begin_object(Layout::Wrap)?;
    json.field_str("version", env!("CARGO_PKG_VERSION"))?;
    json.field_bool("exact", snapshot.exact)?;
    json.field_bool("poisoned", snapshot.poisoned)?;
    // Which path produced this file. `drop` and `atexit` take their readings at
    // different points in process teardown, so two profiles of the same program
    // can legitimately differ; this is how a reader tells which they have.
    json.field_str("shutdown", snapshot.shutdown.as_str())?;
    // Which unwinder produced these frames. The two do not agree about frame
    // count or about where a trace stops, so a reader comparing profiles needs
    // to know whether they are comparing like with like.
    json.field_str("unwinder", snapshot.unwinder.as_str())?;
    json.field_str("mbSemantics", "per-program-point running maximum")?;
    // What the program asked for, and what the profiler cost, written by the
    // native emitter's own functions rather than by copies of them. PLAN.md
    // section 6.7 says *every* profile carries the self-metrics, and two
    // writers producing the same block is how one of them ends up missing a
    // field that was added to the other.
    //
    // The rest of the native format is not projected here, because it is
    // per-point and the DHAT points above are already the projection of it.
    super::native::write_shapes(&mut json, snapshot)?;
    super::native::write_self_metrics(&mut json, snapshot)?;
    json.field_u64("foldedPoints", file.collisions)?;
    // Frames the renderer chose not to show. Trimming makes a profile readable
    // by removing the allocation path and the runtime entry, and a reader
    // comparing this file against a stack they walked themselves needs to know
    // it happened rather than deduce it.
    json.field_u64("trimmedFrames", file.trimmed)?;
    // What the run was configured to do. A ceiling that silently changed the
    // profile — a depth limit that cut the call site off, a live-block ceiling
    // that dropped events — is indistinguishable from a program that behaved
    // that way, unless the file says which limits were in force. Both are
    // recording settings, read back from the engine, so both describe this file.
    //
    // `trim_frames` is deliberately absent. It is a *rendering* setting, and
    // this emitter renders with whatever it was handed: a snapshot from a
    // default profiler written through `write_dhat_v2_with(&Symbolized::new(..))`
    // would have reported `"trimFrames":true` next to `"trimmedFrames":0`,
    // which is the exact contradiction this block exists to prevent
    // **[measured]**. What this file did is `trimmedFrames`, below, and that
    // number is produced by the rendering rather than asserted about it.
    json.key("settings")?;
    json.begin_object(Layout::Inline)?;
    json.field_u64("maxDepth", snapshot.settings.max_depth as u64)?;
    json.field_u64("maxLiveBlocks", snapshot.settings.max_live_blocks as u64)?;
    // Sampling belongs here rather than with the rendering settings above: it
    // changes what every number in this file *means*, and a DHAT viewer showing
    // a sampled profile shows estimates whether or not it knows the word. The
    // key is absent on an exact run, so its presence is the whole signal.
    if let Some(interval) = snapshot.settings.sampling {
        json.field_u64("samplingInterval", interval.get())?;
    }
    json.end_object()?;
    json.field_u64("droppedPoints", snapshot.points_dropped)?;
    json.field_u64("droppedBlocks", snapshot.stats.dropped_blocks)?;
    json.field_u64("unattributedBlocks", snapshot.unattributed_blocks)?;
    // `heapscope::event` or `heapscope::copied` called during a run that counts
    // the other kind. Reported rather than dropped in silence, because the
    // symptom is otherwise a profile with nothing in it and nothing to say the
    // calls were made.
    json.field_u64("refusedEvents", snapshot.stats.refused_events)?;
    json.key("totals")?;
    json.begin_object(Layout::Inline)?;
    json.field_u64("totalBytes", snapshot.stats.total_bytes)?;
    json.field_u64("totalBlocks", snapshot.stats.total_blocks)?;
    json.field_u64("maxBytes", snapshot.stats.max_bytes)?;
    json.field_u64("maxBlocks", snapshot.stats.max_blocks)?;
    json.field_u64("currBytes", snapshot.stats.curr_bytes)?;
    json.field_u64("currBlocks", snapshot.stats.curr_blocks)?;
    json.end_object()?;

    // The module map. Without it the addresses in `ftbl` are uninterpretable
    // the moment this process exits, because the next run maps everything
    // somewhere else. With it, `atos -o <path> -l <load>`, `addr2line -e
    // <path>`, and `llvm-symbolizer` all resolve them — here, or on another
    // machine, or against an archived build a year from now.
    json.key("modules")?;
    json.begin_array(Layout::Wrap)?;
    for module in &snapshot.modules {
        json.begin_object(Layout::Inline)?;
        // The one string in the profile that nobody in the process chose: it is
        // whatever the loader read out of the filesystem. Screened for the same
        // reason the frames are, and separately, because a frame is only
        // rendered for an address that was recorded while every image gets an
        // entry here whether it was ever executed or not.
        let mut path = String::new();
        super::push_display(&mut path, &module.path);
        json.field_str("path", &path)?;
        // `load` is what `atos -l` wants; `start`/`size` bound the executable
        // region a return address can be in; `bias` converts a runtime address
        // to an address in the file. See `symbol::modules` for why these are
        // three numbers and not one.
        json.field_u64("load", module.image_base as u64)?;
        json.field_u64("start", module.start as u64)?;
        json.field_u64("size", module.size as u64)?;
        json.field_u64("bias", module.bias as u64)?;
        // Absent rather than null when the platform or the build gave none: a
        // profile that says "unknown" and one that says nothing are different
        // claims, and only the second is true here.
        if let Some(build_id) = &module.build_id {
            // Screened like the path above it, and for the reason spelled out
            // at the same field in `native.rs`: the screen belongs where a
            // string becomes output, not where it was produced.
            let mut identity = String::new();
            super::push_display(&mut identity, build_id);
            json.field_str("buildId", &identity)?;
        }
        json.end_object()?;
    }
    json.end_array()?;
    json.end_object()?;

    json.end_object()?;
    json.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internals::clock::TimeSource;
    use crate::output::{Counters, ProgramPoint, Shutdown};

    /// A frame format that renders every address to the same string, which is
    /// the worst case the fold exists to handle: symbolization and frame
    /// trimming both do exactly this to unrelated call sites.
    struct AllTheSame;
    impl FrameFormat for AllTheSame {
        fn format(&self, _address: usize, out: &mut String) {
            out.push_str("same");
        }
    }

    /// Renders only the top nibble, so distinct addresses collide in groups.
    struct TopNibble;
    impl FrameFormat for TopNibble {
        fn format(&self, address: usize, out: &mut String) {
            push_hex(out, address >> (usize::BITS - 4));
        }
    }

    /// An address whose top nibble is `nibble` and whose low bits are `low`.
    fn nibbled(nibble: usize, low: usize) -> usize {
        (nibble << (usize::BITS - 4)) | low
    }

    fn point(frames: &[usize], counters: Counters) -> ProgramPoint {
        ProgramPoint {
            kind: PointKind::Recorded,
            frames: frames.to_vec(),
            counters,
            unretired_lifetime: 0,
        }
    }

    fn counters(total_bytes: u64, max_bytes: u64) -> Counters {
        Counters {
            total_bytes,
            total_blocks: 1,
            total_lifetime: 7,
            curr_bytes: 1,
            curr_blocks: 1,
            max_bytes,
            max_blocks: 1,
            at_gmax_bytes: 2,
            at_gmax_blocks: 1,
        }
    }

    /// Only the fields these tests read, over [`Snapshot::default`].
    ///
    /// `unwinder` and `time_source` are written even though they are already the
    /// defaults here: this emitter records both in the file, and
    /// `Strategy::default()` is platform-dependent, so a profile built on
    /// defaults would read differently on Windows.
    fn snapshot(points: Vec<ProgramPoint>) -> Snapshot {
        Snapshot {
            shutdown: Shutdown::Dropped,
            unwinder: crate::unwind::Strategy::FramePointer,
            // A profile with program points recorded captures; the validator
            // rejects one that claims points and no stack walks.
            captures: crate::unwind::CounterSnapshot {
                complete: points.len() as u64,
                ..Default::default()
            },
            time_source: TimeSource::Events,
            time_at_end: 100,
            points,
            command: String::from("test"),
            pid: 1,
            ..Snapshot::default()
        }
    }

    fn emit(snapshot: &Snapshot, format: &dyn FrameFormat) -> String {
        let mut buffer = Vec::new();
        write(snapshot, format, &mut buffer).expect("writing to a Vec cannot fail");
        String::from_utf8(buffer).expect("valid UTF-8")
    }

    #[test]
    fn hexadecimal_rendering_has_no_leading_zeros_and_never_empties() {
        let mut out = String::new();
        push_hex(&mut out, 0);
        assert_eq!(out, "0x0");
        out.clear();
        push_hex(&mut out, 0x1044c81f0);
        assert_eq!(out, "0x1044c81f0");
        out.clear();
        push_hex(&mut out, usize::MAX);
        assert_eq!(out, format!("0x{:x}", usize::MAX));
    }

    #[test]
    fn the_frame_table_starts_with_the_root_and_no_point_refers_to_it() {
        let file = Folded::build(
            &snapshot(vec![
                point(&[0x10, 0x20], counters(100, 100)),
                point(&[0x30], counters(50, 50)),
            ]),
            &RawAddresses,
        );
        assert_eq!(file.ftbl[0], "[root]");
        for point in &file.points {
            assert!(
                !point.fs.contains(&0),
                "index 0 is the viewer's tree root and must not appear in a frame list"
            );
        }
    }

    #[test]
    fn distinct_call_sites_keep_distinct_frame_lists() {
        let file = Folded::build(
            &snapshot(vec![
                point(&[0x10, 0x20], counters(100, 100)),
                point(&[0x10, 0x30], counters(50, 50)),
            ]),
            &RawAddresses,
        );
        assert_eq!(file.points.len(), 2);
        assert_eq!(file.collisions, 0);
        assert_ne!(file.points[0].fs, file.points[1].fs);
    }

    /// The exit criterion from the plan: a rendering that collapses call sites
    /// must not produce a file the viewer refuses.
    #[test]
    fn points_that_render_to_the_same_frames_are_folded_into_one() {
        let file = Folded::build(
            &snapshot(vec![
                point(&[0x10, 0x20], counters(100, 40)),
                point(&[0x30, 0x40], counters(50, 30)),
                point(&[0x50, 0x60], counters(25, 60)),
            ]),
            &AllTheSame,
        );

        assert_eq!(file.points.len(), 1, "three renderings, one frame list");
        assert_eq!(file.collisions, 2);
        let merged = &file.points[0];
        assert_eq!(merged.total_bytes, 175, "cumulative bytes add");
        assert_eq!(merged.total_blocks, 3);
        assert_eq!(merged.total_lifetime, 21);
        assert_eq!(merged.at_gmax_bytes, 6, "bytes at the peak add");
        assert_eq!(merged.at_end_bytes, 3, "bytes at the end add");
        assert_eq!(
            merged.max_bytes, 60,
            "maxima are not summable: three points that peaked at different \
             moments did not jointly peak at their sum"
        );
        assert_eq!(
            merged.max_blocks, 3,
            "each point held one block at t-gmax, and t-gmax is one instant, so \
             the merged point demonstrably held three at once"
        );
    }

    /// The merged maximum must never contradict the merged snapshots. Summing
    /// `gb` while taking the larger `mb` — which is what the plan said to do —
    /// produces a program point that had more bytes live at the global peak
    /// than it ever had live at all.
    #[test]
    fn a_merged_maximum_is_never_smaller_than_what_was_live_at_the_peak() {
        let mut small = counters(100, 10);
        small.at_gmax_bytes = 9;
        small.at_gmax_blocks = 4;
        small.curr_bytes = 3;
        small.curr_blocks = 2;

        let file = Folded::build(
            &snapshot(vec![
                point(&[0x10], small),
                point(&[0x20], small),
                point(&[0x30], small),
            ]),
            &AllTheSame,
        );

        let merged = &file.points[0];
        assert_eq!(merged.at_gmax_bytes, 27);
        assert_eq!(merged.at_end_bytes, 9);
        assert_eq!(
            merged.max_bytes, 27,
            "27 bytes were live at one instant, so the maximum is at least 27"
        );
        assert!(merged.max_bytes >= merged.at_gmax_bytes);
        assert!(merged.max_bytes >= merged.at_end_bytes);
        assert!(merged.max_blocks >= merged.at_gmax_blocks);
        assert!(merged.max_blocks >= merged.at_end_blocks);
    }

    #[test]
    fn folding_is_by_the_whole_frame_list_not_the_innermost_frame() {
        // `0x1...` and `0x2...` render alike under TopNibble only in their top
        // nibble, so these two points share an inner frame but not a list.
        let file = Folded::build(
            &snapshot(vec![
                point(&[nibbled(1, 0), nibbled(2, 0)], counters(10, 10)),
                point(&[nibbled(1, 1), nibbled(3, 0)], counters(20, 20)),
                point(&[nibbled(1, 2), nibbled(2, 2)], counters(30, 30)),
            ]),
            &TopNibble,
        );
        assert_eq!(file.points.len(), 2);
        assert_eq!(file.collisions, 1);
    }

    #[test]
    fn every_emitted_frame_list_is_unique() {
        // The property the viewer enforces, checked on the folded output rather
        // than on the input.
        let file = Folded::build(
            &snapshot(vec![
                point(&[0x10], counters(10, 10)),
                point(&[0x20], counters(20, 20)),
                point(&[0x30], counters(30, 30)),
                point(&[], counters(40, 40)),
                point(&[], counters(50, 50)),
            ]),
            &AllTheSame,
        );
        let mut seen = std::collections::HashSet::new();
        for point in &file.points {
            assert!(seen.insert(point.fs.clone()), "repeated frame list");
        }
    }

    #[test]
    fn a_point_with_no_frames_survives_and_says_why_it_has_none() {
        let file = Folded::build(&snapshot(vec![point(&[], counters(64, 64))]), &RawAddresses);
        assert_eq!(file.points.len(), 1);
        assert_eq!(file.points[0].total_bytes, 64);

        // Not `fs: []`. The viewer draws that as an empty row, which reads as a
        // broken emitter rather than as "the stack could not be walked here".
        let frames = &file.points[0].fs;
        assert_eq!(frames.len(), 1, "{frames:?}");
        assert_eq!(file.ftbl[frames[0] as usize], UNWALKABLE_FRAME);
    }

    #[test]
    fn the_overflow_point_is_labelled_as_overflow_not_as_a_failed_walk() {
        let mut overflow = point(&[], counters(64, 64));
        overflow.kind = PointKind::Overflow;
        let file = Folded::build(&snapshot(vec![overflow]), &RawAddresses);

        let frames = &file.points[0].fs;
        assert_eq!(frames.len(), 1, "{frames:?}");
        assert_eq!(
            file.ftbl[frames[0] as usize], OVERFLOW_FRAME,
            "the overflow point tells the reader to raise the ceiling; a failed \
             stack walk tells them to check their build flags. Rendering both \
             the same way loses the difference."
        );
    }

    #[test]
    fn overflow_and_unwalkable_points_do_not_fold_into_each_other() {
        let mut overflow = point(&[], counters(10, 1));
        overflow.kind = PointKind::Overflow;
        let file = Folded::build(
            &snapshot(vec![point(&[], counters(20, 2)), overflow]),
            &RawAddresses,
        );

        assert_eq!(
            file.points.len(),
            2,
            "two different reasons for having no frames were merged into one row"
        );
        assert_eq!(file.collisions, 0);
    }

    #[test]
    fn points_are_ordered_by_weight_and_the_frame_table_follows_them() {
        let file = Folded::build(
            &snapshot(vec![
                point(&[0x10], counters(10, 10)),
                point(&[0x20], counters(300, 300)),
                point(&[0x30], counters(200, 200)),
            ]),
            &RawAddresses,
        );
        let weights: Vec<u64> = file.points.iter().map(|p| p.total_bytes).collect();
        assert_eq!(weights, [300, 200, 10]);
        // Frame table indices are handed out in the order the points use them,
        // so the heaviest point's frame comes first.
        assert_eq!(file.points[0].fs, [1]);
        assert_eq!(file.ftbl[1], "0x20: ???");
    }

    #[test]
    fn the_order_does_not_depend_on_the_order_points_were_recorded_in() {
        let forwards = Folded::build(
            &snapshot(vec![
                point(&[0x10], counters(10, 10)),
                point(&[0x20], counters(300, 300)),
                point(&[0x30], counters(200, 200)),
            ]),
            &RawAddresses,
        );
        let backwards = Folded::build(
            &snapshot(vec![
                point(&[0x30], counters(200, 200)),
                point(&[0x20], counters(300, 300)),
                point(&[0x10], counters(10, 10)),
            ]),
            &RawAddresses,
        );
        assert_eq!(forwards.points, backwards.points);
        assert_eq!(forwards.ftbl, backwards.ftbl);
    }

    /// Points that weigh the same keep the order the snapshot gave them.
    ///
    /// This test used to assert the opposite — that reversing the input left the
    /// file unchanged — because the tiebreak compared the points' rendered frame
    /// text. Independence from the input reads like the stronger property, and
    /// it was not one: every rendered frame begins with a runtime address, so
    /// the comparison was a reading of where the program had been mapped, and it
    /// changed answer exactly when tied points came from different images or the
    /// load bias moved an address to a different number of hex digits. Two runs
    /// then disagreed with no test able to see it, because a unit test picks its
    /// own addresses and they never move.
    ///
    /// [`Snapshot::points`](super::Snapshot::points) is ordered at the source
    /// now, so the emitter's job here is to be faithful to that order rather
    /// than to invent one of its own.
    #[test]
    fn equal_weights_keep_the_order_the_snapshot_gave_them() {
        let build = |addresses: [usize; 3]| {
            Folded::build(
                &snapshot(
                    addresses
                        .iter()
                        .map(|address| point(&[*address], counters(10, 10)))
                        .collect(),
                ),
                &RawAddresses,
            )
        };
        let forwards = build([0x10, 0x20, 0x30]);
        let backwards = build([0x30, 0x20, 0x10]);

        assert_eq!(forwards.ftbl[1], "0x10: ???");
        assert_eq!(
            backwards.ftbl[1], "0x30: ???",
            "the emitter reordered points that the snapshot had already ordered"
        );
        // Total, which is the part that has to hold: one snapshot, one file.
        assert_eq!(build([0x10, 0x20, 0x30]).ftbl, forwards.ftbl);
    }

    #[test]
    fn a_frame_that_renders_as_the_root_string_still_gets_its_own_index() {
        struct Rooty;
        impl FrameFormat for Rooty {
            fn format(&self, _address: usize, out: &mut String) {
                out.push_str("[root]");
            }
        }
        let file = Folded::build(&snapshot(vec![point(&[0x10], counters(1, 1))]), &Rooty);
        assert_eq!(file.points[0].fs, [1]);
        assert_eq!(file.ftbl, ["[root]", "[root]"]);
    }

    /// A saturated lifetime is reachable — `ProgramPoint::total_lifetime` is a
    /// saturating sum and `Snapshot` is a public type — and merging two of them
    /// must not panic. This runs inside `Profiler::drop`, where a panic while
    /// unwinding aborts the process.
    #[test]
    fn merging_saturates_rather_than_overflowing() {
        let mut enormous = counters(u64::MAX, u64::MAX);
        enormous.total_lifetime = u64::MAX;
        enormous.total_blocks = u64::MAX;
        enormous.curr_bytes = u64::MAX;
        enormous.curr_blocks = u64::MAX;
        enormous.at_gmax_bytes = u64::MAX;
        enormous.at_gmax_blocks = u64::MAX;

        let file = Folded::build(
            &snapshot(vec![point(&[0x10], enormous), point(&[0x20], enormous)]),
            &AllTheSame,
        );

        let merged = &file.points[0];
        assert_eq!(merged.total_bytes, u64::MAX);
        assert_eq!(merged.total_lifetime, u64::MAX);
        assert_eq!(merged.at_gmax_bytes, u64::MAX);
        assert!(merged.max_bytes >= merged.at_gmax_bytes);
    }

    /// The value of `tg` is the whole point of the peak gate, and until this
    /// test existed the emitter could have written any number at all.
    #[test]
    fn the_time_of_the_peak_is_the_engine_s_and_not_the_end_of_the_run() {
        let mut taken = snapshot(vec![point(&[0x10], counters(64, 64))]);
        taken.time_at_end = 9_000;
        taken.stats.time_at_max = 1_234;
        let json = emit(&taken, &RawAddresses);
        assert!(json.contains("\"tg\":1234"), "{json}");
        assert!(json.contains("\"te\":9000"), "{json}");
    }

    /// `tl` must include blocks that were never freed. Losing the
    /// `unretired_lifetime` term is invisible in the file's structure and turns
    /// the viewer's "short-lived" filter inside out.
    #[test]
    fn lifetimes_of_blocks_that_were_never_freed_reach_the_file() {
        let mut held = point(&[0x10], counters(64, 64));
        held.counters.total_lifetime = 40;
        held.unretired_lifetime = 302;
        let json = emit(&snapshot(vec![held]), &RawAddresses);
        assert!(
            json.contains("\"tl\":342"),
            "tl should be 40 freed + 302 still live: {json}"
        );
    }

    #[test]
    fn the_header_describes_a_heap_profile_with_lifetimes_and_no_accesses() {
        let json = emit(
            &snapshot(vec![point(&[0x10], counters(64, 64))]),
            &RawAddresses,
        );
        // `mode` and `verb` are free-form to the viewer but decide what it calls
        // things; `bkacc` false is a statement about what a `GlobalAlloc` shim
        // can know, and `tuth` is the cutoff the "short-lived" sort uses.
        assert!(json.contains("\"mode\":\"heap\""), "{json}");
        assert!(json.contains("\"verb\":\"Allocated\""), "{json}");
        assert!(json.contains("\"bklt\":true"), "{json}");
        assert!(json.contains("\"bkacc\":false"), "{json}");
        assert!(json.contains("\"tuth\":500"), "{json}");
    }

    /// `fs` is innermost-first, like the frames the engine captures. Reversing
    /// it would produce a structurally valid file that means the opposite, and
    /// no viewer would object.
    #[test]
    fn frame_lists_keep_the_innermost_frame_first() {
        let file = Folded::build(
            &snapshot(vec![point(&[0x10, 0x20, 0x30], counters(64, 64))]),
            &RawAddresses,
        );
        let names: Vec<&str> = file.points[0]
            .fs
            .iter()
            .map(|&at| file.ftbl[at as usize].as_str())
            .collect();
        assert_eq!(names, ["0x10: ???", "0x20: ???", "0x30: ???"]);
    }

    #[test]
    fn the_emitted_file_carries_every_field_the_viewer_demands() {
        let json = emit(
            &snapshot(vec![point(&[0x10, 0x20], counters(64, 64))]),
            &RawAddresses,
        );
        // `checkFields` on the top level, plus the two more it demands when
        // `bklt` is true.
        for field in [
            "dhatFileVersion",
            "mode",
            "verb",
            "bklt",
            "bkacc",
            "tu",
            "Mtu",
            "cmd",
            "pid",
            "te",
            "pps",
            "ftbl",
            "tg",
            "tuth",
        ] {
            assert!(json.contains(&format!("\"{field}\":")), "missing {field}");
        }
        // `checkPP`, plus `tl`, which the viewer reads without ever checking.
        for field in [
            "tb", "tbk", "tl", "mb", "mbk", "gb", "gbk", "eb", "ebk", "fs",
        ] {
            assert!(json.contains(&format!("\"{field}\":")), "missing {field}");
        }
        assert!(json.contains("\"dhatFileVersion\":2"));
    }

    // ---- what a renderer is allowed to hide ----

    /// Hides `hidden` frames from the top of every stack, and nothing else.
    /// Stands in for [`Trimmed`](crate::symbol::Trimmed), whose own rules are
    /// tested against real stacks in `symbol::trim`; what is under test here is
    /// what the *emitter* does with an answer, whatever produced it.
    struct Hiding {
        hidden: usize,
    }
    impl FrameFormat for Hiding {
        fn format(&self, address: usize, out: &mut String) {
            RawAddresses.format(address, out);
        }
        fn keep(&self, frames: &[String]) -> std::ops::Range<usize> {
            self.hidden.min(frames.len())..frames.len()
        }
    }

    #[test]
    fn a_renderer_that_hides_nothing_leaves_the_stack_exactly_as_it_was() {
        let file = Folded::build(
            &snapshot(vec![point(&[0x10, 0x20, 0x30], counters(64, 64))]),
            &Hiding { hidden: 0 },
        );
        assert_eq!(file.points[0].fs.len(), 3);
        assert_eq!(file.trimmed, 0);
    }

    /// Hidden frames are frames the process recorded and the file does not
    /// carry. A reader comparing this profile against a stack they walked
    /// themselves must not have to deduce that from the shape of it.
    #[test]
    fn every_frame_a_renderer_hides_is_counted_in_the_profile() {
        let taken = snapshot(vec![
            point(&[0x10, 0x20, 0x30], counters(64, 64)),
            point(&[0x40, 0x50], counters(32, 32)),
        ]);
        let file = Folded::build(&taken, &Hiding { hidden: 1 });
        assert_eq!(file.trimmed, 2, "one frame from each of two stacks");

        let json = emit(&taken, &Hiding { hidden: 1 });
        assert!(json.contains("\"trimmedFrames\":2"), "{json}");
        // And the untrimmed profile says so rather than omitting the field,
        // which is what makes the number readable without knowing the version
        // of this crate that wrote it.
        let whole = emit(&taken, &RawAddresses);
        assert!(whole.contains("\"trimmedFrames\":0"), "{whole}");
    }

    /// PLAN.md section 3.2, and the reason the fold is an exit criterion rather
    /// than a polish item: hiding frames is exactly what makes two distinct
    /// call sites indistinguishable, and a file with a repeated `fs` is one the
    /// viewer refuses to open.
    #[test]
    fn stacks_that_become_identical_once_trimmed_are_folded_rather_than_repeated() {
        let file = Folded::build(
            &snapshot(vec![
                point(&[0x10, 0x99], counters(100, 40)),
                point(&[0x20, 0x99], counters(50, 30)),
            ]),
            &Hiding { hidden: 1 },
        );

        assert_eq!(file.points.len(), 1, "two stacks, one surviving frame list");
        assert_eq!(file.collisions, 1);
        assert_eq!(file.trimmed, 2);
        assert_eq!(file.points[0].total_bytes, 150);
    }

    /// A frameless point is labelled `[unwalkable]`, which says the stack could
    /// not be walked. A renderer must not be able to make a stack that *was*
    /// walked say that, however it answers.
    #[test]
    fn a_renderer_that_keeps_nothing_cannot_make_a_walked_stack_deny_it() {
        struct KeepsNothing;
        impl FrameFormat for KeepsNothing {
            fn format(&self, address: usize, out: &mut String) {
                RawAddresses.format(address, out);
            }
            fn keep(&self, _frames: &[String]) -> std::ops::Range<usize> {
                0..0
            }
        }

        let file = Folded::build(
            &snapshot(vec![point(&[0x10, 0x20], counters(64, 64))]),
            &KeepsNothing,
        );
        let frames = &file.points[0].fs;
        assert_eq!(frames.len(), 1);
        assert_eq!(file.ftbl[frames[0] as usize], "0x10: ???");
        assert_ne!(file.ftbl[frames[0] as usize], UNWALKABLE_FRAME);
    }

    /// `keep` is answered by whoever supplied the renderer, and this runs inside
    /// `Profiler::drop`, where a panic while unwinding aborts the process.
    #[test]
    fn an_answer_outside_the_stack_is_corrected_rather_than_indexed_with() {
        struct Nonsense;
        impl FrameFormat for Nonsense {
            fn format(&self, address: usize, out: &mut String) {
                RawAddresses.format(address, out);
            }
            fn keep(&self, _frames: &[String]) -> std::ops::Range<usize> {
                // Past the end, and backwards. Built field by field because a
                // literal `99..3` is a lint error, which is the compiler making
                // this test's point for it.
                std::ops::Range { start: 99, end: 3 }
            }
        }

        let file = Folded::build(
            &snapshot(vec![point(&[0x10, 0x20], counters(64, 64))]),
            &Nonsense,
        );
        assert_eq!(file.points[0].fs.len(), 1);
        assert_eq!(file.trimmed, 1);
    }

    #[test]
    fn a_range_is_forced_to_keep_at_least_one_frame_and_no_frame_that_is_not_there() {
        assert_eq!(clamp_frames(0..0, 0), 0..0, "an empty stack stays empty");
        assert_eq!(clamp_frames(0..5, 5), 0..5, "an exact answer is untouched");
        assert_eq!(clamp_frames(1..4, 5), 1..4);
        assert_eq!(clamp_frames(0..0, 5), 0..1, "nothing becomes the innermost");
        assert_eq!(clamp_frames(3..3, 5), 3..4, "at the frame it asked for");
        assert_eq!(clamp_frames(0..9, 5), 0..5, "past the end is the end");
        assert_eq!(
            clamp_frames(7..9, 5),
            4..5,
            "starting past the end is the last"
        );
        assert_eq!(
            clamp_frames(std::ops::Range { start: 4, end: 2 }, 5),
            4..5,
            "backwards is one frame forwards"
        );
    }

    #[test]
    fn an_empty_profile_is_still_a_valid_file() {
        let json = emit(&snapshot(Vec::new()), &RawAddresses);
        assert!(json.contains("\"pps\":[]"));
        assert!(json.contains("\"[root]\""));
    }
}
