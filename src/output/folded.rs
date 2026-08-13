//! Folded stacks: the one line-oriented format every flame graph tool reads.
//!
//! ```text
//! main;run;parse;Vec::with_capacity 1048576
//! main;run;collect;HashMap::insert 262144
//! ```
//!
//! One line per distinct stack, outermost frame first, separated by `;`, and a
//! count at the end. That is the whole specification, and it is the reason this
//! emitter exists: `inferno`, `flamegraph.pl`, `speedscope`, and the Firefox
//! Profiler all read it, none of them needs to know anything about this crate,
//! and none of them had a way to open a profile written here.
//!
//! # It is one column, so the column is a parameter
//!
//! A folded file carries a single number per stack, and a heap profile has
//! several that a reader might want drawn. Choosing one silently would make the
//! picture mean something the file does not say, so [`FoldedMetric`] is asked
//! for rather than assumed.
//!
//! Every variant is a counter that **sums to a global total** — the flame
//! graph's own width is a figure that appears elsewhere in the profile, so a
//! reader can check the picture against the summary. That is what rules out the
//! obvious fifth choice: a program point's own peak (`maxBytes`) is a real
//! measurement and sums to nothing, because the points reached their maxima at
//! different instants. [`FoldedMetric::PeakBytes`] is `atGmaxBytes`, what each
//! point held **at the instant the whole heap was largest**, which does sum to
//! the run's peak. Those two are one field apart and answer different
//! questions; the wrong one produces a flame graph whose total exceeds the peak
//! it claims to be showing.
//!
//! # Frames outermost first
//!
//! Captured stacks are innermost first, and every consumer of this format wants
//! the opposite: the flame graph's root is the outermost frame. Reversed here
//! rather than left to the reader, because a folded file that is the wrong way
//! round still renders — upside down, with every stack rooted in whatever
//! allocated — and looks like a profile rather than like a mistake.

use std::io::{self, Write};

use super::dhat_v2::{shown_frames, OVERFLOW_FRAME, UNWALKABLE_FRAME};
use super::{FrameFormat, PointKind, ProgramPoint, Snapshot};

/// Which counter a folded file carries.
///
/// A folded file has one number per stack. This is that number.
///
/// Each of these sums to a figure the profile reports globally, so the total
/// width of the resulting flame graph is checkable against the summary rather
/// than being a number only the picture knows:
///
/// | Metric | Per point | Sums to |
/// |---|---|---|
/// | [`TotalBytes`](FoldedMetric::TotalBytes) | `totalBytes` | `totals.totalBytes` |
/// | [`TotalBlocks`](FoldedMetric::TotalBlocks) | `totalBlocks` | `totals.totalBlocks` |
/// | [`PeakBytes`](FoldedMetric::PeakBytes) | `atGmaxBytes` | `totals.maxBytes` |
/// | [`LiveBytes`](FoldedMetric::LiveBytes) | `atEndBytes` | `totals.currBytes` |
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FoldedMetric {
    /// Bytes allocated over the whole run, whether or not they were freed.
    ///
    /// The default, and the ordinary heap flame graph: where allocation
    /// *volume* went. A site that allocates and frees a megabyte a thousand
    /// times dominates this and contributes nothing to the two below, which is
    /// usually the answer someone chasing allocator time is looking for.
    #[default]
    TotalBytes,
    /// Blocks allocated over the whole run.
    ///
    /// The same question counted per request rather than per byte, which is the
    /// one to draw when the cost being chased is the number of calls rather
    /// than the size of them.
    TotalBlocks,
    /// Bytes each site held at the instant the whole heap was largest.
    ///
    /// What the peak was *made of*. Available only in a mode with block
    /// lifetimes — see [`FoldedMetric::needs_block_lifetimes`].
    ///
    /// This is `atGmaxBytes` and not a point's own `maxBytes`. See the module
    /// documentation for why the distinction decides whether the picture adds
    /// up.
    PeakBytes,
    /// Bytes still live when the profile was written.
    ///
    /// The leak view: everything allocated and not freed, attributed to where it
    /// was allocated. Available only in a mode with block lifetimes.
    LiveBytes,
}

impl FoldedMetric {
    /// Whether this metric exists only in a run that tracks block lifetimes.
    ///
    /// True for [`PeakBytes`](FoldedMetric::PeakBytes) and
    /// [`LiveBytes`](FoldedMetric::LiveBytes). An ad hoc or copy run records
    /// events, and an event is never live and never dies, so neither figure is a
    /// measurement that exists — [`Snapshot::write_folded`] refuses rather than
    /// writing a file of zeroes. Ask this first where the mode is not known in
    /// advance; [`Mode::block_lifetimes`](crate::Mode::block_lifetimes) is the
    /// other half of the comparison.
    pub fn needs_block_lifetimes(self) -> bool {
        matches!(self, Self::PeakBytes | Self::LiveBytes)
    }

    /// What to call this metric in a diagnostic.
    ///
    /// The name of the native profile's field, so that a message naming a
    /// metric names something the reader can find in the file.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TotalBytes => "totalBytes",
            Self::TotalBlocks => "totalBlocks",
            Self::PeakBytes => "atGmaxBytes",
            Self::LiveBytes => "atEndBytes",
        }
    }

    /// This metric's value for one program point.
    fn of(self, point: &ProgramPoint) -> u64 {
        let counters = &point.counters;
        match self {
            Self::TotalBytes => counters.total_bytes,
            Self::TotalBlocks => counters.total_blocks,
            Self::PeakBytes => counters.at_gmax_bytes,
            Self::LiveBytes => counters.curr_bytes,
        }
    }
}

/// Writes `snapshot` as folded stacks.
pub(super) fn write<W: Write>(
    snapshot: &Snapshot,
    format: &dyn FrameFormat,
    metric: FoldedMetric,
    mut out: W,
) -> io::Result<()> {
    // Refused rather than written as zeroes. Everywhere else in this layer a
    // measurement that does not exist is *omitted* — `bklt: false` drops DHAT's
    // `tg`, the native format leaves out `atEndBytes` in a mode with no live
    // blocks — and a folded file has nothing to omit into: the metric is the
    // whole file, so the omission is an empty one, which reads as "this program
    // allocated nothing" rather than as "you asked for a number this run does
    // not have".
    if metric.needs_block_lifetimes() && !snapshot.settings.mode.block_lifetimes() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "a {} run has no `{}`: it records events, which are never live and never die",
                snapshot.settings.mode.as_str(),
                metric.as_str()
            ),
        ));
    }

    // Two buffers reused across points, on the same terms as the DHAT emitter's:
    // `raw` holds what the renderer produced and `rendered` the screened frames.
    let mut raw = String::new();
    let mut rendered: Vec<String> = Vec::new();
    // The line being built, and the stacks already seen. Held rather than
    // streamed because two program points can render onto one stack — the same
    // collapse `dhat_v2::Folded` handles, arriving here through trimming or
    // through a renderer that names two addresses alike — and a folded file with
    // a repeated stack is not wrong, merely one every consumer has to sum for
    // itself. Summing here means the file's line count is its distinct-stack
    // count.
    let mut stack = String::new();
    let mut totals: Vec<(String, u64)> = Vec::with_capacity(snapshot.points.len());
    let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for point in &snapshot.points {
        let count = metric.of(point);
        // A zero-width stack is not something a flame graph can draw, and
        // `inferno` reports it as a malformed line rather than ignoring it. In a
        // heap run this is the ordinary case for the live and at-peak metrics:
        // most call sites hold nothing at the end, and drawing them as slivers
        // of nothing would bury the ones that do.
        if count == 0 {
            continue;
        }

        stack.clear();
        let shown = shown_frames(&point.frames, format, &mut raw, &mut rendered);
        // Outermost first. See the module documentation for why this is not left
        // to the reader.
        for frame in shown.iter().rev() {
            if !stack.is_empty() {
                stack.push(';');
            }
            push_frame(&mut stack, frame);
        }
        // A point with no frames would be a line that is just a number, which is
        // not a stack at all. Both ways of arriving at one get a frame naming
        // which it was, the same pair the DHAT emitter uses.
        if stack.is_empty() {
            let label = match point.kind {
                PointKind::Overflow => OVERFLOW_FRAME,
                PointKind::Recorded => UNWALKABLE_FRAME,
            };
            push_frame(&mut stack, label);
        }

        match index.get(&stack) {
            Some(&at) => totals[at].1 = totals[at].1.saturating_add(count),
            None => {
                index.insert(stack.clone(), totals.len());
                totals.push((stack.clone(), count));
            }
        }
    }

    // In the order the points appear, which `PpTable::sequence` makes a reading
    // of what the program did rather than of where the loader mapped it — so two
    // runs of a deterministic workload produce the same file. Sorting by weight
    // would read more nicely and would put a diff of two profiles at the mercy
    // of a single call site changing rank.
    for (stack, count) in &totals {
        writeln!(out, "{stack} {count}")?;
    }
    Ok(())
}

/// Appends one frame, with the separator escaped.
///
/// `;` is the format's only structure, and it has no escape of its own: every
/// reader splits the line on it. A frame name really can contain one — a path
/// component, or a Rust closure in a crate whose author used one — and left
/// alone it would split one frame into two, producing a flame graph with a level
/// that does not exist and no sign that anything went wrong.
///
/// Rendered in the `\u{3b}` form [`push_display`](super::push_display) uses, and
/// non-reversible for the reason stated there: what is guaranteed is that the
/// output contains no separator this file did not put there, not that the
/// original can be reconstructed. Everything else a frame could carry — a
/// newline, a terminal escape, a bidirectional override — was already screened
/// by `push_display` before this sees it, which is why this handles one
/// character rather than a set.
fn push_frame(out: &mut String, frame: &str) {
    for character in frame.chars() {
        if character == ';' {
            out.push_str("\\u{3b}");
        } else {
            out.push(character);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internals::engine::{GlobalStats, Mode};
    use crate::output::{Counters, RawAddresses, Shutdown};

    /// A renderer that gives every address a short name of its own, so that a
    /// test can read the stacks it expects rather than hexadecimal.
    struct Names;

    impl FrameFormat for Names {
        fn format(&self, address: usize, out: &mut String) {
            out.push_str(match address {
                0x10 => "inner",
                0x20 => "middle",
                0x30 => "outer",
                0x40 => "other",
                _ => "unknown",
            });
        }
    }

    fn point(frames: &[usize], total_bytes: u64) -> ProgramPoint {
        ProgramPoint {
            kind: PointKind::Recorded,
            frames: frames.to_vec(),
            counters: Counters {
                total_bytes,
                total_blocks: 3,
                total_lifetime: 7,
                curr_bytes: total_bytes / 8,
                curr_blocks: 1,
                max_bytes: total_bytes / 2,
                max_blocks: 2,
                at_gmax_bytes: total_bytes / 4,
                at_gmax_blocks: 1,
            },
            unretired_lifetime: 3,
        }
    }

    /// A snapshot whose totals are the sum of its points, which is what the
    /// engine guarantees and what the sum-to-a-global property below rests on.
    fn snapshot(points: Vec<ProgramPoint>) -> Snapshot {
        let mut stats = GlobalStats::default();
        for point in &points {
            stats.total_bytes += point.counters.total_bytes;
            stats.total_blocks += point.counters.total_blocks;
            stats.max_bytes += point.counters.at_gmax_bytes;
            stats.curr_bytes += point.counters.curr_bytes;
        }
        Snapshot {
            stats,
            points,
            shutdown: Shutdown::Dropped,
            ..Snapshot::default()
        }
    }

    fn emit(snapshot: &Snapshot, format: &dyn FrameFormat, metric: FoldedMetric) -> String {
        let mut buffer = Vec::new();
        write(snapshot, format, metric, &mut buffer).expect("writing to a Vec cannot fail");
        String::from_utf8(buffer).expect("valid UTF-8")
    }

    fn folded(snapshot: &Snapshot, metric: FoldedMetric) -> String {
        emit(snapshot, &Names, metric)
    }

    #[test]
    fn a_stack_is_written_outermost_first_with_its_count() {
        let text = folded(
            &snapshot(vec![point(&[0x10, 0x20, 0x30], 1024)]),
            FoldedMetric::TotalBytes,
        );
        assert_eq!(text, "outer;middle;inner 1024\n");
    }

    /// The captured order is innermost first and every consumer wants the
    /// opposite. Getting this backwards still renders, which is why it has a
    /// test of its own rather than being covered incidentally.
    #[test]
    fn the_innermost_frame_is_last() {
        let text = folded(
            &snapshot(vec![point(&[0x10, 0x20, 0x30], 8)]),
            FoldedMetric::TotalBytes,
        );
        let stack = text.trim_end().rsplit_once(' ').expect("a count").0;
        assert_eq!(stack.split(';').next_back(), Some("inner"));
        assert_eq!(stack.split(';').next(), Some("outer"));
    }

    /// The property the whole metric table rests on: what the flame graph is
    /// wide by is a figure the profile reports somewhere else.
    #[test]
    fn every_metric_sums_to_the_global_figure_it_claims() {
        let snapshot = snapshot(vec![
            point(&[0x10, 0x30], 4096),
            point(&[0x20, 0x30], 2048),
            point(&[0x40], 512),
        ]);
        for (metric, total) in [
            (FoldedMetric::TotalBytes, snapshot.stats.total_bytes),
            (FoldedMetric::TotalBlocks, snapshot.stats.total_blocks),
            (FoldedMetric::PeakBytes, snapshot.stats.max_bytes),
            (FoldedMetric::LiveBytes, snapshot.stats.curr_bytes),
        ] {
            let text = folded(&snapshot, metric);
            let summed: u64 = text
                .lines()
                .map(|line| {
                    line.rsplit_once(' ')
                        .expect("every line ends in a count")
                        .1
                        .parse::<u64>()
                        .expect("the count is a number")
                })
                .sum();
            assert_eq!(
                summed,
                total,
                "the folded file for {} sums to {summed}, and the profile reports {total}:\n{text}",
                metric.as_str()
            );
        }
    }

    /// `PeakBytes` is `atGmaxBytes`, not a point's own `maxBytes`. The two are
    /// one field apart and only the first adds up, so this pins the choice
    /// against a fixture where they differ.
    #[test]
    fn the_peak_metric_is_what_was_held_at_the_peak() {
        let text = folded(
            &snapshot(vec![point(&[0x10], 4096)]),
            FoldedMetric::PeakBytes,
        );
        // The fixture's own peak is half and its share of the global peak a
        // quarter, so reading the wrong field doubles the number.
        assert_eq!(text, "inner 1024\n");
    }

    /// Two points that render onto one stack are one line. Trimming is the
    /// ordinary way this happens: two call sites inside the same function differ
    /// by an address the renderer does not show.
    #[test]
    fn stacks_that_render_alike_are_summed_into_one_line() {
        let text = folded(
            &snapshot(vec![point(&[0x10, 0x30], 100), point(&[0x10, 0x30], 25)]),
            FoldedMetric::TotalBytes,
        );
        assert_eq!(text, "outer;inner 125\n");
    }

    /// A zero cannot be drawn, and `inferno` rejects the line rather than
    /// ignoring it. In a heap run most sites hold nothing at the end.
    #[test]
    fn a_stack_with_nothing_in_it_is_left_out() {
        let mut empty = point(&[0x10], 0);
        empty.counters.curr_bytes = 0;
        let text = folded(
            &snapshot(vec![point(&[0x20], 4096), empty]),
            FoldedMetric::TotalBytes,
        );
        assert_eq!(text, "middle 4096\n");
    }

    /// A line that is just a number is not a stack. Both ways of having no
    /// frames get a name that says which one happened.
    #[test]
    fn a_point_with_no_frames_still_names_itself() {
        let mut overflow = point(&[], 64);
        overflow.kind = PointKind::Overflow;
        let text = folded(
            &snapshot(vec![overflow, point(&[], 32)]),
            FoldedMetric::TotalBytes,
        );
        assert!(text.contains("[overflow]"), "{text}");
        assert!(text.contains("[unwalkable]"), "{text}");
        for line in text.lines() {
            assert!(
                line.split(' ')
                    .next()
                    .is_some_and(|frame| !frame.is_empty()),
                "a line began with its count: {line}"
            );
        }
    }

    /// The separator is the format's only structure and has no escape of its
    /// own, so a frame carrying one would invent a level of the flame graph.
    #[test]
    fn a_semicolon_in_a_name_cannot_split_a_frame() {
        struct Awkward;
        impl FrameFormat for Awkward {
            fn format(&self, _address: usize, out: &mut String) {
                out.push_str("weird;name");
            }
        }

        let text = emit(
            &snapshot(vec![point(&[0x10, 0x20], 16)]),
            &Awkward,
            FoldedMetric::TotalBytes,
        );
        let stack = text.trim_end().rsplit_once(' ').expect("a count").0;
        assert_eq!(
            stack.split(';').count(),
            2,
            "the two frames became {} after escaping: {text}",
            stack.split(';').count()
        );
        assert!(text.contains(r"weird\u{3b}name"), "{text}");
    }

    /// A run that records events has no live blocks, so two of the four metrics
    /// are not measurements it took. Refused rather than written as an empty
    /// file, which would read as a program that allocated nothing.
    #[test]
    fn a_run_without_block_lifetimes_refuses_the_metrics_it_has_no_measurement_for() {
        let mut events = snapshot(vec![point(&[0x10], 4096)]);
        events.settings.mode = Mode::AdHoc;

        for metric in [FoldedMetric::PeakBytes, FoldedMetric::LiveBytes] {
            let mut buffer = Vec::new();
            let error = write(&events, &Names, metric, &mut buffer)
                .expect_err("an ad hoc run has no live blocks");
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(
                error.to_string().contains(metric.as_str()),
                "the message does not name the metric it refused: {error}"
            );
            assert!(buffer.is_empty(), "a refused write produced a file anyway");
        }

        // And the two that every mode measures still work, so this refuses by
        // asking about the metric rather than about the mode alone.
        for metric in [FoldedMetric::TotalBytes, FoldedMetric::TotalBlocks] {
            assert!(!folded(&events, metric).is_empty());
        }
    }

    /// The claim on [`FoldedMetric::needs_block_lifetimes`]: it is the check
    /// that predicts the refusal, so a caller can avoid it.
    #[test]
    fn the_published_check_agrees_with_what_the_writer_does() {
        for metric in [
            FoldedMetric::TotalBytes,
            FoldedMetric::TotalBlocks,
            FoldedMetric::PeakBytes,
            FoldedMetric::LiveBytes,
        ] {
            for mode in [Mode::Heap, Mode::AdHoc, Mode::Copy] {
                let mut snapshot = snapshot(vec![point(&[0x10], 4096)]);
                snapshot.settings.mode = mode;
                let refused = write(&snapshot, &Names, metric, &mut Vec::new()).is_err();
                assert_eq!(
                    refused,
                    metric.needs_block_lifetimes() && !mode.block_lifetimes(),
                    "{} in a {} run: refused={refused}",
                    metric.as_str(),
                    mode.as_str()
                );
            }
        }
    }

    /// Two runs of a deterministic workload produce the same file, which is what
    /// PLAN.md section 12's fourth bullet asks of every emitter.
    #[test]
    fn the_order_is_the_order_the_points_are_in() {
        let snapshot = snapshot(vec![
            point(&[0x10], 1),
            point(&[0x20], 4096),
            point(&[0x30], 64),
        ]);
        assert_eq!(folded(&snapshot, FoldedMetric::TotalBytes), {
            let mut expected = String::new();
            expected.push_str("inner 1\n");
            expected.push_str("middle 4096\n");
            expected.push_str("outer 64\n");
            expected
        });
    }

    /// Nothing about the format depends on this crate's renderers, so the
    /// default one has to produce a usable file too.
    #[test]
    fn raw_addresses_still_produce_one_stack_per_line() {
        let text = emit(
            &snapshot(vec![point(&[0x10, 0x20], 512)]),
            &RawAddresses,
            FoldedMetric::TotalBytes,
        );
        assert_eq!(text.lines().count(), 1);
        assert!(text.starts_with("0x20: ???;0x10: ???"), "{text}");
        assert!(text.ends_with(" 512\n"), "{text}");
    }
}
