//! A human-readable summary, for stderr at the end of a run.
//!
//! The DHAT file is the artifact; this is the thing that tells someone whether
//! it is worth opening. It answers three questions and stops: how much was
//! allocated, what was live at the peak, and which call sites account for it.
//!
//! Anything the profiler is unsure about is printed here rather than left in the
//! file for someone to find later — a truncated snapshot, dropped blocks, or a
//! poisoned engine all produce a line that says so.

use std::io::{self, Write};

use super::{FrameFormat, Snapshot};

/// Writes a summary of `snapshot`, showing at most `top` program points.
pub(super) fn write<W: Write>(
    snapshot: &Snapshot,
    format: &dyn FrameFormat,
    mut out: W,
    top: usize,
) -> io::Result<()> {
    let stats = &snapshot.stats;
    // What the run counted decides both the words and the arithmetic. An ad hoc
    // weight is dimensionless, so rendering it in binary units would silently
    // report 1,024 retries as `1.0 KiB`; and a mode with no block lifetimes has
    // nothing to say about the peak or about what was live at the end, so those
    // lines are omitted rather than printed as zeroes.
    let mode = snapshot.settings.mode;
    let lifetimes = mode.block_lifetimes();
    let (_, many, per_count) = mode.units();
    // The file's own `verb`, lowercased once, so the summary and the DHAT viewer
    // describe the same numbers with the same word and there is one place the
    // word comes from.
    let verb = mode.verb().to_lowercase();
    // `counts_bytes`, not `block_lifetimes`: the two agree for `Heap` and
    // `AdHoc` and disagree for `Copy`, which counts bytes and has no lifetimes.
    // A review swapped them and only the copy summary changed, which nothing was
    // reading.
    let amount = |value: u64| {
        if mode.counts_bytes() {
            bytes(value)
        } else {
            format!("{} {many}", count(value))
        }
    };

    // This one goes to a terminal, so the screening described on
    // `push_display` is not a precaution here but the difference between
    // printing a command line and executing whatever it was made of.
    let mut screened = String::new();

    writeln!(out, "heapscope profile")?;
    super::push_display(&mut screened, &snapshot.command);
    writeln!(out, "  command    {screened}")?;
    writeln!(out, "  pid        {}", snapshot.pid)?;
    writeln!(out, "  mode       {mode}")?;
    if lifetimes {
        writeln!(
            out,
            "  time       {} {} at end, peak at {}",
            count(snapshot.time_at_end),
            snapshot.time_source.unit_long(),
            count(stats.time_at_max)
        )?;
    } else {
        writeln!(
            out,
            "  time       {} {} at end",
            count(snapshot.time_at_end),
            snapshot.time_source.unit_long()
        )?;
    }
    writeln!(out)?;
    writeln!(
        out,
        "  {verb:<9}  {} in {} {per_count}",
        amount(stats.total_bytes),
        count(stats.total_blocks)
    )?;
    if lifetimes {
        writeln!(
            out,
            "  at t-gmax  {} in {} {per_count}",
            amount(stats.max_bytes),
            count(stats.max_blocks)
        )?;
        writeln!(
            out,
            "  at t-end   {} in {} {per_count}",
            amount(stats.curr_bytes),
            count(stats.curr_blocks)
        )?;
    }

    write_shapes(&mut out, snapshot)?;

    for warning in warnings(snapshot) {
        writeln!(out, "  warning    {warning}")?;
    }

    write_threads(&mut out, snapshot, &amount, per_count, top)?;
    write_regions(&mut out, snapshot, &amount, per_count, top)?;
    write_overhead(&mut out, snapshot)?;

    if snapshot.points.is_empty() || top == 0 {
        return Ok(());
    }

    // Ties break on position in `Snapshot::points`, which is canonical, so two
    // runs of one program list equal-weight points in the same order. Written
    // out rather than left to the sort's stability: what is at stake is whether
    // the summary diffs cleanly, and a stable sort is not the kind of promise a
    // reader of this line would think to check.
    let mut order: Vec<usize> = (0..snapshot.points.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        let (left, right) = (&snapshot.points[a], &snapshot.points[b]);
        right
            .counters
            .total_bytes
            .cmp(&left.counters.total_bytes)
            .then_with(|| right.counters.total_blocks.cmp(&left.counters.total_blocks))
            .then_with(|| a.cmp(&b))
    });

    let shown = order.len().min(top);

    // Rendered before anything is printed, because the header says how many
    // frames the renderer hid, and that is only known once they all have been
    // through it. Bounded by `top`, which is what makes it affordable here and
    // not in the file emitter.
    let mut stacks: Vec<Vec<String>> = Vec::with_capacity(shown);
    let mut captured = 0usize;
    let mut raw = String::new();
    let mut rendered: Vec<String> = Vec::new();
    for &at in order.iter().take(shown) {
        let frames = &snapshot.points[at].frames;
        captured += frames.len();
        stacks.push(super::dhat_v2::shown_frames(frames, format, &mut raw, &mut rendered).to_vec());
    }
    let kept: usize = stacks.iter().map(Vec::len).sum();

    writeln!(out)?;
    writeln!(
        out,
        "Top {shown} of {} program points, by {many} {verb}. Times are in {}.",
        count(snapshot.points.len() as u64),
        snapshot.time_source.unit_long()
    )?;
    if kept < captured {
        // Deliberately does not say *which* frames, or why. The renderer
        // decided, and this emitter has no more idea what an address means than
        // it has of what a name means — naming the allocation path and the
        // runtime entry here would be printing `Trimmed`'s reasons for whatever
        // renderer happened to be passed in.
        writeln!(
            out,
            "{} of {} frames are not shown, because the frame renderer left \
             them out.",
            count((captured - kept) as u64),
            count(captured as u64)
        )?;
    }

    for (rank, (&at, stack)) in order.iter().take(shown).zip(&stacks).enumerate() {
        let point = &snapshot.points[at];
        let counters = &point.counters;
        writeln!(out)?;
        writeln!(
            out,
            "{:>3}. {} in {} {per_count} ({} of all {many} {verb})",
            rank + 1,
            amount(counters.total_bytes),
            count(counters.total_blocks),
            percent(counters.total_bytes, stats.total_bytes)
        )?;
        // The time unit is named once, in the header. Repeating it here would
        // mean choosing between "1 events" and a pluralisation rule for every
        // unit a future time source might introduce.
        if lifetimes {
            writeln!(
                out,
                "     at t-gmax {}, at t-end {}, peak {}, avg lifetime {}",
                amount(counters.at_gmax_bytes),
                amount(counters.curr_bytes),
                amount(counters.max_bytes),
                count(average(point.total_lifetime(), counters.total_blocks)),
            )?;
        }
        for frame in stack {
            writeln!(out, "       {frame}")?;
        }
        if stack.is_empty() {
            writeln!(out, "       (no frames were captured)")?;
        }
    }

    // For the reason given at the end of `html::write`: `out` is taken by
    // value, so a `BufWriter` handed here is dropped at the end of this
    // function and its final write's error is discarded. Stderr, which is where
    // this usually goes, does not buffer -- but `write_text_summary_with` is
    // public and takes any writer, and a summary that lost its last lines
    // without saying so is exactly what a reader would not notice.
    out.flush()
}

/// What the program asked for, beyond a number of bytes.
///
/// Three lines at most, each of which either answers a question the totals
/// cannot or is left out. A mean allocation size is not one of those answers:
/// a program making a million 24-byte allocations and one 24 MB allocation has
/// the same mean as one making two million 24-byte allocations.
///
/// Nothing at all where nothing was observed, which is every non-heap run — the
/// shim records no allocations there, so there are no shapes to describe — and
/// any heap run that recorded none.
fn write_shapes<W: Write>(out: &mut W, snapshot: &Snapshot) -> io::Result<()> {
    let shapes = &snapshot.shapes;
    if shapes.observed_blocks == 0 {
        return Ok(());
    }

    if let Some((floor, ceiling, blocks)) = shapes.commonest_size() {
        writeln!(
            out,
            "  commonest  {} to {} ({} of {} blocks)",
            bytes(floor as u64),
            bytes(ceiling as u64),
            percent(blocks, shapes.observed_blocks),
            count(shapes.observed_blocks)
        )?;
    }
    // Worth a line because `calloc` may hand back pages that are never faulted
    // in: a run whose bytes are mostly zeroed has a resident size unrelated to
    // its allocated size, which is the first thing a reader gets wrong when a
    // profile and `ps` disagree.
    //
    // The line reports the measurement and stops. It used to append "which may
    // never be faulted in", which is a claim about pages and was therefore false
    // of its own commonest output: a 16-byte block cannot span a page. The
    // reason belongs in prose that can qualify itself; a per-line editorial that
    // is wrong for small values is worse than none.
    if shapes.zeroed_blocks > 0 {
        writeln!(
            out,
            "  zeroed     {} in {} blocks",
            bytes(shapes.zeroed_bytes),
            count(shapes.zeroed_blocks)
        )?;
    }
    // The bytes a moving reallocation copied are real work the program paid for
    // and they appear in none of the sizes it asked for.
    if shapes.reallocs > 0 {
        writeln!(
            out,
            "  reallocs   {}, of which {} moved and copied {}",
            count(shapes.reallocs),
            count(shapes.reallocs_moved),
            bytes(shapes.bytes_copied)
        )?;
    }
    Ok(())
}

/// Who allocated.
///
/// Left out entirely for a single-threaded run: one row that repeats the totals
/// is not a section, and the summary is printed on every run that asks for one.
/// The file carries the row either way.
fn write_threads<W: Write>(
    out: &mut W,
    snapshot: &Snapshot,
    amount: &dyn Fn(u64) -> String,
    per_count: &str,
    top: usize,
) -> io::Result<()> {
    if snapshot.threads.len() < 2 || top == 0 {
        return Ok(());
    }

    let mut order: Vec<&super::ThreadStats> = snapshot.threads.iter().collect();
    order.sort_by_key(|row| std::cmp::Reverse(row.counts.total_bytes));
    let shown = order.len().min(top);
    let width = label_width(order.iter().take(shown).map(|row| thread_label(row).len()));

    writeln!(out)?;
    writeln!(out, "heapscope threads")?;
    for row in order.iter().take(shown) {
        let label = thread_label(row);
        write!(out, "  {label:width$}  ")?;
        write_row(out, snapshot, amount, per_count, &row.counts)?;
    }
    write_remainder(out, order.len(), shown)
}

/// What for. Left out for a run that entered no regions, which is most of them.
fn write_regions<W: Write>(
    out: &mut W,
    snapshot: &Snapshot,
    amount: &dyn Fn(u64) -> String,
    per_count: &str,
    top: usize,
) -> io::Result<()> {
    if snapshot.regions.is_empty() || top == 0 {
        return Ok(());
    }

    let mut order: Vec<&super::RegionStats> = snapshot.regions.iter().collect();
    order.sort_by_key(|row| std::cmp::Reverse(row.counts.total_bytes));
    let shown = order.len().min(top);
    let width = label_width(order.iter().take(shown).map(|row| region_label(row).len()));

    writeln!(out)?;
    writeln!(out, "heapscope regions")?;
    for row in order.iter().take(shown) {
        let label = region_label(row);
        write!(out, "  {label:width$}  ")?;
        write_row(out, snapshot, amount, per_count, &row.counts)?;
        // Only worth a line when it is not the ordinary answer. A region still
        // open when the profile was written is not an error, but it does mean
        // its numbers are a reading taken mid-phase.
        if row.active != 0 {
            writeln!(
                out,
                "  {:width$}  still open {} times",
                "",
                count(row.active)
            )?;
        }
    }
    write_remainder(out, order.len(), shown)
}

/// One attribution row's counters, in the mode's own units.
fn write_row<W: Write>(
    out: &mut W,
    snapshot: &Snapshot,
    amount: &dyn Fn(u64) -> String,
    per_count: &str,
    counts: &super::TallyStats,
) -> io::Result<()> {
    // The mode's own noun for what it counts, so that an ad hoc profile says
    // "events" where a heap one says "blocks" and neither has to be read as the
    // other.
    write!(
        out,
        "{} in {} {per_count} ({})",
        amount(counts.total_bytes),
        count(counts.total_blocks),
        percent(counts.total_bytes, snapshot.stats.total_bytes)
    )?;
    // A row's peak is its own, not its share of the whole heap's. Omitted in a
    // mode with no live blocks, where it would be a zero standing for a
    // measurement that does not exist.
    if snapshot.settings.mode.block_lifetimes() {
        write!(
            out,
            ", {} live, peak {}",
            amount(counts.curr_bytes),
            amount(counts.max_bytes)
        )?;
    }
    writeln!(out)
}

fn write_remainder<W: Write>(out: &mut W, total: usize, shown: usize) -> io::Result<()> {
    if total > shown {
        writeln!(out, "  and {} more", count((total - shown) as u64))?;
    }
    Ok(())
}

/// A name column wide enough for what is in it, and no wider than a terminal.
fn label_width(widths: impl Iterator<Item = usize>) -> usize {
    widths.max().unwrap_or(0).min(32)
}

fn thread_label(row: &super::ThreadStats) -> String {
    if row.overflow {
        // Not one thread. Named so, because a reader who took this for a thread
        // would read a sum over hundreds of them as one thread's appetite.
        return String::from("(threads past the table)");
    }
    match &row.name {
        Some(name) => {
            let mut screened = String::new();
            super::push_display(&mut screened, name);
            format!("#{} {screened}", row.id)
        }
        None => format!("#{} (unnamed)", row.id),
    }
}

fn region_label(row: &super::RegionStats) -> String {
    if row.overflow {
        return String::from("(regions past the table)");
    }
    match &row.name {
        Some(name) => {
            let mut screened = String::new();
            super::push_display(&mut screened, name);
            screened
        }
        None => format!("#{}", row.id),
    }
}

/// What the profiler cost the program it was measuring.
///
/// PLAN.md section 12 promises "honestly measured overhead", and the summary is
/// where someone decides whether to trust a profile at all — so the numbers
/// behind that promise belong here rather than only in the file. Three lines,
/// and each is a measurement rather than an estimate: the memory is what the
/// arena holds, the capture count is exact, and the per-capture time was timed
/// on this machine in this build.
fn write_overhead<W: Write>(out: &mut W, snapshot: &Snapshot) -> io::Result<()> {
    let metrics = &snapshot.metrics;
    if metrics.arena.bytes_reserved == 0 && !metrics.capture_cost.measured() {
        // A snapshot of a profiler that never ran. Nothing was spent, and three
        // lines of zero would say so at more length than the silence does.
        return Ok(());
    }

    writeln!(out)?;
    writeln!(out, "heapscope overhead")?;
    writeln!(
        out,
        "  memory     {} held, {} in use",
        bytes(metrics.arena.bytes_reserved as u64),
        bytes(metrics.arena.bytes_used as u64)
    )?;
    writeln!(
        out,
        "  tables     {} of {} program points, {} of {} live blocks",
        count(metrics.program_points.entries as u64),
        count(metrics.program_points.capacity as u64),
        count(metrics.live_blocks.entries as u64),
        count(metrics.live_blocks.capacity as u64)
    )?;

    let captures = snapshot.captures.complete
        + snapshot.captures.truncated
        + snapshot.captures.suspect
        + snapshot.captures.no_frames;
    if let Some(picos) = metrics.capture_cost.picos_per_capture() {
        // The product, not just the rate, because the product is the number
        // being asked about: how much of the run went into walking stacks. It
        // covers the walk and nothing else — not interning, not the peak gate —
        // and the line says so rather than letting it read as total overhead.
        let total = u128::from(picos) * u128::from(captures) / 1_000;
        writeln!(
            out,
            "  captures   {} walks at {} each = {} of stack walking",
            count(captures),
            picoseconds(picos),
            duration(u64::try_from(total).unwrap_or(u64::MAX))
        )?;
    }
    Ok(())
}

/// A picosecond count as nanoseconds, to two decimal places.
///
/// Two, because a frame-pointer walk costs about 21 ns and rounding that to
/// whole nanoseconds throws away a twentieth of it. Rounded to nearest in
/// integer arithmetic, like every other formatter here.
fn picoseconds(picos: u64) -> String {
    let hundredths = (u128::from(picos) + 5) / 10;
    format!("{}.{:02} ns", hundredths / 100, hundredths % 100)
}

/// A nanosecond count in whatever unit keeps it readable.
fn duration(nanos: u64) -> String {
    const UNITS: [(u64, &str); 4] = [
        (1_000_000_000, "s"),
        (1_000_000, "ms"),
        (1_000, "\u{b5}s"),
        (1, "ns"),
    ];
    for (scale, unit) in UNITS {
        if nanos >= scale {
            // Tenths in integer arithmetic, so the rounding is exact rather
            // than whatever the float formatter decides.
            let tenths = (u128::from(nanos) * 10 + u128::from(scale) / 2) / u128::from(scale);
            return format!("{}.{} {}", tenths / 10, tenths % 10, unit);
        }
    }
    String::from("0.0 ns")
}

/// Everything about this profile that a reader should not have to discover for
/// themselves.
fn warnings(snapshot: &Snapshot) -> Vec<String> {
    let mut warnings = Vec::new();
    if !snapshot.exact {
        warnings.push(String::from(
            "the engine could not reach a quiet point, so the per-point columns \
             need not sum to the totals",
        ));
    }
    if snapshot.poisoned {
        warnings.push(String::from(
            "the profiler reported an internal failure during the run; \
             see stderr for what it was",
        ));
    }
    if snapshot.stats.dropped_blocks > 0 {
        warnings.push(format!(
            "{} allocations went unrecorded because the live-block table was full",
            count(snapshot.stats.dropped_blocks)
        ));
    }
    if snapshot.points_dropped > 0 {
        warnings.push(format!(
            "{} program points appeared while the snapshot was being taken and \
             are missing from it",
            count(snapshot.points_dropped)
        ));
    }
    if snapshot.stats.refused_events > 0 {
        warnings.push(format!(
            "{} calls to heapscope::event or heapscope::copied were refused, \
             because this run counts {}",
            count(snapshot.stats.refused_events),
            match snapshot.settings.mode {
                crate::Mode::Heap => "heap allocations",
                crate::Mode::AdHoc => "ad hoc events",
                crate::Mode::Copy => "copied bytes",
            }
        ));
    }
    if snapshot.unattributed_blocks > 0 {
        warnings.push(format!(
            "{} live blocks could not be attributed to a program point, so their \
             lifetimes are missing",
            count(snapshot.unattributed_blocks)
        ));
    }
    warnings
}

/// `total / count`, rounded to nearest, and zero rather than a division by zero.
///
/// In `u128` because `total` can legitimately be `u64::MAX`: a summed lifetime
/// saturates rather than wrapping, and adding half the divisor to round would
/// then overflow.
fn average(total: u64, count: u64) -> u64 {
    if count == 0 {
        return 0;
    }
    let rounded = (u128::from(total) + u128::from(count) / 2) / u128::from(count);
    u64::try_from(rounded).unwrap_or(u64::MAX)
}

/// A byte count in binary units, to one decimal place.
fn bytes(value: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    if value < 1024 {
        return format!("{value} B");
    }
    // Scaled in integer arithmetic: tenths of a unit, so the rounding is exact
    // rather than whatever the float formatter decides. In `u128` because ten
    // times `u64::MAX` does not fit in a `u64`.
    let mut tenths = u128::from(value) * 10;
    let mut unit = 0;
    while tenths >= 1024 * 10 && unit + 1 < UNITS.len() {
        tenths /= 1024;
        unit += 1;
    }
    format!("{}.{} {}", tenths / 10, tenths % 10, UNITS[unit])
}

/// An integer with thousands separators.
///
/// Shared with the assertion failures in [`crate::stats`] so that a count is
/// grouped one way in this crate rather than two. Note that a *byte* figure is
/// not: this module renders those through [`bytes`] as `4.0 KiB` where a panic
/// message renders `4,096`, which is a deliberate difference — a summary is
/// scanned and a budget is compared against a number the reader wrote.
pub(crate) fn count(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (at, digit) in digits.chars().enumerate() {
        if at > 0 && (digits.len() - at).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

/// `part` as a percentage of `whole`, to one decimal place.
fn percent(part: u64, whole: u64) -> String {
    if whole == 0 {
        return String::from("n/a");
    }
    let tenths = (part as u128 * 1000 + whole as u128 / 2) / whole as u128;
    format!("{}.{}%", tenths / 10, tenths % 10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internals::engine::GlobalStats;
    use crate::output::{Counters, PointKind, ProgramPoint, RawAddresses, Shutdown};

    /// Only the fields these tests read, over [`Snapshot::default`].
    ///
    /// Neither `unwinder` nor `time_source` is among them, though both were
    /// written here for a while with a portability argument attached. The
    /// summary never reads the first at all, and flipping the second to
    /// `Monotonic` changes the rendered unit label with no assertion noticing.
    /// `dhat_v2.rs` pins both, and needs to.
    fn snapshot(points: Vec<ProgramPoint>) -> Snapshot {
        Snapshot {
            stats: GlobalStats {
                curr_bytes: 1_024,
                curr_blocks: 2,
                max_bytes: 8_192,
                max_blocks: 9,
                total_bytes: 1_048_576,
                total_blocks: 1_234,
                time_at_max: 900,
                epoch: 3,
                ..GlobalStats::default()
            },
            shutdown: Shutdown::Dropped,
            // A profile with program points recorded captures; the validator
            // rejects one that claims points and no stack walks.
            captures: crate::unwind::CounterSnapshot {
                complete: points.len() as u64,
                ..Default::default()
            },
            time_at_end: 1_234,
            points,
            command: String::from("target/debug/example --flag"),
            pid: 4242,
            ..Snapshot::default()
        }
    }

    fn point(frames: &[usize], total_bytes: u64) -> ProgramPoint {
        ProgramPoint {
            kind: PointKind::Recorded,
            frames: frames.to_vec(),
            counters: Counters {
                total_bytes,
                total_blocks: 4,
                total_lifetime: 400,
                curr_bytes: 16,
                curr_blocks: 1,
                max_bytes: total_bytes,
                max_blocks: 2,
                at_gmax_bytes: 32,
                at_gmax_blocks: 1,
            },
            unretired_lifetime: 0,
        }
    }

    fn render(snapshot: &Snapshot, top: usize) -> String {
        let mut buffer = Vec::new();
        write(snapshot, &RawAddresses, &mut buffer, top).expect("writing to a Vec cannot fail");
        String::from_utf8(buffer).expect("valid UTF-8")
    }

    /// This output goes straight to a terminal, which acts on what it is sent.
    ///
    /// Both strings a program does not choose reach it: the frame text, which
    /// carries a symbol name read out of a symbol table that may be damaged and
    /// an image path read off the filesystem, and `argv`. Dropping the screen on
    /// either one left the whole suite green until this existed — and this is
    /// the one output where an unescaped `\u{1b}` is not a cosmetic problem.
    #[test]
    fn nothing_reaching_the_terminal_can_command_it() {
        /// Renders every address as a name that is trying to repaint the screen,
        /// which is what a `FrameFormat` over a corrupt symbol table does.
        struct Hostile;
        impl FrameFormat for Hostile {
            fn format(&self, address: usize, out: &mut String) {
                super::super::push_hex(out, address);
                out.push_str(": \u{1b}[2Jcleared\u{202e}gnp.eslaf (/tmp/an\u{1b}[31m image)");
            }
        }

        let mut snapshot = snapshot(vec![point(&[0x1000], 4096)]);
        snapshot.command = String::from("prog \u{1b}[2J --flag \u{202e}gnp.exe\u{0}");

        let mut buffer = Vec::new();
        write(&snapshot, &Hostile, &mut buffer, 5).expect("writing to a Vec cannot fail");
        let summary = String::from_utf8(buffer).expect("valid UTF-8");

        let offenders: Vec<char> = summary
            .chars()
            .filter(|&c| {
                // Not `push_display`'s own predicate: a screen tested against
                // itself agrees with any answer it gives, including a future one
                // that stops escaping something. Stated from the rule instead,
                // the same way `tests/support/display.rs` does it.
                let code = c as u32;
                (code < 0x20 && c != '\n')
                    || (0x7F..=0x9F).contains(&code)
                    || matches!(
                        code,
                        0x061C | 0x200E | 0x200F | 0x202A..=0x202E | 0x2066..=0x2069
                    )
            })
            .collect();
        assert!(
            offenders.is_empty(),
            "the summary carries {offenders:?}, which a terminal will act on:\n{summary}"
        );

        // Escaped rather than dropped: the reader still sees that something was
        // there, and where.
        assert!(summary.contains("\\u{1b}[2Jcleared\\u{202e}"), "{summary}");
        assert!(
            summary.contains("prog \\u{1b}[2J --flag \\u{202e}"),
            "{summary}"
        );
    }

    /// Frames the renderer hides are missing from what someone reads, and this
    /// output has no `trimmedFrames` field for them to look up. So it says so,
    /// in the header, once — a per-point note would be five words repeated on
    /// every stack.
    ///
    /// What it must *not* say is why. `Outermost` below hides frames for a
    /// reason of its own, and an emitter that announced "the allocation path
    /// and the runtime entry" would be printing `Trimmed`'s reasons over
    /// somebody else's decision. That is the same rule the trait documents
    /// about names: nothing here knows what an address means.
    #[test]
    fn frames_the_renderer_hides_are_reported_without_a_reason_invented_for_them() {
        /// Shows only the outermost frame of every stack.
        struct Outermost;
        impl FrameFormat for Outermost {
            fn format(&self, address: usize, out: &mut String) {
                RawAddresses.format(address, out);
            }
            fn keep(&self, frames: &[String]) -> std::ops::Range<usize> {
                frames.len() - 1..frames.len()
            }
        }

        let taken = snapshot(vec![
            point(&[0x10, 0x20, 0x30], 900),
            point(&[0x40, 0x50], 100),
        ]);

        let mut buffer = Vec::new();
        write(&taken, &Outermost, &mut buffer, 5).expect("writing to a Vec cannot fail");
        let text = String::from_utf8(buffer).expect("valid UTF-8");

        assert!(
            text.contains("3 of 5 frames are not shown, because the frame renderer left them out."),
            "{text}"
        );
        assert!(
            !text.contains("allocation path") && !text.contains("runtime entry"),
            "the summary explained a decision it did not make:\n{text}"
        );
        assert!(
            text.contains("0x30"),
            "the kept frames are still printed:\n{text}"
        );
        assert!(text.contains("0x50"), "{text}");
        assert!(
            !text.contains("0x10"),
            "a hidden frame was printed:\n{text}"
        );

        // And a renderer that hides nothing says nothing, so the line is a
        // statement about this profile rather than boilerplate.
        assert!(!render(&taken, 5).contains("not shown"));
    }

    #[test]
    fn byte_counts_are_scaled_and_rounded() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(1023), "1023 B");
        assert_eq!(bytes(1024), "1.0 KiB");
        assert_eq!(bytes(1536), "1.5 KiB");
        assert_eq!(bytes(1_048_576), "1.0 MiB");
        assert_eq!(bytes(1_073_741_824), "1.0 GiB");
        // The largest unit is the last one in the table, not a wrong one past
        // the end, and the arithmetic does not overflow on the way there.
        let largest = bytes(u64::MAX);
        assert!(largest.starts_with("16383."), "{largest}");
        assert!(largest.ends_with(" PiB"), "{largest}");
    }

    #[test]
    fn durations_are_scaled_and_rounded() {
        assert_eq!(duration(0), "0.0 ns");
        assert_eq!(duration(999), "999.0 ns");
        assert_eq!(duration(1_000), "1.0 \u{b5}s");
        assert_eq!(duration(1_500), "1.5 \u{b5}s");
        assert_eq!(duration(295_400), "295.4 \u{b5}s");
        assert_eq!(duration(1_000_000), "1.0 ms");
        assert_eq!(duration(1_500_000_000), "1.5 s");
        // The largest unit is the last one in the table rather than a wrong one
        // past the end, and the arithmetic does not overflow reaching it.
        let largest = duration(u64::MAX);
        assert!(largest.ends_with(" s"), "{largest}");
    }

    /// Two decimal places, because a frame-pointer walk costs about 21 ns and
    /// whole nanoseconds would throw away a twentieth of it.
    #[test]
    fn a_capture_cost_keeps_two_decimal_places() {
        assert_eq!(picoseconds(11_350), "11.35 ns");
        assert_eq!(picoseconds(9_640), "9.64 ns");
        assert_eq!(picoseconds(0), "0.00 ns");
        assert_eq!(picoseconds(5), "0.01 ns", "rounded to nearest, not down");
        assert_eq!(picoseconds(4), "0.00 ns");
        assert_eq!(picoseconds(11_996), "12.00 ns");
        // A `u64` of picoseconds is 213 days; the arithmetic must not wrap on
        // the way to saying so.
        assert!(picoseconds(u64::MAX).ends_with(" ns"));
    }

    /// The promise in PLAN.md section 12 is "honestly measured overhead", and
    /// this is where a reader meets it. Every number in the block is a
    /// measurement: deleting the block, or leaving the capture line out of it,
    /// leaves the claim with nothing behind it.
    #[test]
    fn the_summary_says_what_the_profiler_cost() {
        let mut snapshot = snapshot(vec![point(&[0x10], 1_024)]);
        snapshot.metrics.arena = crate::internals::arena::ArenaStats {
            bytes_reserved: 2 * 1024 * 1024,
            bytes_used: 1024 * 1024,
            chunks: 4,
            refused: 0,
            limit: 512 * 1024 * 1024,
        };
        snapshot.metrics.program_points = crate::output::TableUsage {
            entries: 8,
            capacity: 1_048_576,
            bytes: 4096,
        };
        snapshot.metrics.live_blocks = crate::output::TableUsage {
            entries: 3,
            capacity: 4_194_304,
            bytes: 8192,
        };
        snapshot.metrics.capture_cost = crate::unwind::Cost {
            nanos: 93_000,
            captures: 8_192,
            frames: 7,
            strategy: crate::unwind::Strategy::FramePointer,
        };
        // Spread across all four outcomes, not piled into `complete`. A stack
        // walk that gave up is still a stack walk the program paid for, and
        // with the whole count in one field a summary adding up only that field
        // printed the right number for the wrong reason.
        snapshot.captures = crate::unwind::CounterSnapshot {
            complete: 26_000,
            truncated: 10,
            suspect: 5,
            no_frames: 3,
        };

        let text = render(&snapshot, 1);
        assert!(text.contains("heapscope overhead"), "{text}");
        assert!(text.contains("2.0 MiB held, 1.0 MiB in use"), "{text}");
        assert!(
            text.contains("8 of 1,048,576 program points, 3 of 4,194,304 live blocks"),
            "{text}"
        );
        // 93,000 ns / 8,192 captures = 11.35 ns, times 26,018 walks = 295.4 µs.
        assert!(
            text.contains("26,018 walks at 11.35 ns each = 295.4 \u{b5}s of stack walking"),
            "{text}"
        );
    }

    /// An unmeasured capture cost leaves the line out rather than printing a
    /// free capture, and a profiler that never ran prints no block at all.
    #[test]
    fn an_unmeasured_overhead_is_left_out_rather_than_reported_as_zero() {
        let mut snapshot = snapshot(vec![point(&[0x10], 1_024)]);
        assert!(
            !render(&snapshot, 1).contains("heapscope overhead"),
            "a profiler that reserved nothing and timed nothing printed an \
             overhead block of zeroes"
        );

        snapshot.metrics.arena.bytes_reserved = 65_536;
        let text = render(&snapshot, 1);
        assert!(text.contains("heapscope overhead"), "{text}");
        assert!(
            !text.contains("of stack walking"),
            "an unmeasured capture cost was reported as a rate: {text}"
        );
    }

    /// A row for each thread and region, in the mode's own units, and each
    /// row's peak is its own.
    #[test]
    fn the_summary_says_who_allocated_and_what_for() {
        let mut snapshot = snapshot(vec![point(&[0x10], 1_024)]);
        snapshot.threads = vec![
            crate::output::ThreadStats {
                id: 0,
                overflow: false,
                name: Some(String::from("main")),
                first_seen: 0,
                counts: crate::output::TallyStats {
                    total_bytes: 786_432,
                    total_blocks: 900,
                    curr_bytes: 1_024,
                    curr_blocks: 2,
                    max_bytes: 4_096,
                    max_blocks: 6,
                },
            },
            crate::output::ThreadStats {
                id: 1,
                overflow: false,
                name: None,
                first_seen: 7,
                counts: crate::output::TallyStats {
                    total_bytes: 262_144,
                    total_blocks: 334,
                    curr_bytes: 0,
                    curr_blocks: 0,
                    max_bytes: 2_048,
                    max_blocks: 3,
                },
            },
        ];
        snapshot.regions = vec![crate::output::RegionStats {
            id: 0,
            overflow: false,
            name: Some(String::from("parsing")),
            first_seen: 3,
            entries: 4,
            active: 0,
            counts: crate::output::TallyStats {
                total_bytes: 524_288,
                total_blocks: 500,
                curr_bytes: 0,
                curr_blocks: 0,
                max_bytes: 1_024,
                max_blocks: 2,
            },
        }];

        let text = render(&snapshot, 8);
        assert!(text.contains("heapscope threads"), "{text}");
        assert!(
            text.contains("#0 main")
                && text.contains("768.0 KiB in 900 blocks (75.0%), 1.0 KiB live, peak 4.0 KiB"),
            "{text}"
        );
        assert!(
            text.contains("#1 (unnamed)"),
            "a thread the platform did not name has to be a row rather than a \
             blank: {text}"
        );
        assert!(text.contains("heapscope regions"), "{text}");
        assert!(
            text.contains("parsing") && text.contains("512.0 KiB in 500 blocks (50.0%)"),
            "{text}"
        );
    }

    /// One row that repeats the totals is not a section. The file carries it
    /// either way, so nothing is lost by leaving it out of a summary printed on
    /// every run.
    #[test]
    fn a_single_threaded_run_gets_no_threads_section() {
        let mut snapshot = snapshot(vec![point(&[0x10], 1_024)]);
        snapshot.threads = vec![crate::output::ThreadStats {
            id: 0,
            overflow: false,
            name: Some(String::from("main")),
            first_seen: 0,
            counts: crate::output::TallyStats {
                total_bytes: 1_048_576,
                total_blocks: 1_234,
                curr_bytes: 1_024,
                curr_blocks: 2,
                max_bytes: 8_192,
                max_blocks: 9,
            },
        }];

        let text = render(&snapshot, 8);
        assert!(!text.contains("heapscope threads"), "{text}");
        assert!(
            !text.contains("heapscope regions"),
            "a run that entered no regions was told about regions: {text}"
        );
    }

    /// A region still open when the profile was written is not an error, but it
    /// does mean its numbers are a reading taken mid-phase, so the summary says
    /// so rather than presenting them as final.
    #[test]
    fn a_region_left_open_says_so() {
        let mut snapshot = snapshot(vec![point(&[0x10], 1_024)]);
        snapshot.regions = vec![crate::output::RegionStats {
            id: 0,
            overflow: false,
            name: Some(String::from("serving")),
            first_seen: 1,
            entries: 3,
            active: 2,
            counts: crate::output::TallyStats::default(),
        }];

        let text = render(&snapshot, 8);
        assert!(text.contains("still open 2 times"), "{text}");
    }

    /// The shared row stands for every thread past the table, so it is named as
    /// what it is. Read as one thread, a sum over hundreds of them is a wildly
    /// wrong answer to "which thread is the heavy one".
    #[test]
    fn the_overflow_row_says_it_is_not_a_thread() {
        let mut snapshot = snapshot(vec![point(&[0x10], 1_024)]);
        snapshot.threads = vec![
            crate::output::ThreadStats {
                id: 0,
                overflow: false,
                name: Some(String::from("main")),
                first_seen: 0,
                counts: crate::output::TallyStats::default(),
            },
            crate::output::ThreadStats {
                id: u16::MAX - 1,
                overflow: true,
                name: None,
                first_seen: 0,
                counts: crate::output::TallyStats {
                    total_bytes: 1_048_576,
                    total_blocks: 1_234,
                    ..crate::output::TallyStats::default()
                },
            },
        ];

        let text = render(&snapshot, 8);
        assert!(text.contains("(threads past the table)"), "{text}");
        assert!(
            !text.contains(&format!("#{}", u16::MAX - 1)),
            "the shared row was given an id a reader would take for a thread: {text}"
        );
    }

    /// What the program asked for, in the three lines that answer something the
    /// totals cannot. Each is conditional, because a run with no reallocations
    /// and nothing zeroed should not be told so at length.
    #[test]
    fn the_summary_says_what_the_program_asked_for() {
        let shapes = crate::internals::shape::Shapes::new();
        for _ in 0..8 {
            shapes.record(crate::internals::shape::Shape::of(24).aligned(8));
        }
        shapes.record(crate::internals::shape::Shape::of(4096).zeroed());
        shapes.record_realloc(&crate::internals::shape::Realloc {
            old_address: 0x1000,
            old_size: 100,
            new_address: 0x2000,
            new: crate::internals::shape::Shape::of(400),
        });

        let mut snapshot = snapshot(vec![point(&[0x10], 1_024)]);
        snapshot.shapes = shapes.snapshot();
        let text = render(&snapshot, 1);

        assert!(
            text.contains("commonest  16 B to 31 B (88.9% of 9 blocks)"),
            "{text}"
        );
        assert!(text.contains("zeroed     4.0 KiB in 1 blocks"), "{text}");
        assert!(
            text.contains("reallocs   1, of which 1 moved and copied 100 B"),
            "{text}"
        );

        // A run that asked for nothing zeroed and reallocated nothing says
        // neither, rather than saying zero twice.
        let plain = crate::internals::shape::Shapes::new();
        plain.record(crate::internals::shape::Shape::of(24));
        snapshot.shapes = plain.snapshot();
        let text = render(&snapshot, 1);
        assert!(text.contains("commonest"), "{text}");
        assert!(!text.contains("zeroed"), "{text}");
        assert!(!text.contains("reallocs"), "{text}");
    }

    /// A run with no shapes to describe says nothing about them. Every non-heap
    /// run is one: the shim records no allocations there.
    #[test]
    fn a_run_that_observed_no_allocations_describes_no_shapes() {
        let snapshot = snapshot(vec![point(&[0x10], 1_024)]);
        let text = render(&snapshot, 1);
        assert!(!text.contains("commonest"), "{text}");
    }

    #[test]
    fn counts_are_grouped() {
        assert_eq!(count(0), "0");
        assert_eq!(count(999), "999");
        assert_eq!(count(1_000), "1,000");
        assert_eq!(count(1_234_567), "1,234,567");
        assert_eq!(count(u64::MAX), "18,446,744,073,709,551,615");
    }

    #[test]
    fn percentages_avoid_dividing_by_zero() {
        assert_eq!(percent(1, 0), "n/a");
        assert_eq!(percent(0, 100), "0.0%");
        assert_eq!(percent(50, 100), "50.0%");
        assert_eq!(percent(1, 3), "33.3%");
        assert_eq!(percent(100, 100), "100.0%");
    }

    #[test]
    fn averages_avoid_dividing_by_zero() {
        assert_eq!(average(10, 0), 0);
        assert_eq!(average(10, 4), 3, "rounds to nearest, not toward zero");
        assert_eq!(average(10, 5), 2);
    }

    /// A summed lifetime saturates at `u64::MAX` rather than wrapping, so the
    /// averaging must cope with it. Rounding by adding half the divisor is
    /// exactly where that overflows.
    #[test]
    fn averages_of_a_saturated_total_do_not_overflow() {
        assert_eq!(average(u64::MAX, 1), u64::MAX);
        // Rounds up, so one more than the truncating quotient.
        assert_eq!(average(u64::MAX, 4), u64::MAX / 4 + 1);
        assert_eq!(average(u64::MAX, u64::MAX), 1);
    }

    /// The summary is written through `Snapshot`, so a saturated lifetime must
    /// reach a printed line rather than a panic.
    #[test]
    fn a_saturated_lifetime_prints_rather_than_panicking() {
        let mut held = point(&[0x10], 512);
        held.counters.total_lifetime = u64::MAX;
        held.unretired_lifetime = u64::MAX;
        let text = render(&snapshot(vec![held]), 1);
        assert!(text.contains("avg lifetime"), "{text}");
    }

    #[test]
    fn the_summary_leads_with_the_totals() {
        let text = render(&snapshot(vec![point(&[0x10], 512)]), 5);
        assert!(
            text.contains("allocated  1.0 MiB in 1,234 blocks"),
            "{text}"
        );
        assert!(text.contains("at t-gmax  8.0 KiB in 9 blocks"), "{text}");
        assert!(text.contains("at t-end   1.0 KiB in 2 blocks"), "{text}");
    }

    #[test]
    fn program_points_are_listed_heaviest_first() {
        let text = render(
            &snapshot(vec![
                point(&[0x10], 100),
                point(&[0x20], 900),
                point(&[0x30], 500),
            ]),
            3,
        );
        let first = text.find("0x20").expect("the heaviest point");
        let second = text.find("0x30").expect("the middle point");
        let third = text.find("0x10").expect("the lightest point");
        assert!(first < second && second < third, "{text}");
    }

    #[test]
    fn only_the_requested_number_of_points_is_shown() {
        let points = (1..=10)
            .map(|n| point(&[n * 0x10], n as u64 * 100))
            .collect();
        let text = render(&snapshot(points), 3);
        assert!(text.contains("Top 3 of 10 program points"), "{text}");
        assert_eq!(text.matches("at t-gmax 32 B").count(), 3, "{text}");
    }

    #[test]
    fn a_profile_with_no_points_still_prints_its_totals() {
        let text = render(&snapshot(Vec::new()), 10);
        assert!(text.contains("allocated  1.0 MiB"), "{text}");
        assert!(!text.contains("program points"), "{text}");
    }

    #[test]
    fn a_point_with_no_frames_says_so_rather_than_printing_nothing() {
        let text = render(&snapshot(vec![point(&[], 512)]), 5);
        assert!(text.contains("(no frames were captured)"), "{text}");
    }

    /// Degraded data must announce itself. A profile that quietly omits a
    /// million dropped blocks is worse than no profile.
    #[test]
    fn every_way_the_data_can_be_incomplete_produces_a_warning() {
        let mut snapshot = snapshot(vec![point(&[0x10], 512)]);
        assert!(!render(&snapshot, 1).contains("warning"));

        snapshot.exact = false;
        snapshot.poisoned = true;
        snapshot.stats.dropped_blocks = 7;
        snapshot.points_dropped = 3;
        snapshot.unattributed_blocks = 11;
        let text = render(&snapshot, 1);
        assert_eq!(text.matches("warning").count(), 5, "{text}");
        assert!(text.contains("quiet point"), "{text}");
        assert!(text.contains("internal failure"), "{text}");
        assert!(text.contains("7 allocations went unrecorded"), "{text}");
        assert!(text.contains("3 program points appeared"), "{text}");
        assert!(text.contains("11 live blocks"), "{text}");
    }

    /// The same rule the page is held to at the end of `html::write`.
    ///
    /// `write_text_summary_with` is public and takes any writer, so a caller
    /// can hand it a `BufWriter` -- which this function then drops, discarding
    /// the error from its final write. A summary silently missing its last
    /// lines is exactly what a reader would not notice.
    #[test]
    fn a_sink_that_fails_only_on_flush_still_reports_it() {
        struct FailsOnFlush;
        impl Write for FailsOnFlush {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::new(io::ErrorKind::StorageFull, "no space"))
            }
        }

        let snapshot = snapshot(vec![point(&[0x10], 512)]);
        let result = write(&snapshot, &RawAddresses, FailsOnFlush, 5);
        assert!(
            result.is_err(),
            "a summary that could not be made durable must say so, not return Ok"
        );
    }
}
