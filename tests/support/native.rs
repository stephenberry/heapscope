//! A validator for native profiles.
//!
//! There is no third-party viewer to be stricter than here — this format is
//! ours — so the rules are of a different kind from the DHAT validator's. That
//! one exists because `dh_view.js` checks presence rather than sense and has
//! documented holes. This one exists because the format's *whole claim* is that
//! nothing is lost, and a claim like that is only worth what checks it:
//!
//! - **Every field a reader needs is present**, so that a field silently dropped
//!   from the writer is a failing test rather than a `NaN` in someone's tool.
//! - **The parts agree.** Per-point bytes sum to the totals, each histogram sums
//!   to the requests it describes, every frame index is in range, every module
//!   index is in range. A profile that fails its own arithmetic is reporting an
//!   engine bug, and this is where it surfaces.
//! - **Addresses survived as text.** The format writes them as hexadecimal
//!   strings precisely because `JSON.parse` would round a number above 2^53, so
//!   a validator that accepted a number would accept the bug the choice exists
//!   to prevent.
//! - **A mode carries only what it measured.** Same rule as the DHAT
//!   validator's: block lifetimes are omitted, not zeroed, where there are none.

#![allow(dead_code)]

use super::json::{self, Value};

/// The version this crate writes. A file claiming any other version is one this
/// validator's rules were not written against.
const FORMAT_VERSION: u64 = 1;

/// Per-point fields that exist only where blocks have lifetimes.
const LIFETIME_FIELDS: [&str; 8] = [
    "retiredLifetime",
    "unretiredLifetime",
    "maxBytes",
    "maxBlocks",
    "atGmaxBytes",
    "atGmaxBlocks",
    "atEndBytes",
    "atEndBlocks",
];

/// Checks `text` against every rule, returning one message per problem.
///
/// An empty result means the file is valid.
pub fn problems(text: &str) -> Vec<String> {
    let value = match json::parse(text) {
        Ok(value) => value,
        Err(error) => return vec![format!("not valid JSON: {error}")],
    };
    if value.as_object().is_none() {
        return vec![format!("the document is a {}, not an object", value.kind())];
    }
    let root = &value;

    let mut problems = Vec::new();

    match root.get("format").and_then(Value::as_str) {
        Some("heapscope-profile") => {}
        Some(other) => problems.push(format!(
            "`format` is {other:?}; a tool handed this file would not know what \
             it is"
        )),
        None => problems.push(String::from("missing top-level `format`")),
    }
    match root.get("formatVersion").and_then(Value::as_u64) {
        Some(FORMAT_VERSION) => {}
        Some(other) => problems.push(format!(
            "`formatVersion` is {other}, not {FORMAT_VERSION}; these rules were \
             written against {FORMAT_VERSION}"
        )),
        None => problems.push(String::from("missing integer `formatVersion`")),
    }
    // The rule a reader has to follow travels with the file, because a reader
    // may have the file and not this crate's documentation.
    if !root
        .get("compatibility")
        .and_then(Value::as_str)
        .is_some_and(|rule| rule.contains("ignore unknown fields"))
    {
        problems.push(String::from(
            "`compatibility` does not state the forward-compatibility rule, so \
             nothing in the file tells a reader that unknown fields are safe",
        ));
    }
    if root.get("producer").and_then(Value::as_str).is_none() {
        problems.push(String::from("missing string `producer`"));
    }

    let lifetimes = check_run(root, &mut problems);
    check_settings(root, &mut problems);
    let totals = check_totals(root, lifetimes, &mut problems);
    check_shapes(root, lifetimes, &totals, &mut problems);
    check_self_metrics(root, &mut problems);
    // Whether the counters were read under exclusion decides how strict the
    // row sums below can be, so it is read from the file rather than assumed.
    let exclusive = root
        .get("run")
        .and_then(|run| run.get("exact"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    check_threads(root, lifetimes, exclusive, &totals, &mut problems);
    check_regions(root, lifetimes, &totals, &mut problems);
    let modules = check_modules(root, &mut problems);
    let frames = check_frames(root, modules, &mut problems);
    check_points(root, lifetimes, &totals, frames, &mut problems);

    problems
}

/// Reads `object[field]` as a `0x` hexadecimal string.
///
/// Every address in this format is written as one, because a JSON number is a
/// double in JavaScript and exact only to 2^53. Presence and type are reported
/// separately: the failure being guarded against is a *number*, and a rule that
/// only asked "is there a string here" would report a number as a missing
/// field — the right verdict for the wrong reason, and one that would still
/// pass if the emitter went back to writing numbers under another key.
fn address(object: &Value, path: &str, field: &str, problems: &mut Vec<String>) -> Option<u64> {
    match object.get(field) {
        None => {
            problems.push(format!("`{path}` has no `{field}`"));
            None
        }
        Some(value) => match value.as_str() {
            Some(text) => match text
                .strip_prefix("0x")
                .map(|hex| u64::from_str_radix(hex, 16))
            {
                Some(Ok(address)) => Some(address),
                _ => {
                    problems.push(format!(
                        "`{path}.{field}` is {text:?}, not a `0x` hexadecimal string"
                    ));
                    None
                }
            },
            None => {
                problems.push(format!(
                    "`{path}.{field}` is a {}, not a `0x` hexadecimal string; a \
                     JSON number is a double in JavaScript and would be rounded \
                     above 2^53",
                    value.kind()
                ));
                None
            }
        },
    }
}

/// Panics with every problem if `text` is not a valid native profile.
pub fn assert_valid(text: &str) {
    let problems = problems(text);
    assert!(
        problems.is_empty(),
        "the native profile is invalid:\n  {}\n\nprofile:\n{text}",
        problems.join("\n  ")
    );
}

/// The global counters a file claims, for the cross-checks below.
#[derive(Debug, Default)]
struct Totals {
    total_bytes: u64,
    total_blocks: u64,
    curr_bytes: u64,
    curr_blocks: u64,
    max_bytes: u64,
    dropped_blocks: u64,
    rows_dropped: u64,
}

/// Reads `object[field]` as an integer, reporting its absence.
fn integer(object: &Value, path: &str, field: &str, problems: &mut Vec<String>) -> Option<u64> {
    match object.get(field).and_then(Value::as_u64) {
        Some(value) => Some(value),
        None => {
            problems.push(format!("`{path}` has no integer `{field}`"));
            None
        }
    }
}

/// Reads a nested object, reporting its absence.
fn object<'a>(root: &'a Value, path: &str, problems: &mut Vec<String>) -> Option<&'a Value> {
    let mut current = root;
    for step in path.split('.') {
        match current.get(step) {
            Some(next) if next.as_object().is_some() => current = next,
            Some(next) => {
                problems.push(format!("`{path}` is a {}, not an object", next.kind()));
                return None;
            }
            None => {
                problems.push(format!("missing object `{path}`"));
                return None;
            }
        }
    }
    Some(current)
}

/// Returns whether the run's mode has block lifetimes.
fn check_run(root: &Value, problems: &mut Vec<String>) -> bool {
    let Some(run) = object(root, "run", problems) else {
        return false;
    };

    let mode = run.get("mode").and_then(Value::as_str).unwrap_or_default();
    let lifetimes = match mode {
        "heap" => true,
        "ad-hoc" | "copy" => false,
        other => {
            problems.push(format!(
                "`run.mode` is {other:?}, which is not one of heap, ad-hoc, copy"
            ));
            return false;
        }
    };

    for field in ["command", "shutdown", "timeSource", "timeUnit", "unwinder"] {
        if run.get(field).and_then(Value::as_str).is_none() {
            problems.push(format!("`run` has no string `{field}`"));
        }
    }
    for field in ["pid", "timeAtEnd"] {
        integer(run, "run", field, problems);
    }
    for field in ["exact", "poisoned"] {
        if run.get(field).and_then(Value::as_bool).is_none() {
            problems.push(format!("`run` has no boolean `{field}`"));
        }
    }

    // The same rule the DHAT validator applies to `tg`: an event was never live,
    // so the instant at which live bytes were greatest is not a measurement that
    // exists, and a zero would claim it does.
    if lifetimes {
        integer(run, "run", "timeAtMax", problems);
    } else if run.get("timeAtMax").is_some() {
        problems.push(format!(
            "`run.timeAtMax` is present in {mode} mode, which has no peak; it \
             must be omitted, not zeroed"
        ));
    }
    lifetimes
}

fn check_settings(root: &Value, problems: &mut Vec<String>) {
    let Some(settings) = object(root, "settings", problems) else {
        return;
    };
    for field in ["maxDepth", "maxLiveBlocks"] {
        integer(settings, "settings", field, problems);
    }
    if settings
        .get("trimFrames")
        .and_then(Value::as_bool)
        .is_none()
    {
        problems.push(String::from("`settings` has no boolean `trimFrames`"));
    }
    if integer(settings, "settings", "maxDepth", problems) == Some(0) {
        problems.push(String::from(
            "`settings.maxDepth` is 0, which would record no frames at all; the \
             engine clamps it to at least 1",
        ));
    }
}

fn check_totals(root: &Value, lifetimes: bool, problems: &mut Vec<String>) -> Totals {
    let mut totals = Totals::default();

    if let Some(value) = object(root, "totals", problems) {
        totals.total_bytes = integer(value, "totals", "totalBytes", problems).unwrap_or(0);
        totals.total_blocks = integer(value, "totals", "totalBlocks", problems).unwrap_or(0);
        if lifetimes {
            totals.curr_bytes = integer(value, "totals", "currBytes", problems).unwrap_or(0);
            totals.curr_blocks = integer(value, "totals", "currBlocks", problems).unwrap_or(0);
            totals.max_bytes = integer(value, "totals", "maxBytes", problems).unwrap_or(0);
            integer(value, "totals", "maxBlocks", problems);
            integer(value, "totals", "peaks", problems);
        } else {
            // Every one of these describes a block that was live, and an event
            // never was. Same rule as the per-point lifetime fields.
            for field in ["currBytes", "currBlocks", "maxBytes", "maxBlocks", "peaks"] {
                if value.get(field).is_some() {
                    problems.push(format!(
                        "`totals.{field}` is present in a mode with no live \
                         blocks; it must be omitted, not zeroed"
                    ));
                }
            }
        }
        if totals.curr_bytes > totals.total_bytes {
            problems.push(format!(
                "`totals.currBytes` is {} but only {} bytes were ever allocated",
                totals.curr_bytes, totals.total_bytes
            ));
        }
        if lifetimes && totals.max_bytes > totals.total_bytes {
            problems.push(format!(
                "`totals.maxBytes` is {} but only {} bytes were ever allocated",
                totals.max_bytes, totals.total_bytes
            ));
        }
    }

    if let Some(value) = object(root, "notRecorded", problems) {
        totals.dropped_blocks = integer(value, "notRecorded", "blocks", problems).unwrap_or(0);
        totals.rows_dropped =
            integer(value, "notRecorded", "attributionRows", problems).unwrap_or(0);
        for field in ["programPoints", "unattributedBlocks", "refusedEvents"] {
            integer(value, "notRecorded", field, problems);
        }
    }

    totals
}

fn check_shapes(
    root: &Value,
    records_allocations: bool,
    totals: &Totals,
    problems: &mut Vec<String>,
) {
    let Some(shapes) = object(root, "shapes", problems) else {
        return;
    };
    let Some(observed) = integer(shapes, "shapes", "observedBlocks", problems) else {
        return;
    };

    let mut sized = 0u64;
    match shapes.get("sizeClasses").and_then(Value::as_array) {
        Some(classes) => {
            let mut previous: Option<u64> = None;
            for class in classes {
                let floor = integer(class, "shapes.sizeClasses[]", "atLeast", problems);
                let ceiling = integer(class, "shapes.sizeClasses[]", "atMost", problems);
                let blocks = integer(class, "shapes.sizeClasses[]", "blocks", problems);
                sized += blocks.unwrap_or(0);
                if blocks == Some(0) {
                    problems.push(String::from(
                        "a size class with no blocks was written out; empty \
                         classes are left out so that sixty lines of zero do not \
                         appear in every profile",
                    ));
                }
                if let (Some(floor), Some(ceiling)) = (floor, ceiling) {
                    if floor > ceiling {
                        problems.push(format!(
                            "size class {floor}..={ceiling} is empty by its own bounds"
                        ));
                    }
                    // Ascending, so a reader can bisect and a diff of two
                    // profiles lines up.
                    if previous.is_some_and(|last| floor <= last) {
                        problems.push(format!(
                            "size class {floor}..={ceiling} does not follow the \
                             one before it; classes must ascend"
                        ));
                    }
                    previous = Some(floor);
                }
            }
        }
        None => problems.push(String::from("`shapes` has no array `sizeClasses`")),
    }

    let mut aligned = 0u64;
    match shapes.get("alignments").and_then(Value::as_array) {
        Some(alignments) => {
            for entry in alignments {
                let bytes = integer(entry, "shapes.alignments[]", "bytes", problems);
                aligned += integer(entry, "shapes.alignments[]", "blocks", problems).unwrap_or(0);
                if let Some(bytes) = bytes {
                    if !bytes.is_power_of_two() {
                        problems.push(format!(
                            "an alignment class of {bytes} bytes is not a power \
                             of two, so it is not an alignment class"
                        ));
                    }
                }
            }
        }
        None => problems.push(String::from("`shapes` has no array `alignments`")),
    }

    // The invariant that makes the histograms trustworthy: every request landed
    // in exactly one class of each kind.
    if sized != observed {
        problems.push(format!(
            "the size classes account for {sized} blocks but {observed} requests \
             were observed"
        ));
    }
    if aligned != observed {
        problems.push(format!(
            "the alignment classes account for {aligned} blocks but {observed} \
             requests were observed"
        ));
    }

    if let Some(zeroed) = object(root, "shapes.zeroed", problems) {
        let blocks = integer(zeroed, "shapes.zeroed", "blocks", problems).unwrap_or(0);
        integer(zeroed, "shapes.zeroed", "bytes", problems);
        if blocks > observed {
            problems.push(format!(
                "{blocks} blocks were zeroed out of {observed} observed"
            ));
        }
    }

    if let Some(reallocs) = object(root, "shapes.reallocs", problems) {
        let count = integer(reallocs, "shapes.reallocs", "count", problems).unwrap_or(0);
        let moved = integer(reallocs, "shapes.reallocs", "moved", problems).unwrap_or(0);
        let copied = integer(reallocs, "shapes.reallocs", "bytesCopied", problems).unwrap_or(0);
        integer(reallocs, "shapes.reallocs", "bytesGrown", problems);
        integer(reallocs, "shapes.reallocs", "bytesShrunk", problems);
        if moved > count {
            problems.push(format!(
                "{moved} reallocations moved out of {count} recorded"
            ));
        }
        if moved == 0 && copied > 0 {
            problems.push(format!(
                "{copied} bytes were copied by reallocations, none of which moved; \
                 a resize in place copies nothing"
            ));
        }
    }

    // What ties the histograms to the rest of the file: a request the live-block
    // table had no room for is still a request the program made.
    //
    // Applied on the *mode*, not on whether `observed` happens to be non-zero.
    // Guarding with `observed != 0` was how this rule was written first, and it
    // switched the rule off in precisely the case the comment below claims it
    // catches: a heap profile whose shim passed no shapes at all reports
    // `observedBlocks: 0` and would have skipped the check entirely. A non-heap
    // run records no allocations, so there the zero is the right answer and the
    // rule does not apply.
    //
    // Deliberately a bound rather than an equality. A shape is counted at the
    // top of `record_alloc` and the block counters move at the bottom, under
    // the peak gate, so a profile taken while other threads are still recording
    // — a concurrent shutdown, or a snapshot taken mid-run — catches some of
    // them in between and the two numbers differ by however many threads were in
    // that window. Measured at 1 in 30,320 on the probe's concurrent-shutdown
    // row, which is what turned this rule from an equality into a bound.
    //
    // One part in a thousand still catches everything the rule is for: a shim
    // that passes no shapes at all is out by all of them, and a drop path that
    // counts none is out by the drop count. A single-threaded or already-stopped
    // profile — every hand-built one, and every ordinary run — must be exact,
    // because a thousandth of a small number is zero.
    let recorded = totals.total_blocks + totals.dropped_blocks;
    if records_allocations && observed.abs_diff(recorded).saturating_mul(1_000) > recorded {
        problems.push(format!(
            "{observed} allocation requests were observed but the totals account \
             for {recorded} ({} recorded plus {} dropped), which is further apart \
             than threads in flight can explain",
            totals.total_blocks, totals.dropped_blocks
        ));
    }
    if !records_allocations && observed != 0 {
        problems.push(format!(
            "{observed} allocation requests were observed in a mode where the \
             allocator shim records nothing"
        ));
    }
}

/// One attribution row's counters, on the same terms as `totals`.
///
/// Returns what it read, so the caller can sum the rows and hold them against
/// the run.
fn check_tally(
    row: &Value,
    path: &str,
    lifetimes: bool,
    totals: &Totals,
    problems: &mut Vec<String>,
) -> (u64, u64, u64) {
    let total_bytes = integer(row, path, "totalBytes", problems).unwrap_or(0);
    let total_blocks = integer(row, path, "totalBlocks", problems).unwrap_or(0);

    if !lifetimes {
        // The same rule `totals` follows: an event was never live, so a zero
        // here would be a measurement of something that does not exist.
        for field in ["currBytes", "currBlocks", "maxBytes", "maxBlocks"] {
            if row.get(field).is_some() {
                problems.push(format!(
                    "`{path}` carries `{field}` in a mode with no live blocks"
                ));
            }
        }
        return (total_bytes, total_blocks, 0);
    }

    let curr_bytes = integer(row, path, "currBytes", problems).unwrap_or(0);
    let curr_blocks = integer(row, path, "currBlocks", problems).unwrap_or(0);
    let max_bytes = integer(row, path, "maxBytes", problems).unwrap_or(0);
    let max_blocks = integer(row, path, "maxBlocks", problems).unwrap_or(0);

    // A row cannot hold more than it ever allocated, and cannot be holding more
    // now than the most it ever held. Both are arithmetic the row does for
    // itself, so a row that fails one has a counter moving without its pair.
    if curr_bytes > max_bytes {
        problems.push(format!(
            "`{path}` holds {curr_bytes} bytes, more than its own peak of {max_bytes}"
        ));
    }
    if max_bytes > total_bytes {
        problems.push(format!(
            "`{path}` peaked at {max_bytes} bytes having only ever allocated {total_bytes}"
        ));
    }
    if curr_blocks > max_blocks {
        problems.push(format!(
            "`{path}` holds {curr_blocks} blocks, more than its own peak of {max_blocks}"
        ));
    }
    if max_blocks > total_blocks {
        problems.push(format!(
            "`{path}` peaked at {max_blocks} blocks having only ever allocated {total_blocks}"
        ));
    }
    // A row's live bytes are a subset of the run's at every instant, so its
    // high-water mark cannot be above the run's. This is the one rule here that
    // holds a row against a number from elsewhere in the file, which is what
    // makes it worth having: everything above is arithmetic a row could satisfy
    // while being entirely invented.
    if max_bytes > totals.max_bytes {
        problems.push(format!(
            "`{path}` peaked at {max_bytes} bytes, above the {} the whole heap \
             ever held; a row is a share of the run, not a thing beside it",
            totals.max_bytes
        ));
    }

    (total_bytes, total_blocks, curr_bytes)
}

/// Reads a row's `id`, holding it to being unique within its table.
fn check_row_id(row: &Value, path: &str, seen: &mut Vec<u64>, problems: &mut Vec<String>) {
    let Some(id) = integer(row, path, "id", problems) else {
        return;
    };
    if seen.contains(&id) {
        problems.push(format!(
            "`{path}` uses id {id} twice; a row id is what a reader joins on"
        ));
    }
    seen.push(id);
    if let Some(name) = row.get("name") {
        if name.as_str().is_none() {
            problems.push(format!("`{path}.name` is a {}, not a string", name.kind()));
        }
    }
    // Present only on the shared row, and only ever `true`: a `false` here would
    // be a row saying it is not the thing no row claims to be.
    if let Some(overflow) = row.get("overflow") {
        if overflow.as_bool() != Some(true) {
            problems.push(format!(
                "`{path}.overflow` is {}, but the field exists to mark the \
                 shared row and is left out otherwise",
                overflow.kind()
            ));
        }
        if row.get("name").is_some() {
            problems.push(format!(
                "`{path}` is the shared overflow row and carries a name; it \
                 stands for every thread or region past the table, not one"
            ));
        }
    }
}

/// A row says when it was first seen, unless it is the shared row, which stands
/// for many rows with many first instants and so has no single one.
fn check_first_seen(row: &Value, path: &str, problems: &mut Vec<String>) {
    if row.get("overflow").is_some() {
        if row.get("firstSeen").is_some() {
            problems.push(format!(
                "`{path}` is the shared overflow row and carries a `firstSeen`, \
                 which would be one instant standing for many"
            ));
        }
        return;
    }
    integer(row, path, "firstSeen", problems);
}

/// Who allocated.
///
/// The rule that ties these to the rest of the file: every recorded allocation
/// belongs to exactly one thread, so the rows sum to the run's own totals.
fn check_threads(
    root: &Value,
    lifetimes: bool,
    exclusive: bool,
    totals: &Totals,
    problems: &mut Vec<String>,
) {
    let Some(threads) = root.get("threads").and_then(Value::as_array) else {
        problems.push(String::from("missing array `threads`"));
        return;
    };

    let mut seen = Vec::new();
    let (mut bytes, mut blocks, mut live) = (0u64, 0u64, 0u64);
    for thread in threads {
        check_row_id(thread, "threads[]", &mut seen, problems);
        check_first_seen(thread, "threads[]", problems);
        let counts = check_tally(thread, "threads[]", lifetimes, totals, problems);
        bytes += counts.0;
        blocks += counts.1;
        live += counts.2;
    }

    // A run that recorded anything was recorded *by* something. Without this,
    // an emitter that dropped the array entirely would be caught only by the
    // sums below, which a missing array satisfies trivially at zero.
    if totals.total_blocks > 0 && threads.is_empty() {
        problems.push(String::from(
            "the run recorded blocks and names no thread that allocated them",
        ));
    }

    // A row that did not fit in the space the snapshot reserved is a row whose
    // bytes are missing from the sums below, and the file says how many. This is
    // the same accounting `programPoints` gets, and it is checked rather than
    // assumed: a file that dropped rows and did not say so would look exactly
    // like an emitter that lost them.
    if totals.rows_dropped != 0 {
        return;
    }

    // An **equality** where the flush reached a quiet point, which is every
    // ordinary run: the rows move in the same critical section as the counters
    // they are checked against, so there is no window for them to differ in.
    // Measured across three concurrent-shutdown runs of 34,000 events apiece:
    // exact every time, on all three fields.
    //
    // This rule was a bound at one part in a thousand first, and it should not
    // have been. It passed while the rows were 9% adrift from the totals under
    // a concurrent shutdown, because the tolerance was written to accommodate
    // that rather than the profiler being fixed to make it unnecessary. The
    // tolerance survives only where the file itself says exclusion was not
    // reached, which is what `exact: false` exists to declare.
    let tolerance = if exclusive { 0 } else { 1_000 };
    sums_match(
        "thread rows",
        "totalBytes",
        bytes,
        totals.total_bytes,
        tolerance,
        problems,
    );
    sums_match(
        "thread rows",
        "totalBlocks",
        blocks,
        totals.total_blocks,
        tolerance,
        problems,
    );
    if lifetimes {
        sums_match(
            "thread rows",
            "currBytes",
            live,
            totals.curr_bytes,
            tolerance,
            problems,
        );
    }
}

/// What for.
///
/// Unlike the thread rows these do **not** sum to the totals: an allocation
/// made outside every region belongs to no row, which is where most
/// allocations in most programs happen. What they must not do is exceed them.
fn check_regions(root: &Value, lifetimes: bool, totals: &Totals, problems: &mut Vec<String>) {
    let Some(regions) = root.get("regions").and_then(Value::as_array) else {
        problems.push(String::from("missing array `regions`"));
        return;
    };

    let mut seen = Vec::new();
    let (mut bytes, mut blocks) = (0u64, 0u64);
    for region in regions {
        check_row_id(region, "regions[]", &mut seen, problems);
        check_first_seen(region, "regions[]", problems);
        let entries = integer(region, "regions[]", "entries", problems).unwrap_or(0);
        let active = integer(region, "regions[]", "active", problems).unwrap_or(0);
        if active > entries {
            problems.push(format!(
                "a region is open {active} times having been entered {entries}"
            ));
        }
        if entries == 0 {
            problems.push(String::from(
                "a region row was written for a name that was never entered",
            ));
        }
        let counts = check_tally(region, "regions[]", lifetimes, totals, problems);
        bytes += counts.0;
        blocks += counts.1;
    }

    if bytes > totals.total_bytes || blocks > totals.total_blocks {
        problems.push(format!(
            "the regions account for {bytes} bytes in {blocks} blocks, more than \
             the {} bytes in {} blocks the run recorded; a nested region is \
             attributed to the innermost one only, so the rows cannot overlap",
            totals.total_bytes, totals.total_blocks
        ));
    }
}

/// Holds `rows` against `run`, to within one part in `tolerance`.
///
/// A `tolerance` of zero demands equality. Anything else is a bound of one part
/// in that many, for a profile the file itself says was not read under
/// exclusion.
fn sums_match(
    what: &str,
    field: &str,
    rows: u64,
    run: u64,
    tolerance: u64,
    problems: &mut Vec<String>,
) {
    let apart = rows.abs_diff(run);
    let allowed = run.checked_div(tolerance).unwrap_or(0);
    if apart > allowed {
        problems.push(format!(
            "the {what} account for {rows} `{field}` but the run recorded {run}, \
             which is {apart} apart and more than the {allowed} a profile in this \
             state may be"
        ));
    }
}

/// Returns how many images the module map has.
fn check_modules(root: &Value, problems: &mut Vec<String>) -> usize {
    let Some(modules) = root.get("modules").and_then(Value::as_array) else {
        problems.push(String::from("missing array `modules`"));
        return 0;
    };

    for module in modules {
        // The one field offline resolution cannot be done without: the map
        // exists to name the file to resolve an address against, and a map
        // without paths is a list of numbers. Deleting it left the whole suite
        // green, because the only test looking at these strings was a negative
        // one — and a missing field satisfies "does not contain a bidi
        // override" perfectly.
        match module.get("path").and_then(Value::as_str) {
            Some(path) if !path.is_empty() => {}
            Some(_) => problems.push(String::from("a module has an empty `path`")),
            None => problems.push(String::from(
                "a module has no `path`, so nothing says which file to resolve \
                 its addresses against",
            )),
        }
        // Addresses, and therefore strings. `size` is a count and stays a
        // number, like every other count in the file.
        for field in ["load", "start", "bias"] {
            address(module, "modules[]", field, problems);
        }
        if integer(module, "modules[]", "size", problems) == Some(0) {
            problems.push(String::from(
                "a module covers zero bytes, so no address can be in it",
            ));
        }
    }
    modules.len()
}

fn check_self_metrics(root: &Value, problems: &mut Vec<String>) {
    if object(root, "selfMetrics", problems).is_none() {
        return;
    }

    if let Some(arena) = object(root, "selfMetrics.arena", problems) {
        let reserved = integer(arena, "selfMetrics.arena", "bytesReserved", problems).unwrap_or(0);
        let used = integer(arena, "selfMetrics.arena", "bytesUsed", problems).unwrap_or(0);
        integer(arena, "selfMetrics.arena", "chunks", problems);
        integer(arena, "selfMetrics.arena", "refused", problems);
        let limit = integer(arena, "selfMetrics.arena", "limit", problems).unwrap_or(0);
        if used > reserved {
            problems.push(format!(
                "the arena handed out {used} bytes having reserved only {reserved}"
            ));
        }
        if reserved > limit {
            problems.push(format!(
                "the arena reserved {reserved} bytes against a limit of {limit}"
            ));
        }
    }

    for table in [
        "selfMetrics.programPoints",
        "selfMetrics.liveBlocks",
        "selfMetrics.threads",
        "selfMetrics.regions",
    ] {
        if let Some(usage) = object(root, table, problems) {
            let entries = integer(usage, table, "entries", problems).unwrap_or(0);
            let capacity = integer(usage, table, "capacity", problems).unwrap_or(0);
            integer(usage, table, "bytes", problems);
            if entries > capacity {
                problems.push(format!(
                    "`{table}` holds {entries} entries against a capacity of {capacity}"
                ));
            }
        }
    }

    if let Some(captures) = object(root, "selfMetrics.captures", problems) {
        for field in ["complete", "truncated", "suspect", "noFrames"] {
            integer(captures, "selfMetrics.captures", field, problems);
        }
    }

    // Optional: a process that never started a profiler measured nothing, and
    // the field is absent rather than zero. Where it is present it has to be a
    // measurement.
    if let Some(cost) = root.get("selfMetrics").and_then(|m| m.get("captureCost")) {
        let nanos = integer(cost, "selfMetrics.captureCost", "nanos", problems).unwrap_or(0);
        let captures = integer(cost, "selfMetrics.captureCost", "captures", problems).unwrap_or(0);
        integer(cost, "selfMetrics.captureCost", "frames", problems);
        if cost
            .get("unwinder")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            problems.push(String::from(
                "`selfMetrics.captureCost` does not say which unwinder it \
                 measured, and the two differ by two orders of magnitude",
            ));
        }
        if nanos == 0 || captures == 0 {
            problems.push(String::from(
                "`selfMetrics.captureCost` is present and reads as a free \
                 capture; an unmeasured cost is omitted, not zeroed",
            ));
        }
    }
}

/// Returns how many entries the frame table has.
fn check_frames(root: &Value, modules: usize, problems: &mut Vec<String>) -> usize {
    let Some(frames) = root.get("frames").and_then(Value::as_array) else {
        problems.push(String::from("missing array `frames`"));
        return 0;
    };

    let mut seen = std::collections::BTreeSet::new();
    for frame in frames {
        // An address is the one thing a frame always has, and it is a string
        // for the reason `address` documents.
        if let Some(written) = address(frame, "frames[]", "addr", problems) {
            if !seen.insert(written) {
                problems.push(format!(
                    "frame address {written:#x} appears twice in the table, so \
                     it was resolved twice and points may disagree about it"
                ));
            }
        }
        if let Some(module) = frame.get("module") {
            match module.as_u64() {
                Some(index) if (index as usize) < modules => {}
                Some(index) => problems.push(format!(
                    "a frame names module {index}, but the map has {modules}"
                )),
                None => problems.push(String::from("a frame's `module` is not an integer")),
            }
        }
        // A file address without an image is an offset into nothing.
        if frame.get("fileAddr").is_some() {
            address(frame, "frames[]", "fileAddr", problems);
        }
        if frame.get("fileAddr").is_some() && frame.get("module").is_none() {
            problems.push(String::from(
                "a frame has a `fileAddr` and no `module`, so nothing says which \
                 file the address is in",
            ));
        }
        if frame.get("symbol").is_some() && frame.get("symbolOffset").is_none() {
            problems.push(String::from(
                "a frame has a `symbol` and no `symbolOffset`, so nothing says \
                 how far into it the address is",
            ));
        }
    }
    frames.len()
}

fn check_points(
    root: &Value,
    lifetimes: bool,
    totals: &Totals,
    frames: usize,
    problems: &mut Vec<String>,
) {
    let Some(points) = root.get("points").and_then(Value::as_array) else {
        problems.push(String::from("missing array `points`"));
        return;
    };

    let mut total_bytes = 0u64;
    let mut total_blocks = 0u64;
    let mut at_end_bytes = 0u64;
    let mut at_gmax_bytes = 0u64;

    for point in points {
        match point.get("kind").and_then(Value::as_str) {
            Some("recorded" | "overflow") => {}
            Some(other) => problems.push(format!(
                "a point's `kind` is {other:?}, which is neither `recorded` nor \
                 `overflow`"
            )),
            None => problems.push(String::from(
                "a point has no `kind`, so nothing distinguishes a stack that \
                 could not be walked from the table having filled up",
            )),
        }

        let bytes = integer(point, "points[]", "totalBytes", problems).unwrap_or(0);
        let blocks = integer(point, "points[]", "totalBlocks", problems).unwrap_or(0);
        total_bytes += bytes;
        total_blocks += blocks;

        if lifetimes {
            for field in LIFETIME_FIELDS {
                integer(point, "points[]", field, problems);
            }
            at_end_bytes += point.get("atEndBytes").and_then(Value::as_u64).unwrap_or(0);
            at_gmax_bytes += point
                .get("atGmaxBytes")
                .and_then(Value::as_u64)
                .unwrap_or(0);

            let max = point.get("maxBytes").and_then(Value::as_u64).unwrap_or(0);
            if max > bytes {
                problems.push(format!(
                    "a point peaked at {max} bytes having only ever allocated {bytes}"
                ));
            }
        } else {
            for field in LIFETIME_FIELDS {
                if point.get(field).is_some() {
                    problems.push(format!(
                        "a point carries `{field}` in a mode with no block \
                         lifetimes; it must be omitted, not zeroed"
                    ));
                }
            }
        }

        match point.get("frames").and_then(Value::as_array) {
            Some(list) => {
                for index in list {
                    match index.as_u64() {
                        Some(index) if (index as usize) < frames => {}
                        Some(index) => problems.push(format!(
                            "a point names frame {index}, but the table has {frames}"
                        )),
                        None => {
                            problems.push(String::from("a point's frame index is not an integer"))
                        }
                    }
                }
            }
            None => problems.push(String::from("a point has no `frames` array")),
        }
    }

    // Nothing folds here, so the sums are exact rather than approximate.
    if total_bytes != totals.total_bytes {
        problems.push(format!(
            "the points account for {total_bytes} bytes and the totals say {}",
            totals.total_bytes
        ));
    }
    if total_blocks != totals.total_blocks {
        problems.push(format!(
            "the points account for {total_blocks} blocks and the totals say {}",
            totals.total_blocks
        ));
    }
    if lifetimes {
        if at_end_bytes != totals.curr_bytes {
            problems.push(format!(
                "the points hold {at_end_bytes} bytes at the end and the totals \
                 say {}",
                totals.curr_bytes
            ));
        }
        // The property the peak gate exists to make true: at the instant the
        // heap was largest, the points held exactly the maximum between them.
        if at_gmax_bytes != totals.max_bytes {
            problems.push(format!(
                "the points held {at_gmax_bytes} bytes at the peak and the peak \
                 was {}",
                totals.max_bytes
            ));
        }
    }
}
