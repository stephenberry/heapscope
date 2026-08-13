//! A validator for DHAT version 2 files.
//!
//! It is deliberately **stricter than the viewer**, for two reasons.
//!
//! First, the viewer's own checks have holes: `tl` is read but never checked, so
//! omitting it produces a file that loads perfectly and renders every average
//! lifetime as `NaN`. A validator that only reimplemented `checkFields` would
//! call that file good.
//!
//! Second, the viewer checks presence, not sense. A file where a program point
//! claims to have had more bytes live at the global peak than it ever allocated
//! is structurally fine and semantically impossible; catching that is how a
//! bug in the engine's accounting shows up as a failing test rather than as a
//! number someone eventually disbelieves.
//!
//! The viewer-derived rules are from `dhat/dh_view.js` in the Valgrind tree:
//! `checkFields` on the top level, `checkPP` per program point, the two extra
//! fields required when `bklt` is true, `ftbl[0]` being the tree root, and the
//! `data file contains a repeated location` error that a duplicated frame
//! sequence triggers.

#![allow(dead_code)]

use super::json::{self, Value};

/// Checks `text` against every rule, returning one message per problem.
///
/// An empty result means the file is valid.
pub fn problems(text: &str) -> Vec<String> {
    let value = match json::parse(text) {
        Ok(value) => value,
        Err(error) => return vec![format!("not valid JSON: {error}")],
    };
    let Some(root) = value.as_object() else {
        return vec![format!("the document is a {}, not an object", value.kind())];
    };

    let mut problems = Vec::new();
    let mut require = |field: &str, test: fn(&Value) -> bool, kind: &str| {
        match root.get(field) {
            None => problems.push(format!("missing top-level field `{field}`")),
            Some(value) if !test(value) => problems.push(format!(
                "top-level `{field}` is a {}, expected {kind}",
                value.kind()
            )),
            Some(_) => {}
        };
    };

    // `checkFields` in the viewer, with the types it then assumes.
    require("dhatFileVersion", |v| v.as_u64().is_some(), "an integer");
    require("mode", |v| v.as_str().is_some(), "a string");
    require("verb", |v| v.as_str().is_some(), "a string");
    require("bklt", |v| v.as_bool().is_some(), "a boolean");
    require("bkacc", |v| v.as_bool().is_some(), "a boolean");
    require("tu", |v| v.as_str().is_some(), "a string");
    require("Mtu", |v| v.as_str().is_some(), "a string");
    require("cmd", |v| v.as_str().is_some(), "a string");
    require("pid", |v| v.as_u64().is_some(), "an integer");
    require("te", |v| v.as_u64().is_some(), "an integer");
    require("pps", |v| v.as_array().is_some(), "an array");
    require("ftbl", |v| v.as_array().is_some(), "an array");

    let block_lifetimes = root.get("bklt").and_then(Value::as_bool).unwrap_or(false);
    let block_accesses = root.get("bkacc").and_then(Value::as_bool).unwrap_or(false);
    if block_lifetimes {
        require("tg", |v| v.as_u64().is_some(), "an integer");
        require("tuth", |v| v.as_u64().is_some(), "an integer");
    } else {
        // Stricter than the viewer, which ignores these entirely when `bklt` is
        // false. `dh_main.c` documents them as *omitted* rather than zeroed, and
        // the distinction is the whole reason non-heap modes exist here: an
        // event was never live and never died, so a `tg` of 0 would be a
        // measured instant rather than the absence of one.
        for field in ["tg", "tuth"] {
            if root.contains_key(field) {
                problems.push(format!(
                    "top-level `{field}` is present with `bklt` false; it must be \
                     omitted, not zeroed"
                ));
            }
        }
    }

    // The mode decides the other three, so a file where they disagree describes
    // two different runs. The viewer checks none of this: it would render an
    // ad hoc profile with `verb` "Allocated" and heap columns of `NaN`.
    match root.get("mode").and_then(Value::as_str) {
        Some("heap") => check_mode(root, "Allocated", true, &mut problems),
        Some("ad-hoc") => check_mode(root, "Occurred", false, &mut problems),
        Some("copy") => check_mode(root, "Copied", false, &mut problems),
        Some(other) => problems.push(format!(
            "`mode` is `{other}`; DHAT has exactly three, and the viewer labels \
             every column from it"
        )),
        None => {}
    }

    if root.get("dhatFileVersion").and_then(Value::as_u64) != Some(2) {
        problems.push(String::from(
            "`dhatFileVersion` must be 2; the viewer refuses anything else, and \
             its version check runs *after* the field check, so an old viewer \
             reports a missing field instead",
        ));
    }

    if let (Some(peak), Some(end)) = (
        root.get("tg").and_then(Value::as_u64),
        root.get("te").and_then(Value::as_u64),
    ) {
        if peak > end {
            problems.push(format!(
                "the peak is at time {peak}, after the end of the run at {end}"
            ));
        }
    }

    // The frame table. Index 0 is the viewer's tree root and nothing may point
    // at it: the root is seeded with frame 0 before any program point is read.
    let frames = match root.get("ftbl").and_then(Value::as_array) {
        Some(frames) => frames,
        None => return problems,
    };
    match frames.first() {
        None => problems.push(String::from("`ftbl` is empty; it must start with `[root]`")),
        Some(Value::String(root_frame)) if root_frame == "[root]" => {}
        Some(other) => problems.push(format!("`ftbl[0]` is {other:?}, expected \"[root]\"")),
    }
    for (at, frame) in frames.iter().enumerate() {
        if frame.as_str().is_none() {
            problems.push(format!(
                "`ftbl[{at}]` is a {}, expected a string",
                frame.kind()
            ));
        }
    }

    let points = match root.get("pps").and_then(Value::as_array) {
        Some(points) => points,
        None => return problems,
    };

    let mut sequences: Vec<&[Value]> = Vec::new();
    let mut totals = Totals::default();
    for (at, point) in points.iter().enumerate() {
        check_point(
            point,
            at,
            frames.len(),
            block_lifetimes,
            block_accesses,
            &mut problems,
            &mut totals,
        );
        if let Some(sequence) = point.get("fs").and_then(Value::as_array) {
            if sequences.contains(&sequence) {
                problems.push(format!(
                    "`pps[{at}]` repeats a frame sequence used by an earlier \
                     program point; the viewer refuses such a file with `data \
                     file contains a repeated location`"
                ));
            }
            sequences.push(sequence);
        }
    }

    check_extension(root, &totals, points.len(), &mut problems);
    problems
}

/// Checks that `verb` and `bklt` say what the mode requires.
fn check_mode(
    root: &std::collections::BTreeMap<String, Value>,
    verb: &str,
    block_lifetimes: bool,
    problems: &mut Vec<String>,
) {
    let mode = root.get("mode").and_then(Value::as_str).unwrap_or("?");
    match root.get("verb").and_then(Value::as_str) {
        Some(found) if found == verb => {}
        Some(found) => problems.push(format!(
            "`mode` is `{mode}` but `verb` is `{found}`, expected `{verb}`"
        )),
        None => {}
    }
    if root.get("bklt").and_then(Value::as_bool) != Some(block_lifetimes) {
        problems.push(format!(
            "`mode` is `{mode}`, which {} block lifetimes, but `bklt` says \
             otherwise",
            if block_lifetimes { "has" } else { "has no" }
        ));
    }
    // Ad hoc weights are dimensionless. Left as bytes, a total of 5,000 retries
    // renders in the viewer as five kilobytes.
    let units = root.get("bksu").and_then(Value::as_str);
    if mode == "ad-hoc" && units != Some("events") {
        problems.push(format!(
            "an ad hoc profile names its counts `{}`, expected `events`; the \
             viewer's default is `blocks`, which is a unit these numbers do not \
             have",
            units.unwrap_or("blocks")
        ));
    }
}

/// Sums across program points, for cross-checking against the engine's own
/// global counters.
#[derive(Debug, Default)]
struct Totals {
    total_bytes: u64,
    total_blocks: u64,
    at_gmax_bytes: u64,
    at_gmax_blocks: u64,
    at_end_bytes: u64,
    at_end_blocks: u64,
}

fn check_point(
    point: &Value,
    at: usize,
    frame_count: usize,
    block_lifetimes: bool,
    block_accesses: bool,
    problems: &mut Vec<String>,
    totals: &mut Totals,
) {
    if point.as_object().is_none() {
        problems.push(format!(
            "`pps[{at}]` is a {}, expected an object",
            point.kind()
        ));
        return;
    }

    // Checked before the borrow below, and stricter than the viewer for the
    // same reason the top-level omissions are: an event was never live, so a
    // zero here would be a measurement of something that did not happen.
    if !block_lifetimes {
        for field in ["tl", "mb", "mbk", "gb", "gbk", "eb", "ebk"] {
            if point.get(field).is_some() {
                problems.push(format!(
                    "`pps[{at}].{field}` is present with `bklt` false; it must be \
                     omitted, not zeroed"
                ));
            }
        }
    }

    let mut integer = |field: &str| match point.get(field) {
        None => {
            problems.push(format!("`pps[{at}]` is missing `{field}`"));
            None
        }
        Some(value) => match value.as_u64() {
            Some(number) => Some(number),
            None => {
                problems.push(format!(
                    "`pps[{at}].{field}` is a {}, expected an integer",
                    value.kind()
                ));
                None
            }
        },
    };

    // `checkPP`, plus `tl` — which the viewer reads and never checks.
    let total_bytes = integer("tb");
    let total_blocks = integer("tbk");
    let (
        max_bytes,
        max_blocks,
        at_gmax_bytes,
        at_gmax_blocks,
        at_end_bytes,
        at_end_blocks,
        lifetime,
    ) = if block_lifetimes {
        (
            integer("mb"),
            integer("mbk"),
            integer("gb"),
            integer("gbk"),
            integer("eb"),
            integer("ebk"),
            integer("tl"),
        )
    } else {
        (None, None, None, None, None, None, None)
    };
    if block_accesses {
        integer("rb");
        integer("wb");
    }

    match point.get("fs") {
        None => problems.push(format!("`pps[{at}]` is missing `fs`")),
        Some(Value::Array(sequence)) => {
            for (position, frame) in sequence.iter().enumerate() {
                match frame.as_u64() {
                    None => problems.push(format!(
                        "`pps[{at}].fs[{position}]` is a {}, expected an integer",
                        frame.kind()
                    )),
                    Some(0) => problems.push(format!(
                        "`pps[{at}].fs[{position}]` is 0, which is the viewer's tree \
                         root and may not appear in a program point"
                    )),
                    Some(index) if index as usize >= frame_count => problems.push(format!(
                        "`pps[{at}].fs[{position}]` is {index}, past the end of a \
                         frame table with {frame_count} entries"
                    )),
                    Some(_) => {}
                }
            }
        }
        Some(other) => problems.push(format!(
            "`pps[{at}].fs` is a {}, expected an array",
            other.kind()
        )),
    }

    // Relationships that hold for any honest accounting. None of these is
    // checked by the viewer; all of them would be a bug in the engine.
    let mut ordered = |larger: Option<u64>, smaller: Option<u64>, why: &str| {
        if let (Some(larger), Some(smaller)) = (larger, smaller) {
            if larger < smaller {
                problems.push(format!("`pps[{at}]`: {why} ({larger} < {smaller})"));
            }
        }
    };
    ordered(
        total_bytes,
        max_bytes,
        "more bytes were live at once than were ever allocated",
    );
    ordered(
        total_blocks,
        max_blocks,
        "more blocks were live at once than were ever allocated",
    );
    ordered(
        max_bytes,
        at_gmax_bytes,
        "more bytes were live at the global peak than this point ever had live",
    );
    ordered(
        max_blocks,
        at_gmax_blocks,
        "more blocks were live at the global peak than this point ever had live",
    );
    ordered(
        max_bytes,
        at_end_bytes,
        "more bytes were live at the end than this point ever had live",
    );
    ordered(
        max_blocks,
        at_end_blocks,
        "more blocks were live at the end than this point ever had live",
    );

    // Bytes cannot exist outside of blocks. Every pairing is checked because a
    // bug that loses one counter but not its partner shows up here and nowhere
    // else in the file.
    let mut accompanied = |bytes: Option<u64>, blocks: Option<u64>, when: &str| {
        if let (Some(bytes), Some(blocks)) = (bytes, blocks) {
            if bytes > 0 && blocks == 0 {
                problems.push(format!(
                    "`pps[{at}]`: {bytes} bytes {when} but no blocks holding them"
                ));
            }
        }
    };
    accompanied(total_bytes, total_blocks, "were allocated");
    accompanied(max_bytes, max_blocks, "were live at once");
    accompanied(
        at_gmax_bytes,
        at_gmax_blocks,
        "were live at the global peak",
    );
    accompanied(at_end_bytes, at_end_blocks, "were live at the end");

    // The viewer decompresses `acc` unconditionally, whatever `bkacc` says, and
    // asserts on any value it cannot represent. We never write the field; a
    // validator that ignored it would nonetheless accept a file the viewer
    // refuses, which is the one thing it exists to prevent.
    match point.get("acc") {
        None => {}
        Some(Value::Array(accesses)) => check_accesses(accesses, at, problems),
        Some(other) => problems.push(format!(
            "`pps[{at}].acc` is a {}, expected an array",
            other.kind()
        )),
    }

    if total_blocks == Some(0) {
        problems.push(format!(
            "`pps[{at}]` records no blocks at all and should not have been emitted"
        ));
    }
    if let (Some(lifetime), Some(0)) = (lifetime, total_blocks) {
        if lifetime > 0 {
            problems.push(format!("`pps[{at}]` has a lifetime but no blocks"));
        }
    }

    totals.total_bytes += total_bytes.unwrap_or(0);
    totals.total_blocks += total_blocks.unwrap_or(0);
    totals.at_gmax_bytes += at_gmax_bytes.unwrap_or(0);
    totals.at_gmax_blocks += at_gmax_blocks.unwrap_or(0);
    totals.at_end_bytes += at_end_bytes.unwrap_or(0);
    totals.at_end_blocks += at_end_blocks.unwrap_or(0);
}

/// Checks a run-length encoded access array the way `dh_view.js` decodes it.
///
/// A negative element is a repeat count for the element after it; every value
/// must fit in the viewer's `0..=0xffff` latch, and `normalizeAccess` asserts
/// rather than warning on anything else.
fn check_accesses(accesses: &[Value], at: usize, problems: &mut Vec<String>) {
    let mut index = 0;
    while index < accesses.len() {
        let Value::Number(raw) = &accesses[index] else {
            problems.push(format!(
                "`pps[{at}].acc[{index}]` is a {}, expected an integer",
                accesses[index].kind()
            ));
            return;
        };
        let Ok(value) = raw.parse::<i64>() else {
            problems.push(format!("`pps[{at}].acc[{index}]` is not an integer: {raw}"));
            return;
        };
        if value < 0 {
            // A repeat count, which must be followed by the value to repeat.
            index += 1;
            if index >= accesses.len() {
                problems.push(format!(
                    "`pps[{at}].acc` ends with a repeat count and no value to repeat"
                ));
                return;
            }
            let Some(repeated) = accesses[index].as_u64() else {
                problems.push(format!(
                    "`pps[{at}].acc[{index}]` follows a repeat count and must be an integer"
                ));
                return;
            };
            if repeated > 0xffff {
                problems.push(format!(
                    "`pps[{at}].acc[{index}]` is {repeated}; the viewer asserts above 0xffff"
                ));
            }
        } else if value > 0xffff {
            problems.push(format!(
                "`pps[{at}].acc[{index}]` is {value}; the viewer asserts above 0xffff"
            ));
        }
        index += 1;
    }
}

/// Checks the module map, which is what makes the recorded addresses resolvable
/// after the process that produced them is gone.
///
/// A map that is unsorted, or whose images overlap, gives an address to the
/// wrong image — and the resulting function name is wrong in the most
/// convincing way possible, because it looks like a real answer.
fn check_modules(extension: &Value, problems: &mut Vec<String>) {
    let Some(modules) = extension.get("modules") else {
        problems.push(String::from("the `heapscope` section has no `modules`"));
        return;
    };
    let Some(modules) = modules.as_array() else {
        problems.push(format!(
            "`heapscope.modules` is a {}, expected an array",
            modules.kind()
        ));
        return;
    };

    if modules.is_empty() {
        problems.push(String::from(
            "`heapscope.modules` is empty; every process has at least the image \
             it is running",
        ));
    }

    // The furthest any earlier image reaches, not just the previous one: an
    // image nested inside a larger earlier one is exactly the case a
    // previous-entry comparison misses.
    let mut furthest_end = 0u64;
    let mut previous_start = 0u64;
    for (at, module) in modules.iter().enumerate() {
        let Some(load) = module.get("start").and_then(Value::as_u64) else {
            problems.push(format!("`heapscope.modules[{at}]` has no integer `start`"));
            continue;
        };
        let Some(size) = module.get("size").and_then(Value::as_u64) else {
            problems.push(format!("`heapscope.modules[{at}]` has no integer `size`"));
            continue;
        };
        if size == 0 {
            problems.push(format!(
                "`heapscope.modules[{at}]` has no extent, so no address can be in it"
            ));
        }
        for required in ["load", "bias"] {
            if module.get(required).and_then(Value::as_u64).is_none() {
                problems.push(format!(
                    "`heapscope.modules[{at}]` has no integer `{required}`"
                ));
            }
        }
        // An empty path is not "a path we happen not to know": it is an image
        // whose frames cannot be symbolized by anything, which is the failure
        // this whole map exists to prevent.
        match module.get("path").and_then(Value::as_str) {
            Some(path) if !path.is_empty() => {}
            _ => problems.push(format!(
                "`heapscope.modules[{at}]` has no path, so nothing in it can be \
                 symbolized"
            )),
        }
        if let Some(build_id) = module.get("buildId") {
            match build_id.as_str() {
                Some(text) if text.bytes().all(|b| b.is_ascii_hexdigit()) && !text.is_empty() => {}
                _ => problems.push(format!(
                    "`heapscope.modules[{at}].buildId` should be non-empty hexadecimal"
                )),
            }
        }

        if at > 0 && load < previous_start {
            problems.push(format!(
                "`heapscope.modules[{at}]` starts at {load:#x}, before the \
                 previous entry at {previous_start:#x}; lookups bisect and need \
                 the map sorted"
            ));
        }
        if at > 0 && load < furthest_end {
            problems.push(format!(
                "`heapscope.modules[{at}]` starts at {load:#x}, inside an image \
                 that reaches {furthest_end:#x}; an address in the overlap would \
                 resolve against the wrong file"
            ));
        }
        previous_start = load;
        furthest_end = furthest_end.max(load.saturating_add(size));
    }
}

/// Every value the `shutdown` field is allowed to take.
///
/// Spelled out rather than "any non-empty string" on purpose: the field exists
/// so that a reader can tell a profile written before teardown from one written
/// partway through it, and a value neither the reader nor this list recognises
/// answers that question no better than a missing field would.
const SHUTDOWN_PATHS: &[&str] = &["running", "drop", "atexit", "explicit", "forked-child"];

/// Every unwinder a profile may name.
const UNWINDERS: &[&str] = &["frame-pointer", "system"];

/// Checks that the profile says which unwinder captured its frames.
///
/// The two do not agree about how deep a trace goes or where it stops, so a
/// profile that does not say which it used cannot be compared with another.
fn check_unwinder(extension: &Value, problems: &mut Vec<String>) {
    match extension.get("unwinder").and_then(Value::as_str) {
        Some(name) if UNWINDERS.contains(&name) => {}
        Some(name) => problems.push(format!(
            "`heapscope.unwinder` is {name:?}, which is not one of {UNWINDERS:?}"
        )),
        None => problems.push(String::from(
            "the `heapscope` section has no `unwinder`, so nothing says which \
             unwinder produced these frames",
        )),
    }
}

/// Checks the capture-quality counts.
///
/// Required, not optional. These counters existed for two milestones with
/// nothing incrementing them while four separate comments described the field
/// they were supposed to appear in — so the rule is that the field is present
/// and that it accounts for something.
fn check_captures(extension: &Value, points: usize, problems: &mut Vec<String>) {
    let Some(captures) = extension
        .get("selfMetrics")
        .and_then(|metrics| metrics.get("captures"))
    else {
        problems.push(String::from(
            "the `heapscope.selfMetrics` section has no `captures`, so nothing \
             says how many stack walks came back whole",
        ));
        return;
    };

    let mut total = 0u64;
    for field in ["complete", "truncated", "suspect", "noFrames"] {
        match captures.get(field).and_then(Value::as_u64) {
            Some(count) => total += count,
            None => problems.push(format!(
                "`heapscope.selfMetrics.captures` has no integer `{field}`"
            )),
        }
    }

    // A profile with program points recorded at least that many captures. Zero
    // here alongside real points is the shape of a counter nobody increments.
    if points > 0 && total == 0 {
        problems.push(format!(
            "the profile has {points} program points but recorded 0 captures; \
             the capture counters are not being incremented"
        ));
    }
}

/// Checks that the profile says which path produced it.
fn check_shutdown(extension: &Value, problems: &mut Vec<String>) {
    match extension.get("shutdown").and_then(Value::as_str) {
        Some(path) if SHUTDOWN_PATHS.contains(&path) => {}
        Some(path) => problems.push(format!(
            "`heapscope.shutdown` is {path:?}, which is not one of {SHUTDOWN_PATHS:?}"
        )),
        None => problems.push(String::from(
            "the `heapscope` section has no `shutdown`, so nothing says whether \
             this profile was taken before teardown or partway through it",
        )),
    }
}

/// Cross-checks the per-point columns against the engine's own global counters,
/// which the profile carries in its `heapscope` section.
///
/// This is the invariant the peak gate exists to guarantee: the per-point
/// at-peak bytes must sum to the global peak. Nothing in the DHAT format
/// requires it, and no viewer would notice if it were false.
///
/// The section is **required**, not optional. It is the only thing that makes
/// this check possible, so treating its absence as "nothing to check" would let
/// the strongest rule in the validator disable itself the moment the emitter
/// stopped writing the section.
fn check_extension(
    root: &std::collections::BTreeMap<String, Value>,
    totals: &Totals,
    points: usize,
    problems: &mut Vec<String>,
) {
    let Some(extension) = root.get("heapscope") else {
        problems.push(String::from(
            "the profile has no `heapscope` section, so the per-point columns \
             cannot be cross-checked against the engine's own counters",
        ));
        return;
    };
    check_modules(extension, problems);
    check_shutdown(extension, problems);
    check_unwinder(extension, problems);
    check_captures(extension, points, problems);

    let Some(globals) = extension.get("totals") else {
        problems.push(String::from("the `heapscope` section has no `totals`"));
        return;
    };
    // Two conditions under which the columns legitimately need not agree. Both
    // are stated by the profile itself, and both are rare enough that a test
    // suite where every profile skipped the cross-check would be visibly odd.
    if extension.get("exact").and_then(Value::as_bool) != Some(true) {
        return;
    }
    if extension.get("droppedPoints").and_then(Value::as_u64) != Some(0) {
        return;
    }

    let mut agrees = |field: &str, summed: u64, what: &str| {
        if let Some(global) = globals.get(field).and_then(Value::as_u64) {
            if global != summed {
                problems.push(format!(
                    "the per-point {what} sum to {summed}, but the engine recorded \
                     {global} in `heapscope.totals.{field}`"
                ));
            }
        }
    };
    agrees("totalBytes", totals.total_bytes, "`tb` columns");
    agrees("totalBlocks", totals.total_blocks, "`tbk` columns");
    agrees("maxBytes", totals.at_gmax_bytes, "`gb` columns");
    agrees("maxBlocks", totals.at_gmax_blocks, "`gbk` columns");
    agrees("currBytes", totals.at_end_bytes, "`eb` columns");
    agrees("currBlocks", totals.at_end_blocks, "`ebk` columns");
}

/// Panics with every problem found, or returns quietly.
#[track_caller]
pub fn assert_valid(text: &str) {
    let problems = problems(text);
    assert!(
        problems.is_empty(),
        "the profile is not a valid DHAT v2 file:\n  - {}\n\n{}",
        problems.join("\n  - "),
        excerpt(text)
    );
}

/// The first part of a file, for a failure message.
fn excerpt(text: &str) -> String {
    const LIMIT: usize = 4_000;
    if text.len() <= LIMIT {
        return text.to_string();
    }
    let mut end = LIMIT;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n... ({} bytes in total)", &text[..end], text.len())
}
