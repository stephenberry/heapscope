//! The native emitter, and the claim that the DHAT file is a projection of it.
//!
//! Like `tests/dhat_output.rs`, these build snapshots by hand rather than
//! recording a workload: a program point whose capture found no frames, an
//! address above 2^53, a size class at the top of the range, and counters at
//! `u64::MAX` are not things a workload produces on demand.
//!
//! Two kinds of test are here, and the second is the load-bearing one.
//!
//! The first kind checks the emitter against the validator in
//! `support/native.rs`, and — because a validator that accepts everything makes
//! every such test pass — checks that each of the validator's rules rejects a
//! file that breaks it.
//!
//! The second kind checks that **the DHAT file loses nothing the native file
//! keeps**. PLAN.md section 3.4 calls the DHAT emitter one lossy projection of
//! the native format. Without a test, "projection" is a word: nothing would stop
//! the two emitters drifting until a number appeared in one file and not the
//! other, or appeared in both with different values, and the profile a person
//! opened would be the one that was wrong.

mod support;

use heapscope::internals::shape::{Shape, Shapes};
use heapscope::output::{
    PointKind, ProgramPoint, RegionStats, Snapshot, TableUsage, TallyStats, ThreadStats,
};
use heapscope::symbol::modules::Module;
use heapscope::Mode;
use proptest::prelude::*;
use support::json::{self, Value};
use support::native;
use support::snapshot::{as_mode, hand_built, point};

/// Blocks the snapshot helper will build a shape histogram for.
const HISTOGRAM_LIMIT: u64 = 100_000;

/// A hand-built profile, plus everything the native format carries that DHAT v2
/// has no field for — which is what this suite is about.
fn snapshot(points: Vec<ProgramPoint>) -> Snapshot {
    let mut snapshot = hand_built(points);
    let stats = snapshot.stats;

    // Every observed request either counted toward the totals or was dropped,
    // which is the invariant the validator checks. Built here so that a
    // hand-made snapshot is as coherent as a recorded one.
    //
    // One `record` per block, because that is the only way a histogram is
    // built — the counters are private, deliberately, so that a test cannot
    // construct a distribution no sequence of requests could produce. A block
    // count past `HISTOGRAM_LIMIT` is a saturated counter rather than a number
    // of loop iterations, and such a snapshot carries no shapes at all; the
    // validator accepts that, because it is what a non-heap run has.
    let shapes = Shapes::new();
    if stats.total_blocks <= HISTOGRAM_LIMIT {
        for _ in 0..stats.total_blocks {
            shapes.record(Shape::of(24).aligned(8));
        }
    }

    snapshot.shapes = shapes.snapshot();
    // `SelfMetrics` is `#[non_exhaustive]` too, so this assigns rather than
    // builds: M6 adds a sampling rate to it, and that must not break this file.
    snapshot.metrics.program_points = TableUsage {
        entries: 2,
        capacity: 1024,
        bytes: 4096,
    };
    snapshot.metrics.live_blocks = TableUsage {
        entries: 1,
        capacity: 2048,
        bytes: 8192,
    };
    snapshot.metrics.threads = TableUsage {
        entries: 2,
        capacity: 4096,
        bytes: 192,
    };
    snapshot.metrics.regions = TableUsage {
        entries: 1,
        capacity: 256,
        bytes: 152,
    };
    // Three thread rows, because every recorded allocation belongs to exactly
    // one and the rows have to sum to the run — a hand-built snapshot is held to
    // that the same way a recorded one is. The split is uneven so that a summing
    // mistake cannot land on the right answer by symmetry; the second row is
    // unnamed because most threads are; and the third is the shared overflow
    // row, without which four validator rules about it could never fire.
    let (major, minor, spare) = shares(stats.total_bytes);
    let (major_blocks, minor_blocks, spare_blocks) = shares(stats.total_blocks);
    let (major_live, minor_live, spare_live) = shares(stats.curr_bytes);
    let (major_live_blocks, minor_live_blocks, spare_live_blocks) = shares(stats.curr_blocks);
    // A row's peak is a share of the *run's* peak, not of its own total. The two
    // differ, and taking the wrong one produced rows that had held more than the
    // whole heap ever did — which is a shape no run can take, and which the
    // validator now says so about.
    let (major_peak, minor_peak, spare_peak) = shares(stats.max_bytes);
    let (major_peak_blocks, minor_peak_blocks, spare_peak_blocks) = shares(stats.max_blocks);
    snapshot.threads = vec![
        ThreadStats {
            id: 0,
            overflow: false,
            name: Some(String::from("main")),
            first_seen: 0,
            counts: TallyStats {
                total_bytes: major,
                total_blocks: major_blocks,
                curr_bytes: major_live,
                curr_blocks: major_live_blocks,
                max_bytes: major_peak,
                max_blocks: major_peak_blocks,
            },
        },
        ThreadStats {
            id: 1,
            overflow: false,
            name: None,
            first_seen: 7,
            counts: TallyStats {
                total_bytes: minor,
                total_blocks: minor_blocks,
                curr_bytes: minor_live,
                curr_blocks: minor_live_blocks,
                max_bytes: minor_peak,
                max_blocks: minor_peak_blocks,
            },
        },
        // The shared row: no name, no `firstSeen`, because it stands for many
        // threads with many of each.
        ThreadStats {
            id: u16::MAX - 1,
            overflow: true,
            name: None,
            first_seen: 0,
            counts: TallyStats {
                total_bytes: spare,
                total_blocks: spare_blocks,
                curr_bytes: spare_live,
                curr_blocks: spare_live_blocks,
                max_bytes: spare_peak,
                max_blocks: spare_peak_blocks,
            },
        },
    ];
    // One region, holding a strict subset of the run: an allocation made
    // outside every region belongs to no row, so these do not sum to the
    // totals and the validator must not expect them to.
    snapshot.regions = vec![RegionStats {
        id: 0,
        overflow: false,
        name: Some(String::from("parsing")),
        first_seen: 3,
        entries: 2,
        active: 0,
        counts: TallyStats {
            total_bytes: minor,
            total_blocks: minor_blocks,
            curr_bytes: 0,
            curr_blocks: 0,
            max_bytes: minor_peak,
            max_blocks: minor_peak_blocks,
        },
    }];
    snapshot.modules = vec![
        Module {
            path: String::from("/bin/example"),
            start: 0x1000,
            size: 0x1000,
            // A non-zero bias, so a test asserting on a file address would
            // notice the emitter using the wrong number.
            bias: 0x400,
            image_base: 0x1000,
            build_id: Some(String::from("0badc0ffee")),
        },
        Module {
            path: String::from("/lib/libc.so"),
            start: 0x4000,
            size: 0x2000,
            bias: 0x4000,
            image_base: 0x4000,
            build_id: None,
        },
    ];
    snapshot
}

/// Splits a total unevenly across three rows, keeping the sum exact.
///
/// Uneven so that a validator summing the wrong field, or one row twice, cannot
/// land on the right answer by symmetry. The same proportions are applied to
/// every field, so that a row's live share never exceeds its total share and
/// the rows stay internally coherent for any input — including the saturated
/// counters the property test generates.
fn shares(total: u64) -> (u64, u64, u64) {
    let minor = total / 4;
    let spare = total / 8;
    (total - minor - spare, minor, spare)
}

fn emit(snapshot: &Snapshot) -> String {
    let mut buffer = Vec::new();
    snapshot
        .write_native(&mut buffer)
        .expect("writing to a Vec cannot fail");
    String::from_utf8(buffer).expect("valid UTF-8")
}

fn parse(text: &str) -> Value {
    json::parse(text).expect("the emitter produces valid JSON")
}

/// `text` with one field replaced or removed, so that a rule can be shown to
/// reject the file it exists to reject.
///
/// Textual rather than structural because the test JSON support is a parser and
/// not a writer, and because the damage a rule guards against is what a wrong
/// emitter would *write*, not a tree someone built.
fn damaged_by(text: &str, damage: impl Fn(&str) -> String) -> Vec<String> {
    let damaged = damage(text);
    assert_ne!(damaged, text, "the damage did not change the profile");
    native::problems(&damaged)
}

/// `text` with `from` replaced by `to`, asserting `from` was there to replace.
///
/// A bare `str::replace` that stops matching is a test that silently checks
/// nothing: [`damaged_by`] only sees that *something* changed, so a chain of two
/// replacements still passes when only the second applies — and the rule the
/// test names is then never exercised again. Every literal below is a fragment
/// of the emitter's own output, so any layout change should stop this file
/// loudly rather than quietly.
fn replacing(text: &str, from: &str, to: &str) -> String {
    assert!(
        text.contains(from),
        "the profile has no {from:?} to damage, so this test no longer damages \
         what it names:\n{text}"
    );
    text.replace(from, to)
}

fn rejects(problems: &[String], fragment: &str) {
    assert!(
        problems.iter().any(|problem| problem.contains(fragment)),
        "no problem mentioned {fragment:?}; got {problems:?}"
    );
}

// ---- the emitter ----

#[test]
fn a_recorded_profile_is_valid() {
    let text = emit(&snapshot(vec![
        point(&[0x1500, 0x1600], 4096, 8),
        point(&[0x4500], 512, 1),
    ]));
    native::assert_valid(&text);
}

#[test]
fn every_mode_produces_a_valid_profile() {
    for mode in [Mode::Heap, Mode::AdHoc, Mode::Copy] {
        let snapshot = as_mode(snapshot(vec![point(&[0x1500], 4096, 8)]), mode);
        native::assert_valid(&emit(&snapshot));
    }
}

#[test]
fn a_profile_of_a_run_that_recorded_nothing_is_valid() {
    native::assert_valid(&emit(&Snapshot::default()));
}

/// A frame is resolved against the module map the snapshot carries, and the
/// three parts are separate answers rather than one line of text. `fileAddr` is
/// the number `addr2line` takes, which is the address minus the bias — not an
/// offset from the load address.
#[test]
fn a_frame_carries_its_image_and_its_file_address_apart() {
    let text = emit(&snapshot(vec![point(&[0x1500, 0x4500], 4096, 8)]));
    native::assert_valid(&text);
    let profile = parse(&text);
    let frames = profile.get("frames").and_then(Value::as_array).unwrap();

    assert_eq!(
        frames[0].get("addr").and_then(Value::as_str),
        Some("0x1500")
    );
    assert_eq!(frames[0].get("module").and_then(Value::as_u64), Some(0));
    assert_eq!(
        frames[0].get("fileAddr").and_then(Value::as_str),
        Some("0x1100"),
        "the file address is the runtime address minus the image's bias"
    );

    assert_eq!(frames[1].get("module").and_then(Value::as_u64), Some(1));
    assert_eq!(
        frames[1].get("fileAddr").and_then(Value::as_str),
        Some("0x500")
    );
}

/// An address in no image gets no image, rather than being attributed to
/// whichever one happens to be nearest. This is the case a truncated stack walk
/// produces, and it is exactly where a confident wrong answer does the most
/// damage.
#[test]
fn an_address_in_no_image_is_left_unattributed() {
    let text = emit(&snapshot(vec![point(&[0xDEAD_0000], 4096, 8)]));
    native::assert_valid(&text);
    let profile = parse(&text);
    let frame = &profile.get("frames").and_then(Value::as_array).unwrap()[0];

    assert_eq!(
        frame.get("addr").and_then(Value::as_str),
        Some("0xdead0000")
    );
    assert!(frame.get("module").is_none(), "{frame:?}");
    assert!(frame.get("fileAddr").is_none(), "{frame:?}");
    assert!(
        frame.get("symbol").is_none(),
        "an address in no image was named, which means the module map was not \
         consulted before the platform was: {frame:?}"
    );
}

/// The reason addresses are strings. A JSON number is a double in JavaScript,
/// exact only to 2^53; the bundled viewer of PLAN.md section 6.12 will
/// `JSON.parse` this file, and an address that comes back rounded names the
/// wrong line of the wrong function with nothing about it looking wrong.
#[test]
fn an_address_above_two_to_the_fifty_third_survives_exactly() {
    let address = 0x0020_0000_0000_0001usize;
    assert!(address > (1u64 << 53) as usize);

    let text = emit(&snapshot(vec![point(&[address], 4096, 8)]));
    native::assert_valid(&text);
    let profile = parse(&text);
    let written = profile.get("frames").and_then(Value::as_array).unwrap()[0]
        .get("addr")
        .and_then(Value::as_str)
        .expect("a frame address");

    assert_eq!(written, format!("0x{address:x}"));
    assert_eq!(
        usize::from_str_radix(written.trim_start_matches("0x"), 16).unwrap(),
        address,
        "the address did not survive the round trip"
    );
}

/// Nothing folds here. Two points that a rendering would collapse onto one
/// frame list are two points, because the fold exists to satisfy a viewer that
/// refuses a file with a repeated location and that is not a fact about the run.
#[test]
fn points_that_a_rendering_would_collapse_stay_apart() {
    let text = emit(&snapshot(vec![
        point(&[0x1500], 4096, 8),
        point(&[0x1600], 512, 1),
    ]));
    native::assert_valid(&text);
    let profile = parse(&text);
    let points = profile.get("points").and_then(Value::as_array).unwrap();

    assert_eq!(points.len(), 2);
    assert_ne!(
        points[0].get("frames").and_then(Value::as_array),
        points[1].get("frames").and_then(Value::as_array)
    );
}

/// One entry per distinct address, however many points share it. A program with
/// a thousand call sites shares its outermost frames across all of them, and on
/// Windows every repeat would otherwise be a lock and a dbghelp call.
#[test]
fn a_shared_frame_is_resolved_once() {
    let text = emit(&snapshot(vec![
        point(&[0x1500, 0x1600], 4096, 8),
        point(&[0x1520, 0x1600], 512, 1),
    ]));
    native::assert_valid(&text);
    let profile = parse(&text);

    let frames = profile.get("frames").and_then(Value::as_array).unwrap();
    assert_eq!(frames.len(), 3, "0x1600 was resolved twice: {frames:?}");

    let points = profile.get("points").and_then(Value::as_array).unwrap();
    let outermost = |at: usize| {
        points[at]
            .get("frames")
            .and_then(Value::as_array)
            .unwrap()
            .last()
            .and_then(Value::as_u64)
    };
    assert_eq!(outermost(0), outermost(1));
}

/// DHAT has one `tl` field, so its emitter has to add the two lifetimes
/// together before writing them. Losing that distinction is the kind of loss
/// this format exists to prevent: a site that allocates and holds is not the
/// same as one whose blocks were short-lived.
#[test]
fn the_two_lifetime_totals_stay_apart() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    let profile = parse(&text);
    let first = &profile.get("points").and_then(Value::as_array).unwrap()[0];

    assert_eq!(
        first.get("retiredLifetime").and_then(Value::as_u64),
        Some(17 * 8)
    );
    assert_eq!(
        first.get("unretiredLifetime").and_then(Value::as_u64),
        Some(3)
    );
}

/// The overflow point and a point whose stack could not be walked are both
/// frameless and have opposite remedies — raise the ceiling, or fix the build.
#[test]
fn the_two_frameless_conditions_are_distinguishable() {
    let mut overflow = point(&[], 4096, 8);
    overflow.kind = PointKind::Overflow;
    let text = emit(&snapshot(vec![overflow, point(&[], 512, 1)]));
    native::assert_valid(&text);
    let profile = parse(&text);

    let kinds: Vec<_> = profile
        .get("points")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|point| point.get("kind").and_then(Value::as_str))
        .collect();
    assert_eq!(kinds, [Some("overflow"), Some("recorded")]);
}

/// Every borrowed string reaches a reader through this file: symbol names from a
/// table that may be truncated, image paths from a filesystem where a directory
/// may be named anything, and `argv`.
#[test]
fn strings_the_profiler_did_not_write_are_screened() {
    let mut snapshot = snapshot(vec![point(&[0x1500], 4096, 8)]);
    snapshot.command = String::from("prog \u{1b}[2J \u{202e}gnp.eslaf");
    snapshot.modules[0].path = String::from("/tmp/\u{202e}exe\u{0}");
    let text = emit(&snapshot);

    native::assert_valid(&text);
    for hostile in ['\u{1b}', '\u{202e}', '\u{0}'] {
        assert!(
            !text.contains(hostile),
            "{hostile:?} survived into the profile"
        );
    }
}

// ---- who allocated, and what for ----

/// The rows carry what DHAT v2 has nowhere to put: which thread the bytes
/// belong to and which phase they were allocated in.
#[test]
fn a_profile_names_the_threads_and_regions_that_allocated() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    native::assert_valid(&text);
    let root = parse(&text);

    let threads = root
        .get("threads")
        .and_then(Value::as_array)
        .expect("a profile names the threads that allocated");
    assert_eq!(threads.len(), 3);
    assert_eq!(threads[0].get("name").and_then(Value::as_str), Some("main"));
    assert!(
        threads[1].get("name").is_none(),
        "a thread nobody named must carry no name rather than an empty one, or \
         a reader cannot tell the two apart"
    );

    let regions = root
        .get("regions")
        .and_then(Value::as_array)
        .expect("a profile names the regions the program entered");
    assert_eq!(regions.len(), 1);
    assert_eq!(
        regions[0].get("name").and_then(Value::as_str),
        Some("parsing")
    );
    assert_eq!(regions[0].get("entries").and_then(Value::as_u64), Some(2));
    assert_eq!(
        regions[0].get("active").and_then(Value::as_u64),
        Some(0),
        "a region left open at the end is a fact about the run and has to be \
         in the file"
    );
}

/// The rule that ties the rows to the rest of the file. Without it the section
/// is decorative: any set of numbers would be as good as the recorded ones.
#[test]
fn the_validator_rejects_thread_rows_that_do_not_sum_to_the_totals() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            replacing(
                t,
                r#""firstSeen":0,"name":"main","totalBytes":2560"#,
                r#""firstSeen":0,"name":"main","totalBytes":2559"#,
            )
        }),
        "thread rows account for",
    );
}

/// One byte out is enough, because the sums are an equality wherever the file
/// says the counters were read under exclusion. A tolerance here would be a
/// rule that passes while the rows are adrift, which is what it was before the
/// attribution moved inside the peak gate.
#[test]
fn the_thread_sums_are_exact_rather_than_approximate() {
    let text = emit(&snapshot(vec![point(&[0x1500], 1_000_000, 8)]));
    // Both the row's own total and its peak, so that the only rule left to
    // fire is the sum against the run.
    let problems = damaged_by(&text, |t| replacing(t, "625000", "624999"));
    assert_eq!(
        problems.len(),
        1,
        "one byte in a million should be caught by the sum rule and nothing \
         else: {problems:?}"
    );
    rejects(&problems, "thread rows account for");
}

/// A file that dropped rows says so, and the sums are then unenforceable — so
/// the rule stands down rather than reporting the profiler's own honesty as a
/// defect.
#[test]
fn dropped_rows_suspend_the_sum_rule_rather_than_failing_it() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    let problems = damaged_by(&text, |t| {
        // A row that is internally coherent and simply does not add up, which
        // is exactly what a dropped row leaves behind.
        let short = replacing(
            t,
            r#"{"id":1,"firstSeen":7,"totalBytes":1024,"totalBlocks":2,"currBytes":128,"currBlocks":0,"maxBytes":256,"maxBlocks":0}"#,
            r#"{"id":1,"firstSeen":7,"totalBytes":512,"totalBlocks":1,"currBytes":128,"currBlocks":0,"maxBytes":256,"maxBlocks":0}"#,
        );
        replacing(&short, r#""attributionRows":0"#, r#""attributionRows":1"#)
    });
    assert!(
        problems.is_empty(),
        "a profile that says it dropped rows was rejected for the rows it \
         dropped: {problems:?}"
    );
}

/// A run that recorded blocks was recorded *by* something. Without this rule an
/// emitter that dropped the array entirely would satisfy the sums at zero.
#[test]
fn the_validator_rejects_a_profile_that_names_no_thread() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            let start = t.find(r#""threads":["#).expect("the threads array");
            let end = t[start..].find("\n ],").expect("its end") + start + "\n ],".len();
            format!("{}\"threads\":[],{}", &t[..start], &t[end..])
        }),
        "names no thread",
    );
}

/// A region is attributed to the innermost open one only, so the rows partition
/// the run rather than overlapping. Rows that sum past the totals would mean
/// nesting was being double-counted.
#[test]
fn the_validator_rejects_regions_that_account_for_more_than_the_run() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            replacing(
                t,
                r#""firstSeen":3,"name":"parsing","entries":2,"active":0,"totalBytes":1024"#,
                r#""firstSeen":3,"name":"parsing","entries":2,"active":0,"totalBytes":99999"#,
            )
        }),
        "more than the",
    );
}

/// A row cannot be holding more than the most it ever held, and cannot have
/// peaked above what it ever allocated. Both are arithmetic the row does for
/// itself, so failing either means a counter moved without its pair.
#[test]
fn the_validator_rejects_a_row_holding_more_than_its_own_peak() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            replacing(
                t,
                r#""currBytes":320,"currBlocks":1"#,
                r#""currBytes":4000,"currBlocks":1"#,
            )
        }),
        "more than its own peak",
    );
}

/// A region row exists because a name was entered. One that was never entered
/// is a row the profiler invented.
#[test]
fn the_validator_rejects_a_region_that_was_never_entered() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            replacing(t, r#""entries":2,"active":0"#, r#""entries":0,"active":0"#)
        }),
        "never entered",
    );
}

#[test]
fn the_validator_rejects_a_region_open_more_times_than_it_was_entered() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            replacing(t, r#""entries":2,"active":0"#, r#""entries":2,"active":3"#)
        }),
        "is open 3 times",
    );
}

/// A row id is what a reader joins on, so two rows sharing one is a file that
/// cannot be read rather than one with a cosmetic flaw.
#[test]
fn the_validator_rejects_two_rows_with_the_same_id() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            replacing(t, r#"{"id":1,"firstSeen":7"#, r#"{"id":0,"firstSeen":7"#)
        }),
        "twice",
    );
}

/// The same rule `totals` follows: an event was never live, so a mode with no
/// block lifetimes must omit the live and peak columns rather than zero them.
#[test]
fn an_attribution_row_in_a_non_heap_mode_carries_no_live_columns() {
    for mode in [Mode::AdHoc, Mode::Copy] {
        let text = emit(&as_mode(snapshot(vec![point(&[0x1500], 4096, 8)]), mode));
        native::assert_valid(&text);
        let root = parse(&text);
        for row in root
            .get("threads")
            .and_then(Value::as_array)
            .expect("threads")
        {
            for field in ["currBytes", "currBlocks", "maxBytes", "maxBlocks"] {
                assert!(
                    row.get(field).is_none(),
                    "a {mode} profile's thread row carries `{field}`, which is a \
                     measurement of something an event never had"
                );
            }
        }
        rejects(
            &damaged_by(&text, |t| {
                replacing(
                    t,
                    r#""firstSeen":0,"name":"main","#,
                    r#""firstSeen":0,"name":"main","currBytes":0,"#,
                )
            }),
            "in a mode with no live blocks",
        );
    }
}

/// The `currBytes` half of the row-sum equality carries the loudest claim this
/// chunk makes — that a free brings the *allocating* thread down — and had no
/// test that could see it broken. `totalBlocks` likewise.
#[test]
fn the_validator_rejects_rows_that_do_not_sum_on_every_field() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            replacing(
                t,
                r#""currBytes":320,"currBlocks":1"#,
                r#""currBytes":319,"currBlocks":1"#,
            )
        }),
        "`currBytes` but the run recorded",
    );
    rejects(
        &damaged_by(&text, |t| {
            replacing(
                t,
                r#""totalBytes":2560,"totalBlocks":5"#,
                r#""totalBytes":2560,"totalBlocks":4"#,
            )
        }),
        "`totalBlocks` but the run recorded",
    );
}

/// A row's peak is a share of the run's, not a quantity beside it. This is the
/// one row rule that reaches outside the row, so it is the one that catches an
/// emitter inventing numbers rather than mis-summing real ones.
#[test]
fn the_validator_rejects_a_row_that_peaked_above_the_whole_heap() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            replacing(
                t,
                r#""maxBytes":640,"maxBlocks":1}"#,
                r#""maxBytes":99999,"maxBlocks":1}"#,
            )
        }),
        "above the 1024 the whole heap ever held",
    );
}

/// The remaining per-row arithmetic, each shown to reject the file it exists
/// for. Every one of these was a rule nothing could trip.
#[test]
fn the_validator_rejects_each_kind_of_incoherent_row() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));

    rejects(
        &damaged_by(&text, |t| {
            replacing(
                t,
                r#""maxBytes":640,"maxBlocks":1}"#,
                r#""maxBytes":0,"maxBlocks":1}"#,
            )
        }),
        "more than its own peak",
    );
    rejects(
        &damaged_by(&text, |t| {
            replacing(
                t,
                r#""currBytes":320,"currBlocks":1,"maxBytes":640,"maxBlocks":1"#,
                r#""currBytes":320,"currBlocks":9,"maxBytes":640,"maxBlocks":1"#,
            )
        }),
        "blocks, more than its own peak",
    );
    rejects(
        &damaged_by(&text, |t| {
            replacing(t, r#""name":"main""#, r#""name":4242"#)
        }),
        "not a string",
    );
}

/// The shared row is marked by the *presence* of `overflow`, never by `false`,
/// and carries neither a name nor a first instant — it stands for many threads
/// with many of each. Three rules, none of which could fire until the
/// hand-built profile grew an overflow row.
#[test]
fn the_validator_rejects_a_malformed_overflow_row() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    native::assert_valid(&text);

    rejects(
        &damaged_by(&text, |t| {
            replacing(
                t,
                r#""id":65534,"overflow":true"#,
                r#""id":65534,"overflow":false"#,
            )
        }),
        "exists to mark the",
    );
    rejects(
        &damaged_by(&text, |t| {
            replacing(
                t,
                r#""id":65534,"overflow":true,"#,
                r#""id":65534,"overflow":true,"name":"worker","#,
            )
        }),
        "carries a name",
    );
    rejects(
        &damaged_by(&text, |t| {
            replacing(
                t,
                r#""id":65534,"overflow":true,"#,
                r#""id":65534,"overflow":true,"firstSeen":3,"#,
            )
        }),
        "one instant standing for many",
    );
}

/// A file with no `threads` or `regions` array at all is not a file with empty
/// ones: the reader would see a run nothing allocated in.
#[test]
fn the_validator_rejects_a_file_missing_an_attribution_array() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    for key in ["threads", "regions"] {
        rejects(
            &damaged_by(&text, |t| {
                replacing(t, &format!("\"{key}\":["), &format!("\"{key}Gone\":["))
            }),
            &format!("missing array `{key}`"),
        );
    }
}

// ---- the validator ----
//
// A validator that accepts everything would make every test above pass, so each
// rule is shown to reject a file that breaks it.

#[test]
fn the_validator_rejects_a_file_that_does_not_say_what_it_is() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            t.replace(r#""format":"heapscope-profile""#, r#""format":"something""#)
        }),
        "would not know what it is",
    );
    rejects(
        &damaged_by(&text, |t| {
            t.replace(r#""formatVersion":1"#, r#""formatVersion":99"#)
        }),
        "not 1",
    );
    rejects(
        &damaged_by(&text, |t| t.replace("ignore unknown fields; ", "")),
        "forward-compatibility rule",
    );
}

#[test]
fn the_validator_rejects_a_frame_index_out_of_range() {
    let text = emit(&snapshot(vec![point(&[0x1500, 0x1600], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            t.replace(r#""frames":[0,1]"#, r#""frames":[0,9]"#)
        }),
        "names frame 9",
    );
}

#[test]
fn the_validator_rejects_a_module_index_out_of_range() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| t.replace(r#""module":0"#, r#""module":7"#)),
        "names module 7",
    );
}

#[test]
fn the_validator_rejects_points_that_do_not_sum_to_the_totals() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            t.replace(
                r#""totalBytes":4096,"totalBlocks":8"#,
                r#""totalBytes":9,"totalBlocks":8"#,
            )
        }),
        "the totals say",
    );
}

/// The property the peak gate exists to make true.
#[test]
fn the_validator_rejects_at_peak_columns_that_do_not_sum_to_the_peak() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            t.replace(r#""atGmaxBytes":1024"#, r#""atGmaxBytes":7"#)
        }),
        "at the peak and the peak was",
    );
}

/// Each histogram accounts for every request, which is what makes them
/// trustworthy: a request that landed in no class would be one the reader could
/// not see missing.
#[test]
fn the_validator_rejects_a_histogram_that_loses_a_request() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            t.replace(
                r#"{"atLeast":16,"atMost":31,"blocks":8}"#,
                r#"{"atLeast":16,"atMost":31,"blocks":7}"#,
            )
        }),
        "size classes account for",
    );
    rejects(
        &damaged_by(&text, |t| {
            t.replace(r#"{"bytes":8,"blocks":8}"#, r#"{"bytes":8,"blocks":7}"#)
        }),
        "alignment classes account for",
    );
    rejects(
        &damaged_by(&text, |t| {
            t.replace(r#"{"bytes":8,"blocks":8}"#, r#"{"bytes":7,"blocks":8}"#)
        }),
        "not a power of two",
    );
}

/// The tie between the histograms and the rest of the file. A request the
/// live-block table had no room for is still a request the program made, so the
/// histograms count it and `notRecorded.blocks` explains the difference.
#[test]
fn the_validator_rejects_observed_requests_the_totals_cannot_explain() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            t.replace(r#""observedBlocks":8"#, r#""observedBlocks":9"#)
        }),
        "the totals account for",
    );
}

/// The case the rule above exists for, and the one it was first written not to
/// cover: a heap profile whose shim passed no shapes at all.
///
/// The first version guarded on `observedBlocks != 0`, so that profile — the
/// exact failure the rule's own comment claims to catch — skipped the check
/// entirely. Every other test in this file supplies shapes, so nothing noticed.
#[test]
fn the_validator_rejects_a_heap_profile_that_observed_nothing() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    let stripped = |t: &str| {
        t.replace(r#""observedBlocks":8"#, r#""observedBlocks":0"#)
            .replace(r#"{"atLeast":16,"atMost":31,"blocks":8}"#, "")
            .replace(r#"{"bytes":8,"blocks":8}"#, "")
    };
    rejects(&damaged_by(&text, stripped), "the totals account for");

    // And the mirror rule: a mode where the shim records nothing must observe
    // nothing, so a non-zero count there is equally wrong.
    let ad_hoc = emit(&as_mode(
        snapshot(vec![point(&[0x1500], 4096, 8)]),
        Mode::AdHoc,
    ));
    native::assert_valid(&ad_hoc);
    rejects(
        &damaged_by(&ad_hoc, |t| {
            t.replace(r#""observedBlocks":0"#, r#""observedBlocks":8"#)
        }),
        "where the allocator shim records nothing",
    );
}

/// Every address in the format is a hexadecimal string, not only the one in a
/// frame's `addr`. A module's load address is as capable of exceeding 2^53 as a
/// return address is, and the rule that catches it has to cover all four.
#[test]
fn the_validator_rejects_any_address_written_as_a_number() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));

    for (from, to) in [
        (r#""addr":"0x1500""#, r#""addr":5376"#),
        (r#""fileAddr":"0x1100""#, r#""fileAddr":4352"#),
        (r#""load":"0x1000""#, r#""load":4096"#),
        (r#""start":"0x1000""#, r#""start":4096"#),
        (r#""bias":"0x400""#, r#""bias":1024"#),
    ] {
        rejects(
            &damaged_by(&text, |t| t.replace(from, to)),
            "hexadecimal string",
        );
    }
}

/// The module map's whole purpose is to name the file an address resolves
/// against. Deleting the path left the suite green: the only test reading these
/// strings was a negative one, and a missing field contains no bidi override.
#[test]
fn the_validator_rejects_a_module_map_that_names_no_files() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| t.replace(r#""path":"/bin/example","#, "")),
        "which file to resolve",
    );
    rejects(
        &damaged_by(&text, |t| {
            t.replace(r#""path":"/bin/example""#, r#""path":"""#)
        }),
        "empty `path`",
    );
    rejects(
        &damaged_by(&text, |t| t.replace(r#""size":4096"#, r#""size":0"#)),
        "covers zero bytes",
    );
}

/// The positive half of the same thing: the emitter really writes the paths,
/// which is what a negative screening test cannot show.
#[test]
fn a_module_map_names_every_image_and_where_it_sits() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    let profile = parse(&text);
    let modules = profile.get("modules").and_then(Value::as_array).unwrap();

    assert_eq!(modules.len(), 2);
    assert_eq!(
        modules[0].get("path").and_then(Value::as_str),
        Some("/bin/example")
    );
    assert_eq!(
        modules[1].get("path").and_then(Value::as_str),
        Some("/lib/libc.so")
    );
    assert_eq!(
        modules[0].get("load").and_then(Value::as_str),
        Some("0x1000")
    );
    assert_eq!(
        modules[0].get("bias").and_then(Value::as_str),
        Some("0x400")
    );
    assert_eq!(modules[0].get("size").and_then(Value::as_u64), Some(0x1000));
    assert_eq!(
        modules[0].get("buildId").and_then(Value::as_str),
        Some("0badc0ffee")
    );
    assert!(
        modules[1].get("buildId").is_none(),
        "a build identity nobody supplied must be absent rather than empty"
    );
}

#[test]
fn the_validator_rejects_a_capture_cost_that_reads_as_free() {
    let mut snapshot = snapshot(vec![point(&[0x1500], 4096, 8)]);
    snapshot.metrics.capture_cost = heapscope::unwind::Cost {
        nanos: 1_344,
        captures: 64,
        frames: 11,
        strategy: heapscope::unwind::Strategy::FramePointer,
    };
    let text = emit(&snapshot);
    native::assert_valid(&text);

    rejects(
        &damaged_by(&text, |t| t.replace(r#""nanos":1344"#, r#""nanos":0"#)),
        "reads as a free capture",
    );
    rejects(
        &damaged_by(&text, |t| t.replace(r#","unwinder":"frame-pointer"}"#, "}")),
        "does not say which unwinder",
    );
}

/// Every field of every set, rather than one representative. A negative test
/// that injects one field and generalises leaves the other rules never fired,
/// which is how seven of eight new DHAT rules once shipped dead.
#[test]
fn the_validator_rejects_every_lifetime_field_a_non_heap_profile_cannot_have() {
    let text = emit(&as_mode(
        snapshot(vec![point(&[0x1500], 4096, 8)]),
        Mode::AdHoc,
    ));
    native::assert_valid(&text);

    for field in [
        "retiredLifetime",
        "unretiredLifetime",
        "maxBytes",
        "maxBlocks",
        "atGmaxBytes",
        "atGmaxBlocks",
        "atEndBytes",
        "atEndBlocks",
    ] {
        rejects(
            &damaged_by(&text, |t| {
                t.replace(
                    r#""totalBlocks":8,"#,
                    &format!(r#""totalBlocks":8,"{field}":0,"#),
                )
            }),
            field,
        );
    }
    rejects(
        &damaged_by(&text, |t| {
            t.replace(r#""timeAtEnd":100"#, r#""timeAtEnd":100,"timeAtMax":0"#)
        }),
        "which has no peak",
    );

    // Every global field that describes a live block, not only the two peaks.
    // `currBytes`, `currBlocks` and `peaks` were written unconditionally while
    // the per-point `atEndBytes` beside them was omitted, so an ad hoc profile
    // reported "0 bytes live at the end" — the exact non-measurement the rule
    // one level down exists to keep out.
    //
    // Anchored on the totals object's own layout, which is wrapped where a
    // point is inline, so the injection cannot land in a point instead and
    // pass for the wrong reason.
    for field in ["currBytes", "currBlocks", "maxBytes", "maxBlocks", "peaks"] {
        rejects(
            &damaged_by(&text, |t| {
                t.replace(
                    "\"totalBlocks\":8\n }",
                    &format!("\"totalBlocks\":8,\n  \"{field}\":0\n }}"),
                )
            }),
            "must be omitted, not zeroed",
        );
    }
}

#[test]
fn the_validator_rejects_a_mode_it_does_not_know() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            t.replace(r#""mode":"heap""#, r#""mode":"invented""#)
        }),
        "not one of heap, ad-hoc, copy",
    );
}

#[test]
fn the_validator_rejects_a_size_class_written_out_with_no_blocks() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            t.replace(
                r#"{"atLeast":16,"atMost":31,"blocks":8}"#,
                r#"{"atLeast":16,"atMost":31,"blocks":8},
   {"atLeast":32,"atMost":63,"blocks":0}"#,
            )
        }),
        "empty classes are left out",
    );
}

#[test]
fn the_validator_rejects_a_table_holding_more_than_it_can() {
    let text = emit(&snapshot(vec![point(&[0x1500], 4096, 8)]));
    rejects(
        &damaged_by(&text, |t| {
            t.replace(
                r#""entries":2,"capacity":1024"#,
                r#""entries":2000,"capacity":1024"#,
            )
        }),
        "against a capacity of",
    );
}

#[test]
fn the_validator_rejects_a_reallocation_that_copied_without_moving() {
    let shapes = Shapes::new();
    shapes.record(Shape::of(24).aligned(8));
    let mut snapshot = snapshot(vec![point(&[0x1500], 4096, 1)]);
    snapshot.stats.total_blocks = 1;
    snapshot.points[0].counters.total_blocks = 1;
    snapshot.shapes = shapes.snapshot();
    let text = emit(&snapshot);
    native::assert_valid(&text);

    rejects(
        &damaged_by(&text, |t| {
            t.replace(
                r#""moved":0,"bytesCopied":0"#,
                r#""moved":0,"bytesCopied":99"#,
            )
        }),
        "a resize in place copies nothing",
    );
}

// ---- the DHAT file is a projection of this one ----

/// Every number the DHAT file carries about the run as a whole is in the native
/// file, and equal.
///
/// The list is written out rather than derived, because deriving it from
/// whichever keys the DHAT emitter happens to write would make this test agree
/// with the emitter by construction and prove nothing.
#[test]
fn the_dhat_file_carries_no_number_the_native_file_lacks() {
    let snapshot = snapshot(vec![
        point(&[0x1500, 0x1600], 4096, 8),
        point(&[0x4500], 512, 1),
    ]);

    let mut buffer = Vec::new();
    snapshot
        .write_dhat_v2(&mut buffer)
        .expect("writing to a Vec cannot fail");
    let dhat = parse(&String::from_utf8(buffer).expect("valid UTF-8"));
    let native = parse(&emit(&snapshot));

    let at = |value: &Value, path: &str| -> u64 {
        let mut current = value.clone();
        for step in path.split('.') {
            current = current
                .get(step)
                .unwrap_or_else(|| panic!("{path} is missing at {step}"))
                .clone();
        }
        current
            .as_u64()
            .unwrap_or_else(|| panic!("{path} is not an integer"))
    };

    // Left is the DHAT path, right is the native one.
    for (dhat_path, native_path) in [
        ("pid", "run.pid"),
        ("te", "run.timeAtEnd"),
        ("tg", "run.timeAtMax"),
        ("heapscope.totals.totalBytes", "totals.totalBytes"),
        ("heapscope.totals.totalBlocks", "totals.totalBlocks"),
        ("heapscope.totals.maxBytes", "totals.maxBytes"),
        ("heapscope.totals.maxBlocks", "totals.maxBlocks"),
        ("heapscope.totals.currBytes", "totals.currBytes"),
        ("heapscope.totals.currBlocks", "totals.currBlocks"),
        ("heapscope.droppedBlocks", "notRecorded.blocks"),
        ("heapscope.droppedPoints", "notRecorded.programPoints"),
        (
            "heapscope.unattributedBlocks",
            "notRecorded.unattributedBlocks",
        ),
        ("heapscope.refusedEvents", "notRecorded.refusedEvents"),
        ("heapscope.settings.maxDepth", "settings.maxDepth"),
        ("heapscope.settings.maxLiveBlocks", "settings.maxLiveBlocks"),
    ] {
        assert_eq!(
            at(&dhat, dhat_path),
            at(&native, native_path),
            "`{dhat_path}` in the DHAT file and `{native_path}` in the native \
             file describe the same measurement and disagree"
        );
    }

    for (dhat_path, native_path) in [
        ("mode", "run.mode"),
        ("cmd", "run.command"),
        ("heapscope.shutdown", "run.shutdown"),
        ("heapscope.unwinder", "run.unwinder"),
        ("tu", "run.timeSource"),
    ] {
        let text = |value: &Value, path: &str| -> String {
            let mut current = value.clone();
            for step in path.split('.') {
                current = current.get(step).expect("the field exists").clone();
            }
            current.as_str().expect("a string").to_string()
        };
        assert_eq!(text(&dhat, dhat_path), text(&native, native_path));
    }

    // The module map, which both files carry and nothing compared. It is the one
    // place where a regression would land in the DHAT file alone: if `bias` went
    // back to being an offset from `start` — the M2 bug — only that map would be
    // wrong, every assertion above would still pass, and offline symbolization
    // would silently name the wrong function.
    //
    // The two write it differently on purpose. A DHAT extension is JSON among
    // numbers, and the native format writes addresses as strings because
    // `JSON.parse` rounds above 2^53; comparing them means reading through the
    // representation, which is exactly what a reader of both files has to do.
    let dhat_modules = dhat
        .get("heapscope")
        .and_then(|section| section.get("modules"))
        .and_then(Value::as_array)
        .expect("the DHAT extension carries a module map");
    let native_modules = native
        .get("modules")
        .and_then(Value::as_array)
        .expect("the native file carries a module map");
    assert_eq!(
        dhat_modules.len(),
        native_modules.len(),
        "the two files disagree about how many images were mapped"
    );

    for (dhat_module, native_module) in dhat_modules.iter().zip(native_modules) {
        assert_eq!(
            dhat_module.get("path").and_then(Value::as_str),
            native_module.get("path").and_then(Value::as_str),
        );
        assert_eq!(
            dhat_module.get("buildId").and_then(Value::as_str),
            native_module.get("buildId").and_then(Value::as_str),
        );
        assert_eq!(
            dhat_module.get("size").and_then(Value::as_u64),
            native_module.get("size").and_then(Value::as_u64),
        );
        for field in ["load", "start", "bias"] {
            let from_dhat = dhat_module
                .get(field)
                .and_then(Value::as_u64)
                .unwrap_or_else(|| panic!("the DHAT map's `{field}` is an integer"));
            let from_native = native_module
                .get(field)
                .and_then(Value::as_str)
                .and_then(|text| u64::from_str_radix(text.trim_start_matches("0x"), 16).ok())
                .unwrap_or_else(|| panic!("the native map's `{field}` is a hex string"));
            assert_eq!(
                from_dhat, from_native,
                "the two files disagree about a module's `{field}`"
            );
        }
    }
}

/// Per program point, the native file's numbers are what the DHAT file's are
/// built from — with the one documented difference that DHAT adds the two
/// lifetimes together because it has one field for them.
#[test]
fn every_dhat_program_point_is_a_native_one_with_its_lifetimes_added_up() {
    // One point per frame list, so nothing folds and the two files' points
    // correspond one to one. The DHAT emitter sorts by total bytes, so the
    // comparison is by frame list rather than by position.
    let snapshot = snapshot(vec![
        point(&[0x1500, 0x1600], 4096, 8),
        point(&[0x4500], 512, 1),
    ]);

    let mut buffer = Vec::new();
    snapshot
        .write_dhat_v2_with(&mut buffer, &heapscope::output::RawAddresses)
        .expect("writing to a Vec cannot fail");
    let dhat = parse(&String::from_utf8(buffer).expect("valid UTF-8"));
    let native = parse(&emit(&snapshot));

    let ftbl: Vec<&str> = dhat
        .get("ftbl")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|frame| frame.as_str().unwrap())
        .collect();
    let frames: Vec<&str> = native
        .get("frames")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|frame| frame.get("addr").and_then(Value::as_str).unwrap())
        .collect();

    let dhat_points = dhat.get("pps").and_then(Value::as_array).unwrap();
    let native_points = native.get("points").and_then(Value::as_array).unwrap();
    assert_eq!(dhat_points.len(), native_points.len());

    for dhat_point in dhat_points {
        // `0x1500: ???` in the DHAT rendering; `0x1500` in the native table.
        let stack: Vec<&str> = dhat_point
            .get("fs")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|index| {
                ftbl[index.as_u64().unwrap() as usize]
                    .split(':')
                    .next()
                    .unwrap()
            })
            .collect();
        let native_point = native_points
            .iter()
            .find(|point| {
                let addresses: Vec<&str> = point
                    .get("frames")
                    .and_then(Value::as_array)
                    .unwrap()
                    .iter()
                    .map(|index| frames[index.as_u64().unwrap() as usize])
                    .collect();
                addresses == stack
            })
            .unwrap_or_else(|| panic!("no native point has the frames {stack:?}"));

        let dhat_at = |field: &str| dhat_point.get(field).and_then(Value::as_u64).unwrap();
        let native_at = |field: &str| native_point.get(field).and_then(Value::as_u64).unwrap();

        assert_eq!(dhat_at("tb"), native_at("totalBytes"));
        assert_eq!(dhat_at("tbk"), native_at("totalBlocks"));
        assert_eq!(dhat_at("mb"), native_at("maxBytes"));
        assert_eq!(dhat_at("mbk"), native_at("maxBlocks"));
        assert_eq!(dhat_at("gb"), native_at("atGmaxBytes"));
        assert_eq!(dhat_at("gbk"), native_at("atGmaxBlocks"));
        assert_eq!(dhat_at("eb"), native_at("atEndBytes"));
        assert_eq!(dhat_at("ebk"), native_at("atEndBlocks"));
        assert_eq!(
            dhat_at("tl"),
            native_at("retiredLifetime") + native_at("unretiredLifetime"),
            "DHAT's `tl` is the sum of the two lifetimes the native file keeps \
             apart"
        );
    }
}

/// The other direction of the projection claim. The native file keeps frames
/// the DHAT file's rendering may fold together, so a snapshot whose points
/// collapse under a rendering has fewer points in the DHAT file and all of them
/// here.
#[test]
fn a_fold_that_loses_points_in_the_dhat_file_loses_none_here() {
    // Same frame list, distinct points: what the engine produces when two call
    // sites are distinct by address and identical after trimming.
    let snapshot = snapshot(vec![point(&[0x1500], 4096, 8), point(&[0x1500], 512, 1)]);

    let mut buffer = Vec::new();
    snapshot
        .write_dhat_v2(&mut buffer)
        .expect("writing to a Vec cannot fail");
    let dhat = parse(&String::from_utf8(buffer).expect("valid UTF-8"));
    let native = parse(&emit(&snapshot));

    assert_eq!(
        dhat.get("pps").and_then(Value::as_array).unwrap().len(),
        1,
        "the DHAT emitter must fold, because the viewer refuses a repeated \
         location"
    );
    assert_eq!(
        dhat.get("heapscope")
            .and_then(|extension| extension.get("foldedPoints"))
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        native
            .get("points")
            .and_then(Value::as_array)
            .unwrap()
            .len(),
        2,
        "the native file folded, and folding is not a fact about the run"
    );
}

// ---- the property ----

/// The largest byte count a generated point may carry.
///
/// `u64::MAX / 16` with at most eight points, so the totals cannot overflow the
/// helper that sums them and every generated profile is internally consistent —
/// a generated profile that failed the validator's arithmetic would be testing
/// the generator. The genuinely saturated case is
/// [`a_profile_of_saturated_counters_is_valid`] instead, where one point holds
/// `u64::MAX` and the sum is still exact.
const GENERATED_MAX_BYTES: u64 = u64::MAX / 16;

/// One point at `u64::MAX`, which is what a long-running process's summed
/// lifetime legitimately reaches.
#[test]
fn a_profile_of_saturated_counters_is_valid() {
    let mut saturated = point(&[0x1500], u64::MAX, u64::MAX);
    saturated.counters.total_lifetime = u64::MAX;
    saturated.unretired_lifetime = u64::MAX;
    saturated.counters.max_bytes = u64::MAX;
    saturated.counters.curr_bytes = u64::MAX;
    saturated.counters.at_gmax_bytes = u64::MAX;

    let mut snapshot = snapshot(vec![saturated]);
    // One point, so the totals *are* the point's numbers rather than a sum that
    // wrapped on the way here.
    snapshot.stats.total_bytes = u64::MAX;
    snapshot.stats.total_blocks = u64::MAX;
    snapshot.stats.curr_bytes = u64::MAX;
    snapshot.stats.max_bytes = u64::MAX;
    snapshot.shapes = Default::default();

    let text = emit(&snapshot);
    native::assert_valid(&text);
    assert!(
        text.contains("18446744073709551615"),
        "a saturated counter did not survive into the file: {text}"
    );
}

proptest! {
    // Proptest saves failing seeds to a file next to the test, which means
    // resolving the current directory -- and Miri's filesystem isolation makes
    // that a hard abort, which takes the whole test binary and every other test
    // in this file with it. Persistence is worth keeping natively, where it
    // turns a rare failure into a permanently reproducible one, so it is
    // dropped only under Miri. Same shape as `tests/dhat_output.rs`.
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) { 4 } else { ProptestConfig::default().cases },
        failure_persistence: if cfg!(miri) {
            None
        } else {
            Some(Box::new(
                proptest::test_runner::FileFailurePersistence::default(),
            ))
        },
        ..ProptestConfig::default()
    })]

    /// Whatever a snapshot holds, the file written from it is valid.
    ///
    /// The generators reach the awkward shapes on purpose: points with no
    /// frames, addresses across the whole `usize` range — including above 2^53,
    /// which is why addresses are strings — and every mode.
    #[test]
    fn every_emitted_profile_is_valid(
        stacks in prop::collection::vec(
            prop::collection::vec(prop::num::u64::ANY.prop_map(|a| a as usize), 0..6),
            0..8,
        ),
        bytes in prop::collection::vec(0u64..GENERATED_MAX_BYTES, 0..8),
        blocks in prop::collection::vec(1u64..1_000, 0..8),
        mode in prop_oneof![Just(Mode::Heap), Just(Mode::AdHoc), Just(Mode::Copy)],
    ) {
        let points: Vec<ProgramPoint> = stacks
            .iter()
            .enumerate()
            .map(|(at, frames)| {
                let total = bytes.get(at).copied().unwrap_or(1024);
                let count = blocks.get(at).copied().unwrap_or(1);
                point(frames, total, count)
            })
            .collect();

        let snapshot = as_mode(snapshot(points), mode);
        let profile = emit(&snapshot);
        let problems = native::problems(&profile);
        prop_assert!(problems.is_empty(), "{problems:?}\n{profile}");
    }
}
