//! The DHAT v2 emitter, checked against a validator stricter than the viewer.
//!
//! These tests never touch the engine. Every profile here is built by hand,
//! which is the only way to reach the cases that matter: a program point whose
//! capture found no frames, two points that render to the same frame list, a
//! command line containing a newline, counters at `u64::MAX`. A recorded
//! workload produces none of those on demand.
//!
//! Half the file tests the validator rather than the emitter. A validator that
//! accepts everything would make every other test in this file pass, so each
//! rule it enforces is proved to reject a file that breaks it.

mod support;

use std::collections::BTreeMap;

use heapscope::output::{Counters, FrameFormat, PointKind, ProgramPoint, Snapshot};
use heapscope::symbol::modules::Module;
use heapscope::{Mode, TimeSource};
use proptest::prelude::*;
use support::dhat;
use support::display;
use support::json::{self, Value};
use support::snapshot::{as_mode, hand_built, point};

/// A frame rendering that maps every address to the same string, as
/// symbolization and frame trimming both do to unrelated call sites.
struct OneName;
impl FrameFormat for OneName {
    fn format(&self, _address: usize, out: &mut String) {
        out.push_str("everything");
    }
}

/// A frame rendering whose *name* is hostile, which is what a `FrameFormat`
/// over a damaged symbol table produces.
///
/// Distinct from a hostile module path on purpose. The emitter screens the
/// finished frame, and a test that only ever puts awkward characters in the
/// path proves that the *path* is screened somewhere — which it is, twenty
/// lines later, by its own call. Only a name can show that the frame call site
/// is doing anything.
struct HostileName;
impl FrameFormat for HostileName {
    fn format(&self, address: usize, out: &mut String) {
        out.push_str(&format!(
            "0x{address:x}: \u{1b}[2Jcleared\u{202e}gnp.eslaf\u{0}"
        ));
    }
}

/// A hand-built profile, plus the module map the frame renderer resolves
/// against — which is what this suite is about.
fn snapshot(points: Vec<ProgramPoint>) -> Snapshot {
    let mut snapshot = hand_built(points);
    snapshot.modules = vec![
        Module {
            path: String::from("/bin/example"),
            start: 0x1000,
            size: 0x1000,
            // A non-zero bias, so that a test asserting on a rendered frame
            // would notice the emitter using the wrong number.
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

fn emit(snapshot: &Snapshot) -> String {
    let mut buffer = Vec::new();
    snapshot
        .write_dhat_v2(&mut buffer)
        .expect("writing to a Vec cannot fail");
    String::from_utf8(buffer).expect("the writer must produce UTF-8")
}

fn emit_with(snapshot: &Snapshot, format: &dyn FrameFormat) -> String {
    let mut buffer = Vec::new();
    snapshot
        .write_dhat_v2_with(&mut buffer, format)
        .expect("writing to a Vec cannot fail");
    String::from_utf8(buffer).expect("the writer must produce UTF-8")
}

#[test]
fn a_recorded_program_point_produces_a_valid_profile() {
    let profile = emit(&snapshot(vec![
        point(&[0x1000, 0x2000, 0x3000], 4096, 8),
        point(&[0x1000, 0x4000], 512, 2),
    ]));
    dhat::assert_valid(&profile);
}

#[test]
fn a_profile_with_no_allocations_is_still_valid() {
    let profile = emit(&snapshot(Vec::new()));
    dhat::assert_valid(&profile);
    let parsed = json::parse(&profile).expect("valid JSON");
    assert_eq!(parsed.get("pps").unwrap().as_array().unwrap().len(), 0);
    assert_eq!(parsed.get("ftbl").unwrap().as_array().unwrap().len(), 1);
}

/// The M2 exit criterion. Without the emit-time fold this file would make the
/// viewer throw `data file contains a repeated location` and refuse to open.
#[test]
fn program_points_that_render_alike_are_folded_rather_than_repeated() {
    let profile = emit_with(
        &snapshot(vec![
            point(&[0x1000, 0x2000], 4096, 8),
            point(&[0x3000, 0x4000], 2048, 4),
            point(&[0x5000], 1024, 2),
        ]),
        &OneName,
    );
    dhat::assert_valid(&profile);

    let parsed = json::parse(&profile).expect("valid JSON");
    let points = parsed.get("pps").unwrap().as_array().unwrap();
    assert_eq!(points.len(), 2, "one per distinct *rendered* frame list");
    let bytes: u64 = points
        .iter()
        .map(|p| p.get("tb").unwrap().as_u64().unwrap())
        .sum();
    assert_eq!(bytes, 7168, "folding must not lose bytes");
}

/// A stack in the shape a real one has, so that trimming can be driven through
/// the whole emitter without depending on what this platform's symbol table
/// happens to hold. Every name is verbatim from a profile of
/// `examples/profile_a_program` on macOS aarch64, generics shortened.
struct RealNames;

/// Innermost first: the allocation path, the program, then the runtime entry.
const REAL_STACK: &[(usize, &str)] = &[
    (0x1000, "__rustc::__rust_alloc+0x38"),
    (
        0x2000,
        "<alloc::raw_vec::RawVecInner>::try_allocate_in+0x9c",
    ),
    (0x3000, "<alloc::vec::Vec<u8>>::with_capacity+0x24"),
    (0x4000, "profile_a_program::churn+0x90"),
    (0x5000, "profile_a_program::main+0x174"),
    (
        0x6000,
        "std::sys::backtrace::__rust_begin_short_backtrace::<f, r>+0x18",
    ),
    (0x7000, "std::rt::lang_start_internal+0x3b8"),
    (0x8000, "main+0x24"),
    // Not on the stack above: a second call site, for the fold.
    (0x9000, "profile_a_program::grow_by_pushing+0xa8"),
];

impl FrameFormat for RealNames {
    fn format(&self, address: usize, out: &mut String) {
        let name = REAL_STACK
            .iter()
            .find_map(|&(at, name)| (at == address).then_some(name))
            .unwrap_or("???");
        out.push_str(&format!(
            "0x{address:x}: {name} (/bin/program+0x{address:x})"
        ));
    }
}

fn addresses(count: usize) -> Vec<usize> {
    REAL_STACK.iter().take(count).map(|&(at, _)| at).collect()
}

/// The whole of M4's frame trimming, through the real emitter and the strict
/// validator: what a reader is shown is the program, and the file is still one
/// the viewer will open.
#[test]
fn a_trimmed_profile_shows_the_program_and_not_the_machinery_around_it() {
    use heapscope::symbol::Trimmed;

    let taken = snapshot(vec![point(&addresses(8), 4096, 8)]);
    let profile = emit_with(&taken, &Trimmed::new(RealNames));
    dhat::assert_valid(&profile);

    let parsed = json::parse(&profile).expect("valid JSON");
    let table: Vec<&str> = parsed
        .get("ftbl")
        .and_then(Value::as_array)
        .expect("a frame table")
        .iter()
        .skip(1)
        .filter_map(Value::as_str)
        .collect();

    assert_eq!(
        table,
        [
            "0x3000: <alloc::vec::Vec<u8>>::with_capacity+0x24 (/bin/program+0x3000)",
            "0x4000: profile_a_program::churn+0x90 (/bin/program+0x4000)",
            "0x5000: profile_a_program::main+0x174 (/bin/program+0x5000)",
        ],
        "the emitted frames are not the three that say where the allocation \
         came from"
    );
    assert_eq!(
        parsed
            .get("heapscope")
            .and_then(|section| section.get("trimmedFrames"))
            .and_then(Value::as_u64),
        Some(5),
        "five of eight frames were removed and the file must say so: {profile}"
    );

    // The same snapshot, rendered by the same names without trimming, keeps
    // every one of them. Without this the test above would pass just as well
    // against a capture that never recorded the runtime frames at all.
    let whole = json::parse(&emit_with(&taken, &RealNames)).expect("valid JSON");
    let untrimmed = whole
        .get("ftbl")
        .and_then(Value::as_array)
        .expect("a frame table")
        .len();
    assert_eq!(untrimmed, 9, "the root and eight frames");
}

/// PLAN.md section 3.2, end to end: trimming is what makes two distinct call
/// sites indistinguishable, and a repeated `fs` is a file `dh_view.html`
/// refuses with `data file contains a repeated location`.
#[test]
fn call_sites_that_differ_only_where_trimming_cuts_still_produce_an_openable_file() {
    use heapscope::symbol::Trimmed;

    // Two stacks that share everything a reader is shown and differ only in the
    // allocation path above it: one allocated through `RawVec`, the other
    // directly.
    let through_raw_vec = vec![0x1000, 0x2000, 0x3000, 0x4000];
    let direct = vec![0x1000, 0x3000, 0x4000];

    let taken = snapshot(vec![
        point(&through_raw_vec, 4096, 8),
        point(&direct, 2048, 4),
    ]);
    let profile = emit_with(&taken, &Trimmed::new(RealNames));
    dhat::assert_valid(&profile);

    let parsed = json::parse(&profile).expect("valid JSON");
    let points = parsed.get("pps").and_then(Value::as_array).expect("points");
    assert_eq!(points.len(), 1, "two stacks, one surviving frame list");
    assert_eq!(
        points[0].get("tb").and_then(Value::as_u64),
        Some(6144),
        "folding must not lose bytes"
    );
    assert_eq!(
        parsed
            .get("heapscope")
            .and_then(|section| section.get("foldedPoints"))
            .and_then(Value::as_u64),
        Some(1),
        "the file must say that two call sites became one"
    );
}

/// Trimming reads the *name* of a frame, so a build with no names has nothing
/// to read. It must then leave the stack exactly as it was rather than guess.
#[test]
fn a_profile_with_no_names_in_it_is_not_trimmed_at_all() {
    use heapscope::symbol::Trimmed;

    let taken = snapshot(vec![point(&[0x11, 0x22, 0x33, 0x44], 4096, 8)]);
    let bare = emit_with(&taken, &heapscope::output::RawAddresses);
    let trimmed = emit_with(&taken, &Trimmed::new(heapscope::output::RawAddresses));
    assert_eq!(
        bare, trimmed,
        "a stripped profile must be byte for byte what it was"
    );
}

#[test]
fn a_program_point_with_no_frames_survives_and_is_visible() {
    let profile = emit(&snapshot(vec![point(&[], 64, 1)]));
    dhat::assert_valid(&profile);
    let parsed = json::parse(&profile).expect("valid JSON");
    let points = parsed.get("pps").unwrap().as_array().unwrap();
    assert_eq!(points.len(), 1);

    // The bytes have to go somewhere, and an empty `fs` puts them in a row the
    // viewer draws as blank. The frame table names the reason instead.
    let frames = points[0].get("fs").unwrap().as_array().unwrap();
    assert_eq!(frames.len(), 1);
    let table = parsed.get("ftbl").unwrap().as_array().unwrap();
    let label = table[frames[0].as_u64().unwrap() as usize]
        .as_str()
        .unwrap();
    assert!(label.contains("unwalkable"), "{label:?}");
}

#[test]
fn the_overflow_point_names_itself_in_the_frame_table() {
    let mut overflow = point(&[], 64, 1);
    overflow.kind = PointKind::Overflow;
    let profile = emit(&snapshot(vec![overflow]));
    dhat::assert_valid(&profile);

    let parsed = json::parse(&profile).expect("valid JSON");
    let points = parsed.get("pps").unwrap().as_array().unwrap();
    let frames = points[0].get("fs").unwrap().as_array().unwrap();
    let table = parsed.get("ftbl").unwrap().as_array().unwrap();
    let label = table[frames[0].as_u64().unwrap() as usize]
        .as_str()
        .unwrap();
    assert!(
        label.contains("overflow") && label.contains("program-point table"),
        "a reader seeing this row needs to be told to raise the ceiling: {label:?}"
    );
}

#[test]
fn awkward_text_survives_the_round_trip() {
    // A command line is whatever the user typed, and a path is whatever the
    // filesystem allows. Both reach the file, and the two ways they can be
    // damaged on the way are different enough to be worth separating.
    //
    // Quoting, backslashes, and characters outside the basic multilingual plane
    // are the JSON writer's problem, and they come back byte for byte.
    // Control characters and the bidirectional formatting characters are the
    // display screen's problem, and they come back as visible escapes, because
    // a profile is read in a terminal by someone who did not write the program
    // that produced it. See `output::push_display`.
    let awkward = "prog \"quoted\" \\back\\slash\n\ttab \u{2028}sep\u{2029} µs 😀 \u{1}";
    let mut snapshot = snapshot(vec![point(&[0x1000], 64, 1)]);
    snapshot.command = String::from(awkward);
    let profile = emit(&snapshot);
    dhat::assert_valid(&profile);

    let parsed = json::parse(&profile).expect("valid JSON");
    assert_eq!(
        parsed.get("cmd").unwrap().as_str().unwrap(),
        "prog \"quoted\" \\back\\slash\\u{a}\\u{9}tab \\u{2028}sep\\u{2029} µs 😀 \\u{1}"
    );
    // Which is what the rule says it should be, stated independently.
    assert_eq!(
        parsed.get("cmd").unwrap().as_str().unwrap(),
        display::screen(awkward)
    );
}

/// The screen has to be complete, not merely present.
///
/// The failure it exists to prevent is a terminal acting on a byte from a
/// symbol table, so the check is on the finished artifact: no character that can
/// carry an instruction may appear anywhere in it, whatever field it came
/// through.
#[test]
fn nothing_in_a_finished_profile_can_command_a_terminal() {
    let mut snapshot = snapshot(vec![point(&[0x1234], 64, 1)]);
    snapshot.command = String::from("prog \u{1b}[2J --flag \u{202e}gnp.exe");
    snapshot.modules = vec![Module {
        path: String::from("/tmp/an\u{1b}[31m image\u{202e}os.\u{0}"),
        start: 0x1000,
        size: 0x1000,
        bias: 0x400,
        image_base: 0x1000,
        build_id: None,
    }];

    let profile = emit(&snapshot);
    dhat::assert_valid(&profile);

    let offenders = display::offenders(&profile);
    assert!(
        offenders.is_empty(),
        "the profile carries {offenders:?}, which a terminal or a bidirectional \
         renderer will act on"
    );

    // And the escaped forms are there, so this passed by escaping rather than
    // by dropping the fields.
    assert!(profile.contains("\\\\u{1b}"), "{profile}");
    assert!(profile.contains("\\\\u{202e}"), "{profile}");
}

/// The same, for a frame *name*, through a `FrameFormat` this crate did not
/// write.
///
/// The reason the screen is applied to the finished frame rather than inside
/// `ModuleOffsets` and `Symbolized` is that it should hold for any renderer. If
/// it only held for the two shipped here, a user's own `FrameFormat` — or a
/// future one of ours — would quietly lose the guarantee.
///
/// `HostileName` ignores the module map entirely and the default map's paths
/// are innocuous, so the name is the only thing that can be responsible for
/// what comes out. (The map cannot simply be emptied: the validator rejects
/// that, correctly — every process has at least the image it is running.)
#[test]
fn a_frame_name_from_any_renderer_is_screened() {
    let snapshot = snapshot(vec![point(&[0x1234], 64, 1)]);

    let profile = emit_with(&snapshot, &HostileName);
    dhat::assert_valid(&profile);

    let offenders = display::offenders(&profile);
    assert!(
        offenders.is_empty(),
        "a frame name put {offenders:?} into the profile"
    );
    assert!(
        profile.contains("\\\\u{1b}[2Jcleared\\\\u{202e}"),
        "{profile}"
    );
}

/// `bklt` false means *omitted*, not zeroed. `dh_main.c` documents each of these
/// fields that way, and the distinction is the whole reason a non-heap mode is a
/// mode rather than a filter: an event was never live, so a zero would be a
/// measurement of something that did not happen.
#[test]
fn a_non_heap_profile_omits_every_field_it_has_no_measurement_for() {
    for (mode, name, verb) in [
        (Mode::AdHoc, "ad-hoc", "Occurred"),
        (Mode::Copy, "copy", "Copied"),
    ] {
        let profile = emit(&as_mode(
            snapshot(vec![point(&[0x1000, 0x2000], 4_096, 8)]),
            mode,
        ));
        let parsed = json::parse(&profile).expect("valid JSON");

        assert_eq!(parsed.get("mode").and_then(json::Value::as_str), Some(name));
        assert_eq!(parsed.get("verb").and_then(json::Value::as_str), Some(verb));
        assert_eq!(
            parsed.get("bklt").and_then(json::Value::as_bool),
            Some(false)
        );
        for field in ["tg", "tuth"] {
            assert!(
                parsed.get(field).is_none(),
                "`{field}` survived into a {name} profile:\n{profile}"
            );
        }

        let points = parsed.get("pps").and_then(json::Value::as_array).unwrap();
        let recorded = points.first().expect("the point was emitted");
        for field in ["tb", "tbk", "fs"] {
            assert!(
                recorded.get(field).is_some(),
                "`{field}` is mandatory in every mode and is missing:\n{profile}"
            );
        }
        for field in ["tl", "mb", "mbk", "gb", "gbk", "eb", "ebk"] {
            assert!(
                recorded.get(field).is_none(),
                "`{field}` survived into a {name} program point:\n{profile}"
            );
        }

        dhat::assert_valid(&profile);
    }
}

/// Ad hoc weights are dimensionless and copied bytes are not, so only one of the
/// two renames the viewer's units. Getting this wrong renders 5,000 retries as
/// five kilobytes, in the one place a reader cannot check it.
#[test]
fn only_an_ad_hoc_profile_renames_the_units_it_counts() {
    let ad_hoc = emit(&as_mode(
        snapshot(vec![point(&[0x1000], 5_000, 4)]),
        Mode::AdHoc,
    ));
    let parsed = json::parse(&ad_hoc).expect("valid JSON");
    assert_eq!(parsed.get("bu").and_then(json::Value::as_str), Some("unit"));
    assert_eq!(
        parsed.get("bsu").and_then(json::Value::as_str),
        Some("units")
    );
    assert_eq!(
        parsed.get("bksu").and_then(json::Value::as_str),
        Some("events")
    );

    // Omitted rather than restated, which is what the viewer reads as "these
    // are bytes and blocks".
    for mode in [Mode::Heap, Mode::Copy] {
        let profile = emit(&as_mode(snapshot(vec![point(&[0x1000], 5_000, 4)]), mode));
        let parsed = json::parse(&profile).expect("valid JSON");
        for field in ["bu", "bsu", "bksu"] {
            assert!(
                parsed.get(field).is_none(),
                "a {} profile renamed `{field}`, which it counts in bytes:\n{profile}",
                mode.as_str()
            );
        }
    }
}

#[test]
fn the_time_unit_labels_match_the_time_source() {
    let mut snapshot = snapshot(vec![point(&[0x1000], 64, 1)]);
    let parsed = json::parse(&emit(&snapshot)).expect("valid JSON");
    assert_eq!(parsed.get("tu").unwrap().as_str().unwrap(), "events");
    assert_eq!(parsed.get("Mtu").unwrap().as_str().unwrap(), "Mevent");

    snapshot.time_source = TimeSource::Monotonic;
    let parsed = json::parse(&emit(&snapshot)).expect("valid JSON");
    assert_eq!(parsed.get("tu").unwrap().as_str().unwrap(), "µs");
    assert_eq!(parsed.get("Mtu").unwrap().as_str().unwrap(), "Mµs");
}

#[test]
fn a_degraded_snapshot_says_so_in_the_file() {
    let mut snapshot = snapshot(vec![point(&[0x1000], 64, 1)]);
    snapshot.exact = false;
    snapshot.poisoned = true;
    snapshot.points_dropped = 5;
    snapshot.unattributed_blocks = 9;
    snapshot.stats.dropped_blocks = 11;

    let parsed = json::parse(&emit(&snapshot)).expect("valid JSON");
    let extension = parsed.get("heapscope").expect("the heapscope section");
    assert_eq!(extension.get("exact").unwrap().as_bool(), Some(false));
    assert_eq!(extension.get("poisoned").unwrap().as_bool(), Some(true));
    assert_eq!(extension.get("droppedPoints").unwrap().as_u64(), Some(5));
    assert_eq!(
        extension.get("unattributedBlocks").unwrap().as_u64(),
        Some(9)
    );
    assert_eq!(extension.get("droppedBlocks").unwrap().as_u64(), Some(11));
}

/// Without the module map an address is uninterpretable the moment the process
/// exits, so the frames and the map have to arrive together.
#[test]
fn the_module_map_reaches_the_file_and_the_frames_point_into_it() {
    let profile = emit(&snapshot(vec![point(&[0x1234], 64, 1)]));
    dhat::assert_valid(&profile);
    let parsed = json::parse(&profile).expect("valid JSON");

    let modules = parsed
        .get("heapscope")
        .and_then(|section| section.get("modules"))
        .and_then(|modules| modules.as_array())
        .expect("the module map");
    assert_eq!(modules.len(), 2);
    assert_eq!(
        modules[0].get("path").unwrap().as_str(),
        Some("/bin/example")
    );
    assert_eq!(modules[0].get("load").unwrap().as_u64(), Some(0x1000));
    assert_eq!(modules[0].get("start").unwrap().as_u64(), Some(0x1000));
    assert_eq!(modules[0].get("size").unwrap().as_u64(), Some(0x1000));
    assert_eq!(modules[0].get("bias").unwrap().as_u64(), Some(0x400));
    assert_eq!(
        modules[0].get("buildId").unwrap().as_str(),
        Some("0badc0ffee")
    );
    assert!(
        modules[1].get("buildId").is_none(),
        "an image with no build identity should say nothing rather than null"
    );

    let frame = parsed.get("ftbl").unwrap().as_array().unwrap()[1]
        .as_str()
        .unwrap();
    // The rendered number is the address *in the file*: 0x1234 less the image's
    // bias of 0x400. An offset from the image base would be 0x234, which is the
    // number `addr2line` would resolve to the wrong line.
    assert_eq!(frame, "0x1234: ??? (/bin/example+0xe34)");
}

#[test]
fn an_address_outside_every_image_is_left_unattributed() {
    let profile = emit(&snapshot(vec![point(&[0xDEAD_0000], 64, 1)]));
    dhat::assert_valid(&profile);
    let parsed = json::parse(&profile).expect("valid JSON");
    let frame = parsed.get("ftbl").unwrap().as_array().unwrap()[1]
        .as_str()
        .unwrap();
    assert_eq!(frame, "0xdead0000: ???");
}

/// The heaviest point leads the file whatever order the snapshot held it in,
/// which is what makes the emitter's order a presentation choice rather than an
/// accident of the fold. Points merged by the fold come from anywhere in the
/// snapshot, so without this the file's leading rows would depend on which of
/// several identical stacks was interned first.
///
/// Reproducibility across runs is not this test's subject and never was
/// established here: it comes from `Snapshot::points` being canonically ordered
/// before the emitter sees it. Points that weigh the same are ordered by that,
/// which `equal_weights_keep_the_order_the_snapshot_gave_them` covers.
#[test]
fn the_heaviest_point_leads_whatever_order_the_snapshot_held_them_in() {
    let forwards = emit(&snapshot(vec![
        point(&[0x1000, 0x2000], 4096, 8),
        point(&[0x3000], 512, 2),
        point(&[0x1000, 0x4000], 4096, 4),
    ]));
    let backwards = emit(&snapshot(vec![
        point(&[0x1000, 0x4000], 4096, 4),
        point(&[0x3000], 512, 2),
        point(&[0x1000, 0x2000], 4096, 8),
    ]));
    assert_eq!(forwards, backwards);
}

// ---------------------------------------------------------------------------
// The validator itself. Every rule below must reject something, or the tests
// above prove nothing.
// ---------------------------------------------------------------------------

/// Applies `damage` to a parsed profile and returns the problems the validator
/// finds in the result.
fn damaged_by(profile: &str, damage: impl FnOnce(&mut BTreeMap<String, Value>)) -> Vec<String> {
    let Value::Object(mut root) = json::parse(profile).expect("valid JSON") else {
        panic!("the profile is an object");
    };
    damage(&mut root);
    dhat::problems(&render(&Value::Object(root)))
}

/// Emits a valid profile, applies `damage` to the parsed structure, and returns
/// the problems the validator finds in the result.
fn problems_after(damage: impl FnOnce(&mut BTreeMap<String, Value>)) -> Vec<String> {
    let profile = emit(&snapshot(vec![
        point(&[0x1000, 0x2000], 4096, 8),
        point(&[0x3000], 512, 2),
    ]));
    let Value::Object(mut root) = json::parse(&profile).expect("valid JSON") else {
        panic!("the profile is an object");
    };
    damage(&mut root);
    dhat::problems(&render(&Value::Object(root)))
}

/// Renders a parsed value back to JSON. Only what these tests need.
fn render(value: &Value) -> String {
    match value {
        Value::Null => String::from("null"),
        Value::Bool(value) => value.to_string(),
        Value::Number(raw) => raw.clone(),
        Value::String(text) => {
            let mut out = String::from("\"");
            for character in text.chars() {
                match character {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
            out
        }
        Value::Array(items) => {
            let rendered: Vec<String> = items.iter().map(render).collect();
            format!("[{}]", rendered.join(","))
        }
        Value::Object(members) => {
            let rendered: Vec<String> = members
                .iter()
                .map(|(key, value)| {
                    format!("{}:{}", render(&Value::String(key.clone())), render(value))
                })
                .collect();
            format!("{{{}}}", rendered.join(","))
        }
    }
}

#[test]
fn the_validator_accepts_what_the_emitter_writes() {
    assert!(problems_after(|_| {}).is_empty());
}

#[test]
fn the_validator_rejects_a_missing_mandatory_field() {
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
        let problems = problems_after(|root| {
            root.remove(field);
        });
        assert!(
            problems.iter().any(|p| p.contains(field)),
            "removing `{field}` was not reported: {problems:?}"
        );
    }
}

/// The trap that makes a validator necessary at all: the viewer reads `tl` and
/// never checks for it, so a file without it loads and renders `NaN`.
/// The mode decides `verb` and `bklt`, and the viewer checks neither against it.
/// A file that disagreed with itself would render an ad hoc profile under the
/// word "Allocated" with every heap column showing `NaN`.
#[test]
fn the_validator_rejects_a_mode_that_disagrees_with_the_rest_of_the_file() {
    let wrong_verb = problems_after(|root| {
        root.insert(
            String::from("verb"),
            Value::String(String::from("Occurred")),
        );
    });
    assert!(
        wrong_verb.iter().any(|problem| problem.contains("verb")),
        "{wrong_verb:?}"
    );

    let wrong_lifetimes = problems_after(|root| {
        root.insert(String::from("mode"), Value::String(String::from("ad-hoc")));
        root.insert(
            String::from("verb"),
            Value::String(String::from("Occurred")),
        );
    });
    assert!(
        wrong_lifetimes
            .iter()
            .any(|problem| problem.contains("bklt")),
        "an ad hoc profile claiming block lifetimes was accepted: \
         {wrong_lifetimes:?}"
    );

    let unknown = problems_after(|root| {
        root.insert(
            String::from("mode"),
            Value::String(String::from("sampling")),
        );
    });
    assert!(
        unknown.iter().any(|problem| problem.contains("sampling")),
        "{unknown:?}"
    );
}

/// `bklt: false` means the lifetime fields are *omitted*, and a zero in their
/// place is a measurement of something that did not happen. The viewer ignores
/// them either way, which is exactly why this has to be checked here.
#[test]
fn the_validator_rejects_lifetime_fields_a_non_heap_profile_cannot_have() {
    let ad_hoc = || {
        emit(&as_mode(
            snapshot(vec![point(&[0x1000, 0x2000], 4_096, 8)]),
            Mode::AdHoc,
        ))
    };

    // Every field, not one representative of each set. A review deleted six of
    // the seven per-point rules and one of the two top-level ones with the whole
    // suite still green, because the negative tests injected `tg` and `gb` and
    // generalised from there.
    for field in ["tg", "tuth"] {
        let problems = damaged_by(&ad_hoc(), |root| {
            root.insert(String::from(field), Value::Number(String::from("0")));
        });
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains(&format!("`{field}`"))),
            "a non-heap profile carrying `{field}` was accepted: {problems:?}"
        );
    }

    for field in ["tl", "mb", "mbk", "gb", "gbk", "eb", "ebk"] {
        let problems = damaged_by(&ad_hoc(), |root| {
            let Some(Value::Array(points)) = root.get_mut("pps") else {
                panic!("the profile has program points");
            };
            let Some(Value::Object(first)) = points.first_mut() else {
                panic!("the first point is an object");
            };
            first.insert(String::from(field), Value::Number(String::from("0")));
        });
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains(&format!(".{field}`"))),
            "a non-heap program point carrying `{field}` was accepted: {problems:?}"
        );
    }

    // And the units, which decide whether the viewer calls dimensionless
    // weights bytes.
    let renamed = damaged_by(&ad_hoc(), |root| {
        root.insert(String::from("bksu"), Value::String(String::from("blocks")));
    });
    assert!(
        renamed.iter().any(|problem| problem.contains("events")),
        "{renamed:?}"
    );
}

#[test]
fn the_validator_rejects_a_missing_tl_which_the_viewer_would_accept() {
    let problems = problems_after(|root| {
        let Some(Value::Array(points)) = root.get_mut("pps") else {
            panic!("pps is an array");
        };
        let Value::Object(first) = &mut points[0] else {
            panic!("a program point is an object");
        };
        first.remove("tl");
    });
    assert!(problems.iter().any(|p| p.contains("tl")), "{problems:?}");
}

#[test]
fn the_validator_rejects_a_repeated_frame_sequence() {
    let problems = problems_after(|root| {
        let Some(Value::Array(points)) = root.get_mut("pps") else {
            panic!("pps is an array");
        };
        let first = points[0].get("fs").unwrap().clone();
        let Value::Object(second) = &mut points[1] else {
            panic!("a program point is an object");
        };
        second.insert(String::from("fs"), first);
    });
    assert!(
        problems.iter().any(|p| p.contains("repeated")),
        "{problems:?}"
    );
}

#[test]
fn the_validator_rejects_a_frame_list_that_points_at_the_root() {
    let problems = problems_after(|root| {
        let Some(Value::Array(points)) = root.get_mut("pps") else {
            panic!("pps is an array");
        };
        let Value::Object(first) = &mut points[0] else {
            panic!("a program point is an object");
        };
        first.insert(
            String::from("fs"),
            Value::Array(vec![Value::Number(String::from("0"))]),
        );
    });
    assert!(
        problems.iter().any(|p| p.contains("tree root")),
        "{problems:?}"
    );
}

#[test]
fn the_validator_rejects_a_frame_index_past_the_table() {
    let problems = problems_after(|root| {
        let Some(Value::Array(points)) = root.get_mut("pps") else {
            panic!("pps is an array");
        };
        let Value::Object(first) = &mut points[0] else {
            panic!("a program point is an object");
        };
        first.insert(
            String::from("fs"),
            Value::Array(vec![Value::Number(String::from("999"))]),
        );
    });
    assert!(
        problems.iter().any(|p| p.contains("past the end")),
        "{problems:?}"
    );
}

#[test]
fn the_validator_rejects_a_frame_table_without_the_root() {
    let problems = problems_after(|root| {
        let Some(Value::Array(frames)) = root.get_mut("ftbl") else {
            panic!("ftbl is an array");
        };
        frames[0] = Value::String(String::from("main"));
    });
    assert!(
        problems.iter().any(|p| p.contains("[root]")),
        "{problems:?}"
    );
}

#[test]
fn the_validator_rejects_the_wrong_file_version() {
    let problems = problems_after(|root| {
        root.insert(
            String::from("dhatFileVersion"),
            Value::Number(String::from("1")),
        );
    });
    assert!(
        problems.iter().any(|p| p.contains("must be 2")),
        "{problems:?}"
    );
}

#[test]
fn the_validator_rejects_impossible_counters() {
    let problems = problems_after(|root| {
        let Some(Value::Array(points)) = root.get_mut("pps") else {
            panic!("pps is an array");
        };
        let Value::Object(first) = &mut points[0] else {
            panic!("a program point is an object");
        };
        // More live at the global peak than this point ever had live.
        first.insert(String::from("gb"), Value::Number(String::from("999999")));
    });
    assert!(
        problems.iter().any(|p| p.contains("global peak")),
        "{problems:?}"
    );
}

#[test]
fn the_validator_rejects_columns_that_do_not_sum_to_the_totals() {
    let problems = problems_after(|root| {
        let Some(Value::Array(points)) = root.get_mut("pps") else {
            panic!("pps is an array");
        };
        let Value::Object(first) = &mut points[0] else {
            panic!("a program point is an object");
        };
        let tb = first.get("tb").unwrap().as_u64().unwrap();
        first.insert(String::from("tb"), Value::Number((tb + 1).to_string()));
    });
    assert!(
        problems.iter().any(|p| p.contains("heapscope.totals")),
        "{problems:?}"
    );
}

/// `acc` is never written by this crate, but the viewer decodes it whatever
/// `bkacc` says and *asserts* on a value it cannot represent. A validator that
/// ignored the field would accept a file `dh_view.js` refuses.
#[test]
fn the_validator_rejects_access_counts_the_viewer_would_assert_on() {
    for bad in [
        vec![Value::Number(String::from("70000"))],
        vec![Value::Number(String::from("-3"))],
        vec![
            Value::Number(String::from("-2")),
            Value::Number(String::from("99999")),
        ],
    ] {
        let problems = problems_after(|root| {
            let Some(Value::Array(points)) = root.get_mut("pps") else {
                panic!("pps is an array");
            };
            let Value::Object(first) = &mut points[0] else {
                panic!("a program point is an object");
            };
            first.insert(String::from("acc"), Value::Array(bad));
        });
        assert!(problems.iter().any(|p| p.contains("acc")), "{problems:?}");
    }
}

#[test]
fn the_validator_rejects_bytes_that_are_held_by_no_blocks() {
    let problems = problems_after(|root| {
        let Some(Value::Array(points)) = root.get_mut("pps") else {
            panic!("pps is an array");
        };
        let Value::Object(first) = &mut points[0] else {
            panic!("a program point is an object");
        };
        first.insert(String::from("ebk"), Value::Number(String::from("0")));
        first.insert(String::from("eb"), Value::Number(String::from("64")));
    });
    assert!(
        problems
            .iter()
            .any(|p| p.contains("no blocks holding them")),
        "{problems:?}"
    );
}

/// The cross-check against the engine's own counters is the strongest rule
/// here. Losing the section it depends on must be a failure, not a silent skip.
#[test]
fn the_validator_rejects_a_profile_that_does_not_say_how_it_was_produced() {
    let problems = problems_after(|root| {
        let Some(Value::Object(extension)) = root.get_mut("heapscope") else {
            panic!("the heapscope section is an object");
        };
        extension.remove("shutdown");
    });
    assert!(
        problems.iter().any(|p| p.contains("shutdown")),
        "{problems:?}"
    );
}

#[test]
fn the_validator_rejects_an_unrecognised_shutdown_path() {
    let problems = problems_after(|root| {
        let Some(Value::Object(extension)) = root.get_mut("heapscope") else {
            panic!("the heapscope section is an object");
        };
        extension.insert(
            String::from("shutdown"),
            Value::String(String::from("sometime, somehow")),
        );
    });
    assert!(
        problems.iter().any(|p| p.contains("shutdown")),
        "a value nobody recognises answers the question no better than a \
         missing field: {problems:?}"
    );
}

#[test]
fn the_validator_rejects_a_profile_with_no_heapscope_section() {
    let problems = problems_after(|root| {
        root.remove("heapscope");
    });
    assert!(
        problems.iter().any(|p| p.contains("heapscope")),
        "{problems:?}"
    );
}

/// A misordered or overlapping map resolves an address against the wrong file,
/// and the wrong function name looks exactly like a right one.
#[test]
fn the_validator_rejects_a_module_map_that_would_resolve_addresses_wrongly() {
    let unsorted = problems_after(|root| {
        let Some(Value::Object(section)) = root.get_mut("heapscope") else {
            panic!("the heapscope section");
        };
        section.insert(
            String::from("modules"),
            Value::Array(vec![
                module_entry("/b", 0x9000, 0x100),
                module_entry("/a", 0x1000, 0x100),
            ]),
        );
    });
    assert!(
        unsorted.iter().any(|p| p.contains("before the previous")),
        "{unsorted:?}"
    );

    let anonymous = problems_after(|root| {
        let Some(Value::Object(section)) = root.get_mut("heapscope") else {
            panic!("the heapscope section");
        };
        section.insert(
            String::from("modules"),
            Value::Array(vec![module_entry("", 0x1000, 0x100)]),
        );
    });
    assert!(
        anonymous.iter().any(|p| p.contains("has no path")),
        "{anonymous:?}"
    );

    let empty = problems_after(|root| {
        let Some(Value::Object(section)) = root.get_mut("heapscope") else {
            panic!("the heapscope section");
        };
        section.insert(String::from("modules"), Value::Array(Vec::new()));
    });
    assert!(empty.iter().any(|p| p.contains("is empty")), "{empty:?}");

    let overlapping = problems_after(|root| {
        let Some(Value::Object(section)) = root.get_mut("heapscope") else {
            panic!("the heapscope section");
        };
        section.insert(
            String::from("modules"),
            Value::Array(vec![
                module_entry("/a", 0x1000, 0x8000),
                module_entry("/b", 0x2000, 0x100),
            ]),
        );
    });
    assert!(
        overlapping.iter().any(|p| p.contains("inside an image")),
        "{overlapping:?}"
    );

    let missing = problems_after(|root| {
        let Some(Value::Object(section)) = root.get_mut("heapscope") else {
            panic!("the heapscope section");
        };
        section.remove("modules");
    });
    assert!(
        missing.iter().any(|p| p.contains("no `modules`")),
        "{missing:?}"
    );
}

fn module_entry(path: &str, start: u64, size: u64) -> Value {
    Value::Object(BTreeMap::from([
        (String::from("path"), Value::String(String::from(path))),
        (String::from("load"), Value::Number(start.to_string())),
        (String::from("start"), Value::Number(start.to_string())),
        (String::from("size"), Value::Number(size.to_string())),
        (String::from("bias"), Value::Number(String::from("0"))),
    ]))
}

#[test]
fn the_validator_rejects_a_peak_after_the_end_of_the_run() {
    let problems = problems_after(|root| {
        root.insert(String::from("tg"), Value::Number(String::from("999999")));
    });
    assert!(
        problems.iter().any(|p| p.contains("after the end")),
        "{problems:?}"
    );
}

#[test]
fn the_validator_rejects_text_that_is_not_json_at_all() {
    assert!(!dhat::problems("").is_empty());
    assert!(!dhat::problems("[]").is_empty());
    assert!(!dhat::problems("{\"dhatFileVersion\":2,}").is_empty());
}

/// Our parser and our writer were written by the same hand on the same day, so
/// a bug in the escaping could easily be symmetric and invisible to every test
/// above. Python's `json` shares nothing with either.
///
/// Skipped, loudly, where there is no `python3` on the path.
///
/// The absence is established by running `python3 --version` rather than by
/// letting the real invocation fail, because "the spawn returned an error" is
/// not the same as "the interpreter is not there". Under an emulator — an
/// `amd64` container on Apple silicon, which is how this crate's Linux checks
/// run — spawning a missing binary succeeds and the child fails afterwards,
/// with nothing on stderr. That reported a container without Python as a defect
/// in the JSON writer, which is the mistake `ci/check-dhat-viewer.sh` documents
/// at length for the viewer: a tool that could not be run must never be
/// reported as a tool that said no.
///
/// Skipped quietly under Miri, which can neither create the temporary
/// directory nor spawn the interpreter. Left un-ignored it did more damage than
/// its own failure: an unsupported operation aborts the whole test binary, so
/// this one test suppressed the other thirty-two in this file.
#[test]
#[cfg_attr(miri, ignore = "needs a temporary directory and a python3 subprocess")]
fn a_profile_parses_in_an_independent_json_implementation() {
    let usable = std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_ok_and(|probe| probe.status.success());
    if !usable {
        eprintln!("skipping the independent parser check: no usable python3");
        return;
    }

    let awkward = "prog \"q\" \\b\\ \n\t \u{2028} \u{2029} µs 😀 \u{1}\u{1f} end";
    let mut snapshot = snapshot(vec![point(&[0x1000, 0x2000], 4096, 8)]);
    snapshot.command = String::from(awkward);
    let profile = emit(&snapshot);

    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("profile.json");
    std::fs::write(&path, &profile).expect("writing the profile");

    // Prints the code points of `cmd`, which survives a pipe intact whatever
    // the string contains.
    let script = "import json,sys\n\
                  d=json.load(open(sys.argv[1],encoding='utf-8'))\n\
                  print(' '.join(str(ord(c)) for c in d['cmd']))\n";
    let output = match std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&path)
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            eprintln!("skipping the independent parser check: python3: {error}");
            return;
        }
    };

    assert!(
        output.status.success(),
        "python3 could not read the profile: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Against the screened form, because that is what was written. The check
    // this test exists for is unaffected: every character below is one the
    // screen passes through untouched, so a `\uXXXX` the hand-rolled writer got
    // wrong still shows up here as a different code point.
    let expected: Vec<String> = display::screen(awkward)
        .chars()
        .map(|c| (c as u32).to_string())
        .collect();
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        expected.join(" "),
        "an independent parser read a different string than was written"
    );
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

/// Counters that are internally consistent, because the ones the engine
/// produces are: a point cannot have had more bytes live than it allocated.
fn arbitrary_counters() -> impl Strategy<Value = Counters> {
    (
        1u64..1_000_000,
        1u64..1_000,
        0u64..100,
        0u64..100,
        0u64..100,
    )
        .prop_map(
            |(total_bytes, blocks, live_permille, peak_permille, gmax_permille)| {
                let max_bytes = total_bytes * peak_permille / 100;
                // Bytes and blocks move together. Bytes with no block holding
                // them is not a state the engine can reach — the two are
                // snapshotted from the same instant — so generating one would
                // test the validator rather than the emitter.
                let curr_blocks = blocks * live_permille / 100;
                let at_gmax_blocks = blocks * gmax_permille / 100;
                let held = |bytes: u64, blocks: u64| if blocks == 0 { 0 } else { bytes };
                Counters {
                    total_bytes,
                    total_blocks: blocks,
                    total_lifetime: total_bytes,
                    curr_bytes: held(max_bytes * live_permille / 100, curr_blocks),
                    curr_blocks,
                    max_bytes,
                    max_blocks: blocks,
                    at_gmax_bytes: held(max_bytes * gmax_permille / 100, at_gmax_blocks),
                    at_gmax_blocks,
                }
            },
        )
}

fn arbitrary_point() -> impl Strategy<Value = ProgramPoint> {
    (
        // A small address pool, so that distinct points collide often.
        proptest::collection::vec(
            prop_oneof![Just(0x1000usize), Just(0x2000), Just(0x3000)],
            0..5,
        ),
        arbitrary_counters(),
        0u64..1000,
    )
        .prop_map(|(frames, counters, unretired_lifetime)| ProgramPoint {
            kind: PointKind::Recorded,
            frames,
            counters,
            unretired_lifetime,
        })
}

proptest! {
    // Proptest saves failing seeds to a file next to the test, which means
    // resolving the current directory -- and Miri's filesystem isolation makes
    // that a hard abort, which takes the whole test binary and every other test
    // in this file with it. Persistence is worth keeping natively, where it
    // turns a rare failure into a permanently reproducible one, so it is
    // dropped only under Miri. Same shape as `tests/differential.rs`.
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

    /// Whatever the engine hands the emitter, the file it writes must be one the
    /// viewer will open. This is the property the whole milestone is for.
    #[test]
    fn every_emitted_profile_is_valid(
        points in proptest::collection::vec(arbitrary_point(), 0..12),
        command in ".{0,80}",
        time_at_end in 0u64..u64::MAX,
        mode in prop_oneof![Just(Mode::Heap), Just(Mode::AdHoc), Just(Mode::Copy)],
    ) {
        let mut snapshot = snapshot(points);
        snapshot.command = command;
        snapshot.time_at_end = time_at_end;
        snapshot.stats.time_at_max = time_at_end / 2;
        // Every mode, because each one decides which fields the file carries and
        // the validator refuses a file that carries the wrong set.
        let snapshot = as_mode(snapshot, mode);
        let profile = emit(&snapshot);
        let problems = dhat::problems(&profile);
        prop_assert!(problems.is_empty(), "{problems:?}\n{profile}");
    }

    /// The same, with a rendering that collapses call sites — the case the fold
    /// exists for.
    #[test]
    fn every_emitted_profile_is_valid_even_when_frames_collapse(
        points in proptest::collection::vec(arbitrary_point(), 0..12),
    ) {
        let profile = emit_with(&snapshot(points), &OneName);
        let problems = dhat::problems(&profile);
        prop_assert!(problems.is_empty(), "{problems:?}\n{profile}");
    }

    /// Anything that reaches a string field comes back out as the screen leaves
    /// it, and the screen only ever replaces a character it names.
    ///
    /// `\p{Any}` rather than `.`, which in this generator excludes the line
    /// terminators — the characters most worth generating here.
    #[test]
    fn every_command_line_round_trips(command in r"\p{Any}{0,200}") {
        let mut snapshot = snapshot(vec![point(&[0x1000], 64, 1)]);
        snapshot.command = command.clone();
        let parsed = json::parse(&emit(&snapshot)).expect("valid JSON");
        prop_assert_eq!(parsed.get("cmd").unwrap().as_str().unwrap(), display::screen(&command));
    }

    /// The screen holds for every input, not only the ones a test thought of.
    #[test]
    fn no_command_line_can_put_a_control_character_in_a_profile(
        command in r"\p{Any}{0,200}",
    ) {
        let mut snapshot = snapshot(vec![point(&[0x1000], 64, 1)]);
        snapshot.command = command;
        let profile = emit(&snapshot);
        let offenders = display::offenders(&profile);
        prop_assert!(offenders.is_empty(), "{offenders:?}");
    }

    /// The same for an image path, which is the field a user has the least
    /// control over: it comes from the filesystem by way of the loader.
    #[test]
    fn no_image_path_can_put_a_control_character_in_a_profile(
        path in r"\p{Any}{0,120}",
    ) {
        let mut snapshot = snapshot(vec![point(&[0x1234], 64, 1)]);
        snapshot.modules = vec![Module {
            path,
            start: 0x1000,
            size: 0x1000,
            bias: 0x400,
            image_base: 0x1000,
            build_id: None,
        }];
        let profile = emit(&snapshot);
        let offenders = display::offenders(&profile);
        prop_assert!(offenders.is_empty(), "{offenders:?}");
    }
}
