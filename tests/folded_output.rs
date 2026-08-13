//! Folded stacks, read back as the flame graph tools read them.
//!
//! The format has no version, no header, and no validator: a folded file is
//! lines, and every consumer parses it with one regular expression. That is what
//! makes it worth shipping and also what makes it easy to get quietly wrong —
//! `inferno` and `speedscope` both take `stack count`, and a file that is
//! malformed in the ways this suite checks does not fail to open. It draws the
//! wrong picture.
//!
//! So this suite reads the file the way those tools do, and asserts the two
//! things they cannot:
//!
//! * **Every line is one stack and one count.** A frame that carried an
//!   unescaped separator, a point that rendered to no frames at all, or a count
//!   that is not a number each produce a file that still parses into *something*.
//! * **The counts sum to a figure the profile reports elsewhere.** This is the
//!   property [`FoldedMetric`] is built around and the only self-check the
//!   format admits: a flame graph's total width is a number the reader can find
//!   in the summary, so a metric read out of the wrong field is visible rather
//!   than merely different.
//!
//! What is checked by unit test rather than here is the rendering itself — the
//! frame order, the escaping, the refusal — against renderers a test controls.
//! This suite is about the whole emitter, including the [`Symbolized`] and
//! [`Trimmed`] pair the default path uses, whose output depends on the platform.
//!
//! [`Symbolized`]: heapscope::symbol::Symbolized
//! [`Trimmed`]: heapscope::symbol::Trimmed

mod support;

use heapscope::output::{ProgramPoint, Snapshot};
use heapscope::{FoldedMetric, Mode};
use proptest::prelude::*;
use support::snapshot::{as_mode, hand_built, point};

/// The bound `tests/native_output.rs` generates counters under, for the same
/// reason: the sum of every point has to stay exact, and a generated profile
/// whose totals saturated would be testing the generator.
const GENERATED_MAX_BYTES: u64 = u64::MAX / 16;

fn emit(snapshot: &Snapshot, metric: FoldedMetric) -> String {
    let mut out = Vec::new();
    snapshot
        .write_folded(&mut out, metric)
        .expect("a heap profile can be folded");
    String::from_utf8(out).expect("folded output is UTF-8")
}

/// One line, split the way `inferno` and `speedscope` split it.
///
/// Both take everything up to the **last** space as the stack and the rest as
/// the count, which is what lets a frame contain spaces — and every frame this
/// crate renders does, since `0x1044c81f0: name (/path+0x2c1f0)` has three.
/// Splitting on the first space instead would pass on a file of bare symbols and
/// fail on every real profile, so the test splits the way the tools do rather
/// than the way that is convenient.
fn split(line: &str) -> (Vec<&str>, u64) {
    let (stack, count) = line
        .rsplit_once(' ')
        .unwrap_or_else(|| panic!("a folded line has no count: {line:?}"));
    let count = count
        .parse::<u64>()
        .unwrap_or_else(|error| panic!("`{count}` is not a count ({error}): {line:?}"));
    (stack.split(';').collect(), count)
}

/// Everything wrong with `text` as a folded file, as a flame graph tool would
/// find it.
fn problems(text: &str) -> Vec<String> {
    let mut problems = Vec::new();
    if !text.is_empty() && !text.ends_with('\n') {
        problems.push(String::from("the last line has no newline"));
    }
    for line in text.lines() {
        if line.is_empty() {
            problems.push(String::from("a blank line"));
            continue;
        }
        let Some((stack, count)) = line.rsplit_once(' ') else {
            problems.push(format!("no count: {line:?}"));
            continue;
        };
        if count.parse::<u64>().is_err() {
            problems.push(format!("`{count}` is not a count: {line:?}"));
        }
        if stack.is_empty() {
            problems.push(format!("a count with no stack: {line:?}"));
        }
        // An empty frame is what an unescaped separator produces, and it renders
        // as a nameless band in the middle of the flame graph.
        for frame in stack.split(';') {
            if frame.is_empty() {
                problems.push(format!("an empty frame: {line:?}"));
            }
        }
    }
    problems
}

/// What the profile says the metric adds up to.
fn global(snapshot: &Snapshot, metric: FoldedMetric) -> u64 {
    let stats = &snapshot.stats;
    match metric {
        FoldedMetric::TotalBytes => stats.total_bytes,
        FoldedMetric::TotalBlocks => stats.total_blocks,
        FoldedMetric::PeakBytes => stats.max_bytes,
        FoldedMetric::LiveBytes => stats.curr_bytes,
    }
}

const EVERY_METRIC: [FoldedMetric; 4] = [
    FoldedMetric::TotalBytes,
    FoldedMetric::TotalBlocks,
    FoldedMetric::PeakBytes,
    FoldedMetric::LiveBytes,
];

fn snapshot() -> Snapshot {
    hand_built(vec![
        point(&[0x1000, 0x2000, 0x3000], 4096, 4),
        point(&[0x1500, 0x2000, 0x3000], 2048, 4),
        point(&[0x9000], 1024, 2),
    ])
}

/// The shape every consumer depends on, on the default path.
#[test]
fn every_line_is_a_stack_and_a_count() {
    for metric in EVERY_METRIC {
        let text = emit(&snapshot(), metric);
        assert!(
            !text.is_empty(),
            "{} produced nothing from a profile with counters",
            metric.as_str()
        );
        let found = problems(&text);
        assert!(found.is_empty(), "{}: {found:?}\n{text}", metric.as_str());
    }
}

/// **The property.** A flame graph's total width is a figure the reader can
/// check against the summary.
///
/// This is also what pins each metric to the field it claims. `PeakBytes` reads
/// `atGmaxBytes` and not a point's own `maxBytes`, and the two differ in every
/// fixture here — the second sums to more than the run's peak, because the
/// points reached their maxima at different instants. Nothing about the file's
/// *shape* would show that; the sum is what shows it.
#[test]
fn the_counts_sum_to_the_figure_the_profile_reports() {
    let snapshot = snapshot();
    for metric in EVERY_METRIC {
        let text = emit(&snapshot, metric);
        let summed: u64 = text.lines().map(|line| split(line).1).sum();
        assert_eq!(
            summed,
            global(&snapshot, metric),
            "{} sums to {summed}:\n{text}",
            metric.as_str()
        );
    }
}

/// Stacks are rooted at the outermost frame, which is what puts a flame graph's
/// `main` at the bottom.
///
/// Checked against the recorded frames rather than against a literal, because
/// the default renderer's text is a property of the platform: the two points
/// below share their outer two frames, so whatever those render to, the shared
/// prefix has to be at the *start* of both lines.
#[test]
fn stacks_share_a_prefix_where_the_program_shared_a_caller() {
    let text = emit(&snapshot(), FoldedMetric::TotalBytes);
    let stacks: Vec<Vec<&str>> = text.lines().map(|line| split(line).0).collect();
    let shared = stacks
        .iter()
        .find(|stack| stack.len() > 1)
        .expect("a stack with more than one frame");

    let siblings = stacks
        .iter()
        .filter(|stack| stack.first() == shared.first())
        .count();
    assert!(
        siblings >= 2,
        "the two points that share their outermost frame did not share a prefix: {stacks:?}"
    );
}

/// The default path trims, so the runtime entry sequence every stack carries is
/// not what a reader sees at the root of the flame graph.
///
/// A weak assertion on purpose: trimming reads frame *names*, and on Linux
/// `dladdr` names almost nothing, so a strong claim about which frames survive
/// would be a claim about the platform. What holds everywhere is that asking for
/// no trimming can only produce at least as many frames.
#[test]
fn the_untrimmed_rendering_is_never_shorter() {
    use heapscope::symbol::Symbolized;

    let snapshot = snapshot();
    let trimmed = emit(&snapshot, FoldedMetric::TotalBytes);

    let mut out = Vec::new();
    snapshot
        .write_folded_with(
            &mut out,
            &Symbolized::new(&snapshot.modules),
            FoldedMetric::TotalBytes,
        )
        .expect("the untrimmed rendering");
    let untrimmed = String::from_utf8(out).expect("UTF-8");

    let frames = |text: &str| -> usize { text.lines().map(|line| split(line).0.len()).sum() };
    assert!(
        frames(&untrimmed) >= frames(&trimmed),
        "trimming added frames:\n{trimmed}\n---\n{untrimmed}"
    );
    assert!(problems(&untrimmed).is_empty(), "{untrimmed}");
}

/// A run that records events has no live blocks, so two of the four metrics are
/// not measurements it took. The refusal names the metric, and the two that
/// every mode measures still work.
#[test]
fn an_event_run_refuses_the_metrics_it_never_measured() {
    for mode in [Mode::AdHoc, Mode::Copy] {
        let snapshot = as_mode(snapshot(), mode);
        for metric in [FoldedMetric::PeakBytes, FoldedMetric::LiveBytes] {
            let error = snapshot
                .write_folded(&mut Vec::new(), metric)
                .expect_err("an event is never live");
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
            assert!(error.to_string().contains(metric.as_str()), "{error}");
        }
        for metric in [FoldedMetric::TotalBytes, FoldedMetric::TotalBlocks] {
            let text = emit(&snapshot, metric);
            assert!(problems(&text).is_empty(), "{text}");
            assert_eq!(
                text.lines().map(|line| split(line).1).sum::<u64>(),
                global(&snapshot, metric)
            );
        }
    }
}

/// A refused write must not leave a file where the reader's previous one was.
///
/// This is the first emitter that can fail for a reason other than the disk, and
/// `save_with` creates its temporary *before* calling it — so the cleanup arm is
/// reached here by a path nothing else in the crate takes.
#[test]
#[cfg_attr(miri, ignore = "needs a real filesystem")]
fn a_refused_save_leaves_nothing_behind() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("leaked.folded");
    std::fs::write(&path, b"the file that was already there").expect("the previous file");

    let outcome = as_mode(snapshot(), Mode::AdHoc).save_folded(&path, FoldedMetric::LiveBytes);

    assert!(outcome.is_err(), "an ad hoc run wrote a live-bytes file");
    assert_eq!(
        std::fs::read(&path).expect("reading the previous file back"),
        b"the file that was already there",
        "a refused save replaced the file that was already there"
    );
    let leftovers: Vec<String> = std::fs::read_dir(directory.path())
        .expect("reading the directory back")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".tmp"))
        .collect();
    assert_eq!(
        leftovers,
        Vec::<String>::new(),
        "a refused save left its temporary behind"
    );
}

/// A directory named with a `;` does not add a level to the flame graph.
///
/// The realistic way a separator reaches a frame, and the reason it is worth an
/// integration test rather than only a unit one: every renderer this crate ships
/// puts the *image path* into the frame text, and a path is whatever the
/// filesystem allows. Nothing about the resulting file looks wrong — the stack
/// simply has a frame in it that the program never called.
///
/// Counted rather than matched, because what the frames are called depends on
/// the platform and what they are worth here is only how many there are.
#[test]
fn a_semicolon_in_an_image_path_does_not_invent_a_frame() {
    use heapscope::symbol::modules::Module;

    const FRAMES: usize = 3;

    let mut snapshot = hand_built(vec![point(&[0x1000, 0x1100, 0x1200], 4096, 4)]);
    snapshot.modules = vec![Module {
        path: String::from("/tmp/we;ird/program"),
        start: 0x1000,
        size: 0x1000,
        bias: 0,
        image_base: 0x1000,
        build_id: None,
    }];

    let text = emit(&snapshot, FoldedMetric::TotalBytes);
    let (stack, count) = split(text.lines().next().expect("one line"));
    assert_eq!(count, 4096);
    assert_eq!(
        stack.len(),
        FRAMES,
        "{FRAMES} frames became {} because a path carried the separator:\n{text}",
        stack.len()
    );
    assert!(
        text.contains(r"we\u{3b}ird"),
        "the separator was not escaped: {text}"
    );
}

/// A profile with no points is an empty file rather than a malformed one.
///
/// Worth its own case because "no lines" and "one blank line" are the same
/// number of bytes apart as a trailing `writeln!` that should not have run, and
/// `inferno` reports the second as a parse error.
#[test]
fn a_profile_with_nothing_recorded_is_an_empty_file() {
    assert_eq!(emit(&hand_built(vec![]), FoldedMetric::TotalBytes), "");
}

proptest! {
    // Failing seeds are persisted next to this file, except under Miri, where
    // resolving the current directory is a hard abort that takes the whole test
    // binary with it. Same shape as `tests/native_output.rs`.
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

    /// Whatever a snapshot holds, the folded file parses and adds up.
    ///
    /// The generators reach the shapes a recorded workload does not produce on
    /// demand: points with no frames at all, addresses across the whole `usize`
    /// range, and every mode — including the two where half these metrics are
    /// refused rather than written.
    #[test]
    fn every_folded_file_parses_and_adds_up(
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
        let snapshot = as_mode(hand_built(points), mode);

        for metric in EVERY_METRIC {
            let mut out = Vec::new();
            match snapshot.write_folded(&mut out, metric) {
                Ok(()) => {}
                Err(error) => {
                    // The only refusal, and it is predictable from the outside.
                    prop_assert!(
                        metric.needs_block_lifetimes() && !mode.block_lifetimes(),
                        "{} was refused in a {} run: {error}",
                        metric.as_str(),
                        mode.as_str()
                    );
                    prop_assert!(out.is_empty());
                    continue;
                }
            }
            let text = String::from_utf8(out).expect("UTF-8");
            let found = problems(&text);
            prop_assert!(found.is_empty(), "{}: {found:?}\n{text}", metric.as_str());

            let summed: u64 = text.lines().map(|line| split(line).1).sum();
            prop_assert_eq!(
                summed,
                global(&snapshot, metric),
                "{} sums to {}, and the profile reports {}\n{}",
                metric.as_str(),
                summed,
                global(&snapshot, metric),
                text
            );
        }
    }
}
