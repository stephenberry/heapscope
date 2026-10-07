//! The testing API asserting on a real program, through a real allocator.
//!
//! PLAN.md section 6.9. The unit tests in `src/stats.rs` and `src/baseline.rs`
//! drive an engine directly, which is how the decision table gets exercised
//! exhaustively; nothing there goes through the `#[global_allocator]`, the
//! macros, the panic, or the profile a failure writes. This does.
//!
//! # Why one `#[test]`
//!
//! Because there is one engine per process and `cargo test` runs tests
//! concurrently: a second test allocating during the profiled window would be
//! counted into these totals, and a second test starting a profiler would be
//! refused. That constraint is not an artefact of this file — it is what the
//! module documentation tells users about their own budget tests, and this file
//! is the arrangement it recommends.
//!
//! # Deliberate failures
//!
//! Half of what is worth testing here is that the assertions **fail**. An
//! assertion that cannot fail is the exact defect this API is shaped to avoid,
//! so every macro is run once over the line as well as once under it, through
//! `catch_unwind`, with the panic hook silenced so that a passing run does not
//! print eleven panics that are not failures.
//!
//! The hook is only half of the quiet, and the first version of this file
//! claimed it was all of it. A failing assertion also writes a program-point
//! summary straight to file descriptor 2, which no panic hook and no test
//! harness intercepts: a green run of this file printed 57 lines of profile.
//! So dumping is **off by default here** and switched on only around the
//! failures that are about the dump. That also keeps the ordinals predictable,
//! because a dump that is turned off never claims one.

mod support;

use std::hint::black_box;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use heapscope::{Baseline, HeapStats, Tolerance};
use support::dhat;

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// Exactly one allocation, from a call site of its own.
#[inline(never)]
fn one_block() -> Box<[u8; 512]> {
    black_box(Box::new([0u8; 512]))
}

/// `bytes` live at once, from a call site of its own.
#[inline(never)]
fn hold(bytes: usize) -> Vec<u8> {
    let mut held = Vec::with_capacity(bytes);
    held.resize(bytes, 0xA5);
    black_box(held)
}

/// Runs `body`, and returns the panic message if it panicked.
///
/// The hook is replaced rather than left in place because these panics are the
/// subject, not a failure: without it every run of this file prints four panic
/// messages and five program-point summaries, which is indistinguishable from a
/// test that broke.
fn failure_message(body: impl FnOnce()) -> String {
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let outcome = panic::catch_unwind(AssertUnwindSafe(body));
    panic::set_hook(previous);

    let payload = outcome.expect_err("the assertion passed where it had to fail");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic payload that is a message")
}

/// Sets or clears `name`, without the profiler counting what that costs.
///
/// The reentrancy guard is how the profiler excludes its own bookkeeping from
/// its own figures, and a harness switching a mode between two measurements is
/// the same kind of work: it belongs to the test, not to the program under
/// measurement. Left uncounted, a platform where the environment block is
/// heap-allocated reports the harness as program growth.
fn set_environment(name: &str, value: Option<&str>) {
    let _quiet =
        heapscope::internals::guard::enter().expect("a test thread is not inside the profiler");
    match value {
        Some(value) => std::env::set_var(name, value),
        None => std::env::remove_var(name),
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "needs a real backtrace, and Miri cannot execute inline assembly"
)]
fn the_testing_api_gates_a_real_program() {
    // Somewhere that is not the working directory and that goes away with the
    // test, so that a deliberate failure's profile does not land in the tree.
    let directory = tempfile::tempdir().expect("a temporary directory");
    let dumps = directory.path().join("assert.json");
    let baseline_path = directory.path().join("work.txt");

    // Mutated in a few places below. Safe only because this binary runs one
    // test: `set_var` is a process-wide write, and nothing else here is reading
    // the environment concurrently.
    std::env::set_var(heapscope::stats::DUMP_VARIABLE, "off");
    std::env::remove_var(heapscope::baseline::UPDATE_VARIABLE);

    // ---- with nothing recording, a reading refuses rather than returning zeros ----
    // This is the assertion the whole module is shaped around: were it to
    // return a zeroed reading here, every budget below would pass without ever
    // running the program.
    assert_eq!(
        HeapStats::get().expect_err("an unprofiled process has no counters"),
        heapscope::StatsError::NotRecording
    );
    let unavailable = failure_message(|| heapscope::assert_max_bytes!(0));
    assert!(
        unavailable.contains("no heapscope profiler"),
        "{unavailable}"
    );
    assert!(
        !unavailable.contains("profile written to"),
        "a run that never happened has no profile to write: {unavailable}"
    );

    let profiler = heapscope::Profiler::builder()
        .no_output()
        .build()
        .expect("the profiler should start");

    // ---- a reading reflects what the program actually did ----
    let mark = HeapStats::get().expect("a running heap run has counters");
    let block = one_block();
    let after = HeapStats::get().expect("a running heap run has counters");
    assert_eq!(
        after.total_blocks,
        mark.total_blocks + 1,
        "one Box::new was not recorded as exactly one allocation"
    );
    assert!(
        after.curr_bytes >= mark.curr_bytes + 512,
        "the live bytes do not account for a 512-byte block"
    );
    assert_eq!(after.dropped_blocks, 0, "the ceiling was reached");

    // ---- a bare count is an exact count of the whole run ----
    heapscope::assert_alloc_count!(mark.total_blocks + 1);
    let wrong = failure_message(|| heapscope::assert_alloc_count!(mark.total_blocks));
    // Built from the mark rather than written out, because what the harness
    // allocated before it is not this test's to know; and through
    // `allocations_made`, because in practice it allocated nothing, so the
    // count is one and the sentence is singular.
    assert!(
        wrong.contains(&format!(
            "{}, not {}",
            allocations_made(mark.total_blocks + 1),
            thousands(mark.total_blocks)
        )),
        "{wrong}"
    );

    // A `usize` is what a call site has — `items.len()`, a budget computed from
    // a size — and a macro taking `u64` rejects every one of them. Read fresh:
    // `failure_message` above allocates the panic payload it copies out.
    let counted = HeapStats::get().unwrap().total_blocks as usize;
    heapscope::assert_alloc_count!(counted);
    heapscope::assert_max_bytes!(usize::MAX);
    // `NonZeroU64` met the bound these macros used to have, so it has to meet
    // the narrower one too.
    heapscope::assert_max_bytes!(std::num::NonZeroU64::MAX);

    // Each macro's trailing message, exercised once. Only `assert_max_bytes!`
    // had a test, so the wiring in the other two could be deleted rule by rule
    // with everything green.
    let annotated = failure_message(|| heapscope::assert_alloc_count!(0, "fixture {}", 7));
    assert!(annotated.contains("fixture 7"), "{annotated}");
    assert!(annotated.contains(" made, not 0"), "{annotated}");

    // ---- every form of the count, through the macro ----
    // Each arm is its own expansion, so each is exercised passing, with a
    // trailing comma and with a message: a rule miswired to the wrong body, or
    // a `since:` arm swallowing a bare count, would otherwise compile and pass.
    // Every ceiling arm passes at least once strictly *under* its ceiling,
    // because a ceiling arm wired to the equality passes at the ceiling.
    // One allocation is made after the mark, and nothing after it allocates
    // before the assertions do, because a passing assertion allocates nothing.
    let stage = HeapStats::get().unwrap();
    let staged = one_block();
    let whole = HeapStats::get().unwrap().total_blocks;
    let one = [0u8];
    heapscope::assert_alloc_count!(since: stage, 1);
    heapscope::assert_alloc_count!(since: stage, 1,);
    heapscope::assert_alloc_count!(since: stage, one.len());
    heapscope::assert_alloc_count!(since: stage, 1, "stage {}", 1);
    heapscope::assert_alloc_count!(since: stage, <= 1);
    heapscope::assert_alloc_count!(since: stage, <= 2,);
    heapscope::assert_alloc_count!(since: stage, <= one.len());
    heapscope::assert_alloc_count!(since: stage, <= 1, "stage {}", 1,);
    heapscope::assert_alloc_count!(since: stage, <= 2, "stage {}", 1);
    heapscope::assert_alloc_count!(<= whole);
    heapscope::assert_alloc_count!(<= whole,);
    heapscope::assert_alloc_count!(<= whole + 1);
    heapscope::assert_alloc_count!(<= whole as usize);
    heapscope::assert_alloc_count!(<= whole, "whole run {}", 1);
    heapscope::assert_alloc_count!(<= whole + 1, "whole run {}", 1);
    heapscope::assert_alloc_count!(whole,);
    // A count held in a variable named `since` is still a bare count. The
    // `since:` arms need the colon after it, and no expression has one there.
    let since = whole;
    heapscope::assert_alloc_count!(since);
    heapscope::assert_alloc_count!(since, "counted {}", "since");

    // And failing, with what was counted and from where. Each is read fresh,
    // because the message `failure_message` copies out is an allocation the
    // next count includes.
    // Exactly one so far, which is also the case the grammar has to get right.
    let made = HeapStats::get().unwrap().total_blocks - stage.total_blocks;
    assert_eq!(made, 1, "something allocated between the stage and here");
    let wrong_since = failure_message(|| heapscope::assert_alloc_count!(since: stage, 0));
    assert!(
        wrong_since.contains("1 allocation was made since the mark, not 0"),
        "{wrong_since}"
    );
    let made = HeapStats::get().unwrap().total_blocks - stage.total_blocks;
    let over_since = failure_message(
        || heapscope::assert_alloc_count!(since: stage, <= 0, "while compiling {}", "stage-7"),
    );
    assert!(
        over_since.contains(&format!(
            "{} since the mark, above the ceiling of 0",
            allocations_made(made)
        )),
        "{over_since}"
    );
    assert!(
        over_since.contains("while compiling stage-7"),
        "{over_since}"
    );
    let made = HeapStats::get().unwrap().total_blocks;
    let over_run = failure_message(|| heapscope::assert_alloc_count!(<= 0));
    assert!(
        over_run.contains(&format!(
            "{}, above the ceiling of 0",
            allocations_made(made)
        )),
        "{over_run}"
    );
    // The wording as well as the message, because the message alone is the
    // same whichever body the arm calls.
    let annotated = failure_message(|| heapscope::assert_alloc_count!(<= 0, "fixture {}", 8));
    assert!(annotated.contains("fixture 8"), "{annotated}");
    assert!(annotated.contains("above the ceiling of 0"), "{annotated}");
    let annotated =
        failure_message(|| heapscope::assert_alloc_count!(since: stage, 0, "fixture {}", 9));
    assert!(annotated.contains("fixture 9"), "{annotated}");
    assert!(annotated.contains("since the mark, not 0"), "{annotated}");

    // A mark ahead of the run is refused, through the macro, even against the
    // widest ceiling there is. The fields are public, so a mark can be edited
    // after it is read, and that is all it takes; zero would pass this.
    let mut edited = HeapStats::get().unwrap();
    edited.total_blocks += 1_000;
    let ahead = failure_message(|| heapscope::assert_alloc_count!(since: edited, <= u64::MAX));
    assert!(ahead.contains("the mark records"), "{ahead}");
    assert!(ahead.contains("cannot be known"), "{ahead}");
    // The note about the profile's scope belongs to a profile, and dumping is
    // off here.
    assert!(!over_since.contains("whole run"), "{over_since}");

    // A negative count or ceiling is refused in every form. Converted, `-1`
    // would be `u64::MAX`, a ceiling that cannot fail.
    for refused in [
        failure_message(|| heapscope::assert_alloc_count!(<= -1)),
        failure_message(|| heapscope::assert_alloc_count!(since: stage, <= -1)),
        failure_message(|| heapscope::assert_alloc_count!(since: stage, -1)),
    ] {
        assert!(
            refused.contains("not a byte count or an allocation count"),
            "{refused}"
        );
    }

    // ---- a budget is about the peak, and survives the memory being freed ----
    let held = hold(256 * 1024);
    let peak = HeapStats::get().unwrap().max_bytes;
    assert!(peak >= 256 * 1024, "the peak did not include a 256 KiB Vec");
    drop(held);
    drop(block);
    assert_eq!(
        HeapStats::get().unwrap().max_bytes,
        peak,
        "freeing lowered the recorded peak"
    );

    heapscope::assert_max_bytes!(peak);

    // Dumping on for exactly these failures. Everything else in this file
    // fails with it off, which is what keeps a green run quiet.
    std::env::set_var(heapscope::stats::DUMP_VARIABLE, &dumps);
    let first = failure_message(|| heapscope::assert_alloc_count!(0));
    let over = failure_message(|| heapscope::assert_max_bytes!(peak - 1));
    // `staged` is still live, so both fail.
    let staged_count = failure_message(|| heapscope::assert_alloc_count!(since: stage, <= 0));
    let staged_leak = failure_message(|| heapscope::assert_no_leaks!(since: stage));

    // ---- unset, a dump goes under the build directory, not the source tree ----
    // The directory is found here by looking for Cargo's tag, by name, above
    // this binary, rather than asked of the crate, and whichever layout Cargo
    // ran it from.
    std::env::remove_var(heapscope::stats::DUMP_VARIABLE);
    let defaulted = failure_message(|| heapscope::assert_alloc_count!(0));
    std::env::set_var(heapscope::stats::DUMP_VARIABLE, "off");
    let executable = std::env::current_exe().expect("this test's executable");
    let target = executable
        .ancestors()
        .skip(1)
        .find(|directory| directory.join("CACHEDIR.TAG").is_file())
        .expect("Cargo tags its target directory");
    let written = dumped_profile(&defaulted);
    assert_eq!(
        written.parent(),
        Some(target.join("heapscope").as_path()),
        "{defaulted}"
    );
    assert!(written.is_file(), "{defaulted}");
    std::fs::remove_file(&written).expect("the default dump is removable");
    drop(staged);

    // ---- a failure since a mark says its profile covers the whole run ----
    // A mark holds totals and no program points, so the profile cannot be
    // narrowed to the interval; what it can be is labelled. Whole-run failures
    // have nothing to qualify, and saying it there too would teach readers to
    // skip it.
    const WHOLE_RUN: &str = "cover everything since the run started or its counts last \
                             restarted, not only what followed the mark";
    for message in [&staged_count, &staged_leak] {
        assert!(message.contains("profile written to"), "{message}");
        assert!(message.contains(WHOLE_RUN), "{message}");
    }
    for whole_run in [&first, &over] {
        assert!(!whole_run.contains(WHOLE_RUN), "{whole_run}");
    }

    assert!(over.contains("peak live bytes reached"), "{over}");
    assert!(
        over.contains(&thousands(peak)),
        "the failure must name the peak it measured: {over}"
    );

    // ---- the trailing message reaches the failure ----
    let annotated =
        failure_message(|| heapscope::assert_max_bytes!(0, "while parsing {}", "fixture-7.json"));
    assert!(
        annotated.contains("while parsing fixture-7.json"),
        "{annotated}"
    );

    // ---- a failure writes a profile a viewer can open, and names it ----
    // Both are read back through the message rather than through the name this
    // test would have predicted: what a reader has to go on is the path in the
    // panic, and a message naming a file that is not the one written is the
    // failure the counter exists to prevent.
    let first_dump = dumped_profile(&first);
    let second_dump = dumped_profile(&over);
    assert_eq!(
        first_dump, dumps,
        "the first dump ignored {}",
        "the setting"
    );
    assert_eq!(second_dump, directory.path().join("assert.2.json"));
    assert_ne!(
        first_dump, second_dump,
        "the second dump landed on the first"
    );
    let staged_count_dump = dumped_profile(&staged_count);
    let staged_leak_dump = dumped_profile(&staged_leak);
    assert_eq!(staged_count_dump, directory.path().join("assert.3.json"));
    assert_eq!(staged_leak_dump, directory.path().join("assert.4.json"));

    for dump in [
        &first_dump,
        &second_dump,
        &staged_count_dump,
        &staged_leak_dump,
    ] {
        let text = std::fs::read_to_string(dump).expect("a dumped profile");
        dhat::assert_valid(&text);
    }

    // ---- writing a profile does not change what is being measured ----
    // Verbatim the property `write_text_summary` and `write_native` each had to
    // be given. A failing assertion formats a message, captures a snapshot,
    // prints a summary and writes a whole DHAT file, and none of it may reach
    // the run's totals — otherwise the profile a reader is sent to disagrees
    // with the message that sent them.
    //
    // The window is drawn tightly around the assertion and its panic, and the
    // hook is silenced *outside* it: `failure_message` copies the panic payload
    // out, which is an allocation belonging to this test rather than to the
    // profiler, and one allocation is exactly the size of the defect being
    // looked for.
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let before_failures = HeapStats::get().unwrap();
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| heapscope::assert_max_bytes!(0)));
    let after_failures = HeapStats::get().unwrap();
    panic::set_hook(previous);
    assert!(outcome.is_err(), "a budget of zero passed");
    assert_eq!(
        after_failures.total_blocks, before_failures.total_blocks,
        "a failing assertion recorded its own allocations into the profile"
    );
    assert_eq!(after_failures.total_bytes, before_failures.total_bytes);

    // ---- leaks ----
    let mark = HeapStats::get().unwrap();
    let leaked = one_block();
    let leak = failure_message(|| heapscope::assert_no_leaks!(since: mark));
    assert!(
        leak.contains("1 more block is live than at the mark"),
        "{leak}"
    );
    drop(leaked);
    // The message copied out above is itself live, and a mark is a difference:
    // holding it across the next assertion would report this test's own string
    // as the program's leak, correctly.
    drop(leak);

    let mark = HeapStats::get().unwrap();
    drop(one_block());
    heapscope::assert_no_leaks!(since: mark);
    // The same rule with a message. Reachable only by compiling until now, and
    // dropping the mark from it turns a passing assertion into the whole-run
    // form — which fails on any real test binary, as the next block proves.
    heapscope::assert_no_leaks!(since: mark, "after {} blocks", 1);

    // The bare form is about the whole run, which for a test binary that has
    // been running this long is never clean — which is exactly why the `since`
    // form exists, and why saying so is worth a test rather than a sentence.
    let whole_run = failure_message(|| heapscope::assert_no_leaks!());
    assert!(whole_run.contains("were never freed"), "{whole_run}");
    assert!(
        whole_run.contains("since: mark"),
        "the likeliest cause is an assertion written without a mark, so the \
         message has to name that: {whole_run}"
    );

    // ---- baselines ----
    let missing = failure_message(|| heapscope::assert_baseline!(&baseline_path));
    assert!(missing.contains("there is no baseline"), "{missing}");
    assert!(
        missing.contains(heapscope::baseline::UPDATE_VARIABLE),
        "{missing}"
    );

    // Held across the recording so that every figure in the baseline is
    // non-zero, including the two live ones. A baseline figure of zero cannot be
    // covered by any percentage, which would make the tolerance case below
    // depend on what happened to be live at the moment it ran.
    let held = hold(4 * 1024);

    // Switching the mode is the harness talking to itself, not the program
    // being measured, and on Windows it is not free: `set_var` converts the
    // name and the value to UTF-16, which allocates. Unguarded, that lands
    // between the recording and the check below and makes the pair disagree by
    // 56 bytes in two blocks, measured under Wine — and by nothing at all on
    // unix, which is the only reason this ever looked like a passing test.
    // Held around every call rather than only the ones between the pair, so
    // the profiler sees neither the blocks a switch takes nor the frees the
    // next one performs on them.
    set_environment(heapscope::baseline::UPDATE_VARIABLE, Some("1"));
    heapscope::assert_baseline!(&baseline_path);
    // Switched off by an off spelling rather than by removing the variable, and
    // that is what makes the check below portable. The assertion has to read
    // this variable before it can know which mode it is in, and on unix
    // `var_os` allocates for a *value* and not for a name: remove it and the
    // read is free, so a check that counted its own environment read would
    // still look correct everywhere but Windows. An off spelling costs an
    // `OsString` on every platform. It also exercises `HEAPSCOPE_UPDATE_BASELINE=0`
    // through the macro, which is the setting that once turned a gate into a
    // recorder.
    set_environment(heapscope::baseline::UPDATE_VARIABLE, Some("0"));

    // Immediately, with nothing in between. Recording and then checking must
    // agree exactly, and it only does because the whole assertion — the
    // environment read included — happens inside the reentrancy guard: reading
    // the baseline allocates and so does asking which mode this run is in, so
    // an unguarded check pushes the totals past the numbers it had just
    // written, and every gate in the world drifts upwards each time it runs.
    heapscope::assert_baseline!(&baseline_path);

    let recorded = Baseline::read(&baseline_path).expect("the baseline just written");
    let grown = hold(512 * 1024);
    let regressed = failure_message(|| heapscope::assert_baseline!(&baseline_path));
    assert!(regressed.contains("above the baseline"), "{regressed}");
    assert!(regressed.contains("maxBytes"), "{regressed}");
    assert!(
        regressed.contains(&baseline_path.display().to_string()),
        "{regressed}"
    );
    drop(grown);

    // ---- and a tolerance wide enough to cover the growth lets it through ----
    // Every figure, not just the one that grew most visibly: the first version
    // of this computed the percentage from `maxBytes` alone and the check failed
    // on `maxBlocks`, which is the same mistake as gating on one number.
    let now = HeapStats::get().unwrap();
    let needed = covering_percent(&recorded, &now);
    heapscope::assert_baseline!(&baseline_path, Tolerance::percent(needed));

    let regressed = failure_message(|| {
        heapscope::assert_baseline!(&baseline_path, Tolerance::percent(0), "run {}", 7)
    });
    assert!(regressed.contains("run 7"), "{regressed}");
    drop(held);

    // ---- and the run itself is still sound ----
    let stats = HeapStats::get().expect("the run is still recording");
    assert_eq!(stats.dropped_blocks, 0);
    assert!(stats.total_blocks >= 4);
    assert!(stats.max_bytes >= 512 * 1024);

    drop(profiler);

    // ---- a stopped run keeps its final numbers, and still refuses nonsense ----
    let final_stats = HeapStats::get().expect("a finished run has final counters");
    assert_eq!(final_stats.max_bytes, stats.max_bytes);
    assert_eq!(
        heapscope::EventStats::get().expect_err("a heap run has no event counters"),
        heapscope::StatsError::NotAnEventRun
    );

    // ---- dumping turns off, and the failure still says what it measured ----
    // A finished run still has a profile to write, so this is the setting doing
    // the work rather than there being nothing to dump.
    let before = written_files(directory.path());
    std::env::set_var(heapscope::stats::DUMP_VARIABLE, "off");
    let quiet = failure_message(|| heapscope::assert_max_bytes!(0));
    assert!(quiet.contains("peak live bytes reached"), "{quiet}");
    assert!(
        !quiet.contains("profile written to"),
        "a dump was written with dumping turned off: {quiet}"
    );
    assert_eq!(
        written_files(directory.path()),
        before,
        "a dump was written with dumping turned off"
    );

    assert!(
        !Path::new(heapscope::DEFAULT_OUTPUT_PATH).exists(),
        "a no_output profiler wrote a default profile into the working directory"
    );
    assert_dumps_are_confined(directory.path(), &dumps);
}

/// How many files are in `directory`.
fn written_files(directory: &Path) -> usize {
    std::fs::read_dir(directory)
        .expect("the temporary directory")
        .count()
}

/// Every profile this test wrote is inside the temporary directory.
///
/// A dump path that fell back to the default would litter the repository, which
/// is the failure `examples/cdylib_probe.rs` already documents once.
fn assert_dumps_are_confined(directory: &Path, first: &Path) {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(directory).expect("the temporary directory") {
        found.push(entry.expect("a directory entry").path());
    }
    assert!(found.contains(&first.to_path_buf()), "{found:?}");
    assert!(
        found.len() >= 3,
        "expected several dumps and a baseline: {found:?}"
    );
    for stray in ["heapscope-assert.json", "dhat-heap.json"] {
        assert!(
            !PathBuf::from(stray).exists(),
            "{stray} was written to the working directory"
        );
    }
}

/// The profile a failure says it wrote.
///
/// Read out of the message rather than predicted, because the path in the panic
/// is the whole of what sends a reader to the right file.
fn dumped_profile(message: &str) -> PathBuf {
    const WROTE: &str = "profile written to ";
    let line = message
        .lines()
        .find_map(|line| line.trim_start().strip_prefix(WROTE))
        .unwrap_or_else(|| panic!("no profile was named in the failure:\n{message}"));
    PathBuf::from(line.trim())
}

/// The same grouping the failure messages use, so a test can look for a figure
/// the way it is printed.
///
/// Written out again rather than reached through the crate, deliberately: a
/// check that computes its expectation by calling the code under test agrees
/// with it by construction and checks nothing.
fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::new();
    for (at, digit) in digits.chars().enumerate() {
        if at > 0 && (digits.len() - at).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

/// "N allocations were made", agreeing with N the way the failure messages do.
///
/// Written out here rather than reached through the crate, for the reason
/// [`thousands`] is.
fn allocations_made(count: u64) -> String {
    if count == 1 {
        String::from("1 allocation was made")
    } else {
        format!("{} allocations were made", thousands(count))
    }
}

/// The smallest whole percentage that lets every figure of `now` through
/// against `recorded`.
fn covering_percent(recorded: &Baseline, now: &HeapStats) -> u32 {
    let figures = [
        (recorded.curr_bytes, now.curr_bytes),
        (recorded.curr_blocks, now.curr_blocks),
        (recorded.max_bytes, now.max_bytes),
        (recorded.max_blocks, now.max_blocks),
        (recorded.total_bytes, now.total_bytes),
        (recorded.total_blocks, now.total_blocks),
    ];
    let mut needed = 0;
    for (baseline, measured) in figures {
        if measured <= baseline {
            continue;
        }
        assert!(
            baseline > 0,
            "a figure grew from zero, which no percentage covers"
        );
        // Rounded up, then one more, because the tolerance itself rounds down.
        let over = u128::from(measured - baseline) * 100;
        let percent =
            u32::try_from(over.div_ceil(u128::from(baseline)) + 1).expect("a sane percentage");
        needed = needed.max(percent);
    }
    needed
}
