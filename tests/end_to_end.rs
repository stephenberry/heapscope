//! The profiler recording a real program, through a real `#[global_allocator]`.
//!
//! Every other test exercises a component with synthetic input. This one is the
//! only place where the whole chain runs as a user would run it: the shim
//! intercepts genuine `Vec` and `Box` allocations, the frame-pointer walker
//! captures genuine stacks, and the engine attributes them.
//!
//! It has to be an integration test because it installs a `#[global_allocator]`,
//! and it has to be a *single* test because there is one engine per process and
//! `cargo test` runs tests concurrently — a second profiler would either be
//! refused or blend two recordings.

mod support;

use std::hint::black_box;

use heapscope::internals::engine::Engine;
use support::dhat;
use support::json;

const FLUSH_TIMEOUT: std::time::Duration = Engine::FLUSH_TIMEOUT;

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// Allocates in a way the optimiser cannot elide, from a call site deep enough
/// to produce a distinguishable stack.
#[inline(never)]
fn allocate_vectors(count: usize, bytes: usize) -> Vec<Vec<u8>> {
    let mut kept = Vec::with_capacity(count);
    for _ in 0..count {
        let mut v: Vec<u8> = Vec::with_capacity(bytes);
        v.resize(bytes, 0xAB);
        kept.push(black_box(v));
    }
    kept
}

#[inline(never)]
fn allocate_and_drop(count: usize, bytes: usize) {
    for _ in 0..count {
        let v: Vec<u8> = Vec::with_capacity(bytes);
        black_box(&v);
    }
}

/// One test, doing everything, because the process has one engine.
#[test]
#[cfg_attr(
    miri,
    ignore = "needs a real backtrace, and Miri cannot execute inline assembly"
)]
fn a_real_workload_is_recorded_end_to_end() {
    // Somewhere that is not the working directory, and that goes away with the
    // test. Made before the profiler starts so that a panic below still cleans
    // up, and so that none of it lands in the profile.
    let directory = tempfile::tempdir().expect("a temporary directory");
    let output_path = directory.path().join("dhat-heap.json");

    let profiler = heapscope::Profiler::builder()
        .output(heapscope::Output::dhat_v2(output_path.clone()))
        .build()
        .expect("profiler should start");

    // ---- allocations are observed at all ----
    let before = profiler.stats();
    let kept = allocate_vectors(200, 1024);
    let after_alloc = profiler.stats();

    assert!(
        after_alloc.total_blocks > before.total_blocks,
        "no allocations were recorded through the global allocator"
    );
    assert!(
        after_alloc.total_bytes >= before.total_bytes + 200 * 1024,
        "recorded bytes ({}) do not account for 200 KiB of vectors",
        after_alloc.total_bytes - before.total_bytes
    );
    assert!(
        after_alloc.curr_bytes >= 200 * 1024,
        "live bytes ({}) do not reflect vectors that are still alive",
        after_alloc.curr_bytes
    );

    // ---- freeing brings live bytes back down ----
    let live_at_peak = profiler.stats().curr_bytes;
    drop(kept);
    let after_free = profiler.stats();
    assert!(
        after_free.curr_bytes < live_at_peak,
        "live bytes did not fall after 200 vectors were dropped \
         ({live_at_peak} -> {})",
        after_free.curr_bytes
    );
    assert!(
        after_free.max_bytes >= live_at_peak,
        "the peak was lost when the vectors were freed"
    );

    // ---- churn that never grows the heap must not raise the peak ----
    let peak_before_churn = profiler.stats().max_bytes;
    allocate_and_drop(5_000, 512);
    let after_churn = profiler.stats();
    assert_eq!(
        after_churn.max_bytes, peak_before_churn,
        "allocate-and-immediately-free churn moved the recorded peak"
    );
    assert!(
        after_churn.total_blocks >= after_free.total_blocks + 5_000,
        "churn was not counted toward the cumulative totals"
    );

    // ---- allocations are attributed to distinguishable program points ----
    let mut points = 0usize;
    let mut deepest = 0usize;
    let mut summed_total = 0u64;
    let mut summed_curr = 0u64;
    let mut summed_at_peak = 0u64;
    let flush = heapscope::engine().flush_and_visit(
        FLUSH_TIMEOUT,
        |_id, frames, counters| {
            points += 1;
            deepest = deepest.max(frames.len());
            summed_total += counters.total_bytes;
            summed_curr += counters.curr_bytes;
            summed_at_peak += counters.at_gmax_bytes;
        },
        |_| {},
        |_| {},
    );

    assert!(
        points > 1,
        "every allocation landed on one program point; the unwinder is not \
         distinguishing call sites"
    );
    assert!(
        deepest >= 3,
        "the deepest captured stack was {deepest} frames; frame pointers are \
         probably unavailable"
    );

    // ---- the invariants the whole design exists to guarantee ----
    // Read from the flush, not from a second call: `stats()` on its own is a
    // separate, unsynchronised acquisition, so an event landing in between would
    // break the summation with no bug anywhere.
    assert!(flush.exclusive, "the flush could not reach a quiet point");
    let stats = flush.stats;
    assert_eq!(
        summed_total, stats.total_bytes,
        "per-point cumulative bytes do not sum to the global total"
    );
    assert_eq!(
        summed_curr, stats.curr_bytes,
        "per-point live bytes do not sum to the global live bytes"
    );
    assert_eq!(
        summed_at_peak, stats.max_bytes,
        "per-point at-peak bytes do not sum to the global peak; this is the \
         invariant the peak gate exists to guarantee"
    );

    // ---- multi-threaded traffic keeps those invariants ----
    std::thread::scope(|s| {
        for t in 0..4 {
            s.spawn(move || {
                let mut kept = Vec::new();
                for i in 0..500 {
                    let v: Vec<u8> = Vec::with_capacity(64 + (t * 16) + (i % 32));
                    if i % 3 == 0 {
                        kept.push(black_box(v));
                    } else {
                        black_box(&v);
                    }
                }
                black_box(&kept);
            });
        }
    });

    let mut summed_total = 0u64;
    let mut summed_curr = 0u64;
    let mut summed_at_peak = 0u64;
    let mut summed_at_peak_blocks = 0u64;
    let flush = heapscope::engine().flush_and_visit(
        FLUSH_TIMEOUT,
        |_id, _frames, counters| {
            summed_total += counters.total_bytes;
            summed_curr += counters.curr_bytes;
            summed_at_peak += counters.at_gmax_bytes;
            summed_at_peak_blocks += counters.at_gmax_blocks;
        },
        |_| {},
        |_| {},
    );

    assert!(flush.exclusive, "the flush could not reach a quiet point");
    let stats = flush.stats;
    assert_eq!(summed_total, stats.total_bytes, "cumulative bytes drifted");
    assert_eq!(summed_curr, stats.curr_bytes, "live bytes drifted");
    assert_eq!(
        summed_at_peak, stats.max_bytes,
        "per-point at-peak bytes did not sum to the global peak after \
         concurrent traffic"
    );
    // The blocks half, added in M7 chunk J. PLAN.md section 12's third bullet
    // names `gb` *and* `gbk`, and the two are set by separate lines; every
    // concurrent check in this repository summed the bytes and left the blocks
    // to be assumed.
    assert_eq!(
        summed_at_peak_blocks, stats.max_blocks,
        "per-point at-peak blocks did not sum to the blocks at the peak after \
         concurrent traffic"
    );

    // ---- no internal failures along the way ----
    assert!(
        !heapscope::internals::diagnostic::is_poisoned(),
        "the profiler poisoned itself during a normal workload"
    );
    assert_eq!(
        heapscope::internals::order::violations(),
        0,
        "the lock-order checker reported a violation on a real workload"
    );

    // ---- a second profiler is refused ----
    assert!(
        heapscope::Profiler::new().is_err(),
        "two profilers were allowed to attach to one process"
    );

    // ---- a profile of a real run is a file the viewer will open ----
    // Written while the profiler is still recording, which is the harder case:
    // the snapshot has to be coherent without the run being over.
    // Writing a profile must not add to what is being profiled. Both the
    // snapshot and the writing hold the reentrancy guard for this reason, and
    // the summary path allocates enough — a `format!` per line — that losing
    // the guard shows up immediately.
    let before_output = profiler.stats();

    // Kept for the tier-1 comparison after the profiler is gone, so that the
    // check runs against frames a real workload produced rather than addresses
    // a test chose.
    let recorded = profiler.snapshot();

    let mut live_profile = Vec::new();
    profiler
        .snapshot()
        .write_dhat_v2(&mut live_profile)
        .expect("writing a profile to memory");

    let mut summary = Vec::new();
    profiler
        .snapshot()
        .write_text_summary(&mut summary, 5)
        .expect("writing a summary to memory");

    // The native emitter belongs in this block more than either of the others:
    // it is the most allocation-heavy one in the crate — a `HashMap` keyed by
    // address, a `Vec<Vec<u32>>` of frame indices, and a `String` per resolved
    // symbol. It was left out at first, and every other path that reaches it
    // does so with recording already stopped, so dropping its guard changed
    // nothing any test could see.
    let mut native_profile = Vec::new();
    profiler
        .snapshot()
        .write_native(&mut native_profile)
        .expect("writing a native profile to memory");

    let after_output = profiler.stats();
    assert_eq!(
        after_output.total_blocks,
        before_output.total_blocks,
        "writing a profile recorded {} allocations of its own",
        after_output.total_blocks - before_output.total_blocks
    );

    let live_profile = String::from_utf8(live_profile).expect("the writer produces UTF-8");
    let summary = String::from_utf8(summary).expect("the writer produces UTF-8");
    // A native profile taken mid-run is the harder case for it too: the counters
    // it cross-checks are read in one window while the program is still
    // allocating.
    let native_profile = String::from_utf8(native_profile).expect("the writer produces UTF-8");
    support::native::assert_valid(&native_profile);

    // ---- stopping makes the shim a pass-through, and writes the profile ----
    drop(profiler);
    let after_stop = heapscope::engine().stats();
    allocate_and_drop(1_000, 256);
    assert_eq!(
        heapscope::engine().stats().total_blocks,
        after_stop.total_blocks,
        "allocations were still being recorded after the profiler was dropped"
    );

    // Everything below runs with recording stopped, so the validator's own
    // allocations cannot change what it is validating.
    dhat::assert_valid(&live_profile);

    let written = std::fs::read_to_string(&output_path)
        .expect("dropping the profiler should have written the profile");
    dhat::assert_valid(&written);

    let parsed = json::parse(&written).expect("the file is valid JSON");
    let points = parsed
        .get("pps")
        .and_then(|pps| pps.as_array())
        .expect("the file has program points");
    assert!(
        points.len() > 1,
        "a real workload produced {} program points",
        points.len()
    );

    let frames = parsed
        .get("ftbl")
        .and_then(|ftbl| ftbl.as_array())
        .expect("the file has a frame table");
    assert!(frames.len() > 3, "the frame table holds only the root");
    let mut attributed = 0;
    let mut named = 0;
    for frame in frames.iter().skip(1) {
        let text = frame.as_str().expect("a frame is a string");
        assert!(
            text.starts_with("0x") && text.contains(": "),
            "every frame leads with its address: {text}"
        );
        if text.contains("+0x") {
            attributed += 1;
        }
        if !text.contains(": ???") {
            named += 1;
        }
    }
    assert!(
        attributed > 0,
        "no frame was attributed to a loaded image, so nothing in this profile \
         can be symbolized afterwards"
    );
    // Deliberately reported rather than asserted. How many of a binary's
    // symbols are visible to `dladdr` is a property of the platform and the
    // build, not of this crate: Apple's loader reads the whole symbol table,
    // ELF's reads only what the image exports, and a stripped image has
    // nothing to read. Requiring a name here would make the test fail on
    // configurations where the profiler is working exactly as designed. What
    // *is* asserted, below, is the part that must hold everywhere.
    eprintln!("{named} of {} frames carry a name", frames.len() - 1);
    naming_a_frame_never_costs_the_ability_to_resolve_it(&recorded);
    trimming_drops_the_runtime_and_keeps_the_call_site(&recorded);

    // The cumulative totals in the file must match what the engine reported
    // while it was running: the profile is the same data, not a re-measurement.
    let totals = parsed
        .get("heapscope")
        .and_then(|section| section.get("totals"))
        .expect("the heapscope section");
    assert_eq!(
        totals.get("totalBlocks").and_then(|v| v.as_u64()),
        Some(after_stop.total_blocks),
        "the file disagrees with the engine about how many blocks were allocated"
    );

    assert!(summary.contains("heapscope profile"), "{summary}");
    assert!(summary.contains("at t-gmax"), "{summary}");
    assert!(summary.contains("0x"), "the summary should name call sites");

    the_default_output_trims(&recorded, &parsed, &summary);
    symbolization_resolves_a_recorded_frame(&parsed);
}

/// What `Snapshot::write_dhat_v2` and `write_text_summary` do with no arguments
/// — which is what almost every profile is written by.
///
/// Both wrap the renderer in `Trimmed`, and until this existed nothing checked
/// that they did. Removing `Trimmed` from either one left all twelve suites
/// green, on a run where 118 of 243 frames are trimmed, so the mutation is not
/// vacuous: it changes the file materially and no assertion could see it.
///
/// The reason the gap was easy to leave is worth naming. Every other test of
/// trimming builds its own `Trimmed::new(Symbolized::new(..))` and asks what it
/// produces, which tests the *rule*. Nothing asked what the crate does when
/// nobody chooses a renderer — and that is the only thing a user sees.
///
/// Conditional on the platform naming anything, for the reason the naming count
/// above gives: on Linux `dladdr` sees almost nothing, so there is nothing to
/// trim and the assertion would be false through no fault of this code. The
/// negative branch is not a skip — it pins the other half of the claim, that
/// where nothing can be read nothing is removed.
fn the_default_output_trims(snapshot: &heapscope::Snapshot, written: &json::Value, summary: &str) {
    use heapscope::output::FrameFormat;
    use heapscope::symbol::{Symbolized, Trimmed};

    let names = Symbolized::new(&snapshot.modules);
    let trimmed = Trimmed::new(Symbolized::new(&snapshot.modules));
    let hidden: usize = snapshot
        .points
        .iter()
        .map(|point| {
            let stack: Vec<String> = point
                .frames
                .iter()
                .map(|&address| {
                    let mut frame = String::new();
                    names.format(address, &mut frame);
                    frame
                })
                .collect();
            stack.len() - trimmed.keep(&stack).len()
        })
        .sum();
    let captured: usize = snapshot.points.iter().map(|p| p.frames.len()).sum();
    eprintln!("the trimming rules hide {hidden} of {captured} captured frames");

    let reported = written
        .get("heapscope")
        .and_then(|section| section.get("trimmedFrames"))
        .and_then(json::Value::as_u64)
        .expect("the profile records how many frames the rendering hid");

    if hidden == 0 {
        eprintln!(
            "nothing in this profile can be trimmed, so the default is checked \
             only for leaving it alone"
        );
        assert_eq!(
            reported, 0,
            "the default rendering removed frames the rules would have kept"
        );
        assert!(
            !summary.contains("not shown"),
            "the summary claims frames were hidden and none was:\n{summary}"
        );
        return;
    }

    // Deliberately not `reported == hidden`. `written` came from
    // `Profiler::drop` and `snapshot` from a reading taken earlier, with the
    // profiler still recording in between, so the two hold different sets of
    // program points and an equality would be a flake waiting for a slow
    // machine. What is compared instead is the claim: the rules find something
    // to hide, so the default must have hidden something.
    assert!(
        reported > 0,
        "the trimming rules hide {hidden} frames of this profile and the file \
         written by `write_dhat_v2` hid none, so the default is not using them"
    );
    assert!(
        summary.contains("frames are not shown"),
        "the summary written by `write_text_summary` hid nothing, and the \
         trimming rules would have hidden {hidden} frames:\n{summary}"
    );

    // And the artifact itself, stated in frames the rules never name — the same
    // oracle `trimming_drops_the_runtime_and_keeps_the_call_site` uses, applied
    // to the file rather than to a rendering this test made for itself.
    // Read per program point rather than straight off `ftbl`. That table pools
    // the frames of every point, so a runtime frame in it says only that *some*
    // stack kept one — and one stack is allowed to. `symbol::trim` keeps a stack
    // that never reaches `__rust_begin_short_backtrace` whole on purpose, which
    // `is_anchored` covers for the in-memory stacks and which this check, read
    // flat, could not see at all.
    //
    // The marker itself cannot do the telling apart here — trimming has already
    // removed it from the file. What separates the two cases without it is that
    // a stack trimming keeps whole is one that never ran any of this program's
    // code: it is machinery from end to end, which is why there was no marker on
    // it. Broken trimming produces the opposite shape, startup frames still
    // attached to stacks that do name this binary's own functions.
    //
    // So the rule is: a point may keep runtime frames only if it names nothing
    // of ours.
    //
    // A first attempt classified by whether the *outermost* frame was an entry
    // point, which sounds equivalent and is worse. It reported the ASan case as
    // a failure, because the outermost frame of a macOS thread stack is
    // libsystem's bare `thread_start`, one further out than `_pthread_start`
    // and not named by `is_runtime`. Making it right meant naming that frame —
    // and `clone3` on Linux, and `RtlUserThreadStart` on Windows — which is the
    // per-platform list `symbol::trim` explains it does not keep, growing in a
    // test instead of in the rules. Asking what is ours needs no such list.
    let ftbl: Vec<&str> = written
        .get("ftbl")
        .and_then(json::Value::as_array)
        .expect("the frame table")
        .iter()
        .filter_map(json::Value::as_str)
        .collect();

    let mut survivors: Vec<&str> = Vec::new();
    let (mut checked, mut kept_whole) = (0usize, 0usize);
    for point in written
        .get("pps")
        .and_then(json::Value::as_array)
        .expect("the program points")
    {
        let frames: Vec<&str> = point
            .get("fs")
            .and_then(json::Value::as_array)
            .expect("a program point's frame list")
            .iter()
            .filter_map(json::Value::as_u64)
            .map(|index| ftbl[usize::try_from(index).expect("a frame index that fits")])
            .collect();

        if frames.iter().any(|frame| frame.contains("end_to_end::")) {
            checked += 1;
            survivors.extend(frames.iter().filter(|frame| is_runtime(frame)));
        } else {
            kept_whole += 1;
        }
    }

    assert!(
        survivors.is_empty(),
        "the written profile still carries runtime frames on stacks that name \
         this binary's own functions, across {checked} such program points: \
         {survivors:#?}"
    );
    // Where the platform names our functions, the assertion above is the strong
    // form of the claim and the count says how much it looked at. Where it names
    // none of them, that assertion is vacuously true and says nothing — so which
    // case this run was is stated rather than left to be assumed.
    //
    // Not a skip, by the same argument the `hidden == 0` branch above makes: the
    // claim is still pinned there, by `reported > 0`, which needs no name at
    // all. Both mutations this check exists for are caught in either case —
    // a default renderer that stops trimming makes `reported` zero.
    if checked > 0 {
        eprintln!(
            "{checked} written program points name a function of this binary and \
             carry no runtime frame; {kept_whole} ran none of our code and are \
             kept whole by design"
        );
    } else {
        eprintln!(
            "the platform named no function of this binary in the written \
             profile, so its {kept_whole} program points cannot say whether \
             trimming ran; `trimmedFrames` above is what pins it here"
        );
    }
}

/// Frames a reader of a heap profile would call startup.
///
/// Named independently of the rules that remove them: `symbol::trim` cuts at
/// `__rust_begin_short_backtrace` and at a list of allocation-path prefixes, and
/// none of these appears in either. They are what those cuts *reach*, so a rule
/// that stopped working would leave them in place.
///
/// `std::thread::lifecycle::spawn_unchecked` is deliberately not here, and
/// finding out why is what this oracle was worth writing for. It appears on both
/// sides of the boundary: below the marker on a spawned thread, where it is the
/// machinery that started it, and *above* the marker on the parent, where it is
/// the frame that boxed the closure and is the honest answer to where those
/// bytes came from. The same name, two roles, decided by which side of the
/// marker it falls on — which is why `symbol::trim` cuts by position and not by
/// a list of names, and why a list of names is the wrong oracle for anything but
/// the frames that can only ever be outermost.
fn is_runtime(frame: &str) -> bool {
    [
        "std::rt::lang_start",
        ">::new::thread_start",
        "_pthread_start",
        "__libc_start_main",
    ]
    .iter()
    .any(|machinery| frame.contains(machinery))
}

/// Whether trimming has a boundary to cut this stack at.
///
/// `symbol::trim` cuts at exactly one marker and deliberately not at a list of
/// entry-point names, because such a list is per-platform and wrong outright for
/// a library loaded into a host that is not Rust. Its module documentation
/// states the price of that choice: a stack with no marker — "an allocation made
/// before `main`, or on a thread a C library created" — keeps its runtime
/// frames, "which is the honest answer rather than a guessed one".
///
/// So a stack without the marker cannot be evidence about trimming in either
/// direction, and counting one as a failure asserts the opposite of what the
/// rules say. That is not hypothetical: a spawned thread's teardown path can
/// allocate — `stack_overflow::imp::drop_handler` initializing a `OnceBox`
/// whose first use lands there — producing a stack with no user frame on it at
/// all. It is a first-initialization, so which thread pays for it is a matter
/// of timing; under ASan it is this one, and the assertion below fired on a
/// crate that had done exactly what it documents.
fn is_anchored(stack: &[String]) -> bool {
    stack
        .iter()
        .any(|frame| frame.contains("std::sys::backtrace::__rust_begin_short_backtrace"))
}

/// Tier 1 is only safe as the default because it takes nothing away.
///
/// `Symbolized` renders a name *in front of* what `ModuleOffsets` would have
/// rendered, never instead of it. That is what lets a profile written on a
/// machine with symbols still be resolved on one without, and it is the claim
/// that would break silently: a rendering that dropped the image and offset
/// once it had a name would look better in every summary and would quietly stop
/// producing files this test's own offline symbolization check can use.
///
/// Checked here against real recorded frames rather than in a unit test alone,
/// because the unit test supplies its own symbol table and cannot reach the case
/// where the platform names something unexpected.
fn naming_a_frame_never_costs_the_ability_to_resolve_it(snapshot: &heapscope::Snapshot) {
    use heapscope::output::FrameFormat;

    let bare = heapscope::symbol::ModuleOffsets::new(&snapshot.modules);
    let named = heapscope::symbol::Symbolized::new(&snapshot.modules);

    let mut checked = 0;
    for point in &snapshot.points {
        for &address in &point.frames {
            let (mut without, mut with) = (String::new(), String::new());
            bare.format(address, &mut without);
            named.format(address, &mut with);

            let (runtime_address, image) = without
                .split_once(": ???")
                .expect("ModuleOffsets renders the address, `: ???`, then the image");
            assert!(
                with.starts_with(runtime_address),
                "`{with}` lost the runtime address `{runtime_address}`"
            );
            assert!(
                with.ends_with(image),
                "`{with}` lost the image attribution `{image}`"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "a real workload recorded no frames at all");
}

/// Trimming earns its place only if it removes what nobody reads and keeps what
/// everybody does. Both halves are checked here against stacks this process
/// actually walked, because the rules in `symbol::trim` were written from
/// measured output and a unit test built from the same measurements cannot tell
/// anyone whether the measurement still holds.
///
/// Neither half is stated in the terms the rules are. `symbol::trim` cuts at
/// `__rust_begin_short_backtrace` and at a list of allocation-path prefixes;
/// what is asserted below is that `std::rt::lang_start`, the thread-spawning
/// machinery, and `pthread`'s entry point are *gone* — frames the rules never
/// mention and only reach as a consequence. A rule that stopped working would
/// leave them in place.
///
/// The one place that phrasing has to admit the rules exist is `is_anchored`:
/// a stack the marker never appears on is one the rules keep whole on purpose,
/// so it is counted separately rather than read as a trimming failure. It is
/// not a skip — the count of anchored stacks is asserted to be non-zero, so the
/// exclusion cannot grow until the test is examining nothing.
fn trimming_drops_the_runtime_and_keeps_the_call_site(snapshot: &heapscope::Snapshot) {
    use heapscope::output::FrameFormat;
    use heapscope::symbol::{Symbolized, Trimmed};

    let whole = Symbolized::new(&snapshot.modules);
    let trimmed = Trimmed::new(Symbolized::new(&snapshot.modules));

    let (mut captured, mut shown) = (0usize, 0usize);
    let (mut runtime_before, mut runtime_after) = (0usize, 0usize);
    let (mut sites, mut sites_kept) = (0usize, 0usize);
    let (mut anchored, mut anchorless) = (0usize, 0usize);
    let mut named = 0usize;
    // Frames the platform named as this binary's own code, which is the only
    // honest predicate for "could trimming have had anything to act on".
    //
    // Not `runtime_before`, which this used to be and which is measuring
    // something else entirely: `is_runtime` contains `__libc_start_main`, a
    // glibc `.dynsym` **export** and therefore one of the few names `dladdr`
    // can resolve on ELF. So on aarch64 Linux CI it counted 4 while the run
    // named 1 frame in 173, and the guards below fired on a symbol table that
    // says nothing about whether this crate's own frames are nameable.
    //
    // `__rust_begin_short_backtrace` and every `end_to_end::` function are
    // both non-exported symbols of this test binary, so a symbol table that
    // names one names the other. That is the whole argument, and it is one a
    // reader can check.
    let mut ours = 0usize;

    for point in &snapshot.points {
        let stack: Vec<String> = point
            .frames
            .iter()
            .map(|&address| {
                let mut frame = String::new();
                whole.format(address, &mut frame);
                frame
            })
            .collect();
        let keep = trimmed.keep(&stack);
        assert!(
            keep.end <= stack.len() && (stack.is_empty() || keep.start < keep.end),
            "trimming returned {keep:?} for a stack of {} frames",
            stack.len()
        );
        let kept = &stack[keep];

        captured += stack.len();
        shown += kept.len();
        runtime_before += stack.iter().filter(|frame| is_runtime(frame)).count();
        named += stack
            .iter()
            .filter(|frame| !frame.contains(": ???"))
            .count();
        ours += stack
            .iter()
            .filter(|frame| frame.contains("end_to_end::"))
            .count();
        if is_anchored(&stack) {
            anchored += 1;
            runtime_after += kept.iter().filter(|frame| is_runtime(frame)).count();
        } else {
            anchorless += 1;
        }

        // The frame naming the function that made the allocations. If the
        // platform named it at all, trimming must not be what takes it away.
        if stack.iter().any(|frame| frame.contains("allocate_vectors")) {
            sites += 1;
            if kept.iter().any(|frame| frame.contains("allocate_vectors")) {
                sites_kept += 1;
            }
        }
    }

    assert_eq!(
        runtime_after, 0,
        "{runtime_after} runtime frames survived trimming, across the {anchored} \
         program points whose stacks reach the marker trimming cuts at"
    );

    // The qualification above must not be what makes the claim pass: if every
    // stack were excluded, the assertion would hold having looked at nothing.
    //
    // Conditional on `ours` for the same reason the branch at the end of this
    // function is, and it is the same condition. Reaching the marker means
    // recognising it by name, so on a platform whose symbol table does not name
    // this binary's own functions — `dladdr` on Linux names 0 of 145 frames
    // here — no stack can be anchored and demanding one is demanding a symbol
    // table. Asserting it unconditionally is exactly the regression this branch
    // exists to prevent, and it was written that way first: green on macOS, red
    // on Linux, for a property of the platform rather than of the crate.
    if ours > 0 {
        assert!(
            anchored > 0,
            "the platform named {ours} of this binary's own frames, so the marker \
             was nameable too, yet no stack reached `__rust_begin_short_backtrace` \
             and all {anchorless} were excluded — this test examined nothing"
        );
        eprintln!(
            "{anchored} program points had a marker to trim at and kept no \
             runtime frame; {anchorless} had none and are kept whole by design"
        );
    } else {
        eprintln!(
            "the platform named none of this binary's own frames ({named} of \
             {captured} frames carry any name at all), so no stack here could \
             reach the marker; all {anchorless} program points are excluded, the \
             branch at the end of this function is what pins the claim here, and \
             `tests/symbolize.rs` is what pins trimming on this platform — it \
             resolves the profile offline first, which is what gives the rules \
             names to read"
        );
    }
    assert_eq!(
        sites,
        sites_kept,
        "trimming removed the frame naming the function that allocated, from \
         {} of the {sites} program points that had it",
        sites - sites_kept
    );

    // Whether anything *can* be trimmed is a property of the platform's symbol
    // table, not of this crate — see the note above the naming count. So the
    // claim is conditional on the one thing that settles it: whether this
    // binary's own frames are nameable, which is what both trimming rules read.
    //
    // This condition has to match the one above, and for the same reason. It
    // was `runtime_before` too, and on a run where the only name is libc's
    // `__libc_start_main` — neither the marker nor an allocation-path prefix,
    // and the allocation-path cut is a leading run from the innermost end —
    // nothing is trimmed and `shown == captured`. Fixing only the first guard
    // moves the failure here.
    if ours > 0 {
        assert!(
            shown < captured,
            "{ours} of this binary's own frames were rendered and none was trimmed"
        );
    } else if named == 0 {
        // Not a skip. Where the platform names nothing, the rules must find
        // nothing — that is the documented behaviour on a stripped build and on
        // Linux, and it is the half of the claim this branch can still pin.
        // Without it every Linux run of this test asserts precisely nothing.
        eprintln!(
            "no frame carries a name, so nothing here could be trimmed; \
             {captured} frames kept whole"
        );
        assert_eq!(
            shown, captured,
            "frames were trimmed from a profile in which nothing is named, so \
             something is being removed for a reason it cannot have read"
        );
    } else {
        // Some names, none of them this binary's own. The branch above used to
        // cover this case too, keyed on `runtime_before` while its message
        // spoke of nothing being named — two different conditions that agree
        // only when symbolization is all or nothing. ELF exports fewer symbols
        // than Mach-O, and under a sanitizer build the instrumentation's own
        // are visible as well, so a run naming 30 of 143 frames is ordinary.
        // The allocation-path cut works on the names there are, so frames are
        // legitimately removed while no runtime frame was ever there to remove,
        // and the old assertion called that a defect **\[measured, ASan on
        // x86_64 Linux, and reproduced at `d99525b`\]**.
        //
        // The aarch64 Linux CI runner is the other shape of this: 1 frame named
        // in 173, and that one *is* the runtime's, because `__libc_start_main`
        // is a `.dynsym` export while nothing of ours is. So this branch may
        // not say "none names the runtime" — on that runner the runtime is the
        // only thing named. It reports both counts instead.
        //
        // Nothing further is pinned here, and nothing needs to be: the two
        // assertions above this branch — no runtime frame survives a stack that
        // had a marker, and the frame naming the call site is never the one
        // removed — do not depend on the platform naming a frame of ours.
        eprintln!(
            "{named} of {captured} frames carry a name, {runtime_before} of them \
             the runtime's and none this binary's own, so neither trimming rule \
             has a frame of ours to read; {shown} shown"
        );
    }
    eprintln!("{captured} frames captured, {shown} shown after trimming");
}

/// The M2 exit criterion: a profile must be resolvable back to source-level
/// names *after the fact*, by a tool that was not running when it was recorded.
///
/// The strong form of the check is the second half: it takes a function whose
/// name this test already knows, converts its address through the numbers the
/// *profile* recorded, and requires every symbolizer on the machine to name it.
/// A wrong bias or a wrong load address then produces a different function, or
/// none, rather than something that still looks plausible.
fn symbolization_resolves_a_recorded_frame(profile: &json::Value) {
    let modules = parse_modules(profile);
    assert!(!modules.is_empty(), "the profile carries no module map");

    // ---- the recorded frames are attributed to this binary ----
    let frames: Vec<&str> = profile
        .get("ftbl")
        .and_then(|ftbl| ftbl.as_array())
        .expect("the frame table")
        .iter()
        .skip(1)
        .filter_map(|frame| frame.as_str())
        .collect();

    let executable = std::env::current_exe().expect("the path of this test binary");
    let executable = executable.to_string_lossy().into_owned();
    let mine: Vec<(u64, u64)> = frames
        .iter()
        .filter_map(|frame| parse_frame(frame, &executable))
        .collect();
    assert!(
        !mine.is_empty(),
        "no recorded frame was attributed to this test binary ({executable}); \
         frames were: {frames:#?}"
    );

    let image = modules
        .iter()
        .find(|module| module.path == executable)
        .expect("this binary is in its own module map");

    // Every rendered frame's second number must be what the map says it is, or
    // the two halves of the profile disagree with each other.
    for (address, file_address) in &mine {
        assert_eq!(
            *file_address,
            address - image.bias,
            "a frame rendered a file address that the module map contradicts"
        );
    }

    // ---- a function whose name we already know resolves to that name ----
    let known = allocate_vectors as *const () as u64;
    assert!(
        known >= image.start && known - image.start < image.size,
        "{known:#x} is outside this binary's recorded code region \
         ({:#x}+{:#x})",
        image.start,
        image.size
    );
    let known_in_file = known - image.bias;

    // `nm` reports the address a symbol has *in the file*, which is an
    // independent source of truth for the bias — independent because it comes
    // from the file on disk rather than from the same loader API that produced
    // the map. Without this, a platform where only `atos` is installed never
    // checks the bias at all: `atos` works from the load address and the
    // runtime address, and never looks at it.
    match std::process::Command::new("nm").arg(&executable).output() {
        Ok(output) if output.status.success() => {
            let listing = String::from_utf8_lossy(&output.stdout);
            match listing
                .lines()
                .filter(|line| line.contains("allocate_vectors"))
                .find_map(|line| u64::from_str_radix(line.split_whitespace().next()?, 16).ok())
            {
                Some(in_file) => assert_eq!(
                    known_in_file, in_file,
                    "the profile's bias puts allocate_vectors at {known_in_file:#x} \
                     in the file, but nm says it is at {in_file:#x}"
                ),
                // `nm` ran and named nothing useful, which is not the same as
                // `nm` being absent and is not a statement about the bias. A
                // Windows runner has an `nm` — Git for Windows ships binutils —
                // and an MSVC-linked Rust binary keeps its symbols in the PDB
                // rather than in the image, so the listing has no Rust function
                // in it to compare against. That is a fact about the object
                // format. This was an `expect`, and it fired the first time
                // Windows ever got far enough to run this test.
                //
                // The check is not weakened where it can be made: macOS and
                // both Linux targets name the function, so the comparison still
                // runs there, which is where the bias is a number a user pastes
                // into `addr2line`.
                None => eprintln!(
                    "skipping the nm cross-check: nm ran but does not name \
                     allocate_vectors in {executable}"
                ),
            }
        }
        _ => eprintln!("skipping the nm cross-check: no usable nm on PATH"),
    }

    // On Windows `bias` is the image base rather than the link-time base, so
    // `known_in_file` is a relative virtual address rather than the address the
    // file records — see `Module::bias`, which states the divergence. That is a
    // fact about the platform and not about this profile: the loader **rewrites**
    // `ImageBase` in the mapped optional header when it relocates an image, so
    // the link-time base is not obtainable from memory at all. Measured: ASLR
    // placed this binary at two different bases across two CI runs and the field
    // read from the mapped header equalled the load address both times.
    //
    // `llvm-symbolizer` takes such an address with `--relative-address`.
    // `addr2line` has no equivalent — it resolves against section VMAs, which
    // include the image base — so it is not asked here rather than being asked
    // a question it cannot answer. Closing this needs the link-time base out of
    // the file on disk, which is the same tier-3 requirement the macOS shared
    // cache has.
    let mut candidates = vec![
        // `atos` takes the runtime address and the image's load address.
        (
            "atos",
            vec![
                String::from("-o"),
                executable.clone(),
                String::from("-l"),
                format!("{:#x}", image.load),
                format!("{known:#x}"),
            ],
        ),
        ("llvm-symbolizer", {
            let mut arguments = vec![format!("--obj={executable}")];
            if cfg!(windows) {
                arguments.push(String::from("--relative-address"));
            }
            arguments.push(format!("{known_in_file:#x}"));
            arguments
        }),
    ];
    if !cfg!(windows) {
        candidates.push((
            "addr2line",
            vec![
                String::from("-f"),
                String::from("-C"),
                String::from("-e"),
                executable.clone(),
                format!("{known_in_file:#x}"),
            ],
        ));
    }

    let mut tools = 0;
    for (tool, arguments) in candidates {
        let Ok(output) = std::process::Command::new(tool).args(&arguments).output() else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let resolved = String::from_utf8_lossy(&output.stdout).into_owned();
        if resolved.trim().is_empty() {
            continue;
        }
        tools += 1;
        assert!(
            resolved.contains("allocate_vectors"),
            "{tool} resolved {known:#x} (file address {known_in_file:#x}) to:\n\
             {resolved}\nexpected the function that made the allocations"
        );
    }

    // Hard failure, not a skip: "offline symbolization works end to end" is an
    // M2 exit criterion, and a criterion that quietly stops being checked has
    // stopped being a criterion.
    //
    // The one exception is an environment where spawning a symbolizer is
    // impossible rather than merely unconfigured. `ci/windows-under-wine.sh`
    // runs this as a Windows binary under Wine, so every process it spawns must
    // be a Windows one and no symbolizer installed in the Linux container can
    // be reached. That is a property of the harness, which is why the harness
    // says so itself rather than this test trying to detect it.
    if std::env::var_os("HEAPSCOPE_NO_SUBPROCESS_TOOLS").is_some() {
        eprintln!("skipping the symbolizer check: the harness cannot spawn one");
        return;
    }
    assert!(
        tools > 0,
        "no symbolizer is installed, so offline symbolization was not verified"
    );
}

/// The parts of a module map entry this test uses.
struct MapEntry {
    path: String,
    load: u64,
    start: u64,
    size: u64,
    bias: u64,
}

fn parse_modules(profile: &json::Value) -> Vec<MapEntry> {
    let field = |module: &json::Value, name: &str| {
        module
            .get(name)
            .and_then(|value| value.as_u64())
            .unwrap_or_else(|| panic!("a module map entry has no `{name}`"))
    };
    profile
        .get("heapscope")
        .and_then(|section| section.get("modules"))
        .and_then(|modules| modules.as_array())
        .expect("the profile carries a module map")
        .iter()
        .map(|module| MapEntry {
            path: module
                .get("path")
                .and_then(|path| path.as_str())
                .expect("a module map entry has no `path`")
                .to_string(),
            load: field(module, "load"),
            start: field(module, "start"),
            size: field(module, "size"),
            bias: field(module, "bias"),
        })
        .collect()
}

/// Pulls the runtime address and file address out of a frame belonging to
/// `image`.
///
/// A frame is `0xADDR: name (image+0xFILEADDR)`, where `name` is `???` when the
/// running process could not supply one and an arbitrary demangled path when it
/// could. So the name is skipped rather than matched: read the address off the
/// front, the attribution off the back, and take the last ` (` between them,
/// because a Rust type in the middle can contain parentheses of its own.
fn parse_frame(frame: &str, image: &str) -> Option<(u64, u64)> {
    let (address, rest) = frame.split_once(": ")?;
    let (_name, attribution) = rest.strip_suffix(')')?.rsplit_once(" (")?;
    let (path, file_address) = attribution.rsplit_once("+0x")?;
    if path != image {
        return None;
    }
    Some((
        u64::from_str_radix(address.strip_prefix("0x")?, 16).ok()?,
        u64::from_str_radix(file_address, 16).ok()?,
    ))
}
