//! Regions in a real program, through a real allocator: the row for what no
//! region covered, and the reading that gets every region without a snapshot.
//!
//! The unit tests in `src/internals/engine.rs` and `src/stats.rs` drive an
//! engine directly, which is how each path that moves a row is shown to move
//! the right one. Nothing there goes through `#[global_allocator]`, `region`'s
//! own guard, `Vec` growth as the standard library actually does it, or the
//! files a profile is written to. This does.
//!
//! # Why one `#[test]`
//!
//! One engine per process, and `cargo test` runs tests concurrently: a second
//! test allocating while this one reads would land in these rows. See the
//! `stats` module documentation, which recommends this arrangement to users.

mod support;

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};

use heapscope::output::{OutsideRegions, Snapshot};
use heapscope::{region, RegionBreakdown, StatsError};
use support::native;

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// A `Vec` grown one push at a time, so that it reallocates as the standard
/// library really does it rather than as a test imagines it.
#[inline(never)]
fn grown(len: usize) -> Vec<u64> {
    let mut grown = Vec::new();
    for value in 0..len as u64 {
        grown.push(black_box(value));
    }
    black_box(grown)
}

/// The region rows plus the remainder, per additive column.
fn regions_plus(snapshot: &Snapshot) -> OutsideRegions {
    let mut sum = snapshot.outside_regions;
    for row in &snapshot.regions {
        sum.total_bytes += row.counts.total_bytes;
        sum.total_blocks += row.counts.total_blocks;
        sum.curr_bytes += row.counts.curr_bytes;
        sum.curr_blocks += row.counts.curr_blocks;
    }
    sum
}

#[test]
#[cfg_attr(
    miri,
    ignore = "needs a real backtrace, and Miri cannot execute inline assembly"
)]
fn the_regions_and_what_no_region_covered_add_up_to_the_run() {
    // ---- with nothing recording, the reading refuses rather than being empty ----
    assert_eq!(
        RegionBreakdown::get().expect_err("an unprofiled process has no regions"),
        StatsError::NotRecording
    );

    let profiler = heapscope::Profiler::builder()
        .no_output()
        .build()
        .expect("the profiler should start");

    // Born outside every region and grown inside one: the growth is the
    // block's, so it stays outside.
    let mut born_outside = grown(4);
    // Born in a region and dropped outside it, and born in a region and grown
    // outside it: both come back to the region.
    let (parsed, mut kept) = {
        let parsing = region("parsing");
        assert!(parsing.is_open());
        let parsed = grown(64);
        let outer_block = grown(32);
        let kept = {
            let _lexing = region("parsing/lexing");
            born_outside.extend(0..256);
            // A block of the outer region, freed inside the inner one.
            drop(outer_block);
            grown(16)
        };
        (parsed, kept)
    };
    drop(parsed);
    kept.extend(0..1_024);

    // Another thread while this one is inside a region: what it allocates is
    // outside, because a region is the calling thread's alone, and a region's
    // block that it drops still comes off the region.
    //
    // The thread is spawned before the region opens and told to start by an
    // atomic, because spawning allocates on the *spawning* thread — that is
    // correctly charged to whatever region it is in, and would make "this
    // region recorded nothing" false for a reason that is not the one tested.
    let handed_over = {
        let _region = region("handing over");
        grown(128)
    };
    let go = AtomicBool::new(false);
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !go.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            drop(handed_over);
            black_box(grown(512));
            done.store(true, Ordering::Release);
        });
        let _region = region("waiting");
        go.store(true, Ordering::Release);
        while !done.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    });

    // ---- the cheap reading ----
    let breakdown = RegionBreakdown::get().expect("a running heap run has regions");
    assert_eq!(breakdown.mode, heapscope::Mode::Heap);
    let names: Vec<_> = breakdown
        .regions
        .iter()
        .map(|row| row.name.as_deref())
        .collect();
    assert_eq!(
        names,
        [
            Some("parsing"),
            Some("parsing/lexing"),
            Some("handing over"),
            Some("waiting")
        ],
        "the rows are not the regions the program entered, in the order it \
         entered them"
    );
    let handing_over = breakdown.region("handing over").expect("entered");
    assert!(handing_over.counts.total_bytes > 0);
    assert_eq!(
        handing_over.counts.curr_bytes, 0,
        "a block freed on another thread did not come off the region it was \
         born in"
    );
    assert_eq!(
        breakdown
            .region("waiting")
            .expect("entered")
            .counts
            .total_blocks,
        0,
        "another thread's allocation landed in a region only this thread was in"
    );
    assert!(
        breakdown
            .region("parsing/lexing")
            .expect("entered")
            .counts
            .curr_bytes
            >= (1_024 + 16) * 8,
        "a block born in a region and grown outside it left the region"
    );
    assert!(
        breakdown.outside.curr_bytes >= (4 + 256) * 8,
        "a block born outside every region and grown inside one left the \
         remainder"
    );

    // Taking the reading inside a region does not charge the region for it.
    {
        let _region = region("reading");
        for _ in 0..2 {
            let inside = RegionBreakdown::get().expect("regions");
            assert_eq!(
                inside
                    .region("reading")
                    .expect("entered")
                    .counts
                    .total_blocks,
                0,
                "the reading's own allocations were charged to the region it \
                 was taken in"
            );
        }
    }

    // ---- the snapshot, which reads the totals in the same window ----
    let snapshot = Snapshot::capture();
    assert!(snapshot.exact);
    assert_eq!(snapshot.rows_dropped, 0);
    let sum = regions_plus(&snapshot);
    assert_eq!(sum.total_bytes, snapshot.stats.total_bytes);
    assert_eq!(sum.total_blocks, snapshot.stats.total_blocks);
    assert_eq!(sum.curr_bytes, snapshot.stats.curr_bytes);
    assert_eq!(sum.curr_blocks, snapshot.stats.curr_blocks);
    assert!(
        snapshot.outside_regions.total_bytes < snapshot.stats.total_bytes,
        "the regions recorded nothing, so the sum above proved nothing"
    );

    // ---- and in the files ----
    let mut native_file = Vec::new();
    snapshot
        .write_native(&mut native_file)
        .expect("writing to a Vec cannot fail");
    native::assert_valid(&String::from_utf8(native_file).expect("UTF-8"));

    let mut summary = Vec::new();
    snapshot
        .write_text_summary(&mut summary, 8)
        .expect("writing to a Vec cannot fail");
    let summary = String::from_utf8(summary).expect("UTF-8");
    let section = summary
        .split("heapscope regions")
        .nth(1)
        .unwrap_or_else(|| panic!("no regions section:\n{summary}"));
    assert!(
        section
            .lines()
            .skip(1)
            .take_while(|line| !line.is_empty())
            .last()
            .is_some_and(|line| line.trim_start().starts_with("(no region)")),
        "the regions section does not end with what no region covered:\n{summary}"
    );

    drop(born_outside);
    drop(kept);
    drop(profiler);
}
