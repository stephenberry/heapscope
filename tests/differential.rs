//! Differential testing against a deliberately slow, obviously correct model.
//!
//! PLAN.md section 8.2 calls for a `ReferenceTracker`: one `BTreeMap`, one lock,
//! an eager sweep over every program point on each new peak, no arena, no
//! sharding, no epochs. It is everything the real engine is not, which is the
//! point — it is short enough to read and agree is correct, and it computes the
//! same numbers.
//!
//! The real engine's cleverness is concentrated in one place: the lazy-epoch
//! algorithm, which replaces the reference's `O(#program points)` sweep on every
//! peak with `O(1)` amortised work. That optimisation is invisible in the output
//! if it is right, and produces subtly wrong at-peak values if it is not — the
//! kind of bug that survives code review and unit tests. Comparing every counter
//! against an eager implementation over thousands of generated traces is the
//! check that actually catches it.
//!
//! # What is compared exactly, and what is not
//!
//! **Single-threaded traces are compared field by field.** There is one serial
//! order, so every number is determined and any disagreement is a bug.
//!
//! **Concurrent traces are checked two other ways.** The peak gate makes the
//! result equivalent to *some* serial order, but not to a *known* one, so there
//! is no specific reference run to compare against.
//!
//! The first way is invariants: what must hold regardless of interleaving is
//! that the parts sum to the whole, in particular that per-point at-peak bytes
//! and blocks sum to exactly the global peak, which is the property the gate
//! exists to provide.
//!
//! The second way exists because the first is not enough, which M7 chunk J
//! established by deleting the gate's escalation check and watching the whole
//! suite pass. Summation is self-consistency: a run that recorded the wrong
//! instant as its peak still agrees with itself about it. So the third test
//! gives the run a peak whose value is fixed by the *shape* of the workload
//! rather than by the schedule, and asks whether that is the one recorded.
//!
//! # Restarts
//!
//! `Profiler::reset` is in the generated traces too, because it is the one
//! operation that moves every at-peak column at once without the heap moving:
//! the restart is a peak at whatever level the heap is at, which the engine
//! honours through the same lazy epoch it uses for every other peak. The model
//! does it eagerly, by hand, which is what makes the two worth comparing.

use std::collections::BTreeMap;

use heapscope::internals::clock::TimeSource;
use heapscope::internals::engine::Engine;
use heapscope::internals::pp::Counters;
use heapscope::internals::shape::{Realloc, Shape};
use proptest::prelude::*;

/// The obviously-correct model.
///
/// Deliberately naive: a `BTreeMap` keyed by the frame vector, a full sweep on
/// every peak, and no concurrency at all.
#[derive(Default)]
struct ReferenceTracker {
    /// Live blocks: address to (frames, size, birth). A birth of `None` is a
    /// block carried across a restart, which the window did not allocate and
    /// so gives no lifetime to.
    live: BTreeMap<usize, (Vec<usize>, usize, Option<u64>)>,
    /// Per-program-point counters, keyed by the frames themselves.
    points: BTreeMap<Vec<usize>, Counters>,

    curr_bytes: u64,
    curr_blocks: u64,
    max_bytes: u64,
    max_blocks: u64,
    total_bytes: u64,
    total_blocks: u64,
    /// When the peak was reached, on the clock below.
    time_at_max: u64,
    /// New peaks since the run started or was last restarted.
    peaks: u64,

    /// Mirrors the engine's clock exactly: incremented once per allocation and
    /// once per reallocation, and read without incrementing on a free.
    events: u64,
}

impl ReferenceTracker {
    fn tick(&mut self) -> u64 {
        self.events += 1;
        self.events
    }

    fn point(&mut self, frames: &[usize]) -> &mut Counters {
        self.points.entry(frames.to_vec()).or_default()
    }

    /// The eager sweep: on a new peak, every program point's current values are
    /// copied into its at-peak fields, right now.
    ///
    /// `>=` rather than `>`, so that among several equal peaks the latest is the
    /// one recorded. Valgrind's `dh_main.c` is explicit about this, and the lazy
    /// scheme in the engine only matches an eager one that uses `>=`.
    fn note_peak(&mut self) {
        if self.curr_bytes >= self.max_bytes {
            self.max_bytes = self.curr_bytes;
            self.max_blocks = self.curr_blocks;
            self.time_at_max = self.events;
            self.peaks += 1;
            for counters in self.points.values_mut() {
                counters.at_gmax_bytes = counters.curr_bytes;
                counters.at_gmax_blocks = counters.curr_blocks;
            }
        }
    }

    /// The restart, as its documentation states it: every total back to zero,
    /// everything live kept, and every peak started again from what is live.
    ///
    /// The peak is set here rather than by `note_peak`, because a restart is
    /// not a new peak by the `>=` rule: live bytes have usually fallen since
    /// the old maximum, and the restart lowers the maximum to meet them.
    fn reset(&mut self) {
        self.total_bytes = 0;
        self.total_blocks = 0;
        self.max_bytes = self.curr_bytes;
        self.max_blocks = self.curr_blocks;
        self.time_at_max = self.events;
        self.peaks = 0;
        for counters in self.points.values_mut() {
            counters.total_bytes = 0;
            counters.total_blocks = 0;
            counters.total_lifetime = 0;
            counters.max_bytes = counters.curr_bytes;
            counters.max_blocks = counters.curr_blocks;
            counters.at_gmax_bytes = counters.curr_bytes;
            counters.at_gmax_blocks = counters.curr_blocks;
        }
        for (_, _, birth) in self.live.values_mut() {
            *birth = None;
        }
    }

    /// How long a block born at `birth` has lived, ending at `end`.
    fn lifetime(birth: Option<u64>, end: u64) -> u64 {
        birth.map_or(0, |birth| end - birth)
    }

    fn alloc(&mut self, address: usize, size: usize, frames: &[usize]) {
        let birth = self.tick();
        if self.live.contains_key(&address) {
            // The generator never produces this, but a real allocator cannot
            // either: an address cannot be handed out twice without an
            // intervening free.
            return;
        }
        self.live
            .insert(address, (frames.to_vec(), size, Some(birth)));

        self.curr_bytes += size as u64;
        self.curr_blocks += 1;
        self.total_bytes += size as u64;
        self.total_blocks += 1;

        let counters = self.point(frames);
        counters.curr_bytes += size as u64;
        counters.curr_blocks += 1;
        counters.total_bytes += size as u64;
        counters.total_blocks += 1;
        counters.max_bytes = counters.max_bytes.max(counters.curr_bytes);
        counters.max_blocks = counters.max_blocks.max(counters.curr_blocks);

        self.note_peak();
    }

    /// An event: cumulative totals only.
    ///
    /// No `note_peak`, and that is the property under test rather than a
    /// shortcut. Nothing became live, so live bytes did not move, so no peak can
    /// have occurred — and the engine's peak gate must reach the same
    /// conclusion, including when live bytes already sit exactly at the maximum,
    /// which is where its `>=` rule would otherwise record a new equal peak.
    fn event(&mut self, weight: u64, frames: &[usize]) {
        self.tick();
        self.total_bytes += weight;
        self.total_blocks += 1;

        let counters = self.point(frames);
        counters.total_bytes += weight;
        counters.total_blocks += 1;
    }

    fn free(&mut self, address: usize) {
        let Some((frames, size, birth)) = self.live.remove(&address) else {
            return;
        };
        let lifetime = Self::lifetime(birth, self.events);

        self.curr_bytes -= size as u64;
        self.curr_blocks -= 1;

        let counters = self.point(&frames);
        counters.curr_bytes -= size as u64;
        counters.curr_blocks -= 1;
        counters.total_lifetime += lifetime;
        // A free cannot raise a maximum, so no `max_*` update and no peak check.
    }

    fn realloc(&mut self, old: usize, new: usize, new_size: usize, frames: &[usize]) {
        let Some((old_frames, old_size, old_birth)) = self.live.remove(&old) else {
            self.alloc(new, new_size, frames);
            return;
        };
        let birth = self.tick();
        // The old block's life ends here. Its lifetime has to be recorded
        // because the reallocation also counts as a block, and counting a block
        // without its lifetime deflates the average-lifetime column at exactly
        // the sites a reader looks at first.
        let old_lifetime = Self::lifetime(old_birth, birth);
        self.live
            .insert(new, (old_frames.clone(), new_size, Some(birth)));

        self.curr_bytes = self.curr_bytes - old_size as u64 + new_size as u64;
        self.total_bytes += new_size as u64;
        self.total_blocks += 1;

        // Attributed to the point that made the *original* allocation.
        let counters = self.point(&old_frames);
        counters.curr_bytes = counters.curr_bytes - old_size as u64 + new_size as u64;
        counters.total_bytes += new_size as u64;
        counters.total_blocks += 1;
        counters.total_lifetime += old_lifetime;
        counters.max_bytes = counters.max_bytes.max(counters.curr_bytes);

        self.note_peak();
    }

    /// The end-of-run state, keyed the way the engine reports it.
    ///
    /// A point that neither allocated in the window nor held anything in it is
    /// left out, as the engine leaves it out. One that only holds what it
    /// carried across a restart is kept, because its live and at-peak columns
    /// are part of the run's.
    fn finish(self) -> BTreeMap<Vec<usize>, Counters> {
        self.points
            .into_iter()
            .filter(|(_, counters)| counters.total_blocks != 0 || counters.max_blocks != 0)
            .collect()
    }
}

/// One operation in a generated trace.
#[derive(Clone, Debug)]
enum Op {
    Alloc {
        slot: usize,
        size: usize,
        site: usize,
    },
    Free {
        slot: usize,
    },
    Realloc {
        slot: usize,
        new_size: usize,
        site: usize,
    },
    /// An event the program reported, rather than an allocation the shim saw.
    ///
    /// Mixed into the same traces as the heap operations, which the public API
    /// never does — a run counts one kind of thing — but the engine is what is
    /// under test here, and interleaving them is what makes the property
    /// checkable: an event must move the cumulative totals and *nothing else*,
    /// including in the middle of a growing phase where every allocation around
    /// it is setting a new peak.
    Event {
        weight: u64,
        site: usize,
    },
    /// `Profiler::reset`.
    Reset,
}

/// Slots stand in for addresses, so the generator can produce well-formed
/// traces — no double frees, no frees of never-allocated blocks — without
/// modelling an allocator.
const SLOTS: usize = 64;
const SITES: usize = 12;

/// Records an allocation the way the shim does, with the reentrancy guard held
/// across the call.
///
/// The concurrent traces below run in worker threads, and [`Guard`] is `!Send`,
/// so each has to take its own. Taken per call rather than around a worker's
/// loop for the reason spelled out at the ad hoc site: a guard held across the
/// whole worker refuses every recursive entry the rest of it might legitimately
/// make.
fn guarded_alloc(engine: &Engine, address: usize, shape: Shape, frames: &[usize]) {
    let guard =
        heapscope::internals::guard::enter().expect("this thread is not inside the profiler");
    engine.record_alloc(&guard, address, shape, frames);
}

/// [`guarded_alloc`], for the other half of a reallocation.
fn guarded_realloc(
    engine: &Engine,
    taken: Option<heapscope::internals::live::LiveBlock>,
    realloc: Realloc,
    frames: &[usize],
) {
    let guard =
        heapscope::internals::guard::enter().expect("this thread is not inside the profiler");
    engine.record_realloc_taken(&guard, taken, realloc, frames);
}

/// Allocation sizes, weighted heavily toward a small recurring set.
///
/// This is not cosmetic. With sizes drawn uniformly from a wide range, the total
/// live bytes almost never returns to a value it held before, so *equal peaks*
/// essentially never occur — and equal peaks are precisely where `>=` and `>`
/// differ in the epoch bump. Measured: with uniform sizes, 512 cases of up to
/// 400 operations failed to catch a deliberately reintroduced `>` bug, which
/// only the hand-written cases below detected. Real programs allocate the same
/// handful of sizes over and over, so the recurring set is also the more
/// faithful generator.
fn size() -> impl Strategy<Value = usize> {
    prop_oneof![
        6 => prop::sample::select(vec![16usize, 32, 64, 128, 256, 1024]),
        1 => 1usize..4096,
    ]
}

fn operation() -> impl Strategy<Value = Op> {
    prop_oneof![
        30 => (0..SLOTS, size(), 0..SITES)
            .prop_map(|(slot, size, site)| Op::Alloc { slot, size, site }),
        20 => (0..SLOTS).prop_map(|slot| Op::Free { slot }),
        10 => (0..SLOTS, size(), 0..SITES)
            .prop_map(|(slot, new_size, site)| Op::Realloc { slot, new_size, site }),
        10 => (0u64..4096, 0..SITES).prop_map(|(weight, site)| Op::Event { weight, site }),
        // About one operation in seventy: rare enough that a window after a
        // restart usually grows, shrinks and peaks again before the next, and
        // common enough that a trace of a few hundred operations restarts a
        // few times, carried blocks and all.
        1 => Just(Op::Reset),
    ]
}

/// Call stacks that share prefixes, as real ones do.
fn frames_for(site: usize) -> Vec<usize> {
    let depth = 1 + site % 4;
    (0..depth)
        .map(|d| 0x40_0000 + (site >> d) * 8 + d)
        .collect()
}

/// Compares every counter, naming the first field that disagrees.
fn compare(ops: &[Op]) -> Result<(), String> {
    let engine = Engine::with_limits(1 << 14, 1 << 16);
    assert!(engine.start(TimeSource::Events, || {}));
    let mut reference = ReferenceTracker::default();

    let mut occupied: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    let mut next_address = 0x1_0000_0000usize;
    // `record_event` requires proof that the caller holds the reentrancy guard,
    // because it reaches the peak gate. Taken once for the whole trace: this
    // thread is not inside the shim, so no recursive entry can be refused.
    let guard =
        heapscope::internals::guard::enter().expect("this thread is not inside the profiler");

    for op in ops {
        match *op {
            Op::Alloc { slot, size, site } => {
                if occupied.contains_key(&slot) {
                    continue;
                }
                let address = next_address;
                next_address += 4096;
                let frames = frames_for(site);
                engine.record_alloc(&guard, address, Shape::of(size), &frames);
                reference.alloc(address, size, &frames);
                occupied.insert(slot, (address, size));
            }
            Op::Free { slot } => {
                let Some((address, size)) = occupied.remove(&slot) else {
                    continue;
                };
                engine.record_free(address, size);
                reference.free(address);
            }
            Op::Realloc {
                slot,
                new_size,
                site,
            } => {
                let Some((address, size)) = occupied.remove(&slot) else {
                    continue;
                };
                let new_address = next_address;
                next_address += 4096;
                let frames = frames_for(site);
                let taken = engine.live_blocks().remove(address);
                engine.record_realloc_taken(
                    &guard,
                    taken,
                    Realloc {
                        old_address: address,
                        old_size: size,
                        new_address,
                        new: Shape::of(new_size),
                    },
                    &frames,
                );
                reference.realloc(address, new_address, new_size, &frames);
                occupied.insert(slot, (new_address, new_size));
            }
            Op::Event { weight, site } => {
                let frames = frames_for(site);
                engine.record_event(&guard, weight, &frames);
                reference.event(weight, &frames);
            }
            Op::Reset => {
                engine
                    .reset(&guard)
                    .map_err(|error| format!("the engine refused a restart: {error}"))?;
                reference.reset();
            }
        }
    }

    let mut actual: BTreeMap<Vec<usize>, Counters> = BTreeMap::new();
    let flush = engine.flush_and_visit(
        Engine::FLUSH_TIMEOUT,
        |_id, frames, counters| {
            actual.insert(frames.to_vec(), *counters);
        },
        |_| {},
        |_| {},
    );
    assert!(flush.exclusive, "the flush could not reach a quiet point");
    let stats = flush.stats;
    if stats.curr_bytes != reference.curr_bytes {
        return Err(format!(
            "live bytes: engine {} vs reference {}",
            stats.curr_bytes, reference.curr_bytes
        ));
    }
    if stats.curr_blocks != reference.curr_blocks {
        return Err(format!(
            "live blocks: engine {} vs reference {}",
            stats.curr_blocks, reference.curr_blocks
        ));
    }
    if stats.total_bytes != reference.total_bytes {
        return Err(format!(
            "cumulative bytes: engine {} vs reference {}",
            stats.total_bytes, reference.total_bytes
        ));
    }
    if stats.total_blocks != reference.total_blocks {
        return Err(format!(
            "cumulative blocks: engine {} vs reference {}",
            stats.total_blocks, reference.total_blocks
        ));
    }
    if stats.max_bytes != reference.max_bytes {
        return Err(format!(
            "peak bytes: engine {} vs reference {}",
            stats.max_bytes, reference.max_bytes
        ));
    }
    if stats.max_blocks != reference.max_blocks {
        return Err(format!(
            "blocks at peak: engine {} vs reference {}",
            stats.max_blocks, reference.max_blocks
        ));
    }
    if stats.time_at_max != reference.time_at_max {
        return Err(format!(
            "time of the peak: engine {} vs reference {}",
            stats.time_at_max, reference.time_at_max
        ));
    }
    // The epoch counts peaks over the whole run, and the restart's own is not
    // one the window reached; this is the figure the profile reports.
    let since = engine.last_reset().map_or(0, |reset| reset.epoch);
    if stats.epoch - since != reference.peaks {
        return Err(format!(
            "peaks in the window: engine {} vs reference {}",
            stats.epoch - since,
            reference.peaks
        ));
    }

    let expected = reference.finish();
    for (frames, want) in &expected {
        let Some(got) = actual.get(frames) else {
            return Err(format!("engine lost program point {frames:x?}"));
        };
        let fields: [(&str, u64, u64); 9] = [
            ("total_bytes", got.total_bytes, want.total_bytes),
            ("total_blocks", got.total_blocks, want.total_blocks),
            ("total_lifetime", got.total_lifetime, want.total_lifetime),
            ("curr_bytes", got.curr_bytes, want.curr_bytes),
            ("curr_blocks", got.curr_blocks, want.curr_blocks),
            ("max_bytes", got.max_bytes, want.max_bytes),
            ("max_blocks", got.max_blocks, want.max_blocks),
            ("at_gmax_bytes", got.at_gmax_bytes, want.at_gmax_bytes),
            ("at_gmax_blocks", got.at_gmax_blocks, want.at_gmax_blocks),
        ];
        for (name, got_value, want_value) in fields {
            if got_value != want_value {
                return Err(format!(
                    "program point {frames:x?}: {name} is {got_value}, reference says {want_value}"
                ));
            }
        }
    }
    for frames in actual.keys() {
        if !expected.contains_key(frames) {
            return Err(format!("engine invented program point {frames:x?}"));
        }
    }

    Ok(())
}

proptest! {
    // Shrinking is what makes a failing 400-operation trace debuggable, which is
    // the whole reason PLAN.md section 8.4 asks for property tests rather than a
    // hand-rolled random loop.
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 512 },
        max_shrink_iters: 4096,
        // Proptest saves failing seeds to a file next to the test, which means
        // resolving the current directory -- and Miri's filesystem isolation
        // makes that a hard abort rather than an error. Persistence is worth
        // keeping natively, where it turns a rare failure into a permanently
        // reproducible one, so it is dropped only under Miri.
        failure_persistence: if cfg!(miri) {
            None
        } else {
            Some(Box::new(
                proptest::test_runner::FileFailurePersistence::default(),
            ))
        },
        ..ProptestConfig::default()
    })]

    /// Every counter, on every program point, must match the eager model.
    #[test]
    fn the_engine_agrees_with_the_reference_tracker(
        ops in prop::collection::vec(operation(), 0..400)
    ) {
        if let Err(difference) = compare(&ops) {
            prop_assert!(false, "{difference}");
        }
    }
}

/// The specific hazards the lazy-epoch scheme has to survive, spelled out
/// rather than left to the generator to stumble across.
#[test]
fn hand_written_epoch_hazards_match_the_reference() {
    let cases: Vec<(&str, Vec<Op>)> = vec![
        (
            "a point decremented after the peak",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 1000,
                    site: 0,
                },
                Op::Alloc {
                    slot: 1,
                    size: 500,
                    site: 1,
                },
                Op::Free { slot: 0 },
            ],
        ),
        (
            "a point whose only post-peak event is a free",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 1000,
                    site: 0,
                },
                Op::Alloc {
                    slot: 1,
                    size: 1000,
                    site: 1,
                },
                Op::Free { slot: 1 },
                Op::Free { slot: 0 },
            ],
        ),
        (
            "several peaks between two touches of one point",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 100,
                    site: 0,
                },
                Op::Alloc {
                    slot: 1,
                    size: 100,
                    site: 1,
                },
                Op::Alloc {
                    slot: 2,
                    size: 100,
                    site: 1,
                },
                Op::Alloc {
                    slot: 3,
                    size: 100,
                    site: 1,
                },
                Op::Free { slot: 0 },
            ],
        ),
        (
            "equal peaks, where the latest must win",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 100,
                    site: 0,
                },
                Op::Free { slot: 0 },
                Op::Alloc {
                    slot: 1,
                    size: 100,
                    site: 1,
                },
            ],
        ),
        (
            "a point never touched again after the peak, caught only by the flush",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 700,
                    site: 0,
                },
                Op::Alloc {
                    slot: 1,
                    size: 100,
                    site: 1,
                },
                Op::Free { slot: 1 },
            ],
        ),
        (
            "a shrinking realloc, which is a descent from the peak",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 4000,
                    site: 0,
                },
                Op::Realloc {
                    slot: 0,
                    new_size: 100,
                    site: 1,
                },
                Op::Alloc {
                    slot: 1,
                    size: 50,
                    site: 2,
                },
            ],
        ),
        (
            "a growing realloc that sets a new peak",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 100,
                    site: 0,
                },
                Op::Realloc {
                    slot: 0,
                    new_size: 8000,
                    site: 1,
                },
            ],
        ),
        (
            "a restart below the old peak, which lowers the peak to what is live",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 4000,
                    site: 0,
                },
                Op::Alloc {
                    slot: 1,
                    size: 100,
                    site: 1,
                },
                Op::Free { slot: 0 },
                Op::Reset,
            ],
        ),
        (
            "a point untouched after a restart, whose at-peak figures only the \
             restart's epoch can refresh",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 4000,
                    site: 0,
                },
                Op::Alloc {
                    slot: 1,
                    size: 100,
                    site: 1,
                },
                Op::Free { slot: 1 },
                Op::Alloc {
                    slot: 2,
                    size: 50,
                    site: 1,
                },
                Op::Reset,
                Op::Alloc {
                    slot: 3,
                    size: 50,
                    site: 2,
                },
            ],
        ),
        (
            "a carried block freed after the restart, which has no lifetime",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 256,
                    site: 0,
                },
                Op::Alloc {
                    slot: 1,
                    size: 64,
                    site: 0,
                },
                Op::Reset,
                Op::Alloc {
                    slot: 2,
                    size: 64,
                    site: 1,
                },
                Op::Free { slot: 0 },
                Op::Free { slot: 2 },
            ],
        ),
        (
            "a carried block reallocated after the restart, whose new block is \
             the window's",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 256,
                    site: 0,
                },
                Op::Reset,
                Op::Realloc {
                    slot: 0,
                    new_size: 1024,
                    site: 1,
                },
                Op::Alloc {
                    slot: 1,
                    size: 16,
                    site: 2,
                },
                Op::Free { slot: 0 },
            ],
        ),
        (
            "a point whose carried blocks are all freed after the restart",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 256,
                    site: 0,
                },
                Op::Reset,
                Op::Free { slot: 0 },
            ],
        ),
        (
            "an equal peak right after a restart, where the latest must win",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 128,
                    site: 0,
                },
                Op::Reset,
                Op::Free { slot: 0 },
                Op::Alloc {
                    slot: 1,
                    size: 128,
                    site: 1,
                },
            ],
        ),
        (
            "two restarts with nothing between them",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 512,
                    site: 0,
                },
                Op::Reset,
                Op::Reset,
                Op::Alloc {
                    slot: 1,
                    size: 32,
                    site: 1,
                },
            ],
        ),
        (
            "an event straddled by restarts, which moves totals and no peak",
            vec![
                Op::Alloc {
                    slot: 0,
                    size: 512,
                    site: 0,
                },
                Op::Reset,
                Op::Event {
                    weight: 900,
                    site: 1,
                },
                Op::Reset,
                Op::Event { weight: 7, site: 2 },
            ],
        ),
    ];

    for (name, ops) in cases {
        if let Err(difference) = compare(&ops) {
            panic!("{name}: {difference}");
        }
    }
}

/// Concurrency has no single reference run to compare against, so the check is
/// that the parts sum to the whole — above all that per-point at-peak bytes sum
/// to exactly the global peak, which is what the peak gate exists to provide.
#[test]
fn concurrent_traces_preserve_the_summation_invariants() {
    #[cfg(miri)]
    const ROUNDS: usize = 20;
    #[cfg(not(miri))]
    const ROUNDS: usize = 3_000;
    const THREADS: usize = 8;

    for attempt in 0..if cfg!(miri) { 1 } else { 10 } {
        let engine = Engine::with_limits(1 << 14, 1 << 18);
        assert!(engine.start(TimeSource::Events, || {}));

        std::thread::scope(|s| {
            for t in 0..THREADS {
                let engine = &engine;
                s.spawn(move || {
                    let base = 0x2_0000_0000usize + t * 0x1000_0000;
                    let mut live = Vec::new();
                    for i in 0..ROUNDS {
                        let address = base + i * 128;
                        let size = 32 + (i * 7 + t) % 512;
                        guarded_alloc(
                            engine,
                            address,
                            Shape::of(size),
                            &frames_for((i + t) % SITES),
                        );
                        live.push((address, size));

                        // A mix of immediate and deferred frees, so the heap
                        // both grows and shrinks and the peak moves around.
                        if i % 3 == 0 {
                            if let Some((address, size)) = live.pop() {
                                engine.record_free(address, size);
                            }
                        }
                        if i % 11 == 0 && !live.is_empty() {
                            let (address, size) = live.remove(0);
                            engine.record_free(address, size);
                        }
                    }
                });
            }
        });

        let mut summed_total = 0u64;
        let mut summed_total_blocks = 0u64;
        let mut summed_curr = 0u64;
        let mut summed_at_peak = 0u64;
        let mut summed_at_peak_blocks = 0u64;
        let mut summed_blocks = 0u64;
        let flush = engine.flush_and_visit(
            Engine::FLUSH_TIMEOUT,
            |_id, _frames, counters| {
                summed_total += counters.total_bytes;
                summed_total_blocks += counters.total_blocks;
                summed_curr += counters.curr_bytes;
                summed_at_peak += counters.at_gmax_bytes;
                summed_at_peak_blocks += counters.at_gmax_blocks;
                summed_blocks += counters.curr_blocks;
            },
            |_| {},
            |_| {},
        );
        assert!(flush.exclusive);
        let stats = flush.stats;
        assert_eq!(
            summed_total, stats.total_bytes,
            "attempt {attempt}: cumulative bytes drifted"
        );
        assert_eq!(
            summed_total_blocks, stats.total_blocks,
            "attempt {attempt}: cumulative blocks drifted"
        );
        assert_eq!(
            summed_curr, stats.curr_bytes,
            "attempt {attempt}: live bytes drifted"
        );
        assert_eq!(
            summed_blocks, stats.curr_blocks,
            "attempt {attempt}: live blocks drifted"
        );
        assert_eq!(
            summed_at_peak, stats.max_bytes,
            "attempt {attempt}: per-point at-peak bytes did not sum to the global peak"
        );
        // The `gbk` half of PLAN.md section 12's third bullet, which was
        // unchecked here until M7 chunk J. Every column named in that bullet has
        // a blocks counter beside its bytes counter, and the two are updated by
        // separate lines: the epoch refresh in `internals::pp` sets both, and a
        // defect that reached only the blocks half passed this test.
        assert_eq!(
            summed_at_peak_blocks, stats.max_blocks,
            "attempt {attempt}: per-point at-peak blocks did not sum to the blocks at the peak"
        );
        assert!(
            stats.max_bytes >= stats.curr_bytes,
            "attempt {attempt}: the peak is below the final live bytes"
        );
    }
}

/// Whether every row and point of `engine` sums to its run, read at a quiet
/// point, naming the first figure that does not.
///
/// What a restart that raced a recording thread breaks for good: the restart
/// zeroes the run's totals and each point's and row's, and a thread caught
/// between its update of one and of the other leaves them apart, by an amount
/// nothing after it ever takes back out.
fn sums_agree(engine: &Engine) -> Result<heapscope::internals::engine::GlobalStats, String> {
    let mut points = Counters::default();
    let (mut thread_bytes, mut thread_blocks, mut thread_live) = (0u64, 0u64, 0u64);
    let mut over_peak = 0usize;
    let flush = engine.flush_and_visit(
        Engine::FLUSH_TIMEOUT,
        |_id, _frames, counters| {
            points.total_bytes += counters.total_bytes;
            points.total_blocks += counters.total_blocks;
            points.curr_bytes += counters.curr_bytes;
            points.curr_blocks += counters.curr_blocks;
            points.at_gmax_bytes += counters.at_gmax_bytes;
            points.at_gmax_blocks += counters.at_gmax_blocks;
            if counters.curr_bytes > counters.max_bytes {
                over_peak += 1;
            }
        },
        |thread| {
            thread_bytes += thread.counts.total_bytes;
            thread_blocks += thread.counts.total_blocks;
            thread_live += thread.counts.curr_bytes;
        },
        |_| {},
    );
    if !flush.exclusive {
        return Err(String::from("the flush could not reach a quiet point"));
    }
    let stats = flush.stats;
    let figures = [
        ("points' bytes", points.total_bytes, stats.total_bytes),
        ("points' blocks", points.total_blocks, stats.total_blocks),
        ("points' live bytes", points.curr_bytes, stats.curr_bytes),
        ("points' live blocks", points.curr_blocks, stats.curr_blocks),
        (
            "points' bytes at the peak",
            points.at_gmax_bytes,
            stats.max_bytes,
        ),
        (
            "points' blocks at the peak",
            points.at_gmax_blocks,
            stats.max_blocks,
        ),
        ("threads' bytes", thread_bytes, stats.total_bytes),
        ("threads' blocks", thread_blocks, stats.total_blocks),
        ("threads' live bytes", thread_live, stats.curr_bytes),
    ];
    for (what, parts, whole) in figures {
        if parts != whole {
            return Err(format!("{what} sum to {parts}, and the run says {whole}"));
        }
    }
    if over_peak > 0 {
        return Err(format!(
            "{over_peak} point(s) hold more than their own peak"
        ));
    }
    if stats.max_bytes < stats.curr_bytes {
        return Err(String::from("the peak is below what is live"));
    }
    Ok(stats)
}

/// A restart lands wherever another thread happens to be, so it has to leave
/// the counters as coherent as it found them whatever was in flight.
///
/// This is the test the restart's locking exists for. Workers allocate and free
/// for real, on the shared path and the exclusive one, while another thread
/// restarts the counts over and over, and checks after every restart that the
/// parts still sum to the whole ([`sums_agree`]). Nothing else about the result
/// is fixed by the schedule, and what is checked at the end is what must hold
/// under every schedule: the sums again, the live figures equal to what the
/// workers still hold (which a restart that touched a live counter, or a free
/// of a carried block that went negative, would break), nothing poisoned, and
/// every restart accepted.
///
/// # Why after every restart, and why the workers outlive them
///
/// The first version of this stopped the workers and then reset once more, so
/// the one window it checked began at a quiet point and ended with nothing in
/// it: every total was zero, the epoch had not moved since the restart, and
/// every sum compared zero with zero. A restart that skipped the peak gate
/// passed it. Each restart wipes the totals the one before it got wrong, so a
/// check at the end sees only the last; checking after each one gives every
/// restart its own chance to be caught. And the workers run until the
/// restarting thread is done and then for a tail more, so every restart lands
/// on running threads and the last window has allocations, frees and peaks of
/// its own.
///
/// **Measured:** with the peak gate taken out of `Engine::reset`, this fails
/// in 20 runs of 20, where checking only at the end failed 2 of 20.
#[test]
fn restarts_racing_with_allocating_threads_keep_the_counters_coherent() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;

    #[cfg(miri)]
    const RESETS: usize = 3;
    #[cfg(not(miri))]
    const RESETS: usize = 300;
    #[cfg(miri)]
    const TAIL: usize = 12;
    #[cfg(not(miri))]
    const TAIL: usize = 2_000;
    const THREADS: usize = 6;
    /// The most a worker keeps live while the restarts land, so that the table
    /// a restart sweeps stays the size it is however long they take.
    const HELD: usize = 64;
    // A restart sweeps every slot of both tables, which is nothing natively and
    // most of the run under Miri. So there the tables are as small as the
    // workload allows.
    #[cfg(miri)]
    const LIMITS: (usize, usize) = (1 << 10, 1 << 12);
    #[cfg(not(miri))]
    const LIMITS: (usize, usize) = (1 << 14, 1 << 18);

    let engine = Engine::with_limits(LIMITS.0, LIMITS.1);
    assert!(engine.start(TimeSource::Events, || {}));
    let started = AtomicUsize::new(0);
    let restarting = AtomicBool::new(true);
    let raced = AtomicUsize::new(0);
    let violation: Mutex<Option<String>> = Mutex::new(None);

    let held: Vec<Vec<(usize, usize)>> = std::thread::scope(|s| {
        let (engine, started, restarting) = (&engine, &started, &restarting);
        let (raced, violation) = (&raced, &violation);
        s.spawn(move || {
            // Not before every worker is running, so that even the first
            // restart lands on threads that are recording.
            while started.load(Ordering::Acquire) < THREADS {
                std::thread::yield_now();
            }
            for restart in 1..=RESETS {
                let guard = heapscope::internals::guard::enter()
                    .expect("the restarting thread is not inside the profiler");
                engine
                    .reset(&guard)
                    .expect("a running engine restarts its counts");
                drop(guard);
                if let Err(difference) = sums_agree(engine) {
                    *violation.lock().unwrap() =
                        Some(format!("after restart {restart}: {difference}"));
                    break;
                }
            }
            restarting.store(false, Ordering::Release);
        });

        let workers: Vec<_> = (0..THREADS)
            .map(|t| {
                s.spawn(move || {
                    let base = 0x5_0000_0000usize + t * 0x1000_0000;
                    let mut live: std::collections::VecDeque<(usize, usize)> =
                        std::collections::VecDeque::new();
                    let mut round = |i: usize, capped: bool| {
                        let address = base + i * 128;
                        let size = 32 + (i * 7 + t) % 512;
                        guarded_alloc(
                            engine,
                            address,
                            Shape::of(size),
                            &frames_for((i + t) % SITES),
                        );
                        live.push_back((address, size));
                        // Frees of blocks old and new, so that blocks carried
                        // across a restart are freed after it as well as ones
                        // the window allocated.
                        if i.is_multiple_of(3) {
                            if let Some((address, size)) = live.pop_back() {
                                engine.record_free(address, size);
                            }
                        }
                        if i.is_multiple_of(11) || (capped && live.len() > HELD) {
                            if let Some((address, size)) = live.pop_front() {
                                engine.record_free(address, size);
                            }
                        }
                    };

                    started.fetch_add(1, Ordering::Release);
                    let mut i = 0;
                    while restarting.load(Ordering::Acquire) {
                        round(i, true);
                        raced.fetch_add(1, Ordering::Relaxed);
                        i += 1;
                    }
                    // The tail: uncapped, so the heap grows past where the last
                    // restart left the peak and the window has peaks of its own.
                    for _ in 0..TAIL {
                        round(i, false);
                        i += 1;
                    }
                    live.into_iter().collect()
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("a worker panicked"))
            .collect()
    });

    if let Some(difference) = violation.lock().unwrap().take() {
        panic!("a restart racing with allocations left the counters apart {difference}");
    }
    assert!(
        !heapscope::internals::diagnostic::is_poisoned(),
        "a restart racing with allocations poisoned the profiler"
    );
    assert_eq!(engine.resets(), RESETS as u64);
    assert!(
        raced.load(Ordering::Relaxed) >= RESETS,
        "only {} rounds ran while {RESETS} restarts landed, so the restarts \
         barely raced anything",
        raced.load(Ordering::Relaxed)
    );

    let stats = sums_agree(&engine).unwrap_or_else(|difference| panic!("at the end: {difference}"));
    let reset = engine.last_reset().expect("the run was restarted");
    assert_eq!(reset.count, RESETS as u64);

    // The last window is not empty, or the sums above compare zero with zero.
    assert!(
        stats.total_blocks > 0,
        "nothing was allocated after the last restart"
    );
    assert!(
        stats.epoch > reset.epoch,
        "the heap never peaked after the last restart"
    );
    assert!(
        stats.time_at_max >= reset.at,
        "the window's peak is before the restart that opened it"
    );

    let live_bytes: u64 = held.iter().flatten().map(|&(_, size)| size as u64).sum();
    let live_blocks = held.iter().map(Vec::len).sum::<usize>() as u64;
    assert_eq!(stats.curr_bytes, live_bytes, "live bytes");
    assert_eq!(stats.curr_blocks, live_blocks, "live blocks");
}

/// The peak a run reached must be the peak it recorded, when the threads
/// reaching it are racing for it.
///
/// # Why neither test above covers this
///
/// `concurrent_threads_agree_with_the_reference_tracker` compares every counter
/// exactly, but only with the engine serialized — and serializing sends every
/// event down the exclusive path, which is precisely the path that cannot lose
/// a peak. The shared path's compare-exchange is the thing it skips.
///
/// `concurrent_traces_preserve_the_summation_invariants` does drive the shared
/// path, for real, but every assertion it makes has the form *the parts sum to
/// the whole* — and a lost peak keeps all of them true. The per-point at-peak
/// counters are refreshed from the same epoch the global maximum was recorded
/// at, so they agree with it whatever instant that epoch names, including the
/// wrong one. Summation is self-consistency; it is not correctness.
///
/// **Measured, M7 chunk J:** deleting the gate's escalation check — the line
/// that sends a growth which would reach the recorded maximum down the
/// exclusive path rather than committing it under a shared guard, which is what
/// PLAN.md section 4.3 says the gate is *for* — passed all 673 tests.
///
/// # The oracle
///
/// Every round frees one block per thread and then allocates one of the same
/// size back, so live bytes leaves the maximum and returns to it *exactly*. At
/// the barrier that ends the round the run is therefore at an **equal peak**,
/// and the `>=` rule says the equal peak is the one recorded — so every program
/// point's at-peak counters must equal its current ones, because the instant
/// the epoch names is this one. No reference tracker, no serialization, and no
/// dependence on the interleaving: the answer is fixed by the shape of the
/// round.
///
/// That is `gb` and `gbk` checked against a known answer under real threads,
/// which is what section 12's third bullet claims and what the serialized test
/// can only establish for a serialized engine.
///
/// # Why the round is shaped this way
///
/// An event takes the shared path only when live bytes plus what it is about to
/// allocate still fall short of the recorded maximum. A run that only ever
/// grows never qualifies — the maximum *is* live bytes — and an earlier version
/// of this test, which only allocated, recorded exactly one such event in a
/// whole run and said so rather than passing. Freeing first is what opens the
/// window: after the barrier the heap sits one block per thread below its
/// maximum, so every thread begins its allocation believing no peak is
/// possible, and whichever of them commits last finds one anyway. That last
/// commit is what the gate's escalation check exists for.
///
/// The two sites differ on purpose. The block freed was allocated several
/// rounds ago at another point, so returning to the same total redistributes it
/// — which is what makes a missed equal peak visible at all. Recording the same
/// total again changes no global counter; it changes which points hold it.
///
/// `shared_path_opportunities` counts the window rather than assuming it, for
/// the reason the earlier version proved necessary.
#[test]
fn a_peak_reached_by_racing_threads_is_the_one_recorded() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Barrier, Mutex};

    #[cfg(miri)]
    const ROUNDS: usize = 4;
    #[cfg(not(miri))]
    const ROUNDS: usize = 150;
    const THREADS: usize = 8;
    /// One size throughout, so that freeing one block and allocating another
    /// returns live bytes to exactly the value they left.
    const BLOCK: usize = 512;
    /// Blocks each thread holds before the rounds begin. Also the age of the
    /// block a round frees, which is what puts it at a different program point
    /// from the one the round allocates at.
    const HELD: usize = 4;

    let engine = Engine::with_limits(1 << 14, 1 << 18);
    assert!(engine.start(TimeSource::Events, || {}));

    let barrier = Barrier::new(THREADS);
    let next_address = AtomicUsize::new(0x4_0000_0000);
    let shared_path_opportunities = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let violation: Mutex<Option<String>> = Mutex::new(None);

    std::thread::scope(|s| {
        for t in 0..THREADS {
            let (engine, barrier, next_address) = (&engine, &barrier, &next_address);
            let (stop, violation) = (&stop, &violation);
            let opportunities = &shared_path_opportunities;
            s.spawn(move || {
                // Each thread owns its blocks outright, so a round needs no
                // agreement about who frees what — only that every thread frees
                // one and allocates one.
                let mut held: std::collections::VecDeque<usize> =
                    std::collections::VecDeque::with_capacity(HELD + 1);
                for b in 0..HELD {
                    let address = next_address.fetch_add(4096, Ordering::Relaxed);
                    guarded_alloc(
                        engine,
                        address,
                        Shape::of(BLOCK),
                        &frames_for((t * 5 + b) % SITES),
                    );
                    held.push_back(address);
                }
                barrier.wait();

                for round in 0..ROUNDS {
                    // The descent. Afterwards the heap sits `THREADS` blocks
                    // below its maximum, which is the state the shared path
                    // needs and a growing run never reaches.
                    let address = held.pop_front().expect("every thread holds a block");
                    engine.record_free(address, BLOCK);
                    barrier.wait();

                    // The ascent, from a standing start: every thread arrives
                    // here at once, reads live bytes well below the maximum,
                    // and concludes no peak is possible. Seven of them are
                    // right.
                    let stats = engine.stats();
                    if stats.curr_bytes + (BLOCK as u64) < stats.max_bytes {
                        opportunities.fetch_add(1, Ordering::Relaxed);
                    }
                    let address = next_address.fetch_add(4096, Ordering::Relaxed);
                    guarded_alloc(
                        engine,
                        address,
                        Shape::of(BLOCK),
                        &frames_for((round + t * 3) % SITES),
                    );
                    held.push_back(address);

                    barrier.wait();
                    // Quiet: every thread is here, so nothing is in flight, the
                    // heap is back at exactly its maximum, and the equal peak
                    // that just happened is the one the profile must describe.
                    if t == 0 {
                        let mut stale = 0usize;
                        let mut example = String::new();
                        let flush = engine.flush_and_visit(
                            Engine::FLUSH_TIMEOUT,
                            |_id, frames, counters| {
                                if counters.at_gmax_bytes != counters.curr_bytes
                                    || counters.at_gmax_blocks != counters.curr_blocks
                                {
                                    stale += 1;
                                    example = format!(
                                        "{frames:x?} holds {} bytes in {} blocks but \
                                         reports {} bytes in {} blocks at the peak",
                                        counters.curr_bytes,
                                        counters.curr_blocks,
                                        counters.at_gmax_bytes,
                                        counters.at_gmax_blocks
                                    );
                                }
                            },
                            |_| {},
                            |_| {},
                        );
                        let stats = flush.stats;
                        if stale > 0
                            || stats.max_bytes != stats.curr_bytes
                            || stats.max_blocks != stats.curr_blocks
                        {
                            *violation.lock().unwrap() = Some(format!(
                                "round {round} returned the heap to exactly its maximum, \
                                 so that instant is the peak — but {stale} program \
                                 point(s) still describe an earlier one, and the run \
                                 records a peak of {} bytes in {} blocks while holding \
                                 {} bytes in {} blocks live.\n  {example}",
                                stats.max_bytes,
                                stats.max_blocks,
                                stats.curr_bytes,
                                stats.curr_blocks
                            ));
                            stop.store(true, Ordering::Relaxed);
                        }
                    }
                    // Both barriers are load-bearing. The one above makes the
                    // read quiet; this one publishes `stop` to every thread at
                    // once, so they leave together. One thread breaking out
                    // alone would leave the rest waiting at a barrier nobody
                    // will reach, turning a failure into a hang.
                    barrier.wait();
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                }
            });
        }
    });

    if let Some(difference) = violation.lock().unwrap().take() {
        panic!("{difference}");
    }

    let opportunities = shared_path_opportunities.load(Ordering::Relaxed);
    assert!(
        opportunities > ROUNDS,
        "only {opportunities} allocations began with live bytes far enough below \
         the maximum to reach the gate's shared path, so this run says nothing \
         about that path"
    );

    // The columns the bullet names, summed once at the end. The per-round check
    // above compares each point against a known answer; this one is the whole
    // against the parts, and both are needed: a run could describe the right
    // instant point by point and still disagree with its own totals.
    let mut summed_at_peak_bytes = 0u64;
    let mut summed_at_peak_blocks = 0u64;
    let flush = engine.flush_and_visit(
        Engine::FLUSH_TIMEOUT,
        |_id, _frames, counters| {
            summed_at_peak_bytes += counters.at_gmax_bytes;
            summed_at_peak_blocks += counters.at_gmax_blocks;
        },
        |_| {},
        |_| {},
    );
    assert!(flush.exclusive, "the flush could not reach a quiet point");
    let stats = flush.stats;
    assert_eq!(summed_at_peak_bytes, stats.max_bytes, "`gb` columns");
    assert_eq!(summed_at_peak_blocks, stats.max_blocks, "`gbk` columns");
}

/// Exact comparison under **real threads**, which a concurrent trace cannot
/// otherwise support.
///
/// # Why this needs `serialize_for_testing`
///
/// The engine's linearization point sits inside the peak gate. A reference
/// tracker wrapped around `record_alloc`/`record_free` has its own, different
/// one. Two threads doing `alloc(100)` and `free(100)` can be ordered A-then-B
/// by the gate and B-then-A by the reference, producing legitimately different
/// peaks — so a concurrent trace has no single reference run to compare against.
/// That is not a testing problem to be worked around; it is the same
/// non-linearizability the gate exists to remove, reappearing one layer up.
///
/// Serializing the engine and driving both implementations under one lock gives
/// the two a shared order, which makes every counter comparable exactly.
///
/// # What this proves, and what it does not
///
/// **Proves:** that attribution survives *cross-thread ownership* — blocks
/// allocated on one thread and freed or reallocated on another, program points
/// interned concurrently from several threads, per-thread guard slots and cached
/// stack bounds, and the pointer-sharded live table under a mixed access
/// pattern. All of those differ from the single-threaded case.
///
/// **Does not prove:** that two events executing *simultaneously* agree with a
/// serial model, because under a serial model that question has no answer.
/// `concurrent_traces_preserve_the_summation_invariants` covers the genuinely
/// overlapping case with invariants that hold under any interleaving.
///
/// # The workload is shaped by mutation testing, not by taste
///
/// An earlier version of this test allocated and freed with varying sizes and
/// no reallocations. It passed — and passed just as happily with the epoch's
/// `>=` changed to `>`, and with reallocations misattributed to the resizing
/// call site. It was exercising neither path. Reallocations from a *different*
/// site than allocated, and sizes drawn from a small recurring set so that live
/// bytes returns to values it has held before, are what make those two
/// mutations detectable here.
///
/// One thread also restarts the counts now and then, under the same lock, so
/// that blocks allocated on one thread before a restart are freed and
/// reallocated on others after it, and the restart's sweep over every shard of
/// the live table meets the pointer-sharded access pattern it has to survive.
#[test]
fn concurrent_threads_agree_with_the_reference_tracker() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    #[cfg(miri)]
    const ROUNDS: usize = 15;
    #[cfg(not(miri))]
    const ROUNDS: usize = 1_500;
    const THREADS: usize = 6;
    /// A small recurring set, so equal peaks actually occur. Equal peaks are the
    /// only place the epoch's `>=` rule differs from `>`.
    const SIZES: [usize; 4] = [64, 128, 256, 512];
    /// How often thread 0 restarts the counts: a handful of times a run, so
    /// each window is long enough to grow and peak again in.
    /// Once under Miri, where it sweeps tables of a quarter of a million slots.
    const RESET_EVERY: usize = if cfg!(miri) { 15 } else { 211 };

    let engine = Engine::with_limits(1 << 14, 1 << 18);
    assert!(engine.start(TimeSource::Events, || {}));
    engine.serialize_for_testing();

    // Both implementations advance under one lock, so they share an order.
    let model = Mutex::new(ReferenceTracker::default());
    // Blocks allocated by any thread, available for any other thread to free or
    // reallocate. This is the cross-thread ownership the test exists to check.
    let pool: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let next_address = AtomicUsize::new(0x3_0000_0000);
    let reallocs = AtomicUsize::new(0);
    let frees = AtomicUsize::new(0);
    let equal_peaks = AtomicUsize::new(0);
    let resets = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for t in 0..THREADS {
            let (engine, model, pool) = (&engine, &model, &pool);
            let (next_address, reallocs, frees) = (&next_address, &reallocs, &frees);
            let (equal_peaks, resets) = (&equal_peaks, &resets);
            s.spawn(move || {
                for i in 0..ROUNDS {
                    if t == 0 && i % RESET_EVERY == RESET_EVERY / 2 {
                        let guard = heapscope::internals::guard::enter()
                            .expect("a worker thread is not inside the profiler");
                        let mut model = model.lock().unwrap();
                        engine
                            .reset(&guard)
                            .expect("a running engine restarts its counts");
                        model.reset();
                        drop(model);
                        resets.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }

                    if i % 3 == 0 {
                        let taken = pool.lock().unwrap().pop();
                        if let Some((address, size)) = taken {
                            let mut model = model.lock().unwrap();
                            engine.record_free(address, size);
                            model.free(address);
                            frees.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    }

                    // Free a block and immediately replace it with one of
                    // exactly the same size, returning live bytes to the value
                    // it just held. This is the only way an *equal* peak
                    // occurs, and an equal peak is the only place the epoch's
                    // `>=` rule differs from `>`. Without it this test passed
                    // just as happily with that rule inverted.
                    if i % 7 == 0 {
                        let taken = pool.lock().unwrap().pop();
                        if let Some((address, size)) = taken {
                            let replacement = next_address.fetch_add(4096, Ordering::Relaxed);
                            let frames = frames_for((i + t) % SITES);

                            let mut model = model.lock().unwrap();
                            engine.record_free(address, size);
                            model.free(address);
                            guarded_alloc(engine, replacement, Shape::of(size), &frames);
                            model.alloc(replacement, size, &frames);
                            drop(model);

                            equal_peaks.fetch_add(1, Ordering::Relaxed);
                            pool.lock().unwrap().push((replacement, size));
                            continue;
                        }
                    }

                    if i % 5 == 0 {
                        let taken = pool.lock().unwrap().pop();
                        if let Some((address, size)) = taken {
                            let new_address = next_address.fetch_add(4096, Ordering::Relaxed);
                            let new_size = SIZES[(i + t) % SIZES.len()];
                            // Deliberately a different site from the one that
                            // allocated: the resize must be attributed to the
                            // original, not to whoever triggered it.
                            let frames = frames_for((i + t * 7 + 3) % SITES);

                            let mut model = model.lock().unwrap();
                            let held = engine.live_blocks().remove(address);
                            assert!(held.is_some(), "a pooled block was not tracked");
                            guarded_realloc(
                                engine,
                                held,
                                Realloc {
                                    old_address: address,
                                    old_size: size,
                                    new_address,
                                    new: Shape::of(new_size),
                                },
                                &frames,
                            );
                            model.realloc(address, new_address, new_size, &frames);
                            drop(model);

                            reallocs.fetch_add(1, Ordering::Relaxed);
                            pool.lock().unwrap().push((new_address, new_size));
                            continue;
                        }
                    }

                    // An event, sometimes landing while live bytes sit exactly
                    // at the maximum — which is the only interesting moment for
                    // it. There the engine's `>=` rule would record a new equal
                    // peak for an operation that allocated nothing, and this is
                    // the only test that drives `apply_without_peak` through the
                    // serialized path it has a branch for.
                    if i % 11 == 0 {
                        let weight = 100 + (i % 17) as u64;
                        let frames = frames_for((i + t * 3) % SITES);
                        // `Guard` is `!Send`, so each thread takes its own, and
                        // inside the branch rather than around the loop: a guard
                        // held across the whole worker would refuse every
                        // recursive entry the rest of it might legitimately make.
                        let guard = heapscope::internals::guard::enter()
                            .expect("a worker thread is not inside the profiler");
                        let mut model = model.lock().unwrap();
                        engine.record_event(&guard, weight, &frames);
                        model.event(weight, &frames);
                        drop(model);
                        continue;
                    }

                    let address = next_address.fetch_add(4096, Ordering::Relaxed);
                    let size = SIZES[(i * 3 + t) % SIZES.len()];
                    let frames = frames_for((i + t * 5) % SITES);

                    let mut model = model.lock().unwrap();
                    guarded_alloc(engine, address, Shape::of(size), &frames);
                    model.alloc(address, size, &frames);
                    drop(model);

                    pool.lock().unwrap().push((address, size));
                }
            });
        }
    });

    // The trace has to *end* on an equal peak, and getting there takes two
    // steps that are easy to miss.
    //
    // First, live bytes must actually be *at* the maximum. After a workload with
    // frees it is below, so a same-size replacement creates no peak at all and
    // the epoch never moves. Second, the last peak must be an *equal* one: the
    // at-peak snapshot reflects only the most recent peak, so a trace ending on
    // a strict increase records identical values under `>` and `>=` and
    // overwrites every earlier difference between them.
    {
        let mut model = model.lock().unwrap();

        // Step one: climb back to a fresh peak.
        let deficit = engine
            .stats()
            .max_bytes
            .saturating_sub(engine.stats().curr_bytes)
            + 4096;
        let address = next_address.fetch_add(1 << 20, Ordering::Relaxed);
        let frames = frames_for(0);
        guarded_alloc(&engine, address, Shape::of(deficit as usize), &frames);
        model.alloc(address, deficit as usize, &frames);
        pool.lock().unwrap().push((address, deficit as usize));
        assert_eq!(
            engine.stats().curr_bytes,
            engine.stats().max_bytes,
            "the tail failed to return live bytes to the peak"
        );

        // Step two: replace blocks with same-size ones from *different* call
        // sites. The total returns to the maximum each time, so each is an equal
        // peak, while the distribution across program points changes -- which is
        // what makes `>` and `>=` produce different at-peak values.
        for round in 0..8usize {
            let taken = pool.lock().unwrap().pop();
            let Some((address, size)) = taken else { break };
            let replacement = next_address.fetch_add(4096, Ordering::Relaxed);
            let frames = frames_for((round * 3 + 1) % SITES);

            engine.record_free(address, size);
            model.free(address);
            guarded_alloc(&engine, replacement, Shape::of(size), &frames);
            model.alloc(replacement, size, &frames);
            equal_peaks.fetch_add(1, Ordering::Relaxed);
            pool.lock().unwrap().push((replacement, size));
        }
        assert_eq!(
            engine.stats().curr_bytes,
            engine.stats().max_bytes,
            "the trace does not end at the peak"
        );
    }

    // A workload that silently stopped exercising a path would make this test
    // pass while proving nothing, which is exactly what an earlier version did.
    assert!(
        reallocs.load(Ordering::Relaxed) > ROUNDS / 4,
        "too few reallocations ({}) for this test to say anything about them",
        reallocs.load(Ordering::Relaxed)
    );
    assert!(
        frees.load(Ordering::Relaxed) > ROUNDS / 2,
        "too few frees ({})",
        frees.load(Ordering::Relaxed)
    );
    assert!(
        equal_peaks.load(Ordering::Relaxed) > ROUNDS / 8,
        "too few same-size replacements ({}); without them this test cannot \
         distinguish the epoch's `>=` rule from `>`",
        equal_peaks.load(Ordering::Relaxed)
    );
    assert!(
        resets.load(Ordering::Relaxed) > 0,
        "the run never restarted, so it says nothing about restarts"
    );

    let mut actual: BTreeMap<Vec<usize>, Counters> = BTreeMap::new();
    let flush = engine.flush_and_visit(
        Engine::FLUSH_TIMEOUT,
        |_id, frames, counters| {
            actual.insert(frames.to_vec(), *counters);
        },
        |_| {},
        |_| {},
    );
    assert!(flush.exclusive);
    let stats = flush.stats;

    let reference = model.into_inner().unwrap();
    assert_eq!(stats.curr_bytes, reference.curr_bytes, "live bytes");
    assert_eq!(stats.curr_blocks, reference.curr_blocks, "live blocks");
    assert_eq!(stats.total_bytes, reference.total_bytes, "cumulative bytes");
    assert_eq!(
        stats.total_blocks, reference.total_blocks,
        "cumulative blocks"
    );
    assert_eq!(stats.max_bytes, reference.max_bytes, "peak bytes");
    assert_eq!(stats.max_blocks, reference.max_blocks, "blocks at peak");
    assert_eq!(stats.time_at_max, reference.time_at_max, "time of the peak");
    let since = engine.last_reset().map_or(0, |reset| reset.epoch);
    assert_eq!(stats.epoch - since, reference.peaks, "peaks in the window");

    let expected = reference.finish();
    for (frames, want) in &expected {
        let got = actual
            .get(frames)
            .unwrap_or_else(|| panic!("engine lost program point {frames:x?}"));
        assert_eq!(
            got, want,
            "program point {frames:x?} disagrees with the model"
        );
    }
    assert_eq!(
        actual.len(),
        expected.len(),
        "the engine and the model disagree on how many program points exist"
    );
}

/// A program's allocator keeps handing out addresses it has not used before,
/// for as long as the program runs, and the live table has to keep tracking
/// them with no more than a few live at once.
///
/// This is the shape that once stopped a run being recorded. A removal left a
/// tombstone, which only an insertion landing on it reclaimed, while every
/// insertion that landed on an empty slot used one up; so the empty slots only
/// ever decreased, and a table at its ceiling, unable to clear its tombstones
/// by growing, refused every new block with a handful live. On a loaded CI
/// runner, a test whose workers churned a few million distinct addresses saw
/// about 2.2 million blocks dropped with a few hundred live. A table sized for
/// 1,024 blocks makes the same thing happen within a few thousand.
///
/// Through the engine rather than the table alone, because what a program
/// sees is the engine's counters: a refused block is missing from the live
/// figures and the totals, and counted in `dropped_blocks`.
#[test]
fn churning_distinct_addresses_never_fills_the_live_table() {
    #[cfg(miri)]
    const ALLOCATIONS: usize = 3_000;
    #[cfg(not(miri))]
    const ALLOCATIONS: usize = 200_000;
    /// Blocks live at any time. Far below the 1,024 the table is sized for.
    const LIVE: usize = 8;
    const SIZE: usize = 48;

    let engine = Engine::with_limits(1 << 14, 1 << 10);
    assert!(engine.start(TimeSource::Events, || {}));
    let frames = frames_for(0);
    let address = |i: usize| 0x7_0000_0000usize + i * 64;
    for i in 0..ALLOCATIONS {
        guarded_alloc(&engine, address(i), Shape::of(SIZE), &frames);
        if i >= LIVE {
            engine.record_free(address(i - LIVE), SIZE);
        }
    }

    let stats = engine.stats();
    assert_eq!(stats.dropped_blocks, 0, "the live table refused blocks");
    assert_eq!(stats.total_blocks, ALLOCATIONS as u64);
    assert_eq!(stats.curr_blocks, LIVE as u64);
    assert_eq!(stats.curr_bytes, (LIVE * SIZE) as u64);
}
