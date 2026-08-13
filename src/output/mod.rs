//! Turning recorded state into files people can read.
//!
//! Everything here runs at output time, never on the allocation path, and is
//! organised around one type: a [`Snapshot`] is taken once, and each emitter is
//! a pure function of it. That split is deliberate — it means the emitters can
//! be tested against hand-built snapshots with no engine, no allocator shim and
//! no threads, which is the only way to test the awkward cases (a program point
//! with no frames, two points that collapse onto one frame list, a peak that
//! never moved) reliably.
//!
//! # Taking a snapshot without deadlocking
//!
//! [`Snapshot::capture`] reads the per-point counters through
//! [`Engine::flush_and_visit`](crate::internals::engine::Engine::flush_and_visit),
//! which runs its visitor **with the peak gate held**. The visitor therefore
//! must not allocate: an allocation there re-enters the engine, which blocks on
//! the gate this very thread is holding. That is a hang on Linux and Windows,
//! and on macOS it is worse — the shard locks are `os_unfair_lock`, which kills
//! the process outright on reentrant acquisition rather than deadlocking.
//!
//! So the capture is in two phases. Phase one, under the gate, copies fixed-size
//! counters into storage reserved *before* the gate was taken, and pushes
//! nothing that would make the vector grow. Phase two, after the gate is
//! released, copies each point's frames out of the arena, where they have been
//! immutable since the moment they were interned. The counters are still an
//! atomic snapshot; the frames are exact because interned frames never change.
//!
//! # Writing does not disturb what is being measured
//!
//! Capturing and writing both hold the reentrancy guard, so every allocation
//! this module makes is invisible to the engine. That matters for a profile
//! written mid-run: without it, asking for a summary would add its own string
//! formatting to the numbers the *next* summary reports, and a profiler that
//! changes what it measures is the failure this crate exists to avoid.

mod dhat_v2;
mod folded;
mod html;
mod json;
mod native;
mod text;

use std::io::{self, Write};
use std::path::Path;

use crate::internals::clock::TimeSource;
use crate::internals::engine::Engine;
use crate::internals::pp::PpId;
use crate::symbol::modules::{self, Module};
use crate::symbol::{Symbolized, Trimmed};

pub use crate::internals::engine::{GlobalStats, Settings, Shutdown};
pub use crate::internals::pp::Counters;
pub use crate::internals::shape::{Realloc, Shape, ShapeStats};
pub use crate::internals::site::TallyStats;
pub use dhat_v2::{FrameFormat, RawAddresses};
pub use folded::FoldedMetric;

pub(crate) use dhat_v2::push_hex;
pub(crate) use text::count;

/// How much of one of the profiler's tables is in use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TableUsage {
    /// Entries held.
    pub entries: usize,
    /// Entries the table has room for before it starts refusing.
    pub capacity: usize,
    /// Arena bytes the table's contents occupy: its entries, its index, and
    /// whatever those point at — for the program-point table, the frame lists.
    ///
    /// Not the table's own fixed structure, which is the same size whether it
    /// holds one row or a million and does not come from the arena. So this
    /// answers "what did recording this program cost", which is the question
    /// [`SelfMetrics`] exists for, rather than "how large is this data
    /// structure".
    pub bytes: usize,
}

/// What the profiler cost the program it was measuring.
///
/// PLAN.md section 12 promises "honestly measured overhead". A promise a user
/// cannot check is a claim about our good intentions, so every profile carries
/// the numbers behind it: how much memory the profiler is holding, how full its
/// tables are, and what a stack capture cost on this machine in this build.
///
/// Reading them is what turns a surprising profile into a diagnosable one. A
/// live-block table at capacity explains a `droppedBlocks` count; an arena at
/// its limit explains program points that stopped being interned; a capture cost
/// two orders of magnitude above the frame-pointer figure says the run used the
/// platform unwinder, whatever else it says.
///
/// `#[non_exhaustive]`: what the profiler records about itself grows, and each
/// addition would otherwise be a breaking change rather than a rebuild.
///
/// The sampling rate was once planned to land here and did not. It is on
/// [`Settings::sampling`], because this struct answers what the run *cost* and
/// that field is part of what the run *was* — a reader consulting these numbers
/// to explain a surprising profile and a reader checking whether the numbers are
/// estimates at all are asking different questions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SelfMetrics {
    /// The bump arena every byte of profiler state comes from.
    pub arena: crate::internals::arena::ArenaStats,
    /// The program-point table.
    ///
    /// [`TableUsage::capacity`] is the ceiling past which a new call site is
    /// folded into the `[overflow]` point rather than interned.
    pub program_points: TableUsage,
    /// The live-block table.
    ///
    /// [`TableUsage::entries`] is what the table held when it was read, which is
    /// a moment after the counters were: it is taken outside the flush window,
    /// because nothing cross-checks it and taking the gate for a descriptive
    /// number would make every profile pay for a consistency nothing reads.
    /// Expect it to agree with [`GlobalStats::curr_blocks`] in a stopped run
    /// that dropped nothing, and not to under concurrency.
    pub live_blocks: TableUsage,
    /// The thread table.
    ///
    /// [`TableUsage::capacity`] is the number of threads that get a row of
    /// their own; everything past it shares one, which the profile shows rather
    /// than folding into a real thread.
    pub threads: TableUsage,
    /// The region table, on the same terms as the thread one.
    pub regions: TableUsage,
    /// What one stack capture cost, measured at startup. See
    /// [`Cost`](crate::unwind::Cost).
    pub capture_cost: crate::unwind::Cost,
}

/// One thread, as it stood when the snapshot was taken.
///
/// DHAT v2 has no field for any of this: its per-point counters merge every
/// thread that reached a call site, so "the same site, but one worker is
/// holding all the memory" is not a shape its file can take. See
/// [`Snapshot::threads`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreadStats {
    /// Which row. Assignment order, so row 0 is the first thread that recorded
    /// anything, which is usually but not necessarily the main thread.
    pub id: u16,
    /// Whether this is the shared row for threads past the table's capacity,
    /// rather than one thread.
    pub overflow: bool,
    /// The name the platform had for the thread.
    ///
    /// `None` where it had none, which is the normal state of a thread nobody
    /// named. This is the OS-level name, so it is the one a debugger and
    /// `top -H` show, and on Linux it is subject to the kernel's 15-byte limit.
    pub name: Option<String>,
    /// Clock reading when this thread first recorded something.
    pub first_seen: u64,
    /// What it recorded.
    pub counts: TallyStats,
}

/// One region, as it stood when the snapshot was taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionStats {
    /// Which row, in the order the names were first entered.
    pub id: u16,
    /// Whether this is the shared row for names past the table's capacity.
    pub overflow: bool,
    /// The name the program gave the region.
    pub name: Option<String>,
    /// Clock reading when the name was first entered.
    pub first_seen: u64,
    /// Times entered, on any thread.
    pub entries: u64,
    /// Times entered and not yet left when the snapshot was taken.
    ///
    /// Non-zero in a finished run means a region guard outlived the profiler,
    /// or was leaked. The counters are still sound; the region simply had not
    /// ended yet.
    pub active: u64,
    /// What was allocated while it was the innermost open region on some
    /// thread.
    pub counts: TallyStats,
}

/// What a program point stands for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PointKind {
    /// A call site the profiler interned.
    #[default]
    Recorded,
    /// The synthetic point that absorbs allocations once the program-point
    /// table is full.
    ///
    /// It has no frames, because the whole reason it exists is that there was
    /// nowhere left to put them. PLAN.md section 4.6 requires it to be *visible*
    /// in the output rather than silently folded in with the genuine points: a
    /// profile whose heaviest site is "everything we ran out of room for" is
    /// telling the reader to raise the ceiling, and that is a different message
    /// from any real call site.
    Overflow,
}

/// One program point, as it stood when the snapshot was taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramPoint {
    /// Whether this is a real call site or the overflow point.
    pub kind: PointKind,
    /// Return addresses, innermost first.
    ///
    /// Empty is legal: a capture that found no usable frames still allocates,
    /// and dropping it would lose bytes from the totals.
    pub frames: Vec<usize>,
    /// The recorded counters.
    pub counters: Counters,
    /// Summed lifetime of blocks from this point that were **still alive** when
    /// the snapshot was taken, measured to the snapshot instant.
    ///
    /// Kept separate from [`Counters::total_lifetime`], which only ever counts
    /// blocks that were freed, so that neither number has to lie about what it
    /// measures. Valgrind folds the two together by retiring every live block at
    /// exit; [`ProgramPoint::total_lifetime`] does the same on demand, which has
    /// the advantage of leaving the engine's own state untouched, so writing a
    /// profile twice produces the same numbers both times.
    pub unretired_lifetime: u64,
}

impl ProgramPoint {
    /// Summed lifetime of every block from this point, live ones included.
    ///
    /// This is what DHAT's `tl` field means. Counting only freed blocks would
    /// make a program point that allocates once and never frees look like the
    /// shortest-lived site in the program.
    pub fn total_lifetime(&self) -> u64 {
        self.counters
            .total_lifetime
            .saturating_add(self.unretired_lifetime)
    }
}

/// Everything an emitter needs, read from the engine in one pass.
///
/// `#[non_exhaustive]`: this grows every time the profiler learns to record
/// something. Adding `settings` to it is what forced two other files in this
/// crate to change, and outside the crate that is a breaking change rather than
/// a rebuild. Reading a field is unaffected, which is what emitters do; building
/// one from nothing goes through [`Snapshot::default`] and then assignment.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Snapshot {
    /// Global counters, read in the same window as the per-point ones.
    pub stats: GlobalStats,
    /// Whether the counters are exactly consistent with each other.
    ///
    /// False when the engine could not reach a quiet point within its timeout,
    /// which means an event may have landed between the per-point counters and
    /// the global ones. The profile records the fact rather than presenting a
    /// possibly inconsistent snapshot as exact.
    pub exact: bool,
    /// Whether the profiler reported an internal failure during the run.
    ///
    /// A poisoned engine keeps recording, and its numbers may still be entirely
    /// usable — but the profile says so, because a reader deciding whether to
    /// trust a surprising figure needs to know.
    pub poisoned: bool,
    /// Which path stopped recording, or [`Shutdown::Running`] for a reading
    /// taken mid-run.
    ///
    /// The two automatic paths take their readings at different points in
    /// process teardown and legitimately disagree, so a profile says which one
    /// produced it. See [`Shutdown`].
    pub shutdown: Shutdown,
    /// Which unwinder captured the frames in these program points.
    pub unwinder: crate::unwind::Strategy,
    /// How many captures came back whole, and how many did not.
    ///
    /// The startup probe walks this crate's own frames, which says nothing about
    /// C or C++ dependencies built by someone else, hand-written assembly, JIT
    /// frames, or threads a C library created. These counts turn that false
    /// confidence into a number a reader can check.
    pub captures: crate::unwind::CounterSnapshot,
    /// The time base these readings are in.
    pub time_source: TimeSource,
    /// The clock reading at the moment of the snapshot. DHAT's `te`.
    pub time_at_end: u64,
    /// One entry per program point that has recorded at least one allocation.
    ///
    /// **Ordered by when the program first reached each point**, with the
    /// overflow point last. The order is a property of the run rather than of
    /// the process: it does not move when the program is loaded at a different
    /// address, which is what lets an emitter that writes points in this order
    /// produce the same file twice. An emitter free to choose its own order
    /// should still break ties by position here, so that equal-weight points do
    /// not swap places between runs.
    ///
    /// In a program whose threads race to reach the same call sites, which
    /// thread arrived first is a property of that race and this order follows
    /// it. The counters do not: they are exact under concurrency, by the peak
    /// gate.
    pub points: Vec<ProgramPoint>,
    /// Program points that appeared while the snapshot was being taken and did
    /// not fit in the space reserved for them.
    ///
    /// Only ever non-zero for a snapshot taken while the profiler is running,
    /// and even then only if the program interned thousands of new points in
    /// the microseconds the flush took.
    pub points_dropped: u64,
    /// Live blocks whose program point was not in the snapshot, and whose
    /// lifetime therefore could not be attributed.
    pub unattributed_blocks: u64,
    /// The command line, as `argv` joined by spaces.
    pub command: String,
    /// The process ID.
    pub pid: u32,
    /// Every image mapped into the process, ordered by load address.
    ///
    /// This is what makes the recorded addresses mean anything after the process
    /// exits: address space layout randomization moves everything between runs,
    /// so a bare address is only interpretable alongside the map that says where
    /// each image was.
    pub modules: Vec<Module>,
    /// What the run that produced these readings was configured to do.
    ///
    /// Carried on the snapshot rather than read from the engine when a profile
    /// is written, so that a snapshot taken now and written later describes the
    /// run it came from. It is also what decides the default rendering: a
    /// profiler built with `trim_frames(false)` writes untrimmed profiles from
    /// [`Snapshot::write_dhat_v2`], with no argument at the call site saying so.
    pub settings: Settings,
    /// What the program asked for, beyond a number of bytes: the distribution
    /// of sizes and alignments, the blocks it wanted zeroed, and what
    /// reallocation cost.
    ///
    /// DHAT v2 has no field for any of it, so this reaches the native format
    /// and the text summary and is folded into a few lines of the DHAT file's
    /// own extension block.
    pub shapes: ShapeStats,
    /// What the profiler cost the program it was measuring.
    pub metrics: SelfMetrics,
    /// One row per thread that recorded something, in the order they first did.
    ///
    /// Read **inside** the flush window, and moved under the peak gate, so
    /// these sum to [`Snapshot::stats`] exactly whenever [`Snapshot::exact`] is
    /// true — which is every run that reached a quiet point, including one
    /// snapshotted while other threads are still recording. See
    /// `Engine::attribute`.
    pub threads: Vec<ThreadStats>,
    /// One row per region name the program entered, in the order it first
    /// entered them. Empty for a run that used no regions, which is most of
    /// them.
    pub regions: Vec<RegionStats>,
    /// Attribution rows that appeared while the snapshot was being taken and
    /// did not fit in the space reserved for them.
    ///
    /// The counterpart of [`Snapshot::points_dropped`], and it matters for the
    /// same reason: when this is non-zero the rows no longer sum to
    /// [`Snapshot::stats`], and a reader checking that they do needs to know
    /// which of the two is missing something. Reaching it takes
    /// [`ROW_SLACK`] threads recording their *first* event inside one flush
    /// window — one fewer than the slack, because the shared overflow row is
    /// also emitted.
    pub rows_dropped: u64,
}

/// Extra attribution rows a snapshot reserves room for.
///
/// A row is only added when a thread records for the first time or a region
/// name is entered for the first time, so this is slack against new threads
/// appearing during the flush itself — not against the table's size, which is
/// already known when the space is reserved.
pub const ROW_SLACK: usize = 64;

impl Default for Snapshot {
    /// The reading a profiler that recorded nothing would produce.
    ///
    /// Written out rather than derived because two of the fields have no
    /// meaningful zero of their own: the time base and the capture strategy are
    /// choices, and this states which choice an empty snapshot describes.
    fn default() -> Self {
        Self {
            stats: GlobalStats::default(),
            exact: true,
            poisoned: false,
            shutdown: Shutdown::default(),
            unwinder: crate::unwind::Strategy::default(),
            captures: crate::unwind::CounterSnapshot::default(),
            time_source: TimeSource::default(),
            time_at_end: 0,
            points: Vec::new(),
            points_dropped: 0,
            unattributed_blocks: 0,
            command: String::new(),
            pid: 0,
            modules: Vec::new(),
            settings: Settings::default(),
            shapes: ShapeStats::default(),
            metrics: SelfMetrics::default(),
            threads: Vec::new(),
            regions: Vec::new(),
            rows_dropped: 0,
        }
    }
}

impl Snapshot {
    /// Reads the current state of the process-wide engine.
    ///
    /// Safe to call at any time. Taken while the profiler is running it is a
    /// point-in-time reading and the live-block sweep behind
    /// [`ProgramPoint::unretired_lifetime`] may race with concurrent frees;
    /// taken after the profiler has stopped, which is the normal case, nothing
    /// is moving and the result is exact.
    pub fn capture() -> Self {
        Self::of(crate::engine())
    }

    /// Reads the state of a specific engine. Testing hook.
    pub(crate) fn of(engine: &Engine) -> Self {
        // Keeps this function's own allocations out of the profile it is about
        // to write. Not load-bearing for *correctness* — every path below is
        // safe to re-enter, deliberately, so that a snapshot taken without the
        // guard is still right; it just reports a few of its own allocations.
        let _quiet = crate::internals::guard::enter();

        let time_source = engine.time_source();
        let table = engine.program_points();

        // Reserved *before* the gate is taken. See the module documentation for
        // why the visitors below cannot be allowed to allocate.
        // The leading `u32` is the creation sequence, filled in below once the
        // gate is free. It is not read here because reading it takes a shard
        // lock per point, and every allocating thread in the program is stopped
        // for as long as this window lasts.
        let mut raw: Vec<(u32, PpId, Counters)> = Vec::with_capacity(table.len() * 2 + 64);
        let capacity = raw.capacity();
        let mut points_dropped = 0u64;

        // The attribution rows, on the same terms and for the same reason: they
        // sum to the totals, so they have to be read in the window the totals
        // are. The names are borrowed from the arena and copied afterwards,
        // because a `String` here would allocate under the gate.
        let mut thread_rows: Vec<crate::internals::site::ThreadView> =
            Vec::with_capacity(engine.threads().len() + ROW_SLACK);
        let thread_capacity = thread_rows.capacity();
        let mut region_rows: Vec<crate::internals::site::RegionView> =
            Vec::with_capacity(engine.regions().len() + ROW_SLACK);
        let region_capacity = region_rows.capacity();
        // One counter per closure, summed afterwards: two closures cannot hold
        // a mutable borrow of the same one, and a `Cell` here would be a shared
        // mutable variable inside the gate for the sake of saving an addition.
        let mut thread_rows_dropped = 0u64;
        let mut region_rows_dropped = 0u64;

        let flush = engine.flush_and_visit(
            Engine::FLUSH_TIMEOUT,
            |id, _frames, counters| {
                if raw.len() < capacity {
                    raw.push((0, id, *counters));
                } else {
                    points_dropped += 1;
                }
            },
            |row| {
                if thread_rows.len() < thread_capacity {
                    thread_rows.push(row);
                } else {
                    thread_rows_dropped += 1;
                }
            },
            |row| {
                if region_rows.len() < region_capacity {
                    region_rows.push(row);
                } else {
                    region_rows_dropped += 1;
                }
            },
        );
        let rows_dropped = thread_rows_dropped + region_rows_dropped;
        let time_at_end = engine.clock().now(time_source);

        // Canonical order, and the reason every emitter downstream can be
        // reproducible. `flush_and_visit` walks shards in index order, and a
        // point's shard comes from hashing its return addresses — which address
        // space layout randomization moves on every execution. Two runs of one
        // deterministic program therefore visit the same points in a different
        // order each time, and any emitter that writes them in visit order
        // writes a file that differs everywhere while meaning the same thing.
        // Sorting by creation sequence replaces a reading of where the program
        // was mapped with a reading of what it did.
        for entry in &mut raw {
            entry.0 = table.sequence(entry.1);
        }
        raw.sort_unstable_by_key(|&(sequence, ..)| sequence);

        // Phase two: frames, copied from the arena now that the gate is free.
        let mut points = Vec::with_capacity(raw.len());
        let mut frames = Vec::new();
        for (_, id, counters) in &raw {
            table.frames(*id, &mut frames);
            points.push(ProgramPoint {
                kind: if *id == PpId::OVERFLOW {
                    PointKind::Overflow
                } else {
                    PointKind::Recorded
                },
                frames: frames.clone(),
                counters: *counters,
                unretired_lifetime: 0,
            });
        }

        // Lifetimes of blocks that were never freed. Valgrind retires every
        // live block at exit for exactly this reason: without it, a point that
        // allocates and holds contributes nothing to `tl`, and the viewer's
        // "short-lived" filter then reports the longest-lived sites in the
        // program as the shortest-lived ones.
        let mut index: Vec<(u32, u32)> = raw
            .iter()
            .enumerate()
            .map(|(at, (_, id, _))| (id.as_u32(), at as u32))
            .collect();
        index.sort_unstable();
        let mut unattributed_blocks = 0u64;
        engine.live_blocks().for_each(|_address, block| {
            match index.binary_search_by_key(&block.pp.as_u32(), |&(id, _)| id) {
                Ok(at) => {
                    let point = &mut points[index[at].1 as usize];
                    let lifetime = time_at_end.saturating_sub(block.birth);
                    point.unretired_lifetime = point.unretired_lifetime.saturating_add(lifetime);
                }
                Err(_) => unattributed_blocks += 1,
            }
        });

        // Owned rows, built now that the gate is free. Only the names are
        // copied here; every number came out of the window above.
        let threads: Vec<ThreadStats> = thread_rows
            .iter()
            .map(|row| ThreadStats {
                id: row.id.as_u16(),
                overflow: row.id.is_overflow(),
                name: name_of(row.name.as_bytes()),
                first_seen: row.first_seen,
                counts: row.counts,
            })
            .collect();
        let regions: Vec<RegionStats> = region_rows
            .iter()
            .map(|row| RegionStats {
                id: row.id.as_u16(),
                overflow: row.id.is_overflow(),
                name: name_of(row.name.as_bytes()),
                first_seen: row.first_seen,
                entries: row.entries,
                active: row.active,
                counts: row.counts,
            })
            .collect();

        Snapshot {
            stats: flush.stats,
            exact: flush.exclusive,
            poisoned: crate::internals::diagnostic::is_poisoned(),
            shutdown: engine.shutdown(),
            unwinder: crate::unwind::strategy(),
            captures: crate::unwind::counters().snapshot(),
            time_source,
            time_at_end,
            points,
            points_dropped,
            unattributed_blocks,
            command: command_line(),
            pid: std::process::id(),
            modules: modules::capture(),
            settings: engine.settings(),
            // From the flush rather than from a second read, so that the
            // histograms describe the same instant the totals do.
            shapes: flush.shapes,
            // Read after the flush rather than before it, so the arena and
            // table figures include whatever the run's last events added. They
            // do not include this function's own allocations: it holds the
            // reentrancy guard, and its `Vec`s come from the global allocator
            // rather than from the profiler's arena.
            metrics: SelfMetrics {
                arena: engine.arena().stats(),
                program_points: TableUsage {
                    entries: table.len(),
                    capacity: table.capacity(),
                    bytes: table.bytes(),
                },
                live_blocks: TableUsage {
                    entries: engine.live_blocks().len(),
                    capacity: engine.live_blocks().max_blocks(),
                    bytes: engine.live_blocks().bytes(),
                },
                threads: TableUsage {
                    entries: engine.threads().len(),
                    capacity: engine.threads().capacity(),
                    bytes: engine.threads().bytes(),
                },
                regions: TableUsage {
                    entries: engine.regions().len(),
                    capacity: engine.regions().capacity(),
                    bytes: engine.regions().bytes(),
                },
                capture_cost: crate::unwind::capture_cost(),
            },
            threads,
            regions,
            rows_dropped,
        }
    }

    /// Writes a DHAT format version 2 profile.
    ///
    /// The result opens in Valgrind's `dh_view.html`. Frames are rendered by
    /// [`Symbolized`]: the name the running process knows an address by, where
    /// it knows one, followed by `image + offset` against the snapshot's module
    /// map, which is the form `atos`, `addr2line`, and `llvm-symbolizer`
    /// resolve. The second half is always present, so a profile written on a
    /// stripped build is exactly as resolvable afterwards as one written
    /// without symbolization at all.
    ///
    /// Frames are also [`Trimmed`]: the allocation path above the program and
    /// the runtime entry below it are left out, because they are the same on
    /// every stack and are most of every stack. How many were left out is
    /// recorded in the profile as `trimmedFrames`, and what a trimmed profile
    /// can no longer answer is listed on [`Trimmed`] itself.
    ///
    /// See [`Snapshot::write_dhat_v2_with`] to supply a different rendering:
    /// `Symbolized` alone keeps every frame, and
    /// [`ModuleOffsets`](crate::symbol::ModuleOffsets) leaves names out.
    pub fn write_dhat_v2<W: Write>(&self, out: W) -> io::Result<()> {
        let names = Symbolized::new(&self.modules);
        if self.settings.trim_frames {
            self.write_dhat_v2_with(out, &Trimmed::new(names))
        } else {
            self.write_dhat_v2_with(out, &names)
        }
    }

    /// Writes a DHAT format version 2 profile, rendering frames with `format`.
    ///
    /// Which frames appear is `format`'s decision as much as what they are
    /// called: see [`FrameFormat::keep`], whose default keeps all of them. So
    /// this method trims exactly when the renderer passed to it does — passing
    /// [`Symbolized`] gives a named but complete stack, and wrapping it in
    /// [`Trimmed`] gives what [`Snapshot::write_dhat_v2`] writes.
    pub fn write_dhat_v2_with<W: Write>(&self, out: W, format: &dyn FrameFormat) -> io::Result<()> {
        let _quiet = crate::internals::guard::enter();
        dhat_v2::write(self, format, out)
    }

    /// Writes a DHAT format version 2 profile to `path`, replacing any file
    /// already there.
    ///
    /// The profile is written beside its destination and renamed into place, so
    /// that a full disk, a write error, or a process killed mid-write leaves the
    /// previous profile intact rather than replacing it with a truncated file
    /// that no viewer will open.
    pub fn save_dhat_v2(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.save_with(path, |snapshot, out| snapshot.write_dhat_v2(out))
    }

    /// Writes a native profile: everything recorded, in the shape it was
    /// recorded in.
    ///
    /// The DHAT v2 file is a projection of this one. Where the two disagree in
    /// what they can express, this is the one that is not lossy:
    ///
    /// - Frames are **addresses**, with the image, the file address, and the
    ///   symbol as separate answers rather than joined into a line of text.
    /// - Nothing is trimmed and nothing is folded. Both exist to satisfy
    ///   `dh_view.html`, and neither is a fact about the run.
    /// - The two lifetime totals stay apart, where DHAT has one field for their
    ///   sum.
    /// - What the program asked for beyond a number of bytes — the distribution
    ///   of sizes and alignments, the blocks it wanted zeroed, what
    ///   reallocation copied — and what the profiler itself cost, neither of
    ///   which DHAT v2 has a field for.
    ///
    /// It is JSON, and it is versioned: a reader must ignore fields it does not
    /// know and refuse a `formatVersion` it does not know. That rule is written
    /// into every file rather than only here.
    pub fn write_native<W: Write>(&self, out: W) -> io::Result<()> {
        let _quiet = crate::internals::guard::enter();
        native::write(self, out)
    }

    /// Writes a native profile to `path`, replacing any file already there.
    ///
    /// Written beside its destination and renamed into place, like
    /// [`Snapshot::save_dhat_v2`].
    pub fn save_native(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.save_with(path, |snapshot, out| snapshot.write_native(out))
    }

    /// Writes a self-contained HTML page: the profile, and a viewer for it.
    ///
    /// One file, no build step, and nothing fetched when it is opened — so it
    /// works from a file:// URL on a machine with no network, no Valgrind, and
    /// no tooling of any kind. That is the point of it: Valgrind does not exist
    /// on Windows and does not support Apple Silicon, so for two of the four
    /// supported platforms `dh_view.html` is not something the reader can be
    /// assumed to have.
    ///
    /// It is a **complement** to [`Snapshot::save_dhat_v2`], not a replacement.
    /// DHAT v2 stays the interchange format, and `dh_view.html` is better at the
    /// tree than this is. What this shows that DHAT structurally cannot is the
    /// rest of what was recorded: thread attribution, regions, the distribution
    /// of sizes and alignments, sampling metadata, and what the profiler itself
    /// cost.
    ///
    /// The page carries the native profile verbatim, so it is also the data: a
    /// reader can lift the JSON back out of it without this crate.
    ///
    /// Frames are rendered by [`Symbolized`] and [`Trimmed`], the same pair
    /// [`Snapshot::write_dhat_v2`] uses, and the page offers to show the trimmed
    /// frames as well because the full stacks are in the profile beside it.
    pub fn write_html<W: Write>(&self, out: W) -> io::Result<()> {
        let names = Symbolized::new(&self.modules);
        if self.settings.trim_frames {
            self.write_html_with(out, &Trimmed::new(names))
        } else {
            self.write_html_with(out, &names)
        }
    }

    /// Writes a self-contained HTML page, rendering frames with `format`.
    pub fn write_html_with<W: Write>(&self, out: W, format: &dyn FrameFormat) -> io::Result<()> {
        let _quiet = crate::internals::guard::enter();
        html::write(self, format, out)
    }

    /// Writes a self-contained HTML page to `path`, replacing any file already
    /// there.
    ///
    /// Written beside its destination and renamed into place, like
    /// [`Snapshot::save_dhat_v2`].
    pub fn save_html(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.save_with(path, |snapshot, out| snapshot.write_html(out))
    }

    /// Writes folded stacks: one line per distinct stack, outermost frame
    /// first, separated by `;`, with `metric` as the count at the end.
    ///
    /// ```text
    /// main;run;parse;Vec::with_capacity 1048576
    /// ```
    ///
    /// This is what `inferno`, `flamegraph.pl`, `speedscope`, and the Firefox
    /// Profiler read. None of them knows anything about this crate, and this is
    /// the format that does not ask them to.
    ///
    /// Frames are rendered by [`Symbolized`] and [`Trimmed`], the same pair
    /// [`Snapshot::write_dhat_v2`] uses — a flame graph is read at a glance, so
    /// it is where the nine frames of runtime entry every stack shares cost the
    /// most.
    ///
    /// # Errors
    ///
    /// Beyond what `out` returns: [`InvalidInput`](io::ErrorKind::InvalidInput)
    /// where the run has no measurement for `metric`. A file carries one column
    /// and has nothing to omit into, so a metric that needs block lifetimes is
    /// refused in a mode that has none rather than written as zeroes — see
    /// [`FoldedMetric::needs_block_lifetimes`], which is the check that predicts
    /// this.
    pub fn write_folded<W: Write>(&self, out: W, metric: FoldedMetric) -> io::Result<()> {
        let names = Symbolized::new(&self.modules);
        if self.settings.trim_frames {
            self.write_folded_with(out, &Trimmed::new(names), metric)
        } else {
            self.write_folded_with(out, &names, metric)
        }
    }

    /// Writes folded stacks, rendering frames with `format`.
    ///
    /// Worth reaching for here more than elsewhere: a flame graph groups by the
    /// text of a frame, so what a renderer chooses to show is what the picture
    /// merges. [`ModuleOffsets`](crate::symbol::ModuleOffsets) draws one tower
    /// per image, and [`Symbolized`] without [`Trimmed`] keeps the runtime entry
    /// sequence that [`Snapshot::write_folded`] leaves out.
    pub fn write_folded_with<W: Write>(
        &self,
        out: W,
        format: &dyn FrameFormat,
        metric: FoldedMetric,
    ) -> io::Result<()> {
        let _quiet = crate::internals::guard::enter();
        folded::write(self, format, metric, out)
    }

    /// Writes folded stacks to `path`, replacing any file already there.
    ///
    /// Written beside its destination and renamed into place, like
    /// [`Snapshot::save_dhat_v2`].
    pub fn save_folded(&self, path: impl AsRef<Path>, metric: FoldedMetric) -> io::Result<()> {
        self.save_with(path, |snapshot, out| snapshot.write_folded(out, metric))
    }

    /// Writes through `emit` to `path`, replacing what is there only once the
    /// whole file is written.
    ///
    /// A full disk, a write error, or a process killed mid-write leaves the
    /// previous profile intact rather than replacing it with a truncated file
    /// that no viewer will open.
    fn save_with(
        &self,
        path: impl AsRef<Path>,
        emit: impl Fn(&Self, io::BufWriter<std::fs::File>) -> io::Result<()>,
    ) -> io::Result<()> {
        // Held across the whole operation, including the file system calls.
        // The nested acquisition inside the emitter simply finds the guard
        // already held and does nothing, which is the intended behaviour.
        let _quiet = crate::internals::guard::enter();

        let path = path.as_ref();
        let Some(temporary) = temporary_path(path) else {
            // No file name to write beside, so there is nothing to protect.
            return emit(self, io::BufWriter::new(std::fs::File::create(path)?));
        };

        let written = std::fs::File::create(&temporary)
            .and_then(|file| emit(self, io::BufWriter::new(file)))
            .and_then(|()| std::fs::rename(&temporary, path));
        if written.is_err() {
            // Best effort: the original error is what the caller needs to see.
            let _ = std::fs::remove_file(&temporary);
        }
        written
    }

    /// Writes a human-readable summary of the `top` heaviest program points.
    ///
    /// Frames are rendered by [`Symbolized`] and [`Trimmed`]. This is the
    /// output someone reads without opening anything, so it is where a name
    /// earns the most and where a wall of `lang_start` costs the most.
    pub fn write_text_summary<W: Write>(&self, out: W, top: usize) -> io::Result<()> {
        let names = Symbolized::new(&self.modules);
        if self.settings.trim_frames {
            self.write_text_summary_with(out, &Trimmed::new(names), top)
        } else {
            self.write_text_summary_with(out, &names, top)
        }
    }

    /// Writes a human-readable summary, rendering frames with `format`.
    pub fn write_text_summary_with<W: Write>(
        &self,
        out: W,
        format: &dyn FrameFormat,
        top: usize,
    ) -> io::Result<()> {
        let _quiet = crate::internals::guard::enter();
        text::write(self, format, out, top)
    }
}

/// A sibling of `path` to write to before renaming into place.
///
/// The process ID keeps two *processes* writing to the same directory apart.
/// The counter keeps two calls in **one** process apart, which the pid alone did
/// not: two concurrent writes to the same destination — a `Profiler::drop`
/// racing the exit handler, or simply two threads calling
/// [`Snapshot::save_dhat_v2`] — both created and truncated the same temporary
/// file and both wrote into it at independent offsets, producing an interleaved
/// profile that no viewer will open.
///
/// `None` for a path with no file name, which is not something that can be
/// written to anyway.
fn temporary_path(path: &Path) -> Option<std::path::PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static WRITE: AtomicU64 = AtomicU64::new(0);

    let mut name = path.file_name()?.to_os_string();
    name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        WRITE.fetch_add(1, Ordering::Relaxed)
    ));
    Some(path.with_file_name(name))
}

/// Appends `text` to `out`, escaping anything that would let it show a reader
/// something other than what it says.
///
/// <div class="warning">
///
/// `#[doc(hidden)]` and **not part of the supported surface**. It is public
/// only so that `heapscope-symbolize` — a second output path, in this
/// repository, writing names that came out of someone else's symbol table —
/// applies this rule rather than reimplementing it. The rule is what matters;
/// where a copy of it drifts, one of the two stops guarding anything.
///
/// </div>
///
/// # Why the emitters need this and the demangler cannot provide it
///
/// Three of the strings in a profile are written by someone other than us. Frame
/// names come from a symbol table, which on a stripped, truncated, or mismatched
/// binary holds whatever bytes happen to be at the offset a lookup landed on.
/// Image paths come from the filesystem, where a directory may be named
/// anything at all. The command line is `argv`, which is chosen by whoever
/// started the process.
///
/// Left alone, each of those reaches two places that interpret it. A terminal
/// reads C0 and C1 control sequences, so an escape character in a symbol name
/// repaints the summary that was supposed to be reporting it. Every text
/// renderer implements the bidirectional algorithm, so a right-to-left override
/// reverses the display order of everything after it while leaving the bytes
/// untouched — the "trojan source" shape, and a profile is exactly the kind of
/// artifact that is skimmed rather than read closely.
///
/// [`demangle`](fn@crate::demangle) screens what *it* appends, and cannot be the
/// answer here for two reasons: its documented behaviour on input it does not
/// understand is to refuse, leaving the caller to print the raw symbol, and
/// image paths never go near it. So the screen belongs at the point where a
/// string becomes output, which is here, and it covers every
/// [`FrameFormat`] rather than the ones this crate happens to ship.
///
/// Escaped characters are written in Rust's `\u{2066}` form, which is
/// recognisable and searchable. Deliberately not a reversible encoding: making
/// it one would mean escaping every backslash, which would turn every Windows
/// path in the profile into `C:\\Program Files\\...` to guard against a case
/// that costs nothing here. What is guaranteed is that the output *contains* no
/// character from the escaped set, which is the property that matters, not that
/// the original can be reconstructed from it.
#[doc(hidden)]
pub fn push_display(out: &mut String, text: &str) {
    for character in text.chars() {
        if is_safe_to_display(character) {
            out.push(character);
            continue;
        }
        out.push_str("\\u{");
        // Lower-case hexadecimal without leading zeroes, matching the literal
        // syntax the escape is borrowed from.
        let mut digits = [0u8; 6];
        let mut written = 0;
        let mut value = character as u32;
        while value > 0 || written == 0 {
            digits[written] = b"0123456789abcdef"[(value & 0xF) as usize];
            value >>= 4;
            written += 1;
        }
        for &digit in digits[..written].iter().rev() {
            out.push(digit as char);
        }
        out.push('}');
    }
}

/// Whether a character means only itself.
///
/// Everything outside this is escaped by [`push_display`]. The set is small and
/// specific rather than an allowlist of printable characters, because a profile
/// legitimately contains non-ASCII text: a path under a user's name, a crate
/// with an accented author, `µs` in a unit. Refusing those would make the output
/// worse for the common case in order to guard against the rare one.
fn is_safe_to_display(character: char) -> bool {
    // `is_control` is the Unicode `Cc` category: C0, DEL, and C1. That is the
    // terminal-escape half.
    if character.is_control() {
        return false;
    }
    // The bidirectional formatting characters, which are the reordering half.
    // U+200E and U+200F are marks rather than overrides and cannot by
    // themselves reorder a run, but they are invisible and belong to the same
    // mechanism, and nothing that reaches this function has a use for one.
    !matches!(
        character,
        '\u{061C}'
            | '\u{200E}'
            | '\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2066}'..='\u{2069}'
            // Line separators. JSON tolerates them and JavaScript does not, so
            // the JSON writer escapes them for validity; they are here as well
            // because a line break in the middle of a frame name breaks the
            // text summary's layout too.
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// Renders a row's name, or `None` where the platform or the program gave none.
///
/// Lossy rather than fallible: a thread name the platform hands back is bytes,
/// with no encoding promised, and a mangled name still identifies a thread
/// where no name at all does not.
fn name_of(raw: &[u8]) -> Option<String> {
    if raw.is_empty() {
        return None;
    }
    Some(String::from_utf8_lossy(raw).into_owned())
}

/// `argv`, joined by spaces.
///
/// Lossy on purpose: a path that is not valid UTF-8 is worth reporting with
/// replacement characters rather than not reporting at all, and the field is
/// descriptive rather than something anything parses.
fn command_line() -> String {
    let mut command = String::new();
    for argument in std::env::args_os() {
        if !command.is_empty() {
            command.push(' ');
        }
        command.push_str(&argument.to_string_lossy());
    }
    command
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internals::engine::Engine;
    use crate::internals::shape::Shape;

    /// Releases a spinning worker when it drops, so an assertion failure inside a
    /// `thread::scope` fails the test rather than hanging it. See the copy in
    /// `core::engine`'s tests for what that cost when it was missing.
    struct StopOnDrop<'a>(&'a std::sync::atomic::AtomicBool);

    impl Drop for StopOnDrop<'_> {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The same program, loaded at two different addresses, snapshots its
    /// program points in the same order.
    ///
    /// This is the whole of PLAN.md section 8.1's reproducibility claim at its
    /// source. Every emitter writes `Snapshot::points` in the order it finds
    /// them or breaks its own ties by that order, so if this holds the files
    /// hold, and if it does not no emitter can be fixed without fixing it
    /// twice.
    ///
    /// The two bases stand in for two executions of one binary under address
    /// space layout randomization. They are not decoration: a point's shard
    /// comes from hashing its return addresses, `flush_and_visit` walks shards
    /// in index order, and so the order the engine offers its points in *is* a
    /// reading of where the program was mapped. The assertion below that the
    /// two bases visit in different orders is what keeps this test honest — two
    /// bases that happened to agree would let any implementation pass.
    #[test]
    fn points_are_ordered_by_the_run_and_not_by_where_the_program_was_mapped() {
        use crate::internals::pp::{hash_frames, SHARDS};

        const SITES: usize = 24;
        const LOW: usize = 0x1000_0000;
        const HIGH: usize = 0x7654_0000;

        // One call site, as the program would reach it from a given base.
        fn stack(base: usize, site: usize) -> [usize; 2] {
            [base + site * 0x40, base + 0x8000]
        }

        // The order `PpTable::flush_and_visit` would offer the sites in: shards
        // by index, and within a shard the order they were interned.
        fn visit_order(base: usize) -> Vec<usize> {
            let mut sites: Vec<usize> = (0..SITES).collect();
            sites.sort_by_key(|&site| (hash_frames(&stack(base, site)) as usize) & (SHARDS - 1));
            sites
        }
        assert_ne!(
            visit_order(LOW),
            visit_order(HIGH),
            "the two bases hash into the same visiting order, so this test would \
             pass without ordering anything"
        );

        // Every site allocates the same amount, so no counter can stand in for
        // the ordering under test.
        fn run(engine: &Engine, base: usize) -> Vec<Vec<usize>> {
            engine.start(TimeSource::Events, || {});
            for site in 0..SITES {
                engine.record_alloc_guarded(
                    0x1_0000 + site * 64,
                    Shape::of(64),
                    &stack(base, site),
                );
            }
            let snapshot = Snapshot::of(engine);
            engine.stop(crate::internals::engine::Shutdown::Explicit);
            snapshot
                .points
                .iter()
                .map(|point| point.frames.iter().map(|frame| frame - base).collect())
                .collect()
        }

        static FIRST: Engine = Engine::new();
        static SECOND: Engine = Engine::new();
        let low = run(&FIRST, LOW);
        let high = run(&SECOND, HIGH);

        assert_eq!(low.len(), SITES, "a site was lost before the comparison");
        assert_eq!(
            low, high,
            "two runs of one program put their points in different orders, so no \
             two profiles of it can be diffed"
        );
    }

    /// The live-block table's row count agrees with the counter that says how
    /// many blocks are live.
    ///
    /// [`SelfMetrics::live_blocks`] documents exactly this expectation — "in a
    /// stopped run that dropped nothing" — and it is what a reader checks a
    /// `droppedBlocks` count against. Nothing else makes the two agree: the
    /// counter is maintained on the hot path under the peak gate, and the row
    /// count is read from the table *outside* the flush window, deliberately,
    /// because taking the gate for a descriptive number would make every profile
    /// pay for a consistency nothing reads.
    ///
    /// Frees are interleaved so the two numbers have to agree about the
    /// survivors rather than about a count of allocations, which is the same
    /// number when nothing has been freed.
    #[test]
    fn the_live_block_row_count_agrees_with_the_live_block_counter() {
        // Bounded under Miri, like the snapshot test below: the interpreter
        // walks every row of the live-block table on each capture. Enough
        // blocks to leave the table with holes is all the claim needs.
        #[cfg(miri)]
        const BLOCKS: usize = 24;
        #[cfg(not(miri))]
        const BLOCKS: usize = 512;
        const SIZE: usize = 96;

        let engine = Engine::with_limits(1 << 10, 1 << 12);
        assert!(engine.start(TimeSource::Events, || {}));
        for block in 0..BLOCKS {
            engine.record_alloc_guarded(0x1_0000 + block * 128, Shape::of(SIZE), &[0xA0, 0xB0]);
        }
        for block in (0..BLOCKS).step_by(3) {
            engine.record_free(0x1_0000 + block * 128, SIZE);
        }
        engine.stop(crate::internals::engine::Shutdown::Explicit);

        let snapshot = Snapshot::of(&engine);
        assert_eq!(
            snapshot.stats.dropped_blocks, 0,
            "the table refused a block, which is the one case the claim excludes"
        );
        assert!(
            snapshot.stats.curr_blocks > 0,
            "nothing was left alive, so the comparison below is between two zeroes"
        );
        assert_eq!(
            snapshot.metrics.live_blocks.entries as u64, snapshot.stats.curr_blocks,
            "the table holds {} rows and the counter says {} blocks are live, so a \
             reader cannot tell a full table from a leaking one",
            snapshot.metrics.live_blocks.entries, snapshot.stats.curr_blocks
        );
    }

    /// Reading a profile twice reads the same profile.
    ///
    /// [`ProgramPoint::unretired_lifetime`] says why this is not free: DHAT's
    /// `tl` counts every block's lifetime, live ones included, and Valgrind gets
    /// there by *retiring* every live block at exit. Retiring mutates the thing
    /// being measured, so a second reading of a retired engine is a different
    /// profile — the live blocks are gone from one total and doubled into
    /// another. This crate adds the two totals on demand instead, and the whole
    /// benefit of that decision is the assertion below.
    ///
    /// The live blocks are the point of the test, so it fails rather than passes
    /// if the run left none behind: with nothing unretired there is nothing that
    /// retiring could have moved, and any implementation would pass.
    #[test]
    fn snapshotting_a_stopped_engine_twice_gives_the_same_profile() {
        // Two captures, each sweeping the table: bounded under Miri for the
        // same reason, and four sites still give every point both a freed and
        // an unfreed block.
        #[cfg(miri)]
        const BLOCKS: usize = 16;
        #[cfg(not(miri))]
        const BLOCKS: usize = 256;
        const SIZE: usize = 128;

        let engine = Engine::with_limits(1 << 10, 1 << 12);
        assert!(engine.start(TimeSource::Events, || {}));
        for block in 0..BLOCKS {
            let site = 0xC0 + (block % 4) * 0x10;
            engine.record_alloc_guarded(0x2_0000 + block * 256, Shape::of(SIZE), &[site, 0xB0]);
        }
        // Half freed, half left alive, so both lifetime totals are non-zero and
        // a fold of one into the other would show.
        for block in (0..BLOCKS).step_by(2) {
            engine.record_free(0x2_0000 + block * 256, SIZE);
        }
        engine.stop(crate::internals::engine::Shutdown::Explicit);

        let first = Snapshot::of(&engine);
        let second = Snapshot::of(&engine);

        assert!(
            first
                .points
                .iter()
                .any(|point| point.unretired_lifetime > 0),
            "no block was still alive, so nothing here could have been retired"
        );
        assert!(
            first
                .points
                .iter()
                .any(|point| point.counters.total_lifetime > 0),
            "no block was freed, so the total this one is kept apart from is empty"
        );

        let lifetimes = |snapshot: &Snapshot| -> Vec<(u64, u64, u64)> {
            snapshot
                .points
                .iter()
                .map(|point| {
                    (
                        point.counters.total_lifetime,
                        point.unretired_lifetime,
                        point.total_lifetime(),
                    )
                })
                .collect()
        };
        assert_eq!(
            lifetimes(&first),
            lifetimes(&second),
            "the second reading of a stopped engine disagrees with the first, so \
             taking a snapshot changed what was being measured"
        );

        let render = |snapshot: &Snapshot| -> Vec<u8> {
            let mut out = Vec::new();
            snapshot.write_dhat_v2(&mut out).expect("writing to a Vec");
            out
        };
        assert_eq!(
            render(&first),
            render(&second),
            "two profiles of one stopped run are not the same file"
        );
    }

    /// A snapshot taken of a **running** engine still has rows that sum to its
    /// totals, because both were read in one exclusive window.
    ///
    /// This is the case the window exists for, and the only one that can show
    /// it: a snapshot taken after `Profiler::drop` has stopped the engine and
    /// drained the gate has nothing in flight, so reading the rows a
    /// millisecond later — after the frames are copied and the live-block table
    /// swept — changes no number there. Here threads allocate throughout, and
    /// that same millisecond is thousands of events.
    #[test]
    fn a_running_engine_snapshots_rows_that_sum_to_its_totals() {
        use std::sync::atomic::{AtomicBool, Ordering};

        // Bounded on both sides under Miri. Each snapshot sweeps the live-block
        // table, and an unbounded worker would keep growing the table the
        // interpreter then has to walk.
        #[cfg(miri)]
        const SNAPSHOTS: usize = 3;
        #[cfg(not(miri))]
        const SNAPSHOTS: usize = 50;
        #[cfg(miri)]
        const ROUNDS: usize = 32;
        #[cfg(not(miri))]
        const ROUNDS: usize = usize::MAX;

        let engine = Engine::with_limits(1 << 12, 1 << 16);
        assert!(engine.start(TimeSource::Events, || {}));
        let stop = AtomicBool::new(false);

        std::thread::scope(|scope| {
            for worker in 0..4usize {
                let (engine, stop) = (&engine, &stop);
                scope.spawn(move || {
                    let base = 0x1_0000_0000usize + worker * 0x1000_0000;
                    let mut round = 0usize;
                    while !stop.load(Ordering::Relaxed) && round < ROUNDS {
                        let address = base + (round % 1024) * 128;
                        let size = 64 + round % 512;
                        engine.record_alloc_guarded(address, Shape::of(size), &[0xA0 + worker]);
                        if round.is_multiple_of(3) {
                            engine.record_free(address, size);
                        }
                        round += 1;
                    }
                });
            }

            // Releases the workers however this scope is left, including by a
            // failing assertion below: `thread::scope` joins on unwind, and a
            // worker looping until a flag nobody sets is never joinable, so the
            // test would hang rather than fail.
            let _release = StopOnDrop(&stop);

            let mut checked = 0;
            for _ in 0..SNAPSHOTS {
                let snapshot = Snapshot::of(&engine);
                if !snapshot.exact || snapshot.rows_dropped != 0 {
                    continue;
                }
                let bytes: u64 = snapshot
                    .threads
                    .iter()
                    .map(|row| row.counts.total_bytes)
                    .sum();
                let blocks: u64 = snapshot
                    .threads
                    .iter()
                    .map(|row| row.counts.total_blocks)
                    .sum();
                let live: u64 = snapshot
                    .threads
                    .iter()
                    .map(|row| row.counts.curr_bytes)
                    .sum();
                assert_eq!(
                    bytes, snapshot.stats.total_bytes,
                    "the thread rows describe a different instant from the totals"
                );
                assert_eq!(blocks, snapshot.stats.total_blocks);
                assert_eq!(live, snapshot.stats.curr_bytes);
                checked += 1;
            }
            assert!(
                checked > 0,
                "no snapshot reached a quiet point, so nothing was checked"
            );
        });
    }

    /// What a run was configured with has to reach the snapshot, because that is
    /// what the emitters taking no argument read and what the profile reports.
    ///
    /// This is the half that can be proved without a process: whether the
    /// rendering actually changes is asserted end to end, in `tests/lifecycle.rs`,
    /// where there are real names for the rules to act on. On a platform that
    /// names no frames the two renderings are byte-identical by construction, so
    /// a unit test asserting they differ would be asserting the platform.
    #[test]
    fn the_settings_a_run_was_given_travel_with_its_snapshot() {
        static ENGINE: Engine = Engine::new();
        ENGINE.start(TimeSource::Events, || {
            ENGINE.configure(crate::internals::engine::Settings {
                mode: crate::Mode::Heap,
                max_depth: 8,
                max_live_blocks: 1 << 14,
                trim_frames: false,
                sampling: None,
            })
        });
        let snapshot = Snapshot::of(&ENGINE);
        ENGINE.stop(crate::internals::engine::Shutdown::Explicit);

        assert_eq!(snapshot.settings.max_depth, 8);
        assert_eq!(snapshot.settings.max_live_blocks, 1 << 14);
        assert!(
            !snapshot.settings.trim_frames,
            "`trim_frames(false)` did not reach the snapshot, so the emitters \
             that take no renderer would trim anyway"
        );
        assert!(
            Settings::default().trim_frames,
            "trimming is the documented default"
        );
    }

    /// A snapshot of an engine that recorded nothing still has to be coherent:
    /// this is the profile written by a program that starts and exits.
    #[test]
    fn an_empty_engine_snapshots_cleanly() {
        static ENGINE: Engine = Engine::new();
        ENGINE.start(TimeSource::Events, || {});
        let snapshot = Snapshot::of(&ENGINE);
        ENGINE.stop(crate::internals::engine::Shutdown::Explicit);

        assert!(snapshot.points.is_empty());
        assert_eq!(snapshot.points_dropped, 0);
        assert_eq!(snapshot.unattributed_blocks, 0);
        assert_eq!(snapshot.stats.total_bytes, 0);
        assert!(snapshot.exact);
    }

    /// PLAN.md section 4.6: table capacity exhausted must surface as a synthetic
    /// `[overflow]` point *visible in the output*.
    ///
    /// The emitter side of that was tested against hand-built snapshots with
    /// `kind` already set, which proved nothing about the only line that decides
    /// it. Replacing that line with a constant `PointKind::Recorded` left the
    /// whole suite green, and a real overflow would have been labelled
    /// "the stack could not be walked" — telling the reader to check their build
    /// flags when the true answer is "raise the ceiling".
    #[test]
    fn overflowing_the_program_point_table_produces_a_point_that_says_so() {
        // Small enough that the per-shard ceiling is reached quickly, and a
        // live-block table roomy enough that blocks are not dropped first.
        static ENGINE: Engine = Engine::with_limits(crate::internals::pp::SHARDS * 2, 1 << 16);
        ENGINE.start(TimeSource::Events, || {});

        for i in 1..crate::internals::miri_scale(20_000) {
            ENGINE.record_alloc_guarded(0x8000_0000 + i * 64, Shape::of(64), &[i, i * 3, i * 7]);
        }
        let snapshot = Snapshot::of(&ENGINE);
        ENGINE.stop(crate::internals::engine::Shutdown::Explicit);

        let overflow: Vec<_> = snapshot
            .points
            .iter()
            .filter(|point| point.kind == PointKind::Overflow)
            .collect();
        assert_eq!(
            overflow.len(),
            1,
            "an overflowing run produced {} overflow points, not one",
            overflow.len()
        );
        assert!(
            overflow[0].counters.total_blocks > 0,
            "the overflow point absorbed nothing, so the run never overflowed"
        );
        assert!(
            overflow[0].frames.is_empty(),
            "the overflow point has no frames by construction"
        );
    }

    #[test]
    fn a_snapshot_carries_the_frames_and_counters_of_every_point() {
        static ENGINE: Engine = Engine::new();
        ENGINE.start(TimeSource::Events, || {});
        ENGINE.record_alloc_guarded(0x1000, Shape::of(64), &[0xAA, 0xBB]);
        ENGINE.record_alloc_guarded(0x2000, Shape::of(32), &[0xCC]);
        ENGINE.record_free(0x2000, 32);
        let snapshot = Snapshot::of(&ENGINE);
        ENGINE.stop(crate::internals::engine::Shutdown::Explicit);

        assert_eq!(snapshot.points.len(), 2);
        let allocated: u64 = snapshot.points.iter().map(|p| p.counters.total_bytes).sum();
        assert_eq!(allocated, 96);

        let held = snapshot
            .points
            .iter()
            .find(|p| p.frames == [0xAA, 0xBB])
            .expect("the two-frame point");
        assert_eq!(held.counters.curr_bytes, 64);

        let freed = snapshot
            .points
            .iter()
            .find(|p| p.frames == [0xCC])
            .expect("the one-frame point");
        assert_eq!(freed.counters.curr_bytes, 0);
    }

    /// The reason the live sweep exists: a block that is never freed must still
    /// contribute its lifetime, or every leak looks short-lived.
    #[test]
    fn blocks_still_alive_contribute_their_lifetime() {
        static ENGINE: Engine = Engine::new();
        ENGINE.start(TimeSource::Events, || {});
        // Born at event 1, and the clock advances with each later allocation.
        ENGINE.record_alloc_guarded(0x1000, Shape::of(64), &[0xAA]);
        for address in 0..8 {
            ENGINE.record_alloc_guarded(0x2000 + address * 16, Shape::of(16), &[0xBB]);
        }
        let snapshot = Snapshot::of(&ENGINE);
        ENGINE.stop(crate::internals::engine::Shutdown::Explicit);

        let held = snapshot
            .points
            .iter()
            .find(|p| p.frames == [0xAA])
            .expect("the point that still holds a block");
        assert_eq!(
            held.counters.total_lifetime, 0,
            "nothing was freed, so the engine's own lifetime total is zero"
        );
        assert_eq!(
            held.unretired_lifetime, 8,
            "the block was alive for the eight allocation events that followed"
        );
        assert_eq!(held.total_lifetime(), 8);
    }

    #[test]
    fn capturing_twice_gives_the_same_answer() {
        // The live sweep must not mutate engine state: a profile written twice
        // has to report the same numbers both times.
        static ENGINE: Engine = Engine::new();
        ENGINE.start(TimeSource::Events, || {});
        ENGINE.record_alloc_guarded(0x1000, Shape::of(64), &[0xAA]);
        ENGINE.record_alloc_guarded(0x2000, Shape::of(16), &[0xBB]);
        ENGINE.record_free(0x2000, 16);

        let first = Snapshot::of(&ENGINE);
        let second = Snapshot::of(&ENGINE);
        ENGINE.stop(crate::internals::engine::Shutdown::Explicit);

        assert_eq!(first.points, second.points);
        assert_eq!(first.stats, second.stats);
        assert_eq!(first.time_at_end, second.time_at_end);
    }

    #[test]
    fn the_command_line_is_recorded() {
        let command = command_line();
        assert!(
            !command.is_empty(),
            "argv[0] is always present for a test binary"
        );
    }

    // ---- the display screen ----

    fn screened(text: &str) -> String {
        let mut out = String::new();
        push_display(&mut out, text);
        out
    }

    #[test]
    fn ordinary_text_passes_through_untouched() {
        for text in [
            "core::fmt::write",
            "/Users/someone/Develop/a project/target/debug/program",
            "<alloc::vec::Vec<T,A> as core::ops::drop::Drop>::drop",
            "µs \u{2026} \u{1F600}",
            "C:\\Program Files\\thing.exe",
            "",
        ] {
            assert_eq!(screened(text), text);
        }
    }

    /// The terminal half. A symbol table read at the wrong offset is bytes, and
    /// bytes include the one that starts an escape sequence.
    #[test]
    fn control_characters_are_escaped() {
        assert_eq!(screened("a\u{1b}[2Jb"), "a\\u{1b}[2Jb");
        assert_eq!(screened("a\nb\tc\rd"), "a\\u{a}b\\u{9}c\\u{d}d");
        assert_eq!(screened("a\u{0}b"), "a\\u{0}b");
        assert_eq!(screened("a\u{7f}b"), "a\\u{7f}b");
        // C1, which arrives as two UTF-8 bytes and is still a control.
        assert_eq!(screened("a\u{9b}b"), "a\\u{9b}b");
    }

    /// The reordering half: trojan source. Nothing about the bytes of
    /// `alloc\u{202e}...` says it will display backwards.
    #[test]
    fn bidirectional_overrides_are_escaped() {
        for character in [
            '\u{061C}', '\u{200E}', '\u{200F}', '\u{202A}', '\u{202B}', '\u{202C}', '\u{202D}',
            '\u{202E}', '\u{2066}', '\u{2067}', '\u{2068}', '\u{2069}',
        ] {
            let text = format!("a{character}b");
            let out = screened(&text);
            assert!(
                !out.contains(character),
                "{character:?} survived the screen as `{out}`"
            );
            assert!(out.starts_with('a') && out.ends_with('b'));
        }
    }

    #[test]
    fn the_javascript_line_terminators_are_escaped() {
        assert_eq!(screened("a\u{2028}b\u{2029}c"), "a\\u{2028}b\\u{2029}c");
    }

    /// The escape is Rust's, so it has to look like Rust's: lower case, no
    /// leading zeroes, always at least one digit.
    #[test]
    fn escapes_are_written_the_way_rust_writes_them() {
        assert_eq!(screened("\u{0}"), "\\u{0}");
        assert_eq!(screened("\u{1}"), "\\u{1}");
        assert_eq!(screened("\u{1f}"), "\\u{1f}");
        assert_eq!(screened("\u{202e}"), "\\u{202e}");
        // Every escape this can produce is a valid Rust character literal, and
        // the ones above are the widest it reaches: the escaped set has no
        // member above U+2069.
        assert!(screened("\u{9f}").is_ascii());
    }

    /// The output of the screen must have nothing left to screen.
    #[test]
    fn screening_is_idempotent() {
        for text in [
            "a\u{1b}[2Jb",
            "\u{202e}gnp.eslaf",
            "plain",
            "\u{0}\u{202e}\u{2028}",
        ] {
            let once = screened(text);
            assert_eq!(screened(&once), once);
        }
    }

    /// Escaping must not lose the characters around what it escaped, and it
    /// must not run past the end of a multi-byte character.
    #[test]
    fn nothing_but_the_offending_character_is_disturbed() {
        assert_eq!(screened("é\u{202e}é"), "é\\u{202e}é");
        assert_eq!(screened("\u{1F600}\u{0}"), "\u{1F600}\\u{0}");
    }
}
