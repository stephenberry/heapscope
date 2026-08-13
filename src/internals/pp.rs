//! Program points: interned call stacks and their counters.
//!
//! A *program point* is a distinct call stack that allocated. Every allocation
//! is attributed to one, and the profile is essentially a table of them.
//!
//! # The lazy-epoch algorithm
//!
//! DHAT reports, per program point, the live bytes and blocks at the instant the
//! whole process hit its peak — `gb`/`gbk` in the file format. Valgrind and
//! `dhat-rs` both compute this by sweeping *every* program point on *every* new
//! peak, which is `O(#PPs)` per peak and, during a monotonically growing phase,
//! `O(#PPs)` per allocation.
//!
//! This does it in `O(1)` amortised. A global epoch counter is bumped whenever a
//! new peak is set. Each record remembers the epoch at which it was last
//! touched. On any touch:
//!
//! 1. If the record's epoch is stale, the process peaked at some point since
//!    this record last changed — so the record's *current* values are exactly
//!    what they were at the peak. Copy them into the at-peak fields.
//! 2. Bring the epoch up to date.
//! 3. Apply the change.
//!
//! At output, every record still holding a stale epoch is flushed the same way.
//!
//! # The epoch must bump on `>=`, not `>`
//!
//! Valgrind is explicit about this (`dh_main.c:373-379`): *"The use of `>=`
//! rather than `>` means that if there are multiple equal peaks we record the
//! latest one."* PLAN.md section 4.3 records a model check over 200,000 random
//! traces comparing this scheme against Valgrind's eager sweep:
//!
//! ```text
//! epoch-on->=:  at_gmax mismatches = 0 / 199999      tg mismatches = 0
//! epoch-on->:   at_gmax mismatches = 12110 / 199999  tg mismatches = 12929
//! ```
//!
//! With `>=` the lazy scheme is *exactly* equivalent to eager snapshotting. The
//! bump itself lives in [`super::engine`], which owns the peak; this module owns
//! the reaction to it.
//!
//! # `mb`/`mbk` diverge from Valgrind, deliberately
//!
//! Valgrind assigns a program point's maximum only inside
//! `if (g_curr_bytes >= g_max_bytes)`, so a site that peaked at 4 MB while the
//! whole heap was small records a maximum of zero — despite its own struct
//! comment claiming otherwise. PLAN.md decision 10.3 chooses a true per-point
//! running maximum instead. It is `O(1)`, it is what the field's name means, and
//! the divergence is documented in the README and the output header.

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

use super::arena::{Arena, ArenaVec};
use super::lock::RawLock;
use super::table::{Insert, RawMap};
use super::CachePadded;

/// Number of shards. A compile-time constant so the shard array is a plain
/// `static` with no lazy initialization reachable from the allocator.
pub const SHARDS: usize = 64;

const _: () = assert!(SHARDS.is_power_of_two());

/// Default ceiling on distinct program points.
///
/// Beyond this, allocations are attributed to [`PpId::OVERFLOW`] and the profile
/// says so. A real program has thousands; a million is generous enough that
/// reaching it indicates a pathology worth seeing in the output.
pub const DEFAULT_MAX_PROGRAM_POINTS: usize = 1 << 20;

/// Identifies an interned program point.
///
/// The encoding is `shard + index * SHARDS`, so the shard is recoverable with a
/// mask and no side table.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PpId(u32);

impl PpId {
    /// The synthetic program point that absorbs allocations once the table is
    /// full.
    ///
    /// Kept outside the shards entirely, with its own record and lock. Making
    /// it a normal interned entry would mean it had to exist before the first
    /// allocation, which would put lazy initialization on the path that must
    /// not have any.
    pub const OVERFLOW: PpId = PpId(u32::MAX);

    /// Which shard owns this program point.
    #[inline(always)]
    pub fn shard(self) -> usize {
        self.0 as usize & (SHARDS - 1)
    }

    #[inline(always)]
    fn index(self) -> usize {
        self.0 as usize / SHARDS
    }

    fn new(shard: usize, index: usize) -> Option<Self> {
        let raw = shard.checked_add(index.checked_mul(SHARDS)?)?;
        let raw = u32::try_from(raw).ok()?;
        if raw == PpId::OVERFLOW.0 {
            return None;
        }
        Some(PpId(raw))
    }

    /// The raw value, for output and for stable ordering.
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// Rebuilds an id from its raw value.
    ///
    /// Test-only, and crate-internal: an id that did not come from
    /// [`PpTable::intern`] does not name a record, and [`PpTable::update`]
    /// reports a poison rather than panicking if handed one. This exists so
    /// that the live-block table can be tested without interning real stacks.
    #[cfg(test)]
    pub(crate) const fn from_raw(raw: u32) -> Self {
        PpId(raw)
    }
}

impl fmt::Debug for PpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == PpId::OVERFLOW {
            f.write_str("PpId(overflow)")
        } else {
            write!(f, "PpId({}:{})", self.shard(), self.index())
        }
    }
}

/// Everything recorded about one program point.
///
/// Plain fields rather than atomics: every access happens under the owning
/// shard's lock, so atomics would buy nothing and cost a read-modify-write per
/// field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Bytes ever allocated here. DHAT's `tb`.
    pub total_bytes: u64,
    /// Blocks ever allocated here. DHAT's `tbk`.
    pub total_blocks: u64,
    /// Summed lifetime of blocks freed here. DHAT's `tl`.
    ///
    /// Emitted but never validated by `dh_view.js`; omitting it renders every
    /// average-lifetime cell as `NaN` with no warning (PLAN.md section 3.1).
    pub total_lifetime: u64,

    /// Bytes currently live. Becomes DHAT's `eb` at end of run.
    pub curr_bytes: u64,
    /// Blocks currently live. Becomes DHAT's `ebk`.
    pub curr_blocks: u64,

    /// Greatest `curr_bytes` ever reached. DHAT's `mb`; see the module docs for
    /// why this differs from Valgrind's.
    pub max_bytes: u64,
    /// Greatest `curr_blocks` ever reached. DHAT's `mbk`.
    pub max_blocks: u64,

    /// Bytes live when the whole heap peaked. DHAT's `gb`.
    pub at_gmax_bytes: u64,
    /// Blocks live when the whole heap peaked. DHAT's `gbk`.
    pub at_gmax_blocks: u64,
}

#[derive(Clone, Copy)]
struct Record {
    /// Frames, innermost first, copied into the arena at interning.
    frames: *const usize,
    frames_len: u32,
    /// Position in the order the program first reached its program points.
    ///
    /// This is the only slide-independent identity a record has, and it is what
    /// [`PpTable::sequence`] exists to hand out. It occupies padding that
    /// `frames_len` already left behind, so the record does not grow.
    sequence: u32,
    counters: Counters,
    /// Epoch at which `at_gmax_*` was last refreshed.
    snapshot_epoch: u64,
}

// SAFETY: `frames` points into the arena, which outlives every record and is
// never mutated after interning. Access is serialized by the shard lock.
unsafe impl Send for Record {}
// SAFETY: as above.
unsafe impl Sync for Record {}

impl Record {
    const fn empty() -> Self {
        Self {
            frames: std::ptr::null(),
            frames_len: 0,
            // The overflow record is built from this and never interned, so it
            // never receives a sequence. Last is where it belongs: it is the
            // synthetic point standing in for everything that did not fit.
            sequence: u32::MAX,
            counters: Counters {
                total_bytes: 0,
                total_blocks: 0,
                total_lifetime: 0,
                curr_bytes: 0,
                curr_blocks: 0,
                max_bytes: 0,
                max_blocks: 0,
                at_gmax_bytes: 0,
                at_gmax_blocks: 0,
            },
            snapshot_epoch: 0,
        }
    }

    fn frames(&self) -> &[usize] {
        if self.frames.is_null() || self.frames_len == 0 {
            return &[];
        }
        // SAFETY: `frames` and `frames_len` were set together from an arena
        // slice that is never freed or mutated while the table lives.
        unsafe { std::slice::from_raw_parts(self.frames, self.frames_len as usize) }
    }

    /// Brings `at_gmax_*` up to date before applying a change.
    ///
    /// This is the whole lazy-epoch trick. A stale epoch means the heap peaked
    /// at some moment after this record last changed, so the record's current
    /// values *are* its values at that peak.
    #[inline(always)]
    fn refresh(&mut self, epoch: u64) {
        if self.snapshot_epoch != epoch {
            self.counters.at_gmax_bytes = self.counters.curr_bytes;
            self.counters.at_gmax_blocks = self.counters.curr_blocks;
            self.snapshot_epoch = epoch;
        }
    }
}

struct ShardState {
    /// Frame hash to index within `records`.
    intern: RawMap<u32>,
    records: ArenaVec<Record>,
}

struct Shard {
    lock: RawLock,
    state: std::cell::UnsafeCell<ShardState>,
}

// SAFETY: `state` is reached only while holding `lock`.
unsafe impl Sync for Shard {}

impl Shard {
    const fn new(max_points: usize) -> Self {
        Self {
            lock: RawLock::new(),
            state: std::cell::UnsafeCell::new(ShardState {
                // Intern slots are capped at four per record so that a table
                // full of collisions still terminates its probes.
                intern: RawMap::new((max_points / SHARDS).next_power_of_two() * 4),
                records: ArenaVec::new(max_points / SHARDS),
            }),
        }
    }
}

/// The interned program-point table.
pub struct PpTable {
    shards: [CachePadded<Shard>; SHARDS],
    /// The synthetic point that absorbs allocations once the table is full.
    overflow_lock: RawLock,
    overflow: std::cell::UnsafeCell<Record>,
    /// Handed to the next record created, in order, across all shards.
    ///
    /// Shared rather than per-shard because its whole purpose is to order
    /// records that live in *different* shards. Touched once per distinct
    /// program point and never on a repeat allocation, so it is a cold-path
    /// counter: a program with 5,000 call sites increments it 5,000 times in
    /// its life, whatever its allocation count.
    next_sequence: AtomicU32,
}

// SAFETY: every field is reached only under its own lock.
unsafe impl Sync for PpTable {}

/// What happened when a stack was interned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interned {
    /// An existing program point matched.
    Existing(PpId),
    /// A new program point was created.
    Created(PpId),
    /// The table is full; the caller should use [`PpId::OVERFLOW`].
    Overflowed,
}

impl Interned {
    /// The id to attribute the allocation to.
    pub fn id(self) -> PpId {
        match self {
            Interned::Existing(id) | Interned::Created(id) => id,
            Interned::Overflowed => PpId::OVERFLOW,
        }
    }
}

impl PpTable {
    /// Creates an empty table with the default ceiling.
    pub const fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_PROGRAM_POINTS)
    }

    /// Creates an empty table holding at most `max_points` program points.
    pub const fn with_capacity(max_points: usize) -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const fn shards(max_points: usize) -> [CachePadded<Shard>; SHARDS] {
            // `Shard` is not `Copy`, so the array cannot be built with `[x; N]`.
            // A const fn plus a loop keeps this a compile-time constant.
            // SAFETY: an array of `MaybeUninit` needs no initialization to be
            // valid -- each element is allowed to be uninhabited until written.
            // This is the documented idiom for building an array of a non-`Copy`
            // type element by element.
            let mut array: [std::mem::MaybeUninit<CachePadded<Shard>>; SHARDS] =
                unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            let mut index = 0;
            while index < SHARDS {
                array[index] = std::mem::MaybeUninit::new(CachePadded::new(Shard::new(max_points)));
                index += 1;
            }
            // SAFETY: the loop above wrote every element, and
            // `[MaybeUninit<T>; N]` has the same size and alignment as
            // `[T; N]`.
            unsafe { std::mem::transmute::<_, [CachePadded<Shard>; SHARDS]>(array) }
        }

        Self {
            shards: shards(max_points),
            overflow_lock: RawLock::new(),
            overflow: std::cell::UnsafeCell::new(Record::empty()),
            next_sequence: AtomicU32::new(0),
        }
    }

    /// Finds or creates the program point for `frames`.
    ///
    /// `frames` is the raw return-address array from the unwinder, innermost
    /// first. It is copied into the arena only when a new point is created.
    pub fn intern(&self, arena: &Arena, frames: &[usize]) -> Interned {
        let hash = hash_frames(frames);
        let shard_index = (hash as usize) & (SHARDS - 1);
        let shard = &self.shards[shard_index];

        let _order = super::order::enter(super::order::Level::ProgramPointShard);
        let _guard = shard.lock.lock();
        // SAFETY: `state` is reached only while holding `shard.lock`.
        let state = unsafe { &mut *shard.state.get() };

        let key = RawMap::<u32>::usable_key(hash);
        if let Some(index) = state.intern.get(key) {
            // A hash match is not an identity match. Comparing the frames is
            // what stops two different call stacks from being merged into one
            // program point, which would be silent misattribution.
            if let Some(record) = state.records.get(index as usize) {
                if record.frames() == frames {
                    if let Some(id) = PpId::new(shard_index, index as usize) {
                        return Interned::Existing(id);
                    }
                }
            }
        }

        let Some(stored) = arena.alloc_slice(frames) else {
            return Interned::Overflowed;
        };
        // Claimed before the record is pushed, so that two threads creating
        // points in different shards are ordered by which of them got here
        // first. A gap left by a creation that goes on to fail below costs
        // nothing: the sequence is read only for ordering, never as an index.
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let record = Record {
            frames: stored.as_ptr(),
            frames_len: stored.len() as u32,
            sequence,
            ..Record::empty()
        };
        let Some(index) = state.records.push(arena, record) else {
            return Interned::Overflowed;
        };
        let Ok(index_u32) = u32::try_from(index) else {
            return Interned::Overflowed;
        };
        let Some(id) = PpId::new(shard_index, index) else {
            return Interned::Overflowed;
        };

        // A colliding hash overwrites the older entry's index. The consequence
        // is worse than "a duplicate program point": two stacks that collide
        // and allocate alternately create a *new record on every allocation*,
        // because each lookup finds the other's index, fails the frame
        // comparison, and pushes again. Record creation becomes unbounded until
        // the ceiling is hit.
        //
        // The probability is what makes this acceptable rather than the
        // consequence. Two distinct frame arrays must collide across a full
        // 64-bit hash (less two bits consumed by `usable_key`). The alternative,
        // chaining, would put a second indirection on the hot path to insure
        // against an event that will not occur.
        if state.intern.insert(arena, key, index_u32) == Insert::Full {
            return Interned::Overflowed;
        }

        Interned::Created(id)
    }

    /// Applies `change` to the record for `id`, refreshing its at-peak snapshot
    /// first.
    ///
    /// `epoch` is the current global peak epoch. The caller must hold the peak
    /// gate, in either mode, for the whole of this call — that is what makes
    /// the epoch and the counters consistent.
    #[inline]
    pub fn update(&self, id: PpId, epoch: u64, change: impl FnOnce(&mut Counters)) {
        if id == PpId::OVERFLOW {
            let _order = super::order::enter(super::order::Level::ProgramPointShard);
            let _guard = self.overflow_lock.lock();
            // SAFETY: reached only while holding `overflow_lock`.
            let record = unsafe { &mut *self.overflow.get() };
            record.refresh(epoch);
            change(&mut record.counters);
            update_maxima(&mut record.counters);
            return;
        }

        let shard = &self.shards[id.shard()];
        let _order = super::order::enter(super::order::Level::ProgramPointShard);
        let _guard = shard.lock.lock();
        // SAFETY: reached only while holding `shard.lock`.
        let state = unsafe { &mut *shard.state.get() };
        let Some(record) = state.records.get_mut(id.index()) else {
            // Only reachable from a corrupted id, which would mean a bug
            // elsewhere. Report rather than panic: this runs inside the
            // allocator.
            super::diagnostic::poison("program-point id out of range");
            return;
        };
        record.refresh(epoch);
        change(&mut record.counters);
        update_maxima(&mut record.counters);
    }

    /// Reads the counters for `id`.
    pub fn counters(&self, id: PpId) -> Option<Counters> {
        let _order = super::order::enter(super::order::Level::ProgramPointShard);
        if id == PpId::OVERFLOW {
            let _guard = self.overflow_lock.lock();
            // SAFETY: reached only while holding `overflow_lock`.
            return Some(unsafe { (*self.overflow.get()).counters });
        }
        let shard = &self.shards[id.shard()];
        let _guard = shard.lock.lock();
        // SAFETY: reached only while holding `shard.lock`.
        let state = unsafe { &*shard.state.get() };
        state.records.get(id.index()).map(|record| record.counters)
    }

    /// Reads the creation sequence for `id`: where this program point stands in
    /// the order the program first reached its call sites.
    ///
    /// Sorting by this is what makes a profile reproducible. Every other
    /// identity a record has moves between runs — the shard comes from hashing
    /// return addresses, and address space layout randomization moves those
    /// addresses on every execution, so shard order is a reading of where the
    /// program happened to be mapped. `Snapshot::of` sorts by this instead, and
    /// the emitters inherit the order from the snapshot.
    ///
    /// The overflow point answers [`u32::MAX`], which sorts it last — read from
    /// its record rather than returned from here, so that where the overflow
    /// point belongs is stated once. An id with no record answers the same, and
    /// the flush cannot produce one.
    pub fn sequence(&self, id: PpId) -> u32 {
        let _order = super::order::enter(super::order::Level::ProgramPointShard);
        if id == PpId::OVERFLOW {
            let _guard = self.overflow_lock.lock();
            // SAFETY: reached only while holding `overflow_lock`.
            return unsafe { (*self.overflow.get()).sequence };
        }
        let shard = &self.shards[id.shard()];
        let _guard = shard.lock.lock();
        // SAFETY: reached only while holding `shard.lock`.
        let state = unsafe { &*shard.state.get() };
        state
            .records
            .get(id.index())
            .map_or(u32::MAX, |record| record.sequence)
    }

    /// Reads the frames for `id`.
    ///
    /// The copy into `out` happens **after** the shard lock is released, for the
    /// same reason [`PpTable::flush_and_visit`] copies records out before
    /// calling its visitor: growing `out` allocates, an allocation re-enters the
    /// shim, and the shim reaches this table. Reacquiring a shard lock this
    /// thread already holds is a hang on Linux and Windows and an immediate,
    /// message-free `SIGKILL` on Apple platforms.
    ///
    /// Holding the frames as a pointer across the release is sound because an
    /// interned record's frames are copied into the arena once and never moved,
    /// mutated, or freed while the table lives.
    pub fn frames(&self, id: PpId, out: &mut Vec<usize>) -> bool {
        let _order = super::order::enter(super::order::Level::ProgramPointShard);
        out.clear();
        if id == PpId::OVERFLOW {
            return true;
        }

        let (frames, len) = {
            let shard = &self.shards[id.shard()];
            let _guard = shard.lock.lock();
            // SAFETY: reached only while holding `shard.lock`.
            let state = unsafe { &*shard.state.get() };
            match state.records.get(id.index()) {
                Some(record) => (record.frames, record.frames_len as usize),
                None => return false,
            }
        };

        if !frames.is_null() && len != 0 {
            // SAFETY: `frames` and `frames_len` were set together from an arena
            // slice that outlives the table and is never mutated after
            // interning, so the slice stays valid after the lock is released.
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(frames, len) });
        }
        true
    }

    /// Brings every record's at-peak snapshot up to date and visits them.
    ///
    /// This is the end-of-run flush the lazy-epoch scheme requires: a record
    /// untouched since the last peak still holds a stale epoch, and its current
    /// values are its values at the peak.
    ///
    /// The caller must hold the peak gate exclusively, so that no update is in
    /// flight while the snapshot is taken.
    pub fn flush_and_visit(&self, epoch: u64, mut visit: impl FnMut(PpId, &[usize], &Counters)) {
        for (shard_index, shard) in self.shards.iter().enumerate() {
            let count = {
                let _order = super::order::enter(super::order::Level::ProgramPointShard);
                let _guard = shard.lock.lock();
                // SAFETY: reached only while holding `shard.lock`.
                unsafe { (*shard.state.get()).records.len() }
            };

            for index in 0..count {
                // The record is refreshed and copied out *under* the lock, and
                // the visitor runs after it is released. Calling a visitor while
                // holding a shard lock is a trap: the M2 emitter will allocate,
                // which re-enters the shim, and a visitor that called back into
                // this table would reacquire the same non-reentrant lock — which
                // on Apple is an immediate `SIGKILL` with no message.
                //
                // Copying is cheap: `Counters` is `Copy`, and the frame pointer
                // stays valid because the arena never frees.
                let snapshot = {
                    let _order = super::order::enter(super::order::Level::ProgramPointShard);
                    let _guard = shard.lock.lock();
                    // SAFETY: reached only while holding `shard.lock`.
                    let state = unsafe { &mut *shard.state.get() };
                    state.records.get_mut(index).map(|record| {
                        record.refresh(epoch);
                        (record.counters, record.frames, record.frames_len)
                    })
                };

                let Some((counters, frames, frames_len)) = snapshot else {
                    continue;
                };
                // A point interned for an allocation that was then dropped —
                // because the live-block table was full — has never recorded a
                // block. Emitting it would put a program point with every
                // counter zero in the profile, and `dh_view.js` divides by the
                // block count for its average columns.
                if counters.total_blocks == 0 {
                    continue;
                }
                let Some(id) = PpId::new(shard_index, index) else {
                    continue;
                };
                let frames = if frames.is_null() || frames_len == 0 {
                    &[][..]
                } else {
                    // SAFETY: `frames`/`frames_len` came from one arena slice
                    // that is never freed or mutated after interning, so the
                    // slice outlives this call even with the lock released.
                    unsafe { std::slice::from_raw_parts(frames, frames_len as usize) }
                };
                visit(id, frames, &counters);
            }
        }

        let overflow = {
            let _order = super::order::enter(super::order::Level::ProgramPointShard);
            let _guard = self.overflow_lock.lock();
            // SAFETY: reached only while holding `overflow_lock`.
            let record = unsafe { &mut *self.overflow.get() };
            record.refresh(epoch);
            record.counters
        };
        if overflow.total_blocks > 0 {
            visit(PpId::OVERFLOW, &[], &overflow);
        }
    }

    /// Number of interned program points, excluding the overflow point.
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| {
                let _order = super::order::enter(super::order::Level::ProgramPointShard);
                let _guard = shard.lock.lock();
                // SAFETY: reached only while holding `shard.lock`.
                unsafe { (*shard.state.get()).records.len() }
            })
            .sum()
    }

    /// Whether no program point has been interned.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Program points the table can hold before it starts overflowing.
    ///
    /// Summed from the shards rather than remembered from the request, and the
    /// two differ: each shard takes `max_points / SHARDS`, so a request of
    /// 1,000 across 64 shards is a ceiling of 960. Reporting the request would
    /// let a profile's `droppedPoints` count make sense against a number the
    /// table never enforced — the same trap
    /// [`effective_ceiling`](super::live) closes for live blocks.
    pub fn capacity(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| {
                let _order = super::order::enter(super::order::Level::ProgramPointShard);
                let _guard = shard.lock.lock();
                // SAFETY: reached only while holding `shard.lock`.
                let state = unsafe { &*shard.state.get() };
                state.records.max_capacity()
            })
            .sum()
    }

    /// Arena bytes held by the table, for self-metrics.
    ///
    /// Three things, not two. The intern map and the record vector are the
    /// obvious ones; the **frame lists** are the third, one arena slice per
    /// program point, and leaving them out is not a rounding error. A profile
    /// reports the arena's used bytes beside this, and the difference is read as
    /// what growth abandoned — so every byte of live frame storage the table
    /// failed to claim was being reported as waste. Measured on
    /// `examples/profile_a_program`: the entire 592-byte "waste" of a
    /// nine-point run was its 74 frame slots, and nothing had been abandoned at
    /// all.
    pub fn bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| {
                let _order = super::order::enter(super::order::Level::ProgramPointShard);
                let _guard = shard.lock.lock();
                // SAFETY: reached only while holding `shard.lock`.
                let state = unsafe { &*shard.state.get() };
                let frames: usize = state
                    .records
                    .iter()
                    .map(|record| record.frames_len as usize * std::mem::size_of::<usize>())
                    .sum();
                state.intern.bytes() + state.records.bytes() + frames
            })
            .sum()
    }

    /// Acquires every lock in the table, for a `fork` prepare handler.
    ///
    /// # Safety
    ///
    /// A matching [`PpTable::unlock_all_for_fork`] must run on the same thread,
    /// or the child must reset the locks with [`PpTable::reinit_after_fork`].
    pub unsafe fn lock_all_for_fork(&self) {
        for shard in &self.shards {
            // SAFETY: delegated to the caller's pairing obligation.
            unsafe { shard.lock.raw_lock() };
        }
        // SAFETY: as above.
        unsafe { self.overflow_lock.raw_lock() };
    }

    /// Releases what [`PpTable::lock_all_for_fork`] acquired.
    ///
    /// # Safety
    ///
    /// The calling thread must hold every lock through
    /// [`PpTable::lock_all_for_fork`].
    pub unsafe fn unlock_all_for_fork(&self) {
        // SAFETY: delegated to the caller's obligation.
        unsafe { self.overflow_lock.raw_unlock() };
        for shard in self.shards.iter().rev() {
            // SAFETY: as above.
            unsafe { shard.lock.raw_unlock() };
        }
    }

    /// Re-initializes every lock after a `fork`.
    ///
    /// # Safety
    ///
    /// Call only from a `pthread_atfork` child handler, where the process is
    /// single-threaded.
    pub unsafe fn reinit_after_fork(&self) {
        for shard in &self.shards {
            // SAFETY: delegated to the caller's single-threadedness obligation.
            unsafe { shard.lock.force_reinit() };
        }
        // SAFETY: as above.
        unsafe { self.overflow_lock.force_reinit() };
    }
}

impl Default for PpTable {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for PpTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PpTable")
            .field("program_points", &self.len())
            .finish_non_exhaustive()
    }
}

/// Keeps `max_*` at the running maximum of `curr_*`.
///
/// PLAN.md decision 10.3: a *true* per-point maximum, unlike Valgrind's, which
/// only samples when the whole heap is at its peak.
#[inline(always)]
fn update_maxima(counters: &mut Counters) {
    if counters.curr_bytes > counters.max_bytes {
        counters.max_bytes = counters.curr_bytes;
    }
    if counters.curr_blocks > counters.max_blocks {
        counters.max_blocks = counters.curr_blocks;
    }
}

/// Hashes a frame array.
///
/// FxHash-style: one multiply-and-rotate per frame. Return addresses within one
/// binary share their high bits, so a hash that only mixes forwards would put
/// every stack from the same module into a handful of buckets. The final
/// avalanche is what spreads them.
#[inline]
pub fn hash_frames(frames: &[usize]) -> u64 {
    const SEED: u64 = 0x51_7C_C1_B7_27_22_0A_95;
    let mut hash = frames.len() as u64;
    for &frame in frames {
        hash = (hash.rotate_left(5) ^ frame as u64).wrapping_mul(SEED);
    }
    super::table::mix(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internals::miri_scale;

    fn table() -> (Arena, PpTable) {
        (Arena::new(), PpTable::with_capacity(4096))
    }

    /// The ceiling a profile reports has to be the one the table enforces.
    ///
    /// The same trap `effective_ceiling` closes for live blocks, met again here:
    /// a `droppedPoints` count is only readable against the ceiling that
    /// produced it, so a capacity that is really the occupancy — or really the
    /// request — turns that count into a contradiction. Replacing this with
    /// `len()` left the whole suite green, because every other test that reads
    /// a capacity only checks that the occupancy does not exceed it, which is
    /// trivially true when they are the same number.
    #[test]
    fn the_program_point_ceiling_reported_is_the_one_in_force() {
        // Each of 64 shards takes `1_000 / SHARDS` = 15, so the ceiling is 960
        // rather than the 1,000 asked for.
        let table = PpTable::with_capacity(1_000);
        assert_eq!(table.capacity(), 15 * SHARDS);
        assert_eq!(
            table.len(),
            0,
            "an empty table still has its ceiling, so the two are not the same \
             number"
        );

        let arena = Arena::new();
        table.intern(&arena, &[0xAA]);
        table.intern(&arena, &[0xBB]);
        assert_eq!(table.len(), 2);
        assert_eq!(
            table.capacity(),
            15 * SHARDS,
            "interning moved the ceiling, so it is reporting occupancy"
        );
    }

    #[test]
    fn identical_stacks_intern_to_one_point() {
        let (arena, table) = table();
        let stack = [0x1000usize, 0x2000, 0x3000];

        let first = table.intern(&arena, &stack);
        let second = table.intern(&arena, &stack);

        assert!(matches!(first, Interned::Created(_)));
        assert_eq!(second, Interned::Existing(first.id()));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn distinct_stacks_intern_separately() {
        let (arena, table) = table();
        let count = miri_scale(500);
        let mut ids = Vec::new();
        for i in 0..count {
            let stack = [0x1000 + i * 8, 0x2000, 0x3000];
            ids.push(table.intern(&arena, &stack).id());
        }
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count, "distinct stacks were merged");
        assert_eq!(table.len(), count);
    }

    /// The sequence is the table's only slide-independent ordering, so what it
    /// counts has to be creation and nothing else.
    ///
    /// A sequence bumped on every `intern` rather than on every *new* record
    /// would still be increasing, still be unique per record, and still sort
    /// deterministically — it would simply renumber points by which one
    /// allocated most recently, which is a reading of the workload's timing and
    /// not of its structure.
    #[test]
    fn a_sequence_is_claimed_once_per_point_and_never_by_a_repeat() {
        let (arena, table) = table();
        let first = table.intern(&arena, &[0x10usize]).id();
        let second = table.intern(&arena, &[0x20usize]).id();
        // Between the two below: a repeat of the older point, which must not
        // move it and must not consume a number.
        assert_eq!(
            table.intern(&arena, &[0x10usize]),
            Interned::Existing(first)
        );
        let third = table.intern(&arena, &[0x30usize]).id();

        assert_eq!(table.sequence(first), 0);
        assert_eq!(table.sequence(second), 1);
        assert_eq!(
            table.sequence(third),
            2,
            "re-interning an existing point consumed a sequence number"
        );
        assert_eq!(
            table.sequence(PpId::OVERFLOW),
            u32::MAX,
            "the overflow point has to sort last, after every real point"
        );
    }

    /// Points land in shards by a hash of their addresses, so consecutive
    /// sequence numbers are spread across the table rather than filling it in
    /// order. Sorting by sequence has to recover the creation order anyway.
    #[test]
    fn sequences_order_points_across_shards() {
        let (arena, table) = table();
        let count = miri_scale(200);
        let created: Vec<PpId> = (0..count)
            .map(|i| table.intern(&arena, &[0x4000 + i * 0x40, 0x9000]).id())
            .collect();

        let shards: Vec<usize> = created.iter().map(|id| id.shard()).collect();
        let mut distinct = shards.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert!(
            distinct.len() > 1,
            "every point landed in one shard, so this proves nothing about order \
             across them"
        );
        assert!(
            shards.windows(2).any(|pair| pair[0] > pair[1]),
            "shard order happened to match creation order, so this proves nothing"
        );

        let mut by_sequence = created.clone();
        by_sequence.sort_unstable_by_key(|&id| table.sequence(id));
        assert_eq!(
            by_sequence, created,
            "sorting by sequence lost creation order"
        );
    }

    /// Stacks that differ only in order, or only in length, must not collide.
    #[test]
    fn stack_order_and_length_are_part_of_identity() {
        let (arena, table) = table();
        let a = table.intern(&arena, &[1usize, 2, 3]).id();
        let b = table.intern(&arena, &[3usize, 2, 1]).id();
        let c = table.intern(&arena, &[1usize, 2]).id();
        let d = table.intern(&arena, &[1usize, 2, 3, 0]).id();

        let mut ids = vec![a, b, c, d];
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 4, "stacks differing in order or length collided");
    }

    #[test]
    fn frames_round_trip() {
        let (arena, table) = table();
        let stack = [0xDEAD_BEEFusize, 0xCAFE, 0x1234_5678];
        let id = table.intern(&arena, &stack).id();

        let mut out = Vec::new();
        assert!(table.frames(id, &mut out));
        assert_eq!(out, stack);
    }

    #[test]
    fn an_empty_stack_is_a_valid_program_point() {
        let (arena, table) = table();
        let id = table.intern(&arena, &[]).id();
        let mut out = vec![1, 2, 3];
        assert!(table.frames(id, &mut out));
        assert!(out.is_empty());
    }

    #[test]
    fn counters_accumulate() {
        let (arena, table) = table();
        let id = table.intern(&arena, &[1usize, 2]).id();

        for _ in 0..10 {
            table.update(id, 0, |c| {
                c.total_bytes += 100;
                c.total_blocks += 1;
                c.curr_bytes += 100;
                c.curr_blocks += 1;
            });
        }

        let counters = table.counters(id).unwrap();
        assert_eq!(counters.total_bytes, 1000);
        assert_eq!(counters.total_blocks, 10);
        assert_eq!(counters.curr_bytes, 1000);
        assert_eq!(counters.max_bytes, 1000);
    }

    /// The divergence from Valgrind that PLAN.md decision 10.3 chose: the
    /// maximum is a true per-point running maximum, not a value sampled only
    /// when the whole heap is at its peak.
    #[test]
    fn max_is_a_true_running_maximum() {
        let (arena, table) = table();
        let id = table.intern(&arena, &[1usize]).id();

        table.update(id, 0, |c| {
            c.curr_bytes = 4_000_000;
            c.curr_blocks = 10;
        });
        table.update(id, 0, |c| {
            c.curr_bytes = 100;
            c.curr_blocks = 1;
        });

        let counters = table.counters(id).unwrap();
        assert_eq!(counters.curr_bytes, 100);
        assert_eq!(
            counters.max_bytes, 4_000_000,
            "the peak was forgotten once the point shrank"
        );
        assert_eq!(counters.max_blocks, 10);
    }

    /// The core of the lazy-epoch scheme: a record untouched across a peak must
    /// still report the values it held at that peak.
    #[test]
    fn a_stale_epoch_snapshots_on_the_next_touch() {
        let (arena, table) = table();
        let id = table.intern(&arena, &[1usize]).id();

        // Grow to 500 bytes at epoch 0. `total_blocks` is set too, because a
        // point that has never recorded a block is not emitted at all -- see
        // `flush_and_visit`.
        table.update(id, 0, |c| {
            c.curr_bytes = 500;
            c.curr_blocks = 5;
            c.total_blocks = 5;
        });

        // The heap peaks here; the engine bumps the epoch to 1. This record is
        // not touched at the moment of the peak.

        // Later, the record shrinks. The refresh must capture 500/5 first.
        table.update(id, 1, |c| {
            c.curr_bytes = 10;
            c.curr_blocks = 1;
        });

        let counters = table.counters(id).unwrap();
        assert_eq!(
            counters.at_gmax_bytes, 500,
            "the at-peak snapshot was missed"
        );
        assert_eq!(counters.at_gmax_blocks, 5);
        assert_eq!(counters.curr_bytes, 10);
    }

    /// A record whose only event after the peak is a free is the case most
    /// likely to be missed, because nothing "adds" to trigger a refresh.
    #[test]
    fn a_free_after_the_peak_still_triggers_the_snapshot() {
        let (arena, table) = table();
        let id = table.intern(&arena, &[7usize]).id();

        table.update(id, 0, |c| {
            c.curr_bytes = 900;
            c.curr_blocks = 3;
            c.total_blocks = 3;
        });
        table.update(id, 4, |c| {
            c.curr_bytes -= 900;
            c.curr_blocks -= 3;
        });

        let counters = table.counters(id).unwrap();
        assert_eq!(counters.at_gmax_bytes, 900);
        assert_eq!(counters.curr_bytes, 0);
    }

    /// Several peaks between two touches of one record must collapse to the
    /// most recent, not the first.
    #[test]
    fn multiple_peaks_between_touches_use_the_latest() {
        let (arena, table) = table();
        let id = table.intern(&arena, &[9usize]).id();

        table.update(id, 0, |c| {
            c.curr_bytes = 100;
            c.total_blocks = 1;
        });
        // Epochs 1, 2, and 3 pass without this record being touched.
        table.update(id, 3, |c| c.curr_bytes = 50);

        let counters = table.counters(id).unwrap();
        assert_eq!(
            counters.at_gmax_bytes, 100,
            "the snapshot should hold the value carried through every skipped epoch"
        );
    }

    #[test]
    fn the_end_of_run_flush_updates_untouched_records() {
        let (arena, table) = table();
        let touched = table.intern(&arena, &[1usize]).id();
        let untouched = table.intern(&arena, &[2usize]).id();

        table.update(touched, 0, |c| {
            c.curr_bytes = 100;
            c.total_blocks = 1;
        });
        table.update(untouched, 0, |c| {
            c.curr_bytes = 700;
            c.total_blocks = 1;
        });

        // The heap peaks; epoch becomes 1. Only `touched` is touched again.
        table.update(touched, 1, |c| c.curr_bytes = 1);

        let mut seen = Vec::new();
        table.flush_and_visit(1, |id, _frames, counters| {
            seen.push((id, counters.at_gmax_bytes));
        });

        let at_gmax = |wanted: PpId| seen.iter().find(|(id, _)| *id == wanted).unwrap().1;
        assert_eq!(at_gmax(touched), 100);
        assert_eq!(
            at_gmax(untouched),
            700,
            "a record never touched after the peak was not flushed"
        );
    }

    #[test]
    fn overflow_absorbs_allocations_once_the_table_is_full() {
        let arena = Arena::new();
        // Small enough that the per-shard ceiling is reached quickly.
        let table = PpTable::with_capacity(SHARDS * 2);

        let mut overflowed = 0;
        for i in 0..miri_scale(10_000) {
            if table.intern(&arena, &[i, i * 3, i * 7]) == Interned::Overflowed {
                overflowed += 1;
            }
        }
        assert!(overflowed > 0, "the ceiling was never reached");

        // The overflow point must still account for what it absorbed.
        table.update(PpId::OVERFLOW, 0, |c| {
            c.total_bytes += 64;
            c.total_blocks += 1;
            c.curr_bytes += 64;
            c.curr_blocks += 1;
        });
        let counters = table.counters(PpId::OVERFLOW).unwrap();
        assert_eq!(counters.total_bytes, 64);
        assert_eq!(counters.max_bytes, 64);
    }

    #[test]
    fn overflow_is_visited_by_the_flush_when_it_was_used() {
        let (arena, table) = table();
        table.intern(&arena, &[1usize]);
        table.update(PpId::OVERFLOW, 0, |c| {
            c.total_blocks += 1;
            c.curr_bytes += 32;
        });

        let mut ids = Vec::new();
        table.flush_and_visit(0, |id, _, _| ids.push(id));
        assert!(
            ids.contains(&PpId::OVERFLOW),
            "the overflow point was omitted from the output"
        );
    }

    #[test]
    fn overflow_is_omitted_when_it_was_never_used() {
        let (arena, table) = table();
        table.intern(&arena, &[1usize]);

        let mut ids = Vec::new();
        table.flush_and_visit(0, |id, _, _| ids.push(id));
        assert!(
            !ids.contains(&PpId::OVERFLOW),
            "an unused overflow point should not appear in the profile"
        );
    }

    #[test]
    fn ids_round_trip_through_their_encoding() {
        for shard in [0usize, 1, 31, SHARDS - 1] {
            for index in [0usize, 1, 1000, 65_535] {
                let id = PpId::new(shard, index).unwrap();
                assert_eq!(id.shard(), shard, "shard lost for ({shard}, {index})");
                assert_eq!(id.index(), index, "index lost for ({shard}, {index})");
                assert_ne!(id, PpId::OVERFLOW);
            }
        }
    }

    #[test]
    fn concurrent_interning_agrees_on_identity() {
        #[cfg(miri)]
        const STACKS: usize = 8;
        #[cfg(not(miri))]
        const STACKS: usize = 500;
        const THREADS: usize = 8;

        let arena = Arena::new();
        let table = PpTable::with_capacity(1 << 16);

        let results: Vec<Vec<PpId>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let (arena, table) = (&arena, &table);
                    s.spawn(move || {
                        (0..STACKS)
                            .map(|i| table.intern(arena, &[i, i * 2, i * 3]).id())
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Every thread interned the same stacks, so every thread must have got
        // the same ids. Anything else means two threads created duplicate
        // program points for one call stack, splitting its counters.
        for thread_results in &results[1..] {
            assert_eq!(
                *thread_results, results[0],
                "threads disagreed about program-point identity"
            );
        }
        assert_eq!(table.len(), STACKS);
    }

    /// A point interned for an allocation that was then dropped -- because the
    /// live-block table was full -- has recorded no block, and emitting it would
    /// put a program point with every counter zero into the profile. The DHAT
    /// viewer divides by the block count for its average columns.
    #[test]
    fn points_that_never_recorded_a_block_are_not_emitted() {
        let (arena, table) = table();
        let real = table.intern(&arena, &[1usize]).id();
        let phantom = table.intern(&arena, &[2usize]).id();

        table.update(real, 0, |c| {
            c.total_bytes = 64;
            c.total_blocks = 1;
            c.curr_bytes = 64;
            c.curr_blocks = 1;
        });

        let mut emitted = Vec::new();
        table.flush_and_visit(0, |id, _frames, _counters| emitted.push(id));

        assert!(emitted.contains(&real));
        assert!(
            !emitted.contains(&phantom),
            "a program point with no recorded blocks was emitted"
        );
    }

    /// The visitor must not run while a shard lock is held: the output emitter
    /// allocates, and a visitor that called back into this table would
    /// reacquire a non-reentrant lock -- an immediate SIGKILL on Apple.
    #[test]
    fn the_visitor_may_call_back_into_the_table() {
        let (arena, table) = table();
        for i in 0..50usize {
            let id = table.intern(&arena, &[i]).id();
            table.update(id, 0, |c| {
                c.total_bytes = 8;
                c.total_blocks = 1;
                c.curr_bytes = 8;
                c.curr_blocks = 1;
            });
        }

        let mut checked = 0usize;
        table.flush_and_visit(0, |id, _frames, counters| {
            // Reaching back into the table from inside the visitor deadlocks or
            // dies if the lock is still held.
            let again = table.counters(id).expect("the point should still exist");
            assert_eq!(again.total_bytes, counters.total_bytes);
            checked += 1;
        });
        assert_eq!(checked, 50);
    }
}
