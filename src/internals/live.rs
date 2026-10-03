//! The live-block table: which program point owns each live allocation.
//!
//! # Why it exists at all
//!
//! `GlobalAlloc::dealloc` is handed the `Layout`, so the *size* of a freed block
//! never has to be stored — and using the layout is strictly more robust than a
//! stored copy, which can desync. Three things still require a table:
//!
//! 1. **Attribution of the free.** The layout says how big, not *who allocated*.
//!    Without `pointer -> program point`, a free would decrement whichever site
//!    happened to call `free`, driving per-point live bytes negative and
//!    destroying the at-peak and at-end columns entirely.
//! 2. **Block lifetime.** DHAT's `tl` and its short-lived-block counts need the
//!    instant each block was born.
//! 3. **Membership.** Telling "we never recorded this block" (it predates the
//!    profiler, or sampling skipped it) apart from "we did". A free of an
//!    unknown pointer is simply ignored, which is both the desired behaviour and
//!    the reason no separate pre-start set is needed.
//!
//! # Sharding by pointer, not by thread
//!
//! Deliberate: it makes a cross-thread free — allocate on one thread, free on
//! another, which is overwhelmingly common in Rust — contend no worse than a
//! same-thread one. Sharding by allocating thread would put every free of a
//! producer thread's blocks onto that thread's shard.
//!
//! # Entry size, and a deliberate divergence from PLAN.md
//!
//! PLAN.md section 4.5 sizes the value at `{pp_id: u32, event: u32}` — 8 bytes,
//! for a 16-byte entry. This uses a **64-bit** birth timestamp instead, making
//! the entry 24 bytes.
//!
//! The reason is that a 32-bit event counter wraps after 4.29 billion
//! allocation events, which a busy process reaches in under a minute. Every
//! block that outlives one wrap — typically the long-lived ones allocated at
//! startup, which are exactly the blocks a heap profile is most often read to
//! understand — would report a wrong, smaller lifetime, with nothing to
//! indicate it. That is a silently incorrect number in the output, and PLAN.md
//! section 1 puts correctness first: "never report a number we cannot
//! substantiate".
//!
//! The cost is 50% more memory per live block, which is bounded by
//! [`DEFAULT_MAX_LIVE_BLOCKS`] and visible in the profile's self-metrics.
//!
//! # One address cannot be tracked
//!
//! `RawMap` reserves `0` to mark an empty slot, so a block at null is not
//! tracked. It is not a valid heap block address on any supported platform: an
//! allocator returns null only to signal failure. The table refuses it rather
//! than mapping it onto some other key, because folding would make two
//! distinct addresses share one entry.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::arena::Arena;
use super::lock::RawLock;
use super::pp::PpId;
use super::site::Site;
use super::table::{mix, Insert, RawMap};
use super::CachePadded;

/// Number of shards. A compile-time constant, so the shard array is a plain
/// `static` with no lazy initialization on the allocator path.
pub const SHARDS: usize = 64;

const _: () = assert!(SHARDS.is_power_of_two());

/// Default ceiling on simultaneously live tracked blocks.
///
/// A memory-analysis tool with unbounded memory growth is a contradiction
/// (PLAN.md section 4.5). At 24 bytes per entry and a 50% load factor this is
/// roughly 200 MB of profiler state at the limit; blocks beyond it are counted
/// as dropped and reported rather than silently mis-attributed.
pub const DEFAULT_MAX_LIVE_BLOCKS: usize = 4 * 1024 * 1024;

/// What is remembered about one live allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct LiveBlock {
    /// The clock reading when this block was allocated.
    pub birth: u64,
    /// The program point that allocated it.
    pub pp: PpId,
    /// The thread that allocated it, and the region that was open.
    ///
    /// Held per block rather than only summed per row because a free has to
    /// bring the *allocating* thread's live bytes down, and the freeing thread
    /// is frequently a different one. Without this, "which thread is holding
    /// the memory" would be answerable only for programs that free on the
    /// thread that allocated.
    pub site: Site,
}

/// Attribution is meant to cost nothing in the table that dominates this
/// profiler's memory, and "nothing" is a size. A `u64`, a `u32` and four bytes
/// of [`Site`] is 16 bytes — exactly what the two fields before it occupied,
/// because the `u32` was followed by four bytes of padding either way.
const _: () = assert!(std::mem::size_of::<LiveBlock>() == 16);

impl LiveBlock {
    /// The birth given to every block still live when a run's counts restart.
    ///
    /// [`Engine::reset`](super::engine::Engine::reset) excludes everything
    /// allocated before it from the totals, so a block it finds live is in none
    /// of the allocation counts (`totalBlocks`, DHAT's `tbk`) the profile will
    /// report, though it is in the live and at-peak ones. Its lifetime must therefore count
    /// toward no lifetime total either: DHAT's average lifetime is `tl / tbk`,
    /// and a block in the numerator but not the denominator inflates the one
    /// column a reader uses to find short-lived churn. A warm-up's caches are
    /// exactly the long-lived blocks that would do it.
    ///
    /// The largest value there is, so that the saturating difference
    /// [`LiveBlock::lifetime_at`] takes is zero for it without a branch. The
    /// free path pays nothing for a run that never resets.
    pub const CARRIED: u64 = u64::MAX;

    /// How long this block has lived as of `now`, as far as any lifetime total
    /// counts it.
    ///
    /// Zero for a [carried](LiveBlock::CARRIED) block, and zero rather than a
    /// wrapped value if `now` somehow precedes the birth: a lifetime summed into
    /// `tl` is better short by one block than larger than the run.
    #[inline(always)]
    pub fn lifetime_at(&self, now: u64) -> u64 {
        now.saturating_sub(self.birth)
    }

    /// Whether this block was live when the counts last restarted.
    pub fn is_carried(&self) -> bool {
        self.birth == Self::CARRIED
    }

    /// A block whose thread and region are not known.
    ///
    /// For paths that record without a guard in hand, and for tests about
    /// something other than attribution.
    pub const fn unattributed(birth: u64, pp: PpId) -> Self {
        Self {
            birth,
            pp,
            site: Site::UNATTRIBUTED,
        }
    }
}

struct Shard {
    lock: RawLock,
    blocks: std::cell::UnsafeCell<RawMap<LiveBlock>>,
}

// SAFETY: `blocks` is reached only while holding `lock`.
unsafe impl Sync for Shard {}

impl Shard {
    const fn new(max_blocks: usize) -> Self {
        Self {
            lock: RawLock::new(),
            blocks: std::cell::UnsafeCell::new(RawMap::new(slots_per_shard(max_blocks))),
        }
    }
}

/// Slots to give one shard's map so that `max_blocks` fit across all of them.
///
/// Twice the per-shard block ceiling, because the map grows at a 50% load
/// factor and must be able to hold that many live entries. Saturating rather
/// than wrapping: a ceiling large enough to overflow this is a ceiling nothing
/// can reach anyway, and a silently tiny table would be the worst answer.
const fn slots_per_shard(max_blocks: usize) -> usize {
    let per_shard = max_blocks / SHARDS;
    if per_shard > usize::MAX / 4 {
        usize::MAX / 2
    } else {
        per_shard.next_power_of_two() * 2
    }
}

/// Live blocks a table built for `max_blocks` can actually hold.
///
/// Not `max_blocks`. Each shard rounds its share up to a power of two, so the
/// ceiling in force is at or above the one asked for — 5,000 becomes 8,192,
/// 4,194,304 stays exactly that. Reporting the request instead lets a profile
/// contradict itself: a run asked for 5,000 held 8,192 live blocks while its own
/// `settings` block said 5,000, and the `droppedBlocks` count then made sense
/// against neither number **\[measured\]**.
const fn effective_ceiling(max_blocks: usize) -> usize {
    slots_per_shard(max_blocks) / 2 * SHARDS
}

/// A sharded map from pointer to owning program point.
pub struct LiveBlocks {
    shards: [CachePadded<Shard>; SHARDS],
    /// The ceiling actually in force, which is what a profile reports.
    max_blocks: AtomicUsize,
}

// SAFETY: every shard is reached only under its own lock.
unsafe impl Sync for LiveBlocks {}

impl LiveBlocks {
    /// Creates an empty table with the default ceiling.
    pub const fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_LIVE_BLOCKS)
    }

    /// Creates an empty table holding at most `max_blocks` live blocks.
    pub const fn with_capacity(max_blocks: usize) -> Self {
        const fn shards(max_blocks: usize) -> [CachePadded<Shard>; SHARDS] {
            // SAFETY: an array of `MaybeUninit` needs no initialization to be
            // valid; each element is written by the loop below.
            let mut array: [std::mem::MaybeUninit<CachePadded<Shard>>; SHARDS] =
                unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            let mut index = 0;
            while index < SHARDS {
                array[index] = std::mem::MaybeUninit::new(CachePadded::new(Shard::new(max_blocks)));
                index += 1;
            }
            // SAFETY: every element was initialized above, and
            // `[MaybeUninit<T>; N]` shares its layout with `[T; N]`.
            unsafe { std::mem::transmute::<_, [CachePadded<Shard>; SHARDS]>(array) }
        }

        Self {
            shards: shards(max_blocks),
            max_blocks: AtomicUsize::new(effective_ceiling(max_blocks)),
        }
    }

    /// The ceiling on simultaneously live tracked blocks.
    pub fn max_blocks(&self) -> usize {
        self.max_blocks.load(Ordering::Relaxed)
    }

    /// Changes that ceiling.
    ///
    /// The engine applies this while it is `Starting`, with the shim refusing
    /// every event, which is also why taking each shard's lock here is cheap.
    /// Raising would take effect at any time; *lowering* is the one-way case,
    /// because growth cannot be undone — a ceiling below what a map has already
    /// allocated stops further growth rather than giving memory back.
    pub(crate) fn set_max_blocks(&self, max_blocks: usize) {
        // What is stored is the ceiling this table will actually enforce, not
        // the request. The two differ in both directions: each of the 64 shards
        // rounds its share up to a power of two, and two blocks per shard is the
        // smallest table that exists. A profile reporting the request would be
        // reporting a limit nothing applies.
        self.max_blocks
            .store(effective_ceiling(max_blocks), Ordering::Relaxed);
        let slots = slots_per_shard(max_blocks);
        for shard in &self.shards {
            let _order = super::order::enter(super::order::Level::LiveBlockShard);
            let _guard = shard.lock.lock();
            // SAFETY: reached only while holding `shard.lock`.
            unsafe { &mut *shard.blocks.get() }.set_max_capacity(slots);
        }
    }

    /// Which shard owns `address`.
    ///
    /// The pointer is mixed first, because heap pointers are aligned and
    /// consecutive allocations differ only in a narrow middle range — masking
    /// the raw value would pile whole allocation bursts onto one shard.
    ///
    /// The shard is taken from the **high** bits of the mix, and that detail is
    /// worth more than it looks. `RawMap` indexes with `mix(key) & mask` on the
    /// same key, so taking the shard from the low bits made the two indices the
    /// same function: every key routed to shard *S* also had a home slot
    /// congruent to *S* mod 64, so only 1/64th of the map's slots were ever a
    /// home position and linear probing had to walk the gaps. Measured on
    /// realistic 16-byte-aligned consecutive pointers — 30,000 entries in a
    /// 65,536-slot shard — that was **15 probes per insert on average, 62 at
    /// worst**, against 1 and 18 once the shard came from the high bits.
    ///
    /// Neither existing test caught it: one exercised `RawMap` unsharded, the
    /// other the shard function alone. It only appears when the two compose.
    #[inline(always)]
    fn shard_of(address: usize) -> usize {
        const SHARD_BITS: u32 = SHARDS.trailing_zeros();
        ((mix(address as u64) >> (64 - SHARD_BITS)) as usize) & (SHARDS - 1)
    }

    /// Records a newly allocated block.
    ///
    /// Returns `false` if the table is full, in which case the block is not
    /// tracked and its eventual free will be ignored. Callers count this.
    pub fn insert(&self, arena: &Arena, address: usize, block: LiveBlock) -> bool {
        let shard = &self.shards[Self::shard_of(address)];
        let _order = super::order::enter(super::order::Level::LiveBlockShard);
        let _guard = shard.lock.lock();
        // SAFETY: reached only while holding `shard.lock`.
        let blocks = unsafe { &mut *shard.blocks.get() };
        blocks.insert(arena, address as u64, block) != Insert::Full
    }

    /// Removes and returns the record for `address`.
    ///
    /// `None` means the block was never tracked, which is normal: it may have
    /// been allocated before profiling started, or skipped by sampling.
    #[inline]
    pub fn remove(&self, address: usize) -> Option<LiveBlock> {
        let shard = &self.shards[Self::shard_of(address)];
        let _order = super::order::enter(super::order::Level::LiveBlockShard);
        let _guard = shard.lock.lock();
        // SAFETY: reached only while holding `shard.lock`.
        let blocks = unsafe { &mut *shard.blocks.get() };
        blocks.remove(address as u64)
    }

    /// Reads the record for `address` without removing it.
    pub fn get(&self, address: usize) -> Option<LiveBlock> {
        let shard = &self.shards[Self::shard_of(address)];
        let _order = super::order::enter(super::order::Level::LiveBlockShard);
        let _guard = shard.lock.lock();
        // SAFETY: reached only while holding `shard.lock`.
        let blocks = unsafe { &*shard.blocks.get() };
        blocks.get(address as u64)
    }

    /// Number of tracked live blocks.
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| {
                let _order = super::order::enter(super::order::Level::LiveBlockShard);
                let _guard = shard.lock.lock();
                // SAFETY: reached only while holding `shard.lock`.
                unsafe { (*shard.blocks.get()).len() }
            })
            .sum()
    }

    /// Whether no block is tracked.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Arena bytes held, for self-metrics.
    pub fn bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| {
                let _order = super::order::enter(super::order::Level::LiveBlockShard);
                let _guard = shard.lock.lock();
                // SAFETY: reached only while holding `shard.lock`.
                unsafe { (*shard.blocks.get()).bytes() }
            })
            .sum()
    }

    /// Visits every tracked block.
    ///
    /// Used at output time to attribute blocks still alive at the end of the
    /// run. Order is unspecified.
    pub fn for_each(&self, mut visit: impl FnMut(usize, LiveBlock)) {
        for shard in &self.shards {
            let _order = super::order::enter(super::order::Level::LiveBlockShard);
            let _guard = shard.lock.lock();
            // SAFETY: reached only while holding `shard.lock`.
            let blocks = unsafe { &*shard.blocks.get() };
            blocks.for_each(|address, block| visit(address as usize, block));
        }
    }

    /// Takes every shard's lock, for a caller that needs the whole table in one
    /// state, giving up at `deadline`.
    ///
    /// Acquired in index order and held until the returned value drops, so no
    /// block can be inserted or removed anywhere in the table meanwhile. That is
    /// what [`Engine::reset`](super::engine::Engine::reset) needs: it marks
    /// every live block as carried across a restart, and a block inserted into
    /// a shard it had already walked would be counted in a window it was not
    /// marked out of.
    ///
    /// `None` when a shard could not be had by `deadline`, with every shard
    /// taken so far released. `deadline` of `None` means no deadline, for the
    /// caller whose deadline was not representable.
    ///
    /// Enters [`super::order`] once, for the whole family, rather than once per
    /// shard: sixty-four locks of one level is a same-level reacquisition, which
    /// the checker reports on the recording paths and which is the whole point
    /// here. The level is still entered, so holding these and then taking
    /// anything shallower is caught exactly as it would be for one shard.
    ///
    /// # Deadlock, and why there is a deadline anyway
    ///
    /// None against the recording paths. A path holding a live-block shard waits
    /// for nothing but the arena, which is deeper in [`super::order`] and whose
    /// holders wait on nothing at all, so every holder this waits on is about
    /// to release. The deadline is for a holder that is not running at all: a
    /// thread stopped under a debugger, or descheduled for good. Waited on
    /// without one, it would hold the caller forever, and every thread whose
    /// blocks hash to a shard the caller already holds with it.
    pub fn lock_all_until(&self, deadline: Option<std::time::Instant>) -> Option<AllShards<'_>> {
        let order = super::order::enter(super::order::Level::LiveBlockShard);
        let mut taken: [Option<super::lock::RawGuard<'_>>; SHARDS] = [const { None }; SHARDS];
        for (slot, shard) in taken.iter_mut().zip(&self.shards) {
            let guard = match deadline {
                Some(deadline) => {
                    let left = deadline.saturating_duration_since(std::time::Instant::now());
                    // Dropping `taken` on the way out releases what was held.
                    shard.lock.try_lock_for(left)?
                }
                None => shard.lock.lock(),
            };
            *slot = Some(guard);
        }
        Some(AllShards {
            guards: taken.map(|guard| guard.expect("every shard was taken above")),
            table: self,
            _order: order,
        })
    }

    /// Forgets every tracked block, keeping the allocations.
    pub fn clear(&self) {
        for shard in &self.shards {
            let _order = super::order::enter(super::order::Level::LiveBlockShard);
            let _guard = shard.lock.lock();
            // SAFETY: reached only while holding `shard.lock`.
            let blocks = unsafe { &mut *shard.blocks.get() };
            blocks.clear();
        }
    }

    /// Acquires every shard lock, for a `fork` prepare handler.
    ///
    /// Deliberately does not enter [`super::order`]: taking sixty-four locks of
    /// one family is a same-level reacquisition, which the checker exists to
    /// report on the *recording* paths and which is exactly what a prepare
    /// handler must do.
    ///
    /// # Safety
    ///
    /// A matching [`LiveBlocks::unlock_all_for_fork`] must run on the same
    /// thread, or the child must reset the locks with
    /// [`LiveBlocks::reinit_after_fork`].
    pub unsafe fn lock_all_for_fork(&self) {
        for shard in &self.shards {
            // SAFETY: delegated to the caller's pairing obligation.
            unsafe { shard.lock.raw_lock() };
        }
    }

    /// Releases what [`LiveBlocks::lock_all_for_fork`] acquired.
    ///
    /// # Safety
    ///
    /// The calling thread must hold every shard lock through
    /// [`LiveBlocks::lock_all_for_fork`].
    pub unsafe fn unlock_all_for_fork(&self) {
        for shard in self.shards.iter().rev() {
            // SAFETY: delegated to the caller's obligation.
            unsafe { shard.lock.raw_unlock() };
        }
    }

    /// Re-initializes every shard lock after a `fork`.
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
    }
}

impl Default for LiveBlocks {
    fn default() -> Self {
        Self::new()
    }
}

/// Every shard of a [`LiveBlocks`], locked. Releases them when dropped.
///
/// See [`LiveBlocks::lock_all_until`].
pub struct AllShards<'a> {
    /// Declared first so that the locks are released before the order checker
    /// is told the level was left.
    guards: [super::lock::RawGuard<'a>; SHARDS],
    table: &'a LiveBlocks,
    _order: super::order::Entered,
}

impl AllShards<'_> {
    /// Visits every tracked block with leave to change what is remembered
    /// about it.
    ///
    /// Order is unspecified. The address cannot change, for the reason
    /// [`RawMap::for_each_mut`] gives.
    pub fn for_each_mut(&mut self, mut visit: impl FnMut(usize, &mut LiveBlock)) {
        for shard in &self.table.shards {
            // SAFETY: `self.guards` holds every shard's lock, this one's
            // included, for as long as `self` lives; and `&mut self` means no
            // other borrow of the map can be live through this value.
            let blocks = unsafe { &mut *shard.blocks.get() };
            blocks.for_each_mut(|address, block| visit(address as usize, block));
        }
    }
}

impl fmt::Debug for AllShards<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AllShards")
            .field("shards", &self.guards.len())
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for LiveBlocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiveBlocks")
            .field("live", &self.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internals::miri_scale;

    /// Builds a record without interning a real call stack, so that the table
    /// can be tested independently of the program-point machinery.
    fn block(pp: u32, birth: u64) -> LiveBlock {
        LiveBlock::unattributed(birth, PpId::from_raw(pp))
    }

    #[test]
    fn insert_and_remove_round_trip() {
        let arena = Arena::new();
        let table = LiveBlocks::with_capacity(1 << 16);

        let address = 0x6000_1234_0000usize;
        assert!(table.insert(&arena, address, block(7, 42)));
        assert_eq!(table.len(), 1);

        let removed = table.remove(address).expect("block should be present");
        assert_eq!(removed.birth, 42);
        assert!(table.is_empty());
    }

    /// The membership property: freeing something never recorded is normal and
    /// must be a quiet `None`, not an error and not an underflow.
    #[test]
    fn removing_an_unknown_pointer_is_silent() {
        let table = LiveBlocks::with_capacity(1 << 12);
        assert_eq!(table.remove(0xDEAD_BEEF), None);
        assert!(table.is_empty());
    }

    #[test]
    fn many_blocks_are_all_recoverable() {
        let arena = Arena::new();
        let table = LiveBlocks::with_capacity(1 << 18);

        let base = 0x6000_0000_0000usize;
        let count = miri_scale(20_000);
        for i in 0..count {
            assert!(table.insert(&arena, base + i * 32, block(i as u32 % 100, i as u64)));
        }
        assert_eq!(table.len(), count);

        for i in 0..count {
            let removed = table
                .remove(base + i * 32)
                .unwrap_or_else(|| panic!("block {i} was lost"));
            assert_eq!(removed.birth, i as u64);
        }
        assert!(table.is_empty());
    }

    /// The reason pointers are hashed before sharding: real allocations are
    /// aligned and consecutive, and would otherwise land on one shard.
    #[test]
    fn consecutive_aligned_pointers_spread_across_shards() {
        let mut hit = [0usize; SHARDS];
        let base = 0x6000_1234_0000usize;
        for i in 0..10_000usize {
            hit[LiveBlocks::shard_of(base + i * 16)] += 1;
        }
        let used = hit.iter().filter(|&&count| count > 0).count();
        assert_eq!(used, SHARDS, "only {used}/{SHARDS} shards were used");

        let worst = *hit.iter().max().unwrap();
        let average = 10_000 / SHARDS;
        assert!(
            worst < average * 2,
            "shard distribution is lopsided: worst {worst}, average {average}"
        );
    }

    #[test]
    fn a_64_bit_birth_survives_what_would_wrap_a_32_bit_one() {
        let arena = Arena::new();
        let table = LiveBlocks::with_capacity(1 << 12);

        // A block born just before a 32-bit counter would wrap, freed just
        // after. With a 32-bit birth the computed lifetime would be tiny
        // instead of enormous, silently.
        let birth = u32::MAX as u64 - 10;
        let death = birth + 5_000_000;
        table.insert(
            &arena,
            0x1_0000,
            LiveBlock::unattributed(birth, PpId::OVERFLOW),
        );

        let removed = table.remove(0x1_0000).unwrap();
        assert_eq!(removed.birth, birth);
        assert_eq!(death - removed.birth, 5_000_000, "the lifetime wrapped");
    }

    #[test]
    fn reaching_the_ceiling_reports_rather_than_growing() {
        // Both the ceiling and the attempt count scale, so the ceiling is still
        // reached. Scaling only the count would make this assert nothing.
        let count = miri_scale(200_000);
        let capacity = SHARDS * (count / 3_000).max(2);
        let arena = Arena::new();
        let table = LiveBlocks::with_capacity(capacity);

        let mut tracked = 0;
        let mut dropped = 0;
        for i in 0..count {
            if table.insert(&arena, 0x1_0000 + i * 16, block(1, i as u64)) {
                tracked += 1;
            } else {
                dropped += 1;
            }
        }
        assert!(tracked > 0, "the ceiling was too low to track anything");
        assert!(dropped > 0, "the ceiling was never reached");
    }

    /// The ceiling is a *setting*, and a setting nothing applies is not one.
    ///
    /// Raised rather than lowered, because growth is one way: a table asked for
    /// less than it has already allocated stops growing, which is a different
    /// claim and not the one the builder makes.
    #[test]
    fn a_raised_ceiling_tracks_what_the_first_one_dropped() {
        let count = miri_scale(20_000);
        let narrow = SHARDS * 2;

        let fill = |table: &LiveBlocks| {
            let arena = Arena::new();
            let mut tracked = 0;
            for i in 0..count {
                if table.insert(&arena, 0x1_0000 + i * 16, block(1, i as u64)) {
                    tracked += 1;
                }
            }
            tracked
        };

        let cramped = LiveBlocks::with_capacity(narrow);
        let before = fill(&cramped);

        let raised = LiveBlocks::with_capacity(narrow);
        raised.set_max_blocks(count);
        let after = fill(&raised);

        assert!(
            after > before,
            "raising the ceiling from {narrow} to {count} tracked {after} \
             blocks against {before}, so the setting did nothing"
        );
        assert_eq!(
            raised.max_blocks(),
            effective_ceiling(count),
            "the profile would report a ceiling the table is not using"
        );
    }

    /// The number in a profile has to be the number in force, or `droppedBlocks`
    /// reads as a contradiction against it. Both directions matter: a request
    /// below what 64 shards can express is raised to the smallest real table —
    /// two slots per shard, holding one live block each —
    /// and one that is not a whole power of two per shard is rounded up.
    #[test]
    fn the_ceiling_reported_is_the_one_the_table_enforces() {
        let table = LiveBlocks::with_capacity(1 << 16);

        table.set_max_blocks(1);
        assert_eq!(table.max_blocks(), SHARDS, "the smallest real table");

        table.set_max_blocks(5_000);
        assert_eq!(table.max_blocks(), 8_192, "each shard rounds its share up");

        table.set_max_blocks(DEFAULT_MAX_LIVE_BLOCKS);
        assert_eq!(
            table.max_blocks(),
            DEFAULT_MAX_LIVE_BLOCKS,
            "the default is already a whole power of two per shard"
        );

        assert_eq!(
            LiveBlocks::with_capacity(5_000).max_blocks(),
            8_192,
            "a table built with a ceiling reports it as one told later does"
        );
    }

    /// The reported ceiling is a bound the table respects: it never tracks more.
    #[test]
    fn the_reported_ceiling_is_not_exceeded() {
        let ceiling = LiveBlocks::with_capacity(1);
        let arena = Arena::new();
        let mut tracked = 0usize;
        for i in 0..miri_scale(20_000) {
            if ceiling.insert(&arena, 0x1_0000 + i * 16, block(1, i as u64)) {
                tracked += 1;
            }
        }
        assert!(
            tracked <= ceiling.max_blocks(),
            "{tracked} blocks were tracked under a reported ceiling of {}",
            ceiling.max_blocks()
        );
        assert!(tracked > 0, "nothing was tracked, so this proves nothing");
    }

    #[test]
    fn for_each_sees_every_live_block() {
        let arena = Arena::new();
        let table = LiveBlocks::with_capacity(1 << 16);

        let base = 0x6000_0000_0000usize;
        let count = miri_scale(5_000);
        for i in 0..count {
            table.insert(&arena, base + i * 64, block(1, i as u64));
        }
        for i in (0..count).step_by(2) {
            table.remove(base + i * 64);
        }

        let mut seen = 0;
        table.for_each(|_address, _block| seen += 1);
        assert_eq!(seen, table.len());
        assert_eq!(seen, count / 2, "half the blocks were removed");
    }

    /// Holding every shard is what lets a restart mark the whole table in one
    /// state. What it marks has to be every live block, in every shard, and
    /// only what it marks: the program point and the attribution are what a
    /// later free needs, and losing either would move the wrong counters.
    #[test]
    fn every_shard_can_be_held_and_rewritten_at_once() {
        let arena = Arena::new();
        let table = LiveBlocks::with_capacity(1 << 16);

        let base = 0x7000_0000_0000usize;
        let count = miri_scale(2_000);
        for i in 0..count {
            table.insert(&arena, base + i * 64, block(i as u32 % 7, 10 + i as u64));
        }
        for i in (0..count).step_by(4) {
            table.remove(base + i * 64);
        }

        let mut shards_seen = [false; SHARDS];
        {
            let mut all = table
                .lock_all_until(None)
                .expect("with no deadline, every shard is waited for");
            all.for_each_mut(|address, block| {
                shards_seen[LiveBlocks::shard_of(address)] = true;
                block.birth = LiveBlock::CARRIED;
            });
        }
        assert!(
            shards_seen.iter().filter(|seen| **seen).count() > SHARDS / 2,
            "the blocks landed in too few shards for this to say anything \
             about walking all of them"
        );

        let mut carried = 0;
        table.for_each(|address, block| {
            let i = (address - base) / 64;
            assert!(block.is_carried(), "block {i} was not marked");
            assert_eq!(
                block.pp,
                PpId::from_raw(i as u32 % 7),
                "block {i} moved point"
            );
            carried += 1;
        });
        assert_eq!(carried, count - count.div_ceil(4));

        // Released on drop: the table is usable again from here.
        assert!(table.insert(&arena, base - 64, block(1, 1)));
    }

    /// A shard nobody releases costs the caller its deadline and no more, and
    /// leaves every other shard as it found it.
    ///
    /// The case the deadline exists for is a holder that is not running, which
    /// a test can stand in for by holding a shard itself: the lock is not
    /// reentrant, but `try_lock` on it fails rather than deadlocking.
    #[test]
    fn a_shard_held_past_the_deadline_releases_every_other_one() {
        let arena = Arena::new();
        let table = LiveBlocks::with_capacity(1 << 12);
        let last = SHARDS - 1;
        let held = table.shards[last].lock.lock();

        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(20);
        assert!(
            table.lock_all_until(Some(deadline)).is_none(),
            "a shard that is held was reported taken"
        );
        drop(held);

        // Every shard the failed attempt took was released: each can be taken
        // again, and a block can be inserted wherever it hashes.
        for shard in &table.shards {
            assert!(shard.lock.try_lock().is_some(), "a shard was left locked");
        }
        let base = 0x6000_0000_0000usize;
        for i in 0..miri_scale(256) {
            assert!(table.insert(&arena, base + i * 64, block(1, 1)));
        }
        let later = std::time::Instant::now() + std::time::Duration::from_secs(2);
        assert!(
            table.lock_all_until(Some(later)).is_some(),
            "with nothing held, every shard is taken"
        );
    }

    /// A carried block counts toward no lifetime, and the saturating difference
    /// is what makes that free on the paths that compute one. Pinned here
    /// because the free path relies on it without a branch, and a sentinel at
    /// any other value would quietly hand a warm-up's blocks a lifetime.
    #[test]
    fn a_carried_block_has_no_lifetime() {
        let carried = block(1, LiveBlock::CARRIED);
        assert!(carried.is_carried());
        for now in [0, 1, 1 << 40, u64::MAX - 1, u64::MAX] {
            assert_eq!(carried.lifetime_at(now), 0, "at {now}");
        }

        let born = block(1, 100);
        assert!(!born.is_carried());
        assert_eq!(born.lifetime_at(150), 50);
        assert_eq!(born.lifetime_at(50), 0, "a lifetime never wraps");
    }

    /// The hazard PLAN.md section 4.1 calls out: an address freed on one thread
    /// and immediately reused by an allocation on another must not lose the new
    /// owner's record.
    #[test]
    fn concurrent_insert_and_remove_keep_the_table_consistent() {
        #[cfg(miri)]
        const ROUNDS: usize = 30;
        #[cfg(not(miri))]
        const ROUNDS: usize = 5_000;
        const THREADS: usize = 8;

        let arena = Arena::new();
        let table = LiveBlocks::with_capacity(1 << 18);

        std::thread::scope(|s| {
            for t in 0..THREADS {
                let (arena, table) = (&arena, &table);
                s.spawn(move || {
                    // Disjoint address ranges per thread, so every insert has a
                    // matching remove and the final count must be exactly zero.
                    let base = 0x1_0000_0000usize + t * 0x1000_0000;
                    for i in 0..ROUNDS {
                        let address = base + i * 32;
                        assert!(table.insert(arena, address, block(t as u32, i as u64)));
                        let removed = table.remove(address).expect("own block vanished");
                        assert_eq!(removed.birth, i as u64);
                    }
                });
            }
        });

        assert!(
            table.is_empty(),
            "{} blocks leaked across concurrent churn",
            table.len()
        );
    }

    /// The composition of the two hash steps, which neither
    /// `pointer_like_keys_do_not_pile_into_one_slot` (unsharded map) nor
    /// `consecutive_aligned_pointers_spread_across_shards` (shard function
    /// alone) can see.
    #[test]
    fn sharding_does_not_collapse_the_map_index_within_a_shard() {
        use super::super::table::mix;

        const MAP_SLOTS: usize = 1 << 14;
        let base = 0x6000_1234_0000usize;
        let target_shard = 7usize;

        // Home slots used, within one shard, by the addresses routed to it.
        let mut home_slots = std::collections::BTreeSet::new();
        let mut routed = 0usize;
        // Pure hashing, no locks and no allocation, so this is cheap even under
        // Miri; the sample only has to be large enough for the ratio below to
        // mean something.
        let samples = miri_scale(200_000).max(20_000);
        for i in 0..samples {
            let address = base + i * 16;
            if LiveBlocks::shard_of(address) != target_shard {
                continue;
            }
            routed += 1;
            home_slots.insert(mix(address as u64) as usize & (MAP_SLOTS - 1));
        }

        assert!(
            routed > samples / (SHARDS * 4),
            "too few addresses routed to sample: {routed} of {samples}"
        );
        // With the shard and the slot drawn from the same bits, this was 1/64th
        // of the table. Anything near that means the two indices are correlated
        // again and probe sequences will be long.
        let coverage = home_slots.len() * 100 / MAP_SLOTS.min(routed);
        assert!(
            coverage > 50,
            "addresses in one shard used only {} distinct home slots out of {} \
             ({coverage}% of what they could); the shard index and the map index \
             are correlated",
            home_slots.len(),
            MAP_SLOTS.min(routed)
        );
    }

    /// The reserved address cannot be tracked, and may not disturb the table
    /// when offered.
    #[test]
    fn the_reserved_address_is_refused_rather_than_folded() {
        let arena = Arena::new();
        let table = LiveBlocks::with_capacity(1 << 12);

        table.insert(&arena, 0x1000, block(1, 1));
        assert!(!table.insert(&arena, 0, block(2, 2)), "null was tracked");

        assert_eq!(table.get(0), None);
        assert_eq!(table.remove(0), None);
        assert_eq!(table.len(), 1, "a refused address changed the table");
        assert!(table.get(0x1000).is_some(), "the real block was disturbed");
    }
}
