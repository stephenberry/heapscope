//! An open-addressing hash map that lives in the arena.
//!
//! Two structures in the profiler need a map on the allocator hot path: the
//! live-block table (`pointer -> owning program point`) and the program-point
//! intern table (`frame hash -> program point`). Neither may allocate through
//! the global allocator, so `HashMap` is unusable and this exists instead.
//!
//! # Shape
//!
//! Open addressing with linear probing, power-of-two capacity, and tombstones
//! for deletion. Linear probing rather than anything cleverer because the keys
//! are already well-distributed — pointers are hashed, and program-point keys
//! are hashes to begin with — so the cache locality of a linear scan wins.
//!
//! # Capacity is a hard limit, not a suggestion
//!
//! A memory-analysis tool with unbounded memory growth is a contradiction
//! (PLAN.md section 4.5). Growth stops at a caller-supplied ceiling, after which
//! [`RawMap::insert`] reports [`Insert::Full`] and the caller accounts for the
//! loss in the profile rather than the profiler quietly consuming the machine.

use std::alloc::Layout;
use std::fmt;
use std::ptr::NonNull;

use super::arena::Arena;

/// A key value reserved to mean "this slot has never been used".
const EMPTY: u64 = 0;

/// A key value reserved to mean "this slot held an entry that was removed".
///
/// Probing must continue through a tombstone, because an entry that collided
/// with the removed one may lie beyond it.
const TOMBSTONE: u64 = u64::MAX;

/// Load factor, as a fraction of capacity, at which the table grows.
///
/// Linear probing degrades sharply past about 70%; 50% keeps probe sequences
/// short at a cost of 2x memory, which for an 8-byte value is a good trade.
const MAX_LOAD_NUM: usize = 1;
const MAX_LOAD_DEN: usize = 2;

/// Smallest table, in slots.
const MIN_CAPACITY: usize = 1024;

/// One slot.
///
/// # Partial initialization
///
/// `key` is initialized for every slot as soon as the table is allocated;
/// `value` is initialized **only** in slots whose key is neither [`EMPTY`] nor
/// [`TOMBSTONE`]. Every read must therefore examine `key` first and reach
/// `value` only through a live key. Reading a whole `Entry` — even to discard
/// it — constructs a `V` from uninitialized memory, which is undefined
/// behaviour for any type with validity constraints, and is not hypothetical:
/// Miri caught exactly this in `grow` and `for_each`.
///
/// The alternative, initializing every value, would mean requiring `V: Default`
/// and writing megabytes on every resize to no purpose.
#[repr(C)]
struct Entry<V: Copy> {
    key: u64,
    value: V,
}

/// The outcome of an insertion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Insert {
    /// A new entry was added.
    Added,
    /// An existing entry with the same key was overwritten.
    Replaced,
    /// The table is at its capacity limit and the entry was dropped.
    ///
    /// The caller is expected to count this and surface it in the profile. It
    /// is never an error and never a panic: this runs inside an allocator.
    Full,
}

/// A fixed-ceiling open-addressing map from `u64` to a `Copy` value.
///
/// Not thread-safe on its own. Callers hold the appropriate shard lock.
///
/// # Reserved keys
///
/// `0` and `u64::MAX` mark empty and removed slots, so they cannot be stored.
/// An earlier version of this type quietly folded both onto `1`, which is a
/// worse answer than it looks: folding is not injective, so a genuine key of
/// `1` and a key of `0` became *the same entry*, and the map returned one
/// caller's value to another with nothing to indicate it had happened. Silent
/// data corruption is not an acceptable price for a convenience.
///
/// Both real callers satisfy the constraint by construction — heap pointers are
/// never null and never `u64::MAX` — and anything hashed should be passed
/// through [`RawMap::usable_key`] first.
pub struct RawMap<V: Copy> {
    /// `capacity` entries, or dangling when `capacity == 0`.
    entries: NonNull<Entry<V>>,
    /// Always a power of two, or zero before the first allocation.
    capacity: usize,
    /// Live entries.
    len: usize,
    /// Slots holding tombstones. Counted toward the load factor because they
    /// lengthen probe sequences exactly as live entries do.
    tombstones: usize,
    /// Ceiling on `capacity`, in slots.
    max_capacity: usize,
}

// SAFETY: `RawMap` owns arena memory that nothing else refers to; it carries no
// interior mutability and no thread affinity. Callers provide synchronization.
unsafe impl<V: Copy + Send> Send for RawMap<V> {}
// SAFETY: as above; `&RawMap` grants only reads.
unsafe impl<V: Copy + Sync> Sync for RawMap<V> {}

impl<V: Copy> RawMap<V> {
    /// Creates an empty map that will grow to at most `max_capacity` slots.
    ///
    /// No memory is reserved until the first insertion, so this is `const` and
    /// a map can live in a `static`.
    pub const fn new(max_capacity: usize) -> Self {
        Self {
            entries: NonNull::dangling(),
            capacity: 0,
            len: 0,
            tombstones: 0,
            max_capacity,
        }
    }

    /// Raises or lowers the ceiling on how far this map may grow.
    ///
    /// Raising takes effect at any time, for as long as the map has not already
    /// stopped at its ceiling. *Lowering* is the one-way case: growth cannot be
    /// undone, so a ceiling below the slots already allocated means "grow no
    /// further" rather than giving memory back. It is applied either way rather
    /// than ignored, because a limit that silently does nothing is worse than
    /// one that does less than asked.
    pub fn set_max_capacity(&mut self, slots: usize) {
        self.max_capacity = slots;
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the map holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Slots currently allocated.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes of arena memory currently held, for self-metrics.
    pub fn bytes(&self) -> usize {
        self.capacity * std::mem::size_of::<Entry<V>>()
    }

    /// Looks up `key`.
    ///
    /// See the type documentation for the constraint on `key`.
    pub fn get(&self, key: u64) -> Option<V> {
        if is_reserved(key) {
            return None;
        }
        let index = self.find(key)?;
        // SAFETY: `find` returns `Some` only for a slot below `capacity` whose
        // key matched, which means `insert` wrote `value` there.
        Some(unsafe { std::ptr::addr_of!((*self.slot(index)).value).read() })
    }

    /// Inserts or replaces `key`.
    ///
    /// Growth happens here, from `arena`, and stops at `max_capacity`.
    pub fn insert(&mut self, arena: &Arena, key: u64, value: V) -> Insert {
        // Reported, not asserted. A `debug_assert!` here would panic from inside
        // the allocator shim in debug builds — building the message allocates,
        // which re-enters — which is exactly the hazard `super::diagnostic`
        // exists to avoid. Refusing the entry loses one event and says so.
        if is_reserved(key) {
            return Insert::Full;
        }

        if self.needs_growth() && !self.grow(arena) {
            // Growth failed or the ceiling was reached. An existing key can
            // still be updated in place, which matters: refusing to update a
            // live block's record would corrupt accounting, whereas refusing to
            // add a new one merely loses an event.
            if let Some(index) = self.find(key) {
                // SAFETY: `find` returned a live slot below `capacity`.
                unsafe { std::ptr::addr_of_mut!((*self.slot(index)).value).write(value) };
                return Insert::Replaced;
            }
            return Insert::Full;
        }

        self.insert_no_grow(key, value)
    }

    /// Removes `key`, returning its value.
    pub fn remove(&mut self, key: u64) -> Option<V> {
        if is_reserved(key) {
            return None;
        }
        let index = self.find(key)?;
        // SAFETY: `find` returned a live slot below `capacity`.
        let value = unsafe {
            let slot = self.slot(index);
            let value = std::ptr::addr_of!((*slot).value).read();
            std::ptr::addr_of_mut!((*slot).key).write(TOMBSTONE);
            value
        };
        self.len -= 1;
        self.tombstones += 1;
        Some(value)
    }

    /// Visits every live entry.
    ///
    /// Used at output time to walk the surviving blocks, so iteration order is
    /// deliberately unspecified — callers that need determinism sort afterwards.
    pub fn for_each(&self, mut f: impl FnMut(u64, V)) {
        for index in 0..self.capacity {
            // The `key` field is read on its own, and `value` only once the key
            // proves the slot is live. `grow` initializes every key but leaves
            // values untouched, so reading a whole `Entry` from an empty slot
            // would construct a value out of uninitialized memory — undefined
            // behaviour, and exactly what Miri caught here.
            //
            // SAFETY: `index < capacity`, and `grow` initialized the `key`
            // field of every slot in that range.
            let key = unsafe { std::ptr::addr_of!((*self.slot(index)).key).read() };
            if key != EMPTY && key != TOMBSTONE {
                // SAFETY: a live key means `insert` wrote `value` in this slot.
                let value = unsafe { std::ptr::addr_of!((*self.slot(index)).value).read() };
                f(key, value);
            }
        }
    }

    /// Empties the map, keeping its allocation.
    pub fn clear(&mut self) {
        for index in 0..self.capacity {
            // SAFETY: `index < capacity`; the slot is initialized.
            unsafe { std::ptr::addr_of_mut!((*self.slot(index)).key).write(EMPTY) };
        }
        self.len = 0;
        self.tombstones = 0;
    }

    #[inline(always)]
    fn slot(&self, index: usize) -> *mut Entry<V> {
        // Not a `debug_assert!`: this runs inside the allocator shim, where a
        // panic allocates its own message and re-enters. Every caller derives
        // `index` from `& mask`, so the bound holds by construction; the
        // condition is documented rather than enforced with a panic.
        debug_assert!(
            index < self.capacity,
            "caller must mask the index; this is a contract, not a runtime check"
        );
        // SAFETY: the caller guarantees `index < capacity`, so the offset stays
        // within the single arena block `entries` points at.
        unsafe { self.entries.as_ptr().add(index) }
    }

    /// Returns the index of the live slot holding `key`.
    #[inline]
    fn find(&self, key: u64) -> Option<usize> {
        if self.capacity == 0 {
            return None;
        }
        let mask = self.capacity - 1;
        let mut index = mix(key) as usize & mask;

        // Bounded by capacity: the table is never full of live entries, so a
        // scan of every slot is guaranteed to reach an `EMPTY`. The bound is
        // belt-and-braces against a corrupted `len`.
        for _ in 0..self.capacity {
            // SAFETY: `index <= mask < capacity`.
            let slot_key = unsafe { std::ptr::addr_of!((*self.slot(index)).key).read() };
            // `EMPTY` is tested *first*, and the order is load-bearing. With the
            // key comparison first, a caller looking up key `0` matches the
            // first never-written slot in the probe sequence, and `get` then
            // reads that slot's `value` — which `grow` deliberately leaves
            // uninitialized. That is undefined behaviour, and it is the same
            // partial-initialization hazard documented on `Entry`, reintroduced
            // through the probe rather than through iteration.
            if slot_key == EMPTY {
                return None;
            }
            if slot_key == key {
                return Some(index);
            }
            index = (index + 1) & mask;
        }
        None
    }

    #[inline]
    fn needs_growth(&self) -> bool {
        // Tombstones count: they cost probe length exactly as live entries do,
        // so a table that is half tombstones is as slow as one that is half
        // full, and rehashing is what clears them.
        (self.len + self.tombstones + 1) * MAX_LOAD_DEN > self.capacity * MAX_LOAD_NUM
    }

    fn insert_no_grow(&mut self, key: u64, value: V) -> Insert {
        debug_assert!(self.capacity > 0);
        let mask = self.capacity - 1;
        let mut index = mix(key) as usize & mask;
        // The first tombstone seen, which can be reused if the key turns out to
        // be absent. Probing must still continue past it to find a live match.
        let mut reusable: Option<usize> = None;

        for _ in 0..self.capacity {
            // SAFETY: `index <= mask < capacity`.
            let slot_key = unsafe { std::ptr::addr_of!((*self.slot(index)).key).read() };

            if slot_key == key {
                // SAFETY: as above.
                unsafe { std::ptr::addr_of_mut!((*self.slot(index)).value).write(value) };
                return Insert::Replaced;
            }
            if slot_key == TOMBSTONE {
                reusable.get_or_insert(index);
            } else if slot_key == EMPTY {
                let target = reusable.unwrap_or(index);
                if reusable.is_some() {
                    self.tombstones -= 1;
                }
                // SAFETY: `target` is an index this probe visited, so it is
                // below `capacity`.
                unsafe { self.slot(target).write(Entry { key, value }) };
                self.len += 1;
                return Insert::Added;
            }
            index = (index + 1) & mask;
        }

        // Unreachable while the load factor is enforced, but this runs inside
        // an allocator: reporting fullness beats asserting.
        Insert::Full
    }

    /// Doubles the table, or creates it. Returns `false` if the ceiling is
    /// reached or the arena refuses.
    #[cold]
    fn grow(&mut self, arena: &Arena) -> bool {
        let new_capacity = if self.capacity == 0 {
            MIN_CAPACITY.min(self.max_capacity.next_power_of_two())
        } else {
            match self.capacity.checked_mul(2) {
                Some(doubled) => doubled,
                None => return false,
            }
        };

        if new_capacity > self.max_capacity || new_capacity == 0 {
            return false;
        }

        let Ok(layout) = Layout::array::<Entry<V>>(new_capacity) else {
            return false;
        };
        let Some(memory) = arena.alloc(layout) else {
            return false;
        };
        let entries = memory.cast::<Entry<V>>();

        // Slots are only ever read through `key`, so initializing the key is
        // enough to make every slot well-defined; the value stays uninitialized
        // until something is stored, and is never read before then.
        for index in 0..new_capacity {
            // SAFETY: `index < new_capacity`, the length of the block just
            // allocated, and the write is to a field of a `repr(C)` struct at a
            // known offset.
            unsafe { std::ptr::addr_of_mut!((*entries.as_ptr().add(index)).key).write(EMPTY) };
        }

        // Rehash into the new table. The arena never frees, so the old block
        // stays reserved; it is accounted for in `ArenaStats::bytes_used` and
        // reclaimed wholesale at reset. Buying simplicity with memory is the
        // right trade here — table growth is logarithmic in the number of
        // distinct keys, so the total waste is bounded by the final size.
        let old_entries = self.entries;
        let old_capacity = self.capacity;

        self.entries = entries;
        self.capacity = new_capacity;
        self.len = 0;
        self.tombstones = 0;

        for index in 0..old_capacity {
            // Read `key` alone first; see `for_each` for why reading a whole
            // `Entry` from a slot that was never written is undefined.
            //
            // SAFETY: `index < old_capacity`, and the old block is still valid
            // because the arena never frees. `grow` initialized every key.
            let slot = unsafe { old_entries.as_ptr().add(index) };
            // SAFETY: `slot` is within the old block and its `key` was
            // initialized when that block was created.
            let key = unsafe { std::ptr::addr_of!((*slot).key).read() };
            if key != EMPTY && key != TOMBSTONE {
                // SAFETY: a live key means `value` was written in this slot.
                let value = unsafe { std::ptr::addr_of!((*slot).value).read() };
                self.insert_no_grow(key, value);
            }
        }

        true
    }
}

impl<V: Copy> fmt::Debug for RawMap<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawMap")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("tombstones", &self.tombstones)
            .field("max_capacity", &self.max_capacity)
            .finish()
    }
}

impl<V: Copy> RawMap<V> {
    /// Maps an arbitrary `u64` into the range this map can store.
    ///
    /// Sets the low bit and clears the high bit, so the result is never `0` and
    /// never `u64::MAX`. That costs two bits of a 64-bit hash, which changes
    /// the collision rate by nothing that could ever be measured, and unlike
    /// folding onto a fixed value it never maps two *distinct* useful hashes
    /// together any more often than hashing already does.
    #[inline(always)]
    pub const fn usable_key(raw: u64) -> u64 {
        (raw | 1) & !(1 << 63)
    }
}

/// Whether `key` is one of the two values the table reserves for slot state.
#[inline(always)]
const fn is_reserved(key: u64) -> bool {
    key == EMPTY || key == TOMBSTONE
}

/// Finalizer for 64-bit hashes, from SplitMix64.
///
/// Heap pointers cluster: they are aligned, so the low bits are zero, and
/// consecutive allocations differ only in a narrow middle range. Masking such a
/// value directly would pile every entry into a handful of slots. This spreads
/// the input across all 64 bits in a few instructions.
#[inline(always)]
pub const fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internals::miri_scale;

    fn map(max: usize) -> (Arena, RawMap<u32>) {
        (Arena::new(), RawMap::new(max))
    }

    /// Miri interprets every instruction, so the native loop counts turn a few
    /// of these into minutes each. Scaling them keeps Miri in the ordinary CI
    /// path; the properties under test -- growth, tombstone reuse, key
    /// distribution -- all show up well below the native counts.
    /// Deliberately larger than [`miri_scale`] would give. Two tests below turn
    /// on crossing `MIN_CAPACITY`'s load factor, so a value under 512 would stop
    /// them testing growth and fullness at all — which is exactly what happened
    /// when a blanket scaling was applied.
    #[cfg(miri)]
    const SCALE: usize = 1_500;
    #[cfg(not(miri))]
    const SCALE: usize = 50_000;

    const _: () = assert!(
        SCALE > MIN_CAPACITY,
        "the growth and ceiling tests need a count that crosses MIN_CAPACITY"
    );

    #[test]
    fn insert_and_get_round_trip() {
        let (arena, mut m) = map(1 << 16);
        let count = miri_scale(5000) as u64;
        for i in 1..=count {
            assert_eq!(m.insert(&arena, i, i as u32), Insert::Added);
        }
        assert_eq!(m.len(), count as usize);
        for i in 1..=count {
            assert_eq!(m.get(i), Some(i as u32), "lost key {i}");
        }
        assert_eq!(m.get(count + 1), None);
    }

    #[test]
    fn insert_replaces_an_existing_key_without_growing_len() {
        let (arena, mut m) = map(1 << 12);
        assert_eq!(m.insert(&arena, 42, 1), Insert::Added);
        assert_eq!(m.insert(&arena, 42, 2), Insert::Replaced);
        assert_eq!(m.len(), 1);
        assert_eq!(m.get(42), Some(2));
    }

    #[test]
    fn remove_makes_a_key_absent_without_hiding_its_neighbours() {
        let (arena, mut m) = map(1 << 14);
        let count = miri_scale(2000) as u64;
        for i in 1..=count {
            m.insert(&arena, i, i as u32);
        }
        for i in (2..=count).step_by(2) {
            assert_eq!(m.remove(i), Some(i as u32));
        }
        for i in 1..=count {
            let expected = if i % 2 == 0 { None } else { Some(i as u32) };
            assert_eq!(
                m.get(i),
                expected,
                "wrong result for key {i} after removals"
            );
        }
        assert_eq!(m.len(), count as usize / 2);
    }

    /// The classic open-addressing bug: probing stops at a tombstone and an
    /// entry that collided with the removed key becomes unreachable.
    #[test]
    fn probing_continues_past_tombstones() {
        let (arena, mut m) = map(1 << 12);
        // Force a collision chain by using keys that hash into the same slot.
        // Rather than reverse the hash, insert enough keys that chains are
        // certain, then delete from the middle of the table.
        let count = miri_scale(1000) as u64;
        for i in 0..count {
            m.insert(&arena, i * 7 + 1, i as u32);
        }
        for i in (0..count).step_by(3) {
            m.remove(i * 7 + 1);
        }
        for i in 0..count {
            let key = i * 7 + 1;
            let expected = if i % 3 == 0 { None } else { Some(i as u32) };
            assert_eq!(m.get(key), expected, "key {key} lost behind a tombstone");
        }
    }

    #[test]
    fn reinserting_a_removed_key_reuses_its_tombstone() {
        let (arena, mut m) = map(1 << 12);
        for i in 1..=500u64 {
            m.insert(&arena, i, i as u32);
        }
        let capacity_before = m.capacity();
        // Churn far more than the table could hold if tombstones accumulated
        // without being reclaimed.
        #[cfg(miri)]
        const ROUNDS: u64 = 3;
        #[cfg(not(miri))]
        const ROUNDS: u64 = 50;
        for round in 0..ROUNDS {
            for i in 1..=500u64 {
                m.remove(i);
                m.insert(&arena, i, (i + round) as u32);
            }
        }
        assert_eq!(m.len(), 500);
        assert_eq!(
            m.capacity(),
            capacity_before,
            "steady-state churn grew the table; tombstones are not being reclaimed"
        );
    }

    #[test]
    fn growth_preserves_every_entry() {
        let (arena, mut m) = map(1 << 20);
        let mut expected = Vec::new();
        for i in 0..SCALE as u64 {
            let key = RawMap::<u32>::usable_key(mix(i));
            m.insert(&arena, key, i as u32);
            expected.push((key, i as u32));
        }
        assert!(m.capacity() > MIN_CAPACITY, "the table never grew");
        for (key, value) in expected {
            assert_eq!(m.get(key), Some(value), "key {key} lost across a resize");
        }
    }

    #[test]
    fn reaching_the_ceiling_reports_full_instead_of_growing() {
        let (arena, mut m) = map(MIN_CAPACITY);
        let mut added = 0;
        let mut full = 0;
        for i in 0..10_000u64 {
            match m.insert(&arena, i + 1, i as u32) {
                Insert::Added => added += 1,
                Insert::Full => full += 1,
                Insert::Replaced => unreachable!("keys are distinct"),
            }
        }
        assert!(added > 0, "the ceiling was too low to accept anything");
        assert!(full > 0, "the ceiling was never reached");
        assert!(m.capacity() <= MIN_CAPACITY);
        assert_eq!(m.len(), added);
    }

    /// Refusing to *update* an existing key once full would corrupt accounting,
    /// where refusing to add a new one merely loses an event.
    #[test]
    fn a_full_table_still_updates_existing_keys() {
        let (arena, mut m) = map(MIN_CAPACITY);
        for i in 0..miri_scale(10_000) as u64 {
            m.insert(&arena, i + 1, i as u32);
        }
        let present = (1..=10_000u64).find(|&k| m.get(k).is_some()).unwrap();
        assert_eq!(m.insert(&arena, present, 0xDEAD), Insert::Replaced);
        assert_eq!(m.get(present), Some(0xDEAD));
    }

    /// `usable_key` must never produce a reserved value, for any input at all —
    /// including the two that are themselves reserved.
    #[test]
    fn usable_key_never_produces_a_reserved_value() {
        let interesting = [
            0,
            1,
            2,
            u64::MAX,
            u64::MAX - 1,
            1 << 63,
            (1 << 63) - 1,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
        ];
        for raw in interesting {
            let key = RawMap::<u32>::usable_key(raw);
            assert_ne!(key, EMPTY, "usable_key({raw:#x}) produced the empty marker");
            assert_ne!(
                key, TOMBSTONE,
                "usable_key({raw:#x}) produced the tombstone marker"
            );
        }
        for i in 0..(SCALE * 2) as u64 {
            for raw in [mix(i), mix(i).wrapping_neg(), i] {
                let key = RawMap::<u32>::usable_key(raw);
                assert_ne!(key, EMPTY);
                assert_ne!(key, TOMBSTONE);
            }
        }
    }

    /// The reserved values must stay distinguishable from real keys after
    /// normalisation, so that two distinct inputs do not become one entry.
    #[test]
    fn normalised_reserved_keys_do_not_collide_with_their_neighbours() {
        let (arena, mut m) = map(1 << 12);
        let zero = RawMap::<u32>::usable_key(0);
        let max = RawMap::<u32>::usable_key(u64::MAX);
        assert_ne!(zero, max, "0 and u64::MAX normalised to the same key");

        m.insert(&arena, zero, 10);
        m.insert(&arena, max, 20);
        assert_eq!(m.get(zero), Some(10));
        assert_eq!(m.get(max), Some(20));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn for_each_visits_exactly_the_live_entries() {
        let (arena, mut m) = map(1 << 14);
        let count = miri_scale(3000) as u64;
        for i in 0..count {
            m.insert(&arena, i + 1, i as u32);
        }
        for i in (0..count).step_by(2) {
            m.remove(i + 1);
        }

        let mut seen = Vec::new();
        m.for_each(|key, value| seen.push((key, value)));
        assert_eq!(seen.len(), m.len());
        seen.sort_unstable();
        for (key, value) in seen {
            assert_eq!(m.get(key), Some(value));
        }
    }

    #[test]
    fn clear_empties_without_releasing_the_allocation() {
        let (arena, mut m) = map(1 << 14);
        for i in 0..miri_scale(3000) as u64 {
            m.insert(&arena, i + 1, i as u32);
        }
        let capacity = m.capacity();
        m.clear();
        assert!(m.is_empty());
        assert_eq!(m.capacity(), capacity);
        assert_eq!(m.get(1), None);
        assert_eq!(m.insert(&arena, 1, 5), Insert::Added);
    }

    #[test]
    fn pointer_like_keys_do_not_pile_into_one_slot() {
        // Real heap pointers are 16-byte aligned and consecutive, which is the
        // distribution that destroys a table using the key directly as an index.
        let (arena, mut m) = map(1 << 16);
        let base = 0x0000_6000_1234_0000u64;
        let count = SCALE.min(10_000) as u64;
        for i in 0..count {
            m.insert(&arena, base + i * 16, i as u32);
        }
        for i in 0..count {
            assert_eq!(m.get(base + i * 16), Some(i as u32));
        }
    }

    /// The two reserved values must be *inert*, not merely unlikely. Looking up
    /// key 0 once matched the first never-written slot and read its
    /// uninitialized value, which Miri reports as undefined behaviour.
    #[test]
    fn reserved_keys_are_inert_rather_than_matching_empty_slots() {
        let (arena, mut m) = map(1 << 12);
        for i in 1..=100u64 {
            m.insert(&arena, i, i as u32);
        }

        assert_eq!(m.get(EMPTY), None, "key 0 matched an empty slot");
        assert_eq!(m.get(TOMBSTONE), None, "u64::MAX matched a slot");
        assert_eq!(m.remove(EMPTY), None);
        assert_eq!(m.remove(TOMBSTONE), None);
        assert_eq!(m.insert(&arena, EMPTY, 1), Insert::Full);
        assert_eq!(m.insert(&arena, TOMBSTONE, 1), Insert::Full);
        assert_eq!(m.len(), 100, "a reserved key changed the table's length");
    }

    /// The same lookup against a completely empty table, where every slot in the
    /// probe sequence is uninitialized.
    #[test]
    fn reserved_keys_are_inert_on_an_empty_table() {
        let (arena, mut m) = map(1 << 12);
        m.insert(&arena, 1, 1);
        m.remove(1);
        assert_eq!(m.get(EMPTY), None);
        assert_eq!(m.get(TOMBSTONE), None);
    }
}
