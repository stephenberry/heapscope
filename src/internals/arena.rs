//! A bump arena for profiler state, allocated outside the global allocator.
//!
//! # Why the arena exists
//!
//! Every byte of profiler state — interned program points, frame arrays, module
//! records — has to come from somewhere, and that somewhere cannot be the global
//! allocator, because the global allocator is the thing being instrumented.
//! Allocating from inside `GlobalAlloc::alloc` re-enters the shim, and the
//! reentrancy guard turns that into either lost data or, without care, infinite
//! recursion. This is the structural problem that forced `dhat-rs` onto a
//! hand-rolled mutex; a bump arena removes it at the root.
//!
//! # Where the memory comes from
//!
//! [`std::alloc::System`], which is `malloc`/`HeapAlloc` reached directly. It
//! does not route through `#[global_allocator]`, so it cannot recurse.
//!
//! PLAN.md section 4.1 says the arena requests memory "from the *inner*
//! allocator". Using `System` instead is a deliberate strengthening. The plan's
//! own section 7 states the contract that `Alloc<A>` puts on `A`: *`A` must not
//! allocate through the global allocator*. Routing arena refills through `A`
//! would make the profiler's own correctness depend on a user upholding that
//! contract — and the failure mode when they do not is unbounded recursion
//! inside an allocator, which is close to undiagnosable. `System` is reachable
//! on every supported platform, cannot recurse by construction, and makes the
//! profiler's storage independent of what the program under test chose to use.
//!
//! # Lifetime
//!
//! Nothing is ever freed individually. Chunks are released only by
//! [`Arena::reset`], which invalidates every pointer the arena has handed out
//! and is therefore `unsafe`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::UnsafeCell;
use std::fmt;
use std::ptr::NonNull;

use super::lock::RawLock;

/// Alignment of every chunk. Covers `ChunkHeader` and all natural alignments up
/// to 16 bytes, which is every type this crate stores in the arena.
const CHUNK_ALIGN: usize = 16;

/// First chunk size. Small enough that a profiler attached to a short-lived
/// process does not reserve megabytes it never touches.
const FIRST_CHUNK: usize = 64 * 1024;

/// Chunk sizes double until they reach this, which bounds the waste from the
/// final, partially-filled chunk.
const MAX_CHUNK: usize = 4 * 1024 * 1024;

/// Default ceiling on total arena bytes.
///
/// A memory-analysis tool with unbounded memory growth is a contradiction
/// (PLAN.md section 4.5). Exhaustion is reported, never fatal: the caller
/// accounts for it and degrades.
const DEFAULT_LIMIT: usize = 512 * 1024 * 1024;

#[repr(C)]
struct ChunkHeader {
    /// Next-oldest chunk, or null.
    next: *mut ChunkHeader,
    /// Total size of this chunk in bytes, header included. Retained so that
    /// [`Arena::reset`] can reconstruct the exact `Layout` that `System`
    /// requires for deallocation.
    size: usize,
}

struct ArenaState {
    /// Newest chunk; head of the free-on-reset list.
    chunks: *mut ChunkHeader,
    /// Next unhanded-out byte in the newest chunk.
    cursor: *mut u8,
    /// One past the last usable byte in the newest chunk.
    end: *mut u8,
    /// Size to request for the next chunk.
    next_chunk: usize,

    bytes_reserved: usize,
    bytes_used: usize,
    chunk_count: usize,
    limit: usize,
    /// Requests refused because the limit was reached. Surfaced in self-metrics
    /// so that a truncated profile is visibly truncated.
    refused: usize,
}

/// A thread-safe bump allocator that never touches the global allocator.
pub struct Arena {
    lock: RawLock,
    state: UnsafeCell<ArenaState>,
}

// SAFETY: every access to `state` happens under `lock`, and the memory the
// arena hands out is plain bytes whose ownership passes to the caller.
unsafe impl Send for Arena {}
// SAFETY: as above.
unsafe impl Sync for Arena {}

/// A snapshot of arena occupancy, for the self-metrics block of a profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaStats {
    /// Bytes obtained from the system allocator.
    pub bytes_reserved: usize,
    /// Bytes handed out to callers, including alignment padding.
    pub bytes_used: usize,
    /// Number of chunks currently held.
    pub chunks: usize,
    /// Allocation requests refused because the limit was reached.
    pub refused: usize,
    /// The current ceiling on `bytes_reserved`.
    pub limit: usize,
}

impl Default for ArenaStats {
    /// An arena that has never been asked for anything.
    ///
    /// Not a derived all-zeroes: a limit of zero would say the arena may hold
    /// nothing, which is a different claim from "it holds nothing yet" and is
    /// the one a reader would act on.
    fn default() -> Self {
        Self {
            bytes_reserved: 0,
            bytes_used: 0,
            chunks: 0,
            refused: 0,
            limit: DEFAULT_LIMIT,
        }
    }
}

impl Arena {
    /// Creates an empty arena. No memory is reserved until the first allocation.
    ///
    /// `const` so that the arena can be a plain `static`: the shim is live
    /// before `main`, so nothing it reaches may require lazy initialization.
    pub const fn new() -> Self {
        Self {
            lock: RawLock::new(),
            state: UnsafeCell::new(ArenaState {
                chunks: std::ptr::null_mut(),
                cursor: std::ptr::null_mut(),
                end: std::ptr::null_mut(),
                next_chunk: FIRST_CHUNK,
                bytes_reserved: 0,
                bytes_used: 0,
                chunk_count: 0,
                limit: DEFAULT_LIMIT,
                refused: 0,
            }),
        }
    }

    /// Sets the ceiling on bytes reserved from the system allocator.
    ///
    /// Lowering the limit below what is already reserved does not release
    /// anything; it only prevents further growth.
    pub fn set_limit(&self, limit: usize) {
        let _order = super::order::enter(super::order::Level::Arena);
        let _guard = self.lock.lock();
        // SAFETY: `state` is only ever reached while holding `lock`.
        let state = unsafe { &mut *self.state.get() };
        state.limit = limit;
    }

    /// Allocates `layout` bytes, or returns `None` if the arena limit is
    /// reached or the system allocator refuses.
    ///
    /// Returning `None` rather than panicking is deliberate: this runs inside
    /// the allocator, where a panic would unwind through a `GlobalAlloc` method
    /// and abort the program under test.
    ///
    /// The returned memory is uninitialized.
    pub fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        let _order = super::order::enter(super::order::Level::Arena);
        let _guard = self.lock.lock();
        // SAFETY: `state` is only ever reached while holding `lock`.
        let state = unsafe { &mut *self.state.get() };
        Self::alloc_locked(state, layout)
    }

    /// Allocates space for a `T` and initializes it with `value`.
    ///
    /// Returns a reference valid until [`Arena::reset`]; the arena never frees
    /// individual allocations, so the `'static`-like lifetime is tied to `self`.
    ///
    /// Handing out `&mut` from `&self` is the defining move of a bump allocator
    /// and is sound here for the same reason it is in `bumpalo`: each call
    /// returns a disjoint block, so no two references can alias. `reset`, the
    /// one operation that would invalidate them, is `unsafe`, and `Drop` reaches
    /// it only through `&mut self` — which the borrow checker will not produce
    /// while any of these references is live.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_value<T>(&self, value: T) -> Option<&mut T> {
        let ptr = self.alloc(Layout::new::<T>())?.cast::<T>();
        // SAFETY: `alloc` returned a block of exactly `size_of::<T>()` bytes
        // aligned to `align_of::<T>()`, uniquely owned by this call, and no
        // other reference to it exists.
        unsafe {
            ptr.as_ptr().write(value);
            Some(&mut *ptr.as_ptr())
        }
    }

    /// Copies `slice` into the arena.
    ///
    /// This is the frame-array path: the unwinder writes return addresses into
    /// a fixed stack buffer, and only the frames that turn out to belong to a
    /// newly interned program point are copied here.
    #[allow(clippy::mut_from_ref)] // See `alloc_value`.
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> Option<&mut [T]> {
        if slice.is_empty() {
            return Some(&mut []);
        }
        let layout = Layout::array::<T>(slice.len()).ok()?;
        let ptr = self.alloc(layout)?.cast::<T>();
        // SAFETY: `alloc` returned `slice.len() * size_of::<T>()` bytes aligned
        // for `T` and uniquely owned by this call. `T: Copy` rules out overlap
        // concerns with a source that has a destructor, and the arena block
        // cannot alias `slice` because it was just obtained from `System`.
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), ptr.as_ptr(), slice.len());
            Some(std::slice::from_raw_parts_mut(ptr.as_ptr(), slice.len()))
        }
    }

    /// Reports current occupancy.
    pub fn stats(&self) -> ArenaStats {
        let _order = super::order::enter(super::order::Level::Arena);
        let _guard = self.lock.lock();
        // SAFETY: `state` is only ever reached while holding `lock`.
        let state = unsafe { &*self.state.get() };
        ArenaStats {
            bytes_reserved: state.bytes_reserved,
            bytes_used: state.bytes_used,
            chunks: state.chunk_count,
            refused: state.refused,
            limit: state.limit,
        }
    }

    /// Releases every chunk and returns the arena to its initial state.
    ///
    /// # Safety
    ///
    /// Every pointer and reference previously returned by this arena is
    /// invalidated. The caller must ensure none is still reachable — in
    /// practice, that the profiler has been fully torn down first.
    pub unsafe fn reset(&self) {
        let _order = super::order::enter(super::order::Level::Arena);
        let _guard = self.lock.lock();
        // SAFETY: `state` is only ever reached while holding `lock`.
        let state = unsafe { &mut *self.state.get() };

        let mut chunk = state.chunks;
        while !chunk.is_null() {
            // SAFETY: `chunk` came from this list, so it points at a live
            // `ChunkHeader` written by `grow`.
            let (next, size) = unsafe { ((*chunk).next, (*chunk).size) };
            // SAFETY: `size` and `CHUNK_ALIGN` are exactly the values `grow`
            // used to allocate this block, which is what `System::dealloc`
            // requires.
            unsafe {
                let layout = Layout::from_size_align_unchecked(size, CHUNK_ALIGN);
                System.dealloc(chunk.cast::<u8>(), layout);
            }
            chunk = next;
        }

        state.chunks = std::ptr::null_mut();
        state.cursor = std::ptr::null_mut();
        state.end = std::ptr::null_mut();
        state.next_chunk = FIRST_CHUNK;
        state.bytes_reserved = 0;
        state.bytes_used = 0;
        state.chunk_count = 0;
        state.refused = 0;
    }

    /// Acquires the arena lock, for a `fork` prepare handler.
    ///
    /// # Safety
    ///
    /// A matching [`Arena::unlock_for_fork`] must run on the same thread, or the
    /// child must reset the lock with [`Arena::reinit_after_fork`].
    pub unsafe fn lock_for_fork(&self) {
        // SAFETY: delegated to the caller's pairing obligation.
        unsafe { self.lock.raw_lock() }
    }

    /// Releases what [`Arena::lock_for_fork`] acquired.
    ///
    /// # Safety
    ///
    /// The calling thread must hold the lock through [`Arena::lock_for_fork`].
    pub unsafe fn unlock_for_fork(&self) {
        // SAFETY: delegated to the caller's obligation.
        unsafe { self.lock.raw_unlock() }
    }

    /// Re-initializes the arena lock after a `fork`.
    ///
    /// The child inherits the parent's memory, so the chunks and their contents
    /// remain valid; only the lock may have been orphaned by a thread that does
    /// not exist in the child.
    ///
    /// # Safety
    ///
    /// Call only from a `pthread_atfork` child handler, where the process is
    /// single-threaded by definition.
    pub unsafe fn reinit_after_fork(&self) {
        // SAFETY: delegated to the caller's single-threadedness obligation.
        unsafe { self.lock.force_reinit() }
    }

    fn alloc_locked(state: &mut ArenaState, layout: Layout) -> Option<NonNull<u8>> {
        // A zero-sized request still has to return a non-null aligned pointer.
        // Handing back the (aligned) cursor without advancing it is correct and
        // costs nothing.
        if layout.size() == 0 {
            return Some(NonNull::dangling());
        }

        for _ in 0..2 {
            if let Some(ptr) = Self::bump(state, layout) {
                return Some(ptr);
            }
            if !Self::grow(state, layout) {
                state.refused = state.refused.saturating_add(1);
                return None;
            }
        }
        // A fresh chunk is always sized to fit the request, so the second bump
        // cannot fail. If it somehow does, that is an internal invariant
        // violation: poison and refuse. Deliberately not `debug_assert!` --
        // this runs inside the allocator, where a panic allocates its own
        // message and re-enters.
        super::diagnostic::poison("arena bump failed immediately after a sized grow");
        state.refused = state.refused.saturating_add(1);
        None
    }

    /// Attempts to satisfy `layout` from the current chunk.
    fn bump(state: &mut ArenaState, layout: Layout) -> Option<NonNull<u8>> {
        if state.cursor.is_null() {
            return None;
        }

        let cursor_addr = state.cursor.addr();
        let end_addr = state.end.addr();

        // All arithmetic below is on addresses rather than pointers, and is
        // checked, so a pathological `layout` cannot wrap and produce a pointer
        // outside the chunk.
        let aligned = cursor_addr.checked_add(layout.align() - 1)? & !(layout.align() - 1);
        let new_cursor = aligned.checked_add(layout.size())?;
        if new_cursor > end_addr {
            return None;
        }

        let padding = aligned - cursor_addr;
        // SAFETY: `aligned + size <= end`, and `cursor`/`end` delimit a single
        // live chunk allocated by `grow`, so both offsets stay inside that
        // allocation and retain its provenance.
        let ptr = unsafe { state.cursor.add(padding) };
        // SAFETY: as above.
        state.cursor = unsafe { ptr.add(layout.size()) };
        state.bytes_used += layout.size() + padding;

        NonNull::new(ptr)
    }

    /// Reserves a new chunk large enough for `layout`. Returns `false` if the
    /// limit is reached or the system allocator refuses.
    fn grow(state: &mut ArenaState, layout: Layout) -> bool {
        let header = std::mem::size_of::<ChunkHeader>();
        // Worst case: the header ends just short of the requested alignment, so
        // budget a full alignment's worth of padding.
        let Some(needed) = header
            .checked_add(layout.align())
            .and_then(|n| n.checked_add(layout.size()))
        else {
            return false;
        };

        let size = state.next_chunk.max(needed).next_multiple_of(CHUNK_ALIGN);

        if state.bytes_reserved.saturating_add(size) > state.limit {
            return false;
        }

        let Ok(chunk_layout) = Layout::from_size_align(size, CHUNK_ALIGN) else {
            return false;
        };
        // SAFETY: `chunk_layout` has a non-zero size (it includes at least the
        // header) and a valid power-of-two alignment.
        let raw = unsafe { System.alloc(chunk_layout) };
        let Some(raw) = NonNull::new(raw) else {
            return false;
        };

        let header_ptr = raw.as_ptr().cast::<ChunkHeader>();
        // SAFETY: `raw` points at `size >= size_of::<ChunkHeader>()` bytes
        // aligned to `CHUNK_ALIGN >= align_of::<ChunkHeader>()`, and nothing
        // else refers to it yet.
        unsafe {
            header_ptr.write(ChunkHeader {
                next: state.chunks,
                size,
            });
        }

        state.chunks = header_ptr;
        // SAFETY: `header <= size`, so both offsets are within the block just
        // allocated and carry its provenance.
        unsafe {
            state.cursor = raw.as_ptr().add(header);
            state.end = raw.as_ptr().add(size);
        }
        state.bytes_reserved += size;
        state.chunk_count += 1;
        state.next_chunk = state.next_chunk.saturating_mul(2).min(MAX_CHUNK);

        true
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Arena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.stats();
        f.debug_struct("Arena")
            .field("bytes_reserved", &stats.bytes_reserved)
            .field("bytes_used", &stats.bytes_used)
            .field("chunks", &stats.chunks)
            .finish_non_exhaustive()
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // Every allocation the arena handed out borrows `&self`, so `&mut self`
        // here proves none is outstanding.
        //
        // SAFETY: exclusive access is guaranteed by `&mut self`.
        unsafe { self.reset() }
    }
}

/// A growable array backed by the arena.
///
/// The program-point table needs an indexable, appendable sequence of records,
/// and `Vec` is unusable for the same reason `HashMap` is: it allocates through
/// the global allocator, which is the thing being instrumented.
///
/// # Growth leaks, deliberately
///
/// The arena never frees, so doubling abandons the old block. Total waste is
/// bounded by the final size (a geometric series sums to less than twice it),
/// and everything is reclaimed at once by [`Arena::reset`]. Paying under 2x
/// memory to avoid a free list on the allocator hot path is the right trade;
/// [`ArenaVec::wasted_bytes`] reports the cost so it is not invisible.
pub struct ArenaVec<T: Copy> {
    /// `capacity` elements, or dangling when `capacity == 0`.
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    max_capacity: usize,
    wasted: usize,
}

// SAFETY: `ArenaVec` owns arena memory nothing else refers to, and has no
// interior mutability. Callers provide synchronization.
unsafe impl<T: Copy + Send> Send for ArenaVec<T> {}
// SAFETY: as above; `&ArenaVec` grants only reads.
unsafe impl<T: Copy + Sync> Sync for ArenaVec<T> {}

impl<T: Copy> ArenaVec<T> {
    /// Creates an empty vector that will grow to at most `max_capacity`
    /// elements. Reserves nothing until the first push, so this is `const`.
    pub const fn new(max_capacity: usize) -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            max_capacity,
            wasted: 0,
        }
    }

    /// Appends `value`, returning its index.
    ///
    /// Returns `None` when the ceiling is reached or the arena refuses. This
    /// runs inside the allocator, so refusal is reported, never panicked.
    pub fn push(&mut self, arena: &Arena, value: T) -> Option<usize> {
        if self.len == self.capacity && !self.grow(arena) {
            return None;
        }
        let index = self.len;
        // SAFETY: `index < capacity` because `grow` succeeded or there was
        // already room, so the offset is inside the current block.
        unsafe { self.ptr.as_ptr().add(index).write(value) };
        self.len += 1;
        Some(index)
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Elements this vector will grow to at most.
    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    /// Bytes abandoned by growth.
    ///
    /// Not what a profile reports, and deliberately so. This covers one
    /// `ArenaVec`, and the live-block table is built on [`RawMap`], which
    /// abandons blocks the same way and counts nothing. A profile carries the
    /// complete figure instead, as the difference between the arena's
    /// [`bytes_used`](ArenaStats::bytes_used) and the bytes its tables report
    /// holding: everything growth abandoned is still in the first and no longer
    /// in the second.
    ///
    /// That subtraction is only as good as the tables' own accounting, which is
    /// why [`PpTable::bytes`](super::pp::PpTable::bytes) counts its frame lists
    /// as well as its two containers.
    ///
    /// [`RawMap`]: super::table::RawMap
    pub fn wasted_bytes(&self) -> usize {
        self.wasted
    }

    /// Bytes currently in use.
    pub fn bytes(&self) -> usize {
        self.capacity * std::mem::size_of::<T>()
    }

    /// Borrows the element at `index`.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        // SAFETY: `index < len <= capacity`, and every element below `len` was
        // written by `push`.
        Some(unsafe { &*self.ptr.as_ptr().add(index) })
    }

    /// Mutably borrows the element at `index`.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        // SAFETY: as `get`; `&mut self` makes the borrow exclusive.
        Some(unsafe { &mut *self.ptr.as_ptr().add(index) })
    }

    /// Iterates over every element.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        (0..self.len).map(move |index| {
            // SAFETY: `index < len`; see `get`.
            unsafe { &*self.ptr.as_ptr().add(index) }
        })
    }

    #[cold]
    fn grow(&mut self, arena: &Arena) -> bool {
        let new_capacity = match self.capacity {
            0 => 64.min(self.max_capacity),
            current => current.saturating_mul(2).min(self.max_capacity),
        };
        if new_capacity <= self.capacity {
            return false;
        }

        let Ok(layout) = Layout::array::<T>(new_capacity) else {
            return false;
        };
        let Some(memory) = arena.alloc(layout) else {
            return false;
        };
        let new_ptr = memory.cast::<T>();

        if self.len > 0 {
            // SAFETY: the source holds `len` initialized elements, the
            // destination has room for `new_capacity >= len`, and the two
            // blocks cannot overlap because the destination was just obtained
            // from the arena.
            unsafe { std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len) };
            self.wasted += self.capacity * std::mem::size_of::<T>();
        }

        self.ptr = new_ptr;
        self.capacity = new_capacity;
        true
    }
}

impl<T: Copy> fmt::Debug for ArenaVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArenaVec")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("max_capacity", &self.max_capacity)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internals::miri_scale;

    #[test]
    fn allocations_are_aligned_and_distinct() {
        let arena = Arena::new();
        let mut seen: Vec<(usize, usize)> = Vec::new();

        for align_shift in 0..5 {
            let align = 1usize << align_shift;
            for size in [1usize, 3, 8, 100, 1000] {
                let layout = Layout::from_size_align(size, align).unwrap();
                let ptr = arena.alloc(layout).expect("arena should satisfy this");
                assert!(
                    ptr.as_ptr().addr().is_multiple_of(align),
                    "alignment {align} violated for size {size}"
                );
                let range = (ptr.as_ptr().addr(), ptr.as_ptr().addr() + size);
                for &(start, end) in &seen {
                    assert!(
                        range.1 <= start || range.0 >= end,
                        "arena handed out overlapping blocks: {range:?} overlaps ({start}, {end})"
                    );
                }
                seen.push(range);
            }
        }
    }

    #[test]
    fn written_bytes_survive_later_allocations() {
        let arena = Arena::new();
        let mut blocks = Vec::new();
        // Enough to force several chunk refills.
        for i in 0..miri_scale(2000) as u32 {
            let slice = arena.alloc_slice(&[i; 32]).expect("allocation failed");
            blocks.push((i, slice as *mut [u32]));
        }
        for (i, block) in blocks {
            // SAFETY: nothing has reset the arena, so every block is still live
            // and uniquely owned by this test.
            let block = unsafe { &*block };
            assert!(block.iter().all(|&v| v == i), "arena block was corrupted");
        }
    }

    #[test]
    fn grows_across_chunks() {
        // The *product* is what has to exceed a chunk, so the block size rises
        // as the count falls. Scaling only the count would leave this
        // allocating 12 KiB into a 64 KiB chunk and asserting nothing.
        const BLOCK: usize = 1024;
        let count = miri_scale(10_000);
        assert!(
            count * BLOCK > FIRST_CHUNK,
            "the test no longer allocates past a chunk boundary"
        );
        let arena = Arena::new();
        for _ in 0..count {
            arena
                .alloc(Layout::from_size_align(BLOCK, 8).unwrap())
                .unwrap();
        }
        let stats = arena.stats();
        assert!(stats.chunks > 1, "expected multiple chunks, got {stats:?}");
        assert!(stats.bytes_used >= count * BLOCK);
        assert!(stats.bytes_reserved >= stats.bytes_used);
    }

    #[test]
    fn oversized_request_gets_its_own_chunk() {
        let arena = Arena::new();
        let big = Layout::from_size_align(3 * MAX_CHUNK, 64).unwrap();
        let ptr = arena.alloc(big).expect("oversized request should succeed");
        assert!(ptr.as_ptr().addr().is_multiple_of(64));
        assert!(arena.stats().bytes_reserved >= 3 * MAX_CHUNK);
    }

    #[test]
    fn refuses_rather_than_aborting_when_the_limit_is_reached() {
        // The limit is one chunk, and the count is floored high enough that the
        // requests exceed it. Deriving the limit from the count alone put the
        // ceiling *below* a single chunk under Miri, so the arena refused the
        // very first request and the test asserted nothing.
        const BLOCK: usize = 256;
        let limit = FIRST_CHUNK;
        let count = miri_scale(10_000).max(2 * limit / BLOCK);
        let arena = Arena::new();
        arena.set_limit(limit);

        let mut granted = 0;
        for _ in 0..count {
            if arena
                .alloc(Layout::from_size_align(BLOCK, 8).unwrap())
                .is_some()
            {
                granted += 1;
            }
        }

        let stats = arena.stats();
        assert!(granted > 0, "limit was too low to grant anything");
        assert!(
            stats.bytes_reserved <= limit,
            "arena exceeded its limit: {stats:?}"
        );
        assert!(stats.refused > 0, "expected refusals to be counted");
    }

    #[test]
    fn zero_sized_requests_are_satisfied_without_reserving() {
        let arena = Arena::new();
        for _ in 0..miri_scale(1000) {
            assert!(arena
                .alloc(Layout::from_size_align(0, 8).unwrap())
                .is_some());
        }
        assert_eq!(arena.stats().bytes_reserved, 0);
    }

    #[test]
    fn concurrent_allocation_does_not_hand_out_overlaps() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[cfg(miri)]
        const PER_THREAD: usize = 50;
        #[cfg(not(miri))]
        const PER_THREAD: usize = 5_000;
        const THREADS: usize = 8;

        let arena = Arena::new();
        let failures = AtomicUsize::new(0);

        std::thread::scope(|s| {
            for t in 0..THREADS {
                let arena = &arena;
                let failures = &failures;
                s.spawn(move || {
                    let tag = t as u8;
                    let mut mine = Vec::with_capacity(PER_THREAD);
                    for _ in 0..PER_THREAD {
                        // Each thread writes its own tag over its own block. If
                        // two threads were ever handed overlapping memory, one
                        // would find the other's tag on the readback below.
                        let block = arena.alloc_slice(&[tag; 24]).expect("allocation failed");
                        mine.push(block as *mut [u8]);
                    }
                    for block in mine {
                        // SAFETY: this thread allocated the block and no reset
                        // has occurred, so it is live and exclusively ours.
                        let block = unsafe { &*block };
                        if !block.iter().all(|&v| v == tag) {
                            failures.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
        });

        assert_eq!(
            failures.load(Ordering::Relaxed),
            0,
            "concurrent allocations overlapped"
        );
        assert_eq!(arena.stats().bytes_used, THREADS * PER_THREAD * 24);
    }

    #[test]
    fn arena_vec_appends_and_indexes() {
        let arena = Arena::new();
        let mut v: ArenaVec<u64> = ArenaVec::new(1 << 20);
        let count = miri_scale(10_000);
        for i in 0..count as u64 {
            assert_eq!(v.push(&arena, i * 3), Some(i as usize));
        }
        assert_eq!(v.len(), count);
        for i in 0..count {
            assert_eq!(v.get(i), Some(&(i as u64 * 3)), "element {i} was lost");
        }
        assert_eq!(v.get(count), None);
    }

    #[test]
    fn arena_vec_survives_growth() {
        let arena = Arena::new();
        let mut v: ArenaVec<[u64; 4]> = ArenaVec::new(1 << 16);
        for i in 0..miri_scale(5_000) as u64 {
            v.push(&arena, [i; 4]).unwrap();
        }
        for (i, element) in v.iter().enumerate() {
            assert_eq!(*element, [i as u64; 4], "element {i} corrupted by a resize");
        }
        assert!(v.wasted_bytes() > 0, "growth should have abandoned blocks");
        // The geometric series bounds total waste below the live size.
        assert!(
            v.wasted_bytes() < v.bytes(),
            "growth wasted more than it kept: {} vs {}",
            v.wasted_bytes(),
            v.bytes()
        );
    }

    #[test]
    fn arena_vec_refuses_past_its_ceiling() {
        let arena = Arena::new();
        let mut v: ArenaVec<u32> = ArenaVec::new(100);
        let mut pushed = 0;
        for i in 0..miri_scale(1_000) as u32 {
            if v.push(&arena, i).is_some() {
                pushed += 1;
            }
        }
        assert_eq!(pushed, 100, "the ceiling was not respected");
        assert_eq!(v.len(), 100);
        assert_eq!(v.push(&arena, 0), None);
    }

    #[test]
    fn arena_vec_mutation_is_visible() {
        let arena = Arena::new();
        let mut v: ArenaVec<u64> = ArenaVec::new(1024);
        for i in 0..100u64 {
            v.push(&arena, i).unwrap();
        }
        *v.get_mut(50).unwrap() = 999;
        assert_eq!(v.get(50), Some(&999));
        assert_eq!(v.get(49), Some(&49));
    }
}
