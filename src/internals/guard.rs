//! The reentrancy guard: "am I already inside the allocator shim on this thread?"
//!
//! # The problem this has to survive
//!
//! The obvious implementation is a thread-local flag, which is what `dhat-rs`
//! uses. Consulted from inside the shim *before* the thread is known to be
//! guarded, a thread-local whose initialization allocates recurses without
//! bound: the access initializes, the initialization allocates, the allocation
//! re-enters the shim, which touches the same thread-local, which is still
//! mid-initialization. Each level leaks a block and the stack runs out.
//! `try_with` cannot help — the slot is not "unavailable", it is
//! *mid-initialization*, and the accessor happily starts initializing it again.
//!
//! **Reproduced**, by `tests/cdylib_tls.rs` against a build of this module with
//! one allocating thread-local put in front of the slot lookup:
//!
//! ```text
//! thread '...' has overflowed its stack
//! fatal runtime error: stack overflow, aborting
//! ```
//!
//! # One correction to PLAN.md section 4.7
//!
//! The plan attributes this to dyld specifically: that in a `cdylib` on Apple
//! platforms, `tlv_get_addr` `malloc`s the TLV block before recording it, and
//! that this alone is enough to recurse. **That part does not reproduce.** The
//! same test, against a build whose guard consults a `const`-initialized,
//! destructor-free thread-local — the shape `dhat-rs` actually uses — passes: no
//! recursion, no leak. The reason is that dyld's allocation goes to the C
//! `malloc` in libsystem, and a Rust `#[global_allocator]` does not sit in front
//! of that. It intercepts Rust allocations, not every `malloc` in the process.
//!
//! So the hazard is real and the platform detail behind it was wrong. What
//! matters is *what the thread-local access does*, not which loader resolves it:
//! an access that can allocate, reached before the guard is established, is the
//! bug. Which is why the one thread-local this module does use is touched only
//! from [`claim_slot`], **after** the slot is owned, where a recursive entry
//! finds an owned slot at depth zero and terminates one level down.
//!
//! # What this does instead
//!
//! **The guard does not use thread-local storage at all.**
//!
//! A fixed, statically allocated table holds one slot per thread, keyed by the
//! platform's own thread handle — `pthread_self` on unix, `GetCurrentThreadId`
//! on Windows. Both are register or TEB reads: no allocation, no lazy
//! initialization, no dyld involvement, valid inside a signal handler, and valid
//! during thread teardown after every thread-local has been destroyed. The
//! recursion depth lives in the slot, so entering the guard is a hash, a load,
//! and a compare.
//!
//! Thread-local storage appears exactly once, and only to learn when a thread
//! has exited so its slot can be reused. That use is *not* load-bearing: it is
//! touched while the guard already reads as entered, so if it recurses, the
//! recursive entry is correctly refused and the recursion terminates. If the
//! destructor never runs, one slot leaks and the table is one smaller.
//!
//! # Failure mode
//!
//! When the table is full, [`enter`] refuses. Refusing loses an event; the
//! alternative is recursing inside an allocator. The count of refusals is
//! reported in the profile's self-metrics, so a program that creates more live
//! threads than the table holds can see that it did.

use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicUsize, Ordering};

use super::sampler;
use super::site::{RegionId, Site, ThreadId};

/// Number of thread slots.
///
/// Slots are reclaimed when a thread exits, so this bounds *concurrently live*
/// threads that have allocated, not threads over the process lifetime. At 32
/// bytes per slot the table is 128 KiB of BSS on a 64-bit target, which is not
/// worth a knob.
const SLOTS: usize = 4096;
const SLOT_MASK: usize = SLOTS - 1;

const _: () = assert!(SLOTS.is_power_of_two());

/// Probe distance before declaring the table full.
///
/// With open addressing at low load factor, a free slot is almost always within
/// a few probes. A bounded scan keeps the worst case off the hot path.
const MAX_PROBE: usize = 32;

/// An unclaimed slot. No real thread handle is zero on any supported platform,
/// and [`thread_handle`] maps a hypothetical zero away regardless.
const EMPTY: usize = 0;

struct Slot {
    /// Thread handle owning this slot, or [`EMPTY`].
    owner: AtomicUsize,
    /// Cached stack bounds for the owning thread, as `base` and `span` — the
    /// form the frame-pointer walker checks against directly.
    ///
    /// Cached because asking the platform is not free: on the glibc main thread
    /// `pthread_getattr_np` reads `/proc/self/maps` and **allocates**. It can
    /// therefore only be called with the guard already held, and only once per
    /// thread. `span == 0` means "not yet computed".
    stack_base: AtomicUsize,
    stack_span: AtomicUsize,
    /// Reentrancy depth for the owning thread.
    ///
    /// Only ever written by the owning thread (including from a signal handler
    /// interrupting it, which is precisely the case the depth exists to catch),
    /// so `Relaxed` suffices: a single thread always observes its own writes to
    /// one location in program order.
    depth: AtomicU32,
    /// The row this thread claimed in the profile's thread table, or
    /// [`ThreadId::UNCLAIMED`].
    ///
    /// Cached here rather than in thread-local storage for the reason the whole
    /// module exists: a thread-local read on the allocator path can allocate on
    /// first touch. This word is in a line the thread has just written anyway —
    /// it shares a cache line with `depth`, which `enter` sets on the way in.
    thread: AtomicU16,
    /// The innermost region open on this thread, or [`RegionId::NONE`].
    ///
    /// Written by [`enter_region`] and [`leave_region`] rather than by the
    /// guard, because a region outlives any one trip through the shim. A
    /// two-byte word rather than a stack of them: the nesting lives on the
    /// program's own stack, in each [`Region`](crate::Region) guard's saved
    /// predecessor, so any depth of nesting costs the profiler nothing.
    region: AtomicU16,
    /// How many bytes until this thread's next sample point, and the generator
    /// that draws the next gap.
    ///
    /// Here for the reason everything else in this slot is here: the sampler
    /// needs per-thread state on the allocator path, and a `thread_local!` can
    /// allocate on first touch. See [`sampler`](super::sampler).
    ///
    /// Untouched when sampling is off, which is the default: the engine checks
    /// its interval before asking.
    sampler: sampler::State,
}

impl Slot {
    const fn new() -> Self {
        Self {
            owner: AtomicUsize::new(EMPTY),
            stack_base: AtomicUsize::new(0),
            stack_span: AtomicUsize::new(0),
            depth: AtomicU32::new(0),
            thread: AtomicU16::new(ThreadId::UNCLAIMED.as_u16()),
            region: AtomicU16::new(RegionId::NONE.as_u16()),
            sampler: sampler::State::new(),
        }
    }

    /// Returns the slot to the state a thread that has never used it finds.
    ///
    /// The attribution words are part of the *contents*, not the ownership: a
    /// slot handed to a new thread that kept the last one's row would attribute
    /// its allocations to a thread that has already exited.
    fn clear(&self) {
        self.depth.store(0, Ordering::Relaxed);
        self.stack_span.store(0, Ordering::Relaxed);
        self.stack_base.store(0, Ordering::Relaxed);
        self.thread
            .store(ThreadId::UNCLAIMED.as_u16(), Ordering::Relaxed);
        self.region
            .store(RegionId::NONE.as_u16(), Ordering::Relaxed);
        self.sampler.clear();
    }
}

/// The attribution words are meant to be free, and "free" is a size, so it is
/// asserted rather than asserted-to. A `usize`, two more, a `u32` and two `u16`
/// is 32 bytes with no padding left over, and the sampler's two `u64`s take it
/// to 48; if a future field pushes past that, the table grows again and this
/// stops compiling first.
///
/// At 4,096 slots the table is 192 KiB of BSS, up from 128 KiB when the sampler
/// arrived. That is still not worth a knob, and it is BSS rather than resident
/// memory until a thread touches its slot.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<Slot>() == 48);

static TABLE: [Slot; SLOTS] = {
    // `Slot` is not `Copy`, so the array initializer has to be spelled out this
    // way. It is a compile-time constant either way.
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Slot = Slot::new();
    [INIT; SLOTS]
};

/// Threads that were refused a slot because the table was full.
static REFUSED: AtomicUsize = AtomicUsize::new(0);
/// Slots currently claimed. Reported in self-metrics.
static CLAIMED: AtomicUsize = AtomicUsize::new(0);

/// Returns a non-zero, allocation-free, TLS-free identifier for the calling
/// thread.
///
/// Shared with [`super::order`], which needs per-thread state on the same paths
/// and for the same reason: thread-local storage is not reachable from inside
/// the allocator on every platform.
#[inline(always)]
pub(crate) fn thread_handle() -> usize {
    // `pthread_t` is not one type. Darwin defines it as
    // `struct _opaque_pthread_t *`, a genuine pointer; glibc defines it as
    // `unsigned long int`. Declaring one as the other is an ABI mismatch, so
    // each is spelled out. (Miri rejects exactly this class of mismatch, but
    // only for the paths it executes — it runs the Darwin backend here, so the
    // glibc declaration below is correct by inspection rather than by test.)
    #[cfg(target_vendor = "apple")]
    let raw = {
        extern "C" {
            fn pthread_self() -> *mut std::ffi::c_void;
        }
        // SAFETY: `pthread_self` takes no arguments, cannot fail, and is
        // async-signal-safe. It reads the thread's own control block, which
        // exists for the entire lifetime of any thread that can execute code.
        unsafe { pthread_self() }.addr()
    };

    #[cfg(all(unix, not(target_vendor = "apple")))]
    let raw = {
        extern "C" {
            fn pthread_self() -> std::ffi::c_ulong;
        }
        // SAFETY: as above. The value is an opaque integer handle here, not a
        // pointer, so no provenance is involved.
        unsafe { pthread_self() as usize }
    };

    #[cfg(windows)]
    let raw = {
        #[link(name = "kernel32", kind = "raw-dylib")]
        extern "system" {
            fn GetCurrentThreadId() -> u32;
        }
        // SAFETY: reads a field of the Thread Environment Block; no arguments,
        // no failure mode.
        unsafe { GetCurrentThreadId() as usize }
    };

    // A zero handle would collide with `EMPTY`. No supported platform produces
    // one, but the guard's soundness should not rest on that.
    if raw == EMPTY {
        1
    } else {
        raw
    }
}

/// Fibonacci hashing. Thread handles are pointers on unix, so the low bits are
/// alignment zeros and must not be used directly as an index.
#[inline(always)]
fn slot_index(handle: usize) -> usize {
    // Fibonacci hashing needs a constant that fits the word, and a shift that
    // keeps the *high* half of the product. The 64-bit constant is not merely
    // wrong on a 32-bit target, it does not compile there, and a fixed shift of
    // 32 would discard the whole product.
    #[cfg(target_pointer_width = "64")]
    const GOLDEN: usize = 0x9E37_79B9_7F4A_7C15;
    #[cfg(not(target_pointer_width = "64"))]
    const GOLDEN: usize = 0x9E37_79B9;
    (handle.wrapping_mul(GOLDEN) >> (usize::BITS / 2)) & SLOT_MASK
}

/// Proof that the calling thread holds the guard. Releases it when dropped.
#[must_use = "the guard is released immediately if it is not bound"]
pub struct Guard {
    slot: usize,
    /// Not `Send`: the depth belongs to the thread that incremented it.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Guard {
    /// The calling thread's stack bounds, as an inclusive range.
    ///
    /// Computed once per thread and cached in the guard slot. Requires the
    /// guard — which the caller holds by virtue of having this value — because
    /// the underlying query allocates on the glibc main thread, and an
    /// allocation from inside the shim without the guard held is the recursion
    /// this whole module exists to prevent.
    ///
    /// Returns `None` if the platform declined to say, in which case the walker
    /// falls back to its weaker structural checks.
    pub fn stack_bounds(&self) -> Option<std::ops::Range<usize>> {
        let slot = &TABLE[self.slot];
        let span = slot.stack_span.load(Ordering::Relaxed);
        if span != 0 {
            let base = slot.stack_base.load(Ordering::Relaxed);
            return Some(base..base + span);
        }

        // Cold path, once per thread. Safe to allocate here: the depth is
        // already raised, so any allocation this makes re-enters and is refused.
        let bounds = crate::internals::stack::current_bounds()?;
        let span = bounds.end.checked_sub(bounds.start)?;
        if span == 0 {
            return None;
        }
        slot.stack_base.store(bounds.start, Ordering::Relaxed);
        slot.stack_span.store(span, Ordering::Relaxed);
        Some(bounds)
    }

    /// Who is allocating, and what for.
    ///
    /// Two relaxed loads from a line this thread wrote on the way in. The
    /// thread row may be [`ThreadId::UNCLAIMED`], which the engine resolves
    /// once per thread; the region is whatever [`enter_region`] last left here.
    #[inline]
    pub fn site(&self) -> Site {
        let slot = &TABLE[self.slot];
        Site {
            thread: ThreadId::from_u16(slot.thread.load(Ordering::Relaxed)),
            region: RegionId::from_u16(slot.region.load(Ordering::Relaxed)),
        }
    }

    /// Caches the row this thread claimed, so it claims exactly once.
    ///
    /// Takes the guard by reference rather than being a free function because
    /// the row is only ever claimed from inside the shim, where the answer is
    /// wanted: a thread with no recorded events needs no row.
    #[inline]
    pub fn set_thread(&self, id: ThreadId) {
        TABLE[self.slot]
            .thread
            .store(id.as_u16(), Ordering::Relaxed);
    }

    /// Whether this thread's sampler admits an allocation of `size` bytes,
    /// advancing its countdown either way.
    ///
    /// Only reached when the run has a sampling interval; a run without one
    /// never touches these words. See [`sampler`](super::sampler) for what is
    /// being decided.
    #[inline]
    pub fn admits(&self, size: usize, interval: u64) -> bool {
        TABLE[self.slot].sampler.admits(size, interval)
    }
}

impl Drop for Guard {
    #[inline]
    fn drop(&mut self) {
        let slot = &TABLE[self.slot];
        let depth = slot.depth.load(Ordering::Relaxed);
        // Not a `debug_assert!`. This is the allocator path, where a panic
        // allocates its own message and re-enters the shim. A depth of zero
        // here would mean the guard was released more times than acquired,
        // which is an internal invariant violation, so it poisons instead.
        if depth == 0 {
            crate::internals::diagnostic::poison(
                "reentrancy guard released more times than acquired",
            );
            return;
        }
        slot.depth.store(depth - 1, Ordering::Relaxed);
    }
}

impl fmt::Debug for Guard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Guard").finish_non_exhaustive()
    }
}

/// Attempts to enter the profiler's critical section on this thread.
///
/// Returns `None` if this thread is already inside — meaning the call came from
/// the profiler's own machinery, from a signal handler that interrupted it, or
/// from the platform's thread-local bootstrap — or if no slot is available.
/// In every one of those cases the correct action is to do nothing but forward
/// to the inner allocator.
#[inline]
pub fn enter() -> Option<Guard> {
    let handle = thread_handle();
    let slot = find_slot(handle)?;

    let depth = TABLE[slot].depth.load(Ordering::Relaxed);
    if depth != 0 {
        return None;
    }
    TABLE[slot].depth.store(1, Ordering::Relaxed);

    Some(Guard {
        slot,
        _not_send: std::marker::PhantomData,
    })
}

/// Reports whether this thread is currently inside the guard, without taking it.
///
/// For assertions and tests; not part of the hot path.
pub fn is_entered() -> bool {
    let handle = thread_handle();
    let start = slot_index(handle);
    for probe in 0..MAX_PROBE {
        let slot = &TABLE[(start + probe) & SLOT_MASK];
        match slot.owner.load(Ordering::Acquire) {
            EMPTY => return false,
            owner if owner == handle => return slot.depth.load(Ordering::Relaxed) != 0,
            _ => {}
        }
    }
    false
}

/// Locates this thread's slot, claiming one on first use.
#[inline]
fn find_slot(handle: usize) -> Option<usize> {
    let start = slot_index(handle);

    // Fast path: the slot this thread claimed earlier, usually the first probe.
    for probe in 0..MAX_PROBE {
        let index = (start + probe) & SLOT_MASK;
        match TABLE[index].owner.load(Ordering::Acquire) {
            owner if owner == handle => return Some(index),
            // An empty slot means this thread has not claimed one yet: open
            // addressing guarantees the thread's slot, if any, precedes the
            // first gap in the probe sequence.
            EMPTY => return claim_slot(handle, index),
            _ => {}
        }
    }

    REFUSED.fetch_add(1, Ordering::Relaxed);
    None
}

/// Claims a slot for `handle`, starting at `index`.
#[cold]
fn claim_slot(handle: usize, index: usize) -> Option<usize> {
    let mut index = index;
    for _ in 0..MAX_PROBE {
        match TABLE[index].owner.compare_exchange(
            EMPTY,
            handle,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                TABLE[index].clear();
                CLAIMED.fetch_add(1, Ordering::Relaxed);
                register_release(index);
                return Some(index);
            }
            // Lost the race for this slot; if the winner was this same thread
            // (impossible today, but cheap to handle) use it, otherwise probe on.
            Err(owner) if owner == handle => return Some(index),
            Err(_) => index = (index + 1) & SLOT_MASK,
        }
    }

    REFUSED.fetch_add(1, Ordering::Relaxed);
    None
}

/// Releases `slot` back to the table.
fn release_slot(index: usize) {
    // Everything cached here belongs to the departing thread. Leaving the stack
    // bounds would hand the next claimant a range that is not its own, and the
    // walker would reject every frame it produced -- or, worse, accept an
    // address in another thread's stack. Leaving the thread row would attribute
    // the next claimant's allocations to a thread that has already exited.
    TABLE[index].clear();
    TABLE[index].owner.store(EMPTY, Ordering::Release);
    CLAIMED.fetch_sub(1, Ordering::Relaxed);
}

/// A thread-local whose only job is to run at thread exit and hand the slot back.
struct SlotRelease(Cell<usize>);

impl Drop for SlotRelease {
    fn drop(&mut self) {
        release_slot(self.0.get());
    }
}

thread_local! {
    /// Reclaims this thread's slot when the thread exits.
    ///
    /// This is the crate's only use of thread-local storage on the allocator
    /// path, and it is deliberately not load-bearing. It is touched from
    /// [`claim_slot`], which runs *before* the depth is raised, so on a platform
    /// where the first touch allocates (a macOS cdylib, via `__tlv_bootstrap`),
    /// the recursive entry finds the slot already owned with depth zero and
    /// simply takes the guard — the outer entry then sees a non-zero depth and
    /// refuses. Either way the recursion is one level deep and terminates.
    ///
    /// If the destructor never runs — `_exit`, a fatal signal, a platform that
    /// skips destructors — one slot leaks. That costs 16 bytes and one fewer
    /// concurrent thread, which is an acceptable price for never recursing.
    static RELEASE: Cell<Option<SlotRelease>> = const { Cell::new(None) };
}

fn register_release(index: usize) {
    // `try_with` because a thread may allocate after its own thread-locals have
    // been destroyed, at which point registration is no longer possible and the
    // slot simply leaks.
    let _ = RELEASE.try_with(|cell| {
        if cell.replace(Some(SlotRelease(Cell::new(index)))).is_some() {
            // The thread previously registered a different slot, which means it
            // allocated after teardown and claimed a second one. Nothing to do:
            // the replaced value's `Drop` runs here and releases the old slot.
        }
    });
}

/// Releases the slots of threads that did not survive a `fork`.
///
/// A `fork` child has exactly one thread, but it inherits the whole table. Every
/// slot belonging to a thread that no longer exists is stale, and a stale slot
/// is worse than a leaked one: `pthread_t` values are reused, so a thread the
/// child creates later can hash to a slot whose depth was left non-zero by a
/// thread that died mid-shim. That thread would then be refused the guard
/// forever, and every allocation it made would go unrecorded.
///
/// # Why the ownership is left in place
///
/// The first version of this released the dead slots — set `owner` back to
/// [`EMPTY`] — which is the obvious thing and is wrong, because this is an
/// open-addressed table. Releasing a slot punches a **gap** into a probe
/// sequence, and [`find_slot`] stops at the first gap. A surviving thread whose
/// slot sits after one of those gaps is then handed a *second* slot on its next
/// call, and [`register_release`] replaces its thread-local, whose `Drop`
/// releases the **old** slot — zeroing the depth of the slot a live [`Guard`]
/// still refers to. Demonstrated, not theorised: in that window a nested
/// [`enter`] succeeds on a thread that is already inside the shim (a same-thread
/// lock reacquisition, which is a `SIGKILL` on Apple platforms), and the outer
/// guard's `Drop` poisons the engine and stops recording for the rest of the
/// process.
///
/// So the dead slots keep their owners and lose only their *contents*. That
/// removes the hazard this function exists for — a `pthread_t` value is reused,
/// and a new thread hashing to a slot left at a non-zero depth by a thread that
/// died inside the shim would be refused the guard forever — while leaving
/// every probe sequence exactly as long as it was. The cost is that a child
/// which goes on to create many threads has fewer free slots than a fresh
/// process would; the table is 4096 entries and the overwhelmingly common child
/// calls `exec` immediately.
///
/// The calling thread's slot is untouched for the same reason it always was: it
/// may be holding a live [`Guard`], since a `fork` from inside the shim is
/// unusual but not impossible.
///
/// # Safety
///
/// Call only from a `pthread_atfork` child handler, where the process is
/// single-threaded, so no other thread can be reading a slot as it is cleared.
pub unsafe fn reinit_after_fork() {
    let survivor = thread_handle();
    for slot in &TABLE {
        let owner = slot.owner.load(Ordering::Relaxed);
        if owner == EMPTY || owner == survivor {
            continue;
        }
        // Contents, not ownership. See above.
        slot.clear();
    }
}

/// Makes `id` the innermost region on the calling thread, returning the region
/// it displaced.
///
/// # Why this takes a [`Guard`]
///
/// Not to use it — to require it. A slot is what this writes to, and holding a
/// guard *is* the proof that this thread has one: [`enter`] looks the slot up
/// and returns `None` when it cannot be had, so a caller holding a `Guard`
/// cannot be a caller without a slot. Written first as a fallible call, it had
/// an error branch nothing could reach, three paragraphs of documentation
/// resting on that branch, and a mutation replacing it with `.expect()` that
/// survived the whole suite.
///
/// What it is *not* is a method on [`Guard`]: a region spans arbitrary program
/// code, including many trips in and out of the shim, so it cannot be tied to
/// one guard's lifetime. What it stores is slot state, which outlives every
/// guard the thread takes.
pub fn enter_region(guard: &Guard, id: RegionId) -> RegionId {
    let previous = TABLE[guard.slot]
        .region
        .swap(id.as_u16(), Ordering::Relaxed);
    RegionId::from_u16(previous)
}

/// Restores `previous` as the innermost region on the calling thread.
///
/// Cannot fail, cannot block, and cannot allocate: one relaxed store to a slot
/// this thread already owns. That is what makes it safe to run from a `Drop`
/// reached inside the shim, where taking a lock would deadlock and a
/// thread-local touch could recurse.
///
/// It looks the slot up **without claiming one**. The lookup used by [`enter`]
/// claims on a miss, which reaches a thread-local, and a region guard dropped
/// after its thread's slot has been released — a `Region` outliving the
/// thread-local that reclaims slots, whose destructor order is not specified —
/// would then claim a second slot from inside a destructor. A miss here means
/// the thread has no attribution left to restore, so doing nothing is the whole
/// of the correct behaviour.
pub fn leave_region(previous: RegionId) {
    if let Some(slot) = existing_slot(thread_handle()) {
        TABLE[slot]
            .region
            .store(previous.as_u16(), Ordering::Relaxed);
    }
}

/// Locates this thread's slot without claiming one.
///
/// [`find_slot`]'s lookup half. Split out for [`leave_region`], which must not
/// claim; see there for why.
fn existing_slot(handle: usize) -> Option<usize> {
    let start = slot_index(handle);
    for probe in 0..MAX_PROBE {
        let index = (start + probe) & SLOT_MASK;
        match TABLE[index].owner.load(Ordering::Acquire) {
            owner if owner == handle => return Some(index),
            // Open addressing: this thread's slot, if it has one, precedes the
            // first gap in the probe sequence.
            EMPTY => return None,
            _ => {}
        }
    }
    None
}

/// Enters the guard for a section that releases it from a different function.
///
/// Returns whether this call entered. `false` means the thread was already
/// inside, in which case the caller must **not** call [`leave_unbalanced`].
///
/// This exists for the `fork` handlers, which enter in `prepare` and leave in
/// `parent` or `child`, so a [`Guard`]'s scope cannot express the lifetime. It
/// is deliberately not a general-purpose tool: everything else in this crate
/// binds a [`Guard`] and lets `Drop` do the work.
pub fn enter_unbalanced() -> bool {
    // The `Guard` would release on drop, and this is exactly the case where
    // that is wrong. Its slot is recomputed by `leave_unbalanced` from the same
    // thread handle, so nothing is lost by discarding the value.
    match enter() {
        Some(guard) => {
            std::mem::forget(guard);
            true
        }
        None => false,
    }
}

/// Leaves a section entered by [`enter_unbalanced`].
///
/// Call only on the thread that entered, and only if [`enter_unbalanced`]
/// returned `true`.
pub fn leave_unbalanced() {
    let handle = thread_handle();
    let start = slot_index(handle);
    for probe in 0..MAX_PROBE {
        let slot = &TABLE[(start + probe) & SLOT_MASK];
        match slot.owner.load(Ordering::Acquire) {
            EMPTY => break,
            owner if owner == handle => {
                let depth = slot.depth.load(Ordering::Relaxed);
                if depth == 0 {
                    break;
                }
                slot.depth.store(depth - 1, Ordering::Relaxed);
                return;
            }
            _ => {}
        }
    }
    crate::internals::diagnostic::poison("reentrancy guard left without being entered");
}

/// Counters for the profile's self-metrics block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardStats {
    /// Slots currently claimed by live threads.
    pub claimed: usize,
    /// Events dropped because no slot was available.
    pub refused: usize,
    /// Total slots in the table.
    pub capacity: usize,
}

/// Reports guard-table occupancy.
pub fn stats() -> GuardStats {
    GuardStats {
        claimed: CLAIMED.load(Ordering::Relaxed),
        refused: REFUSED.load(Ordering::Relaxed),
        capacity: SLOTS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entering_twice_on_one_thread_is_refused() {
        let outer = enter().expect("first entry should succeed");
        assert!(is_entered());
        assert!(
            enter().is_none(),
            "reentrant entry was permitted; this is the recursion the guard exists to stop"
        );
        drop(outer);
        assert!(!is_entered());
    }

    #[test]
    fn entry_is_available_again_after_release() {
        for _ in 0..1000 {
            let guard = enter().expect("entry should be available");
            drop(guard);
        }
        assert!(!is_entered());
    }

    #[test]
    fn threads_do_not_block_each_other() {
        #[cfg(miri)]
        const THREADS: usize = 4;
        #[cfg(not(miri))]
        const THREADS: usize = 16;

        let barrier = std::sync::Barrier::new(THREADS);
        std::thread::scope(|s| {
            for _ in 0..THREADS {
                let barrier = &barrier;
                s.spawn(move || {
                    let guard = enter().expect("each thread has its own slot");
                    // Every thread holds the guard simultaneously. A design that
                    // used one global flag would deadlock or mis-refuse here.
                    barrier.wait();
                    assert!(is_entered());
                    drop(guard);
                });
            }
        });
    }

    #[test]
    fn slots_are_reclaimed_when_threads_exit() {
        #[cfg(miri)]
        const ROUNDS: usize = 8;
        #[cfg(not(miri))]
        const ROUNDS: usize = 200;

        let before = stats();
        for _ in 0..ROUNDS {
            std::thread::spawn(|| {
                let _guard = enter().expect("fresh thread should get a slot");
            })
            .join()
            .unwrap();
        }
        let after = stats();

        // Without reclamation this would have consumed `ROUNDS` slots. Allow a
        // small drift for threads the test harness itself may have started.
        assert!(
            after.claimed <= before.claimed + 8,
            "slots leaked: {before:?} -> {after:?}"
        );
        assert_eq!(after.refused, before.refused, "no refusal was expected");
    }

    /// The property the whole design exists for: a recursive call that arrives
    /// while the thread is inside the guard must be refused, whatever the source.
    #[test]
    fn nested_entry_from_within_a_guarded_section_is_refused() {
        fn simulated_reentrancy(depth: usize) -> usize {
            match enter() {
                None => depth,
                Some(_guard) => simulated_reentrancy(depth + 1),
            }
        }
        assert_eq!(
            simulated_reentrancy(0),
            1,
            "recursion should terminate after exactly one successful entry"
        );
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "Miri reports synthetic stack bounds unrelated to its locals"
    )]
    fn cached_stack_bounds_contain_a_local() {
        let guard = enter().unwrap();
        let local = 0u64;
        let address = std::ptr::from_ref(&local).addr();
        let bounds = guard.stack_bounds().expect("bounds should be available");
        assert!(bounds.contains(&address));

        // The second call must come from the cache and agree.
        assert_eq!(guard.stack_bounds(), Some(bounds));
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "Miri reports synthetic stack bounds unrelated to its locals"
    )]
    fn each_thread_caches_its_own_stack() {
        let main_bounds = {
            let guard = enter().unwrap();
            guard.stack_bounds().unwrap()
        };

        let child_bounds = std::thread::spawn(|| {
            let guard = enter().unwrap();
            let local = 0u64;
            let address = std::ptr::from_ref(&local).addr();
            let bounds = guard.stack_bounds().unwrap();
            assert!(
                bounds.contains(&address),
                "a thread was handed another thread's cached stack bounds"
            );
            bounds
        })
        .join()
        .unwrap();

        assert_ne!(child_bounds, main_bounds);
    }
}
