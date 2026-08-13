//! The `GlobalAlloc` shim.
//!
//! Everything the profiler observes passes through here, and the order of
//! operations is load-bearing in three places that are easy to get wrong.
//!
//! # 1. The reentrancy guard comes first — before the inner allocator
//!
//! The obvious shape is "call the inner allocator, then record". That
//! infinitely recurses if the inner allocator itself allocates through the
//! global allocator, which is exactly what wrapping a pool or arena allocator
//! invites. The guard must be held *across* the call into `A`.
//!
//! # 2. On free, the live-block entry goes before the inner free
//!
//! The instant `inner.dealloc(p)` returns, `p` belongs to whoever the allocator
//! hands it to next. Another thread can receive it and record its own entry
//! before the freeing thread gets around to removing the old one — destroying
//! the new owner's record and permanently mis-attributing a live block.
//!
//! `dhat-rs` is accidentally safe here only because it holds one global mutex
//! across both operations. A probe on macOS scored zero cross-thread reuse hits,
//! because Apple's per-thread magazines make immediate reuse rare; glibc's
//! shared arenas make it reachable. The same argument applies to a *moving*
//! `realloc`, which is why the shim removes the old entry before calling the
//! inner allocator and hands it to
//! [`Engine::record_realloc_taken`](crate::internals::engine::Engine::record_realloc_taken).
//!
//! # 3. `alloc_zeroed` must be overridden
//!
//! The default `GlobalAlloc::alloc_zeroed` calls `self.alloc` and then
//! `write_bytes`. Inheriting it would remove `calloc`'s lazy-zero-page fast path
//! from the program under test, changing its resident size and its timing — a
//! profiler that alters what it measures. `dhat-rs` has this bug. Forwarding
//! also avoids double-counting through `self.alloc`.
//!
//! # No allocation ever happens on this path
//!
//! Backtraces land in a fixed-size stack array; every other byte of profiler
//! state comes from a bump arena that reaches the system allocator directly.
//! This is the structural fix for the deadlock class that forced `dhat-rs` onto
//! a hand-rolled mutex.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::internals::engine::Engine;
use crate::internals::guard;
use crate::internals::shape::{Realloc, Shape};

/// Frames captured per allocation, at most.
///
/// The buffer is a stack array because the capture path cannot allocate, so
/// this is a fixed cost on every recorded allocation: 512 bytes of stack.
pub const CAPTURE_DEPTH: usize = 64;

// How many frames the capture machinery and this shim contribute is measured at
// startup, by `crate::unwind::internal_frames`, rather than being a constant
// here. It was a constant of 1 covering both strategies, which is right for the
// frame-pointer walk in a debug build and wrong everywhere else: the platform
// unwinder starts several `heapscope` frames further in, and an optimised
// frame-pointer walk starts one *later*, so it over-skipped in release and
// under-skipped with `Strategy::System`. Its comment claimed frame trimming
// would clean up the leftovers at output time. Frame trimming now exists —
// `symbol::trim` — and that is still not an answer: it reads names, so it does
// nothing on a stripped build or on Linux, and a skip that is too *large*
// removes real frames that no output-time pass can bring back. The measurement
// is what makes the frames right; trimming only decides which right ones to
// show.

/// The process-wide recording engine.
///
/// A plain `static` with no lazy initialization, because the shim is live before
/// `main`. One engine per process matches one profiler per process, which
/// [`Profiler::new`](crate::Profiler) enforces.
static ENGINE: Engine = Engine::new();

/// The recording engine.
pub fn engine() -> &'static Engine {
    &ENGINE
}

/// Set the first time the shim allocates, whatever the engine is doing.
///
/// A profiler whose shim was never installed records nothing and reports zeros,
/// and zeros are the one answer no reader can tell from a real one — see
/// [`StartError::NotInstalled`](crate::StartError::NotInstalled). This is the
/// only evidence available that `Alloc` is the `#[global_allocator]`, because
/// nothing else in the process can be asked.
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// Records that the shim ran.
///
/// One relaxed load and a branch that predicts perfectly because it goes one
/// way exactly once in the life of the process — the same shape as
/// [`unwind::strategy`](crate::unwind::strategy), and chosen for the same
/// reason. The store is guarded rather than unconditional so that the line
/// stays shared in every core's cache after the first allocation; an
/// unconditional store would bounce it between cores on every allocation, which
/// is the cost PLAN.md section 11 already blames for the crate's thread
/// scaling.
///
/// Only `alloc` calls this, not all four methods, because the only thing that
/// ever reads it is [`installed`] and the probe it answers allocates.
#[inline(always)]
fn note_installed() {
    if !INSTALLED.load(Ordering::Relaxed) {
        INSTALLED.store(true, Ordering::Relaxed);
    }
}

/// Whether the shim has allocated at least once in this process.
///
/// Meaningful only just after something is known to have allocated; on its own
/// a `false` says "no allocation has happened yet", not "not installed". The
/// caller is [`Profiler::start`](crate::Profiler), which forces one.
pub(crate) fn installed() -> bool {
    INSTALLED.load(Ordering::Relaxed)
}

/// A `GlobalAlloc` that records every allocation, wrapping another allocator.
///
/// # Contract on `A`
///
/// **`A` must not allocate through the global allocator.** A pool or arena
/// allocator that does will recurse: the shim holds its reentrancy guard across
/// the call into `A`, so the recursive entry is refused and merely goes
/// unrecorded — but if `A`'s own recursion is unbounded, nothing here can stop
/// it. `System` and every allocator that reaches the OS directly (jemalloc,
/// mimalloc, snmalloc) satisfy this.
///
/// # Example
///
/// ```
/// #[global_allocator]
/// static ALLOC: heapscope::Alloc = heapscope::Alloc::system();
/// ```
#[derive(Debug, Default)]
pub struct Alloc<A: GlobalAlloc = System> {
    inner: A,
}

impl Alloc<System> {
    /// Wraps the system allocator.
    pub const fn system() -> Self {
        Self { inner: System }
    }
}

impl<A: GlobalAlloc> Alloc<A> {
    /// Wraps `inner`.
    ///
    /// See the type documentation for the contract `inner` must satisfy.
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }

    /// The wrapped allocator.
    pub fn inner(&self) -> &A {
        &self.inner
    }
}

/// Captures a backtrace for the event being recorded.
///
/// Returns the number of frames written. The guard is required, and not merely
/// as documentation: obtaining the thread's stack bounds can allocate the first
/// time on the glibc main thread.
///
/// `#[inline(always)]`, not `#[inline]`, and that is load-bearing rather than an
/// optimisation. The calibrated skip is measured as "everything below the
/// function that called the capturing one", so it leaves the *caller of this
/// function's caller* as the innermost frame. If this survived as a frame of its
/// own — which it does in a debug build under a plain `#[inline]` — then there
/// would be two `heapscope` frames where the skip accounts for one, and every
/// program point would start inside the profiler.
///
/// [`crate::event`](fn@crate::event) and [`crate::copied`] capture through here too, and are
/// `#[inline(never)]` for the same reason the shim's methods are: they stand
/// where a shim method stands, one frame below the code being profiled.
#[inline(always)]
pub(crate) fn capture(guard: &guard::Guard, buffer: &mut [usize; CAPTURE_DEPTH]) -> usize {
    // One relaxed load of a value written once at startup, and a branch that
    // predicts perfectly because it never changes. See `unwind::SELECTED`.
    let strategy = crate::unwind::strategy();
    // The stack bounds are only meaningful to the frame-pointer walk, and
    // querying them is not free, so the platform unwinder does not pay for them.
    let bounds = match strategy {
        crate::unwind::Strategy::FramePointer => guard.stack_bounds(),
        crate::unwind::Strategy::System => None,
    };
    // The depth limit is applied by handing the unwinder a shorter buffer, not
    // by cutting the result afterwards. Both produce the same frames; only this
    // one makes the profile say what happened, because a walk that stops because
    // the buffer is full reports itself as truncated, and a walk cut silently
    // afterwards would still claim it reached the outermost frame.
    //
    // How short is a question for the strategy, not arithmetic here: on unix the
    // platform unwinder spends the caller's buffer on the frames it is about to
    // discard. See `unwind::depth_room`.
    let skip = crate::unwind::internal_frames();
    let room = buffer.len().min(crate::unwind::depth_room(
        strategy,
        crate::engine().max_depth(),
        skip,
    ));
    let capture = crate::unwind::capture_with(strategy, bounds, skip, &mut buffer[..room]);
    // One relaxed increment. PLAN.md section 5.4 exists because the startup
    // probe walks *our* frames, which says nothing about `cc`-built
    // dependencies, hand-written assembly, or threads a C library created — so
    // the profile reports how many captures actually came back whole.
    crate::unwind::counters().record(capture.outcome);
    capture.len
}

/// # Why every method here is `#[inline(never)]`
///
/// Not an optimisation choice — the frame layout depends on it. The skip the
/// capture applies is measured at startup as "frames below the caller of the
/// function that captured", and in the shim that capturing function is one of
/// these four. If one were inlined into the code that allocated, the capture
/// would start a frame too far out and the profile would lose the call site it
/// exists to name. Verified in both debug and release by
/// `a_recorded_allocation_starts_at_the_code_that_made_it`, which fails in
/// either direction.
///
/// The cost is one call instruction that cannot be elided, against a recorded
/// event that costs tens of nanoseconds. No benchmark covers the shim end to
/// end, so that is a reasoned bound rather than a measured one.
// SAFETY: every method forwards to `self.inner` with exactly the arguments it
// was given and returns exactly what `inner` returned, so the `GlobalAlloc`
// contract holds to precisely the degree `A` upholds it. The recording done
// alongside reads the pointer value but never dereferences it, never retains it
// past the call, and never allocates through this allocator.
unsafe impl<A: GlobalAlloc> GlobalAlloc for Alloc<A> {
    #[inline(never)]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Outside the guard and before anything else, because it has to be true
        // even for an allocation this shim declines to record: the engine is
        // idle when `Profiler::start` asks, which is the whole point of asking
        // then. See `note_installed`.
        note_installed();

        // Acquired *before* the inner call: if `A` allocates, the recursive
        // entry must find the guard already held.
        let guard = guard::enter();

        // SAFETY: forwarding the caller's own valid layout.
        let ptr = unsafe { self.inner.alloc(layout) };

        if let Some(guard) = &guard {
            if !ptr.is_null() && ENGINE.records_allocations() {
                let shape = Shape::of_layout(layout, false);
                // Counted here, captured only if wanted. On a sampled run this
                // is where most allocations stop, before the stack walk that is
                // most of what recording costs.
                if ENGINE.observe(guard, shape) {
                    let mut frames = [0usize; CAPTURE_DEPTH];
                    let len = capture(guard, &mut frames);
                    ENGINE.record_alloc(guard, ptr as usize, shape, &frames[..len]);
                }
            }
        }
        ptr
    }

    #[inline(never)]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let guard = guard::enter();

        // Forwarded rather than inherited. See the module documentation: the
        // default implementation would strip `calloc`'s lazy-zero-page path out
        // of the program being profiled.
        //
        // SAFETY: forwarding the caller's own valid layout.
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };

        if let Some(guard) = &guard {
            if !ptr.is_null() && ENGINE.records_allocations() {
                // The one place `zeroed` is true. Which method the program
                // called is not recoverable from the layout — `alloc` and
                // `alloc_zeroed` are handed identical ones — so it has to be
                // stated here or it is lost.
                let shape = Shape::of_layout(layout, true);
                if ENGINE.observe(guard, shape) {
                    let mut frames = [0usize; CAPTURE_DEPTH];
                    let len = capture(guard, &mut frames);
                    ENGINE.record_alloc(guard, ptr as usize, shape, &frames[..len]);
                }
            }
        }
        ptr
    }

    #[inline(never)]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let guard = guard::enter();

        // Before the inner free, not after. Once `inner.dealloc` returns, this
        // address may already belong to another thread's allocation.
        if guard.is_some() && ENGINE.records_allocations() {
            ENGINE.record_free(ptr as usize, layout.size());
        }

        // SAFETY: forwarding the caller's own valid pointer and layout.
        unsafe { self.inner.dealloc(ptr, layout) }
    }

    #[inline(never)]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let guard = guard::enter();
        let recording = guard.is_some() && ENGINE.records_allocations();

        // Removed before the inner call for the same reason as in `dealloc`: a
        // moving `realloc` frees the old address, after which another thread may
        // own it.
        let taken = if recording {
            ENGINE.live_blocks().remove(ptr as usize)
        } else {
            None
        };

        // SAFETY: forwarding the caller's own valid pointer, layout, and size.
        let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };

        if let Some(guard) = &guard {
            if recording {
                if new_ptr.is_null() {
                    // The reallocation failed, so the original block is still
                    // live and still owned by the program. Put its record back,
                    // or it would leak from the profile's accounting.
                    if let Some(block) = taken {
                        ENGINE
                            .live_blocks()
                            .insert(ENGINE.arena(), ptr as usize, block);
                    }
                } else {
                    let realloc = Realloc {
                        old_address: ptr as usize,
                        old_size: layout.size(),
                        new_address: new_ptr as usize,
                        // `GlobalAlloc::realloc` keeps the alignment of the
                        // original layout and guarantees nothing about the
                        // contents of any growth, so the size is the only
                        // part of the shape that moves.
                        new: Shape::of(new_size).aligned(layout.align()),
                    };
                    // A resize of a block this run already tracks is always
                    // recorded: its live bytes are standing in the counters and
                    // would never come back down otherwise. Only a resize of an
                    // untracked block is a fresh allocation, and only that one
                    // asks the sampler.
                    if ENGINE.observe_realloc(guard, &realloc, taken.is_some()) {
                        let mut frames = [0usize; CAPTURE_DEPTH];
                        let len = capture(guard, &mut frames);
                        ENGINE.record_realloc_taken(guard, taken, realloc, &frames[..len]);
                    }
                }
            }
        }
        new_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An allocator that allocates through the *global* allocator on every call
    /// — the contract violation the type documentation warns about, and the
    /// shape that makes "call inner first, then record" recurse forever.
    struct SelfAllocating;

    // SAFETY: forwards to `System` with unmodified arguments.
    unsafe impl GlobalAlloc for SelfAllocating {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // A `Vec` here goes through whatever `#[global_allocator]` the test
            // binary installed, which is the recursion under test.
            let scratch: Vec<u8> = Vec::with_capacity(16);
            std::hint::black_box(&scratch);
            // SAFETY: forwarding the caller's own valid layout.
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: forwarding the caller's own valid pointer and layout.
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    /// Wrapping an allocator that allocates must not recurse.
    ///
    /// This does not install the wrapper globally — a binary may have only one
    /// global allocator — but it does exercise the guard-before-inner ordering,
    /// which is the property at issue.
    #[test]
    fn a_self_allocating_inner_allocator_does_not_recurse() {
        let alloc = Alloc::new(SelfAllocating);
        let layout = Layout::from_size_align(128, 16).unwrap();

        for _ in 0..100 {
            // SAFETY: a valid non-zero-size layout; the pointer is freed below
            // with the same layout.
            unsafe {
                let ptr = alloc.alloc(layout);
                assert!(!ptr.is_null());
                alloc.dealloc(ptr, layout);
            }
        }
    }

    #[test]
    fn alloc_zeroed_returns_zeroed_memory() {
        let alloc = Alloc::system();
        let layout = Layout::from_size_align(256, 8).unwrap();

        // SAFETY: a valid layout; the block is read within its bounds and freed
        // with the same layout.
        unsafe {
            let ptr = alloc.alloc_zeroed(layout);
            assert!(!ptr.is_null());
            let bytes = std::slice::from_raw_parts(ptr, 256);
            assert!(bytes.iter().all(|&b| b == 0), "alloc_zeroed left garbage");
            alloc.dealloc(ptr, layout);
        }
    }

    /// `alloc_zeroed` must reach the inner allocator's own implementation, not
    /// fall back to `alloc` + `write_bytes`.
    #[test]
    fn alloc_zeroed_forwards_to_the_inner_allocator() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static ZEROED_CALLS: AtomicUsize = AtomicUsize::new(0);
        static PLAIN_CALLS: AtomicUsize = AtomicUsize::new(0);

        struct Counting;

        // SAFETY: forwards to `System` with unmodified arguments.
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                PLAIN_CALLS.fetch_add(1, Ordering::Relaxed);
                // SAFETY: forwarding the caller's own valid layout.
                unsafe { System.alloc(layout) }
            }
            unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
                ZEROED_CALLS.fetch_add(1, Ordering::Relaxed);
                // SAFETY: forwarding the caller's own valid layout.
                unsafe { System.alloc_zeroed(layout) }
            }
            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                // SAFETY: forwarding the caller's own valid pointer and layout.
                unsafe { System.dealloc(ptr, layout) }
            }
        }

        let alloc = Alloc::new(Counting);
        let layout = Layout::from_size_align(64, 8).unwrap();

        // SAFETY: a valid layout, freed with the same layout.
        unsafe {
            let ptr = alloc.alloc_zeroed(layout);
            alloc.dealloc(ptr, layout);
        }

        assert_eq!(ZEROED_CALLS.load(Ordering::Relaxed), 1);
        assert_eq!(
            PLAIN_CALLS.load(Ordering::Relaxed),
            0,
            "alloc_zeroed fell back to alloc, destroying calloc's lazy-zero-page path"
        );
    }

    #[test]
    fn realloc_preserves_contents() {
        let alloc = Alloc::system();
        let layout = Layout::from_size_align(64, 8).unwrap();

        // SAFETY: a valid layout; the block is written and read within bounds,
        // reallocated with its own layout, and freed with the resulting one.
        unsafe {
            let ptr = alloc.alloc(layout);
            assert!(!ptr.is_null());
            std::slice::from_raw_parts_mut(ptr, 64).fill(0xAB);

            let grown = alloc.realloc(ptr, layout, 512);
            assert!(!grown.is_null());
            let bytes = std::slice::from_raw_parts(grown, 64);
            assert!(bytes.iter().all(|&b| b == 0xAB), "realloc lost contents");

            alloc.dealloc(grown, Layout::from_size_align(512, 8).unwrap());
        }
    }

    /// A failing `realloc` leaves the original block live, and the profiler's
    /// record of it must survive too.
    #[test]
    fn a_failed_realloc_restores_the_original_record() {
        struct FailingRealloc;

        // SAFETY: `alloc` and `dealloc` forward to `System`; `realloc` returns
        // null, which the contract permits and which leaves the original block
        // untouched and still owned by the caller.
        unsafe impl GlobalAlloc for FailingRealloc {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                // SAFETY: forwarding the caller's own valid layout.
                unsafe { System.alloc(layout) }
            }
            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                // SAFETY: forwarding the caller's own valid pointer and layout.
                unsafe { System.dealloc(ptr, layout) }
            }
            unsafe fn realloc(&self, _ptr: *mut u8, _layout: Layout, _new: usize) -> *mut u8 {
                std::ptr::null_mut()
            }
        }

        let alloc = Alloc::new(FailingRealloc);
        let layout = Layout::from_size_align(64, 8).unwrap();

        // SAFETY: a valid layout; the null return leaves `ptr` valid, and it is
        // freed with its original layout.
        unsafe {
            let ptr = alloc.alloc(layout);
            assert!(!ptr.is_null());
            assert!(alloc.realloc(ptr, layout, 128).is_null());
            // The original is still valid and still ours to free.
            alloc.dealloc(ptr, layout);
        }
    }

    #[test]
    fn zero_sized_and_large_alignments_pass_through() {
        let alloc = Alloc::system();
        for align in [1usize, 8, 16, 64, 4096] {
            for size in [1usize, 7, 4096] {
                let layout = Layout::from_size_align(size, align).unwrap();
                // SAFETY: a valid layout, freed with the same layout.
                unsafe {
                    let ptr = alloc.alloc(layout);
                    assert!(!ptr.is_null(), "failed for size {size} align {align}");
                    assert!(ptr.addr().is_multiple_of(align));
                    alloc.dealloc(ptr, layout);
                }
            }
        }
    }

    /// The shim's recorded frames must start at the code that allocated.
    ///
    /// This is the property `SKIP_FRAMES` exists for, and until now nothing
    /// checked it. The constant was 1 for both strategies — right for the
    /// frame-pointer walk, wrong for the platform unwinder, which starts
    /// several `heapscope` frames further in — so every profile the latter
    /// produced began with the profiler's own code. Setting the skip to a wrong
    /// value passed the entire suite, including the end-to-end symbolization
    /// test.
    ///
    /// It claims the process-wide engine, so it is the only test that may.
    #[test]
    #[cfg_attr(miri, ignore = "reads real stack memory and calls the platform")]
    fn a_recorded_allocation_starts_at_the_code_that_made_it() {
        use crate::internals::clock::TimeSource;

        /// Allocates through the shim, and reports where it lives.
        #[inline(never)]
        fn allocating_site(alloc: &Alloc, layout: Layout) -> (*mut u8, usize) {
            // SAFETY: a valid layout; the caller frees with the same one.
            let pointer = unsafe { alloc.alloc(layout) };
            let marker = (allocating_site as fn(&Alloc, Layout) -> (*mut u8, usize)) as usize;
            std::hint::black_box((pointer, marker))
        }

        // Exactly what `Profiler::start` does, and the calibration is the part
        // that matters: `INTERNAL_FRAMES` is zero until a strategy is selected,
        // so an engine started without it captures with the wrong skip.
        assert!(
            ENGINE.start(TimeSource::Events, || crate::unwind::select(
                crate::unwind::Strategy::platform_default()
            )),
            "the process-wide engine was already claimed; only one test may do this"
        );

        let alloc = Alloc::system();
        // Distinctive, so the recorded point is unambiguous.
        let layout = Layout::from_size_align(9_973, 8).unwrap();
        let (pointer, marker) = allocating_site(&alloc, layout);
        assert!(!pointer.is_null());

        let mut innermost = None;
        ENGINE.flush_and_visit(
            Engine::FLUSH_TIMEOUT,
            |_id, frames, counters| {
                if counters.total_bytes == 9_973 && !frames.is_empty() {
                    innermost = Some(frames[0]);
                }
            },
            |_| {},
            |_| {},
        );
        // SAFETY: freed with the layout it was allocated with.
        unsafe { alloc.dealloc(pointer, layout) };
        ENGINE.stop(crate::internals::engine::Shutdown::Explicit);

        let innermost = innermost.expect("the allocation was not recorded");
        assert!(
            innermost >= marker && innermost - marker < 8192,
            "the innermost recorded frame is {innermost:#x}, which is not inside \
             the function that allocated ({marker:#x}); the profile would start \
             every program point with heapscope's own frames"
        );
    }
}
