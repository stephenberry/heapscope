//! PLAN.md section 4.6: an allocation from a signal handler.
//!
//! A signal can arrive at any instruction, including one in the middle of the
//! allocator shim, and its handler can allocate — Rust's own panic machinery
//! does, and so does any handler that formats a message. The row states the
//! property as a *design* consequence rather than an accident:
//!
//! > The reentrancy guard covers this by design: a signal arriving while the
//! > interrupted thread is inside the shim sees the guard set and skips.
//!
//! Getting this wrong is not a wrong number. The interrupted thread may be
//! holding a live-block shard lock, and on Apple platforms reacquiring an
//! `os_unfair_lock` on the same thread does not deadlock — it `SIGKILL`s the
//! process, with no message and no core.
//!
//! # How the signal is made to arrive at the right instruction
//!
//! Not by racing. The inner allocator this test installs raises the signal
//! *itself*, from inside `dealloc` — which the shim calls with the guard held.
//! The handler therefore runs, deterministically, on top of a shim frame, which
//! is exactly the interleaving the row is about and one a timing-based test
//! would hit only occasionally.
//!
//! The arming is one-shot: a handler that allocates will free those blocks
//! later, and a `dealloc` that re-raised on the way out would recurse forever.

// Miri cannot deliver a signal: `signal` is a foreign function it does not
// implement on macOS, and the test's whole method is to raise one from inside
// `dealloc` so the handler lands on top of a shim frame. There is no version of
// that an interpreter can stage.
#![cfg(all(unix, not(miri)))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_int;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// `SIGUSR1`. 30 on Darwin; 10 on Linux for every architecture this crate
/// supports (it differs on MIPS, SPARC and Alpha, none of which are targets).
#[cfg(target_vendor = "apple")]
const SIGUSR1: c_int = 30;
#[cfg(all(unix, not(target_vendor = "apple")))]
const SIGUSR1: c_int = 10;

extern "C" {
    fn raise(signal: c_int) -> c_int;
    /// Returns the previous handler, or `SIG_ERR` (`usize::MAX`).
    fn signal(signal: c_int, handler: usize) -> usize;
}

/// Set while a `dealloc` should raise the signal. Cleared by the raising
/// `dealloc` itself, so exactly one does.
static ARMED: AtomicBool = AtomicBool::new(false);

/// An allocator that raises a signal from inside `dealloc`, once, when armed.
struct RaiseOnFree;

// SAFETY: every method forwards to `System` with the caller's own arguments and
// returns exactly what `System` returned. Raising a signal does not touch the
// allocation being freed, and the handler runs to completion before
// `System::dealloc` is called.
unsafe impl GlobalAlloc for RaiseOnFree {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding the caller's own layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ARMED.swap(false, Ordering::SeqCst) {
            // SAFETY: `raise` sends a signal to the calling thread and has no
            // preconditions beyond a valid signal number.
            unsafe { raise(SIGUSR1) };
        }
        // SAFETY: forwarding the caller's own valid pointer and layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding the caller's own layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarding the caller's own pointer, layout and size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: heapscope::Alloc<RaiseOnFree> = heapscope::Alloc::new(RaiseOnFree);

/// How many times the handler ran.
static HANDLER_RUNS: AtomicUsize = AtomicUsize::new(0);
/// Whether the guard was already held when the handler ran.
static GUARD_HELD_IN_HANDLER: AtomicBool = AtomicBool::new(false);
/// The address of a block the handler allocated and leaked.
static HANDLER_BLOCK: AtomicUsize = AtomicUsize::new(0);

/// Size of the blocks this test leaks, large enough not to be served from any
/// small-object cache that might reuse an address.
const BLOCK: usize = 8192;

extern "C" fn on_signal(_signal: c_int) {
    // Re-arm. `signal` has BSD semantics on Darwin and glibc, but the SysV
    // behaviour of resetting to the default disposition on delivery is
    // permitted by C, and re-arming is correct under either.
    // SAFETY: installing a valid `extern "C"` handler for a valid signal.
    unsafe { signal(SIGUSR1, (on_signal as extern "C" fn(c_int)) as usize) };

    // The property under test, read at the one instant it means anything.
    GUARD_HELD_IN_HANDLER.store(heapscope::internals::guard::is_entered(), Ordering::SeqCst);

    // What a real handler does, and what makes this dangerous: allocate. If the
    // guard were not already held this would re-enter the engine from inside a
    // shim call that has not returned — the shim holds the guard across the
    // inner allocator precisely so that it cannot.
    let block = vec![0xABu8; BLOCK];
    HANDLER_BLOCK.store(block.as_ptr() as usize, Ordering::SeqCst);
    std::mem::forget(block);

    HANDLER_RUNS.fetch_add(1, Ordering::SeqCst);
}

/// Allocates a block, leaks it, and returns its address.
fn leak_block() -> usize {
    let block = vec![0x5Au8; BLOCK];
    let address = block.as_ptr() as usize;
    std::mem::forget(block);
    address
}

/// Whether the engine has a live-block entry for `address`.
fn engine_recorded(address: usize) -> bool {
    heapscope::engine().live_blocks().get(address).is_some()
}

#[test]
fn a_signal_handler_that_allocates_cannot_reenter_the_engine() {
    // SAFETY: installing a valid `extern "C"` handler for a valid signal.
    let previous = unsafe { signal(SIGUSR1, (on_signal as extern "C" fn(c_int)) as usize) };
    assert_ne!(previous, usize::MAX, "could not install the signal handler");

    // Nothing here is worth a file, and writing one would only add noise.
    let profiler = heapscope::Profiler::builder()
        .no_output()
        .build()
        .expect("the profiler starts");

    // Starting a profiler allocates — the list of destinations the exit handler
    // keeps — and recording is already on by then. A profiler whose first
    // program point is its own setup is measuring itself.
    assert_eq!(
        heapscope::engine().stats().total_blocks,
        0,
        "starting a profiler recorded {} of its own allocations",
        heapscope::engine().stats().total_blocks
    );

    // Control. Without this the test below would pass just as well against an
    // engine that records nothing at all, which is the shape of a vacuous test.
    let ordinary = leak_block();
    assert!(
        engine_recorded(ordinary),
        "the engine did not record an ordinary allocation, so the check below \
         would prove nothing"
    );

    // Fire the signal from inside the shim: `victim`'s free calls the inner
    // allocator, which raises, all with the guard held.
    let victim = vec![1u8; BLOCK];
    ARMED.store(true, Ordering::SeqCst);
    drop(victim);

    assert_eq!(
        HANDLER_RUNS.load(Ordering::SeqCst),
        1,
        "the handler never ran, so nothing below was tested"
    );
    assert!(
        GUARD_HELD_IN_HANDLER.load(Ordering::SeqCst),
        "a signal handler interrupting the shim found the reentrancy guard \
         free. On Apple platforms the next lock acquisition would SIGKILL the \
         process."
    );

    let from_handler = HANDLER_BLOCK.load(Ordering::SeqCst);
    assert_ne!(from_handler, 0);
    assert!(
        !engine_recorded(from_handler),
        "an allocation made from a signal handler that interrupted the shim was \
         recorded, which means it re-entered the engine"
    );

    // And the guard is balanced afterwards: the handler took none, so the
    // interrupted frame's release still matches its own acquisition.
    assert!(
        !heapscope::internals::guard::is_entered(),
        "the guard was left held after the handler returned"
    );

    // Ordinary recording still works, so the refusal was scoped to the handler
    // rather than having wedged the guard for this thread.
    let after = leak_block();
    assert!(
        engine_recorded(after),
        "the engine stopped recording after the signal handler ran"
    );

    drop(profiler);
}
