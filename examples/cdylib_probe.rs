//! A fixture for `tests/cdylib_tls.rs`, built as a `cdylib`. Not an example.
//!
//! # The hazard this exists to reproduce
//!
//! PLAN.md section 4.7, and the reason the reentrancy guard uses no thread-local
//! storage at all. Rust's `thread_local!` in a **dynamic library** on Apple
//! platforms is reached through dyld's `__tlv_bootstrap`. On a thread's first
//! access, `tlv_get_addr` finds no block for the key, `malloc`s one, and only
//! *then* records it with `pthread_setspecific`. An allocator that consults a
//! thread-local to decide whether it is already inside itself therefore:
//!
//! 1. is entered, and touches its thread-local flag,
//! 2. which allocates, re-entering the allocator,
//! 3. which touches the same thread-local, finds it still unrecorded,
//! 4. and allocates again. Forever.
//!
//! `try_with` does not help: the slot is not *unavailable*, it is
//! mid-initialization, and the accessor cheerfully starts initializing it again.
//! Each level leaks a block and the stack runs out.
//!
//! One correction to that story, established by running it. The recursion needs
//! the thread-local's **initialization to allocate through the Rust global
//! allocator**. A `const`-initialized, destructor-free thread-local — the shape
//! `dhat-rs` uses — does *not* recurse here, because dyld's own TLV allocation
//! calls the C `malloc` in libsystem and a Rust `#[global_allocator]` does not
//! sit in front of that. See `src/core/guard.rs`.
//!
//! # Why it needs all three of these things
//!
//! A `cdylib`, loaded with `dlopen`, allocating on a thread that has **never
//! run any of this library's code before**. Drop any one and the bug hides.
//!
//! The third is the subtle one, and the first version of this fixture got it
//! wrong: it spawned the thread *inside* the library. That does not reproduce
//! anything, because macOS gives an image one TLV block per thread covering all
//! its thread-locals, and `std::thread`'s own startup touches std's — which are
//! compiled into this same library — before any of this code runs. By the time
//! the allocation happened the block already existed and the first-touch path
//! was long gone. Deliberately reintroducing the thread-local guard left that
//! version passing.
//!
//! So the **caller** creates the thread and calls in. Then the first thing this
//! image does on that thread is allocate, and the first thing the allocator does
//! is whatever the guard needs — which, if that were a thread-local, would be
//! the first TLV access for the image.
//!
//! The library's own `#[global_allocator]` covers the Rust code compiled into
//! it, which includes its own copy of heapscope. So the allocations the spawned
//! thread makes here really do run through this crate's shim, in a dynamic
//! library, on a thread with untouched TLS.

use std::sync::atomic::{AtomicUsize, Ordering};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// Blocks the spawned thread allocates. Enough that a per-level leak would be
/// obvious, few enough to stay instant.
const BLOCKS: usize = 512;

/// Result codes, so the caller can tell the failures apart without a panic
/// crossing the FFI boundary.
const OK: i32 = 0;
const COULD_NOT_START: i32 = 1;
const NOT_RECORDED: i32 = 2;
const GUARD_LEFT_HELD: i32 = 3;

static RECORDED: AtomicUsize = AtomicUsize::new(0);

/// Allocates on the **calling** thread and reports whether it worked.
///
/// Call this from a thread the caller created and that has never entered this
/// library before. That is the whole arrangement; see the module documentation.
///
/// # Safety
///
/// None beyond being called at most a few times; it is `extern "C"` only so that
/// `dlsym` can find it.
#[no_mangle]
pub extern "C" fn heapscope_cdylib_allocate_here() -> i32 {
    // The engine in *this* copy of heapscope, which is not the one in the test
    // binary that loaded us.
    let engine = heapscope::engine();
    let before = engine.stats().total_blocks;

    // The first thing this image does on this thread. If the guard consulted
    // thread-local storage, this is where dyld would `malloc` the TLV block,
    // re-enter the allocator, and find the block still unrecorded.
    let mut held: Vec<Vec<u8>> = Vec::with_capacity(BLOCKS);
    for size in 0..BLOCKS {
        held.push(vec![0xCDu8; 64 + size]);
    }
    // Read something back so nothing here can be optimised away.
    let total: usize = held.iter().map(Vec::len).sum();
    RECORDED.store(total, Ordering::SeqCst);

    // The guard must not still be held now the shim has returned. If the first
    // thread-local touch had left a depth behind, every later allocation on this
    // thread would be refused.
    if heapscope::internals::guard::is_entered() {
        return GUARD_LEFT_HELD;
    }

    // The allocations must have been *recorded*, not merely survived. A guard
    // wedged into the entered state produces a live process with an empty
    // profile, which is the quiet version of this failure.
    if engine.stats().total_blocks < before + BLOCKS as u64 {
        return NOT_RECORDED;
    }

    OK
}

/// Starts the profiler inside the loaded library. Separate from the probe so the
/// test can order the two explicitly.
#[no_mangle]
pub extern "C" fn heapscope_cdylib_start() -> i32 {
    // Both halves matter. Forgetting alone left the exit handler armed with the
    // default path, so `cargo test` dropped a `dhat-heap.json` into whatever
    // directory it ran in — the crate's own documented output name, silently
    // overwriting a real profile if one was there.
    match heapscope::Profiler::builder().no_output().build() {
        Ok(profiler) => {
            std::mem::forget(profiler);
            OK
        }
        Err(_) => COULD_NOT_START,
    }
}
