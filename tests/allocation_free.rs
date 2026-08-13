//! Verifies that the primitives reachable from the allocator hot path perform
//! no allocation.
//!
//! This is the load-bearing claim of the whole design. If a lock allocates on
//! first use, the allocator shim recurses into itself, and the failure is a
//! stack overflow at process start on someone else's machine — not something a
//! code review catches.
//!
//! The check has to be an integration test because it installs a
//! `#[global_allocator]`, and a binary may only have one.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use heapscope::internals::lock::RawLock;

/// Miri interprets every instruction, so production iteration counts turn this
/// file into minutes of CI. Every interleaving these tests can expose shows up
/// in the first few hundred rounds.
#[cfg(miri)]
const ROUNDS: usize = 200;
#[cfg(not(miri))]
const ROUNDS: usize = 100_000;

#[cfg(miri)]
const CONTENDED_ROUNDS: usize = 100;
#[cfg(not(miri))]
const CONTENDED_ROUNDS: usize = 20_000;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Whether allocations on *this* thread count toward the measurement.
    ///
    /// A single global on/off switch does not work: `cargo test` runs the tests
    /// in one binary across several threads, so an unrelated test allocating in
    /// parallel lands in the measurement and the result is noise. Participation
    /// is therefore per-thread and opt-in, which also lets a contention test
    /// enrol its workers while excluding the cost of spawning them.
    ///
    /// `const`-initialized and destructor-free, so touching it from inside the
    /// allocator neither allocates nor depends on TLS destructor ordering.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

fn set_counting(on: bool) {
    COUNTING.with(|c| c.set(on));
}

#[inline]
fn counting() -> bool {
    // `try_with` rather than `with`: during thread teardown the slot is gone,
    // and a panic from inside the global allocator would abort the process.
    COUNTING.try_with(|c| c.get()).unwrap_or(false)
}

struct Counting;

// SAFETY: every method forwards to `System` with the same arguments, so the
// allocator contract is upheld exactly as `System` upholds it. The counters are
// observation only and do not affect the returned pointers.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if counting() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: forwarding the caller's own valid layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding the caller's own valid pointer and layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if counting() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: forwarding the caller's own valid layout. Forwarding rather
        // than falling back to `alloc` + `write_bytes` preserves `calloc`'s
        // lazy-zero-page path, which is the same reason the real shim does it.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if counting() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        // SAFETY: forwarding the caller's own valid pointer, layout, and size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Serializes measurements, because the counters themselves are global even
/// though participation is per-thread.
static MEASURING: RawLock = RawLock::new();

/// Opens a measurement window and reports `(count, bytes)` observed inside it.
///
/// `measure` deliberately does *not* enrol the calling thread. Enrolment is
/// always explicit, via [`counted`], for a reason that took a debugging session
/// to find: a helper thread that enrols before the measuring thread has taken
/// `MEASURING` will be counting during some *other* test's window, and on macOS
/// its `Barrier::wait` lazily allocates a `pthread_cond_t`. The result is a
/// handful of phantom allocations attributed to whichever test happened to be
/// running. Enrolling only inside the rendezvous makes that impossible.
fn measure(f: impl FnOnce()) -> (usize, usize) {
    let _guard = MEASURING.lock();
    ALLOCS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    f();
    (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

/// Runs `f` with this thread's allocations counted.
fn counted<R>(f: impl FnOnce() -> R) -> R {
    // Touch the slot before enrolling. `counted` runs *inside* the measurement
    // window, so this does not move the cost outside it — what it does is make
    // any first-touch cost land while `COUNTING` is still false, so it goes
    // uncounted rather than being attributed to the code under test. On every
    // platform measured so far a `const`-initialized destructor-free
    // thread-local costs nothing on first touch anyway; this makes the test
    // independent of that holding.
    let _ = counting();
    set_counting(true);
    let result = f();
    set_counting(false);
    result
}

/// Sanity check on the harness: if this fails, every other test here is
/// vacuously passing.
#[test]
fn the_counter_actually_counts() {
    let (count, bytes) = measure(|| {
        counted(|| {
            let v: Vec<u8> = Vec::with_capacity(4096);
            std::hint::black_box(&v);
        })
    });
    assert!(count >= 1, "harness did not observe a known allocation");
    assert!(bytes >= 4096, "harness did not observe the allocated bytes");
}

/// The first `lock()` on a fresh lock is the case that catches
/// `std::sync::Mutex`, which lazily allocates its platform mutex on Apple.
#[test]
fn first_lock_does_not_allocate() {
    let lock = RawLock::new();
    let (count, bytes) = measure(|| {
        counted(|| {
            let guard = lock.lock();
            drop(guard);
        })
    });
    assert_eq!(
        (count, bytes),
        (0, 0),
        "RawLock allocated on first acquire; this would recurse inside the allocator shim"
    );
}

#[test]
fn steady_state_locking_does_not_allocate() {
    let lock = RawLock::new();
    // Warm up outside the measurement so this test isolates steady state from
    // the first-acquire case above.
    drop(lock.lock());

    let (count, bytes) = measure(|| {
        counted(|| {
            for _ in 0..ROUNDS {
                let guard = lock.lock();
                std::hint::black_box(&guard);
            }
        })
    });
    assert_eq!((count, bytes), (0, 0), "RawLock allocated in steady state");
}

#[test]
fn try_lock_does_not_allocate_on_either_outcome() {
    static LOCK: RawLock = RawLock::new();
    let held_by_main = std::sync::Barrier::new(2);
    let probe_done = std::sync::Barrier::new(2);

    // The failure path has to be probed from another thread, because the lock
    // is not reentrant.
    std::thread::scope(|s| {
        s.spawn(|| {
            held_by_main.wait();
            counted(|| {
                assert!(
                    LOCK.try_lock().is_none(),
                    "try_lock succeeded on a held lock"
                );
            });
            probe_done.wait();
        });

        let (count, bytes) = measure(|| {
            counted(|| {
                drop(
                    LOCK.try_lock()
                        .expect("uncontended try_lock should succeed"),
                );
            });
            let held = LOCK.lock();
            held_by_main.wait();
            probe_done.wait();
            drop(held);
        });

        assert_eq!(
            (count, bytes),
            (0, 0),
            "try_lock allocated on the success or failure path"
        );
    });
}

/// Contended acquisition is the path that blocks in the kernel. On a platform
/// whose blocking path allocated a wait node, this is where it would show.
#[test]
fn contended_locking_does_not_allocate() {
    static LOCK: RawLock = RawLock::new();
    drop(LOCK.lock());

    // Only the workers are enrolled. The measuring thread's own `join` uses
    // `std::thread` machinery that is entitled to allocate, and that is the
    // harness rather than the lock. The barrier outlives the scope so the
    // spawned closures may borrow it.
    let barrier = std::sync::Barrier::new(5);
    let (count, bytes) = std::thread::scope(|s| {
        let barrier = &barrier;
        let handles: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| {
                    barrier.wait();
                    counted(|| {
                        for _ in 0..CONTENDED_ROUNDS {
                            let g = LOCK.lock();
                            std::hint::black_box(&g);
                        }
                    });
                })
            })
            .collect();

        measure(|| {
            barrier.wait();
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    assert_eq!(
        (count, bytes),
        (0, 0),
        "RawLock allocated on the contended (blocking) path"
    );
}

/// The companion to `the_counter_actually_counts`, for *worker* threads.
///
/// Two of the tests above prove their point only if allocations on an enrolled
/// worker are observed. Without this, a bug that silently dropped worker
/// enrolment would turn `contended_locking_does_not_allocate` into a test that
/// passes by measuring nothing at all.
#[test]
fn the_counter_counts_enrolled_worker_threads() {
    let ready = std::sync::Barrier::new(2);
    let done = std::sync::Barrier::new(2);

    std::thread::scope(|s| {
        s.spawn(|| {
            ready.wait();
            counted(|| {
                let v: Vec<u8> = Vec::with_capacity(8192);
                std::hint::black_box(&v);
            });
            done.wait();
        });

        let (count, bytes) = measure(|| {
            ready.wait();
            done.wait();
        });

        assert!(
            count >= 1 && bytes >= 8192,
            "an allocation on an enrolled worker thread was not observed \
             ({count} allocations, {bytes} bytes); the contention tests above \
             would be measuring nothing"
        );
    });
}

/// Reporting an event must not allocate.
///
/// `heapscope::event` and `heapscope::copied` are meant to be left in a hot
/// loop, and the recording one walks the same path the allocator shim does: the reentrancy guard, a stack capture into a
/// fixed array, interning, and the counters. Every one of those is designed to
/// be allocation-free, but until now nothing measured the *path* — only the
/// primitives beneath it. A profiler that allocated once per reported event
/// would be measuring its own instrumentation.
///
/// It would not recurse, and an earlier version of this comment claimed it
/// would: this binary installs `Counting` as its global allocator rather than
/// `heapscope::Alloc`, so nothing here can reenter the shim, and a reporting
/// function is refused at the mode check before it does any work in the one
/// mode where the shim is live.
///
/// The bump arena legitimately grows here, and that does not appear in this
/// count: it reaches [`std::alloc::System`] directly rather than through the
/// global allocator, which is the property that makes the shim safe at all.
///
/// This test claims the process-wide engine, so it is the only test in this file
/// that may.
#[test]
#[cfg_attr(miri, ignore = "starting a profiler captures a real backtrace")]
fn reporting_an_event_does_not_allocate() {
    let profiler = heapscope::Profiler::builder()
        .mode(heapscope::Mode::AdHoc)
        .no_output()
        .build()
        .expect("the profiler must start to measure anything about it");

    // Warm the arena and the program-point table first, so what is measured is
    // the steady state rather than one-time growth. First touch is measured too,
    // separately, because a cold path that allocates through the *global*
    // allocator is the failure this file exists to catch.
    let (cold, _) = measure(|| counted(|| heapscope::event(1)));
    for n in 0..ROUNDS.min(10_000) {
        heapscope::event(n as u64);
    }
    // Both functions. `copied` is refused by this AdHoc run, so what it
    // contributes here is its own body up to the mode check — and that is
    // exactly the surface `event` does not cover, because everything past the
    // check is the shared `record`, inlined into both. A review added an
    // allocation to `copied` alone and nothing failed.
    let (warm, bytes) = measure(|| {
        counted(|| {
            for n in 0..1_000u64 {
                heapscope::event(n);
                heapscope::copied(n as usize);
            }
        })
    });

    // Without this the test passes for the wrong reason: an `event` that
    // recorded nothing — a mode that never took, an engine that never started —
    // allocates nothing either.
    let recorded = profiler.stats().total_blocks;
    drop(profiler);
    assert!(
        recorded > 1_000,
        "only {recorded} events were recorded, so this measured a function that \
         returned immediately"
    );

    assert_eq!(cold, 0, "the first reported event allocated");
    assert_eq!(
        warm, 0,
        "reporting 2,000 events allocated {warm} times ({bytes} bytes)"
    );
}
