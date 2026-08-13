//! `RawLock` — a blocking mutual-exclusion lock that never allocates.
//!
//! # Why not `std::sync::Mutex`
//!
//! On Apple and other non-futex unix targets `std::sync::Mutex` routes through
//! `sys::pal::unix::sync::mutex`, which holds a `OnceBox<pal::Mutex>` whose own
//! doc comment reads *"used to implement synchronization primitives that need
//! allocation"*. A fresh `Mutex`'s first `lock()` performs one 64-byte
//! allocation. Inside a `GlobalAlloc` shim that is exactly the recursion this
//! crate exists to avoid.
//!
//! # Why not a hand-rolled spinlock
//!
//! Three reasons, all of which have bitten real profilers:
//!
//! - **Priority inversion.** Apple deprecated `OSSpinLock` for precisely this,
//!   and `sched_yield` does not donate priority on Darwin.
//! - **Shutdown hangs.** An unbounded spin inside an `atexit` handler, where
//!   other threads are still running, turns process exit into a hang.
//! - **No fairness** under a hot shard.
//!
//! # What this is instead
//!
//! A thin wrapper over the platform's own primitive. Each is allocation-free,
//! statically initializable to a byte pattern of all zeros, and reachable
//! through a library the process already links:
//!
//! | Platform | Primitive |
//! |---|---|
//! | Apple | `os_unfair_lock` |
//! | Other unix | `pthread_mutex_t`, zero-initialized (`PTHREAD_MUTEX_INITIALIZER`) |
//! | Windows | `SRWLOCK` |
//!
//! Every one of those has an all-zero initial state, which is what makes
//! [`RawLock::new`] a `const fn` and lets a shard array be a plain `static`.

use std::fmt;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

/// A mutual-exclusion lock that performs no allocation and can be placed in a
/// `static` without lazy initialization.
///
/// # Not reentrant, and the failure differs by platform
///
/// Acquiring the lock twice from one thread is a programming error. What it
/// does is worth stating precisely, because the platforms disagree and only one
/// of them is debuggable:
///
/// | Platform | Behaviour |
/// |---|---|
/// | Apple | `os_unfair_lock` detects the reacquire and **kills the process with `SIGKILL`**, with no message and no core |
/// | Linux | glibc's zero-initialized mutex is `PTHREAD_MUTEX_NORMAL`, for which POSIX defines relock as **deadlock** |
/// | Windows | Reacquiring an `SRWLOCK` exclusively is **undefined behaviour** |
///
/// The Apple case is the reason this crate documents a global lock order and
/// enforces it in debug builds (see [`super::order`]): on the primary
/// development platform, violating the order does not produce a hang you can
/// attach a debugger to, it produces an instant unattributable death.
///
/// # No destructor
///
/// `RawLock` does not call `pthread_mutex_destroy`. That is sound for the
/// platforms it supports — a glibc default mutex owns no resources beyond its
/// own storage, and neither `os_unfair_lock` nor `SRWLOCK` has a destroy
/// operation at all — and it is what allows the lock to live in a `static` that
/// is never torn down. A port to a platform whose mutex owns heap state would
/// have to revisit this; the `compile_error!` below is what forces that
/// conversation.
pub struct RawLock {
    inner: imp::Lock,
}

// SAFETY: `Send` — every backing primitive is address-stable while unlocked,
// and `RawLock` has no thread affinity of its own. (A *locked* lock must not be
// moved; that obligation is on `raw_lock`, and the guard returned by `lock` is
// `!Send` and borrows `self`, which prevents it through the safe API.)
unsafe impl Send for RawLock {}
// SAFETY: `Sync` — all three primitives are internally synchronized and are
// designed to be shared by reference across threads; that is their entire
// purpose. All mutation goes through them.
unsafe impl Sync for RawLock {}

impl RawLock {
    /// Creates a new unlocked lock.
    ///
    /// This is a `const fn` so that shard arrays can be plain `static`s. No
    /// lazy initialization is reachable from the allocator hot path.
    pub const fn new() -> Self {
        Self {
            inner: imp::Lock::new(),
        }
    }

    /// Acquires the lock, blocking until it is available.
    #[inline]
    pub fn lock(&self) -> RawGuard<'_> {
        // SAFETY: the guard returned here borrows `self` and unlocks exactly
        // once, in its `Drop`. It is `!Send`, so the unlock happens on the
        // acquiring thread, and `self` cannot be moved or dropped while the
        // borrow is live. Reentrancy is not a safety obligation (see the type
        // docs); it is a correctness one.
        unsafe { self.raw_lock() };
        RawGuard {
            lock: self,
            _not_send: PhantomData,
        }
    }

    /// Attempts to acquire the lock without blocking.
    #[inline]
    pub fn try_lock(&self) -> Option<RawGuard<'_>> {
        // SAFETY: as in `lock`; the guard is created only on success.
        if unsafe { self.raw_try_lock() } {
            Some(RawGuard {
                lock: self,
                _not_send: PhantomData,
            })
        } else {
            None
        }
    }

    /// Attempts to acquire the lock, giving up after `timeout`.
    ///
    /// This exists for the shutdown path. When a thread is wedged, or a lock
    /// was orphaned by a `fork` that this crate did not mediate, the profiler
    /// must degrade to partial output rather than hang the process at `exit`.
    ///
    /// The implementation polls rather than blocking on a timed primitive,
    /// because none of the three backing primitives offers a portable timed
    /// acquire. That is acceptable precisely because this is never called from
    /// the hot path.
    pub fn try_lock_for(&self, timeout: Duration) -> Option<RawGuard<'_>> {
        if let Some(guard) = self.try_lock() {
            return Some(guard);
        }
        // `Instant::now() + timeout` *panics* on overflow, and this function is
        // called from the `atexit` handler where a panic is a process abort.
        // A caller passing `Duration::MAX` means "wait as long as it takes", so
        // an unrepresentable deadline degrades to polling without one rather
        // than to a crash.
        let deadline = Instant::now().checked_add(timeout);
        let mut backoff = Duration::from_micros(10);
        loop {
            if let Some(guard) = self.try_lock() {
                return Some(guard);
            }
            let sleep_for = match deadline {
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        return None;
                    }
                    // `Instant: Sub` saturates, so this cannot panic.
                    backoff.min(deadline - now)
                }
                None => backoff,
            };
            std::thread::sleep(sleep_for);
            backoff = (backoff * 2).min(Duration::from_millis(5));
        }
    }

    /// Acquires the lock without producing a guard.
    ///
    /// Used by the `fork` handlers, which must acquire in `prepare` and release
    /// in a different function (`parent`), so a guard's scope cannot express
    /// the lifetime.
    ///
    /// # Safety
    ///
    /// - A matching [`RawLock::raw_unlock`] must run, **on the same thread**.
    ///   Releasing from another thread traps on Apple and is undefined on
    ///   Windows.
    /// - The lock must not be **moved** while held. `pthread_mutex_t` and
    ///   `SRWLOCK` are address-sensitive while locked: a parked waiter is
    ///   queued on the old address and would never be woken.
    ///
    /// Reentrant acquisition is deliberately *not* listed here. It is a
    /// correctness error with platform-specific consequences (see the type
    /// documentation), not a soundness one — and it cannot be a safety
    /// obligation, because the safe [`RawLock::lock`] calls this method and so
    /// could not discharge it.
    #[inline]
    pub unsafe fn raw_lock(&self) {
        // SAFETY: delegated to the caller's obligation; the platform primitive
        // itself is always valid because it is initialized at construction.
        unsafe { self.inner.lock() }
    }

    /// Attempts to acquire the lock without producing a guard, giving up after
    /// `timeout`.
    ///
    /// Returns `true` if the lock was acquired. The guardless form exists for
    /// the `fork` handlers, which acquire in `prepare` and release in `parent`.
    ///
    /// # Safety
    ///
    /// As [`RawLock::raw_lock`], if this returns `true`.
    #[inline]
    pub unsafe fn try_lock_for_raw(&self, timeout: Duration) -> bool {
        match self.try_lock_for(timeout) {
            Some(guard) => {
                // The lock stays held; the caller has taken on the obligation to
                // release it, which is the whole point of the guardless form.
                std::mem::forget(guard);
                true
            }
            None => false,
        }
    }

    /// Attempts to acquire the lock without producing a guard.
    ///
    /// Returns `true` if the lock was acquired.
    ///
    /// # Safety
    ///
    /// As [`RawLock::raw_lock`], if this returns `true`.
    #[inline]
    pub unsafe fn raw_try_lock(&self) -> bool {
        // SAFETY: delegated to the caller's obligation.
        unsafe { self.inner.try_lock() }
    }

    /// Releases the lock.
    ///
    /// # Safety
    ///
    /// The lock must currently be held by the calling thread through
    /// [`RawLock::raw_lock`] or [`RawLock::raw_try_lock`].
    #[inline]
    pub unsafe fn raw_unlock(&self) {
        // SAFETY: delegated to the caller's obligation.
        unsafe { self.inner.unlock() }
    }

    /// Resets the lock to the unlocked state, discarding any ownership record.
    ///
    /// This is the `fork` child handler's tool. A lock held by a thread that
    /// does not exist in the child can never be released; the only recovery is
    /// to overwrite it with the initial state.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that no other thread can observe the lock
    /// during the reset. In practice that means: call this only from a
    /// `pthread_atfork` child handler, where the child process is
    /// single-threaded by definition.
    #[inline]
    pub unsafe fn force_reinit(&self) {
        // SAFETY: delegated to the caller's single-threadedness obligation.
        unsafe { self.inner.force_reinit() }
    }
}

impl Default for RawLock {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for RawLock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Deliberately does not probe the lock state: `try_lock` in a `Debug`
        // impl would make formatting a side-effecting operation.
        f.debug_struct("RawLock").finish_non_exhaustive()
    }
}

/// Proof that a [`RawLock`] is held. Releases the lock when dropped.
///
/// The guard is neither `Send` nor `Sync`: `os_unfair_lock` records the owning
/// thread and traps if released by another, and `pthread_mutex_unlock` from a
/// non-owning thread is undefined. The raw-pointer marker below is what enforces
/// that, since negative impls are not available on stable.
#[must_use = "the lock is released immediately if the guard is not bound"]
pub struct RawGuard<'a> {
    lock: &'a RawLock,
    _not_send: PhantomData<*const ()>,
}

impl Drop for RawGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: the guard's existence proves the lock is held by this thread,
        // and a guard is constructed only after a successful acquire.
        unsafe { self.lock.raw_unlock() }
    }
}

impl fmt::Debug for RawGuard<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawGuard").finish_non_exhaustive()
    }
}

/// Miri backend.
///
/// Miri does shim `os_unfair_lock`, but its shim performs a non-atomic read of
/// the lock word, so the data-race detector reports a race between one thread's
/// `os_unfair_lock_lock` and another's `os_unfair_lock_unlock`. That is a
/// modelling artefact of code we did not write, and the obvious workaround —
/// running Miri with `-Zmiri-disable-data-race-detector` — would switch off the
/// single most valuable check for a crate whose whole difficulty is concurrent
/// shared state.
///
/// So under Miri the platform primitive is replaced by a pure-Rust atomic lock.
/// The race detector then stays on for everything this crate actually owns: the
/// arena, the tables, the epoch algorithm, and the peak gate.
///
/// This is a spinlock, which the module documentation above argues against at
/// some length. The objections there — priority inversion, unbounded spinning
/// during `atexit`, unfairness — are all about wall-clock behaviour on a real
/// machine with a real scheduler. Under Miri there is no real scheduler and no
/// wall clock; execution is interleaved deterministically at each atomic
/// operation. None of the objections apply, and no shipped binary contains this.
#[cfg(miri)]
mod imp {
    use std::sync::atomic::{AtomicU32, Ordering};

    const UNLOCKED: u32 = 0;
    const LOCKED: u32 = 1;

    pub(super) struct Lock {
        state: AtomicU32,
    }

    impl Lock {
        pub(super) const fn new() -> Self {
            Self {
                state: AtomicU32::new(UNLOCKED),
            }
        }

        pub(super) unsafe fn lock(&self) {
            // `_weak` is correct here and only here: a spurious failure just
            // costs another iteration of a loop that was going to spin anyway.
            while self
                .state
                .compare_exchange_weak(UNLOCKED, LOCKED, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                // Miri preempts at atomic operations, so this makes progress.
                std::thread::yield_now();
            }
        }

        pub(super) unsafe fn try_lock(&self) -> bool {
            // Must be the *strong* exchange. `try_lock` reports "someone else
            // holds this", and a spurious failure would be a lie — one that
            // Miri, which fails `_weak` deliberately, catches immediately.
            self.state
                .compare_exchange(UNLOCKED, LOCKED, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        }

        pub(super) unsafe fn unlock(&self) {
            self.state.store(UNLOCKED, Ordering::Release);
        }

        pub(super) unsafe fn force_reinit(&self) {
            // An atomic store rather than a plain write, so that this remains
            // race-free by construction even if a caller is sloppier than the
            // single-threaded `fork` child contract requires.
            self.state.store(UNLOCKED, Ordering::Release);
        }
    }
}

#[cfg(all(target_vendor = "apple", not(miri)))]
mod imp {
    use std::cell::UnsafeCell;
    use std::ffi::c_void;

    // `os_unfair_lock_s` is a single 32-bit opaque word whose zero value is
    // `OS_UNFAIR_LOCK_INIT`.
    #[repr(C)]
    struct OsUnfairLock {
        opaque: u32,
    }

    // These live in libSystem, which every Rust process on Apple platforms
    // already links through `std`.
    extern "C" {
        fn os_unfair_lock_lock(lock: *mut c_void);
        fn os_unfair_lock_trylock(lock: *mut c_void) -> bool;
        fn os_unfair_lock_unlock(lock: *mut c_void);
    }

    pub(super) struct Lock {
        inner: UnsafeCell<OsUnfairLock>,
    }

    impl Lock {
        pub(super) const fn new() -> Self {
            Self {
                inner: UnsafeCell::new(OsUnfairLock { opaque: 0 }),
            }
        }

        #[inline]
        pub(super) unsafe fn lock(&self) {
            // SAFETY: the cell always holds a validly initialized lock.
            unsafe { os_unfair_lock_lock(self.inner.get().cast()) }
        }

        #[inline]
        pub(super) unsafe fn try_lock(&self) -> bool {
            // SAFETY: as above.
            unsafe { os_unfair_lock_trylock(self.inner.get().cast()) }
        }

        #[inline]
        pub(super) unsafe fn unlock(&self) {
            // SAFETY: as above; the caller guarantees ownership.
            unsafe { os_unfair_lock_unlock(self.inner.get().cast()) }
        }

        #[inline]
        pub(super) unsafe fn force_reinit(&self) {
            // SAFETY: the caller guarantees the process is single-threaded
            // (a `fork` child), so this write cannot race.
            unsafe { self.inner.get().write(OsUnfairLock { opaque: 0 }) }
        }
    }
}

/// glibc backend.
///
/// Scoped to Linux rather than to "unix that is not Apple", which is what an
/// earlier version said and which was quietly wrong. On FreeBSD, OpenBSD, and
/// NetBSD `pthread_mutex_t` is a *pointer* and `PTHREAD_MUTEX_INITIALIZER` is
/// null; `pthread_mutex_lock` lazily `calloc`s the real mutex on first use.
/// Over-sized storage would keep that memory-safe, but "allocation-free and
/// statically initialized" — the entire reason this type exists — would become
/// false, and the lazy `calloc` would be a `malloc` inside the allocator shim:
/// precisely the recursion this crate is built to prevent. Failing to compile
/// is the correct outcome, and the `compile_error!` below delivers it.
#[cfg(all(target_os = "linux", not(miri)))]
mod imp {
    use std::cell::UnsafeCell;
    use std::ffi::{c_int, c_void};

    // `pthread_mutex_t` is 40 bytes on x86_64-linux-gnu and 48 on
    // aarch64-linux-gnu. Over-sizing is harmless — the implementation only
    // touches the leading bytes — while under-sizing is memory corruption, so
    // this is deliberately generous. Alignment must be at least that of a
    // pointer, because the robust-mutex list threads a pointer through the
    // first word.
    const MUTEX_BYTES: usize = 64;

    #[repr(C, align(8))]
    struct PthreadMutex {
        // `PTHREAD_MUTEX_INITIALIZER` is all zeros for the default mutex kind
        // on glibc. That is what makes `new` a `const fn`.
        storage: [u8; MUTEX_BYTES],
    }

    extern "C" {
        fn pthread_mutex_lock(mutex: *mut c_void) -> c_int;
        fn pthread_mutex_trylock(mutex: *mut c_void) -> c_int;
        fn pthread_mutex_unlock(mutex: *mut c_void) -> c_int;
    }

    pub(super) struct Lock {
        inner: UnsafeCell<PthreadMutex>,
    }

    impl Lock {
        pub(super) const fn new() -> Self {
            Self {
                inner: UnsafeCell::new(PthreadMutex {
                    storage: [0; MUTEX_BYTES],
                }),
            }
        }

        #[inline]
        pub(super) unsafe fn lock(&self) {
            // SAFETY: the cell always holds a zero-initialized (hence valid)
            // default mutex.
            let rc = unsafe { pthread_mutex_lock(self.inner.get().cast()) };
            if rc != 0 {
                // Deliberately not `debug_assert!`. This can run inside the
                // allocator shim, where building a panic message allocates and
                // re-enters, and where unwinding out of a `GlobalAlloc` method
                // is undefined. A non-zero return means a caller contract
                // violation (`EDEADLK`, `EINVAL`), so the profiler stops
                // recording and says why.
                crate::internals::diagnostic::poison("pthread_mutex_lock failed");
            }
        }

        #[inline]
        pub(super) unsafe fn try_lock(&self) -> bool {
            // SAFETY: as above.
            unsafe { pthread_mutex_trylock(self.inner.get().cast()) == 0 }
        }

        #[inline]
        pub(super) unsafe fn unlock(&self) {
            // SAFETY: as above; the caller guarantees ownership.
            let rc = unsafe { pthread_mutex_unlock(self.inner.get().cast()) };
            if rc != 0 {
                // See `lock`: never panic from a path the shim can reach.
                crate::internals::diagnostic::poison("pthread_mutex_unlock failed");
            }
        }

        #[inline]
        pub(super) unsafe fn force_reinit(&self) {
            // SAFETY: the caller guarantees the process is single-threaded (a
            // `fork` child). Writing the static initializer is exactly what
            // `pthread_mutex_init` with default attributes produces, and it is
            // the only way to reclaim a mutex whose owner no longer exists.
            unsafe {
                self.inner.get().write(PthreadMutex {
                    storage: [0; MUTEX_BYTES],
                })
            }
        }
    }
}

#[cfg(all(windows, not(miri)))]
mod imp {
    use std::cell::UnsafeCell;
    use std::ffi::c_void;

    // `SRWLOCK` is a single pointer-sized field; `SRWLOCK_INIT` is null.
    #[repr(C)]
    struct SrwLock {
        ptr: *mut c_void,
    }

    // `raw-dylib` avoids needing an import library at link time.
    #[link(name = "kernel32", kind = "raw-dylib")]
    extern "system" {
        fn AcquireSRWLockExclusive(lock: *mut c_void);
        fn TryAcquireSRWLockExclusive(lock: *mut c_void) -> u8;
        fn ReleaseSRWLockExclusive(lock: *mut c_void);
    }

    pub(super) struct Lock {
        inner: UnsafeCell<SrwLock>,
    }

    impl Lock {
        pub(super) const fn new() -> Self {
            Self {
                inner: UnsafeCell::new(SrwLock {
                    ptr: std::ptr::null_mut(),
                }),
            }
        }

        #[inline]
        pub(super) unsafe fn lock(&self) {
            // SAFETY: the cell always holds a validly initialized `SRWLOCK`.
            unsafe { AcquireSRWLockExclusive(self.inner.get().cast()) }
        }

        #[inline]
        pub(super) unsafe fn try_lock(&self) -> bool {
            // SAFETY: as above.
            unsafe { TryAcquireSRWLockExclusive(self.inner.get().cast()) != 0 }
        }

        #[inline]
        pub(super) unsafe fn unlock(&self) {
            // SAFETY: as above; the caller guarantees ownership.
            unsafe { ReleaseSRWLockExclusive(self.inner.get().cast()) }
        }

        #[inline]
        pub(super) unsafe fn force_reinit(&self) {
            // Windows has no `fork`, so this is unreachable in practice; it
            // exists to keep the platform backends interchangeable.
            //
            // SAFETY: the caller guarantees no concurrent observers.
            unsafe {
                self.inner.get().write(SrwLock {
                    ptr: std::ptr::null_mut(),
                })
            }
        }
    }
}

// Without this, an unsupported target fails with "cannot find module `imp`",
// repeated once per use site, which tells a porter nothing. PLAN.md section 1.1
// makes musl a permanent non-goal and section 8.8 fixes the supported matrix;
// this is where that decision becomes a compiler error instead of a surprise.
#[cfg(not(any(target_vendor = "apple", target_os = "linux", windows, miri)))]
compile_error!(
    "heapscope supports macOS/iOS, Linux (glibc), and Windows. \
     Other platforms need a `RawLock` backend that is allocation-free and \
     statically initializable -- see the notes on the BSDs in src/core/lock.rs \
     and the musl non-goal in PLAN.md section 1.1."
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    static SHARED: RawLock = RawLock::new();

    #[test]
    fn const_new_is_usable_from_a_static() {
        let guard = SHARED.lock();
        drop(guard);
    }

    #[test]
    fn try_lock_fails_while_held() {
        let lock = RawLock::new();
        let held = lock.lock();
        // Probing from another thread, because a same-thread `try_lock` on a
        // non-recursive lock is a contract violation on some platforms.
        std::thread::scope(|s| {
            s.spawn(|| {
                assert!(lock.try_lock().is_none());
            });
        });
        drop(held);
        assert!(lock.try_lock().is_some());
    }

    #[test]
    fn try_lock_for_times_out_rather_than_hanging() {
        let lock = RawLock::new();
        let held = lock.lock();
        std::thread::scope(|s| {
            s.spawn(|| {
                let start = Instant::now();
                assert!(lock.try_lock_for(Duration::from_millis(30)).is_none());
                assert!(start.elapsed() >= Duration::from_millis(25));
            });
        });
        drop(held);
    }

    #[test]
    fn try_lock_for_succeeds_once_released() {
        let lock = RawLock::new();
        std::thread::scope(|s| {
            let held = lock.lock();
            let waiter = s.spawn(|| lock.try_lock_for(Duration::from_secs(5)).is_some());
            std::thread::sleep(Duration::from_millis(20));
            drop(held);
            assert!(waiter.join().unwrap());
        });
    }

    #[test]
    fn mutual_exclusion_under_contention() {
        const THREADS: usize = 8;
        // Miri interprets every instruction, so the production iteration count
        // would turn one test into several minutes of CI. The interleavings
        // that matter show up in the first few hundred.
        #[cfg(miri)]
        const ITERS: usize = 200;
        #[cfg(not(miri))]
        const ITERS: usize = 20_000;

        static LOCK: RawLock = RawLock::new();
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        // Detects overlap that a plain counter would hide: any observer that
        // sees a non-zero value inside the critical section proves two threads
        // were inside at once.
        static INSIDE: AtomicUsize = AtomicUsize::new(0);
        static OVERLAPS: AtomicUsize = AtomicUsize::new(0);

        COUNTER.store(0, Ordering::Relaxed);
        INSIDE.store(0, Ordering::Relaxed);
        OVERLAPS.store(0, Ordering::Relaxed);

        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in 0..ITERS {
                        let _g = LOCK.lock();
                        if INSIDE.fetch_add(1, Ordering::Relaxed) != 0 {
                            OVERLAPS.fetch_add(1, Ordering::Relaxed);
                        }
                        COUNTER.store(COUNTER.load(Ordering::Relaxed) + 1, Ordering::Relaxed);
                        INSIDE.fetch_sub(1, Ordering::Relaxed);
                    }
                });
            }
        });

        assert_eq!(
            OVERLAPS.load(Ordering::Relaxed),
            0,
            "critical sections overlapped"
        );
        assert_eq!(COUNTER.load(Ordering::Relaxed), THREADS * ITERS);
    }

    #[test]
    fn force_reinit_releases_an_orphaned_lock() {
        let lock = RawLock::new();
        // Acquire on a thread that then exits, which is the shape a `fork`
        // leaves behind: an owner that no longer exists.
        std::thread::scope(|s| {
            s.spawn(|| {
                // SAFETY: released below by `force_reinit`, which is the exact
                // scenario under test.
                unsafe { lock.raw_lock() };
            });
        });
        assert!(lock.try_lock().is_none(), "precondition: lock is orphaned");
        // SAFETY: no other thread can observe the lock at this point.
        unsafe { lock.force_reinit() };
        assert!(
            lock.try_lock().is_some(),
            "force_reinit did not reset the lock"
        );
    }
}
