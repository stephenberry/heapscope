//! The peak gate: a reader-writer lock specialised for "shared is the common
//! case, exclusive is rare and short".
//!
//! # What it is for
//!
//! DHAT reports, per program point, the live bytes and blocks *at the instant
//! the whole process hit its peak*. Making that well-defined under concurrency
//! is the hardest correctness problem in this crate, and PLAN.md section 4.3
//! records what happens without it: over 400,000 modelled two-thread traces,
//! 0.6% violated the profiler's own stated invariant (`sum(pp.gb) > gmax`) and
//! 8.3% silently under-attributed. The cause is not a bug in the epoch trick —
//! it is that decoupling per-point updates from peak detection leaves no single
//! linearization point, so "the values at t-gmax" does not denote anything.
//!
//! The gate supplies that linearization point. Every event that changes live
//! bytes holds it **shared** for the whole of its update, and the moment that
//! defines a new peak holds it **exclusive**. Because the two modes exclude
//! each other, an epoch bump is totally ordered with respect to every update,
//! and the snapshot it implies is a state the program actually passed through.
//!
//! # Why not `std::sync::RwLock`
//!
//! Same reason as [`super::lock::RawLock`]: it can allocate, and this runs
//! inside the allocator. This implementation is const-initializable and
//! allocation-free on every path.
//!
//! # Shape
//!
//! A single word holds the reader count plus a writer flag, and a [`RawLock`]
//! serialises writers *and* gives blocked readers a real kernel wait to sleep
//! on instead of a spin.
//!
//! Readers acquire with a compare-exchange rather than a fetch-add. The
//! difference matters: a fetch-add would briefly raise the count even when the
//! writer flag was already set, and a steady stream of readers doing that could
//! keep a draining writer from ever observing zero. With a compare-exchange the
//! flag being set makes the exchange fail, so once a writer has flagged its
//! intent the count only ever falls.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::lock::RawLock;

/// Set while a writer holds, or is waiting to hold, the gate.
const WRITER: usize = 1 << (usize::BITS - 1);
/// Everything below [`WRITER`] is the count of active readers.
const READERS: usize = WRITER - 1;

/// Spins before yielding while draining readers.
///
/// Reader sections are a compare-exchange and a handful of counter updates,
/// with no syscall and no allocation, so they finish in tens of nanoseconds
/// unless the thread is preempted. Spinning briefly beats a syscall for the
/// common case; yielding afterwards keeps a preempted reader from turning the
/// wait into a burn.
const SPINS_BEFORE_YIELD: u32 = 64;

/// A reader-writer lock with no allocation and const initialization.
pub struct Gate {
    state: AtomicUsize,
    /// Held for the whole of a writer's tenure. Serialises writers against each
    /// other, and is what a blocked reader waits on.
    writer: RawLock,
}

impl Gate {
    /// Creates an unlocked gate.
    pub const fn new() -> Self {
        Self {
            state: AtomicUsize::new(0),
            writer: RawLock::new(),
        }
    }

    /// Acquires shared access, blocking while a writer holds the gate.
    #[inline]
    pub fn read(&self) -> ReadGuard<'_> {
        loop {
            let state = self.state.load(Ordering::Relaxed);
            if state & WRITER == 0 {
                if self
                    .state
                    .compare_exchange_weak(state, state + 1, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    return ReadGuard { gate: self };
                }
                // Either another reader won the race or a writer set the flag
                // between the load and the exchange. Both cases just retry.
                continue;
            }
            // A writer holds or wants the gate. Block on the writer's own lock
            // rather than spinning: this is a real kernel wait, so a reader
            // cannot burn a core while a writer is descheduled.
            drop(self.writer.lock());
        }
    }

    /// Acquires exclusive access, blocking until every reader has finished.
    #[inline]
    pub fn write(&self) -> WriteGuard<'_> {
        // Held for the whole exclusive section: it serialises writers and gives
        // readers something to sleep on.
        let held = self.writer.lock();

        // Announce intent. From here no new reader can enter, so the count is
        // monotonically non-increasing and the drain below terminates.
        self.state.fetch_or(WRITER, Ordering::Acquire);

        let mut spins = 0u32;
        while self.state.load(Ordering::Acquire) & READERS != 0 {
            if spins < SPINS_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }

        WriteGuard {
            gate: self,
            _held: held,
        }
    }

    /// Attempts to acquire exclusive access, giving up after `timeout`.
    ///
    /// The shutdown path uses this so that a wedged reader degrades the profile
    /// to partial output rather than hanging the process at `exit`.
    pub fn write_for(&self, timeout: Duration) -> Option<WriteGuard<'_>> {
        // `Instant::now() + timeout` *panics* on overflow, and this is the
        // shutdown path, where a panic is a process abort. `RawLock::try_lock_for`
        // documents this exact hazard; this had the bug it warns about.
        // `None` means "no representable deadline", i.e. wait indefinitely,
        // which is what a caller passing `Duration::MAX` asked for.
        let deadline = Instant::now().checked_add(timeout);
        // The deadline covers acquiring the writer lock *and* draining readers,
        // rather than allowing `timeout` for each, so the worst case is the
        // timeout the caller asked for and not twice it.
        let held = self.writer.try_lock_for(timeout)?;

        self.state.fetch_or(WRITER, Ordering::Acquire);

        while self.state.load(Ordering::Acquire) & READERS != 0 {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                // Withdraw the intent so readers are not blocked forever by a
                // writer that gave up.
                self.state.fetch_and(!WRITER, Ordering::Release);
                return None;
            }
            std::thread::yield_now();
        }

        Some(WriteGuard {
            gate: self,
            _held: held,
        })
    }

    /// Acquires the gate exclusively for a `fork` prepare handler, giving up
    /// after `timeout`.
    ///
    /// Returns whether the gate was acquired; the caller must release only what
    /// it got.
    ///
    /// # Why this is bounded
    ///
    /// A prepare handler runs *inside* `fork()`, so a handler that waits forever
    /// makes `fork` — and therefore `Command::spawn` — hang, in a caller with no
    /// timeout semantics and no way to find out why. The first version argued
    /// that a reader section is a few atomic operations so the wait is naturally
    /// short. That is true of a reader that is *running*; it says nothing about
    /// one stopped by a debugger, a `SIGSTOP`, or a CPU-throttled container, and
    /// those are precisely the cases the bound is for. Every other off-hot-path
    /// wait on this gate is bounded for the same reason.
    ///
    /// Giving up is safe. The child resets the gate unconditionally, so it is
    /// unaffected either way; what is lost is the guarantee that the child
    /// inherits tables no one was midway through updating, and the child does
    /// not read them.
    ///
    /// # Safety
    ///
    /// If this returns `true`, a matching [`Gate::unlock_for_fork`] must run on
    /// the same thread, or the child must reset the gate with
    /// [`Gate::reinit_after_fork`].
    #[must_use = "the caller must release only the locks it actually acquired"]
    pub unsafe fn lock_for_fork(&self, timeout: Duration) -> bool {
        // `checked_add` because `Instant::now() + Duration::MAX` panics, and
        // this runs where a panic is a process abort.
        let deadline = Instant::now().checked_add(timeout);
        // SAFETY: released by the caller's matching `unlock_for_fork`, or reset
        // by the child, per this function's contract.
        if !unsafe { self.writer.try_lock_for_raw(timeout) } {
            return false;
        }
        self.state.fetch_or(WRITER, Ordering::Acquire);

        while self.state.load(Ordering::Acquire) & READERS != 0 {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                // Withdraw the intent, or readers would be blocked forever by a
                // writer that gave up.
                self.state.fetch_and(!WRITER, Ordering::Release);
                // SAFETY: acquired immediately above.
                unsafe { self.writer.raw_unlock() };
                return false;
            }
            std::thread::yield_now();
        }
        true
    }

    /// Releases what [`Gate::lock_for_fork`] acquired.
    ///
    /// # Safety
    ///
    /// The calling thread must hold the gate through [`Gate::lock_for_fork`].
    pub unsafe fn unlock_for_fork(&self) {
        self.state.fetch_and(!WRITER, Ordering::Release);
        // SAFETY: delegated to the caller's obligation.
        unsafe { self.writer.raw_unlock() };
    }

    /// Resets the gate after a `fork`.
    ///
    /// # Safety
    ///
    /// Call only from a `pthread_atfork` child handler, where the process is
    /// single-threaded, so readers and writers recorded here belong to threads
    /// that no longer exist.
    pub unsafe fn reinit_after_fork(&self) {
        self.state.store(0, Ordering::Release);
        // SAFETY: delegated to the caller's single-threadedness obligation.
        unsafe { self.writer.force_reinit() }
    }
}

impl Default for Gate {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.load(Ordering::Relaxed);
        f.debug_struct("Gate")
            .field("readers", &(state & READERS))
            .field("writer", &(state & WRITER != 0))
            .finish()
    }
}

/// Proof of shared access to a [`Gate`].
#[must_use = "shared access is released immediately if the guard is not bound"]
#[derive(Debug)]
pub struct ReadGuard<'a> {
    gate: &'a Gate,
}

impl Drop for ReadGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        // `Release` pairs with the writer's `Acquire` drain load, so everything
        // this reader did is visible to the writer before it proceeds.
        self.gate.state.fetch_sub(1, Ordering::Release);
    }
}

/// Proof of exclusive access to a [`Gate`].
#[must_use = "exclusive access is released immediately if the guard is not bound"]
#[derive(Debug)]
pub struct WriteGuard<'a> {
    gate: &'a Gate,
    _held: super::lock::RawGuard<'a>,
}

impl Drop for WriteGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.gate.state.fetch_and(!WRITER, Ordering::Release);
        // `_held` is released after this, waking any reader blocked on it.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    #[test]
    fn readers_do_not_exclude_each_other() {
        static GATE: Gate = Gate::new();
        const READERS_N: usize = 8;

        let barrier = std::sync::Barrier::new(READERS_N);
        std::thread::scope(|s| {
            for _ in 0..READERS_N {
                let barrier = &barrier;
                s.spawn(move || {
                    let guard = GATE.read();
                    // Every reader is inside simultaneously. A lock that
                    // serialised them would deadlock here.
                    barrier.wait();
                    drop(guard);
                });
            }
        });
    }

    #[test]
    fn a_writer_excludes_readers() {
        let gate = Gate::new();
        let writing = AtomicBool::new(false);
        let violations = AtomicUsize::new(0);

        #[cfg(miri)]
        const ROUNDS: usize = 20;
        #[cfg(not(miri))]
        const ROUNDS: usize = 2_000;

        std::thread::scope(|s| {
            for _ in 0..4 {
                let (gate, writing, violations) = (&gate, &writing, &violations);
                s.spawn(move || {
                    for _ in 0..ROUNDS {
                        let guard = gate.read();
                        if writing.load(Ordering::Acquire) {
                            violations.fetch_add(1, Ordering::Relaxed);
                        }
                        drop(guard);
                    }
                });
            }
            for _ in 0..2 {
                let (gate, writing, violations) = (&gate, &writing, &violations);
                s.spawn(move || {
                    for _ in 0..ROUNDS {
                        let guard = gate.write();
                        if writing.swap(true, Ordering::AcqRel) {
                            violations.fetch_add(1, Ordering::Relaxed);
                        }
                        writing.store(false, Ordering::Release);
                        drop(guard);
                    }
                });
            }
        });

        assert_eq!(
            violations.load(Ordering::Relaxed),
            0,
            "a reader and a writer, or two writers, were inside at once"
        );
    }

    /// The property the gate exists to provide: a writer observes a state that
    /// no reader is midway through changing.
    ///
    /// Each reader makes a two-step update that is inconsistent in between. If
    /// the gate works, a writer can never observe the halfway state.
    #[test]
    fn a_writer_never_observes_a_half_finished_reader_update() {
        static GATE: Gate = Gate::new();
        static LEFT: AtomicUsize = AtomicUsize::new(0);
        static RIGHT: AtomicUsize = AtomicUsize::new(0);

        #[cfg(miri)]
        const ROUNDS: usize = 20;
        #[cfg(not(miri))]
        const ROUNDS: usize = 5_000;

        LEFT.store(0, Ordering::Relaxed);
        RIGHT.store(0, Ordering::Relaxed);
        let torn = AtomicUsize::new(0);

        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..ROUNDS {
                        let guard = GATE.read();
                        LEFT.fetch_add(1, Ordering::Relaxed);
                        std::hint::spin_loop();
                        RIGHT.fetch_add(1, Ordering::Relaxed);
                        drop(guard);
                    }
                });
            }
            let torn = &torn;
            s.spawn(move || {
                for _ in 0..ROUNDS {
                    let guard = GATE.write();
                    if LEFT.load(Ordering::Relaxed) != RIGHT.load(Ordering::Relaxed) {
                        torn.fetch_add(1, Ordering::Relaxed);
                    }
                    drop(guard);
                }
            });
        });

        assert_eq!(
            torn.load(Ordering::Relaxed),
            0,
            "the gate let a writer see a reader's update half-applied; \
             this is exactly the condition that makes t-gmax undefined"
        );
    }

    /// `Duration::MAX` means "wait as long as it takes", not "abort the
    /// process". An unrepresentable deadline must degrade, not panic.
    #[test]
    fn write_for_with_an_unrepresentable_deadline_does_not_panic() {
        let gate = Gate::new();
        let guard = gate
            .write_for(Duration::MAX)
            .expect("an uncontended gate should be acquired");
        drop(guard);
    }

    #[test]
    fn write_for_times_out_rather_than_hanging() {
        let gate = Gate::new();
        let release = std::sync::Barrier::new(2);

        std::thread::scope(|s| {
            s.spawn(|| {
                let _reader = gate.read();
                release.wait(); // hold the read lock until the writer has given up
            });

            // Give the reader a moment to acquire.
            std::thread::sleep(Duration::from_millis(10));
            let start = Instant::now();
            assert!(
                gate.write_for(Duration::from_millis(50)).is_none(),
                "write_for should have timed out against a held read lock"
            );
            assert!(start.elapsed() >= Duration::from_millis(40));
            release.wait();
        });
    }

    /// A writer that gives up must not leave the gate closed against readers.
    #[test]
    fn a_timed_out_writer_withdraws_its_intent() {
        let gate = Gate::new();
        let release = std::sync::Barrier::new(2);

        std::thread::scope(|s| {
            s.spawn(|| {
                let _reader = gate.read();
                release.wait();
            });

            std::thread::sleep(Duration::from_millis(10));
            assert!(gate.write_for(Duration::from_millis(30)).is_none());
            release.wait();
        });

        // With the reader gone, both modes must be available again.
        drop(gate.read());
        drop(gate.write());
    }

    #[test]
    fn writers_are_serialised() {
        static GATE: Gate = Gate::new();
        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        #[cfg(miri)]
        const ROUNDS: usize = 50;
        #[cfg(not(miri))]
        const ROUNDS: usize = 10_000;
        const THREADS: usize = 4;

        COUNTER.store(0, Ordering::Relaxed);
        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in 0..ROUNDS {
                        let _guard = GATE.write();
                        // A non-atomic read-modify-write, which only produces
                        // the right total if the writes never overlap.
                        let value = COUNTER.load(Ordering::Relaxed);
                        COUNTER.store(value + 1, Ordering::Relaxed);
                    }
                });
            }
        });

        assert_eq!(COUNTER.load(Ordering::Relaxed), THREADS * ROUNDS);
    }
}
