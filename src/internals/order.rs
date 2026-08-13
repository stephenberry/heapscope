//! Lock-order enforcement.
//!
//! # Why this is not optional
//!
//! The profiler holds locks from several families on one code path, and PLAN.md
//! section 4.2 fixes a global order between them. Without enforcement, a single
//! misordered acquisition is a latent deadlock that shows up under load, on
//! someone else's machine, months later.
//!
//! On the primary development platform it is worse than that. `os_unfair_lock`
//! detects a same-thread reacquire and **kills the process with `SIGKILL`** —
//! no message, no core, no stack. An order violation on macOS is therefore not
//! a hang you can attach a debugger to; it is an instant, unattributable death.
//! That is the specific failure this module exists to convert into a sentence.
//!
//! # The order
//!
//! Locks must be acquired in strictly increasing [`Level`]. The levels are
//! chosen so that the real code paths are naturally ordered:
//!
//! - `alloc` records counters under the gate, then releases it before inserting
//!   into the live-block table.
//! - `dealloc` removes the live-block entry first, releases that shard, and only
//!   then takes the gate to apply the counters.
//!
//! Because neither path holds a live-block shard *across* the gate, the two
//! families never nest, and the order below is deliberately permissive about
//! them rather than pretending to a discipline the code does not need.
//!
//! # Cost
//!
//! Enforcement compiles to nothing outside `debug_assertions`. Release builds
//! carry no table, no atomics, and no branch.
//!
//! # It reports; it does not panic
//!
//! A violation is reported through [`super::diagnostic`] and counted, never
//! panicked. This code runs inside the allocator shim, where building a panic
//! message allocates and re-enters, and where unwinding out of a `GlobalAlloc`
//! method is undefined — so an assertion here would corrupt exactly the
//! situation it was trying to diagnose.

use std::fmt;

/// Lock families, in the order they must be acquired.
///
/// Discriminants are explicit because their *relative* values are the contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Level {
    /// A live-block table shard, keyed by pointer.
    LiveBlockShard = 1,
    /// The global peak gate, in either mode.
    PeakGate = 2,
    /// A program-point shard, keyed by program-point id.
    ProgramPointShard = 3,
    /// The region table, taken while a region name is interned.
    ///
    /// Never reached from the allocator path: what that path reads is an id
    /// already interned, sitting in a guard slot. It sits above the shards and
    /// below the arena because interning a name allocates one and nothing else.
    RegionTable = 4,
    /// The arena's own lock, reached when a table grows or a program point is
    /// interned. Deepest, because every other family can need to allocate.
    Arena = 5,
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Level::LiveBlockShard => "live-block shard",
            Level::PeakGate => "peak gate",
            Level::ProgramPointShard => "program-point shard",
            Level::RegionTable => "region table",
            Level::Arena => "arena",
        };
        f.write_str(name)
    }
}

/// Proof that a level was entered. Leaves it on drop.
#[must_use = "the level is left immediately if the token is not bound"]
#[derive(Debug)]
pub struct Entered {
    #[cfg(debug_assertions)]
    level: Level,
    #[cfg(not(debug_assertions))]
    _private: (),
}

impl Drop for Entered {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        checker::leave(self.level);
    }
}

/// Records that the calling thread is acquiring a lock at `level`.
///
/// Reports a violation if a lock at the same or a deeper level is already held.
#[inline(always)]
pub fn enter(level: Level) -> Entered {
    #[cfg(debug_assertions)]
    {
        checker::enter(level);
        Entered { level }
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = level;
        Entered { _private: () }
    }
}

/// Forgets the held sets of threads that did not survive a `fork`.
///
/// Without this a slot left dirty by a thread that died holding a lock makes the
/// checker report a violation the first time a thread in the child hashes to
/// that slot — a diagnostic line about a lock order that was never violated.
///
/// # Safety
///
/// Call only from a `pthread_atfork` child handler, where the process is
/// single-threaded.
pub unsafe fn reinit_after_fork() {
    #[cfg(debug_assertions)]
    checker::reinit_after_fork();
}

/// Violations observed since the process started.
///
/// Always available, so a release build can still be asked whether anything was
/// seen — it just never observes anything, because nothing is checked.
pub fn violations() -> u64 {
    #[cfg(debug_assertions)]
    {
        checker::violations()
    }
    #[cfg(not(debug_assertions))]
    {
        0
    }
}

#[cfg(debug_assertions)]
mod checker {
    use super::Level;
    use crate::internals::diagnostic;
    use crate::internals::guard;
    use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

    /// Per-thread slots holding a bitmask of currently held levels.
    ///
    /// Keyed by the same allocation-free thread handle the reentrancy guard
    /// uses, for the same reason: thread-local storage is not reachable from
    /// inside the allocator on every platform.
    const SLOTS: usize = 512;
    const SLOT_MASK: usize = SLOTS - 1;

    struct Slot {
        owner: AtomicUsize,
        /// Bit `n` set means a lock at level `n` is held by `owner`.
        held: AtomicU32,
    }

    impl Slot {
        const fn new() -> Self {
            Self {
                owner: AtomicUsize::new(0),
                held: AtomicU32::new(0),
            }
        }
    }

    static TABLE: [Slot; SLOTS] = {
        #[allow(clippy::declare_interior_mutable_const)]
        const INIT: Slot = Slot::new();
        [INIT; SLOTS]
    };

    static VIOLATIONS: AtomicU64 = AtomicU64::new(0);

    fn slot_for(handle: usize) -> Option<&'static Slot> {
        // Fibonacci hashing needs a constant that fits the word: the 64-bit value
        // is not merely wrong on a 32-bit target, it does not compile there.
        #[cfg(target_pointer_width = "64")]
        const GOLDEN: usize = 0x9E37_79B9_7F4A_7C15;
        #[cfg(not(target_pointer_width = "64"))]
        const GOLDEN: usize = 0x9E37_79B9;
        let start = (handle.wrapping_mul(GOLDEN) >> (usize::BITS / 2)) & SLOT_MASK;
        for probe in 0..32 {
            let slot = &TABLE[(start + probe) & SLOT_MASK];
            let owner = slot.owner.load(Ordering::Acquire);
            if owner == handle {
                return Some(slot);
            }
            if owner == 0
                && slot
                    .owner
                    .compare_exchange(0, handle, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                slot.held.store(0, Ordering::Relaxed);
                return Some(slot);
            }
            // Either another thread owns this slot, or it won the race to claim
            // it. Both mean: keep probing.
        }
        // A checker that cannot find a slot silently checks nothing. That is the
        // right failure: it is a diagnostic, and it must never be the reason a
        // program stops working.
        None
    }

    pub(super) fn enter(level: Level) {
        let Some(slot) = slot_for(guard::thread_handle()) else {
            return;
        };
        let held = slot.held.load(Ordering::Relaxed);
        let bit = 1u32 << (level as u8);

        // Any bit at or above this level means an out-of-order acquisition.
        let at_or_deeper = held & !(bit - 1);
        if at_or_deeper != 0 {
            VIOLATIONS.fetch_add(1, Ordering::Relaxed);
            diagnostic::report(violation_message(level, at_or_deeper));
        }
        slot.held.store(held | bit, Ordering::Relaxed);
    }

    pub(super) fn leave(level: Level) {
        let Some(slot) = slot_for(guard::thread_handle()) else {
            return;
        };
        let bit = 1u32 << (level as u8);
        let held = slot.held.load(Ordering::Relaxed);
        slot.held.store(held & !bit, Ordering::Relaxed);
    }

    pub(super) fn violations() -> u64 {
        VIOLATIONS.load(Ordering::Relaxed)
    }

    /// Builds a message without allocating or formatting.
    ///
    /// A `format!` here would allocate, from a path the allocator shim reaches.
    /// The set of possible messages is small and known, so they are literals.
    fn violation_message(acquiring: Level, held: u32) -> &'static str {
        let deepest = (31 - held.leading_zeros()) as u8;
        match (acquiring as u8, deepest) {
            (a, h) if a == h => match acquiring {
                Level::LiveBlockShard => "lock order: re-entering a live-block shard",
                Level::PeakGate => "lock order: re-entering the peak gate",
                Level::ProgramPointShard => "lock order: re-entering a program-point shard",
                Level::RegionTable => "lock order: re-entering the region table",
                Level::Arena => "lock order: re-entering the arena",
            },
            _ => match acquiring {
                Level::LiveBlockShard => {
                    "lock order violation: taking a live-block shard while holding a deeper lock"
                }
                Level::PeakGate => {
                    "lock order violation: taking the peak gate while holding a deeper lock"
                }
                Level::ProgramPointShard => {
                    "lock order violation: taking a program-point shard while holding the arena"
                }
                Level::RegionTable => {
                    "lock order violation: taking the region table while holding the arena"
                }
                Level::Arena => "lock order violation: taking the arena out of order",
            },
        }
    }

    /// Clears the held set of every slot except the calling thread's.
    ///
    /// Ownership is left in place, for the reason spelled out in
    /// [`crate::internals::guard::reinit_after_fork`]: this table is open-addressed
    /// too, and releasing a slot punches a gap into a probe sequence that
    /// silently migrates a surviving thread to a different slot.
    pub(super) fn reinit_after_fork() {
        let survivor = guard::thread_handle();
        for slot in &TABLE {
            let owner = slot.owner.load(Ordering::Relaxed);
            if owner == 0 || owner == survivor {
                continue;
            }
            slot.held.store(0, Ordering::Relaxed);
        }
    }

    /// Clears this thread's held set. For tests, which deliberately provoke
    /// violations and must not leave the state dirty for the next one.
    #[cfg(test)]
    pub(super) fn reset_current_thread() {
        if let Some(slot) = slot_for(guard::thread_handle()) {
            slot.held.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;

    /// Serialised because the violation counter is global.
    static SERIALIZE: super::super::lock::RawLock = super::super::lock::RawLock::new();

    /// Holds the serializing lock and suppresses diagnostic output for the
    /// duration of a test.
    ///
    /// These tests provoke violations on purpose; printing them would make a
    /// clean run look like a broken one.
    struct Quiet {
        /// Held for its `Drop`, which releases the serializing lock.
        _guard: super::super::lock::RawGuard<'static>,
    }

    impl Quiet {
        fn new() -> Self {
            let guard = SERIALIZE.lock();
            crate::internals::diagnostic::set_quiet(true);
            checker::reset_current_thread();
            Quiet { _guard: guard }
        }
    }

    impl Drop for Quiet {
        fn drop(&mut self) {
            crate::internals::diagnostic::set_quiet(false);
            checker::reset_current_thread();
        }
    }

    #[test]
    fn increasing_order_is_accepted() {
        let _quiet = Quiet::new();
        let before = violations();

        let gate = enter(Level::PeakGate);
        let shard = enter(Level::ProgramPointShard);
        let arena = enter(Level::Arena);
        drop(arena);
        drop(shard);
        drop(gate);

        assert_eq!(violations(), before, "a correctly ordered path was flagged");
    }

    #[test]
    fn decreasing_order_is_reported() {
        let _quiet = Quiet::new();
        let before = violations();

        let arena = enter(Level::Arena);
        // Taking the gate while holding the arena is the deadlock shape.
        let gate = enter(Level::PeakGate);
        drop(gate);
        drop(arena);

        assert_eq!(
            violations(),
            before + 1,
            "an inverted acquisition was not reported"
        );
        checker::reset_current_thread();
    }

    #[test]
    fn same_level_reentry_is_reported() {
        let _quiet = Quiet::new();
        let before = violations();

        let first = enter(Level::ProgramPointShard);
        // On Apple this is the SIGKILL case; the checker must catch it first.
        let second = enter(Level::ProgramPointShard);
        drop(second);
        drop(first);

        assert_eq!(
            violations(),
            before + 1,
            "same-level reacquisition was not reported"
        );
        checker::reset_current_thread();
    }

    #[test]
    fn levels_are_released_on_drop() {
        let _quiet = Quiet::new();
        let before = violations();

        // Sequential, non-overlapping acquisitions in decreasing order are
        // fine: nothing is held across them.
        for _ in 0..10 {
            drop(enter(Level::Arena));
            drop(enter(Level::PeakGate));
            drop(enter(Level::LiveBlockShard));
        }

        assert_eq!(
            violations(),
            before,
            "released levels were still considered held"
        );
    }

    #[test]
    fn threads_are_tracked_independently() {
        let _quiet = Quiet::new();
        let before = violations();

        let outer = enter(Level::Arena);
        // Another thread holding nothing must be free to take a shallower lock
        // even though this thread holds the deepest one.
        std::thread::scope(|s| {
            s.spawn(|| {
                let gate = enter(Level::PeakGate);
                drop(gate);
            });
        });
        drop(outer);

        assert_eq!(violations(), before, "one thread's locks blocked another's");
    }
}
