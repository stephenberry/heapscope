//! Phases the program names itself.
//!
//! A stack trace answers "where", and a profile of a program with phases is
//! usually a question about "when": the same `Vec::with_capacity` in the same
//! helper is a different cost during parsing than during code generation, and
//! nothing in the stack distinguishes them.
//!
//! ```no_run
//! # fn parse() {}
//! let _profiler = heapscope::Profiler::builder().build().unwrap();
//! {
//!     let _region = heapscope::region("parsing");
//!     parse();
//! } // "parsing" ends here
//! ```
//!
//! # Regions are per thread, and the innermost one wins
//!
//! Entering a region affects **the calling thread only**. A worker pool
//! allocating while the main thread is inside `parsing` records those
//! allocations against no region, and that is the honest answer: the worker was
//! not in the phase, it was running alongside it. The alternative — a
//! process-wide current phase — attributes whatever a background thread happens
//! to be doing to whichever phase some other thread is in, which is a number
//! that looks meaningful and is not.
//!
//! Nesting works to any depth and costs nothing: each guard remembers the
//! region it displaced, on the program's own stack, and puts it back on drop.
//! An allocation is attributed to the innermost open region and to that one
//! only, so the rows in a profile partition the run rather than double-counting
//! it. An outer region does **not** include what its inner regions recorded,
//! because a name can be entered under different parents at different times and
//! a tree built from that would be a shape the run never had.
//!
//! # Cost when nothing is profiling
//!
//! Two atomic loads and a branch — the run state and the poison flag, which is
//! what `Engine::is_running` reads. [`region`] then returns an inert guard
//! whose drop does nothing, so instrumentation can be left in place, which is
//! the point of instrumentation.

use std::marker::PhantomData;

use crate::internals::guard;
use crate::internals::site::RegionId;

/// Opens a named region on the calling thread until the returned guard drops.
///
/// Every allocation this thread makes in the meantime is attributed to `name`,
/// unless a nested region is opened inside it. The profile reports what each
/// region allocated, what it left live, and how much it held at its own peak.
///
/// Names are interned: entering `"parsing"` a thousand times gives one row
/// whose `entries` says a thousand, not a thousand rows. They are cut to 64
/// bytes, and two names that agree that far agree entirely as far as the
/// profile is concerned.
///
/// # Example
///
/// ```no_run
/// # fn load() -> Vec<u8> { Vec::new() }
/// # fn parse(_: &[u8]) {}
/// let _profiler = heapscope::Profiler::builder().build().unwrap();
///
/// let bytes = {
///     let _region = heapscope::region("loading");
///     load()
/// };
/// {
///     let _region = heapscope::region("parsing");
///     parse(&bytes);
/// }
/// ```
///
/// # Binding the guard is not optional
///
/// `heapscope::region("parsing");` — with a semicolon and no binding — opens
/// and closes the region on one line, which is why this is `#[must_use]`. The
/// same is true of `let _ = heapscope::region("parsing")`, which no attribute
/// can catch: `_` drops immediately, where `_region` lives to the end of the
/// scope.
#[must_use = "the region ends when the guard is dropped; bind it with `let _region = ...`"]
pub fn region(name: &str) -> Region {
    let engine = crate::engine();
    // Checked first, so that instrumentation left in a program nobody is
    // profiling costs one acquire load and a predictable branch.
    if !engine.is_running() {
        return Region::inert();
    }

    // Interning takes a lock and touches the arena. `None` means this thread is
    // already inside the profiler — a `Drop` running under the shim, or a signal
    // handler that interrupted one — where taking that lock could deadlock
    // against the thread's own outer acquisition. It also means no slot could
    // be had, which is the other reason attribution is impossible here.
    let Some(guard) = guard::enter() else {
        return Region::inert();
    };
    let id = engine.intern_region(name);
    // The guard is the proof that this thread has a slot to write the region
    // into, which is why it is passed rather than merely held.
    let previous = guard::enter_region(&guard, id);
    engine.regions().enter(id);

    Region {
        id,
        previous: Some(previous),
        _not_send: PhantomData,
    }
}

/// An open region. Closes when dropped.
///
/// Not [`Send`]: the region it displaced is restored on the thread that opened
/// it, because that is the only thread whose attribution it changed. Moving one
/// across threads would restore the wrong predecessor on the wrong thread, and
/// the type system is a better place to say so than the documentation.
#[derive(Debug)]
pub struct Region {
    id: RegionId,
    /// The region this one displaced, or `None` for a guard that never opened
    /// anything and therefore has nothing to close.
    previous: Option<RegionId>,
    _not_send: PhantomData<*const ()>,
}

impl Region {
    /// A guard over a region that was never opened.
    fn inert() -> Region {
        Region {
            id: RegionId::NONE,
            previous: None,
            _not_send: PhantomData,
        }
    }

    /// Whether this guard actually opened a region.
    ///
    /// False in exactly two cases: no profiler was running, or the call came
    /// from a thread the profiler could not enter — one already inside the
    /// shim, or one the guard table had no slot for. Exposed because "my region
    /// has no rows" is otherwise indistinguishable from "my region allocated
    /// nothing".
    pub fn is_open(&self) -> bool {
        self.previous.is_some()
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        let Some(previous) = self.previous else {
            return;
        };
        // Neither call takes a lock or allocates, which is what makes this safe
        // to run from a `Drop` reached inside the allocator shim. The region
        // word is restored first: until it is, allocations made by this
        // thread's own teardown would still land in a region that has ended.
        guard::leave_region(previous);
        crate::engine().regions().leave(self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Instrumentation left in a program nobody is profiling must be inert, and
    /// must say so rather than silently recording into nothing.
    #[test]
    fn a_region_outside_a_run_opens_nothing() {
        let region = region("parsing");
        assert!(!region.is_open());
        assert!(
            crate::engine().regions().is_empty(),
            "a region entered with no profiler running interned a row"
        );
    }

    /// The guard restores what it displaced rather than clearing the slot, or
    /// leaving a nested region would end its parent too.
    #[test]
    fn nesting_restores_the_enclosing_region() {
        let outer = RegionId::from_u16(1);
        let inner = RegionId::from_u16(2);
        let held = guard::enter().expect("this thread is not inside the profiler");

        let before = guard::enter_region(&held, outer);
        let displaced = guard::enter_region(&held, inner);
        assert_eq!(displaced, outer);

        guard::leave_region(displaced);
        let now = guard::enter_region(&held, before);
        assert_eq!(
            now, outer,
            "leaving the inner region did not restore the outer one"
        );
    }

    /// Restoring is a store to a slot this thread already owns, so it must not
    /// go looking for one — and on a thread that has none, it must do nothing
    /// rather than claim one from inside a destructor.
    #[test]
    fn leaving_a_region_on_a_thread_with_no_slot_does_nothing() {
        // A thread that has never entered the guard has no slot. `leave_region`
        // is the only entry point that can be reached in that state, from a
        // `Region` that outlived the thread-local reclaiming its slot.
        std::thread::spawn(|| {
            guard::leave_region(RegionId::from_u16(3));
        })
        .join()
        .expect("restoring a region on a slotless thread panicked");
    }
}
