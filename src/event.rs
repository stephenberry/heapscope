//! Events the program reports about itself.
//!
//! A `GlobalAlloc` shim sees allocations and nothing else. The two functions
//! here let a program hand the profiler something it could not have observed —
//! a weighted occurrence of any kind, or a byte count it copied — and get back
//! the same thing a heap profile gives: the call sites, ranked, with the
//! machinery frames trimmed away.
//!
//! # The mode decides which one records
//!
//! DHAT files carry one `mode`, and the viewer labels every column from it.
//! Summing heap blocks and ad hoc weights into a single `tb` would produce a
//! number with no unit, so a run counts exactly one kind of thing:
//!
//! | Mode | Records | Shim |
//! |---|---|---|
//! | [`Mode::Heap`] | every allocation | on |
//! | [`Mode::AdHoc`] | [`event`] | off |
//! | [`Mode::Copy`] | [`copied`] | off |
//!
//! In either non-heap mode the shim records nothing: an allocation costs the
//! reentrancy guard, one acquire load and one relaxed one, and then goes
//! straight to the inner allocator.
//!
//! # Calling one in the wrong mode
//!
//! Nothing happens, and the profile says how often. A no-op is the only
//! defensible behaviour — instrumentation is meant to be left in place, and a
//! profiler that panics or prints from inside a hot loop is worse than the
//! measurement is worth — but a *silent* no-op turns "I chose the wrong mode"
//! into "my profile is empty and I have no idea why". So the count lands in the
//! profile as `heapscope.refusedEvents` and in the text summary.
//!
//! Calling either with no profiler running is not an error and is not counted:
//! that is the ordinary state of instrumented code in production.

use crate::internals::engine::Mode;
use crate::internals::guard;
use crate::CAPTURE_DEPTH;

/// Records one ad hoc event of `weight` at the caller's call site.
///
/// Only counts during a run built with [`Mode::AdHoc`]; see the module
/// documentation for what happens otherwise.
///
/// The weight means whatever the program says it means — cache misses, bytes
/// sent, rows parsed, retries — and the profile reports the summed weight and
/// the count per call site. DHAT's `tb` and `tbk` carry them, and the viewer
/// labels them `units` and `events` rather than bytes and blocks.
///
/// # Example
///
/// ```no_run
/// # fn parse_row(_: &str) -> usize { 0 }
/// # let rows: Vec<&str> = Vec::new();
/// let profiler = heapscope::Profiler::builder()
///     .mode(heapscope::Mode::AdHoc)
///     .build()
///     .unwrap();
///
/// for row in &rows {
///     heapscope::event(parse_row(row) as u64);
/// }
/// ```
#[inline(never)]
pub fn event(weight: u64) {
    record(Mode::AdHoc, weight);
}

/// Records `bytes` copied at the caller's call site.
///
/// Only counts during a run built with [`Mode::Copy`]; see the module
/// documentation for what happens otherwise.
///
/// Unlike Valgrind's copy mode, which instruments every instruction and so sees
/// `memcpy` itself, this counts what the program says it copied. That is a
/// narrower measurement and a much cheaper one, and it is the only one available
/// to a profiler that works by wrapping the allocator.
#[inline(never)]
pub fn copied(bytes: usize) {
    record(Mode::Copy, bytes as u64);
}

/// The body of both, inlined into them.
///
/// `#[inline(always)]` for the reason [`crate::alloc::capture`] is: the
/// calibrated skip leaves the caller of the capturing function's caller as the
/// innermost frame, and it is [`event`] and [`copied`] that have to stand in
/// that position. A frame of its own here would put `heapscope::event` at the
/// top of every program point in an ad hoc profile.
#[inline(always)]
fn record(required: Mode, weight: u64) {
    let engine = crate::engine();
    // Checked before anything else, so instrumentation left in a program that
    // is not being profiled costs one acquire load, one relaxed one, and a
    // branch that predicts perfectly.
    if !engine.is_running() {
        return;
    }
    if engine.mode() != required {
        engine.refuse_event();
        return;
    }
    // `None` means this thread is already inside the profiler — a call from a
    // `Drop` impl running under the shim, or from a signal handler that
    // interrupted one. Recording from there would reenter the tables this
    // thread is already inside.
    let Some(guard) = guard::enter() else {
        return;
    };
    let mut frames = [0usize; CAPTURE_DEPTH];
    let len = crate::alloc::capture(&guard, &mut frames);
    // The guard is passed on rather than merely still in scope: `record_event`
    // reaches the peak gate, and holding the guard across that is what keeps a
    // signal handler from re-entering it. See its documentation.
    engine.record_event(&guard, weight, &frames[..len]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Instrumentation left in a program nobody is profiling must be inert.
    ///
    /// The engine in a unit-test binary is idle unless a test claims it, and
    /// exactly one test in this crate may — so this is the state every other
    /// test sees, and the one production code sees most of the time.
    #[test]
    fn events_are_inert_when_no_profiler_is_running() {
        let before = crate::engine().stats();
        event(1_000);
        copied(2_000);
        let after = crate::engine().stats();

        assert_eq!(
            after.total_bytes, before.total_bytes,
            "an event was recorded with no profiler running"
        );
        assert_eq!(
            after.refused_events, before.refused_events,
            "an event outside a run was counted as refused; that is the normal \
             state of instrumented code, not a mistake to report"
        );
    }

    #[test]
    fn a_mode_names_itself_the_way_the_viewer_expects() {
        // These strings are the file format, not a label: `dh_view.js` reads
        // `mode` and `verb` straight into the page.
        assert_eq!(Mode::Heap.as_str(), "heap");
        assert_eq!(Mode::AdHoc.as_str(), "ad-hoc");
        assert_eq!(Mode::Copy.as_str(), "copy");
        assert_eq!(Mode::Heap.verb(), "Allocated");
        assert_eq!(Mode::AdHoc.verb(), "Occurred");
        assert_eq!(Mode::Copy.verb(), "Copied");
    }

    /// Only the heap mode has block lifetimes, and only ad hoc renames the
    /// units. Both decide which fields a profile may carry.
    #[test]
    fn only_the_heap_mode_records_block_lifetimes() {
        assert!(Mode::Heap.block_lifetimes());
        assert!(!Mode::AdHoc.block_lifetimes());
        assert!(!Mode::Copy.block_lifetimes());

        assert!(Mode::Heap.records_allocations());
        assert!(!Mode::AdHoc.records_allocations());
        assert!(!Mode::Copy.records_allocations());

        assert_eq!(Mode::Heap.units(), Mode::DEFAULT_UNITS);
        assert_eq!(
            Mode::Copy.units(),
            Mode::DEFAULT_UNITS,
            "copy mode really does count bytes"
        );
        assert_eq!(Mode::AdHoc.units(), ("unit", "units", "events"));

        assert!(Mode::Heap.counts_bytes());
        assert!(Mode::Copy.counts_bytes());
        assert!(
            !Mode::AdHoc.counts_bytes(),
            "an ad hoc weight rendered in binary units reports 1,024 retries \
             as one kibibyte"
        );
    }
}
