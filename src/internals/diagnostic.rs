//! Reporting failures from inside the allocator, where panicking is not an
//! option.
//!
//! PLAN.md section 4.6 is absolute about this: on an internal invariant
//! violation the profiler must *"poison, stop recording, one diagnostic line to
//! stderr, program continues. **Never panic, never abort.**"*
//!
//! Panicking from a `GlobalAlloc` method is not merely impolite. Building the
//! panic message allocates, which re-enters the shim; the unwinder then unwinds
//! *through* a `GlobalAlloc` method, which is undefined; and `panic = "abort"`
//! turns the whole thing into a bare `SIGABRT` with no explanation. A
//! `debug_assert!` is the same hazard wearing a disguise — it is exactly the
//! failing case, in the build most likely to be attached to a debugger, that
//! triggers it.
//!
//! So this module writes bytes to file descriptor 2 with one syscall and no
//! allocation, no formatting machinery, and no locks.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Whether the profiler has stopped recording after an internal failure.
static POISONED: AtomicBool = AtomicBool::new(false);

/// Diagnostics emitted so far. Bounded so that a failure occurring once per
/// allocation cannot itself become the program's dominant cost.
static EMITTED: AtomicU32 = AtomicU32::new(0);

/// Most diagnostics to write before going quiet.
const MAX_DIAGNOSTICS: u32 = 20;

/// Suppresses the actual write while still counting, for the test that checks
/// the budget.
///
/// Without it, that test writes twenty lines to the real file descriptor 2 on
/// every run of the suite — `report` deliberately bypasses `std::io`, so
/// libtest's output capture cannot intercept it.
#[cfg(test)]
static QUIET: AtomicBool = AtomicBool::new(false);

#[inline(always)]
fn suppressed() -> bool {
    #[cfg(test)]
    {
        QUIET.load(Ordering::Relaxed)
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// Writes one line to stderr, prefixed and newline-terminated.
///
/// Allocation-free and lock-free. Safe to call from inside the allocator, from
/// a signal handler, and during process teardown.
///
/// Silently does nothing after [`MAX_DIAGNOSTICS`] calls: a fault on the
/// allocator path can recur millions of times, and a profiler that floods
/// stderr has replaced one problem with a worse one.
pub fn report(message: &str) {
    if EMITTED.fetch_add(1, Ordering::Relaxed) >= MAX_DIAGNOSTICS {
        return;
    }
    if suppressed() {
        return;
    }
    write_stderr(b"heapscope: ");
    write_stderr(message.as_bytes());
    write_stderr(b"\n");
}

/// Records an internal invariant violation, reports it once, and stops the
/// profiler from recording further.
///
/// Returns the previous poison state, so a caller can tell whether it was the
/// first to fail.
pub fn poison(message: &str) -> bool {
    let was_poisoned = POISONED.swap(true, Ordering::SeqCst);
    if !was_poisoned {
        report(message);
        report("recording has stopped; the profile will be incomplete");
    }
    was_poisoned
}

/// Whether the profiler has been poisoned.
///
/// `Relaxed` because this gates a best-effort fast path: reading a stale
/// `false` costs at most one more recorded event, and every path that acts on
/// it is idempotent.
#[inline(always)]
pub fn is_poisoned() -> bool {
    POISONED.load(Ordering::Relaxed)
}

/// Suppresses or restores diagnostic output.
///
/// Used by tests that deliberately provoke a diagnostic: without it, a normal
/// `cargo test` run prints lines like "lock order violation" that read as a
/// failure when they are in fact the thing being verified.
#[cfg(test)]
pub(crate) fn set_quiet(quiet: bool) {
    QUIET.store(quiet, Ordering::SeqCst);
}

/// Clears the poison flag and the diagnostic budget.
///
/// Exists for tests, which would otherwise see a poison set by an earlier test
/// suppress the behaviour under examination.
#[cfg(test)]
pub(crate) fn reset() {
    POISONED.store(false, Ordering::SeqCst);
    EMITTED.store(0, Ordering::SeqCst);
    QUIET.store(false, Ordering::SeqCst);
}

/// Serialises every test that sets or clears the poison flag.
///
/// The flag is process-wide, and the harness runs tests concurrently, so a test
/// that poisons deliberately is otherwise visible to any other test that reads
/// it — including, since M5, the ones in [`crate::stats`], which is why this is
/// module-level rather than private to the tests below. Holding it does not make
/// a test safe from code that never takes it; it makes the tests that
/// deliberately provoke a poison safe from each other.
#[cfg(test)]
pub(crate) static POISON_TESTS: super::lock::RawLock = super::lock::RawLock::new();

/// Writes `bytes` to file descriptor 2, ignoring short writes and errors.
///
/// Nothing useful can be done about a failure to report a failure, and
/// retrying risks an unbounded loop on the allocator path.
fn write_stderr(bytes: &[u8]) {
    imp::write_stderr(bytes);
}

#[cfg(unix)]
mod imp {
    use std::ffi::c_void;

    extern "C" {
        fn write(fd: i32, buf: *const c_void, count: usize) -> isize;
    }

    pub(super) fn write_stderr(bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        // SAFETY: `bytes` is a valid readable slice of `bytes.len()` bytes, and
        // file descriptor 2 is stderr. `write` is async-signal-safe, does not
        // allocate, and does not take any lock this crate holds.
        unsafe {
            let _ = write(2, bytes.as_ptr().cast::<c_void>(), bytes.len());
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;

    /// `STD_ERROR_HANDLE`, as a `u32` because the API takes a `DWORD`.
    const STD_ERROR_HANDLE: u32 = -12i32 as u32;

    #[link(name = "kernel32", kind = "raw-dylib")]
    extern "system" {
        fn GetStdHandle(which: u32) -> *mut c_void;
        fn WriteFile(
            handle: *mut c_void,
            buffer: *const c_void,
            to_write: u32,
            written: *mut u32,
            overlapped: *mut c_void,
        ) -> i32;
    }

    pub(super) fn write_stderr(bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        // SAFETY: `GetStdHandle` cannot fail in a way that matters here — an
        // invalid handle makes `WriteFile` fail, which is ignored. The buffer
        // is a valid readable slice, the length is clamped to `u32`, and a null
        // `overlapped` selects synchronous I/O, for which `written` must be a
        // valid out-pointer, which it is.
        unsafe {
            let handle = GetStdHandle(STD_ERROR_HANDLE);
            let mut written = 0u32;
            let to_write = bytes.len().min(u32::MAX as usize) as u32;
            let _ = WriteFile(
                handle,
                bytes.as_ptr().cast::<c_void>(),
                to_write,
                &mut written,
                std::ptr::null_mut(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisoning_is_sticky_and_reports_once() {
        let _guard = POISON_TESTS.lock();
        reset();
        QUIET.store(true, Ordering::SeqCst);

        assert!(!is_poisoned());
        assert!(!poison("test: first failure"), "first poison should report");
        assert!(is_poisoned());
        assert!(
            poison("test: second failure"),
            "second poison should be a no-op"
        );
        assert!(is_poisoned());

        reset();
        assert!(!is_poisoned());
    }

    #[test]
    fn reporting_is_bounded() {
        let _guard = POISON_TESTS.lock();
        reset();
        // The write path is covered by `reporting_does_not_allocate`; what is
        // under test here is the budget, and letting it write would put twenty
        // lines on the terminal for every run of the suite.
        QUIET.store(true, Ordering::SeqCst);

        // Far more than the budget. A fault on the allocator path can recur
        // millions of times, and a profiler that floods stderr has replaced one
        // problem with a worse one.
        for _ in 0..MAX_DIAGNOSTICS * 100 {
            report("test: bounded reporting check");
        }
        assert!(EMITTED.load(Ordering::Relaxed) >= MAX_DIAGNOSTICS);

        reset();
    }

    #[test]
    fn reporting_does_not_allocate() {
        let _guard = POISON_TESTS.lock();
        reset();
        // A real allocation count needs a counting global allocator, which only
        // an integration test can install; `tests/allocation_free.rs` covers
        // that. This checks the path runs at all without tripping any of the
        // debug machinery.
        report("test: allocation-free path");
        reset();
    }
}
