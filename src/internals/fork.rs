//! Surviving `fork`.
//!
//! # Why detection cannot work
//!
//! Revision 1 of PLAN.md proposed noticing a changed pid on the allocation path.
//! A probe running this crate's own lock design — a background thread hammering
//! the lock, then a `fork` — reported the same thing on all six runs:
//!
//! ```text
//! child: LOCK IS STUCK HELD -> would deadlock forever
//! ```
//!
//! By the time anything in the child *could* notice, the damage is done. The
//! lock is held by a thread that `fork` did not copy, so it can never be
//! released, and the first acquisition in the child — including the one on the
//! path that writes the profile — blocks forever. Detection would also put a
//! `getpid()` on the hot path to fix nothing.
//!
//! # What this does instead
//!
//! `pthread_atfork` registers three handlers that run around the call, in the
//! only process states where the work is possible:
//!
//! | Handler | Runs in | Work |
//! |---|---|---|
//! | `prepare` | parent, before the fork | take every lock, in the global order |
//! | `parent` | parent, after the fork | release every lock |
//! | `child` | child, after the fork | overwrite every lock; stop recording |
//!
//! The cost at runtime is zero: the handlers run only when the program forks.
//!
//! # Sharing the fork window with other libraries
//!
//! POSIX runs prepare handlers in **reverse** registration order. Ours is
//! registered from `Profiler::new`, which is late, so ours runs **first** and
//! every handler registered earlier runs afterwards *on the same thread, with
//! all of this crate's locks held*. Libraries initialised before `main` —
//! OpenSSL, CPython, assorted logging and telemetry crates — do register these,
//! and a handler that allocates would enter the shim, find the guard free, and
//! take a lock the same thread already holds. That is a hang on Linux and a
//! `SIGKILL` on Apple platforms.
//!
//! So the handlers hold the **reentrancy guard** across the whole fork window.
//! Anything that allocates in there — another library's handler, or the C
//! library's own — is forwarded straight to the inner allocator and recorded
//! nowhere, which is exactly what the guard is for everywhere else.
//!
//! # The limit of this
//!
//! A `fork` that races our own `prepare` handler — a second thread forking while
//! the first is inside it — is not recoverable in general, and no
//! `pthread_atfork` user handles it. Neither is a `fork` issued from a signal
//! handler that interrupted a thread inside the shim: `prepare` would block on a
//! lock the interrupted thread holds and cannot release until the handler
//! returns. Both are documented rather than defended against.
//!
//! One cost is worth stating plainly rather than calling it free: any thread
//! blocked inside the shim while a *later* prepare handler waits on a lock that
//! thread holds is a deadlock this arrangement introduces. That is inherent to
//! `pthread_atfork` rather than specific to this crate, and it is why
//! [`Engine::fork_prepare`](crate::internals::engine::Engine::fork_prepare) gives up
//! rather than waiting forever.
//!
//! Windows has no `fork`, so none of this is compiled there.

use std::sync::atomic::{AtomicU8, Ordering};

/// Not yet attempted.
const UNREGISTERED: u8 = 0;
/// A thread is inside [`install`].
const REGISTERING: u8 = 1;
/// Handlers are in place.
const REGISTERED: u8 = 2;
/// The platform refused the registration.
const FAILED: u8 = 3;

static STATE: AtomicU8 = AtomicU8::new(UNREGISTERED);

/// Registers the `fork` handlers, once per process.
///
/// Returns whether the handlers are in place. `false` means the platform
/// declined — `pthread_atfork` can fail with `ENOMEM` — and the caller reports
/// it rather than leaving the user to discover that a forking program hangs.
///
/// Handlers are never unregistered. POSIX offers no way to do so, and the
/// handlers are written to be correct against an idle engine anyway, so a
/// profiler that has stopped leaves nothing dangerous behind.
pub fn install() -> bool {
    match STATE.compare_exchange(
        UNREGISTERED,
        REGISTERING,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => {}
        Err(REGISTERED) => return true,
        Err(FAILED) => return false,
        // Another thread is inside this function. `install` runs *before*
        // `engine().start()`, so two concurrent `Profiler::new` calls genuinely
        // reach here — and answering "no" would report a registration failure
        // that did not happen, telling the user their forking program may wedge
        // when it will not. Wait for the answer instead. The other thread is
        // between a compare-exchange and a `pthread_atfork` call, so the wait
        // is over almost immediately.
        Err(_) => {
            while STATE.load(Ordering::Acquire) == REGISTERING {
                std::thread::yield_now();
            }
            return STATE.load(Ordering::Acquire) == REGISTERED;
        }
    }

    let registered = register();
    STATE.store(
        if registered { REGISTERED } else { FAILED },
        Ordering::Release,
    );
    registered
}

/// Whether the handlers are installed. For tests and self-metrics.
pub fn is_installed() -> bool {
    STATE.load(Ordering::Acquire) == REGISTERED
}

#[cfg(unix)]
fn register() -> bool {
    extern "C" {
        /// POSIX: returns 0 on success, `ENOMEM` on failure. Present in libc on
        /// every supported unix; glibc has provided it outside libpthread since
        /// 2.34, and Darwin has always had it in libSystem.
        fn pthread_atfork(
            prepare: Option<extern "C" fn()>,
            parent: Option<extern "C" fn()>,
            child: Option<extern "C" fn()>,
        ) -> std::ffi::c_int;
    }

    // SAFETY: the three arguments are `extern "C"` functions with the required
    // signature and `'static` lifetime. Registration itself has no
    // preconditions.
    let status = unsafe { pthread_atfork(Some(prepare), Some(parent), Some(child)) };
    status == 0
}

#[cfg(not(unix))]
fn register() -> bool {
    // No `fork`, so nothing to survive. Reported as installed because the
    // guarantee the caller is asking about — "a fork will not wedge this
    // process" — holds vacuously.
    true
}

/// Whether `prepare` entered the reentrancy guard, so that exactly one of
/// `parent`/`child` leaves it.
///
/// A plain static is enough: these three run serialised around one `fork`, on
/// one thread, and a `fork` racing our own `prepare` is already documented as
/// unrecoverable.
#[cfg(unix)]
static GUARD_ENTERED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn prepare() {
    // Held across the whole fork window. See the module documentation: the
    // handlers that run after this one do so with every heapscope lock held,
    // and one of them allocating would otherwise reacquire a lock on this very
    // thread.
    GUARD_ENTERED.store(
        crate::internals::guard::enter_unbalanced(),
        Ordering::Release,
    );

    // SAFETY: paired with `parent` in the parent and discharged by `child` in
    // the child, which is exactly what `pthread_atfork` guarantees about these
    // three functions.
    unsafe { crate::engine().fork_prepare() }
}

#[cfg(unix)]
extern "C" fn parent() {
    // SAFETY: runs on the thread that ran `prepare`, after `fork` returns —
    // including when `fork` failed, which both glibc and Darwin guarantee.
    unsafe { crate::engine().fork_parent() }
    leave_guard();
}

#[cfg(unix)]
extern "C" fn child() {
    // SAFETY: a `fork` child has exactly one thread.
    unsafe { crate::engine().fork_child() }
    leave_guard();
}

#[cfg(unix)]
fn leave_guard() {
    if GUARD_ENTERED.swap(false, Ordering::AcqRel) {
        crate::internals::guard::leave_unbalanced();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installing_twice_reports_success_both_times() {
        assert!(install());
        assert!(install());
        assert!(is_installed());
    }
}
