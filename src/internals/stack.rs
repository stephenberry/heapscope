//! Stack bounds for the calling thread.
//!
//! The frame-pointer walker follows a linked list that lives in memory the
//! program under test is free to corrupt. Alignment and monotonicity checks
//! stop the walk from looping, but they cannot stop it from walking *off the
//! top of the stack* into unmapped memory: a walk of depth 64 with a plausible
//! per-frame bound is still tens of megabytes of reach. Knowing where the stack
//! ends is what turns a possible segfault in the profiled program into a
//! truncated trace and a counter.
//!
//! Every query here is allocation-free except `pthread_getattr_np` on the glibc
//! main thread, which reads `/proc/self/maps`. Callers on the allocator path
//! must therefore already hold the reentrancy guard, and the result is cached
//! per thread so the cost is paid once.

use std::ops::Range;

/// The address range of the calling thread's stack, low bound inclusive and
/// high bound exclusive.
///
/// Returns `None` when the platform declines to say, in which case the walker
/// falls back to its weaker structural checks and reports the traces it
/// produced as suspect.
pub fn current_bounds() -> Option<Range<usize>> {
    imp::current_bounds()
}

#[cfg(target_vendor = "apple")]
mod imp {
    use std::ffi::c_void;
    use std::ops::Range;

    extern "C" {
        fn pthread_self() -> *mut c_void;
        /// Returns the *highest* address of the thread's stack.
        fn pthread_get_stackaddr_np(thread: *mut c_void) -> *mut c_void;
        fn pthread_get_stacksize_np(thread: *mut c_void) -> usize;
    }

    pub(super) fn current_bounds() -> Option<Range<usize>> {
        // SAFETY: all three take the calling thread's own handle, cannot fail
        // for a live thread, and allocate nothing. They are the documented
        // Darwin interface for exactly this question.
        let (high, size) = unsafe {
            let me = pthread_self();
            (
                pthread_get_stackaddr_np(me).addr(),
                pthread_get_stacksize_np(me),
            )
        };
        if high == 0 || size == 0 {
            return None;
        }
        Some(high.checked_sub(size)?..high)
    }
}

#[cfg(all(unix, not(target_vendor = "apple")))]
mod imp {
    use std::ffi::{c_int, c_void};
    use std::ops::Range;

    // `pthread_attr_t` is 56 bytes on x86_64-linux-gnu and 64 on
    // aarch64-linux-gnu. Over-sizing wastes stack; under-sizing corrupts it, so
    // this is deliberately generous. The alignment requirement is that of a
    // pointer.
    #[repr(C, align(8))]
    struct PthreadAttr([u8; 128]);

    // glibc's `pthread_t` is `unsigned long int`, not a pointer. See the note
    // in `super::guard::thread_handle`.
    type PthreadT = std::ffi::c_ulong;

    extern "C" {
        fn pthread_self() -> PthreadT;
        fn pthread_getattr_np(thread: PthreadT, attr: *mut c_void) -> c_int;
        fn pthread_attr_getstack(
            attr: *const c_void,
            stackaddr: *mut *mut c_void,
            stacksize: *mut usize,
        ) -> c_int;
        fn pthread_attr_destroy(attr: *mut c_void) -> c_int;
    }

    pub(super) fn current_bounds() -> Option<Range<usize>> {
        let mut attr = PthreadAttr([0; 128]);
        let mut addr: *mut c_void = std::ptr::null_mut();
        let mut size: usize = 0;

        // SAFETY: `attr` is a sufficiently large, sufficiently aligned, writable
        // buffer for a `pthread_attr_t`. `pthread_getattr_np` initializes it on
        // success, and it is destroyed on every path that initialized it.
        // `pthread_attr_getstack` writes through two valid out-pointers.
        //
        // Note: for the glibc main thread this reads `/proc/self/maps` and
        // allocates. Callers on the allocator path must hold the reentrancy
        // guard, which is why the result is cached per thread.
        unsafe {
            let attr_ptr = std::ptr::from_mut(&mut attr).cast::<c_void>();
            if pthread_getattr_np(pthread_self(), attr_ptr) != 0 {
                return None;
            }
            let rc = pthread_attr_getstack(attr_ptr.cast_const(), &mut addr, &mut size);
            pthread_attr_destroy(attr_ptr);
            if rc != 0 {
                return None;
            }
        }

        // Unlike Darwin, glibc reports the *lowest* address.
        let low = addr.addr();
        if low == 0 || size == 0 {
            return None;
        }
        Some(low..low.checked_add(size)?)
    }
}

#[cfg(windows)]
mod imp {
    use std::ops::Range;

    #[link(name = "kernel32", kind = "raw-dylib")]
    extern "system" {
        fn GetCurrentThreadStackLimits(low: *mut usize, high: *mut usize);
    }

    pub(super) fn current_bounds() -> Option<Range<usize>> {
        let mut low = 0usize;
        let mut high = 0usize;
        // SAFETY: two valid out-pointers to initialized locals. The function
        // cannot fail and allocates nothing.
        unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
        if low == 0 || high <= low {
            return None;
        }
        Some(low..high)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Miri shims the stack-bounds calls but returns a synthetic range with no
    // relationship to where it places locals — it does not model a single
    // contiguous machine stack at all. A test relating a real address to the
    // reported bounds would therefore be checking Miri's fiction rather than
    // this code, so every test in this module runs natively only.

    #[test]
    #[cfg_attr(
        miri,
        ignore = "Miri reports synthetic stack bounds unrelated to its locals"
    )]
    fn bounds_contain_a_local_variable() {
        let local = 0u64;
        let addr = std::ptr::from_ref(&local).addr();
        let bounds = current_bounds().expect("every supported platform reports stack bounds");
        assert!(
            bounds.contains(&addr),
            "a local at {addr:#x} is outside the reported stack {:#x}..{:#x}",
            bounds.start,
            bounds.end
        );
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "Miri reports synthetic stack bounds unrelated to its locals"
    )]
    fn bounds_are_plausible() {
        let bounds = current_bounds().unwrap();
        let size = bounds.end - bounds.start;
        assert!(
            (16 * 1024..=1024 * 1024 * 1024).contains(&size),
            "implausible stack size {size} bytes"
        );
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "Miri reports synthetic stack bounds unrelated to its locals"
    )]
    fn spawned_threads_report_their_own_stack() {
        let main_bounds = current_bounds().unwrap();
        let child_bounds = std::thread::spawn(|| {
            let local = 0u64;
            let addr = std::ptr::from_ref(&local).addr();
            let bounds = current_bounds().expect("spawned thread should report bounds");
            assert!(bounds.contains(&addr));
            bounds
        })
        .join()
        .unwrap();

        assert!(
            child_bounds.start != main_bounds.start || child_bounds.end != main_bounds.end,
            "a spawned thread reported the main thread's stack"
        );
    }
}
