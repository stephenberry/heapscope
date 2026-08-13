//! The platform's own stack unwinder.
//!
//! # Why this exists at all
//!
//! Two unrelated reasons, and it is worth keeping them apart.
//!
//! On **unix** it is an escape hatch, for a build that genuinely cannot supply
//! frame pointers — a large C++ dependency someone else compiled, most likely.
//! It is never selected automatically there, because it is much slower. How
//! much slower depends entirely on the platform, and the difference is large
//! enough to matter (`benches/unwind.rs`, 12 frames):
//!
//! | Platform | Frame-pointer walk | This | Ratio |
//! |---|---|---|---|
//! | x86_64-unknown-linux-gnu | 51 ns | **5,613 ns** | ~110x |
//! | aarch64-apple-darwin | 47 ns | **246 ns** | ~5x |
//!
//! The Darwin numbers were taken on a loaded machine — the baseline
//! `malloc`/`free` measured 29 ns there against the 16.7 ns this project
//! records for an idle one — so read them as a ratio rather than as absolutes.
//!
//! **That two-order-of-magnitude gap between platforms is the important part,
//! and it comes with a caveat that undercuts the whole feature on one of them.**
//! Darwin's `backtrace` is within one order of magnitude of a frame-pointer walk
//! — 5x, per the table above, not "about the same" as this once claimed —
//! because it *is* one: it walks the same chain, in libSystem. So on macOS this is not an
//! escape hatch from missing frame pointers at all — without them it fails
//! exactly as the frame-pointer walk does, and the startup probe refuses to
//! start rather than pretending otherwise. glibc's routes through the unwind
//! tables, which is both why it costs a hundred times more and why it is a real
//! answer on the platform where the problem actually arises: x86_64 Linux with
//! frame pointers omitted.
//!
//! On **Windows** it is the only thing that works, and is therefore the
//! default. Measured under Wine on `x86_64-pc-windows-gnu`, from the same stack
//! at the same instant:
//!
//! | | with `-C force-frame-pointers=yes` | without |
//! |---|---|---|
//! | Hand-walked `rbp` chain | 2 entries, the second a stack address | 1 entry, `0x8` |
//! | `RtlCaptureStackBackTrace` | **9 frames**, all plausible | **9 frames**, identical |
//!
//! That is the Microsoft x64 ABI behaving as specified rather than a bug.
//! Windows requires unwind data (`.pdata`/`.xdata`) for every function, so the
//! platform never needed a linked `rbp` chain and `-C force-frame-pointers=yes`
//! does not produce one that can be walked. `RtlCaptureStackBackTrace` reads
//! those tables, which is why it returns the same frames whether or not the flag
//! is set — and why, unlike the unix system unwinder, it needs no build
//! configuration from the user at all.
//!
//! **The Windows cost is unmeasured.** Wine timings say nothing about the real
//! platform, and this machine is not a Windows machine. Expect a table walk to
//! cost far more than a chain walk; the number goes here when a real Windows
//! machine or CI produces one.
//!
//! # Zero frames
//!
//! Both platform calls can return nothing, and a profiler that silently accepted
//! that would attribute every allocation in the program to one empty program
//! point. Startup probing catches a build where it never works;
//! [`Outcome::NoFrames`](super::Outcome::NoFrames) counts the ones that fail
//! later, so the profile can say how often it happened.

use std::ffi::c_void;

use super::frame_pointer::{Capture, Outcome};

/// Most frames one call will return.
///
/// `RtlCaptureStackBackTrace` documented a limit of 62 for
/// `FramesToSkip + FramesToCapture` on Windows Server 2003 and XP; the current
/// documentation has dropped it. The clamp is kept for those platforms, and it
/// is **not** free: the shim's buffer is
/// [`CAPTURE_DEPTH`](crate::CAPTURE_DEPTH), which is 64, so every capture asks
/// for 61 of the 64 frames it has room for. Deep stacks are therefore cut three
/// frames earlier on Windows than elsewhere, and the capture is reported as
/// truncated rather than complete when that happens.
///
/// The original comment justified this as costing nothing, by reading a default
/// depth of 24 off a constant that sized no buffer. The buffer is the one named
/// above.
#[cfg(windows)]
const WINDOWS_LIMIT: usize = 62;

/// Captures a backtrace with the platform unwinder, skipping `skip` innermost
/// frames.
///
/// Allocation-free on the Windows path. On glibc the *first* call can allocate,
/// because `backtrace` lazily loads the unwinder; the shim holds the reentrancy
/// guard across the capture, so that allocation is forwarded to the inner
/// allocator and recorded nowhere.
///
/// `#[inline(never)]` because the frame layout above this function has to be
/// the same at every optimisation level. [`crate::unwind::calibrate`] measures
/// how many frames the machinery contributes and the shim skips that many; if
/// the platform call moved between inlined and not, the answer would change
/// with the optimisation level in a way nothing would notice. The cost is one
/// call instruction against a platform unwind that costs hundreds of
/// nanoseconds.
#[inline(never)]
pub fn capture(skip: usize, out: &mut [usize]) -> Capture {
    if out.is_empty() {
        return Capture {
            len: 0,
            outcome: Outcome::TruncatedByDepth,
        };
    }

    let Some(captured) = imp::capture(skip, out) else {
        // The platform was asked to skip more frames than the stack has. Those
        // frames are the caller's to discard; handing back what was meant to be
        // discarded, as the Windows path did before its clamp was removed, is
        // worse than an empty capture.
        return Capture {
            len: 0,
            outcome: Outcome::NoFrames,
        };
    };

    let outcome = if captured.len == 0 {
        Outcome::NoFrames
    } else if captured.filled_the_request {
        // Compared against what the *platform* was asked for, not against
        // `out.len()`. Those differ whenever anything is skipped or clamped, and
        // comparing to `out.len()` made truncation unreportable in exactly the
        // configuration the shim uses: `SKIP_FRAMES` is never zero, so `len`
        // could never reach the buffer length, and every truncated capture on
        // both platforms was labelled `Complete`.
        Outcome::TruncatedByDepth
    } else {
        Outcome::Complete
    };
    Capture {
        len: captured.len,
        outcome,
    }
}

/// What one platform call produced.
struct Captured {
    /// Frames written to the caller's buffer.
    len: usize,
    /// Whether the platform returned everything it was asked for, which means
    /// there may be more stack it was not given room for.
    filled_the_request: bool,
}

/// Reinterprets a `usize` buffer as the pointer buffer the platform wants.
///
/// `usize` and `*mut c_void` have the same size and alignment on every supported
/// target, and the addresses are only ever read back as integers afterwards, so
/// nothing depends on the provenance the platform writes.
fn as_pointer_buffer(out: &mut [usize]) -> *mut *mut c_void {
    out.as_mut_ptr().cast::<*mut c_void>()
}

#[cfg(windows)]
mod imp {
    use super::{as_pointer_buffer, Captured, WINDOWS_LIMIT};
    use std::ffi::c_void;

    #[link(name = "kernel32", kind = "raw-dylib")]
    extern "system" {
        /// Walks the caller's stack using the process's unwind tables.
        ///
        /// Exported by `kernel32` as a forwarder to `ntdll`. Returns the number
        /// of frames written, which is a `USHORT` and can be zero.
        fn RtlCaptureStackBackTrace(
            frames_to_skip: u32,
            frames_to_capture: u32,
            backtrace: *mut *mut c_void,
            backtrace_hash: *mut u32,
        ) -> u16;
    }

    pub(super) fn capture(skip: usize, out: &mut [usize]) -> Option<Captured> {
        // Not clamped. `skip.min(WINDOWS_LIMIT - 1)` quietly reinterpreted "skip
        // 1000" as "skip 61" and handed back a frame the caller had asked to
        // discard, labelled as a complete trace. A skip the platform cannot
        // honour is an empty capture.
        if skip >= WINDOWS_LIMIT {
            return None;
        }
        let wanted = out.len().min(WINDOWS_LIMIT - skip);

        // SAFETY: `out` is a live, writable slice of at least `wanted`
        // pointer-sized elements, and the buffer is reinterpreted as the
        // pointer array the function writes. A null hash pointer means "do not
        // compute one", which is documented.
        let captured = unsafe {
            RtlCaptureStackBackTrace(
                skip as u32,
                wanted as u32,
                as_pointer_buffer(out),
                std::ptr::null_mut(),
            )
        };
        let len = usize::from(captured).min(wanted);
        Some(Captured {
            len,
            filled_the_request: len == wanted,
        })
    }
}

#[cfg(unix)]
mod imp {
    use super::{as_pointer_buffer, Captured};
    use std::ffi::{c_int, c_void};

    extern "C" {
        /// Fills `buffer` with return addresses, innermost first, and returns
        /// how many it wrote.
        ///
        /// In libc on glibc and in libSystem on Darwin. Absent on musl, which
        /// PLAN.md section 1.1 makes a permanent non-goal.
        fn backtrace(buffer: *mut *mut c_void, size: c_int) -> c_int;
    }

    pub(super) fn capture(skip: usize, out: &mut [usize]) -> Option<Captured> {
        let wanted = c_int::try_from(out.len()).unwrap_or(c_int::MAX);

        // SAFETY: `out` is a live, writable slice of `wanted` pointer-sized
        // elements, reinterpreted as the pointer array the function writes.
        let raw = unsafe { backtrace(as_pointer_buffer(out), wanted) };
        let raw = usize::try_from(raw).unwrap_or(0).min(out.len());

        // Whether the *platform* filled the buffer, decided before the skip is
        // applied. Afterwards the length is always short by `skip`, so a check
        // there can never see a full buffer.
        let filled_the_request = raw == out.len();

        // `backtrace` has no skip parameter, so the innermost frames are dropped
        // by shifting the rest down. `copy_within` handles the overlap.
        if skip == 0 {
            return Some(Captured {
                len: raw,
                filled_the_request,
            });
        }
        if skip >= raw {
            return None;
        }
        out.copy_within(skip..raw, 0);
        Some(Captured {
            len: raw - skip,
            filled_the_request,
        })
    }
}

#[cfg(not(any(windows, unix)))]
mod imp {
    use super::Captured;

    pub(super) fn capture(_skip: usize, _out: &mut [usize]) -> Option<Captured> {
        Some(Captured {
            len: 0,
            filled_the_request: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The platform unwinder is a foreign function Miri has no shim for, and
    /// reaching one aborts the run rather than failing a test.
    #[test]
    #[cfg_attr(miri, ignore = "calls the platform's unwinder")]
    fn a_real_capture_returns_plausible_addresses() {
        #[inline(never)]
        fn inner(out: &mut [usize]) -> Capture {
            std::hint::black_box(capture(0, out))
        }

        let mut out = [0usize; 32];
        let result = inner(&mut out);

        assert!(result.len > 0, "the platform unwinder returned nothing");
        assert!(
            out[..result.len].iter().all(|&address| address > 0x1000),
            "an address is implausibly low: {:x?}",
            &out[..result.len]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "calls the platform's unwinder")]
    fn skipping_drops_the_innermost_frames() {
        // Both captures are taken from *one* call instruction, reached twice
        // through a loop. Two separate calls would put two different return
        // addresses in this function's own frame, and the comparison below
        // would then be against a frame that was never meant to match — which
        // is exactly what it was doing, passing only because the machinery
        // happened to be one frame deeper in release than in debug.
        #[inline(never)]
        fn inner(runs: &mut [(usize, [usize; 32], usize)]) {
            for run in runs.iter_mut() {
                let skip = std::hint::black_box(run.0);
                let capture = capture(skip, &mut run.1);
                run.2 = std::hint::black_box(capture.len);
            }
        }

        const SKIP: usize = 2;
        let mut runs = [(0usize, [0usize; 32], 0usize), (SKIP, [0usize; 32], 0usize)];
        inner(&mut runs);
        let (_, none, unskipped_len) = runs[0];
        let (_, skipped, skipped_len) = runs[1];

        assert!(unskipped_len > SKIP, "not deep enough to test skipping");
        assert_eq!(
            skipped_len,
            unskipped_len - SKIP,
            "skipping {SKIP} frames should return {SKIP} fewer"
        );
        // Identical stacks, so the shifted trace must match frame for frame.
        // No index here depends on how many frames the machinery contributes,
        // which is the whole point: that number changes with the optimisation
        // level.
        assert_eq!(
            &skipped[..skipped_len],
            &none[SKIP..unskipped_len],
            "the wrong frames were dropped"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "calls the platform's unwinder")]
    fn a_full_buffer_reports_truncation() {
        let mut out = [0usize; 2];
        let result = capture(0, &mut out);
        assert_eq!(result.len, 2);
        assert_eq!(result.outcome, Outcome::TruncatedByDepth);
    }

    #[test]
    fn an_empty_buffer_cannot_be_written_to() {
        let result = capture(0, &mut []);
        assert_eq!(result.len, 0);
        assert_eq!(result.outcome, Outcome::TruncatedByDepth);
    }

    /// A skip larger than the stack must produce nothing, on every platform.
    ///
    /// Windows used to clamp the skip to its own 62-frame limit, so "skip 1000"
    /// became "skip 61" and returned a frame the caller had asked to discard,
    /// labelled `Complete`. Under Wine that went unnoticed because the test
    /// thread's stack was shallower than 61 frames — the assertion held by
    /// accident, from a stack that never reached the clamp.
    #[test]
    #[cfg_attr(miri, ignore = "calls the platform's unwinder")]
    fn skipping_past_the_whole_stack_returns_nothing_rather_than_garbage() {
        for skip in [1000, 100_000, usize::MAX] {
            let mut out = [0usize; 8];
            let result = capture(skip, &mut out);
            assert_eq!(result.len, 0, "skip={skip}");
            assert_eq!(
                result.outcome,
                Outcome::NoFrames,
                "skip={skip}: an empty capture must be reported as empty, not as \
                 a complete trace of zero frames"
            );
        }
    }

    /// Truncation must be reported when anything is skipped, which is the only
    /// configuration the shim ever uses.
    #[test]
    #[cfg_attr(miri, ignore = "calls the platform's unwinder")]
    fn truncation_is_reported_even_when_frames_are_skipped() {
        #[inline(never)]
        fn deep(depth: usize, out: &mut [usize]) -> Capture {
            if depth == 0 {
                return std::hint::black_box(capture(2, out));
            }
            std::hint::black_box(deep(depth - 1, out))
        }

        let mut out = [0usize; 8];
        let result = deep(40, &mut out);
        assert_eq!(
            result.outcome,
            Outcome::TruncatedByDepth,
            "a 40-deep stack into an 8-slot buffer is truncated however many \
             frames are skipped; len={}",
            result.len
        );
    }
}
