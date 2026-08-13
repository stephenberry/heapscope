//! The frame-pointer chain walker.
//!
//! On both supported architectures a frame record is two words at the frame
//! pointer:
//!
//! ```text
//! [fp + 0]  saved frame pointer of the caller
//! [fp + 8]  return address into the caller
//! ```
//!
//! aarch64 stores `x29`/`x30` in that order; x86_64's `push %rbp` followed by
//! the call's pushed return address produces the same layout. One walk serves
//! both.
//!
//! # Cost
//!
//! Measured on aarch64-apple-darwin with `benches/unwind.rs`, against a
//! baseline `malloc`/`free` of a 64-byte block at **16.7 ns**:
//!
//! | | |
//! |---|---|
//! | Fixed cost of a capture | ~5 ns |
//! | Marginal cost per frame | **~1.3 ns** |
//! | 12-frame capture | ~21 ns |
//! | `std::backtrace::Backtrace::force_capture` | **~18,800 ns** |
//!
//! The per-frame figure is measured as a *slope*: the output buffer size is
//! varied while everything else stays byte-identical, so the difference between
//! a 1-frame and a 32-frame capture is exactly the cost of 31 more frames. That
//! avoids subtracting a control that is never quite the same code.
//!
//! A capture therefore costs about the same as the allocation it is recording.
//! The standard library's unwinder costs roughly **900 times** as much, which
//! is why it is never selected automatically: at that price a profiler is not
//! slow, it is unusable, and the user concludes the tool is broken.
//!
//! # Reading memory the program can corrupt
//!
//! The chain lives in the profiled program's stack, which a bug in that program
//! can scribble over. The walk therefore validates every link before
//! dereferencing it, and reports *how* it stopped rather than silently
//! returning a short trace. See [`Outcome`].
//!
//! # Structure
//!
//! The walk policy is generic over a [`FrameSource`], and the unsafe pointer
//! reads live entirely in [`RealStack`]. That split is not decoration: reading
//! a frame record is an integer-to-pointer access with no provenance, which
//! Miri cannot model, so without the split none of this logic could be checked
//! under Miri at all. With it, every branch of the policy is exercised against
//! a synthetic stack built from ordinary Rust memory.

use std::ops::Range;

/// Distance between consecutive frame pointers that is treated as implausible
/// when stack bounds are unavailable.
///
/// Real frames are tens to hundreds of bytes; a megabyte is far beyond any
/// legitimate frame while still permitting large stack-allocated buffers.
const MAX_FRAME_SPAN: usize = 1024 * 1024;

/// Minimum alignment required of a frame pointer.
///
/// This is the alignment needed to read two words, not the ABI's stricter
/// requirement. Being permissive here avoids discarding valid frames from
/// hand-written assembly that keeps a correct chain without full ABI alignment.
const FP_ALIGN: usize = std::mem::align_of::<usize>();

/// How a capture terminated.
///
/// PLAN.md section 5.4 is explicit that validation prevents crashes but not
/// *wrong* traces, and that the honest response is to count the difference
/// rather than to project confidence. Every variant but [`Outcome::Complete`]
/// increments a counter that appears in the profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The chain ended cleanly at the outermost frame.
    Complete,
    /// The depth limit was reached with frames still remaining.
    TruncatedByDepth,
    /// A link failed validation: unaligned, out of the stack, not increasing,
    /// or an implausible span. The frames collected before it are still good.
    Suspect,
    /// The capture produced no frames at all.
    ///
    /// For the frame-pointer walk this means no frame pointer was available: on
    /// x86_64 the program was built without `-C force-frame-pointers=yes`, and
    /// it is a configuration error rather than an empty stack. For the platform
    /// unwinder it means the platform returned nothing, which it is entitled to
    /// do and which must never be mistaken for a complete trace of zero frames —
    /// every allocation in the program would land on one empty program point.
    NoFrames,
}

/// The result of one capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capture {
    /// Number of return addresses written to the output buffer.
    pub len: usize,
    /// How the walk ended.
    pub outcome: Outcome,
}

/// Supplies frame records to the walk.
///
/// Implemented by [`RealStack`] for the live stack and by a synthetic source in
/// the tests, so that the walk policy can be verified without dereferencing
/// addresses no Rust reference points at.
pub trait FrameSource {
    /// Returns `(saved_fp, return_address)` at `fp`, or `None` if `fp` is not
    /// readable.
    ///
    /// Implementations may assume `fp` has already passed the walk's structural
    /// validation.
    fn frame_at(&self, fp: usize) -> Option<(usize, usize)>;
}

/// Reads frame records from the calling thread's real stack.
///
/// Stack bounds are stored pre-reduced to a base and a span rather than as an
/// `Option<Range>`. The check then costs one subtraction and one comparison,
/// with no branch on whether bounds are known — the unknown case is expressed
/// as a span covering the whole address space. Measured, that is worth about
/// 1.2 ns per frame on aarch64-apple-darwin, which roughly halves the cost of
/// the walk's inner loop.
#[derive(Clone, Debug)]
pub struct RealStack {
    /// Lowest readable frame-record address.
    base: usize,
    /// `highest_readable_fp - base`. A frame pointer `fp` is in range exactly
    /// when `fp.wrapping_sub(base) <= span`, which handles `fp < base` without
    /// a second comparison because the subtraction wraps to a huge value.
    span: usize,
    /// Whether real bounds were available, for reporting rather than checking.
    bounded: bool,
}

impl RealStack {
    /// Creates a reader for the calling thread.
    ///
    /// `bounds` should come from [`crate::internals::stack::current_bounds`], cached
    /// per thread — querying it per capture would dominate the cost of the walk.
    ///
    /// When bounds are unavailable the reader accepts any address the walk's
    /// structural checks allow. That is strictly more dangerous, which is why
    /// [`RealStack::is_bounded`] exists and why such captures are reported as
    /// lower quality.
    pub fn new(bounds: Option<Range<usize>>) -> Self {
        const FRAME_RECORD: usize = 2 * std::mem::size_of::<usize>();

        match bounds {
            Some(range) if range.end.saturating_sub(range.start) >= FRAME_RECORD => Self {
                base: range.start,
                // The highest address at which a whole frame record still fits.
                span: (range.end - FRAME_RECORD) - range.start,
                bounded: true,
            },
            // Either no bounds, or a range too small to hold a frame record.
            // Both mean "no useful bound"; the walk's alignment, monotonicity,
            // and span checks are all that remain.
            _ => Self {
                base: 0,
                span: usize::MAX,
                bounded: false,
            },
        }
    }

    /// Whether this reader is backed by real stack bounds from the platform.
    pub fn is_bounded(&self) -> bool {
        self.bounded
    }
}

impl FrameSource for RealStack {
    #[inline(always)]
    fn frame_at(&self, fp: usize) -> Option<(usize, usize)> {
        // One subtraction and one comparison. `wrapping_sub` makes an `fp`
        // below `base` wrap to a value far above `span`, so the low bound is
        // enforced by the same comparison as the high bound.
        if fp.wrapping_sub(self.base) > self.span {
            return None;
        }

        // SAFETY: this is the one genuinely unverifiable read in the crate, and
        // it is unverifiable by nature: a frame pointer is an address the
        // compiler produced, not something any Rust reference points at.
        //
        // What makes it sound in practice:
        //   - `fp` has passed alignment and monotonicity checks in `walk`.
        //   - When the platform reported stack bounds, the two words read here
        //     lie entirely within them, and a thread's stack is mapped for its
        //     whole lifetime. Guard pages are outside the reported range on
        //     every supported platform.
        //   - The read is a single two-word load with no interior padding, so
        //     it cannot straddle the bound just validated.
        //
        // When bounds are unavailable the guarantee weakens to the walk's
        // structural checks, which is why such captures are marked suspect.
        unsafe {
            let ptr = std::ptr::with_exposed_provenance::<[usize; 2]>(fp);
            let [saved_fp, return_address] = ptr.read();
            Some((saved_fp, return_address))
        }
    }
}

/// Reads the calling function's frame pointer.
///
/// Returns `None` on architectures where this crate does not know how to ask,
/// which is reported as [`Outcome::NoFrames`] rather than as an empty
/// stack.
#[inline(always)]
pub fn frame_pointer() -> Option<usize> {
    // Miri cannot execute inline assembly, and reaching it is a hard abort
    // rather than a recoverable error. Reporting "no frame pointer" instead lets
    // anything that merely *touches* this path under Miri degrade the way it
    // would on an unsupported architecture. Tests that need a real capture are
    // ignored under Miri; the walk policy itself is covered there by the
    // synthetic-stack tests.
    #[cfg(miri)]
    {
        return None;
    }

    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        let fp: usize;
        // SAFETY: reads a single register into a general-purpose output. It
        // touches no memory, uses no stack, and modifies no flags, which is
        // what the options assert. `x29` is the frame pointer by the AAPCS64
        // ABI and is maintained by every non-leaf function on the supported
        // targets.
        unsafe {
            std::arch::asm!("mov {}, x29", out(reg) fp, options(nomem, nostack, preserves_flags));
        }
        Some(fp)
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        let fp: usize;
        // SAFETY: as above. `rbp` is read, never written; the compiler reserves
        // it when frame pointers are enabled, which the startup probe requires.
        unsafe {
            std::arch::asm!("mov {}, rbp", out(reg) fp, options(nomem, nostack, preserves_flags));
        }
        Some(fp)
    }
    #[cfg(not(any(miri, target_arch = "aarch64", target_arch = "x86_64")))]
    {
        None
    }
}

/// Walks a frame-pointer chain, writing return addresses into `out`.
///
/// `skip` drops that many innermost frames, which the caller uses to remove the
/// profiler's own frames from every recorded trace.
///
/// Never panics and never allocates.
pub fn walk<S: FrameSource>(
    source: &S,
    start_fp: usize,
    skip: usize,
    out: &mut [usize],
) -> Capture {
    let mut fp = start_fp;
    let mut previous = 0usize;
    let mut written = 0usize;
    let mut skipped = 0usize;

    loop {
        if !is_plausible(fp, previous, source) {
            return Capture {
                len: written,
                // A chain that ends at zero has reached the outermost frame,
                // which is the normal termination and not a defect.
                outcome: if fp == 0 {
                    Outcome::Complete
                } else {
                    Outcome::Suspect
                },
            };
        }

        let Some((saved_fp, return_address)) = source.frame_at(fp) else {
            return Capture {
                len: written,
                outcome: Outcome::Suspect,
            };
        };

        // A zero return address is how the outermost frame of a thread is
        // conventionally terminated.
        if return_address == 0 {
            return Capture {
                len: written,
                outcome: Outcome::Complete,
            };
        }

        if skipped < skip {
            skipped += 1;
        } else {
            if written == out.len() {
                return Capture {
                    len: written,
                    outcome: Outcome::TruncatedByDepth,
                };
            }
            out[written] = return_address;
            written += 1;
        }

        previous = fp;
        fp = saved_fp;
    }
}

/// Structural validation of a candidate frame pointer.
///
/// Rejecting a valid frame costs one truncated trace. Accepting an invalid one
/// risks a read outside the stack, so every check here fails closed.
#[inline(always)]
fn is_plausible<S: FrameSource>(fp: usize, previous: usize, _source: &S) -> bool {
    if fp == 0 || !fp.is_multiple_of(FP_ALIGN) {
        return false;
    }
    if previous != 0 {
        // Stacks grow downward, so each caller's frame sits at a higher address
        // than its callee's. Requiring a strict increase is what makes the walk
        // terminate: a corrupted chain that points back at itself, or at any
        // frame already visited, fails here instead of looping forever.
        if fp <= previous {
            return false;
        }
        if fp - previous > MAX_FRAME_SPAN {
            return false;
        }
    }
    true
}

/// Captures a backtrace from the caller's frame.
///
/// `skip` counts frames to discard from the inside out; the caller's own frame
/// is frame zero.
#[inline(always)]
pub fn capture(source: &RealStack, skip: usize, out: &mut [usize]) -> Capture {
    match frame_pointer() {
        Some(fp) => walk(source, fp, skip, out),
        None => Capture {
            len: 0,
            outcome: Outcome::NoFrames,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stack built out of ordinary Rust memory.
    ///
    /// Lets every branch of the walk policy be tested — including corrupted
    /// chains that would be undefined behaviour to construct on a real stack —
    /// and keeps those tests runnable under Miri.
    struct Synthetic {
        /// Frame records as `(fp, saved_fp, return_address)`.
        frames: Vec<(usize, usize, usize)>,
        /// Addresses that are readable; anything else behaves as unmapped.
        readable: Option<Range<usize>>,
    }

    impl FrameSource for Synthetic {
        fn frame_at(&self, fp: usize) -> Option<(usize, usize)> {
            if let Some(readable) = &self.readable {
                if !readable.contains(&fp) {
                    return None;
                }
            }
            self.frames
                .iter()
                .find(|(at, _, _)| *at == fp)
                .map(|(_, saved, ret)| (*saved, *ret))
        }
    }

    /// Builds a well-formed chain of `depth` frames starting at `base`.
    fn chain(base: usize, depth: usize) -> Synthetic {
        let stride = 64;
        let frames = (0..depth)
            .map(|i| {
                let fp = base + i * stride;
                let saved = if i + 1 == depth { 0 } else { fp + stride };
                (fp, saved, 0x1000 + i)
            })
            .collect();
        Synthetic {
            frames,
            readable: None,
        }
    }

    #[test]
    fn walks_a_well_formed_chain_to_the_end() {
        let source = chain(0x10_000, 5);
        let mut out = [0usize; 16];
        let capture = walk(&source, 0x10_000, 0, &mut out);

        assert_eq!(capture.outcome, Outcome::Complete);
        assert_eq!(capture.len, 5);
        assert_eq!(&out[..5], &[0x1000, 0x1001, 0x1002, 0x1003, 0x1004]);
    }

    #[test]
    fn skip_drops_innermost_frames() {
        let source = chain(0x10_000, 5);
        let mut out = [0usize; 16];
        let capture = walk(&source, 0x10_000, 2, &mut out);

        assert_eq!(capture.outcome, Outcome::Complete);
        assert_eq!(capture.len, 3);
        assert_eq!(&out[..3], &[0x1002, 0x1003, 0x1004]);
    }

    #[test]
    fn depth_limit_is_reported_as_truncation_not_completion() {
        let source = chain(0x10_000, 20);
        let mut out = [0usize; 4];
        let capture = walk(&source, 0x10_000, 0, &mut out);

        assert_eq!(capture.outcome, Outcome::TruncatedByDepth);
        assert_eq!(capture.len, 4);
    }

    /// The property that makes the walk safe to run on memory a buggy program
    /// can corrupt: it must terminate no matter what the chain says.
    #[test]
    fn a_self_referential_chain_terminates() {
        let source = Synthetic {
            frames: vec![(0x10_000, 0x10_000, 0xAAAA)],
            readable: None,
        };
        let mut out = [0usize; 8];
        let capture = walk(&source, 0x10_000, 0, &mut out);

        assert_eq!(capture.outcome, Outcome::Suspect);
        assert_eq!(capture.len, 1, "the one good frame should be kept");
    }

    #[test]
    fn a_cyclic_chain_terminates() {
        let source = Synthetic {
            frames: vec![
                (0x10_000, 0x10_040, 0xA),
                (0x10_040, 0x10_080, 0xB),
                // Points back downward, which cannot happen on a real stack.
                (0x10_080, 0x10_000, 0xC),
            ],
            readable: None,
        };
        let mut out = [0usize; 64];
        let capture = walk(&source, 0x10_000, 0, &mut out);

        assert_eq!(capture.outcome, Outcome::Suspect);
        assert_eq!(capture.len, 3);
    }

    #[test]
    fn a_descending_chain_is_rejected() {
        let source = Synthetic {
            frames: vec![(0x10_000, 0x9_000, 0xA), (0x9_000, 0, 0xB)],
            readable: None,
        };
        let mut out = [0usize; 8];
        let capture = walk(&source, 0x10_000, 0, &mut out);

        assert_eq!(capture.outcome, Outcome::Suspect);
        assert_eq!(capture.len, 1);
    }

    #[test]
    fn an_implausible_span_is_rejected() {
        let far = 0x10_000 + MAX_FRAME_SPAN + 8;
        let source = Synthetic {
            frames: vec![(0x10_000, far, 0xA), (far, 0, 0xB)],
            readable: None,
        };
        let mut out = [0usize; 8];
        let capture = walk(&source, 0x10_000, 0, &mut out);

        assert_eq!(capture.outcome, Outcome::Suspect);
        assert_eq!(capture.len, 1);
    }

    #[test]
    fn an_unaligned_frame_pointer_is_rejected_before_it_is_read() {
        let source = Synthetic {
            frames: vec![(0x10_001, 0, 0xA)],
            readable: None,
        };
        let mut out = [0usize; 8];
        let capture = walk(&source, 0x10_001, 0, &mut out);

        assert_eq!(capture.outcome, Outcome::Suspect);
        assert_eq!(capture.len, 0);
    }

    #[test]
    fn a_zero_return_address_ends_the_walk_cleanly() {
        let source = Synthetic {
            frames: vec![(0x10_000, 0x10_040, 0xA), (0x10_040, 0x10_080, 0)],
            readable: None,
        };
        let mut out = [0usize; 8];
        let capture = walk(&source, 0x10_000, 0, &mut out);

        assert_eq!(capture.outcome, Outcome::Complete);
        assert_eq!(capture.len, 1);
    }

    #[test]
    fn an_unreadable_frame_is_reported_as_suspect() {
        let mut source = chain(0x10_000, 5);
        // Only the first two frames are mapped.
        source.readable = Some(0x10_000..0x10_080);
        let mut out = [0usize; 8];
        let capture = walk(&source, 0x10_000, 0, &mut out);

        assert_eq!(capture.outcome, Outcome::Suspect);
        assert_eq!(capture.len, 2);
    }

    #[test]
    fn skipping_more_frames_than_exist_yields_nothing_but_does_not_fail() {
        let source = chain(0x10_000, 3);
        let mut out = [0usize; 8];
        let capture = walk(&source, 0x10_000, 100, &mut out);

        assert_eq!(capture.outcome, Outcome::Complete);
        assert_eq!(capture.len, 0);
    }

    #[test]
    fn a_zero_start_is_reported_as_complete_not_suspect() {
        let source = chain(0x10_000, 3);
        let mut out = [0usize; 8];
        let capture = walk(&source, 0, 0, &mut out);

        assert_eq!(capture.outcome, Outcome::Complete);
        assert_eq!(capture.len, 0);
    }

    #[test]
    fn an_empty_output_buffer_truncates_immediately() {
        let source = chain(0x10_000, 3);
        let mut out: [usize; 0] = [];
        let capture = walk(&source, 0x10_000, 0, &mut out);

        assert_eq!(capture.outcome, Outcome::TruncatedByDepth);
        assert_eq!(capture.len, 0);
    }
}
