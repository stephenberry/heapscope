//! Dropping the frames that every stack has and none of them is about.
//!
//! A captured stack has three parts. In the middle is the program: the call
//! chain that decided to allocate, which is the whole reason the stack was
//! captured. Above it is the standard allocation path — `Vec` reaching
//! `RawVec` reaching `__rust_alloc` reaching this crate's shim — which is the
//! same handful of frames for every allocation of that kind. Below it is the
//! runtime that started the thread, which is identical for every stack in the
//! process.
//!
//! Measured on this crate's own example program, a debug build on macOS
//! aarch64: the deepest point is seventeen frames, of which three are the
//! allocation path and six are the runtime, leaving eight that say anything
//! **\[measured\]**. Across the whole profile 144 frames become 51 — 89 removed
//! by the rules and four more with a program point that folded into another —
//! and the frame table falls from 55 entries to 37. On a spawned thread it is
//! worse still: the thread entry alone is nine frames of `spawn_unchecked`,
//! `catch_unwind`, and `pthread_start`.
//!
//! # Two rules, each anchored on a marker rather than a threshold
//!
//! **The runtime entry.** `std` marks the boundary itself:
//! `std::sys::backtrace::__rust_begin_short_backtrace` exists so that `std`'s
//! own backtraces can be cut there. Everything from it outwards is dropped,
//! which is the same cut `std` makes at that marker — `sys/backtrace.rs` stops
//! printing when it sees it, so the `FnOnce::call_once` shim just inside it
//! stays visible in both.
//!
//! Two things about the marker are worth stating exactly, because the
//! convenient version of each is wrong. It is declared
//! `#[cfg_attr(feature = "backtrace", inline(never))]`, **not** unconditionally
//! `#[inline(never)]`: the attribute is there for a `std` built with backtrace
//! support, and its own comment says it is "fine to optimize away" otherwise.
//! What is measured here is only that it does survive `opt-level=3`, and thin
//! LTO with one codegen unit, in the toolchains this crate is tested against
//! **\[measured\]**. And it appears *at least* once per thread rather than
//! exactly once: current `std` calls it twice on a spawned thread, once around
//! the spawn hooks and once around the closure. The rule cuts at the innermost,
//! so that costs nothing, but "once" was not true.
//!
//! `std` also honours a second marker, `__rust_end_short_backtrace`, which
//! turns printing back *on* for frames inside it. This deliberately does not:
//! that marker wraps the panic machinery and the allocation-error hook, and an
//! allocation made in there — formatting a panic message — has the formatting
//! frames as its honest answer. Cutting them would remove the only thing that
//! stack says.
//!
//! Deliberately *not* a list of entry-point names (`main`, `start`, `_start`,
//! `__libc_start_main`, `mainCRTStartup`). That list is per-platform, changes
//! under us, and is simply wrong for a library loaded into a host that is not
//! Rust. One marker with one owner rots more slowly than five that we maintain.
//! The cost is that a stack with no marker — an allocation made before `main`,
//! or on a thread a C library created — keeps its runtime frames, which is the
//! honest answer rather than a guessed one.
//!
//! There is a second reason, and it is the stronger one: **a name does not
//! decide whether a frame is machinery — its position does.**
//! `std::thread::lifecycle::spawn_unchecked` appears below the marker on a
//! spawned thread, where it is what started it, and above the marker on the
//! parent, where it is the frame that boxed the closure and is exactly where
//! those bytes came from **\[measured, `tests/end_to_end.rs`\]**. Any rule that
//! matched it by name would be wrong in one of those two places. Cutting at the
//! boundary is right in both.
//!
//! **The allocation path.** The *leading run* of frames whose names begin with
//! one of the `ALLOCATION_PATH` prefixes below, stopping at the first frame
//! that does not match. A run rather than a search for the last match, because
//! a run provably cannot remove a frame that has program code beneath it: a
//! real profile has `<heapscope::profiler::Profiler>::print_summary` four
//! frames in, under `std::io::Stdout::lock`, and that frame is the answer to
//! where the allocation came from **\[measured\]**.
//!
//! The list has a rule to grow by, which matters more than its current
//! contents: **stop at the first frame the program's own source names.** That
//! is what separates the pairs that look alike, all measured on one profile:
//!
//! ```text
//! kept                                                dropped just above it
//! <alloc::vec::Vec<u8>>::with_capacity                <alloc::raw_vec::RawVecInner>::with_capacity_in
//! <alloc::vec::Vec<u64>>::push_mut                    <alloc::raw_vec::RawVec<u64>>::grow_one
//! <hashbrown::raw::RawTableInner>::new_uninitialized  <alloc::alloc::Global as core::alloc::Allocator>::allocate
//! alloc::vec::from_elem::<u32>                        <u32 as …spec_from_elem::SpecFromElem>::from_elem
//! ```
//!
//! `alloc::vec::from_elem` is what `vec![x; n]` expands to, so it is the user's
//! own expression; `SpecFromElem::from_elem` one frame above it is the
//! specialisation machinery. The third row is the case worth having in the
//! list: the first frame a program names is not always its own code —
//! `HashMap::with_capacity` reaches into `hashbrown`, and `hashbrown` asking
//! `Global` for the bytes is where that map's memory comes from. `Box` looks like an exception and is not: what
//! appears is never `Box::new` — that is `#[inline]` — but the compiler's
//! `alloc::boxed::box_new_uninit` helper, or a collection's internal
//! `Box::<T>::try_new_uninit_in`. Neither is written anywhere in a program
//! **\[measured\]**.
//!
//! # What this cannot do, and where
//!
//! Both rules read the *name* of a frame, so where there are no names there is
//! nothing to read and nothing is trimmed. A stack is then left exactly as it
//! was, which is the honest answer rather than a guessed one.
//!
//! That is not only the stripped-binary case. **On Linux it is the ordinary
//! case**: `dladdr` resolves against `.dynsym`, and a Rust executable exports
//! almost none of itself. A recorded profile carried **0 of 70 named frames on
//! aarch64 Linux against 52 of 52 on macOS aarch64** **\[measured\]**, so neither
//! rule fires there at all. `-C link-args=-rdynamic` does not close it: it
//! brings back the `std` and `alloc` generic instantiations that have global
//! linkage and leaves `__rust_alloc`, `__rust_begin_short_backtrace`, and every
//! function in the user's own crate unnamed **\[measured\]**.
//!
//! This is the same fact that makes tier 2 the primary path — see the
//! [module documentation](super) — arriving somewhere new. The frames a Linux
//! profile needs trimmed are exactly the frames only an offline symbolizer can
//! name, so trimming them belongs to whatever does that resolution, not here.
//!
//! # It can make two program points identical
//!
//! Removing frames can collapse two interned points onto one frame list, which
//! is a file `dh_view.html` refuses to open. That is not a hazard introduced
//! here and left to be discovered: the emitter re-keys every point by its final
//! frame list and merges collisions, and reports how many it merged. See
//! `output::dhat_v2`.
//!
//! It is not theoretical either. The example program's `vec![n as u32; 4]`
//! reaches the allocator two ways — `__rust_alloc` when `RawVecInner` takes the
//! general path and `__rust_alloc_zeroed` when the specialisation asks for
//! zeroed memory — and once the allocation path is gone both are the same four
//! frames. Exactly one pair folds there **\[measured\]**.

use std::ops::Range;

use crate::output::FrameFormat;

/// The frame `std` uses to mark where a thread's own code begins.
///
/// Matched as a prefix, because it is generic and the demangled name carries
/// the instantiation: `std::sys::backtrace::__rust_begin_short_backtrace::<fn()
/// -> core::result::Result<(), alloc::boxed::Box<dyn core::error::Error>>, ...>`.
const RUNTIME_ENTRY: &str = "std::sys::backtrace::__rust_begin_short_backtrace";

/// Name prefixes of the frames between a program and this crate's shim.
///
/// Every entry was taken from rendered output rather than from the source of
/// `alloc`, because what matters is the text the demangler produces, and the
/// two manglings do not agree about it. v0 renders an inherent method on a
/// generic type as `<alloc::raw_vec::RawVecInner>::try_allocate_in`, with the
/// path inside the angle brackets; legacy renders the same function as
/// `alloc::raw_vec::RawVecInner::try_allocate_in`, without them. Both are live
/// — a current toolchain emits v0 and the MSRV emits legacy — and both forms
/// are checked against a real profile from each **\[measured, 1.96 and
/// current\]**.
///
/// A trait method is matched against the trait's path, not the whole name.
/// `<u32 as alloc::vec::spec_from_elem::SpecFromElem>::from_elem` puts the
/// *`Self`* type first, so no prefix drawn from the trait's path can ever reach
/// it — see [`on_allocation_path`], which retries after the `" as "`. Without
/// that, the entries below happened to work only where `Self` was itself on the
/// path (`<alloc::alloc::Global as core::alloc::Allocator>::allocate`), and the
/// one real rendering the rule could not handle was the one nobody had written
/// a case for.
///
/// `heapscope` is here as a second line of defence. This crate's own frames are
/// meant to be gone before a stack is ever interned, by the startup calibration
/// in `unwind` — and if that calibration is wrong, its own two tests fail,
/// loudly, before this list ever hides anything. Verified by breaking the
/// calibration in both directions: `alloc`'s
/// `a_recorded_allocation_starts_at_the_code_that_made_it` and `unwind`'s
/// `the_calibrated_skip_lands_on_the_code_that_allocated` both fail, and they
/// read raw captured addresses, entirely upstream of this **\[measured\]**.
const ALLOCATION_PATH: &[&str] = &[
    // The compiler-generated shims that call a `#[global_allocator]`. Named
    // `__rust_alloc` historically and `__rustc::__rust_alloc` currently; both
    // are live, because a profile can be read from either toolchain.
    "__rust_alloc",
    "__rust_realloc",
    "__rustc::__rust_alloc",
    "__rustc::__rust_realloc",
    // `alloc::alloc::{alloc, alloc_zeroed, realloc, exchange_malloc}`,
    // `<alloc::alloc::Global>::alloc_impl_runtime`, and
    // `<alloc::alloc::Global as core::alloc::Allocator>::{allocate, grow}`.
    "alloc::alloc::",
    "<alloc::alloc::",
    // The growth machinery every collection reaches through.
    "alloc::raw_vec::",
    "<alloc::raw_vec::",
    // What a `Box::new` actually looks like once the compiler is done with it,
    // and what a `BTreeMap` node allocation looks like. `Box::new` itself is
    // `#[inline]` and never appears as a frame.
    "alloc::boxed::box_new",
    "<alloc::boxed::Box<",
    // This crate.
    "heapscope::",
    "<heapscope::",
];

/// Traits whose implementations only ever live inside `alloc`.
///
/// Matched against the trait's path in `<Self as Trait>::method`, which
/// [`ALLOCATION_PATH`] deliberately is not. A demangled trait method names the
/// impl, and an impl lives with `Self` as often as with the trait, so "the
/// trait is in `alloc`" does not mean "the function is `alloc`'s":
/// `<program::MyVec as alloc::raw_vec::Grow>::grow` would be the *program's*
/// code, and trimming it would remove the answer.
///
/// The entries here are exempt because nothing outside `alloc` can implement
/// them. They are private specialisation traits — unnameable, unstable, and
/// existing only so that `vec![x; n]` can pick a memcpy or a `calloc` — so
/// every impl is `alloc`'s by construction and the `Self` type is incidental.
///
/// The list is what a real profile produced, and nothing else. A first draft
/// had four entries taken from the names in `alloc`'s source —
/// `spec_from_elem`, `spec_extend`, `spec_from_iter`, `spec_from_iter_nested` —
/// and a profile of `collect`, `extend`, `Vec::from_iter`, `to_vec`, and
/// `into_boxed_slice` showed that three of them never lead a stack at all,
/// while one the source reading had missed does **\[measured\]**. What survives
/// is the two that were observed:
///
/// - `SpecFromElem`, which `vec![x; n]` dispatches to, above the
///   `alloc::vec::from_elem` frame that is the user's own expression.
/// - `ConvertVec`, which `[T]::to_vec` dispatches to; the public `to_vec` is
///   `#[inline]` and never appears.
///
/// `<alloc::vec::Vec<T> as core::iter::traits::collect::FromIterator<T>>
/// ::from_iter` is deliberately absent: `collect()` is what the program wrote,
/// and `FromIterator` is a trait anyone may implement.
const MACHINERY_TRAITS: &[&str] = &[
    "alloc::vec::spec_from_elem::",
    "<[_]>::to_vec_in::ConvertVec",
];

/// Wraps a renderer, hiding the frames that say nothing about where an
/// allocation came from.
///
/// ```text
/// 0x1021e39c0: __rustc::__rust_alloc+0x38 (…)                    ← hidden
/// 0x102261bc4: <alloc::raw_vec::RawVecInner>::try_allocate_in (…) ← hidden
/// 0x10221b0f4: <alloc::raw_vec::RawVecInner>::with_capacity_in(…) ← hidden
/// 0x10220b220: <alloc::vec::Vec<u8>>::with_capacity+0x24 (…)
/// 0x1021e4134: profile_a_program::churn+0x90 (…)
/// 0x1021e3ed0: profile_a_program::main+0x174 (…)
/// 0x1021e4a08: <fn() -> … as core::ops::function::FnOnce<()>>::call_once (…)
/// 0x1021e4e80: std::sys::backtrace::__rust_begin_short_backtrace (…) ← hidden
/// 0x1021e27f0: std::rt::lang_start::<…>::{closure#0}+0x1c (…)        ← hidden
/// 0x102255ca4: std::rt::lang_start_internal+0x3b8 (…)                ← hidden
/// 0x1021e27c8: std::rt::lang_start::<…>+0x54 (…)                     ← hidden
/// 0x1021e41a4: main+0x24 (…)                                         ← hidden
/// 0x187a484e4: start+0x1b50 (/usr/lib/dyld+0x1801344e4)              ← hidden
/// ```
///
/// It renders exactly what the wrapped format renders; the only thing it
/// changes is which frames are asked for. The rules, and the reasoning behind
/// each, are in the [module documentation](self).
///
/// # This hides information
///
/// A trimmed profile has frames in it that the process recorded and the file
/// does not carry. The count is written into the profile's own `heapscope`
/// section as `trimmedFrames` and into the text summary as a line of its own,
/// so a reader is never left to work out *that* something is missing — but the
/// number is one scalar for the whole file, and these are the specific things a
/// trimmed profile can no longer answer:
///
/// - **Whether a block was zeroed.** `alloc` and `alloc_zeroed` differ only in
///   the leading run, so two points that a program cannot tell apart become one
///   — which is arguably the right answer for a call-site-attributed profiler,
///   and is why PLAN.md section 6.8 puts the zeroed/realloc/alignment
///   histograms in the native format rather than leaving them to be inferred
///   from frame names.
/// - **Whether an allocation was a growth or a fresh one** (`__rust_realloc`,
///   `finish_grow`), for the same reason.
/// - **Which points lost frames, and how many.** Only the total is recorded.
/// - The `image + offset` of the removed frames. That is the one place the
///   README's "every frame stays resolvable afterwards" does not hold, and it
///   holds for every frame the file still carries.
///
/// [`Snapshot::write_dhat_v2_with`](crate::Snapshot::write_dhat_v2_with) with a
/// bare [`Symbolized`](super::Symbolized) is the rendering that keeps them.
#[derive(Clone, Copy, Debug, Default)]
pub struct Trimmed<F> {
    inner: F,
}

impl<F> Trimmed<F> {
    /// Renders with `inner`, showing only the frames worth showing.
    pub fn new(inner: F) -> Self {
        Self { inner }
    }
}

impl<F: FrameFormat> FrameFormat for Trimmed<F> {
    fn format(&self, address: usize, out: &mut String) {
        self.inner.format(address, out);
    }

    /// The narrower of these rules and whatever `F` already wanted.
    ///
    /// Intersected rather than replaced, because a decorator that discards the
    /// decision it wraps is not a decorator: `Trimmed<F>` would silently undo
    /// an `F` that trims for reasons of its own, and the failure would look
    /// like frames reappearing for no reason.
    fn keep(&self, frames: &[String]) -> Range<usize> {
        let inner = self.inner.keep(frames);
        let mine = worth_showing(frames);
        inner.start.max(mine.start)..inner.end.min(mine.end)
    }
}

/// The subrange of `frames` — innermost first, as rendered — worth showing.
///
/// Never empty for a non-empty stack. A point with no frames at all is labelled
/// `[unwalkable]` by the emitter, and a stack that *was* walked must not be
/// reduced to a claim that it could not be.
///
/// Public because [`Trimmed`] is not the only way to want these rules: a
/// [`FrameFormat`] that renders in some other shape, or that has trimming of
/// its own to combine with these, can call this directly from its
/// [`keep`](FrameFormat::keep) rather than reimplementing the two cuts.
pub fn worth_showing(frames: &[String]) -> Range<usize> {
    if frames.is_empty() {
        return 0..0;
    }

    // The innermost runtime marker, and everything outside it, goes. Innermost
    // rather than outermost because each marker is a boundary of the same kind:
    // between two of them lies thread-spawning machinery, which is no more
    // interesting than what lies beyond the last one.
    let end = frames
        .iter()
        .position(|frame| name_of(frame).is_some_and(|name| name.starts_with(RUNTIME_ENTRY)))
        .unwrap_or(frames.len())
        // An allocation made inside the marker frame itself would leave nothing
        // at all. Keeping it says where the allocation was, which is more than
        // an empty stack says.
        .max(1);

    // The leading run of allocation-path frames, stopping one short of the end
    // so that a stack which is *entirely* allocation path — a profiler
    // measuring its own machinery — still names something.
    let mut start = 0;
    while start + 1 < end && name_of(&frames[start]).is_some_and(on_allocation_path) {
        start += 1;
    }

    start..end
}

/// The part of a rendered frame that names code, or `None`.
///
/// Frames are rendered `0x1044c81f0: name (image+0x2c1f0)`, the shape Valgrind
/// uses, so the name begins after the first `": "`. A [`FrameFormat`] that
/// produces some other shape yields `None` here and is left entirely alone,
/// which is the right answer: trimming a rendering we cannot read would be
/// guessing.
fn name_of(frame: &str) -> Option<&str> {
    frame.split_once(": ").map(|(_, name)| name)
}

/// Whether `name` is one of the frames between a program and this crate's shim.
///
/// Two attempts, because a demangled name has two places a path can start. An
/// inherent method leads with it — `<alloc::raw_vec::RawVecInner>::grow_one` —
/// but a *trait* method leads with the `Self` type instead, and the trait's
/// path comes after `" as "`:
///
/// ```text
/// <u32 as alloc::vec::spec_from_elem::SpecFromElem>::from_elem::<alloc::alloc::Global>
///        ^ the only place a prefix from ALLOCATION_PATH can match
/// ```
///
/// So the second attempt is what makes trait methods reachable at all. It is
/// deliberately not a search for the prefix *anywhere* in the name: a frame of
/// the program's own that merely mentions `alloc::raw_vec` in a generic
/// argument is not on the allocation path, and matching it would remove the
/// answer.
fn on_allocation_path(name: &str) -> bool {
    if ALLOCATION_PATH
        .iter()
        .any(|prefix| name.starts_with(prefix))
    {
        return true;
    }
    let Some((_, trait_path)) = name
        .strip_prefix('<')
        .and_then(|name| name.split_once(" as "))
    else {
        return false;
    };
    MACHINERY_TRAITS
        .iter()
        .any(|prefix| trait_path.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a stack in the shape the emitter produces: an address, `": "`,
    /// the name, and the image attribution.
    fn stack(names: &[&str]) -> Vec<String> {
        names
            .iter()
            .enumerate()
            .map(|(at, name)| format!("0x{:x}: {name} (/bin/program+0x{:x})", 0x1000 + at, at))
            .collect()
    }

    fn kept(names: &[&str]) -> Vec<String> {
        let frames = stack(names);
        frames[worth_showing(&frames)]
            .iter()
            .map(|frame| {
                let name = name_of(frame).expect("the test builds well-shaped frames");
                name.split_once(" (")
                    .expect("and an image attribution")
                    .0
                    .to_string()
            })
            .collect()
    }

    /// The measured stack from `examples/profile_a_program`, debug build,
    /// macOS aarch64. Verbatim except for shortening the generic arguments.
    const REAL_STACK: &[&str] = &[
        "__rustc::__rust_alloc+0x38",
        "<alloc::raw_vec::RawVecInner>::try_allocate_in+0x9c",
        "<alloc::raw_vec::RawVecInner>::with_capacity_in+0x44",
        "<alloc::vec::Vec<u8>>::with_capacity+0x24",
        "profile_a_program::churn+0x90",
        "profile_a_program::main+0x174",
        "<fn() -> core::result::Result<()> as core::ops::function::FnOnce<()>>::call_once+0x14",
        "std::sys::backtrace::__rust_begin_short_backtrace::<fn() -> core::result::Result<()>>+0x18",
        "std::rt::lang_start::<core::result::Result<()>>::{closure#0}+0x1c",
        "std::rt::lang_start_internal+0x3b8",
        "std::rt::lang_start::<core::result::Result<()>>+0x54",
        "main+0x24",
        "start+0x1b50",
    ];

    #[test]
    fn a_real_stack_keeps_the_program_and_drops_the_rest() {
        assert_eq!(
            kept(REAL_STACK),
            [
                "<alloc::vec::Vec<u8>>::with_capacity+0x24",
                "profile_a_program::churn+0x90",
                "profile_a_program::main+0x174",
                // The `FnOnce` shim survives, because `std` shows it too: the
                // cut is *at* the marker, not before it.
                "<fn() -> core::result::Result<()> as core::ops::function::FnOnce<()>>::call_once+0x14",
            ]
        );
    }

    /// The thread case, which is where the rule earns the most: nine frames of
    /// spawn machinery below one marker.
    #[test]
    fn a_spawned_thread_loses_its_whole_entry_sequence() {
        assert_eq!(
            kept(&[
                "__rustc::__rust_alloc+0x38",
                "<alloc::vec::Vec<u8>>::with_capacity+0x24",
                "probe::allocate_on_a_thread+0x20",
                "probe::main::{closure#0}+0x10",
                "std::sys::backtrace::__rust_begin_short_backtrace::<probe::main::{closure#0}>",
                "std::thread::lifecycle::spawn_unchecked::<probe::main::{closure#0}>",
                "<core::panic::unwind_safe::AssertUnwindSafe<…>>::call_once",
                "std::panicking::catch_unwind::do_call::<…>",
                "__rust_try+0x20",
                "std::thread::lifecycle::spawn_unchecked::<probe::main::{closure#0}>",
                "<std::sys::thread::unix::Thread>::new::thread_start+0x198",
                "_pthread_start+0x88",
                "thread_start+0x8",
            ]),
            [
                "<alloc::vec::Vec<u8>>::with_capacity+0x24",
                "probe::allocate_on_a_thread+0x20",
                "probe::main::{closure#0}+0x10",
            ]
        );
    }

    /// The reason the top rule is a leading run and not a search: a real
    /// profile has a `heapscope` frame in the middle of a stack, and it is the
    /// answer to where the allocation came from.
    #[test]
    fn only_the_leading_run_of_the_allocation_path_is_removed() {
        assert_eq!(
            kept(&[
                "__rustc::__rust_alloc+0x38",
                "<std::sys::sync::once_box::OnceBox<…>>::initialize::<…>",
                "<std::io::stdio::Stdout>::lock+0xd8",
                "<heapscope::profiler::Profiler>::print_summary+0x60",
                "profile_a_program::main+0x274",
            ]),
            [
                "<std::sys::sync::once_box::OnceBox<…>>::initialize::<…>",
                "<std::io::stdio::Stdout>::lock+0xd8",
                "<heapscope::profiler::Profiler>::print_summary+0x60",
                "profile_a_program::main+0x274",
            ]
        );
    }

    /// Every shape a frame on the allocation path is actually written in.
    ///
    /// All measured from real profiles rather than read off `alloc`'s source,
    /// and the four groups exist because each is reached by a different part of
    /// the matching. The `<Self as Trait>` group is the one that matters most:
    /// a rule built only from the first three passes every test anyone would
    /// think to write and still cannot touch the form `vec![x; n]` produces,
    /// because that name leads with `u32` and not with a path.
    #[test]
    fn every_form_the_allocation_path_is_written_in_is_recognised() {
        for name in [
            // A bare path: the compiler's shims, both spellings, because a
            // profile can be read from either toolchain.
            "__rust_alloc+0x38",
            "__rust_alloc_zeroed+0x38",
            "__rust_realloc+0x38",
            "__rustc::__rust_alloc+0x38",
            "__rustc::__rust_alloc_zeroed+0x38",
            "__rustc::__rust_realloc+0x38",
            "alloc::alloc::exchange_malloc+0x10",
            "alloc::raw_vec::finish_grow::<alloc::alloc::Global>+0xb8",
            "alloc::boxed::box_new_uninit+0x3c",
            "heapscope::alloc::record+0x8",
            // A path inside angle brackets: v0's rendering of an inherent
            // method on a generic type.
            "<alloc::alloc::Global>::alloc_impl_runtime+0xa0",
            "<alloc::raw_vec::RawVec<u64>>::grow_one+0x44",
            "<alloc::raw_vec::RawVecInner>::try_allocate_in+0x9c",
            "<alloc::boxed::Box<alloc::collections::btree::node::LeafNode<u32, u32>>>\
             ::try_new_uninit_in+0x38",
            // A trait method whose `Self` type is itself on the path.
            "<alloc::alloc::Global as core::alloc::Allocator>::allocate+0x34",
            "<heapscope::alloc::Alloc as core::alloc::GlobalAlloc>::alloc+0x20",
            // A trait method whose `Self` type is *not*: `u32` first, the trait
            // second. Unreachable by any prefix taken from the trait's path
            // until `on_allocation_path` learned to look after the `" as "`.
            "<u32 as alloc::vec::spec_from_elem::SpecFromElem>::from_elem::\
             <alloc::alloc::Global>+0x88",
            "<u8 as <[_]>::to_vec_in::ConvertVec>::to_vec::<alloc::alloc::Global>+0x40",
        ] {
            assert_eq!(
                kept(&[name, "program::allocates+0x4"]),
                ["program::allocates+0x4"],
                "`{name}` was left in place"
            );
        }
    }

    /// The frames one step further out, which are what the program wrote.
    ///
    /// Each is one frame from the pair its neighbour above is machinery in, so
    /// together with the test above this pins the boundary rather than one side
    /// of it. `alloc::vec::from_elem` is `vec![x; n]`; `collect()` is
    /// `FromIterator::from_iter`, a trait anyone may implement, so matching the
    /// trait path there would take a frame the program named. All measured.
    #[test]
    fn the_first_frame_the_program_named_is_never_removed() {
        for name in [
            "<alloc::vec::Vec<u8>>::with_capacity+0x24",
            "<alloc::vec::Vec<u64>>::push+0x1c",
            "<alloc::vec::Vec<u32>>::reserve+0x9c",
            "<alloc::string::String>::with_capacity+0x24",
            "alloc::vec::from_elem::<u32>+0x18",
            "<alloc::vec::Vec<u64> as core::iter::traits::collect::FromIterator<u64>>\
             ::from_iter::<core::ops::range::Range<u64>>+0x24",
            "alloc::fmt::format::format_inner+0x108",
        ] {
            assert_eq!(
                kept(&["__rustc::__rust_alloc+0x38", name, "program::allocates+0x4"]),
                [name, "program::allocates+0x4"],
                "`{name}` was removed, and it is what the program wrote"
            );
        }
    }

    /// A trait method names an *impl*, and an impl lives with `Self` as often
    /// as with the trait. So "the trait is in `alloc`" cannot mean "the
    /// function is `alloc`'s": here it would be the program's own code, and
    /// removing it would remove the answer.
    ///
    /// This is why the `" as "` retry consults `MACHINERY_TRAITS` — traits
    /// nothing outside `alloc` can implement — and not `ALLOCATION_PATH`.
    #[test]
    fn a_program_implementing_an_alloc_trait_keeps_its_own_frame() {
        for name in [
            "<program::MyVec as alloc::raw_vec::Grow>::grow+0x4",
            "<program::Pool as alloc::alloc::Fill>::fill+0x4",
            "<program::Format as heapscope::output::FrameFormat>::format+0x4",
        ] {
            assert_eq!(
                kept(&[name, "program::main+0x4"]).len(),
                2,
                "`{name}` is the program's own code and was removed"
            );
        }
    }

    /// A frame that merely mentions one of the prefixes is not on the path.
    /// The rule is what a frame *is*, which is where its name starts.
    #[test]
    fn a_name_that_only_contains_a_prefix_is_not_matched() {
        for name in [
            "program::calls_alloc::alloc::wrapper+0x4",
            "<program::MyVec as alloc::raw_vec::Grow>::grow+0x4",
            "not_heapscope::thing+0x4",
        ] {
            assert_eq!(kept(&[name, "program::main+0x4"]).len(), 2, "{name}");
        }
    }

    /// A stripped build renders every frame `0x…: ???`, so there is nothing to
    /// read and nothing may be removed.
    #[test]
    fn a_stack_with_no_names_is_left_exactly_as_it_was() {
        let frames: Vec<String> = (0..6)
            .map(|at| format!("0x{at:x}: ??? (/bin/program+0x{at:x})"))
            .collect();
        assert_eq!(worth_showing(&frames), 0..6);
    }

    /// A `FrameFormat` this crate did not write need not produce the shape this
    /// module reads. Trimming a rendering it cannot parse would be guessing.
    #[test]
    fn a_rendering_in_an_unknown_shape_is_left_alone() {
        let frames: Vec<String> = ["main at prog.rs:1", "churn at prog.rs:9"]
            .iter()
            .map(|frame| String::from(*frame))
            .collect();
        assert_eq!(worth_showing(&frames), 0..2);
    }

    #[test]
    fn an_empty_stack_stays_empty() {
        assert_eq!(worth_showing(&[]), 0..0);
    }

    /// Trimming must never turn a stack that was walked into one that looks as
    /// though it was not: the emitter labels an empty frame list `[unwalkable]`.
    #[test]
    fn a_stack_is_never_trimmed_to_nothing() {
        // Entirely allocation path.
        assert_eq!(
            kept(&[
                "__rustc::__rust_alloc+0x38",
                "<alloc::raw_vec::RawVecInner>::try_allocate_in+0x9c",
                "alloc::alloc::exchange_malloc+0x10",
            ]),
            ["alloc::alloc::exchange_malloc+0x10"]
        );
        // The marker is the innermost frame, so the bottom rule would take
        // everything.
        assert_eq!(
            kept(&[
                "std::sys::backtrace::__rust_begin_short_backtrace::<f>",
                "std::rt::lang_start_internal+0x3b8",
            ]),
            ["std::sys::backtrace::__rust_begin_short_backtrace::<f>"]
        );
        // And both rules at once, on a single frame.
        assert_eq!(
            kept(&["__rustc::__rust_alloc+0x38"]),
            ["__rustc::__rust_alloc+0x38"]
        );
    }

    /// A stack can carry two markers if a thread entry is itself inlined into
    /// one. The innermost is the boundary; what lies between them is spawn
    /// machinery, which is no more interesting than what lies beyond.
    #[test]
    fn the_innermost_runtime_marker_is_the_boundary() {
        assert_eq!(
            kept(&[
                "program::allocates+0x4",
                "std::sys::backtrace::__rust_begin_short_backtrace::<inner>",
                "std::thread::lifecycle::spawn_unchecked::<inner>",
                "std::sys::backtrace::__rust_begin_short_backtrace::<outer>",
                "std::rt::lang_start_internal+0x3b8",
            ]),
            ["program::allocates+0x4"]
        );
    }

    /// The whole point of putting this behind a decorator: choosing it changes
    /// which frames are asked for and nothing else about the rendering.
    #[test]
    fn wrapping_a_renderer_does_not_change_what_it_renders() {
        use crate::output::RawAddresses;

        let bare = RawAddresses;
        let trimmed = Trimmed::new(RawAddresses);
        for address in [0, 0x1000, usize::MAX] {
            let (mut left, mut right) = (String::new(), String::new());
            bare.format(address, &mut left);
            trimmed.format(address, &mut right);
            assert_eq!(left, right);
        }
    }

    /// The default on the trait keeps everything, so a `FrameFormat` written
    /// before this module existed behaves exactly as it did.
    #[test]
    fn a_renderer_that_says_nothing_about_trimming_keeps_every_frame() {
        use crate::output::RawAddresses;
        let frames = stack(REAL_STACK);
        assert_eq!(RawAddresses.keep(&frames), 0..frames.len());
    }

    /// The name begins after the **first** `": "`, and it has to: the image
    /// attribution that follows a name is a filesystem path, and a directory
    /// may be called anything.
    ///
    /// Measured, not imagined — running the example from a directory named
    /// `odd: dir` produces exactly this. Reading from the last `": "` instead
    /// leaves `dir/program+0x1000)`, which matches no rule, so trimming
    /// silently stops working for anyone whose build directory has a colon in
    /// it. That mutation left the whole suite green until this existed.
    #[test]
    fn the_name_is_read_before_the_image_however_the_image_is_spelled() {
        let awkward = String::from(
            "0x1000: __rustc::__rust_alloc+0x38 (/Users/someone/odd: dir/program+0x1000)",
        );
        let site =
            String::from("0x1001: program::allocates+0x4 (/Users/someone/odd: dir/program+0x1001)");
        let frames = vec![awkward, site.clone()];
        assert_eq!(
            worth_showing(&frames),
            1..2,
            "the allocation-path frame was not recognised, so the name was read \
             from the wrong side of the path"
        );

        // And the same for the runtime marker at the bottom.
        let marker = String::from(
            "0x1002: std::sys::backtrace::__rust_begin_short_backtrace::<f> \
             (/Users/someone/odd: dir/program+0x1002)",
        );
        assert_eq!(worth_showing(&[site, marker]), 0..1);
    }

    /// `Trimmed<F>` narrows what `F` already decided rather than replacing it.
    ///
    /// A decorator that discards the decision it wraps is not a decorator, and
    /// the failure would be invisible: frames reappearing for no stated reason.
    #[test]
    fn wrapping_a_renderer_that_already_trims_narrows_rather_than_overrules() {
        use crate::output::RawAddresses;

        /// Hides the innermost two frames, for reasons of its own.
        struct HidesTwo;
        impl FrameFormat for HidesTwo {
            fn format(&self, address: usize, out: &mut String) {
                RawAddresses.format(address, out);
            }
            fn keep(&self, frames: &[String]) -> Range<usize> {
                2.min(frames.len())..frames.len()
            }
        }

        let frames = stack(REAL_STACK);
        let mine = worth_showing(&frames);
        assert_eq!(mine, 3..7, "the rules on their own");
        assert_eq!(
            HidesTwo.keep(&frames),
            2..13,
            "the inner renderer on its own"
        );
        assert_eq!(
            Trimmed::new(HidesTwo).keep(&frames),
            3..7,
            "the narrower of the two, on both ends"
        );

        /// Hides everything except the outermost frame, which is narrower than
        /// these rules at the *start* and wider at the end.
        struct KeepsLast;
        impl FrameFormat for KeepsLast {
            fn format(&self, address: usize, out: &mut String) {
                RawAddresses.format(address, out);
            }
            fn keep(&self, frames: &[String]) -> Range<usize> {
                frames.len() - 1..frames.len()
            }
        }
        // 12..13 intersected with 3..7 is empty; the emitter's `clamp_frames`
        // is what turns that into a frame rather than an `[unwalkable]` claim.
        let intersected = Trimmed::new(KeepsLast).keep(&frames);
        assert!(intersected.start >= intersected.end);
    }
}
