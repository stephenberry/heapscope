//! What the program asked for, beyond a number of bytes.
//!
//! A DHAT v2 file has one number per program point per quantity, and that is the
//! whole of what it can say about an allocation: how many bytes, how many
//! blocks. Three things the shim already knows never reach it.
//!
//! - **The alignment.** `GlobalAlloc` is handed a [`Layout`], and a program that
//!   asks for 64-byte alignment a million times is paying for it in ways a byte
//!   count does not show.
//! - **Whether the block was requested zeroed.** `calloc` may hand back pages
//!   that are never faulted in, so a run whose bytes are mostly zeroed has a
//!   resident size unrelated to its allocated size — which is the first thing a
//!   reader gets wrong when a profile and `ps` disagree.
//! - **What reallocation cost.** A `Vec` that grows by doubling from 8 bytes to
//!   8 MB copies just under 8 MB along the way, and every one of those copies is
//!   invisible in a profile that records only the sizes that were asked for.
//!
//! # Distributions, not just totals
//!
//! `total_bytes / total_blocks` is a mean, and a mean is the wrong summary of
//! allocation sizes: a program making a million 24-byte allocations and one
//! 24 MB allocation has the same mean as one making two million 24-byte
//! allocations, and they are not the same program. So sizes and alignments are
//! bucketed by power of two, which is the resolution at which the answer is
//! actionable ("this is a small-object workload") without being a second copy of
//! the data.
//!
//! # Blocks per class, not bytes
//!
//! Each class counts blocks only. A second array would be a second atomic
//! increment on the allocation path to produce a number the reader can already
//! bound — a class holds sizes in `[2^(k-1), 2^k)`, so its blocks times those
//! two ends brackets its bytes — while the exact byte total across all classes
//! is [`GlobalStats::total_bytes`](super::engine::GlobalStats::total_bytes)
//! already.

use std::alloc::Layout;
use std::sync::atomic::{AtomicU64, Ordering};

/// Power-of-two size classes, one per bit of a `usize` plus one for zero.
pub const SIZE_CLASSES: usize = usize::BITS as usize + 1;

/// Power-of-two alignment classes, one per bit of a `usize`.
pub const ALIGN_CLASSES: usize = usize::BITS as usize;

/// One allocation request, as the program made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Bytes requested.
    pub size: usize,
    /// Alignment requested, in bytes.
    ///
    /// [`Layout`] guarantees a non-zero power of two. Nothing here depends on
    /// that holding: see [`align_class`], which reports a true statement about
    /// any value at all.
    pub align: usize,
    /// Whether the program asked for the block to be zeroed, meaning it arrived
    /// through `alloc_zeroed` rather than `alloc`.
    pub zeroed: bool,
}

impl Shape {
    /// `size` bytes, byte-aligned, not zeroed.
    ///
    /// The shape of an allocation described by its size alone, which is what a
    /// test driving the engine directly has.
    pub const fn of(size: usize) -> Self {
        Self {
            size,
            align: 1,
            zeroed: false,
        }
    }

    /// The layout a [`GlobalAlloc`](std::alloc::GlobalAlloc) method was handed.
    ///
    /// `zeroed` is the caller's to state because it is not part of the layout:
    /// `alloc` and `alloc_zeroed` receive identical `Layout`s and differ only in
    /// which method the program called.
    pub const fn of_layout(layout: Layout, zeroed: bool) -> Self {
        Self {
            size: layout.size(),
            align: layout.align(),
            zeroed,
        }
    }

    /// The same request, aligned to `align`.
    pub const fn aligned(mut self, align: usize) -> Self {
        self.align = align;
        self
    }

    /// The same request, asked for zeroed.
    pub const fn zeroed(mut self) -> Self {
        self.zeroed = true;
        self
    }
}

/// One reallocation, as the shim observed it.
///
/// A struct rather than five positional arguments because four of them are
/// addresses and sizes of the same type, in two old/new pairs, and transposing
/// a pair produces numbers that are wrong without being obviously wrong: a
/// profile reporting that every `Vec` shrank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Realloc {
    /// Where the block was.
    pub old_address: usize,
    /// How large it was.
    pub old_size: usize,
    /// Where it is now. Equal to `old_address` when the allocator resized in
    /// place.
    pub new_address: usize,
    /// What the program asked for this time.
    pub new: Shape,
}

impl Realloc {
    /// Whether the allocator had to move the block, and therefore to copy it.
    pub fn moved(&self) -> bool {
        self.old_address != self.new_address
    }

    /// Bytes the allocator copied, which is what survived the move.
    ///
    /// Zero for a resize in place, which copies nothing. For a move it is
    /// `min(old, new)`: growing copies everything that was there, and shrinking
    /// copies only what fits.
    pub fn bytes_copied(&self) -> u64 {
        if self.moved() {
            self.old_size.min(self.new.size) as u64
        } else {
            0
        }
    }
}

/// Which power-of-two class `size` falls in.
///
/// Class 0 is a zero-byte request and nothing else. Class `k` covers
/// `2^(k-1)..2^k`, so class 1 is one byte, class 4 is 8 through 15, and the
/// class is also the number of bits the size occupies.
pub const fn size_class(size: usize) -> usize {
    (usize::BITS - size.leading_zeros()) as usize
}

/// The smallest size in `class`.
pub const fn size_class_floor(class: usize) -> usize {
    if class == 0 {
        0
    } else {
        1 << (class - 1)
    }
}

/// The largest size in `class`.
pub const fn size_class_ceiling(class: usize) -> usize {
    if class == 0 {
        0
    } else if class >= SIZE_CLASSES - 1 {
        usize::MAX
    } else {
        (1 << class) - 1
    }
}

/// Which power-of-two class `align` falls in.
///
/// The exponent of the largest power of two that divides `align`, which is a
/// true statement about every input rather than only about the powers of two
/// [`Layout`] promises. An alignment of 12 is reported as 4-byte alignment,
/// because a block aligned to 12 *is* aligned to 4; an alignment of 0 is
/// reported as class 0, because no alignment requirement is byte alignment.
/// Neither can occur through this crate's own paths, and neither produces a
/// number that says something untrue if it does.
pub const fn align_class(align: usize) -> usize {
    if align == 0 {
        0
    } else {
        align.trailing_zeros() as usize
    }
}

/// The alignment `class` stands for, in bytes.
pub const fn align_class_bytes(class: usize) -> usize {
    1usize << class
}

/// Running counts of what the program asked for.
///
/// Const-initializable, like everything else the shim reaches, so the engine
/// stays a plain `static`.
#[derive(Debug)]
pub struct Shapes {
    observed: AtomicU64,
    sizes: [AtomicU64; SIZE_CLASSES],
    alignments: [AtomicU64; ALIGN_CLASSES],
    zeroed_blocks: AtomicU64,
    zeroed_bytes: AtomicU64,
    reallocs: AtomicU64,
    reallocs_moved: AtomicU64,
    bytes_copied: AtomicU64,
    bytes_grown: AtomicU64,
    bytes_shrunk: AtomicU64,
}

impl Shapes {
    /// Zeroed counts.
    pub const fn new() -> Self {
        Self {
            observed: AtomicU64::new(0),
            sizes: [const { AtomicU64::new(0) }; SIZE_CLASSES],
            alignments: [const { AtomicU64::new(0) }; ALIGN_CLASSES],
            zeroed_blocks: AtomicU64::new(0),
            zeroed_bytes: AtomicU64::new(0),
            reallocs: AtomicU64::new(0),
            reallocs_moved: AtomicU64::new(0),
            bytes_copied: AtomicU64::new(0),
            bytes_grown: AtomicU64::new(0),
            bytes_shrunk: AtomicU64::new(0),
        }
    }

    /// Counts one allocation request.
    ///
    /// Three relaxed increments for an ordinary allocation — the request count
    /// and one bucket in each array — and two more for one asked for zeroed.
    /// The class arrays spread the traffic that the engine's single
    /// `total_bytes` word concentrates, because a program allocating a mix of
    /// sizes touches a mix of words; `observed` does not, and is one more
    /// contended line on the hot path in exchange for the invariant that ties
    /// the histograms to the totals.
    #[inline]
    pub fn record(&self, shape: Shape) {
        self.observed.fetch_add(1, Ordering::Relaxed);
        // Both indices are in range by construction — `size_class` cannot exceed
        // the bit width and `align_class` counts trailing zeros of a non-zero
        // value — so this is an unconditional store rather than a bounds check
        // whose failure branch nothing could reach.
        self.sizes[size_class(shape.size)].fetch_add(1, Ordering::Relaxed);
        self.alignments[align_class(shape.align)].fetch_add(1, Ordering::Relaxed);
        if shape.zeroed {
            self.zeroed_blocks.fetch_add(1, Ordering::Relaxed);
            self.zeroed_bytes
                .fetch_add(shape.size as u64, Ordering::Relaxed);
        }
    }

    /// Counts one reallocation.
    ///
    /// Separate from [`Shapes::record`], which the caller also invokes for the
    /// resulting block: a reallocation both produces a block of the new shape
    /// and is an event with a cost of its own.
    #[inline]
    pub fn record_realloc(&self, realloc: &Realloc) {
        self.reallocs.fetch_add(1, Ordering::Relaxed);
        if realloc.moved() {
            self.reallocs_moved.fetch_add(1, Ordering::Relaxed);
            self.bytes_copied
                .fetch_add(realloc.bytes_copied(), Ordering::Relaxed);
        }
        let old = realloc.old_size as u64;
        let new = realloc.new.size as u64;
        if new > old {
            self.bytes_grown.fetch_add(new - old, Ordering::Relaxed);
        } else {
            self.bytes_shrunk.fetch_add(old - new, Ordering::Relaxed);
        }
    }

    /// Reads the current counts.
    pub fn snapshot(&self) -> ShapeStats {
        let mut stats = ShapeStats {
            observed_blocks: self.observed.load(Ordering::Relaxed),
            zeroed_blocks: self.zeroed_blocks.load(Ordering::Relaxed),
            zeroed_bytes: self.zeroed_bytes.load(Ordering::Relaxed),
            reallocs: self.reallocs.load(Ordering::Relaxed),
            reallocs_moved: self.reallocs_moved.load(Ordering::Relaxed),
            bytes_copied: self.bytes_copied.load(Ordering::Relaxed),
            bytes_grown: self.bytes_grown.load(Ordering::Relaxed),
            bytes_shrunk: self.bytes_shrunk.load(Ordering::Relaxed),
            sizes: [0; SIZE_CLASSES],
            alignments: [0; ALIGN_CLASSES],
        };
        for (out, counter) in stats.sizes.iter_mut().zip(&self.sizes) {
            *out = counter.load(Ordering::Relaxed);
        }
        for (out, counter) in stats.alignments.iter_mut().zip(&self.alignments) {
            *out = counter.load(Ordering::Relaxed);
        }
        stats
    }
}

impl Default for Shapes {
    fn default() -> Self {
        Self::new()
    }
}

/// A reading of [`Shapes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeStats {
    /// Allocation requests whose shape was counted.
    ///
    /// In a heap run this is
    /// `total_blocks + dropped_blocks`: every request is counted here, including
    /// the ones the live-block table had no room to track. In a mode that
    /// records no allocations it is zero, and `total_blocks` counts events
    /// instead.
    pub observed_blocks: u64,
    /// Blocks per power-of-two size class. See [`size_class`].
    pub sizes: [u64; SIZE_CLASSES],
    /// Blocks per power-of-two alignment class. See [`align_class`].
    pub alignments: [u64; ALIGN_CLASSES],
    /// Blocks the program asked to have zeroed.
    pub zeroed_blocks: u64,
    /// Bytes in those blocks.
    pub zeroed_bytes: u64,
    /// Reallocations observed, whether or not the block was tracked.
    pub reallocs: u64,
    /// Those the allocator could not satisfy in place.
    pub reallocs_moved: u64,
    /// Bytes the allocator copied moving them.
    pub bytes_copied: u64,
    /// Bytes reallocation added to blocks that grew.
    pub bytes_grown: u64,
    /// Bytes reallocation removed from blocks that shrank.
    pub bytes_shrunk: u64,
}

impl Default for ShapeStats {
    fn default() -> Self {
        Self {
            observed_blocks: 0,
            sizes: [0; SIZE_CLASSES],
            alignments: [0; ALIGN_CLASSES],
            zeroed_blocks: 0,
            zeroed_bytes: 0,
            reallocs: 0,
            reallocs_moved: 0,
            bytes_copied: 0,
            bytes_grown: 0,
            bytes_shrunk: 0,
        }
    }
}

impl ShapeStats {
    /// The size classes that recorded anything, as `(floor, ceiling, blocks)`.
    ///
    /// Empty classes are skipped rather than written out as zeroes: a 64-bit
    /// process has 65 of them and a real program uses a handful, so writing all
    /// of them would put sixty lines of zero into every profile to say nothing.
    pub fn size_classes(&self) -> impl Iterator<Item = (usize, usize, u64)> + '_ {
        self.sizes
            .iter()
            .enumerate()
            .filter(|(_, &blocks)| blocks != 0)
            .map(|(class, &blocks)| (size_class_floor(class), size_class_ceiling(class), blocks))
    }

    /// The alignments that were asked for, as `(bytes, blocks)`.
    pub fn alignments_used(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.alignments
            .iter()
            .enumerate()
            .filter(|(_, &blocks)| blocks != 0)
            .map(|(class, &blocks)| (align_class_bytes(class), blocks))
    }

    /// The size class holding the most blocks, as `(floor, ceiling, blocks)`.
    ///
    /// `None` when nothing was recorded. Ties go to the smaller class, which is
    /// the one a reader is more likely to be able to do something about.
    pub fn commonest_size(&self) -> Option<(usize, usize, u64)> {
        self.size_classes().max_by_key(|&(floor, _, blocks)| {
            // `max_by_key` returns the *last* maximum, so the key orders by
            // descending floor to make the first one win.
            (blocks, std::cmp::Reverse(floor))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The class boundaries are the format: a reader converts a class back to a
    /// range of sizes and the two must be inverses.
    #[test]
    fn a_size_lands_in_the_class_that_claims_it() {
        for size in [0usize, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 1023, 1024, 1025] {
            let class = size_class(size);
            assert!(
                size >= size_class_floor(class) && size <= size_class_ceiling(class),
                "{size} landed in class {class}, which covers {}..={}",
                size_class_floor(class),
                size_class_ceiling(class)
            );
        }
        assert_eq!(size_class(0), 0);
        assert_eq!(size_class_floor(0), 0);
        assert_eq!(size_class_ceiling(0), 0);
        assert_eq!(size_class(1), 1);
        assert_eq!(size_class(8), 4);
        assert_eq!(size_class_floor(4), 8);
        assert_eq!(size_class_ceiling(4), 15);
    }

    /// The largest class has no upper power of two to name, and reporting
    /// `2^64 - 1` as its ceiling is the only honest answer that fits.
    #[test]
    fn the_largest_size_class_is_open_at_the_top() {
        let class = size_class(usize::MAX);
        assert_eq!(class, SIZE_CLASSES - 1);
        assert_eq!(size_class_ceiling(class), usize::MAX);
        assert_eq!(size_class_floor(class), 1 << (usize::BITS - 1));
    }

    /// Every class index a shape can produce has to be a valid index, because
    /// `Shapes::record` indexes with it and no branch stands between.
    #[test]
    fn every_class_a_shape_can_produce_is_in_range() {
        for size in [0, 1, usize::MAX, usize::MAX / 2] {
            assert!(size_class(size) < SIZE_CLASSES);
        }
        for align in [0, 1, 2, 4096, 1 << 63, 12, usize::MAX] {
            assert!(align_class(align) < ALIGN_CLASSES, "align {align}");
        }
    }

    /// `align_class` is documented as reporting the largest power of two that
    /// divides its argument, which has to hold for the inputs `Layout` cannot
    /// produce as well as the ones it can.
    #[test]
    fn an_alignment_is_reported_as_a_divisor_it_really_has() {
        assert_eq!(align_class(1), 0);
        assert_eq!(align_class(8), 3);
        assert_eq!(align_class_bytes(3), 8);
        // Not a power of two: 12 = 4 x 3, so 4 is the true claim.
        assert_eq!(align_class_bytes(align_class(12)), 4);
        // No requirement at all is byte alignment, which every block satisfies.
        assert_eq!(align_class_bytes(align_class(0)), 1);
    }

    #[test]
    fn a_shape_carries_what_the_layout_asked_for() {
        let layout = Layout::from_size_align(48, 16).expect("a valid layout");
        let shape = Shape::of_layout(layout, true);
        assert_eq!(shape.size, 48);
        assert_eq!(shape.align, 16);
        assert!(shape.zeroed);
        assert_eq!(Shape::of(48), Shape::of_layout(layout, false).aligned(1));
    }

    /// The invariant the validator checks: every observed request appears once
    /// in each histogram.
    #[test]
    fn each_request_lands_in_exactly_one_class_of_each_kind() {
        let shapes = Shapes::new();
        for size in [0usize, 1, 24, 24, 1000, 1 << 20] {
            shapes.record(Shape::of(size).aligned(8));
        }
        shapes.record(Shape::of(64).aligned(64));

        let stats = shapes.snapshot();
        assert_eq!(stats.observed_blocks, 7);
        assert_eq!(stats.sizes.iter().sum::<u64>(), 7);
        assert_eq!(stats.alignments.iter().sum::<u64>(), 7);
        assert_eq!(
            stats.alignments_used().collect::<Vec<_>>(),
            [(8, 6), (64, 1)]
        );
        assert_eq!(
            stats.commonest_size(),
            Some((16, 31, 2)),
            "two 24-byte blocks"
        );
    }

    /// Zeroed blocks are counted alongside, not instead: a `calloc`ed block is
    /// still an allocation of its size and alignment.
    #[test]
    fn a_zeroed_request_is_counted_as_both() {
        let shapes = Shapes::new();
        shapes.record(Shape::of(100));
        shapes.record(Shape::of(200).zeroed());

        let stats = shapes.snapshot();
        assert_eq!(stats.observed_blocks, 2);
        assert_eq!(stats.zeroed_blocks, 1);
        assert_eq!(stats.zeroed_bytes, 200);
        assert_eq!(stats.sizes.iter().sum::<u64>(), 2);
    }

    #[test]
    fn a_resize_in_place_copies_nothing() {
        let realloc = Realloc {
            old_address: 0x1000,
            old_size: 64,
            new_address: 0x1000,
            new: Shape::of(128),
        };
        assert!(!realloc.moved());
        assert_eq!(realloc.bytes_copied(), 0);

        let shapes = Shapes::new();
        shapes.record_realloc(&realloc);
        let stats = shapes.snapshot();
        assert_eq!(stats.reallocs, 1);
        assert_eq!(stats.reallocs_moved, 0);
        assert_eq!(stats.bytes_copied, 0);
        assert_eq!(stats.bytes_grown, 64);
        assert_eq!(stats.bytes_shrunk, 0);
    }

    /// A move copies what survives it, which is the smaller of the two sizes —
    /// the number a reader is looking for when a realloc-heavy site is at the
    /// top of the profile.
    #[test]
    fn a_move_copies_the_smaller_of_the_two_sizes() {
        let grew = Realloc {
            old_address: 0x1000,
            old_size: 64,
            new_address: 0x2000,
            new: Shape::of(128),
        };
        assert_eq!(grew.bytes_copied(), 64);

        let shrank = Realloc {
            old_address: 0x1000,
            old_size: 128,
            new_address: 0x2000,
            new: Shape::of(64),
        };
        assert_eq!(shrank.bytes_copied(), 64);

        let shapes = Shapes::new();
        shapes.record_realloc(&grew);
        shapes.record_realloc(&shrank);
        let stats = shapes.snapshot();
        assert_eq!(stats.reallocs, 2);
        assert_eq!(stats.reallocs_moved, 2);
        assert_eq!(stats.bytes_copied, 128);
        assert_eq!(stats.bytes_grown, 64);
        assert_eq!(stats.bytes_shrunk, 64);
    }

    /// What doubling actually costs, which is the reason these counters exist.
    #[test]
    fn doubling_from_one_byte_copies_almost_the_final_size() {
        let shapes = Shapes::new();
        let mut size = 1usize;
        while size < 1 << 20 {
            shapes.record_realloc(&Realloc {
                old_address: 0x1000,
                old_size: size,
                new_address: 0x2000,
                new: Shape::of(size * 2),
            });
            size *= 2;
        }
        let stats = shapes.snapshot();
        // 1 + 2 + ... + 2^19 = 2^20 - 1.
        assert_eq!(stats.bytes_copied, (1 << 20) - 1);
        assert_eq!(stats.bytes_grown, (1 << 20) - 1);
    }

    /// The tie-break is documented, so it is asserted. `max_by_key` returns the
    /// *last* maximum, so without the reversed floor in the key this would name
    /// the largest tied class — and a reader is more likely to be able to do
    /// something about the smallest.
    #[test]
    fn the_commonest_size_breaks_a_tie_toward_the_smaller_class() {
        let shapes = Shapes::new();
        for size in [24usize, 24, 4096, 4096, 1_000_000, 1_000_000] {
            shapes.record(Shape::of(size));
        }
        assert_eq!(shapes.snapshot().commonest_size(), Some((16, 31, 2)));
    }

    #[test]
    fn empty_classes_are_left_out_of_the_report() {
        let shapes = Shapes::new();
        shapes.record(Shape::of(24));
        let stats = shapes.snapshot();
        assert_eq!(stats.size_classes().count(), 1);
        assert_eq!(stats.alignments_used().count(), 1);

        assert_eq!(ShapeStats::default().size_classes().count(), 0);
        assert_eq!(ShapeStats::default().commonest_size(), None);
    }
}
