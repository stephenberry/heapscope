//! Core engine primitives. Not a supported surface: this module is public so
//! that integration tests, benchmarks and the reference tracker can observe the
//! engine, and nothing in it carries a stability promise.
//!
//! Everything in this module is reachable from inside the global allocator
//! shim, which imposes two absolute constraints:
//!
//! 1. **Nothing here may allocate through the global allocator.** Storage comes
//!    from a bump arena that calls the wrapped inner allocator directly.
//! 2. **All global state must be const-initializable.** The shim is live before
//!    `main`, so no `OnceLock`-guarded allocation may be reachable from it.

pub mod arena;
pub mod clock;
pub mod diagnostic;
pub mod engine;
pub mod fork;
pub mod gate;
pub mod guard;
pub mod live;
pub mod lock;
pub mod order;
pub mod pp;
pub mod sampler;
pub mod shape;
pub mod site;
pub mod stack;
pub mod table;

/// Pads and aligns a value to its own cache line.
///
/// Sharding only reduces contention if the shards are on different cache lines.
/// They are not, by default, and the numbers are stark:
///
/// | Target | `size_of::<RawLock>()` | Locks per 64-byte line |
/// |---|---|---|
/// | aarch64-apple-darwin | 4 | **16** |
/// | x86_64-pc-windows-msvc | 8 | **8** |
/// | x86_64/aarch64-linux-gnu | 64 | 1, but free to straddle two |
///
/// Sixteen shards sharing one line means touching any one of them invalidates
/// the other fifteen in every other core's cache — the exact false sharing that
/// sharding exists to remove, and it would have made a shard count of 64 an
/// elaborate way to build a single contended lock.
#[derive(Clone, Copy, Default)]
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), repr(align(128)))]
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    repr(align(64))
)]
pub struct CachePadded<T>(pub T);

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for CachePadded<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> CachePadded<T> {
    /// Wraps `value` so that it occupies a cache line of its own.
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for CachePadded<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Scales a test loop count down under Miri.
///
/// Miri interprets every instruction, so native iteration counts turn a handful
/// of these tests into a CI job measured in tens of minutes — one full run took
/// 37.
///
/// **Use this only where the count is arbitrary.** A test whose point is
/// *crossing a threshold* — filling a table to its ceiling, allocating past a
/// chunk boundary, growing a map past its load factor — must have its capacity
/// scaled to match, or it silently stops testing anything. Applying this
/// blanket to every loop broke ten such tests at once, which is the cheapest
/// possible demonstration that the two cases are different.
#[cfg(test)]
pub(crate) const fn miri_scale(native: usize) -> usize {
    if cfg!(miri) && native > 200 {
        200
    } else {
        native
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 128 bytes rather than 64 on the two supported architectures: Apple
    /// M-series cores have 128-byte cache lines, and x86_64 prefetches in
    /// 128-byte pairs, so 64 would leave adjacent shards sharing a prefetch
    /// unit even when they are on distinct lines.
    #[test]
    fn padding_separates_adjacent_elements() {
        let expected = if cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
            128
        } else {
            64
        };
        assert_eq!(std::mem::align_of::<CachePadded<lock::RawLock>>(), expected);
        assert!(std::mem::size_of::<CachePadded<lock::RawLock>>() >= expected);

        let shards: [CachePadded<lock::RawLock>; 4] = Default::default();
        let stride = std::ptr::from_ref(&shards[1]).addr() - std::ptr::from_ref(&shards[0]).addr();
        assert_eq!(
            stride, expected,
            "adjacent shards are {stride} bytes apart and would share a cache line"
        );
    }

    #[test]
    fn padded_locks_still_work() {
        let padded = CachePadded::new(lock::RawLock::new());
        let guard = padded.lock();
        drop(guard);
    }
}
