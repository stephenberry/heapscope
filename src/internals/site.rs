//! Who allocated, and what for: thread and region attribution.
//!
//! A stack trace says *where* an allocation happened. Two questions it cannot
//! answer come up constantly and neither has a field in DHAT v2:
//!
//! - **Which thread?** The same call site reached from four worker threads is
//!   one program point. If one of those threads is the one holding memory, the
//!   profile as DHAT can express it does not say so.
//! - **During which phase?** A program that parses, then plans, then emits
//!   allocates from the same helpers throughout. Attributing to a phase the
//!   program names itself ([`crate::region`](fn@crate::region)) separates them.
//!
//! Both are recorded per *block*, so a free brings the right row back down, and
//! both are summarised per row in [`Tally`].
//!
//! # Why the names are captured here rather than at output time
//!
//! A thread name dies with the thread. By the time a profile is written, every
//! worker that did the allocating is usually gone, and asking the platform then
//! returns the name of whichever thread happens to be writing the file. So the
//! name is copied out of the platform on a thread's **first recorded event**,
//! which puts a `pthread_getname_np` on a cold path inside the allocator shim —
//! once per thread, never again, because the row id is then cached in the
//! thread's guard slot.
//!
//! # Why not `std::thread::current()`
//!
//! It is the obvious source for a *Rust* thread name, and it is unusable here:
//! it **panics** once the thread's local data has been destroyed ("use of
//! std::thread::current() is not possible after the thread's local data has been
//! destroyed" — `library/std/src/thread/current.rs`). Late allocations during
//! thread teardown are exactly the case that reaches it, and a panic inside a
//! `GlobalAlloc` method is undefined behaviour, not a test failure.
//!
//! Asking the platform instead has a second advantage that is not a
//! consolation: the name in the profile is the name `top -H`, `perf`, and a
//! debugger show, because it is the same string. `std::thread::Builder::name`
//! pushes the name to the OS on every platform this crate supports, so a Rust
//! program's own names arrive anyway — subject to the platform's length limit,
//! which on Linux is 15 bytes and is the kernel's, not this crate's.
//!
//! # Cost
//!
//! Attribution adds nothing to the two tables that dominate the profiler's
//! memory, on the targets this crate supports. A [`Site`] is four bytes, which
//! fits in the padding [`LiveBlock`](super::live::LiveBlock) already had, and
//! the two guard-slot words fit in the padding a [`Slot`](super::guard) already
//! had. Both sizes are `const` assertions rather than claims — though the slot
//! one is asserted for 64-bit only, where there was padding to use: on a 32-bit
//! target the slot really does grow, from 16 bytes to 20, and no supported
//! target is 32-bit.
//!
//! What it does add is atomic traffic: up to six read-modify-writes on the
//! thread's own row per recorded allocation, and six more when a region is
//! open. They are uncontended — a row is written almost exclusively by the
//! thread that owns it — which is what makes them affordable next to the global
//! counters they sit beside, which are contended by construction.

use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicU8, AtomicUsize, Ordering};

use super::arena::Arena;
use super::lock::RawLock;
use super::order::{self, Level};

/// Longest name kept for a thread or a region, in bytes.
///
/// Names longer than this are cut, and two names sharing a prefix this long
/// become one row. Both are stated in the output rather than left to be
/// inferred: a row's name is what was kept, not what was asked for.
pub const MAX_NAME: usize = 64;

/// Rows the thread table holds before further threads share the overflow row.
///
/// The same size as the guard's slot table, and for a related reason: a thread
/// that cannot be guarded cannot be recorded at all. This is not the same
/// bound, though, because guard slots are **reclaimed** when a thread exits and
/// these rows are not — a thread's allocations outlive it, so its row has to as
/// well. A program that creates more than this many threads over its lifetime
/// keeps a row for the first [`MAX_THREADS`] and folds the rest into
/// [`ThreadId::OVERFLOW`], which the profile shows as its own row rather than
/// silently merging with a real thread.
pub const MAX_THREADS: usize = 4096;

/// Rows the region table holds before further names share the overflow row.
///
/// Regions are phases a program names itself, so this is generous by two orders
/// of magnitude for the intended use and deliberately finite for the one that
/// is not: `region(&format!("task {i}"))` in a loop is a table of unbounded
/// size, and the overflow row is what keeps that from becoming a memory leak
/// inside a memory profiler.
pub const MAX_REGIONS: usize = 256;

/// Reserved id meaning "no row here".
const UNSET: u16 = u16::MAX;
/// Reserved id for the shared row every unrecordable thread or region lands in.
const OVERFLOW: u16 = u16::MAX - 1;

const _: () = assert!(MAX_THREADS < OVERFLOW as usize);
const _: () = assert!(MAX_REGIONS < OVERFLOW as usize);

/// Which thread a block belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ThreadId(u16);

impl ThreadId {
    /// The row shared by every thread past [`MAX_THREADS`].
    pub const OVERFLOW: ThreadId = ThreadId(OVERFLOW);

    /// No row: the value a guard slot holds before its thread has claimed one.
    ///
    /// Distinct from [`ThreadId::OVERFLOW`], and the distinction is what stops
    /// a thread that failed to claim a row from trying again on every single
    /// allocation for the rest of its life.
    pub const UNCLAIMED: ThreadId = ThreadId(UNSET);

    /// The raw value, for output and for stable ordering.
    pub const fn as_u16(self) -> u16 {
        self.0
    }

    /// Rebuilds an id from a guard slot's raw word.
    ///
    /// Total, because every `u16` names something: a row, the overflow row, or
    /// no row at all.
    pub const fn from_u16(raw: u16) -> ThreadId {
        ThreadId(raw)
    }

    /// Whether this is the shared overflow row.
    pub fn is_overflow(self) -> bool {
        self.0 == OVERFLOW
    }

    /// Whether the owning thread has yet to claim a row.
    pub fn is_unclaimed(self) -> bool {
        self.0 == UNSET
    }
}

impl Default for ThreadId {
    fn default() -> Self {
        ThreadId::UNCLAIMED
    }
}

/// Which region a block was allocated in, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId(u16);

impl RegionId {
    /// The row shared by every region name past [`MAX_REGIONS`].
    pub const OVERFLOW: RegionId = RegionId(OVERFLOW);

    /// Outside every region, which is where most allocations in most programs
    /// happen.
    pub const NONE: RegionId = RegionId(UNSET);

    /// The raw value, for output and for stable ordering.
    pub const fn as_u16(self) -> u16 {
        self.0
    }

    /// Rebuilds an id from a guard slot's raw word. Total, as for
    /// [`ThreadId::from_u16`].
    pub const fn from_u16(raw: u16) -> RegionId {
        RegionId(raw)
    }

    /// Whether this is the shared overflow row.
    pub fn is_overflow(self) -> bool {
        self.0 == OVERFLOW
    }

    /// Whether no region was open.
    pub fn is_none(self) -> bool {
        self.0 == UNSET
    }
}

impl Default for RegionId {
    fn default() -> Self {
        RegionId::NONE
    }
}

/// Who was allocating, and what for.
///
/// Four bytes, which is the whole reason it is a struct: it fits in padding
/// that [`LiveBlock`](super::live::LiveBlock) and the guard's slot table were
/// both already carrying, so recording it costs no memory in either.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Site {
    /// The thread that made the allocation.
    pub thread: ThreadId,
    /// The innermost region open on that thread at the time.
    pub region: RegionId,
}

impl Site {
    /// The attribution of an event recorded outside any run: no thread row, no
    /// region.
    pub const UNATTRIBUTED: Site = Site {
        thread: ThreadId::UNCLAIMED,
        region: RegionId::NONE,
    };
}

/// A name copied out of the platform, or out of the caller's string.
///
/// Inline rather than a pointer into the arena because it is written once,
/// before the row is published, and read without synchronization afterwards.
/// A pointer would need its own ordering and would save nothing: the row is
/// arena-allocated either way.
#[derive(Clone, Copy)]
pub struct Name {
    bytes: [u8; MAX_NAME],
    len: u8,
}

impl Name {
    /// The name of something the platform would not name.
    pub const EMPTY: Name = Name {
        bytes: [0; MAX_NAME],
        len: 0,
    };

    /// Keeps as much of `text` as fits, cut at a character boundary.
    ///
    /// Cut at a boundary rather than at a byte so that the name stays valid
    /// UTF-8: a region name is a Rust `&str`, and truncating one mid-character
    /// would turn a name the program chose into replacement characters in the
    /// profile.
    pub fn of(text: &str) -> Name {
        let mut end = text.len().min(MAX_NAME);
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        Name::of_bytes(&text.as_bytes()[..end])
    }

    /// Keeps as much of `raw` as fits.
    ///
    /// For names that came from the platform, which promises no encoding at
    /// all. They are rendered lossily at output time rather than rejected here,
    /// because a mangled name still identifies a thread and no name does not.
    pub fn of_bytes(raw: &[u8]) -> Name {
        let len = raw.len().min(MAX_NAME);
        let mut bytes = [0u8; MAX_NAME];
        bytes[..len].copy_from_slice(&raw[..len]);
        Name {
            bytes,
            // `MAX_NAME` fits in a `u8`, which the assertion below holds to.
            len: len as u8,
        }
    }

    /// The bytes kept.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    /// Whether the platform gave a name at all.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// How many bytes were kept.
    pub fn len(&self) -> u8 {
        self.len
    }
}

const _: () = assert!(MAX_NAME <= u8::MAX as usize);

impl std::fmt::Debug for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&String::from_utf8_lossy(self.as_bytes()), f)
    }
}

/// What one row recorded.
///
/// The same six counters the global totals carry, per thread and per region.
/// There is deliberately no at-peak column: the global `gb`/`gbk` pair needs
/// every row snapshotted at one instant, which is what the peak gate is for,
/// and putting these rows under that gate would lengthen the one critical
/// section this crate spends its effort keeping short. What is here instead is
/// each row's **own** peak, which is exact without any gate at all — see
/// [`Tally::apply`].
#[derive(Debug, Default)]
pub struct Tally {
    total_bytes: AtomicU64,
    total_blocks: AtomicU64,
    curr_bytes: AtomicU64,
    curr_blocks: AtomicU64,
    max_bytes: AtomicU64,
    max_blocks: AtomicU64,
}

/// One row's counters, read out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TallyStats {
    /// Bytes ever allocated by this thread, or in this region.
    pub total_bytes: u64,
    /// Blocks ever allocated. In a non-heap run, events recorded.
    pub total_blocks: u64,
    /// Bytes still live.
    pub curr_bytes: u64,
    /// Blocks still live.
    pub curr_blocks: u64,
    /// Greatest `curr_bytes` this row ever reached.
    pub max_bytes: u64,
    /// Greatest `curr_blocks` this row ever reached.
    pub max_blocks: u64,
}

impl Tally {
    /// A row that has recorded nothing.
    pub const fn new() -> Self {
        Self {
            total_bytes: AtomicU64::new(0),
            total_blocks: AtomicU64::new(0),
            curr_bytes: AtomicU64::new(0),
            curr_blocks: AtomicU64::new(0),
            max_bytes: AtomicU64::new(0),
            max_blocks: AtomicU64::new(0),
        }
    }

    /// Applies one event's deltas.
    ///
    /// The arguments are the same four numbers the engine's own `Delta` carries,
    /// passed from the one place that applies it, so the row and the global
    /// counters cannot drift apart by someone updating one and forgetting the
    /// other.
    ///
    /// # Why this peak is exact without a lock
    ///
    /// `fetch_add` returns the value the counter held immediately before the
    /// add, so `previous + amount` is a value `curr_bytes` genuinely held at the
    /// instant that add linearized — not an estimate, and not perturbed by a
    /// concurrent free that lands either side of it. Feeding exactly that into
    /// `fetch_max` therefore records the true running maximum of this row, no
    /// matter how many threads are adding at once. The global peak needs the
    /// gate for a different reason: it has to snapshot *other* rows at the same
    /// instant, which no single atomic can do.
    ///
    /// Increments are branched on rather than applied as `fetch_add(0)`, because
    /// a zero read-modify-write costs a cache line in exclusive state exactly
    /// like a real one. A free touches two counters here, not six.
    #[inline]
    pub fn apply(&self, curr_bytes: i64, curr_blocks: i64, total_bytes: u64, total_blocks: u64) {
        if total_bytes != 0 {
            self.total_bytes.fetch_add(total_bytes, Ordering::Relaxed);
        }
        if total_blocks != 0 {
            self.total_blocks.fetch_add(total_blocks, Ordering::Relaxed);
        }
        Self::grow_or_shrink(
            &self.curr_bytes,
            &self.max_bytes,
            curr_bytes,
            "a thread or region row's live bytes went negative",
        );
        Self::grow_or_shrink(
            &self.curr_blocks,
            &self.max_blocks,
            curr_blocks,
            "a thread or region row's live blocks went negative",
        );
    }

    /// Moves `curr` by `delta`, keeping `max` at the largest value `curr` held.
    ///
    /// `underflow` is the sentence reported if the subtraction goes past zero.
    /// A literal rather than something formatted here: this runs inside the
    /// allocator shim, where building a message allocates and re-enters.
    #[inline]
    fn grow_or_shrink(curr: &AtomicU64, max: &AtomicU64, delta: i64, underflow: &'static str) {
        match delta.signum() {
            1 => {
                let amount = delta as u64;
                let now = curr
                    .fetch_add(amount, Ordering::Relaxed)
                    .wrapping_add(amount);
                max.fetch_max(now, Ordering::Relaxed);
            }
            -1 => {
                let amount = delta.unsigned_abs();
                // The same checked discipline the engine's own counters use: an
                // underflow here means a block was freed against a row that
                // never allocated it, which is a defect worth naming rather
                // than a number to clamp quietly.
                let previous = curr.fetch_sub(amount, Ordering::Relaxed);
                if previous < amount {
                    curr.store(0, Ordering::Relaxed);
                    super::diagnostic::poison(underflow);
                }
            }
            _ => {}
        }
    }

    /// Reads the row.
    pub fn snapshot(&self) -> TallyStats {
        TallyStats {
            total_bytes: self.total_bytes.load(Ordering::Relaxed),
            total_blocks: self.total_blocks.load(Ordering::Relaxed),
            curr_bytes: self.curr_bytes.load(Ordering::Relaxed),
            curr_blocks: self.curr_blocks.load(Ordering::Relaxed),
            max_bytes: self.max_bytes.load(Ordering::Relaxed),
            max_blocks: self.max_blocks.load(Ordering::Relaxed),
        }
    }
}

/// Attempts a thread makes to learn its own name before it stops asking.
///
/// It takes more than one, and the reason is specific: on Windows,
/// `std::thread::Builder::name` sets the OS name by encoding it to UTF-16
/// first, and that encoding **allocates** (`Vec::with_capacity` in `to_u16s`,
/// then `SetThreadDescription`). So a spawned thread's very first recorded
/// allocation is std's own name buffer, made a few instructions *before* the
/// name exists, and a profiler that asks once records every worker as unnamed.
/// Measured under Wine: `["main", ""]` where the second row is a thread the
/// program named `hs-worker`.
///
/// Two would do for that case. Eight is margin for a `Vec` that grows, and is
/// still a bound: a thread the platform never names asks eight times over its
/// life and then never again.
pub const NAME_ATTEMPTS: u8 = 8;

/// A thread name, published once by the owning thread — not necessarily on its
/// first try.
///
/// Written and read without a lock, and without `unsafe`: the bytes are atoms,
/// and `state` is the one-shot flag that publishes them. The owning thread
/// writes the bytes, then releases `state`; a reader acquires `state` and reads
/// the bytes only if it saw the release. Nothing writes after that.
#[derive(Debug)]
struct NameCell {
    bytes: [AtomicU8; MAX_NAME],
    /// Valid once `state` reads [`SETTLED`].
    len: AtomicU8,
    /// Attempts made so far, or [`SETTLED`] once the answer is final —
    /// a name was found, or the attempts ran out.
    state: AtomicU8,
}

/// The value [`NameCell::state`] holds once it will not change again.
const SETTLED: u8 = u8::MAX;

const _: () = assert!(NAME_ATTEMPTS < SETTLED);

impl NameCell {
    const fn new() -> Self {
        Self {
            bytes: [const { AtomicU8::new(0) }; MAX_NAME],
            len: AtomicU8::new(0),
            state: AtomicU8::new(0),
        }
    }

    /// Whether the owning thread should ask the platform again.
    #[inline]
    fn unsettled(&self) -> bool {
        self.state.load(Ordering::Relaxed) != SETTLED
    }

    /// Records one attempt by the owning thread, publishing `name` if there was
    /// one to publish.
    ///
    /// Only the owning thread calls this, so the read-modify-write of `state`
    /// needs no atomicity beyond each individual access.
    fn attempt(&self, name: Name) {
        if !name.is_empty() {
            for (slot, byte) in self.bytes.iter().zip(name.as_bytes()) {
                slot.store(*byte, Ordering::Relaxed);
            }
            self.len.store(name.len(), Ordering::Relaxed);
            // Released last, so a reader that sees `SETTLED` sees the bytes.
            self.state.store(SETTLED, Ordering::Release);
            return;
        }
        let made = self.state.load(Ordering::Relaxed).saturating_add(1);
        let next = if made >= NAME_ATTEMPTS { SETTLED } else { made };
        self.state.store(next, Ordering::Release);
    }

    /// The name, empty while the platform has not given one.
    fn get(&self) -> Name {
        if self.state.load(Ordering::Acquire) != SETTLED {
            return Name::EMPTY;
        }
        let len = self.len.load(Ordering::Relaxed) as usize;
        let mut bytes = [0u8; MAX_NAME];
        for (byte, slot) in bytes.iter_mut().zip(self.bytes.iter().take(len)) {
            *byte = slot.load(Ordering::Relaxed);
        }
        Name {
            bytes,
            len: len as u8,
        }
    }
}

/// One thread's row.
#[derive(Debug)]
struct ThreadRecord {
    /// Published by the owning thread, once, on one of its first few events.
    name: NameCell,
    /// Clock reading when this thread first recorded something.
    first_seen: u64,
    tally: Tally,
}

/// One region's row.
#[derive(Debug)]
struct RegionRecord {
    name: Name,
    /// Hash of the kept name, compared before the bytes are.
    hash: u64,
    first_seen: u64,
    /// Times this name was entered, on any thread.
    entries: AtomicU64,
    /// Times it was entered and not yet left.
    active: AtomicU64,
    tally: Tally,
}

/// A thread's row, copied out for output.
///
/// Plain data rather than a borrow, and `Copy`: the output layer collects these
/// under the peak gate, where it may not allocate, and turns them into owned
/// strings afterwards. A name is 65 bytes, so a view is cheap to copy and owes
/// nothing to the table it came from.
#[derive(Clone, Copy, Debug)]
pub struct ThreadView {
    /// Which row.
    pub id: ThreadId,
    /// The name the platform had for the thread, empty if it had none.
    pub name: Name,
    /// Clock reading when the thread first recorded something.
    pub first_seen: u64,
    /// What it recorded.
    pub counts: TallyStats,
}

/// A region's row, copied out for output. See [`ThreadView`].
#[derive(Clone, Copy, Debug)]
pub struct RegionView {
    /// Which row.
    pub id: RegionId,
    /// The name the program gave the region.
    pub name: Name,
    /// Clock reading when the name was first entered.
    pub first_seen: u64,
    /// Times entered, on any thread.
    pub entries: u64,
    /// Times entered and not yet left. Non-zero at end of run means a region
    /// guard was still alive, or was leaked.
    pub active: u64,
    // `first_seen` above is set when the name is *interned*, which for the
    // public API is the first time the program entered it: `region` interns and
    // enters in one call.
    /// What was allocated while it was the innermost open region.
    pub counts: TallyStats,
}

/// The per-thread rows.
///
/// Rows are claimed by a bump counter and published with a release store, so
/// claiming needs no lock: each thread claims exactly once, and no two threads
/// ever contend for the same row.
#[derive(Debug)]
pub struct Threads {
    rows: [AtomicPtr<ThreadRecord>; MAX_THREADS],
    count: AtomicUsize,
    /// Shared by every thread past [`MAX_THREADS`]. Inline rather than
    /// arena-allocated so that it exists before the first allocation does.
    overflow: ThreadRecord,
}

// No `unsafe impl Sync` here, deliberately. `AtomicPtr<T>` is unconditionally
// `Sync` and every other field is an atomic or plain data, so the auto trait
// applies on its own — and writing the impl anyway would switch the auto-trait
// check off permanently, so that a future `Cell` or `*mut T` field became
// `Sync` in silence. `PpTable` and the live-block shards genuinely need theirs;
// this does not. (Verified by deleting it: it compiles and the suite passes,
// which is what says the impl was doing nothing.)
//
// What does need stating is the obligation the raw pointers carry: **the arena
// handed to `Threads::claim` must outlive this table.** The engine satisfies it
// structurally — both are fields of the same `Engine`, and nothing reads a row
// during drop — and it is discharged at the point of use, in the SAFETY comment
// on `Threads::record`.

impl Threads {
    /// An empty table.
    pub const fn new() -> Self {
        Self {
            rows: [const { AtomicPtr::new(std::ptr::null_mut()) }; MAX_THREADS],
            count: AtomicUsize::new(0),
            overflow: ThreadRecord {
                // The shared row is never named: it stands for many threads.
                name: NameCell::new(),
                first_seen: 0,
                tally: Tally::new(),
            },
        }
    }

    /// Claims a row for the calling thread.
    ///
    /// Returns [`ThreadId::OVERFLOW`] when the table is full or the arena is
    /// exhausted. Either way the caller caches the answer, so a thread that
    /// cannot have a row of its own asks once rather than on every allocation.
    pub fn claim(&self, arena: &Arena, name: Name, now: u64) -> ThreadId {
        let index = loop {
            let count = self.count.load(Ordering::Relaxed);
            if count >= MAX_THREADS {
                return ThreadId::OVERFLOW;
            }
            if self
                .count
                .compare_exchange_weak(count, count + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break count;
            }
        };

        let record = ThreadRecord {
            name: NameCell::new(),
            first_seen: now,
            tally: Tally::new(),
        };
        record.name.attempt(name);
        let Some(record) = arena.alloc_value(record) else {
            // The row stays claimed and empty. Giving it back would mean
            // uncounting a bump another thread may already have built on, and
            // an arena with no room for a 128-byte record is a run in more
            // trouble than one lost row. The hole it leaves is counted by
            // `len` and `bytes`, so the self-metrics over-report by one row in
            // a run that has already exhausted its arena.
            return ThreadId::OVERFLOW;
        };
        self.rows[index].store(record, Ordering::Release);
        ThreadId(index as u16)
    }

    /// Whether `id`'s thread should ask the platform for its name again.
    ///
    /// See [`NAME_ATTEMPTS`] for why asking once is not enough. The overflow
    /// row stands for many threads and is never named, so it never asks — as
    /// `OVERFLOW` is past the end of the array, the bounds check is also that
    /// rule.
    ///
    /// # What this costs
    ///
    /// It is consulted once per recorded event, for the life of the process, so
    /// the cost is worth stating exactly: a bounds check on a static array, a
    /// **relaxed** pointer load, and a byte load and compare. Not one load and
    /// a compare, which is what this said first and is what a reader would
    /// otherwise assume.
    ///
    /// It is affordable because the line is not cold: the same event reaches
    /// this record again through [`Threads::tally`], a few hundred nanoseconds
    /// later, so this warms what that needs. On a 128-byte line — the primary
    /// development target — the whole record is one line; on a 64-byte line the
    /// flag and the counters are two, and that is the honest worst case.
    ///
    /// It could be made free by keeping a "settled" bit in the guard slot,
    /// whose word this path has already loaded. That was not done: it would put
    /// a packed flag in the same `u16` that carries the row id, which is the
    /// field whose out-of-range sentinels are what make [`Threads::tally`] safe
    /// without a branch. One load is not worth complicating that.
    ///
    /// The load is relaxed rather than acquiring because **only the owning
    /// thread calls this**, and it published that pointer itself; another
    /// thread reading a stale null would simply see `false`, which is the same
    /// answer it would give for a row it does not own.
    #[inline]
    pub fn wants_name(&self, id: ThreadId) -> bool {
        match self.record_with(id.0 as usize, Ordering::Relaxed) {
            Some(record) => record.name.unsettled(),
            None => false,
        }
    }

    /// Offers `name` to `id`'s row, counting the attempt either way.
    ///
    /// Call only from the thread that owns the row: `NameCell` is written by
    /// its owner and read by everyone.
    pub fn name(&self, id: ThreadId, name: Name) {
        if let Some(record) = self.record_with(id.0 as usize, Ordering::Relaxed) {
            record.name.attempt(name);
        }
    }

    /// The counters for `id`, or `None` if it names no row.
    #[inline]
    pub fn tally(&self, id: ThreadId) -> Option<&Tally> {
        if id.is_overflow() {
            return Some(&self.overflow.tally);
        }
        // `UNCLAIMED` is `u16::MAX`, past the end of the array, so the bounds
        // check below is also the check for "this thread has no row".
        let record = self.record(id.0 as usize)?;
        Some(&record.tally)
    }

    /// Rows claimed so far, the overflow row excluded.
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed).min(MAX_THREADS)
    }

    /// Whether any thread has claimed a row.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rows this table can hold before it overflows.
    pub fn capacity(&self) -> usize {
        MAX_THREADS
    }

    /// Arena bytes the claimed rows occupy.
    pub fn bytes(&self) -> usize {
        self.len() * std::mem::size_of::<ThreadRecord>()
    }

    /// Visits every claimed row in id order, then the overflow row if anything
    /// reached it.
    pub fn visit(&self, mut f: impl FnMut(ThreadView)) {
        for index in 0..self.len() {
            let Some(record) = self.record(index) else {
                // Claimed but not yet published: a thread is between its bump
                // and its release store. It has recorded nothing yet, so there
                // is nothing to report.
                continue;
            };
            f(ThreadView {
                id: ThreadId(index as u16),
                name: record.name.get(),
                first_seen: record.first_seen,
                counts: record.tally.snapshot(),
            });
        }

        let counts = self.overflow.tally.snapshot();
        if counts.total_blocks != 0 {
            f(ThreadView {
                id: ThreadId::OVERFLOW,
                name: Name::EMPTY,
                first_seen: 0,
                counts,
            });
        }
    }

    /// The row at `index`, if one has been published there.
    ///
    /// `ordering` is `Acquire` for a reader that may be another thread, and
    /// `Relaxed` only where the caller is the row's own owner — see
    /// [`Threads::wants_name`], the one place that is true.
    #[inline]
    fn record_with(&self, index: usize, ordering: Ordering) -> Option<&ThreadRecord> {
        let ptr = self.rows.get(index)?.load(ordering);
        // SAFETY: a non-null row was written into the arena and published with a
        // release store, which the acquire load above pairs with; the relaxed
        // case is a thread reading a pointer it published itself, which its own
        // program order orders. The arena outlives this table and never frees a
        // record, so the reference is valid for as long as `self` is — an
        // obligation on whoever calls `Threads::claim`, and one the engine meets
        // by owning both.
        unsafe { ptr.as_ref() }
    }

    #[inline]
    fn record(&self, index: usize) -> Option<&ThreadRecord> {
        self.record_with(index, Ordering::Acquire)
    }
}

impl Default for Threads {
    fn default() -> Self {
        Self::new()
    }
}

/// The per-region rows, interned by name.
///
/// Unlike threads, two calls naming the same region must land on the same row,
/// so this one needs a lock. It is never taken on the allocator path: entering
/// a region is something the program does at a phase boundary, and what the
/// allocator path reads is the interned id, already sitting in a guard slot.
#[derive(Debug)]
pub struct Regions {
    lock: RawLock,
    rows: [AtomicPtr<RegionRecord>; MAX_REGIONS],
    count: AtomicUsize,
    overflow: RegionRecord,
}

// As for `Threads`: no `unsafe impl Sync`, because `RawLock` carries its own and
// every other field is an atomic. Rows here are additionally only ever *created*
// under `lock`, so no two threads publish the same name twice.

impl Regions {
    /// An empty table.
    pub const fn new() -> Self {
        Self {
            lock: RawLock::new(),
            rows: [const { AtomicPtr::new(std::ptr::null_mut()) }; MAX_REGIONS],
            count: AtomicUsize::new(0),
            overflow: RegionRecord {
                name: Name::EMPTY,
                hash: 0,
                first_seen: 0,
                entries: AtomicU64::new(0),
                active: AtomicU64::new(0),
                tally: Tally::new(),
            },
        }
    }

    /// Returns the row for `name`, creating it if this is the first time it has
    /// been seen.
    ///
    /// Names are compared **after** truncation to [`MAX_NAME`], so two names
    /// sharing that long a prefix are one region. That is a deliberate
    /// consequence of bounding the name rather than an accident: the row is what
    /// the profile can name, and two rows a reader cannot tell apart would be
    /// worse than one.
    pub fn intern(&self, arena: &Arena, name: &str, now: u64) -> RegionId {
        let name = Name::of(name);
        let hash = hash_bytes(name.as_bytes());

        let _order = order::enter(Level::RegionTable);
        let _lock = self.lock.lock();

        let count = self.count.load(Ordering::Relaxed);
        for index in 0..count {
            let Some(record) = self.record(index) else {
                continue;
            };
            if record.hash == hash && record.name.as_bytes() == name.as_bytes() {
                return RegionId(index as u16);
            }
        }

        if count >= MAX_REGIONS {
            return RegionId::OVERFLOW;
        }
        let Some(record) = arena.alloc_value(RegionRecord {
            name,
            hash,
            first_seen: now,
            entries: AtomicU64::new(0),
            active: AtomicU64::new(0),
            tally: Tally::new(),
        }) else {
            return RegionId::OVERFLOW;
        };
        self.rows[count].store(record, Ordering::Release);
        self.count.store(count + 1, Ordering::Release);
        RegionId(count as u16)
    }

    /// Counts one entry into `id`.
    pub fn enter(&self, id: RegionId) {
        if let Some(record) = self.row(id) {
            record.entries.fetch_add(1, Ordering::Relaxed);
            record.active.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Counts one exit from `id`.
    pub fn leave(&self, id: RegionId) {
        if let Some(record) = self.row(id) {
            let previous = record.active.fetch_sub(1, Ordering::Relaxed);
            if previous == 0 {
                record.active.store(0, Ordering::Relaxed);
                super::diagnostic::poison("a region was left more times than it was entered");
            }
        }
    }

    /// The counters for `id`, or `None` if it names no row.
    #[inline]
    pub fn tally(&self, id: RegionId) -> Option<&Tally> {
        Some(&self.row(id)?.tally)
    }

    /// Rows interned so far, the overflow row excluded.
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire).min(MAX_REGIONS)
    }

    /// Whether the program entered any region at all. False for most runs.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rows this table can hold before it overflows.
    pub fn capacity(&self) -> usize {
        MAX_REGIONS
    }

    /// Arena bytes the interned rows occupy.
    pub fn bytes(&self) -> usize {
        self.len() * std::mem::size_of::<RegionRecord>()
    }

    /// Visits every row that was **entered**, in id order, then the overflow
    /// row if anything reached it.
    ///
    /// Interning a name and entering it are two steps, and a caller may take
    /// the first without the second — [`Engine::intern_region`] is reachable on
    /// its own. A row in that state is a name the profiler has heard of and a
    /// phase the program never went through, and listing it would be reporting
    /// the second as the first. Same rule as the overflow row: a row appears
    /// once it has something to say.
    ///
    /// [`Engine::intern_region`]: crate::internals::engine::Engine::intern_region
    pub fn visit(&self, mut f: impl FnMut(RegionView)) {
        for index in 0..self.len() {
            let Some(record) = self.record(index) else {
                continue;
            };
            if record.entries.load(Ordering::Relaxed) == 0 {
                continue;
            }
            f(view(RegionId(index as u16), record));
        }

        if self.overflow.entries.load(Ordering::Relaxed) != 0 {
            f(view(RegionId::OVERFLOW, &self.overflow));
        }
    }

    /// Acquires the intern lock, for a `pthread_atfork` prepare handler.
    ///
    /// # Safety
    ///
    /// A matching [`Regions::unlock_for_fork`] must run on the same thread, or
    /// [`Regions::reinit_after_fork`] must reset it in the child.
    pub unsafe fn lock_for_fork(&self) {
        // SAFETY: delegated to the caller's pairing obligation.
        unsafe { self.lock.raw_lock() }
    }

    /// Releases what [`Regions::lock_for_fork`] acquired.
    ///
    /// # Safety
    ///
    /// Call only after a matching [`Regions::lock_for_fork`] on this thread.
    pub unsafe fn unlock_for_fork(&self) {
        // SAFETY: delegated to the caller's obligation.
        unsafe { self.lock.raw_unlock() }
    }

    /// Re-initializes the intern lock after a `fork`.
    ///
    /// The rows themselves are inherited intact — they are arena memory, and a
    /// `fork` copies it — so only the lock may have been orphaned by a thread
    /// that does not exist in the child.
    ///
    /// # Safety
    ///
    /// Call only from a `pthread_atfork` child handler, where the process is
    /// single-threaded.
    pub unsafe fn reinit_after_fork(&self) {
        // SAFETY: delegated to the caller's single-threadedness obligation.
        unsafe { self.lock.force_reinit() }
    }

    #[inline]
    fn row(&self, id: RegionId) -> Option<&RegionRecord> {
        if id.is_overflow() {
            return Some(&self.overflow);
        }
        // `NONE` is `u16::MAX`, past the end of the array, so the bounds check
        // is also the check for "no region was open".
        self.record(id.0 as usize)
    }

    #[inline]
    fn record(&self, index: usize) -> Option<&RegionRecord> {
        let ptr = self.rows.get(index)?.load(Ordering::Acquire);
        // SAFETY: as in `Threads::record`.
        unsafe { ptr.as_ref() }
    }
}

impl Default for Regions {
    fn default() -> Self {
        Self::new()
    }
}

fn view(id: RegionId, record: &RegionRecord) -> RegionView {
    RegionView {
        id,
        name: record.name,
        first_seen: record.first_seen,
        entries: record.entries.load(Ordering::Relaxed),
        active: record.active.load(Ordering::Relaxed),
        counts: record.tally.snapshot(),
    }
}

/// FNV-1a, finished with the table's mixer.
///
/// Not a security hash and not trying to be: it decides whether two short names
/// are worth a `memcmp`, and the `memcmp` is what actually decides.
fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    super::table::mix(hash)
}

/// Copies the calling thread's name out of the platform into `buffer`.
///
/// Returns how many bytes were written; zero means the platform had no name for
/// this thread, which is the normal state of a thread nobody named.
///
/// Safe to call from inside the allocator shim: no path here allocates through
/// the global allocator. On Windows the string the platform hands back is
/// released with `LocalFree`, which is the Win32 heap and not this crate's.
pub fn current_thread_name(buffer: &mut [u8; MAX_NAME]) -> usize {
    platform::current_thread_name(buffer)
}

#[cfg(all(unix, not(miri)))]
mod platform {
    use super::MAX_NAME;

    /// `pthread_t` is a pointer on Darwin and an integer on glibc, so each is
    /// spelled out rather than one being declared as the other. This mirrors
    /// [`super::super::guard::thread_handle`], for the same ABI reason.
    #[cfg(target_vendor = "apple")]
    mod sys {
        use std::ffi::{c_char, c_int, c_void};

        extern "C" {
            pub(super) fn pthread_self() -> *mut c_void;
            pub(super) fn pthread_getname_np(
                thread: *mut c_void,
                name: *mut c_char,
                len: usize,
            ) -> c_int;
        }

        /// # Safety
        ///
        /// `name` must point to `len` writable bytes.
        pub(super) unsafe fn getname(name: *mut c_char, len: usize) -> c_int {
            // SAFETY: `pthread_self` cannot fail, and the buffer contract is
            // the caller's to uphold.
            unsafe { pthread_getname_np(pthread_self(), name, len) }
        }
    }

    #[cfg(not(target_vendor = "apple"))]
    mod sys {
        use std::ffi::{c_char, c_int, c_ulong};

        extern "C" {
            pub(super) fn pthread_self() -> c_ulong;
            pub(super) fn pthread_getname_np(
                thread: c_ulong,
                name: *mut c_char,
                len: usize,
            ) -> c_int;
        }

        /// # Safety
        ///
        /// `name` must point to `len` writable bytes.
        pub(super) unsafe fn getname(name: *mut c_char, len: usize) -> c_int {
            // SAFETY: as above.
            unsafe { pthread_getname_np(pthread_self(), name, len) }
        }
    }

    pub(super) fn current_thread_name(buffer: &mut [u8; MAX_NAME]) -> usize {
        // SAFETY: the buffer is `MAX_NAME` writable bytes owned by the caller,
        // and the platform NUL-terminates within it or fails with `ERANGE`.
        let result = unsafe { sys::getname(buffer.as_mut_ptr().cast(), MAX_NAME) };
        if result != 0 {
            return 0;
        }
        // The platform writes a C string; the length is where the NUL is. A
        // buffer with no NUL at all would mean the platform broke its own
        // contract, and taking the whole buffer is the safe reading of that.
        buffer
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(MAX_NAME)
    }
}

#[cfg(all(windows, not(miri)))]
mod platform {
    use super::MAX_NAME;
    use std::ffi::c_void;

    #[link(name = "kernel32", kind = "raw-dylib")]
    extern "system" {
        fn GetCurrentThread() -> *mut c_void;
        fn GetThreadDescription(thread: *mut c_void, description: *mut *mut u16) -> i32;
        fn LocalFree(mem: *mut c_void) -> *mut c_void;
    }

    pub(super) fn current_thread_name(buffer: &mut [u8; MAX_NAME]) -> usize {
        let mut wide: *mut u16 = std::ptr::null_mut();
        // SAFETY: `GetCurrentThread` returns a pseudo-handle that needs no
        // release, and `description` is a writable pointer slot. On success the
        // callee owns the allocation until `LocalFree`, which happens below.
        let result = unsafe { GetThreadDescription(GetCurrentThread(), &mut wide) };
        if result < 0 || wide.is_null() {
            return 0;
        }

        let mut length = 0usize;
        // SAFETY: the platform returns a NUL-terminated UTF-16 string.
        while unsafe { *wide.add(length) } != 0 {
            length += 1;
        }
        // SAFETY: `length` units were just walked and found to be initialized.
        let units = unsafe { std::slice::from_raw_parts(wide, length) };

        let mut written = 0usize;
        for unit in char::decode_utf16(units.iter().copied()) {
            let unit = unit.unwrap_or(char::REPLACEMENT_CHARACTER);
            let mut encoded = [0u8; 4];
            let encoded = unit.encode_utf8(&mut encoded).as_bytes();
            if written + encoded.len() > MAX_NAME {
                break;
            }
            buffer[written..written + encoded.len()].copy_from_slice(encoded);
            written += encoded.len();
        }

        // SAFETY: `wide` came from `GetThreadDescription`, which documents
        // `LocalFree` as the way to release it, and it is not used again.
        unsafe { LocalFree(wide.cast()) };
        written
    }
}

/// Miri interprets the program rather than running it, and has no
/// implementation of the platform calls above. A run under Miri is checking
/// this crate's memory discipline, not its thread names, so it reports none.
#[cfg(any(miri, not(any(unix, windows))))]
mod platform {
    use super::MAX_NAME;

    pub(super) fn current_thread_name(_buffer: &mut [u8; MAX_NAME]) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arena() -> Arena {
        Arena::new()
    }

    #[test]
    fn an_id_that_names_no_row_finds_none() {
        let threads = Threads::new();
        let regions = Regions::new();

        assert!(
            threads.tally(ThreadId::UNCLAIMED).is_none(),
            "a thread with no row must not share one with row zero"
        );
        assert!(
            regions.tally(RegionId::NONE).is_none(),
            "an allocation outside every region must not land in one"
        );
        assert!(
            threads.tally(ThreadId::OVERFLOW).is_some(),
            "the overflow row exists before any thread claims a real one"
        );
        assert!(regions.tally(RegionId::OVERFLOW).is_some());
    }

    #[test]
    fn a_row_records_its_own_peak_rather_than_its_final_size() {
        let tally = Tally::new();
        tally.apply(1_000, 1, 1_000, 1);
        tally.apply(500, 1, 500, 1);
        tally.apply(-1_400, -2, 0, 0);

        let stats = tally.snapshot();
        assert_eq!(stats.total_bytes, 1_500);
        assert_eq!(stats.total_blocks, 2);
        assert_eq!(stats.curr_bytes, 100);
        assert_eq!(stats.curr_blocks, 0);
        assert_eq!(
            stats.max_bytes, 1_500,
            "the peak is the largest value the row held, not the last one"
        );
        assert_eq!(stats.max_blocks, 2);
    }

    /// The peak is a **high-water mark**, not the largest value it happened to
    /// be set to last.
    ///
    /// The test above cannot see the difference: its last growth *is* its
    /// maximum, so a plain `store` in place of `fetch_max` passes it, passes
    /// the whole suite, and passes the validator too — `curr <= max <= total`
    /// all still hold. This sequence grows, shrinks, then grows to something
    /// smaller than before, which is the only shape that separates the two.
    #[test]
    fn a_peak_that_has_been_passed_does_not_come_back_down() {
        let tally = Tally::new();
        tally.apply(1_000, 1, 1_000, 1);
        tally.apply(-900, -1, 0, 0);
        tally.apply(100, 1, 100, 1);

        let stats = tally.snapshot();
        assert_eq!(stats.curr_bytes, 200);
        assert_eq!(
            stats.max_bytes, 1_000,
            "the row's peak followed it back down; a peak is the largest value \
             it ever held, not the last one it was set to"
        );
        assert_eq!(stats.max_blocks, 1);
    }

    /// A same-size reallocation has a delta of exactly zero. It must not count
    /// as a new peak, and it must not touch the live counters at all.
    #[test]
    fn a_zero_delta_moves_nothing_live() {
        let tally = Tally::new();
        tally.apply(64, 1, 64, 1);
        tally.apply(0, 0, 64, 1);

        let stats = tally.snapshot();
        assert_eq!(
            stats.total_bytes, 128,
            "the resize is still bytes allocated"
        );
        assert_eq!(stats.total_blocks, 2);
        assert_eq!(stats.curr_bytes, 64);
        assert_eq!(stats.curr_blocks, 1);
        assert_eq!(stats.max_bytes, 64);
    }

    /// A thread whose name the platform does not have *yet* has to be asked
    /// again. On Windows the first recorded allocation of a spawned thread is
    /// std's own UTF-16 name buffer, made a few instructions before
    /// `SetThreadDescription` — so a row named on the first attempt is a row
    /// that will be blank for every worker on that platform.
    #[test]
    fn a_name_the_platform_does_not_have_yet_is_asked_for_again() {
        let arena = arena();
        let threads = Threads::new();
        let id = threads.claim(&arena, Name::EMPTY, 0);

        assert!(
            threads.wants_name(id),
            "a row with no name stopped asking after one attempt"
        );
        threads.name(id, Name::EMPTY);
        assert!(threads.wants_name(id));

        threads.name(id, Name::of("hs-worker"));
        assert!(
            !threads.wants_name(id),
            "a row that has its name kept asking for it"
        );

        let mut names = Vec::new();
        threads.visit(|row| names.push(row.name.as_bytes().to_vec()));
        assert_eq!(names, vec![b"hs-worker".to_vec()]);
    }

    /// Asking is bounded. A thread the platform never names must not pay a
    /// platform call on every allocation for the rest of its life.
    #[test]
    fn a_thread_the_platform_never_names_stops_asking() {
        let arena = arena();
        let threads = Threads::new();
        let id = threads.claim(&arena, Name::EMPTY, 0);

        // The claim itself is the first attempt.
        for _ in 1..NAME_ATTEMPTS {
            assert!(threads.wants_name(id));
            threads.name(id, Name::EMPTY);
        }
        assert!(
            !threads.wants_name(id),
            "a nameless thread asks forever after {NAME_ATTEMPTS} attempts"
        );

        let mut names = Vec::new();
        threads.visit(|row| names.push(row.name.as_bytes().to_vec()));
        assert_eq!(
            names,
            vec![Vec::<u8>::new()],
            "giving up must leave the row unnamed, not partly named"
        );
    }

    /// A name found on the first try settles immediately, which is every
    /// platform but Windows and the ordinary case there too.
    #[test]
    fn a_name_the_platform_has_settles_on_the_first_attempt() {
        let arena = arena();
        let threads = Threads::new();
        let id = threads.claim(&arena, Name::of("main"), 0);
        assert!(!threads.wants_name(id));
    }

    /// The shared row stands for many threads with many names, so it is never
    /// named and must never spend attempts trying.
    #[test]
    fn the_overflow_row_never_asks_for_a_name() {
        let threads = Threads::new();
        assert!(!threads.wants_name(ThreadId::OVERFLOW));
        assert!(!threads.wants_name(ThreadId::UNCLAIMED));
    }

    #[test]
    fn claiming_hands_out_rows_in_order_and_then_overflows() {
        let arena = arena();
        let threads = Threads::new();

        let first = threads.claim(&arena, Name::of("worker"), 7);
        let second = threads.claim(&arena, Name::of("other"), 9);
        assert_ne!(first, second);
        assert_eq!(threads.len(), 2);

        let mut seen = Vec::new();
        threads.visit(|row| seen.push((row.id, row.name.as_bytes().to_vec(), row.first_seen)));
        assert_eq!(
            seen,
            vec![
                (first, b"worker".to_vec(), 7),
                (second, b"other".to_vec(), 9)
            ]
        );
    }

    /// The overflow row appears only once something has landed in it, so a run
    /// that never overflowed does not carry a row saying it did.
    #[test]
    fn the_overflow_row_is_reported_only_once_it_is_used() {
        let arena = arena();
        let threads = Threads::new();
        threads.claim(&arena, Name::of("worker"), 0);

        let mut rows = 0;
        threads.visit(|_| rows += 1);
        assert_eq!(rows, 1);

        threads
            .tally(ThreadId::OVERFLOW)
            .expect("the overflow row always exists")
            .apply(16, 1, 16, 1);

        let mut overflowed = Vec::new();
        threads.visit(|row| overflowed.push(row.id));
        assert_eq!(overflowed, vec![ThreadId(0), ThreadId::OVERFLOW]);
    }

    #[test]
    fn interning_the_same_region_name_twice_gives_one_row() {
        let arena = arena();
        let regions = Regions::new();

        let first = regions.intern(&arena, "parsing", 1);
        let again = regions.intern(&arena, "parsing", 2);
        let other = regions.intern(&arena, "planning", 3);

        assert_eq!(first, again, "a region name is a row, not an instance");
        assert_ne!(first, other);
        assert_eq!(regions.len(), 2);

        regions.enter(first);
        regions.enter(again);
        regions.leave(first);

        let mut rows = Vec::new();
        regions.visit(|row| {
            rows.push((
                row.id,
                row.name.as_bytes().to_vec(),
                row.entries,
                row.active,
            ))
        });
        assert_eq!(
            rows,
            vec![(first, b"parsing".to_vec(), 2, 1)],
            "a name that was interned and never entered was reported as a phase \
             the program went through"
        );
        assert_eq!(
            rows[0].2, 2,
            "the row counts every entry, including the one still open"
        );
        assert_eq!(
            regions.len(),
            2,
            "the row still exists and still costs its arena bytes; it is only \
             the profile that leaves it out"
        );
        assert!(!other.is_none());
    }

    /// Two names that differ only past the limit become one row. That is a
    /// consequence of bounding the name, and the test exists so the bound stays
    /// a decision rather than a surprise.
    #[test]
    fn names_are_cut_to_the_limit_and_two_that_share_a_prefix_share_a_row() {
        let arena = arena();
        let regions = Regions::new();

        let long = "r".repeat(MAX_NAME);
        let longer = format!("{long}-and-more");
        let first = regions.intern(&arena, &long, 0);
        let second = regions.intern(&arena, &longer, 0);

        assert_eq!(first, second);
        regions.enter(first);
        let mut names = Vec::new();
        regions.visit(|row| names.push(row.name.as_bytes().to_vec()));
        assert_eq!(names, vec![long.as_bytes().to_vec()]);
    }

    /// A multi-byte character straddling the limit must not be cut in half: the
    /// name is a `&str` the program chose, and half a character is not one.
    #[test]
    fn a_name_is_cut_at_a_character_boundary() {
        let mut text = "a".repeat(MAX_NAME - 1);
        text.push('é'); // two bytes, so it straddles `MAX_NAME`
        let name = Name::of(&text);

        assert_eq!(name.as_bytes().len(), MAX_NAME - 1);
        assert!(
            std::str::from_utf8(name.as_bytes()).is_ok(),
            "a truncated name must still be the string it was cut from"
        );
    }

    #[test]
    fn a_full_region_table_folds_the_rest_into_one_row() {
        let arena = arena();
        let regions = Regions::new();

        for index in 0..MAX_REGIONS {
            let id = regions.intern(&arena, &format!("region {index}"), 0);
            assert!(!id.is_overflow(), "row {index} should have fitted");
        }
        let overflowed = regions.intern(&arena, "one too many", 0);
        assert_eq!(overflowed, RegionId::OVERFLOW);
        assert_eq!(
            regions.intern(&arena, "another too many", 0),
            RegionId::OVERFLOW,
            "every name past the limit shares the one row"
        );
    }

    /// Whatever the platform reports, it must fit in the buffer and must not be
    /// a name the caller has to bounds-check afterwards.
    #[test]
    fn the_platform_name_fits_the_buffer_it_was_given() {
        let mut buffer = [0u8; MAX_NAME];
        let len = current_thread_name(&mut buffer);
        assert!(len <= MAX_NAME);
        assert!(
            buffer[..len].iter().all(|&byte| byte != 0),
            "the length must stop at the terminator, not include it"
        );
    }

    /// The name a thread was given is the name the row carries. Skipped where
    /// the platform is not asked (Miri) or would not answer.
    #[test]
    #[cfg_attr(miri, ignore = "Miri does not implement the platform's naming calls")]
    fn a_named_thread_reports_the_name_the_platform_has() {
        let name = std::thread::Builder::new()
            .name("heapscope-probe".to_string())
            .spawn(|| {
                let mut buffer = [0u8; MAX_NAME];
                let len = current_thread_name(&mut buffer);
                String::from_utf8_lossy(&buffer[..len]).into_owned()
            })
            .expect("spawning a thread")
            .join()
            .expect("the probe thread panicked");

        // Linux caps a thread name at 15 bytes in the kernel, so this checks a
        // prefix rather than the whole string: the limit is the platform's and
        // the profile reports what the platform kept.
        assert!(
            !name.is_empty() && "heapscope-probe".starts_with(&name),
            "expected the thread's own name, got {name:?}"
        );
    }
}
