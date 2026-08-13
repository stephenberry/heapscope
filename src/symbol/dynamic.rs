//! Asking the running process what an address is called.
//!
//! This is tier 1 of the three described in PLAN.md section 6.1, and it is the
//! convenience layer rather than the foundation. The foundation is the [module
//! map](super::modules), which works on a stripped binary, on another machine,
//! and after the process is gone. What this adds is that when the symbols
//! *happen* to be present, a summary printed to stderr at exit can say
//! `core::fmt::write` instead of `program+0x2c1f0`, with no second tool and no
//! second step.
//!
//! # Success is not the same as an answer
//!
//! The one thing that has to be got right here is that `dladdr` reports
//! **success with a null symbol name** on a stripped image. Measured, on the
//! same binary before and after `strip -x`:
//!
//! ```text
//! pub_fn   rc=1 sym=_ZN2dl6pub_fn17ha505993b13ec8b66E   off=0
//! --- after strip -x ---
//! pub_fn   rc=1 sym=<null>                              off=4364011916
//! ```
//!
//! A caller that checks only the return code dereferences a null pointer, and
//! one that checks the pointer but keeps the offset reports a nonsense number.
//! Both are the same mistake: treating "the call worked" as "the address was
//! identified". [`lookup`] returns `None` for that case, and the frame then
//! renders as module and offset, which is true and resolvable.
//!
//! # An answer is not the same as a *close* answer
//!
//! `dladdr` matches the nearest preceding symbol in the dynamic symbol table,
//! and it does not say how near. On a binary stripped of its static symbols,
//! every address in a large private function resolves to whatever exported
//! symbol happens to precede it, which can be thousands of bytes away and in an
//! unrelated part of the file. There is no threshold that separates the two
//! cases: `+0x40` is normal for a real match and plausible for a bad one.
//!
//! So this does not guess. The offset from the symbol is carried out to the
//! renderer and printed, because `parse_header+0x2f18` is a reader's only clue
//! that the name is not to be believed, and hiding it is what would make the
//! output misleading.
//!
//! # `dladdr` will name an address that is in nothing at all
//!
//! Measured on macOS 15, arm64, from a Rust binary and again from a C one:
//!
//! ```text
//! usize::MAX     rc=1  sname=_MergedGlobals.1385  saddr=0x104bc8d28  off=0xfffffffefb4372d7
//! usize::MAX-1   rc=0  sname=<null>
//! usize::MAX-2   rc=0  sname=<null>
//! main + 4GiB    rc=0  sname=<null>
//! ```
//!
//! Exactly one value, because dyld uses `(void *)-1` as a sentinel and the
//! all-ones address matches it before the range check runs. Two things make
//! that worse than a curiosity. It is not a random address: `0xFFFF_FFFF_FFFF_FFFF`
//! is what a truncated stack walk, a poisoned slot, or a misaligned frame
//! pointer produces, so the one address `dladdr` answers wrongly for is the one
//! a profiler is most likely to ask about. And the answer is *confident* — a
//! real symbol name from a real image, with nothing in the result marking it as
//! doubtful except an offset of 18 quintillion.
//!
//! [`lookup`] does not attempt to correct this, because from inside a single
//! `dladdr` call there is nothing to correct it against. The renderer does:
//! [`Symbolized`](super::Symbolized) asks the module map whether the address is
//! in a real image before it asks what the address is called, and the map holds
//! measured bounds for every image's executable region.

/// An address, named.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    /// The name as the linker recorded it, still mangled.
    ///
    /// Demangling is the renderer's business, not the lookup's: the same symbol
    /// table holds Rust, C, and C++ names, and deciding what a name means is a
    /// separate concern from finding it. It is also raw in the stronger sense —
    /// these bytes came out of a file that may be truncated or mismatched, so
    /// nothing may print them without screening. The emitters do that; see
    /// `output::push_display`.
    pub name: String,
    /// How far `address` is past the start of the symbol.
    ///
    /// Zero means the address *is* the symbol's first byte, which is the normal
    /// case for the innermost frame of a call to a function that has not been
    /// inlined. A large value means the match is doubtful; see the module
    /// documentation.
    pub offset: usize,
}

/// Names `address`, if this process can.
///
/// `None` when the platform has no symbol lookup, when the platform reports the
/// address as belonging to nothing, or when the lookup succeeded without
/// producing a usable name — the stripped-binary case, which is common enough
/// that it is the reason the module map exists.
///
/// **A `Some` is not a promise that `address` is in a loaded image.** See the
/// module documentation: `dladdr` names `(void *)-1`. Callers that have a module
/// map should check it first, as [`Symbolized`] does.
///
/// Cheap enough to call per frame on Unix and distinctly not cheap on Windows,
/// where every call goes through one process-wide lock into dbghelp. Callers
/// that resolve a whole profile should cache by address; [`Symbolized`] does.
///
/// [`Symbolized`]: super::Symbolized
pub fn lookup(address: usize) -> Option<Symbol> {
    if !enabled() {
        return None;
    }
    imp::lookup(address)
}

/// The name of the variable that turns in-process symbolization off.
///
/// Set it to `0`, `off`, `no`, or `false` and [`lookup`] finds nothing, so
/// frames render exactly as [`ModuleOffsets`](super::ModuleOffsets) renders
/// them: address, image, and offset, resolvable by `atos`, `addr2line`, or
/// `llvm-symbolizer` afterwards. Nothing else changes. Any other value, or none
/// at all, leaves it on.
///
/// # Why an environment variable, when M5 is where configuration is designed
///
/// Because the reasons to want it off are discovered at the moment a program
/// will not shut down, usually on a machine nobody is building on, and a
/// builder option cannot be reached from there. All three known reasons are of
/// that shape:
///
/// - **dbghelp can go to the network.** A null search path takes its default,
///   which honours `_NT_SYMBOL_PATH`; if that names a symbol server, writing a
///   profile at process exit blocks on it.
/// - **Byte-identical output across machines.** Two runs of the same program on
///   two machines produce different frame text when one has symbols and the
///   other does not, which makes profiles awkward to diff.
/// - **Emulators.** `ci/windows-under-wine.sh` cannot execute `SymFromAddr` at
///   all — see the note in that script — so without this the entire
///   profile-writing half of the Windows suite is unrunnable outside CI.
///
/// The builder option M5 adds supersedes this for programs that can be
/// recompiled; it does not replace it.
pub const DISABLE_VARIABLE: &str = "HEAPSCOPE_SYMBOLIZE";

/// Whether the platform may be asked. Read from the environment once.
fn enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    const UNREAD: u8 = 0;
    const ON: u8 = 1;
    const OFF: u8 = 2;

    // Racing readers both call `var_os` and both store the same answer, which
    // is why this needs no lock. A program that changes the variable while a
    // profile is being written gets one answer or the other, and neither is
    // wrong enough to serialize every frame over.
    static STATE: AtomicU8 = AtomicU8::new(UNREAD);
    match STATE.load(Ordering::Relaxed) {
        ON => return true,
        OFF => return false,
        _ => {}
    }

    let setting = std::env::var_os(DISABLE_VARIABLE);
    // Absent, or not text this can read: leave it on. The failure mode of
    // guessing wrong in this direction is a profile with fewer names in it, and
    // in the other direction it is a profile nobody asked to change.
    let on = !setting
        .as_deref()
        .and_then(|value| value.to_str())
        .is_some_and(reads_as_off);
    STATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
    on
}

/// Whether a setting of [`DISABLE_VARIABLE`] means "off".
///
/// Split out so that a test can reach the rule itself. `enabled` caches its
/// answer in a process-wide static on first use, so a test that set the
/// variable and called [`lookup`] would either be the first to touch it — and
/// change the answer every other test in the process gets — or not be first,
/// and prove nothing.
///
/// `pub(crate)` so that [`crate::stats::is_off`] can be held to *this* rule
/// rather than to a copy of it. The two drifted apart once already — this one
/// folded case and that one did not, so `HEAPSCOPE_UPDATE_BASELINE=FALSE` read
/// as on and rewrote every baseline it was meant to check. Two spellings of one
/// idea in one crate is a variable nobody can remember.
pub(crate) fn reads_as_off(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "off" | "no" | "false"
    )
}

#[cfg(all(unix, not(miri)))]
mod imp {
    use super::Symbol;
    use crate::symbol::dl::DlInfo;

    pub(super) fn lookup(address: usize) -> Option<Symbol> {
        let info = DlInfo::of(address as *const std::ffi::c_void)?;

        // The address is checked before the name is read, because reading the
        // name allocates and this is called per frame. A name whose start is
        // unknown cannot be checked, and one that starts *after* the address it
        // supposedly contains is not describing this address at all. Either way
        // there is nothing to report that would be better than the module map's
        // answer.
        let start = info.symbol_start()?;
        if start > address {
            return None;
        }

        // `None` is the stripped-binary case, where `dladdr` succeeds with a
        // null name. See the module documentation: that is not an answer, and
        // neither is an empty one.
        let name = info.name()?;
        if name.is_empty() {
            return None;
        }

        Some(Symbol {
            name,
            offset: address - start,
        })
    }
}

#[cfg(all(windows, not(miri)))]
mod imp {
    use super::Symbol;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::internals::lock::RawLock;

    // `raw-dylib` means no import library is needed to build against dbghelp,
    // which keeps the build working with nothing installed beyond a toolchain.
    #[link(name = "dbghelp", kind = "raw-dylib")]
    extern "system" {
        fn SymGetOptions() -> u32;
        fn SymSetOptions(options: u32) -> u32;
        fn SymInitializeW(process: *mut c_void, search_path: *const u16, invade: i32) -> i32;
        fn SymFromAddr(
            process: *mut c_void,
            address: u64,
            displacement: *mut u64,
            symbol: *mut SymbolInfo,
        ) -> i32;
    }

    #[link(name = "kernel32", kind = "raw-dylib")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
    }

    /// `SYMOPT_UNDNAME` — undecorate MSVC C++ names. Rust's manglings both
    /// start with an underscore rather than `?`, so dbghelp leaves them exactly
    /// as they are and [`crate::demangle`] still sees what the linker wrote.
    const SYMOPT_UNDNAME: u32 = 0x0000_0002;
    /// `SYMOPT_DEFERRED_LOADS` — do not read a module's symbols until an
    /// address in it is actually asked about. A profile usually touches a
    /// handful of images out of dozens.
    const SYMOPT_DEFERRED_LOADS: u32 = 0x0000_0004;
    /// `SYMOPT_FAIL_CRITICAL_ERRORS` — a missing or unreadable symbol file is a
    /// failed lookup, not a modal dialog on a machine with no one at the
    /// keyboard.
    const SYMOPT_FAIL_CRITICAL_ERRORS: u32 = 0x0000_0200;
    /// `SYMOPT_NO_PROMPTS` — the same, for the symbol server's own prompting.
    const SYMOPT_NO_PROMPTS: u32 = 0x0008_0000;

    /// `MAX_SYM_NAME`. A Rust v0 name with nested generics gets long; this is
    /// what dbghelp itself documents as the ceiling.
    const MAX_NAME: usize = 2000;

    /// `SYMBOL_INFO`, whose `Name` is a flexible array member that the caller is
    /// expected to allocate past the end of. `size_of_struct` must be the size
    /// of *this* structure, counting the single declared name byte and the
    /// trailing padding but not the extra room, which is what dbghelp uses to
    /// tell the structure versions apart.
    #[repr(C)]
    struct SymbolInfo {
        size_of_struct: u32,
        type_index: u32,
        reserved: [u64; 2],
        index: u32,
        size: u32,
        module_base: u64,
        flags: u32,
        value: u64,
        address: u64,
        register: u32,
        scope: u32,
        tag: u32,
        name_length: u32,
        max_name_length: u32,
        name: [u8; 1],
    }

    /// The size dbghelp uses to tell `SYMBOL_INFO`'s versions apart, and the
    /// one number here that cannot be got wrong quietly: a mismatch is a
    /// structure written to the wrong offsets, not a compile error.
    ///
    /// 88 on every Windows target, because every field is fixed-width — there
    /// is no pointer in it — so the layout does not vary with the pointer size.
    /// Checked at compile time because no test in this repository runs on
    /// Windows natively.
    const _: () = assert!(std::mem::size_of::<SymbolInfo>() == 88);

    /// `SYMBOL_INFO` with the room its name field expects to be given.
    #[repr(C)]
    struct SymbolBuffer {
        info: SymbolInfo,
        rest: [u8; MAX_NAME],
    }

    /// dbghelp is documented as single-threaded: every call for a process must
    /// be serialized by the caller. Nothing here is on the allocation path, so
    /// the contention this creates is between two threads writing profiles at
    /// once, which is already rare and already slow.
    ///
    /// The other thing dbghelp must not be called from is `DllMain`, which runs
    /// under the loader lock that dbghelp itself takes. This crate's automatic
    /// profile is written from an `atexit` handler (see `src/profiler.rs`),
    /// which the CRT runs before it unloads anything, so the constraint is met
    /// — but it is met by a decision made elsewhere for other reasons, which is
    /// exactly the kind of thing that stops being true silently.
    static DBGHELP: RawLock = RawLock::new();

    /// Whether `SymInitializeW` has been called yet.
    ///
    /// Not "and whether it worked": see `open_session`. Read and written only
    /// under `DBGHELP`; an atomic rather than a plain `static mut` because that
    /// is what a mutable static costs in safe Rust, not because the ordering is
    /// doing any work.
    static OPENED: AtomicBool = AtomicBool::new(false);

    /// Opens the dbghelp session, once, and does not report whether it worked.
    ///
    /// Must be called with `DBGHELP` held.
    ///
    /// # Why a failed `SymInitializeW` is not a refusal
    ///
    /// This process may already have a dbghelp session, and in a Rust program it
    /// very often does: `std`'s backtrace support calls
    /// `SymInitializeW(GetCurrentProcess(), null, TRUE)` and never calls
    /// `SymCleanup`, so anything that has printed a panic backtrace or built a
    /// `Backtrace` has opened one. `SymInitialize` documents that *"a process
    /// that calls SymInitialize should not call it again unless it calls
    /// SymCleanup first"*, and the second call fails.
    ///
    /// Treating that failure as "dbghelp is unavailable" would give up in
    /// exactly the case where it is *most* available — the session is open and
    /// its modules are loaded — and would do so permanently and silently, so a
    /// program that happened to panic earlier would get a profile with no names
    /// in it and no indication why. `SymFromAddr` asks only that the handle
    /// *"must have been previously passed to the SymInitialize function"*, which
    /// it has been, by whoever got there first. So the lookup proceeds either
    /// way and a genuinely absent dbghelp shows up as `SymFromAddr` returning
    /// zero, which is a cheap answer rather than a crash.
    ///
    /// # What is not solved
    ///
    /// dbghelp requires every call for a process to be serialized. `DBGHELP`
    /// serializes this crate's; `std` has its own lock, and the two know nothing
    /// about each other. A thread formatting a panic backtrace while another
    /// writes a profile makes concurrent dbghelp calls, which Microsoft says
    /// *"will likely result in unexpected behavior or memory corruption"*.
    /// Closing that needs a lock the two crates share, and there is none.
    /// Recorded rather than papered over.
    fn open_session(process: *mut c_void) {
        if OPENED.load(Ordering::Relaxed) {
            return;
        }

        // OR-ed into whatever is already set, never assigned. The option mask is
        // process-wide and shared with every other user of dbghelp in the
        // process, so replacing it would silently change how `std` resolves a
        // backtrace — and would do it even on the path where this crate then
        // decides it cannot proceed.
        //
        // Set before initializing: `SymInitializeW` reads the options while
        // deciding how much work to do up front.
        //
        // SAFETY: no pointer arguments. `SymGetOptions` cannot fail; both are
        // called under `DBGHELP`.
        unsafe {
            let existing = SymGetOptions();
            SymSetOptions(
                existing
                    | SYMOPT_UNDNAME
                    | SYMOPT_DEFERRED_LOADS
                    | SYMOPT_FAIL_CRITICAL_ERRORS
                    | SYMOPT_NO_PROMPTS,
            );
        }

        // A null search path takes dbghelp's default, which is the directory
        // holding the executable followed by `_NT_SYMBOL_PATH`. Honouring that
        // variable is what lets someone point at a symbol store they already
        // have; combined with deferred loads, an unset variable costs nothing.
        // `HEAPSCOPE_SYMBOLIZE=0` is the way out when it names a server that is
        // slow or unreachable.
        //
        // `invade` enumerates the modules already loaded. Without it every
        // lookup would first have to find and register its own module, which is
        // work this crate would be doing worse than the library that is for it.
        //
        // SAFETY: a process handle and a null search path, both of which the
        // call documents as accepted.
        unsafe { SymInitializeW(process, std::ptr::null(), 1) };

        // The result is deliberately discarded. See above.
        OPENED.store(true, Ordering::Relaxed);
    }

    pub(super) fn lookup(address: usize) -> Option<Symbol> {
        // SAFETY: returns a pseudo-handle; no failure mode.
        let process = unsafe { GetCurrentProcess() };

        let _held = DBGHELP.lock();
        open_session(process);

        // Zeroed rather than field-by-field: `SYMBOL_INFO` has padding and
        // reserved words that dbghelp is entitled to look at.
        //
        // SAFETY: `SymbolBuffer` is `repr(C)` over integers and byte arrays, for
        // which every bit pattern is valid.
        let mut buffer: SymbolBuffer = unsafe { std::mem::zeroed() };
        buffer.info.size_of_struct = std::mem::size_of::<SymbolInfo>() as u32;
        // The single byte declared inside `SymbolInfo` is usable too, and this
        // is the length dbghelp will write up to, NUL included.
        buffer.info.max_name_length = (MAX_NAME + 1) as u32;

        let mut displacement = 0u64;

        // Both pointers below are derived from the *whole* buffer and then cast,
        // never from the `info` field or the `name` field. That is not style.
        // A flexible array member is a C idiom with no Rust equivalent: dbghelp
        // writes up to `max_name_length` bytes starting at `name`, which runs
        // past both of those fields and into `rest`. A pointer made by
        // reborrowing `buffer.info` or `buffer.info.name` carries provenance
        // for that field alone, and reading or writing outside it through such
        // a pointer is undefined behaviour regardless of the bytes being there.
        // Casting a whole-buffer pointer keeps provenance over the whole
        // allocation, which is what makes this defined.
        //
        // Ordinarily Miri would be the thing that caught getting this wrong. It
        // cannot: this module is `cfg`-ed out under Miri, because `SymFromAddr`
        // is a foreign function with no shim. So the reasoning is written down
        // instead.
        let whole = &mut buffer as *mut SymbolBuffer;

        // SAFETY: `whole` points at a live `SymbolBuffer`, whose leading bytes
        // are the `SYMBOL_INFO` this call is being given and whose remaining
        // `MAX_NAME` bytes are the room `max_name_length` promises. dbghelp
        // reads no memory at `address` — it consults the loaded modules' symbol
        // tables — so an unmapped address from a bad stack walk is safe to ask
        // about.
        let found = unsafe {
            SymFromAddr(
                process,
                address as u64,
                &mut displacement,
                whole.cast::<SymbolInfo>(),
            )
        };
        if found == 0 {
            return None;
        }

        // `name_length` excludes the terminator, and dbghelp has been known to
        // report a length past the buffer it was given. Clamp rather than trust.
        let length = (buffer.info.name_length as usize).min(MAX_NAME);
        // SAFETY: a fresh pointer over the whole buffer, offset to where the
        // name begins. `length` is at most `MAX_NAME`, and `SymbolBuffer` has
        // `MAX_NAME` bytes after the one declared inside `SymbolInfo`, so the
        // slice ends inside the allocation. The bytes are initialised because
        // the buffer was zeroed before the call.
        let name = unsafe {
            let start = (&buffer as *const SymbolBuffer)
                .cast::<u8>()
                .add(std::mem::offset_of!(SymbolInfo, name));
            std::slice::from_raw_parts(start, length)
        };
        // Stop at a NUL rather than carrying one into the output, in case the
        // reported length disagrees with the string.
        let name = match name.iter().position(|&byte| byte == 0) {
            Some(end) => &name[..end],
            None => name,
        };
        if name.is_empty() {
            return None;
        }

        Some(Symbol {
            name: String::from_utf8_lossy(name).into_owned(),
            // A displacement wider than a pointer would mean dbghelp matched a
            // symbol in a different address space, which it cannot; saturating
            // keeps the arithmetic total on a 32-bit target regardless.
            offset: usize::try_from(displacement).unwrap_or(usize::MAX),
        })
    }
}

/// No symbol lookup to reach: a target with no backend, or Miri.
///
/// Under Miri, `dladdr` and `SymFromAddr` are foreign functions with no shim,
/// and reaching one aborts the whole test binary rather than failing a test —
/// the same trap the module map documents. Reporting nothing found is the
/// truthful answer under an interpreter that has no symbol table to consult,
/// and it keeps every test that renders a frame runnable.
#[cfg(any(miri, not(any(unix, windows))))]
mod imp {
    use super::Symbol;

    pub(super) fn lookup(_address: usize) -> Option<Symbol> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A function this test can take the address of, and which is therefore in
    /// the symbol table of the test binary under any ordinary build.
    #[inline(never)]
    fn a_function_with_a_findable_name() -> usize {
        // Kept from being folded with an identical function by the optimiser,
        // which would make the name found here belong to the other one.
        std::hint::black_box(0x5EED)
    }

    /// The whole point of tier 1, on the one platform configuration the test is
    /// running on.
    ///
    /// Deliberately not asserted unconditionally: a stripped binary, a static
    /// musl target, and Miri all legitimately find nothing, and a test that
    /// demanded a name would be testing the build configuration rather than the
    /// code. What is asserted is that *if* something is found, it is the right
    /// something.
    #[test]
    fn a_local_function_is_named_if_this_build_has_names_at_all() {
        let address = a_function_with_a_findable_name as *const () as usize;
        let Some(symbol) = lookup(address) else {
            return;
        };

        assert!(
            symbol.name.contains("a_function_with_a_findable_name"),
            "the address of a named function resolved to `{}`",
            symbol.name
        );
        assert_eq!(
            symbol.offset, 0,
            "the address *is* the symbol, so nothing should separate them"
        );
    }

    /// The offset has to be measured from the symbol, not invented.
    #[test]
    fn an_address_inside_a_function_reports_how_far_in_it_is() {
        let start = a_function_with_a_findable_name as *const () as usize;
        let Some(base) = lookup(start) else {
            return;
        };
        assert_eq!(base.offset, 0);

        // Four bytes in is inside the first instruction on x86-64 and is the
        // second instruction on a fixed-width instruction set. Either way it is
        // the same function, and it is four bytes past its start.
        let Some(inside) = lookup(start + 4) else {
            return;
        };
        assert_eq!(inside.name, base.name);
        assert_eq!(inside.offset, 4);
    }

    /// A stack walk that goes wrong produces addresses in nothing at all, and
    /// this must answer rather than fault.
    ///
    /// Note what is *not* asserted: that nothing is found. See the module
    /// documentation — `dladdr` names `(void *)-1` on macOS, and the guard
    /// against that lives in the renderer, which has a module map to check
    /// against. Here the claim is only that every one of these returns.
    #[test]
    fn an_address_in_no_image_does_not_fault() {
        // Page zero is never mapped; the rest are addresses no ordinary process
        // has code at, including the two that a truncated stack walk produces.
        for address in [0, 1, 0x1000, usize::MAX, usize::MAX - 1, 1 << 47] {
            let _ = lookup(address);
        }
    }

    /// The specific quirk, pinned so that it is a recorded measurement rather
    /// than a remembered one, and so that the renderer's guard has something to
    /// point at.
    ///
    /// Written as a soft check on purpose: the behaviour belongs to a particular
    /// dyld, and a future one refusing `-1` outright would be a fix, not a
    /// regression. What must never change is that a caller cannot rely on a
    /// `Some` meaning the address was real, so what is asserted is the part that
    /// is true either way.
    #[test]
    fn a_named_sentinel_address_is_still_reported_as_out_of_range_by_the_module_map() {
        let modules = crate::symbol::modules::capture();
        if modules.is_empty() {
            // Miri, or a platform with no loader to ask.
            return;
        }
        assert!(
            crate::symbol::modules::containing(&modules, usize::MAX).is_none(),
            "the module map claims an image holds the last byte of the address space"
        );
        if let Some(symbol) = lookup(usize::MAX) {
            assert!(
                symbol.offset > 1 << 40,
                "`dladdr` named the sentinel address `{}` at a plausible offset \
                 of {:#x}, which would make it indistinguishable from a real match",
                symbol.name,
                symbol.offset
            );
        }
    }

    /// The off switch recognises what it documents.
    ///
    /// The parsing, not the effect: see `reads_as_off` for why the effect
    /// cannot be tested in-process. It is checked where it can be —
    /// `ci/windows-under-wine.sh` runs the entire Windows suite with the
    /// variable set, on a machine where leaving it unset kills the test
    /// process, so the whole run is the assertion.
    #[test]
    fn the_off_switch_recognises_the_spellings_it_documents() {
        for value in ["0", "off", "no", "false", "OFF", "No", " 0 ", "FALSE"] {
            assert!(reads_as_off(value), "`{value}` should turn it off");
        }
        // Anything else leaves symbolization on, including the spellings a
        // reader might expect to work. Guessing at "disabled" or "" would mean
        // a variable someone set for another program silently changing what
        // this one writes.
        for value in ["1", "on", "yes", "true", "", "disabled", "0x0"] {
            assert!(!reads_as_off(value), "`{value}` should leave it on");
        }
    }

    /// Symbolization is on unless somebody turned it off.
    ///
    /// Worth a test of its own because the failure is invisible: inverting the
    /// default so that tier 1 does nothing at all leaves every other test in
    /// this repository green. Nothing asserts that a name *was* found — nothing
    /// can, portably, since `dladdr` on ELF names almost nothing in an
    /// executable — so the switch is the only place the claim can be pinned.
    ///
    /// Written against whatever the environment actually holds, so that it is
    /// also a real check under `ci/windows-under-wine.sh`, which runs the whole
    /// suite with the variable set to `0`.
    #[test]
    fn symbolization_is_on_unless_it_was_turned_off() {
        match std::env::var_os(DISABLE_VARIABLE) {
            None => assert!(
                enabled(),
                "`{DISABLE_VARIABLE}` is unset, so in-process symbolization \
                 should be on; tier 1 is doing nothing at all"
            ),
            Some(value) => {
                let off = value.to_str().is_some_and(reads_as_off);
                assert_eq!(
                    enabled(),
                    !off,
                    "`{DISABLE_VARIABLE}` is set to {value:?} and `enabled` disagrees"
                );
            }
        }
    }

    /// Called once per frame of a profile that can have millions, so it has to
    /// be callable repeatedly without accumulating anything.
    #[test]
    fn repeated_lookups_agree_with_each_other() {
        let address = a_function_with_a_findable_name as *const () as usize;
        let first = lookup(address);
        for _ in 0..64 {
            assert_eq!(lookup(address), first);
        }
    }
}
