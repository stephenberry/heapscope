//! The one binding for `dladdr`, which two backends ask two questions of.
//!
//! [`dynamic`](super::dynamic) wants the symbol an address falls in;
//! [`modules`](super::modules) wants the path of the image the loader itself is
//! in, which is the one image dyld leaves out of its own list. Those are
//! different questions with one answer, and the answer is a `repr(C)` structure
//! that has to match the platform exactly — four pointers, in an order this
//! crate does not get to choose. Declaring it twice, as this used to, is two
//! places to be wrong about a layout that is not ours.

use std::ffi::{c_char, c_void, CStr};

// Declared against whatever the process already links: libSystem on Apple, and
// libc on a glibc since 2.34, which merged libdl into it. Deliberately no
// `#[link(name = "dl")]`. On those two it would name a library that is no longer
// separate and buy nothing; anywhere else it would make the link depend on a
// library that may not be there, to reach one call whose *failure* both callers
// already handle by falling back to the module map. A build that does not happen
// is a worse outcome than the one this module is written to survive.
extern "C" {
    fn dladdr(address: *const c_void, info: *mut DlInfo) -> i32;
}

/// POSIX `Dl_info`. The same four fields in the same order on glibc, musl, and
/// Apple.
///
/// The fields stay private. Every one of them is a pointer into memory owned by
/// the loader, so reading one is an unsafe operation that has to be justified
/// against what `dladdr` promises — and that justification is worth writing once
/// rather than once per caller.
///
/// The pointers stay valid until the image they belong to is unloaded, so a
/// value of this type is only meaningful for as long as that holds: a `dlclose`
/// between [`DlInfo::of`] and a read would leave every one of them dangling.
/// Both callers read what they want and drop it within a few lines, which is why
/// the type carries no lifetime saying so — it would have nothing to borrow
/// from. This is a different hazard from the one
/// [`modules`](super::modules) documents, which is a concurrent `dlopen`
/// producing an image the map missed.
#[repr(C)]
pub(crate) struct DlInfo {
    file_name: *const c_char,
    file_base: *mut c_void,
    symbol_name: *const c_char,
    symbol_address: *mut c_void,
}

impl DlInfo {
    /// What the loader knows about `address`, or `None` if it is in no loaded
    /// image.
    ///
    /// Safe to call with any address, mapped or not: `dladdr` reads no memory at
    /// `address` — it compares the value against the loaded images — which
    /// matters because a truncated stack walk can produce one. Taken as a
    /// pointer rather than a `usize` so that a caller who has one keeps its
    /// provenance rather than laundering it through an integer.
    pub(crate) fn of(address: *const c_void) -> Option<DlInfo> {
        let mut info = DlInfo {
            file_name: std::ptr::null(),
            file_base: std::ptr::null_mut(),
            symbol_name: std::ptr::null(),
            symbol_address: std::ptr::null_mut(),
        };
        // SAFETY: `info` is a live, correctly shaped `Dl_info`, and the address
        // is only ever compared. See above.
        let found = unsafe { dladdr(address, &mut info) };
        (found != 0).then_some(info)
    }

    /// The path of the image the address is in, empty if the loader reported
    /// none.
    ///
    /// Apple only, because that is the only module map that asks: dyld leaves
    /// itself out of its own image list, and this is how that one image gets a
    /// path. Every other backend reads its map from something that already
    /// carries one. Without the gate this is dead code on Linux.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn image_path(&self) -> String {
        // SAFETY: `dladdr` reported success, so `file_name` is null or a
        // NUL-terminated path in loader-owned memory, which is live for as long
        // as the image is — see the note on the type.
        unsafe { string_at(self.file_name) }.unwrap_or_default()
    }

    /// Where the nearest preceding symbol starts, if the loader reported one.
    ///
    /// Separate from [`DlInfo::name`] so that a caller can reject on the address
    /// before paying for the name: a symbol that starts *after* the address it
    /// supposedly contains is not describing that address, and finding that out
    /// should not cost a heap allocation on a path documented as cheap enough to
    /// call per frame.
    pub(crate) fn symbol_start(&self) -> Option<usize> {
        (!self.symbol_address.is_null()).then_some(self.symbol_address as usize)
    }

    /// What the loader calls the symbol, if it named one.
    ///
    /// `None` is the stripped-binary case, where `dladdr` succeeds with a null
    /// name — see [`dynamic`](super::dynamic)'s module documentation for why
    /// that is not an answer.
    pub(crate) fn name(&self) -> Option<String> {
        // SAFETY: `dladdr` reported success, so `symbol_name` is null or a
        // NUL-terminated name in loader-owned memory, which is live for as long
        // as the image is — see the note on the type.
        unsafe { string_at(self.symbol_name) }
    }
}

/// Reads a C string, lossily. `None` for a null pointer.
///
/// `pub(crate)` because [`modules`](super::modules) reads NUL-terminated names
/// out of the loader too — from `_dyld_get_image_name` on Apple and from
/// `dl_iterate_phdr`'s `dl_phdr_info` elsewhere — and had its own copy of this,
/// differing by the single line that decides what a null pointer means.
///
/// # Safety
///
/// `pointer` must be null or point to a NUL-terminated string that stays valid
/// for the duration of the call.
pub(crate) unsafe fn string_at(pointer: *const c_char) -> Option<String> {
    if pointer.is_null() {
        return None;
    }
    // SAFETY: the caller guarantees a valid NUL-terminated string.
    Some(
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned(),
    )
}
