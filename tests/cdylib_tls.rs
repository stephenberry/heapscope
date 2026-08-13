//! PLAN.md section 4.7: the cdylib thread-local path.
//!
//! The reentrancy guard is built the way it is — a static table keyed by the
//! platform's own thread handle, with no thread-local storage on the allocation
//! path — because of one specific failure that only appears in a dynamically
//! loaded library. `examples/cdylib_probe.rs` documents the mechanism; this file
//! is what actually runs it.
//!
//! Until now that design decision was justified by a measurement taken on the
//! platform (`nm -m` showing the TLV routing through `__tlv_bootstrap`) and by
//! reasoning, with no test that put the pieces together. This is the test.
//!
//! Unix only. `dlopen` is the loading mechanism, and the `tlv_get_addr` hazard
//! is a dyld property; Windows has no equivalent path and no equivalent bug.

#![cfg(unix)]

use std::ffi::{c_char, c_int, c_void, CString};
use std::path::PathBuf;

// Just the one helper, rather than `mod support;`. This binary needs a path,
// not the profile validators, and pulling the whole module in to get one would
// compile four large files into every build of this test.
#[path = "support/fixture.rs"]
mod fixture;

extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

/// `RTLD_NOW | RTLD_LOCAL`. Resolving everything up front turns a missing
/// symbol into a clear failure here rather than a crash at the first call.
#[cfg(target_vendor = "apple")]
const FLAGS: c_int = 0x2 | 0x4;
// glibc's `RTLD_LOCAL` is 0, so this is `RTLD_NOW` alone. Spelled out rather
// than reduced, because the two constants are what the call means.
#[cfg(all(unix, not(target_vendor = "apple")))]
const FLAGS: c_int = 0x2;

/// The library's exit codes, mirrored from the fixture.
const OK: i32 = 0;
const COULD_NOT_START: i32 = 1;
const NOT_RECORDED: i32 = 2;
const GUARD_LEFT_HELD: i32 = 3;

fn describe(code: i32) -> &'static str {
    match code {
        OK => "ok",
        COULD_NOT_START => "the profiler inside the library did not start",
        NOT_RECORDED => {
            "the library's allocations were not recorded, so the guard was \
             refusing them — a live process with an empty profile"
        }
        GUARD_LEFT_HELD => {
            "the reentrancy guard was still held after the shim returned, which \
             means the first thread-local touch left a depth behind"
        }
        _ => "an unrecognised failure",
    }
}

// Miri cannot run this, and the reason is not incidental: the test exists to
// exercise the dynamic loader's own thread-local machinery, which means
// `dlopen`ing a real library built for the host and letting the host's loader
// run. Miri interprets the program instead of executing it, so there is no
// loader to exercise; it stops earlier still, at `_NSGetExecutablePath`, which
// its filesystem isolation does not provide. Disabling isolation would not
// help and would weaken the job for every other test in it.
#[test]
#[cfg_attr(
    miri,
    ignore = "dlopens a host library to exercise the platform's loader"
)]
fn a_dynamically_loaded_library_can_allocate_on_a_fresh_thread() {
    let path = library_path();
    let c_path = CString::new(path.to_str().expect("a UTF-8 path")).expect("no interior NUL");

    // SAFETY: a valid NUL-terminated path and valid flags. Loading a library
    // runs its initialisers, which for this fixture is Rust's own runtime setup.
    let handle = unsafe { dlopen(c_path.as_ptr(), FLAGS) };
    assert!(
        !handle.is_null(),
        "could not load {}: {}",
        path.display(),
        last_error()
    );

    let start: extern "C" fn() -> i32 = unsafe { symbol(handle, "heapscope_cdylib_start") };
    let allocate: extern "C" fn() -> i32 =
        unsafe { symbol(handle, "heapscope_cdylib_allocate_here") };

    assert_eq!(
        start(),
        OK,
        "the profiler inside the loaded library did not start"
    );

    // **This** thread must be created here, not inside the library. macOS gives
    // an image one TLV block per thread, and `std::thread`'s own startup touches
    // std's thread-locals — which are compiled into that library — so a thread
    // the library spawned has already been through the first-touch path before
    // any of its code runs. Creating it here means the first thing the image
    // does on this thread is allocate.
    let result = std::thread::spawn(move || allocate())
        .join()
        .expect("the thread calling into the library panicked");

    // If the guard consulted thread-local storage, the call above would not have
    // returned at all: it would recurse until the stack ran out, leaking a block
    // per level. A crash there *is* this test failing; the codes below only
    // distinguish the quieter ways it can go wrong.
    assert_eq!(result, OK, "{}", describe(result));
}

/// Resolves `name` in `handle`, transmuting it to a function pointer.
///
/// # Safety
///
/// `name` must be an `extern "C" fn() -> i32` in the loaded library.
unsafe fn symbol<T>(handle: *mut c_void, name: &str) -> T {
    let c_name = CString::new(name).expect("no interior NUL");
    // SAFETY: a valid handle from `dlopen` and a NUL-terminated symbol name.
    let address = unsafe { dlsym(handle, c_name.as_ptr()) };
    assert!(!address.is_null(), "no symbol {name}: {}", last_error());
    assert_eq!(
        std::mem::size_of::<T>(),
        std::mem::size_of::<*mut c_void>(),
        "the target type is not pointer-sized"
    );
    // SAFETY: delegated to the caller's obligation about the symbol's type.
    unsafe { std::mem::transmute_copy(&address) }
}

fn last_error() -> String {
    // SAFETY: `dlerror` returns a pointer to a static message or null.
    let message = unsafe { dlerror() };
    if message.is_null() {
        return String::from("(no error reported)");
    }
    // SAFETY: a NUL-terminated C string owned by the loader.
    unsafe { std::ffi::CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

/// Path to the compiled `cdylib_probe` example.
///
/// Searched for rather than computed; `support::fixture` says why.
fn library_path() -> PathBuf {
    fixture::example_library(
        "cdylib_probe",
        "plain `cargo test` builds examples, but `cargo test --all-targets` does\n\
         not -- it compiles this one as a test *executable*, so the shared library\n\
         below is never produced. See tests/support/fixture.rs. Either way:\n\
         \x20   cargo build --example cdylib_probe",
    )
}
