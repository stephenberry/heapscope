//! Turning a linker symbol back into the Rust path a person wrote.
//!
//! Two manglings are in circulation and a profiler meets both, often in the
//! same binary. The compiler's default changed under this crate's own feet:
//!
//! ```text
//! rustc 1.96 (this crate's MSRV)   _ZN1p14probe_function17he140cc384555f8bfE
//! rustc 1.97                       _RNvCs785SGTk9yHm_1p14probe_function
//! ```
//!
//! Both measured, same source, no flags. So neither mangling is the legacy
//! case: a profile recorded today can contain a `std` compiled with one and a
//! dependency compiled with the other, and a profile recorded against an MSRV
//! toolchain is entirely the older form. Both demanglers ship, and the entry
//! point below picks between them by prefix rather than by configuration.
//!
//! # Refusing beats guessing
//!
//! [`demangle`] returns `false` and leaves `out` untouched when the input is
//! not a Rust symbol, or is one it cannot parse. The caller then prints the raw
//! symbol, which is ugly but true. The alternative — emitting a partial parse,
//! or a name assembled from whatever was decoded before the input went bad — is
//! a profiler telling you an allocation came from somewhere it did not.
//!
//! # Hostile input is the normal case
//!
//! These parsers do not run on symbols a compiler produced. They run on bytes
//! read out of a stripped, truncated, or mismatched symbol table, and on
//! `dladdr` results that point into the middle of an unrelated image. Both
//! implementations are therefore written to a stricter contract than "correct
//! on valid input": no input of any shape may panic, allocate without bound,
//! recurse without bound, or fail to terminate. `tests/demangle_fuzz.rs` and
//! the fuzz target under `fuzz/` exist to hold that line.

mod legacy;
mod punycode;
mod v0;

/// Appends the demangled form of `symbol` to `out`.
///
/// Returns `false` if `symbol` is not a Rust mangled name, or is malformed, in
/// which case `out` is left exactly as it was found. `out` may already hold
/// text; this appends.
///
/// ```
/// let mut out = String::new();
/// assert!(heapscope::demangle("_ZN4core3fmt5write17hb1f9a4a7f2f1a0c9E", &mut out));
/// assert_eq!(out, "core::fmt::write");
/// ```
pub fn demangle(symbol: &str, out: &mut String) -> bool {
    // Neither mangling can encode a non-ASCII byte. v0 punycodes anything
    // outside `[A-Za-z0-9_]` and legacy writes it as `$u..$`, precisely because
    // a linker symbol has no way to carry one. So a non-ASCII byte here is not
    // an identifier this failed to understand; it is evidence that these bytes
    // are not a mangled symbol.
    //
    // Checked before dispatch because the consequence is not merely a wrong
    // name. v0 copies identifier bytes into the output verbatim, so without
    // this a corrupt symbol table could put a right-to-left override into a
    // profile and reverse the display of everything after it.
    if !symbol.is_ascii() {
        return false;
    }

    // LLVM's ThinLTO marker is removed before parsing rather than after,
    // because it is appended to the whole symbol wherever the symbol happens to
    // end — `_ZN..E.cold.1.llvm.9D1C9369` carries both, and only the second is
    // noise.
    let symbol = strip_thin_lto_marker(symbol);

    let restore_to = out.len();
    let demangled = match strip_prefix(symbol) {
        Some((Mangling::Legacy, body)) => legacy::demangle(body, out),
        Some((Mangling::V0, body)) => v0::demangle(body, out),
        None => false,
    };
    if !demangled {
        return false;
    }

    // A name this returns is going to be written into a JSON file that a
    // browser renders and a text report that a terminal renders.
    //
    // This is not made redundant by the ASCII check above: an escape character
    // is ASCII. v0 copies identifier bytes into the name verbatim, so a
    // corrupt symbol table can still put one there, and that is precisely the
    // case a fuzzer found. Neither mangling can encode a control character in
    // an identifier a compiler produced, so one arriving here means the input
    // was damaged, and passing it on would turn damage in a file into escape
    // sequences in somebody's terminal.
    //
    // Checked once over the finished name rather than at each of the several
    // places a byte can enter it, because the promise being kept is about the
    // result: what this function appends is always safe to display.
    if out[restore_to..].chars().any(char::is_control) {
        out.truncate(restore_to);
        return false;
    }
    true
}

/// Removes the marker LLVM appends to a function it cloned for ThinLTO.
///
/// The marker is `.llvm.` and an upper-case hexadecimal hash, which may carry
/// `@` where the linker joined two of them. It identifies a build rather than a
/// function, and the clone is the same source item as the original, so two
/// symbols differing only in it must render as one name or the profile splits
/// one function across two entries.
///
/// Anything else that merely begins `.llvm.` is left alone: the marker is not
/// reserved, and `.llvm.470a` is somebody else's suffix.
fn strip_thin_lto_marker(symbol: &str) -> &str {
    const MARKER: &str = ".llvm.";
    let Some(at) = symbol.find(MARKER) else {
        return symbol;
    };
    let hash = &symbol[at + MARKER.len()..];
    let hash_byte =
        |byte: u8| byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte) || byte == b'@';
    if hash.bytes().all(hash_byte) {
        &symbol[..at]
    } else {
        symbol
    }
}

/// Whether the bytes after a mangled path are a vendor suffix worth showing.
///
/// A suffix is not part of the path, but it is not noise either: `.cold.1`
/// names the half of a split function that runs rarely, and `$tlv$init` names a
/// thread-local's initialiser rather than the thread-local. Rendering two
/// different pieces of code under one name is the failure this exists to
/// prevent, which is why an unacceptable suffix fails the whole demangling
/// rather than being quietly dropped — dropping it produces exactly that
/// collision.
///
/// Acceptable means graphic ASCII introduced by `.` or `$`. Anything else means
/// the parse did not end where the symbol ends.
fn suffix_is_acceptable(suffix: &str) -> bool {
    if suffix.is_empty() {
        return true;
    }
    if !(suffix.starts_with('.') || suffix.starts_with('$')) {
        return false;
    }
    // Graphic excludes the space, which is not a byte a toolchain puts in a
    // symbol and is a byte that makes two names look like one.
    suffix.bytes().all(|byte| byte.is_ascii_graphic())
}

/// Which mangling a symbol announces itself as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mangling {
    /// `_ZN`-prefixed, length-prefixed path components, `$XX$` escapes.
    Legacy,
    /// `_R`-prefixed, a grammar rather than a flattening.
    V0,
}

/// Splits the mangling marker off the front, returning the body after it.
///
/// The number of leading underscores is a property of the object format, not of
/// the mangling: Mach-O and 32-bit Windows prepend one to every symbol, ELF and
/// 64-bit Windows do not. A symbol also reaches us with no underscore at all
/// when it came from a tool that already stripped it. All three forms are
/// accepted rather than making the caller know which platform it is on.
fn strip_prefix(symbol: &str) -> Option<(Mangling, &str)> {
    // No two of these can match the same string, because each is anchored at
    // index 0 and they disagree by their second byte, so the order is
    // presentational.
    for (marker, mangling) in [
        ("__ZN", Mangling::Legacy),
        ("_ZN", Mangling::Legacy),
        ("ZN", Mangling::Legacy),
        ("__R", Mangling::V0),
        ("_R", Mangling::V0),
        ("R", Mangling::V0),
    ] {
        if let Some(body) = symbol.strip_prefix(marker) {
            return Some((mangling, body));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(symbol: &str) -> Option<String> {
        let mut out = String::new();
        demangle(symbol, &mut out).then_some(out)
    }

    #[test]
    fn a_symbol_that_is_not_mangled_is_refused_rather_than_mangled_further() {
        assert_eq!(rendered("malloc"), None);
        assert_eq!(rendered(""), None);
        assert_eq!(rendered("_Z3foov"), None, "a C++ symbol is not ours");
    }

    /// The object format decides how many underscores a symbol carries, so a
    /// demangler that only understood one of them would work on Linux and go
    /// blank on macOS.
    #[test]
    fn every_platforms_underscore_convention_is_understood() {
        for symbol in [
            "_ZN1p4main17h95b65b0d7d5234a0E",
            "__ZN1p4main17h95b65b0d7d5234a0E",
            "ZN1p4main17h95b65b0d7d5234a0E",
        ] {
            assert_eq!(rendered(symbol).as_deref(), Some("p::main"), "{symbol}");
        }
    }

    /// The contract the callers rely on: a refusal costs them nothing, so they
    /// can try to demangle straight into the buffer they were already building.
    #[test]
    fn a_refusal_leaves_the_output_buffer_untouched() {
        let mut out = String::from("0x1234: ");
        assert!(!demangle("not_a_rust_symbol", &mut out));
        assert_eq!(out, "0x1234: ");
    }

    /// Found by `tests/demangle_fuzz.rs`, which corrupted one byte of a real
    /// symbol. v0 copies identifier bytes into the name verbatim, so before
    /// this check a single stray byte in a symbol table became a control
    /// character in whatever rendered the profile.
    #[test]
    fn a_name_carrying_a_control_character_is_refused_rather_than_displayed() {
        // A v0 identifier with an escape character in the middle of it.
        assert_eq!(rendered("_RNvCs0_1p4na\u{1b}e"), None);
        // The same for legacy, where the byte sits in a literal run.
        assert_eq!(rendered("_ZN1p4na\u{1b}eE"), None);
        // A NUL, which is what the fuzzer actually found.
        assert_eq!(rendered("_RNvCs0_1p4na\0e"), None);
    }

    /// Neither mangling can carry a non-ASCII byte, so one arriving means these
    /// bytes are not a symbol. Found by review: v0 copies identifier bytes into
    /// the name verbatim, so without this a corrupt symbol table could put a
    /// right-to-left override into a profile and reverse the display of
    /// everything after it, in both the terminal report and the viewer.
    #[test]
    fn a_non_ascii_byte_in_the_input_is_refused() {
        assert_eq!(rendered("_RNvCs0_1p7na\u{202e}me"), None, "bidi override");
        assert_eq!(rendered("_RNvCs0_1p6na\u{ad}me"), None, "soft hyphen");
        assert_eq!(rendered("_ZN2\u{fc}E"), None);
        // An escape that *encodes* one is a different matter: that is a
        // compiler-produced spelling of a legitimate identifier, and the
        // reference implementation renders it too.
        assert!(rendered("_ZN9a$u202e$bE").is_some());
    }

    /// LLVM clones a function for ThinLTO and marks the clone. The mark names a
    /// build rather than a function, so two symbols differing only in it are
    /// one function and must render as one name.
    ///
    /// Review found this applied only when the marker began the suffix, and
    /// only for `[0-9A-F]`. It is appended to whatever the symbol already
    /// ended with, and the linker joins two hashes with `@`.
    #[test]
    fn the_thin_lto_marker_is_removed_wherever_it_sits() {
        for symbol in [
            "_ZN1p4main17h95b65b0d7d5234a0E.llvm.9D1C9369",
            "_ZN1p4main17h95b65b0d7d5234a0E.llvm.9D1C9369@@16",
            "_ZN1p4main17h95b65b0d7d5234a0E.llvm.",
        ] {
            assert_eq!(rendered(symbol).as_deref(), Some("p::main"), "{symbol}");
        }
        // Only the marker goes; whatever it was appended to stays.
        assert_eq!(
            rendered("_ZN1p4main17h95b65b0d7d5234a0E.cold.1.llvm.9D1C9369").as_deref(),
            Some("p::main.cold.1")
        );
        // A suffix that merely begins the same way is somebody else's.
        assert_eq!(
            rendered("_ZN1p4main17h95b65b0d7d5234a0E.llvm.470a").as_deref(),
            Some("p::main.llvm.470a")
        );
    }

    /// A suffix that is not a suffix must fail the whole demangling. Dropping
    /// it silently is the one outcome that cannot be allowed: it renders two
    /// different pieces of code under one name, which is the collision the
    /// suffix policy exists to prevent.
    #[test]
    fn an_unshowable_suffix_is_a_refusal_rather_than_a_silent_omission() {
        let plain = "_ZN1p4main17h95b65b0d7d5234a0E";
        assert_eq!(rendered(plain).as_deref(), Some("p::main"));
        for suffix in [".a b", ".cold\u{1}", ".llvm moocow"] {
            let symbol = format!("{plain}{suffix}");
            assert_eq!(rendered(&symbol), None, "{symbol:?}");
        }
        assert_eq!(rendered("_RNvCs0_1p4name.a b"), None);
    }

    #[test]
    fn a_success_appends_rather_than_replaces() {
        let mut out = String::from("0x1234: ");
        assert!(demangle("_ZN1p4main17h95b65b0d7d5234a0E", &mut out));
        assert_eq!(out, "0x1234: p::main");
    }
}
