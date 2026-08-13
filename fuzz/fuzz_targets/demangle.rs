//! A coverage-guided campaign against the demanglers.
//!
//! ```sh
//! cargo +nightly fuzz run demangle -- -max_len=4096
//! ```
//!
//! `tests/demangle_fuzz.rs` checks the same contract on every `cargo test`, but
//! with generators that cannot see inside the parsers. This one is driven by
//! coverage, so it finds the inputs that reach a branch nothing else reaches:
//! a backreference chain that lands exactly on a construct boundary, a
//! punycode delta that overflows on the last digit, a length prefix that stops
//! one byte inside a multi-byte character.
//!
//! Seed it from the checked-in corpus, which starts it at symbols that are
//! already deep in the grammar:
//!
//! ```sh
//! mkdir -p fuzz/corpus/demangle
//! split -l 1 ../tests/data/mangled-symbols.txt fuzz/corpus/demangle/seed-
//! ```
//!
//! The contract is in the assertions below. Panicking is a finding, but so is
//! hanging or exhausting memory, which libFuzzer reports on its own.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Symbols are text. Anything that is not is a case the caller could not
    // have produced, and testing it would only exercise `from_utf8`.
    let Ok(symbol) = std::str::from_utf8(data) else {
        return;
    };

    let mut out = String::from("prefix");
    if !heapscope::demangle(symbol, &mut out) {
        // A refusal must cost the caller nothing, because callers demangle
        // straight into the buffer they are already building.
        assert_eq!(out, "prefix");
        return;
    }
    let name = &out["prefix".len()..];

    // Output is bounded by input. The v0 grammar can re-read constructs it has
    // already read, so without a work budget a short symbol could name an
    // enormous one.
    assert!(
        name.len() <= 64 * symbol.len() + 1024,
        "{} bytes in, {} bytes out",
        symbol.len(),
        name.len()
    );

    // What comes back is going into a terminal and a browser.
    assert!(
        !name.chars().any(char::is_control),
        "control character in {name:?}"
    );

    // Where the reference produces a name, it must be this name. This is the
    // assertion that turns the campaign from a crash hunt into a correctness
    // one: a demangler that never panicked and always returned the wrong name
    // would pass everything above.
    // Refusal is read from `try_demangle`, not by comparing the rendering to
    // the input: `demangle` strips a ThinLTO marker before parsing, so a
    // refused symbol renders as a proper prefix of itself and the comparison
    // would report a crash that is really an oracle bug.
    if let Ok(demangled) = rustc_demangle::try_demangle(symbol) {
        let reference = format!("{demangled:#}");
        if !reference.is_empty() {
            assert_eq!(name, reference, "disagreed with rustc-demangle");
        }
    }
});
