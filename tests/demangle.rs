//! What the demanglers are checked against.
//!
//! `heapscope` reimplements both Rust manglings instead of depending on
//! `rustc-demangle`, because the shipped library has no dependencies. A
//! reimplementation is a claim, and the claim is not "this parses the grammar"
//! but "this produces the same names the rest of the ecosystem produces". So
//! the reference implementation is a dev-dependency and gets run: every symbol
//! in `data/mangled-symbols.txt` goes through both, and they have to agree.
//!
//! The corpus was taken from real binaries with `nm` — this crate's own test
//! executables, built under `rustc` 1.96 for the legacy half and 1.97 for the
//! v0 half, plus a probe crate written to reach constructs that ordinary code
//! does not produce (higher-ranked bounds, `extern` fn pointers, const generics
//! of every scalar type, one-element tuples). It is checked in rather than
//! regenerated because a corpus that changes with the toolchain cannot tell you
//! whether a change in the output came from your edit or from the compiler.

// Every test in this file walks the whole corpus through both implementations,
// which costs 15 minutes under Miri and finds nothing: `src/symbol/demangle`
// contains no `unsafe`, a claim `tests/no_dependencies.rs` checks rather than
// assumes. Miri is here for undefined behaviour, and a job that takes 37
// minutes instead of 14 to prove the same thing is a job people start skipping.
#![cfg_attr(miri, allow(unused_imports))]

use std::collections::BTreeSet;

/// Symbols taken from real binaries, one per line.
const CORPUS: &str = include_str!("data/mangled-symbols.txt");

fn ours(symbol: &str) -> Option<String> {
    let mut out = String::new();
    heapscope::demangle(symbol, &mut out).then_some(out)
}

/// What the reference makes of `symbol`, or `None` if it refused.
///
/// Refusal is read from `try_demangle` rather than by comparing the rendering
/// to the input. Those are not the same test: `demangle` strips a ThinLTO
/// marker before parsing, so a *refused* symbol renders as a proper prefix of
/// itself and comparing would call that a success.
///
/// `{:#}` selects the form without crate disambiguators, which is the one meant
/// for people to read and the one this crate produces.
fn reference(symbol: &str) -> Option<String> {
    let demangled = rustc_demangle::try_demangle(symbol).ok()?;
    let rendered = format!("{demangled:#}");
    // An empty rendering is not a name. The reference produces one for a
    // legacy path whose only component is the disambiguating hash, having
    // stripped the only thing in it; this crate refuses those instead, so that
    // the caller can fall back to showing the raw symbol rather than a frame
    // that looks like it had no symbol at all.
    (!rendered.is_empty()).then_some(rendered)
}

/// Whether a sanitizer is watching, as declared by the harness running this.
///
/// The corpus walks below cost minutes and cannot produce a sanitizer finding:
/// `src/symbol/demangle` contains no `unsafe` at all, no raw memory and no
/// threads, so there is nothing for ASan or TSan to see. That is the same
/// argument the `cfg_attr(miri, ignore)` on each of them already makes, and it
/// is not an argument anyone has to keep true by hand —
/// `tests/no_dependencies.rs` fails if an `unsafe` block appears in that
/// directory.
///
/// Read from the environment because `cfg(sanitize = "..")` is unstable, and
/// set by `ci/sanitizers.sh`. The harness declaring its own constraints is the
/// arrangement `tests/symbolize.rs` uses for the same reason.
fn under_a_sanitizer() -> bool {
    std::env::var_os("HEAPSCOPE_SANITIZER").is_some()
}

/// Says so, loudly, and returns whether the caller should stop.
fn stood_down(what: &str) -> bool {
    if under_a_sanitizer() {
        eprintln!("SKIPPED under a sanitizer: {what} — no unsafe in the demangler to find");
        return true;
    }
    false
}

fn corpus() -> impl Iterator<Item = &'static str> {
    CORPUS
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
}

/// The headline property.
///
/// Where the reference produces a name, this crate must produce the same name.
/// Not a similar one: a profile is read by someone matching a frame against
/// source they are looking at, so a name that differs in a generic argument or
/// a closure index is a wrong answer wearing a plausible face.
#[test]
#[cfg_attr(miri, ignore = "walks the whole corpus; no unsafe here to check")]
fn every_symbol_the_reference_demangles_demangles_identically_here() {
    if stood_down("every_symbol_the_reference_demangles_demangles_identically_here") {
        return;
    }
    let mut checked = 0;
    let mut mismatches = Vec::new();
    for symbol in corpus() {
        let Some(expected) = reference(symbol) else {
            continue;
        };
        checked += 1;
        match ours(symbol) {
            Some(actual) if actual == expected => {}
            other => mismatches.push(format!(
                "  {symbol}\n    reference: {expected}\n    ours:      {}",
                other.as_deref().unwrap_or("<refused>")
            )),
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {checked} symbols disagree with rustc-demangle:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    // A corpus that silently emptied itself would make the assertion above
    // pass without checking anything.
    assert!(checked > 500, "corpus shrank to {checked} usable symbols");
}

/// Both manglings have to be represented, or the test above could be passing
/// entirely on one of them.
#[test]
#[cfg_attr(miri, ignore = "walks the whole corpus; no unsafe here to check")]
fn the_corpus_covers_both_manglings_and_every_construct_that_needs_covering() {
    if stood_down("the_corpus_covers_both_manglings_and_every_construct_that_needs_covering") {
        return;
    }
    let demangled: BTreeSet<String> = corpus().filter_map(ours).collect();
    let all = demangled.iter().cloned().collect::<Vec<_>>().join("\n");

    let legacy = corpus().filter(|s| s.starts_with("_ZN")).count();
    let v0 = corpus().filter(|s| s.starts_with("_R")).count();
    assert!(legacy > 100, "only {legacy} legacy symbols");
    assert!(v0 > 100, "only {v0} v0 symbols");

    for (construct, needle) in [
        ("trait impl", " as "),
        ("closure", "{closure#"),
        ("shim", "{shim:"),
        ("generic argument", "::<"),
        ("higher-ranked bound", "for<'a>"),
        ("function pointer", "fn("),
        ("dyn bound", "dyn "),
        ("associated type binding", "Item = "),
        ("array length", "; 12]"),
        ("slice", "[u16]"),
        ("one-element tuple", "(u8,)"),
        ("raw pointer", "*const "),
        ("mutable reference", "&mut "),
        ("const generic", "::<7>"),
        ("negative const generic", "-42"),
        ("boolean const generic", "::<true>"),
        ("character const generic", "::<'q'>"),
        ("non-ASCII identifier", "ünïcödé"),
        ("vendor suffix", "$tlv$init"),
        ("split-function suffix", ".cold.1"),
        ("string constant", "\"hello\""),
        ("reference constant", "{&mut *\"\"}"),
        ("structural constant", "{[\"\", 7]}"),
        // Review found this rendered as `dyn Trait<u64><Out = u8>` — a trait
        // object whose generic construct arrives through a backreference.
        ("backreferenced dyn generics", "Shape<u64, Out = u8>"),
    ] {
        assert!(
            all.contains(needle),
            "the corpus demangles nothing containing {needle:?} ({construct}), \
             so nothing tests that construct"
        );
    }
}

/// Where this crate deliberately parts company with the reference.
///
/// `rustc-demangle` refuses any symbol whose vendor suffix starts with `$`,
/// which on Mach-O means every thread-local initialiser: the user is shown raw
/// mangled text for a symbol that is perfectly readable. This crate demangles
/// it and keeps the suffix, because the suffix is what distinguishes the
/// initialiser from the thread-local it initialises.
///
/// The test exists so that the divergence stays deliberate. If a future version
/// of the reference starts accepting these, this fails and the two can be
/// reconciled on purpose.
#[test]
#[cfg_attr(miri, ignore = "walks the whole corpus; no unsafe here to check")]
fn the_only_divergence_from_the_reference_is_where_it_refuses_outright() {
    if stood_down("the_only_divergence_from_the_reference_is_where_it_refuses_outright") {
        return;
    }
    let mut improvements = 0;
    for symbol in corpus() {
        if reference(symbol).is_some() {
            continue;
        }
        let Some(actual) = ours(symbol) else {
            continue;
        };
        assert!(
            symbol.contains('$'),
            "demangled {symbol} to {actual} where the reference refused, \
             for a reason other than the known suffix divergence"
        );
        assert_ne!(actual, symbol, "a refusal must not masquerade as a name");
        improvements += 1;
    }
    assert!(
        improvements > 0,
        "the corpus no longer covers the suffix divergence"
    );
}

/// Nothing in the corpus should be anywhere near the parser's work budget, or
/// the budget is not the backstop it is documented to be.
#[test]
#[cfg_attr(miri, ignore = "walks the whole corpus; no unsafe here to check")]
fn real_symbols_are_far_inside_the_limits() {
    if stood_down("real_symbols_are_far_inside_the_limits") {
        return;
    }
    let longest = corpus().map(str::len).max().unwrap_or(0);
    let widest = corpus()
        .filter_map(ours)
        .map(|rendered| rendered.len())
        .max()
        .unwrap_or(0);
    // The budget is 1 << 20 units, spent roughly one per input byte and one
    // per output byte.
    assert!(longest < 4096, "longest input is {longest} bytes");
    assert!(widest < 8192, "widest output is {widest} bytes");
}
