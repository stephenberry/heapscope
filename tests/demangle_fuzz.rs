//! What the demanglers do when the input is not a symbol.
//!
//! PLAN.md section 6.2 pulls this forward from M7 because the threat is not
//! hypothetical. A symbolizer feeds these parsers whatever the symbol table
//! says, and the symbol table is routinely wrong: stripped images give `dladdr`
//! a name belonging to a neighbouring function, a truncated file gives half a
//! symbol, and a profile recorded on one machine can be resolved against a
//! build that is not the one it came from. The parsers are also reachable from
//! anywhere `heapscope::demangle` is called, which is a public function.
//!
//! So the contract is not "correct on valid input". It is that **no input of
//! any shape** may panic, hang, allocate without bound, or overflow the stack.
//! `fuzz/fuzz_targets/demangle.rs` runs a real campaign against the same
//! contract; this file is what holds it on every `cargo test`.
//!
//! Generation is deliberately not uniform random bytes. Random bytes bounce off
//! the four-character prefix check and never reach a parser, which is why the
//! cases below are built from the grammar's own alphabet and from mutations of
//! symbols that really occurred.

use proptest::prelude::*;
use std::time::{Duration, Instant};

const CORPUS: &str = include_str!("data/mangled-symbols.txt");

/// A single call must never take longer than this. Generous enough that a
/// loaded machine does not cause a false failure, tight enough that anything
/// unbounded fails it by orders of magnitude.
const PATIENCE: Duration = Duration::from_secs(2);

/// Whether a sanitizer is watching, as declared by `ci/sanitizers.sh`.
///
/// Same argument as the `cfg_attr(miri, ignore)` markers here: this file drives
/// `src/symbol/demangle`, which contains no `unsafe` at all, no raw memory and
/// no threads, so neither ASan nor TSan can find anything in it. The invariant
/// is enforced rather than trusted, by `tests/no_dependencies.rs`.
///
/// `cfg(sanitize = "..")` is unstable, so the harness says so instead.
fn under_a_sanitizer() -> bool {
    std::env::var_os("HEAPSCOPE_SANITIZER").is_some()
}

fn corpus() -> Vec<&'static str> {
    CORPUS
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect()
}

/// Demangles, and fails rather than returns if it took implausibly long.
fn demangle_promptly(symbol: &str) -> Option<String> {
    let mut out = String::new();
    let started = Instant::now();
    let demangled = heapscope::demangle(symbol, &mut out);
    let elapsed = started.elapsed();
    assert!(
        elapsed < PATIENCE,
        "took {elapsed:?} on {} bytes: {symbol:?}",
        symbol.len()
    );
    demangled.then_some(out)
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

/// Every property that has to hold for every input, in one place so that each
/// generator below can apply all of them.
fn check_invariants(symbol: &str) {
    let first = demangle_promptly(symbol);

    // A refusal must leave the caller's buffer exactly as it found it. Callers
    // demangle straight into the string they are already building, so a
    // half-written name on failure would corrupt the frame around it.
    let mut buffer = String::from("prefix");
    let demangled = heapscope::demangle(symbol, &mut buffer);
    assert_eq!(demangled, first.is_some());
    if demangled {
        assert_eq!(buffer, format!("prefix{}", first.as_ref().unwrap()));
    } else {
        assert_eq!(buffer, "prefix");
    }

    let Some(name) = first else {
        return;
    };

    // Output is bounded by input. The parsers can re-read earlier constructs,
    // so this is the property that says a short symbol cannot name an enormous
    // one. The constant is empirical headroom over the widest real expansion,
    // not a tight bound.
    assert!(
        name.len() <= 64 * symbol.len() + 1024,
        "{} bytes of input produced {} bytes of output",
        symbol.len(),
        name.len()
    );

    // A name is something a person reads. Control characters in one are either
    // corruption or an attempt to write escape sequences into whatever renders
    // the profile.
    assert!(
        !name.chars().any(char::is_control),
        "control character in {name:?}"
    );

    // Where the reference produces a name, it must be this name. Doing this on
    // generated input is what makes it a differential fuzz rather than a
    // smoke test: `tests/demangle.rs` covers symbols that occurred, and this
    // covers symbols that could.
    if let Some(expected) = reference(symbol) {
        assert_eq!(
            name, expected,
            "disagreed with rustc-demangle on {symbol:?}"
        );
    }
}

/// The alphabet a mangled symbol is drawn from, plus a few bytes that are not
/// in it, so that generation covers both the grammar and its edges.
fn symbol_bytes() -> impl Strategy<Value = char> {
    prop_oneof![
        // The tags, weighted up: these are what drive the parsers into their
        // interesting states.
        60 => prop::sample::select(
            "CNMXYIBKLGRQPOASTFDVUEsuvnpbcefhijlmotxyzad".chars().collect::<Vec<_>>()
        ),
        20 => prop::sample::select("0123456789".chars().collect::<Vec<_>>()),
        10 => prop::sample::select("_$.".chars().collect::<Vec<_>>()),
        5 => prop::sample::select("ABDEFGHJKLMNOPQRSTUVWXYZ".chars().collect::<Vec<_>>()),
        5 => any::<char>(),
    ]
}

proptest! {
    // The parsers are cheap, so a wide net costs little — except under Miri,
    // which interprets rather than executes. Four cases there keep the code
    // paths covered at a cost of seconds; `src/symbol/demangle` contains no
    // `unsafe` at all (asserted by `tests/no_dependencies.rs`), so the wide net
    // is not what Miri is for.
    //
    // Failure persistence is dropped under Miri because saving a seed means
    // resolving the current directory, which its isolation makes a hard abort —
    // one that takes every other test in this binary with it.
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) || under_a_sanitizer() { 4 } else { 4096 },
        failure_persistence: if cfg!(miri) {
            None
        } else {
            Some(Box::new(
                proptest::test_runner::FileFailurePersistence::default(),
            ))
        },
        ..ProptestConfig::default()
    })]

    /// Grammar-shaped noise: the right alphabet, no structure.
    #[test]
    fn input_built_from_the_grammars_alphabet_is_survivable(
        prefix in prop::sample::select(vec!["_R", "_ZN", "__R", "__ZN", "ZN", "R", ""]),
        body in prop::collection::vec(symbol_bytes(), 0..80),
    ) {
        let symbol: String = prefix.chars().chain(body).collect();
        check_invariants(&symbol);
    }

    /// Real symbols with bytes damaged, which is what a truncated or mismatched
    /// symbol table actually produces.
    #[test]
    fn a_corrupted_real_symbol_is_survivable(
        index in any::<prop::sample::Index>(),
        edits in prop::collection::vec(
            (any::<prop::sample::Index>(), symbol_bytes()),
            0..8,
        ),
    ) {
        let all = corpus();
        let mut symbol: Vec<char> = index.get(&all).chars().collect();
        for (at, replacement) in edits {
            if symbol.is_empty() {
                break;
            }
            let position = at.index(symbol.len());
            symbol[position] = replacement;
        }
        check_invariants(&symbol.into_iter().collect::<String>());
    }

    /// Real symbols cut short, which is what a truncated read produces. Every
    /// prefix of every symbol is a case the parsers will meet.
    #[test]
    fn a_truncated_real_symbol_is_survivable(
        index in any::<prop::sample::Index>(),
        cut in any::<prop::sample::Index>(),
    ) {
        let all = corpus();
        let symbol = index.get(&all);
        let mut end = cut.index(symbol.len() + 1);
        while !symbol.is_char_boundary(end) {
            end -= 1;
        }
        check_invariants(&symbol[..end]);
    }

    /// Arbitrary text, including text that is not a symbol at all. Most of this
    /// is rejected at the prefix, which is exactly why it is the smallest of
    /// the four generators rather than the only one.
    #[test]
    fn arbitrary_text_is_survivable(symbol in ".*") {
        check_invariants(&symbol);
    }
}

/// Deep nesting is the case that a panic-catching harness cannot save you from:
/// a stack overflow is not a panic, and it takes the process with it.
///
/// Built rather than generated because proptest will not reliably produce a
/// thousand consecutive `R`s, and the depth limit is the thing being checked.
#[test]
#[cfg_attr(
    miri,
    ignore = "builds 100,000-byte symbols; interpreting them takes minutes"
)]
fn nesting_far_past_any_limit_does_not_overflow_the_stack() {
    for depth in [100usize, 1_000, 10_000, 100_000] {
        for (prefix, filler) in [("_RNvCs0_1p3tag", "R"), ("_RINvCs0_1p3tag", "A")] {
            let symbol = format!("{prefix}{}u", filler.repeat(depth));
            let mut out = String::new();
            let _ = heapscope::demangle(&symbol, &mut out);
        }
        // Legacy nests through escapes rather than through types, so its deep
        // case looks nothing like v0's.
        let component = "$LT$".repeat(depth);
        let symbol = format!("_ZN{}{component}E", component.len());
        let mut out = String::new();
        let _ = heapscope::demangle(&symbol, &mut out);
    }
}

/// The same input must always produce the same output. Anything that reads
/// uninitialised memory, or depends on an allocation address, would show up
/// here as a name that changes between calls.
#[test]
#[cfg_attr(miri, ignore = "walks the whole corpus; no unsafe here to check")]
fn demangling_is_deterministic() {
    if under_a_sanitizer() {
        eprintln!(
            "SKIPPED under a sanitizer: the corpus walk — no unsafe in the demangler to find"
        );
        return;
    }
    for symbol in corpus() {
        let mut first = String::new();
        let mut second = String::new();
        assert_eq!(
            heapscope::demangle(symbol, &mut first),
            heapscope::demangle(symbol, &mut second)
        );
        assert_eq!(first, second, "{symbol}");
    }
}
