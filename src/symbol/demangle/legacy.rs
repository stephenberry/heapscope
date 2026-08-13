//! The legacy mangling, which is `rustc`'s default up to and including 1.96.
//!
//! A path is a run of length-prefixed components between `_ZN` and `E`:
//!
//! ```text
//! _ZN1p6nested6deeper6buried17hf6fa4f2218a09129E
//!    ^^ ^^^^^^ ^^^^^^ ^^^^^^ ^^^^^^^^^^^^^^^^^^
//!    p  nested deeper buried  the disambiguating hash
//! ```
//!
//! Components join with `::`, so that demangles to `p::nested::deeper::buried`.
//!
//! The mangling is lossy in a way that shapes everything below: the character
//! set of a linker symbol is much smaller than the character set of a Rust
//! path, so anything outside `[A-Za-z0-9_]` is escaped as `$XX$`, and `::`
//! inside a component is written `..`. Decoding is therefore not a matter of
//! slicing at separators — every component has to be walked byte by byte.
//!
//! ```text
//! _ZN47_$LT$p..Holder$LT$T$GT$$u20$as$u20$p..Shape$GT$4area17h..E
//!         <p::Holder<T> as p::Shape>::area
//! ```

/// Demangles the body of a legacy symbol: everything after `_ZN`.
///
/// Returns `false` without touching `out` if the body is not a well-formed
/// path. The scan runs to completion before a single byte is written, which is
/// what makes that promise cheap to keep.
pub(super) fn demangle(body: &str, out: &mut String) -> bool {
    let Some(path) = Path::scan(body) else {
        return false;
    };

    let mut rest = path.components;
    let mut remaining = path.count;
    let mut first = true;
    while remaining > 0 {
        // `scan` already proved every length prefix and every slice boundary,
        // so these cannot fail; `take_component` returning an Option keeps the
        // two walks reading identically rather than making this one unsafe.
        let Some((component, tail)) = take_component(rest) else {
            return true;
        };
        rest = tail;
        remaining -= 1;
        if !first {
            out.push_str("::");
        }
        first = false;
        write_component(component, out);
    }
    out.push_str(path.suffix);
    true
}

/// A validated legacy path: the components to print and how many there are.
struct Path<'a> {
    /// Positioned at the first length prefix.
    components: &'a str,
    /// Components to print, with any trailing hash already excluded.
    count: usize,
    /// Whatever followed the terminating `E`.
    suffix: &'a str,
}

impl<'a> Path<'a> {
    /// Validates `body` end to end without decoding anything.
    fn scan(body: &'a str) -> Option<Self> {
        let mut rest = body;
        let mut count = 0usize;
        let mut last = "";
        loop {
            if let Some(tail) = rest.strip_prefix('E') {
                rest = tail;
                break;
            }
            let (component, tail) = take_component(rest)?;
            last = component;
            rest = tail;
            count += 1;
        }

        // What follows the terminating `E` is a vendor suffix rather than more
        // path. Anything that is not one means the length prefixes did not
        // describe this symbol and the parse landed short of its end, and a
        // path that stops early is a wrong answer rather than a partial one.
        if !super::suffix_is_acceptable(rest) {
            return None;
        }

        let count = if is_disambiguator(last) {
            count - 1
        } else {
            count
        };
        // A path that was nothing but a hash names nothing. The reference
        // implementation renders that as the empty string; refusing is better,
        // because a caller shown nothing has no way to tell it apart from a
        // frame with no symbol, whereas a caller shown the raw symbol can at
        // least see what was there.
        (count > 0).then_some(Path {
            components: body,
            count,
            suffix: rest,
        })
    }
}

/// Splits one `<byte length><bytes>` component off the front.
fn take_component(rest: &str) -> Option<(&str, &str)> {
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    // A length that does not fit in a `usize` cannot describe a slice of a
    // string that exists, so overflow is a rejection rather than a saturation.
    let length: usize = rest[..digits].parse().ok()?;
    if length == 0 {
        return None;
    }
    let body = &rest[digits..];
    // `is_char_boundary` is false for any index past the end, so this covers
    // truncation and mid-character splits in one check. Well-formed components
    // are ASCII, but nothing guarantees the input is well formed.
    if !body.is_char_boundary(length) {
        return None;
    }
    Some((&body[..length], &body[length..]))
}

/// Whether a component is the trailing hash that makes a symbol unique.
///
/// The form is `h` followed by hexadecimal, currently 16 digits of it. Only the
/// shape is checked, not the width, because the width has changed before, and
/// not even a non-empty width, because `rustc`'s own demangler does not.
///
/// A path really ending in a component like `hedbeef` is mistaken for a hash
/// and dropped. That is inherent to a mangling that marks the hash by
/// convention instead of by syntax, and it is the same trade the reference
/// implementation makes; agreeing with it matters more than being marginally
/// less wrong in a case that does not occur.
fn is_disambiguator(component: &str) -> bool {
    let Some(hex) = component.strip_prefix('h') else {
        return false;
    };
    hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Decodes one component's escapes into `out`.
fn write_component(component: &str, out: &mut String) {
    // A component that would otherwise start with `$` gets an underscore in
    // front of it, because a symbol may not begin with punctuation. Nothing
    // else in the mangling produces a leading `_$`, so this is unambiguous.
    let mut rest = match component.strip_prefix("_$") {
        Some(_) => &component[1..],
        None => component,
    };

    while !rest.is_empty() {
        if let Some(tail) = rest.strip_prefix("..") {
            out.push_str("::");
            rest = tail;
            continue;
        }
        if let Some(tail) = rest.strip_prefix('$') {
            match write_escape(tail, out) {
                Some(remaining) => rest = remaining,
                None => return,
            }
            continue;
        }
        // Everything up to the next byte that could begin an escape or a
        // separator is literal.
        let end = rest.find(['$', '.']).unwrap_or(rest.len());
        if end == 0 {
            // A lone `.`, which the mangling passes through: `{{closure}}` and
            // `vtable.shim` both reach us with real dots in them.
            out.push('.');
            rest = &rest[1..];
            continue;
        }
        out.push_str(&rest[..end]);
        rest = &rest[end..];
    }
}

/// Decodes one `$XX$` escape, given everything after the opening `$`.
///
/// Returns the remainder to keep decoding, or `None` when decoding stopped.
///
/// An escape this does not recognise stops the component: the rest of it goes
/// out exactly as it was written, undecoded. That is not a fallback so much as
/// an admission. `$` is not a byte the mangler emits raw, so an unrecognised
/// escape means the assumption that this is a mangled component was wrong, and
/// continuing to decode past it would be applying a grammar to text that is not
/// in it. Showing the bytes says "I do not know what this is" in the only way
/// that cannot be mistaken for a name.
fn write_escape<'a>(after_dollar: &'a str, out: &mut String) -> Option<&'a str> {
    let stop_here = |out: &mut String| {
        out.push('$');
        out.push_str(after_dollar);
        None
    };
    // No closing `$` at all: there is no escape here to decode.
    let Some(end) = after_dollar.find('$') else {
        return stop_here(out);
    };
    let Some(character) = decode(&after_dollar[..end]) else {
        return stop_here(out);
    };
    out.push(character);
    Some(&after_dollar[end + 1..])
}

/// Maps an escape code to the character it stands for.
fn decode(code: &str) -> Option<char> {
    Some(match code {
        "SP" => '@',
        "BP" => '*',
        "RF" => '&',
        "LT" => '<',
        "GT" => '>',
        "LP" => '(',
        "RP" => ')',
        "C" => ',',
        _ => {
            let hex = code.strip_prefix('u')?;
            // Lower case only. The mangler emits one spelling per code point,
            // so `$uB0$` is not an alternative way of writing `$ub0$` — it is
            // an indication that this text did not come from the mangler, and
            // decoding it would be reading meaning into a coincidence.
            let lower_case_hex = |byte: u8| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte);
            if hex.is_empty() || !hex.bytes().all(lower_case_hex) {
                return None;
            }
            // Bounded by construction: a code longer than eight digits cannot
            // be a scalar value, and rejecting it here keeps `from_str_radix`
            // off inputs it would have to reject anyway.
            if hex.len() > 8 {
                return None;
            }
            let value = u32::from_str_radix(hex, 16).ok()?;
            let character = char::from_u32(value)?;
            // A control character in a symbol name is either corruption or an
            // attempt to write terminal escapes into someone's profile viewer.
            if character.is_control() {
                return None;
            }
            character
        }
    })
}

#[cfg(test)]
mod tests {
    /// Demangles a whole symbol, prefix included, the way callers see it.
    ///
    /// Going in through the public entry point rather than [`super::demangle`]
    /// keeps these tests honest about prefix handling, and means the mangled
    /// strings below can be pasted straight out of `nm`.
    fn rendered(symbol: &str) -> Option<String> {
        let mut out = String::new();
        crate::symbol::demangle(symbol, &mut out).then_some(out)
    }

    /// Assembles a symbol from the components it should contain.
    ///
    /// Symbols copied out of `nm` are written literally below, because their
    /// value is that they are exactly what a compiler produced. Symbols
    /// constructed to reach one branch are built with this instead: a
    /// hand-counted length prefix is a second thing that can be wrong, and when
    /// it is, the test fails for a reason that has nothing to do with the
    /// branch it was written for.
    fn built(components: &[&str]) -> String {
        let mut out = String::from("_ZN");
        for component in components {
            out.push_str(&component.len().to_string());
            out.push_str(component);
        }
        out.push('E');
        out
    }

    #[test]
    fn a_plain_path_joins_its_components_with_colons() {
        assert_eq!(
            rendered("_ZN1p6nested6deeper6buried17hf6fa4f2218a09129E").as_deref(),
            Some("p::nested::deeper::buried")
        );
    }

    #[test]
    fn the_disambiguating_hash_is_dropped() {
        assert_eq!(
            rendered("_ZN1p4main17h95b65b0d7d5234a0E").as_deref(),
            Some("p::main")
        );
    }

    /// A path whose only component is the hash is left with nothing to name.
    /// The reference implementation renders it as the empty string; this
    /// refuses, so the caller falls back to showing the raw symbol.
    #[test]
    fn a_path_that_is_nothing_but_a_hash_is_refused_rather_than_rendered_empty() {
        assert_eq!(rendered(&built(&["hdeadbee"])), None);
        assert_eq!(rendered(&built(&["h"])), None);
    }

    #[test]
    fn a_trailing_component_that_is_not_hexadecimal_is_kept() {
        assert_eq!(
            rendered(&built(&["p", "hunter"])).as_deref(),
            Some("p::hunter"),
            "`hunter` starts with h but is not a hash"
        );
    }

    /// The shape that made escaping necessary in the first place. Taken from a
    /// real build, hash and all.
    #[test]
    fn a_trait_implementation_is_reassembled_from_its_escapes() {
        assert_eq!(
            rendered(
                "_ZN47_$LT$p..Holder$LT$T$GT$$u20$as$u20$p..Shape$GT$4area17hb8b1df8dd112135aE"
            )
            .as_deref(),
            Some("<p::Holder<T> as p::Shape>::area")
        );
    }

    #[test]
    fn an_implementation_for_the_unit_type_round_trips_its_parentheses() {
        assert_eq!(
            rendered(
                "_ZN54_$LT$$LP$$RP$$u20$as$u20$std..process..Termination$GT$6report17h2ce70c1b0b032c89E"
            )
            .as_deref(),
            Some("<() as std::process::Termination>::report")
        );
    }

    #[test]
    fn a_closure_keeps_its_braces() {
        assert_eq!(
            rendered("_ZN1p15makes_a_closure28_$u7b$$u7b$closure$u7d$$u7d$17h1b1ea5f1247a4729E")
                .as_deref(),
            Some("p::makes_a_closure::{{closure}}")
        );
    }

    /// `vtable.shim` is the case that proves a lone dot is not a separator.
    #[test]
    fn a_single_dot_inside_a_component_stays_a_dot() {
        assert_eq!(
            rendered(
                "_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hf497d53e048b4267E"
            )
            .as_deref(),
            Some("core::ops::function::FnOnce::call_once{{vtable.shim}}")
        );
    }

    #[test]
    fn a_non_ascii_identifier_comes_back_out_of_its_hex_escapes() {
        assert_eq!(
            rendered("_ZN1p29_$ufc$n$uef$c$uf6$d$ue9$_name17hd0a10412634eb2dcE").as_deref(),
            Some("p::ünïcödé_name")
        );
    }

    #[test]
    fn every_punctuation_escape_is_understood() {
        for (code, expected) in [
            ("SP", '@'),
            ("BP", '*'),
            ("RF", '&'),
            ("LT", '<'),
            ("GT", '>'),
            ("LP", '('),
            ("RP", ')'),
            ("C", ','),
        ] {
            let symbol = built(&[&format!("_${code}$")]);
            assert_eq!(
                rendered(&symbol).as_deref(),
                Some(expected.to_string().as_str()),
                "${code}$"
            );
        }
    }

    /// LLVM clones and splits functions, and names the results by suffixing the
    /// symbol. The suffix is not part of the path, but its presence must not
    /// cost us the path, and which suffixes survive is a decision rather than
    /// an accident: see [`super::super::append_suffix`].
    #[test]
    fn a_vendor_suffix_after_the_terminator_is_kept_unless_it_is_a_clone_marker() {
        assert_eq!(
            rendered("_ZN1p4main17h95b65b0d7d5234a0E.llvm.4708").as_deref(),
            Some("p::main"),
            "the hash names a build, not a function"
        );
        assert_eq!(
            rendered("_ZN1p4main17h95b65b0d7d5234a0E.cold.1").as_deref(),
            Some("p::main.cold.1")
        );
        assert_eq!(
            rendered("_ZN1p4main17h95b65b0d7d5234a0E$tlv$init").as_deref(),
            Some("p::main$tlv$init")
        );
        // Only the hash itself is a clone marker. A suffix that starts the same
        // way but carries more is carrying something worth keeping.
        assert_eq!(
            rendered("_ZN1p4main17h95b65b0d7d5234a0E.llvm.4708.cold.1").as_deref(),
            Some("p::main.llvm.4708.cold.1")
        );
        assert_eq!(
            rendered("_ZN1p4main17h95b65b0d7d5234a0E.llvm.470u").as_deref(),
            Some("p::main.llvm.470u")
        );
    }

    /// The distinction that makes the suffix rule safe: a dot is a suffix, and
    /// anything else means the length prefixes did not describe this symbol and
    /// the parse landed in the wrong place.
    #[test]
    fn trailing_bytes_that_are_not_a_suffix_are_a_refusal() {
        assert_eq!(rendered("_ZN1p4main17h95b65b0d7d5234a0Etrailing"), None);
    }

    #[test]
    fn a_length_that_runs_past_the_end_is_refused() {
        assert_eq!(rendered("_ZN99pE"), None);
        assert_eq!(rendered("_ZN1"), None, "no body at all");
    }

    #[test]
    fn a_path_with_no_terminator_is_refused() {
        assert_eq!(rendered("_ZN1p4main"), None);
    }

    #[test]
    fn a_zero_length_component_is_refused() {
        assert_eq!(rendered("_ZN0E"), None);
        assert_eq!(rendered("_ZN1p0E"), None);
    }

    #[test]
    fn an_empty_path_is_refused() {
        assert_eq!(rendered("_ZNE"), None);
    }

    /// A length prefix wider than a `usize` must not wrap into a small one.
    #[test]
    fn an_absurd_length_prefix_is_refused_rather_than_wrapped() {
        assert_eq!(rendered("_ZN99999999999999999999999999pE"), None);
    }

    #[test]
    fn a_length_that_would_split_a_character_is_refused() {
        // `ü` is two bytes, so a length of 1 lands inside it.
        assert_eq!(rendered("_ZN1üE"), None);
    }

    #[test]
    fn an_unterminated_escape_is_shown_rather_than_swallowed() {
        assert_eq!(rendered(&built(&["a$bc"])).as_deref(), Some("a$bc"));
    }

    /// An escape we do not know means we are no longer reading what we thought
    /// we were, so the rest of the component goes out as written rather than
    /// being decoded on an assumption that has already proved wrong. The `..`
    /// in the tail staying a `..` is what distinguishes this from carrying on.
    #[test]
    fn an_unrecognised_escape_stops_the_component_rather_than_guessing_past_it() {
        assert_eq!(
            rendered(&built(&["a$ZZ$b..c$LT$d"])).as_deref(),
            Some("a$ZZ$b..c$LT$d")
        );
        // Everything before it was still decoded.
        assert_eq!(
            rendered(&built(&["a..b$LT$c$ZZ$d..e$GT$"])).as_deref(),
            Some("a::b<c$ZZ$d..e$GT$")
        );
    }

    /// A hex escape naming a control character is the one case where showing
    /// the input verbatim is better than decoding it: the decoded form can move
    /// the cursor in whatever terminal prints the report.
    #[test]
    fn a_control_character_escape_is_not_decoded() {
        assert_eq!(rendered(&built(&["$u1b$"])).as_deref(), Some("$u1b$"));
        assert_eq!(rendered(&built(&["$u0$"])).as_deref(), Some("$u0$"));
    }

    #[test]
    fn a_hex_escape_that_is_not_a_character_is_not_decoded() {
        // Surrogates and out-of-range values are both refused by `from_u32`.
        assert_eq!(rendered(&built(&["$ud800$"])).as_deref(), Some("$ud800$"));
        assert_eq!(
            rendered(&built(&["$uffffffff$"])).as_deref(),
            Some("$uffffffff$")
        );
        assert_eq!(
            rendered(&built(&["$u1000000000$"])).as_deref(),
            Some("$u1000000000$"),
            "too many digits to be a scalar value"
        );
    }

    /// Every branch in the component walk has to consume at least one byte, or
    /// a crafted symbol becomes an infinite loop inside a profiler's shutdown.
    #[test]
    fn adjacent_dollars_terminate() {
        assert_eq!(rendered(&built(&["$$"])).as_deref(), Some("$$"));
        assert_eq!(rendered(&built(&["$$$$"])).as_deref(), Some("$$$$"));
        assert_eq!(rendered(&built(&["$"])).as_deref(), Some("$"));
    }

    #[test]
    fn runs_of_dots_terminate_and_pair_up_left_to_right() {
        assert_eq!(rendered(&built(&[".."])).as_deref(), Some("::"));
        assert_eq!(rendered(&built(&["..."])).as_deref(), Some("::."));
        assert_eq!(rendered(&built(&["...."])).as_deref(), Some("::::"));
    }

    #[test]
    fn a_leading_underscore_is_dropped_only_before_a_dollar() {
        assert_eq!(rendered(&built(&["_$LT$"])).as_deref(), Some("<"));
        assert_eq!(
            rendered(&built(&["p", "_name"])).as_deref(),
            Some("p::_name"),
            "an identifier may legitimately start with an underscore"
        );
    }
}
