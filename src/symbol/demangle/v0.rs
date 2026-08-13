//! The v0 mangling, which is `rustc`'s default from 1.97 on.
//!
//! Where the legacy scheme flattens a path to text and escapes what will not
//! fit, v0 encodes the compiler's own structure: types, generic arguments,
//! constants, lifetimes and trait impls all survive it, which is why it can
//! reproduce a name the legacy scheme could only approximate.
//!
//! ```text
//! _RINvCs785SGTk9yHm_1p10generic_fnINtB2_6HolderyEEB2_
//!    I                                                  generic instantiation
//!     Nv                                                a value in a path
//!       Cs785SGTk9yHm_1p                                crate `p`
//!                       10generic_fn                    ...::generic_fn
//!                                   INtB2_6HolderyE     <p::Holder<u64>>
//!                                                  E    end of the args
//!                                                   B2_ instantiating crate
//!
//!   p::generic_fn::<p::Holder<u64>>
//! ```
//!
//! The grammar is RFC 2603's. This is a single pass that parses and prints at
//! the same time, because there is nothing a second pass would learn: every
//! construct is either printed where it is met or skipped entirely.
//!
//! # Why this file is written defensively
//!
//! Two features of the grammar make an ordinary recursive-descent parser a
//! liability when the input is not trustworthy.
//!
//! **Backreferences.** `B<n>_` means "re-read the construct at byte `n`", which
//! is how the encoding avoids repeating a type it has already written. Two
//! backreferences pointing at each other are a cycle, and one pointing at
//! itself is immediate recursion. Requiring every target to lie strictly before
//! the `B` that names it makes progress monotonic, so neither can be written.
//! [`MAX_DEPTH`] would stop both anyway; what the rule adds is that it stops
//! them at the first byte rather than after 256 levels of parsing.
//!
//! **Nesting.** Types contain types with no depth limit in the grammar, so a
//! symbol a few hundred bytes long can describe a structure deep enough to
//! exhaust the stack. [`MAX_DEPTH`] bounds it.
//!
//! Those two together still permit an input that terminates but should not be
//! waited for: backreferences pointing backwards can *share* subtrees, so `n`
//! bytes can describe a tree with `2^n` nodes in it. Neither a depth limit nor
//! the backwards rule catches that, because the structure really is finite and
//! really is shallow. [`BUDGET`] does: it counts work rather than shape, and a
//! symbol that exceeds it is refused.

use super::punycode;

/// Demangles the body of a v0 symbol: everything after `_R`.
///
/// Returns `false` with `out` restored to its previous contents if the body is
/// not a well-formed v0 symbol.
pub(super) fn demangle(body: &str, out: &mut String) -> bool {
    let restore_to = out.len();
    let mut printer = Printer {
        input: body,
        position: 0,
        depth: 0,
        budget: BUDGET,
        bound_lifetimes: 0,
        printing: true,
        out,
    };
    if printer.print_symbol().is_err() {
        out.truncate(restore_to);
        return false;
    }
    true
}

/// How deep a nesting of types, paths and constants may go.
///
/// Deep enough that no symbol a compiler emits comes near it, shallow enough
/// that the stack survives the worst case. Each level is one frame of
/// [`Printer::print_type`] or its neighbours, all of which hold only a handful
/// of locals.
const MAX_DEPTH: u32 = 256;

/// How much work a single symbol may cost before it is refused.
///
/// Spent one unit per byte consumed and per character printed, so it bounds
/// running time and output size together. The largest symbol in a real build
/// measured 921 bytes and rendered to a few kilobytes; this leaves three orders
/// of magnitude of headroom, while still turning the `2^n` blow-up described
/// above into a bounded amount of wasted work.
const BUDGET: u32 = 1 << 20;

/// The parse failed. There is deliberately no detail: nothing downstream can
/// act on the difference between "not a v0 symbol" and "a v0 symbol that ran
/// out of budget", and both lead to the same place, which is showing the caller
/// the raw symbol instead.
struct Invalid;

type Parsed<T = ()> = Result<T, Invalid>;

struct Printer<'a, 'b> {
    input: &'a str,
    position: usize,
    depth: u32,
    budget: u32,
    /// How many lifetimes the enclosing `for<...>` binders have introduced.
    /// Lifetimes are encoded as de Bruijn indices, so a name can only be given
    /// to one by counting outwards from where it is used.
    bound_lifetimes: u64,
    /// False while walking a construct whose text is not wanted. Parsing still
    /// has to happen, because skipping it requires knowing where it ends.
    printing: bool,
    out: &'b mut String,
}

impl<'a> Printer<'a, '_> {
    // ---- the grammar ----------------------------------------------------

    /// `<symbol-name> = <path> [<instantiating-crate>] [<suffix>]`
    fn print_symbol(&mut self) -> Parsed {
        // A leading digit is an encoding version. Version 0 is written by
        // leaving it out, so a digit here names a version that did not exist
        // when this was written, and guessing at it would be worse than saying
        // nothing.
        if self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            return Err(Invalid);
        }

        // A symbol names a value, which is what makes `foo::<T>` print with
        // turbofish where the same construct inside a type does not.
        self.print_path(true)?;

        // What remains is the crate that instantiated this item, which is not
        // part of its name, followed by anything the toolchain appended. Both
        // are optional: only a generic item records where it was instantiated,
        // so most symbols end here.
        if !self.rest().is_empty() && !self.at_suffix() {
            self.skipping(|this| this.print_path(false))?;
        }
        if self.rest().is_empty() {
            return Ok(());
        }
        if self.at_suffix() {
            // Reborrowed off `input` rather than `self` so the printer can be
            // handed its own output buffer.
            let suffix = &self.input[self.position..];
            if !super::suffix_is_acceptable(suffix) {
                return Err(Invalid);
            }
            self.spend(suffix.len().min(u32::MAX as usize) as u32)?;
            if self.printing {
                self.out.push_str(suffix);
            }
            return Ok(());
        }
        // Trailing bytes that are neither mean the parse ended somewhere other
        // than the end of the symbol, so the name it produced is not this
        // symbol's name.
        Err(Invalid)
    }

    /// Whether the cursor is at a vendor suffix rather than more grammar.
    ///
    /// LLVM appends `.cold.1` and friends; the mangling reserves `$` for the
    /// same purpose on targets where `.` is not available. The ThinLTO marker
    /// is already gone by this point, removed from the whole symbol before
    /// parsing began.
    fn at_suffix(&self) -> bool {
        matches!(self.peek(), Some(b'.') | Some(b'$'))
    }

    fn print_path(&mut self, in_value: bool) -> Parsed {
        self.push_depth()?;
        let tag = self.next()?;
        match tag {
            // A crate root.
            b'C' => {
                let _disambiguator = self.disambiguator()?;
                let name = self.ident()?;
                self.print_ident(&name)?;
            }
            // `<parent>::<name>`, in some namespace.
            b'N' => {
                let namespace = self.next()?;
                if !namespace.is_ascii_alphabetic() {
                    return Err(Invalid);
                }
                self.print_path(in_value)?;
                let disambiguator = self.disambiguator()?;
                let name = self.ident()?;
                if namespace.is_ascii_uppercase() {
                    // A compiler-generated item, which has no name a person
                    // wrote. It is rendered in braces so it cannot be mistaken
                    // for one: `{closure#0}`, `{shim:vtable#0}`.
                    self.print("::{")?;
                    match namespace {
                        b'C' => self.print("closure")?,
                        b'S' => self.print("shim")?,
                        other => self.print_char(char::from(other))?,
                    }
                    if !name.is_empty() {
                        self.print(":")?;
                        self.print_ident(&name)?;
                    }
                    self.print("#")?;
                    self.print_u64(disambiguator)?;
                    self.print("}")?;
                } else if !name.is_empty() {
                    self.print("::")?;
                    self.print_ident(&name)?;
                }
            }
            // An item in an impl: inherent (`M`), trait (`X`), or the trait
            // definition itself (`Y`).
            b'M' | b'X' | b'Y' => {
                if tag != b'Y' {
                    // The impl's own path says which module the `impl` block
                    // was written in. That disambiguates two impls in one
                    // crate; it is not part of the item's name, and printing
                    // it would produce a path that does not exist.
                    let _disambiguator = self.disambiguator()?;
                    self.skipping(|this| this.print_path(false))?;
                }
                self.print("<")?;
                self.print_type()?;
                if tag != b'M' {
                    self.print(" as ")?;
                    self.print_path(false)?;
                }
                self.print(">")?;
            }
            // A generic instantiation.
            b'I' => {
                self.print_path(in_value)?;
                // The turbofish exists precisely because `foo<T>` is ambiguous
                // in expression position and `Foo<T>` is not.
                if in_value {
                    self.print("::")?;
                }
                self.print("<")?;
                self.print_list(", ", Self::print_generic_arg)?;
                self.print(">")?;
            }
            b'B' => self.backref(|this| this.print_path(in_value))?,
            _ => return Err(Invalid),
        }
        self.pop_depth();
        Ok(())
    }

    fn print_type(&mut self) -> Parsed {
        self.push_depth()?;
        let tag = self.next()?;
        if let Some(name) = basic_type(tag) {
            self.print(name)?;
            self.pop_depth();
            return Ok(());
        }
        match tag {
            // `&T` and `&mut T`, either of which may carry a lifetime.
            b'R' | b'Q' => {
                self.print("&")?;
                if self.eat(b'L') {
                    let index = self.base62()?;
                    if index != 0 {
                        self.print_lifetime(index)?;
                        self.print(" ")?;
                    }
                }
                if tag == b'Q' {
                    self.print("mut ")?;
                }
                self.print_type()?;
            }
            b'P' => {
                self.print("*const ")?;
                self.print_type()?;
            }
            b'O' => {
                self.print("*mut ")?;
                self.print_type()?;
            }
            // `[T; N]`. The length is written bare: the brackets around it are
            // already unambiguous, so the braces a generic argument would need
            // would only be noise.
            b'A' => {
                self.print("[")?;
                self.print_type()?;
                self.print("; ")?;
                self.print_const(false)?;
                self.print("]")?;
            }
            // `[T]`
            b'S' => {
                self.print("[")?;
                self.print_type()?;
                self.print("]")?;
            }
            // `(T, U)`, where a one-element tuple needs its trailing comma or
            // it reads as parentheses.
            b'T' => {
                self.print("(")?;
                let count = self.print_list(", ", Self::print_type)?;
                if count == 1 {
                    self.print(",")?;
                }
                self.print(")")?;
            }
            b'F' => self.print_fn_sig()?,
            b'D' => self.print_dyn()?,
            b'B' => self.backref(Self::print_type)?,
            // Anything else begins a path naming a type.
            _ => {
                self.position -= 1;
                self.print_path(false)?;
            }
        }
        self.pop_depth();
        Ok(())
    }

    /// `<fn-sig> = [<binder>] ["U"] ["K" <abi>] {<type>} "E" <type>`
    fn print_fn_sig(&mut self) -> Parsed {
        let bound = self.open_binder()?;
        if self.eat(b'U') {
            self.print("unsafe ")?;
        }
        if self.eat(b'K') {
            self.print("extern \"")?;
            if self.eat(b'C') {
                self.print("C")?;
            } else {
                let abi = self.ident()?;
                if abi.is_empty() {
                    return Err(Invalid);
                }
                self.print_abi(&abi)?;
            }
            self.print("\" ")?;
        }
        self.print("fn(")?;
        self.print_list(", ", Self::print_type)?;
        self.print(")")?;
        // A unit return is written but not printed, matching how it is written
        // but not printed in source.
        if self.eat(b'u') {
            // Nothing to say.
        } else {
            self.print(" -> ")?;
            self.print_type()?;
        }
        self.close_binder(bound);
        Ok(())
    }

    /// `<dyn-bounds> = [<binder>] {<dyn-trait>} "E"`, then a lifetime.
    fn print_dyn(&mut self) -> Parsed {
        self.print("dyn ")?;
        let bound = self.open_binder()?;
        self.print_list(" + ", Self::print_dyn_trait)?;
        self.close_binder(bound);
        if !self.eat(b'L') {
            return Err(Invalid);
        }
        let index = self.base62()?;
        if index != 0 {
            self.print(" + ")?;
            self.print_lifetime(index)?;
        }
        Ok(())
    }

    /// One trait in a `dyn` bound, with any associated-type bindings folded
    /// into the same angle brackets its generic arguments use.
    fn print_dyn_trait(&mut self) -> Parsed {
        self.push_depth()?;
        let mut open = self.print_path_leaving_generics_open()?;
        while self.eat(b'p') {
            if open {
                self.print(", ")?;
            } else {
                self.print("<")?;
                open = true;
            }
            let name = self.ident()?;
            self.print_ident(&name)?;
            self.print(" = ")?;
            // A binding names an associated type or an associated constant,
            // and the two are written identically apart from the `K`.
            if self.eat(b'K') {
                self.print_const(true)?;
            } else {
                self.print_type()?;
            }
        }
        if open {
            self.print(">")?;
        }
        self.pop_depth();
        Ok(())
    }

    /// Prints a trait path, returning whether it left a `<` open.
    ///
    /// A `dyn` trait's associated bindings share one pair of angle brackets
    /// with its generic arguments — `dyn Iterator<u8, Item = u32>` — so the
    /// path cannot close the brackets it opened, and has to report that it left
    /// them open instead.
    ///
    /// The reason this is a separate walk rather than a peek at the next byte:
    /// the generic construct can arrive through a backreference. Two `dyn`
    /// types over the same trait reference with different bindings are exactly
    /// that, and `rustc` emits them. Reading only the literal tag renders the
    /// second one as `dyn Trait<u64><Item = u8>`, which is not Rust syntax and
    /// not a name.
    fn print_path_leaving_generics_open(&mut self) -> Parsed<bool> {
        self.push_depth()?;
        let opened = if self.eat(b'B') {
            // Whether generics were left open is a property of what the
            // backreference points at, so the answer has to travel back out of
            // the jump with it.
            let mut opened = false;
            self.backref(|this| {
                opened = this.print_path_leaving_generics_open()?;
                Ok(())
            })?;
            opened
        } else if self.eat(b'I') {
            self.print_path(false)?;
            self.print("<")?;
            self.print_list(", ", Self::print_generic_arg)?;
            true
        } else {
            self.print_path(false)?;
            false
        };
        self.pop_depth();
        Ok(opened)
    }

    fn print_generic_arg(&mut self) -> Parsed {
        if self.eat(b'L') {
            let index = self.base62()?;
            self.print_lifetime(index)
        } else if self.eat(b'K') {
            // A generic argument is a value position: `Foo<{&0}>` is how a
            // structural constant is written there, braces and all. Scalars are
            // unaffected, since [`Self::print_const`] only brackets the
            // constants that need it.
            self.print_const(true)
        } else {
            self.print_type()
        }
    }

    /// `<const> = <type> <const-data> | "p" | <backref>`
    fn print_const(&mut self, in_value: bool) -> Parsed {
        self.push_depth()?;
        let tag = self.next()?;
        match tag {
            // An inferred or erased constant.
            b'p' => self.print("_")?,
            b'b' => match parse_hex(self.hex_payload()?)? {
                0 => self.print("false")?,
                1 => self.print("true")?,
                _ => return Err(Invalid),
            },
            b'c' => {
                let value = parse_hex(self.hex_payload()?)?;
                let value = u32::try_from(value).map_err(|_| Invalid)?;
                let character = char::from_u32(value).ok_or(Invalid)?;
                self.print("'")?;
                self.print_escaped_char(character)?;
                self.print("'")?;
            }
            // Signed and unsigned integers alike. The sign is a leading `n`,
            // not two's complement, so the magnitude is read the same way for
            // both and only the prefix differs.
            b'a' | b'h' | b'i' | b'j' | b'l' | b'm' | b'n' | b'o' | b's' | b't' | b'x' | b'y' => {
                if self.eat(b'n') {
                    self.print("-")?;
                }
                let digits = self.hex_payload()?;
                if digits.len() > 16 {
                    // Wider than a `u64`, so it stays in the base it was
                    // written in. A 39-digit decimal is not more legible than
                    // the bit pattern it came from, and this is the form the
                    // rest of the ecosystem prints.
                    self.print("0x")?;
                    self.print(digits)?;
                } else {
                    self.print_u64(parse_hex(digits)?)?;
                }
            }
            // A `str` constant names a place rather than a value, and a place
            // is written as a dereference of the literal that occupies it.
            b'e' => {
                if in_value {
                    self.print("{")?;
                }
                self.print("*")?;
                self.print_str_literal()?;
                if in_value {
                    self.print("}")?;
                }
            }
            // `&*"..."` is what a shared reference to a `str` constant works
            // out to, and Rust spells that `"..."`. Collapsing it here is what
            // makes a `&'static str` const generic print as the literal
            // somebody wrote, and being a literal it needs no braces either.
            b'R' if self.peek() == Some(b'e') => {
                self.next()?;
                self.print_str_literal()?;
            }
            // Structural constants, which only appear in const-generic
            // positions and are wrapped in braces when they sit in a path,
            // because that is how such a value is written in source.
            b'R' | b'Q' | b'A' | b'T' | b'V' => {
                if in_value {
                    self.print("{")?;
                }
                self.print_structural_const(tag)?;
                if in_value {
                    self.print("}")?;
                }
            }
            b'B' => self.backref(|this| this.print_const(in_value))?,
            _ => return Err(Invalid),
        }
        self.pop_depth();
        Ok(())
    }

    fn print_structural_const(&mut self, tag: u8) -> Parsed {
        match tag {
            // `&C` and `&mut C`.
            b'R' | b'Q' => {
                self.print("&")?;
                if tag == b'Q' {
                    self.print("mut ")?;
                }
                self.print_const(false)?;
            }
            // `[C, C, ...]`
            b'A' => {
                self.print("[")?;
                self.print_list(", ", |this| this.print_const(false))?;
                self.print("]")?;
            }
            // `(C, C)`
            b'T' => {
                self.print("(")?;
                let count = self.print_list(", ", |this| this.print_const(false))?;
                if count == 1 {
                    self.print(",")?;
                }
                self.print(")")?;
            }
            // A struct, enum variant, or unit: `Path { field: C }`.
            b'V' => {
                self.print_path(true)?;
                let shape = self.next()?;
                match shape {
                    // Unit: the path is the whole thing.
                    b'U' => {}
                    // Tuple: `Path(C, C)`.
                    b'T' => {
                        self.print("(")?;
                        self.print_list(", ", |this| this.print_const(false))?;
                        self.print(")")?;
                    }
                    // Braced: `Path { name: C }`.
                    b'S' => {
                        self.print(" { ")?;
                        self.print_list(", ", |this| {
                            let _disambiguator = this.disambiguator()?;
                            let name = this.ident()?;
                            this.print_ident(&name)?;
                            this.print(": ")?;
                            this.print_const(false)
                        })?;
                        self.print(" }")?;
                    }
                    _ => return Err(Invalid),
                }
            }
            _ => return Err(Invalid),
        }
        Ok(())
    }

    /// A `str` constant, stored as the hexadecimal of its UTF-8.
    fn print_str_literal(&mut self) -> Parsed {
        self.print("\"")?;
        self.print_const_str()?;
        self.print("\"")
    }

    /// The contents of a `str` constant, without the quotes around them.
    fn print_const_str(&mut self) -> Parsed {
        let start = self.position;
        let digits = self
            .rest()
            .bytes()
            .take_while(u8::is_ascii_hexdigit)
            .count();
        self.position += digits;
        if !self.eat(b'_') {
            return Err(Invalid);
        }
        if digits % 2 != 0 {
            return Err(Invalid);
        }
        self.spend(digits as u32)?;

        let hex: &str = &self.input[start..start + digits];
        // Decoded a character at a time so an invalid sequence is caught here
        // rather than producing a `String` that lies about its contents.
        let mut bytes: Vec<u8> = Vec::with_capacity(digits / 2);
        for pair in hex.as_bytes().chunks_exact(2) {
            let high = hex_value(pair[0]).ok_or(Invalid)?;
            let low = hex_value(pair[1]).ok_or(Invalid)?;
            bytes.push((high << 4) | low);
        }
        let text = core::str::from_utf8(&bytes).map_err(|_| Invalid)?;
        for character in text.chars() {
            self.print_escaped_char(character)?;
        }
        Ok(())
    }

    /// Writes a character as it would appear inside a Rust literal.
    ///
    /// Delegated to `escape_debug` rather than hand-rolled, because the
    /// interesting part of the decision is which code points are printable, and
    /// that is a Unicode table that changes with the standard. Escaping
    /// everything above ASCII instead would render `'Ñ'` as `'\u{d1}'`, which
    /// is correct and unreadable.
    fn print_escaped_char(&mut self, character: char) -> Parsed {
        for escaped in character.escape_debug() {
            self.print_char(escaped)?;
        }
        Ok(())
    }

    /// An ABI string, which is written with `_` where the real name has `-`.
    fn print_abi(&mut self, ident: &Ident<'_>) -> Parsed {
        if ident.punycode {
            // No ABI name needs punycode, and accepting one here would let a
            // symbol put arbitrary characters inside `extern "..."`.
            return Err(Invalid);
        }
        // Copied to a buffer first so the substitution cannot be defeated by
        // the printer's own chunking.
        let mut text = String::with_capacity(ident.text.len());
        for character in ident.text.chars() {
            text.push(if character == '_' { '-' } else { character });
        }
        self.print(&text)
    }

    // ---- lexical pieces --------------------------------------------------

    /// `<base-62-number> = {<0-9a-zA-Z>} "_"`, offset by one so that the empty
    /// encoding can mean zero.
    fn base62(&mut self) -> Parsed<u64> {
        if self.eat(b'_') {
            return Ok(0);
        }
        let mut value: u64 = 0;
        loop {
            let byte = self.next()?;
            if byte == b'_' {
                return value.checked_add(1).ok_or(Invalid);
            }
            let digit = match byte {
                b'0'..=b'9' => u64::from(byte - b'0'),
                b'a'..=b'z' => u64::from(byte - b'a') + 10,
                b'A'..=b'Z' => u64::from(byte - b'A') + 36,
                _ => return Err(Invalid),
            };
            value = value
                .checked_mul(62)
                .and_then(|scaled| scaled.checked_add(digit))
                .ok_or(Invalid)?;
        }
    }

    /// `<disambiguator> = "s" <base-62-number>`.
    ///
    /// Offset by one on top of the offset [`Self::base62`] already applies, so
    /// that the three shortest encodings are the three smallest values: absent
    /// is 0, `s_` is 1, `s0_` is 2. Getting this wrong renames every closure in
    /// a profile by one.
    fn disambiguator(&mut self) -> Parsed<u64> {
        if self.eat(b's') {
            self.base62()?.checked_add(1).ok_or(Invalid)
        } else {
            Ok(0)
        }
    }

    /// `<undisambiguated-identifier> = ["u"] <decimal-number> ["_"] <bytes>`
    fn ident(&mut self) -> Parsed<Ident<'a>> {
        let punycode = self.eat(b'u');
        let length = self.decimal_number()?;
        // The separator is present when the identifier would otherwise start
        // with something that reads as more of the length.
        self.eat(b'_');
        let start = self.position;
        let end = start.checked_add(length).ok_or(Invalid)?;
        if !self.input.is_char_boundary(end) {
            return Err(Invalid);
        }
        self.position = end;
        self.spend(length.min(u32::MAX as usize) as u32)?;
        Ok(Ident {
            punycode,
            text: &self.input[start..end],
        })
    }

    /// `<decimal-number> = "0" | <1-9> {<0-9>}`
    ///
    /// The alternation is load-bearing rather than stylistic. Numbers here are
    /// not delimited, so `00` has to be two zeroes and not one malformed
    /// number: a closure with an empty name followed by another one is written
    /// exactly that way, and reading it greedily silently loses every symbol
    /// with two nested closures in it.
    fn decimal_number(&mut self) -> Parsed<usize> {
        let start = self.position;
        if self.eat(b'0') {
            return Ok(0);
        }
        let digits = self.rest().bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            return Err(Invalid);
        }
        self.position += digits;
        self.input[start..start + digits]
            .parse()
            .map_err(|_| Invalid)
    }

    /// `{<hex-digit>} "_"`, the payload of a scalar constant, as written.
    ///
    /// Returned undecoded because how wide a value is decides how it is
    /// printed, and that is a question about the digits rather than the number.
    fn hex_payload(&mut self) -> Parsed<&'a str> {
        let start = self.position;
        let digits = self
            .rest()
            .bytes()
            .take_while(u8::is_ascii_hexdigit)
            .count();
        let text = &self.input[start..start + digits];
        self.position += digits;
        self.spend(digits.min(u32::MAX as usize) as u32)?;
        if !self.eat(b'_') {
            return Err(Invalid);
        }
        // Deliberately not rejecting a leading zero. The encoding is canonical,
        // so `0a_` is not a spelling `rustc` produces — but the reference
        // implementation reads it as 10, and refusing a symbol the rest of the
        // ecosystem names is a worse failure than accepting a spelling that
        // will never arrive. The terminating `_` already makes the run
        // unambiguous, so there is nothing here to disambiguate.
        Ok(text)
    }

    /// Re-reads a construct written earlier in the symbol.
    fn backref(&mut self, read: impl FnOnce(&mut Self) -> Parsed) -> Parsed {
        // `next` has already consumed the `B`, so this is where it was.
        let tag_position = self.position - 1;
        let target = usize::try_from(self.base62()?).map_err(|_| Invalid)?;
        // Strictly backwards. Equality would be a construct that contains
        // itself, and anything greater would let a symbol drive the cursor in
        // circles. This single comparison is what makes the recursion below
        // provably finite.
        if target >= tag_position {
            return Err(Invalid);
        }
        let resume_at = self.position;
        self.position = target;
        let result = read(self);
        self.position = resume_at;
        result
    }

    /// Prints elements until the `E` that closes the list, returning how many
    /// there were.
    fn print_list(
        &mut self,
        separator: &str,
        mut element: impl FnMut(&mut Self) -> Parsed,
    ) -> Parsed<usize> {
        let mut count = 0;
        loop {
            if self.eat(b'E') {
                return Ok(count);
            }
            if self.peek().is_none() {
                return Err(Invalid);
            }
            if count > 0 {
                self.print(separator)?;
            }
            let before = self.position;
            element(self)?;
            // An element that consumed nothing would spin here forever. No
            // production in the grammar is empty, so this can only fire on a
            // bug in this file, but the cost of the check is one comparison
            // and the cost of being wrong is a profiler that never exits.
            if self.position == before {
                return Err(Invalid);
            }
            count += 1;
        }
    }

    // ---- lifetimes -------------------------------------------------------

    /// Opens a `for<'a, ...>` binder if one is written here, returning how many
    /// lifetimes it introduced.
    fn open_binder(&mut self) -> Parsed<u64> {
        if !self.eat(b'G') {
            return Ok(0);
        }
        // Offset by one, on the same reasoning as [`Self::disambiguator`]: a
        // binder that bound nothing would not be written at all, so the
        // shortest encoding has to mean one lifetime rather than none.
        let count = self.base62()?.checked_add(1).ok_or(Invalid)?;
        // Each lifetime costs a name in the output, so the budget would catch
        // an absurd count eventually; refusing outright is clearer and keeps
        // the arithmetic below in a range where it obviously cannot wrap.
        if count > u64::from(MAX_DEPTH) {
            return Err(Invalid);
        }
        self.print("for<")?;
        for index in 0..count {
            if index > 0 {
                self.print(", ")?;
            }
            self.bound_lifetimes += 1;
            // Index 1 is always the innermost binding, which is the one just
            // introduced.
            self.print_lifetime(1)?;
        }
        self.print("> ")?;
        Ok(count)
    }

    fn close_binder(&mut self, count: u64) {
        self.bound_lifetimes -= count;
    }

    /// Names a lifetime from its de Bruijn index, counting outwards.
    fn print_lifetime(&mut self, index: u64) -> Parsed {
        self.print("'")?;
        // Index zero is the erased lifetime, which has no name to give.
        if index == 0 {
            return self.print("_");
        }
        // An index reaching past every binder in scope cannot name anything.
        let depth = self.bound_lifetimes.checked_sub(index).ok_or(Invalid)?;
        // The alphabet runs out at 26. Past that the name becomes the index
        // itself rather than a letter with a suffix, because `'a1` reads like a
        // lifetime somebody wrote and `'_26` cannot be mistaken for one.
        if depth < 26 {
            self.print_char(char::from(b'a' + depth as u8))
        } else {
            self.print("_")?;
            self.print_u64(depth)
        }
    }

    // ---- cursor ----------------------------------------------------------

    fn rest(&self) -> &str {
        // `position` only ever moves to a boundary this parser established, so
        // the slice is valid; `get` rather than indexing keeps that a
        // recoverable mistake instead of a panic.
        self.input.get(self.position..).unwrap_or("")
    }

    fn peek(&self) -> Option<u8> {
        self.rest().bytes().next()
    }

    fn next(&mut self) -> Parsed<u8> {
        let byte = self.peek().ok_or(Invalid)?;
        self.position += 1;
        self.spend(1)?;
        Ok(byte)
    }

    fn eat(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn push_depth(&mut self) -> Parsed {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(Invalid);
        }
        self.spend(1)
    }

    fn pop_depth(&mut self) {
        self.depth -= 1;
    }

    fn spend(&mut self, amount: u32) -> Parsed {
        self.budget = self.budget.checked_sub(amount).ok_or(Invalid)?;
        Ok(())
    }

    // ---- output ----------------------------------------------------------

    /// Runs `body` with printing turned off, so the input it covers is
    /// consumed but contributes nothing to the name.
    fn skipping(&mut self, body: impl FnOnce(&mut Self) -> Parsed) -> Parsed {
        let was_printing = self.printing;
        self.printing = false;
        let result = body(self);
        self.printing = was_printing;
        result
    }

    fn print(&mut self, text: &str) -> Parsed {
        self.spend(text.len().min(u32::MAX as usize) as u32)?;
        if self.printing {
            self.out.push_str(text);
        }
        Ok(())
    }

    fn print_char(&mut self, character: char) -> Parsed {
        self.spend(1)?;
        if self.printing {
            self.out.push(character);
        }
        Ok(())
    }

    fn print_ident(&mut self, ident: &Ident<'_>) -> Parsed {
        if !ident.punycode {
            return self.print(ident.text);
        }
        // Charged quadratically, because that is what the work is: decoding
        // inserts each character into the middle of what has been decoded so
        // far, so a descending run of code points moves the whole buffer every
        // time. Spending only the length would leave the budget bounding the
        // size of the output without bounding the time spent producing it,
        // which is not the promise it is here to make.
        let length = ident.text.len().min(u32::MAX as usize) as u32;
        self.spend(length.saturating_mul(length).max(length))?;
        let mut decoded = String::with_capacity(ident.text.len());
        if punycode::decode(ident.text, &mut decoded) {
            if self.printing {
                self.out.push_str(&decoded);
            }
            return Ok(());
        }

        // Undecodable, which means the symbol is damaged rather than that the
        // rest of it is worthless. Showing the encoded form marked as encoded
        // keeps the surrounding path, and cannot be mistaken for a name.
        self.print("punycode{")?;
        match ident.text.rfind('_') {
            // Restored to the delimiter the RFC uses, so what is shown is
            // punycode that another tool would accept rather than the
            // symbol-safe spelling of it.
            Some(delimiter) => {
                self.print(&ident.text[..delimiter])?;
                self.print("-")?;
                self.print(&ident.text[delimiter + 1..])?;
            }
            None => self.print(ident.text)?,
        }
        self.print("}")
    }

    fn print_u64(&mut self, value: u64) -> Parsed {
        // Widest `u64` is 20 digits.
        let mut digits = [0u8; 20];
        let mut written = 0;
        let mut remaining = value;
        loop {
            digits[written] = b'0' + (remaining % 10) as u8;
            written += 1;
            remaining /= 10;
            if remaining == 0 {
                break;
            }
        }
        self.spend(written as u32)?;
        if self.printing {
            for &digit in digits[..written].iter().rev() {
                self.out.push(char::from(digit));
            }
        }
        Ok(())
    }
}

/// An identifier as it was written, which may still need decoding.
struct Ident<'a> {
    punycode: bool,
    text: &'a str,
}

impl Ident<'_> {
    fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// The single-letter types, which cover everything with no parameters.
fn basic_type(tag: u8) -> Option<&'static str> {
    Some(match tag {
        b'a' => "i8",
        b'b' => "bool",
        b'c' => "char",
        b'd' => "f64",
        b'e' => "str",
        b'f' => "f32",
        b'h' => "u8",
        b'i' => "isize",
        b'j' => "usize",
        b'l' => "i32",
        b'm' => "u32",
        b'n' => "i128",
        b'o' => "u128",
        b'p' => "_",
        b's' => "i16",
        b't' => "u16",
        b'u' => "()",
        b'v' => "...",
        b'x' => "i64",
        b'y' => "u64",
        b'z' => "!",
        _ => return None,
    })
}

/// Reads a hexadecimal payload, which the caller has already bounded to 16
/// digits so that it fits.
fn parse_hex(digits: &str) -> Parsed<u64> {
    if digits.is_empty() {
        // No digits at all is how zero is written.
        return Ok(0);
    }
    u64::from_str_radix(digits, 16).map_err(|_| Invalid)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{BUDGET, MAX_DEPTH};

    /// Demangles a whole symbol, prefix included, the way callers see it.
    fn rendered(symbol: &str) -> Option<String> {
        let mut out = String::new();
        crate::symbol::demangle(symbol, &mut out).then_some(out)
    }

    /// The shapes below are checked against `rustc-demangle` on 19,121 symbols
    /// taken from real binaries by `tests/demangle.rs`. These few are repeated
    /// here so that a failure points at the construct rather than at a corpus.
    #[test]
    fn the_shapes_that_appear_in_every_binary() {
        for (symbol, expected) in [
            ("_RNvCs785SGTk9yHm_1p14probe_function", "p::probe_function"),
            (
                "_RINvCsi8n9QKePl5z_1p10generic_fnINtB2_6HolderyEEB2_",
                "p::generic_fn::<p::Holder<u64>>",
            ),
            (
                "_RNCNvCsi8n9QKePl5z_1p15makes_a_closure0B3_",
                "p::makes_a_closure::{closure#0}",
            ),
            (
                "_RNvXCsi8n9QKePl5z_1pINtB2_6HolderpENtB2_5Shape4area",
                "<p::Holder<_> as p::Shape>::area",
            ),
            (
                "_RNvCsi8n9QKePl5z_1pu18ncd_name_d1a1d7d6c",
                "p::ünïcödé_name",
            ),
        ] {
            assert_eq!(rendered(symbol).as_deref(), Some(expected), "{symbol}");
        }
    }

    /// Two encodings differ by one byte and mean adjacent closures. Getting the
    /// offset wrong renames every closure in a profile, which is the kind of
    /// error that looks like a plausible answer.
    #[test]
    fn the_disambiguator_is_offset_by_one_past_the_base_62_offset() {
        for (symbol, expected) in [
            ("_RNCNvCs0_1p4name0", "p::name::{closure#0}"),
            ("_RNCNvCs0_1p4names_0", "p::name::{closure#1}"),
            ("_RNCNvCs0_1p4names0_0", "p::name::{closure#2}"),
        ] {
            assert_eq!(rendered(symbol).as_deref(), Some(expected), "{symbol}");
        }
    }

    /// `<decimal-number> = "0" | <1-9> {<0-9>}`. Two nested closures write two
    /// empty names as `00`, and reading that greedily loses the symbol.
    #[test]
    fn a_zero_length_name_does_not_swallow_the_digit_after_it() {
        assert_eq!(
            rendered("_RNCNCNvCs0_1p4name00").as_deref(),
            Some("p::name::{closure#0}::{closure#0}")
        );
    }

    /// A backreference must point strictly before the `B` that names it. That
    /// is the whole termination argument for a grammar that can otherwise
    /// re-enter itself.
    #[test]
    fn a_backreference_that_does_not_point_backwards_is_refused() {
        // `B` at body offset 0 pointing at body offset 0.
        assert_eq!(rendered("_RB0_"), None);
        assert_eq!(rendered("_RB_"), None);
        // Forwards, past its own position.
        assert_eq!(rendered("_RNvB8_1p4name"), None);
    }

    #[test]
    fn a_backreference_reaching_past_the_end_is_refused() {
        assert_eq!(rendered("_RNvCs0_1p4nameBzzzzzzz_"), None);
    }

    /// A crate disambiguator is eleven base-62 digits of a 64-bit hash, which
    /// is close enough to the top of the range that an unchecked accumulator
    /// would wrap on real symbols rather than on crafted ones.
    #[test]
    fn a_full_width_disambiguator_does_not_overflow() {
        assert_eq!(
            rendered("_RNvCscNHxvxvlkyC_15allocation_free7counted").as_deref(),
            Some("allocation_free::counted")
        );
        // Genuinely past `u64`, which must be a refusal rather than a wrap.
        assert_eq!(rendered("_RNvCszzzzzzzzzzzzzzzzzzzz_1p4name"), None);
    }

    #[test]
    #[cfg_attr(miri, ignore = "256 levels of interpreted recursion, twice")]
    fn nesting_deeper_than_the_limit_is_refused_rather_than_overflowing_the_stack() {
        // Each `R` is one `&`, and each is one level of recursion.
        let deep = format!("_RNvCs0_1p3tag{}u", "R".repeat(MAX_DEPTH as usize + 10));
        assert_eq!(rendered(&deep), None);
        // Just inside the limit still parses, so the guard is not simply
        // rejecting everything.
        let shallow = format!("_RINvCs0_1p3tag{}uEB2_", "R".repeat(32));
        assert!(rendered(&shallow).is_some());
    }

    /// Backreferences can share subtrees, so a short symbol can name a tree
    /// with exponentially many nodes in it. It is finite and it is shallow, so
    /// neither the depth limit nor the backwards rule sees anything wrong; only
    /// a budget on work does.
    #[test]
    #[cfg_attr(miri, ignore = "asserts wall-clock time, which Miri does not model")]
    fn an_exponentially_expanding_symbol_is_refused_in_bounded_time() {
        // Each tuple repeats the previous construct twice, doubling the
        // rendered output at every step.
        let mut symbol = String::from("_RINvCs0_1p3tagu");
        // Backreference targets are offsets into the *body*, which starts after
        // the two-byte `_R`. Getting this wrong is not a harmless off-by-one:
        // it makes the first backreference point at itself, which the
        // strictly-backwards rule rejects three levels of grammar before the
        // budget is ever consulted, and the test then passes without testing
        // anything.
        const PREFIX: usize = "_R".len();
        let mut previous = symbol.len() - PREFIX - 1;
        for _ in 0..40 {
            let next = symbol.len() - PREFIX;
            symbol.push_str(&format!("TB{}_B{}_E", base62(previous), base62(previous)));
            previous = next;
        }
        symbol.push_str("EB2_");

        let started = std::time::Instant::now();
        let result = rendered(&symbol);
        let elapsed = started.elapsed();

        assert_eq!(result, None, "should have run out of budget");
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "took {elapsed:?}, so the budget is not bounding the work"
        );
    }

    /// The inverse of `base62`, for building the test above.
    fn base62(value: usize) -> String {
        const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        // The encoding is offset by one, so zero is the empty string.
        let Some(mut value) = value.checked_sub(1) else {
            return String::new();
        };
        let mut out = Vec::new();
        loop {
            out.push(DIGITS[value % 62]);
            value /= 62;
            if value == 0 {
                break;
            }
        }
        out.reverse();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn the_budget_is_large_enough_that_no_real_symbol_reaches_it() {
        // The longest symbol measured in a real build was 921 bytes.
        const { assert!(BUDGET > 100_000) };
    }

    /// A future mangling version announces itself with a leading digit. Reading
    /// it as version 0 would produce a name from a grammar it was not written
    /// in.
    #[test]
    fn a_future_encoding_version_is_refused_rather_than_guessed_at() {
        assert_eq!(rendered("_R1NvCs0_1p4name"), None);
        assert_eq!(rendered("_R9NvCs0_1p4name"), None);
    }

    #[test]
    fn trailing_bytes_that_are_not_a_suffix_are_a_refusal() {
        assert_eq!(rendered("_RNvCs0_1p4name!!!"), None);
    }

    #[test]
    fn a_vendor_suffix_is_kept_but_an_llvm_clone_marker_is_not() {
        assert_eq!(
            rendered("_RNvCs0_1p4name.cold.1").as_deref(),
            Some("p::name.cold.1")
        );
        assert_eq!(
            rendered("_RNvCs0_1p4name$tlv$init").as_deref(),
            Some("p::name$tlv$init")
        );
        assert_eq!(
            rendered("_RNvCs0_1p4name.llvm.4708").as_deref(),
            Some("p::name")
        );
    }

    #[test]
    fn a_truncated_symbol_is_refused_at_every_length() {
        let whole = "_RINvCsi8n9QKePl5z_1p10generic_fnINtB2_6HolderyEEB2_";
        for end in 3..whole.len() {
            // Not asserting a refusal: a prefix of a symbol can be a symbol,
            // and `_RINv...HolderyEE` is exactly the same item without the
            // instantiating crate on the end. What must hold is that nothing
            // panics or hangs, and that a success is a name rather than the
            // input handed back wearing one.
            let truncated = &whole[..end];
            if let Some(name) = rendered(truncated) {
                assert_ne!(name, truncated);
                assert!(!name.starts_with('_'), "{truncated:?} rendered as {name}");
            }
        }
    }

    /// A `dyn` trait's generic arguments and its associated bindings share one
    /// pair of angle brackets, and the generic construct can arrive through a
    /// backreference. `rustc` emits exactly that for two trait objects over the
    /// same trait reference with different bindings, and reading only the
    /// literal tag renders the second as `dyn Trait<u64><Out = u8>`.
    ///
    /// Found by review, on output from a real compiler; the 201,457-symbol
    /// corpus contains no instance, because `rustc` backreferences the inner
    /// path far more often than the whole generic construct.
    #[test]
    fn a_dyn_trait_reached_through_a_backreference_keeps_one_pair_of_brackets() {
        // Compiled from `fn pair<A: ?Sized, B: ?Sized>()` instantiated with
        // `dyn Shape<u64, Out = u16>` and `dyn Shape<u64, Out = u8>`.
        let symbol = "_RINvCsgPCKjwcKrmH_8dynprobe4pairDINtB2_5ShapeyEp3OuttEL_DBv_p3OuthEL_EB2_";
        assert_eq!(
            rendered(symbol).as_deref(),
            Some(
                "dynprobe::pair::<dyn dynprobe::Shape<u64, Out = u16>, \
                 dyn dynprobe::Shape<u64, Out = u8>>"
            )
        );
    }

    /// A binding can name an associated constant rather than an associated
    /// type. Unstable in the language today, so nothing in the corpus reaches
    /// it, and stable once `associated_const_equality` lands.
    #[test]
    fn a_dyn_associated_constant_binding_is_understood() {
        assert_eq!(
            rendered("_RINvCs0_1p3tagDNtCs0_1p5Traitp6OutputKh7_EL_EB2_").as_deref(),
            Some("p::tag::<dyn p::Trait<Output = 7>>")
        );
    }

    /// A `str` constant is a place, so `&*"lit"` is what a shared reference to
    /// one works out to, and Rust spells that `"lit"`. One of the six defects
    /// fuzzing found, which review pointed out had gained a fix but no test.
    #[test]
    fn a_shared_reference_to_a_string_constant_collapses_to_the_literal() {
        assert_eq!(
            rendered("_RINvCs0_1p1fKRe_E").as_deref(),
            Some("p::f::<\"\">")
        );
        assert_eq!(
            rendered("_RINvCs0_1p1fKRe68656c6c6f_E").as_deref(),
            Some("p::f::<\"hello\">")
        );
        // Not collapsed when the reference is not a shared one, because
        // `&mut ""` is a different type from `&mut *""`.
        assert_eq!(
            rendered("_RINvCs0_1p1fKQe_E").as_deref(),
            Some("p::f::<{&mut *\"\"}>")
        );
        // Nor when the `str` stands alone.
        assert_eq!(
            rendered("_RINvCs0_1p1fKe_E").as_deref(),
            Some("p::f::<{*\"\"}>")
        );
    }

    /// The encoding is canonical, so a leading zero is a spelling `rustc` never
    /// emits — but the reference implementation reads it, and refusing a symbol
    /// the rest of the ecosystem names is the worse failure.
    #[test]
    fn a_constant_written_with_a_leading_zero_is_read_rather_than_refused() {
        assert_eq!(
            rendered("_RINvCs0_1p1fKh0a_E").as_deref(),
            Some("p::f::<10>")
        );
        assert_eq!(
            rendered("_RINvCs0_1p1fKb00_E").as_deref(),
            Some("p::f::<false>")
        );
        assert_eq!(
            rendered("_RINvCs0_1p1fKc061_E").as_deref(),
            Some("p::f::<'a'>")
        );
    }

    /// Punycode decoding inserts into the middle of its output, so its cost is
    /// quadratic in the identifier length. Charging the budget linearly for it
    /// left an identifier admissible at a size where decoding it takes about a
    /// second — inside a profiler's shutdown path. Found by review.
    #[test]
    #[cfg_attr(miri, ignore = "asserts wall-clock time, which Miri does not model")]
    fn a_punycode_identifier_is_bounded_by_work_rather_than_by_length() {
        let oversized = "a_".to_string() + &"z".repeat(4096);
        let symbol = format!("_RNvCs0_1pu{}{oversized}", oversized.len());
        let started = std::time::Instant::now();
        assert_eq!(rendered(&symbol), None);
        assert!(started.elapsed() < std::time::Duration::from_millis(100));

        // An identifier of a size that occurs is unaffected: the real one is 18
        // bytes, and the limit lands three orders of magnitude above that.
        let ordinary = "a_".to_string() + &"z".repeat(500);
        let symbol = format!("_RNvCs0_1pu{}{ordinary}", ordinary.len());
        assert!(rendered(&symbol).is_some());
    }

    /// The alphabet has 26 letters and a binder may bind more lifetimes than
    /// that. Found by `tests/demangle_fuzz.rs`, which built a 40-lifetime
    /// binder out of a real symbol; the naming this produced past `'z` was
    /// plausible enough to read as a lifetime somebody had written.
    #[test]
    fn lifetimes_past_the_alphabet_are_numbered_rather_than_suffixed() {
        // The fuzzer's input, verbatim. Not shortened for readability: the
        // `B2_` at the end is a byte offset into this exact string, so editing
        // any of it silently makes it a different symbol.
        let symbol = "_RINvCs785SGTk9yHm_1p3tagINtNtCs6SjEax68zxx_5alloc5boxed3Box\
DGC_INtNtNtCskumHb0IaX0X_4core3ops8function2FnTRL1_hRL0_tEEp6OutputuEL_EEB2_";
        let rendered = rendered(symbol).unwrap();
        assert!(rendered.contains("'z, '_26, '_27,"), "{rendered}");
        assert!(
            rendered.ends_with("&'_38 u8, &'_39 u16), Output = ()>>>"),
            "{rendered}"
        );
        // Everything up to the boundary is still lettered.
        assert!(rendered.contains("for<'a, 'b, 'c,"), "{rendered}");
    }

    /// A lifetime index counts outwards through the enclosing binders. One that
    /// counts past all of them names nothing.
    #[test]
    fn a_lifetime_index_past_every_binder_is_refused() {
        assert_eq!(rendered("_RINvCs0_1p3tagRL5_uEB2_"), None);
    }

    #[test]
    fn a_non_ascii_byte_where_the_grammar_expects_a_tag_is_refused() {
        assert_eq!(rendered("_RNvCs0_1pé"), None);
        assert_eq!(rendered("_Ré"), None);
    }

    /// An identifier length that lands inside a multi-byte character would
    /// panic a parser that sliced without checking.
    #[test]
    fn an_identifier_length_that_splits_a_character_is_refused() {
        assert_eq!(rendered("_RNvCs0_1p1é"), None);
    }
}
