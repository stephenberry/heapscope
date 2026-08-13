//! JSON that survives a round trip.
//!
//! This tool reads a profile it did not write, adds to it, and writes it back.
//! The native format's compatibility rule — stated in every file it produces —
//! is that *a reader must ignore fields it does not know*, and a rewriter that
//! ignores a field by dropping it has not followed the rule. So the requirement
//! here is not "parse JSON" but **preserve everything not deliberately
//! changed**, which rules out the obvious shapes:
//!
//! * Object members keep their **order** and their duplicates. A `BTreeMap`
//!   would reorder every object in the file and silently merge any repeated key.
//! * Numbers keep their **text**. Parsing to `f64` and reformatting turns
//!   `1e3` into `1000`, rounds anything past 2^53, and would corrupt the one
//!   thing the format is careful about — though addresses are strings here for
//!   exactly that reason, `totalBytes` is a JSON number and legitimately reaches
//!   `u64::MAX`, which no `f64` holds.
//!
//! # Why this is a second parser
//!
//! `tests/support/json.rs` also parses this format, and stays separate on
//! purpose: it is the oracle the emitter is checked against, and an oracle that
//! shares code with what it checks agrees with any answer that code gives. This
//! one has a different job again — it is the only one that has to *write* what
//! it read.

use std::fmt;

/// A JSON value, in the shape it was read.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// The number exactly as it was written. Never reformatted, and interpreted
    /// only where a caller asks for [`Value::as_u64`].
    Number(String),
    String(String),
    Array(Vec<Value>),
    /// Members in file order. A `Vec` rather than a map because order is part of
    /// what is being preserved, and because JSON permits a repeated key that a
    /// map would quietly discard.
    Object(Vec<(String, Value)>),
}

impl Value {
    /// The value of the **first** member named `key`.
    ///
    /// First rather than last, which is the choice `JSON.parse` does not make —
    /// it keeps the last. The difference only shows on a document with a
    /// repeated key, which nothing this tool reads produces; it is written down
    /// so that the answer is a decision rather than an accident of iteration.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(members) => members
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(text) => Some(text),
            _ => None,
        }
    }

    /// The number as a `u64`, or `None` if it is not one.
    ///
    /// Deliberately refuses anything with a fraction or an exponent rather than
    /// rounding it. Every number this tool reads out of a profile is a count, an
    /// index, or a size, and a count that arrived as `1.5` is a file this tool
    /// should not be guessing about.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Number(text) => text.parse().ok(),
            _ => None,
        }
    }

    /// A hexadecimal address string, as the native format writes them.
    ///
    /// The format's own note says why these are strings: a JSON number is a
    /// double in JavaScript, exact only to 2^53, and a 64-bit address is not.
    pub fn as_address(&self) -> Option<u64> {
        let text = self.as_str()?;
        u64::from_str_radix(text.strip_prefix("0x").unwrap_or(text), 16).ok()
    }

    /// Replaces the first member named `key`, or appends one.
    ///
    /// Appending rather than inserting keeps the members this tool did not touch
    /// in the order the file had them, so a diff of a symbolized profile against
    /// its input shows the additions and nothing else.
    pub fn set(&mut self, key: &str, value: Value) {
        let Value::Object(members) = self else {
            return;
        };
        match members.iter_mut().find(|(name, _)| name == key) {
            Some((_, existing)) => *existing = value,
            None => members.push((String::from(key), value)),
        }
    }

    /// A number member, from a `u64`.
    pub fn number(value: u64) -> Value {
        Value::Number(value.to_string())
    }
}

/// Where a document stopped making sense.
#[derive(Debug)]
pub struct ParseError {
    message: String,
    /// Byte offset, which is what a reader can act on: `head -c` reaches it.
    at: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at byte {}", self.message, self.at)
    }
}

impl std::error::Error for ParseError {}

pub fn parse(input: &str) -> Result<Value, ParseError> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        at: 0,
    };
    parser.skip_whitespace();
    let value = parser.value()?;
    parser.skip_whitespace();
    if parser.at != parser.bytes.len() {
        return Err(parser.error("trailing content after the document"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn error(&self, message: impl Into<String>) -> ParseError {
        ParseError {
            message: message.into(),
            at: self.at,
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.bytes.get(self.at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn expect(&mut self, byte: u8) -> Result<(), ParseError> {
        if self.peek() == Some(byte) {
            self.at += 1;
            return Ok(());
        }
        Err(self.error(format!("expected `{}`", byte as char)))
    }

    fn literal(&mut self, word: &str, value: Value) -> Result<Value, ParseError> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            return Ok(value);
        }
        Err(self.error(format!("expected `{word}`")))
    }

    fn value(&mut self) -> Result<Value, ParseError> {
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(byte) => Err(self.error(format!("`{}` begins no JSON value", byte as char))),
            None => Err(self.error("the document ends where a value was expected")),
        }
    }

    fn object(&mut self) -> Result<Value, ParseError> {
        self.expect(b'{')?;
        let mut members = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Value::Object(members));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            members.push((key, self.value()?));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Value::Object(members));
                }
                _ => return Err(self.error("expected `,` or `}`")),
            }
        }
    }

    fn array(&mut self) -> Result<Value, ParseError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_whitespace();
            items.push(self.value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.error("expected `,` or `]`")),
            }
        }
    }

    /// The number's own text, checked for shape but not converted.
    fn number(&mut self) -> Result<Value, ParseError> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        let digits = self.at;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.at += 1;
        }
        if self.at == digits {
            return Err(self.error("a number with no digits"));
        }
        if self.peek() == Some(b'.') {
            self.at += 1;
            let fraction = self.at;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.at += 1;
            }
            if self.at == fraction {
                return Err(self.error("a decimal point with no digits after it"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            let exponent = self.at;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.at += 1;
            }
            if self.at == exponent {
                return Err(self.error("an exponent with no digits"));
            }
        }
        // Every byte consumed above is ASCII, so the range is a character
        // boundary and this cannot panic.
        Ok(Value::Number(
            String::from_utf8_lossy(&self.bytes[start..self.at]).into_owned(),
        ))
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let Some(byte) = self.peek() else {
                return Err(self.error("the document ends inside a string"));
            };
            match byte {
                b'"' => {
                    self.at += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.at += 1;
                    self.escape(&mut out)?;
                }
                0x00..=0x1F => return Err(self.error("a control character inside a string")),
                _ => {
                    // Copy the whole UTF-8 sequence. The input was a `&str`, so
                    // it is valid UTF-8 and the continuation bytes are there.
                    let width = utf8_width(byte);
                    let end = (self.at + width).min(self.bytes.len());
                    out.push_str(&String::from_utf8_lossy(&self.bytes[self.at..end]));
                    self.at = end;
                }
            }
        }
    }

    fn escape(&mut self, out: &mut String) -> Result<(), ParseError> {
        let Some(byte) = self.peek() else {
            return Err(self.error("the document ends inside an escape"));
        };
        self.at += 1;
        let simple = match byte {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                let first = self.hex4()?;
                // A surrogate pair, which is how JSON writes anything above the
                // basic plane — an emoji in a path reaches this.
                let code = if (0xD800..0xDC00).contains(&first) {
                    if !self.bytes[self.at..].starts_with(b"\\u") {
                        return Err(self.error("a high surrogate with no low surrogate"));
                    }
                    self.at += 2;
                    let second = self.hex4()?;
                    if !(0xDC00..0xE000).contains(&second) {
                        return Err(self.error("a high surrogate followed by a non-surrogate"));
                    }
                    0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                } else {
                    first
                };
                // A lone low surrogate is not a character. Replaced rather than
                // refused: this is a name out of someone's symbol table, and
                // refusing the whole profile over one bad byte would be a worse
                // answer than rendering it as one.
                out.push(char::from_u32(code).unwrap_or(char::REPLACEMENT_CHARACTER));
                return Ok(());
            }
            other => {
                return Err(self.error(format!("`\\{}` is not an escape", other as char)));
            }
        };
        out.push(simple);
        Ok(())
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let end = self.at + 4;
        if end > self.bytes.len() {
            return Err(self.error("a `\\u` escape with fewer than four digits"));
        }
        let mut code = 0u32;
        for &byte in &self.bytes[self.at..end] {
            let digit = (byte as char)
                .to_digit(16)
                .ok_or_else(|| self.error("a `\\u` escape with a non-hexadecimal digit"))?;
            code = code * 16 + digit;
        }
        self.at = end;
        Ok(code)
    }
}

fn utf8_width(byte: u8) -> usize {
    match byte {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

/// Renders `value` as JSON.
///
/// Laid out the way the native writer lays out its own files: a container that
/// holds another container goes one member to a line, and one that holds only
/// scalars stays on one. That keeps a symbolized profile as readable as the
/// profile it came from, which matters because reading it is what someone does
/// when a name comes back wrong.
pub fn render(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value, 0);
    out.push('\n');
    out
}

fn write_value(out: &mut String, value: &Value, depth: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(text) => out.push_str(text),
        Value::String(text) => write_string(out, text),
        Value::Array(items) => {
            let wrap = items.iter().any(is_container);
            write_sequence(out, depth, wrap, '[', ']', items.len(), |out, at, depth| {
                write_value(out, &items[at], depth);
            });
        }
        Value::Object(members) => {
            let wrap = members.iter().any(|(_, value)| is_container(value));
            write_sequence(
                out,
                depth,
                wrap,
                '{',
                '}',
                members.len(),
                |out, at, depth| {
                    write_string(out, &members[at].0);
                    out.push(':');
                    write_value(out, &members[at].1, depth);
                },
            );
        }
    }
}

fn is_container(value: &Value) -> bool {
    matches!(value, Value::Array(_) | Value::Object(_))
}

fn write_sequence(
    out: &mut String,
    depth: usize,
    wrap: bool,
    open: char,
    close: char,
    len: usize,
    mut item: impl FnMut(&mut String, usize, usize),
) {
    out.push(open);
    if len == 0 {
        out.push(close);
        return;
    }
    let inner = depth + 1;
    for at in 0..len {
        if at > 0 {
            out.push(',');
        }
        if wrap {
            newline(out, inner);
        }
        item(out, at, inner);
    }
    if wrap {
        newline(out, depth);
    }
    out.push(close);
}

fn newline(out: &mut String, depth: usize) {
    out.push('\n');
    for _ in 0..depth {
        out.push(' ');
    }
}

/// Writes a JSON string, escaping exactly what the library's writer escapes.
///
/// The two-character escapes and the control range are what a parser requires.
/// U+2028 and U+2029 are neither: they are valid JSON and invalid *JavaScript*,
/// and a profile is inlined into a script element by the bundled viewer — so a
/// symbolized profile that stopped escaping them would open in every reader
/// except the one this crate ships.
fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0}'..='\u{1f}' | '\u{2028}' | '\u{2029}' => {
                out.push_str(&format!("\\u{:04x}", character as u32));
            }
            _ => out.push(character),
        }
    }
    out.push('"');
}

/// Every key in the document, with how many times it appears.
///
/// Used by the round-trip test below and by nothing else: it is how a test says
/// "the same members are still there" without depending on their order or on
/// what this tool changed about their values.
#[cfg(test)]
pub fn key_census(value: &Value) -> std::collections::BTreeMap<String, usize> {
    fn walk(value: &Value, census: &mut std::collections::BTreeMap<String, usize>) {
        match value {
            Value::Object(members) => {
                for (key, value) in members {
                    *census.entry(key.clone()).or_default() += 1;
                    walk(value, census);
                }
            }
            Value::Array(items) => items.iter().for_each(|item| walk(item, census)),
            _ => {}
        }
    }
    let mut census = std::collections::BTreeMap::new();
    walk(value, &mut census);
    census
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(text: &str) -> String {
        render(&parse(text).unwrap_or_else(|error| panic!("{text}: {error}")))
    }

    /// **The property this parser exists for.** Parse, render, parse again, and
    /// the value is the same one — including members this tool has never heard
    /// of, in the order the file had them.
    #[test]
    fn a_document_survives_being_read_and_written() {
        let original = r#"{"format":"heapscope-profile","formatVersion":1,
            "unknownToThisTool":{"nested":[1,2,{"deep":true}]},
            "frames":[{"addr":"0x7fffdeadbeef1234","module":0}],
            "empty":{},"emptyList":[],"nothing":null,"awkward":"a\"b\\c\nd\u2028e"}"#;
        let first = parse(original).expect("the original parses");
        let second = parse(&render(&first)).expect("the rendering parses");
        assert_eq!(first, second);
    }

    /// Order is part of what is preserved. A map-backed parser reorders every
    /// object in the file, which turns a one-field addition into a whole-file
    /// diff.
    #[test]
    fn member_order_is_the_order_the_file_had() {
        let text = round_trip(r#"{"zebra":1,"apple":2,"middle":3}"#);
        assert!(
            text.find("zebra") < text.find("apple"),
            "members were reordered: {text}"
        );
        assert!(text.find("apple") < text.find("middle"), "{text}");
    }

    /// A number is text until somebody asks it to be a number. Reformatting one
    /// is how a counter at `u64::MAX` becomes a counter that is merely large.
    #[test]
    fn a_number_too_large_for_a_double_survives_exactly() {
        let text = round_trip(r#"{"totalBytes":18446744073709551615,"rate":1.5e-3}"#);
        assert!(text.contains("18446744073709551615"), "{text}");
        assert!(text.contains("1.5e-3"), "{text}");
    }

    #[test]
    fn addresses_come_back_as_the_numbers_they_name() {
        let value = parse(r#"{"addr":"0x7fffdeadbeef1234","plain":"2a"}"#).expect("parses");
        assert_eq!(
            value.get("addr").and_then(Value::as_address),
            Some(0x7fff_dead_beef_1234)
        );
        assert_eq!(value.get("plain").and_then(Value::as_address), Some(0x2a));
        assert_eq!(value.get("missing").and_then(Value::as_address), None);
    }

    /// A count that arrived as a fraction is a file to complain about, not one
    /// to round.
    #[test]
    fn a_fractional_count_is_not_a_count() {
        let value = parse(r#"{"a":1,"b":1.5,"c":"7"}"#).expect("parses");
        assert_eq!(value.get("a").and_then(Value::as_u64), Some(1));
        assert_eq!(value.get("b").and_then(Value::as_u64), None);
        assert_eq!(value.get("c").and_then(Value::as_u64), None);
    }

    /// A path or a symbol really can hold one, and JSON writes it as a pair of
    /// escapes that a naive reader turns into two replacement characters.
    #[test]
    fn a_character_above_the_basic_plane_survives_its_surrogate_pair() {
        let value = parse(r#"{"path":"/tmp/\ud83d\ude00/lib.so"}"#).expect("parses");
        assert_eq!(
            value.get("path").and_then(Value::as_str),
            Some("/tmp/😀/lib.so")
        );
        assert_eq!(parse(&render(&value)).expect("re-parses"), value);
    }

    #[test]
    fn setting_a_member_replaces_it_and_adding_one_appends() {
        let mut value = parse(r#"{"first":1,"second":2}"#).expect("parses");
        value.set("second", Value::number(9));
        value.set("third", Value::number(3));
        let Value::Object(members) = &value else {
            unreachable!()
        };
        assert_eq!(
            members
                .iter()
                .map(|(key, _)| key.as_str())
                .collect::<Vec<_>>(),
            ["first", "second", "third"]
        );
        assert_eq!(value.get("second").and_then(Value::as_u64), Some(9));
    }

    /// The escapes that are about JavaScript rather than about JSON. A profile
    /// is inlined into a script element by the bundled viewer.
    #[test]
    fn the_javascript_line_terminators_stay_escaped() {
        let text = round_trip("{\"name\":\"a\u{2028}b\u{2029}c\"}");
        assert!(text.contains(r"\u2028"), "{text}");
        assert!(text.contains(r"\u2029"), "{text}");
        assert!(!text.contains('\u{2028}'), "{text}");
    }

    #[test]
    fn malformed_documents_are_refused_with_an_offset() {
        for bad in [
            "{",
            "{\"a\"}",
            "{\"a\":}",
            "[1,]",
            "{\"a\":01x}",
            "\"unterminated",
            "{\"a\":1} trailing",
            "{\"a\":\"\\q\"}",
            "{\"a\":\"\\u00\"}",
        ] {
            assert!(parse(bad).is_err(), "`{bad}` parsed");
        }
    }

    /// The census the round-trip test leans on has to actually count.
    #[test]
    fn the_key_census_sees_every_level() {
        let value = parse(r#"{"a":1,"b":{"a":2},"c":[{"a":3},{"d":4}]}"#).expect("parses");
        let census = key_census(&value);
        assert_eq!(census.get("a"), Some(&3));
        assert_eq!(census.get("d"), Some(&1));
    }
}
