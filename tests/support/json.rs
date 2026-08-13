//! A strict JSON parser, for checking what the profiler writes.
//!
//! Test-only code, and deliberately not the crate's writer read backwards: a
//! parser built to mirror a writer shares the writer's blind spots, and the
//! whole point is to catch the writer being wrong. This one is written against
//! RFC 8259 and refuses everything the grammar refuses — trailing commas, raw
//! control characters inside strings, leading zeros, `NaN`, trailing junk after
//! the document. Several of those are things `serde_json` accepts or rejects by
//! configuration; here they are always errors, because a profile that only parses
//! in a lenient parser is a profile that will not open in the viewer.
//!
//! Duplicate object keys are rejected too. JSON permits them and JavaScript
//! quietly keeps the last, which would let the writer emit `"tb"` twice with
//! different values and still look correct.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fmt;

/// A parsed JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// Kept as written, so that integers stay exact rather than passing through
    /// an `f64` that silently rounds anything past 2^53.
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

impl Value {
    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Object(map) => Some(map),
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

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// The value as a `u64`, if it is a number with no fraction or exponent.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Number(raw) => raw.parse().ok(),
            _ => None,
        }
    }

    /// The named member of an object.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object()?.get(key)
    }

    /// A short name for the kind of value this is, for error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }
}

/// Where in the input a parse failed, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub at: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at byte {}: {}", self.at, self.message)
    }
}

/// Parses a complete JSON document.
pub fn parse(input: &str) -> Result<Value, ParseError> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        at: 0,
    };
    parser.skip_whitespace();
    let value = parser.value(0)?;
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

/// Deep enough for any profile, shallow enough that a hostile file cannot
/// exhaust the stack of the test that reads it.
const MAX_DEPTH: usize = 128;

impl Parser<'_> {
    fn error(&self, message: impl Into<String>) -> ParseError {
        ParseError {
            at: self.at,
            message: message.into(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(byte) = self.peek() {
            // The four the grammar allows, and no others: a vertical tab or a
            // form feed between tokens is not whitespace to a JSON parser.
            if matches!(byte, b' ' | b'\t' | b'\n' | b'\r') {
                self.at += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), ParseError> {
        if self.peek() == Some(byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(self.error(format!("expected `{}`", byte as char)))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, ParseError> {
        if depth > MAX_DEPTH {
            return Err(self.error("nested too deeply"));
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(byte) => Err(self.error(format!("unexpected byte `{}`", byte as char))),
            None => Err(self.error("unexpected end of input")),
        }
    }

    fn literal(&mut self, text: &str, value: Value) -> Result<Value, ParseError> {
        if self.bytes[self.at..].starts_with(text.as_bytes()) {
            self.at += text.len();
            Ok(value)
        } else {
            Err(self.error(format!("expected `{text}`")))
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.expect(b'{')?;
        let mut members = BTreeMap::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Value::Object(members));
        }
        loop {
            self.skip_whitespace();
            let at = self.at;
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            let value = self.value(depth + 1)?;
            if members.insert(key.clone(), value).is_some() {
                self.at = at;
                return Err(self.error(format!("duplicate key `{key}`")));
            }
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

    fn array(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_whitespace();
            items.push(self.value(depth + 1)?);
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

    fn string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"')?;
        let mut text = String::new();
        loop {
            let byte = self
                .peek()
                .ok_or_else(|| self.error("unterminated string"))?;
            match byte {
                b'"' => {
                    self.at += 1;
                    return Ok(text);
                }
                b'\\' => {
                    self.at += 1;
                    let escape = self
                        .peek()
                        .ok_or_else(|| self.error("unterminated escape"))?;
                    self.at += 1;
                    let decoded = match escape {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => self.unicode_escape()?,
                        other => {
                            return Err(self.error(format!("invalid escape `\\{}`", other as char)))
                        }
                    };
                    text.push(decoded);
                }
                0x00..=0x1F => {
                    return Err(self.error(format!(
                        "unescaped control character U+{byte:04X} in a string"
                    )))
                }
                _ => {
                    // A UTF-8 sequence: find its end and copy it whole.
                    let start = self.at;
                    self.at += 1;
                    while self.peek().is_some_and(|next| (0x80..0xC0).contains(&next)) {
                        self.at += 1;
                    }
                    let slice = std::str::from_utf8(&self.bytes[start..self.at])
                        .map_err(|_| self.error("invalid UTF-8 in a string"))?;
                    text.push_str(slice);
                }
            }
        }
    }

    fn unicode_escape(&mut self) -> Result<char, ParseError> {
        let first = self.hex4()?;
        // A high surrogate must be followed by its low half; a lone surrogate is
        // not a character and the parser refuses to invent a replacement.
        if (0xD800..0xDC00).contains(&first) {
            if !self.bytes[self.at..].starts_with(b"\\u") {
                return Err(self.error("high surrogate with no low surrogate"));
            }
            self.at += 2;
            let second = self.hex4()?;
            if !(0xDC00..0xE000).contains(&second) {
                return Err(self.error("high surrogate followed by a non-surrogate"));
            }
            let combined = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00);
            return char::from_u32(combined).ok_or_else(|| self.error("invalid surrogate pair"));
        }
        if (0xDC00..0xE000).contains(&first) {
            return Err(self.error("low surrogate with no high surrogate"));
        }
        char::from_u32(first).ok_or_else(|| self.error("invalid code point"))
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let mut value = 0u32;
        for _ in 0..4 {
            let byte = self.peek().ok_or_else(|| self.error("short \\u escape"))?;
            let digit = (byte as char)
                .to_digit(16)
                .ok_or_else(|| self.error("non-hexadecimal digit in a \\u escape"))?;
            value = value * 16 + digit;
            self.at += 1;
        }
        Ok(value)
    }

    fn number(&mut self) -> Result<Value, ParseError> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.at += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.error("a number may not have a leading zero"));
                }
            }
            Some(b'1'..=b'9') => self.digits(),
            _ => return Err(self.error("expected a digit")),
        }
        if self.peek() == Some(b'.') {
            self.at += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected a digit after the decimal point"));
            }
            self.digits();
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected a digit in the exponent"));
            }
            self.digits();
        }
        let raw = std::str::from_utf8(&self.bytes[start..self.at])
            .expect("digits are ASCII")
            .to_string();
        Ok(Value::Number(raw))
    }

    fn digits(&mut self) {
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.at += 1;
        }
    }
}
