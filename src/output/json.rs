//! A strict JSON writer, hand-rolled because the crate ships no dependencies.
//!
//! It is a *writer*, not a serializer: there is no reflection, no derive, and no
//! value tree. Callers drive it with `begin_object` / `key` / `u64` / `end_object`
//! and it tracks just enough state to place commas, indent, and escape strings
//! correctly.
//!
//! # What "strict" means here
//!
//! Output is always valid RFC 8259 JSON for any input a caller can supply:
//!
//! - Every `str` is escaped, including the C0 control characters that a JSON
//!   parser must reject when they appear raw.
//! - `U+2028` and `U+2029` are escaped as well. JSON does not require it — but
//!   both are line terminators in JavaScript, so a profile containing one in a
//!   file path would be valid JSON that breaks the moment it is embedded in a
//!   script element, which is exactly what the bundled viewer will do.
//! - Rust `str` is UTF-8 by construction, so there is no lone-surrogate case to
//!   handle. That is a guarantee of the input type, not something checked here.
//!
//! Escaping stops there. In particular `<` is *not* escaped: the HTML emitter is
//! responsible for escaping in an HTML context, because doing it here would turn
//! every `Vec<u8>` in a symbolized frame into `Vec<u8>` in a file people
//! read.
//!
//! # Structural misuse
//!
//! Writing a key outside an object, or ending a container that was never begun,
//! is a bug in the caller rather than a runtime condition, so it trips a
//! `debug_assert!`. In release builds the writer does the most reasonable thing
//! it can and keeps going; it never panics from a path that a profile write
//! could reach in production.

use std::io::{self, Write};

/// How a container's members are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layout {
    /// Every member on one line: `{"tb":5,"tbk":1}`.
    ///
    /// Used for the innermost records, where one line per record keeps a large
    /// profile diffable without making it enormous.
    Inline,
    /// One member per line, indented one space per level of nesting.
    Wrap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

#[derive(Clone, Copy, Debug)]
struct Level {
    container: Container,
    layout: Layout,
    /// Whether a member has already been written at this level, which is what
    /// decides between "separator" and "no separator".
    populated: bool,
}

/// Writes JSON to an `io::Write` sink.
#[derive(Debug)]
pub(super) struct JsonWriter<W> {
    out: W,
    stack: Vec<Level>,
    /// Set between `key` and the value it introduces, so that the value does
    /// not emit a separator of its own.
    expecting_value: bool,
}

impl<W: Write> JsonWriter<W> {
    /// Creates a writer that emits to `out`.
    ///
    /// `out` is written to directly; wrap it in a [`std::io::BufWriter`] if it
    /// is a file, because the writer makes many small writes.
    pub(super) fn new(out: W) -> Self {
        Self {
            out,
            stack: Vec::new(),
            expecting_value: false,
        }
    }

    /// Opens an object.
    pub(super) fn begin_object(&mut self, layout: Layout) -> io::Result<()> {
        self.before_value()?;
        self.out.write_all(b"{")?;
        self.push(Container::Object, layout);
        Ok(())
    }

    /// Closes the innermost object.
    pub(super) fn end_object(&mut self) -> io::Result<()> {
        self.end(Container::Object, b"}")
    }

    /// Opens an array.
    pub(super) fn begin_array(&mut self, layout: Layout) -> io::Result<()> {
        self.before_value()?;
        self.out.write_all(b"[")?;
        self.push(Container::Array, layout);
        Ok(())
    }

    /// Closes the innermost array.
    pub(super) fn end_array(&mut self) -> io::Result<()> {
        self.end(Container::Array, b"]")
    }

    /// Writes an object member name. The next call must write its value.
    pub(super) fn key(&mut self, key: &str) -> io::Result<()> {
        debug_assert!(
            matches!(self.stack.last(), Some(l) if l.container == Container::Object),
            "a JSON key is only meaningful inside an object"
        );
        debug_assert!(!self.expecting_value, "two keys in a row: {key}");
        self.before_value()?;
        self.write_string(key)?;
        self.out.write_all(b":")?;
        self.expecting_value = true;
        Ok(())
    }

    /// Writes a string value.
    pub(super) fn string(&mut self, value: &str) -> io::Result<()> {
        self.before_value()?;
        self.write_string(value)
    }

    /// Writes an unsigned integer value.
    pub(super) fn u64(&mut self, value: u64) -> io::Result<()> {
        self.before_value()?;
        let mut buffer = itoa::Buffer::new();
        self.out.write_all(buffer.format(value).as_bytes())
    }

    /// Writes a boolean value.
    pub(super) fn bool(&mut self, value: bool) -> io::Result<()> {
        self.before_value()?;
        self.out.write_all(if value { b"true" } else { b"false" })
    }

    /// Writes an object member whose value is an unsigned integer.
    pub(super) fn field_u64(&mut self, key: &str, value: u64) -> io::Result<()> {
        self.key(key)?;
        self.u64(value)
    }

    /// Writes an object member whose value is a string.
    pub(super) fn field_str(&mut self, key: &str, value: &str) -> io::Result<()> {
        self.key(key)?;
        self.string(value)
    }

    /// Writes an object member whose value is a boolean.
    pub(super) fn field_bool(&mut self, key: &str, value: bool) -> io::Result<()> {
        self.key(key)?;
        self.bool(value)
    }

    /// Terminates the document with a newline and returns the sink.
    ///
    /// Every container must have been closed.
    pub(super) fn finish(mut self) -> io::Result<W> {
        debug_assert!(
            self.stack.is_empty(),
            "{} unclosed JSON container(s)",
            self.stack.len()
        );
        self.out.write_all(b"\n")?;
        self.out.flush()?;
        Ok(self.out)
    }

    fn push(&mut self, container: Container, layout: Layout) {
        self.stack.push(Level {
            container,
            layout,
            populated: false,
        });
    }

    fn end(&mut self, expected: Container, close: &[u8]) -> io::Result<()> {
        let Some(level) = self.stack.pop() else {
            debug_assert!(false, "closed a JSON container that was never opened");
            return Ok(());
        };
        debug_assert_eq!(level.container, expected, "mismatched JSON container close");
        debug_assert!(
            !self.expecting_value,
            "a JSON container was closed with a key still awaiting its value"
        );
        if level.layout == Layout::Wrap && level.populated {
            self.newline()?;
        }
        self.out.write_all(close)?;
        Ok(())
    }

    /// Emits whatever separates this value from the previous one.
    fn before_value(&mut self) -> io::Result<()> {
        if self.expecting_value {
            self.expecting_value = false;
            return Ok(());
        }
        let Some(level) = self.stack.last_mut() else {
            return Ok(());
        };
        let populated = std::mem::replace(&mut level.populated, true);
        let layout = level.layout;
        if populated {
            self.out.write_all(b",")?;
        }
        if layout == Layout::Wrap {
            self.newline()?;
        }
        Ok(())
    }

    /// A newline followed by one space per level of nesting.
    fn newline(&mut self) -> io::Result<()> {
        const SPACES: &[u8; 32] = b"                                ";
        self.out.write_all(b"\n")?;
        let mut remaining = self.stack.len();
        while remaining > 0 {
            let chunk = remaining.min(SPACES.len());
            self.out.write_all(&SPACES[..chunk])?;
            remaining -= chunk;
        }
        Ok(())
    }

    fn write_string(&mut self, value: &str) -> io::Result<()> {
        self.out.write_all(b"\"")?;
        // Written in runs: everything between two characters that need escaping
        // goes out in one call, because the sink may be a file.
        let bytes = value.as_bytes();
        let mut run_start = 0;
        let mut index = 0;
        while index < bytes.len() {
            let byte = bytes[index];
            let escape = match byte {
                b'"' => Escape::Short(b'"'),
                b'\\' => Escape::Short(b'\\'),
                0x08 => Escape::Short(b'b'),
                0x09 => Escape::Short(b't'),
                0x0A => Escape::Short(b'n'),
                0x0C => Escape::Short(b'f'),
                0x0D => Escape::Short(b'r'),
                0x00..=0x1F => Escape::Unicode(u32::from(byte)),
                // U+2028 and U+2029 are `e2 80 a8` and `e2 80 a9` in UTF-8.
                0xE2 if bytes[index..].starts_with(&[0xE2, 0x80, 0xA8]) => Escape::Unicode(0x2028),
                0xE2 if bytes[index..].starts_with(&[0xE2, 0x80, 0xA9]) => Escape::Unicode(0x2029),
                _ => {
                    index += 1;
                    continue;
                }
            };

            self.out.write_all(&bytes[run_start..index])?;
            match escape {
                Escape::Short(c) => {
                    self.out.write_all(&[b'\\', c])?;
                    index += 1;
                }
                Escape::Unicode(code) => {
                    let mut buffer = *b"\\u0000";
                    for (offset, shift) in [(2, 12), (3, 8), (4, 4), (5, 0)] {
                        buffer[offset] = HEX[((code >> shift) & 0xF) as usize];
                    }
                    self.out.write_all(&buffer)?;
                    index += if code < 0x80 { 1 } else { 3 };
                }
            }
            run_start = index;
        }
        self.out.write_all(&bytes[run_start..])?;
        self.out.write_all(b"\"")
    }
}

enum Escape {
    /// A two-character escape such as `\n`.
    Short(u8),
    /// A `\uXXXX` escape for the given code point.
    Unicode(u32),
}

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Formatting for `u64` without going through `format!`.
///
/// `write!` on an `io::Write` is not free, and a profile with a million program
/// points writes on the order of ten million integers. This is the same trick
/// the `itoa` crate uses, in the twenty lines of it that we need.
mod itoa {
    /// A stack buffer that a `u64` is formatted into.
    pub(super) struct Buffer {
        /// 20 digits is the width of `u64::MAX`.
        bytes: [u8; 20],
    }

    impl Buffer {
        pub(super) fn new() -> Self {
            Self { bytes: [0; 20] }
        }

        /// Formats `value` into the buffer and returns it as a string.
        pub(super) fn format(&mut self, value: u64) -> &str {
            let mut cursor = self.bytes.len();
            let mut remaining = value;
            loop {
                cursor -= 1;
                self.bytes[cursor] = b'0' + (remaining % 10) as u8;
                remaining /= 10;
                if remaining == 0 {
                    break;
                }
            }
            // SAFETY: every byte written is an ASCII digit.
            unsafe { std::str::from_utf8_unchecked(&self.bytes[cursor..]) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(f: impl FnOnce(&mut JsonWriter<Vec<u8>>) -> io::Result<()>) -> String {
        let mut writer = JsonWriter::new(Vec::new());
        f(&mut writer).expect("writing to a Vec cannot fail");
        String::from_utf8(writer.finish().expect("finish")).expect("valid UTF-8")
    }

    #[test]
    fn an_inline_object_has_no_whitespace() {
        let json = write(|w| {
            w.begin_object(Layout::Inline)?;
            w.field_u64("tb", 5)?;
            w.field_u64("tbk", 1)?;
            w.end_object()
        });
        assert_eq!(json, "{\"tb\":5,\"tbk\":1}\n");
    }

    #[test]
    fn a_wrapped_object_puts_each_member_on_its_own_line() {
        let json = write(|w| {
            w.begin_object(Layout::Wrap)?;
            w.field_u64("a", 1)?;
            w.field_bool("b", true)?;
            w.end_object()
        });
        assert_eq!(json, "{\n \"a\":1,\n \"b\":true\n}\n");
    }

    #[test]
    fn an_empty_container_stays_on_one_line() {
        // A `Wrap` container with nothing in it would otherwise emit a newline
        // before its closing brace, which reads as a formatting bug.
        assert_eq!(
            write(|w| {
                w.begin_array(Layout::Wrap)?;
                w.end_array()
            }),
            "[]\n"
        );
    }

    #[test]
    fn nesting_indents_by_depth() {
        let json = write(|w| {
            w.begin_object(Layout::Wrap)?;
            w.key("pps")?;
            w.begin_array(Layout::Wrap)?;
            w.begin_object(Layout::Inline)?;
            w.field_u64("tb", 1)?;
            w.end_object()?;
            w.end_array()?;
            w.end_object()
        });
        assert_eq!(json, "{\n \"pps\":[\n  {\"tb\":1}\n ]\n}\n");
    }

    #[test]
    fn indentation_survives_nesting_deeper_than_it_is_written_in() {
        // `newline` emits one space per level from a 32-byte constant, a chunk
        // at a time. No emitter in this crate nests past four levels and no
        // other test past three, so the second iteration of that loop has never
        // run: an indent that silently stopped at 32 spaces would look correct
        // everywhere it is currently reached.
        //
        // Not a fuzz target, because structural drive is out of this writer's
        // contract by design -- misuse trips a `debug_assert!` -- so a campaign
        // over arbitrary call sequences would report the assertions rather than
        // find anything. `fuzz/fuzz_targets/profile.rs` covers the reachable
        // surface, which is well-formed structure and arbitrary text.
        const DEPTH: usize = 40;
        let json = write(|w| {
            for _ in 0..DEPTH {
                w.begin_array(Layout::Wrap)?;
            }
            w.u64(1)?;
            for _ in 0..DEPTH {
                w.end_array()?;
            }
            Ok(())
        });

        let value = json
            .lines()
            .find(|line| line.trim_start() == "1")
            .expect("the value is on a line of its own");
        assert_eq!(
            value.len() - value.trim_start().len(),
            DEPTH,
            "a value {DEPTH} levels down is indented by {} spaces:\n{json}",
            value.len() - value.trim_start().len()
        );
        // And the brackets step back out one level at a time on the way up,
        // which is the same code called with the level already popped.
        assert!(
            json.contains(&format!("\n{}]", " ".repeat(DEPTH - 1))),
            "the innermost bracket is not one level in from its value:\n{json}"
        );
    }

    #[test]
    fn strings_escape_everything_a_parser_would_reject() {
        let json = write(|w| w.string("a\"b\\c\nd\te\u{0}f\u{1f}g"));
        assert_eq!(json, "\"a\\\"b\\\\c\\nd\\te\\u0000f\\u001fg\"\n");
    }

    #[test]
    fn strings_escape_the_javascript_line_terminators() {
        // Valid JSON either way; invalid JavaScript unescaped, and the bundled
        // viewer will inline a profile into a script element.
        let json = write(|w| w.string("a\u{2028}b\u{2029}c"));
        assert_eq!(json, "\"a\\u2028b\\u2029c\"\n");
    }

    #[test]
    fn other_multibyte_characters_pass_through_unescaped() {
        // `µ` is the time unit in `Monotonic` mode, so this is a real case and
        // not a hypothetical one. It also proves the U+2028 detection does not
        // fire on every three-byte sequence that starts with 0xE2.
        let json = write(|w| w.string("µs \u{2026} \u{1F600}"));
        assert_eq!(json, "\"µs \u{2026} \u{1F600}\"\n");
    }

    #[test]
    fn all_control_characters_round_trip_through_a_strict_parser() {
        // Every code point below 0x20 must come out as something a parser
        // accepts, whichever escape form it takes.
        let raw: String = (0u32..0x20).filter_map(char::from_u32).collect();
        let json = write(|w| w.string(&raw));
        assert!(!json.contains(|c: char| (c as u32) < 0x20 && c != '\n'));
        assert_eq!(json.matches("\\u00").count(), 0x20 - 5);
    }

    #[test]
    fn integers_are_formatted_without_allocating() {
        assert_eq!(write(|w| w.u64(0)), "0\n");
        assert_eq!(write(|w| w.u64(u64::MAX)), "18446744073709551615\n");
        assert_eq!(write(|w| w.u64(1_000_000)), "1000000\n");
    }

    #[test]
    fn integer_formatting_agrees_with_the_standard_library() {
        let mut buffer = itoa::Buffer::new();
        for value in [
            0,
            1,
            9,
            10,
            99,
            100,
            u64::from(u32::MAX),
            u64::MAX / 3,
            u64::MAX - 1,
            u64::MAX,
        ] {
            assert_eq!(buffer.format(value), value.to_string());
        }
    }

    #[test]
    fn a_write_error_is_reported_rather_than_swallowed() {
        struct Full;
        impl Write for Full {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::WriteZero, "full"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let mut writer = JsonWriter::new(Full);
        let result = writer
            .begin_object(Layout::Inline)
            .and_then(|()| writer.field_u64("a", 1));
        assert!(result.is_err(), "a failing sink must surface its error");
    }
}
