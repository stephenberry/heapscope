//! An independent statement of the rule `output::push_display` implements.
//!
//! Deliberately a second implementation rather than a call into the crate's own.
//! The screen decides what a reader is allowed to be shown, so a test that asked
//! the implementation what it does would agree with any answer it gave,
//! including a future one that quietly stops escaping something. Written from
//! the rule instead: control characters and the bidirectional formatting
//! characters become `\u{...}`, everything else is left alone.
//!
//! Costs a duplicated ten lines. Buys a test that fails when the rule changes,
//! which is the point — changing it should be a decision, not a diff.

#![allow(dead_code)]

/// What a string should look like once it has reached a profile.
pub fn screen(text: &str) -> String {
    let mut out = String::new();
    for character in text.chars() {
        if is_escaped(character) {
            out.push_str(&format!("\\u{{{:x}}}", character as u32));
        } else {
            out.push(character);
        }
    }
    out
}

/// Every character in `profile` that a terminal or a bidirectional renderer
/// would act on, as a reader would meet them.
///
/// Parses first, deliberately. The file's own layout newlines are the emitter's
/// and are not what this is looking for, and — more to the point — a JSON
/// ` ` escape is invisible in the raw bytes and arrives at the viewer as a
/// line separator all the same. What a reader sees is the decoded string, so
/// that is what gets checked.
pub fn offenders(profile: &str) -> Vec<char> {
    let parsed = super::json::parse(profile).expect("the profile should be valid JSON");
    let mut found = Vec::new();
    walk(&parsed, &mut found);
    found
}

fn walk(value: &super::json::Value, found: &mut Vec<char>) {
    use super::json::Value;
    match value {
        Value::String(text) => found.extend(text.chars().filter(|&c| is_escaped(c))),
        Value::Array(items) => items.iter().for_each(|item| walk(item, found)),
        Value::Object(members) => {
            for (key, item) in members {
                found.extend(key.chars().filter(|&c| is_escaped(c)));
                walk(item, found);
            }
        }
        _ => {}
    }
}

/// Whether the screen should have replaced `character` with an escape.
pub fn is_escaped(character: char) -> bool {
    let code = character as u32;
    // C0, DEL, C1.
    let control = code < 0x20 || (0x7F..=0x9F).contains(&code);
    // Bidirectional marks, embeddings, overrides, and isolates, plus the two
    // line separators.
    let reordering = matches!(
        code,
        0x061C | 0x200E | 0x200F | 0x202A..=0x202E | 0x2066..=0x2069 | 0x2028 | 0x2029
    );
    control || reordering
}
