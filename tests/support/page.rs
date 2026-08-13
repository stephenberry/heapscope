//! Reading the two JSON blocks back out of a bundled viewer page.
//!
//! `tests/html_output.rs` asks whether the page is well formed; `tests/profile_fuzz.rs`
//! asks whether generated text survives the trip into it. Both have to find the
//! same two `<script>` elements first, and both had their own copy of this — the
//! same twelve lines and the same two markers, which is two places for the
//! markers to be updated and one of them to be missed.
//!
//! Stated on the test side rather than imported from `heapscope::output::html`,
//! which is the same rule `support/display.rs` follows: a test that asked the
//! emitter which element it writes would agree with any answer the emitter gave,
//! including a future one that renamed the element the bundled viewer looks for.

#![allow(dead_code)]

/// Where the profile and the sidecar sit in the page.
pub const PROFILE_BLOCK: &str = r#"<script type="application/json" id="heapscope-profile">"#;
pub const DISPLAY_BLOCK: &str = r#"<script type="application/json" id="heapscope-display">"#;

/// The text of the block opened by `tag`, exactly as the page carries it.
///
/// Not unescaped. `\u003c` is valid JSON for `<`, so the block parses as it
/// stands, and undoing the escape by text substitution would corrupt a profile
/// that legitimately contains the six characters `\u003c` — which a symbol or a
/// path really can, and which `tests/profile_fuzz.rs` generates on purpose. Use
/// [`parsed`] where the decoded text is what is wanted.
pub fn block<'a>(page: &'a str, tag: &str) -> &'a str {
    let start = page
        .find(tag)
        .unwrap_or_else(|| panic!("the page has no {tag} block"))
        + tag.len();
    let end = start
        + page[start..]
            .find("</script>")
            .expect("the block is never closed");
    &page[start..end]
}

/// The block, with the one escape the page applies undone, parsed.
///
/// **Not for a caller whose profile can legitimately contain the six characters
/// `\u003c`** — which `tests/profile_fuzz.rs` generates on purpose, and which
/// this would silently turn into a `<`. That suite takes [`block`] and parses it
/// as it stands; this one is for the suites whose text they chose themselves.
pub fn parsed(page: &str, tag: &str) -> super::json::Value {
    let text = block(page, tag).replace(r"\u003c", "<");
    super::json::parse(&text).unwrap_or_else(|error| panic!("{tag} is not valid JSON: {error:?}"))
}
