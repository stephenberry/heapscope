//! Punycode, as the v0 mangling uses it.
//!
//! A linker symbol is `[A-Za-z0-9_]`, and a Rust identifier is any valid
//! identifier character, so the mangling needs a way to write one in the other.
//! v0 borrows the encoding IDNA uses for domain names (RFC 3492), which stores
//! the ASCII characters literally and then encodes the non-ASCII ones as a
//! sequence of deltas describing where to insert them:
//!
//! ```text
//! ünïcödé_name   ->   u18ncd_name_d1a1d7d6c
//!                      ^  ^^^^^^^^ ^^^^^^^^
//!                      |  literal  the four accented characters, as
//!                      |  ASCII    insertion positions and code points
//!                      byte length
//! ```
//!
//! One deviation from the RFC: the delimiter between the two halves is `_`
//! rather than `-`, because `-` cannot appear in a symbol.
//!
//! # Bounds
//!
//! Each delta inserts exactly one character, so the decoded length is bounded
//! by the encoded length and no allocation here can be provoked into growing.
//! Every arithmetic step is checked: the deltas are attacker-controlled, and
//! the RFC's own overflow guidance exists because they can be made to wrap.

/// Decodes `encoded` and appends the result to `out`.
///
/// Returns `false` if `encoded` is not valid punycode, leaving `out` untouched.
pub(super) fn decode(encoded: &str, out: &mut String) -> bool {
    let bytes = encoded.as_bytes();

    // The literal half is everything before the last delimiter. Splitting at
    // the last one rather than the first is what lets the literal half contain
    // delimiters of its own, which it routinely does: `ncd_name` above.
    let (literal, mut deltas) = match encoded.rfind('_') {
        Some(index) => (&bytes[..index], &bytes[index + 1..]),
        None => (&bytes[..0], bytes),
    };

    let mut decoded: Vec<char> = Vec::with_capacity(bytes.len());
    for &byte in literal {
        if !byte.is_ascii() {
            // The literal half is copied through verbatim, so a non-ASCII byte
            // in it means this was never punycode.
            return false;
        }
        decoded.push(byte as char);
    }

    let mut code_point: u32 = INITIAL_N;
    let mut index: u32 = 0;
    let mut bias: u32 = INITIAL_BIAS;

    while !deltas.is_empty() {
        let previous_index = index;
        let mut weight: u32 = 1;
        let mut threshold_scale: u32 = BASE;

        // One variable-length integer, little-endian, with a continuation rule
        // that depends on the running bias.
        loop {
            let Some((&byte, tail)) = deltas.split_first() else {
                return false;
            };
            deltas = tail;
            let Some(digit) = digit_value(byte) else {
                return false;
            };
            let Some(step) = digit.checked_mul(weight) else {
                return false;
            };
            let Some(next) = index.checked_add(step) else {
                return false;
            };
            index = next;

            let threshold = if threshold_scale <= bias {
                T_MIN
            } else if threshold_scale >= bias.saturating_add(T_MAX) {
                T_MAX
            } else {
                threshold_scale - bias
            };
            if digit < threshold {
                break;
            }
            let Some(next_weight) = weight.checked_mul(BASE - threshold) else {
                return false;
            };
            weight = next_weight;
            let Some(next_scale) = threshold_scale.checked_add(BASE) else {
                return false;
            };
            threshold_scale = next_scale;
        }

        // `decoded.len()` is bounded by the input length, so this cast is not
        // the lossy one it looks like.
        let positions = decoded.len() as u32 + 1;
        bias = adapt(index - previous_index, positions, previous_index == 0);
        let Some(next_code_point) = code_point.checked_add(index / positions) else {
            return false;
        };
        code_point = next_code_point;
        index %= positions;

        let Some(character) = char::from_u32(code_point) else {
            return false;
        };
        decoded.insert(index as usize, character);
        index += 1;
    }

    out.extend(decoded);
    true
}

const BASE: u32 = 36;
const T_MIN: u32 = 1;
const T_MAX: u32 = 26;
const SKEW: u32 = 38;
const DAMP: u32 = 700;
const INITIAL_BIAS: u32 = 72;
/// Decoding starts just past ASCII, since ASCII is carried literally.
const INITIAL_N: u32 = 128;

/// RFC 3492's bias adaptation, which keeps the encoding compact when successive
/// code points are close together.
fn adapt(mut delta: u32, positions: u32, first: bool) -> u32 {
    delta /= if first { DAMP } else { 2 };
    delta += delta / positions;
    let mut scale = 0;
    while delta > ((BASE - T_MIN) * T_MAX) / 2 {
        delta /= BASE - T_MIN;
        scale += BASE;
    }
    scale + (((BASE - T_MIN + 1) * delta) / (delta + SKEW))
}

/// `a`-`z` are 0-25 and `0`-`9` are 26-35.
///
/// RFC 3492 treats the two cases as equivalent, since a domain name may be
/// upper-cased in transit. A symbol may not: the mangler writes lower case and
/// nothing rewrites it in between, so an upper-case digit is evidence that this
/// is not the mangler's output rather than an alternative spelling of it.
fn digit_value(byte: u8) -> Option<u32> {
    match byte {
        b'a'..=b'z' => Some(u32::from(byte - b'a')),
        b'0'..=b'9' => Some(u32::from(byte - b'0') + 26),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(encoded: &str) -> Option<String> {
        let mut out = String::new();
        decode(encoded, &mut out).then_some(out)
    }

    /// Straight from a real symbol: `_RNvCs..._1pu18ncd_name_d1a1d7d6c`.
    #[test]
    fn a_real_identifier_from_a_real_binary_round_trips() {
        assert_eq!(
            decoded("ncd_name_d1a1d7d6c").as_deref(),
            Some("ünïcödé_name")
        );
    }

    /// The examples RFC 3492 itself publishes, which exercise insertion
    /// positions our own corpus never reaches.
    #[test]
    fn the_specifications_own_examples_decode() {
        // "Why can't they just speak Chinese?" in Chinese, and a Japanese
        // sentence: both are all-extended, so they have no delimiter.
        assert_eq!(
            decoded("ihqwcrb4cv8a8dqg056pqjye").as_deref(),
            Some("他们为什么不说中文")
        );
        // A mixed string, where the delimiter matters.
        assert_eq!(decoded("3B-ww4c5e180e575a65lsy2b").as_deref(), None);
        assert_eq!(
            decoded("3B_ww4c5e180e575a65lsy2b").as_deref(),
            Some("3年B組金八先生")
        );
    }

    #[test]
    fn an_all_ascii_identifier_needs_no_deltas() {
        assert_eq!(decoded("plain_").as_deref(), Some("plain"));
    }

    #[test]
    fn an_empty_encoding_decodes_to_nothing() {
        assert_eq!(decoded("").as_deref(), Some(""));
    }

    #[test]
    fn a_digit_outside_the_alphabet_is_refused() {
        assert_eq!(decoded("abc_$$$"), None);
        assert_eq!(decoded("abc_é"), None);
    }

    /// A truncated final delta leaves the decoder mid-integer, which must be a
    /// refusal rather than whatever the partial value happened to be.
    #[test]
    fn a_truncated_delta_is_refused() {
        // A digit at or above the threshold means "more digits follow", and at
        // the starting bias the threshold is 1, so a lone `z` demands a
        // continuation that is not there.
        assert_eq!(decoded("a_z"), None);
        assert_eq!(decoded("a_zz"), None);
        // Three is enough to reach the point where the threshold catches up,
        // so this one is well formed and must not be refused.
        assert!(decoded("a_zzz").is_some());
    }

    #[test]
    fn a_delta_that_overflows_is_refused_rather_than_wrapped() {
        assert_eq!(decoded("a_99999999999999"), None);
    }

    /// A code point past the Unicode range, or in the surrogate hole, has no
    /// character to insert.
    #[test]
    fn a_delta_naming_no_character_is_refused() {
        // Long runs of `9` push the code point far past U+10FFFF.
        assert_eq!(decoded("_999999999"), None);
    }

    /// The decoder allocates, so the relationship between input size and
    /// output size is a property worth pinning rather than assuming.
    #[test]
    fn the_decoded_length_never_exceeds_the_encoded_length() {
        for encoded in ["ncd_name_d1a1d7d6c", "ihqwcrb4cv8a8dqg056pqjye", "a_a"] {
            let out = decoded(encoded).expect(encoded);
            assert!(
                out.chars().count() <= encoded.len(),
                "{encoded} decoded to {} characters",
                out.chars().count()
            );
        }
    }

    /// Exhaustive over an alphabet picked to hit every branch: a literal byte,
    /// the delimiter, digits at both ends of the range, and a byte that is not
    /// in the alphabet at all.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "exhaustive over ~50,000 inputs; nothing here is unsafe"
    )]
    fn no_input_of_any_shape_panics() {
        const ALPHABET: &[u8] = b"az09_A-";
        let mut buffer = String::new();
        for length in 0..=4u32 {
            let combinations = (ALPHABET.len() as u64).pow(length);
            for mut n in 0..combinations {
                let candidate: String = (0..length)
                    .map(|_| {
                        let byte = ALPHABET[(n % ALPHABET.len() as u64) as usize];
                        n /= ALPHABET.len() as u64;
                        byte as char
                    })
                    .collect();
                buffer.clear();
                if decode(&candidate, &mut buffer) {
                    assert!(
                        buffer.chars().count() <= candidate.len(),
                        "{candidate:?} grew"
                    );
                }
            }
        }
    }
}
