//! What a hand-written benchmark owes `cargo test`.
//!
//! `cargo test --all-targets` — which is what CI's test job runs on all four
//! platforms — selects bench targets and *runs* them. A `harness = false`
//! benchmark is just a `main`, so unless it says otherwise it runs its whole
//! measurement, in a **debug** build, as part of the test suite.
//!
//! What that cost was not only minutes per platform. `benches/overhead.rs`
//! checks that each configuration's profile names the function that allocated,
//! and `dhat-rs, 5 frames` cannot reach it in an unoptimized build — the frames
//! debug leaves in push the target past a five-frame capture. So the driver
//! failed, correctly, and took `cargo test --all-targets` to exit 101 with it.
//! The check is sound and a five-frame capture of unoptimized code is
//! meaningless, so the answer is not to weaken the check: it is for a benchmark
//! to know when it is not being asked to measure anything.
//!
//! # The flag is `--bench`, and its absence is the signal
//!
//! Cargo tells a `harness = false` target which of the two invocations it is
//! under by passing **`--bench` under `cargo bench`, and nothing at all under
//! `cargo test`** — measured, both ways, rather than assumed:
//!
//! ```text
//! cargo bench --bench overhead   ->  ["…/overhead-065f5658", "--bench"]
//! cargo test --all-targets       ->  ["…/overhead-f59a16d5"]
//! ```
//!
//! This is criterion's protocol too, which is why the two criterion benchmarks
//! here have always been fine under `cargo test` and these two were not. It
//! reads backwards — the *presence* of a flag is what licenses the measurement —
//! and it has to, because the test invocation passes nothing to recognise.
//!
//! One consequence follows from that and is worth stating rather than
//! discovering: running a bench binary straight off the path looks exactly like
//! the test invocation, so it refuses too. The message names the flag.
//!
//! `test = false` in `Cargo.toml` does not do any of this. It sets a default,
//! and `--all-targets` selects bench targets explicitly, which overrides it.
//!
//! Nothing is lost by returning early. `--all-targets` still *builds* both
//! drivers, so a benchmark that stops compiling still fails CI, and that was the
//! only signal running them ever produced.

/// True when nothing asked this benchmark to measure anything.
///
/// The caller should return immediately, printing nothing that reads like a
/// result.
pub fn run_as_a_test() -> bool {
    if std::env::args().any(|argument| argument == "--bench") {
        return false;
    }
    println!(
        "this is a benchmark, not a test: run it with `cargo bench`, or pass \
         `--bench` to run it directly"
    );
    true
}
