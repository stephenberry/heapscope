//! Finding a compiled example from inside a test or bench binary.
//!
//! Several tests here drive a *separate* program rather than a function: the
//! lifecycle suite runs `lifecycle_probe` and reads the profile it leaves
//! behind, the cdylib test `dlopen`s `cdylib_probe`, and the overhead benchmark
//! times fixtures built as examples. All of them have to turn an example's name
//! into a path at runtime.
//!
//! The obvious way to do that is wrong, and was wrong here for four milestones.
//! Cargo puts examples in `<profile>/examples`, and each of these helpers used
//! to reach that directory by walking exactly two levels up from
//! `current_exe()` — `<profile>/deps/<binary>` to `<profile>`. That is not a
//! layout Cargo promises, only one it happened to have. Cargo 1.99.0-nightly
//! builds test binaries into `<profile>/build/<package>/<hash>/out/` instead,
//! which is five levels up, so `parent().parent()` lands in a hash directory
//! and every one of those tests fails on the fixture-missing assertion — with a
//! message blaming the user for not building the fixture they did build.
//!
//! It surfaced while bringing up sanitizers, which need a nightly toolchain.
//! Nothing about sanitizers caused it; nightly was merely the first thing to
//! ask. The comments those helpers carried claimed `current_exe` was the robust
//! choice *because* it survived `--release` and a custom `CARGO_TARGET_DIR` —
//! true, and beside the point, since the fragile part was never the prefix but
//! the fixed distance.
//!
//! So: search rather than assume. Walk up from this binary and take the first
//! ancestor that actually has the fixture under `examples/`. That is correct
//! for both layouts and for whatever the next one is, because it depends only
//! on the part Cargo does document — that examples live in `<profile>/examples`
//! and the binary doing the looking lives somewhere under `<profile>`.
//!
//! # `cargo test --all-targets` does not build these
//!
//! Plain `cargo test` builds examples as programs, so the fixtures are there.
//! `cargo test --all-targets` does not, and the difference is not the one the
//! name suggests: `--all-targets` *selects* examples explicitly, and an explicit
//! selection compiles each one as a **test target** instead. The artifact lands
//! at `<profile>/examples/lifecycle_probe-<hash>` and is never uplifted to the
//! plain name, and a `crate-type = ["cdylib"]` example comes out as an
//! executable rather than `libcdylib_probe.dylib`. Nothing here can run either.
//!
//! So on a cold cache `cargo test --all-targets` fails every test that drives a
//! fixture — 25 in `tests/lifecycle.rs`, all of `tests/cdylib_tls.rs` — on the
//! panic below, which is why that panic names the command. The fix is one build
//! ahead of the test run:
//!
//! ```text
//! cargo build --examples && cargo test --all-targets
//! ```
//!
//! `ci/sanitizers.sh` has always done this; `.github/workflows/ci.yml` did not,
//! and `tests/ci_workflow.rs` is what now keeps it from drifting back. Cargo's
//! `[[bench]]` note in `Cargo.toml` records the same rule from the other side:
//! an explicit selection overrides the `test` key, so `test = false` on the
//! examples would not help.

#![allow(dead_code)]

use std::path::PathBuf;

/// Path to the compiled example `name`, as an executable.
///
/// `remedy` is printed if it is not there, and should say how this particular
/// caller expects the fixture to have been built — the answer differs between a
/// test (where plain `cargo test`, but not `--all-targets`, builds examples for
/// you) and a bench (where nothing does).
pub fn example_binary(name: &str, remedy: &str) -> PathBuf {
    locate(&format!("{name}{}", std::env::consts::EXE_SUFFIX), remedy)
}

/// Path to the compiled example `name`, as a dynamic library.
///
/// A `crate-type = ["cdylib"]` example is still built into `<profile>/examples`,
/// but under the platform's library spelling: `libname.dylib`, `libname.so`,
/// `name.dll`.
pub fn example_library(name: &str, remedy: &str) -> PathBuf {
    locate(
        &format!(
            "{}{name}{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        ),
        remedy,
    )
}

/// The nearest ancestor of this binary with `examples/<file_name>` under it.
///
/// Panics with every directory it looked in. A fixture that was never built and
/// a fixture that moved produce the same symptom, and the list is what tells
/// them apart: a `<profile>/examples` in it that simply lacks the file means the
/// build step was skipped, while no plausible `<profile>` at all means the walk
/// is looking somewhere unexpected.
fn locate(file_name: &str, remedy: &str) -> PathBuf {
    let binary = std::env::current_exe().expect("this binary has a path");

    // `skip(1)` because the first ancestor is the binary itself. The walk ends
    // at the filesystem root, which is reached in a handful of steps and only
    // when the fixture is genuinely absent.
    //
    // Taking the *nearest* match is what keeps this off the source tree: the
    // checkout also has an `examples/` directory, but it holds `foo.rs`, never
    // `foo` or `libfoo.dylib`, so a built fixture is found long before the walk
    // could reach it and an unbuilt one is not falsely found at all.
    let mut searched = Vec::new();
    for ancestor in binary.ancestors().skip(1) {
        let candidate = ancestor.join("examples").join(file_name);
        if candidate.exists() {
            return candidate;
        }
        searched.push(candidate);
    }

    panic!(
        "the fixture `{file_name}` was not found, starting from {}.\n{remedy}\n\
         Looked in:\n{}",
        binary.display(),
        searched
            .iter()
            .map(|path| format!("    {}\n", path.display()))
            .collect::<String>()
    )
}
