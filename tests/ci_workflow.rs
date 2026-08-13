//! The one promise the continuous integration configuration makes to the tests.
//!
//! Several suites here drive a compiled example as a *program*:
//! `tests/lifecycle.rs` runs `lifecycle_probe` twenty-five times and reads the
//! profiles it leaves behind, and `tests/cdylib_tls.rs` `dlopen`s
//! `cdylib_probe`. Both need the example built the ordinary way, and neither can
//! build it itself.
//!
//! `cargo test --all-targets` does not build them that way. It selects examples
//! explicitly, and an explicit selection compiles each one as a *test target*
//! instead: the artifact lands at `<profile>/examples/lifecycle_probe-<hash>`
//! and is never uplifted to the plain name, and a `crate-type = ["cdylib"]`
//! example comes out as an executable rather than a shared library. So on a cold
//! cache the exact command every `Test (…)` job ran failed twenty-six tests on a
//! missing-fixture panic — a message that reads like a defect in this crate and
//! was a defect in the workflow. `tests/support/fixture.rs` documents the
//! mechanism; this file is what stops it coming back.
//!
//! The rule is one line long: anything that runs `cargo test --all-targets` has
//! to build the examples first. `ci/sanitizers.sh` has always obeyed it. This
//! checks that everything else does too, because a rule nobody checks is a
//! comment, and this one was a comment for four milestones.

use std::path::{Path, PathBuf};

/// The workflow, relative to the repository root.
///
/// It is parsed per job, since ordering only means anything within one. The
/// scripts under `ci/` are checked too, but read straight through: source order
/// is not execution order in a shell script — a function can be defined above
/// its caller and called below it — so that half is a check which can miss, not
/// one which can fire wrongly. Worth having anyway, because the arrangement it
/// does catch, a build step added after the test step it was meant to precede,
/// is the one a hurried edit produces.
const WORKFLOW: &str = ".github/workflows/ci.yml";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Command lines with YAML and shell comments removed.
///
/// Comments matter here: this repository explains the `--all-targets` trap in
/// prose next to the steps that avoid it, so a scan that did not strip comments
/// would read those explanations as the commands they describe.
fn commands(text: &str) -> Vec<&str> {
    text.lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect()
}

/// Splits a workflow into `(job_name, command_lines)`.
///
/// Hand-rolled rather than parsed: the file is ours, so the input is not
/// adversarial, and the only structure needed is where one job ends and the next
/// begins. A job is a two-space-indented key inside the top-level `jobs:` block.
fn jobs(workflow: &str) -> Vec<(String, Vec<&str>)> {
    let mut found: Vec<(String, Vec<&str>)> = Vec::new();
    let mut in_jobs = false;
    for line in commands(workflow) {
        if !line.starts_with(char::is_whitespace) && !line.trim().is_empty() {
            in_jobs = line.trim_end() == "jobs:";
            continue;
        }
        if !in_jobs {
            continue;
        }
        let is_job_header = line.starts_with("  ")
            && !line.starts_with("   ")
            && line.trim_end().ends_with(':')
            && !line.trim_start().starts_with('-');
        if is_job_header {
            let name = line.trim().trim_end_matches(':').to_string();
            found.push((name, Vec::new()));
        } else if let Some((_, body)) = found.last_mut() {
            body.push(line);
        }
    }
    found
}

fn runs_all_targets_test(line: &str) -> bool {
    line.contains("cargo test") && line.contains("--all-targets")
}

fn builds_examples(line: &str) -> bool {
    line.contains("cargo build") && line.contains("--examples")
}

/// Nothing runs `cargo test --all-targets` without building the examples first.
///
/// The count at the end is not decoration. Every assertion in this file is
/// inside a loop over things that match a pattern, and a rename that stopped
/// anything from matching would leave a test that passes by looping zero times —
/// which is how `ci/dhat-viewer-check.mjs` sat unrun for a milestone. If the
/// workflow ever legitimately stops using `--all-targets`, this file should be
/// deleted rather than left to pass vacuously.
#[test]
#[cfg_attr(
    miri,
    ignore = "reads the repository from disk; Miri isolation blocks file I/O"
)]
fn every_all_targets_test_run_builds_the_examples_first() {
    let root = repo_root();
    if !root.join(".github").is_dir() {
        // `Cargo.toml` excludes `/.github` from the published package, so an
        // unpacked `.crate` genuinely has nothing to check. Anything that is a
        // checkout has the directory, and then the file below must be there.
        eprintln!("no .github directory, so this is not a checkout; nothing to check");
        return;
    }

    let mut checked = 0;
    for (job, body) in jobs(&read(&root.join(WORKFLOW))) {
        for (index, line) in body.iter().enumerate() {
            if !runs_all_targets_test(line) {
                continue;
            }
            checked += 1;
            assert!(
                body[..index].iter().any(|earlier| builds_examples(earlier)),
                "the `{job}` job of {WORKFLOW} runs\n\
                 \x20   {}\n\
                 without building the examples first, so `tests/lifecycle.rs` and \
                 `tests/cdylib_tls.rs` will fail on a cold cache looking for \
                 fixtures that were compiled as test harnesses instead.\n\
                 Add a `cargo build --locked --examples` step ahead of it.",
                line.trim(),
            );
        }
    }

    for script in shell_scripts(&root.join("ci")) {
        let body = read(&script);
        let lines = commands(&body);
        for (index, line) in lines.iter().enumerate() {
            if !runs_all_targets_test(line) {
                continue;
            }
            checked += 1;
            assert!(
                lines[..index]
                    .iter()
                    .any(|earlier| builds_examples(earlier)),
                "{} runs\n\
                 \x20   {}\n\
                 without building the examples first; see the head of this file.",
                script.display(),
                line.trim(),
            );
        }
    }

    assert!(
        checked > 0,
        "nothing in {WORKFLOW} or ci/ runs `cargo test --all-targets`, so this \
         test checked nothing at all. Either the workflow changed, in which case \
         delete this file rather than let it pass for free, or the scan above \
         stopped recognising the command."
    );
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

fn shell_scripts(directory: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", directory.display()));
    for entry in entries {
        let path = entry.expect("a readable directory entry").path();
        if path.extension().is_some_and(|extension| extension == "sh") {
            found.push(path);
        }
    }
    found.sort();
    found
}
