//! Enforcement of the project's two structural promises.
//!
//! Both are the kind of rule that is stated in a README, believed for a year,
//! and then quietly broken by a convenient one-line addition. A test is the
//! only version of a promise that survives contact with a deadline.

use std::path::{Path, PathBuf};

// Every test here reads the repository from disk. Miri runs with filesystem
// isolation on by default, which turns `open` into a hard machine abort rather
// than an `io::Error`, so under Miri these would abort the whole run. Disabling
// isolation to accommodate them would weaken the Miri job for a reason that has
// nothing to do with what Miri is there to check, so they are ignored instead.
// (The `ignore` reason must be a string literal, not a named constant.)

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Splits a Cargo manifest into `(section_header, body_lines)` pairs.
///
/// Deliberately hand-rolled: pulling in a TOML parser to check that we have no
/// dependencies would be self-defeating. The manifest is ours, so the input is
/// not adversarial; this only has to handle the subset we actually write.
fn sections(manifest: &str) -> Vec<(String, Vec<&str>)> {
    let mut out: Vec<(String, Vec<&str>)> = Vec::new();
    let mut current = String::new();
    let mut body: Vec<&str> = Vec::new();
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            if !current.is_empty() || !body.is_empty() {
                out.push((current.clone(), std::mem::take(&mut body)));
            }
            current = trimmed.trim_matches(['[', ']']).to_string();
        } else {
            body.push(line);
        }
    }
    out.push((current, body));
    out
}

fn is_meaningful(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && !trimmed.starts_with('#')
}

/// `[dependencies]` is empty and stays empty.
///
/// This is the claim the README makes and the one users care about: nothing
/// this crate declares reaches a downstream consumer's build. `[dev-dependencies]`
/// are exempt because Cargo never builds them for consumers.
#[test]
#[cfg_attr(
    miri,
    ignore = "reads the repository from disk; Miri isolation blocks file I/O"
)]
fn shipped_library_has_no_dependencies() {
    let manifest = std::fs::read_to_string(repo_root().join("Cargo.toml")).unwrap();

    for (name, body) in sections(&manifest) {
        // Catches `[dependencies]`, `[target.'cfg(unix)'.dependencies]`, and
        // `[dependencies.foo]`, while leaving dev- and build-dependencies alone.
        let is_runtime_deps = name == "dependencies"
            || name.starts_with("dependencies.")
            || (name.starts_with("target.") && name.ends_with(".dependencies"))
            || (name.starts_with("target.") && name.contains(".dependencies."));
        if !is_runtime_deps {
            continue;
        }

        let entries: Vec<&str> = body.iter().copied().filter(|l| is_meaningful(l)).collect();
        assert!(
            entries.is_empty(),
            "[{name}] must stay empty; found:\n{}\n\n\
             The shipped library has no dependencies outside the standard library. \
             Platform capabilities are reached with `extern \"C\"` declarations against \
             libraries the process already links. If you believe you need a crate here, \
             that is a design discussion, not a patch.",
            entries.join("\n")
        );
    }
}

/// The dev-dependency set is closed.
///
/// Dev-dependencies cost users nothing at build time, but each one is a new
/// MSRV that can drift out from under us and a new supply-chain surface for
/// anyone running our test suite. Adding one should require editing this list,
/// which makes it a decision rather than an accident.
#[test]
#[cfg_attr(
    miri,
    ignore = "reads the repository from disk; Miri isolation blocks file I/O"
)]
fn dev_dependencies_are_an_explicit_allowlist() {
    // `rustc-demangle` is the odd one out, and deliberately so: it is the
    // implementation `src/symbol/demangle` exists in order not to depend on.
    // Reimplementing both Rust manglings is a claim that the reimplementation
    // agrees with the one everyone else uses, and the only way to hold that
    // claim is to run both and compare. `tests/demangle.rs` does, over 600
    // symbols taken from real binaries; `tests/demangle_fuzz.rs` does it again
    // over generated ones. Removing this dependency would turn a checked
    // property back into an assertion.
    //
    // `dhat` is there for the same reason and pays for itself the same way.
    // This crate emits `dhat-rs`'s format and its whole justification is what it
    // does differently, so "cheaper than dhat-rs" is a claim that has to be run
    // rather than asserted. `benches/overhead.rs` runs it, over a workload all
    // three fixtures share and a checksum they have to agree on. It also buys
    // something no test here could: the two implementations count the same run
    // independently, which is the only audit of heapscope's accounting in this
    // repository that heapscope does not carry out itself.
    const ALLOWED: &[&str] = &[
        "proptest",
        "tempfile",
        "criterion",
        "rustc-demangle",
        "dhat",
    ];

    let manifest = std::fs::read_to_string(repo_root().join("Cargo.toml")).unwrap();
    let mut found = Vec::new();
    for (name, body) in sections(&manifest) {
        if name != "dev-dependencies" {
            continue;
        }
        for line in body.iter().filter(|l| is_meaningful(l)) {
            // Only a line that starts at column zero introduces a key. An
            // indented line is a continuation of a multi-line inline table, and
            // treating `  "bar",` as a dependency name would fail the allowlist
            // for something that is not a dependency at all.
            if line.starts_with([' ', '\t']) {
                continue;
            }
            let Some(key) = line.split(['=', '.']).next().map(str::trim) else {
                continue;
            };
            if !key.is_empty() {
                found.push(key.to_string());
            }
        }
    }

    for dep in &found {
        assert!(
            ALLOWED.contains(&dep.as_str()),
            "dev-dependency `{dep}` is not in the allowlist in {}. \
             Add it deliberately, with a reason, or remove it.",
            file!()
        );
    }
}

/// The bundled viewer never acquires a build step.
///
/// The viewer is one hand-written HTML file. The failure mode this guards
/// against is gradual: a helper script, then a bundler, then `node_modules`,
/// and eventually a Rust crate whose tests cannot run without npm. There is no
/// point at which anyone decides to do that, which is exactly why it needs a
/// test rather than a policy.
#[test]
#[cfg_attr(
    miri,
    ignore = "reads the repository from disk; Miri isolation blocks file I/O"
)]
fn repository_has_no_javascript_build_system() {
    const FORBIDDEN: &[&str] = &[
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "node_modules",
    ];

    let root = repo_root();
    let mut offenders = Vec::new();
    walk(&root, &mut |path| {
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if FORBIDDEN.contains(&name) {
                offenders.push(path.to_path_buf());
            }
        }
    });

    assert!(
        offenders.is_empty(),
        "the viewer is a single hand-written file with no build step, ever. Found: {offenders:?}"
    );
}

fn walk(dir: &Path, visit: &mut impl FnMut(&Path)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // `target/` holds whatever our dev-dependencies vendored, and `tmp/` is
        // gitignored scratch space. Neither is ours to police.
        if name == "target" || name == ".git" || name == "tmp" {
            continue;
        }
        visit(&path);
        if path.is_dir() {
            walk(&path, visit);
        }
    }
}

/// The demangler contains no `unsafe`, and that is load-bearing.
///
/// `tests/demangle.rs` and the corpus work in `tests/demangle_fuzz.rs` are
/// skipped or scaled down under Miri, on the stated grounds that Miri is there
/// to find undefined behaviour and there is none to find in a module with no
/// `unsafe` in it. That reasoning stops holding the moment somebody adds one,
/// and nothing would notice — the Miri job would keep passing, having quietly
/// stopped covering the code that needed it.
///
/// So the claim is checked rather than asserted. If this fails, either remove
/// the `unsafe` or remove the Miri exemptions; both are defensible, silently
/// keeping both is not.
#[test]
#[cfg_attr(
    miri,
    ignore = "reads the repository from disk; Miri isolation blocks file I/O"
)]
fn the_demangler_contains_no_unsafe() {
    let directory = repo_root().join("src/symbol/demangle");
    let mut checked = 0;
    let mut offenders = Vec::new();
    for entry in std::fs::read_dir(&directory).expect("the demangler directory") {
        let path = entry.expect("a directory entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        checked += 1;
        let source = std::fs::read_to_string(&path).expect("readable source");
        for (number, line) in source.lines().enumerate() {
            // Comments and string literals are stripped before looking, because
            // the demangler legitimately *prints* the word: `unsafe fn(..)` is
            // a type it has to render, and an `ignore` reason mentions it too.
            // Not a Rust lexer, and it does not need to be — it only has to be
            // wrong in the safe direction, and a stray quote makes it flag more
            // rather than less.
            let code = line.split("//").next().unwrap_or("");
            let outside_strings = code.split('"').step_by(2).collect::<Vec<_>>().join(" ");
            let is_declaration = outside_strings
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .any(|word| word == "unsafe");
            if is_declaration {
                offenders.push(format!(
                    "{}:{}: {}",
                    path.display(),
                    number + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(checked >= 4, "only found {checked} demangler source files");
    assert!(
        offenders.is_empty(),
        "the demangler is no longer free of `unsafe`, so the Miri exemptions in \
         tests/demangle.rs and tests/demangle_fuzz.rs are no longer justified:\n{}",
        offenders.join("\n")
    );
}
