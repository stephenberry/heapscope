#!/usr/bin/env bash
#
# Verifies that the *shipped library* builds on the MSRV.
#
# `cargo check --lib` on the real manifest is not good enough. Cargo still
# parses `[dev-dependencies]` and still resolves them into the lock file, so a
# dev-dependency that raises its own MSRV, bumps its edition, or starts using a
# manifest key the MSRV toolchain cannot parse breaks this job through no action
# of ours — which is exactly the drift PLAN.md section 1.2 promises to contain.
#
# So the check runs against a copy of the crate with every dev-only section
# removed. What remains is precisely what a downstream consumer compiles. A
# dev-dependency now provably cannot affect the MSRV we promise users.
#
# Usage: ci/msrv-check.sh [toolchain]     (default: the MSRV in Cargo.toml)

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

msrv="$(sed -n 's/^rust-version *= *"\([^"]*\)".*/\1/p' Cargo.toml | head -n1)"
if [[ -z "$msrv" ]]; then
    echo "error: no rust-version in Cargo.toml" >&2
    exit 1
fi
toolchain="${1:-$msrv}"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

crate="$workdir/heapscope"
mkdir -p "$crate"
cp -R src "$crate/src"
[[ -f README.md ]] && cp README.md "$crate/"
# Anything the library reaches with `include_str!`/`include_bytes!` must be
# copied here too, or this check fails for a reason that has nothing to do with
# the MSRV. The M7 viewer is the first thing that will need a line added.
[[ -d viewer ]] && cp -R viewer "$crate/viewer"

# Drop every section that exists only for development: dev-dependencies, and the
# bench/test/example targets that need them.
awk '
    /^\[/ {
        section = $0
        gsub(/[][ ]/, "", section)
        # `section` has already had its brackets stripped, so `[[bench]]`
        # arrives here as `bench`. Matching the bracketed spelling as well
        # would be dead code that reads as though it does the work.
        drop = (section == "dev-dependencies" ||
                section ~ /^dev-dependencies\./ ||
                section ~ /\.dev-dependencies$/ ||
                section ~ /\.dev-dependencies\./ ||
                section == "bench" ||
                section == "test" ||
                section == "example")
    }
    !drop { print }
' Cargo.toml > "$crate/Cargo.toml"

echo "==> checking the shipped library on Rust $toolchain"
echo "    (dev-dependencies stripped; this is what a consumer compiles)"
sed -n '/^\[dependencies\]/,/^\[/p' "$crate/Cargo.toml"

cd "$crate"
# A fresh lock file, because the repository's lock file pins dev-dependencies
# that no longer appear in this manifest.
rm -f Cargo.lock
# `--lib` only, deliberately, and NOT `--all-targets`. Target selection is a
# union: `--all-targets` also builds the lib's own `#[cfg(test)]` unit-test
# target, which is compiled against the stripped manifest that has no
# dev-dependencies. The moment a `#[cfg(test)] mod` in src/ uses `proptest` --
# which PLAN.md section 8.4 requires -- this job fails, and a dev-dependency
# would once again be able to break the MSRV floor. That is the exact thing the
# script exists to make impossible.
cargo "+$toolchain" check --lib 2>&1 | sed 's/^/    /'
cargo "+$toolchain" build --lib 2>&1 | sed 's/^/    /'
# `--bins` as well, and deliberately not `--all-targets`: `heapscope-symbolize`
# ships in the package, so `cargo install heapscope` on the MSRV has to work,
# and `--bins` selects it without dragging in the lib's `#[cfg(test)]` target
# that the paragraph above exists to keep out.
cargo "+$toolchain" check --bins 2>&1 | sed 's/^/    /'

echo "==> OK: the shipped library and binary build on Rust $toolchain with no dependencies"
