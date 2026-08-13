#!/usr/bin/env bash
#
# The bundled viewer of PLAN.md section 6.12, checked against real profiles.
#
# Records one profile per mode and runs the page's own logic over the profile it
# carries (see ci/check-bundled-viewer.mjs). One file per mode because the modes
# produce different *shapes* of profile, not the same shape with different
# numbers: an ad hoc or copy run has no block lifetimes, so seven per-point
# fields are simply absent and every reader of them has to cope.
#
# Usage:
#   ci/check-bundled-viewer.sh
#
# Exit codes:
#   0  every page carries a profile its own code reads correctly
#   1  a check failed
#   2  the check could not be run (no node)
#
# The distinction between 1 and 2 matches ci/check-dhat-viewer.sh: "the viewer
# is wrong" must fail loudly, and "there is no JavaScript engine here" must not
# be reported as a defect in this crate.

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if ! command -v node > /dev/null 2>&1; then
  echo "skip: no node on PATH, so the bundled viewer check cannot run" >&2
  exit 2
fi

mkdir -p tmp/viewer-check

pages=()
for mode in heap ad-hoc copy; do
  echo "recording a $mode profile"
  cargo run --locked --release --quiet --example profile_a_program \
    "tmp/viewer-check/$mode.json" "$mode" > /dev/null
  pages+=("tmp/viewer-check/$mode.html")
done

node ci/check-bundled-viewer.mjs "${pages[@]}"
