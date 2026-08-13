#!/usr/bin/env bash
#
# The bundled viewer's controls, clicked in a real browser.
#
# `ci/check-bundled-viewer.sh` covers the half of the page that has no DOM. This
# covers the half that has nothing else: tabs, sorting, trimming, expand,
# collapse, and the twisty on every branch. See ci/check-viewer-interaction.mjs
# for how a browser is driven with nothing installed, and for what this does not
# claim to cover.
#
# Usage:
#   ci/check-viewer-interaction.sh
#
# Exit codes:
#   0  every control did what the profile it is showing says it should
#   1  a check failed
#   2  the check could not be run (no node, or no Chrome)
#
# As in the sibling checks, 2 is not a defect in this crate: a machine with no
# browser must not report the viewer as broken.

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if ! command -v node > /dev/null 2>&1; then
  echo "skip: no node on PATH, so the viewer's controls cannot be driven" >&2
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

node ci/check-viewer-interaction.mjs "${pages[@]}"
