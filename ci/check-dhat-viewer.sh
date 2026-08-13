#!/usr/bin/env bash
#
# The M2 exit criterion, executable: a profile this crate writes must load in
# Valgrind's real `dh_view.html`.
#
# Produces one profile per mode from the example program and runs each through
# the genuine upstream `dh_view.js` under Node (see ci/dhat-viewer-check.mjs for
# how the browser page is stood up headlessly).
#
# Usage:
#   ci/check-dhat-viewer.sh                  # fetches dh_view.js, caches in tmp/
#   DH_VIEW_JS=/path/to/dh_view.js ci/check-dhat-viewer.sh
#
# `dh_view.js` is GPL-licensed and is deliberately not vendored here.
#
# Exit codes:
#   0  every profile loads and renders
#   1  the viewer rejected a profile, or the tools disagree about the numbers
#   2  the check could not be run (no node, or the viewer could not be fetched)
#
# The distinction between 1 and 2 is the point: "the viewer says no" must fail
# loudly, while "sourceware was unreachable" must not be reported as a defect in
# this crate.

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if ! command -v node > /dev/null 2>&1; then
  echo "skip: no node on PATH, so the viewer check cannot run" >&2
  exit 2
fi

viewer="${DH_VIEW_JS:-}"
if [ -z "$viewer" ]; then
  viewer="tmp/dhat-ref/dh_view.js"
  if [ ! -s "$viewer" ]; then
    mkdir -p "$(dirname "$viewer")"
    url="https://sourceware.org/git/?p=valgrind.git;a=blob_plain;f=dhat/dh_view.js;hb=HEAD"
    echo "fetching $url"
    if ! curl -fsS --retry 3 --retry-delay 2 --max-time 60 -o "$viewer" "$url"; then
      rm -f "$viewer"
      echo "skip: could not fetch dh_view.js; set DH_VIEW_JS to a local copy" >&2
      exit 2
    fi
  fi
fi

if [ ! -s "$viewer" ]; then
  echo "skip: '$viewer' is missing or empty" >&2
  exit 2
fi

mkdir -p tmp/viewer-check

# One file per mode. `bklt: false` omits two top-level fields and seven per-point
# ones, so an ad hoc or copy profile is a different *shape* of file for the
# viewer to accept, not the same file with different numbers.
for mode in heap ad-hoc copy; do
  profile="tmp/viewer-check/$mode.json"

  echo "recording a $mode profile"
  cargo run --locked --release --quiet --example profile_a_program "$profile" "$mode" > /dev/null

  # Nothing else runs this example, so the mode argument is checked here or
  # nowhere: swapping two arms of its `match` produced three profiles that all
  # loaded, and no test noticed.
  if ! grep -q "\"mode\":\"$mode\"" "$profile"; then
    echo "error: the $mode profile does not say it is one" >&2
    exit 1
  fi

  echo "loading it in dh_view.js"
  node ci/dhat-viewer-check.mjs "$viewer" "$profile"
done
