#!/usr/bin/env bash
#
# The half of PLAN.md section 4.6's panic row that `cargo test` cannot reach.
#
# The row makes two claims about a program built with `panic = "abort"`, and the
# suite can check neither, because every test binary is built one way and that
# way is `panic = "unwind"`. A claim nobody can run is how this repository has
# been wrong before.
#
#   1. A program that ends normally is profiled exactly as it would be
#      otherwise. This is the claim users depend on -- `panic = "abort"` is an
#      ordinary release setting -- and it is the one that can break silently, the
#      day anything on the shutdown path comes to need unwinding.
#
#   2. A program that *panics* produces no profile at all, because a panic under
#      this setting is an `abort`, and `abort` is defined not to run `atexit`
#      handlers. That is a limitation rather than a defect, and the point of
#      checking it is that it stays stated instead of being discovered by
#      someone whose profile is mysteriously missing.
#
# The row said something else entirely until M7 -- that a panic here "falls back
# to `atexit`" -- and that was corrected by reading `abort`'s definition rather
# than by running anything. This runs it.
#
# Usage:
#   ci/check-panic-abort.sh
#
# Exit codes:
#   0  both claims hold
#   1  one of them does not

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

out="tmp/panic-abort"
rm -rf "$out/normal" "$out/panicking"
mkdir -p "$out/normal" "$out/panicking"

# Composed rather than replaced: CI puts `-D warnings` in the environment, and
# frame pointers are required on x86_64 (PLAN.md section 5.3) exactly as the
# `test` job configures them. The only addition this script exists for is the
# last one.
export RUSTFLAGS="${RUSTFLAGS:-} -C force-frame-pointers=yes -C panic=abort"

# A target directory of its own, because every artifact here is compiled with a
# panic strategy the rest of the tree is not, and sharing would mean rebuilding
# the world in both directions on every alternation. Kept between runs for the
# same reason.
#
# Reached through `cargo run` rather than by naming a path inside it: a script
# that names a path under a target directory runs whatever it finds there, and
# what it finds is not always what this script built.
target="$out/target"

# Through a nested `bash -c` rather than a plain subshell, so that the shell's
# own "Aborted" notification lands in the log file with the rest of the run's
# output. One of the two modes here is *expected* to die by SIGABRT, and the
# parent shell announcing it on stderr, asynchronously and after the check has
# already reported success, reads exactly like the failure this script exists to
# rule out.
#
# Every path handed across is relative to the repository root, and nothing
# changes directory. This runs on the Windows runner too, under the bash that
# ships with git, where an absolute path is `/d/a/...` — a shape the shell
# understands and a Windows executable does not: Rust would read it as rooted on
# whichever drive happens to be current, write the profile somewhere else
# entirely, and this script would report a defect that is really a path.
run_probe() {
  local mode="$1"
  local directory="$out/$2"
  bash -c '
    cargo run --locked --quiet --target-dir "$1" \
      --example lifecycle_probe -- "$2" "$3/dhat-heap.json"
  ' _ "$target" "$mode" "$directory" \
    > "$directory/stdout.txt" 2> "$directory/stderr.txt"
}

echo "==> building the lifecycle probe with -C panic=abort"

# Claim 1. A normal ending, which is the ordinary case and has to be ordinary.
if ! run_probe drop normal; then
  echo "error: a panic=abort build failed to run to completion" >&2
  sed 's/^/    /' "$out/normal/stderr.txt" >&2
  exit 1
fi

profile="$out/normal/dhat-heap.json"
if [[ ! -f "$profile" ]]; then
  echo "error: a panic=abort build wrote no profile from a normal ending." >&2
  echo "       This is the claim users depend on: the setting is an ordinary" >&2
  echo "       release configuration, not an exotic one." >&2
  exit 1
fi
if ! grep -q '"shutdown": *"drop"' "$profile"; then
  echo "error: the profile does not record the drop path it was written from:" >&2
  head -c 400 "$profile" >&2
  exit 1
fi
if ! grep -qE '"totalBytes": *[1-9]' "$profile"; then
  echo "error: a panic=abort build recorded no bytes, so the shim recorded" >&2
  echo "       nothing even though the program ran." >&2
  exit 1
fi
echo "    a normal ending is profiled: $(wc -c < "$profile" | tr -d ' ') bytes, shutdown=drop"

# Claim 2. The limitation, stated and checked. `run_probe` is expected to fail
# here, and a *successful* exit is itself a failure of the check: it would mean
# the panic did not abort, which is not what this build strategy does.
if run_probe panic panicking; then
  echo "error: a panic under panic=abort exited successfully, so the program" >&2
  echo "       did not abort and this build is not what it claims to be." >&2
  exit 1
fi
if [[ -f "$out/panicking/dhat-heap.json" ]]; then
  echo "error: a panicking panic=abort build produced a profile. \`abort\` does" >&2
  echo "       not run \`atexit\` handlers, so something else wrote it -- which" >&2
  echo "       is not a mechanism this crate has -- or the build is not" >&2
  echo "       panic=abort at all." >&2
  exit 1
fi
echo "    a panic writes nothing, as the row says: no profile, no atexit handler"

echo "==> OK: panic=abort profiles a normal ending and, as documented, not a panic"
