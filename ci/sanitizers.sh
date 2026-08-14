#!/usr/bin/env bash
#
# AddressSanitizer and ThreadSanitizer, of PLAN.md section 8.7.
#
# Each sanitizer runs in two parts, and the order is the point. First a
# *positive control*: a probe with a deliberate defect, which the sanitizer must
# report. Only then the suite, which it must not.
#
# The controls are here because this crate is the awkward case for both tools.
# ASan works by replacing `malloc` and putting redzones around what it hands
# back; heapscope installs a `#[global_allocator]` that sits between the program
# and `malloc`. TSan reasons about happens-before through the synchronisation it
# recognises; heapscope's subject is hand-rolled locking. If either composition
# were broken the tool would go quiet, the suite would pass, and the job would
# report success for having watched nothing. A green run is only worth what the
# control proves, so the control runs first and its failure is fatal.
#
# Usage:
#   ci/sanitizers.sh [address|thread|all]     (default: all)
#
# Exit codes:
#   0  every control fired and every suite passed
#   1  a control did not fire, or a suite failed
#   2  the run could not happen here (no nightly toolchain, or no rust-src)
#
# The 1/2 split matches the other scripts in this directory: "the crate is
# broken" must fail loudly, and "this machine has no nightly" must not be
# reported as a defect in the crate.

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

which="${1:-all}"
case "$which" in
  address | thread | all) ;;
  *)
    echo "usage: ci/sanitizers.sh [address|thread|all]" >&2
    exit 2
    ;;
esac

if ! rustup toolchain list 2>/dev/null | grep -q '^nightly'; then
  echo "skip: no nightly toolchain, and -Zsanitizer is nightly-only" >&2
  echo "      rustup toolchain install nightly" >&2
  exit 2
fi

# `-Zbuild-std` needs the standard library's source. It is required for
# ThreadSanitizer rather than merely nice: an uninstrumented std hides its own
# synchronisation from the race detector, which then reports races that are not
# there. A tool that cries wolf is worse here than no tool, because the first
# response to a red TSan job is to disbelieve it.
# Only ThreadSanitizer needs it, so an `address`-only run must not be blocked by
# its absence -- which it was, and the symptom was this script exiting 2 and
# printing nothing anyone connected to the missing component.
if [ "$which" != address ] &&
  ! rustup component list --toolchain nightly --installed 2>/dev/null | grep -q '^rust-src'; then
  echo "skip: nightly has no rust-src, so std cannot be instrumented for TSan" >&2
  echo "      rustup component add rust-src --toolchain nightly" >&2
  echo "      (ci/sanitizers.sh address does not need it)" >&2
  exit 2
fi

# An explicit triple is not optional. `-Zsanitizer` is silently ignored when the
# target is the implicit host, which produces a clean run that instrumented
# nothing — the exact failure this script's controls exist to catch, arriving
# one level lower where they cannot see it.
target="${SANITIZER_TARGET:-$(rustc -vV | sed -n 's/^host: //p')}"

probe_manifest="$root/ci/sanitizer-probe/Cargo.toml"

# Runs the probe and requires `expected` to appear in what the sanitizer said.
# A probe that exits cleanly has not proved anything, so that is a failure too.
require_report() {
  local binary="$1" case="$2" expected="$3"
  local output status
  set +e
  output="$("$binary" "$case" 2>&1)"
  status=$?
  set -e

  if [ "$status" -eq 0 ]; then
    echo "FAIL: the control \`$case\` exited cleanly." >&2
    echo "      The sanitizer did not see a defect in memory that went through" >&2
    echo "      heapscope's global allocator, so a green suite would mean only" >&2
    echo "      that nothing was being watched." >&2
    return 1
  fi
  if printf '%s' "$output" | grep -q "ThreadSanitizer: .*memory layout"; then
    echo "FAIL: ThreadSanitizer could not map its shadow memory here." >&2
    echo "      This is the environment, not the crate: TSan needs a specific" >&2
    echo "      address-space layout, and a kernel with high mmap randomisation" >&2
    echo "      or a container that blocks disabling ASLR denies it one." >&2
    echo "      Remedy: sysctl -w vm.mmap_rnd_bits=28 (and, in Docker," >&2
    echo "      --privileged, since the tunable is not namespaced)." >&2
    printf '%s\n' "$output" >&2
    return 1
  fi
  if ! printf '%s' "$output" | grep -q "$expected"; then
    echo "FAIL: the control \`$case\` failed without reporting \`$expected\`:" >&2
    printf '%s\n' "$output" >&2
    return 1
  fi
  echo "  control \`$case\`: reported, as it must be"
}

require_clean() {
  local binary="$1"
  if ! "$binary" clean > /dev/null 2>&1; then
    echo "FAIL: the probe reported a defect in the case that has none." >&2
    return 1
  fi
  echo "  control \`clean\`: silent, as it must be"
}

run_sanitizer() {
  local sanitizer="$1"
  shift
  # `${extra[@]+...}` rather than a bare `"${extra[@]}"`: bash 3.2, which is
  # what macOS ships, treats an empty array as unset under `set -u`.
  local -a extra=("$@")

  echo
  echo "=== $sanitizer on $target ==="

  # One directory per sanitizer, because they build with different RUSTFLAGS and
  # would otherwise rebuild the world on every alternation. `SANITIZER_TARGET_DIR`
  # exists so that a run in a container can keep its foreign-architecture output
  # off a mounted working tree.
  export CARGO_TARGET_DIR="${SANITIZER_TARGET_DIR:-$root/target}/sanitizer-$sanitizer"
  export RUSTFLAGS="-Zsanitizer=$sanitizer -C force-frame-pointers=yes"

  # `tests/cdylib_tls.rs` `dlopen`s an instrumented shared object, and on an ELF
  # platform that fails with `undefined symbol:
  # __asan_option_detect_stack_use_after_return`. Rust links ASan's runtime
  # *statically* into the executable, so its symbols are not in the dynamic
  # table and the library loaded at run time cannot resolve them. Exporting them
  # is the fix; there is nothing to do on macOS, where the runtime is already a
  # dylib every image can see.
  case "$sanitizer:$target" in
    address:*linux*) RUSTFLAGS="$RUSTFLAGS -C link-arg=-Wl,--export-dynamic" ;;
  esac

  # ASan's use-after-return detection moves locals into a heap-allocated "fake
  # stack", so a local's address is no longer inside the thread's real stack --
  # observed on x86_64 Linux at roughly 3 MB above the reported top. This crate
  # reads those bounds on purpose: `internals::stack` reports them and the
  # frame-pointer walk rejects a frame outside them, which is what stops a walk
  # from following a corrupt chain into unmapped memory. With fake stacks on,
  # five of its unit tests fail for saying something true.
  #
  # Set explicitly rather than left to the platform: macOS defaults it off and
  # Linux defaults it on, so leaving it alone means the two runs check different
  # things while reporting the same word.
  #
  # What this gives up is real and worth naming: ASan will no longer catch a
  # reference to a local that outlived its frame. Nothing else is affected --
  # redzones, quarantine, and every heap check stay on, which is what the
  # controls below exercise.
  #
  # LeakSanitizer rides along with ASan and is on by default on Linux only. It
  # stays on -- a leak check is worth having -- with one suppression, for a
  # fixture that forgets a profiler on purpose. See ci/lsan-suppressions.txt.
  if [ "$sanitizer" = address ]; then
    export ASAN_OPTIONS="detect_stack_use_after_return=0${ASAN_OPTIONS:+:$ASAN_OPTIONS}"
    export LSAN_OPTIONS="suppressions=$root/ci/lsan-suppressions.txt${LSAN_OPTIONS:+:$LSAN_OPTIONS}"
  fi

  # TSan's *deadlock detector* cannot represent this crate, and says so by
  # aborting the process rather than by reporting anything:
  #
  #   ThreadSanitizer: CHECK failed: sanitizer_deadlock_detector.h:67
  #   "((n_all_locks_)) < ((sizeof(all_locks_with_contexts_)/...))" (0x80, 0x80)
  #
  # 0x80 is 128, the fixed size of its per-thread lock table. This crate holds
  # **131** locks: `SHARDS` is 64 for the live-block table and 64 again for the
  # program-point table (`internals/live.rs`, `internals/pp.rs`), plus the peak
  # gate, the arena, and the region intern lock. A test that touches enough
  # shards therefore walks straight past the limit.
  #
  # **Measured**: it killed the run 83 tests into the 526 in the library suite,
  # right after the multi-threaded engine tests, and left the job to sit until
  # something stopped it -- 149 minutes on one run, and 90 on the next, which is
  # what `timeout-minutes` was added to bound. It was never slow. It was dead.
  #
  # What this gives up is nothing this job was relied on for. The *race*
  # detector is untouched, and that is what section 8's sanitizer item asks for
  # -- the first run against the real `os_unfair_lock`, the real allocator and
  # real threads, where Miri only ever sees the `cfg(miri)` backend. Lock
  # *ordering* has its own authority here and always has: `internals/order.rs`
  # enforces the documented order in debug builds, which is a check that knows
  # this crate's four lock families rather than one that runs out of slots at
  # 128.
  if [ "$sanitizer" = thread ]; then
    export TSAN_OPTIONS="detect_deadlocks=0${TSAN_OPTIONS:+:$TSAN_OPTIONS}"
  fi

  cargo +nightly build ${extra[@]+"${extra[@]}"} --target "$target" \
    --manifest-path "$probe_manifest"
  local binary="$CARGO_TARGET_DIR/$target/debug/sanitizer-probe"

  # The crate's own examples, which `--tests` below does not build and which
  # three of the suites need in order to find their fixtures.
  cargo +nightly build ${extra[@]+"${extra[@]}"} --target "$target" --examples

  require_clean "$binary"
  case "$sanitizer" in
    address)
      require_report "$binary" use-after-free "heap-use-after-free"
      require_report "$binary" overflow "heap-buffer-overflow"
      ;;
    thread)
      require_report "$binary" race "data race"
      ;;
  esac

  # `--tests` and not `--all-targets`: the overhead bench cannot run in a debug
  # profile, and would fail here for a reason that has nothing to do with
  # sanitizers. The examples are built above, which `--tests` does not do and
  # which three of the suites need to find their fixtures.
  #
  # Serial, because one process holds one engine: the suites that start a
  # profiler cannot overlap, and under instrumentation the margin is thinner.
  #
  # `HEAPSCOPE_SANITIZER` tells the suite a sanitizer is watching, so that the
  # work which cannot produce a finding can stand down. `cfg(sanitize = "..")`
  # would be the natural way to ask and is unstable, so the harness says so
  # itself — the same arrangement `tests/symbolize.rs` uses to declare that it
  # cannot spawn a subprocess.
  #
  # What stands down is the demangler's corpus walks, on the same argument that
  # already excuses them from Miri: `src/symbol/demangle` contains no `unsafe`
  # at all, so neither ASan nor TSan has anything there to find. That is not a
  # claim anyone has to keep true by hand — `tests/no_dependencies.rs` fails if
  # an `unsafe` block appears in that directory, which is why the exemption
  # cannot quietly outlive its reason. Measured under Miri, where the same cut
  # took the job from 37 minutes to about 15.
  echo "  suite:"
  HEAPSCOPE_SANITIZER="$sanitizer" \
    cargo +nightly test ${extra[@]+"${extra[@]}"} --target "$target" --tests -- --test-threads=1
}

failed=0

if [ "$which" = address ] || [ "$which" = all ]; then
  run_sanitizer address || failed=1
fi

if [ "$which" = thread ] || [ "$which" = all ]; then
  # See the rust-src note above for why ThreadSanitizer alone rebuilds std.
  run_sanitizer thread -Zbuild-std || failed=1
fi

echo
if [ "$failed" -ne 0 ]; then
  echo "sanitizers: FAILED"
  exit 1
fi
echo "sanitizers: every control fired and every suite passed"
