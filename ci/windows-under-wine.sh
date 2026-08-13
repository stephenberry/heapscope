#!/usr/bin/env bash
#
# Runs this crate's tests for Windows, on a machine that is not Windows.
#
# CI already tests `windows-latest` natively, and that is the authority. This
# exists for the same reason `docker run rust:1-slim` exists for Linux: so a
# change to Windows-specific code can be *executed* before it is pushed, rather
# than compiled and hoped for. Private repositories have finite CI minutes, and
# "it compiles for Windows" has twice in this project turned out to mean nothing.
#
# What it is not: proof about Windows. Wine is a reimplementation, and the two
# places this crate touches the platform most directly — `RtlCaptureStackBackTrace`
# and `K32EnumProcessModules` — are exactly the places a reimplementation is most
# likely to differ. Treat a pass here as "no longer obviously broken", and the
# native CI job as the answer.
#
# Usage:  ci/windows-under-wine.sh [cargo test args...]

set -euo pipefail

IMAGE="rust:1-slim"
TARGET="x86_64-pc-windows-gnu"

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker is required (this needs an x86_64 Linux userland with wine)" >&2
    exit 2
fi

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> building and running the $TARGET tests under wine"
echo "    (first run installs mingw-w64 and wine in the container; allow a few minutes)"

# `--platform linux/amd64` because wine needs an x86_64 userland; on an Apple
# Silicon host this runs emulated, which is slow but works.
#
# A container-local CARGO_TARGET_DIR keeps this from clobbering the host's
# `target/`, which holds artifacts for a different architecture entirely.
# Deliberately not `exec`: the note printed after this run has to reach the
# reader, and `exec` would replace this shell before it could.
#
# The `bash -c` argument below is one single-quoted string, so an apostrophe
# anywhere inside it -- including in a comment -- ends the quote and breaks the
# script somewhere else entirely. Write "the tests in foo.rs", not "foo.rs`s
# tests". And no `#` comment may sit between the backslash-continued lines of
# the `docker run` itself; both mistakes were made writing this. `bash -n` on
# this file catches either.
docker run --rm \
    --platform linux/amd64 \
    -v "$repository":/work \
    -w /work \
    -e CARGO_TARGET_DIR=/tmp/target \
    "$IMAGE" \
    bash -euo pipefail -c '
        apt-get update -qq >/dev/null 2>&1
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
            gcc-mingw-w64-x86-64 wine64 >/dev/null 2>&1
        rustup target add '"$TARGET"' >/dev/null 2>&1

        # The test binary is a Windows executable, so every process it spawns
        # is spawned by Wine -- which can only launch Windows binaries. A Linux
        # symbolizer installed in this container is therefore unreachable from
        # inside the tests, whatever is on PATH. `tests/end_to_end.rs` refuses
        # to pass without one rather than skipping quietly, which is right for
        # a real platform and wrong here, so it is told explicitly that no
        # subprocess tooling exists.
        export HEAPSCOPE_NO_SUBPROCESS_TOOLS=1
        export WINEDEBUG=-all WINEPREFIX=/tmp/wine HOME=/tmp
        wine wineboot --init >/dev/null 2>&1 || true

        export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=wine
        # Matches the documented build configuration, and proves the point: the
        # frame-pointer flag changes nothing on Windows, because the default
        # strategy there does not use frame pointers.
        export RUSTFLAGS="-C force-frame-pointers=yes"

        # In-process symbolization is off for this whole run. `SymFromAddr`
        # cannot be executed here: the first call into Wine`s dbghelp dies with
        #
        #   rosetta error: invalid gdt selector index 5    (signal: 5, SIGTRAP)
        #
        # which is Apple`s Rosetta translator refusing a segment-descriptor
        # access, not Wine and not Windows. It aborts the whole test binary, so
        # it does not fail one test -- it produces no results for that binary
        # and none for any binary scheduled after it.
        #
        # This switch rather than a list of skips, because the reach is much
        # wider than the tests that name symbolization: everything that *writes
        # a profile* goes through it, which is all fifteen of `tests/lifecycle.rs`
        # by way of `examples/lifecycle_probe.rs`. Skipping all of those would
        # give up exactly the process-lifecycle matrix that M3 built this
        # harness to check. With symbolization off they all run, and what is
        # lost is one property rather than twenty tests.
        #
        # Nothing is skipped. The tests in `src/symbol/dynamic.rs` were, until
        # the switch existed; now `lookup` returns before it reaches dbghelp, so
        # they run and pass trivially -- and one of them,
        # `symbolization_is_on_unless_it_was_turned_off`, becomes a real check
        # here, because it reads the environment rather than assuming it. If the
        # switch ever stops working, this run says so loudly: either that test
        # fails, or the first `lookup` kills the binary again.
        export HEAPSCOPE_SYMBOLIZE=0

        echo "==> in-process symbolization is disabled here; see the note below"
        cargo test --target '"$TARGET"' "$@"
    ' -- "$@"

cat >&2 <<'SKIPPED'

==> NOT PROVED BY THE RUN ABOVE:

      Windows in-process symbolization. `SymFromAddr` was never called: the run
      sets HEAPSCOPE_SYMBOLIZE=0, so `lookup` returns before reaching dbghelp.
      Every frame above was rendered as image+offset.

    Because this harness cannot execute dbghelp. The first call into Wine's
    dbghelp kills the process outright:

      test symbol::dynamic::tests::a_local_function_is_named_if_this_build_has_names_at_all
      ... rosetta error: invalid gdt selector index 5      (signal: 5, SIGTRAP)

    That message is Apple's Rosetta translator, not Wine and not Windows -- a
    segment-descriptor access it will not translate. It aborts the whole test
    binary rather than failing one test, so the run yields no results for that
    binary and none for any binary after it: the same trap Miri set in M4, one
    layer down.

    Everything else still runs -- no test is skipped -- including all of
    tests/lifecycle.rs, which reaches the same code by writing profiles from a
    probe process. That is the reason for a switch rather than a skip list.

    The native `windows-latest` CI job is the only thing that can exercise
    `SymFromAddr`. To exercise it on this machine, run the containers under QEMU
    instead of Rosetta (OrbStack: `orb config set rosetta false`, which rebuilds
    the VM and slows every amd64 container down). Untested: it is a machine-wide
    setting and not this script's to change.
SKIPPED
