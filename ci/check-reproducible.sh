#!/usr/bin/env bash
#
# PLAN.md section 12's reproducibility claim, checked against real processes.
#
# Records the same workload several times and compares the profiles. What must
# match is everything the *program* did: every program point, every counter,
# every frame, and the order they are written in. What is allowed to differ is
# everything that describes *this execution* rather than the workload, listed
# and justified in `normalize` below.
#
# Two processes rather than two writes from one, because the defect this exists
# to catch cannot appear within a single process. Program points are sharded by
# a hash of their return addresses, and address space layout randomization moves
# those addresses on every execution — so one process hashes into one shard
# order and can only ever agree with itself.
#
# Usage:
#   ci/check-reproducible.sh [runs]        (default: 5)
#
# Exit codes:
#   0  every run produced the same profile
#   1  two runs disagreed
#   2  the check could not be run (no python3)
#
# The 1/2 split matches ci/check-dhat-viewer.sh: "the profiles disagree" must
# fail loudly, and "there is no Python here" must not be reported as a defect in
# this crate.

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if ! command -v python3 > /dev/null 2>&1; then
  echo "skip: no python3 on PATH, so the profiles cannot be compared" >&2
  exit 2
fi

runs="${1:-5}"
out="tmp/reproducible"
rm -rf "$out"
mkdir -p "$out"

# Through `cargo run`, as ci/check-bundled-viewer.sh does, rather than by naming
# a path under `target/`: `CARGO_TARGET_DIR` moves that directory, and a script
# that assumes it will happily run whatever executable it finds at the assumed
# path. Under a container with the source tree mounted, that is the host's
# binary for the host's architecture.
#
# Release, and the same binary every run — only the first invocation builds
# anything. A rebuild between runs would move the addresses for reasons that
# have nothing to do with the claim.
for run in $(seq 1 "$runs"); do
  echo "recording run $run of $runs"
  # The workload prints a summary of its own. Kept out of the way but kept, so
  # a run that dies has something to show for it.
  if ! cargo run --locked --release --quiet --example profile_a_program \
      "$out/run$run.json" heap > "$out/run$run.log" 2>&1; then
    echo "run $run of the workload failed:" >&2
    cat "$out/run$run.log" >&2
    exit 1
  fi
done

python3 - "$out" "$runs" <<'PY'
import json, re, sys

out, runs = sys.argv[1], int(sys.argv[2])


def strip(mapping, keys):
    return {key: value for key, value in mapping.items() if key not in keys}


def normalize(profile, native):
    """Drops what describes this execution rather than the workload.

    Every removal below is a fact about the process, and each one is a fact a
    reader would be wrong to diff:

    `pid` and the command line          -- name the process, not the program.
    module `load`/`start`/`bias`        -- where the loader put each image.
    the runtime address on each frame   -- the same, per frame. The `image +
                                           offset` half of every frame stays,
                                           and it is the half a symbolizer
                                           resolves, so nothing about *which*
                                           code ran is dropped here.
    `captureCost`                       -- a timing measured at start-up.
    `arena` and `programPoints` bytes   -- the profiler's own occupancy, which
                                           moves with the shard distribution
                                           and so with the addresses above.

    Everything else has to match, including the order of `pps`, `points`, and
    the frame tables both of them index.
    """
    profile = dict(profile)
    if native:
        profile["run"] = strip(profile["run"], ("command", "pid"))
        profile["frames"] = [strip(frame, ("addr",)) for frame in profile["frames"]]
        metrics = strip(profile["selfMetrics"], ("captureCost",))
        metrics["arena"] = strip(metrics["arena"], ("bytesUsed",))
        metrics["programPoints"] = strip(metrics["programPoints"], ("bytes",))
        profile["selfMetrics"] = metrics
        modules = profile
    else:
        profile = strip(profile, ("cmd", "pid"))
        profile["ftbl"] = [re.sub(r"^0x[0-9a-f]+: ", "", f) for f in profile["ftbl"]]
        heapscope = dict(profile["heapscope"])
        metrics = strip(heapscope["selfMetrics"], ("captureCost",))
        metrics["arena"] = strip(metrics["arena"], ("bytesUsed",))
        metrics["programPoints"] = strip(metrics["programPoints"], ("bytes",))
        heapscope["selfMetrics"] = metrics
        profile["heapscope"] = heapscope
        modules = profile["heapscope"]
    modules["modules"] = [
        strip(module, ("load", "start", "bias")) for module in modules["modules"]
    ]
    return json.dumps(profile, sort_keys=True)


failed = False
for name, suffix, native in (("DHAT v2", ".json", False), ("native", ".native.json", True)):
    seen = {}
    for run in range(1, runs + 1):
        path = f"{out}/run{run}{suffix}"
        with open(path) as handle:
            seen.setdefault(normalize(json.load(handle), native), []).append(run)
    if len(seen) == 1:
        print(f"  {name}: {runs} runs, one profile")
        continue
    failed = True
    groups = sorted(seen.values(), key=lambda runs: runs[0])
    print(f"  {name}: {runs} runs produced {len(groups)} different profiles")
    for group in groups:
        print(f"    runs {', '.join(str(run) for run in group)}")
    # Name the first disagreement rather than leaving the reader to diff two
    # files of a few hundred kilobytes by eye.
    first, second = (json.loads(key) for key in list(seen)[:2])
    for key in first:
        if first[key] != second[key]:
            print(f"    first difference in {key!r}")
            break

if failed:
    print("profiles of one workload disagree; see PLAN.md section 12", file=sys.stderr)
    sys.exit(1)
print("every run of the workload produced the same profile")
PY
