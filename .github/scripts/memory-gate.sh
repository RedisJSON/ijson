#!/usr/bin/env bash
# Memory gate: run this checkout's memory referee (`benches/memory.rs`) against the library at the
# current tip of the base branch and against this checkout, and fail if any retained or peak byte
# count grows by more than MEMORY_GATE_MAX_PCT percent (default 5). The same harness measures
# both sides, so only the library differs.
#
# usage: .github/scripts/memory-gate.sh [BASE_REF]   (default: $GITHUB_BASE_REF, else master)
set -euo pipefail

BASE_REF=${1:-${GITHUB_BASE_REF:-master}}
ROOT=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d)
trap 'git -C "$ROOT" worktree remove --force "$WORK/base" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT

# Always the branch tip at run time, not the merge base or the commit the PR was opened against.
git -C "$ROOT" fetch --quiet --no-tags origin "+refs/heads/$BASE_REF:refs/remotes/origin/$BASE_REF"
tip=$(git -C "$ROOT" rev-parse "origin/$BASE_REF")
echo "base: $BASE_REF @ $tip; head: $(git -C "$ROOT" rev-parse HEAD)"
git -C "$ROOT" worktree add --quiet --detach "$WORK/base" "$tip"

# The base gets this checkout's harness. Its own Cargo.toml (the library's dependencies) is kept
# when it already declares the memory bench; a base without it predates the harness, so it takes
# this checkout's Cargo.toml.
rm -rf "$WORK/base/benches"
cp -R "$ROOT/benches" "$WORK/base/benches"
if ! grep -q '^name = "memory"' "$WORK/base/Cargo.toml"; then
  cp "$ROOT/Cargo.toml" "$WORK/base/Cargo.toml"
fi

status=0
# The default build's counts repeat exactly; the thread_safe build's vary with the cache's random
# hash seed, so each side runs MEMORY_GATE_RUNS_TS times (default 5) and the gate uses the median.
for cfg in "--no-default-features" "--features thread_safe"; do
  if [ "$cfg" = "--no-default-features" ]; then label=default; runs=1; else label=thread_safe; runs=${MEMORY_GATE_RUNS_TS:-5}; fi
  base_files=() head_files=()
  for i in $(seq 1 "$runs"); do
    # A target dir per side: both sides are the same crate, so a shared one would let cargo reuse
    # one side's binary for the other.
    (cd "$WORK/base" && CARGO_TARGET_DIR="$ROOT/target/memory-gate-base" \
      cargo bench -q $cfg --bench memory -- --json) > "$WORK/base-$label-$i.json"
    (cd "$ROOT" && CARGO_TARGET_DIR="$ROOT/target/memory-gate-head" \
      cargo bench -q $cfg --bench memory -- --json) > "$WORK/head-$label-$i.json"
    base_files+=("$WORK/base-$label-$i.json"); head_files+=("$WORK/head-$label-$i.json")
  done
  python3 "$ROOT/.github/scripts/memory_gate.py" --base "${base_files[@]}" --head "${head_files[@]}" \
    --label "$label" || status=1
done
exit $status
