#!/usr/bin/env bash
# Memory gate: measure the library at the current tip of the base branch and the same tip with this
# checkout's HEAD merged into it, with one harness (the merged tree's `benches/`), and fail if any
# retained or peak byte count grows: at all in the default build, whose counts repeat exactly
# (MEMORY_GATE_MAX_PCT_DEFAULT, default 0), or by more than MEMORY_GATE_MAX_PCT_TS percent in the
# thread_safe build, whose counts vary from run to run (default 5).
#
# Both sides are built from the tip fetched when the gate runs, so the check is always against the
# base branch as it is now, and only HEAD's own changes differ between them: commits that reached
# the base after HEAD branched off are on both sides, in CI and locally alike. Uncommitted changes are
# not measured; commit them first.
#
# usage: .github/scripts/memory-gate.sh [BASE_REF]   (default: $GITHUB_BASE_REF, else master)
#        needs `origin` to point at the repository the base branch lives in, and HEAD's history
#        back to its merge base with the tip (CI: actions/checkout with fetch-depth: 0).
set -euo pipefail

BASE_REF=${1:-${GITHUB_BASE_REF:-master}}
ROOT=$(git rev-parse --show-toplevel)
HEAD_SHA=$(git -C "$ROOT" rev-parse HEAD)
WORK=$(mktemp -d)
cleanup() {
  git -C "$ROOT" worktree remove --force "$WORK/base" >/dev/null 2>&1 || true
  git -C "$ROOT" worktree remove --force "$WORK/head" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

git -C "$ROOT" fetch --quiet --no-tags origin "+refs/heads/$BASE_REF:refs/remotes/origin/$BASE_REF"
tip=$(git -C "$ROOT" rev-parse "origin/$BASE_REF")
git -C "$ROOT" worktree add --quiet --detach "$WORK/base" "$tip"
git -C "$ROOT" worktree add --quiet --detach "$WORK/head" "$tip"
if ! git -C "$WORK/head" -c user.name=memory-gate -c user.email=memory-gate@invalid \
    merge --quiet --no-ff --no-commit "$HEAD_SHA" >/dev/null 2>&1; then
  echo "::error title=Memory gate::HEAD $HEAD_SHA does not merge cleanly into $BASE_REF @ $tip; update the branch"
  exit 1
fi
echo "base: $BASE_REF @ $tip; head: $HEAD_SHA merged into it"

# The base side gets the merged tree's harness. Its own Cargo.toml (the library's dependencies) is
# kept when it already declares the memory bench; a base without it predates the harness, so it
# takes the merged tree's Cargo.toml.
rm -rf "$WORK/base/benches"
cp -R "$WORK/head/benches" "$WORK/base/benches"
if ! grep -q '^name = "memory"' "$WORK/base/Cargo.toml"; then
  cp "$WORK/head/Cargo.toml" "$WORK/base/Cargo.toml"
fi

status=0
# The default build's counts repeat exactly; the thread_safe build's vary with the cache's random
# hash seed, so each side runs MEMORY_GATE_RUNS_TS times (default 5) and the gate uses the median.
for cfg in "--no-default-features" "--features thread_safe"; do
  if [ "$cfg" = "--no-default-features" ]; then
    label=default; runs=1; max=${MEMORY_GATE_MAX_PCT_DEFAULT:-0}
  else
    label=thread_safe; runs=${MEMORY_GATE_RUNS_TS:-5}; max=${MEMORY_GATE_MAX_PCT_TS:-5}
  fi
  base_files=() head_files=()
  for i in $(seq 1 "$runs"); do
    # A target dir per side: both sides are the same crate, so a shared one would let cargo reuse
    # one side's binary for the other.
    (cd "$WORK/base" && CARGO_TARGET_DIR="$ROOT/target/memory-gate-base" \
      cargo bench -q $cfg --bench memory -- --json) > "$WORK/base-$label-$i.json"
    (cd "$WORK/head" && CARGO_TARGET_DIR="$ROOT/target/memory-gate-head" \
      cargo bench -q $cfg --bench memory -- --json) > "$WORK/head-$label-$i.json"
    base_files+=("$WORK/base-$label-$i.json"); head_files+=("$WORK/head-$label-$i.json")
  done
  python3 "$WORK/head/.github/scripts/memory_gate.py" --base "${base_files[@]}" --head "${head_files[@]}" \
    --label "$label" --max-pct "$max" || status=1
done
exit $status
