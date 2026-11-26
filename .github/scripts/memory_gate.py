#!/usr/bin/env python3
"""Compare two `cargo bench --bench memory -- --json` outputs and fail on memory growth.

usage: memory_gate.py --base B1.json [B2.json ...] --head H1.json [H2.json ...]
                      [--max-pct 5] [--label NAME]

Gated figures are the memory a value or call path takes: `size_of` of the core types and, per
fixture, every `*_retained` and `*_peak` byte count plus `ijson_mem_allocated`. The counts come from
a counting allocator. In the default build they are exact and repeat run to run, so one run per side
is enough. The `thread_safe` build's sharded string cache takes a random hash seed, so its tables
grow differently between runs (up to ~13 % on the 1.3 KB fixture); with several runs per side each
figure is the median, which repeats to within ~1.5 %. A figure
that grows by more than --max-pct percent over the base fails the gate. Writes a Markdown table to
stdout (and to $GITHUB_STEP_SUMMARY when set).
"""
import argparse
import json
import os
import statistics
import sys


def gated(key):
    return key.endswith("_retained") or key.endswith("_peak") or key == "ijson_mem_allocated"


def flatten(doc):
    out = {}
    for name, size in doc.get("size_of", []):
        out[("size_of", name)] = size
    for row in doc.get("rows", []):
        for key, value in row.items():
            if gated(key) and isinstance(value, (int, float)):
                out[(row["fixture"], key)] = value
    return out


def median(paths):
    runs = [flatten(json.load(open(p))) for p in paths]
    keys = set().union(*runs)
    return {k: statistics.median([r[k] for r in runs if k in r]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--head", nargs="+", required=True)
    ap.add_argument("--max-pct", type=float, default=float(os.environ.get("MEMORY_GATE_MAX_PCT", 5)))
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    base, head = median(a.base), median(a.head)

    lines = [f"### Memory gate{(' — ' + a.label) if a.label else ''}: HEAD merged into the base branch tip vs the tip "
             f"(fail above +{a.max_pct:g} %; "
             f"median of {len(a.base)} base and {len(a.head)} head runs)", "",
             "| Fixture | Figure | Base | Head | Δ | |", "|---|---|---:|---:|---:|---|"]
    failed, changed = [], 0
    for key in sorted(set(base) | set(head)):
        b, h = base.get(key), head.get(key)
        if b is None or h is None:
            lines.append(f"| {key[0]} | {key[1]} | {b if b is not None else '—'} | "
                         f"{h if h is not None else '—'} | | {'new' if b is None else 'removed'} |")
            continue
        if b == h:
            continue
        changed += 1
        pct = (h - b) * 100.0 / b if b else float("inf")
        bad = pct > a.max_pct
        if bad:
            failed.append((key, pct))
        lines.append(f"| {key[0]} | {key[1]} | {b:,.0f} | {h:,.0f} | {pct:+.2f} % | "
                     f"{'**FAIL**' if bad else ('lower' if pct < 0 else 'ok')} |")
    common = len(set(base) & set(head))
    lines.append("")
    lines.append(f"{common} figures compared, {common - changed} unchanged, {changed} changed, "
                 f"{len(failed)} above +{a.max_pct:g} %.")
    report = "\n".join(lines) + "\n"
    sys.stdout.write(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(report + "\n")
    if failed:
        for (fixture, figure), pct in failed:
            print(f"::error title=Memory gate::{fixture} {figure} grew {pct:+.2f} % "
                  f"(limit +{a.max_pct:g} %)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
