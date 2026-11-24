# Benchmarks

Microbenchmarks for the operations RedisJSON performs on `ijson`, plus a deterministic memory referee. They exist
so that a change to `ijson` can be judged on the paths RedisJSON actually exercises, with time and memory
measured together, before it is proposed upstream.

Nothing here changes the library. `cargo bench` builds with the same features RedisJSON uses
(`default-features = false`; add `--features thread_safe` to measure that configuration).

## What is measured

| Group (`--bench callpaths`) | RedisJSON call path | Benchmarks |
|---|---|---|
| `parse` | `JSON.SET` of a document: `IValueDeserSeed::deserialize_compact_objects`, FPHA float arrays | `compact-objects/{yelp,yahoo2,gmaps-place,int-arrays,parser-0000,pass-100,floats-1024,ints-1024}`, `fpha-f32/floats-1024` |
| `serialise` | `JSON.GET`: compact and indented `serde` output of a document | `compact/*`, `indent/*` |
| `lookup` | JSONPath steps and `get`/`get_mut` by key (`json_path` `get_key`, `ivalue_manager` `follow_path`) | `path-chain-yelp`, `business-22-{hit,miss}`, `wide-128-{hit-short-key,hit-long-key,miss-long-key}`, `typed-array-slice-index-f64-x64` (control) |
| `mutation` | In-place edits: `JSON.SET` of a member, `JSON.ARRINSERT`/`ARRPOP`/`ARRTRIM`, `JSON.NUMINCRBY`, `JSON.STRAPPEND` | `object-insert-remove-22`, `array-push-pop-hetero`, `array-insert-front-remove`, `array-trim-typed-1024-to-512`, `num-incrby-{i64,f64}`, `str-append` |
| `typed` | Homogeneous (typed) arrays: build, retag on a heterogeneous push, slice reads | `build-f64-push-1024`, `build-i64-try-extend-16k`, `retag-on-string-push-1024`, `slice-sum-f64-1024` |
| `clone-drop` | `.clone()` of a value (multi-target `JSON.SET`/`MERGE`/`ARRAPPEND` clone their argument per target; `JSON.GET` of subtrees) and dropping a parsed document | `clone/{yelp,gmaps-place,...}`, `drop/*` |
| `cbor` | RDB save and load: `encode`, `decode`, compressed variants, the streaming encoder/decoder | `encode-yelp`, `decode-yelp`, `encode-compressed-yelp`, `decode-compressed-yelp`, `stream-encode-yelp-fresh-encoder`, `stream-decode-yelp` |
| `mem-defrag` | `JSON.DEBUG MEMORY` (`mem_allocated`) and active defrag (`defrag` copy) | `mem_allocated-yelp`, `defrag-copy-yelp` |

`--bench memory` prints, per fixture, the bytes the allocator sees (peak and retained after parse, clone and
CBOR decode, serialise peak, allocation count), `IValue::mem_allocated()` beside the allocator's truth, the CBOR
payload sizes, and `size_of` for `IValue`, `IString`, `INumber`, `IArray`, `IObject`. It is deterministic: run it
once per build, not under criterion.

`examples/mem_attribution.rs` breaks a parsed document's retained bytes down into objects (len/capacity), arrays,
heap numbers and unique heap strings, and shows the gap to `mem_allocated()` (the shared string table, which
`mem_allocated()` excludes by design).

Fixtures under `benches/fixtures/` are minified copies of RedisJSON's `tests/files` and
`tests/benchmarks/datasets` documents (yelp 38 KB, yahoo2 18 KB, a Google Maps place reply, homogeneous integer
arrays, `parser-0000`, `pass-100`); `floats-1024` and `ints-1024` are generated with a fixed seed.

## Running

```bash
# time, one group
cargo bench --no-default-features --bench callpaths -- lookup
# time, everything, saved as a named baseline
cargo bench --no-default-features --bench callpaths -- --save-baseline base
# memory referee (Markdown table; --json for machine-readable output)
cargo bench --no-default-features --bench memory
cargo bench --no-default-features --bench memory -- --json
# attribution of one or more documents
cargo run --release --no-default-features --example mem_attribution -- benches/fixtures/pass-jsonsl-yelp.json
```

Criterion keeps results under `target/criterion/<group>/<benchmark>/`; `--save-baseline <name>` and
`--baseline <name>` compare two builds of the same checkout.

## Comparing two builds (A/B protocol)

The lookup and mutation routines take 15–60 ns, so they are sensitive to code placement: an unrelated change can
move them by 5–25% on one host and not at all on another. The protocol that has held up:

1. Build the baseline and the candidate from the same commit of everything but the change under test, the same
   toolchain (`rust-toolchain.toml`), features, allocator and machine. Pin the process to one core where the OS
   allows it; keep the host otherwise idle.
2. Run them interleaved, base then candidate, at least three pairs, plus one full pair at the end. Take the
   median delta over the pairs; report the base's own run-to-run spread next to it.
3. Re-run anything that moved by more than 2% in a group the change does not touch. If it reproduces on this host,
   check `nm -S` for the moved symbols and, if available, a top-down (TMA) capture on another machine class
   before calling it a regression: identical instruction streams with different IPC are placement, not the
   change.
4. Decide on both time and memory: a candidate that speeds a target up but grows retained or peak bytes, the
   allocation count, or `size_of` any core type is a trade-off to declare, not a win.
5. Keep thresholds explicit and conservative: a target gain of 3% or more on the benchmarks the change is aimed
   at, no untouched benchmark worse by more than 2% after re-runs, memory identical unless declared.

Preserve behaviour RedisJSON depends on while optimising: number formatting, typed-array tagging and retagging,
string interning semantics, CBOR stream compatibility, defrag. Run the unit tests single-threaded:
`cargo test --lib -- --test-threads=1` (the tests share the global string cache).

## For AI agents

If you are an automated agent asked to optimise `ijson` or to evaluate a change to it, this is the contract:

- **Name the RedisJSON call path first.** Every hypothesis must say which command and which code path in
  RedisJSON it serves (table above). A speed-up on an operation RedisJSON never performs is out of scope.
- **Measure, do not estimate.** Run `--bench callpaths` for the affected groups and `--bench memory` for every
  change, on the same machine for base and candidate, following the A/B protocol above. Never carry numbers
  across machines, never compare two tables printed on different days, never report a delta without the base's
  spread.
- **Report time and memory together**, naming machine class (architecture and microarchitecture only), toolchain,
  features, the pairs run, the median delta, the spread, and the memory table diff. If a routine moved in a group
  the change does not touch, say so and what you did to attribute it.
- **Expect placement noise** on the 15–60 ns benchmarks; the protocol's re-run rule exists for it. Do not tune
  against it and do not weaken a benchmark or test to make a candidate pass.
- **Do not change the harness and the candidate in one commit.** A harness fix is its own commit followed by a
  new baseline; otherwise the comparison measures two things.
- **Check behaviour, not just speed.** The unit tests (single-threaded) and the fuzz targets must pass in both
  the default and `thread_safe` builds; the memory referee must show identical retained bytes unless the change
  declares otherwise.
- **Confirm end to end when a change looks good.** The microbenchmarks predict; a RedisJSON built against the
  changed `ijson` (Cargo `[patch]`, no RedisJSON edits) and driven with commands that exercise the path many
  times per command (wildcard and multi-path `JSON.GET`, multi-target `JSON.SET`/`MERGE`/`ARRAPPEND`, wildcard
  `JSON.NUMINCRBY`) shows whether the gain reaches a command. Single-path commands at pipeline 1 are dominated by
  protocol handling and will not move.
