# Benchmarks

Microbenchmarks for the operations RedisJSON performs on `ijson`, a benchmark of the same operations over real
JSON files, and a deterministic memory referee. They exist so that a change to `ijson` can be judged on the paths
RedisJSON actually exercises, with time and memory measured together, before it is proposed upstream.

## Quick start

```bash
benches/fetch-datasets.sh                                  # datasets, quick preset: 165 MB download, 0.57 GB unpacked
cargo bench --no-default-features --bench callpaths        # call-path microbenchmarks on the bundled fixtures
cargo bench --no-default-features --bench corpus           # the same call paths over the fetched files (~2 min)
```

Compare two commits with criterion baselines: `-- --save-baseline base` on the first, `-- --baseline base` on the
second (see [Comparing two builds](#comparing-two-builds-ab-protocol)); results are kept under `target/criterion/`.
Add `--features thread_safe` to measure that build; the default build is the one RedisJSON uses
(`default-features = false`). More:

```bash
cargo bench --no-default-features --bench callpaths -- lookup     # one group (any criterion name filter)
cargo bench --no-default-features --bench corpus -- --quick       # shorter corpus run
cargo bench --no-default-features --bench memory                  # memory referee (Markdown; -- --json)
cargo run --release --no-default-features --example mem_attribution -- benches/fixtures/pass-jsonsl-yelp.json
```

## What is measured

| Group (`--bench callpaths`) | RedisJSON call path | Benchmarks |
|---|---|---|
| `parse` | `JSON.SET` of a document: `IValueDeserSeed::deserialize_compact_objects`, FPHA float arrays | `compact-objects/{yelp,yahoo2,gmaps-place,int-arrays,parser-0000,pass-100,floats-1024,ints-1024,refsnp-400}`, `fpha-f32/floats-1024` |
| `serialise` | `JSON.GET`: compact and indented `serde` output of a document | `compact/*`, `indent/*` |
| `lookup` | JSONPath steps and `get`/`get_mut` by key (`json_path` `get_key`, `ivalue_manager` `follow_path`) | `path-chain-yelp`, `business-22-{hit,miss}`, `small-8-{hit-last,miss}`, `wide-128-{hit-short-key,hit-long-key,miss-long-key}`, `typed-array-slice-index-f64-x64` (control) |
| `mutation` | In-place edits: `JSON.SET` of a member, `JSON.MERGE`, `JSON.ARRINSERT`/`ARRPOP`/`ARRTRIM`, `JSON.NUMINCRBY`, `JSON.STRAPPEND` | `object-insert-remove-22`, `merge-26-into-54-occupied`, `array-push-pop-hetero`, `array-insert-front-remove`, `array-trim-typed-1024-to-512`, `num-incrby-{i64,f64}`, `str-append` |
| `typed` | Homogeneous (typed) arrays: build, retag on a heterogeneous push, slice reads | `build-f64-push-1024`, `build-i64-try-extend-16k`, `retag-on-string-push-1024`, `slice-sum-f64-1024` |
| `clone-drop` | `.clone()` of a value (multi-target `JSON.SET`/`MERGE`/`ARRAPPEND` clone their argument per target; `JSON.GET` of subtrees); dropping a document (overwrite, `JSON.DEL`, expiry) | `clone/*`, `drop/*`, `drop-last-ref/*` (the only reference, so every string leaves the cache) |
| `cbor` | RDB save and load: `encode`, `decode`, compressed variants, the streaming encoder/decoder | `encode-yelp`, `decode-yelp`, `encode-compressed-yelp`, `decode-compressed-yelp`, `stream-encode-yelp-fresh-encoder`, `stream-decode-yelp` |
| `mem-defrag` | `JSON.DEBUG MEMORY` (`mem_allocated`) and active defrag | `mem_allocated-yelp`, `defrag-copy-yelp`, `defrag-reinit-yelp` (after `defrag_start`'s cache reinit), `defrag-cycle-yelp` (reinit timed too) |

`--bench memory` prints, per fixture, the bytes the allocator sees (peak and retained after parse, clone and
CBOR decode, serialise peak, allocation count), `IValue::mem_allocated()` beside the allocator's truth, the CBOR
payload sizes, and `size_of` for `IValue`, `IString`, `INumber`, `IArray`, `IObject`. It is deterministic: run it
once per build, not under criterion.

`--bench corpus` runs RedisJSON's call paths over files fetched by `fetch-datasets.sh`, one key per document (per
line for `.ndjson`): `corpus-parse` (into an empty string cache), `corpus-drop` (last references),
`corpus-serialise` (compact) and `corpus-clone`, after a memory table (retained bytes per file, string cache
reset per file). Default files: `twitter`, `citm_catalog`, `canada`, `gsoc-2018`, `apache_builds`,
`amazon_cellphones.ndjson`, the 768-float embedding NDJSON `mobilenetv3_large-768.ndjson`, and
`ncbi/refsnp-chrMT.ndjson` when fetched. Without datasets it prints the fetch command and exits 0. Knobs:

| Variable | Default | Meaning |
|---|---|---|
| `IJSON_BENCH_DATASETS` | `benches/datasets` | datasets directory |
| `IJSON_BENCH_FILES` | the list above | comma list of paths or `*`/`?` patterns relative to the directory, e.g. `ncbi/*,*/twitter.json` |
| `IJSON_BENCH_MAX_DOCS` | 1000 | NDJSON documents per file |
| `IJSON_BENCH_MAX_BYTES` | 4194304 | input bytes per file (NDJSON stops before it; a larger `.json` file is skipped) |
| `IJSON_BENCH_MEMORY` | 1 | `0` skips the memory table |

`examples/mem_attribution.rs` breaks a parsed document's retained bytes down into objects (len/capacity), arrays,
heap numbers and unique heap strings, and shows the gap to `mem_allocated()` (the shared string table, which
`mem_allocated()` excludes by design).

### Fixtures

`benches/fixtures/` holds minified copies of RedisJSON's `tests/files` and `tests/benchmarks/datasets` documents
(yelp 38 KB, yahoo2 18 KB, a Google Maps place reply, homogeneous integer arrays, `parser-0000`, `pass-100`);
`floats-1024` and `ints-1024` are generated with a fixed seed. `refsnp-400` is
`refsnp-unsupported-400.ndjson`, 400 distinct NCBI dbSNP RefSNP records, one per line, benchmarked as one
`IValue` (one key) per line:

- source: <https://ftp.ncbi.nlm.nih.gov/snp/archive/b157/JSON/refsnp-unsupported.json.bz2> (dbSNP build 157),
  decompressed lines 1–400, unmodified
- 243,806 bytes, sha256 `1b1737c07ffe700e6099e5d7c44fd67a7444e13f3845b4b3590c4b9f9c0b8ce5`
- a US government work, in the public domain ([NCBI policies](https://www.ncbi.nlm.nih.gov/home/about/policies/))

## Datasets

`benches/fetch-datasets.sh [--dir DIR] [quick|standard|all|FAMILY...]` downloads the
[`datasets-v1`](https://s3.amazonaws.com/benchmarks.redislabs/ijson/datasets-v1/ASSETS.sha256) assets from a public S3 mirror into
`benches/datasets` (git-ignored), checks each against [`datasets.sha256`](datasets.sha256) before unpacking, and
skips assets already verified. Needs `curl`, `zstd`, `tar`. `IJSON_DATASETS_MIRROR=<base URL>` reads another copy
of the same assets (the pinned hashes still apply).

| Family | Preset | Source | Download → unpacked | Licence / privacy |
|---|---|---|---|---|
| `simdjson` | quick | `simdjson/simdjson-data` @ `4197c42`: jsonexamples, small, jsonchecker, minefield | 5 MB → 29 MB | no data licence stated: local measurement only |
| `extras` | quick | data files of other simdjson / Lemire repositories, pinned | 8 MB → 112 MB | as the source repositories |
| `floatdata` | quick | `fastfloat/float-data` @ `4e8922d`, plus 768-float embedding NDJSON | 152 MB → 425 MB | no data licence stated: local measurement only |
| `generated` | standard | simdjson's generators (kostya, large_random, amazon, tweets, timeline, growing), fixed seed | 33 MB → 2.1 GB | derived from simdjson-data |
| `ncbi` | standard | NCBI dbSNP RefSNP JSON, build 157 | 54 MB → 1.2 GB | public domain |
| `gharchive` | all | GH Archive hours 2024-11-01-15 and 2024-03-13 12–19 | 549 MB → 8.3 GB | public GitHub user data (logins, commit e-mails): `all` or by name only; never commit or publish it |
| `pison` | all | Pison (PVLDB 2021) evaluation data | 907 MB → 13 GB | no data licence stated: local measurement only |

Presets: `quick` 165 MB → 0.57 GB (default), `standard` 252 MB → 3.9 GB, `all` 1.7 GB → 25 GB.

## Comparing two builds (A/B protocol)

The lookup and mutation routines take 15–60 ns, so they are sensitive to code placement: an unrelated change can
move them by 5–25% on one host and not at all on another, and an unchanged rebuild can move the serialisation
benchmarks by up to ±30%. The protocol that has held up:

1. Build the baseline and the candidate from the same commit of everything but the change under test, the same
   toolchain (`rust-toolchain.toml`), features, allocator and machine. Keep the host otherwise idle; on Linux pin
   the run to one core (`taskset -c 2 cargo bench ...`).
2. Run them interleaved, base then candidate, three pairs, and take the median delta over the pairs; report the
   base's own run-to-run spread next to it:

   ```bash
   for i in 1 2 3; do
     git switch -q --detach BASE && cargo bench --no-default-features --bench callpaths -- --save-baseline base$i
     git switch -q --detach CAND && cargo bench --no-default-features --bench callpaths -- --baseline base$i
   done
   ```

3. Treat 3% as the noise level. Re-run anything that moved by more than 3%, above all in a group the change does
   not touch. If it reproduces, check `nm -S` for the moved symbols and, if available, a top-down (TMA) capture
   before calling it a regression: identical instruction streams with different IPC are placement, not the change.
4. To take placement out of a comparison, build both sides with every function and branch target aligned to
   64 bytes: `RUSTFLAGS="-C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=6"`. Use
   it for both builds or neither, and do not quote its absolute numbers as the default build's. (Background:
   Mytkowicz et al., [Producing Wrong Data Without Doing Anything Obviously Wrong!](https://sape.inf.usi.ch/publications/asplos09),
   ASPLOS 2009; [LLVM benchmarking tips](https://llvm.org/docs/Benchmarking.html).)
5. Decide on both time and memory: a candidate that speeds a target up but grows retained or peak bytes, the
   allocation count, or `size_of` any core type is a trade-off to declare, not a win.
6. Report on four platform classes, one column each: x86-64 Intel, x86-64 AMD, arm64 Neoverse (Linux) and arm64
   Apple M-series (macOS). Mark a platform you did not measure as not measured; never carry a result from one to
   another (cache lines are 64 B or 128 B, pages 4 KB or 16 KB).

Preserve behaviour RedisJSON depends on while optimising: number formatting, typed-array tagging and retagging,
string interning semantics, CBOR stream compatibility, defrag. Run the unit tests single-threaded:
`cargo test --lib -- --test-threads=1` (the tests share the global string cache).

## For AI agents

If you are an automated agent asked to optimise `ijson` or to evaluate a change to it, this is the contract:

- **Name the RedisJSON call path first.** Every hypothesis must say which command and which code path in
  RedisJSON it serves (table above). A speed-up on an operation RedisJSON never performs is out of scope.
- **Measure, do not estimate.** Run `--bench callpaths` for the affected groups, `--bench corpus` when the
  datasets are fetched, and `--bench memory` for every change, on the same machine for base and candidate,
  following the A/B protocol above. Never carry numbers across machines, never compare two tables printed on
  different days, never report a delta without the base's spread.
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
