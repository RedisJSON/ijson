//! Benchmark referee for the RedisJSON call paths.
//!
//! One group per kind of operation RedisJSON performs on `ijson`: parse (`JSON.SET`), serialise
//! (`JSON.GET`), path lookup, in-place mutation, typed arrays, clone and drop, CBOR (RDB save and
//! load), memory accounting and defrag. Argument shapes follow RedisJSON `9ca1c88`
//! (`redis_json/src/ivalue_manager.rs`, `formatter.rs`, `key_value.rs`, `rdb_persistence.rs`,
//! `defrag.rs`, `json_path/src/select_value.rs`), so each benchmark times the call RedisJSON makes.
//! Fixtures under `benches/fixtures/` are copies of RedisJSON's `tests/benchmarks/datasets`, plus
//! `refsnp-unsupported-400.ndjson` (NCBI dbSNP records, one key per line).
//!
//! The string cache is initialised as RedisJSON does at module load: `init_shared_string_cache(true)`,
//! the locked cache (RedisJSON always enables it, for atomic slot migration), so interning a heap
//! string, or dropping its last reference, takes the cache's mutex as it does in RedisJSON.
//!
//! Timing only. Memory is reported by `benches/memory.rs`.

use std::alloc::{alloc, dealloc, Layout};
use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use ijson::{
    decode, decode_compressed, encode, encode_compressed, Defrag, DefragAllocator, FPHAConfig,
    FloatType, IArray, INumber, IObject, IString, IValue, IValueDeserSeed, StreamDecoder,
    StreamEncoder, StreamEncoderConfig, TryExtend,
};
use serde::Serialize;

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

/// Named JSON documents. The first six are RedisJSON benchmark datasets; the rest are
/// deterministic synthetic shapes that RedisJSON's typed-array code paths depend on.
fn fixtures() -> Vec<(&'static str, String)> {
    vec![
        (
            "yelp",
            include_str!("fixtures/pass-jsonsl-yelp.json").to_string(),
        ),
        (
            "yahoo2",
            include_str!("fixtures/pass-jsonsl-yahoo2.json").to_string(),
        ),
        (
            "gmaps-place",
            include_str!("fixtures/api_replies_q5_gmaps_place.json").to_string(),
        ),
        (
            "int-arrays",
            include_str!("fixtures/homogeneous_integer_arrays.json").to_string(),
        ),
        (
            "parser-0000",
            include_str!("fixtures/parser-0000.json").to_string(),
        ),
        (
            "pass-100",
            include_str!("fixtures/pass-100.json").to_string(),
        ),
        ("floats-1024", float_array_json(1024)),
        ("ints-1024", int_array_json(1024)),
    ]
}

/// Key-per-document fixtures: NDJSON, one `IValue` per line, as `JSON.SET` stores one key per
/// document. `refsnp-400` is 400 NCBI dbSNP RefSNP records (public domain; see benches/README.md).
fn ndjson_fixtures() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![(
        "refsnp-400",
        include_str!("fixtures/refsnp-unsupported-400.ndjson")
            .lines()
            .collect(),
    )]
}

fn ndjson_bytes(lines: &[&str]) -> u64 {
    lines.iter().map(|l| l.len() as u64).sum()
}

fn parse_lines(lines: &[&str]) -> Vec<IValue> {
    lines.iter().map(|l| parse(l, None)).collect()
}

/// Deterministic LCG so the fixture is identical on every machine and run.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

fn float_array_json(n: usize) -> String {
    let mut g = Lcg(0x5eed);
    let items: Vec<String> = (0..n)
        .map(|_| format!("{:.6}", (g.next() % 2_000_000) as f64 / 1000.0 - 1000.0))
        .collect();
    format!("[{}]", items.join(","))
}

fn int_array_json(n: usize) -> String {
    let mut g = Lcg(0x1d);
    let items: Vec<String> = (0..n)
        .map(|_| ((g.next() % 200_000) as i64 - 100_000).to_string())
        .collect();
    format!("[{}]", items.join(","))
}

/// Wide object: 64 short keys (inline, <= 7 bytes) and 64 long keys (interned).
fn wide_object() -> IObject {
    let mut o = IObject::new();
    for i in 0..64 {
        o.insert(format!("k{i:02}"), IValue::from(i as i64))
            .unwrap();
        o.insert(format!("long_key_name_{i:03}"), IValue::from(i as i64))
            .unwrap();
    }
    o
}

// ---------------------------------------------------------------------------------------------
// RedisJSON-shaped operations
// ---------------------------------------------------------------------------------------------

/// `IValueManager::from_str` for `Format::JSON`, `limit_depth = true`, optional FPHA type.
fn parse(s: &str, fpha: Option<FloatType>) -> IValue {
    let mut de = serde_json::Deserializer::from_str(s);
    let v = IValueDeserSeed::new(fpha.map(FPHAConfig::new_with_type))
        .deserialize_compact_objects(&mut de)
        .unwrap();
    de.end().unwrap();
    v
}

/// `KeyValue::serialize_object` with default formatting: serde_json's compact serializer.
fn serialise_compact(v: &IValue) -> Vec<u8> {
    serde_json::to_vec(v).unwrap()
}

/// `KeyValue::serialize_object` with `INDENT`: RedisJSON wraps serde_json's `PrettyFormatter`
/// in `RedisJsonFormatter`; the indent-only configuration is equivalent to this.
fn serialise_indent(v: &IValue) -> Vec<u8> {
    let mut out = Vec::new();
    let fmt = serde_json::ser::PrettyFormatter::with_indent(b"  ");
    let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
    v.serialize(&mut ser).unwrap();
    out
}

/// Pass-through defrag allocator: realloc = copy to a fresh allocation (what RedisJSON's
/// `defrag.rs` does through the module API, minus the allocator's own fragmentation logic).
struct CopyDefrag;
impl DefragAllocator for CopyDefrag {
    unsafe fn realloc_ptr<T>(&mut self, ptr: *mut T, layout: Layout) -> *mut T {
        let new_ptr = self.alloc(layout).cast::<T>();
        std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), new_ptr.cast::<u8>(), layout.size());
        self.free(ptr, layout);
        new_ptr
    }
    unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        alloc(layout)
    }
    unsafe fn free<T>(&mut self, ptr: *mut T, layout: Layout) {
        dealloc(ptr as *mut u8, layout);
    }
}

fn init() {
    // As RedisJSON's module load: `init_ijson_shared_string_cache(true)` (the locked cache).
    let _ = ijson::init_shared_string_cache(true);
}

// ---------------------------------------------------------------------------------------------
// Parse
// ---------------------------------------------------------------------------------------------

/// `arr_trim`'s rotate (`ivalue_manager.rs`): every slice type rotated in place.
fn rotate_left(a: &mut IArray, n: usize) {
    use ijson::array::ArraySliceMut::*;
    match a.as_mut_slice() {
        Heterogeneous(s) => s.rotate_left(n),
        I8(s) => s.rotate_left(n),
        U8(s) => s.rotate_left(n),
        I16(s) => s.rotate_left(n),
        U16(s) => s.rotate_left(n),
        F16(s) => s.rotate_left(n),
        BF16(s) => s.rotate_left(n),
        I32(s) => s.rotate_left(n),
        U32(s) => s.rotate_left(n),
        F32(s) => s.rotate_left(n),
        I64(s) => s.rotate_left(n),
        U64(s) => s.rotate_left(n),
        F64(s) => s.rotate_left(n),
    }
}

fn bench_parse(c: &mut Criterion) {
    init();
    let mut g = c.benchmark_group("parse");
    for (name, text) in fixtures() {
        g.throughput(Throughput::Bytes(text.len() as u64));
        // The parsed value is handed back to criterion and dropped outside the timed section: JSON.SET keeps
        // it, and destruction is the `clone-drop/drop` benchmark's job.
        g.bench_with_input(BenchmarkId::new("compact-objects", name), &text, |b, t| {
            b.iter_with_large_drop(|| parse(t, None))
        });
        // `compact-objects` is the warm case: within a batch every parse after the first finds its
        // strings in the cache, as keys sharing a schema do. A `JSON.SET` of new string values takes
        // the miss path: here each parse runs with no other copy alive, so every string is
        // inserted into the cache, and only the parse is timed (the drop that empties the cache
        // again is not).
        g.bench_with_input(
            BenchmarkId::new("compact-objects-cold", name),
            &text,
            |b, t| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let t0 = Instant::now();
                        let v = black_box(parse(t, None));
                        total += t0.elapsed();
                        drop(v);
                    }
                    total
                })
            },
        );
    }
    // JSON.SET with a float type hint (FPHA): the typed-array detection path with a forced type.
    let floats = float_array_json(1024);
    g.throughput(Throughput::Bytes(floats.len() as u64));
    g.bench_with_input(
        BenchmarkId::new("fpha-f32", "floats-1024"),
        &floats,
        |b, t| b.iter_with_large_drop(|| parse(t, Some(FloatType::F32))),
    );
    // One key per line; the batch of documents is dropped outside the timed section.
    for (name, lines) in ndjson_fixtures() {
        g.throughput(Throughput::Bytes(ndjson_bytes(&lines)));
        g.bench_with_input(
            BenchmarkId::new("compact-objects", name),
            &lines,
            |b, ls| {
                b.iter_batched(
                    || Vec::with_capacity(ls.len()),
                    |mut out| {
                        for l in ls {
                            out.push(parse(l, None));
                        }
                        out
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    g.finish();
}

// ---------------------------------------------------------------------------------------------
// Serialise
// ---------------------------------------------------------------------------------------------

fn bench_serialise(c: &mut Criterion) {
    init();
    let mut g = c.benchmark_group("serialise");
    for (name, text) in fixtures() {
        let doc = parse(&text, None);
        g.throughput(Throughput::Bytes(text.len() as u64));
        g.bench_with_input(BenchmarkId::new("compact", name), &doc, |b, d| {
            b.iter(|| black_box(serialise_compact(d)))
        });
        g.bench_with_input(BenchmarkId::new("indent", name), &doc, |b, d| {
            b.iter(|| black_box(serialise_indent(d)))
        });
    }
    // One JSON.GET per key.
    for (name, lines) in ndjson_fixtures() {
        let docs = parse_lines(&lines);
        g.throughput(Throughput::Bytes(ndjson_bytes(&lines)));
        g.bench_with_input(BenchmarkId::new("compact", name), &docs, |b, ds| {
            b.iter(|| {
                for d in ds {
                    black_box(serialise_compact(d));
                }
            })
        });
        g.bench_with_input(BenchmarkId::new("indent", name), &docs, |b, ds| {
            b.iter(|| {
                for d in ds {
                    black_box(serialise_indent(d));
                }
            })
        });
    }
    g.finish();
}

// ---------------------------------------------------------------------------------------------
// Path traversal (object key lookup, array index) as json_path's select_value drives it
// ---------------------------------------------------------------------------------------------

fn bench_lookup(c: &mut Criterion) {
    init();
    let mut g = c.benchmark_group("lookup");
    let yelp = parse(include_str!("fixtures/pass-jsonsl-yelp.json"), None);
    // `$.businesses[0].id`
    g.bench_function("path-chain-yelp", |b| {
        b.iter(|| {
            let v = yelp
                .as_object()
                .unwrap()
                .get("businesses")
                .unwrap()
                .as_array()
                .unwrap()
                .get(0)
                .unwrap()
                .as_object()
                .unwrap()
                .get("id")
                .unwrap();
            black_box(v)
        })
    });
    let wide = wide_object();
    g.bench_function("wide-128-hit-short-key", |b| {
        b.iter(|| black_box(wide.get("k37")))
    });
    g.bench_function("wide-128-hit-long-key", |b| {
        b.iter(|| black_box(wide.get("long_key_name_037")))
    });
    g.bench_function("wide-128-miss-long-key", |b| {
        b.iter(|| black_box(wide.get("long_key_name_999")))
    });
    // A yelp business (28 keys) is above `SMALL_OBJECT_THRESHOLD` (8), so it is stored with a bucket
    // table like `wide-128`; it is the common RedisJSON object shape. The small-object linear-scan path
    // is measured separately below on an 8-key object.
    let biz = yelp
        .as_object()
        .unwrap()
        .get("businesses")
        .unwrap()
        .as_array()
        .unwrap()
        .get(0)
        .unwrap()
        .clone();
    let biz = biz.into_object().unwrap();
    assert_eq!(biz.len(), 28);
    g.bench_function("business-28-hit", |b| {
        b.iter(|| black_box(biz.get("review_count")))
    });
    g.bench_function("business-28-miss", |b| {
        b.iter(|| black_box(biz.get("not_there")))
    });
    // Small object at the threshold (8 keys, no table): the first eight members of the business.
    let mut small = IObject::with_capacity(8).unwrap();
    for (k, v) in biz.iter().take(8) {
        small.insert(k.clone(), v.clone()).unwrap();
    }
    assert_eq!(small.len(), 8);
    let last_key = small.keys().last().unwrap().clone();
    g.bench_function("small-8-hit-last", |b| {
        b.iter(|| black_box(small.get(last_key.as_str())))
    });
    g.bench_function("small-8-miss", |b| {
        b.iter(|| black_box(small.get("not_there")))
    });
    // Array index on a 1024-element typed array. `IArray::get` returns `None` for typed arrays;
    // RedisJSON reads elements through `ArraySliceRef` (`as_slice`), so that is what is timed. The
    // indices go through `black_box` so the reads cannot be hoisted out of the loop. One read is
    // ~0.6 ns, below what the timer can judge, so each iteration performs 64 strided reads.
    let floats = parse(&float_array_json(1024), None);
    let arr = floats.as_array().unwrap();
    let mut idx = 0usize;
    g.bench_function("typed-array-slice-index-f64-x64", |b| {
        b.iter(|| {
            idx = (idx + 331) & 1023;
            let s: &[f64] = arr.as_slice_of::<f64>().unwrap();
            let base = black_box(idx);
            let mut acc = 0.0;
            for k in 0..64 {
                acc += s[(base + k * 97) & 1023];
            }
            black_box(acc)
        })
    });
    g.finish();
}

// ---------------------------------------------------------------------------------------------
// In-place mutation as ivalue_manager.rs performs it
// ---------------------------------------------------------------------------------------------

fn bench_mutation(c: &mut Criterion) {
    init();
    let mut g = c.benchmark_group("mutation");
    let yelp = parse(include_str!("fixtures/pass-jsonsl-yelp.json"), None);
    let biz = yelp
        .as_object()
        .unwrap()
        .get("businesses")
        .unwrap()
        .as_array()
        .unwrap()
        .get(0)
        .unwrap()
        .clone()
        .into_object()
        .unwrap();

    assert_eq!(biz.len(), 28);

    // JSON.SET $.new_field "v" then JSON.DEL $.new_field on the 28-key business. The pair restores
    // the object, so one object is mutated in place: an `iter_batched` setup clone would dominate the
    // process (cloning and dropping the 28-key object costs several times the routine) and leave the
    // routine's cache state to the setup.
    let mut biz_mut = biz.clone();
    g.bench_function("object-insert-remove-28", |b| {
        b.iter(|| {
            biz_mut.insert("new_field", IValue::from("v")).unwrap();
            black_box(biz_mut.remove("new_field"))
        })
    });
    // JSON.MERGE of a 26-key patch into an object that already has every key (ivalue_manager.rs
    // `merge()`: `map.entry(key.as_str())` then overwrite). After the first command every entry is
    // occupied, so this times 26 occupied-entry lookups plus 26 scalar replacements; the object is
    // mutated in place and keeps its shape (28 + 26 = 54 keys).
    let patch_keys: Vec<String> = (b'a'..=b'z').map(|c| (c as char).to_string()).collect();
    let mut merged = biz.clone();
    for k in &patch_keys {
        merged.insert(k.as_str(), IValue::from(0i64)).unwrap();
    }
    let mut tick = 0i64;
    g.bench_function("merge-26-into-54-occupied", |b| {
        b.iter(|| {
            tick += 1;
            for k in &patch_keys {
                *merged.entry(k.as_str()).unwrap().or_insert(IValue::NULL) = IValue::from(tick);
            }
            black_box(merged.len())
        })
    });
    // JSON.ARRAPPEND / JSON.ARRPOP on the heterogeneous businesses array (10 objects)
    // Both pairs restore the array, so they mutate one array in place (same reasoning as above: a
    // setup clone of 10 objects costs ~10 µs against a ~60 ns routine).
    let businesses = yelp
        .as_object()
        .unwrap()
        .get("businesses")
        .unwrap()
        .clone()
        .into_array()
        .unwrap();
    let mut arr_push = businesses.clone();
    g.bench_function("array-push-pop-hetero", |b| {
        b.iter(|| {
            arr_push.push(IValue::from("appended")).unwrap();
            black_box(arr_push.pop())
        })
    });
    // JSON.ARRINSERT at index 0 then JSON.ARRPOP 0
    let mut arr_front = businesses.clone();
    g.bench_function("array-insert-front-remove", |b| {
        b.iter(|| {
            arr_front.insert(0, IValue::from("front")).unwrap();
            black_box(arr_front.remove(0))
        })
    });
    // JSON.ARRTRIM: truncate a 1024-element typed array to half. `LargeInput` keeps a thousandth
    // of each sample's 8 KB clones alive at once (`SmallInput` keeps a tenth: gigabytes of resident
    // memory at this routine's speed).
    let ints = parse(&int_array_json(1024), None).into_array().unwrap();
    g.bench_function("array-trim-typed-1024-to-512", |b| {
        b.iter_batched(
            || ints.clone(),
            |mut a| {
                a.truncate(512);
                a
            },
            BatchSize::LargeInput,
        )
    });
    // JSON.ARRTRIM keeping the middle half (start 256): `arr_trim` rotates the slice left by
    // `start`, then truncates, as when a list is capped to its last N elements. From index 0 (above)
    // the rotate does nothing.
    g.bench_function("array-trim-typed-1024-middle-512", |b| {
        b.iter_batched(
            || ints.clone(),
            |mut a| {
                rotate_left(&mut a, 256);
                a.truncate(512);
                a
            },
            BatchSize::LargeInput,
        )
    });
    // JSON.NUMINCRBY on an integer field: read, add, replace (number.rs / ivalue_manager.rs)
    // One object mutated in place: repeated INCRBY on the same key is what RedisJSON sees, the value
    // drifting is realistic, and a setup clone would hand the routine an object freshly laid out by
    // the clone code each iteration, so a change to `clone` would move this benchmark.
    let mut biz_i64 = biz.clone();
    g.bench_function("num-incrby-i64", |b| {
        b.iter(|| {
            let v = biz_i64.get_mut("review_count").unwrap();
            let n = v.as_number().unwrap().to_i64().unwrap();
            *v = IValue::from(INumber::from(n.wrapping_add(1)));
        })
    });
    // JSON.NUMINCRBY on a float field
    let mut biz_f64 = biz.clone();
    g.bench_function("num-incrby-f64", |b| {
        b.iter(|| {
            let v = biz_f64.get_mut("latitude").unwrap();
            let n = v.as_number().unwrap().to_f64().unwrap();
            *v = IValue::from(n + 0.5);
        })
    });
    // JSON.STRAPPEND (ivalue_manager.rs `str_append`): concatenate the field's string and the
    // argument, intern the result and replace the field's `IString` with it, dropping the old one.
    // The field is rebuilt from a fixed base so it does not grow across iterations, and the suffix
    // alternates between two of the same length: each iteration interns a string the cache does not
    // hold and drops the previous one at its last reference, removing it from the cache, as every
    // STRAPPEND does. With one suffix every iteration after the first would be a cache hit that
    // replaces a string with itself.
    let mut biz_str = biz.clone();
    let city_base = biz_str
        .get("city")
        .unwrap()
        .as_string()
        .unwrap()
        .as_str()
        .to_owned();
    let suffixes = [" (appended-a)", " (appended-b)"];
    let mut turn = 0usize;
    g.bench_function("str-append", |b| {
        b.iter(|| {
            turn ^= 1;
            let v = biz_str.get_mut("city").unwrap().as_string_mut().unwrap();
            let new_str = [black_box(city_base.as_str()), suffixes[turn]].concat();
            *v = IString::intern(&new_str);
            black_box(new_str.len())
        })
    });
    g.finish();
}

// ---------------------------------------------------------------------------------------------
// Typed numeric arrays: build, extend, retag, slice read
// ---------------------------------------------------------------------------------------------

fn bench_typed_arrays(c: &mut Criterion) {
    init();
    let mut g = c.benchmark_group("typed");
    let mut lcg = Lcg(7);
    let floats: Vec<f64> = (0..1024)
        .map(|_| (lcg.next() % 1_000_000) as f64 / 7.0)
        .collect();
    // 16 K elements: at 1 K the bulk copy takes ~100 ns and allocator jitter dominates the spread.
    let ints: Vec<i64> = (0..16384)
        .map(|_| (lcg.next() % 1_000_000) as i64 - 500_000)
        .collect();

    // The built array is returned and dropped after the batch, outside the timer, as parse and
    // clone do; `LargeInput` keeps only a thousandth of a sample's arrays alive at once.
    g.bench_function("build-f64-push-1024", |b| {
        b.iter_batched(
            || (),
            |()| {
                let mut a = IArray::new();
                for &f in &floats {
                    a.push(f).unwrap();
                }
                a
            },
            BatchSize::LargeInput,
        )
    });
    g.bench_function("build-i64-try-extend-16k", |b| {
        b.iter_batched(
            || (),
            |()| {
                let mut a = IArray::new();
                a.try_extend(ints.iter().copied()).unwrap();
                a
            },
            BatchSize::LargeInput,
        )
    });
    let typed = parse(&float_array_json(1024), None).into_array().unwrap();
    assert!(
        typed.as_slice().is_typed(),
        "fixture must parse as a typed array"
    );
    // `LargeInput`: each output is a 1025-element heterogeneous array; `SmallInput` would keep a
    // tenth of a sample's outputs alive at once (hundreds of MB).
    g.bench_function("retag-on-string-push-1024", |b| {
        b.iter_batched(
            || typed.clone(),
            |mut a| {
                a.push(IValue::from("not a number")).unwrap();
                a
            },
            BatchSize::LargeInput,
        )
    });
    g.bench_function("slice-sum-f64-1024", |b| {
        b.iter(|| {
            let s: &[f64] = typed.as_slice_of::<f64>().unwrap();
            black_box(s.iter().sum::<f64>())
        })
    });
    g.finish();
}

// ---------------------------------------------------------------------------------------------
// Clone and drop
// ---------------------------------------------------------------------------------------------

fn bench_clone_drop(c: &mut Criterion) {
    init();
    let mut g = c.benchmark_group("clone-drop");
    for (name, text) in fixtures()
        .into_iter()
        .filter(|(n, _)| matches!(*n, "yelp" | "gmaps-place" | "floats-1024"))
    {
        // A typed array holds no strings, so its last-reference drop is `drop`'s work; at ~15 ns
        // the two `Instant` reads per iteration would dominate it. Only documents with strings get
        // a `drop-last-ref` variant.
        let has_strings = name != "floats-1024";
        // RedisJSON frees documents it owns (an overwriting JSON.SET, JSON.DEL, expiry): the
        // last reference to every heap string goes, so each one is unregistered from the string
        // cache. `drop` below cannot see that cost: its batch of identical parses, and `doc`,
        // share every interned string. Measured first, while no other copy of the fixture is
        // alive; only the drop is timed.
        if has_strings {
            g.bench_with_input(BenchmarkId::new("drop-last-ref", name), &text, |b, t| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let v = parse(t, None);
                        let t0 = Instant::now();
                        drop(black_box(v));
                        total += t0.elapsed();
                    }
                    total
                })
            });
        }
        let doc = parse(&text, None);
        // Only the clone is timed: `iter` would drop each clone inside the timed loop, adding the
        // `drop` benchmark's cost to this one. Batched, the clones are dropped after each batch;
        // `LargeInput` keeps a thousandth of a sample alive at once (`SmallInput`, which
        // `iter_with_large_drop` uses, keeps a tenth: gigabytes of 8 KB typed-array clones).
        g.bench_with_input(BenchmarkId::new("clone", name), &doc, |b, d| {
            b.iter_batched(|| (), |()| d.clone(), BatchSize::LargeInput)
        });
        // Drop a freshly parsed value, not a clone: a clone's memory layout is set by the clone code,
        // so a change to `clone` would move the `drop` numbers. Parsing in setup is slower but untimed.
        g.bench_with_input(BenchmarkId::new("drop", name), &text, |b, t| {
            b.iter_batched(|| parse(t, None), drop, BatchSize::SmallInput)
        });
    }
    // The same three over one key per line. Keys share strings with each other (field names), so
    // `drop-last-ref` frees each shared string only at its last key.
    for (name, lines) in ndjson_fixtures() {
        g.bench_with_input(BenchmarkId::new("drop-last-ref", name), &lines, |b, ls| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut docs = parse_lines(ls);
                    let t0 = Instant::now();
                    for d in docs.drain(..) {
                        drop(black_box(d));
                    }
                    total += t0.elapsed();
                }
                total
            })
        });
        let docs = parse_lines(&lines);
        // The clones are collected into a vector allocated in setup and dropped after each batch,
        // outside the timed section.
        g.bench_with_input(BenchmarkId::new("clone", name), &docs, |b, ds| {
            b.iter_batched(
                || Vec::with_capacity(ds.len()),
                |mut out| {
                    out.extend(ds.iter().cloned());
                    out
                },
                BatchSize::SmallInput,
            )
        });
        g.bench_with_input(BenchmarkId::new("drop", name), &lines, |b, ls| {
            b.iter_batched(
                || parse_lines(ls),
                |mut docs| {
                    docs.drain(..).for_each(drop);
                    docs
                },
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

// ---------------------------------------------------------------------------------------------
// CBOR stream encode/decode (rdb_persistence.rs), plain and zstd
// ---------------------------------------------------------------------------------------------

fn bench_cbor(c: &mut Criterion) {
    init();
    let mut g = c.benchmark_group("cbor");
    let text = include_str!("fixtures/pass-jsonsl-yelp.json");
    let doc = parse(text, None);
    g.throughput(Throughput::Bytes(text.len() as u64));
    let plain = encode(&doc);
    let compressed = encode_compressed(&doc);
    g.bench_function("encode-yelp", |b| b.iter(|| black_box(encode(&doc))));
    // Decoded values are dropped outside the timed section (RDB load keeps them).
    g.bench_function("decode-yelp", |b| {
        b.iter_with_large_drop(|| decode(&plain).unwrap())
    });
    g.bench_function("encode-compressed-yelp", |b| {
        b.iter(|| black_box(encode_compressed(&doc)))
    });
    g.bench_function("decode-compressed-yelp", |b| {
        b.iter_with_large_drop(|| decode_compressed(&compressed).unwrap())
    });
    // RDB stream: one encoder per save, defining strings once (`local_config` with budget on).
    // The encoder and decoder are built in setup and handed back with the output, so their
    // teardown (the encoder's string table, the decoder's definition table) runs after the batch,
    // outside the timer, like every other output in the suite.
    let cfg = StreamEncoderConfig {
        table_budget_bytes: usize::MAX,
    };
    g.bench_function("stream-encode-yelp-fresh-encoder", |b| {
        b.iter_batched(
            || (StreamEncoder::new(cfg), Vec::with_capacity(64 * 1024)),
            |(mut enc, mut out)| {
                black_box(enc.encode_into(&mut out, &doc));
                (enc, out)
            },
            BatchSize::SmallInput,
        )
    });
    let mut enc = StreamEncoder::new(cfg);
    let mut stream_payload = Vec::new();
    enc.encode_into(&mut stream_payload, &doc);
    g.bench_function("stream-decode-yelp", |b| {
        b.iter_batched(
            StreamDecoder::new,
            |mut dec| {
                let v = dec.decode(&stream_payload, true).unwrap();
                (dec, v)
            },
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

// ---------------------------------------------------------------------------------------------
// Memory accounting and defrag (defrag.rs, JSON.DEBUG MEMORY)
// ---------------------------------------------------------------------------------------------

fn bench_memory_defrag(c: &mut Criterion) {
    init();
    let mut g = c.benchmark_group("mem-defrag");
    let doc = parse(include_str!("fixtures/pass-jsonsl-yelp.json"), None);
    g.bench_function("mem_allocated-yelp", |b| {
        b.iter(|| black_box(doc.mem_allocated()))
    });
    // RedisJSON's `defrag_start` calls `reinit_shared_string_cache()` before a defrag cycle, so each
    // string's re-intern misses and allocates a new copy in the new cache. Without the reinit the
    // copy's strings, shared with `doc`, would all be cache hits and no string would move. The cache
    // is emptied in setup, one defrag per batch so each one starts from an empty cache. The old
    // strings stay referenced by `doc`, as when other keys share them, so they are not freed here;
    // `defrag-reinit-yelp` below times the unshared case. Once the cache has been replaced, `doc`'s
    // strings no longer have entries, so the benchmarks that read `doc` run before this one. (The
    // `thread_safe` build's `IString::defrag` keeps each string where it is.)
    g.bench_function("defrag-copy-yelp", |b| {
        b.iter_batched(
            || {
                ijson::reinit_shared_string_cache();
                doc.clone()
            },
            |d| black_box(d.defrag(&mut CopyDefrag)),
            BatchSize::PerIteration,
        )
    });
    // The same defrag on the only reference to a freshly parsed document: each string is
    // re-interned into the empty cache and each old copy, at its last reference, finds the cache
    // pointing at its replacement and is freed. Only the defrag is timed.
    let text = include_str!("fixtures/pass-jsonsl-yelp.json");
    g.bench_function("defrag-reinit-yelp", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let v = parse(text, None);
                ijson::reinit_shared_string_cache();
                let t0 = Instant::now();
                let d = black_box(v.defrag(&mut CopyDefrag));
                total += t0.elapsed();
                drop(d);
            }
            total
        })
    });
    // The same cycle with the reinit inside the timed section: `defrag_start` runs once per cycle,
    // so a change that moves work into the reinit (presizing the new cache, for example) is
    // charged for it here and not in `defrag-reinit-yelp`.
    g.bench_function("defrag-cycle-yelp", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let v = parse(text, None);
                let t0 = Instant::now();
                ijson::reinit_shared_string_cache();
                let d = black_box(v.defrag(&mut CopyDefrag));
                total += t0.elapsed();
                drop(d);
            }
            total
        })
    });
    g.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
}

criterion_group! {
    name = referee;
    config = config();
    targets = bench_parse, bench_serialise, bench_lookup, bench_mutation, bench_typed_arrays,
              bench_clone_drop, bench_cbor, bench_memory_defrag
}
criterion_main!(referee);
