//! Benchmark referee for the RedisJSON call paths.
//!
//! One group per call path in json-agent-workspace `docs/USAGE-PATTERN.md` §"Call paths to
//! benchmark". Argument shapes follow RedisJSON `9ca1c88` (`redis_json/src/ivalue_manager.rs`,
//! `formatter.rs`, `key_value.rs`, `rdb_persistence.rs`, `json_path/src/select_value.rs`).
//! Fixtures under `benches/fixtures/` are copies of RedisJSON's `tests/benchmarks/datasets`.
//!
//! The string cache is initialised exactly as OSS RedisJSON does at module load:
//! `init_shared_string_cache(is_bigredis)` with `is_bigredis == false`.
//!
//! Timing only. Memory is reported by `benches/memory.rs`.

use std::alloc::{alloc, dealloc, Layout};
use std::hint::black_box;
use std::time::Duration;

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
    // OSS RedisJSON: `init_ijson_shared_string_cache(is_bigredis = false)`.
    let _ = ijson::init_shared_string_cache(false);
}

// ---------------------------------------------------------------------------------------------
// §1 Parse
// ---------------------------------------------------------------------------------------------

fn bench_parse(c: &mut Criterion) {
    init();
    let mut g = c.benchmark_group("parse");
    for (name, text) in fixtures() {
        g.throughput(Throughput::Bytes(text.len() as u64));
        g.bench_with_input(BenchmarkId::new("compact-objects", name), &text, |b, t| {
            b.iter(|| black_box(parse(t, None)))
        });
    }
    // JSON.SET with a float type hint (FPHA): the typed-array detection path with a forced type.
    let floats = float_array_json(1024);
    g.throughput(Throughput::Bytes(floats.len() as u64));
    g.bench_with_input(
        BenchmarkId::new("fpha-f32", "floats-1024"),
        &floats,
        |b, t| b.iter(|| black_box(parse(t, Some(FloatType::F32)))),
    );
    g.finish();
}

// ---------------------------------------------------------------------------------------------
// §2 Serialise
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
    g.finish();
}

// ---------------------------------------------------------------------------------------------
// §3 Path traversal (object key lookup, array index) as json_path's select_value drives it
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
    // Small object (< table threshold): the common RedisJSON shape (yelp business has ~22 keys,
    // yahoo2 result ~10). Use the first yelp business.
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
    g.bench_function("business-22-hit", |b| {
        b.iter(|| black_box(biz.get("review_count")))
    });
    g.bench_function("business-22-miss", |b| {
        b.iter(|| black_box(biz.get("not_there")))
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
// §4 In-place mutation as ivalue_manager.rs performs it
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

    // JSON.SET $.new_field "v" then JSON.DEL $.new_field on a ~22-key object. The pair restores the
    // object, so one object is mutated in place: an `iter_batched` setup clone would dominate the
    // process (clone + drop of a ~22-key object is ~1 µs against a ~0.5 µs routine) and leave the
    // routine's cache state to the setup.
    let mut biz_mut = biz.clone();
    g.bench_function("object-insert-remove-22", |b| {
        b.iter(|| {
            biz_mut.insert("new_field", IValue::from("v")).unwrap();
            black_box(biz_mut.remove("new_field"))
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
    // JSON.ARRTRIM: truncate a 1024-element typed array to half. The setup clone and the drop of the
    // returned array are outside the timed section (`iter_batched` drops its outputs after timing).
    let ints = parse(&int_array_json(1024), None).into_array().unwrap();
    g.bench_function("array-trim-typed-1024-to-512", |b| {
        b.iter_batched(
            || ints.clone(),
            |mut a| {
                a.truncate(512);
                a
            },
            BatchSize::SmallInput,
        )
    });
    // JSON.NUMINCRBY on an integer field: read, add, replace (number.rs / ivalue_manager.rs)
    // One object mutated in place: repeated INCRBY on the same key is what RedisJSON sees, the value
    // drifting is realistic, and a setup clone would make the routine's cache state depend on the
    // clone implementation (harness r4).
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
    // JSON.STRAPPEND: build the appended string from the field's current value and re-intern it; the
    // field is rewritten from a fixed base each time so the string does not grow across iterations.
    let mut biz_str = biz.clone();
    let city_base = biz_str
        .get("city")
        .unwrap()
        .as_string()
        .unwrap()
        .as_str()
        .to_owned();
    g.bench_function("str-append", |b| {
        b.iter(|| {
            let v = biz_str.get_mut("city").unwrap();
            let mut s = String::with_capacity(city_base.len() + 11);
            s.push_str(black_box(&city_base));
            s.push_str(" (appended)");
            *v = IValue::from(IString::from(s));
        })
    });
    g.finish();
}

// ---------------------------------------------------------------------------------------------
// §5 Typed numeric arrays: build, extend, retag, slice read
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

    g.bench_function("build-f64-push-1024", |b| {
        b.iter(|| {
            let mut a = IArray::new();
            for &f in &floats {
                a.push(f).unwrap();
            }
            black_box(a)
        })
    });
    g.bench_function("build-i64-try-extend-16k", |b| {
        b.iter(|| {
            let mut a = IArray::new();
            a.try_extend(ints.iter().copied()).unwrap();
            black_box(a)
        })
    });
    let typed = parse(&float_array_json(1024), None).into_array().unwrap();
    assert!(
        typed.as_slice().is_typed(),
        "fixture must parse as a typed array"
    );
    g.bench_function("retag-on-string-push-1024", |b| {
        b.iter_batched(
            || typed.clone(),
            |mut a| {
                a.push(IValue::from("not a number")).unwrap();
                a
            },
            BatchSize::SmallInput,
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
// §6 Clone and drop
// ---------------------------------------------------------------------------------------------

fn bench_clone_drop(c: &mut Criterion) {
    init();
    let mut g = c.benchmark_group("clone-drop");
    for (name, text) in fixtures()
        .into_iter()
        .filter(|(n, _)| matches!(*n, "yelp" | "gmaps-place" | "floats-1024"))
    {
        let doc = parse(&text, None);
        // The clone is handed back to criterion and dropped outside the timed section: RedisJSON stores the
        // clone on its target, and the destructor is the `drop` benchmark's job.
        g.bench_with_input(BenchmarkId::new("clone", name), &doc, |b, d| {
            b.iter_with_large_drop(|| d.clone())
        });
        // Drop a freshly parsed value, not a clone: the clone implementation must not decide the
        // cache state the drop runs in (harness r4). Parsing in setup is slower but untimed.
        g.bench_with_input(BenchmarkId::new("drop", name), &text, |b, t| {
            b.iter_batched(|| parse(t, None), drop, BatchSize::SmallInput)
        });
    }
    g.finish();
}

// ---------------------------------------------------------------------------------------------
// §7 CBOR stream encode/decode (rdb_persistence.rs), plain and zstd
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
    g.bench_function("decode-yelp", |b| {
        b.iter(|| black_box(decode(&plain).unwrap()))
    });
    g.bench_function("encode-compressed-yelp", |b| {
        b.iter(|| black_box(encode_compressed(&doc)))
    });
    g.bench_function("decode-compressed-yelp", |b| {
        b.iter(|| black_box(decode_compressed(&compressed).unwrap()))
    });
    // RDB stream: one encoder per save, defining strings once (`local_config` with budget on).
    let cfg = StreamEncoderConfig {
        table_budget_bytes: usize::MAX,
    };
    g.bench_function("stream-encode-yelp-fresh-encoder", |b| {
        b.iter_batched(
            || (StreamEncoder::new(cfg), Vec::with_capacity(64 * 1024)),
            |(mut enc, mut out)| {
                black_box(enc.encode_into(&mut out, &doc));
                out
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
            |mut dec| black_box(dec.decode(&stream_payload, true).unwrap()),
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

// ---------------------------------------------------------------------------------------------
// §8 Memory accounting and defrag (defrag.rs, JSON.DEBUG MEMORY)
// ---------------------------------------------------------------------------------------------

fn bench_memory_defrag(c: &mut Criterion) {
    init();
    let mut g = c.benchmark_group("mem-defrag");
    let doc = parse(include_str!("fixtures/pass-jsonsl-yelp.json"), None);
    g.bench_function("mem_allocated-yelp", |b| {
        b.iter(|| black_box(doc.mem_allocated()))
    });
    g.bench_function("defrag-copy-yelp", |b| {
        b.iter_batched(
            || doc.clone(),
            |d| black_box(d.defrag(&mut CopyDefrag)),
            BatchSize::SmallInput,
        )
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
