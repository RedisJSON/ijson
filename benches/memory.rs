//! Memory referee: retained and peak bytes per fixture and per call path, plus the size of the
//! core types. Deterministic (no timing); run once per build.
//!
//! Counts come from `mockalloc` wrapping the system allocator in this binary. `ijson`'s own
//! `IValue::mem_allocated()` is printed beside the measured retained bytes so the accounting
//! RedisJSON reports through `JSON.DEBUG MEMORY` can be checked against the allocator's truth.
//!
//! Usage: `cargo bench --no-default-features --bench memory` (prints Markdown) or add `--json`.

use std::alloc::System;
use std::mem::size_of;

use ijson::{
    decode, encode, encode_compressed, FPHAConfig, FloatType, IArray, INumber, IObject, IString,
    IValue, IValueDeserSeed, StreamEncoder, StreamEncoderConfig,
};
use mockalloc::{record_allocs, AllocInfo, Mockalloc};

#[global_allocator]
static ALLOCATOR: Mockalloc<System> = Mockalloc(System);

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

/// NDJSON fixtures, one RedisJSON key (one `IValue`) per line.
fn ndjson_fixtures() -> Vec<(&'static str, &'static str)> {
    vec![(
        "refsnp-400",
        include_str!("fixtures/refsnp-unsupported-400.ndjson"),
    )]
}

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

fn parse(s: &str, fpha: Option<FloatType>) -> IValue {
    let mut de = serde_json::Deserializer::from_str(s);
    let v = IValueDeserSeed::new(fpha.map(FPHAConfig::new_with_type))
        .deserialize_compact_objects(&mut de)
        .unwrap();
    de.end().unwrap();
    v
}

/// Run `f`, keep its product alive, and report (peak bytes, retained bytes, allocations).
fn measure<T>(f: impl FnOnce() -> T) -> (T, AllocInfo) {
    let mut slot: Option<T> = None;
    let info = record_allocs(|| slot = Some(f()));
    (slot.unwrap(), info)
}

#[derive(serde::Serialize)]
struct Row {
    fixture: &'static str,
    json_bytes: usize,
    parse_peak: u64,
    parse_retained: u64,
    parse_allocs: u64,
    ijson_mem_allocated: f64,
    clone_peak: u64,
    clone_retained: u64,
    serialise_compact_peak: u64,
    cbor_bytes: usize,
    cbor_compressed_bytes: usize,
    cbor_stream_bytes: usize,
    cbor_decode_retained: u64,
}

fn main() {
    let json = std::env::args().any(|a| a == "--json");
    // As RedisJSON's module load: `init_ijson_shared_string_cache(true)` (the locked cache).
    let _ = ijson::init_shared_string_cache(true);

    let mut rows = Vec::new();
    for (fixture, text) in fixtures() {
        // A parse of the same fixture first, dropped: the string cache's table grows to hold this
        // fixture's strings and, in the default build, keeps that capacity after they are dropped,
        // so `parse retained` below counts the document alone, whatever fixture ran before. (The
        // `thread_safe` build's cache shrinks a shard when it empties, so there it still includes
        // the table growth.)
        drop(parse(&text, None));
        let (doc, p) = measure(|| parse(&text, None));
        // Read the self-accounting before any other copy exists: shared strings are divided among
        // live references, so a live clone or decoded copy would halve their attributed bytes.
        let ijson_mem_allocated = doc.mem_allocated();
        let (clone, c) = measure(|| doc.clone());
        drop(clone);
        let (_out, s) = measure(|| serde_json::to_vec(&doc).unwrap());
        let cbor = encode(&doc);
        let cborz = encode_compressed(&doc);
        let mut enc = StreamEncoder::new(StreamEncoderConfig {
            table_budget_bytes: usize::MAX,
        });
        let mut stream = Vec::new();
        enc.encode_into(&mut stream, &doc);
        drop(enc);
        // Decode as an RDB load does, with no other copy of the document alive: every string is
        // allocated, not found in the cache.
        drop(doc);
        let (_decoded, d) = measure(|| decode(&cbor).unwrap());
        rows.push(Row {
            fixture,
            json_bytes: text.len(),
            parse_peak: p.peak_mem(),
            parse_retained: p.mem_leaked(),
            parse_allocs: p.num_allocs(),
            ijson_mem_allocated,
            clone_peak: c.peak_mem(),
            clone_retained: c.mem_leaked(),
            serialise_compact_peak: s.peak_mem(),
            cbor_bytes: cbor.len(),
            cbor_compressed_bytes: cborz.len(),
            cbor_stream_bytes: stream.len(),
            cbor_decode_retained: d.mem_leaked(),
        });
    }
    // Key-per-document fixtures: every figure covers all the keys (the `Vec` holding them is
    // allocated outside the measured window), except serialise peak, the largest single JSON.GET.
    // The CBOR stream is one encoder over all keys, as one RDB save.
    for (fixture, text) in ndjson_fixtures() {
        let lines: Vec<&str> = text.lines().collect();
        // Untimed warm parse, as above.
        drop(lines.iter().map(|l| parse(l, None)).collect::<Vec<_>>());
        let mut docs = Vec::with_capacity(lines.len());
        let ((), p) = measure(|| docs.extend(lines.iter().map(|l| parse(l, None))));
        let ijson_mem_allocated = docs.iter().map(IValue::mem_allocated).sum();
        let mut clones = Vec::with_capacity(docs.len());
        let ((), c) = measure(|| clones.extend(docs.iter().cloned()));
        drop(clones);
        let serialise_compact_peak = docs
            .iter()
            .map(|d| measure(|| serde_json::to_vec(d).unwrap()).1.peak_mem())
            .max()
            .unwrap_or(0);
        let cbor: Vec<Vec<u8>> = docs.iter().map(encode).collect();
        let cbor_compressed_bytes = docs.iter().map(|d| encode_compressed(d).len()).sum();
        let mut enc = StreamEncoder::new(StreamEncoderConfig {
            table_budget_bytes: usize::MAX,
        });
        let mut stream = Vec::new();
        for d in &docs {
            enc.encode_into(&mut stream, d);
        }
        drop(enc);
        // As above: no other copy alive during the decode.
        drop(docs);
        let mut decoded = Vec::with_capacity(cbor.len());
        let ((), d) = measure(|| decoded.extend(cbor.iter().map(|b| decode(b).unwrap())));
        rows.push(Row {
            fixture,
            json_bytes: lines.iter().map(|l| l.len()).sum(),
            parse_peak: p.peak_mem(),
            parse_retained: p.mem_leaked(),
            parse_allocs: p.num_allocs(),
            ijson_mem_allocated,
            clone_peak: c.peak_mem(),
            clone_retained: c.mem_leaked(),
            serialise_compact_peak,
            cbor_bytes: cbor.iter().map(Vec::len).sum(),
            cbor_compressed_bytes,
            cbor_stream_bytes: stream.len(),
            cbor_decode_retained: d.mem_leaked(),
        });
    }

    let sizes = [
        ("IValue", size_of::<IValue>()),
        ("IString", size_of::<IString>()),
        ("INumber", size_of::<INumber>()),
        ("IArray", size_of::<IArray>()),
        ("IObject", size_of::<IObject>()),
        ("Option<IValue>", size_of::<Option<IValue>>()),
    ];

    if json {
        #[derive(serde::Serialize)]
        struct Out<'a> {
            size_of: Vec<(&'a str, usize)>,
            rows: &'a [Row],
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&Out {
                size_of: sizes.to_vec(),
                rows: &rows
            })
            .unwrap()
        );
        return;
    }
    println!("## size_of (bytes)\n");
    println!("| Type | Bytes |\n|---|---|");
    for (n, s) in sizes {
        println!("| {n} | {s} |");
    }
    println!("\n## Per fixture (bytes; retained = allocated and still live after the operation)\n");
    println!("| Fixture | JSON | parse peak | parse retained | parse allocs | ijson mem_allocated | clone peak | clone retained | serialise peak | CBOR | CBOR+zstd | CBOR stream | CBOR decode retained |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for r in &rows {
        println!(
            "| {} | {} | {} | {} | {} | {:.0} | {} | {} | {} | {} | {} | {} | {} |",
            r.fixture,
            r.json_bytes,
            r.parse_peak,
            r.parse_retained,
            r.parse_allocs,
            r.ijson_mem_allocated,
            r.clone_peak,
            r.clone_retained,
            r.serialise_compact_peak,
            r.cbor_bytes,
            r.cbor_compressed_bytes,
            r.cbor_stream_bytes,
            r.cbor_decode_retained
        );
    }
}
