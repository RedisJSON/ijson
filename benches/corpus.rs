//! Corpus benchmark: RedisJSON's call paths over real JSON files.
//!
//! The files come from `benches/fetch-datasets.sh`. The bench reads them from `IJSON_BENCH_DATASETS`,
//! or from `benches/datasets` when that variable is unset. If neither exists it prints the fetch
//! command and exits 0, so a plain `cargo bench` never fails for want of data.
//!
//! Per file, one RedisJSON key per document: a `.json` file is one document, and each line of an
//! `.ndjson`/`.jsonl` file is one document, as `JSON.SET` would store it. Groups:
//!
//! - `corpus-parse`: `IValueDeserSeed::deserialize_compact_objects` (RedisJSON `JSON.SET`) of every
//!   document into an empty string cache; the documents are dropped outside the timed section.
//! - `corpus-drop`: drop of those documents while they hold the last reference to their strings
//!   (an overwriting `JSON.SET`, `JSON.DEL`, expiry).
//! - `corpus-serialise`: compact `serde_json` output of every document (`JSON.GET`).
//! - `corpus-clone`: `.clone()` of every document (multi-target `JSON.SET`/`MERGE`/`ARRAPPEND`);
//!   the clones are dropped outside the timed section.
//!
//! Before the timed runs it prints a memory table: bytes the allocator holds after parsing each
//! file (requested sizes, counted by the allocator below) and after cloning it, beside
//! `IValue::mem_allocated()`. The string cache is reinitialised before every file, so each file's
//! figures include the cache growth it causes and nothing left over from the previous file.
//!
//! Knobs (environment): `IJSON_BENCH_DATASETS` (directory), `IJSON_BENCH_FILES` (comma list of
//! paths or `*`/`?` patterns relative to the directory, replacing the default list),
//! `IJSON_BENCH_MAX_DOCS` (NDJSON documents per file, default 1000), `IJSON_BENCH_MAX_BYTES` (input
//! bytes per file, default 4 MiB: NDJSON stops before the limit, a larger `.json` file is skipped),
//! `IJSON_BENCH_MEMORY=0` (skip the memory table). Criterion's own options (`--quick`,
//! `--save-baseline`, `--baseline`, a name filter) work as usual.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, SamplingMode, Throughput};
use ijson::{IValue, IValueDeserSeed};

/// Representative files of the `quick` preset (plus NCBI when fetched), relative to the datasets
/// directory. Missing ones are skipped.
const DEFAULT_FILES: &[&str] = &[
    "simdjson-data/jsonexamples/twitter.json",
    "simdjson-data/jsonexamples/citm_catalog.json",
    "simdjson-data/jsonexamples/canada.json",
    "simdjson-data/jsonexamples/gsoc-2018.json",
    "simdjson-data/jsonexamples/apache_builds.json",
    "simdjson-data/jsonexamples/amazon_cellphones.ndjson",
    "floatdata/mobilenetv3_large-768.ndjson",
    "ncbi/refsnp-chrMT.ndjson",
];

// ---------------------------------------------------------------------------------------------
// Allocator: the system allocator, counting requested bytes only while `COUNTING` is set (the
// memory table). During the timed runs the cost is one relaxed load and a predictable branch.
// ---------------------------------------------------------------------------------------------

struct Counting;
static COUNTING: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicI64 = AtomicI64::new(0);
static ALLOCS: AtomicU64 = AtomicU64::new(0);

fn grow(n: i64) {
    let now = LIVE.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if COUNTING.load(Ordering::Relaxed) && !p.is_null() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            grow(l.size() as i64);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if COUNTING.load(Ordering::Relaxed) && !p.is_null() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            grow(l.size() as i64);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            LIVE.fetch_sub(l.size() as i64, Ordering::Relaxed);
        }
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        let q = System.realloc(p, l, new_size);
        if COUNTING.load(Ordering::Relaxed) && !q.is_null() {
            // Counted as an allocation, as `mockalloc` (used by `--bench memory`) records it.
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            grow(new_size as i64 - l.size() as i64);
        }
        q
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

struct Mem {
    retained: i64,
    peak: i64,
    allocs: u64,
}

/// Run `f` with counting on; bytes are relative to the start of the window.
fn measure(f: impl FnOnce()) -> Mem {
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    ALLOCS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::SeqCst);
    f();
    COUNTING.store(false, Ordering::SeqCst);
    Mem {
        retained: LIVE.load(Ordering::Relaxed),
        peak: PEAK.load(Ordering::Relaxed),
        allocs: ALLOCS.load(Ordering::Relaxed),
    }
}

// ---------------------------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------------------------

struct Input {
    label: String,
    docs: Vec<String>,
    bytes: usize,
    file_bytes: u64,
}

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a number, got {v:?}")),
        Err(_) => default,
    }
}

fn datasets_dir() -> Option<PathBuf> {
    let dir = match std::env::var_os("IJSON_BENCH_DATASETS") {
        Some(d) => PathBuf::from(d),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("benches/datasets"),
    };
    dir.is_dir().then_some(dir)
}

/// `*` matches any run of characters (including `/`), `?` exactly one.
fn wildcard(pat: &[u8], s: &[u8]) -> bool {
    let (mut p, mut i, mut star, mut mark) = (0, 0, None, 0);
    while i < s.len() {
        if p < pat.len() && (pat[p] == b'?' || pat[p] == s[i]) {
            p += 1;
            i += 1;
        } else if p < pat.len() && pat[p] == b'*' {
            star = Some(p);
            mark = i;
            p += 1;
        } else if let Some(sp) = star {
            p = sp + 1;
            mark += 1;
            i = mark;
        } else {
            return false;
        }
    }
    pat[p..].iter().all(|&c| c == b'*')
}

fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let path = e.path();
        if e.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        if path.is_dir() {
            walk(&path, root, out);
        } else if let Ok(rel) = path.strip_prefix(root) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

fn select_files(root: &Path) -> Vec<PathBuf> {
    let Ok(list) = std::env::var("IJSON_BENCH_FILES") else {
        let mut out = Vec::new();
        for f in DEFAULT_FILES {
            let p = root.join(f);
            if p.is_file() {
                out.push(p);
            } else {
                println!("corpus: {f} not in the datasets directory, skipped");
            }
        }
        return out;
    };
    let mut all = None;
    let mut out = Vec::new();
    for entry in list.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        if entry.contains(['*', '?']) {
            let all = all.get_or_insert_with(|| {
                let mut v = Vec::new();
                walk(root, root, &mut v);
                v.sort();
                v
            });
            let hits: Vec<_> = all
                .iter()
                .filter(|r| wildcard(entry.as_bytes(), r.as_bytes()))
                .collect();
            if hits.is_empty() {
                println!("corpus: no file matches {entry:?}");
            }
            out.extend(hits.into_iter().map(|r| root.join(r)));
        } else {
            let p = root.join(entry); // an absolute entry replaces `root`
            if p.is_file() {
                out.push(p);
            } else {
                println!("corpus: {entry} not found, skipped");
            }
        }
    }
    out
}

fn load(path: &Path, max_docs: usize, max_bytes: usize) -> Result<Input, String> {
    let file_bytes = fs::metadata(path).map_err(|e| e.to_string())?.len();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let docs = if matches!(ext, "ndjson" | "jsonl") {
        let mut docs = Vec::new();
        let mut bytes = 0;
        let rd = BufReader::new(fs::File::open(path).map_err(|e| e.to_string())?);
        for line in rd.lines() {
            let line = line.map_err(|e| e.to_string())?;
            if line.trim().is_empty() {
                continue;
            }
            if docs.len() >= max_docs || bytes + line.len() > max_bytes {
                break;
            }
            bytes += line.len();
            docs.push(line);
        }
        if docs.is_empty() {
            return Err(format!("no line fits IJSON_BENCH_MAX_BYTES={max_bytes}"));
        }
        docs
    } else {
        if file_bytes > max_bytes as u64 {
            return Err(format!(
                "{file_bytes} bytes > IJSON_BENCH_MAX_BYTES={max_bytes}"
            ));
        }
        vec![fs::read_to_string(path).map_err(|e| e.to_string())?]
    };
    for d in &docs {
        parse(d).map_err(|e| format!("not parsed the way JSON.SET parses: {e}"))?;
    }
    Ok(Input {
        label: path.file_name().unwrap().to_string_lossy().into_owned(),
        bytes: docs.iter().map(String::len).sum(),
        docs,
        file_bytes,
    })
}

// ---------------------------------------------------------------------------------------------
// RedisJSON-shaped operations (as in benches/callpaths.rs)
// ---------------------------------------------------------------------------------------------

/// `IValueManager::from_str` for `Format::JSON`, no FPHA type.
fn parse(s: &str) -> serde_json::Result<IValue> {
    let mut de = serde_json::Deserializer::from_str(s);
    let v = IValueDeserSeed::new(None).deserialize_compact_objects(&mut de)?;
    de.end()?;
    Ok(v)
}

fn parse_all(input: &Input) -> Vec<IValue> {
    let mut out = Vec::with_capacity(input.docs.len());
    for d in &input.docs {
        out.push(parse(d).unwrap());
    }
    out
}

fn fresh_cache() {
    // Every document of the previous file is dropped before this runs.
    ijson::reinit_shared_string_cache();
}

// ---------------------------------------------------------------------------------------------
// Memory table
// ---------------------------------------------------------------------------------------------

fn memory_table(inputs: &[Input]) {
    println!(
        "\n## corpus memory (bytes requested from the allocator; string cache reset per file)\n"
    );
    println!("| File | docs | input bytes | file bytes | parse retained | retained / input | parse peak | parse allocs | ijson mem_allocated | clone retained |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for input in inputs {
        fresh_cache();
        let mut docs = Vec::with_capacity(input.docs.len());
        let p = measure(|| {
            for d in &input.docs {
                docs.push(parse(d).unwrap());
            }
        });
        let mem_allocated: f64 = docs.iter().map(IValue::mem_allocated).sum();
        let mut clones = Vec::with_capacity(docs.len());
        let c = measure(|| clones.extend(docs.iter().cloned()));
        println!(
            "| {} | {} | {} | {} | {} | {:.2} | {} | {} | {:.0} | {} |",
            input.label,
            input.docs.len(),
            input.bytes,
            input.file_bytes,
            p.retained,
            p.retained as f64 / input.bytes as f64,
            p.peak,
            p.allocs,
            mem_allocated,
            c.retained
        );
        drop(clones);
        drop(docs);
    }
    fresh_cache();
    println!();
}

// ---------------------------------------------------------------------------------------------
// Timed runs
// ---------------------------------------------------------------------------------------------

fn bench_file(c: &mut Criterion, input: &Input) {
    fresh_cache();
    let id = || BenchmarkId::from_parameter(&input.label);
    let tp = Throughput::Bytes(input.bytes as u64);

    // Parse and drop first, while no other copy of the file's strings is alive.
    let mut g = c.benchmark_group("corpus-parse");
    g.sampling_mode(SamplingMode::Flat).throughput(tp.clone());
    g.bench_function(id(), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut out = Vec::with_capacity(input.docs.len());
                let t0 = Instant::now();
                for d in &input.docs {
                    out.push(parse(black_box(d)).unwrap());
                }
                total += t0.elapsed();
                drop(out);
            }
            total
        })
    });
    g.finish();

    let mut g = c.benchmark_group("corpus-drop");
    g.sampling_mode(SamplingMode::Flat).throughput(tp.clone());
    g.bench_function(id(), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut docs = parse_all(input);
                let t0 = Instant::now();
                for d in docs.drain(..) {
                    drop(black_box(d));
                }
                total += t0.elapsed();
            }
            total
        })
    });
    g.finish();

    let live = parse_all(input);
    let mut g = c.benchmark_group("corpus-serialise");
    g.sampling_mode(SamplingMode::Flat).throughput(tp.clone());
    g.bench_function(id(), |b| {
        b.iter(|| {
            for d in &live {
                black_box(serde_json::to_vec(d).unwrap());
            }
        })
    });
    g.finish();

    let mut g = c.benchmark_group("corpus-clone");
    g.sampling_mode(SamplingMode::Flat).throughput(tp);
    g.bench_function(id(), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut out = Vec::with_capacity(live.len());
                let t0 = Instant::now();
                out.extend(live.iter().cloned());
                total += t0.elapsed();
                drop(black_box(out));
            }
            total
        })
    });
    g.finish();
    drop(live);
    fresh_cache();
}

fn main() {
    let Some(root) = datasets_dir() else {
        println!(
            "corpus: no datasets; fetch them with `benches/fetch-datasets.sh` (or set IJSON_BENCH_DATASETS), then \
             `cargo bench --no-default-features --bench corpus`"
        );
        return;
    };
    // As RedisJSON's module load: `init_ijson_shared_string_cache(true)` (the locked cache).
    let _ = ijson::init_shared_string_cache(true);

    let max_docs = env_usize("IJSON_BENCH_MAX_DOCS", 1000);
    let max_bytes = env_usize("IJSON_BENCH_MAX_BYTES", 4 << 20);
    let mut inputs: Vec<Input> = Vec::new();
    for path in select_files(&root) {
        match load(&path, max_docs, max_bytes) {
            Ok(mut input) => {
                if inputs.iter().any(|i| i.label == input.label) {
                    let parent = path.parent().and_then(Path::file_name);
                    let parent = parent.map(|p| p.to_string_lossy()).unwrap_or_default();
                    input.label = format!("{parent}-{}", input.label);
                }
                inputs.push(input);
            }
            Err(e) => println!("corpus: {} skipped: {e}", path.display()),
        }
    }
    if inputs.is_empty() {
        println!("corpus: no input files under {}", root.display());
        return;
    }
    println!("corpus: {} file(s) from {}", inputs.len(), root.display());
    for i in &inputs {
        println!(
            "  {:<36} {:>6} doc(s) {:>10} bytes",
            i.label,
            i.docs.len(),
            i.bytes
        );
    }
    if std::env::var("IJSON_BENCH_MEMORY").as_deref() != Ok("0") {
        memory_table(&inputs);
    }

    let mut c = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .configure_from_args();
    for input in &inputs {
        bench_file(&mut c, input);
    }
    c.final_summary();
}
