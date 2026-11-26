//! Attribute retained memory of a parsed document to its components and compare with
//! `IValue::mem_allocated()` (what RedisJSON reports through JSON.DEBUG MEMORY).
//!
//! `cargo run --release --no-default-features --example mem_attribution -- <file.json>...`
//!
//! Each file is parsed once and dropped before the measured parse, so in the default build the
//! string cache's table already has room for the file's strings and `retained` is the document
//! alone, whatever file came before it. The `thread_safe` build's cache shrinks a shard when it
//! empties, so there the gap to `mem_allocated()` is the table growth the parse causes.

use std::alloc::System;
use std::collections::HashSet;

use ijson::{DestructuredRef, IValue, IValueDeserSeed};
use mockalloc::{record_allocs, Mockalloc};

#[global_allocator]
static ALLOCATOR: Mockalloc<System> = Mockalloc(System);

#[derive(Default, Debug)]
struct Stats {
    objects: usize,
    object_len: usize,
    object_cap: usize,
    arrays: usize,
    array_len: usize,
    array_cap: usize,
    typed_arrays: usize,
    heap_numbers: usize,
    string_refs: usize,
    /// References to strings that own no heap allocation: the empty string, and in the default
    /// build strings of up to 7 bytes, which are stored inline in the handle.
    no_heap_string_refs: usize,
    unique_heap_strings: usize,
    unique_heap_string_bytes: usize,
    uniq: HashSet<usize>,
}

/// Whether a string of `len` bytes is stored without a heap allocation. The `thread_safe` build's
/// `IString` (`string.rs`) has no inline form: every non-empty string is on the heap.
fn no_heap(len: usize) -> bool {
    len == 0 || (!cfg!(feature = "thread_safe") && len <= 7)
}

fn walk(v: &IValue, s: &mut Stats) {
    match v.destructure_ref() {
        DestructuredRef::Null | DestructuredRef::Bool(_) => {}
        DestructuredRef::Number(_) => {
            // INumber stores small ints inline; anything else is a heap box (mem_allocated > 0)
            if v.mem_allocated() > 0.0 {
                s.heap_numbers += 1;
            }
        }
        DestructuredRef::String(st) => {
            s.string_refs += 1;
            if no_heap(st.len()) {
                s.no_heap_string_refs += 1;
            } else {
                let p = st.as_str().as_ptr() as usize;
                if s.uniq.insert(p) {
                    s.unique_heap_strings += 1;
                    s.unique_heap_string_bytes += st.len();
                }
            }
        }
        DestructuredRef::Array(a) => {
            s.arrays += 1;
            s.array_len += a.len() as usize;
            s.array_cap += a.capacity() as usize;
            if a.as_slice().is_typed() {
                s.typed_arrays += 1;
            } else {
                for item in a.iter() {
                    walk(item.as_ref(), s);
                }
            }
        }
        DestructuredRef::Object(o) => {
            s.objects += 1;
            s.object_len += o.len() as usize;
            s.object_cap += o.capacity() as usize;
            for (k, val) in o.iter() {
                s.string_refs += 1;
                if no_heap(k.len()) {
                    s.no_heap_string_refs += 1;
                } else {
                    let p = k.as_str().as_ptr() as usize;
                    if s.uniq.insert(p) {
                        s.unique_heap_strings += 1;
                        s.unique_heap_string_bytes += k.len();
                    }
                }
                walk(val, s);
            }
        }
    }
}

fn parse(text: &str) -> IValue {
    let mut de = serde_json::Deserializer::from_str(text);
    let v = IValueDeserSeed::new(None)
        .deserialize_compact_objects(&mut de)
        .unwrap();
    de.end().unwrap();
    v
}

fn main() {
    // As RedisJSON's module load: `init_ijson_shared_string_cache(true)` (the locked cache).
    let _ = ijson::init_shared_string_cache(true);
    println!("| file | JSON bytes | retained (allocator) | mem_allocated() | gap | objects (len/cap) | arrays (len/cap, typed) | heap numbers | string refs (no heap) | unique heap strings (bytes) |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).unwrap();
        drop(parse(&text));
        let mut slot: Option<IValue> = None;
        let info = record_allocs(|| slot = Some(parse(&text)));
        let doc = slot.unwrap();
        let mut s = Stats::default();
        walk(&doc, &mut s);
        let retained = info.mem_leaked();
        let acc = doc.mem_allocated();
        let name = std::path::Path::new(&path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        println!(
            "| {} | {} | {} | {:.0} | {:.0} | {} ({}/{}) | {} ({}/{}, {} typed) | {} | {} ({}) | {} ({}) |",
            name,
            text.len(),
            retained,
            acc,
            (retained as f64 - acc).round() + 0.0, // + 0.0: an exact match prints as 0, not -0
            s.objects,
            s.object_len,
            s.object_cap,
            s.arrays,
            s.array_len,
            s.array_cap,
            s.typed_arrays,
            s.heap_numbers,
            s.string_refs,
            s.no_heap_string_refs,
            s.unique_heap_strings,
            s.unique_heap_string_bytes
        );
    }
}
