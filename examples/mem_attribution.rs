//! Attribute retained memory of a parsed document to its components and compare with
//! `IValue::mem_allocated()` (what RedisJSON reports through JSON.DEBUG MEMORY).
//!
//! `cargo run --release --no-default-features --example mem_attribution -- <file.json>...`

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
    inline_string_refs: usize,
    unique_heap_strings: usize,
    unique_heap_string_bytes: usize,
    uniq: HashSet<usize>,
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
            if st.len() <= 7 {
                s.inline_string_refs += 1;
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
                if k.len() <= 7 {
                    s.inline_string_refs += 1;
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

fn main() {
    let _ = ijson::init_shared_string_cache(false);
    println!("| file | JSON bytes | retained (allocator) | mem_allocated() | gap | objects (len/cap) | arrays (len/cap, typed) | heap numbers | string refs (inline) | unique heap strings (bytes) |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).unwrap();
        let mut slot: Option<IValue> = None;
        let info = record_allocs(|| {
            let mut de = serde_json::Deserializer::from_str(&text);
            let v = IValueDeserSeed::new(None).deserialize_compact_objects(&mut de).unwrap();
            de.end().unwrap();
            slot = Some(v);
        });
        let doc = slot.unwrap();
        let mut s = Stats::default();
        walk(&doc, &mut s);
        let retained = info.mem_leaked();
        let acc = doc.mem_allocated();
        let name = std::path::Path::new(&path).file_name().unwrap().to_string_lossy().into_owned();
        println!(
            "| {} | {} | {} | {:.0} | {:.0} | {} ({}/{}) | {} ({}/{}, {} typed) | {} | {} ({}) | {} ({}) |",
            name,
            text.len(),
            retained,
            acc,
            retained as f64 - acc,
            s.objects,
            s.object_len,
            s.object_cap,
            s.arrays,
            s.array_len,
            s.array_cap,
            s.typed_arrays,
            s.heap_numbers,
            s.string_refs,
            s.inline_string_refs,
            s.unique_heap_strings,
            s.unique_heap_string_bytes
        );
    }
}
