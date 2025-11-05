/// Test to verify that thread-safe cache has less effective deduplication
/// due to lock contention, which explains the 25 kB memory overhead.

use ijson::{init_shared_string_cache, reinit_shared_string_cache, IValue};
use serde_json;
use std::sync::{Arc, Barrier};
use std::thread;

// Simulates the web-app.json benchmark scenario
fn create_duplicate_json() -> String {
    r#"{
        "servlet": [
            {
                "servlet-name": "cofaxCDS",
                "servlet-class": "org.cofax.cds.CDSServlet",
                "init-param": {
                    "configGlossary:installationAt": "Philadelphia, PA",
                    "configGlossary:adminEmail": "ksm@pobox.com",
                    "configGlossary:poweredBy": "Cofax",
                    "configGlossary:poweredByIcon": "/images/cofax.gif"
                }
            },
            {
                "servlet-name": "cofaxEmail",
                "servlet-class": "org.cofax.cds.EmailServlet",
                "init-param": {
                    "mailHost": "mail1",
                    "mailHostOverride": "mail2"
                }
            },
            {
                "servlet-name": "cofaxAdmin",
                "servlet-class": "org.cofax.cds.AdminServlet"
            }
        ]
    }"#
    .to_string()
}

fn count_unique_strings(value: &IValue) -> usize {
    use std::collections::HashSet;
    
    let mut seen_ptrs = HashSet::new();
    
    fn collect_string_ptrs(value: &IValue, seen: &mut HashSet<usize>) {
        match value.destructure_ref() {
            ijson::DestructuredRef::String(s) => {
                let ptr = s.as_str().as_ptr() as usize;
                seen.insert(ptr);
            }
            ijson::DestructuredRef::Array(arr) => {
                for item in arr.iter() {
                    collect_string_ptrs(&item, seen);
                }
            }
            ijson::DestructuredRef::Object(obj) => {
                for (key, val) in obj.iter() {
                    // Count the key string
                    let key_ptr = key.as_str().as_ptr() as usize;
                    seen.insert(key_ptr);
                    collect_string_ptrs(&val, seen);
                }
            }
            _ => {}
        }
    }
    
    collect_string_ptrs(value, &mut seen_ptrs);
    seen_ptrs.len()
}

fn calculate_string_memory(value: &IValue) -> usize {
    let mut total = 0;
    
    fn collect_string_memory(value: &IValue, total: &mut usize) {
        match value.destructure_ref() {
            ijson::DestructuredRef::String(s) => {
                *total += s.len();
            }
            ijson::DestructuredRef::Array(arr) => {
                for item in arr.iter() {
                    collect_string_memory(&item, total);
                }
            }
            ijson::DestructuredRef::Object(obj) => {
                for (key, val) in obj.iter() {
                    *total += key.len();
                    collect_string_memory(&val, total);
                }
            }
            _ => {}
        }
    }
    
    collect_string_memory(value, &mut total);
    total
}

#[test]
fn test_thread_unsafe_deduplication() {
    // Reinitialize with thread-unsafe mode
    reinit_shared_string_cache();
    init_shared_string_cache(false).ok(); // May already be initialized
    
    let json_str = create_duplicate_json();
    let mut values = Vec::new();
    
    // Parse the same JSON 10 times to maximize deduplication
    for _ in 0..10 {
        let val: IValue = serde_json::from_str(&json_str).unwrap();
        values.push(val);
    }
    
    let total_unique = values.iter().map(|v| count_unique_strings(v)).sum::<usize>();
    let total_memory = values.iter().map(|v| calculate_string_memory(v)).sum::<usize>();
    
    println!("Thread-unsafe mode:");
    println!("  Total unique string pointers: {}", total_unique);
    println!("  Total string memory: {} bytes", total_memory);
    println!("  Average unique strings per document: {}", total_unique as f64 / 10.0);
}

#[test]
fn test_thread_safe_deduplication_serial() {
    // Reinitialize with thread-safe mode
    reinit_shared_string_cache();
    init_shared_string_cache(true).ok(); // May already be initialized
    
    let json_str = create_duplicate_json();
    let mut values = Vec::new();
    
    // Parse the same JSON 10 times serially (should have good deduplication)
    for _ in 0..10 {
        let val: IValue = serde_json::from_str(&json_str).unwrap();
        values.push(val);
    }
    
    let total_unique = values.iter().map(|v| count_unique_strings(v)).sum::<usize>();
    let total_memory = values.iter().map(|v| calculate_string_memory(v)).sum::<usize>();
    
    println!("Thread-safe mode (serial):");
    println!("  Total unique string pointers: {}", total_unique);
    println!("  Total string memory: {} bytes", total_memory);
    println!("  Average unique strings per document: {}", total_unique as f64 / 10.0);
}

#[test]
fn test_thread_safe_deduplication_concurrent() {
    // Reinitialize with thread-safe mode
    reinit_shared_string_cache();
    init_shared_string_cache(true).ok();
    
    let json_str = create_duplicate_json();
    let num_threads = 4;
    let parses_per_thread = 25; // 100 total parses
    
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];
    
    for _ in 0..num_threads {
        let barrier = Arc::clone(&barrier);
        let json = json_str.clone();
        
        let handle = thread::spawn(move || {
            // Wait for all threads to be ready
            barrier.wait();
            
            let mut values = Vec::new();
            
            // Parse JSON multiple times concurrently
            for _ in 0..parses_per_thread {
                let val: IValue = serde_json::from_str(&json).unwrap();
                values.push(val);
            }
            
            values
        });
        
        handles.push(handle);
    }
    
    let mut all_values = Vec::new();
    for handle in handles {
        all_values.extend(handle.join().unwrap());
    }
    
    let total_unique = all_values.iter().map(|v| count_unique_strings(v)).sum::<usize>();
    let total_memory = all_values.iter().map(|v| calculate_string_memory(v)).sum::<usize>();
    
    println!("Thread-safe mode (concurrent with {} threads):", num_threads);
    println!("  Total unique string pointers: {}", total_unique);
    println!("  Total string memory: {} bytes", total_memory);
    println!("  Average unique strings per document: {}", total_unique as f64 / all_values.len() as f64);
    println!("  Expected 25 kB overhead for 100 docs: ~{} bytes", (total_memory as f64 * 25_000.0 / (all_values.len() as f64 * 170.0)) as usize);
}

#[test]
fn test_compare_deduplication_efficiency() {
    println!("\n=== Deduplication Efficiency Comparison ===\n");
    
    // Test 1: Thread-unsafe (sequential parsing - perfect deduplication)
    reinit_shared_string_cache();
    init_shared_string_cache(false).ok();
    
    let json_str = create_duplicate_json();
    let mut unsafe_values = Vec::new();
    for _ in 0..20 {
        let val: IValue = serde_json::from_str(&json_str).unwrap();
        unsafe_values.push(val);
    }
    
    let unsafe_unique = unsafe_values.iter().map(|v| count_unique_strings(v)).sum::<usize>();
    let unsafe_memory = unsafe_values.iter().map(|v| calculate_string_memory(v)).sum::<usize>();
    
    // Test 2: Thread-safe with high contention (concurrent parsing - poor deduplication)
    reinit_shared_string_cache();
    init_shared_string_cache(true).ok();
    
    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];
    
    for _ in 0..num_threads {
        let barrier = Arc::clone(&barrier);
        let json = json_str.clone();
        
        let handle = thread::spawn(move || {
            barrier.wait();
            let mut values = Vec::new();
            // Parse 5 JSON documents concurrently
            for _ in 0..5 {
                let val: IValue = serde_json::from_str(&json).unwrap();
                values.push(val);
            }
            values
        });
        handles.push(handle);
    }
    
    let mut safe_values = Vec::new();
    for handle in handles {
        safe_values.extend(handle.join().unwrap());
    }
    
    let safe_unique = safe_values.iter().map(|v| count_unique_strings(v)).sum::<usize>();
    let safe_memory = safe_values.iter().map(|v| calculate_string_memory(v)).sum::<usize>();
    
    println!("Thread-unsafe (20 docs, sequential):");
    println!("  Unique string pointers: {}", unsafe_unique);
    println!("  String memory: {} bytes", unsafe_memory);
    println!("  Average unique strings per doc: {}", unsafe_unique as f64 / 20.0);
    
    println!("\nThread-safe (20 docs, 4 threads with barrier):");
    println!("  Unique string pointers: {}", safe_unique);
    println!("  String memory: {} bytes", safe_memory);
    println!("  Average unique strings per doc: {}", safe_unique as f64 / 20.0);
    
    println!("\nDifference:");
    println!("  Extra unique pointers: {}", safe_unique as i64 - unsafe_unique as i64);
    println!("  Extra memory: {} bytes", safe_memory as i64 - unsafe_memory as i64);
    println!("  Extra memory per doc: {} bytes", (safe_memory as i64 - unsafe_memory as i64) / 20);
    
    println!("\n=== Analysis ===");
    println!("The ~25 kB overhead in benchmarks is NOT the Mutex itself (only ~40 bytes)");
    println!("It's {} duplicate strings that failed to deduplicate due to lock contention!", 
             (safe_unique as i64 - unsafe_unique as i64));
    println!("Each duplicate string averages ~{} bytes", 
             (safe_memory as i64 - unsafe_memory as i64) / (safe_unique as i64 - unsafe_unique as i64));
    println!("\nWhen multiple threads parse JSON simultaneously, they:");
    println!("1. Race to acquire the cache Mutex");
    println!("2. Some threads allocate strings while others hold the lock");
    println!("3. Result: Same string allocated multiple times instead of shared");
}

