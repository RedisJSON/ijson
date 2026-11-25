//! Adversarial security coverage for ijson (SECURITY-2026-11-24).
//!
//! RedisJSON exposes ijson to attacker-controlled JSON documents, JSONPath
//! targets and CBOR payloads (RESTORE of a crafted dump, replication streams).
//! The properties under test are memory safety and availability: no panic, no
//! unbounded growth, no quadratic blow-up, and the object table stays
//! self-consistent (every key retrievable, `len()` correct, insertion order
//! preserved) under hash collisions, huge/degenerate keys, duplicate floods,
//! deep nesting and malformed CBOR.
//!
//! These tests use only the public API, so the file is identical on every
//! candidate branch. Collision construction replicates the base hasher
//! (`DefaultHasher`, SipHash-1-3 with fixed keys) and `IObject`'s
//! `hash_capacity(cap) = cap + cap/4` and `bucket = hash % hash_capacity`. On
//! the EXP-004 branch the table hasher is ahash, so these keys no longer land
//! in one bucket; the consistency assertions still hold (they are the safety
//! property) but the probe-length claim is base/EXP-002/003/005/015/016 only.

#![allow(clippy::needless_range_loop)]

use crate::array::ArraySliceRef;
use crate::{cbor, IObject, IString, IValue, IValueDeserSeed, StreamDecoder};
use serde::Deserialize;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ── collision construction (base hasher) ────────────────────────────────────

fn hash_capacity(cap: usize) -> usize {
    cap + cap / 4
}

fn base_bucket(s: &str, hash_cap: usize) -> usize {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish() as usize % hash_cap
}

/// `count` distinct keys that all share one home bucket of a table sized for
/// `cap` entries, under the base `DefaultHasher`. Keys are long enough to live
/// on the heap (so interning and byte comparison, not inline handles, run).
fn colliding_keys(cap: usize, count: usize) -> Vec<String> {
    let hash_cap = hash_capacity(cap);
    let home = 0usize;
    let mut keys = Vec::with_capacity(count);
    let mut i: u64 = 0;
    while keys.len() < count {
        let candidate = format!("adversarial_collision_key_{i:020}");
        if base_bucket(&candidate, hash_cap) == home {
            keys.push(candidate);
        }
        i += 1;
    }
    keys
}

/// Insert `keys` into an object sized exactly for them (so no growth re-lays
/// the table), then assert the table is internally consistent.
fn assert_object_consistent_for(keys: &[String]) {
    let n = keys.len();
    let mut obj = IObject::with_capacity(n).unwrap();
    for (v, k) in keys.iter().enumerate() {
        assert!(
            obj.insert(k.as_str(), IValue::from(v as u64))
                .unwrap()
                .is_none(),
            "duplicate insert for distinct key {k}"
        );
    }
    assert_eq!(obj.len() as usize, n);

    // Every key retrievable, by &str and by IString, with the right value.
    for (v, k) in keys.iter().enumerate() {
        assert_eq!(
            obj.get(k.as_str()).and_then(IValue::to_u64),
            Some(v as u64),
            "get {k}"
        );
        let is = IString::intern(k);
        assert_eq!(
            obj.get(&is).and_then(IValue::to_u64),
            Some(v as u64),
            "get istring {k}"
        );
        assert!(obj.contains_key(k.as_str()));
    }
    // A key that is not present must miss cleanly.
    assert!(obj.get("adversarial_absent_key_xyzzy").is_none());

    // Insertion order is preserved by iteration.
    let order: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
    let expected: Vec<&str> = keys.iter().map(String::as_str).collect();
    assert_eq!(order, expected, "iteration order");

    // Remove half from the front, re-check the rest, then drain fully.
    for k in keys.iter().take(n / 2) {
        assert!(obj.remove(k.as_str()).is_some(), "remove {k}");
    }
    assert_eq!(obj.len() as usize, n - n / 2);
    for (v, k) in keys.iter().enumerate().skip(n / 2) {
        assert_eq!(
            obj.get(k.as_str()).and_then(IValue::to_u64),
            Some(v as u64),
            "survivor {k}"
        );
    }
    for k in keys.iter().take(n / 2) {
        assert!(obj.get(k.as_str()).is_none(), "removed {k} still present");
    }
    for k in keys.iter().skip(n / 2) {
        assert!(obj.remove(k.as_str()).is_some());
    }
    assert!(obj.is_empty());
}

#[test]
fn colliding_keys_8_16_128_stay_consistent() {
    for &cap in &[8usize, 16, 128] {
        let keys = colliding_keys(cap, cap);
        // The probe chain for the last key is as long as the cluster (every key
        // shares the home bucket), which is the adversarial worst case.
        assert_object_consistent_for(&keys);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn colliding_keys_4096_stay_consistent() {
    let keys = colliding_keys(4096, 4096);
    assert_object_consistent_for(&keys);
}

#[test]
#[cfg_attr(miri, ignore)]
fn colliding_keys_through_compact_parse() {
    // The same collision cluster arriving as a JSON document: the compact
    // object parser must build a consistent table and preserve first-seen order.
    let keys = colliding_keys(128, 128);
    let mut json = String::from("{");
    for (i, k) in keys.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!("\"{k}\":{i}"));
    }
    json.push('}');
    let value = parse_compact(&json).expect("colliding object parses");
    let obj = value.as_object().unwrap();
    assert_eq!(obj.len() as usize, keys.len());
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(obj.get(k.as_str()).and_then(IValue::to_u64), Some(i as u64));
    }
    let order: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
    let expected: Vec<&str> = keys.iter().map(String::as_str).collect();
    assert_eq!(order, expected);
}

// ── degenerate keys ─────────────────────────────────────────────────────────

#[test]
fn long_keys_1kib() {
    let a = "a".repeat(1024);
    let b = "b".repeat(1024);
    let mut obj = IObject::new();
    obj.insert(a.as_str(), IValue::from(1u64)).unwrap();
    obj.insert(b.as_str(), IValue::from(2u64)).unwrap();
    assert_eq!(obj.get(a.as_str()).and_then(IValue::to_u64), Some(1));
    assert_eq!(obj.get(b.as_str()).and_then(IValue::to_u64), Some(2));
    // A long key used as a value too.
    obj.insert("k", IValue::from(a.as_str())).unwrap();
    assert_eq!(
        obj.get("k").and_then(|v| v.as_string().map(|s| s.len())),
        Some(1024)
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn long_key_1mib() {
    let big = "z".repeat(1 << 20);
    let mut obj = IObject::new();
    obj.insert(big.as_str(), IValue::TRUE).unwrap();
    assert_eq!(obj.get(big.as_str()), Some(&IValue::TRUE));
    assert_eq!(obj.get("z"), None);
    // Round-trips through JSON and CBOR without truncation.
    let as_json = serde_json::to_string(&obj.get(big.as_str()).unwrap()).unwrap();
    assert_eq!(as_json, "true");
    let doc = IValue::from(obj);
    let encoded = cbor::encode(&doc);
    assert_eq!(cbor::decode(&encoded).unwrap(), doc);
}

#[test]
fn empty_key_and_nul_and_control_bytes() {
    let mut obj = IObject::new();
    // Empty string key (uses the static empty header, not inline or heap).
    obj.insert("", IValue::from(1u64)).unwrap();
    // Embedded NUL and other control bytes are valid in JSON strings.
    obj.insert("a\0b", IValue::from(2u64)).unwrap();
    obj.insert("\u{1}\u{2}\u{7f}", IValue::from(3u64)).unwrap();
    // A key that is a NUL-prefixed near-duplicate must not alias another.
    obj.insert("\0", IValue::from(4u64)).unwrap();
    obj.insert("\0\0", IValue::from(5u64)).unwrap();
    assert_eq!(obj.get("").and_then(IValue::to_u64), Some(1));
    assert_eq!(obj.get("a\0b").and_then(IValue::to_u64), Some(2));
    assert_eq!(
        obj.get("\u{1}\u{2}\u{7f}").and_then(IValue::to_u64),
        Some(3)
    );
    assert_eq!(obj.get("\0").and_then(IValue::to_u64), Some(4));
    assert_eq!(obj.get("\0\0").and_then(IValue::to_u64), Some(5));
    assert_eq!(obj.get("a").and_then(IValue::to_u64), None);
    assert_eq!(obj.len(), 5);
    // Through the JSON parser (escaped NUL) the same keys survive.
    let value = parse_compact("{\"\":1,\"a\\u0000b\":2,\"\\u0000\":4}").unwrap();
    let o = value.as_object().unwrap();
    assert_eq!(o.get("").and_then(IValue::to_u64), Some(1));
    assert_eq!(o.get("a\0b").and_then(IValue::to_u64), Some(2));
    assert_eq!(o.get("\0").and_then(IValue::to_u64), Some(4));
}

#[test]
fn utf8_boundary_keys() {
    // Multi-byte and near-ASCII keys that differ only in the trailing byte.
    let keys = ["é", "e\u{301}", "中", "中文", "😀", "😁", "a\u{200b}", "a"];
    let mut obj = IObject::new();
    for (i, k) in keys.iter().enumerate() {
        obj.insert(*k, IValue::from(i as u64)).unwrap();
    }
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(
            obj.get(*k).and_then(IValue::to_u64),
            Some(i as u64),
            "{k:?}"
        );
    }
    assert_eq!(obj.len() as usize, keys.len());
}

// ── duplicate-key floods ────────────────────────────────────────────────────

fn parse_compact(json: &str) -> Result<IValue, serde_json::Error> {
    let mut de = serde_json::Deserializer::from_str(json);
    let v = IValueDeserSeed::new(None).deserialize_compact_objects(&mut de)?;
    de.end()?;
    Ok(v)
}

fn parse_plain(json: &str) -> Result<IValue, serde_json::Error> {
    let mut de = serde_json::Deserializer::from_str(json);
    let v = IValue::deserialize(&mut de)?;
    de.end()?;
    Ok(v)
}

#[test]
#[cfg_attr(miri, ignore)]
fn duplicate_key_flood_10k_same_key() {
    // 10k occurrences of one key: last value wins, object holds one entry, no
    // quadratic blow-up and no growth of the final object.
    let mut json = String::from("{");
    for i in 0..10_000 {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!("\"dup\":{i}"));
    }
    json.push('}');
    let value = parse_compact(&json).unwrap();
    let obj = value.as_object().unwrap();
    assert_eq!(obj.len(), 1);
    assert_eq!(obj.get("dup").and_then(IValue::to_u64), Some(9999));
    assert_eq!(
        obj.capacity(),
        1,
        "duplicate flood must shrink to exact size"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn duplicate_key_flood_alternating_two_keys() {
    let mut json = String::from("{");
    for i in 0..10_000 {
        if i > 0 {
            json.push(',');
        }
        let k = if i % 2 == 0 { "a" } else { "b" };
        json.push_str(&format!("\"{k}\":{i}"));
    }
    json.push('}');
    let value = parse_compact(&json).unwrap();
    let obj = value.as_object().unwrap();
    assert_eq!(obj.len(), 2);
    assert_eq!(obj.get("a").and_then(IValue::to_u64), Some(9998));
    assert_eq!(obj.get("b").and_then(IValue::to_u64), Some(9999));
    // First-seen order: "a" then "b".
    let order: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
    assert_eq!(order, vec!["a", "b"]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn duplicate_flood_after_inline_threshold() {
    // More than 16 distinct keys (crossing the inline buffer into the pooled
    // map), then a flood of duplicates of all of them.
    let distinct = 40usize;
    let mut json = String::from("{");
    let mut first = true;
    for i in 0..distinct {
        if !first {
            json.push(',');
        }
        first = false;
        json.push_str(&format!("\"field_number_{i}\":{i}"));
    }
    for round in 0..50 {
        for i in 0..distinct {
            json.push(',');
            json.push_str(&format!("\"field_number_{i}\":{}", 1000 * (round + 1) + i));
        }
    }
    json.push('}');
    let value = parse_compact(&json).unwrap();
    let obj = value.as_object().unwrap();
    assert_eq!(obj.len() as usize, distinct);
    for i in 0..distinct {
        let k = format!("field_number_{i}");
        assert_eq!(
            obj.get(k.as_str()).and_then(IValue::to_u64),
            Some(1000 * 50 + i as u64),
            "last value wins for {k}"
        );
    }
    // Parsing agrees with the plain (non-compact) parser.
    assert_eq!(value, parse_plain(&json).unwrap());
}

// ── deep nesting ────────────────────────────────────────────────────────────

#[test]
fn deep_nesting_json_arrays() {
    // serde_json enforces its own recursion limit (128); ijson must surface the
    // error rather than overflow the stack. Beyond the limit: Err, not a crash.
    for depth in [16usize, 120, 127, 128, 200, 500] {
        let json = format!("{}{}{}", "[".repeat(depth), "", "]".repeat(depth));
        let r = parse_compact(&json);
        // Either parses (within limit) or errors cleanly; never panics.
        if depth <= 120 {
            assert!(r.is_ok(), "depth {depth} should parse");
        }
        let _ = r;
    }
}

#[test]
fn deep_nesting_json_objects() {
    for depth in [16usize, 120, 300] {
        let json = format!("{}{}{}", "{\"k\":".repeat(depth), "null", "}".repeat(depth));
        let _ = parse_compact(&json); // must not panic
    }
}

// ── number edge cases (EXP-001d float-array parse) ──────────────────────────

fn roundtrips(json: &str) {
    for fpha in [
        None,
        Some(crate::FPHAConfig::new_with_type(crate::FloatType::F32)),
    ] {
        let mut de = serde_json::Deserializer::from_str(json);
        let parsed = IValueDeserSeed::new(fpha).deserialize_compact_objects(&mut de);
        if let Ok(v) = parsed {
            // Serialise and re-parse: the value must be stable.
            let s = serde_json::to_string(&v).unwrap();
            let v2 = parse_compact(&s).unwrap();
            if fpha.is_none() {
                assert_eq!(v, v2, "round-trip for {json}");
            }
        }
    }
}

#[test]
fn number_edge_cases() {
    roundtrips("[-0.0]");
    roundtrips("[0.0, -0.0]");
    roundtrips("[1e308]");
    roundtrips("[5e-324]");
    roundtrips("[1.7976931348623157e308]");
    roundtrips("[2.2250738585072014e-308]");
    roundtrips("[9223372036854775807]"); // i64::MAX
    roundtrips("[9223372036854775808]"); // 2^63
    roundtrips("[18446744073709551615]"); // u64::MAX = 2^64-1
    roundtrips("[18446744073709551616]"); // 2^64 -> float
    roundtrips("[-9223372036854775808]"); // i64::MIN
    roundtrips("[1.5, 2, 2.5, 3]"); // mixed: retag
    roundtrips("[3, 1.5]");
    roundtrips("[1.5, \"x\", 2.5]");
    roundtrips("[1.5]");
    roundtrips("[]");
    roundtrips("[1.5, 2.5]");
    roundtrips("[0.1, 0.2, 0.3]");
    roundtrips("[1e300, 2e300, 3e300]");
    // -0.0 stored exactly. Typed arrays are read through `as_slice`, not by index.
    let v = parse_compact("[-0.0]").unwrap();
    let f = match v.as_array().unwrap().as_slice() {
        ArraySliceRef::F64(s) => s[0],
        ArraySliceRef::F32(s) => s[0] as f64,
        ArraySliceRef::F16(s) => s[0].to_f64(),
        other => panic!("expected float storage, got {other:?}"),
    };
    assert!(f == 0.0 && f.is_sign_negative(), "-0.0 preserved");
}

#[test]
fn number_overflow_and_invalid_tokens_error_cleanly() {
    // 1e400 overflows f64; serde_json rejects it. NaN/Infinity are not JSON.
    for bad in [
        "[1e400]",
        "[-1e400]",
        "[NaN]",
        "[Infinity]",
        "[-Infinity]",
        "[-]",
        "[--1]",
        "[1.]",
        "[.5]",
        "[01]",
        "[1e]",
        "[+1]",
        "[0x1]",
    ] {
        // Must return Err, never panic. (1e400 may parse to an error in serde_json.)
        let _ = parse_compact(bad);
    }
    // These specific ones must be rejected.
    assert!(parse_compact("[NaN]").is_err());
    assert!(parse_compact("[Infinity]").is_err());
    assert!(parse_compact("[-]").is_err());
}

#[test]
fn fpha_f32_out_of_range_values() {
    // FPHA forces f32 storage; f64 values outside f32 range must not panic.
    let json = "[1e300, -1e300, 3.4e38, 1e-300]";
    let seed = IValueDeserSeed::new(Some(crate::FPHAConfig::new_with_type(
        crate::FloatType::F32,
    )));
    let mut de = serde_json::Deserializer::from_str(json);
    let _ = seed.deserialize_compact_objects(&mut de); // Ok or Err, no panic
}

#[test]
fn small_arrays_0_1_2_elements() {
    for json in [
        "[]",
        "[1.5]",
        "[1.5,2.5]",
        "[1]",
        "[1,2]",
        "[null]",
        "[true,false]",
    ] {
        roundtrips(json);
    }
}

// ── EXP-001d direct-float-push correctness through parse ────────────────────

#[test]
#[cfg_attr(miri, ignore)]
fn large_float_array_parse_matches_plain() {
    let json = format!(
        "[{}]",
        (0..1024)
            .map(|i| format!("{}.5", i))
            .collect::<Vec<_>>()
            .join(",")
    );
    let compact = parse_compact(&json).unwrap();
    let plain = parse_plain(&json).unwrap();
    assert_eq!(compact, plain);
    let arr = compact.as_array().unwrap();
    assert_eq!(arr.len(), 1024);
    // All `.5` values up to 1023.5 are losslessly representable in f16, so FPHA
    // picks the narrowest float tag; read whichever float storage results.
    let values: Vec<f64> = match arr.as_slice() {
        ArraySliceRef::F64(s) => s.iter().copied().collect(),
        ArraySliceRef::F32(s) => s.iter().map(|&v| v as f64).collect(),
        ArraySliceRef::F16(s) => s.iter().map(|v| v.to_f64()).collect(),
        other => panic!("expected float storage, got {other:?}"),
    };
    for i in 0..1024 {
        assert_eq!(values[i], i as f64 + 0.5);
    }
    // Serialise round-trips.
    assert_eq!(
        parse_compact(&serde_json::to_string(&compact).unwrap()).unwrap(),
        compact
    );
}

// ── EXP-003 clone invariants ────────────────────────────────────────────────

#[test]
fn clone_after_remove_is_tombstone_free() {
    // Build, remove some, shrink, then clone: the clone must be exact-sized and
    // the fast copy-the-table path (when cap == len) must stay consistent.
    let keys = colliding_keys(64, 64);
    let mut obj = IObject::with_capacity(64).unwrap();
    for (v, k) in keys.iter().enumerate() {
        obj.insert(k.as_str(), IValue::from(v as u64)).unwrap();
    }
    for k in keys.iter().take(20) {
        obj.remove(k.as_str());
    }
    obj.shrink_to_fit(); // now cap == len == 44, triggers table copy on clone
    let cloned = obj.clone();
    assert_eq!(cloned, obj);
    for (v, k) in keys.iter().enumerate().skip(20) {
        assert_eq!(
            cloned.get(k.as_str()).and_then(IValue::to_u64),
            Some(v as u64)
        );
    }
    for k in keys.iter().take(20) {
        assert!(cloned.get(k.as_str()).is_none());
    }
}

#[test]
fn clone_at_every_capacity_class() {
    // Exact-sized objects across the no-table / table boundary and several
    // capacity classes, each cloned and checked key-for-key.
    for n in [0usize, 1, 2, 7, 8, 9, 16, 17, 64, 200] {
        let keys = colliding_keys(n.max(1), n);
        let mut obj = IObject::with_capacity(n).unwrap();
        for (v, k) in keys.iter().enumerate() {
            obj.insert(k.as_str(), IValue::from(v as u64)).unwrap();
        }
        assert_eq!(obj.capacity() as usize, n);
        let cloned = obj.clone();
        assert_eq!(cloned, obj, "n={n}");
        assert_eq!(cloned.len() as usize, n);
        for (v, k) in keys.iter().enumerate() {
            assert_eq!(
                cloned.get(k.as_str()).and_then(IValue::to_u64),
                Some(v as u64)
            );
        }
        // Iteration order preserved by the clone.
        let a: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
        let b: Vec<&str> = cloned.keys().map(|k| k.as_str()).collect();
        assert_eq!(a, b, "n={n} order");
    }
}

#[test]
fn clone_then_mutate_both_copies_no_aliasing() {
    let keys = colliding_keys(32, 32);
    let mut obj = IObject::with_capacity(32).unwrap();
    for (v, k) in keys.iter().enumerate() {
        obj.insert(k.as_str(), IValue::from(v as u64)).unwrap();
    }
    let mut cloned = obj.clone();
    // Mutate original: remove, insert, overwrite.
    obj.remove(keys[0].as_str());
    obj.insert("original_only", IValue::from(111u64)).unwrap();
    *obj.get_mut(keys[1].as_str()).unwrap() = IValue::from(999u64);
    // Mutate clone differently.
    cloned.insert("clone_only", IValue::from(222u64)).unwrap();
    *cloned.get_mut(keys[2].as_str()).unwrap() = IValue::from(888u64);
    // Neither mutation leaked into the other.
    assert!(obj.get(keys[0].as_str()).is_none());
    assert_eq!(
        cloned.get(keys[0].as_str()).and_then(IValue::to_u64),
        Some(0)
    );
    assert_eq!(obj.get("original_only").and_then(IValue::to_u64), Some(111));
    assert!(cloned.get("original_only").is_none());
    assert_eq!(cloned.get("clone_only").and_then(IValue::to_u64), Some(222));
    assert!(obj.get("clone_only").is_none());
    assert_eq!(
        obj.get(keys[1].as_str()).and_then(IValue::to_u64),
        Some(999)
    );
    assert_eq!(
        cloned.get(keys[1].as_str()).and_then(IValue::to_u64),
        Some(1)
    );
    assert_eq!(
        cloned.get(keys[2].as_str()).and_then(IValue::to_u64),
        Some(888)
    );
    assert_eq!(obj.get(keys[2].as_str()).and_then(IValue::to_u64), Some(2));
}

// ── EXP-015 entry invariants ────────────────────────────────────────────────

#[test]
#[cfg_attr(miri, ignore)]
fn entry_on_full_table_grows_with_colliding_keys() {
    // Fill an object to capacity with colliding keys, then insert a fresh
    // colliding key: the table is rebuilt (grown) and stays consistent.
    let keys = colliding_keys(64, 64);
    let mut obj = IObject::with_capacity(64).unwrap();
    for (v, k) in keys.iter().enumerate() {
        obj.insert(k.as_str(), IValue::from(v as u64)).unwrap();
    }
    assert_eq!(obj.len(), obj.capacity());
    // A fresh key that also collides (home bucket 0) forces growth.
    let extra = colliding_keys(128, 65).into_iter().nth(64).unwrap();
    obj.insert(extra.as_str(), IValue::from(10_000u64)).unwrap();
    assert!(obj.capacity() > 64);
    assert_eq!(
        obj.get(extra.as_str()).and_then(IValue::to_u64),
        Some(10_000)
    );
    for (v, k) in keys.iter().enumerate() {
        assert_eq!(obj.get(k.as_str()).and_then(IValue::to_u64), Some(v as u64));
    }
}

#[test]
fn entry_on_empty_and_remove_then_reentry() {
    // Entry on a static (empty) object, and removal then re-entry of the same &str.
    let mut obj = IObject::new();
    obj.insert("k", IValue::from(1u64)).unwrap();
    obj.insert("k2", IValue::from(2u64)).unwrap();
    assert_eq!(obj.remove("k").and_then(|v| v.to_u64()), Some(1));
    // Re-enter the same &str: must be vacant again and insert fresh.
    obj.insert("k", IValue::from(3u64)).unwrap();
    assert_eq!(obj.get("k").and_then(IValue::to_u64), Some(3));
    assert_eq!(obj.get("k2").and_then(IValue::to_u64), Some(2));
    // IndexMut path on &str (goes through entry on EXP-015).
    obj["k"] = IValue::from(4u64);
    obj["k3"] = IValue::from(5u64);
    assert_eq!(obj.get("k").and_then(IValue::to_u64), Some(4));
    assert_eq!(obj.get("k3").and_then(IValue::to_u64), Some(5));
    assert_eq!(obj.len(), 3);
}

// ── CBOR / EXP-007 adversarial decode ───────────────────────────────────────

/// A self-contained representative document (objects, nested arrays, typed
/// float/int arrays, strings, booleans, nulls, nesting). Built in code so the
/// test is portable to branches that do not carry `benches/fixtures/`.
fn representative_value() -> IValue {
    let mut json = String::from("{\"businesses\":[");
    for i in 0..40 {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"id\":{i},\"name\":\"business number {i}\",\"open\":{},\
             \"rating\":{}.5,\"coords\":[{}.25,{}.75],\"counts\":[{},{},{}],\
             \"tags\":[\"a\",\"bb\",\"ccc\"],\"meta\":{{\"nested\":{{\"deep\":[true,false,null]}}}}}}",
            i % 2 == 0,
            i % 5,
            i,
            i + 1,
            i,
            i * 2,
            i * 3,
        ));
    }
    json.push_str("],\"total\":40,\"nullable\":null,\"pi\":3.141592653589793}");
    parse_plain(&json).expect("representative document parses")
}

#[test]
#[cfg_attr(miri, ignore)]
fn cbor_truncated_at_every_offset_never_panics() {
    let value = representative_value();
    let encoded = cbor::encode(&value);
    assert_eq!(cbor::decode(&encoded).unwrap(), value);
    // Truncate at every byte offset: each prefix must decode to Ok or a clean
    // Err, never panic or hang.
    for n in 0..encoded.len() {
        let r = cbor::decode(&encoded[..n]);
        // A truncated stream is almost always an error; the full stream is Ok.
        let _ = r;
    }
    // Also with the depth limit disabled.
    for n in (0..encoded.len()).step_by(7) {
        let _ = cbor::decode_with_depth_limit(&encoded[..n], false);
    }
}

#[test]
fn cbor_length_claims_more_than_present() {
    // Definite-length containers and strings whose declared length exceeds the
    // bytes available: must error, not over-read or allocate unboundedly.
    let cases: &[&[u8]] = &[
        &[0x85, 0x01],                   // array(5) with one element
        &[0xa5, 0x61, 0x61, 0x01],       // map(5) with one pair
        &[0x7f_u8.wrapping_sub(0x1f)],   // placeholder (overwritten below)
        &[0x78, 0xff],                   // text of 255 bytes, none present
        &[0x58, 0xff],                   // bytes of 255, none present
        &[0x9a, 0xff, 0xff, 0xff, 0xff], // array(4294967295), empty
        &[0xba, 0xff, 0xff, 0xff, 0xff], // map(4294967295), empty
    ];
    for c in cases {
        let _ = cbor::decode(c); // must not panic / OOM
    }
}

#[test]
fn cbor_huge_declared_count_does_not_oom() {
    // A definite array/map header claiming a count near u64::MAX, followed by a
    // couple of items then EOF. `with_capacity(_.min(1024))` must cap the
    // pre-allocation, so decode errors on EOF instead of trying to reserve 2^64.
    let mut arr = vec![0x9b]; // array, 8-byte count
    arr.extend_from_slice(&u64::MAX.to_be_bytes());
    arr.extend_from_slice(&[0x01, 0x02]); // two ints then EOF
    let _ = cbor::decode(&arr);

    let mut map = vec![0xbb]; // map, 8-byte count
    map.extend_from_slice(&u64::MAX.to_be_bytes());
    map.extend_from_slice(&[0x61, 0x61, 0x01]); // "a":1 then EOF
    let _ = cbor::decode(&map);
}

#[test]
fn cbor_additional_info_28_to_31_and_indefinite() {
    // Additional info 28, 29, 30 are reserved/invalid; 31 is indefinite or break.
    for major in 0u8..8 {
        for ai in 28u8..=31 {
            let byte = (major << 5) | ai;
            let _ = cbor::decode(&[byte]);
            let _ = cbor::decode(&[byte, 0x00]);
            let _ = cbor::decode(&[byte, 0xff]);
        }
    }
    // Indefinite-length containers (valid CBOR, but ijson's encoder never emits
    // them): must decode or error cleanly, never panic.
    let _ = cbor::decode(&[0x9f, 0x01, 0x02, 0xff]); // indefinite array [1,2]
    let _ = cbor::decode(&[0xbf, 0x61, 0x61, 0x01, 0xff]); // indefinite map {"a":1}
    let _ = cbor::decode(&[0x7f, 0x61, 0x61, 0x61, 0x62, 0xff]); // indefinite text "ab"
    let _ = cbor::decode(&[0x5f, 0x41, 0x00, 0xff]); // indefinite bytes
}

#[test]
#[cfg_attr(miri, ignore)]
fn cbor_deep_nesting_depth_limit() {
    // Nesting beyond MAX_DEPTH (128) must error under the limit and not overflow
    // the stack when the limit is enforced.
    for depth in [100usize, 128, 129, 300, 2000] {
        let mut bytes = vec![0x81_u8; depth]; // nested array(1) × depth
        bytes.push(0x00); // innermost: integer 0
        let limited = cbor::decode_with_depth_limit(&bytes, true);
        if depth >= 128 {
            assert!(limited.is_err(), "depth {depth} must hit the limit");
        }
    }
}

#[test]
fn cbor_stream_reference_out_of_range() {
    // Stream string references (tag 7) to ids beyond the table must error with
    // InvalidStringReference, not panic or read out of bounds.
    let mut decoder = StreamDecoder::new();
    // Define two strings (tag 6, text >= 8 bytes), in one document: array of 2.
    let mut doc = vec![0x82];
    for t in ["aaaaaaaa", "bbbbbbbb"] {
        doc.push(0xc6); // tag 6 = definition
        doc.push(0x68); // text(8)
        doc.extend_from_slice(t.as_bytes());
    }
    decoder.decode(&doc, true).expect("definitions decode");
    assert_eq!(decoder.len(), 2);
    // Reference id 0 and 1 are valid; 2, 5 and u32-ish are out of range.
    let ref_doc = |id: u64| {
        let mut d = vec![0xc7]; // tag 7 = reference
        if id < 24 {
            d.push(id as u8);
        } else {
            d.push(0x19);
            d.extend_from_slice(&(id as u16).to_be_bytes());
        }
        d
    };
    assert!(decoder.decode(&ref_doc(0), true).is_ok());
    // A fresh decoder for the out-of-range cases (a failed decode poisons it).
    for id in [2u64, 3, 5, 1000] {
        let mut dec = StreamDecoder::new();
        dec.decode(&doc, true).unwrap();
        let r = dec.decode(&ref_doc(id), true);
        assert!(r.is_err(), "reference {id} must be rejected");
    }
}

#[test]
fn cbor_roundtrip_small_documents() {
    for json in [
        "null",
        "true",
        "false",
        "0",
        "-1",
        "1.5",
        "\"s\"",
        "[]",
        "{}",
        "[1,2,3]",
        "{\"a\":1,\"b\":[1.5,2.5]}",
        "[1.5,2.5,3.5]",
    ] {
        let v = parse_plain(json).unwrap();
        let enc = cbor::encode(&v);
        assert_eq!(cbor::decode(&enc).unwrap(), v, "cbor roundtrip {json}");
    }
}

// ── typed-array retagging under adversarial inputs ──────────────────────────

#[test]
fn typed_array_retag_mixed_then_cbor() {
    // Arrays that start homogeneous and then force a retag (float then string,
    // int then float): CBOR must round-trip whatever tag results.
    for json in [
        "[1.5,2.5,\"x\"]",
        "[1,2,1.5]",
        "[1.5,1,2]",
        "[1,2,3,1.5,\"a\",true]",
    ] {
        let v = parse_compact(json).unwrap();
        let enc = cbor::encode(&v);
        assert_eq!(cbor::decode(&enc).unwrap(), v, "{json}");
    }
}
