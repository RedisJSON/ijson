/// Compares CBOR and zstd-compressed CBOR sizes vs JSON for representative IValue documents.
///
/// Run with:
///   cargo run --example size_profile
use ijson::{cbor, FPHAConfig, FloatType, IValue, IValueDeserSeed};
use serde::de::DeserializeSeed;
use serde::Deserialize;

struct Case {
    name: &'static str,
    value: IValue,
}

fn json_size(v: &IValue) -> usize {
    serde_json::to_string(v).unwrap().len()
}

fn cbor_size(v: &IValue) -> usize {
    cbor::encode(v).len()
}

fn cbor_zstd_size(v: &IValue) -> usize {
    cbor::encode_compressed(v).len()
}

fn from_json(s: &str) -> IValue {
    IValue::deserialize(&mut serde_json::Deserializer::from_str(s)).unwrap()
}

fn from_json_fpha(s: &str, ft: FloatType) -> IValue {
    let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(ft)));
    seed.deserialize(&mut serde_json::Deserializer::from_str(s))
        .unwrap()
}

fn make_cases() -> Vec<Case> {
    // ── typed float arrays ──────────────────────────────────────────────────
    let n = 1000usize;
    let fp32_json = format!(
        "[{}]",
        (0..n)
            .map(|i| format!("{:.6}", i as f32 * 0.001))
            .collect::<Vec<_>>()
            .join(",")
    );
    let fp64_json = format!(
        "[{}]",
        (0..n)
            .map(|i| format!("{:.15}", i as f64 * 0.001))
            .collect::<Vec<_>>()
            .join(",")
    );

    // ── string-heavy object ─────────────────────────────────────────────────
    let string_obj_json = format!(
        "{{{}}}",
        (0..50)
            .map(|i| format!("\"key_{i}\":\"value_{i}_some_longer_string_here\""))
            .collect::<Vec<_>>()
            .join(",")
    );

    // ── heterogeneous object ────────────────────────────────────────────────
    let hetero_json = r#"{
        "name": "Alice",
        "age": 30,
        "scores": [1, 2, 3, null, true, "bonus"],
        "meta": {"active": true, "level": 42}
    }"#;

    // ── nested typed arrays ─────────────────────────────────────────────────
    let small_fp32 = format!(
        "[{}]",
        (0..100)
            .map(|i| format!("{:.4}", i as f32 * 0.1))
            .collect::<Vec<_>>()
            .join(",")
    );
    let nested_fp32_json = format!("{{\"a\":{small_fp32},\"b\":{small_fp32},\"label\":\"test\"}}");

    // ── big mixed JSON: array of 200 records, each with scalars + fp32 embedding ─
    // Simulates a realistic workload: repeated schema, repeated key names,
    // mix of strings / integers / booleans, and a typed float sub-array.
    let big_mixed_json = {
        let records: Vec<String> = (0..200)
            .map(|i| {
                let embedding: String = (0..32)
                    .map(|j| format!("{:.6}", (i as f32 * 0.01 + j as f32 * 0.001).sin()))
                    .collect::<Vec<_>>()
                    .join(",");
                format!(
                    r#"{{"id":{i},"name":"user_{i}","active":{},"score":{:.4},"tags":["alpha","beta","gamma"],"embedding":[{embedding}]}}"#,
                    i % 2 == 0,
                    i as f64 * 1.5,
                )
            })
            .collect();
        format!("[{}]", records.join(","))
    };

    // ── repeated strings: 500 objects sharing the same schema and many identical values ─
    // Targets RED-141886: string-reuse gap between in-memory and RDB representation.
    // Keys ("status", "region", "tier", "owner") and values ("active", "us-east-1",
    // "premium", "team-a") repeat across every record, stressing string deduplication.
    let repeated_strings_json = {
        let statuses = ["active", "inactive", "pending"];
        let regions = ["us-east-1", "eu-west-1", "ap-southeast-1"];
        let tiers = ["free", "standard", "premium"];
        let owners = ["team-a", "team-b", "team-c", "team-d"];
        let records: Vec<String> = (0..500)
            .map(|i| {
                format!(
                    r#"{{"id":{i},"status":"{}","region":"{}","tier":"{}","owner":"{}","count":{}}}"#,
                    statuses[i % statuses.len()],
                    regions[i % regions.len()],
                    tiers[i % tiers.len()],
                    owners[i % owners.len()],
                    i * 10,
                )
            })
            .collect();
        format!("[{}]", records.join(","))
    };

    vec![
        Case {
            name: "FP32 array (1000 elements)",
            value: from_json_fpha(&fp32_json, FloatType::F32),
        },
        Case {
            name: "FP64 array (1000 elements)",
            value: from_json_fpha(&fp64_json, FloatType::F64),
        },
        Case {
            name: "Heterogeneous array (1000 numbers, no hint)",
            value: from_json(&fp32_json),
        },
        Case {
            name: "String-heavy object (50 keys)",
            value: from_json(&string_obj_json),
        },
        Case {
            name: "Mixed object (hetero)",
            value: from_json(hetero_json),
        },
        Case {
            name: "Nested FP32 arrays + string",
            value: from_json_fpha(&nested_fp32_json, FloatType::F32),
        },
        Case {
            name: "Big mixed JSON (200 records, hetero embed)",
            value: from_json(&big_mixed_json),
        },
        Case {
            name: "Repeated strings (500 records, RED-141886)",
            value: from_json(&repeated_strings_json),
        },
    ]
}

fn pct(new: usize, base: usize) -> String {
    let p = (new as f64 - base as f64) / base as f64 * 100.0;
    let sign = if p < 0.0 { "" } else { "+" };
    format!("{sign}{p:.1}%")
}

fn main() {
    let cases = make_cases();

    let name_w = 42usize;
    let col_w = 12usize;

    println!(
        "\n{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {:>col_w$}",
        "Document", "JSON (B)", "cbor (B)", "cbor Δ%", "cbor+z (B)",
    );
    println!(
        "{:<name_w$} {:>col_w$}",
        "", "cbor+z Δ%",
    );
    println!("{}", "-".repeat(name_w + col_w * 5 + 4));

    for c in &cases {
        let j = json_size(&c.value);
        let cb = cbor_size(&c.value);
        let cbz = cbor_zstd_size(&c.value);
        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {:>col_w$}",
            c.name,
            j,
            cb,
            pct(cb, j),
            cbz,
        );
        println!(
            "{:<name_w$} {:>col_w$}",
            "",
            pct(cbz, j),
        );
        println!();
    }

    println!("Δ%: relative to JSON size. Negative = smaller than JSON.");
    println!();
}
