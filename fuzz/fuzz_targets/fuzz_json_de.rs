#![no_main]

use arbitrary::Arbitrary;
use ijson::IValue;
use libfuzzer_sys::fuzz_target;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Arbitrary, Debug)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(HashMap<String, JsonValue>),
}

impl JsonValue {
    fn to_json_string(&self) -> String {
        match self {
            JsonValue::Null => "null".to_string(),
            JsonValue::Bool(b) => b.to_string(),
            JsonValue::Number(n) => {
                if n.is_finite() {
                    n.to_string()
                } else {
                    "0".to_string()
                }
            }
            JsonValue::String(s) => format!("\"{}\"", s),
            JsonValue::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_json_string()).collect();
                format!("[{}]", items.join(","))
            }
            JsonValue::Object(obj) => {
                let items: Vec<String> = obj
                    .iter()
                    .map(|(k, v)| {
                        let key = k.clone();
                        format!("\"{}\":{}", key, v.to_json_string())
                    })
                    .collect();
                format!("{{{}}}", items.join(","))
            }
        }
    }
}

fuzz_target!(|value: JsonValue| {
    let json_string = value.to_json_string();
    let mut deserializer = serde_json::Deserializer::from_str(&json_string);
    let _ = IValue::deserialize(&mut deserializer);
});
