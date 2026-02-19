#![no_main]

use ijson::IValue;
use ijson_fuzz::JsonValue;
use libfuzzer_sys::fuzz_target;
use serde::Deserialize;

fuzz_target!(|value: JsonValue| {
    let json_string = value.to_json_string();
    let mut deserializer = serde_json::Deserializer::from_str(&json_string);
    let _ = IValue::deserialize(&mut deserializer);
});
