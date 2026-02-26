#![no_main]

use ijson::IValue;
use ijson_fuzz::JsonValue;
use libfuzzer_sys::fuzz_target;
use serde::Deserialize;

fuzz_target!(|value: JsonValue| {
    let json_string = serde_json::to_string(&value).unwrap();
    let mut deserializer = serde_json::Deserializer::from_str(&json_string);
    let _ = IValue::deserialize(&mut deserializer);
});
