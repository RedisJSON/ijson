#![no_main]

use arbitrary_json::ArbitraryValue;
use ijson::IValue;
use libfuzzer_sys::fuzz_target;
use serde::Deserialize;

fuzz_target!(|value: ArbitraryValue| {
    let json_string = value.to_string();
    let mut deserializer = serde_json::Deserializer::from_str(&json_string);
    let _ = IValue::deserialize(&mut deserializer);
});
