#![no_main]

use ijson::IValue;
use libfuzzer_sys::fuzz_target;
use serde::Deserialize;

fuzz_target!(|data: &str| {
    if data.is_empty() {
        return;
    }
    let mut deserializer = serde_json::Deserializer::from_str(data);
    let _ = IValue::deserialize(&mut deserializer);
});
