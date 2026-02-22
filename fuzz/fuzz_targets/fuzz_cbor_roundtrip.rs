#![no_main]

use ijson::{cbor, IValue};
use ijson_fuzz::JsonValue;
use libfuzzer_sys::fuzz_target;
use serde::Deserialize;

fuzz_target!(|value: JsonValue| {
    let json_string = value.to_json_string();
    let mut deserializer = serde_json::Deserializer::from_str(&json_string);
    let Ok(original) = IValue::deserialize(&mut deserializer) else {
        return;
    };

    let encoded = cbor::encode(&original);
    let decoded = cbor::decode(&encoded).expect("encode->decode round-trip must not fail");

    assert_eq!(
        original, decoded,
        "round-trip mismatch for input: {json_string}"
    );
});
