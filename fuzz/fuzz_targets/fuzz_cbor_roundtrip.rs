#![no_main]

use arbitrary_json::ArbitraryValue;
use ijson::{cbor, IValue};
use libfuzzer_sys::fuzz_target;
use serde::Deserialize;

fuzz_target!(|value: ArbitraryValue| {
    let json_string = value.to_string();
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
