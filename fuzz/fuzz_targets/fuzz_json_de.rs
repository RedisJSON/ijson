#![no_main]

use arbitrary_json::ArbitraryValue;
use ijson::{IValue, IValueDeserSeed};
use libfuzzer_sys::fuzz_target;
use serde::Deserialize;

fuzz_target!(|value: ArbitraryValue| {
    let json_string = value.to_string();
    let mut deserializer = serde_json::Deserializer::from_str(&json_string);
    let ordinary = IValue::deserialize(&mut deserializer);
    let mut deserializer = serde_json::Deserializer::from_str(&json_string);
    let buffered = IValueDeserSeed::new(None).deserialize_compact_objects(&mut deserializer);
    match (ordinary, buffered) {
        (Ok(ordinary), Ok(buffered)) => assert_eq!(ordinary, buffered),
        (Err(_), Err(_)) => {}
        _ => panic!("buffered objects changed whether JSON parsing succeeds"),
    }
});
