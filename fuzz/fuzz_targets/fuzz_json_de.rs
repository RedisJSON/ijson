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
    let counted =
        IValueDeserSeed::new(None).deserialize_with_object_hints(&mut deserializer, &json_string);
    match (ordinary, counted) {
        (Ok(ordinary), Ok(counted)) => assert_eq!(ordinary, counted),
        (Err(_), Err(_)) => {}
        _ => panic!("object hints changed whether JSON parsing succeeds"),
    }
});
