use ijson::{array::ArraySliceRef, FPHAConfig, FloatType, IValue, IValueDeserSeed};
use serde::de::DeserializeSeed;

fn parse(s: &str, counted: bool) -> Result<IValue, serde_json::Error> {
    let mut de = serde_json::Deserializer::from_str(s);
    let value = if counted {
        IValueDeserSeed::new(None).deserialize_compact_objects(&mut de)?
    } else {
        IValueDeserSeed::new(None).deserialize(&mut de)?
    };
    de.end()?;
    Ok(value)
}

#[test]
fn buffered_objects_preserve_values_errors_and_capacity() {
    for input in [
        r#"{"a":1,"b":2,"c":3,"d":4,"e":5}"#,
        r#"{"x":[{}, {"nested":{"a":null,"b":true}}],"str":"{[colon:]}"}"#,
        r#"{"escaped\"key":"slash\\\" colon: } {","unicode":"שלום","n":-1.25e4}"#,
        r#"[{"a":1,"a":{"b":2}},{"z":[]},[1,2,3]]"#,
        "{}",
        "[]",
        "null",
        "123",
        "\"hello\"",
    ] {
        let counted = parse(input, true).unwrap();
        let ordinary = parse(input, false).unwrap();
        assert_eq!(
            serde_json::to_value(&counted).unwrap(),
            serde_json::to_value(&ordinary).unwrap()
        );
    }
    let input = r#"{"a":1,"b":2,"c":3,"d":4,"e":5}"#;
    let mut value = parse(input, true).unwrap();
    assert_eq!(value.as_object().unwrap().capacity(), 5);
    assert_eq!(
        parse(input, false).unwrap().as_object().unwrap().capacity(),
        8
    );
    value
        .as_object_mut()
        .unwrap()
        .insert("f", IValue::NULL)
        .unwrap();
    assert_eq!(value.as_object().unwrap().len(), 6);
    let nested = parse(r#"{"x":[{"a":1,"b":2,"c":3}] }"#, true).unwrap();
    assert_eq!(nested.as_object().unwrap().capacity(), 1);
    assert_eq!(nested["x"][0].as_object().unwrap().capacity(), 3);
    for input in [
        "",
        "{",
        "[",
        "{\"a\":}",
        "{\"a\":1,}",
        "[1,]",
        "{} {}",
        "{\"a\":\"\\q\"}",
        "{\"a\":1]",
        "\"unterminated",
    ] {
        assert!(parse(input, true).is_err(), "{input}");
        assert!(parse(input, false).is_err(), "{input}");
    }
    let deep = format!("{}0{}", "[".repeat(150), "]".repeat(150));
    assert!(parse(&deep, true).is_err());
}

#[test]
fn repeated_keys_use_exact_storage_and_preserve_last_value() {
    // Escaped names compare equal after decoding. Replaced values may themselves
    // contain objects, whose buffers must not interfere with the next field.
    let input = r#"{"a":{"x":1},"\u0061":{"y":2,"z":3},"next":{"p":4}}"#;
    let value = parse(input, true).unwrap();
    assert_eq!(value, parse(input, false).unwrap());
    assert_eq!(
        serde_json::to_string(&value).unwrap(),
        serde_json::to_string(&parse(input, false).unwrap()).unwrap()
    );
    assert_eq!(value.as_object().unwrap().capacity(), 2);
    assert_eq!(value["a"].as_object().unwrap().capacity(), 2);
    assert_eq!(value["next"].as_object().unwrap().capacity(), 1);

    let input = format!("{{{}\"same\":2}}", "\"same\":1,".repeat(100_000));
    let value = parse(&input, true).unwrap();
    assert_eq!(value.as_object().unwrap().capacity(), 1);
    assert_eq!(value, parse(r#"{"same":2}"#, false).unwrap());
}

#[test]
fn buffered_objects_preserve_typed_arrays() {
    let input = r#"{"values":[0.5,1.0,1.5],"nested":[{"x":1,"y":2}]}"#;
    for fp_type in [
        FloatType::F16,
        FloatType::BF16,
        FloatType::F32,
        FloatType::F64,
    ] {
        let seed = || IValueDeserSeed::new(Some(FPHAConfig::new_with_type(fp_type)));
        let mut de = serde_json::Deserializer::from_str(input);
        let value = seed().deserialize_compact_objects(&mut de).unwrap();
        de.end().unwrap();
        let ordinary = seed()
            .deserialize(&mut serde_json::Deserializer::from_str(input))
            .unwrap();
        assert_eq!(value, ordinary);
        assert_eq!(value["nested"][0].as_object().unwrap().capacity(), 2);
        let array = value["values"].as_array().unwrap().as_slice();
        assert!(matches!(
            (fp_type, array),
            (FloatType::F16, ArraySliceRef::F16(_))
                | (FloatType::BF16, ArraySliceRef::BF16(_))
                | (FloatType::F32, ArraySliceRef::F32(_))
                | (FloatType::F64, ArraySliceRef::F64(_))
        ));
    }
}

#[test]
fn buffered_objects_preserve_escaped_strings() {
    // Exercise escaped keys and values at varied lengths.
    for padding in [0, 1, 15, 16, 31, 32, 63, 64, 65, 256, 4096] {
        for slashes in 0..8 {
            for suffix in ["", "\"", "\" :{}[] שלום"] {
                let text = format!("{}{}{}", "x".repeat(padding), "\\".repeat(slashes), suffix);
                let text = serde_json::to_string(&text).unwrap();
                let input = format!("{{{text}:{text},\"next\":{{\"a\":1,\"b\":2}}}}");
                let value = parse(&input, true).unwrap();
                assert_eq!(value, parse(&input, false).unwrap());
                assert_eq!(value.as_object().unwrap().capacity(), 2);
                assert_eq!(value["next"].as_object().unwrap().capacity(), 2);
            }
        }
    }
    for input in ["\"", "\"abc\\", "{\"key\":\"abc\\", "\"\\q\"", "\"\\u12\""] {
        assert!(parse(input, true).is_err(), "{input}");
        assert!(parse(input, false).is_err(), "{input}");
    }
}
