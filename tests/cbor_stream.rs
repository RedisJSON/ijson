use ciborium::value::Value;
use ijson::array::ArraySliceRef;
use ijson::{
    decode, decode_with_depth_limit, encode, CborDecodeError, DocUse, IArray, IObject, IString,
    IValue, StreamDecoder, StreamEncoder, StreamEncoderConfig,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::{Mutex, MutexGuard, PoisonError};

// Serializes the tests: they look at owner counts and allocator behavior.
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    // One failing test must not make the others fail on a poisoned lock.
    TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

#[ctor::ctor]
fn init_cache() {
    ijson::init_shared_string_cache(true).unwrap();
}

fn config() -> StreamEncoderConfig {
    StreamEncoderConfig {
        table_budget_bytes: 64 << 20,
    }
}

fn json(text: &str) -> IValue {
    serde_json::from_str(text).unwrap()
}

fn cbor_bytes(value: &Value) -> Vec<u8> {
    let mut bytes = Vec::new();
    ciborium::into_writer(value, &mut bytes).unwrap();
    bytes
}

fn define(text: &str) -> Value {
    Value::Tag(6, Box::new(Value::Text(text.into())))
}

fn reference(id: u64) -> Value {
    Value::Tag(7, Box::new(Value::Integer(id.into())))
}

fn encode_doc(encoder: &mut StreamEncoder, value: &IValue) -> (Vec<u8>, DocUse) {
    let mut out = Vec::new();
    let used = encoder.encode_into(&mut out, value);
    (out, used)
}

/// A document as the framing layer stores it: the id its first definition gets,
/// and the payload.
struct Doc {
    base_id: u32,
    bytes: Vec<u8>,
    used: DocUse,
}

impl Doc {
    fn carries_definitions(&self) -> bool {
        self.used.defs > 0
    }

    fn is_stream_doc(&self) -> bool {
        self.used.defs + self.used.refs > 0
    }
}

fn encode_stream(encoder: &mut StreamEncoder, values: &[IValue]) -> Vec<Doc> {
    values
        .iter()
        .map(|value| {
            let base_id = encoder.next_id();
            let (bytes, used) = encode_doc(encoder, value);
            Doc {
                base_id,
                bytes,
                used,
            }
        })
        .collect()
}

/// What a reader does with one document: stream documents must start where its
/// table ends, other documents are plain CBOR.
fn read_doc(decoder: &mut StreamDecoder, doc: &Doc) -> Result<IValue, String> {
    if doc.is_stream_doc() && doc.base_id as usize != decoder.len() {
        return Err(format!(
            "base id {} but the table holds {}",
            doc.base_id,
            decoder.len()
        ));
    }
    decoder.decode(&doc.bytes, true).map_err(|e| e.to_string())
}

fn read_all(docs: &[&Doc]) -> Result<(Vec<IValue>, StreamDecoder), String> {
    let mut decoder = StreamDecoder::new();
    let values = docs
        .iter()
        .map(|doc| read_doc(&mut decoder, doc))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((values, decoder))
}

#[test]
fn encoded_bytes_match_the_wire_format_built_from_cbor_values() {
    let _guard = serial();
    let field = "a field name over the minimum";
    let value = "a value that is long enough to share";
    let values = [
        json(&format!(
            r#"{{"{field}":"{value}","list":["{value}","{field}"]}}"#
        )),
        json(&format!(r#"{{"{field}":["{value}","{value}"]}}"#)),
        // A field used as a value, a value used as a field, and a document that shares nothing.
        json(&format!(
            r#"{{"plain":"{field}","{value}":null,"{field}":1}}"#
        )),
        json("[1, 2, 3]"),
    ];
    let mut encoder = StreamEncoder::new(config());
    let docs = encode_stream(&mut encoder, &values);

    assert_eq!(
        docs[0].bytes,
        cbor_bytes(&Value::Map(vec![
            (define(field), define(value)),
            (
                Value::Text("list".into()),
                Value::Array(vec![reference(1), reference(0)]),
            ),
        ]))
    );
    assert_eq!(
        docs[1].bytes,
        cbor_bytes(&Value::Map(vec![(
            reference(0),
            Value::Array(vec![reference(1), reference(1)])
        )]))
    );
    assert_eq!(docs[3].bytes, encode(&values[3]));
    let counts: Vec<_> = docs
        .iter()
        .map(|doc| (doc.used.defs, doc.used.refs))
        .collect();
    assert_eq!(counts, [(2, 2), (0, 3), (0, 3), (0, 0)]);
    assert_eq!(
        docs.iter().map(|doc| doc.base_id).collect::<Vec<_>>(),
        [0, 2, 2, 2]
    );
    let stats = encoder.stats();
    assert_eq!((stats.defs, stats.refs), (2, 8));
    assert_eq!(stats.def_bytes, (field.len() + value.len()) as u64);

    let (decoded, decoder) = read_all(&docs.iter().collect::<Vec<_>>()).unwrap();
    assert_eq!(decoded, values);
    assert_eq!(decoder.len(), 2);
    assert_eq!(decoder.stats().refs, stats.refs);
}

#[test]
fn documents_without_shared_strings_are_byte_identical_to_plain_encoding() {
    let _guard = serial();
    let mut typed = IObject::new();
    typed
        .insert("f32", IArray::try_from(vec![1.5f32, 2.5]).unwrap())
        .unwrap();
    typed
        .insert("u8", IArray::try_from(vec![1u8, 2, 3]).unwrap())
        .unwrap();
    let typed: IValue = typed.into();
    let cases = [
        IValue::NULL,
        true.into(),
        (-5i64).into(),
        u64::MAX.into(),
        1.5f64.into(),
        IString::from("").into(),
        IString::from("short").into(),
        json(r#"{"a":[1,2,{"b":"c"}],"d":"only one owner of this long string"}"#),
        typed,
    ];
    let mut encoder = StreamEncoder::new(config());
    for value in cases {
        let (bytes, used) = encode_doc(&mut encoder, &value);
        assert_eq!(bytes, encode(&value), "{value:?}");
        assert_eq!(used, DocUse::default());
        assert_eq!(decode(&bytes).unwrap(), value);
    }
    assert_eq!(encoder.next_id(), 0);
}

#[test]
fn decoding_follows_byte_order_for_the_first_key_and_value() {
    let _guard = serial();
    let (key, value) = ("the first key of the map", "the first value of the map");
    let mut decoder = StreamDecoder::new();
    let first = cbor_bytes(&Value::Map(vec![(define(key), define(value))]));
    assert_eq!(
        decoder.decode(&first, true).unwrap(),
        json(&format!(r#"{{"{key}":"{value}"}}"#))
    );
    let both = cbor_bytes(&Value::Array(vec![
        reference(0),
        reference(1),
        reference(0),
    ]));
    assert_eq!(
        decoder.decode(&both, true).unwrap(),
        json(&format!(r#"["{key}","{value}","{key}"]"#))
    );

    let mut decoder = StreamDecoder::new();
    let array = cbor_bytes(&Value::Array(vec![
        define("first element, id zero"),
        define("second element, id one"),
        reference(0),
        Value::Map(vec![(define("a key defined late"), reference(1))]),
        reference(2),
    ]));
    assert_eq!(
        decoder.decode(&array, true).unwrap(),
        json(
            r#"["first element, id zero","second element, id one","first element, id zero",
                {"a key defined late":"second element, id one"},"a key defined late"]"#
        )
    );
    assert_eq!(decoder.len(), 3);

    // Every definition takes the next id, even when its text repeats.
    let mut decoder = StreamDecoder::new();
    let twice = cbor_bytes(&Value::Array(vec![
        define("the same text twice"),
        define("the same text twice"),
        reference(1),
    ]));
    decoder.decode(&twice, true).unwrap();
    assert_eq!(decoder.len(), 2);
}

#[test]
fn strings_kept_inline_are_never_registered() {
    let _guard = serial();
    let mut encoder = StreamEncoder::new(config());
    for text in ["", "a", "short", "1234567"] {
        let owner = IString::from(text);
        let value: IValue = IArray::try_from(vec![
            IValue::from(owner.clone()),
            IValue::from(owner.clone()),
        ])
        .unwrap()
        .into();
        let (bytes, used) = encode_doc(&mut encoder, &value);
        assert_eq!(bytes, encode(&value), "{text:?}");
        assert_eq!(used, DocUse::default());
    }
    assert_eq!(encoder.next_id(), 0);

    let owner = IString::from("12345678");
    let value: IValue = IArray::try_from(vec![IValue::from(owner.clone()), owner.clone().into()])
        .unwrap()
        .into();
    let (_, used) = encode_doc(&mut encoder, &value);
    assert_eq!(used, DocUse { defs: 1, refs: 1 });
    drop(owner);
}

#[test]
fn poisoned_encoder_writes_plain_documents() {
    let _guard = serial();
    let text = "a string shared by many docs";
    let docs = [
        json(&format!(r#"["{text}"]"#)),
        json(&format!(r#"["{text}","{text}"]"#)),
    ];
    let mut encoder = StreamEncoder::new(config());
    assert!(!encoder.is_poisoned());
    let (_, used) = encode_doc(&mut encoder, &docs[0]);
    assert_eq!(used.defs, 1);

    encoder.poison();
    assert!(encoder.is_poisoned());
    for doc in &docs {
        let (bytes, used) = encode_doc(&mut encoder, doc);
        assert_eq!(bytes, encode(doc));
        assert_eq!(used, DocUse::default());
    }
    assert_eq!(encoder.next_id(), 1);
}

#[test]
fn table_budget_freezes_new_definitions_but_keeps_references() {
    let _guard = serial();
    let strings: Vec<IString> = (0..40)
        .map(|index| IString::from(format!("budget limited string {index:04}")))
        .collect();
    let value = |range: std::ops::Range<usize>| -> IValue {
        IArray::try_from(
            strings[range]
                .iter()
                .cloned()
                .map(IValue::from)
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .into()
    };
    let budget = 700;
    let mut encoder = StreamEncoder::new(StreamEncoderConfig {
        table_budget_bytes: budget,
        ..config()
    });
    let first = value(0..40);
    let (bytes, used) = encode_doc(&mut encoder, &first);
    let stats = encoder.stats();
    assert!(stats.defs > 0 && stats.defs < 40);
    assert_eq!(used.defs, stats.defs);
    let items: Vec<Value> = ciborium::from_reader(bytes.as_slice()).unwrap();
    assert_eq!(
        items.iter().filter(|item| item.is_text()).count(),
        40 - stats.defs as usize
    );
    let mut decoder = StreamDecoder::new();
    assert_eq!(decoder.decode(&bytes, true).unwrap(), first);

    let registered = stats.defs;
    let (bytes, used) = encode_doc(&mut encoder, &first);
    assert_eq!(used.defs, 0);
    assert_eq!(used.refs, registered);
    assert_eq!(decoder.decode(&bytes, true).unwrap(), first);
    assert_eq!(encoder.stats().defs, registered);
    assert!(!encoder.is_poisoned());
}

#[test]
fn a_plain_array_is_not_an_object_key() {
    let _guard = serial();
    let text = || Value::Text("abcdefgh".into());
    let arrays = [
        Value::Array(vec![Value::Integer(6.into()), text()]),
        Value::Array(vec![Value::Integer(7.into()), Value::Integer(0.into())]),
        Value::Array(vec![Value::Integer(6.into())]),
        Value::Array(vec![]),
    ];
    for key in arrays {
        let doc = Value::Map(vec![(key, Value::Integer(1.into()))]);
        assert_eq!(
            StreamDecoder::new().decode(&cbor_bytes(&doc), true),
            Err(CborDecodeError::InvalidValue),
            "{doc:?}"
        );
    }
}

#[test]
fn reference_ids_must_be_integers_of_defined_entries() {
    let _guard = serial();
    let defined = define("the only defined string");
    let payloads = vec![
        Value::Integer(1.into()),
        Value::Integer(u64::MAX.into()),
        Value::Integer((-1).into()),
        Value::Integer((-(u64::MAX as i128) - 1).try_into().unwrap()),
        Value::Tag(2, Box::new(Value::Bytes(vec![1; 9]))),
        Value::Tag(3, Box::new(Value::Bytes(vec![0]))),
        Value::Float(0.0),
        Value::Text("0".into()),
        Value::Bytes(vec![0]),
        Value::Bool(false),
        Value::Null,
        Value::Array(vec![]),
        Value::Map(vec![]),
        reference(0),
    ];
    for payload in payloads {
        let forged = Value::Tag(7, Box::new(payload));
        for doc in [
            Value::Array(vec![defined.clone(), forged.clone()]),
            Value::Map(vec![(defined.clone(), forged.clone())]),
            Value::Array(vec![
                defined.clone(),
                Value::Map(vec![(forged.clone(), Value::Null)]),
            ]),
        ] {
            assert_eq!(
                StreamDecoder::new().decode(&cbor_bytes(&doc), true),
                Err(CborDecodeError::InvalidStringReference),
                "{doc:?}"
            );
        }
    }
    assert_eq!(
        StreamDecoder::new().decode(&cbor_bytes(&reference(0)), true),
        Err(CborDecodeError::InvalidStringReference)
    );
    assert!(StreamDecoder::new()
        .decode(&[0xda, 0x00, 0x00, 0x00, 0x07, 0x00], true)
        .is_err());

    // Any CBOR integer form names an id, here a bignum zero.
    let bignum_zero = Value::Tag(7, Box::new(Value::Tag(2, Box::new(Value::Bytes(vec![0])))));
    let doc = Value::Array(vec![defined.clone(), bignum_zero]);
    assert_eq!(
        StreamDecoder::new()
            .decode(&cbor_bytes(&doc), true)
            .unwrap(),
        json(r#"["the only defined string","the only defined string"]"#)
    );
}

#[test]
fn decoding_rejects_malformed_definitions() {
    let _guard = serial();
    let wrong_payloads = [
        Value::Text("1234567".into()),
        Value::Text(String::new()),
        Value::Integer(8.into()),
        Value::Bytes(b"12345678".to_vec()),
        Value::Array(vec![Value::Text("12345678".into())]),
        Value::Null,
        define("12345678"),
    ];
    for payload in wrong_payloads {
        let forged = Value::Tag(6, Box::new(payload));
        for doc in [
            forged.clone(),
            Value::Map(vec![(forged.clone(), Value::Null)]),
        ] {
            let mut decoder = StreamDecoder::new();
            assert_eq!(
                decoder.decode(&cbor_bytes(&doc), true),
                Err(CborDecodeError::InvalidValue),
                "{doc:?}"
            );
            assert_eq!(decoder.len(), 0);
        }
    }
    // Invalid UTF-8 text never reaches the table either.
    let mut decoder = StreamDecoder::new();
    let invalid = [0xC6, 0x68, 0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8];
    assert!(decoder.decode(&invalid, true).is_err());
    assert_eq!(decoder.len(), 0);

    let mut decoder = StreamDecoder::new();
    assert_eq!(
        decoder
            .decode(&cbor_bytes(&define("12345678")), true)
            .unwrap(),
        IValue::from("12345678")
    );
    assert_eq!(decoder.len(), 1);
}

#[test]
fn decoding_rejects_trailing_bytes() {
    let _guard = serial();
    let mut bytes = cbor_bytes(&define("a definition with trailing bytes"));
    bytes.push(0xF6);
    let mut decoder = StreamDecoder::new();
    assert!(decoder.decode(&bytes, true).is_err());
}

#[test]
fn stream_tags_need_a_stream_decoder() {
    let _guard = serial();
    let define_bytes = cbor_bytes(&define("a defined stream string"));
    assert_eq!(decode(&define_bytes), Err(CborDecodeError::InvalidValue));
    let bytes_payload = cbor_bytes(&Value::Tag(6, Box::new(Value::Bytes(vec![]))));
    assert_eq!(decode(&bytes_payload), Err(CborDecodeError::UnknownTag(6)));
    let reference_bytes = cbor_bytes(&reference(0));
    assert_eq!(decode(&reference_bytes), Err(CborDecodeError::InvalidValue));
}

#[test]
fn typed_arrays_decode_as_before_in_stream_mode() {
    let _guard = serial();
    let mut object = IObject::new();
    object
        .insert(
            "a long enough shared key",
            IArray::try_from(vec![1.5f32, -2.5]).unwrap(),
        )
        .unwrap();
    object
        .insert(
            "a long enough shared key too",
            IArray::try_from(vec![7u16, 8, 9]).unwrap(),
        )
        .unwrap();
    let value: IValue = object.into();
    let again = value.clone();
    let mut encoder = StreamEncoder::new(config());
    let (bytes, used) = encode_doc(&mut encoder, &value);
    assert_eq!(used.defs, 2);
    let decoded = StreamDecoder::new().decode(&bytes, true).unwrap();
    assert_eq!(decoded, value);
    let object = decoded.as_object().unwrap();
    let array = |key: &str| object.get(key).unwrap().as_array().unwrap();
    assert!(matches!(
        array("a long enough shared key").as_slice(),
        ArraySliceRef::F32(_)
    ));
    assert!(matches!(
        array("a long enough shared key too").as_slice(),
        ArraySliceRef::U16(_)
    ));
    drop(again);

    // A tag that is neither a string tag nor a typed array is refused.
    let unknown = cbor_bytes(&Value::Tag(0x10001, Box::new(Value::Integer(0.into()))));
    assert_eq!(
        StreamDecoder::new().decode(&unknown, true),
        Err(CborDecodeError::InvalidValue)
    );
}

/// Wraps `leaf` in `depth` single element arrays.
fn nested_arrays(leaf: IValue, depth: usize) -> IValue {
    (0..depth).fold(leaf, |inner, _| {
        IArray::try_from(vec![inner]).unwrap().into()
    })
}

/// Wraps `leaf` in `depth` single entry objects that all use `key`.
fn nested_objects(key: &IString, leaf: IValue, depth: usize) -> IValue {
    (0..depth).fold(leaf, |inner, _| {
        let mut object = IObject::new();
        object.insert(key.clone(), inner).unwrap();
        object.into()
    })
}

#[test]
fn nesting_limit_treats_definitions_and_references_like_inline_strings() {
    let _guard = serial();
    let leaf = IString::from("a string at the bottom of the document");
    let key = IString::from("a shared key at every level of the document");
    let shallow: IValue =
        IArray::try_from(vec![IValue::from(leaf.clone()), IValue::from(key.clone())])
            .unwrap()
            .into();
    for depth in [126, 127, 128, 300] {
        for limit_depth in [true, false] {
            let cases = [
                ("leaf", nested_arrays(leaf.clone().into(), depth)),
                ("key", nested_objects(&key, IValue::NULL, depth)),
            ];
            for (what, doc) in cases {
                let expected = decode_with_depth_limit(&encode(&doc), limit_depth);
                assert_eq!(
                    expected.is_ok(),
                    !limit_depth || depth < 128,
                    "{what} {depth}"
                );

                // The document holds the first use of its strings.
                let mut encoder = StreamEncoder::new(config());
                let (bytes, used) = encode_doc(&mut encoder, &doc);
                assert!(used.defs > 0, "{what} {depth}");
                let mut decoder = StreamDecoder::new();
                assert_eq!(
                    decoder.decode(&bytes, limit_depth),
                    expected,
                    "define {what} {depth} {limit_depth}"
                );
                if expected.is_ok() {
                    assert_eq!(decoder.len() as u64, used.defs, "{what} {depth}");
                }

                // A shallow document defined them, so this one only references.
                let mut encoder = StreamEncoder::new(config());
                let mut decoder = StreamDecoder::new();
                let (bytes, _) = encode_doc(&mut encoder, &shallow);
                decoder.decode(&bytes, true).unwrap();
                let (bytes, used) = encode_doc(&mut encoder, &doc);
                assert!(used.refs > 0 && used.defs == 0, "{what} {depth}");
                assert_eq!(
                    decoder.decode(&bytes, limit_depth),
                    expected,
                    "reference {what} {depth} {limit_depth}"
                );
            }
        }
    }
}

fn string_pool(rng: &mut StdRng) -> Vec<IString> {
    let mut pool: Vec<IString> = ["", "a", "key", "héllo", "1234567"]
        .into_iter()
        .map(IString::from)
        .collect();
    for index in 0..40 {
        let len = rng.gen_range(8..64);
        pool.push(IString::from(format!("m{index}-{}", "é".repeat(len / 2))));
    }
    for index in 0..8 {
        let len = rng.gen_range(64..2112);
        pool.push(IString::from(format!("l{index}-{}", "x".repeat(len))));
    }
    pool
}

fn random_value(rng: &mut StdRng, pool: &[IString], depth: usize) -> IValue {
    match rng.gen_range(0..if depth >= 4 { 6 } else { 9 }) {
        0 => IValue::NULL,
        1 => rng.gen::<bool>().into(),
        2 => rng.gen_range(-1000..1000i64).into(),
        3 => (rng.gen_range(0..1000) as f64 / 8.0).into(),
        4 => pool[rng.gen_range(0..pool.len())].clone().into(),
        5 => IString::from(format!(
            "unique string {} {}",
            rng.gen::<u64>(),
            rng.gen::<u64>()
        ))
        .into(),
        6 => {
            let items: Vec<IValue> = (0..rng.gen_range(0..6))
                .map(|_| random_value(rng, pool, depth + 1))
                .collect();
            IArray::try_from(items).unwrap().into()
        }
        _ => {
            let mut object = IObject::new();
            for _ in 0..rng.gen_range(0..6) {
                let key = pool[rng.gen_range(0..pool.len())].clone();
                object
                    .insert(key, random_value(rng, pool, depth + 1))
                    .unwrap();
            }
            object.into()
        }
    }
}

#[test]
fn random_documents_round_trip_in_order() {
    let _guard = serial();
    let configs = [
        config(),
        StreamEncoderConfig {
            table_budget_bytes: 600,
            ..config()
        },
        StreamEncoderConfig {
            table_budget_bytes: 600_000,
            ..config()
        },
    ];
    for seed in 0..8 {
        let mut rng = StdRng::seed_from_u64(seed);
        let pool = string_pool(&mut rng);
        let values: Vec<IValue> = (0..60).map(|_| random_value(&mut rng, &pool, 0)).collect();
        for config in configs {
            let mut encoder = StreamEncoder::new(config);
            let docs = encode_stream(&mut encoder, &values);
            assert!(docs.iter().any(Doc::carries_definitions));
            assert!(docs.iter().any(|doc| !doc.is_stream_doc()));

            let (decoded, mut decoder) = read_all(&docs.iter().collect::<Vec<_>>()).unwrap();
            assert_eq!(decoded, values, "seed {seed} {config:?}");
            let stats = encoder.stats();
            assert_eq!(decoder.len() as u64, stats.defs);
            assert_eq!(decoder.stats().defs, stats.defs);
            assert_eq!(decoder.stats().refs, stats.refs);
            assert_eq!(decoder.stats().def_bytes, stats.def_bytes);

            let forged = cbor_bytes(&reference(decoder.len() as u64));
            assert_eq!(
                decoder.decode(&forged, true),
                Err(CborDecodeError::InvalidStringReference)
            );
        }
    }
}

#[test]
fn a_fresh_encoder_and_decoder_make_a_document_stand_alone() {
    let _guard = serial();
    let value = json(
        r#"[{"a repeated object key": 1}, {"a repeated object key": 2}, {"a repeated object key": 3}]"#,
    );
    let mut bytes = Vec::new();
    let used = StreamEncoder::new(config()).encode_into(&mut bytes, &value);
    assert_eq!(used, DocUse { defs: 1, refs: 2 });
    assert!(bytes.len() < encode(&value).len());
    assert_eq!(StreamDecoder::new().decode(&bytes, true).unwrap(), value);
    // A second document written by another fresh encoder restarts its ids at 0.
    let mut other = Vec::new();
    StreamEncoder::new(config()).encode_into(&mut other, &value);
    assert_eq!(other, bytes);
}
