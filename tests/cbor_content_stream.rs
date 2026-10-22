use ijson::{
    ContentDictionary, DocUse, StreamDecoder, StreamEncoder, StreamEncoderConfig, StreamStrings,
    StringDictionary,
};
use mockalloc::{record_allocs, Mockalloc};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Serialize, Serializer};

#[global_allocator]
static ALLOCATOR: Mockalloc<std::alloc::System> = Mockalloc(std::alloc::System);

fn config(budget: usize) -> StreamEncoderConfig {
    StreamEncoderConfig {
        table_budget_bytes: budget,
    }
}

struct Document<'a, 'e> {
    strings: &'a StreamStrings<'e>,
    key: &'a str,
    values: &'a [String],
}

impl Serialize for Document<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        struct Values<'a, 'e>(&'a Document<'a, 'e>);
        impl Serialize for Values<'_, '_> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut seq = serializer.serialize_seq(Some(self.0.values.len()))?;
                for value in self.0.values {
                    seq.serialize_element(&self.0.strings.string(value))?;
                }
                seq.end()
            }
        }
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry(&self.strings.string(self.key), &Values(self))?;
        map.end()
    }
}

#[test]
fn external_strings_are_owned_and_keys_and_values_share_ids() {
    let mut encoder = StreamEncoder::with_dictionary(config(4096), ContentDictionary::default());
    let key = String::from("shared object key");
    let values = vec![
        String::from("shared object key"),
        String::from("another shared value"),
    ];
    let mut first = Vec::new();
    let used = encoder
        .encode_with(&mut first, |strings, out| {
            ciborium::into_writer(
                &Document {
                    strings,
                    key: &key,
                    values: &values,
                },
                out,
            )
        })
        .unwrap();
    assert_eq!(used, DocUse { defs: 2, refs: 1 });
    drop((key, values));

    let mut second = Vec::new();
    let used = encoder
        .encode_with(&mut second, |strings, out| {
            ciborium::into_writer(&strings.string(&String::from("another shared value")), out)
        })
        .unwrap();
    assert_eq!(used, DocUse { defs: 0, refs: 1 });
    assert_eq!(second, [0xc7, 1]);
    let mut decoder = StreamDecoder::new();
    let first = decoder.decode(&first, true).unwrap();
    assert_eq!(
        serde_json::to_value(first).unwrap(),
        serde_json::json!({"shared object key": ["shared object key", "another shared value"]})
    );
    assert_eq!(
        decoder
            .decode(&second, true)
            .unwrap()
            .as_string()
            .unwrap()
            .as_str(),
        "another shared value"
    );
}

#[test]
fn content_budget_counts_owned_bytes_and_resets_release_them() {
    let mut encoder = StreamEncoder::with_dictionary(config(256), ContentDictionary::default());
    let mut decoder = StreamDecoder::new();
    let mut resets = 0;
    for index in 0..20 {
        if encoder.reset_if_needed() {
            decoder.reset();
            resets += 1;
        }
        let text = format!("{index:04}{}", "x".repeat(96));
        let mut bytes = Vec::new();
        let used = encoder
            .encode_with(&mut bytes, |strings, out| {
                ciborium::into_writer(&strings.string(&text), out)
            })
            .unwrap();
        assert!(used.defs <= 1);
        assert!(encoder.next_id() <= 1);
        assert_eq!(
            decoder
                .decode(&bytes, true)
                .unwrap()
                .as_string()
                .unwrap()
                .as_str(),
            text
        );
    }
    assert!(resets > 0);
}

#[test]
fn failed_serialization_discards_bytes_and_poisons_later_references() {
    let mut encoder = StreamEncoder::with_dictionary(config(4096), ContentDictionary::default());
    let mut bytes = vec![42];
    let result = encoder.encode_with(&mut bytes, |strings, out| {
        ciborium::into_writer(&strings.string("a definition in a failed document"), out).unwrap();
        Err::<(), _>("serializer failed")
    });
    assert_eq!(result, Err("serializer failed"));
    assert_eq!(bytes, [42]);
    assert!(encoder.is_poisoned());

    let mut later = Vec::new();
    let used = encoder
        .encode_with(&mut later, |strings, out| {
            ciborium::into_writer(&strings.string("a definition in a failed document"), out)
        })
        .unwrap();
    assert_eq!(used, DocUse::default());
    assert_eq!(
        ijson::decode(&later).unwrap().as_string().unwrap().as_str(),
        "a definition in a failed document"
    );
}

#[test]
fn dictionary_policy_does_not_change_the_wire_format() {
    let json = r#"{"shared object key":["shared string value","shared string value",42]}"#;
    let values: Vec<ijson::IValue> = (0..2)
        .map(|_| serde_json::from_str::<ijson::IValue>(json).unwrap())
        .collect();
    let mut address = StreamEncoder::new(config(4096));
    let mut content = StreamEncoder::with_dictionary(config(4096), ContentDictionary::default());
    let mut decoder = StreamDecoder::new();
    for value in &values {
        let (mut a, mut c) = (Vec::new(), Vec::new());
        assert_eq!(
            address.encode_into(&mut a, value),
            content.encode_into(&mut c, value)
        );
        assert_eq!(a, c);
        assert_eq!(decoder.decode(&c, true).unwrap(), *value);
    }
}

#[test]
fn owned_dictionary_growth_and_repeated_resets_fit_the_allocation_budget() {
    let texts: Vec<_> = (0..2000)
        .map(|i| format!("{i:08}{}", "x".repeat(i % 100)))
        .collect();
    for budget in [512, 4096, 65536] {
        let mut resets = 0;
        let allocations = record_allocs(|| {
            let mut dictionary = ContentDictionary::default();
            for text in &texts {
                if !dictionary.insert(&text.into(), dictionary.len() as u32, budget) {
                    dictionary.clear();
                    resets += 1;
                    assert!(dictionary.insert(&text.into(), 0, budget));
                }
            }
        });
        assert!(resets > 0);
        assert!(
            allocations.peak_mem() <= budget as u64,
            "budget {budget}, peak {}",
            allocations.peak_mem()
        );
    }
}

#[test]
fn a_disabled_content_dictionary_writes_valid_plain_values() {
    for budget in [0, 4096] {
        let mut encoder =
            StreamEncoder::with_dictionary(config(budget), ContentDictionary::default());
        let mut bytes = Vec::new();
        let used = encoder
            .encode_with(&mut bytes, |strings, out| {
                ciborium::into_writer(&strings.string("already defined text"), out)
            })
            .unwrap();
        assert_eq!(used.defs, u64::from(budget != 0));
        bytes.clear();
        let used = encoder
            .encode_with(&mut bytes, |strings, out| {
                ciborium::into_writer(&strings.string("already defined text"), out)
            })
            .unwrap();
        assert_eq!(used.refs, u64::from(budget != 0));
        assert_eq!(used.defs, 0);
    }
}
