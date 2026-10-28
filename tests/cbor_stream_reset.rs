use ijson::{IArray, IString, IValue, StreamDecoder, StreamEncoder, StreamEncoderConfig};

#[ctor::ctor]
fn init_cache() {
    ijson::init_shared_string_cache(true).unwrap();
}

fn config(budget: usize) -> StreamEncoderConfig {
    StreamEncoderConfig {
        table_budget_bytes: budget,
    }
}

fn repeated(text: &IString) -> IValue {
    IArray::try_from(vec![IValue::from(text.clone()), text.clone().into()])
        .unwrap()
        .into()
}

#[test]
fn reset_reuses_ids_and_reader_handles_but_preserves_statistics() {
    let owners: Vec<_> = (0..100)
        .map(|i| IString::from(format!("reset string {i:04}")))
        .collect();
    let mut encoder = StreamEncoder::new(config(84));
    let mut decoder = StreamDecoder::new();
    let mut resets = 0;
    let mut loaded = Vec::new();
    for owner in &owners {
        if encoder.reset_if_needed() {
            decoder.reset();
            assert_eq!(encoder.next_id(), 0);
            assert!(decoder.is_empty());
            resets += 1;
        }
        let value = repeated(owner);
        let mut bytes = Vec::new();
        let used = encoder.encode_into(&mut bytes, &value);
        loaded.push(decoder.decode(&bytes, true).unwrap());
        assert_eq!(loaded.last(), Some(&value));
        assert!(encoder.next_id() <= 3);
        assert!(decoder.len() <= 3);
        assert!(used.defs <= 1);
    }
    assert!(resets > 0);
    assert!(encoder.stats().defs > 3);
    assert_eq!(decoder.stats().defs, encoder.stats().defs);
    assert_eq!(decoder.stats().def_bytes, encoder.stats().def_bytes);
    assert_eq!(loaded, owners.iter().map(repeated).collect::<Vec<_>>());
}

#[test]
fn a_full_table_requests_a_reset_only_after_a_miss_and_a_poisoned_encoder_never_does() {
    let owners: Vec<_> = (0..4)
        .map(|i| IString::from(format!("budget string {i:04}")))
        .collect();
    for poisoned in [false, true] {
        let mut encoder = StreamEncoder::new(config(84));
        for owner in &owners[..3] {
            encoder.encode_into(&mut Vec::new(), &repeated(owner));
        }
        for owner in &owners[..3] {
            assert!(!encoder.reset_if_needed());
            assert_eq!(
                encoder.encode_into(&mut Vec::new(), &repeated(owner)).refs,
                2
            );
        }
        encoder.encode_into(&mut Vec::new(), &repeated(&owners[3]));
        if poisoned {
            encoder.poison();
        }
        assert_eq!(encoder.reset_if_needed(), !poisoned);
    }
}

#[test]
fn unusable_budget_does_not_request_resets() {
    let owners: Vec<_> = (0..4).map(|i| IString::from(format!("{i:016}"))).collect();
    for budget in [0, 1, 83] {
        let mut encoder = StreamEncoder::new(config(budget));
        for owner in &owners {
            encoder.encode_into(&mut Vec::new(), &repeated(owner));
            assert!(!encoder.reset_if_needed());
        }
        assert_eq!(encoder.stats().defs, 0);
    }
}

#[test]
fn a_failed_decoder_refuses_every_later_document_even_after_a_reset() {
    let mut decoder = StreamDecoder::new();
    let define_one = [0xc6, 0x74]
        .into_iter()
        .chain(*b"the first definition")
        .collect::<Vec<u8>>();
    decoder.decode(&define_one, true).unwrap();
    assert!(decoder.decode(&[0xc7, 9], true).is_err());
    // Later documents could resolve ids the failed one was meant to define.
    assert!(decoder.decode(&[0xc7, 0], true).is_err());
    decoder.reset();
    assert!(decoder.decode(&[0xf6], true).is_err());
}
