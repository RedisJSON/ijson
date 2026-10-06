use ijson::{
    decode, encode, IArray, IString, IValue, StreamDecoder, StreamEncoder, StreamEncoderConfig,
};
use mockalloc::{record_allocs, Mockalloc};
use std::alloc::System;

#[global_allocator]
static ALLOCATOR: Mockalloc<System> = Mockalloc(System);

fn stream_config() -> StreamEncoderConfig {
    StreamEncoderConfig {
        table_budget_bytes: 64 << 20,
        max_def_bytes: 0,
    }
}

/// An array whose elements all share one string.
fn shared_elements(count: usize) -> IValue {
    let shared = IString::from("a string shared by every element of the array");
    IArray::try_from(vec![IValue::from(shared); count])
        .unwrap()
        .into()
}

#[test]
fn encoding_borrows_large_string_content() {
    let value: IValue = IString::from("x".repeat(4 * 1024 * 1024)).into();
    let allocations = record_allocs(|| {
        let bytes = encode(&value);
        assert!(bytes.len() > 4 * 1024 * 1024);
        std::hint::black_box(&bytes);
    });
    assert!(
        allocations.peak_mem() < 6 * 1024 * 1024,
        "encoding retained an extra string copy: {} bytes",
        allocations.peak_mem()
    );
}

#[test]
fn encoding_borrows_array_elements() {
    let value: IValue = IArray::try_from(vec![IValue::NULL; 262_144])
        .unwrap()
        .into();
    let allocations = record_allocs(|| {
        let bytes = encode(&value);
        assert!(bytes.len() > 262_144);
        std::hint::black_box(&bytes);
    });
    assert!(
        allocations.peak_mem() < 1024 * 1024,
        "encoding retained an intermediate array tree: {} bytes",
        allocations.peak_mem()
    );
}

#[test]
fn decoding_builds_array_without_an_intermediate_tree() {
    let value: IValue = IArray::try_from(vec![IValue::NULL; 262_144])
        .unwrap()
        .into();
    let bytes = encode(&value);
    let allocations = record_allocs(|| {
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded, value);
    });
    assert!(
        allocations.peak_mem() < 4 * 1024 * 1024,
        "decoding retained an intermediate tree: {} bytes",
        allocations.peak_mem()
    );
}

#[test]
fn stream_encoding_borrows_large_string_content() {
    let shared = IString::from("x".repeat(4 * 1024 * 1024));
    let value: IValue = shared.clone().into();
    let mut encoder = StreamEncoder::new(stream_config());
    let allocations = record_allocs(|| {
        let mut out = Vec::new();
        assert_eq!(encoder.encode_into(&mut out, &value, true).defs, 1);
        assert!(out.len() > 4 * 1024 * 1024);
        std::hint::black_box(&out);
    });
    assert!(
        allocations.peak_mem() < 6 * 1024 * 1024,
        "defining retained an extra string copy: {} bytes",
        allocations.peak_mem()
    );

    let allocations = record_allocs(|| {
        let mut out = Vec::new();
        assert_eq!(encoder.encode_into(&mut out, &value, true).refs, 1);
        assert_eq!(out.len(), 2);
    });
    assert!(
        allocations.peak_mem() < 64,
        "referencing allocated {} bytes",
        allocations.peak_mem()
    );
    drop(shared);
}

#[test]
fn stream_encoding_borrows_array_elements() {
    let value: IValue = IArray::try_from(vec![IValue::NULL; 262_144])
        .unwrap()
        .into();
    let mut encoder = StreamEncoder::new(stream_config());
    let allocations = record_allocs(|| {
        let mut out = Vec::new();
        assert_eq!(encoder.encode_into(&mut out, &value, true).defs, 0);
        assert!(out.len() > 262_144);
        std::hint::black_box(&out);
    });
    assert!(
        allocations.peak_mem() < 1024 * 1024,
        "encoding retained an intermediate array tree: {} bytes",
        allocations.peak_mem()
    );
}

#[test]
fn stream_encoding_of_shared_strings_needs_less_memory_than_plain_encoding() {
    let value = shared_elements(262_144);
    let plain = record_allocs(|| {
        std::hint::black_box(encode(&value));
    });
    let mut encoder = StreamEncoder::new(stream_config());
    let stream = record_allocs(|| {
        let mut out = Vec::new();
        let used = encoder.encode_into(&mut out, &value, true);
        assert_eq!((used.defs, used.refs), (1, 262_143));
        std::hint::black_box(&out);
    });
    assert!(
        stream.peak_mem() < plain.peak_mem() / 8,
        "stream {} bytes, plain {} bytes",
        stream.peak_mem(),
        plain.peak_mem()
    );
}

#[test]
fn stream_decoding_builds_array_without_an_intermediate_tree() {
    let value = shared_elements(262_144);
    let mut bytes = Vec::new();
    StreamEncoder::new(stream_config()).encode_into(&mut bytes, &value, true);
    let inline = encode(&value);
    let plain = record_allocs(|| {
        assert_eq!(decode(&inline).unwrap(), value);
    });
    let mut decoder = StreamDecoder::new();
    let stream = record_allocs(|| {
        assert_eq!(decoder.decode(&bytes, true).unwrap(), value);
    });
    // The stream pays for one chunk of the ID table, 64Ki handles of 8 bytes.
    let chunk = 512 * 1024;
    assert!(
        stream.peak_mem() <= plain.peak_mem() + chunk + 64 * 1024,
        "stream {} bytes, plain {} bytes",
        stream.peak_mem(),
        plain.peak_mem()
    );
}
