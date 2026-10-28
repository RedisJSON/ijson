#![no_main]

use arbitrary::Arbitrary;
use ijson::StreamDecoder;
use libfuzzer_sys::fuzz_target;

// One document of the stream, decoded with or without the nesting limit.
#[derive(Arbitrary, Debug)]
struct Document {
    limit_depth: bool,
    bytes: Vec<u8>,
}

// A stream document that defines three strings, so that the input can also
// use the references 0 to 2.
fn decoder_with_definitions() -> StreamDecoder {
    let mut document = vec![0x83];
    for text in ["aaaaaaaa", "bbbbbbbb", "cccccccc"] {
        document.extend_from_slice(&[0xC6, 0x68]);
        document.extend_from_slice(text.as_bytes());
    }
    let mut decoder = StreamDecoder::new();
    decoder
        .decode(&document, true)
        .expect("the definitions are a valid stream document");
    decoder
}

// Reads the documents in order with one decoder, as a loader does: the table
// grows from document to document, and after a document that failed the decoder
// must refuse every other one.
fuzz_target!(|documents: Vec<Document>| {
    let mut decoder = decoder_with_definitions();
    let mut failed = false;
    for document in documents {
        let known = decoder.len();
        let result = decoder.decode(&document.bytes, document.limit_depth);
        assert!(
            !failed || result.is_err(),
            "a decoder that failed decoded another document"
        );
        failed |= result.is_err();
        assert!(decoder.len() >= known, "the table shrank");
        assert_eq!(decoder.len() as u64, decoder.stats().defs);
    }
});
