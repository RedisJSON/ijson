//! Size and round-trip checks of the stream writer on small corpora.
//!
//! A total counts what a module writes per record before compression: the 1 byte marker
//! every record starts with, the 4 byte base ID of a stream record, and the CBOR payload.
//! Ratios hold at any size while the table budget does not bind, so these are small versions
//! of the corpora a live-server benchmark loads. Documents are shuffled because Redis saves in
//! hash order. Every corpus also has to decode back to the same documents, record by record,
//! as a loader would, with the default budget and with a tiny one.
use ijson::{
    DocUse, IArray, IObject, IString, IValue, StreamDecoder, StreamEncoder, StreamEncoderConfig,
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::sync::{Mutex, MutexGuard, PoisonError};

// Serializes the tests: they look at owner counts and hold the corpora in memory.
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    // One failing test must not make the others fail on a poisoned lock.
    TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

#[ctor::ctor]
fn init_cache() {
    ijson::init_shared_string_cache(true).unwrap();
}

// The marker byte and the base ID that a stream record starts with.
const STREAM_HEADER_LEN: usize = 5;
// What the module writes with: a 256 MiB address table budget in both the saving
// process and its forked children.
const TABLE_BUDGET: usize = 256 << 20;
// Small enough to freeze after a few dozen strings.
const TINY_BUDGET: usize = 2048;
// Ids from this value on need a four byte CBOR integer.
const FOUR_BYTE_IDS: u64 = 65_536;

/// What a writer produced for a corpus.
struct Written {
    bytes: u64,
    stream_docs: u64,
    defs: u64,
}

impl Written {
    fn ratio(&self, inline: u64) -> f64 {
        self.bytes as f64 / inline as f64
    }
}

fn config(table_budget_bytes: usize) -> StreamEncoderConfig {
    StreamEncoderConfig { table_budget_bytes }
}

fn inline_bytes(docs: &[IValue]) -> u64 {
    docs.iter()
        .map(|doc| 1 + ijson::encode(doc).len() as u64)
        .sum()
}

/// Writes the documents in order the way the module frames them, decodes every record the way a
/// loader does and checks that it equals its document.
fn write_corpus(docs: &[IValue], config: StreamEncoderConfig) -> Written {
    let mut encoder = StreamEncoder::new(config);
    let mut decoder = StreamDecoder::new();
    let (mut bytes, mut stream_docs) = (0, 0);
    for (index, doc) in docs.iter().enumerate() {
        let base_id = encoder.next_id();
        let mut record = vec![0; STREAM_HEADER_LEN];
        if encoder.encode_into(&mut record, doc) == DocUse::default() {
            // Self-contained: the marker is the only framing.
            bytes += (record.len() - (STREAM_HEADER_LEN - 1)) as u64;
        } else {
            assert_eq!(base_id as usize, decoder.len(), "document {index}");
            bytes += record.len() as u64;
            stream_docs += 1;
        }
        let decoded = decoder
            .decode(&record[STREAM_HEADER_LEN..], true)
            .unwrap_or_else(|error| panic!("document {index}: {error}"));
        assert!(decoded == *doc, "document {index} did not round trip");
    }
    Written {
        bytes,
        stream_docs,
        defs: encoder.stats().defs,
    }
}

/// Writes the corpus with the default and with a tiny budget. Both stay inside the worst case:
/// 1 byte per definition and 4 bytes of base ID per stream record more than the same documents
/// written inline. The inline and default totals come back for the ratios.
fn measure(docs: &[IValue]) -> (u64, Written) {
    let inline = inline_bytes(docs);
    let default = write_corpus(docs, config(TABLE_BUDGET));
    let tiny = write_corpus(docs, config(TINY_BUDGET));
    for written in [&default, &tiny] {
        let worst = inline + 4 * written.stream_docs + written.defs;
        assert!(
            written.bytes <= worst,
            "{} bytes, worst case {worst}",
            written.bytes
        );
    }
    (inline, default)
}

/// Fisher-Yates, seeded, standing in for the hash order Redis saves in.
fn scatter(docs: &mut [IValue]) {
    docs.shuffle(&mut StdRng::seed_from_u64(217_194));
}

fn text(tag: &str, index: usize, len: usize) -> IString {
    let mut text = format!("{tag}-{index:08}-");
    text.push_str(&"x".repeat(len - text.len()));
    IString::from(text)
}

fn object(entries: Vec<(&str, IValue)>) -> IValue {
    let mut object = IObject::new();
    for (key, value) in entries {
        object.insert(key, value).unwrap();
    }
    object.into()
}

fn array(items: impl Iterator<Item = IString>) -> IValue {
    IArray::try_from(items.map(IValue::from).collect::<Vec<_>>())
        .unwrap()
        .into()
}

/// More distinct 1 KiB strings than a two byte id can name, 256 to a document, each document
/// stored 4 times.
fn cardinality() -> Vec<IValue> {
    let strings: Vec<IString> = (0..258 * 256).map(|i| text("card", i, 1024)).collect();
    let mut docs = Vec::new();
    for (group, items) in strings.chunks(256).enumerate() {
        for copy in 0..4 {
            docs.push(object(vec![
                ("group", (group as i64).into()),
                ("copy", copy.into()),
                ("items", array(items.iter().cloned())),
            ]));
        }
    }
    scatter(&mut docs);
    docs
}

/// 16 large strings, each in all 4 documents.
fn pool() -> Vec<IValue> {
    let strings: Vec<IString> = (0..16).map(|i| text("pool", i, 256 * 1024)).collect();
    let mut docs: Vec<IValue> = (0..4)
        .map(|copy| {
            object(vec![
                ("copy", copy.into()),
                ("pool", array(strings.iter().cloned())),
            ])
        })
        .collect();
    scatter(&mut docs);
    docs
}

#[test]
fn ids_past_65535_round_trip_and_shrink_the_corpus() {
    let _guard = serial();
    let (inline, written) = measure(&cardinality());
    assert!(written.defs > FOUR_BYTE_IDS, "{} entries", written.defs);
    let ratio = written.ratio(inline);
    assert!(ratio <= 0.30, "{ratio:.4} of inline");
}

#[test]
fn strings_above_64_kib_round_trip_and_shrink_a_fourfold_pool() {
    let _guard = serial();
    let (inline, written) = measure(&pool());
    let ratio = written.ratio(inline);
    assert!(ratio <= 0.27, "{ratio:.4} of inline");
}
