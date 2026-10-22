//! The size numbers the plan sets for the stream writer, on small corpora.
//!
//! A total counts what a module writes per record before compression: the 1 byte marker
//! every record starts with, the 4 byte base ID of a stream record, and the CBOR payload.
//! Ratios hold at any size while the table budget does not bind, so these are small versions
//! of the corpora a live-server benchmark loads. Documents are shuffled because Redis saves in
//! hash order. Every writer mode also has to decode back to the same documents, record by
//! record, as a loader would.
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
// The earlier dictionary held at most this many strings.
const OLD_ENTRY_CAP: u64 = 65_536;

/// What a writer mode produced for a corpus.
struct Written {
    bytes: u64,
    stream_docs: u64,
    defs: u64,
    refs: u64,
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
    let stats = encoder.stats();
    Written {
        bytes,
        stream_docs,
        defs: stats.defs,
        refs: stats.refs,
    }
}

/// Writes the corpus with the default and a tiny budget. The inline and default totals
/// come back for the plan's ratios; the mode with a tiny budget only has to stay inside the
/// worst case of the design: 1 byte per definition and 4 bytes of base ID per stream record more
/// than the same documents written inline. (The plan's 5 bytes count the marker too.)
fn measure(docs: &[IValue]) -> (u64, Written) {
    let inline = inline_bytes(docs);
    let mut modes = [config(TABLE_BUDGET), config(TINY_BUDGET)]
        .map(|config| {
            let written = write_corpus(docs, config);
            let worst = inline + 4 * written.stream_docs + written.defs;
            assert!(
                written.bytes <= worst,
                "{config:?}: {} bytes, worst case {worst}",
                written.bytes
            );
            written
        })
        .into_iter();
    (inline, modes.next().unwrap())
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

/// More distinct 1 KiB strings than the earlier dictionary held entries, 256 to a document, each
/// document stored 4 times.
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

/// One string per document that nothing else holds.
fn unique_long() -> Vec<IValue> {
    (0..200)
        .map(|doc| {
            object(vec![
                ("doc", (doc as i64).into()),
                ("payload", text("uniq", doc, 16 * 1024).into()),
            ])
        })
        .collect()
}

/// A large string in 2 documents next to 32 smaller ones in 32 documents each: the shape on which
/// a ranking of strings by their saving once inflated the file elevenfold. The earlier
/// evaluation used 32 MiB and 1 MiB strings; these are a thousandth of that.
fn mixed_size() -> Vec<IValue> {
    let payload = |string: IString| object(vec![("payload", string.into())]);
    let large = text("large", 0, 32 * 1024);
    let mut docs: Vec<IValue> = (0..2).map(|_| payload(large.clone())).collect();
    for index in 0..32 {
        let small = text("small", index, 1024);
        docs.extend((0..32).map(|_| payload(small.clone())));
    }
    scatter(&mut docs);
    docs
}

/// Bodies that another owner holds outside the documents and no document repeats: the writer
/// defines each one and never references it. The owners come back so they outlive the writing.
fn ghost() -> (Vec<IValue>, Vec<IString>) {
    let bodies: Vec<IString> = (0..500).map(|i| text("ghost", i, 4096)).collect();
    let docs = bodies
        .iter()
        .enumerate()
        .map(|(doc, body)| {
            object(vec![
                ("doc", (doc as i64).into()),
                ("body", body.clone().into()),
            ])
        })
        .collect();
    (docs, bodies)
}

#[test]
fn cardinality_beyond_the_old_entry_cap_stays_within_0_30_of_inline() {
    let _guard = serial();
    let (inline, written) = measure(&cardinality());
    {
        let mode = "stream";
        assert!(
            written.defs > OLD_ENTRY_CAP,
            "{mode}: {} entries",
            written.defs
        );
        let ratio = written.ratio(inline);
        assert!(ratio <= 0.30, "{mode}: {ratio:.4} of inline");
    }
}

#[test]
fn a_fourfold_pool_stays_within_0_27_of_inline() {
    let _guard = serial();
    let (inline, written) = measure(&pool());
    {
        let mode = "stream";
        let ratio = written.ratio(inline);
        assert!(ratio <= 0.27, "{mode}: {ratio:.4} of inline");
    }
}

#[test]
fn unique_long_strings_cost_exactly_what_inline_costs() {
    let _guard = serial();
    let (inline, written) = measure(&unique_long());
    {
        let mode = "stream";
        assert_eq!(written.bytes, inline, "{mode}");
        assert_eq!(
            (written.stream_docs, written.defs, written.refs),
            (0, 0, 0),
            "{mode}"
        );
    }
}

#[test]
fn mixed_size_reuse_never_exceeds_1_001_of_inline() {
    let _guard = serial();
    let (inline, written) = measure(&mixed_size());
    {
        let mode = "stream";
        let ratio = written.ratio(inline);
        assert!(ratio <= 1.001, "{mode}: {ratio:.4} of inline");
    }
}

#[test]
fn a_defined_string_that_is_never_repeated_costs_the_worst_case_exactly() {
    let _guard = serial();
    let (docs, _outside_owners) = ghost();
    let (inline, written) = measure(&docs);
    {
        let mode = "stream";
        assert_eq!(
            (written.defs, written.refs, written.stream_docs),
            (docs.len() as u64, 0, docs.len() as u64),
            "{mode}"
        );
        assert_eq!(
            written.bytes,
            inline + 4 * written.stream_docs + written.defs,
            "{mode}"
        );
    }
}
