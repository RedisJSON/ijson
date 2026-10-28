//! First-use string streams: strings shared between the documents of one ordered
//! stream.
//!
//! [`StreamEncoder`] writes a string that several values own inline once, wrapped
//! in tag 6 (a definition), and every later occurrence as tag 7 followed by the ID
//! the definition received (a reference). IDs are implicit: the n-th definition of
//! a stream has ID n, so [`StreamDecoder`] rebuilds the same table by reading the
//! documents in the order they were written. Documents with neither tag are plain
//! CBOR.
//!
//! The encoder recognizes a string by the address of its shared allocation, so it
//! never hashes or copies string bytes to find repeats. An address identifies a
//! string only while that string is alive. Callers must keep every defined
//! allocation alive and at the same address until the stream ends.
//!
//! Callers may explicitly reset both dictionaries between documents. IDs then restart,
//! while statistics and the definition-byte allowance remain cumulative.
//!
//! The address-based behavior above is the default. [`ContentDictionary`] accepts
//! ordinary string slices and owns their contents for serializers outside ijson.

use std::cell::RefCell;
use std::fmt;

use ciborium::tag::RequireAny;
use hashbrown::HashMap;
use serde::Serialize;

use super::{EncodedValue, StringRefs, TAG_DEF, TAG_REF};
use crate::{CborDecodeError, IString, IValue};

mod dictionary;
pub use dictionary::{AddressDictionary, ContentDictionary, StreamString, StringDictionary};

/// The encoder as the serializers see it, borrowed for one string at a time.
pub(super) type EncoderCell<'a, D = AddressDictionary> = RefCell<&'a mut StreamEncoder<D>>;

/// The decoder as the deserializers see it, borrowed for one string at a time.
pub(super) type DecoderCell<'a> = RefCell<&'a mut StreamDecoder>;

/// Shortest string, in UTF-8 bytes, that is defined and referenced. Strings of up to 7 bytes are
/// stored inside their handles and have no address to recognize them by, so every string that
/// has an address qualifies.
const MIN_DEFINED_LEN: usize = 8;

/// Settings for a [`StreamEncoder`].
#[derive(Debug, Clone, Copy)]
pub struct StreamEncoderConfig {
    /// Upper bound on the heap the encoder uses, in bytes: its ID table and the
    /// old and new allocations while it grows. When one more definition would
    /// exceed it no more strings are defined: strings already defined keep being
    /// referenced and everything else is written inline. Zero defines nothing.
    /// An explicit reset allows new definitions to reuse the table allocation.
    /// With [`ContentDictionary`], this also includes owned string bytes.
    /// Fixed dictionary-object storage and the output buffer are outside this budget.
    pub table_budget_bytes: usize,
}

/// The definitions and references in one encoded document. A document with
/// neither is plain CBOR.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DocUse {
    /// Strings the document defined. Their IDs follow the encoder's
    /// [`StreamEncoder::next_id`] from before the document.
    pub defs: u64,
    /// References to strings defined earlier, in this document or before it.
    pub refs: u64,
}

/// Counters of a [`StreamEncoder`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamEncoderStats {
    /// Strings defined.
    pub defs: u64,
    /// UTF-8 bytes of the strings defined.
    pub def_bytes: u64,
    /// References written.
    pub refs: u64,
    /// Bytes the references saved compared with writing their text. Each definition adds one
    /// byte, its tag, so the stream is `saved_bytes - defs` bytes smaller than plain CBOR.
    pub saved_bytes: u64,
}

/// Bytes of a CBOR head that carries `value`.
fn cbor_head_len(value: u64) -> u64 {
    match value {
        0..=23 => 1,
        24..=0xff => 2,
        0x100..=0xffff => 3,
        0x1_0000..=0xffff_ffff => 5,
        _ => 9,
    }
}

/// Counters of a [`StreamDecoder`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamDecoderStats {
    /// Definitions read, equal to the length of the ID table.
    /// After resets this remains cumulative and can exceed the current table length.
    pub defs: u64,
    /// UTF-8 bytes of the strings defined.
    pub def_bytes: u64,
    /// References resolved.
    pub refs: u64,
}

/// What the table remembers of a registered string.
#[derive(Debug, Clone, Copy)]
struct Slot {
    id: u32,
    /// Length of the string when it was registered, to notice a recycled address.
    len: u32,
}

type AddressTable = HashMap<usize, Slot>;

fn buckets_for_capacity(capacity: usize) -> usize {
    match capacity {
        0 => 0,
        1..=7 => capacity + 1,
        _ => capacity / 7 * 8,
    }
}

/// Buckets of the table that replaces one of `capacity` entries when it is full.
fn next_buckets(capacity: usize) -> usize {
    match capacity {
        0 => 4,
        _ => 2 * buckets_for_capacity(capacity),
    }
}

/// Bytes allocated for `buckets` buckets: 16 for the entry and 1 control byte
/// each, and a trailing group of control bytes of at most 16.
fn table_alloc_bytes(buckets: usize) -> usize {
    buckets * 17 + 16
}

fn table_bytes(capacity: usize) -> usize {
    match buckets_for_capacity(capacity) {
        0 => 0,
        buckets => table_alloc_bytes(buckets),
    }
}

/// How one string occurrence is written.
enum Choice {
    Text,
    Define,
    Reference(u32),
}

/// Encodes documents into a stream in which shared strings are defined once.
///
/// A string is defined at its first use when at least two values own it and it
/// has at least 8 bytes, and is referenced at every use after that. Write the
/// documents of a stream in the order they are encoded, and have the reader
/// decode them in that order: a reference is only valid once the document with
/// the definition has been read.
///
/// A document that was encoded but not written must be followed by
/// [`poison`](Self::poison), because its definitions would shift the IDs of every
/// later document.
///
/// The default [`AddressDictionary`] requires stable allocations; a
/// [`ContentDictionary`] owns copies and admits strings without owner counts.
pub struct StreamEncoder<D: StringDictionary = AddressDictionary> {
    /// Heap budget for the address table, including growth allocations.
    budget: usize,
    /// Maps string allocation addresses to IDs and lengths; does not own the strings.
    // The default dictionary has the address-only behavior described above.
    table: D,
    reset_pending: bool,
    /// Sticky failure state: encode later strings inline, without DEF or REF.
    poisoned: bool,
    /// Stream counters for reporting.
    /// Definition counts provide cumulative stream positions across resets.
    stats: StreamEncoderStats,
}

impl StreamEncoder {
    /// Creates an encoder that has defined nothing yet.
    pub fn new(config: StreamEncoderConfig) -> Self {
        Self::with_dictionary(config, AddressDictionary::default())
    }

    /// Creates an encoder with the supplied dictionary, clearing any old entries.
    pub fn with_dictionary<D: StringDictionary>(
        config: StreamEncoderConfig,
        mut dictionary: D,
    ) -> StreamEncoder<D> {
        dictionary.clear();
        StreamEncoder {
            budget: config.table_budget_bytes,
            table: dictionary,
            reset_pending: false,
            poisoned: false,
            stats: StreamEncoderStats::default(),
        }
    }
}

impl<D: StringDictionary> StreamEncoder<D> {
    /// Appends the CBOR encoding of `value` to `out`, after whatever `out`
    /// already holds, and reports what the document used.
    ///
    /// When the returned [`DocUse`] is zero the bytes are exactly those of
    /// [`encode`](crate::encode). If an address turns out to be reused by a string of
    /// another length, the encoder is poisoned and that string and the rest of the
    /// document are written inline.
    pub fn encode_into(&mut self, out: &mut Vec<u8>, value: &IValue) -> DocUse {
        let (defs, refs) = (self.stats.defs, self.stats.refs);
        let encoder = RefCell::new(&mut *self);
        let cbor = EncodedValue {
            value,
            strings: &encoder,
        };
        ciborium::into_writer(&cbor, out).expect("write to Vec never fails");
        DocUse {
            defs: self.stats.defs - defs,
            refs: self.stats.refs - refs,
        }
    }

    /// The ID the next definition receives, which is the number of strings
    /// defined so far. Read it before encoding a document to learn the ID of its
    /// first definition. A poisoned encoder defines nothing more, so what it
    /// reports no longer matters.
    /// Counts the current dictionary; `stats().defs` counts the entire stream.
    pub fn next_id(&self) -> u32 {
        // IDs stop below `u32::MAX`, see `has_room`.
        // Strings in the ID table.
        self.table.len() as u32
    }

    /// Reuses the table after a budget miss. Call only between definition-capable documents,
    /// and signal the reader before the next document that introduces definitions.
    pub fn reset_if_needed(&mut self) -> bool {
        if !self.reset_pending || self.poisoned {
            return false;
        }
        self.table.clear();
        self.reset_pending = false;
        true
    }

    /// Stops all sharing: every later string is written inline, with no
    /// definitions and no references. The encoder does this itself when it finds an
    /// address that was reused by a string of another length.
    pub fn poison(&mut self) {
        self.poisoned = true;
    }

    /// Whether [`poison`](Self::poison) was called or the encoder poisoned itself.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Serializes one complete CBOR document using [`StreamStrings::string`] for
    /// each string occurrence, including object keys. The callback must serialize
    /// directly in wire order, without pre-serializing or discarding definitions.
    ///
    /// On error, appended bytes are discarded and the encoder is poisoned, so later
    /// documents cannot reference definitions that were never written.
    pub fn encode_with<E>(
        &mut self,
        out: &mut Vec<u8>,
        encode: impl FnOnce(&StreamStrings<'_>, &mut Vec<u8>) -> Result<(), E>,
    ) -> Result<DocUse, E> {
        let (defs, refs) = (self.stats.defs, self.stats.refs);
        let start = out.len();
        let mut choose = |string: &StreamString<'_>| self.choose(string);
        let strings = StreamStrings {
            choose: RefCell::new(&mut choose),
        };
        if let Err(error) = encode(&strings, out) {
            out.truncate(start);
            self.poison();
            return Err(error);
        }
        Ok(DocUse {
            defs: self.stats.defs - defs,
            refs: self.stats.refs - refs,
        })
    }

    /// Counters and the current size of the ID table.
    pub fn stats(&self) -> StreamEncoderStats {
        self.stats
    }

    fn choose(&mut self, string: &StreamString<'_>) -> Choice {
        if self.poisoned {
            return Choice::Text;
        }
        // Occurrences left inline because the string was shorter than 8 bytes,
        // which includes the empty string and every string kept inside its handle.
        let len = string.as_str().len();
        if len < MIN_DEFINED_LEN {
            // The empty string and strings kept inside their handle, all below `MIN_DEFINED_LEN`.
            return Choice::Text;
        }
        if let Some((id, defined_len)) = self.table.get(string) {
            // Times a registered address belonged to a string of another length, which
            // means its string was freed and the address reused. Each one poisons the
            // encoder: that string and every later one is written inline. Zero proves
            // nothing about lifetime safety: a string of the same length
            // at a reused address goes unnoticed.
            // only a length change shows a recycled address; an equal length is
            // prevented by the caller keeping the allocation alive, never by this check.
            if defined_len != len {
                self.poisoned = true;
                return Choice::Text;
            }
            self.stats.refs += 1;
            // Text of 8 or more bytes takes at least 9; a reference takes its tag and at most 5.
            self.stats.saved_bytes +=
                cbor_head_len(len as u64) + len as u64 - 1 - cbor_head_len(u64::from(id));
            return Choice::Reference(id);
        }
        // Cumulative definition positions must remain representable in a u32 stream header.
        if self.stats.defs >= u64::from(u32::MAX) {
            return Choice::Text;
        }
        if !self.table.can_define(string) {
            return Choice::Text;
        }
        // Occurrences left inline because the table had no room or could not represent the string.
        if u32::try_from(len).is_err() {
            return Choice::Text;
        }
        let id = self.next_id();
        if !self.table.insert(string, id, self.budget) {
            self.reset_pending = !self.table.is_empty();
            return Choice::Text;
        }
        self.stats.defs += 1;
        self.stats.def_bytes += len as u64;
        Choice::Define
    }
}

impl<D: StringDictionary> fmt::Debug for StreamEncoder<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamEncoder")
            .field("next_id", &self.next_id())
            .field("poisoned", &self.poisoned)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

/// String wrappers for a single document, borrowed from its stream encoder.
pub struct StreamStrings<'a> {
    choose: RefCell<&'a mut dyn FnMut(&StreamString<'_>) -> Choice>,
}

impl fmt::Debug for StreamStrings<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamStrings").finish_non_exhaustive()
    }
}

impl<'e> StreamStrings<'e> {
    /// Wraps a string without registering it until the wrapper is serialized.
    pub fn string<'a, T: Into<StreamString<'a>> + 'a>(
        &'a self,
        string: T,
    ) -> impl Serialize + 'a + use<'a, 'e, T> {
        struct StringValue<'a, 'e>(&'a StreamStrings<'e>, StreamString<'a>);
        impl Serialize for StringValue<'_, '_> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let choice = (self.0.choose.borrow_mut())(&self.1);
                serialize_choice(choice, self.1.as_str(), serializer)
            }
        }
        StringValue(self, string.into())
    }
}

/// Writes one string occurrence of a document as text, a definition or a
/// reference.
pub(super) fn serialize_string<S: serde::Serializer, D: StringDictionary>(
    encoder: &EncoderCell<'_, D>,
    string: &IString,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    // The choice is made, and a definition registered, right before the bytes are
    // written, and no borrow is held while they are.
    let choice = encoder.borrow_mut().choose(&string.into());
    serialize_choice(choice, string.as_str(), serializer)
}

fn serialize_choice<S: serde::Serializer>(
    choice: Choice,
    string: &str,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match choice {
        Choice::Text => serializer.serialize_str(string),
        Choice::Define => RequireAny(TAG_DEF, string).serialize(serializer),
        Choice::Reference(id) => RequireAny(TAG_REF, id).serialize(serializer),
    }
}

/// Decodes the documents of a stream written by a [`StreamEncoder`].
///
/// Documents must be decoded in the order they were encoded. Each definition
/// adds a string to the table, and a reference to an ID the table does not hold
/// is an error. After any error the decoder refuses further documents, since the
/// failed one may have held definitions that later references depend on.
#[derive(Default)]
pub struct StreamDecoder {
    entries: Vec<IString>,
    stats: StreamDecoderStats,
    failed: bool,
}

impl StreamDecoder {
    /// Creates a decoder with an empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of strings defined so far, which is the ID the next definition
    /// receives. A document whose encoder-side [`next_id`](StreamEncoder::next_id)
    /// differs from this was not decoded in order.
    /// Counts only the current dictionary after a reset.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing was defined yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Releases this dictionary's handles without discarding statistics or clearing failures.
    /// Already decoded values retain their own string handles.
    pub fn reset(&mut self) {
        self.entries.clear();
    }

    /// Decodes one document, which must be the whole of `payload`.
    ///
    /// Tags 6 and 7 are definitions and references here, and no other tag is a
    /// string tag. A definition must hold text of at least 8 bytes; a reference
    /// must hold a nonnegative integer below [`len`](Self::len). Typed arrays and
    /// the nesting limit behave as in
    /// [`decode_with_depth_limit`](crate::decode_with_depth_limit).
    /// Trailing bytes are an error.
    pub fn decode(&mut self, payload: &[u8], limit_depth: bool) -> Result<IValue, CborDecodeError> {
        if self.failed {
            return Err(CborDecodeError::DecodeError);
        }
        let mut rest = payload;
        let decoder = RefCell::new(&mut *self);
        let decoded = super::decode_one_value(&mut rest, limit_depth, StringRefs::Stream(&decoder));
        let result = match decoded {
            Ok(_) if !rest.is_empty() => Err(CborDecodeError::DecodeError),
            other => other,
        };
        self.failed |= result.is_err();
        result
    }

    /// Counters of the documents decoded so far.
    pub fn stats(&self) -> StreamDecoderStats {
        self.stats
    }

    pub(super) fn define_str(&mut self, text: &str) -> Result<IString, CborDecodeError> {
        if text.len() < MIN_DEFINED_LEN {
            return Err(CborDecodeError::InvalidValue);
        }
        let string = IString::from(text);
        self.stats.defs += 1;
        self.stats.def_bytes += string.len() as u64;
        self.entries.push(string.clone());
        Ok(string)
    }

    pub(super) fn reference_id(&mut self, id: usize) -> Result<IString, CborDecodeError> {
        let string = self
            .entries
            .get(id)
            .cloned()
            .ok_or(CborDecodeError::InvalidStringReference)?;
        self.stats.refs += 1;
        Ok(string)
    }
}

impl fmt::Debug for StreamDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamDecoder")
            .field("len", &self.len())
            .field("failed", &self.failed)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IArray;
    use mockalloc::record_allocs;

    fn config() -> StreamEncoderConfig {
        StreamEncoderConfig {
            table_budget_bytes: 1 << 20,
        }
    }

    /// An array that holds every string twice, so that each has two owners.
    fn shared_array(texts: &[&str]) -> IValue {
        let strings: Vec<IValue> = texts
            .iter()
            .chain(texts)
            .map(|text| IValue::from(IString::from(*text)))
            .collect();
        IArray::try_from(strings).unwrap().into()
    }

    #[test]
    fn budget_model_covers_what_the_allocator_sees() {
        let mut model_peak = 0;
        let mut final_bytes = 0;
        let allocations = record_allocs(|| {
            let mut table = AddressTable::default();
            for index in 1..=3_000usize {
                if table.len() == table.capacity() {
                    let old = table_bytes(table.capacity());
                    model_peak =
                        model_peak.max(old + table_alloc_bytes(next_buckets(table.capacity())));
                }
                table.insert(index * 8, Slot { id: 0, len: 0 });
            }
            final_bytes = table_bytes(table.capacity());
        });
        let peak = allocations.peak_mem() as usize;
        assert!(peak <= model_peak, "model {model_peak} below actual {peak}");
        assert!(
            model_peak - peak <= 64,
            "model {model_peak} far above actual {peak}"
        );
        assert!(final_bytes <= peak);
    }

    #[test]
    fn repeated_resets_keep_writer_allocations_within_budget() {
        let texts: Vec<_> = (0..40)
            .map(|i| format!("reset allocation string {i:04}"))
            .collect();
        let refs: Vec<_> = texts.iter().map(String::as_str).collect();
        let value = shared_array(&refs);
        let mut bytes = Vec::with_capacity(4096);
        let mut resets = 0;
        let allocations = record_allocs(|| {
            let mut encoder = StreamEncoder::new(StreamEncoderConfig {
                table_budget_bytes: 700,
                ..config()
            });
            for _ in 0..64 {
                resets += usize::from(encoder.reset_if_needed());
                bytes.clear();
                encoder.encode_into(&mut bytes, &value);
                assert!(encoder.table.used_bytes() <= 700);
            }
            assert!(encoder.stats().defs > 40);
        });
        assert!(resets > 0);
        assert!(allocations.peak_mem() <= 700);
    }

    #[test]
    fn cumulative_definition_positions_do_not_wrap_after_resets() {
        let first = shared_array(&["a previously defined string"]);
        let second = shared_array(&["a string beyond the position limit"]);
        let mut encoder = StreamEncoder::new(config());
        encoder.encode_into(&mut Vec::new(), &first);
        encoder.stats.defs = u64::from(u32::MAX);
        assert_eq!(
            encoder.encode_into(&mut Vec::new(), &second),
            DocUse::default()
        );
        assert!(!encoder.reset_if_needed());
        assert_eq!(encoder.encode_into(&mut Vec::new(), &first).refs, 2);
        assert_eq!(encoder.stats().defs, u64::from(u32::MAX));
    }

    #[test]
    fn saved_bytes_are_the_exact_difference_from_the_plain_encoding() {
        // Lengths and ids on both sides of each CBOR head size.
        let texts: Vec<String> = [8, 23, 24, 255, 256, 65_535, 65_536]
            .iter()
            .map(|&len| "s".repeat(len))
            .chain((0..30).map(|i| format!("one of many ids {i:02}")))
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let value = shared_array(&refs);
        let mut encoder = StreamEncoder::new(config());
        let mut shared = Vec::new();
        encoder.encode_into(&mut shared, &value);
        let stats = encoder.stats();
        assert_eq!(stats.refs, texts.len() as u64);
        assert_eq!(
            crate::encode(&value).len() as u64,
            shared.len() as u64 + stats.saved_bytes - stats.defs
        );
    }

    #[test]
    fn table_growth_stays_within_the_budget_at_scale() {
        const STRINGS: usize = 60_000;
        let strings: Vec<IString> = (0..STRINGS)
            .map(|index| IString::from(format!("budget shared string {index:06}")))
            .collect();
        let doc: IValue = IArray::try_from(
            strings
                .iter()
                .cloned()
                .map(IValue::from)
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .into();
        // Returns how many strings were defined and the bytes the table holds.
        let encode_with_budget = |budget| {
            let mut encoder = StreamEncoder::new(StreamEncoderConfig {
                table_budget_bytes: budget,
            });
            let mut bytes = Vec::new();
            let used = encoder.encode_into(&mut bytes, &doc);
            assert_eq!(used.defs, encoder.table.len() as u64);
            assert!(encoder.table.used_bytes() <= budget, "{encoder:?}");
            assert_eq!(StreamDecoder::new().decode(&bytes, true).unwrap(), doc);
            (used.defs, encoder.table.used_bytes())
        };
        let (all, needed) = encode_with_budget(1 << 30);
        assert_eq!(all, STRINGS as u64);
        // Half of what the whole table needs leaves no room for it to grow to that size.
        let (defined, _) = encode_with_budget(needed / 2);
        assert!(defined > 0 && defined < STRINGS as u64, "{defined}");
    }
}
