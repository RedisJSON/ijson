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

use std::cell::RefCell;
use std::fmt;

use ciborium::tag::RequireAny;
use ciborium::value::Value;
use hashbrown::HashMap;
use serde::Serialize;

use super::{reference_id, EncodedValue, StringRefs, TAG_DEF, TAG_REF};
use crate::{CborDecodeError, IString, IValue};

/// The encoder as the serializers see it, borrowed for one string at a time.
pub(super) type EncoderCell<'a> = RefCell<&'a mut StreamEncoder>;

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
    pub table_budget_bytes: usize,
    /// Bound on the UTF-8 bytes of the strings the encoder defines; zero sets no bound. A
    /// definition that would pass it is written inline, and the strings defined so far keep
    /// being referenced. A reader keeps every defined string until the end of the stream, so
    /// this bound is also the most memory the strings of the stream can pin in a reader.
    pub max_def_bytes: usize,
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
}

/// Counters of a [`StreamDecoder`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamDecoderStats {
    /// Definitions read, equal to the length of the ID table.
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
/// The caller must keep every defined string allocation alive and at the same
/// address until the stream ends. The encoder stores addresses, not owned handles;
/// dropping or moving those strings can make later references resolve to wrong text.
pub struct StreamEncoder {
    /// Heap budget for the address table, including growth allocations.
    budget: usize,
    /// Cumulative UTF-8 definition payload limit; zero means unlimited, not a total RAM bound.
    max_def_bytes: usize,
    /// Maps string allocation addresses to IDs and lengths; does not own the strings.
    table: AddressTable,
    /// Whether this document may add definitions; temporary Flex documents may only reference.
    allow_define: bool,
    /// Sticky failure state: encode later strings inline, without DEF or REF.
    poisoned: bool,
    /// Stream counters for reporting; definition bytes also enforce max_def_bytes.
    stats: StreamEncoderStats,
}

impl StreamEncoder {
    /// Creates an encoder that has defined nothing yet.
    pub fn new(config: StreamEncoderConfig) -> Self {
        Self {
            budget: config.table_budget_bytes,
            max_def_bytes: config.max_def_bytes,
            table: AddressTable::default(),
            allow_define: false,
            poisoned: false,
            stats: StreamEncoderStats::default(),
        }
    }

    /// The ID the next definition receives, which is the number of strings
    /// defined so far. Read it before encoding a document to learn the ID of its
    /// first definition. A poisoned encoder defines nothing more, so what it
    /// reports no longer matters.
    pub fn next_id(&self) -> u32 {
        // IDs stop below `u32::MAX`, see `has_room`.
        // Strings in the ID table.
        self.table.len() as u32
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

    /// Appends the CBOR encoding of `value` to `out`, after whatever `out`
    /// already holds, and reports what the document used.
    ///
    /// When the returned [`DocUse`] is zero the bytes are exactly those of
    /// [`encode`](crate::encode). With `allow_define` false no string is defined,
    /// but strings defined earlier are still referenced. If an address turns out to
    /// be reused by a string of another length, the encoder is poisoned and that
    /// string and the rest of the document are written inline.
    pub fn encode_into(&mut self, out: &mut Vec<u8>, value: &IValue, allow_define: bool) -> DocUse {
        let (defs, refs) = (self.stats.defs, self.stats.refs);
        self.allow_define = allow_define;
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

    /// Counters and the current size of the ID table.
    pub fn stats(&self) -> StreamEncoderStats {
        self.stats
    }

    /// Heap bytes held by the ID table.
    fn used_bytes(&self) -> usize {
        table_bytes(self.table.capacity())
    }

    /// Whether one more definition fits the budget, counting the old and the new
    /// allocation while the table grows. No allocation can fail gracefully, so the
    /// budget is checked in advance. A table that is full and cannot grow stays so,
    /// because nothing is added to it from then on.
    fn has_room(&self) -> bool {
        let table_growth = if self.table.len() == self.table.capacity() {
            table_alloc_bytes(next_buckets(self.table.capacity()))
        } else {
            0
        };
        (table_growth == 0 || self.used_bytes() + table_growth <= self.budget)
            && self.table.len() < u32::MAX as usize
    }

    fn choose(&mut self, string: &IString) -> Choice {
        if self.poisoned {
            return Choice::Text;
        }
        // Occurrences left inline because the string was shorter than 8 bytes,
        // which includes the empty string and every string kept inside its handle.
        let Some((address, len, owners)) = string.heap_identity() else {
            // The empty string and strings kept inside their handle, all below `MIN_DEFINED_LEN`.
            return Choice::Text;
        };
        if len < MIN_DEFINED_LEN {
            return Choice::Text;
        }
        if let Some(&slot) = self.table.get(&address.get()) {
            // Times a registered address belonged to a string of another length, which
            // means its string was freed and the address reused. Each one poisons the
            // encoder: that string and every later one is written inline. Zero proves
            // nothing about lifetime safety: a string of the same length
            // at a reused address goes unnoticed.
            // only a length change shows a recycled address; an equal length is
            // prevented by the caller keeping the allocation alive, never by this check.
            if slot.len as usize != len {
                self.poisoned = true;
                return Choice::Text;
            }
            self.stats.refs += 1;
            return Choice::Reference(slot.id);
        }
        if !self.allow_define {
            return Choice::Text;
        }
        // Occurrences left inline because the string had a single owner.
        if owners < 2 {
            return Choice::Text;
        }
        // Occurrences left inline because the table had no room, could not
        // represent the string, or the bound on defined bytes
        // ([`StreamEncoderConfig::max_def_bytes`]) would have been passed.
        let Ok(slot_len) = u32::try_from(len) else {
            return Choice::Text;
        };
        if self.max_def_bytes != 0 && self.stats.def_bytes + len as u64 > self.max_def_bytes as u64
        {
            return Choice::Text;
        }
        if !self.has_room() {
            return Choice::Text;
        }
        let id = self.next_id();
        self.table.insert(address.get(), Slot { id, len: slot_len });
        self.stats.defs += 1;
        self.stats.def_bytes += len as u64;
        Choice::Define
    }
}

impl fmt::Debug for StreamEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamEncoder")
            .field("next_id", &self.next_id())
            .field("poisoned", &self.poisoned)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

/// Writes one string occurrence of a document as text, a definition or a
/// reference.
pub(super) fn serialize_string<S: serde::Serializer>(
    encoder: &EncoderCell<'_>,
    string: &IString,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    // The choice is made, and a definition registered, right before the bytes are
    // written, and no borrow is held while they are.
    let choice = encoder.borrow_mut().choose(string);
    match choice {
        Choice::Text => serializer.serialize_str(string.as_str()),
        Choice::Define => RequireAny(TAG_DEF, string.as_str()).serialize(serializer),
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
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing was defined yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
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

    pub(super) fn resolve(&mut self, tag: u64, inner: Value) -> Result<IString, CborDecodeError> {
        if tag == TAG_DEF {
            self.define(inner)
        } else {
            self.reference(inner)
        }
    }

    fn define(&mut self, inner: Value) -> Result<IString, CborDecodeError> {
        let Value::Text(text) = inner else {
            return Err(CborDecodeError::InvalidValue);
        };
        if text.len() < MIN_DEFINED_LEN {
            return Err(CborDecodeError::InvalidValue);
        }
        let string = IString::from(text);
        self.stats.defs += 1;
        self.stats.def_bytes += string.len() as u64;
        self.entries.push(string.clone());
        Ok(string)
    }

    fn reference(&mut self, inner: Value) -> Result<IString, CborDecodeError> {
        let string = self
            .entries
            .get(reference_id(inner)?)
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
            max_def_bytes: 0,
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
    fn bucket_model_follows_the_growth_of_the_table() {
        let mut table = AddressTable::default();
        assert_eq!(table.capacity(), 0);
        for index in 1..=5_000usize {
            let capacity = table.capacity();
            let predicted = (table.len() == capacity).then(|| next_buckets(capacity));
            table.insert(index * 8, Slot { id: 0, len: 0 });
            match predicted {
                Some(buckets) => assert_eq!(buckets_for_capacity(table.capacity()), buckets),
                None => assert_eq!(table.capacity(), capacity),
            }
        }
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
    fn definitions_are_registered_in_the_order_their_bytes_are_written() {
        let texts = [
            "first string defined",
            "second string defined",
            "third string defined",
        ];
        let strings: Vec<IString> = texts.iter().map(|text| IString::from(*text)).collect();
        // Keys come before their values, and arrays in element order.
        let mut object = crate::IObject::new();
        object
            .insert(strings[2].clone(), strings[1].clone())
            .unwrap();
        let items: Vec<IValue> = vec![
            object.into(),
            strings[0].clone().into(),
            strings[2].clone().into(),
        ];
        let doc: IValue = IArray::try_from(items).unwrap().into();
        let mut encoder = StreamEncoder::new(config());
        let mut out = Vec::new();
        assert_eq!(encoder.encode_into(&mut out, &doc, true).defs, 3);
        let ids: Vec<u32> = [2, 1, 0]
            .iter()
            .map(|index| {
                let address = strings[*index].heap_identity().unwrap().0;
                encoder.table[&address.get()].id
            })
            .collect();
        assert_eq!(ids, [0, 1, 2]);
        let decoded = StreamDecoder::new().decode(&out, true).unwrap();
        assert_eq!(decoded, doc);
    }

    #[test]
    fn a_registered_address_with_another_length_poisons_the_encoder() {
        let doc = shared_array(&["a string that gets registered"]);
        let mut encoder = StreamEncoder::new(config());
        let mut out = Vec::new();
        assert_eq!(
            encoder.encode_into(&mut out, &doc, true),
            DocUse { defs: 1, refs: 1 }
        );
        let address = doc.as_array().unwrap()[0]
            .as_string()
            .unwrap()
            .heap_identity()
            .unwrap()
            .0;
        // As if the string had been freed and another one of this length took its place.
        encoder.table.get_mut(&address.get()).unwrap().len += 1;

        out.clear();
        assert_eq!(encoder.encode_into(&mut out, &doc, true), DocUse::default());
        assert_eq!(out, crate::encode(&doc));
        assert!(encoder.is_poisoned());
        assert_eq!((encoder.stats().defs, encoder.stats().refs), (1, 1));
    }

    #[test]
    fn budget_of_seven_hundred_bytes_holds_fourteen_entries() {
        let texts: Vec<String> = (0..40)
            .map(|index| format!("budget string {index:04}"))
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let doc = shared_array(&refs);
        let mut encoder = StreamEncoder::new(StreamEncoderConfig {
            table_budget_bytes: 700,
            ..config()
        });
        let mut bytes = Vec::new();
        let used = encoder.encode_into(&mut bytes, &doc, true);
        // 84, 152, 288 and 560 bytes for 4, 8, 16 and 32 buckets: the move from 16
        // to 32 buckets needs 288 + 560 bytes at once.
        assert_eq!(used, DocUse { defs: 14, refs: 14 });
        assert_eq!(encoder.table.len(), 14);
        assert_eq!(encoder.used_bytes(), 288);
        // The 26 strings that found no room are met twice.
        let items: Vec<Value> = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(items.iter().filter(|item| item.is_text()).count(), 52);
    }

    #[test]
    fn table_growth_respects_the_budget_at_scale() {
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
        let registered = |budget| {
            let mut encoder = StreamEncoder::new(StreamEncoderConfig {
                table_budget_bytes: budget,
                ..config()
            });
            let mut bytes = Vec::new();
            let used = encoder.encode_into(&mut bytes, &doc, true);
            assert_eq!(used.defs, encoder.table.len() as u64);
            assert!(encoder.used_bytes() <= budget, "{encoder:?}");
            assert_eq!(StreamDecoder::new().decode(&bytes, true).unwrap(), doc);
            (used.defs, used.defs < STRINGS as u64)
        };
        // The table fits the larger budget; the smaller budget leaves no room for it to grow.
        assert_eq!(registered(3_670_016), (STRINGS as u64, false));
        let (defined, ran_out_of_room) = registered(2_000_000);
        assert!(ran_out_of_room);
        assert!((28_000..STRINGS as u64).contains(&defined), "{defined}");
    }
}
