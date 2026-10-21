//! CBOR encode/decode for [`IValue`], preserving typed array tags via RFC 8746.
//!
//! Typed homogeneous arrays are encoded as `Tag(rfc8746_tag, Bytes(raw_le_bytes))`.
//! BF16 arrays use private tag `0x10000` (no standard RFC 8746 equivalent).
//!
//! Use [`encode`] / [`decode`] for raw CBOR, or [`encode_compressed`] /
//! [`decode_compressed`] for zstd-compressed CBOR.
//!
//! [`StreamEncoder`] and [`StreamDecoder`] share strings across the documents
//! of one ordered stream: the first use of a string defines it inline and later
//! uses refer to it by an implicit ID.

use std::fmt;

use bytemuck::Pod;
#[cfg(test)]
use ciborium::value::Integer;
use ciborium::value::Value;
use ciborium_ll::tag;
use half::{bf16, f16};
use serde::de::DeserializeSeed;
use serde::Serialize;

/// Converts a typed slice to/from little-endian bytes using each type's own
/// `to_le_bytes` / `from_le_bytes`, which are correct on every host endianness.
trait LeBytes: Pod + Copy {
    fn slice_to_le_bytes(s: &[Self]) -> Vec<u8>;
    fn slice_from_le_bytes(bytes: &[u8]) -> Result<Vec<Self>, CborDecodeError>;
}

macro_rules! impl_le_bytes {
    ($($t:ty => $n:literal),* $(,)?) => {$(
        impl LeBytes for $t {
            fn slice_to_le_bytes(s: &[Self]) -> Vec<u8> {
                s.iter().flat_map(|v| v.to_le_bytes()).collect()
            }
            fn slice_from_le_bytes(bytes: &[u8]) -> Result<Vec<Self>, CborDecodeError> {
                if bytes.len() % $n != 0 {
                    return Err(CborDecodeError::CastError);
                }
                Ok(bytes
                    .chunks_exact($n)
                    .map(|c| {
                        // SAFETY: `chunks_exact($n)` guarantees every chunk
                        // is exactly $n bytes, matching [u8; $n].
                        let arr: [u8; $n] = c.try_into().unwrap();
                        Self::from_le_bytes(arr)
                    })
                    .collect())
            }
        }
    )*};
}

impl_le_bytes!(
    i8  => 1, u8  => 1,
    i16 => 2, u16 => 2, f16 => 2, bf16 => 2,
    i32 => 4, u32 => 4, f32 => 4,
    i64 => 8, u64 => 8, f64 => 8,
);

use crate::array::ArraySliceRef;
use crate::{DestructuredRef, IArray, INumber, IObject, IString, IValue};

mod stream;
pub use stream::{
    AddressDictionary, ContentDictionary, DocUse, StreamDecoder, StreamDecoderStats, StreamEncoder,
    StreamEncoderConfig, StreamEncoderStats, StreamString, StreamStrings, StringDictionary,
};
use stream::{DecoderCell, EncoderCell};

use tag::{
    TYPED_F16_LE as TAG_F16_LE, TYPED_F32_LE as TAG_F32_LE, TYPED_F64_LE as TAG_F64_LE,
    TYPED_I16_LE as TAG_I16_LE, TYPED_I32_LE as TAG_I32_LE, TYPED_I64_LE as TAG_I64_LE,
    TYPED_I8 as TAG_I8, TYPED_U16_LE as TAG_U16_LE, TYPED_U32_LE as TAG_U32_LE,
    TYPED_U64_LE as TAG_U64_LE, TYPED_U8 as TAG_U8,
};

/// Private CBOR tag for BF16 arrays (no RFC 8746 standard tag exists for BF16).
const TAG_BF16_LE: u64 = 0x10000;

/// CBOR tag that defines a stream string: it wraps the text, which takes the next ID.
pub(crate) const TAG_DEF: u64 = 6;

/// CBOR tag that references an earlier stream definition by its ID.
pub(crate) const TAG_REF: u64 = 7;

const MAX_DEPTH: u32 = 128;

/// Reads the ID of a string reference: a nonnegative integer that fits `usize`.
fn reference_id(value: Value) -> Result<usize, CborDecodeError> {
    let Value::Integer(id) = value else {
        return Err(CborDecodeError::InvalidStringReference);
    };
    usize::try_from(id).map_err(|_| CborDecodeError::InvalidStringReference)
}

/// Error returned when CBOR decoding fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CborDecodeError {
    /// The CBOR stream was malformed or could not be parsed.
    DecodeError,
    /// An unrecognised CBOR tag was encountered where a typed array was expected.
    UnknownTag(u64),
    /// A CBOR map key was not a text string.
    InvalidValue,
    /// An array allocation failed.
    AllocError,
    /// Nesting depth exceeded the limit.
    DepthLimitExceeded,
    /// Failed to reinterpret a byte slice.
    CastError,
    /// Zstd decompression failed.
    DecompressError,
    /// A stream string reference had an invalid payload or an out-of-range ID.
    InvalidStringReference,
}

impl fmt::Display for CborDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CborDecodeError::DecodeError => write!(f, "CBOR decode error"),
            CborDecodeError::UnknownTag(t) => write!(f, "unknown CBOR tag: {t}"),
            CborDecodeError::InvalidValue => write!(f, "unexpected CBOR value type"),
            CborDecodeError::AllocError => write!(f, "memory allocation failed"),
            CborDecodeError::DepthLimitExceeded => write!(f, "nesting depth limit exceeded"),
            CborDecodeError::CastError => write!(f, "failed to cast byte slice"),
            CborDecodeError::DecompressError => write!(f, "zstd decompression failed"),
            CborDecodeError::InvalidStringReference => write!(f, "invalid string reference"),
        }
    }
}

impl std::error::Error for CborDecodeError {}

// ── Encode ────────────────────────────────────────────────────────────────────

/// Encodes an [`IValue`] tree into CBOR bytes, preserving typed array tags via
/// RFC 8746.
pub fn encode(value: &IValue) -> Vec<u8> {
    let mut out = Vec::new();
    encode_into(value, &mut out);
    out
}

/// Appends CBOR bytes to an existing buffer, preserving its prefix and typed array tags.
pub fn encode_into(value: &IValue, out: &mut Vec<u8>) {
    let cbor = EncodedValue {
        value,
        strings: Inline,
    };
    ciborium::into_writer(&cbor, out).expect("write to Vec never fails");
}

/// Encodes an [`IValue`] tree as CBOR and then compresses it with zstd (level 3).
///
/// Use [`decode_compressed`] to decode the output.
pub fn encode_compressed(value: &IValue) -> Vec<u8> {
    let raw = encode(value);
    zstd::bulk::Compressor::default()
        .compress(&raw)
        .expect("zstd compress")
}

#[cfg(test)]
fn ivalue_to_cbor(value: &IValue) -> Value {
    match value.destructure_ref() {
        DestructuredRef::Null => Value::Null,
        DestructuredRef::Bool(b) => Value::Bool(b),
        DestructuredRef::Number(n) => number_to_cbor(n),
        DestructuredRef::String(s) => string_to_cbor(s),
        DestructuredRef::Array(a) => array_to_cbor(a),
        DestructuredRef::Object(o) => object_to_cbor(o),
    }
}

#[cfg(test)]
fn string_to_cbor(value: &IString) -> Value {
    Value::Text(value.as_str().to_owned())
}

#[cfg(test)]
fn number_to_cbor(n: &INumber) -> Value {
    if n.has_decimal_point() {
        Value::Float(n.to_f64().unwrap())
    } else if let Some(i) = n.to_i64() {
        Value::Integer(Integer::from(i))
    } else {
        Value::Integer(Integer::from(n.to_u64().unwrap()))
    }
}

#[cfg(test)]
fn array_to_cbor(a: &IArray) -> Value {
    match a.as_slice() {
        ArraySliceRef::Heterogeneous(s) => Value::Array(s.iter().map(ivalue_to_cbor).collect()),
        ArraySliceRef::I8(s) => typed_le_tag(TAG_I8, s),
        ArraySliceRef::U8(s) => typed_le_tag(TAG_U8, s),
        ArraySliceRef::I16(s) => typed_le_tag(TAG_I16_LE, s),
        ArraySliceRef::U16(s) => typed_le_tag(TAG_U16_LE, s),
        ArraySliceRef::F16(s) => typed_le_tag(TAG_F16_LE, s),
        ArraySliceRef::BF16(s) => typed_le_tag(TAG_BF16_LE, s),
        ArraySliceRef::I32(s) => typed_le_tag(TAG_I32_LE, s),
        ArraySliceRef::U32(s) => typed_le_tag(TAG_U32_LE, s),
        ArraySliceRef::F32(s) => typed_le_tag(TAG_F32_LE, s),
        ArraySliceRef::I64(s) => typed_le_tag(TAG_I64_LE, s),
        ArraySliceRef::U64(s) => typed_le_tag(TAG_U64_LE, s),
        ArraySliceRef::F64(s) => typed_le_tag(TAG_F64_LE, s),
    }
}

#[cfg(test)]
fn object_to_cbor(o: &IObject) -> Value {
    Value::Map(
        o.iter()
            .map(|(k, v)| (string_to_cbor(k), ivalue_to_cbor(v)))
            .collect(),
    )
}

#[cfg(test)]
fn typed_le_tag<T: LeBytes>(tag: u64, s: &[T]) -> Value {
    Value::Tag(tag, Box::new(Value::Bytes(T::slice_to_le_bytes(s))))
}

/// How the strings of a document are written. It is a type parameter, not a value checked
/// per string, so that a document encoded without a stream runs code with no trace of the
/// stream: a per-string branch alone made the plain encoder about 40% slower on an array of
/// eight million small values.
trait WriteStrings: Copy {
    fn write_string<S: serde::Serializer>(
        self,
        string: &IString,
        serializer: S,
    ) -> Result<S::Ok, S::Error>;
}

/// Every string is text.
#[derive(Clone, Copy)]
struct Inline;

impl WriteStrings for Inline {
    #[inline(always)]
    fn write_string<S: serde::Serializer>(
        self,
        string: &IString,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(string.as_str())
    }
}

/// A string is text, a definition or a reference, as the stream encoder decides.
impl<D: StringDictionary> WriteStrings for &EncoderCell<'_, D> {
    fn write_string<S: serde::Serializer>(
        self,
        string: &IString,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        stream::serialize_string(self, string, serializer)
    }
}

struct EncodedValue<'a, W> {
    value: &'a IValue,
    strings: W,
}

struct EncodedString<'a, W>(&'a IString, W);

impl<W: WriteStrings> serde::Serialize for EncodedString<'_, W> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.1.write_string(self.0, serializer)
    }
}

struct EncodedBytes<'a>(&'a [u8]);

impl serde::Serialize for EncodedBytes<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(self.0)
    }
}

fn serialize_typed<T: LeBytes, S: serde::Serializer>(
    tag: u64,
    values: &[T],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let converted;
    let bytes = if cfg!(target_endian = "little") {
        bytemuck::cast_slice(values)
    } else {
        converted = T::slice_to_le_bytes(values);
        &converted
    };
    ciborium::tag::RequireAny(tag, EncodedBytes(bytes)).serialize(serializer)
}

impl<W: WriteStrings> serde::Serialize for EncodedValue<'_, W> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self.value.destructure_ref() {
            DestructuredRef::Null => serializer.serialize_unit(),
            DestructuredRef::Bool(value) => serializer.serialize_bool(value),
            DestructuredRef::Number(value) => {
                if value.has_decimal_point() {
                    serializer.serialize_f64(value.to_f64().unwrap())
                } else if let Some(value) = value.to_i64() {
                    serializer.serialize_i64(value)
                } else {
                    serializer.serialize_u64(value.to_u64().unwrap())
                }
            }
            DestructuredRef::String(value) => {
                EncodedString(value, self.strings).serialize(serializer)
            }
            DestructuredRef::Object(value) => {
                let mut map = serializer.serialize_map(Some(value.len() as usize))?;
                for (key, value) in value.iter() {
                    map.serialize_entry(
                        &EncodedString(key, self.strings),
                        &EncodedValue {
                            value,
                            strings: self.strings,
                        },
                    )?;
                }
                map.end()
            }
            DestructuredRef::Array(value) => match value.as_slice() {
                ArraySliceRef::Heterogeneous(values) => {
                    let mut seq = serializer.serialize_seq(Some(values.len()))?;
                    for value in values {
                        seq.serialize_element(&EncodedValue {
                            value,
                            strings: self.strings,
                        })?;
                    }
                    seq.end()
                }
                ArraySliceRef::I8(values) => serialize_typed(TAG_I8, values, serializer),
                ArraySliceRef::U8(values) => serialize_typed(TAG_U8, values, serializer),
                ArraySliceRef::I16(values) => serialize_typed(TAG_I16_LE, values, serializer),
                ArraySliceRef::U16(values) => serialize_typed(TAG_U16_LE, values, serializer),
                ArraySliceRef::F16(values) => serialize_typed(TAG_F16_LE, values, serializer),
                ArraySliceRef::BF16(values) => serialize_typed(TAG_BF16_LE, values, serializer),
                ArraySliceRef::I32(values) => serialize_typed(TAG_I32_LE, values, serializer),
                ArraySliceRef::U32(values) => serialize_typed(TAG_U32_LE, values, serializer),
                ArraySliceRef::F32(values) => serialize_typed(TAG_F32_LE, values, serializer),
                ArraySliceRef::I64(values) => serialize_typed(TAG_I64_LE, values, serializer),
                ArraySliceRef::U64(values) => serialize_typed(TAG_U64_LE, values, serializer),
                ArraySliceRef::F64(values) => serialize_typed(TAG_F64_LE, values, serializer),
            },
        }
    }
}

/// How the tagged string references of a document resolve.
#[derive(Clone, Copy)]
enum StringRefs<'a> {
    /// No tag is a string reference.
    None,
    /// Definitions and references of a stream use tags 6 and 7.
    Stream(&'a DecoderCell<'a>),
}

impl StringRefs<'_> {
    fn is_string_tag(self, tag: u64) -> bool {
        match self {
            Self::None => false,
            Self::Stream(_) => tag == TAG_DEF || tag == TAG_REF,
        }
    }

    /// Resolves the content of a tag accepted by [`Self::is_string_tag`].
    fn resolve(self, tag: u64, inner: Value) -> Result<IString, CborDecodeError> {
        match self {
            Self::None => Err(CborDecodeError::UnknownTag(tag)),
            Self::Stream(decoder) => decoder.borrow_mut().resolve(tag, inner),
        }
    }
}

struct DecodeContext<'a> {
    refs: StringRefs<'a>,
    limit_depth: bool,
    error: std::cell::Cell<Option<CborDecodeError>>,
}

impl DecodeContext<'_> {
    fn fail<E: serde::de::Error>(&self, error: CborDecodeError) -> E {
        self.error.set(Some(error.clone()));
        E::custom(error)
    }
}

#[derive(Clone, Copy)]
struct ValueSeed<'a, 'd> {
    context: &'a DecodeContext<'d>,
    depth: u32,
}

impl ValueSeed<'_, '_> {
    fn convert<E: serde::de::Error>(self, value: Value) -> Result<IValue, E> {
        leaf_to_ivalue(value, self.context.refs).map_err(|error| self.context.fail(error))
    }

    fn child(self) -> Self {
        Self {
            depth: self.depth + 1,
            ..self
        }
    }
}

impl<'de> serde::de::DeserializeSeed<'de> for ValueSeed<'_, '_> {
    type Value = IValue;

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<IValue, D::Error> {
        if self.context.limit_depth && self.depth >= MAX_DEPTH {
            return Err(self.context.fail(CborDecodeError::DepthLimitExceeded));
        }
        deserializer.deserialize_any(self)
    }
}

impl<'de> serde::de::Visitor<'de> for ValueSeed<'_, '_> {
    type Value = IValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value with CBOR typed arrays or stream string references")
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<IValue, E> {
        Ok(IValue::NULL)
    }
    fn visit_none<E: serde::de::Error>(self) -> Result<IValue, E> {
        Ok(IValue::NULL)
    }
    fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<IValue, E> {
        Ok(value.into())
    }
    fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<IValue, E> {
        self.convert(value.into())
    }
    fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<IValue, E> {
        self.convert(value.into())
    }
    fn visit_i128<E: serde::de::Error>(self, value: i128) -> Result<IValue, E> {
        self.convert(value.into())
    }
    fn visit_u128<E: serde::de::Error>(self, value: u128) -> Result<IValue, E> {
        self.convert(value.into())
    }
    fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<IValue, E> {
        self.convert(value.into())
    }
    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<IValue, E> {
        Ok(IString::from(value).into())
    }
    fn visit_bytes<E: serde::de::Error>(self, _: &[u8]) -> Result<IValue, E> {
        Err(self.context.fail(CborDecodeError::InvalidValue))
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut access: A) -> Result<IValue, A::Error> {
        let mut array = IArray::with_capacity(access.size_hint().unwrap_or(0).min(1024))
            .map_err(|_| self.context.fail(CborDecodeError::AllocError))?;
        while let Some(value) = access.next_element_seed(self.child())? {
            array
                .push(value)
                .map_err(|_| self.context.fail(CborDecodeError::AllocError))?;
        }
        Ok(array.into())
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut access: A) -> Result<IValue, A::Error> {
        let mut object = IObject::with_capacity(access.size_hint().unwrap_or(0).min(1024))
            .map_err(|_| self.context.fail(CborDecodeError::AllocError))?;
        while let Some(key) = access.next_key_seed(KeySeed {
            context: self.context,
        })? {
            let value = access.next_value_seed(self.child())?;
            object
                .insert(key, value)
                .map_err(|_| self.context.fail(CborDecodeError::AllocError))?;
        }
        Ok(object.into())
    }

    fn visit_enum<A: serde::de::EnumAccess<'de>>(self, access: A) -> Result<IValue, A::Error> {
        use serde::de::VariantAccess;
        // Keep ciborium's tag and bignum normalization; valid tags hold one byte string or ID.
        // Stream definitions hold one text string.
        // The variant name is always "@@TAGGED@@"; skip it without allocating a String.
        let (_, data): (serde::de::IgnoredAny, _) = access.variant()?;
        // The payload is a 2-tuple: element 1 is the tag number, element 2 its content.
        data.tuple_variant(2, TaggedValue { seed: self })
    }
}

/// Reads a stream string tag (6 or 7) directly from the CBOR reader, without a `Value`.
/// Returns `None`, with nothing consumed after the tag, when `tag` is not a stream string tag.
fn stream_string<'c, 'de, A: serde::de::SeqAccess<'de>>(
    context: &DecodeContext<'c>,
    tag: u64,
    access: &mut A,
) -> Result<Option<IString>, A::Error> {
    // Without a stream decoder no tag is a string tag.
    let StringRefs::Stream(cell) = context.refs else {
        return Ok(None);
    };
    let missing = || <A::Error as serde::de::Error>::custom("expected val");
    let result = match tag {
        // Definition: the text is interned and takes the next ID.
        TAG_DEF => access
            .next_element_seed(DefText { cell })?
            .ok_or_else(missing)?,
        // Reference: an ID below the number of definitions, resolved to the shared handle.
        TAG_REF => {
            let id = access.next_element_seed(RefId)?.ok_or_else(missing)?;
            id.and_then(|id| cell.borrow_mut().reference_id(id))
        }
        _ => return Ok(None),
    };
    // A refused definition or reference fails the whole decode with its own error kind.
    result.map(Some).map_err(|error| context.fail(error))
}

struct TaggedValue<'a, 'd> {
    seed: ValueSeed<'a, 'd>,
}

impl<'de> serde::de::Visitor<'de> for TaggedValue<'_, '_> {
    type Value = IValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a valid CBOR item")
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut access: A) -> Result<IValue, A::Error> {
        // Element 1 of the tag pair: the tag number.
        let tag: u64 = access
            .next_element()?
            .ok_or_else(|| <A::Error as serde::de::Error>::custom("expected tag"))?;
        // Tags 6 and 7 are read here, with no intermediate `Value`.
        if let Some(string) = stream_string(self.seed.context, tag, &mut access)? {
            return Ok(string.into());
        }
        // Typed arrays keep ciborium's value path.
        let inner: Value = access
            .next_element()?
            .ok_or_else(|| <A::Error as serde::de::Error>::custom("expected val"))?;
        self.seed.convert(Value::Tag(tag, Box::new(inner)))
    }
}

/// Implements the scalar `Visitor` methods a seed refuses, each as `self.reject()`.
macro_rules! reject_scalars {
    ($($name:ident: $ty:ty),* $(,)?) => {$(
        fn $name<E: serde::de::Error>(self, _: $ty) -> Result<Self::Value, E> {
            self.reject()
        }
    )*};
}

/// Implements the argument-less `Visitor` methods a seed refuses.
macro_rules! reject_empty {
    ($($name:ident),* $(,)?) => {$(
        fn $name<E: serde::de::Error>(self) -> Result<Self::Value, E> {
            self.reject()
        }
    )*};
}

/// Implements the `seq`, `map` and `enum` `Visitor` methods a seed refuses.
macro_rules! reject_compound {
    () => {};
    (seq $($rest:tt)*) => {
        fn visit_seq<A: serde::de::SeqAccess<'de>>(self, _: A) -> Result<Self::Value, A::Error> {
            self.reject()
        }
        reject_compound!($($rest)*);
    };
    (map $($rest:tt)*) => {
        fn visit_map<A: serde::de::MapAccess<'de>>(self, _: A) -> Result<Self::Value, A::Error> {
            self.reject()
        }
        reject_compound!($($rest)*);
    };
    (enum $($rest:tt)*) => {
        fn visit_enum<A: serde::de::EnumAccess<'de>>(self, _: A) -> Result<Self::Value, A::Error> {
            self.reject()
        }
        reject_compound!($($rest)*);
    };
}

/// The text of a definition, borrowed from the reader and interned once.
/// Non-text content yields `Err(InvalidValue)` inside `Ok`, so the caller gets a
/// `CborDecodeError` and not a serde error.
struct DefText<'c> {
    cell: &'c DecoderCell<'c>,
}

impl DefText<'_> {
    fn reject<E>(self) -> Result<Result<IString, CborDecodeError>, E> {
        Ok(Err(CborDecodeError::InvalidValue))
    }
}

impl<'de> serde::de::DeserializeSeed<'de> for DefText<'_> {
    type Value = Result<IString, CborDecodeError>;

    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> serde::de::Visitor<'de> for DefText<'_> {
    type Value = Result<IString, CborDecodeError>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a stream definition text")
    }

    fn visit_str<E: serde::de::Error>(self, text: &str) -> Result<Self::Value, E> {
        Ok(self.cell.borrow_mut().define_str(text))
    }

    reject_scalars!(
        visit_u64: u64,
        visit_i64: i64,
        visit_i128: i128,
        visit_u128: u128,
        visit_f64: f64,
        visit_bool: bool,
        visit_bytes: &[u8],
    );
    reject_empty!(visit_unit, visit_none);
    reject_compound!(seq map enum);
}

/// The ID of a reference. Anything but an unsigned integer is `InvalidStringReference`.
struct RefId;

impl RefId {
    fn reject<E>(self) -> Result<Result<usize, CborDecodeError>, E> {
        Ok(Err(CborDecodeError::InvalidStringReference))
    }
}

impl<'de> serde::de::DeserializeSeed<'de> for RefId {
    type Value = Result<usize, CborDecodeError>;

    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> serde::de::Visitor<'de> for RefId {
    type Value = Result<usize, CborDecodeError>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a stream reference id")
    }

    fn visit_u64<E: serde::de::Error>(self, id: u64) -> Result<Self::Value, E> {
        Ok(usize::try_from(id).map_err(|_| CborDecodeError::InvalidStringReference))
    }

    fn visit_i128<E: serde::de::Error>(self, id: i128) -> Result<Self::Value, E> {
        Ok(usize::try_from(id).map_err(|_| CborDecodeError::InvalidStringReference))
    }
    fn visit_u128<E: serde::de::Error>(self, id: u128) -> Result<Self::Value, E> {
        Ok(usize::try_from(id).map_err(|_| CborDecodeError::InvalidStringReference))
    }

    reject_scalars!(
        visit_i64: i64,
        visit_f64: f64,
        visit_bool: bool,
        visit_str: &str,
        visit_bytes: &[u8],
    );
    reject_empty!(visit_unit, visit_none);
    reject_compound!(seq map enum);
}

/// An object key: text, or a stream string tag. Inline text is interned directly from the reader.
struct KeySeed<'a, 'd> {
    context: &'a DecodeContext<'d>,
}

impl KeySeed<'_, '_> {
    fn reject<E: serde::de::Error>(self) -> Result<IString, E> {
        Err(self.context.fail(CborDecodeError::InvalidValue))
    }
}

impl<'de> serde::de::DeserializeSeed<'de> for KeySeed<'_, '_> {
    type Value = IString;

    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<IString, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> serde::de::Visitor<'de> for KeySeed<'_, '_> {
    type Value = IString;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an object key")
    }

    fn visit_str<E: serde::de::Error>(self, key: &str) -> Result<IString, E> {
        Ok(IString::from(key))
    }

    fn visit_enum<A: serde::de::EnumAccess<'de>>(self, access: A) -> Result<IString, A::Error> {
        use serde::de::VariantAccess;
        // A tagged key arrives as the same ("@@TAGGED@@", (tag, content)) enum as a value:
        // skip the name, then read the 2-tuple through `visit_seq` below.
        let (_, data): (serde::de::IgnoredAny, _) = access.variant()?;
        data.tuple_variant(2, self)
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut access: A) -> Result<IString, A::Error> {
        let tag: u64 = access
            .next_element()?
            .ok_or_else(|| <A::Error as serde::de::Error>::custom("expected tag"))?;
        // Only stream string tags can be keys; any other tag is invalid.
        match stream_string(self.context, tag, &mut access)? {
            Some(string) => Ok(string),
            None => Err(self.context.fail(CborDecodeError::InvalidValue)),
        }
    }

    reject_scalars!(
        visit_u64: u64,
        visit_i64: i64,
        visit_i128: i128,
        visit_u128: u128,
        visit_f64: f64,
        visit_bool: bool,
        visit_bytes: &[u8],
    );
    reject_empty!(visit_unit, visit_none);
    reject_compound!(map);
}

// ── Decode ────────────────────────────────────────────────────────────────────

/// Decodes an [`IValue`] tree from CBOR bytes produced by [`encode`].
pub fn decode(bytes: &[u8]) -> Result<IValue, CborDecodeError> {
    decode_with_depth_limit(bytes, true)
}

/// Decodes an [`IValue`] tree, optionally enforcing the standard nesting limit.
pub fn decode_with_depth_limit(bytes: &[u8], limit_depth: bool) -> Result<IValue, CborDecodeError> {
    decode_one_value(&mut &*bytes, limit_depth, StringRefs::None)
}

/// Decodes the first value of `bytes` and leaves the bytes after it in `bytes`.
fn decode_one_value(
    bytes: &mut &[u8],
    limit_depth: bool,
    refs: StringRefs<'_>,
) -> Result<IValue, CborDecodeError> {
    let recursion_limit = if limit_depth {
        MAX_DEPTH as usize
    } else {
        usize::MAX
    };
    let mut scratch = [0u8; 4096];
    let mut deserializer = ciborium::de::deserializer_from_reader_with_buffer_and_recursion_limit(
        bytes,
        &mut scratch,
        recursion_limit,
    );
    let context = DecodeContext {
        refs,
        limit_depth,
        error: std::cell::Cell::new(None),
    };
    ValueSeed {
        context: &context,
        depth: 0,
    }
    .deserialize(&mut deserializer)
    .map_err(|error| match error {
        ciborium::de::Error::RecursionLimitExceeded => CborDecodeError::DepthLimitExceeded,
        _ => context.error.take().unwrap_or(CborDecodeError::DecodeError),
    })
}

/// Decodes an [`IValue`] tree from bytes produced by [`encode_compressed`].
pub fn decode_compressed(bytes: &[u8]) -> Result<IValue, CborDecodeError> {
    let raw = zstd::decode_all(bytes).map_err(|_| CborDecodeError::DecompressError)?;
    decode(&raw)
}

/// Converts what the visitor leaves to ciborium: numbers (ciborium has already turned bignums into
/// integers) and tags. The nesting limit was checked before the visitor ran.
fn leaf_to_ivalue(val: Value, refs: StringRefs<'_>) -> Result<IValue, CborDecodeError> {
    match val {
        Value::Float(f) => Ok(INumber::try_from(f).map(Into::into).unwrap_or(IValue::NULL)),
        Value::Integer(i) => {
            if let Ok(v) = i64::try_from(i.clone()) {
                Ok(IValue::from(v))
            } else if let Ok(v) = u64::try_from(i) {
                Ok(IValue::from(v))
            } else {
                Err(CborDecodeError::InvalidValue)
            }
        }
        Value::Tag(tag, inner) if refs.is_string_tag(tag) => {
            refs.resolve(tag, *inner).map(Into::into)
        }
        Value::Tag(tag, inner) => decode_typed_array(tag, *inner),
        _ => Err(CborDecodeError::InvalidValue),
    }
}

fn decode_typed_array(tag: u64, inner: Value) -> Result<IValue, CborDecodeError> {
    let bytes = match inner {
        Value::Bytes(b) => b,
        _ => return Err(CborDecodeError::InvalidValue),
    };
    match tag {
        TAG_U8 => decode_le_array::<u8>(&bytes),
        TAG_I8 => decode_le_array::<i8>(&bytes),
        TAG_U16_LE => decode_le_array::<u16>(&bytes),
        TAG_I16_LE => decode_le_array::<i16>(&bytes),
        TAG_F16_LE => decode_le_array::<f16>(&bytes),
        TAG_BF16_LE => decode_le_array::<bf16>(&bytes),
        TAG_U32_LE => decode_le_array::<u32>(&bytes),
        TAG_I32_LE => decode_le_array::<i32>(&bytes),
        TAG_F32_LE => decode_le_array::<f32>(&bytes),
        TAG_U64_LE => decode_le_array::<u64>(&bytes),
        TAG_I64_LE => decode_le_array::<i64>(&bytes),
        TAG_F64_LE => decode_le_array::<f64>(&bytes),
        other => Err(CborDecodeError::UnknownTag(other)),
    }
}

fn decode_le_array<T>(bytes: &[u8]) -> Result<IValue, CborDecodeError>
where
    T: LeBytes,
    IArray: TryFrom<Vec<T>>,
{
    IArray::try_from(T::slice_from_le_bytes(bytes)?)
        .map(Into::into)
        .map_err(|_| CborDecodeError::AllocError)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[test]
    fn plain_encoding_appends_without_changing_prefix() {
        let value = crate::IValue::from("hello");
        let mut out = vec![0, 42];
        super::encode_into(&value, &mut out);
        assert_eq!(out, [0, 42, 0x65, b'h', b'e', b'l', b'l', b'o']);
    }

    use super::*;
    use crate::array::ArraySliceRef;
    use crate::IValueDeserSeed;
    use serde::de::DeserializeSeed;

    #[test]
    fn compatibility_encoder_matches_tree_bytes() {
        let mut cases: Vec<IValue> = vec![
            IValue::NULL,
            true.into(),
            false.into(),
            i64::MIN.into(),
            u64::MAX.into(),
            (-0.0f64).into(),
            1.5f64.into(),
            f64::MIN_POSITIVE.into(),
            f64::MAX.into(),
            IString::from("shared value").into(),
            serde_json::from_str(r#"{"shared field":["shared value",null,42,{},[],"literal"]}"#)
                .unwrap(),
        ];
        macro_rules! typed {
            ($($values:expr),+ $(,)?) => { $(cases.push(IArray::try_from($values).unwrap().into());)+ };
        }
        typed!(
            vec![i8::MIN, i8::MAX],
            vec![0u8, u8::MAX],
            vec![i16::MIN, i16::MAX],
            vec![0u16, u16::MAX],
            vec![i32::MIN, i32::MAX],
            vec![0u32, u32::MAX],
            vec![i64::MIN, i64::MAX],
            vec![0u64, u64::MAX],
            vec![f16::from_f32(1.5), f16::NEG_INFINITY],
            vec![bf16::from_f32(1.5), bf16::NAN],
            vec![1.5f32, f32::INFINITY],
            vec![1.5f64, f64::NAN]
        );
        for value in cases {
            let mut expected = Vec::new();
            ciborium::into_writer(&ivalue_to_cbor(&value), &mut expected).unwrap();
            let actual = encode(&value);
            assert_eq!(actual, expected);
            let decoded = decode(&actual).unwrap();
            assert_eq!(encode(&decoded), expected);
        }
    }

    #[test]
    fn compatibility_decoder_matches_legacy_inputs() {
        let json = |text: &str| serde_json::from_str::<IValue>(text).unwrap();
        let cases: Vec<(Vec<u8>, IValue)> = vec![
            (vec![0x9f, 0xf6, 0xf5, 0xff], json("[null,true]")),
            (vec![0x7f, 0x61, b'a', 0x61, b'b', 0xff], json("\"ab\"")),
            (
                vec![0xbf, 0x61, b'a', 1, 0x61, b'a', 2, 0xff],
                json("{\"a\":2}"),
            ),
            (
                vec![0xd8, TAG_U16_LE as u8, 0x5f, 0x41, 1, 0x41, 0, 0xff],
                IArray::try_from(vec![1u16]).unwrap().into(),
            ),
            (vec![0xc2, 0x41, 0], json("0")),
            (vec![0xc3, 0x41, 0], json("-1")),
            (vec![0xfb, 0x7f, 0xf0, 0, 0, 0, 0, 0, 0], IValue::NULL),
            (vec![0xf6, 0xf5], IValue::NULL),
        ];
        for (bytes, expected) in cases {
            for limit in [true, false] {
                assert_eq!(
                    decode_with_depth_limit(&bytes, limit),
                    Ok(expected.clone()),
                    "{bytes:?}"
                );
            }
        }
        // The tag of the former dictionary references is no string tag now.
        for limit in [true, false] {
            assert_eq!(
                decode_with_depth_limit(&[0xda, 0, 1, 0, 1, 0xc2, 0x41, 0], limit),
                Err(CborDecodeError::InvalidValue)
            );
        }
        // Nesting: the limit rejects 128 levels of arrays, no limit accepts any.
        for (depth, limited_ok) in [(126, true), (127, true), (128, false), (300, false)] {
            let mut bytes = vec![0x81; depth];
            bytes.push(0xf6);
            assert!(decode_with_depth_limit(&bytes, false).is_ok(), "{depth}");
            let limited = decode_with_depth_limit(&bytes, true);
            if limited_ok {
                assert!(limited.is_ok(), "{depth}");
            } else {
                assert_eq!(limited, Err(CborDecodeError::DepthLimitExceeded), "{depth}");
            }
        }
    }

    #[test]
    fn stream_decoder_matches_the_tree_reference() {
        let def = |text: &str| Value::Tag(TAG_DEF, Box::new(Value::Text(text.into())));
        let reference = |id: u64| Value::Tag(TAG_REF, Box::new(Value::Integer(id.into())));
        let wrap =
            |depth: usize, leaf: Value| (0..depth).fold(leaf, |inner, _| Value::Array(vec![inner]));
        let wrap_keys = |depth: usize, key: Value, leaf: Value| {
            (0..depth).fold(leaf, |inner, _| Value::Map(vec![(key.clone(), inner)]))
        };
        let mut cases = vec![
            Value::Array(vec![def("12345678"), reference(0), reference(0)]),
            Value::Map(vec![
                (def("key key key"), def("value value")),
                (reference(0), reference(1)),
            ]),
            Value::Array(vec![reference(0)]),
            Value::Array(vec![def("12345678"), reference(1)]),
            def("1234567"),
            Value::Tag(TAG_DEF, Box::new(Value::Integer(8.into()))),
            Value::Tag(TAG_REF, Box::new(Value::Text("0".into()))),
            // Formerly the tag of dictionary references; no string tag now.
            Value::Tag(0x10001, Box::new(Value::Integer(0.into()))),
            Value::Tag(TAG_U16_LE, Box::new(Value::Bytes(vec![1, 0]))),
            Value::Array(vec![]),
        ];
        for depth in [126, 127, 128, 300] {
            cases.push(wrap(depth, def("deepest leaf")));
            cases.push(wrap(
                depth,
                Value::Array(vec![def("leaf and"), reference(0)]),
            ));
            cases.push(wrap_keys(depth, def("every level key"), Value::Null));
            cases.push(wrap_keys(depth, def("every level key"), reference(0)));
        }
        let json = |text: &str| serde_json::from_str::<IValue>(text).unwrap();
        // What each of the first cases decodes to, and how many strings it defines.
        let fixed: Vec<(Result<IValue, CborDecodeError>, usize)> = vec![
            (Ok(json(r#"["12345678","12345678","12345678"]"#)), 1),
            (Ok(json(r#"{"key key key":"value value"}"#)), 2),
            (Err(CborDecodeError::InvalidStringReference), 0),
            (Err(CborDecodeError::InvalidStringReference), 1),
            (Err(CborDecodeError::InvalidValue), 0),
            (Err(CborDecodeError::InvalidValue), 0),
            (Err(CborDecodeError::InvalidStringReference), 0),
            (Err(CborDecodeError::InvalidValue), 0),
            (Ok(IArray::try_from(vec![1u16]).unwrap().into()), 0),
            (Ok(json("[]")), 0),
        ];
        let depths = [126, 127, 128, 300];
        for (index, case) in cases.iter().enumerate() {
            let mut bytes = Vec::new();
            ciborium::into_writer(case, &mut bytes).unwrap();
            for limit in [true, false] {
                let mut decoder = StreamDecoder::new();
                let actual = decoder.decode(&bytes, limit);
                if let Some((expected, defined)) = fixed.get(index) {
                    assert_eq!(&actual, expected, "{case:?} limit {limit}");
                    assert_eq!(decoder.len(), *defined, "{case:?}");
                    continue;
                }
                // Four cases per depth: two arrays that define one string (the second holds the
                // leaf in one more array), two maps that define one string per level.
                let (depth, kind) = (depths[(index - fixed.len()) / 4], (index - fixed.len()) % 4);
                let levels = depth + usize::from(kind == 1);
                if limit && levels > 127 {
                    assert_eq!(
                        actual,
                        Err(CborDecodeError::DepthLimitExceeded),
                        "{depth} {kind}"
                    );
                } else {
                    assert!(actual.is_ok(), "{depth} {kind} limit {limit}");
                    assert_eq!(
                        decoder.len(),
                        if kind < 2 { 1 } else { depth },
                        "{depth} {kind}"
                    );
                }
            }
        }
    }

    fn round_trip(value: &IValue) -> IValue {
        let bytes = encode(value);
        decode(&bytes).expect("decode should succeed")
    }

    #[test]
    fn test_null() {
        assert_eq!(round_trip(&IValue::NULL), IValue::NULL);
    }

    #[test]
    fn test_bool() {
        let t: IValue = true.into();
        let f: IValue = false.into();
        assert_eq!(round_trip(&t), t);
        assert_eq!(round_trip(&f), f);
    }

    #[test]
    fn test_numbers() {
        let cases: Vec<IValue> = vec![
            0i64.into(),
            42i64.into(),
            (-1i64).into(),
            i64::MAX.into(),
            u64::MAX.into(),
            1.5f64.into(),
            (-3.14f64).into(),
        ];
        for v in &cases {
            assert_eq!(round_trip(v), *v);
        }
    }

    #[test]
    fn test_string() {
        let v: IValue = IString::from("hello world").into();
        assert_eq!(round_trip(&v), v);
    }

    #[test]
    fn test_heterogeneous_array() {
        let mut arr = IArray::new();
        arr.push(IValue::NULL).unwrap();
        arr.push(IValue::from(true)).unwrap();
        arr.push(IValue::from(42i64)).unwrap();
        arr.push(IValue::from(IString::from("hi"))).unwrap();
        let v: IValue = arr.into();
        let result = round_trip(&v);
        let result_arr = result.as_array().unwrap();
        assert!(matches!(
            result_arr.as_slice(),
            ArraySliceRef::Heterogeneous(_)
        ));
        assert_eq!(result_arr.len(), 4);
    }

    #[test]
    fn test_f32_array_preserves_tag() {
        let seed = IValueDeserSeed::new(Some(crate::FPHAConfig::new_with_type(
            crate::FloatType::F32,
        )));
        let json = r#"[1.5, 2.5, 3.5]"#;
        let mut de = serde_json::Deserializer::from_str(json);
        let v = seed.deserialize(&mut de).unwrap();
        assert!(matches!(
            v.as_array().unwrap().as_slice(),
            ArraySliceRef::F32(_)
        ));

        let result = round_trip(&v);
        let arr = result.as_array().unwrap();
        assert!(
            matches!(arr.as_slice(), ArraySliceRef::F32(_)),
            "F32 tag should survive CBOR encode/decode"
        );
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_f16_array_preserves_tag() {
        let seed = IValueDeserSeed::new(Some(crate::FPHAConfig::new_with_type(
            crate::FloatType::F16,
        )));
        let json = r#"[0.5, 1.0, 1.5]"#;
        let mut de = serde_json::Deserializer::from_str(json);
        let v = seed.deserialize(&mut de).unwrap();
        assert!(matches!(
            v.as_array().unwrap().as_slice(),
            ArraySliceRef::F16(_)
        ));

        let result = round_trip(&v);
        let arr = result.as_array().unwrap();
        assert!(
            matches!(arr.as_slice(), ArraySliceRef::F16(_)),
            "F16 tag should survive CBOR encode/decode"
        );
    }

    #[test]
    fn test_bf16_array_preserves_tag() {
        let seed = IValueDeserSeed::new(Some(crate::FPHAConfig::new_with_type(
            crate::FloatType::BF16,
        )));
        let json = r#"[1.0, 2.0, 3.0]"#;
        let mut de = serde_json::Deserializer::from_str(json);
        let v = seed.deserialize(&mut de).unwrap();

        let result = round_trip(&v);
        let arr = result.as_array().unwrap();
        assert!(
            matches!(arr.as_slice(), ArraySliceRef::BF16(_)),
            "BF16 tag should survive CBOR encode/decode"
        );
    }

    #[test]
    fn test_f64_array_preserves_tag() {
        let seed = IValueDeserSeed::new(Some(crate::FPHAConfig::new_with_type(
            crate::FloatType::F64,
        )));
        let json = r#"[1.0, 2.0, 3.0]"#;
        let mut de = serde_json::Deserializer::from_str(json);
        let v = seed.deserialize(&mut de).unwrap();

        let result = round_trip(&v);
        let arr = result.as_array().unwrap();
        assert!(
            matches!(arr.as_slice(), ArraySliceRef::F64(_)),
            "F64 tag should survive CBOR encode/decode"
        );
    }

    #[test]
    fn test_nested_object_with_typed_arrays() {
        let seed = IValueDeserSeed::new(Some(crate::FPHAConfig::new_with_type(
            crate::FloatType::F32,
        )));
        let json = r#"{"a": [1.0, 2.0], "b": "text", "c": [3.0, 4.0]}"#;
        let mut de = serde_json::Deserializer::from_str(json);
        let v = seed.deserialize(&mut de).unwrap();

        let result = round_trip(&v);
        let obj = result.as_object().unwrap();
        let a = obj.get("a").unwrap().as_array().unwrap();
        let c = obj.get("c").unwrap().as_array().unwrap();
        assert!(matches!(a.as_slice(), ArraySliceRef::F32(_)));
        assert!(matches!(c.as_slice(), ArraySliceRef::F32(_)));
        assert_eq!(obj.get("b").unwrap().as_string().unwrap().as_str(), "text");
    }

    #[test]
    fn test_optional_depth_limit() {
        let mut value = IValue::NULL;
        for _ in 0..300 {
            value = IArray::try_from(vec![value]).unwrap().into();
        }

        let bytes = encode(&value);
        assert_eq!(decode(&bytes), Err(CborDecodeError::DepthLimitExceeded));
        assert_eq!(decode_with_depth_limit(&bytes, false).unwrap(), value);
    }

    #[test]
    fn test_compressed_round_trip() {
        let seed = IValueDeserSeed::new(Some(crate::FPHAConfig::new_with_type(
            crate::FloatType::F32,
        )));
        let json = r#"[1.5, 2.5, 3.5, 4.5, 5.5]"#;
        let mut de = serde_json::Deserializer::from_str(json);
        let v = seed.deserialize(&mut de).unwrap();

        let bytes = encode_compressed(&v);
        let result = decode_compressed(&bytes).expect("decode_compressed should succeed");
        assert!(matches!(
            result.as_array().unwrap().as_slice(),
            ArraySliceRef::F32(_)
        ));
    }

    #[test]
    fn test_small_integers_compact() {
        // Small integers should be encoded more compactly in CBOR than custom binary.
        let v: IValue = 42i64.into();
        let cbor_bytes = encode(&v);
        // 42 fits in a single CBOR byte (major type 0, value 24 triggers 1-byte header + 1-byte value)
        // Either way it's much smaller than the custom binary's fixed 9 bytes.
        assert!(
            cbor_bytes.len() < 9,
            "expected CBOR to be smaller than 9-byte fixed encoding"
        );
    }

    #[test]
    fn test_misaligned_typed_array_is_rejected() {
        // Hand-craft valid CBOR: Tag(85, Bytes[13 bytes]).
        // 13 bytes is not divisible by 4 (F32 element size), so decode must
        // return an error rather than silently truncating to 3 elements.
        //
        // CBOR layout:
        //   0xD8 0x55       — tag 85 (RFC 8746 F32-LE), 2-byte form (tag >= 24)
        //   0x4D            — byte string, length 13 (0x40 | 13)
        //   [0u8; 13]       — 13 zero bytes (misaligned: 13 % 4 != 0)
        let mut bytes = vec![0xD8, 0x55, 0x4D];
        bytes.extend_from_slice(&[0u8; 13]);
        assert_eq!(
            decode(&bytes),
            Err(CborDecodeError::CastError),
            "F32 typed array with 13 bytes (not a multiple of 4) must be rejected"
        );
    }

    #[test]
    fn test_misaligned_f16_array_is_rejected() {
        // Tag 84 (F16-LE), byte string of 5 bytes (not divisible by 2).
        // 0xD8 0x54 — tag 84; 0x45 — byte string length 5
        let mut bytes = vec![0xD8, 0x54, 0x45];
        bytes.extend_from_slice(&[0u8; 5]);
        assert_eq!(
            decode(&bytes),
            Err(CborDecodeError::CastError),
            "F16 typed array with 5 bytes (not a multiple of 2) must be rejected"
        );
    }
}
