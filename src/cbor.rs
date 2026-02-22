//! CBOR encode/decode for [`IValue`], preserving typed array tags via RFC 8746.
//!
//! Typed homogeneous arrays are encoded as `Tag(rfc8746_tag, Bytes(raw_le_bytes))`.
//! BF16 arrays use private tag `0x10000` (no standard RFC 8746 equivalent).
//!
//! Use [`encode`] / [`decode`] for raw CBOR, or [`encode_compressed`] /
//! [`decode_compressed`] for zstd-compressed CBOR.

use std::fmt;

use ciborium::value::{Integer, Value};
use half::{bf16, f16};

use crate::array::ArraySliceRef;
use crate::{DestructuredRef, IArray, INumber, IObject, IString, IValue};

// RFC 8746 typed array tags (little-endian variants where applicable).
// Bit layout within the tag byte: 0b010_f_s_e_ll
//   f=0 integer, f=1 float | s=0 unsigned, s=1 signed | e=1 little-endian | ll = width
const TAG_U8: u64 = 64; // uint8 (endianness irrelevant)
const TAG_I8: u64 = 72; // sint8 (endianness irrelevant)
const TAG_U16_LE: u64 = 69;
const TAG_I16_LE: u64 = 77;
const TAG_F16_LE: u64 = 84; // IEEE 754 binary16 LE
const TAG_U32_LE: u64 = 70;
const TAG_I32_LE: u64 = 78;
const TAG_F32_LE: u64 = 85; // IEEE 754 binary32 LE
const TAG_U64_LE: u64 = 71;
const TAG_I64_LE: u64 = 79;
const TAG_F64_LE: u64 = 86; // IEEE 754 binary64 LE

/// Private CBOR tag for BF16 arrays (no RFC 8746 standard tag exists for BF16).
const TAG_BF16_LE: u64 = 0x10000;

const MAX_DEPTH: u32 = 128;

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
        }
    }
}

impl std::error::Error for CborDecodeError {}

// ── Encode ────────────────────────────────────────────────────────────────────

/// Encodes an [`IValue`] tree into CBOR bytes, preserving typed array tags via
/// RFC 8746.
pub fn encode(value: &IValue) -> Vec<u8> {
    let cbor = ivalue_to_cbor(value);
    let mut out = Vec::new();
    ciborium::into_writer(&cbor, &mut out).expect("write to Vec never fails");
    out
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

fn ivalue_to_cbor(value: &IValue) -> Value {
    match value.destructure_ref() {
        DestructuredRef::Null => Value::Null,
        DestructuredRef::Bool(b) => Value::Bool(b),
        DestructuredRef::Number(n) => number_to_cbor(n),
        DestructuredRef::String(s) => Value::Text(s.as_str().to_owned()),
        DestructuredRef::Array(a) => array_to_cbor(a),
        DestructuredRef::Object(o) => object_to_cbor(o),
    }
}

fn number_to_cbor(n: &INumber) -> Value {
    if n.has_decimal_point() {
        Value::Float(n.to_f64().unwrap())
    } else if let Some(i) = n.to_i64() {
        Value::Integer(Integer::from(i))
    } else {
        Value::Integer(Integer::from(n.to_u64().unwrap()))
    }
}

fn array_to_cbor(a: &IArray) -> Value {
    match a.as_slice() {
        ArraySliceRef::Heterogeneous(s) => Value::Array(s.iter().map(ivalue_to_cbor).collect()),
        ArraySliceRef::I8(s) => typed_le_tag(TAG_I8, s),
        ArraySliceRef::U8(s) => Value::Tag(TAG_U8, Box::new(Value::Bytes(s.to_vec()))),
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

fn object_to_cbor(o: &IObject) -> Value {
    Value::Map(
        o.iter()
            .map(|(k, v)| (Value::Text(k.as_str().to_owned()), ivalue_to_cbor(v)))
            .collect(),
    )
}

trait ToLeBytes {
    fn to_le_bytes_vec(&self) -> impl AsRef<[u8]>;
}

macro_rules! impl_to_le_bytes {
    ($ty:ty) => {
        impl ToLeBytes for $ty {
            fn to_le_bytes_vec(&self) -> impl AsRef<[u8]> {
                self.to_le_bytes()
            }
        }
    };
}

impl_to_le_bytes!(i8);
impl_to_le_bytes!(i16);
impl_to_le_bytes!(u16);
impl_to_le_bytes!(f16);
impl_to_le_bytes!(bf16);
impl_to_le_bytes!(i32);
impl_to_le_bytes!(u32);
impl_to_le_bytes!(f32);
impl_to_le_bytes!(i64);
impl_to_le_bytes!(u64);
impl_to_le_bytes!(f64);

fn typed_le_tag<T: ToLeBytes>(tag: u64, s: &[T]) -> Value {
    let mut bytes = Vec::new();
    for v in s {
        bytes.extend_from_slice(v.to_le_bytes_vec().as_ref());
    }
    Value::Tag(tag, Box::new(Value::Bytes(bytes)))
}

// ── Decode ────────────────────────────────────────────────────────────────────

/// Decodes an [`IValue`] tree from CBOR bytes produced by [`encode`].
pub fn decode(bytes: &[u8]) -> Result<IValue, CborDecodeError> {
    let cbor: Value = ciborium::from_reader(bytes).map_err(|_| CborDecodeError::DecodeError)?;
    cbor_to_ivalue(cbor, 0)
}

/// Decodes an [`IValue`] tree from bytes produced by [`encode_compressed`].
pub fn decode_compressed(bytes: &[u8]) -> Result<IValue, CborDecodeError> {
    let raw = zstd::decode_all(bytes).map_err(|_| CborDecodeError::DecompressError)?;
    decode(&raw)
}

fn cbor_to_ivalue(val: Value, depth: u32) -> Result<IValue, CborDecodeError> {
    if depth >= MAX_DEPTH {
        return Err(CborDecodeError::DepthLimitExceeded);
    }
    match val {
        Value::Null => Ok(IValue::NULL),
        Value::Bool(b) => Ok(b.into()),
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
        Value::Text(s) => Ok(IString::from(s.as_str()).into()),
        Value::Array(arr) => {
            let hint = arr.len().min(1024);
            let mut out = IArray::with_capacity(hint).map_err(|_| CborDecodeError::AllocError)?;
            for v in arr {
                let iv = cbor_to_ivalue(v, depth + 1)?;
                out.push(iv).map_err(|_| CborDecodeError::AllocError)?;
            }
            Ok(out.into())
        }
        Value::Map(entries) => {
            let mut obj = IObject::with_capacity(entries.len());
            for (k, v) in entries {
                let key = match k {
                    Value::Text(s) => s,
                    _ => return Err(CborDecodeError::InvalidValue),
                };
                let val = cbor_to_ivalue(v, depth + 1)?;
                obj.insert(&key, val);
            }
            Ok(obj.into())
        }
        Value::Tag(tag, inner) => decode_typed_array(tag, *inner),
        Value::Bytes(_) => Err(CborDecodeError::InvalidValue),
        _ => Err(CborDecodeError::InvalidValue),
    }
}

fn decode_typed_array(tag: u64, inner: Value) -> Result<IValue, CborDecodeError> {
    let bytes = match inner {
        Value::Bytes(b) => b,
        _ => return Err(CborDecodeError::InvalidValue),
    };
    match tag {
        TAG_U8 => IArray::try_from(bytes.as_slice())
            .map(Into::into)
            .map_err(|_| CborDecodeError::AllocError),
        TAG_I8 => decode_le_array::<i8>(&bytes, 1),
        TAG_U16_LE => decode_le_array::<u16>(&bytes, 2),
        TAG_I16_LE => decode_le_array::<i16>(&bytes, 2),
        TAG_F16_LE => decode_le_array::<f16>(&bytes, 2),
        TAG_BF16_LE => decode_le_array::<bf16>(&bytes, 2),
        TAG_U32_LE => decode_le_array::<u32>(&bytes, 4),
        TAG_I32_LE => decode_le_array::<i32>(&bytes, 4),
        TAG_F32_LE => decode_le_array::<f32>(&bytes, 4),
        TAG_U64_LE => decode_le_array::<u64>(&bytes, 8),
        TAG_I64_LE => decode_le_array::<i64>(&bytes, 8),
        TAG_F64_LE => decode_le_array::<f64>(&bytes, 8),
        other => Err(CborDecodeError::UnknownTag(other)),
    }
}

trait FromLeBytes: Copy + Sized + 'static {
    fn from_le_bytes_slice(s: &[u8]) -> Self;
}

macro_rules! impl_from_le_bytes {
    ($ty:ty) => {
        impl FromLeBytes for $ty {
            fn from_le_bytes_slice(s: &[u8]) -> Self {
                Self::from_le_bytes(s.try_into().unwrap())
            }
        }
    };
}

impl_from_le_bytes!(i8);
impl_from_le_bytes!(i16);
impl_from_le_bytes!(u16);
impl_from_le_bytes!(f16);
impl_from_le_bytes!(bf16);
impl_from_le_bytes!(i32);
impl_from_le_bytes!(u32);
impl_from_le_bytes!(f32);
impl_from_le_bytes!(i64);
impl_from_le_bytes!(u64);
impl_from_le_bytes!(f64);

fn decode_le_array<T>(bytes: &[u8], elem_size: usize) -> Result<IValue, CborDecodeError>
where
    T: FromLeBytes,
    IArray: TryFrom<Vec<T>>,
{
    let mut vec: Vec<T> = Vec::with_capacity(bytes.len() / elem_size);
    for chunk in bytes.chunks_exact(elem_size) {
        vec.push(T::from_le_bytes_slice(chunk));
    }
    IArray::try_from(vec)
        .map(Into::into)
        .map_err(|_| CborDecodeError::AllocError)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::ArraySliceRef;
    use crate::IValueDeserSeed;
    use serde::de::DeserializeSeed;

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
}
