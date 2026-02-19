use std::fmt;

use bytemuck;
use half::{bf16, f16};

use crate::array::ArraySliceRef;
use crate::{DestructuredRef, IArray, INumber, IObject, IString, IValue};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tag {
    Null = 0x00,
    False = 0x01,
    True = 0x02,
    I64 = 0x03,
    U64 = 0x04,
    F64 = 0x05,
    String = 0x06,
    Object = 0x07,
    ArrayHetero = 0x08,
    ArrayI8 = 0x10,
    ArrayU8 = 0x11,
    ArrayI16 = 0x12,
    ArrayU16 = 0x13,
    ArrayF16 = 0x14,
    ArrayBF16 = 0x15,
    ArrayI32 = 0x16,
    ArrayU32 = 0x17,
    ArrayF32 = 0x18,
    ArrayI64 = 0x19,
    ArrayU64 = 0x1A,
    ArrayF64 = 0x1B,
}

impl TryFrom<u8> for Tag {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0x00 => Ok(Tag::Null),
            0x01 => Ok(Tag::False),
            0x02 => Ok(Tag::True),
            0x03 => Ok(Tag::I64),
            0x04 => Ok(Tag::U64),
            0x05 => Ok(Tag::F64),
            0x06 => Ok(Tag::String),
            0x07 => Ok(Tag::Object),
            0x08 => Ok(Tag::ArrayHetero),
            0x10 => Ok(Tag::ArrayI8),
            0x11 => Ok(Tag::ArrayU8),
            0x12 => Ok(Tag::ArrayI16),
            0x13 => Ok(Tag::ArrayU16),
            0x14 => Ok(Tag::ArrayF16),
            0x15 => Ok(Tag::ArrayBF16),
            0x16 => Ok(Tag::ArrayI32),
            0x17 => Ok(Tag::ArrayU32),
            0x18 => Ok(Tag::ArrayF32),
            0x19 => Ok(Tag::ArrayI64),
            0x1A => Ok(Tag::ArrayU64),
            0x1B => Ok(Tag::ArrayF64),
            other => Err(other),
        }
    }
}

/// Error returned when decoding fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryDecodeError {
    /// The input was too short to decode the next value.
    UnexpectedEof,
    /// An unknown type tag was encountered.
    UnknownTag(u8),
    /// A string was not valid UTF-8.
    InvalidUtf8,
    /// An array allocation failed.
    AllocError,
    /// Nesting depth exceeded the limit.
    DepthLimitExceeded,
    /// Decompression failed (zstd error).
    DecompressError,
    /// Failed to cast slice.
    CastError,
}

const MAX_DEPTH: u32 = 128;

impl fmt::Display for BinaryDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryDecodeError::UnexpectedEof => write!(f, "unexpected end of input"),
            BinaryDecodeError::UnknownTag(t) => write!(f, "unknown type tag: 0x{:02X}", t),
            BinaryDecodeError::InvalidUtf8 => write!(f, "invalid UTF-8 in string"),
            BinaryDecodeError::AllocError => write!(f, "memory allocation failed"),
            BinaryDecodeError::DepthLimitExceeded => write!(f, "nesting depth limit exceeded"),
            BinaryDecodeError::DecompressError => write!(f, "zstd decompression failed"),
            BinaryDecodeError::CastError => write!(f, "failed to cast slice"),
        }
    }
}

/// Encodes an [`IValue`] tree into a compact binary representation that
/// preserves the [`ArrayTag`](crate::array::ArrayTag) of every array.
pub fn encode(value: &IValue) -> Vec<u8> {
    let mut out = Vec::new();
    encode_into(value, &mut out);
    out
}

/// Encodes an [`IValue`] tree and compresses the result with zstd (level 3).
///
/// Use [`decode_compressed`] to decode the output.
pub fn encode_compressed(value: &IValue) -> Vec<u8> {
    let raw = encode(value);
    zstd::bulk::Compressor::default()
        .compress(&raw)
        .expect("zstd compress")
}

/// Decodes an [`IValue`] tree from bytes produced by [`encode_compressed`].
pub fn decode_compressed(bytes: &[u8]) -> Result<IValue, BinaryDecodeError> {
    let raw = zstd::decode_all(bytes).map_err(|_| BinaryDecodeError::DecompressError)?;
    decode(&raw)
}

fn push_tag(tag: Tag, out: &mut Vec<u8>) {
    out.push(tag as u8);
}

fn encode_into(value: &IValue, out: &mut Vec<u8>) {
    match value.destructure_ref() {
        DestructuredRef::Null => push_tag(Tag::Null, out),
        DestructuredRef::Bool(false) => push_tag(Tag::False, out),
        DestructuredRef::Bool(true) => push_tag(Tag::True, out),
        DestructuredRef::Number(n) => encode_number(n, out),
        DestructuredRef::String(s) => encode_string(s, out),
        DestructuredRef::Array(a) => encode_array(a, out),
        DestructuredRef::Object(o) => encode_object(o, out),
    }
}

fn encode_number(n: &INumber, out: &mut Vec<u8>) {
    if n.has_decimal_point() {
        push_tag(Tag::F64, out);
        out.extend_from_slice(&n.to_f64().unwrap().to_le_bytes());
    } else if let Some(v) = n.to_i64() {
        push_tag(Tag::I64, out);
        out.extend_from_slice(&v.to_le_bytes());
    } else {
        push_tag(Tag::U64, out);
        out.extend_from_slice(&n.to_u64().unwrap().to_le_bytes());
    }
}

fn encode_string(s: &IString, out: &mut Vec<u8>) {
    let bytes = s.as_str().as_bytes();
    push_tag(Tag::String, out);
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn encode_array(a: &IArray, out: &mut Vec<u8>) {
    let len = a.len() as u32;
    match a.as_slice() {
        ArraySliceRef::Heterogeneous(s) => {
            push_tag(Tag::ArrayHetero, out);
            out.extend_from_slice(&len.to_le_bytes());
            for v in s {
                encode_into(v, out);
            }
        }
        ArraySliceRef::I8(s) => {
            push_tag(Tag::ArrayI8, out);
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(bytemuck::cast_slice::<i8, u8>(s));
        }
        ArraySliceRef::U8(s) => {
            push_tag(Tag::ArrayU8, out);
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(s);
        }
        ArraySliceRef::I16(s) => encode_typed_array(Tag::ArrayI16, s, out),
        ArraySliceRef::U16(s) => encode_typed_array(Tag::ArrayU16, s, out),
        ArraySliceRef::F16(s) => encode_typed_array(Tag::ArrayF16, s, out),
        ArraySliceRef::BF16(s) => encode_typed_array(Tag::ArrayBF16, s, out),
        ArraySliceRef::I32(s) => encode_typed_array(Tag::ArrayI32, s, out),
        ArraySliceRef::U32(s) => encode_typed_array(Tag::ArrayU32, s, out),
        ArraySliceRef::F32(s) => encode_typed_array(Tag::ArrayF32, s, out),
        ArraySliceRef::I64(s) => encode_typed_array(Tag::ArrayI64, s, out),
        ArraySliceRef::U64(s) => encode_typed_array(Tag::ArrayU64, s, out),
        ArraySliceRef::F64(s) => encode_typed_array(Tag::ArrayF64, s, out),
    }
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

fn encode_typed_array<T: ToLeBytes>(tag: Tag, s: &[T], out: &mut Vec<u8>) {
    push_tag(tag, out);
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    for v in s {
        out.extend_from_slice(v.to_le_bytes_vec().as_ref());
    }
}

fn encode_object(o: &IObject, out: &mut Vec<u8>) {
    push_tag(Tag::Object, out);
    out.extend_from_slice(&(o.len() as u32).to_le_bytes());
    for (k, v) in o {
        let key_bytes = k.as_str().as_bytes();
        out.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(key_bytes);
        encode_into(v, out);
    }
}

/// Decodes an [`IValue`] tree from bytes produced by [`encode`].
///
/// # Errors
///
/// Returns [`BinaryDecodeError`] if the bytes are malformed.
pub fn decode(bytes: &[u8]) -> Result<IValue, BinaryDecodeError> {
    let mut cur = 0usize;
    decode_value(bytes, &mut cur, 0)
}

fn read_u8(bytes: &[u8], cur: &mut usize) -> Result<u8, BinaryDecodeError> {
    bytes
        .get(*cur)
        .copied()
        .map(|b| {
            *cur += 1;
            b
        })
        .ok_or(BinaryDecodeError::UnexpectedEof)
}

fn read_u32(bytes: &[u8], cur: &mut usize) -> Result<u32, BinaryDecodeError> {
    let end = cur.checked_add(4).ok_or(BinaryDecodeError::UnexpectedEof)?;
    let slice = bytes
        .get(*cur..end)
        .ok_or(BinaryDecodeError::UnexpectedEof)?;
    *cur = end;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_i64(bytes: &[u8], cur: &mut usize) -> Result<i64, BinaryDecodeError> {
    let end = cur.checked_add(8).ok_or(BinaryDecodeError::UnexpectedEof)?;
    let slice = bytes
        .get(*cur..end)
        .ok_or(BinaryDecodeError::UnexpectedEof)?;
    *cur = end;
    Ok(i64::from_le_bytes(slice.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], cur: &mut usize) -> Result<u64, BinaryDecodeError> {
    let end = cur.checked_add(8).ok_or(BinaryDecodeError::UnexpectedEof)?;
    let slice = bytes
        .get(*cur..end)
        .ok_or(BinaryDecodeError::UnexpectedEof)?;
    *cur = end;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

fn read_f64(bytes: &[u8], cur: &mut usize) -> Result<f64, BinaryDecodeError> {
    let end = cur.checked_add(8).ok_or(BinaryDecodeError::UnexpectedEof)?;
    let slice = bytes
        .get(*cur..end)
        .ok_or(BinaryDecodeError::UnexpectedEof)?;
    *cur = end;
    Ok(f64::from_le_bytes(slice.try_into().unwrap()))
}

fn read_bytes<'a>(
    bytes: &'a [u8],
    cur: &mut usize,
    n: usize,
) -> Result<&'a [u8], BinaryDecodeError> {
    let end = cur.checked_add(n).ok_or(BinaryDecodeError::UnexpectedEof)?;
    let slice = bytes
        .get(*cur..end)
        .ok_or(BinaryDecodeError::UnexpectedEof)?;
    *cur = end;
    Ok(slice)
}

fn decode_value(bytes: &[u8], cur: &mut usize, depth: u32) -> Result<IValue, BinaryDecodeError> {
    if depth >= MAX_DEPTH {
        return Err(BinaryDecodeError::DepthLimitExceeded);
    }
    let raw_tag = read_u8(bytes, cur)?;
    let tag = Tag::try_from(raw_tag).map_err(BinaryDecodeError::UnknownTag)?;
    match tag {
        Tag::Null => Ok(IValue::NULL),
        Tag::False => Ok(false.into()),
        Tag::True => Ok(true.into()),
        Tag::I64 => Ok(read_i64(bytes, cur)?.into()),
        Tag::U64 => Ok(read_u64(bytes, cur)?.into()),
        Tag::F64 => {
            let v = read_f64(bytes, cur)?;
            Ok(INumber::try_from(v).map(Into::into).unwrap_or(IValue::NULL))
        }
        Tag::String => {
            let len = read_u32(bytes, cur)? as usize;
            let raw = read_bytes(bytes, cur, len)?;
            let s = std::str::from_utf8(raw).map_err(|_| BinaryDecodeError::InvalidUtf8)?;
            Ok(IString::from(s).into())
        }
        Tag::Object => {
            let count = read_u32(bytes, cur)? as usize;
            // Each entry needs at least 5 bytes: 4-byte key-len + 1-byte value tag.
            let hint = count.min((bytes.len() - *cur) / 5);
            let mut obj = IObject::with_capacity(hint);
            for _ in 0..count {
                let key_len = read_u32(bytes, cur)? as usize;
                let key_raw = read_bytes(bytes, cur, key_len)?;
                let key =
                    std::str::from_utf8(key_raw).map_err(|_| BinaryDecodeError::InvalidUtf8)?;
                let val = decode_value(bytes, cur, depth + 1)?;
                obj.insert(key, val);
            }
            Ok(obj.into())
        }
        Tag::ArrayHetero => {
            let count = read_u32(bytes, cur)? as usize;
            // Each element needs at least 1 byte (tag).
            let hint = count.min(bytes.len() - *cur);
            let mut arr = IArray::with_capacity(hint).map_err(|_| BinaryDecodeError::AllocError)?;
            for _ in 0..count {
                let v = decode_value(bytes, cur, depth + 1)?;
                arr.push(v).map_err(|_| BinaryDecodeError::AllocError)?;
            }
            Ok(arr.into())
        }
        Tag::ArrayI8 => {
            let count = read_u32(bytes, cur)? as usize;
            let raw = read_bytes(bytes, cur, count)?;
            let typed: &[i8] = bytemuck::try_cast_slice::<u8, i8>(raw)
                .map_err(|_| BinaryDecodeError::CastError)?;
            IArray::try_from(typed)
                .map(Into::into)
                .map_err(|_| BinaryDecodeError::AllocError)
        }
        Tag::ArrayU8 => {
            let count = read_u32(bytes, cur)? as usize;
            let raw = read_bytes(bytes, cur, count)?;
            IArray::try_from(raw)
                .map(Into::into)
                .map_err(|_| BinaryDecodeError::AllocError)
        }
        Tag::ArrayI16 => decode_primitive_array::<i16>(bytes, cur, 2),
        Tag::ArrayU16 => decode_primitive_array::<u16>(bytes, cur, 2),
        Tag::ArrayF16 => {
            let count = read_u32(bytes, cur)? as usize;
            let byte_len = count
                .checked_mul(2)
                .ok_or(BinaryDecodeError::UnexpectedEof)?;
            let raw = read_bytes(bytes, cur, byte_len)?;
            let vec: Vec<f16> = raw
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes(c.try_into().unwrap()))
                .collect();
            IArray::try_from(vec)
                .map(Into::into)
                .map_err(|_| BinaryDecodeError::AllocError)
        }
        Tag::ArrayBF16 => {
            let count = read_u32(bytes, cur)? as usize;
            let byte_len = count
                .checked_mul(2)
                .ok_or(BinaryDecodeError::UnexpectedEof)?;
            let raw = read_bytes(bytes, cur, byte_len)?;
            let vec: Vec<bf16> = raw
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes(c.try_into().unwrap()))
                .collect();
            IArray::try_from(vec)
                .map(Into::into)
                .map_err(|_| BinaryDecodeError::AllocError)
        }
        Tag::ArrayI32 => decode_primitive_array::<i32>(bytes, cur, 4),
        Tag::ArrayU32 => decode_primitive_array::<u32>(bytes, cur, 4),
        Tag::ArrayF32 => decode_primitive_array::<f32>(bytes, cur, 4),
        Tag::ArrayI64 => decode_primitive_array::<i64>(bytes, cur, 8),
        Tag::ArrayU64 => decode_primitive_array::<u64>(bytes, cur, 8),
        Tag::ArrayF64 => decode_primitive_array::<f64>(bytes, cur, 8),
    }
}

trait FromLeBytes: Copy + Sized + 'static {
    fn from_le_bytes_slice(s: &[u8]) -> Self;
}

macro_rules! impl_from_le_bytes {
    ($ty:ty, $size:expr) => {
        impl FromLeBytes for $ty {
            fn from_le_bytes_slice(s: &[u8]) -> Self {
                Self::from_le_bytes(s.try_into().unwrap())
            }
        }
    };
}

impl_from_le_bytes!(i16, 2);
impl_from_le_bytes!(u16, 2);
impl_from_le_bytes!(i32, 4);
impl_from_le_bytes!(u32, 4);
impl_from_le_bytes!(f32, 4);
impl_from_le_bytes!(i64, 8);
impl_from_le_bytes!(u64, 8);
impl_from_le_bytes!(f64, 8);

fn decode_primitive_array<T>(
    bytes: &[u8],
    cur: &mut usize,
    elem_size: usize,
) -> Result<IValue, BinaryDecodeError>
where
    T: FromLeBytes,
    IArray: TryFrom<Vec<T>>,
{
    let count = read_u32(bytes, cur)? as usize;
    let byte_len = count
        .checked_mul(elem_size)
        .ok_or(BinaryDecodeError::UnexpectedEof)?;
    let raw = read_bytes(bytes, cur, byte_len)?;
    let mut vec: Vec<T> = Vec::with_capacity(count);
    for chunk in raw.chunks_exact(elem_size) {
        vec.push(T::from_le_bytes_slice(chunk));
    }
    IArray::try_from(vec)
        .map(Into::into)
        .map_err(|_| BinaryDecodeError::AllocError)
}

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
        let v: IValue = IValue::NULL;
        assert_eq!(round_trip(&v), v);
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
            "F32 tag should survive encode/decode"
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
            "F16 tag should survive encode/decode"
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
            "BF16 tag should survive encode/decode"
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
            "F64 tag should survive encode/decode"
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
        assert!(
            matches!(a.as_slice(), ArraySliceRef::F32(_)),
            "nested F32 array 'a' should survive"
        );
        assert!(
            matches!(c.as_slice(), ArraySliceRef::F32(_)),
            "nested F32 array 'c' should survive"
        );
        assert_eq!(obj.get("b").unwrap().as_string().unwrap().as_str(), "text");
    }

    #[test]
    fn test_truncated_input_returns_error() {
        let v: IValue = 42i64.into();
        let bytes = encode(&v);
        for len in 0..bytes.len() {
            assert!(
                decode(&bytes[..len]).is_err(),
                "truncated at {len} should fail"
            );
        }
    }

    #[test]
    fn test_unknown_tag_returns_error() {
        let bytes = [0xFF];
        assert_eq!(decode(&bytes), Err(BinaryDecodeError::UnknownTag(0xFF)));
    }

    #[test]
    fn test_object_huge_count_does_not_oom() {
        // TAG_OBJECT with count=0x94940606 (~2.5 billion) followed by no actual data.
        // Must return an error, not OOM.
        let bytes = [0x07, 0x06, 0x06, 0x94, 0x94];
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn test_hetero_array_huge_count_does_not_oom() {
        // TAG_ARRAY_HETERO with count=0xFFFFFFFF followed by no data.
        let bytes = [0x08, 0xFF, 0xFF, 0xFF, 0xFF];
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn test_depth_limit() {
        // Build MAX_DEPTH+1 levels of nested single-element hetero arrays.
        // Each level: ArrayHetero tag (1) + count=1 (4) = 5 bytes, then recurse.
        let mut bytes: Vec<u8> = Vec::new();
        for _ in 0..=super::MAX_DEPTH {
            bytes.push(super::Tag::ArrayHetero as u8);
            bytes.extend_from_slice(&1u32.to_le_bytes());
        }
        bytes.push(super::Tag::Null as u8);
        assert_eq!(decode(&bytes), Err(BinaryDecodeError::DepthLimitExceeded));
    }

    #[test]
    fn test_depth_limit_exact() {
        // MAX_DEPTH-1 array wrappers: the leaf is decoded at depth=MAX_DEPTH-1, which is allowed.
        let mut bytes: Vec<u8> = Vec::new();
        for _ in 0..super::MAX_DEPTH - 1 {
            bytes.push(super::Tag::ArrayHetero as u8);
            bytes.extend_from_slice(&1u32.to_le_bytes());
        }
        bytes.push(super::Tag::Null as u8);
        assert!(decode(&bytes).is_ok());
    }
}
