use std::fmt;

use half::{bf16, f16};

use crate::array::ArraySliceRef;
use crate::{DestructuredRef, IArray, INumber, IObject, IString, IValue};

const TAG_NULL: u8 = 0x00;
const TAG_FALSE: u8 = 0x01;
const TAG_TRUE: u8 = 0x02;
const TAG_I64: u8 = 0x03;
const TAG_U64: u8 = 0x04;
const TAG_F64: u8 = 0x05;
const TAG_STRING: u8 = 0x06;
const TAG_OBJECT: u8 = 0x07;
const TAG_ARRAY_HETERO: u8 = 0x08;
const TAG_ARRAY_I8: u8 = 0x10;
const TAG_ARRAY_U8: u8 = 0x11;
const TAG_ARRAY_I16: u8 = 0x12;
const TAG_ARRAY_U16: u8 = 0x13;
const TAG_ARRAY_F16: u8 = 0x14;
const TAG_ARRAY_BF16: u8 = 0x15;
const TAG_ARRAY_I32: u8 = 0x16;
const TAG_ARRAY_U32: u8 = 0x17;
const TAG_ARRAY_F32: u8 = 0x18;
const TAG_ARRAY_I64: u8 = 0x19;
const TAG_ARRAY_U64: u8 = 0x1A;
const TAG_ARRAY_F64: u8 = 0x1B;

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

fn encode_into(value: &IValue, out: &mut Vec<u8>) {
    match value.destructure_ref() {
        DestructuredRef::Null => out.push(TAG_NULL),
        DestructuredRef::Bool(false) => out.push(TAG_FALSE),
        DestructuredRef::Bool(true) => out.push(TAG_TRUE),
        DestructuredRef::Number(n) => encode_number(n, out),
        DestructuredRef::String(s) => encode_string(s, out),
        DestructuredRef::Array(a) => encode_array(a, out),
        DestructuredRef::Object(o) => encode_object(o, out),
    }
}

fn encode_number(n: &INumber, out: &mut Vec<u8>) {
    if n.has_decimal_point() {
        out.push(TAG_F64);
        out.extend_from_slice(&n.to_f64().unwrap().to_le_bytes());
    } else if let Some(v) = n.to_i64() {
        out.push(TAG_I64);
        out.extend_from_slice(&v.to_le_bytes());
    } else {
        out.push(TAG_U64);
        out.extend_from_slice(&n.to_u64().unwrap().to_le_bytes());
    }
}

fn encode_string(s: &IString, out: &mut Vec<u8>) {
    let bytes = s.as_str().as_bytes();
    out.push(TAG_STRING);
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn encode_array(a: &IArray, out: &mut Vec<u8>) {
    let len = a.len() as u32;
    match a.as_slice() {
        ArraySliceRef::Heterogeneous(s) => {
            out.push(TAG_ARRAY_HETERO);
            out.extend_from_slice(&len.to_le_bytes());
            for v in s {
                encode_into(v, out);
            }
        }
        ArraySliceRef::I8(s) => {
            out.push(TAG_ARRAY_I8);
            out.extend_from_slice(&len.to_le_bytes());
            let bytes = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()) };
            out.extend_from_slice(bytes);
        }
        ArraySliceRef::U8(s) => {
            out.push(TAG_ARRAY_U8);
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(s);
        }
        ArraySliceRef::I16(s) => encode_typed_array(TAG_ARRAY_I16, s, out),
        ArraySliceRef::U16(s) => encode_typed_array(TAG_ARRAY_U16, s, out),
        ArraySliceRef::F16(s) => encode_typed_array(TAG_ARRAY_F16, s, out),
        ArraySliceRef::BF16(s) => encode_typed_array(TAG_ARRAY_BF16, s, out),
        ArraySliceRef::I32(s) => encode_typed_array(TAG_ARRAY_I32, s, out),
        ArraySliceRef::U32(s) => encode_typed_array(TAG_ARRAY_U32, s, out),
        ArraySliceRef::F32(s) => encode_typed_array(TAG_ARRAY_F32, s, out),
        ArraySliceRef::I64(s) => encode_typed_array(TAG_ARRAY_I64, s, out),
        ArraySliceRef::U64(s) => encode_typed_array(TAG_ARRAY_U64, s, out),
        ArraySliceRef::F64(s) => encode_typed_array(TAG_ARRAY_F64, s, out),
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

fn encode_typed_array<T: ToLeBytes>(tag: u8, s: &[T], out: &mut Vec<u8>) {
    out.push(tag);
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    for v in s {
        out.extend_from_slice(v.to_le_bytes_vec().as_ref());
    }
}

fn encode_object(o: &IObject, out: &mut Vec<u8>) {
    out.push(TAG_OBJECT);
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
    let slice = bytes.get(*cur..end).ok_or(BinaryDecodeError::UnexpectedEof)?;
    *cur = end;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_i64(bytes: &[u8], cur: &mut usize) -> Result<i64, BinaryDecodeError> {
    let end = cur.checked_add(8).ok_or(BinaryDecodeError::UnexpectedEof)?;
    let slice = bytes.get(*cur..end).ok_or(BinaryDecodeError::UnexpectedEof)?;
    *cur = end;
    Ok(i64::from_le_bytes(slice.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], cur: &mut usize) -> Result<u64, BinaryDecodeError> {
    let end = cur.checked_add(8).ok_or(BinaryDecodeError::UnexpectedEof)?;
    let slice = bytes.get(*cur..end).ok_or(BinaryDecodeError::UnexpectedEof)?;
    *cur = end;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

fn read_f64(bytes: &[u8], cur: &mut usize) -> Result<f64, BinaryDecodeError> {
    let end = cur.checked_add(8).ok_or(BinaryDecodeError::UnexpectedEof)?;
    let slice = bytes.get(*cur..end).ok_or(BinaryDecodeError::UnexpectedEof)?;
    *cur = end;
    Ok(f64::from_le_bytes(slice.try_into().unwrap()))
}

fn read_bytes<'a>(bytes: &'a [u8], cur: &mut usize, n: usize) -> Result<&'a [u8], BinaryDecodeError> {
    let end = cur.checked_add(n).ok_or(BinaryDecodeError::UnexpectedEof)?;
    let slice = bytes.get(*cur..end).ok_or(BinaryDecodeError::UnexpectedEof)?;
    *cur = end;
    Ok(slice)
}

fn decode_value(bytes: &[u8], cur: &mut usize, depth: u32) -> Result<IValue, BinaryDecodeError> {
    if depth >= MAX_DEPTH {
        return Err(BinaryDecodeError::DepthLimitExceeded);
    }
    match read_u8(bytes, cur)? {
        TAG_NULL => Ok(IValue::NULL),
        TAG_FALSE => Ok(false.into()),
        TAG_TRUE => Ok(true.into()),
        TAG_I64 => Ok(read_i64(bytes, cur)?.into()),
        TAG_U64 => Ok(read_u64(bytes, cur)?.into()),
        TAG_F64 => {
            let v = read_f64(bytes, cur)?;
            Ok(INumber::try_from(v)
                .map(Into::into)
                .unwrap_or(IValue::NULL))
        }
        TAG_STRING => {
            let len = read_u32(bytes, cur)? as usize;
            let raw = read_bytes(bytes, cur, len)?;
            let s = std::str::from_utf8(raw).map_err(|_| BinaryDecodeError::InvalidUtf8)?;
            Ok(IString::from(s).into())
        }
        TAG_OBJECT => {
            let count = read_u32(bytes, cur)? as usize;
            // Each entry needs at least 5 bytes: 4-byte key-len + 1-byte value tag.
            let hint = count.min((bytes.len() - *cur) / 5);
            let mut obj = IObject::with_capacity(hint);
            for _ in 0..count {
                let key_len = read_u32(bytes, cur)? as usize;
                let key_raw = read_bytes(bytes, cur, key_len)?;
                let key = std::str::from_utf8(key_raw).map_err(|_| BinaryDecodeError::InvalidUtf8)?;
                let val = decode_value(bytes, cur, depth + 1)?;
                obj.insert(key, val);
            }
            Ok(obj.into())
        }
        TAG_ARRAY_HETERO => {
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
        TAG_ARRAY_I8 => {
            let count = read_u32(bytes, cur)? as usize;
            let raw = read_bytes(bytes, cur, count)?;
            let typed: &[i8] = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, count) };
            IArray::try_from(typed).map(Into::into).map_err(|_| BinaryDecodeError::AllocError)
        }
        TAG_ARRAY_U8 => {
            let count = read_u32(bytes, cur)? as usize;
            let raw = read_bytes(bytes, cur, count)?;
            IArray::try_from(raw).map(Into::into).map_err(|_| BinaryDecodeError::AllocError)
        }
        TAG_ARRAY_I16 => decode_primitive_array::<i16>(bytes, cur, 2),
        TAG_ARRAY_U16 => decode_primitive_array::<u16>(bytes, cur, 2),
        TAG_ARRAY_F16 => {
            let count = read_u32(bytes, cur)? as usize;
            let byte_len = count.checked_mul(2).ok_or(BinaryDecodeError::UnexpectedEof)?;
            let raw = read_bytes(bytes, cur, byte_len)?;
            let vec: Vec<f16> = raw
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes(c.try_into().unwrap()))
                .collect();
            IArray::try_from(vec).map(Into::into).map_err(|_| BinaryDecodeError::AllocError)
        }
        TAG_ARRAY_BF16 => {
            let count = read_u32(bytes, cur)? as usize;
            let byte_len = count.checked_mul(2).ok_or(BinaryDecodeError::UnexpectedEof)?;
            let raw = read_bytes(bytes, cur, byte_len)?;
            let vec: Vec<bf16> = raw
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes(c.try_into().unwrap()))
                .collect();
            IArray::try_from(vec).map(Into::into).map_err(|_| BinaryDecodeError::AllocError)
        }
        TAG_ARRAY_I32 => decode_primitive_array::<i32>(bytes, cur, 4),
        TAG_ARRAY_U32 => decode_primitive_array::<u32>(bytes, cur, 4),
        TAG_ARRAY_F32 => decode_primitive_array::<f32>(bytes, cur, 4),
        TAG_ARRAY_I64 => decode_primitive_array::<i64>(bytes, cur, 8),
        TAG_ARRAY_U64 => decode_primitive_array::<u64>(bytes, cur, 8),
        TAG_ARRAY_F64 => decode_primitive_array::<f64>(bytes, cur, 8),
        tag => Err(BinaryDecodeError::UnknownTag(tag)),
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

fn decode_primitive_array<T>(bytes: &[u8], cur: &mut usize, elem_size: usize) -> Result<IValue, BinaryDecodeError>
where
    T: FromLeBytes,
    IArray: TryFrom<Vec<T>>,
{
    let count = read_u32(bytes, cur)? as usize;
    let byte_len = count.checked_mul(elem_size).ok_or(BinaryDecodeError::UnexpectedEof)?;
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
        assert!(matches!(result_arr.as_slice(), ArraySliceRef::Heterogeneous(_)));
        assert_eq!(result_arr.len(), 4);
    }

    #[test]
    fn test_f32_array_preserves_tag() {
        let seed = IValueDeserSeed::new(Some(crate::FPHAConfig::new_with_type(crate::FloatType::F32)));
        let json = r#"[1.5, 2.5, 3.5]"#;
        let mut de = serde_json::Deserializer::from_str(json);
        let v = seed.deserialize(&mut de).unwrap();
        assert!(matches!(v.as_array().unwrap().as_slice(), ArraySliceRef::F32(_)));

        let result = round_trip(&v);
        let arr = result.as_array().unwrap();
        assert!(matches!(arr.as_slice(), ArraySliceRef::F32(_)), "F32 tag should survive encode/decode");
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_f16_array_preserves_tag() {
        let seed = IValueDeserSeed::new(Some(crate::FPHAConfig::new_with_type(crate::FloatType::F16)));
        let json = r#"[0.5, 1.0, 1.5]"#;
        let mut de = serde_json::Deserializer::from_str(json);
        let v = seed.deserialize(&mut de).unwrap();
        assert!(matches!(v.as_array().unwrap().as_slice(), ArraySliceRef::F16(_)));

        let result = round_trip(&v);
        let arr = result.as_array().unwrap();
        assert!(matches!(arr.as_slice(), ArraySliceRef::F16(_)), "F16 tag should survive encode/decode");
    }

    #[test]
    fn test_bf16_array_preserves_tag() {
        let seed = IValueDeserSeed::new(Some(crate::FPHAConfig::new_with_type(crate::FloatType::BF16)));
        let json = r#"[1.0, 2.0, 3.0]"#;
        let mut de = serde_json::Deserializer::from_str(json);
        let v = seed.deserialize(&mut de).unwrap();

        let result = round_trip(&v);
        let arr = result.as_array().unwrap();
        assert!(matches!(arr.as_slice(), ArraySliceRef::BF16(_)), "BF16 tag should survive encode/decode");
    }

    #[test]
    fn test_f64_array_preserves_tag() {
        let seed = IValueDeserSeed::new(Some(crate::FPHAConfig::new_with_type(crate::FloatType::F64)));
        let json = r#"[1.0, 2.0, 3.0]"#;
        let mut de = serde_json::Deserializer::from_str(json);
        let v = seed.deserialize(&mut de).unwrap();

        let result = round_trip(&v);
        let arr = result.as_array().unwrap();
        assert!(matches!(arr.as_slice(), ArraySliceRef::F64(_)), "F64 tag should survive encode/decode");
    }

    #[test]
    fn test_nested_object_with_typed_arrays() {
        let seed = IValueDeserSeed::new(Some(crate::FPHAConfig::new_with_type(crate::FloatType::F32)));
        let json = r#"{"a": [1.0, 2.0], "b": "text", "c": [3.0, 4.0]}"#;
        let mut de = serde_json::Deserializer::from_str(json);
        let v = seed.deserialize(&mut de).unwrap();

        let result = round_trip(&v);
        let obj = result.as_object().unwrap();
        let a = obj.get("a").unwrap().as_array().unwrap();
        let c = obj.get("c").unwrap().as_array().unwrap();
        assert!(matches!(a.as_slice(), ArraySliceRef::F32(_)), "nested F32 array 'a' should survive");
        assert!(matches!(c.as_slice(), ArraySliceRef::F32(_)), "nested F32 array 'c' should survive");
        assert_eq!(obj.get("b").unwrap().as_string().unwrap().as_str(), "text");
    }

    #[test]
    fn test_truncated_input_returns_error() {
        let v: IValue = 42i64.into();
        let bytes = encode(&v);
        for len in 0..bytes.len() {
            assert!(decode(&bytes[..len]).is_err(), "truncated at {len} should fail");
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
        // Each level: TAG_ARRAY_HETERO (1) + count=1 (4) = 5 bytes, then recurse.
        let mut bytes: Vec<u8> = Vec::new();
        for _ in 0..=super::MAX_DEPTH {
            bytes.push(TAG_ARRAY_HETERO);
            bytes.extend_from_slice(&1u32.to_le_bytes());
        }
        bytes.push(TAG_NULL);
        assert_eq!(decode(&bytes), Err(BinaryDecodeError::DepthLimitExceeded));
    }

    #[test]
    fn test_depth_limit_exact() {
        // MAX_DEPTH-1 array wrappers: the leaf is decoded at depth=MAX_DEPTH-1, which is allowed.
        let mut bytes: Vec<u8> = Vec::new();
        for _ in 0..super::MAX_DEPTH - 1 {
            bytes.push(TAG_ARRAY_HETERO);
            bytes.extend_from_slice(&1u32.to_le_bytes());
        }
        bytes.push(TAG_NULL);
        assert!(decode(&bytes).is_ok());
    }
}
