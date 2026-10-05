use std::convert::TryFrom;
use std::fmt::{self, Formatter};
use std::marker::PhantomData;

use serde::de::{
    DeserializeSeed, EnumAccess, Error as SError, Expected, IntoDeserializer, MapAccess, SeqAccess,
    Unexpected, VariantAccess, Visitor,
};
use serde::{forward_to_deserialize_any, Deserialize, Deserializer};
use serde_json::error::Error;

use crate::{DestructuredRef, FloatType, IArray, INumber, IObject, IString, IValue};

#[derive(Debug, Clone, Copy)]
/// Configuration for floating point homogeneous arrays.
pub struct FPHAConfig {
    /// Floating point type for homogeneous arrays.
    pub fpha_type: FloatType,
}

impl FPHAConfig {
    /// Creates a new [`FPHAConfig`] with the given floating point type.
    pub fn new_with_type(fpha_type: FloatType) -> Self {
        Self { fpha_type }
    }
}

/// Seed for deserializing an [`IValue`].
#[derive(Debug, Default)]
pub struct IValueDeserSeed {
    /// Optional FPHA configuration for homogeneous arrays.
    pub fpha_config: Option<FPHAConfig>,
}

impl IValueDeserSeed {
    /// Creates a new [`IValueDeserSeed`] with the given floating point type enforcment type for homogeneous arrays.
    pub fn new(fpha_config: Option<FPHAConfig>) -> Self {
        IValueDeserSeed { fpha_config }
    }

    /// Deserializes using temporary object buffers, then moves each object's
    /// unique fields into exactly sized storage. Arrays keep normal growth.
    /// The caller must still check for trailing input with the deserializer.
    pub fn deserialize_compact_objects<'de, D>(self, deserializer: D) -> Result<IValue, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut buffers = Vec::new();
        deserializer.deserialize_any(ValueVisitor {
            fpha_config: self.fpha_config,
            buffers: Some(&mut buffers),
        })
    }
}

/// Temporary parser storage; independent of the final object's table threshold.
pub(crate) const OBJECT_BUFFER_INLINE_CAPACITY: usize = 16;

// IndexMap preserves insertion order and replaces duplicate values. Reuse the
// existing hash builder and retain drained maps for later sibling objects.
pub(crate) type ObjectBuffer =
    indexmap::IndexMap<IString, IValue, hashbrown::hash_map::DefaultHashBuilder>;

impl<'de> DeserializeSeed<'de> for IValueDeserSeed {
    type Value = IValue;

    fn deserialize<D>(self, deserializer: D) -> Result<IValue, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Pass hint to a custom visitor
        deserializer.deserialize_any(ValueVisitor::new(self.fpha_config))
    }
}

impl<'de> Deserialize<'de> for IValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(ValueVisitor::new(None))
    }
}

impl<'de> Deserialize<'de> for INumber {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(NumberVisitor)
    }
}

impl<'de> Deserialize<'de> for IString {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(StringVisitor)
    }
}

impl<'de> Deserialize<'de> for IArray {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(ArrayVisitor {
            fpha_config: None,
            buffers: None,
        })
    }
}

impl<'de> Deserialize<'de> for IObject {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(ObjectVisitor {
            fpha_config: None,
            buffers: None,
        })
    }
}

struct ValueVisitor<'a> {
    fpha_config: Option<FPHAConfig>,
    // Pool of drained maps that retain capacity for reuse while parsing objects.
    // Maps are popped when needed and returned after use; indices do not identify
    // JSON objects or nesting levels. None selects ordinary deserialization.
    buffers: Option<&'a mut Vec<ObjectBuffer>>,
}

impl ValueVisitor<'_> {
    fn new(fpha_config: Option<FPHAConfig>) -> Self {
        ValueVisitor {
            fpha_config,
            buffers: None,
        }
    }
}

// Nested visitors share the pool of reusable object buffers.
impl<'de> DeserializeSeed<'de> for ValueVisitor<'_> {
    type Value = IValue;

    fn deserialize<D>(self, deserializer: D) -> Result<IValue, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for ValueVisitor<'_> {
    type Value = IValue;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("any valid JSON value")
    }

    #[inline]
    fn visit_bool<E: SError>(self, value: bool) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_i64<E: SError>(self, value: i64) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_u64<E: SError>(self, value: u64) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_f64<E: SError>(self, value: f64) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_str<E: SError>(self, value: &str) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_string<E: SError>(self, value: String) -> Result<IValue, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_none<E: SError>(self) -> Result<IValue, E> {
        Ok(IValue::NULL)
    }

    #[inline]
    fn visit_some<D>(self, deserializer: D) -> Result<IValue, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.deserialize(deserializer)
    }

    #[inline]
    fn visit_unit<E: SError>(self) -> Result<IValue, E> {
        Ok(IValue::NULL)
    }

    #[inline]
    fn visit_seq<V>(self, visitor: V) -> Result<IValue, V::Error>
    where
        V: SeqAccess<'de>,
    {
        ArrayVisitor {
            fpha_config: self.fpha_config,
            buffers: self.buffers,
        }
        .visit_seq(visitor)
        .map(Into::into)
    }

    fn visit_map<V>(self, visitor: V) -> Result<IValue, V::Error>
    where
        V: MapAccess<'de>,
    {
        ObjectVisitor {
            fpha_config: self.fpha_config,
            buffers: self.buffers,
        }
        .visit_map(visitor)
        .map(Into::into)
    }
}

struct NumberVisitor;

impl<'de> Visitor<'de> for NumberVisitor {
    type Value = INumber;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("JSON number")
    }

    #[inline]
    fn visit_i64<E: SError>(self, value: i64) -> Result<INumber, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_u64<E: SError>(self, value: u64) -> Result<INumber, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_f64<E: SError>(self, value: f64) -> Result<INumber, E> {
        INumber::try_from(value).map_err(|_| E::invalid_value(Unexpected::Float(value), &self))
    }
}

struct StringVisitor;

impl<'de> Visitor<'de> for StringVisitor {
    type Value = IString;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("JSON string")
    }

    #[inline]
    fn visit_str<E: SError>(self, value: &str) -> Result<IString, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_string<E: SError>(self, value: String) -> Result<Self::Value, E> {
        Ok(value.into())
    }

    #[inline]
    fn visit_bytes<E: SError>(self, value: &[u8]) -> Result<Self::Value, E> {
        match std::str::from_utf8(value) {
            Ok(s) => Ok(s.into()),
            Err(_) => Err(SError::invalid_value(Unexpected::Bytes(value), &self)),
        }
    }

    #[inline]
    fn visit_byte_buf<E: SError>(self, value: Vec<u8>) -> Result<Self::Value, E> {
        match String::from_utf8(value) {
            Ok(s) => Ok(s.into()),
            Err(e) => Err(SError::invalid_value(
                Unexpected::Bytes(&e.into_bytes()),
                &self,
            )),
        }
    }
}

struct ArrayVisitor<'a> {
    fpha_config: Option<FPHAConfig>,
    buffers: Option<&'a mut Vec<ObjectBuffer>>,
}

impl<'de> Visitor<'de> for ArrayVisitor<'_> {
    type Value = IArray;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("JSON array")
    }

    #[inline]
    fn visit_seq<V>(mut self, mut visitor: V) -> Result<IArray, V::Error>
    where
        V: SeqAccess<'de>,
    {
        let mut arr = IArray::with_capacity(visitor.size_hint().unwrap_or(0))
            .map_err(|_| SError::custom("Failed to allocate array"))?;
        while let Some(v) = visitor.next_element_seed(ValueVisitor {
            fpha_config: self.fpha_config,
            buffers: self.buffers.as_deref_mut(),
        })? {
            match self.fpha_config {
                Some(FPHAConfig { fpha_type }) => arr.push_with_fp_type(v, fpha_type),
                None => arr.push(v).map_err(Into::into),
            }
            .map_err(|e| SError::custom(e.to_string()))?;
        }
        Ok(arr)
    }
}

struct ObjectVisitor<'a> {
    fpha_config: Option<FPHAConfig>,
    buffers: Option<&'a mut Vec<ObjectBuffer>>,
}

impl<'de> Visitor<'de> for ObjectVisitor<'_> {
    type Value = IObject;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("JSON object")
    }

    fn visit_map<V>(self, mut map_access: V) -> Result<IObject, V::Error>
    where
        V: MapAccess<'de>,
    {
        if let Some(buffer_pool) = self.buffers {
            // Keep up to 16 unique fields inline while parsing. Final objects
            // still use their own eight-field threshold for hash tables.
            let mut inline_entries =
                smallvec::SmallVec::<[(IString, IValue); OBJECT_BUFFER_INLINE_CAPACITY]>::new();
            let mut indexed_entries: Option<ObjectBuffer> = None;
            // Nested values reuse the same pool, but each active object owns its entries.
            while let Some((key, value)) = map_access.next_entry_seed(
                PhantomData::<IString>,
                ValueVisitor {
                    fpha_config: self.fpha_config,
                    buffers: Some(&mut *buffer_pool),
                },
            )? {
                let entries_map = match &mut indexed_entries {
                    Some(entries_map) => entries_map,
                    None => {
                        // Duplicate keys replace their value without consuming another slot.
                        if let Some((_, existing_value)) = inline_entries
                            .iter_mut()
                            .find(|(existing_key, _)| *existing_key == key)
                        {
                            *existing_value = value;
                            continue;
                        }
                        if inline_entries.len() < OBJECT_BUFFER_INLINE_CAPACITY {
                            inline_entries.push((key, value));
                            continue;
                        }
                        // The next unique field exceeds inline storage: move to a pooled map.
                        let mut entries_map = buffer_pool.pop().unwrap_or_default();
                        entries_map
                            .try_reserve(inline_entries.len() + 1)
                            .map_err(SError::custom)?;
                        entries_map.extend(inline_entries.drain(..));
                        indexed_entries.insert(entries_map)
                    }
                };
                // Reserve only for new keys; inserting a duplicate replaces its value.
                if entries_map.len() == entries_map.capacity() && !entries_map.contains_key(&key) {
                    entries_map.try_reserve(1).map_err(SError::custom)?;
                }
                entries_map.insert(key, value);
            }
            // Allocate only for validated, unique fields; move their values.
            if let Some(mut entries_map) = indexed_entries {
                let object = IObject::from_unique_entries(&mut entries_map)
                    .map_err(|_| SError::custom("Failed to allocate object"))?;
                // Construction drains the map; retain its capacity for another object.
                // Pooling is optional: a failed reservation must not reject
                // an object that was already parsed successfully.
                if buffer_pool.try_reserve(1).is_ok() {
                    buffer_pool.push(entries_map);
                }
                return Ok(object);
            }
            return IObject::from_unique_inline_entries(inline_entries)
                .map_err(|_| SError::custom("Failed to allocate object"));
        }

        // Ordinary deserialization inserts directly into a growing object.
        let mut object = IObject::with_capacity(map_access.size_hint().unwrap_or(0))
            .map_err(|_| SError::custom("Failed to allocate object"))?;
        while let Some((key, value)) = map_access.next_entry_seed(
            PhantomData::<IString>,
            ValueVisitor {
                fpha_config: self.fpha_config,
                buffers: None,
            },
        )? {
            object
                .insert(key, value)
                .map_err(|e| SError::custom(e.to_string()))?;
        }
        Ok(object)
    }
}

macro_rules! deserialize_number {
    ($method:ident) => {
        fn $method<V>(self, visitor: V) -> Result<V::Value, Error>
        where
            V: Visitor<'de>,
        {
            if let Some(v) = self.as_number() {
                v.deserialize_any(visitor)
            } else {
                Err(self.invalid_type(&visitor))
            }
        }
    };
}

impl<'de> Deserializer<'de> for &'de IValue {
    type Error = Error;

    #[inline]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.destructure_ref() {
            DestructuredRef::Null => visitor.visit_unit(),
            DestructuredRef::Bool(v) => visitor.visit_bool(v),
            DestructuredRef::Number(v) => v.deserialize_any(visitor),
            DestructuredRef::String(v) => v.deserialize_any(visitor),
            DestructuredRef::Array(v) => v.deserialize_any(visitor),
            DestructuredRef::Object(v) => v.deserialize_any(visitor),
        }
    }

    deserialize_number!(deserialize_i8);
    deserialize_number!(deserialize_i16);
    deserialize_number!(deserialize_i32);
    deserialize_number!(deserialize_i64);
    deserialize_number!(deserialize_u8);
    deserialize_number!(deserialize_u16);
    deserialize_number!(deserialize_u32);
    deserialize_number!(deserialize_u64);
    deserialize_number!(deserialize_f32);
    deserialize_number!(deserialize_f64);

    #[inline]
    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if self.is_null() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    #[inline]
    fn deserialize_enum<V>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.destructure_ref() {
            DestructuredRef::String(v) => v.deserialize_enum(name, variants, visitor),
            DestructuredRef::Object(v) => v.deserialize_enum(name, variants, visitor),
            other => Err(SError::invalid_type(other.unexpected(), &"string or map")),
        }
    }

    #[inline]
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if let Some(v) = self.to_bool() {
            visitor.visit_bool(v)
        } else {
            Err(self.invalid_type(&visitor))
        }
    }

    fn deserialize_char<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if let Some(v) = self.as_string() {
            v.deserialize_str(visitor)
        } else {
            Err(self.invalid_type(&visitor))
        }
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_bytes<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.destructure_ref() {
            DestructuredRef::String(v) => v.deserialize_bytes(visitor),
            DestructuredRef::Array(v) => v.deserialize_bytes(visitor),
            other => Err(other.invalid_type(&visitor)),
        }
    }

    fn deserialize_byte_buf<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_bytes(visitor)
    }

    fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if self.is_null() {
            visitor.visit_unit()
        } else {
            Err(self.invalid_type(&visitor))
        }
    }

    fn deserialize_unit_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_unit(visitor)
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if let Some(v) = self.as_array() {
            v.deserialize_seq(visitor)
        } else {
            Err(self.invalid_type(&visitor))
        }
    }

    fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if let Some(v) = self.as_object() {
            v.deserialize_map(visitor)
        } else {
            Err(self.invalid_type(&visitor))
        }
    }

    fn deserialize_struct<V>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.destructure_ref() {
            DestructuredRef::Array(v) => v.deserialize_struct(name, fields, visitor),
            DestructuredRef::Object(v) => v.deserialize_struct(name, fields, visitor),
            other => Err(other.invalid_type(&visitor)),
        }
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_unit()
    }
}

impl<'de> Deserializer<'de> for &'de INumber {
    type Error = Error;

    #[inline]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if self.has_decimal_point() {
            visitor.visit_f64(self.to_f64().unwrap())
        } else if let Some(v) = self.to_i64() {
            visitor.visit_i64(v)
        } else {
            visitor.visit_u64(self.to_u64().unwrap())
        }
    }

    #[inline]
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}

impl<'de> Deserializer<'de> for &'de IString {
    type Error = Error;

    #[inline]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_borrowed_str(self.as_str())
    }

    fn deserialize_enum<V>(
        self,
        _name: &str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_enum(EnumDeserializer {
            variant: self,
            value: None,
        })
    }

    #[inline]
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct seq tuple
        tuple_struct map struct identifier ignored_any
    }
}

impl<'de> Deserializer<'de> for &'de IArray {
    type Error = Error;

    #[inline]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        use crate::array::ArraySliceRef;
        let len = self.len() as usize;

        macro_rules! deserialize_typed_array {
            ($variant:ident, $slice:expr) => {{
                let mut deserializer = ArrayAccess {
                    iter: Iter::$variant($slice.iter()),
                };
                let seq = visitor.visit_seq(&mut deserializer)?;
                let remaining = deserializer.remaining_len();
                if remaining == 0 {
                    Ok(seq)
                } else {
                    Err(SError::invalid_length(len, &"fewer elements in array"))
                }
            }};
        }

        match self.as_slice() {
            ArraySliceRef::Heterogeneous(slice) => deserialize_typed_array!(Heterogeneous, slice),
            ArraySliceRef::I8(slice) => deserialize_typed_array!(I8, slice),
            ArraySliceRef::U8(slice) => deserialize_typed_array!(U8, slice),
            ArraySliceRef::I16(slice) => deserialize_typed_array!(I16, slice),
            ArraySliceRef::U16(slice) => deserialize_typed_array!(U16, slice),
            ArraySliceRef::F16(slice) => deserialize_typed_array!(F16, slice),
            ArraySliceRef::BF16(slice) => deserialize_typed_array!(BF16, slice),
            ArraySliceRef::I32(slice) => deserialize_typed_array!(I32, slice),
            ArraySliceRef::U32(slice) => deserialize_typed_array!(U32, slice),
            ArraySliceRef::F32(slice) => deserialize_typed_array!(F32, slice),
            ArraySliceRef::I64(slice) => deserialize_typed_array!(I64, slice),
            ArraySliceRef::U64(slice) => deserialize_typed_array!(U64, slice),
            ArraySliceRef::F64(slice) => deserialize_typed_array!(F64, slice),
        }
    }

    #[inline]
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}

impl<'de> Deserializer<'de> for &'de IObject {
    type Error = Error;

    #[inline]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        let len = self.len() as usize;
        let mut deserializer = ObjectAccess::new(self);
        let seq = visitor.visit_map(&mut deserializer)?;
        let remaining = deserializer.iter.len();
        if remaining == 0 {
            Ok(seq)
        } else {
            Err(SError::invalid_length(len, &"fewer elements in object"))
        }
    }

    #[inline]
    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        let mut iter = self.iter();
        let (variant, value) = iter
            .next()
            .ok_or_else(|| SError::invalid_value(Unexpected::Map, &"object with a single key"))?;
        // enums are encoded in json as maps with a single key:value pair
        if iter.next().is_some() {
            return Err(SError::invalid_value(
                Unexpected::Map,
                &"object with a single key",
            ));
        }
        visitor.visit_enum(EnumDeserializer {
            variant,
            value: Some(value),
        })
    }

    #[inline]
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct seq tuple
        tuple_struct map struct identifier ignored_any
    }
}

trait MaybeUnexpected<'de>: Sized {
    fn invalid_type<E>(self, exp: &dyn Expected) -> E
    where
        E: SError,
    {
        SError::invalid_type(self.unexpected(), exp)
    }

    fn unexpected(self) -> Unexpected<'de>;
}

impl<'de> MaybeUnexpected<'de> for &'de IValue {
    fn unexpected(self) -> Unexpected<'de> {
        self.destructure_ref().unexpected()
    }
}

impl<'de> MaybeUnexpected<'de> for DestructuredRef<'de> {
    fn unexpected(self) -> Unexpected<'de> {
        match self {
            Self::Null => Unexpected::Unit,
            Self::Bool(b) => Unexpected::Bool(b),
            Self::Number(v) => v.unexpected(),
            Self::String(v) => v.unexpected(),
            Self::Array(v) => v.unexpected(),
            Self::Object(v) => v.unexpected(),
        }
    }
}

impl<'de> MaybeUnexpected<'de> for &'de INumber {
    fn unexpected(self) -> Unexpected<'de> {
        if self.has_decimal_point() {
            Unexpected::Float(self.to_f64().unwrap())
        } else if let Some(v) = self.to_i64() {
            Unexpected::Signed(v)
        } else {
            Unexpected::Unsigned(self.to_u64().unwrap())
        }
    }
}

impl<'de> MaybeUnexpected<'de> for &'de IString {
    fn unexpected(self) -> Unexpected<'de> {
        Unexpected::Str(self.as_str())
    }
}

impl<'de> MaybeUnexpected<'de> for &'de IArray {
    fn unexpected(self) -> Unexpected<'de> {
        Unexpected::Seq
    }
}

impl<'de> MaybeUnexpected<'de> for &'de IObject {
    fn unexpected(self) -> Unexpected<'de> {
        Unexpected::Map
    }
}

struct EnumDeserializer<'de> {
    variant: &'de IString,
    value: Option<&'de IValue>,
}

impl<'de> EnumAccess<'de> for EnumDeserializer<'de> {
    type Error = Error;
    type Variant = VariantDeserializer<'de>;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self::Variant), Error>
    where
        V: DeserializeSeed<'de>,
    {
        let variant = self.variant.into_deserializer();
        let visitor = VariantDeserializer { value: self.value };
        seed.deserialize(variant).map(|v| (v, visitor))
    }
}

impl<'de> IntoDeserializer<'de, Error> for &'de IString {
    type Deserializer = Self;

    fn into_deserializer(self) -> Self::Deserializer {
        self
    }
}

struct VariantDeserializer<'de> {
    value: Option<&'de IValue>,
}

impl<'de> VariantAccess<'de> for VariantDeserializer<'de> {
    type Error = Error;

    fn unit_variant(self) -> Result<(), Error> {
        if let Some(value) = self.value {
            Deserialize::deserialize(value)
        } else {
            Ok(())
        }
    }

    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value, Error>
    where
        T: DeserializeSeed<'de>,
    {
        if let Some(value) = self.value {
            seed.deserialize(value)
        } else {
            Err(SError::invalid_type(
                Unexpected::UnitVariant,
                &"newtype variant",
            ))
        }
    }

    fn tuple_variant<V>(self, _len: usize, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.value.map(IValue::destructure_ref) {
            Some(DestructuredRef::Array(v)) => v.deserialize_any(visitor),
            Some(other) => Err(SError::invalid_type(other.unexpected(), &"tuple variant")),
            None => Err(SError::invalid_type(
                Unexpected::UnitVariant,
                &"tuple variant",
            )),
        }
    }

    fn struct_variant<V>(
        self,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.value.map(IValue::destructure_ref) {
            Some(DestructuredRef::Object(v)) => v.deserialize_any(visitor),
            Some(other) => Err(SError::invalid_type(other.unexpected(), &"struct variant")),
            None => Err(SError::invalid_type(
                Unexpected::UnitVariant,
                &"struct variant",
            )),
        }
    }
}

struct ArrayAccess<'de> {
    iter: Iter<'de>,
}

enum Iter<'de> {
    Heterogeneous(std::slice::Iter<'de, IValue>),
    I8(std::slice::Iter<'de, i8>),
    U8(std::slice::Iter<'de, u8>),
    I16(std::slice::Iter<'de, i16>),
    U16(std::slice::Iter<'de, u16>),
    F16(std::slice::Iter<'de, half::f16>),
    BF16(std::slice::Iter<'de, half::bf16>),
    I32(std::slice::Iter<'de, i32>),
    U32(std::slice::Iter<'de, u32>),
    F32(std::slice::Iter<'de, f32>),
    I64(std::slice::Iter<'de, i64>),
    U64(std::slice::Iter<'de, u64>),
    F64(std::slice::Iter<'de, f64>),
}

impl<'de> ArrayAccess<'de> {
    fn remaining_len(&self) -> usize {
        match &self.iter {
            Iter::Heterogeneous(it) => it.len(),
            Iter::I8(it) => it.len(),
            Iter::U8(it) => it.len(),
            Iter::I16(it) => it.len(),
            Iter::U16(it) => it.len(),
            Iter::F16(it) => it.len(),
            Iter::BF16(it) => it.len(),
            Iter::I32(it) => it.len(),
            Iter::U32(it) => it.len(),
            Iter::F32(it) => it.len(),
            Iter::I64(it) => it.len(),
            Iter::U64(it) => it.len(),
            Iter::F64(it) => it.len(),
        }
    }
}

impl<'de> SeqAccess<'de> for ArrayAccess<'de> {
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Error>
    where
        T: DeserializeSeed<'de>,
    {
        use serde::de::IntoDeserializer as _;
        match &mut self.iter {
            Iter::Heterogeneous(it) => match it.next() {
                Some(v) => seed.deserialize(v).map(Some),
                None => Ok(None),
            },
            Iter::I8(it) => match it.next() {
                Some(v) => seed.deserialize((*v).into_deserializer()).map(Some),
                None => Ok(None),
            },
            Iter::U8(it) => match it.next() {
                Some(v) => seed.deserialize((*v).into_deserializer()).map(Some),
                None => Ok(None),
            },
            Iter::I16(it) => match it.next() {
                Some(v) => seed.deserialize((*v).into_deserializer()).map(Some),
                None => Ok(None),
            },
            Iter::U16(it) => match it.next() {
                Some(v) => seed.deserialize((*v).into_deserializer()).map(Some),
                None => Ok(None),
            },
            // For f16/bf16, feed as f32 so downstream types (f16/f32/f64) can convert
            Iter::F16(it) => match it.next() {
                Some(v) => {
                    let f: f32 = v.to_f32();
                    seed.deserialize(f.into_deserializer()).map(Some)
                }
                None => Ok(None),
            },
            Iter::BF16(it) => match it.next() {
                Some(v) => {
                    let f: f32 = v.to_f32();
                    seed.deserialize(f.into_deserializer()).map(Some)
                }
                None => Ok(None),
            },
            Iter::I32(it) => match it.next() {
                Some(v) => seed.deserialize((*v).into_deserializer()).map(Some),
                None => Ok(None),
            },
            Iter::U32(it) => match it.next() {
                Some(v) => seed.deserialize((*v).into_deserializer()).map(Some),
                None => Ok(None),
            },
            Iter::F32(it) => match it.next() {
                Some(v) => seed.deserialize((*v).into_deserializer()).map(Some),
                None => Ok(None),
            },
            Iter::I64(it) => match it.next() {
                Some(v) => seed.deserialize((*v).into_deserializer()).map(Some),
                None => Ok(None),
            },
            Iter::U64(it) => match it.next() {
                Some(v) => seed.deserialize((*v).into_deserializer()).map(Some),
                None => Ok(None),
            },
            Iter::F64(it) => match it.next() {
                Some(v) => seed.deserialize((*v).into_deserializer()).map(Some),
                None => Ok(None),
            },
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining_len())
    }
}

struct ObjectAccess<'de> {
    iter: <&'de IObject as IntoIterator>::IntoIter,
    value: Option<&'de IValue>,
}

impl<'de> ObjectAccess<'de> {
    fn new(obj: &'de IObject) -> Self {
        ObjectAccess {
            iter: obj.into_iter(),
            value: None,
        }
    }
}

impl<'de> MapAccess<'de> for ObjectAccess<'de> {
    type Error = Error;

    fn next_key_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Error>
    where
        T: DeserializeSeed<'de>,
    {
        if let Some((key, value)) = self.iter.next() {
            self.value = Some(value);
            seed.deserialize(key).map(Some)
        } else {
            Ok(None)
        }
    }

    fn next_value_seed<T>(&mut self, seed: T) -> Result<T::Value, Error>
    where
        T: DeserializeSeed<'de>,
    {
        if let Some(value) = self.value.take() {
            seed.deserialize(value)
        } else {
            Err(SError::custom("value is missing"))
        }
    }

    fn size_hint(&self) -> Option<usize> {
        match self.iter.size_hint() {
            (lower, Some(upper)) if lower == upper => Some(upper),
            _ => None,
        }
    }
}

/// Converts an [`IValue`] to an arbitrary type using that type's [`serde::Deserialize`]
/// implementation.
///
/// # Errors
///
/// Will return `Error` if `value` fails to deserialize.
pub fn from_value<'de, T>(value: &'de IValue) -> Result<T, Error>
where
    T: Deserialize<'de>,
{
    T::deserialize(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::ArraySliceRef;
    use serde::de::DeserializeSeed;

    #[test]
    #[cfg(not(miri))]
    fn compact_objects_do_not_reserve_for_unvalidated_or_duplicate_fields() {
        for (input, valid) in [
            (format!("{{{}", ":".repeat(100_000)), false),
            (r#"{"a":{"x":1},"b":[{"y":2},"#.to_owned(), false),
            (
                format!("{{{}\"same\":2}}", "\"same\":1,".repeat(100_000)),
                true,
            ),
        ] {
            // Build the input outside the measured scope. Repetition must not
            // translate into a proportional temporary or final object allocation.
            let allocations = mockalloc::record_allocs(|| {
                let mut de = serde_json::Deserializer::from_str(&input);
                let result = IValueDeserSeed::new(None).deserialize_compact_objects(&mut de);
                assert_eq!(result.is_ok(), valid);
                if let Ok(value) = result {
                    assert_eq!(value.as_object().unwrap().capacity(), 1);
                    de.end().unwrap();
                }
            });
            assert!(allocations.peak_mem() < 64 * 1024);
            allocations.result().unwrap();
        }
    }

    #[test]
    fn test_deserialize_with_f64_fp() {
        let json = r#"[1.5, 2.5, 3.5]"#;
        let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(FloatType::F64)));
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let value = seed.deserialize(&mut deserializer).unwrap();

        let arr = value.as_array().unwrap();
        assert!(matches!(arr.as_slice(), ArraySliceRef::F64(_)));
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_deserialize_with_f32_fp() {
        let json = r#"[1.5, 2.5, 3.5]"#;
        let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(FloatType::F32)));
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let value = seed.deserialize(&mut deserializer).unwrap();

        let arr = value.as_array().unwrap();
        assert!(matches!(arr.as_slice(), ArraySliceRef::F32(_)));
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_deserialize_with_f16_fp() {
        let json = r#"[0.5, 1.0, 1.5]"#;
        let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(FloatType::F16)));
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let value = seed.deserialize(&mut deserializer).unwrap();

        let arr = value.as_array().unwrap();
        assert!(matches!(arr.as_slice(), ArraySliceRef::F16(_)));
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_deserialize_with_bf16_fp() {
        let json = r#"[0.5, 1.0, 2.0]"#;
        let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(FloatType::BF16)));
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let value = seed.deserialize(&mut deserializer).unwrap();

        let arr = value.as_array().unwrap();
        assert!(matches!(arr.as_slice(), ArraySliceRef::BF16(_)));
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_deserialize_mixed_array_with_fp() {
        let json = r#"[1, "string", 3.5]"#;
        let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(FloatType::F32)));
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let value = seed.deserialize(&mut deserializer).unwrap();

        let arr = value.as_array().unwrap();
        assert!(matches!(arr.as_slice(), ArraySliceRef::Heterogeneous(_)));
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_deserialize_integer_array_with_fp() {
        let json = r#"[1, 2, 3]"#;
        let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(FloatType::F32)));
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let value = seed.deserialize(&mut deserializer).unwrap();

        let arr = value.as_array().unwrap();
        assert!(matches!(arr.as_slice(), ArraySliceRef::F32(_)));
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_deserialize_f16_value_overflow_rejected() {
        let json = r#"[0.5, 100000.0, 1.5]"#;
        let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(FloatType::F16)));
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let _error = seed.deserialize(&mut deserializer).unwrap_err();
    }

    #[test]
    fn test_deserialize_bf16_value_overflow_rejected() {
        let json = r#"[1e39, 2e39]"#;
        let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(FloatType::BF16)));
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let _error = seed.deserialize(&mut deserializer).unwrap_err();
    }

    #[test]
    fn test_deserialize_f32_value_overflow_rejected() {
        let json = r#"[1e39, 2e39]"#;
        let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(FloatType::F32)));
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let _error = seed.deserialize(&mut deserializer).unwrap_err();
    }

    #[test]
    fn test_fpha_outer_array_of_objects_succeeds() {
        // The classic embedding use-case: outer array holds objects, not numbers.
        // Before the fix, push_with_fp_type would error on the object element.
        let json = r#"[{"embedding": [1.0, 2.0]}, {"embedding": [3.0, 4.0]}]"#;
        let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(FloatType::F16)));
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let value = seed.deserialize(&mut deserializer).unwrap();

        let arr = value.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert!(matches!(arr.as_slice(), ArraySliceRef::Heterogeneous(_)));

        // Inner arrays should still be typed f16
        assert!(matches!(
            arr[0]
                .as_object()
                .unwrap()
                .get("embedding")
                .unwrap()
                .as_array()
                .unwrap()
                .as_slice(),
            ArraySliceRef::F16(_)
        ));
    }

    #[test]
    fn test_fpha_outer_array_of_nested_arrays_succeeds() {
        // Outer array holds inner float arrays; outer must become heterogeneous.
        let json = r#"[[1.0, 2.0], [3.0, 4.0]]"#;
        let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(FloatType::F16)));
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let value = seed.deserialize(&mut deserializer).unwrap();

        let arr = value.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert!(matches!(arr.as_slice(), ArraySliceRef::Heterogeneous(_)));
        // Inner arrays should still be typed f16
        assert!(matches!(
            arr[0].as_array().unwrap().as_slice(),
            ArraySliceRef::F16(_)
        ));
    }

    #[test]
    fn test_ser_deser_roundtrip_preserves_type() {
        let json = r#"[0.2, 1.0, 1.2]"#;

        for fp_type in [FloatType::F16, FloatType::BF16, FloatType::F32] {
            let seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(fp_type)));
            let mut de = serde_json::Deserializer::from_str(json);
            let original = seed.deserialize(&mut de).unwrap();

            let serialized = serde_json::to_string(&original).unwrap();

            let reload_seed = IValueDeserSeed::new(Some(FPHAConfig::new_with_type(fp_type)));
            let mut de = serde_json::Deserializer::from_str(&serialized);
            let roundtripped = reload_seed.deserialize(&mut de).unwrap();

            let arr = roundtripped.as_array().unwrap();
            assert_eq!(arr.len(), 3);
            let roundtrip_tag = arr.as_slice().type_tag();
            assert_eq!(
                roundtrip_tag,
                fp_type.into(),
                "roundtrip should preserve {fp_type}"
            );
        }
    }
}
