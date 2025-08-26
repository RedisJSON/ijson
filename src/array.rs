//! Functionality relating to the JSON array type

use std::alloc::{alloc, dealloc, realloc, Layout, LayoutError};
use std::cmp::{self, Ordering, PartialOrd};
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::iter::FromIterator;
use std::ops::{Index, IndexMut};
use std::slice::SliceIndex;

use crate::thin::{ThinMut, ThinMutExt, ThinRef, ThinRefExt};
use crate::{Defrag, DefragAllocator};

use super::value::{IValue, TypeTag};

/// Tag indicating the type of elements stored in a typed array
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArrayTag {
    /// Array contains heterogeneous IValue objects
    Heterogeneous = 0,
    /// Array contains i8 values
    I8 = 1,
    /// Array contains u8 values
    U8 = 2,
    /// Array contains i16 values
    I16 = 3,
    /// Array contains u16 values
    U16 = 4,
    /// Array contains i32 values
    I32 = 5,
    /// Array contains u32 values
    U32 = 6,
    /// Array contains f32 values
    F32 = 7,
    /// Array contains i64 values
    I64 = 8,
    /// Array contains u64 values
    U64 = 9,
    /// Array contains f64 values
    F64 = 10,
}

impl Default for ArrayTag {
    fn default() -> Self {
        Self::Heterogeneous
    }
}

impl ArrayTag {
    fn from_type<T>() -> Self {
        use ArrayTag::*;
        match std::any::type_name::<T>() {
            "i8" => I8,
            "u8" => U8,
            "i16" => I16,
            "u16" => U16,
            "i32" => I32,
            "u32" => U32,
            "f32" => F32,
            "i64" => I64,
            "u64" => U64,
            "f64" => F64,
            _ => Heterogeneous,
        }
    }
}

/// Enum representing different types of array slices that can be returned from typed arrays
#[derive(Clone, PartialEq)]
pub enum ArraySliceRef<'a> {
    /// Heterogeneous array containing IValue objects
    Heterogeneous(&'a [IValue]),
    /// Typed array of i8 values
    I8(&'a [i8]),
    /// Typed array of u8 values
    U8(&'a [u8]),
    /// Typed array of i16 values
    I16(&'a [i16]),
    /// Typed array of u16 values
    U16(&'a [u16]),
    /// Typed array of i32 values
    I32(&'a [i32]),
    /// Typed array of u32 values
    U32(&'a [u32]),
    /// Typed array of f32 values
    F32(&'a [f32]),
    /// Typed array of i64 values
    I64(&'a [i64]),
    /// Typed array of u64 values
    U64(&'a [u64]),
    /// Typed array of f64 values
    F64(&'a [f64]),
}

impl<'a> ArraySliceRef<'a> {
    /// Returns the length of the slice regardless of type
    pub fn len(&self) -> usize {
        use ArraySliceRef::*;
        match self {
            Heterogeneous(slice) => slice.len(),
            I8(slice) => slice.len(),
            U8(slice) => slice.len(),
            I16(slice) => slice.len(),
            U16(slice) => slice.len(),
            I32(slice) => slice.len(),
            U32(slice) => slice.len(),
            F32(slice) => slice.len(),
            I64(slice) => slice.len(),
            U64(slice) => slice.len(),
            F64(slice) => slice.len(),
        }
    }

    /// Returns true if the slice is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the type tag of the array slice
    pub fn type_tag(&self) -> ArrayTag {
        use ArrayTag::*;
        match self {
            Self::Heterogeneous(_) => Heterogeneous,
            Self::I8(_) => I8,
            Self::U8(_) => U8,
            Self::I16(_) => I16,
            Self::U16(_) => U16,
            Self::I32(_) => I32,
            Self::U32(_) => U32,
            Self::F32(_) => F32,
            Self::I64(_) => I64,
            Self::U64(_) => U64,
            Self::F64(_) => F64,
        }
    }

    /// Returns true if this is a heterogeneous array slice
    pub fn is_heterogeneous(&self) -> bool {
        matches!(self, Self::Heterogeneous(_))
    }

    /// Returns true if this is a typed array slice
    pub fn is_typed(&self) -> bool {
        !self.is_heterogeneous()
    }
}

impl PartialOrd for ArraySliceRef<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use ArraySliceRef::*;
        match (self, other) {
            (Heterogeneous(a), Heterogeneous(b)) => a.partial_cmp(b),
            (I8(a), I8(b)) => a.partial_cmp(b),
            (U8(a), U8(b)) => a.partial_cmp(b),
            (I16(a), I16(b)) => a.partial_cmp(b),
            (U16(a), U16(b)) => a.partial_cmp(b),
            (I32(a), I32(b)) => a.partial_cmp(b),
            (U32(a), U32(b)) => a.partial_cmp(b),
            (F32(a), F32(b)) => a.partial_cmp(b),
            (I64(a), I64(b)) => a.partial_cmp(b),
            (U64(a), U64(b)) => a.partial_cmp(b),
            (F64(a), F64(b)) => a.partial_cmp(b),
            _ => None, // Different types are not comparable
        }
    }
}

macro_rules! from_impl {
    ($(($ty:ty, $variant:ident)),*) => {
        $(impl<'a> From<&'a [$ty]> for ArraySliceRef<'a> {
            fn from(slice: &'a [$ty]) -> Self {
                ArraySliceRef::$variant(slice)
            }
        })*
    };
}

from_impl!(
    (i8, I8),
    (u8, U8),
    (i16, I16),
    (u16, U16),
    (i32, I32),
    (u32, U32),
    (f32, F32),
    (i64, I64),
    (u64, U64),
    (f64, F64),
    (IValue, Heterogeneous)
);

impl Debug for ArraySliceRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use ArraySliceRef::*;
        match self {
            Heterogeneous(slice) => Debug::fmt(slice, f),
            I8(slice) => Debug::fmt(slice, f),
            U8(slice) => Debug::fmt(slice, f),
            I16(slice) => Debug::fmt(slice, f),
            U16(slice) => Debug::fmt(slice, f),
            I32(slice) => Debug::fmt(slice, f),
            U32(slice) => Debug::fmt(slice, f),
            F32(slice) => Debug::fmt(slice, f),
            I64(slice) => Debug::fmt(slice, f),
            U64(slice) => Debug::fmt(slice, f),
            F64(slice) => Debug::fmt(slice, f),
        }
    }
}

/// Mutable slice of the array contents
#[derive(Debug)]
pub enum ArraySliceMut<'a> {
    /// Heterogeneous array containing mutable IValue objects
    Heterogeneous(&'a mut [IValue]),
    /// Typed array of mutable i8 values
    I8(&'a mut [i8]),
    /// Typed array of mutable u8 values
    U8(&'a mut [u8]),
    /// Typed array of mutable i16 values
    I16(&'a mut [i16]),
    /// Typed array of mutable u16 values
    U16(&'a mut [u16]),
    /// Typed array of mutable i32 values
    I32(&'a mut [i32]),
    /// Typed array of mutable u32 values
    U32(&'a mut [u32]),
    /// Typed array of mutable f32 values
    F32(&'a mut [f32]),
    /// Typed array of mutable i64 values
    I64(&'a mut [i64]),
    /// Typed array of mutable u64 values
    U64(&'a mut [u64]),
    /// Typed array of mutable f64 values
    F64(&'a mut [f64]),
}

#[repr(C)]
#[repr(align(8))]
struct Header {
    /// Packed field:
    /// bits 0-29: length,
    /// bits 30-59: capacity,
    /// bits 60-63: type tag
    packed: u64,
}

impl Header {
    const LEN_MASK: u64 = (1u64 << 30) - 1;
    const LEN_SHIFT: u64 = 0;
    const CAP_MASK: u64 = (1u64 << 30) - 1;
    const CAP_SHIFT: u64 = 30;
    const TAG_MASK: u64 = 0xF;
    const TAG_SHIFT: u64 = 60;

    const fn new(len: usize, cap: usize, tag: ArrayTag) -> Self {
        assert!(len <= Self::LEN_MASK as usize, "Length exceeds 30-bit limit");
        assert!(cap <= Self::CAP_MASK as usize, "Capacity exceeds 30-bit limit");

        let packed = ((len as u64) & Self::LEN_MASK) << Self::LEN_SHIFT
            | ((cap as u64) & Self::CAP_MASK) << Self::CAP_SHIFT
            | ((tag as u64) & Self::TAG_MASK) << Self::TAG_SHIFT;

        Self { packed }
    }

    fn len(&self) -> usize {
        ((self.packed >> Self::LEN_SHIFT) & Self::LEN_MASK) as usize
    }

    fn cap(&self) -> usize {
        ((self.packed >> Self::CAP_SHIFT) & Self::CAP_MASK) as usize
    }

    fn type_tag(&self) -> ArrayTag {
        let tag_value = ((self.packed >> Self::TAG_SHIFT) & Self::TAG_MASK) as u8;
        // Safety: We only store valid ArrayTag values
        unsafe { std::mem::transmute(tag_value) }
    }

    fn set_len(&mut self, len: usize) {
        assert!(len <= Self::LEN_MASK as usize, "Length exceeds 30-bit limit");
        self.packed = (self.packed & !(Self::LEN_MASK << Self::LEN_SHIFT))
            | (((len as u64) & Self::LEN_MASK) << Self::LEN_SHIFT);
    }

    fn set_cap(&mut self, cap: usize) {
        assert!(cap <= Self::CAP_MASK as usize, "Capacity exceeds 30-bit limit");
        self.packed = (self.packed & !(Self::CAP_MASK << Self::CAP_SHIFT))
            | (((cap as u64) & Self::CAP_MASK) << Self::CAP_SHIFT);
    }

}

trait HeaderRef<'a>: ThinRefExt<'a, Header> {
    fn array_ptr(&self) -> *const IValue {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr().add(1).cast::<IValue>() }
    }
    fn raw_array_ptr(&self) -> *const u8 {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr().add(1).cast::<u8>() }
    }
    fn items_slice(&self) -> ArraySliceRef<'a> {
        use ArraySliceRef::*;
        // Safety: Header `len` must be accurate
        match self.type_tag() {
            ArrayTag::I8 => I8(unsafe { std::slice::from_raw_parts(self.raw_array_ptr().cast::<i8>(), self.len()) }),
            ArrayTag::U8 => U8(unsafe { std::slice::from_raw_parts(self.raw_array_ptr().cast::<u8>(), self.len()) }),
            ArrayTag::I16 => I16(unsafe { std::slice::from_raw_parts(self.raw_array_ptr().cast::<i16>(), self.len()) }),
            ArrayTag::U16 => U16(unsafe { std::slice::from_raw_parts(self.raw_array_ptr().cast::<u16>(), self.len()) }),
            ArrayTag::I32 => I32(unsafe { std::slice::from_raw_parts(self.raw_array_ptr().cast::<i32>(), self.len()) }),
            ArrayTag::U32 => U32(unsafe { std::slice::from_raw_parts(self.raw_array_ptr().cast::<u32>(), self.len()) }),
            ArrayTag::I64 => I64(unsafe { std::slice::from_raw_parts(self.raw_array_ptr().cast::<i64>(), self.len()) }),
            ArrayTag::U64 => U64(unsafe { std::slice::from_raw_parts(self.raw_array_ptr().cast::<u64>(), self.len()) }),
            ArrayTag::F32 => F32(unsafe { std::slice::from_raw_parts(self.raw_array_ptr().cast::<f32>(), self.len()) }),
            ArrayTag::F64 => F64(unsafe { std::slice::from_raw_parts(self.raw_array_ptr().cast::<f64>(), self.len()) }),
            _ => Heterogeneous(unsafe { std::slice::from_raw_parts(self.array_ptr(), self.len()) }),
        }
    }
}

trait HeaderMut<'a>: ThinMutExt<'a, Header> {
    fn array_ptr_mut(mut self) -> *mut IValue {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr_mut().add(1).cast::<IValue>() }
    }
    fn raw_array_ptr_mut(mut self) -> *mut u8 {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr_mut().add(1).cast::<u8>() }
    }
    fn items_slice_mut(self) -> ArraySliceMut<'a> {
        use ArraySliceMut::*;
        let len = self.len();
        match self.type_tag() {
            ArrayTag::I8 => I8(unsafe { std::slice::from_raw_parts_mut(self.raw_array_ptr_mut().cast::<i8>(), len) }),
            ArrayTag::U8 => U8(unsafe { std::slice::from_raw_parts_mut(self.raw_array_ptr_mut().cast::<u8>(), len) }),
            ArrayTag::I16 => I16(unsafe { std::slice::from_raw_parts_mut(self.raw_array_ptr_mut().cast::<i16>(), len) }),
            ArrayTag::U16 => U16(unsafe { std::slice::from_raw_parts_mut(self.raw_array_ptr_mut().cast::<u16>(), len) }),
            ArrayTag::I32 => I32(unsafe { std::slice::from_raw_parts_mut(self.raw_array_ptr_mut().cast::<i32>(), len) }),
            ArrayTag::U32 => U32(unsafe { std::slice::from_raw_parts_mut(self.raw_array_ptr_mut().cast::<u32>(), len) }),
            ArrayTag::I64 => I64(unsafe { std::slice::from_raw_parts_mut(self.raw_array_ptr_mut().cast::<i64>(), len) }),
            ArrayTag::U64 => U64(unsafe { std::slice::from_raw_parts_mut(self.raw_array_ptr_mut().cast::<u64>(), len) }),
            ArrayTag::F32 => F32(unsafe { std::slice::from_raw_parts_mut(self.raw_array_ptr_mut().cast::<f32>(), len) }),
            ArrayTag::F64 => F64(unsafe { std::slice::from_raw_parts_mut(self.raw_array_ptr_mut().cast::<f64>(), len) }),
            _ => Heterogeneous(unsafe { std::slice::from_raw_parts_mut(self.array_ptr_mut(), len) }),
        }
    }

    // Safety: Space must already be allocated for the item
    unsafe fn push(&mut self, item: IValue) {
        use ArrayTag::*;
        let index = self.len();
        match self.type_tag() {
            Heterogeneous => self.reborrow().array_ptr_mut().add(index).write(item),
            _ => panic!("Can only push IValue items onto heterogeneous arrays"),
        }
        self.set_len(index + 1);
    }
    fn pop(&mut self) -> Option<IValue> {
        if self.len() == 0 {
            None
        } else {
            use ArrayTag::*;

            let new_len = self.len() - 1;
            self.set_len(new_len);
            let index = new_len;

            let array_type = self.type_tag();
            // Safety: We just checked that an item exists
            match array_type {
                Heterogeneous => Some(unsafe { self.reborrow().array_ptr_mut().add(index).read() }),
                I8 => Some(unsafe { self.reborrow().raw_array_ptr_mut().cast::<i8>().add(index).read() }).map(IValue::from),
                U8 => Some(unsafe { self.reborrow().raw_array_ptr_mut().cast::<u8>().add(index).read() }).map(IValue::from),
                I16 => Some(unsafe { self.reborrow().raw_array_ptr_mut().cast::<i16>().add(index).read() }).map(IValue::from),
                U16 => Some(unsafe { self.reborrow().raw_array_ptr_mut().cast::<u16>().add(index).read() }).map(IValue::from),
                I32 => Some(unsafe { self.reborrow().raw_array_ptr_mut().cast::<i32>().add(index).read() }).map(IValue::from),
                U32 => Some(unsafe { self.reborrow().raw_array_ptr_mut().cast::<u32>().add(index).read() }).map(IValue::from),
                F32 => Some(unsafe { self.reborrow().raw_array_ptr_mut().cast::<f32>().add(index).read() }).map(IValue::from),
                I64 => Some(unsafe { self.reborrow().raw_array_ptr_mut().cast::<i64>().add(index).read() }).map(IValue::from),
                U64 => Some(unsafe { self.reborrow().raw_array_ptr_mut().cast::<u64>().add(index).read() }).map(IValue::from),
                F64 => Some(unsafe { self.reborrow().raw_array_ptr_mut().cast::<f64>().add(index).read() }).map(IValue::from),
            }
        }
    }
}

impl<'a, T: ThinRefExt<'a, Header>> HeaderRef<'a> for T {}
impl<'a, T: ThinMutExt<'a, Header>> HeaderMut<'a> for T {}

/// Iterator over [`IValue`]s returned from [`IArray::into_iter`]
pub struct IntoIter {
    reversed_array: IArray,
}

impl Iterator for IntoIter {
    type Item = IValue;

    fn next(&mut self) -> Option<Self::Item> {
        self.reversed_array.pop()
    }
}

impl ExactSizeIterator for IntoIter {
    fn len(&self) -> usize {
        self.reversed_array.len()
    }
}

impl Debug for IntoIter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoIter")
            .field("reversed_array", &self.reversed_array)
            .finish()
    }
}

/// The `IArray` type is similar to a `Vec<IValue>`. The primary difference is
/// that the length and capacity are stored _inside_ the heap allocation, so that
/// the `IArray` itself can be a single pointer.
#[repr(transparent)]
#[derive(Clone)]
pub struct IArray(pub(crate) IValue);

value_subtype_impls!(IArray, into_array, as_array, as_array_mut);

static EMPTY_HEADER: Header = Header::new(0, 0, ArrayTag::Heterogeneous);

impl ArrayTag {
    fn element_size(self) -> usize {
        use ArrayTag::*;
        use std::mem::size_of;
        match self {
            Heterogeneous => size_of::<IValue>(),
            I8 => size_of::<i8>(),
            U8 => size_of::<u8>(),
            I16 => size_of::<i16>(),
            U16 => size_of::<u16>(),
            I32 => size_of::<i32>(),
            U32 => size_of::<u32>(),
            F32 => size_of::<f32>(),
            I64 => size_of::<i64>(),
            U64 => size_of::<u64>(),
            F64 => size_of::<f64>(),
        }
    }
}

impl IArray {
    fn layout(cap: usize, tag: ArrayTag) -> Result<Layout, LayoutError> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<u8>(cap * tag.element_size())?)?
            .0
            .pad_to_align())
    }

    fn alloc(cap: usize, tag: ArrayTag) -> *mut Header {
        unsafe {
            let ptr = alloc(Self::layout(cap, tag).unwrap()).cast::<Header>();
            ptr.write(Header::new(0, cap, tag));
            ptr
        }
    }

    fn realloc(ptr: *mut Header, new_cap: usize) -> *mut Header {
        unsafe {
            let tag = (*ptr).type_tag();
            let old_layout = Self::layout((*ptr).cap(), tag).unwrap();
            let new_layout = Self::layout(new_cap, tag).unwrap();
            let ptr = realloc(ptr.cast::<u8>(), old_layout, new_layout.size()).cast::<Header>();
            (*ptr).set_cap(new_cap);
            ptr
        }
    }

    fn dealloc(ptr: *mut Header) {
        unsafe {
            let tag = (*ptr).type_tag();
            let layout = Self::layout((*ptr).cap(), tag).unwrap();
            dealloc(ptr.cast(), layout);
        }
    }

    /// Constructs a new empty `IArray`. Does not allocate.
    #[must_use]
    pub fn new() -> Self {
        unsafe { IArray(IValue::new_ref(&EMPTY_HEADER, TypeTag::ArrayOrFalse)) }
    }

    /// Constructs a new `IArray` with the specified capacity. At least that many items
    /// can be added to the array without reallocating.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_capacity_and_tag(cap, ArrayTag::Heterogeneous)
    }

    /// Constructs a new `IArray` with the specified capacity and array type.
    #[must_use]
    fn with_capacity_and_tag(cap: usize, tag: ArrayTag) -> Self {
        if cap == 0 {
            Self::new()
        } else {
            IArray(unsafe { IValue::new_ptr(Self::alloc(cap, tag).cast(), TypeTag::ArrayOrFalse) })
        }
    }

    fn header(&self) -> ThinRef<'_, Header> {
        unsafe { ThinRef::new(self.0.ptr().cast()) }
    }

    // Safety: must not be static
    unsafe fn header_mut(&mut self) -> ThinMut<'_, Header> {
        ThinMut::new(self.0.ptr().cast())
    }

    fn is_static(&self) -> bool {
        self.capacity() == 0
    }
    /// Returns the capacity of the array. This is the maximum number of items the array
    /// can hold without reallocating.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.header().cap()
    }

    /// Returns the number of items currently stored in the array.
    #[must_use]
    pub fn len(&self) -> usize {
        self.header().len()
    }



    /// Returns `true` if the array is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrows a slice of the array contents
    #[must_use]
    pub fn as_slice(&self) -> ArraySliceRef<'_> {
        self.header().items_slice()
    }

    /// Borrows a mutable slice of the array contents
    pub fn as_mut_slice(&mut self) -> ArraySliceMut<'_> {
        if self.is_static() {
            ArraySliceMut::Heterogeneous(&mut [])
        } else {
            unsafe { self.header_mut().items_slice_mut() }
        }
    }

    fn resize_internal(&mut self, cap: usize) {
        if self.is_static() || cap == 0 {
            let tag = if self.is_static() {
                ArrayTag::Heterogeneous
            } else {
                self.header().type_tag()
            };
            *self = Self::with_capacity_and_tag(cap, tag);
        } else {
            unsafe {
                let new_ptr = Self::realloc(self.0.ptr().cast(), cap);
                self.0.set_ptr(new_ptr.cast());
            }
        }
    }

    /// Reserves space for at least this many additional items.
    pub fn reserve(&mut self, additional: usize) {
        let hd = self.header();
        let current_capacity = hd.cap();
        let desired_capacity = hd.len().checked_add(additional).unwrap();
        if current_capacity >= desired_capacity {
            return;
        }
        self.resize_internal(cmp::max(current_capacity * 2, desired_capacity.max(4)));
    }

    /// Truncates the array by removing items until it is no longer than the specified
    /// length. The capacity is unchanged.
    pub fn truncate(&mut self, len: usize) {
        if self.is_static() {
            return;
        }
        unsafe {
            let mut hd = self.header_mut();
            if hd.type_tag() == ArrayTag::Heterogeneous {
                while hd.len() > len {
                    hd.pop();
                }
            } else {
                // we don't need to drop primitives
                if len < hd.len() {
                    hd.set_len(len);
                }
            }
        }
    }

    /// Removes all items from the array. The capacity is unchanged.
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Inserts a new item into the array at the specified index. Any existing items
    /// on or after this index will be shifted down to accomodate this. For large
    /// arrays, insertions near the front will be slow as it will require shifting
    /// a large number of items.
    pub fn insert(&mut self, index: usize, item: impl Into<IValue>) {
        self.reserve(1);

        unsafe {
            // Safety: cannot be static after calling `reserve`
            let mut hd = self.header_mut();
            assert!(index <= hd.len());

            // Safety: We just reserved enough space for at least one extra item
            hd.push(item.into());
            if index < hd.len() {
                use ArraySliceMut::*;
                match hd.reborrow().items_slice_mut() {
                    Heterogeneous(slice) => slice[index..].rotate_right(1),
                    _ => panic!("Can only push IValue items onto heterogeneous arrays"),
                };
            }
        }
    }

    /// Removes and returns the item at the specified index from the array. Any
    /// items after this index will be shifted back up to close the gap. For large
    /// arrays, removals from near the front will be slow as it will require shifting
    /// a large number of items.
    ///
    /// If the order of the array is unimporant, consider using [`IArray::swap_remove`].
    ///
    /// If the index is outside the array bounds, `None` is returned.
    pub fn remove(&mut self, index: usize) -> Option<IValue> {
        if index < self.len() {
            // Safety: cannot be static if index <= len
            unsafe {
                use ArraySliceMut::*;
                let mut hd = self.header_mut();
                match hd.reborrow().items_slice_mut() {
                    Heterogeneous(slice) => slice[index..].rotate_left(1),
                    I8(slice) => slice[index..].rotate_left(1),
                    U8(slice) => slice[index..].rotate_left(1),
                    I16(slice) => slice[index..].rotate_left(1),
                    U16(slice) => slice[index..].rotate_left(1),
                    I32(slice) => slice[index..].rotate_left(1),
                    U32(slice) => slice[index..].rotate_left(1),
                    F32(slice) => slice[index..].rotate_left(1),
                    I64(slice) => slice[index..].rotate_left(1),
                    U64(slice) => slice[index..].rotate_left(1),
                    F64(slice) => slice[index..].rotate_left(1),
                };
                hd.pop()
            }
        } else {
            None
        }
    }

    /// Removes and returns the item at the specified index from the array by
    /// first swapping it with the item currently at the end of the array, and
    /// then popping that last item.
    ///
    /// This can be more efficient than [`IArray::remove`] for large arrays,
    /// but will change the ordering of items within the array.
    ///
    /// If the index is outside the array bounds, `None` is returned.
    pub fn swap_remove(&mut self, index: usize) -> Option<IValue> {
        if index < self.len() {
            // Safety: cannot be static if index <= len
            unsafe {
                use ArraySliceMut::*;
                let mut hd = self.header_mut();
                let last_index = hd.len() - 1;
                match hd.reborrow().items_slice_mut() {
                    Heterogeneous(slice) => slice.swap(index, last_index),
                    I8(slice) => slice.swap(index, last_index),
                    U8(slice) => slice.swap(index, last_index),
                    I16(slice) => slice.swap(index, last_index),
                    U16(slice) => slice.swap(index, last_index),
                    I32(slice) => slice.swap(index, last_index),
                    U32(slice) => slice.swap(index, last_index),
                    F32(slice) => slice.swap(index, last_index),
                    I64(slice) => slice.swap(index, last_index),
                    U64(slice) => slice.swap(index, last_index),
                    F64(slice) => slice.swap(index, last_index),
                };
                hd.pop()
            }
        } else {
            None
        }
    }

    /// Pushes a new item onto the back of the array.
    pub fn push(&mut self, item: impl Into<IValue>) {
        self.reserve(1);
        // Safety: We just reserved enough space for at least one extra item
        unsafe {
            self.header_mut().push(item.into());
        }
    }

    /// Pops the last item from the array and returns it. If the array is
    /// empty, `None` is returned.
    pub fn pop(&mut self) -> Option<IValue> {
        if self.is_static() {
            None
        } else {
            // Safety: not static
            unsafe { self.header_mut().pop() }
        }
    }

    fn reverse(&mut self) {
        use ArraySliceMut::*;
        match self.as_mut_slice() {
            Heterogeneous(slice) => slice.reverse(),
            I8(slice) => slice.reverse(),
            U8(slice) => slice.reverse(),
            I16(slice) => slice.reverse(),
            U16(slice) => slice.reverse(),
            I32(slice) => slice.reverse(),
            U32(slice) => slice.reverse(),
            F32(slice) => slice.reverse(),
            I64(slice) => slice.reverse(),
            U64(slice) => slice.reverse(),
            F64(slice) => slice.reverse(),
        }
    }

    /// Shrinks the memory allocation used by the array such that its
    /// capacity becomes equal to its length.
    pub fn shrink_to_fit(&mut self) {
        self.resize_internal(self.len());
    }

    pub(crate) fn clone_impl(&self) -> IValue {
        let hd = self.header();
        let len = hd.len();
        let tag = hd.type_tag();

        // Preserve the original capacity, not just the length
        let mut res = Self::with_capacity_and_tag(len, tag);

        if len > 0 {
            if tag == ArrayTag::Heterogeneous {
                // For heterogeneous arrays, clone IValue objects
                let ArraySliceRef::Heterogeneous(src) = hd.items_slice() else {
                    unreachable!()
                };
                unsafe {
                    // Safety: we cannot be static if len > 0
                    let mut res_hd = res.header_mut();
                    for v in src {
                        // Safety: we reserved enough space at the start
                        res_hd.push(v.clone());
                    }
                }
            } else {
                // For typed arrays, copy raw primitive data
                unsafe {
                    let src_ptr = hd.raw_array_ptr();
                    let dst_ptr = res.header_mut().raw_array_ptr_mut();
                    let element_size = tag.element_size();
                    let total_bytes = len * element_size;
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, total_bytes);

                    // Update the length
                    res.header_mut().set_len(len);
                }
            }
        }
        res.0
    }
    pub(crate) fn drop_impl(&mut self) {
        self.clear();
        if !self.is_static() {
            unsafe {
                Self::dealloc(self.0.ptr().cast());
                self.0.set_ref(&EMPTY_HEADER);
            }
        }
    }

    pub(crate) fn mem_allocated(&self) -> usize {
        if self.is_static() {
            0
        } else {
            let tag = self.header().type_tag();
            let layout_size = Self::layout(self.capacity(), tag).unwrap().size();
            if let ArraySliceRef::Heterogeneous(slice) = self.as_slice() {
                // For heterogeneous arrays, include memory allocated by contained IValue objects
                layout_size + slice.iter().map(IValue::mem_allocated).sum::<usize>()
            } else {
                // For typed arrays, just return the layout size since primitives don't allocate additional memory
                layout_size
            }
        }
    }
}

impl<A: DefragAllocator> Defrag<A> for IArray {
    fn defrag(mut self, defrag_allocator: &mut A) -> Self {
        if self.is_static() {
            return self;
        }
        match self.as_mut_slice() {
            ArraySliceMut::Heterogeneous(slice) => {
                for i in 0..slice.len() {
                    unsafe {
                        let val = slice.as_ptr().add(i).read();
                        let val = val.defrag(defrag_allocator);
                        std::ptr::write(slice.as_ptr().add(i) as *mut IValue, val);
                    }
                }
            }
            _ => {}, // typed arrays don't need defragmentation
        }
        
        unsafe {
            let header = &*self.0.ptr().cast::<Header>();
            let tag = header.type_tag();
            let new_ptr = defrag_allocator.realloc_ptr(
                self.0.ptr(),
                Self::layout(header.cap(), tag)
                    .expect("layout is expected to return a valid value"),
            );
            self.0.set_ptr(new_ptr.cast());
        }
        self
    }
}

impl IntoIterator for IArray {
    type Item = IValue;
    type IntoIter = IntoIter;

    fn into_iter(mut self) -> Self::IntoIter {
        self.reverse();
        IntoIter {
            reversed_array: self,
        }
    }
}

// impl Deref for IArray {
//     type Target = [IValue];

//     fn deref(&self) -> &Self::Target {
//         self.as_slice()
//     }
// }

// impl DerefMut for IArray {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         self.as_mut_slice()
//     }
// }

// impl Borrow<[IValue]> for IArray {
//     fn borrow(&self) -> &[IValue] {
//         self.as_slice()
//     }
// }

// impl BorrowMut<[IValue]> for IArray {
//     fn borrow_mut(&mut self) -> &mut [IValue] {
//         self.as_mut_slice()
//     }
// }

impl Hash for IArray {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use ArraySliceRef::*;
        match self.as_slice() {
            Heterogeneous(slice) => slice.hash(state),
            I8(slice) => slice.hash(state),
            U8(slice) => slice.hash(state),
            I16(slice) => slice.hash(state),
            U16(slice) => slice.hash(state),
            I32(slice) => slice.hash(state),
            U32(slice) => slice.hash(state),
            I64(slice) => slice.hash(state),
            U64(slice) => slice.hash(state),
            F32(slice) => slice.iter().map(|f| IValue::from(*f)).collect::<Vec<_>>().hash(state),
            F64(slice) => slice.iter().map(|f| IValue::from(*f)).collect::<Vec<_>>().hash(state),
        }
    }
}

macro_rules! extend_impl {
    ($($ty:ty),*) => {
        $(impl Extend<$ty> for IArray {
            fn extend<T: IntoIterator<Item = $ty>>(&mut self, iter: T) {
                let expected_tag = ArrayTag::from_type::<$ty>();
                let actual_tag = self.header().type_tag();
                if actual_tag != expected_tag {
                    panic!("Type tag mismatch, expected {:?} but found {:?}", expected_tag, actual_tag);
                }
                let iter = iter.into_iter();
                self.reserve(iter.size_hint().0);

                // For typed arrays, store raw primitive values directly
                unsafe {
                    let start_index = self.header().len();
                    let hd = self.header_mut();
                    let array_ptr = hd.raw_array_ptr_mut().cast::<$ty>();
                    let mut index = start_index;

                    for v in iter {
                        array_ptr.add(index).write(v);
                        index += 1;
                    }

                    // Update length after writing all values
                    let mut hd = self.header_mut();
                    hd.set_len(index);
                }
            }
        })*
    };
}

extend_impl!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl<U: Into<IValue>> Extend<U> for IArray {
    default fn extend<T: IntoIterator<Item = U>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for v in iter {
            self.push(v);
        }
    }
}

impl<U: Into<IValue>> FromIterator<U> for IArray {
    default fn from_iter<T: IntoIterator<Item = U>>(iter: T) -> Self {
        let mut res = IArray::new();
        res.extend(iter);
        res
    }
}

macro_rules! from_iter_impl {
    ($($ty:ty),*) => {
        $(impl FromIterator<$ty> for IArray {
            fn from_iter<T: IntoIterator<Item = $ty>>(iter: T) -> Self {
                let iter = iter.into_iter();
                let mut res = IArray::with_capacity_and_tag(iter.size_hint().0, ArrayTag::from_type::<$ty>());
                res.extend(iter);
                res
            }
        })*
    };
}

from_iter_impl!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

// impl AsRef<[IValue]> for IArray {
//     fn as_ref(&self) -> &[IValue] {
//         self.as_slice()
//     }
// }

impl PartialEq for IArray {
    fn eq(&self, other: &Self) -> bool {
        if self.0.raw_eq(&other.0) {
            true
        } else if self.header().type_tag() != other.header().type_tag() {
            false
        } else {
            use ArraySliceRef::*;
            match (self.as_slice(), other.as_slice()) {
                (Heterogeneous(a), Heterogeneous(b)) => a == b,
                (I8(a), I8(b)) => a == b,
                (U8(a), U8(b)) => a == b,
                (I16(a), I16(b)) => a == b,
                (U16(a), U16(b)) => a == b,
                (I32(a), I32(b)) => a == b,
                (U32(a), U32(b)) => a == b,
                (F32(a), F32(b)) => a == b,
                (I64(a), I64(b)) => a == b,
                (U64(a), U64(b)) => a == b,
                (F64(a), F64(b)) => a == b,
                _ => false, // Different types should never reach here due to the type_tag check above
            }
        }
    }
}

impl Eq for IArray {}
impl PartialOrd for IArray {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.0.raw_eq(&other.0) {
            Some(Ordering::Equal)
        } else {
            self.as_slice().partial_cmp(&other.as_slice())
        }
    }
}

impl<I: SliceIndex<[IValue]>> Index<I> for IArray {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        match self.as_slice() {
            ArraySliceRef::Heterogeneous(slice) => Index::index(slice, index),
            _ => panic!("Invalid index access"),
        }
    }
}

impl<I: SliceIndex<[IValue]>> IndexMut<I> for IArray {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        match self.as_mut_slice() {
            ArraySliceMut::Heterogeneous(slice) => IndexMut::index_mut(slice, index),
            _ => panic!("Invalid index access"),
        }
    }
}

impl Debug for IArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

macro_rules! from_vec_impl {
    ($($ty:ty),*) => {
        $(impl From<Vec<$ty>> for IArray {
            fn from(other: Vec<$ty>) -> Self {
                let mut res = IArray::with_capacity_and_tag(other.len(), ArrayTag::from_type::<$ty>());
                res.extend(other.into_iter());
                res
            }
        })*
    };
}

from_vec_impl!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl<T: Into<IValue>> From<Vec<T>> for IArray {
    default fn from(other: Vec<T>) -> Self {
        let mut res = IArray::with_capacity(other.len());
        res.extend(other.into_iter().map(Into::into));
        res
    }
}

macro_rules! from_slice_impl {
    ($($ty:ty),*) => {
        $(impl From<&[$ty]> for IArray {
            fn from(other: &[$ty]) -> Self {
                let mut res = IArray::with_capacity_and_tag(other.len(), ArrayTag::from_type::<$ty>());
                res.extend(other.iter().cloned());
                res
            }
        })*
    };
}

from_slice_impl!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl<T: Into<IValue> + Clone> From<&[T]> for IArray {
    default fn from(other: &[T]) -> Self {
        let mut res = IArray::with_capacity(other.len());
        res.extend(other.iter().cloned().map(Into::into));
        res
    }
}

/// Iterator item that can hold either a reference to an IValue or an owned IValue
/// This avoids deep copying for heterogeneous arrays while still providing owned values for primitives
#[derive(Debug)]
pub enum ArrayIterItem<'a> {
    /// Reference to an IValue (for heterogeneous arrays - no deep copy)
    Borrowed(&'a IValue),
    /// Owned IValue (for primitive types - converted from primitive)
    Owned(IValue),
}

impl<'a> ArrayIterItem<'a> {
    /// Get a reference to the IValue
    pub fn as_ref(&self) -> &IValue {
        match self {
            ArrayIterItem::Borrowed(val) => val,
            ArrayIterItem::Owned(val) => val,
        }
    }

    /// Convert to an owned IValue
    pub fn into_owned(self) -> IValue {
        match self {
            ArrayIterItem::Borrowed(val) => val.clone(),
            ArrayIterItem::Owned(val) => val,
        }
    }
}

/// Mutable iterator item that can hold either a mutable reference to an IValue or a mutable reference to a primitive type
/// This allows mutable iteration over both heterogeneous and homogeneous arrays
#[derive(Debug)]
pub enum ArrayIterItemMut<'a> {
    /// Mutable reference to an IValue (for heterogeneous arrays)
    Heterogeneous(&'a mut IValue),
    /// Mutable reference to an i8 value
    I8(&'a mut i8),
    /// Mutable reference to a u8 value
    U8(&'a mut u8),
    /// Mutable reference to an i16 value
    I16(&'a mut i16),
    /// Mutable reference to a u16 value
    U16(&'a mut u16),
    /// Mutable reference to an i32 value
    I32(&'a mut i32),
    /// Mutable reference to a u32 value
    U32(&'a mut u32),
    /// Mutable reference to an f32 value
    F32(&'a mut f32),
    /// Mutable reference to an i64 value
    I64(&'a mut i64),
    /// Mutable reference to a u64 value
    U64(&'a mut u64),
    /// Mutable reference to an f64 value
    F64(&'a mut f64),
}

impl<'a> ArrayIterItemMut<'a> {
    /// Get a reference to the underlying value as an IValue
    /// For primitive types, this creates a temporary IValue
    pub fn as_ivalue(&self) -> IValue {
        match self {
            ArrayIterItemMut::Heterogeneous(val) => (*val).clone(),
            ArrayIterItemMut::I8(val) => IValue::from(**val),
            ArrayIterItemMut::U8(val) => IValue::from(**val),
            ArrayIterItemMut::I16(val) => IValue::from(**val),
            ArrayIterItemMut::U16(val) => IValue::from(**val),
            ArrayIterItemMut::I32(val) => IValue::from(**val),
            ArrayIterItemMut::U32(val) => IValue::from(**val),
            ArrayIterItemMut::F32(val) => IValue::from(**val),
            ArrayIterItemMut::I64(val) => IValue::from(**val),
            ArrayIterItemMut::U64(val) => IValue::from(**val),
            ArrayIterItemMut::F64(val) => IValue::from(**val),
        }
    }

    /// Set the value from an IValue, if the types are compatible
    /// Returns true if the assignment was successful, false otherwise
    pub fn set_from_ivalue(&mut self, value: &IValue) -> bool {
        match self {
            ArrayIterItemMut::Heterogeneous(val) => {
                **val = value.clone();
                true
            }
            ArrayIterItemMut::I8(val) => {
                if let Some(v) = value.to_i64() {
                    if v >= i8::MIN as i64 && v <= i8::MAX as i64 {
                        **val = v as i8;
                        return true;
                    }
                }
                false
            }
            ArrayIterItemMut::U8(val) => {
                if let Some(v) = value.to_u64() {
                    if v <= u8::MAX as u64 {
                        **val = v as u8;
                        return true;
                    }
                }
                false
            }
            ArrayIterItemMut::I16(val) => {
                if let Some(v) = value.to_i64() {
                    if v >= i16::MIN as i64 && v <= i16::MAX as i64 {
                        **val = v as i16;
                        return true;
                    }
                }
                false
            }
            ArrayIterItemMut::U16(val) => {
                if let Some(v) = value.to_u64() {
                    if v <= u16::MAX as u64 {
                        **val = v as u16;
                        return true;
                    }
                }
                false
            }
            ArrayIterItemMut::I32(val) => {
                if let Some(v) = value.to_i64() {
                    if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                        **val = v as i32;
                        return true;
                    }
                }
                false
            }
            ArrayIterItemMut::U32(val) => {
                if let Some(v) = value.to_u64() {
                    if v <= u32::MAX as u64 {
                        **val = v as u32;
                        return true;
                    }
                }
                false
            }
            ArrayIterItemMut::F32(val) => {
                if let Some(v) = value.to_f64_lossy() {
                    **val = v as f32;
                    return true;
                }
                false
            }
            ArrayIterItemMut::I64(val) => {
                if let Some(v) = value.to_i64() {
                    **val = v;
                    return true;
                }
                false
            }
            ArrayIterItemMut::U64(val) => {
                if let Some(v) = value.to_u64() {
                    **val = v;
                    return true;
                }
                false
            }
            ArrayIterItemMut::F64(val) => {
                if let Some(v) = value.to_f64_lossy() {
                    **val = v;
                    return true;
                }
                false
            }
        }
    }
}

impl<'a> AsRef<IValue> for ArrayIterItem<'a> {
    fn as_ref(&self) -> &IValue {
        self.as_ref()
    }
}

impl<'a> std::ops::Deref for ArrayIterItem<'a> {
    type Target = IValue;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a> From<ArrayIterItem<'a>> for IValue {
    fn from(item: ArrayIterItem<'a>) -> Self {
        item.into_owned()
    }
}

impl<'a> PartialEq<IValue> for ArrayIterItem<'a> {
    fn eq(&self, other: &IValue) -> bool {
        self.as_ref() == other
    }
}

impl<'a> PartialEq<ArrayIterItem<'a>> for IValue {
    fn eq(&self, other: &ArrayIterItem<'a>) -> bool {
        self == other.as_ref()
    }
}

// Implement FromIterator for Vec<IValue> from ArrayIterItem
impl<'a> FromIterator<ArrayIterItem<'a>> for Vec<IValue> {
    fn from_iter<T: IntoIterator<Item = ArrayIterItem<'a>>>(iter: T) -> Self {
        iter.into_iter().map(|item| item.into_owned()).collect()
    }
}

// Add Serialize support for ArrayIterItem
impl<'a> serde::Serialize for ArrayIterItem<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.as_ref().serialize(serializer)
    }
}

/// Iterator over array elements, yielding ArrayIterItem objects
/// This avoids deep copying for heterogeneous arrays containing nested arrays/objects
#[derive(Debug)]
pub enum ArrayIter<'a> {
    /// Iterator over heterogeneous array
    Heterogeneous(std::slice::Iter<'a, IValue>),
    /// Iterator over i8 array
    I8(std::slice::Iter<'a, i8>),
    /// Iterator over u8 array
    U8(std::slice::Iter<'a, u8>),
    /// Iterator over i16 array
    I16(std::slice::Iter<'a, i16>),
    /// Iterator over u16 array
    U16(std::slice::Iter<'a, u16>),
    /// Iterator over i32 array
    I32(std::slice::Iter<'a, i32>),
    /// Iterator over u32 array
    U32(std::slice::Iter<'a, u32>),
    /// Iterator over f32 array
    F32(std::slice::Iter<'a, f32>),
    /// Iterator over i64 array
    I64(std::slice::Iter<'a, i64>),
    /// Iterator over u64 array
    U64(std::slice::Iter<'a, u64>),
    /// Iterator over f64 array
    F64(std::slice::Iter<'a, f64>),
}

/// Mutable iterator over array elements, yielding ArrayIterItemMut objects
/// This allows mutable iteration over both heterogeneous and homogeneous arrays
#[derive(Debug)]
pub enum ArrayIterMut<'a> {
    /// Mutable iterator over heterogeneous array
    Heterogeneous(std::slice::IterMut<'a, IValue>),
    /// Mutable iterator over i8 array
    I8(std::slice::IterMut<'a, i8>),
    /// Mutable iterator over u8 array
    U8(std::slice::IterMut<'a, u8>),
    /// Mutable iterator over i16 array
    I16(std::slice::IterMut<'a, i16>),
    /// Mutable iterator over u16 array
    U16(std::slice::IterMut<'a, u16>),
    /// Mutable iterator over i32 array
    I32(std::slice::IterMut<'a, i32>),
    /// Mutable iterator over u32 array
    U32(std::slice::IterMut<'a, u32>),
    /// Mutable iterator over f32 array
    F32(std::slice::IterMut<'a, f32>),
    /// Mutable iterator over i64 array
    I64(std::slice::IterMut<'a, i64>),
    /// Mutable iterator over u64 array
    U64(std::slice::IterMut<'a, u64>),
    /// Mutable iterator over f64 array
    F64(std::slice::IterMut<'a, f64>),
}

impl<'a> Iterator for ArrayIter<'a> {
    type Item = ArrayIterItem<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        use ArrayIter::*;
        match self {
            // For heterogeneous arrays, return borrowed references (no deep copy!)
            Heterogeneous(iter) => iter.next().map(ArrayIterItem::Borrowed),
            // For primitive types, create owned IValues
            I8(iter) => iter.next().map(|&v| ArrayIterItem::Owned(IValue::from(v))),
            U8(iter) => iter.next().map(|&v| ArrayIterItem::Owned(IValue::from(v))),
            I16(iter) => iter.next().map(|&v| ArrayIterItem::Owned(IValue::from(v))),
            U16(iter) => iter.next().map(|&v| ArrayIterItem::Owned(IValue::from(v))),
            I32(iter) => iter.next().map(|&v| ArrayIterItem::Owned(IValue::from(v))),
            U32(iter) => iter.next().map(|&v| ArrayIterItem::Owned(IValue::from(v))),
            F32(iter) => iter.next().map(|&v| ArrayIterItem::Owned(IValue::from(v))),
            I64(iter) => iter.next().map(|&v| ArrayIterItem::Owned(IValue::from(v))),
            U64(iter) => iter.next().map(|&v| ArrayIterItem::Owned(IValue::from(v))),
            F64(iter) => iter.next().map(|&v| ArrayIterItem::Owned(IValue::from(v))),
        }
    }
}

impl<'a> Iterator for ArrayIterMut<'a> {
    type Item = ArrayIterItemMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        use ArrayIterMut::*;
        match self {
            // For heterogeneous arrays, return mutable references
            Heterogeneous(iter) => iter.next().map(ArrayIterItemMut::Heterogeneous),
            // For primitive types, return mutable references to the primitives
            I8(iter) => iter.next().map(ArrayIterItemMut::I8),
            U8(iter) => iter.next().map(ArrayIterItemMut::U8),
            I16(iter) => iter.next().map(ArrayIterItemMut::I16),
            U16(iter) => iter.next().map(ArrayIterItemMut::U16),
            I32(iter) => iter.next().map(ArrayIterItemMut::I32),
            U32(iter) => iter.next().map(ArrayIterItemMut::U32),
            F32(iter) => iter.next().map(ArrayIterItemMut::F32),
            I64(iter) => iter.next().map(ArrayIterItemMut::I64),
            U64(iter) => iter.next().map(ArrayIterItemMut::U64),
            F64(iter) => iter.next().map(ArrayIterItemMut::F64),
        }
    }
}

impl<'a> ExactSizeIterator for ArrayIter<'a> {
    fn len(&self) -> usize {
        use ArrayIter::*;
        match self {
            Heterogeneous(iter) => iter.len(),
            I8(iter) => iter.len(),
            U8(iter) => iter.len(),
            I16(iter) => iter.len(),
            U16(iter) => iter.len(),
            I32(iter) => iter.len(),
            U32(iter) => iter.len(),
            F32(iter) => iter.len(),
            I64(iter) => iter.len(),
            U64(iter) => iter.len(),
            F64(iter) => iter.len(),
        }
    }
}

impl<'a> ExactSizeIterator for ArrayIterMut<'a> {
    fn len(&self) -> usize {
        use ArrayIterMut::*;
        match self {
            Heterogeneous(iter) => iter.len(),
            I8(iter) => iter.len(),
            U8(iter) => iter.len(),
            I16(iter) => iter.len(),
            U16(iter) => iter.len(),
            I32(iter) => iter.len(),
            U32(iter) => iter.len(),
            F32(iter) => iter.len(),
            I64(iter) => iter.len(),
            U64(iter) => iter.len(),
            F64(iter) => iter.len(),
        }
    }
}



impl IArray {
    /// Returns an iterator over the array elements
    ///
    /// For heterogeneous arrays, this returns references to IValues instead of cloning them,
    /// which avoids expensive deep copies for nested arrays and objects. For primitive arrays,
    /// it creates owned IValues from the primitive values.
    ///
    /// The returned iterator yields `ArrayIterItem` which can be dereferenced to `&IValue`
    /// or converted to owned `IValue` when needed.
    pub fn iter(&self) -> ArrayIter<'_> {
        use ArraySliceRef::*;
        match self.as_slice() {
            Heterogeneous(slice) => ArrayIter::Heterogeneous(slice.iter()),
            I8(slice) => ArrayIter::I8(slice.iter()),
            U8(slice) => ArrayIter::U8(slice.iter()),
            I16(slice) => ArrayIter::I16(slice.iter()),
            U16(slice) => ArrayIter::U16(slice.iter()),
            I32(slice) => ArrayIter::I32(slice.iter()),
            U32(slice) => ArrayIter::U32(slice.iter()),
            F32(slice) => ArrayIter::F32(slice.iter()),
            I64(slice) => ArrayIter::I64(slice.iter()),
            U64(slice) => ArrayIter::U64(slice.iter()),
            F64(slice) => ArrayIter::F64(slice.iter()),
        }
    }

    /// Gets a reference to the element at the given index.
    /// Only works for heterogeneous arrays (arrays containing IValue objects).
    /// For typed arrays, use `as_slice()` to get direct access to the underlying slice.
    pub fn get(&self, index: usize) -> Option<&IValue> {
        match self.as_slice() {
            ArraySliceRef::Heterogeneous(slice) => slice.get(index),
            _ => None,
        }
    }

    /// Gets a mutable reference to the element at the given index.
    /// Only works for heterogeneous arrays (arrays containing IValue objects).
    /// For typed arrays, use `as_mut_slice()` to get direct access to the underlying slice.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut IValue> {
        match self.as_mut_slice() {
            ArraySliceMut::Heterogeneous(slice) => slice.get_mut(index),
            _ => None,
        }
    }
}

impl<'a> IntoIterator for &'a IArray {
    type Item = ArrayIterItem<'a>;
    type IntoIter = ArrayIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IArray {
    /// Returns a mutable iterator over the array elements.
    /// Works for both heterogeneous arrays (arrays containing IValue objects) and homogeneous arrays (typed arrays).
    /// For heterogeneous arrays, yields mutable references to IValues.
    /// For typed arrays, yields mutable references to the primitive types.
    pub fn iter_mut(&mut self) -> ArrayIterMut<'_> {
        use ArraySliceMut::*;
        match self.as_mut_slice() {
            Heterogeneous(slice) => ArrayIterMut::Heterogeneous(slice.iter_mut()),
            I8(slice) => ArrayIterMut::I8(slice.iter_mut()),
            U8(slice) => ArrayIterMut::U8(slice.iter_mut()),
            I16(slice) => ArrayIterMut::I16(slice.iter_mut()),
            U16(slice) => ArrayIterMut::U16(slice.iter_mut()),
            I32(slice) => ArrayIterMut::I32(slice.iter_mut()),
            U32(slice) => ArrayIterMut::U32(slice.iter_mut()),
            F32(slice) => ArrayIterMut::F32(slice.iter_mut()),
            I64(slice) => ArrayIterMut::I64(slice.iter_mut()),
            U64(slice) => ArrayIterMut::U64(slice.iter_mut()),
            F64(slice) => ArrayIterMut::F64(slice.iter_mut()),
        }
    }
}

impl<'a> IntoIterator for &'a mut IArray {
    type Item = ArrayIterItemMut<'a>;
    type IntoIter = ArrayIterMut<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl Default for IArray {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec_memory_safety() {
        // Test the from_vec implementation for memory safety
        println!("Creating i32 vec");
        let vec_i32 = vec![1i32, 2, 3, 4, 5];
        println!("Converting to IArray");
        let arr = IArray::from(vec_i32);
        println!("Checking length: {}", arr.len());
        assert_eq!(arr.len(), 5);
        println!("Checking type tag: {:?}", arr.header().type_tag());
        assert_eq!(arr.header().type_tag(), ArrayTag::I32);
        println!("i32 array test passed");

        // Test drop by letting arr go out of scope
        drop(arr);
        println!("Drop completed successfully");

        // Test other types
        let vec_f64 = vec![1.0f64, 2.0, 3.0];
        let arr2 = IArray::from(vec_f64);
        assert_eq!(arr2.len(), 3);
        assert_eq!(arr2.header().type_tag(), ArrayTag::F64);

        let vec_u8 = vec![1u8, 2, 3];
        let arr3 = IArray::from(vec_u8);
        assert_eq!(arr3.len(), 3);
        assert_eq!(arr3.header().type_tag(), ArrayTag::U8);

        println!("All typed array tests passed");
    }

    #[test]
    fn test_header_packing() {
        // Test that the header correctly packs and unpacks values
        let header = Header::new(123, 456, ArrayTag::I32);
        assert_eq!(header.len(), 123);
        assert_eq!(header.cap(), 456);
        assert_eq!(header.type_tag(), ArrayTag::I32);

        // Test maximum values (30 bits = 1,073,741,823)
        let max_30_bit = (1usize << 30) - 1;
        let header_max = Header::new(max_30_bit, max_30_bit, ArrayTag::F64);
        assert_eq!(header_max.len(), max_30_bit);
        assert_eq!(header_max.cap(), max_30_bit);
        assert_eq!(header_max.type_tag(), ArrayTag::F64);

        // Test that the header is 64 bits (8 bytes)
        assert_eq!(std::mem::size_of::<Header>(), 8);
    }

    #[test]
    fn test_typed_slice() {
        // Test heterogeneous array
        let hetero_array: IArray = vec![IValue::NULL, IValue::TRUE, IValue::FALSE].into();
        match hetero_array.as_slice() {
            ArraySliceRef::Heterogeneous(slice) => {
                assert_eq!(slice.len(), 3);
                assert_eq!(slice[0], IValue::NULL);
                assert_eq!(slice[1], IValue::TRUE);
                assert_eq!(slice[2], IValue::FALSE);
            }
            _ => panic!("Expected heterogeneous slice"),
        }

        // Test typed i32 array
        let i32_array: IArray = vec![1i32, 2i32, 3i32].into();
        match i32_array.as_slice() {
            ArraySliceRef::I32(slice) => {
                assert_eq!(slice.len(), 3);
                assert_eq!(slice[0], 1);
                assert_eq!(slice[1], 2);
                assert_eq!(slice[2], 3);
            }
            _ => panic!("Expected i32 slice"),
        }

        // Test typed f64 array
        let f64_array: IArray = vec![1.0f64, 2.5f64, 3.14f64].into();
        match f64_array.as_slice() {
            ArraySliceRef::F64(slice) => {
                assert_eq!(slice.len(), 3);
                assert_eq!(slice[0], 1.0);
                assert_eq!(slice[1], 2.5);
                assert_eq!(slice[2], 3.14);
            }
            _ => panic!("Expected f64 slice"),
        }
    }

    #[test]
    fn test_iteration() {
        // Test iteration over heterogeneous array
        let hetero_array: IArray = vec![IValue::NULL, IValue::TRUE, IValue::FALSE].into();
        let collected: Vec<IValue> = hetero_array.iter().collect();
        assert_eq!(collected, vec![IValue::NULL, IValue::TRUE, IValue::FALSE]);

        // Test iteration over typed i32 array
        let i32_array: IArray = vec![1i32, 2i32, 3i32].into();
        let collected: Vec<IValue> = i32_array.iter().collect();
        assert_eq!(collected, vec![IValue::from(1i32), IValue::from(2i32), IValue::from(3i32)]);

        // Test iteration over typed f64 array
        let f64_array: IArray = vec![1.0f64, 2.5f64, 3.14f64].into();
        let collected: Vec<IValue> = f64_array.iter().collect();
        assert_eq!(collected, vec![IValue::from(1.0f64), IValue::from(2.5f64), IValue::from(3.14f64)]);

        // Test IntoIterator for &IArray
        let values: Vec<IValue> = (&i32_array).into_iter().collect();
        assert_eq!(values, vec![IValue::from(1i32), IValue::from(2i32), IValue::from(3i32)]);
    }

    #[test]
    fn test_iter_efficient_behavior() {
        // Test that iter() now avoids deep copies for heterogeneous arrays
        let original_string = IValue::from("test_string");
        let nested_array = IValue::from(vec![IValue::from(1), IValue::from(2)]);
        let hetero_array: IArray = vec![original_string.clone(), nested_array.clone(), IValue::NULL].into();

        // Iterate using the new efficient iterator
        let mut iter = hetero_array.iter();

        // First item should be a borrowed reference to the string
        let first_item = iter.next().unwrap();
        assert_eq!(*first_item, original_string);

        // For strings, both should point to the same data (no deep copy)
        if let (Some(original_str), Some(iter_str)) = (original_string.as_string(), first_item.as_string()) {
            assert_eq!(original_str.as_ptr(), iter_str.as_ptr());
        }

        // Second item should be a borrowed reference to the nested array (no deep copy!)
        let second_item = iter.next().unwrap();
        assert_eq!(*second_item, nested_array);

        // Third item should be NULL
        let third_item = iter.next().unwrap();
        assert_eq!(*third_item, IValue::NULL);

        // Test that we can convert to owned values when needed
        let collected: Vec<IValue> = hetero_array.iter().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], original_string);
        assert_eq!(collected[1], nested_array);
        assert_eq!(collected[2], IValue::NULL);
    }

    #[test]
    fn test_iter_creates_new_ivalues_for_primitives() {
        // Test that iter() creates new IValue instances for primitive types
        let i32_array: IArray = vec![42i32, 100i32].into();

        // Iterate and collect values
        let collected: Vec<IValue> = i32_array.iter().collect();

        // Verify the values are correct
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].to_i64(), Some(42));
        assert_eq!(collected[1].to_i64(), Some(100));

        // These should be newly created IValue instances from the primitive data
        assert!(collected[0].is_number());
        assert!(collected[1].is_number());
    }

    #[test]
    fn test_mutable_iteration() {
        // Test mutable iteration over heterogeneous array
        let mut hetero_array: IArray = vec![IValue::from(1), IValue::from(2), IValue::from(3)].into();

        // Test iter_mut() method
        for mut_item in hetero_array.iter_mut() {
            match mut_item {
                ArrayIterItemMut::Heterogeneous(value) => {
                    if let Some(n) = value.to_f64_lossy() {
                        *value = IValue::from(n + 1.0);
                    }
                }
                _ => {} // Should not happen for heterogeneous array
            }
        }

        let collected: Vec<IValue> = hetero_array.iter().collect();
        assert_eq!(collected, vec![IValue::from(2), IValue::from(3), IValue::from(4)]);

        // Test IntoIterator for &mut IArray on heterogeneous array
        for mut_item in &mut hetero_array {
            match mut_item {
                ArrayIterItemMut::Heterogeneous(value) => {
                    if let Some(n) = value.to_f64_lossy() {
                        *value = IValue::from(n * 2.0);
                    }
                }
                _ => {} // Should not happen for heterogeneous array
            }
        }

        let collected: Vec<IValue> = hetero_array.iter().collect();
        assert_eq!(collected, vec![IValue::from(4), IValue::from(6), IValue::from(8)]);

        // Test mutable iteration over typed arrays
        let mut i32_array: IArray = vec![1i32, 2i32, 3i32].into();
        for mut_item in i32_array.iter_mut() {
            match mut_item {
                ArrayIterItemMut::I32(value) => {
                    *value += 10;
                }
                _ => {} // Should not happen for i32 array
            }
        }

        // Verify the changes
        match i32_array.as_slice() {
            ArraySliceRef::I32(slice) => {
                assert_eq!(slice, &[11, 12, 13]);
            }
            _ => panic!("Expected i32 array"),
        }
    }

    #[test]
    fn test_mutable_iteration_on_typed_array() {
        let mut i32_array: IArray = vec![1i32, 2i32, 3i32].into();

        // This should now work (no longer panics)
        let mut count = 0;
        for mut_item in &mut i32_array {
            match mut_item {
                ArrayIterItemMut::I32(value) => {
                    *value *= 2;
                    count += 1;
                }
                _ => panic!("Expected i32 values"),
            }
        }

        assert_eq!(count, 3);

        // Verify the changes
        match i32_array.as_slice() {
            ArraySliceRef::I32(slice) => {
                assert_eq!(slice, &[2, 4, 6]);
            }
            _ => panic!("Expected i32 array"),
        }
    }

    #[test]
    fn test_mutable_iteration_different_types() {
        // Test f64 array
        let mut f64_array: IArray = vec![1.5f64, 2.5f64, 3.5f64].into();
        for mut_item in f64_array.iter_mut() {
            match mut_item {
                ArrayIterItemMut::F64(value) => {
                    *value += 0.5;
                }
                _ => panic!("Expected f64 values"),
            }
        }

        match f64_array.as_slice() {
            ArraySliceRef::F64(slice) => {
                assert_eq!(slice, &[2.0, 3.0, 4.0]);
            }
            _ => panic!("Expected f64 array"),
        }

        // Test u8 array
        let mut u8_array: IArray = vec![10u8, 20u8, 30u8].into();
        for mut_item in u8_array.iter_mut() {
            match mut_item {
                ArrayIterItemMut::U8(value) => {
                    *value = value.saturating_add(5);
                }
                _ => panic!("Expected u8 values"),
            }
        }

        match u8_array.as_slice() {
            ArraySliceRef::U8(slice) => {
                assert_eq!(slice, &[15, 25, 35]);
            }
            _ => panic!("Expected u8 array"),
        }
    }

    #[test]
    fn test_typed_array_iteration() {
        let i32_array: IArray = vec![1i32, 2i32, 3i32].into();
        let f64_array: IArray = vec![1.0f64, 2.5f64, 3.14f64].into();

        let i32_values: Vec<IValue> = i32_array.iter().collect();
        assert_eq!(i32_values, vec![IValue::from(1i32), IValue::from(2i32), IValue::from(3i32)]);

        let f64_values: Vec<IValue> = f64_array.iter().collect();
        assert_eq!(f64_values, vec![IValue::from(1.0f64), IValue::from(2.5f64), IValue::from(3.14f64)]);

        // Test that we can convert typed arrays to heterogeneous arrays for deserialization
        let hetero_from_i32: IArray = i32_array.iter().collect();
        assert!(matches!(hetero_from_i32.as_slice(), ArraySliceRef::Heterogeneous(_)));

        let hetero_from_f64: IArray = f64_array.iter().collect();
        assert!(matches!(hetero_from_f64.as_slice(), ArraySliceRef::Heterogeneous(_)));

        // Verify the converted arrays have the same values
        let hetero_i32_values: Vec<IValue> = hetero_from_i32.iter().collect();
        assert_eq!(hetero_i32_values, i32_values);

        let hetero_f64_values: Vec<IValue> = hetero_from_f64.iter().collect();
        assert_eq!(hetero_f64_values, f64_values);
    }

    #[test]
    fn test_typed_array_deserialization() {
        use serde::Deserialize;
        
        // Test that we can deserialize from a typed i32 array directly
        let i32_array: IArray = vec![1i32, 2i32, 3i32].into();
        let deserialized_vec: Vec<i32> = Deserialize::deserialize(&i32_array).unwrap();
        assert_eq!(deserialized_vec, vec![1, 2, 3]);

        // Test with f64 array
        let f64_array: IArray = vec![1.5f64, 2.5f64, 3.5f64].into();
        let deserialized_f64: Vec<f64> = Deserialize::deserialize(&f64_array).unwrap();
        assert_eq!(deserialized_f64, vec![1.5, 2.5, 3.5]);

        // Test with u8 array
        let u8_array: IArray = vec![10u8, 20u8, 30u8].into();
        let deserialized_u8: Vec<u8> = Deserialize::deserialize(&u8_array).unwrap();
        assert_eq!(deserialized_u8, vec![10, 20, 30]);

        // Test that heterogeneous arrays still work
        let hetero_array: IArray = vec![IValue::from(1), IValue::from(2), IValue::from(3)].into();
        let deserialized_hetero: Vec<i32> = Deserialize::deserialize(&hetero_array).unwrap();
        assert_eq!(deserialized_hetero, vec![1, 2, 3]);
    }

    #[mockalloc::test]
    fn can_create() {
        let x = IArray::new();
        let y = IArray::with_capacity(10);

        assert_eq!(x, y);
    }

    #[mockalloc::test]
    fn can_collect() {
        let x = vec![IValue::NULL, IValue::TRUE, IValue::FALSE];
        let y: IArray = x.iter().cloned().collect();

        assert_eq!(y.as_slice(), x.as_slice().into());
    }

    #[mockalloc::test]
    fn can_push_insert() {
        let mut x = IArray::new();
        x.insert(0, IValue::NULL);
        x.push(IValue::TRUE);
        x.insert(1, IValue::FALSE);

        assert_eq!(x.as_slice(), [IValue::NULL, IValue::FALSE, IValue::TRUE].as_slice().into());
    }

    #[mockalloc::test]
    fn can_nest() {
        let x: IArray = vec![IValue::NULL, IValue::TRUE, IValue::FALSE].into();
        let y: IArray = vec![
            IValue::NULL,
            x.clone().into(),
            IValue::FALSE,
            x.clone().into(),
        ]
        .into();

        assert_eq!(&y[1], x.as_ref());
    }

    #[mockalloc::test]
    fn can_pop_remove() {
        let mut x: IArray = vec![IValue::NULL, IValue::TRUE, IValue::FALSE].into();
        assert_eq!(x.remove(1), Some(IValue::TRUE));
        assert_eq!(x.pop(), Some(IValue::FALSE));

        assert_eq!(x.as_slice(), [IValue::NULL].as_slice().into());
    }

    #[mockalloc::test]
    fn can_swap_remove() {
        let mut x: IArray = vec![IValue::NULL, IValue::TRUE, IValue::FALSE].into();
        assert_eq!(x.swap_remove(0), Some(IValue::NULL));

        assert_eq!(x.as_slice(), [IValue::FALSE, IValue::TRUE].as_slice().into());
    }

    #[mockalloc::test]
    fn can_index() {
        let mut x: IArray = vec![IValue::NULL, IValue::TRUE, IValue::FALSE].into();
        assert_eq!(x[1], IValue::TRUE);
        x[1] = IValue::FALSE;
        assert_eq!(x[1], IValue::FALSE);
    }

    #[mockalloc::test]
    fn can_truncate_and_shrink() {
        let mut x: IArray =
            vec![IValue::NULL, IValue::TRUE, IArray::with_capacity(10).into()].into();
        x.truncate(2);
        assert_eq!(x.len(), 2);
        assert_eq!(x.capacity(), 3);
        x.shrink_to_fit();
        assert_eq!(x.len(), 2);
        assert_eq!(x.capacity(), 2);
    }

    // Too slow for miri
    #[cfg(not(miri))]
    #[mockalloc::test]
    fn stress_test() {
        use rand::prelude::*;

        for i in 0..10 {
            // We want our test to be random but for errors to be reproducible
            let mut rng = StdRng::seed_from_u64(i);
            let mut arr = IArray::new();

            for j in 0..1000 {
                let index = rng.gen_range(0..arr.len() + 1);
                if rng.gen() {
                    arr.insert(index, j);
                } else {
                    arr.remove(index);
                }
            }
        }
    }
}
