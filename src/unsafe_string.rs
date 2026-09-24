//! Functionality relating to the JSON string type

use hashbrown::HashSet;
use std::alloc::{alloc, dealloc, Layout, LayoutError};
use std::borrow::Borrow;
use std::cell::{Cell, UnsafeCell};
use std::cmp::Ordering;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::mem::transmute;
use std::ops::Deref;
use std::ptr::{addr_of_mut, copy_nonoverlapping, NonNull};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::{
    thin::{ThinMut, ThinMutExt, ThinRef, ThinRefExt},
    value::{TypeTag, ALIGNMENT, TAG_SIZE_BITS},
    Defrag, DefragAllocator, IValue,
};

#[repr(C)]
struct Header {
    // We use 32 bits for the length, which allows up to 4 GiB (safely covers 512MB)
    len: u32,
}

trait HeaderRef<'a>: ThinRefExt<'a, Header> {
    fn len(&self) -> usize {
        self.len as usize
    }
    fn str_ptr(&self) -> *const u8 {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr().add(1).cast() }
    }
    fn bytes(&self) -> &'a [u8] {
        // Safety: Header `len` must be accurate
        unsafe { std::slice::from_raw_parts(self.str_ptr(), self.len()) }
    }
    fn str(&self) -> &'a str {
        // Safety: UTF-8 enforced on construction
        unsafe { std::str::from_utf8_unchecked(self.bytes()) }
    }
}

trait HeaderMut<'a>: ThinMutExt<'a, Header> {
    fn str_ptr_mut(mut self) -> *mut u8 {
        // Safety: pointers to the end of structs are allowed
        unsafe { self.ptr_mut().add(1).cast() }
    }
}

impl<'a, T: ThinRefExt<'a, Header>> HeaderRef<'a> for T {}
impl<'a, T: ThinMutExt<'a, Header>> HeaderMut<'a> for T {}

// Constants for inline string storage
const INLINE_STRING_MAX_LEN: usize = 7;
/// Check if a string can be stored inline
fn can_inline_string(s: &str) -> bool {
    let bytes = s.as_bytes();
    bytes.len() <= INLINE_STRING_MAX_LEN
}

enum StringCache {
    ThreadSafe(Mutex<HashSet<WeakIString>>),
    ThreadUnsafe(UnsafeCell<HashSet<WeakIString>>),
}

static mut STRING_CACHE: OnceLock<StringCache> = OnceLock::new();

pub(crate) fn reinit_cache() {
    // The cache now owns reference counts; live entries must survive a reset.
    get_cache_guard().shrink();
}

pub(crate) fn init_cache(thread_safe: bool) -> Result<(), String> {
    let s_c = unsafe { &*addr_of_mut!(STRING_CACHE) };
    s_c.set(if thread_safe {
        StringCache::ThreadSafe(Mutex::new(HashSet::new()))
    } else {
        StringCache::ThreadUnsafe(UnsafeCell::new(HashSet::new()))
    })
    .map_err(|_| "Cache is already initialized".to_owned())
}

fn get_cache() -> &'static StringCache {
    // SAFETY: the static is never assigned after initialization. OnceLock and
    // Mutex synchronize the thread-safe path without aliasing mutable references.
    let s_c = unsafe { &*addr_of_mut!(STRING_CACHE) };
    s_c.get_or_init(|| StringCache::ThreadUnsafe(UnsafeCell::new(HashSet::new())))
}

enum CacheGuard {
    ThreadUnsafe(&'static mut HashSet<WeakIString>),
    ThreadSafe(MutexGuard<'static, HashSet<WeakIString>>),
}

impl CacheGuard {
    fn get_or_insert(&mut self, value: &str, f: impl FnOnce(&str) -> WeakIString) -> &WeakIString {
        match self {
            CacheGuard::ThreadSafe(c_g) => c_g.get_or_insert_with(value, |val| f(val)),
            CacheGuard::ThreadUnsafe(c_g) => c_g.get_or_insert_with(value, |val| f(val)),
        }
    }

    fn get_val(&self, val: &str) -> Option<&WeakIString> {
        match self {
            CacheGuard::ThreadSafe(c_g) => c_g.get(val),
            CacheGuard::ThreadUnsafe(c_g) => c_g.get(val),
        }
    }

    fn remove_val(&mut self, val: &str) -> bool {
        match self {
            CacheGuard::ThreadSafe(c_g) => c_g.remove(val),
            CacheGuard::ThreadUnsafe(c_g) => c_g.remove(val),
        }
    }

    #[cfg(test)]
    fn check_if_empty(&self) -> bool {
        match self {
            CacheGuard::ThreadSafe(c_g) => c_g.is_empty(),
            CacheGuard::ThreadUnsafe(c_g) => c_g.is_empty(),
        }
    }

    fn shrink(&mut self) {
        match self {
            CacheGuard::ThreadSafe(c_g) => c_g.shrink_to_fit(),
            CacheGuard::ThreadUnsafe(c_g) => c_g.shrink_to_fit(),
        }
    }
}

fn get_cache_guard() -> CacheGuard {
    let s_c = get_cache();
    match s_c {
        StringCache::ThreadUnsafe(s_c) => {
            // SAFETY: callers selecting this mode must externally serialize all
            // cache operations, as required by init_shared_string_cache(false).
            CacheGuard::ThreadUnsafe(unsafe { &mut *s_c.get() })
        }
        StringCache::ThreadSafe(s_c) => {
            CacheGuard::ThreadSafe(s_c.lock().expect("Mutex lock should succeed"))
        }
    }
}

struct SharedHeader {
    data: NonNull<Header>,
    rc: Cell<u32>,
}

struct WeakIString {
    // Low bit set: SharedHeader; otherwise a unique Header. Both are aligned.
    // Mutated only with the cache guard held; string content/hash never changes.
    ptr: Cell<NonNull<u8>>,
}

impl PartialEq for WeakIString {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Eq for WeakIString {}
impl Hash for WeakIString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl Deref for WeakIString {
    type Target = str;
    fn deref(&self) -> &str {
        self.borrow()
    }
}

impl Borrow<str> for WeakIString {
    fn borrow(&self) -> &str {
        self.header().str()
    }
}
impl WeakIString {
    fn shared(&self) -> Option<&SharedHeader> {
        let ptr = self.ptr.get().as_ptr();
        if ptr.addr() & 1 == 0 {
            None
        } else {
            // SAFETY: the tagged pointer owns a live Box<SharedHeader>; the cache
            // guard prevents promotion/demotion while this reference is used.
            Some(unsafe { &*ptr.with_addr(ptr.addr() & !1).cast::<SharedHeader>() })
        }
    }

    fn data(&self) -> NonNull<Header> {
        self.shared()
            .map_or_else(|| self.ptr.get().cast(), |s| s.data)
    }

    fn header(&self) -> ThinRef<'_, Header> {
        // SAFETY: the cache entry always refers to a live immutable string.
        unsafe { ThinRef::new(self.data().as_ptr()) }
    }

    fn value(&self) -> IString {
        // SAFETY: the caller has registered ownership of this aligned allocation.
        unsafe {
            IString(IValue::new_ptr(
                self.data().as_ptr().cast(),
                TypeTag::StringOrNull,
            ))
        }
    }

    fn upgrade(&self) -> IString {
        if let Some(shared) = self.shared() {
            shared.rc.set(
                shared
                    .rc
                    .get()
                    .checked_add(1)
                    .expect("string reference count overflow"),
            );
        } else {
            let shared = Box::new(SharedHeader {
                data: self.data(),
                rc: Cell::new(2),
            });
            let ptr = Box::into_raw(shared).cast::<u8>();
            // SAFETY: Box pointers are non-null; bit 0 is free by alignment.
            self.ptr
                .set(unsafe { NonNull::new_unchecked(ptr.with_addr(ptr.addr() | 1)) });
        }
        self.value()
    }

    // Returns true when the departing owner was the last one.
    fn release(&self) -> bool {
        let Some(shared) = self.shared() else {
            return true;
        };
        let remaining = shared.rc.get() - 1;
        shared.rc.set(remaining);
        if remaining == 1 {
            let ptr = self.ptr.get().as_ptr();
            self.ptr.set(shared.data.cast());
            // SAFETY: no other code accesses the descriptor under this guard.
            // The string remains alive, and the table entry is now unique.
            unsafe {
                drop(Box::from_raw(
                    ptr.with_addr(ptr.addr() & !1).cast::<SharedHeader>(),
                ));
            }
        }
        false
    }
}

/// The `IString` type is an interned, immutable string, and is where this crate
/// gets its name.
///
/// Cloning an `IString` looks up its ownership in the cache. It can be converted
/// from `&str` or `String`. Comparisons between `IString`s use a pointer
/// comparison.
///
/// The memory backing an `IString` is reference counted, so that unlike many
/// string interning libraries, memory is not leaked as new strings are interned.
/// One hash-set stores unique pointers or shared descriptors. In thread-safe
/// mode, interning, cloning and dropping take the cache mutex.
///
/// Given the nature of `IString` it is better to intern a string once and reuse
/// it, rather than continually convert from `&str` to `IString`.
#[repr(transparent)]
#[derive(Clone)]
pub struct IString(pub(crate) IValue);

value_subtype_impls!(IString, into_string, as_string, as_string_mut);

#[repr(align(8))]
struct EmptyHeader(Header);
static EMPTY_HEADER: EmptyHeader = EmptyHeader(Header { len: 0 });

impl IString {
    fn layout(len: usize) -> Result<Layout, LayoutError> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<u8>(len)?)?
            .0
            .align_to(ALIGNMENT)?
            .pad_to_align())
    }

    fn alloc<A: FnOnce(Layout) -> *mut u8>(s: &str, allocator: A) -> *mut Header {
        assert!((s.len()) < u32::MAX as usize);
        unsafe {
            let ptr = allocator(
                Self::layout(s.len()).expect("layout is expected to return a valid value"),
            )
            .cast::<Header>();
            let ptr = NonNull::new(ptr)
                .unwrap_or_else(|| std::alloc::handle_alloc_error(Self::layout(s.len()).unwrap()))
                .as_ptr();
            ptr.write(Header {
                len: s.len() as u32,
            });
            let hd = ThinMut::new(ptr);
            copy_nonoverlapping(s.as_ptr(), hd.str_ptr_mut(), s.len());
            ptr
        }
    }

    fn dealloc<D: FnOnce(*mut u8, Layout)>(ptr: *mut Header, deallocator: D) {
        unsafe {
            let hd = ThinRef::new(ptr);
            let layout = Self::layout(hd.len()).unwrap();
            deallocator(ptr.cast::<u8>(), layout);
        }
    }

    fn intern_with_allocator<A: FnOnce(Layout) -> *mut u8>(s: &str, allocator: A) -> Self {
        if s.is_empty() {
            return Self::new();
        }

        let mut cache = get_cache_guard();
        let mut inserted = false;
        let entry = cache.get_or_insert(s, |s| {
            inserted = true;
            WeakIString {
                ptr: Cell::new(
                    NonNull::new(Self::alloc(s, allocator).cast())
                        .expect("string allocation failed"),
                ),
            }
        });
        if inserted {
            entry.value()
        } else {
            entry.upgrade()
        }
    }

    /// Create an inline string by storing bytes in upper bits
    /// Safety: String must be < 8 bytes and valid UTF-8
    unsafe fn new_inline_string(s: &str) -> Self {
        // 1 byte for the tag(3 bits for tag and rest for the length), 7 bytes for the string
        let bytes = s.as_bytes();
        let mut data_bytes = [0u8; 8];

        // Set the length in the first byte (after tag bits)
        data_bytes[0] = (s.len() << TAG_SIZE_BITS) as u8;
        data_bytes[1..1 + bytes.len()].copy_from_slice(bytes);
        let data: usize = usize::from_ne_bytes(data_bytes);

        Self(IValue::new_ptr(data as *mut u8, TypeTag::InlineString))
    }

    /// Converts a `&str` to an `IString` by interning it in the global string cache.
    #[must_use]
    pub fn intern(s: &str) -> Self {
        if s.is_empty() {
            return Self::new();
        } else if can_inline_string(s) {
            unsafe { Self::new_inline_string(s) }
        } else {
            Self::intern_with_allocator(s, |layout| unsafe { alloc(layout) })
        }
    }

    fn is_inline(&self) -> bool {
        (self.0.ptr_usize() % ALIGNMENT) == TypeTag::InlineString as usize
    }

    fn header(&self) -> ThinRef<'_, Header> {
        // SAFETY: a non-inline string owns a live immutable header.
        unsafe { ThinRef::new(self.0.ptr().cast()) }
    }

    /// Returns the length (in bytes) of this string.
    #[must_use]
    pub fn len(&self) -> usize {
        if self.is_inline() {
            let data = self.0.ptr_usize() as u64;
            let len_data = (data & 0xFF) >> TAG_SIZE_BITS;
            len_data as usize
        } else {
            self.header().len()
        }
    }

    /// Returns `true` if this is the empty string "".
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Extract string from inline storage
    /// Safety: Must be called on inline string(strings are valid UTF-8)
    unsafe fn extract_inline_str(&self) -> &str {
        let data_ptr = &self.0 as *const IValue as *const u8;
        let bytes: &[u8; 8] = transmute(data_ptr);
        str::from_utf8_unchecked(&bytes[1..self.len() + 1])
    }

    /// Obtains a `&str` from this `IString`. This is a cheap operation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        if self.is_inline() {
            unsafe { self.extract_inline_str() }
        } else {
            self.header().str()
        }
    }

    /// Obtains a byte slice from this `IString`. This is a cheap operation.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.as_str().as_bytes()
    }

    /// Returns the empty string.
    #[must_use]
    pub fn new() -> Self {
        unsafe { IString(IValue::new_ref(&EMPTY_HEADER.0, TypeTag::StringOrNull)) }
    }

    pub(crate) fn clone_impl(&self) -> IValue {
        if self.is_empty() {
            Self::new().0
        } else if self.is_inline() {
            unsafe { self.0.raw_copy() }
        } else {
            let cache = get_cache_guard();
            cache
                .get_val(self.as_str())
                .expect("live string missing from cache")
                .upgrade()
                .0
        }
    }

    pub(crate) fn drop_impl(&mut self) {
        if !self.is_empty() && !self.is_inline() {
            let mut cache = get_cache_guard();
            if cache
                .get_val(self.as_str())
                .expect("live string missing from cache")
                .release()
            {
                cache.remove_val(self.as_str());

                // Shrink the cache if it is empty in tests to verify no memory leaks
                #[cfg(test)]
                if cache.check_if_empty() {
                    cache.shrink();
                }
                // SAFETY: the cache entry was removed after its last owner departed.
                Self::dealloc(unsafe { self.0.ptr().cast() }, |ptr, layout| unsafe {
                    dealloc(ptr, layout)
                });
            }
        }
    }

    pub(crate) fn mem_allocated(&self) -> usize {
        if self.is_empty() || self.is_inline() {
            0
        } else {
            let cache = get_cache_guard();
            let entry = cache
                .get_val(self.as_str())
                .expect("live string missing from cache");
            Self::layout(self.len()).unwrap().size()
                + if entry.shared().is_some() {
                    std::mem::size_of::<SharedHeader>()
                } else {
                    0
                }
        }
    }
}

impl Deref for IString {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for IString {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for IString {
    fn from(other: &str) -> Self {
        Self::intern(other)
    }
}

impl From<&mut str> for IString {
    fn from(other: &mut str) -> Self {
        Self::intern(other)
    }
}

impl From<String> for IString {
    fn from(other: String) -> Self {
        Self::intern(other.as_str())
    }
}

impl From<&String> for IString {
    fn from(other: &String) -> Self {
        Self::intern(other.as_str())
    }
}

impl From<&mut String> for IString {
    fn from(other: &mut String) -> Self {
        Self::intern(other.as_str())
    }
}

impl From<IString> for String {
    fn from(other: IString) -> Self {
        other.as_str().into()
    }
}

impl PartialEq for IString {
    fn eq(&self, other: &Self) -> bool {
        if self.0.raw_eq(&other.0) {
            // if we have the same exact point we know they are equals.
            return true;
        }
        // otherwise we need to compare the strings.
        let s1 = self.as_str();
        let s2 = other.as_str();
        let res = s1 == s2;
        res
    }
}

impl PartialEq<str> for IString {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<IString> for str {
    fn eq(&self, other: &IString) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<String> for IString {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<IString> for String {
    fn eq(&self, other: &IString) -> bool {
        self == other.as_str()
    }
}

impl Default for IString {
    fn default() -> Self {
        Self::new()
    }
}

impl Eq for IString {}
impl Ord for IString {
    fn cmp(&self, other: &Self) -> Ordering {
        if self == other {
            Ordering::Equal
        } else {
            self.as_str().cmp(other.as_str())
        }
    }
}
impl PartialOrd for IString {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Hash for IString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl Debug for IString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_str(), f)
    }
}

impl<A: DefragAllocator> Defrag<A> for IString {
    fn defrag(mut self, defrag_allocator: &mut A) -> Self {
        if self.is_empty() || self.is_inline() {
            return self;
        }
        let cache = get_cache_guard();
        let entry = cache
            .get_val(self.as_str())
            .expect("live string missing from cache");
        if entry.shared().is_none() {
            // SAFETY: this is the sole owner, and the cache guard prevents a new
            // owner. Relocation updates both the cache and this value together.
            unsafe {
                let ptr =
                    defrag_allocator.realloc_ptr(self.0.ptr(), Self::layout(self.len()).unwrap());
                entry
                    .ptr
                    .set(NonNull::new(ptr).expect("defrag allocation failed"));
                self.0.set_ptr(ptr);
            }
        }
        // ponytail: shared records stay pinned; moving them needs stable handles
        // or a forwarding mechanism to preserve every owner's pointer.
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockalloc::record_allocs;

    fn assert_no_allocs<F: FnOnce()>(f: F) {
        let alloc_info = record_allocs(f);
        assert_eq!(
            alloc_info.num_allocs(),
            0,
            "Expected zero allocations, but {} occurred",
            alloc_info.num_allocs()
        );
    }

    #[test]
    fn test_inline_string_as_str() {
        assert_no_allocs(|| {
            let s = IString::intern("hello");
            assert_eq!(s.as_str(), "hello");
        });
    }

    #[mockalloc::test]
    fn can_intern() {
        let x = IString::intern("foofoofoo");
        let y = IString::intern("bar");
        let z = IString::intern("foofoofoo");

        assert_eq!(x.as_ptr(), z.as_ptr());
        assert_ne!(x.as_ptr(), y.as_ptr());
        assert_eq!(x.as_str(), "foofoofoo");
        assert_eq!(y.as_str(), "bar");
    }

    #[test]
    fn default_interns_string() {
        assert_no_allocs(|| {
            let x = IString::intern("");
            let y = IString::new();
            let z = IString::intern("foo");

            assert_eq!(x.as_ptr(), y.as_ptr());
            assert_ne!(x.as_ptr(), z.as_ptr());
        });
    }

    #[mockalloc::test]
    fn test_inline_strings() {
        // Test strings that should be stored inline (≤ 7 bytes)
        let short_strings = ["", "a", "hi", "hello", "world", "1234567", "12345678"];

        for s in &short_strings {
            let istr = IString::intern(s);

            if s.is_empty() {
                // Empty strings use static header, not inline
                assert!(!istr.is_inline());
            } else if s.len() <= INLINE_STRING_MAX_LEN {
                assert!(istr.is_inline(), "String '{}' should be inline", s);
                assert_eq!(istr.as_str(), *s);
                assert_eq!(istr.len(), s.len());
                assert_eq!(istr.as_bytes(), s.as_bytes());

                // Inline strings should have minimal memory overhead
                assert_eq!(istr.mem_allocated(), 0);
            } else {
                assert!(!istr.is_inline(), "String '{}' should not be inline", s);
            }
        }
    }

    #[mockalloc::test]
    fn test_heap_strings() {
        // Test strings that should be stored on heap (> 7 bytes)
        let long_string = "a".repeat(100);
        let long_strings = ["12345678", "toolongstring", &long_string];

        for s in &long_strings {
            let istr = IString::intern(s);
            assert!(!istr.is_inline(), "String '{}' should not be inline", s);
            assert_eq!(istr.as_str(), *s);
            assert_eq!(istr.len(), s.len());
            assert_eq!(istr.as_bytes(), s.as_bytes());

            // Heap strings should have memory overhead
            assert!(istr.mem_allocated() > 0);
        }
    }

    #[mockalloc::test]
    fn test_utf8_boundary_safety() {
        // Test that we don't inline strings that would break UTF-8 boundaries
        let emoji = "🦀"; // 4 bytes in UTF-8
        let multi_emoji = "🦀🔥"; // 8 bytes in UTF-8 - too long for inline

        let crab = IString::intern(emoji);
        assert!(crab.is_inline(), "Single emoji should be inline");
        assert_eq!(crab.as_str(), emoji);

        let fire_crab = IString::intern(multi_emoji);
        assert!(!fire_crab.is_inline(), "Two emojis should not be inline");
        assert_eq!(fire_crab.as_str(), multi_emoji);
    }

    #[test]
    fn test_inline_string_cloning() {
        assert_no_allocs(|| {
            let original = IString::intern("hello");
            assert!(original.is_inline());

            let cloned = original.clone();
            assert!(cloned.is_inline());
            assert_eq!(original.as_str(), cloned.as_str());

            // Both should point to the same inline data
            assert_eq!(original.0.ptr_usize(), cloned.0.ptr_usize());
        });
    }
    #[mockalloc::test]
    fn unique_promotes_and_demotes_without_moving_bytes() {
        assert_eq!(std::mem::size_of::<Header>(), 4);
        assert_eq!(std::mem::size_of::<WeakIString>(), 8);
        let first = IString::intern("Redis is very fast. "); // 20 bytes
        let borrowed = first.as_str();
        let address = borrowed.as_ptr();
        assert_eq!(first.mem_allocated(), 24);
        assert!(get_cache_guard()
            .get_val(borrowed)
            .unwrap()
            .shared()
            .is_none());

        let second = IString::intern(borrowed);
        let third = first.clone();
        assert_eq!(second.as_ptr(), address);
        assert_eq!(third.as_ptr(), address);
        assert_eq!(
            first.mem_allocated(),
            24 + std::mem::size_of::<SharedHeader>()
        );
        assert_eq!(
            get_cache_guard()
                .get_val(borrowed)
                .unwrap()
                .shared()
                .unwrap()
                .rc
                .get(),
            3
        );
        reinit_cache();
        drop(first);
        assert_eq!(second.as_str(), "Redis is very fast. ");
        drop(third);
        assert!(get_cache_guard()
            .get_val(second.as_str())
            .unwrap()
            .shared()
            .is_none());
        assert_eq!(second.mem_allocated(), 24);
        let promoted_again = second.clone();
        assert_eq!(promoted_again.as_ptr(), address);
        drop(second);
        assert_eq!(promoted_again.as_str(), "Redis is very fast. ");
        drop(promoted_again);
        assert!(get_cache_guard().get_val("Redis is very fast. ").is_none());
    }

    #[mockalloc::test]
    fn defrag_moves_unique_but_preserves_shared_pointers() {
        struct MovingAllocator(usize);
        impl DefragAllocator for MovingAllocator {
            unsafe fn realloc_ptr<T>(&mut self, ptr: *mut T, layout: Layout) -> *mut T {
                self.0 += 1;
                // SAFETY: the caller supplies a live allocation with this layout.
                unsafe {
                    let new = self.alloc(layout);
                    copy_nonoverlapping(ptr.cast(), new, layout.size());
                    self.free(ptr, layout);
                    new.cast()
                }
            }
            unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
                // SAFETY: layout comes from the live string allocation.
                unsafe {
                    NonNull::new(alloc(layout))
                        .unwrap_or_else(|| std::alloc::handle_alloc_error(layout))
                        .as_ptr()
                }
            }
            unsafe fn free<T>(&mut self, ptr: *mut T, layout: Layout) {
                // SAFETY: the caller has finished using this allocation.
                unsafe {
                    dealloc(ptr.cast(), layout);
                }
            }
        }
        let mut allocator = MovingAllocator(0);
        for text in ["", "short", "שלום עולם 🌍", "long\0string\nwith controls"] {
            let original = IString::intern(text);
            let relocated = original.defrag(&mut allocator);
            assert_eq!(relocated.as_str(), text);
            let shared = relocated.clone();
            let address = shared.as_ptr();
            let moves = allocator.0;
            let relocated = relocated.defrag(&mut allocator);
            assert_eq!(allocator.0, moves);
            if text.len() > 7 {
                assert_eq!(relocated.as_ptr(), address);
            }
            assert_eq!(shared.as_str(), text);
        }
        assert_eq!(allocator.0, 2);
    }

    // Run in its own test process because cache initialization is process-wide.
    #[test]
    #[ignore = "run separately with --ignored --exact"]
    fn concurrent_promotion() {
        init_cache(true).unwrap();
        let keep = IString::intern("concurrently shared string");
        let address = keep.as_ptr() as usize;
        let threads: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        let value = IString::intern("concurrently shared string");
                        let cloned = value.clone();
                        assert_eq!(cloned.as_ptr() as usize, address);
                        drop(value);
                        assert_eq!(cloned.as_str(), "concurrently shared string");
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert!(get_cache_guard()
            .get_val(keep.as_str())
            .unwrap()
            .shared()
            .is_none());
        drop(keep);
        assert!(get_cache_guard().check_if_empty());
    }
}
