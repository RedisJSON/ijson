//! Functionality relating to the JSON string type

use hashbrown::HashSet;
use std::alloc::{alloc, dealloc, Layout, LayoutError};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::mem;
use std::ops::Deref;
use std::ptr::{addr_of_mut, copy_nonoverlapping, NonNull};
use std::sync::atomic::AtomicUsize;
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::thin::{ThinMut, ThinMutExt, ThinRef, ThinRefExt};
use crate::{Defrag, DefragAllocator};

use super::value::{IValue, TypeTag};

#[repr(C)]
#[repr(align(4))]
struct Header {
    rc: AtomicUsize,
    // We use 48 bits for the length.
    len_lower: u32,
    len_upper: u16,
}

trait HeaderRef<'a>: ThinRefExt<'a, Header> {
    fn len(&self) -> usize {
        (u64::from(self.len_lower) | (u64::from(self.len_upper) << 32)) as usize
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

enum StringCache {
    ThreadSafe(Mutex<HashSet<WeakIString>>),
    ThreadUnsafe(HashSet<WeakIString>),
}

static mut STRING_CACHE: OnceLock<StringCache> = OnceLock::new();

pub(crate) fn reinit_cache() {
    let s_c = get_cache_mut();
    match s_c {
        StringCache::ThreadUnsafe(s_c) => *s_c = HashSet::new(),
        StringCache::ThreadSafe(s_c) => {
            let mut s_c: std::sync::MutexGuard<'_, HashSet<WeakIString>> =
                s_c.lock().expect("Mutex lock should succeed");
            *s_c = HashSet::new();
        }
    }
}

pub(crate) fn init_cache(thread_safe: bool) -> Result<(), String> {
    let s_c = unsafe { &*addr_of_mut!(STRING_CACHE) };
    s_c.set(if thread_safe {
        StringCache::ThreadSafe(Mutex::new(HashSet::new()))
    } else {
        StringCache::ThreadUnsafe(HashSet::new())
    })
    .map_err(|_| "Cache is already initialized".to_owned())
}

fn get_cache_mut() -> &'static mut StringCache {
    let s_c = unsafe { &mut *addr_of_mut!(STRING_CACHE) };
    s_c.get_or_init(|| StringCache::ThreadUnsafe(HashSet::new()));
    s_c.get_mut().unwrap()
}

fn is_thread_safe() -> bool {
    match get_cache_mut() {
        StringCache::ThreadSafe(_) => true,
        StringCache::ThreadUnsafe(_) => false,
    }
}

enum CacheGuard {
    ThreadUnsafe(&'static mut HashSet<WeakIString>),
    ThreadSafe(MutexGuard<'static, HashSet<WeakIString>>),
}

impl CacheGuard {
    fn get_or_insert<'a>(
        &mut self,
        value: &str,
        f: Box<dyn FnOnce(&str) -> WeakIString + 'a>,
    ) -> &WeakIString {
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

    #[cfg(test)]
    fn shrink(&mut self) {
        match self {
            CacheGuard::ThreadSafe(c_g) => c_g.shrink_to_fit(),
            CacheGuard::ThreadUnsafe(c_g) => c_g.shrink_to_fit(),
        }
    }
}

fn get_cache_guard() -> CacheGuard {
    let s_c = get_cache_mut();
    match s_c {
        StringCache::ThreadUnsafe(s_c) => CacheGuard::ThreadUnsafe(s_c),
        StringCache::ThreadSafe(s_c) => {
            CacheGuard::ThreadSafe(s_c.lock().expect("Mutex lock should succeed"))
        }
    }
}

struct WeakIString {
    ptr: NonNull<Header>,
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
    fn header(&self) -> ThinMut<Header> {
        // Safety: pointer is always valid
        unsafe { ThinMut::new(self.ptr.as_ptr()) }
    }
    fn upgrade(&self) -> IString {
        unsafe {
            self.header()
                .rc
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            IString(IValue::new_ptr(
                self.ptr.as_ptr().cast::<u8>(),
                TypeTag::StringOrNull,
            ))
        }
    }
}

/// The `IString` type is an interned, immutable string, and is where this crate
/// gets its name.
///
/// Cloning an `IString` is cheap, and it can be easily converted from `&str` or
/// `String` types. Comparisons between `IString`s is a simple pointer
/// comparison.
///
/// The memory backing an `IString` is reference counted, so that unlike many
/// string interning libraries, memory is not leaked as new strings are interned.
/// Interning uses `DashSet`, an implementation of a concurrent hash-set, allowing
/// many strings to be interned concurrently without becoming a bottleneck.
///
/// Given the nature of `IString` it is better to intern a string once and reuse
/// it, rather than continually convert from `&str` to `IString`.
#[repr(transparent)]
#[derive(Clone)]
pub struct IString(pub(crate) IValue);

value_subtype_impls!(IString, into_string, as_string, as_string_mut);

static EMPTY_HEADER: Header = Header {
    len_lower: 0,
    len_upper: 0,
    rc: AtomicUsize::new(0),
};

impl IString {
    fn layout(len: usize) -> Result<Layout, LayoutError> {
        Ok(Layout::new::<Header>()
            .extend(Layout::array::<u8>(len)?)?
            .0
            .pad_to_align())
    }

    fn alloc<A: FnOnce(Layout) -> *mut u8>(s: &str, allocator: A) -> *mut Header {
        assert!((s.len() as u64) < (1 << 48));
        unsafe {
            let ptr = allocator(
                Self::layout(s.len()).expect("layout is expected to return a valid value"),
            )
            .cast::<Header>();
            ptr.write(Header {
                len_lower: s.len() as u32,
                len_upper: ((s.len() as u64) >> 32) as u16,
                rc: AtomicUsize::new(0),
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

        let k = cache.get_or_insert(
            s,
            Box::new(|s| WeakIString {
                ptr: unsafe { NonNull::new_unchecked(Self::alloc(s, allocator)) },
            }),
        );
        k.upgrade()
    }

    /// Converts a `&str` to an `IString` by interning it in the global string cache.
    #[must_use]
    pub fn intern(s: &str) -> Self {
        Self::intern_with_allocator(s, |layout| unsafe { alloc(layout) })
    }

    fn header(&self) -> ThinMut<Header> {
        unsafe { ThinMut::new(self.0.ptr().cast()) }
    }

    /// Returns the length (in bytes) of this string.
    #[must_use]
    pub fn len(&self) -> usize {
        self.header().len()
    }

    /// Returns `true` if this is the empty string "".
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Obtains a `&str` from this `IString`. This is a cheap operation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.header().str()
    }

    /// Obtains a byte slice from this `IString`. This is a cheap operation.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.header().bytes()
    }

    /// Returns the empty string.
    #[must_use]
    pub fn new() -> Self {
        unsafe { IString(IValue::new_ref(&EMPTY_HEADER, TypeTag::StringOrNull)) }
    }

    pub(crate) fn clone_impl(&self) -> IValue {
        if self.is_empty() {
            Self::new().0
        } else {
            self.header()
                .rc
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            unsafe { self.0.raw_copy() }
        }
    }

    fn drop_impl_with_deallocator<D: FnOnce(*mut u8, Layout)>(&mut self, deallocator: D) {
        if !self.is_empty() {
            let hd = self.header();

            if is_thread_safe() {
                // Optimization for the thread safe case, we want to avoid locking the cache if the ref count
                // is not potentially going to reach zero.
                let mut rc = hd.rc.load(std::sync::atomic::Ordering::Relaxed);
                while rc > 1 {
                    match hd.rc.compare_exchange_weak(
                        rc,
                        rc - 1,
                        std::sync::atomic::Ordering::Relaxed,
                        std::sync::atomic::Ordering::Relaxed,
                    ) {
                        Ok(_) => return,
                        Err(new_rc) => rc = new_rc,
                    }
                }
            }

            let mut cache = get_cache_guard();
            if hd.rc.fetch_sub(1, std::sync::atomic::Ordering::Relaxed) == 1 {
                // Reference count reached zero, free the string
                if let Some(element) = cache.get_val(hd.str()) {
                    // we can not simply remove the element from the cache, while we
                    // perform active defrag, the element might be in the cache but will
                    // point to another (newer) value. In this case we do not want to remove it.
                    if element.ptr.as_ptr().cast() == unsafe { self.0.ptr() } {
                        cache.remove_val(hd.str());
                    }
                }

                // Shrink the cache if it is empty in tests to verify no memory leaks
                #[cfg(test)]
                if cache.check_if_empty() {
                    cache.shrink();
                }
                Self::dealloc(unsafe { self.0.ptr().cast() }, deallocator);
            }
        }
    }

    pub(crate) fn drop_impl(&mut self) {
        self.drop_impl_with_deallocator(|ptr, layout| unsafe { dealloc(ptr, layout) });
    }

    pub(crate) fn mem_allocated(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            Self::layout(self.len()).unwrap().size()
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
        let new = Self::intern_with_allocator(self.as_str(), |layout| unsafe {
            defrag_allocator.alloc(layout)
        });
        self.drop_impl_with_deallocator(|ptr, layout| unsafe {
            defrag_allocator.free(ptr, layout)
        });
        mem::forget(self);
        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[mockalloc::test]
    fn can_intern() {
        let x = IString::intern("foo");
        let y = IString::intern("bar");
        let z = IString::intern("foo");

        assert_eq!(x.as_ptr(), z.as_ptr());
        assert_ne!(x.as_ptr(), y.as_ptr());
        assert_eq!(x.as_str(), "foo");
        assert_eq!(y.as_str(), "bar");
    }

    #[mockalloc::test]
    fn default_interns_string() {
        let x = IString::intern("");
        let y = IString::new();
        let z = IString::intern("foo");

        assert_eq!(x.as_ptr(), y.as_ptr());
        assert_ne!(x.as_ptr(), z.as_ptr());
    }
}
