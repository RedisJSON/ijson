//! What a stream encoder sees when the address of a freed string is given to
//! another string.
//!
//! Whether a system allocator hands a freed block to the next allocation is up to
//! it, so these tests install one that always does: the block freed last goes to
//! the next allocation of the same layout. Reuse then happens every time instead
//! of when the allocator feels like it.

use ijson::{
    encode, DocUse, IArray, IString, IValue, StreamDecoder, StreamEncoder, StreamEncoderConfig,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// How many freed blocks are kept back for reuse; the others are freed at once.
const SLOTS: usize = 64;

/// Freed blocks, the most recently freed last.
struct Freed {
    len: usize,
    blocks: [Option<(usize, Layout)>; SLOTS],
}

impl Freed {
    /// Takes back the block freed last that has exactly this layout.
    fn take(&mut self, layout: Layout) -> Option<*mut u8> {
        let index = (0..self.len).rev().find(|&index| {
            self.blocks[index].is_some_and(|(_, freed_layout)| freed_layout == layout)
        })?;
        let (address, _) = self.blocks[index].take()?;
        self.blocks.copy_within(index + 1..self.len, index);
        self.len -= 1;
        self.blocks[self.len] = None;
        Some(address as *mut u8)
    }

    /// Keeps a freed block, unless there is no room for it.
    fn keep(&mut self, address: *mut u8, layout: Layout) -> bool {
        if self.len == SLOTS {
            return false;
        }
        self.blocks[self.len] = Some((address as usize, layout));
        self.len += 1;
        true
    }
}

thread_local! {
    // Only the thread that asked for reuse gets it. Others, such as the test
    // harness, allocate at any time and would take the block that is waiting.
    static RECYCLING: Cell<bool> = const { Cell::new(false) };
}

fn recycling() -> bool {
    RECYCLING.try_with(Cell::get).unwrap_or(false)
}

struct Recycler {
    busy: AtomicBool,
    freed: UnsafeCell<Freed>,
}

// SAFETY: `freed` is only reached through `locked`, which serializes the access.
unsafe impl Sync for Recycler {}

impl Recycler {
    fn locked<R>(&self, access: impl FnOnce(&mut Freed) -> R) -> R {
        while self.busy.swap(true, Ordering::Acquire) {
            std::hint::spin_loop();
        }
        // SAFETY: `busy` was false and is now ours, so no other thread holds a
        // reference into `freed` until the store below.
        let result = access(unsafe { &mut *self.freed.get() });
        self.busy.store(false, Ordering::Release);
        result
    }

    fn free_kept_blocks(&self) {
        self.locked(|freed| {
            for block in freed.blocks[..freed.len]
                .iter_mut()
                .filter_map(Option::take)
            {
                // SAFETY: the block came from `System.alloc` with this layout and
                // was kept back instead of being freed.
                unsafe { System.dealloc(block.0 as *mut u8, block.1) };
            }
            freed.len = 0;
        });
    }
}

// SAFETY: every block comes from `System` and goes back to it with the layout it
// was allocated with; blocks kept back are handed out only for that same layout.
unsafe impl GlobalAlloc for Recycler {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if recycling() {
            if let Some(block) = self.locked(|freed| freed.take(layout)) {
                return block;
            }
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, address: *mut u8, layout: Layout) {
        if recycling() && self.locked(|freed| freed.keep(address, layout)) {
            return;
        }
        System.dealloc(address, layout)
    }
}

#[global_allocator]
static ALLOCATOR: Recycler = Recycler {
    busy: AtomicBool::new(false),
    freed: UnsafeCell::new(Freed {
        len: 0,
        blocks: [None; SLOTS],
    }),
};

/// Reuse of freed blocks on this thread lasts as long as this value lives.
struct Recycling;

impl Recycling {
    fn start() -> Self {
        RECYCLING.with(|recycling| recycling.set(true));
        Recycling
    }
}

impl Drop for Recycling {
    fn drop(&mut self) {
        RECYCLING.with(|recycling| recycling.set(false));
        ALLOCATOR.free_kept_blocks();
    }
}

// Serializes the tests: another thread's allocations could take the freed block.
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    // One failing test must not make the others fail on a poisoned lock.
    TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

#[ctor::ctor]
fn init_cache() {
    ijson::init_shared_string_cache(true).unwrap();
}

fn config() -> StreamEncoderConfig {
    StreamEncoderConfig {
        table_budget_bytes: 8 << 20,
    }
}

/// An array that holds a clone of each of `strings`, in order. Each clone is one
/// more owner of its string.
fn array_of(strings: &[&IString]) -> IValue {
    let items: Vec<IValue> = strings.iter().map(|s| IValue::from((*s).clone())).collect();
    IArray::try_from(items).unwrap().into()
}

/// Strings of 33 to 40 bytes need blocks of one layout, so a block freed by one
/// of them is handed to any other.
const FIRST_LEN: usize = 33;

/// An encoder that defined a string and whose string was freed afterwards, and a
/// new string that was given the freed address.
struct Reused {
    encoder: StreamEncoder,
    first_bytes: Vec<u8>,
    /// The text that was defined.
    old_text: String,
    intruder: IString,
    /// Whether the intruder sits where the defined string did.
    at_old_address: bool,
}

fn free_and_reuse(intruder_len: usize) -> Reused {
    let mut encoder = StreamEncoder::new(config());
    let old_text = "o".repeat(FIRST_LEN);
    let original = IString::from(old_text.as_str());
    let address = original.as_ptr();
    let first = array_of(&[&original, &original]);
    let mut first_bytes = Vec::new();
    let used = encoder.encode_into(&mut first_bytes, &first);
    assert_eq!(used, DocUse { defs: 1, refs: 1 });

    // The document goes first and the string last, so that its block is the one
    // the allocator hands out next.
    drop(first);
    drop(original);
    let intruder = IString::from("i".repeat(intruder_len));
    Reused {
        encoder,
        first_bytes,
        old_text,
        at_old_address: intruder.as_ptr() == address,
        intruder,
    }
}

#[test]
fn a_reused_address_with_another_length_poisons_the_encoder() {
    let _guard = serial();
    let _recycling = Recycling::start();
    let mut reused = free_and_reuse(40);
    assert!(reused.at_old_address);

    let second = array_of(&[&reused.intruder, &reused.intruder]);
    let mut second_bytes = Vec::new();
    let used = reused.encoder.encode_into(&mut second_bytes, &second);
    assert_eq!(used, DocUse::default());
    assert_eq!(second_bytes, encode(&second));
    assert!(reused.encoder.is_poisoned());
    let mut decoder = StreamDecoder::new();
    decoder.decode(&reused.first_bytes, true).unwrap();
    assert_eq!(decoder.decode(&second_bytes, true).unwrap(), second);
}

#[test]
fn a_stale_hit_after_a_definition_in_the_same_document_keeps_the_definition() {
    let _guard = serial();
    let _recycling = Recycling::start();
    let mut reused = free_and_reuse(40);
    assert!(reused.at_old_address);

    // The new string needs two owners to be defined, and is met before the
    // address that was reused.
    let fresh = IString::from("a string defined before the stale hit");
    let second = array_of(&[&fresh, &reused.intruder, &fresh]);
    let mut second_bytes = vec![2, 0, 0, 0, 0];
    let used = reused.encoder.encode_into(&mut second_bytes, &second);
    // The definition written before the stale hit stays; everything after it is inline.
    assert_eq!(used, DocUse { defs: 1, refs: 0 });
    assert_eq!(second_bytes[..5], [2, 0, 0, 0, 0]);
    assert_eq!(reused.encoder.stats().defs, 2);
    assert!(reused.encoder.is_poisoned());
    let mut decoder = StreamDecoder::new();
    decoder.decode(&reused.first_bytes, true).unwrap();
    assert_eq!(decoder.decode(&second_bytes[5..], true).unwrap(), second);
    assert_eq!(decoder.len(), 2);
}

// Known limit: the encoder sees a reused address only when the new string has another
// length. Callers keep every defined string alive until the stream ends, as `StreamEncoder`
// documents.
#[test]
fn a_reused_address_with_the_same_length_is_not_detected() {
    let _guard = serial();
    let _recycling = Recycling::start();
    let mut reused = free_and_reuse(FIRST_LEN);
    assert!(reused.at_old_address);

    let second = array_of(&[&reused.intruder, &reused.intruder]);
    let mut second_bytes = Vec::new();
    let used = reused.encoder.encode_into(&mut second_bytes, &second);
    assert_eq!(used, DocUse { defs: 0, refs: 2 });
    assert_eq!(second_bytes, [0x82, 0xC7, 0x00, 0xC7, 0x00]);
    assert!(!reused.encoder.is_poisoned());

    // The reader resolves the reference to the string that was freed.
    let mut decoder = StreamDecoder::new();
    decoder.decode(&reused.first_bytes, true).unwrap();
    let decoded = decoder.decode(&second_bytes, true).unwrap();
    let old = IString::from(reused.old_text.as_str());
    assert_eq!(decoded, array_of(&[&old, &old]));
    assert_ne!(decoded, second);
}
