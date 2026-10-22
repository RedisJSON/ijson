//! An encoder must never write to a string it encodes, so
//! that a forked child working on its parent's pages copies none of them.
//!
//! Every string here gets a mapping of its own. The child makes the first page of
//! each mapping, which holds the string's header, read-only and then encodes: a
//! write to a header ends it. Cloning a string outside the encoder updates its owner
//! count, so it serves as the control that the child does notice a write.
#![cfg(all(
    any(target_os = "linux", target_os = "macos"),
    target_pointer_width = "64"
))]

use ijson::{IObject, IString, IValue, StreamEncoder, StreamEncoderConfig};
use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_void;
use std::ptr::null_mut;

// std links libc already; only these calls and constants are needed.
extern "C" {
    fn mmap(
        address: *mut c_void,
        len: usize,
        protection: i32,
        flags: i32,
        file: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(address: *mut c_void, len: usize) -> i32;
    fn mprotect(address: *mut c_void, len: usize, protection: i32) -> i32;
    fn sysconf(name: i32) -> i64;
    fn signal(signal: i32, handler: usize) -> usize;
    fn fork() -> i32;
    fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    fn _exit(code: i32) -> !;
}

const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 2;
const MAP_FAILED: *mut c_void = !0 as *mut c_void;
const SIGSEGV: i32 = 11;
#[cfg(target_os = "linux")]
const MAP_ANONYMOUS: i32 = 0x20;
#[cfg(target_os = "linux")]
const SC_PAGESIZE: i32 = 30;
#[cfg(target_os = "linux")]
const SIGBUS: i32 = 7;
#[cfg(target_os = "macos")]
const MAP_ANONYMOUS: i32 = 0x1000;
#[cfg(target_os = "macos")]
const SC_PAGESIZE: i32 = 29;
#[cfg(target_os = "macos")]
const SIGBUS: i32 = 10;

fn page_size() -> usize {
    // SAFETY: `sysconf` only reads its integer argument.
    unsafe { sysconf(SC_PAGESIZE) as usize }
}

fn round_up_to_pages(len: usize) -> usize {
    let page = page_size();
    (len + page - 1) / page * page
}

/// Allocations from this size on get a mapping of their own, so that the page
/// that holds a string header holds nothing else.
const MAPPED_FROM: usize = 8192;

fn gets_own_mapping(layout: Layout) -> bool {
    layout.size() >= MAPPED_FROM && layout.align() <= page_size()
}

struct Mappings;

// SAFETY: a block is either a fresh private mapping of its rounded size, which
// `dealloc` unmaps again from the same layout, or comes from `System` and goes
// back to it.
unsafe impl GlobalAlloc for Mappings {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !gets_own_mapping(layout) {
            return System.alloc(layout);
        }
        let address = mmap(
            null_mut(),
            round_up_to_pages(layout.size()),
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        );
        if address == MAP_FAILED {
            null_mut()
        } else {
            address.cast()
        }
    }

    unsafe fn dealloc(&self, address: *mut u8, layout: Layout) {
        if gets_own_mapping(layout) {
            munmap(address.cast(), round_up_to_pages(layout.size()));
        } else {
            System.dealloc(address, layout)
        }
    }
}

#[global_allocator]
static ALLOCATOR: Mappings = Mappings;

#[ctor::ctor]
fn init_cache() {
    ijson::init_shared_string_cache(true).unwrap();
}

const DOCS: usize = 128;
const STRING_BYTES: usize = 20 * 1024;

const EXIT_OK: i32 = 0;
const EXIT_WROTE_TO_A_STRING: i32 = 1;
const EXIT_PROTECT_FAILED: i32 = 2;
const EXIT_UNEXPECTED_OUTPUT: i32 = 3;

extern "C" fn on_fault(_signal: i32) {
    // SAFETY: `_exit` may be called from a signal handler and ends the process
    // at once.
    unsafe { _exit(EXIT_WROTE_TO_A_STRING) }
}

/// Runs in the forked child and returns its exit code.
fn encode_in_child(
    clone_control: bool,
    strings: &[IString],
    docs: &[IValue],
    string_pages: &[usize],
) -> i32 {
    // SAFETY: the handler only calls `_exit`.
    unsafe {
        signal(SIGSEGV, on_fault as usize);
        signal(SIGBUS, on_fault as usize);
    }
    for &page in string_pages {
        // SAFETY: `page` is the start of a mapping of this process that holds a
        // string, and nothing else lives on its first page.
        if unsafe { mprotect(page as *mut c_void, page_size(), PROT_READ) } != 0 {
            return EXIT_PROTECT_FAILED;
        }
    }
    if clone_control {
        std::mem::forget(strings[0].clone());
    }
    let mut encoder = StreamEncoder::new(StreamEncoderConfig {
        table_budget_bytes: 64 << 20,
    });
    let mut out = Vec::new();
    for doc in docs {
        out.clear();
        encoder.encode_into(&mut out, doc);
    }
    let stats = encoder.stats();
    if stats.defs == DOCS as u64 && stats.refs == DOCS as u64 {
        EXIT_OK
    } else {
        EXIT_UNEXPECTED_OUTPUT
    }
}

/// Forks, runs the child and returns its exit code, or `None` when a signal ended it.
fn run_child(
    clone_control: bool,
    strings: &[IString],
    docs: &[IValue],
    string_pages: &[usize],
) -> Option<i32> {
    // SAFETY: the child encodes, then leaves through `_exit` without running the
    // parent's test harness or any destructor, which would write to the strings.
    let pid = unsafe { fork() };
    assert!(pid >= 0, "fork failed");
    if pid == 0 {
        let code = encode_in_child(clone_control, strings, docs, string_pages);
        // SAFETY: `_exit` is always safe to call; it ends the child immediately.
        unsafe { _exit(code) };
    }
    let mut status = 0;
    // SAFETY: `status` is a valid pointer and `pid` is a child of this process.
    let waited = unsafe { waitpid(pid, &mut status, 0) };
    assert_eq!(waited, pid, "waitpid failed");
    (status & 0x7f == 0).then_some((status >> 8) & 0xff)
}

#[test]
fn an_encoder_never_writes_to_a_string() {
    // Distinct 20 KiB strings: each one has a mapping of its own.
    let strings: Vec<IString> = (0..DOCS)
        .map(|index| IString::from(format!("{index:08}{}", "x".repeat(STRING_BYTES - 8))))
        .collect();
    // Each string is the "a" entry of one document and the "b" entry of the
    // previous one, so it is defined once and referenced once.
    let docs: Vec<IValue> = (0..DOCS)
        .map(|index| {
            let mut object = IObject::new();
            object.insert("a", strings[index].clone()).unwrap();
            object
                .insert("b", strings[(index + 1) % DOCS].clone())
                .unwrap();
            IValue::from(object)
        })
        .collect();
    let string_pages: Vec<usize> = strings
        .iter()
        .map(|string| {
            let bytes = string.as_ptr() as usize;
            let page = bytes & !(page_size() - 1);
            // The header, 8 or 16 bytes, sits at the start of the mapping.
            assert!(bytes - page <= 16, "a string does not start its mapping");
            page
        })
        .collect();

    assert_eq!(
        run_child(false, &strings, &docs, &string_pages),
        Some(EXIT_OK),
        "the encoder wrote to a string"
    );
    // The control: holding handles updates owner counts, which the child must notice.
    assert_eq!(
        run_child(true, &strings, &docs, &string_pages),
        Some(EXIT_WROTE_TO_A_STRING),
        "the child did not notice a write to a read-only string"
    );
}
