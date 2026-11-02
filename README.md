# ijson

[![CI Status](https://github.com/Diggsey/ijson/workflows/CI/badge.svg)](https://github.com/Diggsey/ijson/actions?query=workflow%3ACI)
[![Documentation](https://docs.rs/ijson/badge.svg)](https://docs.rs/ijson)
[![crates.io](https://img.shields.io/crates/v/ijson.svg)](https://crates.io/crates/ijson)
[![codecov](https://codecov.io/gh/Diggsey/ijson/branch/master/graph/badge.svg?token=XZ1UCUKSYB)](https://codecov.io/gh/Diggsey/ijson)

This crate offers a replacement for `serde-json`'s `Value` type, which is
significantly more memory efficient.

As a ballpark figure, it will typically use half as much memory as
`serde-json` when deserializing a value and the memory footprint of cloning
a value is more than 7x smaller.

## Memory savings

These graphs show memory savings as a function of JSON size (in bytes). The
JSON is randomly generated using the template in the `test_data` folder
using the javascript `dummyjson` tool.

![Peak memory usage when deserializing](graphs/graph00.png)
![Total allocations when deserializing](graphs/graph01.png)
![Memory overhead of cloning](graphs/graph02.png)
![Total allocations when cloning](graphs/graph03.png)

You can reproduce them yourself by installing `dummy-json` from NPM, and then
running these commands in the root directory:

```
cargo run --example generate --release
cargo run --example comparison --release > comparison.csv
```

The `comparison.xlsx` Excel file uses this CSV as a data-source to generate
the graphs.

## Usage

### `IValue`

The primary type exposed by this crate is the `IValue` type. It is guaranteed
to be pointer-sized and has a niche (so `Option<IValue>` is also guaranteed
to be pointer-sized).

Compared to `serde_json::Value` this type is a struct rather than an enum, as
this is necessary to achieve the important size reductions. This means that
you cannot directly `match` on an `IValue` to determine its type.

Instead, an `IValue` offers several ways to get at the inner type:

- Destructuring using `IValue::destructure[{_ref,_mut}]()`

  These methods return wrapper enums which you _can_ directly match on, so
  these methods are the most direct replacement for matching on a `Value`.

- Borrowing using `IValue::as_{array,object,string,number}[_mut]()`

  These methods return an `Option` of the corresponding reference if the
  type matches the one expected. These methods exist for the variants
  which are not `Copy`.

- Converting using `IValue::into_{array,object,string,number}()`

  These methods return a `Result` of the corresponding type (or the
  original `IValue` if the type is not the one expected). These methods
  also exist for the variants which are not `Copy`.

- Getting using `IValue::to_{bool,{i,u,f}{32,64}}[_lossy]}()`

  These methods return an `Option` of the corresponding type. These
  methods exist for types where the return value would be `Copy`.

You can also check the type of the inner value without specifically
accessing it using one of these methods:

- Checking using `IValue::is_{null,bool,number,string,array,object,true,false}()`

  These methods exist for all types.

- Getting the type with `IValue::type_()`

  This method returns the `ValueType` enum, which has a variant for each of the
  six JSON types.

### INumber

The `INumber` type represents a JSON number. It is decoupled from any specific
representation, and internally uses several. There is no way to determine the
internal representation: instead the caller is expected to convert the number
using one of the fallible `to_xxx` functions and handle the cases where the
number does not convert to the desired type.

Special floating point values (eg. NaN, Infinity, etc.) cannot be stored within
an `INumber`.

Whilst `INumber` does not consider `2.0` and `2` to be different numbers (ie.
they will compare equal) it does allow you to distinguish them using the
method `INumber::has_decimal_point()`. That said, calling `to_i32` on
`2.0` will succeed with the value `2`.

Currently `INumber` can store any number representable with an `f64`, `i64` or
`u64`. It is expected that in the future it will be further expanded to store
integers and possibly decimals to arbitrary precision, but that is not currently
the case.

Any number representable with an `i8` or a `u8` can be stored in an `INumber`
without a heap allocation (so JSON byte arrays are relatively efficient).
Integers up to 24 bits can be stored with a 4-byte heap allocation.

### IString

The `IString` type is an interned, immutable string, and is where this crate
gets its name.

Cloning an `IString` is cheap, and it can be easily converted from `&str` or
`String` types. Comparisons between `IString`s is a simple pointer
comparison.

The memory backing an `IString` is reference counted, so that unlike many
string interning libraries, memory is not leaked as new strings are interned.
Interning uses `DashSet`, an implementation of a concurrent hash-set, allowing
many strings to be interned concurrently without becoming a bottleneck.

Given the nature of `IString` it is better to intern a string once and reuse
it, rather than continually convert from `&str` to `IString`.

### IArray

The `IArray` type is similar to a `Vec<IValue>`. The primary difference is
that the length and capacity are stored _inside_ the heap allocation, so that
the `IArray` itself can be a single pointer.

### IObject

The `IObject` type is similar to a `HashMap<IString, IValue>`. As with the
`IArray`, the length and capacity are stored _inside_ the heap allocation.
In addition, `IObject`s preserve the insertion order of their elements, in
case that is important in the original JSON.

Removing from an `IObject` will disrupt the insertion order.

### String streams

`StreamEncoder` shares strings across the documents of one ordered stream. A
string with at least two owners and at least 8 bytes is
written inline once, wrapped in CBOR tag 6 (a definition), and every later
occurrence is written as tag 7 plus an ID (a reference). IDs are implicit: the
n-th definition of the stream is ID n, so `StreamDecoder` rebuilds the table by
decoding the documents in the order they were encoded. A reference costs 2, 3, 4
or 6 bytes for IDs below 24, 256, 65,536 and 2^32. Typed numeric arrays keep their
existing CBOR tags, and `encode`/`decode` and the compressed APIs keep their
standalone formats.

```rust
use ijson::{IValue, StreamDecoder, StreamEncoder, StreamEncoderConfig};

let mut encoder = StreamEncoder::new(StreamEncoderConfig {
    table_budget_bytes: 64 << 20,
});
let first: IValue = serde_json::from_str(r#"{"k":"a string stored in two documents"}"#).unwrap();
let second: IValue = serde_json::from_str(r#"{"k":"a string stored in two documents"}"#).unwrap();

let (mut first_bytes, mut second_bytes) = (Vec::new(), Vec::new());
encoder.encode_into(&mut first_bytes, &first); // defines the string
encoder.encode_into(&mut second_bytes, &second); // references it

let mut decoder = StreamDecoder::new();
assert_eq!(decoder.decode(&first_bytes, true).unwrap(), first);
assert_eq!(decoder.decode(&second_bytes, true).unwrap(), second);
```

The encoder recognizes a string by the address of its shared allocation, so it
neither hashes nor copies string bytes and only reads them. The caller must keep
all defined string allocations alive and at the same address until the stream
ends. The encoder holds no owned handles. A freed string whose address another
string of the same length takes over goes unnoticed, and the reader gets the old
text; only a string of another length at a registered address is noticed, which
poisons the encoder: that string and every later one is written inline. The table budget bounds the
encoder's memory: when one more definition
would exceed it, new strings stay inline while earlier ones keep being
referenced. If a document that defined strings is not written, call `poison()`;
every later document is then plain CBOR. `StreamEncoder::next_id` before a
document and `StreamDecoder::len` before reading it must agree, which lets a
framing layer detect skipped, repeated or reordered documents.

To reuse a full table, call `StreamEncoder::reset_if_needed()` between documents
that may define strings. It clears the table only after an eligible definition
missed the table budget, preserving its allocation. Signal that reset in your
framing and call `StreamDecoder::reset()` before decoding the next document
that introduces definitions. Inline documents can precede that marker; they do
not use the dictionary. Both dictionaries then assign IDs from 0 again.

With resets, use cumulative `stats().defs` for framing positions rather than the
current table length, and require a reset-bearing record to introduce a definition.
Otherwise, skipping a reset-only record could silently reuse old IDs. Statistics
stay cumulative across the entire stream. Reader resets release old dictionary handles while loaded
values keep their own. Neither reset API clears a poisoned or failed state.

### Standalone documents

A fresh `StreamEncoder` and a fresh `StreamDecoder` per document make that document stand
alone: ids start at 0, nothing else is needed to read it, and strings that repeat inside the
document are defined once. A string only counts as repeated here if it has two or more owners,
so a document that holds a shared string once also gets a definition that nothing references.
Write the plain bytes when the definitions do not pay: each definition costs 1 byte and
`stats().saved_bytes` is what the references saved, so keep the stream form only when
`saved_bytes > defs`.

```rust
use ijson::{IValue, StreamDecoder, StreamEncoder, StreamEncoderConfig};

let value: IValue = serde_json::from_str(
    r#"[{"a repeated object key": 1}, {"a repeated object key": 2}]"#,
).unwrap();
let mut bytes = Vec::new();
let mut encoder = StreamEncoder::new(StreamEncoderConfig { table_budget_bytes: 1 << 20 });
encoder.encode_into(&mut bytes, &value);
let stats = encoder.stats();
assert!(stats.saved_bytes > stats.defs);
assert_eq!(bytes.len() as u64, ijson::encode(&value).len() as u64 - stats.saved_bytes + stats.defs);
assert_eq!(StreamDecoder::new().decode(&bytes, true).unwrap(), value);
```

#### Other value representations

`StreamEncoder<D>` accepts any implementation of `StringDictionary`, defaulting
to `AddressDictionary`. `ContentDictionary` matches ordinary `&str` values by
content and owns copies, so sources can be freed after each document. Its budget
includes copied UTF-8 bytes, table storage and simultaneous growth allocations.
It admits all strings of at least 8 bytes, including strings that occur only once.

```rust
use ijson::{ContentDictionary, StreamDecoder, StreamEncoder, StreamEncoderConfig};

let mut encoder = StreamEncoder::with_dictionary(
    StreamEncoderConfig { table_budget_bytes: 4096 },
    ContentDictionary::default(),
);
let mut bytes = Vec::new();
encoder.encode_with(&mut bytes, |strings, out| {
    ciborium::into_writer(&strings.string("text from another value representation"), out)
}).unwrap();
assert_eq!(
    StreamDecoder::new().decode(&bytes, true).unwrap().as_string().unwrap().as_str(),
    "text from another value representation",
);
```

A custom Serde serializer wraps each participating string value and object key
with `strings.string(...)`. Wrapping does not register a definition; serialization
does. Serialize one complete CBOR document directly in wire order, without
pre-serializing, discarding or reordering encoded pieces. Propagate errors:
`encode_with` discards appended bytes and poisons the encoder on failure.
Unwrapped strings remain ordinary inline CBOR.

A custom dictionary supplies identity, eligibility, storage and allocation
accounting. Its input is a `StreamString`: text plus an optional interned allocation
identity and owner count. Ordinary string slices have no allocation identity.
A dictionary that borrows its strings, like `AddressDictionary`, needs the caller to keep the
source storage alive; one that owns copies, like `ContentDictionary`, does not.
Choose one dictionary for the stream. Pass a fresh dictionary, or ensure retained
allocations already fit the budget. `Box<dyn StringDictionary>` supports choosing
the implementation at initialization without runtime type checks.

All dictionaries use the same DEF/REF format and existing `StreamDecoder`.
The decoder reconstructs its own `IString` table from definitions; it does not
need the writer's cache or value representation.

## Technical details

### `IValue`

The six JSON types are broken down into those with a small fixed set of values:

- null
- bool

And those without:

- number
- string
- array
- object

Conveniently, this means we only need to distinguish between four different
heap allocated types (those without) and this can be done using only 2 bits.

We make sure our heap allocations have an alignment of at least 4 (which
is generally the case _anyway_) and this leaves us the two lower bits of
a pointer to store a "tag" value.

As an added bonus, the alignment of 4 means there are 3 constant pointer
values (other than the null pointer) which can never be returned from
`alloc`:

- 0x1
- 0x2
- 0x3

These three pointer values map neatly onto the fixed values `null`, `false`
and `true` respectively. And with that, we've covered all the possible JSON
types!

All that's left is to find a way to store numbers, strings, arrays and
objects behind a thin pointer.

### `INumber`

It's not uncommon to store byte arrays in JSON. If we need to a heap
allocation for every single byte in such an array it would be extremely
inefficient. Also, some numbers are more common than others (0, 1, -1).

As a result, we need a way to encode numbers more efficiently the smaller
they are, and ideally encode all possible byte values without a heap
allocation. But we only have a single pointer to work with, and we've
already used the tag bits!

The good news is that there just aren't that many byte values (256 to
be exact) and even if we extend the range to signed bytes too, it's
only 384. We can simply reserve a static array in our binary for these
small integers.

In practice we use a nice round 512-entry array storing values from
`-128` to `383` which more than covers the byte value range. When we
need to store one of these numbers we simply set our pointer to the
appropriate entry in the array, and skip any allocating or freeing.

### `IString`

As mentioned previously, strings are interned. As well as saving a ton
of memory when keys are repeated many times in arrays of objects, this
also makes it trivial to store a string with a single pointer, we just
set the pointer to the location of the interned string.

We also use a similar trick as for numbers to store the empty string
more cheaply: we just declare a static variable to be the empty string,
and use pointers to that.

### `IArray`

This works just like a `Vec`, but we reserve extra space at the beginning
of the allocation to store the length and capacity.

We again use the static variable optimization so that the empty `Vec` does
not require an allocation.

### `IObject`

Same idea as the `IArray` and the same static variable optimization.

Internally we actually store two arrays in the allocation: the first is
a simple array of `IValue`s and the second is the hash-table itself.
The hash table just stored indices into the first array.

This simplifies the hash table implementation whilst also allowing us to
preserve the insertion order and makes iteration very cheap (since we
don't need to skip over empty entries).

New values are always pushed onto the end of the array before their
index is inserted into the hash table. Removed values are first swapped
to the end of the array (see `Vec::swap_remove`) so that removals are
still constant time.

The hash values are not stored, since the keys (`IString`s) are interned
and so the hash function can be a very fast operation that only looks
at the pointer value.
