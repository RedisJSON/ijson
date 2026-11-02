use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use ijson::{IObject, IString, IValue};

// A string from before a cache reset and the same text interned afterwards are two allocations.
// They must still compare, order and hash as the same string.

fn init_cache() {
    // Another test in this binary may have initialized it already.
    let _ = ijson::init_shared_string_cache(true);
}

fn hash_of<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn retained_object_keys_still_match_after_cache_reset() {
    init_cache();
    let key = IString::from("retained object key");
    let mut object = IObject::new();
    object.insert(key.clone(), 42).unwrap();

    ijson::reinit_shared_string_cache();

    let replacement = IString::from("retained object key");
    assert_eq!(object.get(&replacement), Some(&IValue::from(42)));
}

#[test]
fn a_string_decoded_after_cache_reset_equals_the_retained_one() {
    init_cache();
    let retained = IValue::from("retained string across encode and decode");
    let bytes = ijson::encode(&retained);

    ijson::reinit_shared_string_cache();

    let decoded = ijson::decode(&bytes).unwrap();
    assert_eq!(decoded, retained);
    assert_eq!(
        decoded.partial_cmp(&retained),
        Some(std::cmp::Ordering::Equal)
    );
    assert_eq!(hash_of(&decoded), hash_of(&retained));
}

#[test]
fn equal_strings_from_either_side_of_a_cache_reset_hash_and_order_the_same() {
    init_cache();
    let before = IString::from("string interned on both sides of a reset");

    ijson::reinit_shared_string_cache();

    let after = IString::from(before.as_str());
    assert_eq!(before, after);
    assert_eq!(before.cmp(&after), std::cmp::Ordering::Equal);
    assert_eq!(hash_of(&before), hash_of(&after));

    let values: HashSet<IValue> = [IValue::from(before)].into_iter().collect();
    assert!(values.contains(&IValue::from(after)));
}
