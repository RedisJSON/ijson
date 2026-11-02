use ijson::IString;

// Dropping the last handle of a string from before a reset must not evict its replacement.
#[test]
fn a_string_kept_across_a_cache_reset_is_not_the_one_interned_afterwards() {
    ijson::init_shared_string_cache(true).unwrap();
    let before_reset = IString::from("string retained across cache reset");
    ijson::reinit_shared_string_cache();
    let after_reset = IString::from(before_reset.as_str());
    assert_ne!(before_reset.as_ptr(), after_reset.as_ptr());
    drop(before_reset);
    assert_eq!(
        IString::from(after_reset.as_str()).as_ptr(),
        after_reset.as_ptr()
    );
}
