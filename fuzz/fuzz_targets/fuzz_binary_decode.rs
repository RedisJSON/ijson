#![no_main]

use ijson::binary::decode;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    println!("data: {:?}", data);
    let _ = decode(data);
});
