#![no_main]

use arbitrary::Arbitrary;
use ijson::{IObject, IValue};
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

// Small key space (u8 -> "k{n}") so op sequences repeatedly hit the same keys,
// reliably crossing the small-object table threshold in both directions and
// forcing hash collisions / Robin-Hood displacement on the u32 table.
#[derive(Arbitrary, Debug)]
enum Op {
    Insert(u8, u64),
    Remove(u8),
    Get(u8),
}

fuzz_target!(|ops: Vec<Op>| {
    let mut obj = IObject::new();
    let mut oracle: HashMap<u8, u64> = HashMap::new();

    for op in ops {
        match op {
            Op::Insert(id, v) => {
                let k = format!("k{id}");
                // insert() only errors on OOM / 2^30 overflow; neither is reachable here.
                let prev = obj
                    .insert(k.as_str(), IValue::from(v))
                    .unwrap()
                    .and_then(|old| old.to_u64());
                assert_eq!(prev, oracle.insert(id, v));
            }
            Op::Remove(id) => {
                let k = format!("k{id}");
                let removed = obj.remove(k.as_str()).and_then(|v| v.to_u64());
                assert_eq!(removed, oracle.remove(&id));
            }
            Op::Get(id) => {
                let k = format!("k{id}");
                let got = obj.get(k.as_str()).and_then(|v| v.to_u64());
                assert_eq!(got, oracle.get(&id).copied());
            }
        }
        assert_eq!(obj.len(), oracle.len());
    }

    // Final agreement: every oracle key present with the right value.
    for (id, v) in &oracle {
        let k = format!("k{id}");
        assert_eq!(obj.get(k.as_str()).and_then(|x| x.to_u64()), Some(*v));
    }
});
