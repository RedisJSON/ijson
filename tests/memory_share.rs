use ijson::IValue;

#[test]
fn proportional_memory_across_documents() {
    let text = "proportional allocation accounting test string";
    let first = IValue::from(text);
    let allocation = first.mem_allocated();
    let mut docs: Vec<IValue> = (0..99).map(|_| IValue::from(text)).collect();
    docs.push(first);
    assert!(docs[0].mem_allocated() < 1.0);
    assert!((docs.iter().map(IValue::mem_allocated).sum::<f64>() - allocation).abs() < 1e-9);
    let one = docs.pop().unwrap();
    drop(docs);
    assert_eq!(one.mem_allocated(), allocation);
    drop(one);

    // A field name and nested values share one allocation across documents.
    let json = format!(r#"{{"{text}":["{text}","{text}"]}}"#);
    let doc: IValue = serde_json::from_str(&json).unwrap();
    let original = doc.mem_allocated();
    let other = IValue::from(text);
    assert!((doc.mem_allocated() + other.mem_allocated() - original).abs() < 1e-9);
    assert!((other.mem_allocated() - allocation / 4.0).abs() < 1e-9);
    drop(other);
    assert!((doc.mem_allocated() - original).abs() < 1e-9);

    for json in ["null", "true", "[]", "{}", r#""short""#] {
        let value: IValue = serde_json::from_str(json).unwrap();
        assert_eq!(value.mem_allocated(), 0.0);
    }
}
