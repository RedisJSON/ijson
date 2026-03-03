use arbitrary::Arbitrary;
use serde::{Deserialize, Serialize};

#[derive(Debug, Arbitrary, Serialize, Deserialize)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Integer(u64),
    Float(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn to_json_string(&self) -> String {
        match self {
            JsonValue::Null => "null".to_string(),
            JsonValue::Bool(b) => b.to_string(),
            JsonValue::Integer(n) => n.to_string(),
            JsonValue::Float(n) => n.to_string(),
            JsonValue::Str(s) => {
                format!(r#""{}"#, s.replace('\\', r"\\").replace('"', r#"\""#))
            }
            JsonValue::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_json_string()).collect();
                format!("[{}]", items.join(","))
            }
            JsonValue::Object(obj) => {
                let items: Vec<String> = obj
                    .iter()
                    .map(|(k, v)| {
                        format!(
                            r#""{}":{}"#,
                            k.replace('\\', r"\\").replace('"', r#"\""#),
                            v.to_json_string()
                        )
                    })
                    .collect();
                format!("{{{}}}", items.join(","))
            }
        }
    }
}
