use arbitrary::Arbitrary;
use serde::{Deserialize, Serialize};

#[derive(Debug, Arbitrary, Serialize, Deserialize)]
pub enum JsonValue {
    #[serde(rename = "null")]
    Null,
    Bool(bool),
    Integer(u64),
    Float(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}