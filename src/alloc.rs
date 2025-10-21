use std::error::Error;
use std::fmt;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct AllocError;

impl Error for AllocError {}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("memory allocation failed")
    }
}
