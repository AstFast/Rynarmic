pub mod a32;
pub mod thumb;

use core::fmt;

/// Why decoding failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The encoding is illegal, or lies in an instruction space this stage does not cover.
    Undefined(u32),
    /// Legal but not implemented yet. Carries the encoding and a reason, so the gap can be
    /// filled in later without re-deriving it.
    Unsupported { encoding: u32, reason: &'static str },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Undefined(enc) => write!(f, "undefined instruction: {enc:#010x}"),
            DecodeError::Unsupported { encoding, reason } => {
                write!(f, "unsupported instruction {encoding:#010x}: {reason}")
            }
        }
    }
}

impl std::error::Error for DecodeError {}
