/// Error types for the library.

/// Main error type for the library.
#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
}

/// Error kinds for the library.
#[derive(Debug)]
pub enum ErrorKind {
    /// Used when the user pass a logical  invalid parameter to a function.
    InvalidParameter(String),
}

impl Error {
    /// Create a error with the kind `InvalidParameter`.
    /// # Arguments
    /// * `msg` - The error message.
    pub fn invalid_parameter<T: ToString>(msg: T) -> Self {
        Self {
            kind: ErrorKind::InvalidParameter(msg.to_string()),
        }
    }
}
