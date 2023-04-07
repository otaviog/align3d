/// Main error type for the library.
#[derive(Debug)]
pub enum A3dError {
    /// Used when the user pass a logical  invalid parameter to a function.
    InvalidParameter(String),
    Io(std::io::Error),
    Parser(String),
    Assertion(String),
}

impl std::fmt::Display for A3dError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            A3dError::Io(err) => write!(f, "IO error: {}", err),
            A3dError::Parser(err) => write!(f, "Parser error: {}", err),
            A3dError::InvalidParameter(err) => write!(f, "Parameter error: {}", err),
            A3dError::Assertion(err) => write!(f, "Assertion err,or: {}", err)
        }
    }
}

impl A3dError {
    /// Create a error with the kind `InvalidParameter`.
    /// # Arguments
    /// * `msg` - The error message.
    pub fn invalid_parameter<T: ToString>(msg: T) -> Self {
        A3dError::InvalidParameter(msg.to_string())
    }
}

impl std::error::Error for A3dError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            A3dError::Io(err) => Some(err),
            A3dError::Parser(_) => None,
            A3dError::InvalidParameter(_) => None,
            A3dError::Assertion(_) => None
        }
    }
}
