use std::io;

#[derive(Debug)]
pub enum LoadError {
    IO(io::Error),
    ParseError(String),
}

impl From<io::Error> for LoadError {
    fn from(err: io::Error) -> Self {
        LoadError::IO(err)
    }
}
