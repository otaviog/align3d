#[derive(Debug)]
pub struct Error {
    pub message: String
}

impl From<opencv::Error> for Error {
    fn from(value: opencv::Error) -> Self {
        Self {
            message: value.message.clone()
        }
    }
}