#[derive(Debug, Clone)]
pub enum KesaErrorType {
    UnknownTypeError,
    FileNotFoundError,
}

#[derive(Debug, Clone)]
pub struct KesaError {
    pub message: String,
    pub errortype: KesaErrorType,
}

impl KesaError {
    pub fn KesaUnknownTypeError(msg: String) -> KesaError {
        println!("KesaUnknownTypeError: {}", &msg);
        KesaError {
            message: msg,
            errortype: KesaErrorType::UnknownTypeError,
        }
    }

    pub fn FileNotFoundError(msg: String) -> KesaError {
        println!("KesaFileNotFoundError: {}", &msg);
        KesaError {
            message: msg,
            errortype: KesaErrorType::FileNotFoundError,
        }
    }
}
