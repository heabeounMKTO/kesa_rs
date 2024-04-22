use crate::label::Embeddings;
use anyhow::{Error, Result};
use std::ffi::OsStr;
use std::path::PathBuf;

use super::candle_backend::CandleModel;
use super::onnx_backend::OnnxModel;
use super::tch_backend::TchModel;

#[derive(Debug, Clone)]
pub enum ComputeBackendType {
    OnnxModel,
    CandleModel,
    TchModel,
}

pub trait InferenceModel: Sized {
    fn run(&self, image: image::DynamicImage) -> Result<Embeddings, Error>;
    fn warmup(&self);
}

/// infers model type from filename
pub fn get_backend(input_path: &str) -> Result<ComputeBackendType, Error> {
    let _input2path = PathBuf::from(&input_path).to_owned();
    let extension = _input2path.extension().unwrap().to_owned();
    let model_type = match extension.to_str() {
        Some("onnx") => ComputeBackendType::OnnxModel,
        Some("torchscript") => ComputeBackendType::TchModel,
        Some("safetensors") => ComputeBackendType::CandleModel,
        _ => panic!("Unkown/Unsuppourted Model type {:#?}", _input2path),
    };
    Ok(model_type)
}
