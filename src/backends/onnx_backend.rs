use crate::fileutils::get_config_from_name;
use crate::image_utils;
use crate::label::Embeddings;
use crate::model::DatasetInfo;
use anyhow::{Error, Result};
use image::DynamicImage;
use ndarray::{array, s, Array, ArrayBase, Axis, CowArray, Dim, IxDyn, OwnedRepr};
use ort::{
    inputs, CPUExecutionProvider, CUDAExecutionProvider, GraphOptimizationLevel, Session,
    SessionOutputs, Value,
};
use rand::Rng;
use spinners::{Spinner, Spinners};
use std::io::Read;
use std::path::{Path, PathBuf};

use super::compute_backends::InferenceModel;

/// onnx model instance for inference (loads a mf model once)
#[derive(Debug)]
pub struct OnnxModel {
    pub model_details: DatasetInfo,
    pub model: ort::Session,
    pub is_fp16: bool,
}

impl OnnxModel {
    pub fn new(
        model_details: DatasetInfo,
        model: ort::Session,
        is_fp16: bool,
    ) -> Result<OnnxModel, Error> {
        Ok(OnnxModel {
            model_details,
            model,
            is_fp16,
        })
    }
}

impl InferenceModel for OnnxModel {
    /// go on , do a forward pass
    fn run(&self, input_image: image::DynamicImage) -> Result<Embeddings, Error> {
        if self.is_fp16 {
            todo!()
        } else {
            let _input_img = image_utils::preprocess_imagef32(&input_image, 640)?;
            let _inference: Result<SessionOutputs, ort::Error> =
                self.model.run(inputs!["images" => _input_img.view()]?);
            let _embeddings: Embeddings = match _inference {
                Ok(result) => {
                    // TODO: change "output" => "output0" for yolov9
                    // or maybe just export with the output being output
                    let _arr = result["output"]
                        .extract_tensor::<f32>()?
                        .view()
                        .t()
                        .into_owned();
                    let embeddings = Embeddings::new(_arr.t().to_owned());
                    embeddings
                }
                // returns a vec with a single numer on error :|
                Err(e) => {
                    println!("cannot find any detections! {}", e);
                    let mut rnd = rand::thread_rng();
                    let random_num = vec![rnd.gen::<u8>() as usize];
                    Embeddings::new(Array::zeros(IxDyn(&random_num)))
                }
            };
            Ok(_embeddings)
        }
    }
    /// when i dont want to self.run() whenever i run load_onnx_model() :|
    fn warmup(&self) {
        let mut _dummy_input: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> =
            Array::ones((1, 3, 640, 640));
    }
}

/// *[DEPRECATED]* lmao (blud really lasted a day)
/// just use OnnxModel instead
/// i was a fool
pub struct OnnxInference<'a> {
    onnx: &'a OnnxModel,
    input_image: image::DynamicImage,
    is_fp16: bool,
}

impl<'a> OnnxInference<'a> {
    pub fn new(onnx: &OnnxModel, input: image::DynamicImage) -> Result<OnnxInference, Error> {
        Ok(OnnxInference {
            onnx,
            is_fp16: onnx.is_fp16.to_owned(),
            input_image: input,
        })
    }

    pub fn run_inference(&self) -> Result<Embeddings, Error> {
        if self.is_fp16 {
            todo!()
        } else {
            // TODO: dynamically get image size from config or filename?! , 640 for now amen.
            let _input_img = image_utils::preprocess_imagef32(&self.input_image, 640)?;
            let _inference: Result<SessionOutputs, ort::Error> =
                self.onnx.model.run(inputs!["images" => _input_img.view()]?);
            let _embeddings: Embeddings = match _inference {
                Ok(result) => {
                    let _arr = result["output"]
                        .extract_tensor::<f32>()?
                        .view()
                        .t()
                        .into_owned();
                    let embeddings = Embeddings::new(_arr.t().to_owned());
                    embeddings
                }
                // returns a vec with a single numer on error :|
                Err(e) => {
                    println!("cannot find any detections! {}", e);
                    let mut rnd = rand::thread_rng();
                    let random_num = vec![rnd.gen::<u8>() as usize];
                    Embeddings::new(Array::zeros(IxDyn(&random_num)))
                }
            };

            // transposed to [2,7], so it's
            // [idk?? (0.0) , x1, y1, x2, y2, class, confidence]
            Ok(_embeddings)
        }
    }
}
/// creates a onnx env
pub fn init_onnx_backend() -> Result<(), anyhow::Error> {
    ort::init()
        .with_execution_providers([
            CUDAExecutionProvider::default().with_device_id(0).build(),
            CPUExecutionProvider::default().build(),
        ])
        .commit()?;
    Ok(())
}
/// deprecated, just use run() in OnnxModel
pub fn run_warmup(
    onnx_model: &OnnxModel,
    img: image::DynamicImage,
    fp16: bool,
) -> Result<Embeddings, Error> {
    let detection: OnnxInference = OnnxInference::new(onnx_model, img)?;
    let results: Embeddings = detection.run_inference()?;
    Ok(results)
}

pub fn load_onnx_model(
    model_path: &str,
    image_path: &str,
    load_fp16: bool,
    config_path: Option<&str>,
) -> Result<OnnxModel, Error> {
    // panic if we cant load the model
    // cos what is the point of cannot load and continue?
    // why are we here? just to suffer?
    let model: ort::Session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file(&model_path)
        .unwrap();
    let model_yaml_config_path = get_config_from_name(&config_path, &model_path)
        .expect("Cannot Find model Configuration file");

    let model_details =
        serde_yaml::from_reader(std::fs::File::open(&model_yaml_config_path)?).unwrap();
    // again, we must panic if something happens to model loading
    // phob lok nis ber load model ort jenh
    // nhom sok chet ort mean phob lok
    let mut spinna = Spinner::new(
        Spinners::Dots12,
        format!("Loading Model {:?}", &model_path).into(),
    );
    let loaded_model: OnnxModel = OnnxModel::new(model_details, model, false).unwrap();
    let mut _dummy_input: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> =
        Array::ones((1, 3, 640, 640));
    let original_img = image::open(Path::new(image_path)).unwrap();
    println!("\nRunning Warmup");
    // runs a forward pass on a random image from the folder
    let _ = &loaded_model.run(original_img);
    spinna.stop_with_symbol("✅");
    // println!("loaded_model {:#?}", &loaded_model);
    Ok(loaded_model)
}
