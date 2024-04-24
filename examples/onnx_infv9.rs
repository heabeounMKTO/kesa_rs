extern crate kesa;
use kesa::backends::onnx_backend::{init_onnx_backend, load_onnx_model};
use anyhow::{Result, Error};
use clap::Parser;
use kesa::backends::compute_backends::InferenceModel;
use kesa::fileutils::get_all_images;
use kesa::fileutils::open_image;
use ort::Tensor;

#[derive(Parser, Debug)]
struct CliArguments {
    #[arg(long, required=true)]
    weights: String,

    #[arg(long, required=true)]
    folder: String
}


fn main() -> Result<(), Error> {
    let args = CliArguments::parse();
    let all_imgs = get_all_images(&args.folder);


    let _init = init_onnx_backend()?;
    let load_model = load_onnx_model(&args.weights, all_imgs[0].to_owned().to_str().unwrap(), false, None)?;
    let orig_img = open_image(&all_imgs[0])?;
    let mut _r = load_model.run(orig_img)?;
    _r.data.swap_axes(2,1);
    let _tnsr = Tensor::try_from(_r.data)?; 
    _tnsr.view();
    println!("LOADED MODEL : {:#?}",_tnsr.shape());
    Ok(())
}
