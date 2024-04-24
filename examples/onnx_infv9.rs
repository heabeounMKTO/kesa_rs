extern crate kesa;
use kesa::backends::compute_backends::ComputeBackendType;
use kesa::backends::onnx_backend::{init_onnx_backend, load_onnx_model};
use anyhow::{Result, Error};
use clap::{ArgAction, Parser,Args };
use kesa::fileutils::get_all_images;


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
    let load_model = load_onnx_model(&args.weights, all_imgs[0].to_owned().to_str().unwrap(), false, None);
    println!("LOADED MODEL : {:#?}", load_model);
    Ok(())
}
