extern crate kesa;
use anyhow::{bail, Error, Result};

use kesa::{backends::tch_backend::{self, TchModel}, image_utils};
fn load_tch(input: &str, device: Option<tch::Device>) -> Result<TchModel, Error> {
    let cuda = device.unwrap_or(tch::Device::cuda_if_available());
    let loaded_model = TchModel::new(&input, 640, 640, cuda);
    
    // for n in 0..3 {
    //     loaded_model.warmup()?;
    // }
    let imgpath = "/media/hbpopos/penisf/275k_img/kesa_test/8s.jpeg";
    let _img2 = image::open(imgpath)?;
    let preproc_img = image_utils::preprocess_imagef32(&_img2, 640)?;
    let mut _pimg2 = tch::Tensor::try_from(preproc_img)?;
    println!("pimg2 {:?}", _pimg2);
    let test_inf = loaded_model.run(&_pimg2, 0.7, 0.6, "yolov9");
    println!("test_inf: {:#?}", test_inf);
    Ok(loaded_model)
}

pub fn main() -> Result<(), Error> {
    load_tch("test/card_det_40k_640_final-converted.torchscript", None)?;
    Ok(())
}
