extern crate kesa;
use anyhow::{bail, Error, Result};

use kesa::backends::tch_backend::{self, TchModel};
fn load_tch(input: &str, device: Option<tch::Device>) -> Result<TchModel, Error> {
    let cuda = device.unwrap_or(tch::Device::cuda_if_available());
    let loaded_model = TchModel::new(&input, 640, 640, cuda);
    // for n in 0..3 {
    //     loaded_model.warmup()?;
    // }
    let image = tch::vision::image::load("/home/hbpopos/Downloads/1.jpg")?;
    let test_inf = loaded_model.run(&image, 0.7, 0.6, "yolov5");
    println!("test_inf: {:#?}", test_inf);
    Ok(loaded_model)
}

pub fn main() -> Result<(), Error> {
    load_tch("test/cardDetv1.0_yolov5.torchscript", None)?;
    Ok(())
}
