extern crate kesa;
use anyhow::{bail, Error, Result};

use kesa::backends::tch_backend::{self, TchModel};
fn load_tch(input: &str, device: Option<tch::Device>) -> Result<TchModel, Error> {
    let cuda = device.unwrap_or(tch::Device::cuda_if_available());
    let loaded_model = TchModel::new(&input, 640, 640, cuda);
    // for n in 0..3 {
    //     loaded_model.warmup()?;
    // }
    let image = tch::vision::image::load("/media/hbpopos/penisf/275k_img/kesa_test/1f62eafa-de5f-492c-8b13-1be9971a4fa2.jpeg")?;
    let test_inf = loaded_model.run(&image, 0.1, 0.6, "yolov9");
    println!("test_inf: {:#?}", test_inf);
    Ok(loaded_model)
}

pub fn main() -> Result<(), Error> {
    load_tch("test/card_det_40k_640_final.torchscript", None)?;
    Ok(())
}
