extern crate kesa;
use anyhow::{Result, Error, bail};

use kesa::backends::tch_backend::{self, TchModel};
fn load_tch(input: &str, device: Option<tch::Device>) -> Result<TchModel, Error> {
    let cuda = device.unwrap_or(tch::Device::cuda_if_available());
    let loaded_model = TchModel::new(
        &input, 640, 640, cuda
    );
    for n in 0..3 {
        loaded_model.warmup()?;
    }
    Ok(loaded_model)
}


pub fn main() -> Result<(), Error> {
    load_tch("test/cardDetv1.556_640x640.torchscript", None)?;  
    Ok(())
}
