extern crate kesa;

use kesa::backends::tch_backend::{self, TchModel};
fn load_tch(input: &str, device: Option<tch::Device>) -> TchModel {
    let cuda = device.unwrap_or(tch::Device::cuda_if_available());
    let loaded_model = TchModel::new(
        &input, 640, 640, cuda
    );
    for n in 0..3 {
        loaded_model.warmup().unwrap();
    }
    loaded_model
}


pub fn main() {
    load_tch("test/cardDetv1.556_640x640.torchscript", None);  
}
