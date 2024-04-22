extern crate kesa;
use anyhow::{bail, Error, Result};

use image::GenericImageView;
use kesa::{
    backends::tch_backend::{self, TchModel},
    fileutils::get_config_from_name,
    image_utils,
    label::{Shape, YoloAnnotation},
    output::OutputFormat,
};

use kesa::output;
fn load_tch(input: &str, device: Option<tch::Device>) -> Result<TchModel, Error> {
    let cuda = device.unwrap_or(tch::Device::cuda_if_available());
    let loaded_model = TchModel::new(&input, 640, 640, cuda);
    for n in 0..4 {
       loaded_model.warmupfp16()?;
    } 

    let imgpath = "/media/hbpopos/penisf/275k_img/kesa_test/8s.jpeg";
    let _img2 = image::open(imgpath)?;
    let all_classes = vec![
        "10C", "10D", "10H", "10S", "2C", "2D", "2H", "2S", "3C", "3D", "3H", "3S", "4C", "4D",
        "4H", "4S", "5C", "5D", "5H", "5S", "6C", "6D", "6H", "6S", "7C", "7D", "7H", "7S", "8C",
        "8D", "8H", "8S", "9C", "9D", "9H", "9S", "AC", "AD", "AH", "AS", "JC", "JD", "JH", "JS",
        "KC", "KD", "KH", "KS", "QC", "QD", "QH", "QS",
    ];
    let _ac = all_classes.iter().map(|x| String::from(*x)).collect();
    let preproc_img = image_utils::preprocess_imagef16(&_img2, 640)?;
    let mut _pimg2 = tch::Tensor::try_from(preproc_img)?;
    println!("pimg2 {:?}", _pimg2);
    let test_inf = loaded_model.run_fp16(&_pimg2, 0.7, 0.6, "yolov9")?;
    println!(
        "testinf[1] to yolo: {:?}",
        test_inf[0].to_shape(&_ac, &_img2.dimensions(), &(640, 640))?
    );
    Ok(loaded_model)
}

pub fn main() -> Result<(), Error> {
    load_tch(
        "test/card_det_40k_640_final-converted_fp16.torchscript",
        None,
    )?;
    Ok(())
}
