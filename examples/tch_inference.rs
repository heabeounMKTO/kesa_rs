extern crate kesa;
use anyhow::{bail, Error, Result};

use image::GenericImageView;
use kesa::{
    backends::tch_backend::{self, TchModel},
    fileutils::get_config_from_name,
    image_utils,
    label::{LabelmeAnnotation, Shape, YoloAnnotation},
    output::OutputFormat,
};

use kesa::output;
fn load_tch(input: &str, device: Option<tch::Device>) -> Result<TchModel, Error> {
    let cuda = device.unwrap_or(tch::Device::cuda_if_available());
    let loaded_model = TchModel::new(&input, 640, 640, cuda);
    for n in 0..4 {
        loaded_model.warmup_gpu()?;
    }
    let imgpath = "/media/hbpopos/penisf/275k_img/kesa_test/8s.jpeg";
    let _img2 = image::open(imgpath)?;
    let all_classes = vec![
        "10C", "10D", "10H", "10S", "2C", "2D", "2H", "2S", "3C", "3D", "3H", "3S", "4C", "4D",
        "4H", "4S", "5C", "5D", "5H", "5S", "6C", "6D", "6H", "6S", "7C", "7D", "7H", "7S", "8C",
        "8D", "8H", "8S", "9C", "9D", "9H", "9S", "AC", "AD", "AH", "AS", "JC", "JD", "JH", "JS",
        "KC", "KD", "KH", "KS", "QC", "QD", "QH", "QS",
    ];
    let _ac: Vec<String> = all_classes.iter().map(|x| String::from(*x)).collect();
    let preproc_img = image_utils::preprocess_imagef16(&_img2, 640)?;
    let mut _pimg2 = tch::Tensor::try_from(preproc_img)?;

    let mut _img3 = tch::vision::image::load_and_resize(imgpath, 640, 640)?;
    // _img3 = _img3.unsqueeze(0).to_kind(tch::Kind::Float).to_device(loaded_model.device).g_div_scalar(255.0); 

    println!("pimg2 {:?}", _img2.dimensions());
    let mut test_inf = loaded_model._run_fp16(&_img3, 0.7, 0.6)?;
    println!("test_inf: {:?}", &test_inf);
    let uhhh = test_inf[0]
        .to_normalized(&(640, 640))
        .to_screen(&(690, 1035))
        .to_shape(&_ac, &(690, 1035))?;
    let _2vec: Vec<Shape> = vec![uhhh];
    let _lm: LabelmeAnnotation = LabelmeAnnotation::from_shape_vec(imgpath, &_img2, &_2vec)?;
    println!(
        "labelme anno w:{:?} h:{:?}",
        _lm.imageWidth, _lm.imageHeight
    );
    Ok(loaded_model)
}

pub fn main() -> Result<(), Error> {
    load_tch(
        "test/card_det_40k_640_final-converted.torchscript",
        None,
    )?;
    Ok(())
}
