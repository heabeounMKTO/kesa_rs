use tch::{self, vision::image};
use tch::Tensor;
use tch::kind;
use tch::IValue;
use std::io;
use anyhow::{Result, Error};

use crate::label::YoloAnnotation;

pub struct TchModel {
    model: tch::CModule,
    device: tch::Device,
    w: i64,
    h: i64 
}


impl TchModel{
    pub fn new(weights: &str, w: i64, h: i64, device: tch::Device) -> TchModel {
        let mut model = tch::CModule::load_on_device(weights, device).unwrap();
        model.set_eval();
        TchModel {
            model: model,
            device: device,
            w: w,
            h: h
        }
    }


    pub fn run(&self, image: &tch::Tensor, conf_thresh: f32, iou_thresh: f64) -> Result<Vec<YoloAnnotation>, Error>    {
        todo!()
    } 
}
