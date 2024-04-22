use crate::label::Xyxy;
use crate::label::YoloAnnotation;
use anyhow::bail;
use anyhow::{Error, Result};
use conv::TryInto;
use sorted_list::Tuples;
use std::io;
use tch::kind;
use tch::IValue;
use tch::Tensor;
use tch::{self, vision::image};



pub struct TchModel {
    model: tch::CModule,
    device: tch::Device,
    w: i64,
    h: i64,
}

impl TchModel {
    pub fn new(weights: &str, w: i64, h: i64, device: tch::Device) -> TchModel {
        let mut model = tch::CModule::load_on_device(weights, device).unwrap();
        model.set_eval();
        TchModel {
            model: model,
            device: device,
            w: w,
            h: h,
        }
    }
    pub fn warmup(&self) -> Result<(), Error> {
        let mut img: Tensor = Tensor::zeros(&[3, 640, 640], kind::INT64_CUDA);
        img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .g_div_scalar(255.0);
        let pred: tch::IValue = self.model.forward_is(&[tch::IValue::Tensor(img)])?;
        let _toTensorList = Vec::<tch::Tensor>::try_from(pred)?;
        println!(
            "warmup tensor list: {:?}",
            self.non_max_suppression(&_toTensorList[0].get(0), 0.8,0.7)
        );
        Ok(())
    }


    /// for testing yolov5 models 
    pub fn warmupv5(&self) -> Result<(), Error> {
        let mut img: Tensor = Tensor::zeros(&[3, 640, 640], kind::INT64_CUDA);
        img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .g_div_scalar(255.0);
        let pred = self.model.forward_is(&[tch::IValue::Tensor(img)])?;
        let pred_T = tch::Tensor::try_from(pred)?;
        println!("pred_t, {:?}", pred_T);
        // todo!()
        Ok(())
    }

    pub fn run(&self, image: &tch::Tensor, conf_thresh: f32, iou_thresh: f64) -> Result<(), Error> {
        let mut img = tch::vision::image::resize(&image, self.w, self.h)?;
        img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .g_div_scalar(255.0);
        let pred = self.model.forward_is(&[tch::IValue::Tensor(img)])?;
        let pred_T = tch::Tensor::try_from(pred)?;
        println!("pred_t, {:?}", pred_T);
        // todo!()
        Ok(())
    }
    
    fn iou(&self, b1: &YoloAnnotation, b2: &YoloAnnotation) -> f32 {
        let b1_area = (b1.w - b1.xmin + 1.) * (b1.h - b1.ymin + 1.);
        let b2_area = (b2.w - b2.xmin + 1.) * (b2.h - b2.ymin + 1.);
        let i_xmin = b1.xmin.max(b2.xmin);
        let i_w = b1.w.min(b2.w);
        let i_ymin = b1.ymin.max(b2.ymin);
        let i_h = b1.h.min(b2.h);
        let i_area = (i_w - i_xmin + 1.).max(0.) * (i_h - i_ymin + 1.).max(0.);
        i_area / (b1_area + b2_area - i_area) 
    }

       fn non_max_suppression(
        &self,
        pred: &Tensor,
        conf_thresh: f64,
        iou_thresh: f64,
    ) -> Vec<YoloAnnotation> {
        let (npreds, pred_size) = pred.size2().unwrap();
        let nclasses = (pred_size - 5) as usize;
        let mut bboxes: Vec<Vec<YoloAnnotation>> = (0..nclasses).map(|_| vec![]).collect();

        for index in 0..npreds {
            let pred = Vec::<f64>::try_from(pred.get(index)).expect("cannot convert");
            let confidence = pred[4];

            if confidence > conf_thresh {
                let mut class_index = 0;

                for i in 0..nclasses {
                    if pred[5 + i] > pred[5 + class_index] {
                        class_index = i;
                    }
                }
                if pred[5 + class_index] > 0. {

                    let bbox = YoloAnnotation {
                        xmin: (pred[0] - pred[2] / 2. ) as f32,
                        ymin: (pred[1] - pred[3] / 2.) as f32,
                        w: (pred[0] + pred[2] / 2.) as f32,
                        h: (pred[1] + pred[3] / 2.) as f32,
                        confidence: confidence as f32,
                        class: class_index as i64,
                    };
                    bboxes[class_index].push(bbox);

                }

            }
        }
        for bboxes_for_class in bboxes.iter_mut() {
            bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
            let mut current_index = 0;
            for index in 0..bboxes_for_class.len() {
                let mut drop = false;
                for prev_index in 0..current_index {
                    let iou = self.iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);

                    if iou > iou_thresh as f32 {
                        drop = true;
                        break;
                    }
                }
                if !drop {
                    bboxes_for_class.swap(current_index, index);
                    current_index += 1;
                }
            }
            bboxes_for_class.truncate(current_index);
        }

        let mut result = vec![];

        for bboxes_for_class in bboxes.iter() {
            for bbox in bboxes_for_class.iter() {
                result.push(*bbox);
            }
        }
        println!("Result: {:?}", &result);
        result
    } 
}
