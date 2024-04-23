use crate::label::Xyxy;
use crate::label::{CoordinateType, YoloBbox};
use anyhow::bail;
use anyhow::{Error, Result};
use conv::TryInto;
use half::f16;
use sorted_list::Tuples;
use std::io;
use tch::kind;
use tch::IValue;
use tch::Kind;
use tch::Tensor;
use tch::{self, vision::image};

#[derive(Debug)]
pub struct TchModel {
    pub model: tch::CModule,
    pub device: tch::Device,
    pub w: i64,
    pub h: i64,
}

impl TchModel {
    pub fn new(weights: &str, w: i64, h: i64, device: tch::Device) -> TchModel {
        let mut model = tch::CModule::load_on_device(weights, device).unwrap();
        // model.set_eval();
        TchModel {
            model: model,
            device: device,
            w: w,
            h: h,
        }
    }

    /// forward pass with zeroes
    pub fn warmup_gpu_fp16(&self) -> Result<(), Error> {
        println!("[info]::torch_backend: running gpu fp16 warmup");

        // half cuda cos there aint any in tch
        // nhom kherng yerng ai >:(
        let HALF_CUDA: (Kind, tch::Device) = (Kind::Half, tch::Device::Cuda(0));
        let mut img: Tensor = Tensor::zeros([3, 640, 640], HALF_CUDA);
        img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Half)
            .to_device(self.device);
        let t1 = std::time::Instant::now();
        let pred: tch::IValue = self.model.forward_is(&[tch::IValue::Tensor(img)])?;
        println!(
            "[info]::torch_backend: warmup_gpu_fp16 time: {:?}",
            t1.elapsed()
        );
        Ok(())
    }

    /// fp32 warmup on gpu
    pub fn warmup_gpu(&self) -> Result<(), Error> {
        println!("[info]::torch_backend: running gpu warmup");
        let FLOAT_CUDA: (Kind, tch::Device) = (Kind::Float, tch::Device::Cuda(0));
        let mut img: Tensor = Tensor::zeros([3, 640, 640], FLOAT_CUDA);
        img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.device);
        let t1 = std::time::Instant::now();
        let pred: tch::IValue = self.model.forward_is(&[tch::IValue::Tensor(img)])?;
        println!("[info]::torch_backend: warmup_gpu time: {:?}", t1.elapsed());
        Ok(())
    }

    /// forward pass with zeroes , for CPU inference
    pub fn warmup(&self) -> Result<(), Error> {
        println!("[info]::torch_backend: running cpu warmup");
        let mut img: Tensor = Tensor::zeros(&[3, 640, 640], kind::FLOAT_CPU);
        img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .g_div_scalar(255.0);
        let t1 = std::time::Instant::now();
        let pred: tch::IValue = self.model.forward_is(&[tch::IValue::Tensor(img)])?;
        println!("[info]::torch_backend: warmup time: {:?}", t1.elapsed());
        Ok(())
    }

    /// [testing only] for testing yolov5 models
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
        Ok(())
    }

    /// runs inference in 16bit presicion (fp16)
    pub fn run_fp16(
        &self,
        image: &tch::Tensor,
        conf_thresh: f32,
        iou_thresh: f32,
        yolo_version: &str,
    ) -> Result<Vec<YoloBbox>, Error> {
        let img = image.to_kind(tch::Kind::Half).to_device(self.device);
        match yolo_version {
            "yolov9" => {
                let pred = self
                    .model
                    .forward_ts(&[img])
                    .unwrap()
                    .to_device(self.device);
                let _transposed_o = pred.transpose(2, 1);
                // let t1 = std::time::Instant::now();
                let results = self.nms_yolov9(&_transposed_o.get(0), conf_thresh, iou_thresh);
                // println!("inference time: {:?}", t1.elapsed());
                results
            }
            _ => {
                todo!()
            }
        }
    }

    /// runs inference in 32bit (fp32) precision :)
    pub fn run(
        &self,
        image: &tch::Tensor,
        conf_thresh: f32,
        iou_thresh: f32,
        yolo_version: &str,
    ) -> Result<Vec<YoloBbox>, Error> {
        let img = image.to_kind(tch::Kind::Float).to_device(self.device);
        match yolo_version {
            "yolov9" => {
                let pred = self
                    .model
                    .forward_ts(&[img])
                    .unwrap()
                    .to_device(self.device);
                // turns [1,56, 8400] to [1, 8400, 56]
                let _transposed_o = pred.transpose(2, 1);
                self.nms_yolov9(&_transposed_o.get(0), conf_thresh, iou_thresh)
            }
            _ => {
                let pred = self
                    .model
                    .forward_ts(&[img])
                    .unwrap()
                    .to_device(self.device);
                let _transposed_o = pred.transpose(2, 1);
                println!("transpoed: {:?}", _transposed_o);
                self.non_max_suppression(&_transposed_o.get(0), conf_thresh, iou_thresh)
            }
        }
    }

    fn iou(&self, b1: &YoloBbox, b2: &YoloBbox) -> f32 {
        let b1_area = (b1.xyxy.x2 - b1.xyxy.x1 + 1.) * (b1.xyxy.y2 - b1.xyxy.y1 + 1.);
        let b2_area = (b2.xyxy.x2 - b2.xyxy.x1 + 1.) * (b2.xyxy.y2 - b2.xyxy.y1 + 1.);
        let i_xmin = b1.xyxy.x1.max(b2.xyxy.x1);
        let i_w = b1.xyxy.x2.min(b2.xyxy.x2);
        let i_ymin = b1.xyxy.y1.max(b2.xyxy.y1);
        let i_h = b1.xyxy.y2.min(b2.xyxy.y2);
        let i_area = (i_w - i_xmin + 1.).max(0.) * (i_h - i_ymin + 1.).max(0.);
        i_area / (b1_area + b2_area - i_area)
    }

    /// NMS for yolov9 models :)
    fn nms_yolov9(
        &self,
        pred: &Tensor,
        conf_thresh: f32,
        iou_thresh: f32,
    ) -> Result<Vec<YoloBbox>, Error> {
        // yolov9 transposed output [1, 8400, 56]
        let (npreds, preds_size) = pred.size2().unwrap();
        let nclasses = (preds_size - 4) as usize;
        let mut bboxes: Vec<Vec<YoloBbox>> = (0..nclasses).map(|_| vec![]).collect();
        for index in 0..npreds {
            let pred = Vec::<f32>::try_from(pred.get(index))?;
            let confidence = *pred[5..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
            if confidence > conf_thresh {
                let mut class_index = 0;
                for i in 0..nclasses {
                    if pred[4 + i] > pred[4 + class_index] {
                        class_index = i;
                    }
                }

                // NOTE: this outputs from the image inference size i.e 640
                // so to get back coordinates for original image
                // you can normalize these coordinates [x,y]/640 then multiply
                // it by its dimension i.e [x, y]*[imagewidth, imageheight]
                if pred[4 + class_index] > 0. {
                    let xyxy: Xyxy = Xyxy {
                        coordinate_type: CoordinateType::Screen,
                        x1: (pred[0] - pred[2] / 2.0),
                        y1: (pred[1] - pred[3] / 2.0),
                        x2: (pred[0] + pred[2] / 2.0),
                        y2: (pred[0] + pred[3] / 2.0),
                    };
                    let bbox: YoloBbox = YoloBbox {
                        class: class_index as i64,
                        xyxy: xyxy,
                        confidence: confidence,
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
                    if iou > iou_thresh {
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
        Ok(result)
    }

    fn non_max_suppression(
        &self,
        pred: &Tensor,
        conf_thresh: f32,
        iou_thresh: f32,
    ) -> Result<Vec<YoloBbox>, Error> {
        let (npreds, pred_size) = pred.size2().unwrap();
        let nclasses = (pred_size - 5) as usize;
        let mut bboxes: Vec<Vec<YoloBbox>> = (0..nclasses).map(|_| vec![]).collect();
        for index in 0..npreds {
            let pred = Vec::<f32>::try_from(pred.get(index)).expect("cannot convert");
            let confidence = pred[4];
            println!("confidence {:?}", &confidence);
            if confidence > conf_thresh {
                let mut class_index = 0;

                for i in 0..nclasses {
                    if pred[5 + i] > pred[5 + class_index] {
                        class_index = i;
                    }
                }
                println!("Class index {:?}", &class_index);
                if pred[5 + class_index] > 0. {
                    let xyxy: Xyxy = Xyxy {
                        coordinate_type: CoordinateType::Screen,
                        x1: (pred[0] - pred[2] / 2.0),
                        y1: (pred[1] - pred[3] / 2.0),
                        x2: (pred[0] + pred[2] / 2.0),
                        y2: (pred[0] + pred[3] / 2.0),
                    };
                    let bbox: YoloBbox = YoloBbox {
                        class: class_index as i64,
                        xyxy: xyxy,
                        confidence: confidence,
                    };
                    bboxes[class_index].push(bbox);
                }
            }
        }
        // println!("DEBUG bboxes , {:?}", &bboxes);
        for bboxes_for_class in bboxes.iter_mut() {
            bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
            let mut current_index = 0;
            for index in 0..bboxes_for_class.len() {
                let mut drop = false;
                for prev_index in 0..current_index {
                    let iou = self.iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);

                    if iou > iou_thresh {
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
        Ok(result)
    }
}
