use serde_derive::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io;
use tch::kind;
use tch::IValue;
use tch::Tensor;
use tch::{self, vision::image};

#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub xmin: f64,
    pub ymin: f64,
    pub xmax: f64,
    pub ymax: f64,
    pub conf: f64,
    pub cls: usize,
}

pub struct YOLO {
    model: tch::CModule,
    device: tch::Device,
    h: i64,
    w: i64,
}

impl BBox {
    pub fn conf(&self) -> f64 {
        return self.conf;
    }
    pub fn cls(&self) -> i64 {
        return self.cls as i64;
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Detections {
    pub confidence: f64,
    pub cardClass: String,
    pub card: usize,
}

impl Detections {
    pub fn new(confidence: f64, cardClass: String, card: usize) -> Detections {
        return Detections {
            confidence: confidence,
            cardClass: cardClass,
            card: card,
        };
    }
}

impl YOLO {
    pub fn new(weights: &str, h: i64, w: i64, device: tch::Device) -> YOLO {
        let mut model = tch::CModule::load_on_device(weights, device).unwrap();
        model.set_eval();
        YOLO {
            model,
            h,
            w,
            device,
        }
    }

    pub fn predict_batch(
        &self,
        images: &Vec<&tch::Tensor>,
        conf_thresh: f64,
        iou_thresh: f64,
    ) -> Vec<Vec<BBox>> {
        let img: Vec<tch::Tensor> = images
            .into_iter()
            .map(|x| tch::vision::image::resize(&x, self.w, self.h).unwrap())
            .collect();
        let img = tch::Tensor::stack(&img, 0);
        let img = img
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .g_div_scalar(255.);
        let pred = self
            .model
            .forward_ts(&[img])
            .unwrap()
            .to_device(self.device);
        let (amount, _, _) = pred.size3().unwrap();
        let results = (0..amount)
            .map(|x| self.non_max_suppression(&pred.get(x), conf_thresh, iou_thresh))
            .collect();
        results
    }

    pub fn predict_ivalue(
        &self,
        image: &tch::Tensor,
        conf_thresh: f64,
        iou_thresh: f64,
    ) -> Vec<BBox> {
        let img = tch::vision::image::resize(&image, self.w, self.h).unwrap();
        let img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .g_div_scalar(255.0);
        let pred = self.model.forward_is(&[IValue::from(img)]).unwrap();

        let pred_T = tch::Tensor::try_from(pred).unwrap();
        return self.non_max_suppression(&pred_T.get(0), conf_thresh, iou_thresh);
    }

    pub fn predict(&self, image: &tch::Tensor, conf_thresh: f64, iou_thresh: f64) -> Vec<BBox> {
        let img = tch::vision::image::resize(&image, self.w, self.h).unwrap();
        let img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .g_div_scalar(255.);
        let start = std::time::Instant::now();
        let pred = self
            .model
            .forward_ts(&[img])
            .unwrap()
            .to_device(self.device);
        println!("Inference Time: {:?}", start.elapsed());
        let result = self.non_max_suppression(&pred.get(0), conf_thresh, iou_thresh);
        result
    }

    pub fn warmup(&self) {
        let img: Tensor = Tensor::zeros(&[3, 640, 640], kind::INT64_CUDA);
        let img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .g_div_scalar(255.);
        let start = std::time::Instant::now();
        let pred = self
            .model
            .forward_ts(&[img])
            .unwrap()
            .to_device(self.device);
        println!("Inference Time: {:?}", start.elapsed());
        let result = self.non_max_suppression(&pred.get(0), 0.1, 0.1);
    }

    fn iou(&self, b1: &BBox, b2: &BBox) -> f64 {
        let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
        let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
        let i_xmin = b1.xmin.max(b2.xmin);
        let i_xmax = b1.xmax.min(b2.xmax);
        let i_ymin = b1.ymin.max(b2.ymin);
        let i_ymax = b1.ymax.min(b2.ymax);
        let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
        i_area / (b1_area + b2_area - i_area)
    }

    fn non_max_suppression(&self, pred: &Tensor, conf_thresh: f64, iou_thresh: f64) -> Vec<BBox> {
        let (npreds, pred_size) = pred.size2().unwrap();
        let nclasses = (pred_size - 5) as usize;
        let mut bboxes: Vec<Vec<BBox>> = (0..nclasses).map(|_| vec![]).collect();

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
                    let bbox = BBox {
                        xmin: pred[0] - pred[2] / 2.,
                        ymin: pred[1] - pred[3] / 2.,
                        xmax: pred[0] + pred[2] / 2.,
                        ymax: pred[1] + pred[3] / 2.,
                        conf: confidence,
                        cls: class_index,
                    };
                    bboxes[class_index].push(bbox);
                }
            }
        }

        for bboxes_for_class in bboxes.iter_mut() {
            bboxes_for_class.sort_by(|b1, b2| b2.conf.partial_cmp(&b1.conf).unwrap());
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

        return result;
    }
}
