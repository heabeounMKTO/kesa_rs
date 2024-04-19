use anyhow::{Error, Result};
use sorted_list::Tuples;
use std::io;
use std::result;
use tch::kind;
use tch::IValue;
use tch::Tensor;
use tch::{self, vision::image};
use crate::label::YoloAnnotation;
use anyhow::{bail};
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
        let mut pred = self.model.forward_is(&[IValue::from(img)])?;
        
        let (tensor1, tensor2): (Vec<Tensor>,Vec<Tensor>) = match &pred {
            IValue::Tuple(ivalues) => match &ivalues {
                [<IValue::Tensor(t1)> <IValue::Tensor(t2)>] => (t1.shallow_clone(), t2.shallow_clone()),  
                _ => bail!("unexpected ivalue output {:?}", ivalues),
            }
            _ => bail!("unexpected tch output"),
        };

        Ok(())

    }

    pub fn run(&self, image: &tch::Tensor, conf_thresh: f32, iou_thresh: f64) -> Result<(), Error> {
        let mut img = tch::vision::image::resize(&image, self.w, self.h)?;
        img = img
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .g_div_scalar(255.0);
        let pred = self.model.forward_is(&[IValue::from(img)]).unwrap();
        let pred_T = tch::Tensor::try_from(pred)?;
        println!("pred_t, {:?}", pred_T);
        // todo!()
        Ok(())
    }
}
