use crate::kesa_utils::file_utils::{get_all_json, get_model_classes_from_yaml};
use crate::kesa_utils::kesa_error::KesaError;
use crate::yolo::yolo;
use std::str::FromStr;

pub struct LabelSettings {
    pub model: yolo::YOLO,
    pub batch_size: u16,
    /*damn bro if u have a batchsize of more than 65535
    and using this shitty software u must have some
    beefy ass pc u better pay me or
    hire someone to change lmaos
    */
}
