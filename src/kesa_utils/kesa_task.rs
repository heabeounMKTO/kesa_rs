use crate::convert_label::convert::{ConvertSettings, ConvertTarget};

use std::collections::HashMap;
use std::str::FromStr;

use super::kesa_error::KesaError;

#[derive(Debug, Clone)]
pub enum KesaTaskType {
    KesaConvert,
    KesaLabel,
    KesaAugment,
}

#[derive(Debug, Clone)]
pub struct KesaConvert {
    settings: ConvertSettings,
}

impl KesaConvert {
    pub fn new_convert_setting(
        target: ConvertTarget,
        classes: HashMap<String, i32>,
        input_folder: String,
    ) -> ConvertSettings {
        ConvertSettings {
            target,
            classes,
            input_folder,
        }
    }
}

// TODO:
#[derive(Debug, Clone)]
pub struct KesaLabel {
    settings: String,
}

// TODO:
#[derive(Debug, Clone)]
pub struct KesaAugment {
    settings: String,
}

#[derive(Debug, Clone)]
pub struct KesaTask {
    task_type: KesaTaskType,
}

impl FromStr for KesaTaskType {
    type Err = KesaError;
    fn from_str(task_type: &str) -> Result<Self, Self::Err> {
        match task_type.to_lowercase().as_str() {
            "convert" => Ok(KesaTaskType::KesaConvert),
            "label" => Ok(KesaTaskType::KesaLabel),
            "augment" => Ok(KesaTaskType::KesaAugment),
            _ => Err(KesaError::KesaUnknownTypeError(String::from(format!(
                "\nkesa does not suppourt task type '{}'",
                task_type
            )))),
        }
    }
}
