use crate::kesa_utils::file_utils::{get_all_json, get_model_classes_from_yaml};
use crate::kesa_utils::kesa_error::KesaError;
use std::str::FromStr;

#[derive(Clone, Debug, Copy)]
pub enum ConvertTarget {
    Yolo,
    Pascal,
    Coco,
}

impl FromStr for ConvertTarget {
    type Err = KesaError;
    fn from_str(input_type: &str) -> Result<Self, Self::Err> {
        match input_type.to_lowercase().as_str() {
            "yolo" => Ok(ConvertTarget::Yolo),
            "coco" => Ok(ConvertTarget::Coco),
            "pascal" => Ok(ConvertTarget::Pascal),
            _ => Err(KesaError::KesaUnknownTypeError(String::from(format!(
                "kesa does not suppourt '{}' for conversion",
                input_type
            )))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConvertSettings {
    target: ConvertTarget,
    classes: Vec<String>,
    input_folder: String,
}
impl ConvertSettings {
    pub fn new(
        target: ConvertTarget,
        classes: Vec<String>,
        input_folder: String,
    ) -> ConvertSettings {
        ConvertSettings {
            target,
            classes,
            input_folder,
        }
    }
}

pub fn convert(settings: ConvertSettings) {
    println!(
        "target: {:?}\nclasses: {:?}\ni_f: {:?} ",
        settings.target, settings.classes, settings.input_folder
    );
    let all_jsons = get_all_json(&settings.input_folder.as_str());
}
