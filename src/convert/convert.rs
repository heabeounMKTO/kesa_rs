use crate::kesa_utils::file_utils::{get_all_json, get_model_classes_from_yaml};

pub enum ConvertTarget {
    Yolo,
    Pascal,
    Coco,
}

pub struct ConvertSettings {
    target: ConvertTarget,
    classes: Vec<String>,
    input_folder: &'static str,
}
impl ConvertSettings {
    pub fn new(target: ConvertTarget, classes: Vec<String>, input_folder: &'static str) -> ConvertSettings {
        ConvertSettings { target, classes, input_folder }
    }
}

pub fn convert(settings: ConvertSettings) {
    let all_jsons = get_all_json(settings.input_folder);
}
