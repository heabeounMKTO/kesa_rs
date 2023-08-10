use crate::convert_label::label_structs::YoloLabel;
use crate::kesa_utils::file_utils::{
    get_all_json, get_model_classes_from_yaml, read_shapes_from_json,
};
use crate::kesa_utils::kesa_error::KesaError;
use conv::ValueFrom;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use std::fs::File;
use std::str::FromStr;
use std::{cmp::min, fmt::Write};
use std::collections::HashMap;

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
                "\nkesa does not suppourt target '{}' for conversion\n",
                input_type
            )))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConvertSettings {
    pub target: ConvertTarget,
    pub classes: HashMap<String, i32>, //NOTE: reads into HashMap for less complex indexing nonsense
    pub input_folder: String,
}
impl ConvertSettings {
    pub fn new(
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

pub fn convert(settings: ConvertSettings) {
    println!(
        "conversion target: {:#?}\nclasses: {:#?}\nfolder: {:#?} ",
        &settings.target,
        &settings.classes.len(),
        &settings.input_folder
    );
    let all_jsons = get_all_json(&settings.input_folder.as_str());
    let start = std::time::Instant::now();
    let total = u64::value_from(all_jsons.as_ref().unwrap().len());
    let progress = ProgressBar::new(total.unwrap());
    for mut json_path in all_jsons.unwrap() {
        progress.inc(1);
        let shapes = read_shapes_from_json(json_path.as_path().to_str().unwrap());
        json_path.set_extension("txt");
        let mut txtfile = File::create(&json_path).expect("failed in creating file");
        for annotations in &shapes.unwrap() {
            let penis: YoloLabel = annotations.convert2yolo(settings.classes.to_owned());
        }
    }
    progress.finish_with_message("conversion done!");
    println!("Conversion done in {:#?}", start.elapsed());
}
