use crate::convert_label::label_structs::YoloLabel;
use crate::kesa_utils::file_utils::{self, get_all_json, read_shapes_from_json, LabelExportFolderDetails};
use crate::kesa_utils::kesa_error::KesaError;

use conv::ValueFrom;
use indicatif::ProgressBar;
use std::fs::File;
use std::io::Write;
use std::str::FromStr;

use std::collections::HashMap;

#[derive(Clone, Debug, Copy)]
pub enum ConvertTarget {
    Yolo,
    Pascal,
    Coco,
    LabelMe,
}

impl FromStr for ConvertTarget {
    type Err = KesaError;
    fn from_str(input_type: &str) -> Result<Self, Self::Err> {
        match input_type.to_lowercase().as_str() {
            "yolo" => Ok(ConvertTarget::Yolo),
            "coco" => Ok(ConvertTarget::Coco),
            "pascal" => Ok(ConvertTarget::Pascal),
            "labelme" => Ok(ConvertTarget::LabelMe),
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
            let yolo_labels: YoloLabel = annotations.convert2yolo(settings.classes.to_owned());
            txtfile
                .write_all(
                    format!(
                        "{:?} {:?} {:?} {:?} {:?}",
                        &yolo_labels.label_index.to_owned(),
                        &yolo_labels.x.to_owned(),
                        &yolo_labels.y.to_owned(),
                        &yolo_labels.w.to_owned(),
                        &yolo_labels.h.to_owned()
                    )
                    .as_bytes(),
                )
                .expect("Error In writing file to txt");
            txtfile
                .write_all("\n".as_bytes())
                .expect("Error in writing space!");
        }
    }
    let ayylmao: LabelExportFolderDetails = file_utils::create_export_folder(Some(String::from("export"))).unwrap();
    println!("LabelExportFolderDetails : {:?}", ayylmao.get_train_image_and_label_path());
    progress.finish_with_message("conversion done!");
    println!("Conversion done in {:#?}", start.elapsed());
}
