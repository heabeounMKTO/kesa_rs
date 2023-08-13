use crate::convert_label::label_structs::{GenericAnnotation, GenericLabelPoints, LabelMeLabel};
use anyhow;

use serde_derive::{Deserialize, Serialize};
use serde_json::Result;
use serde_yaml::{self};
use std::collections::HashMap;

use std::fmt::Write;
use std::path::Path;
use std::{ffi::OsStr, fs, path::PathBuf};

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelDetails {
    pub names: Vec<String>,
    pub input_size: Vec<String>, // TODO: add input_size to python export scripts
}


#[derive(Debug)]
pub struct LabelExportFolderDetails {
    pub train_path: String,
    pub valid_path: String,
    pub test_path: String,
    pub data_yml_path: String,
}

impl LabelExportFolderDetails {
    pub fn get_train_image_and_label_path(&self) -> Vec<String>{
        let img_pth = format!("{}{}", self.train_path, "/images");
        let label_path = format!("{}{}", self.train_path, "/labels");
        let ayylmao: Vec<String> = vec![img_pth, label_path];
        ayylmao
    }
    pub fn get_valid_image_and_label_path(&self) -> Vec<String>{
        let img_pth = format!("{}{}", self.valid_path, "/images");
        let label_path = format!("{}{}", self.valid_path, "/labels");
        let ayylmao: Vec<String> = vec![img_pth, label_path];
        ayylmao
    }
    pub fn get_test_image_and_label_path(&self) -> Vec<String>{
        let img_pth = format!("{}{}", self.test_path, "/images");
        let label_path = format!("{}{}", self.test_path, "/labels");
        let ayylmao: Vec<String> = vec![img_pth, label_path];
        ayylmao
    }
}

pub fn create_export_folder(
    export_path: Option<String>,
) -> anyhow::Result<LabelExportFolderDetails> {
    let export_settings: LabelExportFolderDetails = match export_path {
        Some(export_path) => {
            let train_path = format!("{}/train", &export_path);
            let valid_path = format!("{}/valid", &export_path);
            let test_path = format!("{}/test", &export_path);
            let data_yml_path = format!("{}/data.yaml", &export_path);
            // fs::create_dir_all(&test_path).expect("cannot create train_path");
            // fs::create_dir_all(&valid_path).expect("cannot create valid_path");
            // fs::create_dir_all(&test_path).expect("cannot create test path");

            fs::create_dir_all(format!("{}/train/images", &export_path))
                .expect("cannot create train_path/image");
            fs::create_dir_all(format!("{}/valid/images", &export_path))
                .expect("cannot create valid_path/image");
            fs::create_dir_all(format!("{}/valid/images", &export_path))
                .expect("cannot create test_path/image");

            fs::create_dir_all(format!("{}/train/labels", &export_path))
                .expect("cannot create train_path/labels");
            fs::create_dir_all(format!("{}/valid/labels", &export_path))
                .expect("cannot create valid_path/labels");
            fs::create_dir_all(format!("{}/valid/labels", &export_path))
                .expect("cannot create test_path/labels");

            fs::File::create(&data_yml_path).expect("cannot create data.yaml");
            LabelExportFolderDetails {
                train_path,
                valid_path,
                test_path,
                data_yml_path,
            }
        }
        None => {
            let train_path = String::from("export/train");
            let valid_path = String::from("export/valid");
            let test_path = String::from("export/test");
            let data_yml_path = String::from("export/data.yaml");
            fs::create_dir_all(&test_path).expect("cannot create train_path");
            fs::create_dir_all(&valid_path).expect("cannot create valid_path");
            fs::create_dir_all(&test_path).expect("cannot create test path");
            fs::File::create(&data_yml_path).expect("cannot create data.yaml");
            LabelExportFolderDetails {
                train_path,
                valid_path,
                test_path,
                data_yml_path,
            }
        }
    };
    Ok(export_settings)
}

pub fn class_vec2hash(input_vec: Vec<String>) -> Result<HashMap<String, i32>> {
    let mut result: HashMap<String, i32> = HashMap::new();
    for (idx, label_name) in input_vec.iter().enumerate() {
        result.insert(String::from(label_name), idx as i32);
    }
    Ok(result)
}

pub fn get_model_classes_from_yaml(input: &str) -> anyhow::Result<HashMap<String, i32>> {
    let f = std::fs::File::open(input)?;
    let model_deets: ModelDetails = serde_yaml::from_reader(f)?;
    Ok(class_vec2hash(model_deets.names).unwrap())
}

pub fn get_model_config_from_yaml(input: &str) -> anyhow::Result<ModelDetails> {
    let f = std::fs::File::open(input)?;
    let model_config: ModelDetails = serde_yaml::from_reader(f)?;
    Ok(model_config)
}

pub fn get_all_json(input: &str) -> anyhow::Result<Vec<PathBuf>> {
    let mut result = vec![];
    for path in fs::read_dir(input)? {
        let path = path?.path();
        if let Some("json") = path.extension().and_then(OsStr::to_str) {
            result.push(path.to_owned())
        }
    }
    Ok(result)
}

pub fn get_extension_from_str(input_str: &str) -> Option<&str> {
    Path::new(input_str).extension().and_then(OsStr::to_str)
}

pub fn get_image_from_json_path(input_json_path: &str) -> anyhow::Result<PathBuf> {
    let extension = get_extension_from_str(input_json_path).unwrap();
    println!("{:?}", extension);
    Ok(PathBuf::from(input_json_path))
}

pub fn read_shapes_from_json(input_json: &str) -> anyhow::Result<Vec<GenericAnnotation>> {
    let mut result = vec![];
    let contents =
        fs::read_to_string(input_json).expect(&format!("Couldn't find file: {:?}", &input_json));
    let readed_json: LabelMeLabel = serde_json::from_str(&contents)
        .expect(&format!("Couldn't read content in file: {:?}", &contents));

    for shape in readed_json.shapes.to_owned() {
        let (label, image_width, image_height, image_path, x1y1, x2y2): (
            String,
            i32,
            i32,
            String,
            GenericLabelPoints,
            GenericLabelPoints,
        );

        label = shape.label.to_owned();
        image_path = readed_json.imagePath.to_owned();
        image_height = readed_json.image_height().to_owned();
        image_width = readed_json.image_width().to_owned();
        x1y1 = GenericLabelPoints::new(shape.points[0].x.to_owned(), shape.points[0].y.to_owned());

        x2y2 = GenericLabelPoints::new(shape.points[1].x.to_owned(), shape.points[1].y.to_owned());
        let anno =
            GenericAnnotation::new(&label, image_width, image_height, image_path, x1y1, x2y2);
        result.push(anno.to_owned());
    }
    Ok(result)
}
