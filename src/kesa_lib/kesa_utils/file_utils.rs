use crate::convert_label::label_structs::{GenericAnnotation, GenericLabelPoints, LabelMeLabel};
use anyhow;

use owo_colors::colors::xterm::FuchsiaPink;
use serde_derive::{Deserialize, Serialize};
use serde_json::Result;
use serde_yaml::{self};
use std::collections::HashMap;

use std::default;
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

#[derive(Debug, Clone, Copy)]
pub struct LabelPortions {
    pub train: f32,
    pub valid: f32,
    pub test: f32
}

impl LabelPortions {
    pub fn new(train_set: f32, valid_set: f32, test_set: f32) -> LabelPortions{
        LabelPortions {
            train: train_set,
            valid: valid_set,
            test: test_set
        } 
    }
}


impl LabelExportFolderDetails {
    pub fn get_train_image_and_label_path(&self) -> Vec<String> {
        let img_pth = format!("{}{}", self.train_path, "/images");
        let label_path = format!("{}{}", self.train_path, "/labels");
        let ayylmao: Vec<String> = vec![img_pth, label_path];
        ayylmao
    }
    pub fn get_valid_image_and_label_path(&self) -> Vec<String> {
        let img_pth = format!("{}{}", self.valid_path, "/images");
        let label_path = format!("{}{}", self.valid_path, "/labels");
        let ayylmao: Vec<String> = vec![img_pth, label_path];
        ayylmao
    }
    pub fn get_test_image_and_label_path(&self) -> Vec<String> {
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
    // not used, use find_filetype instead
    // edit aight nvm we will have a separate thing for txt lmao
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
    // println!("{:?}", extension);
    Ok(PathBuf::from(input_json_path))
}
pub fn get_all_txt(input: &str) -> anyhow::Result<Vec<PathBuf>> {
    // not used, use find_filetype instead
    // edit aight nvm we will have a separate thing for txt lmao
    let mut result = vec![];
    for path in fs::read_dir(input)? {
        let path = path?.path();
        if let Some("txt") = path.extension().and_then(OsStr::to_str) {
            result.push(path.to_owned())
        }
    }
    Ok(result)
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

// MOVE UR SHIT BRAH
pub fn move_labels_to_export_folder(
    input_folder: &str,
    output_folder: &str,
    export_portions: LabelPortions
) {
    // will add handling of files according to convert format later,
    // currently it's just for YOLO format
    println!(
        "Moving labels from {:?} to {:?}",
        &input_folder, &output_folder
    );

    let mut all_txt = get_all_txt(&input_folder);

    let train_ratio:f32 = {
        ((all_txt.as_ref().unwrap().len() as f32)*&export_portions.train).floor()
    };
    
    let valid_ratio:f32 = {
        ((all_txt.as_ref().unwrap().len() as f32)*&export_portions.valid).floor()
    };
    
    let test_ratio:f32 = {
        ((all_txt.as_ref().unwrap().len() as f32)*&export_portions.test).floor()
    };
    
    dbg!(all_txt.as_ref().unwrap().len() as f32,train_ratio, valid_ratio, test_ratio); 
    let train_batch = all_txt.split_off(train_ratio as i64);
    let valid_batch = all_txt.split_off(valid_ratio as i64);
    let test_batch = all_txt.split_off(valid_ratio as i64);
    dbg!(train_batch, valid_batch, test_batch);    
    for txtfiles in all_txt.unwrap() {
        let stem = txtfiles.as_path().file_stem().unwrap();
        
    }
}
