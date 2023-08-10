use crate::convert_label::label_structs::{GenericAnnotation, GenericLabelPoints, LabelMeLabel};
use anyhow;

use serde_derive::{Deserialize, Serialize};
use serde_json::{Result};
use serde_yaml::{self};
use std::collections::HashMap;



use std::{fmt::Write};
use std::{ffi::OsStr, fs, path::PathBuf};

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelDetails {
    pub names: Vec<String>,
    pub input_size: Vec<String>, // TODO: add input_size to python export scripts
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
