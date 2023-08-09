use std::{fs, io, path, path::PathBuf, ffi::OsStr, env::consts::OS};
use serde_json::{Result,Value, json};
use std::collections::HashMap;
use serde_derive::{Deserialize, Serialize};
use serde_yaml::{self};
use anyhow;

use crate::convert::label_structs::{GenericAnnotation, LabelMeLabel, GenericLabelPoints};


#[derive(Debug, Serialize, Deserialize)]
pub struct ModelDetails{
    pub names: Vec<String>,
}

pub fn get_model_classes_from_yaml(input: &str) -> anyhow::Result<Vec<String>>{
    let f = std::fs::File::open(input)?;
    let model_deets: ModelDetails = serde_yaml::from_reader(f)?;
    Ok(model_deets.names)
}


pub fn get_all_json(input: &str) -> anyhow::Result<Vec<PathBuf>>{
    let mut result = vec![];
    for path in fs::read_dir(input)? {
        let path = path?.path();
        if let Some("json") = path.extension()
                              .and_then(OsStr::to_str){
            result.push(path.to_owned())                        
        }
    }
    Ok(result)
}

pub fn read_shapes_from_json(input_json: &str) 
    -> anyhow::Result<Vec<GenericAnnotation>> {
        let mut result = vec![];
        let contents = fs::read_to_string(input_json)
                        .expect(&format!("Couldn't find file {:?}", &input_json));
        let readed_json: LabelMeLabel = serde_json::from_str(&contents)
                        .expect(&format!("Couldn't read {:?}", &contents));
        
        for shape in readed_json.shapes.to_owned(){
            let (mut label,
                 mut image_width,
                 mut image_height,
                 mut image_path,
                 mut x1y1,
                 mut x2y2) : (String, i32, i32, String, GenericLabelPoints, GenericLabelPoints);

            label = shape.label.to_owned();
            image_path = readed_json.imagePath.to_owned();
            image_height = shape.image_height().to_owned();
        }
        
        Ok(result)
    }