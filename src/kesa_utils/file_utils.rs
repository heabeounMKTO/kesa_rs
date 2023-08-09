use std::{fs::File, io, path, path::PathBuf, ffi::OsStr};
use serde_json::{Result,Value, json};
use std::collections::HashMap;
use serde_derive::{Deserialize, Serialize};
use serde_yaml::{self};
use anyhow;


#[derive(Debug, Serialize, Deserialize)]
pub struct ModelDetails{
    pub names: Vec<String>,
}

pub fn get_classes_from_yaml(input: &str) -> anyhow::Result<Vec<String>>{
    let f = std::fs::File::open(input)?;
    let model_deets: ModelDetails = serde_yaml::from_reader(f)?;
    Ok(model_deets.names)
}