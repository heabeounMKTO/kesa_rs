use crate::label::{read_labels_from_file, YoloAnnotation};
use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_yaml::{self, Value};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatasetInfo {
    pub names: Vec<String>,
    pub nc: i64,
    pub train: String,
    pub val: String,
    pub test: String,
}

// creates a data.yaml strucc
impl DatasetInfo {
    pub fn new(
        export_options: &ExportFolderOptions,
        all_classes: &Vec<String>,
    ) -> Result<DatasetInfo, Error> {
        Ok(DatasetInfo {
            names: all_classes.to_owned(),
            nc: all_classes.len().to_owned() as i64,
            train: export_options.train_img.to_owned(),
            val: export_options.val_img.to_owned(),
            test: export_options.test_img.to_owned(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExportFolderOptions {
    pub train_img: String,
    pub train_label: String,

    pub val_img: String,
    pub val_label: String,

    pub test_img: String,
    pub test_label: String,

    // yoooo
    pub train_ratio: f32,
    pub val_ratio: f32,
    pub test_ratio: f32,
    pub export_folder: PathBuf,
}

impl ExportFolderOptions {
    pub fn new(export_path: &str, split_train_val: f32) -> Result<ExportFolderOptions, Error> {
        let split_rem = 1.0 - split_train_val;
        let split_test = split_rem / 3.0;
        let split_val = split_rem - split_test;
        Ok(ExportFolderOptions {
            train_img: format!("{}/train/images", &export_path),
            train_label: format!("{}/train/labels", &export_path),
            val_img: format!("{}/val/images", &export_path),
            val_label: format!("{}/val/labels", &export_path),
            test_img: format!("{}/test/images", &export_path),
            test_label: format!("{}/test/labels", &export_path),
            train_ratio: split_train_val,
            val_ratio: split_val,
            test_ratio: split_test,
            export_folder: fs::canonicalize(export_path).unwrap(),
        })
    }

    pub fn create_folders(&self) -> Result<(), Error> {
        fs::create_dir_all(&self.train_img.to_owned())?;
        fs::create_dir_all(&self.train_label.to_owned())?;
        fs::create_dir_all(&self.val_label.to_owned())?;
        fs::create_dir_all(&self.val_img.to_owned())?;
        fs::create_dir_all(&self.test_img.to_owned())?;
        fs::create_dir_all(&self.test_label.to_owned())?;
        Ok(())
    }
}

pub fn get_all_jsons(input: &str) -> Result<Vec<PathBuf>, Error> {
    let all_jsons: Vec<PathBuf> = fs::read_dir(&input)
        .unwrap()
        .filter_map(|f| f.ok())
        .filter(|f| match f.path().extension() {
            None => false,
            Some(ex) => ex == "json",
        })
        .map(|f| f.path())
        .collect();
    Ok(all_jsons)
}

/// takes a all classes label list and then puts it in a mf HASHMAP AHHHHHHHHHHHHHHHHHH
pub fn get_all_classes_hash(label_list: &Vec<String>) -> Result<HashMap<String, i64>> {
    let mut result: HashMap<String, i64> = HashMap::new();
    for (idx, label_name) in label_list.iter().enumerate() {
        result.insert(String::from(label_name), idx as i64);
    }
    Ok(result)
}

/// get all classes from a folder input
pub fn get_all_classes(input: &Vec<PathBuf>) -> Result<Vec<String>, Error> {
    let mut label_list: Vec<String> = vec![];
    for x in input.iter() {
        let _json = read_labels_from_file(x.to_str().expect("can't convert PathBuf to str"));
        for y in _json?.shapes.into_iter() {
            label_list.push(y.label.to_owned());
        }
    }
    label_list.sort();
    label_list.dedup();
    Ok(label_list)
}

/// writes to txt
pub fn write_yolo_txt(input_yolo: Vec<YoloAnnotation>, file_path: &PathBuf) -> Result<(), Error> {
    // println!("pathbuf: {:?}", file_path);
    let mut _txt_file_name = file_path.to_owned();
    _txt_file_name.set_extension("txt");
    let mut txtfile = fs::File::create(&_txt_file_name).expect("cannot create file!");
    for shape in input_yolo.iter() {
        txtfile
            .write_all(
                format!(
                    "{:?} {:?} {:?} {:?} {:?}",
                    &shape.class.to_owned(),
                    &shape.xmin.to_owned(),
                    &shape.ymin.to_owned(),
                    &shape.w.to_owned(),
                    &shape.h.to_owned()
                )
                .as_bytes(),
            )
            .expect("Error in writing txt file:");
        txtfile
            .write_all("\n".as_bytes())
            .expect("error in writing space to txt")
    }
    Ok(())
}

pub fn write_data_yaml(
    export_options: &ExportFolderOptions,
    all_classes: &Vec<String>,
) -> Result<(), Error> {
    let data_yaml: DatasetInfo = DatasetInfo::new(export_options, all_classes)?;
    let yaml_fname = export_options.export_folder.to_owned().join("data.yaml");
    let mut yaml_file = fs::File::create(yaml_fname).unwrap();
    serde_yaml::to_writer(&mut yaml_file, &data_yaml)?;
    // println!("YAML DIR: {:?}", export_options.export_folder.to_owned().join("data.yaml"));
    Ok(())
}
