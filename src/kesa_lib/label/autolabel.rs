use super::autolabel_struct::LabelSettings;
use crate::{kesa_utils::file_utils::get_image_from_json_path, yolo::yolo};

pub fn get_labels_from_img() {
    let test_string = String::from("test\\dragon-1683001691664.json");
    get_image_from_json_path(&test_string);
}
