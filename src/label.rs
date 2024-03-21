/* anything that's label related */
use anyhow::{Error, Result};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs;

use crate::output::OutputFormat;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoloAnnotation {
    pub class: i64,
    pub xmin: f32,
    pub ymin: f32,
    pub w: f32,
    pub h: f32,
}
impl YoloAnnotation {
    pub fn new(class: i64, xmin: f32, ymin: f32, w: f32, h: f32) -> YoloAnnotation {
        YoloAnnotation {
            class,
            xmin,
            ymin,
            w,
            h,
        }
    }
    /// for creating placeholder
    pub fn dummy() -> YoloAnnotation {
        YoloAnnotation {
            class: 1,
            xmin: 10.0,
            ymin: 10.0,
            w: 10.0,
            h: 10.0,
        }
    }
}

impl OutputFormat for YoloAnnotation {
    fn to_yolo_vec(&self) -> std::result::Result<Vec<YoloAnnotation>, anyhow::Error> {
        todo!()
    }
    fn to_yolo(&self) -> std::result::Result<YoloAnnotation, anyhow::Error> {
        // this might end very badly
        panic!("Invalid Operation , cannot convert `YoloAnnotation` to `YoloAnnotation`")
    }
    fn to_shape(
        &self,
        all_classes: &Vec<String>,
        original_dimension: &(u32, u32),
    ) -> std::result::Result<Vec<Shape>, anyhow::Error> {
        todo!()
    }
    fn to_labelme(
        &self,
        all_classes: &Vec<String>,
        original_dimension: &(u32, u32),
        filename: &str,
        image_file: &DynamicImage,
    ) -> std::result::Result<LabelmeAnnotation, anyhow::Error> {
        todo!()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelmeAnnotation {
    pub version: String,
    pub flags: Option<HashMap<String, String>>,
    pub shapes: Vec<Shape>,
    pub imagePath: String,
    pub imageData: String,
    pub imageWidth: i64,
    pub imageHeight: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shape {
    pub label: String,
    pub points: Vec<Vec<f32>>,
    pub group_id: Option<String>,
    pub shape_type: String,
    pub flags: HashMap<String, String>,
}

pub struct xyxy {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl LabelmeAnnotation {
    /// converts labelme annotation to yolo shape
    pub fn to_yolo(&self, class_hash: &HashMap<String, i64>) -> Result<Vec<YoloAnnotation>, Error> {
        let mut yolo_label_list: Vec<YoloAnnotation> = vec![];
        for shape in self.shapes.iter() {
            let temp_xyxy: xyxy = get_xyxy_from_shape(&shape);
            let x = ((temp_xyxy.x1 + temp_xyxy.x2) / 2.0) / self.imageWidth as f32;
            let y = ((temp_xyxy.y1 + temp_xyxy.y2) / 2.0) / self.imageHeight as f32;
            let w = (temp_xyxy.x2 - temp_xyxy.x1) / self.imageWidth as f32;
            let h = (temp_xyxy.y2 - temp_xyxy.y1) / self.imageHeight as f32;
            let label_index = class_hash.get(&shape.label).expect("cannot find index!");
            let yolo_struct: YoloAnnotation = YoloAnnotation {
                class: *label_index, // deref bih, uh
                xmin: x,
                ymin: y,
                w: w,
                h: h,
            };
            yolo_label_list.push(yolo_struct);
        }
        Ok(yolo_label_list)
    }
}

pub fn get_xyxy_from_shape(input_shape: &Shape) -> xyxy {
    xyxy {
        x1: input_shape.points[0][0].to_owned(),
        y1: input_shape.points[0][1].to_owned(),
        x2: input_shape.points[1][0].to_owned(),
        y2: input_shape.points[1][1].to_owned(),
    }
}

pub fn read_labels_from_file(filename: &str) -> Result<LabelmeAnnotation, Error> {
    let json_filename = fs::read_to_string(filename).expect("READ ERROR: cannot read jsonfile !");
    let read_json_to_struct: LabelmeAnnotation = serde_json::from_str(&json_filename)
        .expect("LABELME ERROR: cannot read json to label me struct!");
    Ok(read_json_to_struct)
}
