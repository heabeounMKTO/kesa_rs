use serde_derive::{Deserialize, Serialize};
use serde_json::{json, Result, Value};
use sorted_list::SortedList;
use std::collections::HashMap;
use std::{ffi::OsStr, fs, io, path, path::PathBuf};

#[derive(Debug, Clone)]
pub struct GenericAnnotation {
    pub label: String,
    pub image_width: i32,
    pub image_height: i32,
    pub image_path: String,
    pub x1y1: GenericLabelPoints,
    pub x2y2: GenericLabelPoints,
}

impl GenericAnnotation {
    pub fn new(
        label: &str,
        image_width: i32,
        image_height: i32,
        image_path: String,
        x1y1: GenericLabelPoints,
        x2y2: GenericLabelPoints,
    ) -> GenericAnnotation {
        return GenericAnnotation {
            label: String::from(label),
            image_width: image_width,
            image_height: image_height,
            image_path: image_path,
            x1y1: x1y1,
            x2y2: x2y2,
        };
    }

    pub fn label(&self) -> String {
        String::from(&self.label)
    }
    pub fn image_width(&self) -> i32 {
        self.image_width
    }
    pub fn image_height(&self) -> i32 {
        self.image_height
    }
    pub fn x1y1(&self) -> GenericLabelPoints {
        self.x1y1
    }
    pub fn x2y2(&self) -> GenericLabelPoints {
        self.x2y2
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct GenericLabelPoints {
    pub x: f32,
    pub y: f32,
}

impl GenericLabelPoints {
    pub fn new(x: f32, y: f32) -> GenericLabelPoints {
        GenericLabelPoints { x: x, y: y }
    }
}

#[derive(Debug, Clone)]
// xywh
pub struct YoloLabel {
    pub label_index: i32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl YoloLabel {
    pub fn new(label_index: i32, x: f32, y: f32, w: f32, h: f32) -> YoloLabel {
        return YoloLabel {
            label_index: label_index,
            x: x,
            y: y,
            w: w,
            h: h,
        };
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LabelMeShapes {
    pub label: String,
    pub points: Vec<GenericLabelPoints>,
    pub shape_type: String,
    pub flags: HashMap<String, String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LabelMeLabel {
    pub version: String,
    pub flags: HashMap<String, String>,
    pub shapes: Vec<LabelMeShapes>,
    pub imagePath: String, //must match actual LabelMe json, so keysmust match
    pub imageData: String,
    pub imageHeight: i32,
    pub imageWidth: i32,
}

impl LabelMeLabel {
    pub fn version(&self) -> String {
        String::from(&self.version)
    }
    pub fn image_width(&self) -> i32 {
        self.imageWidth
    }
    pub fn image_height(&self) -> i32 {
        self.imageHeight
    }
}
