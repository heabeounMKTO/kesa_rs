use std::{fs, io, path, path::PathBuf, ffi::OsStr};
use serde_derive::{Deserialize, Serialize};
use serde_json::{Result,Value, json};
use std::collections::HashMap;
use sorted_list::SortedList;

#[derive(Debug, Clone)]
pub struct GenericAnnotation{
    pub label: String,
    pub image_width: i32,
    pub image_height: i32,
    pub image_path: i32,
    pub x1y1: [f32;2],
    pub x2y2: [f32;2]
}

impl GenericAnnotation{
    pub fn new(label: &str, image_width: i32, image_height: i32, image_path: String, x1y1: GenericLabelPoints, x2y2: GenericLabelPoints) -> Annotation{
        return Annotation{
           label: String::from(label),
           image_width: image_width,
           image_height: image_height,
           image_path: image_path,
           x1y1: GenericLabelPoints,
           x2y2: GenericLabelPoints
         }
    }

    pub fn label(&self) -> String {String::from(&self.label)}
    pub fn image_width(&self) -> i32 {self.image_width}
    pub fn image_height(&self) -> i32 {self.image_height}
    pub fn x1y1(&self) -> GenericLabelPoints {self.x1y1}
    pub fn x2y2(&self) -> GenericLabelPoints {self.x2y2} 
}

pub struct GenericLabelPoints{
    pub x: f32,
    pub y: f32
}

impl GenericLabelPoints {
    pub fn new(x: f32, y:f32){
        GenericLabelPoints{
            x: x,
            y: y
        }
    }
}


#[derive(Debug, Clone)]
// xywh
pub struct YoloLabel{
    pub label_index: i32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32
}

impl YoloLabel{
    pub fn new(label_index: i32,x: f32, y: f32, w: f32, h: f32) -> xywh{
        return xywh {
            label_index: label_index,
            x: x,
            y: y,
            w: w,
            h: h,
         }

    }
}

