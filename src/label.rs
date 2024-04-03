/* anything that's label related */
use crate::image_utils::dynimg2string;
use crate::output::OutputFormat;
use anyhow::{Error, Result};
use image::DynamicImage;
use ndarray::{ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
/// struct for storing generic xyxy's
/// for conversion between normalized
/// and screen coordinates.
#[derive(Debug)]
pub struct Xyxy {
    pub coordinate_type: CoordinateType,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl Xyxy {
    pub fn new(coordinate_type: CoordinateType, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Xyxy {
            coordinate_type,
            x1,
            y1,
            x2,
            y2,
        }
    }

    /// get center coordinates/values
    /// it is mapped as [x, y]
    pub fn get_center_xy(&self) -> Vec<f32> {
        vec![((self.x1 + self.x2) / 2.0), ((self.y1 + self.y2) / 2.0)]
    }

    /// coordinates from yolo
    pub fn from_yolo(input_yolo: &YoloAnnotation) -> Result<Self, Error> {
        Ok(Xyxy {
            coordinate_type: CoordinateType::Screen,
            x1: input_yolo.xmin,
            y1: input_yolo.ymin,
            x2: input_yolo.w,
            y2: input_yolo.h,
        })
    }
    pub fn to_screen(&self, img_dims: &(u32, u32)) -> Result<Self, Error> {
        match &self.coordinate_type {
            CoordinateType::Screen => {
                panic!("Given Coordinate is already screen!")
            }
            CoordinateType::Normalized => Ok(Xyxy {
                coordinate_type: CoordinateType::Screen,
                x1: self.x1 * img_dims.0 as f32,
                y1: self.y1 * img_dims.1 as f32,
                x2: self.x2 * img_dims.0 as f32,
                y2: self.y2 * img_dims.1 as f32,
            }),
        }
    }

    pub fn points(&self) -> Vec<Vec<f32>> {
        vec![vec![self.x1, self.y1], vec![self.x2, self.y2]]
    }
    /// returns normalized coordinates
    pub fn to_normalized(&self, img_dims: &(u32, u32)) -> Result<Self, Error> {
        match &self.coordinate_type {
            CoordinateType::Normalized => {
                panic!("Given Coordinates is already normalized!")
            }
            CoordinateType::Screen => Ok(Xyxy {
                coordinate_type: CoordinateType::Normalized,
                x1: self.x1 / img_dims.0 as f32,
                y1: self.y1 / img_dims.1 as f32,
                x2: self.x2 / img_dims.0 as f32,
                y2: self.y2 / img_dims.1 as f32,
            }),
        }
    }
}

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
        inference_dimension: &(u32, u32),
    ) -> std::result::Result<Vec<Shape>, anyhow::Error> {
        todo!()
    }
    fn to_labelme(
        &self,
        all_classes: &Vec<String>,
        original_dimension: &(u32, u32),
        filename: &str,
        image_file: &DynamicImage,
        inference_dimension: &(u32, u32),
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

impl LabelmeAnnotation {
    pub fn get_xyxy(&self) -> Result<Vec<Xyxy>, Error> {
        let mut all_xyxys: Vec<Xyxy> = vec![];
        for shape in self.shapes.iter() {
            all_xyxys.push(Xyxy::new(
                CoordinateType::Screen,
                shape.points[0][0],
                shape.points[0][1],
                shape.points[1][0],
                shape.points[1][1],
            ));
        }
        Ok(all_xyxys)
    }

    pub fn update_shapes(&mut self) {
        todo!()
    }

    /// converts labelme annotation to yolo shape
    pub fn to_yolo(&self, class_hash: &HashMap<String, i64>) -> Result<Vec<YoloAnnotation>, Error> {
        let mut yolo_label_list: Vec<YoloAnnotation> = vec![];
        for shape in self.shapes.iter() {
            let temp_xyxy: Xyxy = get_xyxy_from_shape(&shape, CoordinateType::Normalized);
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

/// parsed directrly from the json file eh
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shape {
    pub label: String,
    pub points: Vec<Vec<f32>>,
    pub group_id: Option<String>,
    pub shape_type: String,
    pub flags: HashMap<String, String>,
}

impl Shape {
    pub fn update_points_from_xyxy(&mut self, new_xyxy: Xyxy) {
        let x1y1 = vec![new_xyxy.x1, new_xyxy.y1];
        let x2y2 = vec![new_xyxy.x2, new_xyxy.y2];
        self.points = vec![x1y1, x2y2];
    }
}

pub fn get_xyxy_from_shape(input_shape: &Shape, coordinate_type: CoordinateType) -> Xyxy {
    Xyxy {
        coordinate_type,
        x1: input_shape.points[0][0].to_owned(),
        y1: input_shape.points[0][1].to_owned(),
        x2: input_shape.points[1][0].to_owned(),
        y2: input_shape.points[1][1].to_owned(),
    }
}

/// TODO: SHAPE MASK  for segmentation tasks , unimplemnetd for now :|
pub enum ShapeType {
    Rectangle,
    Mask,
}

#[derive(Debug)]
pub enum CoordinateType {
    Screen,
    Normalized,
}

// not sure where to put these, just leaving it here for now :|
#[derive(Debug, Clone)]
pub struct Embeddings {
    data: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
}
impl Embeddings {
    pub fn new(data: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>) -> Self {
        Self { data }
    }

    pub fn to_vec(&self) -> Result<Vec<Vec<f32>>, Error> {
        let mut results_vec: Vec<Vec<f32>> = vec![];
        for n in 0..self.data().shape()[0] {
            let _a = self.data().view().to_owned().select(Axis(0), &[n]);
            let _v = Vec::from_iter(_a);
            results_vec.push(_v);
        }
        Ok(results_vec)
    }

    pub fn data(&self) -> &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
        &self.data
    }
}

impl OutputFormat for Embeddings {
    fn to_shape(
        &self,
        all_classes: &Vec<String>,
        original_dimension: &(u32, u32),
        inference_dimension: &(u32, u32),
    ) -> Result<Vec<Shape>, Error> {
        let raw_output_vec = self.to_yolo_vec();
        let g_id = self.to_vec()?;
        let mut shape_vec: Vec<Shape> = vec![];
        let shape = String::from("rectangle");
        let flags: HashMap<String, String> = HashMap::new();
        for _yolo in raw_output_vec.into_iter() {
            for (idx, elem) in _yolo.iter().enumerate() {
                // we getting the confidence with this one
                let gid_idx = g_id[idx][6].to_owned() * 100.0;

                let class_index = elem.class as usize;
                let class_name = all_classes[class_index].to_owned();
                let xy_coords: Vec<Vec<f32>> = Xyxy::from_yolo(&elem)?
                    .to_normalized(inference_dimension)?
                    .to_screen(original_dimension)?
                    .points();
                let _shape = Shape {
                    label: class_name,
                    points: xy_coords,
                    shape_type: shape.to_owned(),
                    group_id: Some(gid_idx.to_string()),
                    flags: flags.to_owned(),
                };
                shape_vec.push(_shape);
            }
        }
        // println!("my penis: {:#?}", raw_output_vec);
        Ok(shape_vec)
    }
    fn to_yolo(&self) -> Result<YoloAnnotation, anyhow::Error> {
        todo!()
    }
    fn to_yolo_vec(&self) -> Result<Vec<YoloAnnotation>, anyhow::Error> {
        let mut yolo_arr: Vec<YoloAnnotation> = vec![];
        let raw_output_vec = self.to_vec();
        for detection in raw_output_vec.into_iter() {
            for elm in detection {
                let res = YoloAnnotation::new(
                    elm[5] as i64, //class
                    elm[1],        // xmin
                    elm[2],        // xmax
                    elm[3],        // w
                    elm[4],        // h
                );
                yolo_arr.push(res);
            }
        }
        Ok(yolo_arr)
    }
    fn to_labelme(
        &self,
        all_classes: &Vec<String>,
        original_dimension: &(u32, u32),
        filename: &str,
        image_file: &DynamicImage,
        inference_dimension: &(u32, u32),
    ) -> Result<LabelmeAnnotation, anyhow::Error> {
        let all_shapes: Vec<Shape> = self.to_shape(all_classes, 
                                                   original_dimension,
                                                   inference_dimension)?;
        let version: String = String::from("5.1.1");
        let _file = PathBuf::from(&filename);
        let flags: HashMap<String, String> = HashMap::new();
        let base64img: String = dynimg2string(image_file).unwrap();
        println!("IMAGE_PATH : {:?}", PathBuf::from(&filename).file_name());
        Ok(LabelmeAnnotation {
            version,
            flags: Some(flags),
            shapes: all_shapes,
            imageWidth: original_dimension.0.to_owned() as i64,
            imageHeight: original_dimension.1.to_owned() as i64,
            imageData: base64img,
            // TODO: change to filename instead
            // of the whole mf directory
            imagePath: _file.file_name().unwrap().to_string_lossy().to_string()
        })
    }
}

pub fn read_labels_from_file(filename: &str) -> Result<LabelmeAnnotation, Error> {
    let json_filename = fs::read_to_string(filename).expect("READ ERROR: cannot read jsonfile !");
    let read_json_to_struct: LabelmeAnnotation = serde_json::from_str(&json_filename)
        .expect("LABELME ERROR: cannot read json to label me struct!");
    Ok(read_json_to_struct)
}
