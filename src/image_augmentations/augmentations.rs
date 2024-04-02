use crate::image_utils::open_image;
use crate::label::{YoloAnnotation, Xyxy, CoordinateType, LabelmeAnnotation};
use anyhow::{Error, Result};
use clap::Subcommand;
use image::{self, DynamicImage, imageops, GenericImageView};
use sorted_list::Tuples;
use crate::fileutils::{write_yolo_to_txt, write_labelme_to_json};
use crate::label::Shape;
use uuid::Uuid;
use std::collections::HashMap;
use std::path::PathBuf;
use ndarray::prelude::*;
use rand::prelude::*;

#[derive(Debug)]
pub enum AugmentationType {
    FlipHorizontal,
    FlipVeritcal,
}

#[derive(Debug)]
pub struct ImageAugmentation {
    pub image: DynamicImage,
    pub coords: LabelmeAnnotation 
}

impl ImageAugmentation {
    /// write a augmented label & image 
    /// , takes a export path 
    /// write the filename using UUID to avoid overwriting a old one 
    pub fn write_annotations(&mut self, write_dir: &PathBuf, class_hash: &HashMap<String, i64>) -> Result<(), Error> {
        let anno_uuid = Uuid::new_v4().to_string();
        let mut img_path = write_dir.clone();
        let mut label_path = write_dir.clone();

        let img_fname =format!("{}.png", &anno_uuid); 
        img_path.push(&img_fname);
        label_path.push(format!("{}.json", &anno_uuid));
        println!("imgpath: {:?}\nlabelpath: {:?}", &img_path, &label_path);
        self.image.save(&img_path)?;
        
        self.coords.imagePath = img_fname.to_owned();
        
        // println!("self.imagePath {:?}", self.coords.imagePath);
        let yolo_anno = self.coords.to_yolo(class_hash)?;
        write_labelme_to_json(&self.coords, &img_path)?;
        write_yolo_to_txt(yolo_anno, &img_path)?;
        Ok(())
    }

    pub fn new(image: DynamicImage, coords: LabelmeAnnotation) -> ImageAugmentation {
         ImageAugmentation { image:  image, coords: coords }        
    }
    
    /// adds random amount of brightness in a given range 
    /// negative values subtract brightness
    pub fn random_brightness(&mut self, range: (i32, i32)) {
        let mut rng = rand::thread_rng();
        let p = imageops::brighten(&self.image, rng.gen_range(range.0..range.1));
        self.image = DynamicImage::ImageRgba8(p);
    }

    /// flips an image and it's annotation 
    /// vertically , or "along the y axis ☝️🤓" 
    /// for u nerds out there
    /// subtract y coordinates by image height then 
    /// multiplies the coords by `[[1 , -1], [1 , -1]]`
    /// to flip along the y axis
    pub fn flip_v(&mut self) {
       let flipped_v_image = imageops::flip_vertical(&self.image); 
       self.image = DynamicImage::ImageRgba8(flipped_v_image);
       for shape in self.coords.shapes.iter_mut() {
            
            // subtract y coord by height and mult by -1
            shape.points[0][1] = ( shape.points[0][1] - (self.image.dimensions().1 as f32)) * -1.0; 
            shape.points[1][1] = ( shape.points[1][1] - (self.image.dimensions().1 as f32)) * -1.0;
            println!("shape: {:?}", &shape);
       } 

    }

    /// flips an image and it's annotation 
    /// horizontally , or "along the x axis ☝️🤓" 
    /// for u nerds out there
    /// subtract y coordinates by image height then 
    /// multiplies the coords by [[-1 , 1], [-1 , 1]]
    /// to flip along the x axis
    pub fn flip_h(&mut self) {
       let flipped_h_image = imageops::flip_horizontal(&self.image); 
       self.image = DynamicImage::ImageRgba8(flipped_h_image);
       for shape in self.coords.shapes.iter_mut() {
            
            // subtract x coord by width and mult by -1
            // we dont use ndarrays here sir
            shape.points[0][0] = ( shape.points[0][0] - (self.image.dimensions().0 as f32)) * -1.0; 
            shape.points[1][0] = ( shape.points[1][0] - (self.image.dimensions().0 as f32)) * -1.0;
            println!("shape: {:?}", &shape);
       } 
    }
}
