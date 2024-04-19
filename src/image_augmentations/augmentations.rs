use crate::fileutils::{write_labelme_to_json, write_yolo_to_txt};
use crate::image_utils::{dynimg2string_png, open_image};
use crate::label::Shape;
use crate::label::{CoordinateType, LabelmeAnnotation, Xyxy, YoloAnnotation};
use anyhow::{Error, Result};
use clap::Subcommand;
use image::imageops::colorops;
use image::{self, imageops, DynamicImage, GenericImageView, GenericImage};
use ndarray::prelude::*;
use rand::prelude::*;
use sorted_list::Tuples;
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Debug)]
pub enum AugmentationType {
    FlipHorizontal,
    FlipVeritcal,
    RandomBrightness,
    UnSharpen,
    HueRotate30,
    HueRotate60,
    HueRotate90,
    HueRotate120,
    HueRotate180,
    HueRotate210,
    HueRotate270,
    Grayscale,
    Rotate90
}

#[derive(Debug)]
pub struct ImageAugmentation {
    pub image: DynamicImage,
    pub coords: LabelmeAnnotation,
}


/// yea CHATGPT IS A LYING MF
fn rotate_90_degrees_ccw(img: &DynamicImage) -> DynamicImage {
    // Get the dimensions of the image
    let (width, height) = img.dimensions();

    // Create a new image buffer for the rotated image
    let mut rotated_img = image::DynamicImage::new_rgba8(height, width);

    // Iterate over each pixel in the original image and copy it to the rotated image
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            rotated_img.put_pixel(height - y - 1, x, pixel);
        }
    }
    rotated_img
}


impl ImageAugmentation {
    /// write a augmented label & image
    /// , takes a export path
    /// write the filename using UUID to avoid overwriting a old one
    pub fn write_annotations(
        &mut self,
        write_dir: &PathBuf,
        class_hash: &HashMap<String, i64>,
    ) -> Result<(), Error> {
        let anno_uuid = Uuid::new_v4().to_string();
        let mut img_path = write_dir.clone();
        let mut label_path = write_dir.clone();

        let img_fname = format!("{}.png", &anno_uuid);
        img_path.push(&img_fname);
        label_path.push(format!("{}.json", &anno_uuid));
        self.image.save(&img_path)?;
        self.coords.imageData = dynimg2string_png(&self.image)?;
        self.coords.imagePath = img_fname.to_owned();
        let yolo_anno = self.coords.to_yolo(class_hash)?;

        write_labelme_to_json(&self.coords, &img_path)?;
        // TODO: add option to export
        // yolo directly
        // write_yolo_to_txt(yolo_anno, &img_path)?;
        Ok(())
    }

    /// what do u want me to explain 😠
    pub fn new(image: DynamicImage, coords: LabelmeAnnotation) -> ImageAugmentation {
        ImageAugmentation {
            image: image,
            coords: coords,
        }
    }

    pub fn grayscale(&mut self) {
        let _gscale = colorops::grayscale_alpha(&self.image);

        self.image = DynamicImage::ImageLumaA8(_gscale);
    }

    pub fn huerotate(&mut self, rotate_degree: i32) {
        let _hrotate = colorops::huerotate(&self.image, rotate_degree);

        self.image = DynamicImage::ImageRgba8(_hrotate);
    }

    pub fn unsharpen(&mut self, sigma: f32, threshold: i32) {
        let _unsharpen = imageops::unsharpen(&self.image, sigma, threshold);
        self.image = DynamicImage::ImageRgba8(_unsharpen);
    }
    

    /// full credit to stackoverflow guy for this python code! :)
    /// 
    /// https://stackoverflow.com/questions/71960632/how-to-rotate-a-rectangle-bounding-box-together-with-an-image
    /// ```code
    /// # assuming [[x1,y1], [x2,y2]]
    /// def rotate_90(bbox, img_width):
    ///  xmin,ymin = bbox[0]
    ///  xmax,ymax = bbox[1]
    ///  new_xmin = ymin
    ///  new_ymin = img_width-xmax
    ///  new_xmax = ymax
    ///  new_ymax = img_width-xmin
    ///  return [[new_xmin, new_ymin], 
    ///         [new_xmax, new_ymax]]
    /// ````
    pub fn rotate_90_counterclockwise(&mut self) {
        // this is a HACKy solution , but it works for now
        // first rotate it by 90 degrees clockwise, 
        // then flip h and flip v.
        //
        // TODO: actual ccw function but not now :| 
        let mut _rotate_90_ccw = imageops::rotate90(&self.image);
        _rotate_90_ccw = imageops::flip_vertical(&_rotate_90_ccw);
        _rotate_90_ccw = imageops::flip_horizontal(&_rotate_90_ccw);
        self.image = DynamicImage::ImageRgba8(_rotate_90_ccw);
    
        for shape in self.coords.shapes.iter_mut() {
           let new_x1 = shape.points[0][1].to_owned();   
           let new_y1 = &self.image.dimensions().0 - shape.points[1][0] as u32;
           let new_x2 = shape.points[1][1].to_owned();
           let new_y2 = &self.image.dimensions().0 - shape.points[0][0] as u32;

           shape.points[0][0] = new_x1;
           shape.points[0][1] = new_y1 as f32;
           shape.points[1][0] = new_x2;
           shape.points[1][1] = new_y2 as f32;
        }
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
    /// ```text
    /// subtracts y coordinates by image height then
    /// multiplies the coords by `[[1 , -1], [1 , -1]]`
    /// to flip along the y axis
    /// ````
    pub fn flip_v(&mut self) {
        let flipped_v_image = imageops::flip_vertical(&self.image);
        self.image = DynamicImage::ImageRgba8(flipped_v_image);
        for shape in self.coords.shapes.iter_mut() {
            // subtract y coord by height and mult by -1
            shape.points[0][1] = (shape.points[0][1] - (self.image.dimensions().1 as f32)) * - 1.0;
            shape.points[1][1] = (shape.points[1][1] - (self.image.dimensions().1 as f32)) * - 1.0;
        }
    }

    /// flips an image and it's annotation
    /// horizontally , or "along the x axis ☝️🤓"
    /// for u nerds out there
    /// ```text
    /// subtract y coordinates by image height then
    /// multiplies the coords by [[-1 , 1], [-1 , 1]]
    /// to flip along the x axis
    /// ```
    pub fn flip_h(&mut self) {
        let flipped_h_image = imageops::flip_horizontal(&self.image);
        self.image = DynamicImage::ImageRgba8(flipped_h_image);
        for shape in self.coords.shapes.iter_mut() {
            // subtract x coord by width and mult by -1
            // we dont use ndarrays here sir
            shape.points[0][0] = (shape.points[0][0] - (self.image.dimensions().0 as f32)) * - 1.0;
            shape.points[1][0] = (shape.points[1][0] - (self.image.dimensions().0 as f32)) * - 1.0;
        }
    }
}
