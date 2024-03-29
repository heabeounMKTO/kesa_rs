use crate::image_utils::open_image;
use crate::label::{YoloAnnotation, Xyxy, CoordinateType};
use anyhow::{Error, Result};
use clap::Subcommand;
use image::{self, DynamicImage, imageops, GenericImageView};
use crate::label::Shape;



#[derive(Debug)]
pub enum AugmentationType {
    FlipHorizontal,
    FlipVeritcal,
}

#[derive(Debug)]
pub struct ImageAugmentation {
    pub image: DynamicImage,
    pub coords: Vec<Xyxy> 
}

impl ImageAugmentation {
    pub fn new(image: DynamicImage, coords: Vec<Xyxy>) -> ImageAugmentation {
         ImageAugmentation { image:  image, coords: coords }        
    }
    

    /// flips an image and it's annotation 
    /// horizontally, or "along the x axis ☝️🤓" 
    /// for u nerds out there
    pub fn flip_h(
        &mut self,
    ) {
        let flipped_h_image = imageops::flip_horizontal(&self.image);
        self.image = DynamicImage::ImageRgba8(flipped_h_image);
    }

    /// flips an image and it's annotation 
    /// vertically , or "along the y axis ☝️🤓" 
    /// for u nerds out there
    pub fn flip_v(&mut self) {
        self.image = DynamicImage::ImageRgba8(imageops::flip_vertical(&self.image));
        let flipped_v_coord: Vec<Xyxy> = vec![];
        for coord in self.coords.iter() {
            match coord.coordinate_type {
                CoordinateType::Screen => {
                    todo!()
                },
                CoordinateType::Normalized => {
                    todo!()
                }
            }
        }
        self.coords = flipped_v_coord;
    }
}
