use crate::image_utils::open_image;
use crate::label::{YoloAnnotation, Xyxy};
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


    /// flips horizontally,
    /// or "along the x axis ☝️🤓" for u nerds out there
    pub fn flip_h(
        &mut self,
    ) {
        let flipped_h_image = imageops::flip_horizontal(&self.image);
        self.image = DynamicImage::ImageRgba8(flipped_h_image);
    }

}
