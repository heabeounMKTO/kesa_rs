use crate::image_utils::open_image;
use crate::label::YoloAnnotation;
use anyhow::{Error, Result};
use clap::Subcommand;
use image::{self, DynamicImage};

#[derive(Debug)]
pub enum AugmentationType {
    FlipHorizontal,
    FlipVeritcal,
}

pub struct ImageAugmentation {
    Image: DynamicImage,
    Augmentation: Vec<AugmentationType>,
}

impl ImageAugmentation {
    pub fn flip_h(
        image: &DynamicImage,
        augmentations: Vec<AugmentationType>,
    ) -> Result<DynamicImage, Error> {
        todo!()
    }
}
