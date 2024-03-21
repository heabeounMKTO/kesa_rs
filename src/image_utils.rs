use anyhow::{Error, Result};
use image::io::Reader;
use image::DynamicImage;
use std::io::Cursor;

/// handles opening images ,
/// avoids crashing
/// if image failed to open
/// for whatever Reason
pub fn open_image(image_path: &str) -> Result<DynamicImage, Error> {
    let _open_image = image::open(image_path)?;
    Ok(_open_image)
}
