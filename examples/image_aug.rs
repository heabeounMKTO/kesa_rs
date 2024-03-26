extern crate kesa;

use image::GenericImageView;
use image::ImageFormat;
use image::imageops;

use anyhow::{Result, Error};
use kesa::image_utils::open_image;


pub fn main() -> Result<(), Error>{
    let read_img = open_image("test/test2.png")?;
    let flip_v = imageops::flip_horizontal(&read_img);
    read_img.save("test/test2_save.png")?;
    flip_v.save("test/tflip_v.png")?;
    println!("flip : {:?}", read_img.dimensions());
    Ok(())
}
