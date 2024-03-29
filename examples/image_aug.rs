extern crate kesa;

use image::GenericImageView;
use image::ImageFormat;
use kesa::fileutils::get_all_classes;
use kesa::fileutils::get_all_jsons;
use kesa::image_augmentations;
use image::imageops;

use anyhow::{Result, Error};
use kesa::image_augmentations::augmentations::ImageAugmentation;
use kesa::image_utils::open_image;
use kesa::label::read_labels_from_file;
use kesa::fileutils::get_all_classes_hash;

pub fn main() -> Result<(), Error>{
    let read_img = open_image("test/test2.png")?;
    let all_json = get_all_jsons("test")?;
    let all_classes = get_all_classes(&all_json)?;
    let all_classes_hash = get_all_classes_hash(&all_classes)?;
    let read_annotations = read_labels_from_file("test/test2.json")?.get_xyxy()?; 
    let mut aug = ImageAugmentation {
        image: read_img,
        coords: read_annotations
    };
    aug.flip_h();
    println!("readed annos: {:?}", &aug.coords);
    // let flip_v = imageops::flip_horizontal(&read_img);
    // read_img.save("test/test2_save.jpeg")?;
    // flip_v.save_with_format("test/tflip_v.jpeg", ImageFormat::Jpeg)?;
    // println!("flip : {:?}", read_img.dimensions());
    Ok(())
}
