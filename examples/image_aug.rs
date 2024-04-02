extern crate kesa;

use image::GenericImageView;
use image::ImageFormat;
use kesa::fileutils::get_all_classes;
use kesa::fileutils::get_all_jsons;
use kesa::image_augmentations;
use image::imageops;

use std::path::PathBuf;

use anyhow::{Result, Error};
use kesa::image_augmentations::augmentations::ImageAugmentation;
use kesa::image_utils::open_image;
use kesa::label::read_labels_from_file;
use kesa::fileutils::get_all_classes_hash;

pub fn main() -> Result<(), Error>{
    let read_img = open_image("test/test.png")?;
    let all_json = get_all_jsons("test")?;
    let all_classes = get_all_classes(&all_json)?;
    let all_classes_hash = get_all_classes_hash(&all_classes)?;
    // get xyxy from image
    let read_annotations = read_labels_from_file("test/test.json")?; 
    let read_anno_xyxy = read_annotations.to_owned().get_xyxy()?;
    // construct and augment 
    // println!("readed annos: {:?}", &read_annotations);
    println!("readed xyxy: {:?}", &read_anno_xyxy);
    let mut aug = ImageAugmentation {
        image: read_img,
        coords: read_annotations 
    };
    aug.flip_v();
    aug.flip_h();
    aug.random_brightness((-100, 100));
    aug.write_annotations(&PathBuf::from("test"), &all_classes_hash);
    // let flip_v = imageops::flip_horizontal(&read_img);
    // read_img.save("test/test2_save.jpeg")?;
    // flip_v.save_with_format("test/tflip_v.jpeg", ImageFormat::Jpeg)?;
    // println!("flip : {:?}", read_img.dimensions());
    Ok(())
}
