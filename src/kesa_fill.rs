mod fileutils;
mod image_augmentations;
mod image_utils;
mod label;
mod output;
mod splash;

use anyhow::{bail, Error, Result};
use clap::Parser;
use fileutils::{get_all_classes, open_image, ExportFolderOptions, get_json_from_image,get_all_images, write_labelme_to_json};
use image::{DynamicImage, GenericImageView};
use indicatif::ProgressBar;
use kesa::image_utils::dynimg2string;
use label::{read_labels_from_file,Shape, LabelmeAnnotation};
use rand::distributions::{Distribution, Uniform};
use rayon::prelude::*;
use spinoff::{spinners, Color, Spinner};
use splash::print_splash;
use std::collections::HashMap;
use std::fs::Metadata;
use std::{fs, path::PathBuf};

use crate::fileutils::{get_all_classes_hash, get_all_jsons, write_data_yaml, write_yolo_to_txt};

#[derive(Parser, Debug)]
struct CliArguments {
    #[arg(long)]
    folder: String,

    #[arg(long)]
    workers: Option<i64>
}

fn main() -> Result<(), Error> {
    print_splash();
    let args = CliArguments::parse();
    let workers = match &args.workers {
        Some(_) => args.workers,
        None => Some(4)
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(workers.unwrap().try_into().unwrap())
        .build_global()
        .unwrap();

    let mut spinner0 = Spinner::new(spinners::Dots, "[info]::kesa_fill: collecting images", Color::White);
    let mut all_images = get_all_images(&args.folder);




    spinner0.success(format!("[info]::kesa_fill: found {:?} images", &all_images.len()).as_str()); 
    let prog = ProgressBar::new(all_images.len().to_owned() as u64);
    all_images.par_iter_mut().for_each(|img| {
        prog.inc(1);
        let _json_path = get_json_from_image(&img).expect("[info]::kesa_fill: cannot convert to json path");
        match std::fs::metadata(&_json_path) {
            Ok(_) => {},
            Err(_) => {
                let _f = create_empty_annotation_from_image(&img);
                match _f {
                    Ok(label) => {
                       write_labelme_to_json(&label,&img); 
                    },
                    Err(e) => {
                        eprintln!("[error]::kesa_fill: {:?}\ncannot create labelme json for file {:?}",e, &img);
                    }
                }
            }
        }
    });

    prog.finish_with_message("[info]::kesa_fill: filled empty images!\n");
    Ok(())

}



fn create_empty_annotation_from_image(input_img: &PathBuf) -> Result<LabelmeAnnotation, Error> {
    let read_img = open_image(&input_img)?;
    let b64img = dynimg2string(&read_img)?;
    let _empty_shape: Vec<Shape> = vec![];
    Ok(
        LabelmeAnnotation::new(
            None, 
            _empty_shape,
            input_img.file_name().unwrap().to_string_lossy().to_string(),
            b64img,
            read_img.dimensions().0 as i64,
            read_img.dimensions().1 as i64
        )
    )
}
