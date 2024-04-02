mod fileutils;
mod image_augmentations;
mod image_utils;
mod label;
mod output;
mod splash;

use anyhow::{bail, Error, Result};
use clap::Parser;
use fileutils::{get_all_classes, open_image, ExportFolderOptions};
use image::DynamicImage;
use image_augmentations::augmentations::{AugmentationType, ImageAugmentation};
use indicatif::ProgressBar;
use label::{read_labels_from_file, LabelmeAnnotation};
use rayon::prelude::*;
use spinoff::{spinners, Color, Spinner};
use splash::print_splash;
use std::collections::HashMap;
use std::{fs, path::PathBuf};

use crate::fileutils::{get_all_classes_hash, get_all_jsons, write_data_yaml, write_yolo_to_txt};

#[derive(Parser, Debug)]
struct CliArguments {
    #[arg(long)]
    folder: String,

    #[arg(long)]
    workers: Option<i64>,

    #[arg(long)]
    /// export folder
    export: Option<String>,

    #[arg(long)]
    /// export format , labelme or yolo?!
    format: Option<String>,
}

fn main() -> Result<(), Error> {
    print_splash();
    let args = CliArguments::parse();
    let workers = match &args.workers {
        Some(ref _i64) => args.workers,
        None => Some(4),
    };

    let export_format = match &args.format {
        Some(ref String) => args.format,
        None => Some(String::from("labelme")),
    }
    .unwrap();

    println!("export format {:?}", &export_format);

    rayon::ThreadPoolBuilder::new()
        .num_threads(workers.unwrap().try_into().unwrap())
        .build_global()
        .unwrap();
    let mut spinner0 = Spinner::new(spinners::Hearts, "collecting jsons..", Color::White);
    let all_json = get_all_jsons(&args.folder)?;
    let all_classes = get_all_classes(&all_json)?;
    let classes_hash = get_all_classes_hash(&all_classes)?;
    spinner0.success(format!("found {:?} json files", &all_json.len()).as_str());
    let prog = ProgressBar::new(all_json.len().to_owned() as u64);
    all_json.par_iter().for_each(|file| {
        prog.inc(1);
        // fuck handing <Result>
        create_augmentations(&file, &classes_hash, &export_format, &args.folder);
    });
    prog.finish_with_message("created augmentations!\n");
    Ok(())
}

fn create_augmentations(
    json_path: &PathBuf,
    class_hash: &HashMap<String, i64>,
    export_format: &str,
    export_folder: &str,
) -> Result<(), Error> {
    let label = read_labels_from_file(json_path.to_str().unwrap())?;
    let img = open_image(&PathBuf::from(&label.imagePath))?;

    let mut aug = ImageAugmentation {
        image: img,
        coords: label,
    };

    aug.flip_v();
    aug.flip_h();
    aug.flip_v();
    aug.write_annotations(&PathBuf::from(export_folder), class_hash)?;
    Ok(())
}
