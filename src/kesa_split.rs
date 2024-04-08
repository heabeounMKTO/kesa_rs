mod fileutils;
mod image_augmentations;
mod image_utils;
mod label;
mod output;
mod splash;

use anyhow::{bail, Error, Result};
use clap::Parser;
use fileutils::{get_all_classes, ExportFolderOptions};
use image_augmentations::augmentations::AugmentationType;
use indicatif::ProgressBar;
use kesa::fileutils::get_all_images;
use label::read_labels_from_file;
use rayon::prelude::*;
use spinoff::{spinners, Color, Spinner};
use splash::print_splash;
use std::collections::HashMap;
use std::{fs, path::PathBuf};

use crate::fileutils::{
    get_all_classes_hash, get_all_jsons, get_all_txts, write_data_yaml, write_yolo_to_txt,
};

#[derive(Parser, Debug)]
struct CliArguments {
    #[arg(long)]
    folder: String,

    #[arg(long)]
    workers: Option<i64>,

    #[arg(long)]
    export: Option<String>,

    #[arg(long)]
    ext: String,
}

fn main() -> Result<(), Error> {
    print_splash();
    let args = CliArguments::parse();
    let workers = match &args.workers {
        Some(ref _i64) => args.workers,
        None => Some(4),
    };

    let export = match &args.export {
        Some(ref _string) => args.export,
        None => Some(String::from("export")),
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(workers.unwrap().try_into().unwrap())
        .build_global()
        .unwrap();
    let export_options = ExportFolderOptions::new(export.unwrap().as_str(), 0.7)?;
    export_options.create_folders()?;

    let all_txt = get_all_txts(&args.folder)?;

    let all_json = get_all_jsons(&args.folder)?;
    let all_classes = get_all_classes(&all_json)?;

    let train_split = all_txt.len().to_owned() as f32 * export_options.train_ratio;
    let val_split =
        all_txt.len().to_owned() as f32 * (export_options.train_ratio + export_options.val_ratio);

    let train_batch = all_txt[0..train_split as usize].to_vec();
    let val_batch = all_txt[train_split as usize..val_split as usize].to_vec();
    let test_batch = all_txt[val_split as usize..].to_vec();

    write_data_yaml(&export_options, &all_classes)?;

    move_txt_files(
        train_batch,
        &args.folder,
        &export_options,
        "train",
        &args.ext,
    )?;
    move_txt_files(val_batch, &args.folder, &export_options, "valid", &args.ext)?;
    move_txt_files(test_batch, &args.folder, &export_options, "test", &args.ext)?;
    Ok(())
}

fn move_txt_files(
    input_txt_array: Vec<PathBuf>,
    orig_path: &str,
    export_options: &ExportFolderOptions,
    batch: &str,
    img_ext: &str, // will change this later to match or something idk
) -> Result<(), Error> {
    println!("moving files  to `{}` batch", &batch);
    let prog = ProgressBar::new(input_txt_array.len().to_owned() as u64);
    for orig_txt_file in input_txt_array.iter() {
        prog.inc(1);
        let orig_txt_file = orig_txt_file.to_owned();
        let mut orig_img_file = orig_txt_file.clone();
        orig_img_file.set_extension(img_ext);

        match batch {
            "train" => {
                let dest_image = PathBuf::from(&export_options.train_img)
                    .join(orig_img_file.file_name().to_owned().unwrap());
                let dest_label = PathBuf::from(&export_options.train_label)
                    .join(orig_txt_file.file_name().to_owned().unwrap());
                fs::rename(orig_img_file, dest_image)?;
                fs::rename(orig_txt_file, dest_label)?;
            }
            "valid" => {
                let dest_image = PathBuf::from(&export_options.val_img)
                    .join(orig_img_file.file_name().to_owned().unwrap());
                let dest_label = PathBuf::from(&export_options.val_label)
                    .join(orig_txt_file.file_name().to_owned().unwrap());
                fs::rename(orig_img_file, dest_image)?;
                fs::rename(orig_txt_file, dest_label)?;
            }
            "test" => {
                let dest_image = PathBuf::from(&export_options.test_img)
                    .join(orig_img_file.file_name().to_owned().unwrap());
                let dest_label = PathBuf::from(&export_options.test_label)
                    .join(orig_txt_file.file_name().to_owned().unwrap());
                fs::rename(orig_img_file, dest_image)?;
                fs::rename(orig_txt_file, dest_label)?;
            }
            _ => {
                bail!("unrecognized batch name {:?}", batch)
            }
        }
    }
    prog.finish_with_message("files moved !\n");
    println!("files moving done!");
    Ok(())
}
