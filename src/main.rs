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
use label::read_labels_from_file;
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

    #[arg(long, use_value_delimiter = true, value_delimiter = ',')]
    augment: Option<Vec<String>>,
    /// TODO: augment struct that takes kwargs** (idk what it's called)

    #[arg(long)]
    workers: Option<i64>,

    #[arg(long)]
    export: Option<String>,
}

fn main() -> Result<(), Error> {
    print_splash();
    let args = CliArguments::parse();
    let workers = match &args.workers {
        Some(ref _i64) => args.workers,
        None => Some(8),
    };

    let export = match &args.export {
        Some(ref _string) => args.export,
        None => Some(String::from("export")),
    };
    // set workers if not set, is 8 by default
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers.unwrap().try_into().unwrap())
        .build_global()
        .unwrap();

    let export_options = ExportFolderOptions::new(export.unwrap().as_str(), 0.7)?;
    println!("export options: {:#?}", export_options);
    let mut spinner0 = Spinner::new(spinners::Hearts, "creating export paths", Color::White);

    export_options.create_folders()?;
    spinner0.success("created export paths");

    let mut spinner = Spinner::new(
        spinners::Monkey,
        format!("searching for .json files in {:?}", &args.folder),
        Color::White,
    );
    let all_json = get_all_jsons(&args.folder)?;
    let all_classes = get_all_classes(&all_json)?;
    spinner.success(format!("found {:?} json files", &all_json.len()).as_str());

    let prog = ProgressBar::new(all_json.len().to_owned() as u64);
    let class_hash = get_all_classes_hash(&all_classes)?;
    println!("starting conversion !\n");
    all_json.par_iter().for_each(|file| {
        prog.inc(1);
        convert_labelme2yolo(file, &class_hash)
    });

    prog.finish_with_message("conversion done !\n");

    // split array into 3
    let train_split = all_json.len().to_owned() as f32 * export_options.train_ratio;
    let val_split =
        all_json.len().to_owned() as f32 * (export_options.train_ratio + export_options.val_ratio);

    let train_batch = all_json[0..train_split as usize].to_vec();
    let val_batch = all_json[train_split as usize..val_split as usize].to_vec();
    let test_batch = all_json[val_split as usize..].to_vec();

    write_data_yaml(&export_options, &all_classes)?;
    move_files(train_batch, &args.folder, &export_options, "train")?;
    move_files(val_batch, &args.folder, &export_options, "valid")?;
    move_files(test_batch, &args.folder, &export_options, "test")?;
    Ok(())
}

fn convert_labelme2yolo(json: &PathBuf, class_hash: &HashMap<String, i64>) -> () {
    // de-serialize from file to struct
    let all_shapes = read_labels_from_file(json.to_str().unwrap()).expect("read shapes error");
    // convert to yolo txt format
    let all_yolo = all_shapes
        .to_yolo(&class_hash)
        .expect("cannot convert yolo");
    let _write = write_yolo_to_txt(all_yolo, &json);
}

fn move_files(
    input_array: Vec<PathBuf>,
    orig_path: &str,
    export_options: &ExportFolderOptions,
    batch: &str,
) -> Result<(), Error> {
    println!("moving files to `{}` batch", &batch);
    let prog = ProgressBar::new(input_array.len().to_owned() as u64);
    for orig_json_file in input_array.iter() {
        prog.inc(1);
        let read_json_file = read_labels_from_file(orig_json_file.to_owned().to_str().unwrap())?;
        let mut orig_txt_file = orig_json_file.to_owned();
        orig_txt_file.set_extension("txt");
        // use imagePath from labelme so we dont have to do some png jpeg and jpg lookup bullshit
        let orig_image_file =
            PathBuf::from(format!("{}/{}", &orig_path, &read_json_file.imagePath));

        // println!("txt: {:?} , img: {:?}", &orig_txt_file, &orig_image_file);

        match batch {
            "train" => {
                let dest_image = format!(
                    "{}/{}",
                    &export_options.train_img, &read_json_file.imagePath
                );
                let dest_label = PathBuf::from(&export_options.train_label)
                    .join(orig_txt_file.file_name().to_owned().unwrap());
                // println!("orig img: {:?} , orig label: {:?}", &orig_image_file, &orig_txt_file);
                // println!("dest img: {:?} , dest label: {:?}", &dest_image, &dest_label);
                fs::rename(orig_image_file, dest_image)?;
                fs::rename(orig_txt_file, dest_label)?;
            }
            "valid" => {
                let dest_image =
                    format!("{}/{}", &export_options.val_img, &read_json_file.imagePath);
                let dest_label = PathBuf::from(&export_options.val_label)
                    .join(orig_txt_file.file_name().to_owned().unwrap());
                fs::rename(orig_image_file, dest_image)?;
                fs::rename(orig_txt_file, dest_label)?;
            }
            "test" => {
                let dest_image =
                    format!("{}/{}", &export_options.test_img, &read_json_file.imagePath);
                let dest_label = PathBuf::from(&export_options.test_label)
                    .join(orig_txt_file.file_name().to_owned().unwrap());
                fs::rename(orig_image_file, dest_image)?;
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
