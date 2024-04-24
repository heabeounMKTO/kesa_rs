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
use rand::distributions::{Distribution, Uniform};
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
}

fn main() -> Result<(), Error> {
    print_splash();
    let args = CliArguments::parse();
}
