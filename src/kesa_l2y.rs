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

use crate::fileutils::{get_all_classes_hash, get_all_jsons, write_data_yaml, write_yolo_txt};

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
        None => Some(4),
    };
}
