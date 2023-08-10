mod convert_label;
mod kesa_utils;
mod label;
mod yolo;
use crate::kesa_utils::file_utils::get_model_classes_from_yaml;
use anyhow::Error;
use anyhow::Result;
use clap::Parser;
use convert_label::convert::{convert, ConvertSettings, ConvertTarget};
use kesa_utils::kesa_splash;
use std::str::FromStr;
use std::{env::args, fs::File};


#[derive(Parser, Debug)]
struct CliArguments {
    #[arg(long)]
    task: String,
    
    #[arg(long)]
    folder: String,

    #[arg(long)]
    target: String,

    #[arg(long)]
    classes_file: String,
}

fn main() {
    kesa_splash::print_splash();    
    let penis = CliArguments::parse();
    let label_classes: Vec<String> = get_model_classes_from_yaml(&penis.classes_file).unwrap();
    let target: ConvertTarget = ConvertTarget::from_str(&penis.target).unwrap();
    let settings: ConvertSettings = ConvertSettings::new(target, label_classes, penis.folder);
    convert(settings);
}
