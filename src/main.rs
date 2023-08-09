mod convert_label;
mod kesa_utils;
mod yolo;
use clap::Parser;
use convert_label::convert::{convert, ConvertSettings, ConvertTarget};
use std::{env::args, fs::File};
use anyhow::bail;
use crate::kesa_utils::file_utils::get_model_classes_from_yaml;

#[derive(Parser, Debug)]
struct CliArguments {
    #[arg(short, long)]
    folder: String,

    #[arg(short, long)]
    target: String,

    #[arg(short, long)]
    classes_file: String,
}

fn main() {
    println!("Hello, world!");
    let penis = CliArguments::parse();
    let label_classes: Vec<String> = get_model_classes_from_yaml(&penis.classes_file).unwrap();
    let target: anyhow::Result<ConvertTarget> = match penis.target.to_lowercase().as_str() {
       "yolo" => Ok(ConvertTarget::Yolo),
       "coco" => Ok(ConvertTarget::Coco),
       "pascal" => Ok(ConvertTarget::Pascal),
       _ => Err("format not recognized!")
    };
    
    let settings: ConvertSettings =
        ConvertSettings::new(ConvertTarget::Yolo, label_classes, penis.folder);
    convert(settings);
}
