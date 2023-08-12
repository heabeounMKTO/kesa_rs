use kesa_lib::*;
use kesa_lib::kesa_utils::file_utils::get_model_classes_from_yaml;

use clap::Parser;
use convert_label::convert::{convert, ConvertTarget};
use kesa_utils::kesa_splash;

use kesa_utils::kesa_task::KesaConvert;
use kesa_utils::kesa_task::KesaTaskType;
use owo_colors::OwoColorize;
use std::collections::HashMap;
use std::str::FromStr;

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
    let label_classes: HashMap<String, i32> =
        get_model_classes_from_yaml(&penis.classes_file).unwrap();
    let task: KesaTaskType = KesaTaskType::from_str(&penis.task).unwrap();
    let target: ConvertTarget = ConvertTarget::from_str(&penis.target).unwrap();
    println!(
        "Kesa Running: {:?}",
        &task.bright_white().on_bright_purple()
    );

    // do ya thing kessy
    match task {
        KesaTaskType::KesaConvert => {
            let convert_setting =
                KesaConvert::new_convert_setting(target, label_classes, penis.folder);
            convert(convert_setting);
        }
        KesaTaskType::KesaLabel => {
            todo!()
        }
        KesaTaskType::KesaAugment => {
            todo!()
        }
        _ => println!("unknown task type!"),
    };
}
