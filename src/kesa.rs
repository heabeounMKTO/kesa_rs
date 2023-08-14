use clap::Subcommand;
use kesa_lib::kesa_utils::file_utils::LabelPortions;
use kesa_lib::kesa_utils::file_utils::get_model_classes_from_yaml;
use kesa_lib::*;

use clap::Parser;
use convert_label::convert::{convert, ConvertTarget};
use kesa_lib::kesa_utils::file_utils::move_labels_to_export_folder;
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
    export_folder: Option<String>,

    #[command(subcommand)]
    label_portions: Option<LabelPortionsOption>,

    #[arg(long)]
    target: String,

    #[arg(long)]
    classes_file: String,
}

#[derive(Debug, Subcommand)]
enum LabelPortionsOption {
    SplitPortions{
        #[arg(long)]
        train: f32,
        
        #[arg(long)]
        valid: f32,
        
        #[arg(long)]
        test: f32,
    },
}


fn main() {
    kesa_splash::print_splash();
    let penis = CliArguments::parse();
   
    let label_classes: HashMap<String, i32> =
        get_model_classes_from_yaml(&penis.classes_file).unwrap();

    let task: KesaTaskType = KesaTaskType::from_str(&penis.task).unwrap();
    
    let export_folder = match &penis.export_folder {
        Some(ref String) => penis.export_folder,
        None => Some(String::from("export")),
    };
    
    let target: ConvertTarget = ConvertTarget::from_str(&penis.target).unwrap();
    
    let portions: LabelPortions = match &penis.label_portions {
       Some(LabelPortionsOption::SplitPortions { train, valid, test }) => {
            LabelPortions::new(train.to_owned(), valid.to_owned(), test.to_owned())
    },
       _ => {
            LabelPortions::new(0.7, 0.25, 0.05)
       }
    };

    println!(
        "Kesa Running: {:?}",
        &task.bright_white().on_bright_purple()
    );
    

    // do ya thing kessy
    match task {
        KesaTaskType::KesaConvert => {
            let convert_setting = KesaConvert::new_convert_setting(
                target,
                label_classes,
                penis.folder,
                export_folder.unwrap(),
            );
            convert(&convert_setting);
            move_labels_to_export_folder(
                &convert_setting.input_folder,
                &convert_setting.export_folder,
                portions
            );
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
