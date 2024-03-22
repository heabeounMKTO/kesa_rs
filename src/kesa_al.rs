mod backends;
mod fileutils;
mod image_utils;
mod label;
mod model;
mod output;
mod plotting;
mod splash;
use splash::print_splash;
use crate::{
    backends::{
        candle_backend::CandleModel, compute_backends::InferenceModel, tch_backend::TchModel,
    },
    fileutils::{get_all_images, write_labelme_to_json},
    label::LabelmeAnnotation,
    model::DatasetInfo,
    output::OutputFormat,
};
use anyhow::{Error, Result};
use backends::{
    compute_backends::{get_backend, ComputeBackendType},
    onnx_backend::{init_onnx_backend, load_onnx_model},
};
use clap::{ArgAction, Parser};
use fileutils::{open_image, write_yolo_to_txt};
use image::{DynamicImage, GenericImageView};
use indicatif::ProgressBar;
use label::{Embeddings, Shape, YoloAnnotation};
use lazy_static::lazy_static;
use ndarray::{s, ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use plotting::draw_dummy_graph;
use rayon::prelude::*;
use spinners::{Spinner, Spinners};
use std::fs::{self, File};
use std::io::BufReader;
use std::io::{Read, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct CliArguments {
    #[arg(long)]
    /// target folder
    ///
    folder: String,

    #[arg(long, action=ArgAction::SetTrue)]
    /// moves empty images to a separate folder
    ///
    sort: bool,

    #[arg(long, action=ArgAction::SetFalse)]
    /// fp16 inference , for GPUs only !
    fp_16: bool,

    #[arg(long)]
    /// inference image size of the
    /// model provided
    imgsize: u32,

    // SetFlase means true :|
    #[arg(long, action=ArgAction::SetFalse)]
    /// if supplied ,
    /// outputs yolo txt files
    /// instead of LabelMe jsons
    txt: bool,

    #[arg(long)]
    /// amount of threads to use
    /// defaults to
    /// 2
    /// puttin it on high thread count will
    /// cause allocation errors on gpu, if you are using gpu
    workers: Option<i64>,

    #[arg(long)]
    /// weights to be used
    weights: String,
}

lazy_static! {
    pub static ref IMG_SIZE: u32 = {
        let args = CliArguments::parse();
        let imgsz = args.imgsize;
        imgsz
    };
}

fn main() -> Result<(), Error> {
    print_splash();
    println!("Running Autolabeling");
    let init_onnx = init_onnx_backend()?;
    let args = CliArguments::parse();
    let workers = match &args.workers {
        Some(ref i64) => args.workers,
        None => Some(2),
    };

    let front_sort_dir = format!("{}/front", &args.folder);
    let back_sort_dir = format!("{}/back", &args.folder);

    match &args.sort {
        true => {
            fs::create_dir_all(&front_sort_dir)?;
            fs::create_dir_all(&back_sort_dir)?;
        }
        false => println!(""),
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers.unwrap().try_into().unwrap())
        .build_global()
        .unwrap();
    let all_imgs = get_all_images(&args.folder);
    let model_type: ComputeBackendType = get_backend(&args.weights)?;
    println!("Detected model format : {:#?}", &model_type);
    match model_type {
        ComputeBackendType::OnnxModel => {
            let load_model = load_onnx_model(
                &args.weights,
                all_imgs[0].to_owned().to_str().unwrap(),
                false,
                None,
            )?;
            println!("leme get a uhh : {:?}", &load_model.model);
            let prog = ProgressBar::new(all_imgs.to_owned().len() as u64);
            all_imgs.par_iter().for_each(|image_path| {
                let orig_img = open_image(&image_path);
                match orig_img {
                    Ok(orig_img) => {
                        let input_img = orig_img.clone();
                        let detections = load_model.run(input_img);
                        match detections {
                            Ok(results) => {
                                process_results(
                                    image_path.to_owned().to_str().unwrap(),
                                    results,
                                    &args.txt,
                                    &orig_img,
                                    &load_model.model_details.names,
                                )
                                .unwrap();
                                // move file if sort
                                match &args.sort {
                                    true => {
                                        move_to_sort_dir(&image_path, &front_sort_dir);
                                    }
                                    false => println!("skipping sorting"),
                                }
                            }
                            Err(e) => {
                                match &args.sort {
                                    true => {
                                        move_to_sort_dir(&image_path, &back_sort_dir);
                                    }
                                    false => println!("skipping sorting"),
                                }
                                let _ = e;
                            }
                        }
                        prog.inc(1);
                    }
                    Err(e) => {
                        eprintln!("cannot open image,\nError: {:?}", e);
                    }
                }
            });
            prog.finish_with_message("\nLabeling done!");
        }
        ComputeBackendType::CandleModel => {
            todo!()
        }
        ComputeBackendType::TchModel => {
            todo!()
        }
    };
    // draw_dummy_graph();
    Ok(())
}

/// moves a image in the form of a 
/// *&PathBuf* to a dir provided as a string
fn move_to_sort_dir(image_path: &PathBuf, sorting_dir: &str) {
    let original_img_file = image_path.to_owned();
    let mut original_json_file = image_path.to_owned();
    original_json_file.set_extension("json");
    let sorted_img_file =
        PathBuf::from(sorting_dir).join(image_path.file_name().to_owned().unwrap());
    let mut sorted_json_file = sorted_img_file.to_owned();
    sorted_json_file.set_extension("json");
    fs::rename(original_img_file, sorted_img_file);
    fs::rename(original_json_file, sorted_json_file);
}

/// process Embeddings (a ndarray) output
// TODO: refactor:: input_image to pathbuf or &str
fn process_results(
    image_path: &str,
    results: Embeddings,
    txt: &bool,
    original_image: &DynamicImage,
    all_classes: &Vec<String>,
) -> Result<(), Error> {
    let img_pathbuf = PathBuf::from(&image_path);
    match &txt {
        true => {
            let res_labelme: LabelmeAnnotation = results.to_labelme(
                all_classes,
                &original_image.dimensions(),
                image_path.to_owned().as_str(),
                &original_image,
                &(*IMG_SIZE, *IMG_SIZE),
            )?;
            write_labelme_to_json(&res_labelme, &img_pathbuf)?
        }
        false => {
            let res_yolo: Vec<YoloAnnotation> = results.to_yolo_vec()?;
            write_yolo_to_txt(res_yolo, &img_pathbuf)?;
        }
    }
    Ok(())
}
