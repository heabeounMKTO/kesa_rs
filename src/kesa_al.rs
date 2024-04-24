mod backends;
mod fileutils;
mod image_utils;
mod label;
mod model;
mod output;
mod plotting;
mod splash;
use crate::{
    backends::{candle_backend::CandleModel, compute_backends::InferenceModel},
    fileutils::{get_all_images, write_labelme_to_json},
    label::LabelmeAnnotation,
    model::DatasetInfo,
    output::OutputFormat,
};
use anyhow::{Error, Result};
use backends::compute_backends::{get_backend, ComputeBackendType};
#[cfg(feature = "onnxruntime")]
use backends::onnx_backend::{init_onnx_backend, load_onnx_model};

#[cfg(feature = "torch")]
use backends::tch_backend::TchModel;
#[cfg(feature = "torch")]
use tch::Device::{Cpu, Cuda};

use clap::{ArgAction, Parser,Args };
use fileutils::{open_image, write_yolo_to_txt};
use image::{DynamicImage, GenericImageView};
use indicatif::ProgressBar;
use label::{Embeddings, Shape, YoloAnnotation};
use lazy_static::lazy_static;
use ndarray::{s, ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use plotting::draw_dummy_graph;
use rayon::prelude::*;
use spinners::{Spinner, Spinners};
use splash::print_splash;
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

    #[arg(long, action=ArgAction::SetTrue)]
    /// fp16 inference,
    /// for torch backend
    /// , for GPUs only !
    fp_16: bool,

    #[arg(long)]
    /// compute device
    /// input cpu or 0 1 2 etc for gpus
    device: Option<u32>,

    #[arg(long, num_args(2))]
    /// inference image size of the
    /// model provided
    imgsize: Vec<u32>,

    #[arg(long, action=ArgAction::SetTrue)]
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
    pub static ref IMG_SIZE: (u32, u32) = {
        let args = CliArguments::parse();
        (args.imgsize[0] , args.imgsize[1])
    };
}

fn main() -> Result<(), Error> {
    print_splash();
    println!("Running Autolabeling");
    let args = CliArguments::parse();
    let workers = match &args.workers {
        Some(ref i64) => args.workers,
        None => Some(2),
    };
    #[cfg(feature = "torch")]
    let device: tch::Device = match &args.device {
        Some(ref fuck) => tch::Device::Cuda(args.device.unwrap() as usize),
        None => tch::Device::Cpu,
    };

    let front_sort_dir = format!("{}/front", &args.folder);
    let back_sort_dir = format!("{}/back", &args.folder);

    match &args.sort {
        true => {
            fs::create_dir_all(&front_sort_dir)?;
            fs::create_dir_all(&back_sort_dir)?;
        }
        false => (),
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers.unwrap().try_into().unwrap())
        .build_global()
        .unwrap();
    let all_imgs = get_all_images(&args.folder);
    let model_type: ComputeBackendType = get_backend(&args.weights)?;
    println!(
        "[info]::kesa_al: detected model format : {:#?}",
        &model_type
    );

    match model_type {

        #[cfg(feature = "onnxruntime")]
        ComputeBackendType::OnnxModel => {
            let init_onnx = init_onnx_backend()?;
            let load_model = load_onnx_model(
                &args.weights,
                all_imgs[0].to_owned().to_str().unwrap(),
                false,
                None,
            )?;
            println!("[info]::kesa_al: onnx_model {:#?}", &load_model.model);
            let prog = ProgressBar::new(all_imgs.to_owned().len() as u64);
            all_imgs.par_iter().for_each(|image_path| {
                let orig_img = open_image(&image_path);
                match orig_img {
                    Ok(orig_img) => {
                        let input_img = orig_img.clone();
                        let detections = load_model.run(input_img);
                        match detections {
                            Ok(results) => {
                                process_onnx_results(
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
                                    false => (),
                                }
                            }
                            Err(e) => {
                                match &args.sort {
                                    true => {
                                        move_to_sort_dir(&image_path, &back_sort_dir);
                                    }
                                    false => (),
                                }
                                let _ = e;
                            }
                        }
                        prog.inc(1);
                    }
                    Err(e) => {
                        eprintln!("[error]::kesa_al: cannot open image,\nError: {:?}", e);
                    }
                }
            });
            prog.finish_with_message("\nLabeling done!");
        }
        ComputeBackendType::CandleModel => {
            todo!()
        }
        #[cfg(feature = "torch")]
        ComputeBackendType::TchModel => {
           /*
            *
            * temporary all_class vec            
            * 
           */
        let all_classes = vec![
                    "10C", "10D", "10H", "10S", "2C", "2D", "2H", "2S", "3C", "3D", "3H", "3S", "4C", "4D",
                    "4H", "4S", "5C", "5D", "5H", "5S", "6C", "6D", "6H", "6S", "7C", "7D", "7H", "7S", "8C",
                    "8D", "8H", "8S", "9C", "9D", "9H", "9S", "AC", "AD", "AH", "AS", "JC", "JD", "JH", "JS",
                    "KC", "KD", "KH", "KS", "QC", "QD", "QH", "QS",
                ];
                let _ac: Vec<String> = all_classes.iter().map(|x| String::from(*x)).collect();
        /*********************************************/

            let torch_model = TchModel::new(
                &args.weights,
                IMG_SIZE.0 as i64,
                IMG_SIZE.1 as i64,
                device,
            );
            println!("[info]::kesa_al: torch_model {:#?}", &torch_model);
            match &torch_model.device {
                tch::Device::Cuda(_) => match &args.fp_16 {
                    true => {
                        torch_model.warmup_gpu_fp16()?;
                    }
                    false => {
                        torch_model.warmup_gpu()?;
                    }
                },
                tch::Device::Cpu => {
                    torch_model.warmup()?;
                }
                _ => {
                    todo!()
                }
            }
            let prog = ProgressBar::new(all_imgs.to_owned().len() as u64);
            all_imgs.iter().for_each(|image_path| {
                let orig_img = open_image(&image_path);
                // println!("[info]::kesa_al: imgsize {:?}", orig_img.as_ref().unwrap().dimensions());
                match orig_img {
                    Ok(orig_img) => {
                        match &torch_model.device {
                                tch::Device::Cuda(_) => match &args.fp_16 {
                                    true => {
                                        let preproc_img = image_utils::preprocess_imagef16(&orig_img.clone(), 640)
                                            .expect("[error]::kesa_al: unable to preprocess image");
                                        
                                        let mut _input_tensor = tch::Tensor::try_from(preproc_img)
                                            .expect("[error]::kesa_al: cannot convert preprocessed image to tensor");
                                        
                                        let mut detections = torch_model.run_fp16(&_input_tensor, 0.8, 0.7, "yolov9")
                                            .expect("[error]::kesa_al: error getting detections!");
                                        
                                        let mut shapes_vec: Vec<Shape> = vec![];
                                        println!("-------"); 
                                        println!("detections::raw {:?}", &detections[0]);
                                        println!("detections::noramlzied {:?}", &detections[0].to_normalized(&IMG_SIZE));
                                        println!("-------"); 


                                        // for detection in detections.iter_mut() {
                                        //     let lblme = detection.to_normalized(&IMG_SIZE)
                                        //                          .to_screen(&(orig_img.dimensions().0 , orig_img.dimensions().1))
                                        //                          .to_shape(&_ac, &(orig_img.dimensions().0 , orig_img.dimensions().1)).expect("[error]::kesa_al: conversion to shape failed");
                                        //     shapes_vec.push(lblme);
                                        // }

                                        // let labelme = LabelmeAnnotation::from_shape_vec(image_path.to_owned().to_str().unwrap(), 
                                        //                                                 &orig_img, 
                                        //                                                 &shapes_vec).expect("[error]::kesa_al: cannot convert to labelme");
                                        // println!("debug_lableme: {:?}" , &labelme.shapes);
                                        // println!("debug_lableme: w{:?} h{:?}" , &labelme.imageWidth, &labelme.imageHeight);
                                        // write_labelme_to_json(&labelme, &image_path).expect("[error]::kesa_al: cannot write to labelme");
                                        // println!("detections: {:#?}", &labelme);
                                    },
                                    false => {
                                        let preproc_img = image_utils::preprocess_imagef32(&orig_img.clone(), 640)
                                            .expect("[error]::kesa_al: unable to preprocess image");
                                        
                                        let mut _input_tensor = tch::Tensor::try_from(preproc_img)
                                            .expect("[error]::kesa_al: cannot convert preprocessed image to tensor");
                                        
                                        let mut detections = torch_model.run(&_input_tensor, 0.8, 0.7, "yolov9")
                                            .expect("[error]::kesa_al: error getting detections!");
                                        
                                        let mut shapes_vec: Vec<Shape> = vec![];
                                        println!("-------"); 
                                        println!("detections::raw {:?}", &detections[0]);
                                        println!("detections::noramlzied {:?}", &detections[0].to_normalized(&IMG_SIZE));
                                        println!("-------"); 
                                    }
                                },
                                tch::Device::Cpu => {
                                    todo!()
                                },
                                _ => {
                                    todo!()
                                }

                        }
                    },
                    Err(e) => {
                        eprintln!("[error]::kesa_al: cannot open image,\nError: {:?}", e);
                    }
                }
            });

            prog.finish_with_message("\nLabeling done!");
        }
        _ => panic!("[error]::kesa_al: cannot infer model type!"),
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
fn process_onnx_results(
    image_path: &str,
    results: Embeddings,
    txt: &bool,
    original_image: &DynamicImage,
    all_classes: &Vec<String>,
) -> Result<(), Error> {
    let img_pathbuf = PathBuf::from(&image_path);
    match &txt {
        false => {
            let res_labelme: LabelmeAnnotation = results.to_labelme(
                all_classes,
                &original_image.dimensions(),
                image_path.to_owned().as_str(),
                &original_image,
                &(IMG_SIZE.0, IMG_SIZE.1),
            )?;
            write_labelme_to_json(&res_labelme, &img_pathbuf)?
        }
        true => {
            let res_yolo: Vec<YoloAnnotation> = results.to_yolo_vec()?;
            write_yolo_to_txt(res_yolo, &img_pathbuf)?;
        }
    }
    Ok(())
}