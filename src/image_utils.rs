use anyhow::{Error, Result};
use image::io::Reader;
use image::DynamicImage;
use std::io::Cursor;
use base64::{
    engine::{self, general_purpose},
    Engine as _,
};
use half::f16;
use image::{imageops::FilterType,  GenericImageView, ImageBuffer, ImageFormat};
use ndarray::{s, Array, ArrayBase, Axis, Dim, IxDyn, OwnedRepr};
use std::path::Path;
/// handles opening images ,
/// avoids crashing
/// if image failed to open
/// for whatever Reason
pub fn open_image(image_path: &str) -> Result<DynamicImage, Error> {
    let _open_image = image::open(image_path)?;
    Ok(_open_image)
}


/* stolen my own code
* yet agian lmaooo */

/// preprocess an image for detection,
/// casted to fp16 for gpu inferences
/// # Arguments
/// * params:
///     - `image_source` : a image::DynamicImage loaded thru whatever method
///     - `target_size`: target image resize
pub fn preprocess_imagef16(
    image_source: &DynamicImage,
    target_size: u32,
) -> Result<ArrayBase<OwnedRepr<f16>, Dim<[usize; 4]>>, Error> {
    let img = image_source.resize_exact(target_size, target_size, FilterType::Triangle);
    let mut _dummy_input: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> =
        Array::ones((1, 3, target_size as usize, target_size as usize));
    let mut _dummy_input = _dummy_input.mapv(f16::from_f32);
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        _dummy_input[[0, 0, y, x]] = f16::from_f32(r as f32 / 255.0);
        _dummy_input[[0, 1, y, x]] = f16::from_f32(g as f32 / 255.0);
        _dummy_input[[0, 2, y, x]] = f16::from_f32(b as f32 / 255.0);
    }
    Ok(_dummy_input)
}

/// preprocess an image for detection
/// # Arguments
/// * params:
///     - `image_source` : a image::DynamicImage loaded thru whatever method
///     - `target_size`: target image resize
pub fn preprocess_imagef32(
    image_source: &DynamicImage,
    target_size: u32,
) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>, Error> {
    let img = image_source.resize_exact(target_size, target_size, FilterType::Triangle);
    let mut _dummy_input: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> =
        Array::ones((1, 3, target_size as usize, target_size as usize));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        _dummy_input[[0, 0, y, x]] = r as f32 / 255.0;
        _dummy_input[[0, 1, y, x]] = g as f32 / 255.0;
        _dummy_input[[0, 2, y, x]] = b as f32 / 255.0;
    }
    Ok(_dummy_input)
}

// pub fn ndarray_dynamic_img<T: image::Primitive>(ndarray: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>) -> image::DynamicImage where T: Clone,  {
//         {
//             let (width, height, _, _) = ndarray.dim();
//             let mut img_buffer = ImageBuffer::new(width as u32, height as u32);

//             for (x, y, pixel) in img_buffer.enumerate_pixels_mut() {
//                 let color: [T; 3] = ndarray[[x as usize, y as usize, ..]].into();
//                 *pixel = image::Rgb(color);
//             }
//             DynamicImage::ImageRgb8(img_buffer)
//     }
// }
//

pub fn dynimg2string(input_image: &DynamicImage) -> Result<String, Error> {
    let mut image_data: Vec<u8> = Vec::new();
    input_image
        .write_to(&mut Cursor::new(&mut image_data), ImageFormat::Jpeg)
        .unwrap();
    let resb64 = general_purpose::STANDARD.encode(image_data);
    Ok(resb64)
}
