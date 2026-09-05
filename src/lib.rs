//! See the [dssim-core](https://lib.rs/dssim-core) crate if you'd like to use only the library part.
#![doc(html_logo_url = "https://kornel.ski/dssim/logo.png")]
#![allow(clippy::manual_range_contains)]

pub use dssim_core::*;
use imgref::{Img, ImgVec};
use load_image::ImageData;
use std::path::Path;

fn load(attr: &Dssim, path: &Path) -> Result<DssimImage<f32>, load_image::Error> {
    let img = load_image::load_path(path)?;
    Ok(match img.bitmap {
        ImageData::RGB8(ref bitmap) => attr.create_image(&Img::new(bitmap.to_rgblu(), img.width, img.height)),
        ImageData::RGB16(ref bitmap) => attr.create_image(&Img::new(bitmap.to_rgblu(), img.width, img.height)),
        ImageData::RGBA8(ref bitmap) => attr.create_image(&Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        ImageData::RGBA16(ref bitmap) => attr.create_image(&Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        ImageData::GRAY8(ref bitmap) => attr.create_image(&Img::new(bitmap.to_rgblu(), img.width, img.height)),
        ImageData::GRAY16(ref bitmap) => attr.create_image(&Img::new(bitmap.to_rgblu(), img.width, img.height)),
        ImageData::GRAYA8(ref bitmap) => attr.create_image(&Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        ImageData::GRAYA16(ref bitmap) => attr.create_image(&Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
    }.expect("infallible"))
}

/// Decode a file to 8-bit RGBA (sRGB, non-premultiplied) **without** color
/// profile conversion — the raw input format the GPU path consumes. Note:
/// unlike [`load_image`], this skips ICC profile handling; images with
/// non-sRGB profiles will differ from the CPU path's decoding.
pub fn load_image_rgba(path: impl AsRef<Path>) -> Result<ImgVec<RGBAPLU>, load_image::Error> {
    let img = load_image::load_path(path)?;
    match img.bitmap {
        ImageData::RGBA8(ref bitmap) => Ok(Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        ImageData::RGB8(ref bitmap) => Ok(Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        ImageData::GRAY8(ref bitmap) => Ok(Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        ImageData::GRAYA8(ref bitmap) => Ok(Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        // 16-bit inputs: the GPU path is 8-bit for now (u16 LUT is a
        // profiling-justified extension, plan §4 Phase 6).
        ImageData::RGBA16(ref bitmap) => Ok(Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        ImageData::RGB16(ref bitmap) => Ok(Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        ImageData::GRAY16(ref bitmap) => Ok(Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        ImageData::GRAYA16(ref bitmap) => Ok(Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
    }
}

/// Load PNG or JPEG image from the given path. Applies color profiles and converts to `sRGB`.
#[inline]
pub fn load_image(attr: &Dssim, path: impl AsRef<Path>) -> Result<DssimImage<f32>, load_image::Error> {
    load(attr, path.as_ref())
}
