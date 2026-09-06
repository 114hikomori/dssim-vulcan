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

/// Decode a file to 8-bit RGBA (sRGB, non-premultiplied) — the raw input
/// format the GPU path consumes. This calls the **same** `load_image::load_path`
/// as the CPU [`load`], so embedded ICC profiles are applied identically on
/// both paths. (An earlier note here claimed this skipped ICC and "would
/// differ from the CPU path's decoding" — that was wrong; corrected per audit
/// F13.) The only difference from [`load`] is the returned form (raw
/// `ImgVec<RGBAPLU>` for the GPU vs an immediate `DssimImage`) and the
/// pixel-format trait (`to_rgbaplu` vs `to_rgblu`), which is equivalent for the
/// downstream Lab conversion.
pub fn load_image_rgba(path: impl AsRef<Path>) -> Result<ImgVec<RGBAPLU>, load_image::Error> {
    let img = load_image::load_path(path)?;
    match img.bitmap {
        ImageData::RGBA8(ref bitmap) => Ok(Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        ImageData::RGB8(ref bitmap) => Ok(Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        ImageData::GRAY8(ref bitmap) => Ok(Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        ImageData::GRAYA8(ref bitmap) => Ok(Img::new(bitmap.to_rgbaplu(), img.width, img.height)),
        // 16-bit inputs: FULL precision. `to_rgbaplu()` maps each u16 through
        // the 65536-entry linear LUT (dssim-core/src/linear.rs), so the GPU
        // decode path is NOT truncated to 8-bit. BH9: an earlier note here
        // claimed "the GPU path is 8-bit for now" -- false; pinned by
        // load_image_rgba_preserves_16bit_precision.
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


#[cfg(test)]
mod bit_depth {
    /// BH9: the GPU decode path (`load_image_rgba`) must preserve full 16-bit
    /// precision, not truncate to 8-bit. Encode a 16-bit PNG whose red channel
    /// steps by 1 LSB (all four values share the high byte 0x01), decode it, and
    /// assert the f32 outputs are all distinct -- an 8-bit truncation would
    /// collapse them to one value. This pins the claim that the earlier
    /// "GPU path is 8-bit for now" comment got wrong.
    #[test]
    fn load_image_rgba_preserves_16bit_precision() {
        let dir = std::env::temp_dir().join(format!("dssim-16bit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("grad16.png");
        let px: Vec<rgb::RGBA16> = (0x0100u16..0x0104)
            .map(|r| rgb::RGBA16::new(r, 0x8000, 0x4000, 0xFFFF))
            .collect();
        lodepng::encode_file(&path, &px, 4, 1, lodepng::ColorType::RGBA, 16).unwrap();

        let decoded = crate::load_image_rgba(&path).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (4, 1));
        let rs: Vec<f32> = decoded.pixels().map(|p| p.r).collect();
        for i in 1..rs.len() {
            assert!(
                rs[i] != rs[i - 1],
                "16-bit red values collapsed to 8-bit: {rs:?}"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
