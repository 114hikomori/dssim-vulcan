//! Multi-scale GPU DSSIM orchestration + CPU f64 pooling (plan §6 Phase E /
//! VULKAN_PORT_PLAN §4 Phase 5).
//!
//! Pipeline order per scale, exactly as the CPU implementation
//! (`make_scales_recursive` + `compare_inner`): the current RGBAPLU scale is
//! converted to Lab planes on the CPU (hybrid strategy — Phase F moves this),
//! its statistics are computed on the GPU, the SSIM map comes back and is
//! pooled here in f64, and the RGBAPLU image is 2×2 box-downsampled on the
//! CPU for the next scale.
//!
//! Pooling (`DEFAULT_WEIGHTS`, power term, MAD, `to_dssim`) is transcribed
//! from dssim.rs:85,299-326,439 rather than extracted: extraction would
//! restructure the AGPL production path for the GPU's benefit, and the
//! transcription is pinned by the locked-value tests in tests/phase_e.rs.

use std::sync::Arc;

use imgref::ImgVec;

use crate::blur::BlurPipelines;
use crate::context::Context;
use crate::ssim::{ssim_combine_pipelines, SsimPipelines};
use crate::Result;

// `Downsample` is trait-scoped on dssim-core's image types.
use dssim_core::Downsample as _;

/// Pooling weights, transcribed from dssim.rs:85 (`DEFAULT_WEIGHTS`).
pub const DEFAULT_WEIGHTS: [f64; 5] = [0.028, 0.197, 0.322, 0.298, 0.155];

/// Final DSSIM conversion, transcribed from dssim.rs:439.
pub fn to_dssim(ssim: f64) -> f64 {
    1.0 / ssim.max(f64::EPSILON) - 1.0
}

/// Per-scale pooling, transcribed from dssim.rs:299-302 (f64, power term,
/// mean absolute deviation). `scale_n` is the post-reverse scale index
/// (0 = original image).
pub fn pool_scale(map: &[f32], width: usize, height: usize, scale_n: usize) -> f64 {
    let sum = map.iter().fold(0.0f64, |s, i| s + f64::from(*i));
    let len = (width * height) as f64;
    let avg = (sum / len).max(0.0).powf((0.5f64).powf(scale_n as f64));
    1.0 - map.iter().fold(0.0f64, |s, i| s + (avg - f64::from(*i)).abs()) / len
}

/// A GPU-side analog of `dssim_core::Dssim`: holds the compiled pipelines
/// once and drives the multi-scale comparison. Reuses the CPU path for
/// Lab conversion, downsampling, and the score formula.
pub struct GpuSsim {
    /// Kept alive: the pipelines' descriptor machinery borrows this device.
    #[allow(dead_code)]
    context: Arc<Context>,
    blur: BlurPipelines,
    ssim: SsimPipelines,
}

impl GpuSsim {
    pub fn new(context: Arc<Context>) -> Result<Self> {
        let blur = BlurPipelines::new(&context)?;
        let ssim = SsimPipelines::new(&context)?;
        Ok(Self { context, blur, ssim })
    }

    /// Materialize all pyramid scales of one image: per scale, GPU Lab
    /// conversion + GPU statistics, then CPU 2×2 downsample for the next.
    ///
    /// Scale-count semantics replicate dssim-core exactly: one scale per
    /// weight (plus the one the CPU generates but never uses — it is dropped
    /// by the weight zip in `compare_inner`, so it is simply not generated
    /// here), stopping when `Downsample` returns None (w<8 || h<8).
    pub fn create_image(&self, src: &ImgVec<dssim_core::RGBAPLU>) -> Result<GpuSsimImage> {
        let mut scales: Vec<ScaleData> = Vec::new();
        let mut current: Option<ImgVec<dssim_core::RGBAPLU>> = Some(src.clone());

        for _ in 0..DEFAULT_WEIGHTS.len() {
            let img = match current.take() {
                Some(img) => img,
                None => break,
            };
            let (w, h) = (img.width(), img.height());
            let mut inter = Vec::with_capacity(w * h * 4);
            for px in img.pixels() {
                inter.extend_from_slice(&[px.r, px.g, px.b, px.a]);
            }
            let planes = crate::color::rgba_to_lab_gpu(&self.context, &inter, w, h, 3)?;
            scales.push(self.make_scale(&planes, w, h, 3)?);
            current = img.downsample();
        }

        Ok(GpuSsimImage { scales })
    }

    /// Gray (1-channel) variant: linear-light f32 planes, matching
    /// `GBitmap::to_lab` (the ×1.16 branch).
    pub fn create_image_gray(&self, src: &ImgVec<f32>) -> Result<GpuSsimImage> {
        let mut scales: Vec<ScaleData> = Vec::new();
        let mut current: Option<ImgVec<f32>> = Some(src.clone());

        for _ in 0..DEFAULT_WEIGHTS.len() {
            let img = match current.take() {
                Some(img) => img,
                None => break,
            };
            let (w, h) = (img.width(), img.height());
            let input: Vec<f32> = img.pixels().collect();
            let planes = crate::color::rgba_to_lab_gpu(&self.context, &input, w, h, 1)?;
            scales.push(self.make_scale(&planes, w, h, 1)?);
            current = img.downsample();
        }

        Ok(GpuSsimImage { scales })
    }

    /// GPU statistics for one scale's Lab planes, replicating
    /// `DssimChan::preprocess` (dssim.rs:118-139): chroma pre-blur in place
    /// (luma untouched), then mu = blur(img), sq_blur = blur_mul(img, img).
    fn make_scale(&self, planes: &[f32], w: usize, h: usize, channels: usize) -> Result<ScaleData> {
        let pixels = w * h;
        let mut img_all = Vec::with_capacity(pixels * channels);
        let mut mu_all = Vec::with_capacity(pixels * channels);
        let mut sq_all = Vec::with_capacity(pixels * channels);
        for c in 0..channels {
            let mut data = planes[c * pixels..(c + 1) * pixels].to_vec();
            if c > 0 {
                // Chroma pre-blur: the CPU mutates img in place via
                // blur_in_place; blur() is arithmetically identical
                // (blur.rs tests assert src2 == dst), so overwrite.
                data = self.blur.blur(&data, w, h, w)?;
            }
            let mu = self.blur.blur(&data, w, h, w)?;
            let sq = self.blur.blur_mul(&data, &data, w, h, w, w)?;
            img_all.extend(data);
            mu_all.extend(mu);
            sq_all.extend(sq);
        }
        Ok(ScaleData {
            width: w,
            height: h,
            channels,
            img: img_all,
            mu: mu_all,
            sq_blur: sq_all,
        })
    }

    /// Compare a reference image with a modified one; returns the final
    /// DSSIM score. Mirrors `Dssim::compare` + `compare_inner`:
    /// per scale, cross = blur_mul(ref.img, mod.img) on the preprocessed
    /// planes, SSIM map on GPU, pooled here in f64 with the scale weights.
    pub fn compare(&self, reference: &GpuSsimImage, modified: &GpuSsimImage) -> Result<f64> {
        let mut ssim_sum = 0.0f64;
        let mut weight_sum = 0.0f64;

        for (n, (weight, (ref_scale, mod_scale))) in DEFAULT_WEIGHTS
            .iter()
            .zip(reference.scales.iter().zip(modified.scales.iter()))
            .enumerate()
        {
            let pixels = ref_scale.width * ref_scale.height;
            let channels = ref_scale.channels;
            assert_eq!(channels, mod_scale.channels, "channel count mismatch");

            let mut cross_all = Vec::with_capacity(channels * pixels);
            for c in 0..channels {
                let cross = self.blur.blur_mul(
                    &ref_scale.img[c * pixels..(c + 1) * pixels],
                    &mod_scale.img[c * pixels..(c + 1) * pixels],
                    ref_scale.width,
                    ref_scale.height,
                    ref_scale.width,
                    mod_scale.width,
                )?;
                cross_all.extend(cross);
            }

            let map = ssim_combine_pipelines(
                &self.ssim,
                &ref_scale.mu,
                &ref_scale.sq_blur,
                &mod_scale.mu,
                &mod_scale.sq_blur,
                &cross_all,
                ref_scale.width,
                ref_scale.height,
                channels,
            )?;

            let pooled = pool_scale(&map, ref_scale.width, ref_scale.height, n);
            ssim_sum = pooled.mul_add(*weight, ssim_sum);
            weight_sum += weight;
        }

        Ok(to_dssim(ssim_sum / weight_sum))
    }
}

/// One pyramid scale of statistics for one image.
struct ScaleData {
    /// Preprocessed channel planes (post chroma pre-blur), tightly packed —
    /// the CPU cross-blur reads exactly these (`DssimChan.img`).
    img: Vec<f32>,
    mu: Vec<f32>,
    sq_blur: Vec<f32>,
    width: usize,
    height: usize,
    channels: usize,
}

/// A GPU-processed image — analog of `dssim_core::DssimImage`.
pub struct GpuSsimImage {
    scales: Vec<ScaleData>,
}

impl GpuSsimImage {
    pub fn width(&self) -> usize {
        self.scales[0].width
    }

    pub fn height(&self) -> usize {
        self.scales[0].height
    }
}
