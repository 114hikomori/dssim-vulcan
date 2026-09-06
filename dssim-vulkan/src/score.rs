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
use crate::pipeline::{dispatch_sequence, Pass};
use crate::ssim::SsimPipelines;
use crate::transfer::Buffer;
use crate::Result;

use ash::vk;

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
    color: crate::color::ColorPipelines,
    ssim: SsimPipelines,
}

impl GpuSsim {
    pub fn new(context: Arc<Context>) -> Result<Self> {
        let blur = BlurPipelines::new(&context)?;
        let color = crate::color::ColorPipelines::new(&context)?;
        let ssim = SsimPipelines::new(&context)?;
        Ok(Self { context, blur, color, ssim })
    }

    /// Materialize the whole pyramid in ONE GPU submit: per scale, staging
    /// copy -> GPU Lab -> per-channel pre-blur/mu/sq_blur chains (all
    /// GPU-resident; results stay on the device for `compare`). The CPU only
    /// interleaves the input planes and runs the 2x2 downsample between
    /// scales. Scale-count semantics replicate dssim-core exactly: one scale
    /// per weight, stopping when `Downsample` returns None (w<8 || h<8).
    pub fn create_image(&self, src: &ImgVec<dssim_core::RGBAPLU>) -> Result<GpuSsimImage> {
        let mut current: Option<ImgVec<dssim_core::RGBAPLU>> = Some(src.clone());

        let mut passes: Vec<Pass> = Vec::new();

        let mut keep: Vec<ScaleKeep> = Vec::new();

        for _ in 0..DEFAULT_WEIGHTS.len() {
            let img = match current.take() {
                Some(img) => img,
                None => break,
            };
            let (w, h) = (img.width(), img.height());
            let pixels = w * h;

            // Staging: interleaved RGBA (CPU-owned data, one copy to device).
            let mut inter = Vec::with_capacity(pixels * 4);
            for px in img.pixels() {
                inter.extend_from_slice(&[px.r, px.g, px.b, px.a]);
            }
            let staging = self.context.alloc_buffer(
                "s.staging",
                (inter.len() * 4) as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                gpu_allocator::MemoryLocation::CpuToGpu,
            )?;
            crate::transfer::write_mapped(&staging, &pack_f32(&inter))?;
            let rgba_buf = self.context.alloc_buffer(
                "s.rgba",
                (inter.len() * 4) as u64,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            passes.push(Pass::CopyBuffer { src: staging.clone(), dst: rgba_buf.clone() });

            // img_all: Lab planes written straight in by the shader (plane 0 =
            // raw L, planes 1,2 = chroma, pre-blurred in place below); plane
            // stride = width. There is deliberately no separate `lab` buffer
            // and no lab->img copy: the old copy lacked TRANSFER_SRC/DST usage
            // on both buffers (F25, a spec violation that only survived because
            // validation was off). Writing plane 0 directly removes the copy.
            let img_all = self.context.alloc_buffer(
                "s.img",
                (pixels * 3 * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            self.color.lab_into(&mut passes, rgba_buf.clone(), img_all.clone(), w, h, 3);
            let tmp = self.context.alloc_buffer(
                "s.tmp",
                (pixels * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            for c in 1..3u32 {
                let off = c * pixels as u32;
                self.blur.h5_into(&mut passes, img_all.clone(), tmp.clone(), w, h, w, off, 0);
                self.blur.v5_into(&mut passes, tmp.clone(), img_all.clone(), w, h, off);
            }

            // mu (blur of img) and sq (blur_mul(img, img)) per channel. Both
            // stay device-local: `compare` reads them only on-device via the
            // combine (F27 — mu was host-visible for no reason, wasting the
            // scarce BAR-resident pool on dGPUs).
            let mu_all = self.context.alloc_buffer(
                "s.mu",
                (pixels * 3 * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            let sq_all = self.context.alloc_buffer(
                "s.sq",
                (pixels * 3 * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            for c in 0..3u32 {
                let off = c * pixels as u32;
                self.blur.h5_into(&mut passes, img_all.clone(), tmp.clone(), w, h, w, off, 0);
                self.blur.v5_into(&mut passes, tmp.clone(), mu_all.clone(), w, h, off);
                self.blur.h5_mul_into(&mut passes, img_all.clone(), img_all.clone(), tmp.clone(), w, h, w, off, off, 0);
                self.blur.v5_into(&mut passes, tmp.clone(), sq_all.clone(), w, h, off);
            }

            keep.push(ScaleKeep {
                width: w,
                height: h,
                channels: 3,
                img: img_all,
                mu: mu_all,
                sq: sq_all,
            });
            current = img.downsample();
        }

        dispatch_sequence(&self.context, &passes)?;


        Ok(GpuSsimImage {
            scales: keep
                .into_iter()
                .map(|k| ScaleData {
                    width: k.width,
                    height: k.height,
                    channels: k.channels,
                    img: k.img,
                    mu: k.mu,
                    sq_blur: k.sq,
                })
                .collect(),
        })
    }

    /// Gray (1-channel) variant: linear-light f32 planes, matching
    /// `GBitmap::to_lab` (the x1.16 branch). One GPU submit for all scales.
    pub fn create_image_gray(&self, src: &ImgVec<f32>) -> Result<GpuSsimImage> {
        let mut passes: Vec<Pass> = Vec::new();
        
        let mut keep: Vec<ScaleKeep> = Vec::new();
        let mut current: Option<ImgVec<f32>> = Some(src.clone());

        for _ in 0..DEFAULT_WEIGHTS.len() {
            let img = match current.take() {
                Some(img) => img,
                None => break,
            };
            let (w, h) = (img.width(), img.height());
            let pixels = w * h;

            let input: Vec<f32> = img.pixels().collect();
            let staging = self.context.alloc_buffer(
                "g.staging",
                (input.len() * 4) as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                gpu_allocator::MemoryLocation::CpuToGpu,
            )?;
            crate::transfer::write_mapped(&staging, &pack_f32(&input))?;
            let gray_buf = self.context.alloc_buffer(
                "g.gray",
                (pixels * 4) as u64,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            passes.push(Pass::CopyBuffer { src: staging.clone(), dst: gray_buf.clone() });

            let lab = self.context.alloc_buffer(
                "g.lab",
                (pixels * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            self.color.lab_into(&mut passes, gray_buf.clone(), lab.clone(), w, h, 1);

            let mu = self.context.alloc_buffer(
                "g.mu",
                (pixels * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            let sq = self.context.alloc_buffer(
                "g.sq",
                (pixels * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            let tmp = self.context.alloc_buffer(
                "g.tmp",
                (pixels * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            self.blur.h5_into(&mut passes, lab.clone(), tmp.clone(), w, h, w, 0, 0);
            self.blur.v5_into(&mut passes, tmp.clone(), mu.clone(), w, h, 0);
            self.blur.h5_mul_into(&mut passes, lab.clone(), lab.clone(), tmp.clone(), w, h, w, 0, 0, 0);
            self.blur.v5_into(&mut passes, tmp.clone(), sq.clone(), w, h, 0);

            keep.push(ScaleKeep { width: w, height: h, channels: 1, img: lab, mu, sq });
            current = img.downsample();
        }

        dispatch_sequence(&self.context, &passes)?;

        Ok(GpuSsimImage {
            scales: keep
                .into_iter()
                .map(|k| ScaleData {
                    width: k.width,
                    height: k.height,
                    channels: k.channels,
                    img: k.img,
                    mu: k.mu,
                    sq_blur: k.sq,
                })
                .collect(),
        })
    }

    /// Compare a reference image with a modified one; returns the final
    /// DSSIM score. ONE GPU submit for all scales: per scale, the three
    /// channel cross-blurs and the SSIM combine run on the GPU-resident
    /// planes; only the per-scale SSIM maps come back for CPU pooling in f64.
    /// Note the scale-0 map is full-resolution (≈16 MB at 2048²; ≈4/3·P0·4B
    /// summed over scales), so this is not a negligible transfer — pooling on
    /// the GPU is a standing non-goal (AGENTS.md §8) pending profiling.
    pub fn compare(&self, reference: &GpuSsimImage, modified: &GpuSsimImage) -> Result<f64> {
        let mut passes: Vec<Pass> = Vec::new();

        let mut map_readbacks: Vec<(Buffer, usize, usize, usize)> = Vec::new(); // (rb, w, h, scale_n)

        for (n, (ref_scale, mod_scale)) in reference
            .scales
            .iter()
            .zip(modified.scales.iter())
            .enumerate()
        {
            let pixels = ref_scale.width * ref_scale.height;
            let channels = ref_scale.channels;
            assert_eq!(channels, mod_scale.channels, "channel count mismatch");
            // F29: the cross-blur (`h5_mul_into`) derives src2's stride from
            // src1's, so a ref/mod shape mismatch reads out of bounds on the
            // GPU instead of the old CPU slice panic. The CLI guards sizes
            // upstream, but this is a public library API — assert here too.
            assert_eq!(
                (ref_scale.width, ref_scale.height),
                (mod_scale.width, mod_scale.height),
                "reference and modified scales must have matching dimensions"
            );

            let tmp = self.context.alloc_buffer(
                "c.tmp",
                (pixels * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            let cross_all = self.context.alloc_buffer(
                "c.cross",
                (pixels * channels * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;

            for c in 0..channels as u32 {
                let off = c * pixels as u32;
                self.blur.h5_mul_into(&mut passes, ref_scale.img.clone(), mod_scale.img.clone(), tmp.clone(), ref_scale.width, ref_scale.height, ref_scale.width, off, off, 0);
                self.blur.v5_into(&mut passes, tmp.clone(), cross_all.clone(), ref_scale.width, ref_scale.height, off);
            }

            let map_dst = self.context.alloc_buffer(
                "c.map",
                (pixels * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            let map_rb = self.context.alloc_buffer(
                "c.map_rb",
                (pixels * 4) as u64,
                vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
            )?;
            self.ssim.combine_into(
                &mut passes,
                ref_scale.mu.clone(),
                ref_scale.sq_blur.clone(),
                mod_scale.mu.clone(),
                mod_scale.sq_blur.clone(),
                cross_all.clone(),
                map_dst.clone(),
                ref_scale.width,
                ref_scale.height,
                channels,
            );

            passes.push(Pass::CopyBuffer { src: map_dst.clone(), dst: map_rb.clone() });
            map_readbacks.push((map_rb, ref_scale.width, ref_scale.height, n));
        }

        dispatch_sequence(&self.context, &passes)?;

        // Pool on CPU from the downloaded maps (order: scale n ascending —
        // map_readbacks was built in scale order).
        let mut ssim_sum = 0.0f64;
        let mut weight_sum = 0.0f64;
        for (rb, w, h, n) in &map_readbacks {
            let map = crate::transfer::read_bytes(rb, w * h * 4)?;
            let map: Vec<f32> = map
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
                let pooled = pool_scale(&map, *w, *h, *n);
                ssim_sum = pooled.mul_add(DEFAULT_WEIGHTS[*n], ssim_sum);
            weight_sum += DEFAULT_WEIGHTS[*n];
        }
        

        Ok(to_dssim(ssim_sum / weight_sum))
    }
}

fn pack_f32(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// One pyramid scale of statistics for one image. All buffers hold
/// `channels` concatenated planes (plane stride = width) and stay on the
/// device between `create_image` and `compare`.
struct ScaleData {
    width: usize,
    height: usize,
    channels: usize,
    /// Preprocessed channel planes (plane 0 = raw L, planes 1.. = chroma
    /// pre-blur) — the cross-blur reads exactly these (`DssimChan.img`).
    img: Buffer,
    mu: Buffer,
    sq_blur: Buffer,
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


/// Per-scale allocation keepers for both the RGB and gray paths (F30: these
/// were two near-identical structs; `channels` is 1 for gray). Staging and
/// transient buffers are dropped at each scale's scope end — the clones held
/// in `passes` keep the GPU memory alive until the submit completes.
struct ScaleKeep {
    width: usize,
    height: usize,
    channels: usize,
    img: Buffer,
    mu: Buffer,
    sq: Buffer,
}
