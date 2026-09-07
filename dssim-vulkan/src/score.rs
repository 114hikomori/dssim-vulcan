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
use crate::Error;

use ash::vk;

// `Downsample` is trait-scoped on dssim-core's image types.
use dssim_core::Downsample as _;

/// Pooling weights, transcribed from dssim.rs:85 (`DEFAULT_WEIGHTS`).
pub const DEFAULT_WEIGHTS: [f64; 5] = [0.028, 0.197, 0.322, 0.298, 0.155];

/// Adaptive submit granularity (F28). A single batched submit keeps every
/// scale's per-scale transients (`staging`/`rgba`/`tmp` in create;
/// `tmp`/`cross`/`map` in compare) alive until the one fence returns -- great
/// for small images (fewest fences, the Phase-H overhead win) but a VRAM
/// liability at 4K+ (create footprint ~96 B/px summed over the pyramid vs
/// ~72 B/px if transients free per scale). At or above this scale-0 pixel
/// count, flush after every scale so only one scale's transients are live at a
/// time. The persistent outputs (`img`/`mu`/`sq`) still accumulate across all
/// scales -- `compare` needs them -- so this removes the *excess*, not the
/// algorithm's inherent working set. 6 M px (~2450^2) keeps 2048^2 (4.2 M px,
/// ~400 MB) on the fast single-submit path and splits 4K (16.8 M px) and up.
const SPLIT_SUBMIT_MIN_PIXELS: usize = 6_000_000;

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

/// Tier 5: how the multi-scale pyramid's per-scale RGBA is produced.
///
/// - [`PrepMode::Cpu`] (default): the CPU runs `dssim_core::Downsample` between
///   scales and uploads each scale's RGBA. This is the original, fully-tested
///   path and stays the default — the GPU-side downsample is a standing non-goal
///   (AGENTS.md §8) made opt-in.
/// - [`PrepMode::Device`]: upload only level-0 RGBA once, then build the rest of
///   the pyramid on the GPU with a 2x2 box downsample that is a bitwise-exact
///   transcription of the CPU's `Average4` (Mode A parity contract). Cuts the
///   host->VRAM transfer and the CPU downsample at large sizes; opt-in only.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrepMode {
    Cpu,
    Device,
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
    /// Tier 5: the device-side downsample pipeline (only dispatched when
    /// `prep_mode == Device`, but built once regardless for simplicity).
    downsample: crate::downsample::DownsamplePipelines,
    /// Tier 5: CPU-prep (default) vs device-prep (opt-in).
    prep_mode: PrepMode,
    /// Scale-0 pixel count at/above which `create_image`/`compare` flush per
    /// scale instead of batching all scales into one submit (F28). Defaults to
    /// `SPLIT_SUBMIT_MIN_PIXELS`; a test seam lowers it to exercise the split
    /// path on a small image (CI-safe) and prove split == batch == CPU.
    split_min_pixels: usize,
}

impl GpuSsim {
    pub fn new(context: Arc<Context>) -> Result<Self> {
        Self::with_prep_mode(context, PrepMode::Cpu)
    }

    /// Tier 5: build a `GpuSsim` choosing the pyramid-prep mode. `PrepMode::Cpu`
    /// is the default/`new()` behavior; `PrepMode::Device` opts into the
    /// GPU-side downsample (bitwise-equal, Mode A parity contract).
    pub fn with_prep_mode(context: Arc<Context>, prep_mode: PrepMode) -> Result<Self> {
        let blur = BlurPipelines::new(&context)?;
        let color = crate::color::ColorPipelines::new(&context)?;
        let ssim = SsimPipelines::new(&context)?;
        let downsample = crate::downsample::DownsamplePipelines::new(&context)?;
        Ok(Self {
            context,
            blur,
            color,
            ssim,
            downsample,
            prep_mode,
            split_min_pixels: SPLIT_SUBMIT_MIN_PIXELS,
        })
    }

    /// The active pyramid-prep mode.
    pub fn prep_mode(&self) -> PrepMode {
        self.prep_mode
    }

    /// Test-only: force the split-submit path by lowering the threshold (0 =
    /// always flush per scale). Not part of the public contract.
    #[doc(hidden)]
    pub fn set_split_threshold_for_test(&mut self, min_pixels: usize) {
        self.split_min_pixels = min_pixels;
    }

    /// Materialize the whole pyramid in ONE GPU submit: per scale, staging
    /// copy -> GPU Lab -> per-channel pre-blur/mu/sq_blur chains (all
    /// GPU-resident; results stay on the device for `compare`). The CPU only
    /// interleaves the input planes and runs the 2x2 downsample between
    /// scales. Scale-count semantics replicate dssim-core exactly: one scale
    /// per weight, stopping when `Downsample` returns None (w<8 || h<8).
    pub fn create_image(&self, src: &ImgVec<dssim_core::RGBAPLU>) -> Result<GpuSsimImage> {
        let mut passes: Vec<Pass> = Vec::new();
        let flush_per_scale = src.width() * src.height() >= self.split_min_pixels;
        let keep = self.push_rgb_scales_mode(src, &mut passes, flush_per_scale)?;
        if !passes.is_empty() {
            dispatch_sequence(&self.context, &passes)?;
        }
        Ok(to_image(keep))
    }

    /// Dispatch pyramid-prep to the active [`PrepMode`] (Tier 5). Shared by
    /// `create_image` and `create_image_pair` so both honor the mode.
    fn push_rgb_scales_mode<'p>(
        &'p self,
        src: &ImgVec<dssim_core::RGBAPLU>,
        passes: &mut Vec<Pass<'p>>,
        flush_per_scale: bool,
    ) -> Result<Vec<ScaleKeep>> {
        match self.prep_mode {
            PrepMode::Cpu => self.push_rgb_scales(src, passes, flush_per_scale),
            PrepMode::Device => self.push_rgb_scales_device(src, passes, flush_per_scale),
        }
    }

    /// T11: build both pyramids in ONE submit. Reference and modified are
    /// independent, so in batch mode their passes accumulate into a single
    /// `dispatch_sequence` -- one submit + one fence instead of two, halving the
    /// create-side fixed cost that dominates small/medium images. Large images
    /// still flush per scale (memory-bounded), so the merge only applies where
    /// it helps. The CLI's 1-vs-N streaming (original reused across modifieds)
    /// can't use this; it serves single-pair callers and the bench.
    pub fn create_image_pair(
        &self,
        reference: &ImgVec<dssim_core::RGBAPLU>,
        modified: &ImgVec<dssim_core::RGBAPLU>,
    ) -> Result<(GpuSsimImage, GpuSsimImage)> {
        // BH3: same-size precondition. compare() requires matching dims (F29);
        // without this guard a small-reference + large-modified call would build
        // the ENTIRE large pyramid in batch mode (the threshold below reads only
        // `reference`) -- exactly the transient pile-up F28 exists to cap -- and
        // the mismatch would only surface later as a PANIC in compare, not an
        // Err. Reject up front.
        if reference.width() != modified.width() || reference.height() != modified.height() {
            return Err(Error::InvalidInput(format!(
                "create_image_pair: reference {}x{} and modified {}x{} must have the same size",
                reference.width(),
                reference.height(),
                modified.width(),
                modified.height()
            )));
        }
        let mut passes: Vec<Pass> = Vec::new();
        // BH18: the pair accumulates BOTH pyramids' transients in one submit, so
        // its batch-mode peak is ~2x a single create's. Halve the effective
        // threshold so a pair flushes per scale at half the single-image size,
        // keeping the peak comparable to the F28 single-image guarantee. (Sizes
        // are equal here, so one threshold covers both.)
        let pixels = reference.width() * reference.height();
        let flush_per_scale = pixels >= self.split_min_pixels / 2;
        let keep_a = self.push_rgb_scales_mode(reference, &mut passes, flush_per_scale)?;
        let keep_b = self.push_rgb_scales_mode(modified, &mut passes, flush_per_scale)?;
        if !passes.is_empty() {
            dispatch_sequence(&self.context, &passes)?;
        }
        Ok((to_image(keep_a), to_image(keep_b)))
    }

    /// Push one RGB image's whole pyramid into a shared `passes` buffer (so a
    /// pair can be built in one submit). Returns the per-scale keepers. In split
    /// mode it dispatches+clears after each scale to bound transient VRAM; in
    /// batch mode it only accumulates.
    fn push_rgb_scales<'p>(
        &'p self,
        src: &ImgVec<dssim_core::RGBAPLU>,
        passes: &mut Vec<Pass<'p>>,
        flush_per_scale: bool,
    ) -> Result<Vec<ScaleKeep>> {
        let mut current: Option<ImgVec<dssim_core::RGBAPLU>> = Some(src.clone());
        let mut keep: Vec<ScaleKeep> = Vec::new();

        for _ in 0..DEFAULT_WEIGHTS.len() {
            let img = match current.take() {
                Some(img) => img,
                None => break,
            };
            let (w, h) = (img.width(), img.height());
            let pixels = w * h;

            // Upload the interleaved RGBA. RGBAPLU is repr(C) [f32;4] in
            // r,g,b,a order, so this is a byte-for-byte copy of the contiguous
            // pixel buffer (size assert guards it; parity suites catch a layout
            // error). T7: on unified memory the shader's source buffer is
            // host-visible, so write it directly -- no staging, no copy.
            const _: () = assert!(std::mem::size_of::<dssim_core::RGBAPLU>() == 16);
            let (cow, _, _) = img.as_ref().to_contiguous_buf();
            let px: &[dssim_core::RGBAPLU] = &cow;
            let src =
                unsafe { std::slice::from_raw_parts(px.as_ptr() as *const u8, px.len() * 16) };
            let write_rgba_into = |buf: &Buffer| -> Result<()> {
                crate::transfer::write_mapped_f32_with(buf, pixels * 4, |dst| {
                    let dstb = unsafe {
                        std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len() * 4)
                    };
                    dstb.copy_from_slice(src);
                })
            };
            let rgba_buf = if self.context.is_unified_memory() {
                let b = self.context.alloc_buffer(
                    "s.rgba",
                    (pixels * 4 * 4) as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )?;
                write_rgba_into(&b)?;
                b
            } else {
                let staging = self.context.alloc_buffer(
                    "s.staging",
                    (pixels * 4 * 4) as u64,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )?;
                write_rgba_into(&staging)?;
                let b = self.context.alloc_buffer(
                    "s.rgba",
                    (pixels * 4 * 4) as u64,
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                    gpu_allocator::MemoryLocation::GpuOnly,
                )?;
                passes.push(Pass::CopyBuffer { src: staging.clone(), dst: b.clone() });
                b
            };

            // img_all: Lab planes written straight in by the shader (plane 0 =
            // raw L, planes 1,2 = chroma, pre-blurred in place below); plane
            // stride = width. No separate lab buffer / copy (F25).
            let img_all = self.context.alloc_buffer(
                "s.img",
                (pixels * 3 * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            self.color.lab_into(passes, rgba_buf.clone(), img_all.clone(), w, h, 3);
            let tmp = self.context.alloc_buffer(
                "s.tmp",
                (pixels * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            for c in 1..3u32 {
                let off = c * pixels as u32;
                self.blur.h5_into(passes, img_all.clone(), tmp.clone(), w, h, w, off, 0);
                self.blur.v5_into(passes, tmp.clone(), img_all.clone(), w, h, off);
            }

            // mu (blur of img) and sq (blur_mul(img, img)) per channel, both
            // device-local (compare reads them on-device; F27).
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
                self.blur.h5_into(passes, img_all.clone(), tmp.clone(), w, h, w, off, 0);
                self.blur.v5_into(passes, tmp.clone(), mu_all.clone(), w, h, off);
                self.blur.h5_mul_into(passes, img_all.clone(), img_all.clone(), tmp.clone(), w, h, w, off, off, 0);
                self.blur.v5_into(passes, tmp.clone(), sq_all.clone(), w, h, off);
            }

            keep.push(ScaleKeep {
                width: w,
                height: h,
                channels: 3,
                img: img_all,
                mu: mu_all,
                sq: sq_all,
            });
            // F28: in split mode, submit this scale now and drop the pass
            // records so its transients free; outputs survive via `keep`.
            if flush_per_scale {
                dispatch_sequence(&self.context, passes)?;
                passes.clear();
            }
            current = img.downsample();
        }
        Ok(keep)
    }

    /// Upload interleaved RGBA (4 floats/px) for one scale: zero-copy write into
    /// a host-visible buffer on unified/ReBAR devices, else staging + CopyBuffer.
    /// `src_bytes` must be exactly `pixels * 16` bytes. Shared by the device-prep
    /// path (level-0 only); the CPU-prep path inlines the equivalent per scale.
    fn upload_rgba_interleaved<'p>(
        &'p self,
        pixels: usize,
        src_bytes: &[u8],
        passes: &mut Vec<Pass<'p>>,
    ) -> Result<Buffer> {
        let write_into = |buf: &Buffer| -> Result<()> {
            crate::transfer::write_mapped_f32_with(buf, pixels * 4, |dst| {
                let dstb = unsafe {
                    std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len() * 4)
                };
                dstb.copy_from_slice(src_bytes);
            })
        };
        if self.context.is_unified_memory() {
            let b = self.context.alloc_buffer(
                "d.rgba",
                (pixels * 4 * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
            )?;
            write_into(&b)?;
            Ok(b)
        } else {
            let staging = self.context.alloc_buffer(
                "d.staging",
                (pixels * 4 * 4) as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                gpu_allocator::MemoryLocation::CpuToGpu,
            )?;
            write_into(&staging)?;
            let b = self.context.alloc_buffer(
                "d.rgba",
                (pixels * 4 * 4) as u64,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            passes.push(Pass::CopyBuffer { src: staging.clone(), dst: b.clone() });
            Ok(b)
        }
    }

    /// Tier 5 (`PrepMode::Device`): upload only level-0 RGBA, then build the rest
    /// of the pyramid on the GPU with the bitwise-exact 2x2 box downsample. The
    /// per-scale Lab/blur work is identical to `push_rgb_scales`; only the
    /// between-scale step changes from "CPU downsample + upload" to "GPU
    /// downsample pass". Scale-count/cutoff semantics replicate dssim-core
    /// exactly (one scale per weight; stop when the current scale is below 8px in
    /// either dim, matching `Downsample` returning None).
    fn push_rgb_scales_device<'p>(
        &'p self,
        src: &ImgVec<dssim_core::RGBAPLU>,
        passes: &mut Vec<Pass<'p>>,
        flush_per_scale: bool,
    ) -> Result<Vec<ScaleKeep>> {
        const _: () = assert!(std::mem::size_of::<dssim_core::RGBAPLU>() == 16);
        let mut keep: Vec<ScaleKeep> = Vec::new();
        let mut w = src.width();
        let mut h = src.height();

        let (cow, _, _) = src.as_ref().to_contiguous_buf();
        let px: &[dssim_core::RGBAPLU] = &cow;
        let src_bytes =
            unsafe { std::slice::from_raw_parts(px.as_ptr() as *const u8, px.len() * 16) };
        let mut rgba_buf = self.upload_rgba_interleaved(w * h, src_bytes, passes)?;

        for _scale in 0..DEFAULT_WEIGHTS.len() {
            let pixels = w * h;

            let img_all = self.context.alloc_buffer(
                "d.img",
                (pixels * 3 * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            self.color.lab_into(passes, rgba_buf.clone(), img_all.clone(), w, h, 3);
            let tmp = self.context.alloc_buffer(
                "d.tmp",
                (pixels * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            for c in 1..3u32 {
                let off = c * pixels as u32;
                self.blur.h5_into(passes, img_all.clone(), tmp.clone(), w, h, w, off, 0);
                self.blur.v5_into(passes, tmp.clone(), img_all.clone(), w, h, off);
            }
            let mu_all = self.context.alloc_buffer(
                "d.mu",
                (pixels * 3 * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            let sq_all = self.context.alloc_buffer(
                "d.sq",
                (pixels * 3 * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            for c in 0..3u32 {
                let off = c * pixels as u32;
                self.blur.h5_into(passes, img_all.clone(), tmp.clone(), w, h, w, off, 0);
                self.blur.v5_into(passes, tmp.clone(), mu_all.clone(), w, h, off);
                self.blur.h5_mul_into(passes, img_all.clone(), img_all.clone(), tmp.clone(), w, h, w, off, off, 0);
                self.blur.v5_into(passes, tmp.clone(), sq_all.clone(), w, h, off);
            }
            keep.push(ScaleKeep {
                width: w,
                height: h,
                channels: 3,
                img: img_all,
                mu: mu_all,
                sq: sq_all,
            });

            // Stop exactly where CPU `downsample()` returns None: current scale
            // below 8px in either dim, or we've produced all weighted scales.
            if w < 8 || h < 8 || keep.len() == DEFAULT_WEIGHTS.len() {
                if flush_per_scale {
                    dispatch_sequence(&self.context, passes)?;
                    passes.clear();
                }
                break;
            }
            let (nw, nh) = (w / 2, h / 2);
            let next_rgba = self.context.alloc_buffer(
                "d.rgba_next",
                (nw * nh * 4 * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            self.downsample.downsample_into(passes, rgba_buf.clone(), next_rgba.clone(), w, h);
            rgba_buf = next_rgba;
            w = nw;
            h = nh;

            if flush_per_scale {
                dispatch_sequence(&self.context, passes)?;
                passes.clear();
            }
        }
        Ok(keep)
    }

    /// Gray (1-channel) variant: linear-light f32 planes, matching
    /// `GBitmap::to_lab` (the x1.16 branch). One GPU submit for all scales.
    pub fn create_image_gray(&self, src: &ImgVec<f32>) -> Result<GpuSsimImage> {
        let mut passes: Vec<Pass> = Vec::new();
        // F28: same adaptive granularity as the RGB path.
        let flush_per_scale = src.width() * src.height() >= self.split_min_pixels;

        let mut keep: Vec<ScaleKeep> = Vec::new();
        let mut current: Option<ImgVec<f32>> = Some(src.clone());

        for _ in 0..DEFAULT_WEIGHTS.len() {
            let img = match current.take() {
                Some(img) => img,
                None => break,
            };
            let (w, h) = (img.width(), img.height());
            let pixels = w * h;

            // Gray pixels are already f32, so the upload is a direct slice copy.
            // T7: on unified memory write the shader's source buffer directly.
            let (cow, _, _) = img.as_ref().to_contiguous_buf();
            let write_gray_into = |buf: &Buffer| -> Result<()> {
                crate::transfer::write_mapped_f32_with(buf, pixels, |dst| dst.copy_from_slice(&cow))
            };
            let gray_buf = if self.context.is_unified_memory() {
                let b = self.context.alloc_buffer(
                    "g.gray",
                    (pixels * 4) as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )?;
                write_gray_into(&b)?;
                b
            } else {
                let staging = self.context.alloc_buffer(
                    "g.staging",
                    (pixels * 4) as u64,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )?;
                write_gray_into(&staging)?;
                let b = self.context.alloc_buffer(
                    "g.gray",
                    (pixels * 4) as u64,
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                    gpu_allocator::MemoryLocation::GpuOnly,
                )?;
                passes.push(Pass::CopyBuffer { src: staging.clone(), dst: b.clone() });
                b
            };

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
            if flush_per_scale {
                dispatch_sequence(&self.context, &passes)?;
                passes.clear();
            }
            current = img.downsample();
        }

        if !passes.is_empty() {
            dispatch_sequence(&self.context, &passes)?;
        }

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
    /// DSSIM score. Per scale, the three channel cross-blurs and the SSIM
    /// combine run on the GPU-resident planes; only the per-scale SSIM maps
    /// come back for CPU pooling in f64. Small/medium images batch all scales
    /// into ONE submit (F28 threshold); large images flush per scale to cap
    /// transient VRAM. Note the scale-0 map is full-resolution (≈16 MB at
    /// 2048²; ≈4/3·P0·4B summed over scales), so this is not a negligible
    /// transfer — pooling on the GPU is a standing non-goal (AGENTS.md §8)
    /// pending profiling.
    pub fn compare(&self, reference: &GpuSsimImage, modified: &GpuSsimImage) -> Result<f64> {
        let mut passes: Vec<Pass> = Vec::new();
        // F28: same adaptive granularity as create_image.
        let scale0_pixels = reference
            .scales
            .first()
            .map(|s| s.width * s.height)
            .unwrap_or(0);
        let flush_per_scale = scale0_pixels >= self.split_min_pixels;

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
                mod_scale.mu.clone(),
                ref_scale.sq_blur.clone(),
                mod_scale.sq_blur.clone(),
                cross_all.clone(),
                map_dst.clone(),
                ref_scale.width,
                ref_scale.height,
                channels,
            );

            passes.push(Pass::CopyBuffer { src: map_dst.clone(), dst: map_rb.clone() });
            map_readbacks.push((map_rb, ref_scale.width, ref_scale.height, n));
            // F28: in split mode, submit this scale now; the map readback
            // survives via `map_readbacks`, and this scale's tmp/cross/map_dst
            // free when `passes` is cleared.
            if flush_per_scale {
                dispatch_sequence(&self.context, &passes)?;
                passes.clear();
            }
        }

        if !passes.is_empty() {
            dispatch_sequence(&self.context, &passes)?;
        }

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

    /// Tier 4 (H23): compare one already-resident reference against many
    /// modifieds, reusing the reference's GPU pyramid across every comparison
    /// (the original is created once and never re-uploaded). This is the
    /// resident-reference / batch pattern the CLI's 1-vs-N streaming relies on;
    /// the per-modified cost is `create_image` + `compare`, with the reference's
    /// create amortized over the whole batch. Each modified's transient pyramid
    /// is freed before the next, so peak VRAM stays ~one image + the reference.
    pub fn compare_many(
        &self,
        reference: &GpuSsimImage,
        modifieds: &[ImgVec<dssim_core::RGBAPLU>],
    ) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(modifieds.len());
        for m in modifieds {
            let mg = self.create_image(m)?;
            out.push(self.compare(reference, &mg)?);
        }
        Ok(out)
    }
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

/// Wrap a scale's keepers into the public per-image form.
fn to_image(keep: Vec<ScaleKeep>) -> GpuSsimImage {
    GpuSsimImage {
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
    }
}
