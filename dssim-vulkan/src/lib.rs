//! `dssim-vulkan` — Vulkan compute backend for DSSIM, porting the
//! `dssim-core` kernels to GPU without changing their CPU-path semantics.
//!
//! Plan: `dssim-vulkan-fable-plan.md` / `VULKAN_PORT_PLAN.md`. Through Phase H
//! (M10): the full multi-scale DSSIM runs on the GPU — Lab conversion, blur
//! statistics, and the SSIM combine — driven by [`GpuSsim`] (`create_image` /
//! `compare`) with CPU f64 pooling, exposed via the `--gpu` CLI flag. See
//! `CHECKPOINT.md` for the current milestone and `AUDIT_GPU_PORT.md` for the
//! rolling correctness review.

pub mod blur;
pub mod color;
pub mod context;
pub mod downsample;
pub mod error;
pub mod pipeline;
pub mod score;
pub mod ssim;
pub mod transfer;

pub use blur::{blur_gpu, blur_mul_gpu, BlurPipelines};
pub use color::rgba_to_lab_gpu;
pub use context::Context;
pub use downsample::DownsamplePipelines;
pub use error::{Error, Result};
pub use pipeline::ComputePipeline;
pub use score::{to_dssim, GpuSsim, GpuSsimImage, PrepMode};
pub use ssim::ssim_combine_gpu;
pub use transfer::Buffer;

use std::sync::Arc;

use ash::vk;

/// Phase B smoke test: run the `×2` shader over `input` and return the GPU's
/// output. Exercises upload → dispatch → download end to end.
pub fn run_smoke(context: &Arc<Context>, input: &[f32]) -> Result<Vec<f32>> {
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();

    let input_buf = context.upload_buffer(
        "smoke.in",
        &bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let output_buf = context.alloc_buffer(
        "smoke.out",
        bytes.len() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        gpu_allocator::MemoryLocation::GpuOnly,
    )?;

    let shader = smoke_shader();
    let pipeline = ComputePipeline::new(context, "smoke_double", shader, 2, 4)?;
    let count_bytes = (input.len() as u32).to_le_bytes();
    pipeline.dispatch(&[&input_buf, &output_buf], input.len() as u32, &count_bytes)?;

    let out_bytes = context.download_buffer(&output_buf)?;
    let out: Vec<f32> = out_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(out.len(), input.len(), "GPU returned the wrong element count");
    Ok(out)
}

/// The compiled smoke shader. Compiled from `shaders/smoke_double.comp` with
/// `glslc --target-env=vulkan1.3 -O`; recompile when the source changes.
const fn smoke_shader() -> &'static [u8] {
    include_bytes!("../shaders/smoke_double.comp.spv")
}
