//! `dssim-vulkan` — Vulkan compute backend for DSSIM, porting the
//! `dssim-core` kernels to GPU without changing their CPU-path semantics.
//!
//! Plan: `dssim-vulkan-fable-plan.md` / `VULKAN_PORT_PLAN.md`. This module is
//! currently Phase B: the minimal runtime (context, allocator, staging
//! transfers, compute dispatch) proven by the ×2 smoke shader.

pub mod blur;
pub mod color;
pub mod context;
pub mod error;
pub mod pipeline;
pub mod score;
pub mod ssim;
pub mod transfer;

pub use blur::{blur_gpu, blur_mul_gpu, BlurPipelines};
pub use color::rgba_to_lab_gpu;
pub use context::Context;
pub use error::{Error, Result};
pub use pipeline::ComputePipeline;
pub use score::{to_dssim, GpuSsim, GpuSsimImage};
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
