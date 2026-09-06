//! GPU color conversion — the Vulkan counterpart of `tolab.rs`'s
//! `rgb_to_lab` (RGBA premultiplied linear → planar Lab) and `GBitmap::to_lab`
//! (gray). All constants come from `dssim_core::tolab::LAB_GPU_CONSTANTS`,
//! computed by the same Rust expressions as the CPU path. The sRGB gamma LUT
//! and premultiplication stay on the CPU (hybrid strategy: the CPU needs the
//! linear RGBAPLU image for its 2×2 downsample; plan §6 Phase F parity is
//! defined on the Lab planes).

use std::sync::Arc;

use ash::vk;

use crate::context::Context;
use crate::pipeline::{dispatch_sequence, ComputePipeline, Pass};
use crate::transfer::Buffer;
use crate::Result;

/// Push constant block for rgba_to_lab.comp (112 bytes). Field names mirror
/// the shader's block members.
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(non_snake_case)]
struct LabPC {
    dims: [u32; 4], // width, height, channels(1|3), 0
    kX: [f32; 4],
    kY: [f32; 4],
    kZ: [f32; 4],
    kC: [f32; 4], // K, EPSILON, 16/116, 1.05
    kA: [f32; 4], // 500/220, 86.2/220, 200/220, 107.9/220
    kG: [f32; 4], // K*1.16, 0, 0, 0
}

const _: () = assert!(std::mem::size_of::<LabPC>() == 112);

fn pc_bytes(width: usize, height: usize, channels: usize, c: &[f32; 18]) -> Vec<u8> {
    let pc = LabPC {
        dims: [width as u32, height as u32, channels as u32, 0],
        kX: [c[0], c[1], c[2], 0.0],
        kY: [c[3], c[4], c[5], 0.0],
        kZ: [c[6], c[7], c[8], 0.0],
        kC: [c[9], c[10], c[11], c[12]],
        kA: [c[13], c[14], c[15], c[16]],
        kG: [c[17], 0.0, 0.0, 0.0],
    };
    // repr(C) pod of plain f32/u32 — no padding, safe to view as bytes.
    unsafe {
        std::slice::from_raw_parts(&pc as *const LabPC as *const u8, std::mem::size_of::<LabPC>())
    }
    .to_vec()
}

fn pack(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Convert premultiplied-linear RGBA pixels (4 floats/pixel, `in_rgba`)
/// to `channels` planar Lab planes (L,a,b for 3ch; single plane for gray,
/// where `in_rgba` holds 1 float/pixel) on the GPU. Mirrors
/// `ToLABBitmap for ImgRef<RGBAPLU>` / `GBitmap::to_lab` semantics.
///
/// Returns `channels` tightly-packed planes concatenated.
pub fn rgba_to_lab_gpu(
    context: &Arc<Context>,
    in_rgba: &[f32],
    width: usize,
    height: usize,
    channels: usize,
) -> Result<Vec<f32>> {
    assert!(channels == 1 || channels == 3, "DSSIM uses 1 or 3 channels");
    let pixels = width * height;
    let expected = if channels == 3 { pixels * 4 } else { pixels };
    assert_eq!(in_rgba.len(), expected, "input size mismatch");

    let pipelines = ColorPipelines::new(context)?;

    let src = context.upload_buffer("lab.src", &pack(in_rgba), vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let dst = context.alloc_buffer(
        "lab.dst",
        (pixels * channels * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        gpu_allocator::MemoryLocation::GpuOnly,
    )?;

    let mut passes: Vec<Pass> = Vec::new();
    pipelines.lab_into(&mut passes, src.clone(), dst.clone(), width, height, channels);
    dispatch_sequence(context, &passes)?;

    let out_bytes = context.download_buffer(&dst)?;
    let mut out = Vec::with_capacity(pixels * channels);
    for c in out_bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*c));
    }
    Ok(out)
}

/// The rgba_to_lab pipeline, created once per GpuSsim and reused.
pub struct ColorPipelines {
    to_lab: ComputePipeline,
}

impl ColorPipelines {
    pub fn new(context: &Arc<Context>) -> Result<Self> {
        Ok(Self {
            to_lab: ComputePipeline::new(
                context,
                "rgba_to_lab",
                include_bytes!("../shaders/rgba_to_lab.comp.spv"),
                2,
                112,
            )?,
        })
    }

    /// Push the rgba_to_lab dispatch into a sequence: reads `src`
    /// (interleaved RGBA, 4 floats/pixel, for 3ch; single plane for 1ch),
    /// writes `dst` (channels planes, plane stride = width).
    pub(crate) fn lab_into<'a>(
        &'a self,
        passes: &mut Vec<Pass<'a>>,
        src: Buffer,
        dst: Buffer,
        width: usize,
        height: usize,
        channels: usize,
    ) {
            let pc = pc_bytes(width, height, channels, &dssim_core::tolab::LAB_GPU_CONSTANTS);
        passes.push(Pass::Compute {
            pipeline: &self.to_lab,
            buffers: vec![src, dst],
            push: pc,
            groups: ((width * height) as u32).div_ceil(64),
        });
    }
}
