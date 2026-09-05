//! GPU SSIM combine — the Vulkan counterpart of `dssim.rs`'s
//! `compare_scale_3ch` (3-channel) and `compare_scale` (1-channel).
//! Statistics inputs are the blurred planes (mu, img_sq_blur, img1_img2_blur)
//! that `blur_gpu` produces; the output is the per-pixel SSIM map. Pooling
//! stays on the CPU (plan §6 Phase D).

use std::sync::Arc;

use ash::vk;

use crate::context::Context;
use crate::pipeline::{dispatch_sequence, ComputePipeline, Pass};
use crate::transfer::Buffer;
use crate::Result;

/// Push constant block shared by both combine shaders (48 bytes),
/// mirroring BlurPC: uvec4 dims / uvec4 dims2 / vec4 k.
#[repr(C)]
#[derive(Copy, Clone)]
struct SsimPC {
    dims: [u32; 4],  // width, height, 0, 0
    dims2: [u32; 4], // dst_stride, 0, 0, 0
    k: [f32; 4],     // c1 = 0.01*0.01, c2 = 0.03*0.03, inv3 = 1.0/3.0, unused
}

const _: () = assert!(std::mem::size_of::<SsimPC>() == 48);

fn pc_bytes(width: usize, height: usize) -> Vec<u8> {
    let pc = SsimPC {
        dims: [width as u32, height as u32, 0, 0],
        dims2: [width as u32, 0, 0, 0],
        k: [0.01 * 0.01, 0.03 * 0.03, 1.0 / 3.0, 0.0],
    };
    // repr(C) pod of plain f32/u32 — no padding, safe to view as bytes.
    unsafe {
        std::slice::from_raw_parts(&pc as *const SsimPC as *const u8, std::mem::size_of::<SsimPC>())
    }
    .to_vec()
}

fn pack(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn include_spv(name: &str) -> &'static [u8] {
    match name {
        "ssim_combine_3ch" => include_bytes!("../shaders/ssim_combine_3ch.comp.spv"),
        "ssim_combine_1ch" => include_bytes!("../shaders/ssim_combine_1ch.comp.spv"),
        _ => unreachable!(),
    }
}

fn upload(context: &Arc<Context>, name: &str, data: &[f32]) -> Result<Buffer> {
    context.upload_buffer(name, &pack(data), vk::BufferUsageFlags::STORAGE_BUFFER)
}

/// GPU SSIM combine over blurred statistics.
///
/// `mu` and `sq_blur` hold `2 * num_channels` concatenated planes —
/// (original, modified) per channel, plane stride = width; `cross` holds
/// `num_channels` planes. Channel order matches `to_lab()` (L, a, b).
///
/// * `num_channels == 3` mirrors `compare_scale_3ch` (color inputs).
/// * `num_channels == 1` mirrors `compare_scale` (gray inputs).
///
/// Returns the tightly packed `width × height` SSIM map.
pub fn ssim_combine_gpu(
    context: &Arc<Context>,
    mu: &[f32],
    sq_blur: &[f32],
    cross: &[f32],
    width: usize,
    height: usize,
    num_channels: usize,
) -> Result<Vec<f32>> {
    assert!(num_channels == 1 || num_channels == 3, "DSSIM uses 1 or 3 channels");
    assert_eq!(mu.len(), 2 * num_channels * width * height);
    assert_eq!(sq_blur.len(), 2 * num_channels * width * height);
    assert_eq!(cross.len(), num_channels * width * height);

    let pixels = width * height;
    let shader = match num_channels {
        3 => include_spv("ssim_combine_3ch"),
        _ => include_spv("ssim_combine_1ch"),
    };
    let pipeline = ComputePipeline::new(context, "ssim_combine", shader, 6, 48)?;

    // Both shaders take the same six bindings: (mu_o, mu_m, sq_o, sq_m,
    // cross, dst). For 3ch the first four arrive as 3-plane concatenations;
    // for 1ch each is a single plane.
    let (mu_o, mu_m) = mu.split_at(pixels * num_channels);
    let (sq_o, sq_m) = sq_blur.split_at(pixels * num_channels);
    let mu_o_buf = upload(context, "ssim.mu_o", mu_o)?;
    let mu_m_buf = upload(context, "ssim.mu_m", mu_m)?;
    let sq_o_buf = upload(context, "ssim.sq_o", sq_o)?;
    let sq_m_buf = upload(context, "ssim.sq_m", sq_m)?;
    let cross_buf = upload(context, "ssim.cross", cross)?;
    let dst = context.alloc_buffer(
        "ssim.dst",
        (pixels * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        gpu_allocator::MemoryLocation::GpuOnly,
    )?;

    let pass = Pass {
        pipeline: &pipeline,
        buffers: vec![&mu_o_buf, &mu_m_buf, &sq_o_buf, &sq_m_buf, &cross_buf, &dst],
        push: pc_bytes(width, height),
        groups: (pixels as u32).div_ceil(64),
    };
    dispatch_sequence(context, &[pass])?;

    let out_bytes = context.download_buffer(&dst)?;
    let mut out = Vec::with_capacity(pixels);
    for c in out_bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*c));
    }
    Ok(out)
}
