//! Vulkan blur dispatches — the GPU counterparts of `dssim-core::blur`
//! (`blur`, `blur_in_place`, `blur_mul`). Each is one submit with two passes:
//! H5 then V5, with a barrier between them (the V5 pass reads the H5 output),
//! matching the CPU's sequential H→V structure. Kernel weights are the exact
//! f32 values from `dssim_core::blur::K5_REF`, passed via push constants so
//! the shader never re-derives them.

use std::sync::Arc;

use ash::vk;

use crate::context::Context;
use crate::pipeline::{dispatch_sequence, ComputePipeline, Pass};
use crate::Result;

/// Push constant block shared by the three blur shaders (56 bytes):
/// uvec4 dims / uvec4 dims2 / vec4 k1 / vec2 k2 (std430-style packing).
#[repr(C)]
#[derive(Copy, Clone)]
struct BlurPC {
    dims: [u32; 4],   // width, height, src_stride, src2_stride
    dims2: [u32; 4],  // dst_stride, 0, 0, 0
    k1: [f32; 4],     // K5_OUTER, K5_INNER, K5_MID, K5_EDGE_CENTER
    k2: [f32; 2],     // K5_EDGE_NEAR, K5_EDGE_FAR
}

const _: () = assert!(std::mem::size_of::<BlurPC>() == 56);

fn pc_bytes(width: usize, height: usize, src_stride: usize, src2_stride: usize, dst_stride: usize, k5_ref: &[f32; 6]) -> Vec<u8> {
    let pc = BlurPC {
        dims: [width as u32, height as u32, src_stride as u32, src2_stride as u32],
        dims2: [dst_stride as u32, 0, 0, 0],
        k1: [k5_ref[0], k5_ref[1], k5_ref[2], k5_ref[3]],
        k2: [k5_ref[4], k5_ref[5]],
    };
    // repr(C) pod of plain f32/u32 — no padding, safe to view as bytes.
    unsafe {
        std::slice::from_raw_parts(&pc as *const BlurPC as *const u8, std::mem::size_of::<BlurPC>())
    }
    .to_vec()
}

/// High-level GPU blur over host data: upload → H5 → V5 → download.
/// Mirrors `dssim_core::blur::blur` semantics: `data` may be strided
/// (`stride >= width`); the result is tightly packed `width × height`.
pub fn blur_gpu(
    context: &Arc<Context>,
    data: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Vec<f32>> {
    assert!(width > 0 && height > 0);
    assert!(
        data.len() >= stride * (height - 1) + width,
        "buffer smaller than strided image"
    );

    let k5_ref = dssim_core::blur::K5_REF;
    let pipeline_h = ComputePipeline::new(context, "blur_h5", include_spv("blur_h5"), 2, 56)?;
    let pipeline_v = ComputePipeline::new(context, "blur_v5", include_spv("blur_v5"), 2, 56)?;

    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let src = context.upload_buffer("blur.src", &bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let tmp = context.alloc_buffer(
        "blur.tmp",
        (width * height * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        gpu_allocator::MemoryLocation::GpuOnly,
    )?;
    let dst = context.alloc_buffer(
        "blur.dst",
        (width * height * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        gpu_allocator::MemoryLocation::GpuOnly,
    )?;
    let pixels = (width * height) as u32;

    // Pass 1: H5, src (strided) -> tmp (tight). Pass 2: V5, tmp -> dst.
    let h_pass = Pass {
        pipeline: &pipeline_h,
        buffers: vec![&src, &tmp],
        push: pc_bytes(width, height, stride, 0, width, &k5_ref),
        groups: pixels.div_ceil(64),
    };
    let v_pass = Pass {
        pipeline: &pipeline_v,
        buffers: vec![&tmp, &dst],
        push: pc_bytes(width, height, width, 0, width, &k5_ref),
        groups: pixels.div_ceil(64),
    };
    dispatch_sequence(context, &[h_pass, v_pass])?;

    let out_bytes = context.download_buffer(&dst)?;
    let mut out = Vec::with_capacity(width * height);
    for c in out_bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*c));
    }
    debug_assert_eq!(out.len(), width * height);
    Ok(out)
}

/// High-level GPU `blur_mul` over host data: `blur(src1 * src2)`, fused
/// multiply into the horizontal pass exactly like `dssim_core::blur::blur_mul`.
pub fn blur_mul_gpu(
    context: &Arc<Context>,
    src1: &[f32],
    src2: &[f32],
    width: usize,
    height: usize,
    stride1: usize,
    stride2: usize,
) -> Result<Vec<f32>> {
    assert!(width > 0 && height > 0);
    assert!(
        src1.len() >= stride1 * (height - 1) + width && src2.len() >= stride2 * (height - 1) + width,
        "blur_mul inputs smaller than their strided images"
    );

    let k5_ref = dssim_core::blur::K5_REF;
    let pipeline_hm = ComputePipeline::new(context, "blur_h5_mul", include_spv("blur_h5_mul"), 3, 56)?;
    let pipeline_v = ComputePipeline::new(context, "blur_v5", include_spv("blur_v5"), 2, 56)?;

    let pack = |s: &[f32]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let s1 = context.upload_buffer("blur_mul.src1", &pack(src1), vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let s2 = context.upload_buffer("blur_mul.src2", &pack(src2), vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let tmp = context.alloc_buffer(
        "blur_mul.tmp",
        (width * height * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        gpu_allocator::MemoryLocation::GpuOnly,
    )?;
    let dst = context.alloc_buffer(
        "blur_mul.dst",
        (width * height * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        gpu_allocator::MemoryLocation::GpuOnly,
    )?;
    let pixels = (width * height) as u32;

    let h_pass = Pass {
        pipeline: &pipeline_hm,
        buffers: vec![&s1, &s2, &tmp],
        push: pc_bytes(width, height, stride1, stride2, width, &k5_ref),
        groups: pixels.div_ceil(64),
    };
    let v_pass = Pass {
        pipeline: &pipeline_v,
        buffers: vec![&tmp, &dst],
        push: pc_bytes(width, height, width, 0, width, &k5_ref),
        groups: pixels.div_ceil(64),
    };
    dispatch_sequence(context, &[h_pass, v_pass])?;

    let out_bytes = context.download_buffer(&dst)?;
    let mut out = Vec::with_capacity(width * height);
    for c in out_bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*c));
    }
    Ok(out)
}

fn include_spv(name: &str) -> &'static [u8] {
    match name {
        "blur_h5" => include_bytes!("../shaders/blur_h5.comp.spv"),
        "blur_h5_mul" => include_bytes!("../shaders/blur_h5_mul.comp.spv"),
        "blur_v5" => include_bytes!("../shaders/blur_v5.comp.spv"),
        _ => unreachable!(),
    }
}
