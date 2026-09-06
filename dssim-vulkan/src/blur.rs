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
use crate::transfer::Buffer;
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

/// The three blur pipelines, created once per [`GpuSsim`](crate::GpuSsim) and
/// reused across every scale and image (pipeline creation is expensive;
/// dispatches are cheap).
pub struct BlurPipelines {
    context: Arc<Context>,
    h5: ComputePipeline,
    h5_mul: ComputePipeline,
    v5: ComputePipeline,
}

impl BlurPipelines {
    pub fn new(context: &Arc<Context>) -> Result<Self> {
        Ok(Self {
            context: context.clone(),
            h5: ComputePipeline::new(context, "blur_h5", include_spv("blur_h5"), 2, 56)?,
            h5_mul: ComputePipeline::new(context, "blur_h5_mul", include_spv("blur_h5_mul"), 3, 56)?,
            v5: ComputePipeline::new(context, "blur_v5", include_spv("blur_v5"), 2, 56)?,
        })
    }

    /// Push a single H5 pass into a sequence (plane-offset aware): reads
    /// plane `src_off` of `src`, writes plane `dst_off` of `dst`.
    pub(crate) fn h5_into<'a>(
        &'a self,
        passes: &mut Vec<Pass<'a>>,
        src: Buffer,
        tmp: Buffer,
        width: usize,
        height: usize,
        stride: usize,
        src_off: u32,
        dst_off: u32,
    ) {
        let k5_ref = dssim_core::blur::K5_REF;
        passes.push(Pass::Compute {
            pipeline: &self.h5,
            buffers: vec![src.clone(), tmp.clone()],
            push: pc_bytes_off(width, height, stride, src_off, width, dst_off, &k5_ref),
            groups: ((width * height) as u32).div_ceil(64),
        });
    }

    /// Push a single V5 pass into a sequence (tight src/dst + plane offsets).
    pub(crate) fn v5_into<'a>(
        &'a self,
        passes: &mut Vec<Pass<'a>>,
        tmp: Buffer,
        dst: Buffer,
        width: usize,
        height: usize,
        _stride: usize,
        dst_off: u32,
    ) {
        let k5_ref = dssim_core::blur::K5_REF;
        passes.push(Pass::Compute {
            pipeline: &self.v5,
            buffers: vec![tmp.clone(), dst.clone()],
            push: pc_bytes_off(width, height, width, 0, width, dst_off, &k5_ref),
            groups: ((width * height) as u32).div_ceil(64),
        });
    }

    /// Push a single fused H5-multiply pass into a sequence (plane-offset aware).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn h5_mul_into<'a>(
        &'a self,
        passes: &mut Vec<Pass<'a>>,
        src1: Buffer,
        src2: Buffer,
        tmp: Buffer,
        width: usize,
        height: usize,
        stride1: usize,
        src1_off: u32,
        src2_off: u32,
        dst_off: u32,
    ) {
        let k5_ref = dssim_core::blur::K5_REF;
        passes.push(Pass::Compute {
            pipeline: &self.h5_mul,
            buffers: vec![src1, src2, tmp],
            push: pc_mul_bytes(width, height, stride1, src1_off, stride1, src2_off, width, dst_off, &k5_ref),
            groups: ((width * height) as u32).div_ceil(64),
        });
    }
    /// GPU `blur` of host data: upload → H5 → V5 → download. `data` may be
    /// strided (`stride >= width`); result is tightly packed.
    pub fn blur(&self, data: &[f32], width: usize, height: usize, stride: usize) -> Result<Vec<f32>> {
        assert!(width > 0 && height > 0);
        assert!(
            data.len() >= stride * (height - 1) + width,
            "buffer smaller than strided image"
        );

        let k5_ref = dssim_core::blur::K5_REF;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let src = self.context.upload_buffer("blur.src", &bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let tmp = self.context.alloc_buffer(
            "blur.tmp",
            (width * height * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            gpu_allocator::MemoryLocation::GpuOnly,
        )?;
        let dst = self.context.alloc_buffer(
            "blur.dst",
            (width * height * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::GpuOnly,
        )?;
        let pixels = (width * height) as u32;

        let h_pass = Pass::Compute {
            pipeline: &self.h5,
            buffers: vec![src.clone(), tmp.clone()],
            push: pc_bytes(width, height, stride, 0, width, &k5_ref),
            groups: pixels.div_ceil(64),
        };
        let v_pass = Pass::Compute {
            pipeline: &self.v5,
            buffers: vec![tmp.clone(), dst.clone()],
            push: pc_bytes(width, height, width, 0, width, &k5_ref),
            groups: pixels.div_ceil(64),
        };
        dispatch_sequence(&self.context, &[h_pass, v_pass])?;

        read_f32s(&self.context, &dst, width * height)
    }

    /// GPU `blur_mul`: `blur(src1 * src2)` with the multiply fused into the
    /// horizontal pass, mirroring `dssim_core::blur::blur_mul`.
    pub fn blur_mul(
        &self,
        src1: &[f32],
        src2: &[f32],
        width: usize,
        height: usize,
        stride1: usize,
        stride2: usize,
    ) -> Result<Vec<f32>> {
        assert!(width > 0 && height > 0);
        assert!(
            src1.len() >= stride1 * (height - 1) + width
                && src2.len() >= stride2 * (height - 1) + width,
            "blur_mul inputs smaller than their strided images"
        );

        let k5_ref = dssim_core::blur::K5_REF;
        let pack = |s: &[f32]| s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        let s1 = self.context.upload_buffer("blur_mul.src1", &pack(src1), vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let s2 = self.context.upload_buffer("blur_mul.src2", &pack(src2), vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let tmp = self.context.alloc_buffer(
            "blur_mul.tmp",
            (width * height * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            gpu_allocator::MemoryLocation::GpuOnly,
        )?;
        let dst = self.context.alloc_buffer(
            "blur_mul.dst",
            (width * height * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::GpuOnly,
        )?;
        let pixels = (width * height) as u32;

        let h_pass = Pass::Compute {
            pipeline: &self.h5_mul,
            buffers: vec![s1.clone(), s2.clone(), tmp.clone()],
            push: pc_mul_bytes(width, height, stride1, 0, stride2, 0, width, 0, &k5_ref),
            groups: pixels.div_ceil(64),
        };
        let v_pass = Pass::Compute {
            pipeline: &self.v5,
            buffers: vec![tmp.clone(), dst.clone()],
            push: pc_bytes(width, height, width, 0, width, &k5_ref),
            groups: pixels.div_ceil(64),
        };
        dispatch_sequence(&self.context, &[h_pass, v_pass])?;

        read_f32s(&self.context, &dst, width * height)
    }
}

fn read_f32s(context: &Arc<Context>, dst: &Buffer, count: usize) -> Result<Vec<f32>> {
    let out_bytes = context.download_buffer(dst)?;
    let mut out = Vec::with_capacity(count);
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
    let pipelines = BlurPipelines::new(context)?;
    pipelines.blur(data, width, height, stride)
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
    let pipelines = BlurPipelines::new(context)?;
    pipelines.blur_mul(src1, src2, width, height, stride1, stride2)
}

/// Push-constant bytes with src/dst plane offsets (byte offsets into the
/// bound buffers): dims = (width, height, src_stride_elems, src_off_elems),
/// dims2 = (dst_stride_elems, dst_off_elems).
#[allow(clippy::too_many_arguments)]
fn pc_bytes_off(
    width: usize,
    height: usize,
    src_stride: usize,
    src_off: u32,
    dst_stride: usize,
    dst_off: u32,
    k5_ref: &[f32; 6],
) -> Vec<u8> {
    let pc = BlurPC {
        dims: [width as u32, height as u32, src_stride as u32, src_off],
        dims2: [dst_stride as u32, dst_off, 0, 0],
        k1: [k5_ref[0], k5_ref[1], k5_ref[2], k5_ref[3]],
        k2: [k5_ref[4], k5_ref[5]],
    };
    unsafe {
        std::slice::from_raw_parts(&pc as *const BlurPC as *const u8, std::mem::size_of::<BlurPC>())
    }
    .to_vec()
}

#[allow(clippy::too_many_arguments)]
fn pc_mul_bytes(
    width: usize,
    height: usize,
    src1_stride: usize,
    src1_off: u32,
    src2_stride: usize,
    src2_off: u32,
    dst_stride: usize,
    dst_off: u32,
    k5_ref: &[f32; 6],
) -> Vec<u8> {
    let pc = BlurPC {
        dims: [width as u32, height as u32, src1_stride as u32, src1_off],
        dims2: [dst_stride as u32, src2_stride as u32, src2_off, dst_off],
        k1: [k5_ref[0], k5_ref[1], k5_ref[2], k5_ref[3]],
        k2: [k5_ref[4], k5_ref[5]],
    };
    unsafe {
        std::slice::from_raw_parts(&pc as *const BlurPC as *const u8, std::mem::size_of::<BlurPC>())
    }
    .to_vec()
}
