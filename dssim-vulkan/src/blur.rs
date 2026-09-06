//! Vulkan blur dispatches — the GPU counterparts of `dssim-core::blur`
//! (`blur`, `blur_in_place`, `blur_mul`). Kernel weights are the exact f32
//! values from `dssim_core::blur::K5_REF`, passed via push constants so the
//! shader never re-derives them.
//!
//! Two shapes live here (BH33: the module used to claim "each is one submit
//! with two passes," which is only the host wrappers):
//! - The `*_into` methods (`h5_into`, `v5_into`, `h5_mul_into`) push individual
//!   passes into a CALLER-OWNED `Vec<Pass>` so the whole pyramid (many passes
//!   across scales/channels) goes out in ONE `dispatch_sequence` — this is the
//!   production path used by `GpuSsim::create_image`/`compare`.
//! - `blur`/`blur_mul` (and the `*_gpu` free functions) are self-contained
//!   host-in/host-out helpers that upload, run H5→V5 as one two-pass submit,
//!   and download — used by the blur parity tests and as a reference.

use std::sync::Arc;

use ash::vk;

use crate::context::Context;
use crate::pipeline::{dispatch_sequence, ComputePipeline, Pass};
use crate::transfer::Buffer;
use crate::Result;

/// Push constant block shared by the three blur shaders (56 bytes):
/// uvec4 dims / uvec4 dims2 / vec4 k1 / vec2 k2 (std430-style packing).
/// Field meanings are per-shader; the two builders below are the single
/// source of truth (F33: an earlier shared `pc_bytes` let the non-mul h5
/// shader read `dims.w` as `src_off` while the builder named it
/// `src2_stride` — one wrong argument from silent corruption).
#[repr(C)]
#[derive(Copy, Clone)]
struct BlurPC {
    // h5/v5: width, height, src_stride, src_off | h5_mul: ..., src1_stride, src1_off
    dims: [u32; 4],
    // h5/v5: dst_stride, dst_off, 0, 0 | h5_mul: dst_stride, src2_stride, src2_off, dst_off
    dims2: [u32; 4],
    k1: [f32; 4],     // K5_OUTER, K5_INNER, K5_MID, K5_EDGE_CENTER
    k2: [f32; 2],     // K5_EDGE_NEAR, K5_EDGE_FAR
}

const _: () = assert!(std::mem::size_of::<BlurPC>() == 56);

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
    // The parameter list mirrors the shader's push-constant fields one-for-one;
    // grouping them would obscure that mapping, which is the whole point here.
    #[allow(clippy::too_many_arguments)]
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
        // BH11: descriptors bind the WHOLE buffer, so a wrong offset is an
        // in-allocation OOB that standard validation cannot see (GPU-Assisted is
        // off). Assert the buffers are big enough for the max index this
        // dispatch's (stride, offset, w, h) touches -- the class behind both
        // historical M10 bugs.
        debug_assert!(
            src.size >= ((src_off as usize) + (height - 1) * stride + width) as u64 * 4,
            "h5_into: src ({} B) too small for off={src_off} stride={stride} {width}x{height}",
            src.size
        );
        debug_assert!(
            tmp.size >= ((dst_off as usize) + height * width) as u64 * 4,
            "h5_into: tmp ({} B) too small for dst_off={dst_off} {width}x{height}",
            tmp.size
        );
        let k5_ref = dssim_core::blur::K5_REF;
        passes.push(Pass::Compute {
            pipeline: &self.h5,
            buffers: vec![src.clone(), tmp.clone()],
            push: pc_bytes_off(width, height, stride, src_off, width, dst_off, &k5_ref),
            groups: ((width * height) as u32).div_ceil(64),
        });
    }

    /// Push a single V5 pass into a sequence. The source (`tmp`) is always a
    /// tight single plane (stride == width), so no stride parameter is taken
    /// (F33: an earlier dead `_stride` param implied a stride could differ).
    pub(crate) fn v5_into<'a>(
        &'a self,
        passes: &mut Vec<Pass<'a>>,
        tmp: Buffer,
        dst: Buffer,
        width: usize,
        height: usize,
        dst_off: u32,
    ) {
        // BH11: tmp is read tight (stride == width) from element 0; dst plane at
        // dst_off. Assert both cover the max index the shader touches.
        debug_assert!(
            tmp.size >= (height * width) as u64 * 4,
            "v5_into: tmp ({} B) too small for {width}x{height}",
            tmp.size
        );
        debug_assert!(
            dst.size >= ((dst_off as usize) + height * width) as u64 * 4,
            "v5_into: dst ({} B) too small for dst_off={dst_off} {width}x{height}",
            dst.size
        );
        let k5_ref = dssim_core::blur::K5_REF;
        passes.push(Pass::Compute {
            pipeline: &self.v5,
            buffers: vec![tmp.clone(), dst.clone()],
            push: pc_bytes_off(width, height, width, 0, width, dst_off, &k5_ref),
            groups: ((width * height) as u32).div_ceil(64),
        });
    }

    /// Push a single fused H5-multiply pass into a sequence (plane-offset aware).
    ///
    /// BH12: `src2` is read with the SAME stride as `src1` (`stride1`) -- the
    /// shader supports independent strides (dims2.y) but this wrapper does not
    /// expose one, because every caller blurs two planes of identical layout. If
    /// a future caller needs `stride2 != stride1`, thread it through
    /// `pc_mul_bytes`'s 5th argument (no shader change required).
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
        // BH11: assert all three buffers cover the max index the shader touches
        // (src2 sized with stride1, per BH12's assumption).
        debug_assert!(
            src1.size >= ((src1_off as usize) + (height - 1) * stride1 + width) as u64 * 4,
            "h5_mul_into: src1 ({} B) too small for off={src1_off} stride={stride1}",
            src1.size
        );
        debug_assert!(
            src2.size >= ((src2_off as usize) + (height - 1) * stride1 + width) as u64 * 4,
            "h5_mul_into: src2 ({} B) too small (assumes stride2 == stride1)",
            src2.size
        );
        debug_assert!(
            tmp.size >= ((dst_off as usize) + height * width) as u64 * 4,
            "h5_mul_into: tmp ({} B) too small for dst_off={dst_off}",
            tmp.size
        );
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
        // BH24: the length assert below passes even for stride < width, but then
        // rows overlap and the shader reads in-bounds garbage where the CPU
        // (imgref new_stride) would panic. Require stride >= width.
        assert!(stride >= width, "blur: stride ({stride}) must be >= width ({width})");
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
            push: pc_bytes_off(width, height, stride, 0, width, 0, &k5_ref),
            groups: pixels.div_ceil(64),
        };
        let v_pass = Pass::Compute {
            pipeline: &self.v5,
            buffers: vec![tmp.clone(), dst.clone()],
            push: pc_bytes_off(width, height, width, 0, width, 0, &k5_ref),
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
        // BH24: see blur(). stride < width passes the length assert but overlaps
        // rows -> in-bounds garbage on the GPU.
        assert!(
            stride1 >= width && stride2 >= width,
            "blur_mul: strides ({stride1},{stride2}) must be >= width ({width})"
        );
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
            push: pc_bytes_off(width, height, width, 0, width, 0, &k5_ref),
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

/// Push-constant bytes with src/dst plane offsets. BH21: the offsets are
/// ELEMENT offsets into the bound buffers (f32 units), NOT byte offsets -- the
/// shader indexes `buffer[off + y*stride + x]` directly. dims =
/// (width, height, src_stride_elems, src_off_elems), dims2 =
/// (dst_stride_elems, dst_off_elems). A caller that "fixed" this to pass
/// `off*4` would shift plane reads by 3x the plane size -- in-descriptor-range,
/// validation-blind OOB.
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
