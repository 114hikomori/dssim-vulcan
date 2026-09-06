//! GPU SSIM combine — the Vulkan counterpart of `dssim.rs`'s
//! `compare_scale_3ch` (3-channel) and `compare_scale` (1-channel). The output
//! is the per-pixel SSIM map; pooling stays on the CPU (plan §6 Phase D).
//!
//! BH33: two entry points. `combine_into` pushes the combine dispatch into a
//! caller-owned `Vec<Pass>` -- the production path, where `GpuSsim::compare`
//! reads device-resident mu/sq/cross planes produced by the blur `*_into`
//! methods (not host buffers). `ssim_combine_gpu`/`ssim_combine_pipelines` are
//! the host-in/host-out form used by the parity tests.

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
        // BH31: these duplicate dssim-core's SSIM constants (dssim.rs:405-407:
        // c1=0.01^2, c2=0.03^2, inv3=1/3), which are local `let`s, not `pub`.
        // Verified equal today; the parity suites (phase_e, ssim_parity) catch
        // drift if a CPU-side change moves the score past 5e-6. A full fix (a
        // `pub` constant in dssim-core, or extending the LAB_GPU_CONSTANTS push
        // channel to carry these) is deferred -- see AUDIT_DEEP_BUGHUNT BH31.
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

/// The two SSIM combine pipelines (3-channel and 1-channel), created once
/// per [`GpuSsim`](crate::GpuSsim) and reused for every scale.
pub struct SsimPipelines {
    context: Arc<Context>,
    combine3: ComputePipeline,
    combine1: ComputePipeline,
}

impl SsimPipelines {
    pub fn new(context: &Arc<Context>) -> Result<Self> {
        Ok(Self {
            context: context.clone(),
            combine3: ComputePipeline::new(context, "ssim_combine_3ch", include_spv("ssim_combine_3ch"), 6, 48)?,
            combine1: ComputePipeline::new(context, "ssim_combine_1ch", include_spv("ssim_combine_1ch"), 6, 48)?,
        })
    }
}

/// GPU SSIM combine over blurred statistics, via prebuilt pipelines.
///
/// `mu_o`/`mu_m` hold `num_channels` concatenated planes each (original /
/// modified), likewise `sq_o`/`sq_m`; `cross` holds `num_channels` planes.
/// Plane stride = width, channel order matches `to_lab()` (L, a, b).
///
/// * `num_channels == 3` mirrors `compare_scale_3ch` (color inputs).
/// * `num_channels == 1` mirrors `compare_scale` (gray inputs).
///
/// Returns the tightly packed `width × height` SSIM map.
#[allow(clippy::too_many_arguments)]
pub fn ssim_combine_pipelines(
    pipelines: &SsimPipelines,
    // BH22: parameter order == binding order == the shader's set-0 layout
    // (mu_o, mu_m, sq_o, sq_m, cross, dst). Previously the params were
    // (mu_o, sq_o, mu_m, sq_m) while the vec bound (mu_o, mu_m, sq_o, sq_m) --
    // a swap of two same-length stat buffers that no length assert could catch,
    // producing a finite, plausible, WRONG map on any future "tidy up" edit.
    mu_o: &[f32],
    mu_m: &[f32],
    sq_o: &[f32],
    sq_m: &[f32],
    cross: &[f32],
    width: usize,
    height: usize,
    num_channels: usize,
) -> Result<Vec<f32>> {
    assert!(num_channels == 1 || num_channels == 3, "DSSIM uses 1 or 3 channels");
    let pixels = width * height;
    assert_eq!(mu_o.len(), num_channels * pixels);
    assert_eq!(mu_m.len(), num_channels * pixels);
    assert_eq!(sq_o.len(), num_channels * pixels);
    assert_eq!(sq_m.len(), num_channels * pixels);
    assert_eq!(cross.len(), num_channels * pixels);

    let context = &pipelines.context;
    let pipeline = match num_channels {
        3 => &pipelines.combine3,
        _ => &pipelines.combine1,
    };

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

    let pass = Pass::Compute {
        pipeline,
        buffers: vec![mu_o_buf.clone(), mu_m_buf.clone(), sq_o_buf.clone(), sq_m_buf.clone(), cross_buf.clone(), dst.clone()],
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

/// Convenience wrapper over [`ssim_combine_pipelines`]: `mu` and `sq_blur`
/// hold `2 * num_channels` concatenated planes — (original, modified) per
/// channel, plane stride = width; `cross` holds `num_channels` planes.
pub fn ssim_combine_gpu(
    context: &Arc<Context>,
    mu: &[f32],
    sq_blur: &[f32],
    cross: &[f32],
    width: usize,
    height: usize,
    num_channels: usize,
) -> Result<Vec<f32>> {
    let pipelines = SsimPipelines::new(context)?;
    let pixels = width * height;
    let (mu_o, mu_m) = mu.split_at(pixels * num_channels);
    let (sq_o, sq_m) = sq_blur.split_at(pixels * num_channels);
    ssim_combine_pipelines(
        &pipelines, mu_o, mu_m, sq_o, sq_m, cross, width, height, num_channels,
    )
}

impl SsimPipelines {
    /// Push the SSIM combine dispatch into a sequence (buffers already
    /// GPU-resident; map lands in `dst`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn combine_into<'a>(
        &'a self,
        passes: &mut Vec<Pass<'a>>,
        // BH22: parameter order == binding order == the shader's set-0 layout
        // (mu_o, mu_m, sq_o, sq_m, cross, dst). See ssim_combine_pipelines.
        mu_o: Buffer,
        mu_m: Buffer,
        sq_o: Buffer,
        sq_m: Buffer,
        cross: Buffer,
        dst: Buffer,
        width: usize,
        height: usize,
        num_channels: usize,
    ) {
        assert!(num_channels == 1 || num_channels == 3, "DSSIM uses 1 or 3 channels");
        // BH11: the five stat buffers each hold `num_channels` planes of pixels;
        // dst holds one plane. Assert they cover the dispatch (descriptors bind
        // the whole buffer, so a wrong plane is an invisible in-allocation OOB).
        let pixels = width * height;
        let stat_bytes = (num_channels * pixels * 4) as u64;
        debug_assert!(
            mu_o.size >= stat_bytes
                && mu_m.size >= stat_bytes
                && sq_o.size >= stat_bytes
                && sq_m.size >= stat_bytes
                && cross.size >= stat_bytes,
            "combine_into: a stat buffer is smaller than {num_channels} planes"
        );
        debug_assert!(
            dst.size >= (pixels * 4) as u64,
            "combine_into: dst ({} B) too small for {width}x{height}",
            dst.size
        );
        let pipeline = match num_channels {
            3 => &self.combine3,
            _ => &self.combine1,
        };
        passes.push(Pass::Compute {
            pipeline,
            buffers: vec![mu_o, mu_m, sq_o, sq_m, cross, dst],
            push: pc_bytes(width, height),
            groups: ((width * height) as u32).div_ceil(64),
        });
    }
}
