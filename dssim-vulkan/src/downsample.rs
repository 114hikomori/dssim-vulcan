//! Tier 5 (opt-in device prep): the 2x2 box downsample that builds the pyramid
//! on the GPU instead of on the CPU. The shader is an exact transcription of
//! `dssim_core::image::Average4` for `RGBAPLU` (`((a+b)+c)+d) * 0.25` per
//! component, `precise`), so a device-built pyramid is bitwise equal to the
//! CPU-downsampled one. Off by default — see [`crate::score::PrepMode`].

use std::sync::Arc;

use crate::context::Context;
use crate::pipeline::{ComputePipeline, Pass};
use crate::transfer::Buffer;
use crate::Result;

/// Push constants: input (width, height). Output is (width/2, height/2).
#[repr(C)]
#[derive(Copy, Clone)]
struct DownsamplePC {
    dims: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<DownsamplePC>() == 16);

fn pc_bytes(width: usize, height: usize) -> Vec<u8> {
    let pc = DownsamplePC {
        dims: [width as u32, height as u32, 0, 0],
    };
    unsafe {
        std::slice::from_raw_parts(
            &pc as *const DownsamplePC as *const u8,
            std::mem::size_of::<DownsamplePC>(),
        )
    }
    .to_vec()
}

/// The single downsample pipeline, created once per [`GpuSsim`](crate::GpuSsim)
/// and reused across every scale (only used when `PrepMode::Device`).
pub struct DownsamplePipelines {
    downsample: ComputePipeline,
}

impl DownsamplePipelines {
    pub fn new(context: &Arc<Context>) -> Result<Self> {
        Ok(Self {
            downsample: ComputePipeline::new(
                context,
                "downsample",
                include_bytes!("../shaders/downsample.comp.spv"),
                2,
                16,
            )?,
        })
    }

    /// Push a 2x2 box downsample of interleaved RGBA into a sequence: reads a
    /// tightly packed `width x height` RGBA buffer, writes `width/2 x height/2`.
    pub(crate) fn downsample_into<'a>(
        &'a self,
        passes: &mut Vec<Pass<'a>>,
        src: Buffer,
        dst: Buffer,
        width: usize,
        height: usize,
    ) {
        let ow = width / 2;
        let oh = height / 2;
        // BH11-style size guard: src holds width*height RGBA (16 B/px), dst
        // holds (width/2)*(height/2) RGBA.
        debug_assert!(
            src.size >= (width * height * 16) as u64,
            "downsample_into: src ({} B) too small for {width}x{height}",
            src.size
        );
        debug_assert!(
            dst.size >= (ow * oh * 16) as u64,
            "downsample_into: dst ({} B) too small for {ow}x{oh}",
            dst.size
        );
        passes.push(Pass::Compute {
            pipeline: &self.downsample,
            buffers: vec![src, dst],
            push: pc_bytes(width, height),
            groups: ((ow * oh) as u32).div_ceil(64),
        });
    }
}
