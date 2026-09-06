//! Compute pipeline management: SPIR-V loading, descriptor sets, push
//! constants, and single-pass dispatches (plan §4 Phase 1 item 4+).

use std::sync::Arc;

use ash::vk;

use crate::context::Context;
use crate::transfer::{buffer_barrier, Buffer};
use crate::{Error, Result};

/// One compute pipeline over a compiled SPIR-V module, plus its descriptor
/// set layout. Buffers are bound per-dispatch through a descriptor pool.
pub struct ComputePipeline {
    context: Arc<Context>,
    pub(crate) pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pub(crate) push_constant_size: u32,
    binding_count: u32,
}

impl ComputePipeline {
    /// Create a pipeline from SPIR-V code with `binding_count` storage
    /// buffers in set 0 and an optional push-constant block (`size` bytes,
    /// stage COMPUTE).
    pub fn new(
        context: &Arc<Context>,
        name: &str,
        spirv_code: &[u8],
        binding_count: u32,
        push_constant_size: u32,
    ) -> Result<Self> {
        if !spirv_code.len().is_multiple_of(4) {
            return Err(Error::Shader("SPIR-V code length must be a multiple of 4".into()));
        }
        unsafe {
            let device = &context.device;
            let code_words: Vec<u32> = spirv_code
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            let create_info = vk::ShaderModuleCreateInfo {
                code_size: spirv_code.len(),
                p_code: code_words.as_ptr(),
                ..Default::default()
            };
            let module = device
                .create_shader_module(&create_info, None)
                .map_err(Error::Vulkan)?;

            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..binding_count)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let descriptor_set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(Error::Vulkan)?;

            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: push_constant_size,
            };
            let empty_ranges: [vk::PushConstantRange; 0] = [];
            let push_constant_ranges: &[vk::PushConstantRange] = if push_constant_size > 0 {
                std::slice::from_ref(&push_constant_range)
            } else {
                &empty_ranges
            };
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                .push_constant_ranges(push_constant_ranges);
            let pipeline_layout = device
                .create_pipeline_layout(&layout_info, None)
                .map_err(Error::Vulkan)?;

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(pipeline_layout)],
                    None,
                )
                .map_err(|(_, e)| Error::Vulkan(e))?[0];

            device.destroy_shader_module(module, None);

            // Fixed descriptor-pool cap, sized to the batch plan (F5). Each
            // pass in a `dispatch_sequence` allocates one set from this
            // pipeline's pool; the pool is reset after every submit, so the
            // cap must cover the most a single sequence ever uses for one
            // pipeline. Worst case is `create_image`'s `v5` usage: (2 chroma +
            // channels mu + channels sq) per scale x scales = (2+3+3)*5 = 40
            // for the 5-scale / 3-channel DSSIM plan. BH19: `create_image_pair`
            // (T11) accumulates BOTH images' passes in one submit, doubling the
            // per-pipeline peak to ~80 -- still under 128, but the headroom is
            // now ~1.6x, not 3x. This is NOT dynamic: exceeding it fails
            // `allocate_descriptor_sets` mid-sequence (and BH1's guard now
            // recycles the pool on that error path), so raise it together with
            // any change to the batching plan.
            const MAX_SETS_PER_POOL: u32 = 128;
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: binding_count * MAX_SETS_PER_POOL,
            }];
            let descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(MAX_SETS_PER_POOL)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .map_err(Error::Vulkan)?;

            context.name_object(pipeline, &format!("dssim:pipeline:{name}"));
            Ok(Self {
                context: context.clone(),
                pipeline,
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
                push_constant_size,
                binding_count,
            })
        }
    }

    /// Dispatch the pipeline once over `count` invocations with the given
    /// buffers bound in order and `push_constants` copied into the push
    /// constant block. Blocks until the GPU work completes (Phase B
    /// determinism-over-speed policy, plan §3.4).
    pub fn dispatch(
        &self,
        buffers: &[&Buffer],
        count: u32,
        push_constants: &[u8],
    ) -> Result<()> {
        let pass = Pass::Compute {
            pipeline: self,
            buffers: buffers.iter().map(|b| (*b).clone()).collect(),
            push: push_constants.to_vec(),
            groups: count.div_ceil(64),
        };
        dispatch_sequence(&self.context, &[pass])
    }

    /// Record one bound-and-dispatched pass into an open command buffer.
    /// The descriptor set is allocated from this pipeline's pool and freed
    /// by the pool reset in [`dispatch_sequence`].
    fn record_pass(&self, cb: vk::CommandBuffer, buffers: &[Buffer], push: &[u8], groups: u32) -> Result<()> {
        // BH26: caller-bug guards. These run inside the record closure, so a
        // panic would bypass submit_one_shot's cleanup and leave the buffer
        // recording -- return Err instead (the record API's Result contract).
        if buffers.len() as u32 != self.binding_count {
            return Err(Error::Shader(format!(
                "record_pass: {} buffers for {} bindings",
                buffers.len(),
                self.binding_count
            )));
        }
        // BH30: require an exact, 4-aligned push. Overflow alone was checked
        // before; a SHORT push leaves the shader's tail constants undefined,
        // and size%4!=0 violates VUID-vkCmdPushConstants-size-00369. All
        // builders emit exact sizes today, so this only catches future drift.
        if push.len() as u32 != self.push_constant_size {
            return Err(Error::Shader(format!(
                "record_pass: push {} bytes != declared {} (short or overflowing)",
                push.len(),
                self.push_constant_size
            )));
        }
        if !push.len().is_multiple_of(4) {
            return Err(Error::Shader("record_pass: push size not a multiple of 4".into()));
        }
        unsafe {
            let device = &self.context.device;

            let set_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
            let set = device
                .allocate_descriptor_sets(&set_info)
                .map_err(Error::Vulkan)?[0];

            let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|b| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(b.buffer)
                        .offset(0)
                        .range(b.size)
                })
                .collect();
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos)];
            device.update_descriptor_sets(&writes, &[]);

            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&set),
                &[],
            );
            if !push.is_empty() {
                device.cmd_push_constants(
                    cb,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            device.cmd_dispatch(cb, groups, 1, 1);
        }
        Ok(())
    }
}

/// One step inside a [`dispatch_sequence`]: either a compute dispatch or a
/// device-to-device buffer copy (e.g. staging upload, result readback).
pub enum Pass<'a> {
    Compute {
        pipeline: &'a ComputePipeline,
        buffers: Vec<Buffer>,
        push: Vec<u8>,
        groups: u32,
    },
    CopyBuffer {
        src: Buffer,
        dst: Buffer,
    },
}

/// Record all passes into ONE command buffer (barriers between passes keep
/// writes visible to the next pass), submit, and wait for the fence.
/// Determinism-over-speed: no overlap, no pipelining. Many passes per submit
/// is the point — one fence wait amortizes the whole sequence.
pub fn dispatch_sequence(context: &Arc<Context>, passes: &[Pass<'_>]) -> Result<()> {
    // BH1: recycle each used pipeline's descriptor pool on EVERY exit path. A
    // mid-sequence failure (record_pass Err, submit error, timing-read error)
    // previously returned before the reset loop, leaking the sets allocated so
    // far toward MAX_SETS_PER_POOL -- after enough failures the pipeline bricks
    // until process exit. The guard resets on drop: success, error, or unwind.
    struct PoolReset<'a> {
        pipelines: Vec<&'a ComputePipeline>,
    }
    impl Drop for PoolReset<'_> {
        fn drop(&mut self) {
            unsafe {
                for p in self.pipelines.drain(..) {
                    // Best-effort: a reset failure (e.g. device-lost) must not
                    // mask the original error the caller is already returning.
                    let _ = p.context.device.reset_descriptor_pool(
                        p.descriptor_pool,
                        vk::DescriptorPoolResetFlags::empty(),
                    );
                }
            }
        }
    }
    let mut reset = PoolReset { pipelines: Vec::new() };
    let timed = context.timing_enabled.load(std::sync::atomic::Ordering::Relaxed);

    context.submit_one_shot(|cb| unsafe {
        let device = &context.device;
        if timed {
            device.cmd_reset_query_pool(cb, context.perf_query_pool, 0, 2);
            device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                context.perf_query_pool,
                0,
            );
        }
        for pass in passes {
            // Over-barriered on purpose (correctness first): every buffer the
            // pass touches becomes visible to every later consumer.
            let involved: Vec<&Buffer> = match pass {
                Pass::Compute { buffers, .. } => buffers.iter().collect(),
                Pass::CopyBuffer { src, dst } => vec![src, dst],
            };
            match pass {
                Pass::Compute { pipeline, buffers, push, groups } => {
                    if !reset.pipelines.iter().any(|p| std::ptr::eq(*p, *pipeline)) {
                        reset.pipelines.push(pipeline);
                    }
                    pipeline.record_pass(cb, buffers, push, *groups)?;
                }
                Pass::CopyBuffer { src, dst } => {
                    // F32/BH26: every call site copies a whole buffer; a size
                    // mismatch is a bug, not something to silently truncate.
                    // Return Err (not assert): a panic inside this closure would
                    // bypass submit_one_shot's cleanup and leave the buffer
                    // in the recording state.
                    if src.size != dst.size {
                        return Err(Error::Shader(format!(
                            "CopyBuffer size mismatch ({} vs {}): would silently truncate",
                            src.size, dst.size
                        )));
                    }
                    let region = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: src.size,
                    };
                    device.cmd_copy_buffer(cb, src.buffer, dst.buffer, std::slice::from_ref(&region));
                }
            }
            let barriers: Vec<vk::BufferMemoryBarrier<'_>> = involved
                .iter()
                .map(|b| {
                    buffer_barrier(
                        b.buffer,
                        b.size,
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::TRANSFER_READ
                            | vk::AccessFlags::HOST_READ,
                    )
                })
                .collect();
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
        }
        if timed {
            device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                context.perf_query_pool,
                1,
            );
        }
        Ok(())
    })?;

    // T9-lite: the fence already waited inside submit_one_shot, so the two
    // timestamps are ready. Accumulate the GPU-busy delta (ticks) for this
    // submit; the bench converts to ms via Context::gpu_elapsed_ms.
    if timed {
        unsafe {
            let mut ts = [0u64; 2];
            context
                .device
                .get_query_pool_results(
                    context.perf_query_pool,
                    0,
                    &mut ts,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(Error::Vulkan)?;
            context
                .perf_gpu_ns
                .fetch_add(ts[1].wrapping_sub(ts[0]), std::sync::atomic::Ordering::Relaxed);
        }
    }

    // BH1: descriptor-pool recycling happens in `reset`'s Drop (which also
    // covers the error paths), so there is nothing left to do but return; the
    // guard drops now, after the fence has already been waited inside
    // submit_one_shot (so no set is in-flight when its pool is reset).
    Ok(())
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            let device = &self.context.device;
            let _ = device.device_wait_idle();
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
