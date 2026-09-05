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

            // Enough sets for the dispatches this pipeline will do per pass;
            // grows via a new pool if ever exceeded (Phase B: fixed 64).
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: binding_count * 64,
            }];
            let descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(64)
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
        let pass = Pass {
            pipeline: self,
            buffers: buffers.to_vec(),
            push: push_constants.to_vec(),
            groups: count.div_ceil(64),
        };
        dispatch_sequence(&self.context, &[pass])
    }

    /// Record one bound-and-dispatched pass into an open command buffer.
    /// The descriptor set is allocated from this pipeline's pool and freed
    /// by the pool reset in [`dispatch_sequence`].
    fn record_pass(&self, cb: vk::CommandBuffer, buffers: &[&Buffer], push: &[u8], groups: u32) -> Result<()> {
        assert_eq!(buffers.len() as u32, self.binding_count, "buffer count must match binding count");
        if push.len() as u32 > self.push_constant_size {
            return Err(Error::Shader("push constant overflow".into()));
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

/// One pipeline pass inside a [`dispatch_sequence`].
pub struct Pass<'a> {
    pub pipeline: &'a ComputePipeline,
    pub buffers: Vec<&'a Buffer>,
    pub push: Vec<u8>,
    pub groups: u32,
}

/// Record all passes into ONE command buffer (barriers between passes keep
/// writes visible to the next pass), submit, and wait for the fence.
/// Determinism-over-speed: no overlap, no pipelining.
pub fn dispatch_sequence(context: &Arc<Context>, passes: &[Pass<'_>]) -> Result<()> {
    struct ResetPool<'a>(&'a ComputePipeline);
    let mut used_pipelines: Vec<ResetPool<'_>> = Vec::new();

    context.submit_one_shot(|cb| unsafe {
        let device = &context.device;
        for pass in passes {
            if !used_pipelines.iter().any(|ResetPool(p)| std::ptr::eq(*p, pass.pipeline)) {
                used_pipelines.push(ResetPool(pass.pipeline));
            }
            pass.pipeline.record_pass(cb, &pass.buffers, &pass.push, pass.groups)?;
            // Shader writes -> later reads (next pass / transfer / HOST).
            let barriers: Vec<vk::BufferMemoryBarrier<'_>> = pass
                .buffers
                .iter()
                .map(|b| {
                    buffer_barrier(
                        b.buffer,
                        b.size,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::TRANSFER_READ
                            | vk::AccessFlags::HOST_READ,
                    )
                })
                .collect();
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
        }
        Ok(())
    })?;

    // Descriptor sets were consumed; recycle the pools.
    for ResetPool(pipeline) in &used_pipelines {
        unsafe {
            pipeline
                .context
                .device
                .reset_descriptor_pool(pipeline.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(Error::Vulkan)?;
        }
    }
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
