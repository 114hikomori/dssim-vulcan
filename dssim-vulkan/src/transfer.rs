//! Resource creation and staging transfer helpers (plan §4 Phase 1:
//! "staging upload / download … with correct rowPitch handling" — buffers
//! here; images come with the first image-based kernel).

use std::sync::Arc;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::context::Context;
use crate::{Error, Result};

/// A Vulkan buffer with its backing allocation. Freed on drop.
pub struct Buffer {
    context: Arc<Context>,
    pub(crate) buffer: vk::Buffer,
    pub(crate) allocation: Option<Allocation>,
    pub size: u64,
}

impl Buffer {
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if let Some(allocation) = self.allocation.take() {
                let mut allocator = self.context.allocator.lock().expect("allocator mutex poisoned");
                let _ = allocator
                    .as_mut()
                    .expect("allocator not yet dropped")
                    .free(allocation);
                self.context.device.destroy_buffer(self.buffer, None);
            }
        }
    }
}

impl Context {
    /// Allocate a buffer of `size` bytes with `usage`, in the given memory
    /// location. The buffer is debug-named `name`.
    pub(crate) fn alloc_buffer(
        self: &Arc<Self>,
        name: &str,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Buffer> {
        unsafe {
            let info = vk::BufferCreateInfo {
                size,
                usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let buffer = self.device.create_buffer(&info, None).map_err(Error::Vulkan)?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);

            let allocation = {
                let mut allocator = self.allocator.lock().expect("allocator mutex poisoned");
                allocator
                    .as_mut()
                    .expect("allocator not yet dropped")
                    .allocate(&AllocationCreateDesc {
                        name,
                        requirements,
                        location,
                        linear: true,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })
            };
            let allocation = match allocation {
                Ok(a) => a,
                Err(e) => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(Error::Allocator(e));
                }
            };

            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(Error::Vulkan)?;

            self.name_object(buffer, name);
            Ok(Buffer {
                context: self.clone(),
                buffer,
                allocation: Some(allocation),
                size,
            })
        }
    }

    /// Upload `data` into a fresh device-local buffer via a staging copy.
    pub fn upload_buffer(
        self: &std::sync::Arc<Self>,
        name: &str,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer> {
        let staging = self.alloc_buffer(
            &format!("{name}.staging"),
            data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        write_mapped(&staging, data)?;

        let dst = self.alloc_buffer(
            name,
            data.len() as u64,
            usage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
        )?;

        self.submit_one_shot(|cb| unsafe {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: data.len() as u64,
            };
            self.device.cmd_copy_buffer(cb, staging.buffer, dst.buffer(), std::slice::from_ref(&region));
            // Staging write -> device copy read, and copy write -> shader read.
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[
                    buffer_barrier(
                        dst.buffer(),
                        data.len() as u64,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ,
                    ),
                ],
                &[],
            );
            Ok(())
        })?;

        Ok(dst)
    }

    /// Copy a device-local buffer back to the host and return its contents.
    pub fn download_buffer(self: &std::sync::Arc<Self>, src: &Buffer) -> Result<Vec<u8>> {
        let readback = self.alloc_buffer(
            "readback",
            src.size,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )?;

        self.submit_one_shot(|cb| unsafe {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: src.size,
            };
            self.device.cmd_copy_buffer(cb, src.buffer, readback.buffer(), std::slice::from_ref(&region));
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[
                    buffer_barrier(
                        readback.buffer(),
                        src.size,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::HOST_READ,
                    ),
                ],
                &[],
            );
            Ok(())
        })?;

        read_bytes(&readback, src.size as usize)
    }
}

pub(crate) fn buffer_barrier<'a>(
    buffer: vk::Buffer,
    size: u64,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) -> vk::BufferMemoryBarrier<'a> {
    vk::BufferMemoryBarrier {
        src_access_mask: src_access,
        dst_access_mask: dst_access,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        buffer,
        offset: 0,
        size,
        ..Default::default()
    }
}

/// Write `data` into a host-mapped allocation.
pub(crate) fn write_mapped(buffer: &Buffer, data: &[u8]) -> Result<()> {
    let allocation = buffer.allocation.as_ref().expect("allocation present until drop");
    let ptr = allocation
        .mapped_ptr()
        .ok_or_else(|| Error::Allocator(gpu_allocator::AllocationError::FailedToMap("allocation not host-mapped".into())))?;
    debug_assert!(data.len() as u64 <= buffer.size, "staging write exceeds buffer size");
    // SAFETY: the allocation is at least `data.len()` bytes (we sized it),
    // host-visible (CpuToGpu), and currently mapped by gpu-allocator.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr().cast::<u8>(), data.len());
    }
    Ok(())
}

/// Read `len` bytes from a host-mapped allocation.
pub(crate) fn read_bytes(buffer: &Buffer, len: usize) -> Result<Vec<u8>> {
    let allocation = buffer.allocation.as_ref().expect("allocation present until drop");
    let ptr = allocation
        .mapped_ptr()
        .ok_or_else(|| Error::Allocator(gpu_allocator::AllocationError::FailedToMap("allocation not host-mapped".into())))?;
    // SAFETY: allocation is at least `len` bytes, host-visible (GpuToCpu),
    // mapped; the queue finished (fence) before this read.
    Ok(unsafe { std::slice::from_raw_parts(ptr.as_ptr().cast::<u8>(), len) }.to_vec())
}
