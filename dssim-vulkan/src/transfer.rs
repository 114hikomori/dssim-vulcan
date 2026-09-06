//! Resource creation and staging transfer helpers (plan §4 Phase 1:
//! "staging upload / download … with correct rowPitch handling" — buffers
//! here; images come with the first image-based kernel).

use std::sync::Arc;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::context::Context;
use crate::{Error, Result};

/// A Vulkan buffer with its backing allocation. Freed on drop. Cloneable:
/// `Arc`-shared inner state means clones alias the same GPU memory (needed
/// so pipeline `Pass` records can own buffers that outlive local scopes).
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}

pub struct BufferInner {
    context: Arc<Context>,
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: u64,
}

impl std::ops::Deref for Buffer {
    type Target = BufferInner;
    fn deref(&self) -> &BufferInner {
        &self.inner
    }
}

impl Buffer {
    pub fn buffer(&self) -> vk::Buffer {
        self.inner.buffer
    }

    pub fn size(&self) -> u64 {
        self.inner.size
    }
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        unsafe {
            if let Some(allocation) = self.allocation.take() {
                // F28 measurement: this Buffer's bytes leave the live set.
                self.context
                    .live_bytes
                    .fetch_sub(self.size as usize, std::sync::atomic::Ordering::Relaxed);
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

            // F28 measurement: track live + peak allocation bytes (a
            // deterministic proxy for VRAM footprint).
            let live = self
                .live_bytes
                .fetch_add(size as usize, std::sync::atomic::Ordering::Relaxed)
                + size as usize;
            self.peak_bytes.fetch_max(live, std::sync::atomic::Ordering::Relaxed);

            self.name_object(buffer, name);
            Ok(Buffer {
                inner: Arc::new(BufferInner {
                    context: self.clone(),
                    buffer,
                    allocation: Some(allocation),
                    size,
                }),
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

/// Flush (host->device, after a write) or invalidate (device->host, before a
/// read) a host-mapped allocation's range. F1: correctness must not depend on
/// the allocation being HOST_COHERENT -- on coherent memory this is a no-op, so
/// we skip the driver call; on non-coherent memory it is required or the GPU
/// may read stale bytes / the host may read pre-invalidate garbage.
fn sync_host_range(buffer: &Buffer, flush: bool) -> Result<()> {
    let allocation = buffer.allocation.as_ref().expect("allocation present until drop");
    if allocation
        .memory_properties()
        .intersects(vk::MemoryPropertyFlags::HOST_COHERENT)
    {
        return Ok(());
    }
    let range = vk::MappedMemoryRange {
        memory: unsafe { allocation.memory() },
        offset: allocation.offset(),
        size: allocation.size(),
        ..Default::default()
    };
    let device = &buffer.context.device;
    unsafe {
        if flush {
            device.flush_mapped_memory_ranges(&[range])
        } else {
            device.invalidate_mapped_memory_ranges(&[range])
        }
    }
    .map_err(Error::Vulkan)?;
    Ok(())
}

/// Write `data` into a host-mapped allocation, then flush (F1).
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
    sync_host_range(buffer, true)
}

/// Fill a host-mapped staging buffer's first `count` f32 slots directly via
/// `f`, avoiding an intermediate `Vec<f32>`/`Vec<u8>` and a second memcpy
/// (the small-image `create_image` upload was dominated by exactly those).
/// Flushes afterwards (F1).
pub(crate) fn write_mapped_f32_with<F: FnOnce(&mut [f32])>(buffer: &Buffer, count: usize, f: F) -> Result<()> {
    let allocation = buffer.allocation.as_ref().expect("allocation present until drop");
    let ptr = allocation
        .mapped_ptr()
        .ok_or_else(|| Error::Allocator(gpu_allocator::AllocationError::FailedToMap("allocation not host-mapped".into())))?;
    debug_assert!((count * 4) as u64 <= buffer.size, "staging f32 write exceeds buffer size");
    // SAFETY: allocation is host-visible, mapped, and at least `count*4` bytes.
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().cast::<f32>(), count) };
    f(slice);
    sync_host_range(buffer, true)
}

/// Read `len` bytes from a host-mapped allocation, invalidating first (F1).
pub(crate) fn read_bytes(buffer: &Buffer, len: usize) -> Result<Vec<u8>> {
    sync_host_range(buffer, false)?;
    let allocation = buffer.allocation.as_ref().expect("allocation present until drop");
    let ptr = allocation
        .mapped_ptr()
        .ok_or_else(|| Error::Allocator(gpu_allocator::AllocationError::FailedToMap("allocation not host-mapped".into())))?;
    // SAFETY: allocation is at least `len` bytes, host-visible (GpuToCpu),
    // mapped; the queue finished (fence) before this read.
    Ok(unsafe { std::slice::from_raw_parts(ptr.as_ptr().cast::<u8>(), len) }.to_vec())
}

#[cfg(test)]
mod upload_diag {
    // Diagnostic (run manually -- timing is machine-dependent, so #[ignore]):
    // is the create_image upload cost first-touch page faults on the fresh
    // per-scale staging allocation, or the writes themselves? Writes 64 MB
    // (~scale-0 staging at 2048^2) to a fresh buffer, then again to the SAME
    // (now-resident) buffer. If first >> warm, the cure is a persistent
    // pre-touched staging ring (T2), not a faster interleave.
    //   cargo test -p dssim-vulkan upload_diag -- --ignored --nocapture
    use super::*;
    use std::time::Instant;

    fn fill(dst: &mut [f32]) {
        for (i, d) in dst.iter_mut().enumerate() {
            *d = (i & 1023) as f32;
        }
    }

    fn timed_write(ctx: &Arc<Context>, bytes: usize, n: usize, tag: &str) {
        let b = ctx
            .alloc_buffer(
                "diag.staging",
                bytes as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
            )
            .unwrap();
        let t = Instant::now();
        write_mapped_f32_with(&b, n, fill).unwrap();
        eprintln!("{tag} first-write-to-fresh={:.2}ms", t.elapsed().as_secs_f64() * 1000.0);
        let t = Instant::now();
        write_mapped_f32_with(&b, n, fill).unwrap();
        eprintln!("{tag} warm-write(same-buffer)={:.2}ms", t.elapsed().as_secs_f64() * 1000.0);
        let t = Instant::now();
        write_mapped_f32_with(&b, n, fill).unwrap();
        eprintln!("{tag} warm-write-2={:.2}ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    #[test]
    #[ignore = "timing diagnostic; run with --ignored --nocapture"]
    fn first_touch_vs_warm() {
        let ctx = Arc::new(Context::new().expect("context"));
        eprintln!("device: {}", ctx.device_name());
        let bytes = 64 * 1024 * 1024;
        timed_write(&ctx, bytes, bytes / 4, "64MB");
        let bytes2 = 8 * 1024 * 1024;
        timed_write(&ctx, bytes2, bytes2 / 4, "8MB");

        // Is the ~2.3GB/s the WC mapped memory, or the fill loop's own CPU
        // cost? Same loop into a plain cached Vec<f32> on the heap.
        let n = bytes / 4;
        let mut v = vec![0f32; n];
        let t = Instant::now();
        fill(&mut v);
        eprintln!("cached-Vec same-loop={:.2}ms", t.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(&v);
    }
}
