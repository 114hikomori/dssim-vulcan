//! Vulkan context: instance, validation, device selection, compute queue,
//! command pool, and memory allocator (plan §6 Phase B / VULKAN_PORT_PLAN.md
//! §4 Phase 1).

use std::ffi::CStr;
use std::sync::{Mutex, MutexGuard};

use ash::ext::debug_utils;
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::Error;

pub(crate) const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Instance/device creation is serialized: some drivers' enumeration paths
/// misbehave under concurrent instance creation (observed: AMD Windows driver
/// returning INCOMPLETE during parallel test startup).
static CREATE_LOCK: Mutex<()> = Mutex::new(());

fn create_lock() -> MutexGuard<'static, ()> {
    CREATE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// An initialized Vulkan context bound to one physical device.
///
/// Selection policy (plan §3.1): discrete > integrated > CPU/lavapipe,
/// requiring a compute-capable queue family.
pub struct Context {
    instance: ash::Instance,
    /// Instance-level VK_EXT_debug_utils loader + validation messenger.
    debug_instance_loader: Option<debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    /// Device-level VK_EXT_debug_utils loader for object naming.
    pub(crate) debug_device_loader: Option<debug_utils::Device>,
    /// Needed by future image/feature queries (Phase C+).
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    device_name: String,
    device_type: vk::PhysicalDeviceType,
    /// All compute-capable devices found at creation, best first.
    pub device_candidates: Vec<String>,
    pub(crate) device: ash::Device,
    /// Needed for multi-queue sync decisions (Phase E).
    #[allow(dead_code)]
    pub(crate) queue_family_index: u32,
    queue: vk::Queue,
    pub(crate) command_pool: vk::CommandPool,
    /// gpu-allocator wants `&mut` and must be dropped before the device
    /// (its Drop frees remaining memory blocks), hence `Option` + explicit
    /// teardown order in `Drop`.
    pub(crate) allocator: Mutex<Option<Allocator>>,
    /// Must be the LAST field: dropping `Entry` unloads the Vulkan library,
    /// and every other field's teardown calls into it first.
    #[allow(dead_code)] // read implicitly: must outlive all other fields' Drop
    entry: ash::Entry,
}

impl Context {
    /// Create a context on the best device per the selection policy.
    pub fn new() -> Result<Self, Error> {
        Self::new_inner(None)
    }

    /// Create a context pinned to a specific candidate index (as listed in
    /// `device_candidates`). Useful for testing across multiple GPUs.
    pub fn new_with_device(candidate_index: usize) -> Result<Self, Error> {
        Self::new_inner(Some(candidate_index))
    }

    fn new_inner(pin: Option<usize>) -> Result<Self, Error> {
        let _serial = create_lock();
        unsafe {
            let entry = ash::Entry::load().map_err(Error::Loader)?;

            // Validation layers in development builds (disable with
            // DSSIM_VK_NO_VALIDATION), only if the layer is installed.
            let want_validation =
                cfg!(debug_assertions) && std::env::var_os("DSSIM_VK_NO_VALIDATION").is_none();
            let instance_layers: Vec<std::ffi::CString> = entry
                .enumerate_instance_layer_properties()
                .map_err(Error::Vulkan)?
                .iter()
                .filter_map(|l| l.layer_name_as_c_str().ok().map(|c| c.to_owned()))
                .collect();
            let validation_enabled =
                want_validation && instance_layers.iter().any(|l| l.as_c_str() == VALIDATION_LAYER);
            // F26: make the validation state explicit so a silent skip (layer
            // present on disk but not registered / VK_LAYER_PATH unset) is
            // visible instead of quietly weakening every "GPU-validated" claim.
            if cfg!(debug_assertions) {
                if validation_enabled {
                    eprintln!("[dssim-vulkan] validation layer: ENABLED");
                } else if want_validation {
                    eprintln!(
                        "[dssim-vulkan] validation layer: WANTED but NOT FOUND \
                         (set VK_LAYER_PATH to the SDK's Bin dir to enable it)"
                    );
                }
            }

            let instance_extensions = entry
                .enumerate_instance_extension_properties(None)
                .map_err(Error::Vulkan)?;
            let has_debug_utils = instance_extensions
                .iter()
                .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == c"VK_EXT_debug_utils"));

            let extension_names: Vec<_> = has_debug_utils
                .then(|| c"VK_EXT_debug_utils".as_ptr())
                .into_iter()
                .collect();
            let layer_names: Vec<_> = validation_enabled
                .then_some(VALIDATION_LAYER.as_ptr())
                .into_iter()
                .collect();

            let app_info = vk::ApplicationInfo {
                api_version: vk::make_api_version(0, 1, 3, 0),
                ..Default::default()
            };
            let create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                pp_enabled_layer_names: layer_names.as_ptr(),
                enabled_layer_count: layer_names.len() as u32,
                pp_enabled_extension_names: extension_names.as_ptr(),
                enabled_extension_count: extension_names.len() as u32,
                ..Default::default()
            };
            let instance = entry.create_instance(&create_info, None).map_err(Error::Vulkan)?;

            // Debug messenger for validation output.
            let (debug_instance_loader, debug_messenger) = if has_debug_utils {
                let loader = debug_utils::Instance::new(&entry, &instance);
                let messenger = loader
                    .create_debug_utils_messenger(&debug_messenger_create_info(), None)
                    .map_err(Error::Vulkan)?;
                (Some(loader), Some(messenger))
            } else {
                (None, None)
            };

            // Enumerate and rank devices: discrete > integrated > everything
            // else; all candidates must have a compute queue family.
            let physical_devices = instance
                .enumerate_physical_devices()
                .map_err(Error::Vulkan)?;
            if physical_devices.is_empty() {
                return Err(Error::NoDevice);
            }
            let mut candidates: Vec<_> = physical_devices
                .into_iter()
                .filter_map(|pd| {
                    let props = instance.get_physical_device_properties(pd);
                    let queue_family_index = queue_family_with_compute(&instance, pd)?;
                    let name = props
                        .device_name_as_c_str()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned();
                    Some((device_type_rank(props.device_type), name, pd, queue_family_index))
                })
                .collect();
            candidates.sort_by_key(|(rank, name, ..)| std::cmp::Reverse((*rank, name.clone())));
            let device_candidates: Vec<String> =
                candidates.iter().map(|(_, name, ..)| name.clone()).collect();

            let (_, _, physical_device, queue_family_index) = match pin {
                Some(i) => {
                    let (rank, name, pd, qfi) = candidates.get(i).ok_or(Error::NoDevice)?;
                    (*rank, name.clone(), *pd, *qfi)
                }
                None => {
                    let (rank, name, pd, qfi) = candidates.first().ok_or(Error::NoDevice)?;
                    (*rank, name.clone(), *pd, *qfi)
                }
            };
            let props = instance.get_physical_device_properties(physical_device);
            let device_name = props
                .device_name_as_c_str()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let device_type = props.device_type;

            let queue_priorities = [1.0];
            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);
            let device_info = vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
            let device = instance
                .create_device(physical_device, &device_info, None)
                .map_err(Error::Vulkan)?;
            let queue = device.get_device_queue(queue_family_index, 0);

            let command_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(queue_family_index),
                    None,
                )
                .map_err(Error::Vulkan)?;

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .map_err(Error::Allocator)?;            let debug_device_loader = has_debug_utils.then(|| debug_utils::Device::new(&instance, &device));

            // Name the physical device for RenderDoc/Nsight sessions.
            if let (Some(loader), Some(_)) = (&debug_device_loader, has_debug_utils.then_some(())) {
                name_object(loader, physical_device, "dssim:physical_device");
            }

            Ok(Self {
                entry,
                instance,
                debug_instance_loader,
                debug_messenger,
                debug_device_loader,
                physical_device,
                device_name,
                device_type,
                device_candidates,
                device,
                queue_family_index,
                queue,
                command_pool,
                allocator: Mutex::new(Some(allocator)),
            })
        }
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn device_type(&self) -> vk::PhysicalDeviceType {
        self.device_type
    }

    /// Name any Vulkan object for debugging tools, when the instance supports
    /// VK_EXT_debug_utils.
    pub(crate) fn name_object<T: vk::Handle>(&self, object: T, name: &str) {
        if let Some(loader) = &self.debug_device_loader {
            name_object(loader, object, name);
        }
    }

    /// Record a one-shot command buffer on the compute queue and wait for it.
    /// `record` may fail (e.g. descriptor allocation); the error aborts the
    /// partially-recorded buffer and is returned.
    pub(crate) fn submit_one_shot(
        &self,
        record: impl FnOnce(vk::CommandBuffer) -> Result<(), Error>,
    ) -> Result<(), Error> {
        unsafe {
            let cb = self
                .device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                    command_pool: self.command_pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: 1,
                    ..Default::default()
                })
                .map_err(Error::Vulkan)?[0];

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(Error::Vulkan)?;

            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            let recorded = (|| {
                self.device.begin_command_buffer(cb, &begin_info).map_err(Error::Vulkan)?;
                record(cb)?;
                self.device.end_command_buffer(cb).map_err(Error::Vulkan)
            })();
            if let Err(e) = recorded {
                self.device.destroy_fence(fence, None);
                self.device.free_command_buffers(self.command_pool, std::slice::from_ref(&cb));
                return Err(e);
            }

            let submit = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: std::slice::from_ref(&cb).as_ptr(),
                ..Default::default()
            };
            let result = (|| -> Result<(), Error> {
                self.device
                    .queue_submit(self.queue, std::slice::from_ref(&submit), fence)
                    .map_err(Error::Vulkan)?;
                self.device
                    .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                    .map_err(Error::Vulkan)?;
                Ok(())
            })();
            // F2: release the fence + command buffer on EVERY path, including
            // a submit/wait error -- the previous `?` skipped this cleanup and
            // leaked both.
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, std::slice::from_ref(&cb));
            result
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            // Drop the allocator while the device is still valid: its Drop
            // frees any remaining memory blocks through vkFreeMemory.
            if let Ok(mut guard) = self.allocator.lock() {
                drop(guard.take());
            }
            if let (Some(loader), Some(messenger)) = (&self.debug_instance_loader, self.debug_messenger) {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
            // `entry` drops last (field order), unloading the library.
        }
    }
}

fn debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT {
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        pfn_user_callback: Some(debug_callback),
        ..Default::default()
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut core::ffi::c_void,
) -> vk::Bool32 {
    let sev = if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR { "ERROR" } else { "WARN" };
    let msg = unsafe {
        if callback_data.is_null() {
            String::new()
        } else {
            let p = (*callback_data).p_message;
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    };
    eprintln!("[vulkan {sev}] {msg}");
    vk::FALSE
}

/// Attach a debug name to a Vulkan object (plan §4 Phase 1: "debug naming").
pub(crate) fn name_object<T: vk::Handle>(
    loader: &debug_utils::Device,
    object: T,
    name: &str,
) {
    let mut name_buf = name.as_bytes().to_vec();
    name_buf.push(0);
    let name_c = CStr::from_bytes_with_nul(&name_buf).unwrap_or_default();
    let name_info = vk::DebugUtilsObjectNameInfoEXT {
        object_type: T::TYPE,
        object_handle: object.as_raw(),
        p_object_name: name_c.as_ptr(),
        ..Default::default()
    };
    // Best effort: naming is diagnostics, never load-bearing.
    let _ = unsafe { loader.set_debug_utils_object_name(&name_info) };
}

fn device_type_rank(t: vk::PhysicalDeviceType) -> u32 {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => 3,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
        _ => 1,
    }
}

fn queue_family_with_compute(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Option<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|i| i as u32)
}
