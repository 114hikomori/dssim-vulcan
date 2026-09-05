//! Error type shared by the Vulkan backend.

#[derive(Debug)]
pub enum Error {
    /// The Vulkan loader (`vulkan-1.dll` / `libvulkan.so`) could not be loaded.
    Loader(ash::LoadingError),
    /// A Vulkan API call returned an error result.
    Vulkan(ash::vk::Result),
    /// The memory allocator failed.
    Allocator(gpu_allocator::AllocationError),
    /// No suitable compute-capable Vulkan device was found.
    NoDevice,
    /// A shader or pipeline problem.
    Shader(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Loader(e) => write!(f, "Vulkan loader: {e}"),
            Error::Vulkan(e) => write!(f, "Vulkan error: {e}"),
            Error::Allocator(e) => write!(f, "allocator: {e}"),
            Error::NoDevice => write!(f, "no compute-capable Vulkan device found"),
            Error::Shader(msg) => write!(f, "shader error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
