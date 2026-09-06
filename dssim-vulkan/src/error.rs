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
    /// BH3: a public-API caller passed arguments the backend can't honor (e.g.
    /// a mismatched-size pair to `create_image_pair`). Distinct from `Shader`
    /// (not a pipeline issue) and surfaced as `Err`, not a panic.
    InvalidInput(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Loader(e) => write!(f, "Vulkan loader: {e}"),
            Error::Vulkan(e) => write!(f, "Vulkan error: {e}"),
            Error::Allocator(e) => write!(f, "allocator: {e}"),
            Error::NoDevice => write!(f, "no compute-capable Vulkan device found"),
            Error::Shader(msg) => write!(f, "shader error: {msg}"),
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    // F6: expose the wrapped cause so callers (e.g. main.rs's `e.source()`)
    // can print the underlying loader/Vulkan/allocator error, not just the
    // top-level message. Critical for field diagnosis of device-lost /
    // allocator failures.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Loader(e) => Some(e),
            Error::Vulkan(e) => Some(e),
            Error::Allocator(e) => Some(e),
            Error::NoDevice => None,
            Error::Shader(_) => None,
            Error::InvalidInput(_) => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
