//! Ganbate's portable scene data and camera. Game-specific content belongs to clients.

pub mod anim;
#[cfg(target_vendor = "apple")]
mod apple;
pub mod cli;
pub mod gpu;
#[cfg(target_vendor = "apple")]
mod metal;
pub mod render;
#[cfg(any(target_os = "linux", target_os = "windows", target_os = "android"))]
pub mod vulkan;
#[cfg(any(
    target_os = "linux",
    target_os = "windows",
    target_os = "macos",
    target_os = "ios"
))]
pub mod window;

pub use ganbate_assets as assets;
pub use ganbate_core::*;
