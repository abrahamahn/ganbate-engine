//! Portable scene math, geometry and object transforms, plus the data layer every tool shares:
//! JSON (read and canonical write), content digests and identifiers.

pub mod digest;
pub mod id;
pub mod json;
pub mod preview;
pub mod timing;
pub mod world;
pub mod xr;

pub use preview::{Camera, Rgb, Scene, Triangle, Vec3, Vertex};
