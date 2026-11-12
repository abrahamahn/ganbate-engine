//! Ganbate's stylized renderer: soft cel shading, a time-of-day sky and weather, water reflections
//! and painterly post-processing, built on the backend-neutral [`crate::gpu`] layer and shaders
//! compiled by Ganbate's own compiler. The look is data ([`Look`], usually from a game's Look
//! file); nothing game-specific lives here.
/// A generational handle type (ids are never 0).
macro_rules! handle {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(u32);
        impl $name {
            /// The raw id, for hosts that pass handles across a language boundary.
            pub const fn raw(self) -> u32 {
                self.0
            }
            pub const fn from_raw(raw: u32) -> Self {
                Self(raw)
            }
        }
    };
}

pub mod ao;
#[cfg(test)]
mod bench;
mod lights;
pub mod look;
pub mod math;
mod models;
mod props;
mod renderer;
mod scene;
pub mod sky;

pub use lights::{
    GameLight, LightDesc, MAX_GAME_LIGHTS, MAX_SHADOW_CONE, MAX_SHADOWED_LIGHTS, SpotCone,
};
pub use look::{
    CameraExposure, DetailPreset, DetailRule, Focus, Grass, Look, LookFile, SurfaceModel,
    SurfaceRule, Terrain, ToneMap, View,
};
pub use models::{ModelId, ModelInstance};
pub use props::{
    BlendMode, MaterialDesc, MeshData, PROP_SHADER_PARAMS, Pick, PropInstance, PropMaterial,
    PropMesh, PropShader, PropTexture, WorldPick,
};
pub use renderer::{FrameStats, Output, Quality, Renderer};
pub use sky::{Environment, Palette, SkyKey, TimeOfDay, Weather};
