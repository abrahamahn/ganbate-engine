//! 3D Spatial Audio Attenuation, Listener Azimuth Panning, and Air-Absorption Filtering.
//!
//! Uses `ganbate_core::Vec3` and `ganbate_core::Camera` (+Y up, right-handed world space) to
//! compute distance rolloff, stereo azimuth pan (`-1.0..+1.0`), and distance-dependent air
//! absorption low-pass cutoff with zero genre or scene hardcoding.

use ganbate_core::{Camera, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistanceModel {
    /// Web Audio / OpenAL clamped inverse distance:
    /// `ref / (ref + rolloff * (clamp(d, ref, max) - ref))`
    InverseClamped,
    /// Clamped linear falloff from `1.0` at `ref_distance` to `0.0` at `max_distance`.
    LinearClamped,
    /// Clamped exponential rolloff `(clamp(d, ref, max) / ref) ^ (-rolloff)`.
    ExponentialClamped,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Listener3D {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
}

impl Default for Listener3D {
    fn default() -> Self {
        Self {
            position: Vec3(0.0, 0.0, 0.0),
            forward: Vec3(0.0, 0.0, -1.0),
            up: Vec3(0.0, 1.0, 0.0),
        }
    }
}

impl Listener3D {
    /// Constructs a 3D audio listener aligned with a `ganbate_core::Camera`.
    pub fn from_camera(camera: &Camera) -> Self {
        let forward = camera
            .target
            .minus(camera.eye)
            .unit()
            .unwrap_or(Vec3(0.0, 0.0, -1.0));
        Self {
            position: camera.eye,
            forward,
            up: Vec3(0.0, 1.0, 0.0),
        }
    }

    /// Right-handed listener right unit vector (`forward x up`).
    pub fn right_axis(&self) -> Vec3 {
        let fwd = self.forward.unit().unwrap_or(Vec3(0.0, 0.0, -1.0));
        let up = self.up.unit().unwrap_or(Vec3(0.0, 1.0, 0.0));
        fwd.cross(up).unit().unwrap_or(Vec3(1.0, 0.0, 0.0))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialEmitter3D {
    pub position: Vec3,
    pub ref_distance: f32,
    pub max_distance: f32,
    pub rolloff_factor: f32,
    pub model: DistanceModel,
    /// High-frequency air absorption strength (`0.0 = disabled`, `1.0 = outdoor/humid air`).
    pub air_absorption: f32,
}

impl Default for SpatialEmitter3D {
    fn default() -> Self {
        Self {
            position: Vec3(0.0, 0.0, 0.0),
            ref_distance: 1.0,
            max_distance: 40.0,
            rolloff_factor: 1.0,
            model: DistanceModel::InverseClamped,
            air_absorption: 0.35,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialMixParams {
    pub distance: f32,
    pub distance_gain: f32,
    pub stereo_pan: f32,
    pub lowpass_cutoff_hz: f32,
}

/// Evaluates 3D distance attenuation, stereo azimuth pan, and air-absorption LPF cutoff.
pub fn evaluate_spatial_3d(listener: &Listener3D, emitter: &SpatialEmitter3D) -> SpatialMixParams {
    let delta = emitter.position.minus(listener.position);
    let dist_sq = delta.dot(delta);
    let distance = if dist_sq.is_finite() && dist_sq > 0.0 {
        dist_sq.sqrt()
    } else {
        0.0
    };

    let ref_d = emitter.ref_distance.max(0.05);
    let max_d = emitter.max_distance.max(ref_d + 0.05);
    let rolloff = emitter.rolloff_factor.clamp(0.0, 16.0);
    let clamped_d = distance.clamp(ref_d, max_d);

    let distance_gain = match emitter.model {
        DistanceModel::InverseClamped => {
            let denom = ref_d + rolloff * (clamped_d - ref_d);
            if denom > 1e-6 {
                (ref_d / denom).clamp(0.0, 1.0)
            } else {
                1.0
            }
        }
        DistanceModel::LinearClamped => {
            let span = (max_d - ref_d).max(1e-5);
            (1.0 - rolloff * (clamped_d - ref_d) / span).clamp(0.0, 1.0)
        }
        DistanceModel::ExponentialClamped => (clamped_d / ref_d).powf(-rolloff).clamp(0.0, 1.0),
    };

    // Collapse harsh hard-panning when an emitter passes directly through the listener's head
    let stereo_pan = if let Some(dir) = delta.unit() {
        let right = listener.right_axis();
        let near_blend = (distance / ref_d).clamp(0.0, 1.0);
        (dir.dot(right) * near_blend).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    let air = emitter.air_absorption.clamp(0.0, 1.0);
    let norm_dist = ((clamped_d - ref_d) / (max_d - ref_d).max(1e-5)).clamp(0.0, 1.0);
    let lowpass_cutoff_hz = (20_000.0 * (1.0 - 0.88 * air * norm_dist)).clamp(1_800.0, 20_000.0);

    SpatialMixParams {
        distance,
        distance_gain,
        stereo_pan,
        lowpass_cutoff_hz,
    }
}
