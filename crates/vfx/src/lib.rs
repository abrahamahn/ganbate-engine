//! Data-driven 3D particle emitter specification (`ganbate.vfx/1`) and deterministic simulator.

pub mod noise;

use std::collections::BTreeMap;

use ganbate_core::json::{self, Value as JsonValue};
pub use noise::{SplitMix64, curl_noise_3d, scalar_noise_grad_3d};

pub const VFX_SCHEMA: &str = "ganbate.vfx/1";
pub const DEFAULT_FIXED_DT: f32 = 1.0 / 60.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendMode {
    #[default]
    Alpha,
    Additive,
}

impl BlendMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Alpha => "alpha",
            Self::Additive => "additive",
        }
    }

    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "alpha" => Ok(Self::Alpha),
            "additive" => Ok(Self::Additive),
            other => Err(format!("unsupported blendMode '{other}'")),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SpawnShape {
    #[default]
    Point,
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: [f32; 3],
    },
    Cone {
        angle_deg: f32,
        radius: f32,
    },
    Disc {
        radius: f32,
    },
}

impl SpawnShape {
    pub fn validate(&self) -> Result<(), String> {
        match *self {
            Self::Point => Ok(()),
            Self::Sphere { radius } | Self::Disc { radius } => {
                if !radius.is_finite() || radius < 0.0 {
                    return Err("shape radius must be finite and >= 0".into());
                }
                Ok(())
            }
            Self::Box { half_extents } => {
                if !half_extents.into_iter().all(|v| v.is_finite() && v >= 0.0) {
                    return Err("box halfExtents must be finite and >= 0".into());
                }
                Ok(())
            }
            Self::Cone { angle_deg, radius } => {
                if !angle_deg.is_finite() || !(0.0..=180.0).contains(&angle_deg) {
                    return Err("cone angleDeg must lie in 0..=180".into());
                }
                if !radius.is_finite() || radius < 0.0 {
                    return Err("cone radius must be finite and >= 0".into());
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Burst {
    pub time_s: f32,
    pub count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurlNoiseSpec {
    pub amplitude: f32,
    pub frequency: f32,
    pub scroll_speed: f32,
}

impl Default for CurlNoiseSpec {
    fn default() -> Self {
        Self {
            amplitude: 0.0,
            frequency: 1.0,
            scroll_speed: 0.4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarPoint {
    pub t: f32,
    pub value: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorStop {
    pub t: f32,
    pub rgba: [f32; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmitterSpec {
    pub id: String,
    pub enabled: bool,
    pub blend_mode: BlendMode,
    pub max_particles: u32,
    pub spawn_rate: f32,
    pub bursts: Vec<Burst>,
    pub lifetime_s: [f32; 2],
    pub shape: SpawnShape,
    pub offset: [f32; 3],
    pub direction: [f32; 3],
    pub speed: [f32; 2],
    pub gravity: [f32; 3],
    pub drag: f32,
    pub curl_noise: CurlNoiseSpec,
    pub initial_rotation_deg: [f32; 2],
    pub angular_velocity_deg: [f32; 2],
    pub size_base: [f32; 2],
    pub size_over_life: Vec<ScalarPoint>,
    pub color_over_life: Vec<ColorStop>,
}

impl Default for EmitterSpec {
    fn default() -> Self {
        Self {
            id: "emitter_0".into(),
            enabled: true,
            blend_mode: BlendMode::Alpha,
            max_particles: 128,
            spawn_rate: 16.0,
            bursts: Vec::new(),
            lifetime_s: [1.0, 2.0],
            shape: SpawnShape::Point,
            offset: [0.0, 0.0, 0.0],
            direction: [0.0, 1.0, 0.0],
            speed: [0.4, 1.2],
            gravity: [0.0, -1.5, 0.0],
            drag: 0.25,
            curl_noise: CurlNoiseSpec::default(),
            initial_rotation_deg: [-180.0, 180.0],
            angular_velocity_deg: [-45.0, 45.0],
            size_base: [0.03, 0.06],
            size_over_life: vec![
                ScalarPoint { t: 0.0, value: 0.4 },
                ScalarPoint { t: 0.2, value: 1.0 },
                ScalarPoint { t: 1.0, value: 0.0 },
            ],
            color_over_life: vec![
                ColorStop {
                    t: 0.0,
                    rgba: [1.0, 1.0, 1.0, 0.0],
                },
                ColorStop {
                    t: 0.15,
                    rgba: [1.0, 1.0, 1.0, 0.9],
                },
                ColorStop {
                    t: 1.0,
                    rgba: [1.0, 1.0, 1.0, 0.0],
                },
            ],
        }
    }
}

impl EmitterSpec {
    /// Gentle drifting botanical petals with wide horizontal canopy and curl turbulence.
    pub fn preset_blossom_drift() -> Self {
        Self {
            id: "blossom_drift".into(),
            enabled: true,
            blend_mode: BlendMode::Alpha,
            max_particles: 160,
            spawn_rate: 18.0,
            bursts: Vec::new(),
            lifetime_s: [4.5, 7.0],
            shape: SpawnShape::Box {
                half_extents: [3.5, 0.4, 2.5],
            },
            offset: [0.0, 3.2, 0.0],
            direction: [0.25, -0.9, 0.15],
            speed: [0.25, 0.55],
            gravity: [0.08, -0.32, 0.03],
            drag: 0.85,
            curl_noise: CurlNoiseSpec {
                amplitude: 0.55,
                frequency: 0.65,
                scroll_speed: 0.35,
            },
            initial_rotation_deg: [-180.0, 180.0],
            angular_velocity_deg: [-95.0, 95.0],
            size_base: [0.028, 0.048],
            size_over_life: vec![
                ScalarPoint { t: 0.0, value: 0.6 },
                ScalarPoint { t: 0.2, value: 1.0 },
                ScalarPoint {
                    t: 0.85,
                    value: 0.95,
                },
                ScalarPoint { t: 1.0, value: 0.3 },
            ],
            color_over_life: vec![
                ColorStop {
                    t: 0.0,
                    rgba: [0.99, 0.82, 0.88, 0.0],
                },
                ColorStop {
                    t: 0.12,
                    rgba: [0.99, 0.82, 0.88, 0.92],
                },
                ColorStop {
                    t: 0.85,
                    rgba: [0.96, 0.72, 0.82, 0.82],
                },
                ColorStop {
                    t: 1.0,
                    rgba: [0.92, 0.64, 0.76, 0.0],
                },
            ],
        }
    }

    /// Rising additive warm embers with buoyant updraft and curl swirl.
    pub fn preset_ember_updraft() -> Self {
        Self {
            id: "ember_updraft".into(),
            enabled: true,
            blend_mode: BlendMode::Additive,
            max_particles: 96,
            spawn_rate: 22.0,
            bursts: Vec::new(),
            lifetime_s: [1.6, 3.0],
            shape: SpawnShape::Disc { radius: 0.35 },
            offset: [0.0, 0.0, 0.0],
            direction: [0.0, 1.0, 0.0],
            speed: [0.45, 1.15],
            gravity: [0.0, 0.45, 0.0],
            drag: 0.4,
            curl_noise: CurlNoiseSpec {
                amplitude: 0.75,
                frequency: 1.35,
                scroll_speed: 0.8,
            },
            initial_rotation_deg: [0.0, 360.0],
            angular_velocity_deg: [-140.0, 140.0],
            size_base: [0.012, 0.026],
            size_over_life: vec![
                ScalarPoint { t: 0.0, value: 0.5 },
                ScalarPoint {
                    t: 0.25,
                    value: 1.0,
                },
                ScalarPoint { t: 1.0, value: 0.0 },
            ],
            color_over_life: vec![
                ColorStop {
                    t: 0.0,
                    rgba: [1.0, 0.92, 0.58, 0.0],
                },
                ColorStop {
                    t: 0.1,
                    rgba: [1.0, 0.78, 0.32, 0.95],
                },
                ColorStop {
                    t: 0.7,
                    rgba: [0.96, 0.38, 0.12, 0.7],
                },
                ColorStop {
                    t: 1.0,
                    rgba: [0.55, 0.12, 0.04, 0.0],
                },
            ],
        }
    }

    /// High-energy radial celebratory burst with crisp gravity arc.
    pub fn preset_radial_burst() -> Self {
        Self {
            id: "radial_burst".into(),
            enabled: true,
            blend_mode: BlendMode::Additive,
            max_particles: 120,
            spawn_rate: 0.0,
            bursts: vec![Burst {
                time_s: 0.0,
                count: 72,
            }],
            lifetime_s: [0.65, 1.35],
            shape: SpawnShape::Cone {
                angle_deg: 58.0,
                radius: 0.08,
            },
            offset: [0.0, 0.05, 0.0],
            direction: [0.0, 1.0, 0.0],
            speed: [1.8, 3.8],
            gravity: [0.0, -4.8, 0.0],
            drag: 0.9,
            curl_noise: CurlNoiseSpec {
                amplitude: 0.3,
                frequency: 1.8,
                scroll_speed: 0.6,
            },
            initial_rotation_deg: [-180.0, 180.0],
            angular_velocity_deg: [-240.0, 240.0],
            size_base: [0.018, 0.038],
            size_over_life: vec![
                ScalarPoint { t: 0.0, value: 1.0 },
                ScalarPoint { t: 0.5, value: 0.8 },
                ScalarPoint { t: 1.0, value: 0.0 },
            ],
            color_over_life: vec![
                ColorStop {
                    t: 0.0,
                    rgba: [1.0, 0.96, 0.78, 1.0],
                },
                ColorStop {
                    t: 0.45,
                    rgba: [0.96, 0.76, 0.28, 0.9],
                },
                ColorStop {
                    t: 1.0,
                    rgba: [0.85, 0.42, 0.14, 0.0],
                },
            ],
        }
    }

    /// Soft expanding volumetric mist plume.
    pub fn preset_mist_plume() -> Self {
        Self {
            id: "mist_plume".into(),
            enabled: true,
            blend_mode: BlendMode::Alpha,
            max_particles: 64,
            spawn_rate: 8.0,
            bursts: Vec::new(),
            lifetime_s: [3.0, 5.0],
            shape: SpawnShape::Sphere { radius: 0.45 },
            offset: [0.0, 0.2, 0.0],
            direction: [0.0, 0.4, 0.0],
            speed: [0.08, 0.22],
            gravity: [0.0, 0.04, 0.0],
            drag: 1.2,
            curl_noise: CurlNoiseSpec {
                amplitude: 0.22,
                frequency: 0.45,
                scroll_speed: 0.18,
            },
            initial_rotation_deg: [-180.0, 180.0],
            angular_velocity_deg: [-18.0, 18.0],
            size_base: [0.22, 0.42],
            size_over_life: vec![
                ScalarPoint { t: 0.0, value: 0.5 },
                ScalarPoint { t: 0.5, value: 1.0 },
                ScalarPoint {
                    t: 1.0,
                    value: 1.35,
                },
            ],
            color_over_life: vec![
                ColorStop {
                    t: 0.0,
                    rgba: [0.76, 0.82, 0.92, 0.0],
                },
                ColorStop {
                    t: 0.3,
                    rgba: [0.76, 0.82, 0.92, 0.22],
                },
                ColorStop {
                    t: 1.0,
                    rgba: [0.70, 0.76, 0.88, 0.0],
                },
            ],
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("emitter id must not be empty".into());
        }
        if self.max_particles == 0 {
            return Err(format!("emitter '{}' maxParticles must be > 0", self.id));
        }
        if !self.spawn_rate.is_finite() || self.spawn_rate < 0.0 {
            return Err(format!(
                "emitter '{}' spawnRate must be finite and >= 0",
                self.id
            ));
        }
        if !self.lifetime_s[0].is_finite()
            || !self.lifetime_s[1].is_finite()
            || self.lifetime_s[0] <= 0.0
            || self.lifetime_s[1] < self.lifetime_s[0]
        {
            return Err(format!(
                "emitter '{}' lifetimeS must satisfy 0 < min <= max",
                self.id
            ));
        }
        self.shape.validate()?;
        if !self.speed[0].is_finite() || !self.speed[1].is_finite() || self.speed[1] < self.speed[0]
        {
            return Err(format!(
                "emitter '{}' speed must satisfy min <= max",
                self.id
            ));
        }
        if !self.size_base[0].is_finite()
            || !self.size_base[1].is_finite()
            || self.size_base[0] < 0.0
            || self.size_base[1] < self.size_base[0]
        {
            return Err(format!(
                "emitter '{}' sizeBase must satisfy 0 <= min <= max",
                self.id
            ));
        }
        for stop in &self.size_over_life {
            if !stop.t.is_finite() || !(0.0..=1.0).contains(&stop.t) || !stop.value.is_finite() {
                return Err(format!(
                    "emitter '{}' sizeOverLife points must lie in t=0..=1",
                    self.id
                ));
            }
        }
        for stop in &self.color_over_life {
            if !stop.t.is_finite()
                || !(0.0..=1.0).contains(&stop.t)
                || !stop.rgba.into_iter().all(f32::is_finite)
            {
                return Err(format!(
                    "emitter '{}' colorOverLife stops must lie in t=0..=1",
                    self.id
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VfxAsset {
    pub schema: String,
    pub id: String,
    pub name: String,
    pub seed: u64,
    pub duration_s: f32,
    pub looping: bool,
    pub warmup_s: f32,
    pub budget_max_particles: u32,
    pub emitters: Vec<EmitterSpec>,
}

impl Default for VfxAsset {
    fn default() -> Self {
        Self {
            schema: VFX_SCHEMA.into(),
            id: "untitled_vfx".into(),
            name: "Untitled Effect".into(),
            seed: 1,
            duration_s: 4.0,
            looping: true,
            warmup_s: 0.0,
            budget_max_particles: 512,
            emitters: vec![EmitterSpec::default()],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    pub emitter_index: usize,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub age_s: f32,
    pub lifetime_s: f32,
    pub size_base: f32,
    pub size: f32,
    pub rotation_deg: f32,
    pub angular_velocity_deg: f32,
    pub rgba: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VfxStats {
    pub active_particles: usize,
    pub additive_particles: usize,
    pub alpha_particles: usize,
    pub estimated_overdraw_score: f32,
    pub within_budget: bool,
}

#[derive(Clone, Debug)]
struct EmitterState {
    spawn_accum: f32,
    fired_bursts: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct VfxSimulator {
    asset: VfxAsset,
    rng: SplitMix64,
    time_s: f32,
    emitter_states: Vec<EmitterState>,
    particles: Vec<Particle>,
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-6 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sample_scalar_curve(points: &[ScalarPoint], t: f32) -> f32 {
    if points.is_empty() {
        return 1.0;
    }
    let u = t.clamp(0.0, 1.0);
    if u <= points[0].t {
        return points[0].value;
    }
    for pair in points.windows(2) {
        let a = pair[0];
        let b = pair[1];
        if u <= b.t {
            let span = (b.t - a.t).max(1e-6);
            let k = ((u - a.t) / span).clamp(0.0, 1.0);
            return a.value + (b.value - a.value) * k;
        }
    }
    points[points.len() - 1].value
}

fn sample_color_curve(stops: &[ColorStop], t: f32) -> [f32; 4] {
    if stops.is_empty() {
        return [1.0, 1.0, 1.0, 1.0];
    }
    let u = t.clamp(0.0, 1.0);
    if u <= stops[0].t {
        return stops[0].rgba;
    }
    for pair in stops.windows(2) {
        let a = pair[0];
        let b = pair[1];
        if u <= b.t {
            let span = (b.t - a.t).max(1e-6);
            let k = ((u - a.t) / span).clamp(0.0, 1.0);
            return [
                a.rgba[0] + (b.rgba[0] - a.rgba[0]) * k,
                a.rgba[1] + (b.rgba[1] - a.rgba[1]) * k,
                a.rgba[2] + (b.rgba[2] - a.rgba[2]) * k,
                a.rgba[3] + (b.rgba[3] - a.rgba[3]) * k,
            ];
        }
    }
    stops[stops.len() - 1].rgba
}

fn sample_spawn_pos_and_dir(spec: &EmitterSpec, rng: &mut SplitMix64) -> ([f32; 3], [f32; 3]) {
    let axis = normalize3(spec.direction);
    let helper = if axis[1].abs() < 0.92 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tangent = normalize3(cross3(helper, axis));
    let bitangent = cross3(axis, tangent);

    let (local_pos, dir) = match spec.shape {
        SpawnShape::Point => ([0.0, 0.0, 0.0], axis),
        SpawnShape::Sphere { radius } => {
            let theta = rng.range(0.0, std::f32::consts::TAU);
            let z = rng.range(-1.0, 1.0);
            let ring = (1.0 - z * z).max(0.0).sqrt();
            let unit = [ring * theta.cos(), z, ring * theta.sin()];
            let r = radius * rng.next_f32().cbrt();
            let pos = [unit[0] * r, unit[1] * r, unit[2] * r];
            let blended = normalize3([
                axis[0] * 0.65 + unit[0] * 0.35,
                axis[1] * 0.65 + unit[1] * 0.35,
                axis[2] * 0.65 + unit[2] * 0.35,
            ]);
            (pos, blended)
        }
        SpawnShape::Box { half_extents } => (
            [
                rng.range(-half_extents[0], half_extents[0]),
                rng.range(-half_extents[1], half_extents[1]),
                rng.range(-half_extents[2], half_extents[2]),
            ],
            axis,
        ),
        SpawnShape::Disc { radius } => {
            let theta = rng.range(0.0, std::f32::consts::TAU);
            let r = radius * rng.next_f32().sqrt();
            let u = r * theta.cos();
            let v = r * theta.sin();
            (
                [
                    tangent[0] * u + bitangent[0] * v,
                    tangent[1] * u + bitangent[1] * v,
                    tangent[2] * u + bitangent[2] * v,
                ],
                axis,
            )
        }
        SpawnShape::Cone { angle_deg, radius } => {
            let theta = rng.range(0.0, std::f32::consts::TAU);
            let r = radius * rng.next_f32().sqrt();
            let u = r * theta.cos();
            let v = r * theta.sin();
            let pos = [
                tangent[0] * u + bitangent[0] * v,
                tangent[1] * u + bitangent[1] * v,
                tangent[2] * u + bitangent[2] * v,
            ];
            let half_rad = (angle_deg * 0.5).to_radians();
            let cos_max = half_rad.cos();
            let cos_phi = cos_max + (1.0 - cos_max) * rng.next_f32();
            let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt();
            let az = rng.range(0.0, std::f32::consts::TAU);
            let cx = sin_phi * az.cos();
            let cy = sin_phi * az.sin();
            let cone_dir = normalize3([
                tangent[0] * cx + bitangent[0] * cy + axis[0] * cos_phi,
                tangent[1] * cx + bitangent[1] * cy + axis[1] * cos_phi,
                tangent[2] * cx + bitangent[2] * cy + axis[2] * cos_phi,
            ]);
            (pos, cone_dir)
        }
    };

    (
        [
            spec.offset[0] + local_pos[0],
            spec.offset[1] + local_pos[1],
            spec.offset[2] + local_pos[2],
        ],
        dir,
    )
}

impl VfxSimulator {
    pub fn new(asset: VfxAsset) -> Self {
        let mut sim = Self {
            rng: SplitMix64::new(asset.seed),
            time_s: 0.0,
            emitter_states: asset
                .emitters
                .iter()
                .map(|e| EmitterState {
                    spawn_accum: 0.0,
                    fired_bursts: vec![false; e.bursts.len()],
                })
                .collect(),
            particles: Vec::new(),
            asset,
        };
        sim.reset();
        sim
    }

    pub fn asset(&self) -> &VfxAsset {
        &self.asset
    }

    pub fn time_s(&self) -> f32 {
        self.time_s
    }

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    pub fn reset(&mut self) {
        self.rng = SplitMix64::new(self.asset.seed);
        self.time_s = 0.0;
        self.particles.clear();
        self.emitter_states = self
            .asset
            .emitters
            .iter()
            .map(|e| EmitterState {
                spawn_accum: 0.0,
                fired_bursts: vec![false; e.bursts.len()],
            })
            .collect();
        if self.asset.warmup_s > 0.0 {
            self.advance_fixed(self.asset.warmup_s);
            self.time_s = 0.0;
            for state in &mut self.emitter_states {
                state.fired_bursts.fill(false);
            }
        }
    }

    fn advance_fixed(&mut self, total_dt: f32) {
        if !total_dt.is_finite() || total_dt <= 0.0 {
            return;
        }
        let mut remaining = total_dt;
        while remaining > DEFAULT_FIXED_DT {
            self.step(DEFAULT_FIXED_DT);
            remaining -= DEFAULT_FIXED_DT;
        }
        if remaining > 1e-6 {
            self.step(remaining);
        }
    }

    /// Deterministically resets to the asset seed (plus warmup) and advances to `target_time_s`
    /// using fixed `1/60s` substeps so scrubbing backward or forward produces bit-identical poses.
    pub fn scrub_to(&mut self, target_time_s: f32) {
        self.reset();
        let clamped = if target_time_s.is_finite() && target_time_s > 0.0 {
            target_time_s
        } else {
            0.0
        };
        // Even at t=0.0, fire any t=0 bursts cleanly via a zero-dt burst pass.
        self.fire_due_bursts(0.0);
        if clamped > 0.0 {
            self.advance_fixed(clamped);
        }
    }

    fn count_for_emitter(&self, emitter_index: usize) -> usize {
        self.particles
            .iter()
            .filter(|p| p.emitter_index == emitter_index)
            .count()
    }

    fn spawn_one(&mut self, emitter_index: usize) {
        if self.particles.len() >= self.asset.budget_max_particles as usize {
            return;
        }
        let max_for_emitter = self.asset.emitters[emitter_index].max_particles as usize;
        if self.count_for_emitter(emitter_index) >= max_for_emitter {
            return;
        }
        let spec = &self.asset.emitters[emitter_index];
        let (position, dir) = sample_spawn_pos_and_dir(spec, &mut self.rng);
        let speed = self.rng.range(spec.speed[0], spec.speed[1]);
        let lifetime_s = self
            .rng
            .range(spec.lifetime_s[0], spec.lifetime_s[1])
            .max(1e-3);
        let size_base = self.rng.range(spec.size_base[0], spec.size_base[1]);
        let rotation_deg = self
            .rng
            .range(spec.initial_rotation_deg[0], spec.initial_rotation_deg[1]);
        let angular_velocity_deg = self
            .rng
            .range(spec.angular_velocity_deg[0], spec.angular_velocity_deg[1]);
        let size = size_base * sample_scalar_curve(&spec.size_over_life, 0.0);
        let rgba = sample_color_curve(&spec.color_over_life, 0.0);

        self.particles.push(Particle {
            emitter_index,
            position,
            velocity: [dir[0] * speed, dir[1] * speed, dir[2] * speed],
            age_s: 0.0,
            lifetime_s,
            size_base,
            size,
            rotation_deg,
            angular_velocity_deg,
            rgba,
        });
    }

    fn fire_due_bursts(&mut self, cycle_time_s: f32) {
        for emitter_index in 0..self.asset.emitters.len() {
            if !self.asset.emitters[emitter_index].enabled {
                continue;
            }
            let burst_count = self.asset.emitters[emitter_index].bursts.len();
            for b_idx in 0..burst_count {
                let burst = self.asset.emitters[emitter_index].bursts[b_idx];
                if !self.emitter_states[emitter_index].fired_bursts[b_idx]
                    && cycle_time_s >= burst.time_s
                {
                    self.emitter_states[emitter_index].fired_bursts[b_idx] = true;
                    for _ in 0..burst.count {
                        self.spawn_one(emitter_index);
                    }
                }
            }
        }
    }

    pub fn step(&mut self, dt_s: f32) {
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return;
        }
        let duration = self.asset.duration_s.max(1e-3);
        let prev_time = self.time_s;
        self.time_s += dt_s;

        let cycle_time = if self.asset.looping {
            let prev_cycle = (prev_time / duration).floor() as u64;
            let next_cycle = (self.time_s / duration).floor() as u64;
            if next_cycle > prev_cycle {
                for state in &mut self.emitter_states {
                    state.fired_bursts.fill(false);
                }
            }
            self.time_s % duration
        } else {
            self.time_s
        };

        let emitting_window = self.asset.looping || self.time_s <= duration;
        if emitting_window {
            self.fire_due_bursts(cycle_time);
            for emitter_index in 0..self.asset.emitters.len() {
                if !self.asset.emitters[emitter_index].enabled {
                    continue;
                }
                let rate = self.asset.emitters[emitter_index].spawn_rate;
                if rate > 0.0 {
                    self.emitter_states[emitter_index].spawn_accum += rate * dt_s;
                    let count = self.emitter_states[emitter_index].spawn_accum.floor() as u32;
                    self.emitter_states[emitter_index].spawn_accum -= count as f32;
                    for _ in 0..count {
                        self.spawn_one(emitter_index);
                    }
                }
            }
        }

        let sim_time = self.time_s;
        let emitters = &self.asset.emitters;
        self.particles.retain_mut(|p| {
            p.age_s += dt_s;
            if p.age_s >= p.lifetime_s {
                return false;
            }
            let spec = &emitters[p.emitter_index];
            let damp = (1.0 - spec.drag.max(0.0) * dt_s).clamp(0.0, 1.0);
            let mut accel = spec.gravity;
            if spec.curl_noise.amplitude.abs() > 1e-6 {
                let curl = curl_noise_3d(
                    p.position,
                    spec.curl_noise.frequency,
                    sim_time,
                    spec.curl_noise.scroll_speed,
                );
                accel[0] += curl[0] * spec.curl_noise.amplitude;
                accel[1] += curl[1] * spec.curl_noise.amplitude;
                accel[2] += curl[2] * spec.curl_noise.amplitude;
            }
            p.velocity[0] = (p.velocity[0] + accel[0] * dt_s) * damp;
            p.velocity[1] = (p.velocity[1] + accel[1] * dt_s) * damp;
            p.velocity[2] = (p.velocity[2] + accel[2] * dt_s) * damp;
            p.position[0] += p.velocity[0] * dt_s;
            p.position[1] += p.velocity[1] * dt_s;
            p.position[2] += p.velocity[2] * dt_s;
            p.rotation_deg += p.angular_velocity_deg * dt_s;

            let life_t = (p.age_s / p.lifetime_s).clamp(0.0, 1.0);
            p.size = p.size_base * sample_scalar_curve(&spec.size_over_life, life_t);
            p.rgba = sample_color_curve(&spec.color_over_life, life_t);
            true
        });
    }

    pub fn stats(&self) -> VfxStats {
        let mut additive_particles = 0usize;
        let mut alpha_particles = 0usize;
        let mut overdraw = 0.0f32;
        for p in &self.particles {
            let mode = self.asset.emitters[p.emitter_index].blend_mode;
            match mode {
                BlendMode::Additive => {
                    additive_particles += 1;
                    overdraw += p.size * p.size * 1.35;
                }
                BlendMode::Alpha => {
                    alpha_particles += 1;
                    overdraw += p.size * p.size;
                }
            }
        }
        VfxStats {
            active_particles: self.particles.len(),
            additive_particles,
            alpha_particles,
            estimated_overdraw_score: overdraw,
            within_budget: self.particles.len() <= self.asset.budget_max_particles as usize,
        }
    }
}

fn json_num(v: f32) -> JsonValue {
    JsonValue::Number(v as f64)
}

fn json_vec2(v: [f32; 2]) -> JsonValue {
    JsonValue::Array(vec![json_num(v[0]), json_num(v[1])])
}

fn json_vec3(v: [f32; 3]) -> JsonValue {
    JsonValue::Array(vec![json_num(v[0]), json_num(v[1]), json_num(v[2])])
}

fn json_vec4(v: [f32; 4]) -> JsonValue {
    JsonValue::Array(vec![
        json_num(v[0]),
        json_num(v[1]),
        json_num(v[2]),
        json_num(v[3]),
    ])
}

fn req_str<'a>(obj: &'a JsonValue, key: &str) -> Result<&'a str, String> {
    obj.get(key)
        .ok_or_else(|| format!("missing string '{key}'"))?
        .string()
}

fn req_f32(obj: &JsonValue, key: &str) -> Result<f32, String> {
    obj.get(key)
        .ok_or_else(|| format!("missing number '{key}'"))?
        .float()
}

fn opt_f32(obj: &JsonValue, key: &str, default: f32) -> Result<f32, String> {
    match obj.get(key) {
        Some(v) => v.float(),
        None => Ok(default),
    }
}

fn opt_bool(obj: &JsonValue, key: &str, default: bool) -> Result<bool, String> {
    match obj.get(key) {
        Some(JsonValue::Bool(b)) => Ok(*b),
        Some(_) => Err(format!("expected bool for '{key}'")),
        None => Ok(default),
    }
}

fn opt_u32(obj: &JsonValue, key: &str, default: u32) -> Result<u32, String> {
    match obj.get(key) {
        Some(v) => u32::try_from(v.unsigned()?).map_err(|_| format!("'{key}' exceeds u32")),
        None => Ok(default),
    }
}

fn opt_vec2(obj: &JsonValue, key: &str, default: [f32; 2]) -> Result<[f32; 2], String> {
    let Some(val) = obj.get(key) else {
        return Ok(default);
    };
    let arr = val.array()?;
    if arr.len() != 2 {
        return Err(format!("'{key}' must have length 2"));
    }
    Ok([arr[0].float()?, arr[1].float()?])
}

fn opt_vec3(obj: &JsonValue, key: &str, default: [f32; 3]) -> Result<[f32; 3], String> {
    let Some(val) = obj.get(key) else {
        return Ok(default);
    };
    let arr = val.array()?;
    if arr.len() != 3 {
        return Err(format!("'{key}' must have length 3"));
    }
    Ok([arr[0].float()?, arr[1].float()?, arr[2].float()?])
}

fn shape_to_json(shape: SpawnShape) -> JsonValue {
    let mut map = BTreeMap::new();
    match shape {
        SpawnShape::Point => {
            map.insert("kind".into(), JsonValue::String("point".into()));
        }
        SpawnShape::Sphere { radius } => {
            map.insert("kind".into(), JsonValue::String("sphere".into()));
            map.insert("radius".into(), json_num(radius));
        }
        SpawnShape::Box { half_extents } => {
            map.insert("kind".into(), JsonValue::String("box".into()));
            map.insert("halfExtents".into(), json_vec3(half_extents));
        }
        SpawnShape::Cone { angle_deg, radius } => {
            map.insert("kind".into(), JsonValue::String("cone".into()));
            map.insert("angleDeg".into(), json_num(angle_deg));
            map.insert("radius".into(), json_num(radius));
        }
        SpawnShape::Disc { radius } => {
            map.insert("kind".into(), JsonValue::String("disc".into()));
            map.insert("radius".into(), json_num(radius));
        }
    }
    JsonValue::Object(map)
}

fn shape_from_json(value: Option<&JsonValue>) -> Result<SpawnShape, String> {
    let Some(v) = value else {
        return Ok(SpawnShape::Point);
    };
    match req_str(v, "kind")? {
        "point" => Ok(SpawnShape::Point),
        "sphere" => Ok(SpawnShape::Sphere {
            radius: opt_f32(v, "radius", 0.25)?,
        }),
        "box" => Ok(SpawnShape::Box {
            half_extents: opt_vec3(v, "halfExtents", [0.5, 0.5, 0.5])?,
        }),
        "cone" => Ok(SpawnShape::Cone {
            angle_deg: opt_f32(v, "angleDeg", 35.0)?,
            radius: opt_f32(v, "radius", 0.05)?,
        }),
        "disc" => Ok(SpawnShape::Disc {
            radius: opt_f32(v, "radius", 0.25)?,
        }),
        other => Err(format!("unsupported spawn shape '{other}'")),
    }
}

impl VfxAsset {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != VFX_SCHEMA {
            return Err(format!(
                "unsupported VFX schema '{}' (expected '{VFX_SCHEMA}')",
                self.schema
            ));
        }
        if self.id.trim().is_empty() {
            return Err("vfx id must not be empty".into());
        }
        if !self.duration_s.is_finite() || self.duration_s <= 0.0 {
            return Err("durationS must be finite and > 0".into());
        }
        if !self.warmup_s.is_finite() || self.warmup_s < 0.0 {
            return Err("warmupS must be finite and >= 0".into());
        }
        if self.budget_max_particles == 0 {
            return Err("budgetMaxParticles must be > 0".into());
        }
        for emitter in &self.emitters {
            emitter.validate()?;
        }
        Ok(())
    }

    pub fn to_json_value(&self) -> JsonValue {
        let mut root = BTreeMap::new();
        root.insert("schema".into(), JsonValue::String(self.schema.clone()));
        root.insert("id".into(), JsonValue::String(self.id.clone()));
        root.insert("name".into(), JsonValue::String(self.name.clone()));
        root.insert("seed".into(), JsonValue::Number(self.seed as f64));
        root.insert("durationS".into(), json_num(self.duration_s));
        root.insert("looping".into(), JsonValue::Bool(self.looping));
        root.insert("warmupS".into(), json_num(self.warmup_s));
        root.insert(
            "budgetMaxParticles".into(),
            JsonValue::Number(self.budget_max_particles as f64),
        );
        root.insert(
            "emitters".into(),
            JsonValue::Array(
                self.emitters
                    .iter()
                    .map(|e| {
                        let mut em = BTreeMap::new();
                        em.insert("id".into(), JsonValue::String(e.id.clone()));
                        em.insert("enabled".into(), JsonValue::Bool(e.enabled));
                        em.insert(
                            "blendMode".into(),
                            JsonValue::String(e.blend_mode.as_str().into()),
                        );
                        em.insert(
                            "maxParticles".into(),
                            JsonValue::Number(e.max_particles as f64),
                        );
                        em.insert("spawnRate".into(), json_num(e.spawn_rate));
                        em.insert(
                            "bursts".into(),
                            JsonValue::Array(
                                e.bursts
                                    .iter()
                                    .map(|b| {
                                        let mut bm = BTreeMap::new();
                                        bm.insert("timeS".into(), json_num(b.time_s));
                                        bm.insert(
                                            "count".into(),
                                            JsonValue::Number(b.count as f64),
                                        );
                                        JsonValue::Object(bm)
                                    })
                                    .collect(),
                            ),
                        );
                        em.insert("lifetimeS".into(), json_vec2(e.lifetime_s));
                        em.insert("shape".into(), shape_to_json(e.shape));
                        em.insert("offset".into(), json_vec3(e.offset));
                        em.insert("direction".into(), json_vec3(e.direction));
                        em.insert("speed".into(), json_vec2(e.speed));
                        em.insert("gravity".into(), json_vec3(e.gravity));
                        em.insert("drag".into(), json_num(e.drag));
                        let mut cn = BTreeMap::new();
                        cn.insert("amplitude".into(), json_num(e.curl_noise.amplitude));
                        cn.insert("frequency".into(), json_num(e.curl_noise.frequency));
                        cn.insert("scrollSpeed".into(), json_num(e.curl_noise.scroll_speed));
                        em.insert("curlNoise".into(), JsonValue::Object(cn));
                        em.insert(
                            "initialRotationDeg".into(),
                            json_vec2(e.initial_rotation_deg),
                        );
                        em.insert(
                            "angularVelocityDeg".into(),
                            json_vec2(e.angular_velocity_deg),
                        );
                        em.insert("sizeBase".into(), json_vec2(e.size_base));
                        em.insert(
                            "sizeOverLife".into(),
                            JsonValue::Array(
                                e.size_over_life
                                    .iter()
                                    .map(|pt| {
                                        let mut pm = BTreeMap::new();
                                        pm.insert("t".into(), json_num(pt.t));
                                        pm.insert("value".into(), json_num(pt.value));
                                        JsonValue::Object(pm)
                                    })
                                    .collect(),
                            ),
                        );
                        em.insert(
                            "colorOverLife".into(),
                            JsonValue::Array(
                                e.color_over_life
                                    .iter()
                                    .map(|st| {
                                        let mut sm = BTreeMap::new();
                                        sm.insert("t".into(), json_num(st.t));
                                        sm.insert("rgba".into(), json_vec4(st.rgba));
                                        JsonValue::Object(sm)
                                    })
                                    .collect(),
                            ),
                        );
                        JsonValue::Object(em)
                    })
                    .collect(),
            ),
        );
        JsonValue::Object(root)
    }

    pub fn to_canonical_json(&self) -> Result<String, String> {
        self.validate()?;
        json::to_canonical(&self.to_json_value())
    }

    pub fn from_json_str(raw: &str) -> Result<Self, String> {
        let val = json::parse(raw.as_bytes()).map_err(|e| format!("invalid JSON: {e}"))?;
        Self::from_json_value(&val)
    }

    pub fn from_json_value(value: &JsonValue) -> Result<Self, String> {
        if !matches!(value, JsonValue::Object(_)) {
            return Err("vfx root must be a JSON object".into());
        }
        let schema = req_str(value, "schema")?.to_string();
        let id = req_str(value, "id")?.to_string();
        let name = match value.get("name") {
            Some(v) => v.string()?.to_string(),
            None => id.clone(),
        };
        let seed = match value.get("seed") {
            Some(v) => v.unsigned()?,
            None => 1,
        };
        let duration_s = req_f32(value, "durationS")?;
        let looping = opt_bool(value, "looping", true)?;
        let warmup_s = opt_f32(value, "warmupS", 0.0)?;
        let budget_max_particles = opt_u32(value, "budgetMaxParticles", 512)?;

        let mut emitters = Vec::new();
        if let Some(emit_val) = value.get("emitters") {
            for item in emit_val.array()? {
                if !matches!(item, JsonValue::Object(_)) {
                    return Err("emitter must be a JSON object".into());
                }
                let eid = req_str(item, "id")?.to_string();
                let enabled = opt_bool(item, "enabled", true)?;
                let blend_mode = match item.get("blendMode") {
                    Some(v) => BlendMode::parse(v.string()?)?,
                    None => BlendMode::Alpha,
                };
                let max_particles = opt_u32(item, "maxParticles", 128)?;
                let spawn_rate = opt_f32(item, "spawnRate", 0.0)?;
                let mut bursts = Vec::new();
                if let Some(bval) = item.get("bursts") {
                    for b in bval.array()? {
                        if !matches!(b, JsonValue::Object(_)) {
                            return Err("burst must be a JSON object".into());
                        }
                        bursts.push(Burst {
                            time_s: req_f32(b, "timeS")?,
                            count: opt_u32(b, "count", 0)?,
                        });
                    }
                }
                let lifetime_s = opt_vec2(item, "lifetimeS", [1.0, 2.0])?;
                let shape = shape_from_json(item.get("shape"))?;
                let offset = opt_vec3(item, "offset", [0.0, 0.0, 0.0])?;
                let direction = opt_vec3(item, "direction", [0.0, 1.0, 0.0])?;
                let speed = opt_vec2(item, "speed", [0.5, 1.0])?;
                let gravity = opt_vec3(item, "gravity", [0.0, 0.0, 0.0])?;
                let drag = opt_f32(item, "drag", 0.0)?;
                let curl_noise = if let Some(cn_val) = item.get("curlNoise") {
                    if !matches!(cn_val, JsonValue::Object(_)) {
                        return Err("curlNoise must be a JSON object".into());
                    }
                    CurlNoiseSpec {
                        amplitude: opt_f32(cn_val, "amplitude", 0.0)?,
                        frequency: opt_f32(cn_val, "frequency", 1.0)?,
                        scroll_speed: opt_f32(cn_val, "scrollSpeed", 0.4)?,
                    }
                } else {
                    CurlNoiseSpec::default()
                };
                let initial_rotation_deg = opt_vec2(item, "initialRotationDeg", [-180.0, 180.0])?;
                let angular_velocity_deg = opt_vec2(item, "angularVelocityDeg", [0.0, 0.0])?;
                let size_base = opt_vec2(item, "sizeBase", [0.03, 0.05])?;

                let mut size_over_life = Vec::new();
                if let Some(sval) = item.get("sizeOverLife") {
                    for pt in sval.array()? {
                        if !matches!(pt, JsonValue::Object(_)) {
                            return Err("sizeOverLife point must be a JSON object".into());
                        }
                        size_over_life.push(ScalarPoint {
                            t: req_f32(pt, "t")?,
                            value: req_f32(pt, "value")?,
                        });
                    }
                }
                let mut color_over_life = Vec::new();
                if let Some(cval) = item.get("colorOverLife") {
                    for st in cval.array()? {
                        if !matches!(st, JsonValue::Object(_)) {
                            return Err("colorOverLife stop must be a JSON object".into());
                        }
                        let arr = st
                            .get("rgba")
                            .ok_or("colorOverLife stop missing rgba")?
                            .array()?;
                        if arr.len() != 4 {
                            return Err("colorOverLife rgba must have length 4".into());
                        }
                        color_over_life.push(ColorStop {
                            t: req_f32(st, "t")?,
                            rgba: [
                                arr[0].float()?,
                                arr[1].float()?,
                                arr[2].float()?,
                                arr[3].float()?,
                            ],
                        });
                    }
                }

                emitters.push(EmitterSpec {
                    id: eid,
                    enabled,
                    blend_mode,
                    max_particles,
                    spawn_rate,
                    bursts,
                    lifetime_s,
                    shape,
                    offset,
                    direction,
                    speed,
                    gravity,
                    drag,
                    curl_noise,
                    initial_rotation_deg,
                    angular_velocity_deg,
                    size_base,
                    size_over_life,
                    color_over_life,
                });
            }
        }

        let asset = Self {
            schema,
            id,
            name,
            seed,
            duration_s,
            looping,
            warmup_s,
            budget_max_particles,
            emitters,
        };
        asset.validate()?;
        Ok(asset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_roundtrip_and_deterministic_scrubbing() {
        let asset = VfxAsset {
            schema: VFX_SCHEMA.into(),
            id: "festival_canopy".into(),
            name: "Festival Canopy".into(),
            seed: 42,
            duration_s: 6.0,
            looping: true,
            warmup_s: 0.5,
            budget_max_particles: 300,
            emitters: vec![
                EmitterSpec::preset_blossom_drift(),
                EmitterSpec::preset_ember_updraft(),
                EmitterSpec::preset_radial_burst(),
            ],
        };
        let raw = asset.to_canonical_json().expect("canonical json");
        let parsed = VfxAsset::from_json_str(&raw).expect("valid vfx asset");
        assert_eq!(parsed, asset);

        let mut sim_a = VfxSimulator::new(parsed.clone());
        let mut sim_b = VfxSimulator::new(parsed);
        sim_a.scrub_to(1.75);
        sim_b.scrub_to(4.10);
        sim_b.scrub_to(1.75);
        assert_eq!(sim_a.particles(), sim_b.particles());
        assert!(!sim_a.particles().is_empty());

        let stats = sim_a.stats();
        assert!(stats.within_budget);
        assert!(stats.active_particles <= 300);
        assert!(stats.additive_particles > 0);
        assert!(stats.alpha_particles > 0);
    }

    #[test]
    fn curl_noise_is_smooth_and_bounded() {
        let v0 = curl_noise_3d([0.25, 1.1, -0.4], 1.2, 0.5, 0.4);
        let v1 = curl_noise_3d([0.26, 1.1, -0.4], 1.2, 0.5, 0.4);
        for c in v0.into_iter().chain(v1) {
            assert!(c.is_finite());
            assert!(c.abs() < 10.0);
        }
        let delta =
            ((v0[0] - v1[0]).powi(2) + (v0[1] - v1[1]).powi(2) + (v0[2] - v1[2]).powi(2)).sqrt();
        assert!(delta < 0.5);
    }
}
