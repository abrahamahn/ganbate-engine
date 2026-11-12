//! Omni Stereo XR views, sanitized `Head + 2-Hand` upper-body spatial poses, and a deterministic
//! `120Hz` (`8.333ms`) rolling-window frame budget governor.
//!
//! Pure `std` math and `ganbate_core::json::Value` serialization with zero game-domain coupling.

use std::collections::{BTreeMap, VecDeque};

use crate::Vec3;
use crate::json::Value;
use crate::world::Transform;

/// Unit quaternion `[x, y, z, w]` in right-handed coordinates (`+Y` up, `-Z` forward).
pub type Quat = [f32; 4];

/// Column-major `4x4` homogeneous matrix.
pub type Mat4 = [f32; 16];

/// Default adult interpupillary distance (`63mm`) in metres.
pub const DEFAULT_IPD_METRES: f32 = 0.063;
/// Minimum realistic human IPD (`50mm`) in metres.
pub const MIN_IPD_METRES: f32 = 0.050;
/// Maximum realistic human IPD (`78mm`) in metres.
pub const MAX_IPD_METRES: f32 = 0.078;

/// Target frame budget (`8.333333ms`) for locked `120Hz` XR and high-refresh rendering.
pub const TARGET_120HZ_MS: f32 = 8.333_333;
/// Required sustained frame-time headroom (`< 6.8ms`) before stepping quality back up at `120Hz`.
pub const HEADROOM_120HZ_MS: f32 = 6.8;

/// Target frame budget (`11.111111ms`) for locked `90Hz` standalone/PC VR headsets.
pub const TARGET_90HZ_VR_MS: f32 = 11.111_111;
/// Required sustained frame-time headroom (`< 9.0ms`) before stepping quality back up at `90Hz` VR.
pub const HEADROOM_90HZ_VR_MS: f32 = 9.0;

/// Maximum finite coordinate magnitude accepted from network spatial poses (`10km`).
pub const MAX_POSE_COORD_METRES: f32 = 10_000.0;

/// Maximum quality tier index managed by [`FrameBudgetGovernor`].
pub const MAX_QUALITY_TIER: u8 = 3;

const DEFAULT_GOVERNOR_WINDOW: usize = 16;
const DOWNGRADE_STREAK_FRAMES: u32 = 4;
const DOWNGRADE_COOLDOWN_FRAMES: u32 = 6;
const UPGRADE_STREAK_FRAMES: u32 = 24;
const UPGRADE_COOLDOWN_FRAMES: u32 = 12;

/// A lens focused `distance` metres from the eye: nothing nearer blurs, and past it the scene
/// softens, reaching its full blur `depth` metres further on, `blur` (the blur's radius as a share
/// of the image height) across. A foreground subject in focus keeps the scenery behind it soft.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Focus {
    pub distance: f32,
    pub depth: f32,
    pub blur: f32,
}

impl Focus {
    /// Whether it blurs anything: a positive depth and blur at a finite distance.
    pub fn blurs(&self) -> bool {
        self.distance.is_finite() && self.depth > 0.0 && self.blur > 0.0 && self.blur.is_finite()
    }
}

/// A named camera. `fov` is the vertical field of view in degrees.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub name: String,
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub fov: f32,
    /// Which way is up in the image (pick another than +Y to look straight down).
    pub up: [f32; 3],
    /// Lens shift: where the view axis (the target) lands in the image, in normalized device
    /// units with x right and y up; `[0.0, 0.2]` puts it a tenth of the image height above the
    /// centre. The frustum is skewed, not rotated, so verticals stay vertical.
    pub shift: [f32; 2],
    /// Near clip distance in metres.
    pub near: f32,
    /// Depth of field: what lies past the focus softens (`None`: everything sharp).
    pub focus: Option<Focus>,
}

impl Default for View {
    fn default() -> Self {
        Self {
            name: String::new(),
            eye: [0.0, 0.0, 5.0],
            target: [0.0; 3],
            fov: 50.0,
            up: [0.0, 1.0, 0.0],
            shift: [0.0; 2],
            near: 0.08,
            focus: None,
        }
    }
}

/// Parallel-axis stereo camera pair derived from a mono [`View`].
#[derive(Clone, Debug, PartialEq)]
pub struct StereoViewPair {
    pub left: View,
    pub right: View,
    pub ipd_metres: f32,
    pub right_axis: [f32; 3],
}

/// Clamps `ipd_metres` to the realistic human range `0.050..=0.078` (`DEFAULT_IPD_METRES` on `NaN`).
#[inline]
pub fn clamp_ipd_metres(ipd_metres: f32) -> f32 {
    if ipd_metres.is_nan() {
        DEFAULT_IPD_METRES
    } else {
        ipd_metres.clamp(MIN_IPD_METRES, MAX_IPD_METRES)
    }
}

#[inline]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    if !v.iter().all(|x| x.is_finite()) {
        return None;
    }
    let len_sq = f64::from(v[0]) * f64::from(v[0])
        + f64::from(v[1]) * f64::from(v[1])
        + f64::from(v[2]) * f64::from(v[2]);
    if !len_sq.is_finite() || len_sq < 1e-12 {
        return None;
    }
    let inv = 1.0 / len_sq.sqrt();
    Some([
        (f64::from(v[0]) * inv) as f32,
        (f64::from(v[1]) * inv) as f32,
        (f64::from(v[2]) * inv) as f32,
    ])
}

/// Computes parallel-axis `left` and `right` stereo [`View`]s offset along the camera's right
/// vector by `±0.5 * ipd_metres` (`ipd_metres` clamped to `0.050..=0.078`, default `0.063`),
/// preserving camera orientation and projection/lens parameters without toe-in distortion.
pub fn compute_stereo_views(mono: &View, ipd_metres: f32) -> StereoViewPair {
    let ipd = clamp_ipd_metres(ipd_metres);
    let eye = if mono.eye.iter().all(|x| x.is_finite()) {
        mono.eye
    } else {
        [0.0, 0.0, 5.0]
    };
    let raw_target = if mono.target.iter().all(|x| x.is_finite()) {
        mono.target
    } else {
        [0.0, 0.0, 0.0]
    };

    let (forward, base_target) = match normalize3(sub3(raw_target, eye)) {
        Some(fwd) => (fwd, raw_target),
        None => ([0.0, 0.0, -1.0], [eye[0], eye[1], eye[2] - 1.0]),
    };

    let up_hint = normalize3(mono.up).unwrap_or([0.0, 1.0, 0.0]);
    let right_axis = normalize3(cross3(forward, up_hint)).unwrap_or_else(|| {
        let fallback_up = if forward[1] < -0.9 {
            [0.0, 0.0, -1.0]
        } else if forward[1] > 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        normalize3(cross3(forward, fallback_up)).unwrap_or([1.0, 0.0, 0.0])
    });

    let half = 0.5 * ipd;
    let offset = [
        right_axis[0] * half,
        right_axis[1] * half,
        right_axis[2] * half,
    ];

    let fov = if mono.fov.is_finite() {
        mono.fov.clamp(1.0, 175.0)
    } else {
        50.0
    };
    let near = if mono.near.is_finite() && mono.near > 1e-4 {
        mono.near
    } else {
        0.08
    };
    let shift = [
        if mono.shift[0].is_finite() {
            mono.shift[0]
        } else {
            0.0
        },
        if mono.shift[1].is_finite() {
            mono.shift[1]
        } else {
            0.0
        },
    ];

    let (left_name, right_name) = if mono.name.is_empty() {
        ("left".into(), "right".into())
    } else {
        (
            format!("{}:left", mono.name),
            format!("{}:right", mono.name),
        )
    };

    StereoViewPair {
        left: View {
            name: left_name,
            eye: sub3(eye, offset),
            target: sub3(base_target, offset),
            fov,
            up: up_hint,
            shift,
            near,
            focus: mono.focus,
        },
        right: View {
            name: right_name,
            eye: add3(eye, offset),
            target: add3(base_target, offset),
            fov,
            up: up_hint,
            shift,
            near,
            focus: mono.focus,
        },
        ipd_metres: ipd,
        right_axis,
    }
}

/// Sanitizes an arbitrary quaternion `[x, y, z, w]` into a finite unit quaternion, falling back
/// to identity `[0, 0, 0, 1]` when non-finite or degenerate.
#[inline]
pub fn sanitize_quat(q: Quat) -> Quat {
    if !q.iter().all(|v| v.is_finite()) {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let len_sq = q.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
    if !len_sq.is_finite() || len_sq < 1e-16 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let inv = 1.0 / len_sq.sqrt();
    let out = [
        (f64::from(q[0]) * inv) as f32,
        (f64::from(q[1]) * inv) as f32,
        (f64::from(q[2]) * inv) as f32,
        (f64::from(q[3]) * inv) as f32,
    ];
    if out.iter().all(|v| v.is_finite()) {
        out
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// Sanitizes a world/stage position vector so non-finite or extreme network values cannot poison
/// scene transforms.
#[inline]
pub fn sanitize_position(p: Vec3) -> Vec3 {
    let clean = |v: f32| {
        if v.is_finite() {
            v.clamp(-MAX_POSE_COORD_METRES, MAX_POSE_COORD_METRES)
        } else {
            0.0
        }
    };
    Vec3(clean(p.0), clean(p.1), clean(p.2))
}

/// Sanitizes an optional direction vector into a finite unit [`Vec3`], returning `None` if
/// non-finite or near-zero length.
#[inline]
pub fn sanitize_direction(d: Vec3) -> Option<Vec3> {
    normalize3([d.0, d.1, d.2]).map(|[x, y, z]| Vec3(x, y, z))
}

fn sanitize_f64_coord(n: f64) -> f32 {
    if n.is_finite() {
        n.clamp(
            f64::from(-MAX_POSE_COORD_METRES),
            f64::from(MAX_POSE_COORD_METRES),
        ) as f32
    } else {
        0.0
    }
}

fn sanitize_f64_quat(raw: [f64; 4]) -> Quat {
    if !raw.iter().all(|v| v.is_finite()) {
        return [0.0, 0.0, 0.0, 1.0];
    }
    // Scale by max abs component first so large f64 inputs (e.g. 1e200) never overflow len_sq.
    let max_abs = raw.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    if max_abs < 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let scaled = raw.map(|v| v / max_abs);
    let len = scaled.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if !len.is_finite() || len < 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    sanitize_quat(scaled.map(|v| (v / len) as f32))
}

fn parse_num_array<const N: usize>(v: &Value, field: &str) -> Result<[f64; N], String> {
    let arr = v
        .array()
        .map_err(|_| format!("{field}: expected an array of {N} numbers"))?;
    if arr.len() != N {
        return Err(format!("{field}: expected {N} numbers, got {}", arr.len()));
    }
    let mut out = [0.0f64; N];
    for (i, item) in arr.iter().enumerate() {
        let Value::Number(n) = item else {
            return Err(format!("{field}[{i}]: expected number"));
        };
        out[i] = *n;
    }
    Ok(out)
}

/// Genre-agnostic 6-DoF spatial pose (`position` in metres + unit `orientation` quaternion `[x, y, z, w]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialPose6D {
    pub position: Vec3,
    pub orientation: Quat,
}

impl Default for SpatialPose6D {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl SpatialPose6D {
    pub const IDENTITY: Self = Self {
        position: Vec3(0.0, 0.0, 0.0),
        orientation: [0.0, 0.0, 0.0, 1.0],
    };

    /// Constructs a pose with finite position and unit quaternion sanitization.
    #[inline]
    pub fn new(position: Vec3, orientation: Quat) -> Self {
        Self {
            position: sanitize_position(position),
            orientation: sanitize_quat(orientation),
        }
    }

    /// Returns a sanitized copy guaranteed to produce a valid affine [`Transform`].
    #[inline]
    pub fn sanitized(self) -> Self {
        Self::new(self.position, self.orientation)
    }

    /// Converts this pose into a validated engine [`Transform`].
    #[inline]
    pub fn to_transform(self) -> Transform {
        let clean = self.sanitized();
        Transform::from_trs(
            [clean.position.0, clean.position.1, clean.position.2],
            clean.orientation,
            [1.0, 1.0, 1.0],
        )
        .unwrap_or(Transform::IDENTITY)
    }

    /// Forward unit vector (`-Z` rotated by `self.orientation`).
    pub fn forward(self) -> Vec3 {
        let [x, y, z, w] = self.sanitized().orientation;
        // Column 2 of rotation matrix is local +Z; local -Z is its negation.
        let fx = -(2.0 * (x * z + y * w));
        let fy = -(2.0 * (y * z - x * w));
        let fz = -(1.0 - 2.0 * (x * x + y * y));
        sanitize_direction(Vec3(fx, fy, fz)).unwrap_or(Vec3(0.0, 0.0, -1.0))
    }

    /// Serializes into compact `ganbate_core::json::Value` (`{"p": [x, y, z], "q": [qx, qy, qz, qw]}`).
    pub fn to_json_value(&self) -> Value {
        let clean = self.sanitized();
        Value::Object(BTreeMap::from([
            (
                "p".into(),
                Value::Array(vec![
                    Value::Number(f64::from(clean.position.0)),
                    Value::Number(f64::from(clean.position.1)),
                    Value::Number(f64::from(clean.position.2)),
                ]),
            ),
            (
                "q".into(),
                Value::Array(
                    clean
                        .orientation
                        .iter()
                        .map(|&c| Value::Number(f64::from(c)))
                        .collect(),
                ),
            ),
        ]))
    }

    /// Decodes from `ganbate_core::json::Value`, sanitizing non-finite coordinates and
    /// unnormalized or zero-length quaternions.
    pub fn from_json_value(v: &Value) -> Result<Self, String> {
        if !matches!(v, Value::Object(_)) {
            return Err("SpatialPose6D expected a JSON object".into());
        }
        let pos_val = v
            .get("p")
            .or_else(|| v.get("position"))
            .ok_or_else(|| "SpatialPose6D missing 'p' / 'position'".to_string())?;
        let quat_val = v
            .get("q")
            .or_else(|| v.get("orientation"))
            .or_else(|| v.get("rotation"))
            .ok_or_else(|| "SpatialPose6D missing 'q' / 'orientation'".to_string())?;

        let raw_p = parse_num_array::<3>(pos_val, "p")?;
        let raw_q = parse_num_array::<4>(quat_val, "q")?;

        Ok(Self {
            position: Vec3(
                sanitize_f64_coord(raw_p[0]),
                sanitize_f64_coord(raw_p[1]),
                sanitize_f64_coord(raw_p[2]),
            ),
            orientation: sanitize_f64_quat(raw_q),
        })
    }
}

/// Genre-agnostic `Head + 2-Hand` upper-body spatial IK pose contract with optional eye-gaze ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpperBodyIkPose {
    pub head: SpatialPose6D,
    pub left_hand: Option<SpatialPose6D>,
    pub right_hand: Option<SpatialPose6D>,
    pub gaze_dir: Option<Vec3>,
}

impl Default for UpperBodyIkPose {
    fn default() -> Self {
        Self {
            head: SpatialPose6D::new(Vec3(0.0, 1.6, 0.0), [0.0, 0.0, 0.0, 1.0]),
            left_hand: None,
            right_hand: None,
            gaze_dir: None,
        }
    }
}

impl UpperBodyIkPose {
    /// Returns a sanitized pose where all positions are finite/bounded, all quaternions are unit
    /// length, and any degenerate `gaze_dir` is dropped cleanly rather than poisoning transforms.
    pub fn sanitized(self) -> Self {
        Self {
            head: self.head.sanitized(),
            left_hand: self.left_hand.map(SpatialPose6D::sanitized),
            right_hand: self.right_hand.map(SpatialPose6D::sanitized),
            gaze_dir: self.gaze_dir.and_then(sanitize_direction),
        }
    }

    /// Serializes into compact `ganbate_core::json::Value`.
    pub fn to_json_value(&self) -> Value {
        let clean = self.sanitized();
        let mut map = BTreeMap::from([("head".into(), clean.head.to_json_value())]);
        if let Some(lh) = clean.left_hand {
            map.insert("left_hand".into(), lh.to_json_value());
        }
        if let Some(rh) = clean.right_hand {
            map.insert("right_hand".into(), rh.to_json_value());
        }
        if let Some(gaze) = clean.gaze_dir {
            map.insert(
                "gaze_dir".into(),
                Value::Array(vec![
                    Value::Number(f64::from(gaze.0)),
                    Value::Number(f64::from(gaze.1)),
                    Value::Number(f64::from(gaze.2)),
                ]),
            );
        }
        Value::Object(map)
    }

    /// Decodes from `ganbate_core::json::Value`, sanitizing all poses and gaze vectors.
    pub fn from_json_value(v: &Value) -> Result<Self, String> {
        if !matches!(v, Value::Object(_)) {
            return Err("UpperBodyIkPose expected a JSON object".into());
        }
        let head_val = v
            .get("head")
            .or_else(|| v.get("h"))
            .ok_or_else(|| "UpperBodyIkPose missing 'head'".to_string())?;
        let head = SpatialPose6D::from_json_value(head_val)?;

        let left_hand = match v
            .get("left_hand")
            .or_else(|| v.get("leftHand"))
            .or_else(|| v.get("lh"))
        {
            None | Some(Value::Null) => None,
            Some(lh) => Some(SpatialPose6D::from_json_value(lh)?),
        };

        let right_hand = match v
            .get("right_hand")
            .or_else(|| v.get("rightHand"))
            .or_else(|| v.get("rh"))
        {
            None | Some(Value::Null) => None,
            Some(rh) => Some(SpatialPose6D::from_json_value(rh)?),
        };

        let gaze_dir = match v
            .get("gaze_dir")
            .or_else(|| v.get("gazeDir"))
            .or_else(|| v.get("gaze"))
        {
            None | Some(Value::Null) => None,
            Some(gv) => {
                let raw = parse_num_array::<3>(gv, "gaze_dir")?;
                if !raw.iter().all(|x| x.is_finite()) {
                    None
                } else {
                    let max_abs = raw.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
                    if max_abs < 1e-12 {
                        None
                    } else {
                        let s = raw.map(|x| (x / max_abs) as f32);
                        sanitize_direction(Vec3(s[0], s[1], s[2]))
                    }
                }
            }
        };

        Ok(Self {
            head,
            left_hand,
            right_hand,
            gaze_dir,
        })
    }
}

/// Dynamic quality budget emitted by [`FrameBudgetGovernor`] to hold a locked `120Hz` (`8.333ms`)
/// frame cadence across flat, mobile, and stereo XR displays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityBudget {
    pub resolution_scale: f32,
    pub max_dynamic_lights: u32,
    pub ambient_particles: bool,
    pub shadow_tier: u8,
}

impl Default for QualityBudget {
    fn default() -> Self {
        Self::for_tier(MAX_QUALITY_TIER)
    }
}

impl QualityBudget {
    /// Returns the deterministic quality preset for `tier` (`0` = maximum GPU relief, `3` = full `120Hz` fidelity).
    pub const fn for_tier(tier: u8) -> Self {
        match tier {
            0 => Self {
                resolution_scale: 0.67,
                max_dynamic_lights: 2,
                ambient_particles: false,
                shadow_tier: 0,
            },
            1 => Self {
                resolution_scale: 0.80,
                max_dynamic_lights: 4,
                ambient_particles: false,
                shadow_tier: 1,
            },
            2 => Self {
                resolution_scale: 0.90,
                max_dynamic_lights: 6,
                ambient_particles: true,
                shadow_tier: 2,
            },
            _ => Self {
                resolution_scale: 1.0,
                max_dynamic_lights: 8,
                ambient_particles: true,
                shadow_tier: 3,
            },
        }
    }

    pub fn to_json_value(&self) -> Value {
        Value::Object(BTreeMap::from([
            (
                "resolution_scale".into(),
                Value::Number(f64::from(self.resolution_scale)),
            ),
            (
                "max_dynamic_lights".into(),
                Value::Number(f64::from(self.max_dynamic_lights)),
            ),
            (
                "ambient_particles".into(),
                Value::Bool(self.ambient_particles),
            ),
            (
                "shadow_tier".into(),
                Value::Number(f64::from(self.shadow_tier)),
            ),
        ]))
    }

    pub fn from_json_value(v: &Value) -> Result<Self, String> {
        let scale = v
            .get("resolution_scale")
            .ok_or("QualityBudget missing resolution_scale")?
            .float()?
            .clamp(0.5, 1.0);
        let lights = v
            .get("max_dynamic_lights")
            .ok_or("QualityBudget missing max_dynamic_lights")?
            .unsigned()?
            .min(8) as u32;
        let particles = match v.get("ambient_particles") {
            Some(Value::Bool(b)) => *b,
            _ => return Err("QualityBudget missing boolean ambient_particles".into()),
        };
        let shadow_tier = v
            .get("shadow_tier")
            .ok_or("QualityBudget missing shadow_tier")?
            .unsigned()?
            .min(u64::from(MAX_QUALITY_TIER)) as u8;
        Ok(Self {
            resolution_scale: scale,
            max_dynamic_lights: lights,
            ambient_particles: particles,
            shadow_tier,
        })
    }
}

/// Deterministic rolling-window frame-time governor (`120Hz` `8.333ms` default, or `90Hz` VR
/// `11.111ms`) with hysteresis.
///
/// Steps down one tier at a time on sustained over-budget frames to prevent XR motion sickness,
/// holds steady inside the `[headroom_ms, target_ms]` hysteresis deadband, and steps back up only
/// after sustained headroom (`< headroom_ms`).
#[derive(Clone, Debug)]
pub struct FrameBudgetGovernor {
    window_capacity: usize,
    target_ms: f32,
    headroom_ms: f32,
    samples_ms: VecDeque<f32>,
    tier: u8,
    over_budget_streak: u32,
    headroom_streak: u32,
    cooldown_frames: u32,
}

impl Default for FrameBudgetGovernor {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameBudgetGovernor {
    pub fn new() -> Self {
        Self::with_window(DEFAULT_GOVERNOR_WINDOW)
    }

    /// Creates a `90Hz` VR (`11.111ms` budget, `9.0ms` recovery headroom) closed-loop governor.
    pub fn new_90hz_vr() -> Self {
        Self::with_target_ms(
            DEFAULT_GOVERNOR_WINDOW,
            TARGET_90HZ_VR_MS,
            HEADROOM_90HZ_VR_MS,
        )
    }

    pub fn with_window(window_capacity: usize) -> Self {
        Self::with_target_ms(window_capacity, TARGET_120HZ_MS, HEADROOM_120HZ_MS)
    }

    /// Creates a closed-loop governor with custom rolling window, frame-time target, and recovery headroom.
    pub fn with_target_ms(window_capacity: usize, target_ms: f32, headroom_ms: f32) -> Self {
        let cap = window_capacity.clamp(4, 240);
        let safe_target = if target_ms.is_finite() && target_ms >= 1.0 {
            target_ms
        } else {
            TARGET_120HZ_MS
        };
        let safe_headroom = if headroom_ms.is_finite() && headroom_ms > 0.0 {
            headroom_ms.min(safe_target * 0.95)
        } else {
            safe_target * 0.82
        };
        Self {
            window_capacity: cap,
            target_ms: safe_target,
            headroom_ms: safe_headroom,
            samples_ms: VecDeque::with_capacity(cap),
            tier: MAX_QUALITY_TIER,
            over_budget_streak: 0,
            headroom_streak: 0,
            cooldown_frames: 0,
        }
    }

    /// Target frame budget in milliseconds (e.g. `8.333ms` at `120Hz` or `11.111ms` at `90Hz` VR).
    #[inline]
    pub fn target_ms(&self) -> f32 {
        self.target_ms
    }

    /// Recovery headroom threshold in milliseconds.
    #[inline]
    pub fn headroom_ms(&self) -> f32 {
        self.headroom_ms
    }

    /// Current quality tier (`0..=3`).
    #[inline]
    pub fn tier(&self) -> u8 {
        self.tier
    }

    /// Current [`QualityBudget`].
    #[inline]
    pub fn budget(&self) -> QualityBudget {
        QualityBudget::for_tier(self.tier)
    }

    /// Number of valid frame samples currently in the rolling window.
    #[inline]
    pub fn sample_count(&self) -> usize {
        self.samples_ms.len()
    }

    /// Rolling-window mean frame duration in milliseconds (`0.0` when empty).
    pub fn rolling_mean_ms(&self) -> f32 {
        if self.samples_ms.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.samples_ms.iter().map(|&x| f64::from(x)).sum();
        (sum / (self.samples_ms.len() as f64)) as f32
    }

    /// Resets rolling samples and restores the top quality tier.
    pub fn reset(&mut self) {
        self.samples_ms.clear();
        self.tier = MAX_QUALITY_TIER;
        self.over_budget_streak = 0;
        self.headroom_streak = 0;
        self.cooldown_frames = 0;
    }

    /// Records a completed frame duration in milliseconds and returns the governed [`QualityBudget`].
    pub fn record_frame_ms(&mut self, dt_ms: f32) -> QualityBudget {
        if dt_ms.is_nan() || dt_ms < 0.0 {
            return self.budget();
        }
        let clamped_dt = if dt_ms.is_finite() {
            dt_ms.min(250.0)
        } else {
            250.0
        };

        if self.samples_ms.len() == self.window_capacity {
            self.samples_ms.pop_front();
        }
        self.samples_ms.push_back(clamped_dt);

        if self.cooldown_frames > 0 {
            self.cooldown_frames -= 1;
        }

        if clamped_dt > self.target_ms {
            self.over_budget_streak = self.over_budget_streak.saturating_add(1);
            self.headroom_streak = 0;
        } else if clamped_dt < self.headroom_ms {
            self.headroom_streak = self.headroom_streak.saturating_add(1);
            self.over_budget_streak = 0;
        } else {
            // Hysteresis deadband `[headroom_ms, target_ms]`: neither upgrade nor downgrade builds streak.
            self.over_budget_streak = 0;
            self.headroom_streak = 0;
        }

        let mean = self.rolling_mean_ms();
        let half_window = (self.window_capacity / 2).max(4);

        if self.cooldown_frames == 0 && self.tier > 0 {
            let sustained_spike = self.over_budget_streak >= DOWNGRADE_STREAK_FRAMES;
            let sustained_mean_over = self.samples_ms.len() >= half_window
                && mean > self.target_ms
                && clamped_dt > self.target_ms;
            if sustained_spike || sustained_mean_over {
                self.tier -= 1;
                self.over_budget_streak = 0;
                self.headroom_streak = 0;
                self.cooldown_frames = DOWNGRADE_COOLDOWN_FRAMES;
                return self.budget();
            }
        }

        if self.cooldown_frames == 0
            && self.tier < MAX_QUALITY_TIER
            && self.headroom_streak >= UPGRADE_STREAK_FRAMES
            && self.samples_ms.len() >= half_window
            && mean < self.headroom_ms
        {
            self.tier += 1;
            self.headroom_streak = 0;
            self.over_budget_streak = 0;
            self.cooldown_frames = UPGRADE_COOLDOWN_FRAMES;
        }

        self.budget()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    #[test]
    fn stereo_views_offset_along_camera_right_preserve_orientation_and_clamp_ipd() {
        let mono = View {
            name: "main_cam".into(),
            eye: [0.0, 1.65, 3.0],
            target: [0.0, 1.65, 0.0],
            fov: 62.0,
            up: [0.0, 1.0, 0.0],
            shift: [0.05, -0.02],
            near: 0.05,
            focus: Some(Focus {
                distance: 3.0,
                depth: 2.0,
                blur: 0.01,
            }),
        };

        // Default 63mm IPD looking along -Z has +X right vector.
        let pair = compute_stereo_views(&mono, DEFAULT_IPD_METRES);
        assert!((pair.ipd_metres - 0.063).abs() < 1e-6);
        assert!((pair.right_axis[0] - 1.0).abs() < 1e-6);
        assert!((pair.left.eye[0] - (-0.0315)).abs() < 1e-6);
        assert!((pair.right.eye[0] - 0.0315).abs() < 1e-6);
        assert_eq!(pair.left.name, "main_cam:left");
        assert_eq!(pair.right.name, "main_cam:right");
        assert_eq!(pair.left.fov, mono.fov);
        assert_eq!(pair.right.fov, mono.fov);
        assert_eq!(pair.left.shift, mono.shift);
        assert_eq!(pair.right.near, mono.near);
        assert_eq!(pair.left.focus, mono.focus);

        // Parallel forward vectors (zero toe-in distortion).
        let fwd_mono = normalize3(sub3(mono.target, mono.eye)).unwrap();
        let fwd_left = normalize3(sub3(pair.left.target, pair.left.eye)).unwrap();
        let fwd_right = normalize3(sub3(pair.right.target, pair.right.eye)).unwrap();
        for i in 0..3 {
            assert!((fwd_left[i] - fwd_mono[i]).abs() < 1e-6);
            assert!((fwd_right[i] - fwd_mono[i]).abs() < 1e-6);
        }

        // Arbitrary yawed/pitched view still separates by exact clamped IPD perpendicular to view ray.
        let angled = View {
            name: String::new(),
            eye: [2.0, 2.2, 2.0],
            target: [-1.0, 1.0, -2.0],
            ..mono.clone()
        };
        let low_ipd = compute_stereo_views(&angled, 0.010);
        assert!((low_ipd.ipd_metres - MIN_IPD_METRES).abs() < 1e-6);
        let sep = sub3(low_ipd.right.eye, low_ipd.left.eye);
        let dist = (sep[0] * sep[0] + sep[1] * sep[1] + sep[2] * sep[2]).sqrt();
        assert!((dist - MIN_IPD_METRES).abs() < 1e-5);
        let fwd_ang = normalize3(sub3(angled.target, angled.eye)).unwrap();
        let dot_fwd = sep[0] * fwd_ang[0] + sep[1] * fwd_ang[1] + sep[2] * fwd_ang[2];
        assert!(dot_fwd.abs() < 1e-5);

        let high_ipd = compute_stereo_views(&angled, 0.250);
        assert!((high_ipd.ipd_metres - MAX_IPD_METRES).abs() < 1e-6);
        let nan_ipd = compute_stereo_views(&angled, f32::NAN);
        assert!((nan_ipd.ipd_metres - DEFAULT_IPD_METRES).abs() < 1e-6);

        // Straight-down degenerate up vector still resolves cleanly without NaNs.
        let top_down = View {
            eye: [0.0, 5.0, 0.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            ..View::default()
        };
        let td_pair = compute_stereo_views(&top_down, DEFAULT_IPD_METRES);
        assert!(td_pair.left.eye.iter().all(|v| v.is_finite()));
        assert!(td_pair.right.eye.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn upper_body_ik_pose_roundtrips_json_and_sanitizes_malformed_quaternions() {
        let valid_pose = UpperBodyIkPose {
            head: SpatialPose6D::new(Vec3(0.0, 1.68, -0.1), [0.0, 0.382_683_43, 0.0, 0.923_879_5]),
            left_hand: Some(SpatialPose6D::new(
                Vec3(-0.28, 1.12, -0.42),
                [0.0, 0.0, 0.0, 1.0],
            )),
            right_hand: Some(SpatialPose6D::new(
                Vec3(0.28, 1.14, -0.38),
                [0.0, 0.0, 0.0, 5.0], // non-unit -> normalized to [0,0,0,1]
            )),
            gaze_dir: Some(Vec3(0.0, 0.0, -4.0)), // non-unit -> normalized to [0,0,-1]
        };

        let encoded = valid_pose.to_json_value();
        let compact = json::to_compact(&encoded).expect("compact json");
        let reparsed = json::parse(compact.as_bytes()).expect("parse compact json");
        let decoded = UpperBodyIkPose::from_json_value(&reparsed).expect("decode UpperBodyIkPose");

        assert_eq!(
            decoded.right_hand.unwrap().orientation,
            [0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(decoded.gaze_dir, Some(Vec3(0.0, 0.0, -1.0)));
        assert!((decoded.head.position.1 - 1.68).abs() < 1e-5);
        assert!(decoded.head.to_transform().valid());
        assert!(decoded.left_hand.unwrap().to_transform().valid());
        assert!(decoded.right_hand.unwrap().to_transform().valid());

        // Malformed in-memory network pose: NaN, Infinity, zero quaternion, zero gaze.
        let poisoned = UpperBodyIkPose {
            head: SpatialPose6D {
                position: Vec3(f32::NAN, f32::INFINITY, -999_999.0),
                orientation: [0.0, 0.0, 0.0, 0.0],
            },
            left_hand: Some(SpatialPose6D {
                position: Vec3(1.0, 2.0, 3.0),
                orientation: [f32::NAN, 1.0, 0.0, 1.0],
            }),
            right_hand: Some(SpatialPose6D {
                position: Vec3(0.2, 1.0, -0.3),
                orientation: [3.0, 4.0, 0.0, 0.0],
            }),
            gaze_dir: Some(Vec3(0.0, 0.0, 0.0)),
        };
        let clean = poisoned.sanitized();
        assert_eq!(clean.head.position, Vec3(0.0, 0.0, -MAX_POSE_COORD_METRES));
        assert_eq!(clean.head.orientation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(clean.left_hand.unwrap().orientation, [0.0, 0.0, 0.0, 1.0]);
        let rh_q = clean.right_hand.unwrap().orientation;
        assert!((rh_q[0] - 0.6).abs() < 1e-6 && (rh_q[1] - 0.8).abs() < 1e-6);
        assert_eq!(clean.gaze_dir, None);
        assert!(clean.head.to_transform().valid());
        assert!(clean.left_hand.unwrap().to_transform().valid());
        assert!(clean.right_hand.unwrap().to_transform().valid());

        // Malformed JSON Value with extreme f64 / zero quaternion / NaN never poisons scene transforms.
        let bad_json = Value::Object(BTreeMap::from([
            (
                "head".into(),
                Value::Object(BTreeMap::from([
                    (
                        "p".into(),
                        Value::Array(vec![
                            Value::Number(f64::NAN),
                            Value::Number(1e200),
                            Value::Number(-1.5),
                        ]),
                    ),
                    (
                        "q".into(),
                        Value::Array(vec![
                            Value::Number(0.0),
                            Value::Number(0.0),
                            Value::Number(0.0),
                            Value::Number(0.0),
                        ]),
                    ),
                ])),
            ),
            (
                "gaze_dir".into(),
                Value::Array(vec![
                    Value::Number(0.0),
                    Value::Number(0.0),
                    Value::Number(0.0),
                ]),
            ),
        ]));
        let from_bad = UpperBodyIkPose::from_json_value(&bad_json).unwrap();
        assert_eq!(
            from_bad.head.position,
            Vec3(0.0, MAX_POSE_COORD_METRES, -1.5)
        );
        assert_eq!(from_bad.head.orientation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(from_bad.gaze_dir, None);
        assert!(from_bad.head.to_transform().valid());
    }

    #[test]
    fn frame_budget_governor_enforces_120hz_with_hysteresis() {
        let mut gov = FrameBudgetGovernor::new();
        assert_eq!(gov.tier(), 3);
        assert_eq!(gov.budget().max_dynamic_lights, 8);
        assert!(gov.budget().ambient_particles);

        // Single transient hitch (> 8.333ms once) does NOT thrash quality.
        for _ in 0..8 {
            gov.record_frame_ms(6.0);
        }
        let after_single_spike = gov.record_frame_ms(11.5);
        assert_eq!(gov.tier(), 3);
        assert_eq!(after_single_spike.resolution_scale, 1.0);

        // Sustained > 8.333ms frames step down smoothly one tier at a time.
        let mut observed_tiers = vec![gov.tier()];
        for _ in 0..30 {
            gov.record_frame_ms(9.6);
            if observed_tiers.last().copied() != Some(gov.tier()) {
                observed_tiers.push(gov.tier());
            }
        }
        assert_eq!(observed_tiers, vec![3, 2, 1, 0]);
        let min_budget = gov.budget();
        assert!((min_budget.resolution_scale - 0.67).abs() < 1e-6);
        assert_eq!(min_budget.max_dynamic_lights, 2);
        assert!(!min_budget.ambient_particles);
        assert_eq!(min_budget.shadow_tier, 0);

        // Frames inside hysteresis deadband [6.8ms, 8.333ms] hold steady and do NOT upgrade prematurely.
        for _ in 0..60 {
            gov.record_frame_ms(7.4);
        }
        assert_eq!(gov.tier(), 0);

        // Sustained headroom (< 6.8ms) steps quality back up smoothly tier by tier.
        for _ in 0..24 {
            gov.record_frame_ms(5.6);
        }
        assert_eq!(gov.tier(), 1);
        for _ in 0..24 {
            gov.record_frame_ms(5.6);
        }
        assert_eq!(gov.tier(), 2);
        for _ in 0..24 {
            gov.record_frame_ms(5.6);
        }
        assert_eq!(gov.tier(), 3);
        assert_eq!(gov.budget(), QualityBudget::default());

        // Budget JSON roundtrip works cleanly.
        let b_json = min_budget.to_json_value();
        assert_eq!(QualityBudget::from_json_value(&b_json).unwrap(), min_budget);
    }

    #[test]
    fn pose_forward_and_governor_edge_inputs_stay_deterministic() {
        let identity_fwd = SpatialPose6D::IDENTITY.forward();
        assert!((identity_fwd.0).abs() < 1e-6);
        assert!((identity_fwd.1).abs() < 1e-6);
        assert!((identity_fwd.2 - (-1.0)).abs() < 1e-6);

        // +90 deg yaw around +Y rotates -Z forward to -X.
        let half_90 = 45.0_f32.to_radians();
        let yaw_left = SpatialPose6D::new(
            Vec3(0.0, 1.6, 0.0),
            [0.0, half_90.sin(), 0.0, half_90.cos()],
        );
        let left_fwd = yaw_left.forward();
        assert!((left_fwd.0 - (-1.0)).abs() < 1e-5);
        assert!(left_fwd.1.abs() < 1e-5);
        assert!(left_fwd.2.abs() < 1e-5);

        assert!(UpperBodyIkPose::from_json_value(&Value::String("bad".into())).is_err());

        let mut gov = FrameBudgetGovernor::with_window(8);
        assert_eq!(gov.sample_count(), 0);
        assert_eq!(gov.rolling_mean_ms(), 0.0);
        // Negative and NaN durations are ignored safely.
        gov.record_frame_ms(-5.0);
        gov.record_frame_ms(f32::NAN);
        assert_eq!(gov.sample_count(), 0);
        // Infinite stall clamps cleanly without poisoning mean with Inf/NaN.
        for _ in 0..4 {
            gov.record_frame_ms(f32::INFINITY);
        }
        assert!(gov.rolling_mean_ms().is_finite());
        assert_eq!(gov.tier(), 2);
        gov.reset();
        assert_eq!(gov.tier(), MAX_QUALITY_TIER);
        assert_eq!(gov.sample_count(), 0);
    }

    #[test]
    fn vr_90hz_governor_steps_down_holds_deadband_and_recovers() {
        let mut gov = FrameBudgetGovernor::new_90hz_vr();
        assert!((gov.target_ms() - TARGET_90HZ_VR_MS).abs() < 1e-6);
        assert!((gov.headroom_ms() - HEADROOM_90HZ_VR_MS).abs() < 1e-6);
        for _ in 0..32 {
            gov.record_frame_ms(13.8);
        }
        assert_eq!(gov.tier(), 0);
        for _ in 0..16 {
            gov.record_frame_ms(10.0);
            assert_eq!(gov.tier(), 0);
        }
        for _ in 0..96 {
            gov.record_frame_ms(7.8);
        }
        assert_eq!(gov.tier(), MAX_QUALITY_TIER);
    }
}
