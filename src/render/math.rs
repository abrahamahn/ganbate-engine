//! Small column-major matrix and vector helpers for the renderer (element (row, col) is at
//! `m[col * 4 + row]`). World space is right-handed and Y-up, in metres.

pub type V3 = [f32; 3];
pub type M4 = [f32; 16];

pub const IDENTITY: M4 = [
    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
];

#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
pub fn length(a: V3) -> f32 {
    dot(a, a).sqrt()
}
#[inline]
pub fn normalize(a: V3) -> V3 {
    let l = length(a);
    if l > 1e-12 {
        scale(a, 1.0 / l)
    } else {
        [0.0, 1.0, 0.0]
    }
}
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
#[inline]
pub fn lerp3(a: V3, b: V3, t: f32) -> V3 {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}
#[inline]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `a * b`.
#[inline]
pub fn mul(a: &M4, b: &M4) -> M4 {
    let mut out = [0.0; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    out
}
/// `m * (x, y, z, w)`.
#[inline]
pub fn transform(m: &M4, v: [f32; 4]) -> [f32; 4] {
    std::array::from_fn(|r| (0..4).map(|k| m[k * 4 + r] * v[k]).sum())
}
#[inline]
pub fn transform_point(m: &M4, p: V3) -> V3 {
    std::array::from_fn(|r| (((0.0 + m[r] * p[0]) + m[4 + r] * p[1]) + m[8 + r] * p[2]) + m[12 + r])
}

/// Right-handed view matrix: the camera looks down its -Z axis.
pub fn look_at(eye: V3, target: V3, up: V3) -> M4 {
    let f = normalize(sub(target, eye));
    let mut s = cross(f, up);
    if length(s) < 1e-5 {
        s = cross(f, [0.0, 0.0, 1.0]);
    }
    let s = normalize(s);
    let u = cross(s, f);
    [
        s[0],
        u[0],
        -f[0],
        0.0,
        s[1],
        u[1],
        -f[1],
        0.0,
        s[2],
        u[2],
        -f[2],
        0.0,
        -dot(s, eye),
        -dot(u, eye),
        dot(f, eye),
        1.0,
    ]
}

/// Vulkan clip space (y down) with reversed depth and an infinite far plane: depth is 1 at
/// `near` and approaches 0 at infinity. Use a `Greater` depth test and clear depth to 0.
pub fn perspective(fov_y_radians: f32, aspect: f32, near: f32) -> M4 {
    let t = 1.0 / (0.5 * fov_y_radians).tan();
    let mut m = [0.0; 16];
    m[0] = t / aspect;
    m[5] = -t;
    m[11] = -1.0;
    m[14] = near;
    m
}

/// Perspective projection to depth 0..1 (near..far), no y flip: a spot light's shadow map, looked
/// up through the same matrix (see [`ortho`]).
pub fn perspective_depth(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> M4 {
    let t = 1.0 / (0.5 * fov_y_radians).tan();
    let mut m = [0.0; 16];
    m[0] = t / aspect;
    m[5] = t;
    m[10] = far / (near - far);
    m[11] = -1.0;
    m[14] = near * far / (near - far);
    m
}

/// Orthographic projection to depth 0..1 (near..far), no y flip. Used for shadow maps, whose
/// lookups use the same matrix so the orientation cancels out.
pub fn ortho(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> M4 {
    let mut m = [0.0; 16];
    m[0] = 2.0 / (r - l);
    m[5] = 2.0 / (t - b);
    m[10] = -1.0 / (f - n);
    m[12] = -(r + l) / (r - l);
    m[13] = -(t + b) / (t - b);
    m[14] = -n / (f - n);
    m[15] = 1.0;
    m
}

/// Mirror about the horizontal plane `y = h`.
pub fn reflect_y(h: f32) -> M4 {
    let mut m = IDENTITY;
    m[5] = -1.0;
    m[13] = 2.0 * h;
    m
}

pub fn invert(m: &M4) -> Option<M4> {
    let a = m;
    let mut inv = [0.0f32; 16];
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15]
        + a[9] * a[7] * a[14]
        + a[13] * a[6] * a[11]
        - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15]
        - a[8] * a[7] * a[14]
        - a[12] * a[6] * a[11]
        + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15]
        + a[8] * a[7] * a[13]
        + a[12] * a[5] * a[11]
        - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14]
        - a[8] * a[6] * a[13]
        - a[12] * a[5] * a[10]
        + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15]
        - a[9] * a[3] * a[14]
        - a[13] * a[2] * a[11]
        + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15]
        + a[8] * a[3] * a[14]
        + a[12] * a[2] * a[11]
        - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15]
        - a[8] * a[3] * a[13]
        - a[12] * a[1] * a[11]
        + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14]
        + a[8] * a[2] * a[13]
        + a[12] * a[1] * a[10]
        - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15]
        + a[5] * a[3] * a[14]
        + a[13] * a[2] * a[7]
        - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15]
        - a[4] * a[3] * a[14]
        - a[12] * a[2] * a[7]
        + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15]
        + a[4] * a[3] * a[13]
        + a[12] * a[1] * a[7]
        - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14]
        - a[4] * a[2] * a[13]
        + a[12] * a[1] * a[6]
        + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11]
        - a[5] * a[3] * a[10]
        - a[9] * a[2] * a[7]
        + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11]
        + a[4] * a[3] * a[10]
        + a[8] * a[2] * a[7]
        - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11]
        - a[4] * a[3] * a[9]
        - a[8] * a[1] * a[7]
        + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10]
        + a[4] * a[2] * a[9]
        + a[8] * a[1] * a[6]
        - a[8] * a[2] * a[5];
    let det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if det.abs() < 1e-30 || !det.is_finite() {
        return None;
    }
    let d = 1.0 / det;
    Some(inv.map(|v| v * d))
}

/// sRGB-encoded 8-bit hex colour (`"#rrggbb"`) to linear RGB.
pub fn hex(s: &str) -> Option<V3> {
    let s = s.strip_prefix('#').unwrap_or(s);
    if s.len() != 6 || !s.is_ascii() {
        return None;
    }
    let c = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).ok();
    Some([
        srgb_to_linear(c(0)?),
        srgb_to_linear(c(2)?),
        srgb_to_linear(c(4)?),
    ])
}
pub fn srgb_to_linear(v: u8) -> f32 {
    let c = f32::from(v) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
pub fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Six inward-facing clip planes `a*x + b*y + c*z + d >= 0` extracted from a column-major
/// view-projection matrix whose clip volume is `-w <= x <= w`, `-w <= y <= w`, `0 <= z <= w`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    pub planes: [[f32; 4]; 6],
}

fn normalize_plane(p: [f32; 4]) -> [f32; 4] {
    let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    if len > 1e-12 {
        let inv = 1.0 / len;
        [p[0] * inv, p[1] * inv, p[2] * inv, p[3] * inv]
    } else {
        p
    }
}

impl Frustum {
    /// Extracts the six inward-pointing clip planes from `vp`.
    pub fn from_view_proj(vp: &M4) -> Self {
        let row = |r: usize| [vp[r], vp[4 + r], vp[8 + r], vp[12 + r]];
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);
        let plus = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let minus = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
        Self {
            planes: [
                normalize_plane(plus(r3, r0)),
                normalize_plane(minus(r3, r0)),
                normalize_plane(plus(r3, r1)),
                normalize_plane(minus(r3, r1)),
                normalize_plane(r2),
                normalize_plane(minus(r3, r2)),
            ],
        }
    }

    /// True when world-space point `p` lies inside or on the frustum boundary.
    pub fn contains_point(&self, p: V3) -> bool {
        self.planes
            .iter()
            .all(|pl| pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3] >= -1e-5)
    }

    /// Conservative AABB-frustum intersection test with zero false negatives. Empty AABBs
    /// (`min > max`) return `false`.
    pub fn intersects_aabb(&self, min: V3, max: V3) -> bool {
        if min[0] > max[0] || min[1] > max[1] || min[2] > max[2] {
            return false;
        }
        for pl in &self.planes {
            let px = if pl[0] >= 0.0 { max[0] } else { min[0] };
            let py = if pl[1] >= 0.0 { max[1] } else { min[1] };
            let pz = if pl[2] >= 0.0 { max[2] } else { min[2] };
            if pl[0] * px + pl[1] * py + pl[2] * pz + pl[3] < -1e-4 {
                return false;
            }
        }
        true
    }

    /// Conservative AABB test for depth-clamped orthographic passes (shadow cascades and top-down
    /// ground maps), omitting the `z >= 0` near plane so casters clamped to depth `0` are kept.
    pub fn intersects_aabb_clamped(&self, min: V3, max: V3) -> bool {
        if min[0] > max[0] || min[1] > max[1] || min[2] > max[2] {
            return false;
        }
        for i in [0, 1, 2, 3, 5] {
            let pl = &self.planes[i];
            let px = if pl[0] >= 0.0 { max[0] } else { min[0] };
            let py = if pl[1] >= 0.0 { max[1] } else { min[1] };
            let pz = if pl[2] >= 0.0 { max[2] } else { min[2] };
            if pl[0] * px + pl[1] * py + pl[2] * pz + pl[3] < -1e-4 {
                return false;
            }
        }
        true
    }
}

/// Shortest Euclidean distance from `eye` to axis-aligned box `(min, max)` (`0` inside).
pub fn aabb_distance(eye: V3, min: V3, max: V3) -> f32 {
    if min[0] > max[0] || min[1] > max[1] || min[2] > max[2] {
        return f32::INFINITY;
    }
    let dx = (min[0] - eye[0]).max(0.0).max(eye[0] - max[0]);
    let dy = (min[1] - eye[1]).max(0.0).max(eye[1] - max[1]);
    let dz = (min[2] - eye[2]).max(0.0).max(eye[2] - max[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// World-space diagonal length of axis-aligned box `(min, max)` (`0` when empty).
pub fn aabb_extent(min: V3, max: V3) -> f32 {
    if min[0] > max[0] || min[1] > max[1] || min[2] > max[2] {
        0.0
    } else {
        length(sub(max, min))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inverse_round_trips() {
        let v = look_at([3.0, 2.0, 5.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]);
        let p = mul(&perspective(1.0, 1.5, 0.1), &v);
        let i = invert(&p).unwrap();
        let id = mul(&p, &i);
        for (k, x) in id.iter().enumerate() {
            let want = if k % 5 == 0 { 1.0 } else { 0.0 };
            assert!((x - want).abs() < 1e-4, "{k}: {x}");
        }
    }
    #[test]
    fn reversed_depth_is_one_at_near() {
        let p = perspective(1.0, 1.0, 0.5);
        let c = transform(&p, [0.0, 0.0, -0.5, 1.0]);
        assert!((c[2] / c[3] - 1.0).abs() < 1e-6);
        let far = transform(&p, [0.0, 0.0, -1e6, 1.0]);
        assert!(far[2] / far[3] < 1e-5);
    }
    #[test]
    fn shadow_depth_runs_from_near_to_far() {
        let p = perspective_depth(1.2, 1.0, 0.1, 8.0);
        let depth = |z: f32| {
            let c = transform(&p, [0.0, 0.0, z, 1.0]);
            c[2] / c[3]
        };
        assert!(depth(-0.1).abs() < 1e-6);
        assert!((depth(-8.0) - 1.0).abs() < 1e-6);
        assert!(depth(-1.0) < depth(-2.0));
        let up = transform(&p, [0.0, (0.6f32).tan(), -1.0, 1.0]);
        assert!((up[1] / up[3] - 1.0).abs() < 1e-5);
    }
    #[test]
    fn perspective_frustum_matches_reversed_z_clip_volume() {
        let v = look_at([2.0, 3.0, 10.0], [2.0, 2.5, 0.0], [0.0, 1.0, 0.0]);
        let mut p = perspective(55f32.to_radians(), 16.0 / 9.0, 0.5);
        p[8] += 0.12;
        p[9] -= 0.08;
        let vp = mul(&p, &v);
        let f = Frustum::from_view_proj(&vp);

        for ix in -8..=12 {
            for iy in -6..=10 {
                for iz in -30..=15 {
                    let pt = [ix as f32 * 2.3, iy as f32 * 1.7, iz as f32 * 3.1];
                    let c = transform(&vp, [pt[0], pt[1], pt[2], 1.0]);
                    let clip_in = c[3] > 0.0
                        && c[0] >= -c[3]
                        && c[0] <= c[3]
                        && c[1] >= -c[3]
                        && c[1] <= c[3]
                        && c[2] >= 0.0
                        && c[2] <= c[3];
                    if clip_in {
                        assert!(f.contains_point(pt), "false negative on point {pt:?}");
                        let min = [pt[0] - 0.25, pt[1] - 0.25, pt[2] - 0.25];
                        let max = [pt[0] + 0.25, pt[1] + 0.25, pt[2] + 0.25];
                        assert!(f.intersects_aabb(min, max), "false negative on box {pt:?}");
                    }
                    let well_out = c[3] < -0.5
                        || c[0] < -c[3] - 2.0
                        || c[0] > c[3] + 2.0
                        || c[1] < -c[3] - 2.0
                        || c[1] > c[3] + 2.0
                        || c[2] > c[3] + 0.5;
                    if well_out {
                        assert!(!f.contains_point(pt), "false positive on point {pt:?}");
                    }
                }
            }
        }
        assert!(f.contains_point([2.0, -45.0, -1000.0]) || f.contains_point([2.0, -47.0, -1000.0]));
        assert!(!f.contains_point([2.0, 3.0, 12.0]));
        assert!(!f.contains_point([2.0, 3.0, 9.8]));
        assert!(!f.intersects_aabb([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]));
    }
    #[test]
    fn ortho_and_mirrored_frusta_classify_inside_and_outside_accurately() {
        let lv = look_at([20.0, 30.0, 20.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = mul(&ortho(-15.0, 15.0, -12.0, 12.0, 2.0, 80.0), &lv);
        let f = Frustum::from_view_proj(&vp);

        assert!(f.contains_point([0.0, 0.0, 0.0]));
        assert!(f.intersects_aabb([-1.0; 3], [1.0; 3]));
        assert!(!f.intersects_aabb([60.0, 0.0, -60.0], [65.0, 5.0, -55.0]));
        assert!(!f.intersects_aabb([-80.0, -80.0, -80.0], [-70.0, -70.0, -70.0]));
        assert!(!f.contains_point([25.0, 38.0, 25.0]));
        assert!(f.intersects_aabb_clamped([24.5, 37.0, 24.5], [25.5, 38.0, 25.5]));

        let cam = mul(
            &perspective(50f32.to_radians(), 1.5, 0.2),
            &look_at([0.0, 3.0, 8.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        );
        let refl_vp = mul(&cam, &reflect_y(0.0));
        let rf = Frustum::from_view_proj(&refl_vp);
        for ix in -5..=5 {
            for iy in -2..=8 {
                for iz in -10..=4 {
                    let pt = [ix as f32 * 1.5, iy as f32 * 1.2, iz as f32 * 1.8];
                    let c = transform(&refl_vp, [pt[0], pt[1], pt[2], 1.0]);
                    let clip_in = c[3] > 0.0
                        && c[0].abs() <= c[3]
                        && c[1].abs() <= c[3]
                        && c[2] >= 0.0
                        && c[2] <= c[3];
                    if clip_in {
                        assert!(
                            rf.intersects_aabb(
                                [pt[0] - 0.1, pt[1] - 0.1, pt[2] - 0.1],
                                [pt[0] + 0.1, pt[1] + 0.1, pt[2] + 0.1],
                            ),
                            "mirrored frustum false negative at {pt:?}"
                        );
                    }
                }
            }
        }
    }
}
