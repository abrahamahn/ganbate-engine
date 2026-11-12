//! Vectors, quaternions, 3×3 matrices, rigid transforms and boxes for the physics engine.
//!
//! Units are metres, kilograms and seconds; +Y is up and coordinates are right-handed, as in the
//! rest of Ganbate. Everything a step computes uses only `+ - * /` and `sqrt` (IEEE-exact on every
//! platform, wasm included) and [`atan2`] below, never the platform's `sin`, `cos` or `atan`, so a
//! simulation given the same inputs produces the same bits everywhere.

use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 3-vector of `f32`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 1.0, 1.0);
    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self::new(v, v, v)
    }
    #[inline]
    pub fn from_array(a: [f32; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }
    #[inline]
    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
    #[inline]
    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }
    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }
    /// The unit vector along `self`, or `None` when it is too short to have a direction or not
    /// finite.
    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let (n, l) = self.normalize_and_length();
        (l > 0.0).then_some(n)
    }
    /// The unit vector along `self`, or zero when it has no direction.
    #[inline]
    pub fn normalize_or_zero(self) -> Self {
        self.try_normalize().unwrap_or(Self::ZERO)
    }
    /// The unit vector along `self` and its length (zero vector and 0 when it is too short to
    /// have a direction or not finite).
    #[inline]
    pub fn normalize_and_length(self) -> (Self, f32) {
        let l = self.length();
        if l > 1e-12 && l <= f32::MAX {
            (self * (1.0 / l), l)
        } else if l == f32::INFINITY && self.is_finite() {
            // The squared length overflowed (a component above about 1.8e19): the direction is
            // still there, after scaling by the largest component.
            let m = self.abs().max_element();
            let s = self * (1.0 / m);
            let ls = s.length();
            (s * (1.0 / ls), ls * m)
        } else {
            (Self::ZERO, 0.0)
        }
    }
    #[inline]
    pub fn mul_elem(self, o: Self) -> Self {
        Self::new(self.x * o.x, self.y * o.y, self.z * o.z)
    }
    #[inline]
    pub fn div_elem(self, o: Self) -> Self {
        Self::new(self.x / o.x, self.y / o.y, self.z / o.z)
    }
    #[inline]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
    #[inline]
    pub fn min(self, o: Self) -> Self {
        Self::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }
    #[inline]
    pub fn max(self, o: Self) -> Self {
        Self::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }
    #[inline]
    pub fn min_element(self) -> f32 {
        self.x.min(self.y).min(self.z)
    }
    #[inline]
    pub fn max_element(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }
    /// The index (0, 1 or 2) of the largest component; ties go to the lower index.
    #[inline]
    pub fn max_axis(self) -> usize {
        if self.x >= self.y && self.x >= self.z {
            0
        } else if self.y >= self.z {
            1
        } else {
            2
        }
    }
    #[inline]
    pub fn lerp(self, o: Self, t: f32) -> Self {
        self + (o - self) * t
    }
    #[inline]
    pub fn distance(self, o: Self) -> f32 {
        (o - self).length()
    }
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    /// Two unit vectors that complete `self` (which must be a unit vector) to a right-handed
    /// orthonormal basis `(t1, t2, self)` (Duff et al. 2017, branch on the sign of `z`).
    #[inline]
    pub fn orthonormal_basis(self) -> (Self, Self) {
        let n = self;
        let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
        let a = -1.0 / (sign + n.z);
        let b = n.x * n.y * a;
        let t1 = Self::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        let t2 = Self::new(b, sign + n.y * n.y * a, -n.y);
        (t1, t2)
    }
    /// Any unit vector perpendicular to `self` (which need not be a unit vector).
    #[inline]
    pub fn any_perpendicular(self) -> Self {
        let a = self.abs();
        let other = if a.x <= a.y && a.x <= a.z {
            Self::X
        } else if a.y <= a.z {
            Self::Y
        } else {
            Self::Z
        };
        self.cross(other).normalize_or_zero()
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}
impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}
impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, s: f32) -> Self {
        Self::new(self.x / s, self.y / s, self.z / s)
    }
}
impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl MulAssign<f32> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, s: f32) {
        *self = *self * s;
    }
}
impl Index<usize> for Vec3 {
    type Output = f32;
    #[inline]
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index {i} out of range"),
        }
    }
}
impl IndexMut<usize> for Vec3 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index {i} out of range"),
        }
    }
}

/// A rotation as a unit quaternion `(x, y, z, w)`, `w` the scalar part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Self = Self::new(0.0, 0.0, 0.0, 1.0);

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
    #[inline]
    pub fn from_array(a: [f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }
    #[inline]
    pub fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
    #[inline]
    pub fn vector(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
    /// A rotation of `angle` radians about the unit vector `axis` (the engine's own sine and
    /// cosine, so the same bits on every platform).
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let (s, c) = sin_cos(angle * 0.5);
        let a = axis.normalize_or_zero();
        Self::new(a.x * s, a.y * s, a.z * s, c).normalize()
    }
    /// A kinematic body's angular velocity is prescribed, so its turn over a substep is a
    /// rotation through |ω|·h. The normalized differential used for dynamic bodies instead
    /// turns through 2·atan(|ω|·h/2), short of a door's requested angle.
    pub(crate) fn from_angular_velocity(w: Vec3, h: f32) -> Self {
        let [x, y, z] = w.to_array().map(f64::from);
        // The length and angle of finite f32 inputs can exceed f32::MAX. Reduce the angle
        // before passing it to the deterministic engine sine and cosine.
        let speed = (x * x + y * y + z * z).sqrt();
        if speed == 0.0 {
            return Self::IDENTITY;
        }
        let axis = Vec3::new((x / speed) as f32, (y / speed) as f32, (z / speed) as f32);
        let angle = (speed * f64::from(h)) % std::f64::consts::TAU;
        Self::from_axis_angle(axis, angle as f32)
    }
    /// The shortest rotation taking unit vector `from` onto unit vector `to`.
    #[inline]
    pub fn from_rotation_arc(from: Vec3, to: Vec3) -> Self {
        let d = from.dot(to);
        if d < -0.999_999 {
            let axis = from.any_perpendicular();
            return Self::new(axis.x, axis.y, axis.z, 0.0);
        }
        let c = from.cross(to);
        Self::new(c.x, c.y, c.z, 1.0 + d).normalize()
    }
    #[inline]
    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w
    }
    #[inline]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    /// The unit quaternion along `self`; the identity when `self` has no length.
    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l > 1e-12 {
            let s = 1.0 / l;
            Self::new(self.x * s, self.y * s, self.z * s, self.w * s)
        } else {
            Self::IDENTITY
        }
    }
    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }
    /// Rotates `v` by this (unit) quaternion.
    #[inline]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let q = self.vector();
        let t = q.cross(v) * 2.0;
        v + t * self.w + q.cross(t)
    }
    /// Rotates `v` by the inverse of this (unit) quaternion.
    #[inline]
    pub fn inverse_rotate(self, v: Vec3) -> Vec3 {
        self.conjugate().rotate(v)
    }
    /// Advances this orientation by angular velocity `w` (radians per second, world frame) over
    /// `h` seconds: `q + h/2 · (w, 0) ⊗ q`, renormalized.
    #[inline]
    pub fn integrate(self, w: Vec3, h: f32) -> Self {
        let half = h * 0.5;
        let dq = Self::new(w.x * half, w.y * half, w.z * half, 0.0) * self;
        Self::new(self.x + dq.x, self.y + dq.y, self.z + dq.z, self.w + dq.w).normalize()
    }
    /// The rotation matrix.
    #[inline]
    pub fn to_mat3(self) -> Mat3 {
        let (x, y, z, w) = (self.x, self.y, self.z, self.w);
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, xy, xz) = (x * x2, x * y2, x * z2);
        let (yy, yz, zz) = (y * y2, y * z2, z * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        Mat3::from_cols(
            Vec3::new(1.0 - (yy + zz), xy + wz, xz - wy),
            Vec3::new(xy - wz, 1.0 - (xx + zz), yz + wx),
            Vec3::new(xz + wy, yz - wx, 1.0 - (xx + yy)),
        )
    }
    /// The quaternion of a rotation matrix (Shepperd's method).
    pub fn from_mat3(m: &Mat3) -> Self {
        let [c0, c1, c2] = m.cols;
        let (m00, m11, m22) = (c0.x, c1.y, c2.z);
        let trace = m00 + m11 + m22;
        let q = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Self::new(
                (c1.z - c2.y) / s,
                (c2.x - c0.z) / s,
                (c0.y - c1.x) / s,
                0.25 * s,
            )
        } else if m00 > m11 && m00 > m22 {
            let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
            Self::new(
                0.25 * s,
                (c1.x + c0.y) / s,
                (c2.x + c0.z) / s,
                (c1.z - c2.y) / s,
            )
        } else if m11 > m22 {
            let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
            Self::new(
                (c1.x + c0.y) / s,
                0.25 * s,
                (c2.y + c1.z) / s,
                (c2.x - c0.z) / s,
            )
        } else {
            let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
            Self::new(
                (c2.x + c0.z) / s,
                (c2.y + c1.z) / s,
                0.25 * s,
                (c0.y - c1.x) / s,
            )
        };
        q.normalize()
    }
    /// The rotation's angle about `axis` (a unit vector), in `(-π, π]`, by swing-twist
    /// decomposition. Deterministic: uses [`atan2`].
    #[inline]
    pub fn twist_angle(self, axis: Vec3) -> f32 {
        // q and -q are the same rotation; take the one with w >= 0 so the angle stays in (-π, π].
        let (s, c) = (self.vector().dot(axis), self.w);
        let (s, c) = if c < 0.0 { (-s, -c) } else { (s, c) };
        2.0 * atan2(s, c)
    }
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
}

impl Mul for Quat {
    type Output = Self;
    /// Hamilton product: `(a * b).rotate(v) == a.rotate(b.rotate(v))`.
    #[inline]
    fn mul(self, b: Self) -> Self {
        let a = self;
        Self::new(
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        )
    }
}

/// A matrix whose determinant is at most this fraction of the product of its column lengths is
/// singular (see [`Mat3::inverse`]).
const SINGULAR: f32 = 1e-12;

/// A 3×3 matrix stored as three columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub cols: [Vec3; 3],
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat3 {
    pub const IDENTITY: Self = Self {
        cols: [Vec3::X, Vec3::Y, Vec3::Z],
    };
    pub const ZERO: Self = Self {
        cols: [Vec3::ZERO; 3],
    };

    #[inline]
    pub const fn from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> Self {
        Self { cols: [c0, c1, c2] }
    }
    #[inline]
    pub fn from_diagonal(d: Vec3) -> Self {
        Self::from_cols(
            Vec3::new(d.x, 0.0, 0.0),
            Vec3::new(0.0, d.y, 0.0),
            Vec3::new(0.0, 0.0, d.z),
        )
    }
    /// The matrix of `v × _`.
    #[inline]
    pub fn skew(v: Vec3) -> Self {
        Self::from_cols(
            Vec3::new(0.0, v.z, -v.y),
            Vec3::new(-v.z, 0.0, v.x),
            Vec3::new(v.y, -v.x, 0.0),
        )
    }
    /// `a bᵀ`.
    #[inline]
    pub fn outer(a: Vec3, b: Vec3) -> Self {
        Self::from_cols(a * b.x, a * b.y, a * b.z)
    }
    #[inline]
    pub fn row(&self, i: usize) -> Vec3 {
        Vec3::new(self.cols[0][i], self.cols[1][i], self.cols[2][i])
    }
    #[inline]
    pub fn diagonal(&self) -> Vec3 {
        Vec3::new(self.cols[0].x, self.cols[1].y, self.cols[2].z)
    }
    #[inline]
    pub fn mul_vec(&self, v: Vec3) -> Vec3 {
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z
    }
    #[inline]
    pub fn transpose(&self) -> Self {
        Self::from_cols(self.row(0), self.row(1), self.row(2))
    }
    #[inline]
    pub fn determinant(&self) -> f32 {
        self.cols[0].dot(self.cols[1].cross(self.cols[2]))
    }
    /// The inverse, or `None` when the matrix is singular (or not finite).
    ///
    /// Singular is relative to the matrix's own size: the determinant is negligible next to the
    /// product of the column lengths, which bounds it (Hadamard). The matrix is first scaled to a
    /// largest element of 1, so tiny matrices (the inertia of a millimetre ball is about 1e-11
    /// kg·m², its determinant 1e-31) and huge ones invert without underflow or overflow.
    pub fn inverse(&self) -> Option<Self> {
        let largest = self.cols[0]
            .abs()
            .max(self.cols[1].abs())
            .max(self.cols[2].abs())
            .max_element();
        if !(largest > 0.0 && largest <= f32::MAX) {
            return None;
        }
        let s = 1.0 / largest;
        let [a, b, c] = self.cols.map(|col| col * s);
        let (r0, r1, r2) = (b.cross(c), c.cross(a), a.cross(b));
        let det = a.dot(r0);
        let bound = SINGULAR * (a.length() * b.length() * c.length());
        if det.is_nan() || det.abs() <= bound {
            return None;
        }
        // inverse(M) = s · inverse(s M).
        let inv = s / det;
        let m = Self::from_cols(r0 * inv, r1 * inv, r2 * inv).transpose();
        m.is_finite().then_some(m)
    }
    /// The inverse of a symmetric matrix, or zero when it is singular (a body that cannot turn
    /// about some axis).
    pub fn inverse_or_zero(&self) -> Self {
        self.inverse().unwrap_or(Self::ZERO)
    }
    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        Self::from_cols(self.cols[0] * s, self.cols[1] * s, self.cols[2] * s)
    }
    /// `R M Rᵀ`: this matrix (an inertia tensor in a local frame) seen in a frame rotated by `q`.
    #[inline]
    pub fn rotated(&self, q: Quat) -> Self {
        let r = q.to_mat3();
        r * *self * r.transpose()
    }
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.cols.iter().all(|c| c.is_finite())
    }
}

/// A symmetric 3×3 matrix stored as its six upper-triangle components (24 bytes vs 36 for [`Mat3`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SymMat3 {
    pub xx: f32,
    pub yy: f32,
    pub zz: f32,
    pub xy: f32,
    pub xz: f32,
    pub yz: f32,
}

impl SymMat3 {
    pub const ZERO: Self = Self {
        xx: 0.0,
        yy: 0.0,
        zz: 0.0,
        xy: 0.0,
        xz: 0.0,
        yz: 0.0,
    };

    #[inline]
    pub const fn new(xx: f32, yy: f32, zz: f32, xy: f32, xz: f32, yz: f32) -> Self {
        Self {
            xx,
            yy,
            zz,
            xy,
            xz,
            yz,
        }
    }

    #[inline]
    pub const fn from_diagonal(d: Vec3) -> Self {
        Self::new(d.x, d.y, d.z, 0.0, 0.0, 0.0)
    }

    #[inline]
    pub fn from_mat3(m: &Mat3) -> Self {
        Self::new(
            m.cols[0].x,
            m.cols[1].y,
            m.cols[2].z,
            m.cols[1].x,
            m.cols[2].x,
            m.cols[2].y,
        )
    }

    #[inline]
    pub fn to_mat3(self) -> Mat3 {
        Mat3::from_cols(
            Vec3::new(self.xx, self.xy, self.xz),
            Vec3::new(self.xy, self.yy, self.yz),
            Vec3::new(self.xz, self.yz, self.zz),
        )
    }

    #[inline]
    pub fn diagonal(self) -> Vec3 {
        Vec3::new(self.xx, self.yy, self.zz)
    }

    #[inline]
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3::new(self.xx, self.xy, self.xz) * v.x
            + Vec3::new(self.xy, self.yy, self.yz) * v.y
            + Vec3::new(self.xz, self.yz, self.zz) * v.z
    }
}

impl Mul for Mat3 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self::from_cols(
            self.mul_vec(o.cols[0]),
            self.mul_vec(o.cols[1]),
            self.mul_vec(o.cols[2]),
        )
    }
}
impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        self.mul_vec(v)
    }
}
impl Add for Mat3 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::from_cols(
            self.cols[0] + o.cols[0],
            self.cols[1] + o.cols[1],
            self.cols[2] + o.cols[2],
        )
    }
}
impl Sub for Mat3 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::from_cols(
            self.cols[0] - o.cols[0],
            self.cols[1] - o.cols[1],
            self.cols[2] - o.cols[2],
        )
    }
}

/// A rigid transform: rotate, then translate. `iso.point(p) = rotation · p + position`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Iso {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Iso {
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    #[inline]
    pub const fn new(position: Vec3, rotation: Quat) -> Self {
        Self { position, rotation }
    }
    #[inline]
    pub const fn from_position(position: Vec3) -> Self {
        Self::new(position, Quat::IDENTITY)
    }
    #[inline]
    pub fn point(&self, p: Vec3) -> Vec3 {
        self.rotation.rotate(p) + self.position
    }
    #[inline]
    pub fn vector(&self, v: Vec3) -> Vec3 {
        self.rotation.rotate(v)
    }
    #[inline]
    pub fn inverse_point(&self, p: Vec3) -> Vec3 {
        self.rotation.inverse_rotate(p - self.position)
    }
    #[inline]
    pub fn inverse_vector(&self, v: Vec3) -> Vec3 {
        self.rotation.inverse_rotate(v)
    }
    /// `self ∘ local`: first `local`, then `self`.
    #[inline]
    pub fn mul(&self, local: &Iso) -> Iso {
        Iso::new(
            self.point(local.position),
            (self.rotation * local.rotation).normalize(),
        )
    }
    #[inline]
    pub fn inverse(&self) -> Iso {
        let r = self.rotation.conjugate();
        Iso::new(r.rotate(-self.position), r)
    }
    /// `self⁻¹ ∘ other`: `other` expressed in this frame.
    #[inline]
    pub fn inverse_mul(&self, other: &Iso) -> Iso {
        Iso::new(
            self.inverse_point(other.position),
            (self.rotation.conjugate() * other.rotation).normalize(),
        )
    }
    /// The column-major 4×4 matrix (as in `ganbate_core::world::Transform`).
    pub fn to_matrix(&self) -> [f32; 16] {
        let m = self.rotation.to_mat3();
        let [c0, c1, c2] = m.cols;
        let p = self.position;
        [
            c0.x, c0.y, c0.z, 0.0, c1.x, c1.y, c1.z, 0.0, c2.x, c2.y, c2.z, 0.0, p.x, p.y, p.z, 1.0,
        ]
    }
    /// The rigid part of a column-major 4×4 matrix: its translation and the rotation of its
    /// normalized basis (scale is dropped; shear makes the result approximate).
    pub fn from_matrix(m: &[f32; 16]) -> Iso {
        let c0 = Vec3::new(m[0], m[1], m[2]).normalize_or_zero();
        let c1 = Vec3::new(m[4], m[5], m[6]).normalize_or_zero();
        let c2 = Vec3::new(m[8], m[9], m[10]).normalize_or_zero();
        Iso::new(
            Vec3::new(m[12], m[13], m[14]),
            Quat::from_mat3(&Mat3::from_cols(c0, c1, c2)),
        )
    }
}

/// An axis-aligned box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// The empty box: union with it changes nothing; it overlaps nothing.
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    #[inline]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }
    #[inline]
    pub fn from_center_half(center: Vec3, half: Vec3) -> Self {
        Self::new(center - half, center + half)
    }
    pub fn from_points(points: &[Vec3]) -> Self {
        points.iter().fold(Self::EMPTY, |b, &p| b.include(p))
    }
    #[inline]
    pub fn include(self, p: Vec3) -> Self {
        Self::new(self.min.min(p), self.max.max(p))
    }
    #[inline]
    pub fn union(self, o: Self) -> Self {
        Self::new(self.min.min(o.min), self.max.max(o.max))
    }
    #[inline]
    pub fn expand(self, margin: f32) -> Self {
        Self::new(
            self.min - Vec3::splat(margin),
            self.max + Vec3::splat(margin),
        )
    }
    /// This box swept along `d`: the union of it and it moved by `d`.
    #[inline]
    pub fn sweep(self, d: Vec3) -> Self {
        Self::new(self.min + d.min(Vec3::ZERO), self.max + d.max(Vec3::ZERO))
    }
    #[inline]
    pub fn overlaps(&self, o: &Self) -> bool {
        self.min.x <= o.max.x
            && self.max.x >= o.min.x
            && self.min.y <= o.max.y
            && self.max.y >= o.min.y
            && self.min.z <= o.max.z
            && self.max.z >= o.min.z
    }
    #[inline]
    pub fn contains(&self, o: &Self) -> bool {
        self.min.x <= o.min.x
            && self.min.y <= o.min.y
            && self.min.z <= o.min.z
            && self.max.x >= o.max.x
            && self.max.y >= o.max.y
            && self.max.z >= o.max.z
    }
    #[inline]
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.y >= self.min.y
            && p.z >= self.min.z
            && p.x <= self.max.x
            && p.y <= self.max.y
            && p.z <= self.max.z
    }
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    #[inline]
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }
    /// Half the surface area: the cost measure of a bounding-volume hierarchy.
    #[inline]
    pub fn half_area(&self) -> f32 {
        let d = self.max - self.min;
        d.x * d.y + d.y * d.z + d.z * d.x
    }
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.min.is_finite()
            && self.max.is_finite()
            && self.min.x <= self.max.x
            && self.min.y <= self.max.y
            && self.min.z <= self.max.z
    }
    /// Where a ray `origin + t·dir` enters this box, for `t` in `[0, max_t]` (slab test; a ray
    /// starting inside enters at 0).
    #[inline]
    pub fn ray_entry(&self, origin: Vec3, inv_dir: Vec3, max_t: f32) -> Option<f32> {
        let mut t0 = 0.0f32;
        let mut t1 = max_t;
        for i in 0..3 {
            let (o, inv) = (origin[i], inv_dir[i]);
            let (lo, hi) = (self.min[i], self.max[i]);
            if inv.is_infinite() {
                // Parallel to this slab: inside it or never.
                if o < lo || o > hi {
                    return None;
                }
                continue;
            }
            let (mut a, mut b) = ((lo - o) * inv, (hi - o) * inv);
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            t0 = t0.max(a);
            t1 = t1.min(b);
            if t0 > t1 {
                return None;
            }
        }
        Some(t0)
    }
}

/// The reciprocal of each component, infinite (with the component's sign) where it is zero: what
/// [`Aabb::ray_entry`] takes.
#[inline]
pub fn inverse_direction(d: Vec3) -> Vec3 {
    let inv = |v: f32| {
        if v == 0.0 {
            if v.is_sign_negative() {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            1.0 / v
        }
    };
    Vec3::new(inv(d.x), inv(d.y), inv(d.z))
}

/// `(sin x, cos x)` from basic arithmetic only: the argument is reduced to `[-π/4, π/4]` with a
/// three-part `π/2` and the quadrant, then Cephes' single-precision polynomials run on the rest
/// (error within a few units in the last place for `|x|` up to thousands of radians), so rotations
/// built from angles are identical on every platform.
pub fn sin_cos(x: f32) -> (f32, f32) {
    if !x.is_finite() {
        return (f32::NAN, f32::NAN);
    }
    // π/2 in three parts with few mantissa bits each (Cephes' DP1-DP3, doubled), so `k` times
    // the first two is exact for any `k` below 2^13.
    const PIO2_1: f32 = 1.570_312_5;
    const PIO2_2: f32 = 4.837_513e-4;
    const PIO2_3: f32 = 7.549_79e-8;
    let k = (x * std::f32::consts::FRAC_2_PI).round();
    let r = ((x - k * PIO2_1) - k * PIO2_2) - k * PIO2_3;
    let z = r * r;
    let s = ((-1.951_529_6e-4 * z + 8.332_161e-3) * z - 1.666_665_5e-1) * z * r + r;
    let c = ((2.443_315_7e-5 * z - 1.388_731_6e-3) * z + 4.166_664_6e-2) * z * z - 0.5 * z + 1.0;
    // The quadrant: k mod 4, also for negative k.
    match (k as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// `atan2(y, x)` in `(-π, π]` from basic arithmetic only (the polynomial for `atan` on `[0, 1]`
/// of Abramowitz and Stegun 4.4.49, absolute error below 1e-5 rad), so joint angles are identical
/// on every platform.
pub fn atan2(y: f32, x: f32) -> f32 {
    use std::f32::consts::{FRAC_PI_2, PI};
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }
    let (ax, ay) = (x.abs(), y.abs());
    let (num, den, swap) = if ay > ax {
        (ax, ay, true)
    } else {
        (ay, ax, false)
    };
    let z = num / den;
    let z2 = z * z;
    let mut a = z
        * (0.999_866
            + z2 * (-0.330_299_5 + z2 * (0.180_141 + z2 * (-0.085_133 + z2 * 0.020_835_1))));
    if swap {
        a = FRAC_PI_2 - a;
    }
    if x < 0.0 {
        a = PI - a;
    }
    if y < 0.0 { -a } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }
    fn close_v(a: Vec3, b: Vec3, tol: f32) -> bool {
        (a - b).length() <= tol
    }

    #[test]
    fn quaternions_rotate_compose_and_round_trip_matrices() {
        let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.7);
        let r = Quat::from_axis_angle(Vec3::new(-2.0, 0.5, 1.0), -1.3);
        let v = Vec3::new(0.3, -1.2, 2.5);
        assert!(close_v((q * r).rotate(v), q.rotate(r.rotate(v)), 1e-5));
        assert!(close_v(q.inverse_rotate(q.rotate(v)), v, 1e-5));
        assert!(close_v(q.to_mat3().mul_vec(v), q.rotate(v), 1e-5));
        let back = Quat::from_mat3(&q.to_mat3());
        assert!(back.dot(q).abs() > 0.99999);
        let arc = Quat::from_rotation_arc(Vec3::X, Vec3::new(0.0, 0.6, 0.8));
        assert!(close_v(arc.rotate(Vec3::X), Vec3::new(0.0, 0.6, 0.8), 1e-5));
        let flip = Quat::from_rotation_arc(Vec3::Y, -Vec3::Y);
        assert!(close_v(flip.rotate(Vec3::Y), -Vec3::Y, 1e-5));
    }

    #[test]
    fn integrating_a_spin_turns_by_its_angle() {
        // A quarter turn about +Y at 1 rad/s, in many small steps.
        let mut q = Quat::IDENTITY;
        let steps = 1000;
        let h = std::f32::consts::FRAC_PI_2 / steps as f32;
        for _ in 0..steps {
            q = q.integrate(Vec3::Y, h);
        }
        assert!(
            close_v(q.rotate(Vec3::X), -Vec3::Z, 2e-3),
            "{:?}",
            q.rotate(Vec3::X)
        );
        assert!(close(
            q.twist_angle(Vec3::Y),
            std::f32::consts::FRAC_PI_2,
            2e-3
        ));
    }

    #[test]
    fn matrices_invert_and_rotate_inertia() {
        let m = Mat3::from_cols(
            Vec3::new(4.0, 1.0, 0.5),
            Vec3::new(1.0, 3.0, 0.2),
            Vec3::new(0.5, 0.2, 2.0),
        );
        let p = m * m.inverse().unwrap();
        for i in 0..3 {
            assert!(close_v(p.cols[i], Mat3::IDENTITY.cols[i], 1e-5));
        }
        assert!(Mat3::ZERO.inverse().is_none());
        let q = Quat::from_axis_angle(Vec3::Z, 0.9);
        let d = Mat3::from_diagonal(Vec3::new(1.0, 2.0, 3.0));
        let w = d.rotated(q);
        // The rotated tensor maps a rotated axis as the original mapped the axis.
        assert!(close_v(
            w.mul_vec(q.rotate(Vec3::X)),
            q.rotate(d.mul_vec(Vec3::X)),
            1e-5
        ));
        assert!(close_v(
            Mat3::skew(Vec3::new(1.0, 2.0, 3.0)).mul_vec(Vec3::new(-1.0, 0.5, 2.0)),
            Vec3::new(1.0, 2.0, 3.0).cross(Vec3::new(-1.0, 0.5, 2.0)),
            1e-6
        ));
    }

    #[test]
    fn isometries_compose_and_invert() {
        let a = Iso::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_axis_angle(Vec3::Y, 0.4),
        );
        let b = Iso::new(
            Vec3::new(-2.0, 0.5, 1.0),
            Quat::from_axis_angle(Vec3::X, -1.1),
        );
        let p = Vec3::new(0.2, -0.7, 1.5);
        assert!(close_v(a.mul(&b).point(p), a.point(b.point(p)), 1e-5));
        assert!(close_v(a.inverse().point(a.point(p)), p, 1e-5));
        assert!(close_v(
            a.inverse_mul(&a.mul(&b)).point(p),
            b.point(p),
            1e-5
        ));
        let m = a.to_matrix();
        let back = Iso::from_matrix(&m);
        assert!(close_v(back.point(p), a.point(p), 1e-5));
    }

    #[test]
    fn inverses_do_not_depend_on_the_matrix_scale() {
        let m = Mat3::from_cols(
            Vec3::new(4.0, 1.0, 0.5),
            Vec3::new(1.0, 3.0, 0.2),
            Vec3::new(0.5, 0.2, 2.0),
        );
        let unit = m.inverse().unwrap();
        // From a 0.1 mm ball's inertia (determinant far below f32's smallest number) to a
        // planet's.
        for scale in [1e-17f32, 1e-11, 1e-3, 1e3, 1e12, 1e30] {
            let inv = m
                .scale(scale)
                .inverse()
                .unwrap_or_else(|| panic!("scale {scale}"));
            for i in 0..3 {
                let want = unit.cols[i] * (1.0 / scale);
                assert!(
                    (inv.cols[i] - want).length() <= 1e-5 * want.length(),
                    "scale {scale}: {:?} against {want:?}",
                    inv.cols[i]
                );
            }
        }
        // Singular at any scale: two equal columns, a zero column, or nothing at all.
        let flat = Mat3::from_cols(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        for scale in [1e-20f32, 1.0, 1e20] {
            assert!(flat.scale(scale).inverse().is_none(), "scale {scale}");
            let mut z = m.scale(scale);
            z.cols[1] = Vec3::ZERO;
            assert!(z.inverse().is_none(), "scale {scale}");
        }
        assert!(Mat3::ZERO.inverse().is_none());
        let mut bad = m;
        bad.cols[2].y = f32::NAN;
        assert!(bad.inverse().is_none());
        bad.cols[2].y = f32::INFINITY;
        assert!(bad.inverse().is_none());
        assert_eq!(Mat3::ZERO.inverse_or_zero(), Mat3::ZERO);
    }

    #[test]
    fn normalizing_keeps_huge_directions_and_refuses_non_finite_ones() {
        // The squared length of these overflows f32; the direction does not.
        let n = Vec3::new(1e20, 0.0, 1e20).try_normalize().unwrap();
        assert!(
            close_v(n, Vec3::new(1.0, 0.0, 1.0) * (0.5f32.sqrt()), 1e-6),
            "{n:?}"
        );
        let (n, l) = Vec3::new(0.0, -3e19, 4e19).normalize_and_length();
        assert!(
            close_v(n, Vec3::new(0.0, -0.6, 0.8), 1e-6) && close(l, 5e19, 5e13),
            "{n:?} {l}"
        );
        for bad in [
            Vec3::new(f32::INFINITY, 0.0, 0.0),
            Vec3::new(1.0, f32::NEG_INFINITY, 0.0),
            Vec3::new(f32::NAN, 1.0, 0.0),
            Vec3::ZERO,
            Vec3::splat(1e-13),
        ] {
            assert_eq!(bad.try_normalize(), None, "{bad:?}");
            assert_eq!(bad.normalize_and_length(), (Vec3::ZERO, 0.0), "{bad:?}");
            assert_eq!(bad.normalize_or_zero(), Vec3::ZERO, "{bad:?}");
        }
        // Ordinary vectors are untouched, bit for bit.
        let v = Vec3::new(0.3, -1.2, 2.5);
        let l = v.length();
        assert_eq!(v.try_normalize(), Some(v * (1.0 / l)));
    }

    #[test]
    fn bases_are_orthonormal_for_every_direction() {
        for n in [
            Vec3::X,
            -Vec3::X,
            Vec3::Y,
            -Vec3::Y,
            Vec3::Z,
            -Vec3::Z,
            Vec3::new(1.0, 1.0, 1.0).normalize_or_zero(),
            Vec3::new(-0.3, 0.1, -0.95).normalize_or_zero(),
        ] {
            let (t1, t2) = n.orthonormal_basis();
            assert!(close(t1.length(), 1.0, 1e-5) && close(t2.length(), 1.0, 1e-5));
            assert!(
                close(t1.dot(n), 0.0, 1e-5)
                    && close(t2.dot(n), 0.0, 1e-5)
                    && close(t1.dot(t2), 0.0, 1e-5)
            );
            assert!(close_v(t1.cross(t2), n, 1e-5), "right-handed for {n:?}");
            assert!(close(n.any_perpendicular().dot(n), 0.0, 1e-5));
        }
    }

    #[test]
    fn deterministic_sin_cos_matches_the_platform_closely() {
        let mut worst = 0.0f32;
        for i in -20_000..=20_000 {
            let x = i as f32 * 0.005_123;
            let (s, c) = sin_cos(x);
            worst = worst.max((s - x.sin()).abs()).max((c - x.cos()).abs());
        }
        assert!(worst < 1e-6, "worst error {worst}");
        assert_eq!(sin_cos(0.0), (0.0, 1.0));
        assert!(sin_cos(f32::NAN).0.is_nan() && sin_cos(f32::INFINITY).1.is_nan());
        let q = Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        assert!(close_v(q.rotate(Vec3::X), Vec3::new(0.0, 0.0, -1.0), 1e-6));
    }

    #[test]
    fn deterministic_atan2_matches_the_platform_closely() {
        let mut worst = 0.0f32;
        for i in 0..=720 {
            let a = -std::f32::consts::PI + i as f32 * (std::f32::consts::TAU / 720.0);
            for r in [0.001f32, 1.0, 250.0] {
                let (y, x) = (a.sin() * r, a.cos() * r);
                worst = worst.max((atan2(y, x) - y.atan2(x)).abs());
            }
        }
        assert!(worst < 2e-5, "worst error {worst}");
        assert_eq!(atan2(0.0, 0.0), 0.0);
    }

    #[test]
    fn boxes_sweep_overlap_and_cast_rays() {
        let b = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(b.overlaps(&Aabb::new(Vec3::splat(0.5), Vec3::splat(2.0))));
        assert!(!b.overlaps(&Aabb::new(Vec3::splat(1.5), Vec3::splat(2.0))));
        let s = b.sweep(Vec3::new(3.0, 0.0, -2.0));
        assert_eq!(
            s,
            Aabb::new(Vec3::new(-1.0, -1.0, -3.0), Vec3::new(4.0, 1.0, 1.0))
        );
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let t = b.ray_entry(Vec3::new(-5.0, 0.0, 0.0), inverse_direction(dir), 100.0);
        assert!(close(t.unwrap(), 4.0, 1e-6));
        assert!(
            b.ray_entry(Vec3::new(-5.0, 2.0, 0.0), inverse_direction(dir), 100.0)
                .is_none()
        );
        assert_eq!(
            b.ray_entry(Vec3::ZERO, inverse_direction(dir), 100.0),
            Some(0.0)
        );
        assert!(Aabb::EMPTY.union(b) == b && !Aabb::EMPTY.overlaps(&b));
    }
}
