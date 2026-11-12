//! Portable scene math, geometry and object transforms, plus the data layer every tool shares:
//! JSON (read and canonical write), content digests and identifiers.
pub mod digest;
pub mod id;
pub mod json;
pub mod timing;
pub mod world;
pub mod xr;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3(pub f32, pub f32, pub f32);

impl Vec3 {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self(x, y, z)
    }
    #[inline]
    pub fn dot(self, b: Self) -> f32 {
        self.0 * b.0 + self.1 * b.1 + self.2 * b.2
    }
    #[inline]
    pub fn cross(self, b: Self) -> Self {
        Self(
            self.1 * b.2 - self.2 * b.1,
            self.2 * b.0 - self.0 * b.2,
            self.0 * b.1 - self.1 * b.0,
        )
    }
    #[inline]
    pub fn minus(self, b: Self) -> Self {
        Self(self.0 - b.0, self.1 - b.1, self.2 - b.2)
    }
    #[inline]
    pub fn unit(self) -> Option<Self> {
        let len_sq = self.dot(self);
        if len_sq.is_finite() && len_sq > 1e-12 {
            let inv = 1.0 / len_sq.sqrt();
            Some(Self(self.0 * inv, self.1 * inv, self.2 * inv))
        } else {
            None
        }
    }
}

/// RGB values in the output target's 0..1 range.
pub type Rgb = [f32; 3];

#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub positions: [Vec3; 3],
    pub color: Rgb,
}

#[derive(Default, Debug)]
pub struct Scene {
    pub triangles: Vec<Triangle>,
}

impl Scene {
    /// Bounding sphere for framing a static scene. Compute once when loading.
    pub fn bounds(&self) -> Option<(Vec3, f32)> {
        let mut low = [f32::INFINITY; 3];
        let mut high = [f32::NEG_INFINITY; 3];
        for t in &self.triangles {
            for Vec3(x, y, z) in t.positions {
                for (i, v) in [x, y, z].into_iter().enumerate() {
                    if !v.is_finite() {
                        return None;
                    }
                    low[i] = low[i].min(v);
                    high[i] = high[i].max(v);
                }
            }
        }
        if self.triangles.is_empty() {
            return None;
        }
        let center = std::array::from_fn::<_, 3, _>(|i| low[i] * 0.5 + high[i] * 0.5);
        let radius = (0..3)
            .map(|i| (high[i] - center[i]).powi(2))
            .sum::<f32>()
            .sqrt();
        radius
            .is_finite()
            .then_some((Vec3(center[0], center[1], center[2]), radius.max(0.01)))
    }
    pub fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, color: Rgb) {
        self.triangles.push(Triangle {
            positions: [a, b, c],
            color,
        });
    }
    pub fn quad(&mut self, corners: [Vec3; 4], color: Rgb) {
        let [a, b, c, d] = corners;
        self.triangle(a, b, c, color);
        self.triangle(a, c, d, color);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub vertical_fov_degrees: f32,
    pub near: f32,
    pub far: f32,
}

/// Vulkan homogeneous clip coordinates. +Y in world space points up.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub clip: [f32; 4],
    pub rgba: [f32; 4],
}

impl Camera {
    /// Orbit an enclosing sphere at a fixed distance; angles are in radians.
    pub fn orbit((target, radius): (Vec3, f32), angle: f32) -> Self {
        let d = radius * 2.8;
        Self {
            eye: Vec3(
                target.0 + d * angle.sin(),
                target.1 + d * 0.35,
                target.2 + d * angle.cos(),
            ),
            target,
            vertical_fov_degrees: 48.,
            near: radius * 0.02,
            far: radius * 8.,
        }
    }
    /// Column-major world-to-clip matrix. Vulkan uses a 0..1 depth range.
    pub fn clip_matrix(self, width: u32, height: u32) -> Result<[f32; 16], &'static str> {
        if width == 0
            || height == 0
            || !self.near.is_finite()
            || !self.far.is_finite()
            || self.near <= 0.0
            || self.far <= self.near
            || !(0.0 < self.vertical_fov_degrees && self.vertical_fov_degrees < 179.0)
        {
            return Err("invalid viewport or camera frustum");
        }
        let forward = self
            .target
            .minus(self.eye)
            .unit()
            .ok_or("invalid camera direction")?;
        let right = forward
            .cross(Vec3(0.0, 1.0, 0.0))
            .unit()
            .ok_or("camera direction parallel to world up")?;
        let up = right.cross(forward);
        let scale = 1.0 / (0.5 * self.vertical_fov_degrees.to_radians()).tan();
        let aspect = width as f32 / height as f32;
        let depth = self.far / (self.far - self.near);
        let row = |v: Vec3, k: f32| [v.0 * k, v.1 * k, v.2 * k, -self.eye.dot(v) * k];
        let mut z = row(forward, depth);
        z[3] -= self.near * depth;
        let rows = [
            row(right, scale / aspect),
            row(up, -scale),
            z,
            row(forward, 1.0),
        ];
        Ok(std::array::from_fn(|i| rows[i % 4][i / 4]))
    }
    pub fn vertices(
        self,
        scene: &Scene,
        width: u32,
        height: u32,
    ) -> Result<Vec<Vertex>, &'static str> {
        let m = self.clip_matrix(width, height)?;
        let mut vertices = Vec::with_capacity(scene.triangles.len() * 3);
        for tri in &scene.triangles {
            for p in tri.positions {
                let clip = std::array::from_fn(|j| {
                    m[j] * p.0 + m[j + 4] * p.1 + m[j + 8] * p.2 + m[j + 12]
                });
                vertices.push(Vertex {
                    clip,
                    rgba: [tri.color[0], tri.color[1], tri.color[2], 1.0],
                });
            }
        }
        Ok(vertices)
    }
}
