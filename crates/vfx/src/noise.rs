//! Deterministic SplitMix64 PRNG and analytic 3D divergence-free curl noise.

/// Deterministic 64-bit SplitMix64 generator used so timeline scrubbing reproduces identical
/// particle trajectories across native and WASM targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0.0, 1.0)`.
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    /// Uniform float in `[-1.0, 1.0)`.
    pub fn next_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    /// Uniform float between `min` and `max`.
    pub fn range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

fn hash_lattice(ix: i32, iy: i32, iz: i32, channel: u64) -> f32 {
    let mut h = (ix as i64 as u64).wrapping_mul(0x9E37_79B1_85EB_CA87);
    h ^= (iy as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= (iz as i64 as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= channel.wrapping_mul(0x27D4_EB2F_1656_67C5);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    let unit = ((h >> 40) as u32 as f32) * (1.0 / ((1u32 << 24) as f32));
    unit * 2.0 - 1.0
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn fade_deriv(t: f32) -> f32 {
    30.0 * t * t * (t * (t - 2.0) + 1.0)
}

/// Evaluates smooth 3D value noise and its analytic spatial gradient `[dn/dx, dn/dy, dn/dz]`.
pub fn scalar_noise_grad_3d(p: [f32; 3], channel: u64) -> (f32, [f32; 3]) {
    let fx = p[0].floor();
    let fy = p[1].floor();
    let fz = p[2].floor();
    let ix = fx as i32;
    let iy = fy as i32;
    let iz = fz as i32;

    let tx = p[0] - fx;
    let ty = p[1] - fy;
    let tz = p[2] - fz;

    let u = fade(tx);
    let v = fade(ty);
    let w = fade(tz);
    let du = fade_deriv(tx);
    let dv = fade_deriv(ty);
    let dw = fade_deriv(tz);

    let a = hash_lattice(ix, iy, iz, channel);
    let b = hash_lattice(ix + 1, iy, iz, channel);
    let c = hash_lattice(ix, iy + 1, iz, channel);
    let d = hash_lattice(ix + 1, iy + 1, iz, channel);
    let e = hash_lattice(ix, iy, iz + 1, channel);
    let f = hash_lattice(ix + 1, iy, iz + 1, channel);
    let g = hash_lattice(ix, iy + 1, iz + 1, channel);
    let h = hash_lattice(ix + 1, iy + 1, iz + 1, channel);

    let k0 = a;
    let k1 = b - a;
    let k2 = c - a;
    let k3 = e - a;
    let k4 = a - b - c + d;
    let k5 = a - c - e + g;
    let k6 = a - b - e + f;
    let k7 = -a + b + c - d + e - f - g + h;

    let value =
        k0 + k1 * u + k2 * v + k3 * w + k4 * u * v + k5 * v * w + k6 * w * u + k7 * u * v * w;

    let dx = du * (k1 + k4 * v + k6 * w + k7 * v * w);
    let dy = dv * (k2 + k4 * u + k5 * w + k7 * u * w);
    let dz = dw * (k3 + k6 * u + k5 * v + k7 * u * v);
    (value, [dx, dy, dz])
}

/// Evaluates divergence-free 3D curl noise (`nabla x psi`) for smooth turbulent advection.
pub fn curl_noise_3d(pos: [f32; 3], frequency: f32, time_s: f32, scroll_speed: f32) -> [f32; 3] {
    let freq = frequency.max(1e-4);
    let scroll = time_s * scroll_speed;
    let sample = [
        pos[0] * freq + scroll * 0.37,
        pos[1] * freq + scroll * 0.73,
        pos[2] * freq - scroll * 0.29,
    ];
    let (_, gx) = scalar_noise_grad_3d(sample, 1);
    let (_, gy) = scalar_noise_grad_3d(
        [sample[0] + 31.416, sample[1] - 47.853, sample[2] + 12.793],
        2,
    );
    let (_, gz) = scalar_noise_grad_3d(
        [sample[0] - 19.137, sample[1] + 23.619, sample[2] - 53.271],
        3,
    );

    // nabla x psi = [d psi_z / dy - d psi_y / dz, d psi_x / dz - d psi_z / dx, d psi_y / dx - d psi_x / dy]
    [gz[1] - gy[2], gx[2] - gz[0], gy[0] - gx[1]]
}
