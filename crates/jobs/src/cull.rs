//! SIMD Structure-of-Arrays (SoA) frustum, distance, and subpixel screen-size culling,
//! Arvo 9-multiply AABB transforms, and `O(N)` 16-bit radix sort for GPU Early-Z submission.

use crate::pool::JobPool;

/// Axis-aligned bounding box in 3D space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub const ZERO: Self = Self {
        min: [0.0; 3],
        max: [0.0; 3],
    };

    #[inline]
    pub const fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn center_extents(&self) -> ([f32; 3], [f32; 3]) {
        let cx = 0.5 * (self.min[0] + self.max[0]);
        let cy = 0.5 * (self.min[1] + self.max[1]);
        let cz = 0.5 * (self.min[2] + self.max[2]);
        let ex = 0.5 * (self.max[0] - self.min[0]).max(0.0);
        let ey = 0.5 * (self.max[1] - self.min[1]).max(0.0);
        let ez = 0.5 * (self.max[2] - self.min[2]).max(0.0);
        ([cx, cy, cz], [ex, ey, ez])
    }
}

/// Transforms a local-space [`Aabb`] by a column-major `4x4` affine world matrix using
/// Jim Arvo's 9-multiply method (9 multiplies instead of transforming all 8 corners = 96 multiplies).
#[inline]
pub fn transform_aabb(local: &Aabb, world_col_major: &[f32; 16]) -> Aabb {
    let mut min = [
        world_col_major[12],
        world_col_major[13],
        world_col_major[14],
    ];
    let mut max = min;
    for col in 0..3 {
        let lo = local.min[col];
        let hi = local.max[col];
        let base = col * 4;
        for row in 0..3 {
            let m = world_col_major[base + row];
            let a = m * lo;
            let b = m * hi;
            if a < b {
                min[row] += a;
                max[row] += b;
            } else {
                min[row] += b;
                max[row] += a;
            }
        }
    }
    Aabb { min, max }
}

/// Normalized inward-facing plane `n.x * x + n.y * y + n.z * z + d >= 0`, with precomputed `|n|`
/// for branchless center-extent AABB testing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub n: [f32; 3],
    pub d: f32,
    pub abs_n: [f32; 3],
}

impl Plane {
    #[inline]
    pub fn from_unnormalized(a: f32, b: f32, c: f32, d: f32) -> Self {
        let len_sq = a * a + b * b + c * c;
        let inv = if len_sq > 1e-24 {
            1.0 / len_sq.sqrt()
        } else {
            0.0
        };
        let nx = a * inv;
        let ny = b * inv;
        let nz = c * inv;
        Self {
            n: [nx, ny, nz],
            d: d * inv,
            abs_n: [nx.abs(), ny.abs(), nz.abs()],
        }
    }
}

/// Six-plane camera view frustum (`[Left, Right, Bottom, Top, Near, Far]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Extracts inward-facing frustum planes from a column-major view-projection matrix using
    /// `[0, 1]` clip-space depth (Vulkan, Metal, WebGPU).
    pub fn from_view_proj_zero_to_one(vp: &[f32; 16]) -> Self {
        let r = |row: usize, col: usize| vp[col * 4 + row];
        let left = Plane::from_unnormalized(
            r(3, 0) + r(0, 0),
            r(3, 1) + r(0, 1),
            r(3, 2) + r(0, 2),
            r(3, 3) + r(0, 3),
        );
        let right = Plane::from_unnormalized(
            r(3, 0) - r(0, 0),
            r(3, 1) - r(0, 1),
            r(3, 2) - r(0, 2),
            r(3, 3) - r(0, 3),
        );
        let bottom = Plane::from_unnormalized(
            r(3, 0) + r(1, 0),
            r(3, 1) + r(1, 1),
            r(3, 2) + r(1, 2),
            r(3, 3) + r(1, 3),
        );
        let top = Plane::from_unnormalized(
            r(3, 0) - r(1, 0),
            r(3, 1) - r(1, 1),
            r(3, 2) - r(1, 2),
            r(3, 3) - r(1, 3),
        );
        let near = Plane::from_unnormalized(r(2, 0), r(2, 1), r(2, 2), r(2, 3));
        let far = Plane::from_unnormalized(
            r(3, 0) - r(2, 0),
            r(3, 1) - r(2, 1),
            r(3, 2) - r(2, 2),
            r(3, 3) - r(2, 3),
        );
        Self {
            planes: [left, right, bottom, top, near, far],
        }
    }

    /// Extracts inward-facing frustum planes from a column-major view-projection matrix using
    /// `[-1, 1]` clip-space depth (OpenGL, WebGL2).
    pub fn from_view_proj_neg_one_to_one(vp: &[f32; 16]) -> Self {
        let r = |row: usize, col: usize| vp[col * 4 + row];
        let mut f = Self::from_view_proj_zero_to_one(vp);
        f.planes[4] = Plane::from_unnormalized(
            r(3, 0) + r(2, 0),
            r(3, 1) + r(2, 1),
            r(3, 2) + r(2, 2),
            r(3, 3) + r(2, 3),
        );
        f
    }

    /// Scalar single-AABB frustum intersection test (`true` if inside or intersecting).
    #[inline]
    pub fn contains_aabb(&self, aabb: &Aabb) -> bool {
        let (c, e) = aabb.center_extents();
        for p in &self.planes {
            let s = c[0] * p.n[0] + c[1] * p.n[1] + c[2] * p.n[2] + p.d;
            let r = e[0] * p.abs_n[0] + e[1] * p.abs_n[1] + e[2] * p.abs_n[2];
            if s + r < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Optional distance, screen-space subpixel contribution, and Early-Z depth quantization settings.
#[derive(Clone, Copy, Debug)]
pub struct CullConfig {
    /// World-space camera origin for distance and screen-size tests.
    pub camera_pos: [f32; 3],
    /// Maximum draw distance (`f32::INFINITY` disables distance culling).
    pub max_distance: f32,
    /// Minimum projected diameter in pixels (`0.0` disables subpixel culling).
    pub min_screen_px: f32,
    /// Vertical projection scale `cot(fov_y / 2)` (typically `proj[5].abs()`).
    pub proj_scale_y: f32,
    /// Viewport height in physical pixels.
    pub viewport_height_px: f32,
    /// Depth span mapped onto `0..=65535` for front-to-back Early-Z sorting.
    pub depth_range: f32,
}

impl Default for CullConfig {
    fn default() -> Self {
        Self {
            camera_pos: [0.0; 3],
            max_distance: f32::INFINITY,
            min_screen_px: 0.0,
            proj_scale_y: 1.732_050_8,
            viewport_height_px: 1080.0,
            depth_range: 1000.0,
        }
    }
}

/// Telemetry counters returned by a culling pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CullStats {
    pub total: u32,
    pub visible: u32,
    pub frustum_culled: u32,
    pub distance_or_subpixel_culled: u32,
}

impl CullStats {
    #[inline]
    pub fn merge(&mut self, other: Self) {
        self.total += other.total;
        self.visible += other.visible;
        self.frustum_culled += other.frustum_culled;
        self.distance_or_subpixel_culled += other.distance_or_subpixel_culled;
    }
}

/// 8-wide Structure-of-Arrays (SoA) AABB packet (`192` bytes, 32-byte aligned for AVX2).
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
pub struct AabbSoa8 {
    pub cx: [f32; 8],
    pub cy: [f32; 8],
    pub cz: [f32; 8],
    pub ex: [f32; 8],
    pub ey: [f32; 8],
    pub ez: [f32; 8],
}

impl Default for AabbSoa8 {
    fn default() -> Self {
        Self {
            cx: [0.0; 8],
            cy: [0.0; 8],
            cz: [0.0; 8],
            ex: [0.0; 8],
            ey: [0.0; 8],
            ez: [0.0; 8],
        }
    }
}

/// Packs a slice of [`Aabb`]s into 8-wide [`AabbSoa8`] packets, reusing `out`'s allocation.
pub fn pack_aabbs_soa(aabbs: &[Aabb], out: &mut Vec<AabbSoa8>) {
    let batches = aabbs.len().div_ceil(8);
    out.clear();
    out.reserve(batches);
    for chunk in aabbs.chunks(8) {
        let mut b = AabbSoa8::default();
        for (lane, aabb) in chunk.iter().enumerate() {
            let (c, e) = aabb.center_extents();
            b.cx[lane] = c[0];
            b.cy[lane] = c[1];
            b.cz[lane] = c[2];
            b.ex[lane] = e[0];
            b.ey[lane] = e[1];
            b.ez[lane] = e[2];
        }
        out.push(b);
    }
}

#[derive(Clone, Copy)]
struct DerivedCullParams {
    use_dist: bool,
    max_dist: f32,
    use_subpx: bool,
    subpx_k_sq: f32,
    inv_depth_range: f32,
    cam: [f32; 3],
}

impl DerivedCullParams {
    fn new(cfg: &CullConfig) -> Self {
        let use_dist = cfg.max_distance.is_finite() && cfg.max_distance >= 0.0;
        let use_subpx = cfg.min_screen_px > 0.0
            && cfg.proj_scale_y > 0.0
            && cfg.viewport_height_px > 0.0
            && cfg.min_screen_px.is_finite();
        let subpx_k = if use_subpx {
            (cfg.proj_scale_y * cfg.viewport_height_px) / cfg.min_screen_px
        } else {
            0.0
        };
        let inv_depth_range = if cfg.depth_range > 1e-6 && cfg.depth_range.is_finite() {
            1.0 / cfg.depth_range
        } else {
            1.0 / 1000.0
        };
        Self {
            use_dist,
            max_dist: cfg.max_distance,
            use_subpx,
            subpx_k_sq: subpx_k * subpx_k,
            inv_depth_range,
            cam: cfg.camera_pos,
        }
    }

    #[inline]
    fn quantize_depth(&self, near_dist: f32) -> u16 {
        let t = (near_dist * self.inv_depth_range).clamp(0.0, 1.0);
        (t * 65535.0) as u16
    }
}

/// Culls `count` instances packed into `batches` (`base_instance` added to output indices),
/// appending `(instance_index, quantized_depth_u16)` to `out_visible` and returning [`CullStats`].
///
/// Automatically dispatches to 8-wide AVX2 or 4-wide SSE2 on `x86_64`, 4-wide NEON on `aarch64`,
/// 4-wide `simd128` on `wasm32`, or the portable scalar implementation.
pub fn cull_frustum_soa(
    frustum: &Frustum,
    cfg: &CullConfig,
    batches: &[AabbSoa8],
    count: usize,
    out_visible: &mut Vec<(u32, u16)>,
) -> CullStats {
    cull_frustum_soa_offset(frustum, cfg, batches, count, 0, out_visible)
}

fn cull_frustum_soa_offset(
    frustum: &Frustum,
    cfg: &CullConfig,
    batches: &[AabbSoa8],
    count: usize,
    base_instance: u32,
    out_visible: &mut Vec<(u32, u16)>,
) -> CullStats {
    let clamped_count = count.min(batches.len() * 8);
    if clamped_count == 0 {
        return CullStats::default();
    }
    let params = DerivedCullParams::new(cfg);

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: Runtime CPUID check confirmed AVX2 support.
            return unsafe {
                cull_frustum_soa_avx2(
                    frustum,
                    &params,
                    batches,
                    clamped_count,
                    base_instance,
                    out_visible,
                )
            };
        }
        // SAFETY: SSE2 is baseline on all x86_64 targets.
        return unsafe {
            cull_frustum_soa_sse2(
                frustum,
                &params,
                batches,
                clamped_count,
                base_instance,
                out_visible,
            )
        };
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on all AArch64 targets.
        return unsafe {
            cull_frustum_soa_neon(
                frustum,
                &params,
                batches,
                clamped_count,
                base_instance,
                out_visible,
            )
        };
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return cull_frustum_soa_wasm_simd128(
            frustum,
            &params,
            batches,
            clamped_count,
            base_instance,
            out_visible,
        );
    }

    #[allow(unreachable_code)]
    cull_frustum_soa_scalar_inner(
        frustum,
        &params,
        batches,
        clamped_count,
        base_instance,
        out_visible,
    )
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn emit_batch_results(
    frustum_mask: u8,
    keep_mask: u8,
    valid_lanes: usize,
    base_idx: u32,
    near_dists: &[f32; 8],
    params: &DerivedCullParams,
    stats: &mut CullStats,
    out_visible: &mut Vec<(u32, u16)>,
) {
    let valid_mask = ((1u16 << valid_lanes) - 1) as u8;
    let f_mask = frustum_mask & valid_mask;
    let k_mask = f_mask & keep_mask;

    let f_pass = f_mask.count_ones();
    let k_pass = k_mask.count_ones();
    stats.total += valid_lanes as u32;
    stats.frustum_culled += (valid_lanes as u32) - f_pass;
    stats.distance_or_subpixel_culled += f_pass - k_pass;
    stats.visible += k_pass;

    let mut m = k_mask;
    while m != 0 {
        let lane = m.trailing_zeros() as usize;
        out_visible.push((
            base_idx + lane as u32,
            params.quantize_depth(near_dists[lane]),
        ));
        m &= m - 1;
    }
}

/// Portable scalar reference implementation of [`cull_frustum_soa`].
pub fn cull_frustum_soa_scalar(
    frustum: &Frustum,
    cfg: &CullConfig,
    batches: &[AabbSoa8],
    count: usize,
    out_visible: &mut Vec<(u32, u16)>,
) -> CullStats {
    let clamped = count.min(batches.len() * 8);
    if clamped == 0 {
        return CullStats::default();
    }
    let params = DerivedCullParams::new(cfg);
    cull_frustum_soa_scalar_inner(frustum, &params, batches, clamped, 0, out_visible)
}

fn cull_frustum_soa_scalar_inner(
    frustum: &Frustum,
    params: &DerivedCullParams,
    batches: &[AabbSoa8],
    count: usize,
    base_instance: u32,
    out_visible: &mut Vec<(u32, u16)>,
) -> CullStats {
    let mut stats = CullStats::default();
    let num_batches = count.div_ceil(8);
    for (bi, b) in batches[..num_batches].iter().enumerate() {
        let valid = (count - bi * 8).min(8);
        let mut frustum_mask = 0u8;
        let mut keep_mask = 0xffu8;
        let mut near_dists = [0.0f32; 8];

        for (lane, near_slot) in near_dists.iter_mut().enumerate().take(valid) {
            let (cx, cy, cz) = (b.cx[lane], b.cy[lane], b.cz[lane]);
            let (ex, ey, ez) = (b.ex[lane], b.ey[lane], b.ez[lane]);
            let mut inside = true;
            for (pi, p) in frustum.planes.iter().enumerate() {
                let s = cx * p.n[0] + cy * p.n[1] + cz * p.n[2] + p.d;
                if pi == 4 {
                    *near_slot = s;
                }
                let r = ex * p.abs_n[0] + ey * p.abs_n[1] + ez * p.abs_n[2];
                inside &= s + r >= 0.0;
            }
            if inside {
                frustum_mask |= 1 << lane;
            }
            if params.use_dist || params.use_subpx {
                let dx = cx - params.cam[0];
                let dy = cy - params.cam[1];
                let dz = cz - params.cam[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let mut ok = true;
                if params.use_dist {
                    let lim = params.max_dist + ex + ey + ez;
                    ok &= dist_sq <= lim * lim;
                }
                if params.use_subpx {
                    let r_sq = ex * ex + ey * ey + ez * ez;
                    ok &= r_sq * params.subpx_k_sq >= dist_sq;
                }
                if !ok {
                    keep_mask &= !(1 << lane);
                }
            }
        }

        emit_batch_results(
            frustum_mask,
            keep_mask,
            valid,
            base_instance + (bi * 8) as u32,
            &near_dists,
            params,
            &mut stats,
            out_visible,
        );
    }
    stats
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cull_frustum_soa_avx2(
    frustum: &Frustum,
    params: &DerivedCullParams,
    batches: &[AabbSoa8],
    count: usize,
    base_instance: u32,
    out_visible: &mut Vec<(u32, u16)>,
) -> CullStats {
    use std::arch::x86_64::*;

    let mut stats = CullStats::default();
    let zero = _mm256_setzero_ps();
    let cam_x = _mm256_set1_ps(params.cam[0]);
    let cam_y = _mm256_set1_ps(params.cam[1]);
    let cam_z = _mm256_set1_ps(params.cam[2]);
    let max_d = _mm256_set1_ps(params.max_dist);
    let subpx_k_sq = _mm256_set1_ps(params.subpx_k_sq);

    let planes: [(__m256, __m256, __m256, __m256, __m256, __m256, __m256); 6] =
        std::array::from_fn(|i| {
            let p = &frustum.planes[i];
            (
                _mm256_set1_ps(p.n[0]),
                _mm256_set1_ps(p.n[1]),
                _mm256_set1_ps(p.n[2]),
                _mm256_set1_ps(p.d),
                _mm256_set1_ps(p.abs_n[0]),
                _mm256_set1_ps(p.abs_n[1]),
                _mm256_set1_ps(p.abs_n[2]),
            )
        });

    let num_batches = count.div_ceil(8);
    let mut near_dists = [0.0f32; 8];

    for (bi, b) in batches[..num_batches].iter().enumerate() {
        let valid = (count - bi * 8).min(8);
        // SAFETY: `AabbSoa8` is `#[repr(C, align(32))]` with `[f32; 8]` fields.
        let cx = unsafe { _mm256_load_ps(b.cx.as_ptr()) };
        let cy = unsafe { _mm256_load_ps(b.cy.as_ptr()) };
        let cz = unsafe { _mm256_load_ps(b.cz.as_ptr()) };
        let ex = unsafe { _mm256_load_ps(b.ex.as_ptr()) };
        let ey = unsafe { _mm256_load_ps(b.ey.as_ptr()) };
        let ez = unsafe { _mm256_load_ps(b.ez.as_ptr()) };

        let mut vis = _mm256_cmp_ps(zero, zero, _CMP_EQ_OQ);
        for (pi, &(nx, ny, nz, pd, anx, any, anz)) in planes.iter().enumerate() {
            let s = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(cx, nx), _mm256_mul_ps(cy, ny)),
                _mm256_add_ps(_mm256_mul_ps(cz, nz), pd),
            );
            if pi == 4 {
                unsafe { _mm256_storeu_ps(near_dists.as_mut_ptr(), s) };
            }
            let r = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(ex, anx), _mm256_mul_ps(ey, any)),
                _mm256_mul_ps(ez, anz),
            );
            let ge = _mm256_cmp_ps(_mm256_add_ps(s, r), zero, _CMP_GE_OQ);
            vis = _mm256_and_ps(vis, ge);
        }
        let frustum_mask = _mm256_movemask_ps(vis) as u8;

        let mut keep_mask = 0xffu8;
        if frustum_mask != 0 && (params.use_dist || params.use_subpx) {
            let dx = _mm256_sub_ps(cx, cam_x);
            let dy = _mm256_sub_ps(cy, cam_y);
            let dz = _mm256_sub_ps(cz, cam_z);
            let dist_sq = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
                _mm256_mul_ps(dz, dz),
            );
            let mut keep = _mm256_cmp_ps(zero, zero, _CMP_EQ_OQ);
            if params.use_dist {
                let lim = _mm256_add_ps(_mm256_add_ps(max_d, ex), _mm256_add_ps(ey, ez));
                let lim_sq = _mm256_mul_ps(lim, lim);
                keep = _mm256_and_ps(keep, _mm256_cmp_ps(dist_sq, lim_sq, _CMP_LE_OQ));
            }
            if params.use_subpx {
                let r_sq = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(ex, ex), _mm256_mul_ps(ey, ey)),
                    _mm256_mul_ps(ez, ez),
                );
                let lhs = _mm256_mul_ps(r_sq, subpx_k_sq);
                keep = _mm256_and_ps(keep, _mm256_cmp_ps(lhs, dist_sq, _CMP_GE_OQ));
            }
            keep_mask = _mm256_movemask_ps(keep) as u8;
        }

        emit_batch_results(
            frustum_mask,
            keep_mask,
            valid,
            base_instance + (bi * 8) as u32,
            &near_dists,
            params,
            &mut stats,
            out_visible,
        );
    }
    stats
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn cull_frustum_soa_sse2(
    frustum: &Frustum,
    params: &DerivedCullParams,
    batches: &[AabbSoa8],
    count: usize,
    base_instance: u32,
    out_visible: &mut Vec<(u32, u16)>,
) -> CullStats {
    use std::arch::x86_64::*;

    let mut stats = CullStats::default();
    let zero = _mm_setzero_ps();
    let cam_x = _mm_set1_ps(params.cam[0]);
    let cam_y = _mm_set1_ps(params.cam[1]);
    let cam_z = _mm_set1_ps(params.cam[2]);
    let max_d = _mm_set1_ps(params.max_dist);
    let subpx_k_sq = _mm_set1_ps(params.subpx_k_sq);

    let planes: [(__m128, __m128, __m128, __m128, __m128, __m128, __m128); 6] =
        std::array::from_fn(|i| {
            let p = &frustum.planes[i];
            (
                _mm_set1_ps(p.n[0]),
                _mm_set1_ps(p.n[1]),
                _mm_set1_ps(p.n[2]),
                _mm_set1_ps(p.d),
                _mm_set1_ps(p.abs_n[0]),
                _mm_set1_ps(p.abs_n[1]),
                _mm_set1_ps(p.abs_n[2]),
            )
        });

    let num_batches = count.div_ceil(8);
    let mut near_dists = [0.0f32; 8];

    for (bi, b) in batches[..num_batches].iter().enumerate() {
        let valid = (count - bi * 8).min(8);
        let mut frustum_mask = 0u8;
        let mut keep_mask = 0xffu8;

        for half in 0..2 {
            let off = half * 4;
            let cx = unsafe { _mm_loadu_ps(b.cx.as_ptr().add(off)) };
            let cy = unsafe { _mm_loadu_ps(b.cy.as_ptr().add(off)) };
            let cz = unsafe { _mm_loadu_ps(b.cz.as_ptr().add(off)) };
            let ex = unsafe { _mm_loadu_ps(b.ex.as_ptr().add(off)) };
            let ey = unsafe { _mm_loadu_ps(b.ey.as_ptr().add(off)) };
            let ez = unsafe { _mm_loadu_ps(b.ez.as_ptr().add(off)) };

            let mut vis = _mm_cmpeq_ps(zero, zero);
            for (pi, &(nx, ny, nz, pd, anx, any, anz)) in planes.iter().enumerate() {
                let s = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(cx, nx), _mm_mul_ps(cy, ny)),
                    _mm_add_ps(_mm_mul_ps(cz, nz), pd),
                );
                if pi == 4 {
                    unsafe { _mm_storeu_ps(near_dists.as_mut_ptr().add(off), s) };
                }
                let r = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(ex, anx), _mm_mul_ps(ey, any)),
                    _mm_mul_ps(ez, anz),
                );
                vis = _mm_and_ps(vis, _mm_cmpge_ps(_mm_add_ps(s, r), zero));
            }
            let f_half = (_mm_movemask_ps(vis) as u8) & 0x0f;
            frustum_mask |= f_half << off;

            if f_half != 0 && (params.use_dist || params.use_subpx) {
                let dx = _mm_sub_ps(cx, cam_x);
                let dy = _mm_sub_ps(cy, cam_y);
                let dz = _mm_sub_ps(cz, cam_z);
                let dist_sq = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)),
                    _mm_mul_ps(dz, dz),
                );
                let mut keep = _mm_cmpeq_ps(zero, zero);
                if params.use_dist {
                    let lim = _mm_add_ps(_mm_add_ps(max_d, ex), _mm_add_ps(ey, ez));
                    keep = _mm_and_ps(keep, _mm_cmple_ps(dist_sq, _mm_mul_ps(lim, lim)));
                }
                if params.use_subpx {
                    let r_sq = _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(ex, ex), _mm_mul_ps(ey, ey)),
                        _mm_mul_ps(ez, ez),
                    );
                    keep = _mm_and_ps(keep, _mm_cmpge_ps(_mm_mul_ps(r_sq, subpx_k_sq), dist_sq));
                }
                let k_half = (_mm_movemask_ps(keep) as u8) & 0x0f;
                keep_mask = (keep_mask & !(0x0fu8 << off)) | (k_half << off);
            }
        }

        emit_batch_results(
            frustum_mask,
            keep_mask,
            valid,
            base_instance + (bi * 8) as u32,
            &near_dists,
            params,
            &mut stats,
            out_visible,
        );
    }
    stats
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn cull_frustum_soa_neon(
    frustum: &Frustum,
    params: &DerivedCullParams,
    batches: &[AabbSoa8],
    count: usize,
    base_instance: u32,
    out_visible: &mut Vec<(u32, u16)>,
) -> CullStats {
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn movemask_u32x4(v: uint32x4_t) -> u8 {
        let mut lanes = [0u32; 4];
        unsafe { vst1q_u32(lanes.as_mut_ptr(), v) };
        ((lanes[0] >> 31) as u8)
            | (((lanes[1] >> 31) as u8) << 1)
            | (((lanes[2] >> 31) as u8) << 2)
            | (((lanes[3] >> 31) as u8) << 3)
    }

    let mut stats = CullStats::default();
    let zero = vdupq_n_f32(0.0);
    let cam_x = vdupq_n_f32(params.cam[0]);
    let cam_y = vdupq_n_f32(params.cam[1]);
    let cam_z = vdupq_n_f32(params.cam[2]);
    let max_d = vdupq_n_f32(params.max_dist);
    let subpx_k_sq = vdupq_n_f32(params.subpx_k_sq);

    let planes: [(
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
    ); 6] = std::array::from_fn(|i| {
        let p = &frustum.planes[i];
        (
            vdupq_n_f32(p.n[0]),
            vdupq_n_f32(p.n[1]),
            vdupq_n_f32(p.n[2]),
            vdupq_n_f32(p.d),
            vdupq_n_f32(p.abs_n[0]),
            vdupq_n_f32(p.abs_n[1]),
            vdupq_n_f32(p.abs_n[2]),
        )
    });

    let num_batches = count.div_ceil(8);
    let mut near_dists = [0.0f32; 8];

    for (bi, b) in batches[..num_batches].iter().enumerate() {
        let valid = (count - bi * 8).min(8);
        let mut frustum_mask = 0u8;
        let mut keep_mask = 0xffu8;

        for half in 0..2 {
            let off = half * 4;
            let cx = unsafe { vld1q_f32(b.cx.as_ptr().add(off)) };
            let cy = unsafe { vld1q_f32(b.cy.as_ptr().add(off)) };
            let cz = unsafe { vld1q_f32(b.cz.as_ptr().add(off)) };
            let ex = unsafe { vld1q_f32(b.ex.as_ptr().add(off)) };
            let ey = unsafe { vld1q_f32(b.ey.as_ptr().add(off)) };
            let ez = unsafe { vld1q_f32(b.ez.as_ptr().add(off)) };

            let mut vis = vdupq_n_u32(u32::MAX);
            for (pi, &(nx, ny, nz, pd, anx, any, anz)) in planes.iter().enumerate() {
                let s = vaddq_f32(
                    vaddq_f32(vmulq_f32(cx, nx), vmulq_f32(cy, ny)),
                    vaddq_f32(vmulq_f32(cz, nz), pd),
                );
                if pi == 4 {
                    unsafe { vst1q_f32(near_dists.as_mut_ptr().add(off), s) };
                }
                let r = vaddq_f32(
                    vaddq_f32(vmulq_f32(ex, anx), vmulq_f32(ey, any)),
                    vmulq_f32(ez, anz),
                );
                vis = vandq_u32(vis, vcgeq_f32(vaddq_f32(s, r), zero));
            }
            let f_half = unsafe { movemask_u32x4(vis) };
            frustum_mask |= f_half << off;

            if f_half != 0 && (params.use_dist || params.use_subpx) {
                let dx = vsubq_f32(cx, cam_x);
                let dy = vsubq_f32(cy, cam_y);
                let dz = vsubq_f32(cz, cam_z);
                let dist_sq = vaddq_f32(
                    vaddq_f32(vmulq_f32(dx, dx), vmulq_f32(dy, dy)),
                    vmulq_f32(dz, dz),
                );
                let mut keep = vdupq_n_u32(u32::MAX);
                if params.use_dist {
                    let lim = vaddq_f32(vaddq_f32(max_d, ex), vaddq_f32(ey, ez));
                    keep = vandq_u32(keep, vcleq_f32(dist_sq, vmulq_f32(lim, lim)));
                }
                if params.use_subpx {
                    let r_sq = vaddq_f32(
                        vaddq_f32(vmulq_f32(ex, ex), vmulq_f32(ey, ey)),
                        vmulq_f32(ez, ez),
                    );
                    keep = vandq_u32(keep, vcgeq_f32(vmulq_f32(r_sq, subpx_k_sq), dist_sq));
                }
                let k_half = unsafe { movemask_u32x4(keep) };
                keep_mask = (keep_mask & !(0x0fu8 << off)) | (k_half << off);
            }
        }

        emit_batch_results(
            frustum_mask,
            keep_mask,
            valid,
            base_instance + (bi * 8) as u32,
            &near_dists,
            params,
            &mut stats,
            out_visible,
        );
    }
    stats
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn cull_frustum_soa_wasm_simd128(
    frustum: &Frustum,
    params: &DerivedCullParams,
    batches: &[AabbSoa8],
    count: usize,
    base_instance: u32,
    out_visible: &mut Vec<(u32, u16)>,
) -> CullStats {
    use core::arch::wasm32::*;

    let mut stats = CullStats::default();
    let zero = f32x4_splat(0.0);
    let cam_x = f32x4_splat(params.cam[0]);
    let cam_y = f32x4_splat(params.cam[1]);
    let cam_z = f32x4_splat(params.cam[2]);
    let max_d = f32x4_splat(params.max_dist);
    let subpx_k_sq = f32x4_splat(params.subpx_k_sq);

    let planes: [(v128, v128, v128, v128, v128, v128, v128); 6] = std::array::from_fn(|i| {
        let p = &frustum.planes[i];
        (
            f32x4_splat(p.n[0]),
            f32x4_splat(p.n[1]),
            f32x4_splat(p.n[2]),
            f32x4_splat(p.d),
            f32x4_splat(p.abs_n[0]),
            f32x4_splat(p.abs_n[1]),
            f32x4_splat(p.abs_n[2]),
        )
    });

    let num_batches = count.div_ceil(8);
    let mut near_dists = [0.0f32; 8];

    for (bi, b) in batches[..num_batches].iter().enumerate() {
        let valid = (count - bi * 8).min(8);
        let mut frustum_mask = 0u8;
        let mut keep_mask = 0xffu8;

        for half in 0..2 {
            let off = half * 4;
            // SAFETY: `off` is 0 or 4 inside `[f32; 8]`.
            let cx = unsafe { v128_load(b.cx.as_ptr().add(off).cast()) };
            let cy = unsafe { v128_load(b.cy.as_ptr().add(off).cast()) };
            let cz = unsafe { v128_load(b.cz.as_ptr().add(off).cast()) };
            let ex = unsafe { v128_load(b.ex.as_ptr().add(off).cast()) };
            let ey = unsafe { v128_load(b.ey.as_ptr().add(off).cast()) };
            let ez = unsafe { v128_load(b.ez.as_ptr().add(off).cast()) };

            let mut vis = i32x4_splat(-1);
            for (pi, &(nx, ny, nz, pd, anx, any, anz)) in planes.iter().enumerate() {
                let s = f32x4_add(
                    f32x4_add(f32x4_mul(cx, nx), f32x4_mul(cy, ny)),
                    f32x4_add(f32x4_mul(cz, nz), pd),
                );
                if pi == 4 {
                    unsafe { v128_store(near_dists.as_mut_ptr().add(off).cast(), s) };
                }
                let r = f32x4_add(
                    f32x4_add(f32x4_mul(ex, anx), f32x4_mul(ey, any)),
                    f32x4_mul(ez, anz),
                );
                vis = v128_and(vis, f32x4_ge(f32x4_add(s, r), zero));
            }
            let f_half = (i32x4_bitmask(vis)) & 0x0f;
            frustum_mask |= f_half << off;

            if f_half != 0 && (params.use_dist || params.use_subpx) {
                let dx = f32x4_sub(cx, cam_x);
                let dy = f32x4_sub(cy, cam_y);
                let dz = f32x4_sub(cz, cam_z);
                let dist_sq = f32x4_add(
                    f32x4_add(f32x4_mul(dx, dx), f32x4_mul(dy, dy)),
                    f32x4_mul(dz, dz),
                );
                let mut keep = i32x4_splat(-1);
                if params.use_dist {
                    let lim = f32x4_add(f32x4_add(max_d, ex), f32x4_add(ey, ez));
                    keep = v128_and(keep, f32x4_le(dist_sq, f32x4_mul(lim, lim)));
                }
                if params.use_subpx {
                    let r_sq = f32x4_add(
                        f32x4_add(f32x4_mul(ex, ex), f32x4_mul(ey, ey)),
                        f32x4_mul(ez, ez),
                    );
                    keep = v128_and(keep, f32x4_ge(f32x4_mul(r_sq, subpx_k_sq), dist_sq));
                }
                let k_half = i32x4_bitmask(keep) & 0x0f;
                keep_mask = (keep_mask & !(0x0fu8 << off)) | (k_half << off);
            }
        }

        emit_batch_results(
            frustum_mask,
            keep_mask,
            valid,
            base_instance + (bi * 8) as u32,
            &near_dists,
            params,
            &mut stats,
            out_visible,
        );
    }
    stats
}

/// Zero-allocation steady-state parallel SIMD [`cull_frustum_soa`] across `pool`, reusing
/// `chunk_scratch` and writing deterministically ordered results into `out_visible`.
pub fn par_cull_frustum_into(
    pool: &JobPool,
    frustum: &Frustum,
    cfg: &CullConfig,
    batches: &[AabbSoa8],
    count: usize,
    chunk_scratch: &mut Vec<(Vec<(u32, u16)>, CullStats)>,
    out_visible: &mut Vec<(u32, u16)>,
) -> CullStats {
    out_visible.clear();
    let clamped_count = count.min(batches.len() * 8);
    if clamped_count == 0 {
        return CullStats::default();
    }
    let num_batches = clamped_count.div_ceil(8);
    let min_batches_per_chunk = 256usize; // 2,048 AABBs minimum per parallel job
    let max_workers = pool.worker_count().min(8);
    if max_workers <= 1 || num_batches < min_batches_per_chunk * 2 {
        return cull_frustum_soa_offset(frustum, cfg, batches, clamped_count, 0, out_visible);
    }

    let num_chunks = (num_batches / min_batches_per_chunk).clamp(2, max_workers);
    let chunk_batches = num_batches.div_ceil(num_chunks);
    if chunk_scratch.len() < num_chunks {
        chunk_scratch.resize_with(num_chunks, || {
            (Vec::with_capacity(chunk_batches * 4), CullStats::default())
        });
    }

    struct SyncSlicePtr<T>(*mut T);
    impl<T> SyncSlicePtr<T> {
        #[inline]
        fn ptr(&self) -> *mut T {
            self.0
        }
    }
    // SAFETY: Each `ci` in `0..num_chunks` accesses its own disjoint `chunk_scratch[ci]`.
    unsafe impl<T: Send> Sync for SyncSlicePtr<T> {}
    unsafe impl<T: Send> Send for SyncSlicePtr<T> {}

    let scratch_ptr = SyncSlicePtr(chunk_scratch.as_mut_ptr());
    pool.par_for_chunks(num_chunks, 1, |_, range| {
        let base = scratch_ptr.ptr();
        for ci in range {
            let b_start = ci * chunk_batches;
            let b_end = (b_start + chunk_batches).min(num_batches);
            let elem_start = b_start * 8;
            let elem_count = (clamped_count - elem_start).min((b_end - b_start) * 8);
            // SAFETY: `ci` is uniquely owned by this `range` slice.
            let slot = unsafe { &mut *base.add(ci) };
            slot.0.clear();
            slot.1 = if b_start < b_end && elem_count > 0 {
                cull_frustum_soa_offset(
                    frustum,
                    cfg,
                    &batches[b_start..b_end],
                    elem_count,
                    elem_start as u32,
                    &mut slot.0,
                )
            } else {
                CullStats::default()
            };
        }
    });

    let mut stats = CullStats::default();
    let total_vis: usize = chunk_scratch[..num_chunks]
        .iter()
        .map(|(v, _)| v.len())
        .sum();
    out_visible.reserve(total_vis);
    for (v, s) in &chunk_scratch[..num_chunks] {
        out_visible.extend_from_slice(v);
        stats.merge(*s);
    }
    stats
}

/// Runs SIMD [`cull_frustum_soa`] in parallel across `pool`, preserving deterministic ascending
/// instance order prior to optional Early-Z depth sorting.
pub fn par_cull_frustum(
    pool: &JobPool,
    frustum: &Frustum,
    cfg: &CullConfig,
    batches: &[AabbSoa8],
    count: usize,
) -> (Vec<(u32, u16)>, CullStats) {
    let mut scratch = Vec::new();
    let mut out = Vec::new();
    let stats = par_cull_frustum_into(pool, frustum, cfg, batches, count, &mut scratch, &mut out);
    (out, stats)
}

/// Stable linear-time `O(N)` 2-pass 8-bit LSD counting radix sort on 16-bit quantized depth keys,
/// ordering visible opaque draws front-to-back so GPU Early-Z rejects occluded fragments before
/// running expensive PBR pixel shaders.
pub fn sort_front_to_back_radix(items: &mut [(u32, u16)], scratch: &mut Vec<(u32, u16)>) {
    let n = items.len();
    if n <= 1 {
        return;
    }
    if n <= 32 {
        // Tiny arrays fit in L1; insertion/slice sort avoids histogram setup.
        items.sort_by_key(|&(_, d)| d);
        return;
    }

    scratch.resize(n, (0, 0));

    // Single histogram scan computes both low-byte and high-byte frequency tables at once.
    let mut hist_lo = [0u32; 256];
    let mut hist_hi = [0u32; 256];
    for &(_, key) in items.iter() {
        hist_lo[(key & 0xff) as usize] += 1;
        hist_hi[(key >> 8) as usize] += 1;
    }

    let mut pos_lo = [0usize; 256];
    let mut pos_hi = [0usize; 256];
    let (mut acc_lo, mut acc_hi) = (0usize, 0usize);
    for i in 0..256 {
        pos_lo[i] = acc_lo;
        acc_lo += hist_lo[i] as usize;
        pos_hi[i] = acc_hi;
        acc_hi += hist_hi[i] as usize;
    }

    // Pass 1: scatter `items` -> `scratch` by low 8 bits.
    for &item in items.iter() {
        let b = (item.1 & 0xff) as usize;
        scratch[pos_lo[b]] = item;
        pos_lo[b] += 1;
    }

    // Pass 2: scatter `scratch` -> `items` by high 8 bits.
    for &item in scratch[..n].iter() {
        let b = (item.1 >> 8) as usize;
        items[pos_hi[b]] = item;
        pos_hi[b] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perspective_zero_to_one(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
        let f = 1.0 / (0.5 * fov_y_rad).tan();
        let nf = 1.0 / (near - far);
        // Column-major right-handed perspective looking down -Z with [0, 1] depth:
        [
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            far * nf,
            -1.0,
            0.0,
            0.0,
            far * near * nf,
            0.0,
        ]
    }

    #[test]
    fn arvo_transform_aabb_matches_eight_corner_reference() {
        let local = Aabb::new([-1.5, -0.4, 0.2], [2.3, 1.8, 3.1]);
        // Non-uniform scale + rotation + translation (column-major).
        let c = 0.6f32.cos();
        let s = 0.6f32.sin();
        let m: [f32; 16] = [
            2.0 * c,
            2.0 * s,
            0.0,
            0.0,
            -1.5 * s,
            1.5 * c,
            0.5,
            0.0,
            0.3,
            -0.7,
            1.2,
            0.0,
            12.5,
            -4.2,
            8.0,
            1.0,
        ];
        let got = transform_aabb(&local, &m);

        let mut ref_min = [f32::INFINITY; 3];
        let mut ref_max = [f32::NEG_INFINITY; 3];
        for mask in 0..8 {
            let x = if mask & 1 == 0 {
                local.min[0]
            } else {
                local.max[0]
            };
            let y = if mask & 2 == 0 {
                local.min[1]
            } else {
                local.max[1]
            };
            let z = if mask & 4 == 0 {
                local.min[2]
            } else {
                local.max[2]
            };
            for r in 0..3 {
                let p = m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r];
                ref_min[r] = ref_min[r].min(p);
                ref_max[r] = ref_max[r].max(p);
            }
        }
        for i in 0..3 {
            assert!((got.min[i] - ref_min[i]).abs() < 1e-5);
            assert!((got.max[i] - ref_max[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn gribb_hartmann_culls_outside_boxes_and_keeps_inside_boxes() {
        let vp = perspective_zero_to_one(std::f32::consts::FRAC_PI_3, 16.0 / 9.0, 0.5, 100.0);
        let frustum = Frustum::from_view_proj_zero_to_one(&vp);

        // Centered in front of camera at z = -10.
        assert!(frustum.contains_aabb(&Aabb::new([-0.5, -0.5, -10.5], [0.5, 0.5, -9.5])));
        // Behind the camera at z = +5.
        assert!(!frustum.contains_aabb(&Aabb::new([-0.5, -0.5, 4.5], [0.5, 0.5, 5.5])));
        // Beyond the far plane at z = -150.
        assert!(!frustum.contains_aabb(&Aabb::new([-1.0, -1.0, -155.0], [1.0, 1.0, -145.0])));
        // Far to the left at x = -80, z = -10.
        assert!(!frustum.contains_aabb(&Aabb::new([-82.0, -1.0, -11.0], [-78.0, 1.0, -9.0])));
    }

    #[test]
    fn simd_culling_matches_scalar_reference_bit_for_bit() {
        let vp = perspective_zero_to_one(std::f32::consts::FRAC_PI_3, 16.0 / 9.0, 0.25, 250.0);
        let frustum = Frustum::from_view_proj_zero_to_one(&vp);
        let cfg = CullConfig {
            camera_pos: [0.0, 0.0, 0.0],
            max_distance: 160.0,
            min_screen_px: 2.0,
            proj_scale_y: vp[5],
            viewport_height_px: 1080.0,
            depth_range: 250.0,
        };

        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next_f32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32) / ((1u32 << 24) as f32)
        };

        let count = 4099; // Deliberately non-multiple of 8 to test tail-lane masking.
        let mut aabbs = Vec::with_capacity(count);
        for _ in 0..count {
            let cx = (next_f32() - 0.5) * 200.0;
            let cy = (next_f32() - 0.5) * 120.0;
            let cz = -next_f32() * 280.0 + 20.0;
            let e = 0.02 + next_f32() * 3.0;
            aabbs.push(Aabb::new(
                [cx - e, cy - e, cz - e],
                [cx + e, cy + e, cz + e],
            ));
        }

        let mut soa = Vec::new();
        pack_aabbs_soa(&aabbs, &mut soa);

        let mut simd_out = Vec::new();
        let simd_stats = cull_frustum_soa(&frustum, &cfg, &soa, count, &mut simd_out);

        let mut scalar_out = Vec::new();
        let scalar_stats = cull_frustum_soa_scalar(&frustum, &cfg, &soa, count, &mut scalar_out);

        assert_eq!(simd_stats, scalar_stats);
        assert_eq!(simd_out, scalar_out);
        assert!(simd_stats.visible > 100 && simd_stats.frustum_culled > 1000);
        assert!(simd_stats.distance_or_subpixel_culled > 50);
    }

    #[test]
    fn parallel_cull_and_radix_sort_produce_exact_front_to_back_order() {
        let vp = perspective_zero_to_one(1.0, 1.5, 0.1, 500.0);
        let frustum = Frustum::from_view_proj_zero_to_one(&vp);
        let cfg = CullConfig {
            max_distance: 300.0,
            depth_range: 300.0,
            ..CullConfig::default()
        };

        let count = 8195;
        let aabbs: Vec<Aabb> = (0..count)
            .map(|i| {
                let z = -1.0 - ((i * 73) % 280) as f32;
                let x = (((i * 19) % 40) as f32) - 20.0;
                Aabb::new([x - 0.5, -0.5, z - 0.5], [x + 0.5, 0.5, z + 0.5])
            })
            .collect();
        let mut soa = Vec::new();
        pack_aabbs_soa(&aabbs, &mut soa);

        let pool = JobPool::new(4);
        let (mut par_vis, par_stats) = par_cull_frustum(&pool, &frustum, &cfg, &soa, count);

        let mut seq_vis = Vec::new();
        let seq_stats = cull_frustum_soa(&frustum, &cfg, &soa, count, &mut seq_vis);

        assert_eq!(par_stats, seq_stats);
        assert_eq!(par_vis, seq_vis);

        let mut expected_sorted = seq_vis.clone();
        expected_sorted.sort_by_key(|&(_, d)| d); // Rust `sort_by_key` is stable.

        let mut scratch = Vec::new();
        sort_front_to_back_radix(&mut par_vis, &mut scratch);
        assert_eq!(par_vis, expected_sorted);
    }
}
