//! Stereo XR camera views, 120 Hz QoS frame-budget governor, cooked-world edits, screen
//! projection, and screen/3D-ray picking.

use super::{floats, matrix, with, write_floats};
use ganbate_engine::xr::{
    DEFAULT_IPD_METRES, FrameBudgetGovernor, QualityBudget, clamp_ipd_metres, compute_stereo_views,
};

fn governor_snapshot_floats(gov: &FrameBudgetGovernor, budget: QualityBudget) -> [f32; 6] {
    [
        budget.resolution_scale,
        budget.max_dynamic_lights as f32,
        if budget.ambient_particles { 1.0 } else { 0.0 },
        f32::from(budget.shadow_tier),
        f32::from(gov.tier()),
        gov.rolling_mean_ms(),
    ]
}

/// Configures stereo XR eye selection for subsequent `ganbate_rt_render`, `ganbate_rt_project`,
/// and `ganbate_rt_pick*` calls:
/// - `eye = 0`: mono (`rt.view` directly)
/// - `eye = -1`: left eye (`compute_stereo_views(&rt.view, ipd_metres).left`)
/// - `eye = 1`: right eye (`compute_stereo_views(&rt.view, ipd_metres).right`)
///
/// `ipd_metres <= 0.0` or non-finite defaults to `DEFAULT_IPD_METRES` (`0.063m`); otherwise it is
/// clamped to the realistic human range `0.050..=0.078`.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_set_stereo_eye(eye: i32, ipd_metres: f32) -> u32 {
    with(0, |rt| {
        if !(-1..=1).contains(&eye) {
            return Err(format!(
                "set_stereo_eye: eye must be -1, 0, or 1, got {eye}"
            ));
        }
        let ipd = if !ipd_metres.is_finite() || ipd_metres <= 0.0 {
            DEFAULT_IPD_METRES
        } else {
            clamp_ipd_metres(ipd_metres)
        };
        rt.stereo_eye = eye;
        rt.stereo_ipd_metres = ipd;
        Ok(1)
    })
}

/// Computes the parallel-axis stereo [`View`] pair for the current mono camera and IPD, writing
/// 16 little-endian floats at `out`:
/// `[left.eye(3), left.target(3), right.eye(3), right.target(3), right_axis(3), ipd_metres]`.
///
/// # Safety
/// `out` must address 64 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_stereo_views(out: *mut u8) -> u32 {
    with(0, |rt| {
        if out.is_null() {
            return Err("stereo_views: null output buffer".into());
        }
        let pair = compute_stereo_views(&rt.view, rt.stereo_ipd_metres);
        let mut v = [0.0f32; 16];
        v[0..3].copy_from_slice(&pair.left.eye);
        v[3..6].copy_from_slice(&pair.left.target);
        v[6..9].copy_from_slice(&pair.right.eye);
        v[9..12].copy_from_slice(&pair.right.target);
        v[12..15].copy_from_slice(&pair.right_axis);
        v[15] = pair.ipd_metres;
        // SAFETY: the caller passes 64 writable bytes.
        unsafe { write_floats(out, &v) };
        Ok(1)
    })
}

/// Records a frame duration (`dt_ms`, milliseconds) into the `120Hz` (`8.333ms`)
/// [`FrameBudgetGovernor`] and, when `out` is non-null, writes 6 floats:
/// `[resolution_scale, max_dynamic_lights, ambient_particles, shadow_tier, tier, rolling_mean_ms]`.
///
/// # Safety
/// `out` must be null or address 24 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_governor_record(dt_ms: f32, out: *mut u8) -> u32 {
    with(0, |rt| {
        let budget = rt.frame_governor.record_frame_ms(dt_ms);
        if !out.is_null() {
            let snap = governor_snapshot_floats(&rt.frame_governor, budget);
            // SAFETY: when non-null, the caller passes 24 writable bytes.
            unsafe { write_floats(out, &snap) };
        }
        Ok(1)
    })
}

/// Queries the current `120Hz` [`FrameBudgetGovernor`] state without recording a sample, writing
/// 6 floats at `out`:
/// `[resolution_scale, max_dynamic_lights, ambient_particles, shadow_tier, tier, rolling_mean_ms]`.
///
/// # Safety
/// `out` must address 24 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_governor_query(out: *mut u8) -> u32 {
    with(0, |rt| {
        if out.is_null() {
            return Err("governor_query: null output buffer".into());
        }
        let snap = governor_snapshot_floats(&rt.frame_governor, rt.frame_governor.budget());
        // SAFETY: the caller passes 24 writable bytes.
        unsafe { write_floats(out, &snap) };
        Ok(1)
    })
}

/// Resets the `120Hz` [`FrameBudgetGovernor`] rolling window and restores top quality tier (`3`).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_governor_reset() -> u32 {
    with(0, |rt| {
        rt.frame_governor.reset();
        Ok(1)
    })
}

/// Where the world point (`x`, `y`, `z`) lands in the image: writes pixel x, pixel y (device
/// pixels, top-left origin) and the depth in metres as 3 floats at `out`. Returns 0 (without
/// logging) when the point is behind the camera.
///
/// # Safety
/// `out` must address 12 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_project(x: f32, y: f32, z: f32, out: *mut u8) -> u32 {
    with(0, |rt| {
        let size = rt.renderer.size();
        let active_view = rt.active_view();
        let Some(p) = rt.renderer.project(&active_view, size, [x, y, z]) else {
            return Ok(0);
        };
        // SAFETY: the caller passes 12 writable bytes.
        unsafe { write_floats(out, &p) };
        Ok(1)
    })
}

/// How many instances the cooked world has.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_world_instances() -> u32 {
    with(0, |rt| Ok(rt.renderer.world_instances() as u32))
}

/// Writes world instance `index`'s matrix (16 floats, column-major) at `out`; returns 1 when it
/// is shown, 2 when hidden, 0 (after logging) when there is no such instance.
///
/// # Safety
/// `out` must address 64 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_world_instance(index: u32, out: *mut u8) -> u32 {
    with(0, |rt| {
        let (m, visible) = rt
            .renderer
            .world_instance(index as usize)
            .ok_or_else(|| format!("world_instance: no world instance {index}"))?;
        // SAFETY: the caller passes 64 writable bytes.
        unsafe { write_floats(out, &m) };
        Ok(if visible { 1 } else { 2 })
    })
}

/// Moves world instance `index` to the matrix at `matrix` (16 floats) and shows (`visible` 1)
/// or hides (0) it. Returns 1, or 0 after logging why not.
///
/// # Safety
/// `matrix` must address 64 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_set_world_instance(
    index: u32,
    matrix_at: *const u8,
    visible: u32,
) -> u32 {
    // SAFETY: the caller passes 16 readable floats.
    let m = unsafe { floats(matrix_at, 16) };
    with(0, |rt| {
        let m = matrix(&m)?;
        rt.renderer
            .set_world_instance(&mut rt.device, index as usize, m, visible != 0)?;
        Ok(1)
    })
}

/// Moves world instances to new matrices, keeping whether each is shown (what physics does after
/// a step). Returns 1, or 0 after logging why not.
pub(crate) fn move_world_instances(moves: &[(u32, [f32; 16])]) -> u32 {
    with(0, |rt| {
        for &(index, m) in moves {
            let visible = rt
                .renderer
                .world_instance(index as usize)
                .is_none_or(|(_, v)| v);
            rt.renderer
                .set_world_instance(&mut rt.device, index as usize, m, visible)?;
        }
        Ok(1)
    })
}

/// Moves world lamp `index` to (`x`, `y`, `z`). Returns 1, or 0 after logging why not.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_set_world_light(index: u32, x: f32, y: f32, z: f32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .set_world_light(&mut rt.device, index as usize, [x, y, z])?;
        Ok(1)
    })
}

/// The cooked world's instance under device pixel (`x`, `y`): returns its index plus one and
/// writes the distance in metres and the world point (4 floats) at `out`; returns 0 when no
/// instance is there.
///
/// # Safety
/// `out` must address 16 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_pick_world(x: f32, y: f32, out: *mut u8) -> u32 {
    with(0, |rt| {
        let size = rt.renderer.size();
        let active_view = rt.active_view();
        let Some(hit) = rt.renderer.pick_world(&active_view, size, (x, y)) else {
            return Ok(0);
        };
        let p = hit.point;
        // SAFETY: the caller passes 16 writable bytes.
        unsafe { write_floats(out, &[hit.distance, p[0], p[1], p[2]]) };
        Ok(hit.instance as u32 + 1)
    })
}

/// The visible prop under device pixel (`x`, `y`) on one of the pick `layers` (as
/// `Renderer::pick` chooses it): returns its instance and writes the distance in metres and
/// the world point (4 floats) at `out`; returns 0 (without logging) when no prop is there.
///
/// # Safety
/// `out` must address 16 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_pick(x: f32, y: f32, layers: u32, out: *mut u8) -> u32 {
    with(0, |rt| {
        let size = rt.renderer.size();
        let active_view = rt.active_view();
        let Some(hit) = rt.renderer.pick(&active_view, size, (x, y), layers) else {
            return Ok(0);
        };
        let p = hit.point;
        // SAFETY: the caller passes 16 writable bytes.
        unsafe { write_floats(out, &[hit.distance, p[0], p[1], p[2]]) };
        Ok(hit.instance.raw())
    })
}

/// The visible prop along world-space ray `(ox, oy, oz)` -> `(dx, dy, dz)` on one of the pick
/// `layers` (as `Renderer::pick_along_ray` chooses it): returns its instance and writes the
/// distance in metres and the world point (4 floats) at `out`; returns 0 (without logging) when
/// no prop is there or the ray is degenerate.
///
/// # Safety
/// `out` must address 16 writable bytes when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_pick_ray(
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    layers: u32,
    out: *mut u8,
) -> u32 {
    with(0, |rt| {
        if out.is_null() {
            return Ok(0);
        }
        let Some(hit) = rt
            .renderer
            .pick_along_ray([ox, oy, oz], [dx, dy, dz], layers)
        else {
            return Ok(0);
        };
        let p = hit.point;
        // SAFETY: the caller passes 16 writable bytes.
        unsafe { write_floats(out, &[hit.distance, p[0], p[1], p[2]]) };
        Ok(hit.instance.raw())
    })
}

/// The cooked world's instance along world-space ray `(ox, oy, oz)` -> `(dx, dy, dz)`: returns
/// its index plus one and writes the distance in metres and the world point (4 floats) at `out`;
/// returns 0 when no instance is there or the ray is degenerate.
///
/// # Safety
/// `out` must address 16 writable bytes when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_pick_world_ray(
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    out: *mut u8,
) -> u32 {
    with(0, |rt| {
        if out.is_null() {
            return Ok(0);
        }
        let Some(hit) = rt.renderer.pick_world_along_ray([ox, oy, oz], [dx, dy, dz]) else {
            return Ok(0);
        };
        let p = hit.point;
        // SAFETY: the caller passes 16 writable bytes.
        unsafe { write_floats(out, &[hit.distance, p[0], p[1], p[2]]) };
        Ok(hit.instance as u32 + 1)
    })
}
