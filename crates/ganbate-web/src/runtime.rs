//! Library mode: a host page (a game with its own interface) drives the renderer instead of
//! the player's fly camera. `web/ganbate.js` wraps these exports as `createEngine`.
//!
//! The host loads a cooked world (or none: a scene of props only) and a Look, then each frame
//! sets the camera, the clock and the weather, moves its props and calls [`ganbate_rt_render`].
//! Props (meshes, textures, materials and placed instances) are the engine's
//! `Renderer::create_prop_*` objects; their handles are never 0, so 0 always means failure, and
//! the reason has been sent to the host's `log` import at level 2.
//!
//! Memory: buffers the host passes are read during the call only. Floats and words are read
//! byte by byte (little-endian), so pointers need no alignment. A world handed to
//! [`ganbate_rt_start`] comes from `ganbate_alloc` and becomes the engine's.

use crate::{give, install_panic_hook, say};
use ganbate_engine::anim::{
    BoneOffset, ClipLayer, LookAtChain, MorphOverride, Pose, PoseSpec, TwoBoneIk, normalize_q,
    pose_model_clip,
};
use ganbate_engine::assets::model::{self, Model};
use ganbate_engine::gpu::Device;
use ganbate_engine::render::{
    Environment, Focus, FrameStats, LookFile, MaterialDesc, MeshData, Output, PROP_SHADER_PARAMS,
    PropInstance, PropMaterial, PropMesh, PropTexture, Quality, Renderer, TimeOfDay, View, Weather,
};
use ganbate_engine::render::{GameLight, LightDesc, ModelId, ModelInstance, SpotCone};
use ganbate_engine::world::World;
use ganbate_engine::xr::{
    DEFAULT_IPD_METRES, FrameBudgetGovernor, QualityBudget, clamp_ipd_metres, compute_stereo_views,
};
use std::cell::RefCell;
use std::collections::HashMap;

struct Runtime {
    device: Device<'static>,
    renderer: Renderer,
    view: View,
    /// Stereo eye selector (`0` = mono, `-1` = left eye, `+1` = right eye).
    stereo_eye: i32,
    /// Interpupillary distance in metres (`0.050..=0.078`, default `0.063`).
    stereo_ipd_metres: f32,
    /// Deterministic `120Hz` (`8.333ms`) rolling-window frame budget governor.
    frame_governor: FrameBudgetGovernor,
    time: TimeOfDay,
    weather: Weather,
    seconds: f32,
    /// The world's lamps, brighter (or dimmer) than the time of day lights them.
    lamp_boost: f32,
    frames: u32,
    /// Parsed models waiting for their textures (handle = index + 1).
    pending: Vec<Option<Model>>,
    /// Snapshotted poses per model instance for zero-copy in-Rust cross-fading.
    crossfade_from: HashMap<u32, Pose>,
}

impl Runtime {
    fn active_view(&self) -> View {
        match self.stereo_eye {
            -1 => compute_stereo_views(&self.view, self.stereo_ipd_metres).left,
            1 => compute_stereo_views(&self.view, self.stereo_ipd_metres).right,
            _ => self.view.clone(),
        }
    }
}

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

fn quality_from_code(code: u32) -> Quality {
    match code {
        0 => Quality::realtime(),
        1 => Quality::still(),
        2 => Quality::hybrid_stage(),
        3 => Quality::low_cost(),
        _ => Quality::realtime(),
    }
}

fn frame_stats_words(stats: FrameStats) -> [u32; 9] {
    [
        stats.world_draws_total,
        stats.world_draws_submitted,
        stats.world_draws_culled,
        u32::from(stats.shadow_static_cached),
        stats.shadow_world_draws,
        u32::from(stats.ground_cached),
        u32::from(stats.reflection_drawn),
        u32::from(stats.ssao_drawn),
        u32::from(stats.fused_tonemap),
    ]
}

thread_local! {
    static RT: RefCell<Option<Runtime>> = const { RefCell::new(None) };
}

/// Runs `f` on the runtime; errors (and a missing runtime) are logged and give `fail`.
fn with<T>(fail: T, f: impl FnOnce(&mut Runtime) -> Result<T, String>) -> T {
    RT.with(|cell| match cell.borrow_mut().as_mut() {
        None => {
            say(2, "the engine is not started (ganbate_rt_start)");
            fail
        }
        Some(rt) => match f(rt) {
            Ok(v) => v,
            Err(e) => {
                say(2, &e);
                fail
            }
        },
    })
}

/// `len` bytes at `ptr`, or nothing for a zero length.
///
/// # Safety
/// `ptr` must address `len` readable bytes for the duration of the call.
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        // SAFETY: the caller guarantees the range is readable.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// `count` little-endian f32 at `ptr` (any alignment).
///
/// # Safety
/// As [`bytes`] for `count * 4` bytes.
unsafe fn floats(ptr: *const u8, count: usize) -> Vec<f32> {
    // SAFETY: forwarded.
    unsafe { bytes(ptr, count * 4) }
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// `count` little-endian u32 at `ptr` (any alignment).
///
/// # Safety
/// As [`bytes`] for `count * 4` bytes.
unsafe fn words(ptr: *const u8, count: usize) -> Vec<u32> {
    // SAFETY: forwarded.
    unsafe { bytes(ptr, count * 4) }
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn triples(v: &[f32]) -> Vec<[f32; 3]> {
    v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn matrix(v: &[f32]) -> Result<[f32; 16], String> {
    let m: [f32; 16] = v.try_into().map_err(|_| "a matrix needs 16 floats")?;
    if m.iter().all(|x| x.is_finite()) {
        Ok(m)
    } else {
        Err("the matrix is not finite".into())
    }
}

/// Starts the engine on the page's canvas with the `.gbew` world at `world` (ownership passes
/// from `ganbate_alloc`; a zero length takes the world `ganbate_load_world` decoded, whose images the
/// host may have decoded with `ganbate_world_set_image`, or none) and the Look file at `look`.
/// `quality` selects the rendering preset (`0` = realtime, `1` = still, `2` = hybrid stage,
/// `3` = low-cost fast-path; unknown codes fall back to realtime). Returns 1, or 0 after
/// logging why not.
///
/// # Safety
/// `world` must come from `ganbate_alloc(world_len)` fully written (or be null with a zero length);
/// `look` must address `look_len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_start(
    world: *mut u8,
    world_len: usize,
    look: *const u8,
    look_len: usize,
    quality: u32,
) -> u32 {
    install_panic_hook();
    let world = if world_len == 0 || world.is_null() {
        Ok(crate::WORLD
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default())
    } else {
        // SAFETY: the caller hands over an initialized allocation of `world_len` bytes.
        let owned = unsafe { Vec::from_raw_parts(world, world_len, world_len) };
        ganbate_engine::assets::cooked::decode_scene(&owned)
            .map(|(w, table)| {
                crate::ENTITIES.with(|cell| *cell.borrow_mut() = crate::entities_json(&w, &table));
                crate::physics::prepare(&w, &table);
                w
            })
            .map_err(|e| format!("world: {e}"))
    };
    // SAFETY: the caller passes a readable range.
    let look = unsafe { bytes(look, look_len) };
    match world.and_then(|w| start(&w, look, quality)) {
        Ok(rt) => {
            say(
                0,
                &format!(
                    "Ganbate engine | {} | {} triangles, {} lamps",
                    rt.device.name(),
                    rt.renderer.triangles(),
                    rt.renderer.lights()
                ),
            );
            RT.with(|cell| *cell.borrow_mut() = Some(rt));
            1
        }
        Err(e) => {
            say(2, &e);
            0
        }
    }
}

fn start(world: &World, look: &[u8], quality: u32) -> Result<Runtime, String> {
    let file = if look.is_empty() {
        LookFile::default()
    } else {
        LookFile::parse(look).map_err(|e| format!("Look file: {e}"))?
    };
    let mut device = Device::for_canvas()?;
    let quality = quality_from_code(quality);
    let (w, h) = device.surface_size();
    let format = device.surface_format();
    let renderer = Renderer::new(
        &mut device,
        world,
        file.look.clone(),
        quality,
        (w.max(1), h.max(1)),
        Output::Surface(format),
    )?;
    Ok(Runtime {
        device,
        renderer,
        view: file.views.first().cloned().unwrap_or_default(),
        stereo_eye: 0,
        stereo_ipd_metres: DEFAULT_IPD_METRES,
        frame_governor: FrameBudgetGovernor::new(),
        time: file.time,
        weather: file.weather,
        seconds: 0.0,
        lamp_boost: 1.0,
        frames: 0,
        pending: vec![],
        crossfade_from: HashMap::new(),
    })
}

/// Stops the engine and releases its GPU objects. Returns 1.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_stop() -> u32 {
    RT.with(|cell| {
        if let Some(mut rt) = cell.borrow_mut().take() {
            rt.renderer.destroy_props(&mut rt.device);
        }
    });
    1
}

/// Replaces the Look (the file's look block; its views, time and weather are ignored).
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_set_look(ptr: *const u8, len: usize) -> u32 {
    // SAFETY: the caller passes a readable range.
    let text = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        let file = LookFile::parse(text).map_err(|e| format!("Look file: {e}"))?;
        rt.renderer.set_look(file.look);
        Ok(1)
    })
}

/// Sets the camera from 17 floats at `ptr`: eye (3), target (3), up (3), vertical field of view
/// in degrees, lens shift (2, normalized device units, y up), near clip distance, and the focus
/// (distance, depth, blur, then 1 for a focus or 0 for none).
///
/// # Safety
/// `ptr` must address 68 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_set_view(ptr: *const u8) -> u32 {
    // SAFETY: the caller passes 17 floats.
    let v = unsafe { floats(ptr, 17) };
    with(0, |rt| {
        if v.iter().any(|x| !x.is_finite()) {
            return Err("set_view: values must be finite".into());
        }
        if !(v[9] > 0.0 && v[9] < 180.0 && v[12] > 0.0) {
            return Err(format!(
                "set_view: fov {} or near {} out of range",
                v[9], v[12]
            ));
        }
        let focus = (v[16] > 0.5).then_some(Focus {
            distance: v[13],
            depth: v[14],
            blur: v[15],
        });
        if focus.is_some_and(|f| !f.blurs() || f.distance < 0.0) {
            return Err(format!(
                "set_view: focus distance {}, depth {} or blur {} out of range",
                v[13], v[14], v[15]
            ));
        }
        rt.view = View {
            name: "host".into(),
            eye: [v[0], v[1], v[2]],
            target: [v[3], v[4], v[5]],
            up: [v[6], v[7], v[8]],
            fov: v[9],
            shift: [v[10], v[11]],
            near: v[12],
            focus,
        };
        Ok(1)
    })
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

/// Sets the time of day (`hours`, 0..24) and the animation clock (`seconds`: water, clouds,
/// grass and weather move with it).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_set_clock(hours: f32, seconds: f32) -> u32 {
    with(0, |rt| {
        if !hours.is_finite() || !seconds.is_finite() {
            return Err("set_clock: values must be finite".into());
        }
        rt.time.hours = hours.rem_euclid(24.0);
        rt.seconds = seconds;
        Ok(1)
    })
}

/// Lights the world's lamps `boost` times as brightly as the time of day does, their glow and the
/// light they cast alike (1: as the hour has them; a game's flare swells them for a moment).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_set_lamp_boost(boost: f32) -> u32 {
    with(0, |rt| {
        if !boost.is_finite() || boost < 0.0 {
            return Err("set_lamp_boost: a finite boost of 0 or more".into());
        }
        rt.lamp_boost = boost;
        Ok(1)
    })
}

/// Sets the weather to a preset (`clear`, `cloudy`, `overcast`, `mist`, `rain`, `storm`,
/// `petals`) named by the UTF-8 text at `ptr`.
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_set_weather(ptr: *const u8, len: usize) -> u32 {
    // SAFETY: the caller passes a readable range.
    let name = String::from_utf8_lossy(unsafe { bytes(ptr, len) }).into_owned();
    with(0, |rt| {
        rt.weather = Weather::preset(&name).ok_or_else(|| format!("unknown weather {name:?}"))?;
        Ok(1)
    })
}

/// Renders and presents a frame. Returns 1 when presented, 2 when the canvas had no image to
/// draw into (nothing to do), 0 after logging an error.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_render() -> u32 {
    with(0, |rt| {
        let Some(target) = rt.device.acquire()? else {
            return Ok(2);
        };
        let size = rt.device.surface_size();
        if size != rt.renderer.size() && size.0 > 0 && size.1 > 0 {
            rt.renderer.resize(&mut rt.device, size)?;
        }
        let mut env = Environment::resolve(
            &rt.time,
            &rt.weather,
            &rt.renderer.look().palette,
            rt.seconds,
        );
        env.lamp_level *= rt.lamp_boost;
        let active_view = rt.active_view();
        let mut enc = rt.device.encoder();
        rt.renderer.encode(
            &mut rt.device,
            &mut enc,
            &active_view,
            &env,
            [0.0, 0.0],
            0,
            Some(target),
        )?;
        rt.device.present(enc)?;
        rt.frames = rt.frames.wrapping_add(1);
        Ok(1)
    })
}

/// Frames presented so far.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_frames() -> u32 {
    RT.with(|cell| cell.borrow().as_ref().map_or(0, |rt| rt.frames))
}

/// Completes the world's baked vertex ambient occlusion now, rewriting the vertices it changes,
/// for a capture that must not depend on how many frames ran before it. Returns 1, or 0 after
/// logging why not.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_finish_ao() -> u32 {
    with(0, |rt| {
        rt.renderer.finish_ao(&mut rt.device)?;
        Ok(1)
    })
}

/// Loads the world's baked ambient occlusion from a sidecar `ganbate-build bake-ao` made for it
/// (`len` bytes at `ptr`), so frames bake only around what moves. Returns 1, or 0 (and reports
/// why) when it was baked for another world: the AO is then baked live as before.
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_ao_import(ptr: *const u8, len: usize) -> u32 {
    // SAFETY: the caller passes a readable range.
    let data = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        rt.renderer.import_ao(&mut rt.device, data)?;
        Ok(1)
    })
}

/// Does up to `samples` vertices' worth of the world's AO bake between frames (occluders first,
/// then samples nearest the camera), so the page can spend its idle time, measured by its own
/// clock, instead of waiting for frames. Returns 1 when it did work, 0 when there was none
/// (converged, or world edits still settling).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_step_ao(samples: u32) -> u32 {
    with(0, |rt| {
        let eye = rt.view.eye;
        Ok(u32::from(rt.renderer.step_ao(
            &mut rt.device,
            samples as usize,
            eye,
        )?))
    })
}

/// How far the world's baked vertex ambient occlusion has converged, in thousandths (`1000`
/// when complete or without a running engine). It refines a slice per frame after start and
/// after world edits, nearest the camera first.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_ao_progress() -> u32 {
    RT.with(|cell| {
        cell.borrow().as_ref().map_or(1000, |rt| {
            (rt.renderer.ao_progress() * 1000.0).round() as u32
        })
    })
}

/// Switches the renderer's quality preset at runtime (`0` = realtime, `1` = still,
/// `2` = hybrid stage, `3` = low-cost fast-path; unknown codes fall back to realtime).
/// Returns `1` on success, `0` on error.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_set_quality(quality: u32) -> u32 {
    with(0, |rt| {
        let preset = quality_from_code(quality);
        rt.renderer.set_quality(&mut rt.device, preset)?;
        Ok(1)
    })
}

/// Writes the latest [`FrameStats`] recorded by `Renderer::encode` as 9 little-endian `u32`
/// words (`36` bytes) at `out`:
/// `[world_draws_total, world_draws_submitted, world_draws_culled, shadow_static_cached, shadow_world_draws, ground_cached, reflection_drawn, ssao_drawn, fused_tonemap]`.
///
/// # Safety
/// `out` must address 36 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_frame_stats(out: *mut u8) -> u32 {
    with(0, |rt| {
        if out.is_null() {
            return Err("frame_stats: null output buffer".into());
        }
        let words = frame_stats_words(rt.renderer.last_frame_stats());
        let mut buf = [0u8; 36];
        for (i, w) in words.iter().enumerate() {
            buf[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        // SAFETY: the caller passes 36 writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), out, 36) };
        Ok(1)
    })
}

/// Writes what the last frame did with the room's cached picture as three words at `out`: 1 when
/// it started from the picture as it was (0: drew it whole), the regions of it drawn again, and 1
/// when pixels of it that moved props' shadows reach were shaded again.
///
/// # Safety
/// `out` must address 12 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_cache_stats(out: *mut u8) -> u32 {
    with(0, |rt| {
        if out.is_null() {
            return Err("cache_stats: null output buffer".into());
        }
        let s = rt.renderer.last_frame_stats();
        let words = [
            u32::from(s.world_cached),
            s.cache_regions,
            u32::from(s.reshaded),
        ];
        let mut buf = [0u8; 12];
        for (i, w) in words.iter().enumerate() {
            buf[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        // SAFETY: the caller passes 12 writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), out, 12) };
        Ok(1)
    })
}

/// Writes the size the renderer draws at (device pixels) as two words at `out`.
///
/// # Safety
/// `out` must address 8 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_size(out: *mut u8) -> u32 {
    with(0, |rt| {
        let (w, h) = rt.renderer.size();
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&w.to_le_bytes());
        b[4..].copy_from_slice(&h.to_le_bytes());
        // SAFETY: the caller passes 8 writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), out, 8) };
        Ok(1)
    })
}

/// A texture of `width` x `height` RGBA8 pixels (`len` = width * height * 4 bytes at `ptr`),
/// sRGB colour (`srgb` 1) or linear data such as normal maps (0), with a full mip chain.
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_texture_create(
    width: u32,
    height: u32,
    ptr: *const u8,
    len: usize,
    srgb: u32,
) -> u32 {
    // SAFETY: the caller passes a readable range.
    let data = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        if width == 0
            || height == 0
            || data.len() as u64 != u64::from(width) * u64::from(height) * 4
        {
            return Err(format!(
                "texture: {} bytes for {width}x{height} RGBA8",
                data.len()
            ));
        }
        let t =
            rt.renderer
                .create_prop_texture(&mut rt.device, (width, height), data, srgb != 0)?;
        Ok(t.raw())
    })
}

/// A texture from a PNG file (`len` bytes at `ptr`), decoded by the engine: sRGB colour
/// (`srgb` 1) or linear data (0), with a full mip chain.
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_texture_png(ptr: *const u8, len: usize, srgb: u32) -> u32 {
    // SAFETY: the caller passes a readable range.
    let data = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        let image = ganbate_engine::assets::png::decode(data).map_err(|e| format!("PNG: {e}"))?;
        let t = rt.renderer.create_prop_texture(
            &mut rt.device,
            (image.width, image.height),
            &image.rgba,
            srgb != 0,
        )?;
        Ok(t.raw())
    })
}

/// Replaces a texture's pixels (same size).
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_texture_update(
    texture: u32,
    ptr: *const u8,
    len: usize,
) -> u32 {
    // SAFETY: the caller passes a readable range.
    let data = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        rt.renderer
            .update_prop_texture(&mut rt.device, PropTexture::from_raw(texture), data)?;
        Ok(1)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_texture_destroy(texture: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .destroy_prop_texture(&mut rt.device, PropTexture::from_raw(texture));
        Ok(1)
    })
}

/// A mesh: `vertices` positions (3 floats each) at `positions`, normals likewise at `normals`
/// (null for flat normals), uvs (2 floats each) at `uvs` (null for none), and `index_count`
/// u32 indices at `indices` (counter-clockwise triangles).
///
/// # Safety
/// Each non-null pointer must address the floats or words its count implies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_mesh_create(
    positions: *const u8,
    vertices: usize,
    normals: *const u8,
    uvs: *const u8,
    indices: *const u8,
    index_count: usize,
) -> u32 {
    // SAFETY: the caller passes readable arrays of the stated sizes.
    let (p, n, uv, idx) = unsafe {
        (
            triples(&floats(positions, vertices * 3)),
            if normals.is_null() {
                vec![]
            } else {
                triples(&floats(normals, vertices * 3))
            },
            if uvs.is_null() {
                vec![]
            } else {
                floats(uvs, vertices * 2)
                    .chunks_exact(2)
                    .map(|c| [c[0], c[1]])
                    .collect()
            },
            words(indices, index_count),
        )
    };
    with(0, |rt| {
        let m = rt.renderer.create_prop_mesh(
            &mut rt.device,
            &MeshData {
                positions: &p,
                normals: &n,
                uvs: &uv,
                indices: &idx,
            },
        )?;
        Ok(m.raw())
    })
}

/// New positions (and normals, or null for flat ones) for a mesh's `vertices` vertices.
///
/// # Safety
/// As [`ganbate_rt_mesh_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_mesh_update(
    mesh: u32,
    positions: *const u8,
    normals: *const u8,
    vertices: usize,
) -> u32 {
    // SAFETY: the caller passes readable arrays of the stated sizes.
    let (p, n) = unsafe {
        (
            triples(&floats(positions, vertices * 3)),
            if normals.is_null() {
                vec![]
            } else {
                triples(&floats(normals, vertices * 3))
            },
        )
    };
    with(0, |rt| {
        rt.renderer
            .update_prop_mesh(&mut rt.device, PropMesh::from_raw(mesh), &p, &n)?;
        Ok(1)
    })
}

/// Destroys a mesh and every instance of it.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_mesh_destroy(mesh: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .destroy_prop_mesh(&mut rt.device, PropMesh::from_raw(mesh));
        Ok(1)
    })
}

/// A material from its JSON description (see `MaterialDesc::from_json`).
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_material_create(ptr: *const u8, len: usize) -> u32 {
    // SAFETY: the caller passes a readable range.
    let text = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        let desc = MaterialDesc::from_json(text)?;
        Ok(rt
            .renderer
            .create_prop_material(&mut rt.device, &desc)?
            .raw())
    })
}

/// Compiles a game's prop shader: WGSL defining `fn shade(in: ShadeInput) -> vec4f` (UTF-8,
/// `len` bytes at `ptr`). Returns its handle; 0 with the compiler's message on failure.
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_shader_create(ptr: *const u8, len: usize) -> u32 {
    // SAFETY: the caller passes a readable range.
    let text = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        let source = std::str::from_utf8(text).map_err(|e| format!("shader: {e}"))?;
        Ok(rt
            .renderer
            .create_prop_shader(&mut rt.device, source)?
            .raw())
    })
}

/// A material's twenty prop-shader parameters (floats at `ptr`), without a rebuild.
///
/// # Safety
/// `ptr` must address 80 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_material_params(material: u32, ptr: *const u8) -> u32 {
    // SAFETY: the caller passes twenty floats.
    let v = unsafe { floats(ptr, PROP_SHADER_PARAMS) };
    with(0, |rt| {
        let mut params = [0.0f32; PROP_SHADER_PARAMS];
        params.copy_from_slice(&v);
        rt.renderer
            .set_prop_material_params(PropMaterial::from_raw(material), &params)?;
        Ok(1)
    })
}

/// Changes a material in place (every instance that uses it changes with it).
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_material_update(
    material: u32,
    ptr: *const u8,
    len: usize,
) -> u32 {
    // SAFETY: the caller passes a readable range.
    let text = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        let desc = MaterialDesc::from_json(text)?;
        rt.renderer.update_prop_material(
            &mut rt.device,
            PropMaterial::from_raw(material),
            &desc,
        )?;
        Ok(1)
    })
}

/// Destroys a material and every instance that uses it.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_material_destroy(material: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .destroy_prop_material(&mut rt.device, PropMaterial::from_raw(material));
        Ok(1)
    })
}

/// Places `mesh` with `material` at the column-major model matrix (16 floats) at `matrix`.
///
/// # Safety
/// `matrix` must address 64 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_instance_create(
    mesh: u32,
    material: u32,
    matrix: *const u8,
) -> u32 {
    // SAFETY: the caller passes 16 floats.
    let m = unsafe { floats(matrix, 16) };
    with(0, |rt| {
        let i = rt.renderer.create_prop_instance(
            PropMesh::from_raw(mesh),
            PropMaterial::from_raw(material),
            &self::matrix(&m)?,
        )?;
        Ok(i.raw())
    })
}

/// Moves an instance and shows (`visible` 1) or hides it.
///
/// # Safety
/// `matrix` must address 64 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_instance_set(
    instance: u32,
    matrix: *const u8,
    visible: u32,
) -> u32 {
    // SAFETY: the caller passes 16 floats.
    let m = unsafe { floats(matrix, 16) };
    with(0, |rt| {
        rt.renderer.set_prop_instance(
            PropInstance::from_raw(instance),
            &self::matrix(&m)?,
            visible != 0,
        )?;
        Ok(1)
    })
}

/// Moves many instances at once: `count` records of 18 words at `ptr`, each the instance, 1 to
/// show it or 0 to hide it, and its 16-float model matrix. Stops at the first bad record.
///
/// # Safety
/// `ptr` must address `count * 72` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_instances_set(ptr: *const u8, count: usize) -> u32 {
    // SAFETY: the caller passes `count` records.
    let raw = unsafe { bytes(ptr, count * 72) };
    with(0, |rt| {
        for r in raw.chunks_exact(72) {
            let word = |i: usize| u32::from_le_bytes([r[i], r[i + 1], r[i + 2], r[i + 3]]);
            let m: Vec<f32> = r[8..]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            rt.renderer.set_prop_instance(
                PropInstance::from_raw(word(0)),
                &matrix(&m)?,
                word(4) != 0,
            )?;
        }
        Ok(1)
    })
}

/// Gives an instance another material.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_instance_material(instance: u32, material: u32) -> u32 {
    with(0, |rt| {
        rt.renderer.set_prop_instance_material(
            PropInstance::from_raw(instance),
            PropMaterial::from_raw(material),
        )?;
        Ok(1)
    })
}

/// Multiplies an instance's material colour and alpha by (`r`, `g`, `b`, `a`).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_instance_tint(instance: u32, r: f32, g: f32, b: f32, a: f32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .set_prop_instance_tint(PropInstance::from_raw(instance), [r, g, b, a])?;
        Ok(1)
    })
}

/// The pick layers (bits) an instance answers to; 0: never picked.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_instance_pick(instance: u32, layers: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .set_prop_instance_pick(PropInstance::from_raw(instance), layers)?;
        Ok(1)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_instance_destroy(instance: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .destroy_prop_instance(PropInstance::from_raw(instance));
        Ok(1)
    })
}

/// A game light from 15 floats: position (3), range, linear colour (3), intensity, a spot's
/// direction (3, all zero for a point light), its inner and outer half-angles (radians), 1 to
/// follow the lamps (lit only as the evening lights them), and how much of its light the props'
/// shadows take (0 casts none).
fn light_desc(v: &[f32]) -> LightDesc {
    let at = |i: usize| v.get(i).copied().unwrap_or(0.0);
    let direction = [at(8), at(9), at(10)];
    LightDesc {
        position: [at(0), at(1), at(2)],
        range: at(3),
        color: [at(4), at(5), at(6)],
        intensity: at(7),
        spot: direction.iter().any(|d| *d != 0.0).then_some(SpotCone {
            direction,
            inner: at(11),
            outer: at(12),
        }),
        follows_lamps: at(13) > 0.5,
        shadow: at(14),
    }
}

/// Hangs a game light (see `light_desc` for the 15 floats at `ptr`).
///
/// # Safety
/// `ptr` must address 60 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_light_create(ptr: *const u8) -> u32 {
    // SAFETY: the caller passes 15 floats.
    let v = unsafe { floats(ptr, 15) };
    with(0, |rt| Ok(rt.renderer.create_light(&light_desc(&v))?.raw()))
}

/// Moves, recolours or reshapes a game light (15 floats at `ptr`, as `ganbate_rt_light_create`).
///
/// # Safety
/// `ptr` must address 60 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_light_set(light: u32, ptr: *const u8) -> u32 {
    // SAFETY: the caller passes 15 floats.
    let v = unsafe { floats(ptr, 15) };
    with(0, |rt| {
        rt.renderer
            .set_light(GameLight::from_raw(light), &light_desc(&v))?;
        Ok(1)
    })
}

/// Puts a game light out.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_light_destroy(light: u32) -> u32 {
    with(0, |rt| {
        rt.renderer.destroy_light(GameLight::from_raw(light));
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

/// # Safety
/// `out` must address `v.len() * 4` writable bytes.
unsafe fn write_floats(out: *mut u8, v: &[f32]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    // SAFETY: forwarded.
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), out, b.len()) };
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A model's contents as JSON, for the host: images to decode (type, colour or data, size),
/// nodes (name, parent, mesh), meshes (name, morph target names), clips (name, duration and the
/// nodes they animate) and material names. Indices are positions in these lists.
fn describe(m: &Model) -> String {
    let list = |items: Vec<String>| format!("[{}]", items.join(","));
    let opt = |o: Option<usize>| o.map_or("null".to_string(), |v| v.to_string());
    let images = m
        .images
        .iter()
        .map(|i| {
            format!(
                r#"{{"name":{},"mime":{},"srgb":{},"size":{}}}"#,
                json_string(&i.name),
                json_string(&i.mime),
                i.srgb,
                i.data.len()
            )
        })
        .collect();
    let nodes = m
        .nodes
        .iter()
        .map(|n| {
            format!(
                r#"{{"name":{},"parent":{},"mesh":{},"skin":{}}}"#,
                json_string(&n.name),
                opt(n.parent),
                opt(n.mesh),
                opt(n.skin)
            )
        })
        .collect();
    let meshes = m
        .meshes
        .iter()
        .map(|me| {
            format!(
                r#"{{"name":{},"targets":{},"targetNames":{}}}"#,
                json_string(&me.name),
                me.weights.len(),
                list(me.target_names.iter().map(|t| json_string(t)).collect())
            )
        })
        .collect();
    let clips = m
        .clips
        .iter()
        .map(|c| {
            // Each node a channel moves, once, in the order first met.
            let mut nodes: Vec<usize> = Vec::new();
            for channel in &c.channels {
                if !nodes.contains(&channel.node) {
                    nodes.push(channel.node);
                }
            }
            format!(
                r#"{{"name":{},"duration":{},"nodes":{}}}"#,
                json_string(&c.name),
                c.duration,
                list(nodes.iter().map(usize::to_string).collect())
            )
        })
        .collect();
    let materials = m.materials.iter().map(|x| json_string(&x.name)).collect();
    format!(
        r#"{{"images":{},"nodes":{},"meshes":{},"clips":{},"materials":{}}}"#,
        list(images),
        list(nodes),
        list(meshes),
        list(clips),
        list(materials)
    )
}

fn pending(rt: &mut Runtime, id: u32) -> Result<&mut Model, String> {
    rt.pending
        .get_mut((id as usize).wrapping_sub(1))
        .and_then(Option::as_mut)
        .ok_or_else(|| format!("no parsed model {id}"))
}

/// Parses a GLB model (`len` bytes at `ptr`, read during the call). Returns a parsed-model
/// handle for [`ganbate_rt_model_describe`], [`ganbate_rt_model_image`] and [`ganbate_rt_model_create`].
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_parse(ptr: *const u8, len: usize) -> u32 {
    // SAFETY: the caller passes a readable range.
    let data = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        let m = model::load_glb(data).map_err(|e| format!("model: {e}"))?;
        let slot = rt.pending.iter().position(Option::is_none);
        let i = match slot {
            Some(i) => {
                rt.pending[i] = Some(m);
                i
            }
            None => {
                rt.pending.push(Some(m));
                rt.pending.len() - 1
            }
        };
        Ok(i as u32 + 1)
    })
}

/// Writes a parsed model's description (JSON, see `describe`) to `out` if it fits in `cap`
/// bytes; returns its length.
///
/// # Safety
/// `out` must address `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_describe(parsed: u32, out: *mut u8, cap: usize) -> u32 {
    with(0, |rt| {
        let text = describe(pending(rt, parsed)?);
        // SAFETY: forwarded.
        Ok(unsafe { give(text.as_bytes(), out, cap) })
    })
}

/// Copies image `index` of a parsed model (encoded PNG, JPEG or WebP) to `out` if it fits in
/// `cap` bytes; returns its length.
///
/// # Safety
/// `out` must address `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_image(
    parsed: u32,
    index: u32,
    out: *mut u8,
    cap: usize,
) -> u32 {
    with(0, |rt| {
        let m = pending(rt, parsed)?;
        let image = m
            .images
            .get(index as usize)
            .ok_or_else(|| format!("no image {index}"))?;
        // SAFETY: forwarded.
        Ok(unsafe { give(&image.data, out, cap) })
    })
}

/// Uploads a parsed model with its textures (`count` texture handles at `textures`, one per
/// image, 0 where an image could not be decoded). Returns the model handle; the parsed handle
/// is spent either way.
///
/// # Safety
/// `textures` must address `count` words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_create(
    parsed: u32,
    textures: *const u8,
    count: usize,
) -> u32 {
    // SAFETY: the caller passes `count` words.
    let tex: Vec<Option<PropTexture>> = unsafe { words(textures, count) }
        .into_iter()
        .map(|t| (t != 0).then(|| PropTexture::from_raw(t)))
        .collect();
    with(0, |rt| {
        let slot = (parsed as usize).wrapping_sub(1);
        let m = rt
            .pending
            .get_mut(slot)
            .and_then(Option::take)
            .ok_or_else(|| format!("no parsed model {parsed}"))?;
        Ok(rt.renderer.create_model(&mut rt.device, m, &tex)?.raw())
    })
}

/// The prop material made for the model's material `index` (to restyle it).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_material(model: u32, index: u32) -> u32 {
    with(0, |rt| {
        Ok(rt
            .renderer
            .model_material(ModelId::from_raw(model), index as usize)?
            .raw())
    })
}

/// Destroys a model and its instances (textures stay the host's).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_destroy(model: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .destroy_model(&mut rt.device, ModelId::from_raw(model));
        Ok(1)
    })
}

/// Places a model at the column-major matrix (16 floats) at `matrix`, in its rest pose.
///
/// # Safety
/// `matrix` must address 64 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_instance(model: u32, matrix: *const u8) -> u32 {
    // SAFETY: the caller passes 16 floats.
    let m = unsafe { floats(matrix, 16) };
    with(0, |rt| {
        let i = rt.renderer.create_model_instance(
            &mut rt.device,
            ModelId::from_raw(model),
            &self::matrix(&m)?,
        )?;
        Ok(i.raw())
    })
}

/// Moves a model instance and shows (`visible` 1) or hides it.
///
/// # Safety
/// `matrix` must address 64 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_instance_set(
    instance: u32,
    matrix: *const u8,
    visible: u32,
) -> u32 {
    // SAFETY: the caller passes 16 floats.
    let m = unsafe { floats(matrix, 16) };
    with(0, |rt| {
        rt.renderer.set_model_instance(
            ModelInstance::from_raw(instance),
            &self::matrix(&m)?,
            visible != 0,
        )?;
        Ok(1)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_instance_destroy(instance: u32) -> u32 {
    with(0, |rt| {
        rt.crossfade_from.remove(&instance);
        rt.renderer
            .destroy_model_instance(&mut rt.device, ModelInstance::from_raw(instance));
        Ok(1)
    })
}

/// Snapshots `instance`'s current pose inside Rust WASM memory so subsequent calls to
/// [`ganbate_rt_pose_model_clip`] can Hermite cross-fade from it without JS joint copies.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_snapshot_pose(instance: u32) -> u32 {
    with(0, |rt| {
        let (pose, _) = rt
            .renderer
            .model_pose_and_model(ModelInstance::from_raw(instance))?;
        let snap = pose.clone();
        rt.crossfade_from.insert(instance, snap);
        Ok(1)
    })
}

/// Evaluates skeletal clip layers, Hermite cross-fade from [`ganbate_rt_model_snapshot_pose`],
/// Look-At chains, analytical Two-Bone IK, additive bone offsets, morph overrides, and optional
/// GPU skinning/morph commit in a single WASM call.
///
/// Packet layout (`float_count` little-endian `f32` values at `cmd_ptr`):
/// - Header (`9` floats): `[flags, clip_count, fade_elapsed_s, fade_duration_s, fade_mask_count,
///   look_at_count, ik_count, offset_count, morph_count]`
///   where `flags` bits are `1` (`reset_to_rest`), `2` (`commit_update`), `4` (`use_crossfade`).
/// - `clip_count * 3`: `[clip_idx, time_s, weight]`
/// - `fade_mask_count`: `[node_idx]`
/// - `look_at_count`: `[joint_count, tx, ty, tz, fx, fy, fz, ux, uy, uz, max_yaw, max_pitch,
///   weight, (node_idx, share) * joint_count]`
/// - `ik_count * 10`: `[root, mid, end, tx, ty, tz, px, py, pz, weight]`
/// - `offset_count * 11`: `[node, tx, ty, tz, qx, qy, qz, qw, sx, sy, sz]`
/// - `morph_count * 3`: `[node, target, weight]`
///
/// # Safety
/// `cmd_ptr` must address `float_count * 4` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_pose_model_clip(
    instance: u32,
    cmd_ptr: *const u8,
    float_count: usize,
) -> u32 {
    if float_count < 9 {
        say(2, "pose_model_clip: packet needs at least 9 header floats");
        return 0;
    }
    // SAFETY: forwarded from caller.
    let f = unsafe { floats(cmd_ptr, float_count) };
    with(0, |rt| {
        if f.iter().any(|x| !x.is_finite()) {
            return Err("pose_model_clip: non-finite packet value".into());
        }
        let flags = f[0] as u32;
        let clip_count = f[1] as usize;
        let fade_elapsed = f[2];
        let fade_duration = f[3];
        let mask_count = f[4] as usize;
        let look_at_count = f[5] as usize;
        let ik_count = f[6] as usize;
        let offset_count = f[7] as usize;
        let morph_count = f[8] as usize;

        let mut cursor = 9usize;
        let take = |cur: &mut usize, n: usize| -> Result<&[f32], String> {
            let end = cur
                .checked_add(n)
                .filter(|&e| e <= f.len())
                .ok_or_else(|| "pose_model_clip: truncated command packet".to_string())?;
            let slice = &f[*cur..end];
            *cur = end;
            Ok(slice)
        };

        let raw_clips = take(&mut cursor, clip_count * 3)?;
        let clips: Vec<ClipLayer> = raw_clips
            .chunks_exact(3)
            .map(|c| ClipLayer {
                clip: c[0] as usize,
                time: c[1],
                weight: c[2],
            })
            .collect();

        let raw_mask = take(&mut cursor, mask_count)?;
        let mask: Vec<usize> = raw_mask.iter().map(|&n| n as usize).collect();

        let mut look_at = Vec::with_capacity(look_at_count);
        for _ in 0..look_at_count {
            let hdr = take(&mut cursor, 13)?;
            let joints_n = hdr[0] as usize;
            let raw_joints = take(&mut cursor, joints_n * 2)?;
            let nodes = raw_joints
                .chunks_exact(2)
                .map(|j| (j[0] as usize, j[1]))
                .collect();
            look_at.push(LookAtChain {
                nodes,
                target_model: [hdr[1], hdr[2], hdr[3]],
                forward: [hdr[4], hdr[5], hdr[6]],
                up: [hdr[7], hdr[8], hdr[9]],
                max_yaw: hdr[10],
                max_pitch: hdr[11],
                weight: hdr[12],
            });
        }

        let raw_ik = take(&mut cursor, ik_count * 10)?;
        let two_bone_ik: Vec<TwoBoneIk> = raw_ik
            .chunks_exact(10)
            .map(|k| TwoBoneIk {
                root: k[0] as usize,
                mid: k[1] as usize,
                end: k[2] as usize,
                target_model: [k[3], k[4], k[5]],
                pole_model: [k[6], k[7], k[8]],
                weight: k[9],
            })
            .collect();

        let raw_offsets = take(&mut cursor, offset_count * 11)?;
        let bone_offsets: Vec<BoneOffset> = raw_offsets
            .chunks_exact(11)
            .map(|o| BoneOffset {
                node: o[0] as usize,
                translation: [o[1], o[2], o[3]],
                rotation: [o[4], o[5], o[6], o[7]],
                scale: [o[8], o[9], o[10]],
            })
            .collect();

        let raw_morphs = take(&mut cursor, morph_count * 3)?;
        let morph_overrides: Vec<MorphOverride> = raw_morphs
            .chunks_exact(3)
            .map(|m| MorphOverride {
                node: m[0] as usize,
                target: m[1] as usize,
                weight: m[2],
            })
            .collect();

        let snap = if (flags & 4) != 0 {
            rt.crossfade_from.get(&instance).cloned()
        } else {
            None
        };
        let crossfade = ((flags & 4) != 0).then_some((
            fade_elapsed,
            fade_duration,
            (!mask.is_empty()).then_some(mask.as_slice()),
        ));

        let handle = ModelInstance::from_raw(instance);
        {
            let (pose, model) = rt.renderer.model_pose_and_model(handle)?;
            pose_model_clip(
                pose,
                snap.as_ref(),
                model,
                &PoseSpec {
                    reset_to_rest: (flags & 1) != 0,
                    clips: &clips,
                    crossfade,
                    look_at: &look_at,
                    two_bone_ik: &two_bone_ik,
                    bone_offsets: &bone_offsets,
                    morph_overrides: &morph_overrides,
                },
            );
        }
        if (flags & 2) != 0 {
            rt.renderer.update_model_instance(&mut rt.device, handle)?;
        }
        Ok(1)
    })
}

/// Resets an instance's pose to the model's rest pose.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_rest(instance: u32) -> u32 {
    with(0, |rt| {
        let (pose, model) = rt
            .renderer
            .model_pose_and_model(ModelInstance::from_raw(instance))?;
        *pose = ganbate_engine::anim::Pose::rest(model);
        Ok(1)
    })
}

/// Mixes clip `clip` at `time` seconds into the instance's pose with `weight` (1 replaces what
/// the clip animates).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_clip(instance: u32, clip: u32, time: f32, weight: f32) -> u32 {
    with(0, |rt| {
        let (pose, model) = rt
            .renderer
            .model_pose_and_model(ModelInstance::from_raw(instance))?;
        if clip as usize >= model.clips.len() {
            return Err(format!("no clip {clip}"));
        }
        pose.apply(model, clip as usize, time, weight);
        Ok(1)
    })
}

/// Writes node `node`'s local transform as 10 floats at `out`: translation (3), rotation
/// quaternion x, y, z, w (4), scale (3).
///
/// # Safety
/// `out` must address 40 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_node_get(instance: u32, node: u32, out: *mut u8) -> u32 {
    with(0, |rt| {
        let (pose, _) = rt
            .renderer
            .model_pose_and_model(ModelInstance::from_raw(instance))?;
        let n = node as usize;
        if n >= pose.translation.len() {
            return Err(format!("no node {node}"));
        }
        let mut v = [0.0f32; 10];
        v[..3].copy_from_slice(&pose.translation[n]);
        v[3..7].copy_from_slice(&pose.rotation[n]);
        v[7..].copy_from_slice(&pose.scale[n]);
        // SAFETY: the caller passes 40 writable bytes.
        unsafe { write_floats(out, &v) };
        Ok(1)
    })
}

/// Sets node `node`'s local transform from 10 floats at `ptr` (as `ganbate_rt_model_node_get`
/// writes them; the rotation is normalized).
///
/// # Safety
/// `ptr` must address 40 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_node_set(
    instance: u32,
    node: u32,
    ptr: *const u8,
) -> u32 {
    // SAFETY: the caller passes 10 floats.
    let v = unsafe { floats(ptr, 10) };
    with(0, |rt| {
        if v.iter().any(|x| !x.is_finite()) {
            return Err("model_node_set: values must be finite".into());
        }
        let (pose, _) = rt
            .renderer
            .model_pose_and_model(ModelInstance::from_raw(instance))?;
        let n = node as usize;
        if n >= pose.translation.len() {
            return Err(format!("no node {node}"));
        }
        pose.translation[n] = [v[0], v[1], v[2]];
        pose.rotation[n] = normalize_q([v[3], v[4], v[5], v[6]]);
        pose.scale[n] = [v[7], v[8], v[9]];
        Ok(1)
    })
}

/// Sets morph target `target` of node `node`'s mesh to `weight`.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_morph(
    instance: u32,
    node: u32,
    target: u32,
    weight: f32,
) -> u32 {
    with(0, |rt| {
        let (pose, _) = rt
            .renderer
            .model_pose_and_model(ModelInstance::from_raw(instance))?;
        let w = pose
            .weights
            .get_mut(node as usize)
            .and_then(|w| w.get_mut(target as usize))
            .ok_or_else(|| format!("node {node} has no morph target {target}"))?;
        if !weight.is_finite() {
            return Err("model_morph: the weight must be finite".into());
        }
        *w = weight;
        Ok(1)
    })
}

/// Applies the instance's pose: node transforms, skinning and morphing.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_update(instance: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .update_model_instance(&mut rt.device, ModelInstance::from_raw(instance))?;
        Ok(1)
    })
}

/// Writes node `node`'s world matrix (16 floats, as of the last update) at `out`.
///
/// # Safety
/// `out` must address 64 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_node_world(
    instance: u32,
    node: u32,
    out: *mut u8,
) -> u32 {
    with(0, |rt| {
        let m = rt
            .renderer
            .model_node_world(ModelInstance::from_raw(instance), node as usize)?;
        // SAFETY: the caller passes 64 writable bytes.
        unsafe { write_floats(out, &m) };
        Ok(1)
    })
}

/// Shows (`visible` 1) or hides the meshes of node `node`.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_node_visible(instance: u32, node: u32, visible: u32) -> u32 {
    with(0, |rt| {
        rt.renderer.set_model_node_visible(
            ModelInstance::from_raw(instance),
            node as usize,
            visible != 0,
        )?;
        Ok(1)
    })
}

/// Tints every part of a model instance.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_tint(instance: u32, r: f32, g: f32, b: f32, a: f32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .set_model_instance_tint(ModelInstance::from_raw(instance), [r, g, b, a])?;
        Ok(1)
    })
}

/// The model instance a picked prop instance belongs to (0 for none); writes the node it draws
/// as a word at `out`.
///
/// # Safety
/// `out` must address 4 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_owner(prop: u32, out: *mut u8) -> u32 {
    with(0, |rt| {
        let Some((i, node)) = rt.renderer.model_part_owner(PropInstance::from_raw(prop)) else {
            return Ok(0);
        };
        let b = (node as u32).to_le_bytes();
        // SAFETY: the caller passes 4 writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), out, 4) };
        Ok(i.raw())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_from_code_maps_all_presets_and_preserves_realtime_fallback() {
        assert_eq!(quality_from_code(0), Quality::realtime());
        assert_eq!(quality_from_code(1), Quality::still());
        assert_eq!(quality_from_code(2), Quality::hybrid_stage());
        assert_eq!(quality_from_code(3), Quality::low_cost());
        assert_eq!(quality_from_code(4), Quality::realtime());
        assert_eq!(quality_from_code(u32::MAX), Quality::realtime());
    }

    #[test]
    fn frame_stats_words_packs_nine_little_endian_words_exact() {
        let stats = FrameStats {
            world_draws_total: 42,
            world_draws_submitted: 27,
            world_draws_culled: 15,
            shadow_static_cached: true,
            shadow_world_draws: 11,
            ground_cached: false,
            reflection_drawn: false,
            ssao_drawn: true,
            fused_tonemap: true,
            world_cached: false,
        };
        assert_eq!(frame_stats_words(stats), [42, 27, 15, 1, 11, 0, 0, 1, 1]);
    }
}
