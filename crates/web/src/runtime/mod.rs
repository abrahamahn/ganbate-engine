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

mod models;
mod picking;
mod props;

pub(crate) use picking::{ganbate_rt_set_world_light, move_world_instances};

use crate::{install_panic_hook, say};
use ganbate_engine::anim::Pose;
use ganbate_engine::assets::model::Model;
use ganbate_engine::gpu::Device;
use ganbate_engine::render::{
    Environment, Focus, FrameStats, LookFile, Output, Quality, Renderer, TimeOfDay, View, Weather,
};
use ganbate_engine::world::World;
use ganbate_engine::xr::{DEFAULT_IPD_METRES, FrameBudgetGovernor, compute_stereo_views};
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

/// Draws the main pass as several while passes are timed (`on` 1), so a device's timing shows
/// each part ([`ganbate_engine::render::Renderer::set_split_passes`]). Returns 1, or 0 before
/// `ganbate_rt_start`.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_set_split_passes(on: u32) -> u32 {
    with(0, |rt| {
        rt.renderer.set_split_passes(on != 0);
        Ok(1)
    })
}

/// Draws the fused tone-map output grading each pixel once (`on` 1) or five times (0)
/// ([`ganbate_engine::render::Renderer::set_grade_once`]): the same picture, a fifth of the
/// grading and 32 bytes a pixel more memory traffic, for Arm Mali GPUs. Returns 1 when frames now
/// grade once; 0 when they keep the fused pass (`on` 0, or a device that cannot read 32-bit float
/// textures), before `ganbate_rt_start` and after logging an error.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_set_grade_once(on: u32) -> u32 {
    with(0, |rt| {
        let once = rt.renderer.set_grade_once(&mut rt.device, on != 0)?;
        Ok(u32::from(once))
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

/// Writes what the last frame did with the room's cached picture as eight words at `out`: 1 when
/// it started from the picture as it was (0: drew it whole), the regions of it drawn again, 1
/// when pixels of it that moved props' shadows reach were shaded again; then 1 when the water
/// read its lamp light and shadow as kept with the picture (0: computed them), 1 when those were
/// drawn whole, the regions of them drawn again, and the footprints of moved props' shadows
/// within which the water computed them; then the pixels of the region the moved props'
/// shadows were sought in and shaded again within (0: none).
///
/// # Safety
/// `out` must address 32 writable bytes.
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
            u32::from(s.water_kept),
            u32::from(s.water_statics_whole),
            s.water_statics_regions,
            s.water_footprints,
            s.reshade_pixels,
        ];
        let mut buf = [0u8; 32];
        for (i, w) in words.iter().enumerate() {
            buf[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        // SAFETY: the caller passes 32 writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), out, 32) };
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

/// # Safety
/// `out` must address `v.len() * 4` writable bytes.
unsafe fn write_floats(out: *mut u8, v: &[f32]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    // SAFETY: forwarded.
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), out, b.len()) };
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
            ..FrameStats::default()
        };
        assert_eq!(frame_stats_words(stats), [42, 27, 15, 1, 11, 0, 0, 1, 1]);
    }
}
