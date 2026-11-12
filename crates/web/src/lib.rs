//! Ganbate's web module: the stylized renderer drawing into a browser canvas through WebGPU or
//! WebGL 2. It has two modes. The player (this file) flies through a cooked world with the
//! controls of `ganbate-player`; library mode (`runtime.rs`, `ganbate_rt_*`) lets a host page such as
//! a game drive the camera, clock, weather and props itself.
//!
//! `web/ganbate.js` instantiates this module, copies the world (`.gbew`) into memory allocated
//! with [`ganbate_alloc`], calls [`ganbate_load_world`] and [`ganbate_start`] (with the Look file and the
//! page's query string), forwards input to [`ganbate_key`], [`ganbate_look`] and [`ganbate_move`], and
//! calls [`ganbate_frame`] from `requestAnimationFrame`. Messages and the status line go back
//! through the host imports `log` and `status`.
//!
//! Query parameters read here: `view=NAME`, `time=HOURS`, `weather=PRESET`, `seconds=S`
//! (start of the animation clock), `paused=1`, `quality=high`.
#![cfg_attr(not(target_arch = "wasm32"), allow(rustdoc::broken_intra_doc_links))]
#![cfg(target_arch = "wasm32")]

mod graph;
mod physics;
mod runtime;
mod selftest;

use ganbate_engine::gpu::Device;
use ganbate_engine::render::{
    Environment, LookFile, Output, Quality, Renderer, TimeOfDay, View, Weather, math,
};
use ganbate_engine::world::World;
use std::cell::RefCell;
use std::collections::HashSet;

#[link(wasm_import_module = "ganbate")]
unsafe extern "C" {
    fn log(level: u32, text: *const u8, len: usize);
    fn status(text: *const u8, len: usize);
}

/// Sends a message to the page: level 0 info, 1 warning, 2 error.
pub(crate) fn say(level: u32, msg: &str) {
    // SAFETY: the host reads the UTF-8 text during the call.
    unsafe { log(level, msg.as_ptr(), msg.len()) }
}

const HELP: &str = "W/A/S/D fly, Q/E down/up, arrows or drag look, Shift faster, 1-9 views, \
T/G time +/-, C day cycle, Y next weather, P pause motion. Touch: drag to look, two fingers to \
move.";

/// Key codes from the host: lowercase ASCII for characters, these for the rest.
const LEFT: u32 = 1;
const RIGHT: u32 = 2;
const UP: u32 = 3;
const DOWN: u32 = 4;
const SHIFT: u32 = 5;

struct Fly {
    pos: math::V3,
    yaw: f32,
    pitch: f32,
}
impl Fly {
    fn from_view(v: &View) -> Self {
        let d = math::normalize(math::sub(v.target, v.eye));
        Self {
            pos: v.eye,
            yaw: d[0].atan2(-d[2]),
            pitch: d[1].clamp(-1.0, 1.0).asin(),
        }
    }
    fn forward(&self) -> math::V3 {
        let c = self.pitch.cos();
        [c * self.yaw.sin(), self.pitch.sin(), -c * self.yaw.cos()]
    }
    fn view(&self, fov: f32) -> View {
        // The target sits 30 m ahead: the renderer sizes shadow cascades from it.
        View {
            name: "free".into(),
            eye: self.pos,
            target: math::add(self.pos, math::scale(self.forward(), 30.0)),
            fov,
            ..View::default()
        }
    }
}

fn clock(hours: f32) -> String {
    let m = (hours.rem_euclid(24.0) * 60.0).round() as u32 % (24 * 60);
    format!("{:02}:{:02}", m / 60, m % 60)
}

struct App {
    device: Device<'static>,
    renderer: Renderer,
    views: Vec<View>,
    /// The preset view, exactly as the Look file states it, until the camera moves.
    preset: Option<View>,
    fly: Fly,
    fov: f32,
    time: TimeOfDay,
    weather: Weather,
    weather_name: String,
    held: HashSet<u32>,
    look: (f32, f32),
    stick: (f32, f32),
    cycle: bool,
    paused: bool,
    seconds: f32,
    frames: u64,
    fps: (u32, f32, f32),
    status_age: f32,
}

thread_local! {
    /// The world decoded by `ganbate_load_world`, until the player or the engine starts with it.
    pub(crate) static WORLD: RefCell<Option<World>> = const { RefCell::new(None) };
    /// The loaded world's entity table as JSON (empty for worlds cooked without entities).
    pub(crate) static ENTITIES: RefCell<String> = const { RefCell::new(String::new()) };
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
}

pub(crate) fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| say(2, &format!("panic: {info}"))));
}

/// Memory for the host to copy a file into; ownership passes back with [`ganbate_load_world`].
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Releases memory from [`ganbate_alloc`] that was not handed to the player.
///
/// # Safety
/// `ptr` and `len` must come from one `ganbate_alloc(len)` call and not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_free(ptr: *mut u8, len: usize) {
    // SAFETY: the caller passes a pointer and capacity from `ganbate_alloc`.
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len.max(1)) });
}

/// Takes the `.gbew` bytes at `ptr` (from `ganbate_alloc(len)`, fully written) and decodes them.
/// Returns 1 on success; on failure it logs the reason and returns 0.
///
/// # Safety
/// `ptr` must come from `ganbate_alloc(len)` with all `len` bytes initialized; the player owns
/// and frees the memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_load_world(ptr: *mut u8, len: usize) -> u32 {
    install_panic_hook();
    // SAFETY: the caller hands over an initialized allocation of `len` bytes.
    let bytes = unsafe { Vec::from_raw_parts(ptr, len, len.max(1)) };
    let world = ganbate_engine::assets::cooked::decode_scene(&bytes);
    drop(bytes);
    match world {
        Ok((w, table)) => {
            ENTITIES.with(|cell| *cell.borrow_mut() = entities_json(&w, &table));
            physics::prepare(&w, &table);
            say(
                0,
                &format!(
                    "world: {} meshes, {} instances, {} materials, {} textures, {} lights",
                    w.meshes.len(),
                    w.instances.len(),
                    w.materials.len(),
                    w.textures.len(),
                    w.lights.len()
                ),
            );
            WORLD.with(|cell| *cell.borrow_mut() = Some(w));
            1
        }
        Err(e) => {
            say(2, &format!("cannot decode the world: {e}"));
            0
        }
    }
}

/// An entity table as JSON: `{"entities": [{"id", "name", "parent", "transform", "instances":
/// [first, count], "lights": [first, count], "bounds": [[lo], [hi]]}], "instanceSpecs":
/// [{"material", "bounds"}], "lamps": [[x, y, z]], "data": {...}}`, or `""` when it is empty.
pub(crate) fn entities_json(
    world: &ganbate_engine::world::World,
    table: &ganbate_engine::world::EntityTable,
) -> String {
    use ganbate_engine::json::{Value, to_compact};
    use std::collections::BTreeMap;
    if table.entities.is_empty() && table.data.is_empty() {
        return String::new();
    }
    let n = |v: u32| Value::Number(f64::from(v));
    let arr3 = |v: [f32; 3]| Value::Array(v.iter().map(|x| Value::Number(f64::from(*x))).collect());
    let inst_bounds: Vec<Option<([f32; 3], [f32; 3])>> = world
        .instances
        .iter()
        .map(|inst| {
            let mesh = world.meshes.get(inst.mesh as usize)?;
            let mut lo = [f32::INFINITY; 3];
            let mut hi = [f32::NEG_INFINITY; 3];
            for p in &mesh.positions {
                let w = inst.transform.point(*p);
                for (k, v) in [w.0, w.1, w.2].into_iter().enumerate() {
                    lo[k] = lo[k].min(v);
                    hi[k] = hi[k].max(v);
                }
            }
            (lo[0] <= hi[0]).then_some((lo, hi))
        })
        .collect();
    let instance_specs: Vec<Value> = world
        .instances
        .iter()
        .enumerate()
        .map(|(idx, inst)| {
            let mat_name = world
                .meshes
                .get(inst.mesh as usize)
                .and_then(|m| m.material)
                .and_then(|mi| world.materials.get(mi as usize))
                .map_or("", |mat| mat.name.as_str());
            let b = inst_bounds
                .get(idx)
                .copied()
                .flatten()
                .map_or(Value::Null, |(lo, hi)| {
                    Value::Array(vec![arr3(lo), arr3(hi)])
                });
            Value::Object(BTreeMap::from([
                ("material".to_string(), Value::String(mat_name.to_string())),
                ("bounds".into(), b),
            ]))
        })
        .collect();
    let entities = table
        .entities
        .iter()
        .map(|e| {
            let mut lo = [f32::INFINITY; 3];
            let mut hi = [f32::NEG_INFINITY; 3];
            for i in e.instances.clone() {
                if let Some((ilo, ihi)) = inst_bounds.get(i as usize).copied().flatten() {
                    for k in 0..3 {
                        lo[k] = lo[k].min(ilo[k]);
                        hi[k] = hi[k].max(ihi[k]);
                    }
                }
            }
            let bounds_val = if lo[0] <= hi[0] {
                Value::Array(vec![arr3(lo), arr3(hi)])
            } else {
                Value::Null
            };
            Value::Object(BTreeMap::from([
                ("id".to_string(), Value::String(format!("{:016x}", e.id))),
                ("name".into(), Value::String(e.name.clone())),
                ("parent".into(), e.parent.map_or(Value::Null, n)),
                (
                    "transform".into(),
                    Value::Array(
                        e.transform
                            .0
                            .iter()
                            .map(|v| Value::Number(f64::from(*v)))
                            .collect(),
                    ),
                ),
                (
                    "instances".into(),
                    Value::Array(vec![
                        n(e.instances.start),
                        n(e.instances.end - e.instances.start),
                    ]),
                ),
                (
                    "lights".into(),
                    Value::Array(vec![n(e.lights.start), n(e.lights.end - e.lights.start)]),
                ),
                ("bounds".into(), bounds_val),
            ]))
        })
        .collect();
    let lamps: Vec<Value> = world
        .lights
        .iter()
        .map(|l| arr3([l.position.0, l.position.1, l.position.2]))
        .collect();
    let data = ganbate_engine::json::parse(table.data.as_bytes()).unwrap_or(Value::Null);
    to_compact(&Value::Object(BTreeMap::from([
        ("entities".to_string(), Value::Array(entities)),
        ("instanceSpecs".into(), Value::Array(instance_specs)),
        ("lamps".into(), Value::Array(lamps)),
        ("data".into(), data),
    ])))
    .unwrap_or_default()
}

/// Writes the loaded world's entity table as JSON (see `entities_json`) to `out` when it fits
/// in `cap` bytes; returns its length (0: the world has none).
///
/// # Safety
/// `out` must address `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_world_entities(out: *mut u8, cap: usize) -> u32 {
    ENTITIES.with(|cell| {
        // SAFETY: forwarded from the caller.
        unsafe { give(cell.borrow().as_bytes(), out, cap) }
    })
}

/// Copies `bytes` to `out` when they fit in `cap`; returns their length either way.
///
/// # Safety
/// `out` must address `cap` writable bytes.
pub(crate) unsafe fn give(bytes: &[u8], out: *mut u8, cap: usize) -> u32 {
    if bytes.len() <= cap && !out.is_null() {
        // SAFETY: the caller passes `cap` writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len()) };
    }
    bytes.len() as u32
}

/// Lists the loaded world's still-encoded images as JSON (`[{"texture", "mime", "srgb",
/// "width", "height", "size"}]`) for the host to decode, written to `out` when it fits in `cap`
/// bytes; returns the length. The host decodes what it can and hands the pixels back with
/// [`ganbate_world_set_image`]; PNGs left encoded are decoded by the engine.
///
/// # Safety
/// `out` must address `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_world_images(out: *mut u8, cap: usize) -> u32 {
    let text = WORLD.with(|cell| {
        let world = cell.borrow();
        let items: Vec<String> = world
            .iter()
            .flat_map(|w| w.textures.iter().enumerate())
            .filter_map(|(i, t)| {
                t.encoded.as_ref().map(|e| {
                    format!(
                        r#"{{"texture":{i},"mime":"{}","srgb":{},"width":{},"height":{},"size":{}}}"#,
                        e.mime,
                        t.srgb,
                        t.width,
                        t.height,
                        e.data.len()
                    )
                })
            })
            .collect();
        format!("[{}]", items.join(","))
    });
    // SAFETY: forwarded.
    unsafe { give(text.as_bytes(), out, cap) }
}

/// Copies texture `texture`'s encoded image to `out` when it fits in `cap` bytes; returns its
/// length (0 when the texture is not encoded).
///
/// # Safety
/// `out` must address `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_world_image(texture: u32, out: *mut u8, cap: usize) -> u32 {
    WORLD.with(|cell| {
        let world = cell.borrow();
        let Some(e) = world
            .as_ref()
            .and_then(|w| w.textures.get(texture as usize))
            .and_then(|t| t.encoded.as_ref())
        else {
            return 0;
        };
        // SAFETY: forwarded.
        unsafe { give(&e.data, out, cap) }
    })
}

/// Replaces texture `texture`'s encoded image with its decoded pixels (`width` x `height`
/// RGBA8, `len` bytes at `ptr`, read during the call; the engine makes the mips). Returns 1, or
/// 0 after logging why not.
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_world_set_image(
    texture: u32,
    width: u32,
    height: u32,
    ptr: *const u8,
    len: usize,
) -> u32 {
    // SAFETY: the caller passes a readable range.
    let rgba = if len == 0 || ptr.is_null() {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len) }
    };
    let done = WORLD.with(|cell| {
        let mut world = cell.borrow_mut();
        let t = world
            .as_mut()
            .and_then(|w| w.textures.get_mut(texture as usize))
            .ok_or_else(|| format!("no texture {texture} in the loaded world"))?;
        if (width, height) != (t.width, t.height)
            || len as u64 != u64::from(width) * u64::from(height) * 4
        {
            return Err(format!(
                "texture {texture}: {width}x{height} pixels ({len} bytes) for a {}x{} image",
                t.width, t.height
            ));
        }
        t.encoded = None;
        t.levels = vec![rgba.to_vec()];
        Ok(())
    });
    match done {
        Ok(()) => 1,
        Err(e) => {
            say(2, &e);
            0
        }
    }
}

fn param<'q>(query: &'q str, key: &str) -> Option<&'q str> {
    query
        .trim_start_matches('?')
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
}

/// Creates the device and the renderer for the loaded world. `look` is the Look file (may be
/// empty) and `query` the page's query string. Returns 1 on success, 0 after logging why not.
///
/// # Safety
/// The pointers must address `look_len` and `query_len` readable bytes (or be dangling with a
/// zero length).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_start(
    look: *const u8,
    look_len: usize,
    query: *const u8,
    query_len: usize,
) -> u32 {
    install_panic_hook();
    // SAFETY: the host passes readable ranges for the duration of the call.
    let (look, query) = unsafe {
        (
            if look_len == 0 {
                &[][..]
            } else {
                std::slice::from_raw_parts(look, look_len)
            },
            if query_len == 0 {
                &[][..]
            } else {
                std::slice::from_raw_parts(query, query_len)
            },
        )
    };
    let query = String::from_utf8_lossy(query).into_owned();
    match start(look, &query) {
        Ok(app) => {
            say(
                0,
                &format!(
                    "Ganbate web player | {} | {} triangles, {} lamps | {}x{}\n{HELP}",
                    app.device.name(),
                    app.renderer.triangles(),
                    app.renderer.lights(),
                    app.renderer.size().0,
                    app.renderer.size().1
                ),
            );
            APP.with(|cell| *cell.borrow_mut() = Some(app));
            1
        }
        Err(e) => {
            say(2, &e);
            0
        }
    }
}

fn start(look: &[u8], query: &str) -> Result<App, String> {
    let world = WORLD
        .with(|cell| cell.borrow_mut().take())
        .ok_or("no world: call ganbate_load_world first")?;
    let file = if look.is_empty() {
        LookFile::default()
    } else {
        LookFile::parse(look).map_err(|e| format!("Look file: {e}"))?
    };
    let mut device = Device::for_canvas()?;
    let quality = if param(query, "quality") == Some("high") {
        Quality::still()
    } else {
        Quality::realtime()
    };
    let (w, h) = device.surface_size();
    let format = device.surface_format();
    let renderer = Renderer::new(
        &mut device,
        &world,
        file.look.clone(),
        quality,
        (w.max(1), h.max(1)),
        Output::Surface(format),
    )?;
    drop(world);
    let views = if file.views.is_empty() {
        let (lo, hi) = renderer.bounds();
        let c = [
            (lo[0] + hi[0]) * 0.5,
            (lo[1] + hi[1]) * 0.5,
            (lo[2] + hi[2]) * 0.5,
        ];
        let r = ((hi[0] - lo[0]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt() * 0.4;
        vec![View {
            name: "overview".into(),
            eye: [c[0] + r, c[1] + r * 0.6, c[2] + r],
            target: c,
            fov: 45.0,
            ..View::default()
        }]
    } else {
        file.views.clone()
    };
    let view = param(query, "view")
        .and_then(|name| views.iter().find(|v| v.name == name))
        .unwrap_or(&views[0])
        .clone();
    let mut time = file.time;
    if let Some(h) = param(query, "time").and_then(|t| t.parse::<f32>().ok()) {
        time.hours = h;
    }
    let mut weather = file.weather;
    let mut weather_name = Weather::PRESETS
        .iter()
        .find(|p| Weather::preset(p) == Some(weather))
        .map_or("custom", |p| p)
        .to_string();
    if let Some(name) = param(query, "weather") {
        weather = Weather::preset(name).ok_or_else(|| format!("unknown weather {name:?}"))?;
        weather_name = name.to_string();
    }
    let seconds = param(query, "seconds")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    Ok(App {
        device,
        renderer,
        fly: Fly::from_view(&view),
        fov: view.fov,
        preset: Some(view),
        views,
        time,
        weather,
        weather_name,
        held: HashSet::new(),
        look: (0.0, 0.0),
        stick: (0.0, 0.0),
        cycle: false,
        paused: param(query, "paused") == Some("1"),
        seconds,
        frames: 0,
        fps: (0, 0.0, 0.0),
        status_age: 1.0,
    })
}

fn with_app<T>(f: impl FnOnce(&mut App) -> T) -> Option<T> {
    APP.with(|cell| cell.borrow_mut().as_mut().map(f))
}

/// One frame, `dt` seconds after the previous one. Returns 1, or 0 after logging an error.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_frame(dt: f32) -> u32 {
    match with_app(|app| app.frame(dt)) {
        Some(Ok(())) => 1,
        Some(Err(e)) => {
            say(2, &e);
            0
        }
        None => 0,
    }
}

/// Runs the GPU-layer self-test (see `selftest.rs`) and presents its image. Call it from a
/// `requestAnimationFrame` callback. Returns 1, or 0 after logging an error.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_selftest() -> u32 {
    install_panic_hook();
    match selftest::run() {
        Ok(report) => {
            say(0, &report);
            1
        }
        Err(e) => {
            say(2, &format!("self-test: {e}"));
            0
        }
    }
}

/// Frames presented so far.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_frames() -> u32 {
    with_app(|app| app.frames as u32).unwrap_or(0)
}

/// A key went down (`down` = 1) or up.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_key(code: u32, down: u32) {
    with_app(|app| app.key(code, down != 0));
}

/// Releases every held key (the page lost focus).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_blur() {
    with_app(|app| {
        app.held.clear();
        app.stick = (0.0, 0.0);
    });
}

/// Turns the camera by a pointer or touch drag of `dx`, `dy` CSS pixels.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_look(dx: f32, dy: f32) {
    with_app(|app| {
        app.look.0 += dx;
        app.look.1 += dy;
    });
}

/// Sets the analog move input (two-finger drag): `forward` and `right` in -1..1.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_move(forward: f32, right: f32) {
    with_app(|app| app.stick = (forward.clamp(-1.0, 1.0), right.clamp(-1.0, 1.0)));
}

impl App {
    fn key(&mut self, code: u32, down: bool) {
        if !down {
            self.held.remove(&code);
            return;
        }
        if !self.held.insert(code) {
            return;
        }
        match char::from_u32(code) {
            Some(d @ '1'..='9') => {
                if let Some(v) = self.views.get(d as usize - '1' as usize) {
                    self.fly = Fly::from_view(v);
                    self.fov = v.fov;
                    self.preset = Some(v.clone());
                }
            }
            Some('y') => {
                let presets = Weather::PRESETS;
                let next = presets
                    .iter()
                    .position(|p| *p == self.weather_name)
                    .map_or(0, |i| (i + 1) % presets.len());
                self.weather_name = presets[next].to_string();
                self.weather = Weather::preset(presets[next]).unwrap_or(self.weather);
                self.status_age = 1.0;
            }
            Some('c') => self.cycle = !self.cycle,
            Some('p' | ' ') => self.paused = !self.paused,
            Some('h') => say(0, HELP),
            _ => {}
        }
    }

    fn frame(&mut self, dt: f32) -> Result<(), String> {
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 0.1)
        } else {
            0.0
        };
        let held = self.held.clone();
        let down = |k: u32| held.contains(&k);
        let speed = if down(SHIFT) { 16.0 } else { 4.0 } * dt;
        let f = self.fly.forward();
        let flat = math::normalize([f[0], 0.0, f[2]]);
        let right = math::normalize(math::cross(flat, [0.0, 1.0, 0.0]));
        let mut step = [0.0f32; 3];
        for (k, dir) in [
            ('w', flat),
            ('s', math::scale(flat, -1.0)),
            ('d', right),
            ('a', math::scale(right, -1.0)),
            ('e', [0.0, 1.0, 0.0]),
            ('q', [0.0, -1.0, 0.0]),
        ] {
            if down(k as u32) {
                step = math::add(step, dir);
            }
        }
        step = math::add(step, math::scale(flat, self.stick.0));
        step = math::add(step, math::scale(right, self.stick.1));
        let mut moved = step != [0.0; 3];
        self.fly.pos = math::add(self.fly.pos, math::scale(step, speed));
        let turn = 1.3 * dt;
        for (k, dyaw, dpitch) in [
            (LEFT, -turn, 0.0),
            (RIGHT, turn, 0.0),
            (UP, 0.0, turn),
            (DOWN, 0.0, -turn),
        ] {
            if down(k) {
                self.fly.yaw += dyaw;
                self.fly.pitch = (self.fly.pitch + dpitch).clamp(-1.5, 1.5);
                moved = true;
            }
        }
        if self.look != (0.0, 0.0) {
            self.fly.yaw += self.look.0 * 0.004;
            self.fly.pitch = (self.fly.pitch - self.look.1 * 0.004).clamp(-1.5, 1.5);
            self.look = (0.0, 0.0);
            moved = true;
        }
        if moved {
            self.preset = None;
        }
        if down('t' as u32) {
            self.time.hours = (self.time.hours + 1.5 * dt).rem_euclid(24.0);
        }
        if down('g' as u32) {
            self.time.hours = (self.time.hours - 1.5 * dt).rem_euclid(24.0);
        }
        if self.cycle {
            self.time.hours = (self.time.hours + dt * 24.0 / 120.0).rem_euclid(24.0);
        }
        if !self.paused {
            self.seconds += dt;
        }
        let env = Environment::resolve(
            &self.time,
            &self.weather,
            &self.renderer.look().palette,
            self.seconds,
        );
        let Some(target) = self.device.acquire()? else {
            return Ok(());
        };
        let size = self.device.surface_size();
        if size != self.renderer.size() {
            self.renderer.resize(&mut self.device, size)?;
        }
        let view = match &self.preset {
            Some(v) => v.clone(),
            None => self.fly.view(self.fov),
        };
        let mut enc = self.device.encoder();
        // A constant noise seed: without temporal filtering a per-frame seed flickers.
        self.renderer.encode(
            &mut self.device,
            &mut enc,
            &view,
            &env,
            [0.0, 0.0],
            0,
            Some(target),
        )?;
        self.device.present(enc)?;
        self.frames += 1;
        self.fps.0 += 1;
        self.fps.1 += dt;
        if self.fps.1 >= 0.5 {
            self.fps.2 = self.fps.0 as f32 / self.fps.1;
            self.fps = (0, 0.0, self.fps.2);
        }
        self.status_age += dt;
        if self.status_age >= 0.25 {
            self.status_age = 0.0;
            let text = format!(
                "{} · {}{}{} · {:.0} fps · {} · {}x{}",
                clock(self.time.hours),
                self.weather_name,
                if self.cycle { " · cycle" } else { "" },
                if self.paused { " · paused" } else { "" },
                self.fps.2,
                self.preset.as_ref().map_or("free", |v| v.name.as_str()),
                size.0,
                size.1
            );
            // SAFETY: the host reads the UTF-8 text during the call.
            unsafe { status(text.as_ptr(), text.len()) };
        }
        Ok(())
    }
}
