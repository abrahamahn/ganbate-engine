//! Minimal standalone player. The example scene is also used by the offscreen oracle.
#[cfg(any(target_os = "linux", target_os = "windows"))]
#[path = "../../examples/common/gallery.rs"]
mod demo;

#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (mut limit, mut scene_path) = (None, None);
    let mut look_path: Option<String> = None;
    let mut high = false;
    let mut size = (1600u32, 900u32);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--look" => look_path = Some(args.next().ok_or("missing look file path")?),
            "--high" => high = true,
            "--size" => {
                let v = args.next().ok_or("missing WxH")?;
                let (w, h) = v.split_once('x').ok_or("size must be WxH")?;
                size = (w.parse()?, h.parse()?);
            }
            "--frames" => {
                let n = args.next().ok_or("missing frame count")?.parse::<u64>()?;
                if n == 0 {
                    return Err("frame count must be positive".into());
                }
                limit = Some(n);
            }
            "--scene" => scene_path = Some(args.next().ok_or("missing cooked scene path")?),
            "--help" => {
                println!(
                    "ganbate-player [--frames N] [--scene world.gbew] [--look look.json] [--high] [--size WxH]\n\
                     Without a Look file or materials: Space or click pauses, arrows orbit, Esc closes.\n{}",
                    stylized::HELP
                );
                return Ok(());
            }
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    if let Some(p) = &scene_path {
        let world = ganbate_engine::assets::cooked::decode(&ganbate_engine::assets::read_asset(
            std::path::Path::new(p),
        )?)?;
        if look_path.is_some() || !world.materials.is_empty() {
            return stylized::run(world, look_path.as_deref(), limit, high, size, p);
        }
    }
    observatory(scene_path, limit)
}

/// The built-in sculpture (or an unlit scene) through the Vulkan presenter.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn observatory(
    scene_path: Option<String>,
    limit: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    use ganbate_engine::{
        Camera, Vec3,
        vulkan::present::{DesktopRenderer, PresentStatus},
        window::{Event, Key, Window},
    };
    use std::time::{Duration, Instant};
    let asset = scene_path
        .map(|p| -> Result<_, String> {
            let world = ganbate_engine::assets::cooked::decode(
                &ganbate_engine::assets::read_asset(std::path::Path::new(&p))?,
            )?;
            let scene = world.scene()?;
            let bounds = scene.bounds().ok_or("static scene has no finite bounds")?;
            Ok((scene, bounds))
        })
        .transpose()?;
    let window = Window::new("Ganbate Player", 800, 480)?;
    let mut gpu = DesktopRenderer::new(&window)?;
    println!(
        "Ganbate Player | {} | space/click: pause | arrows: orbit | Esc: close",
        gpu.device_name()
    );
    let mut presenter = gpu.presenter();
    let mut bounds = None;
    if let Some((scene, sphere)) = asset {
        presenter.set_static_scene(&scene)?;
        bounds = Some(sphere);
    }
    let mut last = Instant::now();
    let mut animation = 0.45f32;
    let mut angle = 0.4f32;
    let mut paused = false;
    'player: loop {
        while let Some(event) = window.poll() {
            match event {
                Event::Close | Event::Pressed(Key::Escape) => break 'player,
                Event::Pressed(Key::Space) | Event::Clicked(_, _) => paused = !paused,
                Event::Pressed(Key::Left) => angle -= 0.15,
                Event::Pressed(Key::Right) => angle += 0.15,
                _ => {}
            }
        }
        let now = Instant::now();
        if !paused && window.visible() {
            animation += (now - last).as_secs_f32().min(0.1) * 0.65;
        }
        last = now;
        let camera = Camera {
            eye: Vec3(angle.sin() * 5.7, 2.0, angle.cos() * 5.7),
            target: Vec3(0.0, 0.15, 0.0),
            vertical_fov_degrees: 43.0,
            near: 0.1,
            far: 60.0,
        };
        let clear = [0.03, 0.045, 0.07, 1.0];
        let status = if let Some(bounds) = bounds {
            presenter.draw_static(Camera::orbit(bounds, angle), clear)?
        } else {
            presenter.draw(&demo::gallery(animation), camera, clear)?
        };
        if limit.is_some_and(|n| presenter.stats().frames >= n) {
            break;
        }
        if status != PresentStatus::Presented {
            std::thread::sleep(Duration::from_millis(12));
        }
    }
    println!("Presentation {:?}", presenter.stats());
    Ok(())
}

#[cfg(target_os = "macos")]
fn observatory(_: Option<String>, _: Option<u64>) -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "on macOS the player runs the stylized renderer on Metal: pass --scene world.gbew \
         (a scene with materials, or add --look look.json); the built-in observatory needs \
         the Vulkan presenter"
            .into(),
    )
}

/// The stylized renderer in a window: fly around, scrub the time of day, change the weather.
#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
mod stylized {
    use ganbate_engine::gpu::Device;
    use ganbate_engine::render::{
        Environment, LookFile, Output, Quality, Renderer, TimeOfDay, View, Weather, math,
    };
    use ganbate_engine::window::{Event, Key, Window};
    use std::collections::HashSet;
    use std::time::{Duration, Instant};

    pub const HELP: &str = "With a Look file: W/A/S/D fly, Q/E down/up, arrows or left-drag look, \
Shift faster, 1-9 views, T/G time +/-, C day cycle, Y next weather, P pause motion, H help, Esc quit.";

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

    pub fn run(
        world: ganbate_engine::world::World,
        look_path: Option<&str>,
        limit: Option<u64>,
        high: bool,
        size: (u32, u32),
        scene_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = match look_path {
            Some(p) => LookFile::parse(&ganbate_engine::assets::read_asset(
                std::path::Path::new(p),
            )?)?,
            None => LookFile::default(),
        };
        let window = Window::new("Ganbate Player", size.0, size.1)?;
        let mut device = Device::for_window(&window)?;
        let quality = if high {
            Quality::still()
        } else {
            Quality::realtime()
        };
        let (surface_size, surface_format) = (device.surface_size(), device.surface_format());
        let mut renderer = Renderer::new(
            &mut device,
            &world,
            file.look.clone(),
            quality,
            surface_size,
            Output::Surface(surface_format),
        )?;
        let views = if file.views.is_empty() {
            let (lo, hi) = renderer.bounds();
            let c = [
                (lo[0] + hi[0]) * 0.5,
                (lo[1] + hi[1]) * 0.5,
                (lo[2] + hi[2]) * 0.5,
            ];
            vec![View {
                name: "overview".into(),
                eye: [c[0] + 40.0, c[1] + 25.0, c[2] + 40.0],
                target: c,
                fov: 50.0,
                ..View::default()
            }]
        } else {
            file.views.clone()
        };
        let mut fly = Fly::from_view(&views[0]);
        let mut fov = views[0].fov;
        let mut time: TimeOfDay = file.time;
        let presets = Weather::PRESETS;
        let mut weather_index = 0usize;
        let mut weather: Weather = file.weather;
        let mut held: HashSet<Key> = HashSet::new();
        let (mut dragging, mut last_pointer) = (false, None::<(i32, i32)>);
        let (mut cycle, mut paused) = (false, false);
        let mut seconds = 0.0f32;
        let mut frames = 0u64;
        let (mut fps_frames, mut fps_since, mut fps) = (0u32, Instant::now(), 0.0f32);
        let mut last = Instant::now();
        println!(
            "Ganbate Player | {} | {} | {} triangles, {} lamps\n{HELP}",
            device.name(),
            scene_name,
            renderer.triangles(),
            renderer.lights()
        );
        'frames: loop {
            while let Some(event) = window.poll() {
                match event {
                    Event::Close | Event::Pressed(Key::Escape) => break 'frames,
                    Event::Pressed(k) => {
                        if !held.insert(k) {
                            continue;
                        }
                        match k {
                            Key::Char(d @ b'1'..=b'9') => {
                                if let Some(v) = views.get((d - b'1') as usize) {
                                    fly = Fly::from_view(v);
                                    fov = v.fov;
                                }
                            }
                            Key::Char(b'y') => {
                                weather_index = (weather_index + 1) % presets.len();
                                weather =
                                    Weather::preset(presets[weather_index]).unwrap_or(weather);
                            }
                            Key::Char(b'c') => cycle = !cycle,
                            Key::Char(b'p') | Key::Space => paused = !paused,
                            Key::Char(b'h') => println!("{HELP}"),
                            _ => {}
                        }
                    }
                    Event::Released(k) => {
                        held.remove(&k);
                    }
                    Event::Clicked(x, y) => {
                        dragging = true;
                        last_pointer = Some((x, y));
                    }
                    Event::Unclicked(..) => dragging = false,
                    Event::Focus(false) => {
                        // Releases that happen while unfocused never arrive.
                        held.clear();
                        dragging = false;
                    }
                    Event::Pointer(x, y) => {
                        if dragging {
                            if let Some((px, py)) = last_pointer {
                                fly.yaw += (x - px) as f32 * 0.004;
                                fly.pitch = (fly.pitch - (y - py) as f32 * 0.004).clamp(-1.5, 1.5);
                            }
                        }
                        last_pointer = Some((x, y));
                    }
                    _ => {}
                }
            }
            let now = Instant::now();
            let dt = (now - last).as_secs_f32().min(0.1);
            last = now;
            let down = |k: Key| held.contains(&k);
            let speed = if down(Key::Shift) { 16.0 } else { 4.0 } * dt;
            let f = fly.forward();
            let flat = math::normalize([f[0], 0.0, f[2]]);
            let right = math::normalize(math::cross(flat, [0.0, 1.0, 0.0]));
            let mut step = [0.0f32; 3];
            for (k, dir) in [
                (b'w', flat),
                (b's', math::scale(flat, -1.0)),
                (b'd', right),
                (b'a', math::scale(right, -1.0)),
                (b'e', [0.0, 1.0, 0.0]),
                (b'q', [0.0, -1.0, 0.0]),
            ] {
                if down(Key::Char(k)) {
                    step = math::add(step, dir);
                }
            }
            fly.pos = math::add(fly.pos, math::scale(step, speed));
            let turn = 1.3 * dt;
            if down(Key::Left) {
                fly.yaw -= turn;
            }
            if down(Key::Right) {
                fly.yaw += turn;
            }
            if down(Key::Up) {
                fly.pitch = (fly.pitch + turn).min(1.5);
            }
            if down(Key::Down) {
                fly.pitch = (fly.pitch - turn).max(-1.5);
            }
            if down(Key::Char(b't')) {
                time.hours = (time.hours + 1.5 * dt).rem_euclid(24.0);
            }
            if down(Key::Char(b'g')) {
                time.hours = (time.hours - 1.5 * dt).rem_euclid(24.0);
            }
            if cycle {
                time.hours = (time.hours + dt * 24.0 / 120.0).rem_euclid(24.0);
            }
            if !paused {
                seconds += dt;
            }
            if !window.visible() {
                std::thread::sleep(Duration::from_millis(30));
                continue;
            }
            let env = Environment::resolve(&time, &weather, &renderer.look().palette, seconds);
            let Some(frame) = device.acquire()? else {
                std::thread::sleep(Duration::from_millis(16));
                continue;
            };
            let surface = device.surface_size();
            if surface != renderer.size() {
                renderer.resize(&mut device, surface)?;
            }
            let mut enc = device.encoder();
            renderer.encode(
                &mut device,
                &mut enc,
                &fly.view(fov),
                &env,
                [0.0, 0.0],
                // A constant noise seed: without temporal filtering a per-frame seed flickers.
                0,
                Some(frame),
            )?;
            device.present(enc)?;
            frames += 1;
            fps_frames += 1;
            let since = fps_since.elapsed().as_secs_f32();
            if since >= 0.5 {
                fps = fps_frames as f32 / since;
                fps_frames = 0;
                fps_since = Instant::now();
                window.set_title(&format!(
                    "Ganbate Player | {} | {} | {:.0} fps | {}",
                    clock(time.hours),
                    presets
                        .iter()
                        .find(|p| Weather::preset(p) == Some(weather))
                        .unwrap_or(&"custom"),
                    fps,
                    device.name()
                ));
            }
            if limit.is_some_and(|n| frames >= n) {
                break;
            }
        }
        let errors = device.validation_errors();
        println!(
            "{frames} frames, last {fps:.1} fps, {} validation errors",
            errors.len()
        );
        if let Some(e) = errors.first() {
            return Err(e.clone().into());
        }
        Ok(())
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
fn main() {
    eprintln!("This version of the presented player requires Windows, Linux/X11 or macOS.");
    std::process::exit(1);
}
