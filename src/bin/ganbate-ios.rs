//! The stylized renderer as an iPhone app. It loads `world.gbew` and, if present, `look.json`
//! from the app bundle (`scripts/apple/build-ios.sh` copies them in) and renders on Metal.
//!
//! Touch controls: drag one finger to look around, drag two fingers to move (up/down moves
//! forward/back, sideways strafes). Buttons step the time of day by an hour, cycle the
//! weather presets and cycle the Look file's camera views.
#[cfg(target_os = "ios")]
fn main() {
    ganbate_engine::window::ios::run(app::setup)
}

#[cfg(not(target_os = "ios"))]
fn main() {
    eprintln!(
        "ganbate-ios is the iPhone app: build and install it on a Mac with scripts/apple/build-ios.sh"
    );
    std::process::exit(1);
}

#[cfg(target_os = "ios")]
mod app {
    use ganbate_engine::cli::{FlyCamera, load_look_or_default, load_world, views_or_overview};
    use ganbate_engine::gpu::Device;
    use ganbate_engine::render::{
        Environment, Output, Quality, Renderer, TimeOfDay, View, Weather, math,
    };
    use ganbate_engine::window::ios::{Input, TouchPhase, Window};
    use std::time::Instant;

    /// Radians of turn per pixel of one-finger drag.
    const LOOK: f32 = 0.0025;
    /// Metres of travel per pixel of two-finger drag.
    const MOVE: f32 = 0.02;

    struct App {
        window: &'static Window,
        device: Device<'static>,
        renderer: Renderer,
        views: Vec<View>,
        view_index: usize,
        fly: FlyCamera,
        fov: f32,
        time: TimeOfDay,
        weather_index: usize,
        weather: Weather,
        /// Fingers on the screen: (touch id, x, y) in pixels.
        touches: Vec<(usize, f32, f32)>,
        buttons: [usize; 3],
        started: Instant,
    }

    pub fn setup(window: &'static Window) -> Box<dyn FnMut()> {
        match App::new(window) {
            Ok(mut app) => {
                let mut failed = false;
                Box::new(move || {
                    if failed {
                        return;
                    }
                    if let Err(e) = app.frame() {
                        // Stop at the first error instead of repeating it every refresh.
                        eprintln!("ganbate-ios: stopped rendering: {e}");
                        failed = true;
                    }
                })
            }
            Err(e) => {
                eprintln!("ganbate-ios: cannot start: {e}");
                Box::new(|| {})
            }
        }
    }

    impl App {
        fn new(window: &'static Window) -> Result<Self, String> {
            let world_path = Window::resource("world.gbew")
                .ok_or("world.gbew is missing from the app bundle (see build-ios.sh)")?;
            let world = load_world(&world_path)?;
            let look_res = Window::resource("look.json");
            let file = load_look_or_default(look_res.as_deref())?;
            let mut device = Device::for_window(window)?;
            let (size, format) = (device.surface_size(), device.surface_format());
            let renderer = Renderer::new(
                &mut device,
                &world,
                file.look.clone(),
                Quality::realtime(),
                size,
                Output::Surface(format),
            )?;
            let views = views_or_overview(&file, renderer.bounds(), 50.0);
            let buttons = [
                window.add_button("Time +1h"),
                window.add_button("Weather"),
                window.add_button("View"),
            ];
            eprintln!(
                "ganbate-ios: {} | {} triangles, {} lamps",
                device.name(),
                renderer.triangles(),
                renderer.lights()
            );
            Ok(Self {
                window,
                fly: FlyCamera::from_view(&views[0]),
                fov: views[0].fov,
                views,
                view_index: 0,
                time: file.time,
                weather_index: 0,
                weather: file.weather,
                device,
                renderer,
                touches: Vec::new(),
                buttons,
                started: Instant::now(),
            })
        }

        fn input(&mut self, input: Input) {
            match input {
                Input::Touch {
                    id,
                    x,
                    y,
                    phase: TouchPhase::Began,
                } => {
                    self.touches.retain(|t| t.0 != id);
                    self.touches.push((id, x, y));
                }
                Input::Touch {
                    id,
                    phase: TouchPhase::Ended,
                    ..
                } => self.touches.retain(|t| t.0 != id),
                Input::Touch {
                    id,
                    x,
                    y,
                    phase: TouchPhase::Moved,
                } => {
                    let fingers = self.touches.len();
                    let Some(t) = self.touches.iter_mut().find(|t| t.0 == id) else {
                        return;
                    };
                    let (dx, dy) = (x - t.1, y - t.2);
                    (t.1, t.2) = (x, y);
                    if fingers == 1 {
                        self.fly.yaw += dx * LOOK;
                        self.fly.pitch = (self.fly.pitch - dy * LOOK).clamp(-1.5, 1.5);
                    } else {
                        // Each finger moves the camera by half its own drag.
                        let f = self.fly.forward();
                        let flat = math::normalize([f[0], 0.0, f[2]]);
                        let right = math::normalize(math::cross(flat, [0.0, 1.0, 0.0]));
                        let share = MOVE / fingers as f32;
                        let step = math::add(
                            math::scale(flat, -dy * share),
                            math::scale(right, dx * share),
                        );
                        self.fly.pos = math::add(self.fly.pos, step);
                    }
                }
                Input::Button(b) if b == self.buttons[0] => {
                    self.time.hours = (self.time.hours + 1.0).rem_euclid(24.0);
                }
                Input::Button(b) if b == self.buttons[1] => {
                    let presets = Weather::PRESETS;
                    self.weather_index = (self.weather_index + 1) % presets.len();
                    self.weather =
                        Weather::preset(presets[self.weather_index]).unwrap_or(self.weather);
                    self.window
                        .set_button_title(self.buttons[1], presets[self.weather_index]);
                }
                Input::Button(b) if b == self.buttons[2] => {
                    self.view_index = (self.view_index + 1) % self.views.len();
                    let v = &self.views[self.view_index];
                    self.fly = FlyCamera::from_view(v);
                    self.fov = v.fov;
                    self.window.set_button_title(self.buttons[2], &v.name);
                }
                _ => {}
            }
        }

        fn frame(&mut self) -> Result<(), String> {
            while let Some(i) = self.window.poll_input() {
                self.input(i);
            }
            if !self.window.visible() {
                return Ok(());
            }
            let seconds = self.started.elapsed().as_secs_f32();
            let env = Environment::resolve(
                &self.time,
                &self.weather,
                &self.renderer.look().palette,
                seconds,
            );
            let Some(frame) = self.device.acquire()? else {
                return Ok(());
            };
            let surface = self.device.surface_size();
            if surface != self.renderer.size() {
                self.renderer.resize(&mut self.device, surface)?;
            }
            let mut enc = self.device.encoder();
            self.renderer.encode(
                &mut self.device,
                &mut enc,
                &self.fly.view(self.fov),
                &env,
                [0.0, 0.0],
                0,
                Some(frame),
            )?;
            self.device.present(enc)
        }
    }
}
