//! Render stills of a cooked scene with Ganbate's unified stylized & photoreal renderer.
//!
//! ganbate-render --scene world.gbew [--look look.json] [--view NAME | --all-views]
//!            [--time HOURS] [--weather PRESET] [--width W] [--height H] [--samples N]
//!            [--seconds S] [--out FILE.png | --out-dir DIR]
//!            [--realtime-quality | --hybrid-quality]
//!            [--surface MODEL] [--tonemap stylized|agx|aces] [--ev100 EV]
//!            [--sequence START_HOURS END_HOURS FRAMES]   (a time-lapse of one view, numbered
//!                                                         frames in --out-dir)
use ganbate_engine::cli::{Args, load_look_or_default, load_world, overview_view};
use ganbate_engine::gpu::Device;
use ganbate_engine::render::{
    Environment, Output, Quality, Renderer, SurfaceModel, ToneMap, View, Weather,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("ganbate-render: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = Args::from_env();
    let mut scene = None;
    let mut look_path = None;
    let mut view_name = None;
    let mut all_views = false;
    let mut hours = None;
    let mut weather = None;
    let (mut width, mut height, mut samples) = (1920u32, 1080u32, 8u32);
    let mut seconds = 12.0f32;
    let mut out = PathBuf::from("render.png");
    let mut out_dir: Option<PathBuf> = None;
    let mut realtime = false;
    let mut hybrid = false;
    let mut surface_override = None;
    let mut tonemap_override = None;
    let mut ev100_override = None;
    let mut sequence: Option<(f32, f32, u32)> = None;
    while let Some(a) = args.next_arg() {
        match a.as_str() {
            "--scene" => scene = Some(args.path(&a)?),
            "--look" => look_path = Some(args.path(&a)?),
            "--view" => view_name = Some(args.value(&a)?),
            "--all-views" => all_views = true,
            "--time" => hours = Some(args.parsed::<f32>(&a)?),
            "--weather" => weather = Some(args.value(&a)?),
            "--width" => width = args.parsed(&a)?,
            "--height" => height = args.parsed(&a)?,
            "--samples" => samples = args.parsed(&a)?,
            "--seconds" => seconds = args.parsed(&a)?,
            "--out" => out = args.path(&a)?,
            "--out-dir" => out_dir = Some(args.path(&a)?),
            "--realtime-quality" => realtime = true,
            "--hybrid-quality" => hybrid = true,
            "--surface" => {
                let s = args.value(&a)?;
                surface_override = Some(
                    SurfaceModel::parse(&s)
                        .ok_or_else(|| format!("--surface: unknown surface model {s:?}"))?,
                );
            }
            "--tonemap" => {
                let s = args.value(&a)?;
                tonemap_override = Some(
                    ToneMap::parse(&s)
                        .ok_or_else(|| format!("--tonemap: unknown tonemap {s:?}"))?,
                );
            }
            "--ev100" => ev100_override = Some(args.parsed::<f32>(&a)?),
            "--sequence" => {
                let start: f32 = args.parsed("--sequence start")?;
                let end: f32 = args.parsed("--sequence end")?;
                let frames: u32 = args.parsed("--sequence frames")?;
                sequence = Some((start, end, frames.max(2)));
            }
            _ => return Err(format!("unknown argument {a}")),
        }
    }
    let scene = scene.ok_or("--scene is required")?;
    if sequence.is_some() && (all_views || out_dir.is_none()) {
        return Err("--sequence renders one view (--view NAME) into --out-dir".into());
    }
    let started = Instant::now();
    let world = load_world(&scene)?;
    let mut file = load_look_or_default(look_path.as_deref())?;
    if let Some(h) = hours {
        file.time.hours = h;
    }
    if let Some(w) = &weather {
        file.weather = Weather::preset(w)
            .ok_or_else(|| format!("unknown weather {w:?}; try {:?}", Weather::PRESETS))?;
    }
    if let Some(s) = surface_override {
        file.look.surface_model = s;
    }
    if let Some(tm) = tonemap_override {
        file.look.tonemap = tm;
    }
    if let Some(ev) = ev100_override {
        file.look.ev100 = Some(ev);
    }
    let mut device = Device::headless()?;
    // Offline stills on slow (software) devices can take minutes per submission.
    device.set_submit_timeout(std::time::Duration::from_secs(1800));
    let quality = if hybrid {
        Quality::hybrid_stage()
    } else if realtime {
        Quality::realtime()
    } else {
        Quality::still()
    };
    let mut renderer = Renderer::new(
        &mut device,
        &world,
        file.look.clone(),
        quality,
        (width, height),
        Output::Still,
    )?;
    eprintln!(
        "{}: {} triangles, {} lights, water at {:?}, loaded in {:.1}s on {}",
        scene.display(),
        renderer.triangles(),
        renderer.lights(),
        renderer.water_level(),
        started.elapsed().as_secs_f32(),
        device.name()
    );
    let views: Vec<View> = if all_views {
        file.views.clone()
    } else if let Some(name) = &view_name {
        vec![
            file.views
                .iter()
                .find(|v| &v.name == name)
                .cloned()
                .ok_or_else(|| format!("no view named {name:?}"))?,
        ]
    } else if let Some(v) = file.views.first() {
        vec![v.clone()]
    } else {
        vec![overview_view(renderer.bounds(), 45.0)]
    };
    if let Some((start, end, frames)) = sequence {
        let dir = out_dir.clone().ok_or("--sequence needs --out-dir")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let view = &views[0];
        for i in 0..frames {
            let t = Instant::now();
            let mut time = file.time;
            time.hours = start + (end - start) * i as f32 / (frames - 1) as f32;
            // Motion (water, clouds, rain, petals) runs at 24 frames a second.
            let env = Environment::resolve(
                &time,
                &file.weather,
                &file.look.palette,
                seconds + i as f32 / 24.0,
            );
            let pixels = renderer.render_still(&mut device, view, &env, samples)?;
            let png = ganbate_engine::assets::png::encode(width, height, &pixels)?;
            write(&dir.join(format!("{}-{i:05}.png", view.name)), &png)?;
            eprintln!(
                "frame {}/{frames} at {:.2}h in {:.1}s",
                i + 1,
                time.hours,
                t.elapsed().as_secs_f32()
            );
        }
        return Ok(());
    }
    let env = Environment::resolve(&file.time, &file.weather, &file.look.palette, seconds);
    for view in &views {
        let t = Instant::now();
        let pixels = renderer.render_still(&mut device, view, &env, samples)?;
        let png = ganbate_engine::assets::png::encode(width, height, &pixels)?;
        let path = match &out_dir {
            Some(dir) => {
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                let weather = weather.as_deref().unwrap_or("default");
                dir.join(format!(
                    "{}-{:04.1}h-{weather}.png",
                    view.name, file.time.hours
                ))
            }
            None if views.len() > 1 => {
                let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("render");
                out.with_file_name(format!("{stem}-{}.png", view.name))
            }
            None => out.clone(),
        };
        write(&path, &png)?;
        eprintln!(
            "{} ({}x{}, {} samples) in {:.1}s -> {}",
            view.name,
            width,
            height,
            samples,
            t.elapsed().as_secs_f32(),
            path.display()
        );
    }
    let errors = device.validation_errors();
    if !errors.is_empty() {
        return Err(format!(
            "{} validation errors, first: {}",
            errors.len(),
            errors[0]
        ));
    }
    Ok(())
}

fn write(path: &Path, data: &[u8]) -> Result<(), String> {
    ganbate_engine::assets::write_asset(path, data).map_err(|e| format!("{}: {e}", path.display()))
}
