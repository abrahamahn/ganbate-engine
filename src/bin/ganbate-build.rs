//! Offline asset cooker and inspector.
use ganbate_engine::{
    assets::{cooked, glb, read_asset},
    world::{AlphaMode, Emission, Shading, World},
};
use ganbate_pipeline::project::{Kind, MANIFEST, Project};
use std::{ffi::OsString, path::Path};

fn summary(w: &World) -> Result<String, String> {
    let texels: u64 = w
        .textures
        .iter()
        .flat_map(|t| &t.levels)
        .map(|l| l.len() as u64)
        .sum();
    let encoded: u64 = w
        .textures
        .iter()
        .filter_map(|t| t.encoded.as_ref())
        .map(|e| e.data.len() as u64)
        .sum();
    let images = if encoded > 0 {
        format!(", {encoded} encoded image bytes")
    } else {
        String::new()
    };
    Ok(format!(
        "{} meshes | {} instances | {} triangles | {} materials | {} textures ({texels} texel bytes{images}) | {} lights",
        w.meshes.len(),
        w.instances.len(),
        w.validate()?,
        w.materials.len(),
        w.textures.len(),
        w.lights.len(),
    ))
}

/// What `dest` is, if it is a file of project `p` that already exists (its manifest, or anything
/// in its content, schema or editor folders): a cook must not replace authored files.
fn project_file(p: &Project, dest: &Path) -> Option<String> {
    let dest = dest.canonicalize().ok()?;
    if p.root
        .join(MANIFEST)
        .canonicalize()
        .is_ok_and(|m| m == dest)
    {
        return Some("the project's manifest".into());
    }
    let m = &p.manifest;
    for (what, folders) in [
        ("content", &m.content),
        ("schema", &m.schemas),
        ("editor", &m.editor),
    ] {
        for f in folders {
            if p.root
                .join(f)
                .canonicalize()
                .is_ok_and(|d| dest.starts_with(d))
            {
                return Some(format!("a file in the project's {what} folder {f}"));
            }
        }
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<OsString> = std::env::args_os().skip(1).collect();
    // Cook options: --compact quantizes geometry, --keep-images keeps PNG, JPEG and WebP
    // images encoded (for hosts that decode them, such as browsers). --force lets import and
    // cook-scene replace a file of the project that already exists.
    let mut flag = |name: &str| {
        let before = args.len();
        args.retain(|a| a != name);
        args.len() != before
    };
    let (compact, keep_images, force) = (flag("--compact"), flag("--keep-images"), flag("--force"));
    match args.as_slice() {
        [cmd, source, dest] if cmd == "cook" => {
            let w = glb::import_with(&read_asset(Path::new(source))?, keep_images)?;
            let data = if compact {
                cooked::encode_compact(&w)?
            } else {
                cooked::encode(&w)?
            };
            if Path::new(dest).canonicalize().ok() == Some(Path::new(source).canonicalize()?) {
                return Err("cook destination must differ from the source".into());
            }
            ganbate_engine::assets::write_asset(Path::new(dest), &data)?;
            println!("{} | {} bytes", summary(&w)?, data.len());
        }
        [cmd, file] if cmd == "model" => {
            // A run-time model as `createEngine().loadModel` reads it.
            let data = read_asset(Path::new(file))?;
            let m = ganbate_engine::assets::model::load_glb(&data)?;
            let prims = m.meshes.iter().flat_map(|x| &x.primitives);
            let triangles: usize = prims.clone().map(|p| p.indices.len() / 3).sum();
            let vertices: usize = prims.map(|p| p.positions.len()).sum();
            let image_bytes: usize = m.images.iter().map(|i| i.data.len()).sum();
            println!(
                "model | {} nodes | {} meshes | {vertices} vertices | {triangles} triangles | {} materials | {} textures ({image_bytes} image bytes) | {} skins | {} clips | {} bytes",
                m.nodes.len(),
                m.meshes.len(),
                m.materials.len(),
                m.textures.len(),
                m.skins.len(),
                m.clips.len(),
                data.len(),
            );
            for (i, mesh) in m.meshes.iter().enumerate() {
                for (j, p) in mesh.primitives.iter().enumerate() {
                    let material = p
                        .material
                        .map_or("none".to_string(), |k| m.materials[k].name.clone());
                    let mut traits = vec![format!("material {material}")];
                    if !p.uvs.is_empty() {
                        traits.push("uv".into());
                    }
                    if !p.joints.is_empty() {
                        traits.push("skinned".into());
                    }
                    if !p.targets.is_empty() {
                        traits.push(format!("{} targets", p.targets.len()));
                    }
                    println!(
                        "  mesh {i} {} primitive {j}: {} vertices, {} triangles [{}]",
                        mesh.name,
                        p.positions.len(),
                        p.indices.len() / 3,
                        traits.join(", ")
                    );
                }
            }
            for c in &m.clips {
                println!(
                    "  clip {}: {:.2} s, {} channels",
                    c.name,
                    c.duration,
                    c.channels.len()
                );
            }
        }
        [cmd, project, scene, dest] if cmd == "cook-scene" => {
            // A scene of a project (by asset id or path) into a world that keeps its entities.
            let mut p = Project::open(Path::new(project))?;
            if let Some(what) = project_file(&p, Path::new(dest)).filter(|_| !force) {
                return Err(format!(
                    "{} is {what}; write the cooked scene elsewhere (such as build/), or pass --force to replace it",
                    Path::new(dest).display()
                )
                .into());
            }
            p.scan(None)?;
            let s = scene.to_str().ok_or("scene is not UTF-8")?;
            let id = if ganbate_engine::id::is_asset_id(s) {
                s.to_string()
            } else {
                let rel = s.replace('\\', "/");
                p.find_path(&rel)
                    .ok_or_else(|| format!("no scene {s} in the project"))?
                    .id
                    .clone()
            };
            let mut registry = ganbate_scene::schema::engine_registry();
            for (path, file) in p.schema_files()? {
                registry
                    .add(&file, false)
                    .map_err(|e| format!("{path}: {e}"))?;
            }
            let cooked = ganbate_pipeline::cook::cook_scene(&p, &registry, &id, keep_images)?;
            for w in &cooked.warnings {
                eprintln!("warning: {w}");
            }
            let data = cooked::encode_scene(&cooked.world, &cooked.table, compact)?;
            if let Some(dir) = Path::new(dest)
                .parent()
                .filter(|d| !d.as_os_str().is_empty())
            {
                std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            }
            ganbate_engine::assets::write_asset(Path::new(dest), &data)?;
            println!(
                "{} | {} entities | {} bytes",
                summary(&cooked.world)?,
                cooked.table.entities.len(),
                data.len()
            );
        }
        [cmd, project, model, dest] if cmd == "import" => {
            // A model of a project as a new scene of separate objects (one entity per node).
            let mut p = Project::open(Path::new(project))?;
            let mut ids = ganbate_engine::id::Ids::from_os();
            p.scan(Some(&mut ids))?;
            let m = model
                .to_str()
                .ok_or("model is not UTF-8")?
                .replace('\\', "/");
            let entry = p
                .find_path(&m)
                .filter(|e| e.kind == Kind::Model)
                .ok_or_else(|| format!("no model {m} in the project"))?
                .clone();
            let rel = dest
                .to_str()
                .ok_or("destination is not UTF-8")?
                .replace('\\', "/");
            // Checked before the import, which can take a while.
            p.check_document_path(&rel, Kind::Scene)?;
            if !force && p.root.join(&rel).exists() {
                return Err(format!("{rel} already exists; pass --force to replace it").into());
            }
            let bytes = read_asset(&p.root.join(&entry.path))?;
            let nodes = ganbate_engine::assets::glb::import_nodes(&bytes, keep_images)?;
            let name = Path::new(&rel)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("scene")
                .trim_end_matches(".scene.json")
                .to_string();
            let scene_id = ids.asset();
            let doc = ganbate_pipeline::cook::scene_from_model(
                &nodes, &entry.id, &scene_id, &name, &mut ids,
            );
            p.write_document_with(&rel, Kind::Scene, &doc, force)?;
            println!("{rel}: {} entities from {}", nodes.nodes.len(), entry.path);
        }
        [cmd, file, out] if cmd == "bake-ao" => {
            // The world's vertex AO, baked here once so runtimes load it instead of baking it.
            let data = read_asset(Path::new(file))?;
            let (w, _) = cooked::decode_scene(&data)?;
            let started = std::time::Instant::now();
            let baked = ganbate_engine::render::ao::bake_world(&w)?;
            std::fs::write(out, &baked)?;
            println!(
                "{}: baked AO in {:.1} s, {} bytes",
                Path::new(out).display(),
                started.elapsed().as_secs_f32(),
                baked.len()
            );
        }
        [cmd, file] if cmd == "inspect" => {
            let data = read_asset(Path::new(file))?;
            let (w, table) = cooked::decode_scene(&data)?;
            let version = cooked::version(&data).unwrap_or_default();
            println!("GBEW v{version} | {} | {} bytes", summary(&w)?, data.len());
            if !table.entities.is_empty() {
                println!("  {} entities", table.entities.len());
                for e in table.entities.iter().take(64) {
                    println!(
                        "  entity {:016x} {:?}: instances {:?}, lamps {:?}",
                        e.id, e.name, e.instances, e.lights
                    );
                }
            }
            for (i, m) in w.materials.iter().enumerate() {
                let mut traits = vec![
                    match m.shading {
                        Shading::Standard => "standard",
                        Shading::Foliage => "foliage",
                        Shading::Water => "water",
                        Shading::Unlit => "unlit",
                    }
                    .to_string(),
                ];
                if m.emission == Emission::Lamp {
                    traits.push("lamp".into());
                }
                match m.alpha {
                    AlphaMode::Opaque => {}
                    AlphaMode::Mask(c) => traits.push(format!("mask {c}")),
                    AlphaMode::Blend => traits.push("blend".into()),
                }
                for (name, map) in [
                    ("base", m.base_color_texture),
                    ("normal", m.normal_texture),
                    ("emissive", m.emissive_texture),
                ] {
                    if let Some(t) = map {
                        let t = &w.textures[t as usize];
                        traits.push(format!("{name} {}x{}", t.width, t.height));
                    }
                }
                println!("  material {i}: {} [{}]", m.name, traits.join(", "));
            }
        }
        [cmd, file, steps] if cmd == "simulate" => {
            simulate(Path::new(file), steps.to_str().unwrap_or(""))?;
        }
        _ => {
            return Err(
                "usage: ganbate-build cook [--compact] [--keep-images] source.glb output.gbew \
                        | cook-scene [--compact] [--keep-images] [--force] PROJECT SCENE output.gbew \
                        | import [--force] PROJECT content/model.glb content/new.scene.json \
                        | inspect file.gbew | bake-ao file.gbew output.gbao | model file.glb \
                        | simulate file.gbew STEPS"

                    .into(),
            );
        }
    }
    Ok(())
}

/// Steps a cooked world's authored physics `steps` times at 60 Hz, as Ganbate Studio's Simulate does,
/// and prints the state hash and where every moving body ended up. A browser that steps the same
/// world the same number of times reports the same hash (`physicsHash()` in the web library).
fn simulate(file: &Path, steps: &str) -> Result<(), Box<dyn std::error::Error>> {
    let steps: u32 = steps
        .parse()
        .map_err(|_| format!("STEPS must be a whole number, not {steps:?}"))?;
    let data = read_asset(file)?;
    let (w, table) = cooked::decode_scene(&data)?;
    let mut physics =
        ganbate_physics::scene::build(&w, &table, ganbate_physics::Settings::default())?;
    // The web library receives 1/60 as a JavaScript number and narrows it to f32 the same way.
    let dt = (1.0f64 / 60.0) as f32;
    for _ in 0..steps {
        physics.world.step(dt);
    }
    println!("hash {:016x}", physics.world.state_hash());
    println!(
        "{} bodies, {} awake, {} contacts after {steps} steps",
        physics.world.body_count(),
        physics.world.awake_body_count(),
        physics.world.contact_count()
    );
    for b in &physics.bodies {
        if physics.world.body_kind(b.body) == Some(ganbate_physics::BodyKind::Static) {
            continue;
        }
        let name = &table.entities[b.record as usize].name;
        if let Some(pose) = physics.world.body_pose(b.body) {
            let p = pose.position;
            println!("  {name:?}: at {:.4} {:.4} {:.4}", p.x, p.y, p.z);
        }
    }
    Ok(())
}
