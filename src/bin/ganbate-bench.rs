//! Uncapped offscreen throughput on the selected Vulkan driver, plus optional `--compute`
//! microbenchmarks for physics, work-stealing SIMD culling, BC/ASTC texture compression, and JSON.
#[cfg(any(target_os = "linux", target_os = "windows", target_os = "android"))]
#[path = "../../examples/common/gallery.rs"]
mod demo;

#[cfg(any(target_os = "linux", target_os = "windows", target_os = "android"))]
fn run_compute_benchmarks() {
    use std::sync::Arc;
    use std::time::Instant;

    use ganbate_assets::bc;
    use ganbate_core::{json, world::TextureCompression};
    use ganbate_jobs::{
        Aabb, CullConfig, Frustum, JobPool, cull_frustum_soa, cull_frustum_soa_scalar,
        pack_aabbs_soa, par_cull_frustum_into, sort_front_to_back_radix,
    };
    use ganbate_physics::{
        BodyDesc, BodyKind, Character, CharacterSettings, ColliderDesc, PhysicsWorld, Quat,
        Settings, Shape, Vec3, hull::ConvexHull,
    };

    println!("\n--- Ganbate Compute, Physics, SIMD Culling & Memory Suite ---");

    // 1. Deterministic Physics: Settled World + Fully Asleep Fast-Path + Active Pile
    let mut pts = Vec::with_capacity(64);
    for i in 0..64 {
        let u = (i as f32 + 0.5) / 64.0;
        let y = 1.0 - 2.0 * u;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let phi = i as f32 * 2.399_963_1;
        pts.push(Vec3::new(
            0.45 * r * phi.cos(),
            0.45 * y,
            0.45 * r * phi.sin(),
        ));
    }
    let hull = Arc::new(ConvexHull::from_points(&pts).expect("valid 64-pt hull"));

    let mut w1 = PhysicsWorld::new(Settings::default());
    let ground = w1
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..BodyDesc::default()
        })
        .unwrap();
    w1.add_collider(
        ground,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(200.0, 0.5, 200.0),
        }),
    )
    .unwrap();
    for i in 0..2500 {
        let b = w1
            .create_body(&BodyDesc {
                kind: BodyKind::Static,
                position: Vec3::new(
                    ((i % 50) as f32 - 25.0) * 3.0,
                    0.5,
                    ((i / 50) as f32) * 3.0 + 20.0,
                ),
                ..BodyDesc::default()
            })
            .unwrap();
        w1.add_collider(
            b,
            &ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(0.8, 0.5, 0.8),
            }),
        )
        .unwrap();
    }
    for i in 0..400 {
        let b = w1
            .create_body(&BodyDesc {
                kind: BodyKind::Dynamic,
                position: Vec3::new(
                    ((i % 20) as f32 - 10.0) * 2.2,
                    0.5,
                    -20.0 - (i / 20) as f32 * 2.2,
                ),
                awake: false,
                ..BodyDesc::default()
            })
            .unwrap();
        w1.add_collider(
            b,
            &ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(0.5, 0.5, 0.5),
            }),
        )
        .unwrap();
    }
    for i in 0..60 {
        let layer = i / 12;
        let slot = i % 12;
        let b = w1
            .create_body(&BodyDesc {
                kind: BodyKind::Dynamic,
                position: Vec3::new(
                    ((slot % 4) as f32 - 1.5) * 1.05 + (layer as f32) * 0.03,
                    0.6 + layer as f32 * 1.05,
                    ((slot / 4) as f32 - 1.0) * 1.05,
                ),
                ..BodyDesc::default()
            })
            .unwrap();
        let shape = if i % 3 == 0 {
            Shape::Hull(hull.clone())
        } else {
            Shape::Box {
                half_extents: Vec3::new(0.45, 0.45, 0.45),
            }
        };
        w1.add_collider(b, &ColliderDesc::new(shape)).unwrap();
    }

    let steps1 = 120usize;
    let t1 = Instant::now();
    for _ in 0..steps1 {
        w1.step(1.0 / 60.0);
    }
    let w1_us = t1.elapsed().as_secs_f64() * 1e6 / steps1 as f64;
    let hash1 = w1.state_hash();

    // Let the 60 boxes finish settling to sleep, then time 1,000 fully-settled steps.
    for _ in 0..360 {
        w1.step(1.0 / 60.0);
    }
    let asleep_iters = 1000usize;
    let t_sleep = Instant::now();
    for _ in 0..asleep_iters {
        w1.step(1.0 / 60.0);
    }
    let asleep_ns = t_sleep.elapsed().as_secs_f64() * 1e9 / asleep_iters as f64;

    let mut w2 = PhysicsWorld::new(Settings::default());
    let g2 = w2
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..BodyDesc::default()
        })
        .unwrap();
    w2.add_collider(
        g2,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(50.0, 0.5, 50.0),
        }),
    )
    .unwrap();
    for i in 0..120 {
        let layer = i / 16;
        let slot = i % 16;
        let b = w2
            .create_body(&BodyDesc {
                kind: BodyKind::Dynamic,
                position: Vec3::new(
                    ((slot % 4) as f32 - 1.5) * 0.95 + (layer as f32) * 0.04,
                    0.55 + layer as f32 * 0.98,
                    ((slot / 4) as f32 - 1.5) * 0.95,
                ),
                rotation: Quat::new(0.05, 0.1, -0.05, 1.0).normalize(),
                ..BodyDesc::default()
            })
            .unwrap();
        let shape = match i % 4 {
            0 => Shape::Box {
                half_extents: Vec3::new(0.42, 0.42, 0.42),
            },
            1 => Shape::Hull(hull.clone()),
            2 => Shape::Capsule {
                half_height: 0.2,
                radius: 0.3,
            },
            _ => Shape::Sphere { radius: 0.42 },
        };
        w2.add_collider(b, &ColliderDesc::new(shape)).unwrap();
    }
    let mut chars: Vec<Character> = (0..4)
        .map(|k| {
            Character::new(
                Vec3::new(-4.0 + k as f32 * 2.5, 1.2, 4.0),
                CharacterSettings::default(),
            )
        })
        .collect();
    let steps2 = 100usize;
    let t2 = Instant::now();
    for s in 0..steps2 {
        let dir = if s < 50 { -0.06 } else { 0.06 };
        for ch in &mut chars {
            ch.move_and_slide(&mut w2, Vec3::new(0.0, -0.15, dir), 1.0 / 60.0);
        }
        w2.step(1.0 / 60.0);
    }
    let w2_us = t2.elapsed().as_secs_f64() * 1e6 / steps2 as f64;
    let hash2 = w2.state_hash();

    println!(
        "Physics settled world (2961 bodies): {:.1} us/step (hash={:#018x}) | fully-asleep step: {:.1} ns/step (awake={})",
        w1_us,
        hash1,
        asleep_ns,
        w1.awake_body_count()
    );
    println!(
        "Physics active pile   (120 mixed + 4 chars): {:.1} us/step (hash={:#018x})",
        w2_us, hash2
    );

    // 2. SIMD SoA Frustum + Distance + Subpixel Culling + Work-Stealing Pool + O(N) Radix Sort
    let n_cull = 65_536usize;
    let aabbs: Vec<Aabb> = (0..n_cull)
        .map(|i| {
            let x = (((i * 73) % 400) as f32) - 200.0;
            let y = (((i * 31) % 100) as f32) - 50.0;
            let z = -(((i * 97) % 350) as f32) + 40.0;
            let r = 0.05 + ((i % 17) as f32) * 0.15;
            Aabb::new([x - r, y - r, z - r], [x + r, y + r, z + r])
        })
        .collect();
    let mut soa = Vec::new();
    pack_aabbs_soa(&aabbs, &mut soa);
    let f = 1.0 / (0.5f32).tan();
    let (near, far, aspect) = (0.2f32, 300.0f32, 16.0f32 / 9.0);
    let nf = 1.0 / (near - far);
    let vp = [
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
    ];
    let frustum = Frustum::from_view_proj_zero_to_one(&vp);
    let cfg = CullConfig {
        camera_pos: [0.0; 3],
        max_distance: 220.0,
        min_screen_px: 1.5,
        proj_scale_y: f,
        viewport_height_px: 1080.0,
        depth_range: 220.0,
    };

    let mut vis = Vec::with_capacity(n_cull / 2);
    let cull_iters = 50usize;

    let t_scalar = Instant::now();
    for _ in 0..cull_iters {
        vis.clear();
        cull_frustum_soa_scalar(&frustum, &cfg, &soa, n_cull, &mut vis);
    }
    let scalar_us = t_scalar.elapsed().as_secs_f64() * 1e6 / cull_iters as f64;

    let t_simd = Instant::now();
    let mut stats = Default::default();
    for _ in 0..cull_iters {
        vis.clear();
        stats = cull_frustum_soa(&frustum, &cfg, &soa, n_cull, &mut vis);
    }
    let simd_us = t_simd.elapsed().as_secs_f64() * 1e6 / cull_iters as f64;

    let pool = JobPool::new(4);
    let mut par_scratch = Vec::new();
    let mut par_vis = Vec::with_capacity(n_cull / 2);
    // Warm up worker threads and scratch capacities once.
    par_cull_frustum_into(
        &pool,
        &frustum,
        &cfg,
        &soa,
        n_cull,
        &mut par_scratch,
        &mut par_vis,
    );
    let t_par = Instant::now();
    for _ in 0..cull_iters {
        par_cull_frustum_into(
            &pool,
            &frustum,
            &cfg,
            &soa,
            n_cull,
            &mut par_scratch,
            &mut par_vis,
        );
    }
    let par_us = t_par.elapsed().as_secs_f64() * 1e6 / cull_iters as f64;

    let mut sort_scratch = Vec::new();
    let mut cmp_copy = par_vis.clone();
    let t_std_sort = Instant::now();
    for _ in 0..20 {
        cmp_copy.copy_from_slice(&vis);
        cmp_copy.sort_unstable_by_key(|&(_, d)| d);
    }
    let std_sort_us = t_std_sort.elapsed().as_secs_f64() * 1e6 / 20.0;

    let t_radix = Instant::now();
    for _ in 0..20 {
        cmp_copy.copy_from_slice(&vis);
        sort_front_to_back_radix(&mut cmp_copy, &mut sort_scratch);
    }
    let radix_us = t_radix.elapsed().as_secs_f64() * 1e6 / 20.0;

    println!(
        "Culling 65,536 AABBs (vis={}, frustum_culled={}, dist/subpx={}): scalar {:.1} us | 1T SIMD {:.1} us | {}T JobPool+SIMD {:.1} us",
        stats.visible,
        stats.frustum_culled,
        stats.distance_or_subpixel_culled,
        scalar_us,
        simd_us,
        pool.worker_count(),
        par_us
    );
    println!(
        "Early-Z depth sort ({} visible draws): O(N log N) sort_unstable {:.1} us -> O(N) 16-bit radix sort {:.1} us",
        vis.len(),
        std_sort_us,
        radix_us
    );

    // 3. Hardware Block-Compressed Texture Memory Footprint (512x512 RGBA8 vs BC1 / BC7 / ASTC 4x4)
    let rgba512 = vec![180u8; 512 * 512 * 4];
    let bc1 = bc::encode_rgba8(TextureCompression::Bc1, &rgba512, 512, 512).unwrap();
    let bc7 = bc::encode_rgba8(TextureCompression::Bc7, &rgba512, 512, 512).unwrap();
    let astc = bc::encode_rgba8(TextureCompression::Astc4x4, &rgba512, 512, 512).unwrap();
    println!(
        "VRAM 512x512 base mip: RGBA8 {} KiB -> BC1 {} KiB ({:.1}x) | BC7 {} KiB ({:.1}x) | ASTC 4x4 {} KiB ({:.1}x)",
        rgba512.len() / 1024,
        bc1.len() / 1024,
        rgba512.len() as f64 / bc1.len() as f64,
        bc7.len() / 1024,
        rgba512.len() as f64 / bc7.len() as f64,
        astc.len() / 1024,
        rgba512.len() as f64 / astc.len() as f64,
    );

    // 4. Core JSON Parse + Canonical Serialize (500 entities)
    let mut json_src = String::from("{\"entities\":[");
    for i in 0..500 {
        if i > 0 {
            json_src.push(',');
        }
        json_src.push_str(&format!(
            "{{\"id\":\"ent_{i:04}\",\"name\":\"PropObject_{i}\",\"visible\":true,\"transform\":[1,0,0,0,0,1,0,0,0,0,1,0,{},{},0,1],\"tags\":[\"prop\",\"dynamic\"]}}",
            i,
            i * 2
        ));
    }
    json_src.push_str("],\"version\":1}");
    let jiters = 60usize;
    let t_jp = Instant::now();
    let mut val = json::parse(json_src.as_bytes()).unwrap();
    for _ in 0..jiters {
        val = json::parse(json_src.as_bytes()).unwrap();
    }
    let jp_us = t_jp.elapsed().as_secs_f64() * 1e6 / jiters as f64;
    let t_jc = Instant::now();
    let mut canon = String::new();
    for _ in 0..jiters {
        canon = json::to_canonical(&val).unwrap();
    }
    let jc_us = t_jc.elapsed().as_secs_f64() * 1e6 / jiters as f64;
    println!(
        "Core JSON (500 entities, {} B): parse {:.1} us | to_canonical {:.1} us",
        canon.len(),
        jp_us,
        jc_us
    );
}

#[cfg(any(target_os = "linux", target_os = "windows", target_os = "android"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ganbate_engine::{
        Camera, assets,
        timing::Distribution,
        vulkan::{VulkanRenderer, bench},
    };
    let mut args = std::env::args().skip(1);
    let (mut w, mut h, mut warmup, mut frames) = (1920, 1080, 40, 240);
    let mut scene_path = None;
    let mut compute = false;
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "ganbate-bench [--scene world.gbew] [--width 1920] [--height 1080] [--warmup 40] [--frames 240] [--compute]"
            );
            return Ok(());
        }
        if flag == "--compute" || flag == "--suite" {
            compute = true;
            continue;
        }
        let value = args.next().ok_or("missing option value")?;
        match flag.as_str() {
            "--scene" => scene_path = Some(value),
            "--width" => w = value.parse()?,
            "--height" => h = value.parse()?,
            "--warmup" => warmup = value.parse()?,
            "--frames" => frames = value.parse()?,
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    let scene = if let Some(path) = scene_path {
        assets::cooked::decode(&assets::read_asset(std::path::Path::new(&path))?)?.scene()?
    } else {
        demo::gallery(0.65)
    };
    let bounds = scene
        .bounds()
        .ok_or("benchmark needs a finite nonempty scene")?;
    let mut gpu = VulkanRenderer::new()?;
    println!(
        "Ganbate offscreen | {} | {w}x{h} | {} triangles | {warmup} warmup + {frames} measured | 1 static upload | 2 frames in flight",
        gpu.device_name(),
        scene.triangles.len()
    );
    let run = bench::run(&mut gpu, &scene, (w, h), warmup, frames, |i| {
        Camera::orbit(
            bounds,
            0.4 + (i % 360) as f32 * std::f32::consts::TAU / 360.,
        )
    })?;
    println!(
        "Completed throughput {:.2} FPS in {:.3} s",
        run.throughput(),
        run.seconds
    );
    for (name, samples) in [
        ("CPU prepare + submit", &run.cpu_ms),
        ("GPU draw", &run.gpu_ms),
        ("CPU wait for slot", &run.fence_ms),
    ] {
        if let Some(d) = Distribution::from_ms(samples) {
            println!(
                "{name}: p50 {:.3} ms | p95 {:.3} ms | p99 {:.3} ms (n={})",
                d.p50, d.p95, d.p99, d.count
            );
        } else {
            println!("{name}: unavailable");
        }
    }
    println!(
        "Render throughput only. Display pacing, gameplay, networking, UI and asset loads are outside this run."
    );
    if compute {
        run_compute_benchmarks();
    }
    Ok(())
}
#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "android")))]
fn main() {
    eprintln!("ganbate-bench needs a supported Vulkan host");
    std::process::exit(1);
}
