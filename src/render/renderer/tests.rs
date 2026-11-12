use super::*;
use crate::Vec3;
use crate::render::lights::{LightDesc, SpotCone};
use crate::render::look::SurfaceModel;
use crate::render::props::{MaterialDesc, MeshData};
use crate::render::sky::{TimeOfDay, Weather};
use crate::world::{AlphaMode, Instance, Material, Mesh, PointLight, Shading, Transform, World};
use ganbate_shader::{
    BindingKind, CompileOptions, SampleType, ShaderLibrary, SpecValue, TextureView,
    compile_glsl_with_options, compile_msl_with_options, compile_wgsl_with_options,
    compile_with_options,
};

#[test]
fn a_lamp_lands_in_the_cluster_of_every_point_it_lights() {
    // A camera at the origin looking down -z; a lamp 10 m ahead with a 3 m reach.
    let v = look_at([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
    let p = perspective(60f32.to_radians(), 1.0, 0.1);
    let vp = mul(&p, &v);
    let (lamp, r) = ([0.0f32, 0.0, -10.0], 3.0f32);
    let centre = transform(&vp, [lamp[0], lamp[1], lamp[2], 1.0])[3];
    assert!(
        (centre - 10.0).abs() < 1e-3,
        "clip w is the view depth, got {centre}"
    );
    let (s0, s1) = (cluster_slice(centre - r), cluster_slice(centre + r));
    for z in [7.5f32, 8.0, 10.0, 12.0, 12.9] {
        let w = transform(&vp, [0.0, 0.0, -z, 1.0])[3];
        let s = cluster_slice(w);
        assert!(
            s0 <= s && s <= s1,
            "a point {z} m away is in slice {s}, the lamp in {s0}..={s1}"
        );
    }
}

#[test]
fn crowded_clusters_keep_their_nearest_lamps_alike() {
    // Room for them all: each keeps every lamp it has (up to the cap a cluster holds).
    assert_eq!(
        cluster_cap([3, 200, 0].into_iter(), 1000),
        MAX_LIGHTS_PER_CLUSTER
    );
    // Room for 30: the most each can keep so that all fit, the short lists whole.
    let cap = cluster_cap([4, 50, 50].into_iter(), 30);
    assert_eq!(cap, 13);
    assert!(4 + 2 * cap <= 30 && 4 + 2 * (cap + 1) > 30);
    assert_eq!(cluster_cap([5, 5].into_iter(), 0), 0);
}

#[test]
fn lamp_clusters_slice_view_depth_in_log_steps() {
    // Everything nearer than the first slice's far edge, and the camera's own plane, is in it.
    assert_eq!(cluster_slice(-1.0), 0);
    assert_eq!(cluster_slice(CLUSTER_NEAR), 0);
    // Beyond the far edge, the last slice.
    assert_eq!(cluster_slice(CLUSTER_FAR * 4.0), CLUSTER_SLICES - 1);
    // Never back, and a doubling of depth moves `cluster_scale` slices.
    let mut last = 0;
    for i in 1..4000 {
        let s = cluster_slice(i as f32 * 0.125);
        assert!(s >= last);
        last = s;
    }
    let (a, b) = (cluster_slice(3.0), cluster_slice(6.0));
    assert!((b - a) as f32 >= cluster_scale().floor() && (b - a) as f32 <= cluster_scale().ceil());
}

#[test]
fn grass_spacing_matches_the_shader() {
    assert!(SCENE.contains(&format!("const GRASS_SPACING = {GRASS_SPACING};")));
}

#[test]
fn receivers_outside_the_view_get_none() {
    let view_proj = mul(
        &perspective(60f32.to_radians(), 4.0 / 3.0, 0.1),
        &look_at([0.0, 1.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    );
    let map = ortho(-40.0, 40.0, -40.0, 40.0, -40.0, 40.0);
    let maps = [
        (map, 0.1, 0.5),
        (map, 0.1, 0.5),
        (map, 0.1, 0.0),
        (map, 0.1, 0.0),
    ];
    let draw = |lo: V3, hi: V3| DrawItem {
        first_index: 0,
        count: 6,
        material: 0,
        water: false,
        bounds: (lo, hi),
        max_extent: 2.0,
    };
    let draws = [
        draw([-1.0, 0.0, -1.0], [1.0, 0.5, 1.0]),
        draw([30.0, 0.0, -1.0], [32.0, 0.5, 1.0]),
        draw([-1.0, 0.0, 9.0], [1.0, 0.5, 11.0]),
    ];
    let size = (64, 48);
    let r = ReceiverBounds::new(7, &draws, &maps, &view_proj, size);
    assert_eq!((r.key, r.maps.len(), r.screen.len()), (7, 3, 3));

    let seen = draws[0].bounds;
    let before: [([f32; 4], f32); 4] = std::array::from_fn(|v| {
        let (m, _, grow) = &maps[v];
        let (b, _, far) = map_bounds((seen.0.map(|x| x - grow), seen.1.map(|x| x + grow)), m);
        (b, far)
    });
    assert_eq!(r.maps[0], before);
    let screen = screen_rect(&corners(seen), &view_proj, size, 2.0);
    assert!(screen.is_some());
    assert_eq!(r.screen[0], screen);

    // Beside the view and behind the eye: the picture holds none of them.
    let none = ([1.0, 1.0, -1.0, -1.0], f32::NEG_INFINITY);
    for i in [1, 2] {
        assert_eq!(r.maps[i], [none; 4], "draw {i}");
        assert_eq!(r.screen[i], None, "draw {i}");
    }
}

#[test]
fn the_cached_picture_shaders_compile() {
    compile(SCENE, Stage::Fragment, "fs_reshade_mask").expect("fs_reshade_mask");
    compile(SCENE, Stage::Fragment, "fs_main_cache").expect("fs_main_cache");
    compile(SCENE, Stage::Fragment, "fs_main_opaque_cache").expect("fs_main_opaque_cache");
    compile(POST, Stage::Fragment, "fs_clear").expect("fs_clear");
    // The mask on every backend the player draws with.
    let composed = ganbate_shader::compose(SCENE, &shader_modules()).expect("compose");
    let opts = CompileOptions::default();
    let entry = "fs_reshade_mask";
    compile_glsl_with_options(&composed, Stage::Fragment, entry, &opts).expect("glsl");
    compile_msl_with_options(&composed, Stage::Fragment, entry, &opts).expect("msl");
    compile_wgsl_with_options(&composed, Stage::Fragment, entry, &opts).expect("wgsl");
}

#[test]
fn the_cached_water_reads_its_light_within_webgl_units() {
    let composed = ganbate_shader::compose(SCENE, &shader_modules()).expect("compose");
    let glsl = |entry: &str, specialize: &[(&str, SpecValue)]| {
        let opts = CompileOptions {
            imports: &[],
            specialize,
        };
        let c = compile_glsl_with_options(&composed, Stage::Fragment, entry, &opts)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        let g = c.glsl.expect("glsl");
        (
            g.units.len() + g.storage.len(),
            sorted_bindings(&c.bindings),
        )
    };
    let (opaque_units, opaque) = glsl("fs_main_opaque", &[]);
    let (water_units, water) = glsl("fs_water", &WATER_STATICS);
    let (_, statics) = glsl("fs_water_static", &[]);
    // Where props moved since its light was kept, the water computes it as `fs_main_opaque`
    // does: it reads all that reads, plus the kept light (a unit) and the moved props'
    // footprints (a uniform block).
    assert!(
        water_units == opaque_units + 1 && water_units <= 16,
        "WebGL 2 fragment texture units: fs_water {water_units}, fs_main_opaque {opaque_units}"
    );
    let kept = [
        (
            0,
            17,
            BindingKind::UniformBuffer {
                size: RESHADE_BYTES as u32,
            },
        ),
        (
            0,
            19,
            BindingKind::Texture {
                view: TextureView::D2,
                sample: SampleType::Float,
            },
        ),
    ]
    .map(|(group, binding, kind)| ganbate_shader::Binding {
        group,
        binding,
        kind,
    });
    assert_eq!(water, sorted_bindings(&[&opaque[..], &kept].concat()));
    // The statics take the scene's group, which must not hold the target they draw into.
    assert_eq!(statics, opaque);
    for (entry, spec) in [("fs_water", &WATER_STATICS[..]), ("fs_water_static", &[])] {
        let opts = CompileOptions {
            imports: &[],
            specialize: spec,
        };
        compile_specialized(SCENE, Stage::Fragment, entry, spec).expect(entry);
        compile_msl_with_options(&composed, Stage::Fragment, entry, &opts).expect(entry);
        compile_wgsl_with_options(&composed, Stage::Fragment, entry, &opts).expect(entry);
    }
}

/// The grade-once passes compile for every backend the player draws with, and the output
/// reading the grades (`GRADED`) binds the frame, the depth and the grades alone: nothing the
/// grading reads (the bloom, its sampler) is left in it.
#[test]
fn the_grade_once_shaders_compile_everywhere() {
    let composed = ganbate_shader::compose(POST, &shader_modules()).expect("compose");
    for (entry, specialize) in [
        ("fs_grade", &[][..]),
        ("fs_grade_plain", &[]),
        ("fs_tonemap_output", &GRADED),
    ] {
        let opts = CompileOptions {
            imports: &[],
            specialize,
        };
        compile_specialized(POST, Stage::Fragment, entry, specialize).expect(entry);
        compile_glsl_with_options(&composed, Stage::Fragment, entry, &opts).expect(entry);
        compile_msl_with_options(&composed, Stage::Fragment, entry, &opts).expect(entry);
        compile_wgsl_with_options(&composed, Stage::Fragment, entry, &opts).expect(entry);
    }
    let read = compile_specialized(POST, Stage::Fragment, "fs_tonemap_output", &GRADED)
        .expect("fs_tonemap_output");
    let bound: Vec<_> = sorted_bindings(&read.bindings)
        .iter()
        .map(|b| b.binding)
        .collect();
    assert_eq!(bound, [0, 1, 3]);
}

#[test]
fn moved_shadows_reach_the_water_in_view() {
    let view_proj = mul(
        &perspective(60f32.to_radians(), 4.0 / 3.0, 0.1),
        &look_at([0.0, 2.0, 6.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    );
    // A light straight above.
    let map = mul(
        &ortho(-40.0, 40.0, -40.0, 40.0, -40.0, 40.0),
        &look_at([0.0, 20.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    );
    let maps = [
        (map, 0.1, 0.5),
        (map, 0.1, 0.5),
        (map, 0.1, 0.0),
        (map, 0.1, 0.0),
    ];
    let draw = |lo: V3, hi: V3, water: bool| DrawItem {
        first_index: 0,
        count: 6,
        material: 0,
        water,
        bounds: (lo, hi),
        max_extent: 2.0,
    };
    let draws = [
        draw([-4.0, -0.5, -4.0], [4.0, -0.4, 4.0], false),
        draw([-3.0, 0.0, -3.0], [3.0, 0.0, 3.0], true),
        // Behind the eye.
        draw([-3.0, 0.0, 20.0], [3.0, 0.0, 26.0], true),
    ];
    let r = ReceiverBounds::new(7, &draws, &maps, &view_proj, (64, 48));
    // The cached picture's re-shade never reaches the water; the water in view is apart.
    let none = ([1.0, 1.0, -1.0, -1.0], f32::NEG_INFINITY);
    assert_eq!(r.maps[1], [none; 4]);
    assert_eq!(r.water.len(), 1);
    let footprint = |lo: V3, hi: V3| {
        let (rect, near, _) = map_bounds((lo, hi), &map);
        vec![(0, rect, near)]
    };
    assert!(reached(
        &footprint([1.0, 0.5, 1.0], [1.5, 1.0, 1.5]),
        &r.water[0]
    ));
    // Beside the water, and under it.
    assert!(!reached(
        &footprint([20.0, 0.5, 1.0], [21.0, 1.0, 1.5]),
        &r.water[0]
    ));
    assert!(!reached(
        &footprint([1.0, -2.0, 1.0], [1.5, -1.5, 1.5]),
        &r.water[0]
    ));
}

#[test]
fn specialize_brdf_permutations_all_backends() {
    let mut lib = ShaderLibrary::new();
    for (name, src) in shader_modules() {
        lib.add(name, src);
    }
    let composed = lib.compose(SCENE).expect("compose scene");

    let full = compile_with_options(
        &composed,
        Stage::Fragment,
        "fs_main",
        &CompileOptions::default(),
    )
    .expect("full uber-shader");

    for fixed in 0..6 {
        let specs = [("FIXED_SURFACE_MODEL", SpecValue::I32(fixed))];
        let opts = CompileOptions {
            imports: &[],
            specialize: &specs,
        };
        let spv = compile_with_options(&composed, Stage::Fragment, "fs_main", &opts)
            .expect("specialized spirv");
        assert!(
            spv.words.len() < full.words.len(),
            "fixed model {fixed} should prune unused BRDF lobes ({} vs {})",
            spv.words.len(),
            full.words.len()
        );
        compile_msl_with_options(&composed, Stage::Fragment, "fs_main", &opts)
            .expect("specialized msl");
        compile_wgsl_with_options(&composed, Stage::Fragment, "fs_main", &opts)
            .expect("specialized wgsl");
        compile_glsl_with_options(&composed, Stage::Fragment, "fs_main", &opts)
            .expect("specialized glsl");
    }
}

fn make_quad_mesh(y: f32, half: f32) -> Mesh {
    Mesh {
        positions: vec![
            Vec3(-half, y, -half),
            Vec3(-half, y, half),
            Vec3(half, y, half),
            Vec3(half, y, -half),
        ],
        normals: vec![Vec3(0.0, 1.0, 0.0); 4],
        uvs: vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        color: [1.0, 1.0, 1.0],
        indices: vec![0, 1, 2, 0, 2, 3],
        material: Some(0),
    }
}

fn sphere_prop_data() -> (Vec<V3>, Vec<V3>, Vec<[f32; 2]>, Vec<u32>) {
    let (stacks, slices) = (10u32, 16u32);
    let (mut pos, mut nrm, mut uvs, mut idx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..=stacks {
        let v = i as f32 / stacks as f32;
        let phi = v * std::f32::consts::PI;
        for j in 0..=slices {
            let u = j as f32 / slices as f32;
            let theta = u * std::f32::consts::TAU;
            let n = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
            pos.push(math::scale(n, 0.85));
            nrm.push(n);
            uvs.push([u, v]);
        }
    }
    let row = slices + 1;
    for i in 0..stacks {
        for j in 0..slices {
            let a = i * row + j;
            let b = a + row;
            idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (pos, nrm, uvs, idx)
}

#[test]
fn hybrid_stage_culls_caches_and_renders_unified_surface_models() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    world.materials.push(Material {
        name: "pedestal".into(),
        base_color: [0.55, 0.53, 0.50, 1.0],
        roughness: 0.65,
        metallic: 0.0,
        alpha: AlphaMode::Opaque,
        shading: Shading::Standard,
        ..Material::default()
    });
    world.meshes.push(make_quad_mesh(0.0, 4.0));
    let mut behind = math::IDENTITY;
    behind[12] = 0.0;
    behind[14] = 95.0; // Far behind the eye at z = 4 looking towards origin.
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(behind),
    });

    let mut look = Look::default();
    let mut renderer = Renderer::new(
        &mut device,
        &world,
        look.clone(),
        Quality::hybrid_stage(),
        (64, 64),
        Output::Still,
    )
    .expect("create hybrid renderer");

    let (sp, sn, suv, sidx) = sphere_prop_data();
    let sphere_mesh = renderer
        .create_prop_mesh(
            &mut device,
            &MeshData {
                positions: &sp,
                normals: &sn,
                uvs: &suv,
                indices: &sidx,
            },
        )
        .expect("sphere mesh");
    let sphere_mat = renderer
        .create_prop_material(
            &mut device,
            &MaterialDesc {
                name: "probe".into(),
                base_color: [0.75, 0.22, 0.18, 1.0],
                roughness: 0.25,
                metallic: 0.1,
                ..MaterialDesc::default()
            },
        )
        .expect("sphere mat");
    let mut sphere_xf = math::IDENTITY;
    sphere_xf[13] = 0.9;
    let _sphere_inst = renderer
        .create_prop_instance(sphere_mesh, sphere_mat, &sphere_xf)
        .expect("sphere instance");

    let view = View {
        name: "probe".into(),
        eye: [0.0, 1.8, 3.6],
        target: [0.0, 0.85, 0.0],
        fov: 45.0,
        ..View::default()
    };
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 14.5,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &look.palette,
        1.0,
    );

    let img1 = renderer
        .render_still(&mut device, &view, &env, 1)
        .expect("frame 1");
    let stats1 = renderer.last_frame_stats();
    assert!(stats1.world_draws_culled >= 1);
    assert!(!stats1.reflection_drawn);
    assert!(!stats1.ssao_drawn);
    assert!(stats1.fused_tonemap);
    assert!(!stats1.shadow_static_cached);

    let _img2 = renderer
        .render_still(&mut device, &view, &env, 1)
        .expect("frame 2");
    let stats2 = renderer.last_frame_stats();
    assert!(stats2.shadow_static_cached);
    assert!(stats2.ground_cached);

    // Verify unified surface models + tonemap curves + EV100 exposure produce distinct valid frames.
    let avg_luma = |px: &[u8]| -> f64 {
        px.chunks_exact(4)
            .map(|c| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64)
            .sum::<f64>()
            / (px.len() / 4) as f64
    };
    assert!(avg_luma(&img1) > 5.0);

    let mut model_renders = Vec::new();
    for model in SurfaceModel::ALL {
        renderer
            .update_prop_material(
                &mut device,
                sphere_mat,
                &MaterialDesc {
                    name: "probe".into(),
                    base_color: [0.78, 0.24, 0.20, 1.0],
                    roughness: 0.22,
                    metallic: if model == SurfaceModel::StandardGGX {
                        0.85
                    } else {
                        0.0
                    },
                    surface: Some(model),
                    clearcoat: Some(0.95),
                    clearcoat_roughness: Some(0.06),
                    sheen: Some(0.9),
                    subsurface: Some(0.8),
                    ..MaterialDesc::default()
                },
            )
            .expect("update prop surface");
        let px = renderer
            .render_still(&mut device, &view, &env, 1)
            .expect("render surface model");
        assert!(avg_luma(&px) > 5.0);
        model_renders.push(px);
    }
    assert_ne!(
        model_renders[SurfaceModel::StylizedRamp as usize],
        model_renders[SurfaceModel::StandardGGX as usize]
    );
    assert_ne!(
        model_renders[SurfaceModel::StandardGGX as usize],
        model_renders[SurfaceModel::CharlieCloth as usize]
    );

    look.tonemap = ToneMap::AgX;
    look.ev100 = Some(13.0);
    renderer.set_look(look.clone());
    let agx_ev13 = renderer
        .render_still(&mut device, &view, &env, 1)
        .expect("agx ev13");

    look.tonemap = ToneMap::ACESFitted;
    look.ev100 = Some(11.5); // Brighter camera exposure (+1.5 EV stops).
    renderer.set_look(look);
    let aces_ev11 = renderer
        .render_still(&mut device, &view, &env, 1)
        .expect("aces ev11.5");
    assert!(avg_luma(&aces_ev11) > avg_luma(&agx_ev13));

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    renderer.destroy_props(&mut device);
}

#[test]
fn low_cost_quality_invariants_and_budget_ordering() {
    let low = Quality::low_cost();
    let hybrid = Quality::hybrid_stage();
    let realtime = Quality::realtime();
    let still = Quality::still();

    assert_eq!(low.shadow_size, 512);
    assert_eq!(low.reflection_scale, 0.0);
    assert_eq!(low.ao_scale, 0.25);
    assert_eq!(low.ao_samples, 0);
    assert_eq!(low.tile_size, 32);
    assert_eq!(low.bloom_levels, 2);
    assert!(!low.painterly);
    assert!(!low.fxaa);
    assert!(!low.particles);
    assert_eq!(low.ground_size, 256);
    assert_eq!(low.world_size, 512);
    assert_eq!(low.grass, 0.0);
    assert_eq!(low.light_shadow_size, 256);

    assert!(low.shadow_size < hybrid.shadow_size && hybrid.shadow_size < realtime.shadow_size);
    assert!(low.reflection_scale < hybrid.reflection_scale);
    assert!(hybrid.reflection_scale < realtime.reflection_scale);
    assert!(low.bloom_levels < hybrid.bloom_levels && hybrid.bloom_levels < realtime.bloom_levels);
    assert!(low.ground_size < hybrid.ground_size && hybrid.ground_size <= realtime.ground_size);
    assert!(low.world_size < hybrid.world_size && hybrid.world_size <= realtime.world_size);
    assert!(low.grass < hybrid.grass && hybrid.grass < realtime.grass);
    assert!(
        low.light_shadow_size < hybrid.light_shadow_size
            && hybrid.light_shadow_size < realtime.light_shadow_size
            && realtime.light_shadow_size < still.light_shadow_size
    );
}

#[test]
fn set_quality_switches_dynamically_skips_water_reflection_and_preserves_props() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    world.materials.push(Material {
        name: "stone".into(),
        base_color: [0.5, 0.5, 0.52, 1.0],
        roughness: 0.7,
        metallic: 0.0,
        alpha: AlphaMode::Opaque,
        shading: Shading::Standard,
        ..Material::default()
    });
    world.materials.push(Material {
        name: "pond".into(),
        base_color: [0.12, 0.34, 0.42, 0.85],
        roughness: 0.08,
        metallic: 0.0,
        alpha: AlphaMode::Opaque,
        shading: Shading::Water,
        ..Material::default()
    });
    world.meshes.push(make_quad_mesh(-0.4, 5.0));
    let mut water_mesh = make_quad_mesh(0.0, 4.0);
    water_mesh.material = Some(1);
    world.meshes.push(water_mesh);
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    world.instances.push(Instance {
        mesh: 1,
        transform: Transform(math::IDENTITY),
    });

    let look = Look::default();
    let mut renderer = Renderer::new(
        &mut device,
        &world,
        look.clone(),
        Quality::realtime(),
        (64, 64),
        Output::Still,
    )
    .expect("create realtime renderer");
    assert_eq!(renderer.quality(), Quality::realtime());

    let (sp, sn, suv, sidx) = sphere_prop_data();
    let prop_mesh = renderer
        .create_prop_mesh(
            &mut device,
            &MeshData {
                positions: &sp,
                normals: &sn,
                uvs: &suv,
                indices: &sidx,
            },
        )
        .expect("prop mesh");
    let prop_mat = renderer
        .create_prop_material(
            &mut device,
            &MaterialDesc {
                name: "buoy".into(),
                base_color: [0.85, 0.32, 0.16, 1.0],
                roughness: 0.3,
                ..MaterialDesc::default()
            },
        )
        .expect("prop material");
    let mut prop_xf = math::IDENTITY;
    prop_xf[13] = 0.8;
    let _prop_inst = renderer
        .create_prop_instance(prop_mesh, prop_mat, &prop_xf)
        .expect("prop instance");

    let view = View {
        name: "pond".into(),
        eye: [0.0, 2.0, 4.2],
        target: [0.0, 0.2, 0.0],
        fov: 48.0,
        ..View::default()
    };
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 15.0,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &look.palette,
        1.0,
    );

    let rt_px = renderer
        .render_still(&mut device, &view, &env, 1)
        .expect("render realtime");
    let rt_stats = renderer.last_frame_stats();
    assert!(rt_stats.reflection_drawn, "realtime draws water reflection");
    assert!(rt_stats.ssao_drawn, "realtime draws SSAO");
    assert!(
        !rt_stats.fused_tonemap,
        "realtime uses painterly post chain"
    );

    renderer
        .set_quality(&mut device, Quality::low_cost())
        .expect("switch to low_cost");
    assert_eq!(renderer.quality(), Quality::low_cost());
    // No-op re-apply of identical quality must succeed immediately.
    renderer
        .set_quality(&mut device, Quality::low_cost())
        .expect("idempotent low_cost");

    let lc_px = renderer
        .render_still(&mut device, &view, &env, 1)
        .expect("render low_cost");
    let lc_stats = renderer.last_frame_stats();
    assert!(
        !lc_stats.reflection_drawn,
        "low_cost skips water reflection when reflection_scale == 0.0"
    );
    assert!(!lc_stats.ssao_drawn, "low_cost skips SSAO");
    assert!(lc_stats.fused_tonemap, "low_cost uses fused LUT tonemap");

    renderer
        .set_quality(&mut device, Quality::hybrid_stage())
        .expect("switch to hybrid_stage");
    assert_eq!(renderer.quality(), Quality::hybrid_stage());
    let hy_px = renderer
        .render_still(&mut device, &view, &env, 1)
        .expect("render hybrid_stage");
    let hy_stats = renderer.last_frame_stats();
    assert!(
        hy_stats.reflection_drawn,
        "hybrid_stage restores water reflection when water is visible"
    );
    assert!(!hy_stats.ssao_drawn);
    assert!(hy_stats.fused_tonemap);

    for px in [&rt_px, &lc_px, &hy_px] {
        let luma = px
            .chunks_exact(4)
            .map(|c| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64)
            .sum::<f64>()
            / (px.len() / 4) as f64;
        assert!(luma > 5.0);
    }

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    renderer.destroy_props(&mut device);
}

/// One frame as RGBA8 sRGB: a still's linear readback encoded here, or a presented frame's
/// `Rgba8Srgb` `target` read back.
fn frame_srgb(
    device: &mut Device<'_>,
    r: &mut Renderer,
    view: &View,
    env: &Environment,
    target: Option<(TextureId, BufferId)>,
) -> Vec<u8> {
    let mut enc = device.encoder();
    r.encode(
        device,
        &mut enc,
        view,
        env,
        [0.0; 2],
        0,
        target.map(|t| t.0),
    )
    .expect("encode");
    if let Some((texture, buffer)) = target {
        enc.copy_texture_to_buffer(texture, 0, 0, buffer, 0);
    }
    device.submit(enc).expect("submit");
    if let Some((_, buffer)) = target {
        return device.read_buffer(buffer).expect("read");
    }
    let linear = device
        .read_buffer(r.targets.readback.expect("readback"))
        .expect("read");
    linear
        .chunks_exact(2)
        .map(|h| {
            let v = f16_to_f32(u16::from_le_bytes([h[0], h[1]]));
            (math::linear_to_srgb(v) * 255.0).round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// A presented frame, whose water reads its lamp light and shadow as the cached picture
/// keeps them (`cache_water`), shows what a still (computing them) shows, to the step its
/// own sRGB encoding rounds differently: by day and at night under a lamp and a spot light,
/// and while a ball under the spot (its shadow on the water) moves every frame, as a
/// dealer's idle does, which the water follows without drawing what it keeps again. Held,
/// and while the ball settles into the picture, it does not change. The pond stands alone:
/// with nothing else in the picture no frame re-shades it (`fs_reshade_mask`, which llvmpipe
/// crashes on).
#[test]
fn the_cached_water_matches_the_water_computed_each_frame() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    world.materials.push(Material {
        name: "pond".into(),
        base_color: [0.12, 0.34, 0.42, 0.85],
        roughness: 0.08,
        shading: Shading::Water,
        ..Material::default()
    });
    world.meshes.push(make_quad_mesh(0.0, 4.0));
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    world.lights.push(PointLight {
        position: Vec3(-1.2, 0.5, 0.8),
        color: [1.0, 0.8, 0.6],
        intensity: 6.0,
        range: 4.0,
    });
    let look = Look::default();
    let env = |hours| {
        let time = TimeOfDay {
            hours,
            ..TimeOfDay::default()
        };
        Environment::resolve(&time, &Weather::default(), &look.palette, 1.0)
    };
    let (day, night) = (env(15.0), env(22.5));
    let size = (80, 56);
    // No mirror: a presented frame keeps the water's reflection while the world holds, so
    // the ball's would stay where it was.
    let quality = Quality {
        reflection_scale: 0.0,
        ..Quality::hybrid_stage()
    };
    let mut still = Renderer::new(
        &mut device,
        &world,
        look.clone(),
        quality,
        size,
        Output::Still,
    )
    .expect("still renderer");
    let mut live = Renderer::new(
        &mut device,
        &world,
        look.clone(),
        quality,
        size,
        Output::Surface(Format::Rgba8Srgb),
    )
    .expect("presented renderer");
    let target = device
        .create_texture(&TextureDesc::new(
            size.0,
            size.1,
            Format::Rgba8Srgb,
            TextureUsage::COLOR_TARGET | TextureUsage::COPY_SRC,
        ))
        .expect("target");
    let readback = device
        .create_buffer(&BufferDesc {
            size: u64::from(size.0 * size.1 * 4),
            usage: BufferUsage::COPY_DST,
            memory: Memory::Readback,
        })
        .expect("readback");
    let out = Some((target, readback));
    let ball_at = |x: f32| {
        let mut m = math::IDENTITY;
        (m[0], m[5], m[10]) = (0.35, 0.35, 0.35);
        (m[12], m[13], m[14]) = (x, 0.6, 0.3);
        m
    };
    let spot = LightDesc {
        position: [0.0, 2.5, 0.3],
        range: 6.0,
        color: [0.9, 0.95, 1.0],
        intensity: 30.0,
        spot: Some(SpotCone {
            direction: [0.0, -1.0, 0.0],
            inner: 0.5,
            outer: 0.8,
        }),
        follows_lamps: false,
        shadow: 1.0,
    };
    let (sp, sn, suv, sidx) = sphere_prop_data();
    let mut balls = vec![];
    for r in [&mut still, &mut live] {
        r.create_light(&spot).expect("spot");
        let mesh = r
            .create_prop_mesh(
                &mut device,
                &MeshData {
                    positions: &sp,
                    normals: &sn,
                    uvs: &suv,
                    indices: &sidx,
                },
            )
            .expect("ball mesh");
        let material = r
            .create_prop_material(
                &mut device,
                &MaterialDesc {
                    name: "ball".into(),
                    base_color: [0.8, 0.3, 0.2, 1.0],
                    ..MaterialDesc::default()
                },
            )
            .expect("ball material");
        balls.push(
            r.create_prop_instance(mesh, material, &ball_at(-0.4))
                .expect("ball"),
        );
    }
    live.finish_ao(&mut device).expect("ao");
    let view = View {
        name: "pond".into(),
        eye: [0.0, 2.0, 4.2],
        target: [0.0, 0.2, 0.0],
        fov: 48.0,
        ..View::default()
    };

    let differ = |a: &[u8], b: &[u8]| {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .flat_map(|(p, q)| (0..3).map(move |c| p[c].abs_diff(q[c])))
            .max()
            .unwrap_or(0)
    };
    // A frame drawing the picture whole computes the water's light; the next keeps it.
    let computed = frame_srgb(&mut device, &mut still, &view, &day, None);
    let first = frame_srgb(&mut device, &mut live, &view, &day, out);
    let s = live.last_frame_stats();
    assert!(!s.world_cached && !s.water_kept && !s.water_statics_whole);
    let d = differ(&first, &computed);
    assert!(
        d <= 1,
        "the frame drawing the cached picture differs by {d}"
    );
    let held = frame_srgb(&mut device, &mut live, &view, &day, out);
    let s = live.last_frame_stats();
    assert!(s.world_cached && s.water_kept && s.water_statics_whole);
    assert!(
        held == first,
        "the first frame keeping the water's light changed"
    );
    let mut settled = false;
    for _ in 0..10 {
        let frame = frame_srgb(&mut device, &mut live, &view, &day, out);
        let s = live.last_frame_stats();
        assert!(s.water_kept && !s.water_statics_whole);
        settled |= s.cache_regions > 0 && s.water_statics_regions > 0;
        assert!(frame == first, "a frame changed as the ball settled");
    }
    assert!(settled, "the ball never settled into the picture");

    // The ball leaves the picture and moves on every frame, back over where its shadow was.
    let mut frames = vec![];
    for x in [0.5, 0.2, 0.5, -0.4] {
        for (r, ball) in [&mut still, &mut live].into_iter().zip(&balls) {
            r.set_prop_instance(*ball, &ball_at(x), true).expect("move");
        }
        let moved = frame_srgb(&mut device, &mut still, &view, &day, None);
        let frame = frame_srgb(&mut device, &mut live, &view, &day, out);
        let s = live.last_frame_stats();
        assert!(s.water_kept && !s.water_statics_whole && s.water_footprints > 0);
        assert!(!s.reshaded);
        let d = differ(&frame, &moved);
        assert!(d <= 1, "the frame with the ball at {x} differs by {d}");
        frames.push(moved);
    }
    assert!(
        frames[0] != computed && frames[1] != frames[0],
        "the ball did not move"
    );
    assert!(frames[3] == computed, "the ball did not return");

    // At night the lamp and the spot light the water.
    let computed = frame_srgb(&mut device, &mut still, &view, &night, None);
    let frame = frame_srgb(&mut device, &mut live, &view, &night, out);
    assert!(!live.last_frame_stats().water_kept);
    let d = differ(&frame, &computed);
    assert!(d <= 1, "the night frame differs by {d}");
    let frame = frame_srgb(&mut device, &mut live, &view, &night, out);
    let s = live.last_frame_stats();
    assert!(s.water_kept && s.water_statics_whole);
    let d = differ(&frame, &computed);
    assert!(d <= 1, "the held night frame differs by {d}");
    for (r, ball) in [&mut still, &mut live].into_iter().zip(&balls) {
        r.set_prop_instance(*ball, &ball_at(0.3), true)
            .expect("move");
    }
    let moved = frame_srgb(&mut device, &mut still, &view, &night, None);
    assert!(moved != computed, "the ball's shadow did not move at night");
    let frame = frame_srgb(&mut device, &mut live, &view, &night, out);
    assert!(live.last_frame_stats().water_footprints > 0);
    let d = differ(&frame, &moved);
    assert!(
        d <= 1,
        "the night frame after the ball moved differs by {d}"
    );

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    still.destroy_props(&mut device);
    live.destroy_props(&mut device);
}

/// Draws a frame of a renderer presenting to `target` (an offscreen RGBA8 texture standing
/// in for the canvas) and reads it back.
fn present(
    device: &mut Device<'_>,
    r: &mut Renderer,
    view: &View,
    env: &Environment,
    (target, readback): (TextureId, BufferId),
) -> Vec<u8> {
    let mut enc = device.encoder();
    r.encode(device, &mut enc, view, env, [0.0; 2], 0, Some(target))
        .expect("frame");
    enc.copy_texture_to_buffer(target, 0, 0, readback, 0);
    device.submit(enc).expect("submit");
    device.read_buffer(readback).expect("read back")
}

/// An offscreen RGBA8 target standing in for a `size` canvas, and a buffer to read it into.
fn canvas(device: &mut Device<'_>, size: (u32, u32)) -> (TextureId, BufferId) {
    canvas_as(device, size, Format::Rgba8Unorm)
}

/// [`canvas`] in `format`.
fn canvas_as(device: &mut Device<'_>, (w, h): (u32, u32), format: Format) -> (TextureId, BufferId) {
    let usage = TextureUsage::COLOR_TARGET | TextureUsage::COPY_SRC;
    let target = device
        .create_texture(&TextureDesc::new(w, h, format, usage))
        .expect("target");
    let readback = device
        .create_buffer(&BufferDesc {
            size: u64::from(w * h * format.texel_size()),
            usage: BufferUsage::COPY_DST,
            memory: Memory::Readback,
        })
        .expect("readback");
    (target, readback)
}

/// Asserts a frame drawn through the caches is the same frame drawn whole (`what` says
/// which): byte for byte on lavapipe (llvmpipe), where that holds; on another driver, whose
/// compilers may round the cache's shaders apart from the whole frame's (unchecked), within
/// one 8-bit step on 99.9 % of pixels.
fn assert_same_frame(device: &Device<'_>, cached: &[u8], whole: &[u8], what: &str) {
    let (off, largest) = off_by_more_than_a_step(cached, whole);
    if device.name().starts_with("llvmpipe") {
        let pixels = cached.chunks_exact(4).zip(whole.chunks_exact(4));
        let differ = pixels.filter(|(p, q)| p != q).count();
        assert_eq!(largest, 0, "{what}: {differ} pixels differ");
    } else {
        assert!(
            off <= 0.001,
            "{what}: {:.3} % of pixels off by more than a step (up to {largest})",
            off * 100.0
        );
    }
}

/// Of two pictures, the share of pixels where a channel differs by more than one 8-bit step,
/// and the largest difference.
fn off_by_more_than_a_step(a: &[u8], b: &[u8]) -> (f64, u8) {
    let (mut off, mut largest) = (0usize, 0u8);
    for (p, q) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let d = p
            .iter()
            .zip(q)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0);
        off += usize::from(d > 1);
        largest = largest.max(d);
    }
    (off as f64 / (a.len() / 4) as f64, largest)
}

/// A glowing ball by a matte one on a floor at half past five in the evening, drawn in `look`
/// by a `size` renderer presenting to `format`, with the view and the light to draw it in.
fn two_balls(
    device: &mut Device<'_>,
    look: &Look,
    size: (u32, u32),
    format: Format,
) -> (Renderer, View, Environment) {
    let mut world = World::default();
    world.materials.push(Material {
        name: "floor".into(),
        base_color: [0.6, 0.58, 0.55, 1.0],
        ..Material::default()
    });
    world.meshes.push(make_quad_mesh(0.0, 4.0));
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    let mut r = Renderer::new(
        device,
        &world,
        look.clone(),
        Quality::realtime(),
        size,
        Output::Surface(format),
    )
    .expect("renderer");
    r.finish_ao(device).expect("ao");
    let (sp, sn, suv, sidx) = sphere_prop_data();
    let sphere = r
        .create_prop_mesh(
            device,
            &MeshData {
                positions: &sp,
                normals: &sn,
                uvs: &suv,
                indices: &sidx,
            },
        )
        .expect("mesh");
    for (color, emissive, x) in [
        ([0.2, 0.5, 0.9, 1.0], [6.0, 3.0, 1.0], 0.5),
        ([0.8; 4], [0.0; 3], -0.6),
    ] {
        let material = r
            .create_prop_material(
                device,
                &MaterialDesc {
                    name: "ball".into(),
                    base_color: color,
                    emissive,
                    ..MaterialDesc::default()
                },
            )
            .expect("material");
        let mut at = math::IDENTITY;
        for k in [0, 5, 10] {
            at[k] = 0.4;
        }
        at[12..15].copy_from_slice(&[x, 0.45, 0.0]);
        r.create_prop_instance(sphere, material, &at).expect("ball");
    }
    let view = View {
        name: "balls".into(),
        eye: [0.0, 1.4, 2.6],
        target: [0.0, 0.4, 0.0],
        fov: 45.0,
        ..View::default()
    };
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 17.5,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &look.palette,
        1.0,
    );
    (r, view, env)
}

/// A Look whose grade changes nothing is tone-mapped without grading, and the picture is the
/// one its grade gives: within one 8-bit step on 99.9 % of pixels of the same frame graded a
/// hair from the identity, which takes the full grade.
#[test]
fn an_identity_grade_is_skipped_without_changing_the_picture() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let identity = Look {
        painterly: 0.0,
        saturation: 1.0,
        contrast: 1.0,
        lift: [0.0; 3],
        gain: [1.0; 3],
        ..Look::default()
    };
    assert!(identity.grade_is_identity() && !Look::default().grade_is_identity());
    let size = (128, 96);
    let (mut r, view, env) = two_balls(&mut device, &identity, size, Format::Rgba8Unorm);
    let out = canvas(&mut device, size);
    let hair = 1.0 + f32::EPSILON;
    for tonemap in ToneMap::ALL {
        let identity = Look {
            tonemap,
            ..identity.clone()
        };
        r.set_look(identity.clone());
        let plain = present(&mut device, &mut r, &view, &env, out);
        for graded in [
            Look {
                saturation: hair,
                ..identity.clone()
            },
            Look {
                contrast: hair,
                ..identity.clone()
            },
        ] {
            assert!(!graded.grade_is_identity());
            r.set_look(graded);
            let full = present(&mut device, &mut r, &view, &env, out);
            let (off, largest) = off_by_more_than_a_step(&plain, &full);
            assert!(
                off <= 0.001,
                "{tonemap:?}: {:.3} % of pixels off by more than a step (up to {largest})",
                off * 100.0
            );
        }
    }
    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    r.destroy_props(&mut device);
}

/// The fused tone-map output drawn grading each pixel once (`set_grade_once`: `fs_grade`
/// into a 32-bit float target, then `fs_tonemap_output` reading it) is the fused pass's
/// picture, compared in 32-bit floats: bit for bit on lavapipe, for an identity grade and a
/// full one, with sharpening and edge smoothing each on and off. On another driver, whose
/// compilers may round the two shaders apart, within one 8-bit step on 99.9 % of pixels.
#[test]
fn grading_once_draws_the_fused_picture() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let size = (128, 96);
    let look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let (mut fused, view, env) = two_balls(&mut device, &look, size, GRADED_FORMAT);
    let (mut once, ..) = two_balls(&mut device, &look, size, GRADED_FORMAT);
    assert!(once.set_grade_once(&mut device, true).expect("grade once"));
    let out = [(); 2].map(|_| canvas_as(&mut device, size, GRADED_FORMAT));
    let unorm8 = |frame: &[u8]| -> Vec<u8> {
        let channels = frame.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]);
        let to_8 = |c| (f32::from_le_bytes(c) * 255.0).round().clamp(0.0, 255.0) as u8;
        channels.map(to_8).collect()
    };
    let mut seen = vec![];
    for smoothing in [false, true] {
        let quality = Quality {
            fxaa: smoothing,
            ..Quality::realtime()
        };
        for identity in [false, true] {
            for sharpen in [0.0, 0.6] {
                let mut look = Look {
                    sharpen,
                    ..look.clone()
                };
                if identity {
                    (look.saturation, look.contrast) = (1.0, 1.0);
                    (look.lift, look.gain) = ([0.0; 3], [1.0; 3]);
                }
                let [a, b] = [(&mut fused, out[0]), (&mut once, out[1])].map(|(r, out)| {
                    r.set_quality(&mut device, quality).expect("quality");
                    r.set_look(look.clone());
                    present(&mut device, r, &view, &env, out)
                });
                let what = format!("smoothing {smoothing}, identity {identity}, sharpen {sharpen}");
                let stats = [&fused, &once].map(|r| r.last_frame_stats().graded_once);
                assert_eq!(stats, [false, true], "{what}");
                if device.name().starts_with("llvmpipe") {
                    let floats = a.chunks_exact(4).zip(b.chunks_exact(4));
                    let differ = floats.filter(|(x, y)| x != y).count();
                    assert_eq!(differ, 0, "{what}: {differ} floats differ");
                } else {
                    assert_same_frame(&device, &unorm8(&a), &unorm8(&b), &what);
                }
                // Each setting draws a picture of its own, so each is compared.
                let mut h = Fnv::new();
                h.write(&a);
                assert!(!seen.contains(&h.finish()), "{what}: as another setting");
                seen.push(h.finish());
            }
        }
    }
    once.resize(&mut device, (size.0 + 2, size.1))
        .expect("resize");
    assert!(once.grade_once(), "a resized renderer still grades once");
    assert!(!once.set_grade_once(&mut device, false).expect("fused"));
    assert!(!once.grade_once());
    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut fused, &mut once] {
        r.destroy_props(&mut device);
    }
}

/// The presented picture drawn through the caches frame after frame (the cached picture, the
/// kept props' shadow maps) against the same scene drawn whole: a second renderer whose
/// cached picture and props' shadow maps are thrown away before each of its frames. They
/// are byte-identical on lavapipe (`assert_same_frame`) while a prop that casts no shadow
/// moves, while a caster rests, is re-posed by a hair every frame (as a breathing character)
/// and moves a long way.
#[test]
fn cached_frames_match_frames_drawn_whole() {
    use crate::render::lights::{LightDesc, SpotCone};
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    world.materials.push(Material {
        name: "floor".into(),
        base_color: [0.6, 0.58, 0.55, 1.0],
        roughness: 0.8,
        alpha: AlphaMode::Opaque,
        shading: Shading::Standard,
        ..Material::default()
    });
    world.meshes.push(make_quad_mesh(0.0, 4.0));
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    // The fused tone-map output, as the table draws.
    let look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let (w, h) = (128, 96);
    let view = View {
        name: "table".into(),
        eye: [0.0, 1.9, 2.4],
        target: [0.0, 0.7, 0.0],
        fov: 45.0,
        ..View::default()
    };
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 15.0,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &look.palette,
        1.0,
    );
    let (sp, sn, suv, sidx) = sphere_prop_data();
    let place = |x: f32, z: f32, size: f32| {
        let mut m = math::IDENTITY;
        for k in [0, 5, 10] {
            m[k] = size;
        }
        m[12..15].copy_from_slice(&[x, 1.15, z]);
        m
    };
    // A table slab that settles into the cached picture, a figure over it casting shadows,
    // a pawn that casts none, and a spot light casting shadows.
    let make = |device: &mut Device<'_>| {
        let mut r = Renderer::new(
            device,
            &world,
            look.clone(),
            Quality::realtime(),
            (w, h),
            Output::Surface(Format::Rgba8Unorm),
        )
        .expect("renderer");
        r.finish_ao(device).expect("ao");
        let sphere = r
            .create_prop_mesh(
                device,
                &MeshData {
                    positions: &sp,
                    normals: &sn,
                    uvs: &suv,
                    indices: &sidx,
                },
            )
            .expect("mesh");
        let mut material = |name: &str, base_color, casts_shadow| {
            let desc = MaterialDesc {
                name: name.into(),
                base_color,
                casts_shadow,
                ..MaterialDesc::default()
            };
            r.create_prop_material(device, &desc).expect("material")
        };
        let wood = material("table", [0.5, 0.3, 0.2, 1.0], true);
        let skin = material("figure", [0.8, 0.6, 0.5, 1.0], true);
        let glass = material("pawn", [0.3, 0.5, 0.8, 1.0], false);
        let mut top = math::IDENTITY;
        top[0] = 1.2;
        top[5] = 0.06;
        top[10] = 0.8;
        top[13] = 0.75;
        r.create_prop_instance(sphere, wood, &top).expect("table");
        let figure = r
            .create_prop_instance(sphere, skin, &place(0.0, 0.1, 0.25))
            .expect("figure");
        let pawn = r
            .create_prop_instance(sphere, glass, &place(-0.45, 0.2, 0.1))
            .expect("pawn");
        r.create_light(&LightDesc {
            position: [0.3, 3.0, 0.4],
            range: 8.0,
            color: [1.0, 0.9, 0.8],
            intensity: 6.0,
            spot: Some(SpotCone {
                direction: [-0.1, -1.0, -0.1],
                inner: 0.35,
                outer: 0.6,
            }),
            follows_lamps: false,
            shadow: 1.0,
        })
        .expect("spot");
        (r, [figure, pawn], canvas(device, (w, h)))
    };
    let (mut cached, cached_props, cached_out) = make(&mut device);
    let (mut whole, whole_props, whole_out) = make(&mut device);

    // Draws a frame of both with the figure `figure` metres along and the pawn `pawn`
    // metres along, compares them, and returns what the cached frame kept and drew.
    let mut frame = |device: &mut Device<'_>, figure: f32, pawn: f32| {
        for (r, [f, p]) in [(&mut cached, cached_props), (&mut whole, whole_props)] {
            r.set_prop_instance(f, &place(figure, 0.1, 0.25), true)
                .expect("figure");
            r.set_prop_instance(p, &place(pawn - 0.45, 0.2, 0.1), true)
                .expect("pawn");
        }
        let a = present(device, &mut cached, &view, &env, cached_out);
        whole.world_cache = None;
        whole.prop_shadows_for = None;
        let b = present(device, &mut whole, &view, &env, whole_out);
        let what = format!("figure at {figure} m, pawn at {pawn} m");
        assert_same_frame(device, &a, &b, &what);
        (cached.last_frame_stats(), a)
    };

    frame(&mut device, 0.0, 0.0);
    // The pawn moves every frame and the table settles: the maps are kept.
    for k in 1..12 {
        let stats = frame(&mut device, 0.0, k as f32 * 1e-3).0;
        assert!(stats.prop_shadows_kept && !stats.reshaded, "frame {k}");
    }
    // The figure re-posed by a hair (a twentieth of a millimetre) every frame: drawn again.
    for k in 1..6 {
        let stats = frame(&mut device, k as f32 * 5e-5, 0.0).0;
        assert!(!stats.prop_shadows_kept, "hair {k}");
    }
    // Resting where the maps were last drawn, it settles into the cached picture under them.
    for k in 0..12 {
        let stats = frame(&mut device, 25e-5, 0.0).0;
        assert!(stats.prop_shadows_kept && !stats.reshaded, "rest {k}");
    }
    // Moved a long way, its shadow moves over the cached picture, shaded again.
    let before = frame(&mut device, 25e-5, 0.0).1;
    let (stats, after) = frame(&mut device, 0.3, 0.0);
    assert!(!stats.prop_shadows_kept && stats.reshaded);
    assert!(off_by_more_than_a_step(&before, &after).0 > 0.01);

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut cached, &mut whole] {
        r.destroy_props(&mut device);
    }
}

/// Every point of a receiver whose lookups `fs_reshade_mask` finds in a moved prop's
/// footprint lies in the pixels the points `shadow_volume` gives project to (as
/// `reshade_plan` rounds them out): in an oblique key light's cascade (its lookups off the
/// surface by up to the receivers' margin) and in a spot light's map, on a floor filling the
/// picture, a table and a wall across the spot's eye. On the floor those pixels are a small
/// part of the floor's. Near the eye, the key light's far cascade is not looked up.
#[test]
fn the_reshade_region_holds_every_point_that_sees_a_footprint() {
    let size = (160, 120);
    let (w, h) = (size.0 as f32, size.1 as f32);
    let view_proj = mul(
        &perspective(50f32.to_radians(), w / h, 0.1),
        &look_at([0.0, 1.6, 2.6], [0.0, 0.5, -0.4], [0.0, 1.0, 0.0]),
    );
    let key = mul(
        &ortho(-6.0, 6.0, -6.0, 6.0, 1.0, 60.0),
        &look_at(
            scale(normalize([0.7, 1.0, 0.4]), 30.0),
            [0.0; 3],
            [0.0, 1.0, 0.0],
        ),
    );
    let spot = mul(
        &math::perspective_depth(1.4, 1.0, 0.05, 6.0),
        &look_at([0.4, 2.2, -0.2], [0.4, 0.0, -0.2], [0.0, 0.0, 1.0]),
    );
    let grow = 0.04;
    let maps = [
        (key, 0.01, grow),
        (key, 0.01, grow),
        (spot, 0.02, 0.0),
        (spot, 0.02, 0.0),
    ];
    // A dealer's upper body over the table.
    let caster: (V3, V3) = ([-0.25, 0.8, -0.7], [0.25, 1.5, -0.35]);
    let floor = ([-8.0, 0.0, -8.0], [8.0, 0.0, 8.0]);
    let table = ([-1.2, 0.0, -0.9], [1.2, 0.76, 0.6]);
    let wall = ([-3.0, 0.0, -1.1], [3.0, 3.0, -1.0]);
    let mut seed = 0x2545_f491_u32;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32
    };
    let area = |r: Option<Rect>| r.map_or(0, |r| (r[2] - r[0]) * (r[3] - r[1]));
    let volume = |b, f: &Footprint| {
        let mut points = vec![];
        shadow_volume(b, f, &maps, &mut points);
        points
    };
    let reach = |b, f: &Footprint| {
        let points = volume(b, f);
        if points.is_empty() {
            return None;
        }
        screen_rect(&points, &view_proj, size, 1.0)
    };
    for v in [0, 2] {
        let (m, margin, grow) = &maps[v];
        let (r, near, _) = map_bounds(caster, m);
        let rect = [r[0] - margin, r[1] - margin, r[2] + margin, r[3] + margin];
        let f: Footprint = (v, rect, near);
        for (name, b) in [("floor", floor), ("table", table), ("wall", wall)] {
            let pixels = reach(b, &f);
            // Half the points anywhere in the box, half where the shadows fall.
            let near_shadows = (b.0.map(|x| x.max(-3.0)), b.1.map(|x| x.min(3.0)));
            let mut seen = 0;
            for i in 0..100_000 {
                let (lo, hi) = if i % 2 == 0 { b } else { near_shadows };
                let p: V3 = std::array::from_fn(|k| math::lerp(lo[k], hi[k], rand()));
                let off = normalize([rand() - 0.5, rand() - 0.5, rand() - 0.5]);
                let at = add(p, scale(off, grow * rand()));
                let c = transform(m, [at[0], at[1], at[2], 1.0]);
                // Behind a spot's eye, outside its cone: its shadows change nothing there.
                if c[3] <= 1e-6 {
                    continue;
                }
                let q = [c[0] / c[3], c[1] / c[3], c[2] / c[3]];
                let inside = rect[0] <= q[0] && q[0] <= rect[2];
                if !(inside && rect[1] <= q[1] && q[1] <= rect[3] && q[2] >= near) {
                    continue;
                }
                // Past a spot's range its shadows change nothing.
                if v == 2 && q[2] > 1.0 {
                    continue;
                }
                // Seen at the centre of a pixel there, that pixel is in the region.
                let s = transform(&view_proj, [p[0], p[1], p[2], 1.0]);
                let (x, y) = ((s[0] / s[3] * 0.5 + 0.5) * w, (s[1] / s[3] * 0.5 + 0.5) * h);
                if s[3] < 0.1 || x < 0.5 || y < 0.5 || x > w - 0.5 || y > h - 0.5 {
                    continue;
                }
                seen += 1;
                let r = pixels.unwrap_or_else(|| panic!("map {v}, {name}: {p:?} unheld"));
                let held = |lo: u32, hi: u32, z: f32| lo as f32 + 0.5 <= z && z <= hi as f32 - 0.5;
                assert!(
                    held(r[0], r[2], x) && held(r[1], r[3], y),
                    "map {v}, {name}: {p:?} at ({x}, {y}) outside {r:?}"
                );
            }
            assert!(
                seen > 1000,
                "map {v}, {name}: {seen} points see the footprint"
            );
        }
        // The floor's picture is all of it; the shadow's is a small part.
        let whole = screen_rect(&corners(floor), &view_proj, size, 2.0);
        assert_eq!(area(whole), size.0 * size.1);
        let shadow = area(reach(floor, &f));
        assert!(
            shadow * 8 < area(whole),
            "map {v}: the floor's region {shadow} of {} pixels",
            area(whole)
        );
        if v == 0 {
            let eye = [0.0, 1.6, 2.6];
            let points = volume(table, &(1, rect, near));
            assert!(near_cascade_only(&points, eye, 18.0, m, *grow));
            assert!(!near_cascade_only(&points, eye, 3.0, m, *grow));
        }
    }
}

/// Past a game light's range (depth 1 in its map) its share is 1 whatever moved: a wall
/// there sees nothing of a caster's footprint in the spot's map, nor does any receiver of a
/// caster past it, and a box across the range sees it only up to the range.
#[test]
fn nothing_past_a_spot_lights_range_is_reshaded() {
    let spot = mul(
        &math::perspective_depth(1.0, 1.0, 0.05, 3.0),
        &look_at([0.0, 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 1.0, 0.0]),
    );
    let key = (math::IDENTITY, 0.0, 0.0);
    let maps = [key, key, (spot, 0.02, 0.0), (spot, 0.02, 0.0)];
    let footprint = |caster: (V3, V3)| -> Footprint {
        let (r, near, _) = map_bounds(caster, &spot);
        (
            2,
            [r[0] - 0.02, r[1] - 0.02, r[2] + 0.02, r[3] + 0.02],
            near,
        )
    };
    let points = |b, f: &Footprint| {
        let mut points = vec![];
        shadow_volume(b, f, &maps, &mut points);
        points
    };
    let wall = |z: f32| ([-2.0, 0.0, z - 0.05], [2.0, 2.0, z + 0.05]);
    let ball = footprint(([-0.1, 0.9, -1.1], [0.1, 1.1, -0.9]));
    let across = points(([-2.0, 0.0, -3.5], [2.0, 2.0, -2.5]), &ball);
    assert!(!across.is_empty());
    let deepest = across
        .iter()
        .map(|p| transform(&spot, [p[0], p[1], p[2], 1.0]))
        .map(|c| c[2] / c[3])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(deepest < 1.0 + 1e-4, "{deepest}");
    assert!(points(wall(-3.5), &ball).is_empty());
    let beyond = footprint(([-0.1, 0.9, -3.4], [0.1, 1.1, -3.2]));
    assert!(points(wall(-4.0), &beyond).is_empty());
}

/// A caster re-posed by a hair every frame (a breathing dealer), then moved a little, low
/// over a floor filling the picture, by a slab settled into the picture under its shadow and
/// another far off, under a spot light: the cached frames are those drawn whole, byte for
/// byte on lavapipe (`assert_same_frame`), and the region re-shaded is around its shadows,
/// under half the floor's picture (all of it).
#[test]
fn a_caster_over_a_wide_floor_reshades_around_its_shadow() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    world.materials.push(Material {
        name: "floor".into(),
        base_color: [0.6, 0.58, 0.55, 1.0],
        roughness: 0.8,
        ..Material::default()
    });
    world.meshes.push(make_quad_mesh(0.0, 10.0));
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    let look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let (w, h) = (128, 96);
    let view = View {
        name: "floor".into(),
        eye: [0.0, 2.4, 2.6],
        target: [0.0, 0.0, -0.2],
        fov: 50.0,
        ..View::default()
    };
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 15.0,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &look.palette,
        1.0,
    );
    let (sp, sn, suv, sidx) = sphere_prop_data();
    let place = |at: V3, size: V3| {
        let mut m = math::IDENTITY;
        (m[0], m[5], m[10]) = (size[0], size[1], size[2]);
        m[12..15].copy_from_slice(&at);
        m
    };
    let ball_at = |x: f32| place([x, 0.45, 0.0], [0.3; 3]);
    let make = |device: &mut Device<'_>| {
        let mut r = Renderer::new(
            device,
            &world,
            look.clone(),
            Quality::realtime(),
            (w, h),
            Output::Surface(Format::Rgba8Unorm),
        )
        .expect("renderer");
        r.finish_ao(device).expect("ao");
        let sphere = r
            .create_prop_mesh(
                device,
                &MeshData {
                    positions: &sp,
                    normals: &sn,
                    uvs: &suv,
                    indices: &sidx,
                },
            )
            .expect("mesh");
        let mut material = |name: &str, base_color| {
            let desc = MaterialDesc {
                name: name.into(),
                base_color,
                ..MaterialDesc::default()
            };
            r.create_prop_material(device, &desc).expect("material")
        };
        let stone = material("slab", [0.4, 0.42, 0.45, 1.0]);
        let skin = material("ball", [0.8, 0.6, 0.5, 1.0]);
        for at in [[-0.6, 0.06, 0.4], [-3.0, 0.06, -2.5]] {
            r.create_prop_instance(sphere, stone, &place(at, [0.6, 0.06, 0.6]))
                .expect("slab");
        }
        let ball = r
            .create_prop_instance(sphere, skin, &ball_at(0.0))
            .expect("ball");
        r.create_light(&LightDesc {
            position: [0.2, 2.6, 0.3],
            range: 6.0,
            color: [1.0, 0.9, 0.8],
            intensity: 5.0,
            spot: Some(SpotCone {
                direction: [-0.1, -1.0, -0.1],
                inner: 0.4,
                outer: 0.7,
            }),
            follows_lamps: false,
            shadow: 1.0,
        })
        .expect("spot");
        (r, ball, canvas(device, (w, h)))
    };
    let (mut cached, cached_ball, cached_out) = make(&mut device);
    let (mut whole, whole_ball, whole_out) = make(&mut device);
    let mut frame = |device: &mut Device<'_>, x: f32| {
        for (r, ball) in [(&mut cached, cached_ball), (&mut whole, whole_ball)] {
            r.set_prop_instance(ball, &ball_at(x), true).expect("ball");
        }
        let a = present(device, &mut cached, &view, &env, cached_out);
        whole.world_cache = None;
        whole.prop_shadows_for = None;
        let b = present(device, &mut whole, &view, &env, whole_out);
        assert_same_frame(device, &a, &b, &format!("ball at {x} m"));
        (cached.last_frame_stats(), a)
    };
    // At rest, the slabs settle into the picture.
    for _ in 0..12 {
        frame(&mut device, 0.0);
    }
    let small = |s: &FrameStats| s.reshaded && s.reshade_pixels * 2 < w * h;
    for k in 1..6 {
        let stats = frame(&mut device, k as f32 * 5e-5).0;
        assert!(small(&stats), "hair {k}: {} pixels", stats.reshade_pixels);
    }
    let before = frame(&mut device, 25e-5).1;
    let (stats, after) = frame(&mut device, 0.12);
    assert!(small(&stats), "moved: {} pixels", stats.reshade_pixels);
    assert!(off_by_more_than_a_step(&before, &after).0 > 0.005);

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut cached, &mut whole] {
        r.destroy_props(&mut device);
    }
}

/// Two casters far down a wide floor moved by a hair every frame and then visibly: one whose
/// shadow lies across where the key light's near cascade blends into the far one (0.85 to 1
/// times the split from the eye), one whose shadow lies past the split, in the far cascade
/// alone. Every cached frame is the frame drawn whole, byte for byte on lavapipe
/// (`assert_same_frame`), the mask passing over the far cascade's footprints where a pixel's
/// shadow reads only the near one.
#[test]
fn shadows_across_the_cascade_split_reshade_exactly() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    world.materials.push(Material {
        name: "floor".into(),
        base_color: [0.6, 0.58, 0.55, 1.0],
        roughness: 0.8,
        ..Material::default()
    });
    world.meshes.push(make_quad_mesh(0.0, 40.0));
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    let look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let (w, h) = (192, 128);
    let view = View {
        name: "far".into(),
        eye: [0.0, 6.0, 0.0],
        target: [0.0, 0.0, -10.0],
        fov: 50.0,
        ..View::default()
    };
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 15.0,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &look.palette,
        1.0,
    );
    let (sp, sn, suv, sidx) = sphere_prop_data();
    let place = |at: V3, size: f32| {
        let mut m = math::IDENTITY;
        (m[0], m[5], m[10]) = (size, size, size);
        m[12..15].copy_from_slice(&at);
        m
    };
    // Where on the floor (y 0) the key light casts a point's shadow, and how far from the eye.
    let l = normalize(env.key_dir);
    let shadow_at = |p: V3| sub(p, scale(l, p[1] / l[1]));
    let from_eye = |p: V3| math::length(sub(shadow_at(p), view.eye));
    // The casters (radius 1.1 m, 0.15 m over the floor) along -z, their shadows' centres at
    // 0.925 and 1.35 times the split.
    let (size, y) = (1.3, 1.25);
    let make = |device: &mut Device<'_>| {
        let mut r = Renderer::new(
            device,
            &world,
            look.clone(),
            Quality::realtime(),
            (w, h),
            Output::Surface(Format::Rgba8Unorm),
        )
        .expect("renderer");
        r.finish_ao(device).expect("ao");
        let sphere = r
            .create_prop_mesh(
                device,
                &MeshData {
                    positions: &sp,
                    normals: &sn,
                    uvs: &suv,
                    indices: &sidx,
                },
            )
            .expect("mesh");
        let desc = MaterialDesc {
            name: "ball".into(),
            base_color: [0.8, 0.6, 0.5, 1.0],
            ..MaterialDesc::default()
        };
        let skin = r.create_prop_material(device, &desc).expect("material");
        let balls = [0, 1].map(|_| {
            r.create_prop_instance(sphere, skin, &place([0.0, y, -10.0], size))
                .expect("ball")
        });
        (r, balls, canvas(device, (w, h)))
    };
    let (mut cached, cached_balls, cached_out) = make(&mut device);
    let (mut whole, whole_balls, whole_out) = make(&mut device);
    let split = cached.cascades(&view, &env).split;
    let ats = [0.925, 1.35].map(|share| {
        let z = (0..400)
            .map(|k| -5.0 - 0.1 * k as f32)
            .min_by(|a, b| {
                let off = |z: f32| (from_eye([0.0, y, z]) - share * split).abs();
                off(*a).total_cmp(&off(*b))
            })
            .expect("a place");
        [0.0, y, z]
    });
    // The first casts from short of the blend to past the split, the second past the split:
    // the shadows of their boxes' corners from the eye.
    let reach = |at: V3| {
        let r = 0.85 * size;
        let corners = corners((at.map(|x| x - r), at.map(|x| x + r)));
        let d = corners.map(from_eye);
        (
            d.iter().copied().fold(f32::INFINITY, f32::min),
            d.iter().copied().fold(0.0, f32::max),
        )
    };
    let (near, far) = reach(ats[0]);
    assert!(
        near < 0.85 * split && split < far,
        "{near}..{far} of {split}"
    );
    assert!(reach(ats[1]).0 > split, "{:?} of {split}", reach(ats[1]));
    let mut frame = |device: &mut Device<'_>, x: f32| {
        for (r, balls) in [(&mut cached, cached_balls), (&mut whole, whole_balls)] {
            for (ball, at) in balls.into_iter().zip(ats) {
                let at = [at[0] + x, at[1], at[2]];
                r.set_prop_instance(ball, &place(at, size), true)
                    .expect("ball");
            }
        }
        let a = present(device, &mut cached, &view, &env, cached_out);
        whole.world_cache = None;
        whole.prop_shadows_for = None;
        let b = present(device, &mut whole, &view, &env, whole_out);
        assert_same_frame(device, &a, &b, &format!("balls at {x} m"));
        (cached.last_frame_stats(), a)
    };
    for _ in 0..12 {
        frame(&mut device, 0.0);
    }
    for k in 1..6 {
        let stats = frame(&mut device, k as f32 * 5e-5).0;
        assert!(stats.reshaded, "hair {k}");
    }
    let before = frame(&mut device, 25e-5).1;
    let (stats, after) = frame(&mut device, 0.6);
    assert!(stats.reshaded);
    assert!(off_by_more_than_a_step(&before, &after).0 > 0.002);

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut cached, &mut whole] {
        r.destroy_props(&mut device);
    }
}

/// A card table seen from a seat, for `reshade_plan`: 17 world draws (the room) and 40
/// settled props (the table, chairs, chips, cards) under the key light and a spot casting
/// shadows, and the footprints of the dealer breathing (her box in the three maps) and of ten
/// chips sliding (thirty).
struct CardTable {
    r: Renderer,
    settled: Vec<(V3, V3)>,
    eye: Eye,
    size: (u32, u32),
    cas: Cascades,
    /// Toward the key light, as the frame block holds it.
    key: V3,
    cases: [(&'static str, Vec<Footprint>); 2],
}

fn card_table(device: &mut Device<'_>) -> CardTable {
    let mut world = World::default();
    world.materials.push(Material::default());
    world.meshes.push(make_quad_mesh(0.0, 6.0));
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    let size = (1920, 1080);
    let mut r = Renderer::new(
        device,
        &world,
        Look::default(),
        Quality::realtime(),
        size,
        Output::Surface(Format::Rgba8Unorm),
    )
    .expect("renderer");
    r.create_light(&LightDesc {
        position: [0.0, 2.6, -0.2],
        range: 6.0,
        color: [1.0, 0.9, 0.8],
        intensity: 5.0,
        spot: Some(SpotCone {
            direction: [0.0, -1.0, 0.0],
            inner: 0.5,
            outer: 0.8,
        }),
        follows_lamps: false,
        shadow: 1.0,
    })
    .expect("spot");
    // The room: floor, ceiling, four walls, a bar, its stools and shelves, a rug, the table's
    // stand and two lamps over it.
    let rooms: [(V3, V3); 17] = [
        ([-6.0, 0.0, -6.0], [6.0, 0.0, 6.0]),
        ([-6.0, 3.2, -6.0], [6.0, 3.2, 6.0]),
        ([-6.0, 0.0, -6.0], [6.0, 3.2, -5.9]),
        ([-6.0, 0.0, 5.9], [6.0, 3.2, 6.0]),
        ([-6.0, 0.0, -6.0], [-5.9, 3.2, 6.0]),
        ([5.9, 0.0, -6.0], [6.0, 3.2, 6.0]),
        ([-4.0, 0.0, -5.5], [4.0, 1.1, -4.6]),
        ([-3.5, 0.0, -4.4], [-3.0, 0.8, -3.9]),
        ([-1.0, 0.0, -4.4], [-0.5, 0.8, -3.9]),
        ([1.5, 0.0, -4.4], [2.0, 0.8, -3.9]),
        ([-4.0, 1.4, -5.9], [4.0, 2.6, -5.6]),
        ([-5.9, 0.4, -3.0], [-5.4, 2.4, 3.0]),
        ([5.4, 0.4, -3.0], [5.9, 2.4, 3.0]),
        ([-2.5, 0.0, -2.0], [2.5, 0.01, 2.0]),
        ([-0.4, 0.0, -0.6], [0.4, 0.7, 0.2]),
        ([-0.9, 2.2, -0.5], [-0.6, 3.2, -0.2]),
        ([0.6, 2.2, -0.5], [0.9, 3.2, -0.2]),
    ];
    r.scene.draws = rooms
        .iter()
        .map(|&bounds| DrawItem {
            first_index: 0,
            count: 6,
            material: 0,
            water: false,
            bounds,
            max_extent: 1.0,
        })
        .collect();
    // The table, nine chairs, nine stacks of chips and nine pairs of cards, the board and
    // the dealer's tray.
    let mut settled: Vec<(V3, V3)> = vec![([-1.3, 0.0, -1.0], [1.3, 0.78, 0.6])];
    for k in 0..9 {
        let a = std::f32::consts::PI * (0.15 + 0.7 * k as f32 / 8.0);
        let at = |d: f32, lo: V3, hi: V3| {
            let (x, z) = (d * a.cos(), d * a.sin() * 0.7 - 0.2);
            ([x + lo[0], lo[1], z + lo[2]], [x + hi[0], hi[1], z + hi[2]])
        };
        settled.push(at(1.6, [-0.25, 0.0, -0.25], [0.25, 1.0, 0.25]));
        settled.push(at(1.0, [-0.03, 0.76, -0.03], [0.03, 0.82, 0.03]));
        settled.push(at(0.8, [-0.06, 0.76, -0.04], [0.06, 0.765, 0.04]));
    }
    for k in 0..12 {
        let x = -0.5 + k as f32 * 0.09;
        settled.push(([x - 0.035, 0.76, -0.25], [x + 0.035, 0.765, -0.15]));
    }
    settled.push(([-0.6, 0.76, -0.7], [-0.4, 0.8, -0.55]));
    let view = View {
        name: "seat".into(),
        eye: [0.0, 1.45, 1.6],
        target: [0.0, 0.8, -0.3],
        fov: 50.0,
        ..View::default()
    };
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 15.0,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &Look::default().palette,
        1.0,
    );
    let cas = r.cascades(&view, &env);
    let eye = r.eye(&view, [0.0; 2], size);
    let maps = r.prop_shadow_maps(&cas);
    r.receivers = Some(ReceiverBounds::new(
        0,
        &r.scene.draws,
        &maps,
        &eye.view_proj,
        size,
    ));
    // The dealer's upper body as it was and is, across the table.
    let dealer = ([-0.3, 0.75, -1.2], [0.3, 1.65, -0.8]);
    let dealer = r.shadow_footprints(&[dealer], &cas, true);
    let chips: Vec<(V3, V3)> = (0..10)
        .map(|k| {
            let x = -0.3 + 0.06 * k as f32;
            ([x - 0.02, 0.76, -0.05], [x + 0.02, 0.8, 0.03])
        })
        .collect();
    let chips = r.shadow_footprints(&chips, &cas, true);
    assert_eq!((dealer.len(), chips.len()), (3, 30));
    let cases = [("dealer", dealer), ("ten chips", chips)];
    CardTable {
        r,
        settled,
        eye,
        size,
        cas,
        key: normalize(env.key_dir),
        cases,
    }
}

/// The region `reshade_plan` gives holds every pixel where a receiver can see one of the
/// footprints, as each footprint's `shadow_volume` alone bounds them, whatever work it saves:
/// passing receivers it holds already, bounding each map's footprints as one and, past its
/// budget, taking receivers whole (with none, the region before the volumes bounded it: every
/// receiver in reach whole). The dealer's region is the one the volumes give, the ten chips'
/// a little more.
#[test]
fn the_reshade_plan_holds_every_pixel_a_footprint_reaches() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let CardTable {
        r,
        settled,
        eye,
        size,
        cas,
        cases,
        ..
    } = card_table(&mut device);
    let receivers = r.receivers.as_ref().expect("receivers");
    let maps = r.prop_shadow_maps(&cas);
    let (m0, _, grow0) = &maps[0];
    // Each receiver in view and in reach of the footprints, its bounds in the maps and its
    // pixels.
    let in_reach = |footprints: &[Footprint]| {
        let draws = (r
            .scene
            .draws
            .iter()
            .zip(&receivers.maps)
            .zip(&receivers.screen))
        .filter_map(|((d, m), s)| s.map(|s| (d.bounds, *m, s)));
        let settled = settled.iter().filter_map(|b| {
            let bounds = std::array::from_fn(|v| {
                let (m, _, grow) = &maps[v];
                let (r, _, far) = map_bounds((b.0.map(|x| x - grow), b.1.map(|x| x + grow)), m);
                (r, far)
            });
            let s = screen_rect(&corners(*b), &eye.view_proj, size, 2.0)?;
            Some((*b, bounds, s))
        });
        draws
            .chain(settled)
            .filter(|(_, bounds, _)| reached(footprints, bounds))
            .collect::<Vec<_>>()
    };
    let area = |r: Rect| (r[2] - r[0]) * (r[3] - r[1]);
    let holds = |a: Rect, b: Rect| union(a, b) == a;
    for (name, footprints) in &cases {
        let receivers = in_reach(footprints);
        let mut points = vec![];
        let volumes = receivers.iter().flat_map(|(b, bounds, s)| {
            let reaching = footprints
                .iter()
                .filter(|f| reached(std::slice::from_ref(f), bounds));
            reaching
                .filter_map(|f| {
                    shadow_volume(*b, f, &maps, &mut points);
                    if f.0 == 1 && near_cascade_only(&points, eye.pos, cas.split, m0, *grow0) {
                        return None;
                    }
                    screen_rect(&points, &eye.view_proj, size, 1.0).and_then(|r| intersect(r, *s))
                })
                .collect::<Vec<_>>()
        });
        let exact = volumes.reduce(union).expect("pixels in reach");
        let whole = receivers.iter().map(|(_, _, s)| *s).reduce(union);
        let plan = |budget| {
            let plan = r.reshade_plan(footprints, &settled, &eye, size, &cas, budget);
            plan.expect("a plan").region
        };
        let (unbounded, bounded, none) = (plan(u32::MAX), plan(MAX_RESHADE_PAIRS), plan(0));
        assert!(
            holds(unbounded, exact),
            "{name}: {unbounded:?} misses {exact:?}"
        );
        assert!(
            holds(bounded, unbounded),
            "{name}: {bounded:?} misses {unbounded:?}"
        );
        assert_eq!(Some(none), whole, "{name}");
        assert!(
            area(bounded) * 3 < area(none),
            "{name}: {bounded:?} of {none:?}"
        );
        if *name == "dealer" {
            assert_eq!(bounded, exact);
        } else {
            assert!(
                area(bounded) * 10 < area(exact) * 11,
                "{bounded:?} for {exact:?}"
            );
        }
    }
}

/// The planes bounding the convex hull of `points`: through three of them, with all of them
/// on the inner side (`h · [p, 1] >= 0`), normals of unit length.
fn hull_planes(points: &[V3]) -> Vec<[f32; 4]> {
    let n = points.len();
    let triples =
        (0..n).flat_map(|i| (i + 1..n).flat_map(move |j| (j + 1..n).map(move |k| [i, j, k])));
    triples
        .filter_map(|[i, j, k]| {
            let (a, b, c) = (points[i], points[j], points[k]);
            let normal = cross(sub(b, a), sub(c, a));
            let length = math::length(normal);
            if length < 1e-9 {
                return None;
            }
            let normal = scale(normal, 1.0 / length);
            let h = [normal[0], normal[1], normal[2], -math::dot(normal, a)];
            let side = |p: &V3| math::dot(normal, *p) + h[3];
            let (lo, hi) = points
                .iter()
                .map(side)
                .fold((0.0f32, 0.0f32), |(lo, hi), d| (lo.min(d), hi.max(d)));
            let flat = 1e-6;
            match (lo >= -flat, hi <= flat) {
                (true, _) => Some(h),
                (_, true) => Some(h.map(|x| -x)),
                _ => None,
            }
        })
        .collect()
}

/// At the card table, with the key light's cascades as a frame makes them: wherever a point
/// on a receiver looks up the dealer's footprint in a cascade (its lookup moved off the
/// surface along a normal up to 1.001 long, as `shadow_cascade` moves it), the point itself
/// passes `fs_reshade_mask`'s first test (the footprint grown by `lookup_margins`) and lies
/// in the convex hull of the points `shadow_volume` gives. And where a point passes as
/// reading only the near cascade with those margins (`near_cascade_only` in scene.wgsl), so
/// does each of its lookups, as `shadow_taps` reads it: short of the blend, inside the near
/// cascade's map, at the receivers and by the map's borders.
#[test]
fn every_key_lookup_in_a_footprint_is_in_its_shadow_volume() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let CardTable {
        r,
        settled,
        eye,
        cas,
        key,
        cases,
        ..
    } = card_table(&mut device);
    let maps = r.prop_shadow_maps(&cas);
    let margins = Renderer::lookup_margins(&maps);
    let mut seed = 0x9e37_79b9_u32;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32
    };
    // A normal as the picture's aux gives it back (up to 1.001 long, as often a full one).
    let normal = |i: usize, rand: &mut dyn FnMut() -> f32| {
        let dir = normalize([rand() - 0.5, rand() - 0.5, rand() - 0.5]);
        scale(dir, if i % 4 < 2 { 1.001 } else { 1.001 * rand() })
    };
    // Where `shadow_cascade` looks up cascade `v` for a point `p` with normal `n`.
    let lookup = |v: usize, p: V3, n: V3| {
        let grow = 1.0 + 1.5 * (1.0 - math::dot(n, key).abs());
        add(p, scale(n, cas.offset[v] * grow))
    };
    let clip = |v: usize, p: V3| {
        let c = transform(&maps[v].0, [p[0], p[1], p[2], 1.0]);
        [c[0] / c[3], c[1] / c[3], c[2] / c[3]]
    };
    // `near_cascade_only` in scene.wgsl: a point `dist` from the eye, its lookup in the near
    // cascade within `m` of clip coordinates `c` there.
    let near_only = |c: V3, dist: f32, m: [f32; 2]| {
        let e = [m[0] + 1e-4, m[1] + 1e-4];
        dist < cas.split * 0.85 * (1.0 - 1e-4)
            && c[0].abs() < 0.996 - e[0]
            && c[1].abs() < 0.996 - e[0]
            && e[1] < c[2]
            && c[2] < 1.0 - e[1]
    };
    let receivers = [
        ("floor", r.scene.draws[0].bounds),
        ("rug", r.scene.draws[13].bounds),
        ("stand", r.scene.draws[14].bounds),
        ("table", settled[0]),
    ];
    let dealer = &cases[0].1;
    let (mut found, mut near_skips) = ([0; 2], 0);
    for v in 0..2 {
        let f = dealer.iter().find(|f| f.0 == v).expect("a footprint");
        let (_, rect, near) = *f;
        for (name, b) in receivers {
            let mut points = vec![];
            shadow_volume(b, f, &maps, &mut points);
            let hull = hull_planes(&points);
            // Half the points anywhere on the receiver, half under the dealer.
            let under = (b.0.map(|x| x.max(-1.5)), b.1.map(|x| x.min(1.5)));
            for i in 0..40_000 {
                let (lo, hi) = if i % 2 == 0 { b } else { under };
                let p: V3 = std::array::from_fn(|k| math::lerp(lo[k], hi[k], rand()));
                let n = normal(i, &mut rand);
                let q = clip(v, lookup(v, p, n));
                let inside = rect[0] <= q[0] && q[0] <= rect[2] && rect[1] <= q[1];
                if !(inside && q[1] <= rect[3] && q[2] >= near) {
                    continue;
                }
                found[v] += 1;
                let [xy, depth] = margins[v];
                let at = clip(v, p);
                assert!(
                    rect[0] - xy <= at[0]
                        && at[0] <= rect[2] + xy
                        && rect[1] - xy <= at[1]
                        && at[1] <= rect[3] + xy
                        && at[2] >= near - depth,
                    "map {v}, {name}: {p:?} (normal {n:?}) fails the early test"
                );
                let side = |h: &[f32; 4]| h[0] * p[0] + h[1] * p[1] + h[2] * p[2] + h[3];
                let outside = hull.iter().map(side).fold(0.0f32, f32::min);
                assert!(
                    outside >= 0.0,
                    "map {v}, {name}: {p:?} (normal {n:?}) {outside} m out of its volume"
                );
                let dist = math::length(sub(p, eye.pos));
                if v == 1 && near_only(clip(0, p), dist, margins[0]) {
                    near_skips += 1;
                    assert!(
                        near_only(clip(0, lookup(0, p, n)), dist, [0.0; 2]),
                        "{name}: {p:?} (normal {n:?}) reads the far cascade"
                    );
                }
            }
        }
    }
    assert!(found[0] > 5000 && found[1] > 5000, "{found:?}");
    assert!(near_skips > 1000, "{near_skips} near the eye");
    // By the near cascade's borders: points whose clip coordinates there lie within a few
    // margins of one (inside and out).
    let inv = invert(&maps[0].0).expect("a cascade");
    let [xy, depth] = margins[0];
    let mut held = 0;
    for i in 0..100_000 {
        let mut c: V3 = [2.0 * rand() - 1.0, 2.0 * rand() - 1.0, rand()];
        let off = (2.0 * rand() - 0.5) * 3.0;
        let high = rand() < 0.5;
        c[i % 3] = match (i % 3, high) {
            (2, true) => 1.0 - off * depth,
            (2, false) => off * depth,
            (_, true) => 0.996 - off * xy,
            (_, false) => off * xy - 0.996,
        };
        let w = transform(&inv, [c[0], c[1], c[2], 1.0]);
        let p = [w[0] / w[3], w[1] / w[3], w[2] / w[3]];
        if near_only(clip(0, p), 0.0, margins[0]) {
            held += 1;
            let n = normal(i, &mut rand);
            assert!(
                near_only(clip(0, lookup(0, p, n)), 0.0, [0.0; 2]),
                "{p:?} (normal {n:?}) looks up past the near cascade's border"
            );
        }
    }
    assert!(held > 10_000, "{held} by the borders");
}

/// What `reshade_plan` costs the CPU a frame at the card table, while the dealer breathes
/// and while ten chips slide. A timing, not a check:
/// `cargo test --release -p ganbate-engine --lib reshade_plan_cost -- --ignored --nocapture`.
#[test]
#[ignore = "a timing"]
fn reshade_plan_cost() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let CardTable {
        r,
        settled,
        eye,
        size,
        cas,
        cases,
        ..
    } = card_table(&mut device);
    for (name, footprints) in &cases {
        let plan =
            |footprints| r.reshade_plan(footprints, &settled, &eye, size, &cas, MAX_RESHADE_PAIRS);
        let region = plan(footprints).map_or([0; 4], |p| p.region);
        let runs = 2000;
        let best = (0..7)
            .map(|_| {
                let start = std::time::Instant::now();
                for _ in 0..runs {
                    std::hint::black_box(plan(std::hint::black_box(footprints)));
                }
                start.elapsed().as_secs_f64() / f64::from(runs)
            })
            .fold(f64::INFINITY, f64::min);
        println!(
            "reshade_plan, {name}: {:.1} us a frame, region {region:?} ({} pixels)",
            best * 1e6,
            (region[2] - region[0]) * (region[3] - region[1])
        );
    }
}
