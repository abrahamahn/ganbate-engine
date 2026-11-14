use super::*;

#[test]
fn air_sky_and_flower_shaders_compile_on_all_backends() {
    let opts = CompileOptions::default();
    for (src, stage, entry) in [
        (POST, Stage::Fragment, "fs_air"),
        (POST, Stage::Fragment, "fs_tonemap_output"),
        (POST, Stage::Fragment, "fs_tonemap_output_plain"),
        (POST, Stage::Fragment, "fs_tonemap_final"),
        (WORLD_FX, Stage::Fragment, "fs_sky"),
        (WORLD_FX, Stage::Vertex, "vs_particle"),
        (SCENE, Stage::Vertex, "vs_flower"),
        (SCENE, Stage::Fragment, "fs_flower"),
    ] {
        compile(src, stage, entry).unwrap_or_else(|e| panic!("{entry} spirv: {e}"));
        let composed = ganbate_shader::compose(src, &shader_modules()).expect("compose");
        let glsl = compile_glsl_with_options(&composed, stage, entry, &opts)
            .unwrap_or_else(|e| panic!("{entry} glsl: {e}"));
        let g = glsl.glsl.expect("glsl");
        assert!(
            g.units.len() <= 16,
            "{entry} uses {} WebGL texture units",
            g.units.len()
        );
        compile_msl_with_options(&composed, stage, entry, &opts)
            .unwrap_or_else(|e| panic!("{entry} msl: {e}"));
        compile_wgsl_with_options(&composed, stage, entry, &opts)
            .unwrap_or_else(|e| panic!("{entry} wgsl: {e}"));
    }
}

fn paint_allocated(r: &Renderer) -> bool {
    let tex = r.targets.ldr.is_some() && r.targets.paint.is_some();
    let slots_all = r
        .slots
        .iter()
        .all(|s| s.paint.is_some() && s.output.is_some());
    let slots_none = r
        .slots
        .iter()
        .all(|s| s.paint.is_none() && s.output.is_none());
    assert_eq!(r.targets.ldr.is_some(), r.targets.paint.is_some());
    assert!(
        (tex && slots_all) || (!tex && slots_none),
        "paint targets and slot bind groups must be allocated together"
    );
    tex
}

#[test]
fn paint_targets_allocate_lazily_across_still_surface_resize_and_set_quality() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let size = (128, 96);
    let plain_look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let painted_look = Look {
        painterly: 15.0,
        painterly_far: 1.4,
        ..Look::default()
    };

    // 1. Output::Still: non-painted frame leaves ldr/paint None; enabling DoF and painterly
    // lazily allocates them and matches fresh Output::Still renderers byte for byte.
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
    let mut still_lazy = Renderer::new(
        &mut device,
        &world,
        plain_look.clone(),
        Quality::realtime(),
        size,
        Output::Still,
    )
    .expect("still lazy");
    assert!(!paint_allocated(&still_lazy));
    let view = View {
        name: "still".into(),
        eye: [0.0, 1.4, 2.6],
        target: [0.0, 0.2, 0.0],
        fov: 45.0,
        ..View::default()
    };
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 17.5,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &plain_look.palette,
        1.0,
    );
    let plain_still = still_lazy
        .render_still(&mut device, &view, &env, 1)
        .expect("plain still");
    assert!(!paint_allocated(&still_lazy));
    assert!(!still_lazy.last_frame_stats().painted);

    let dof_view = View {
        focus: Some(Focus {
            distance: 1.5,
            depth: 0.2,
            blur: 0.05,
        }),
        ..view.clone()
    };
    let lazy_dof_still = still_lazy
        .render_still(&mut device, &dof_view, &env, 1)
        .expect("lazy dof still");
    assert!(paint_allocated(&still_lazy));
    assert!(still_lazy.last_frame_stats().painted);
    assert_ne!(lazy_dof_still, plain_still);

    let mut still_fresh_dof = Renderer::new(
        &mut device,
        &world,
        plain_look.clone(),
        Quality::realtime(),
        size,
        Output::Still,
    )
    .expect("still fresh dof");
    let fresh_dof_still = still_fresh_dof
        .render_still(&mut device, &dof_view, &env, 1)
        .expect("fresh dof still");
    assert_same_frame(
        &device,
        &lazy_dof_still,
        &fresh_dof_still,
        "Output::Still lazy DoF",
    );

    still_lazy.set_look(painted_look.clone());
    let lazy_painted_still = still_lazy
        .render_still(&mut device, &view, &env, 1)
        .expect("lazy painted still");
    let mut still_fresh_paint = Renderer::new(
        &mut device,
        &world,
        painted_look.clone(),
        Quality::realtime(),
        size,
        Output::Still,
    )
    .expect("still fresh paint");
    let fresh_painted_still = still_fresh_paint
        .render_still(&mut device, &view, &env, 1)
        .expect("fresh painted still");
    assert_same_frame(
        &device,
        &lazy_painted_still,
        &fresh_painted_still,
        "Output::Still lazy painterly",
    );

    // 2. Output::Surface (2 frame slots): non-painted uncached + cached frames leave ldr/paint None.
    let (mut r, view, env) = two_balls(&mut device, &plain_look, size, Format::Rgba8Unorm);
    let (mut whole, ..) = two_balls(&mut device, &plain_look, size, Format::Rgba8Unorm);
    let out_a = canvas(&mut device, size);
    let out_b = canvas(&mut device, size);
    assert!(!paint_allocated(&r));
    for frame in 0..2 {
        let a = present(&mut device, &mut r, &view, &env, out_a);
        whole.forget_world_cache();
        let b = present(&mut device, &mut whole, &view, &env, out_b);
        assert!(!paint_allocated(&r), "frame {frame}");
        assert!(!r.last_frame_stats().painted, "frame {frame}");
        assert_eq!(r.last_frame_stats().world_cached, frame > 0);
        assert_same_frame(&device, &a, &b, &format!("plain frame {frame}"));
    }

    // Fused tonemap (`!quality.painterly`) never allocates ldr/paint even when Look requests paint + DoF.
    let fused_q = Quality {
        painterly: false,
        ..Quality::realtime()
    };
    r.set_quality(&mut device, fused_q).expect("fused quality");
    r.set_look(painted_look.clone());
    let dof_balls = View {
        focus: Some(Focus {
            distance: 1.2,
            depth: 0.15,
            blur: 0.05,
        }),
        ..view.clone()
    };
    let _ = present(&mut device, &mut r, &dof_balls, &env, out_a);
    assert!(!paint_allocated(&r));
    assert!(r.last_frame_stats().fused_tonemap && !r.last_frame_stats().painted);

    // Restore realtime quality and plain look; turn DoF blur on across both slots.
    r.set_quality(&mut device, Quality::realtime())
        .expect("realtime");
    r.set_look(plain_look.clone());
    assert!(!paint_allocated(&r));
    for slot_pass in 0..2 {
        let a = present(&mut device, &mut r, &dof_balls, &env, out_a);
        whole.forget_world_cache();
        let b = present(&mut device, &mut whole, &dof_balls, &env, out_b);
        assert!(paint_allocated(&r), "DoF slot pass {slot_pass}");
        assert!(r.last_frame_stats().painted, "DoF slot pass {slot_pass}");
        assert_same_frame(&device, &a, &b, &format!("surface DoF pass {slot_pass}"));
    }

    // 3. resize resets ldr/paint to None until needed again, then matches a fresh resized renderer.
    let resized = (160, 120);
    r.resize(&mut device, resized).expect("resize");
    whole.resize(&mut device, resized).expect("resize whole");
    assert!(!paint_allocated(&r));
    let out_ra = canvas(&mut device, resized);
    let out_rb = canvas(&mut device, resized);
    let a = present(&mut device, &mut r, &view, &env, out_ra);
    whole.forget_world_cache();
    let b = present(&mut device, &mut whole, &view, &env, out_rb);
    assert!(!paint_allocated(&r));
    assert_same_frame(&device, &a, &b, "resized plain");

    r.set_look(painted_look.clone());
    whole.set_look(painted_look.clone());
    for slot_pass in 0..2 {
        let a = present(&mut device, &mut r, &view, &env, out_ra);
        whole.forget_world_cache();
        let b = present(&mut device, &mut whole, &view, &env, out_rb);
        assert!(
            paint_allocated(&r),
            "resized painterly slot pass {slot_pass}"
        );
        assert!(r.last_frame_stats().painted);
        assert_same_frame(
            &device,
            &a,
            &b,
            &format!("resized painterly pass {slot_pass}"),
        );
    }

    // 4. set_quality resets ldr/paint to None until needed again, then matches byte for byte.
    let tweaked_q = Quality {
        fxaa: false,
        ao_scale: 1.0,
        ..Quality::realtime()
    };
    r.set_look(plain_look);
    r.set_quality(&mut device, tweaked_q).expect("set_quality");
    whole
        .set_quality(&mut device, tweaked_q)
        .expect("set_quality whole");
    assert!(!paint_allocated(&r));
    let _ = present(&mut device, &mut r, &view, &env, out_ra);
    assert!(!paint_allocated(&r));

    r.set_look(painted_look.clone());
    whole.set_look(painted_look);
    for slot_pass in 0..2 {
        let a = present(&mut device, &mut r, &dof_balls, &env, out_ra);
        whole.forget_world_cache();
        let b = present(&mut device, &mut whole, &dof_balls, &env, out_rb);
        assert!(paint_allocated(&r), "set_quality slot pass {slot_pass}");
        assert!(r.last_frame_stats().painted);
        assert_same_frame(
            &device,
            &a,
            &b,
            &format!("set_quality painterly+DoF pass {slot_pass}"),
        );
    }

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [
        &mut r,
        &mut whole,
        &mut still_lazy,
        &mut still_fresh_dof,
        &mut still_fresh_paint,
    ] {
        r.destroy_props(&mut device);
    }
}

#[test]
fn tonemap_passes_read_near_sharpen_without_binding_depth() {
    for (entry, spec, expected) in [
        ("fs_tonemap_output", &[][..], &[0u32, 3, 4, 5][..]),
        ("fs_tonemap_output_plain", &[], &[0, 3, 4, 5]),
        ("fs_tonemap_output", &GRADED, &[0, 3]),
        ("fs_tonemap_final", &[], &[0, 3, 4, 5, 6]),
    ] {
        let c = compile_specialized(POST, Stage::Fragment, entry, spec).expect(entry);
        let bound: Vec<u32> = sorted_bindings(&c.bindings)
            .iter()
            .map(|b| b.binding)
            .collect();
        assert_eq!(bound, expected, "{entry} ({spec:?})");
    }
    for m in 0..3 {
        let s = [("TONEMAP_MODE", SpecValue::I32(m))];
        for entry in [
            "fs_tonemap_output",
            "fs_tonemap_output_plain",
            "fs_grade",
            "fs_grade_plain",
        ] {
            let c = compile_specialized(POST, Stage::Fragment, entry, &s).expect(entry);
            let bound: Vec<u32> = sorted_bindings(&c.bindings)
                .iter()
                .map(|b| b.binding)
                .collect();
            assert_eq!(bound, &[0u32, 3, 4, 5], "{entry} TONEMAP_MODE={m}");
        }
    }
}

#[test]
fn reshade_mask_omits_frag_depth_and_matches_still_frames() {
    use crate::render::lights::{LightDesc, SpotCone};

    let spv = compile(SCENE, Stage::Fragment, "fs_reshade_mask").expect("spirv");
    assert!(spv.outputs.is_empty(), "fs_reshade_mask has colour outputs");
    let mut idx = 5;
    let mut saw_kill = false;
    while idx < spv.words.len() {
        let word = spv.words[idx];
        let count = (word >> 16) as usize;
        let op = word & 0xffff;
        assert!(count > 0 && idx + count <= spv.words.len());
        // OpExecutionMode (16): mode 12 is DepthReplacing.
        if op == 16 && count >= 3 {
            assert_ne!(
                spv.words[idx + 2],
                12,
                "fs_reshade_mask emitted SPIR-V DepthReplacing"
            );
        }
        // OpDecorate (71) / OpMemberDecorate (72): Decoration::BuiltIn (11) FragDepth (22).
        if op == 71 && count >= 4 && spv.words[idx + 2] == 11 {
            assert_ne!(
                spv.words[idx + 3],
                22,
                "fs_reshade_mask decorated FragDepth"
            );
        }
        if op == 72 && count >= 5 && spv.words[idx + 3] == 11 {
            assert_ne!(
                spv.words[idx + 4],
                22,
                "fs_reshade_mask member-decorated FragDepth"
            );
        }
        // OpKill (252).
        if op == 252 {
            saw_kill = true;
        }
        idx += count;
    }
    assert!(saw_kill, "fs_reshade_mask must discard unchanged pixels");

    let composed = ganbate_shader::compose(SCENE, &shader_modules()).expect("compose");
    let opts = CompileOptions::default();
    let glsl = compile_glsl_with_options(&composed, Stage::Fragment, "fs_reshade_mask", &opts)
        .expect("glsl")
        .glsl
        .expect("glsl source");
    assert!(!glsl.source.contains("gl_FragDepth"));
    assert!(glsl.source.contains("discard"));
    let msl = compile_msl_with_options(&composed, Stage::Fragment, "fs_reshade_mask", &opts)
        .expect("msl")
        .msl
        .expect("msl source");
    assert!(!msl.source.contains("[[depth("));
    assert!(msl.source.contains("discard_fragment()"));
    let wgsl = compile_wgsl_with_options(&composed, Stage::Fragment, "fs_reshade_mask", &opts)
        .expect("wgsl")
        .wgsl
        .expect("wgsl source");
    assert!(!wgsl.source.contains("frag_depth"));
    assert!(wgsl.source.contains("discard"));

    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    world.materials.push(Material {
        name: "floor".into(),
        base_color: [0.62, 0.6, 0.56, 1.0],
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
    let look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let size = (80, 60);
    let view = View {
        name: "reshade".into(),
        eye: [0.0, 1.9, 2.4],
        target: [0.0, 0.6, 0.0],
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
    let place = |x: f32, y: f32, z: f32, s: f32| {
        let mut m = math::IDENTITY;
        for k in [0, 5, 10] {
            m[k] = s;
        }
        m[12..15].copy_from_slice(&[x, y, z]);
        m
    };
    let setup = |device: &mut Device<'_>, output: Output| {
        let mut r = Renderer::new(
            device,
            &world,
            look.clone(),
            Quality::realtime(),
            size,
            output,
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
        let receiver_mat = r
            .create_prop_material(
                device,
                &MaterialDesc {
                    name: "slab".into(),
                    base_color: [0.55, 0.36, 0.24, 1.0],
                    casts_shadow: true,
                    ..MaterialDesc::default()
                },
            )
            .expect("slab mat");
        let caster_mat = r
            .create_prop_material(
                device,
                &MaterialDesc {
                    name: "caster".into(),
                    base_color: [0.82, 0.62, 0.5, 1.0],
                    casts_shadow: true,
                    ..MaterialDesc::default()
                },
            )
            .expect("caster mat");
        let mut slab = math::IDENTITY;
        (slab[0], slab[5], slab[10], slab[13]) = (1.1, 0.06, 0.8, 0.65);
        r.create_prop_instance(sphere, receiver_mat, &slab)
            .expect("slab");
        let caster = r
            .create_prop_instance(sphere, caster_mat, &place(-0.25, 1.15, 0.05, 0.24))
            .expect("caster");
        r.create_light(&LightDesc {
            position: [0.25, 2.8, 0.35],
            range: 8.0,
            color: [1.0, 0.92, 0.82],
            intensity: 6.0,
            spot: Some(SpotCone {
                direction: [-0.1, -1.0, -0.1],
                inner: 0.35,
                outer: 0.65,
            }),
            follows_lamps: false,
            shadow: 1.0,
        })
        .expect("spot");
        (r, caster)
    };

    let (mut cached, cached_caster) = setup(&mut device, Output::Surface(Format::Rgba8Unorm));
    let (mut whole, whole_caster) = setup(&mut device, Output::Surface(Format::Rgba8Unorm));
    let (mut cached_srgb, srgb_caster) = setup(&mut device, Output::Surface(Format::Rgba8Srgb));
    let (mut still, still_caster) = setup(&mut device, Output::Still);
    let cached_out = canvas(&mut device, size);
    let whole_out = canvas(&mut device, size);
    let srgb_tex = device
        .create_texture(&TextureDesc::new(
            size.0,
            size.1,
            Format::Rgba8Srgb,
            TextureUsage::COLOR_TARGET | TextureUsage::COPY_SRC,
        ))
        .expect("srgb target");
    let srgb_rb = device
        .create_buffer(&BufferDesc {
            size: u64::from(size.0 * size.1 * 4),
            usage: BufferUsage::COPY_DST,
            memory: Memory::Readback,
        })
        .expect("srgb readback");

    // Settle both props into the cached picture first across all presented renderers.
    for _ in 0..10 {
        for (r, tex) in [
            (&mut cached, cached_out.0),
            (&mut whole, whole_out.0),
            (&mut cached_srgb, srgb_tex),
        ] {
            let mut enc = device.encoder();
            r.encode(&mut device, &mut enc, &view, &env, [0.0; 2], 0, Some(tex))
                .expect("settle encode");
            device.submit(enc).expect("settle submit");
        }
    }
    assert!(cached.last_frame_stats().world_cached);

    // Move the shadow-casting sphere across the settled slab and floor: every reshaded frame
    // must match a whole-drawn surface frame byte-for-byte, and match Output::Still within the
    // single sRGB quantization step of Still's f16 readback.
    for x in [0.22, -0.32, 0.05] {
        let m = place(x, 1.15, 0.05, 0.24);
        for (r, c) in [
            (&mut cached, cached_caster),
            (&mut whole, whole_caster),
            (&mut cached_srgb, srgb_caster),
            (&mut still, still_caster),
        ] {
            r.set_prop_instance(c, &m, true).expect("move caster");
        }
        let a = present(&mut device, &mut cached, &view, &env, cached_out);
        let stats = cached.last_frame_stats();
        assert!(
            stats.world_cached && stats.reshaded && stats.reshade_pixels > 0,
            "expected reshade at x={x}: {stats:?}"
        );
        whole.world_cache = None;
        whole.prop_shadows_for = None;
        let b = present(&mut device, &mut whole, &view, &env, whole_out);
        assert_same_frame(&device, &a, &b, &format!("reshade vs whole at x={x}"));

        let srgb_live = frame_srgb(
            &mut device,
            &mut cached_srgb,
            &view,
            &env,
            Some((srgb_tex, srgb_rb)),
        );
        assert!(cached_srgb.last_frame_stats().reshaded);
        let srgb_still = frame_srgb(&mut device, &mut still, &view, &env, None);
        let max_step = srgb_live
            .chunks_exact(4)
            .zip(srgb_still.chunks_exact(4))
            .flat_map(|(p, q)| (0..3).map(move |ch| p[ch].abs_diff(q[ch])))
            .max()
            .unwrap_or(0);
        assert!(
            max_step <= 1,
            "reshaded frame vs Output::Still differed by {max_step} at x={x}"
        );
    }

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut cached, &mut whole, &mut cached_srgb, &mut still] {
        r.destroy_props(&mut device);
    }
}

pub(super) fn box_world_mesh(lo: V3, hi: V3, material: u32) -> Mesh {
    let [x0, y0, z0] = lo;
    let [x1, y1, z1] = hi;
    let faces: [([V3; 4], V3); 6] = [
        (
            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
            [0.0, 0.0, 1.0],
        ),
        (
            [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
            [0.0, 0.0, -1.0],
        ),
        (
            [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
            [1.0, 0.0, 0.0],
        ),
        (
            [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
            [-1.0, 0.0, 0.0],
        ),
        (
            [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
            [0.0, 1.0, 0.0],
        ),
        (
            [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
            [0.0, -1.0, 0.0],
        ),
    ];
    let mut positions = Vec::with_capacity(24);
    let mut normals = Vec::with_capacity(24);
    let mut uvs = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (pts, normal) in faces {
        let base = positions.len() as u32;
        for (i, p) in pts.into_iter().enumerate() {
            positions.push(Vec3(p[0], p[1], p[2]));
            normals.push(Vec3(normal[0], normal[1], normal[2]));
            uvs.push([(i & 1) as f32, (i >> 1) as f32]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    Mesh {
        positions,
        normals,
        uvs,
        color: [1.0, 1.0, 1.0],
        indices,
        material: Some(material),
    }
}

/// Regression for Wave F Part 1: tall walls, pillars, and water extending above a shadowed
/// downward spotlight (`y = 2.5`, `c.w <= 1e-6`) must be clipped against `c.w >= 1e-6` and cone
/// bounds by `receiver_map_bounds` (and ignored above the spotlight in `fs_reshade_mask`) so
/// `reshade_pixels` stays tightly bounded to the lit floor instead of exhausting the pair budget
/// across 80+ tall draws, while matching whole-drawn frames byte for byte.
#[test]
fn spotlight_above_light_receivers_stay_culled_and_match_whole() {
    use crate::render::lights::{LightDesc, SpotCone};

    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    world.materials.push(Material {
        name: "stone".into(),
        base_color: [0.64, 0.6, 0.56, 1.0],
        roughness: 0.78,
        shading: Shading::Standard,
        ..Material::default()
    });
    world.materials.push(Material {
        name: "water".into(),
        base_color: [0.14, 0.3, 0.42, 1.0],
        roughness: 0.08,
        shading: Shading::Water,
        ..Material::default()
    });
    // Floor receiver under the spotlight cone.
    world
        .meshes
        .push(box_world_mesh([-6.0, -0.1, -6.0], [6.0, 0.0, 6.0], 0));
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    // Water draw spanning behind/above the spotlight height along a side basin so `r.water` is
    // also exercised against `receiver_map_bounds`.
    world
        .meshes
        .push(box_world_mesh([-5.5, -0.2, -5.0], [-3.2, 3.2, 3.0], 1));
    world.instances.push(Instance {
        mesh: 1,
        transform: Transform(math::IDENTITY),
    });
    // 80 tall wall/pillar draws outside the spotlight cone extending from `y = 0` to `y = 6.0`
    // (above the spotlight at `y = 2.5`), spanning most of the viewport background!
    for i in 0..80 {
        let row = i / 20;
        let col = i % 20;
        let x = -4.8 + col as f32 * 0.5;
        let z = -3.8 - row as f32 * 0.55;
        let mesh_idx = world.meshes.len() as u32;
        world.meshes.push(box_world_mesh(
            [x - 0.18, 0.0, z - 0.18],
            [x + 0.18, 6.0, z + 0.18],
            0,
        ));
        world.instances.push(Instance {
            mesh: mesh_idx,
            transform: Transform(math::IDENTITY),
        });
    }

    let look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let size = (120, 90);
    let full_pixels = size.0 * size.1;
    let view = View {
        name: "spot_cull".into(),
        eye: [0.0, 2.2, 4.2],
        target: [0.0, 1.4, -1.5],
        fov: 55.0,
        ..View::default()
    };
    let mut env = Environment::resolve(
        &TimeOfDay {
            hours: 21.5,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &look.palette,
        1.0,
    );
    env.key_intensity = 0.0;
    env.lamp_level = 1.0;

    let (sp, sn, suv, sidx) = sphere_prop_data();
    let place = |pos: V3, scale_v: V3| {
        let mut m = math::IDENTITY;
        (m[0], m[5], m[10]) = (scale_v[0], scale_v[1], scale_v[2]);
        m[12..15].copy_from_slice(&pos);
        m
    };
    let setup = |device: &mut Device<'_>| {
        let mut r = Renderer::new(
            device,
            &world,
            look.clone(),
            Quality::realtime(),
            size,
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
        let mat = r
            .create_prop_material(
                device,
                &MaterialDesc {
                    name: "prop".into(),
                    base_color: [0.78, 0.62, 0.5, 1.0],
                    casts_shadow: true,
                    ..MaterialDesc::default()
                },
            )
            .expect("mat");
        // Settled tall column prop outside the cone reaching `y = 5.6` (above spotlight `y = 2.5`).
        r.create_prop_instance(sphere, mat, &place([1.9, 2.8, -0.4], [0.35, 2.8, 0.35]))
            .expect("tall settled prop");
        let caster = r
            .create_prop_instance(sphere, mat, &place([-0.14, 1.05, 0.2], [0.18; 3]))
            .expect("caster");
        r.create_light(&LightDesc {
            position: [0.0, 2.5, 0.2],
            range: 5.5,
            color: [1.0, 0.9, 0.78],
            intensity: 8.0,
            spot: Some(SpotCone {
                direction: [0.0, -1.0, 0.0],
                inner: 0.22,
                outer: 0.42,
            }),
            follows_lamps: false,
            shadow: 1.0,
        })
        .expect("spot");
        (r, caster, canvas(device, size))
    };

    let (mut cached, c_caster, c_out) = setup(&mut device);
    let (mut whole, w_caster, w_out) = setup(&mut device);

    for _ in 0..10 {
        let _ = present(&mut device, &mut cached, &view, &env, c_out);
        let _ = present(&mut device, &mut whole, &view, &env, w_out);
    }
    assert!(cached.last_frame_stats().world_cached);

    for x in [0.14, -0.1, 0.02] {
        let xf = place([x, 1.05, 0.2], [0.18; 3]);
        cached
            .set_prop_instance(c_caster, &xf, true)
            .expect("move cached");
        whole
            .set_prop_instance(w_caster, &xf, true)
            .expect("move whole");
        let a = present(&mut device, &mut cached, &view, &env, c_out);
        let stats = cached.last_frame_stats();
        assert!(
            stats.world_cached
                && stats.reshaded
                && stats.reshade_pixels > 0
                && stats.reshade_pixels < full_pixels / 3,
            "expected tight spotlight reshade (< {}), got {stats:?}",
            full_pixels / 3
        );
        assert_eq!(
            cached.water_statics.as_ref().map(Vec::len),
            Some(0),
            "water outside spotlight cone must accumulate 0 moved boxes"
        );
        whole.world_cache = None;
        whole.prop_shadows_for = None;
        let b = present(&mut device, &mut whole, &view, &env, w_out);
        assert_same_frame(&device, &a, &b, &format!("spotlight reshade at x={x}"));
    }

    for (tex, buf) in [c_out, w_out] {
        device.destroy_texture(tex);
        device.destroy_buffer(buf);
    }
    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut cached, &mut whole] {
        r.destroy_props(&mut device);
    }
}

/// Regression for Wave F Part 1: when a low-angle key light casts a foreground prop's shadow
/// toward and behind the camera (`min eye_clip.w < 0`) across a large floor AABB (`z = -20..20`),
/// `volume_screen_rect` clips the shadow prism against the 6 camera frustum planes so
/// `reshade_pixels` stays `< full_pixels / 2` instead of blowing up to the whole screen, while
/// matching whole-drawn rendering byte for byte.
#[test]
fn behind_camera_low_angle_shadow_prism_stays_tight_and_matches_whole() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let floor_bounds = ([-12.0, -0.2, -20.0], [12.0, 0.0, 20.0]);
    let mut world = World::default();
    world.materials.push(Material {
        name: "floor".into(),
        base_color: [0.66, 0.63, 0.58, 1.0],
        roughness: 0.82,
        shading: Shading::Standard,
        ..Material::default()
    });
    world
        .meshes
        .push(box_world_mesh(floor_bounds.0, floor_bounds.1, 0));
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });

    let look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let size = (120, 90);
    let full_pixels = size.0 * size.1;
    let view = View {
        name: "behind_cam".into(),
        eye: [0.0, 1.6, 3.0],
        target: [0.0, 0.9, -3.0],
        fov: 55.0,
        ..View::default()
    };
    let mut env = Environment::resolve(
        &TimeOfDay {
            hours: 20.5,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &look.palette,
        1.0,
    );
    // Low key light ahead (`-Z`), casting foreground shadows along `+Z` from `z = -1.2` across the
    // visible floor and past `eye.z = 3.0` (`min_w < 0`)!
    env.key_dir = normalize([0.08, 0.08, -1.0]);
    env.key_intensity = 1.4;

    let (sp, sn, suv, sidx) = sphere_prop_data();
    let place = |x: f32, y: f32, z: f32, r_s: f32| {
        let mut m = math::IDENTITY;
        (m[0], m[5], m[10]) = (r_s, r_s, r_s);
        m[12..15].copy_from_slice(&[x, y, z]);
        m
    };
    let setup = |device: &mut Device<'_>| {
        let mut r = Renderer::new(
            device,
            &world,
            look.clone(),
            Quality::realtime(),
            size,
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
        let mat = r
            .create_prop_material(
                device,
                &MaterialDesc {
                    name: "caster".into(),
                    base_color: [0.8, 0.55, 0.45, 1.0],
                    casts_shadow: true,
                    ..MaterialDesc::default()
                },
            )
            .expect("mat");
        let caster = r
            .create_prop_instance(sphere, mat, &place(-0.2, 0.22, -1.2, 0.22))
            .expect("caster");
        (r, caster, canvas(device, size))
    };

    let (mut cached, c_caster, c_out) = setup(&mut device);
    let (mut whole, w_caster, w_out) = setup(&mut device);

    for _ in 0..10 {
        let _ = present(&mut device, &mut cached, &view, &env, c_out);
        let _ = present(&mut device, &mut whole, &view, &env, w_out);
    }
    assert!(cached.last_frame_stats().world_cached);

    // Confirm `shadow_volume` on the floor genuinely produces vertices behind the camera (`w < 0`).
    let eye = cached.eye(&view, [0.0; 2], size);
    let cas = cached.cascades(&view, &env);
    let maps = cached.prop_shadow_maps(&cas);
    let moved_box = ([-0.45, 0.0, -1.45], [0.45, 0.44, -0.95]);
    let fps = cached.shadow_footprints(&[moved_box], &cas, true);
    let mut pts = Vec::new();
    let mut min_w = f32::INFINITY;
    for f in &fps {
        shadow_volume(floor_bounds, f, &maps, &mut pts);
        for p in &pts {
            let c = transform(&eye.view_proj, [p[0], p[1], p[2], 1.0]);
            min_w = min_w.min(c[3]);
        }
    }
    assert!(
        min_w < 0.0,
        "expected shadow_volume to extend behind camera (`w < 0`), got min_w = {min_w}"
    );

    for x in [0.18, -0.22, 0.05] {
        let xf = place(x, 0.22, -1.2, 0.22);
        cached
            .set_prop_instance(c_caster, &xf, true)
            .expect("move cached");
        whole
            .set_prop_instance(w_caster, &xf, true)
            .expect("move whole");
        let a = present(&mut device, &mut cached, &view, &env, c_out);
        let stats = cached.last_frame_stats();
        assert!(
            stats.world_cached
                && stats.reshaded
                && stats.reshade_pixels > 0
                && stats.reshade_pixels < full_pixels / 2,
            "expected frustum-clipped prism (< {}), got {stats:?}",
            full_pixels / 2
        );
        whole.world_cache = None;
        whole.prop_shadows_for = None;
        let b = present(&mut device, &mut whole, &view, &env, w_out);
        assert_same_frame(&device, &a, &b, &format!("behind-camera prism at x={x}"));
    }

    for (tex, buf) in [c_out, w_out] {
        device.destroy_texture(tex);
        device.destroy_buffer(buf);
    }
    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut cached, &mut whole] {
        r.destroy_props(&mut device);
    }
}

/// Regression for Wave F Part 3: specialized `TONEMAP_MODE` pipelines (`Stylized`, `AgX`,
/// `ACESFitted`) x `(identity grade, non-identity grade)` x `(vignette = 0.0, vignette = 0.25)`
/// match `set_grade_once(true)` byte for byte (`0` differing bytes).
#[test]
fn specialized_tonemap_modes_and_vignette_guard_match_grade_once() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    world.materials.push(Material {
        name: "terracotta".into(),
        base_color: [0.85, 0.42, 0.28, 1.0],
        roughness: 0.45,
        ..Material::default()
    });
    world.materials.push(Material {
        name: "lantern".into(),
        base_color: [1.0, 0.9, 0.7, 1.0],
        emissive: [6.0, 3.8, 1.4],
        roughness: 0.3,
        ..Material::default()
    });
    world
        .meshes
        .push(box_world_mesh([-4.0, -0.1, -4.0], [4.0, 0.0, 4.0], 0));
    world
        .meshes
        .push(box_world_mesh([-0.5, 0.0, -0.8], [0.5, 1.1, 0.2], 1));
    for mesh in 0..2 {
        world.instances.push(Instance {
            mesh,
            transform: Transform(math::IDENTITY),
        });
    }

    let size = (64, 48);
    let base_look = Look {
        painterly: 0.0,
        sharpen: 0.35,
        ..Look::default()
    };
    let mut fused = Renderer::new(
        &mut device,
        &world,
        base_look.clone(),
        Quality::realtime(),
        size,
        Output::Surface(Format::Rgba8Unorm),
    )
    .expect("fused");
    let mut once = Renderer::new(
        &mut device,
        &world,
        base_look.clone(),
        Quality::realtime(),
        size,
        Output::Surface(Format::Rgba8Unorm),
    )
    .expect("once");
    fused.finish_ao(&mut device).expect("ao");
    once.finish_ao(&mut device).expect("ao");
    assert!(once.set_grade_once(&mut device, true).expect("grade_once"));

    let f_out = canvas(&mut device, size);
    let o_out = canvas(&mut device, size);
    let view = View {
        name: "tm".into(),
        eye: [0.9, 1.5, 2.5],
        target: [0.0, 0.5, -0.2],
        fov: 50.0,
        ..View::default()
    };

    for tonemap in ToneMap::ALL {
        for identity in [true, false] {
            for vignette in [0.0, 0.25] {
                let mut look = base_look.clone();
                look.tonemap = tonemap;
                look.vignette = vignette;
                if identity {
                    look.saturation = 1.0;
                    look.contrast = 1.0;
                    look.lift = [0.0; 3];
                    look.gain = [1.0; 3];
                    assert!(look.grade_is_identity());
                } else {
                    look.saturation = 1.18;
                    look.contrast = 1.12;
                    look.lift = [0.02, 0.01, 0.03];
                    look.gain = [1.04, 0.98, 0.95];
                    assert!(!look.grade_is_identity());
                }
                let env = Environment::resolve(
                    &TimeOfDay {
                        hours: 17.4,
                        ..TimeOfDay::default()
                    },
                    &Weather::default(),
                    &look.palette,
                    1.0,
                );
                fused.set_look(look.clone());
                once.set_look(look);
                let a = present(&mut device, &mut fused, &view, &env, f_out);
                let b = present(&mut device, &mut once, &view, &env, o_out);
                assert!(!fused.last_frame_stats().graded_once);
                assert!(once.last_frame_stats().graded_once);
                assert_eq!(
                    a, b,
                    "tonemap={tonemap:?} identity={identity} vignette={vignette}"
                );
            }
        }
    }

    for (tex, buf) in [f_out, o_out] {
        device.destroy_texture(tex);
        device.destroy_buffer(buf);
    }
    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
}

/// Regression for `[MAJOR-1]`: a foreground opaque draw grazing `2e-6 m` outside a `[20, 12, 20]`
/// downward spotlight cone in front of a brightly lit background draw inside the cone must remain
/// in `receiver_map_bounds` (via `eps` and `spot_margin`) so `fs_main_opaque_cache` redraws the
/// foreground occluder instead of letting the background draw overwrite `cache_hdr` through it.
#[test]
fn spotlight_cone_graze_foreground_occluder_does_not_punch_hole() {
    use crate::render::lights::{LightDesc, SpotCone};
    use crate::render::renderer::cache::{OUT_OF_REACH, receiver_map_bounds};

    let Ok(mut device) = Device::headless() else {
        return;
    };
    let outer = 0.45_f32;
    let x_cone_at_y10 = 20.0 + 2.0 * outer.tan();
    let x_graze = x_cone_at_y10 + 2.0e-6;
    let back_bounds = ([21.22, 8.2, 18.8], [21.38, 10.2, 21.2]);
    let front_bounds = ([x_graze, 10.0, 19.3], [x_graze + 0.05, 10.015, 20.7]);

    let mut world = World::default();
    world.materials.push(Material {
        name: "bright_floor".into(),
        base_color: [0.92, 0.88, 0.8, 1.0],
        roughness: 0.65,
        shading: Shading::Standard,
        ..Material::default()
    });
    world.materials.push(Material {
        name: "dark_occluder".into(),
        base_color: [0.02, 0.02, 0.03, 1.0],
        roughness: 0.95,
        shading: Shading::Standard,
        ..Material::default()
    });
    world
        .meshes
        .push(box_world_mesh(back_bounds.0, back_bounds.1, 0));
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    world
        .meshes
        .push(box_world_mesh(front_bounds.0, front_bounds.1, 1));
    world.instances.push(Instance {
        mesh: 1,
        transform: Transform(math::IDENTITY),
    });
    // Off-screen world bounds posts so `cascades()` depth range stays invariant as `caster` moves.
    for (lo, hi) in [
        ([17.0, 7.5, 17.0], [17.2, 12.5, 17.2]),
        ([22.8, 7.5, 22.8], [23.0, 12.5, 23.0]),
    ] {
        let idx = world.meshes.len() as u32;
        world.meshes.push(box_world_mesh(lo, hi, 1));
        world.instances.push(Instance {
            mesh: idx,
            transform: Transform(math::IDENTITY),
        });
    }

    let look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let size = (120, 90);
    let view = View {
        name: "spot_graze".into(),
        eye: [20.6, 10.85, 20.0],
        target: [21.25, 9.35, 20.0],
        fov: 38.0,
        ..View::default()
    };
    let mut env = Environment::resolve(
        &TimeOfDay {
            hours: 21.5,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &look.palette,
        1.0,
    );
    env.key_intensity = 0.0;
    env.lamp_level = 1.0;

    let (sp, sn, suv, sidx) = sphere_prop_data();
    let place = |pos: V3, scale_v: V3| {
        let mut m = math::IDENTITY;
        (m[0], m[5], m[10]) = (scale_v[0], scale_v[1], scale_v[2]);
        m[12..15].copy_from_slice(&pos);
        m
    };
    let setup = |device: &mut Device<'_>| {
        let mut r = Renderer::new(
            device,
            &world,
            look.clone(),
            Quality::realtime(),
            size,
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
        let mat = r
            .create_prop_material(
                device,
                &MaterialDesc {
                    name: "caster".into(),
                    base_color: [0.6, 0.55, 0.5, 1.0],
                    casts_shadow: true,
                    ..MaterialDesc::default()
                },
            )
            .expect("mat");
        let caster = r
            .create_prop_instance(
                sphere,
                mat,
                &place([20.0 + 0.92 * outer.tan(), 11.0, 20.0], [0.16; 3]),
            )
            .expect("caster");
        r.create_light(&LightDesc {
            position: [20.0, 12.0, 20.0],
            range: 6.0,
            color: [1.0, 0.95, 0.85],
            intensity: 24.0,
            spot: Some(SpotCone {
                direction: [0.0, -1.0, 0.0],
                inner: 0.2,
                outer,
            }),
            follows_lamps: false,
            shadow: 1.0,
        })
        .expect("spot");
        (r, caster, canvas(device, size))
    };

    let (mut cached, c_caster, c_out) = setup(&mut device);
    let (mut whole, w_caster, w_out) = setup(&mut device);

    let spot_m = cached.game_lights.shadow_views()[0];
    let light_texel = 2.0 / cached.quality.light_shadow_size.max(16) as f32;
    let spot_margin = 6.0 * light_texel;
    assert_ne!(
        receiver_map_bounds(front_bounds, &spot_m, Some(spot_margin)),
        OUT_OF_REACH,
        "2e-6 m cone-grazing foreground occluder must stay in receiver_map_bounds"
    );

    for _ in 0..10 {
        let _ = present(&mut device, &mut cached, &view, &env, c_out);
        let _ = present(&mut device, &mut whole, &view, &env, w_out);
    }
    assert!(cached.last_frame_stats().world_cached);

    for (dx, z) in [(0.04, 19.88), (-0.03, 20.12), (0.05, 20.0)] {
        let xf = place([20.0 + (0.92 + dx) * outer.tan(), 11.0, z], [0.16; 3]);
        cached
            .set_prop_instance(c_caster, &xf, true)
            .expect("move cached");
        whole
            .set_prop_instance(w_caster, &xf, true)
            .expect("move whole");
        let a = present(&mut device, &mut cached, &view, &env, c_out);
        let stats = cached.last_frame_stats();
        assert!(
            stats.world_cached && stats.reshaded && stats.reshade_pixels > 0,
            "expected cached spotlight reshade, got {stats:?}"
        );
        whole.world_cache = None;
        whole.prop_shadows_for = None;
        whole.forget_ao();
        let b = present(&mut device, &mut whole, &view, &env, w_out);
        assert_same_frame(&device, &a, &b, &format!("cone-graze dx={dx} z={z}"));
    }

    for (tex, buf) in [c_out, w_out] {
        device.destroy_texture(tex);
        device.destroy_buffer(buf);
    }
    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut cached, &mut whole] {
        r.destroy_props(&mut device);
    }
}
