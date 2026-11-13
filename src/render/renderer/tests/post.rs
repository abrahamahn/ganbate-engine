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
