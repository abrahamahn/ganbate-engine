use super::*;

#[test]
fn air_sky_and_flower_shaders_compile_on_all_backends() {
    let opts = CompileOptions::default();
    for (src, stage, entry) in [
        (POST, Stage::Fragment, "fs_air"),
        (WORLD_FX, Stage::Fragment, "fs_sky"),
        (SCENE, Stage::Vertex, "vs_flower"),
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
