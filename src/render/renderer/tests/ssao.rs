use super::*;
use crate::render::look::DetailRule;
use crate::render::props::PropInstance;

/// Makes `r`'s AO target a copy source (and the frame slots that bind it anew), and a buffer to
/// read it into.
fn readable_ao(device: &mut Device<'_>, r: &mut Renderer) -> BufferId {
    let (w, h) = r.targets.ao;
    let usage = TextureUsage::COLOR_TARGET | TextureUsage::SAMPLED | TextureUsage::COPY_SRC;
    let ao = device
        .create_texture(&TextureDesc::new(w, h, Format::R16Float, usage))
        .expect("ao");
    device.destroy_texture(std::mem::replace(&mut r.targets.ao_tex, ao));
    r.reset_prop_slots(device);
    Renderer::destroy_slots(device, std::mem::take(&mut r.slots));
    let count = match r.output {
        Output::Still => 1,
        Output::Surface(_) => 2,
    };
    for _ in 0..count {
        let slot = r.create_slot(device).expect("slot");
        r.slots.push(slot);
    }
    device
        .create_buffer(&BufferDesc {
            size: u64::from(w * h * 2),
            usage: BufferUsage::COPY_DST,
            memory: Memory::Readback,
        })
        .expect("ao readback")
}

/// [`present`], reading back the AO too (`readable_ao`), with the kernel's reach in the AO's
/// changed regions block set to `reach` when given (after `encode` wrote it).
fn present_with_ao(
    device: &mut Device<'_>,
    r: &mut Renderer,
    (view, env): (&View, &Environment),
    (target, readback): (TextureId, BufferId),
    ao: BufferId,
    reach: Option<f32>,
) -> (Vec<u8>, Vec<u8>) {
    let slot = device.frame_slot() % r.slots.len();
    let mut enc = device.encoder();
    r.encode(device, &mut enc, view, env, [0.0; 2], 0, Some(target))
        .expect("frame");
    if let Some(reach) = reach {
        let block = r.slots[slot].ao_changed;
        device
            .write_buffer(block, 4, &reach.to_le_bytes())
            .expect("reach");
    }
    enc.copy_texture_to_buffer(target, 0, 0, readback, 0);
    enc.copy_texture_to_buffer(r.targets.ao_tex, 0, 0, ao, 0);
    device.submit(enc).expect("submit");
    let frame = device.read_buffer(readback).expect("read back");
    (frame, device.read_buffer(ao).expect("read back the AO"))
}

fn assert_same_ao(device: &Device<'_>, kept: &[u8], whole: &[u8], what: &str) {
    assert_eq!(kept.len(), whole.len(), "{what}: AO byte length");
    let llvmpipe = device.name().starts_with("llvmpipe");
    let texels = kept.len() / 2;
    let differ = kept
        .chunks_exact(2)
        .zip(whole.chunks_exact(2))
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differ == 0 || !llvmpipe && differ * 1000 <= texels,
        "{what}: {differ} of {texels} AO texels differ"
    );
}

/// The AO texels a frame of `r` computes, of all: its AO target cleared to a mark first (the next
/// frame computes it whole).
fn recomputed_texels(
    device: &mut Device<'_>,
    r: &mut Renderer,
    frame: (&View, &Environment),
    out: (TextureId, BufferId),
    ao: BufferId,
) -> (usize, usize) {
    let mut enc = device.encoder();
    enc.begin_pass(PassDesc {
        colors: vec![Attachment::clear(r.targets.ao_tex, [-1.0; 4])],
        depth: None,
    });
    enc.end_pass();
    device.submit(enc).expect("mark");
    let (_, texels) = present_with_ao(device, r, frame, out, ao, None);
    r.forget_ao();
    let value = |t: &[u8]| f16_to_f32(u16::from_le_bytes([t[0], t[1]]));
    let computed = texels.chunks_exact(2).filter(|t| value(t) != -1.0).count();
    (computed, texels.len() / 2)
}

#[test]
fn the_ao_mask_and_ssao_shaders_compile_on_all_backends() {
    let composed = ganbate_shader::compose(POST, &shader_modules()).expect("compose");
    let opts = CompileOptions::default();
    for entry in [
        "fs_ao_grass_half",
        "fs_ao_grass_tile",
        "fs_ao_grass_union",
        "fs_ao_grass_coarse",
        "fs_ssao",
    ] {
        let spv =
            compile(POST, Stage::Fragment, entry).unwrap_or_else(|e| panic!("{entry} spirv: {e}"));
        if entry != "fs_ssao" {
            assert!(
                spv.bindings.iter().all(|b| b.binding != 1),
                "{entry} must not bind depth_tex (binding 1)"
            );
        }
        let glsl = compile_glsl_with_options(&composed, Stage::Fragment, entry, &opts)
            .unwrap_or_else(|e| panic!("{entry} glsl: {e}"));
        let g = glsl.glsl.expect("glsl");
        assert!(
            g.units.len() <= 16,
            "{entry} uses {} WebGL texture units",
            g.units.len()
        );
        compile_msl_with_options(&composed, Stage::Fragment, entry, &opts)
            .unwrap_or_else(|e| panic!("{entry} msl: {e}"));
        compile_wgsl_with_options(&composed, Stage::Fragment, entry, &opts)
            .unwrap_or_else(|e| panic!("{entry} wgsl: {e}"));
    }
}

/// `AO_KERNEL_REACH` holds every kernel point of `fs_ssao` (lifted off its surface as it lifts
/// them), in radii, and not by much more; `AoChanged` holds `MAX_AO_REGIONS`.
#[test]
fn the_ao_kernel_lies_within_its_reach() {
    let start = POST.find("const KERNEL").expect("kernel");
    let body = &POST[start..start + POST[start..].find(");").expect("kernel end")];
    let reach = body
        .split("vec3f(")
        .skip(2)
        .map(|point| {
            let k: Vec<f32> = point
                .split([',', ')'])
                .take(3)
                .map(|v| v.trim().parse().expect("number"))
                .collect();
            math::length([k[0], k[1], k[2] + 0.08])
        })
        .fold(0.0, f32::max);
    assert!(
        reach <= AO_KERNEL_REACH && reach > AO_KERNEL_REACH - 0.01,
        "{reach}"
    );
    assert!(POST.contains(&format!("rects: array<vec4f, {MAX_AO_REGIONS}>")));
}

/// Every point of a box lies in the pixels `box_rect` gives it, where the box lies in front of the
/// eye, across the plane of the eye, beside it and above it; a narrow box crossing behind the eye
/// on one side of the screen stays bounded to that side rather than covering the whole viewport.
#[test]
fn a_box_lies_in_its_rect() {
    let size = (160, 120);
    let view_proj = mul(
        &perspective(50f32.to_radians(), 160.0 / 120.0, 0.1),
        &look_at([0.0, 1.5, 3.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]),
    );
    let boxes = [
        ([-0.5, 0.0, -0.5], [0.5, 0.3, 0.5]),
        ([-2.0, 0.0, -1.0], [2.0, 0.4, 6.0]),
        ([1.5, 0.0, 2.0], [4.0, 0.4, 8.0]),
        ([-3.0, 1.4, 2.5], [3.0, 1.6, 3.5]),
    ];
    for b in boxes {
        let r = box_rect(b, &view_proj, size, 0.0);
        for i in 0..=8 {
            for j in 0..=8 {
                for k in 0..=8 {
                    let f = [i, j, k].map(|n| n as f32 / 8.0);
                    let p: V3 = std::array::from_fn(|a| b.0[a] + (b.1[a] - b.0[a]) * f[a]);
                    let c = transform(&view_proj, [p[0], p[1], p[2], 1.0]);
                    if c[3] <= 0.1 {
                        continue;
                    }
                    let x = (c[0] / c[3] * 0.5 + 0.5) * size.0 as f32;
                    let y = (c[1] / c[3] * 0.5 + 0.5) * size.1 as f32;
                    if (0.0..size.0 as f32).contains(&x) && (0.0..size.1 as f32).contains(&y) {
                        let r = r.unwrap_or_else(|| panic!("{b:?}: no rect for {p:?}"));
                        let inside = r[0] as f32 <= x
                            && x <= r[2] as f32
                            && r[1] as f32 <= y
                            && y <= r[3] as f32;
                        assert!(inside, "{b:?}: {p:?} at ({x}, {y}) outside {r:?}");
                    }
                }
            }
        }
    }
    let side_rail =
        box_rect(([0.3, 0.0, 1.5], [1.5, 0.4, 5.0]), &view_proj, size, 2.0).expect("side rail");
    assert!(
        side_rail[0] > size.0 / 2,
        "clipped rail stays on right half: {side_rail:?}"
    );
}

/// Joined regions hold every pixel of the regions joined, however many there are, and are at most
/// [`MAX_AO_REGIONS`]; `union_area` counts the pixels they hold.
#[test]
fn joined_regions_hold_every_pixel() {
    let size = (96, 64);
    let mut seed = 7u32;
    let mut next = |n: u32| {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) % n
    };
    for count in [3, 20, 90, 400] {
        let rects: Vec<Rect> = (0..count)
            .map(|_| {
                let (x, y) = (next(size.0), next(size.1));
                [
                    x,
                    y,
                    (x + 1 + next(12)).min(size.0),
                    (y + 1 + next(9)).min(size.1),
                ]
            })
            .collect();
        let joined = join_regions(rects.clone(), size);
        assert!(joined.len() <= MAX_AO_REGIONS, "{count}: {}", joined.len());
        let holds = |set: &[Rect], x, y| {
            set.iter()
                .any(|r| r[0] <= x && x < r[2] && r[1] <= y && y < r[3])
        };
        let mut held = 0;
        for y in 0..size.1 {
            for x in 0..size.0 {
                assert!(
                    !holds(&rects, x, y) || holds(&joined, x, y),
                    "{count}: ({x}, {y})"
                );
                held += u32::from(holds(&joined, x, y));
            }
        }
        assert_eq!(union_area(&joined), held, "{count}");
    }
}

/// A garden for the AO kept from frame to frame: a floor, a lawn growing grass that sways, a pond,
/// a slab settled on the floor, a ball by it and one standing in the pond; a camera that holds
/// still. Drawn by `r` into the canvas `out`, its AO read into `ao`.
struct AoGarden {
    r: Renderer,
    ball: PropInstance,
    view: View,
    look: Look,
    out: (TextureId, BufferId),
    ao: BufferId,
}

impl AoGarden {
    fn placed(at: V3, size: V3) -> M4 {
        let mut m = math::IDENTITY;
        (m[0], m[5], m[10]) = (size[0], size[1], size[2]);
        m[12..15].copy_from_slice(&at);
        m
    }

    fn ball_at(x: f32) -> M4 {
        Self::placed([x, 0.18, 1.15], [0.18; 3])
    }

    fn env(&self, seconds: f32) -> Environment {
        let time = TimeOfDay {
            hours: 15.0,
            ..TimeOfDay::default()
        };
        Environment::resolve(&time, &Weather::default(), &self.look.palette, seconds)
    }

    fn frame(
        &mut self,
        device: &mut Device<'_>,
        x: f32,
        env: &Environment,
        reach: Option<f32>,
    ) -> (Vec<u8>, Vec<u8>) {
        self.r
            .set_prop_instance(self.ball, &Self::ball_at(x), true)
            .expect("ball");
        let frame = (&self.view, env);
        present_with_ao(device, &mut self.r, frame, self.out, self.ao, reach)
    }
}

fn ao_garden(device: &mut Device<'_>) -> AoGarden {
    let mut world = World::default();
    for (name, shading, base_color) in [
        ("floor", Shading::Standard, [0.6, 0.58, 0.55, 1.0]),
        ("lawn", Shading::Standard, [0.3, 0.5, 0.2, 1.0]),
        ("pond", Shading::Water, [0.12, 0.34, 0.42, 0.85]),
    ] {
        world.materials.push(Material {
            name: name.into(),
            base_color,
            shading,
            ..Material::default()
        });
    }
    for (material, y, half, x, z) in [
        (0, 0.0, 3.0, 0.0, 0.0),
        (1, 0.01, 0.6, -1.6, -1.2),
        (2, 0.02, 0.6, 1.4, -0.6),
    ] {
        world.meshes.push(Mesh {
            material: Some(material),
            ..make_quad_mesh(y, half)
        });
        let mut at = math::IDENTITY;
        (at[12], at[14]) = (x, z);
        world.instances.push(Instance {
            mesh: material,
            transform: Transform(at),
        });
    }
    let mut look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    look.details.push(DetailRule {
        patterns: vec!["lawn".into()],
        preset: DetailPreset::Grass,
        scale: [1.0, 1.0],
        strength: 1.0,
        variation: 0.0,
        color: None,
        gloss: 0.0,
    });
    look.grass.density = 1.0;
    look.grass.height = 0.3;
    look.grass.width = 0.04;
    look.grass.sway = 1.0;
    look.grass.flowers = 0.2;
    let size = (160, 120);
    let quality = Quality {
        reflection_scale: 0.0,
        ..Quality::realtime()
    };
    let mut r = Renderer::new(
        device,
        &world,
        look.clone(),
        quality,
        size,
        Output::Surface(Format::Rgba8Unorm),
    )
    .expect("renderer");
    r.finish_ao(device).expect("baked ao");
    let ao = readable_ao(device, &mut r);
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
    let material = r
        .create_prop_material(
            device,
            &MaterialDesc {
                name: "ball".into(),
                base_color: [0.8, 0.3, 0.2, 1.0],
                ..MaterialDesc::default()
            },
        )
        .expect("material");
    let placed = AoGarden::placed;
    r.create_prop_instance(
        sphere,
        material,
        &placed([0.0, 0.15, 0.5], [0.6, 0.15, 0.4]),
    )
    .expect("slab");
    r.create_prop_instance(sphere, material, &placed([1.4, 0.2, -0.6], [0.2; 3]))
        .expect("ball in the pond");
    let ball = r
        .create_prop_instance(sphere, material, &AoGarden::ball_at(-0.4))
        .expect("ball");
    let view = View {
        name: "garden".into(),
        eye: [0.0, 1.6, 3.0],
        target: [0.0, 0.1, 0.0],
        fov: 50.0,
        ..View::default()
    };
    let out = canvas(device, size);
    AoGarden {
        r,
        ball,
        view,
        look,
        out,
        ao,
    }
}

/// Over 22 frames, the AO kept with the cached picture and computed again only where what it reads
/// changed equals the AO computed whole for every texel on lavapipe: while a ball casting a shadow
/// moves by the slab, stops, settles into the picture and moves off again, the grass sways and the
/// pond ripples with the clock, and the camera holds.
#[test]
fn kept_ao_matches_the_ao_computed_whole() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut kept = ao_garden(&mut device);
    let mut whole = ao_garden(&mut device);
    let llvmpipe = device.name().starts_with("llvmpipe");
    let mut changed = 0;
    let mut last = vec![];
    for (frame, x) in (0..6)
        .map(|k| k as f32 * 0.015)
        .chain([0.075; 12])
        .chain((1..5).map(|k| 0.075 - k as f32 * 0.02))
        .enumerate()
    {
        let env = kept.env(frame as f32 / 60.0);
        let (a, a_ao) = kept.frame(&mut device, x - 0.4, &env, None);
        let stats = kept.r.last_frame_stats();
        whole.r.forget_world_cache();
        whole.r.forget_prop_shadows();
        let (b, b_ao) = whole.frame(&mut device, x - 0.4, &env, None);
        assert!(!whole.r.last_frame_stats().ao_kept);
        assert_eq!(stats.ao_kept, frame > 0, "frame {frame}");
        assert!(stats.ssao_drawn, "frame {frame}: the pond ripples");
        let what = format!("frame {frame}, ball at {x} m");
        assert_same_ao(&device, &a_ao, &b_ao, &what);
        let (off, largest) = off_by_more_than_a_step(&a, &b);
        assert!(
            largest <= 1 || !llvmpipe && off <= 0.001,
            "{what}: up to {largest} steps apart"
        );
        changed += usize::from(frame > 0 && b_ao != last);
        last = b_ao;
    }
    assert!(changed > 10, "{changed}");
    kept.r
        .set_prop_instance(kept.ball, &AoGarden::ball_at(-0.4), true)
        .expect("ball");
    let env = kept.env(1.0);
    let frame = (&kept.view, &env);
    let (computed, texels) = recomputed_texels(&mut device, &mut kept.r, frame, kept.out, kept.ao);
    assert!(
        computed > 0 && computed * 2 < texels,
        "{computed} of {texels}"
    );
    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for g in [&mut kept, &mut whole] {
        g.r.destroy_props(&mut device);
    }
}

/// The kept AO must be computed again wherever a kernel point can read what changed: drawn with
/// the kernel's reach taken as nothing, the AO around the moving ball goes stale.
#[test]
fn kept_ao_without_the_kernels_reach_goes_stale() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut kept = ao_garden(&mut device);
    let mut whole = ao_garden(&mut device);
    let mut stale = 0;
    for frame in 0..5 {
        let env = kept.env(frame as f32 / 60.0);
        let x = frame as f32 * 0.03 - 0.4;
        let (_, a) = kept.frame(&mut device, x, &env, Some(0.0));
        whole.r.forget_world_cache();
        whole.r.forget_prop_shadows();
        let (_, b) = whole.frame(&mut device, x, &env, None);
        stale += a
            .chunks_exact(2)
            .zip(b.chunks_exact(2))
            .filter(|(p, q)| p != q)
            .count();
    }
    assert!(stale > 0, "no AO texel went stale");
    for g in [&mut kept, &mut whole] {
        g.r.destroy_props(&mut device);
    }
}

fn plain_floor_pair(
    device: &mut Device<'_>,
    size: (u32, u32),
    quality: Quality,
    look: Look,
) -> [(Renderer, PropInstance, (TextureId, BufferId), BufferId); 2] {
    let mut world = World::default();
    world.materials.push(Material {
        name: "floor".into(),
        base_color: [0.62, 0.6, 0.56, 1.0],
        ..Material::default()
    });
    world.meshes.push(Mesh {
        material: Some(0),
        ..make_quad_mesh(0.0, 4.0)
    });
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    // Anchor land height up to y = 2.5 behind the camera so resizing or removing props below 2.5 m
    // keeps the key-light cascade depth bounds (`world_key`) fixed across kept frames.
    world.meshes.push(Mesh {
        material: Some(0),
        ..make_quad_mesh(2.5, 0.2)
    });
    let mut back = math::IDENTITY;
    back[14] = 8.0;
    world.instances.push(Instance {
        mesh: 1,
        transform: Transform(back),
    });
    let (sp, sn, suv, sidx) = sphere_prop_data();
    std::array::from_fn(|_| {
        let mut r = Renderer::new(
            device,
            &world,
            look.clone(),
            quality,
            size,
            Output::Surface(Format::Rgba8Unorm),
        )
        .expect("renderer");
        r.finish_ao(device).expect("baked ao");
        let ao = readable_ao(device, &mut r);
        let mesh = r
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
                    name: "ball".into(),
                    base_color: [0.75, 0.35, 0.25, 1.0],
                    ..MaterialDesc::default()
                },
            )
            .expect("mat");
        // Keep an off-view caster behind the eye so destroying `ball` does not flip
        // `has_prop_shadows` (`world_key`) from true to false.
        r.create_prop_instance(mesh, mat, &AoGarden::placed([0.0, 0.2, 6.0], [0.2; 3]))
            .expect("anchor");
        let ball = r
            .create_prop_instance(mesh, mat, &AoGarden::placed([-0.9, 0.2, 0.4], [0.2; 3]))
            .expect("ball");
        let out = canvas(device, size);
        (r, ball, out, ao)
    })
}

/// When a live prop jumps across the screen by far more than the AO kernel reach between two kept
/// frames so its old and new screen rects are completely disjoint, both regions recompute and match
/// whole-frame AO byte for byte.
#[test]
fn disjoint_live_prop_jump_updates_both_old_and_new_ao_regions() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let quality = Quality {
        reflection_scale: 0.0,
        ao_scale: 1.0,
        ..Quality::realtime()
    };
    let size = (160, 120);
    let [
        (mut kept, k_ball, k_out, k_ao),
        (mut whole, w_ball, w_out, w_ao),
    ] = plain_floor_pair(&mut device, size, quality, look.clone());
    let view = View {
        name: "floor".into(),
        eye: [0.0, 1.7, 2.8],
        target: [0.0, 0.1, 0.2],
        fov: 52.0,
        ..View::default()
    };
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 14.0,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &look.palette,
        0.0,
    );
    let eye = kept.eye(&view, [0.0; 2], size);
    let left_rect = box_rect(
        ([-1.1, 0.0, 0.2], [-0.7, 0.4, 0.6]),
        &eye.view_proj,
        size,
        4.0,
    )
    .expect("left rect");
    let right_rect = box_rect(
        ([0.7, 0.0, 0.2], [1.1, 0.4, 0.6]),
        &eye.view_proj,
        size,
        4.0,
    )
    .expect("right rect");
    assert!(
        left_rect[2] + 30 < right_rect[0],
        "old and new rects must be wide apart: {left_rect:?} vs {right_rect:?}"
    );

    for (step, x) in [-0.9, -0.88, 0.9, 0.88].into_iter().enumerate() {
        let m = AoGarden::placed([x, 0.2, 0.4], [0.2; 3]);
        kept.set_prop_instance(k_ball, &m, true).expect("kept ball");
        whole
            .set_prop_instance(w_ball, &m, true)
            .expect("whole ball");
        let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
        whole.forget_world_cache();
        whole.forget_prop_shadows();
        let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
        let stats = kept.last_frame_stats();
        assert_eq!(stats.ao_kept, step > 0, "step {step}");
        assert!(stats.ssao_drawn, "step {step}");
        if step > 0 {
            assert!(stats.ao_changed_pixels > 0, "step {step}");
        }
        let what = format!("disjoint jump step {step} (x = {x})");
        assert_same_ao(&device, &a_ao, &b_ao, &what);
        assert_same_frame(&device, &a, &b, &what);
    }

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut kept, &mut whole] {
        r.destroy_props(&mut device);
    }
}

/// Removing a live prop (`destroy_prop_instance` -> `vacated_live`) between kept frames recomputes
/// the vacated ground and recovers its unoccluded AO byte for byte.
#[test]
fn removed_live_prop_recovers_unoccluded_ao() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let quality = Quality {
        reflection_scale: 0.0,
        ao_scale: 1.0,
        ..Quality::realtime()
    };
    let size = (160, 120);
    let [
        (mut kept, k_ball, k_out, k_ao),
        (mut whole, w_ball, w_out, w_ao),
    ] = plain_floor_pair(&mut device, size, quality, look.clone());
    let view = View {
        name: "floor".into(),
        eye: [0.0, 1.5, 2.4],
        target: [0.0, 0.1, 0.2],
        fov: 50.0,
        ..View::default()
    };
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 14.0,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &look.palette,
        0.0,
    );

    // Frame 0 (whole) and Frame 1 (kept, slight move so the ball stays live and occludes floor).
    let mut with_ball_ao = Vec::new();
    for (step, x) in [0.0, 0.02].into_iter().enumerate() {
        let m = AoGarden::placed([x, 0.22, 0.3], [0.24; 3]);
        kept.set_prop_instance(k_ball, &m, true).expect("kept");
        whole.set_prop_instance(w_ball, &m, true).expect("whole");
        let (_, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
        whole.forget_world_cache();
        whole.forget_prop_shadows();
        let (_, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
        assert_eq!(kept.last_frame_stats().ao_kept, step > 0);
        assert_same_ao(&device, &a_ao, &b_ao, "with live ball");
        with_ball_ao = a_ao;
    }

    // Remove the live prop between kept frames.
    kept.destroy_prop_instance(k_ball);
    whole.destroy_prop_instance(w_ball);
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
    let stats = kept.last_frame_stats();
    assert!(stats.ao_kept && stats.ssao_drawn && stats.ao_changed_pixels > 0);
    whole.forget_world_cache();
    whole.forget_prop_shadows();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
    assert_ne!(
        a_ao, with_ball_ao,
        "vacated prop must change AO back to unoccluded"
    );
    assert_same_ao(&device, &a_ao, &b_ao, "after live prop removal");
    assert_same_frame(&device, &a, &b, "after live prop removal");

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut kept, &mut whole] {
        r.destroy_props(&mut device);
    }
}

/// Swaying grass blades both covering previously solid (`ao < 1.0`) pixels and uncovering
/// previously grass-covered solid pixels between kept frames (with zero CPU `ao_regions` and no
/// water) stay byte-for-byte identical to whole-frame AO on lavapipe.
#[test]
fn swaying_grass_covers_and_uncovers_solid_pixels_without_stale_ao() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    world.materials.push(Material {
        name: "lawn".into(),
        base_color: [0.3, 0.5, 0.22, 1.0],
        ..Material::default()
    });
    world.meshes.push(Mesh {
        material: Some(0),
        ..make_quad_mesh(0.0, 3.0)
    });
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    let mut look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    look.details.push(DetailRule {
        patterns: vec!["lawn".into()],
        preset: DetailPreset::Grass,
        scale: [1.0, 1.0],
        strength: 1.0,
        variation: 0.0,
        color: None,
        gloss: 0.0,
    });
    look.grass.density = 1.0;
    look.grass.height = 0.45;
    look.grass.width = 0.065;
    look.grass.sway = 1.6;
    look.grass.flowers = 0.25;
    let size = (160, 120);
    let quality = Quality {
        reflection_scale: 0.0,
        ao_scale: 1.0,
        ao_samples: 12,
        ..Quality::realtime()
    };
    let (sp, sn, suv, sidx) = sphere_prop_data();
    let mut make = || {
        let mut r = Renderer::new(
            &mut device,
            &world,
            look.clone(),
            quality,
            size,
            Output::Surface(Format::Rgba8Unorm),
        )
        .expect("renderer");
        r.finish_ao(&mut device).expect("baked ao");
        let ao = readable_ao(&mut device, &mut r);
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
            .expect("mesh");
        let mat = r
            .create_prop_material(
                &mut device,
                &MaterialDesc {
                    name: "boulder".into(),
                    base_color: [0.55, 0.52, 0.5, 1.0],
                    ..MaterialDesc::default()
                },
            )
            .expect("mat");
        // Stationary boulder on the grassy ground so its contact crease has strong AO (< 1.0)
        // right where swaying blades cross back and forth.
        r.create_prop_instance(
            mesh,
            mat,
            &AoGarden::placed([0.0, 0.24, 0.0], [0.5, 0.24, 0.4]),
        )
        .expect("boulder");
        let out = canvas(&mut device, size);
        (r, out, ao)
    };
    let (mut kept, k_out, k_ao) = make();
    let (mut whole, w_out, w_ao) = make();
    let view = View {
        name: "close_meadow".into(),
        eye: [0.0, 0.85, 1.35],
        target: [0.0, 0.14, 0.0],
        fov: 48.0,
        ..View::default()
    };
    let make_env = |seconds: f32| {
        let w = Weather {
            wind: 1.6,
            ..Weather::default()
        };
        Environment::resolve(
            &TimeOfDay {
                hours: 15.0,
                ..TimeOfDay::default()
            },
            &w,
            &look.palette,
            seconds,
        )
    };

    // Settle the boulder with paused time so CPU `ao_regions` become completely empty.
    let env0 = make_env(0.0);
    let mut prev_ao = vec![];
    for _ in 0..10 {
        prev_ao = present_with_ao(&mut device, &mut kept, (&view, &env0), k_out, k_ao, None).1;
    }
    assert!(kept.last_frame_stats().ao_kept);
    assert!(!kept.last_frame_stats().ssao_drawn);
    assert_eq!(kept.last_frame_stats().ao_changed_pixels, 0);

    let decode = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    };
    let mut covered_occluded = 0usize;
    let mut uncovered_occluded = 0usize;
    for (step, t) in [0.22, 0.55, 0.95, 1.4, 1.9, 2.45].into_iter().enumerate() {
        let env = make_env(t);
        let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
        let stats = kept.last_frame_stats();
        assert!(stats.ao_kept, "step {step}");
        assert_eq!(stats.ao_changed_pixels, 0, "step {step}: zero CPU regions");
        assert!(stats.ssao_drawn, "step {step}: grass swayed");

        whole.forget_world_cache();
        whole.forget_prop_shadows();
        let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
        let what = format!("swaying grass step {step} (t = {t})");
        assert_same_ao(&device, &a_ao, &b_ao, &what);
        assert_same_frame(&device, &a, &b, &what);

        let before = decode(&prev_ao);
        let after = decode(&a_ao);
        for (&u, &v) in before.iter().zip(&after) {
            if u < 0.98 && v == 1.0 {
                covered_occluded += 1;
            } else if u == 1.0 && v < 0.98 {
                uncovered_occluded += 1;
            }
        }
        prev_ao = a_ao;
    }
    assert!(
        covered_occluded > 0 && uncovered_occluded > 0,
        "expected grass to both cover ({covered_occluded}) and uncover ({uncovered_occluded}) occluded solid pixels"
    );

    // Pause time again: SSAO pass skips completely while remaining byte-identical.
    let env_hold = make_env(2.45);
    let (a, a_ao) = present_with_ao(
        &mut device,
        &mut kept,
        (&view, &env_hold),
        k_out,
        k_ao,
        None,
    );
    let stats = kept.last_frame_stats();
    assert!(stats.ao_kept && !stats.ssao_drawn && stats.ao_changed_pixels == 0);
    whole.forget_world_cache();
    whole.forget_prop_shadows();
    let (b, b_ao) = present_with_ao(
        &mut device,
        &mut whole,
        (&view, &env_hold),
        w_out,
        w_ao,
        None,
    );
    assert_same_ao(&device, &a_ao, &b_ao, "paused grass skip");
    assert_same_frame(&device, &a, &b, "paused grass skip");

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut kept, &mut whole] {
        r.destroy_props(&mut device);
    }
}

/// `set_look` (adding grassy detail rules or changing `ao_radius`) and `set_quality` (switching
/// between half-res `ao_scale: 0.5` and full-res `ao_scale: 1.0`) invalidate `ao_for` and keep
/// half-res and full-res AO byte-for-byte identical to whole-frame AO.
#[test]
fn look_and_quality_changes_invalidate_kept_ao_at_half_and_full_res() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let base_look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let half_res = Quality {
        reflection_scale: 0.0,
        ao_scale: 0.5,
        ao_samples: 8,
        ..Quality::realtime()
    };
    let size = (160, 120);
    let [
        (mut kept, k_ball, k_out, mut k_ao),
        (mut whole, w_ball, w_out, mut w_ao),
    ] = plain_floor_pair(&mut device, size, half_res, base_look.clone());
    let view = View {
        name: "floor".into(),
        eye: [0.0, 1.3, 2.2],
        target: [0.0, 0.1, 0.1],
        fov: 50.0,
        ..View::default()
    };
    let make_env = |look: &Look, seconds: f32| {
        Environment::resolve(
            &TimeOfDay {
                hours: 15.0,
                ..TimeOfDay::default()
            },
            &Weather::default(),
            &look.palette,
            seconds,
        )
    };

    // Half-res kept AO warm-up + kept step.
    let env0 = make_env(&base_look, 0.0);
    present_with_ao(&mut device, &mut kept, (&view, &env0), k_out, k_ao, None);
    let (_, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env0), k_out, k_ao, None);
    assert!(kept.last_frame_stats().ao_kept);
    whole.forget_world_cache();
    whole.forget_prop_shadows();
    let (_, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env0), w_out, w_ao, None);
    assert_eq!(a_ao.len(), 80 * 60 * 2);
    assert_same_ao(&device, &a_ao, &b_ao, "half-res kept");

    // 1. `set_look` making floor grassy invalidates `ao_for`.
    let mut grassy_look = base_look.clone();
    grassy_look.details.push(DetailRule {
        patterns: vec!["floor".into()],
        preset: DetailPreset::Grass,
        scale: [1.0, 1.0],
        strength: 1.0,
        variation: 0.0,
        color: None,
        gloss: 0.0,
    });
    grassy_look.grass.density = 1.0;
    grassy_look.grass.height = 0.32;
    grassy_look.grass.sway = 1.0;
    kept.set_look(grassy_look.clone());
    whole.set_look(grassy_look.clone());
    for (step, t) in [0.1, 0.45].into_iter().enumerate() {
        let env = make_env(&grassy_look, t);
        let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
        assert_eq!(
            kept.last_frame_stats().ao_kept,
            step > 0,
            "grassy step {step}"
        );
        whole.forget_world_cache();
        whole.forget_prop_shadows();
        let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
        assert_same_ao(&device, &a_ao, &b_ao, "after grassy set_look");
        assert_same_frame(&device, &a, &b, "after grassy set_look");
    }

    // 2. `set_look` changing `ao_radius` invalidates `ao_for`.
    let mut wide_ao_look = grassy_look.clone();
    wide_ao_look.ao_radius *= 1.8;
    kept.set_look(wide_ao_look.clone());
    whole.set_look(wide_ao_look.clone());
    let env = make_env(&wide_ao_look, 0.45);
    let (_, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
    assert!(!kept.last_frame_stats().ao_kept);
    whole.forget_world_cache();
    whole.forget_prop_shadows();
    let (_, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "after ao_radius set_look");

    // 3. `set_quality` switching to full-res AO (`ao_scale: 1.0`, `ao_samples: 16`).
    let full_res = Quality {
        ao_scale: 1.0,
        ao_samples: 16,
        ..half_res
    };
    device.destroy_buffer(k_ao);
    device.destroy_buffer(w_ao);
    kept.set_quality(&mut device, full_res)
        .expect("kept quality");
    whole
        .set_quality(&mut device, full_res)
        .expect("whole quality");
    k_ao = readable_ao(&mut device, &mut kept);
    w_ao = readable_ao(&mut device, &mut whole);
    for (step, (x, t)) in [(-0.4, 0.5), (-0.25, 0.85)].into_iter().enumerate() {
        let m = AoGarden::placed([x, 0.2, 0.4], [0.2; 3]);
        kept.set_prop_instance(k_ball, &m, true).expect("kept ball");
        whole
            .set_prop_instance(w_ball, &m, true)
            .expect("whole ball");
        let env = make_env(&wide_ao_look, t);
        let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
        assert_eq!(a_ao.len(), 160 * 120 * 2);
        assert_eq!(
            kept.last_frame_stats().ao_kept,
            step > 0,
            "full-res step {step}"
        );
        whole.forget_world_cache();
        whole.forget_prop_shadows();
        let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
        assert_same_ao(&device, &a_ao, &b_ao, "after full-res set_quality");
        let (off, largest) = off_by_more_than_a_step(&a, &b);
        let llvmpipe = device.name().starts_with("llvmpipe");
        assert!(
            largest <= 1 || !llvmpipe && off <= 0.001,
            "after full-res set_quality: up to {largest} steps apart"
        );
    }

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut kept, &mut whole] {
        r.destroy_props(&mut device);
    }
}

/// A quiet kept frame skips the SSAO pass completely (`ao_kept == true`, `ssao_drawn == false`,
/// `ao_changed_pixels == 0`); when CPU `ao_regions` cover >= 50% of the viewport, `encode` falls
/// back cleanly to whole-frame SSAO (`ao_kept == false`, `ao_changed_pixels == width * height`)
/// and resumes keeping on the next quiet frame.
#[test]
fn quiet_frame_skips_ssao_and_half_screen_change_falls_back_to_whole_ssao() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let quality = Quality {
        reflection_scale: 0.0,
        ao_scale: 1.0,
        ..Quality::realtime()
    };
    let size = (160, 120);
    let full_pixels = size.0 * size.1;
    let [
        (mut kept, k_ball, k_out, k_ao),
        (mut whole, w_ball, w_out, w_ao),
    ] = plain_floor_pair(&mut device, size, quality, look.clone());
    let view = View {
        name: "floor".into(),
        eye: [0.0, 1.4, 2.2],
        target: [0.0, 0.5, 0.0],
        fov: 50.0,
        ..View::default()
    };
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 14.0,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &look.palette,
        0.0,
    );

    // Frame 0 (whole) -> Frame 1 (quiet: live prop unmoved, no water, no grass).
    present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
    let stats = kept.last_frame_stats();
    assert!(stats.ao_kept && !stats.ssao_drawn && stats.ao_changed_pixels == 0);
    whole.forget_world_cache();
    whole.forget_prop_shadows();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "quiet skip");
    assert_same_frame(&device, &a, &b, "quiet skip");

    // Expand the live prop right in front of the eye so `ao_regions` cover >= 50% of the screen.
    let big = AoGarden::placed([0.0, 0.9, 1.25], [1.1, 0.85, 0.35]);
    kept.set_prop_instance(k_ball, &big, true)
        .expect("kept big");
    whole
        .set_prop_instance(w_ball, &big, true)
        .expect("whole big");
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
    let stats = kept.last_frame_stats();
    assert!(
        !stats.ao_kept && stats.ssao_drawn && stats.ao_changed_pixels == full_pixels,
        "expected >=50% whole-pass fallback, got {stats:?}"
    );
    whole.forget_world_cache();
    whole.forget_prop_shadows();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "whole-pass fallback");
    assert_same_frame(&device, &a, &b, "whole-pass fallback");

    // Next frame with the large prop stationary immediately resumes keeping & skipping SSAO.
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
    let stats = kept.last_frame_stats();
    assert!(stats.ao_kept && !stats.ssao_drawn && stats.ao_changed_pixels == 0);
    whole.forget_world_cache();
    whole.forget_prop_shadows();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "resume after fallback");
    assert_same_frame(&device, &a, &b, "resume after fallback");

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut kept, &mut whole] {
        r.destroy_props(&mut device);
    }
}

/// Across odd/prime and `% 4` / `% 8` boundary viewport dimensions (`601x1309`..`604x1312` plus
/// `133x101` and `135x103` covering all four `half_size % 4` remainders `{0, 1, 2, 3}`), with
/// grass and flowers reaching the viewport edges, both world water and a `Shading::Water` prop
/// (`water_props_rect`), a dry prop inside `water_rect`, and grass sway toggled on and off, kept
/// `ao_tex` and output pixels match whole-frame SSAO byte-for-byte on lavapipe.
#[test]
fn odd_viewport_grass_boundaries_and_water_props_match_whole_ao() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    for (name, shading, base_color) in [
        ("lawn", Shading::Standard, [0.28, 0.48, 0.2, 1.0]),
        ("pond", Shading::Water, [0.1, 0.32, 0.42, 0.9]),
    ] {
        world.materials.push(Material {
            name: name.into(),
            base_color,
            shading,
            ..Material::default()
        });
    }
    for (material, y, half, x, z) in [(0, 0.0, 5.0, 0.0, 0.0), (1, 0.02, 0.55, 0.45, -0.35)] {
        world.meshes.push(Mesh {
            material: Some(material),
            ..make_quad_mesh(y, half)
        });
        let mut at = math::IDENTITY;
        (at[12], at[14]) = (x, z);
        world.instances.push(Instance {
            mesh: material,
            transform: Transform(at),
        });
    }
    let make_look = |sway: f32| {
        let mut look = Look {
            painterly: 0.0,
            ..Look::default()
        };
        look.details.push(DetailRule {
            patterns: vec!["lawn".into()],
            preset: DetailPreset::Grass,
            scale: [1.0, 1.0],
            strength: 1.0,
            variation: 0.0,
            color: None,
            gloss: 0.0,
        });
        look.grass.density = 1.0;
        look.grass.height = 0.36;
        look.grass.width = 0.055;
        look.grass.flowers = 0.25;
        look.grass.sway = sway;
        look
    };
    let sway_look = make_look(1.4);
    let still_look = make_look(0.0);
    let quality = Quality {
        reflection_scale: 0.0,
        ao_scale: 0.5,
        ao_samples: 10,
        ..Quality::realtime()
    };
    let init_size = (133, 101);
    let (sp, sn, suv, sidx) = sphere_prop_data();
    let mut build = || {
        let mut r = Renderer::new(
            &mut device,
            &world,
            sway_look.clone(),
            quality,
            init_size,
            Output::Surface(Format::Rgba8Unorm),
        )
        .expect("renderer");
        r.finish_ao(&mut device).expect("baked ao");
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
            .expect("mesh");
        let stone = r
            .create_prop_material(
                &mut device,
                &MaterialDesc {
                    name: "stone".into(),
                    base_color: [0.58, 0.55, 0.5, 1.0],
                    ..MaterialDesc::default()
                },
            )
            .expect("stone");
        let basin_water = r
            .create_prop_material(
                &mut device,
                &MaterialDesc {
                    name: "basin_water".into(),
                    base_color: [0.14, 0.38, 0.46, 1.0],
                    shading: Shading::Water,
                    casts_shadow: false,
                    ..MaterialDesc::default()
                },
            )
            .expect("basin_water");
        // Dry boulder overlapping the world pond (`in_water_rect && !water`) and edge boulder.
        r.create_prop_instance(
            mesh,
            stone,
            &AoGarden::placed([0.45, 0.18, -0.35], [0.22, 0.18, 0.22]),
        )
        .expect("pond boulder");
        r.create_prop_instance(
            mesh,
            stone,
            &AoGarden::placed([0.52, 0.2, 0.55], [0.26, 0.2, 0.24]),
        )
        .expect("edge boulder");
        // Scene prop using `Shading::Water` (`water_props_rect`) on the left side of the lawn.
        r.create_prop_instance(
            mesh,
            basin_water,
            &AoGarden::placed([-0.48, 0.08, 0.15], [0.3, 0.05, 0.3]),
        )
        .expect("water prop");
        let mover = r
            .create_prop_instance(
                mesh,
                stone,
                &AoGarden::placed([-0.15, 0.16, 0.3], [0.16; 3]),
            )
            .expect("mover");
        (r, mover)
    };
    let (mut kept, k_mover) = build();
    let (mut whole, w_mover) = build();
    let view = View {
        name: "boundary_meadow".into(),
        eye: [0.0, 1.15, 1.85],
        target: [0.0, 0.08, 0.0],
        fov: 52.0,
        ..View::default()
    };
    let make_env = |look: &Look, seconds: f32| {
        let w = Weather {
            wind: 1.5,
            ..Weather::default()
        };
        Environment::resolve(
            &TimeOfDay {
                hours: 15.0,
                ..TimeOfDay::default()
            },
            &w,
            &look.palette,
            seconds,
        )
    };

    let sizes = [
        (133, 101),
        (135, 103),
        (601, 1309),
        (602, 1310),
        (603, 1311),
        (604, 1312),
    ];
    for size in sizes {
        let ao_scale = if size.0 < 200 { 1.0 } else { 0.5 };
        let q = Quality {
            ao_scale,
            ..quality
        };
        kept.set_quality(&mut device, q).expect("kept quality");
        whole.set_quality(&mut device, q).expect("whole quality");
        kept.resize(&mut device, size).expect("kept resize");
        whole.resize(&mut device, size).expect("whole resize");
        kept.set_look(sway_look.clone());
        whole.set_look(sway_look.clone());
        let k_ao = readable_ao(&mut device, &mut kept);
        let w_ao = readable_ao(&mut device, &mut whole);
        let k_out = canvas(&mut device, size);
        let w_out = canvas(&mut device, size);

        for (step, (x, t)) in [(-0.15, 0.1), (-0.08, 0.45), (-0.08, 0.85)]
            .into_iter()
            .enumerate()
        {
            let m = AoGarden::placed([x, 0.16, 0.3], [0.16; 3]);
            kept.set_prop_instance(k_mover, &m, true)
                .expect("kept mover");
            whole
                .set_prop_instance(w_mover, &m, true)
                .expect("whole mover");
            let env = make_env(&sway_look, t);
            let (a, a_ao) =
                present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
            assert_eq!(
                kept.last_frame_stats().ao_kept,
                step > 0,
                "{size:?} sway step {step}"
            );
            whole.forget_ao();
            let (b, b_ao) =
                present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
            assert!(!whole.last_frame_stats().ao_kept);
            let what = format!("odd grass/water {size:?} sway step {step}");
            assert_same_ao(&device, &a_ao, &b_ao, &what);
            assert_same_frame(&device, &a, &b, &what);
        }

        // Toggle grass sway off (`sway = 0.0`): water still ripples inside `water_rect`, while
        // non-water grass & dry tiles skip grass union/coarse work and match whole AO.
        kept.set_look(still_look.clone());
        whole.set_look(still_look.clone());
        for (step, t) in [1.1, 1.5].into_iter().enumerate() {
            let env = make_env(&still_look, t);
            let (a, a_ao) =
                present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
            assert_eq!(
                kept.last_frame_stats().ao_kept,
                step > 0,
                "{size:?} still-grass step {step}"
            );
            whole.forget_ao();
            let (b, b_ao) =
                present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
            assert!(!whole.last_frame_stats().ao_kept);
            let what = format!("odd grass/water {size:?} still step {step}");
            assert_same_ao(&device, &a_ao, &b_ao, &what);
            assert_same_frame(&device, &a, &b, &what);
        }

        for (tex, buf) in [k_out, w_out] {
            device.destroy_texture(tex);
            device.destroy_buffer(buf);
        }
        device.destroy_buffer(k_ao);
        device.destroy_buffer(w_ao);
    }

    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut kept, &mut whole] {
        r.destroy_props(&mut device);
    }
}

/// Regression for Yozakura Harbor `/test`: when a world ocean mesh extends underneath an opaque
/// deck so `Renderer::water_rect` spans the whole viewport, the `.g` visible-water channel in the
/// `Rgba8Unorm` AO pyramid skips dry deck pixels in `fs_ssao`, reuses the water mask with zero
/// pyramid passes on quiet water-only frames, recaptures when props move or opaque water props
/// draw, and matches whole-frame SSAO byte-for-byte throughout.
#[test]
fn occluded_ocean_under_opaque_deck_skips_dry_pixels_and_reuses_water_mask() {
    assert_eq!(crate::render::renderer::ssao::AO_CHANGED_BYTES, 288);

    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    world.materials.push(Material {
        name: "deck".into(),
        base_color: [0.58, 0.48, 0.38, 1.0],
        roughness: 0.75,
        shading: Shading::Standard,
        ..Material::default()
    });
    world.materials.push(Material {
        name: "ocean".into(),
        base_color: [0.08, 0.28, 0.38, 0.88],
        shading: Shading::Water,
        alpha: AlphaMode::Blend,
        ..Material::default()
    });
    world.meshes.push(Mesh {
        material: Some(0),
        ..make_quad_mesh(0.35, 7.0)
    });
    let mut deck_xf = math::IDENTITY;
    deck_xf[14] = 5.2;
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(deck_xf),
    });
    world.meshes.push(Mesh {
        material: Some(1),
        ..make_quad_mesh(0.0, 12.0)
    });
    world.instances.push(Instance {
        mesh: 1,
        transform: Transform(math::IDENTITY),
    });

    let size = (256, 192);
    let look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    let quality = Quality {
        ao_scale: 0.5,
        ..Quality::realtime()
    };
    let view = View {
        name: "harbor_deck".into(),
        eye: [0.0, 2.4, 3.6],
        target: [0.0, 0.35, -1.2],
        fov: 52.0,
        ..View::default()
    };
    let make_env = |seconds: f32| {
        Environment::resolve(
            &TimeOfDay {
                hours: 16.0,
                ..TimeOfDay::default()
            },
            &Weather::default(),
            &look.palette,
            seconds,
        )
    };

    let (sp, sn, suv, sidx) = sphere_prop_data();
    let setup = |device: &mut Device<'_>| {
        let mut r = Renderer::new(
            device,
            &world,
            look.clone(),
            quality,
            size,
            Output::Surface(Format::Rgba8Unorm),
        )
        .expect("renderer");
        r.finish_ao(device).expect("ao");
        for id in [
            r.targets.ao_grass_half,
            r.targets.ao_grass[0],
            r.targets.ao_grass[1],
            r.targets.ao_grass_union,
            r.targets.ao_grass_coarse,
        ] {
            assert_eq!(
                device.texture_desc(id).expect("mask desc").format,
                Format::Rgba8Unorm
            );
        }
        let mesh = r
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
        let prop_mat = r
            .create_prop_material(
                device,
                &MaterialDesc {
                    name: "crate".into(),
                    base_color: [0.75, 0.52, 0.34, 1.0],
                    ..MaterialDesc::default()
                },
            )
            .expect("prop mat");
        let water_mat = r
            .create_prop_material(
                device,
                &MaterialDesc {
                    name: "fountain".into(),
                    base_color: [0.15, 0.42, 0.52, 1.0],
                    shading: Shading::Water,
                    ..MaterialDesc::default()
                },
            )
            .expect("water mat");
        r.create_prop_instance(
            mesh,
            prop_mat,
            &AoGarden::placed([-0.7, 0.6, 0.6], [0.25; 3]),
        )
        .expect("pillar");
        let mover = r
            .create_prop_instance(
                mesh,
                prop_mat,
                &AoGarden::placed([0.6, 0.6, 0.8], [0.22; 3]),
            )
            .expect("mover");
        r.set_prop_instance(mover, &math::IDENTITY, false)
            .expect("hide mover");
        let fountain = r
            .create_prop_instance(mesh, water_mat, &math::IDENTITY)
            .expect("fountain");
        r.set_prop_instance(fountain, &math::IDENTITY, false)
            .expect("hide fountain");
        let ao = readable_ao(device, &mut r);
        let out = canvas(device, size);
        (r, mover, fountain, out, ao)
    };

    let (mut kept, k_mover, k_fountain, k_out, k_ao) = setup(&mut device);
    let (mut whole, w_mover, w_fountain, w_out, w_ao) = setup(&mut device);

    let eye = kept.eye(&view, [0.0; 2], size);
    let frustum = Frustum::from_view_proj(&eye.view_proj);
    let w_rect = kept
        .water_rect(&eye, &frustum, size, true)
        .expect("water rect");
    let w_area = (w_rect[2] - w_rect[0]) * (w_rect[3] - w_rect[1]);
    assert!(
        w_area * 4 >= size.0 * size.1 * 3,
        "expected unoccluded ocean rect to cover >=75% of viewport, got {w_rect:?}"
    );

    let env0 = make_env(0.1);
    for _ in 0..10 {
        let _ = present_with_ao(&mut device, &mut kept, (&view, &env0), k_out, k_ao, None);
        let _ = present_with_ao(&mut device, &mut whole, (&view, &env0), w_out, w_ao, None);
    }
    assert!(kept.last_frame_stats().world_cached);
    kept.forget_ao();

    // Frame 0 (`!ao_kept`): captures water into slot 1; union/coarse deferred to first kept frame.
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env0), k_out, k_ao, None);
    assert!(!kept.last_frame_stats().ao_kept);
    assert_eq!(kept.targets.ao_grass_cur, 1);
    assert_eq!(kept.targets.ao_grass_water, [false, true]);
    assert!(!kept.targets.ao_union_water);
    whole.forget_ao();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env0), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "occluded ocean frame 0");
    assert_same_frame(&device, &a, &b, "occluded ocean frame 0");

    // Frame 1 (`ao_kept`): reuses slot 1 without recapturing half/tile, builds union + coarse once.
    let env1 = make_env(0.55);
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env1), k_out, k_ao, None);
    assert!(kept.last_frame_stats().ao_kept);
    assert_eq!(kept.targets.ao_grass_cur, 1);
    assert_eq!(kept.targets.ao_grass_water, [false, true]);
    assert!(kept.targets.ao_union_water);
    whole.forget_ao();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env1), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "occluded ocean frame 1");
    assert_same_frame(&device, &a, &b, "occluded ocean frame 1");

    // Frame 2 (`ao_kept`, quiet water-only): 0 pyramid passes run (`ao_grass_cur` stays 1).
    let env2 = make_env(1.05);
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env2), k_out, k_ao, None);
    assert!(kept.last_frame_stats().ao_kept);
    assert_eq!(kept.targets.ao_grass_cur, 1);
    assert!(kept.targets.ao_union_water);
    whole.forget_ao();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env2), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "occluded ocean frame 2");
    assert_same_frame(&device, &a, &b, "occluded ocean frame 2");

    // Verify `fs_ssao` discards the ~80% dry deck pixels over the occluded ocean!
    let (computed, total) =
        recomputed_texels(&mut device, &mut kept, (&view, &make_env(1.4)), k_out, k_ao);
    assert!(
        computed > 0 && computed * 3 < total,
        "expected fs_ssao to skip occluded deck pixels, got {computed}/{total}"
    );

    // Re-warm `kept`, then exercise moving deck props, opaque water props, and >50% `!keep_ao`.
    let _ = present_with_ao(
        &mut device,
        &mut kept,
        (&view, &make_env(1.6)),
        k_out,
        k_ao,
        None,
    );
    for (step, (mx, show_water, huge, t)) in [
        (Some(0.45), false, false, 1.9),
        (Some(-0.15), true, false, 2.3),
        (Some(-0.15), true, false, 2.7),
        (None, false, true, 3.1),
        (None, false, false, 3.5),
        (None, false, false, 3.9),
    ]
    .into_iter()
    .enumerate()
    {
        for (r, m_id, f_id) in [
            (&mut kept, k_mover, k_fountain),
            (&mut whole, w_mover, w_fountain),
        ] {
            if huge {
                r.set_prop_instance(m_id, &AoGarden::placed([0.0, 1.6, 1.9], [1.6; 3]), true)
                    .expect("huge mover");
            } else if let Some(x) = mx {
                r.set_prop_instance(m_id, &AoGarden::placed([x, 0.6, 0.8], [0.22; 3]), true)
                    .expect("mover");
            } else {
                r.set_prop_instance(m_id, &math::IDENTITY, false)
                    .expect("hide mover");
            }
            if show_water {
                r.set_prop_instance(f_id, &AoGarden::placed([0.55, 0.58, 1.3], [0.2; 3]), true)
                    .expect("show fountain");
            } else {
                r.set_prop_instance(f_id, &math::IDENTITY, false)
                    .expect("hide fountain");
            }
        }
        let env = make_env(t);
        let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env), k_out, k_ao, None);
        if huge {
            assert!(
                !kept.last_frame_stats().ao_kept,
                "huge fallback step {step}"
            );
        }
        whole.forget_ao();
        let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env), w_out, w_ao, None);
        let what = format!("occluded ocean dynamic step {step}");
        assert_same_ao(&device, &a_ao, &b_ao, &what);
        assert_same_frame(&device, &a, &b, &what);
    }

    for (tex, buf) in [k_out, w_out] {
        device.destroy_texture(tex);
        device.destroy_buffer(buf);
    }
    device.destroy_buffer(k_ao);
    device.destroy_buffer(w_ao);
    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut kept, &mut whole] {
        r.destroy_props(&mut device);
    }
}

/// In a dry scene (`!water_drawn`) with swaying grass (`sway > 0`), hiding or moving an opaque
/// prop while `env.seconds` is paused (`!grass_changed`) uncovers grass blades and must recapture
/// `ao_grass[cur_idx]` so that when the clock unpauses on a subsequent frame and those blades sway
/// out of their uncovered tiles, `ao_grass_union` includes the uncovered tiles and matches
/// whole-frame SSAO byte-for-byte.
#[test]
fn unpausing_after_prop_uncovers_swaying_grass_matches_whole_ao() {
    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut world = World::default();
    world.materials.push(Material {
        name: "stone_floor".into(),
        base_color: [0.48, 0.46, 0.44, 1.0],
        ..Material::default()
    });
    world.materials.push(Material {
        name: "tuft".into(),
        base_color: [0.3, 0.5, 0.22, 1.0],
        ..Material::default()
    });
    world.meshes.push(Mesh {
        material: Some(0),
        ..make_quad_mesh(0.0, 4.0)
    });
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(math::IDENTITY),
    });
    // High corner canopy so `scene.land_max.y` (`2.0`) already encloses `blocker` (`y < 1.0`),
    // keeping `cascades()` and `world_key` invariant when `blocker` hides or moves.
    world.meshes.push(Mesh {
        material: Some(0),
        ..make_quad_mesh(2.0, 0.1)
    });
    let mut post_xf = math::IDENTITY;
    post_xf[12] = -3.5;
    post_xf[14] = -3.5;
    world.instances.push(Instance {
        mesh: 1,
        transform: Transform(post_xf),
    });
    // Small grass planter so blade roots stay on the left while peak gusts (`dir = (0.8, 0.6)`) bend
    // tips across the ray toward the boulder's dark contact crease, then retract out on trough gusts.
    world.meshes.push(Mesh {
        material: Some(1),
        ..make_quad_mesh(0.002, 0.14)
    });
    let mut tuft_xf = math::IDENTITY;
    tuft_xf[12] = -0.22;
    tuft_xf[14] = -0.15;
    world.instances.push(Instance {
        mesh: 2,
        transform: Transform(tuft_xf),
    });

    let mut look = Look {
        painterly: 0.0,
        ..Look::default()
    };
    look.details.push(DetailRule {
        patterns: vec!["tuft".into()],
        preset: DetailPreset::Grass,
        scale: [1.0, 1.0],
        strength: 1.0,
        variation: 0.0,
        color: None,
        gloss: 0.0,
    });
    look.grass.density = 1.0;
    look.grass.height = 0.65;
    look.grass.width = 0.08;
    look.grass.sway = 1.4;
    look.grass.flowers = 0.0;

    let size = (240, 180);
    let quality = Quality {
        reflection_scale: 0.0,
        ao_scale: 1.0,
        ao_samples: 12,
        ..Quality::realtime()
    };
    let view = View {
        name: "paused_uncover".into(),
        eye: [0.0, 1.6, 3.2],
        target: [0.0, 0.1, 0.0],
        fov: 50.0,
        ..View::default()
    };
    let make_env = |seconds: f32| {
        let w = Weather {
            wind: 6.0,
            ..Weather::default()
        };
        Environment::resolve(
            &TimeOfDay {
                hours: 15.0,
                ..TimeOfDay::default()
            },
            &w,
            &look.palette,
            seconds,
        )
    };

    let (sp, sn, suv, sidx) = sphere_prop_data();
    let setup = |device: &mut Device<'_>| {
        let mut r = Renderer::new(
            device,
            &world,
            look.clone(),
            quality,
            size,
            Output::Surface(Format::Rgba8Unorm),
        )
        .expect("renderer");
        r.finish_ao(device).expect("baked ao");
        let ao = readable_ao(device, &mut r);
        let mesh = r
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
                    name: "stone".into(),
                    base_color: [0.55, 0.52, 0.5, 1.0],
                    ..MaterialDesc::default()
                },
            )
            .expect("mat");
        // Stationary boulder on the camera ray behind the bent blade tips (`[0.23, 0.50, 0.19]`) so
        // its floor contact crease (`z ~ -0.80..-1.35`) has dark screen-space AO `< 1.0`.
        r.create_prop_instance(
            mesh,
            mat,
            &AoGarden::placed([0.35, 0.25, -1.1], [0.45, 0.3, 0.3]),
        )
        .expect("boulder");
        let blocker = r
            .create_prop_instance(mesh, mat, &math::IDENTITY)
            .expect("blocker");
        r.set_prop_instance(blocker, &math::IDENTITY, false)
            .expect("hide blocker");
        let out = canvas(device, size);
        (r, blocker, out, ao)
    };

    let (mut kept, k_blocker, k_out, k_ao) = setup(&mut device);
    let (mut whole, w_blocker, w_out, w_ao) = setup(&mut device);

    // Warm up so the stationary boulder settles into `world_cache`.
    let env0 = make_env(0.2);
    for _ in 0..10 {
        let _ = present_with_ao(&mut device, &mut kept, (&view, &env0), k_out, k_ao, None);
        let _ = present_with_ao(&mut device, &mut whole, (&view, &env0), w_out, w_ao, None);
    }
    assert!(kept.last_frame_stats().world_cached);

    // Case 1: Peak gust (`t = 1.8`) bends blade tips right onto the camera ray in front of the
    // boulder's dark floor crease; `blocker` sits closer on that same ray so it occludes ONLY the
    // bent tip tiles on Frame 1, hides on Frame 2 at constant `t = 1.8`, and Frame 3 unpauses to
    // trough gust (`t = 4.2`) where the tips retract out of those tiles!
    let cover_xf = AoGarden::placed([0.2, 0.66, 0.64], [0.16, 0.16, 0.1]);
    for (r, b) in [(&mut kept, k_blocker), (&mut whole, w_blocker)] {
        r.set_prop_instance(b, &cover_xf, true)
            .expect("show blocker");
    }
    let env_t1 = make_env(1.8);
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env_t1), k_out, k_ao, None);
    let slot_f1 = kept.targets.ao_grass_cur;
    whole.forget_ao();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env_t1), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "paused grass frame 1 (covered)");
    assert_same_frame(&device, &a, &b, "paused grass frame 1 (covered)");

    // Frame 2: hide blocker at constant `t = 1.8` (`!grass_changed`, `!regions.is_empty()`).
    for (r, b) in [(&mut kept, k_blocker), (&mut whole, w_blocker)] {
        r.set_prop_instance(b, &math::IDENTITY, false)
            .expect("hide blocker");
    }
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env_t1), k_out, k_ao, None);
    let stats_f2 = kept.last_frame_stats();
    assert!(
        stats_f2.ao_kept && stats_f2.ssao_drawn && stats_f2.ao_changed_pixels > 0,
        "f2 stats: {stats_f2:?}"
    );
    assert_eq!(
        kept.targets.ao_grass_cur,
        1 - slot_f1,
        "uncovering swaying grass at constant seconds must recapture active grass slot"
    );
    whole.forget_ao();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env_t1), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "paused grass frame 2 (uncovered)");
    assert_same_frame(&device, &a, &b, "paused grass frame 2 (uncovered)");

    // Frame 3: unpause clock (`t = 4.2`, grass retracts out of uncovered tip tiles, 0 CPU regions).
    let env_t3 = make_env(4.2);
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env_t3), k_out, k_ao, None);
    let stats_f3 = kept.last_frame_stats();
    assert!(stats_f3.ao_kept && stats_f3.ssao_drawn && stats_f3.ao_changed_pixels == 0);
    whole.forget_ao();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env_t3), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "unpaused grass frame 3");
    assert_same_frame(&device, &a, &b, "unpaused grass frame 3");

    // Case 2: Move blocker aside at constant `t = 1.8`, hold one completely quiet frame at `t = 1.8`,
    // then unpause to `t = 4.2`.
    for (r, b) in [(&mut kept, k_blocker), (&mut whole, w_blocker)] {
        r.set_prop_instance(b, &cover_xf, true)
            .expect("show blocker 2");
    }
    let env_t4 = make_env(1.8);
    let _ = present_with_ao(&mut device, &mut kept, (&view, &env_t4), k_out, k_ao, None);
    whole.forget_ao();
    let _ = present_with_ao(&mut device, &mut whole, (&view, &env_t4), w_out, w_ao, None);

    let aside_xf = AoGarden::placed([-0.85, 0.25, 0.6], [0.16, 0.2, 0.16]);
    for (r, b) in [(&mut kept, k_blocker), (&mut whole, w_blocker)] {
        r.set_prop_instance(b, &aside_xf, true)
            .expect("move blocker aside");
    }
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env_t4), k_out, k_ao, None);
    whole.forget_ao();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env_t4), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "moved aside at paused t=1.8");
    assert_same_frame(&device, &a, &b, "moved aside at paused t=1.8");

    // Quiet paused frame (`0` pyramid passes, `0` ssao pass) before unpausing.
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env_t4), k_out, k_ao, None);
    assert!(kept.last_frame_stats().ao_kept && !kept.last_frame_stats().ssao_drawn);
    whole.forget_ao();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env_t4), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "quiet paused frame before unpause");
    assert_same_frame(&device, &a, &b, "quiet paused frame before unpause");

    let env_t5 = make_env(4.2);
    let (a, a_ao) = present_with_ao(&mut device, &mut kept, (&view, &env_t5), k_out, k_ao, None);
    whole.forget_ao();
    let (b, b_ao) = present_with_ao(&mut device, &mut whole, (&view, &env_t5), w_out, w_ao, None);
    assert_same_ao(&device, &a_ao, &b_ao, "unpaused after quiet frame");
    assert_same_frame(&device, &a, &b, "unpaused after quiet frame");

    for (tex, buf) in [k_out, w_out] {
        device.destroy_texture(tex);
        device.destroy_buffer(buf);
    }
    device.destroy_buffer(k_ao);
    device.destroy_buffer(w_ao);
    let errs = device.validation_errors();
    assert!(errs.is_empty(), "validation errors: {errs:?}");
    for r in [&mut kept, &mut whole] {
        r.destroy_props(&mut device);
    }
}
