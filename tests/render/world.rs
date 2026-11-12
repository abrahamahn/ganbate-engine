use super::*;
use ganbate_engine::render::{
    DetailPreset, DetailRule, Look, MaterialDesc, MeshData, Output, Quality,
};

#[test]
#[ignore = "requires a Vulkan driver"]
fn accumulated_stills_and_resize() {
    let mut device = Device::headless().expect("device");
    let mut quality = Quality::still();
    quality.shadow_size = 512;
    let mut r = Renderer::new(
        &mut device,
        &world(),
        Look::default(),
        quality,
        (W, H),
        Output::Still,
    )
    .expect("renderer");
    let env = Environment::resolve(
        &TimeOfDay::default(),
        &Weather::default(),
        &r.look().palette,
        1.0,
    );
    let a = r
        .render_still(&mut device, &view(), &env, 4)
        .expect("four samples");
    assert_eq!(a.len(), (W * H * 4) as usize);
    r.resize(&mut device, (W / 2, H / 2)).expect("resize");
    let b = r
        .render_still(&mut device, &view(), &env, 1)
        .expect("after resize");
    assert_eq!(b.len(), (W * H) as usize);
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

/// Lens shift translates the image without rotating the camera, and a camera can look straight
/// down when it names another up direction.
#[test]
#[ignore = "requires a Vulkan driver"]
fn lens_shift_and_looking_straight_down() {
    let mut device = Device::headless().expect("device");
    let mut box_ = material("box", [0.9, 0.05, 0.05, 1.0], Shading::Standard);
    box_.metallic = 0.0;
    let world = World {
        instances: (0..2)
            .map(|mesh| Instance {
                mesh,
                transform: Transform::IDENTITY,
            })
            .collect(),
        meshes: vec![quad(0.0, 20.0, 0), cube([0.0, 0.5, 0.0], 0.5, 1)],
        materials: vec![
            material("ground", [0.6, 0.6, 0.6, 1.0], Shading::Standard),
            box_,
        ],
        textures: vec![],
        lights: vec![],
    };
    let mut quality = Quality::realtime();
    quality.shadow_size = 512;
    let mut r = Renderer::new(
        &mut device,
        &world,
        Look::default(),
        quality,
        (W, H),
        Output::Still,
    )
    .expect("renderer");
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 15.0,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &r.look().palette,
        3.0,
    );
    // The centroid of red-dominant pixels.
    let red = |px: &[u8]| {
        let (mut sx, mut sy, mut n) = (0.0, 0.0, 0.0);
        for y in 0..H {
            for x in 0..W {
                let i = ((y * W + x) * 4) as usize;
                if u32::from(px[i]) > u32::from(px[i + 1]) + 40 {
                    sx += x as f32 + 0.5;
                    sy += y as f32 + 0.5;
                    n += 1.0;
                }
            }
        }
        assert!(n > 10.0, "the box is in view");
        [sx / n, sy / n]
    };
    let size = (W, H);
    let plain = view();
    let base = red(&r
        .render_still(&mut device, &plain, &env, 1)
        .expect("render"));
    for shift in [[0.0, 0.2], [0.3, 0.0], [-0.25, -0.15]] {
        let shifted = View { shift, ..view() };
        // The view axis lands `shift` from the centre (y up), whatever the depth.
        let target = r.project(&shifted, size, shifted.target).expect("in front");
        let want = [
            W as f32 * 0.5 * (1.0 + shift[0]),
            H as f32 * 0.5 * (1.0 - shift[1]),
        ];
        assert!(
            (target[0] - want[0]).abs() < 0.01 && (target[1] - want[1]).abs() < 0.01,
            "{shift:?}: {target:?} vs {want:?}"
        );
        let moved = red(&r
            .render_still(&mut device, &shifted, &env, 1)
            .expect("render"));
        let (dx, dy) = (moved[0] - base[0], moved[1] - base[1]);
        let (wx, wy) = (shift[0] * W as f32 * 0.5, -shift[1] * H as f32 * 0.5);
        eprintln!("shift {shift:?}: the box moved {dx:.2}, {dy:.2} px (want {wx:.2}, {wy:.2})");
        assert!((dx - wx).abs() < 1.0 && (dy - wy).abs() < 1.0);
    }
    // Straight down, with north (-z) at the top of the image.
    let down = View {
        eye: [0.0, 12.0, 0.0],
        target: [0.0, 0.0, 0.0],
        up: [0.0, 0.0, -1.0],
        ..view()
    };
    let top = red(&r.render_still(&mut device, &down, &env, 1).expect("render"));
    assert!(
        (top[0] - W as f32 * 0.5).abs() < 2.0 && (top[1] - H as f32 * 0.5).abs() < 2.0,
        "{top:?}"
    );
    let north = r.project(&down, size, [0.0, 0.0, -3.0]).expect("in front");
    let east = r.project(&down, size, [3.0, 0.0, 0.0]).expect("in front");
    assert!(
        north[1] < H as f32 * 0.5 && east[0] > W as f32 * 0.5,
        "{north:?} {east:?}"
    );
    // The same view with the default up (parallel to the line of sight) still renders.
    let degenerate = View {
        up: [0.0, 1.0, 0.0],
        ..down.clone()
    };
    let px = r
        .render_still(&mut device, &degenerate, &env, 1)
        .expect("render");
    assert!(region(&px, 0, 0, W, H) > 20.0, "not black");
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

/// Depth of field: past the focus the scene softens (a far wall of black and white squares loses
/// its edges), what is nearer stays exactly as it was, and overlays are drawn sharp over the blur.
#[test]
#[ignore = "requires a Vulkan driver"]
fn focus_softens_what_lies_past_it() {
    use ganbate_engine::render::{BlendMode, Focus};
    let mut device = Device::headless().expect("device");
    let mut wall = material("wall", [1.0; 4], Shading::Standard);
    wall.base_color_texture = Some(0);
    // 2 m squares on a wall 40 m out, facing the camera; a red box 10 m out.
    let far = Mesh {
        positions: vec![
            Vec3(-40.0, -5.0, -30.0),
            Vec3(40.0, -5.0, -30.0),
            Vec3(40.0, 35.0, -30.0),
            Vec3(-40.0, 35.0, -30.0),
        ],
        normals: vec![Vec3(0.0, 0.0, 1.0); 4],
        uvs: vec![[0.0, 10.0], [20.0, 10.0], [20.0, 0.0], [0.0, 0.0]],
        indices: vec![0, 1, 2, 0, 2, 3],
        color: [0.5; 3],
        material: Some(0),
    };
    let world = World {
        instances: (0..2)
            .map(|mesh| Instance {
                mesh,
                transform: Transform::IDENTITY,
            })
            .collect(),
        meshes: vec![far, cube([0.0, 0.5, 0.0], 0.8, 1)],
        materials: vec![
            wall,
            material("box", [0.9, 0.05, 0.05, 1.0], Shading::Standard),
        ],
        textures: vec![Texture {
            width: 2,
            height: 2,
            srgb: true,
            levels: vec![
                vec![
                    0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255,
                ],
                vec![128, 128, 128, 255],
            ],
            encoded: None,
        }],
        lights: vec![],
    };
    let mut quality = Quality::realtime();
    quality.shadow_size = 512;
    let mut r = Renderer::new(
        &mut device,
        &world,
        Look::default(),
        quality,
        (W, H),
        Output::Still,
    )
    .expect("renderer");
    // A green badge over the wall, drawn on top as an overlay.
    let badge_mesh = r
        .create_prop_mesh(
            &mut device,
            &MeshData {
                positions: &[
                    [-14.0, 0.0, -29.0],
                    [-10.0, 0.0, -29.0],
                    [-10.0, 4.0, -29.0],
                    [-14.0, 4.0, -29.0],
                ],
                normals: &[],
                uvs: &[],
                indices: &[0, 1, 2, 0, 2, 3],
            },
        )
        .expect("badge mesh");
    let green = r
        .create_prop_material(
            &mut device,
            &MaterialDesc {
                name: "badge".into(),
                base_color: [0.0, 1.0, 0.0, 1.0],
                overlay: true,
                blend: BlendMode::Alpha,
                depth_test: false,
                ..MaterialDesc::default()
            },
        )
        .expect("badge");
    let mut identity = [0.0f32; 16];
    for i in [0, 5, 10, 15] {
        identity[i] = 1.0;
    }
    r.create_prop_instance(badge_mesh, green, &identity)
        .expect("badge instance");
    let env = Environment::resolve(
        &TimeOfDay {
            hours: 12.0,
            ..TimeOfDay::default()
        },
        &Weather::default(),
        &r.look().palette,
        3.0,
    );
    let sharp = r
        .render_still(&mut device, &view(), &env, 1)
        .expect("render");
    let focused = View {
        focus: Some(Focus {
            distance: 12.0,
            depth: 6.0,
            blur: 0.08,
        }),
        ..view()
    };
    let soft = r
        .render_still(&mut device, &focused, &env, 1)
        .expect("render");
    // How much neighbouring pixels differ across a region: the squares' edges.
    let edges = |px: &[u8], x0: u32, y0: u32, x1: u32, y1: u32| {
        let mut sum = 0.0;
        let mut n = 0.0;
        for y in y0..y1 {
            for x in x0..x1 {
                sum += (luma(px, x + 1, y) - luma(px, x, y)).abs()
                    + (luma(px, x, y + 1) - luma(px, x, y)).abs();
                n += 1.0;
            }
        }
        sum / n
    };
    // Right of the badge, which is drawn alike in both.
    let (crisp, blurred) = (edges(&sharp, 60, 2, 120, 18), edges(&soft, 60, 2, 120, 18));
    eprintln!("the far wall's edges: {crisp:.1} sharp, {blurred:.1} with the focus at 12 m");
    assert!(crisp > 8.0, "the squares show: {crisp}");
    assert!(blurred < crisp * 0.4, "{blurred} vs {crisp}");
    // The box, nearer than the focus, is untouched.
    let mut most = 0u8;
    for y in 33..40 {
        for x in 60..68 {
            let i = ((y * W + x) * 4) as usize;
            for c in 0..3 {
                most = most.max(sharp[i + c].abs_diff(soft[i + c]));
            }
        }
    }
    assert!(most <= 1, "the box changed by {most}");
    let i = ((H / 2 * W + W / 2) * 4) as usize;
    assert!(
        u32::from(soft[i]) > u32::from(soft[i + 1]) + 40,
        "the box is in view"
    );
    // The badge keeps its exact colour over the blur.
    let at = r
        .project(&focused, (W, H), [-12.0, 2.0, -29.0])
        .expect("badge in front");
    let b = ((at[1] as u32 * W + at[0] as u32) * 4) as usize;
    assert_eq!(&soft[b..b + 3], &[0, 255, 0], "the badge stays sharp");
    assert_eq!(&soft[b..b + 3], &sharp[b..b + 3]);
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

#[test]
#[ignore = "requires a Vulkan driver"]
fn world_instances_move_hide_and_pick_like_a_recooked_world() {
    let mut device = Device::headless().expect("device");
    let base = world();
    let look = || Look::default();
    let still = |device: &mut Device<'_>, w: &World| {
        let mut r = Renderer::new(device, w, look(), Quality::still(), (W, H), Output::Still)
            .expect("renderer");
        render(device, &mut r, 15.0, "clear")
    };
    // The red box (instance 1) 2.5 m to the right and 0.5 m up, cooked that way.
    let moved_at = Transform::from_trs([2.5, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0; 3]).unwrap();
    let mut recooked = base.clone();
    recooked.instances[1].transform = moved_at;
    let expected_moved = still(&mut device, &recooked);
    // The same box hidden: cooked without it.
    let mut without = base.clone();
    without.instances.remove(1);
    let expected_hidden = still(&mut device, &without);
    // Instances in another cooked order draw the same picture (draws group by material).
    let mut shuffled = base.clone();
    shuffled.instances.reverse();
    let original = still(&mut device, &base);
    let reordered = still(&mut device, &shuffled);
    assert!(
        difference(&original, &reordered) < 0.01,
        "order changed the picture: {}",
        difference(&original, &reordered)
    );

    let mut r = Renderer::new(
        &mut device,
        &base,
        look(),
        Quality::still(),
        (W, H),
        Output::Still,
    )
    .expect("renderer");
    assert_eq!(r.world_instances(), base.instances.len());
    let first = render(&mut device, &mut r, 15.0, "clear");
    assert!(difference(&first, &original) < 0.01);
    // Pick the box where it stands, then move it.
    let at = r
        .project(&view(), (W, H), [0.0, 1.0, 1.0])
        .expect("in view");
    let hit = r
        .pick_world(&view(), (W, H), (at[0], at[1]))
        .expect("the box is under its projection");
    assert_eq!(hit.instance, 1, "{hit:?}");
    r.set_world_instance(&mut device, 1, moved_at.0, true)
        .expect("move");
    assert_eq!(r.world_instance(1), Some((moved_at.0, true)));
    let moved = render(&mut device, &mut r, 15.0, "clear");
    let d = difference(&moved, &expected_moved);
    assert!(d < 0.05, "moved world differs from the recooked one by {d}");
    assert!(difference(&moved, &first) > 0.2, "nothing moved");
    // The old place now picks the ground; the new one the box.
    let old = r
        .pick_world(&view(), (W, H), (at[0], at[1]))
        .expect("ground");
    assert_ne!(old.instance, 1);
    let new_at = r
        .project(&view(), (W, H), [2.5, 1.5, 1.0])
        .expect("in view");
    assert_eq!(
        r.pick_world(&view(), (W, H), (new_at[0], new_at[1]))
            .map(|p| p.instance),
        Some(1)
    );
    // Hidden, it is gone from the picture and from picking.
    r.set_world_instance(&mut device, 1, moved_at.0, false)
        .expect("hide");
    let hidden = render(&mut device, &mut r, 15.0, "clear");
    let d = difference(&hidden, &expected_hidden);
    assert!(
        d < 0.05,
        "hidden world differs from one cooked without it by {d}"
    );
    assert_ne!(
        r.pick_world(&view(), (W, H), (new_at[0], new_at[1]))
            .map(|p| p.instance),
        Some(1)
    );
    // Moving the lamp moves its light.
    r.set_world_instance(&mut device, 1, Transform::IDENTITY.0, true)
        .expect("back");
    r.set_world_light(&mut device, 0, [-3.0, 1.6, 1.0])
        .expect("lamp");
    assert!(
        r.set_world_instance(&mut device, 99, Transform::IDENTITY.0, true)
            .is_err()
    );
    assert!(r.set_world_light(&mut device, 5, [0.0; 3]).is_err());
    let mut bad = Transform::IDENTITY.0;
    bad[0] = f32::NAN;
    assert!(r.set_world_instance(&mut device, 1, bad, true).is_err());
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

#[test]
#[ignore = "requires a Vulkan driver"]
fn moved_land_keeps_its_distant_shallows() {
    // The world map (shallows and foam past the ground map) follows land moved with
    // `set_world_instance`: moved 150 m along X, beyond where the map first reached, the land's
    // far shallows (red here) show as in a world cooked with the land there.
    let mut device = Device::headless().expect("device");
    let (w, h) = (256, 144);
    let look = || Look {
        details: vec![DetailRule {
            patterns: vec!["land".into()],
            preset: DetailPreset::Terrain,
            scale: [0.3, 0.3],
            strength: 1.0,
            variation: 0.1,
            color: None,
            gloss: 0.0,
        }],
        fog: 0.0,
        water_reflection: 0.0,
        water_shallow: [1.0, 0.0, 0.0],
        ..Look::default()
    };
    let quality = || {
        let mut q = Quality::realtime();
        q.shadow_size = 512;
        q.ground_size = 256;
        q.world_size = 512;
        q
    };
    let shift = 150.0;
    let far = View {
        name: "offshore".into(),
        eye: [shift, 25.0, 190.0],
        target: [shift, 0.0, 0.0],
        fov: 40.0,
        ..View::default()
    };
    let noon = TimeOfDay {
        hours: 12.0,
        ..TimeOfDay::default()
    };
    let red = |image: &[u8]| {
        image
            .chunks(4)
            .filter(|p| {
                let (r, g, b) = (f32::from(p[0]), f32::from(p[1]), f32::from(p[2]));
                r > 1.2 * g + 6.0 && r > 1.2 * b + 6.0
            })
            .count()
    };
    let moved_at = Transform::from_trs([shift, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0; 3]).unwrap();
    let mut cooked = shore();
    cooked.instances[0].transform = moved_at;
    let mut r = Renderer::new(
        &mut device,
        &cooked,
        look(),
        quality(),
        (w, h),
        Output::Still,
    )
    .expect("renderer");
    let env = Environment::resolve(&noon, &Weather::default(), &r.look().palette, 2.0);
    let expected = r.render_still(&mut device, &far, &env, 1).expect("render");
    // Cooked where it was and drawn once (the world map included), then moved.
    let mut r = Renderer::new(
        &mut device,
        &shore(),
        look(),
        quality(),
        (w, h),
        Output::Still,
    )
    .expect("renderer");
    r.render_still(&mut device, &far, &env, 1).expect("render");
    r.set_world_instance(&mut device, 0, moved_at.0, true)
        .expect("move");
    let moved = r.render_still(&mut device, &far, &env, 1).expect("render");
    let (want, got) = (red(&expected), red(&moved));
    let d = difference(&moved, &expected);
    eprintln!("far red pixels: {want} cooked there, {got} moved there; difference {d}");
    assert!(want > 40, "no shallows in the cooked world: {want}");
    assert!(
        d < 0.05,
        "moved land differs from land cooked there by {d} ({got} / {want} red pixels)"
    );
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}
