use super::*;
use ganbate_engine::render::{Look, MaterialDesc, MeshData, Output, Quality};

/// Props: a run-time mesh drawn twice from one batch, moved, recoloured, hidden, picked and
/// projected, casting a shadow on the world. A prop renders like the same mesh cooked into the
/// world.
#[test]
#[ignore = "requires a Vulkan driver"]
fn props_draw_move_pick_and_cast_shadows() {
    // Late afternoon: long shadows.
    const HOURS: f32 = 16.5;
    let mut device = Device::headless().expect("device");
    let mut quality = Quality::realtime();
    quality.shadow_size = 512;
    quality.ground_size = 256;
    quality.world_size = 256;
    let ground = material("ground", [0.6, 0.6, 0.6, 1.0], Shading::Standard);
    let red = MaterialDesc {
        name: "prop-red".into(),
        base_color: [0.9, 0.05, 0.05, 1.0],
        ..MaterialDesc::default()
    };
    let new = |device: &mut Device<'_>, world: &World| {
        Renderer::new(
            device,
            world,
            Look::default(),
            quality,
            (W, H),
            Output::Still,
        )
        .expect("renderer")
    };
    let instance = |mesh| Instance {
        mesh,
        transform: Transform::IDENTITY,
    };
    // The reference: the cube cooked into the world at x = 2.
    let mut cooked_red = material("prop-red", red.base_color, Shading::Standard);
    cooked_red.metallic = red.metallic;
    cooked_red.roughness = red.roughness;
    let mut reference = new(
        &mut device,
        &World {
            instances: vec![instance(0), instance(1)],
            meshes: vec![quad(0.0, 20.0, 0), cube([2.0, 0.5, 0.0], 0.5, 1)],
            materials: vec![ground.clone(), cooked_red],
            textures: vec![],
            lights: vec![],
        },
    );
    let cooked = render(&mut device, &mut reference, HOURS, "clear");
    let mut r = new(
        &mut device,
        &World {
            instances: vec![instance(0)],
            meshes: vec![quad(0.0, 20.0, 0)],
            materials: vec![ground],
            textures: vec![],
            lights: vec![],
        },
    );
    // A unit cube in model space.
    let c = cube([0.0, 0.0, 0.0], 0.5, 0);
    let positions: Vec<[f32; 3]> = c.positions.iter().map(|p| [p.0, p.1, p.2]).collect();
    let normals: Vec<[f32; 3]> = c.normals.iter().map(|n| [n.0, n.1, n.2]).collect();
    let mesh = r
        .create_prop_mesh(
            &mut device,
            &MeshData {
                positions: &positions,
                normals: &normals,
                uvs: &c.uvs,
                indices: &c.indices,
            },
        )
        .expect("mesh");
    let red = r.create_prop_material(&mut device, &red).expect("material");
    let at = |x: f32, y: f32, z: f32| {
        let mut m = [0.0f32; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        m[12] = x;
        m[13] = y;
        m[14] = z;
        m
    };
    let right = r
        .create_prop_instance(mesh, red, &at(2.0, 0.5, 0.0))
        .expect("right");
    // Pixels that differ visibly between two frames.
    let differing = |a: &[u8], b: &[u8]| {
        a.chunks(4)
            .zip(b.chunks(4))
            .filter(|(p, q)| (0..3).any(|c| p[c].abs_diff(q[c]) > 12))
            .count()
    };
    let alone = render(&mut device, &mut r, HOURS, "clear");
    let off = differing(&alone, &cooked);
    // Without its shadow the prop no longer matches: the shadow pass draws it.
    let mut unshadowed = r.prop_material(red).expect("desc").clone();
    unshadowed.casts_shadow = false;
    r.update_prop_material(&mut device, red, &unshadowed)
        .expect("no shadow");
    let missing = differing(&render(&mut device, &mut r, HOURS, "clear"), &cooked);
    eprintln!(
        "pixels unlike the cooked cube: {off} as a prop, {missing} without its shadow \
         (mean difference {:.3})",
        difference(&alone, &cooked)
    );
    assert!(
        off <= 8,
        "a prop renders like the cooked mesh: {off} pixels differ"
    );
    assert!(missing > off + 20, "the prop's shadow: {missing} vs {off}");
    unshadowed.casts_shadow = true;
    r.update_prop_material(&mut device, red, &unshadowed)
        .expect("shadow again");
    let left = r
        .create_prop_instance(mesh, red, &at(-2.0, 0.5, 0.0))
        .expect("left");
    // Red-dominant pixels (in the bright afternoon light the cube reads as a warm pink).
    let reddish = |px: &[u8], x0: u32, x1: u32| {
        let mut n = 0;
        for y in 0..H {
            for x in x0..x1 {
                let i = ((y * W + x) * 4) as usize;
                let (r, g, b) = (px[i], px[i + 1], px[i + 2]);
                if u32::from(r) > u32::from(g) + 40 && u32::from(r) > u32::from(b) + 40 {
                    n += 1;
                }
            }
        }
        n
    };
    let both = render(&mut device, &mut r, HOURS, "clear");
    let (l, rr) = (reddish(&both, 0, W / 2), reddish(&both, W / 2, W));
    eprintln!("red pixels: {l} left, {rr} right");
    assert!(l > 30 && rr > 30, "both instances drawn: {l} / {rr}");
    // Projection lands on the cube, and picking there finds it.
    let size = (W, H);
    let p = r.project(&view(), size, [2.0, 0.5, 0.0]).expect("in front");
    assert!(
        p[0] > W as f32 / 2.0 && p[1] > 0.0 && p[1] < H as f32,
        "{p:?}"
    );
    // Up in the world is up in the image, and the projected centre is on the drawn cube.
    let top = r.project(&view(), size, [2.0, 1.0, 0.0]).expect("in front");
    assert!(top[1] < p[1], "image rows run down: {top:?} {p:?}");
    let i = ((p[1] as u32 * W + p[0] as u32) * 4) as usize;
    assert!(
        u32::from(both[i]) > u32::from(both[i + 1]) + 40,
        "the cube is under its projection: {:?}",
        &both[i..i + 4]
    );
    let hit = r
        .pick(&view(), size, (p[0], p[1]), u32::MAX)
        .expect("a pick");
    assert_eq!(hit.instance, right);
    assert!(
        (hit.point[0] - 2.0).abs() < 0.6 && hit.distance > 5.0,
        "{hit:?}"
    );
    assert!(
        r.pick(&view(), size, (2.0, 2.0), u32::MAX).is_none(),
        "sky has no prop"
    );
    // Hide one, move the other far left: the right half has no red.
    r.set_prop_instance(right, &at(2.0, 0.5, 0.0), false)
        .expect("hide");
    r.set_prop_instance(left, &at(-3.0, 0.5, 0.0), true)
        .expect("move");
    let moved = render(&mut device, &mut r, HOURS, "clear");
    let (l2, r2) = (reddish(&moved, 0, W / 2), reddish(&moved, W / 2, W));
    eprintln!("after hiding and moving: {l2} left, {r2} right");
    assert!(l2 > 30 && r2 == 0, "{l2} / {r2}");
    r.destroy_prop_instance(left);
    assert_eq!(r.prop_instances(), 1);
    // Recolour through the material: the hidden instance stays hidden, so show it again.
    let mut blue = r.prop_material(red).expect("desc").clone();
    blue.base_color = [0.05, 0.1, 0.9, 1.0];
    r.update_prop_material(&mut device, red, &blue)
        .expect("update");
    r.set_prop_instance(right, &at(2.0, 0.5, 0.0), true)
        .expect("show");
    let recoloured = render(&mut device, &mut r, HOURS, "clear");
    assert_eq!(
        reddish(&recoloured, 0, W),
        0,
        "no red after the material changed"
    );
    r.destroy_props(&mut device);
    assert_eq!(r.prop_instances(), 0);
    assert!(r.pick(&view(), size, (p[0], p[1]), u32::MAX).is_none());
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

/// A renderer without a cooked world draws the sky, and props on their own. Prop bounds widen
/// the shadow depth range: a prop floor 30 m above the (placeholder) world bounds still takes
/// a prop cube's shadow.
#[test]
#[ignore = "requires a Vulkan driver"]
fn empty_world_with_props() {
    let mut device = Device::headless().expect("device");
    let empty = World {
        instances: vec![],
        meshes: vec![],
        materials: vec![],
        textures: vec![],
        lights: vec![],
    };
    let mut quality = Quality::realtime();
    quality.shadow_size = 512;
    let mut r = Renderer::new(
        &mut device,
        &empty,
        Look::default(),
        quality,
        (W, H),
        Output::Still,
    )
    .expect("renderer");
    let sky = render(&mut device, &mut r, 12.0, "clear");
    assert!(region(&sky, 0, 0, W, H) > 40.0, "the sky is drawn");
    let mesh_of = |device: &mut Device<'_>, r: &mut Renderer, m: &Mesh| {
        let positions: Vec<[f32; 3]> = m.positions.iter().map(|p| [p.0, p.1, p.2]).collect();
        r.create_prop_mesh(
            device,
            &MeshData {
                positions: &positions,
                normals: &[],
                uvs: &[],
                indices: &m.indices,
            },
        )
        .expect("mesh")
    };
    let cube_mesh = mesh_of(&mut device, &mut r, &cube([0.0, 0.0, 0.0], 0.5, 0));
    let floor_mesh = mesh_of(&mut device, &mut r, &quad(0.0, 6.0, 0));
    let mut red = MaterialDesc {
        name: "red".into(),
        base_color: [0.9, 0.05, 0.05, 1.0],
        ..MaterialDesc::default()
    };
    let red_id = r.create_prop_material(&mut device, &red).expect("red");
    let grey = r
        .create_prop_material(
            &mut device,
            &MaterialDesc {
                name: "floor".into(),
                base_color: [0.6, 0.6, 0.6, 1.0],
                ..MaterialDesc::default()
            },
        )
        .expect("grey");
    let at = |y: f32| {
        let mut m = [0.0f32; 16];
        for i in [0, 5, 10, 15] {
            m[i] = 1.0;
        }
        m[13] = y;
        m
    };
    let cube = r
        .create_prop_instance(cube_mesh, red_id, &at(0.5))
        .expect("cube");
    let with = render(&mut device, &mut r, 12.0, "clear");
    let i = (((H / 2) * W + W / 2) * 4) as usize;
    assert!(
        u32::from(with[i]) > u32::from(with[i + 1]) + 40,
        "the prop is drawn: {:?}",
        &with[i..i + 4]
    );
    r.set_prop_instance(cube, &at(33.0), true).expect("raise");
    r.create_prop_instance(floor_mesh, grey, &at(30.0))
        .expect("floor");
    let lit = |device: &mut Device<'_>, r: &mut Renderer| {
        let v = View {
            eye: [0.0, 37.0, 9.0],
            target: [0.0, 30.5, 0.0],
            ..view()
        };
        let time = TimeOfDay {
            hours: 15.5,
            ..TimeOfDay::default()
        };
        let env = Environment::resolve(
            &time,
            &Weather::preset("clear").expect("clear"),
            &r.look().palette,
            3.0,
        );
        r.render_still(device, &v, &env, 1).expect("render")
    };
    let shadowed = lit(&mut device, &mut r);
    red.casts_shadow = false;
    r.update_prop_material(&mut device, red_id, &red)
        .expect("no shadow");
    let unshadowed = lit(&mut device, &mut r);
    let darker = shadowed
        .chunks(4)
        .zip(unshadowed.chunks(4))
        .filter(|(a, b)| u32::from(a[1]) + 12 < u32::from(b[1]))
        .count();
    eprintln!("pixels darkened by the prop's shadow on the prop floor: {darker}");
    assert!(darker > 25, "the cube shades the floor: {darker}");
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

/// Prop layers: blended props mix with the scene by alpha (tinted per instance), additive
/// ones add, overlays land over the finished image in their exact colour (on top of
/// everything without a depth test, hidden behind nearer surfaces with one) in draw order,
/// and picking follows what is drawn on top.
#[test]
#[ignore = "requires a Vulkan driver"]
fn prop_layers_blend_overlay_tint_and_pick() {
    use ganbate_engine::render::BlendMode;
    let mut device = Device::headless().expect("device");
    let world = World {
        instances: vec![Instance {
            mesh: 0,
            transform: Transform::IDENTITY,
        }],
        meshes: vec![quad(0.0, 20.0, 0)],
        materials: vec![material(
            "ground",
            [0.15, 0.15, 0.15, 1.0],
            Shading::Standard,
        )],
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
    let mesh_of = |device: &mut Device<'_>, r: &mut Renderer, m: &Mesh| {
        let positions: Vec<[f32; 3]> = m.positions.iter().map(|p| [p.0, p.1, p.2]).collect();
        let normals: Vec<[f32; 3]> = m.normals.iter().map(|n| [n.0, n.1, n.2]).collect();
        r.create_prop_mesh(
            device,
            &MeshData {
                positions: &positions,
                normals: &normals,
                uvs: &m.uvs,
                indices: &m.indices,
            },
        )
        .expect("mesh")
    };
    let cube_mesh = mesh_of(&mut device, &mut r, &cube([0.0, 0.0, 0.0], 0.5, 0));
    // A 1.2 m card facing +z (the camera).
    let card = Mesh {
        positions: vec![
            Vec3(-0.6, -0.6, 0.0),
            Vec3(0.6, -0.6, 0.0),
            Vec3(0.6, 0.6, 0.0),
            Vec3(-0.6, 0.6, 0.0),
        ],
        normals: vec![Vec3(0.0, 0.0, 1.0); 4],
        uvs: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        indices: vec![0, 1, 2, 0, 2, 3],
        color: [0.5; 3],
        material: Some(0),
    };
    let card_mesh = mesh_of(&mut device, &mut r, &card);
    let at = |x: f32, y: f32, z: f32| {
        let mut m = [0.0f32; 16];
        for i in [0, 5, 10, 15] {
            m[i] = 1.0;
        }
        m[12] = x;
        m[13] = y;
        m[14] = z;
        m
    };
    let new_material = |device: &mut Device<'_>, r: &mut Renderer, d: MaterialDesc| {
        r.create_prop_material(device, &d).expect("material")
    };
    let white = new_material(
        &mut device,
        &mut r,
        MaterialDesc {
            name: "white".into(),
            base_color: [0.2, 0.2, 0.2, 1.0],
            ..MaterialDesc::default()
        },
    );
    let cube_at = r
        .create_prop_instance(cube_mesh, white, &at(0.0, 0.5, 0.0))
        .expect("cube");
    let size = (W, H);
    let centre = r.project(&view(), size, [0.0, 0.5, 0.5]).expect("in front");
    let (cx, cy) = (centre[0] as u32, centre[1] as u32);
    let px = |img: &[u8]| {
        let i = ((cy * W + cx) * 4) as usize;
        [img[i], img[i + 1], img[i + 2]]
    };
    let plain = render(&mut device, &mut r, 15.0, "clear");
    let base = px(&plain);
    eprintln!("grey cube: {base:?}");
    assert!(base.iter().all(|&c| c < 230), "not saturated: {base:?}");

    // Blended: a red card in front of the cube, faded by its instance tint.
    let glass = new_material(
        &mut device,
        &mut r,
        MaterialDesc {
            name: "glass".into(),
            base_color: [0.6, 0.02, 0.02, 1.0],
            shading: Shading::Unlit,
            blend: BlendMode::Alpha,
            ..MaterialDesc::default()
        },
    );
    let pane = r
        .create_prop_instance(card_mesh, glass, &at(0.0, 0.5, 1.0))
        .expect("pane");
    let mut seen = vec![];
    for a in [1.0, 0.5, 0.0] {
        r.set_prop_instance_tint(pane, [1.0, 1.0, 1.0, a])
            .expect("tint");
        seen.push(px(&render(&mut device, &mut r, 15.0, "clear")));
    }
    eprintln!("red pane at alpha 1, 0.5, 0: {seen:?}");
    let redness = |p: [u8; 3]| i32::from(p[0]) - i32::from(p[1]);
    assert!(redness(seen[0]) > 100, "opaque red: {:?}", seen[0]);
    assert!(
        redness(seen[1]) > 30 && redness(seen[1]) < redness(seen[0]) - 30,
        "half red: {:?}",
        seen[1]
    );
    assert!(
        seen[2].iter().zip(base).all(|(a, b)| a.abs_diff(b) <= 2),
        "invisible: {:?} vs {base:?}",
        seen[2]
    );
    r.set_prop_instance(pane, &at(0.0, 0.5, 1.0), false)
        .expect("hide");

    // Additive: adds blue.
    let glow = new_material(
        &mut device,
        &mut r,
        MaterialDesc {
            name: "glow".into(),
            base_color: [0.0, 0.0, 0.3, 1.0],
            shading: Shading::Unlit,
            blend: BlendMode::Add,
            ..MaterialDesc::default()
        },
    );
    let flare = r
        .create_prop_instance(card_mesh, glow, &at(0.0, 0.5, 1.0))
        .expect("flare");
    let added = px(&render(&mut device, &mut r, 15.0, "clear"));
    eprintln!("blue glow over the cube: {added:?}");
    assert!(
        added[2] > base[2].saturating_add(2) || base[2] >= 250,
        "{added:?} vs {base:?}"
    );
    assert!(added[0].abs_diff(base[0]) <= 3, "only blue is added");
    r.destroy_prop_instance(flare);

    // Overlays behind the cube: on top without a depth test, hidden with one; exact colour.
    let overlay = |device: &mut Device<'_>, r: &mut Renderer, c: [f32; 4], depth_test, order| {
        r.create_prop_material(
            device,
            &MaterialDesc {
                name: "badge".into(),
                base_color: c,
                overlay: true,
                blend: BlendMode::Alpha,
                depth_test,
                order,
                ..MaterialDesc::default()
            },
        )
        .expect("overlay")
    };
    let green = overlay(&mut device, &mut r, [0.0, 1.0, 0.0, 1.0], false, 0.0);
    // Behind the cube on the line of sight through the measured pixel.
    let eye = view().eye;
    let front = [0.0, 0.5, 0.5];
    let ray = [front[0] - eye[0], front[1] - eye[1], front[2] - eye[2]];
    let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
    let behind = |d: f32| {
        at(
            front[0] + ray[0] / len * d,
            front[1] + ray[1] / len * d,
            front[2] + ray[2] / len * d,
        )
    };
    let badge = r
        .create_prop_instance(card_mesh, green, &behind(2.0))
        .expect("badge");
    let full = render(&mut device, &mut r, 15.0, "clear");
    let on_top = px(&full);
    eprintln!("green overlay behind the cube, no depth test: {on_top:?}");
    assert!(
        on_top[0] <= 2 && on_top[1] >= 253 && on_top[2] <= 2,
        "exact colour on top: {on_top:?}"
    );
    let hit = r.pick(&view(), size, (cx as f32, cy as f32), u32::MAX);
    assert_eq!(
        hit.map(|h| h.instance),
        Some(badge),
        "the overlay on top is picked"
    );
    r.set_prop_instance_pick(badge, 2).expect("layer 2");
    let hit = r.pick(&view(), size, (cx as f32, cy as f32), 1);
    assert_eq!(
        hit.map(|h| h.instance),
        Some(cube_at),
        "layer 1 skips the badge"
    );
    let mut tested = r.prop_material(green).expect("desc").clone();
    tested.depth_test = true;
    r.update_prop_material(&mut device, green, &tested)
        .expect("depth test");
    let hidden = px(&render(&mut device, &mut r, 15.0, "clear"));
    assert!(
        hidden.iter().zip(base).all(|(a, b)| a.abs_diff(b) <= 2),
        "hidden behind the cube: {hidden:?} vs {base:?}"
    );
    // Draw order: blue (order 1) over green (order 0), then the other way round.
    tested.depth_test = false;
    r.update_prop_material(&mut device, green, &tested)
        .expect("on top");
    let blue = overlay(&mut device, &mut r, [0.0, 0.0, 1.0, 1.0], false, 1.0);
    let second = r
        .create_prop_instance(card_mesh, blue, &behind(1.5))
        .expect("blue");
    let order01 = px(&render(&mut device, &mut r, 15.0, "clear"));
    tested.order = 2.0;
    r.update_prop_material(&mut device, green, &tested)
        .expect("order");
    let order21 = px(&render(&mut device, &mut r, 15.0, "clear"));
    eprintln!("blue over green: {order01:?}; green over blue: {order21:?}");
    assert!(order01[2] >= 253 && order01[1] <= 2, "{order01:?}");
    assert!(order21[1] >= 253 && order21[2] <= 2, "{order21:?}");
    // An order between two others draws between them: blue at 1.5 over green at 1.25.
    tested.order = 1.25;
    r.update_prop_material(&mut device, green, &tested)
        .expect("order 1.25");
    let mut over = r.prop_material(blue).expect("blue").clone();
    over.order = 1.5;
    r.update_prop_material(&mut device, blue, &over)
        .expect("order 1.5");
    let between = px(&render(&mut device, &mut r, 15.0, "clear"));
    assert!(between[2] >= 253 && between[1] <= 2, "{between:?}");
    tested.order = 2.0;
    r.update_prop_material(&mut device, green, &tested)
        .expect("order back");
    r.set_prop_instance_pick(badge, 1).expect("layer 1");
    let hit = r.pick(&view(), size, (cx as f32, cy as f32), u32::MAX);
    assert_eq!(
        hit.map(|h| h.instance),
        Some(badge),
        "the last drawn overlay is picked"
    );
    r.destroy_prop_instance(second);
    r.destroy_prop_instance(badge);
    // Opaque overlays writing depth: the nearer red one, drawn first, hides the farther yellow
    // one drawn after it; testing alone, the later one covers it.
    let solid = |device: &mut Device<'_>, r: &mut Renderer, c: [f32; 4], order| {
        r.create_prop_material(
            device,
            &MaterialDesc {
                name: "solid".into(),
                base_color: c,
                overlay: true,
                depth_test: true,
                depth_write: true,
                order,
                ..MaterialDesc::default()
            },
        )
        .expect("solid overlay")
    };
    let near = solid(&mut device, &mut r, [1.0, 0.0, 0.0, 1.0], 0.0);
    let far = solid(&mut device, &mut r, [1.0, 1.0, 0.0, 1.0], 1.0);
    r.create_prop_instance(card_mesh, near, &behind(-2.0))
        .expect("near");
    r.create_prop_instance(card_mesh, far, &behind(-1.0))
        .expect("far");
    let written = px(&render(&mut device, &mut r, 15.0, "clear"));
    let mut unwritten = r.prop_material(far).expect("desc").clone();
    unwritten.depth_write = false;
    r.update_prop_material(
        &mut device,
        near,
        &MaterialDesc {
            base_color: [1.0, 0.0, 0.0, 1.0],
            order: 0.0,
            ..unwritten.clone()
        },
    )
    .expect("near without writing");
    let tested = px(&render(&mut device, &mut r, 15.0, "clear"));
    eprintln!("near red over far yellow, writing depth: {written:?}; only testing: {tested:?}");
    assert!(written[0] >= 253 && written[1] <= 2, "{written:?}");
    assert!(tested[0] >= 253 && tested[1] >= 253, "{tested:?}");
    // Writing depth is for opaque overlays that test it.
    for bad in [
        MaterialDesc {
            depth_write: true,
            ..MaterialDesc::default()
        },
        MaterialDesc {
            overlay: true,
            depth_test: false,
            depth_write: true,
            ..MaterialDesc::default()
        },
        MaterialDesc {
            overlay: true,
            depth_write: true,
            blend: BlendMode::Alpha,
            ..MaterialDesc::default()
        },
    ] {
        assert!(
            r.create_prop_material(&mut device, &bad).is_err(),
            "{bad:?}"
        );
    }
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

/// A game's prop shader colours an overlay by its own rule from its parameters (left half the
/// first colour, right half the second plus the last four), takes new parameters without a
/// rebuild, and is refused when it reads anything but its input, or draws anything but an
/// overlay.
#[test]
#[ignore = "requires a Vulkan driver"]
fn prop_shaders_colour_overlays_from_their_params() {
    use ganbate_engine::render::BlendMode;
    let mut device = Device::headless().expect("device");
    let world = World {
        instances: vec![Instance {
            mesh: 0,
            transform: Transform::IDENTITY,
        }],
        meshes: vec![quad(0.0, 20.0, 0)],
        materials: vec![material(
            "ground",
            [0.15, 0.15, 0.15, 1.0],
            Shading::Standard,
        )],
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
    let card = r
        .create_prop_mesh(
            &mut device,
            &MeshData {
                positions: &[
                    [-1.2, -0.6, 0.0],
                    [1.2, -0.6, 0.0],
                    [1.2, 0.6, 0.0],
                    [-1.2, 0.6, 0.0],
                ],
                normals: &[[0.0, 0.0, 1.0]; 4],
                uvs: &[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
                indices: &[0, 1, 2, 0, 2, 3],
            },
        )
        .expect("card");
    let halves = r
        .create_prop_shader(
            &mut device,
            "fn shade(in: ShadeInput) -> vec4f {
               let c = select(in.p1.rgb + in.p4.rgb, in.p0.rgb, in.uv.x < 0.5);
               return vec4f(c * in.tint.rgb, 1.0);
             }",
        )
        .expect("shader");
    let mut params = [0.0f32; ganbate_engine::render::PROP_SHADER_PARAMS];
    params[..3].copy_from_slice(&[1.0, 0.0, 0.0]);
    params[4..7].copy_from_slice(&[0.0, 0.0, 1.0]);
    let desc = MaterialDesc {
        name: "halves".into(),
        overlay: true,
        depth_test: false,
        blend: BlendMode::Alpha,
        shader: Some(halves),
        params,
        ..MaterialDesc::default()
    };
    let material = r
        .create_prop_material(&mut device, &desc)
        .expect("material");
    let mut at = [0.0f32; 16];
    for i in [0, 5, 10, 15] {
        at[i] = 1.0;
    }
    at[13] = 1.0;
    at[14] = 2.0;
    r.create_prop_instance(card, material, &at)
        .expect("instance");
    let size = (W, H);
    let spot = |r: &Renderer, p: [f32; 3]| {
        let s = r.project(&view(), size, p).expect("in front");
        ((s[1] as u32 * W + s[0] as u32) * 4) as usize
    };
    let (at_left, at_right) = (spot(&r, [-0.6, 1.0, 2.0]), spot(&r, [0.6, 1.0, 2.0]));
    let pixel = |img: &[u8], i: usize| [img[i], img[i + 1], img[i + 2]];
    let img = render(&mut device, &mut r, 15.0, "clear");
    let (left, right) = (pixel(&img, at_left), pixel(&img, at_right));
    eprintln!("prop shader halves: {left:?} {right:?}");
    assert_eq!((left, right), ([255, 0, 0], [0, 0, 255]));

    // New parameters, no rebuild: green on the left now, and the last four (`p4`) add red on
    // the right.
    params[..3].copy_from_slice(&[0.0, 1.0, 0.0]);
    params[16..19].copy_from_slice(&[1.0, 0.0, 0.0]);
    r.set_prop_material_params(material, &params)
        .expect("params");
    let img = render(&mut device, &mut r, 15.0, "clear");
    assert_eq!(
        (pixel(&img, at_left), pixel(&img, at_right)),
        ([0, 255, 0], [255, 0, 255])
    );

    // A shader reads all three of its material's textures as pictures: the base (red) on the
    // left, the emissive (blue) on the right, the normal (green) sampled at another uv where
    // the base's alpha is low.
    let solid = |r: &mut Renderer, device: &mut Device<'_>, c: [u8; 4]| {
        r.create_prop_texture(device, (2, 2), &[c, c, c, c].concat(), true)
            .expect("texture")
    };
    let (red, green, blue) = (
        solid(&mut r, &mut device, [255, 0, 0, 255]),
        solid(&mut r, &mut device, [0, 255, 0, 255]),
        solid(&mut r, &mut device, [0, 0, 255, 255]),
    );
    let pictures = r
        .create_prop_shader(
            &mut device,
            "fn shade(in: ShadeInput) -> vec4f {
               let elsewhere = textureSample(normal_tex, material_sampler, in.uv * 0.5);
               let left = select(elsewhere.rgb, in.base.rgb, in.base.a > 0.5);
               let c = select(in.emissive_map.rgb, left, in.uv.x < 0.5);
               return vec4f(c * (in.normal_map.g + 0.0) , 1.0);
             }",
        )
        .expect("pictures");
    r.update_prop_material(
        &mut device,
        material,
        &MaterialDesc {
            shader: Some(pictures),
            base_texture: Some(red),
            normal_texture: Some(green),
            emissive_texture: Some(blue),
            ..desc.clone()
        },
    )
    .expect("textured");
    let img = render(&mut device, &mut r, 15.0, "clear");
    assert_eq!(
        (pixel(&img, at_left), pixel(&img, at_right)),
        ([255, 0, 0], [0, 0, 255]),
        "the base on the left, the emissive on the right"
    );

    // Refused: a shader that reads the frame block, one that does not compile, and a prop
    // shader on a material drawn in the scene.
    let reads_frame = r
        .create_prop_shader(
            &mut device,
            "fn shade(in: ShadeInput) -> vec4f { return vec4f(frame.camera.xyz, 1.0); }",
        )
        .expect_err("reads the frame");
    assert!(reads_frame.contains("reads only"), "{reads_frame}");
    assert!(
        r.create_prop_shader(&mut device, "fn shade(in: ShadeInput) -> vec4f { oops }")
            .is_err()
    );
    let in_scene = r
        .create_prop_material(
            &mut device,
            &MaterialDesc {
                overlay: false,
                ..desc.clone()
            },
        )
        .expect_err("not an overlay");
    assert!(in_scene.contains("overlays only"), "{in_scene}");
}
