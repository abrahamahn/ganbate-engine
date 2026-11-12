use super::*;
use ganbate_engine::render::{
    DetailPreset, DetailRule, LightDesc, Look, MaterialDesc, MeshData, Output, Quality, SpotCone,
    Terrain, math,
};

#[test]
#[ignore = "requires a Vulkan driver"]
fn time_of_day_weather_and_lamps() {
    let mut device = Device::headless().expect("device");
    let mut quality = Quality::realtime();
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
    assert_eq!(r.lights(), 1);
    assert!((r.water_level().unwrap() + 0.3).abs() < 1e-4);

    let noon = render(&mut device, &mut r, 12.0, "clear");
    // The sky at the top of the frame is blue and bright at noon (on average: a few clouds).
    let channel = |c: usize| -> f32 {
        let top = &noon[..(8 * W * 4) as usize];
        top.chunks(4).map(|p| f32::from(p[c])).sum::<f32>() / (8 * W) as f32
    };
    let (red, blue) = (channel(0), channel(2));
    assert!(blue > red + 15.0, "noon sky red {red} blue {blue}");
    assert!(region(&noon, 0, 0, W, 8) > 90.0);
    let midnight = render(&mut device, &mut r, 0.0, "clear");
    assert!(
        region(&midnight, 0, 0, W, 8) < region(&noon, 0, 0, W, 8) * 0.5,
        "midnight sky should be much darker"
    );
    // At midnight the lamp (right of centre, near the box) lights the ground around it.
    let near = region(&midnight, 84, 34, 104, 50);
    let far = region(&midnight, 8, 40, 28, 56);
    assert!(near > far + 8.0, "lamp-lit ground {near} vs unlit {far}");
    for weather in Weather::PRESETS {
        let px = render(&mut device, &mut r, 17.5, weather);
        assert!(region(&px, 0, 0, W, H) > 5.0, "{weather} rendered black");
    }
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

/// A game's lights: a spot lights the ground inside its cone at midnight and not outside it, a
/// light that follows the lamps is dark by day, and one put out leaves the frame as it was.
#[test]
#[ignore = "requires a Vulkan driver"]
fn game_lights_spot_and_follow_the_lamps() {
    let mut device = Device::headless().expect("device");
    let world = World {
        instances: vec![Instance {
            mesh: 0,
            transform: Transform::IDENTITY,
        }],
        meshes: vec![quad(0.0, 20.0, 0)],
        materials: vec![material("ground", [0.4, 0.4, 0.4, 1.0], Shading::Standard)],
        textures: vec![],
        lights: vec![],
    };
    let mut r = Renderer::new(
        &mut device,
        &world,
        Look::default(),
        Quality::realtime(),
        (W, H),
        Output::Still,
    )
    .expect("renderer");
    let dark = render(&mut device, &mut r, 0.0, "clear");
    // Straight down onto the ground in front of the camera, a narrow cone.
    let spot = LightDesc {
        position: [0.0, 3.0, 0.0],
        range: 8.0,
        color: [1.0, 0.85, 0.6],
        intensity: 40.0,
        spot: Some(SpotCone {
            direction: [0.0, -1.0, 0.0],
            inner: 0.25,
            outer: 0.4,
        }),
        follows_lamps: false,
        shadow: 0.0,
    };
    let light = r.create_light(&spot).expect("light");
    assert_eq!(r.game_lights(), 1);
    let lit = render(&mut device, &mut r, 0.0, "clear");
    // Where the cone's middle and a point well outside it land in the frame.
    let pixel = |p: [f32; 3]| {
        let q = r.project(&view(), (W, H), p).expect("in front");
        (q[0] as u32, q[1] as u32)
    };
    let (centre, aside) = (pixel([0.0, 0.0, 0.0]), pixel([3.0, 0.0, 0.0]));
    let at = |px: &[u8], (x, y): (u32, u32)| region(px, x - 2, y - 2, x + 2, y + 2);
    assert!(at(&lit, centre) > at(&dark, centre) + 20.0, "in the cone");
    assert!(
        (at(&lit, aside) - at(&dark, aside)).abs() < 3.0,
        "outside it"
    );
    // Following the lamps, it is dark at noon like them.
    let noon = render(&mut device, &mut r, 12.0, "clear");
    r.set_light(
        light,
        &LightDesc {
            follows_lamps: true,
            ..spot
        },
    )
    .expect("follow");
    let noon_follow = render(&mut device, &mut r, 12.0, "clear");
    assert!(
        at(&noon, centre) > at(&noon_follow, centre) + 5.0,
        "by day it follows the lamps"
    );
    r.destroy_light(light);
    assert_eq!(r.game_lights(), 0);
    let out = render(&mut device, &mut r, 0.0, "clear");
    assert!(difference(&out, &dark) < 0.5, "put out");
    assert!(r.set_light(light, &spot).is_err());
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

/// A spot casting shadows: a prop between it and the ground takes its light off the ground behind
/// the prop, as much as the light asks, and nowhere else; at most two lights cast at once.
#[test]
#[ignore = "requires a Vulkan driver"]
fn game_lights_cast_the_props_shadows() {
    let mut device = Device::headless().expect("device");
    let world = World {
        instances: vec![Instance {
            mesh: 0,
            transform: Transform::IDENTITY,
        }],
        meshes: vec![quad(0.0, 20.0, 0)],
        materials: vec![material("ground", [0.5, 0.5, 0.5, 1.0], Shading::Standard)],
        textures: vec![],
        lights: vec![],
    };
    let mut quality = Quality::realtime();
    quality.shadow_size = 512;
    quality.light_shadow_size = 512;
    let mut r = Renderer::new(
        &mut device,
        &world,
        Look::default(),
        quality,
        (W, H),
        Output::Still,
    )
    .expect("renderer");
    // A metre cube on the ground at the middle.
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
    let white = MaterialDesc {
        name: "prop-white".into(),
        base_color: [0.8, 0.8, 0.8, 1.0],
        ..MaterialDesc::default()
    };
    let white = r
        .create_prop_material(&mut device, &white)
        .expect("material");
    let mut place = math::IDENTITY;
    place[13] = 0.5;
    r.create_prop_instance(mesh, white, &place).expect("cube");
    // From up on the left, aimed past the cube: its shadow falls on the ground to its right.
    let spot = LightDesc {
        position: [-3.0, 3.0, 0.0],
        range: 12.0,
        color: [1.0, 0.9, 0.8],
        intensity: 60.0,
        spot: Some(SpotCone {
            direction: [4.5, -3.0, 0.0],
            inner: 0.3,
            outer: 0.6,
        }),
        follows_lamps: false,
        shadow: 0.0,
    };
    let light = r.create_light(&spot).expect("light");
    let pixel = |p: [f32; 3]| {
        let q = r.project(&view(), (W, H), p).expect("in front");
        (q[0] as u32, q[1] as u32)
    };
    let (behind, beside) = (pixel([1.4, 0.0, 0.0]), pixel([1.4, 0.0, 1.6]));
    let at = |px: &[u8], (x, y): (u32, u32)| region(px, x - 1, y - 1, x + 2, y + 2);
    let open = render(&mut device, &mut r, 0.0, "clear");
    let mut frame = |shadow: f32| {
        r.set_light(light, &LightDesc { shadow, ..spot })
            .expect("set");
        render(&mut device, &mut r, 0.0, "clear")
    };
    let full = frame(1.0);
    let half = frame(0.5);
    let none = frame(0.0);
    eprintln!(
        "behind the cube: {:.1} open, {:.1} half, {:.1} full shadow; beside it {:.1} / {:.1}",
        at(&open, behind),
        at(&half, behind),
        at(&full, behind),
        at(&open, beside),
        at(&full, beside),
    );
    assert!(
        at(&open, behind) > at(&full, behind) + 15.0,
        "the cube's shadow"
    );
    let mid = at(&half, behind);
    assert!(
        mid < at(&open, behind) - 4.0 && mid > at(&full, behind) + 4.0,
        "half a shadow"
    );
    assert!(
        (at(&open, beside) - at(&full, beside)).abs() < 3.0,
        "lit beside it"
    );
    assert!(difference(&none, &open) < 0.5, "none cast");
    // Two lights cast at most; a third waits until one stops.
    let shadowed = LightDesc {
        shadow: 1.0,
        ..spot
    };
    r.set_light(light, &shadowed).expect("first");
    let second = r.create_light(&shadowed).expect("second");
    assert!(r.create_light(&shadowed).is_err(), "a third");
    let third = r.create_light(&spot).expect("unshadowed");
    assert!(r.set_light(third, &shadowed).is_err(), "made a third");
    r.destroy_light(second);
    r.set_light(third, &shadowed).expect("now there is room");
    render(&mut device, &mut r, 0.0, "clear");
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

#[test]
#[ignore = "requires a Vulkan driver"]
fn rain_stops_under_a_roof() {
    // A camera under a wide roof in a storm sees no drops; in the open it does. Each case
    // compares the frame with and without particles.
    let mut device = Device::headless().expect("device");
    let (w, h) = (320, 180);
    let view = View {
        name: "under".into(),
        eye: [0.0, 1.5, 0.0],
        target: [0.0, 1.0, -10.0],
        fov: 60.0,
        ..View::default()
    };
    let mut frame = |roof: bool, particles: bool| {
        let mut meshes = vec![quad(0.0, 40.0, 0)];
        if roof {
            meshes.push(quad(3.0, 40.0, 1));
        }
        let world = World {
            instances: (0..meshes.len() as u32)
                .map(|mesh| Instance {
                    mesh,
                    transform: Transform::IDENTITY,
                })
                .collect(),
            meshes,
            materials: vec![
                material("ground", [0.4, 0.5, 0.3, 1.0], Shading::Standard),
                material("roof", [0.5, 0.3, 0.2, 1.0], Shading::Standard),
            ],
            textures: vec![],
            lights: vec![],
        };
        let mut quality = Quality::realtime();
        quality.shadow_size = 512;
        quality.particles = particles;
        let mut r = Renderer::new(
            &mut device,
            &world,
            Look::default(),
            quality,
            (w, h),
            Output::Still,
        )
        .expect("renderer");
        let time = TimeOfDay {
            hours: 12.0,
            ..TimeOfDay::default()
        };
        let storm = Weather::preset("storm").expect("preset");
        let env = Environment::resolve(&time, &storm, &r.look().palette, 2.0);
        r.render_still(&mut device, &view, &env, 1).expect("render")
    };
    let open = difference(&frame(false, true), &frame(false, false));
    let covered = difference(&frame(true, true), &frame(true, false));
    eprintln!("drops change the open frame by {open:.3}, the covered one by {covered:.4}");
    assert!(open > 0.2, "drops should show in the open: {open}");
    assert!(
        covered < open * 0.05,
        "drops under the roof: {covered} vs {open} in the open"
    );
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

#[test]
#[ignore = "requires a Vulkan driver"]
fn grass_and_surface_detail() {
    // A lawn with a tiled roof slab over part of it, seen from just above the grass.
    let mut device = Device::headless().expect("device");
    let (w, h) = (192, 108);
    let roof = Mesh {
        positions: vec![
            Vec3(-3.0, 2.0, -9.0),
            Vec3(3.0, 2.0, -9.0),
            Vec3(3.0, 3.5, -12.0),
            Vec3(-3.0, 3.5, -12.0),
        ],
        normals: vec![Vec3(0.0, 0.894, 0.447); 4],
        uvs: vec![[0.0, 0.0]; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
        color: [0.5; 3],
        material: Some(1),
    };
    let meshes = vec![quad(0.0, 30.0, 0), roof];
    let world = World {
        instances: (0..meshes.len() as u32)
            .map(|mesh| Instance {
                mesh,
                transform: Transform::IDENTITY,
            })
            .collect(),
        meshes,
        materials: vec![
            material("lawn", [0.2, 0.3, 0.1, 1.0], Shading::Standard),
            material("roof-tile", [0.1, 0.15, 0.3, 1.0], Shading::Standard),
        ],
        textures: vec![],
        lights: vec![],
    };
    let view = View {
        name: "meadow".into(),
        eye: [0.0, 1.2, 0.0],
        target: [0.0, 1.0, -10.0],
        fov: 60.0,
        ..View::default()
    };
    let rule = |pattern: &str, preset| DetailRule {
        patterns: vec![pattern.into()],
        preset,
        scale: [0.3, 0.34],
        strength: 1.0,
        variation: 0.1,
        color: None,
        gloss: 0.0,
    };
    let mut frame = |grass: f32, details: bool| {
        let mut look = Look::default();
        look.grass.density = grass;
        if details {
            look.details = vec![
                rule("lawn", DetailPreset::Grass),
                rule("roof-*", DetailPreset::RoofTiles),
            ];
        }
        let mut quality = Quality::realtime();
        quality.shadow_size = 512;
        quality.ground_size = 256;
        let mut r = Renderer::new(&mut device, &world, look, quality, (w, h), Output::Still)
            .expect("renderer");
        let time = TimeOfDay {
            hours: 12.0,
            ..TimeOfDay::default()
        };
        let env = Environment::resolve(&time, &Weather::default(), &r.look().palette, 2.0);
        r.render_still(&mut device, &view, &env, 1).expect("render")
    };
    let plain = frame(0.0, false);
    let detailed = frame(0.0, true);
    let grassy = frame(1.0, true);
    // Roof tiles and grassy ground change the picture; grass blades change it again.
    let detail = difference(&plain, &detailed);
    let blades = difference(&detailed, &grassy);
    eprintln!("detail changes the frame by {detail:.2}, grass blades by {blades:.2}");
    assert!(detail > 1.0, "surface detail had no effect: {detail}");
    assert!(blades > 1.0, "no grass blades: {blades}");
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

#[test]
#[ignore = "requires a Vulkan driver"]
fn colour_rules_recolour_any_material() {
    // A colour-only rule (preset `none`) recolours foliage, which takes no surface detail.
    let mut device = Device::headless().expect("device");
    let (w, h) = (96, 54);
    let meshes = vec![quad(0.0, 30.0, 0), cube([0.0, 1.5, -6.0], 1.5, 1)];
    let world = World {
        instances: (0..meshes.len() as u32)
            .map(|mesh| Instance {
                mesh,
                transform: Transform::IDENTITY,
            })
            .collect(),
        meshes,
        materials: vec![
            material("ground", [0.3, 0.3, 0.3, 1.0], Shading::Standard),
            material("leaf-1", [0.1, 0.3, 0.25, 1.0], Shading::Foliage),
        ],
        textures: vec![],
        lights: vec![],
    };
    let view = View {
        name: "tree".into(),
        eye: [0.0, 2.0, 2.0],
        target: [0.0, 1.5, -6.0],
        fov: 50.0,
        ..View::default()
    };
    let mut frame = |rules: Vec<DetailRule>| {
        let look = Look {
            details: rules,
            ..Look::default()
        };
        let mut quality = Quality::realtime();
        quality.shadow_size = 512;
        quality.ground_size = 256;
        quality.world_size = 256;
        let mut r = Renderer::new(&mut device, &world, look, quality, (w, h), Output::Still)
            .expect("renderer");
        let env = Environment::resolve(
            &TimeOfDay::default(),
            &Weather::default(),
            &r.look().palette,
            2.0,
        );
        r.render_still(&mut device, &view, &env, 1).expect("render")
    };
    let red = |image: &[u8]| {
        image
            .chunks(4)
            .filter(|p| u32::from(p[0]) > 2 * u32::from(p[1]) + 20)
            .count()
    };
    let plain = frame(vec![]);
    let recoloured = frame(vec![DetailRule {
        patterns: vec!["leaf-*".into()],
        preset: DetailPreset::None,
        scale: [1.0, 1.0],
        strength: 1.0,
        variation: 0.0,
        color: Some([0.8, 0.05, 0.05]),
        gloss: 0.0,
    }]);
    let (before, after) = (red(&plain), red(&recoloured));
    eprintln!("red pixels: {before} before, {after} with the rule");
    assert!(
        before == 0 && after > 300,
        "foliage not recoloured: {before} / {after}"
    );
}

#[test]
#[ignore = "requires a Vulkan driver"]
fn lattice_bars_shade_a_lamp() {
    // A glowing paper box; a lattice rule draws dark bars across it, glow included.
    let mut device = Device::headless().expect("device");
    let (w, h) = (96, 54);
    let mut paper = material("paper-lamp", [0.9, 0.85, 0.7, 1.0], Shading::Standard);
    paper.emissive = [2.0, 1.6, 1.0];
    let meshes = vec![quad(0.0, 30.0, 0), cube([0.0, 1.0, -4.0], 0.8, 1)];
    let world = World {
        instances: (0..meshes.len() as u32)
            .map(|mesh| Instance {
                mesh,
                transform: Transform::IDENTITY,
            })
            .collect(),
        meshes,
        materials: vec![
            material("ground", [0.2, 0.2, 0.2, 1.0], Shading::Standard),
            paper,
        ],
        textures: vec![],
        lights: vec![],
    };
    let view = View {
        name: "lamp".into(),
        eye: [0.0, 1.2, -1.0],
        target: [0.0, 1.0, -4.0],
        fov: 50.0,
        ..View::default()
    };
    let mut frame = |rules: Vec<DetailRule>| {
        let look = Look {
            details: rules,
            ..Look::default()
        };
        let mut quality = Quality::realtime();
        quality.shadow_size = 512;
        quality.ground_size = 256;
        quality.world_size = 256;
        let mut r = Renderer::new(&mut device, &world, look, quality, (w, h), Output::Still)
            .expect("renderer");
        let time = TimeOfDay {
            hours: 22.0,
            ..TimeOfDay::default()
        };
        let env = Environment::resolve(&time, &Weather::default(), &r.look().palette, 2.0);
        r.render_still(&mut device, &view, &env, 1).expect("render")
    };
    let bright = |image: &[u8]| image.chunks(4).filter(|p| p[0] > 200 && p[1] > 160).count();
    let plain = frame(vec![]);
    let barred = frame(vec![DetailRule {
        patterns: vec!["paper-*".into()],
        preset: DetailPreset::Lattice,
        scale: [0.2, 0.2],
        strength: 0.25,
        variation: 0.0,
        color: None,
        gloss: 0.0,
    }]);
    let ribbed = frame(vec![DetailRule {
        patterns: vec!["paper-*".into()],
        preset: DetailPreset::Lattice,
        scale: [0.0, 0.2],
        strength: 0.25,
        variation: 0.0,
        color: None,
        gloss: 0.0,
    }]);
    let (before, after, ribs) = (bright(&plain), bright(&barred), bright(&ribbed));
    eprintln!("bright pixels: {before} plain, {after} with a lattice, {ribs} with ribs only");
    assert!(
        before > 500 && after < before * 9 / 10,
        "no bars: {before} / {after}"
    );
    // Ribs alone (no bars across) cover less than the full lattice, but still some.
    assert!(
        ribs > after && ribs < before,
        "ribs: {before} / {ribs} / {after}"
    );
}

#[test]
#[ignore = "requires a Vulkan driver"]
fn tide_marks_stain_what_stands_in_water() {
    // A pale block standing in a sea at 0 m: with tide marks on, its foot is darker.
    let mut device = Device::headless().expect("device");
    let (w, h) = (96, 54);
    let meshes = vec![quad(0.0, 200.0, 0), cube([0.0, 0.5, -4.0], 1.0, 1)];
    let world = World {
        instances: (0..meshes.len() as u32)
            .map(|mesh| Instance {
                mesh,
                transform: Transform::IDENTITY,
            })
            .collect(),
        meshes,
        materials: vec![
            material("sea", [0.1, 0.2, 0.3, 1.0], Shading::Water),
            material("pier", [0.8, 0.8, 0.8, 1.0], Shading::Standard),
        ],
        textures: vec![],
        lights: vec![],
    };
    let view = View {
        name: "pier".into(),
        eye: [0.0, 0.8, -1.0],
        target: [0.0, 0.4, -4.0],
        fov: 50.0,
        ..View::default()
    };
    let mut frame = |tide: f32| {
        let look = Look {
            tide,
            tide_color: [0.2, 0.25, 0.15],
            ..Look::default()
        };
        let mut quality = Quality::realtime();
        quality.shadow_size = 512;
        quality.ground_size = 256;
        quality.world_size = 256;
        let mut r = Renderer::new(&mut device, &world, look, quality, (w, h), Output::Still)
            .expect("renderer");
        let time = TimeOfDay {
            hours: 12.0,
            ..TimeOfDay::default()
        };
        let env = Environment::resolve(&time, &Weather::default(), &r.look().palette, 2.0);
        r.render_still(&mut device, &view, &env, 1).expect("render")
    };
    let luma = |image: &[u8]| image.chunks(4).map(|p| u64::from(p[1])).sum::<u64>();
    let (dry, marked) = (luma(&frame(0.0)), luma(&frame(0.8)));
    eprintln!("green sum: {dry} without tide marks, {marked} with");
    assert!(marked * 100 < dry * 97, "no tide marks: {dry} / {marked}");
}

#[test]
#[ignore = "requires a Vulkan driver"]
fn terrain_beaches_and_distant_shallows() {
    // Terrain turns to sand above the water line, and 200 m away, well past the ground map,
    // the world map still tints the shallows (red here, so they can be counted).
    let mut device = Device::headless().expect("device");
    let (w, h) = (256, 144);
    let world = shore();
    let rule = |preset| DetailRule {
        patterns: vec!["land".into()],
        preset,
        scale: [0.3, 0.3],
        strength: 1.0,
        variation: 0.1,
        color: None,
        gloss: 0.0,
    };
    let mut frame = |preset, view: &View, shallow| {
        let look = Look {
            details: vec![rule(preset)],
            fog: 0.0,
            water_reflection: 0.0,
            water_shallow: shallow,
            terrain: Terrain {
                sand: [0.2, 0.2, 0.9],
                wet_sand: [0.1, 0.1, 0.6],
                ..Terrain::default()
            },
            ..Look::default()
        };
        let mut quality = Quality::realtime();
        quality.shadow_size = 512;
        quality.ground_size = 256;
        quality.world_size = 512;
        let mut r = Renderer::new(&mut device, &world, look, quality, (w, h), Output::Still)
            .expect("renderer");
        let time = TimeOfDay {
            hours: 12.0,
            ..TimeOfDay::default()
        };
        let env = Environment::resolve(&time, &Weather::default(), &r.look().palette, 2.0);
        r.render_still(&mut device, view, &env, 1).expect("render")
    };
    let count = |image: &[u8], test: &dyn Fn(f32, f32, f32) -> bool| {
        image
            .chunks(4)
            .filter(|p| test(f32::from(p[0]), f32::from(p[1]), f32::from(p[2])))
            .count()
    };
    // Sand (blue here) along the beach, seen from just above it.
    let near = View {
        name: "beach".into(),
        eye: [0.0, 10.0, 10.0],
        target: [0.0, 0.0, -20.0],
        fov: 60.0,
        ..View::default()
    };
    let blue = |r: f32, g: f32, b: f32| b > 1.5 * r + 10.0 && b > 1.5 * g + 10.0;
    let red_shallows = [1.0, 0.0, 0.0];
    let grassy = count(&frame(DetailPreset::Grass, &near, red_shallows), &blue);
    let sandy = count(&frame(DetailPreset::Terrain, &near, red_shallows), &blue);
    // Red shallows along the far shore. Out there the red mixes with the water's blue into a
    // magenta band (sharpening, which fades out by 18 m, no longer saturates it), so count
    // red-leaning pixels.
    let far = View {
        name: "offshore".into(),
        eye: [0.0, 25.0, 190.0],
        target: [0.0, 0.0, 0.0],
        fov: 40.0,
        ..View::default()
    };
    let red = |r: f32, g: f32, b: f32| r > 1.2 * g + 6.0 && r > 1.2 * b + 6.0;
    let shallows = count(&frame(DetailPreset::Terrain, &far, red_shallows), &red);
    let plain = count(
        &frame(DetailPreset::Terrain, &far, Look::default().water_shallow),
        &red,
    );
    eprintln!(
        "sand pixels: {grassy} with grass, {sandy} with terrain; \
         far red pixels: {shallows} with red shallows, {plain} without"
    );
    assert!(sandy > grassy + 1000, "no beach: {grassy} / {sandy}");
    assert!(
        shallows > 40 && plain < 5,
        "no shallows past the ground map: {shallows} / {plain}"
    );
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}
