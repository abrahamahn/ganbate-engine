use super::*;
use ganbate_engine::render::{Look, Output, Quality};

/// A run-time model: a static box on a node and a skinned bar bent by a clip; poses move what
/// is drawn, node visibility hides parts, and a node's world transform follows the pose.
#[test]
#[ignore = "requires a Vulkan driver"]
fn models_pose_skin_and_place() {
    use ganbate_engine::anim::Pose;
    use ganbate_engine::assets::model::{
        Channel, Clip, Interpolation, Material as ModelMaterial, Mesh as ModelMesh, Model, Node,
        Path, Primitive, Skin,
    };
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
    let mut r = Renderer::new(
        &mut device,
        &world,
        Look::default(),
        Quality::realtime(),
        (W, H),
        Output::Still,
    )
    .expect("renderer");
    let node = |name: &str, parent: Option<usize>, t: [f32; 3]| Node {
        name: name.into(),
        parent,
        children: vec![],
        translation: t,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
        mesh: None,
        skin: None,
        weights: None,
    };
    let to_prim = |m: &Mesh, joints: Vec<[u16; 4]>, weights: Vec<[f32; 4]>, material| Primitive {
        positions: m.positions.iter().map(|p| [p.0, p.1, p.2]).collect(),
        normals: m.normals.iter().map(|n| [n.0, n.1, n.2]).collect(),
        uvs: m.uvs.clone(),
        indices: m.indices.clone(),
        material: Some(material),
        joints,
        weights,
        targets: vec![],
    };
    // A 0.3 m thick, 2 m tall bar of two blocks: the lower follows the root, the upper the
    // elbow.
    let mut bar_prim = to_prim(&cube([0.0, 0.5, 0.0], 0.5, 0), vec![], vec![], 1);
    let upper = to_prim(&cube([0.0, 1.5, 0.0], 0.5, 0), vec![], vec![], 1);
    let n = bar_prim.positions.len() as u32;
    bar_prim.joints = vec![[0, 0, 0, 0]; bar_prim.positions.len()];
    bar_prim
        .joints
        .extend(vec![[1, 0, 0, 0]; upper.positions.len()]);
    bar_prim.positions.extend(upper.positions);
    bar_prim.normals.extend(upper.normals);
    bar_prim.uvs.extend(upper.uvs);
    bar_prim.indices.extend(upper.indices.iter().map(|i| i + n));
    for p in &mut bar_prim.positions {
        p[0] *= 0.3;
        p[2] *= 0.3;
    }
    bar_prim.weights = vec![[1.0, 0.0, 0.0, 0.0]; bar_prim.positions.len()];
    let mut model = Model {
        nodes: vec![
            node("root", None, [0.0; 3]),
            node("elbow", Some(0), [0.0, 1.0, 0.0]),
            node("bar", None, [0.0; 3]),
            node("box", Some(0), [1.5, 0.3, 0.0]),
            node("tip", Some(1), [0.0, 1.0, 0.0]),
        ],
        ..Model::default()
    };
    model.nodes[0].children = vec![1, 3];
    model.nodes[1].children = vec![4];
    model.nodes[2].mesh = Some(0);
    model.nodes[2].skin = Some(0);
    model.nodes[3].mesh = Some(1);
    model.roots = vec![0, 2];
    model.meshes = vec![
        ModelMesh {
            name: "bar".into(),
            primitives: vec![bar_prim],
            weights: vec![],
            target_names: vec![],
        },
        ModelMesh {
            name: "box".into(),
            primitives: vec![to_prim(&cube([0.0; 3], 0.3, 0), vec![], vec![], 0)],
            weights: vec![],
            target_names: vec![],
        },
    ];
    let unlit = |name: &str, c: [f32; 4]| ModelMaterial {
        name: name.into(),
        base_color: c,
        unlit: true,
        ..ModelMaterial::default()
    };
    model.materials = vec![
        unlit("blue", [0.05, 0.05, 0.9, 1.0]),
        unlit("red", [0.9, 0.05, 0.05, 1.0]),
    ];
    let mut inv = [0.0f32; 16];
    for i in [0, 5, 10, 15] {
        inv[i] = 1.0;
    }
    let mut inv_elbow = inv;
    inv_elbow[13] = -1.0;
    model.skins = vec![Skin {
        name: "arm".into(),
        joints: vec![0, 1],
        inverse_bind: vec![inv, inv_elbow],
    }];
    let s = std::f32::consts::FRAC_1_SQRT_2;
    model.clips = vec![Clip {
        name: "bend".into(),
        channels: vec![Channel {
            node: 1,
            path: Path::Rotation,
            interpolation: Interpolation::Linear,
            times: vec![0.0, 1.0],
            // 90 degrees about z: the upper half points to -x.
            values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, s, s],
            width: 4,
        }],
        duration: 1.0,
    }];
    let id = r.create_model(&mut device, model, &[]).expect("model");
    let mut place = [0.0f32; 16];
    for i in [0, 5, 10, 15] {
        place[i] = 1.0;
    }
    let inst = r
        .create_model_instance(&mut device, id, &place)
        .expect("instance");
    let size = (W, H);
    let colour_at = |img: &[u8], p: [f32; 3], r: &Renderer| {
        let q = r.project(&view(), size, p).expect("in front");
        let i = ((q[1] as u32 * W + q[0] as u32) * 4) as usize;
        [img[i], img[i + 1], img[i + 2]]
    };
    let red = |c: [u8; 3]| u32::from(c[0]) > u32::from(c[2]) + 60;
    let blue = |c: [u8; 3]| u32::from(c[2]) > u32::from(c[0]) + 60;
    let rest = render(&mut device, &mut r, 15.0, "clear");
    assert!(red(colour_at(&rest, [0.0, 1.6, 0.15], &r)), "upright bar");
    assert!(
        blue(colour_at(&rest, [1.5, 0.3, 0.3], &r)),
        "the box on its node"
    );
    // Bend: the upper half now lies along -x at 1 m.
    {
        let (pose, model) = r.model_pose_and_model(inst).expect("pose");
        pose.apply(model, 0, 1.0, 1.0);
    }
    r.update_model_instance(&mut device, inst).expect("update");
    let bent = render(&mut device, &mut r, 15.0, "clear");
    assert!(
        !red(colour_at(&bent, [0.0, 1.8, 0.15], &r)),
        "nothing upright now"
    );
    assert!(
        red(colour_at(&bent, [-0.6, 1.0, 0.15], &r)),
        "bent to the side"
    );
    let tip = r.model_node_world(inst, 4).expect("tip");
    assert!(
        (tip[12] + 1.0).abs() < 1e-4 && (tip[13] - 1.0).abs() < 1e-4,
        "{:?}",
        &tip[12..15]
    );
    // Move the instance, hide the box node, and the model's rest is untouched.
    place[12] = -2.0;
    r.set_model_instance(inst, &place, true).expect("move");
    r.set_model_node_visible(inst, 3, false).expect("hide");
    let moved = render(&mut device, &mut r, 15.0, "clear");
    assert!(
        red(colour_at(&moved, [-2.6, 1.0, 0.15], &r)),
        "moved with its pose"
    );
    assert!(!blue(colour_at(&moved, [-0.5, 0.3, 0.3], &r)), "box hidden");
    let mut want = Pose::rest(r.model(id).expect("model"));
    want.apply(r.model(id).expect("model"), 0, 1.0, 1.0);
    assert_eq!(
        *r.model_pose_mut(inst).expect("pose"),
        want,
        "moving keeps the pose"
    );
    // A second instance starts at rest; destroying the model takes both.
    let second = r
        .create_model_instance(&mut device, id, &math_translate(2.0))
        .expect("second");
    let two = render(&mut device, &mut r, 15.0, "clear");
    assert!(
        red(colour_at(&two, [2.0, 1.6, 0.15], &r)),
        "the second is upright"
    );
    // The first goes, and with it its mesh's skin: the second still bends on its own joints.
    r.destroy_model_instance(&mut device, inst);
    {
        let (pose, model) = r.model_pose_and_model(second).expect("pose");
        pose.apply(model, 0, 1.0, 1.0);
    }
    r.update_model_instance(&mut device, second)
        .expect("update");
    let alone = render(&mut device, &mut r, 15.0, "clear");
    assert!(
        red(colour_at(&alone, [1.4, 1.0, 0.15], &r)),
        "the second bends"
    );
    assert!(
        !red(colour_at(&alone, [-2.6, 1.0, 0.15], &r)),
        "the first is gone"
    );
    r.destroy_model(&mut device, id);
    assert!(r.model_node_world(second, 0).is_err());
    assert_eq!(r.prop_instances(), 0);
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

/// Morph targets on a model node with two primitives: a target stretches two boxes whose top
/// corners lie scattered through their vertices, another lifts a third box in the second
/// primitive only. Each weight moves what it should, whichever changed last.
#[test]
#[ignore = "requires a Vulkan driver"]
fn models_morph_scattered_vertices() {
    use ganbate_engine::anim::morphed_vertices;
    use ganbate_engine::assets::model::{
        Material as ModelMaterial, Mesh as ModelMesh, Model, MorphTarget, Node, Primitive,
    };
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
    let mut r = Renderer::new(
        &mut device,
        &world,
        Look::default(),
        Quality::realtime(),
        (W, H),
        Output::Still,
    )
    .expect("renderer");
    // Boxes 0.6 m across on the ground; a target raises every top corner 1 m.
    let boxes = |xs: &[f32]| {
        let mut p = Primitive {
            material: Some(0),
            ..Primitive::default()
        };
        let mut raise = vec![];
        for &x in xs {
            let m = cube([x, 0.3, 0.0], 0.3, 0);
            let base = p.positions.len() as u32;
            for v in &m.positions {
                p.positions.push([v.0, v.1, v.2]);
                raise.push([0.0, if v.1 > 0.5 { 1.0 } else { 0.0 }, 0.0]);
            }
            p.normals.extend(m.normals.iter().map(|n| [n.0, n.1, n.2]));
            p.uvs.extend(&m.uvs);
            p.indices.extend(m.indices.iter().map(|i| i + base));
        }
        (p, raise)
    };
    let (mut pair, stretch) = boxes(&[-1.6, 0.0]);
    let (mut single, lift) = boxes(&[1.6]);
    let none = |p: &Primitive| MorphTarget {
        positions: vec![[0.0; 3]; p.positions.len()],
        normals: vec![],
    };
    pair.targets = vec![
        MorphTarget {
            positions: stretch,
            normals: vec![],
        },
        none(&pair),
    ];
    single.targets = vec![
        none(&single),
        MorphTarget {
            positions: lift,
            normals: vec![],
        },
    ];
    let mut model = Model {
        nodes: vec![Node {
            name: "boxes".into(),
            parent: None,
            children: vec![],
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            mesh: Some(0),
            skin: None,
            weights: None,
        }],
        roots: vec![0],
        ..Model::default()
    };
    model.meshes = vec![ModelMesh {
        name: "boxes".into(),
        primitives: vec![pair, single],
        weights: vec![0.0, 0.0],
        target_names: vec!["stretch".into(), "lift".into()],
    }];
    model.materials = vec![ModelMaterial {
        name: "red".into(),
        base_color: [0.9, 0.05, 0.05, 1.0],
        unlit: true,
        ..ModelMaterial::default()
    }];
    let id = r.create_model(&mut device, model, &[]).expect("model");
    // The top corners, scattered through both boxes' faces, are gathered first.
    let gathered = &r.model(id).expect("model").meshes[0].primitives[0];
    assert_eq!(
        morphed_vertices(gathered),
        (0..24).collect::<Vec<u32>>(),
        "moved vertices first"
    );
    let mut place = [0.0f32; 16];
    for i in [0, 5, 10, 15] {
        place[i] = 1.0;
    }
    let inst = r
        .create_model_instance(&mut device, id, &place)
        .expect("instance");
    let size = (W, H);
    let red_at = |img: &[u8], p: [f32; 3], r: &Renderer| {
        let q = r.project(&view(), size, p).expect("in front");
        let i = ((q[1] as u32 * W + q[0] as u32) * 4) as usize;
        u32::from(img[i]) > u32::from(img[i + 2]) + 60
    };
    let tall = [[-1.6, 1.2, 0.3], [0.0, 1.2, 0.3], [1.6, 1.2, 0.3]];
    let mut weigh = |r: &mut Renderer, weights: [f32; 2]| {
        r.model_pose_mut(inst).expect("pose").weights[0] = weights.to_vec();
        r.update_model_instance(&mut device, inst).expect("update");
        let img = render(&mut device, r, 15.0, "clear");
        tall.map(|p| red_at(&img, p, r))
    };
    assert_eq!(weigh(&mut r, [0.0, 0.0]), [false; 3], "all at rest");
    assert_eq!(weigh(&mut r, [1.0, 0.0]), [true, true, false], "stretched");
    assert_eq!(weigh(&mut r, [1.0, 1.0]), [true; 3], "and lifted");
    assert_eq!(
        weigh(&mut r, [0.0, 1.0]),
        [false, false, true],
        "lifted only"
    );
    assert_eq!(weigh(&mut r, [0.0, 0.0]), [false; 3], "at rest again");
    r.destroy_model(&mut device, id);
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

fn math_translate(x: f32) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    for i in [0, 5, 10, 15] {
        m[i] = 1.0;
    }
    m[12] = x;
    m
}
