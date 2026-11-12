//! What a frame costs the CPU at a resting card table: a dealer (a skinned, morphed figure of
//! eight primitives on one 70-joint skin) re-posed every frame, about sixty props at rest (chips,
//! cards, seats, the table), three chips sliding, and a spot light casting the props' shadows.
//! Each part of the frame is timed on its own (posing the dealer, a morph upload,
//! `prepare_props`, `track_cached_props` and `commit_prop_shadows`), and so is the presented
//! view's whole `encode`, recording only (the GPU's work is outside it). Digests of what the
//! frames decided (batches, bounds, the cached picture's changes) and of stills of the same
//! frames read back show that two builds draw the same.
//!
//! Ignored by default (it wants an optimised build):
//! `cargo test --release -p ganbate-engine --lib render::bench -- --ignored --nocapture`.
//! Set `GANBATE_BENCH_FRAMES` to a file to write each still's hash there.
use std::time::Instant;

use super::lights::{LightDesc, SpotCone};
use super::math::{self, M4, V3};
use super::props::{
    CacheView, MaterialDesc, MeshData, PropInstance, PropMaterial, PropMesh, ShadowMap,
    commit_prop_shadows, track_cached_props,
};
use super::{
    Environment, Look, ModelId, ModelInstance, Output, Quality, Renderer, TimeOfDay, View, Weather,
};
use crate::Vec3;
use crate::anim;
use crate::assets::model::{
    Material as ModelMaterial, Mesh as ModelMesh, Model, MorphTarget, Node, Primitive, Skin,
};
use crate::gpu::{BufferDesc, BufferUsage, Device, Format, Memory, TextureDesc, TextureUsage};
use crate::world::{AlphaMode, Instance, Material, Mesh, Shading, Transform, World};

const SIZE: (u32, u32) = (256, 144);
const WARMUP: usize = 30;
const FRAMES: usize = 600;
/// Stills drawn and read back to show the frames are the same.
const STILLS: usize = 60;

fn node(name: String, parent: Option<usize>, translation: V3) -> Node {
    Node {
        name,
        parent,
        children: vec![],
        translation,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
        mesh: None,
        skin: None,
        weights: None,
    }
}

/// The dealer's skeleton: 70 joints (spine, face, hair, arms with fingers, legs).
struct Skeleton {
    nodes: Vec<Node>,
    spine: Vec<usize>,
    face: Vec<usize>,
    hair: Vec<usize>,
    /// Per side: shoulder, upper arm, forearm, hand, then a finger's three bones.
    arms: [Vec<usize>; 2],
    legs: [Vec<usize>; 2],
}

fn skeleton() -> Skeleton {
    let mut nodes: Vec<Node> = vec![];
    let mut add = |name: String, parent: Option<usize>, t: V3| {
        nodes.push(node(name, parent, t));
        let id = nodes.len() - 1;
        if let Some(p) = parent {
            nodes[p].children.push(id);
        }
        id
    };
    let hips = add("hips".into(), None, [0.0, 0.95, 0.0]);
    let mut spine = vec![hips];
    for (k, y) in [0.12, 0.12, 0.12, 0.15, 0.1].into_iter().enumerate() {
        let parent = *spine.last().expect("hips");
        spine.push(add(format!("spine{k}"), Some(parent), [0.0, y, 0.0]));
    }
    let (chest, head) = (spine[3], spine[5]);
    let face = (0..12)
        .map(|k| {
            let a = k as f32 * 0.5;
            add(
                format!("face{k}"),
                Some(head),
                [0.04 * a.cos(), 0.03 * a.sin(), 0.08],
            )
        })
        .collect();
    let mut hair = vec![];
    for side in [-1.0f32, 1.0] {
        let mut at = head;
        for k in 0..3 {
            at = add(
                format!("hair{side}{k}"),
                Some(at),
                [side * 0.05, -0.06, -0.06],
            );
            hair.push(at);
        }
    }
    let arms = [-1.0f32, 1.0].map(|s| {
        let shoulder = add(format!("shoulder{s}"), Some(chest), [s * 0.08, 0.1, 0.0]);
        let upper = add(format!("upper{s}"), Some(shoulder), [s * 0.12, 0.0, 0.0]);
        let fore = add(format!("fore{s}"), Some(upper), [s * 0.26, 0.0, 0.0]);
        let hand = add(format!("hand{s}"), Some(fore), [s * 0.24, 0.0, 0.0]);
        let mut arm = vec![shoulder, upper, fore, hand];
        for f in 0..5 {
            let mut at = hand;
            for (k, t) in [
                [s * 0.04, 0.0, -0.03 + 0.015 * f as f32],
                [s * 0.03, 0.0, 0.0],
                [s * 0.025, 0.0, 0.0],
            ]
            .into_iter()
            .enumerate()
            {
                at = add(format!("finger{s}{f}{k}"), Some(at), t);
                arm.push(at);
            }
        }
        arm
    });
    let legs = [-1.0f32, 1.0].map(|s| {
        let thigh = add(format!("thigh{s}"), Some(hips), [s * 0.1, -0.05, 0.0]);
        let shin = add(format!("shin{s}"), Some(thigh), [0.0, -0.42, 0.0]);
        let foot = add(format!("foot{s}"), Some(shin), [0.0, -0.4, 0.0]);
        let toe = add(format!("toe{s}"), Some(foot), [0.0, -0.05, 0.12]);
        vec![thigh, shin, foot, toe]
    });
    Skeleton {
        nodes,
        spine,
        face,
        hair,
        arms,
        legs,
    }
}

/// A tube of `rings` rings of `segs` vertices along the rest positions of `chain`, each vertex
/// weighted between the two joints of its ring's span (the head's front between the head and a
/// face joint).
fn tube(
    rest: &[M4],
    chain: &[usize],
    face: &[usize],
    radius: f32,
    rings: u32,
    segs: u32,
) -> Primitive {
    let at = |j: usize| [rest[j][12], rest[j][13], rest[j][14]];
    let mut p = Primitive::default();
    let spans = (chain.len() - 1).max(1) as f32;
    for r in 0..rings {
        let u = r as f32 / (rings - 1) as f32 * spans;
        let k = (u.floor() as usize).min(chain.len().saturating_sub(2));
        let (a, b) = (chain[k], chain[(k + 1).min(chain.len() - 1)]);
        let w = (u - k as f32).clamp(0.0, 1.0);
        let centre = math::lerp3(at(a), at(b), w);
        for s in 0..segs {
            let t = s as f32 / segs as f32 * std::f32::consts::TAU;
            let n = [t.cos(), 0.0, t.sin()];
            p.positions.push(math::add(centre, math::scale(n, radius)));
            p.normals.push(n);
            p.uvs
                .push([s as f32 / segs as f32, r as f32 / rings as f32]);
            let front = n[2] > 0.3 && !face.is_empty() && b == *chain.last().expect("chain");
            if front {
                let f = face[s as usize % face.len()];
                p.joints.push([b as u16, f as u16, 0, 0]);
                p.weights.push([0.7, 0.3, 0.0, 0.0]);
            } else {
                p.joints.push([a as u16, b as u16, 0, 0]);
                p.weights.push([1.0 - w, w, 0.0, 0.0]);
            }
        }
    }
    for r in 0..rings - 1 {
        for s in 0..segs {
            let a = r * segs + s;
            let b = r * segs + (s + 1) % segs;
            let (c, d) = (a + segs, b + segs);
            p.indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    p
}

/// A morph target moving the vertices of `p` that `moves` picks.
fn target(p: &Primitive, moves: impl Fn(V3, V3) -> bool, by: V3) -> MorphTarget {
    MorphTarget {
        positions: p
            .positions
            .iter()
            .zip(&p.normals)
            .map(|(&v, &n)| if moves(v, n) { by } else { [0.0; 3] })
            .collect(),
        normals: vec![],
    }
}

/// The dealer: one skinned mesh of eight primitives (about 36 000 vertices), the body's with four
/// face targets and a breath.
fn dealer() -> (Model, usize) {
    let Skeleton {
        mut nodes,
        spine,
        face,
        hair,
        arms,
        legs,
    } = skeleton();
    let mut probe = Model {
        nodes: nodes.clone(),
        roots: vec![0],
        ..Model::default()
    };
    let rest = anim::Pose::rest(&probe).globals(&probe);
    let head = spine[5];
    let mut body = tube(&rest, &spine, &face, 0.13, 140, 100);
    let head_y = rest[head][13];
    let chest_y = rest[spine[3]][13];
    let face_target = |dy: f32, side: f32| {
        target(
            &body,
            move |v, n| v[1] > head_y - 0.06 && n[2] > 0.5 && n[0] * side >= 0.0,
            [0.0, dy, 0.004],
        )
    };
    let breath = target(
        &body,
        |v, n| (v[1] - chest_y).abs() < 0.08 && n[2] > 0.0,
        [0.0, 0.0, 0.01],
    );
    body.targets = vec![
        face_target(-0.004, -1.0),
        face_target(-0.004, 1.0),
        face_target(0.006, 0.0),
        face_target(0.002, 0.0),
        breath,
    ];
    body.material = Some(0);
    let mut hair_chain = vec![head];
    hair_chain.extend(&hair[..3]);
    let mut prims = vec![body];
    for (chain, radius, rings, segs, material) in [
        (hair_chain, 0.1, 50, 100, 1),
        (spine[1..5].to_vec(), 0.15, 50, 100, 2),
        (legs[0].clone(), 0.09, 40, 50, 2),
        (legs[1].clone(), 0.09, 40, 50, 2),
        (arms[0][..9].to_vec(), 0.05, 50, 40, 0),
        (arms[1][..9].to_vec(), 0.05, 50, 40, 0),
        (face[9..11].to_vec(), 0.012, 20, 20, 3),
    ] {
        let mut p = tube(&rest, &chain, &[], radius, rings, segs);
        p.material = Some(material);
        prims.push(p);
    }
    let joints: Vec<usize> = (0..nodes.len()).collect();
    let inverse_bind = joints
        .iter()
        .map(|&j| math::invert(&rest[j]).expect("invertible rest pose"))
        .collect();
    let mut figure = node("figure".into(), None, [0.0; 3]);
    figure.mesh = Some(0);
    figure.skin = Some(0);
    nodes.push(figure);
    let figure = nodes.len() - 1;
    probe.nodes = nodes;
    probe.roots = vec![0, figure];
    probe.meshes = vec![ModelMesh {
        name: "dealer".into(),
        weights: vec![0.0; prims[0].targets.len()],
        target_names: vec![],
        primitives: prims,
    }];
    probe.skins = vec![Skin {
        name: "dealer".into(),
        joints,
        inverse_bind,
    }];
    probe.materials = ["skin", "hair", "cloth", "eye"]
        .map(|name| ModelMaterial {
            name: name.into(),
            metallic: 0.0,
            roughness: 0.6,
            ..ModelMaterial::default()
        })
        .to_vec();
    (probe, figure)
}

/// A unit cube's faces (24 vertices), as prop mesh data.
fn cube() -> (Vec<V3>, Vec<V3>, Vec<u32>) {
    let (mut pos, mut nrm, mut idx) = (vec![], vec![], vec![]);
    for axis in 0..3 {
        for side in [-1.0f32, 1.0] {
            let mut n = [0.0; 3];
            n[axis] = side;
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let base = pos.len() as u32;
            for (a, b) in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
                let mut p = [0.0; 3];
                p[axis] = side * 0.5;
                p[u] = a;
                p[v] = b * side;
                pos.push(p);
                nrm.push(n);
            }
            idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    (pos, nrm, idx)
}

/// A unit cylinder of `segs` sides, capped, as prop mesh data.
fn cylinder(segs: u32) -> (Vec<V3>, Vec<V3>, Vec<u32>) {
    let (mut pos, mut nrm, mut idx) = (vec![], vec![], vec![]);
    for s in 0..segs {
        let t = s as f32 / segs as f32 * std::f32::consts::TAU;
        let n = [t.cos(), 0.0, t.sin()];
        pos.extend([[n[0], -0.5, n[2]], [n[0], 0.5, n[2]]]);
        nrm.extend([n, n]);
    }
    for s in 0..segs {
        let (a, b) = (s * 2, (s + 1) % segs * 2);
        idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
    }
    for (y, flip) in [(-0.5f32, true), (0.5, false)] {
        let centre = pos.len() as u32;
        pos.push([0.0, y, 0.0]);
        nrm.push([0.0, y.signum(), 0.0]);
        for s in 0..segs {
            let t = s as f32 / segs as f32 * std::f32::consts::TAU;
            pos.push([t.cos(), y, t.sin()]);
            nrm.push([0.0, y.signum(), 0.0]);
        }
        for s in 0..segs {
            let (a, b) = (centre + 1 + s, centre + 1 + (s + 1) % segs);
            if flip {
                idx.extend_from_slice(&[centre, a, b]);
            } else {
                idx.extend_from_slice(&[centre, b, a]);
            }
        }
    }
    (pos, nrm, idx)
}

fn placed(at: V3, size: V3, turn: f32) -> M4 {
    let (s, c) = turn.sin_cos();
    [
        c * size[0],
        0.0,
        -s * size[0],
        0.0,
        0.0,
        size[1],
        0.0,
        0.0,
        s * size[2],
        0.0,
        c * size[2],
        0.0,
        at[0],
        at[1],
        at[2],
        1.0,
    ]
}

fn quat(axis: V3, angle: f32) -> [f32; 4] {
    let (s, c) = (angle * 0.5).sin_cos();
    let a = math::normalize(axis);
    [a[0] * s, a[1] * s, a[2] * s, c]
}

/// The scene at the table, and what each frame changes in it.
struct Table {
    renderer: Renderer,
    model: ModelId,
    dealer: ModelInstance,
    /// The figure's node (its morph weights) and the joints posed each frame.
    figure: usize,
    posed: Vec<(usize, V3, f32, f32)>,
    sliding: Vec<PropInstance>,
    view: View,
    env: Environment,
}

impl Table {
    fn new(device: &mut Device<'_>, output: Output) -> Self {
        let mut world = World::default();
        world.materials.push(Material {
            name: "floor".into(),
            base_color: [0.4, 0.3, 0.25, 1.0],
            roughness: 0.8,
            alpha: AlphaMode::Opaque,
            shading: Shading::Standard,
            ..Material::default()
        });
        let half = 4.0;
        world.meshes.push(Mesh {
            positions: vec![
                Vec3(-half, 0.0, -half),
                Vec3(-half, 0.0, half),
                Vec3(half, 0.0, half),
                Vec3(half, 0.0, -half),
            ],
            normals: vec![Vec3(0.0, 1.0, 0.0); 4],
            uvs: vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            color: [1.0, 1.0, 1.0],
            indices: vec![0, 1, 2, 0, 2, 3],
            material: Some(0),
        });
        world.instances.push(Instance {
            mesh: 0,
            transform: Transform(math::IDENTITY),
        });
        let look = Look {
            painterly: 0.0,
            ..Look::default()
        };
        let mut r = Renderer::new(
            device,
            &world,
            look.clone(),
            Quality::realtime(),
            SIZE,
            output,
        )
        .expect("renderer");
        r.finish_ao(device).expect("ao");

        let mesh =
            |r: &mut Renderer, device: &mut Device<'_>, (p, n, i): (Vec<V3>, Vec<V3>, Vec<u32>)| {
                r.create_prop_mesh(
                    device,
                    &MeshData {
                        positions: &p,
                        normals: &n,
                        uvs: &[],
                        indices: &i,
                    },
                )
                .expect("mesh")
            };
        let material = |r: &mut Renderer, device: &mut Device<'_>, name: &str, base_color| {
            let desc = MaterialDesc {
                name: name.into(),
                base_color,
                ..MaterialDesc::default()
            };
            r.create_prop_material(device, &desc).expect("material")
        };
        let chip: PropMesh = mesh(&mut r, device, cylinder(24));
        let block: PropMesh = mesh(&mut r, device, cube());
        let felt = material(&mut r, device, "felt", [0.1, 0.4, 0.2, 1.0]);
        let wood = material(&mut r, device, "wood", [0.4, 0.25, 0.15, 1.0]);
        let card_face = material(&mut r, device, "card", [0.95, 0.95, 0.92, 1.0]);
        let card_back = material(&mut r, device, "card back", [0.6, 0.1, 0.15, 1.0]);
        let chips: Vec<PropMaterial> = [
            [0.9, 0.9, 0.9, 1.0],
            [0.8, 0.1, 0.1, 1.0],
            [0.1, 0.5, 0.2, 1.0],
            [0.05, 0.05, 0.05, 1.0],
        ]
        .map(|c| material(&mut r, device, "chip", c))
        .to_vec();
        let mut place = |mesh, material, m: M4| {
            r.create_prop_instance(mesh, material, &m)
                .expect("instance")
        };
        place(block, felt, placed([0.0, 0.76, 0.0], [2.4, 0.08, 1.4], 0.0));
        for k in 0..6 {
            let a = k as f32 / 6.0 * std::f32::consts::TAU;
            let at = [1.5 * a.cos(), 0.45, 0.95 * a.sin()];
            place(block, wood, placed(at, [0.45, 0.9, 0.45], a));
        }
        // Six stacks of eight chips before the seats.
        for k in 0..48 {
            let (seat, level) = (k / 8, k % 8);
            let a = seat as f32 / 6.0 * std::f32::consts::TAU;
            let at = [
                0.85 * a.cos(),
                0.802 + level as f32 * 0.0042,
                0.55 * a.sin(),
            ];
            place(chip, chips[k % 4], placed(at, [0.02, 0.004, 0.02], 0.0));
        }
        // Five board cards and three dealt.
        for k in 0..8 {
            let at = [
                -0.3 + k as f32 * 0.08,
                0.8005,
                if k < 5 { 0.0 } else { 0.3 },
            ];
            let material = if k < 5 { card_face } else { card_back };
            place(
                block,
                material,
                placed(at, [0.063, 0.0006, 0.088], 0.1 * k as f32),
            );
        }
        let sliding = (0..3)
            .map(|k| {
                place(
                    chip,
                    chips[k],
                    placed([0.2 * k as f32, 0.802, 0.2], [0.02, 0.004, 0.02], 0.0),
                )
            })
            .collect();

        let (model, figure) = dealer();
        let posed = posed_joints(&model);
        let id = r.create_model(device, model, &[]).expect("model");
        let dealer = r
            .create_model_instance(device, id, &placed([0.0, 0.0, -1.1], [1.0; 3], 0.0))
            .expect("dealer");
        r.create_light(&LightDesc {
            position: [0.0, 2.6, 0.0],
            range: 6.0,
            color: [1.0, 0.92, 0.8],
            intensity: 8.0,
            spot: Some(SpotCone {
                direction: [0.0, -1.0, 0.0],
                inner: 0.5,
                outer: 0.8,
            }),
            follows_lamps: false,
            shadow: 1.0,
        })
        .expect("spot");
        let view = View {
            name: "seat".into(),
            eye: [0.0, 1.55, 1.9],
            target: [0.0, 0.85, -0.3],
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
        Self {
            renderer: r,
            model: id,
            dealer,
            figure,
            posed,
            sliding,
            view,
            env,
        }
    }

    /// Frame `f`'s idle: the dealer breathes, sways, blinks and talks; the chips slide on.
    fn pose(&mut self, device: &mut Device<'_>, f: usize) {
        let t = f as f32 / 60.0;
        let r = &mut self.renderer;
        let (pose, _) = r.model_pose_and_model(self.dealer).expect("pose");
        for &(node, axis, amount, rate) in &self.posed {
            pose.rotation[node] = quat(axis, amount * (t * rate).sin());
        }
        let blink = if f % 180 < 8 { 1.0 } else { 0.0 };
        let w = &mut pose.weights[self.figure];
        w[0] = blink;
        w[1] = blink;
        w[2] = 0.3 + 0.3 * (t * 3.1).sin();
        w[3] = 0.5 + 0.5 * (t * 0.7).sin();
        w[4] = 0.5 + 0.5 * (t * 1.4).sin();
        r.update_model_instance(device, self.dealer)
            .expect("update");
        for (k, &chip) in self.sliding.iter().enumerate() {
            let at = [0.2 * k as f32 - 0.1 + 0.002 * f as f32 % 0.3, 0.802, 0.2];
            r.set_prop_instance(chip, &placed(at, [0.02, 0.004, 0.02], 0.0), true)
                .expect("chip");
        }
    }
}

/// The joints an idle moves: the spine, head, shoulders and arms, the hair and a few fingers.
fn posed_joints(model: &Model) -> Vec<(usize, V3, f32, f32)> {
    let wanted = [
        "spine0",
        "spine1",
        "spine2",
        "spine3",
        "spine4",
        "shoulder-1",
        "shoulder1",
        "upper-1",
        "upper1",
        "fore-1",
        "fore1",
        "hand-1",
        "hand1",
        "hair-10",
        "hair-11",
        "hair10",
        "hair11",
        "finger-100",
        "finger100",
        "face0",
        "face3",
    ];
    wanted
        .iter()
        .enumerate()
        .filter_map(|(k, name)| {
            let node = model.nodes.iter().position(|n| n.name == *name)?;
            let axis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]][k % 3];
            Some((
                node,
                axis,
                0.02 + 0.01 * (k % 4) as f32,
                0.8 + 0.37 * k as f32,
            ))
        })
        .collect()
}

/// The tenth percentile (what a frame costs without interruptions, on a busy machine) and the
/// median of samples in microseconds.
fn summary(name: &str, samples: &mut [f64]) -> String {
    samples.sort_by(f64::total_cmp);
    let p10 = samples[samples.len() / 10];
    let median = samples[samples.len() / 2];
    format!("{name}: p10 {p10:.1} us, median {median:.1} us")
}

fn micros(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e6
}

/// The presented view's encode, recording only: llvmpipe crashes re-shading the cached picture
/// (`fs_reshade_mask`), so these frames are recorded and dropped, and the frames read back are
/// stills ([`stills`]). Returns the timings and a digest of what each frame decided: its batches
/// and the props' bounds, and its statistics.
fn encodes(device: &mut Device<'_>, lines: &mut Vec<String>) -> u64 {
    let mut table = Table::new(device, Output::Surface(Format::Rgba8Unorm));
    let usage = TextureUsage::COLOR_TARGET | TextureUsage::COPY_SRC;
    let target = device
        .create_texture(&TextureDesc::new(SIZE.0, SIZE.1, Format::Rgba8Unorm, usage))
        .expect("target");
    let (mut encode, mut posing) = (vec![], vec![]);
    let mut digest = super::renderer::Fnv::new();
    let mut enc = device.encoder();
    for f in 0..WARMUP + FRAMES {
        let t = Instant::now();
        table.pose(device, f);
        let pose_us = micros(t);
        enc.clear();
        let r = &mut table.renderer;
        let t = Instant::now();
        r.encode(
            device,
            &mut enc,
            &table.view,
            &table.env,
            [0.0; 2],
            0,
            Some(target),
        )
        .expect("encode");
        let encode_us = micros(t);
        let s = r.last_frame_stats();
        digest.write(&r.props.digest().to_le_bytes());
        digest.write(format!("{s:?}").as_bytes());
        if f >= WARMUP {
            encode.push(encode_us);
            posing.push(pose_us);
        }
    }
    lines.push(summary("encode (whole, CPU)", &mut encode));
    lines.push(summary("update_model_instance + chips", &mut posing));
    table.renderer.destroy_props(device);
    digest.finish()
}

/// `prepare_props` alone, then `track_cached_props` and (when casters moved)
/// `commit_prop_shadows` after it, as `encode` calls them. Returns a digest of the batches and
/// of what the cached picture follows: regions, moved and settled bounds, casters moved.
fn prop_steps(device: &mut Device<'_>, lines: &mut Vec<String>) -> u64 {
    let mut table = Table::new(device, Output::Surface(Format::Rgba8Unorm));
    let eye = table.view.eye;
    let view_proj = {
        let aspect = SIZE.0 as f32 / SIZE.1 as f32;
        let p = math::perspective(table.view.fov.to_radians(), aspect, 0.05);
        math::mul(&p, &math::look_at(eye, table.view.target, [0.0, 1.0, 0.0]))
    };
    let (mut prepare, mut track, mut commit) = (vec![], vec![], vec![]);
    let mut digest = super::renderer::Fnv::new();
    for f in 0..WARMUP + FRAMES {
        table.pose(device, f);
        let r = &mut table.renderer;
        let t = Instant::now();
        r.prepare_props(device, f % 2, eye).expect("prepare");
        let prepare_us = micros(t);
        let maps: Vec<ShadowMap> = r
            .game_lights
            .shadow_views()
            .iter()
            .map(|m| (*m, r.quality.light_shadow_size.max(16)))
            .collect();
        let view = CacheView {
            view_proj: &view_proj,
            size: SIZE,
        };
        let t = Instant::now();
        let changes = track_cached_props(&mut r.props, f == 0, &view, &maps, 0.0);
        let track_us = micros(t);
        let t = Instant::now();
        if changes.casters_moved || f == 0 {
            commit_prop_shadows(&mut r.props);
        }
        let commit_us = micros(t);
        digest.write(&r.props.digest().to_le_bytes());
        digest.write(
            format!(
                "{:?} {:?} {:?} {}",
                changes.regions, changes.moved, changes.settled, changes.casters_moved
            )
            .as_bytes(),
        );
        if f >= WARMUP {
            prepare.push(prepare_us);
            track.push(track_us);
            commit.push(commit_us);
        }
    }
    lines.push(summary("prepare_props", &mut prepare));
    lines.push(summary("track_cached_props", &mut track));
    lines.push(summary("commit_prop_shadows", &mut commit));
    table.renderer.destroy_props(device);
    digest.finish()
}

/// The body's morph upload: its face and breath targets re-weighted every frame, against the
/// device write it makes (on Vulkan a submission of its own, waited for).
fn morph_upload(device: &mut Device<'_>, lines: &mut Vec<String>) {
    let mut table = Table::new(device, Output::Surface(Format::Rgba8Unorm));
    let r = &mut table.renderer;
    let p = r.model(table.model).expect("model").meshes[0].primitives[0].clone();
    let body = r
        .create_prop_mesh(
            device,
            &MeshData {
                positions: &p.positions,
                normals: &p.normals,
                uvs: &p.uvs,
                indices: &p.indices,
            },
        )
        .expect("body");
    r.set_prop_mesh_skin(body, &p.joints, &p.weights)
        .expect("skin");
    let scratch = device
        .create_buffer(&BufferDesc {
            size: p.positions.len() as u64 * 32,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            memory: Memory::Device,
        })
        .expect("scratch");
    let moved = anim::morphed_vertices(&p);
    let (first, count) = (
        moved[0] as usize,
        (moved[moved.len() - 1] - moved[0]) as usize + 1,
    );
    let (mut morph, mut write) = (vec![], vec![]);
    for f in 0..WARMUP + FRAMES {
        let t = f as f32 / 60.0;
        let weights = [
            0.0,
            0.0,
            0.3 + 0.3 * (t * 3.1).sin(),
            0.5 + 0.5 * (t * 0.7).sin(),
            0.5 + 0.5 * (t * 1.4).sin(),
        ];
        let (pos, nrm) = anim::morph_range(&p, &weights, &moved, first, count);
        let t = Instant::now();
        r.update_prop_mesh_range(device, body, first, &pos, &nrm)
            .expect("morph");
        let morph_us = micros(t);
        let bytes = vec![0u8; count * 32];
        let t = Instant::now();
        device
            .write_buffer(scratch, first as u64 * 32, &bytes)
            .expect("write");
        if f >= WARMUP {
            morph.push(morph_us);
            write.push(micros(t));
        }
    }
    device.destroy_buffer(scratch);
    let n = p.positions.len();
    lines.push(summary(
        &format!("morph upload ({count} of {n} vertices)"),
        &mut morph,
    ));
    lines.push(summary("  of which the device write alone", &mut write));
    table.renderer.destroy_props(device);
}

/// Stills of the same frames, read back: each frame's pixels hashed (into
/// `GANBATE_BENCH_FRAMES` when set) and all of them into one digest.
fn stills(device: &mut Device<'_>) -> u64 {
    let mut table = Table::new(device, Output::Still);
    let mut digest = super::renderer::Fnv::new();
    let mut frames = String::new();
    for f in 0..STILLS {
        table.pose(device, f);
        let pixels = table
            .renderer
            .render_still(device, &table.view, &table.env, 1)
            .expect("still");
        let mut h = super::renderer::Fnv::new();
        h.write(&pixels);
        let line = format!(
            "{f} {:016x} {:016x}\n",
            h.finish(),
            table.renderer.props.digest()
        );
        digest.write(line.as_bytes());
        frames.push_str(&line);
    }
    if let Ok(path) = std::env::var("GANBATE_BENCH_FRAMES") {
        std::fs::write(&path, &frames).expect("write frames");
    }
    table.renderer.destroy_props(device);
    digest.finish()
}

#[test]
#[ignore = "a benchmark: run it in release with --ignored --nocapture"]
fn a_frame_at_the_table() {
    let Ok(mut device) = Device::headless() else {
        eprintln!("bench: no headless device");
        return;
    };
    let device = &mut device;
    let mut lines = vec![];
    let encoded = encodes(device, &mut lines);
    let stepped = prop_steps(device, &mut lines);
    morph_upload(device, &mut lines);
    let drawn = stills(device);
    lines.push(format!(
        "digests: encode {encoded:016x}, prop steps {stepped:016x}, stills {drawn:016x}"
    ));
    for l in &lines {
        println!("bench: {l}");
    }
}
