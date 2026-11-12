use super::*;
use crate::assets::model::{Clip, Mesh, MorphTarget, Node, Skin};

fn node(name: &str, parent: Option<usize>, t: V3) -> Node {
    Node {
        name: name.into(),
        parent,
        children: vec![],
        translation: t,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
        mesh: None,
        skin: None,
        weights: None,
    }
}

fn close(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
}

/// A two-bone arm: root at the origin, elbow 1 m up, with a skinned two-vertex strip and a
/// morph target.
fn arm() -> Model {
    let mut m = Model::default();
    m.nodes.push(node("root", None, [0.0; 3]));
    m.nodes.push(node("elbow", Some(0), [0.0, 1.0, 0.0]));
    m.nodes[0].children.push(1);
    let mut skinned = node("strip", None, [5.0, 0.0, 0.0]);
    skinned.mesh = Some(0);
    skinned.skin = Some(0);
    m.nodes.push(skinned);
    m.roots = vec![0, 2];
    m.meshes.push(Mesh {
        name: "strip".into(),
        primitives: vec![Primitive {
            positions: vec![[0.0, 0.5, 0.0], [0.0, 1.5, 0.0], [0.1, 1.0, 0.0]],
            normals: vec![[1.0, 0.0, 0.0]; 3],
            uvs: vec![],
            indices: vec![0, 1, 2],
            material: None,
            joints: vec![[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
            weights: vec![
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
            ],
            targets: vec![MorphTarget {
                positions: vec![[0.0, 0.0, 1.0]; 3],
                normals: vec![],
            }],
        }],
        weights: vec![0.0],
        target_names: vec!["push".into()],
    });
    // Bind pose: joints at their rest globals.
    let inv = |y: f32| {
        let mut i = math::IDENTITY;
        i[13] = -y;
        i
    };
    m.skins.push(Skin {
        name: "arm".into(),
        joints: vec![0, 1],
        inverse_bind: vec![inv(0.0), inv(1.0)],
    });
    // Bend the elbow 90 degrees about z over one second; push the morph in.
    let half = std::f32::consts::FRAC_PI_4;
    m.clips.push(Clip {
        name: "bend".into(),
        channels: vec![
            Channel {
                node: 1,
                path: Path::Rotation,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 1.0],
                values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, half.sin(), half.cos()],
                width: 4,
            },
            Channel {
                node: 2,
                path: Path::Weights,
                interpolation: Interpolation::Step,
                times: vec![0.0, 0.5],
                values: vec![0.0, 1.0],
                width: 1,
            },
        ],
        duration: 1.0,
    });
    m
}

#[test]
fn a_morph_range_matches_the_whole_mesh_morphed() {
    let mut p = arm().meshes[0].primitives[0].clone();
    // The target moves only the middle vertex now (the others' deltas are zero).
    p.targets = vec![
        MorphTarget {
            positions: vec![[0.0; 3], [0.0, 0.0, 1.0], [0.0; 3]],
            normals: vec![[0.0; 3], [0.0, 1.0, 0.0], [0.0; 3]],
        },
        MorphTarget {
            positions: vec![[0.0; 3]; 3],
            normals: vec![],
        },
    ];
    let moved = morphed_vertices(&p);
    assert_eq!(moved, vec![1]);
    let weights = [0.5, 1.0];
    let (all_pos, all_nrm) = deform(&p, &weights, None);
    let (pos, nrm) = morph_range(&p, &weights, &moved, 1, 1);
    assert!(close(&pos[0], &all_pos[1]), "{pos:?}");
    assert!(close(&nrm[0], &all_nrm[1]), "{nrm:?}");
    // A wider range carries its unmoved vertices as modelled.
    let (pos, _) = morph_range(&p, &weights, &moved, 0, 3);
    assert!(close(&pos.concat(), &all_pos.concat()));
}

#[test]
fn gathering_puts_the_moved_vertices_first_and_deforms_as_before() {
    let m = arm();
    let mut p = m.meshes[0].primitives[0].clone();
    // Two more vertices and a second triangle; the target moves only the last and the middle.
    p.positions.extend([[0.2, 0.5, 0.0], [0.3, 1.5, 0.0]]);
    p.normals.extend([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    p.uvs = vec![[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]];
    p.indices.extend([2, 3, 4]);
    p.joints.extend([[1, 0, 0, 0], [0, 1, 0, 0]]);
    p.weights
        .extend([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]);
    let push = [0.0, 0.0, 1.0];
    p.targets = vec![
        MorphTarget {
            positions: vec![[0.0; 3], push, [0.0; 3], [0.0; 3], push],
            normals: vec![],
        },
        MorphTarget {
            positions: vec![[0.0; 3]; 5],
            normals: vec![[0.0; 3]; 5],
        },
    ];
    assert_eq!(moving_targets(&p), vec![true, false]);
    let before = p.clone();
    gather_morphed(&mut p);
    assert_eq!(morphed_vertices(&p), vec![0, 1]);
    // The moved vertices first in their order, then the rest in theirs.
    let order = [1, 4, 0, 2, 3];
    for (new, &old) in order.iter().enumerate() {
        assert_eq!(p.positions[new], before.positions[old]);
        assert_eq!(p.normals[new], before.normals[old]);
        assert_eq!(p.uvs[new], before.uvs[old]);
        assert_eq!(p.joints[new], before.joints[old]);
        assert_eq!(p.weights[new], before.weights[old]);
        assert_eq!(
            p.targets[0].positions[new],
            before.targets[0].positions[old]
        );
    }
    // Every triangle keeps its corners, so it draws and deforms as before.
    let globals = Pose::rest(&m).globals(&m);
    let joints = joint_matrices(&m, 0, &globals);
    let (was, _) = deform(&before, &[1.0, 0.0], Some(&joints));
    let (now, _) = deform(&p, &[1.0, 0.0], Some(&joints));
    for (a, b) in before.indices.iter().zip(&p.indices) {
        assert!(close(&was[*a as usize], &now[*b as usize]));
    }
    // Gathered once, it stays as it is.
    let gathered = p.clone();
    gather_morphed(&mut p);
    assert_eq!(p, gathered);
}

#[test]
fn only_a_moving_target_s_weight_changes_a_primitive() {
    let moving = [true, false];
    assert!(!morph_changes(&[0.0, 0.0], &[0.0, 0.7], &moving));
    assert!(morph_changes(&[0.0, 0.0], &[0.2, 0.0], &moving));
    assert!(morph_changes(&[0.0], &[0.0, 0.0], &moving));
}

#[test]
fn rest_pose_globals_and_bind_pose_skinning() {
    let m = arm();
    let pose = Pose::rest(&m);
    let g = pose.globals(&m);
    assert!(close(&g[1][12..15], &[0.0, 1.0, 0.0]));
    assert!(close(&g[2][12..15], &[5.0, 0.0, 0.0]));
    // In the bind pose skinning changes nothing (and ignores the mesh node's own offset).
    let joints = joint_matrices(&m, 0, &g);
    let (p, n) = deform(&m.meshes[0].primitives[0], &pose.weights[2], Some(&joints));
    assert!(close(
        &p.concat(),
        &m.meshes[0].primitives[0].positions.concat()
    ));
    assert!(close(&n[0], &[1.0, 0.0, 0.0]));
}

#[test]
fn clips_bend_skin_and_morph() {
    let m = arm();
    let mut pose = Pose::rest(&m);
    pose.apply(&m, 0, 1.0, 1.0);
    let g = pose.globals(&m);
    let joints = joint_matrices(&m, 0, &g);
    let (p, n) = deform(&m.meshes[0].primitives[0], &pose.weights[2], Some(&joints));
    // The tip (0, 1.5) bound to the elbow swings to (-0.5, 1) and the morph pushes z by 1.
    assert!(close(&p[1], &[-0.5, 1.0, 1.0]), "{:?}", p[1]);
    // The root vertex stays (plus the morph).
    assert!(close(&p[0], &[0.0, 0.5, 1.0]), "{:?}", p[0]);
    // A normal bound half to the elbow turns 45 degrees.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    assert!(close(&n[2], &[s, s, 0.0]), "{:?}", n[2]);
    // Half a clip: 45 degrees, step weights already switched at 0.5.
    let mut half = Pose::rest(&m);
    half.apply(&m, 0, 0.5, 1.0);
    assert_eq!(half.weights[2], vec![1.0]);
    let q = half.rotation[1];
    let angle = 2.0 * q[3].acos();
    assert!(
        (angle - std::f32::consts::FRAC_PI_4).abs() < 1e-3,
        "{angle}"
    );
    // Weight 0.5 of the full bend from rest is the same 45 degrees.
    let mut mixed = Pose::rest(&m);
    mixed.apply(&m, 0, 1.0, 0.5);
    assert!(close(&mixed.rotation[1], &half.rotation[1]));
    assert_eq!(mixed.weights[2], vec![0.5]);
    // Times outside the clip clamp; unknown clips and zero weights change nothing.
    let mut late = Pose::rest(&m);
    late.apply(&m, 0, 9.0, 1.0);
    assert!(close(&late.rotation[1], &pose.rotation[1]));
    let rest = Pose::rest(&m);
    let mut same = rest.clone();
    same.apply(&m, 7, 0.5, 1.0);
    same.apply(&m, 0, 0.5, 0.0);
    assert_eq!(same, rest);
}

#[test]
fn cubic_spline_hits_its_keys_and_eases_between() {
    let ch = Channel {
        node: 0,
        path: Path::Translation,
        interpolation: Interpolation::CubicSpline,
        times: vec![0.0, 2.0],
        // (in, value, out) per key; zero tangents: a smoothstep from 0 to 4 in x.
        values: vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        width: 3,
    };
    let mut v = [0.0; 3];
    sample(&ch, 1.0, &mut v);
    assert!(close(&v, &[2.0, 0.0, 0.0]), "{v:?}");
    sample(&ch, 0.5, &mut v);
    assert!(close(&v, &[4.0 * 0.15625, 0.0, 0.0]), "{v:?}");
    sample(&ch, 2.0, &mut v);
    assert!(close(&v, &[4.0, 0.0, 0.0]));
}

#[test]
fn quaternion_helpers() {
    let a = [0.0, 0.0, 0.0, 1.0];
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let b = [0.0, s, 0.0, s]; // 90 degrees about y
    // The shorter arc: -b is the same rotation.
    assert!(close(&slerp(a, b.map(|v| -v), 0.5), &slerp(a, b, 0.5)));
    let m = compose([1.0, 2.0, 3.0], b, [2.0; 3]);
    let p = math::transform_point(&m, [1.0, 0.0, 0.0]);
    assert!(close(&p, &[1.0, 2.0, 1.0]), "{p:?}");
    let r = mul_q(b, b); // 180 degrees about y
    assert!(close(&r, &[0.0, 1.0, 0.0, 0.0]));
}

#[test]
fn hermite_crossfade_and_single_call_pose_stack() {
    assert_eq!(hermite_fade(0.0, 0.1), 0.0);
    assert!(close(&[hermite_fade(0.05, 0.1)], &[0.5]));
    assert_eq!(hermite_fade(0.1, 0.1), 1.0);
    assert_eq!(hermite_fade(0.2, 0.1), 1.0);
    assert_eq!(hermite_fade(0.05, 0.0), 1.0);

    let m = arm();
    let rest = Pose::rest(&m);
    let mut bent = Pose::rest(&m);
    bent.apply(&m, 0, 1.0, 1.0);

    let mut pose = Pose::rest(&m);
    let layers = [ClipLayer {
        clip: 0,
        time: 1.0,
        weight: 1.0,
    }];
    let mask = [1usize];
    let offsets = [BoneOffset::euler(0, 0.0, 0.25, 0.0)];
    let morphs = [MorphOverride {
        node: 2,
        target: 0,
        weight: 0.75,
    }];
    pose_model_clip(
        &mut pose,
        Some(&rest),
        &m,
        &PoseSpec {
            reset_to_rest: true,
            clips: &layers,
            crossfade: Some((0.05, 0.1, Some(&mask))),
            look_at: &[],
            two_bone_ik: &[],
            bone_offsets: &offsets,
            morph_overrides: &morphs,
        },
    );
    // Halfway Hermite cross-fade on elbow (90 deg -> 45 deg), plus 0.25 rad yaw on root and 0.75 morph.
    let angle = 2.0 * pose.rotation[1][3].acos();
    assert!(
        (angle - std::f32::consts::FRAC_PI_4).abs() < 1e-3,
        "{angle}"
    );
    assert!(close(&pose.rotation[0], &quat_from_euler(0.0, 0.25, 0.0)));
    assert!(close(&pose.weights[2], &[0.75]));
    assert_ne!(pose.rotation[1], bent.rotation[1]);
}

#[test]
fn look_at_chain_splits_yaw_and_pitch_and_clamps() {
    let mut m = Model::default();
    m.nodes.push(node("spine", None, [0.0, 1.0, 0.0]));
    m.nodes.push(node("neck", Some(0), [0.0, 0.3, 0.0]));
    m.nodes.push(node("head", Some(1), [0.0, 0.2, 0.0]));
    m.nodes[0].children.push(1);
    m.nodes[1].children.push(2);
    m.roots = vec![0];

    let mut pose = Pose::rest(&m);
    // Target to the character's left (+X) and slightly above the head (+Y, +Z).
    pose.apply_look_at(
        &m,
        &LookAtChain {
            nodes: vec![(1, 0.35), (2, 0.65)],
            target_model: [1.0, 2.0, 1.0],
            forward: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            max_yaw: 1.2,
            max_pitch: 0.9,
            weight: 1.0,
        },
    );
    let wr = pose.world_rotations(&m);
    let head_fwd = rotate_vec(wr[2], [0.0, 0.0, 1.0]);
    let expected = math::normalize(math::sub([1.0, 2.0, 1.0], [0.0, 1.5, 0.0]));
    assert!(
        math::dot(head_fwd, expected) > 0.995,
        "head_fwd={head_fwd:?}, expected={expected:?}"
    );
}

#[test]
fn two_bone_ik_reaches_target_in_pole_plane() {
    let mut m = Model::default();
    m.nodes.push(node("shoulder", None, [0.0, 0.0, 0.0]));
    m.nodes.push(node("elbow", Some(0), [0.0, 1.0, 0.0]));
    m.nodes.push(node("wrist", Some(1), [0.0, 1.0, 0.0]));
    m.nodes[0].children.push(1);
    m.nodes[1].children.push(2);
    m.roots = vec![0];

    let mut pose = Pose::rest(&m);
    let target = [0.6, 1.2, 0.4];
    let pole = [0.0, 1.0, 2.0];
    pose.solve_two_bone_ik(
        &m,
        &TwoBoneIk {
            root: 0,
            mid: 1,
            end: 2,
            target_model: target,
            pole_model: pole,
            weight: 1.0,
        },
    );
    let g = pose.globals(&m);
    let wrist = [g[2][12], g[2][13], g[2][14]];
    let elbow = [g[1][12], g[1][13], g[1][14]];
    assert!(close(&wrist, &target), "wrist={wrist:?}, target={target:?}");
    // Elbow bends toward +Z pole.
    assert!(elbow[2] > 0.2, "elbow={elbow:?}");
}

#[test]
fn blend_spaces_1d_and_2d_produce_exact_convex_pose_combinations() {
    let mut m = Model::default();
    m.nodes.push(node("root", None, [0.0, 0.0, 0.0]));
    m.roots = vec![0];
    for x in [0.0f32, 2.0, 6.0] {
        m.clips.push(Clip {
            name: format!("c_{x}"),
            duration: 1.0,
            channels: vec![Channel {
                node: 0,
                path: Path::Translation,
                interpolation: Interpolation::Step,
                times: vec![0.0],
                values: vec![x, 0.0, 0.0],
                width: 3,
            }],
        });
    }
    let bs1 = [
        BlendSample1D {
            clip: 0,
            position: 0.0,
            rate: 1.0,
        },
        BlendSample1D {
            clip: 1,
            position: 1.0,
            rate: 1.0,
        },
        BlendSample1D {
            clip: 2,
            position: 2.0,
            rate: 1.0,
        },
    ];
    let layers_1d = blend_space_1d_layers(&bs1, 1.5, 0.0);
    let mut pose = Pose::rest(&m);
    pose_model_clip(
        &mut pose,
        None,
        &m,
        &PoseSpec {
            reset_to_rest: true,
            clips: &layers_1d,
            ..PoseSpec::default()
        },
    );
    assert!(close(&pose.translation[0], &[4.0, 0.0, 0.0]));

    // 2D Gradient Band Interpolation: hits exact sample at corners and equal 1/3 blend at centroid
    let bs2 = [
        BlendSample2D {
            clip: 0,
            position: [0.0, 0.0],
            rate: 1.0,
        },
        BlendSample2D {
            clip: 1,
            position: [1.0, 0.0],
            rate: 1.0,
        },
        BlendSample2D {
            clip: 2,
            position: [0.0, 1.0],
            rate: 1.0,
        },
    ];
    let corner_layers = blend_space_2d_layers(&bs2, [1.0, 0.0], 0.0);
    pose_model_clip(
        &mut pose,
        None,
        &m,
        &PoseSpec {
            reset_to_rest: true,
            clips: &corner_layers,
            ..PoseSpec::default()
        },
    );
    assert!(close(&pose.translation[0], &[2.0, 0.0, 0.0]));

    let mid_layers = blend_space_2d_layers(&bs2, [0.5, 0.5], 0.0);
    pose_model_clip(
        &mut pose,
        None,
        &m,
        &PoseSpec {
            reset_to_rest: true,
            clips: &mid_layers,
            ..PoseSpec::default()
        },
    );
    // At [0.5, 0.5], all three samples have symmetric influence 0.5 -> normalized 1/3 each -> (0 + 2 + 6)/3.
    assert!(close(&pose.translation[0], &[8.0 / 3.0, 0.0, 0.0]));
}
