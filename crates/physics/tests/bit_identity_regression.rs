//! Bit-identical simulation regression suite (`state_hash()`, events, queries, character).

use std::sync::Arc;

use ganbate_physics::hull::ConvexHull;
use ganbate_physics::mesh::{Heightfield, TriangleMesh};
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, Character, CharacterSettings, ColliderDesc, DistanceOptions,
    HingeOptions, Iso, JointDesc, JointKind, Motor, PhysicsWorld, Quat, Settings, Shape,
    SliderOptions, Spring, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn build_complex_world() -> (PhysicsWorld, BodyId, BodyId, BodyId, BodyId) {
    let mut w = PhysicsWorld::new(Settings::default());

    // 1. Static ground + sensor zone + mesh ramp + heightfield hills.
    let ground = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        ground,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(60.0, 0.5, 60.0),
        }),
    )
    .unwrap();

    let mut sensor_desc = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::new(4.0, 3.0, 4.0),
    });
    sensor_desc.local = Iso::from_position(Vec3::new(0.0, 3.5, 0.0));
    sensor_desc.sensor = true;
    w.add_collider(ground, &sensor_desc).unwrap();

    // Ramp mesh.
    let ramp_verts = vec![
        Vec3::new(6.0, 0.0, -3.0),
        Vec3::new(12.0, 2.5, -3.0),
        Vec3::new(12.0, 2.5, 3.0),
        Vec3::new(6.0, 0.0, 3.0),
        Vec3::new(12.0, 0.0, -3.0),
        Vec3::new(12.0, 0.0, 3.0),
    ];
    let ramp_tris = vec![[0, 2, 1], [0, 3, 2], [1, 2, 5], [1, 5, 4]];
    let ramp_mesh = TriangleMesh::new(ramp_verts, ramp_tris).unwrap();
    let mut ramp_col = ColliderDesc::new(Shape::Mesh(Arc::new(ramp_mesh)));
    ramp_col.local = Iso::from_position(Vec3::new(0.0, 0.5, 0.0));
    w.add_collider(ground, &ramp_col).unwrap();

    // Heightfield patch.
    let mut heights = Vec::with_capacity(9 * 9);
    for r in 0..9 {
        for c in 0..9 {
            let dx = c as f32 - 4.0;
            let dz = r as f32 - 4.0;
            heights.push(0.15 * ((dx * dx + dz * dz) * 0.2));
        }
    }
    let hf = Heightfield::new(9, 9, heights, Vec3::new(1.0, 1.0, 1.0)).unwrap();
    let hf_body = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(-18.0, 0.0, -4.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        hf_body,
        &ColliderDesc::new(Shape::Heightfield(Arc::new(hf))),
    )
    .unwrap();

    // 2. Convex hulls: sheared prism & octahedron.
    let prism_hull = Arc::new(
        ConvexHull::from_points(&[
            Vec3::new(-0.4, -0.3, -0.4),
            Vec3::new(0.4, -0.3, -0.4),
            Vec3::new(0.4, -0.3, 0.4),
            Vec3::new(-0.4, -0.3, 0.4),
            Vec3::new(-0.25, 0.35, -0.3),
            Vec3::new(0.45, 0.35, -0.3),
            Vec3::new(0.45, 0.35, 0.35),
            Vec3::new(-0.25, 0.35, 0.35),
        ])
        .unwrap(),
    );
    let oct_hull = Arc::new(
        ConvexHull::from_points(&[
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.6, 0.0),
            Vec3::new(0.0, -0.6, 0.0),
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::new(0.0, 0.0, -0.5),
        ])
        .unwrap(),
    );

    // 3. Settling distant island (sleeps quickly and stays asleep until woken).
    let mut sleeper_top = ground;
    for i in 0..4 {
        let b = w
            .create_body(&BodyDesc {
                kind: BodyKind::Dynamic,
                position: Vec3::new(30.0, 0.4 + i as f32 * 0.82, 30.0),
                ..Default::default()
            })
            .unwrap();
        w.add_collider(
            b,
            &ColliderDesc::new(Shape::Box {
                half_extents: Vec3::splat(0.4),
            }),
        )
        .unwrap();
        sleeper_top = b;
    }

    // 4. Active central stack of mixed shapes + locked axes.
    for i in 0..8 {
        let lock_rot = match i % 4 {
            1 => [false, true, false],
            2 => [true, false, true],
            3 => [true, true, true],
            _ => [false, false, false],
        };
        let b = w
            .create_body(&BodyDesc {
                kind: BodyKind::Dynamic,
                position: Vec3::new(
                    (i % 2) as f32 * 0.12,
                    0.6 + i as f32 * 0.95,
                    (i / 2) as f32 * 0.08 - 0.1,
                ),
                rotation: Quat::from_axis_angle(Vec3::new(0.2, 0.9, 0.3).normalize_or_zero(), 0.25),
                lock_rotation: lock_rot,
                ..Default::default()
            })
            .unwrap();
        let shape = match i % 5 {
            0 => Shape::Box {
                half_extents: Vec3::new(0.45, 0.35, 0.4),
            },
            1 => Shape::Sphere { radius: 0.4 },
            2 => Shape::Capsule {
                half_height: 0.25,
                radius: 0.25,
            },
            3 => Shape::Hull(prism_hull.clone()),
            _ => Shape::Hull(oct_hull.clone()),
        };
        w.add_collider(b, &ColliderDesc::new(shape)).unwrap();
    }

    // Balls rolling down the mesh ramp and onto heightfield.
    for i in 0..4 {
        let b = w
            .create_body(&BodyDesc {
                kind: BodyKind::Dynamic,
                position: Vec3::new(10.5 - i as f32 * 0.8, 3.2, -1.5 + i as f32 * 1.0),
                ..Default::default()
            })
            .unwrap();
        let shape = if i % 2 == 0 {
            Shape::Sphere { radius: 0.35 }
        } else {
            Shape::Hull(prism_hull.clone())
        };
        w.add_collider(b, &ColliderDesc::new(shape)).unwrap();
    }
    for i in 0..3 {
        let b = w
            .create_body(&BodyDesc {
                kind: BodyKind::Dynamic,
                position: Vec3::new(-14.0 + i as f32 * 1.1, 2.5, -1.0 + i as f32 * 0.9),
                ..Default::default()
            })
            .unwrap();
        w.add_collider(
            b,
            &ColliderDesc::new(if i == 1 {
                Shape::Capsule {
                    half_height: 0.3,
                    radius: 0.25,
                }
            } else {
                Shape::Hull(oct_hull.clone())
            }),
        )
        .unwrap();
    }

    // 5. Moving kinematic platform + cargo + hanging jointed pendulum.
    let platform = w
        .create_body(&BodyDesc {
            kind: BodyKind::Kinematic,
            position: Vec3::new(-6.0, 1.0, 8.0),
            linear_velocity: Vec3::new(1.5, 0.0, 0.0),
            angular_velocity: Vec3::new(0.0, 0.3, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        platform,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(2.0, 0.25, 2.0),
        }),
    )
    .unwrap();
    let cargo = w
        .create_body(&BodyDesc {
            kind: BodyKind::Dynamic,
            position: Vec3::new(-6.0, 1.65, 8.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        cargo,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::splat(0.35),
        }),
    )
    .unwrap();

    // 6. All 5 joint kinds.
    let anchor = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(0.0, 8.0, -10.0),
            ..Default::default()
        })
        .unwrap();
    let j_rev = w
        .create_body(&BodyDesc {
            kind: BodyKind::Dynamic,
            position: Vec3::new(1.0, 8.0, -10.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        j_rev,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.15, 0.15),
        }),
    )
    .unwrap();
    w.create_joint(&JointDesc {
        body_a: anchor,
        body_b: j_rev,
        frame_a: Iso::IDENTITY,
        frame_b: Iso::from_position(Vec3::new(-1.0, 0.0, 0.0)),
        kind: JointKind::Hinge(HingeOptions {
            limit: Some((-1.2, 1.2)),
            motor: Some(Motor {
                speed: 2.0,
                max_force: 15.0,
            }),
            spring: None,
        }),
        collide_connected: false,
        user_data: 1,
    })
    .unwrap();

    let j_ball = w
        .create_body(&BodyDesc {
            kind: BodyKind::Dynamic,
            position: Vec3::new(2.0, 7.0, -10.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(j_ball, &ColliderDesc::new(Shape::Sphere { radius: 0.25 }))
        .unwrap();
    w.create_joint(&JointDesc {
        body_a: j_rev,
        body_b: j_ball,
        frame_a: Iso::from_position(Vec3::new(0.5, 0.0, 0.0)),
        frame_b: Iso::from_position(Vec3::new(-0.5, 1.0, 0.0)),
        kind: JointKind::Ball,
        collide_connected: false,
        user_data: 2,
    })
    .unwrap();

    let j_prism = w
        .create_body(&BodyDesc {
            kind: BodyKind::Dynamic,
            position: Vec3::new(-3.0, 7.5, -10.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        j_prism,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::splat(0.25),
        }),
    )
    .unwrap();
    w.create_joint(&JointDesc {
        body_a: anchor,
        body_b: j_prism,
        frame_a: Iso::from_position(Vec3::new(-3.0, 0.0, 0.0)),
        frame_b: Iso::IDENTITY,
        kind: JointKind::Slider(SliderOptions {
            limit: Some((-1.5, 0.5)),
            motor: None,
            spring: Some(Spring {
                hertz: 3.0,
                damping_ratio: 0.4,
            }),
        }),
        collide_connected: false,
        user_data: 3,
    })
    .unwrap();

    let j_fixed = w
        .create_body(&BodyDesc {
            kind: BodyKind::Dynamic,
            position: Vec3::new(-3.0, 6.8, -10.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(j_fixed, &ColliderDesc::new(Shape::Hull(prism_hull.clone())))
        .unwrap();
    w.create_joint(&JointDesc {
        body_a: j_prism,
        body_b: j_fixed,
        frame_a: Iso::from_position(Vec3::new(0.0, -0.35, 0.0)),
        frame_b: Iso::from_position(Vec3::new(0.0, 0.35, 0.0)),
        kind: JointKind::Fixed,
        collide_connected: false,
        user_data: 4,
    })
    .unwrap();

    let j_dist = w
        .create_body(&BodyDesc {
            kind: BodyKind::Dynamic,
            position: Vec3::new(-6.0, 0.3, 8.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(j_dist, &ColliderDesc::new(Shape::Sphere { radius: 0.2 }))
        .unwrap();
    w.create_joint(&JointDesc {
        body_a: platform,
        body_b: j_dist,
        frame_a: Iso::from_position(Vec3::new(0.0, -0.3, 0.0)),
        frame_b: Iso::IDENTITY,
        kind: JointKind::Distance(DistanceOptions {
            length: 0.6,
            min: 0.2,
            max: 0.8,
            spring: Some(Spring {
                hertz: 4.0,
                damping_ratio: 0.5,
            }),
        }),
        collide_connected: false,
        user_data: 5,
    })
    .unwrap();

    // 7. Thin wall + fast CCD bullets (hull + box).
    let wall = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(0.0, 2.0, -22.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        wall,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(4.0, 2.0, 0.05),
        }),
    )
    .unwrap();
    let bullet = w
        .create_body(&BodyDesc {
            kind: BodyKind::Dynamic,
            position: Vec3::new(-0.5, 2.0, -15.0),
            linear_velocity: Vec3::new(0.0, 0.0, -45.0),
            bullet: true,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(bullet, &ColliderDesc::new(Shape::Hull(prism_hull)))
        .unwrap();

    (w, sleeper_top, platform, cargo, bullet)
}

#[test]
fn bit_identical_world_and_character_hashes() {
    let (mut w, sleeper_top, platform, cargo, bullet) = build_complex_world();

    let mut ch = Character::new(
        Vec3::new(-2.0, 1.2, 2.0),
        CharacterSettings {
            half_height: 0.5,
            radius: 0.3,
            ..Default::default()
        },
    );

    let checkpoints = [1u32, 5, 15, 30, 60, 90, 120, 150, 200, 260];
    let mut hashes = Vec::new();
    let mut event_counts = Vec::new();

    for step in 1..=260 {
        if step == 70 {
            // Stop the kinematic platform so its island can settle to sleep.
            w.set_linear_velocity(platform, Vec3::ZERO);
            w.set_angular_velocity(platform, Vec3::ZERO);
        }
        if step == 140 {
            // Change substep count while some islands are asleep and some awake!
            w.settings.substeps = 2;
        }
        if step == 160 {
            w.settings.substeps = 4;
            // Wake the distant sleeping stack with an impulse and restart the platform!
            w.apply_impulse_at(
                sleeper_top,
                Vec3::new(-3.5, 2.0, 1.0),
                Vec3::new(30.1, 3.0, 30.0),
            );
            w.set_linear_velocity(platform, Vec3::new(-2.0, 0.2, 0.0));
        }
        if step == 190 {
            w.set_body_pose(bullet, Iso::from_position(Vec3::new(0.5, 2.2, -16.0)));
            w.set_linear_velocity(bullet, Vec3::new(0.0, 0.0, -50.0));
        }
        if step == 220 {
            w.destroy_body(cargo);
        }

        let wish = Vec3::new(
            if step < 130 { 2.5 } else { -1.8 },
            -4.0,
            if step % 60 < 30 { -1.5 } else { 1.5 },
        );
        ch.move_and_slide(&mut w, wish * DT, DT);
        w.step(DT);

        if checkpoints.contains(&step) {
            hashes.push(w.state_hash());
            let ev = w.events();
            event_counts.push((
                ev.contacts.len(),
                ev.sensors.len(),
                w.awake_body_count(),
                w.contact_count(),
            ));
        }
    }

    let ch_pos = ch.position;
    let ch_bits = [ch_pos.x.to_bits(), ch_pos.y.to_bits(), ch_pos.z.to_bits()];

    let expected_hashes: [u64; 10] = [
        0x50b5_19bd_b301_7e73,
        0xda52_8909_0a78_153a,
        0x025f_94a1_3650_507c,
        0x5548_8067_9ac4_8b32,
        0x989d_1d0b_c372_3eae,
        0x746d_ff59_e12c_e200,
        0xf799_79b6_8131_9a44,
        0x6f12_9fcd_ff69_474f,
        0xf1c4_67e2_8365_995c,
        0x3a8b_3f4b_2ee6_6913,
    ];
    let expected_events: [(usize, usize, usize, usize); 10] = [
        (4, 7, 27, 4),
        (1, 0, 27, 6),
        (2, 0, 27, 9),
        (0, 0, 27, 14),
        (0, 0, 21, 19),
        (0, 0, 20, 16),
        (1, 0, 18, 21),
        (0, 0, 16, 22),
        (2, 0, 15, 22),
        (1, 0, 15, 22),
    ];
    let expected_ch_bits: [u32; 3] = [0xbe9a_108c, 0x3f51_f6c1, 0x3fd1_e792];

    assert_eq!(hashes.as_slice(), &expected_hashes, "state_hash drifted!");
    assert_eq!(
        event_counts.as_slice(),
        &expected_events,
        "event/awake/contact counts drifted!"
    );
    assert_eq!(ch_bits, expected_ch_bits, "character position drifted!");
}
