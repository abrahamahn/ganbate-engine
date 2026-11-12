//! Queries against one-sided triangles (review finding QUERY-ONE-SIDED). Mesh and heightfield
//! triangles are one-sided for contacts, continuous collision and rays: only their front faces
//! stop anything. `shape_cast` swept every triangle two-sided, so a cast from below a floor hit
//! its underside where a ray, and a body thrown the same way, pass through; and `overlap`, which
//! runs the narrow phase, missed a shape straddling a floor once its centre was below it.
//! `contacts_with` stays the narrow phase's, one-sided: the character pushes itself out of
//! surfaces along its normals. Every scene here is synthetic.

use std::sync::Arc;

use ganbate_physics::mesh::{Heightfield, TriangleMesh};
use ganbate_physics::{
    BodyDesc, BodyKind, ColliderDesc, Iso, PhysicsWorld, QueryFilter, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

/// A 20 m floor at y = 0, front faces up: a mesh of four quads, or a heightfield of 1 m cells.
fn floor(mesh: bool) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let (shape, at) = if mesh {
        let mut v = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                v.push(Vec3::new(
                    i as f32 * 10.0 - 10.0,
                    0.0,
                    j as f32 * 10.0 - 10.0,
                ));
            }
        }
        let mut t = Vec::new();
        for j in 0..2u32 {
            for i in 0..2u32 {
                let a = j * 3 + i;
                t.push([a, a + 3, a + 1]);
                t.push([a + 1, a + 3, a + 4]);
            }
        }
        (
            Shape::Mesh(Arc::new(TriangleMesh::new(v, t).unwrap())),
            Vec3::ZERO,
        )
    } else {
        (
            Shape::Heightfield(Arc::new(
                Heightfield::new(21, 21, vec![0.0; 21 * 21], Vec3::ONE).unwrap(),
            )),
            Vec3::new(-10.0, 0.0, -10.0),
        )
    };
    let b = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: at,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(shape)).unwrap();
    w.step(DT);
    w
}

fn shapes() -> [Shape; 3] {
    [
        Shape::Sphere { radius: 0.3 },
        Shape::Box {
            half_extents: Vec3::splat(0.3),
        },
        Shape::Capsule {
            half_height: 0.2,
            radius: 0.3,
        },
    ]
}

/// From below, a cast up through the floor finds nothing, as a ray does; from above both find
/// the floor. Beside the floor's border (an active edge, where a hit keeps its own normal) the
/// cast from below hit the underside at fraction 0.425; on a seam inside the floor the hit took
/// the face's normal and was let through as one moving away from it.
#[test]
fn a_shape_cast_from_below_a_one_sided_floor_passes_through_it_as_a_ray_does() {
    let all = QueryFilter::default();
    for mesh in [true, false] {
        let w = floor(mesh);
        for shape in shapes() {
            let reach = shape.aabb(&Iso::IDENTITY).max.y;
            for (x, z) in [(1.0, 1.0), (9.9, 1.5), (-9.9, 9.9)] {
                let below = w.shape_cast(
                    &shape,
                    &Iso::from_position(Vec3::new(x, -2.0, z)),
                    Vec3::new(0.0, 4.0, 0.0),
                    &all,
                );
                let ray = w.raycast(Vec3::new(x, -2.0, z), Vec3::Y, 4.0, &all);
                assert!(
                    below.is_none() && ray.is_none(),
                    "mesh {mesh}, {shape:?} from below at ({x}, {z}): cast {below:?}, ray {ray:?}"
                );
                let above = w
                    .shape_cast(
                        &shape,
                        &Iso::from_position(Vec3::new(x, 2.0, z)),
                        Vec3::new(0.0, -4.0, 0.0),
                        &all,
                    )
                    .expect("the floor, from above");
                assert!(
                    (above.fraction - (2.0 - reach) / 4.0).abs() < 1e-3
                        && (above.normal - Vec3::Y).length() < 1e-4,
                    "mesh {mesh}, {shape:?} from above at ({x}, {z}): {above:?}"
                );
            }
        }
    }
}

/// What the cast now predicts is what the world does: a ball thrown up from below passes
/// through the floor.
#[test]
fn a_ball_thrown_up_through_a_one_sided_floor_passes_as_the_cast_predicts() {
    for mesh in [true, false] {
        let mut w = floor(mesh);
        let cast = w.shape_cast(
            &Shape::Sphere { radius: 0.3 },
            &Iso::from_position(Vec3::new(9.9, -2.0, 1.5)),
            Vec3::new(0.0, 4.0, 0.0),
            &QueryFilter::default(),
        );
        assert!(cast.is_none(), "mesh {mesh}: {cast:?}");
        let b = w
            .create_body(&BodyDesc {
                position: Vec3::new(9.9, -2.0, 1.5),
                linear_velocity: Vec3::new(0.0, 3.0, 0.0),
                ..Default::default()
            })
            .unwrap();
        w.add_collider(b, &ColliderDesc::new(Shape::Sphere { radius: 0.3 }))
            .unwrap();
        for _ in 0..90 {
            w.step(DT);
        }
        let p = w.body_pose(b).unwrap().position;
        assert!(p.y > 2.0, "mesh {mesh}: the ball ends at {p:?}");
    }
}

/// A shape crossing the floor overlaps it wherever its centre is (with the centre below it the
/// narrow phase gives no contact, and overlap found nothing); one wholly below or above it, or
/// only touching it, does not.
#[test]
fn overlap_finds_a_shape_straddling_a_one_sided_floor() {
    let all = QueryFilter::default();
    for mesh in [true, false] {
        let w = floor(mesh);
        for shape in shapes() {
            let reach = shape.aabb(&Iso::IDENTITY).max.y;
            let count = |y: f32| {
                let mut n = 0;
                w.overlap(
                    &shape,
                    &Iso::from_position(Vec3::new(1.0, y, 1.0)),
                    &all,
                    &mut |_| {
                        n += 1;
                        true
                    },
                );
                n
            };
            for k in -5..=5 {
                let y = 0.05 * k as f32;
                assert_eq!(count(y), 1, "mesh {mesh}, {shape:?} centred at y = {y}");
            }
            for y in [
                reach,
                -reach,
                reach + 0.01,
                -reach - 0.01,
                reach + 0.3,
                -reach - 0.3,
            ] {
                assert_eq!(count(y), 0, "mesh {mesh}, {shape:?} centred at y = {y}");
            }
        }
    }
}

/// At the border of a floor the box can straddle its plane while touching only the outside
/// edge or corner. Strict overlap distinguishes this from moving across the border.
#[test]
fn touching_a_floor_boundary_from_either_side_is_not_an_overlap() {
    let shape = Shape::Box {
        half_extents: Vec3::splat(0.5),
    };
    for mesh in [true, false] {
        let w = floor(mesh);
        for y in [-0.25, 0.25] {
            for z in [0.0, 10.5] {
                for inset in [0.0, 0.01] {
                    let at = Iso::from_position(Vec3::new(10.5 - inset, y, z - inset));
                    let mut count = 0;
                    w.overlap(&shape, &at, &QueryFilter::default(), &mut |_| {
                        count += 1;
                        true
                    });
                    assert_eq!(count, usize::from(inset > 0.0), "{mesh} {at:?}");
                }
            }
        }
    }
}

/// `contacts_with` is the narrow phase's view: a shape whose centre is below the floor has no
/// contact with it (a character pushed out along such a normal would be pushed through).
#[test]
fn contacts_with_stays_one_sided() {
    for mesh in [true, false] {
        let w = floor(mesh);
        for shape in shapes() {
            let mut below = 0;
            w.contacts_with(
                &shape,
                &Iso::from_position(Vec3::new(1.0, -0.1, 1.0)),
                0.02,
                &QueryFilter::default(),
                &mut |_, _| below += 1,
            );
            let mut above = Vec::new();
            w.contacts_with(
                &shape,
                &Iso::from_position(Vec3::new(1.0, 0.1, 1.0)),
                0.02,
                &QueryFilter::default(),
                &mut |_, m| above.push(m.normal),
            );
            assert!(
                below == 0 && !above.is_empty() && above.iter().all(|n| n.y < -0.99),
                "mesh {mesh}, {shape:?}: {below} manifolds from below, normals {above:?} from above"
            );
        }
    }
}
