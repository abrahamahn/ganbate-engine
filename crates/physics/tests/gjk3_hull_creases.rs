//! `ConvexHull::from_faces` and coplanar neighbouring faces (review finding
//! HULL-COPLANAR-NEIGHBOURS).
//!
//! The hull module's layout rule says neighbouring faces are never coplanar, so that every edge
//! is a real crease. `from_faces` did not check it: a box whose top was given as two triangles
//! (what a triangulated mesh export gives) was accepted with 7 faces, and a box resting across
//! the split got its contact points on one triangle only (3 points, or 1 point outside its own
//! footprint) and tipped over. Now `from_faces` refuses such faces (`from_points` merges them),
//! and a flat, back-to-back "hull" too. Every shape here is synthetic.

use std::sync::Arc;

use ganbate_physics::collide::collide;
use ganbate_physics::hull::ConvexHull;
use ganbate_physics::{Iso, Quat, Shape, Vec3};

/// The six quads of `ConvexHull::cuboid`'s vertex numbering (bit 0 is +x, bit 1 +y, bit 2 +z).
const QUADS: [[u32; 4]; 6] = [
    [0, 4, 6, 2],
    [1, 3, 7, 5],
    [0, 1, 5, 4],
    [2, 6, 7, 3],
    [0, 2, 3, 1],
    [4, 5, 7, 6],
];

fn build(vertices: Vec<Vec3>, loops: &[Vec<u32>]) -> Result<ConvexHull, String> {
    let refs: Vec<&[u32]> = loops.iter().map(|l| l.as_slice()).collect();
    ConvexHull::from_faces(vertices, &refs)
}

/// The quads with the ones listed in `split` cut into two triangles.
fn split_quads(split: &[usize]) -> Vec<Vec<u32>> {
    let mut loops = Vec::new();
    for (i, q) in QUADS.iter().enumerate() {
        if split.contains(&i) {
            loops.push(vec![q[0], q[1], q[2]]);
            loops.push(vec![q[0], q[2], q[3]]);
        } else {
            loops.push(q.to_vec());
        }
    }
    loops
}

fn refused_as_coplanar(r: Result<ConvexHull, String>, what: &str) {
    match r {
        Ok(h) => panic!("{what}: accepted with {} faces", h.faces.len()),
        Err(e) => assert!(e.contains("coplanar"), "{what}: {e}"),
    }
}

#[test]
fn coplanar_neighbouring_faces_are_refused() {
    let v = ConvexHull::cuboid(Vec3::new(1.0, 0.5, 1.0)).vertices;
    refused_as_coplanar(build(v.clone(), &split_quads(&[3])), "top split");
    refused_as_coplanar(build(v.clone(), &split_quads(&[2])), "bottom split");
    refused_as_coplanar(
        build(v.clone(), &split_quads(&[0, 1, 2, 3, 4, 5])),
        "triangles",
    );
    // The same triangles turned and moved, so rounding puts them a hair off one plane.
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut unit = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    for case in 0..100 {
        let axis = Vec3::new(unit() - 0.5, unit() - 0.5, unit() - 0.5);
        let q = Quat::from_axis_angle(axis.normalize_or_zero(), 6.0 * unit());
        let at = Vec3::new(unit() - 0.5, unit() - 0.5, unit() - 0.5) * 200.0;
        let half = Vec3::new(0.1 + unit(), 0.1 + unit(), 0.1 + unit());
        let v: Vec<Vec3> = ConvexHull::cuboid(half)
            .vertices
            .iter()
            .map(|&p| q.rotate(p) + at)
            .collect();
        refused_as_coplanar(
            build(v.clone(), &split_quads(&[case % 6])),
            &format!("case {case}"),
        );
        // Given as quads, the same corners build.
        build(v, &split_quads(&[])).unwrap_or_else(|e| panic!("case {case}: {e}"));
    }
    // A hexagonal prism whose caps are fans of triangles.
    let mut v = Vec::new();
    for k in 0..6 {
        let (s, c) = (k as f32 * std::f32::consts::FRAC_PI_3).sin_cos();
        v.push(Vec3::new(c, -0.4, s));
        v.push(Vec3::new(c, 0.4, s));
    }
    let mut loops: Vec<Vec<u32>> = (0..6u32)
        .map(|k| {
            let n = (k + 1) % 6;
            vec![2 * k, 2 * k + 1, 2 * n + 1, 2 * n]
        })
        .collect();
    let bottom: Vec<u32> = (0..6u32).map(|k| 2 * k).collect();
    let top: Vec<u32> = (0..6u32).rev().map(|k| 2 * k + 1).collect();
    build(
        v.clone(),
        &[loops.clone(), vec![bottom.clone(), top.clone()]].concat(),
    )
    .expect("the prism with whole caps");
    for k in 1..5 {
        loops.push(vec![top[0], top[k], top[k + 1]]);
    }
    loops.push(bottom);
    refused_as_coplanar(build(v, &loops), "prism with a fanned cap");
}

#[test]
fn a_flat_hull_is_refused() {
    // A square given as its two sides: closed, convex and planar, but with no inside.
    let v = vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 0.0, 1.0), Vec3::Z];
    let e = ConvexHull::from_faces(v, &[&[0, 3, 2, 1], &[0, 1, 2, 3]]).unwrap_err();
    assert!(e.contains("flat"), "{e}");
}

#[test]
fn real_creases_build_however_thin_or_shallow() {
    // Boxes at the extremes Shape::Box allows: plates, needles and tiny cubes, whose thin sides
    // lie within rounding of their big neighbours' planes but cross them square.
    for half in [
        Vec3::new(1e5, 1e-4, 1e5),
        Vec3::new(1e-4, 1e-4, 1e5),
        Vec3::splat(1e-4),
        Vec3::splat(1e5),
        Vec3::new(3.0, 0.02, 0.7),
    ] {
        let h = ConvexHull::cuboid(half);
        assert_eq!(h.faces.len(), 6, "{half:?}");
    }
    // A box with a roof whose ridge rises 0.5 mm over a 2 m span: a shallow but real crease.
    let mut v = ConvexHull::cuboid(Vec3::new(1.0, 0.5, 1.0)).vertices;
    v.push(Vec3::new(0.0, 0.5005, -1.0));
    v.push(Vec3::new(0.0, 0.5005, 1.0));
    let roof = vec![
        vec![0, 4, 6, 2],
        vec![1, 3, 7, 5],
        vec![0, 1, 5, 4],
        vec![2, 6, 9, 8],
        vec![8, 9, 7, 3],
        vec![0, 2, 8, 3, 1],
        vec![4, 5, 7, 9, 6],
    ];
    let h = build(v, &roof).expect("a shallow roof");
    assert_eq!(h.faces.len(), 7);
    // The octahedron and a pyramid.
    let v = vec![Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
    let faces = [
        vec![0, 2, 4],
        vec![1, 4, 2],
        vec![0, 4, 3],
        vec![1, 3, 4],
        vec![0, 5, 2],
        vec![1, 2, 5],
        vec![0, 3, 5],
        vec![1, 5, 3],
    ];
    build(v, &faces).expect("the octahedron");
    let v = vec![
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(-1.0, 0.0, 1.0),
        Vec3::new(0.0, 1.5, 0.0),
    ];
    let faces = [
        vec![0, 1, 2, 3],
        vec![0, 4, 1],
        vec![1, 4, 2],
        vec![2, 4, 3],
        vec![3, 4, 0],
    ];
    build(v, &faces).expect("the pyramid");
}

/// The triangles of a mesh export go through `from_points`, which merges them: a box resting
/// across the diagonal of the top gets the four corners of its own bottom.
#[test]
fn a_box_rests_on_four_points_on_a_hull_built_from_a_mesh_exports_corners() {
    let corners = ConvexHull::cuboid(Vec3::new(1.0, 0.5, 1.0)).vertices;
    let hull = ConvexHull::from_points(&corners).unwrap();
    assert_eq!(hull.faces.len(), 6);
    let small = Shape::Box {
        half_extents: Vec3::new(0.3, 0.1, 0.3),
    };
    let iso = Iso::new(
        Vec3::new(0.1, 0.599, -0.1),
        Quat::from_axis_angle(Vec3::Y, 0.3),
    );
    let mut out = Vec::new();
    collide(
        &small,
        &iso,
        &Shape::Hull(Arc::new(hull)),
        &Iso::IDENTITY,
        0.02,
        &mut out,
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].count, 4, "{:?}", out[0]);
    for p in out[0].points() {
        let local = iso.inverse_point(p.point);
        assert!(
            local.x.abs() <= 0.3 + 1e-4 && local.z.abs() <= 0.3 + 1e-4,
            "{p:?}"
        );
    }
}
