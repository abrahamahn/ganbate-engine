//! The character on level geometry made of many small triangles: flat floors of triangle meshes
//! and heightfields with centimetre cells, and a slope made of them. Each triangle is swept on its
//! own, so the seams between coplanar neighbours (inactive edges, which the narrow phase reads as
//! the face) must not stop it.

use std::sync::Arc;

use ganbate_physics::mesh::{Heightfield, TriangleMesh};
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, Character, CharacterSettings, ColliderDesc, PhysicsWorld, Shape,
    Vec3,
};

const DT: f32 = 1.0 / 60.0;

/// A square grid of `n` × `n` cells of `cell` metres from `origin`, two triangles a cell, facing
/// up, with the height of each vertex from `height(x, z)`.
fn grid_mesh(n: usize, cell: f32, origin: Vec3, height: impl Fn(f32, f32) -> f32) -> TriangleMesh {
    let mut vertices = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let x = origin.x + i as f32 * cell;
            let z = origin.z + j as f32 * cell;
            vertices.push(Vec3::new(x, origin.y + height(x, z), z));
        }
    }
    let k = |i: usize, j: usize| (j * (n + 1) + i) as u32;
    let mut triangles = Vec::new();
    for j in 0..n {
        for i in 0..n {
            triangles.push([k(i, j), k(i, j + 1), k(i + 1, j + 1)]);
            triangles.push([k(i, j), k(i + 1, j + 1), k(i + 1, j)]);
        }
    }
    TriangleMesh::new(vertices, triangles).unwrap()
}

fn add_static(w: &mut PhysicsWorld, shape: Shape, position: Vec3) -> BodyId {
    let b = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(shape)).unwrap();
    b
}

/// A flat floor at y = 0 from (-5, -5) to (5, 5): a triangle mesh, or a heightfield.
fn fine_floor(heightfield: bool, cell: f32) -> PhysicsWorld {
    let mut w = PhysicsWorld::default();
    let n = (10.0 / cell).round() as usize;
    if heightfield {
        let samples = (n + 1) as u32;
        let field = Heightfield::new(
            samples,
            samples,
            vec![0.0; (n + 1) * (n + 1)],
            Vec3::new(cell, 1.0, cell),
        )
        .unwrap();
        add_static(
            &mut w,
            Shape::Heightfield(Arc::new(field)),
            Vec3::new(-5.0, 0.0, -5.0),
        );
    } else {
        let mesh = grid_mesh(n, cell, Vec3::new(-5.0, 0.0, -5.0), |_, _| 0.0);
        add_static(&mut w, Shape::Mesh(Arc::new(mesh)), Vec3::ZERO);
    }
    w
}

fn stand() -> f32 {
    let s = CharacterSettings::default();
    s.half_height + s.radius + s.skin
}

/// Walks `c` at `velocity` (m/s) for `moves` moves with the game's usual vertical speed (a small
/// push down while grounded, gravity otherwise); returns how far each move went horizontally.
fn walk(w: &mut PhysicsWorld, c: &mut Character, velocity: Vec3, moves: usize) -> Vec<f32> {
    let mut vy = 0.0f32;
    let mut out = Vec::new();
    for _ in 0..moves {
        vy = if c.grounded { -0.1 } else { vy - 9.81 * DT };
        let r = c.move_and_slide(w, velocity * DT + Vec3::Y * (vy * DT), DT);
        w.step(DT);
        out.push(Vec3::new(r.moved.x, 0.0, r.moved.z).length());
    }
    out
}

/// Walking straight across a flat floor of 3 to 8 cm triangles, at headings where the seams of
/// the triangles around the one underfoot meet the capsule's skin, it never slows down. (Each of
/// those seams used to stop a slide with its own tilted edge normal, and four of them took every
/// slide iteration: 55 of 432 walks stood still for good.)
#[test]
fn walking_across_flat_floors_of_small_triangles_never_slows_down() {
    let mut stalls = Vec::new();
    // Walks the review found stuck for good, and their neighbours.
    for (heightfield, cell, speed, degrees) in [
        (false, 0.05f32, 3.0f32, 42.0f32),
        (false, 0.05, 3.0, 12.0),
        (false, 0.03, 3.0, 7.0),
        (false, 0.03, 1.5, 32.0),
        (false, 0.08, 3.0, 27.0),
        (false, 0.05, 6.0, 32.0),
        (true, 0.03, 1.5, 57.0),
        (true, 0.03, 3.0, 47.0),
        (true, 0.05, 1.5, 62.0),
        (true, 0.08, 3.0, 52.0),
        (true, 0.05, 6.0, 57.0),
    ] {
        let mut w = fine_floor(heightfield, cell);
        let mut c = Character::new(Vec3::new(-4.3, stand(), -4.3), CharacterSettings::default());
        let a = degrees.to_radians();
        let velocity = Vec3::new(a.cos(), 0.0, a.sin()) * speed;
        let moves = (3.0 / speed / DT).round() as usize;
        let travel = walk(&mut w, &mut c, velocity, moves);
        let slow = travel.iter().filter(|&&d| d < 0.999 * speed * DT).count();
        let label = format!(
            "{} of {cell} m cells at {speed} m/s, heading {degrees}°",
            if heightfield { "heightfield" } else { "mesh" }
        );
        if slow > 0 {
            stalls.push(format!(
                "{label}: {slow} of {moves} moves slowed, ended at {:?}",
                c.position
            ));
        }
        assert!(
            c.grounded && (c.feet().y - c.settings.skin).abs() < 5e-4,
            "{label}: on the floor a skin above it: {:?}",
            c.position
        );
    }
    assert!(stalls.is_empty(), "{stalls:#?}");
}

/// Walking up a 20° slope of 5 cm triangles, straight up it or across it at 30° and 45°, it keeps
/// climbing every move. (It used to stand still for about a hundred moves at a time at 30° and
/// 45°.)
#[test]
fn climbing_a_slope_of_small_triangles_never_stalls() {
    for degrees in [0.0f32, 30.0, 45.0] {
        let mut w = PhysicsWorld::default();
        let slope = 20.0f32.to_radians();
        let rise = slope.sin() / slope.cos();
        // Flat for x < 1, then rising at 20° towards +X.
        let mesh = grid_mesh(140, 0.05, Vec3::new(-1.0, 0.0, -3.5), move |x, _| {
            ((x - 1.0) * rise).max(0.0)
        });
        add_static(&mut w, Shape::Mesh(Arc::new(mesh)), Vec3::ZERO);
        let mut c = Character::new(Vec3::new(0.0, stand(), -2.0), CharacterSettings::default());
        let h = degrees.to_radians();
        let speed = 3.0f32;
        let travel = walk(&mut w, &mut c, Vec3::new(h.cos(), 0.0, h.sin()) * speed, 80);
        let stuck = travel.iter().filter(|&&d| d < 0.25 * speed * DT).count();
        assert!(
            stuck == 0,
            "heading {degrees}°: {stuck} of 80 moves (nearly) stood still; ended at {:?}",
            c.position
        );
        // Climbing at 20° costs the walk nothing horizontally: 80 moves at 3 m/s is 4 m.
        let x = c.position.x;
        assert!(
            x > h.cos() * 3.5 && c.grounded,
            "heading {degrees}°: climbed to {:?}",
            c.position
        );
        assert!(
            (c.position.y - stand() - (x - 1.0) * rise).abs() < 0.05,
            "heading {degrees}°: on the slope: {:?}",
            c.position
        );
    }
}

/// Walking over rolling terrain of 3 to 10 cm cells, whose ridges between triangles are real
/// (active) edges only a fraction of a degree sharp: it keeps walking. The ridges near the
/// capsule's bottom come within the casts' tolerance of its skin while it passes over them a skin
/// above, and must not stop it. (With only the seams between coplanar triangles read as faces it
/// still stood still for up to 150 moves at a time.)
#[test]
fn walking_over_rolling_terrain_of_small_triangles_never_stalls() {
    let mut stalls = Vec::new();
    for (amplitude, cell, speed, degrees) in [
        (0.05f32, 0.03f32, 1.5f32, 23.0f32),
        (0.05, 0.03, 1.5, 203.0),
        (0.05, 0.03, 4.0, 63.0),
        (0.3, 0.05, 1.5, 43.0),
        (0.6, 0.05, 1.5, 23.0),
        (0.3, 0.1, 1.5, 43.0),
    ] {
        let mut w = PhysicsWorld::default();
        let samples = (12.0 / cell) as u32 + 1;
        let mut heights = Vec::new();
        for j in 0..samples {
            for i in 0..samples {
                let (x, z) = (i as f32 * cell, j as f32 * cell);
                heights.push(amplitude * (x * 0.9).sin() * (z * 0.7).cos());
            }
        }
        let field =
            Heightfield::new(samples, samples, heights, Vec3::new(cell, 1.0, cell)).unwrap();
        add_static(
            &mut w,
            Shape::Heightfield(Arc::new(field)),
            Vec3::new(-6.0, 0.0, -6.0),
        );
        let mut c = Character::new(
            Vec3::new(0.0, amplitude + stand() + 0.2, 0.0),
            CharacterSettings::default(),
        );
        let a = degrees.to_radians();
        let velocity = Vec3::new(a.cos(), 0.0, a.sin()) * speed;
        let moves = (4.0 / speed / DT) as usize + 20;
        let travel = walk(&mut w, &mut c, velocity, moves);
        let stuck = travel.iter().filter(|&&d| d < 0.25 * speed * DT).count();
        if stuck > 0 {
            stalls.push(format!(
                "{amplitude} m hills of {cell} m cells at {speed} m/s, heading {degrees}°: \
                 {stuck} of {moves} moves (nearly) stood still, ended at {:?}",
                c.position
            ));
        }
    }
    assert!(stalls.is_empty(), "{stalls:#?}");
}
