//! Physics for a cooked scene: the `rigid_body` and `collider` components an author set in Ganbate
//! Studio (carried in the cooked world's entity table) become bodies and colliders.
//!
//! An entity with a `rigid_body` is a body of the kind that says. A `collider` on an entity
//! without one belongs to the body of its nearest ancestor that has one, so a cart's wheels move
//! with the cart (and a mesh collider there is refused unless that body is static); with no such
//! ancestor it is static scenery. Field defaults match `crates/scene/src/engine.schema.json`.
//!
//! A body's pose is the rigid part of its entity's cooked transform. A mirrored transform
//! (negative determinant) has no rotation of its own, so one axis is turned round first, the one
//! that leaves the smallest rotation, and the mirror stays in that axis's scale
//! ([`SceneBody::scale`]). Collider sizes are in their entity's frame and scale with it along the
//! collider's own axes after its `rotation`: exactly whenever the scale runs along those axes
//! (any turn under an even scale, quarter turns under an uneven one, no shear from a parent);
//! otherwise the drawn shape is sheared. A box turned against the uneven scale of an entity that
//! its parents do not shear keeps its turn with each edge as long as drawn, while that box stays
//! close to the drawn one: at most a quarter larger, and each inside the other doubled about
//! their centre. Any other sheared box is exactly what is drawn: a box when its drawn edges are
//! square after all, otherwise the convex hull of its eight drawn corners (boxes and hulls
//! collide alike, as polytopes). A capsule keeps its drawn axis and takes the largest stretch
//! across it, and a ball always takes the largest stretch. `hull` and `mesh` colliders take the
//! entity's own mesh geometry from the cooked world, with mirrored instances' triangles wound so
//! their front faces stay in front; a hull is made of at most [`MAX_HULL_VERTICES`] of the mesh's
//! extreme points.
//!
//! After each step a game (or the editor) moves the instances and lamps of the entities each body
//! carries ([`SceneBody::carries`]) with [`ScenePhysics::instance_matrix`] or
//! [`ScenePhysics::motion`]: rigidly, so a mirrored or sheared entity keeps its shape, and not at
//! all until its body moves. [`ScenePhysics::moved_since`] names the bodies that moved, so
//! sleeping and still bodies cost nothing. A host that steps once a frame uses
//! [`ScenePhysics::advance`], which never runs more than [`MAX_STEPS_PER_CALL`] steps in one call.

use std::sync::Arc;

use ganbate_core::json::{self, Value};
use ganbate_core::world::{EntityRecord, EntityTable, World};

use crate::hull::{ConvexHull, MAX_HULL_VERTICES};
use crate::math::{Iso, Mat3, Quat, Vec3, sin_cos};
use crate::mesh::TriangleMesh;
use crate::shape::{MAX_EXTENT, MIN_EXTENT, Shape};
use crate::world::{BodyDesc, BodyId, BodyKind, ColliderDesc, PhysicsWorld, Settings};

/// Most steps [`ScenePhysics::advance`] runs in one call. A host that fell further behind (a tab
/// in the background, a breakpoint, a count of milliseconds passed as steps) drops the rest
/// instead of freezing while it catches up: at 60 Hz this is about a second.
pub const MAX_STEPS_PER_CALL: u32 = 64;

/// Longest step [`ScenePhysics::advance`] takes, in seconds; a longer one is shortened to it.
pub const MAX_STEP: f32 = 1.0 / 15.0;

/// A body made for an entity.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneBody {
    /// The entity's record in the table.
    pub record: u32,
    pub entity: u64,
    pub body: BodyId,
    /// The entity's scale along the body's axes when cooked (bodies are rigid; the scale rides
    /// along): the body's rotation times this scale gives the cooked transform's axes. A mirrored
    /// entity has one negative axis.
    pub scale: Vec3,
    /// The records whose instances and lamps move with this body, in table order: its entity's,
    /// then, for a body made from a `rigid_body`, every descendant that has no `rigid_body` of
    /// its own and none on an ancestor below this one.
    pub carries: Vec<u32>,
    /// The body's pose when built: the cooked instances are placed relative to it.
    pose: Iso,
    /// The entity's cooked world transform (column-major).
    cooked: [f32; 16],
}

/// A physics world built from a cooked scene, and which entity each body stands for.
#[derive(Clone)]
pub struct ScenePhysics {
    pub world: PhysicsWorld,
    pub bodies: Vec<SceneBody>,
}

impl ScenePhysics {
    /// Steps the world `steps` times by `dt` seconds each, as a host does once a frame: at most
    /// [`MAX_STEPS_PER_CALL`] steps, each at most [`MAX_STEP`] long. Returns how many steps ran,
    /// none when `dt` is not a positive number. Each step is exactly [`PhysicsWorld::step`], so
    /// a run of calls computes what the same steps taken one by one compute.
    pub fn advance(&mut self, dt: f32, steps: u32) -> u32 {
        // NaN takes no step: `f32::min` below would turn it into `MAX_STEP`.
        if dt.is_nan() || dt <= 0.0 {
            return 0;
        }
        let dt = dt.min(MAX_STEP);
        let steps = steps.min(MAX_STEPS_PER_CALL);
        for _ in 0..steps {
            self.world.step(dt);
        }
        steps
    }

    /// How body `i` has moved since it was built: the rigid motion from its pose then to its pose
    /// now, exactly the identity until it moves; `None` once the body is destroyed.
    pub fn motion(&self, i: usize) -> Option<Iso> {
        let b = &self.bodies[i];
        let now = self.world.body_pose(b.body)?;
        Some(if now == b.pose {
            Iso::IDENTITY
        } else {
            now.mul(&b.pose.inverse())
        })
    }

    /// Body `i`'s entity's world matrix now (column-major): its cooked transform carried by the
    /// body's [motion](ScenePhysics::motion); `None` once the body is destroyed.
    pub fn entity_matrix(&self, i: usize) -> Option<[f32; 16]> {
        self.instance_matrix(i, &self.bodies[i].cooked)
    }

    /// Where an instance that body `i` carries ([`SceneBody::carries`]), cooked at `cooked`
    /// (column-major), is now: carried rigidly by the body's [motion](ScenePhysics::motion), so
    /// a mirrored or sheared instance keeps its shape, and exactly `cooked` until the body moves.
    /// `None` once the body is destroyed.
    pub fn instance_matrix(&self, i: usize, cooked: &[f32; 16]) -> Option<[f32; 16]> {
        let m = self.motion(i)?;
        Some(if m == Iso::IDENTITY {
            *cooked
        } else {
            mul(&m.to_matrix(), cooked)
        })
    }

    /// Every body's pose now, in [`ScenePhysics::bodies`] order (`None` for a destroyed body).
    /// Taken right after [`build`], it says where every instance is drawn: where it was cooked.
    pub fn poses(&self) -> Vec<Option<Iso>> {
        self.bodies
            .iter()
            .map(|b| self.world.body_pose(b.body))
            .collect()
    }

    /// The bodies (indices into [`ScenePhysics::bodies`]) whose pose is no longer the one in
    /// `shown`, one entry per body, which is brought up to date: after a step, the bodies whose
    /// instances a host must move. Sleeping and still bodies are not listed. Start `shown` from
    /// [`ScenePhysics::poses`] when the instances are where they were cooked.
    pub fn moved_since(&self, shown: &mut [Option<Iso>]) -> Vec<usize> {
        let mut moved = Vec::new();
        for (i, (b, s)) in self.bodies.iter().zip(shown.iter_mut()).enumerate() {
            let now = self.world.body_pose(b.body);
            if now != *s {
                *s = now;
                moved.push(i);
            }
        }
        moved
    }
}

fn mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    out
}

fn num(c: &Value, key: &str, default: f32) -> f32 {
    c.get(key).and_then(|v| v.float().ok()).unwrap_or(default)
}

fn flag(c: &Value, key: &str) -> bool {
    matches!(c.get(key), Some(Value::Bool(true)))
}

fn vec3(c: &Value, key: &str, default: Vec3) -> Vec3 {
    let Some(a) = c.get(key).and_then(|v| v.array().ok()) else {
        return default;
    };
    let f = |i: usize| a.get(i).and_then(|v| v.float().ok());
    match (f(0), f(1), f(2)) {
        (Some(x), Some(y), Some(z)) => Vec3::new(x, y, z),
        _ => default,
    }
}

fn quat(c: &Value, key: &str) -> Quat {
    let Some(a) = c.get(key).and_then(|v| v.array().ok()) else {
        return Quat::IDENTITY;
    };
    let f = |i: usize| a.get(i).and_then(|v| v.float().ok());
    match (f(0), f(1), f(2), f(3)) {
        (Some(x), Some(y), Some(z), Some(w)) => Quat::new(x, y, z, w).normalize(),
        _ => Quat::IDENTITY,
    }
}

fn axes(c: &Value, key: &str) -> [bool; 3] {
    let Some(o) = c.get(key) else {
        return [false; 3];
    };
    [flag(o, "x"), flag(o, "y"), flag(o, "z")]
}

fn bits(c: &Value, key: &str, default: u32) -> u32 {
    c.get(key)
        .and_then(|v| v.unsigned().ok())
        .map_or(default, |v| v.min(u32::MAX as u64) as u32)
}

/// The linear part of a column-major 4×4 matrix.
fn linear(m: &[f32; 16]) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(m[0], m[1], m[2]),
        Vec3::new(m[4], m[5], m[6]),
        Vec3::new(m[8], m[9], m[10]),
    )
}

/// The lengths of a linear map's columns, refusing a map that squashes some direction flat.
fn column_lengths(a: &Mat3) -> Result<Vec3, String> {
    let len = Vec3::new(a.cols[0].length(), a.cols[1].length(), a.cols[2].length());
    if !len.is_finite() || len.min_element() <= 1e-6 {
        return Err("a zero scale cannot have physics".into());
    }
    // The columns are finite here, so is the determinant.
    if a.determinant().abs() <= 1e-6 * len.x * len.y * len.z {
        return Err("its transform is flat, which physics cannot simulate".into());
    }
    Ok(len)
}

/// Splits a linear map into a rotation and a scale along the rotated axes: the rotation of its
/// normalized axes (exact without shear, close to it with some). A mirror (negative
/// determinant) has no rotation of its own, so first the axis that points most backwards is
/// turned round (the rotation left is then the smallest; ties within rounding go to the first
/// axis) and its scale becomes negative.
fn split(a: &Mat3) -> Result<(Quat, Vec3), String> {
    let len = column_lengths(a)?;
    let mut cols = a.cols;
    let mut scale = len;
    if a.determinant() < 0.0 {
        let back = -Vec3::new(cols[0].x / len.x, cols[1].y / len.y, cols[2].z / len.z);
        let most = back.max_element();
        let k = (0..3).find(|&k| back[k] >= most - 1e-6).unwrap_or(0);
        cols[k] = -cols[k];
        scale[k] = -scale[k];
    }
    let [c0, c1, c2] = cols;
    let folded = [
        c0.x, c0.y, c0.z, 0.0, c1.x, c1.y, c1.z, 0.0, c2.x, c2.y, c2.z, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    Ok((Iso::from_matrix(&folded).rotation, scale))
}

/// The pose of a body for an entity cooked at `m` (its translation, and its rotation as
/// [`split`] finds it), and the entity's scale along the body's axes.
fn body_frame(m: &[f32; 16]) -> Result<(Iso, Vec3), String> {
    let (rotation, scale) = split(&linear(m))?;
    Ok((Iso::new(Vec3::new(m[12], m[13], m[14]), rotation), scale))
}

/// How an entity cooked at `m` maps into the frame of a body at `pose`: points go to `a·p + t`.
fn relative(pose: &Iso, m: &[f32; 16]) -> (Mat3, Vec3) {
    let a = linear(m);
    let r = |v: Vec3| pose.rotation.inverse_rotate(v);
    (
        Mat3::from_cols(r(a.cols[0]), r(a.cols[1]), r(a.cols[2])),
        pose.inverse_point(Vec3::new(m[12], m[13], m[14])),
    )
}

/// `q` seen in the mirror that turns round the one axis `scale` has negative (if any): how a
/// turned box looks in a mirrored entity. Its axial part flips like a pseudovector: the component
/// along the mirrored axis stays, the other two change sign.
fn mirrored(q: Quat, scale: Vec3) -> Quat {
    match (scale.x < 0.0, scale.y < 0.0, scale.z < 0.0) {
        (true, _, _) => Quat::new(q.x, -q.y, -q.z, q.w),
        (_, true, _) => Quat::new(-q.x, q.y, -q.z, q.w),
        (_, _, true) => Quat::new(-q.x, -q.y, q.z, q.w),
        _ => q,
    }
}

/// Columns whose cosines are all within this are square: rounding, or a shear too small to matter
/// (it moves the corners of a box by a millimetre per metre at most).
const SQUARE: f32 = 1e-3;

/// How far from square `m`'s columns are: the largest |cosine| between two of them.
fn skew(m: &Mat3) -> f32 {
    let [x, y, z] = m.cols.map(Vec3::normalize_or_zero);
    x.dot(y).abs().max(x.dot(z).abs()).max(y.dot(z).abs())
}

/// Whether the box of half extents `half` turned by `turn` may stand for the drawn box whose half
/// edges are the columns of `e`, about the same centre: it is at most a quarter larger, the drawn
/// box lies inside it doubled and it lies inside the drawn box doubled.
fn close_to(e: &Mat3, turn: Quat, half: Vec3) -> bool {
    // False for NaN too.
    let small = half.x * half.y * half.z <= 1.25 * e.determinant().abs();
    if !small {
        return false;
    }
    let Some(to_drawn) = e.inverse() else {
        return false;
    };
    let r = turn.to_mat3();
    // How far the drawn box reaches along each of the box's axes, relative to the box.
    let drawn_along_box = (0..3).map(|i| {
        let axis = r.cols[i];
        (e.cols[0].dot(axis).abs() + e.cols[1].dot(axis).abs() + e.cols[2].dot(axis).abs())
            / half[i]
    });
    // How far the box reaches along each of the drawn box's own coordinates, in which the drawn
    // box is the cube [-1, 1]³.
    let w = to_drawn * Mat3::from_cols(r.cols[0] * half.x, r.cols[1] * half.y, r.cols[2] * half.z);
    let box_along_drawn = (0..3).map(|i| {
        let row = w.row(i).abs();
        row.x + row.y + row.z
    });
    drawn_along_box
        .chain(box_along_drawn)
        .all(|reach| reach <= 2.0)
}

/// The shape and turn of a box collider with half extents `half` and turn `rotation`, on an
/// entity that maps into its body's frame by `a` (split into `turn` and `scale`), placed at its
/// drawn centre (see the module notes): the box keeping its turn when the entity is not sheared
/// and that box is [`close_to`] the drawn one, otherwise exactly what is drawn, a box or the hull
/// of the drawn corners.
fn box_shape(
    a: &Mat3,
    turn: Quat,
    scale: Vec3,
    rotation: Quat,
    half: Vec3,
) -> Result<(Shape, Quat), String> {
    // The box keeping its turn, each edge as long as drawn: exactly what is drawn whenever the
    // scale runs along its axes. A mirror only changes how it sits.
    let q = mirrored(rotation, scale);
    let edge = |e: Vec3| q.rotate(e).mul_elem(scale.abs()).length();
    let kept_half = half.mul_elem(Vec3::new(edge(Vec3::X), edge(Vec3::Y), edge(Vec3::Z)));
    let kept = (
        Shape::Box {
            half_extents: kept_half,
        },
        turn * q,
    );
    // The drawn box: its half edges in the body's frame.
    let e = Mat3::from_cols(
        a.mul_vec(rotation.rotate(Vec3::X)) * half.x,
        a.mul_vec(rotation.rotate(Vec3::Y)) * half.y,
        a.mul_vec(rotation.rotate(Vec3::Z)) * half.z,
    );
    if skew(a) <= SQUARE && close_to(&e, kept.1, kept_half) {
        return Ok(kept);
    }
    let len = Vec3::new(e.cols[0].length(), e.cols[1].length(), e.cols[2].length());
    // Half the drawn box's thickness across each pair of its faces.
    let det = e.determinant().abs();
    let across = Vec3::new(
        det / e.cols[1].cross(e.cols[2]).length(),
        det / e.cols[2].cross(e.cols[0]).length(),
        det / e.cols[0].cross(e.cols[1]).length(),
    );
    if !(across.min_element() >= MIN_EXTENT && len.max_element() <= MAX_EXTENT) {
        return Err(format!(
            "sheared box half thickness {across:?} or edge lengths {len:?} outside the supported range"
        ));
    }
    if skew(&e) <= SQUARE {
        // Drawn square after all (the entity's own shear and the box's turn cancel out): the box
        // along the drawn edges.
        let x = e.cols[0] / len.x;
        let y = (e.cols[1] - x * x.dot(e.cols[1])).normalize_or_zero();
        let frame = Mat3::from_cols(x, y, x.cross(y));
        return Ok((Shape::Box { half_extents: len }, Quat::from_mat3(&frame)));
    }
    // Sheared: the hull of the eight drawn corners is exactly what is drawn.
    let hull = ConvexHull::parallelepiped(&e)?;
    Ok((Shape::Hull(Arc::new(hull)), Quat::IDENTITY))
}

/// A box, sphere or capsule collider `c` of an entity that maps into its body's frame by `a` and
/// `t`, and its placement on the body. Its `center` is placed exactly. Its size scales with the
/// entity along the collider's own axes after its `rotation`, which is exact whenever the scale
/// runs along those axes (any turn under an even scale, quarter turns under an uneven one, no
/// shear from a parent); otherwise the drawn shape is sheared: a box is exactly what is drawn
/// unless the box keeping its turn is close to it ([`box_shape`]); a capsule keeps its drawn axis
/// and takes the largest stretch across it; a ball always takes the largest stretch.
fn primitive(name: &str, c: &Value, a: &Mat3, t: Vec3) -> Result<(Shape, Iso), String> {
    let center = t + a.mul_vec(vec3(c, "center", Vec3::ZERO));
    let rotation = quat(c, "rotation");
    // The entity's own turn and scale in the body's frame. The shapes are symmetric, so a mirror
    // only changes how a turned box sits.
    let (turn, scale) = split(a)?;
    let stretch = scale.abs();
    Ok(match name {
        "box" => {
            let half = vec3(c, "half_extents", Vec3::splat(0.5)).abs();
            let (shape, turn) = box_shape(a, turn, scale, rotation, half)?;
            (shape, Iso::new(center, turn))
        }
        "sphere" => (
            Shape::Sphere {
                radius: num(c, "radius", 0.5) * stretch.max_element(),
            },
            Iso::new(center, turn),
        ),
        _ => {
            let drawn = Mat3::from_cols(
                a.mul_vec(rotation.rotate(Vec3::X)),
                a.mul_vec(rotation.rotate(Vec3::Y)),
                a.mul_vec(rotation.rotate(Vec3::Z)),
            );
            let len = column_lengths(&drawn)?;
            let y = drawn.cols[1] / len.y;
            let z = drawn.cols[2] - y * drawn.cols[2].dot(y);
            let z = z / z.length();
            (
                Shape::Capsule {
                    half_height: num(c, "half_height", 0.5) * len.y,
                    radius: num(c, "radius", 0.5) * len.x.max(len.z),
                },
                Iso::new(center, Quat::from_mat3(&Mat3::from_cols(y.cross(z), y, z))),
            )
        }
    })
}

/// The directions a detailed mesh is sampled along for its hull: the six axes (so the hull keeps
/// the mesh's bounds), then a Fibonacci spiral over the sphere, [`MAX_HULL_VERTICES`] in all.
fn hull_directions() -> Vec<Vec3> {
    let mut dirs = vec![Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
    let n = MAX_HULL_VERTICES - dirs.len();
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    for i in 0..n {
        let y = 1.0 - (2 * i + 1) as f32 / n as f32;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let (s, c) = sin_cos(i as f32 * golden_angle);
        dirs.push(Vec3::new(r * c, y, r * s));
    }
    dirs
}

/// The points a hull is made from: all of them when there are at most [`MAX_HULL_VERTICES`],
/// otherwise the farthest along each of [`hull_directions`] (the first of equals), in their
/// original order. Their hull then has at most that many vertices, comes out the same on every
/// platform and keeps the mesh's bounds; it lies within the mesh's hull, close to it.
fn hull_points(points: Vec<Vec3>) -> Vec<Vec3> {
    if points.len() <= MAX_HULL_VERTICES || points.iter().any(|p| !p.is_finite()) {
        return points;
    }
    let mut keep = vec![false; points.len()];
    for d in hull_directions() {
        let mut best = 0;
        let mut best_d = f32::NEG_INFINITY;
        for (i, p) in points.iter().enumerate() {
            let s = p.dot(d);
            if s > best_d {
                best_d = s;
                best = i;
            }
        }
        keep[best] = true;
    }
    points
        .into_iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(p))
        .collect()
}

/// Adds entity `e`'s collider `c` to `owner`'s body. `moves_with` names the owner's entity when
/// it is not `e` (the collider rides on an ancestor's body).
fn add_collider(
    physics: &mut PhysicsWorld,
    world: &World,
    owner: &SceneBody,
    moves_with: Option<&str>,
    e: &EntityRecord,
    c: &Value,
) -> Result<(), String> {
    let kind = physics.body_kind(owner.body).unwrap_or(BodyKind::Static);
    let shape_name = c
        .get("shape")
        .and_then(|s| s.string().ok())
        .unwrap_or("box");
    let (a, t) = relative(&owner.pose, &e.transform.0);
    let (shape, local) = match shape_name {
        "box" | "sphere" | "capsule" => primitive(shape_name, c, &a, t)?,
        "hull" | "mesh" => {
            column_lengths(&a)?;
            if shape_name == "mesh" && kind != BodyKind::Static {
                return Err(match moves_with {
                    None => "a mesh collider can only be static; use hull for moving bodies".into(),
                    Some(owner) => format!(
                        "a mesh collider can only be static, and this one moves with {owner}; \
                         use hull"
                    ),
                });
            }
            // The entity's own geometry, in the body's frame.
            let instances = world
                .instances
                .get(e.instances.start as usize..e.instances.end as usize)
                .ok_or("its instances are not in the world")?;
            let mut points = Vec::new();
            let mut triangles = Vec::new();
            for inst in instances {
                let mesh = world
                    .meshes
                    .get(inst.mesh as usize)
                    .ok_or("an instance's mesh is not in the world")?;
                let base = points.len() as u32;
                for p in &mesh.positions {
                    let wp = inst.transform.point(*p);
                    points.push(owner.pose.inverse_point(Vec3::new(wp.0, wp.1, wp.2)));
                }
                // A mirrored instance turns its triangles inside out: wind them the other way
                // round so the faces drawn as fronts stay fronts.
                let mirrored = linear(&inst.transform.0).determinant() < 0.0;
                for t in mesh.indices.chunks_exact(3) {
                    let [i, j, k] = [t[0], t[1], t[2]].map(|v| base + v);
                    triangles.push(if mirrored { [i, k, j] } else { [i, j, k] });
                }
            }
            if points.is_empty() {
                return Err(format!(
                    "a {shape_name} collider needs the entity to have a mesh"
                ));
            }
            let shape = if shape_name == "hull" {
                Shape::Hull(Arc::new(ConvexHull::from_points(&hull_points(points))?))
            } else {
                Shape::Mesh(Arc::new(TriangleMesh::new(points, triangles)?))
            };
            (shape, Iso::IDENTITY)
        }
        other => return Err(format!("unknown collider shape {other:?}")),
    };
    physics.add_collider(
        owner.body,
        &ColliderDesc {
            shape,
            local,
            density: num(c, "density", 1000.0),
            friction: num(c, "friction", 0.6),
            restitution: num(c, "restitution", 0.0),
            sensor: flag(c, "sensor"),
            layers: bits(c, "layers", 1),
            mask: bits(c, "mask", u32::MAX),
            user_data: e.id,
        },
    )?;
    Ok(())
}

/// Builds the physics of a cooked world from its entity table.
pub fn build(
    world: &World,
    table: &EntityTable,
    settings: Settings,
) -> Result<ScenePhysics, String> {
    let data = if table.data.is_empty() {
        Value::Object(Default::default())
    } else {
        json::parse(table.data.as_bytes())?
    };
    let entities = data.get("entities");
    let mut physics = PhysicsWorld::new(settings);
    let mut bodies: Vec<SceneBody> = Vec::new();
    // For each record so far, the body (index into `bodies`) of its nearest ancestor or self
    // with a `rigid_body`: the body that carries it. Parents come before their children.
    let mut carrier: Vec<Option<usize>> = Vec::with_capacity(table.entities.len());
    let none = Value::Object(Default::default());
    for (record, e) in table.entities.iter().enumerate() {
        let key = format!("{:016x}", e.id);
        let components = entities
            .and_then(|m| m.get(&key))
            .and_then(|v| v.get("components"));
        let rigid = components.and_then(|c| c.get("rigid_body"));
        let collider = components.and_then(|c| c.get("collider"));
        let inherited = e
            .parent
            .and_then(|p| carrier.get(p as usize).copied().flatten());
        let at = |m: String| format!("{} ({key}): {m}", e.name);
        if rigid.is_none() && (collider.is_none() || inherited.is_some()) {
            if let Some(i) = inherited {
                bodies[i].carries.push(record as u32);
                if let Some(c) = collider {
                    let owner = &table.entities[bodies[i].record as usize];
                    let with = format!("{} ({:016x})", owner.name, owner.id);
                    add_collider(&mut physics, world, &bodies[i], Some(&with), e, c).map_err(at)?;
                }
            }
            carrier.push(inherited);
            continue;
        }
        let (pose, scale) = body_frame(&e.transform.0).map_err(at)?;
        let kind = match rigid
            .and_then(|r| r.get("kind"))
            .and_then(|k| k.string().ok())
        {
            None if rigid.is_some() => BodyKind::Dynamic,
            None => BodyKind::Static,
            Some("dynamic") => BodyKind::Dynamic,
            Some("kinematic") => BodyKind::Kinematic,
            Some("static") => BodyKind::Static,
            Some(other) => return Err(at(format!("unknown body kind {other:?}"))),
        };
        let r = rigid.unwrap_or(&none);
        let body = physics
            .create_body(&BodyDesc {
                kind,
                position: pose.position,
                rotation: pose.rotation,
                linear_velocity: vec3(r, "velocity", Vec3::ZERO),
                angular_velocity: vec3(r, "angular_velocity", Vec3::ZERO),
                linear_damping: num(r, "linear_damping", 0.0),
                angular_damping: num(r, "angular_damping", 0.05),
                gravity_scale: num(r, "gravity_scale", 1.0),
                can_sleep: !flag(r, "never_sleeps"),
                awake: true,
                bullet: flag(r, "bullet"),
                lock_rotation: axes(r, "lock_rotation"),
                lock_translation: axes(r, "lock_translation"),
                user_data: e.id,
            })
            .map_err(at)?;
        bodies.push(SceneBody {
            record: record as u32,
            entity: e.id,
            body,
            scale,
            carries: vec![record as u32],
            pose: physics.body_pose(body).unwrap_or(pose),
            cooked: e.transform.0,
        });
        let index = bodies.len() - 1;
        if let Some(c) = collider {
            add_collider(&mut physics, world, &bodies[index], None, e, c).map_err(at)?;
        }
        // Static scenery (a collider with no body above it) carries only its own entity.
        carrier.push(rigid.is_some().then_some(index));
    }
    Ok(ScenePhysics {
        world: physics,
        bodies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ganbate_core::world::{EntityRecord, Instance, Mesh, Transform};

    fn record(id: u64, name: &str, t: [f32; 3], instances: std::ops::Range<u32>) -> EntityRecord {
        EntityRecord {
            id,
            name: name.into(),
            parent: None,
            transform: Transform::from_trs(t, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]).unwrap(),
            instances,
            lights: 0..0,
        }
    }

    #[test]
    fn components_become_bodies_and_instances_follow_them() {
        // A ground entity with a box collider (static) and a crate with a rigid body above it.
        let cube = Mesh {
            positions: vec![
                ganbate_core::Vec3(-0.5, -0.5, -0.5),
                ganbate_core::Vec3(0.5, 0.5, 0.5),
            ],
            indices: vec![],
            ..Default::default()
        };
        let world = World {
            meshes: vec![cube],
            instances: vec![Instance {
                mesh: 0,
                transform: Transform::from_trs([0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0; 3])
                    .unwrap(),
            }],
            ..Default::default()
        };
        let table = EntityTable {
            entities: vec![record(1, "ground", [0.0, -0.5, 0.0], 0..0), record(2, "crate", [0.0, 3.0, 0.0], 0..1)],
            data: r#"{"entities":{"0000000000000001":{"components":{"collider":{"half_extents":[20,0.5,20]}}},"0000000000000002":{"components":{"collider":{"shape":"box"},"rigid_body":{}}}}}"#.into(),
        };
        let mut s = build(&world, &table, Settings::default()).unwrap();
        assert_eq!(s.bodies.len(), 2);
        assert_eq!(s.world.body_kind(s.bodies[0].body), Some(BodyKind::Static));
        assert_eq!(s.world.body_kind(s.bodies[1].body), Some(BodyKind::Dynamic));
        // A cubic metre of water.
        assert!((s.world.mass(s.bodies[1].body).unwrap() - 1000.0).abs() < 1e-2);
        for _ in 0..30 {
            s.world.step(1.0 / 60.0);
        }
        let m = s
            .instance_matrix(1, &world.instances[0].transform.0)
            .unwrap();
        let y = s.world.body_pose(s.bodies[1].body).unwrap().position.y;
        assert!(y < 3.0, "the crate fell: {y}");
        assert!(
            (m[13] - y).abs() < 1e-5,
            "its instance follows: {} vs {y}",
            m[13]
        );
        // Destroyed, it has no place.
        s.world.destroy_body(s.bodies[1].body);
        assert_eq!(s.instance_matrix(1, &world.instances[0].transform.0), None);
        assert_eq!(s.entity_matrix(1), None);
    }

    #[test]
    fn bad_components_are_refused_with_the_entity_named() {
        let world = World::default();
        let table = EntityTable {
            entities: vec![record(7, "lamp post", [0.0, 0.0, 0.0], 0..0)],
            data:
                r#"{"entities":{"0000000000000007":{"components":{"collider":{"shape":"mesh"}}}}}"#
                    .into(),
        };
        let err = build(&world, &table, Settings::default()).err().unwrap();
        assert!(
            err.contains("lamp post") && err.contains("needs the entity to have a mesh"),
            "{err}"
        );
    }

    #[test]
    fn a_mirror_is_folded_into_the_axis_that_leaves_the_smallest_turn() {
        let m = |q: Quat, s: [f32; 3]| {
            Transform::from_trs([1.0, 2.0, 3.0], q.to_array(), s)
                .unwrap()
                .0
        };
        // A plain mirror keeps no turn at all, whichever axis it is on.
        for (k, s) in [[-2.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -2.0]]
            .into_iter()
            .enumerate()
        {
            let (pose, scale) = body_frame(&m(Quat::IDENTITY, s)).unwrap();
            assert_eq!(pose, Iso::new(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY));
            assert_eq!(scale[k], -2.0);
        }
        // Turned and mirrored: a proper rotation that with the signed scale gives the transform.
        let q = Quat::from_axis_angle(Vec3::new(1.0, -2.0, 0.5), 2.0);
        let cooked = m(q, [1.5, -0.5, 2.0]);
        let (pose, scale) = body_frame(&cooked).unwrap();
        let r = pose.rotation.to_mat3();
        assert!(r.determinant() > 0.0);
        for k in 0..3 {
            let want = linear(&cooked).cols[k];
            assert!((r.cols[k] * scale[k] - want).length() < 1e-5, "axis {k}");
        }
        // Zero and flat scales are refused.
        assert!(body_frame(&m(Quat::IDENTITY, [1.0, 0.0, 1.0])).is_err());
        let mut flat = m(Quat::IDENTITY, [1.0; 3]);
        flat[8..11].copy_from_slice(&[1.0, 1.0, 0.0]);
        flat[0..3].copy_from_slice(&[1.0, 1.0, 0.0]);
        assert!(body_frame(&flat).is_err());
    }

    /// A ball of `n × 2n` points (latitude, longitude) of radius 2 about (1, 0, -1).
    fn ball_points(n: usize) -> Vec<Vec3> {
        let mut out = Vec::new();
        for i in 0..=n {
            let (st, ct) = sin_cos(std::f32::consts::PI * i as f32 / n as f32);
            for j in 0..2 * n {
                let (sp, cp) = sin_cos(std::f32::consts::PI * j as f32 / n as f32);
                out.push(Vec3::new(
                    1.0 + 2.0 * st * cp,
                    2.0 * ct,
                    -1.0 + 2.0 * st * sp,
                ));
            }
        }
        out
    }

    #[test]
    fn a_detailed_hull_is_reduced_to_extreme_points_within_the_limit() {
        let points = ball_points(40);
        assert!(points.len() > 3000);
        let reduced = hull_points(points.clone());
        assert!(
            (100..=MAX_HULL_VERTICES).contains(&reduced.len()),
            "{} points kept",
            reduced.len()
        );
        // Every kept point is one of the mesh's, in the mesh's order.
        let mut from = 0;
        for p in &reduced {
            from += points[from..]
                .iter()
                .position(|q| q == p)
                .expect("an input point")
                + 1;
        }
        // The same bounds, and close to the full hull along any direction.
        let bounds = |v: &[Vec3]| crate::math::Aabb::from_points(v);
        assert_eq!(bounds(&reduced), bounds(&points));
        for d in [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-0.3, 0.9, 0.2),
            Vec3::new(0.7, -0.1, -0.7),
        ] {
            let d = d.normalize_or_zero();
            let reach = |v: &[Vec3]| v.iter().fold(f32::NEG_INFINITY, |m, p| m.max(p.dot(d)));
            let (full, kept) = (reach(&points), reach(&reduced));
            assert!(
                kept <= full && full - kept < 0.03,
                "{d:?}: {kept} of {full}"
            );
        }
        // The same every time; and few points pass through untouched.
        assert_eq!(hull_points(points), reduced);
        let few = ball_points(8);
        assert!(few.len() <= MAX_HULL_VERTICES);
        assert_eq!(hull_points(few.clone()), few);
    }
}
