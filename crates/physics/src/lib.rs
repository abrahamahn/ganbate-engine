//! Ganbate's rigid-body physics, with no dependencies beyond `ganbate-core`.
//!
//! - [`math`]: vectors, quaternions, matrices, rigid transforms, boxes.
//! - [`shape`]: spheres, capsules, boxes, convex hulls ([`hull`]), triangle meshes and
//!   heightfields ([`mesh`]); their bounds and mass.
//! - [`collide`]: contact manifolds between shapes; [`gjk`]: distance and casts; [`ray`]: rays.
//! - [`tree`]: the dynamic bounding-volume tree behind the broad phase and queries.
//! - [`world`]: bodies, colliders, joints ([`joint`]), contacts and the step; [`solver`]: the
//!   soft-step constraint solver; [`query`]: rays, shape casts and overlaps on a world.
//! - [`scene`]: bodies and colliders from a cooked scene's `rigid_body` and `collider` components.
//! - [`character`]: a kinematic capsule that walks, slides, climbs steps and pushes bodies.
//!
//! A simulation steps in fixed time slices and gives the same bits for the same inputs on every
//! platform (see [`math`]).

#![forbid(unsafe_code)]

mod ccd;
pub mod character;
pub mod collide;
pub mod gjk;
pub mod hull;
pub mod joint;
pub mod math;
pub mod mesh;
pub mod query;
pub mod ray;
pub mod scene;
pub mod shape;
pub(crate) mod small_buf;
pub mod solver;
pub mod tree;
pub mod world;

pub use character::{Character, CharacterSettings, MoveResult};
pub use joint::{
    DistanceOptions, HingeOptions, JointDesc, JointKind, Motor, SliderOptions, Spring,
};
pub use math::{Aabb, Iso, Mat3, Quat, Vec3};
pub use query::{RaycastHit, ShapeCastHit};
pub use shape::{Convex, ConvexShape, MassProperties, Shape};
pub use world::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, ColliderId, ContactEvent, Events, JointId,
    PhysicsWorld, QueryFilter, SensorEvent, Settings,
};
