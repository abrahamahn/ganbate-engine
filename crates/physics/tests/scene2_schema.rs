//! The engine schema's `rigid_body` and `collider` docs (what Ganbate Studio and the agent tools show
//! authors) against what `ganbate_physics::scene::build` does with the components, and the schema
//! file as canonical text (editors and tools compare it byte for byte).

use ganbate_core::json::{self, Value};
use ganbate_core::world::{EntityRecord, EntityTable, Instance, Mesh, Transform, World};
use ganbate_physics::scene;
use ganbate_physics::{BodyKind, ColliderId, Settings};

const SCHEMA: &str = include_str!("../../scene/src/engine.schema.json");

fn schema() -> Value {
    json::parse(SCHEMA.as_bytes()).expect("the engine schema parses")
}

/// The `doc` of component `kind`, or of its field `field`.
fn doc(kind: &str, field: Option<&str>) -> String {
    let s = schema();
    let mut v = s
        .get("components")
        .and_then(|c| c.get(kind))
        .unwrap_or_else(|| panic!("no component {kind}"));
    if let Some(f) = field {
        v = v
            .get("fields")
            .and_then(|fs| fs.get(f))
            .unwrap_or_else(|| panic!("no field {kind}.{f}"));
    }
    v.get("doc")
        .and_then(|d| d.string().ok())
        .unwrap_or_else(|| panic!("{kind} {field:?} has no doc"))
        .to_string()
}

#[test]
fn the_schema_is_canonical_text() {
    let canonical = json::to_canonical(&schema()).unwrap();
    assert!(
        canonical == SCHEMA,
        "engine.schema.json is not what ganbate_core::json::to_canonical writes; first difference at \
         byte {}",
        canonical
            .bytes()
            .zip(SCHEMA.bytes())
            .position(|(a, b)| a != b)
            .unwrap_or(canonical.len().min(SCHEMA.len()))
    );
}

fn record(id: u64, parent: Option<u32>, instances: std::ops::Range<u32>) -> EntityRecord {
    EntityRecord {
        id,
        name: format!("entity {id}"),
        parent,
        transform: Transform::IDENTITY,
        instances,
        lights: 0..0,
    }
}

#[test]
fn a_collider_without_a_body_is_scenery_only_when_no_ancestor_has_one() {
    // What the builder does: a collider on a child of a dynamic body is part of that body.
    let s = scene::build(
        &World::default(),
        &EntityTable {
            entities: vec![record(1, None, 0..0), record(2, Some(0), 0..0)],
            data: r#"{"entities":{"0000000000000001":{"components":{"rigid_body":{}}},"0000000000000002":{"components":{"collider":{}}}}}"#.into(),
        },
        Settings::default(),
    )
    .unwrap();
    let wheel = ColliderId {
        index: 0,
        generation: 0,
    };
    let body = s.world.collider_body(wheel).unwrap();
    assert_eq!(s.world.body_kind(body), Some(BodyKind::Dynamic));
    // What the docs say.
    let rigid = doc("rigid_body", None);
    let collider = doc("collider", None);
    for (what, text) in [("rigid_body", &rigid), ("collider", &collider)] {
        assert!(
            !text.contains("Without it, an entity with a collider is static scenery"),
            "{what}: {text}"
        );
        assert!(
            text.contains("nearest body above it") && text.contains("static scenery"),
            "{what} should say a collider without a body is part of the nearest body above it, \
             and scenery only with none: {text}"
        );
    }
}

#[test]
fn mesh_colliders_are_documented_where_the_builder_takes_them() {
    // The builder refuses a mesh collider that would move with a body above it...
    let floor = Mesh {
        positions: vec![
            ganbate_core::Vec3(-1.0, 0.0, -1.0),
            ganbate_core::Vec3(1.0, 0.0, -1.0),
            ganbate_core::Vec3(0.0, 0.0, 1.0),
        ],
        indices: vec![0, 2, 1],
        ..Default::default()
    };
    let world = World {
        meshes: vec![floor],
        instances: vec![Instance {
            mesh: 0,
            transform: Transform::IDENTITY,
        }],
        ..Default::default()
    };
    let table = |kind: &str| EntityTable {
        entities: vec![record(1, None, 0..0), record(2, Some(0), 0..1)],
        data: format!(
            r#"{{"entities":{{"0000000000000001":{{"components":{{"rigid_body":{{"kind":"{kind}"}}}}}},"0000000000000002":{{"components":{{"collider":{{"shape":"mesh"}}}}}}}}}}"#
        ),
    };
    assert!(scene::build(&world, &table("kinematic"), Settings::default()).is_err());
    // ...and takes one on a static body, or as scenery.
    assert!(scene::build(&world, &table("static"), Settings::default()).is_ok());
    let alone = EntityTable {
        entities: vec![record(1, None, 0..1)],
        data: r#"{"entities":{"0000000000000001":{"components":{"collider":{"shape":"mesh"}}}}}"#
            .into(),
    };
    assert!(scene::build(&world, &alone, Settings::default()).is_ok());
    // The docs say so, where the shape is described and where it is picked.
    for text in [doc("collider", None), doc("collider", Some("shape"))] {
        assert!(!text.contains("mesh: static scenery only"), "{text}");
        assert!(
            text.contains("mesh only on a static body or as scenery"),
            "mesh colliders are allowed on static bodies too, and only there: {text}"
        );
    }
}
