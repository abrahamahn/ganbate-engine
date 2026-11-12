//! Component schemas: the shape of each kind of component, so an editor can draw an inspector
//! for it, a cook can refuse bad data, and a game can add its own kinds without engine code.
//!
//! A schema file is JSON:
//!
//! ```json
//! { "ganbate": "schema", "version": 1,
//!   "components": {
//!     "arena.spawn": { "label": "Spawn point", "doc": "Where a player enters.",
//!       "fields": { "index": { "type": "int", "min": 0, "max": 10, "default": 0 },
//!                   "facing": { "type": "float", "unit": "deg", "default": 0 } } } } }
//! ```
//!
//! Field types: `bool`, `int` and `float` (optional `min`, `max`, `step`, `unit`), `string`,
//! `enum` (`values`), `vec2`, `vec3`, `vec4`, `quat`, `rgb` and `rgba` (linear colour), `asset`
//! (an asset id; `kind` names the asset kind it accepts), `entity` (an entity id), `array`
//! (`items` gives the element field, optional `max`) and `object` (`fields`). Every field may
//! have `label`, `doc`, `default`, `hidden` and `readonly`. Stored components are sparse: a
//! missing field reads as its default.
//!
//! Component kinds without a dot (`transform`) belong to the engine; a game's kinds carry its
//! namespace (`arena.spawn`).
use std::collections::BTreeMap;
use std::sync::OnceLock;

use ganbate_core::id::{is_asset_id, is_entity_id};
use ganbate_core::json::Value;

#[derive(Clone, Debug, PartialEq)]
pub enum FieldType {
    Bool,
    Int {
        min: Option<f64>,
        max: Option<f64>,
    },
    Float {
        min: Option<f64>,
        max: Option<f64>,
    },
    String,
    Enum(Vec<String>),
    /// Vectors and colours: `len` numbers. Quaternions are unit-length x, y, z, w.
    Numbers {
        len: usize,
        quat: bool,
        color: bool,
    },
    Asset {
        kind: Option<String>,
    },
    Entity,
    Array {
        items: Box<Field>,
        max: Option<usize>,
    },
    Object(BTreeMap<String, Field>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub ty: FieldType,
    pub default: Value,
    /// Everything else the file said (label, doc, unit, step, hidden...), for editors.
    pub meta: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComponentSchema {
    pub name: String,
    pub fields: BTreeMap<String, Field>,
    pub meta: BTreeMap<String, Value>,
}

/// Every component kind a project knows, from the engine's schema and the game's.
#[derive(Clone, Debug, Default)]
pub struct Registry {
    components: BTreeMap<String, ComponentSchema>,
    /// The schema files' texts as loaded, for sending to editors.
    sources: Vec<Value>,
}

/// A problem found in data: where, and what.
#[derive(Clone, Debug, PartialEq)]
pub struct Issue {
    pub path: String,
    pub message: String,
}

fn num(v: Option<&Value>) -> Result<Option<f64>, String> {
    match v {
        None => Ok(None),
        Some(Value::Number(n)) => Ok(Some(*n)),
        Some(_) => Err("expected a number".into()),
    }
}

pub fn valid_kind_name(name: &str, engine: bool) -> bool {
    let ok_char = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '.';
    !name.is_empty()
        && name.len() <= 64
        && name.chars().all(ok_char)
        && !name.starts_with('.')
        && !name.ends_with('.')
        && !name.contains("..")
        && name.contains('.') != engine
}

fn parse_field(v: &Value, at: &str) -> Result<Field, String> {
    let err = |m: &str| format!("{at}: {m}");
    let Value::Object(map) = v else {
        return Err(err("a field is an object"));
    };
    let ty_name = map
        .get("type")
        .and_then(|t| {
            if let Value::String(s) = t {
                Some(s.as_str())
            } else {
                None
            }
        })
        .ok_or_else(|| err("missing type"))?;
    let numbers = |len, quat, color| FieldType::Numbers { len, quat, color };
    let ty = match ty_name {
        "bool" => FieldType::Bool,
        "int" => FieldType::Int {
            min: num(map.get("min")).map_err(|e| err(&e))?,
            max: num(map.get("max")).map_err(|e| err(&e))?,
        },
        "float" => FieldType::Float {
            min: num(map.get("min")).map_err(|e| err(&e))?,
            max: num(map.get("max")).map_err(|e| err(&e))?,
        },
        "string" => FieldType::String,
        "enum" => {
            let values = map
                .get("values")
                .and_then(|v| v.array().ok())
                .ok_or_else(|| err("an enum needs values"))?
                .iter()
                .map(|v| v.string().map(String::from))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| err(&e))?;
            if values.is_empty() {
                return Err(err("an enum needs values"));
            }
            FieldType::Enum(values)
        }
        "vec2" => numbers(2, false, false),
        "vec3" => numbers(3, false, false),
        "vec4" => numbers(4, false, false),
        "quat" => numbers(4, true, false),
        "rgb" => numbers(3, false, true),
        "rgba" => numbers(4, false, true),
        "asset" => FieldType::Asset {
            kind: match map.get("kind") {
                Some(Value::String(s)) => Some(s.clone()),
                None => None,
                Some(_) => return Err(err("kind is a string")),
            },
        },
        "entity" => FieldType::Entity,
        "array" => FieldType::Array {
            items: Box::new(parse_field(
                map.get("items")
                    .ok_or_else(|| err("an array needs items"))?,
                &format!("{at}[]"),
            )?),
            max: match map.get("max") {
                Some(v) => Some(v.integer().map_err(|e| err(&e))?),
                None => None,
            },
        },
        "object" => FieldType::Object(parse_fields(
            map.get("fields")
                .ok_or_else(|| err("an object needs fields"))?,
            at,
        )?),
        other => return Err(err(&format!("unknown type {other:?}"))),
    };
    let mut field = Field {
        default: Value::Null,
        ty,
        meta: map
            .iter()
            .filter(|(k, _)| !matches!(k.as_str(), "type" | "default" | "fields" | "items"))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    };
    let explicit = map.get("default");
    field.default = explicit
        .cloned()
        .unwrap_or_else(|| implicit_default(&field));
    // Every stored component that lacks the field reads its default, so the default must be a
    // value the field accepts, whether the file gave it or not.
    let mut issues = vec![];
    check_value(
        &field,
        &field.default,
        &format!("{at}.default"),
        &mut issues,
    );
    if let Some(i) = issues.first() {
        return Err(match explicit {
            Some(_) => format!("{}: {}", i.path, i.message),
            None => format!("{at}: no default fits the field ({}); give one", i.message),
        });
    }
    Ok(field)
}

fn parse_fields(v: &Value, at: &str) -> Result<BTreeMap<String, Field>, String> {
    let Value::Object(map) = v else {
        return Err(format!("{at}: fields is an object"));
    };
    let mut out = BTreeMap::new();
    for (name, f) in map {
        if !valid_kind_name(name, true) {
            return Err(format!("{at}: invalid field name {name:?}"));
        }
        out.insert(name.clone(), parse_field(f, &format!("{at}.{name}"))?);
    }
    Ok(out)
}

/// Zero moved into `[min, max]` (when no number fits, a value outside it, which `parse_field`
/// then refuses).
fn clamp_zero(min: Option<f64>, max: Option<f64>) -> f64 {
    let low = min.map_or(0.0, |m| m.max(0.0));
    max.map_or(low, |m| low.min(m))
}

fn implicit_default(f: &Field) -> Value {
    let zeros = |n: usize| Value::Array(vec![Value::Number(0.0); n]);
    match &f.ty {
        FieldType::Bool => Value::Bool(false),
        // The whole number nearest zero within the range.
        FieldType::Int { min, max } => {
            Value::Number(clamp_zero(min.map(f64::ceil), max.map(f64::floor)))
        }
        FieldType::Float { min, max } => Value::Number(clamp_zero(*min, *max)),
        FieldType::String => Value::String(String::new()),
        FieldType::Enum(values) => Value::String(values[0].clone()),
        FieldType::Numbers {
            len, quat: true, ..
        } => {
            let mut v = vec![Value::Number(0.0); *len];
            v[3] = Value::Number(1.0);
            Value::Array(v)
        }
        FieldType::Numbers {
            len, color: true, ..
        } => Value::Array(vec![Value::Number(1.0); *len]),
        FieldType::Numbers { len, .. } => zeros(*len),
        FieldType::Asset { .. } | FieldType::Entity => Value::Null,
        FieldType::Array { .. } => Value::Array(vec![]),
        FieldType::Object(fields) => Value::Object(
            fields
                .iter()
                .map(|(k, f)| (k.clone(), f.default.clone()))
                .collect(),
        ),
    }
}

/// Checks `v` against `field`, appending problems to `issues`.
pub fn check_value(field: &Field, v: &Value, at: &str, issues: &mut Vec<Issue>) {
    let mut bad = |m: String| {
        issues.push(Issue {
            path: at.to_string(),
            message: m,
        })
    };
    let range = |n: f64, min: &Option<f64>, max: &Option<f64>| {
        min.is_none_or(|lo| n >= lo) && max.is_none_or(|hi| n <= hi)
    };
    match (&field.ty, v) {
        (FieldType::Bool, Value::Bool(_)) => {}
        (FieldType::Int { min, max }, Value::Number(n)) => {
            if n.fract() != 0.0 || n.abs() > 9_007_199_254_740_991.0 {
                bad(format!("{n} is not an integer"));
            } else if !range(*n, min, max) {
                bad(format!("{n} is out of range"));
            }
        }
        (FieldType::Float { min, max }, Value::Number(n)) => {
            if !range(*n, min, max) {
                bad(format!("{n} is out of range"));
            }
        }
        (FieldType::String, Value::String(_)) => {}
        (FieldType::Enum(values), Value::String(s)) => {
            if !values.contains(s) {
                bad(format!("{s:?} is not one of {values:?}"));
            }
        }
        (FieldType::Numbers { len, quat, .. }, Value::Array(a)) => {
            if a.len() != *len || !a.iter().all(|x| matches!(x, Value::Number(_))) {
                bad(format!("expected {len} numbers"));
            } else if *quat {
                let q: f64 = a
                    .iter()
                    .map(|x| if let Value::Number(n) = x { n * n } else { 0.0 })
                    .sum();
                if (q.sqrt() - 1.0).abs() > 1e-3 {
                    bad("a rotation quaternion has unit length".into());
                }
            }
        }
        (FieldType::Asset { .. }, Value::Null) | (FieldType::Entity, Value::Null) => {}
        (FieldType::Asset { .. }, Value::String(s)) => {
            if !is_asset_id(s) {
                bad(format!("{s:?} is not an asset id"));
            }
        }
        (FieldType::Entity, Value::String(s)) => {
            if !is_entity_id(s) {
                bad(format!("{s:?} is not an entity id"));
            }
        }
        (FieldType::Array { items, max }, Value::Array(a)) => {
            if max.is_some_and(|m| a.len() > m) {
                bad(format!("more than {} items", max.unwrap_or(0)));
            }
            for (i, x) in a.iter().enumerate() {
                check_value(items, x, &format!("{at}[{i}]"), issues);
            }
        }
        (FieldType::Object(fields), Value::Object(map)) => {
            check_fields(fields, map, at, issues);
        }
        (ty, other) => bad(format!(
            "expected {}, found {}",
            type_name(ty),
            value_name(other)
        )),
    }
}

fn check_fields(
    fields: &BTreeMap<String, Field>,
    map: &BTreeMap<String, Value>,
    at: &str,
    issues: &mut Vec<Issue>,
) {
    for (k, v) in map {
        match fields.get(k) {
            Some(f) => check_value(f, v, &format!("{at}.{k}"), issues),
            None => issues.push(Issue {
                path: format!("{at}.{k}"),
                message: "no such field".into(),
            }),
        }
    }
}

fn type_name(t: &FieldType) -> &'static str {
    match t {
        FieldType::Bool => "a boolean",
        FieldType::Int { .. } => "an integer",
        FieldType::Float { .. } => "a number",
        FieldType::String | FieldType::Enum(_) => "a string",
        FieldType::Numbers { .. } | FieldType::Array { .. } => "an array",
        FieldType::Asset { .. } => "an asset id",
        FieldType::Entity => "an entity id",
        FieldType::Object(_) => "an object",
    }
}
fn value_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

impl Registry {
    /// Adds the component kinds of one schema file. `engine` files may only define names
    /// without a dot; a game's only names with one.
    pub fn add(&mut self, file: &Value, engine: bool) -> Result<(), String> {
        if file.get("ganbate").and_then(|v| v.string().ok()) != Some("schema") {
            return Err("not a schema file (\"ganbate\": \"schema\")".into());
        }
        if file.get("version").and_then(|v| v.unsigned().ok()) != Some(1) {
            return Err("unsupported schema version".into());
        }
        let Some(Value::Object(kinds)) = file.get("components") else {
            return Err("a schema file has components".into());
        };
        let mut added = BTreeMap::new();
        for (name, def) in kinds {
            if !valid_kind_name(name, engine) {
                return Err(format!(
                    "{name:?}: {}",
                    if engine {
                        "engine component names have no dot"
                    } else {
                        "a game's component names carry its namespace (game.kind)"
                    }
                ));
            }
            if self.components.contains_key(name) || added.contains_key(name) {
                return Err(format!("{name:?} is defined twice"));
            }
            let Value::Object(map) = def else {
                return Err(format!("{name}: a component is an object"));
            };
            let fields = parse_fields(
                map.get("fields").unwrap_or(&Value::Object(BTreeMap::new())),
                name,
            )?;
            let meta = map
                .iter()
                .filter(|(k, _)| k.as_str() != "fields")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            added.insert(
                name.clone(),
                ComponentSchema {
                    name: name.clone(),
                    fields,
                    meta,
                },
            );
        }
        self.components.extend(added);
        self.sources.push(file.clone());
        Ok(())
    }

    pub fn get(&self, kind: &str) -> Option<&ComponentSchema> {
        self.components.get(kind)
    }
    pub fn kinds(&self) -> impl Iterator<Item = &str> {
        self.components.keys().map(String::as_str)
    }
    pub fn sources(&self) -> &[Value] {
        &self.sources
    }

    /// A component of `kind` with every field at its default.
    pub fn default_component(&self, kind: &str) -> Option<Value> {
        self.get(kind).map(|c| {
            Value::Object(
                c.fields
                    .iter()
                    .map(|(k, f)| (k.clone(), f.default.clone()))
                    .collect(),
            )
        })
    }

    /// Checks a stored (sparse) component; unknown kinds are an issue.
    pub fn check_component(&self, kind: &str, v: &Value, at: &str) -> Vec<Issue> {
        let mut issues = vec![];
        match (self.get(kind), v) {
            (None, _) => issues.push(Issue {
                path: at.into(),
                message: format!("unknown component kind {kind:?}"),
            }),
            (Some(c), Value::Object(map)) => check_fields(&c.fields, map, at, &mut issues),
            (Some(_), _) => issues.push(Issue {
                path: at.into(),
                message: "a component is an object".into(),
            }),
        }
        issues
    }

    /// Reads field `field` of a stored component, falling back to the schema default.
    pub fn field<'a>(&'a self, kind: &str, stored: &'a Value, field: &str) -> Option<&'a Value> {
        stored
            .get(field)
            .or_else(|| self.get(kind)?.fields.get(field).map(|f| &f.default))
    }
}

/// The engine's own component kinds.
pub const ENGINE_SCHEMA: &str = include_str!("engine.schema.json");

/// A registry holding the engine's component kinds.
pub fn engine_registry() -> Registry {
    static BASE: OnceLock<Registry> = OnceLock::new();
    BASE.get_or_init(|| {
        let mut r = Registry::default();
        let file =
            ganbate_core::json::parse(ENGINE_SCHEMA.as_bytes()).expect("engine schema parses");
        r.add(&file, true).expect("engine schema is valid");
        r
    })
    .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ganbate_core::json::parse;

    fn game() -> Value {
        parse(
            br#"{"ganbate":"schema","version":1,"components":{
              "test.seat":{"label":"Seat","fields":{
                "index":{"type":"int","min":0,"max":10},
                "side":{"type":"enum","values":["left","right"]},
                "tint":{"type":"rgb"},
                "anchor":{"type":"entity"},
                "art":{"type":"asset","kind":"model"},
                "marks":{"type":"array","items":{"type":"vec2"},"max":3},
                "pose":{"type":"object","fields":{"lean":{"type":"float","min":-1,"max":1,"default":0.25}}}}}}}"#,
        )
        .unwrap()
    }

    #[test]
    fn engine_schema_loads_and_has_defaults() {
        let r = engine_registry();
        let t = r.default_component("transform").unwrap();
        assert_eq!(
            ganbate_core::json::to_compact(&t).unwrap(),
            r#"{"rotation": [0, 0, 0, 1], "scale": [1, 1, 1], "translation": [0, 0, 0]}"#
        );
        for kind in ["transform", "mesh", "light", "camera"] {
            assert!(r.get(kind).is_some(), "{kind}");
        }
    }

    #[test]
    fn game_kinds_need_a_namespace_and_engine_kinds_none() {
        let mut r = engine_registry();
        let bad = parse(br#"{"ganbate":"schema","version":1,"components":{"seat":{"fields":{}}}}"#)
            .unwrap();
        assert!(r.add(&bad, false).is_err());
        assert!(r.add(&game(), false).is_ok());
        assert!(r.add(&game(), false).is_err(), "defined twice");
    }

    #[test]
    fn values_are_checked_against_fields() {
        let mut r = Registry::default();
        r.add(&game(), false).unwrap();
        let ok = parse(br#"{"index":3,"side":"left","tint":[1,0.5,0],"anchor":"00000000000000ab","art":null,"marks":[[0,1]],"pose":{"lean":-0.5}}"#).unwrap();
        assert_eq!(r.check_component("test.seat", &ok, "c"), vec![]);
        let bad = parse(br#"{"index":3.5,"side":"up","tint":[1,0],"anchor":"zz","marks":[[0,1],[1,2],[2,3],[3,4]],"pose":{"lean":2},"extra":1}"#).unwrap();
        let issues = r.check_component("test.seat", &bad, "c");
        let paths: Vec<_> = issues.iter().map(|i| i.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "c.anchor",
                "c.extra",
                "c.index",
                "c.marks",
                "c.pose.lean",
                "c.side",
                "c.tint"
            ]
        );
        assert_eq!(r.check_component("test.none", &ok, "c").len(), 1);
        let pose = r.default_component("test.seat").unwrap();
        assert_eq!(
            pose.get("pose").and_then(|p| p.get("lean")),
            Some(&Value::Number(0.25))
        );
        let sparse = parse(br#"{"index":2}"#).unwrap();
        assert_eq!(
            r.field("test.seat", &sparse, "side"),
            Some(&Value::String("left".into()))
        );
    }

    #[test]
    fn a_bad_default_is_refused() {
        let mut r = Registry::default();
        let f = parse(br#"{"ganbate":"schema","version":1,"components":{"test.x":{"fields":{"a":{"type":"int","max":3,"default":9}}}}}"#).unwrap();
        assert!(r.add(&f, false).unwrap_err().contains("out of range"));
    }

    #[test]
    fn implicit_defaults_fit_their_field() {
        let mut r = Registry::default();
        let f = parse(
            br#"{"ganbate":"schema","version":1,"components":{"test.x":{"fields":{
              "below":{"type":"float","min":-5,"max":-1},
              "half":{"type":"int","min":0.5},
              "low":{"type":"int","max":-2.5},
              "around":{"type":"int","min":-3,"max":3},
              "above":{"type":"float","min":2}}}}}"#,
        )
        .unwrap();
        r.add(&f, false).unwrap();
        let d = r.default_component("test.x").unwrap();
        assert_eq!(
            ganbate_core::json::to_compact(&d).unwrap(),
            r#"{"above": 2, "around": 0, "below": -1, "half": 1, "low": -3}"#
        );
        assert_eq!(r.check_component("test.x", &d, "c"), vec![]);
        // A range no value fits leaves nothing to fall back on: the schema is refused.
        for field in [
            r#"{"type":"int","min":0.25,"max":0.75}"#,
            r#"{"type":"float","min":2,"max":1}"#,
            r#"{"type":"int","min":1e300}"#,
        ] {
            let f = parse(
                format!(r#"{{"ganbate":"schema","version":1,"components":{{"test.y":{{"fields":{{"n":{field}}}}}}}}}"#)
                    .as_bytes(),
            )
            .unwrap();
            let e = Registry::default().add(&f, false).unwrap_err();
            assert!(e.contains("test.y.n: no default fits"), "{field}: {e}");
        }
    }
}
