//! Edits to a graph: a patch is a list of operations applied in order. As JSON, an array of
//! objects:
//!
//! ```text
//! {"op": "put", "node": {...}}                      add a node, or replace the one with its id
//! {"op": "remove", "id": "cup"}                     delete a node and its relations
//! {"op": "set", "id": "cup", "key": "position", "value": [0, 0.76, 0]}
//!                                                   set one field (null deletes an extra field)
//! {"op": "place", "id": "cup", "rel": "on_top_of", "to": "desk", "anchor": "left"}
//!                                                   stand a node by a relation; the solver finds where
//! {"op": "face", "id": "person", "to": "desk"}      turn a node toward another (null: stop facing)
//! {"op": "graph", "key": "environment", "value": {...}}
//!                                                   set a section of the graph's own data
//! ```
//!
//! A field an operation does not know is refused, so a misspelt edit fails instead of vanishing.
use std::collections::BTreeMap;

use ganbate_core::json::{self, Value};

use crate::{ANCHORS, Bounds, Edge, Graph, Node, Relation};

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Put(Node),
    Remove(String),
    Set {
        id: String,
        key: String,
        value: Value,
    },
    Place {
        id: String,
        rel: Relation,
        to: String,
        anchor: Option<String>,
        clearance: Option<f32>,
    },
    Face {
        id: String,
        to: Option<String>,
    },
    Graph {
        key: String,
        value: Value,
    },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Patch(pub Vec<Op>);

/// What a patch changed: nodes it moved, turned or reshaped (`moved`), nodes it placed by a
/// relation (`placed`, which the solver grounds), and nodes that lost or changed what they stand
/// on (`dropped`, which settle by gravity).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Applied {
    pub moved: Vec<String>,
    pub placed: Vec<String>,
    pub dropped: Vec<String>,
}

fn add(list: &mut Vec<String>, id: &str) {
    if !list.iter().any(|m| m == id) {
        list.push(id.to_owned());
    }
}

fn op_fields(v: &Value, allowed: &[&str]) -> Result<BTreeMap<String, Value>, String> {
    let Value::Object(m) = v else {
        return Err("an operation must be an object".into());
    };
    match m.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => Err(format!(
            "operation \"{}\" has an unknown field \"{k}\"",
            op_name(m)
        )),
        None => Ok(m.clone()),
    }
}

fn op_name(m: &BTreeMap<String, Value>) -> &str {
    m.get("op").and_then(|o| o.string().ok()).unwrap_or("?")
}

fn required(m: &BTreeMap<String, Value>, key: &str) -> Result<String, String> {
    match m.get(key) {
        Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
        _ => Err(format!(
            "operation \"{}\" needs a nonempty \"{key}\"",
            op_name(m)
        )),
    }
}

impl Op {
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let kind = v
            .get("op")
            .and_then(|o| o.string().ok())
            .unwrap_or_default();
        Ok(match kind {
            "put" => {
                let m = op_fields(v, &["op", "node"])?;
                Self::Put(Node::from_json(m.get("node").unwrap_or(&Value::Null))?)
            }
            "remove" => Self::Remove(required(&op_fields(v, &["op", "id"])?, "id")?),
            "set" => {
                let m = op_fields(v, &["op", "id", "key", "value"])?;
                let value = m
                    .get("value")
                    .cloned()
                    .ok_or("operation \"set\" needs \"value\"")?;
                Self::Set {
                    id: required(&m, "id")?,
                    key: required(&m, "key")?,
                    value,
                }
            }
            "place" => {
                let m = op_fields(v, &["op", "id", "rel", "to", "anchor", "clearance"])?;
                let rel = Relation::parse(&required(&m, "rel")?)?;
                if !rel.places() {
                    return Err("\"place\" takes a placing relation; use \"face\"".into());
                }
                let anchor = m
                    .get("anchor")
                    .map(|a| a.string().map(str::to_owned))
                    .transpose()?;
                if let Some(a) = &anchor {
                    if rel != Relation::OnTopOf || !ANCHORS.contains(&a.as_str()) {
                        return Err(format!(
                            "anchor \"{a}\" needs on_top_of and one of {ANCHORS:?}"
                        ));
                    }
                }
                let clearance = m.get("clearance").map(Value::float).transpose()?;
                if clearance.is_some_and(|c| c < crate::spatial::MIN_GAP) {
                    return Err(format!(
                        "clearance must be at least {} m",
                        crate::spatial::MIN_GAP
                    ));
                }
                Self::Place {
                    id: required(&m, "id")?,
                    rel,
                    to: required(&m, "to")?,
                    anchor,
                    clearance,
                }
            }
            "face" => {
                let m = op_fields(v, &["op", "id", "to"])?;
                let to = match m.get("to") {
                    None | Some(Value::Null) => None,
                    Some(_) => Some(required(&m, "to")?),
                };
                Self::Face {
                    id: required(&m, "id")?,
                    to,
                }
            }
            "graph" => {
                let m = op_fields(v, &["op", "key", "value"])?;
                let key = required(&m, "key")?;
                if matches!(key.as_str(), "schema" | "nodes" | "edges") {
                    return Err(format!("\"{key}\" changes only through node operations"));
                }
                let value = m
                    .get("value")
                    .cloned()
                    .ok_or("operation \"graph\" needs \"value\"")?;
                Self::Graph { key, value }
            }
            other => return Err(format!("unknown operation \"{other}\"")),
        })
    }
}

impl Patch {
    pub fn parse(text: &str) -> Result<Self, String> {
        Self::from_json(&json::parse(text.as_bytes())?)
    }

    pub fn from_json(v: &Value) -> Result<Self, String> {
        let ops = v
            .array()
            .map_err(|_| "a patch must be an array of operations")?;
        Ok(Self(
            ops.iter().map(Op::from_json).collect::<Result<_, _>>()?,
        ))
    }
}

fn node_mut<'g>(graph: &'g mut Graph, id: &str) -> Result<&'g mut Node, String> {
    graph
        .node_mut(id)
        .ok_or_else(|| format!("no node \"{id}\""))
}

/// Moves `id` by `delta` together with everything standing on it, calling `moved` for each
/// node it carries (not for `id`).
pub(crate) fn carry(graph: &mut Graph, id: &str, delta: [f32; 3], moved: &mut dyn FnMut(&str)) {
    let mut stack = vec![id.to_owned()];
    let mut seen: Vec<String> = Vec::new();
    while let Some(at) = stack.pop() {
        if seen.contains(&at) {
            continue;
        }
        if let Some(n) = graph.node_mut(&at) {
            for (p, d) in n.position.iter_mut().zip(delta) {
                *p += d;
            }
        }
        if at != id {
            moved(&at);
        }
        stack.extend(riders(graph, &at));
        seen.push(at);
    }
}

/// Nodes standing directly on `id`.
fn riders(graph: &Graph, id: &str) -> Vec<String> {
    graph
        .edges
        .iter()
        .filter(|e| e.rel == Relation::OnTopOf && e.to == id)
        .map(|e| e.from.clone())
        .collect()
}

fn set(
    graph: &mut Graph,
    id: &str,
    key: &str,
    value: Value,
    applied: &mut Applied,
) -> Result<(), String> {
    let what = format!("node \"{id}\".{key}");
    match key {
        "id" => return Err("a node's id cannot change; put a new node instead".into()),
        "position" => {
            let to = crate::vec3(&value, &what)?;
            let from = node_mut(graph, id)?.position;
            let delta = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
            carry(graph, id, delta, &mut |m| add(&mut applied.dropped, m));
            add(&mut applied.moved, id);
            // A later position wins over an earlier placement in the same patch.
            applied.placed.retain(|p| p != id);
        }
        "yawDeg" => {
            node_mut(graph, id)?.yaw_deg = crate::number(&value, &what)?;
            add(&mut applied.moved, id);
        }
        "type" => match value {
            Value::String(s) if !s.is_empty() => node_mut(graph, id)?.kind = s,
            _ => return Err(format!("{what} must be a nonempty string")),
        },
        "bounds" => {
            let bounds = match value {
                Value::Null => None,
                b => Some(Bounds::from_json(&b, id)?),
            };
            node_mut(graph, id)?.bounds = bounds;
            add(&mut applied.moved, id);
            for rider in riders(graph, id) {
                add(&mut applied.dropped, &rider);
            }
        }
        _ => {
            let extra = &mut node_mut(graph, id)?.extra;
            match value {
                Value::Null => extra.remove(key),
                v => extra.insert(key.to_owned(), v),
            };
        }
    }
    Ok(())
}

/// Applies `patch` to `graph` in order; on an error the graph may be partly changed, so apply to
/// a copy (as [`crate::spatial::commit`] does).
pub fn apply(graph: &mut Graph, patch: &Patch) -> Result<Applied, String> {
    let mut applied = Applied::default();
    for op in &patch.0 {
        match op {
            Op::Put(node) => {
                match graph.node(&node.id).map(|n| n.position) {
                    Some(from) => {
                        let to = node.position;
                        let delta = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
                        carry(graph, &node.id, delta, &mut |m| {
                            add(&mut applied.dropped, m)
                        });
                        if let Some(existing) = graph.node_mut(&node.id) {
                            *existing = node.clone();
                        }
                    }
                    None => graph.nodes.push(node.clone()),
                }
                add(&mut applied.moved, &node.id);
                applied.placed.retain(|p| p != &node.id);
            }
            Op::Remove(id) => {
                if graph.node(id).is_none() {
                    return Err(format!("no node \"{id}\""));
                }
                // What stood on it settles on whatever is below.
                for rider in riders(graph, id) {
                    add(&mut applied.dropped, &rider);
                }
                graph.nodes.retain(|n| &n.id != id);
                graph.edges.retain(|e| &e.from != id && &e.to != id);
                for list in [
                    &mut applied.moved,
                    &mut applied.placed,
                    &mut applied.dropped,
                ] {
                    list.retain(|m| m != id);
                }
            }
            Op::Set { id, key, value } => set(graph, id, key, value.clone(), &mut applied)?,
            Op::Place {
                id,
                rel,
                to,
                anchor,
                clearance,
            } => {
                if id == to || graph.node(to).is_none() || graph.node(id).is_none() {
                    return Err(format!("cannot place \"{id}\" {} \"{to}\"", rel.name()));
                }
                if *rel == Relation::OnTopOf && (graph.rests_on(to, id)) {
                    return Err(format!(
                        "\"{to}\" stands on \"{id}\"; it cannot also hold it"
                    ));
                }
                graph.edges.retain(|e| !(&e.from == id && e.rel.places()));
                graph.edges.push(Edge {
                    from: id.clone(),
                    rel: *rel,
                    to: to.clone(),
                    anchor: anchor.clone(),
                    clearance: *clearance,
                });
                add(&mut applied.moved, id);
                add(&mut applied.placed, id);
            }
            Op::Face { id, to } => {
                node_mut(graph, id)?;
                graph
                    .edges
                    .retain(|e| !(&e.from == id && e.rel == Relation::Facing));
                if let Some(to) = to {
                    if to == id || graph.node(to).is_none() {
                        return Err(format!("\"{id}\" cannot face \"{to}\""));
                    }
                    graph.edges.push(Edge {
                        from: id.clone(),
                        rel: Relation::Facing,
                        to: to.clone(),
                        anchor: None,
                        clearance: None,
                    });
                }
            }
            Op::Graph { key, value } => {
                match value {
                    Value::Null => graph.extra.remove(key),
                    v => graph.extra.insert(key.clone(), v.clone()),
                };
                // New stage bounds may leave nodes outside: check them all.
                if key == "stage" {
                    for n in &graph.nodes {
                        add(&mut applied.moved, &n.id);
                    }
                }
            }
        }
    }
    Ok(applied)
}
