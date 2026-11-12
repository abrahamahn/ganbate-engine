//! Scenes and prefabs: entities with components, parents and prefab instances, kept in a
//! `Document` so every change is an undoable op and every file is canonical JSON.
//!
//! ```json
//! { "ganbate": "scene", "version": 1, "id": "<asset id>", "name": "Harbour",
//!   "entities": {
//!     "<entity id>": {
//!       "name": "Lantern 3", "parent": "<entity id>", "order": "V",
//!       "components": { "transform": { "translation": [1, 0, 2] }, "mesh": { "model": "<asset id>" } },
//!       "prefab": { "asset": "<asset id>",
//!                   "overrides": { "<prefab entity id>": { "name": "...", "components": { ... }, "removed": true } } } } } }
//! ```
//!
//! A prefab file has `"ganbate": "prefab"` and exactly one root entity (one without a parent). An
//! entity with `prefab` is an instance: it takes the prefab root's components under its own
//! (its own win, field by field), and the prefab's other entities appear beneath it with ids
//! derived from the instance's id and theirs (`ganbate_core::id::derived`), changed by the
//! instance's `overrides`. Instances nest. Entities may be parented under those derived ids.
//! Components are sparse: a missing field reads as the schema's default.
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::rc::Rc;

use ganbate_core::id::{derived, is_asset_id, is_entity_id};
use ganbate_core::json::Value;

use crate::doc::{Op, Path};
use crate::order;
use crate::schema::{Issue, Registry};

pub const SCENE_VERSION: u64 = 1;
/// Deepest nesting of prefab instances inside prefabs.
pub const MAX_PREFAB_DEPTH: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Scene,
    Prefab,
}
impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Scene => "scene",
            Kind::Prefab => "prefab",
        }
    }
}

fn s(v: &str) -> Value {
    Value::String(v.into())
}
fn obj(pairs: impl IntoIterator<Item = (String, Value)>) -> Value {
    Value::Object(pairs.into_iter().collect())
}
pub fn path(parts: &[&str]) -> Path {
    parts.iter().map(|p| p.to_string()).collect()
}

/// An empty scene or prefab document.
pub fn new_document(kind: Kind, id: &str, name: &str) -> Value {
    obj([
        ("ganbate".into(), s(kind.name())),
        ("version".into(), Value::Number(SCENE_VERSION as f64)),
        ("id".into(), s(id)),
        ("name".into(), s(name)),
        ("entities".into(), Value::Object(BTreeMap::new())),
    ])
}

pub fn kind(root: &Value) -> Option<Kind> {
    match root.get("ganbate")?.string().ok()? {
        "scene" => Some(Kind::Scene),
        "prefab" => Some(Kind::Prefab),
        _ => None,
    }
}

pub fn entities(root: &Value) -> &BTreeMap<String, Value> {
    static EMPTY: BTreeMap<String, Value> = BTreeMap::new();
    match root.get("entities") {
        Some(Value::Object(m)) => m,
        _ => &EMPTY,
    }
}

fn str_field<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(|x| x.string().ok())
}

pub fn parent_of<'a>(root: &'a Value, id: &str) -> Option<&'a str> {
    str_field(entities(root).get(id)?, "parent")
}

/// Sibling order: by order key (entities without one last), then name, then id.
type SortKey<'a> = (bool, &'a str, &'a str, &'a str);

fn entity_sort_key<'a>(id: &'a str, e: &'a Value) -> SortKey<'a> {
    let order = str_field(e, "order");
    let name = str_field(e, "name").unwrap_or("");
    (order.is_none(), order.unwrap_or(""), name, id)
}

/// The entities directly under `parent` (`None`: the roots), in sibling order.
pub fn children(root: &Value, parent: Option<&str>) -> Vec<String> {
    let mut keys: Vec<SortKey> = entities(root)
        .iter()
        .filter(|(_, e)| str_field(e, "parent") == parent)
        .map(|(id, e)| entity_sort_key(id, e))
        .collect();
    keys.sort_unstable();
    keys.into_iter().map(|k| k.3.to_string()).collect()
}

/// A document's own entities by parent, each list in sibling order, built in one pass so that
/// walking the whole tree costs no more than sorting it.
struct Tree<'a> {
    /// The entities without a parent.
    roots: Vec<&'a str>,
    /// The entities by the id their `parent` names: one of the document's, one an instance
    /// shows (a derived id) or one that does not exist.
    kids: BTreeMap<&'a str, Vec<&'a str>>,
}

impl<'a> Tree<'a> {
    fn new(root: &'a Value) -> Self {
        let mut roots = vec![];
        let mut kids: BTreeMap<&str, Vec<SortKey>> = BTreeMap::new();
        for (id, e) in entities(root) {
            let key = entity_sort_key(id, e);
            match str_field(e, "parent") {
                Some(p) => kids.entry(p).or_default().push(key),
                None => roots.push(key),
            }
        }
        // Ids are unique, so the order is total.
        let ordered = |mut keys: Vec<SortKey<'a>>| -> Vec<&'a str> {
            keys.sort_unstable();
            keys.into_iter().map(|k| k.3).collect()
        };
        Self {
            roots: ordered(roots),
            kids: kids.into_iter().map(|(p, k)| (p, ordered(k))).collect(),
        }
    }

    /// The entities whose parent is `id`, in sibling order.
    fn kids(&self, id: &str) -> &[&'a str] {
        self.kids.get(id).map_or(&[], Vec::as_slice)
    }

    /// `id` and the document's entities beneath it, parents before children, each once even
    /// where parents form a cycle.
    fn subtree(&self, id: &str) -> Vec<String> {
        let mut out = vec![id];
        let mut seen = BTreeSet::from([id]);
        let mut i = 0;
        while i < out.len() {
            for k in self.kids(out[i]) {
                if seen.insert(*k) {
                    out.push(k);
                }
            }
            i += 1;
        }
        out.into_iter().map(String::from).collect()
    }
}

/// `id` and the document's own entities beneath it (through `parent` links, not through what
/// prefab instances show), parents before children. Each appears once, even in a cycle.
pub fn subtree(root: &Value, id: &str) -> Vec<String> {
    Tree::new(root).subtree(id)
}

fn push(issues: &mut Vec<Issue>, path: impl Into<String>, message: impl Into<String>) {
    issues.push(Issue {
        path: path.into(),
        message: message.into(),
    });
}

/// Checks a scene or prefab document; returns every problem found.
pub fn check(root: &Value, registry: &Registry) -> Vec<Issue> {
    let mut issues = vec![];
    let Some(kind) = kind(root) else {
        push(&mut issues, "ganbate", "not a scene or prefab");
        return issues;
    };
    if root.get("version").and_then(|v| v.unsigned().ok()) != Some(SCENE_VERSION) {
        push(&mut issues, "version", "unsupported version");
    }
    if !str_field(root, "id").is_some_and(is_asset_id) {
        push(&mut issues, "id", "missing or invalid asset id");
    }
    if root.get("name").and_then(|v| v.string().ok()).is_none() {
        push(&mut issues, "name", "missing name");
    }
    let ents = match root.get("entities") {
        Some(Value::Object(m)) => m,
        _ => {
            push(&mut issues, "entities", "missing entities object");
            return issues;
        }
    };
    let mut roots = 0;
    for (id, e) in ents {
        let at = format!("entities.{id}");
        if !is_entity_id(id) {
            push(&mut issues, &at, "invalid entity id");
        }
        let Value::Object(map) = e else {
            push(&mut issues, &at, "an entity is an object");
            continue;
        };
        for key in map.keys() {
            if !matches!(
                key.as_str(),
                "name" | "parent" | "order" | "components" | "prefab"
            ) {
                push(&mut issues, format!("{at}.{key}"), "unknown entity key");
            }
        }
        if !matches!(map.get("name"), Some(Value::String(_))) {
            push(&mut issues, format!("{at}.name"), "missing name");
        }
        match map.get("parent") {
            None => roots += 1,
            Some(Value::String(p)) if is_entity_id(p) && p != id => {}
            Some(_) => push(&mut issues, format!("{at}.parent"), "invalid parent"),
        }
        if let Some(o) = map.get("order") {
            if !o.string().is_ok_and(order::valid) {
                push(&mut issues, format!("{at}.order"), "invalid order key");
            }
        }
        match map.get("components") {
            None => {}
            Some(Value::Object(comps)) => {
                for (k, c) in comps {
                    for i in registry.check_component(k, c, &format!("{at}.components.{k}")) {
                        issues.push(i);
                    }
                }
            }
            Some(_) => issues.push(Issue {
                path: format!("{at}.components"),
                message: "components is an object".into(),
            }),
        }
        if let Some(p) = map.get("prefab") {
            check_prefab_link(p, &format!("{at}.prefab"), registry, &mut issues);
        }
    }
    // Cycles among the document's own entities: every entity whose parents lead into one.
    for id in in_parent_cycles(ents) {
        push(&mut issues, format!("entities.{id}.parent"), "parent cycle");
    }
    if kind == Kind::Prefab && roots != 1 {
        issues.push(Issue {
            path: "entities".into(),
            message: format!("a prefab has exactly one root entity, not {roots}"),
        });
    }
    issues
}

/// The entities, in id order, whose chain of `parent` links among `ents` runs into a cycle
/// (those in it and those beneath it). Each entity is walked once, so a long chain costs no
/// more than a short one.
fn in_parent_cycles(ents: &BTreeMap<String, Value>) -> Vec<&str> {
    let mut loops: HashMap<&str, bool> = HashMap::with_capacity(ents.len());
    let mut path = Vec::new();
    let mut on_path = HashSet::new();
    for id in ents.keys() {
        let id = id.as_str();
        if loops.contains_key(id) {
            continue;
        }
        path.clear();
        on_path.clear();
        let mut at = id;
        let looped = loop {
            if let Some(&known) = loops.get(at) {
                break known;
            }
            if !on_path.insert(at) {
                break true;
            }
            path.push(at);
            match ents.get(at).and_then(|e| str_field(e, "parent")) {
                Some(p) if ents.contains_key(p) => at = p,
                _ => break false,
            }
        };
        for &x in &path {
            loops.insert(x, looped);
        }
    }
    ents.keys()
        .map(String::as_str)
        .filter(|&id| loops.get(id).copied().unwrap_or(false))
        .collect()
}

fn check_prefab_link(p: &Value, at: &str, registry: &Registry, issues: &mut Vec<Issue>) {
    if !str_field(p, "asset").is_some_and(is_asset_id) {
        push(
            issues,
            format!("{at}.asset"),
            "missing or invalid prefab asset id",
        );
    }
    if let Value::Object(link) = p {
        for key in link.keys() {
            if !matches!(key.as_str(), "asset" | "overrides") {
                push(issues, format!("{at}.{key}"), "unknown prefab link key");
            }
        }
    }
    let Some(over) = p.get("overrides") else {
        return;
    };
    let Value::Object(over) = over else {
        push(issues, format!("{at}.overrides"), "overrides is an object");
        return;
    };
    for (inner, o) in over {
        let oat = format!("{at}.overrides.{inner}");
        if !is_entity_id(inner) {
            push(issues, oat.clone(), "invalid prefab entity id");
        }
        let Value::Object(m) = o else {
            push(issues, oat, "an override is an object");
            continue;
        };
        for (k, v) in m {
            match (k.as_str(), v) {
                ("name", Value::String(_)) | ("removed", Value::Bool(_)) => {}
                ("components", Value::Object(comps)) => {
                    for (kind, c) in comps {
                        issues.extend(registry.check_component(
                            kind,
                            c,
                            &format!("{oat}.components.{kind}"),
                        ));
                    }
                }
                _ => issues.push(Issue {
                    path: format!("{oat}.{k}"),
                    message: "unknown or invalid override".into(),
                }),
            }
        }
    }
}

// ---- Edits -------------------------------------------------------------------------------

/// Ops that create entity `id`.
pub fn create(
    id: &str,
    name: &str,
    parent: Option<&str>,
    order_key: Option<&str>,
    components: BTreeMap<String, Value>,
) -> Vec<Op> {
    let mut e = BTreeMap::from([("name".to_string(), s(name))]);
    if let Some(p) = parent {
        e.insert("parent".into(), s(p));
    }
    if let Some(o) = order_key {
        e.insert("order".into(), s(o));
    }
    if !components.is_empty() {
        e.insert("components".into(), Value::Object(components));
    }
    vec![Op::Set {
        path: path(&["entities", id]),
        value: Value::Object(e),
    }]
}

/// `ids` and everything beneath them once prefab instances are expanded: the document's own
/// entities, what the instances among them show, and the entities added under that. `load`
/// returns a prefab document by asset id; an instance whose prefab does not flatten is taken
/// to show nothing.
fn beneath(root: &Value, ids: &[&str], load: &dyn Fn(&str) -> Option<Value>) -> BTreeSet<String> {
    let tree = Tree::new(root);
    let ents = entities(root);
    let mut flattener = Flattener::new(load);
    let mut seen = BTreeSet::new();
    let mut todo: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
    while let Some(x) = todo.pop() {
        if seen.contains(&x) {
            continue;
        }
        todo.extend(tree.kids(&x).iter().map(|k| k.to_string()));
        if let Some(e) = ents.get(&x) {
            if let Some(link) = e.get("prefab") {
                if let Ok(shown) = flattener.expand(&x, e, link) {
                    todo.extend(shown.entities.into_iter().map(|r| r.id));
                }
            }
        }
        seen.insert(x);
    }
    seen
}

/// Ops that delete `ids` and everything beneath them, including the entities added under what
/// an instance among them shows (`load` returns a prefab document by asset id).
pub fn delete(root: &Value, ids: &[&str], load: &dyn Fn(&str) -> Option<Value>) -> Vec<Op> {
    let ents = entities(root);
    beneath(root, ids, load)
        .into_iter()
        .filter(|id| ents.contains_key(id))
        .map(|id| Op::Remove {
            path: path(&["entities", &id]),
        })
        .collect()
}

/// Ops that move `id` under `parent` (`None`: to the top) with `order_key`. Refuses a parent
/// beneath `id`, also one that an instance beneath it shows or one added under that (`load`
/// returns a prefab document by asset id).
pub fn reparent(
    root: &Value,
    id: &str,
    parent: Option<&str>,
    order_key: Option<&str>,
    load: &dyn Fn(&str) -> Option<Value>,
) -> Result<Vec<Op>, String> {
    if !entities(root).contains_key(id) {
        return Err(format!("no entity {id}"));
    }
    if let Some(p) = parent {
        if beneath(root, &[id], load).contains(p) {
            return Err("an entity cannot move beneath itself".into());
        }
    }
    let mut ops = vec![match parent {
        Some(p) => Op::Set {
            path: path(&["entities", id, "parent"]),
            value: s(p),
        },
        None => Op::Remove {
            path: path(&["entities", id, "parent"]),
        },
    }];
    ops.push(match order_key {
        Some(o) => Op::Set {
            path: path(&["entities", id, "order"]),
            value: s(o),
        },
        None => Op::Remove {
            path: path(&["entities", id, "order"]),
        },
    });
    Ok(ops)
}

/// The op that sets one field of a component (`field` may go into object fields).
pub fn set_field(id: &str, component: &str, field: &[&str], value: Value) -> Op {
    let mut p = path(&["entities", id, "components", component]);
    p.extend(field.iter().map(|f| f.to_string()));
    Op::Set { path: p, value }
}

/// Ops that copy `ids` (with what is beneath them) beside the originals, under new ids from
/// `next_id`; returns the ops and the new ids of the copied tops.
pub fn duplicate(
    root: &Value,
    ids: &[&str],
    mut next_id: impl FnMut() -> String,
) -> (Vec<Op>, Vec<String>) {
    let tree = Tree::new(root);
    let mut map = BTreeMap::new();
    let mut tops = vec![];
    let mut order_ids = vec![];
    for id in ids {
        for x in tree.subtree(id) {
            if !map.contains_key(&x) {
                map.insert(x.clone(), next_id());
                order_ids.push(x);
            }
        }
        tops.push(map[*id].clone());
    }
    let mut ops = vec![];
    for old in order_ids {
        let Some(Value::Object(e)) = entities(root).get(&old) else {
            continue;
        };
        let mut e = e.clone();
        if let Some(Value::String(p)) = e.get("parent") {
            if let Some(newp) = map.get(p) {
                e.insert("parent".into(), s(newp));
            }
        }
        if ids.contains(&old.as_str()) {
            if let Some(Value::String(n)) = e.get("name") {
                let n = format!("{n} copy");
                e.insert("name".into(), s(&n));
            }
            e.remove("order");
        }
        ops.push(Op::Set {
            path: path(&["entities", &map[&old]]),
            value: Value::Object(e),
        });
    }
    (ops, tops)
}

// ---- Prefab expansion ---------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// Written in this document.
    Own,
    /// Shown by prefab instance `instance` from entity `inner` of prefab `prefab`.
    Prefab {
        instance: String,
        prefab: String,
        inner: String,
    },
}

/// An entity as it exists once prefab instances are expanded.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    pub id: String,
    pub name: String,
    pub parent: Option<String>,
    pub components: BTreeMap<String, Value>,
    pub source: Source,
}

/// Merges `over` into `base`: objects field by field, anything else replaced.
pub fn merge(base: &mut Value, over: &Value) {
    match (base, over) {
        (Value::Object(b), Value::Object(o)) => merge_into(b, o),
        (b, o) => *b = o.clone(),
    }
}

fn merge_into(base: &mut BTreeMap<String, Value>, over: &BTreeMap<String, Value>) {
    for (k, v) in over {
        match base.get_mut(k) {
            Some(slot) => merge(slot, v),
            None => {
                base.insert(k.clone(), v.clone());
            }
        }
    }
}

fn components_of(e: &Value) -> BTreeMap<String, Value> {
    match e.get("components") {
        Some(Value::Object(m)) => m.clone(),
        _ => BTreeMap::new(),
    }
}

/// Every entity of `root` with prefab instances expanded, parents before children: the
/// document's own entities depth first in sibling order (then those whose parent does not
/// exist, each with what is beneath it), then what each instance shows, in the order the
/// instances came, with every entity added under one of those entities right after it.
/// `load` returns a prefab document by asset id; each prefab is loaded and flattened once.
///
/// Refuses parent cycles (also one that runs through what an instance shows, which would
/// otherwise leave the instance nowhere), prefabs that are missing, broken, contain
/// themselves or nest deeper than `MAX_PREFAB_DEPTH`, and more than `MAX_ENTITIES` entities.
pub fn flatten(
    root: &Value,
    load: &dyn Fn(&str) -> Option<Value>,
) -> Result<Vec<Resolved>, String> {
    let done = Flattener::new(load).document(root, true);
    match done.failures.into_iter().next() {
        Some(f) => Err(format!("{}: {}", f.path, f.message)),
        None => Ok(done.resolved),
    }
}

/// `check`, plus what only expanding the prefab instances shows (`load` returns a prefab
/// document by asset id): parent cycles that run through what an instance shows, prefabs that
/// are missing or broken, and more entities than one scene may have. An editor host checks
/// edits with this so that it never accepts a document `flatten` refuses.
pub fn check_with_prefabs(
    root: &Value,
    registry: &Registry,
    load: &dyn Fn(&str) -> Option<Value>,
) -> Vec<Issue> {
    let mut issues = check(root, registry);
    let mut known: BTreeSet<(String, String)> = issues
        .iter()
        .map(|i| (i.path.clone(), i.message.clone()))
        .collect();
    for f in Flattener::new(load).document(root, false).failures {
        if known.insert((f.path.clone(), f.message.clone())) {
            issues.push(Issue {
                path: f.path,
                message: f.message,
            });
        }
    }
    issues
}

/// Most entities a scene or prefab may have once its instances are expanded: what one cooked
/// entity table holds.
pub const MAX_ENTITIES: usize = ganbate_core::world::MAX_ENTITIES;
/// Most prefab entities one `flatten` reads and makes over all its nesting levels, so a few
/// small files cannot keep it busy for hours. Expanding `MAX_ENTITIES` entities through every
/// level of nesting stays within it.
const MAX_WORK: usize = MAX_ENTITIES * (MAX_PREFAB_DEPTH + 1);

/// A prefab flattened once for all its instances in one `flatten`.
struct Flat {
    resolved: Vec<Resolved>,
    /// Where the prefab's root is in `resolved`.
    root: usize,
    /// Prefab levels it spans, itself included (1: it holds no instances).
    height: usize,
}

/// What an instance shows: its components (the prefab root's beneath its own) and the
/// prefab's other entities under derived ids, parents before children.
struct Shown {
    components: BTreeMap<String, Value>,
    entities: Vec<Resolved>,
    /// The prefab's `Flat::height`.
    height: usize,
}

/// Why a document does not flatten: where (a path like an `Issue`'s) and what.
struct Failure {
    path: String,
    message: String,
}

#[derive(Default)]
struct Done {
    resolved: Vec<Resolved>,
    /// Prefab levels beneath the document (0: it holds no instances).
    height: usize,
    failures: Vec<Failure>,
}

impl Done {
    fn fail(&mut self, path: impl Into<String>, message: impl Into<String>) {
        self.failures.push(Failure {
            path: path.into(),
            message: message.into(),
        });
    }
}

struct Flattener<'l> {
    load: &'l dyn Fn(&str) -> Option<Value>,
    /// Prefabs flattened so far, by asset id.
    flat: BTreeMap<String, Rc<Flat>>,
    /// The prefabs being flattened, outermost first.
    stack: Vec<String>,
    /// Prefab entities read and made so far, over all levels.
    work: usize,
}

impl<'l> Flattener<'l> {
    fn new(load: &'l dyn Fn(&str) -> Option<Value>) -> Self {
        Self {
            load,
            flat: BTreeMap::new(),
            stack: vec![],
            work: 0,
        }
    }

    fn spend(&mut self, n: usize) -> Result<(), String> {
        self.work = self.work.saturating_add(n);
        if self.work > MAX_WORK {
            return Err(format!(
                "expanding prefabs goes through more than {MAX_WORK} entities"
            ));
        }
        Ok(())
    }

    /// Flattens a scene or prefab document. With `strict`, stops at the first failure;
    /// otherwise notes every failure it can and goes on (an instance that fails shows nothing).
    fn document(&mut self, root: &Value, strict: bool) -> Done {
        let mut done = Done::default();
        let own = entities(root);
        if let Err(m) = self.spend(own.len()) {
            done.fail("entities", m);
            return done;
        }
        let too_many = || format!("more than {MAX_ENTITIES} entities once prefabs are expanded");
        let mut total = own.len();
        if total > MAX_ENTITIES {
            done.fail("entities", too_many());
            return done;
        }
        // Expand every instance before placing anything: an entity added under what an
        // instance shows goes right after that entity, and a parent cycle can run through it.
        let mut shown: BTreeMap<&str, Shown> = BTreeMap::new();
        // Every id the instances show.
        let mut shown_ids: HashSet<String> = HashSet::new();
        for (id, e) in own {
            let Some(link) = e.get("prefab") else {
                continue;
            };
            let at = format!("entities.{id}.prefab");
            let x = match self.expand(id, e, link) {
                Ok(x) => x,
                Err(m) => {
                    done.fail(at, m);
                    if strict {
                        return done;
                    }
                    continue;
                }
            };
            // Derived ids are hashes: two entities can share one only by a forged collision.
            let mut ids = HashSet::with_capacity(x.entities.len());
            if let Some(r) = x.entities.iter().find(|r| {
                own.contains_key(&r.id) || shown_ids.contains(&r.id) || !ids.insert(&r.id)
            }) {
                done.fail(at, format!("shows entity {}, which already exists", r.id));
                if strict {
                    return done;
                }
                continue;
            }
            total += x.entities.len();
            if total > MAX_ENTITIES {
                done.fail("entities", too_many());
                return done;
            }
            done.height = done.height.max(x.height);
            shown_ids.extend(x.entities.iter().map(|r| r.id.clone()));
            shown.insert(id, x);
        }
        let tree = Tree::new(root);
        let mut place = Placer {
            own,
            tree: &tree,
            shown,
            out: Vec::with_capacity(total),
            placed: HashSet::with_capacity(own.len()),
            queue: VecDeque::new(),
        };
        for r in &tree.roots {
            place.walk(r);
        }
        // Entities whose parent does not exist go after those reached from the roots.
        for (id, e) in own {
            if let Some(p) = str_field(e, "parent") {
                if !own.contains_key(p) && !shown_ids.contains(p) {
                    place.walk(id);
                }
            }
        }
        while let Some(instance) = place.queue.pop_front() {
            let entities = place
                .shown
                .remove(instance)
                .map(|x| x.entities)
                .unwrap_or_default();
            for r in entities {
                let kids = tree.kids(&r.id);
                place.out.push(r);
                for k in kids {
                    place.walk(k);
                }
            }
        }
        // Whatever is left was never reached: its parents lead into a cycle, through the
        // document's own entities or through what one of its instances shows.
        for id in own.keys() {
            if !place.placed.contains(id.as_str()) {
                done.fail(format!("entities.{id}.parent"), "parent cycle");
                if strict {
                    return done;
                }
            }
        }
        done.resolved = place.out;
        done
    }

    /// What instance `id` (entity `e`, prefab link `link`) shows.
    fn expand(&mut self, id: &str, e: &Value, link: &Value) -> Result<Shown, String> {
        let asset = str_field(link, "asset").ok_or("a prefab link needs an asset")?;
        let flat = self.prefab(asset)?;
        self.spend(flat.resolved.len())?;
        let prefab_root = flat
            .resolved
            .get(flat.root)
            .ok_or_else(|| format!("prefab {asset} has no root"))?;
        // The instance takes the prefab root's components beneath its own.
        let mut components = prefab_root.components.clone();
        if let Some(Value::Object(own)) = e.get("components") {
            merge_into(&mut components, own);
        }
        let no_overrides = BTreeMap::new();
        let overrides = match link.get("overrides") {
            Some(Value::Object(m)) => m,
            _ => &no_overrides,
        };
        let mut removed = BTreeSet::new();
        let mut entities = Vec::with_capacity(flat.resolved.len().saturating_sub(1));
        for (i, r) in flat.resolved.iter().enumerate() {
            if i == flat.root {
                continue;
            }
            let o = overrides.get(&r.id);
            let parent_removed = r.parent.as_deref().is_some_and(|p| removed.contains(p));
            if parent_removed || o.and_then(|o| o.get("removed")) == Some(&Value::Bool(true)) {
                removed.insert(r.id.as_str());
                continue;
            }
            let mut c = r.components.clone();
            // Override components merge field by field; anything but an object is no override
            // (`check` reports it).
            if let Some(Value::Object(oc)) = o.and_then(|o| o.get("components")) {
                merge_into(&mut c, oc);
            }
            let parent = match r.parent.as_deref() {
                Some(p) if p != prefab_root.id => derived(id, p),
                _ => id.to_string(),
            };
            entities.push(Resolved {
                id: derived(id, &r.id),
                name: o
                    .and_then(|o| str_field(o, "name"))
                    .unwrap_or(&r.name)
                    .to_string(),
                parent: Some(parent),
                components: c,
                source: Source::Prefab {
                    instance: id.to_string(),
                    prefab: asset.to_string(),
                    inner: r.id.clone(),
                },
            });
        }
        Ok(Shown {
            components,
            entities,
            height: flat.height,
        })
    }

    /// Prefab `asset` flattened: loaded and flattened the first time, kept for the others.
    fn prefab(&mut self, asset: &str) -> Result<Rc<Flat>, String> {
        if self.stack.iter().any(|a| a == asset) {
            return Err(format!("prefab {asset} contains itself"));
        }
        if let Some(f) = self.flat.get(asset) {
            // Flattened before at a shallower level; it must still fit at this one.
            if self.stack.len() + f.height > MAX_PREFAB_DEPTH {
                return Err("prefabs nest too deeply".into());
            }
            return Ok(Rc::clone(f));
        }
        if self.stack.len() >= MAX_PREFAB_DEPTH {
            return Err("prefabs nest too deeply".into());
        }
        let doc = (self.load)(asset).ok_or_else(|| format!("missing prefab {asset}"))?;
        if kind(&doc) != Some(Kind::Prefab) {
            return Err(format!("{asset} is not a prefab"));
        }
        self.stack.push(asset.to_string());
        let done = self.document(&doc, true);
        self.stack.pop();
        if let Some(f) = done.failures.first() {
            return Err(format!("in prefab {asset}: {}: {}", f.path, f.message));
        }
        let root = done
            .resolved
            .iter()
            .position(|r| r.parent.is_none())
            .ok_or_else(|| format!("prefab {asset} has no root"))?;
        let flat = Rc::new(Flat {
            resolved: done.resolved,
            root,
            height: done.height + 1,
        });
        self.flat.insert(asset.to_string(), Rc::clone(&flat));
        Ok(flat)
    }
}

/// Places a document's own entities and what its instances show, parents first.
struct Placer<'r> {
    own: &'r BTreeMap<String, Value>,
    tree: &'r Tree<'r>,
    /// What each instance not yet placed shows.
    shown: BTreeMap<&'r str, Shown>,
    out: Vec<Resolved>,
    placed: HashSet<&'r str>,
    /// Instances placed whose shown entities wait to be placed, in the order they came.
    queue: VecDeque<&'r str>,
}

impl<'r> Placer<'r> {
    /// Places `start` and the document's entities beneath it, depth first in sibling order.
    fn walk(&mut self, start: &'r str) {
        let mut todo = vec![start];
        while let Some(id) = todo.pop() {
            let Some(e) = self.own.get(id) else {
                continue;
            };
            if !self.placed.insert(id) {
                continue;
            }
            let components = match self.shown.get_mut(id) {
                Some(x) => {
                    self.queue.push_back(id);
                    std::mem::take(&mut x.components)
                }
                None => components_of(e),
            };
            self.out.push(Resolved {
                id: id.to_string(),
                name: str_field(e, "name").unwrap_or("").to_string(),
                parent: str_field(e, "parent").map(String::from),
                components,
                source: Source::Own,
            });
            todo.extend(self.tree.kids(id).iter().rev());
        }
    }
}

// ---- Transforms ---------------------------------------------------------------------------

fn numbers<const N: usize>(v: Option<&Value>, default: [f32; N]) -> [f32; N] {
    let Some(Value::Array(a)) = v else {
        return default;
    };
    if a.len() != N {
        return default;
    }
    let mut out = default;
    for (o, x) in out.iter_mut().zip(a) {
        if let Value::Number(n) = x {
            *o = *n as f32;
        }
    }
    out
}

/// The local matrix (column-major) of a stored `transform` component.
pub fn local_matrix(transform: Option<&Value>) -> [f32; 16] {
    let t = numbers(transform.and_then(|t| t.get("translation")), [0.0; 3]);
    let q = numbers(
        transform.and_then(|t| t.get("rotation")),
        [0.0, 0.0, 0.0, 1.0],
    );
    let sc = numbers(transform.and_then(|t| t.get("scale")), [1.0; 3]);
    ganbate_core::world::Transform::from_trs(t, q, sc)
        .unwrap_or(ganbate_core::world::Transform::IDENTITY)
        .0
}

/// World transforms of resolved entities in `resolved` order (parents must precede children,
/// as `flatten` gives).
pub fn world_transforms(resolved: &[Resolved]) -> Vec<ganbate_core::world::Transform> {
    use ganbate_core::world::Transform;
    let mut index_of: HashMap<&str, usize> = HashMap::with_capacity(resolved.len());
    let mut out: Vec<Transform> = Vec::with_capacity(resolved.len());
    for (i, r) in resolved.iter().enumerate() {
        let local = Transform(local_matrix(r.components.get("transform")));
        let world = match r.parent.as_deref().and_then(|p| index_of.get(p).copied()) {
            Some(pi) => out[pi].compose(local),
            None => local,
        };
        index_of.insert(&r.id, i);
        out.push(world);
    }
    out
}

/// World matrices of resolved entities (parents must precede children, as `flatten` gives).
pub fn world_matrices(resolved: &[Resolved]) -> BTreeMap<String, [f32; 16]> {
    resolved
        .iter()
        .zip(world_transforms(resolved))
        .map(|(r, t)| (r.id.clone(), t.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{Document, History};
    use crate::schema::engine_registry;
    use ganbate_core::id::Ids;
    use ganbate_core::json::{parse, to_canonical};

    const SCENE_ID: &str = "0123456789abcdef0123456789abcdef";
    const PREFAB_ID: &str = "fedcba9876543210fedcba9876543210";

    fn tr(x: f64) -> BTreeMap<String, Value> {
        BTreeMap::from([(
            "transform".to_string(),
            parse(format!(r#"{{"translation":[{x},0,0]}}"#).as_bytes()).unwrap(),
        )])
    }

    fn prefab() -> Value {
        let mut p = new_document(Kind::Prefab, PREFAB_ID, "Lamp post");
        let mut d = Document::new(p.clone()).unwrap();
        d.apply_all(&create("00000000000000a1", "Post", None, None, tr(0.0)))
            .unwrap();
        d.apply_all(&create(
            "00000000000000a2",
            "Lamp",
            Some("00000000000000a1"),
            None,
            tr(2.0),
        ))
        .unwrap();
        d.apply_all(&create(
            "00000000000000a3",
            "Bulb",
            Some("00000000000000a2"),
            None,
            tr(0.5),
        ))
        .unwrap();
        p = d.root().clone();
        p
    }

    /// Runs `f` on its own thread and fails the test if it has not returned within `secs`, so
    /// a walk that never ends fails instead of hanging the run.
    fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
        use std::sync::mpsc::RecvTimeoutError;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        match rx.recv_timeout(std::time::Duration::from_secs(secs)) {
            Ok(v) => v,
            Err(RecvTimeoutError::Timeout) => panic!("did not finish within {secs} s"),
            Err(RecvTimeoutError::Disconnected) => panic!("panicked"),
        }
    }

    /// A scene whose entities' parents are given as (id, parent) pairs; ids are entity ids
    /// written without leading zeros.
    fn scene_of(parents: &[(u64, Option<u64>)]) -> Value {
        let mut ents = BTreeMap::new();
        for (id, parent) in parents {
            let mut e = BTreeMap::from([("name".to_string(), s(&format!("E{id:x}")))]);
            if let Some(p) = parent {
                e.insert("parent".into(), s(&format!("{p:016x}")));
            }
            ents.insert(format!("{id:016x}"), Value::Object(e));
        }
        let mut root = new_document(Kind::Scene, SCENE_ID, "S");
        if let Value::Object(m) = &mut root {
            m.insert("entities".into(), Value::Object(ents));
        }
        root
    }

    #[test]
    fn walks_end_on_parent_cycles() {
        // A and B name each other as parent (a clean merge of "A under B" and "B under A");
        // C is under A, D is a root.
        let (a, b, c, d) = (0xa1, 0xa2, 0xa3, 0xa4);
        let doc = scene_of(&[(a, Some(b)), (b, Some(a)), (c, Some(a)), (d, None)]);
        let id = |n: u64| format!("{n:016x}");
        let (sub, deleted, copied, moved, issues) = within(10, move || {
            let mut ids = Ids::seeded(3);
            (
                subtree(&doc, &id(a)),
                delete(&doc, &[&id(b)], &|_| None).len(),
                duplicate(&doc, &[&id(a)], || ids.entity()).0.len(),
                reparent(&doc, &id(d), Some(&id(c)), None, &|_| None).is_ok(),
                check(&doc, &engine_registry()),
            )
        });
        assert_eq!(sub, [id(a), id(b), id(c)]);
        assert_eq!((deleted, copied, moved), (3, 3, true));
        let cycles: Vec<&str> = issues
            .iter()
            .filter(|i| i.message == "parent cycle")
            .map(|i| i.path.as_str())
            .collect();
        let want: Vec<String> = [a, b, c]
            .iter()
            .map(|n| format!("entities.{}.parent", id(*n)))
            .collect();
        assert_eq!(cycles, want);
    }

    #[test]
    fn long_parent_chains_are_walked_in_linear_time() {
        // 30,000 entities each the parent of the next. Walking every entity's ancestors
        // separately takes about 450 million steps; one walk with memory takes 30,000.
        let n = 30_000u64;
        let parents: Vec<(u64, Option<u64>)> =
            (1..=n).map(|i| (i, (i > 1).then_some(i - 1))).collect();
        let doc = scene_of(&parents);
        let (issues, below) = within(30, move || {
            let issues = check(&doc, &engine_registry());
            let below = subtree(&doc, &format!("{:016x}", 1)).len();
            (issues, below)
        });
        assert_eq!(issues, vec![]);
        assert_eq!(below, n as usize);
    }

    #[test]
    fn edits_build_a_valid_scene_and_undo_to_the_same_bytes() {
        let reg = engine_registry();
        let mut doc = Document::new(new_document(Kind::Scene, SCENE_ID, "Test")).unwrap();
        let empty = to_canonical(doc.root()).unwrap();
        let mut h = History::new(64);
        let mut ids = Ids::seeded(1);
        let a = ids.entity();
        let b = ids.entity();
        h.edit(
            &mut doc,
            "Add",
            None,
            create(&a, "A", None, Some("V"), tr(1.0)),
        )
        .unwrap();
        h.edit(
            &mut doc,
            "Add",
            None,
            create(&b, "B", Some(&a), None, tr(2.0)),
        )
        .unwrap();
        h.edit(
            &mut doc,
            "Move",
            Some("drag"),
            vec![set_field(
                &b,
                "transform",
                &["translation"],
                parse(b"[3,0,0]").unwrap(),
            )],
        )
        .unwrap();
        assert_eq!(check(doc.root(), &reg), vec![]);
        let flat = flatten(doc.root(), &|_| None).unwrap();
        let m = world_matrices(&flat);
        assert_eq!(m[&b][12], 4.0);
        // A cycle is refused.
        assert!(reparent(doc.root(), &a, Some(&b), None, &|_| None).is_err());
        let ops = delete(doc.root(), &[&a], &|_| None);
        assert_eq!(ops.len(), 2);
        h.edit(&mut doc, "Delete", None, ops).unwrap();
        assert!(entities(doc.root()).is_empty());
        while h.undo(&mut doc).unwrap().is_some() {}
        assert_eq!(to_canonical(doc.root()).unwrap(), empty);
    }

    #[test]
    fn checking_finds_bad_entities() {
        let reg = engine_registry();
        let bad = parse(format!(
            r#"{{"ganbate":"scene","version":1,"id":"{SCENE_ID}","name":"x","entities":{{
              "00000000000000a1":{{"name":"A","parent":"00000000000000a2"}},
              "00000000000000a2":{{"name":"B","parent":"00000000000000a1","order":"a0"}},
              "00000000000000a3":{{"name":"D","prefab":{{"asset":"{PREFAB_ID}","k":{{"k":{{"k":{{}}}}}}}}}},
              "zz":{{"name":"C","colour":1,"components":{{"transform":{{"scale":[1,1]}},"nope":{{}}}}}}}}}}"#
        ).as_bytes())
        .unwrap();
        let issues = check(&bad, &reg);
        let text: Vec<String> = issues
            .iter()
            .map(|i| format!("{}: {}", i.path, i.message))
            .collect();
        for want in [
            "entities.zz: invalid entity id",
            "entities.zz.colour: unknown entity key",
            "entities.zz.components.transform.scale: expected 3 numbers",
            "entities.zz.components.nope: unknown component kind \"nope\"",
            "entities.00000000000000a2.order: invalid order key",
            "entities.00000000000000a1.parent: parent cycle",
            "entities.00000000000000a3.prefab.k: unknown prefab link key",
        ] {
            assert!(
                text.iter().any(|t| t == want),
                "missing {want:?} in {text:#?}"
            );
        }
    }

    #[test]
    fn prefab_instances_expand_with_overrides_and_derived_ids() {
        let reg = engine_registry();
        let p = prefab();
        assert_eq!(check(&p, &reg), vec![]);
        let inst = "00000000000000b1";
        let scene = parse(format!(
            r#"{{"ganbate":"scene","version":1,"id":"{SCENE_ID}","name":"S","entities":{{
              "{inst}":{{"name":"Lamp post 1","components":{{"transform":{{"translation":[10,0,0]}}}},
                 "prefab":{{"asset":"{PREFAB_ID}","overrides":{{
                    "00000000000000a2":{{"name":"Big lamp","components":{{"transform":{{"translation":[0,3,0]}}}}}},
                    "00000000000000a3":{{"removed":true}}}}}}}},
              "00000000000000c1":{{"name":"Flag","parent":"{}"}} }}}}"#,
            derived(inst, "00000000000000a2")
        ).as_bytes())
        .unwrap();
        assert_eq!(check(&scene, &reg), vec![]);
        let load = |id: &str| (id == PREFAB_ID).then(|| p.clone());
        let flat = flatten(&scene, &load).unwrap();
        let names: Vec<&str> = flat.iter().map(|r| r.name.as_str()).collect();
        // The flag was added under the lamp the instance shows, so it comes after the lamp.
        assert_eq!(names, ["Lamp post 1", "Big lamp", "Flag"]);
        let lamp = flat.iter().find(|r| r.name == "Big lamp").unwrap();
        assert_eq!(lamp.id, derived(inst, "00000000000000a2"));
        assert_eq!(lamp.parent.as_deref(), Some(inst));
        let m = world_matrices(&flat);
        // The instance root moved to x=10; the lamp's override put it 3 m up, no x offset.
        assert_eq!((m[&lamp.id][12], m[&lamp.id][13]), (10.0, 3.0));
        assert_eq!(m["00000000000000c1"], m[&lamp.id]);
        // A prefab that contains itself is refused.
        let mut selfish = p.clone();
        merge(
            &mut selfish,
            &parse(format!(r#"{{"entities":{{"00000000000000a2":{{"prefab":{{"asset":"{PREFAB_ID}"}}}}}}}}"#).as_bytes()).unwrap(),
        );
        let load2 = |id: &str| (id == PREFAB_ID).then(|| selfish.clone());
        assert!(
            flatten(&selfish, &load2)
                .unwrap_err()
                .contains("contains itself")
        );
    }

    /// A scene or prefab document whose entities are the JSON object `entities`.
    fn doc_with(kind: Kind, id: &str, entities: &str) -> Value {
        let text = format!(
            r#"{{"ganbate":"{}","version":1,"id":"{id}","name":"D","entities":{entities}}}"#,
            kind.name()
        );
        parse(text.as_bytes()).unwrap()
    }

    const POLE_ID: &str = "00000000000000000000000000000f00";
    const MODEL_ID: &str = "11111111111111111111111111111111";

    /// A pole whose root carries a mesh, with a cloth 1 m up.
    fn pole() -> Value {
        doc_with(
            Kind::Prefab,
            POLE_ID,
            &format!(
                r#"{{"00000000000000f1":{{"name":"Pole","components":{{"mesh":{{"model":"{MODEL_ID}"}}}}}},
                   "00000000000000f2":{{"name":"Cloth","parent":"00000000000000f1",
                      "components":{{"transform":{{"translation":[0,1,0]}}}}}}}}"#
            ),
        )
    }

    fn translation(m: &[f32; 16]) -> [f32; 3] {
        [m[12], m[13], m[14]]
    }

    #[test]
    fn entities_under_what_an_instance_shows_follow_it_and_expand() {
        // The lamp post (Post > Lamp at x=2 > Bulb at x=0.5) stands at x=10. A flag (with a
        // pennant 1 m along z) and an instance of the pole are added under the lamp it shows.
        let (lamp_post, pole) = (prefab(), pole());
        let b1 = "00000000000000b1";
        let lamp = derived(b1, "00000000000000a2");
        let scene = doc_with(
            Kind::Scene,
            SCENE_ID,
            &format!(
                r#"{{"{b1}":{{"name":"Lamp post 1","prefab":{{"asset":"{PREFAB_ID}"}},
                       "components":{{"transform":{{"translation":[10,0,0]}}}}}},
                   "00000000000000c1":{{"name":"Flag","parent":"{lamp}"}},
                   "00000000000000c2":{{"name":"Pennant","parent":"00000000000000c1",
                       "components":{{"transform":{{"translation":[0,0,1]}}}}}},
                   "00000000000000b2":{{"name":"Pole 1","parent":"{lamp}","order":"V",
                       "prefab":{{"asset":"{POLE_ID}"}}}}}}"#
            ),
        );
        assert_eq!(check(&scene, &engine_registry()), vec![]);
        let load = |id: &str| match id {
            PREFAB_ID => Some(lamp_post.clone()),
            POLE_ID => Some(pole.clone()),
            _ => None,
        };
        let flat = flatten(&scene, &load).unwrap();
        for (i, r) in flat.iter().enumerate() {
            if let Some(p) = &r.parent {
                assert!(
                    flat[..i].iter().any(|q| &q.id == p),
                    "{} before its parent",
                    r.name
                );
            }
        }
        let names: Vec<&str> = flat.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "Lamp post 1",
                "Lamp",
                "Pole 1",
                "Flag",
                "Pennant",
                "Bulb",
                "Cloth"
            ]
        );
        let m = world_matrices(&flat);
        assert_eq!(translation(&m[&lamp]), [12.0, 0.0, 0.0]);
        assert_eq!(translation(&m["00000000000000c1"]), [12.0, 0.0, 0.0]);
        assert_eq!(translation(&m["00000000000000c2"]), [12.0, 0.0, 1.0]);
        // The pole instance under the lamp takes its prefab's root mesh and shows its cloth.
        let pole_1 = flat.iter().find(|r| r.id == "00000000000000b2").unwrap();
        assert!(pole_1.components.contains_key("mesh"));
        let cloth = derived("00000000000000b2", "00000000000000f2");
        let c = flat.iter().find(|r| r.id == cloth).unwrap();
        assert_eq!(c.parent.as_deref(), Some("00000000000000b2"));
        assert_eq!(translation(&m[&cloth]), [12.0, 1.0, 0.0]);
    }

    #[test]
    fn flatten_refuses_parent_cycles_in_scenes_and_prefabs() {
        let doc = scene_of(&[(0xa1, Some(0xa2)), (0xa2, Some(0xa1)), (0xa4, None)]);
        let e = within(10, move || flatten(&doc, &|_| None)).unwrap_err();
        assert_eq!(e, "entities.00000000000000a1.parent: parent cycle");
        // A prefab whose two non-root entities name each other; the scene using it is clean.
        let p = doc_with(
            Kind::Prefab,
            PREFAB_ID,
            r#"{"00000000000000a0":{"name":"R"},
                "00000000000000a1":{"name":"A","parent":"00000000000000a2"},
                "00000000000000a2":{"name":"B","parent":"00000000000000a1"}}"#,
        );
        assert!(!check(&p, &engine_registry()).is_empty());
        let scene = doc_with(
            Kind::Scene,
            SCENE_ID,
            &format!(r#"{{"00000000000000b1":{{"name":"I","prefab":{{"asset":"{PREFAB_ID}"}}}}}}"#),
        );
        assert_eq!(check(&scene, &engine_registry()), vec![]);
        let e = within(10, move || {
            flatten(&scene, &move |id: &str| {
                (id == PREFAB_ID).then(|| p.clone())
            })
        })
        .unwrap_err();
        assert_eq!(
            e,
            format!(
                "entities.00000000000000b1.prefab: in prefab {PREFAB_ID}: \
                 entities.00000000000000a1.parent: parent cycle"
            )
        );
    }

    #[test]
    fn a_cycle_through_what_an_instance_shows_is_refused() {
        let reg = engine_registry();
        let p = prefab();
        let load = |id: &str| (id == PREFAB_ID).then(|| p.clone());
        let b1 = "00000000000000b1";
        let lamp = derived(b1, "00000000000000a2");
        // The instance beneath the lamp it shows: directly, or through an entity added there.
        for entities in [
            format!(
                r#"{{"{b1}":{{"name":"I","parent":"{lamp}","prefab":{{"asset":"{PREFAB_ID}"}}}}}}"#
            ),
            format!(
                r#"{{"{b1}":{{"name":"I","parent":"00000000000000c1","prefab":{{"asset":"{PREFAB_ID}"}}}},
                   "00000000000000c1":{{"name":"Flag","parent":"{lamp}"}}}}"#
            ),
        ] {
            let scene = doc_with(Kind::Scene, SCENE_ID, &entities);
            // The document alone cannot tell: the lamp's id is derived from the prefab's.
            assert_eq!(check(&scene, &reg), vec![]);
            let e = flatten(&scene, &load).unwrap_err();
            assert_eq!(e, format!("entities.{b1}.parent: parent cycle"));
            let issues = check_with_prefabs(&scene, &reg, &load);
            assert!(
                issues.contains(&Issue {
                    path: format!("entities.{b1}.parent"),
                    message: "parent cycle".into(),
                }),
                "{issues:?}"
            );
        }
    }

    #[test]
    fn an_entity_with_an_id_an_instance_shows_is_refused() {
        let p = prefab();
        let b1 = "00000000000000b1";
        let lamp = derived(b1, "00000000000000a2");
        let scene = doc_with(
            Kind::Scene,
            SCENE_ID,
            &format!(
                r#"{{"{b1}":{{"name":"I","prefab":{{"asset":"{PREFAB_ID}"}}}},
                   "{lamp}":{{"name":"Copy of the lamp"}}}}"#
            ),
        );
        let e = flatten(&scene, &|id: &str| (id == PREFAB_ID).then(|| p.clone())).unwrap_err();
        assert_eq!(
            e,
            format!("entities.{b1}.prefab: shows entity {lamp}, which already exists")
        );
    }

    /// A lamp post b1 at x=10 with, under the lamp it shows, a flag c1 (and a pennant c2 under
    /// that) and a pole instance b2 with a streamer c3 under the cloth it shows; d1 stands apart.
    fn decorated_lamp_post() -> Value {
        let lamp = derived("00000000000000b1", "00000000000000a2");
        let cloth = derived("00000000000000b2", "00000000000000f2");
        doc_with(
            Kind::Scene,
            SCENE_ID,
            &format!(
                r#"{{"00000000000000b1":{{"name":"Lamp post 1","prefab":{{"asset":"{PREFAB_ID}"}},
                       "components":{{"transform":{{"translation":[10,0,0]}}}}}},
                   "00000000000000c1":{{"name":"Flag","parent":"{lamp}"}},
                   "00000000000000c2":{{"name":"Pennant","parent":"00000000000000c1"}},
                   "00000000000000b2":{{"name":"Pole 1","parent":"{lamp}","prefab":{{"asset":"{POLE_ID}"}}}},
                   "00000000000000c3":{{"name":"Streamer","parent":"{cloth}"}},
                   "00000000000000d1":{{"name":"Bench"}}}}"#
            ),
        )
    }

    #[test]
    fn deleting_an_instance_takes_what_was_added_under_what_it_shows() {
        let reg = engine_registry();
        let (lamp_post, pole) = (prefab(), pole());
        let load = |id: &str| match id {
            PREFAB_ID => Some(lamp_post.clone()),
            POLE_ID => Some(pole.clone()),
            _ => None,
        };
        let scene = decorated_lamp_post();
        assert_eq!(check_with_prefabs(&scene, &reg, &load), vec![]);
        let ops = delete(&scene, &["00000000000000b1"], &load);
        let gone: Vec<&str> = ops.iter().map(|op| op.path()[1].as_str()).collect();
        assert_eq!(
            gone,
            [
                "00000000000000b1",
                "00000000000000b2",
                "00000000000000c1",
                "00000000000000c2",
                "00000000000000c3"
            ]
        );
        let mut doc = Document::new(scene.clone()).unwrap();
        doc.apply_all(&ops).unwrap();
        assert_eq!(check_with_prefabs(doc.root(), &reg, &load), vec![]);
        let names: Vec<String> = flatten(doc.root(), &load)
            .unwrap()
            .into_iter()
            .map(|r| r.name)
            .collect();
        assert_eq!(names, ["Bench"]);
        // Without its prefab an instance shows nothing, so only its own subtree goes.
        assert_eq!(delete(&scene, &["00000000000000b1"], &|_| None).len(), 1);
    }

    #[test]
    fn reparent_refuses_a_parent_that_an_instance_beneath_shows() {
        let (lamp_post, pole) = (prefab(), pole());
        let load = |id: &str| match id {
            PREFAB_ID => Some(lamp_post.clone()),
            POLE_ID => Some(pole.clone()),
            _ => None,
        };
        let scene = decorated_lamp_post();
        let lamp = derived("00000000000000b1", "00000000000000a2");
        let bulb = derived("00000000000000b1", "00000000000000a3");
        let cloth = derived("00000000000000b2", "00000000000000f2");
        // Beneath what it shows, directly, through an entity added there, or through a nested
        // instance's shown entity.
        for parent in [&lamp, &bulb, "00000000000000c2", &cloth, "00000000000000c3"] {
            let r = reparent(&scene, "00000000000000b1", Some(parent), None, &load);
            assert_eq!(
                r.unwrap_err(),
                "an entity cannot move beneath itself",
                "{parent}"
            );
        }
        // Other moves stay allowed: the flag onto the bulb, the bench under the lamp.
        for (id, parent) in [("00000000000000c1", &bulb), ("00000000000000d1", &lamp)] {
            let ops = reparent(&scene, id, Some(parent), None, &load).unwrap();
            let mut doc = Document::new(scene.clone()).unwrap();
            doc.apply_all(&ops).unwrap();
            assert!(flatten(doc.root(), &load).is_ok());
        }
    }

    #[test]
    fn override_components_that_are_not_an_object_change_nothing() {
        let p = prefab();
        let scene = doc_with(
            Kind::Scene,
            SCENE_ID,
            &format!(
                r#"{{"00000000000000b1":{{"name":"I","prefab":{{"asset":"{PREFAB_ID}",
                     "overrides":{{"00000000000000a2":{{"components":"oops"}}}}}}}}}}"#
            ),
        );
        let issues = check(&scene, &engine_registry());
        assert_eq!(
            issues[0].path,
            "entities.00000000000000b1.prefab.overrides.00000000000000a2.components"
        );
        let flat = flatten(&scene, &|id: &str| (id == PREFAB_ID).then(|| p.clone())).unwrap();
        let lamp = flat.iter().find(|r| r.name == "Lamp").unwrap();
        assert_eq!(lamp.components, tr(2.0));
    }

    #[test]
    fn flatten_takes_near_linear_time() {
        // 40,000 entities, 100 roots and about three children to each entity (the shape the
        // review timed: 16,000 took 12 s in a release build while flatten scanned every entity
        // for each one), a chain 5,000 deep, and 100 lamp posts with a flag under each lamp.
        // Done in well under a second even unoptimised; the bound only catches quadratic work.
        let n = 40_000u64;
        let mut parents: Vec<(u64, Option<u64>)> = (1..=n)
            .map(|i| (i, (i > 100).then(|| (i - 101) / 3 + 1)))
            .collect();
        parents.extend((n + 1..=n + 5_000).map(|i| (i, Some(i - 1))));
        let mut doc = scene_of(&parents);
        if let Some(Value::Object(ents)) = doc.get("entities").cloned().as_mut() {
            for k in 0..100u64 {
                let inst = format!("{:016x}", 400 * k + 1);
                let link = parse(format!(r#"{{"asset":"{PREFAB_ID}"}}"#).as_bytes()).unwrap();
                if let Some(Value::Object(e)) = ents.get_mut(&inst) {
                    e.insert("prefab".into(), link);
                }
                let flag = BTreeMap::from([
                    ("name".to_string(), s("Flag")),
                    ("parent".to_string(), s(&derived(&inst, "00000000000000a2"))),
                ]);
                ents.insert(format!("{:016x}", 0xf000_0000 + k), Value::Object(flag));
            }
            if let Value::Object(root) = &mut doc {
                root.insert("entities".into(), Value::Object(ents.clone()));
            }
        }
        let p = prefab();
        let flat = within(60, move || {
            flatten(&doc, &move |id: &str| (id == PREFAB_ID).then(|| p.clone()))
        })
        .unwrap();
        // Own entities, flags, and the lamp and bulb each instance shows.
        assert_eq!(flat.len(), 45_000 + 100 + 200);
        let mut seen = BTreeSet::new();
        for r in &flat {
            assert!(r.parent.as_ref().is_none_or(|p| seen.contains(p.as_str())));
            seen.insert(r.id.as_str());
        }
    }

    #[test]
    fn prefab_expansion_is_bounded_and_loads_each_prefab_once() {
        // P0 to P15, each a root with four child instances of the next (P15 a root alone):
        // a few kilobytes of files that expand to about 1.4e9 entities.
        let ids: Vec<String> = (0..16).map(|k| format!("{:032x}", 0xe00 + k)).collect();
        let mut prefabs = BTreeMap::new();
        for (k, id) in ids.iter().enumerate() {
            let mut ents = String::from(r#"{"00000000000000a0":{"name":"R"}"#);
            if let Some(next) = ids.get(k + 1) {
                for c in 1..=4 {
                    ents += &format!(
                        r#","00000000000000a{c}":{{"name":"C","parent":"00000000000000a0","prefab":{{"asset":"{next}"}}}}"#
                    );
                }
            }
            ents.push('}');
            prefabs.insert(id.clone(), doc_with(Kind::Prefab, id, &ents));
        }
        // Flattens one instance of prefab `top`; gives the result and the loads of each prefab.
        let instance_of = move |top: usize| {
            let scene = doc_with(
                Kind::Scene,
                SCENE_ID,
                &format!(
                    r#"{{"00000000000000b1":{{"name":"I","prefab":{{"asset":"{}"}}}}}}"#,
                    ids[top]
                ),
            );
            let loads = std::cell::RefCell::new(BTreeMap::<String, usize>::new());
            let load = |id: &str| {
                *loads.borrow_mut().entry(id.to_string()).or_default() += 1;
                prefabs.get(id).cloned()
            };
            let result = flatten(&scene, &load);
            (result, loads.into_inner())
        };
        let (result, loads) = within(60, move || {
            let deep = instance_of(0);
            (deep, instance_of(12))
        });
        let (deep, deep_loads) = result;
        let e = deep.unwrap_err();
        assert!(
            e.ends_with("entities: more than 250000 entities once prefabs are expanded"),
            "{e}"
        );
        assert_eq!(deep_loads.len(), 16);
        assert!(deep_loads.values().all(|n| *n == 1), "{deep_loads:?}");
        // Four levels fit: 1 + 4 + 16 + 64 entities, each prefab loaded once.
        let (shallow, shallow_loads) = loads;
        assert_eq!(shallow.unwrap().len(), 85);
        assert_eq!(shallow_loads.values().collect::<Vec<_>>(), [&1; 4]);
    }

    #[test]
    fn duplicate_copies_subtrees_with_new_ids() {
        let mut doc = Document::new(new_document(Kind::Scene, SCENE_ID, "D")).unwrap();
        doc.apply_all(&create("00000000000000a1", "Tree", None, None, tr(1.0)))
            .unwrap();
        doc.apply_all(&create(
            "00000000000000a2",
            "Leaves",
            Some("00000000000000a1"),
            None,
            BTreeMap::new(),
        ))
        .unwrap();
        let mut ids = Ids::seeded(9);
        let (ops, tops) = duplicate(doc.root(), &["00000000000000a1"], || ids.entity());
        doc.apply_all(&ops).unwrap();
        assert_eq!(entities(doc.root()).len(), 4);
        let copy = &tops[0];
        assert_eq!(
            str_field(&entities(doc.root())[copy], "name"),
            Some("Tree copy")
        );
        assert_eq!(children(doc.root(), Some(copy)).len(), 1);
        assert_eq!(check(doc.root(), &engine_registry()), vec![]);
    }
}
