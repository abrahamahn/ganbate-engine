//! Editable documents: a JSON object changed only by `Op`s, each of which yields its own
//! inverse, so every edit can be undone, redone, sent to other views of the same document and
//! saved as canonical text.
use std::collections::BTreeMap;

use ganbate_core::json::Value;

/// Object keys from the document root to a value. Arrays are values, not containers that
/// edits reach into: an edit replaces a whole array, so concurrent views never disagree about
/// an index.
pub type Path = Vec<String>;

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// Put `value` at `path`, creating missing parent objects, replacing what was there.
    Set { path: Path, value: Value },
    /// Take away the key at `path` (nothing happens if it is absent).
    Remove { path: Path },
}

impl Op {
    pub fn path(&self) -> &Path {
        match self {
            Op::Set { path, .. } | Op::Remove { path } => path,
        }
    }
}

/// A document and its revision, which counts the changes applied since it was opened.
#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    root: Value,
    revision: u64,
}

pub fn get<'a>(root: &'a Value, path: &[String]) -> Option<&'a Value> {
    path.iter().try_fold(root, |v, key| v.get(key))
}

fn parent_object<'a>(
    root: &'a mut Value,
    path: &[String],
    create: bool,
) -> Result<Option<&'a mut BTreeMap<String, Value>>, String> {
    let mut v = root;
    for key in path {
        let Value::Object(map) = v else {
            return Err(format!("{key:?} is inside a value that is not an object"));
        };
        if !map.contains_key(key) {
            if !create {
                return Ok(None);
            }
            map.insert(key.clone(), Value::Object(BTreeMap::new()));
        }
        v = map.get_mut(key).expect("present");
    }
    match v {
        Value::Object(map) => Ok(Some(map)),
        _ => Err(format!("{:?} is not an object", path.join("/"))),
    }
}

impl Document {
    pub fn new(root: Value) -> Result<Self, String> {
        if !matches!(root, Value::Object(_)) {
            return Err("a document is a JSON object".into());
        }
        Ok(Self { root, revision: 0 })
    }
    /// Reopens a document changed outside the editor. In-flight edits to its old contents
    /// must see a different revision, even when both files had never been edited in Studio.
    pub fn reload(&mut self, root: Value) -> Result<(), String> {
        if !matches!(root, Value::Object(_)) {
            return Err("a document is a JSON object".into());
        }
        self.root = root;
        self.revision += 1;
        Ok(())
    }
    pub fn root(&self) -> &Value {
        &self.root
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn get(&self, path: &[String]) -> Option<&Value> {
        get(&self.root, path)
    }

    /// Applies `op` and returns the op that undoes it.
    pub fn apply(&mut self, op: &Op) -> Result<Op, String> {
        let path = op.path();
        let Some((last, parents)) = path.split_last() else {
            return Err("an edit needs a path below the root".into());
        };
        // Undoing a set that created parent objects removes the outermost one it created, so
        // an undone document is exactly the one before.
        let missing = (1..=path.len()).find(|n| self.get(&path[..*n]).is_none());
        let inverse = match (self.get(path), missing) {
            (Some(old), _) => Op::Set {
                path: path.clone(),
                value: old.clone(),
            },
            (None, Some(n)) if matches!(op, Op::Set { .. }) => Op::Remove {
                path: path[..n].to_vec(),
            },
            (None, _) => Op::Remove { path: path.clone() },
        };
        match op {
            Op::Set { value, .. } => {
                let map = parent_object(&mut self.root, parents, true)?.expect("created");
                map.insert(last.clone(), value.clone());
            }
            Op::Remove { .. } => {
                if let Some(map) = parent_object(&mut self.root, parents, false)? {
                    map.remove(last);
                }
            }
        }
        self.revision += 1;
        Ok(inverse)
    }

    /// Takes back an edit that never became visible (`undo` from `apply_all`), restoring the
    /// revision it had, so other views of the document are not made stale by it.
    pub fn rollback(&mut self, undo: &[Op], revision: u64) -> Result<(), String> {
        self.apply_all(undo)?;
        self.revision = revision;
        Ok(())
    }

    /// Applies `ops` in order, all or nothing; returns their inverses in undo order. A batch
    /// that fails leaves the document and its revision as they were, so other views of it are
    /// not made stale by an edit that never happened.
    pub fn apply_all(&mut self, ops: &[Op]) -> Result<Vec<Op>, String> {
        let start = self.revision;
        let mut inverses = Vec::with_capacity(ops.len());
        for op in ops {
            match self.apply(op) {
                Ok(inv) => inverses.push(inv),
                Err(e) => {
                    for inv in inverses.iter().rev() {
                        self.apply(inv).expect("an inverse applies");
                    }
                    self.revision = start;
                    return Err(e);
                }
            }
        }
        inverses.reverse();
        Ok(inverses)
    }
}

/// One undoable step: what it did and how to take it back.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub label: String,
    /// Steps with the same key that follow each other merge into one (a dragged slider).
    pub merge: Option<String>,
    pub ops: Vec<Op>,
    pub undo: Vec<Op>,
}

/// Undo and redo stacks for one document.
#[derive(Clone, Debug, Default)]
pub struct History {
    undo: Vec<Step>,
    redo: Vec<Step>,
    limit: usize,
    /// The undo depth at the last save, or None when that state is no longer reachable.
    saved_at: Option<usize>,
    /// Whether the next step may merge into the top one.
    open: bool,
}

impl History {
    pub fn new(limit: usize) -> Self {
        Self {
            undo: vec![],
            redo: vec![],
            limit: limit.max(1),
            saved_at: Some(0),
            open: false,
        }
    }

    /// Applies an edit and records it; returns the ops other views need to apply.
    pub fn edit(
        &mut self,
        doc: &mut Document,
        label: &str,
        merge: Option<&str>,
        ops: Vec<Op>,
    ) -> Result<Vec<Op>, String> {
        let undo = doc.apply_all(&ops)?;
        self.record(label, merge, ops.clone(), undo);
        Ok(ops)
    }

    /// Records an edit already applied to the document (`undo` from `Document::apply_all`), so
    /// a caller can check the result before it enters the history.
    pub fn record(&mut self, label: &str, merge: Option<&str>, ops: Vec<Op>, undo: Vec<Op>) {
        self.redo.clear();
        if self.saved_at.is_some_and(|s| s > self.undo.len()) {
            self.saved_at = None;
        }
        let mergeable = self.open
            && merge.is_some()
            && self
                .undo
                .last()
                .is_some_and(|top| top.merge.as_deref() == merge)
            && self.saved_at != Some(self.undo.len());
        if mergeable {
            let top = self.undo.last_mut().expect("top");
            top.ops.extend(ops);
            let mut combined = undo;
            combined.append(&mut top.undo);
            top.undo = combined;
        } else {
            self.undo.push(Step {
                label: label.into(),
                merge: merge.map(Into::into),
                ops,
                undo,
            });
            if self.undo.len() > self.limit {
                self.undo.remove(0);
                self.saved_at = self.saved_at.and_then(|s| s.checked_sub(1));
            }
        }
        self.open = merge.is_some();
    }

    /// Ends merging: the next edit starts a new step even if its merge key matches.
    pub fn close(&mut self) {
        self.open = false;
    }

    pub fn undo(&mut self, doc: &mut Document) -> Result<Option<(String, Vec<Op>)>, String> {
        self.open = false;
        let Some(step) = self.undo.pop() else {
            return Ok(None);
        };
        // A step that does not apply stays where it was (apply_all left the document as is).
        if let Err(e) = doc.apply_all(&step.undo) {
            self.undo.push(step);
            return Err(e);
        }
        let out = (step.label.clone(), step.undo.clone());
        self.redo.push(step);
        Ok(Some(out))
    }

    pub fn redo(&mut self, doc: &mut Document) -> Result<Option<(String, Vec<Op>)>, String> {
        self.open = false;
        let Some(step) = self.redo.pop() else {
            return Ok(None);
        };
        if let Err(e) = doc.apply_all(&step.ops) {
            self.redo.push(step);
            return Err(e);
        }
        let out = (step.label.clone(), step.ops.clone());
        self.undo.push(step);
        Ok(Some(out))
    }

    pub fn mark_saved(&mut self) {
        self.saved_at = Some(self.undo.len());
        self.open = false;
    }
    /// Whether the document differs from what was last saved.
    pub fn dirty(&self) -> bool {
        self.saved_at != Some(self.undo.len())
    }
    pub fn can_undo(&self) -> Option<&str> {
        self.undo.last().map(|s| s.label.as_str())
    }
    pub fn can_redo(&self) -> Option<&str> {
        self.redo.last().map(|s| s.label.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ganbate_core::json::parse;

    fn p(s: &str) -> Path {
        s.split('/').map(String::from).collect()
    }
    fn num(n: f64) -> Value {
        Value::Number(n)
    }

    #[test]
    fn ops_return_their_inverses() {
        let start = parse(br#"{"a":{"b":1},"c":[1,2]}"#).unwrap();
        let mut d = Document::new(start.clone()).unwrap();
        let i1 = d
            .apply(&Op::Set {
                path: p("a/b"),
                value: num(2.0),
            })
            .unwrap();
        let i2 = d
            .apply(&Op::Set {
                path: p("x/y/z"),
                value: num(3.0),
            })
            .unwrap();
        let i3 = d.apply(&Op::Remove { path: p("c") }).unwrap();
        assert_eq!(d.get(&p("x/y/z")), Some(&num(3.0)));
        assert_eq!(d.revision(), 3);
        for inv in [i3, i2, i1] {
            d.apply(&inv).unwrap();
        }
        // Exactly the document before, created parents included.
        assert_eq!(d.root(), &start);
    }

    #[test]
    fn a_failed_batch_changes_nothing() {
        let start = parse(br#"{"a":{"b":1},"s":"text"}"#).unwrap();
        let mut d = Document::new(start.clone()).unwrap();
        let r = d.apply_all(&[
            Op::Set {
                path: p("a/b"),
                value: num(5.0),
            },
            Op::Set {
                path: p("s/inner"),
                value: num(1.0),
            },
        ]);
        assert!(r.is_err());
        assert_eq!(d.root(), &start);
    }

    #[test]
    fn a_failed_batch_keeps_the_revision() {
        // The first op applies and is undone when the second fails; a view at revision 1 must
        // still be current afterwards.
        let mut d = Document::new(parse(br#"{"e":{"name":"A"}}"#).unwrap()).unwrap();
        d.apply(&Op::Set {
            path: p("e/name"),
            value: Value::String("B".into()),
        })
        .unwrap();
        let before = d.clone();
        let r = d.apply_all(&[
            Op::Set {
                path: p("e/name"),
                value: Value::String("X".into()),
            },
            Op::Set {
                path: p("e/name/inner"),
                value: num(1.0),
            },
        ]);
        assert!(r.unwrap_err().contains("is not an object"));
        assert_eq!(d, before);
        assert_eq!(d.revision(), 1);
    }

    #[test]
    fn undo_redo_merge_and_dirty() {
        let mut d = Document::new(parse(br#"{"v":0}"#).unwrap()).unwrap();
        let mut h = History::new(100);
        assert!(!h.dirty());
        for i in 1..=5 {
            h.edit(
                &mut d,
                "Drag",
                Some("slider"),
                vec![Op::Set {
                    path: p("v"),
                    value: num(f64::from(i)),
                }],
            )
            .unwrap();
        }
        h.close();
        h.edit(
            &mut d,
            "Name",
            None,
            vec![Op::Set {
                path: p("name"),
                value: Value::String("n".into()),
            }],
        )
        .unwrap();
        assert!(h.dirty());
        assert_eq!(h.can_undo(), Some("Name"));
        h.undo(&mut d).unwrap();
        assert_eq!(d.get(&p("name")), None);
        // The five drags undo as one step.
        h.undo(&mut d).unwrap();
        assert_eq!(d.get(&p("v")), Some(&num(0.0)));
        assert!(!h.dirty());
        assert_eq!(h.can_undo(), None);
        h.redo(&mut d).unwrap();
        assert_eq!(d.get(&p("v")), Some(&num(5.0)));
        h.mark_saved();
        assert!(!h.dirty());
        h.undo(&mut d).unwrap();
        assert!(h.dirty());
        // A new edit after an undo drops the redo branch, and the saved state with it.
        h.edit(
            &mut d,
            "Other",
            None,
            vec![Op::Set {
                path: p("v"),
                value: num(9.0),
            }],
        )
        .unwrap();
        assert_eq!(h.can_redo(), None);
        assert!(h.dirty());
    }

    #[test]
    fn history_keeps_its_limit() {
        let mut d = Document::new(parse(br#"{}"#).unwrap()).unwrap();
        let mut h = History::new(3);
        for i in 0..10 {
            h.edit(
                &mut d,
                "Set",
                None,
                vec![Op::Set {
                    path: p("k"),
                    value: num(f64::from(i)),
                }],
            )
            .unwrap();
        }
        let mut undone = 0;
        while h.undo(&mut d).unwrap().is_some() {
            undone += 1;
        }
        assert_eq!(undone, 3);
        assert_eq!(d.get(&p("k")), Some(&num(6.0)));
        assert!(h.dirty());
    }
    #[test]
    fn a_step_that_does_not_apply_stays_in_the_history() {
        let mut doc = Document::new(parse(br#"{"entities":{"a":{"name":"A"}}}"#).unwrap()).unwrap();
        let mut h = History::new(8);
        let ops = vec![Op::Set {
            path: vec!["entities".into(), "a".into(), "name".into()],
            value: Value::String("B".into()),
        }];
        let undo = doc.apply_all(&ops).unwrap();
        h.record("Rename", None, ops, undo);
        // Something else turned "a" into a string, so the undo cannot reach its name.
        doc.apply_all(&[Op::Set {
            path: vec!["entities".into(), "a".into()],
            value: Value::String("not an object".into()),
        }])
        .unwrap();
        let before = doc.root().clone();
        assert!(h.undo(&mut doc).is_err());
        assert_eq!(doc.root(), &before, "the failed undo changed nothing");
        assert_eq!(h.can_undo(), Some("Rename"), "and the step is still there");
        assert_eq!(h.can_redo(), None);
    }
}
