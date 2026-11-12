//! A project folder: its manifest (`ganbate.project.json`) and the index of its assets by id.
//!
//! Engine documents (`*.scene.json`, `*.prefab.json`, `*.material.json`, `*.data.json`) carry
//! their own `id`. Every other asset file (models, images, sounds, Look files) has a sidecar
//! `<file>.meta.json` holding its id, kind and import settings; `scan` writes a missing
//! sidecar with a fresh id when asked to. Two files claiming one id are an error naming both.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ganbate_core::id::{Ids, is_asset_id};
use ganbate_core::json::{self, Value};

pub const MANIFEST: &str = "ganbate.project.json";
pub const PROJECT_VERSION: u64 = 1;
/// Most files a scan visits, and deepest folder nesting.
const MAX_FILES: usize = 200_000;
const MAX_DEPTH: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Scene,
    Prefab,
    Material,
    Model,
    Texture,
    Look,
    Sound,
    Data,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Scene => "scene",
            Kind::Prefab => "prefab",
            Kind::Material => "material",
            Kind::Model => "model",
            Kind::Texture => "texture",
            Kind::Look => "look",
            Kind::Sound => "sound",
            Kind::Data => "data",
        }
    }
    /// Whether files of this kind are engine documents with their own id.
    pub fn self_identified(self) -> bool {
        matches!(
            self,
            Kind::Scene | Kind::Prefab | Kind::Material | Kind::Data
        )
    }
    /// The kind a file name implies, if it is an asset at all.
    pub fn of_file(name: &str) -> Option<Kind> {
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".meta.json") || lower.ends_with(".schema.json") {
            return None;
        }
        for (suffix, kind) in [
            (".scene.json", Kind::Scene),
            (".prefab.json", Kind::Prefab),
            (".material.json", Kind::Material),
            (".data.json", Kind::Data),
            (".look.json", Kind::Look),
            (".glb", Kind::Model),
            (".png", Kind::Texture),
            (".jpg", Kind::Texture),
            (".jpeg", Kind::Texture),
            (".webp", Kind::Texture),
            (".wav", Kind::Sound),
            (".ogg", Kind::Sound),
        ] {
            if lower.ends_with(suffix) {
                return Some(kind);
            }
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub id: String,
    pub kind: Kind,
    /// Relative to the project root, with `/` separators.
    pub path: String,
}

#[derive(Clone, Debug)]
pub struct Manifest {
    pub id: String,
    pub name: String,
    pub content: Vec<String>,
    pub schemas: Vec<String>,
    pub editor: Vec<String>,
    /// Optional external or staging art directories (`ganbate-art`, `tripo3d-queue`, etc.)
    /// inspected read-only by Studio's intake bridge and never auto-indexed into `content`.
    pub intake: Vec<String>,
    pub startup: Option<String>,
    pub platforms: BTreeMap<String, Value>,
}

fn strings(v: Option<&Value>, key: &str) -> Result<Vec<String>, String> {
    match v {
        None => Ok(vec![]),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| x.string().map(String::from))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{key}: {e}")),
        Some(_) => Err(format!("{key}: expected a list of folders")),
    }
}

fn relative_folder(f: &str) -> Result<(), String> {
    let p = Path::new(f);
    if f.is_empty()
        || p.is_absolute()
        || p.components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(format!("{f:?} is not a folder inside the project"));
    }
    Ok(())
}

fn intake_folder(f: &str) -> Result<(), String> {
    let trimmed = f.trim();
    if trimmed.is_empty() || trimmed.len() > 1024 || f.contains('\0') {
        return Err(format!("{f:?} is not a valid intake directory path"));
    }
    Ok(())
}

impl Manifest {
    pub fn parse(v: &Value) -> Result<Self, String> {
        let Value::Object(map) = v else {
            return Err("the manifest is a JSON object".into());
        };
        for k in map.keys() {
            if !matches!(
                k.as_str(),
                "ganbate"
                    | "version"
                    | "id"
                    | "name"
                    | "content"
                    | "schemas"
                    | "editor"
                    | "intake"
                    | "startup"
                    | "platforms"
            ) {
                return Err(format!("unknown manifest key {k:?}"));
            }
        }
        if v.get("ganbate").and_then(|x| x.string().ok()) != Some("project") {
            return Err("not a project manifest (\"ganbate\": \"project\")".into());
        }
        if v.get("version").and_then(|x| x.unsigned().ok()) != Some(PROJECT_VERSION) {
            return Err("unsupported project version".into());
        }
        let id = v
            .get("id")
            .and_then(|x| x.string().ok())
            .filter(|s| is_asset_id(s))
            .ok_or("the manifest needs an asset id")?
            .to_string();
        let name = v
            .get("name")
            .and_then(|x| x.string().ok())
            .unwrap_or("")
            .to_string();
        let content = strings(v.get("content"), "content")?;
        let schemas = strings(v.get("schemas"), "schemas")?;
        let editor = strings(v.get("editor"), "editor")?;
        let intake = strings(v.get("intake"), "intake")?;
        for f in content.iter().chain(&schemas).chain(&editor) {
            relative_folder(f)?;
        }
        if intake.len() > 32 {
            return Err("intake: at most 32 directories may be configured".into());
        }
        for f in &intake {
            intake_folder(f)?;
        }
        if content.is_empty() {
            return Err("the manifest lists no content folders".into());
        }
        let startup = match v.get("startup") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) if is_asset_id(s) => Some(s.clone()),
            Some(_) => return Err("startup is an asset id".into()),
        };
        let platforms = match v.get("platforms") {
            None => BTreeMap::new(),
            Some(Value::Object(m)) => m.clone(),
            Some(_) => return Err("platforms is an object".into()),
        };
        Ok(Self {
            id,
            name,
            content,
            schemas,
            editor,
            intake,
            startup,
            platforms,
        })
    }

    pub fn to_value(&self) -> Value {
        let list = |v: &[String]| Value::Array(v.iter().cloned().map(Value::String).collect());
        let mut m = BTreeMap::from([
            ("ganbate".to_string(), Value::String("project".into())),
            ("version".into(), Value::Number(PROJECT_VERSION as f64)),
            ("id".into(), Value::String(self.id.clone())),
            ("name".into(), Value::String(self.name.clone())),
            ("content".into(), list(&self.content)),
            ("schemas".into(), list(&self.schemas)),
            ("editor".into(), list(&self.editor)),
        ]);
        if !self.intake.is_empty() {
            m.insert("intake".into(), list(&self.intake));
        }
        if let Some(s) = &self.startup {
            m.insert("startup".into(), Value::String(s.clone()));
        }
        if !self.platforms.is_empty() {
            m.insert("platforms".into(), Value::Object(self.platforms.clone()));
        }
        Value::Object(m)
    }
}

/// What a scan found besides the index.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScanReport {
    /// Sidecars written for files that had none.
    pub created: Vec<String>,
    /// Files without a sidecar when `scan` was not allowed to write one.
    pub unregistered: Vec<String>,
    /// Problems that did not stop the scan (unreadable documents, bad sidecars).
    pub problems: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Project {
    pub root: PathBuf,
    pub manifest: Manifest,
    by_id: BTreeMap<String, Entry>,
}

/// Writes `v` as canonical JSON (atomic). Text the reader refuses (too large, too deep, too
/// many values) is not written: no tool could open that file again.
pub fn write_json(path: &Path, v: &Value) -> Result<(), String> {
    // `to_canonical` enforces the reader's byte, depth, value-count and finite-number limits.
    let text =
        json::to_canonical(v).map_err(|e| format!("{}: not written: {e}", path.display()))?;
    debug_assert!(json::parse(text.as_bytes()).is_ok());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    ganbate_assets::write_asset(path, text.as_bytes())
        .map_err(|e| format!("{}: {e}", path.display()))
}

pub fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = ganbate_assets::read_asset(path)?;
    json::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn slash_path(root: &Path, p: &Path) -> Option<String> {
    let rel = p.strip_prefix(root).ok()?;
    let parts: Option<Vec<&str>> = rel.components().map(|c| c.as_os_str().to_str()).collect();
    Some(parts?.join("/"))
}

impl Project {
    /// Creates a new project in `root` (which must not hold a manifest yet).
    pub fn create(root: &Path, name: &str, ids: &mut Ids) -> Result<Self, String> {
        let manifest_path = root.join(MANIFEST);
        if manifest_path.exists() {
            return Err(format!("{} already exists", manifest_path.display()));
        }
        let manifest = Manifest {
            id: ids.asset(),
            name: name.into(),
            content: vec!["content".into()],
            schemas: vec!["schemas".into()],
            editor: vec!["editor".into()],
            intake: vec![],
            startup: None,
            platforms: BTreeMap::new(),
        };
        for d in ["content", "schemas", "editor"] {
            std::fs::create_dir_all(root.join(d)).map_err(|e| e.to_string())?;
        }
        let ignore = root.join(".gitignore");
        if !ignore.exists() {
            std::fs::write(&ignore, "/.ganbate/\n/build/\n").map_err(|e| e.to_string())?;
        }
        write_json(&manifest_path, &manifest.to_value())?;
        Ok(Self {
            root: root.to_path_buf(),
            manifest,
            by_id: BTreeMap::new(),
        })
    }

    /// Opens the project in `root` without scanning it.
    pub fn open(root: &Path) -> Result<Self, String> {
        let real_root = root
            .canonicalize()
            .map_err(|e| format!("{}: {e}", root.display()))?;
        within_project(&real_root, &root.join(MANIFEST))?;
        let manifest = Manifest::parse(&read_json(&root.join(MANIFEST))?)?;
        Ok(Self {
            root: root.to_path_buf(),
            manifest,
            by_id: BTreeMap::new(),
        })
    }

    /// Rebuilds the index from the content folders. With `ids`, files without a sidecar get
    /// one (with a fresh id); without, they are reported as unregistered.
    pub fn scan(&mut self, mut ids: Option<&mut Ids>) -> Result<ScanReport, String> {
        let mut report = ScanReport::default();
        let real_root = self
            .root
            .canonicalize()
            .map_err(|e| format!("{}: {e}", self.root.display()))?;
        let files = project_files(&self.root, &real_root, &self.manifest.content)?;
        let mut by_id = BTreeMap::new();
        let mut pending = Vec::new();
        for file in files {
            let Some(name) = file.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let rel = slash_path(&self.root, &file).ok_or("a file outside the project")?;
            let Some(kind) = Kind::of_file(name) else {
                if let Some(source) = rel.strip_suffix(".meta.json") {
                    if Kind::of_file(source).is_some_and(|k| !k.self_identified())
                        && !std::fs::symlink_metadata(self.root.join(source))
                            .is_ok_and(|m| m.is_file())
                    {
                        report.problems.push(format!("{rel}: its source file {source} is missing; move a source and its sidecar together"));
                    }
                }
                continue;
            };
            within_project(&real_root, &file)?;
            let id = if kind.self_identified() {
                match read_json(&file) {
                    Ok(v) => match v.get("id").and_then(|x| x.string().ok()) {
                        Some(id) if is_asset_id(id) => id.to_string(),
                        _ => {
                            report.problems.push(format!("{rel}: no valid \"id\""));
                            continue;
                        }
                    },
                    Err(e) => {
                        report.problems.push(e);
                        continue;
                    }
                }
            } else {
                let meta = sidecar_path(&file);
                match std::fs::symlink_metadata(&meta) {
                    Ok(_) => {
                        within_project(&real_root, &meta)?;
                        match read_json(&meta).and_then(|v| sidecar_id(&v, kind)) {
                            Ok(id) => id,
                            Err(e) => {
                                report.problems.push(format!("{rel}.meta.json: {e}"));
                                continue;
                            }
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        pending.push((file, rel, kind));
                        continue;
                    }
                    Err(e) => return Err(format!("{}: {e}", meta.display())),
                }
            };
            index_entry(&mut by_id, id, kind, rel)?;
        }
        // Detect all existing duplicates before assigning new identities. If another writer
        // creates a sidecar during the scan, index its actual identity without replacing it.
        for (file, rel, kind) in pending {
            let Some(ids) = ids.as_deref_mut() else {
                report.unregistered.push(rel);
                continue;
            };
            let id = loop {
                let id = ids.asset();
                if !by_id.contains_key(&id) {
                    break id;
                }
            };
            within_project(&real_root, &file)?;
            let meta = sidecar_path(&file);
            let text = json::to_canonical(&new_sidecar(&id, kind))?;
            let id = match ganbate_assets::create_asset(&meta, text.as_bytes()) {
                Ok(()) => {
                    report.created.push(rel.clone());
                    id
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    within_project(&real_root, &meta)?;
                    match read_json(&meta).and_then(|v| sidecar_id(&v, kind)) {
                        Ok(id) => id,
                        Err(e) => {
                            report.problems.push(format!("{rel}.meta.json: {e}"));
                            continue;
                        }
                    }
                }
                Err(e) => return Err(format!("{}: {e}", meta.display())),
            };
            index_entry(&mut by_id, id, kind, rel)?;
        }
        self.by_id = by_id;
        Ok(report)
    }

    pub fn entries(&self) -> impl Iterator<Item = &Entry> {
        self.by_id.values()
    }
    pub fn entry(&self, id: &str) -> Option<&Entry> {
        self.by_id.get(id)
    }
    pub fn path_of(&self, id: &str) -> Option<PathBuf> {
        self.entry(id).map(|e| self.root.join(&e.path))
    }
    pub fn find_path(&self, rel: &str) -> Option<&Entry> {
        self.by_id.values().find(|e| e.path == rel)
    }

    /// Reads an engine document (scene, prefab, material) by id.
    pub fn read_document(&self, id: &str) -> Result<Value, String> {
        let p = self.path_of(id).ok_or_else(|| format!("no asset {id}"))?;
        read_json(&p)
    }

    /// Checks that `rel` can hold an asset file that a scan finds again: a path in a content
    /// folder, not in a hidden folder nor deeper than a scan looks, with a recognized asset
    /// extension.
    pub fn check_asset_path(&self, rel: &str) -> Result<Kind, String> {
        relative_folder(rel)?;
        let path = Path::new(rel);
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let kind = Kind::of_file(name)
            .ok_or_else(|| format!("{rel}: unsupported asset file extension"))?;
        let below: Vec<&Path> = self
            .manifest
            .content
            .iter()
            .filter_map(|f| path.strip_prefix(f).ok())
            .collect();
        if below.is_empty() {
            return Err(format!(
                "{rel} is not in a content folder ({}), so a scan would not find it",
                self.manifest.content.join(", ")
            ));
        }
        for r in below {
            if r.components().count() > MAX_DEPTH + 1 {
                return Err(format!("{rel}: folders nest too deeply"));
            }
            if r.components()
                .any(|c| c.as_os_str().to_str().is_none_or(|s| s.starts_with('.')))
            {
                return Err(format!("{rel}: scans skip hidden files and folders"));
            }
        }
        Ok(kind)
    }

    /// Updates the indexed path of an asset after its file (and sidecar, if any) moved on disk.
    pub fn relocate_entry(&mut self, id: &str, rel: &str) -> Result<(), String> {
        let kind = self
            .by_id
            .get(id)
            .ok_or_else(|| format!("no asset {id}"))?
            .kind;
        if self.check_asset_path(rel)? != kind {
            return Err(format!("{rel} is not a {} file", kind.name()));
        }
        if let Some(other) = self.find_path(rel)
            && other.id != id
        {
            return Err(format!(
                "{rel} already holds {} {}",
                other.kind.name(),
                other.id
            ));
        }
        self.by_id.get_mut(id).expect("entry").path = rel.to_string();
        Ok(())
    }

    /// Indexes a sidecar-identified asset written into a content folder.
    pub fn register_sidecar_asset(
        &mut self,
        id: String,
        kind: Kind,
        rel: String,
    ) -> Result<(), String> {
        if kind.self_identified() {
            return Err(format!("{} files carry their own id", kind.name()));
        }
        if !is_asset_id(&id) {
            return Err("invalid asset id".into());
        }
        if self.check_asset_path(&rel)? != kind {
            return Err(format!("{rel} is not a {} file", kind.name()));
        }
        if let Some(other) = self.find_path(&rel) {
            return Err(format!(
                "{rel} already holds {} {}",
                other.kind.name(),
                other.id
            ));
        }
        index_entry(&mut self.by_id, id, kind, rel)
    }

    /// Checks that `rel` can hold a `kind` document that a scan finds again: a path in a content
    /// folder, not in a hidden folder nor deeper than a scan looks, whose file name ends as the
    /// kind's do (`.scene.json`, `.prefab.json`, `.material.json`).
    pub fn check_document_path(&self, rel: &str, kind: Kind) -> Result<(), String> {
        relative_folder(rel)?;
        if !kind.self_identified() && kind != Kind::Look {
            return Err(format!("{rel}: {} files are not documents", kind.name()));
        }
        let path = Path::new(rel);
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if Kind::of_file(name) != Some(kind) {
            return Err(format!(
                "{rel}: a {0} document's file name ends in .{0}.json",
                kind.name()
            ));
        }
        // The folders below each content folder that holds the path, as a scan walks them.
        let below: Vec<&Path> = self
            .manifest
            .content
            .iter()
            .filter_map(|f| path.strip_prefix(f).ok())
            .collect();
        if below.is_empty() {
            return Err(format!(
                "{rel} is not in a content folder ({}), so a scan would not find it",
                self.manifest.content.join(", ")
            ));
        }
        for r in below {
            if r.components().count() > MAX_DEPTH + 1 {
                return Err(format!("{rel}: folders nest too deeply"));
            }
            if r.components()
                .any(|c| c.as_os_str().to_str().is_none_or(|s| s.starts_with('.')))
            {
                return Err(format!("{rel}: scans skip hidden files and folders"));
            }
        }
        Ok(())
    }

    /// `write_document_with` without `overwrite`: creates a document, or saves one over its own
    /// file, and refuses to replace any other file.
    pub fn write_document(&mut self, rel: &str, kind: Kind, doc: &Value) -> Result<(), String> {
        self.write_document_with(rel, kind, doc, false)
    }

    /// Writes a document (canonical text, atomic) to `rel` and indexes it. `rel` must pass
    /// `check_document_path`. A new file is created and the document's own file is replaced
    /// (a save); any other file already at `rel` is replaced only when `overwrite` is true,
    /// and whatever the index had there leaves it.
    pub fn write_document_with(
        &mut self,
        rel: &str,
        kind: Kind,
        doc: &Value,
        overwrite: bool,
    ) -> Result<(), String> {
        self.check_document_path(rel, kind)?;
        if kind == Kind::Look {
            return Err("a Look has a sidecar id; use create_look or save_look".into());
        }
        let id = doc
            .get("id")
            .and_then(|x| x.string().ok())
            .filter(|s| is_asset_id(s))
            .ok_or("a document needs an id")?
            .to_string();
        if let Some(e) = self.by_id.get(&id) {
            if e.path != rel {
                return Err(format!("{id} is already {}", e.path));
            }
        }
        let file = self.root.join(rel);
        if !overwrite && file.exists() && !self.holds(rel, &id) {
            let what = match self.find_path(rel) {
                Some(e) => format!(" (it holds {} {})", e.kind.name(), e.id),
                None => String::new(),
            };
            return Err(format!(
                "{rel} already exists{what}; not replacing it with document {id}"
            ));
        }
        write_json(&file, doc)?;
        self.by_id.retain(|other, e| *other == id || e.path != rel);
        self.by_id.insert(
            id.clone(),
            Entry {
                id,
                kind,
                path: rel.into(),
            },
        );
        Ok(())
    }

    /// Creates an empty Look in a content folder. Install the sidecar before the source,
    /// without replacing either file. If creation stopped between the two, retrying reuses
    /// that sidecar so references retain their original identity.
    pub fn create_look(&mut self, rel: &str, proposed_id: &str) -> Result<String, String> {
        self.check_document_path(rel, Kind::Look)?;
        if !is_asset_id(proposed_id) {
            return Err("invalid Look id".into());
        }
        if self
            .entry(proposed_id)
            .is_some_and(|entry| entry.path != rel)
        {
            return Err(format!(
                "asset id {proposed_id} is already used by another file"
            ));
        }
        let file = self.root.join(rel);
        if std::fs::symlink_metadata(&file).is_ok() {
            return Err(format!("{rel} already exists"));
        }
        let parent = file.parent().ok_or("a Look needs a content folder")?;
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        let meta = sidecar_path(&file);
        let text = json::to_canonical(&new_sidecar(proposed_id, Kind::Look))?;
        match ganbate_assets::create_asset(&meta, text.as_bytes()) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(format!("{}: {e}", meta.display())),
        }
        let id = sidecar_id(&read_json(&meta)?, Kind::Look)?;
        if self.entry(&id).is_some_and(|other| other.path != rel) {
            return Err(format!("asset id {id} is already used by another file"));
        }
        ganbate_assets::create_asset(&file, b"{}\n").map_err(|e| format!("{rel}: {e}"))?;
        self.by_id.insert(
            id.clone(),
            Entry {
                id: id.clone(),
                kind: Kind::Look,
                path: rel.into(),
            },
        );
        Ok(id)
    }

    /// Saves a Look under its existing sidecar identity. Its JSON has no embedded asset id.
    pub fn save_look(&self, id: &str, doc: &Value) -> Result<(), String> {
        let entry = self.entry(id).ok_or_else(|| format!("no asset {id}"))?;
        if entry.kind != Kind::Look {
            return Err(format!("{} is not a Look", entry.path));
        }
        let file = self.root.join(&entry.path);
        let meta = read_json(&sidecar_path(&file))?;
        if sidecar_id(&meta, Kind::Look)? != id {
            return Err(format!("{} no longer has asset id {id}", entry.path));
        }
        write_json(&file, doc)
    }

    /// Whether the file at `rel` is document `id`'s own: indexed there, or holding that id.
    fn holds(&self, rel: &str, id: &str) -> bool {
        self.by_id.get(id).is_some_and(|e| e.path == rel)
            || read_json(&self.root.join(rel))
                .is_ok_and(|v| v.get("id").and_then(|x| x.string().ok()) == Some(id))
    }

    /// The schema files the project declares (`*.schema.json` in its schema folders).
    pub fn schema_files(&self) -> Result<Vec<(String, Value)>, String> {
        let real_root = self
            .root
            .canonicalize()
            .map_err(|e| format!("{}: {e}", self.root.display()))?;
        let files = project_files(&self.root, &real_root, &self.manifest.schemas)?;
        let mut out = vec![];
        for f in files {
            if f.to_str().is_some_and(|s| s.ends_with(".schema.json")) {
                let rel = slash_path(&self.root, &f).unwrap_or_default();
                within_project(&real_root, &f)?;
                out.push((rel, read_json(&f)?));
            }
        }
        Ok(out)
    }

    /// Reads the `.meta.json` sidecar of a sidecar-identified asset (`model`, `texture`, `sound`,
    /// `look`), checking that neither the source nor its sidecar escapes the project folder.
    pub fn read_sidecar(&self, id: &str) -> Result<Value, String> {
        let entry = self.entry(id).ok_or_else(|| format!("no asset {id}"))?;
        if entry.kind.self_identified() {
            return Err(format!("{} has no .meta.json sidecar", entry.path));
        }
        let real_root = self
            .root
            .canonicalize()
            .map_err(|e| format!("{}: {e}", self.root.display()))?;
        let file = self.root.join(&entry.path);
        within_project(&real_root, &file)?;
        let meta = sidecar_path(&file);
        within_project(&real_root, &meta)?;
        let doc = read_json(&meta)?;
        if sidecar_id(&doc, entry.kind)? != id {
            return Err(format!("{} no longer has asset id {id}", entry.path));
        }
        Ok(doc)
    }

    /// Updates the editable authoring metadata (`import`, `lods`, `collider`, `sockets`) of a
    /// sidecar-identified asset while strictly preserving its `id` and `kind`.
    pub fn write_sidecar_meta(
        &self,
        id: &str,
        meta: &crate::ddc::AssetMeta,
    ) -> Result<Value, String> {
        let existing = self.read_sidecar(id)?;
        let entry = self.entry(id).expect("checked in read_sidecar");
        let updated = meta.apply_to_sidecar(&existing)?;
        if sidecar_id(&updated, entry.kind)? != id {
            return Err("sidecar identity mismatch".into());
        }
        let meta_file = sidecar_path(&self.root.join(&entry.path));
        write_json(&meta_file, &updated)?;
        Ok(updated)
    }

    /// Reads the non-destructive [`crate::ddc::ModelImportSettings`] for model `id` (or defaults
    /// when no custom settings are present).
    pub fn model_import_settings(&self, id: &str) -> crate::ddc::ModelImportSettings {
        self.read_sidecar(id)
            .and_then(|v| crate::ddc::ModelImportSettings::from_sidecar(&v))
            .unwrap_or_default()
    }
}

pub fn sidecar_path(file: &Path) -> PathBuf {
    let mut s = file.as_os_str().to_owned();
    s.push(".meta.json");
    PathBuf::from(s)
}

pub fn new_sidecar(id: &str, kind: Kind) -> Value {
    Value::Object(BTreeMap::from([
        ("ganbate".to_string(), Value::String("meta".into())),
        ("version".into(), Value::Number(1.0)),
        ("id".into(), Value::String(id.into())),
        ("kind".into(), Value::String(kind.name().into())),
        ("import".into(), Value::Object(BTreeMap::new())),
    ]))
}

fn sidecar_id(m: &Value, kind: Kind) -> Result<String, String> {
    if m.get("ganbate").and_then(|x| x.string().ok()) != Some("meta") {
        return Err("not a sidecar (\"ganbate\": \"meta\")".into());
    }
    if m.get("kind").and_then(|x| x.string().ok()) != Some(kind.name()) {
        return Err(format!("its kind is not {}", kind.name()));
    }
    m.get("id")
        .and_then(|x| x.string().ok())
        .filter(|s| is_asset_id(s))
        .map(String::from)
        .ok_or_else(|| "no valid id".into())
}

fn within_project(real_root: &Path, path: &Path) -> Result<(), String> {
    let actual = path
        .canonicalize()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if !actual.starts_with(real_root) {
        return Err(format!("{} leads outside the project", path.display()));
    }
    Ok(())
}

fn project_files(
    root: &Path,
    real_root: &Path,
    folders: &[String],
) -> Result<Vec<PathBuf>, String> {
    let mut files = vec![];
    for folder in folders {
        let dir = root.join(folder);
        match std::fs::symlink_metadata(&dir) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("{}: {e}", dir.display())),
            Ok(_) => within_project(real_root, &dir)?,
        }
        if dir.is_dir() {
            walk(&dir, 0, &mut files)?;
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn index_entry(
    entries: &mut BTreeMap<String, Entry>,
    id: String,
    kind: Kind,
    rel: String,
) -> Result<(), String> {
    if let Some(other) = entries.get(&id) {
        return Err(format!(
            "{} and {rel} have the same id {id}; give the copy a new id",
            other.path
        ));
    }
    entries.insert(
        id.clone(),
        Entry {
            id,
            kind,
            path: rel,
        },
    );
    Ok(())
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if depth > MAX_DEPTH {
        return Err(format!("{}: folders nest too deeply", dir.display()));
    }
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for e in entries {
        let e = e.map_err(|e| e.to_string())?;
        let name = e.file_name();
        if name.to_str().is_none_or(|n| n.starts_with('.')) {
            continue;
        }
        let ty = e.file_type().map_err(|e| e.to_string())?;
        if ty.is_dir() {
            walk(&e.path(), depth + 1, out)?;
        } else if ty.is_file() {
            if out.len() >= MAX_FILES {
                return Err("too many files in the content folders".into());
            }
            out.push(e.path());
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A fresh folder under the system temp directory, removed on drop.
    pub struct TempDir(pub PathBuf);
    impl TempDir {
        pub fn new(tag: &str) -> Self {
            let mut ids = Ids::from_os();
            let p = std::env::temp_dir().join(format!("ganbate-test-{tag}-{}", ids.entity()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_new_project_scans_creates_sidecars_and_finds_documents() {
        let t = TempDir::new("project");
        let mut ids = Ids::seeded(3);
        let mut p = Project::create(&t.0, "Sample", &mut ids).unwrap();
        std::fs::create_dir_all(t.0.join("content/models")).unwrap();
        std::fs::write(t.0.join("content/models/box.glb"), b"glTF").unwrap();
        std::fs::write(t.0.join("content/notes.txt"), b"not an asset").unwrap();
        let scene = ganbate_scene::scene::new_document(
            ganbate_scene::scene::Kind::Scene,
            &ids.asset(),
            "Main",
        );
        p.write_document("content/main.scene.json", Kind::Scene, &scene)
            .unwrap();

        // Without an id source the model is only reported.
        let mut q = Project::open(&t.0).unwrap();
        let r = q.scan(None).unwrap();
        assert_eq!(r.unregistered, ["content/models/box.glb"]);
        assert_eq!(q.entries().count(), 1);

        let r = q.scan(Some(&mut ids)).unwrap();
        assert_eq!(r.created, ["content/models/box.glb"]);
        assert!(t.0.join("content/models/box.glb.meta.json").exists());
        let kinds: Vec<_> = q.entries().map(|e| (e.kind, e.path.as_str())).collect();
        assert!(kinds.contains(&(Kind::Model, "content/models/box.glb")));
        assert!(kinds.contains(&(Kind::Scene, "content/main.scene.json")));
        // A second scan finds the same ids and writes nothing.
        let before: Vec<_> = q.entries().cloned().collect();
        let r = q.scan(Some(&mut ids)).unwrap();
        assert!(r.created.is_empty());
        assert_eq!(before, q.entries().cloned().collect::<Vec<_>>());
        // Moving a file with its sidecar keeps its id.
        let id = q.find_path("content/models/box.glb").unwrap().id.clone();
        std::fs::rename(
            t.0.join("content/models/box.glb"),
            t.0.join("content/box.glb"),
        )
        .unwrap();
        std::fs::rename(
            t.0.join("content/models/box.glb.meta.json"),
            t.0.join("content/box.glb.meta.json"),
        )
        .unwrap();
        q.scan(None).unwrap();
        assert_eq!(q.entry(&id).unwrap().path, "content/box.glb");
    }

    #[test]
    fn nested_or_repeated_folders_list_each_file_once() {
        let t = TempDir::new("nested");
        let mut ids = Ids::seeded(5);
        let mut p = Project::create(&t.0, "Nested", &mut ids).unwrap();
        std::fs::create_dir_all(t.0.join("content/models")).unwrap();
        std::fs::write(t.0.join("content/models/box.glb"), b"glTF").unwrap();
        let scene = ganbate_scene::scene::new_document(
            ganbate_scene::scene::Kind::Scene,
            &ids.asset(),
            "S",
        );
        p.write_document("content/models/s.scene.json", Kind::Scene, &scene)
            .unwrap();
        std::fs::write(
            t.0.join("schemas/game.schema.json"),
            br#"{"ganbate":"schema","version":1,"components":{}}"#,
        )
        .unwrap();
        // A folder inside another, and a folder listed twice.
        p.manifest.content = vec!["content/models".into(), "content".into()];
        p.manifest.schemas = vec!["schemas".into(), "schemas".into()];
        write_json(&t.0.join(MANIFEST), &p.manifest.to_value()).unwrap();
        let mut q = Project::open(&t.0).unwrap();
        let r = q.scan(Some(&mut ids)).unwrap();
        assert_eq!(r.created, ["content/models/box.glb"]);
        let paths: Vec<&str> = q.entries().map(|e| e.path.as_str()).collect();
        assert_eq!(paths.len(), 2, "{paths:?}");
        assert!(paths.contains(&"content/models/s.scene.json"));
        let schemas = q.schema_files().unwrap();
        assert_eq!(schemas.len(), 1, "{schemas:?}");
    }

    #[test]
    fn documents_replace_only_their_own_file_unless_asked() {
        let t = TempDir::new("replace");
        let mut ids = Ids::seeded(6);
        let mut p = Project::create(&t.0, "Replace", &mut ids).unwrap();
        let scene = |id: &str, name: &str| {
            ganbate_scene::scene::new_document(ganbate_scene::scene::Kind::Scene, id, name)
        };
        let name_of = |v: &Value| {
            v.get("name")
                .and_then(|n| n.string().ok())
                .map(String::from)
        };
        let (a, b) = (ids.asset(), ids.asset());
        p.write_document("content/a.scene.json", Kind::Scene, &scene(&a, "A"))
            .unwrap();
        // A save replaces the document's own file.
        p.write_document("content/a.scene.json", Kind::Scene, &scene(&a, "A2"))
            .unwrap();
        // Another document is not written over it, nor over any other file.
        let err = p
            .write_document("content/a.scene.json", Kind::Scene, &scene(&b, "B"))
            .unwrap_err();
        assert!(err.contains("content/a.scene.json"), "{err}");
        assert_eq!(
            name_of(&p.read_document(&a).unwrap()).as_deref(),
            Some("A2")
        );
        assert!(p.entry(&b).is_none());
        std::fs::write(t.0.join("content/stray.scene.json"), b"not json").unwrap();
        let err = p
            .write_document("content/stray.scene.json", Kind::Scene, &scene(&b, "B"))
            .unwrap_err();
        assert!(err.contains("content/stray.scene.json"), "{err}");
        assert_eq!(
            std::fs::read(t.0.join("content/stray.scene.json")).unwrap(),
            b"not json"
        );
        // Paths a scan would not find the document at again are refused.
        let deep = format!("content/{}b.scene.json", "d/".repeat(MAX_DEPTH + 1));
        for bad in [
            "content/b.glb",
            "content/b.prefab.json",
            "schemas/b.scene.json",
            "b.scene.json",
            "content/.drafts/b.scene.json",
            &deep,
        ] {
            let err = p
                .write_document(bad, Kind::Scene, &scene(&b, "B"))
                .unwrap_err();
            assert!(err.contains(bad), "{bad}: {err}");
            assert!(!t.0.join(bad).exists(), "{bad}");
        }
        // Asked to, a document replaces another, which leaves the index.
        p.write_document_with("content/a.scene.json", Kind::Scene, &scene(&b, "B"), true)
            .unwrap();
        assert!(p.entry(&a).is_none());
        assert_eq!(p.entry(&b).unwrap().path, "content/a.scene.json");
        let mut q = Project::open(&t.0).unwrap();
        q.scan(None).unwrap();
        assert_eq!(q.find_path("content/a.scene.json").unwrap().id, b);
        // A project opened without a scan still saves a document over its own file.
        let mut r = Project::open(&t.0).unwrap();
        r.write_document("content/a.scene.json", Kind::Scene, &scene(&b, "B2"))
            .unwrap();
        assert_eq!(
            name_of(&r.read_document(&b).unwrap()).as_deref(),
            Some("B2")
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_scan_never_registers_assets_through_an_outside_content_folder_or_sidecar() {
        use std::os::unix::fs::symlink;
        let root = TempDir::new("scan-root");
        let outside = TempDir::new("scan-outside");
        let mut ids = Ids::seeded(17);
        let mut project = Project::create(&root.0, "Confined", &mut ids).unwrap();
        let content = root.0.join("content");
        std::fs::remove_dir_all(&content).unwrap();
        std::fs::write(outside.0.join("foreign.glb"), b"outside").unwrap();
        symlink(&outside.0, &content).unwrap();
        assert!(
            project
                .scan(Some(&mut ids))
                .unwrap_err()
                .contains("outside")
        );
        assert!(!outside.0.join("foreign.glb.meta.json").exists());
        std::fs::remove_file(&content).unwrap();
        std::fs::create_dir(&content).unwrap();
        std::fs::write(content.join("local.glb"), b"inside").unwrap();
        let foreign = outside.0.join("foreign.json");
        write_json(
            &foreign,
            &new_sidecar("abcdef0123456789abcdef0123456789", Kind::Model),
        )
        .unwrap();
        symlink(&foreign, content.join("local.glb.meta.json")).unwrap();
        assert!(
            project
                .scan(Some(&mut ids))
                .unwrap_err()
                .contains("outside")
        );
    }

    #[test]
    fn documents_too_large_to_read_back_are_not_written() {
        use ganbate_assets::glb::{NodeWorld, SourceNode};
        use ganbate_core::world::Transform;
        let t = TempDir::new("large");
        let mut ids = Ids::seeded(8);
        let mut p = Project::create(&t.0, "Large", &mut ids).unwrap();
        // A model of 90,000 placed parts with long names: its scene is larger than the reader
        // takes (`ganbate_core::json::MAX_BYTES`, about 400 bytes a part).
        let part = |i: usize| SourceNode {
            name: format!("Part {i} {}", "with a long descriptive name ".repeat(10)),
            parent: None,
            local: Transform::from_trs([i as f32, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0; 3])
                .unwrap(),
            meshes: 0..1,
            light: None,
        };
        let model = NodeWorld {
            nodes: (0..90_000).map(part).collect(),
            ..NodeWorld::default()
        };
        let (model_id, scene_id) = (ids.asset(), ids.asset());
        let doc = crate::cook::scene_from_model(&model, &model_id, &scene_id, "Large", &mut ids);
        let err = p
            .write_document("content/large.scene.json", Kind::Scene, &doc)
            .unwrap_err();
        assert!(err.contains("content/large.scene.json"), "{err}");
        assert!(!t.0.join("content/large.scene.json").exists());
        assert!(p.entry(&scene_id).is_none());
    }

    #[test]
    fn copied_ids_and_bad_manifests_are_refused() {
        let t = TempDir::new("dupe");
        let mut ids = Ids::seeded(4);
        let mut p = Project::create(&t.0, "Dupe", &mut ids).unwrap();
        let doc = ganbate_scene::scene::new_document(
            ganbate_scene::scene::Kind::Scene,
            &ids.asset(),
            "A",
        );
        p.write_document("content/a.scene.json", Kind::Scene, &doc)
            .unwrap();
        std::fs::copy(
            t.0.join("content/a.scene.json"),
            t.0.join("content/b.scene.json"),
        )
        .unwrap();
        let err = p.scan(None).unwrap_err();
        assert!(err.contains("same id"), "{err}");

        for bad in [
            r#"{"ganbate":"project","version":1,"id":"00","content":["content"]}"#,
            r#"{"ganbate":"project","version":1,"id":"0123456789abcdef0123456789abcdef","content":["../out"]}"#,
            r#"{"ganbate":"project","version":1,"id":"0123456789abcdef0123456789abcdef","content":[]}"#,
            r#"{"ganbate":"project","version":1,"id":"0123456789abcdef0123456789abcdef","content":["c"],"typo":1}"#,
        ] {
            assert!(
                Manifest::parse(&json::parse(bad.as_bytes()).unwrap()).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn data_documents_are_self_identified_and_scanned_without_sidecars() {
        assert_eq!(Kind::Data.name(), "data");
        assert!(Kind::Data.self_identified());
        assert_eq!(Kind::of_file("hud.data.json"), Some(Kind::Data));
        assert_eq!(Kind::of_file("HUD.DATA.JSON"), Some(Kind::Data));
        assert_eq!(Kind::of_file("hud.data.json.meta.json"), None);

        let t = TempDir::new("data-doc");
        let mut ids = Ids::seeded(17);
        let mut p = Project::create(&t.0, "Observatory", &mut ids).unwrap();
        let id = ids.asset();
        let doc = json::parse(
            format!(
                r#"{{"ganbate":"data","version":1,"id":"{id}","schema":"hud.anchors","name":"Observatory HUD","values":{{}}}}"#
            )
            .as_bytes(),
        )
        .unwrap();
        p.write_document("content/data/hud.data.json", Kind::Data, &doc)
            .unwrap();
        assert!(t.0.join("content/data/hud.data.json").is_file());
        assert!(!t.0.join("content/data/hud.data.json.meta.json").exists());

        let mut reopened = Project::open(&t.0).unwrap();
        let report = reopened.scan(Some(&mut ids)).unwrap();
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        assert!(report.created.is_empty(), "{:?}", report.created);
        let entry = reopened.entry(&id).expect("scanned data document");
        assert_eq!(entry.kind, Kind::Data);
        assert_eq!(entry.path, "content/data/hud.data.json");
        assert_eq!(reopened.read_document(&id).unwrap(), doc);
    }
}
