//! Asset catalog metadata, reference ("where used") queries, non-destructive move/rename,
//! sidecar `.meta.json` authoring (`meta-get` / `meta-set`), external art intake bridge
//! (`intake-list` / `intake-preview` / `intakeIndex` ingest), and direct file import.
use super::*;
use ganbate_pipeline::ddc::AssetMeta;
use ganbate_pipeline::project::{MANIFEST, new_sidecar, sidecar_path};
use ganbate_scene::schema::{Field, FieldType};

/// Maximum raw file size accepted by base64 `asset-import` / `intake-preview` (16 MiB).
pub const MAX_IMPORT_BYTES: usize = 16 * 1024 * 1024;
const MAX_INTAKE_DEPTH: usize = 8;
const MAX_INTAKE_FILES: usize = 2_000;

struct AssetRef {
    target: String,
    doc: Option<String>,
    doc_path: String,
    doc_kind: &'static str,
    entity: Option<String>,
    entity_name: Option<String>,
    field: String,
}

impl Studio {
    pub(super) fn asset_bytes(&self, rel: &str) -> u64 {
        if self.inside(rel).is_err() {
            return 0;
        }
        std::fs::metadata(self.project.root.join(rel)).map_or(0, |m| m.len())
    }

    pub(super) fn plain_asset_path(&self, path: &str) -> Result<(String, Kind), String> {
        if path
            .split('/')
            .any(|p| p.is_empty() || p.starts_with('.') || p.contains(['\\', ':', '\0']))
        {
            return Err(format!(
                "{path:?} is not a plain path (names separated by /, none starting with a dot)"
            ));
        }
        let Some(folder) = self
            .project
            .manifest
            .content
            .iter()
            .find(|c| path.starts_with(&format!("{c}/")))
        else {
            return Err("put assets in a content folder".into());
        };
        let depth = path[folder.len() + 1..].matches('/').count();
        if depth > MAX_FOLDER_DEPTH {
            return Err(format!(
                "{path} is {depth} folders deep in {folder}; a project keeps to {MAX_FOLDER_DEPTH}"
            ));
        }
        let kind = self.project.check_asset_path(path)?;
        self.inside(path)?;
        if !kind.self_identified() {
            self.inside(&format!("{path}.meta.json"))?;
        }
        Ok((path.to_string(), kind))
    }

    pub(super) fn asset_usage(&self, from: ClientId, msg: &Value) -> Result<Outbox, String> {
        if let Some(op) = msg.get("op").and_then(|v| v.string().ok()) {
            return match op {
                "meta-get" => self.asset_meta_get(from, msg),
                "meta-set" => self.asset_meta_set(from, msg),
                "intake-list" => self.asset_intake_list(from, msg),
                "intake-preview" => self.asset_intake_preview(from, msg),
                "semantic-catalog" => self.asset_semantic_catalog(from, msg),
                other => Err(format!("unknown asset-usage op {other:?}")),
            };
        }
        let req = msg.get("req").cloned().unwrap_or(Value::Null);
        match msg.get("id") {
            None | Some(Value::Null) => {
                let mut counts: BTreeMap<String, u64> = BTreeMap::new();
                self.visit_project_references(|r| {
                    *counts.entry(r.target).or_insert(0) += 1;
                });
                let counts_obj =
                    Value::Object(counts.into_iter().map(|(k, v)| (k, n(v))).collect());
                Ok(vec![(
                    from,
                    obj(vec![
                        ("type", s("asset-usage")),
                        ("req", req),
                        ("counts", counts_obj),
                    ]),
                )])
            }
            Some(v) => {
                let target = v
                    .string()
                    .ok()
                    .filter(|id| is_asset_id(id))
                    .ok_or("missing or invalid asset id")?;
                let mut uses = Vec::new();
                self.visit_project_references(|r| {
                    if r.target == target {
                        uses.push(obj(vec![
                            ("doc", r.doc.as_deref().map_or(Value::Null, s)),
                            ("docPath", s(&r.doc_path)),
                            ("docKind", s(r.doc_kind)),
                            ("entity", r.entity.as_deref().map_or(Value::Null, s)),
                            (
                                "entityName",
                                r.entity_name.as_deref().map_or(Value::Null, s),
                            ),
                            ("field", s(&r.field)),
                        ]));
                    }
                });
                Ok(vec![(
                    from,
                    obj(vec![
                        ("type", s("asset-usage")),
                        ("id", s(target)),
                        ("req", req),
                        ("uses", Value::Array(uses)),
                    ]),
                )])
            }
        }
    }

    /// Reads the non-destructive `.meta.json` authoring settings (`import`, `lods`, `collider`,
    /// `sockets`) of an asset.
    pub fn asset_meta_get(&self, from: ClientId, msg: &Value) -> Result<Outbox, String> {
        let req = msg.get("req").cloned().unwrap_or(Value::Null);
        let id = msg
            .get("id")
            .and_then(|v| v.string().ok())
            .filter(|id| is_asset_id(id))
            .ok_or("missing or invalid asset id")?;
        let entry = self
            .project
            .entry(id)
            .ok_or_else(|| format!("no asset {id}"))?;
        if entry.kind.self_identified() {
            return Ok(vec![(
                from,
                obj(vec![
                    ("type", s("asset-usage")),
                    ("op", s("meta-get")),
                    ("id", s(id)),
                    ("path", s(&entry.path)),
                    ("kind", s(entry.kind.name())),
                    ("selfIdentified", Value::Bool(true)),
                    ("meta", Value::Object(BTreeMap::new())),
                    ("req", req),
                ]),
            )]);
        }
        self.inside(&entry.path)?;
        self.inside(&format!("{}.meta.json", entry.path))?;
        let sidecar = self.project.read_sidecar(id)?;
        let meta = AssetMeta::from_sidecar(&sidecar)?;
        Ok(vec![(
            from,
            obj(vec![
                ("type", s("asset-usage")),
                ("op", s("meta-get")),
                ("id", s(id)),
                ("path", s(&entry.path)),
                ("kind", s(entry.kind.name())),
                ("selfIdentified", Value::Bool(false)),
                ("meta", meta.to_value()),
                ("sidecar", sidecar),
                ("req", req),
            ]),
        )])
    }

    /// Validates and writes updated `.meta.json` authoring settings (`import`, `lods`, `collider`,
    /// `sockets`) for an asset without mutating its source file or 32-hex identity.
    pub fn asset_meta_set(&self, from: ClientId, msg: &Value) -> Result<Outbox, String> {
        let req = msg.get("req").cloned().unwrap_or(Value::Null);
        let id = msg
            .get("id")
            .and_then(|v| v.string().ok())
            .filter(|id| is_asset_id(id))
            .ok_or("missing or invalid asset id")?;
        let entry = self
            .project
            .entry(id)
            .ok_or_else(|| format!("no asset {id}"))?;
        if entry.kind.self_identified() {
            return Err(format!("{} has no .meta.json sidecar", entry.path));
        }
        self.inside(&entry.path)?;
        self.inside(&format!("{}.meta.json", entry.path))?;
        let meta_val = msg.get("meta").ok_or("meta-set needs a meta object")?;
        let meta = AssetMeta::from_sidecar(meta_val)?;
        let sidecar = self.project.write_sidecar_meta(id, &meta)?;
        Ok(vec![(
            from,
            obj(vec![
                ("type", s("asset-usage")),
                ("op", s("meta-set")),
                ("id", s(id)),
                ("path", s(&entry.path)),
                ("kind", s(entry.kind.name())),
                ("meta", meta.to_value()),
                ("sidecar", sidecar),
                ("req", req),
            ]),
        )])
    }

    /// Scans read-only external art directories configured in `manifest.intake` (`ganbate-art`,
    /// `tripo3d-queue`, etc.) for supported raw assets.
    pub fn asset_intake_list(&self, from: ClientId, msg: &Value) -> Result<Outbox, String> {
        let req = msg.get("req").cloned().unwrap_or(Value::Null);
        let mut roots = Vec::with_capacity(self.project.manifest.intake.len());
        let mut items = Vec::new();
        for (idx, configured) in self.project.manifest.intake.iter().enumerate() {
            let raw_dir = resolve_intake_dir(&self.project.root, configured);
            let label = intake_label(configured);
            let Ok(real_intake) = std::fs::canonicalize(&raw_dir) else {
                roots.push(obj(vec![
                    ("index", n(idx as u64)),
                    ("path", s(configured)),
                    ("label", s(&label)),
                    ("available", Value::Bool(false)),
                    ("count", n(0)),
                ]));
                continue;
            };
            if !real_intake.is_dir() {
                roots.push(obj(vec![
                    ("index", n(idx as u64)),
                    ("path", s(configured)),
                    ("label", s(&label)),
                    ("available", Value::Bool(false)),
                    ("count", n(0)),
                ]));
                continue;
            }
            let mut found = Vec::new();
            walk_intake_dir(
                &real_intake,
                &real_intake,
                0,
                &mut found,
                MAX_INTAKE_FILES.saturating_sub(items.len()),
            )?;
            found.sort_by(|a, b| a.0.cmp(&b.0));
            let count = found.len() as u64;
            for (rel_path, kind, bytes) in found {
                let file_name = rel_path
                    .rsplit('/')
                    .next()
                    .unwrap_or(rel_path.as_str())
                    .to_string();
                items.push(obj(vec![
                    ("intakeIndex", n(idx as u64)),
                    ("rootLabel", s(&label)),
                    ("intakePath", s(&rel_path)),
                    ("name", s(&file_name)),
                    ("kind", s(kind.name())),
                    ("bytes", n(bytes)),
                ]));
            }
            roots.push(obj(vec![
                ("index", n(idx as u64)),
                ("path", s(configured)),
                ("label", s(&label)),
                ("available", Value::Bool(true)),
                ("count", n(count)),
            ]));
        }
        Ok(vec![(
            from,
            obj(vec![
                ("type", s("asset-usage")),
                ("op", s("intake-list")),
                ("roots", Value::Array(roots)),
                ("items", Value::Array(items)),
                ("req", req),
            ]),
        )])
    }

    /// Reads bounded preview bytes from a configured `manifest.intake` directory so Studio can
    /// render interactive 3D / texture / waveform previews before ingesting.
    pub fn asset_intake_preview(&self, from: ClientId, msg: &Value) -> Result<Outbox, String> {
        let req = msg.get("req").cloned().unwrap_or(Value::Null);
        let (idx, rel, file, kind) = self.resolve_intake_candidate(msg)?;
        let bytes = ganbate_assets::read_asset(&file)?;
        if bytes.len() > MAX_IMPORT_BYTES {
            return Err(
                "intake preview exceeds 16 MiB preview limit; import directly to disk".into(),
            );
        }
        Ok(vec![(
            from,
            obj(vec![
                ("type", s("asset-usage")),
                ("op", s("intake-preview")),
                ("intakeIndex", n(idx as u64)),
                ("intakePath", s(&rel)),
                ("kind", s(kind.name())),
                ("bytes", n(bytes.len() as u64)),
                ("dataBase64", s(&ganbate_core::digest::base64(&bytes))),
                ("req", req),
            ]),
        )])
    }

    /// Synthesizes a `ganbate.semantic.catalog.v2` affordance catalog across all placeable
    /// project assets (`model`, `prefab`, `character`), merging any `.meta.json` sidecar
    /// collider/socket settings for the Tri-Modal Semantic World Graph solver.
    pub fn asset_semantic_catalog(&self, from: ClientId, msg: &Value) -> Result<Outbox, String> {
        let req = msg.get("req").cloned().unwrap_or(Value::Null);
        let mut entries = Vec::new();
        for entry in self.project.entries() {
            if !matches!(entry.kind.name(), "model" | "prefab" | "character") {
                continue;
            }
            let file_name = entry.path.rsplit('/').next().unwrap_or(&entry.path);
            let stem = file_name
                .strip_suffix(".prefab.json")
                .or_else(|| file_name.strip_suffix(".character.json"))
                .or_else(|| file_name.strip_suffix(".glb"))
                .unwrap_or(file_name);
            let hay = format!(
                "{} {} {}",
                entry.path.to_lowercase(),
                stem.to_lowercase(),
                entry.kind.name()
            );

            let role = if entry.kind.name() == "character"
                || hay.contains("character")
                || hay.contains("avatar")
                || hay.contains("dealer")
                || hay.contains("rei")
                || hay.contains("rigged")
            {
                "articulated_agent"
            } else if hay.contains("table")
                || hay.contains("desk")
                || hay.contains("counter")
                || hay.contains("altar")
                || hay.contains("pedestal")
            {
                "support_surface"
            } else if hay.contains("chair")
                || hay.contains("stool")
                || hay.contains("seat")
                || hay.contains("bench")
                || hay.contains("cushion")
                || hay.contains("zabuton")
            {
                "seating"
            } else if hay.contains("lantern")
                || hay.contains("lamp")
                || hay.contains("beacon")
                || hay.contains("light")
                || hay.contains("sconce")
            {
                "luminaire"
            } else if hay.contains("vessel")
                || hay.contains("cup")
                || hay.contains("teapot")
                || hay.contains("bottle")
                || hay.contains("chip")
                || hay.contains("tray")
                || hay.contains("bowl")
                || hay.contains("vase")
            {
                "surface_prop"
            } else {
                "architectural_prop"
            };

            let (mut hx, mut hy, mut hz, mut surface_y, shape) = match role {
                "support_surface" => (1.3_f64, 0.45_f64, 0.85_f64, 0.9_f64, "aabb"),
                "surface_prop" => (0.16, 0.18, 0.16, 0.36, "cylinder"),
                "seating" => (0.32, 0.45, 0.32, 0.48, "cylinder"),
                "luminaire" => (0.28, 0.55, 0.28, 1.1, "cylinder"),
                "articulated_agent" => (0.35, 0.88, 0.35, 1.76, "cylinder"),
                _ => (0.45, 0.9, 0.45, 1.8, "aabb"),
            };

            if !entry.kind.self_identified()
                && let Ok(sidecar) = self.project.read_sidecar(&entry.id)
                && let Ok(meta) = AssetMeta::from_sidecar(&sidecar)
                && let Some(ref col) = meta.collider
            {
                let ext = col.half_extents;
                if ext.iter().any(|&v| v > 0.0) {
                    hx = (ext[0] as f64).max(0.04);
                    hy = (ext[1] as f64).max(0.04);
                    hz = (ext[2] as f64).max(0.04);
                    surface_y = (hy * 2.0).max(0.08);
                } else if col.radius > 0.0 && col.height > 0.0 {
                    hx = (col.radius as f64).max(0.04);
                    hz = (col.radius as f64).max(0.04);
                    hy = ((col.height as f64) * 0.5).max(0.04);
                    surface_y = (col.height as f64).max(0.08);
                }
            }

            let domain = if role == "surface_prop" {
                "table_surface"
            } else {
                "floor"
            };
            let default_rel = if domain == "table_surface" {
                "on_top_of"
            } else if role == "seating" || role == "articulated_agent" {
                "in_front_of"
            } else {
                "beside"
            };
            let academic_collection = match role {
                "articulated_agent" => "characters",
                "support_surface" | "surface_prop" | "seating" => "tabletop",
                "luminaire" => "lighting",
                _ => "architecture",
            };
            let radius = hx.max(hz);

            entries.push(obj(vec![
                ("schema", s("ganbate.semantic.node.v2")),
                ("id", s(stem)),
                ("type", s(stem)),
                ("role", s(role)),
                ("academicCollection", s(academic_collection)),
                ("path", s(&entry.path)),
                ("assetId", s(&entry.id)),
                (
                    "bounds",
                    obj(vec![
                        ("shape", s(shape)),
                        (
                            "halfExtents",
                            Value::Array(vec![
                                Value::Number(hx),
                                Value::Number(hy),
                                Value::Number(hz),
                            ]),
                        ),
                        ("radius", Value::Number(radius)),
                        ("height", Value::Number(hy * 2.0)),
                        ("surfaceY", Value::Number(surface_y)),
                    ]),
                ),
                (
                    "affordances",
                    obj(vec![
                        ("domain", s(domain)),
                        ("defaultRelation", s(default_rel)),
                        ("supportsOnTopOf", Value::Bool(role == "support_surface")),
                        ("emitsLight", Value::Bool(role == "luminaire")),
                        (
                            "orientToTarget",
                            Value::Bool(role == "seating" || role == "articulated_agent"),
                        ),
                    ]),
                ),
            ]));
        }

        Ok(vec![(
            from,
            obj(vec![
                ("type", s("asset-usage")),
                ("op", s("semantic-catalog")),
                ("schema", s("ganbate.semantic.catalog.v2")),
                ("entries", Value::Array(entries)),
                ("req", req),
            ]),
        )])
    }

    fn resolve_intake_candidate(
        &self,
        msg: &Value,
    ) -> Result<(usize, String, PathBuf, Kind), String> {
        let idx = msg
            .get("intakeIndex")
            .and_then(|v| v.unsigned().ok())
            .ok_or("missing intakeIndex")? as usize;
        let configured = self
            .project
            .manifest
            .intake
            .get(idx)
            .ok_or_else(|| format!("intakeIndex {idx} is out of range"))?;
        let rel = msg
            .get("intakePath")
            .and_then(|v| v.string().ok())
            .ok_or("missing intakePath")?;
        validate_plain_rel_path(rel)?;
        let file_name = rel.rsplit('/').next().unwrap_or(rel);
        let kind = Kind::of_file(file_name)
            .ok_or_else(|| format!("{rel}: unsupported asset file extension"))?;
        let raw_dir = resolve_intake_dir(&self.project.root, configured);
        let real_intake =
            std::fs::canonicalize(&raw_dir).map_err(|e| format!("{}: {e}", raw_dir.display()))?;
        let candidate = real_intake.join(rel);
        let actual = std::fs::canonicalize(&candidate).map_err(|e| format!("{rel}: {e}"))?;
        if !actual.starts_with(&real_intake) {
            return Err(format!("{rel} leads outside intake folder"));
        }
        if !actual.is_file() {
            return Err(format!("{rel} is not a regular file"));
        }
        Ok((idx, rel.to_string(), actual, kind))
    }

    fn visit_project_references(&self, mut emit: impl FnMut(AssetRef)) {
        if let Some(startup) = &self.project.manifest.startup {
            emit(AssetRef {
                target: startup.clone(),
                doc: None,
                doc_path: MANIFEST.into(),
                doc_kind: "project",
                entity: None,
                entity_name: None,
                field: "startup".into(),
            });
        }
        for entry in self.project.entries() {
            match entry.kind {
                Kind::Scene | Kind::Prefab => {
                    let root = if let Some(open) = self.open.get(&entry.id)
                        && open.unavailable.is_none()
                    {
                        open.doc.root().clone()
                    } else if self.inside(&entry.path).is_ok()
                        && let Ok(disk) = self.project.read_document(&entry.id)
                    {
                        disk
                    } else {
                        continue;
                    };
                    visit_scene_references(
                        &self.registry,
                        &entry.id,
                        &entry.path,
                        entry.kind.name(),
                        &root,
                        &mut emit,
                    );
                }
                Kind::Material => {
                    if self.inside(&entry.path).is_ok()
                        && let Ok(disk) = self.project.read_document(&entry.id)
                    {
                        visit_material_references(&entry.id, &entry.path, &disk, "", &mut emit);
                    }
                }
                Kind::Data => {
                    let root = if let Some(open) = self.open.get(&entry.id)
                        && open.unavailable.is_none()
                    {
                        open.doc.root().clone()
                    } else if self.inside(&entry.path).is_ok()
                        && let Ok(disk) = self.project.read_document(&entry.id)
                    {
                        disk
                    } else {
                        continue;
                    };
                    visit_data_references(&self.registry, &entry.id, &entry.path, &root, &mut emit);
                }
                _ => {}
            }
        }
    }

    pub(super) fn asset_move(&mut self, from: ClientId, msg: &Value) -> Result<Outbox, String> {
        let req = msg.get("req").cloned().unwrap_or(Value::Null);
        let id = msg
            .get("id")
            .and_then(|v| v.string().ok())
            .filter(|id| is_asset_id(id))
            .ok_or("missing or invalid asset id")?
            .to_string();
        let entry = self
            .project
            .entry(&id)
            .ok_or_else(|| format!("no asset {id}"))?
            .clone();
        let to_raw = msg
            .get("to")
            .and_then(|v| v.string().ok())
            .ok_or("give a target path")?;
        let (to, to_kind) = self.plain_asset_path(to_raw)?;
        if to_kind != entry.kind {
            return Err(format!(
                "cannot move {} ({}) to {} ({})",
                entry.path,
                entry.kind.name(),
                to,
                to_kind.name()
            ));
        }
        if to == entry.path {
            return Ok(vec![(
                from,
                obj(vec![
                    ("type", s("asset-moved")),
                    ("id", s(&id)),
                    ("kind", s(entry.kind.name())),
                    ("from", s(&entry.path)),
                    ("to", s(&to)),
                    ("req", req),
                ]),
            )]);
        }

        self.inside(&entry.path)?;
        if !entry.kind.self_identified() {
            self.inside(&format!("{}.meta.json", entry.path))?;
        }
        let from_file = self.project.root.join(&entry.path);
        if !std::fs::symlink_metadata(&from_file).is_ok_and(|m| m.is_file()) {
            return Err(format!("{} is missing on disk", entry.path));
        }
        let from_meta = sidecar_path(&from_file);
        if !entry.kind.self_identified()
            && !std::fs::symlink_metadata(&from_meta).is_ok_and(|m| m.is_file())
        {
            return Err(format!("{}.meta.json is missing on disk", entry.path));
        }

        let to_file = self.project.root.join(&to);
        let to_meta = sidecar_path(&to_file);
        if std::fs::symlink_metadata(&to_file).is_ok() || self.project.find_path(&to).is_some() {
            return Err(format!("{to} already exists"));
        }
        if !entry.kind.self_identified() && std::fs::symlink_metadata(&to_meta).is_ok() {
            return Err(format!("{to}.meta.json already exists"));
        }

        if let Some(parent) = to_file.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        self.inside(&to)?;
        if !entry.kind.self_identified() {
            self.inside(&format!("{to}.meta.json"))?;
        }

        std::fs::rename(&from_file, &to_file)
            .map_err(|e| format!("{} -> {to}: {e}", entry.path))?;
        if !entry.kind.self_identified()
            && let Err(e) = std::fs::rename(&from_meta, &to_meta)
        {
            let _ = std::fs::rename(&to_file, &from_file);
            return Err(format!("{}.meta.json -> {to}.meta.json: {e}", entry.path));
        }

        self.project.relocate_entry(&id, &to)?;
        if let Some(open) = self.open.get_mut(&id) {
            open.path = to.clone();
        }

        let assets = obj(vec![("type", s("assets")), ("assets", self.assets())]);
        let mut out: Outbox = self.clients.keys().map(|c| (*c, assets.clone())).collect();
        out.push((
            from,
            obj(vec![
                ("type", s("asset-moved")),
                ("id", s(&id)),
                ("kind", s(entry.kind.name())),
                ("from", s(&entry.path)),
                ("to", s(&to)),
                ("req", req),
            ]),
        ));
        if let Some(open) = self.open.get(&id)
            && !open.watchers.is_empty()
        {
            let doc_msg = self.doc_message(&id);
            out.extend(open.watchers.iter().map(|w| (*w, doc_msg.clone())));
        }
        Ok(out)
    }

    pub(super) fn asset_import(&mut self, from: ClientId, msg: &Value) -> Result<Outbox, String> {
        let req = msg.get("req").cloned().unwrap_or(Value::Null);
        let path_raw = msg
            .get("path")
            .and_then(|v| v.string().ok())
            .ok_or("give the imported asset a path")?;
        let (path, kind) = self.plain_asset_path(path_raw)?;
        let custom_meta = match msg.get("meta") {
            None | Some(Value::Null) => None,
            Some(v) => Some(AssetMeta::from_sidecar(v)?),
        };
        let bytes = if msg.get("intakeIndex").is_some() {
            let (_idx, intake_rel, intake_file, intake_kind) =
                self.resolve_intake_candidate(msg)?;
            if intake_kind != kind {
                return Err(format!(
                    "cannot ingest {} ({}) into {} ({})",
                    intake_rel,
                    intake_kind.name(),
                    path,
                    kind.name()
                ));
            }
            let b = ganbate_assets::read_asset(&intake_file)?;
            if b.is_empty() {
                return Err("intake file is empty".into());
            }
            b
        } else {
            let b64 = msg
                .get("dataBase64")
                .and_then(|v| v.string().ok())
                .ok_or("asset-import needs dataBase64 or intakeIndex")?;
            if b64.len() > MAX_IMPORT_BYTES / 3 * 4 + 4096 {
                return Err("asset exceeds 16 MiB import limit".into());
            }
            let b = ganbate_core::digest::base64_decode(b64)?;
            if b.is_empty() {
                return Err("imported file is empty".into());
            }
            if b.len() > MAX_IMPORT_BYTES {
                return Err("asset exceeds 16 MiB import limit".into());
            }
            b
        };

        let dest = self.project.root.join(&path);
        if std::fs::symlink_metadata(&dest).is_ok() || self.project.find_path(&path).is_some() {
            return Err(format!("{path} already exists"));
        }
        if !kind.self_identified() && std::fs::symlink_metadata(sidecar_path(&dest)).is_ok() {
            return Err(format!("{path}.meta.json already exists"));
        }

        let id = match kind {
            Kind::Scene | Kind::Prefab => {
                let mut doc = json::parse(&bytes).map_err(|e| format!("{path}: {e}"))?;
                let id = self.prepare_imported_document(&mut doc, kind)?;
                let issues = scene::check(&doc, &self.registry);
                if let Some(first) = issues.first() {
                    return Err(format!("{path} · {}: {}", first.path, first.message));
                }
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("{}: {e}", parent.display()))?;
                }
                self.inside(&path)?;
                self.project.write_document(&path, kind, &doc)?;
                id
            }
            Kind::Material => {
                let mut doc = json::parse(&bytes).map_err(|e| format!("{path}: {e}"))?;
                let id = self.prepare_imported_material(&mut doc)?;
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("{}: {e}", parent.display()))?;
                }
                self.inside(&path)?;
                self.project.write_document(&path, Kind::Material, &doc)?;
                id
            }
            Kind::Data => {
                let mut doc = json::parse(&bytes).map_err(|e| format!("{path}: {e}"))?;
                let id = self.prepare_imported_document(&mut doc, Kind::Data)?;
                let issues = validate_data(&doc, &self.registry);
                if let Some((at, message)) = issues.iter().next() {
                    let prefix = if at.is_empty() {
                        path.clone()
                    } else {
                        format!("{path} · {at}")
                    };
                    return Err(format!("{prefix}: {message}"));
                }
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("{}: {e}", parent.display()))?;
                }
                self.inside(&path)?;
                self.project.write_document(&path, Kind::Data, &doc)?;
                id
            }
            Kind::Look => {
                let doc = json::parse(&bytes).map_err(|e| format!("{path}: {e}"))?;
                validate_look(&doc).map_err(|e| format!("{path}: {e}"))?;
                let canonical = json::to_canonical(&doc)?;
                self.install_sidecar_asset(
                    &path,
                    Kind::Look,
                    canonical.as_bytes(),
                    custom_meta.as_ref(),
                )?
            }
            Kind::Model => {
                ganbate_assets::glb::import_nodes(&bytes, true)
                    .map_err(|e| format!("{path}: {e}"))?;
                self.install_sidecar_asset(&path, Kind::Model, &bytes, custom_meta.as_ref())?
            }
            Kind::Texture => {
                validate_texture_header(&path, &bytes)?;
                self.install_sidecar_asset(&path, Kind::Texture, &bytes, custom_meta.as_ref())?
            }
            Kind::Sound => {
                validate_sound_header(&path, &bytes)?;
                self.install_sidecar_asset(&path, Kind::Sound, &bytes, custom_meta.as_ref())?
            }
        };

        let assets = obj(vec![("type", s("assets")), ("assets", self.assets())]);
        let mut out: Outbox = self.clients.keys().map(|c| (*c, assets.clone())).collect();
        out.push((
            from,
            obj(vec![
                ("type", s("asset-imported")),
                ("id", s(&id)),
                ("kind", s(kind.name())),
                ("path", s(&path)),
                ("req", req),
            ]),
        ));
        Ok(out)
    }

    pub(super) fn fresh_asset_id(&mut self) -> String {
        loop {
            let candidate = self.ids.asset();
            if self.project.entry(&candidate).is_none() {
                return candidate;
            }
        }
    }

    fn prepare_imported_document(&mut self, doc: &mut Value, kind: Kind) -> Result<String, String> {
        let existing_id = doc
            .get("id")
            .and_then(|v| v.string().ok())
            .filter(|id| is_asset_id(id) && self.project.entry(id).is_none())
            .map(String::from);
        let id = existing_id.unwrap_or_else(|| self.fresh_asset_id());
        let Value::Object(map) = doc else {
            return Err(format!("a {} document is a JSON object", kind.name()));
        };
        map.insert("id".into(), Value::String(id.clone()));
        Ok(id)
    }

    fn prepare_imported_material(&mut self, doc: &mut Value) -> Result<String, String> {
        let existing_id = doc
            .get("id")
            .and_then(|v| v.string().ok())
            .filter(|id| is_asset_id(id) && self.project.entry(id).is_none())
            .map(String::from);
        let id = existing_id.unwrap_or_else(|| self.fresh_asset_id());
        let Value::Object(map) = doc else {
            return Err("a material document is a JSON object".into());
        };
        map.insert("id".into(), Value::String(id.clone()));
        Ok(id)
    }

    fn install_sidecar_asset(
        &mut self,
        rel: &str,
        kind: Kind,
        bytes: &[u8],
        custom_meta: Option<&AssetMeta>,
    ) -> Result<String, String> {
        let dest = self.project.root.join(rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        self.inside(rel)?;
        self.inside(&format!("{rel}.meta.json"))?;
        let id = self.fresh_asset_id();
        let meta_path = sidecar_path(&dest);
        let base_sidecar = new_sidecar(&id, kind);
        let sidecar = match custom_meta {
            Some(m) => m.apply_to_sidecar(&base_sidecar)?,
            None => base_sidecar,
        };
        let meta_text = json::to_canonical(&sidecar)?;
        ganbate_assets::create_asset(&meta_path, meta_text.as_bytes())
            .map_err(|e| format!("{rel}.meta.json: {e}"))?;
        if let Err(e) = ganbate_assets::create_asset(&dest, bytes) {
            let _ = std::fs::remove_file(&meta_path);
            return Err(format!("{rel}: {e}"));
        }
        if let Err(e) = self
            .project
            .register_sidecar_asset(id.clone(), kind, rel.to_string())
        {
            let _ = std::fs::remove_file(&dest);
            let _ = std::fs::remove_file(&meta_path);
            return Err(e);
        }
        Ok(id)
    }
}

fn resolve_intake_dir(project_root: &Path, configured: &str) -> PathBuf {
    let p = Path::new(configured);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        project_root.join(p)
    }
}

fn intake_label(configured: &str) -> String {
    let trimmed = configured.trim_end_matches(['/', '\\']);
    Path::new(trimmed)
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(configured)
        .to_string()
}

fn validate_plain_rel_path(rel: &str) -> Result<(), String> {
    if rel.is_empty()
        || rel.split('/').any(|part| {
            part.is_empty()
                || part == "."
                || part == ".."
                || part.starts_with('.')
                || part.contains(['\\', ':', '\0'])
        })
    {
        return Err(format!("{rel:?} is not a plain relative intake path"));
    }
    Ok(())
}

fn walk_intake_dir(
    real_root: &Path,
    dir: &Path,
    depth: usize,
    out: &mut Vec<(String, Kind, u64)>,
    remaining: usize,
) -> Result<(), String> {
    if depth > MAX_INTAKE_DEPTH || out.len() >= remaining {
        return Ok(());
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(());
    };
    for entry in entries.flatten() {
        if out.len() >= remaining {
            break;
        }
        let fname = entry.file_name();
        let Some(name_str) = fname.to_str() else {
            continue;
        };
        if name_str.starts_with('.') {
            continue;
        }
        let path = entry.path();
        let Ok(actual) = std::fs::canonicalize(&path) else {
            continue;
        };
        if !actual.starts_with(real_root) {
            continue;
        }
        if actual.is_dir() {
            walk_intake_dir(real_root, &actual, depth + 1, out, remaining)?;
        } else if actual.is_file()
            && let Some(kind) = Kind::of_file(name_str)
            && let Ok(rel) = actual.strip_prefix(real_root)
            && let Some(parts) = rel
                .components()
                .map(|c| c.as_os_str().to_str())
                .collect::<Option<Vec<_>>>()
        {
            let bytes = std::fs::metadata(&actual).map_or(0, |m| m.len());
            out.push((parts.join("/"), kind, bytes));
        }
    }
    Ok(())
}

fn validate_texture_header(path: &str, bytes: &[u8]) -> Result<(), String> {
    let lower = path.to_ascii_lowercase();
    let ok = if lower.ends_with(".png") {
        bytes.starts_with(b"\x89PNG\r\n\x1a\n")
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        bytes.starts_with(b"\xff\xd8\xff")
    } else if lower.ends_with(".webp") {
        bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
    } else {
        false
    };
    if !ok {
        return Err(format!("{path}: file header does not match image format"));
    }
    Ok(())
}

fn validate_sound_header(path: &str, bytes: &[u8]) -> Result<(), String> {
    let lower = path.to_ascii_lowercase();
    let ok = if lower.ends_with(".wav") {
        bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
    } else if lower.ends_with(".ogg") {
        bytes.starts_with(b"OggS")
    } else {
        false
    };
    if !ok {
        return Err(format!("{path}: file header does not match audio format"));
    }
    Ok(())
}

fn visit_field_refs(field: &Field, value: &Value, path: &str, emit: &mut dyn FnMut(&str, &str)) {
    match (&field.ty, value) {
        (FieldType::Asset { .. }, Value::String(id)) if is_asset_id(id) => {
            emit(id, path);
        }
        (FieldType::Array { items, .. }, Value::Array(values)) => {
            for (index, item) in values.iter().enumerate() {
                visit_field_refs(items, item, &format!("{path}[{index}]"), emit);
            }
        }
        (FieldType::Object(fields), Value::Object(values)) => {
            for (name, inner) in values {
                if let Some(sub) = fields.get(name) {
                    visit_field_refs(sub, inner, &format!("{path}.{name}"), emit);
                }
            }
        }
        _ => {}
    }
}

fn visit_component_refs(
    registry: &Registry,
    holder: &Value,
    prefix: &str,
    emit: &mut dyn FnMut(&str, &str),
) {
    let Some(Value::Object(components)) = holder.get("components") else {
        return;
    };
    for (comp_name, stored) in components {
        let Some(schema) = registry.get(comp_name) else {
            continue;
        };
        for (field_name, field_def) in &schema.fields {
            if let Some(value) = stored.get(field_name) {
                let field_path = if prefix.is_empty() {
                    format!("{comp_name}.{field_name}")
                } else {
                    format!("{prefix}.{comp_name}.{field_name}")
                };
                visit_field_refs(field_def, value, &field_path, emit);
            }
        }
    }
}

fn visit_scene_references(
    registry: &Registry,
    doc_id: &str,
    doc_path: &str,
    doc_kind: &'static str,
    root: &Value,
    emit: &mut impl FnMut(AssetRef),
) {
    let Some(Value::Object(entities)) = root.get("entities") else {
        return;
    };
    for (ent_id, entity) in entities {
        let ent_name = entity
            .get("name")
            .and_then(|v| v.string().ok())
            .unwrap_or("Entity")
            .to_string();
        visit_component_refs(registry, entity, "", &mut |target, field| {
            emit(AssetRef {
                target: target.to_string(),
                doc: Some(doc_id.to_string()),
                doc_path: doc_path.to_string(),
                doc_kind,
                entity: Some(ent_id.clone()),
                entity_name: Some(ent_name.clone()),
                field: field.to_string(),
            });
        });
        if let Some(link) = entity.get("prefab") {
            if let Some(Ok(prefab_id)) = link.get("asset").map(Value::string)
                && is_asset_id(prefab_id)
            {
                emit(AssetRef {
                    target: prefab_id.to_string(),
                    doc: Some(doc_id.to_string()),
                    doc_path: doc_path.to_string(),
                    doc_kind,
                    entity: Some(ent_id.clone()),
                    entity_name: Some(ent_name.clone()),
                    field: "prefab".into(),
                });
            }
            if let Some(Value::Object(overrides)) = link.get("overrides") {
                for (inner, data) in overrides {
                    let prefix = format!("overrides.{inner}");
                    visit_component_refs(registry, data, &prefix, &mut |target, field| {
                        emit(AssetRef {
                            target: target.to_string(),
                            doc: Some(doc_id.to_string()),
                            doc_path: doc_path.to_string(),
                            doc_kind,
                            entity: Some(ent_id.clone()),
                            entity_name: Some(ent_name.clone()),
                            field: field.to_string(),
                        });
                    });
                }
            }
        }
    }
}

fn visit_material_references(
    doc_id: &str,
    doc_path: &str,
    value: &Value,
    prefix: &str,
    emit: &mut impl FnMut(AssetRef),
) {
    match value {
        Value::String(id) if !prefix.is_empty() && prefix != "id" && is_asset_id(id) => {
            emit(AssetRef {
                target: id.clone(),
                doc: Some(doc_id.to_string()),
                doc_path: doc_path.to_string(),
                doc_kind: "material",
                entity: None,
                entity_name: None,
                field: prefix.to_string(),
            });
        }
        Value::Object(map) => {
            for (k, v) in map {
                if prefix.is_empty() && k == "id" {
                    continue;
                }
                let next = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                visit_material_references(doc_id, doc_path, v, &next, emit);
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                visit_material_references(doc_id, doc_path, v, &format!("{prefix}[{i}]"), emit);
            }
        }
        _ => {}
    }
}

fn visit_data_references(
    registry: &Registry,
    doc_id: &str,
    doc_path: &str,
    root: &Value,
    emit: &mut impl FnMut(AssetRef),
) {
    let Some(schema_name) = root.get("schema").and_then(|v| v.string().ok()) else {
        return;
    };
    let Some(schema) = registry.get(schema_name) else {
        return;
    };
    let Some(values) = root.get("values") else {
        return;
    };
    let doc_name = root
        .get("name")
        .and_then(|v| v.string().ok())
        .map(String::from);
    for (field_name, field_def) in &schema.fields {
        if let Some(value) = values.get(field_name) {
            visit_field_refs(field_def, value, field_name, &mut |target, field| {
                emit(AssetRef {
                    target: target.to_string(),
                    doc: Some(doc_id.to_string()),
                    doc_path: doc_path.to_string(),
                    doc_kind: "data",
                    entity: None,
                    entity_name: doc_name.clone(),
                    field: field.to_string(),
                });
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ganbate_pipeline::project::write_json;

    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn parse_msg(text: &str) -> Value {
        json::parse(text.as_bytes()).unwrap()
    }

    fn find_type<'a>(out: &'a Outbox, ty: &str) -> Vec<&'a (ClientId, Value)> {
        out.iter()
            .filter(|(_, m)| m.get("type").and_then(|v| v.string().ok()) == Some(ty))
            .collect()
    }

    fn sample_offset_glb() -> Vec<u8> {
        let doc = r#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":44}],"bufferViews":[{"buffer":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":6}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"nodes":[{"name":"Prop","mesh":0,"translation":[4,3,-2]}],"scenes":[{"nodes":[0]}],"scene":0}"#;
        let mut j = doc.as_bytes().to_vec();
        while j.len() % 4 != 0 {
            j.push(b' ');
        }
        let mut b = Vec::new();
        for v in [[0f32, 0., 0.], [2., 0., 0.], [0., 4., 0.]] {
            for f in v {
                b.extend_from_slice(&f.to_le_bytes());
            }
        }
        for idx in [0u16, 1, 2] {
            b.extend_from_slice(&idx.to_le_bytes());
        }
        while b.len() % 4 != 0 {
            b.push(0);
        }
        let mut out = b"glTF".to_vec();
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(28 + j.len() as u32 + b.len() as u32).to_le_bytes());
        for (id, chunk) in [(0x4e4f534au32, j), (0x004e4942, b)] {
            out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&chunk);
        }
        out
    }

    #[test]
    fn meta_get_and_meta_set_round_trip_without_mutating_glb_or_asset_id_and_recook_scene() {
        let dir =
            std::env::temp_dir().join(format!("ganbate-studio-meta-{}", Ids::from_os().entity()));
        let _tmp = TempDir(dir.clone());
        std::fs::create_dir_all(&dir).unwrap();
        let mut ids = Ids::seeded(42);
        Project::create(&dir, "Meta Authoring Test", &mut ids).unwrap();
        let (mut st, _) = Studio::open(&dir, ids).unwrap();
        let client = st.connect("model-studio");

        let glb_bytes = sample_offset_glb();
        let glb_b64 = ganbate_core::digest::base64(&glb_bytes);
        let imp = st.handle(
            client,
            &parse_msg(&format!(
                r#"{{"type":"asset-import","path":"content/models/pillar.glb","dataBase64":"{glb_b64}","req":"r-imp"}}"#
            )),
        );
        let model_id = find_type(&imp, "asset-imported")[0]
            .1
            .get("id")
            .unwrap()
            .string()
            .unwrap()
            .to_string();

        let meta_initial = st.handle(
            client,
            &parse_msg(&format!(
                r#"{{"type":"asset-usage","op":"meta-get","id":"{model_id}","req":"r-get-1"}}"#
            )),
        );
        let m0 = &find_type(&meta_initial, "asset-usage")[0].1;
        assert_eq!(m0.get("op"), Some(&s("meta-get")));
        assert_eq!(m0.get("req"), Some(&s("r-get-1")));
        assert!(m0.get("meta").unwrap().get("import").is_some());

        // Update sidecar meta with bottom_center pivot, target_height 2.0, custom LODs, collider, and named sockets.
        let set_out = st.handle(
            client,
            &parse_msg(&format!(
                r#"{{"type":"asset-usage","op":"meta-set","id":"{model_id}","req":"r-set","meta":{{"import":{{"pivot":"bottom_center","scale":1.0,"target_height":2.0,"offset":[0,0,0],"active_lod":0}},"lods":[1.0,0.6,0.3,0.15],"collider":{{"shape":"box","center":[0,1,0],"half_extents":[0.5,1.0,0.25]}},"sockets":{{"top_surface":{{"translation":[0,2,0],"rotation":[0,0,0,1]}}}}}}}}"#
            )),
        );
        let set_reply = &find_type(&set_out, "asset-usage")[0].1;
        assert_eq!(set_reply.get("op"), Some(&s("meta-set")));
        assert_eq!(set_reply.get("req"), Some(&s("r-set")));

        // Source .glb bytes and 32-hex ID are untouched.
        assert_eq!(
            std::fs::read(dir.join("content/models/pillar.glb")).unwrap(),
            glb_bytes
        );
        assert_eq!(
            st.project.entry(&model_id).unwrap().path,
            "content/models/pillar.glb"
        );

        let meta_after = st.handle(
            client,
            &parse_msg(&format!(
                r#"{{"type":"asset-usage","op":"meta-get","id":"{model_id}","req":"r-get-2"}}"#
            )),
        );
        let m1 = find_type(&meta_after, "asset-usage")[0]
            .1
            .get("meta")
            .unwrap();
        assert_eq!(
            m1.get("import").unwrap().get("pivot"),
            Some(&s("bottom_center"))
        );
        assert_eq!(m1.get("lods").unwrap().array().unwrap().len(), 4);
        assert_eq!(m1.get("collider").unwrap().get("shape"), Some(&s("box")));
        assert!(
            m1.get("sockets")
                .unwrap()
                .get("top_surface")
                .unwrap()
                .get("translation")
                .is_some()
        );

        // Create and cook a scene referencing the model; verify DDC cache file is emitted and cooked model uses normalized height.
        let created = st.handle(
            client,
            &parse_msg(
                r#"{"type":"create","kind":"scene","path":"content/scenes/test.scene.json","name":"Test"}"#,
            ),
        );
        let scene_id = find_type(&created, "created")[0]
            .1
            .get("doc")
            .unwrap()
            .string()
            .unwrap()
            .to_string();
        st.handle(
            client,
            &parse_msg(&format!(r#"{{"type":"open","doc":"{scene_id}"}}"#)),
        );
        st.handle(
            client,
            &parse_msg(&format!(
                r#"{{"type":"edit","doc":"{scene_id}","rev":0,"ops":[{{"set":["entities","00000000000000a1"],"value":{{"name":"Pillar","components":{{"transform":{{}},"mesh":{{"model":"{model_id}"}}}}}}}}]}}"#
            )),
        );
        let cooked_out = st.handle(
            client,
            &parse_msg(&format!(r#"{{"type":"cook","doc":"{scene_id}"}}"#)),
        );
        let cooked_msg = &find_type(&cooked_out, "cooked")[0].1;
        let cache_file = cooked_msg.get("file").unwrap().string().unwrap();
        let cooked_bytes = std::fs::read(dir.join(cache_file)).unwrap();
        let decoded = ganbate_assets::cooked::decode(&cooked_bytes).unwrap();
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for inst in &decoded.instances {
            for p in &decoded.meshes[inst.mesh as usize].positions {
                let wp = inst.transform.point(*p);
                min_y = min_y.min(wp.1);
                max_y = max_y.max(wp.1);
            }
        }
        assert!(
            min_y.abs() < 1e-4,
            "expected bottom_center min_y=0, got {min_y}"
        );
        assert!(
            (max_y - 2.0).abs() < 1e-4,
            "expected target_height=2.0, got {max_y}"
        );
        assert!(dir.join(".ganbate/cache/ddc").is_dir());
    }

    #[test]
    fn intake_bridge_lists_previews_imports_and_blocks_traversal_and_symlinks() {
        let base =
            std::env::temp_dir().join(format!("ganbate-studio-intake-{}", Ids::from_os().entity()));
        let _tmp = TempDir(base.clone());
        let proj_dir = base.join("project");
        let intake_dir = base.join("external-art");
        let secret_dir = base.join("secret-outside");
        std::fs::create_dir_all(intake_dir.join("props")).unwrap();
        std::fs::create_dir_all(&secret_dir).unwrap();

        let glb_bytes = sample_offset_glb();
        std::fs::write(intake_dir.join("props/archway.glb"), &glb_bytes).unwrap();
        std::fs::write(secret_dir.join("escape.glb"), &glb_bytes).unwrap();
        #[cfg(unix)]
        {
            let _ = std::os::unix::fs::symlink(
                secret_dir.join("escape.glb"),
                intake_dir.join("props/linked_escape.glb"),
            );
        }

        std::fs::create_dir_all(&proj_dir).unwrap();
        let mut ids = Ids::seeded(77);
        Project::create(&proj_dir, "Intake Test", &mut ids).unwrap();
        let manifest_path = proj_dir.join(MANIFEST);
        let mut manifest_json = ganbate_pipeline::project::read_json(&manifest_path).unwrap();
        if let Value::Object(ref mut map) = manifest_json {
            map.insert(
                "intake".into(),
                Value::Array(vec![
                    s(intake_dir.to_str().unwrap()),
                    s("../missing-intake-folder"),
                ]),
            );
        }
        write_json(&manifest_path, &manifest_json).unwrap();

        let (mut st, _) = Studio::open(&proj_dir, Ids::seeded(88)).unwrap();
        let client = st.connect("artist");

        // 1. List configured intake roots (symlink escaping outside intake_dir is omitted).
        let listed = st.handle(
            client,
            &parse_msg(r#"{"type":"asset-usage","op":"intake-list","req":"il-1"}"#),
        );
        let reply = &find_type(&listed, "asset-usage")[0].1;
        let roots = reply.get("roots").unwrap().array().unwrap();
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0].get("available"), Some(&Value::Bool(true)));
        assert_eq!(roots[0].get("count"), Some(&n(1)));
        assert_eq!(roots[1].get("available"), Some(&Value::Bool(false)));
        let items = reply.get("items").unwrap().array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].get("intakePath"), Some(&s("props/archway.glb")));
        assert_eq!(items[0].get("kind"), Some(&s("model")));

        // 2. Preview valid intake item & reject path traversal / symlink escapes.
        let prev = st.handle(
            client,
            &parse_msg(
                r#"{"type":"asset-usage","op":"intake-preview","intakeIndex":0,"intakePath":"props/archway.glb","req":"ip-1"}"#,
            ),
        );
        let prev_msg = &find_type(&prev, "asset-usage")[0].1;
        let b64 = prev_msg.get("dataBase64").unwrap().string().unwrap();
        assert_eq!(ganbate_core::digest::base64_decode(b64).unwrap(), glb_bytes);

        let traversal = st.handle(
            client,
            &parse_msg(
                r#"{"type":"asset-usage","op":"intake-preview","intakeIndex":0,"intakePath":"../secret-outside/escape.glb"}"#,
            ),
        );
        assert!(!find_type(&traversal, "error").is_empty());

        #[cfg(unix)]
        {
            let sym_escape = st.handle(
                client,
                &parse_msg(
                    r#"{"type":"asset-usage","op":"intake-preview","intakeIndex":0,"intakePath":"props/linked_escape.glb"}"#,
                ),
            );
            assert!(!find_type(&sym_escape, "error").is_empty());
        }

        // 3. Ingest external art directly into content/ with initial normalized .meta.json!
        let ingested = st.handle(
            client,
            &parse_msg(
                r#"{"type":"asset-import","intakeIndex":0,"intakePath":"props/archway.glb","path":"content/models/archway.glb","meta":{"import":{"pivot":"bottom_center","scale":0.01}},"req":"ing-1"}"#,
            ),
        );
        let imp_msg = &find_type(&ingested, "asset-imported")[0].1;
        let new_id = imp_msg.get("id").unwrap().string().unwrap();
        assert!(proj_dir.join("content/models/archway.glb").is_file());
        assert!(
            proj_dir
                .join("content/models/archway.glb.meta.json")
                .is_file()
        );
        let settings = st.project.model_import_settings(new_id);
        assert_eq!(
            settings.pivot,
            ganbate_pipeline::ddc::PivotPreset::BottomCenter
        );
        assert!((settings.scale - 0.01).abs() < 1e-6);

        // 4. Query Tri-Modal semantic catalog (`op: "semantic-catalog"`).
        let sem = st.handle(
            client,
            &parse_msg(r#"{"type":"asset-usage","op":"semantic-catalog","req":"sem-1"}"#),
        );
        let sem_msg = &find_type(&sem, "asset-usage")[0].1;
        assert_eq!(sem_msg.get("op"), Some(&s("semantic-catalog")));
        assert_eq!(
            sem_msg.get("schema"),
            Some(&s("ganbate.semantic.catalog.v2"))
        );
        let sem_entries = sem_msg.get("entries").unwrap().array().unwrap();
        assert!(!sem_entries.is_empty());
        assert!(
            sem_entries
                .iter()
                .any(|e| e.get("id") == Some(&s("archway"))
                    && e.get("role") == Some(&s("architectural_prop")))
        );
    }
}
