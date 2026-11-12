# Editor: Ganbate Studio

**Decision.** The editor is **Ganbate Studio**: a local Rust host (`ganbate-studio`) that owns the project folder, and a
browser interface whose viewport is the engine itself (the same wasm runtime and renderer a web game ships). The
host is the only writer of project files and the keeper of undo history; the interface, a running game and a
phone are all clients of the same edit protocol. A native viewport client joins later for work the browser cannot
hold (very large worlds, console devkits).

People build content here instead of asking for code: placement, terrain, vegetation, lighting, Look, cameras,
materials, characters and game stages.

## 1. Why this shape

| Option | Time to first useful edit | Same pixels as shipping | Zero-dependency cost | Scales to open worlds and consoles | Text, IME, accessibility |
| --- | --- | --- | --- | --- | --- |
| **A. Local Rust host + browser UI + wasm viewport (chosen)** | weeks | yes for web; native via a viewport client | small: std-only HTTP/WebSocket host, hand-written DOM UI | host and protocol scale; browser viewport limited by wasm memory, so add a native viewport client | the browser's |
| B. Native Rust editor with its own UI toolkit | many months (fonts, shaping, IME, docking, widgets first) | yes on native | very large | best long-term | must be built per OS |
| C. Editor pages inside each game's web app | days | yes | none for the engine, but per game | no: every game rebuilds tools, nothing is shared | the browser's |
| D. A DCC (Blender, Maya) as the editor, the engine as exporter target | days | no: DCC viewport differs | none | weak: no game data, no live game | the DCC's |
| E. Hosted cloud editor | months | yes for web | needs a service | needs servers | the browser's |

A wins on what matters now: a first useful editor soon, true pixels for the first shipping target (the web),
reuse across games, and an architecture that does not need replacing later. The browser is the interface, not the
authority: the documents, history, import and cook live in Rust, so a native viewport or a second interface can
join without moving logic. Frostbite's and RAGE's tools talking to a running game, Insomniac's web-based tools
and Unreal's Live Link are the precedent for this split.

The rejected options keep a role: **C** is how a game's current tools work, and their contents move into
documents and plugins (§6). **D** stays the place to model meshes; the editor assembles, it does not model.

## 2. Processes and protocol

```text
             project folder (JSON documents, sources, sidecars)
                         |  read / write (only the host writes)
                 +-------+--------+
                 |   ganbate-studio   |  asset database, import cache, cook, documents + undo,
                 |   (Rust host)  |  validation, live-link hub, serves the interface
                 +---+-------+----+
        WebSocket    |       |    WebSocket (opt-in on the LAN, with a pairing token)
          +----------+       +----------------------+
          |                                         |
   browser interface                       live targets
   panels in the DOM                       a running game (web or native) with the live-link
   viewport = ganbate-web (wasm)               client; a phone on the same network; a native
                                           viewport client (later)
```

- **Host.** `ganbate-studio <project>` serves the interface at `http://localhost:<port>/`, reads and indexes the
  project, opens documents on request and keeps each open document's history. It is written with the standard
  library only: HTTP/1.1 and WebSocket (RFC 6455). It scans the project when it starts; files added or changed
  while it runs are seen after a restart (re-scanning by polling is planned), except that a document reopened with
  no unsaved edits is read again, and a save over a file another tool changed since asks first. It listens on
  localhost, where it answers only requests addressed to this machine at its own port (`Host` localhost, 127.0.0.1
  or [::1]) and WebSocket upgrades from the editor's own origin, which closes cross-site and DNS-rebinding attacks;
  a port forward must keep the same port number. Started with `--lan`, it also listens on the network and prints a
  128-bit pairing token that every client, this machine's browser included, presents once in the URL; a cookie
  carries it after that, and ten wrong tokens in a minute lock a peer out for the rest of the minute. Each request
  has a deadline, connections and per-client queues are bounded, and a client that stops reading is dropped.
- **Edits are ops.** A client sends
  `{"type": "edit", "doc": id, "rev": n, "label": "Move", "merge": "drag:7f...", "req": 7, "ops": [{"set": ["entities",
  "2b8f0c71d4e6a935", "components", "transform", "translation"], "value": [4.5, 0, -2]}, {"remove": [...]}]}`: set
  and remove ops (`ganbate_scene::doc`) on paths of 1 to 64 keys; in scenes and prefabs only `entities` and `name` may
  change. The host checks the revision, the schema and the scene rules (no parent cycles, valid ids), applies the
  ops all-or-nothing, refuses an edit that adds a validation problem (problems the document already had do not block
  edits), records the inverse for undo and broadcasts the ops with the new revision to every client watching that
  document. A client whose revision is stale gets the current document instead. The protocol's messages are listed
  at the top of `../ganbate-studio/server/ganbate-studio/src/session.rs`: `hello`/`welcome`, `open`/`doc`, `close`, `edit` (answered with
  `ops` or `error`), `delete` (an edit whose ops the host works out, through what prefab instances show), `undo`,
  `redo`, `save`/`saved` (with `force` to overwrite a file changed on disk), `ids`, `create`/`created`, `import`,
  `cook`/`cooked`, and `preview` and `select`, which are relayed to the other clients.
- **Previews are not edits.** While a gizmo drags, the interface moves the entity in its own viewport only
  (and may stream preview transforms to live targets); one op is sent on release. A slider sends ops with a merge
  key, so a whole drag undoes as one step.
- **Undo and redo** are per document, host-side, with labels ("Move Lantern 3"), a limit, merging of consecutive
  same-key edits and a saved marker for the dirty dot. Ctrl+Z in any window undoes the focused document.
- **Save** writes canonical text atomically (a temporary file, then a rename). In M1 unsaved edits live only in the
  host's memory, so a host crash loses them; autosave to `.ganbate/autosave/` is planned (M2). The project files change
  only on Save.

## 3. Panels

| Panel | What it does |
| --- | --- |
| Viewport | The scene drawn by the engine (the Look, time and weather of the scene's environment). Fly and orbit cameras, frame selection, the Look's named views as bookmarks, view modes (lit, unlit, wireframe, bounds, entity colours), grid and snapping. Click to select (world and props), box select, gizmos. |
| Outliner | The entity tree: search, filter by component kind, show/hide and lock per entity, drag to reparent and reorder, prefab instances marked, derived entities shown under their instance. |
| Inspector | Built from schemas: one widget per field type (number with unit and range, vector, rotation as Euler angles, colour picker in sRGB while storing linear, asset slot with picker and drop, entity slot with an eyedropper, enum, arrays, nested objects). Multi-select edits common fields. Overridden prefab fields are marked with revert and apply. Add and remove components. |
| Assets | Folders and search over the asset database, thumbnails, import by drop, kinds filtered, drag into the viewport to place, "where used", rename and move without breaking references. |
| Console | Engine and host messages, validation issues with links to the entity and field, cook output, live-link status. |
| History | The focused document's undo stack; click to jump. |
| Changes | Files changed since the last commit, a diff of canonical JSON per entity, and validation before commit. Commits stay in git. |

## 4. Tools

| Tool | Edits | Writes | Milestone |
| --- | --- | --- | --- |
| Placement | move, rotate, scale (Ctrl snaps a drag in 0.25 m, 15° and 0.25× steps), duplicate, delete, add entity | scene and prefab documents | M1 |
| Placement, more | group, snap to ground and grid, align, place and replace with prefab | scene and prefab documents | M3 |
| Lighting and Look | lights as entities; the Look's sky, time, weather, palette, post-processing, detail rules, live with a time-of-day scrubber | scene, Look | M2 |
| Materials | shading model, colours, textures, alpha, detail preset; prop shaders with live parameters | material documents | M2 |
| Cameras | camera entities, view bookmarks, framing guides for interfaces (safe areas, the game's UI boxes) | scene | M2 |
| Game stages | the game's component kinds placed and edited; anchors that tell the game's interface where things are on screen; preview of game states through the game's plugin | scene, prefab | M3 |
| Sequencer | tracks over time: transforms, camera cuts and blends, clips, cues (sound, particles, game events), scrubbing in the viewport | sequence documents | M3 |
| Characters | animation graphs, look-at and IK targets, spring chains; the character creator: body, parts, morphs, colours, outfits with a library of presets | animation graph, character documents | M6 |
| Effects | particle emitters and their curves, previewed in place; sound banks and mix snapshots | VFX, sound bank documents | M7 |
| Landscape | heightfield sculpt (raise, smooth, flatten, noise, erode), layer paint (grass, rock, sand, path), foliage paint and scatter rules (density, slope, height, spacing, random scale), water bodies | terrain documents and images | M5 |

Every tool edits documents through ops, so every tool gets undo, save, live link and git diffs without extra work.

## 5. Shared machinery

- **Prefab overrides.** Editing an entity shown by an instance writes an override on the instance; the inspector
  marks it, "Revert" removes it, "Apply to prefab" moves it into the prefab document (one undo step across both
  documents). Prefab edits update every open scene that uses it.
- **Play in Editor.** For a web game, PIE opens the game's own page in the viewport frame with the live-link
  client attached and the unsaved documents served by the host, so the game runs on what is being edited. For a
  native game, the host starts the game's player binary with `--live-link`. The game decides what "play" means
  (a fixture, a replay, a bot hand) through its editor plugin.
- **Live link.** A running game or a phone subscribes to documents. Ops that a runtime can apply directly
  (transforms, light values, Look values, material parameters, game component fields) apply in the next frame;
  structural changes (new meshes, imports) trigger an incremental cook and a hot reload of that package. A
  phone on the LAN joins by pairing code, which is how the editor previews on real devices. Browsers give WebGPU
  only to secure pages and refuse `ws:` connections from `https:` pages, so a phone opens the host's own preview
  page over the network (plain HTTP on the LAN, where the viewport falls back to WebGL 2), and a game served over
  HTTPS follows edits through a relay: the host connects out to a WebSocket endpoint the game's server offers, which
  forwards ops to pages that present the pairing code. The engine needs no TLS of its own for either.
- **Validation everywhere.** The host validates every edit (schema, id format, parent cycles) and refuses one that
  adds a problem; `ganbate-build cook-scene` refuses a scene with any problem. Validation on save, checks that the
  referenced parents, prefabs and assets exist, and `ganbate-build check` for commits and CI are planned (M2).
- **Selection and references.** Entities are addressed by id everywhere (outliner, inspector, viewport, console
  links, the game's plugin), so a selection survives renames and moves.
- **Agents and people share one edit path.** An AI agent edits through the `ganbate` command (`scene inspect`,
  `validate`, `apply` with `--dry-run`, `undo`, `diff`, and `render`; short JSON answers with stable error codes),
  which uses the editor's ops, validation and canonical writer, so an agent's change and a person's change are the
  same kind of thing in git and in review. Several edits form one transaction that writes nothing if any of them
  fails. While the host runs, the command sends its transaction through the host (M2), so the change shows live,
  enters the editor's undo history and can never overwrite a document someone has unsaved edits in. An agent can
  also *propose* instead of apply (M3): the viewport shows the proposal as a ghost preview (new and moved entities
  drawn translucent, changed fields marked in the inspector), and a person accepts all of it or some of its ops,
  adjusts it, or rejects it; what is accepted is an ordinary undoable edit. Every transaction records who made it,
  a person or a named agent, so the history doubles as an audit log. Manual editing is never second to an agent's:
  an agent never overwrites a person's work without that person accepting it.

## 6. Absorbing a game's existing tools

A game that already has tuning pages (layout tuners, motion and audio mixers, camera and light inspectors,
review workbenches) moves them in three ways:

| Today | Becomes |
| --- | --- |
| Constants in code (camera profiles, light rigs, colours, layout numbers, timings) | fields of the game's component kinds or engine documents, edited in the inspector |
| A page with sliders that writes to browser storage or a download | the same values in project documents, so they are versioned, reviewed and shipped |
| A custom preview (simulate a game state, replay a recorded session) | a game editor plugin that drives the viewport and PIE |

Tools move one at a time; the old page reads the new documents until it is removed.

## 7. Game editor plugins

A game extends the editor with JavaScript modules listed in its manifest. The plugin API (versioned, documented in
`editor/plugin.d.ts` when it lands in M3) lets a plugin:

- register inspectors for its component kinds (or rely on the schema-drawn default);
- draw in the viewport (gizmos, guides, labels) through overlay props;
- add panels and commands (with undo through ops);
- provide preview states for PIE and the viewport (for example "show the state after event N of this fixture");
- add asset kinds with their own document schema.

Plugins are project code, trusted like the game's own code. They never write files directly; they send ops.

### Authoring a camera

Orbit the viewport and set **Lens** to the desired vertical field of view. **View guides** previews the viewport at `9:16`, `16:9`, or `4:3` with an optional margin inset without modifying the scene. **+ Camera** in the Outliner saves that view as a named scene entity. To change an existing shot, select its camera and choose **Align camera to view**; the transform and lens undo together. **Look through** follows the saved view, including roll and authored near plane, lens shift, and depth-of-field focus edited in the Inspector. A camera under a transformed parent keeps its local scale when aligned.

### Rescanning project assets

Use **Rescan** in Assets after adding or re-exporting files in the project's content folders.
Move a model or Look together with its `.meta.json` file to keep its id. Every open editor
receives the updated paths. If a file changes elsewhere while its document has unsaved edits,
Studio keeps the draft and asks before replacing the disk version. If the source disappears
or changes identity, Save is unavailable until it is restored and rescanned. **Download draft**
exports a local copy. The report lists new, moved and missing assets and unresolved references
in open scenes. A duplicate id leaves the previous asset list in use.

### Authoring a Look

In **Assets**, choose **+ Look** to create a style, or **Edit Look** beside an existing one.
A scene's environment also offers **Edit Look** for its assigned asset. The scene stays in
the viewport while the inspector edits shading, post-processing, sky, weather, time, terrain,
grass, detail rules and named views. Undo, redo and save apply to the document shown in the
toolbar. **Back to scene** returns to the scene's own history. Look identity stays in its
`.meta.json` sidecar, so saving an existing style keeps references to it intact.
A scene hour or weather preset takes precedence over that part of the Look.

## 8. First slice (M1)

The first release is the smallest editor that ends script-driven placement:

1. `ganbate-studio` serves the interface and opens a project; the asset database indexes it and creates sidecars.
2. A static GLB imports as a scene with one entity per node (names and hierarchy kept) instead of being merged;
   skinned, morphed and animated models are refused (they stay run-time models).
3. The viewport draws the scene (cooked on the fly, entities kept) with the Look named by the scene's
   `environment` component; click selects an entity.
4. Outliner, schema inspector and console work; the move/rotate/scale gizmo edits transforms with undo and redo.
5. Save writes canonical JSON; reopening gives the same scene; `ganbate-build cook-scene PROJECT SCENE out.gbew` cooks
   the saved scene into one GBEW v4 file (world plus entity table), which `ganbate-render`, the player (`--scene`) and
   the web runtime read. A per-platform package cook (`ganbate-build cook-project`) is not built yet (see the data page).

M1 has the Viewport (orbit, frame, click to select, the gizmo), the Outliner (tree and filter), the Inspector
(schema fields, add and remove components), Assets (list, open, import as scene, new scene) and the Console; the
rest of each panel comes with later milestones.

Acceptance (met by `../ganbate-studio/client/e2e/run.ts` in headless Chromium on WebGPU and WebGL 2): on the synthetic sample, click
an object to select it, drag it with the gizmo, undo and redo, move a parent and see its child follow, rename
through the inspector, add a component (re-cooking the scene), save, reload and find the move, then cook the saved
scene with `ganbate-build` and render it with `ganbate-render`.
