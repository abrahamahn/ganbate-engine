# Ganbate Engine & Ganbate Studio — Developer Onboarding Manual

Welcome to **Ganbate Engine**. This manual is written for engineers, technical artists, and studio operators opening the codebase or editor for the first time. Read it top-to-bottom in 15 minutes to understand **how the engine is structured, who owns what, what works today, how to edit scenes in Ganbate Studio, how to drive the runtime from code, and how to direct AI agents safely**.

---

## 1. The 60-Second Mental Model

Ganbate is a **general-purpose, zero-third-party-dependency Rust 3D game engine** paired with a browser-based visual editor (**Ganbate Studio**), a deterministic 60 Hz physics engine, a multi-target shader compiler, and a structured JSON CLI (`ganbate`) built for AI agents and build scripts.

Four rules govern everything in this repository:

1. **Zero external Rust dependencies.** The core engine, codecs (`GLB`, `PNG`, `zlib`, `SHA-256`, canonical JSON), physics solver, HTTP/WebSocket Studio host, and `WGSL → SPIR-V / MSL / WGSL / GLSL ES 3.00` shader compiler are written in-house on top of `std` and direct OS/GPU ABI bindings (`Vulkan`, `Metal`, `WebGPU`, `WebGL 2`, `X11`, `Win32`, `AppKit`, `UIKit`).
2. **Engine vs. Game vs. Data ("The 3 Questions").** Before writing code, ask:
   - *Could a second, unrelated game use this mechanism unchanged?* → Put it in **`ganbate-engine`**.
   - *Does this encode one title's rules, network authority, economy, characters, or HTML menus?* → Put it in **that game's repository** (never in `ganbate-engine`).
   - *Is this a placement, camera angle, sky palette, material tint, or timing curve someone will tune?* → Put it in **canonical JSON content documents** (`.scene.json`, `.prefab.json`, `.look.json`) edited in **Ganbate Studio**, not hardcoded constants.
3. **Content is human-readable JSON in Git; runtime worlds are compact binary (`.gbew`).** Authoring files format canonically so Git diffs stay line-by-line readable. Cooking (`ganbate-build`) packs them into `.gbew` binary worlds (~7 bytes/triangle with `--compact`) while keeping an **entity table** (GBEW v4) so game code can still find, move, hide, or pick individual authored objects at runtime without extra draw calls.
4. **One viewport everywhere.** Ganbate Studio's 3D viewport runs the exact same `ganbate-web` WASM runtime that web games ship in production. What you frame and light in the editor is pixel-for-pixel what players see.

---

## 2. Architecture & Crate Ownership Map

### The Six Engine Layers

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  GAME REPOSITORY (outside ganbate-engine)                                   │
│  Game rules, authoritative server, HTML/DOM UI, art assets, project tools   │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 6 · TOOLS, WEB HOST & AGENT CLI                                      │
│  crates/ganbate-studio + editor/  │  crates/ganbate-agent (`ganbate` CLI)   │
│  crates/web + web/        │  src/bin/ganbate-{build,render,player}  │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 5 · RUNTIME PRESENTATION & SIMULATION                                │
│  src/render/* (Stylized Look renderer, props, shadows, water, sky/weather)  │
│  src/anim.rs  (GPU skinning, weighted clip poses, sparse CPU facial morphs) │
│  crates/physics (Deterministic 60 Hz rigid bodies, joints, KCC)     │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 4 · SCENE DOCUMENTS & ASSET PIPELINE                                 │
│  crates/scene    (Entities, component schemas, prefabs, op undo)    │
│  crates/pipeline (`ganbate.project.json`, `.meta.json`, scene cook) │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 3 · SHADER COMPILER & BINARY CODECS                                  │
│  crates/shader (WGSL subset → SPIR-V, MSL, WebGPU WGSL, GLSL ES 3) │
│  crates/assets (GLB loader, PNG/zlib codec, `.gbew` v1–v4 format)   │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 · UNIFIED GPU RHI & PLATFORM WINDOWS                               │
│  src/gpu/* → src/vulkan, src/metal, src/gpu/webgpu, src/gpu/webgl           │
│  src/window/* (X11, Win32, macOS AppKit, iOS UIKit)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 1 · PORTABLE CORE                                                    │
│  crates/core (Math, Transform, SHA-256, permanent IDs, canon JSON)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Who Owns What Inside `ganbate-engine`

| Path / Crate | Binary / Output | Owns | Touch When… |
| --- | --- | --- | --- |
| `crates/core` | lib | Permanent 64/128-bit IDs, SHA-256 digests, canonical JSON parser/writer, frame-pacing math, `EntityTable` | Changing IDs, canonical JSON rules, or shared world metadata |
| `crates/shader` | lib + `ganbate-shaderc` example | Custom WGSL subset parser, type checker, reflection, and codegen to SPIR-V, Metal MSL, WebGPU WGSL, and WebGL 2 GLSL ES 3.00 | Adding shader builtins, fixing backend codegen, or extending `ShadeInput` |
| `crates/assets` | lib | Embedded glTF 2.0 (`.glb`) static & skinned model loaders, PNG/zlib decoder, `.gbew` v1–v4 binary world encoder/decoder (`--compact` mesh compression) | Supporting new glTF attributes/codecs or changing `.gbew` wire layout |
| `crates/scene` | lib | `.scene.json` and `.prefab.json` data model, `engine.schema.json`, base-62 sibling ordering, prefab flattening/overrides, invertible `Op` undo/redo | Adding built-in engine component schemas (`camera`, `light`, `rigid_body`, etc.) |
| `crates/pipeline` | lib | `ganbate.project.json` discovery, `.meta.json` sidecar generation/indexing, object-preserving GLB→scene import, cooking scenes into GBEW v4 | Changing project indexing, asset sidecars, or scene cooking |
| `crates/physics` | lib | Deterministic 60 Hz soft-step rigid-body solver (`+`, `-`, `*`, `/`, `sqrt` only), GJK/EPA, shapes, 5 joint types, CCD, ray/shape/overlap queries, character controller | Modifying collision shapes, physics stepping, joints, or character movement |
| `src/` (`ganbate-engine` root) | lib | Unified RHI (`src/gpu/`), Vulkan/Metal/WebGPU/WebGL 2 backends, stylized `Look` renderer (`src/render/`), GPU skinning & morphs (`src/anim.rs`), native windowing (`src/window/`) | Changing rendering passes, Look parameters, dynamic props, lights, or GPU backends |
| `crates/web` + `web/` | `ganbate_web.wasm`, `ganbate.js`, `ganbate.d.ts` | Browser WASM runtime (`ganbate_rt_*` exports) and JS/TS wrapper supporting standalone viewer mode and `createEngine()` library mode | Adding or updating methods exposed to TypeScript web games or the Studio viewport |
| `crates/ganbate-studio` + `editor/` | `ganbate-studio` (`:7480`) | Local Rust HTTP + WebSocket project server (`crates/ganbate-studio`) and browser UI (`editor/studio.js`, Asset Library, Outliner, Inspector, Viewport gizmos) | Adding editor panels, gizmos, asset previews, rescan rules, or `project-tools.json` hooks |
| `crates/ganbate-agent` | `ganbate` | Single-line JSON CLI (`scene inspect`, `validate`, `apply`, `undo`, `diff`, `render`) + `.ganbate/history/` undo journals | Extending AI-agent scene editing, validation, or headless verification |
| `src/bin/ganbate-build.rs` | `ganbate-build` | Asset CLI (`import`, `cook`, `cook-scene`, `inspect`, `model`, `simulate`) | Adding offline asset build/cook subcommands |
| `src/bin/ganbate-render.rs` | `ganbate-render` | Headless Vulkan offscreen PNG/PPM still & 24 fps time-lapse renderer | Rendering reference stills, thumbnails, or CI regression diffs |
| `src/bin/ganbate-player.rs` | `ganbate-player` | Native desktop fly-through viewer (X11 / Win32 / macOS Metal) | Testing native swapchain presentation and interactive scene fly-throughs |
| `src/bin/ganbate-bench.rs` & `ganbate-framecheck.rs` | `ganbate-bench`, `ganbate-framecheck` | Offscreen throughput benchmark and real-device 120 Hz / 240 Hz displayed-frame cadence verifier | Qualifying device performance and renderer cutover gates |

---

## 3. Honest Status: What Works Today vs. Roadmap

Always distinguish **what runs today** from **planned milestones** (full breakdown in [`design/02-runtime.md`](design/02-runtime.md) and [`roadmap.md`](roadmap.md)):

| Area | Status Today | Details |
| --- | --- | --- |
| **GPU RHI & 4 Backends** | **Shipped (M0)** | Vulkan, Metal, WebGPU, and WebGL 2 behind one `gpu::Device` interface; custom `ganbate-shader` compiler builds every pass across all four targets. (Linux Vulkan + Chromium WebGPU/WebGL 2 tested in CI; Apple Metal & Windows Win32 compile-checked, awaiting hardware qualification.) |
| **Stylized World & Prop Renderer** | **Shipped (M0–M1)** | Soft cel brushwork, 2 shadow cascades, tiled point lights + 64 dynamic game lights (up to 2 shadow-casting spots), planar water with shallows/foam, sun/moon sky + 7 weathers (`clear`, `cloudy`, `overcast`, `mist`, `rain`, `storm`, `petals`), wind-swaying grass, SSAO, outlines, bloom, DoF, overlays, custom `ShadeInput` prop shaders, GPU skinning, and sparse CPU facial morphs. |
| **Scene Identity, Cook & Studio Editor** | **Shipped (M1)** | `ganbate.project.json`, permanent `.meta.json` asset IDs, object-preserving GLB→scene import, GBEW v4 entity tables, Ganbate Studio (`localhost:7480`) with Asset Library, Outliner, schema-driven Inspector, gizmos, undo/redo, EN/KO UI, and `project-tools.json` workspace integration. |
| **Deterministic 60 Hz Physics** | **Shipped (ahead of M9)** | Spheres, capsules, boxes, convex hulls, triangle meshes, heightfields; static/kinematic/dynamic bodies; 5 joints; CCD; ray/shape/overlap queries; kinematic character controller; identical 64-bit `state_hash` across native and WASM. Press **`P`** in Studio to simulate live. |
| **AI Agent Scene CLI (`ganbate`)** | **Shipped** | Structured single-line JSON `inspect`, `validate`, `apply` (with `--dry-run`), `diff`, `undo`, and offscreen `render`. |
| **Live Link, Device Loss & Packages** | **Next (M2)** | Live WebSocket sync into a running external game tab, content-addressed platform packages, GPU context-loss recovery, adaptive resolution/idle pacing. |
| **Sequencer, Sub-views & Variants** | **Planned (M3–M4)** | Multi-track timeline editor, vertex-stage card/prop deformers, secondary camera sub-views, cosmetic prefab variants, full device frame-gate qualification. |
| **Terrain, Anim Graphs & Engine Audio** | **Planned (M5–M8)** | Heightfield sculpting & foliage brushes (M5), animation state machines + look-at/IK + spring bones + character assembly (M6), native audio buses & particles (M7), native iOS/Android app shells (M8). |

---

## 4. Ganbate Studio Manual (Visual Editor)

Ganbate Studio consists of a **local Rust host** (`ganbate-studio`) that owns disk I/O, asset indexing, cooking, and undo history, paired with a **browser interface** (`editor/`) whose central viewport runs `ganbate-web`.

### 4.1 Build & Launch on a Sample Project

From the `ganbate-engine` root:

```sh
# 1. Build the browser runtime used by Ganbate Studio's 3D viewport (once, or after engine render changes)
rustup target add wasm32-unknown-unknown
scripts/build-web.sh web/dist

# 2. Generate a synthetic sample project to explore safely
cargo run --release --example studio_sample -- ~/ganbate-sample

# 3. Launch Ganbate Studio and open http://localhost:7480/
cargo run --release -p ganbate-studio -- ~/ganbate-sample
```

To open **any real game project**, pass the directory containing its `ganbate.project.json` instead of `~/ganbate-sample`:

```sh
cargo run --release -p ganbate-studio -- /path/to/game/project
```

### 4.2 Anatomy of a Ganbate Project Folder

```text
my-game-project/
├── ganbate.project.json          # Project manifest: id, name, startup scene id, content[], editor[], schemas[]
├── content/
│   ├── scenes/
│   │   └── main.scene.json       # Canonical JSON scene (entities, parent tree, components)
│   ├── looks/
│   │   ├── dusk.look.json        # Visual style, sky, weather, detail rules, named views
│   │   └── dusk.look.json.meta.json
│   └── models/
│       ├── lantern.glb
│       └── lantern.glb.meta.json # Permanent 128-bit asset ID — keep alongside file when moving/renaming!
├── editor/
│   └── project-tools.json        # Optional bridge to game-specific web tools (e.g., http://127.0.0.1:5173)
└── .ganbate/                     # Local uncommitted cache: cooked previews & agent undo journals
```

> **Critical `.meta.json` Rule:** Scenes and prefabs store their permanent ID inside their own JSON. Non-document files (`.glb`, `.png`, `.webp`, `.look.json`, `.wav`) store their permanent 128-bit ID in `<filename>.meta.json`. When renaming or moving an asset in the Studio **Assets** panel, Studio moves both files automatically so no scene reference breaks. If you move an asset in a shell, always `git mv` its `.meta.json` alongside it!

### 4.3 Viewport & Editor Controls

| Action | Shortcut / Input |
| --- | --- |
| **Select object** | `Left-click` in Viewport or Outliner |
| **Move / Rotate / Scale gizmo** | `W` (Move) · `E` (Rotate) · `R` (Scale) |
| **Snap gizmo increments** | Hold `Ctrl` while dragging gizmo |
| **Orbit camera** | `Right-drag` |
| **Pan camera** | `Shift + Right-drag` or `Middle-drag` |
| **Zoom camera** | `Mouse wheel` |
| **Frame selected entity** | `F` |
| **Simulate 60 Hz physics** | `P` (press `P` again to stop and restore exact authored transforms) |
| **Undo / Redo** | `Ctrl+Z` / `Ctrl+Shift+Z` (tracked on the Rust host) |
| **Duplicate entity** | `Ctrl+D` |
| **Delete entity** | `Delete` (when Viewport or Outliner has focus) |
| **Save open document** | `Ctrl+S` |
| **Switch UI language** | **EN / 한국어** toggle in top-right toolbar |

### 4.4 Common Editor Workflows

1. **Importing a Static 3D Environment vs. Animated Character:**
   - **Static architectural/environment `.glb`:** Select it in **Assets** → click **Import as scene**. Every glTF node becomes an editable entity with its hierarchy and transform preserved (`mesh.static: true`). Cooking batches identical materials automatically, so keeping objects separate costs zero runtime draw calls.
   - **Skinned / morphed / animated character `.glb`:** Keep it as a **runtime model** (do *not* flatten into static scene nodes). Reference it from game code via `engine.loadModel(url)` or preview it in the Asset Library.
2. **Authoring Cameras & Looks:**
   - Orbit the viewport to frame a shot → select a Camera entity in the Outliner → click **Align camera to view** (or click **+ Camera** to create a new shot from your current viewpoint). Click **Look through** to lock the viewport to that camera.
   - Select the scene's atmosphere/environment entity → click **Edit Look** to tune cel shading, outlines, bloom, exposure, sky latitude/time, weather, and material surface rules (`wood`, `stone`, `roof_tiles`, `lacquer`, `grass`, `terrain`). Remember to save both the Look and the Scene.
3. **Bridging Game-Specific Editors (`project-tools.json`):**
   - Place `project-tools.json` inside a folder listed in `ganbate.project.json`'s `"editor"` array:
     ```json
     {
       "version": 1,
       "origin": "http://127.0.0.1:5173",
       "tools": [
         { "label": "Character Expressions", "path": "/dev/expressions" },
         { "label": "Live Operations", "path": "/dev/ops", "open": "tab" }
       ]
     }
     ```
   - Clicking **Project tools** embeds the game's own local web tool panel (or opens `"open": "tab"` in a new browser tab). Keep the game server on a distinct hostname (`127.0.0.1` vs `localhost`) so Studio's session cookie is never sent to the game server.
4. **Testing on a Phone Over Wi-Fi (`--lan`):**
   - Run `cargo run --release -p ganbate-studio -- --lan /path/to/project`. Studio prints a pairing URL with a one-time token; open that URL once on your phone or desktop browser to set the pairing cookie. (Note: plain HTTP over LAN uses the **WebGL 2** backend on mobile browsers because browsers restrict WebGPU to `localhost` or `https://`.)

---

## 5. Asset & Cooking Pipeline (`ganbate-build` & `ganbate-render`)

### Key `ganbate-build` Commands

```sh
# Import a static GLB into an editable canonical JSON scene (1 entity per glTF node)
cargo run --release --bin ganbate-build -- import ./my-project content/models/room.glb content/scenes/room.scene.json

# Cook an authored scene into a compact browser-ready GBEW v4 world (with entity table!)
cargo run --release --bin ganbate-build -- cook-scene --compact --keep-images \
  ./my-project content/scenes/room.scene.json dist/room.gbew

# Cook a single static GLB directly to GBEW v3 (without a .scene.json project)
cargo run --release --bin ganbate-build -- cook --compact --keep-images source.glb dist/room.gbew

# Inspect meshes, materials, textures, lights, and GBEW v4 entity table inside a cooked world
cargo run --release --bin ganbate-build -- inspect dist/room.gbew

# Validate a runtime character/prop GLB (prints nodes, primitives, materials, skins, morphs, clips)
cargo run --release --bin ganbate-build -- model character.glb

# Run 120 ticks (2.0 s at 60 Hz) of deterministic headless physics and print final 64-bit state hash
cargo run --release --bin ganbate-build -- simulate dist/room.gbew 120
```

- **Why `--compact --keep-images` for web builds?**
  - `--compact` quantizes positions to a 0.5 mm grid, encodes normals as 12-bit octahedral vectors, delta-codes indices, and deflates streams (~7 bytes/triangle — roughly **5× smaller**).
  - `--keep-images` preserves compressed PNG/JPEG/WebP payloads instead of expanding uncompressed mipmaps on disk; browsers decode JPEG/WebP natively, the engine decodes PNG directly (avoiding HTML canvas premultiplied-alpha corruption), and mipmaps generate on the GPU.
  - Note: Export uncompressed GLBs (without external Draco or `meshopt` extensions) before passing them to `ganbate-build`; `ganbate-build --compact` applies Ganbate's own zero-dependency mesh optimization and `.gbew` compression.

### Headless Stills & Time-Lapses (`ganbate-render`)

```sh
# Render every named camera view in look.json at 1080p with 16x supersampling
cargo run --release --bin ganbate-render -- \
  --scene dist/room.gbew --look content/looks/dusk.look.json \
  --all-views --time 20.5 --weather clear --width 1920 --height 1080 --samples 16 --out-dir renders

# Render a 24 fps time-lapse sequence from 17:30 to 21:00 (84 frames)
cargo run --release --bin ganbate-render -- \
  --scene dist/room.gbew --look content/looks/dusk.look.json \
  --sequence 17.5 21.0 84 --out-dir timelapse
```

*(On headless Linux servers without a physical GPU, prefix with `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json` to run on Mesa lavapipe.)*

---

## 6. Driving the Engine from Game Code (`createEngine`)

Web games embed `ganbate_web.wasm` + `ganbate.js` (`ganbate.d.ts` provides full TypeScript types) and drive rendering on demand:

### 6.1 Core Lifecycle, Camera, World Entities & Props

```ts
import { createEngine } from './ganbate.js';

const engine = await createEngine({
  canvas,
  wasm: '/assets/ganbate_web.wasm',
  world: '/assets/harbor.gbew', // optional: omit for a props-only stage
  look: lookJsonObject,
});

// 1. Match canvas backing resolution to CSS box * devicePixelRatio
engine.fitCanvas();

// 2. Frame camera (supports off-center lens `shift` for UI clearance + depth-of-field `focus`)
engine.setView({
  eye: [0, 3.2, 7.5],
  target: [0, 0.9, 0],
  fov: 42,
  shift: [0, 0.08],
  focus: { distance: 8.5, depth: 6.0, blur: 0.015 },
});

// 3. Atmosphere & world entity control (GBEW v4)
engine.setClock(20.25, performance.now() / 1000);
engine.setWeather('petals');
const entities = engine.worldEntities(); // authored IDs, names, parents, instance handles

// 4. Dynamic game props & lights
const mesh = engine.createMesh({ positions, normals, uvs, indices });
const tex = engine.createTexture(imageBitmapOrCanvas);
const mat = engine.createMaterial({ name: 'lacquer', color: [1, 1, 1, 1], baseTexture: tex });
const prop = engine.createInstance(mesh, mat, columnMajorMat4);

const keySpot = engine.createLight({
  position: [0, 3.5, 1.2],
  direction: [0, -1, -0.3],
  color: [1.0, 0.92, 0.8],
  intensity: 14,
  innerAngle: 0.35,
  outerAngle: 0.65,
  shadow: 0.85, // up to 2 spot lights cast real-time 1024px 8-tap shadows across opaque props
});

// 5. Skinned & morphed GLB characters
const heroModel = await engine.loadModel('/assets/character.glb');
const hero = engine.createModelInstance(heroModel, rootMat4);
engine.modelClip(hero, 'idle_breathe', timeSeconds, 1.0); // weight 1.0 = base pose, <1.0 = cross-fade blend
engine.modelMorph(hero, 'FaceMesh', morphTargetIndex, 0.75); // sparse CPU morph updates ONLY affected vertices
engine.updateModel(hero);
const rightHandMat4 = engine.modelNodeWorld(hero, 'Hand_R'); // socket transform for held props!

// 6. Draw frame, project 3D anchors to DOM CSS pixels, and pick under pointer
engine.render();
const hudPos = engine.project([0, 1.4, 0]); // -> [cssX, cssY, depth] or null if behind camera
const hit = engine.pick(pointerEvent.offsetX, pointerEvent.offsetY);
```

### 6.2 Custom Overlay Prop Shaders (`ShadeInput`)

Games can write custom fragment effects in Ganbate's portable WGSL subset without touching engine code. `ganbate-shader` compiles them at runtime to WebGPU WGSL, WebGL 2 GLSL ES 3.00, Vulkan SPIR-V, or Metal MSL:

```wgsl
fn shade(in: ShadeInput) -> vec4f {
  // in.uv, in.world_pos, in.normal, in.tint, in.color
  // in.base, in.normal_map, in.emissive_map (pre-sampled at in.uv)
  // in.params[0..19] (20 dynamic f32 uniforms updated cheaply via engine.setMaterialParams)
  let pulse = 0.5 + 0.5 * sin(in.params[0] * 4.0);
  let alpha = in.base.a * in.tint.a;
  return vec4f(in.base.rgb * (1.0 + 0.35 * pulse), alpha);
}
```

### 6.3 Deterministic Runtime Physics

```ts
engine.physicsStart();
engine.physicsStep(1 / 60, 1); // advances cooked rigid_body/collider entities and syncs transforms
const stateHash = engine.physicsHash(); // identical 64-bit hash on Chrome, Safari, Linux & Windows
```

---

## 7. Working With AI Agents (`ganbate` CLI)

When an AI coding agent edits a `.scene.json` file by hand with regex or raw text tools, it risks inventing invalid schema fields, breaking base-62 sibling order, or clobbering permanent hex IDs. **Always instruct AI agents to use the `ganbate` CLI (`crates/ganbate-agent`) instead.**

### Build the CLI

```sh
cargo build --release -p ganbate-agent --bin ganbate
export PATH="$PWD/target/release:$PATH"
```

### The Standard Agent Loop (`Inspect → Dry-Run → Apply → Render → Verify`)

Every `ganbate` command returns **one line of machine-parseable JSON** (`{"ok":true,...}` or `{"ok":false,"error":{"code":"...","message":"..."}}`).

```sh
P=/path/to/project
S=content/scenes/main.scene.json

# 1. Inspect hierarchy or a specific subtree/component
ganbate scene inspect "$P" "$S" --depth 2
ganbate scene inspect "$P" "$S" --entity "Courtyard/Lantern" --defaults

# 2. Validate against engine + project schemas before & after edits
ganbate scene validate "$P" "$S"

# 3. Preview an atomic multi-op edit transaction (--dry-run writes nothing, predicts exact new IDs)
ganbate scene apply "$P" "$S" --dry-run - <<'EOF'
{
  "label": "Warm courtyard lantern and add stone bench",
  "edits": [
    { "op": "set", "entity": "Courtyard/Lantern", "path": "components.light.intensity", "value": 24 },
    { "op": "create", "name": "Stone Bench", "parent": "Courtyard",
      "components": { "transform": { "translation": [2.5, 0.0, -1.2] } } }
  ]
}
EOF

# 4. Apply for real (records inverse ops & pre/post SHA-256 in $P/.ganbate/history/<scene-id>.jsonl)
ganbate scene apply "$P" "$S" edits.json

# 5. Visually verify by cooking in memory and rendering a PNG frame
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ganbate render "$P" "$S" --out preview.png --size 960x540 --samples 4

# 6. Diff against Git HEAD or roll back byte-for-byte if dissatisfied
git show "HEAD:$S" | ganbate scene diff - "$P/$S"
ganbate scene undo "$P" "$S"
```

### Single-Writer Rule Alongside Ganbate Studio

Until live CLI-to-Studio IPC lands in **M2**:
- **Do not edit the same scene in an unsaved Studio browser tab and via `ganbate scene apply` at the same time.**
- Save (`Ctrl+S`) in Studio before running `ganbate scene apply`, then reload the Studio tab (`Ctrl+R`) to pick up the agent's canonical changes. Both Studio (`changed-since` disk guard) and `ganbate scene undo` (`changed-since` SHA-256 check) refuse to silently overwrite edits made by the other.

---

## 8. Where to Read Deeper

- **Design Decisions (read in order):** [`design/README.md`](design/README.md)
  - [`01-studio-workflow.md`](design/01-studio-workflow.md) — DCC → Import → Studio → Cook → Runtime roles
  - [`02-runtime.md`](design/02-runtime.md) — Every subsystem's status (`present` / `partial` / `absent`), file evidence & priority
  - [`03-editor.md`](design/03-editor.md) — Ganbate Studio host/browser protocol & panel architecture
  - [`04-data.md`](design/04-data.md) — Canonical JSON documents, permanent IDs, `.meta.json` sidecars & GBEW v4
  - [`05-boundary.md`](design/05-boundary.md) — Exact Engine vs. Game ownership rules
  - [`06-physics.md`](design/06-physics.md) — Soft-step solver math & cross-platform determinism rules
- **Interactive Layer Map:** Download and open [`architecture.html`](architecture.html) in a browser
- **Milestones & Acceptance Gates:** [`roadmap.md`](roadmap.md)
