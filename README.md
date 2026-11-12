# Ganbate Engine

**Ganbate Engine** is a general-purpose Rust 3D game engine, visual editor (**Ganbate Studio**), asset pipeline, deterministic physics solver, and structured AI-agent CLI designed to power multiple cross-platform games from one zero-third-party-dependency codebase.

Game rules, network protocols, accounts, HTML menus, and title-specific artwork belong in each game's own repository—never inside `ganbate-engine`.

- **New here? Start with the [Developer Onboarding Manual (`docs/ONBOARDING.md`)](docs/ONBOARDING.md)** for a 15-minute walkthrough of ownership, Studio controls, cooking, TypeScript integration, and AI-agent workflows.
- **Architecture & Decisions:** [Design Index (`docs/design/README.md`)](docs/design/README.md) · [Interactive 6-Layer Map (`docs/architecture.html`)](docs/architecture.html) · [Milestones (`docs/roadmap.md`)](docs/roadmap.md)

![The native player on a Linux virtual display](docs/player.png)

---

## Repository & Crate Map

The core workspace uses **zero third-party Cargo crates**, calling OS/GPU ABIs directly (`Vulkan`, `Metal`, `WebGPU`, `WebGL 2`, `X11`, `Win32`, `AppKit`, `UIKit`) and compiling shaders in-house.

| Path | Output | Responsibility |
| --- | --- | --- |
| `src/` (`ganbate-engine`) | lib + native bins | Unified GPU RHI (`gpu`), Vulkan/Metal/WebGPU/WebGL 2 backends, stylized `Look` renderer (`render`), GPU skinning & sparse facial morphs (`anim`), native windows (`window`) |
| `crates/ganbate-core` | lib | Math, permanent 64/128-bit IDs, SHA-256 digests, canonical JSON reader/writer, frame pacing, `EntityTable` |
| `crates/ganbate-shader` | lib | Custom WGSL subset compiler → Vulkan `SPIR-V`, Apple `MSL`, `WebGPU` WGSL, and `WebGL 2` GLSL ES 3.00 |
| `crates/ganbate-assets` | lib | Embedded glTF 2.0 (`.glb`) loader, PNG/zlib codecs, cooked `.gbew` (v1–v4) world format (`--compact` ~7 B/tri) |
| `crates/ganbate-scene` | lib | Canonical `.scene.json` & `.prefab.json` documents, component schemas, prefab overrides, invertible `Op` undo/redo |
| `crates/ganbate-pipeline` | lib | `ganbate.project.json` discovery, `.meta.json` permanent ID sidecars, object-preserving GLB import, GBEW v4 cooking |
| `crates/ganbate-physics` | lib | Deterministic 60 Hz rigid-body solver (`+`, `-`, `*`, `/`, `sqrt` only), 5 joints, CCD, spatial queries, character controller |
| `crates/ganbate-web` + `web/` | `ganbate_web.wasm`, `ganbate.js`, `ganbate.d.ts` | Browser WASM runtime linking both WebGPU and WebGL 2; supports standalone viewer and `createEngine()` library mode |
| `../ganbate-editor/crates/ganbate-studio` + `../ganbate-editor/editor/` | `ganbate-studio` | Local HTTP/WebSocket editor host (`localhost:7480`) + browser UI (Asset Library, Outliner, Inspector, `ganbate-web` viewport) |
| `../ganbate-editor/crates/ganbate-agent` | `ganbate` | Single-line JSON CLI for AI agents & scripts (`scene inspect`, `validate`, `apply`, `undo`, `diff`, `render`) |
| `src/bin/ganbate-*` | `ganbate-player`, `ganbate-render`, `ganbate-build`, `ganbate-bench`, `ganbate-framecheck`, `ganbate-ios` | Native desktop player, headless renderer, asset importer/cooker, throughput benchmark, and 120/240 Hz display-pacing verifier |

---

## 1. Edit a Scene in Ganbate Studio (Visual Editor — `../ganbate-editor`)

Ganbate Studio (`../ganbate-editor`) pairs a local Rust host (`ganbate-studio`) that owns a project folder (`ganbate.project.json`) and undo history with a browser UI (`editor/`) whose 3D viewport runs the live `ganbate-web` runtime ([data design](docs/design/04-data.md)).

```sh
scripts/build-web.sh web/dist                                                         # 1. Build the browser viewport runtime
cargo run --release --example studio_sample -- ~/ganbate-sample                       # 2. Generate a synthetic sample project
cargo run --release --manifest-path ../ganbate-editor/Cargo.toml -p ganbate-studio -- ~/ganbate-sample  # 3. Start host, then open http://localhost:7480/
```

### Studio Controls & Workflows
- **Import & Assets:** Open a scene from **Assets**. Static `.glb` environments import via **Import as scene** (one entity per glTF node with names and hierarchy preserved; skinned/morphed/animated `.glb` files stay runtime models). Every non-document asset carries a permanent ID in `<file>.meta.json`—keep sidecars next to their source files when moving or renaming.
- **Viewport & Gizmos:** Click to select; **`W` / `E` / `R`** switch Move / Rotate / Scale gizmos (hold **`Ctrl`** to snap); **Right-drag** orbits, **`Shift`+Right-drag** or **Middle-drag** pans, **Wheel** zooms, **`F`** frames selection, **`P`** toggles 60 Hz physics simulation.
- **Editing & History:** Schema-driven Inspector; **`Ctrl+Z` / `Ctrl+Shift+Z`** undo/redo on the host; **`Ctrl+S`** saves canonical JSON; **`Ctrl+D`** duplicates; **`Delete`** removes selected entities (text inputs keep Backspace/Delete). Structural edits re-cook automatically; transform moves apply immediately. Toggle **EN / 한국어** at top right.
- **LAN & Phone Preview:** By default the host listens only on loopback (`localhost:7480`). Pass `--lan` to listen on the local network and print a pairing URL with a one-time token (saved in a browser cookie).
- **Game-Specific Workspace Bridge (`project-tools.json`):** Any folder listed in `ganbate.project.json`'s `"editor"` array can register external web tools:
  ```json
  {"version":1,"origin":"http://127.0.0.1:5173","tools":[{"label":"Level rules","path":"/tools/level-rules"}]}
  ```
  Choose **Project tools** to open a workspace inline (or add `"open":"tab"` for a toolbar tab link). Keep the game server on a different hostname from Studio (`127.0.0.1` vs `localhost`, or pass `?toolOrigin=http://127.0.0.1:5174`) so Studio's pairing cookie never reaches the game server. Pass `?asset=path/to/file` to reveal or open an asset directly.

---

## 2. Cook Worlds & Inspect Models (`ganbate-build`)

```sh
# Import a static GLB into an editable scene, then cook to GBEW v4 (preserves entity table!)
cargo run --locked --offline --release --bin ganbate-build -- import ./project content/model.glb content/model.scene.json
cargo run --locked --offline --release --bin ganbate-build -- cook-scene --compact --keep-images ./project content/model.scene.json world.gbew

# Or cook a single static GLB directly to GBEW v3, inspect its contents, or validate a runtime skinned GLB
cargo run --locked --offline --release --bin ganbate-build -- cook --compact --keep-images source.glb scene.gbew
cargo run --locked --offline --release --bin ganbate-build -- inspect scene.gbew
cargo run --locked --offline --release --bin ganbate-build -- model character.glb
```

- **Supported inputs:** Embedded uncompressed glTF 2.0 triangle meshes, PBR material factors, PNG/JPEG/WebP base-colour/normal/emissive textures, `KHR_materials_emissive_strength`, `KHR_lights_punctual` point lights, alpha modes, double-sidedness, `extras.ganbate.shading` (`standard`, `foliage`, `water`, `unlit`), and `extras.ganbate.emission` (`lamp`, scaled by time of day). External geometry compression codecs (`KHR_draco_mesh_compression`, `EXT_meshopt_compression`) are intentionally excluded and rejected explicitly in favour of native `--compact` `.gbew` cooking and `ganbate-assets::mesh_opt`.
- **Browser compression (`--compact --keep-images`):** Shrinks worlds to ~1/5th size (~7 bytes/triangle via 0.5 mm grid quantization, 12-bit octahedral normals, 1/8192 UVs, delta coding, and deflate) while keeping PNG/JPEG/WebP files intact (`EXT_texture_webp` supported). Browsers decode JPEG/WebP natively while the engine decodes PNG directly (avoiding 2D canvas premultiplied-alpha loss) and builds mip chains on the GPU.

---

## 3. Run in a Browser (WebGPU & WebGL 2)

`crates/ganbate-web` (`ganbate_web.wasm`) links **both** WebGPU (Chrome/Edge 113+, Safari 26 on macOS/iPhone/iPad) and WebGL 2 (older iOS, Android GPUs without WebGPU, Linux Firefox/Chrome) with zero third-party crates. `web/ganbate.js` selects WebGPU automatically (`?api=webgl2` forces WebGL 2; `?api=webgpu` disables fallback).

```sh
rustup target add wasm32-unknown-unknown
scripts/build-web.sh                  # writes web/dist/{index.html,ganbate.js,ganbate.d.ts,ganbate_web.wasm}
```

*(Without rustup's prebuilt `wasm32` stdlib, build std from source via `GANBATE_WEB_TOOLCHAIN=build-std GANBATE_WEB_RUSTC=rustc-1.91 GANBATE_WEB_CARGO=cargo-1.91 GANBATE_WEB_RUST_SRC=/path/to/rust-src-1.91/library scripts/build-web.sh`.)*

Serve over `localhost` or `https` (required by WebGPU) outside this repo alongside your cooked assets:

```sh
scripts/build-web.sh ~/ganbate-site
cp world.gbew look.json ~/ganbate-site/
python3 -m http.server -d ~/ganbate-site 8000
# Open http://localhost:8000/?scene=world.gbew&look=look.json
```

URL parameters: `scene`, `look`, `view=NAME`, `time=HOURS`, `weather=PRESET`, `seconds=S`, `paused=1`, `quality=high`, `dpr=N` (default 2, 1.5 on touch), `scale=F`, `size=WxH`. Controls: `W/A/S/D`, `Q/E`, arrows/drag, `Shift`, `1–9` views, `T/G` time, `C` cycle, `Y` weather, `P` pause, `H` help (1-finger look / 2-finger move on touch).

### Library Mode: Driving the Engine from TypeScript / JavaScript

Games with their own UI import `createEngine` (`web/ganbate.d.ts` ships full TypeScript declarations):

```js
import { createEngine } from './ganbate.js';
const engine = await createEngine({ canvas, wasm: 'ganbate_web.wasm', world: 'world.gbew', look });
engine.fitCanvas();                                   // canvas size = CSS box * devicePixelRatio
engine.setView({ eye: [0, 4, 10], target: [0, 0.5, 0], fov: 45, shift: [0, 0.1] });
engine.setClock(18.5, performance.now() / 1000);     // time of day, animation clock
engine.setWeather('clear');
engine.setLampBoost(1.9);                             // momentary lamp swell (1.0 = normal hour)
const mesh = engine.createMesh({ positions, normals, uvs, indices });
const art = engine.createTexture(canvasOrImageBitmap);
const felt = engine.createMaterial({ name: 'felt', color: [0.1, 0.3, 0.2, 1], baseTexture: art });
const chip = engine.createInstance(mesh, felt, matrix); // 16 numbers, column-major
engine.setInstances([{ instance: chip, matrix, visible: true }]);
engine.render();
engine.project([0, 1, 0]);                            // 3D world point -> CSS pixels for DOM anchors
engine.pick(event.offsetX, event.offsetY);            // triangle-accurate prop picking
```

- **Cooked Entity Access:** `engine.worldEntities()`, `setWorldInstance`, `pickWorld`, and `setWorldLight` inspect, move, hide, and pick GBEW v4 scene entities at runtime.
- **Dynamic Lights & Shadows:** `engine.createLight` places up to 64 tiled point or spot lights; up to 2 spots cast 8-tap filtered 1024px real-time prop shadows (`shadow` factor).
- **Skinned & Morphed Runtime Models:** `engine.loadModel(url)` loads GLBs; `createModelInstance`, `modelClip` (weighted cross-fades), `setModelNode`, `modelMorph` (gathered sparse CPU vertex morphs via `anim::gather_morphed`), `updateModel`, and `modelNodeWorld` (socket attachments) animate characters without mutating static body geometry.
- **Custom Overlay Prop Shaders:** `engine.createShader(source)` compiles portable WGSL (`fn shade(in: ShadeInput) -> vec4f` using `src/render/shaders/prop_shader_input.wgsl`) with 20 per-frame `params` (`engine.setMaterialParams`) across WebGPU, WebGL 2, Vulkan, and Metal.
- **Camera Lens Shift & Depth of Field:** Views and `.look.json` cameras accept `up`, off-center `shift` (frame around HUD chrome without rotating perspective), `near`, and `focus: { distance, depth, blur }`.

---

## 4. Stylized Renderer & Native Player (`ganbate-player` & `ganbate-render`)

```sh
# Interactive native Vulkan player (Linux X11/XWayland or Windows Win32; pass --frames 120 for smoke runs)
cargo run --locked --offline --release --bin ganbate-player
cargo run --locked --offline --release --bin ganbate-player -- --scene world.gbew --look look.json

# Offscreen reference sculpture (PPM) or stylized Look render (PNG stills & 24 fps time-lapses)
cargo run --locked --offline --release --example observatory -- scene.ppm
cargo run --locked --offline --release --bin ganbate-render -- \
  --scene world.gbew --look look.json --all-views --time 20.5 --weather clear --width 1920 --height 1080 --samples 16 --out-dir renders
cargo run --locked --offline --release --bin ganbate-render -- \
  --scene world.gbew --look look.json --sequence 18.0 22.0 96 --out-dir timelapse
```

`ganbate_engine::render` is driven entirely by JSON **Look** files: soft cel lighting + brushwork, 2 shadow cascades, tiled point lights, planar water reflections with shallows/foam/glitter, astronomical sun/moon sky, 7 weather presets (`clear`, `cloudy`, `overcast`, `mist`, `rain`, `storm`, `petals` occluded by roofs), wind-animated grass/flowers, SSAO, outlines, aerial perspective, light shafts, bloom, grading, sharpening, and procedural surface detail rules (`roof_tiles`, `wood`, `plaster`, `stone`, `lacquer`, `grass`, `terrain`, `lattice`).

Shader inspection CLI:
```sh
cargo run -p ganbate-shader --example ganbate-shaderc -- shader.wgsl fragment fs_main out.spv
```

---

## 5. Deterministic 60 Hz Physics (`ganbate-physics`)

`crates/ganbate-physics` provides rigid-body simulation with zero dependencies beyond `ganbate-core` ([physics design](docs/design/06-physics.md)):
- **Shapes & Bodies:** Spheres, capsules, boxes, convex hulls, triangle meshes, heightfields; static, kinematic, and dynamic bodies; sleeping islands; CCD for fast-moving objects.
- **Joints, Queries & Characters:** Ball, hinge, slider, fixed, and distance joints (limits, motors, springs); sensors & contact events; `raycast`, `shape_cast`, and `overlap`; kinematic character controller (slides, climbs steps, rejects steep slopes, rides moving platforms).
- **Bit-Identical Determinism:** Steps execute only `+`, `−`, `×`, `÷`, and `sqrt`, producing the exact same 64-bit `state_hash` across native and WASM builds.
- **In Studio & Web:** Author with `rigid_body` and `collider` components; press **`P`** in Studio to simulate at 60 Hz, or call `physicsStart()`, `physicsStep(dt, steps)`, `physicsStop()`, and `physicsHash()` on the web.

```sh
cargo run --release --bin ganbate-build -- simulate scene.gbew 120
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo run --release --example physics_demo -- physics.png
```

---

## 6. AI Agent CLI (`ganbate`)

`crates/ganbate-agent` (`ganbate`) lets AI agents and automation inspect, validate, edit, diff, undo, and render scenes without touching engine source or corrupting canonical JSON. Every command outputs a single JSON line (`{"ok":true,...}` or `{"ok":false,"error":{"code":"...","message":"..."}}`).

```sh
cargo run --release --example studio_sample -- ~/ganbate-sample
cargo build --release -p ganbate-agent --bin ganbate && export PATH="$PWD/target/release:$PATH"
P=~/ganbate-sample S=content/scenes/blocks.scene.json

ganbate scene inspect "$P" "$S" --entity "Tower/Tower top" --defaults
ganbate scene validate "$P" "$S"
ganbate scene apply "$P" "$S" --dry-run - <<'EOF'
{"label": "Brighter lamp, a bench", "edits": [
  {"op": "set", "entity": "Lamp", "path": "components.light.intensity", "value": 30},
  {"op": "create", "name": "Bench", "parent": "Tower", "components": {"transform": {"translation": [1, 0, 0]}}}
]}
EOF
ganbate render "$P" "$S" --out blocks.png --size 960x540 --samples 4
git show "HEAD:$S" | ganbate scene diff - "$P/$S"
ganbate scene undo "$P" "$S"
```

- **Undo & Safety:** Written transactions log inverse ops and pre/post file hashes in `PROJECT/.ganbate/history/<scene-id>.jsonl`. Both Ganbate Studio and `ganbate scene undo` refuse to overwrite concurrent disk changes (`changed-since`); save in Studio before running `ganbate`, then reload the Studio page.

---

## 7. Apple Metal (macOS & iOS)

`ganbate-shader` emits Metal Shading Language (`compile_msl`) matching WGSL buffer layouts byte-for-byte, and `src/metal` binds Metal directly through the Objective-C runtime (`aarch64-apple-darwin`, `aarch64-apple-ios`, `aarch64-apple-ios-sim`). *(Status: compiled and MSL-validated against clang; hardware device qualification pending.)*

```sh
# macOS (requires Xcode Command Line Tools)
scripts/apple/build-mac.sh --shaders --test
target/aarch64-apple-darwin/release/ganbate-player --scene world.gbew --look look.json

# iOS Simulator & Device (requires full Xcode + Apple Development team)
scripts/apple/build-ios.sh --sim --world world.gbew --look look.json
GANBATE_IOS_TEAM=ABCDE12345 scripts/apple/build-ios.sh --device --world world.gbew --look look.json
```

*(Use `MTL_DEBUG_LAYER=1` for Metal validation and `GANBATE_METAL_DEVICE` to select a Mac GPU. See `scripts/apple/build-ios.sh` for provisioning profile details.)*

---

## 8. Verify & Benchmark

```sh
# Unit, Vulkan, presentation, renderer, agent, browser & Studio end-to-end suites
cargo test --workspace --locked --offline
cargo test --locked --offline --test vulkan -- --ignored
cargo test --locked --offline --test present -- --ignored
cargo test --locked --offline --test gpu --test render -- --ignored
cargo test --locked --offline -p ganbate-agent --test binary -- --ignored
node editor/test/run.cjs --out editor-test
node editor/test/run.cjs --api webgl2 --out editor-test-gl

# Cross-backend PSNR oracle (WebGPU/SwiftShader & WebGL 2/ANGLE vs. native Vulkan/llvmpipe)
cargo run --release --example synthetic_world -- scenes
cargo build --release --bin ganbate-render
node web/test/run.cjs --dist web/dist --out web-test --ganbate-render target/release/ganbate-render \
  --case name=synthetic,scene=scenes/synthetic.gbew,look=scenes/synthetic-look.json,view=test,time=17.5,weather=clear,size=640x360

# Throughput benchmark & real-device 120 Hz / 240 Hz displayed-frame gate
cargo run --release --locked --offline --bin ganbate-bench -- --scene scene.gbew --width 1920 --height 1080 --warmup 40 --frames 240
cargo run --release --locked --offline --bin ganbate-framecheck -- 120 1200 presented-ms.txt
```

Vulkan ABI declarations are adapted from [Khronos Vulkan-Headers v1.3.275](https://github.com/KhronosGroup/Vulkan-Headers/tree/v1.3.275) under [docs/Vulkan-Registry-LICENSE.txt](docs/Vulkan-Registry-LICENSE.txt).
