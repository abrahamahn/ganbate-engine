# Runtime: every subsystem, what exists, why, and when

**Decision.** Ganbate's runtime is one Rust engine for every platform, built in layers: a portable core, platform and
GPU layers, then systems (scene, rendering, animation, audio, physics, UI, VFX, sequencing), then the game's code
on top. The renderer and GPU layer are the strongest parts today; scene identity, the asset pipeline and the edit
loop arrived with milestone 1; animation graphs, audio, text, physics and streaming are the big gaps. Priorities
follow what the first game project needs to leave its old renderer and to build its content in the editor (P0),
then what native apps and a second, non-card game need (P1), then open worlds and consoles (P2).

Status words: **present** (works and is tested), **partial** (works for today's uses; named gaps), **absent**.
Evidence paths are in this repository. "Hardware" means a real GPU; everything recorded so far ran on software
rasterizers (lavapipe/llvmpipe, SwiftShader, ANGLE on llvmpipe) unless a row says otherwise.

## 1. Foundation

| Subsystem | Status | Evidence | Why it is needed | Priority |
| --- | --- | --- | --- | --- |
| Core types and math | partial | `crates/core/src/lib.rs`, `world.rs`, `src/render/math.rs`, `src/anim.rs` | Three vocabularies today (core `Vec3`/`Transform`, render arrays, animation quaternions); one math library with bounds, rays, frusta and quaternions is the base of picking, culling, physics and tools | P0: unify as systems need it |
| Data layer: JSON read and canonical write, digests, ids | present (M1) | `crates/core/src/json.rs`, `digest.rs`, `id.rs` | Every authored file, every content address, every entity and asset reference | P0 |
| Reflection and schemas | present for data (M1) | `crates/scene/src/schema.rs`, `engine.schema.json` | Inspectors, validation, the cook and runtime readers come from one description; games add kinds without engine code. Rust-side typed components with generated schemas come with the gameplay framework | P0 data, P1 Rust types |
| Serialization of runtime state (save games, replays) | absent | | Story progress, local saves, deterministic replays of presentation | P1 |
| Platform layer (windows, lifecycle, files, clocks) | partial | `src/window/*` (X11, Win32, AppKit, UIKit), `src/apple/objc.rs`, `crates/web` | One window per device, no Android, no Wayland, no app lifecycle hooks (background, memory warnings) | P1 (M8): lifecycle and Android |
| Input: keys, pointer, touch, gamepad, text, actions | partial | `src/window/mod.rs:19-43` | Missing modifiers, wheel, right button, gamepad, text input and IME, action mapping. On the web the host page owns input today | P1 (native apps, second game), gamepad P1 |
| Frame loop, timing and pacing | partial | `crates/core/src/timing.rs`, `src/bin/ganbate-framecheck.rs` | Render on demand with per-state cadences (motion at the display's own rate, 120 Hz and faster displays included; then 30, 15, a few and 0 frames per second as less moves), an adaptive resolution governor with device-class floors, visibility throttling, fixed-step simulation, presentation-time feedback for the device frame gates | P0 (M2) |
| Loading policy (priorities, deadlines, fallbacks, preloading) | absent | | Critical content before decoration; a defined fallback when an asset is late or fails | P0 (M2) |
| Job system and threads | absent | (single-threaded by design: `src/vulkan/rhi/mod.rs:14-16`) | Cooking, decoding, streaming, animation and culling off the main thread; wasm threads where the browser allows; a single-thread fallback | P1 (P0 for the editor host's cook) |
| Memory: arenas, pools, budgets | absent | one GPU allocation per resource (`src/vulkan/rhi/resources.rs:66-120`) | Zero per-frame allocation, zero growth over two-hour sessions, per-tier GPU memory budgets | P0 budgets, P1 allocators |
| Logging, crash capture, telemetry | partial | errors as strings; Vulkan validation log | An engine log with levels and sinks (editor console, files), crash breadcrumbs, renderer/backend/adapter/revision in bug reports | P0 (M2) |
| Localization of engine text | absent | | In-scene text in each player's language (Korean, English and Japanese first): string tables, run-time switching, captions | P1 (editor P0; M8) |

## 2. Assets and content

| Subsystem | Status | Evidence | Why | Priority |
| --- | --- | --- | --- | --- |
| Project and asset database (ids, sidecars, index) | present (M1) | `crates/pipeline/src/project.rs` | References survive renames; the editor finds everything; duplicate ids are caught | P0 |
| Importers: glTF static and animated, PNG | partial | `crates/assets/src/glb.rs`, `model.rs`, `png.rs` | External geometry codecs (`KHR_draco_mesh_compression`, `EXT_meshopt_compression`) are intentionally excluded and refused (export uncompressed GLBs and cook with `--compact`); no spot or directional lights; no texture transforms or extra UV sets | P0: import keeps objects (done); P1: remaining glTF attributes |
| Import that keeps objects | present (M1, static models) | `glb::import_nodes`, `ganbate_pipeline::cook::scene_from_model` | One entity per node with name, parent and transform; ends script-driven placement | P0 |
| Cook: scenes to worlds with entity tables | present (M1) | `ganbate_pipeline::cook::cook_scene`, GBEW v4 (`crates/assets/src/cooked.rs`) | The runtime keeps authored identity; objects move, hide and pick at run time | P0 |
| Incremental, cached, parallel cook | absent | | Large projects cook in seconds after a change; the editor re-cooks only what changed | P0 for the editor, P1 at scale |
| Per-platform packages and texture compression | absent | no block formats in `src/gpu/types.rs:47-80` | VRAM and download budgets (a phone room must stay small); BC on desktop, ASTC on mobile, ETC2 fallback, Basis-style transcoding for the web | P0 packages (M2); P1 texture compression |
| Compressed geometry for models | partial | compact GBEW coding (`cooked.rs`, `--compact`) | Run-time models are loaded as plain GLB, about 3.5 times larger than compressed sources; cook models into the compact coding too | P0 (download budget) |
| Content-addressed publication | partial (M1 names) | `ganbate_pipeline::cook::content_name` | Published files never change; rollbacks point at older manifests | P0 (M2) |
| Cook provenance (engine revision, settings, source hashes stamped in packages) | absent | | A cooked world records which engine cooked it, so a runtime can refuse what it cannot read and a team can rebuild exactly | P0 (M2) |
| Import settings for common fix-ups (UV repair, normal treatments, lamps from emissive clusters, unit scale, rotation order) | absent | games fix models in export scripts or at load | Fix-ups run identically every import, with names, instead of in scripts | P0 (M2) |
| Streaming and partial loading | absent | whole-file decode (`cooked.rs`: `decode_scene`) | Next hand and next room stream in the background; open worlds stream cells by distance | P1, P2 for cells |
| Large-world coordinates (origin shifting per region) | absent | | `f32` keeps millimetre precision only within about 10 km of the origin; open worlds rebase rendering and physics per region | P1 (M9) |
| Video playback | absent | | Intro and story movies, video textures on screens | P2 |
| Hot reload | partial (M1: the host broadcasts confirmed ops to every client watching a document and the editor's viewport applies moves live; no runtime live-link client yet, M2) | `../ganbate-studio/server/ganbate-studio`, `../ganbate-studio/client/` | Change a Look, material or placement and see it in the running game; shaders and textures reload without restart | P0 |
| Hostile-input hardening of cooked files on wasm32 | partial | overflow in `padded()` and a 1 GiB reservation (`cooked.rs`: `padded` and the reservation in `decode_mesh`), unchecked accessor ranges (`model.rs:350-395`) | A corrupt download must fail with a message, never abort the page | P0 (M2) |

## 3. Scene

| Subsystem | Status | Evidence | Why | Priority |
| --- | --- | --- | --- | --- |
| Entities, hierarchy, components (authoring) | present (M1) | `crates/scene/src/scene.rs` | Every placeable thing has an id, a parent and typed components | P0 |
| Prefabs with overrides and nesting | present (M1 data), editor UI M3 | `scene::flatten` | Seats, lamps, tables and trees reused with per-instance changes | P0 |
| Runtime entity access for game code | partial (M1) | entity table (`ganbate_core::world::EntityTable`, read with `ganbate_assets::cooked::decode_scene`), `engine.worldEntities()` | Games find their markers and anchors by id or kind instead of constants | P0 |
| Runtime transform hierarchy for props (parents, inherited visibility, batched updates) | absent | games build their own node graphs over flat instances | Every game stage needs it; doing it in the engine removes per-frame allocation and duplicate code | P0 (M3) |
| Runtime entity world (spawn, destroy, query, component storage) | absent | | Gameplay that creates and moves things while it runs, beyond props | P1 |
| Levels, layers and sub-scenes | absent | | A House with optional cutaway layers, per-orientation variants, stage-specific props; open-world cells | P0 layers, P2 cells |
| Spatial queries (raycast, overlap, nearest) | partial | world and prop picking (`Renderer::pick_world`, `Renderer::pick`) | Selection, placement on surfaces, gameplay queries | P0 picking, P1 general |

## 4. Rendering

The renderer is Ganbate's strongest system: one stylized pipeline on Vulkan, Metal, WebGPU and WebGL 2, data-driven by
Look files, with props, overlays, prop shaders and game lights for game-drawn content.

| Subsystem | Status | Evidence | Why | Priority |
| --- | --- | --- | --- | --- |
| GPU layer (RHI) with four backends | present | `src/gpu/*`, `src/vulkan/*`, `src/metal/*`, `src/gpu/webgpu`, `src/gpu/webgl` | One API everywhere; automatic barriers; same misuse rejected on every backend | P0 |
| GPU features: compressed, integer, 3D, MSAA, stencil formats; timestamps; debug labels; multi-surface | absent | `src/gpu/types.rs:47-80` | Texture budgets, GPU profiling, readable captures, multiple editor viewports | P1 (one viewport switched between device shapes in M2) |
| Device loss and context restore | absent | detection only (`src/vulkan/rhi/mod.rs:373-390`) | Phones lose GPU contexts; the game must recover without a reload | P0 |
| Shader compiler (WGSL subset to SPIR-V, MSL, WGSL, GLSL ES) | present | `crates/shader` | One shader source on every backend, compiled at run time even in the browser | P0 |
| Shader modules, specialization, caches, hot reload | absent | sources joined by concatenation; prop-shader errors reported 1,360 lines off | Material editing with good errors, permutations without recompiling, cached pipelines on phones | P0 error positions, P1 the rest |
| Stylized shading, shadows, water, sky, weather, grass, terrain detail, post | present | `src/render/*` | The game's whole look as data | P0 |
| Props, overlays, blend layers, prop shaders, game lights, picking, projection | present | `src/render/props.rs`, `lights.rs` | Everything a game draws at run time | P0 |
| World objects that move, hide and pick | present (M1) | `GpuScene::set_instance`, `Renderer::pick_world` | Editing and run-time changes of cooked scenes; draws grouped by material | P0 |
| Unified scene representation (world and props as one instance system) | partial | ground and world maps draw the cooked world only (`renderer.rs`: `draw_ground`) | Props that grass grows on, terrain as entities, one culling path | P1 |
| Visibility groups and layers of the cooked world | partial (M1 per instance) | `Renderer::set_world_instance` | Whole groups (roofs, exterior, a courtyard) switched per camera stop or quality tier in one call | P0 (M2) |
| Render layers and sub-views (a second camera and viewport over the main image) | absent | games fake close-ups by placing them far below the world under a swapped view | Close-ups, picture-in-picture, portrait renders, the editor's camera previews | P0 (M3) |
| Vertex-stage effects: deformers (bend, curl, sculpt), billboards and sprites | absent | prop shaders are fragment only (`src/render/shaders/prop_shader.wgsl`) | Cards that bend, faces that turn to the camera, halos and particles without CPU work every frame | P0 (M3) |
| Decals, vertex colours and per-instance draw order for blended props | absent | `GanbateMeshData` has no colour stream; order is per material | Soft contact shadows, rims and sweeps on surfaces; cards that sort by deal order | P0 (M3) |
| Image-based light and reflections for props | absent | metallic surfaces have no environment term | Brass, leather and lacquer that read as materials | P1 |
| Per-view grading (exposure, tinted vignette) and timed Look and light blends | partial | `Look` exposure and vignette are scalars | Exposure per screen orientation; a room that dims and warms over half a second at a big moment | P0 (M2, M3) |
| Shader standard library and validation entry for game shaders | partial | prop shaders compile at run time; errors carry engine offsets | sRGB, rounded-rectangle, ring and glow helpers; `ganbate-build check-shader` with the game's own line numbers | P0 (M2) |
| Culling, LOD, HLOD, GPU-driven draws | absent | | Draw-call and triangle budgets at scale (the game's world is already half the triangle cap); open worlds | P1, P2 |
| Heightfield terrain: sculpting, layer painting, holes, water bodies | partial | terrain detail shading by the Look (`src/render/look.rs`); terrain is ordinary meshes today | Landscapes edited in the editor, and terrain the ground map, grass and shallows follow | P0 for the first game's landscape (M5); open-world scale P2 |
| Foliage: instanced vegetation painted or scattered by rules, with LOD | partial | grass from the ground map; trees are ordinary meshes | Forests and gardens placed by brush or rule within a draw and memory budget | P0 for the first game's gardens (M5); open-world scale P2 |
| Material and prefab variants (cosmetics) | absent | | Table themes, card backs, frames and outfits swapped at run time as data, not code | P0 (M3) |
| Many small scenes in one view (thumbnails, multi-table views) | absent | one world, one view | Lists of live tables, tournament overviews, store previews | P0 (M3 with sub-views) |
| Text rendering (fonts, shaping, Hangul and Latin, signed-distance glyphs; outline, ellipsis, icons) | absent | games draw text into canvas textures | Localized in-scene text without canvas round trips; native apps have no canvas | P1 |
| Procedural meshes (box, frustum, sphere, disc, extrude and bevel, strips, rounded rectangles) | absent | only in tests and examples | Prefabs built from parameters; a game's code-built props converted to assets | P0 (M2) |
| Baked 2D art (vector and procedural textures cooked to images) | absent | games paint SVG and canvas art at run time | Card faces, felts and prints cooked once per platform, not painted on every device | P1 |
| Planar and screen-space reflections, refraction | partial | water reflection | Wet stone, lacquer, glass | P2 |
| Accessibility render modes (reduced transparency, high contrast, reduced motion) | absent | | Fully authored low-intensity and high-contrast modes; a reduced-motion variant of every sequence (M3) | P0 reduced motion (M3), P1 the rest |
| Render on demand and quality tiers | partial | `Quality::still/realtime`; the host decides when to draw | Battery and thermal: draw only what changes; an adaptive resolution governor | P0 |
| Still capture (thumbnails, backdrops, store art) | present | `Renderer::render_still`, `ganbate-render` | Backdrops and art rendered from the live scene | P0 |

## 5. Animation and characters

| Subsystem | Status | Evidence | Why | Priority |
| --- | --- | --- | --- | --- |
| Skinned models, GPU skinning, morph targets, clips | present | `src/anim.rs`, `src/render/models.rs` | Characters, dealers, avatars | P0 |
| Pose mixing by weight, node overrides, sockets (node world transforms) | present | `anim::Pose`, `modelNodeWorld` | Cross-fades and held objects (cards in a hand) | P0 |
| Animation graph: state machines, blend trees, layers, masks | absent | games hand-write cross-fades | Characters authored as data instead of script code; reuse across characters | P0 (M6) |
| Look-at, two-bone IK, foot and hand placement | absent | | Gaze toward players, hands on objects, sitting characters | P0 (M6) |
| Spring bones and secondary motion | absent | games simulate in their own code | Hair, cloth panels, accessories, body secondary motion as data per character | P0 (M6) |
| Facial expression sets and presets | partial | morph weights per node | Composable facial controls (blink, gaze, brows, mouth) and named expressions | P0 (M6) |
| Retargeting between skeletons | absent | | One motion library across a roster of characters | P1 |
| Animation events and cues, root motion, seeking | partial | clips sample at any time; no events | Sequenced beats, sound on contact, deterministic scrubbing in the editor | P0 (M3) |
| Bulk pose input and output, pose snapshots, socket queries at a clip time | absent | one wasm call per joint per frame | Blending from wherever a character is now; where a held object will be at a release time | P0 (M3) |
| Model validation and scaling (shared skins, one draw per skinned instance today) | partial | no `Model::validate`; each skinned instance duplicated and drawn alone | A roster of seated characters within draw budgets; hostile models fail cleanly | P0 (M6) |
| Cloth and hair simulation | absent | | Authored or spring-based in the first game; simulation for later games | P2 |
| Character assembly (body, parts, outfits, morph sliders) | absent | | Avatar customization and cosmetics; one identity across 3D and portrait | P0 (M6) |
| Animation compression and cooked clips | absent | raw f32 keys | Download and memory budgets for large clip libraries | P1 |

## 6. Simulation, audio, input, UI, effects

| Subsystem | Status | Evidence | Why | Priority |
| --- | --- | --- | --- | --- |
| Physics: collision shapes, broad and narrow phase, rigid bodies, joints, queries, character controller | present | `crates/physics` ([physics](06-physics.md)) | A second, non-card game with movement and collision; open worlds; props that fall, roll and swing in any game | P1 (built ahead of M9) |
| Raycasts and overlaps against scene geometry | present | the renderer's picking against triangles; the physics world's `raycast`, `shape_cast`, `overlap` and `contacts_with` (`crates/physics/src/query.rs`) against its colliders | Placement in the editor, gameplay queries | P0 |
| Audio engine: buses, mixing, voices, scheduling, streaming, 3D panning | absent | the game's web audio lives in the game | Native apps have no Web Audio; one event-keyed audio system with buses, voice limits per family, music arrangements, ambience beds and ducking for every game | P0 (M7) |
| Haptics | absent | | Turn and table-feel channels on phones | P1 |
| Input actions and rebinding | absent | | Keyboard, touch and gamepad mapped to actions; gamepad focus for native and console | P1 |
| UI: world-space text and badges, anchors to an outside interface | partial | overlays and `project`; games build text textures | World UI from the engine; anchors the DOM or a native UI can follow | P0 anchors (M3), P1 text |
| UI: full in-engine interface with accessibility bridges | absent | | Native apps and consoles need menus without a browser, exposed to each platform's accessibility API | P1 native, P2 console |
| VFX: particles (CPU first, GPU later), trails, flashes | absent in engine | games run their own CPU particles on props | One particle system with presets edited in the editor | P0 (M7) |
| Decals and contact shadows | partial | game lights cast prop shadows | Soft contact shadows under objects | P1 |

## 7. Gameplay, sequencing, AI

| Subsystem | Status | Evidence | Why | Priority |
| --- | --- | --- | --- | --- |
| Sequencer: tracks, keys, curves, cue tracks, camera cuts and blends | absent | games hard-code timings | Choreographed moments with many channels (motion, camera, light, VFX, sound, haptics, interface hints) authored and scrubbed in the editor | P0 (M3) |
| Camera system: rigs, profiles, blends, shake, framing for screen shapes | partial | `View` with lens shift and focus | Camera stops and flights, per-orientation framing, lean-ins at big beats | P0 (M2) |
| Gameplay framework: game-module lifecycle, events, timers, state machines as data | absent | | Game code that reacts to its server and drives presentation without owning engine internals | P1 |
| Scripting | absent (by choice) | | Data (sequences, state machines, graphs) first; a sandboxed scripting runtime only when designers need logic that data cannot express | P2 |
| Navigation and AI | absent | | A second game with movement; ambient characters in larger worlds | P1 navigation, P2 crowds |
| Dialogue and narrative | absent | | Lines as data (speaker, text per language, voice, timing) played from sequences with voice and captions on one clock (engine, M7); the story, its branching and the facts it reacts to stay the game's | P1 (M7) |
| Networking | out of scope for game protocols | games own protocols and authority | The engine exposes clocks, snapshots and replay hooks and, for games that want it, a transport (reliable and unreliable messages over WebSocket, WebTransport and UDP) without game semantics | P1 transport (M9) |

## 8. Tools and diagnostics

| Subsystem | Status | Evidence | Why | Priority |
| --- | --- | --- | --- | --- |
| Editor host and interface | present (M1) | `../ganbate-studio/server/ganbate-studio`, `../ganbate-studio/client/` | All content work in the editor | P0 |
| Profiler: CPU and GPU scopes, counters, frame traces | absent | legacy benchmark only (`src/vulkan/bench.rs`) | Frame budgets on devices; find the slow pass | P0 (M2: engine stats) |
| Debug draw (lines, boxes, labels in the viewport) | partial | the editor's 2D overlay | Visualize bounds, anchors, camera paths, colliders | P0 (M2) |
| Frame pacing gate | present | `ganbate-framecheck`, `timing::Pacing` | The device frame-rate gate for switching renderers | P0 |
| Regression oracle (offscreen stills, PSNR against native) | present | `tests/render.rs`, `web/test/run.cjs` | Same pixels on every backend | P0 |

## 9. Platforms

| Platform | Status | Evidence | Next step | Priority |
| --- | --- | --- | --- | --- |
| Web (WebGPU, WebGL 2) | partial | `crates/web`, `web/sdk/ganbate.js`; headless Chromium only | Real devices: iPhone Safari 26, Android Chrome, desktop browsers; device loss; frame traces | P0 |
| Linux (Vulkan, X11) | partial | presented under Xvfb on lavapipe | A hardware GPU run; Wayland | P1 |
| Windows (Vulkan, Win32) | partial | cross-built; Wine only | A native hardware run; D3D12 is a separate decision | P1 |
| macOS (Metal, AppKit) | partial | compiled, never run | First hardware run | P1 |
| iOS (Metal, UIKit) | partial | compiled, never run | First device run; app lifecycle; native app shell | P1 (M8) |
| Android (Vulkan) | absent | offscreen compile only | Window, lifecycle, input, packaging | P1 (M8) |
| Consoles | absent | | Private platform modules (SDKs under NDA cannot enter a public repository or this zero-dependency rule): D3D12/GDK, AGC/PSSL, NVN; certification requirements | P2 |
| App and console services (accounts, saves, achievements, stores, entitlements) | absent | | Native apps and consoles require their platform's services | P1 apps (M8), P2 consoles (M10) |

## 10. The order this implies

1. **Now (milestones 1-4; P0 to leave the old renderer):** scene identity and the editor loop (done), Look, light, material and camera
   editing with live link, sequences and anchors, device loss, render on demand, profiler scopes, cooked model
   compression, the remaining parity gaps of the first game, and device qualification.
2. **Next (milestones 5-7; P0 to build the first game's content in the editor):** landscape and vegetation;
   animation graphs, look-at and IK, spring bones, character assembly; the audio engine and the particle system.
3. **Then (P1):** native app shells on iOS and Android, input actions, text rendering, texture compression,
   streaming, jobs and navigation for a second game (its physics came early: rigid bodies, joints, queries and a
   character controller are done).
4. **Later (P2):** open-world cells and HLOD, GPU-driven rendering, console platform modules, scripting.

The [roadmap](../roadmap.md) turns this into milestones with acceptance tests.
