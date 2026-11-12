# Engine roadmap

**Two goals advance together.** A first game project leaves its old web renderer for Ganbate completely, and all of
its content work moves into Ganbate Studio. Each editor milestone turns one slice of the game's hand-written content
into documents *and* closes that slice's remaining gaps on Ganbate, so the parity work is done once, in data, not twice.
The game decides when its old renderer stops taking new visual work (the first game proposes milestone 2); the old
renderer is switched off per surface and platform when Ganbate passes the device gates, and deleted after a release
without a switch-back. Every milestone ships something usable
and ends with tests that prove it. The [design](design/README.md) explains the parts.

| Milestone | Ships | Status |
| --- | --- | --- |
| M0 Foundation | Stylized renderer on Vulkan, Metal, WebGPU and WebGL 2; Look files; props, overlays, prop shaders, game lights; animated glTF models; web library mode | done |
| M1 Scene and editor foundation | Scene and prefab documents, schemas, undo; asset database; import that keeps objects; cook with entity tables; world objects that move, hide and pick; Ganbate Studio host and interface | done |
| M2 Look, light, material and camera | Editing the style and lighting live on a running game and a phone | next |
| M3 Game stages and sequences | Game components, plugins, anchors, prefab overrides, sequencer, play in editor | |
| M4 Parity and switch | The first game fully on Ganbate per surface and platform; old renderer deleted | |
| M5 Landscape and vegetation | Terrain sculpt and paint, foliage, water bodies | |
| M6 Characters | Animation graphs, look-at and IK, spring bones, expressions, character creator | |
| M7 Effects and audio | Particle system and editor; audio engine with buses, banks and mixes | |
| M8 Native apps | iOS and Android app shells, text, native interface with accessibility, haptics | |
| M9 Prove reuse and open worlds | A second, non-card game with movement and collision; streaming cells, HLOD, jobs | |
| Physics (ahead of M9) | Rigid bodies, colliders, joints, queries, a character controller; Simulate in Ganbate Studio; native and browser state-hash checks | done |
| M10 Consoles | Private platform modules and certification work | |

The six milestones this roadmap replaces map onto it: *Player* is M0 (native presentation on Linux; Windows
cross-built), *Assets and world* is M1, *Animation and replay* is M3 (cues, seeking) and M6, *Editor and builder*
is M1 to M3, *Platform qualification* is M4 and M8, *Prove reuse* is M9.

## M1 Scene and editor foundation (done)

- `ganbate-core`: canonical JSON writer, SHA-256, ids. `ganbate-scene`: documents with ops and undo, schemas, scenes,
  prefabs with overrides, order keys. `ganbate-pipeline`: projects, sidecars, asset index, object-keeping import, scene
  cook to GBEW v4 with an entity table. Renderer: draws grouped by material; world instances move, hide and pick;
  lamps move. Web runtime: `worldEntities`, `setWorldInstance`, `pickWorld`, `setWorldLight`. `ganbate-studio` host and
  the `editor/` interface. `ganbate-build cook-scene` and `import`.
- **Acceptance (met):** on the synthetic sample (`examples/studio_sample.rs`), in headless Chromium on WebGPU and
  WebGL 2, `editor/test/run.cjs` clicks an object in the viewport to select it, drags it with the gizmo, undoes and
  redoes with the keyboard, moves a parent and sees its child follow, renames through the inspector, adds a
  component (re-cooking the scene), saves canonical JSON, reloads and finds the move, then cooks the saved scene
  with `ganbate-build` and renders it with `ganbate-render`. The driver test
  `world_instances_move_hide_and_pick_like_a_recooked_world` shows a moved world draws the same pixels as one
  cooked that way.

## Physics verification (2026-11-03)

The Linux x86-64 run used Rust 1.94.0 on Ubuntu 24.04 under WSL2. All 698 default workspace tests passed,
including all 439 physics tests. The 43 tests that require a driver or display ran separately in release mode.
Mesa lavapipe with Vulkan validation passed 41 driver tests and the agent command's PNG render test.
The window presentation test passed under Xvfb, including resizing, hiding and restoring the window.
The shader tests passed with `glslangValidator` and `spirv-val` available.

The physics regressions include a turntable spun for ten simulated minutes, a thin rod hinged across its width,
a closed mesh room hit from 96 starting throws, and a rider separated from a platform by a 5 mm teleport.
A resting hexagonal prism kept the same four contact IDs for 300 settled steps at each of three positions on a
triangle mesh. The native `physics_demo` rendered the ball striking its pyramid of 15 blocks at five points
over three simulated seconds (final state hash `664d75dcfa0fa7de`).

Ganbate Studio passed all 50 end-to-end checks on each of WebGPU and WebGL 2 in headless Chromium 156.
The saved scene produced the same state hash, `c9df1e2e6456329b`, on both graphics backends and in the native
`ganbate-build simulate` command after 120 steps. The first browser call requested 2³² steps and exercised
the 64-step cap. Both editor runs included overlapping scene loads and a host disconnect and reconnect.
The web renderer also passed 35 API checks on each backend. For the synthetic scene, the images matched
the native Vulkan reference at 47.7 dB PSNR on WebGPU (SwiftShader) and 51.6 dB on WebGL 2 (ANGLE on llvmpipe).

The workspace passed formatting and native Clippy. The WebAssembly build passed Clippy. Windows x86-64,
macOS arm64, iOS arm64 and Android arm64 passed Rust compile checks. Physical-device qualification remains
part of M4 and M8.

## M2 Look, light, material and camera

The Studio can create, edit, undo and save Look assets while previewing them in the scene. Its Outliner can save a viewport as a named camera or update an existing shot. The [editor design](design/03-editor.md) describes both workflows. The remaining M2 work includes material documents, screen-shape profiles, live linking and packaging.

- Look editor with every Look field in the inspector, a time-of-day and weather scrubber, view bookmarks.
- Light entities edited live (point and spot, follow-the-lamps, shadows); material documents and a material editor
  (colours, textures, shading model, alpha, detail preset, prop shader parameters).
- Camera entities and camera profiles per screen shape (portrait, landscape, desktop), with interface safe areas
  drawn in the viewport.
- Live link: a running web game and a phone on the network follow the editor's ops (the host's `--lan` and its pairing token exist since M1; M2 adds the live-link client in
  the web runtime and the phone preview page);
  structural changes cook incrementally and hot-reload. Browsers give WebGPU only to secure pages and refuse `ws://`
  from `https://` pages, so a phone opens the studio's own preview page over the network (WebGL 2 there) and a
  game served over HTTPS follows edits through a relay on the game's own server (see the editor design).
- Engine pieces the game slice needs: visibility groups (layers) of the cooked world; per-view grading; procedural
  mesh builders; import settings for the usual fix-ups; a shader standard
  library and `ganbate-build check-shader`; cook provenance (engine revision and source hashes in every package);
  compressed cooked geometry for worlds and models (the first game's download budget needs it); Look rules for
  characters (cel ramps, rim and sheen in the model's rest space); clearcoat and glass approximated as Look rules
  (lacquer, chip edges).
- Measuring and debugging: engine stats (GPU time, draws, triangles, memory) in the web library and natively, which
  the device gates of M4 read; a debug-draw layer (lines, boxes, labels) in the viewport; an engine log with levels
  and sinks shown in the editor's console; a headless engine for games' unit tests; hardened decoding of cooked files
  on wasm32 (every count bounded by the bytes that remain).
- Device-loss and context-restore recovery in the web runtime. Autosave of unsaved documents; validation on save
  and of references (parents, prefabs, assets); `ganbate-build check` for commits and CI; versions on the web library
  and the edit protocol; re-scanning the project while the host runs. Render on demand with per-state cadences, an adaptive resolution governor with device-class floors, and a
  loading policy with priorities and fallbacks in the web library.
- Packaging and publishing: `ganbate-build cook-project` turns a project into one package per platform (the manifest's
  `platforms` settings take effect), and `ganbate-build publish` writes content-addressed files and an append-only list.
  Look files gain the `ganbate`/`version` header and a sidecar id.
- Agents: the `ganbate` command sends its transactions through a running host (live, undoable, never over unsaved
  work), and `ganbate benchmark` reports a scene's engine stats from a camera for a device profile.
- **Game slice:** the game's world, light rigs, camera stops and flights, and material colours come from documents;
  per-stop cutaways are entity layers. The game decides whether its old renderer stops taking new visual work here
  (the first game proposes so).
- **Acceptance:** change the Look's evening palette, a lamp's colour, a material and a camera framing in the editor
  and see each on a running game in a browser and on a phone without reloading; publish; the game loads the
  published package; switching the device back to the old renderer still works.

## M3 Game stages and sequences

- Game component schemas and editor plugins loaded from the project; prefab override UI (marks, revert, apply).
- Anchors: named points the runtime projects every frame for an outside interface (DOM today, native later).
- Sequencer: tracks for transforms, camera cuts and blends, clips, light and material parameters, cue tracks
  (sound, particles, game events, haptics, interface hints); scrubbing in the viewport; the same pose at the same
  time in the editor and in the runtime.
- Play in editor: the game's own page runs inside the editor on unsaved documents, with fixtures the game's plugin
  provides.
- Animation events, clip seeking without duplicate cues, bulk pose input and output, pose snapshots and socket
  queries at a clip time.
- Runtime pieces stages need: a transform hierarchy for props in the web library, render layers and sub-views with
  their own camera (also many small scenes in one view: thumbnails, a multi-table view), vertex-stage deformers and
  billboards, decals and vertex colours, per-instance draw order, projection through a chosen view (a camera's rest
  pose), timed Look and light parameter blends; picking that returns every hit under a point, not only the nearest;
  an overlap check in the editor that lists objects intersecting each other.
- Agent proposals: a transaction an agent proposes appears as a ghost preview with its changed fields, and a
  person accepts all or part of it. A measured comparison, on a fixed set of scene tasks, of agents editing through
  the `ganbate` command against agents editing code: tokens, retries, time and regressions per accepted change.
- Material and prefab variants swapped at run time (cosmetics: table themes, card backs, frames) as data; a
  reduced-motion variant for every sequence, chosen by the player's setting.
- **Game slice:** table layouts per seat count and orientation, choreography timings, camera emphasis, table
  pieces and the remaining table-game stages built from prefabs and sequences.
- **Acceptance:** change a seat anchor and a choreography timing in the editor, play the game's fixture in PIE, and
  see the outside interface follow the anchors; a saved sequence scrubbed to time t gives the same pose in the
  editor and in a packaged player.

## M4 Parity and switch

- Close every remaining gap the game's surfaces need on Ganbate (each listed in the game's migration plan).
- Device qualification per platform: sustained frame-rate gates with `ganbate-framecheck` on real displays, two-hour
  memory soaks, load-time and download budgets, input latency. Budgets become data: per-platform profiles in the
  project manifest (frame time, draws, memory, download size) that the cook and the gates check.
- Switch the default renderer per surface and platform behind the game's flag, observe a release, then delete the
  old renderer's code per surface.
- **Acceptance:** recorded device evidence for each platform switched; the old renderer's package is gone from the
  game's dependencies; switching back is no longer needed for a full release cycle.

## M5 Landscape and vegetation

- Heightfield terrain entities: sculpt (raise, lower, smooth, flatten, noise), layer paint, holes; coast and water
  bodies; terrain in the ground and world maps so grass and shallows follow it.
- Foliage: instanced vegetation painted or scattered by rules (density, slope, height, spacing, scale), with LOD.
- One render scene for world and props, so any entity can take part in the ground map.
- **Acceptance:** sculpt a hill, paint a path and plant a grove in the editor; save; the cooked scene draws the
  same; foliage stays within its draw and memory budget at the target view.

## M6 Characters

- Animation graphs (states, transitions, blend spaces, layers, masks), look-at and two-bone IK, spring chains,
  facial expression sets, retargeting across a roster.
- Character assembly and a character creator: body, parts, morph sliders, colours, outfits; presets as documents;
  one identity across 3D, portrait renders and reduced forms.
- **Acceptance:** build a character's idle, gaze and reaction states in the editor without code; customize a
  roster character and render its portrait from the same document.

## M7 Effects and audio

- Particle system (CPU first, GPU where available) with an effect editor and presets.
- Audio engine: buses, voice limits per family, scheduling on an audio clock, music arrangements that persist
  across scenes, ambience beds with crossfades and ducking, positional panning; a Web Audio backend and native
  backends; sound banks and mixes edited in the editor.
- Dialogue: lines as data (speaker, text per language, voice clip, timing) played from sequences, with voice and
  captions on the same clock; the game owns the story and its branching.
- **Acceptance:** an effect and its sound authored in the editor play in sync from a sequence cue on web and native.

## M8 Native apps

- iOS and Android app shells around the native runtime (Metal, Vulkan), app lifecycle, haptics, native audio.
- Text rendering (fonts, shaping; Latin, Hangul and Japanese first) and an in-engine interface layer with platform
  accessibility bridges, for screens that cannot be DOM; string tables and switching language at run time;
  captions.
- **Acceptance:** the first game's table plays in a native app on an iPhone and an Android phone and passes the
  same device gates as the web build.

## M9 Prove reuse and open worlds

- A second sample game, not a card game, with movement and collision: a runtime entity world and gameplay framework
  (spawning, components at run time, game-loop hooks), navigation, input actions and gamepads, a network transport for
  multiplayer. Physics and the character controller were built ahead of this milestone (see
  [physics](design/06-physics.md)).
- Streaming cells, HLOD, large-world coordinates (origin shifting per region), a job system, a semantic three-way
  merge driver for scene files.
- **Acceptance:** two unrelated games built from the same engine revision with no game names in the engine API;
  a roaming sample crosses cells without holes or unbounded memory growth.

## M10 Consoles

- Platform modules kept outside the public repository (D3D12 and the console SDKs), the same RHI and platform
  interfaces, platform services (accounts, saves, achievements, stores), and certification requirements.
- **Acceptance:** the second sample game runs on a console devkit from the same content.
