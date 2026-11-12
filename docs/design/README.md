# Ganbate design

> **New to Ganbate Engine or Ganbate Studio?** Start with the practical **[Developer Onboarding Manual (`../ONBOARDING.md`)](../ONBOARDING.md)** for crate ownership, what runs today, Studio keyboard shortcuts, cooking commands, `createEngine()` code recipes, and the `ganbate` AI-agent CLI. Come back here when you want the architectural rationale behind each subsystem.

**Ganbate is becoming a studio engine: one Rust runtime, one editor (Ganbate Studio) and one content pipeline that
several games share.** A game keeps its rules, its server protocol and its art; the engine owns every mechanism a
second game could use unchanged. People make and change content in the editor, not by asking for code: placement,
terrain, vegetation, lighting, the Look, cameras, materials, characters and game stages.

The first game project moves onto Ganbate completely, and its old renderer is deleted once Ganbate matches it on every
platform. Long term the same engine carries a cross-platform open-world game, consoles included.

This folder is the design. It is written to be read in order; each page starts with what it decides.

| Page | Decides |
| --- | --- |
| [1. Studio workflow](01-studio-workflow.md) | How a AAA studio's roles and pipeline (DCC → import → editor → cook → runtime) map onto Ganbate, and how engine and game code stay apart |
| [2. Runtime](02-runtime.md) | Every runtime subsystem: what exists today (with evidence), why each is needed, and its priority |
| [3. Editor](03-editor.md) | Ganbate Studio's shape (a local Rust host plus a browser interface whose viewport is the engine), panels, tools and shared machinery |
| [4. Data](04-data.md) | Projects, documents (scene, prefab, schema, material, Look...), ids, the cook, platform packages and immutable publication |
| [5. Boundary](05-boundary.md) | What belongs to the engine and what to a game, the extension points, and how a game's current code is sorted |
| [6. Physics](06-physics.md) | Rigid bodies, collision, the soft-step solver, determinism, queries, characters, and physics in the editor |
| [Roadmap](../roadmap.md) | Milestones that interleave "replace the game's old renderer" with "build the editors", each with a shippable result and acceptance tests |

## The decisions in one screen

1. **Objects stay objects.** Importing never merges a model's objects; a scene keeps one entity per object with a
   permanent id. The cook keeps that identity in the cooked world's entity table, and the renderer groups draws by
   material at load, so separate objects cost no extra draw calls. Moving one tree is a gizmo drag, not a script.
2. **Content is text, the runtime reads binary.** Scenes, prefabs, schemas, materials, Looks and sequences are
   canonical JSON that diffs and merges line by line; the cook turns them into per-platform packages named by
   their content hash. A published file never changes.
3. **The editor is a local host plus a browser interface.** `ganbate-studio` owns the project folder and the undo
   history and speaks one edit protocol to the interface, to a running game and to a phone. The viewport is the
   engine's own web runtime, so what the editor shows is what a web player sees with the same graphics API (a phone
   previewing over plain HTTP on the LAN draws with WebGL 2, not WebGPU); a native viewport client joins
   later for open worlds and consoles.
4. **Data is described by schemas.** Engine components and a game's components share one schema format, so the
   inspector, validation, the cook and runtime readers come from one description. Games extend the engine with
   schemas, code and editor plugins, never by editing engine code.
5. **Everything is ours, platforms excepted.** The engine keeps zero third-party crates and writes its own codecs,
   compiler and tools. Console SDKs, which cannot be avoided or published, live in separate private platform
   modules behind the same interfaces.
6. **Each milestone ships.** Editor milestones convert one slice of a game's hand-coded content into data, and the
   same milestone closes that slice's renderer gaps, so both goals advance together and every step is reversible.

## Status

Milestone 1 (scene and editor foundation), deterministic rigid-body physics, and the structured agent CLI are
implemented in this repository: `ganbate-core` (JSON writer, digests, ids), `ganbate-scene` (documents, schemas,
prefabs, undo), `ganbate-pipeline` (project, asset database, import that keeps objects, scene cook to GBEW v4 with an
entity table), renderer support for moving, hiding and picking world objects, `ganbate-physics` (60 Hz soft-step
solver, joints, queries, character controller), `../ganbate-studio/server/ganbate-studio` (host) plus `../ganbate-studio/client/` (interface + Asset Library),
and `../ganbate-studio/server/ganbate-agent` (`ganbate` JSON CLI), checked end to end by `../ganbate-studio/client/e2e/run.ts` and `cargo test --workspace`.
See the [roadmap](../roadmap.md) for what each later milestone adds.

## Words used here

| Word | Meaning |
| --- | --- |
| Entity | One placeable thing in a scene, with a permanent 64-bit id and components |
| Component | A typed bundle of fields on an entity (`transform`, `mesh`, `light`, or a game's own kind) |
| Schema | The description of a component kind's fields, types, ranges and defaults |
| Prefab | A reusable group of entities; an instance can override fields of what it shows |
| Document | Any authored JSON file the editor opens: scene, prefab, material, Look, sequence |
| Op | One change to a document (`set` or `remove` at a path); every edit is a list of ops with an inverse |
| Cook | Turning documents and sources into what a platform's runtime loads |
| Look | A game's whole visual style as data: shading, post-processing, sky, time, weather, views, detail rules |
| Live link | A running game or phone following the editor's changes as they happen |
| GBEW | The engine's cooked world file (`.gbew`): meshes, materials, textures and lamps; version 4 adds the entity table |
| Entity table | The part of a cooked world that lists each entity's id, name, parent and the instances and lamps it produced, plus the components the renderer does not draw, so game code finds and moves objects at run time |
| Prop | An object a game creates while it runs (a mesh, a material and an instance), drawn and lit like the cooked world; a prop shader is a small game-written shader for one prop's effect |
| Sidecar | The small `<file>.meta.json` beside a source file that holds its permanent id and import settings |
| DCC | A content-creation tool such as Blender, where meshes, rigs, clips and textures are made |
| HLOD | Simplified stand-ins for groups of distant objects, so far areas draw cheaply |
