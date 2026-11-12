# Boundary: what is the engine's and what is the game's

**Decision.** A thing belongs to the engine when another game could use it unchanged. It belongs to the game when
it encodes that game's rules, policy or identity. Any number someone will want to tune is data, edited in the
editor, whichever side reads it. The engine repository never contains a game's names, assets, protocols or rules.

## 1. Three questions

Ask in order:

1. **Would a different game use it as is?** A matrix library, a GLB importer, a particle system, spring bones,
   text rendering, a camera rig: yes, engine. A card game's dealing order, a game's chat policy: no, game.
2. **Is it mechanism or policy?** "Blend two animation clips over 0.3 s" is mechanism (engine). "When the player
   wins, the dealer smiles" is policy (game). "Hide what this player may not see" is policy that must stay with
   the game's server and client, never inside a renderer.
3. **Is it a number or a choice someone will tune?** Camera framings, light positions and colours, material
   colours, layout offsets, timings, particle counts, sound levels: data. The engine provides the document type
   and its editor; the game provides the values.

Most of a game's hand-written presentation code splits three ways along these lines: the mechanism moves into the
engine, the numbers become documents, and a small amount of policy stays as game code that drives both.

## 2. Typical outcomes

| In a game today | Goes to | As |
| --- | --- | --- |
| Matrix, quaternion and node-hierarchy helpers | engine | `ganbate-core` math, scene hierarchy |
| Fixing up models at load (UVs, normals, merged objects, missing lights) | engine | import settings and import tools; the fix happens once, at import |
| Building meshes in code (cards, discs, rings, ribbons) | engine | procedural mesh builders and prefab assets |
| CPU particle emitters | engine | the VFX system; the game keeps presets as documents |
| Spring or "secondary" motion on a character | engine | spring chains in the animation system; per-character values as data |
| Clip cross-fades, look-at, facial weights | engine | animation graph, IK and look-at, expression sets |
| A frame scheduler and a resolution governor | engine | the frame loop and quality tiers |
| World-space labels, badges, text on objects | engine | text and world UI |
| Screen positions of 3D things for an HTML interface | engine | anchors: named points the runtime projects every frame |
| Shaders written for one effect | engine library, or the game's own prop shader | materials, prop shaders |
| Camera profiles per screen shape, light rigs, colours | data | camera entities, light entities, materials, the Look |
| Layout constants (where seats, cards, pieces go) | data | anchor and marker entities in a prefab, the game's components |
| Timings and curves of a choreographed moment | data | sequences with cue tracks |
| Which moment plays when (game events to presentation) | game | game code that triggers sequences and sets component values |
| Rules, hidden information, settlement, networking | game | game code and the game's server; the engine never sees hidden state |
| Menus, forms, account screens | game | the game's UI (the DOM on the web) |

## 3. Extension points

A game extends Ganbate in six ways and never by editing engine code:

| Extension | Lives in | Example |
| --- | --- | --- |
| Component schemas | `schemas/*.schema.json` | a spawn point, a seat, a trigger volume, with fields the inspector draws |
| Game code | the game's repository | reads its components from the cooked entity table, drives entities, plays sequences, answers its server |
| Editor plugins | `editor/*.js`, listed in the manifest | a custom inspector, a viewport guide, "preview this game state" |
| Asset kinds | a schema for a new document type plus a plugin | a dialogue file, a table layout |
| Prop shaders | WGSL in the game's materials | a card's sheen, a hologram |
| Looks | `*.look.json` | the whole style of the game |

Runtime access to authored data goes through the cooked scene: the entity table names every entity, its
instances and lamps, and carries every component the renderer does not draw. A web game reads it with
`engine.worldEntities()`; a native game reads the same table through `ganbate_assets::cooked::decode_scene`.

## 4. Contracts across the boundary

- **Versioned APIs.** The Rust crates and the schema format are versioned; the web library (`web/sdk/ganbate.js`,
  `web/sdk/ganbate.d.ts`) and the edit protocol will be in M2 (a `version` constant the library exports, a `protocol` number
  in `hello` and `welcome`). Additive changes keep old games working; a breaking change bumps the version and ships
  a migration note.
- **A game pins the engine.** A web game vendors a built runtime with its engine revision; a native game pins the
  crates. Upgrading is a reviewed change in the game's repository, rolled out per platform.
- **Fixtures stay with the game.** A game's tests run its own fixtures through its own adapters against the
  pinned engine. The engine's tests use synthetic, genre-neutral content only.
- **Reversible migrations.** Moving a game from another renderer onto Ganbate happens per surface and per platform
  behind a switch, and each step can be switched back until the old path is deleted.

## 5. Sorting a game's existing code

1. **Inventory** what the old renderer draws, file by file, and what the Ganbate path already matches.
2. **Classify** each module with the three questions; mark mixed modules section by section.
3. **Extract mechanisms** into the engine one at a time, each with synthetic tests in this repository, and delete
   the game's copy once its stage uses the engine's.
4. **Convert numbers to documents** as the editor milestone for that kind of data lands, so the values move into
   files the game reads and the editor edits.
5. **Keep policy** as small game code that triggers sequences, sets component values and answers the server.
6. **Delete** the old renderer's code per surface after its switch has stayed on through a release without
   being switched back.

A game keeps its classification and migration plan in its own repository; the engine's design only fixes the
rules above.
