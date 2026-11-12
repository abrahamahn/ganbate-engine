# Studio workflow: how Ganbate works the way large studios do, at a small studio's size

**Decision.** Ganbate copies what makes large studios productive — tools as part of the engine, data instead of code
for anything tunable, a pipeline from content tools to a cooked runtime with live feedback, and a hard line between
engine and game — and leaves out what only pays at a large studio's size. The editor is where every role that is
not engine programming works.

## 1. Roles, and who fills them here

A large team splits the work across many specialists. Ganbate keeps the split as *kinds of work*, because each kind
needs its own tool, even when one person or an agent does several.

| Kind of work | In a large studio | In Ganbate |
| --- | --- | --- |
| Engine programming | engine, graphics, platform, audio, physics, network programmers | agents writing Rust in this repository, against synthetic tests |
| Tools programming | a tools team building editors and pipelines | the same, in `../ganbate-studio` (`server/ganbate-studio` + `client/`) and `ganbate-pipeline` |
| Gameplay programming | gameplay engineers in the game's code | agents in the game's repository, through the engine's extension points |
| Art direction, environment and level art | art director, environment artists, level designers | the founder in the editor: placement, landscape, vegetation, lighting, Look |
| Character art and animation | character artists, riggers, animators | modelling in a content tool; assembly, animation graphs, expressions and customization in the editor |
| Technical art | tech artists: shaders, import rules, budgets | material and Look documents, import settings, prop shaders, budget reports |
| Game and narrative design | systems, level and narrative designers | the game's component schemas, sequences and stage plugins in the editor |
| Audio | sound designers, composers | audio tools for the sounds; sound banks, cues and mixes in the editor |
| QA and release | testers, build engineers | the engine's regression oracle and device evidence; the game's release pipeline |

The test for every tool: could someone who does that kind of work use it without asking a programmer for a code
change? Where the answer is no, the tool is missing, not the person.

## 2. The pipeline

```text
 content tools           import                editor assembly          cook                    runtime
 (Blender, image,  -->  sidecar settings,  -->  scenes, prefabs,   -->  per platform,     -->  web, desktop,
  audio tools)          objects kept,           materials, Look,        deterministic,          mobile players
                        cached derived data     lights, cameras,        content-addressed,      and the game's
                                                sequences, stages       published once          own code
        ^                                               |                                            |
        |                                               +---- live link: edits reach a running ------+
        +------------ re-export with the same object names keeps every reference -------------------+
```

1. **Content tools** model meshes, rigs, clips, textures and sounds. They export with rules that keep identity:
   one named object per thing that will be placed, separate named clips and shape keys, metres and Y up, no
   joining of objects for convenience. A DCC is not where a level is assembled.
2. **Import** reads a source file with its sidecar settings and produces documents and cached derived data. It
   never merges objects. Fix-ups that used to live in scripts (UV repair, normal treatments, light extraction from
   emissive geometry) become import settings with names, so they run identically every time.
3. **Editor assembly** places entities, instances prefabs, paints terrain and foliage, lights, sets the Look,
   frames cameras, writes sequences and configures game stages. Everything is a document with undo.
4. **Cook** turns documents and sources into per-platform packages: geometry coded compactly, textures in each
   platform's formats, entity tables, sequences, sound banks. Cooking is deterministic and named by content.
5. **Runtime** loads packages. The game's code reads the entity table and drives presentation from its own rules
   and server.

**Re-import must not break placement.** References go to asset ids and to node paths inside a model, so changing
a mesh in the content tool and exporting again updates every placed instance without touching the scene.

## 3. Data-driven development and live tuning

Large engines converge on the same idea: code defines *kinds* of things and their behaviour; data defines *which*
things, *where*, and *how much*. Unreal has reflected properties and data assets, Unity has serialized components
and ScriptableObjects, Frostbite and RAGE describe their content in large typed data sets read by generic systems.

Ganbate's version:

- **Schemas** describe every component kind (engine and game) once; the inspector, validation, the cook and the
  runtime readers follow from them.
- **Documents** hold the values: scenes, prefabs, materials, Looks, sequences, sound banks, character presets.
- **Live tuning** sends the same ops the editor saves to a running game or a phone, so tuning happens on the target
  with real timing, not in a separate preview. What a runtime cannot apply in place (new meshes) is cooked
  incrementally and hot-reloaded.
- **Code reads data, never constants.** A camera framing, a light colour, a timing or a layout offset in code is a
  missing field in a schema.

## 4. One engine, several games

| Engine | How games are separated | Lesson for Ganbate |
| --- | --- | --- |
| Unity | The engine is a product; each game is a project of assets, scripts and packages; the editor is extended by editor scripts | Projects, packages and editor extensions are the right shape; a game should never need the engine's source changed |
| Unreal | Engine source plus game modules and plugins; licensees often modify the engine, creating long-lived forks | Give games real extension points (schemas, code, plugins) so nobody forks the engine |
| RAGE | A shared core with each title's large game layer and data formats on top | Shared core, title-specific data; keep title formats out of the core |
| Frostbite | One engine across many studios and genres; well-documented costs when a game's genre differs from what the engine was built for | Prove reuse early with a different kind of game, before the API sets around the first one |
| Godot, Bevy | Text scene formats, reflection-driven inspectors, asset processors with sidecar metadata | Text-first data and sidecars work for small teams and for git |

Ganbate therefore:

1. Keeps the engine repository free of any game's names, assets and rules; samples are synthetic.
2. Gives games extension points instead of forks: component schemas, game code against versioned APIs, editor
   plugins, prop shaders and Looks ([boundary](05-boundary.md)).
3. Builds a second, non-card game sample at M9 to prove the API is general. That is later than the Frostbite
   lesson above advises, a trade accepted to ship the first game sooner. Until then every engine API ships with a
   synthetic, genre-neutral sample and test, and no game's concepts enter the engine API, before or after M9.
4. Treats tools as engine features: an engine change that adds a field also adds its schema, so the editor shows it.

## 5. Where Ganbate deliberately differs

- **Zero third-party code.** Codecs, the shader compiler, the editor host and its protocol are first-party. Only
  platform SDKs that cannot be avoided (consoles) live outside, in private platform modules.
- **A web interface for the editor.** A local Rust host plus the browser gives a usable editor in weeks and shows
  exactly what web players see; a native viewport client follows for open worlds ([editor](03-editor.md)).
- **Text-first everything.** Canonical JSON for all authored data, so a solo founder, agents and git can all read,
  diff and review content.
- **Evidence over claims.** Every platform claim names the device and driver that produced it; software
  rasterizer runs prove appearance, never device performance.
