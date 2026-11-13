//! The stylized renderer's frame: two shadow cascades, a top-down ground map (what is highest
//! around the view and whether grass grows there: rain cover, grass placement, water depth), a
//! mirrored reflection pass for water, the main forward pass (world, grass and flowers, sky,
//! weather particles) into HDR colour plus normals, then ambient occlusion, air (outlines,
//! aerial perspective, mist, light shafts), bloom, tone mapping and grading, the optional
//! painterly filter and the final output with sharpening (a swapchain image or a readback
//! target).

use crate::gpu::{
    Address, Attachment, BindGroupId, Blend, BufferDesc, BufferId, BufferUsage, ColorTarget,
    Compare, Cull, DepthState, Device, Encoder, Filter, Format, Load, Memory, PassDesc, PipelineId,
    RenderPipelineDesc, Resource, Result, SamplerDesc, SamplerId, TextureDesc, TextureId,
    TextureUsage, Topology, VertexAttribute, VertexFormat, VertexLayout, f32_bytes,
};
use ganbate_shader::{Compiled, SpecValue, Stage};

use super::look::{DetailPreset, Focus, Look, ToneMap, View};
use super::math::{
    self, Frustum, M4, V3, aabb_distance, add, cross, invert, look_at, mul, normalize, ortho,
    perspective, reflect_y, scale, sub, transform,
};
use super::props::{
    CacheChanges, CacheView, PropSet, Rect, SHADOW_DRIFT_TEXELS, ShadowMap, commit_prop_shadows,
    corners, merge_boxes, screen_rect, track_cached_props, union,
};
use super::scene::{DrawItem, GpuScene};
use super::sky::Environment;

const COMMON: &str = include_str!("../shaders/common.wgsl");
const BRDF: &str = include_str!("../shaders/brdf.wgsl");
const SCENE_DETAIL: &str = include_str!("../shaders/scene_detail.wgsl");
const SCENE_GRASS: &str = include_str!("../shaders/scene_grass.wgsl");
const SCENE: &str = include_str!("../shaders/scene.wgsl");
const WORLD_FX: &str = include_str!("../shaders/world_fx.wgsl");
const POST: &str = include_str!("../shaders/post.wgsl");
const PROP_SHADER_INPUT: &str = include_str!("../shaders/prop_shader_input.wgsl");
const PROP_SHADER: &str = include_str!("../shaders/prop_shader.wgsl");
const HDR: Format = Format::Rgba16Float;
const LUT_SIZE: u32 = 16;
/// The lamps a pixel loops over are those of its cluster: its screen tile cut into
/// `CLUSTER_SLICES` slices of view depth, evenly spaced in log depth from `CLUSTER_NEAR` to
/// `CLUSTER_FAR` metres. A tile alone holds every lamp whose sphere it sees, near or far: in the
/// Yozakura house (617 lamps) most tiles held the 96 a list keeps, and each pixel read them all.
const CLUSTER_SLICES: u32 = 16;
const CLUSTER_NEAR: f32 = 0.5;
const CLUSTER_FAR: f32 = 400.0;
const MAX_LIGHTS_PER_CLUSTER: usize = 96;
/// Room for a tile's clusters' lists together, in light indices (24 a cluster on average): a lamp
/// joins every slice its sphere crosses, so the lists hold several times the lamps a tile sees.
const CLUSTER_ENTRIES_PER_TILE: u64 = CLUSTER_SLICES as u64 * 24;
/// Frames a cached world may lag the converging AO by (a quarter second at 60 frames a second).
const AO_CACHE_REFRESH_FRAMES: u32 = 15;
/// Byte offset of `camera.w` (the clock's seconds) in the frame block: after five matrices.
const CAMERA_SECONDS: usize = 5 * 64 + 12;

/// FNV-1a over bytes: a fingerprint of what the cached world was drawn for.
pub(in crate::render) use crate::digest::Fnv1a64 as Fnv;

/// Vertex AO samples a live frame may bake (each casts `AoConfig::rays` rays): about 3 ms in
/// wasm, inside a 120 Hz frame. Nearest the camera first, so what is in view settles first.
const AO_FRAME_SAMPLES: usize = 400;
/// Frames a world must stay unmoved before its AO is re-baked, so a drag is not slowed by it.
const AO_SETTLE_FRAMES: u32 = 12;
/// The near clip distance of `view`, kept positive.
fn near(view: &View) -> f32 {
    if view.near > 1e-4 { view.near } else { 1e-4 }
}

/// `view.up`, or a perpendicular fallback when it is (nearly) along the line of sight.
fn camera_up(view: &View) -> V3 {
    let fwd = normalize(sub(view.target, view.eye));
    if math::length(cross(fwd, view.up)) > 1e-4 {
        view.up
    } else if fwd[1].abs() > 0.9 {
        [0.0, 0.0, -1.0]
    } else {
        [0.0, 1.0, 0.0]
    }
}
/// The ground map covers this far (metres) around a point just ahead of the camera.
const GROUND_REACH: f32 = 48.0;
/// The world map covers the land's bounds plus this margin (metres) on every side, so shallows
/// and foam reach out past the outermost shore.
const WORLD_MARGIN: f32 = 16.0;
/// ...but reaches at most this far (metres) from where the land's detail is densest, so distant
/// scenery does not spread its texels thin.
const WORLD_MAX_REACH: f32 = 560.0;
/// The grass instance grid is at most GRASS_SIDE x GRASS_SIDE cells of 13 cm around the camera
/// (see scene.wgsl); `Quality::grass` scales its area.
const GRASS_SIDE: u32 = 640;
/// Metres between grass blades' cells (`GRASS_SPACING` in scene.wgsl).
const GRASS_SPACING: f32 = 0.13;
const RAIN_BOX: f32 = 16.0;
const PETAL_BOX: f32 = 14.0;
const GROUND_FORMAT: Format = Format::Rgba8Unorm;
/// The widest a defocused pixel spreads, in pixels: past it the taps would scatter.
const MAX_FOCUS_BLUR_PX: f32 = 24.0;
/// Size of the per-view uniform buffer holding the `Frame` block.
const FRAME_BYTES: u64 = 1536;

/// How much work each frame does. [`Quality::still`] is for offline images,
/// [`Quality::realtime`] for the interactive player.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quality {
    pub shadow_size: u32,
    pub reflection_scale: f32,
    pub ao_scale: f32,
    pub ao_samples: u32,
    pub tile_size: u32,
    pub bloom_levels: u32,
    pub painterly: bool,
    pub fxaa: bool,
    pub particles: bool,
    /// Texels per side of the top-down ground map.
    pub ground_size: u32,
    /// Texels per side of the world map (shallows and foam beyond the ground map).
    pub world_size: u32,
    /// Fraction of the full grass area drawn around the camera (0 = no grass).
    pub grass: f32,
    /// Texels per side of each game light's shadow map (see [`LightDesc::shadow`]).
    ///
    /// [`LightDesc::shadow`]: super::LightDesc::shadow
    pub light_shadow_size: u32,
}
impl Quality {
    pub fn still() -> Self {
        Self {
            shadow_size: 4096,
            reflection_scale: 1.0,
            ao_scale: 1.0,
            ao_samples: 16,
            tile_size: 16,
            bloom_levels: 6,
            painterly: true,
            fxaa: false,
            particles: true,
            ground_size: 1024,
            world_size: 2048,
            grass: 1.0,
            light_shadow_size: 2048,
        }
    }
    pub fn realtime() -> Self {
        Self {
            shadow_size: 2048,
            // A quarter of the view: the water's ripples blur the mirror past that detail, and at
            // half the pass cost the table 6 ms of an 18 ms frame (780x1328, Radeon 780M).
            reflection_scale: 0.25,
            ao_scale: 0.5,
            ao_samples: 8,
            tile_size: 32,
            bloom_levels: 5,
            painterly: true,
            fxaa: true,
            particles: true,
            ground_size: 1024,
            world_size: 1024,
            grass: 0.6,
            light_shadow_size: 1024,
        }
    }
    /// Hybrid realtime budget tuned for interactive 60 fps stages: baked vertex AO instead of
    /// SSAO, cached static shadow cascades, frustum + distance chunk culling, water-visibility
    /// reflection skip, and single-pass fused 3D LUT tone mapping + output.
    pub fn hybrid_stage() -> Self {
        Self {
            shadow_size: 1024,
            reflection_scale: 0.2,
            ao_scale: 0.25,
            ao_samples: 0,
            tile_size: 32,
            bloom_levels: 4,
            painterly: false,
            fxaa: true,
            particles: true,
            ground_size: 512,
            world_size: 1024,
            grass: 0.4,
            light_shadow_size: 512,
        }
    }

    /// Ultra-lean fast-path budget tuned for 120 Hz desktop, mobile WebGPU, and 90 Hz stereo VR
    /// stages: skips planar water reflections, SSAO, Kuwahara painterly filtering, FXAA, weather
    /// particles, and procedural grass while shrinking shadow/ground/world caches and bloom mips.
    pub fn low_cost() -> Self {
        Self {
            shadow_size: 512,
            reflection_scale: 0.0,
            ao_scale: 0.25,
            ao_samples: 0,
            tile_size: 32,
            bloom_levels: 2,
            painterly: false,
            fxaa: false,
            particles: false,
            ground_size: 256,
            world_size: 512,
            grass: 0.0,
            light_shadow_size: 256,
        }
    }
}

/// Per-frame visibility, cache, and pass-skip telemetry recorded by [`Renderer::encode`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    pub world_draws_total: u32,
    pub world_draws_submitted: u32,
    pub world_draws_culled: u32,
    pub shadow_static_cached: bool,
    pub shadow_world_draws: u32,
    pub ground_cached: bool,
    pub reflection_drawn: bool,
    pub ssao_drawn: bool,
    pub fused_tonemap: bool,
    /// The frame started from the cached world rather than drawing it.
    pub world_cached: bool,
    /// Regions of the cached picture drawn again this frame (where props settled, left or
    /// moved their shadows).
    pub cache_regions: u32,
    /// Pixels of the cached picture that moved props' shadows can reach were shaded again.
    pub reshaded: bool,
    /// The pixels of the region they were sought in (`fs_reshade_mask`) and shaded again within.
    pub reshade_pixels: u32,
    /// The water read its lamp light and shadow as kept with the cached picture (`cache_water`)
    /// rather than computing them, as a still does.
    pub water_kept: bool,
    /// The water's kept lamp light and shadow were drawn whole this frame.
    pub water_statics_whole: bool,
    /// Regions of them drawn again this frame, with the cached picture's.
    pub water_statics_regions: u32,
    /// Footprints of the shadows of props moved since they were drawn whole (one shadow map
    /// apiece), within which the water computes them.
    pub water_footprints: u32,
    /// The props' shadow maps (the key light's prop cascades, a game's lights') were kept from
    /// the last frame: no caster moved, and the lights' views held.
    pub prop_shadows_kept: bool,
    /// The painterly filter ran (it is skipped where it would change nothing).
    pub painted: bool,
    /// The fused tone-map output graded each pixel once, in a pass of its own
    /// ([`Renderer::set_grade_once`]).
    pub graded_once: bool,
    /// The ambient occlusion was kept from the last frame and computed again only where what it
    /// reads changed and on the water (false: computed whole, or not at all).
    pub ao_kept: bool,
    /// Pixels of the CPU changed regions (`union_area(&ao_regions)`) when `ao_kept` is true,
    /// `width * height` when >= 50% CPU region coverage triggers the whole-pass fallback, or 0
    /// when skipped or drawn whole from scratch.
    pub ao_changed_pixels: u32,
}

/// Where the final pass writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// An offscreen RGBA16F target read back as linear, display-referred colour.
    Still,
    /// A presentation surface of this format (the final pass encodes sRGB when the format
    /// does not).
    Surface(Format),
}

/// Which of the world's draws one `draw_world` call makes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WorldPart {
    All,
    /// Opaque draws, nearest first (the main view, with the pipeline that never discards).
    Opaque,
    /// The alpha-tested draws, in scene order.
    Masked,
    /// The water without an alpha test, in scene order (its surface moves, so a cached world
    /// leaves it out; it takes the pipeline that never discards, so what stands in front of it
    /// hides it before it is shaded).
    Water,
    /// The water with an alpha test.
    MaskedWater,
}

pub(in crate::render) struct Pipelines {
    pub(in crate::render) scene: PipelineId,
    /// The scene shading without the alpha test, for opaque world surfaces in the main view: a
    /// shader that never discards lets the GPU drop hidden fragments before shading them.
    scene_opaque: PipelineId,
    /// The water in a frame that keeps the cached picture (`fs_water`): `water_opaque` with its
    /// lamp light and shadow read from `cache_water` where no prop moved since they were kept.
    water: PipelineId,
    /// Uncached opaque water in the main view (`fs_water_opaque`).
    water_opaque: PipelineId,
    /// Alpha-tested water in the main view (`fs_water_masked`).
    water_masked: PipelineId,
    /// Draws the water's lamp light and shadow into `cache_water` (`fs_water_static`).
    water_static: PipelineId,
    /// The scene shading for props (model-space vertices placed per instance).
    pub(in crate::render) scene_prop: PipelineId,
    /// The same without the alpha test, for opaque prop materials in the main view.
    pub(in crate::render) scene_prop_opaque: PipelineId,
    /// Water shading for alpha-tested and opaque props in the main view.
    pub(in crate::render) water_prop: PipelineId,
    pub(in crate::render) water_prop_opaque: PipelineId,
    /// The same four drawing into the cached picture, which also keeps the shadows moved props
    /// can change (`fs_main_cache`): world cut and opaque, props cut and opaque.
    cache_scene: [PipelineId; 2],
    pub(in crate::render) cache_prop: [PipelineId; 2],
    /// Blended dry and water props over the scene without writing depth: alpha, additive.
    pub(in crate::render) scene_prop_blend: [PipelineId; 2],
    pub(in crate::render) water_prop_blend: [PipelineId; 2],
    /// Props over the finished image: [opaque, alpha, additive][without the scene's depth, testing
    /// it, testing and writing it (opaque only; the others repeat the test)].
    pub(in crate::render) overlay: [[PipelineId; 3]; 3],
    /// What a game's prop shader is built against: the output format, the prop vertex layout,
    /// and the resources the built-in overlay fragment reads (a prop shader may read no others,
    /// so it takes the same bind groups).
    pub(in crate::render) output_format: Format,
    pub(in crate::render) prop_layout: VertexLayout,
    /// What a prop shader's fragment stage reads: an overlay's frame bindings and a scene prop's
    /// material group (its draws bind a material made for `scene_prop`).
    pub(in crate::render) prop_shader_bindings: Vec<ganbate_shader::Binding>,
    /// The props' vertex stage (`vs_prop`): every prop shader's, which never calls `shade`.
    pub(in crate::render) prop_vs: Compiled,
    /// Copies the cached world back into the frame's colour, aux and depth targets.
    restore: PipelineId,
    /// Empties a region of the cached picture (colour, aux and depth) before it is drawn again.
    cache_clear: PipelineId,
    /// Marks the cached picture's pixels moved props' shadows can reach (`fs_reshade_mask`):
    /// depth 0 there, 1 elsewhere, so a re-shade pass draws only there.
    reshade_mask: PipelineId,
    shadow: PipelineId,
    shadow_masked: PipelineId,
    pub(in crate::render) shadow_prop: PipelineId,
    pub(in crate::render) shadow_prop_masked: PipelineId,
    ground: PipelineId,
    grass: PipelineId,
    flower: PipelineId,
    sky: PipelineId,
    particle: PipelineId,
    ao_grass_tile: PipelineId,
    ao_grass_union: PipelineId,
    ao_grass_coarse: PipelineId,
    ssao: PipelineId,
    air: PipelineId,
    bloom_down: PipelineId,
    bloom_up: PipelineId,
    tonemap: PipelineId,
    tonemap_final: PipelineId,
    /// `fs_tonemap` and `fs_final` in one pass, where no painterly filter runs between them; then
    /// the same for a Look whose grade changes nothing ([`Look::grade_is_identity`]).
    tonemap_output: [PipelineId; 2],
    /// The same in two passes ([`Renderer::set_grade_once`]), made when first chosen: `fs_grade`
    /// and `fs_grade_plain` into [`GRADED_FORMAT`], then `fs_tonemap_output` compiled with
    /// `GRADED`, which reads what they wrote.
    grade_once: Option<([PipelineId; 2], PipelineId)>,
    paint: PipelineId,
    output: PipelineId,
}

pub(in crate::render) struct Targets {
    width: u32,
    height: u32,
    hdr: TextureId,
    aux: TextureId,
    depth: TextureId,
    /// The world's part of the main pass as last drawn ([`Renderer::world_cache`]).
    cache_hdr: TextureId,
    cache_aux: TextureId,
    cache_depth: TextureId,
    /// Where a frame marks the cached picture's pixels to shade again (`fs_reshade_mask`).
    reshade_depth: TextureId,
    /// The shadows moved props can change, as the cached picture was shaded with them
    /// (`CacheOut::shadow`).
    cache_shadow: TextureId,
    /// The water's lamp light and shadow where the cached picture shows it (`fs_water_static`),
    /// for a presented view on a device that reads 32-bit float textures.
    cache_water: Option<TextureId>,
    refl: (u32, u32),
    pub(in crate::render) refl_hdr: TextureId,
    refl_aux: TextureId,
    refl_depth: TextureId,
    refl_cleared: bool,
    ao: (u32, u32),
    /// The AO, kept from frame to frame while what it was computed for (`ao_for`) holds.
    ao_tex: TextureId,
    ao_for: Option<u64>,
    ao_cleared: bool,
    /// Hierarchical 1/8-scale and 1/64-scale visible-grass coverage masks (`Format::R8Unorm`).
    ao_grass: [TextureId; 2],
    ao_grass_cur: usize,
    ao_grass_valid: [bool; 2],
    ao_grass_time: u32,
    ao_grass_sway: u32,
    ao_grass_union: TextureId,
    ao_grass_coarse: TextureId,
    lit: TextureId,
    bloom: TextureId,
    bloom_size: (u32, u32),
    bloom_levels: u32,
    ldr: Option<TextureId>,
    paint: Option<TextureId>,
    /// Each pixel's grade, where frames grade once ([`Renderer::set_grade_once`]).
    graded: Option<TextureId>,
    still: Option<TextureId>,
    readback: Option<BufferId>,
    tiles: [(u32, u32); 2],
}

/// Per-frame-slot uniform buffers, light tiles and the bind groups that reference them.
pub(in crate::render) struct Slot {
    pub(in crate::render) frame: [BufferId; 2],
    pub(in crate::render) tiles: [BufferId; 2],
    pub(in crate::render) index: [BufferId; 2],
    /// What each view's light tiles were last binned for ([`Renderer::tile_key`]): while that
    /// holds they are neither binned nor written again.
    tiles_for: [Option<u64>; 2],
    scene: [BindGroupId; 2],
    sky: [BindGroupId; 2],
    /// The water's group in a frame that keeps the cached picture (`fs_water`, with `cache_water`)
    /// and the `Reshade` block it binds: the footprints of props moved since its lamp light and
    /// shadow were kept.
    water: Option<(BindGroupId, BufferId)>,
    grass: BindGroupId,
    flower: BindGroupId,
    particle: BindGroupId,
    restore: BindGroupId,
    /// `fs_reshade_mask`'s group and its `Reshade` block.
    mask: BindGroupId,
    reshade: BufferId,
    ao_grass_tile: BindGroupId,
    ao_grass_union: BindGroupId,
    ao_grass_coarse: BindGroupId,
    /// `fs_ssao`'s group and its `AoChanged` block.
    ssao: BindGroupId,
    ao_changed: BufferId,
    air: BindGroupId,
    down: Vec<BindGroupId>,
    up: Vec<BindGroupId>,
    tonemap: BindGroupId,
    tonemap_final: BindGroupId,
    tonemap_output: [BindGroupId; 2],
    /// `Pipelines::grade_once`'s groups, where frames grade once.
    grade_once: Option<([BindGroupId; 2], BindGroupId)>,
    paint: Option<BindGroupId>,
    output: Option<BindGroupId>,
}

pub struct Renderer {
    /// What the cached world was drawn for (view, size, frame inputs, world revision); `None`
    /// until it is drawn and after anything that changes it without changing those.
    world_cache: Option<u64>,
    /// The part of the grass grid this frame draws ([`Renderer::visible_grass`]).
    grass_grid: [u32; 4],
    /// The world's draws' bounds for re-shading the cached picture, as of `world_cache`.
    receivers: Option<ReceiverBounds>,
    /// The water's lamp light and shadow kept with the cached picture (`cache_water`): `None`
    /// until drawn whole for it as it stands; then the boxes of props moved since that the water
    /// follows (within their shadows' footprints `fs_water` computes them).
    water_statics: Option<Vec<(V3, V3)>>,
    /// The AO refinement the cached world shows, and the frames since it was drawn.
    cache_ao_revision: u64,
    cache_age: u32,
    /// The main pass ends and starts again between what it draws, so each part is timed as a pass
    /// of its own on a device ([`Renderer::set_split_passes`]).
    split_passes: bool,
    pub(in crate::render) scene: GpuScene,
    pub(in crate::render) pipelines: Pipelines,
    pub(in crate::render) quality: Quality,
    pub(in crate::render) output: Output,
    pub(in crate::render) look: Look,
    pub(in crate::render) targets: Targets,
    pub(in crate::render) slots: Vec<Slot>,
    pub(in crate::render) materials: Vec<BindGroupId>,
    /// Bind groups for alpha-masked materials in the masked shadow pipeline.
    pub(in crate::render) shadow_materials: Vec<Option<BindGroupId>>,
    pub(in crate::render) shadow_map: TextureId,
    shadow_layers_ready: bool,
    static_shadows_vp: Option<[M4; 2]>,
    /// What the props' shadow maps were last drawn for; while it holds and no caster moved, they
    /// are kept.
    prop_shadows_for: Option<PropShadowViews>,
    /// The game lights' shadow maps, a layer for each light casting shadows.
    pub(in crate::render) light_shadow_map: TextureId,
    /// Every layer of the light shadow map has been written (cleared) once.
    light_shadows_ready: bool,
    pub(in crate::render) ground_depth: TextureId,
    pub(in crate::render) ground_mask: TextureId,
    ground_cached_vp: Option<M4>,
    /// The world map: the whole land seen from above once, for shallows and foam on shores
    /// beyond the ground map. Drawn on the first frame and again when the Look changes or a
    /// world instance moves.
    pub(in crate::render) world_depth: TextureId,
    pub(in crate::render) world_mask: TextureId,
    world: Ground,
    pub(in crate::render) world_drawn: bool,
    /// A world instance moved since the last frame: bounds and the world map follow.
    pub(in crate::render) world_moved: bool,
    /// Frames since a world instance last moved (the AO bake waits for edits to pause).
    ao_quiet_frames: u32,
    /// Per material: the procedural detail block pushed with each draw (see `Draw` in
    /// scene.wgsl), from the Look's detail rules.
    pub(in crate::render) details: Vec<[f32; 16]>,
    /// Per material: grows grass (the ground map's mask).
    pub(in crate::render) grassy: Vec<bool>,
    pub(in crate::render) lut: TextureId,
    lut_dirty: bool,
    pub(in crate::render) dummy: TextureId,
    pub(in crate::render) linear_clamp: SamplerId,
    pub(in crate::render) shadow_sampler: SamplerId,
    /// Fallback textures and the material sampler, shared with prop materials.
    pub(in crate::render) white: TextureId,
    pub(in crate::render) flat: TextureId,
    pub(in crate::render) material_sampler: SamplerId,
    pub(in crate::render) props: super::props::Props,
    pub(in crate::render) game_lights: super::lights::GameLights,
    pub(in crate::render) models: super::models::Models,
    stats: FrameStats,
}

/// Builds the `Frame` uniform block in WGSL layout.
struct Block(Vec<f32>);
impl Block {
    fn m(&mut self, m: &M4) {
        self.0.extend_from_slice(m);
    }
    fn v(&mut self, v: V3, w: f32) {
        self.0.extend_from_slice(&[v[0], v[1], v[2], w]);
    }
    fn f(&mut self, a: f32, b: f32, c: f32, d: f32) {
        self.0.extend_from_slice(&[a, b, c, d]);
    }
}

/// The top-down ground map for one frame.
struct Ground {
    vp: M4,
    top: f32,
    range: f32,
    size: u32,
}

/// A camera for one frame: view-projection, its inverse and the eye.
#[derive(Clone, Copy)]
pub(in crate::render) struct Eye {
    pub(in crate::render) view_proj: M4,
    pub(in crate::render) inv: M4,
    pos: V3,
    near: f32,
}

impl Eye {
    /// This eye narrowed to the pixels `r` of a `size` picture: what it culls lies outside them.
    fn within(&self, r: &Rect, size: (u32, u32)) -> Eye {
        let (w, h) = (size.0 as f32, size.1 as f32);
        let (x0, x1) = (r[0] as f32 / w * 2.0 - 1.0, r[2] as f32 / w * 2.0 - 1.0);
        let (y0, y1) = (r[1] as f32 / h * 2.0 - 1.0, r[3] as f32 / h * 2.0 - 1.0);
        let (sx, sy) = ((x1 - x0) * 0.5, (y1 - y0) * 0.5);
        let (cx, cy) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        #[rustfmt::skip]
        let narrow: M4 = [
            1.0 / sx, 0.0, 0.0, 0.0,
            0.0, 1.0 / sy, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -cx / sx, -cy / sy, 0.0, 1.0,
        ];
        Eye {
            view_proj: mul(&narrow, &self.view_proj),
            inv: self.inv,
            pos: self.pos,
            near: self.near,
        }
    }
}

/// The views the props' shadow maps are drawn from: the key light's cascades while it shines on
/// casters (layers 2, 3 of its map), and each shadowed game light's.
#[derive(PartialEq)]
struct PropShadowViews {
    cascades: Option<[M4; 2]>,
    lights: Vec<M4>,
}

struct Cascades {
    mats: [M4; 2],
    split: f32,
    offset: [f32; 2],
    radius: [f32; 2],
    bias: [f32; 2],
}

fn halton(i: u32, base: u32) -> f32 {
    let (mut f, mut r, mut i) = (1.0, 0.0, i);
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let frac = f32::from(h & 0x3ff);
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + frac / 1024.0) * 2f32.powi(i32::from(e) - 15),
    }
}

/// The world map's top-down projection: the land's XZ bounds plus a margin, no further than
/// `WORLD_MAX_REACH` from the land's centre (square, so texels are square), depth 0 above the
/// highest point of the world and 1 below the lowest.
fn world_view(scene: &GpuScene, size: u32) -> Ground {
    let c = scene.land_centre;
    let span = |k: usize| {
        (
            (scene.land_min[k] - WORLD_MARGIN).max(c[k] - WORLD_MAX_REACH),
            (scene.land_max[k] + WORLD_MARGIN).min(c[k] + WORLD_MAX_REACH),
        )
    };
    let ((x0, x1), (z0, z1)) = (span(0), span(2));
    let (x, z) = (0.5 * (x0 + x1), 0.5 * (z0 + z1));
    let r = 0.5 * (x1 - x0).max(z1 - z0);
    let top = scene.max[1] + 1.0;
    let range = (top - scene.min[1] + 1.0).max(2.0);
    let v = look_at([x, top, z], [x, top - 1.0, z], [0.0, 0.0, -1.0]);
    Ground {
        vp: mul(&ortho(-r, r, -r, r, 0.0, range), &v),
        top,
        range,
        size,
    }
}

/// The detail block of a material named `name` (its first matching Look rule) and whether grass
/// grows on it.
pub(in crate::render) fn detail_block(look: &Look, name: &str) -> ([f32; 16], bool) {
    let rule = (!name.is_empty())
        .then(|| look.details.iter().find(|r| r.matches(name)))
        .flatten();
    let srule = (!name.is_empty())
        .then(|| look.surface_rules.iter().find(|r| r.matches(name)))
        .flatten();
    let mut d = [0.0f32; 16];
    d[11] = -1.0;
    d[15] = -1.0;
    if let Some(r) = rule {
        let c = r.color.unwrap_or([0.0; 3]);
        // A lattice may leave out one direction of bars with a zero size.
        let floor = if r.preset == DetailPreset::Lattice {
            0.0
        } else {
            1e-3
        };
        d[..4].copy_from_slice(&[
            r.preset as u32 as f32,
            r.scale[0].max(floor),
            r.scale[1].max(floor),
            r.strength,
        ]);
        d[4..8].copy_from_slice(&[r.variation, c[0], c[1], c[2]]);
        d[8] = if r.color.is_some() { 1.0 } else { 0.0 };
        d[9] = r.gloss;
    }
    if let Some(sr) = srule {
        if let Some(s) = sr.surface {
            d[10] = (s as u32 + 1) as f32;
        }
        if let Some(w) = sr.clearcoat.or(sr.sheen).or(sr.subsurface) {
            d[11] = w;
        }
        if let Some(p) = sr.clearcoat_roughness.or(sr.sheen_tint) {
            d[15] = p;
        }
    }
    let grassy =
        rule.is_some_and(|r| matches!(r.preset, DetailPreset::Grass | DetailPreset::Terrain));
    (d, grassy)
}

#[allow(clippy::excessive_precision)]
fn tonemap_cpu(c: V3, mode: ToneMap) -> V3 {
    let dot3 = |a: V3, b: V3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    match mode {
        ToneMap::Stylized => c.map(|v| {
            let x = (v * 0.8).max(0.0);
            let a = x * (2.51 * x + 0.03);
            let b = x * (2.43 * x + 0.59) + 0.14;
            (a / b).clamp(0.0, 1.0)
        }),
        ToneMap::AgX => {
            let x = c.map(|v| v.max(1e-10));
            let inset = [
                dot3(x, [0.84247906, 0.07843360, 0.07922375]),
                dot3(x, [0.04232824, 0.87846864, 0.07916613]),
                dot3(x, [0.04237565, 0.07843360, 0.87914297]),
            ];
            let lin = inset.map(|v| {
                let ev = ((v.max(1e-10).log2() + 12.47393) / 16.5).clamp(0.0, 1.0);
                let x2 = ev * ev;
                let x4 = x2 * x2;
                let p = 15.5 * x4 * x2 - 40.14 * x4 * ev + 31.96 * x4 - 6.868 * x2 * ev
                    + 0.4298 * x2
                    + 0.1191 * ev
                    - 0.00232;
                p.max(0.0).powf(2.2)
            });
            [
                dot3(lin, [1.19687901, -0.09802088, -0.09902974]).clamp(0.0, 1.0),
                dot3(lin, [-0.05289685, 1.15190313, -0.09896118]).clamp(0.0, 1.0),
                dot3(lin, [-0.05297164, -0.09804345, 1.15107367]).clamp(0.0, 1.0),
            ]
        }
        ToneMap::ACESFitted => {
            let x = c.map(|v| v.max(0.0));
            let ap1 = [
                dot3(x, [0.59719, 0.35458, 0.04823]),
                dot3(x, [0.07600, 0.90834, 0.01566]),
                dot3(x, [0.02840, 0.13383, 0.83777]),
            ];
            let rrt = ap1.map(|v| {
                (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081)
            });
            [
                dot3(rrt, [1.60475, -0.53108, -0.07367]).clamp(0.0, 1.0),
                dot3(rrt, [-0.10208, 1.10813, -0.00605]).clamp(0.0, 1.0),
                dot3(rrt, [-0.00327, -0.07276, 1.07602]).clamp(0.0, 1.0),
            ]
        }
    }
}

/// Bakes a `16x16x16` colour-grading 3D LUT into a `256x16` RGBA8 strip matching `fs_tonemap`.
fn bake_lut(look: &Look) -> Vec<u8> {
    let n = LUT_SIZE as usize;
    let mut out = vec![0u8; n * n * n * 4];
    for g_i in 0..n {
        for b_i in 0..n {
            for r_i in 0..n {
                let u = [
                    r_i as f32 / (n - 1) as f32,
                    g_i as f32 / (n - 1) as f32,
                    b_i as f32 / (n - 1) as f32,
                ];
                let hdr = u.map(|v| v / (1.0 - v).max(1e-4));
                let mut col = tonemap_cpu(hdr, look.tonemap);
                let lum = 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2];
                for ch in &mut col {
                    *ch = (lum + (*ch - lum) * look.saturation).max(0.0);
                    let g = ch.powf(1.0 / 2.2);
                    let c = ((g - 0.5) * look.contrast + 0.5).clamp(0.0, 1.0);
                    *ch = c.powf(2.2);
                }
                for (k, ch) in col.iter_mut().enumerate() {
                    *ch = (*ch * look.gain[k] + look.lift[k] * (1.0 - *ch)).clamp(0.0, 1.0);
                }
                let px = (g_i * (n * n) + b_i * n + r_i) * 4;
                out[px] = (col[0] * 255.0).round() as u8;
                out[px + 1] = (col[1] * 255.0).round() as u8;
                out[px + 2] = (col[2] * 255.0).round() as u8;
                out[px + 3] = 255;
            }
        }
    }
    out
}

/// The detail block of each world material and whether grass grows on it.
fn material_details(look: &Look, scene: &GpuScene) -> (Vec<[f32; 16]>, Vec<bool>) {
    scene.names.iter().map(|n| detail_block(look, n)).unzip()
}

mod cache;
mod encode;
mod lifecycle;
mod pipelines;
mod ssao;
#[cfg(test)]
mod tests;

use cache::*;
pub use pipelines::compose_shader;
use pipelines::*;
pub(in crate::render) use pipelines::{prop_shader_pipeline, prop_shader_stage};
use ssao::*;
