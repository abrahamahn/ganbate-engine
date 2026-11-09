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
use ganbate_shader::{Compiled, Stage};

use super::look::{DetailPreset, Focus, Look, ToneMap, View};
use super::math::{
    self, Frustum, M4, V3, aabb_distance, add, cross, invert, look_at, mul, normalize, ortho,
    perspective, reflect_y, scale, sub, transform,
};
use super::scene::{DrawItem, GpuScene};
use super::sky::Environment;

const COMMON: &str = include_str!("shaders/common.wgsl");
const SCENE: &str = include_str!("shaders/scene.wgsl");
const WORLD_FX: &str = include_str!("shaders/world_fx.wgsl");
const POST: &str = include_str!("shaders/post.wgsl");
const PROP_SHADER_INPUT: &str = include_str!("shaders/prop_shader_input.wgsl");
const PROP_SHADER: &str = include_str!("shaders/prop_shader.wgsl");
const HDR: Format = Format::Rgba16Float;
const LUT_SIZE: u32 = 16;
const MAX_LIGHTS_PER_TILE: usize = 96;
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
    /// The alpha-tested and water draws, in scene order.
    Rest,
}

pub(super) struct Pipelines {
    pub(super) scene: PipelineId,
    /// The scene shading without the alpha test, for opaque world surfaces in the main view: a
    /// shader that never discards lets the GPU drop hidden fragments before shading them.
    scene_opaque: PipelineId,
    /// The scene shading for props (model-space vertices placed per instance).
    pub(super) scene_prop: PipelineId,
    /// The same without the alpha test, for opaque prop materials in the main view.
    pub(super) scene_prop_opaque: PipelineId,
    /// The same, blended over the scene without writing depth: alpha, additive.
    pub(super) scene_prop_blend: [PipelineId; 2],
    /// Props over the finished image: [opaque, alpha, additive][without the scene's depth, testing
    /// it, testing and writing it (opaque only; the others repeat the test)].
    pub(super) overlay: [[PipelineId; 3]; 3],
    /// What a game's prop shader is built against: the output format, the prop vertex layout,
    /// and the resources the built-in overlay fragment reads (a prop shader may read no others,
    /// so it takes the same bind groups).
    pub(super) output_format: Format,
    pub(super) prop_layout: VertexLayout,
    /// What a prop shader's fragment stage reads: an overlay's frame bindings and a scene prop's
    /// material group (its draws bind a material made for `scene_prop`).
    pub(super) prop_shader_bindings: Vec<ganbate_shader::Binding>,
    /// The props' vertex stage (`vs_prop`): every prop shader's, which never calls `shade`.
    pub(super) prop_vs: Compiled,
    shadow: PipelineId,
    shadow_masked: PipelineId,
    pub(super) shadow_prop: PipelineId,
    pub(super) shadow_prop_masked: PipelineId,
    ground: PipelineId,
    grass: PipelineId,
    flower: PipelineId,
    sky: PipelineId,
    particle: PipelineId,
    ssao: PipelineId,
    air: PipelineId,
    bloom_down: PipelineId,
    bloom_up: PipelineId,
    tonemap: PipelineId,
    tonemap_final: PipelineId,
    paint: PipelineId,
    output: PipelineId,
}

pub(super) struct Targets {
    width: u32,
    height: u32,
    hdr: TextureId,
    aux: TextureId,
    depth: TextureId,
    refl: (u32, u32),
    pub(super) refl_hdr: TextureId,
    refl_aux: TextureId,
    refl_depth: TextureId,
    refl_cleared: bool,
    ao: (u32, u32),
    ao_tex: TextureId,
    ao_cleared: bool,
    lit: TextureId,
    bloom: TextureId,
    bloom_size: (u32, u32),
    bloom_levels: u32,
    ldr: TextureId,
    paint: TextureId,
    still: Option<TextureId>,
    readback: Option<BufferId>,
    tiles: [(u32, u32); 2],
}

/// Per-frame-slot uniform buffers, light tiles and the bind groups that reference them.
pub(super) struct Slot {
    pub(super) frame: [BufferId; 2],
    pub(super) tiles: [BufferId; 2],
    pub(super) index: [BufferId; 2],
    scene: [BindGroupId; 2],
    sky: [BindGroupId; 2],
    grass: BindGroupId,
    flower: BindGroupId,
    particle: BindGroupId,
    ssao: BindGroupId,
    air: BindGroupId,
    down: Vec<BindGroupId>,
    up: Vec<BindGroupId>,
    tonemap: BindGroupId,
    tonemap_final: BindGroupId,
    paint: BindGroupId,
    output: BindGroupId,
}

pub struct Renderer {
    pub(super) scene: GpuScene,
    pub(super) pipelines: Pipelines,
    pub(super) quality: Quality,
    pub(super) output: Output,
    pub(super) look: Look,
    pub(super) targets: Targets,
    pub(super) slots: Vec<Slot>,
    pub(super) materials: Vec<BindGroupId>,
    /// Bind groups for alpha-masked materials in the masked shadow pipeline.
    pub(super) shadow_materials: Vec<Option<BindGroupId>>,
    pub(super) shadow_map: TextureId,
    shadow_layers_ready: bool,
    static_shadows_vp: Option<[M4; 2]>,
    /// The game lights' shadow maps, a layer for each light casting shadows.
    pub(super) light_shadow_map: TextureId,
    /// Every layer of the light shadow map has been written (cleared) once.
    light_shadows_ready: bool,
    pub(super) ground_depth: TextureId,
    pub(super) ground_mask: TextureId,
    ground_cached_vp: Option<M4>,
    /// The world map: the whole land seen from above once, for shallows and foam on shores
    /// beyond the ground map. Drawn on the first frame and again when the Look changes or a
    /// world instance moves.
    pub(super) world_depth: TextureId,
    pub(super) world_mask: TextureId,
    world: Ground,
    pub(super) world_drawn: bool,
    /// A world instance moved since the last frame: bounds and the world map follow.
    pub(super) world_moved: bool,
    /// Frames since a world instance last moved (the AO bake waits for edits to pause).
    ao_quiet_frames: u32,
    /// Per material: the procedural detail block pushed with each draw (see `Draw` in
    /// scene.wgsl), from the Look's detail rules.
    pub(super) details: Vec<[f32; 16]>,
    /// Per material: grows grass (the ground map's mask).
    pub(super) grassy: Vec<bool>,
    pub(super) lut: TextureId,
    lut_dirty: bool,
    pub(super) dummy: TextureId,
    pub(super) linear_clamp: SamplerId,
    pub(super) shadow_sampler: SamplerId,
    /// Fallback textures and the material sampler, shared with prop materials.
    pub(super) white: TextureId,
    pub(super) flat: TextureId,
    pub(super) material_sampler: SamplerId,
    pub(super) props: super::props::Props,
    pub(super) game_lights: super::lights::GameLights,
    pub(super) models: super::models::Models,
    stats: FrameStats,
}

fn shader_modules() -> [(&'static str, &'static str); 2] {
    let (common_mod, brdf_mod) = COMMON
        .split_once("// --- module: brdf ---")
        .expect("brdf module delimiter in common.wgsl");
    [("common", common_mod), ("brdf", brdf_mod)]
}

fn compile(body: &str, stage: Stage, entry: &str) -> Result<Compiled> {
    let source = ganbate_shader::compose(body, &shader_modules()).map_err(|e| e.to_string())?;
    crate::gpu::compile_shader(&source, stage, entry)
}

/// The scene shader with a game's prop shader (`shade`) and the overlay entry point that calls it.
pub(super) fn prop_shader_source(user: &str) -> Result<String> {
    let raw = format!(
        "{SCENE}\n{PROP_SHADER_INPUT}\n// --- prop shader\n{user}\n// --- prop shader \
         entry\n{PROP_SHADER}"
    );
    ganbate_shader::compose(&raw, &shader_modules()).map_err(|e| e.to_string())
}

/// A game's prop shader's fragment stage compiled, checked to read only what its draw binds: an
/// overlay's frame and a scene prop's material (its three textures). Its vertex stage is every
/// prop's (`Pipelines::prop_vs`).
pub(super) fn prop_shader_stage(pipelines: &Pipelines, user: &str) -> Result<Compiled> {
    let source = prop_shader_source(user)?;
    let fs = crate::gpu::compile_shader(&source, Stage::Fragment, "fs_prop_shader")?;
    if sorted_bindings(&fs.bindings) != pipelines.prop_shader_bindings {
        return Err(
            "a prop shader reads only its material's textures and the draw's parameters \
             (`ShadeInput`); pass anything else, the time among it, in `params`"
                .into(),
        );
    }
    Ok(fs)
}

/// The bind group a prop's material binds (its textures and sampler).
const MATERIAL_GROUP: u32 = 1;

/// Bindings by group and binding, for comparing stages.
fn sorted_bindings(bindings: &[ganbate_shader::Binding]) -> Vec<ganbate_shader::Binding> {
    let mut out = bindings.to_vec();
    out.sort_by_key(|b| (b.group, b.binding));
    out
}

/// One overlay pipeline of a game's prop shader: `blend` 0 opaque, 1 alpha, 2 additive; `depth` 0
/// drawn on top, 1 testing the scene's depth, 2 testing and writing it (opaque).
pub(super) fn prop_shader_pipeline(
    device: &mut Device<'_>,
    pipelines: &Pipelines,
    fs: &Compiled,
    blend: usize,
    depth: usize,
) -> Result<PipelineId> {
    let blend = [Blend::Replace, Blend::Premultiplied, Blend::Additive]
        .get(blend)
        .copied()
        .ok_or("prop_shader_pipeline: no such blend")?;
    device.create_render_pipeline(&RenderPipelineDesc {
        vertex: &pipelines.prop_vs,
        fragment: Some(fs),
        vertex_layout: Some(pipelines.prop_layout.clone()),
        topology: Topology::TriangleList,
        cull: Cull::None,
        front_ccw: true,
        depth: Some(DepthState {
            format: Format::Depth32Float,
            compare: if depth > 0 {
                Compare::Greater
            } else {
                Compare::Always
            },
            write: depth == 2,
            bias_constant: 0.0,
            bias_slope: 0.0,
            clamp: false,
        }),
        colors: vec![color(pipelines.output_format, blend)],
    })
}

fn color(format: Format, blend: Blend) -> ColorTarget {
    ColorTarget {
        format,
        blend,
        write_mask: 0xF,
    }
}

fn fullscreen(
    device: &mut Device<'_>,
    vs: &Compiled,
    entry: &str,
    format: Format,
    blend: Blend,
) -> Result<PipelineId> {
    let fs = compile(POST, Stage::Fragment, entry)?;
    device.create_render_pipeline(&RenderPipelineDesc {
        vertex: vs,
        fragment: Some(&fs),
        vertex_layout: None,
        topology: Topology::TriangleList,
        cull: Cull::None,
        front_ccw: true,
        depth: None,
        colors: vec![color(format, blend)],
    })
}

impl Pipelines {
    fn new(device: &mut Device<'_>, output: Format) -> Result<Self> {
        let layout = VertexLayout {
            stride: 32,
            attributes: vec![
                VertexAttribute {
                    location: 0,
                    format: VertexFormat::F32x3,
                    offset: 0,
                },
                VertexAttribute {
                    location: 1,
                    format: VertexFormat::F32x3,
                    offset: 12,
                },
                VertexAttribute {
                    location: 2,
                    format: VertexFormat::F32x2,
                    offset: 24,
                },
            ],
        };
        let world_layout = VertexLayout {
            stride: 36,
            attributes: vec![
                layout.attributes[0],
                layout.attributes[1],
                layout.attributes[2],
                VertexAttribute {
                    location: 3,
                    format: VertexFormat::F32,
                    offset: 32,
                },
            ],
        };
        let reverse_depth = |compare, write| DepthState {
            format: Format::Depth32Float,
            compare,
            write,
            bias_constant: 0.0,
            bias_slope: 0.0,
            clamp: false,
        };
        let mrt = || vec![color(HDR, Blend::Replace), color(HDR, Blend::Replace)];
        let vs = compile(SCENE, Stage::Vertex, "vs_main")?;
        let fs = compile(SCENE, Stage::Fragment, "fs_main")?;
        let scene = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&fs),
            vertex_layout: Some(world_layout.clone()),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(reverse_depth(Compare::Greater, true)),
            colors: mrt(),
        })?;
        let opaque_fs = compile(SCENE, Stage::Fragment, "fs_main_opaque")?;
        let scene_opaque = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&opaque_fs),
            vertex_layout: Some(world_layout),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(reverse_depth(Compare::Greater, true)),
            colors: mrt(),
        })?;
        let vs = compile(SCENE, Stage::Vertex, "vs_prop")?;
        let prop_vs = vs.clone();
        let scene_prop = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&fs),
            vertex_layout: Some(layout.clone()),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(reverse_depth(Compare::Greater, true)),
            colors: mrt(),
        })?;
        let scene_prop_opaque = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&opaque_fs),
            vertex_layout: Some(layout.clone()),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(reverse_depth(Compare::Greater, true)),
            colors: mrt(),
        })?;
        // Blended props leave the aux target (normals for outlines and occlusion) alone.
        let mut scene_prop_blend = [scene_prop; 2];
        for (i, blend) in [Blend::Premultiplied, Blend::Additive]
            .into_iter()
            .enumerate()
        {
            scene_prop_blend[i] = device.create_render_pipeline(&RenderPipelineDesc {
                vertex: &vs,
                fragment: Some(&fs),
                vertex_layout: Some(layout.clone()),
                topology: Topology::TriangleList,
                cull: Cull::None,
                front_ccw: true,
                depth: Some(reverse_depth(Compare::Greater, false)),
                colors: vec![
                    color(HDR, blend),
                    ColorTarget {
                        format: HDR,
                        blend: Blend::Replace,
                        write_mask: 0,
                    },
                ],
            })?;
        }
        let overlay_fs = compile(SCENE, Stage::Fragment, "fs_overlay")?;
        let prop_shader_bindings = sorted_bindings(
            &overlay_fs
                .bindings
                .iter()
                .filter(|b| b.group != MATERIAL_GROUP)
                .chain(fs.bindings.iter().filter(|b| b.group == MATERIAL_GROUP))
                .copied()
                .collect::<Vec<_>>(),
        );
        let mut overlay = [[scene_prop; 3]; 3];
        for (i, blend) in [Blend::Replace, Blend::Premultiplied, Blend::Additive]
            .into_iter()
            .enumerate()
        {
            for (j, (compare, write)) in [
                (Compare::Always, false),
                (Compare::Greater, false),
                (Compare::Greater, true),
            ]
            .into_iter()
            .enumerate()
            {
                // Only an opaque overlay writes depth.
                if write && i > 0 {
                    overlay[i][j] = overlay[i][1];
                    continue;
                }
                overlay[i][j] = device.create_render_pipeline(&RenderPipelineDesc {
                    vertex: &vs,
                    fragment: Some(&overlay_fs),
                    vertex_layout: Some(layout.clone()),
                    topology: Topology::TriangleList,
                    cull: Cull::None,
                    front_ccw: true,
                    depth: Some(reverse_depth(compare, write)),
                    colors: vec![color(output, blend)],
                })?;
            }
        }
        let shadow_depth = DepthState {
            format: Format::Depth32Float,
            compare: Compare::Less,
            write: true,
            bias_constant: 1.0,
            bias_slope: 1.75,
            clamp: true,
        };
        let vs = compile(WORLD_FX, Stage::Vertex, "vs_shadow_prop")?;
        let shadow_prop = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: None,
            vertex_layout: Some(VertexLayout {
                stride: 32,
                attributes: vec![layout.attributes[0]],
            }),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(shadow_depth),
            colors: vec![],
        })?;
        let vs = compile(WORLD_FX, Stage::Vertex, "vs_shadow_prop_masked")?;
        let fs = compile(WORLD_FX, Stage::Fragment, "fs_shadow_prop_masked")?;
        let shadow_prop_masked = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&fs),
            vertex_layout: Some(VertexLayout {
                stride: 32,
                attributes: vec![layout.attributes[0], layout.attributes[2]],
            }),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(shadow_depth),
            colors: vec![],
        })?;
        let vs = compile(WORLD_FX, Stage::Vertex, "vs_shadow")?;
        let shadow = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: None,
            vertex_layout: Some(VertexLayout {
                stride: 36,
                attributes: vec![layout.attributes[0]],
            }),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(DepthState {
                format: Format::Depth32Float,
                compare: Compare::Less,
                write: true,
                bias_constant: 1.0,
                bias_slope: 1.75,
                clamp: true,
            }),
            colors: vec![],
        })?;
        let vs = compile(WORLD_FX, Stage::Vertex, "vs_shadow_masked")?;
        let fs = compile(WORLD_FX, Stage::Fragment, "fs_shadow_masked")?;
        let shadow_masked = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&fs),
            vertex_layout: Some(VertexLayout {
                stride: 36,
                attributes: vec![layout.attributes[0], layout.attributes[2]],
            }),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(DepthState {
                format: Format::Depth32Float,
                compare: Compare::Less,
                write: true,
                bias_constant: 1.0,
                bias_slope: 1.75,
                clamp: true,
            }),
            colors: vec![],
        })?;
        let vs = compile(WORLD_FX, Stage::Vertex, "vs_ground")?;
        let fs = compile(WORLD_FX, Stage::Fragment, "fs_ground")?;
        let ground = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&fs),
            vertex_layout: Some(VertexLayout {
                stride: 36,
                attributes: vec![layout.attributes[0], layout.attributes[1]],
            }),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(DepthState {
                format: Format::Depth32Float,
                compare: Compare::Less,
                write: true,
                bias_constant: 0.0,
                bias_slope: 0.0,
                clamp: true,
            }),
            colors: vec![color(GROUND_FORMAT, Blend::Replace)],
        })?;
        let blades = |device: &mut Device<'_>, vs: &str, fs: &str| -> Result<PipelineId> {
            let vs = compile(SCENE, Stage::Vertex, vs)?;
            let fs = compile(SCENE, Stage::Fragment, fs)?;
            device.create_render_pipeline(&RenderPipelineDesc {
                vertex: &vs,
                fragment: Some(&fs),
                vertex_layout: None,
                topology: Topology::TriangleList,
                cull: Cull::None,
                front_ccw: true,
                depth: Some(reverse_depth(Compare::Greater, true)),
                colors: mrt(),
            })
        };
        let grass = blades(device, "vs_grass", "fs_grass")?;
        let flower = blades(device, "vs_flower", "fs_flower")?;
        let vs = compile(WORLD_FX, Stage::Vertex, "vs_sky")?;
        let fs = compile(WORLD_FX, Stage::Fragment, "fs_sky")?;
        let sky = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&fs),
            vertex_layout: None,
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(reverse_depth(Compare::GreaterEqual, false)),
            colors: mrt(),
        })?;
        let vs = compile(WORLD_FX, Stage::Vertex, "vs_particle")?;
        let fs = compile(WORLD_FX, Stage::Fragment, "fs_particle")?;
        let particle = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&fs),
            vertex_layout: None,
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(reverse_depth(Compare::Greater, false)),
            colors: vec![
                color(HDR, Blend::Alpha),
                ColorTarget {
                    format: HDR,
                    blend: Blend::Replace,
                    write_mask: 0,
                },
            ],
        })?;
        let fsvs = compile(POST, Stage::Vertex, "vs_fullscreen")?;
        Ok(Self {
            scene,
            scene_opaque,
            scene_prop,
            scene_prop_opaque,
            scene_prop_blend,
            overlay,
            output_format: output,
            prop_layout: layout,
            prop_shader_bindings,
            prop_vs,
            shadow,
            shadow_masked,
            shadow_prop,
            shadow_prop_masked,
            ground,
            grass,
            flower,
            sky,
            particle,
            ssao: fullscreen(device, &fsvs, "fs_ssao", Format::R16Float, Blend::Replace)?,
            air: fullscreen(device, &fsvs, "fs_air", HDR, Blend::Replace)?,
            bloom_down: fullscreen(device, &fsvs, "fs_bloom_down", HDR, Blend::Replace)?,
            bloom_up: fullscreen(device, &fsvs, "fs_bloom_up", HDR, Blend::Additive)?,
            tonemap: fullscreen(device, &fsvs, "fs_tonemap", HDR, Blend::Replace)?,
            tonemap_final: fullscreen(device, &fsvs, "fs_tonemap_final", output, Blend::Replace)?,
            paint: fullscreen(device, &fsvs, "fs_paint", HDR, Blend::Replace)?,
            output: fullscreen(device, &fsvs, "fs_final", output, Blend::Replace)?,
        })
    }
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
pub(super) struct Eye {
    pub(super) view_proj: M4,
    pub(super) inv: M4,
    pos: V3,
    near: f32,
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
pub(super) fn detail_block(look: &Look, name: &str) -> ([f32; 16], bool) {
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

impl Renderer {
    pub fn new(
        device: &mut Device<'_>,
        world: &crate::world::World,
        look: Look,
        quality: Quality,
        size: (u32, u32),
        output: Output,
    ) -> Result<Self> {
        let out_format = match output {
            Output::Still => HDR,
            Output::Surface(f) => f,
        };
        let pipelines = Pipelines::new(device, out_format)?;
        let scene = GpuScene::upload(device, world)?;
        let sampled = TextureUsage::SAMPLED | TextureUsage::COPY_DST;
        let solid = |device: &mut Device<'_>, format, texel: [u8; 4]| -> Result<TextureId> {
            let t = device.create_texture(&TextureDesc::new(1, 1, format, sampled))?;
            device.write_texture(t, 0, 0, &texel)?;
            Ok(t)
        };
        let white = solid(device, Format::Rgba8Srgb, [255; 4])?;
        let flat = solid(device, Format::Rgba8Unorm, [128, 128, 255, 255])?;
        let dummy = solid(device, Format::Rgba8Unorm, [0; 4])?;
        let material_sampler = device.create_sampler(&SamplerDesc {
            anisotropy: 8.0,
            ..SamplerDesc::default()
        })?;
        let linear_clamp = device.create_sampler(&SamplerDesc {
            address: Address::Clamp,
            ..SamplerDesc::default()
        })?;
        let shadow_sampler = device.create_sampler(&SamplerDesc {
            address: Address::Clamp,
            compare: Some(Compare::LessEqual),
            mip_filter: Filter::Nearest,
            ..SamplerDesc::default()
        })?;
        let mut materials = vec![];
        let mut shadow_materials = vec![];
        for m in &scene.materials {
            shadow_materials.push(if m.params[0] >= 0.0 {
                Some(device.create_bind_group(
                    pipelines.shadow_masked,
                    1,
                    &[
                        (
                            0,
                            Resource::Texture(m.base.map_or(white, |i| scene.textures[i])),
                        ),
                        (3, Resource::Sampler(material_sampler)),
                    ],
                )?)
            } else {
                None
            });
            let tex = |i: Option<usize>, fallback| i.map_or(fallback, |i| scene.textures[i]);
            materials.push(device.create_bind_group(
                pipelines.scene,
                1,
                &[
                    (0, Resource::Texture(tex(m.base, white))),
                    (1, Resource::Texture(tex(m.normal, flat))),
                    (2, Resource::Texture(tex(m.emission, white))),
                    (3, Resource::Sampler(material_sampler)),
                ],
            )?);
        }
        let shadow_size = quality.shadow_size.max(16);
        let shadow_map = device.create_texture(&TextureDesc {
            width: shadow_size,
            height: shadow_size,
            layers: 4,
            mips: 1,
            format: Format::Depth32Float,
            usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
        })?;
        let light_shadow_map = device.create_texture(&TextureDesc {
            width: quality.light_shadow_size.max(16),
            height: quality.light_shadow_size.max(16),
            layers: super::lights::MAX_SHADOWED_LIGHTS as u32,
            mips: 1,
            format: Format::Depth32Float,
            usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
        })?;
        let ground_size = quality.ground_size.max(64);
        let ground_depth = device.create_texture(&TextureDesc {
            width: ground_size,
            height: ground_size,
            layers: 1,
            mips: 1,
            format: Format::Depth32Float,
            usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
        })?;
        let ground_mask = device.create_texture(&TextureDesc {
            width: ground_size,
            height: ground_size,
            layers: 1,
            mips: 1,
            format: GROUND_FORMAT,
            usage: TextureUsage::COLOR_TARGET | TextureUsage::SAMPLED,
        })?;
        let world_size = quality.world_size.max(64);
        let world_depth = device.create_texture(&TextureDesc {
            width: world_size,
            height: world_size,
            layers: 1,
            mips: 1,
            format: Format::Depth32Float,
            usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
        })?;
        let world_mask = device.create_texture(&TextureDesc {
            width: world_size,
            height: world_size,
            layers: 1,
            mips: 1,
            format: GROUND_FORMAT,
            usage: TextureUsage::COLOR_TARGET | TextureUsage::SAMPLED,
        })?;
        let world = world_view(&scene, world_size);
        let (details, grassy) = material_details(&look, &scene);
        let lut = device.create_texture(&TextureDesc::new(
            LUT_SIZE * LUT_SIZE,
            LUT_SIZE,
            Format::Rgba8Unorm,
            sampled,
        ))?;
        device.write_texture(lut, 0, 0, &bake_lut(&look))?;
        let targets = Self::create_targets(device, &quality, output, size)?;
        let mut r = Self {
            scene,
            pipelines,
            quality,
            output,
            look,
            targets,
            slots: vec![],
            materials,
            shadow_materials,
            shadow_map,
            shadow_layers_ready: false,
            static_shadows_vp: None,
            light_shadow_map,
            light_shadows_ready: false,
            ground_depth,
            ground_mask,
            ground_cached_vp: None,
            world_depth,
            world_mask,
            world,
            world_drawn: false,
            world_moved: false,
            ao_quiet_frames: AO_SETTLE_FRAMES,
            details,
            grassy,
            lut,
            lut_dirty: false,
            dummy,
            linear_clamp,
            shadow_sampler,
            white,
            flat,
            material_sampler,
            props: super::props::Props::default(),
            game_lights: super::lights::GameLights::default(),
            models: super::models::Models::default(),
            stats: FrameStats::default(),
        };
        let slots = match output {
            Output::Still => 1,
            Output::Surface(_) => 2,
        };
        for _ in 0..slots {
            let s = r.create_slot(device)?;
            r.slots.push(s);
        }
        Ok(r)
    }

    pub fn look(&self) -> &Look {
        &self.look
    }
    pub fn set_look(&mut self, look: Look) {
        (self.details, self.grassy) = material_details(&look, &self.scene);
        self.look = look;
        self.world_drawn = false;
        self.ground_cached_vp = None;
        self.lut_dirty = true;
        self.restyle_props();
    }
    /// Current per-frame rendering quality budget.
    pub fn quality(&self) -> Quality {
        self.quality
    }
    /// Visibility, cache, and pass-skip telemetry recorded during the most recent [`Renderer::encode`].
    pub fn last_frame_stats(&self) -> FrameStats {
        self.stats
    }
    /// Advances the baked vertex AO by up to `samples` vertices' worth of work outside a frame
    /// (reading or building occluders first, then sampling nearest `eye`), so a host can bake in
    /// idle time with its own clock. Returns whether there was work to do: false once converged,
    /// and while world edits have not yet paused.
    pub fn step_ao(&mut self, device: &mut Device<'_>, samples: usize, eye: V3) -> Result<bool> {
        if self.ao_quiet_frames < AO_SETTLE_FRAMES || self.scene.ao.is_idle() {
            return Ok(false);
        }
        self.scene.step_ao(device, samples, Some(eye))?;
        Ok(true)
    }

    /// Completes the baked vertex AO now (a capture that must not depend on frame count).
    pub fn finish_ao(&mut self, device: &mut Device<'_>) -> Result<()> {
        self.scene.finish_ao(device).map(|_| ())
    }
    /// Converged fraction of the baked vertex AO in `[0, 1]` (it refines a slice per live
    /// frame after start and after world edits).
    pub fn ao_progress(&self) -> f32 {
        self.scene.ao.progress()
    }
    pub fn size(&self) -> (u32, u32) {
        (self.targets.width, self.targets.height)
    }
    pub fn triangles(&self) -> usize {
        self.scene.triangles
    }
    pub fn lights(&self) -> usize {
        self.scene.lights.len()
    }
    pub fn water_level(&self) -> Option<f32> {
        self.scene.water_level
    }
    /// World bounds of everything drawn.
    pub fn bounds(&self) -> (V3, V3) {
        (self.scene.min, self.scene.max)
    }

    fn create_targets(
        device: &mut Device<'_>,
        q: &Quality,
        output: Output,
        (width, height): (u32, u32),
    ) -> Result<Targets> {
        let rt = TextureUsage::COLOR_TARGET | TextureUsage::SAMPLED;
        let tex = |d: &mut Device<'_>, w: u32, h: u32, f: Format, u| {
            d.create_texture(&TextureDesc::new(w.max(1), h.max(1), f, u))
        };
        let scaled = |s: f32| {
            (
                ((width as f32 * s).round() as u32).max(1),
                ((height as f32 * s).round() as u32).max(1),
            )
        };
        let refl = scaled(q.reflection_scale);
        let ao = scaled(q.ao_scale);
        let bloom_size = ((width / 2).max(1), (height / 2).max(1));
        let max_levels = 32 - bloom_size.0.min(bloom_size.1).leading_zeros();
        let bloom_levels = q.bloom_levels.clamp(1, max_levels.max(1));
        let depth_usage = TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED;
        let mut t = Targets {
            width,
            height,
            hdr: tex(device, width, height, HDR, rt)?,
            aux: tex(device, width, height, HDR, rt)?,
            depth: tex(device, width, height, Format::Depth32Float, depth_usage)?,
            refl,
            refl_hdr: tex(device, refl.0, refl.1, HDR, rt)?,
            refl_aux: tex(device, refl.0, refl.1, HDR, rt)?,
            refl_depth: tex(device, refl.0, refl.1, Format::Depth32Float, depth_usage)?,
            refl_cleared: false,
            ao,
            ao_tex: tex(device, ao.0, ao.1, Format::R16Float, rt)?,
            ao_cleared: false,
            lit: tex(device, width, height, HDR, rt)?,
            bloom: device.create_texture(&TextureDesc {
                width: bloom_size.0,
                height: bloom_size.1,
                layers: 1,
                mips: bloom_levels,
                format: HDR,
                usage: rt,
            })?,
            bloom_size,
            bloom_levels,
            ldr: tex(device, width, height, HDR, rt)?,
            paint: tex(device, width, height, HDR, rt)?,
            still: None,
            readback: None,
            tiles: [
                (width.div_ceil(q.tile_size), height.div_ceil(q.tile_size)),
                (refl.0.div_ceil(q.tile_size), refl.1.div_ceil(q.tile_size)),
            ],
        };
        if output == Output::Still {
            t.still = Some(tex(
                device,
                width,
                height,
                HDR,
                rt | TextureUsage::COPY_SRC,
            )?);
            t.readback = Some(device.create_buffer(&BufferDesc {
                size: u64::from(width) * u64::from(height) * 8,
                usage: BufferUsage::COPY_DST,
                memory: Memory::Readback,
            })?);
        }
        Ok(t)
    }

    fn create_slot(&self, device: &mut Device<'_>) -> Result<Slot> {
        let t = &self.targets;
        let p = &self.pipelines;
        let upload = |d: &mut Device<'_>, size: u64, usage| {
            d.create_buffer(&BufferDesc {
                size: size.max(16),
                usage,
                memory: Memory::Upload,
            })
        };
        let mut frame = vec![];
        let mut tiles = vec![];
        let mut index = vec![];
        for v in 0..2 {
            let (tx, ty) = t.tiles[v];
            let n = u64::from(tx) * u64::from(ty);
            frame.push(upload(device, FRAME_BYTES, BufferUsage::UNIFORM)?);
            tiles.push(upload(device, n * 8, BufferUsage::STORAGE)?);
            index.push(upload(
                device,
                n * MAX_LIGHTS_PER_TILE as u64 * 4,
                BufferUsage::STORAGE,
            )?);
        }
        let buf = |id| Resource::Buffer {
            id,
            offset: 0,
            size: 0,
        };
        let mut scene = vec![];
        let mut sky = vec![];
        let mut blades = vec![];
        for v in 0..2 {
            let reflection = if v == 0 { t.refl_hdr } else { self.dummy };
            let entries = [
                (0, buf(frame[v])),
                (1, buf(self.scene.light_buffer)),
                (2, buf(tiles[v])),
                (3, buf(index[v])),
                (4, Resource::Texture(self.shadow_map)),
                (5, Resource::Sampler(self.shadow_sampler)),
                (6, Resource::Texture(reflection)),
                (7, Resource::Sampler(self.linear_clamp)),
                (8, Resource::Texture(self.ground_depth)),
                (9, Resource::Texture(self.ground_mask)),
                (10, Resource::Texture(self.world_depth)),
                (14, Resource::Texture(self.light_shadow_map)),
            ];
            scene.push(device.create_bind_group(p.scene, 0, &entries)?);
            sky.push(device.create_bind_group(p.sky, 0, &entries[..1])?);
            if v == 0 {
                blades.push(device.create_bind_group(p.grass, 0, &entries)?);
                blades.push(device.create_bind_group(p.flower, 0, &entries)?);
                blades.push(device.create_bind_group(p.particle, 0, &entries)?);
            }
        }
        fn pair<T: Copy>(v: Vec<T>) -> [T; 2] {
            [v[0], v[1]]
        }
        let (frame, tiles, index) = (pair(frame), pair(tiles), pair(index));
        let (scene, sky) = (pair(scene), pair(sky));
        let (grass, flower, particle) = (blades[0], blades[1], blades[2]);
        let post = |d: &mut Device<'_>, pipe, src: Resource, extra: Resource| {
            d.create_bind_group(
                pipe,
                0,
                &[
                    (0, buf(frame[0])),
                    (1, Resource::Texture(t.depth)),
                    (2, Resource::Texture(t.aux)),
                    (3, src),
                    (4, extra),
                    (5, Resource::Sampler(self.linear_clamp)),
                ],
            )
        };
        let none = Resource::Texture(self.dummy);
        let ssao = post(device, p.ssao, none, none)?;
        let air = post(
            device,
            p.air,
            Resource::Texture(t.hdr),
            Resource::Texture(t.ao_tex),
        )?;
        let mut down = vec![];
        let mut up = vec![];
        for level in 0..t.bloom_levels {
            let src = if level == 0 {
                Resource::Texture(t.lit)
            } else {
                Resource::TextureMip(t.bloom, level - 1)
            };
            down.push(post(device, p.bloom_down, src, none)?);
            if level + 1 < t.bloom_levels {
                up.push(post(
                    device,
                    p.bloom_up,
                    Resource::TextureMip(t.bloom, level + 1),
                    none,
                )?);
            }
        }
        let tonemap = post(
            device,
            p.tonemap,
            Resource::Texture(t.lit),
            Resource::TextureMip(t.bloom, 0),
        )?;
        let tonemap_final = device.create_bind_group(
            p.tonemap_final,
            0,
            &[
                (0, buf(frame[0])),
                (1, Resource::Texture(t.depth)),
                (2, Resource::Texture(t.aux)),
                (3, Resource::Texture(t.lit)),
                (4, Resource::TextureMip(t.bloom, 0)),
                (5, Resource::Sampler(self.linear_clamp)),
                (6, Resource::Texture(self.lut)),
            ],
        )?;
        let paint = post(device, p.paint, Resource::Texture(t.ldr), none)?;
        let final_src = if self.quality.painterly {
            t.paint
        } else {
            t.ldr
        };
        let output = post(device, p.output, Resource::Texture(final_src), none)?;
        Ok(Slot {
            frame,
            tiles,
            index,
            scene,
            sky,
            grass,
            flower,
            particle,
            ssao,
            air,
            down,
            up,
            tonemap,
            tonemap_final,
            paint,
            output,
        })
    }

    fn destroy_slots(device: &mut Device<'_>, slots: Vec<Slot>) {
        for s in slots {
            for g in s
                .scene
                .into_iter()
                .chain(s.sky)
                .chain([s.grass, s.flower, s.particle, s.ssao, s.air])
                .chain([s.tonemap, s.tonemap_final, s.paint, s.output])
            {
                device.destroy_bind_group(g);
            }
            for g in s.down.into_iter().chain(s.up) {
                device.destroy_bind_group(g);
            }
            for b in s.frame.into_iter().chain(s.tiles).chain(s.index) {
                device.destroy_buffer(b);
            }
        }
    }

    fn destroy_targets(device: &mut Device<'_>, old: Targets) {
        for t in [
            old.hdr,
            old.aux,
            old.depth,
            old.refl_hdr,
            old.refl_aux,
            old.refl_depth,
            old.ao_tex,
            old.lit,
            old.bloom,
            old.ldr,
            old.paint,
        ]
        .into_iter()
        .chain(old.still)
        {
            device.destroy_texture(t);
        }
        if let Some(b) = old.readback {
            device.destroy_buffer(b);
        }
    }

    /// Recreate size-dependent targets (e.g. after a window resize).
    pub fn resize(&mut self, device: &mut Device<'_>, size: (u32, u32)) -> Result<()> {
        if size == (self.targets.width, self.targets.height) {
            return Ok(());
        }
        let new_targets = Self::create_targets(device, &self.quality, self.output, size)?;
        self.reset_prop_slots(device);
        Self::destroy_slots(device, std::mem::take(&mut self.slots));
        let old = std::mem::replace(&mut self.targets, new_targets);
        Self::destroy_targets(device, old);
        let slots = match self.output {
            Output::Still => 1,
            Output::Surface(_) => 2,
        };
        for _ in 0..slots {
            let s = self.create_slot(device)?;
            self.slots.push(s);
        }
        Ok(())
    }

    /// Reconfigure quality-dependent targets, shadow maps, and ground/world caches at runtime
    /// without losing uploaded world geometry, props, skinned models, lights, or [`Look`] state.
    pub fn set_quality(&mut self, device: &mut Device<'_>, quality: Quality) -> Result<()> {
        if quality == self.quality {
            return Ok(());
        }
        let size = (self.targets.width, self.targets.height);
        let new_targets = Self::create_targets(device, &quality, self.output, size)?;

        let old_shadow_size = self.quality.shadow_size.max(16);
        let new_shadow_size = quality.shadow_size.max(16);
        let new_shadow_map = if new_shadow_size != old_shadow_size {
            Some(device.create_texture(&TextureDesc {
                width: new_shadow_size,
                height: new_shadow_size,
                layers: 4,
                mips: 1,
                format: Format::Depth32Float,
                usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
            })?)
        } else {
            None
        };

        let old_light_shadow_size = self.quality.light_shadow_size.max(16);
        let new_light_shadow_size = quality.light_shadow_size.max(16);
        let new_light_shadow_map = if new_light_shadow_size != old_light_shadow_size {
            Some(device.create_texture(&TextureDesc {
                width: new_light_shadow_size,
                height: new_light_shadow_size,
                layers: super::lights::MAX_SHADOWED_LIGHTS as u32,
                mips: 1,
                format: Format::Depth32Float,
                usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
            })?)
        } else {
            None
        };

        let old_ground_size = self.quality.ground_size.max(64);
        let new_ground_size = quality.ground_size.max(64);
        let new_ground = if new_ground_size != old_ground_size {
            let depth = device.create_texture(&TextureDesc {
                width: new_ground_size,
                height: new_ground_size,
                layers: 1,
                mips: 1,
                format: Format::Depth32Float,
                usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
            })?;
            let mask = device.create_texture(&TextureDesc {
                width: new_ground_size,
                height: new_ground_size,
                layers: 1,
                mips: 1,
                format: GROUND_FORMAT,
                usage: TextureUsage::COLOR_TARGET | TextureUsage::SAMPLED,
            })?;
            Some((depth, mask))
        } else {
            None
        };

        let old_world_size = self.quality.world_size.max(64);
        let new_world_size = quality.world_size.max(64);
        let new_world = if new_world_size != old_world_size {
            let depth = device.create_texture(&TextureDesc {
                width: new_world_size,
                height: new_world_size,
                layers: 1,
                mips: 1,
                format: Format::Depth32Float,
                usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
            })?;
            let mask = device.create_texture(&TextureDesc {
                width: new_world_size,
                height: new_world_size,
                layers: 1,
                mips: 1,
                format: GROUND_FORMAT,
                usage: TextureUsage::COLOR_TARGET | TextureUsage::SAMPLED,
            })?;
            Some((depth, mask))
        } else {
            None
        };

        self.reset_prop_slots(device);
        Self::destroy_slots(device, std::mem::take(&mut self.slots));
        let old_targets = std::mem::replace(&mut self.targets, new_targets);
        Self::destroy_targets(device, old_targets);

        if let Some(shadow_map) = new_shadow_map {
            let old = std::mem::replace(&mut self.shadow_map, shadow_map);
            device.destroy_texture(old);
            self.shadow_layers_ready = false;
            self.static_shadows_vp = None;
        }
        if let Some(light_shadow_map) = new_light_shadow_map {
            let old = std::mem::replace(&mut self.light_shadow_map, light_shadow_map);
            device.destroy_texture(old);
            self.light_shadows_ready = false;
        }
        if let Some((depth, mask)) = new_ground {
            let old_depth = std::mem::replace(&mut self.ground_depth, depth);
            let old_mask = std::mem::replace(&mut self.ground_mask, mask);
            device.destroy_texture(old_depth);
            device.destroy_texture(old_mask);
            self.ground_cached_vp = None;
        }
        if let Some((depth, mask)) = new_world {
            let old_depth = std::mem::replace(&mut self.world_depth, depth);
            let old_mask = std::mem::replace(&mut self.world_mask, mask);
            device.destroy_texture(old_depth);
            device.destroy_texture(old_mask);
            self.world = world_view(&self.scene, new_world_size);
            self.world_drawn = false;
        }

        self.quality = quality;
        let slots = match self.output {
            Output::Still => 1,
            Output::Surface(_) => 2,
        };
        for _ in 0..slots {
            let s = self.create_slot(device)?;
            self.slots.push(s);
        }
        Ok(())
    }

    pub(super) fn eye(&self, view: &View, jitter: [f32; 2], (w, h): (u32, u32)) -> Eye {
        let v = look_at(view.eye, view.target, camera_up(view));
        let mut p = perspective(view.fov.to_radians(), w as f32 / h as f32, near(view));
        // Column 2 skews the frustum: x_ndc gains -p[8] and y_ndc (pointing down) -p[9].
        p[8] += 2.0 * jitter[0] / w as f32 - view.shift[0];
        p[9] += 2.0 * jitter[1] / h as f32 + view.shift[1];
        let view_proj = mul(&p, &v);
        Eye {
            view_proj,
            inv: invert(&view_proj).unwrap_or(math::IDENTITY),
            pos: view.eye,
            near: near(view),
        }
    }

    /// The ground map's top-down orthographic projection: centred on the view target (at most
    /// 40 m ahead of the eye, so the camera is always inside the map), depth 0 above the
    /// highest point of the world and 1 below the lowest. The centre snaps to whole texels so
    /// grass, rain cover and foam do not shimmer as the camera moves.
    fn ground_view(&self, view: &View) -> Ground {
        let to = sub(view.target, view.eye);
        let d = math::length(to);
        let centre = if d <= 40.0 {
            view.target
        } else {
            add(view.eye, scale(to, 40.0 / d))
        };
        let size = self.quality.ground_size.max(64);
        let texel = 2.0 * GROUND_REACH / size as f32;
        let (x, z) = (
            (centre[0] / texel).round() * texel,
            (centre[2] / texel).round() * texel,
        );
        let top = self.scene.max[1] + 1.0;
        let range = (top - self.scene.min[1] + 1.0).max(2.0);
        let v = look_at([x, top, z], [x, top - 1.0, z], [0.0, 0.0, -1.0]);
        let r = GROUND_REACH;
        Ground {
            vp: mul(&ortho(-r, r, -r, r, 0.0, range), &v),
            top,
            range,
            size,
        }
    }

    fn cascades(&self, view: &View, env: &Environment) -> Cascades {
        let (w, h) = (self.targets.width as f32, self.targets.height as f32);
        let focus = math::length(sub(view.target, view.eye)).max(1.0);
        let split = (focus * 1.5).clamp(18.0, 140.0);
        let far = (focus * 4.5).clamp(90.0, 420.0);
        let l = normalize(env.key_dir);
        let size = self.quality.shadow_size as f32;
        let fwd = normalize(sub(view.target, view.eye));
        let right = normalize(cross(fwd, camera_up(view)));
        let up = cross(right, fwd);
        let ty = (0.5 * view.fov.to_radians()).tan();
        let tx = ty * w / h;
        let [shift_x, shift_y] = view.shift;
        let mut out = Cascades {
            mats: [math::IDENTITY; 2],
            split,
            offset: [0.0; 2],
            radius: [0.0; 2],
            bias: [0.0; 2],
        };
        // Water casts no shadow, so a wide sea does not stretch the depth range.
        let (mut smin, mut smax) = (self.scene.land_min, self.scene.land_max);
        if let Some((lo, hi)) = self.props_bounds() {
            for k in 0..3 {
                smin[k] = smin[k].min(lo[k]);
                smax[k] = smax[k].max(hi[k]);
            }
        }
        for (c, (d0, d1)) in [(near(view), split), (split * 0.85, far)]
            .into_iter()
            .enumerate()
        {
            let mut pts = vec![];
            for d in [d0, d1] {
                for sx in [-1.0, 1.0] {
                    for sy in [-1.0, 1.0] {
                        pts.push(add(
                            add(view.eye, scale(fwd, d)),
                            add(
                                scale(right, (sx - shift_x) * d * tx),
                                scale(up, (sy - shift_y) * d * ty),
                            ),
                        ));
                    }
                }
            }
            let centre = scale(pts.iter().fold([0.0; 3], |a, p| add(a, *p)), 1.0 / 8.0);
            let radius = pts
                .iter()
                .map(|p| math::length(sub(*p, centre)))
                .fold(0.0f32, f32::max)
                .max(1.0);
            let up_hint = if l[1].abs() > 0.98 {
                [0.0, 0.0, 1.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let back = 1000.0;
            // The light's view is anchored at the world origin, so it only changes when the
            // light moves. The cascade centre is then snapped to whole texels in that frame,
            // and the radius to a fixed step, so shadows don't crawl as the camera moves.
            let lv = look_at(scale(l, back), [0.0; 3], up_hint);
            let radius = (radius * 2.0).ceil() * 0.5;
            let texel = 2.0 * radius / size;
            let lc = transform(&lv, [centre[0], centre[1], centre[2], 1.0]);
            let snap = |x: f32| (x / texel).round() * texel - x;
            let (dx, dy) = (snap(lc[0]), snap(lc[1]));
            // Depth range from the scene bounds along the light.
            let (mut zmin, mut zmax) = (f32::INFINITY, f32::NEG_INFINITY);
            for i in 0..8 {
                let p = [
                    if i & 1 == 0 { smin[0] } else { smax[0] },
                    if i & 2 == 0 { smin[1] } else { smax[1] },
                    if i & 4 == 0 { smin[2] } else { smax[2] },
                ];
                let z = -transform(&lv, [p[0], p[1], p[2], 1.0])[2];
                zmin = zmin.min(z);
                zmax = zmax.max(z);
            }
            let near = (zmin - 5.0).max(1.0);
            let farz = (zmax + 5.0).max(near + 1.0);
            let proj = ortho(
                lc[0] + dx - radius,
                lc[0] + dx + radius,
                lc[1] + dy - radius,
                lc[1] + dy + radius,
                near,
                farz,
            );
            out.mats[c] = mul(&proj, &lv);
            out.offset[c] = texel * 1.5;
            let filter_texels = if c == 0 { 1.6 } else { 1.1 };
            out.radius[c] = filter_texels * env.shadow_softness / size;
            out.bias[c] = (texel * 0.6) / (farz - near);
        }
        out
    }

    /// Light lists per screen tile for one camera. Lights are sorted nearest first so a full
    /// tile keeps the lights that matter most.
    fn tile_lights(
        &self,
        eye: &Eye,
        (w, h): (u32, u32),
        (tx, ty): (u32, u32),
        on: bool,
    ) -> (Vec<u32>, Vec<u32>) {
        let ts = self.quality.tile_size as f32;
        let mut lists: Vec<Vec<u32>> = vec![vec![]; (tx * ty) as usize];
        // The lamps while they are lit, and a game's lights (after them in the light buffer):
        // those that follow the lamps while they are lit, the rest always.
        let lamps = self.scene.lights.len();
        let game = self.game_lights.records();
        let mut order: Vec<usize> = (0..lamps)
            .filter(|_| on)
            .chain(
                game.iter()
                    .enumerate()
                    .filter(|(_, l)| on || l[13] > 0.5)
                    .map(|(k, _)| lamps + k),
            )
            .collect();
        let light = |i: usize| -> &[f32; 16] {
            if i < lamps {
                &self.scene.lights[i]
            } else {
                &game[i - lamps]
            }
        };
        if !order.is_empty() {
            let dist = |i: usize| {
                let l = light(i);
                math::length(sub([l[0], l[1], l[2]], eye.pos))
            };
            order.sort_by(|a, b| dist(*a).total_cmp(&dist(*b)));
            for i in order {
                let l = light(i);
                let r = l[3];
                let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                let mut behind = 0;
                let mut straddle = false;
                for k in 0..8 {
                    let p = [
                        l[0] + if k & 1 == 0 { -r } else { r },
                        l[1] + if k & 2 == 0 { -r } else { r },
                        l[2] + if k & 4 == 0 { -r } else { r },
                    ];
                    let c = transform(&eye.view_proj, [p[0], p[1], p[2], 1.0]);
                    if c[3] <= eye.near * 0.5 {
                        behind += 1;
                        straddle = true;
                        continue;
                    }
                    let (sx, sy) = (c[0] / c[3], c[1] / c[3]);
                    x0 = x0.min(sx);
                    x1 = x1.max(sx);
                    y0 = y0.min(sy);
                    y1 = y1.max(sy);
                }
                if behind == 8 {
                    continue;
                }
                if straddle {
                    (x0, y0, x1, y1) = (-1.0, -1.0, 1.0, 1.0);
                }
                if x1 < -1.0 || y1 < -1.0 || x0 > 1.0 || y0 > 1.0 {
                    continue;
                }
                let px = |v: f32, size: u32| ((v * 0.5 + 0.5) * size as f32 / ts).floor();
                let ax = px(x0.max(-1.0), w).max(0.0) as u32;
                let bx = (px(x1.min(1.0), w) as u32).min(tx - 1);
                let ay = px(y0.max(-1.0), h).max(0.0) as u32;
                let by = (px(y1.min(1.0), h) as u32).min(ty - 1);
                for y in ay..=by {
                    for x in ax..=bx {
                        let list = &mut lists[(y * tx + x) as usize];
                        if list.len() < MAX_LIGHTS_PER_TILE {
                            list.push(i as u32);
                        }
                    }
                }
            }
        }
        let mut tiles = Vec::with_capacity(lists.len() * 2);
        let mut index = vec![];
        for list in lists {
            tiles.push(index.len() as u32);
            tiles.push(list.len() as u32);
            index.extend(list);
        }
        if index.is_empty() {
            index.push(0);
        }
        (tiles, index)
    }

    #[allow(clippy::too_many_arguments)]
    fn frame_block(
        &self,
        eye: &Eye,
        env: &Environment,
        cas: &Cascades,
        ground: &Ground,
        (w, h): (u32, u32),
        tiles: (u32, u32),
        reflection: bool,
        has_prop_shadows: bool,
        has_reflection: bool,
        seed: u32,
    ) -> Vec<u8> {
        let l = &self.look;
        let mut b = Block(Vec::with_capacity(248));
        b.m(&eye.view_proj);
        b.m(&eye.inv);
        b.m(&cas.mats[0]);
        b.m(&cas.mats[1]);
        b.m(&ground.vp);
        b.v(eye.pos, env.seconds);
        b.v(normalize(env.key_dir), env.key_intensity);
        b.v(env.key_color, 0.0);
        b.v(scale(env.ambient_sky, env.ambient), env.lamp_level);
        // Ground ambient plus the key light bounced up off sunlit ground.
        let bounce = scale(env.key_color, env.key_intensity * l.bounce);
        b.v(
            add(scale(env.ambient_ground, env.ambient), bounce),
            env.wetness,
        );
        b.v(l.shadow_tint, l.shadow_strength);
        b.f(
            l.ramp_center,
            l.ramp_softness.max(0.002),
            l.rim_strength,
            l.rim_power,
        );
        b.f(
            l.specular,
            l.specular_size,
            l.foliage_wrap,
            l.foliage_translucency,
        );
        b.f(
            l.texture_softness,
            l.normal_map_strength,
            l.lamp_light,
            l.lamp_glow,
        );
        b.f(
            self.scene.water_level.unwrap_or(-1e6),
            l.water_reflection,
            l.water_ripples,
            env.rain,
        );
        b.v(l.water_color, if reflection { 1.0 } else { 0.0 });
        b.v(env.zenith, env.stars);
        b.v(env.horizon, env.clouds);
        b.v(env.glow, env.sun_disc);
        b.v(normalize(env.glow_dir), env.moon_disc);
        b.v(normalize(env.sun_dir), l.cloud_scale);
        b.v(normalize(env.moon_dir), l.moon_size);
        b.v(env.cloud_lit, l.star_density);
        b.v(env.cloud_shadow, env.petals);
        b.v(env.fog_color, env.fog_density);
        let fog_base = l.fog_height + self.scene.water_level.unwrap_or(0.0);
        b.f(fog_base, l.fog_falloff, env.mist, l.fog);
        b.f(env.wind[0], env.wind[1], 1.0, seed as f32);
        b.f(w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32);
        b.f(cas.split, cas.offset[0], cas.offset[1], eye.near);
        b.f(cas.radius[0], cas.radius[1], cas.bias[0], cas.bias[1]);
        b.f(
            ground.top,
            ground.range,
            ground.size as f32,
            l.grass.density,
        );
        b.f(l.brush, l.brush_scale.max(0.01), l.sun_shafts, 0.0);
        b.v(l.water_shallow, l.foam);
        b.f(
            l.sparkle,
            self.grass_side() as f32,
            if has_prop_shadows { 1.0 } else { 0.0 },
            if has_reflection { 1.0 } else { 0.0 },
        );
        let g = &l.grass;
        b.f(g.height, g.width, g.flowers, g.sway);
        b.v(g.base, 0.0);
        b.v(g.tip, 0.0);
        for c in g.flower_colors {
            b.v(c, 0.0);
        }
        let w = &self.world;
        b.m(&w.vp);
        b.f(
            w.top,
            w.range,
            w.size as f32,
            if self.world_drawn { 1.0 } else { 0.0 },
        );
        let t = &l.terrain;
        b.f(t.beach, t.rock_slope[0], t.rock_slope[1], t.patch_size);
        b.v(t.sand, t.patches);
        b.v(t.wet_sand, 0.0);
        b.v(t.rock, 0.0);
        b.v(t.meadow, 0.0);
        b.f(l.lamp_foliage, l.ambient_occlusion, 0.0, 0.0);
        let tide = if self.scene.water_level.is_some() {
            l.tide
        } else {
            0.0
        };
        let [r, g, bl] = l.tide_color;
        b.f(tide, r, g, bl);
        let mut bytes = f32_bytes(&b.0);
        for u in [
            tiles.0,
            tiles.1,
            self.quality.tile_size,
            (self.scene.lights.len() + self.game_lights.records().len()) as u32,
        ] {
            bytes.extend_from_slice(&u.to_le_bytes());
        }
        // The game lights' shadow maps and unified surface model parameters.
        let mut b = Block(Vec::with_capacity(44));
        let views = self.game_lights.shadow_views();
        for layer in 0..super::lights::MAX_SHADOWED_LIGHTS {
            b.m(views.get(layer).unwrap_or(&math::IDENTITY));
        }
        b.f(
            1.0 / self.quality.light_shadow_size.max(16) as f32,
            0.0,
            0.0,
            0.0,
        );
        b.f(
            l.surface_model as u32 as f32,
            l.clearcoat,
            l.sheen,
            l.subsurface,
        );
        b.v(l.subsurface_color, l.clearcoat_roughness);
        bytes.extend_from_slice(&f32_bytes(&b.0));
        debug_assert!(bytes.len() as u64 <= FRAME_BYTES);
        bytes
    }

    /// Grass instances to draw: the instance grid scaled by quality and the Look's density,
    /// or 0 when no material grows grass.
    fn grass_instances(&self) -> u32 {
        let side = self.grass_side();
        side * side
    }

    /// Rows of the grass instance grid this frame (0 when nothing grows grass).
    fn grass_side(&self) -> u32 {
        let g = &self.look.grass;
        if g.density <= 0.0 || self.quality.grass <= 0.0 || !self.grassy.iter().any(|&x| x) {
            return 0;
        }
        ((GRASS_SIDE as f32 * self.quality.grass.min(1.0).sqrt()).round() as u32 / 2 * 2).max(2)
    }

    /// A ground (or world) map: the highest solid surfaces seen from above through `vp`, and
    /// which of them grow grass. Roofs and solid things count; foliage, alpha-masked cut-outs
    /// and water do not (rain falls through trees, grass grows under them). On terrain, beach
    /// sand and bare rock grow no grass (see `fs_ground`).
    fn draw_ground(&self, enc: &mut Encoder, depth: TextureId, mask: TextureId, vp: &M4) {
        enc.begin_pass(PassDesc {
            colors: vec![Attachment::clear(mask, [0.0; 4])],
            depth: Some(Attachment::clear(depth, [1.0; 4])),
        });
        enc.set_pipeline(self.pipelines.ground);
        enc.set_vertex_buffer(self.scene.vertices, 0);
        enc.set_index_buffer(self.scene.indices, 0);
        let t = &self.look.terrain;
        let mut gp = [0.0f32; 24];
        gp[..16].copy_from_slice(vp);
        gp[17] = 1.0;
        gp[19] = self.scene.water_level.unwrap_or(-1.0e6);
        gp[20..].copy_from_slice(&[t.beach, t.rock_slope[0], t.rock_slope[1], t.patch_size]);
        let terrain = DetailPreset::Terrain as u32 as f32;
        let frustum = Frustum::from_view_proj(vp);
        for d in self.scene.draws.iter() {
            let foliage = self.scene.materials[d.material].params[3] == 1.0;
            if d.water || foliage || self.shadow_materials[d.material].is_some() {
                continue;
            }
            if !frustum.intersects_aabb_clamped(d.bounds.0, d.bounds.1) {
                continue;
            }
            gp[16] = if self.grassy[d.material] { 1.0 } else { 0.0 };
            gp[18] = if self.details[d.material][0] == terrain {
                1.0
            } else {
                0.0
            };
            enc.push_constants(&f32_bytes(&gp));
            enc.draw_indexed(d.count, 1, d.first_index, 0, 0);
        }
        enc.end_pass();
    }

    /// Draws visible static world chunks against `eye`, returning `(total, submitted)` chunk counts.
    /// Draws the visible world draws of `part`: all of them in scene order, or (for the main view)
    /// the opaque ones nearest first and then the rest. The caller binds the pipeline and frame
    /// group for the part.
    fn draw_world(
        &self,
        enc: &mut Encoder,
        model_push: &mut [f32; 32],
        eye: &Eye,
        water: bool,
        part: WorldPart,
    ) -> (u32, u32) {
        enc.set_vertex_buffer(self.scene.vertices, 0);
        enc.set_index_buffer(self.scene.indices, 0);
        let frustum = Frustum::from_view_proj(&eye.view_proj);
        let hybrid = self.quality.ao_samples == 0 && !self.quality.painterly;
        let (mut total, mut submitted) = (0u32, 0u32);
        let mut nearest: Vec<(f32, usize)> = vec![];
        let mut bound = usize::MAX;
        for (i, d) in self.scene.draws.iter().enumerate() {
            if d.water && !water {
                continue;
            }
            // Without an alpha test and off the water, a draw is opaque.
            let opaque = !d.water && self.scene.materials[d.material].params[0] < 0.0;
            match part {
                WorldPart::Opaque if !opaque => continue,
                WorldPart::Rest if opaque => continue,
                _ => {}
            }
            total += 1;
            if !frustum.intersects_aabb(d.bounds.0, d.bounds.1) {
                continue;
            }
            if hybrid && !d.water && d.max_extent < 160.0 {
                let dist = aabb_distance(eye.pos, d.bounds.0, d.bounds.1);
                if (d.max_extent < 2.0 && dist > 140.0) || (d.max_extent < 12.0 && dist > 260.0) {
                    continue;
                }
            }
            submitted += 1;
            if part == WorldPart::Opaque {
                nearest.push((aabb_distance(eye.pos, d.bounds.0, d.bounds.1), i));
            } else {
                self.draw_item(enc, model_push, d, &mut bound);
            }
        }
        // Nearest first: what stands behind then fails the depth test before it is shaded.
        nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, i) in nearest {
            self.draw_item(enc, model_push, &self.scene.draws[i], &mut bound);
        }
        (total, submitted)
    }

    fn draw_item(
        &self,
        enc: &mut Encoder,
        model_push: &mut [f32; 32],
        d: &DrawItem,
        bound: &mut usize,
    ) {
        if *bound != d.material {
            enc.set_bind_group(1, self.materials[d.material]);
            *bound = d.material;
        }
        let m = &self.scene.materials[d.material];
        let detail = &self.details[d.material];
        model_push[..16].copy_from_slice(detail);
        // A colour-only rule (preset `none`) recolours the material whatever its shading,
        // foliage included; patterned presets take the colour in their own way.
        let mut base = m.base_color;
        if detail[0] == 0.0 && detail[8] > 0.5 {
            base[..3].copy_from_slice(&detail[5..8]);
        }
        model_push[16..20].copy_from_slice(&base);
        model_push[20..24].copy_from_slice(&m.emissive);
        model_push[24..28].copy_from_slice(&m.params);
        model_push[28..32].copy_from_slice(&m.flags);
        enc.push_constants(&f32_bytes(model_push));
        enc.draw_indexed(d.count, 1, d.first_index, 0, 0);
    }

    fn post_pass(
        enc: &mut Encoder,
        target: TextureId,
        mip: u32,
        load: Load,
        pipeline: PipelineId,
        group: BindGroupId,
        push: [f32; 16],
    ) {
        enc.begin_pass(PassDesc {
            colors: vec![Attachment {
                texture: target,
                mip,
                layer: 0,
                load,
                store: true,
            }],
            depth: None,
        });
        enc.set_pipeline(pipeline);
        enc.set_bind_group(0, group);
        enc.push_constants(&f32_bytes(&push));
        enc.draw(3, 1, 0, 0);
        enc.end_pass();
    }

    /// Record one frame. `target` is the swapchain texture for [`Output::Surface`] renderers
    /// and ignored for stills (which write their readback target). `jitter` is a sub-pixel
    /// offset in pixels; `seed` varies per accumulated sample.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &mut Device<'_>,
        enc: &mut Encoder,
        view: &View,
        env: &Environment,
        jitter: [f32; 2],
        seed: u32,
        target: Option<TextureId>,
    ) -> Result<()> {
        let slot_index = match self.output {
            Output::Still => 0,
            Output::Surface(_) => device.frame_slot() % self.slots.len(),
        };
        // The world map is drawn in this frame's first passes, before anything reads it. After
        // a move it is drawn again, over the land's new bounds (as a cook of the moved world).
        if std::mem::take(&mut self.world_moved) {
            self.scene.refresh_world();
            self.world = world_view(&self.scene, self.world.size);
            self.world_drawn = false;
            self.static_shadows_vp = None;
            self.ground_cached_vp = None;
            self.ao_quiet_frames = 0;
        } else {
            self.ao_quiet_frames = self.ao_quiet_frames.saturating_add(1);
        }
        // Baked AO: a still completes it before drawing; a live view refines it a slice per
        // frame, nearest the camera first, once edits have paused (a drag stays smooth).
        match self.output {
            Output::Still => {
                self.scene.finish_ao(device)?;
            }
            Output::Surface(_) if self.ao_quiet_frames >= AO_SETTLE_FRAMES => {
                self.scene
                    .step_ao(device, AO_FRAME_SAMPLES, Some(view.eye))?;
            }
            Output::Surface(_) => {}
        }
        if std::mem::take(&mut self.lut_dirty) {
            device.write_texture(self.lut, 0, 0, &bake_lut(&self.look))?;
        }
        let draw_world = !self.world_drawn;
        self.world_drawn = true;
        let first_shadow_layers = !self.shadow_layers_ready;
        self.shadow_layers_ready = true;
        // Every layer of the light shadow map holds depth before any is read.
        let first_light_shadows = !self.light_shadows_ready;
        self.light_shadows_ready = true;
        // Props first: their bounds widen the shadow cascades' depth range.
        self.prepare_props(device, slot_index, view.eye)?;
        let has_prop_shadows = self.has_prop_shadows();
        // A game's lights after the lamps in the light buffer, written when they changed.
        if self.game_lights.dirty {
            let data: Vec<f32> = self
                .game_lights
                .records()
                .iter()
                .flatten()
                .copied()
                .collect();
            if !data.is_empty() {
                let at = (self.scene.lights.len() * 64) as u64;
                device.write_buffer(self.scene.light_buffer, at, &crate::gpu::f32_bytes(&data))?;
            }
            self.game_lights.dirty = false;
        }
        let size = (self.targets.width, self.targets.height);
        let eye = self.eye(view, jitter, size);
        let main_frustum = Frustum::from_view_proj(&eye.view_proj);
        let cas = self.cascades(view, env);
        let ground = self.ground_view(view);
        let lamps_on = env.lamp_level > 0.001;
        let water_visible = self.quality.reflection_scale > 0.0
            && self.look.water_reflection > 0.001
            && (self.has_water_props()
                || self
                    .scene
                    .draws
                    .iter()
                    .any(|d| d.water && main_frustum.intersects_aabb(d.bounds.0, d.bounds.1)));
        // Mirror camera for the water reflection (skipped when no water enters the view frustum).
        let refl_eye = self.scene.water_level.filter(|_| water_visible).map(|h| {
            let r = reflect_y(h);
            let base = self.eye(view, [0.0, 0.0], self.targets.refl);
            let view_proj = mul(&base.view_proj, &r);
            Eye {
                view_proj,
                inv: invert(&view_proj).unwrap_or(math::IDENTITY),
                pos: [view.eye[0], 2.0 * h - view.eye[1], view.eye[2]],
                near: base.near,
            }
        });
        let has_reflection = refl_eye.is_some();
        {
            let tiles_dim = self.targets.tiles;
            let refl_size = self.targets.refl;
            let slot = &self.slots[slot_index];
            let (tiles, index) = self.tile_lights(&eye, size, tiles_dim[0], lamps_on);
            device.write_buffer(slot.tiles[0], 0, &crate::gpu::u32_bytes(&tiles))?;
            device.write_buffer(slot.index[0], 0, &crate::gpu::u32_bytes(&index))?;
            let block = self.frame_block(
                &eye,
                env,
                &cas,
                &ground,
                size,
                tiles_dim[0],
                false,
                has_prop_shadows,
                has_reflection,
                seed,
            );
            device.write_buffer(slot.frame[0], 0, &block)?;
            if let Some(re) = &refl_eye {
                let (tiles, index) = self.tile_lights(re, refl_size, tiles_dim[1], lamps_on);
                device.write_buffer(slot.tiles[1], 0, &crate::gpu::u32_bytes(&tiles))?;
                device.write_buffer(slot.index[1], 0, &crate::gpu::u32_bytes(&index))?;
                let block = self.frame_block(
                    re,
                    env,
                    &cas,
                    &ground,
                    refl_size,
                    tiles_dim[1],
                    true,
                    has_prop_shadows,
                    has_reflection,
                    seed,
                );
                device.write_buffer(slot.frame[1], 0, &block)?;
            }
        }
        let mut stats = FrameStats::default();

        // Shadow cascades: layers 0..2 cache static world depth; layers 2..4 hold dynamic props.
        if env.key_intensity > 0.0 {
            if self.static_shadows_vp == Some(cas.mats) {
                stats.shadow_static_cached = true;
            } else {
                for (layer, m) in cas.mats.iter().enumerate() {
                    let shadow_frustum = Frustum::from_view_proj(m);
                    enc.begin_pass(PassDesc {
                        colors: vec![],
                        depth: Some(Attachment {
                            texture: self.shadow_map,
                            mip: 0,
                            layer: layer as u32,
                            load: Load::Clear([1.0; 4]),
                            store: true,
                        }),
                    });
                    enc.set_pipeline(self.pipelines.shadow);
                    enc.set_vertex_buffer(self.scene.vertices, 0);
                    enc.set_index_buffer(self.scene.indices, 0);
                    enc.push_constants(&f32_bytes(m));
                    let masked = |d: &&DrawItem| self.shadow_materials[d.material].is_some();
                    for d in self.scene.draws.iter().filter(|d| {
                        !d.water
                            && !masked(d)
                            && shadow_frustum.intersects_aabb_clamped(d.bounds.0, d.bounds.1)
                    }) {
                        stats.shadow_world_draws += 1;
                        enc.draw_indexed(d.count, 1, d.first_index, 0, 0);
                    }
                    let mut bound = false;
                    for d in self.scene.draws.iter().filter(|d| {
                        !d.water
                            && masked(d)
                            && shadow_frustum.intersects_aabb_clamped(d.bounds.0, d.bounds.1)
                    }) {
                        if !bound {
                            enc.set_pipeline(self.pipelines.shadow_masked);
                            bound = true;
                        }
                        let mat = &self.scene.materials[d.material];
                        if let Some(g) = self.shadow_materials[d.material] {
                            enc.set_bind_group(1, g);
                        }
                        let mut push = [0.0f32; 20];
                        push[..16].copy_from_slice(m);
                        push[16..].copy_from_slice(&[
                            mat.params[0],
                            mat.base_color[3],
                            mat.flags[0],
                            0.0,
                        ]);
                        enc.push_constants(&f32_bytes(&push));
                        stats.shadow_world_draws += 1;
                        enc.draw_indexed(d.count, 1, d.first_index, 0, 0);
                    }
                    enc.end_pass();
                }
                self.static_shadows_vp = Some(cas.mats);
            }
            if has_prop_shadows || first_shadow_layers {
                for (layer, m) in cas.mats.iter().enumerate() {
                    enc.begin_pass(PassDesc {
                        colors: vec![],
                        depth: Some(Attachment {
                            texture: self.shadow_map,
                            mip: 0,
                            layer: (layer + 2) as u32,
                            load: Load::Clear([1.0; 4]),
                            store: true,
                        }),
                    });
                    if has_prop_shadows {
                        self.draw_prop_shadows(enc, slot_index, m);
                    }
                    enc.end_pass();
                }
            }
        } else {
            self.static_shadows_vp = None;
            let clear_layers = if first_shadow_layers { 4 } else { 2 };
            for layer in 0..clear_layers {
                enc.begin_pass(PassDesc {
                    colors: vec![],
                    depth: Some(Attachment {
                        texture: self.shadow_map,
                        mip: 0,
                        layer,
                        load: Load::Clear([1.0; 4]),
                        store: true,
                    }),
                });
                enc.end_pass();
            }
        }

        // A game's shadowed lights: the props seen from each, on a layer apiece.
        let views = self.game_lights.shadow_views();
        for layer in 0..super::lights::MAX_SHADOWED_LIGHTS {
            let light_vp = views.get(layer);
            if light_vp.is_none() && !first_light_shadows {
                continue;
            }
            enc.begin_pass(PassDesc {
                colors: vec![],
                depth: Some(Attachment {
                    texture: self.light_shadow_map,
                    mip: 0,
                    layer: layer as u32,
                    load: Load::Clear([1.0; 4]),
                    store: true,
                }),
            });
            if let Some(m) = light_vp {
                self.draw_prop_shadows(enc, slot_index, m);
            }
            enc.end_pass();
        }

        // Rain and petals wrap around a box just ahead of the camera.
        let fwd = normalize(sub(view.target, view.eye));
        let boxes = [RAIN_BOX, PETAL_BOX].map(|b| add(view.eye, scale(fwd, b * 0.5 - 2.0)));
        // The ground map around the view (cached while snapped texel grid stays fixed), and the world map.
        if self.ground_cached_vp == Some(ground.vp) {
            stats.ground_cached = true;
        } else {
            self.draw_ground(enc, self.ground_depth, self.ground_mask, &ground.vp);
            self.ground_cached_vp = Some(ground.vp);
        }
        if draw_world {
            self.draw_ground(enc, self.world_depth, self.world_mask, &self.world.vp);
        }

        let mut push = [0.0f32; 32];
        let clear = [0.0, 0.0, 0.0, 0.0];
        // Water reflection.
        if let Some(re) = &refl_eye {
            stats.reflection_drawn = true;
            self.targets.refl_cleared = true;
            let (refl_hdr, refl_aux, refl_depth) = (
                self.targets.refl_hdr,
                self.targets.refl_aux,
                self.targets.refl_depth,
            );
            enc.begin_pass(PassDesc {
                colors: vec![
                    Attachment::clear(refl_hdr, clear),
                    Attachment::clear(refl_aux, clear),
                ],
                depth: Some(Attachment::clear(refl_depth, clear)),
            });
            enc.set_pipeline(self.pipelines.scene);
            enc.set_bind_group(0, self.slots[slot_index].scene[1]);
            self.draw_world(enc, &mut push, re, false, WorldPart::All);
            self.draw_props(enc, slot_index, 1);
            enc.set_pipeline(self.pipelines.sky);
            enc.set_bind_group(0, self.slots[slot_index].sky[1]);
            enc.draw(3, 1, 0, 0);
            enc.end_pass();
        } else if !self.targets.refl_cleared {
            self.targets.refl_cleared = true;
            enc.begin_pass(PassDesc {
                colors: vec![Attachment::clear(self.targets.refl_hdr, clear)],
                depth: None,
            });
            enc.end_pass();
        }

        let need_ao_clear = self.quality.ao_samples == 0 && !self.targets.ao_cleared;
        if need_ao_clear {
            self.targets.ao_cleared = true;
        }

        let slot = &self.slots[slot_index];
        let t = &self.targets;
        let p = &self.pipelines;

        // Main pass: world, sky, weather.
        enc.begin_pass(PassDesc {
            colors: vec![
                Attachment::clear(t.hdr, clear),
                Attachment::clear(t.aux, clear),
            ],
            depth: Some(Attachment::clear(t.depth, clear)),
        });
        // The game's props first: the table and its players stand nearest and cover most of
        // the screen, so the world behind them fails the depth test before it is shaded.
        self.draw_props(enc, slot_index, 0);
        enc.set_pipeline(p.scene_opaque);
        enc.set_bind_group(0, slot.scene[0]);
        let (opaque_total, opaque_submitted) =
            self.draw_world(enc, &mut push, &eye, true, WorldPart::Opaque);
        enc.set_pipeline(p.scene);
        enc.set_bind_group(0, slot.scene[0]);
        let (rest_total, rest_submitted) =
            self.draw_world(enc, &mut push, &eye, true, WorldPart::Rest);
        let (total_draws, submitted_draws) =
            (opaque_total + rest_total, opaque_submitted + rest_submitted);
        stats.world_draws_total = total_draws;
        stats.world_draws_submitted = submitted_draws;
        stats.world_draws_culled = total_draws.saturating_sub(submitted_draws);
        let blades = self.grass_instances();
        if blades > 0 {
            enc.set_pipeline(p.grass);
            enc.set_bind_group(0, slot.grass);
            enc.draw(15, blades, 0, 0);
            if self.look.grass.flowers > 0.0 {
                enc.set_pipeline(p.flower);
                enc.set_bind_group(0, slot.flower);
                enc.draw(6, blades, 0, 0);
            }
        }
        enc.set_pipeline(p.sky);
        enc.set_bind_group(0, slot.sky[0]);
        enc.draw(3, 1, 0, 0);
        if self.quality.particles {
            enc.set_pipeline(p.particle);
            enc.set_bind_group(0, slot.particle);
            let ground = self.scene.water_level.unwrap_or(self.scene.min[1]);
            let particle_push = |params: [f32; 4], tint: [f32; 4], centre: V3| {
                let mut pp = [0.0f32; 12];
                pp[..4].copy_from_slice(&params);
                pp[4..8].copy_from_slice(&tint);
                pp[8..11].copy_from_slice(&centre);
                pp
            };
            let rain = (env.rain * 26000.0) as u32;
            if rain > 0 {
                let pp = particle_push(
                    [0.0, ground, RAIN_BOX, 1.0],
                    [0.7, 0.75, 0.85, 0.3],
                    boxes[0],
                );
                enc.push_constants(&f32_bytes(&pp));
                enc.draw(6, rain, 0, 0);
            }
            let petals = (env.petals * 3500.0) as u32;
            if petals > 0 {
                let pp = particle_push(
                    [1.0, ground, PETAL_BOX, 1.0],
                    [0.95, 0.5, 0.62, 0.9],
                    boxes[1],
                );
                enc.push_constants(&f32_bytes(&pp));
                enc.draw(6, petals, 0, 0);
            }
        }
        self.draw_blended_props(enc, slot_index);
        enc.end_pass();

        let l = &self.look;
        let px_scale = t.height as f32 / 1080.0;
        // Ambient occlusion (skipped when `ao_samples == 0`, relying on baked vertex AO).
        if self.quality.ao_samples > 0 {
            stats.ssao_drawn = true;
            let mut pc = [0.0f32; 16];
            pc[..4].copy_from_slice(&[
                l.ao_radius,
                self.quality.ao_samples as f32,
                t.ao.0 as f32,
                t.ao.1 as f32,
            ]);
            Self::post_pass(enc, t.ao_tex, 0, Load::DontCare, p.ssao, slot.ssao, pc);
        } else if need_ao_clear {
            enc.begin_pass(PassDesc {
                colors: vec![Attachment::clear(t.ao_tex, [1.0; 4])],
                depth: None,
            });
            enc.end_pass();
        }
        // Air.
        let mut pc = [0.0f32; 16];
        pc[..4].copy_from_slice(&[
            l.ambient_occlusion,
            l.outline,
            (1.4 * px_scale).max(1.0),
            0.0,
        ]);
        pc[4..7].copy_from_slice(&l.ao_tint);
        pc[8..11].copy_from_slice(&l.outline_color);
        Self::post_pass(enc, t.lit, 0, Load::DontCare, p.air, slot.air, pc);
        // Bloom down and up.
        for level in 0..t.bloom_levels {
            let (sw, sh) = if level == 0 {
                (t.width, t.height)
            } else {
                (
                    (t.bloom_size.0 >> (level - 1)).max(1),
                    (t.bloom_size.1 >> (level - 1)).max(1),
                )
            };
            let mut pc = [0.0f32; 16];
            pc[..4].copy_from_slice(&[
                1.0 / sw as f32,
                1.0 / sh as f32,
                l.bloom_threshold,
                if level == 0 { 1.0 } else { 0.0 },
            ]);
            Self::post_pass(
                enc,
                t.bloom,
                level,
                Load::DontCare,
                p.bloom_down,
                slot.down[level as usize],
                pc,
            );
        }
        for level in (0..t.bloom_levels.saturating_sub(1)).rev() {
            let (sw, sh) = (
                (t.bloom_size.0 >> (level + 1)).max(1),
                (t.bloom_size.1 >> (level + 1)).max(1),
            );
            let mut pc = [0.0f32; 16];
            pc[..4].copy_from_slice(&[1.0 / sw as f32, 1.0 / sh as f32, 1.0, 0.0]);
            Self::post_pass(
                enc,
                t.bloom,
                level,
                Load::Load,
                p.bloom_up,
                slot.up[level as usize],
                pc,
            );
        }
        let (dest, encode_srgb) = match self.output {
            Output::Still => (t.still.ok_or("still target missing")?, false),
            Output::Surface(f) => (target.ok_or("no surface texture")?, !f.is_srgb()),
        };
        let exposure = l.effective_exposure() * env.exposure;
        if !self.quality.painterly {
            stats.fused_tonemap = true;
            let mut pc = [0.0f32; 16];
            pc[..4].copy_from_slice(&[exposure, l.bloom, l.vignette, l.sharpen.max(0.0)]);
            pc[4] = if encode_srgb { 1.0 } else { 0.0 };
            pc[5] = if self.quality.fxaa { 1.0 } else { 0.0 };
            Self::post_pass(
                enc,
                dest,
                0,
                Load::DontCare,
                p.tonemap_final,
                slot.tonemap_final,
                pc,
            );
        } else {
            // Tone mapping and grading.
            let mut pc = [0.0f32; 16];
            pc[..4].copy_from_slice(&[exposure, l.bloom, l.saturation, l.contrast]);
            pc[4..7].copy_from_slice(&l.lift);
            pc[7] = l.vignette;
            pc[8..11].copy_from_slice(&l.gain);
            pc[11] = l.tonemap as u32 as f32;
            Self::post_pass(enc, t.ldr, 0, Load::DontCare, p.tonemap, slot.tonemap, pc);
            let mut pc = [0.0f32; 16];
            pc[..4].copy_from_slice(&[
                (l.painterly * px_scale * 0.6).clamp(0.0, 6.0),
                (l.painterly * l.painterly_far.max(0.0) * px_scale).clamp(0.0, 6.0),
                0.85,
                0.0,
            ]);
            // Depth of field, drawn with the painterly filter.
            if let Some(f) = view.focus.filter(Focus::blurs) {
                pc[4..8].copy_from_slice(&[
                    f.distance.max(0.0),
                    f.depth,
                    (f.blur * t.height as f32).min(MAX_FOCUS_BLUR_PX),
                    1.0,
                ]);
            }
            Self::post_pass(enc, t.paint, 0, Load::DontCare, p.paint, slot.paint, pc);
            let mut pc = [0.0f32; 16];
            pc[0] = if self.quality.fxaa { 1.0 } else { 0.0 };
            pc[1] = if encode_srgb { 1.0 } else { 0.0 };
            pc[2] = l.sharpen.max(0.0);
            Self::post_pass(enc, dest, 0, Load::DontCare, p.output, slot.output, pc);
        }
        if self.has_overlays() {
            enc.begin_pass(PassDesc {
                colors: vec![Attachment {
                    texture: dest,
                    mip: 0,
                    layer: 0,
                    load: Load::Load,
                    store: true,
                }],
                depth: Some(Attachment {
                    texture: t.depth,
                    mip: 0,
                    layer: 0,
                    load: Load::Load,
                    store: false,
                }),
            });
            self.draw_overlays(enc, slot_index, encode_srgb);
            enc.end_pass();
        }
        if let (Output::Still, Some(still), Some(rb)) = (self.output, t.still, t.readback) {
            enc.copy_texture_to_buffer(still, 0, 0, rb, 0);
        }
        self.stats = stats;
        Ok(())
    }

    /// Render a still: `samples` jittered frames averaged in linear display space. Returns
    /// RGBA8 sRGB pixels, top row first.
    pub fn render_still(
        &mut self,
        device: &mut Device<'_>,
        view: &View,
        env: &Environment,
        samples: u32,
    ) -> Result<Vec<u8>> {
        if self.output != Output::Still {
            return Err("render_still needs a renderer created with Output::Still".into());
        }
        let (w, h) = self.size();
        let n = (w * h * 4) as usize;
        let mut acc = vec![0.0f32; n];
        let samples = samples.max(1);
        for s in 0..samples {
            let jitter = if samples == 1 {
                [0.0, 0.0]
            } else {
                [halton(s + 1, 2) - 0.5, halton(s + 1, 3) - 0.5]
            };
            let mut enc = device.encoder();
            self.encode(device, &mut enc, view, env, jitter, s, None)?;
            device.submit(enc)?;
            let bytes = device.read_buffer(self.targets.readback.ok_or("no readback")?)?;
            for (i, a) in acc.iter_mut().enumerate() {
                let h = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                *a += f16_to_f32(h);
            }
        }
        let inv = 1.0 / samples as f32;
        let mut out = vec![0u8; n];
        for (i, (o, a)) in out.iter_mut().zip(&acc).enumerate() {
            let v = a * inv;
            if i % 4 == 3 {
                *o = 255;
            } else {
                // Ordered dither before quantizing to avoid banding in skies.
                let px = (i / 4) as u32;
                let d = (halton(px % 4096 + 1, 5) - 0.5) / 255.0;
                *o = ((math::linear_to_srgb(v) + d) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec3;
    use crate::render::look::SurfaceModel;
    use crate::render::props::{MaterialDesc, MeshData};
    use crate::render::sky::{TimeOfDay, Weather};
    use crate::world::{AlphaMode, Instance, Material, Mesh, Shading, Transform, World};
    use ganbate_shader::{
        CompileOptions, ShaderLibrary, SpecValue, compile_glsl_with_options,
        compile_msl_with_options, compile_wgsl_with_options, compile_with_options,
    };

    #[test]
    fn specialize_brdf_permutations_all_backends() {
        let mut lib = ShaderLibrary::new();
        for (name, src) in shader_modules() {
            lib.add(name, src);
        }
        let composed = lib.compose(SCENE).expect("compose scene");

        let full = compile_with_options(
            &composed,
            Stage::Fragment,
            "fs_main",
            &CompileOptions::default(),
        )
        .expect("full uber-shader");

        for fixed in 0..6 {
            let specs = [("FIXED_SURFACE_MODEL", SpecValue::I32(fixed))];
            let opts = CompileOptions {
                imports: &[],
                specialize: &specs,
            };
            let spv = compile_with_options(&composed, Stage::Fragment, "fs_main", &opts)
                .expect("specialized spirv");
            assert!(
                spv.words.len() < full.words.len(),
                "fixed model {fixed} should prune unused BRDF lobes ({} vs {})",
                spv.words.len(),
                full.words.len()
            );
            compile_msl_with_options(&composed, Stage::Fragment, "fs_main", &opts)
                .expect("specialized msl");
            compile_wgsl_with_options(&composed, Stage::Fragment, "fs_main", &opts)
                .expect("specialized wgsl");
            compile_glsl_with_options(&composed, Stage::Fragment, "fs_main", &opts)
                .expect("specialized glsl");
        }
    }

    fn make_quad_mesh(y: f32, half: f32) -> Mesh {
        Mesh {
            positions: vec![
                Vec3(-half, y, -half),
                Vec3(-half, y, half),
                Vec3(half, y, half),
                Vec3(half, y, -half),
            ],
            normals: vec![Vec3(0.0, 1.0, 0.0); 4],
            uvs: vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            color: [1.0, 1.0, 1.0],
            indices: vec![0, 1, 2, 0, 2, 3],
            material: Some(0),
        }
    }

    fn sphere_prop_data() -> (Vec<V3>, Vec<V3>, Vec<[f32; 2]>, Vec<u32>) {
        let (stacks, slices) = (10u32, 16u32);
        let (mut pos, mut nrm, mut uvs, mut idx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..=stacks {
            let v = i as f32 / stacks as f32;
            let phi = v * std::f32::consts::PI;
            for j in 0..=slices {
                let u = j as f32 / slices as f32;
                let theta = u * std::f32::consts::TAU;
                let n = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
                pos.push(math::scale(n, 0.85));
                nrm.push(n);
                uvs.push([u, v]);
            }
        }
        let row = slices + 1;
        for i in 0..stacks {
            for j in 0..slices {
                let a = i * row + j;
                let b = a + row;
                idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        (pos, nrm, uvs, idx)
    }

    #[test]
    fn hybrid_stage_culls_caches_and_renders_unified_surface_models() {
        let Ok(mut device) = Device::headless() else {
            return;
        };
        let mut world = World::default();
        world.materials.push(Material {
            name: "pedestal".into(),
            base_color: [0.55, 0.53, 0.50, 1.0],
            roughness: 0.65,
            metallic: 0.0,
            alpha: AlphaMode::Opaque,
            shading: Shading::Standard,
            ..Material::default()
        });
        world.meshes.push(make_quad_mesh(0.0, 4.0));
        let mut behind = math::IDENTITY;
        behind[12] = 0.0;
        behind[14] = 95.0; // Far behind the eye at z = 4 looking towards origin.
        world.instances.push(Instance {
            mesh: 0,
            transform: Transform(math::IDENTITY),
        });
        world.instances.push(Instance {
            mesh: 0,
            transform: Transform(behind),
        });

        let mut look = Look::default();
        let mut renderer = Renderer::new(
            &mut device,
            &world,
            look.clone(),
            Quality::hybrid_stage(),
            (64, 64),
            Output::Still,
        )
        .expect("create hybrid renderer");

        let (sp, sn, suv, sidx) = sphere_prop_data();
        let sphere_mesh = renderer
            .create_prop_mesh(
                &mut device,
                &MeshData {
                    positions: &sp,
                    normals: &sn,
                    uvs: &suv,
                    indices: &sidx,
                },
            )
            .expect("sphere mesh");
        let sphere_mat = renderer
            .create_prop_material(
                &mut device,
                &MaterialDesc {
                    name: "probe".into(),
                    base_color: [0.75, 0.22, 0.18, 1.0],
                    roughness: 0.25,
                    metallic: 0.1,
                    ..MaterialDesc::default()
                },
            )
            .expect("sphere mat");
        let mut sphere_xf = math::IDENTITY;
        sphere_xf[13] = 0.9;
        let _sphere_inst = renderer
            .create_prop_instance(sphere_mesh, sphere_mat, &sphere_xf)
            .expect("sphere instance");

        let view = View {
            name: "probe".into(),
            eye: [0.0, 1.8, 3.6],
            target: [0.0, 0.85, 0.0],
            fov: 45.0,
            ..View::default()
        };
        let env = Environment::resolve(
            &TimeOfDay {
                hours: 14.5,
                ..TimeOfDay::default()
            },
            &Weather::default(),
            &look.palette,
            1.0,
        );

        let img1 = renderer
            .render_still(&mut device, &view, &env, 1)
            .expect("frame 1");
        let stats1 = renderer.last_frame_stats();
        assert!(stats1.world_draws_culled >= 1);
        assert!(!stats1.reflection_drawn);
        assert!(!stats1.ssao_drawn);
        assert!(stats1.fused_tonemap);
        assert!(!stats1.shadow_static_cached);

        let _img2 = renderer
            .render_still(&mut device, &view, &env, 1)
            .expect("frame 2");
        let stats2 = renderer.last_frame_stats();
        assert!(stats2.shadow_static_cached);
        assert!(stats2.ground_cached);

        // Verify unified surface models + tonemap curves + EV100 exposure produce distinct valid frames.
        let avg_luma = |px: &[u8]| -> f64 {
            px.chunks_exact(4)
                .map(|c| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64)
                .sum::<f64>()
                / (px.len() / 4) as f64
        };
        assert!(avg_luma(&img1) > 5.0);

        let mut model_renders = Vec::new();
        for model in SurfaceModel::ALL {
            renderer
                .update_prop_material(
                    &mut device,
                    sphere_mat,
                    &MaterialDesc {
                        name: "probe".into(),
                        base_color: [0.78, 0.24, 0.20, 1.0],
                        roughness: 0.22,
                        metallic: if model == SurfaceModel::StandardGGX {
                            0.85
                        } else {
                            0.0
                        },
                        surface: Some(model),
                        clearcoat: Some(0.95),
                        clearcoat_roughness: Some(0.06),
                        sheen: Some(0.9),
                        subsurface: Some(0.8),
                        ..MaterialDesc::default()
                    },
                )
                .expect("update prop surface");
            let px = renderer
                .render_still(&mut device, &view, &env, 1)
                .expect("render surface model");
            assert!(avg_luma(&px) > 5.0);
            model_renders.push(px);
        }
        assert_ne!(
            model_renders[SurfaceModel::StylizedRamp as usize],
            model_renders[SurfaceModel::StandardGGX as usize]
        );
        assert_ne!(
            model_renders[SurfaceModel::StandardGGX as usize],
            model_renders[SurfaceModel::CharlieCloth as usize]
        );

        look.tonemap = ToneMap::AgX;
        look.ev100 = Some(13.0);
        renderer.set_look(look.clone());
        let agx_ev13 = renderer
            .render_still(&mut device, &view, &env, 1)
            .expect("agx ev13");

        look.tonemap = ToneMap::ACESFitted;
        look.ev100 = Some(11.5); // Brighter camera exposure (+1.5 EV stops).
        renderer.set_look(look);
        let aces_ev11 = renderer
            .render_still(&mut device, &view, &env, 1)
            .expect("aces ev11.5");
        assert!(avg_luma(&aces_ev11) > avg_luma(&agx_ev13));

        let errs = device.validation_errors();
        assert!(errs.is_empty(), "validation errors: {errs:?}");
        renderer.destroy_props(&mut device);
    }

    #[test]
    fn low_cost_quality_invariants_and_budget_ordering() {
        let low = Quality::low_cost();
        let hybrid = Quality::hybrid_stage();
        let realtime = Quality::realtime();
        let still = Quality::still();

        assert_eq!(low.shadow_size, 512);
        assert_eq!(low.reflection_scale, 0.0);
        assert_eq!(low.ao_scale, 0.25);
        assert_eq!(low.ao_samples, 0);
        assert_eq!(low.tile_size, 32);
        assert_eq!(low.bloom_levels, 2);
        assert!(!low.painterly);
        assert!(!low.fxaa);
        assert!(!low.particles);
        assert_eq!(low.ground_size, 256);
        assert_eq!(low.world_size, 512);
        assert_eq!(low.grass, 0.0);
        assert_eq!(low.light_shadow_size, 256);

        assert!(low.shadow_size < hybrid.shadow_size && hybrid.shadow_size < realtime.shadow_size);
        assert!(low.reflection_scale < hybrid.reflection_scale);
        assert!(hybrid.reflection_scale < realtime.reflection_scale);
        assert!(
            low.bloom_levels < hybrid.bloom_levels && hybrid.bloom_levels < realtime.bloom_levels
        );
        assert!(low.ground_size < hybrid.ground_size && hybrid.ground_size <= realtime.ground_size);
        assert!(low.world_size < hybrid.world_size && hybrid.world_size <= realtime.world_size);
        assert!(low.grass < hybrid.grass && hybrid.grass < realtime.grass);
        assert!(
            low.light_shadow_size < hybrid.light_shadow_size
                && hybrid.light_shadow_size < realtime.light_shadow_size
                && realtime.light_shadow_size < still.light_shadow_size
        );
    }

    #[test]
    fn set_quality_switches_dynamically_skips_water_reflection_and_preserves_props() {
        let Ok(mut device) = Device::headless() else {
            return;
        };
        let mut world = World::default();
        world.materials.push(Material {
            name: "stone".into(),
            base_color: [0.5, 0.5, 0.52, 1.0],
            roughness: 0.7,
            metallic: 0.0,
            alpha: AlphaMode::Opaque,
            shading: Shading::Standard,
            ..Material::default()
        });
        world.materials.push(Material {
            name: "pond".into(),
            base_color: [0.12, 0.34, 0.42, 0.85],
            roughness: 0.08,
            metallic: 0.0,
            alpha: AlphaMode::Opaque,
            shading: Shading::Water,
            ..Material::default()
        });
        world.meshes.push(make_quad_mesh(-0.4, 5.0));
        let mut water_mesh = make_quad_mesh(0.0, 4.0);
        water_mesh.material = Some(1);
        world.meshes.push(water_mesh);
        world.instances.push(Instance {
            mesh: 0,
            transform: Transform(math::IDENTITY),
        });
        world.instances.push(Instance {
            mesh: 1,
            transform: Transform(math::IDENTITY),
        });

        let look = Look::default();
        let mut renderer = Renderer::new(
            &mut device,
            &world,
            look.clone(),
            Quality::realtime(),
            (64, 64),
            Output::Still,
        )
        .expect("create realtime renderer");
        assert_eq!(renderer.quality(), Quality::realtime());

        let (sp, sn, suv, sidx) = sphere_prop_data();
        let prop_mesh = renderer
            .create_prop_mesh(
                &mut device,
                &MeshData {
                    positions: &sp,
                    normals: &sn,
                    uvs: &suv,
                    indices: &sidx,
                },
            )
            .expect("prop mesh");
        let prop_mat = renderer
            .create_prop_material(
                &mut device,
                &MaterialDesc {
                    name: "buoy".into(),
                    base_color: [0.85, 0.32, 0.16, 1.0],
                    roughness: 0.3,
                    ..MaterialDesc::default()
                },
            )
            .expect("prop material");
        let mut prop_xf = math::IDENTITY;
        prop_xf[13] = 0.8;
        let _prop_inst = renderer
            .create_prop_instance(prop_mesh, prop_mat, &prop_xf)
            .expect("prop instance");

        let view = View {
            name: "pond".into(),
            eye: [0.0, 2.0, 4.2],
            target: [0.0, 0.2, 0.0],
            fov: 48.0,
            ..View::default()
        };
        let env = Environment::resolve(
            &TimeOfDay {
                hours: 15.0,
                ..TimeOfDay::default()
            },
            &Weather::default(),
            &look.palette,
            1.0,
        );

        let rt_px = renderer
            .render_still(&mut device, &view, &env, 1)
            .expect("render realtime");
        let rt_stats = renderer.last_frame_stats();
        assert!(rt_stats.reflection_drawn, "realtime draws water reflection");
        assert!(rt_stats.ssao_drawn, "realtime draws SSAO");
        assert!(
            !rt_stats.fused_tonemap,
            "realtime uses painterly post chain"
        );

        renderer
            .set_quality(&mut device, Quality::low_cost())
            .expect("switch to low_cost");
        assert_eq!(renderer.quality(), Quality::low_cost());
        // No-op re-apply of identical quality must succeed immediately.
        renderer
            .set_quality(&mut device, Quality::low_cost())
            .expect("idempotent low_cost");

        let lc_px = renderer
            .render_still(&mut device, &view, &env, 1)
            .expect("render low_cost");
        let lc_stats = renderer.last_frame_stats();
        assert!(
            !lc_stats.reflection_drawn,
            "low_cost skips water reflection when reflection_scale == 0.0"
        );
        assert!(!lc_stats.ssao_drawn, "low_cost skips SSAO");
        assert!(lc_stats.fused_tonemap, "low_cost uses fused LUT tonemap");

        renderer
            .set_quality(&mut device, Quality::hybrid_stage())
            .expect("switch to hybrid_stage");
        assert_eq!(renderer.quality(), Quality::hybrid_stage());
        let hy_px = renderer
            .render_still(&mut device, &view, &env, 1)
            .expect("render hybrid_stage");
        let hy_stats = renderer.last_frame_stats();
        assert!(
            hy_stats.reflection_drawn,
            "hybrid_stage restores water reflection when water is visible"
        );
        assert!(!hy_stats.ssao_drawn);
        assert!(hy_stats.fused_tonemap);

        for px in [&rt_px, &lc_px, &hy_px] {
            let luma = px
                .chunks_exact(4)
                .map(|c| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64)
                .sum::<f64>()
                / (px.len() / 4) as f64;
            assert!(luma > 5.0);
        }

        let errs = device.validation_errors();
        assert!(errs.is_empty(), "validation errors: {errs:?}");
        renderer.destroy_props(&mut device);
    }
}
