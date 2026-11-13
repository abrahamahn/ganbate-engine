use super::*;

pub(super) fn shader_modules() -> [(&'static str, &'static str); 4] {
    [
        ("common", COMMON),
        ("brdf", BRDF),
        ("scene_detail", SCENE_DETAIL),
        ("scene_grass", SCENE_GRASS),
    ]
}

/// Expands the stylized renderer's `#import` library (`common`, `brdf`, `scene_detail`,
/// `scene_grass`) into `body`.
pub fn compose_shader(body: &str) -> Result<String> {
    ganbate_shader::compose(body, &shader_modules()).map_err(|e| e.to_string())
}

pub(super) fn compile(body: &str, stage: Stage, entry: &str) -> Result<Compiled> {
    compile_specialized(body, stage, entry, &[])
}

/// [`compile`] with the shader's module `const`s set.
pub(super) fn compile_specialized(
    body: &str,
    stage: Stage,
    entry: &str,
    specialize: &[(&str, SpecValue)],
) -> Result<Compiled> {
    let source = compose_shader(body)?;
    crate::gpu::compile_shader_specialized(&source, stage, entry, specialize)
}

/// `fs_water` as the water's pipeline compiles it: its lamp light and shadow read from
/// `cache_water` where no prop moved since they were kept, computed elsewhere.
pub(super) const WATER_STATICS: [(&str, SpecValue); 1] = [("WATER_STATICS", SpecValue::Bool(true))];

/// `cache_water`'s format: 32-bit floats hand the water the very values computing them gives, so
/// its frames match those that compute them byte for byte (16-bit floats moved some pixels a few
/// steps). Devices that cannot read them (WebGPU without `float32-filterable`) compute them.
pub(super) const WATER_STATICS_FORMAT: Format = Format::Rgba32Float;

/// `fs_grade`'s target: 32-bit floats hold each pixel's grade as computing it gives, so the output
/// read from them is the fused pass's byte for byte. WebGL 2 renders them with
/// `EXT_color_buffer_float`, as it does the 16-bit float targets the renderer always has.
pub(super) const GRADED_FORMAT: Format = Format::Rgba32Float;

/// `fs_tonemap_output` reading the grades `fs_grade` wrote.
pub(super) const GRADED: [(&str, SpecValue); 1] = [("GRADED", SpecValue::Bool(true))];

/// The scene shader with a game's prop shader (`shade`) and the overlay entry point that calls it.
pub(in crate::render) fn prop_shader_source(user: &str) -> Result<String> {
    let raw = format!(
        "{SCENE}\n{PROP_SHADER_INPUT}\n// --- prop shader\n{user}\n// --- prop shader \
         entry\n{PROP_SHADER}"
    );
    ganbate_shader::compose(&raw, &shader_modules()).map_err(|e| e.to_string())
}

/// A game's prop shader's fragment stage compiled, checked to read only what its draw binds: an
/// overlay's frame and a scene prop's material (its three textures). Its vertex stage is every
/// prop's (`Pipelines::prop_vs`).
pub(in crate::render) fn prop_shader_stage(pipelines: &Pipelines, user: &str) -> Result<Compiled> {
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
pub(super) const MATERIAL_GROUP: u32 = 1;

/// Bindings by group and binding, for comparing stages.
pub(super) fn sorted_bindings(
    bindings: &[ganbate_shader::Binding],
) -> Vec<ganbate_shader::Binding> {
    let mut out = bindings.to_vec();
    out.sort_by_key(|b| (b.group, b.binding));
    out
}

/// One overlay pipeline of a game's prop shader: `blend` 0 opaque, 1 alpha, 2 additive; `depth` 0
/// drawn on top, 1 testing the scene's depth, 2 testing and writing it (opaque).
pub(in crate::render) fn prop_shader_pipeline(
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

pub(super) fn color(format: Format, blend: Blend) -> ColorTarget {
    ColorTarget {
        format,
        blend,
        write_mask: 0xF,
    }
}

pub(super) fn fullscreen(
    device: &mut Device<'_>,
    vs: &Compiled,
    entry: &str,
    format: Format,
    blend: Blend,
) -> Result<PipelineId> {
    fullscreen_specialized(device, vs, entry, &[], format, blend)
}

/// [`fullscreen`] with the post shader's module `const`s set.
pub(super) fn fullscreen_specialized(
    device: &mut Device<'_>,
    vs: &Compiled,
    entry: &str,
    specialize: &[(&str, SpecValue)],
    format: Format,
    blend: Blend,
) -> Result<PipelineId> {
    let fs = compile_specialized(POST, Stage::Fragment, entry, specialize)?;
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
    pub(super) fn new(device: &mut Device<'_>, output: Format) -> Result<Self> {
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
        let world_layout_cache = world_layout.clone();
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
        let water = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&compile_specialized(
                SCENE,
                Stage::Fragment,
                "fs_water",
                &WATER_STATICS,
            )?),
            vertex_layout: Some(world_layout_cache.clone()),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(reverse_depth(Compare::Greater, true)),
            colors: mrt(),
        })?;
        let water_opaque_fs = compile(SCENE, Stage::Fragment, "fs_water_opaque")?;
        let water_masked_fs = compile(SCENE, Stage::Fragment, "fs_water_masked")?;
        let water_opaque = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&water_opaque_fs),
            vertex_layout: Some(world_layout_cache.clone()),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(reverse_depth(Compare::Greater, true)),
            colors: mrt(),
        })?;
        let water_masked = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&water_masked_fs),
            vertex_layout: Some(world_layout_cache.clone()),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(reverse_depth(Compare::Greater, true)),
            colors: mrt(),
        })?;
        // Tested against the cached picture's depth (a copy) and writing its own, so where water
        // lies over water the nearer keeps its light, as in the main pass.
        let water_static = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&compile(SCENE, Stage::Fragment, "fs_water_static")?),
            vertex_layout: Some(world_layout_cache.clone()),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(reverse_depth(Compare::Greater, true)),
            colors: vec![color(WATER_STATICS_FORMAT, Blend::Replace)],
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
        let water_prop = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&water_masked_fs),
            vertex_layout: Some(layout.clone()),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(reverse_depth(Compare::Greater, true)),
            colors: mrt(),
        })?;
        let water_prop_opaque = device.create_render_pipeline(&RenderPipelineDesc {
            vertex: &vs,
            fragment: Some(&water_opaque_fs),
            vertex_layout: Some(layout.clone()),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(reverse_depth(Compare::Greater, true)),
            colors: mrt(),
        })?;
        // The cached picture's pipelines: the same shading with the shadows moved props can
        // change written to a third target.
        let mrt3 = || {
            vec![
                color(HDR, Blend::Replace),
                color(HDR, Blend::Replace),
                color(HDR, Blend::Replace),
            ]
        };
        let cache_fs = [
            compile(SCENE, Stage::Fragment, "fs_main_cache")?,
            compile(SCENE, Stage::Fragment, "fs_main_opaque_cache")?,
        ];
        let world_vs = compile(SCENE, Stage::Vertex, "vs_main")?;
        let mut cache_scene = [scene; 2];
        let mut cache_prop = [scene_prop; 2];
        for (i, fs) in cache_fs.iter().enumerate() {
            for (out, vs, vertex_layout) in [
                (&mut cache_scene[i], &world_vs, world_layout_cache.clone()),
                (&mut cache_prop[i], &vs, layout.clone()),
            ] {
                *out = device.create_render_pipeline(&RenderPipelineDesc {
                    vertex: vs,
                    fragment: Some(fs),
                    vertex_layout: Some(vertex_layout),
                    topology: Topology::TriangleList,
                    cull: Cull::None,
                    front_ccw: true,
                    depth: Some(reverse_depth(Compare::Greater, true)),
                    colors: mrt3(),
                })?;
            }
        }
        // Blended props leave the aux target (normals for outlines and occlusion) alone.
        let blend_fs = compile(SCENE, Stage::Fragment, "fs_main_blend")?;
        let mut scene_prop_blend = [scene_prop; 2];
        let mut water_prop_blend = [water_prop; 2];
        for (i, blend) in [Blend::Premultiplied, Blend::Additive]
            .into_iter()
            .enumerate()
        {
            for (out, frag) in [
                (&mut scene_prop_blend[i], &blend_fs),
                (&mut water_prop_blend[i], &water_masked_fs),
            ] {
                *out = device.create_render_pipeline(&RenderPipelineDesc {
                    vertex: &vs,
                    fragment: Some(frag),
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
            water,
            water_opaque,
            water_masked,
            water_static,
            scene_prop,
            scene_prop_opaque,
            water_prop,
            water_prop_opaque,
            cache_scene,
            cache_prop,
            scene_prop_blend,
            water_prop_blend,
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
            // Colour and aux only: the depth is copied, so the GPU keeps rejecting what lies
            // behind the picture early (a shader-written depth would leave it none).
            restore: device.create_render_pipeline(&RenderPipelineDesc {
                vertex: &fsvs,
                fragment: Some(&compile(POST, Stage::Fragment, "fs_restore")?),
                vertex_layout: None,
                topology: Topology::TriangleList,
                cull: Cull::None,
                front_ccw: true,
                depth: Some(DepthState {
                    format: Format::Depth32Float,
                    compare: Compare::Always,
                    write: false,
                    bias_constant: 0.0,
                    bias_slope: 0.0,
                    clamp: false,
                }),
                colors: vec![color(HDR, Blend::Replace), color(HDR, Blend::Replace)],
            })?,
            cache_clear: device.create_render_pipeline(&RenderPipelineDesc {
                vertex: &fsvs,
                fragment: Some(&compile(POST, Stage::Fragment, "fs_clear")?),
                vertex_layout: None,
                topology: Topology::TriangleList,
                cull: Cull::None,
                front_ccw: true,
                depth: Some(DepthState {
                    format: Format::Depth32Float,
                    compare: Compare::Always,
                    write: true,
                    bias_constant: 0.0,
                    bias_slope: 0.0,
                    clamp: false,
                }),
                colors: vec![
                    color(HDR, Blend::Replace),
                    color(HDR, Blend::Replace),
                    color(HDR, Blend::Replace),
                ],
            })?,
            reshade_mask: device.create_render_pipeline(&RenderPipelineDesc {
                vertex: &fsvs,
                fragment: Some(&compile(SCENE, Stage::Fragment, "fs_reshade_mask")?),
                vertex_layout: None,
                topology: Topology::TriangleList,
                cull: Cull::None,
                front_ccw: true,
                depth: Some(DepthState {
                    format: Format::Depth32Float,
                    compare: Compare::Always,
                    write: true,
                    bias_constant: 0.0,
                    bias_slope: 0.0,
                    clamp: false,
                }),
                colors: vec![],
            })?,
            ao_grass_half: fullscreen(
                device,
                &fsvs,
                "fs_ao_grass_half",
                Format::R8Unorm,
                Blend::Replace,
            )?,
            ao_grass_tile: fullscreen(
                device,
                &fsvs,
                "fs_ao_grass_tile",
                Format::R8Unorm,
                Blend::Replace,
            )?,
            ao_grass_union: fullscreen(
                device,
                &fsvs,
                "fs_ao_grass_union",
                Format::R8Unorm,
                Blend::Replace,
            )?,
            ao_grass_coarse: fullscreen(
                device,
                &fsvs,
                "fs_ao_grass_coarse",
                Format::R8Unorm,
                Blend::Replace,
            )?,
            ssao: fullscreen(device, &fsvs, "fs_ssao", Format::R16Float, Blend::Replace)?,
            air: fullscreen(device, &fsvs, "fs_air", HDR, Blend::Replace)?,
            bloom_down: fullscreen(device, &fsvs, "fs_bloom_down", HDR, Blend::Replace)?,
            bloom_up: fullscreen(device, &fsvs, "fs_bloom_up", HDR, Blend::Additive)?,
            tonemap: fullscreen(device, &fsvs, "fs_tonemap", HDR, Blend::Replace)?,
            tonemap_final: fullscreen(device, &fsvs, "fs_tonemap_final", output, Blend::Replace)?,
            tonemap_output: [
                fullscreen(device, &fsvs, "fs_tonemap_output", output, Blend::Replace)?,
                fullscreen(
                    device,
                    &fsvs,
                    "fs_tonemap_output_plain",
                    output,
                    Blend::Replace,
                )?,
            ],
            grade_once: None,
            paint: fullscreen(device, &fsvs, "fs_paint", HDR, Blend::Replace)?,
            output: fullscreen(device, &fsvs, "fs_final", output, Blend::Replace)?,
        })
    }

    /// The `grade_once` pipelines, for an `output` target.
    pub(super) fn grade_once(
        device: &mut Device<'_>,
        output: Format,
    ) -> Result<([PipelineId; 2], PipelineId)> {
        let fsvs = compile(POST, Stage::Vertex, "vs_fullscreen")?;
        let mut pass = |entry| fullscreen(device, &fsvs, entry, GRADED_FORMAT, Blend::Replace);
        let grade = [pass("fs_grade")?, pass("fs_grade_plain")?];
        let read = fullscreen_specialized(
            device,
            &fsvs,
            "fs_tonemap_output",
            &GRADED,
            output,
            Blend::Replace,
        )?;
        Ok((grade, read))
    }
}
