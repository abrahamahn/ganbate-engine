//! Prop textures, meshes, materials, custom WGSL prop shaders, instances, and dynamic game lights.

use super::{bytes, floats, matrix, with, words};
use ganbate_engine::render::{
    GameLight, LightDesc, MaterialDesc, MeshData, PROP_SHADER_PARAMS, PropInstance, PropMaterial,
    PropMesh, PropTexture, SpotCone,
};

fn triples(v: &[f32]) -> Vec<[f32; 3]> {
    v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// A texture of `width` x `height` RGBA8 pixels (`len` = width * height * 4 bytes at `ptr`),
/// sRGB colour (`srgb` 1) or linear data such as normal maps (0), with a full mip chain.
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_texture_create(
    width: u32,
    height: u32,
    ptr: *const u8,
    len: usize,
    srgb: u32,
) -> u32 {
    // SAFETY: the caller passes a readable range.
    let data = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        if width == 0
            || height == 0
            || data.len() as u64 != u64::from(width) * u64::from(height) * 4
        {
            return Err(format!(
                "texture: {} bytes for {width}x{height} RGBA8",
                data.len()
            ));
        }
        let t =
            rt.renderer
                .create_prop_texture(&mut rt.device, (width, height), data, srgb != 0)?;
        Ok(t.raw())
    })
}

/// A texture from a PNG file (`len` bytes at `ptr`), decoded by the engine: sRGB colour
/// (`srgb` 1) or linear data (0), with a full mip chain.
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_texture_png(ptr: *const u8, len: usize, srgb: u32) -> u32 {
    // SAFETY: the caller passes a readable range.
    let data = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        let image = ganbate_engine::assets::png::decode(data).map_err(|e| format!("PNG: {e}"))?;
        let t = rt.renderer.create_prop_texture(
            &mut rt.device,
            (image.width, image.height),
            &image.rgba,
            srgb != 0,
        )?;
        Ok(t.raw())
    })
}

/// Replaces a texture's pixels (same size).
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_texture_update(
    texture: u32,
    ptr: *const u8,
    len: usize,
) -> u32 {
    // SAFETY: the caller passes a readable range.
    let data = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        rt.renderer
            .update_prop_texture(&mut rt.device, PropTexture::from_raw(texture), data)?;
        Ok(1)
    })
}

/// As [`ganbate_rt_texture_create`], from the image the host set for this call (a canvas or an
/// image bitmap of `width` x `height`), which the browser copies to the GPU itself. 0 when the
/// host has no such image or the browser refused it: the host then sends the pixels instead.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_texture_create_source(width: u32, height: u32, srgb: u32) -> u32 {
    with(0, |rt| {
        let t = rt.renderer.create_prop_texture_with(
            &mut rt.device,
            (width, height),
            srgb != 0,
            |d, t| d.write_texture_source(t, 0, 0, (width, height)),
        )?;
        Ok(t.raw())
    })
}

/// As [`ganbate_rt_texture_update`], from the image the host set for this call, which must be
/// `width` x `height`, the texture's size.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_texture_update_source(texture: u32, width: u32, height: u32) -> u32 {
    with(0, |rt| {
        rt.renderer.update_prop_texture_with(
            &mut rt.device,
            PropTexture::from_raw(texture),
            |d, t| d.write_texture_source(t, 0, 0, (width, height)),
        )?;
        Ok(1)
    })
}

/// A `width` x `height` texture of one colour (straight alpha, sRGB-encoded channels from 0 to
/// 1, like a canvas colour), filled on the GPU with no pixels from the host, with a full mip
/// chain; a later update of the same size replaces it.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_texture_create_clear(
    width: u32,
    height: u32,
    srgb: u32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
) -> u32 {
    with(0, |rt| {
        let t = rt.renderer.create_prop_texture_clear(
            &mut rt.device,
            (width, height),
            srgb != 0,
            [r, g, b, a],
        )?;
        Ok(t.raw())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_texture_destroy(texture: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .destroy_prop_texture(&mut rt.device, PropTexture::from_raw(texture));
        Ok(1)
    })
}

/// A mesh: `vertices` positions (3 floats each) at `positions`, normals likewise at `normals`
/// (null for flat normals), uvs (2 floats each) at `uvs` (null for none), and `index_count`
/// u32 indices at `indices` (counter-clockwise triangles).
///
/// # Safety
/// Each non-null pointer must address the floats or words its count implies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_mesh_create(
    positions: *const u8,
    vertices: usize,
    normals: *const u8,
    uvs: *const u8,
    indices: *const u8,
    index_count: usize,
) -> u32 {
    // SAFETY: the caller passes readable arrays of the stated sizes.
    let (p, n, uv, idx) = unsafe {
        (
            triples(&floats(positions, vertices * 3)),
            if normals.is_null() {
                vec![]
            } else {
                triples(&floats(normals, vertices * 3))
            },
            if uvs.is_null() {
                vec![]
            } else {
                floats(uvs, vertices * 2)
                    .chunks_exact(2)
                    .map(|c| [c[0], c[1]])
                    .collect()
            },
            words(indices, index_count),
        )
    };
    with(0, |rt| {
        let m = rt.renderer.create_prop_mesh(
            &mut rt.device,
            &MeshData {
                positions: &p,
                normals: &n,
                uvs: &uv,
                indices: &idx,
            },
        )?;
        Ok(m.raw())
    })
}

/// New positions (and normals, or null for flat ones) for a mesh's `vertices` vertices.
///
/// # Safety
/// As [`ganbate_rt_mesh_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_mesh_update(
    mesh: u32,
    positions: *const u8,
    normals: *const u8,
    vertices: usize,
) -> u32 {
    // SAFETY: the caller passes readable arrays of the stated sizes.
    let (p, n) = unsafe {
        (
            triples(&floats(positions, vertices * 3)),
            if normals.is_null() {
                vec![]
            } else {
                triples(&floats(normals, vertices * 3))
            },
        )
    };
    with(0, |rt| {
        rt.renderer
            .update_prop_mesh(&mut rt.device, PropMesh::from_raw(mesh), &p, &n)?;
        Ok(1)
    })
}

/// Destroys a mesh and every instance of it.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_mesh_destroy(mesh: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .destroy_prop_mesh(&mut rt.device, PropMesh::from_raw(mesh));
        Ok(1)
    })
}

/// A material from its JSON description (see `MaterialDesc::from_json`).
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_material_create(ptr: *const u8, len: usize) -> u32 {
    // SAFETY: the caller passes a readable range.
    let text = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        let desc = MaterialDesc::from_json(text)?;
        Ok(rt
            .renderer
            .create_prop_material(&mut rt.device, &desc)?
            .raw())
    })
}

/// Compiles a game's prop shader: WGSL defining `fn shade(in: ShadeInput) -> vec4f` (UTF-8,
/// `len` bytes at `ptr`). Returns its handle; 0 with the compiler's message on failure.
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_shader_create(ptr: *const u8, len: usize) -> u32 {
    // SAFETY: the caller passes a readable range.
    let text = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        let source = std::str::from_utf8(text).map_err(|e| format!("shader: {e}"))?;
        Ok(rt
            .renderer
            .create_prop_shader(&mut rt.device, source)?
            .raw())
    })
}

/// A material's twenty prop-shader parameters (floats at `ptr`), without a rebuild.
///
/// # Safety
/// `ptr` must address 80 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_material_params(material: u32, ptr: *const u8) -> u32 {
    // SAFETY: the caller passes twenty floats.
    let v = unsafe { floats(ptr, PROP_SHADER_PARAMS) };
    with(0, |rt| {
        let mut params = [0.0f32; PROP_SHADER_PARAMS];
        params.copy_from_slice(&v);
        rt.renderer
            .set_prop_material_params(PropMaterial::from_raw(material), &params)?;
        Ok(1)
    })
}

/// Changes a material in place (every instance that uses it changes with it).
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_material_update(
    material: u32,
    ptr: *const u8,
    len: usize,
) -> u32 {
    // SAFETY: the caller passes a readable range.
    let text = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        let desc = MaterialDesc::from_json(text)?;
        rt.renderer.update_prop_material(
            &mut rt.device,
            PropMaterial::from_raw(material),
            &desc,
        )?;
        Ok(1)
    })
}

/// Destroys a material and every instance that uses it.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_material_destroy(material: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .destroy_prop_material(&mut rt.device, PropMaterial::from_raw(material));
        Ok(1)
    })
}

/// Places `mesh` with `material` at the column-major model matrix (16 floats) at `matrix`.
///
/// # Safety
/// `matrix` must address 64 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_instance_create(
    mesh: u32,
    material: u32,
    matrix: *const u8,
) -> u32 {
    // SAFETY: the caller passes 16 floats.
    let m = unsafe { floats(matrix, 16) };
    with(0, |rt| {
        let i = rt.renderer.create_prop_instance(
            PropMesh::from_raw(mesh),
            PropMaterial::from_raw(material),
            &self::matrix(&m)?,
        )?;
        Ok(i.raw())
    })
}

/// Moves an instance and shows (`visible` 1) or hides it.
///
/// # Safety
/// `matrix` must address 64 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_instance_set(
    instance: u32,
    matrix: *const u8,
    visible: u32,
) -> u32 {
    // SAFETY: the caller passes 16 floats.
    let m = unsafe { floats(matrix, 16) };
    with(0, |rt| {
        rt.renderer.set_prop_instance(
            PropInstance::from_raw(instance),
            &self::matrix(&m)?,
            visible != 0,
        )?;
        Ok(1)
    })
}

/// Moves many instances at once: `count` records of 18 words at `ptr`, each the instance, 1 to
/// show it or 0 to hide it, and its 16-float model matrix. Stops at the first bad record.
///
/// # Safety
/// `ptr` must address `count * 72` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_instances_set(ptr: *const u8, count: usize) -> u32 {
    // SAFETY: the caller passes `count` records.
    let raw = unsafe { bytes(ptr, count * 72) };
    with(0, |rt| {
        for r in raw.chunks_exact(72) {
            let word = |i: usize| u32::from_le_bytes([r[i], r[i + 1], r[i + 2], r[i + 3]]);
            let m: Vec<f32> = r[8..]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            rt.renderer.set_prop_instance(
                PropInstance::from_raw(word(0)),
                &matrix(&m)?,
                word(4) != 0,
            )?;
        }
        Ok(1)
    })
}

/// Gives an instance another material.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_instance_material(instance: u32, material: u32) -> u32 {
    with(0, |rt| {
        rt.renderer.set_prop_instance_material(
            PropInstance::from_raw(instance),
            PropMaterial::from_raw(material),
        )?;
        Ok(1)
    })
}

/// Multiplies an instance's material colour and alpha by (`r`, `g`, `b`, `a`).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_instance_tint(instance: u32, r: f32, g: f32, b: f32, a: f32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .set_prop_instance_tint(PropInstance::from_raw(instance), [r, g, b, a])?;
        Ok(1)
    })
}

/// The pick layers (bits) an instance answers to; 0: never picked.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_instance_pick(instance: u32, layers: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .set_prop_instance_pick(PropInstance::from_raw(instance), layers)?;
        Ok(1)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_instance_destroy(instance: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .destroy_prop_instance(PropInstance::from_raw(instance));
        Ok(1)
    })
}

/// A game light from 15 floats: position (3), range, linear colour (3), intensity, a spot's
/// direction (3, all zero for a point light), its inner and outer half-angles (radians), 1 to
/// follow the lamps (lit only as the evening lights them), and how much of its light the props'
/// shadows take (0 casts none).
fn light_desc(v: &[f32]) -> LightDesc {
    let at = |i: usize| v.get(i).copied().unwrap_or(0.0);
    let direction = [at(8), at(9), at(10)];
    LightDesc {
        position: [at(0), at(1), at(2)],
        range: at(3),
        color: [at(4), at(5), at(6)],
        intensity: at(7),
        spot: direction.iter().any(|d| *d != 0.0).then_some(SpotCone {
            direction,
            inner: at(11),
            outer: at(12),
        }),
        follows_lamps: at(13) > 0.5,
        shadow: at(14),
    }
}

/// Hangs a game light (see `light_desc` for the 15 floats at `ptr`).
///
/// # Safety
/// `ptr` must address 60 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_light_create(ptr: *const u8) -> u32 {
    // SAFETY: the caller passes 15 floats.
    let v = unsafe { floats(ptr, 15) };
    with(0, |rt| Ok(rt.renderer.create_light(&light_desc(&v))?.raw()))
}

/// Moves, recolours or reshapes a game light (15 floats at `ptr`, as `ganbate_rt_light_create`).
///
/// # Safety
/// `ptr` must address 60 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_light_set(light: u32, ptr: *const u8) -> u32 {
    // SAFETY: the caller passes 15 floats.
    let v = unsafe { floats(ptr, 15) };
    with(0, |rt| {
        rt.renderer
            .set_light(GameLight::from_raw(light), &light_desc(&v))?;
        Ok(1)
    })
}

/// Puts a game light out.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_light_destroy(light: u32) -> u32 {
    with(0, |rt| {
        rt.renderer.destroy_light(GameLight::from_raw(light));
        Ok(1)
    })
}
