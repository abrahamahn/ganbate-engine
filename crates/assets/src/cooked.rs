//! GBEW cooked worlds. Little-endian, deterministic bytes; every count is bounded and checked
//! against the remaining input before allocation, and trailing bytes are rejected.
//!
//! Version 3 (written): header `GBEW`, version, then counts of meshes, instances, materials,
//! textures and lights, followed by those sections in that order.
//! - mesh: position count, index count, flags (1 normals, 2 UVs, 4 16-bit indices, 32 coded),
//!   colour (3 f32), material (`u32::MAX` for none), then either the plain arrays (positions
//!   and normals as 3 f32, UVs as 2 f32, indices as u16 (zero-padded to 4 bytes) or u32) or,
//!   when coded, the stream's length and a zlib stream zero-padded to 4 bytes holding, in
//!   order: the indices with vertices numbered in order of first use, each as a LEB128 varint
//!   (0 for the next new vertex, otherwise how far below it); positions on a 0.5 mm grid; then
//!   octahedral normals with 12-bit components; then UVs on a 1/8192 grid. Each attribute is
//!   coded as zigzag deltas from the previous vertex, stored byte plane by byte plane (all low
//!   bytes first): 4 planes per position or UV component, 2 per normal component. An index
//!   code takes at most 5 bytes, and a file's coded meshes decode to at most `MAX_BYTES`
//!   together (as much mesh data as an exact file holds).
//! - instance: mesh, column-major affine transform (16 f32).
//! - material: name byte length, UTF-8 name zero-padded to 4 bytes, base colour (4 f32),
//!   base colour texture, normal texture, normal scale, emissive (3 f32), emissive texture,
//!   metallic, roughness, alpha mode (0 opaque, 1 mask, 2 blend), alpha cutoff (0 unless mask),
//!   double-sided (0/1), shading (0 standard, 1 foliage, 2 water, 3 unlit), emission
//!   (0 constant, 1 lamp). Absent textures are `u32::MAX`.
//! - texture: width, height, sRGB (0/1), kind. Kind 0: level count (the complete chain, or 1
//!   for the top level alone), then each level's RGBA8 texels. Kind 1: an encoded image (MIME
//!   type length, the type zero-padded to 4 bytes, data length, the data zero-padded to 4).
//!   Kind 2: block-compressed mip chain (codec id `0=BC1`, `1=BC7`, `2=ASTC 4x4`, level count,
//!   then each level's compressed block bytes).
//! - light: position (3 f32), colour (3 f32), intensity, range.
//!
//! Version 4 (written by `encode_scene`): version 3 followed by an entity table: the entity
//! count, then per entity its 64-bit id (low then high word), name (byte length, UTF-8 zero-padded
//! to 4 bytes), parent (`u32::MAX` for none), world transform (16 f32), first instance and
//! instance count, first light and light count; then the table's data (byte length, UTF-8
//! canonical JSON zero-padded to 4 bytes). `decode` reads version 4 and ignores the table.
//!
//! Version 2 (read only): as version 3 with mesh flags 1 and 2 only (f32 positions and
//! normals, u32 indices) and textures without a kind (always the complete chain).
//!
//! Version 1 (read only): mesh and instance counts, then meshes as position count, index
//! count, colour, positions and indices, then instances as above.
use ganbate_core::{
    Vec3,
    world::{
        AlphaMode, Emission, EncodedImage, EntityRecord, EntityTable, IMAGE_TYPES, Instance,
        MAX_ENTITIES, MAX_ENTITY_DATA, MAX_LIGHTS, MAX_MATERIALS, MAX_NAME_BYTES, MAX_TEXTURE_SIZE,
        MAX_TEXTURES, Material, Mesh, PointLight, Shading, Texture, TextureCompression, Transform,
        World,
    },
};
const MAGIC: &[u8; 4] = b"GBEW";
pub const VERSION: u32 = 3;
/// The version of a world with an entity table (`encode_scene`).
pub const SCENE_VERSION: u32 = 4;
/// Mesh flags.
const NORMALS: u32 = 1;
const UVS: u32 = 2;
const SHORT_INDICES: u32 = 4;
const CODED: u32 = 32;
/// Coded meshes: the position grid (metres), the UV grid and the octahedral normal range.
const POSITION_GRID: f32 = 0.0005;
const UV_GRID: f32 = 1.0 / 8192.0;
const NORMAL_RANGE: f32 = 2047.0;
/// Grid values stay within ±2^30, so deltas fit 32-bit zigzag codes.
const GRID_LIMIT: f64 = 1_073_741_823.0;
pub const MAX_BYTES: usize = 256 * 1024 * 1024;
const MAX_MESHES: usize = 65_536;
/// Most instances a cooked world holds.
pub const MAX_INSTANCES: usize = 250_000;
const NONE: u32 = u32::MAX;
// Smallest encoded records, for validating counts before allocation.
const MESH_BYTES: u64 = 28;
const INSTANCE_BYTES: u64 = 68;
const MATERIAL_BYTES: u64 = 76;
const TEXTURE_BYTES: u64 = 20;
const LIGHT_BYTES: u64 = 32;
const ENTITY_BYTES: u64 = 96;

fn put_u32(v: &mut Vec<u8>, n: usize) -> Result<(), String> {
    v.extend_from_slice(
        &u32::try_from(n)
            .map_err(|_| "count exceeds u32")?
            .to_le_bytes(),
    );
    Ok(())
}
fn put_floats(v: &mut Vec<u8>, values: &[f32]) {
    for n in values {
        v.extend_from_slice(&n.to_le_bytes())
    }
}
fn put_index(v: &mut Vec<u8>, i: Option<u32>) {
    v.extend_from_slice(&i.unwrap_or(NONE).to_le_bytes());
}
/// `len` rounded up to a multiple of 4, in 64 bits (a 32-bit length near `u32::MAX` would
/// wrap in a 32-bit `usize`).
fn padded(len: usize) -> u64 {
    (len as u64).next_multiple_of(4)
}
/// Bytes of `k` 16-bit indices, zero-padded to 4.
fn short_index_bytes(k: usize) -> u64 {
    (2 * k as u64).next_multiple_of(4)
}

/// Bytes per vertex of a coded mesh's attribute planes.
fn plane_bytes(normals: bool, uvs: bool) -> u64 {
    12 + 4 * u64::from(normals) + 8 * u64::from(uvs)
}
/// Most bytes a coded mesh's stream inflates to: an index code of a 32-bit value takes at most
/// 5 bytes, then the attribute planes.
fn stream_bytes(n: usize, k: usize, normals: bool, uvs: bool) -> u64 {
    5 * k as u64 + n as u64 * plane_bytes(normals, uvs)
}
/// Bytes a coded mesh decodes to. The coded meshes of one file decode to at most `MAX_BYTES`
/// together, as much mesh data as an exact file can hold.
fn decoded_bytes(n: usize, k: usize, normals: bool, uvs: bool) -> u64 {
    n as u64 * (12 + 12 * u64::from(normals) + 8 * u64::from(uvs)) + 4 * k as u64
}

/// Version of an GBEW file, if it has the GBEW magic.
pub fn version(data: &[u8]) -> Option<u32> {
    let (magic, rest) = data.split_first_chunk::<4>()?;
    let (v, _) = rest.split_first_chunk::<4>()?;
    (magic == MAGIC).then(|| u32::from_le_bytes(*v))
}

/// Octahedral encoding of a unit vector: two components in -1..1.
fn oct_encode(n: Vec3) -> [f32; 2] {
    let Vec3(x, y, z) = n;
    let l1 = x.abs() + y.abs() + z.abs();
    if l1.is_nan() || l1 <= 0.0 || !l1.is_finite() {
        return [0.0, 0.0];
    }
    let (mut px, mut py) = (x / l1, y / l1);
    if z < 0.0 {
        let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
        (px, py) = ((1.0 - py.abs()) * sign(px), (1.0 - px.abs()) * sign(py));
    }
    [px, py]
}

fn oct_decode([mut px, mut py]: [f32; 2]) -> Vec3 {
    let z = 1.0 - px.abs() - py.abs();
    if z < 0.0 {
        let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
        (px, py) = ((1.0 - py.abs()) * sign(px), (1.0 - px.abs()) * sign(py));
    }
    let l = (px * px + py * py + z * z).sqrt();
    if l > 0.0 {
        Vec3(px / l, py / l, z / l)
    } else {
        Vec3(0.0, 1.0, 0.0)
    }
}

fn pad4(v: &mut Vec<u8>) {
    v.resize(v.len().next_multiple_of(4), 0);
}

fn zigzag(v: i64) -> u32 {
    ((v << 1) ^ (v >> 63)) as u32
}

fn unzigzag(u: u32) -> i64 {
    i64::from(u >> 1) ^ -i64::from(u & 1)
}

fn put_varint(v: &mut Vec<u8>, mut n: u64) {
    while n >= 0x80 {
        v.push((n as u8 & 0x7F) | 0x80);
        n >>= 7;
    }
    v.push(n as u8);
}

fn on_grid(c: f32, step: f32, what: &str) -> Result<i64, String> {
    let q = (f64::from(c) / f64::from(step)).round();
    if q.is_nan() || q.abs() > GRID_LIMIT {
        return Err(format!(
            "{what} {c} is too far out to code; cook it exactly"
        ));
    }
    Ok(q as i64)
}

/// A mesh's coded stream (see the module notes) and its vertex count.
fn code_mesh(m: &Mesh) -> Result<(usize, Vec<u8>), String> {
    let has_normals = !m.normals.is_empty();
    let has_uvs = !m.uvs.is_empty();
    let mut first = vec![u32::MAX; m.positions.len()];
    let mut order = Vec::with_capacity(m.positions.len());
    let est_bytes = m.indices.len().saturating_mul(2).saturating_add(
        m.positions
            .len()
            .saturating_mul(plane_bytes(has_normals, has_uvs) as usize),
    );
    let mut v = Vec::with_capacity(est_bytes);
    let mut next: u64 = 0;
    for &i in &m.indices {
        let i = i as usize;
        if first[i] == u32::MAX {
            first[i] = order.len() as u32;
            order.push(i);
        }
        let k = u64::from(first[i]);
        if k == next {
            put_varint(&mut v, 0);
            next += 1;
        } else {
            put_varint(&mut v, next - k);
        }
    }
    drop(first);

    let n = order.len();
    // Positions: 3 components, 4 byte planes.
    {
        let count = n * 3;
        let start = v.len();
        v.resize(start + count * 4, 0);
        let (p0, rest) = v[start..].split_at_mut(count);
        let (p1, rest) = rest.split_at_mut(count);
        let (p2, p3) = rest.split_at_mut(count);
        let mut prev = [0i64; 3];
        for (vi, &orig) in order.iter().enumerate() {
            let Vec3(x, y, z) = m.positions[orig];
            let base = vi * 3;
            for (c, val) in [x, y, z].into_iter().enumerate() {
                let q = on_grid(val, POSITION_GRID, "position")?;
                let bytes = zigzag(q - prev[c]).to_le_bytes();
                prev[c] = q;
                let idx = base + c;
                p0[idx] = bytes[0];
                p1[idx] = bytes[1];
                p2[idx] = bytes[2];
                p3[idx] = bytes[3];
            }
        }
    }
    // Normals: 2 octahedral components, 2 byte planes.
    if has_normals {
        let count = n * 2;
        let start = v.len();
        v.resize(start + count * 2, 0);
        let (n0, n1) = v[start..].split_at_mut(count);
        let mut prev = [0i64; 2];
        for (vi, &orig) in order.iter().enumerate() {
            let oct = oct_encode(m.normals[orig]);
            let base = vi * 2;
            for (c, val) in oct.into_iter().enumerate() {
                let q = (val * NORMAL_RANGE).round() as i64;
                let bytes = zigzag(q - prev[c]).to_le_bytes();
                prev[c] = q;
                let idx = base + c;
                n0[idx] = bytes[0];
                n1[idx] = bytes[1];
            }
        }
    }
    // UVs: 2 components, 4 byte planes.
    if has_uvs {
        let count = n * 2;
        let start = v.len();
        v.resize(start + count * 4, 0);
        let (u0, rest) = v[start..].split_at_mut(count);
        let (u1, rest) = rest.split_at_mut(count);
        let (u2, u3) = rest.split_at_mut(count);
        let mut prev = [0i64; 2];
        for (vi, &orig) in order.iter().enumerate() {
            let uv = m.uvs[orig];
            let base = vi * 2;
            for (c, val) in uv.into_iter().enumerate() {
                let q = on_grid(val, UV_GRID, "uv")?;
                let bytes = zigzag(q - prev[c]).to_le_bytes();
                prev[c] = q;
                let idx = base + c;
                u0[idx] = bytes[0];
                u1[idx] = bytes[1];
                u2[idx] = bytes[2];
                u3[idx] = bytes[3];
            }
        }
    }
    Ok((n, crate::zlib::compress(&v)))
}

/// Reads a coded mesh's stream back into its arrays without intermediate plane/delta buffers.
#[allow(clippy::type_complexity)]
fn decode_mesh(
    data: &[u8],
    n: usize,
    k: usize,
    normals: bool,
    uvs: bool,
) -> Result<(Vec<Vec3>, Vec<Vec3>, Vec<[f32; 2]>, Vec<u32>), String> {
    // Each index code takes at least a byte and each vertex its planes: counts the stream
    // cannot hold fail before anything is reserved for them.
    if (data.len() as u64) < k as u64 + n as u64 * plane_bytes(normals, uvs) {
        return Err("truncated coded mesh".into());
    }
    let mut at = 0usize;
    let mut indices = Vec::with_capacity(k);
    let mut next: u64 = 0;
    for _ in 0..k {
        let b0 = *data.get(at).ok_or("truncated mesh indices")?;
        at += 1;
        let code: u64 = if b0 < 0x80 {
            u64::from(b0)
        } else {
            let mut c = u64::from(b0 & 0x7F);
            let mut shift = 7;
            loop {
                let b = *data.get(at).ok_or("truncated mesh indices")?;
                at += 1;
                if shift > 28 {
                    return Err("overlong index code".into());
                }
                c |= u64::from(b & 0x7F) << shift;
                shift += 7;
                if b & 0x80 == 0 {
                    break;
                }
            }
            c
        };
        let index = if code == 0 {
            next += 1;
            next - 1
        } else {
            next.checked_sub(code)
                .ok_or("index code before the first vertex")?
        };
        if index >= n as u64 {
            return Err("index outside the mesh".into());
        }
        indices.push(index as u32);
    }
    if next != n as u64 {
        return Err("coded mesh has unused vertices".into());
    }

    let take_planes = |at: &mut usize, count: usize, width: usize| -> Result<&[u8], String> {
        let end = count
            .checked_mul(width)
            .and_then(|b| b.checked_add(*at))
            .ok_or("truncated mesh attributes")?;
        let bytes = data.get(*at..end).ok_or("truncated mesh attributes")?;
        *at = end;
        Ok(bytes)
    };

    let pos_count = n.checked_mul(3).ok_or("truncated mesh attributes")?;
    let p_bytes = take_planes(&mut at, pos_count, 4)?;
    let (p0, rest) = p_bytes.split_at(pos_count);
    let (p1, rest) = rest.split_at(pos_count);
    let (p2, p3) = rest.split_at(pos_count);
    let mut positions = Vec::with_capacity(n);
    let mut prev_p = [0i64; 3];
    let pos_step = f64::from(POSITION_GRID);
    for vi in 0..n {
        let base = vi * 3;
        let z0 = u32::from_le_bytes([p0[base], p1[base], p2[base], p3[base]]);
        let z1 = u32::from_le_bytes([p0[base + 1], p1[base + 1], p2[base + 1], p3[base + 1]]);
        let z2 = u32::from_le_bytes([p0[base + 2], p1[base + 2], p2[base + 2], p3[base + 2]]);
        prev_p[0] += unzigzag(z0);
        prev_p[1] += unzigzag(z1);
        prev_p[2] += unzigzag(z2);
        positions.push(Vec3(
            (prev_p[0] as f64 * pos_step) as f32,
            (prev_p[1] as f64 * pos_step) as f32,
            (prev_p[2] as f64 * pos_step) as f32,
        ));
    }

    let normals = if normals {
        let norm_count = n.checked_mul(2).ok_or("truncated mesh attributes")?;
        let n_bytes = take_planes(&mut at, norm_count, 2)?;
        let (n0, n1) = n_bytes.split_at(norm_count);
        let mut out_normals = Vec::with_capacity(n);
        let mut prev_n = [0i64; 2];
        for vi in 0..n {
            let base = vi * 2;
            let z0 = u32::from(u16::from_le_bytes([n0[base], n1[base]]));
            let z1 = u32::from(u16::from_le_bytes([n0[base + 1], n1[base + 1]]));
            prev_n[0] += unzigzag(z0);
            prev_n[1] += unzigzag(z1);
            out_normals.push(oct_decode([
                (prev_n[0] as f32 / NORMAL_RANGE).clamp(-1.0, 1.0),
                (prev_n[1] as f32 / NORMAL_RANGE).clamp(-1.0, 1.0),
            ]));
        }
        out_normals
    } else {
        vec![]
    };

    let uvs = if uvs {
        let uv_count = n.checked_mul(2).ok_or("truncated mesh attributes")?;
        let u_bytes = take_planes(&mut at, uv_count, 4)?;
        let (u0, rest) = u_bytes.split_at(uv_count);
        let (u1, rest) = rest.split_at(uv_count);
        let (u2, u3) = rest.split_at(uv_count);
        let mut out_uvs = Vec::with_capacity(n);
        let mut prev_u = [0i64; 2];
        let uv_step = f64::from(UV_GRID);
        for vi in 0..n {
            let base = vi * 2;
            let z0 = u32::from_le_bytes([u0[base], u1[base], u2[base], u3[base]]);
            let z1 = u32::from_le_bytes([u0[base + 1], u1[base + 1], u2[base + 1], u3[base + 1]]);
            prev_u[0] += unzigzag(z0);
            prev_u[1] += unzigzag(z1);
            out_uvs.push([
                (prev_u[0] as f64 * uv_step) as f32,
                (prev_u[1] as f64 * uv_step) as f32,
            ]);
        }
        out_uvs
    } else {
        vec![]
    };

    if at != data.len() {
        return Err("trailing bytes in a coded mesh".into());
    }
    Ok((positions, normals, uvs, indices))
}

/// Encodes a world exactly: f32 positions and normals (16-bit indices where they fit, and
/// encoded images as they are), so decoding gives the same world back.
pub fn encode(w: &World) -> Result<Vec<u8>, String> {
    encode_with(w, false)
}

/// Encodes a world compactly for delivery: each mesh coded (see the module notes) with
/// positions within 0.25 mm, normals within about 0.05 degrees and UVs within 1/16384 of the
/// original. Vertices come back in order of first use, without the unused ones.
pub fn encode_compact(w: &World) -> Result<Vec<u8>, String> {
    encode_with(w, true)
}

/// Encodes a cooked scene: the world (compactly when `compact`) and its entity table, as
/// version 4. Readers of version 3 cannot load it; `encode` still writes version 3.
pub fn encode_scene(w: &World, table: &EntityTable, compact: bool) -> Result<Vec<u8>, String> {
    table.validate(w)?;
    let mut v = encode_with(w, compact)?;
    v[4..8].copy_from_slice(&SCENE_VERSION.to_le_bytes());
    put_u32(&mut v, table.entities.len())?;
    for e in &table.entities {
        v.extend_from_slice(&(e.id as u32).to_le_bytes());
        v.extend_from_slice(&((e.id >> 32) as u32).to_le_bytes());
        put_u32(&mut v, e.name.len())?;
        v.extend_from_slice(e.name.as_bytes());
        pad4(&mut v);
        put_index(&mut v, e.parent);
        put_floats(&mut v, &e.transform.0);
        for n in [
            e.instances.start,
            e.instances.end - e.instances.start,
            e.lights.start,
            e.lights.end - e.lights.start,
        ] {
            v.extend_from_slice(&n.to_le_bytes());
        }
    }
    put_u32(&mut v, table.data.len())?;
    v.extend_from_slice(table.data.as_bytes());
    pad4(&mut v);
    if v.len() > MAX_BYTES {
        return Err("cooked file too large".into());
    }
    Ok(v)
}

fn encode_with(w: &World, compact: bool) -> Result<Vec<u8>, String> {
    w.validate()?;
    if w.meshes.len() > MAX_MESHES || w.instances.len() > MAX_INSTANCES {
        return Err("too many meshes or instances".into());
    }
    let mesh_est: usize = w
        .meshes
        .iter()
        .map(|m| {
            let base = 28 + m.indices.len() * 2;
            if compact {
                base + m.positions.len() * 8
            } else {
                base + m.positions.len() * 12 + m.normals.len() * 12 + m.uvs.len() * 8
            }
        })
        .sum();
    let tex_est: usize = w
        .textures
        .iter()
        .map(|t| match &t.encoded {
            Some(e) => 28 + e.mime.len() + e.data.len(),
            None => 20 + t.levels.iter().map(Vec::len).sum::<usize>(),
        })
        .sum();
    let cap = (28usize)
        .saturating_add(mesh_est)
        .saturating_add(w.instances.len().saturating_mul(68))
        .saturating_add(w.materials.len().saturating_mul(88))
        .saturating_add(tex_est)
        .saturating_add(w.lights.len().saturating_mul(32))
        .min(MAX_BYTES);
    let mut v = Vec::with_capacity(cap);
    v.extend_from_slice(MAGIC);
    put_u32(&mut v, VERSION as usize)?;
    for n in [
        w.meshes.len(),
        w.instances.len(),
        w.materials.len(),
        w.textures.len(),
        w.lights.len(),
    ] {
        put_u32(&mut v, n)?;
    }
    // What the coded meshes decode to, as `decode` counts it.
    let mut budget = MAX_BYTES as u64;
    for m in &w.meshes {
        let mut flags = 0;
        if !m.normals.is_empty() {
            flags |= NORMALS;
        }
        if !m.uvs.is_empty() {
            flags |= UVS;
        }
        if compact {
            let (n, stream) = code_mesh(m)?;
            let size = decoded_bytes(n, m.indices.len(), flags & NORMALS != 0, flags & UVS != 0);
            budget = budget
                .checked_sub(size)
                .ok_or("compact meshes decode to more than a file may hold")?;
            put_u32(&mut v, n)?;
            put_u32(&mut v, m.indices.len())?;
            put_u32(&mut v, (flags | CODED) as usize)?;
            put_floats(&mut v, &m.color);
            put_index(&mut v, m.material);
            put_u32(&mut v, stream.len())?;
            v.extend_from_slice(&stream);
            pad4(&mut v);
        } else {
            let n = m.positions.len();
            if n <= 65_536 {
                flags |= SHORT_INDICES;
            }
            put_u32(&mut v, n)?;
            put_u32(&mut v, m.indices.len())?;
            put_u32(&mut v, flags as usize)?;
            put_floats(&mut v, &m.color);
            put_index(&mut v, m.material);
            for &Vec3(x, y, z) in m.positions.iter().chain(&m.normals) {
                put_floats(&mut v, &[x, y, z]);
            }
            for uv in &m.uvs {
                put_floats(&mut v, uv);
            }
            if flags & SHORT_INDICES != 0 {
                for &i in &m.indices {
                    v.extend_from_slice(&(i as u16).to_le_bytes());
                }
                pad4(&mut v);
            } else {
                for i in &m.indices {
                    v.extend_from_slice(&i.to_le_bytes());
                }
            }
        }
        if v.len() > MAX_BYTES {
            return Err("cooked file too large".into());
        }
    }
    for i in &w.instances {
        v.extend_from_slice(&i.mesh.to_le_bytes());
        put_floats(&mut v, &i.transform.0);
    }
    for m in &w.materials {
        put_u32(&mut v, m.name.len())?;
        v.extend_from_slice(m.name.as_bytes());
        v.resize(v.len() + (padded(m.name.len()) as usize - m.name.len()), 0);
        put_floats(&mut v, &m.base_color);
        put_index(&mut v, m.base_color_texture);
        put_index(&mut v, m.normal_texture);
        put_floats(&mut v, &[m.normal_scale]);
        put_floats(&mut v, &m.emissive);
        put_index(&mut v, m.emissive_texture);
        put_floats(&mut v, &[m.metallic, m.roughness]);
        let (mode, cutoff) = match m.alpha {
            AlphaMode::Opaque => (0, 0.),
            AlphaMode::Mask(c) => (1, c),
            AlphaMode::Blend => (2, 0.),
        };
        put_u32(&mut v, mode)?;
        put_floats(&mut v, &[cutoff]);
        put_u32(&mut v, usize::from(m.double_sided))?;
        put_u32(&mut v, m.shading as usize)?;
        put_u32(&mut v, m.emission as usize)?;
    }
    for t in &w.textures {
        put_u32(&mut v, t.width as usize)?;
        put_u32(&mut v, t.height as usize)?;
        put_u32(&mut v, usize::from(t.srgb))?;
        if let Some(comp) = t.compression() {
            put_u32(&mut v, 2)?;
            put_u32(&mut v, comp.codec_id() as usize)?;
            put_u32(&mut v, t.levels.len())?;
            for level in &t.levels {
                v.extend_from_slice(level);
            }
        } else {
            match &t.encoded {
                None => {
                    put_u32(&mut v, 0)?;
                    put_u32(&mut v, t.levels.len())?;
                    for level in &t.levels {
                        v.extend_from_slice(level);
                    }
                }
                Some(e) => {
                    put_u32(&mut v, 1)?;
                    put_u32(&mut v, e.mime.len())?;
                    v.extend_from_slice(e.mime.as_bytes());
                    pad4(&mut v);
                    put_u32(&mut v, e.data.len())?;
                    v.extend_from_slice(&e.data);
                    pad4(&mut v);
                }
            }
        }
        if v.len() > MAX_BYTES {
            return Err("cooked file too large".into());
        }
    }
    for l in &w.lights {
        let Vec3(x, y, z) = l.position;
        put_floats(&mut v, &[x, y, z]);
        put_floats(&mut v, &l.color);
        put_floats(&mut v, &[l.intensity, l.range]);
    }
    if v.len() > MAX_BYTES {
        return Err("cooked file too large".into());
    }
    Ok(v)
}

struct Read<'a>(&'a [u8]);
impl<'a> Read<'a> {
    #[inline]
    fn bytes(&mut self, n: u64) -> Result<&'a [u8], String> {
        let n = usize::try_from(n).map_err(|_| "truncated cooked scene")?;
        let (b, left) = self.0.split_at_checked(n).ok_or("truncated cooked scene")?;
        self.0 = left;
        Ok(b)
    }
    #[inline]
    fn word(&mut self) -> Result<[u8; 4], String> {
        Ok(self.bytes(4)?.try_into().expect("four bytes"))
    }
    #[inline]
    fn uint(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.word()?))
    }
    #[inline]
    fn floats<const N: usize>(&mut self) -> Result<[f32; N], String> {
        let raw = self.bytes(4 * N as u64)?;
        let mut v = [0.0; N];
        for (f, c) in v.iter_mut().zip(raw.chunks_exact(4)) {
            *f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        Ok(v)
    }
    fn vec3s(&mut self, n: usize) -> Result<Vec<Vec3>, String> {
        let raw = self.bytes(12 * n as u64)?;
        Ok(raw
            .chunks_exact(12)
            .map(|c| {
                Vec3(
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                    f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                    f32::from_le_bytes([c[8], c[9], c[10], c[11]]),
                )
            })
            .collect())
    }
    fn pairs(&mut self, n: usize) -> Result<Vec<[f32; 2]>, String> {
        let raw = self.bytes(8 * n as u64)?;
        Ok(raw
            .chunks_exact(8)
            .map(|c| {
                [
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                    f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                ]
            })
            .collect())
    }
    fn u32s(&mut self, n: usize) -> Result<Vec<u32>, String> {
        let raw = self.bytes(4 * n as u64)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
    #[inline]
    fn index(&mut self) -> Result<Option<u32>, String> {
        Ok(Some(self.uint()?).filter(|&i| i != NONE))
    }
    #[inline]
    fn choice(&mut self, options: u32) -> Result<u32, String> {
        Some(self.uint()?)
            .filter(|&v| v < options)
            .ok_or_else(|| "invalid enumeration in cooked scene".into())
    }
    /// Fail before allocating `bytes` that the input cannot hold.
    #[inline]
    fn holds(&self, bytes: u64) -> Result<(), String> {
        if bytes > self.0.len() as u64 {
            return Err("cooked counts exceed file size".into());
        }
        Ok(())
    }
}

fn instances(r: &mut Read, count: usize) -> Result<Vec<Instance>, String> {
    let raw = r.bytes(INSTANCE_BYTES * count as u64)?;
    Ok(raw
        .chunks_exact(INSTANCE_BYTES as usize)
        .map(|c| {
            let mesh = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let mut m = [0.0f32; 16];
            for (dst, w) in m.iter_mut().zip(c[4..].chunks_exact(4)) {
                *dst = f32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            }
            Instance {
                mesh,
                transform: Transform(m),
            }
        })
        .collect())
}

fn decode_v1(r: &mut Read) -> Result<World, String> {
    let meshes = r.uint()? as usize;
    let instance_count = r.uint()? as usize;
    if meshes > MAX_MESHES || instance_count > MAX_INSTANCES {
        return Err("invalid world counts".into());
    }
    r.holds(20 * meshes as u64 + INSTANCE_BYTES * instance_count as u64)?;
    let mut world = World {
        meshes: Vec::with_capacity(meshes),
        ..World::default()
    };
    for _ in 0..meshes {
        let n = r.uint()? as usize;
        let k = r.uint()? as usize;
        let color = r.floats()?;
        if k % 3 != 0 {
            return Err("invalid mesh ranges".into());
        }
        r.holds(n as u64 * 12 + k as u64 * 4)?;
        let positions = r.vec3s(n)?;
        let indices = r.u32s(k)?;
        world.meshes.push(Mesh {
            positions,
            indices,
            color,
            ..Mesh::default()
        });
    }
    world.instances = instances(r, instance_count)?;
    Ok(world)
}

fn decode_v2(r: &mut Read, version: u32) -> Result<World, String> {
    let [meshes, instance_count, materials, textures, lights] =
        [(); 5].map(|()| r.uint().map(|n| n as usize));
    let (meshes, instance_count, materials, textures, lights) =
        (meshes?, instance_count?, materials?, textures?, lights?);
    if meshes > MAX_MESHES
        || instance_count > MAX_INSTANCES
        || materials > MAX_MATERIALS
        || textures > MAX_TEXTURES
        || lights > MAX_LIGHTS
    {
        return Err("invalid world counts".into());
    }
    r.holds(
        MESH_BYTES * meshes as u64
            + INSTANCE_BYTES * instance_count as u64
            + MATERIAL_BYTES * materials as u64
            + TEXTURE_BYTES * textures as u64
            + LIGHT_BYTES * lights as u64,
    )?;
    let mut world = World {
        meshes: Vec::with_capacity(meshes),
        materials: Vec::with_capacity(materials),
        textures: Vec::with_capacity(textures),
        lights: Vec::with_capacity(lights),
        ..World::default()
    };
    let known = if version >= 3 {
        NORMALS | UVS | SHORT_INDICES | CODED
    } else {
        NORMALS | UVS
    };
    let mut budget = MAX_BYTES as u64;
    for _ in 0..meshes {
        let n = r.uint()? as usize;
        let k = r.uint()? as usize;
        let flags = r.uint()?;
        let color = r.floats()?;
        let material = r.index()?;
        if k % 3 != 0
            || flags & !known != 0
            || (flags & SHORT_INDICES != 0 && (n > 65_536 || flags & CODED != 0))
        {
            return Err("invalid mesh ranges".into());
        }
        let (normals, uvs) = (flags & NORMALS != 0, flags & UVS != 0);
        let (positions, normals, uvs, indices) = if flags & CODED != 0 {
            let len = r.uint()? as usize;
            let raw = r.bytes(padded(len))?;
            if raw[len..].iter().any(|&b| b != 0) {
                return Err("nonzero mesh stream padding".into());
            }
            // Every vertex is used, so a mesh has at most as many vertices as indices.
            if n > k || k > MAX_BYTES {
                return Err("invalid coded mesh counts".into());
            }
            // What the mesh decodes to comes out of the file's budget before its stream is
            // inflated, so a small file cannot claim (or inflate to) more than it may hold.
            budget = budget
                .checked_sub(decoded_bytes(n, k, normals, uvs))
                .ok_or("coded meshes decode to more than a file may hold")?;
            let limit = usize::try_from(stream_bytes(n, k, normals, uvs))
                .map_err(|_| "invalid coded mesh counts")?;
            let expect = k
                .saturating_mul(2)
                .saturating_add(n.saturating_mul(plane_bytes(normals, uvs) as usize))
                .min(limit);
            let data = crate::zlib::decompress_with_capacity(&raw[..len], limit, expect)?;
            decode_mesh(&data, n, k, normals, uvs)?
        } else {
            let n64 = n as u64;
            let index_bytes = if flags & SHORT_INDICES != 0 {
                short_index_bytes(k)
            } else {
                4 * k as u64
            };
            let per_vertex = 12 + 12 * u64::from(normals) + 8 * u64::from(uvs);
            r.holds(n64 * per_vertex + index_bytes)?;
            let positions = r.vec3s(n)?;
            let normals = if normals { r.vec3s(n)? } else { vec![] };
            let uvs = if uvs { r.pairs(n)? } else { vec![] };
            let indices = if flags & SHORT_INDICES != 0 {
                let raw = r.bytes(short_index_bytes(k))?;
                raw.chunks_exact(2)
                    .take(k)
                    .map(|c| u32::from(u16::from_le_bytes([c[0], c[1]])))
                    .collect()
            } else {
                r.u32s(k)?
            };
            (positions, normals, uvs, indices)
        };
        world.meshes.push(Mesh {
            positions,
            normals,
            uvs,
            indices,
            color,
            material,
        });
    }
    world.instances = instances(r, instance_count)?;
    for _ in 0..materials {
        let len = r.uint()? as usize;
        if len > MAX_NAME_BYTES {
            return Err("material name too long".into());
        }
        let raw = r.bytes(padded(len))?;
        if raw[len..].iter().any(|&b| b != 0) {
            return Err("nonzero material name padding".into());
        }
        let name =
            String::from_utf8(raw[..len].to_vec()).map_err(|_| "material name is not UTF-8")?;
        let base_color = r.floats()?;
        let base_color_texture = r.index()?;
        let normal_texture = r.index()?;
        let [normal_scale] = r.floats()?;
        let emissive = r.floats()?;
        let emissive_texture = r.index()?;
        let [metallic, roughness] = r.floats()?;
        let mode = r.choice(3)?;
        let [cutoff] = r.floats()?;
        if mode != 1 && cutoff.to_bits() != 0 {
            return Err("alpha cutoff without alpha mask".into());
        }
        let alpha = [AlphaMode::Opaque, AlphaMode::Mask(cutoff), AlphaMode::Blend][mode as usize];
        let double_sided = r.choice(2)? == 1;
        let shading = [
            Shading::Standard,
            Shading::Foliage,
            Shading::Water,
            Shading::Unlit,
        ][r.choice(4)? as usize];
        let emission = [Emission::Constant, Emission::Lamp][r.choice(2)? as usize];
        world.materials.push(Material {
            name,
            base_color,
            base_color_texture,
            normal_texture,
            normal_scale,
            emissive,
            emissive_texture,
            metallic,
            roughness,
            alpha,
            double_sided,
            shading,
            emission,
        });
    }
    for _ in 0..textures {
        let [width, height] = [r.uint()?, r.uint()?];
        let srgb = r.choice(2)? == 1;
        let limit = 1..=MAX_TEXTURE_SIZE;
        if !limit.contains(&width) || !limit.contains(&height) {
            return Err("invalid texture size".into());
        }
        let kind = if version >= 3 { r.choice(3)? } else { 0 };
        if kind == 1 {
            let len = r.uint()? as usize;
            if len > 64 {
                return Err("image type too long".into());
            }
            let raw = r.bytes(padded(len))?;
            let mime = std::str::from_utf8(&raw[..len]).map_err(|_| "image type is not UTF-8")?;
            if !IMAGE_TYPES.contains(&mime) || raw[len..].iter().any(|&b| b != 0) {
                return Err(format!("unsupported image type {mime:?}"));
            }
            let size = r.uint()? as usize;
            r.holds(padded(size))?;
            let data = r.bytes(padded(size))?;
            if data[size..].iter().any(|&b| b != 0) {
                return Err("nonzero image padding".into());
            }
            world.textures.push(Texture {
                width,
                height,
                srgb,
                levels: vec![],
                encoded: Some(EncodedImage {
                    mime: mime.to_string(),
                    data: data[..size].to_vec(),
                }),
            });
            continue;
        }
        if kind == 2 {
            let codec = TextureCompression::from_codec_id(r.uint()?)
                .ok_or_else(|| "unsupported compressed texture codec".to_string())?;
            let count = r.uint()? as usize;
            let full = Texture::level_count(width, height);
            if count != full && count != 1 {
                return Err("invalid texture size or mip chain".into());
            }
            let sizes: Vec<u64> = (0..count)
                .map(|i| {
                    let (w, h) = Texture::level_extent(width, height, i);
                    codec.mip_byte_len(w, h) as u64
                })
                .collect();
            r.holds(sizes.iter().sum())?;
            let levels = sizes
                .into_iter()
                .map(|n| r.bytes(n).map(<[u8]>::to_vec))
                .collect::<Result<_, _>>()?;
            world
                .textures
                .push(Texture::compressed(width, height, srgb, codec, levels));
            continue;
        }
        let count = r.uint()? as usize;
        let full = Texture::level_count(width, height);
        if count != full && !(version >= 3 && count == 1) {
            return Err("invalid texture size or mip chain".into());
        }
        let sizes: Vec<u64> = (0..count)
            .map(|i| {
                let (w, h) = Texture::level_extent(width, height, i);
                u64::from(w) * u64::from(h) * 4
            })
            .collect();
        r.holds(sizes.iter().sum())?;
        let levels = sizes
            .into_iter()
            .map(|n| r.bytes(n).map(<[u8]>::to_vec))
            .collect::<Result<_, _>>()?;
        world.textures.push(Texture {
            width,
            height,
            srgb,
            levels,
            encoded: None,
        });
    }
    for _ in 0..lights {
        let [x, y, z] = r.floats()?;
        let color = r.floats()?;
        let [intensity, range] = r.floats()?;
        world.lights.push(PointLight {
            position: Vec3(x, y, z),
            color,
            intensity,
            range,
        });
    }
    Ok(world)
}

fn entity_table(r: &mut Read, w: &World) -> Result<EntityTable, String> {
    let count = r.uint()? as usize;
    if count > MAX_ENTITIES {
        return Err("too many entities".into());
    }
    r.holds(ENTITY_BYTES * count as u64)?;
    let mut entities = Vec::with_capacity(count);
    for _ in 0..count {
        let lo = u64::from(r.uint()?);
        let hi = u64::from(r.uint()?);
        let len = r.uint()? as usize;
        if len > MAX_NAME_BYTES {
            return Err("entity name too long".into());
        }
        let raw = r.bytes(padded(len))?;
        if raw[len..].iter().any(|&b| b != 0) {
            return Err("nonzero entity name padding".into());
        }
        let name =
            String::from_utf8(raw[..len].to_vec()).map_err(|_| "entity name is not UTF-8")?;
        let parent = r.index()?;
        let transform = Transform(r.floats()?);
        let [first, n, first_light, lights] = [(); 4].map(|()| r.uint());
        let (first, n, first_light, lights) = (first?, n?, first_light?, lights?);
        let end = first
            .checked_add(n)
            .ok_or("entity instances out of range")?;
        let light_end = first_light
            .checked_add(lights)
            .ok_or("entity lights out of range")?;
        entities.push(EntityRecord {
            id: lo | hi << 32,
            name,
            parent,
            transform,
            instances: first..end,
            lights: first_light..light_end,
        });
    }
    let len = r.uint()? as usize;
    if len > MAX_ENTITY_DATA {
        return Err("entity data too large".into());
    }
    let raw = r.bytes(padded(len))?;
    if raw[len..].iter().any(|&b| b != 0) {
        return Err("nonzero entity data padding".into());
    }
    let data = String::from_utf8(raw[..len].to_vec()).map_err(|_| "entity data is not UTF-8")?;
    let table = EntityTable { entities, data };
    table.validate(w)?;
    Ok(table)
}

/// Decode an GBEW file of version 1, 2, 3 or 4 (a version 4 file's entity table is checked and
/// dropped; `decode_scene` keeps it).
pub fn decode(data: &[u8]) -> Result<World, String> {
    decode_scene(data).map(|(w, _)| w)
}

/// Decode an GBEW file with its entity table (empty before version 4).
pub fn decode_scene(data: &[u8]) -> Result<(World, EntityTable), String> {
    if data.len() > MAX_BYTES {
        return Err("cooked file too large".into());
    }
    let mut r = Read(data);
    if &r.word()? != MAGIC {
        return Err("unsupported GBEW header or version".into());
    }
    let version = r.uint()?;
    let world = match version {
        1 => decode_v1(&mut r)?,
        v @ (2..=4) => decode_v2(&mut r, v)?,
        _ => return Err("unsupported GBEW header or version".into()),
    };
    world.validate()?;
    let table = if version == SCENE_VERSION {
        entity_table(&mut r, &world)?
    } else {
        EntityTable::default()
    };
    if !r.0.is_empty() {
        return Err("trailing bytes after scene".into());
    }
    Ok((world, table))
}
