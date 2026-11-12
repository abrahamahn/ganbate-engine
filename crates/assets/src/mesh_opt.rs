//! Mesh vertex deduplication, GPU post-transform vertex cache reordering (Tom Forsyth's
//! linear-time algorithm), and pre-transform sequential vertex fetch remapping.
use std::collections::HashMap;

use ganbate_core::{Vec3, world::Mesh};

const CACHE_SIZE: usize = 32;
/// Above this vertex valence, per-step score delta propagation across all incident triangles is
/// skipped and candidate selection uses a lazy cursor so pathological star/cone hubs still run
/// in strict O(T + V) worst-case time.
const MAX_HUB_DEGREE: u32 = 64;

#[inline]
fn canon_bits(f: f32) -> u32 {
    if f == 0.0 { 0 } else { f.to_bits() }
}

#[inline]
fn unique_tri_verts(a: u32, b: u32, c: u32) -> ([u32; 3], usize) {
    let mut out = [a, 0, 0];
    let mut n = 1;
    if b != a {
        out[n] = b;
        n += 1;
    }
    if c != a && c != b {
        out[n] = c;
        n += 1;
    }
    (out, n)
}

#[inline]
fn valence_score(active_tris: u32, lut: &[f32; 64]) -> f32 {
    if active_tris == 0 {
        0.0
    } else if (active_tris as usize) < lut.len() {
        lut[active_tris as usize]
    } else {
        2.0 / (active_tris as f32).sqrt()
    }
}

/// Deduplicates identical vertices, reorders triangles for a 32-entry GPU post-transform vertex
/// cache, and remaps vertex buffers into sequential first-fetch order.
pub fn optimize_mesh(mesh: &mut Mesh) {
    let v_count = mesh.positions.len();
    let has_normals = !mesh.normals.is_empty();
    let has_uvs = !mesh.uvs.is_empty();
    if (has_normals && mesh.normals.len() != v_count)
        || (has_uvs && mesh.uvs.len() != v_count)
        || mesh.indices.len() % 3 != 0
        || mesh.indices.iter().any(|&i| (i as usize) >= v_count)
    {
        return;
    }
    if mesh.indices.is_empty() {
        mesh.positions.clear();
        mesh.normals.clear();
        mesh.uvs.clear();
        return;
    }

    // 1. Exact bit-key vertex deduplication + unused-vertex stripping in deterministic index order.
    let mut remap_in = vec![u32::MAX; v_count];
    let mut dedup_map: HashMap<[u32; 8], u32> = HashMap::with_capacity(v_count);
    let mut dedup_pos: Vec<Vec3> = Vec::with_capacity(v_count);
    let mut dedup_norm: Vec<Vec3> = Vec::with_capacity(if has_normals { v_count } else { 0 });
    let mut dedup_uv: Vec<[f32; 2]> = Vec::with_capacity(if has_uvs { v_count } else { 0 });

    for idx in &mut mesh.indices {
        let old = *idx as usize;
        let mut mapped = remap_in[old];
        if mapped == u32::MAX {
            let p = mesh.positions[old];
            let n = if has_normals {
                mesh.normals[old]
            } else {
                Vec3(0.0, 0.0, 0.0)
            };
            let uv = if has_uvs { mesh.uvs[old] } else { [0.0, 0.0] };
            let key = [
                canon_bits(p.0),
                canon_bits(p.1),
                canon_bits(p.2),
                canon_bits(n.0),
                canon_bits(n.1),
                canon_bits(n.2),
                canon_bits(uv[0]),
                canon_bits(uv[1]),
            ];
            mapped = *dedup_map.entry(key).or_insert_with(|| {
                let id = dedup_pos.len() as u32;
                dedup_pos.push(p);
                if has_normals {
                    dedup_norm.push(n);
                }
                if has_uvs {
                    dedup_uv.push(uv);
                }
                id
            });
            remap_in[old] = mapped;
        }
        *idx = mapped;
    }
    drop(dedup_map);
    drop(remap_in);

    let num_verts = dedup_pos.len();
    let num_tris = mesh.indices.len() / 3;

    // 2. Linear-time Tom Forsyth post-transform vertex cache triangle reordering.
    let ordered_indices = if num_tris > 1 {
        let mut cache_score = [0.0f32; CACHE_SIZE];
        let mut valence_lut = [0.0f32; 64];
        for (pos, slot) in cache_score.iter_mut().enumerate() {
            *slot = if pos < 3 {
                0.75
            } else {
                let t = (CACHE_SIZE - pos) as f32 / (CACHE_SIZE - 3) as f32;
                t * t.sqrt()
            };
        }
        for (val, slot) in valence_lut.iter_mut().enumerate().skip(1) {
            *slot = 2.0 / (val as f32).sqrt();
        }

        let mut active_tris = vec![0u32; num_verts];
        for tri in mesh.indices.chunks_exact(3) {
            let (uv, uc) = unique_tri_verts(tri[0], tri[1], tri[2]);
            for &v in &uv[..uc] {
                active_tris[v as usize] += 1;
            }
        }

        let mut offsets = Vec::with_capacity(num_verts + 1);
        let mut total_adj = 0u32;
        for &deg in &active_tris {
            offsets.push(total_adj);
            total_adj += deg;
        }
        offsets.push(total_adj);

        let mut adj = vec![0u32; total_adj as usize];
        let mut write_pos = offsets[..num_verts].to_vec();
        for (t, tri) in mesh.indices.chunks_exact(3).enumerate() {
            let (uv, uc) = unique_tri_verts(tri[0], tri[1], tri[2]);
            for &v in &uv[..uc] {
                let slot = &mut write_pos[v as usize];
                adj[*slot as usize] = t as u32;
                *slot += 1;
            }
        }
        // Reuse `write_pos` as a lazy unemitted cursor for super-hub vertices (`deg > MAX_HUB_DEGREE`).
        let mut hub_cursor = write_pos;
        hub_cursor.copy_from_slice(&offsets[..num_verts]);

        let mut vert_score = Vec::with_capacity(num_verts);
        for &deg in &active_tris {
            vert_score.push(valence_score(deg, &valence_lut));
        }

        let mut tri_score = Vec::with_capacity(num_tris);
        let mut best_tri = 0u32;
        let mut best_score = -1.0f32;
        for (t, tri) in mesh.indices.chunks_exact(3).enumerate() {
            let (uv, uc) = unique_tri_verts(tri[0], tri[1], tri[2]);
            let mut s = 0.0f32;
            for &v in &uv[..uc] {
                s += vert_score[v as usize];
            }
            tri_score.push(s);
            if s > best_score {
                best_score = s;
                best_tri = t as u32;
            }
        }

        let mut emitted = vec![false; num_tris];
        let mut out_indices = Vec::with_capacity(mesh.indices.len());
        let mut cache = [u32::MAX; CACHE_SIZE];
        let mut cache_len = 0usize;
        let mut scan_cursor = 0usize;

        for _ in 0..num_tris {
            if best_tri == u32::MAX {
                while emitted[scan_cursor] {
                    scan_cursor += 1;
                }
                best_tri = scan_cursor as u32;
            }

            let t = best_tri as usize;
            emitted[t] = true;
            let i0 = mesh.indices[t * 3];
            let i1 = mesh.indices[t * 3 + 1];
            let i2 = mesh.indices[t * 3 + 2];
            out_indices.push(i0);
            out_indices.push(i1);
            out_indices.push(i2);

            let (uverts, ucount) = unique_tri_verts(i0, i1, i2);
            for &v in &uverts[..ucount] {
                active_tris[v as usize] -= 1;
            }

            let mut next_cache = [u32::MAX; CACHE_SIZE + 3];
            next_cache[..ucount].copy_from_slice(&uverts[..ucount]);
            let mut next_len = ucount;
            for &cv in &cache[..cache_len] {
                if !uverts[..ucount].contains(&cv) {
                    next_cache[next_len] = cv;
                    next_len += 1;
                }
            }

            for (pos, &cv) in next_cache[..next_len].iter().enumerate() {
                let v = cv as usize;
                let rem = active_tris[v];
                let new_s = if rem == 0 {
                    0.0
                } else {
                    let c_score = if pos < CACHE_SIZE {
                        cache_score[pos]
                    } else {
                        0.0
                    };
                    c_score + valence_score(rem, &valence_lut)
                };
                let delta = new_s - vert_score[v];
                vert_score[v] = new_s;
                let start = offsets[v];
                let end = offsets[v + 1];
                if delta != 0.0 && end - start <= MAX_HUB_DEGREE {
                    for &adj_t in &adj[start as usize..end as usize] {
                        let ti = adj_t as usize;
                        if !emitted[ti] {
                            tri_score[ti] += delta;
                        }
                    }
                }
            }

            cache_len = next_len.min(CACHE_SIZE);
            cache[..cache_len].copy_from_slice(&next_cache[..cache_len]);

            best_tri = u32::MAX;
            let mut best_s = -1.0f32;
            for &cv in &cache[..cache_len] {
                let v = cv as usize;
                if active_tris[v] == 0 {
                    continue;
                }
                let start = offsets[v];
                let end = offsets[v + 1];
                if end - start <= MAX_HUB_DEGREE {
                    for &adj_t in &adj[start as usize..end as usize] {
                        let ti = adj_t as usize;
                        if !emitted[ti] && tri_score[ti] > best_s {
                            best_s = tri_score[ti];
                            best_tri = adj_t;
                        }
                    }
                } else {
                    let mut cur = hub_cursor[v];
                    while cur < end && emitted[adj[cur as usize] as usize] {
                        cur += 1;
                    }
                    hub_cursor[v] = cur;
                    let probe_end = end.min(cur + 16);
                    for &adj_t in &adj[cur as usize..probe_end as usize] {
                        let ti = adj_t as usize;
                        if !emitted[ti] && tri_score[ti] > best_s {
                            best_s = tri_score[ti];
                            best_tri = adj_t;
                        }
                    }
                }
            }
        }
        out_indices
    } else {
        std::mem::take(&mut mesh.indices)
    };

    // 3. Sequential pre-transform vertex fetch remapping.
    let mut fetch_remap = vec![u32::MAX; num_verts];
    let mut final_pos = Vec::with_capacity(num_verts);
    let mut final_norm = Vec::with_capacity(if has_normals { num_verts } else { 0 });
    let mut final_uv = Vec::with_capacity(if has_uvs { num_verts } else { 0 });
    let mut final_indices = ordered_indices;

    for idx in &mut final_indices {
        let old = *idx as usize;
        let mut new_i = fetch_remap[old];
        if new_i == u32::MAX {
            new_i = final_pos.len() as u32;
            fetch_remap[old] = new_i;
            final_pos.push(dedup_pos[old]);
            if has_normals {
                final_norm.push(dedup_norm[old]);
            }
            if has_uvs {
                final_uv.push(dedup_uv[old]);
            }
        }
        *idx = new_i;
    }

    mesh.positions = final_pos;
    mesh.normals = final_norm;
    mesh.uvs = final_uv;
    mesh.indices = final_indices;
}

fn cluster_mesh_grid(mesh: &Mesh, lo: Vec3, cell_size: f32) -> Mesh {
    let v_count = mesh.positions.len();
    let has_normals = mesh.normals.len() == v_count;
    let has_uvs = mesh.uvs.len() == v_count;
    let inv = if cell_size > 1e-9 {
        1.0 / cell_size
    } else {
        1.0
    };

    let mut cell_of: HashMap<[i32; 3], u32> = HashMap::new();
    let mut remap = Vec::with_capacity(v_count);
    let mut sum_pos: Vec<[f64; 3]> = Vec::new();
    let mut sum_norm: Vec<[f64; 3]> = Vec::new();
    let mut sum_uv: Vec<[f64; 2]> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();

    for i in 0..v_count {
        let p = mesh.positions[i];
        let key = [
            ((p.0 - lo.0) * inv).floor() as i32,
            ((p.1 - lo.1) * inv).floor() as i32,
            ((p.2 - lo.2) * inv).floor() as i32,
        ];
        let cid = *cell_of.entry(key).or_insert_with(|| {
            let id = counts.len() as u32;
            sum_pos.push([0.0; 3]);
            if has_normals {
                sum_norm.push([0.0; 3]);
            }
            if has_uvs {
                sum_uv.push([0.0; 2]);
            }
            counts.push(0);
            id
        });
        let idx = cid as usize;
        sum_pos[idx][0] += f64::from(p.0);
        sum_pos[idx][1] += f64::from(p.1);
        sum_pos[idx][2] += f64::from(p.2);
        if has_normals {
            let n = mesh.normals[i];
            sum_norm[idx][0] += f64::from(n.0);
            sum_norm[idx][1] += f64::from(n.1);
            sum_norm[idx][2] += f64::from(n.2);
        }
        if has_uvs {
            let uv = mesh.uvs[i];
            sum_uv[idx][0] += f64::from(uv[0]);
            sum_uv[idx][1] += f64::from(uv[1]);
        }
        counts[idx] += 1;
        remap.push(cid);
    }

    let c_len = counts.len();
    let mut positions = Vec::with_capacity(c_len);
    let mut normals = Vec::with_capacity(if has_normals { c_len } else { 0 });
    let mut uvs = Vec::with_capacity(if has_uvs { c_len } else { 0 });
    for i in 0..c_len {
        let inv_n = 1.0 / f64::from(counts[i].max(1));
        positions.push(Vec3(
            (sum_pos[i][0] * inv_n) as f32,
            (sum_pos[i][1] * inv_n) as f32,
            (sum_pos[i][2] * inv_n) as f32,
        ));
        if has_normals {
            let (nx, ny, nz) = (sum_norm[i][0], sum_norm[i][1], sum_norm[i][2]);
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            if len > 1e-9 {
                normals.push(Vec3(
                    (nx / len) as f32,
                    (ny / len) as f32,
                    (nz / len) as f32,
                ));
            } else {
                normals.push(Vec3(0.0, 1.0, 0.0));
            }
        }
        if has_uvs {
            uvs.push([(sum_uv[i][0] * inv_n) as f32, (sum_uv[i][1] * inv_n) as f32]);
        }
    }

    let mut indices = Vec::new();
    let mut seen_tris: HashMap<[u32; 3], ()> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        let a = remap[tri[0] as usize];
        let b = remap[tri[1] as usize];
        let c = remap[tri[2] as usize];
        if a == b || b == c || a == c {
            continue;
        }
        let pa = positions[a as usize];
        let pb = positions[b as usize];
        let pc = positions[c as usize];
        let ux = pb.0 - pa.0;
        let uy = pb.1 - pa.1;
        let uz = pb.2 - pa.2;
        let vx = pc.0 - pa.0;
        let vy = pc.1 - pa.1;
        let vz = pc.2 - pa.2;
        let cx = uy * vz - uz * vy;
        let cy = uz * vx - ux * vz;
        let cz = ux * vy - uy * vx;
        if (cx * cx + cy * cy + cz * cz) <= 1e-18 {
            continue;
        }
        // Canonicalize cycle order so duplicate collapsed triangles are emitted once.
        let key = if a < b && a < c {
            [a, b, c]
        } else if b < a && b < c {
            [b, c, a]
        } else {
            [c, a, b]
        };
        if seen_tris.insert(key, ()).is_none() {
            indices.push(a);
            indices.push(b);
            indices.push(c);
        }
    }

    Mesh {
        positions,
        normals,
        uvs,
        indices,
        color: mesh.color,
        material: mesh.material,
    }
}

/// Simplifies `mesh` toward `target_ratio` (`0.01..=1.0`) of its original triangle count using
/// adaptive spatial vertex clustering followed by [`optimize_mesh`].
pub fn decimate_mesh(mesh: &Mesh, target_ratio: f32) -> Mesh {
    let mut base = mesh.clone();
    optimize_mesh(&mut base);
    let orig_tris = base.indices.len() / 3;
    if !target_ratio.is_finite() || target_ratio >= 0.999 || orig_tris <= 4 {
        return base;
    }
    let ratio = target_ratio.clamp(0.01, 1.0);
    let target_tris = ((orig_tris as f32) * ratio).round().max(1.0) as usize;

    let mut lo = Vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut hi = Vec3(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &p in &base.positions {
        lo.0 = lo.0.min(p.0);
        lo.1 = lo.1.min(p.1);
        lo.2 = lo.2.min(p.2);
        hi.0 = hi.0.max(p.0);
        hi.1 = hi.1.max(p.1);
        hi.2 = hi.2.max(p.2);
    }
    let span = (hi.0 - lo.0).max(hi.1 - lo.1).max(hi.2 - lo.2);
    if !span.is_finite() || span <= 1e-6 {
        return base;
    }

    let mut low_res = 2u32;
    let mut high_res = 256u32;
    let mut best: Option<(usize, Mesh)> = None;
    while low_res <= high_res {
        let mid = (low_res + high_res) / 2;
        let candidate = cluster_mesh_grid(&base, lo, span / (mid as f32));
        let tris = candidate.indices.len() / 3;
        if tris > 0 {
            let diff = tris.abs_diff(target_tris);
            if best.as_ref().is_none_or(|(b_diff, _)| diff < *b_diff) {
                best = Some((diff, candidate));
            }
        }
        if tris < target_tris {
            low_res = mid + 1;
        } else if tris > target_tris {
            if mid == 0 {
                break;
            }
            high_res = mid - 1;
        } else {
            break;
        }
    }

    match best {
        Some((_, mut simplified)) if simplified.indices.len() < base.indices.len() => {
            optimize_mesh(&mut simplified);
            simplified
        }
        _ => base,
    }
}

/// Extracts up to `max_points` deterministic extreme support vertices across sphere directions for
/// convex-hull physics collider fitting.
pub fn extreme_hull_points(positions: &[Vec3], max_points: usize) -> Vec<[f32; 3]> {
    if positions.is_empty() || max_points == 0 {
        return Vec::new();
    }
    let cap = max_points.clamp(4, 64);
    let mut dirs: Vec<[f32; 3]> = Vec::with_capacity(cap);
    let presets: &[[f32; 3]] = &[
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, 1.0],
        [0.0, -1.0, -1.0],
    ];
    for d in presets.iter().take(cap) {
        dirs.push(*d);
    }
    if cap > dirs.len() {
        let extra = cap - dirs.len();
        let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        for i in 0..extra {
            let y = 1.0 - ((i as f32 + 0.5) / extra as f32) * 2.0;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden * (i as f32);
            dirs.push([theta.cos() * r, y, theta.sin() * r]);
        }
    }

    let mut seen: HashMap<[u32; 3], ()> = HashMap::new();
    let mut out = Vec::new();
    for d in dirs {
        let mut best: Option<Vec3> = None;
        let mut best_dot = f32::NEG_INFINITY;
        for &p in positions {
            if !p.0.is_finite() || !p.1.is_finite() || !p.2.is_finite() {
                continue;
            }
            let dot = p.0 * d[0] + p.1 * d[1] + p.2 * d[2];
            if dot > best_dot {
                best_dot = dot;
                best = Some(p);
            }
        }
        if let Some(p) = best {
            let key = [canon_bits(p.0), canon_bits(p.1), canon_bits(p.2)];
            if seen.insert(key, ()).is_none() {
                out.push([p.0, p.1, p.2]);
                if out.len() >= cap {
                    break;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acmr_fifo32(indices: &[u32]) -> f32 {
        let mut cache = [u32::MAX; 32];
        let mut write = 0usize;
        let mut misses = 0usize;
        for &idx in indices {
            if !cache.contains(&idx) {
                misses += 1;
                cache[write] = idx;
                write = (write + 1) % 32;
            }
        }
        misses as f32 / (indices.len() / 3).max(1) as f32
    }

    #[test]
    fn deduplicates_improves_acmr_and_orders_vertex_fetch() {
        // 32x32 unindexed soup of quads (2048 triangles, 6144 duplicate vertices -> 1089 unique),
        // laid out column-scrambled to defeat naive scanline order.
        let side = 32usize;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        for step in 0..(side * side) {
            // Scramble traversal order.
            let cell = (step * 293) % (side * side);
            let (gx, gy) = (cell % side, cell / side);
            let quad = [
                (gx, gy),
                (gx + 1, gy),
                (gx + 1, gy + 1),
                (gx, gy),
                (gx + 1, gy + 1),
                (gx, gy + 1),
            ];
            for (x, y) in quad {
                let id = positions.len() as u32;
                positions.push(Vec3(x as f32, 0.0, y as f32));
                normals.push(Vec3(0.0, 1.0, 0.0));
                uvs.push([x as f32 / side as f32, y as f32 / side as f32]);
                indices.push(id);
            }
        }
        let mut mesh = Mesh {
            positions,
            normals,
            uvs,
            indices,
            color: [1.0, 1.0, 1.0],
            material: None,
        };
        let before_acmr = acmr_fifo32(&mesh.indices);
        assert!(before_acmr > 2.5, "soup ACMR: {before_acmr}");

        optimize_mesh(&mut mesh);

        assert_eq!(mesh.positions.len(), (side + 1) * (side + 1));
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        assert_eq!(mesh.uvs.len(), mesh.positions.len());
        assert_eq!(mesh.indices.len(), side * side * 6);

        let after_acmr = acmr_fifo32(&mesh.indices);
        assert!(
            after_acmr < 0.75,
            "expected optimized ACMR < 0.75, got {after_acmr} (was {before_acmr})"
        );

        // Verify vertices are strictly in sequential first-use order.
        let mut max_seen = 0u32;
        for &idx in &mesh.indices {
            assert!(idx <= max_seen);
            if idx == max_seen {
                max_seen += 1;
            }
        }
        assert_eq!(max_seen as usize, mesh.positions.len());
    }

    #[test]
    fn preserves_hard_edges_and_uv_seams_while_handling_degenerates() {
        // Two triangles sharing position (1,0,0) with different normals must NOT merge that vertex,
        // plus one degenerate triangle and an unreferenced vertex.
        let mut mesh = Mesh {
            positions: vec![
                Vec3(99.0, 99.0, 99.0), // unused
                Vec3(0.0, 0.0, 0.0),
                Vec3(1.0, 0.0, -0.0), // -0.0 normalizes to +0.0
                Vec3(0.0, 1.0, 0.0),
                Vec3(1.0, 0.0, 0.0), // duplicate of index 2 with same normal
                Vec3(1.0, 0.0, 0.0), // hard edge: different normal!
                Vec3(2.0, 0.0, 0.0),
                Vec3(1.0, 1.0, 0.0),
            ],
            normals: vec![
                Vec3(0.0, 1.0, 0.0),
                Vec3(0.0, 0.0, 1.0),
                Vec3(0.0, 0.0, 1.0),
                Vec3(0.0, 0.0, 1.0),
                Vec3(0.0, 0.0, 1.0),
                Vec3(1.0, 0.0, 0.0),
                Vec3(1.0, 0.0, 0.0),
                Vec3(1.0, 0.0, 0.0),
            ],
            uvs: vec![],
            indices: vec![1, 2, 3, 4, 4, 3, 5, 6, 7],
            color: [0.5, 0.5, 0.5],
            material: Some(0),
        };
        optimize_mesh(&mut mesh);
        // Unused vertex 0 stripped, vertex 4 merged with 2, vertex 5 kept distinct -> 6 vertices.
        assert_eq!(mesh.positions.len(), 6);
        assert_eq!(mesh.normals.len(), 6);
        assert_eq!(mesh.indices.len(), 9);
    }

    #[test]
    fn decimate_mesh_reduces_lod_triangles_monotonically_and_extracts_hull_points() {
        let side = 24usize;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        for gz in 0..side {
            for gx in 0..side {
                let quad = [
                    (gx, gz),
                    (gx + 1, gz),
                    (gx + 1, gz + 1),
                    (gx, gz),
                    (gx + 1, gz + 1),
                    (gx, gz + 1),
                ];
                for (x, z) in quad {
                    let id = positions.len() as u32;
                    let fx = x as f32 / side as f32;
                    let fz = z as f32 / side as f32;
                    let fy = (fx * std::f32::consts::PI).sin() * 0.4;
                    positions.push(Vec3(fx * 4.0, fy, fz * 4.0));
                    normals.push(Vec3(0.0, 1.0, 0.0));
                    uvs.push([fx, fz]);
                    indices.push(id);
                }
            }
        }
        let lod0 = Mesh {
            positions,
            normals,
            uvs,
            indices,
            color: [0.8, 0.8, 0.8],
            material: Some(0),
        };
        let t0 = lod0.indices.len() / 3;
        let lod1 = decimate_mesh(&lod0, 0.5);
        let lod2 = decimate_mesh(&lod0, 0.25);
        let lod3 = decimate_mesh(&lod0, 0.125);
        let (t1, t2, t3) = (
            lod1.indices.len() / 3,
            lod2.indices.len() / 3,
            lod3.indices.len() / 3,
        );
        assert!(
            t1 < t0 && t2 < t1 && t3 < t2 && t3 > 0,
            "{t0} -> {t1} -> {t2} -> {t3}"
        );
        assert_eq!(lod2.normals.len(), lod2.positions.len());
        assert_eq!(lod2.uvs.len(), lod2.positions.len());

        let hull = extreme_hull_points(&lod0.positions, 24);
        assert!(
            hull.len() >= 4 && hull.len() <= 24,
            "hull len: {}",
            hull.len()
        );
    }
}
