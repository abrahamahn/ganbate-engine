//! Deterministic pure-Rust block texture codecs (`BC1`, `BC7` Mode 6, and `ASTC 4x4` LDR).
//!
//! Every codec operates on 4x4 RGBA8 blocks (`[[u8; 4]; 16]` in row-major order) and handles
//! arbitrary non-multiple-of-4 mip extents by clamping border texels during encoding and cropping
//! out-of-bounds texels during decoding.

use crate::mip::{self, Filter};
use ganbate_core::world::{MAX_TEXTURE_SIZE, Texture, TextureCompression};

const BC7_WEIGHT4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
const ASTC_WEIGHT2: [u32; 4] = [0, 21, 43, 64];
const ASTC_WEIGHT3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];

fn rgb565_from_rgb8(r: u8, g: u8, b: u8) -> u16 {
    let r5 = ((u16::from(r) * 31 + 127) / 255) & 0x1F;
    let g6 = ((u16::from(g) * 63 + 127) / 255) & 0x3F;
    let b5 = ((u16::from(b) * 31 + 127) / 255) & 0x1F;
    (r5 << 11) | (g6 << 5) | b5
}

fn rgb565_to_rgb8(c: u16) -> [u8; 3] {
    let r5 = ((c >> 11) & 0x1F) as u8;
    let g6 = ((c >> 5) & 0x3F) as u8;
    let b5 = (c & 0x1F) as u8;
    [
        (r5 << 3) | (r5 >> 2),
        (g6 << 2) | (g6 >> 4),
        (b5 << 3) | (b5 >> 2),
    ]
}

fn bc1_palette(c0: u16, c1: u16) -> [[u8; 4]; 4] {
    let [r0, g0, b0] = rgb565_to_rgb8(c0);
    let [r1, g1, b1] = rgb565_to_rgb8(c1);
    let p0 = [r0, g0, b0, 255];
    let p1 = [r1, g1, b1, 255];
    if c0 > c1 {
        let p2 = [
            ((2 * u16::from(r0) + u16::from(r1) + 1) / 3) as u8,
            ((2 * u16::from(g0) + u16::from(g1) + 1) / 3) as u8,
            ((2 * u16::from(b0) + u16::from(b1) + 1) / 3) as u8,
            255,
        ];
        let p3 = [
            ((u16::from(r0) + 2 * u16::from(r1) + 1) / 3) as u8,
            ((u16::from(g0) + 2 * u16::from(g1) + 1) / 3) as u8,
            ((u16::from(b0) + 2 * u16::from(b1) + 1) / 3) as u8,
            255,
        ];
        [p0, p1, p2, p3]
    } else {
        let p2 = [
            ((u16::from(r0) + u16::from(r1)) / 2) as u8,
            ((u16::from(g0) + u16::from(g1)) / 2) as u8,
            ((u16::from(b0) + u16::from(b1)) / 2) as u8,
            255,
        ];
        [p0, p1, p2, [0, 0, 0, 0]]
    }
}

fn eval_bc1_candidate(
    texels: &[[u8; 4]; 16],
    mut c0: u16,
    mut c1: u16,
    punchthrough: bool,
) -> (u32, [u8; 8]) {
    if punchthrough {
        if c0 > c1 {
            std::mem::swap(&mut c0, &mut c1);
        }
        let pal = bc1_palette(c0, c1);
        let mut indices = 0u32;
        let mut err = 0u32;
        for (i, t) in texels.iter().enumerate() {
            if t[3] < 128 {
                indices |= 3 << (2 * i);
                err += u32::from(t[3]) * u32::from(t[3]);
            } else {
                let mut best_k = 0u32;
                let mut best_d = u32::MAX;
                for (k, p) in pal[..3].iter().enumerate() {
                    let dr = i32::from(t[0]) - i32::from(p[0]);
                    let dg = i32::from(t[1]) - i32::from(p[1]);
                    let db = i32::from(t[2]) - i32::from(p[2]);
                    let d = (dr * dr + dg * dg + db * db) as u32;
                    if d < best_d {
                        best_d = d;
                        best_k = k as u32;
                    }
                }
                indices |= best_k << (2 * i);
                let da = 255 - u32::from(t[3]);
                err += best_d + da * da;
            }
        }
        let [b0, b1] = c0.to_le_bytes();
        let [b2, b3] = c1.to_le_bytes();
        let [b4, b5, b6, b7] = indices.to_le_bytes();
        (err, [b0, b1, b2, b3, b4, b5, b6, b7])
    } else {
        if c0 < c1 {
            std::mem::swap(&mut c0, &mut c1);
        }
        let pal = bc1_palette(c0, c1);
        let count = if c0 > c1 { 4 } else { 3 };
        let mut indices = 0u32;
        let mut err = 0u32;
        for (i, t) in texels.iter().enumerate() {
            let mut best_k = 0u32;
            let mut best_d = u32::MAX;
            for (k, p) in pal[..count].iter().enumerate() {
                let dr = i32::from(t[0]) - i32::from(p[0]);
                let dg = i32::from(t[1]) - i32::from(p[1]);
                let db = i32::from(t[2]) - i32::from(p[2]);
                let d = (dr * dr + dg * dg + db * db) as u32;
                if d < best_d {
                    best_d = d;
                    best_k = k as u32;
                }
            }
            indices |= best_k << (2 * i);
            err += best_d;
        }
        let [b0, b1] = c0.to_le_bytes();
        let [b2, b3] = c1.to_le_bytes();
        let [b4, b5, b6, b7] = indices.to_le_bytes();
        (err, [b0, b1, b2, b3, b4, b5, b6, b7])
    }
}

/// Encodes a 4x4 RGBA8 block into an 8-byte `BC1` (`DXT1`) block using PCA endpoint fitting,
/// bounding-box fallback, and 1-pass least-squares refinement.
pub fn encode_bc1_block(texels: &[[u8; 4]; 16]) -> [u8; 8] {
    let punchthrough = texels.iter().any(|t| t[3] < 128);
    let mut active = [[0f32; 3]; 16];
    let mut active_n = 0usize;
    let mut min_c = [255f32; 3];
    let mut max_c = [0f32; 3];
    let mut mean = [0f32; 3];
    for t in texels {
        if punchthrough && t[3] < 128 {
            continue;
        }
        let rgb = [f32::from(t[0]), f32::from(t[1]), f32::from(t[2])];
        for c in 0..3 {
            min_c[c] = min_c[c].min(rgb[c]);
            max_c[c] = max_c[c].max(rgb[c]);
            mean[c] += rgb[c];
        }
        active[active_n] = rgb;
        active_n += 1;
    }
    if active_n == 0 {
        return [0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF];
    }
    let inv_n = 1.0 / active_n as f32;
    for m in &mut mean {
        *m *= inv_n;
    }

    // Bounding-box candidate (with slight inset).
    let span = [
        max_c[0] - min_c[0],
        max_c[1] - min_c[1],
        max_c[2] - min_c[2],
    ];
    let inset = [
        span[0] * (1.0 / 16.0),
        span[1] * (1.0 / 16.0),
        span[2] * (1.0 / 16.0),
    ];
    let bb0 = rgb565_from_rgb8(
        (min_c[0] + inset[0]).clamp(0.0, 255.0).round() as u8,
        (min_c[1] + inset[1]).clamp(0.0, 255.0).round() as u8,
        (min_c[2] + inset[2]).clamp(0.0, 255.0).round() as u8,
    );
    let bb1 = rgb565_from_rgb8(
        (max_c[0] - inset[0]).clamp(0.0, 255.0).round() as u8,
        (max_c[1] - inset[1]).clamp(0.0, 255.0).round() as u8,
        (max_c[2] - inset[2]).clamp(0.0, 255.0).round() as u8,
    );
    let mut best = eval_bc1_candidate(texels, bb0, bb1, punchthrough);
    let raw0 = rgb565_from_rgb8(min_c[0] as u8, min_c[1] as u8, min_c[2] as u8);
    let raw1 = rgb565_from_rgb8(max_c[0] as u8, max_c[1] as u8, max_c[2] as u8);
    let cand_raw = eval_bc1_candidate(texels, raw0, raw1, punchthrough);
    if cand_raw.0 < best.0 {
        best = cand_raw;
    }

    // 3x3 covariance + 6 deterministic power iterations for the principal RGB axis.
    let mut cov = [0f32; 6]; // xx, xy, xz, yy, yz, zz
    for rgb in &active[..active_n] {
        let dx = rgb[0] - mean[0];
        let dy = rgb[1] - mean[1];
        let dz = rgb[2] - mean[2];
        cov[0] += dx * dx;
        cov[1] += dx * dy;
        cov[2] += dx * dz;
        cov[3] += dy * dy;
        cov[4] += dy * dz;
        cov[5] += dz * dz;
    }
    let mut axis = span;
    let axis_norm2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    if axis_norm2 > 1e-4 {
        for _ in 0..6 {
            let nx = cov[0] * axis[0] + cov[1] * axis[1] + cov[2] * axis[2];
            let ny = cov[1] * axis[0] + cov[3] * axis[1] + cov[4] * axis[2];
            let nz = cov[2] * axis[0] + cov[4] * axis[1] + cov[5] * axis[2];
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            if len <= 1e-6 {
                break;
            }
            axis = [nx / len, ny / len, nz / len];
        }
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if len > 1e-6 {
            axis = [axis[0] / len, axis[1] / len, axis[2] / len];
            let mut min_t = f32::INFINITY;
            let mut max_t = f32::NEG_INFINITY;
            for rgb in &active[..active_n] {
                let t = (rgb[0] - mean[0]) * axis[0]
                    + (rgb[1] - mean[1]) * axis[1]
                    + (rgb[2] - mean[2]) * axis[2];
                min_t = min_t.min(t);
                max_t = max_t.max(t);
            }
            let e0 = [
                (mean[0] + axis[0] * min_t).clamp(0.0, 255.0),
                (mean[1] + axis[1] * min_t).clamp(0.0, 255.0),
                (mean[2] + axis[2] * min_t).clamp(0.0, 255.0),
            ];
            let e1 = [
                (mean[0] + axis[0] * max_t).clamp(0.0, 255.0),
                (mean[1] + axis[1] * max_t).clamp(0.0, 255.0),
                (mean[2] + axis[2] * max_t).clamp(0.0, 255.0),
            ];
            let p0 = rgb565_from_rgb8(
                e0[0].round() as u8,
                e0[1].round() as u8,
                e0[2].round() as u8,
            );
            let p1 = rgb565_from_rgb8(
                e1[0].round() as u8,
                e1[1].round() as u8,
                e1[2].round() as u8,
            );
            let cand_pca = eval_bc1_candidate(texels, p0, p1, punchthrough);
            if cand_pca.0 < best.0 {
                best = cand_pca;
            }

            // 1-pass least-squares endpoint refit from the best opaque 4-level assignment.
            if !punchthrough {
                let c0 = u16::from_le_bytes([best.1[0], best.1[1]]);
                let c1 = u16::from_le_bytes([best.1[2], best.1[3]]);
                let idx_bits = u32::from_le_bytes([best.1[4], best.1[5], best.1[6], best.1[7]]);
                if c0 != c1 {
                    const W: [f32; 4] = [0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0];
                    let (mut a00, mut a01, mut a11) = (0f32, 0f32, 0f32);
                    let mut b0 = [0f32; 3];
                    let mut b1 = [0f32; 3];
                    for (i, t) in texels.iter().enumerate() {
                        let w = W[((idx_bits >> (2 * i)) & 3) as usize];
                        let omw = 1.0 - w;
                        a00 += omw * omw;
                        a01 += omw * w;
                        a11 += w * w;
                        for c in 0..3 {
                            let v = f32::from(t[c]);
                            b0[c] += omw * v;
                            b1[c] += w * v;
                        }
                    }
                    let det = a00 * a11 - a01 * a01;
                    if det > 1e-4 {
                        let inv_det = 1.0 / det;
                        let mut r0 = [0u8; 3];
                        let mut r1 = [0u8; 3];
                        for c in 0..3 {
                            r0[c] = ((a11 * b0[c] - a01 * b1[c]) * inv_det)
                                .clamp(0.0, 255.0)
                                .round() as u8;
                            r1[c] = ((a00 * b1[c] - a01 * b0[c]) * inv_det)
                                .clamp(0.0, 255.0)
                                .round() as u8;
                        }
                        let q0 = rgb565_from_rgb8(r0[0], r0[1], r0[2]);
                        let q1 = rgb565_from_rgb8(r1[0], r1[1], r1[2]);
                        let cand_ls = eval_bc1_candidate(texels, q0, q1, false);
                        if cand_ls.0 < best.0 {
                            best = cand_ls;
                        }
                    }
                }
            }
        }
    }

    best.1
}

/// Decodes an 8-byte `BC1` (`DXT1`) block into 16 RGBA8 texels.
pub fn decode_bc1_block(block: &[u8; 8]) -> [[u8; 4]; 16] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let indices = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let pal = bc1_palette(c0, c1);
    let mut out = [[0u8; 4]; 16];
    for (i, dst) in out.iter_mut().enumerate() {
        *dst = pal[((indices >> (2 * i)) & 3) as usize];
    }
    out
}

fn bc7_interp(e0: u8, e1: u8, w: u32) -> u8 {
    (((64 - w) * u32::from(e0) + w * u32::from(e1) + 32) >> 6) as u8
}

fn quantize_with_pbit(v: f32, pbit: u8) -> u8 {
    let p = i32::from(pbit & 1);
    let vi = v.clamp(0.0, 255.0).round() as i32;
    let c7 = ((vi - p + 1) >> 1).clamp(0, 127) as u8;
    (c7 << 1) | (pbit & 1)
}

fn eval_bc7_mode6_pbits(
    texels: &[[u8; 4]; 16],
    f0: [f32; 4],
    f1: [f32; 4],
    p0: u8,
    p1: u8,
) -> (u32, [u8; 4], [u8; 4], [u8; 16]) {
    let e0 = [
        quantize_with_pbit(f0[0], p0),
        quantize_with_pbit(f0[1], p0),
        quantize_with_pbit(f0[2], p0),
        quantize_with_pbit(f0[3], p0),
    ];
    let e1 = [
        quantize_with_pbit(f1[0], p1),
        quantize_with_pbit(f1[1], p1),
        quantize_with_pbit(f1[2], p1),
        quantize_with_pbit(f1[3], p1),
    ];
    let mut pal = [[0u8; 4]; 16];
    for (k, &w) in BC7_WEIGHT4.iter().enumerate() {
        pal[k] = [
            bc7_interp(e0[0], e1[0], w),
            bc7_interp(e0[1], e1[1], w),
            bc7_interp(e0[2], e1[2], w),
            bc7_interp(e0[3], e1[3], w),
        ];
    }
    let mut indices = [0u8; 16];
    let mut total_err = 0u32;
    for (i, t) in texels.iter().enumerate() {
        let mut best_k = 0u8;
        let mut best_d = u32::MAX;
        for (k, p) in pal.iter().enumerate() {
            let dr = i32::from(t[0]) - i32::from(p[0]);
            let dg = i32::from(t[1]) - i32::from(p[1]);
            let db = i32::from(t[2]) - i32::from(p[2]);
            let da = i32::from(t[3]) - i32::from(p[3]);
            let d = (dr * dr + dg * dg + db * db + da * da) as u32;
            if d < best_d {
                best_d = d;
                best_k = k as u8;
                if d == 0 {
                    break;
                }
            }
        }
        indices[i] = best_k;
        total_err += best_d;
    }
    (total_err, e0, e1, indices)
}

fn pack_bc7_mode6(
    mut e0: [u8; 4],
    mut e1: [u8; 4],
    mut p0: u8,
    mut p1: u8,
    mut indices: [u8; 16],
) -> [u8; 16] {
    if indices[0] >= 8 {
        std::mem::swap(&mut e0, &mut e1);
        std::mem::swap(&mut p0, &mut p1);
        for idx in &mut indices {
            *idx = 15 - *idx;
        }
    }
    let mut bits: u128 = 1 << 6;
    bits |= u128::from(e0[0] >> 1) << 7;
    bits |= u128::from(e1[0] >> 1) << 14;
    bits |= u128::from(e0[1] >> 1) << 21;
    bits |= u128::from(e1[1] >> 1) << 28;
    bits |= u128::from(e0[2] >> 1) << 35;
    bits |= u128::from(e1[2] >> 1) << 42;
    bits |= u128::from(e0[3] >> 1) << 49;
    bits |= u128::from(e1[3] >> 1) << 56;
    bits |= u128::from(p0 & 1) << 63;
    bits |= u128::from(p1 & 1) << 64;
    bits |= u128::from(indices[0] & 0x7) << 65;
    for (i, &idx) in indices[1..].iter().enumerate() {
        bits |= u128::from(idx & 0xF) << (68 + i * 4);
    }
    bits.to_le_bytes()
}

/// Encodes a 4x4 RGBA8 block into a 16-byte `BC7` Mode 6 block (7-bit RGBA endpoints + 1-bit
/// unique endpoint P-bits + 4-bit per-texel weights).
pub fn encode_bc7_block(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    let first = texels[0];
    if texels[1..].iter().all(|&t| t == first) {
        // For a solid block, setting P0=0 and P1=1 lets odd/even channel LSBs interpolate within <= 1 LSB,
        // while testing all 4 P-bit combinations finds an exact 0-error match whenever all 4 channels share parity.
        let f = [
            f32::from(first[0]),
            f32::from(first[1]),
            f32::from(first[2]),
            f32::from(first[3]),
        ];
        let mut best = (u32::MAX, [0u8; 4], [0u8; 4], 0u8, 0u8, [0u8; 16]);
        for (p0, p1) in [(0, 0), (1, 1), (0, 1), (1, 0)] {
            let f0 = [
                f[0] - f32::from(p1),
                f[1] - f32::from(p1),
                f[2] - f32::from(p1),
                f[3] - f32::from(p1),
            ];
            let f1 = [
                f[0] + f32::from(p1),
                f[1] + f32::from(p1),
                f[2] + f32::from(p1),
                f[3] + f32::from(p1),
            ];
            let (err, e0, e1, indices) = eval_bc7_mode6_pbits(
                texels,
                if p0 == p1 { f } else { f0 },
                if p0 == p1 { f } else { f1 },
                p0,
                p1,
            );
            if err < best.0 {
                best = (err, e0, e1, p0, p1, indices);
                if err == 0 {
                    break;
                }
            }
        }
        return pack_bc7_mode6(best.1, best.2, best.3, best.4, best.5);
    }

    let mut min_c = [255f32; 4];
    let mut max_c = [0f32; 4];
    let mut mean = [0f32; 4];
    for t in texels {
        for c in 0..4 {
            let v = f32::from(t[c]);
            min_c[c] = min_c[c].min(v);
            max_c[c] = max_c[c].max(v);
            mean[c] += v;
        }
    }
    for m in &mut mean {
        *m *= 1.0 / 16.0;
    }

    // 4D covariance matrix + 6 deterministic power iterations.
    let mut cov = [[0f32; 4]; 4];
    for t in texels {
        let d = [
            f32::from(t[0]) - mean[0],
            f32::from(t[1]) - mean[1],
            f32::from(t[2]) - mean[2],
            f32::from(t[3]) - mean[3],
        ];
        for r in 0..4 {
            for c in 0..4 {
                cov[r][c] += d[r] * d[c];
            }
        }
    }
    let mut axis = [
        max_c[0] - min_c[0],
        max_c[1] - min_c[1],
        max_c[2] - min_c[2],
        max_c[3] - min_c[3],
    ];
    for _ in 0..6 {
        let mut next = [0f32; 4];
        for r in 0..4 {
            next[r] = cov[r][0] * axis[0]
                + cov[r][1] * axis[1]
                + cov[r][2] * axis[2]
                + cov[r][3] * axis[3];
        }
        let len =
            (next[0] * next[0] + next[1] * next[1] + next[2] * next[2] + next[3] * next[3]).sqrt();
        if len <= 1e-6 {
            break;
        }
        axis = [next[0] / len, next[1] / len, next[2] / len, next[3] / len];
    }

    let axis_len =
        (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2] + axis[3] * axis[3]).sqrt();
    let (mut f0, mut f1) = if axis_len > 1e-6 {
        let dir = [
            axis[0] / axis_len,
            axis[1] / axis_len,
            axis[2] / axis_len,
            axis[3] / axis_len,
        ];
        let mut min_t = f32::INFINITY;
        let mut max_t = f32::NEG_INFINITY;
        for t in texels {
            let proj = (f32::from(t[0]) - mean[0]) * dir[0]
                + (f32::from(t[1]) - mean[1]) * dir[1]
                + (f32::from(t[2]) - mean[2]) * dir[2]
                + (f32::from(t[3]) - mean[3]) * dir[3];
            min_t = min_t.min(proj);
            max_t = max_t.max(proj);
        }
        (
            [
                (mean[0] + dir[0] * min_t).clamp(0.0, 255.0),
                (mean[1] + dir[1] * min_t).clamp(0.0, 255.0),
                (mean[2] + dir[2] * min_t).clamp(0.0, 255.0),
                (mean[3] + dir[3] * min_t).clamp(0.0, 255.0),
            ],
            [
                (mean[0] + dir[0] * max_t).clamp(0.0, 255.0),
                (mean[1] + dir[1] * max_t).clamp(0.0, 255.0),
                (mean[2] + dir[2] * max_t).clamp(0.0, 255.0),
                (mean[3] + dir[3] * max_t).clamp(0.0, 255.0),
            ],
        )
    } else {
        (min_c, max_c)
    };

    let mut best = (u32::MAX, [0u8; 4], [0u8; 4], 0u8, 0u8, [0u8; 16]);
    for (p0, p1) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        let (err, e0, e1, indices) = eval_bc7_mode6_pbits(texels, f0, f1, p0, p1);
        if err < best.0 {
            best = (err, e0, e1, p0, p1, indices);
        }
    }

    // 1-pass least-squares endpoint refinement from the 4-bit weights.
    if best.0 > 0 {
        let (mut a00, mut a01, mut a11) = (0f32, 0f32, 0f32);
        let mut b0 = [0f32; 4];
        let mut b1 = [0f32; 4];
        for (i, t) in texels.iter().enumerate() {
            let w = BC7_WEIGHT4[usize::from(best.5[i])] as f32 * (1.0 / 64.0);
            let omw = 1.0 - w;
            a00 += omw * omw;
            a01 += omw * w;
            a11 += w * w;
            for c in 0..4 {
                let v = f32::from(t[c]);
                b0[c] += omw * v;
                b1[c] += w * v;
            }
        }
        let det = a00 * a11 - a01 * a01;
        if det > 1e-4 {
            let inv_det = 1.0 / det;
            for c in 0..4 {
                f0[c] = ((a11 * b0[c] - a01 * b1[c]) * inv_det).clamp(0.0, 255.0);
                f1[c] = ((a00 * b1[c] - a01 * b0[c]) * inv_det).clamp(0.0, 255.0);
            }
            for (p0, p1) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let (err, e0, e1, indices) = eval_bc7_mode6_pbits(texels, f0, f1, p0, p1);
                if err < best.0 {
                    best = (err, e0, e1, p0, p1, indices);
                }
            }
        }
    }

    pack_bc7_mode6(best.1, best.2, best.3, best.4, best.5)
}

/// Decodes a 16-byte `BC7` Mode 6 block into 16 RGBA8 texels.
pub fn decode_bc7_block(block: &[u8; 16]) -> Result<[[u8; 4]; 16], String> {
    let bits = u128::from_le_bytes(*block);
    let mode = bits.trailing_zeros();
    if mode != 6 {
        return Err(format!("unsupported BC7 mode {mode} (expected Mode 6)"));
    }
    let r0 = ((bits >> 7) & 0x7F) as u8;
    let r1 = ((bits >> 14) & 0x7F) as u8;
    let g0 = ((bits >> 21) & 0x7F) as u8;
    let g1 = ((bits >> 28) & 0x7F) as u8;
    let b0 = ((bits >> 35) & 0x7F) as u8;
    let b1 = ((bits >> 42) & 0x7F) as u8;
    let a0 = ((bits >> 49) & 0x7F) as u8;
    let a1 = ((bits >> 56) & 0x7F) as u8;
    let p0 = ((bits >> 63) & 1) as u8;
    let p1 = ((bits >> 64) & 1) as u8;

    let e0 = [
        (r0 << 1) | p0,
        (g0 << 1) | p0,
        (b0 << 1) | p0,
        (a0 << 1) | p0,
    ];
    let e1 = [
        (r1 << 1) | p1,
        (g1 << 1) | p1,
        (b1 << 1) | p1,
        (a1 << 1) | p1,
    ];

    let mut out = [[0u8; 4]; 16];
    for (i, dst) in out.iter_mut().enumerate() {
        let idx = if i == 0 {
            ((bits >> 65) & 0x7) as usize
        } else {
            ((bits >> (68 + (i - 1) * 4)) & 0xF) as usize
        };
        let w = BC7_WEIGHT4[idx];
        *dst = [
            bc7_interp(e0[0], e1[0], w),
            bc7_interp(e0[1], e1[1], w),
            bc7_interp(e0[2], e1[2], w),
            bc7_interp(e0[3], e1[3], w),
        ];
    }
    Ok(out)
}

fn astc_interp_ldr(e0: u8, e1: u8, w: u32) -> u8 {
    let c0 = u32::from(e0) * 257;
    let c1 = u32::from(e1) * 257;
    let c16 = (c0 * (64 - w) + c1 * w + 32) >> 6;
    (c16 >> 8) as u8
}

fn astc_blue_contract(r: u8, g: u8, b: u8, a: u8) -> [u8; 4] {
    let rb = ((i32::from(r) + i32::from(b)) >> 1).clamp(0, 255) as u8;
    let gb = ((i32::from(g) + i32::from(b)) >> 1).clamp(0, 255) as u8;
    [rb, gb, b, a]
}

/// Encodes a 4x4 RGBA8 block into a 16-byte Khronos-compliant `ASTC 4x4` LDR block:
/// - Solid blocks use a constant-colour LDR Void-Extent block (`0xDFC`).
/// - Opaque blocks (`alpha == 255`) use 1-partition `CEM 8` (`LDR RGB Direct`, full 8-bit endpoints)
///   with a 4x4 single-plane 3-bit weight grid (`block_mode = 0x053`).
/// - Translucent blocks use 1-partition `CEM 12` (`LDR RGBA Direct`, full 8-bit endpoints)
///   with a 4x4 single-plane 2-bit weight grid (`block_mode = 0x042`).
pub fn encode_astc4x4_block(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    let first = texels[0];
    if texels[1..].iter().all(|&t| t == first) {
        let mut out = [0u8; 16];
        // Void-Extent LDR 2D constant block: bits[11:0] = 0xDFC, bits[63:12] = all ones.
        out[0..8].copy_from_slice(&0xFFFF_FFFF_FFFF_FDFCu64.to_le_bytes());
        let r16 = u16::from(first[0]) * 257;
        let g16 = u16::from(first[1]) * 257;
        let b16 = u16::from(first[2]) * 257;
        let a16 = u16::from(first[3]) * 257;
        out[8..10].copy_from_slice(&r16.to_le_bytes());
        out[10..12].copy_from_slice(&g16.to_le_bytes());
        out[12..14].copy_from_slice(&b16.to_le_bytes());
        out[14..16].copy_from_slice(&a16.to_le_bytes());
        return out;
    }

    let opaque = texels.iter().all(|t| t[3] == 255);
    let mut min_c = [255f32; 4];
    let mut max_c = [0f32; 4];
    let mut mean = [0f32; 4];
    for t in texels {
        for c in 0..4 {
            let v = f32::from(t[c]);
            min_c[c] = min_c[c].min(v);
            max_c[c] = max_c[c].max(v);
            mean[c] += v;
        }
    }
    for m in &mut mean {
        *m *= 1.0 / 16.0;
    }

    let dims = if opaque { 3 } else { 4 };
    let mut cov = [[0f32; 4]; 4];
    for t in texels {
        let d = [
            f32::from(t[0]) - mean[0],
            f32::from(t[1]) - mean[1],
            f32::from(t[2]) - mean[2],
            if opaque {
                0.0
            } else {
                f32::from(t[3]) - mean[3]
            },
        ];
        for r in 0..dims {
            for c in 0..dims {
                cov[r][c] += d[r] * d[c];
            }
        }
    }
    let mut axis = [
        max_c[0] - min_c[0],
        max_c[1] - min_c[1],
        max_c[2] - min_c[2],
        if opaque { 0.0 } else { max_c[3] - min_c[3] },
    ];
    for _ in 0..6 {
        let mut next = [0f32; 4];
        for r in 0..dims {
            for c in 0..dims {
                next[r] += cov[r][c] * axis[c];
            }
        }
        let len =
            (next[0] * next[0] + next[1] * next[1] + next[2] * next[2] + next[3] * next[3]).sqrt();
        if len <= 1e-6 {
            break;
        }
        axis = [next[0] / len, next[1] / len, next[2] / len, next[3] / len];
    }
    let axis_len =
        (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2] + axis[3] * axis[3]).sqrt();
    let (mut e0, mut e1) = if axis_len > 1e-6 {
        let dir = [
            axis[0] / axis_len,
            axis[1] / axis_len,
            axis[2] / axis_len,
            axis[3] / axis_len,
        ];
        let mut min_t = f32::INFINITY;
        let mut max_t = f32::NEG_INFINITY;
        for t in texels {
            let mut proj = 0f32;
            for c in 0..dims {
                proj += (f32::from(t[c]) - mean[c]) * dir[c];
            }
            min_t = min_t.min(proj);
            max_t = max_t.max(proj);
        }
        (
            [
                (mean[0] + dir[0] * min_t).clamp(0.0, 255.0).round() as u8,
                (mean[1] + dir[1] * min_t).clamp(0.0, 255.0).round() as u8,
                (mean[2] + dir[2] * min_t).clamp(0.0, 255.0).round() as u8,
                if opaque {
                    255
                } else {
                    (mean[3] + dir[3] * min_t).clamp(0.0, 255.0).round() as u8
                },
            ],
            [
                (mean[0] + dir[0] * max_t).clamp(0.0, 255.0).round() as u8,
                (mean[1] + dir[1] * max_t).clamp(0.0, 255.0).round() as u8,
                (mean[2] + dir[2] * max_t).clamp(0.0, 255.0).round() as u8,
                if opaque {
                    255
                } else {
                    (mean[3] + dir[3] * max_t).clamp(0.0, 255.0).round() as u8
                },
            ],
        )
    } else {
        (
            [
                min_c[0] as u8,
                min_c[1] as u8,
                min_c[2] as u8,
                if opaque { 255 } else { min_c[3] as u8 },
            ],
            [
                max_c[0] as u8,
                max_c[1] as u8,
                max_c[2] as u8,
                if opaque { 255 } else { max_c[3] as u8 },
            ],
        )
    };

    // Enforce s1 >= s0 so ASTC LDR Direct mode uses straight RGBA endpoints without blue-contraction.
    let s0 = u16::from(e0[0]) + u16::from(e0[1]) + u16::from(e0[2]);
    let s1 = u16::from(e1[0]) + u16::from(e1[1]) + u16::from(e1[2]);
    if s1 < s0 {
        std::mem::swap(&mut e0, &mut e1);
    }

    if opaque {
        // 4x4 weight grid, single plane, 3-bit weights (R=7, H=0 -> block_mode = 0x053),
        // 1 partition, CEM = 8 (LDR RGB Direct).
        let mut weights = [0u8; 16];
        for (i, t) in texels.iter().enumerate() {
            let mut best_k = 0u8;
            let mut best_d = u32::MAX;
            for (k, &w) in ASTC_WEIGHT3.iter().enumerate() {
                let dr = i32::from(t[0]) - i32::from(astc_interp_ldr(e0[0], e1[0], w));
                let dg = i32::from(t[1]) - i32::from(astc_interp_ldr(e0[1], e1[1], w));
                let db = i32::from(t[2]) - i32::from(astc_interp_ldr(e0[2], e1[2], w));
                let d = (dr * dr + dg * dg + db * db) as u32;
                if d < best_d {
                    best_d = d;
                    best_k = k as u8;
                }
            }
            weights[i] = best_k;
        }
        let mut bits: u128 = 0x053 | (8u128 << 13);
        let endpoints = [e0[0], e1[0], e0[1], e1[1], e0[2], e1[2]];
        for (i, &ep) in endpoints.iter().enumerate() {
            bits |= u128::from(ep) << (17 + i * 8);
        }
        for (i, &w) in weights.iter().enumerate() {
            for b in 0..3 {
                if (w >> b) & 1 != 0 {
                    bits |= 1u128 << (127 - (i * 3 + b));
                }
            }
        }
        bits.to_le_bytes()
    } else {
        // 4x4 weight grid, single plane, 2-bit weights (R=4, H=0 -> block_mode = 0x042),
        // 1 partition, CEM = 12 (LDR RGBA Direct).
        let mut weights = [0u8; 16];
        for (i, t) in texels.iter().enumerate() {
            let mut best_k = 0u8;
            let mut best_d = u32::MAX;
            for (k, &w) in ASTC_WEIGHT2.iter().enumerate() {
                let dr = i32::from(t[0]) - i32::from(astc_interp_ldr(e0[0], e1[0], w));
                let dg = i32::from(t[1]) - i32::from(astc_interp_ldr(e0[1], e1[1], w));
                let db = i32::from(t[2]) - i32::from(astc_interp_ldr(e0[2], e1[2], w));
                let da = i32::from(t[3]) - i32::from(astc_interp_ldr(e0[3], e1[3], w));
                let d = (dr * dr + dg * dg + db * db + da * da) as u32;
                if d < best_d {
                    best_d = d;
                    best_k = k as u8;
                }
            }
            weights[i] = best_k;
        }
        let mut bits: u128 = 0x042 | (12u128 << 13);
        let endpoints = [e0[0], e1[0], e0[1], e1[1], e0[2], e1[2], e0[3], e1[3]];
        for (i, &ep) in endpoints.iter().enumerate() {
            bits |= u128::from(ep) << (17 + i * 8);
        }
        for (i, &w) in weights.iter().enumerate() {
            for b in 0..2 {
                if (w >> b) & 1 != 0 {
                    bits |= 1u128 << (127 - (i * 2 + b));
                }
            }
        }
        bits.to_le_bytes()
    }
}

/// Decodes a 16-byte `ASTC 4x4` LDR block (Void-Extent constant blocks and 1-partition 4x4
/// single-plane `LDR RGB Direct` / `LDR RGBA Direct` blocks) into 16 RGBA8 texels.
pub fn decode_astc4x4_block(block: &[u8; 16]) -> Result<[[u8; 4]; 16], String> {
    let bits = u128::from_le_bytes(*block);
    // Void-Extent LDR 2D block: bits[11:0] == 0xDFC.
    if (bits & 0xFFF) == 0xDFC {
        let r = ((bits >> 72) & 0xFF) as u8;
        let g = ((bits >> 88) & 0xFF) as u8;
        let b = ((bits >> 104) & 0xFF) as u8;
        let a = ((bits >> 120) & 0xFF) as u8;
        return Ok([[r, g, b, a]; 16]);
    }

    let block_mode = (bits & 0x7FF) as u32;
    let part_minus_one = ((bits >> 11) & 0x3) as u32;
    if part_minus_one != 0 {
        return Err(format!(
            "unsupported ASTC 4x4 multi-partition count {}",
            part_minus_one + 1
        ));
    }
    if (block_mode >> 10) & 1 != 0 {
        return Err("unsupported ASTC 4x4 dual-plane block".into());
    }
    // Decode 2D block mode row 0: bits[1:0] != 0, bits[3:2] == 0 -> width = B + 4, height = A + 2.
    let low2 = block_mode & 3;
    let mid2 = (block_mode >> 2) & 3;
    if low2 == 0 || mid2 != 0 {
        return Err(format!(
            "unsupported ASTC 4x4 block mode 0x{block_mode:03x}"
        ));
    }
    let r_code = (low2 << 1) | ((block_mode >> 4) & 1);
    let a_grid = (block_mode >> 5) & 3;
    let b_grid = (block_mode >> 7) & 3;
    let high_prec = (block_mode >> 9) & 1;
    let grid_w = b_grid + 4;
    let grid_h = a_grid + 2;
    if grid_w != 4 || grid_h != 4 {
        return Err(format!("unsupported ASTC weight grid {grid_w}x{grid_h}"));
    }

    let weight_bits_per_texel = match (high_prec, r_code) {
        (0, 2) => 1usize,
        (0, 4) => 2usize,
        (0, 7) => 3usize,
        (1, 4) => 4usize,
        _ => {
            return Err(format!(
                "unsupported ASTC weight range (H={high_prec}, R={r_code})"
            ));
        }
    };

    let cem = ((bits >> 13) & 0xF) as u32;
    let (e0, e1) = match cem {
        8 => {
            if 17 + 48 + 16 * weight_bits_per_texel > 128 {
                return Err("ASTC CEM 8 endpoints exceed block budget".into());
            }
            let mut v = [0u8; 6];
            for (i, dst) in v.iter_mut().enumerate() {
                *dst = ((bits >> (17 + i * 8)) & 0xFF) as u8;
            }
            let s0 = u16::from(v[0]) + u16::from(v[2]) + u16::from(v[4]);
            let s1 = u16::from(v[1]) + u16::from(v[3]) + u16::from(v[5]);
            if s1 >= s0 {
                ([v[0], v[2], v[4], 255], [v[1], v[3], v[5], 255])
            } else {
                (
                    astc_blue_contract(v[1], v[3], v[5], 255),
                    astc_blue_contract(v[0], v[2], v[4], 255),
                )
            }
        }
        12 => {
            if 17 + 64 + 16 * weight_bits_per_texel > 128 {
                return Err("ASTC CEM 12 endpoints exceed block budget".into());
            }
            let mut v = [0u8; 8];
            for (i, dst) in v.iter_mut().enumerate() {
                *dst = ((bits >> (17 + i * 8)) & 0xFF) as u8;
            }
            let s0 = u16::from(v[0]) + u16::from(v[2]) + u16::from(v[4]);
            let s1 = u16::from(v[1]) + u16::from(v[3]) + u16::from(v[5]);
            if s1 >= s0 {
                ([v[0], v[2], v[4], v[6]], [v[1], v[3], v[5], v[7]])
            } else {
                (
                    astc_blue_contract(v[1], v[3], v[5], v[7]),
                    astc_blue_contract(v[0], v[2], v[4], v[6]),
                )
            }
        }
        _ => return Err(format!("unsupported ASTC colour endpoint mode {cem}")),
    };

    const ASTC_WEIGHT1: [u32; 2] = [0, 64];
    const ASTC_WEIGHT4: [u32; 16] = [0, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64];

    let mut out = [[0u8; 4]; 16];
    for (i, dst) in out.iter_mut().enumerate() {
        let mut q = 0usize;
        for b in 0..weight_bits_per_texel {
            let bit = ((bits >> (127 - (i * weight_bits_per_texel + b))) & 1) as usize;
            q |= bit << b;
        }
        let w = match weight_bits_per_texel {
            1 => ASTC_WEIGHT1[q],
            2 => ASTC_WEIGHT2[q],
            3 => ASTC_WEIGHT3[q],
            4 => ASTC_WEIGHT4[q],
            _ => unreachable!(),
        };
        *dst = [
            astc_interp_ldr(e0[0], e1[0], w),
            astc_interp_ldr(e0[1], e1[1], w),
            astc_interp_ldr(e0[2], e1[2], w),
            astc_interp_ldr(e0[3], e1[3], w),
        ];
    }
    Ok(out)
}

fn extract_block(rgba: &[u8], width: u32, height: u32, bx: u32, by: u32) -> [[u8; 4]; 16] {
    let mut block = [[0u8; 4]; 16];
    let w_max = width - 1;
    let h_max = height - 1;
    for py in 0..4u32 {
        let sy = (by * 4 + py).min(h_max);
        for px in 0..4u32 {
            let sx = (bx * 4 + px).min(w_max);
            let off = ((sy * width + sx) * 4) as usize;
            block[(py * 4 + px) as usize] =
                [rgba[off], rgba[off + 1], rgba[off + 2], rgba[off + 3]];
        }
    }
    block
}

fn scatter_block(block: &[[u8; 4]; 16], out: &mut [u8], width: u32, height: u32, bx: u32, by: u32) {
    for py in 0..4u32 {
        let dy = by * 4 + py;
        if dy >= height {
            break;
        }
        for px in 0..4u32 {
            let dx = bx * 4 + px;
            if dx >= width {
                break;
            }
            let off = ((dy * width + dx) * 4) as usize;
            out[off..off + 4].copy_from_slice(&block[(py * 4 + px) as usize]);
        }
    }
}

/// Compresses a single RGBA8 mip level of `width` by `height` texels into `compression` blocks.
pub fn encode_rgba8(
    compression: TextureCompression,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let limit = 1..=MAX_TEXTURE_SIZE;
    if !limit.contains(&width) || !limit.contains(&height) {
        return Err(format!("invalid {width}x{height} texture extent"));
    }
    let expected = u64::from(width) * u64::from(height) * 4;
    if rgba.len() as u64 != expected {
        return Err(format!(
            "expected {expected} RGBA8 bytes for {width}x{height}, got {}",
            rgba.len()
        ));
    }
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let mut out = Vec::with_capacity(compression.mip_byte_len(width, height));
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block = extract_block(rgba, width, height, bx, by);
            match compression {
                TextureCompression::Bc1 => out.extend_from_slice(&encode_bc1_block(&block)),
                TextureCompression::Bc7 => out.extend_from_slice(&encode_bc7_block(&block)),
                TextureCompression::Astc4x4 => out.extend_from_slice(&encode_astc4x4_block(&block)),
            }
        }
    }
    Ok(out)
}

/// Decompresses a single `compression` mip level of `width` by `height` texels into RGBA8 bytes.
pub fn decode_rgba8(
    compression: TextureCompression,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let limit = 1..=MAX_TEXTURE_SIZE;
    if !limit.contains(&width) || !limit.contains(&height) {
        return Err(format!("invalid {width}x{height} texture extent"));
    }
    let expected = compression.mip_byte_len(width, height);
    if data.len() != expected {
        return Err(format!(
            "expected {expected} {} bytes for {width}x{height}, got {}",
            compression.name(),
            data.len()
        ));
    }
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let block_bytes = compression.block_bytes() as usize;
    let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
    let mut idx = 0usize;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let slice = &data[idx..idx + block_bytes];
            idx += block_bytes;
            let decoded = match compression {
                TextureCompression::Bc1 => {
                    let b: &[u8; 8] = slice.try_into().expect("8-byte BC1 block");
                    decode_bc1_block(b)
                }
                TextureCompression::Bc7 => {
                    let b: &[u8; 16] = slice.try_into().expect("16-byte BC7 block");
                    decode_bc7_block(b)?
                }
                TextureCompression::Astc4x4 => {
                    let b: &[u8; 16] = slice.try_into().expect("16-byte ASTC block");
                    decode_astc4x4_block(b)?
                }
            };
            scatter_block(&decoded, &mut out, width, height, bx, by);
        }
    }
    Ok(out)
}

/// Compresses a [`Texture`] (generating its full RGBA8 mip chain first if only level 0 or an
/// embedded PNG is present) into hardware [`TextureCompression`] blocks.
pub fn compress_texture(
    texture: &Texture,
    compression: TextureCompression,
) -> Result<Texture, String> {
    if texture.compression() == Some(compression) {
        return Ok(texture.clone());
    }
    let source = if texture.is_compressed() {
        decompress_texture(texture)?
    } else if let Some(enc) = &texture.encoded {
        if enc.mime != "image/png" {
            return Err(format!(
                "cannot block-compress host-encoded {} texture without RGBA8 texels",
                enc.mime
            ));
        }
        let img = crate::png::decode(&enc.data)?;
        if img.width != texture.width || img.height != texture.height {
            return Err("PNG dimensions do not match texture header".into());
        }
        let filter = if texture.srgb {
            Filter::Srgb
        } else {
            Filter::Linear
        };
        mip::texture(texture.width, texture.height, img.rgba, filter)?
    } else if texture.levels.len() == 1 && Texture::level_count(texture.width, texture.height) > 1 {
        let filter = if texture.srgb {
            Filter::Srgb
        } else {
            Filter::Linear
        };
        mip::texture(
            texture.width,
            texture.height,
            texture.levels[0].clone(),
            filter,
        )?
    } else {
        texture.clone()
    };

    let mut levels = Vec::with_capacity(source.levels.len());
    for (i, rgba) in source.levels.iter().enumerate() {
        let (w, h) = Texture::level_extent(source.width, source.height, i);
        levels.push(encode_rgba8(compression, rgba, w, h)?);
    }
    Ok(Texture::compressed(
        source.width,
        source.height,
        source.srgb,
        compression,
        levels,
    ))
}

/// Compresses every texture in `world` in place, using [`Filter::Normal`] mip downsampling for
/// normal-map textures when mip chains must be built from level 0.
pub fn compress_world_textures(
    world: &mut ganbate_core::world::World,
    compression: TextureCompression,
) -> Result<(), String> {
    let mut normal_used = vec![false; world.textures.len()];
    let mut color_used = vec![false; world.textures.len()];
    for m in &world.materials {
        if let Some(i) = m.normal_texture
            && let Some(slot) = normal_used.get_mut(i as usize)
        {
            *slot = true;
        }
        for slot_idx in [m.base_color_texture, m.emissive_texture]
            .into_iter()
            .flatten()
        {
            if let Some(slot) = color_used.get_mut(slot_idx as usize) {
                *slot = true;
            }
        }
    }
    for (i, tex) in world.textures.iter_mut().enumerate() {
        if tex.compression() == Some(compression) {
            continue;
        }
        if !tex.is_compressed() && normal_used[i] && !color_used[i] && !tex.srgb {
            let rgba0 = if let Some(enc) = &tex.encoded {
                if enc.mime != "image/png" {
                    return Err(format!(
                        "texture {i}: cannot block-compress host-encoded {} texture",
                        enc.mime
                    ));
                }
                let img = crate::png::decode(&enc.data)?;
                if img.width != tex.width || img.height != tex.height {
                    return Err(format!("texture {i}: PNG dimensions do not match header"));
                }
                Some(img.rgba)
            } else if tex.levels.len() == 1 && Texture::level_count(tex.width, tex.height) > 1 {
                Some(tex.levels[0].clone())
            } else {
                None
            };
            if let Some(rgba) = rgba0 {
                let chained = mip::texture(tex.width, tex.height, rgba, Filter::Normal)?;
                *tex = compress_texture(&chained, compression)?;
                continue;
            }
        }
        *tex = compress_texture(tex, compression)?;
    }
    Ok(())
}

/// Decompresses a block-compressed [`Texture`] back into uncompressed RGBA8 mip levels.
pub fn decompress_texture(texture: &Texture) -> Result<Texture, String> {
    let Some(comp) = texture.compression() else {
        return Ok(texture.clone());
    };
    let mut levels = Vec::with_capacity(texture.levels.len());
    for (i, data) in texture.levels.iter().enumerate() {
        let (w, h) = Texture::level_extent(texture.width, texture.height, i);
        levels.push(decode_rgba8(comp, data, w, h)?);
    }
    Ok(Texture {
        width: texture.width,
        height: texture.height,
        srgb: texture.srgb,
        levels,
        encoded: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ganbate_core::world::{Material, World};

    fn gradient_rgba(w: u32, h: u32, with_alpha: bool) -> Vec<u8> {
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w.max(1)) as u8;
                let g = ((y * 255) / h.max(1)) as u8;
                let b = (((x + y) * 127) / (w + h).max(1)) as u8;
                let a = if with_alpha {
                    if (x + y) % 7 == 0 { 0 } else { 220 }
                } else {
                    255
                };
                out.extend_from_slice(&[r, g, b, a]);
            }
        }
        out
    }

    #[test]
    fn bc1_bc7_and_astc4x4_roundtrip_gradients_and_odd_extents() {
        // 1. Collinear smooth gradient across normal and odd/sub-block extents: exact 1D line fit.
        for &(w, h) in &[(16, 16), (7, 5), (2, 1), (1, 1)] {
            let mut linear_grad = Vec::with_capacity((w * h * 4) as usize);
            for y in 0..h {
                for x in 0..w {
                    let t = ((x + y) * 255) / (w + h).max(1);
                    linear_grad.extend_from_slice(&[
                        t as u8,
                        (255 - t) as u8,
                        (64 + t / 2) as u8,
                        255,
                    ]);
                }
            }
            for comp in [
                TextureCompression::Bc1,
                TextureCompression::Bc7,
                TextureCompression::Astc4x4,
            ] {
                let enc = encode_rgba8(comp, &linear_grad, w, h).expect("encode");
                assert_eq!(enc.len(), comp.mip_byte_len(w, h));
                let dec = decode_rgba8(comp, &enc, w, h).expect("decode");
                assert_eq!(dec.len(), linear_grad.len());
                let max_err = linear_grad
                    .iter()
                    .zip(dec.iter())
                    .map(|(&a, &b)| a.abs_diff(b))
                    .max()
                    .unwrap_or(0);
                let bound = match comp {
                    TextureCompression::Bc1 => 24,
                    TextureCompression::Bc7 => 10,
                    TextureCompression::Astc4x4 => 14,
                };
                assert!(
                    max_err <= bound,
                    "{comp:?} {w}x{h} collinear max channel error {max_err} > {bound}"
                );
            }
        }

        // 2. 2D orthogonal RGB gradient at 64x64: verify high PSNR / low RMSE across all codecs.
        let ortho = gradient_rgba(64, 64, false);
        for comp in [
            TextureCompression::Bc1,
            TextureCompression::Bc7,
            TextureCompression::Astc4x4,
        ] {
            let enc = encode_rgba8(comp, &ortho, 64, 64).expect("encode 64x64");
            let dec = decode_rgba8(comp, &enc, 64, 64).expect("decode 64x64");
            let mse: f64 = ortho
                .iter()
                .zip(dec.iter())
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum::<f64>()
                / (ortho.len() as f64);
            let rmse = mse.sqrt();
            let max_rmse = match comp {
                TextureCompression::Bc1 => 6.0,
                TextureCompression::Bc7 => 4.0,
                TextureCompression::Astc4x4 => 4.5,
            };
            assert!(
                rmse <= max_rmse,
                "{comp:?} 64x64 rmse {rmse:.2} > {max_rmse}"
            );
        }

        // 3. Discontinuous alpha punch-through on BC1 (1-bit) and smooth/mixed alpha on BC7 & ASTC 4x4.
        let alpha_src = gradient_rgba(32, 32, true);
        for comp in [TextureCompression::Bc7, TextureCompression::Astc4x4] {
            let enc = encode_rgba8(comp, &alpha_src, 32, 32).expect("encode alpha");
            let dec = decode_rgba8(comp, &enc, 32, 32).expect("decode alpha");
            let alpha_rmse = (alpha_src
                .chunks_exact(4)
                .zip(dec.chunks_exact(4))
                .map(|(a, b)| {
                    let d = f64::from(a[3]) - f64::from(b[3]);
                    d * d
                })
                .sum::<f64>()
                / (32.0 * 32.0))
                .sqrt();
            assert!(
                alpha_rmse <= 35.0,
                "{comp:?} punch-through alpha_rmse {alpha_rmse:.2}"
            );
        }
        let bc1_enc = encode_rgba8(TextureCompression::Bc1, &alpha_src, 32, 32).unwrap();
        let bc1_dec = decode_rgba8(TextureCompression::Bc1, &bc1_enc, 32, 32).unwrap();
        assert_eq!(bc1_dec[3], 0);
        assert_eq!(bc1_dec[7], 255);
    }

    #[test]
    fn cooked_world_preserves_compressed_mip_chains_at_4x_to_8x_vram_reduction() {
        let rgba = gradient_rgba(64, 64, false);
        let raw_tex = crate::mip::texture(64, 64, rgba, Filter::Srgb).unwrap();
        let raw_vram: usize = raw_tex.levels.iter().map(Vec::len).sum();

        for comp in [
            TextureCompression::Bc1,
            TextureCompression::Bc7,
            TextureCompression::Astc4x4,
        ] {
            let mut world = World::default();
            world.textures.push(raw_tex.clone());
            world.materials.push(Material {
                name: "mat".into(),
                base_color_texture: Some(0),
                ..Material::default()
            });
            compress_world_textures(&mut world, comp).unwrap();
            let comp_tex = &world.textures[0];
            assert_eq!(comp_tex.compression(), Some(comp));
            let comp_vram: usize = comp_tex.levels.iter().map(Vec::len).sum();
            assert!(
                comp_vram * 3 < raw_vram,
                "{comp:?}: {comp_vram} vs {raw_vram}"
            );

            let bytes = crate::cooked::encode(&world).unwrap();
            let decoded_world = crate::cooked::decode(&bytes).unwrap();
            assert_eq!(decoded_world.textures[0], *comp_tex);

            let unpacked = decompress_texture(&decoded_world.textures[0]).unwrap();
            assert_eq!(unpacked.levels.len(), raw_tex.levels.len());
            assert_eq!(unpacked.levels[0].len(), 64 * 64 * 4);
        }
    }
}
