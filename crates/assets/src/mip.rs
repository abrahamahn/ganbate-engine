//! Complete mip chains for RGBA8 textures. Each level halves the previous one (rounding down)
//! with a box filter; for an odd source size the last texel of a row or column also takes the
//! leftover source texel, so nothing is dropped. sRGB colour is averaged in linear light and
//! normal maps are renormalized. Alpha is always averaged linearly.
use std::sync::OnceLock;

use ganbate_core::world::{MAX_TEXTURE_SIZE, Texture};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// sRGB-encoded colour (base colour, emissive).
    Srgb,
    /// Linear data.
    Linear,
    /// Tangent-space normals encoded as `n * 0.5 + 0.5`.
    Normal,
}

struct MipTables {
    to_linear: [f32; 256],
    unorm_f32: [f32; 256],
    norm_f32: [f32; 256],
    thresh: [f32; 257],
    coarse: [u8; 4096],
}

static TABLES: OnceLock<MipTables> = OnceLock::new();

fn srgb_to_linear_ref(v: u8) -> f32 {
    let c = f32::from(v) / 255.;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_ref(l: f32) -> u8 {
    let c = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1. / 2.4) - 0.055
    };
    unorm(c)
}

#[inline]
fn unorm(v: f32) -> u8 {
    (v.clamp(0., 1.) * 255. + 0.5) as u8
}

#[inline]
fn tables() -> &'static MipTables {
    TABLES.get_or_init(|| {
        let to_linear = std::array::from_fn(|v| srgb_to_linear_ref(v as u8));
        let unorm_f32 = std::array::from_fn(|v| f32::from(v as u8) / 255.);
        let norm_f32 = std::array::from_fn(|v| f32::from(v as u8) / 127.5 - 1.);

        let mut thresh = [0.0f32; 257];
        thresh[256] = f32::INFINITY;
        let one_bits = 1.0f32.to_bits();
        let mut lo_start = 0u32;
        for k in 1u8..=255 {
            let mut lo = lo_start;
            let mut hi = one_bits;
            while lo < hi {
                let mid = lo + ((hi - lo) >> 1);
                if linear_to_srgb_ref(f32::from_bits(mid)) >= k {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            thresh[usize::from(k)] = f32::from_bits(lo);
            lo_start = lo;
        }

        let mut coarse = [0u8; 4096];
        let bucket = |x: f32| (x.sqrt() * 4095.0) as usize;
        for k in (0u8..=255).rev() {
            let min_l = if k == 0 {
                f32::from_bits(1)
            } else {
                thresh[usize::from(k)]
            };
            let max_bits = if k == 255 {
                one_bits - 1
            } else {
                thresh[usize::from(k) + 1].to_bits() - 1
            };
            let max_l = f32::from_bits(max_bits);
            let b0 = bucket(min_l);
            let b1 = bucket(max_l);
            coarse[b0..=b1].fill(k);
        }

        MipTables {
            to_linear,
            unorm_f32,
            norm_f32,
            thresh,
            coarse,
        }
    })
}

#[inline]
pub fn srgb_to_linear(v: u8) -> f32 {
    tables().to_linear[usize::from(v)]
}

#[inline]
fn linear_to_srgb_with(l: f32, t: &MipTables) -> u8 {
    if l <= 0.0 || l.is_nan() {
        return 0;
    }
    if l >= 1.0 {
        return 255;
    }
    let idx = (l.sqrt() * 4095.0) as usize;
    let mut k = usize::from(t.coarse[idx]);
    while l >= t.thresh[k + 1] {
        k += 1;
    }
    k as u8
}

#[inline]
pub fn linear_to_srgb_fast(l: f32) -> u8 {
    linear_to_srgb_with(l, tables())
}

#[inline]
pub fn linear_to_srgb(l: f32) -> u8 {
    linear_to_srgb_with(l, tables())
}

#[inline]
fn encode_normal(r: f32, g: f32, b: f32) -> [u8; 3] {
    let len = (r * r + g * g + b * b).sqrt();
    let v = if len > 1e-6 {
        [r / len, g / len, b / len]
    } else {
        [0., 0., 1.]
    };
    v.map(|c| unorm(c * 0.5 + 0.5))
}

/// Source texels covered by destination texel `d` along one axis.
#[inline]
fn taps(d: usize, src: usize, dst: usize) -> std::ops::Range<usize> {
    2 * d..if d + 1 == dst { src } else { 2 * d + 2 }
}

#[inline]
fn downsample_border_texel(
    src: &[u8],
    sw: usize,
    xs: std::ops::Range<usize>,
    ys: std::ops::Range<usize>,
    filter: Filter,
    t: &MipTables,
) -> [u8; 4] {
    let mut acc = [0f32; 4];
    let n = (ys.len() * xs.len()) as f32;
    match filter {
        Filter::Srgb => {
            for sy in ys {
                for p in src[(sy * sw + xs.start) * 4..(sy * sw + xs.end) * 4].chunks_exact(4) {
                    acc[0] += t.to_linear[usize::from(p[0])];
                    acc[1] += t.to_linear[usize::from(p[1])];
                    acc[2] += t.to_linear[usize::from(p[2])];
                    acc[3] += t.unorm_f32[usize::from(p[3])];
                }
            }
            let [r, g, b, a] = acc.map(|v| v / n);
            [
                linear_to_srgb_with(r, t),
                linear_to_srgb_with(g, t),
                linear_to_srgb_with(b, t),
                unorm(a),
            ]
        }
        Filter::Linear => {
            for sy in ys {
                for p in src[(sy * sw + xs.start) * 4..(sy * sw + xs.end) * 4].chunks_exact(4) {
                    acc[0] += t.unorm_f32[usize::from(p[0])];
                    acc[1] += t.unorm_f32[usize::from(p[1])];
                    acc[2] += t.unorm_f32[usize::from(p[2])];
                    acc[3] += t.unorm_f32[usize::from(p[3])];
                }
            }
            acc.map(|v| unorm(v / n))
        }
        Filter::Normal => {
            for sy in ys {
                for p in src[(sy * sw + xs.start) * 4..(sy * sw + xs.end) * 4].chunks_exact(4) {
                    acc[0] += t.norm_f32[usize::from(p[0])];
                    acc[1] += t.norm_f32[usize::from(p[1])];
                    acc[2] += t.norm_f32[usize::from(p[2])];
                    acc[3] += t.unorm_f32[usize::from(p[3])];
                }
            }
            let [r, g, b, a] = acc.map(|v| v / n);
            let rgb = encode_normal(r, g, b);
            [rgb[0], rgb[1], rgb[2], unorm(a)]
        }
    }
}

fn downsample(
    src: &[u8],
    (sw, sh): (usize, usize),
    (dw, dh): (usize, usize),
    filter: Filter,
    t: &MipTables,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(dw * dh * 4);
    let fast_w = if sw == 2 * dw {
        dw
    } else {
        dw.saturating_sub(1)
    };
    let fast_h = if sh == 2 * dh {
        dh
    } else {
        dh.saturating_sub(1)
    };
    let row_bytes = sw * 4;

    match filter {
        Filter::Srgb => {
            for y in 0..fast_h {
                let r0 = &src[(2 * y) * row_bytes..(2 * y + 1) * row_bytes];
                let r1 = &src[(2 * y + 1) * row_bytes..(2 * y + 2) * row_bytes];
                for x in 0..fast_w {
                    let bx = x * 8;
                    let p00 = &r0[bx..bx + 4];
                    let p01 = &r0[bx + 4..bx + 8];
                    let p10 = &r1[bx..bx + 4];
                    let p11 = &r1[bx + 4..bx + 8];
                    let r = (((t.to_linear[usize::from(p00[0])]
                        + t.to_linear[usize::from(p01[0])])
                        + t.to_linear[usize::from(p10[0])])
                        + t.to_linear[usize::from(p11[0])])
                        * 0.25;
                    let g = (((t.to_linear[usize::from(p00[1])]
                        + t.to_linear[usize::from(p01[1])])
                        + t.to_linear[usize::from(p10[1])])
                        + t.to_linear[usize::from(p11[1])])
                        * 0.25;
                    let b = (((t.to_linear[usize::from(p00[2])]
                        + t.to_linear[usize::from(p01[2])])
                        + t.to_linear[usize::from(p10[2])])
                        + t.to_linear[usize::from(p11[2])])
                        * 0.25;
                    let a = (((t.unorm_f32[usize::from(p00[3])]
                        + t.unorm_f32[usize::from(p01[3])])
                        + t.unorm_f32[usize::from(p10[3])])
                        + t.unorm_f32[usize::from(p11[3])])
                        * 0.25;
                    out.extend_from_slice(&[
                        linear_to_srgb_with(r, t),
                        linear_to_srgb_with(g, t),
                        linear_to_srgb_with(b, t),
                        unorm(a),
                    ]);
                }
                for x in fast_w..dw {
                    out.extend_from_slice(&downsample_border_texel(
                        src,
                        sw,
                        taps(x, sw, dw),
                        2 * y..2 * y + 2,
                        Filter::Srgb,
                        t,
                    ));
                }
            }
        }
        Filter::Linear => {
            for y in 0..fast_h {
                let r0 = &src[(2 * y) * row_bytes..(2 * y + 1) * row_bytes];
                let r1 = &src[(2 * y + 1) * row_bytes..(2 * y + 2) * row_bytes];
                for x in 0..fast_w {
                    let bx = x * 8;
                    let p00 = &r0[bx..bx + 4];
                    let p01 = &r0[bx + 4..bx + 8];
                    let p10 = &r1[bx..bx + 4];
                    let p11 = &r1[bx + 4..bx + 8];
                    let ch = |c: usize| {
                        unorm(
                            (((t.unorm_f32[usize::from(p00[c])]
                                + t.unorm_f32[usize::from(p01[c])])
                                + t.unorm_f32[usize::from(p10[c])])
                                + t.unorm_f32[usize::from(p11[c])])
                                * 0.25,
                        )
                    };
                    out.extend_from_slice(&[ch(0), ch(1), ch(2), ch(3)]);
                }
                for x in fast_w..dw {
                    out.extend_from_slice(&downsample_border_texel(
                        src,
                        sw,
                        taps(x, sw, dw),
                        2 * y..2 * y + 2,
                        Filter::Linear,
                        t,
                    ));
                }
            }
        }
        Filter::Normal => {
            for y in 0..fast_h {
                let r0 = &src[(2 * y) * row_bytes..(2 * y + 1) * row_bytes];
                let r1 = &src[(2 * y + 1) * row_bytes..(2 * y + 2) * row_bytes];
                for x in 0..fast_w {
                    let bx = x * 8;
                    let p00 = &r0[bx..bx + 4];
                    let p01 = &r0[bx + 4..bx + 8];
                    let p10 = &r1[bx..bx + 4];
                    let p11 = &r1[bx + 4..bx + 8];
                    let r = ((((0.0 + t.norm_f32[usize::from(p00[0])])
                        + t.norm_f32[usize::from(p01[0])])
                        + t.norm_f32[usize::from(p10[0])])
                        + t.norm_f32[usize::from(p11[0])])
                        * 0.25;
                    let g = ((((0.0 + t.norm_f32[usize::from(p00[1])])
                        + t.norm_f32[usize::from(p01[1])])
                        + t.norm_f32[usize::from(p10[1])])
                        + t.norm_f32[usize::from(p11[1])])
                        * 0.25;
                    let b = ((((0.0 + t.norm_f32[usize::from(p00[2])])
                        + t.norm_f32[usize::from(p01[2])])
                        + t.norm_f32[usize::from(p10[2])])
                        + t.norm_f32[usize::from(p11[2])])
                        * 0.25;
                    let a = (((t.unorm_f32[usize::from(p00[3])]
                        + t.unorm_f32[usize::from(p01[3])])
                        + t.unorm_f32[usize::from(p10[3])])
                        + t.unorm_f32[usize::from(p11[3])])
                        * 0.25;
                    let rgb = encode_normal(r, g, b);
                    out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], unorm(a)]);
                }
                for x in fast_w..dw {
                    out.extend_from_slice(&downsample_border_texel(
                        src,
                        sw,
                        taps(x, sw, dw),
                        2 * y..2 * y + 2,
                        Filter::Normal,
                        t,
                    ));
                }
            }
        }
    }

    for y in fast_h..dh {
        let ys = taps(y, sh, dh);
        for x in 0..dw {
            out.extend_from_slice(&downsample_border_texel(
                src,
                sw,
                taps(x, sw, dw),
                ys.clone(),
                filter,
                t,
            ));
        }
    }
    out
}

/// Bytes in a complete chain, or `None` past the size limits.
pub fn chain_bytes(width: u32, height: u32) -> Option<u64> {
    let limit = 1..=MAX_TEXTURE_SIZE;
    if !limit.contains(&width) || !limit.contains(&height) {
        return None;
    }
    Some(
        (0..Texture::level_count(width, height))
            .map(|i| {
                let (w, h) = Texture::level_extent(width, height, i);
                u64::from(w) * u64::from(h) * 4
            })
            .sum(),
    )
}

/// Build a texture with a complete chain from its top level. sRGB filtering marks it sRGB.
pub fn texture(width: u32, height: u32, rgba: Vec<u8>, filter: Filter) -> Result<Texture, String> {
    if chain_bytes(width, height).is_none()
        || rgba.len() as u64 != u64::from(width) * u64::from(height) * 4
    {
        return Err(format!("invalid {width}x{height} RGBA8 texture"));
    }
    let t = tables();
    let count = Texture::level_count(width, height);
    let mut levels = Vec::with_capacity(count);
    levels.push(rgba);
    for i in 1..count {
        let (sw, sh) = Texture::level_extent(width, height, i - 1);
        let (dw, dh) = Texture::level_extent(width, height, i);
        let next = downsample(
            &levels[i - 1],
            (sw as usize, sh as usize),
            (dw as usize, dh as usize),
            filter,
            t,
        );
        levels.push(next);
    }
    Ok(Texture {
        width,
        height,
        srgb: filter == Filter::Srgb,
        levels,
        encoded: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_to_srgb_fast_is_bit_identical_to_powf_reference() {
        let t = tables();
        for v in 0u8..=255 {
            assert_eq!(srgb_to_linear(v).to_bits(), srgb_to_linear_ref(v).to_bits());
        }
        // Special float cases.
        for special in [
            0.0f32,
            -0.0,
            -1.0,
            -1e-38,
            f32::from_bits(1),
            0.003_130_8,
            0.5,
            1.0,
            f32::from_bits(1.0f32.to_bits() - 1),
            f32::from_bits(1.0f32.to_bits() + 1),
            2.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            assert_eq!(
                linear_to_srgb_fast(special),
                linear_to_srgb_ref(special),
                "special={special:?}"
            );
        }
        // Exact ULP neighborhood around every single threshold in 1..=255.
        for k in 1..=255 {
            let center = t.thresh[k].to_bits();
            for delta in -64i32..=64 {
                let bits = center.saturating_add_signed(delta);
                let l = f32::from_bits(bits);
                assert_eq!(
                    linear_to_srgb_fast(l),
                    linear_to_srgb_ref(l),
                    "threshold {k} delta {delta} l={l}"
                );
            }
        }
        // 2,000,000 samples across all float bit patterns in [0.0, 1.0] + linear steps.
        let one_bits = 1.0f32.to_bits();
        let step = (one_bits / 1_000_000).max(1);
        let mut bits = 0u32;
        while bits <= one_bits {
            let l = f32::from_bits(bits);
            assert_eq!(
                linear_to_srgb_fast(l),
                linear_to_srgb_ref(l),
                "bits={bits:#x}"
            );
            bits = bits.saturating_add(step);
        }
        for i in 0..=1_000_000u32 {
            let l = i as f32 / 1_000_000.0;
            assert_eq!(linear_to_srgb_fast(l), linear_to_srgb_ref(l), "l={l}");
        }
    }
}
