//! PNG decoding to RGBA8 for untrusted input, and lossless RGBA8 encoding. Decoding accepts
//! non-interlaced greyscale, RGB, paletted, grey-alpha and RGBA images at their standard bit
//! depths (16-bit samples are rounded to 8). Every CRC and the zlib Adler-32 are checked.
use crate::zlib;

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
pub const MAX_DIMENSION: u32 = 16_384;
/// 128 MiB of RGBA8 output.
pub const MAX_PIXELS: u64 = 1 << 25;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

const CRC_TABLE: [[u32; 256]; 4] = {
    let mut table = [[0u32; 256]; 4];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[0][n] = c;
        n += 1;
    }
    let mut t = 1;
    while t < 4 {
        let mut i = 0;
        while i < 256 {
            let prev = table[t - 1][i];
            table[t][i] = table[0][(prev & 0xff) as usize] ^ (prev >> 8);
            i += 1;
        }
        t += 1;
    }
    table
};

pub fn crc32(parts: &[&[u8]]) -> u32 {
    let mut c = !0u32;
    for part in parts {
        let mut chunks = part.chunks_exact(4);
        for w in &mut chunks {
            let d = c ^ u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            c = CRC_TABLE[3][(d & 0xff) as usize]
                ^ CRC_TABLE[2][((d >> 8) & 0xff) as usize]
                ^ CRC_TABLE[1][((d >> 16) & 0xff) as usize]
                ^ CRC_TABLE[0][(d >> 24) as usize];
        }
        for &b in chunks.remainder() {
            c = CRC_TABLE[0][((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8);
        }
    }
    !c
}

#[derive(Clone, Copy)]
struct Header {
    width: u32,
    height: u32,
    depth: u8,
    color: u8,
}
impl Header {
    fn parse(b: &[u8]) -> Result<Self, String> {
        let &[
            w0,
            w1,
            w2,
            w3,
            h0,
            h1,
            h2,
            h3,
            depth,
            color,
            compression,
            filter,
            interlace,
        ] = b
        else {
            return Err("invalid IHDR length".into());
        };
        let (width, height) = (
            u32::from_be_bytes([w0, w1, w2, w3]),
            u32::from_be_bytes([h0, h1, h2, h3]),
        );
        if compression != 0 || filter != 0 {
            return Err("unsupported PNG compression or filter method".into());
        }
        match interlace {
            0 => {}
            1 => {
                return Err(
                    "interlaced (Adam7) PNGs are unsupported; save without interlacing".into(),
                );
            }
            _ => return Err("invalid PNG interlace method".into()),
        }
        if width == 0
            || height == 0
            || width > MAX_DIMENSION
            || height > MAX_DIMENSION
            || u64::from(width) * u64::from(height) > MAX_PIXELS
        {
            return Err(format!("PNG size {width}x{height} outside limits"));
        }
        let valid = match color {
            0 => matches!(depth, 1 | 2 | 4 | 8 | 16),
            3 => matches!(depth, 1 | 2 | 4 | 8),
            2 | 4 | 6 => matches!(depth, 8 | 16),
            _ => false,
        };
        if !valid {
            return Err(format!(
                "invalid PNG colour type {color} at bit depth {depth}"
            ));
        }
        Ok(Self {
            width,
            height,
            depth,
            color,
        })
    }
    fn channels(self) -> usize {
        match self.color {
            2 => 3,
            4 => 2,
            6 => 4,
            _ => 1,
        }
    }
    fn stride(self) -> usize {
        (self.width as usize * self.channels() * usize::from(self.depth)).div_ceil(8)
    }
    /// Bytes per complete pixel for filtering, at least one.
    fn bpp(self) -> usize {
        (self.channels() * usize::from(self.depth)).div_ceil(8)
    }
}

#[inline]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (i32::from(a), i32::from(b), i32::from(c));
    let pa = (ib - ic).abs();
    let pb = (ia - ic).abs();
    let pc = (ia + ib - 2 * ic).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Reverse the per-row filters in place; rows are a filter byte followed by `stride` bytes.
fn unfilter(raw: &mut [u8], stride: usize, bpp: usize) -> Result<(), String> {
    let row = stride + 1;
    let split = bpp.min(stride);
    let nrows = raw.len() / row;
    if nrows == 0 {
        return Ok(());
    }
    // First row (`up` is all zeros).
    {
        let (filter, line) = raw[..row].split_first_mut().expect("nonempty row");
        match *filter {
            0 | 2 => {}
            1 | 4 => {
                for i in split..stride {
                    line[i] = line[i].wrapping_add(line[i - bpp]);
                }
            }
            3 => {
                for i in split..stride {
                    line[i] = line[i].wrapping_add(line[i - bpp] >> 1);
                }
            }
            f => return Err(format!("invalid PNG filter type {f}")),
        }
    }
    for y in 1..nrows {
        let (done, rest) = raw.split_at_mut(y * row);
        let prev = &done[done.len() - stride..];
        let (filter, line) = rest[..row].split_first_mut().expect("nonempty row");
        match *filter {
            0 => {}
            1 => {
                for i in split..stride {
                    line[i] = line[i].wrapping_add(line[i - bpp]);
                }
            }
            2 => {
                for (v, &u) in line.iter_mut().zip(prev) {
                    *v = v.wrapping_add(u);
                }
            }
            3 => {
                for i in 0..split {
                    line[i] = line[i].wrapping_add(prev[i] >> 1);
                }
                for i in split..stride {
                    line[i] = line[i]
                        .wrapping_add(((u16::from(line[i - bpp]) + u16::from(prev[i])) >> 1) as u8);
                }
            }
            4 => {
                for i in 0..split {
                    line[i] = line[i].wrapping_add(prev[i]);
                }
                for i in split..stride {
                    line[i] = line[i].wrapping_add(paeth(line[i - bpp], prev[i], prev[i - bpp]));
                }
            }
            f => return Err(format!("invalid PNG filter type {f}")),
        }
    }
    Ok(())
}

#[inline]
fn sample(line: &[u8], k: usize, depth: u8) -> u16 {
    match depth {
        8 => u16::from(line[k]),
        16 => u16::from_be_bytes([line[2 * k], line[2 * k + 1]]),
        d => {
            let bit = k * usize::from(d);
            let shift = 8 - usize::from(d) - bit % 8;
            u16::from((line[bit / 8] >> shift) & ((1u8 << d) - 1))
        }
    }
}
#[inline]
fn to8(v: u16, depth: u8) -> u8 {
    match depth {
        16 => ((u32::from(v) + 128) / 257) as u8,
        8 => v as u8,
        d => (u32::from(v) * 255 / ((1 << d) - 1)) as u8,
    }
}

fn expand(h: Header, raw: &[u8], palette: &[u8], trns: Option<&[u8]>) -> Result<Vec<u8>, String> {
    let (w, stride, d) = (h.width as usize, h.stride(), h.depth);
    let mut out = Vec::with_capacity(w * h.height as usize * 4);
    let key = |i: usize| trns.map(|t| u16::from_be_bytes([t[2 * i], t[2 * i + 1]]));
    match (h.color, d) {
        (2, 8) if trns.is_none() => {
            for line in raw.chunks_exact(stride + 1).map(|r| &r[1..]) {
                for p in line.chunks_exact(3) {
                    out.extend_from_slice(&[p[0], p[1], p[2], 255]);
                }
            }
        }
        (0, 8) if trns.is_none() => {
            for line in raw.chunks_exact(stride + 1).map(|r| &r[1..]) {
                for &v in &line[..w] {
                    out.extend_from_slice(&[v, v, v, 255]);
                }
            }
        }
        (4, 8) => {
            for line in raw.chunks_exact(stride + 1).map(|r| &r[1..]) {
                for p in line.chunks_exact(2) {
                    out.extend_from_slice(&[p[0], p[0], p[0], p[1]]);
                }
            }
        }
        (3, 8) => {
            for line in raw.chunks_exact(stride + 1).map(|r| &r[1..]) {
                for &idx in &line[..w] {
                    let i = usize::from(idx);
                    let rgb = palette
                        .get(3 * i..3 * i + 3)
                        .ok_or("PNG palette index out of range")?;
                    let a = trns.and_then(|t| t.get(i).copied()).unwrap_or(255);
                    out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], a]);
                }
            }
        }
        (0, _) => {
            let k = key(0);
            for line in raw.chunks_exact(stride + 1).map(|r| &r[1..]) {
                for x in 0..w {
                    let g = sample(line, x, d);
                    let v = to8(g, d);
                    out.extend_from_slice(&[v, v, v, if k == Some(g) { 0 } else { 255 }]);
                }
            }
        }
        (2, _) => {
            let k = trns.map(|_| [key(0), key(1), key(2)]);
            for line in raw.chunks_exact(stride + 1).map(|r| &r[1..]) {
                for x in 0..w {
                    let s = [0, 1, 2].map(|c| sample(line, 3 * x + c, d));
                    let a = if k == Some(s.map(Some)) { 0 } else { 255 };
                    out.extend_from_slice(&[to8(s[0], d), to8(s[1], d), to8(s[2], d), a]);
                }
            }
        }
        (3, _) => {
            for line in raw.chunks_exact(stride + 1).map(|r| &r[1..]) {
                for x in 0..w {
                    let i = usize::from(sample(line, x, d));
                    let rgb = palette
                        .get(3 * i..3 * i + 3)
                        .ok_or("PNG palette index out of range")?;
                    let a = trns.and_then(|t| t.get(i).copied()).unwrap_or(255);
                    out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], a]);
                }
            }
        }
        (4, _) => {
            for line in raw.chunks_exact(stride + 1).map(|r| &r[1..]) {
                for x in 0..w {
                    let v = to8(sample(line, 2 * x, d), d);
                    out.extend_from_slice(&[v, v, v, to8(sample(line, 2 * x + 1, d), d)]);
                }
            }
        }
        (_, 8) => {
            for line in raw.chunks_exact(stride + 1).map(|r| &r[1..]) {
                out.extend_from_slice(&line[..4 * w]);
            }
        }
        _ => {
            for line in raw.chunks_exact(stride + 1).map(|r| &r[1..]) {
                for x in 0..w {
                    let s = [0, 1, 2, 3].map(|c| to8(sample(line, 4 * x + c, d), d));
                    out.extend_from_slice(&s);
                }
            }
        }
    }
    Ok(out)
}

pub fn decode(data: &[u8]) -> Result<Image, String> {
    let mut rest = data.strip_prefix(&SIGNATURE).ok_or("not a PNG file")?;
    let mut header: Option<Header> = None;
    let mut palette: &[u8] = &[];
    let mut trns: Option<&[u8]> = None;
    let mut single_idat: Option<&[u8]> = None;
    let mut multi_idat = Vec::new();
    // 0: before image data, 1: inside the IDAT run, 2: after it.
    let mut stage = 0;
    loop {
        if rest.len() < 12 {
            return Err("truncated PNG chunk".into());
        }
        let len = u32::from_be_bytes(rest[..4].try_into().expect("four bytes")) as usize;
        let kind: [u8; 4] = rest[4..8].try_into().expect("four bytes");
        if len > 0x7fff_ffff || rest.len() - 12 < len {
            return Err("truncated PNG chunk".into());
        }
        let body = &rest[8..8 + len];
        let crc = u32::from_be_bytes(rest[8 + len..12 + len].try_into().expect("four bytes"));
        rest = &rest[12 + len..];
        if !kind.iter().all(u8::is_ascii_alphabetic) {
            return Err("invalid PNG chunk type".into());
        }
        let name = std::str::from_utf8(&kind).expect("ascii");
        if crc32(&[&kind, body]) != crc {
            return Err(format!("CRC mismatch in PNG {name} chunk"));
        }
        if stage == 1 && &kind != b"IDAT" {
            stage = 2;
        }
        let Some(h) = header else {
            if &kind != b"IHDR" {
                return Err("first PNG chunk must be IHDR".into());
            }
            header = Some(Header::parse(body)?);
            continue;
        };
        match &kind {
            b"IHDR" => return Err("duplicate IHDR chunk".into()),
            b"PLTE" => {
                if !palette.is_empty() || trns.is_some() || stage != 0 {
                    return Err("misplaced or repeated PLTE chunk".into());
                }
                if h.color == 0 || h.color == 4 {
                    return Err("PLTE chunk in a greyscale PNG".into());
                }
                if len == 0 || len % 3 != 0 || len > 768 {
                    return Err("invalid PLTE length".into());
                }
                palette = body;
            }
            b"tRNS" => {
                let valid = match h.color {
                    0 => len == 2,
                    2 => len == 6,
                    3 => !palette.is_empty() && len <= palette.len() / 3,
                    _ => return Err("tRNS chunk in a PNG with an alpha channel".into()),
                };
                if !valid || trns.is_some() || stage != 0 {
                    return Err("invalid tRNS chunk".into());
                }
                trns = Some(body);
            }
            b"IDAT" => {
                if stage == 2 {
                    return Err("PNG IDAT chunks are not consecutive".into());
                }
                if h.color == 3 && palette.is_empty() {
                    return Err("paletted PNG without a PLTE chunk".into());
                }
                stage = 1;
                if multi_idat.is_empty() {
                    match single_idat {
                        None => single_idat = Some(body),
                        Some(first) => {
                            multi_idat.reserve(first.len() + body.len());
                            multi_idat.extend_from_slice(first);
                            multi_idat.extend_from_slice(body);
                            single_idat = None;
                        }
                    }
                } else {
                    multi_idat.extend_from_slice(body);
                }
            }
            b"IEND" => {
                if len != 0 || !rest.is_empty() {
                    return Err("invalid IEND or data after it".into());
                }
                break;
            }
            _ if kind[0].is_ascii_uppercase() => {
                return Err(format!("unsupported critical PNG chunk {name}"));
            }
            _ => {}
        }
    }
    let h = header.ok_or("missing IHDR")?;
    if stage == 0 {
        return Err("PNG has no image data".into());
    }
    let stride = h.stride();
    let expected = (stride + 1) * h.height as usize;
    let idat_bytes = single_idat.unwrap_or(&multi_idat);
    let mut raw = zlib::decompress_with_capacity(idat_bytes, expected, expected)?;
    drop(multi_idat);
    if raw.len() != expected {
        return Err("PNG image data is shorter than its size".into());
    }
    unfilter(&mut raw, stride, h.bpp())?;
    let rgba = if h.color == 6 && h.depth == 8 {
        let row = stride + 1;
        let height = h.height as usize;
        for y in 0..height {
            raw.copy_within(y * row + 1..y * row + 1 + stride, y * stride);
        }
        raw.truncate(stride * height);
        raw
    } else {
        expand(h, &raw, palette, trns)?
    };
    Ok(Image {
        width: h.width,
        height: h.height,
        rgba,
    })
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out.extend_from_slice(&crc32(&[kind, body]).to_be_bytes());
}

#[inline]
fn signed_abs(v: u8) -> u64 {
    u64::from((v as i8).unsigned_abs())
}

/// Choose the row filter with the smallest sum of absolute signed bytes.
fn filter_rows(rows: &[u8], stride: usize, bpp: usize, adaptive: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows.len() + rows.len() / stride.max(1));
    if !adaptive {
        for line in rows.chunks_exact(stride) {
            out.push(0);
            out.extend_from_slice(line);
        }
        return out;
    }

    let split = bpp.min(stride);
    let zero = vec![0u8; stride];
    let mut candidate = vec![0u8; stride];
    let mut best = vec![0u8; stride];
    for (y, line) in rows.chunks_exact(stride).enumerate() {
        let up = if y == 0 {
            &zero[..]
        } else {
            &rows[(y - 1) * stride..y * stride]
        };

        // Filter 0: None.
        let mut best_score: u64 = line.iter().map(|&v| signed_abs(v)).sum();
        let mut best_filter = 0u8;
        best.copy_from_slice(line);

        // Filter 1: Sub.
        {
            let mut score = 0u64;
            for i in 0..split {
                let v = line[i];
                candidate[i] = v;
                score += signed_abs(v);
            }
            for i in split..stride {
                let v = line[i].wrapping_sub(line[i - bpp]);
                candidate[i] = v;
                score += signed_abs(v);
            }
            if score < best_score {
                best_score = score;
                best_filter = 1;
                std::mem::swap(&mut best, &mut candidate);
            }
        }

        // Filter 2: Up.
        {
            let mut score = 0u64;
            for i in 0..stride {
                let v = line[i].wrapping_sub(up[i]);
                candidate[i] = v;
                score += signed_abs(v);
            }
            if score < best_score {
                best_score = score;
                best_filter = 2;
                std::mem::swap(&mut best, &mut candidate);
            }
        }

        // Filter 3: Average.
        {
            let mut score = 0u64;
            for i in 0..split {
                let v = line[i].wrapping_sub(up[i] >> 1);
                candidate[i] = v;
                score += signed_abs(v);
            }
            for i in split..stride {
                let avg = ((u16::from(line[i - bpp]) + u16::from(up[i])) >> 1) as u8;
                let v = line[i].wrapping_sub(avg);
                candidate[i] = v;
                score += signed_abs(v);
            }
            if score < best_score {
                best_score = score;
                best_filter = 3;
                std::mem::swap(&mut best, &mut candidate);
            }
        }

        // Filter 4: Paeth.
        if best_score > 0 {
            let mut score = 0u64;
            for i in 0..split {
                let v = line[i].wrapping_sub(up[i]);
                candidate[i] = v;
                score += signed_abs(v);
            }
            let mut aborted = false;
            for i in split..stride {
                let v = line[i].wrapping_sub(paeth(line[i - bpp], up[i], up[i - bpp]));
                candidate[i] = v;
                score += signed_abs(v);
                if score >= best_score {
                    aborted = true;
                    break;
                }
            }
            if !aborted && score < best_score {
                best_filter = 4;
                std::mem::swap(&mut best, &mut candidate);
            }
        }

        out.push(best_filter);
        out.extend_from_slice(&best);
    }
    out
}

#[inline]
fn hash_rgba(px: u32) -> usize {
    (px.wrapping_mul(0x9e37_79b9) >> 23) as usize
}

/// Encode RGBA8 losslessly, choosing greyscale, paletted (1, 2, 4 or 8 bits), grey-alpha,
/// RGB or RGBA, whichever represents the pixels exactly in the fewest bytes.
pub fn encode(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    if width == 0
        || height == 0
        || width > MAX_DIMENSION
        || height > MAX_DIMENSION
        || u64::from(width) * u64::from(height) > MAX_PIXELS
    {
        return Err(format!("PNG size {width}x{height} outside limits"));
    }
    let (w, h) = (width as usize, height as usize);
    if rgba.len() != w * h * 4 {
        return Err("RGBA buffer does not match the image size".into());
    }

    // Single-pass scan for greyscale, opacity, and up to 256 unique colours using a stack table.
    let mut grey = true;
    let mut opaque = true;
    let mut table = [u64::MAX; 512];
    let mut colours = [[0u8; 4]; 257];
    let mut n_colours = 0usize;
    for p in rgba.chunks_exact(4) {
        if grey && (p[0] != p[1] || p[1] != p[2]) {
            grey = false;
        }
        if opaque && p[3] != 255 {
            opaque = false;
        }
        if n_colours <= 256 {
            let c = [p[0], p[1], p[2], p[3]];
            let px = u32::from_le_bytes(c);
            let mut slot = hash_rgba(px);
            loop {
                let entry = table[slot];
                if entry == u64::MAX {
                    colours[n_colours] = c;
                    table[slot] = ((n_colours as u64) << 32) | u64::from(px);
                    n_colours += 1;
                    break;
                }
                if (entry as u32) == px {
                    break;
                }
                slot = (slot + 1) & 511;
            }
        } else if !grey && !opaque {
            break;
        }
    }

    let paletted = !(grey && opaque) && n_colours <= 256;
    let (color, depth, channels) = if grey && opaque {
        (0u8, 8u8, 1)
    } else if paletted {
        let depth = match n_colours {
            0..=2 => 1,
            3..=4 => 2,
            5..=16 => 4,
            _ => 8,
        };
        (3, depth, 1)
    } else if grey {
        (4, 8, 2)
    } else if opaque {
        (2, 8, 3)
    } else {
        (6, 8, 4)
    };
    let mut out = Vec::with_capacity(64 + (w * h) / 2);
    out.extend_from_slice(&SIGNATURE);
    let mut ihdr = [0u8; 13];
    ihdr[..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = depth;
    ihdr[9] = color;
    chunk(&mut out, b"IHDR", &ihdr);

    let stride = (w * channels * usize::from(depth)).div_ceil(8);
    let filtered = if paletted {
        let colours = &mut colours[..n_colours];
        // Translucent entries lead so tRNS is short.
        colours.sort_by_key(|c| c[3] == 255);
        table.fill(u64::MAX);
        let mut plte = [0u8; 768];
        for (i, &c) in colours.iter().enumerate() {
            plte[3 * i] = c[0];
            plte[3 * i + 1] = c[1];
            plte[3 * i + 2] = c[2];
            let px = u32::from_le_bytes(c);
            let mut slot = hash_rgba(px);
            while table[slot] != u64::MAX {
                slot = (slot + 1) & 511;
            }
            table[slot] = ((i as u64) << 32) | u64::from(px);
        }
        chunk(&mut out, b"PLTE", &plte[..3 * n_colours]);
        let translucent = colours.iter().take_while(|c| c[3] != 255).count();
        if translucent > 0 {
            let mut alpha = [0u8; 256];
            for (i, c) in colours[..translucent].iter().enumerate() {
                alpha[i] = c[3];
            }
            chunk(&mut out, b"tRNS", &alpha[..translucent]);
        }

        let lookup = |p: &[u8]| -> u8 {
            let px = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
            let mut slot = hash_rgba(px);
            loop {
                let entry = table[slot];
                if (entry as u32) == px && entry != u64::MAX {
                    return (entry >> 32) as u8;
                }
                slot = (slot + 1) & 511;
            }
        };

        let mut rows = Vec::with_capacity(stride * h);
        if depth == 8 {
            for p in rgba.chunks_exact(4) {
                rows.push(lookup(p));
            }
        } else {
            let per_byte = 8 / usize::from(depth);
            let mut packed = vec![0u8; stride];
            for line in rgba.chunks_exact(w * 4) {
                packed.fill(0);
                for (x, p) in line.chunks_exact(4).enumerate() {
                    let i = lookup(p);
                    let shift = 8 - usize::from(depth) * (x % per_byte + 1);
                    packed[x / per_byte] |= i << shift;
                }
                rows.extend_from_slice(&packed);
            }
        }
        filter_rows(&rows, stride, 1, false)
    } else if color == 6 {
        filter_rows(rgba, stride, 4, true)
    } else {
        let mut rows = Vec::with_capacity(stride * h);
        match color {
            0 => {
                for p in rgba.chunks_exact(4) {
                    rows.push(p[0]);
                }
            }
            4 => {
                for p in rgba.chunks_exact(4) {
                    rows.extend_from_slice(&[p[0], p[3]]);
                }
            }
            _ => {
                for p in rgba.chunks_exact(4) {
                    rows.extend_from_slice(&p[..3]);
                }
            }
        }
        filter_rows(&rows, stride, channels.max(1), true)
    };
    chunk(&mut out, b"IDAT", &zlib::compress(&filtered));
    chunk(&mut out, b"IEND", &[]);
    Ok(out)
}
