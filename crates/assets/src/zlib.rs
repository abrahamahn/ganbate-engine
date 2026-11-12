//! zlib streams (RFC 1950) with DEFLATE (RFC 1951): a bounded decoder for untrusted input and
//! an LZ77 + Huffman encoder.

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// Transmission order of the code length code lengths.
const ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

const PRIMARY_BITS: u32 = 9;
const PRIMARY_SIZE: usize = 1 << PRIMARY_BITS;
const PRIMARY_MASK: usize = PRIMARY_SIZE - 1;
/// Worst-case secondary `u16` entries for a 286-symbol complete Huffman tree (<= 2,592).
const LIT_SUB_CAP: usize = 2688;
/// Worst-case secondary `u16` entries for a 30-symbol complete Huffman tree (<= 258).
const DIST_SUB_CAP: usize = 320;

const FIXED_LIT_LEN: [u8; 288] = {
    let mut lit = [8u8; 288];
    let mut i = 144;
    while i < 256 {
        lit[i] = 9;
        i += 1;
    }
    while i < 280 {
        lit[i] = 7;
        i += 1;
    }
    lit
};
const FIXED_DIST_LEN: [u8; 30] = [5u8; 30];

const fn reverse16(mut v: u16, len: u8) -> u16 {
    let mut out = 0u16;
    let mut i = 0;
    while i < len {
        out = (out << 1) | (v & 1);
        v >>= 1;
        i += 1;
    }
    out
}

const fn build_fixed_primary(lengths: &[u8]) -> [u16; PRIMARY_SIZE] {
    let mut counts = [0u16; 16];
    let mut i = 0;
    while i < lengths.len() {
        counts[lengths[i] as usize] += 1;
        i += 1;
    }
    counts[0] = 0;
    let mut next = [0u16; 16];
    let mut code = 0u16;
    let mut len = 1;
    while len < 16 {
        code = (code + counts[len - 1]) << 1;
        next[len] = code;
        len += 1;
    }
    let mut primary = [0u16; PRIMARY_SIZE];
    let mut sym = 0;
    while sym < lengths.len() {
        let l = lengths[sym];
        if l != 0 {
            let c = next[l as usize];
            next[l as usize] += 1;
            let rev = reverse16(c, l) as usize;
            let entry = ((l as u16) << 12) | (sym as u16);
            let step = 1usize << l;
            let mut idx = rev;
            while idx < PRIMARY_SIZE {
                primary[idx] = entry;
                idx += step;
            }
        }
        sym += 1;
    }
    primary
}

const FIXED_LIT_PRIMARY: [u16; PRIMARY_SIZE] = build_fixed_primary(&FIXED_LIT_LEN);
const FIXED_DIST_PRIMARY: [u16; PRIMARY_SIZE] = build_fixed_primary(&FIXED_DIST_LEN);

pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        let mut iter = chunk.chunks_exact(16);
        for c in &mut iter {
            b += a * 16
                + 16 * u32::from(c[0])
                + 15 * u32::from(c[1])
                + 14 * u32::from(c[2])
                + 13 * u32::from(c[3])
                + 12 * u32::from(c[4])
                + 11 * u32::from(c[5])
                + 10 * u32::from(c[6])
                + 9 * u32::from(c[7])
                + 8 * u32::from(c[8])
                + 7 * u32::from(c[9])
                + 6 * u32::from(c[10])
                + 5 * u32::from(c[11])
                + 4 * u32::from(c[12])
                + 3 * u32::from(c[13])
                + 2 * u32::from(c[14])
                + u32::from(c[15]);
            a += u32::from(c[0])
                + u32::from(c[1])
                + u32::from(c[2])
                + u32::from(c[3])
                + u32::from(c[4])
                + u32::from(c[5])
                + u32::from(c[6])
                + u32::from(c[7])
                + u32::from(c[8])
                + u32::from(c[9])
                + u32::from(c[10])
                + u32::from(c[11])
                + u32::from(c[12])
                + u32::from(c[13])
                + u32::from(c[14])
                + u32::from(c[15]);
        }
        for &x in iter.remainder() {
            a += u32::from(x);
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    count: u32,
}
impl Bits<'_> {
    #[inline]
    fn refill(&mut self) {
        if self.count <= 56 && self.pos + 8 <= self.data.len() {
            let word = u64::from_le_bytes(
                self.data[self.pos..self.pos + 8]
                    .try_into()
                    .expect("eight bytes"),
            );
            let bytes = ((64 - self.count) >> 3) as usize;
            let bits_added = (bytes as u32) << 3;
            let mask = u64::MAX >> (64 - bits_added);
            self.buf |= (word & mask) << self.count;
            self.pos += bytes;
            self.count += bits_added;
            return;
        }
        while self.count <= 56 {
            let Some(&byte) = self.data.get(self.pos) else {
                break;
            };
            self.buf |= u64::from(byte) << self.count;
            self.count += 8;
            self.pos += 1;
        }
    }
    #[inline]
    fn take(&mut self, n: u32) -> Result<u32, String> {
        if self.count < n {
            self.refill();
            if self.count < n {
                return Err("truncated deflate stream".into());
            }
        }
        let v = (self.buf & ((1u64 << n) - 1)) as u32;
        self.buf >>= n;
        self.count -= n;
        Ok(v)
    }
    /// Drop the partial byte and return buffered whole bytes to the input.
    fn align(&mut self) {
        self.pos -= (self.count / 8) as usize;
        self.buf = 0;
        self.count = 0;
    }
}

/// Two-level `u16` Huffman lookup table (`(len << 12) | sym` in primary for `len <= 9`,
/// `((9 + sub_bits) << 12) | sub_offset` for `10..=15`) with zero heap allocations.
struct HuffRef<'a> {
    primary: &'a [u16; PRIMARY_SIZE],
    sub: &'a [u16],
}

impl HuffRef<'_> {
    #[inline]
    fn decode(&self, bits: &mut Bits) -> Result<u16, String> {
        if bits.count < 15 {
            bits.refill();
        }
        let entry = self.primary[(bits.buf as usize) & PRIMARY_MASK];
        let top = u32::from(entry >> 12);
        if top == 0 {
            return if bits.count == 0 {
                Err("truncated deflate stream".into())
            } else {
                Err("invalid Huffman code".into())
            };
        }
        if top <= PRIMARY_BITS {
            if top > bits.count {
                return Err("truncated deflate stream".into());
            }
            bits.buf >>= top;
            bits.count -= top;
            return Ok(entry & 0x0fff);
        }
        let sub_bits = top - PRIMARY_BITS;
        let sub_off = usize::from(entry & 0x0fff);
        let sub_idx = ((bits.buf >> PRIMARY_BITS) as usize) & ((1usize << sub_bits) - 1);
        let sub_entry = self.sub[sub_off + sub_idx];
        let len = u32::from(sub_entry >> 12);
        if len == 0 {
            return Err("invalid Huffman code".into());
        }
        if len > bits.count {
            return Err("truncated deflate stream".into());
        }
        bits.buf >>= len;
        bits.count -= len;
        Ok(sub_entry & 0x0fff)
    }
}

/// A code may only be incomplete when it is a single one-bit code (or, for distances, empty).
fn usable(lengths: &[u8], left: i32) -> bool {
    left == 0 || lengths.iter().all(|&l| l <= 1)
}

/// Builds a two-level Huffman table into caller-owned stack arrays; returns `(left, sub_used)`.
fn build_huffman<'a>(
    lengths: &[u8],
    primary: &'a mut [u16; PRIMARY_SIZE],
    sub: &'a mut [u16],
    incomplete_err: Option<&'static str>,
) -> Result<HuffRef<'a>, String> {
    let mut counts = [0u16; 16];
    for &l in lengths {
        counts[usize::from(l)] += 1;
    }
    let mut left = 1i32;
    for &c in &counts[1..] {
        left = (left << 1) - i32::from(c);
        if left < 0 {
            return Err("over-subscribed Huffman code".into());
        }
    }
    if let Some(msg) = incomplete_err {
        if !usable(lengths, left) {
            return Err(msg.into());
        }
    } else if left != 0 {
        return Err("incomplete code length code".into());
    }

    primary.fill(0);
    counts[0] = 0;
    let mut base_next = [0u32; 16];
    let mut code = 0u32;
    for len in 1..16 {
        code = (code + u32::from(counts[len - 1])) << 1;
        base_next[len] = code;
    }

    let has_long = counts[(PRIMARY_BITS as usize + 1)..]
        .iter()
        .any(|&c| c != 0);
    let mut sub_used = 0usize;
    if has_long {
        let mut max_sub = [0u8; PRIMARY_SIZE];
        let mut next = base_next;
        for &l in lengths {
            if u32::from(l) > PRIMARY_BITS {
                let len = usize::from(l);
                let c = next[len];
                next[len] += 1;
                let rev = (c.reverse_bits() >> (32 - u32::from(l))) as usize;
                let prefix = rev & PRIMARY_MASK;
                if l > max_sub[prefix] {
                    max_sub[prefix] = l;
                }
            }
        }
        for (prefix, &ml) in max_sub.iter().enumerate() {
            if u32::from(ml) > PRIMARY_BITS {
                let sub_bits = u32::from(ml) - PRIMARY_BITS;
                let size = 1usize << sub_bits;
                if sub_used + size > sub.len() {
                    return Err("Huffman subtable overflow".into());
                }
                sub[sub_used..sub_used + size].fill(0);
                primary[prefix] = (u16::from(ml) << 12) | (sub_used as u16);
                sub_used += size;
            }
        }
    }

    let mut next = base_next;
    for (symbol, &l) in lengths.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let len = usize::from(l);
        let c = next[len];
        next[len] += 1;
        let rev = (c.reverse_bits() >> (32 - u32::from(l))) as usize;
        let entry = (u16::from(l) << 12) | (symbol as u16);
        if u32::from(l) <= PRIMARY_BITS {
            let step = 1usize << l;
            let mut idx = rev;
            while idx < PRIMARY_SIZE {
                primary[idx] = entry;
                idx += step;
            }
        } else {
            let p_entry = primary[rev & PRIMARY_MASK];
            let sub_bits = u32::from(p_entry >> 12) - PRIMARY_BITS;
            let sub_off = usize::from(p_entry & 0x0fff);
            let step = 1usize << (u32::from(l) - PRIMARY_BITS);
            let end = 1usize << sub_bits;
            let mut s_idx = rev >> PRIMARY_BITS;
            while s_idx < end {
                sub[sub_off + s_idx] = entry;
                s_idx += step;
            }
        }
    }

    Ok(HuffRef {
        primary,
        sub: &sub[..sub_used],
    })
}

fn inflate_dynamic(bits: &mut Bits, out: &mut Vec<u8>, limit: usize) -> Result<(), String> {
    let nlit = bits.take(5)? as usize + 257;
    let ndist = bits.take(5)? as usize + 1;
    let ncode = bits.take(4)? as usize + 4;
    if nlit > 286 || ndist > 30 {
        return Err("invalid dynamic block code counts".into());
    }
    let mut cl = [0u8; 19];
    for &i in &ORDER[..ncode] {
        cl[i] = bits.take(3)? as u8;
    }
    let mut cl_primary = [0u16; PRIMARY_SIZE];
    let clcode = build_huffman(&cl, &mut cl_primary, &mut [], None)?;

    let mut lengths = [0u8; 316];
    let mut i = 0;
    while i < nlit + ndist {
        let symbol = clcode.decode(bits)?;
        if symbol < 16 {
            lengths[i] = symbol as u8;
            i += 1;
            continue;
        }
        let (value, repeat) = match symbol {
            16 if i > 0 => (lengths[i - 1], 3 + bits.take(2)?),
            16 => return Err("repeated code length without a predecessor".into()),
            17 => (0, 3 + bits.take(3)?),
            _ => (0, 11 + bits.take(7)?),
        };
        let end = i + repeat as usize;
        if end > nlit + ndist {
            return Err("code lengths overrun the dynamic block header".into());
        }
        lengths[i..end].fill(value);
        i = end;
    }
    if lengths[256] == 0 {
        return Err("dynamic block has no end-of-block code".into());
    }

    let mut lit_primary = [0u16; PRIMARY_SIZE];
    let mut lit_sub = [0u16; LIT_SUB_CAP];
    let lit = build_huffman(
        &lengths[..nlit],
        &mut lit_primary,
        &mut lit_sub,
        Some("incomplete literal/length code"),
    )?;

    let mut dist_primary = [0u16; PRIMARY_SIZE];
    let mut dist_sub = [0u16; DIST_SUB_CAP];
    let dist = build_huffman(
        &lengths[nlit..nlit + ndist],
        &mut dist_primary,
        &mut dist_sub,
        Some("incomplete distance code"),
    )?;

    inflate_block(bits, out, limit, &lit, &dist)
}

fn inflate_block(
    bits: &mut Bits,
    out: &mut Vec<u8>,
    limit: usize,
    lit: &HuffRef,
    dist: &HuffRef,
) -> Result<(), String> {
    let overflow = || "decompressed data exceeds its expected size".to_string();
    loop {
        if bits.count < 48 {
            bits.refill();
        }
        let symbol = lit.decode(bits)?;
        if symbol < 256 {
            if out.len() >= limit {
                return Err(overflow());
            }
            out.push(symbol as u8);
            continue;
        }
        if symbol == 256 {
            return Ok(());
        }
        let s = usize::from(symbol - 257);
        if s >= 29 {
            return Err("invalid length symbol".into());
        }
        let lextra = u32::from(LEN_EXTRA[s]);
        let len = usize::from(LEN_BASE[s])
            + if lextra != 0 {
                bits.take(lextra)? as usize
            } else {
                0
            };
        let d = usize::from(dist.decode(bits)?);
        if d >= 30 {
            return Err("invalid distance symbol".into());
        }
        let dextra = u32::from(DIST_EXTRA[d]);
        let distance = usize::from(DIST_BASE[d])
            + if dextra != 0 {
                bits.take(dextra)? as usize
            } else {
                0
            };
        if distance > out.len() {
            return Err("distance reaches before the start of the stream".into());
        }
        if out.len() + len > limit {
            return Err(overflow());
        }
        let start = out.len() - distance;
        if distance >= len {
            out.extend_from_within(start..start + len);
        } else if distance == 1 {
            let b = out[start];
            out.resize(out.len() + len, b);
        } else {
            let mut rem = len;
            while rem > 0 {
                let chunk = rem.min(out.len() - start);
                out.extend_from_within(start..start + chunk);
                rem -= chunk;
            }
        }
    }
}

fn inflate_into(bits: &mut Bits, out: &mut Vec<u8>, limit: usize) -> Result<(), String> {
    let fixed_lit = HuffRef {
        primary: &FIXED_LIT_PRIMARY,
        sub: &[],
    };
    let fixed_dist = HuffRef {
        primary: &FIXED_DIST_PRIMARY,
        sub: &[],
    };
    loop {
        let last = bits.take(1)? == 1;
        match bits.take(2)? {
            0 => {
                bits.align();
                let header = bits
                    .data
                    .get(bits.pos..bits.pos + 4)
                    .ok_or("truncated stored block")?;
                let len = usize::from(u16::from_le_bytes([header[0], header[1]]));
                if u16::from_le_bytes([header[2], header[3]]) != !(len as u16) {
                    return Err("stored block length check failed".into());
                }
                bits.pos += 4;
                let raw = bits
                    .data
                    .get(bits.pos..bits.pos + len)
                    .ok_or("truncated stored block")?;
                if out.len() + len > limit {
                    return Err("decompressed data exceeds its expected size".into());
                }
                out.extend_from_slice(raw);
                bits.pos += len;
            }
            1 => inflate_block(bits, out, limit, &fixed_lit, &fixed_dist)?,
            2 => inflate_dynamic(bits, out, limit)?,
            _ => return Err("invalid deflate block type".into()),
        }
        if last {
            break;
        }
    }
    Ok(())
}

/// Inflate a raw DEFLATE stream with an expected capacity hint, bounded against untrusted input.
pub fn inflate_with_capacity(data: &[u8], cap: usize) -> Result<Vec<u8>, String> {
    let limit = crate::cooked::MAX_BYTES;
    let init_cap = cap.min(limit).min(data.len().saturating_mul(32));
    let mut out = Vec::with_capacity(init_cap);
    let mut bits = Bits {
        data,
        pos: 0,
        buf: 0,
        count: 0,
    };
    inflate_into(&mut bits, &mut out, limit)?;
    Ok(out)
}

/// Decompress a complete zlib stream that inflates to at most `limit` bytes, pre-sizing for `cap`
/// while bounding initial allocation against the compressed input length.
pub fn decompress_with_capacity(data: &[u8], limit: usize, cap: usize) -> Result<Vec<u8>, String> {
    let [cmf, flg, ..] = *data else {
        return Err("truncated zlib header".into());
    };
    if cmf & 0x0f != 8 || cmf >> 4 > 7 || (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err("invalid zlib header".into());
    }
    if flg & 0x20 != 0 {
        return Err("zlib preset dictionaries are unsupported".into());
    }
    let init_cap = cap.min(limit).min(data.len().saturating_mul(32));
    let mut out = Vec::with_capacity(init_cap);
    let mut bits = Bits {
        data: &data[2..],
        pos: 0,
        buf: 0,
        count: 0,
    };
    inflate_into(&mut bits, &mut out, limit)?;
    bits.align();
    match bits.data.get(bits.pos..) {
        Some(trailer) if trailer.len() == 4 => {
            if u32::from_be_bytes(trailer.try_into().expect("four bytes")) != adler32(&out) {
                return Err("Adler-32 checksum mismatch".into());
            }
            Ok(out)
        }
        Some(trailer) if trailer.len() > 4 => Err("data after the zlib stream".into()),
        _ => Err("missing Adler-32 checksum".into()),
    }
}

/// Decompress a complete zlib stream that inflates to at most `limit` bytes. Allocation is
/// bounded by both `limit` and the input size, and the Adler-32 checksum must match.
pub fn decompress(data: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    decompress_with_capacity(data, limit, data.len().saturating_mul(4))
}

// Compression.

const WINDOW: usize = 1 << 15;
const HASH_BITS: u32 = 15;
const NONE: u32 = u32::MAX;
const MAX_CHAIN: usize = 96;
const NICE_LENGTH: usize = 128;
const LAZY_LENGTH: usize = 32;
const BLOCK_TOKENS: usize = 1 << 15;

const LEN_SYM: [u8; 256] = {
    let mut lut = [0u8; 256];
    let mut s = 0usize;
    while s < 29 {
        let start = (LEN_BASE[s] - 3) as usize;
        let end = if s + 1 < 29 {
            (LEN_BASE[s + 1] - 3) as usize
        } else {
            256
        };
        let mut k = start;
        while k < end {
            lut[k] = s as u8;
            k += 1;
        }
        s += 1;
    }
    lut
};

const DIST_SYM: [u8; 512] = {
    let mut lut = [0u8; 512];
    let mut d = 1u16;
    while d <= 256 {
        let mut s = 0usize;
        while s + 1 < 16 && DIST_BASE[s + 1] <= d {
            s += 1;
        }
        lut[(d - 1) as usize] = s as u8;
        d += 1;
    }
    let mut hi = 2usize;
    while hi < 256 {
        let dist = ((hi << 7) + 1) as u16;
        let mut s = 16usize;
        while s + 1 < 30 && DIST_BASE[s + 1] <= dist {
            s += 1;
        }
        lut[256 + hi] = s as u8;
        hi += 1;
    }
    lut
};

const fn build_canonical_fixed<const N: usize>(lengths: &[u8; N]) -> [(u16, u8); N] {
    let mut counts = [0u16; 16];
    let mut i = 0;
    while i < N {
        counts[lengths[i] as usize] += 1;
        i += 1;
    }
    counts[0] = 0;
    let mut next = [0u16; 16];
    let mut code = 0u16;
    let mut len = 1;
    while len < 16 {
        code = (code + counts[len - 1]) << 1;
        next[len] = code;
        len += 1;
    }
    let mut out = [(0u16, 0u8); N];
    let mut sym = 0;
    while sym < N {
        let l = lengths[sym];
        if l != 0 {
            let c = next[l as usize];
            next[l as usize] += 1;
            out[sym] = (reverse16(c, l), l);
        }
        sym += 1;
    }
    out
}

const FIXED_LIT_CODES: [(u16, u8); 288] = build_canonical_fixed(&FIXED_LIT_LEN);
const FIXED_DIST_CODES: [(u16, u8); 30] = build_canonical_fixed(&FIXED_DIST_LEN);

#[derive(Clone, Copy)]
struct Token {
    /// Literal byte when `dist` is 0, otherwise match length.
    len: u16,
    dist: u16,
}

struct Writer {
    out: Vec<u8>,
    buf: u64,
    count: u32,
}
impl Writer {
    #[inline]
    fn put(&mut self, value: u32, n: u32) {
        self.buf |= u64::from(value) << self.count;
        self.count += n;
        if self.count >= 32 {
            self.out.extend_from_slice(&(self.buf as u32).to_le_bytes());
            self.buf >>= 32;
            self.count -= 32;
        }
    }
    /// Huffman codes are pre-reversed into LSB-first wire order.
    #[inline]
    fn code(&mut self, (rev_code, len): (u16, u8)) {
        self.put(u32::from(rev_code), u32::from(len));
    }
    fn align(&mut self) {
        while self.count >= 8 {
            self.out.push(self.buf as u8);
            self.buf >>= 8;
            self.count -= 8;
        }
        if self.count > 0 {
            self.out.push(self.buf as u8);
        }
        self.buf = 0;
        self.count = 0;
    }
}

#[inline]
fn hash(d: &[u8], i: usize) -> usize {
    let v = u32::from(d[i]) << 16 | u32::from(d[i + 1]) << 8 | u32::from(d[i + 2]);
    (v.wrapping_mul(2_654_435_761) >> (32 - HASH_BITS)) as usize
}

#[inline]
fn match_len(a: &[u8], b: &[u8], max: usize) -> usize {
    let mut k = 0;
    while k + 8 <= max {
        let va = u64::from_le_bytes(a[k..k + 8].try_into().expect("eight bytes"));
        let vb = u64::from_le_bytes(b[k..k + 8].try_into().expect("eight bytes"));
        let diff = va ^ vb;
        if diff != 0 {
            return k + (diff.trailing_zeros() >> 3) as usize;
        }
        k += 8;
    }
    while k < max && a[k] == b[k] {
        k += 1;
    }
    k
}

struct Matcher<'a> {
    data: &'a [u8],
    head: Vec<u32>,
    prev: Vec<u32>,
}
impl Matcher<'_> {
    #[inline]
    fn insert(&mut self, i: usize) {
        if i + 3 <= self.data.len() {
            let h = hash(self.data, i);
            self.prev[i % WINDOW] = self.head[h];
            self.head[h] = i as u32;
        }
    }
    /// Longest earlier match at `i` within the window, searched before `i` is inserted.
    fn find(&self, i: usize) -> (usize, usize) {
        let d = self.data;
        let max = (d.len() - i).min(258);
        if max < 3 {
            return (0, 0);
        }
        let (mut best, mut dist) = (2, 0);
        let di = &d[i..];
        let b0 = di[0];
        let b1 = di[1];
        let mut b_best = di[best];
        let nice = NICE_LENGTH.min(max);
        let mut c = self.head[hash(d, i)];
        for _ in 0..MAX_CHAIN {
            if c == NONE || i - c as usize >= WINDOW {
                break;
            }
            let c0 = c as usize;
            if d[c0 + best] == b_best && d[c0] == b0 && d[c0 + 1] == b1 {
                let len = match_len(&d[c0..], di, max);
                if len > best {
                    (best, dist) = (len, i - c0);
                    if len >= nice {
                        break;
                    }
                    b_best = di[best];
                }
            }
            let next = self.prev[c0 % WINDOW];
            if next == NONE || next >= c {
                break;
            }
            c = next;
        }
        if best >= 3 { (best, dist) } else { (0, 0) }
    }
}

fn tokens(data: &[u8]) -> Vec<Token> {
    let mut m = Matcher {
        data,
        head: vec![NONE; 1 << HASH_BITS],
        prev: vec![NONE; WINDOW],
    };
    let mut out = Vec::with_capacity(data.len() / 2);
    let literal = |b: u8| Token {
        len: u16::from(b),
        dist: 0,
    };
    // One-step lazy evaluation: a match is deferred when the next position has a longer one.
    let (mut pending, mut prev) = (false, (0, 0));
    let mut i = 0;
    while i < data.len() {
        let cur = if prev.0 >= LAZY_LENGTH {
            (0, 0)
        } else {
            m.find(i)
        };
        m.insert(i);
        if prev.0 >= 3 && cur.0 <= prev.0 {
            out.push(Token {
                len: prev.0 as u16,
                dist: prev.1 as u16,
            });
            let end = i - 1 + prev.0;
            for j in i + 1..end {
                m.insert(j);
            }
            i = end;
            pending = false;
            prev = (0, 0);
        } else {
            if pending {
                out.push(literal(data[i - 1]));
            }
            pending = true;
            prev = cur;
            i += 1;
        }
    }
    if pending {
        out.push(literal(data[data.len() - 1]));
    }
    out
}

/// Huffman code lengths no longer than `max`: frequencies are flattened until the tree fits.
fn code_lengths(freqs: &[u32], max: u8, lengths: &mut [u8]) {
    let mut f = [0u64; 288];
    for (dst, &src) in f[..freqs.len()].iter_mut().zip(freqs) {
        *dst = u64::from(src);
    }
    let f = &mut f[..freqs.len()];
    let mut used = [0usize; 288];
    let mut parent = [usize::MAX; 576];
    let mut depth = [0u8; 576];
    let mut heap = std::collections::BinaryHeap::with_capacity(freqs.len());
    loop {
        let mut ulen = 0usize;
        for (s, &w) in f.iter().enumerate() {
            if w > 0 {
                used[ulen] = s;
                ulen += 1;
            }
        }
        lengths.fill(0);
        if ulen == 0 {
            return;
        }
        if ulen == 1 {
            lengths[used[0]] = 1;
            return;
        }
        // Nodes: leaves first, then internal nodes in creation order.
        parent[..2 * ulen].fill(usize::MAX);
        heap.clear();
        for (k, &s) in used[..ulen].iter().enumerate() {
            heap.push(std::cmp::Reverse((f[s], k)));
        }
        let mut next = ulen;
        while heap.len() > 1 {
            let std::cmp::Reverse((wa, a)) = heap.pop().expect("two nodes");
            let std::cmp::Reverse((wb, b)) = heap.pop().expect("two nodes");
            parent[a] = next;
            parent[b] = next;
            heap.push(std::cmp::Reverse((wa + wb, next)));
            next += 1;
        }
        depth[..next].fill(0);
        for node in (0..next.saturating_sub(1)).rev() {
            depth[node] = depth[parent[node]].saturating_add(1);
        }
        let mut fits = true;
        for (k, &s) in used[..ulen].iter().enumerate() {
            let d = depth[k];
            lengths[s] = d;
            if d > max {
                fits = false;
            }
        }
        if fits {
            return;
        }
        for x in f.iter_mut() {
            if *x > 0 {
                *x = (*x >> 1).max(1);
            }
        }
    }
}

fn canonical(lengths: &[u8], out: &mut [(u16, u8)]) {
    let mut counts = [0u16; 16];
    for &l in lengths {
        counts[usize::from(l)] += 1;
    }
    counts[0] = 0;
    let mut next = [0u16; 16];
    let mut code = 0u16;
    for len in 1..16 {
        code = (code + counts[len - 1]) << 1;
        next[len] = code;
    }
    for (slot, &l) in out[..lengths.len()].iter_mut().zip(lengths) {
        if l == 0 {
            *slot = (0, 0);
        } else {
            let c = next[usize::from(l)];
            next[usize::from(l)] += 1;
            let rev = (u32::from(c).reverse_bits() >> (32 - u32::from(l))) as u16;
            *slot = (rev, l);
        }
    }
}

/// Run-length coded code lengths: (symbol, extra bits value, extra bit count).
fn rle(lengths: &[u8]) -> Vec<(u8, u8, u8)> {
    let mut out = Vec::with_capacity(lengths.len());
    let mut i = 0;
    while i < lengths.len() {
        let v = lengths[i];
        let mut run = lengths[i..].iter().take_while(|&&x| x == v).count();
        i += run;
        if v == 0 {
            while run >= 11 {
                let r = run.min(138);
                out.push((18, (r - 11) as u8, 7));
                run -= r;
            }
            if run >= 3 {
                out.push((17, (run - 3) as u8, 3));
                run = 0;
            }
        } else {
            out.push((v, 0, 0));
            run -= 1;
            while run >= 3 {
                let r = run.min(6);
                out.push((16, (r - 3) as u8, 2));
                run -= r;
            }
        }
        out.extend(std::iter::repeat_n((v, 0, 0), run));
    }
    out
}

/// At least two used symbols make every Huffman code complete.
fn ensure_two(freqs: &mut [u32]) {
    let mut used = freqs.iter().filter(|&&x| x > 0).count();
    for f in freqs.iter_mut() {
        if used >= 2 {
            break;
        }
        if *f == 0 {
            *f = 1;
            used += 1;
        }
    }
}

#[inline]
fn len_symbol(len: u16) -> usize {
    usize::from(LEN_SYM[(len - 3) as usize])
}
#[inline]
fn dist_symbol(dist: u16) -> usize {
    let d1 = (dist - 1) as usize;
    usize::from(if d1 < 256 {
        DIST_SYM[d1]
    } else {
        DIST_SYM[256 + (d1 >> 7)]
    })
}

fn write_tokens(w: &mut Writer, tokens: &[Token], lit: &[(u16, u8)], dist: &[(u16, u8)]) {
    for t in tokens {
        if t.dist == 0 {
            w.code(lit[usize::from(t.len)]);
            continue;
        }
        let s = len_symbol(t.len);
        w.code(lit[257 + s]);
        let le = u32::from(LEN_EXTRA[s]);
        if le != 0 {
            w.put(u32::from(t.len - LEN_BASE[s]), le);
        }
        let d = dist_symbol(t.dist);
        w.code(dist[d]);
        let de = u32::from(DIST_EXTRA[d]);
        if de != 0 {
            w.put(u32::from(t.dist - DIST_BASE[d]), de);
        }
    }
    w.code(lit[256]);
}

/// Emit one block as stored, fixed or dynamic Huffman, whichever is smallest.
fn write_block(w: &mut Writer, tokens: &[Token], raw: &[u8], last: bool) {
    let mut lf = [0u32; 286];
    let mut df = [0u32; 30];
    let mut extra = 0u64;
    for t in tokens {
        if t.dist == 0 {
            lf[usize::from(t.len)] += 1;
        } else {
            let (s, d) = (len_symbol(t.len), dist_symbol(t.dist));
            lf[257 + s] += 1;
            df[d] += 1;
            extra += u64::from(LEN_EXTRA[s] + DIST_EXTRA[d]);
        }
    }
    lf[256] = 1;
    let cost = |f: &[u32], l: &[u8]| -> u64 {
        f.iter()
            .zip(l)
            .map(|(&f, &l)| u64::from(f) * u64::from(l))
            .sum()
    };
    let fixed_cost = 3 + cost(&lf, &FIXED_LIT_LEN) + cost(&df, &FIXED_DIST_LEN) + extra;

    let (mut lf2, mut df2) = (lf, df);
    ensure_two(&mut lf2);
    ensure_two(&mut df2);
    let mut ll = [0u8; 286];
    let mut dl = [0u8; 30];
    code_lengths(&lf2, 15, &mut ll);
    code_lengths(&df2, 15, &mut dl);
    let nlit = ll
        .iter()
        .rposition(|&l| l != 0)
        .map_or(257, |p| (p + 1).max(257));
    let ndist = dl.iter().rposition(|&l| l != 0).map_or(1, |p| p + 1);
    let mut combined = [0u8; 316];
    combined[..nlit].copy_from_slice(&ll[..nlit]);
    combined[nlit..nlit + ndist].copy_from_slice(&dl[..ndist]);
    let runs = rle(&combined[..nlit + ndist]);
    let mut cf = [0u32; 19];
    for &(sym, _, _) in &runs {
        cf[usize::from(sym)] += 1;
    }
    ensure_two(&mut cf);
    let mut cl = [0u8; 19];
    code_lengths(&cf, 7, &mut cl);
    let ncl = ORDER
        .iter()
        .rposition(|&i| cl[i] != 0)
        .map_or(4, |p| (p + 1).max(4));
    let dynamic_cost = 17
        + 3 * ncl as u64
        + cost(&cf, &cl)
        + runs.iter().map(|r| u64::from(r.2)).sum::<u64>()
        + cost(&lf, &ll)
        + cost(&df, &dl)
        + extra;
    let stored_cost = (raw.len() as u64 + 5 * (raw.len() as u64 / 65535 + 1)) * 8 + 10;

    if stored_cost <= fixed_cost.min(dynamic_cost) {
        let mut chunks = raw.chunks(65535).peekable();
        if chunks.peek().is_none() {
            w.put(u32::from(last), 3);
            w.align();
            w.out.extend_from_slice(&[0, 0, 0xff, 0xff]);
        }
        while let Some(chunk) = chunks.next() {
            w.put(u32::from(last && chunks.peek().is_none()), 3);
            w.align();
            let n = chunk.len() as u16;
            w.out.extend_from_slice(&n.to_le_bytes());
            w.out.extend_from_slice(&(!n).to_le_bytes());
            w.out.extend_from_slice(chunk);
        }
    } else if fixed_cost <= dynamic_cost {
        w.put(u32::from(last) | 2, 3);
        write_tokens(w, tokens, &FIXED_LIT_CODES, &FIXED_DIST_CODES);
    } else {
        w.put(u32::from(last) | 4, 3);
        w.put(nlit as u32 - 257, 5);
        w.put(ndist as u32 - 1, 5);
        w.put(ncl as u32 - 4, 4);
        for &i in &ORDER[..ncl] {
            w.put(u32::from(cl[i]), 3);
        }
        let mut cc = [(0u16, 0u8); 19];
        canonical(&cl, &mut cc);
        for &(sym, value, bits) in &runs {
            w.code(cc[usize::from(sym)]);
            if bits != 0 {
                w.put(u32::from(value), u32::from(bits));
            }
        }
        let mut lit_codes = [(0u16, 0u8); 286];
        let mut dist_codes = [(0u16, 0u8); 30];
        canonical(&ll, &mut lit_codes);
        canonical(&dl, &mut dist_codes);
        write_tokens(w, tokens, &lit_codes, &dist_codes);
    }
}

/// Compress `data` into a zlib stream with a 32 KiB window.
pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity((data.len() / 2).max(16));
    out.extend_from_slice(&[0x78, 0x9c]);
    let mut w = Writer {
        out,
        buf: 0,
        count: 0,
    };
    let all = tokens(data);
    let blocks = all.len().div_ceil(BLOCK_TOKENS).max(1);
    let mut pos = 0;
    for k in 0..blocks {
        let chunk =
            &all[(k * BLOCK_TOKENS).min(all.len())..((k + 1) * BLOCK_TOKENS).min(all.len())];
        let span: usize = chunk
            .iter()
            .map(|t| if t.dist == 0 { 1 } else { usize::from(t.len) })
            .sum();
        write_block(&mut w, chunk, &data[pos..pos + span], k + 1 == blocks);
        pos += span;
    }
    w.align();
    w.out.extend_from_slice(&adler32(data).to_be_bytes());
    w.out
}
