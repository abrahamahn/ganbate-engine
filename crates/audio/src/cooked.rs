//! `GBEA` cooked binary SoundBank archives (`.gbea`).
//!
//! Little-endian, deterministic bytes mirroring `GBEW` (`ganbate-assets/src/cooked.rs`):
//! every count and byte length is bounded and checked against the remaining slice before
//! allocation, 4-byte padding bytes must be zero, and trailing bytes are rejected.
//!
//! Binary layout (Version 1):
//! - `0..4`:   magic `b"GBEA"`
//! - `4..8`:   version `1u32` (LE)
//! - `8..12`:  canonical `.soundbank.json` UTF-8 byte length (LE `u32`)
//! - `12..16`: cooked audio clip count (LE `u32`, sorted lexicographically by `asset_id`)
//! - JSON payload bytes zero-padded to a 4-byte boundary
//! - `clip_count` clip records in ascending `asset_id` order:
//!   - `id_len: u32` + UTF-8 `asset_id` zero-padded to 4 bytes
//!   - `codec_tag: u32` (`1 = QOA v1.0`, `2 = WAV PCM16`)
//!   - `sample_rate: u32`
//!   - `frame_count: u32`
//!   - `data_len: u32` + encoded clip stream zero-padded to 4 bytes

use std::collections::BTreeMap;

use crate::codec::{AudioBuffer, decode_qoa, decode_wav, encode_qoa, encode_wav_pcm16};
use crate::schema::SoundBank;

pub const GBEA_MAGIC: &[u8; 4] = b"GBEA";
pub const GBEA_VERSION: u32 = 1;
pub const MAX_GBEA_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_GBEA_CLIPS: usize = 16_384;

const CODEC_QOA: u32 = 1;
const CODEC_WAV_PCM16: u32 = 2;
const MIN_CLIP_RECORD_BYTES: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GbeaCodec {
    Qoa,
    WavPcm16,
}

#[derive(Clone, Debug)]
pub struct CookedSoundBank {
    pub bank: SoundBank,
    pub clips: BTreeMap<String, AudioBuffer>,
}

/// Returns the `.gbea` format version if `data` begins with `b"GBEA"`.
pub fn gbea_version(data: &[u8]) -> Option<u32> {
    let (magic, rest) = data.split_first_chunk::<4>()?;
    let (ver, _) = rest.split_first_chunk::<4>()?;
    (magic == GBEA_MAGIC).then(|| u32::from_le_bytes(*ver))
}

fn push_u32(out: &mut Vec<u8>, value: usize, label: &str) -> Result<(), String> {
    let v = u32::try_from(value).map_err(|_| format!("{label} exceeds u32"))?;
    out.extend_from_slice(&v.to_le_bytes());
    Ok(())
}

fn push_padded_bytes(out: &mut Vec<u8>, bytes: &[u8], label: &str) -> Result<(), String> {
    push_u32(out, bytes.len(), label)?;
    out.extend_from_slice(bytes);
    let pad = (4 - (bytes.len() & 3)) & 3;
    if pad > 0 {
        out.extend_from_slice(&[0u8; 3][..pad]);
    }
    Ok(())
}

/// Cooks a validated `SoundBank` and optional decoded audio buffers into deterministic `.gbea`
/// binary archive bytes (defaulting to compact QOA v1.0 compression).
pub fn cook_soundbank_gbea(
    bank: &SoundBank,
    sources: &BTreeMap<String, AudioBuffer>,
) -> Result<Vec<u8>, String> {
    cook_soundbank_gbea_with_codec(bank, sources, GbeaCodec::Qoa)
}

/// Cooks a `SoundBank` and audio buffers into deterministic `.gbea` bytes using the chosen codec.
pub fn cook_soundbank_gbea_with_codec(
    bank: &SoundBank,
    sources: &BTreeMap<String, AudioBuffer>,
    codec: GbeaCodec,
) -> Result<Vec<u8>, String> {
    if sources.len() > MAX_GBEA_CLIPS {
        return Err(format!(
            "clip count {} exceeds MAX_GBEA_CLIPS ({MAX_GBEA_CLIPS})",
            sources.len()
        ));
    }
    let json_text = bank.to_canonical_json()?;
    let json_bytes = json_text.as_bytes();

    // Only cook clips that correspond to declared SoundBank assets (deterministic BTreeMap order)
    let mut cooked_entries: Vec<(&str, &AudioBuffer, Vec<u8>)> = Vec::new();
    for (asset_id, buf) in sources {
        if !bank.assets.contains_key(asset_id) {
            return Err(format!(
                "cannot cook clip {asset_id:?}: not declared in soundbank assets"
            ));
        }
        let encoded = match codec {
            GbeaCodec::Qoa => encode_qoa(buf)?,
            GbeaCodec::WavPcm16 => encode_wav_pcm16(buf),
        };
        cooked_entries.push((asset_id.as_str(), buf, encoded));
    }

    let mut out = Vec::with_capacity(32 + json_bytes.len() + cooked_entries.len() * 64);
    out.extend_from_slice(GBEA_MAGIC);
    out.extend_from_slice(&GBEA_VERSION.to_le_bytes());
    push_u32(&mut out, json_bytes.len(), "soundbank json length")?;
    push_u32(&mut out, cooked_entries.len(), "clip count")?;

    out.extend_from_slice(json_bytes);
    let json_pad = (4 - (json_bytes.len() & 3)) & 3;
    if json_pad > 0 {
        out.extend_from_slice(&[0u8; 3][..json_pad]);
    }

    let codec_tag = match codec {
        GbeaCodec::Qoa => CODEC_QOA,
        GbeaCodec::WavPcm16 => CODEC_WAV_PCM16,
    };

    for (asset_id, buf, encoded) in &cooked_entries {
        push_padded_bytes(&mut out, asset_id.as_bytes(), "asset_id length")?;
        out.extend_from_slice(&codec_tag.to_le_bytes());
        out.extend_from_slice(&buf.sample_rate.to_le_bytes());
        push_u32(&mut out, buf.frames.len(), "clip frame count")?;
        push_padded_bytes(&mut out, encoded, "encoded clip length")?;
    }

    if out.len() > MAX_GBEA_BYTES {
        return Err(format!(
            "cooked GBEA size {} exceeds MAX_GBEA_BYTES ({MAX_GBEA_BYTES})",
            out.len()
        ));
    }
    Ok(out)
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn take(&mut self, len: usize, ctx: &str) -> Result<&'a [u8], String> {
        if len > self.remaining() {
            return Err(format!("{ctx}: truncated GBEA stream"));
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    fn u32(&mut self, ctx: &str) -> Result<u32, String> {
        let b = self.take(4, ctx)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn padded_bytes(&mut self, max_len: usize, ctx: &str) -> Result<&'a [u8], String> {
        let len = usize::try_from(self.u32(ctx)?).map_err(|_| format!("{ctx}: length overflow"))?;
        if len > max_len {
            return Err(format!("{ctx}: length {len} exceeds limit {max_len}"));
        }
        let payload = self.take(len, ctx)?;
        let pad = (4 - (len & 3)) & 3;
        let pad_bytes = self.take(pad, ctx)?;
        if pad_bytes.iter().any(|&b| b != 0) {
            return Err(format!("{ctx}: non-zero 4-byte alignment padding"));
        }
        Ok(payload)
    }
}

/// Decodes and validates a `.gbea` archive into its canonical `SoundBank` and decoded `AudioBuffer` map.
pub fn decode_soundbank_gbea(data: &[u8]) -> Result<CookedSoundBank, String> {
    if data.len() > MAX_GBEA_BYTES {
        return Err(format!(
            "GBEA archive size {} exceeds MAX_GBEA_BYTES ({MAX_GBEA_BYTES})",
            data.len()
        ));
    }
    let mut r = Reader { data, pos: 0 };
    let magic = r.take(4, "magic")?;
    if magic != GBEA_MAGIC {
        return Err("invalid GBEA magic header".into());
    }
    let version = r.u32("version")?;
    if version != GBEA_VERSION {
        return Err(format!("unsupported GBEA version {version}"));
    }
    let json_len =
        usize::try_from(r.u32("json_len")?).map_err(|_| "json_len overflow".to_string())?;
    let clip_count =
        usize::try_from(r.u32("clip_count")?).map_err(|_| "clip_count overflow".to_string())?;
    if clip_count > MAX_GBEA_CLIPS {
        return Err(format!(
            "GBEA clip_count {clip_count} exceeds MAX_GBEA_CLIPS ({MAX_GBEA_CLIPS})"
        ));
    }

    let json_bytes = r.take(json_len, "soundbank json")?;
    let json_pad = (4 - (json_len & 3)) & 3;
    let pad_slice = r.take(json_pad, "soundbank json padding")?;
    if pad_slice.iter().any(|&b| b != 0) {
        return Err("non-zero padding after soundbank json".into());
    }

    if clip_count.saturating_mul(MIN_CLIP_RECORD_BYTES) > r.remaining() {
        return Err("GBEA clip_count exceeds remaining archive bytes".into());
    }

    let bank = SoundBank::from_json_bytes(json_bytes)?;
    let mut clips = BTreeMap::new();
    let mut prev_id: Option<String> = None;

    for idx in 0..clip_count {
        let id_bytes = r.padded_bytes(96, &format!("clip[{idx}].asset_id"))?;
        let asset_id = std::str::from_utf8(id_bytes)
            .map_err(|_| format!("clip[{idx}]: invalid UTF-8 asset_id"))?
            .to_string();
        if let Some(prev) = &prev_id {
            if &asset_id <= prev {
                return Err(format!(
                    "clip[{idx}] asset_id {asset_id:?} is not strictly sorted after {prev:?}"
                ));
            }
        }
        if !bank.assets.contains_key(&asset_id) {
            return Err(format!(
                "clip[{idx}] references undeclared soundbank asset {asset_id:?}"
            ));
        }

        let codec_tag = r.u32(&format!("clip[{idx}].codec"))?;
        let expected_sr = r.u32(&format!("clip[{idx}].sample_rate"))?;
        let expected_frames = usize::try_from(r.u32(&format!("clip[{idx}].frame_count"))?)
            .map_err(|_| format!("clip[{idx}]: frame_count overflow"))?;
        let stream = r.padded_bytes(MAX_GBEA_BYTES, &format!("clip[{idx}].data"))?;

        let decoded = match codec_tag {
            CODEC_QOA => decode_qoa(stream)?,
            CODEC_WAV_PCM16 => decode_wav(stream)?,
            other => return Err(format!("clip[{idx}]: unknown codec tag {other}")),
        };
        if decoded.sample_rate != expected_sr || decoded.frames.len() != expected_frames {
            return Err(format!(
                "clip[{idx}] header/stream mismatch (expected {} Hz / {} frames, got {} Hz / {} frames)",
                expected_sr,
                expected_frames,
                decoded.sample_rate,
                decoded.frames.len()
            ));
        }

        prev_id = Some(asset_id.clone());
        clips.insert(asset_id, decoded);
    }

    if r.remaining() != 0 {
        return Err(format!(
            "GBEA archive has {} unexpected trailing bytes",
            r.remaining()
        ));
    }

    Ok(CookedSoundBank { bank, clips })
}
