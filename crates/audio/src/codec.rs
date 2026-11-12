//! Zero-dependency WAV (8/16/24/32-bit PCM & 32-bit IEEE float) and QOA v1.0 codecs,
//! plus Hermite 4-point cubic resampling for sample-rate conversion and pitch shifting.

/// Stereo `f32` audio buffer (`frames` of `[left, right]` in nominal `-1.0..=1.0`).
#[derive(Clone, Debug, PartialEq)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub frames: Vec<[f32; 2]>,
}

/// Maximum decoded frame count (10 minutes at 192 kHz) to bound untrusted binary inputs.
pub const MAX_DECODED_FRAMES: usize = 192_000 * 600;

impl AudioBuffer {
    pub fn silence(sample_rate: u32, frame_count: usize) -> Self {
        Self {
            sample_rate: sample_rate.max(1),
            frames: vec![[0.0, 0.0]; frame_count],
        }
    }

    pub fn from_mono(sample_rate: u32, samples: &[f32]) -> Self {
        Self {
            sample_rate: sample_rate.max(1),
            frames: samples.iter().map(|&x| [x, x]).collect(),
        }
    }

    pub fn duration_ms(&self) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        (self.frames.len() as f32 * 1_000.0) / (self.sample_rate as f32)
    }

    /// Resamples / pitch-shifts via 4-point Hermite cubic interpolation.
    /// `rate_scale > 1.0` plays faster / higher pitch; `< 1.0` plays slower / lower pitch.
    pub fn resample_hermite(&self, target_sample_rate: u32, rate_scale: f32) -> Self {
        let target_sr = target_sample_rate.max(1);
        if self.frames.is_empty() {
            return Self::silence(target_sr, 0);
        }
        let scale = rate_scale.clamp(0.125, 8.0) as f64;
        let step = (self.sample_rate as f64 / target_sr as f64) * scale;
        if (step - 1.0).abs() < 1e-9 {
            return Self {
                sample_rate: target_sr,
                frames: self.frames.clone(),
            };
        }
        let out_len = ((self.frames.len() as f64) / step).ceil() as usize;
        let out_len = out_len.min(MAX_DECODED_FRAMES);
        let mut out = Vec::with_capacity(out_len);
        let n = self.frames.len() as isize;
        let sample_at = |idx: isize| -> [f32; 2] {
            let clamped = idx.clamp(0, n - 1) as usize;
            self.frames[clamped]
        };

        for i in 0..out_len {
            let pos = (i as f64) * step;
            let idx = pos.floor() as isize;
            let t = (pos - (idx as f64)) as f32;
            let y0 = sample_at(idx - 1);
            let y1 = sample_at(idx);
            let y2 = sample_at(idx + 1);
            let y3 = sample_at(idx + 2);
            out.push([
                hermite4(t, y0[0], y1[0], y2[0], y3[0]),
                hermite4(t, y0[1], y1[1], y2[1], y3[1]),
            ]);
        }
        Self {
            sample_rate: target_sr,
            frames: out,
        }
    }
}

#[inline]
fn hermite4(t: f32, y0: f32, y1: f32, y2: f32, y3: f32) -> f32 {
    let c0 = y1;
    let c1 = 0.5 * (y2 - y0);
    let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
    ((c3 * t + c2) * t + c1) * t + c0
}

// ---- WAV Codec ------------------------------------------------------------------------------

/// Decodes a RIFF/WAVE file (1 or 2+ channels; 8/16/24/32-bit PCM or 32-bit IEEE float)
/// into a stereo `AudioBuffer`. Multi-channel (>2) inputs fold channels 0 & 1 to L/R.
pub fn decode_wav(bytes: &[u8]) -> Result<AudioBuffer, String> {
    if bytes.len() < 44 {
        return Err("WAV too short (< 44 bytes)".into());
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("invalid RIFF/WAVE header".into());
    }
    let mut offset = 12usize;
    let mut format_tag: Option<u16> = None;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut data_slice: Option<&[u8]> = None;

    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_len = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        offset += 8;
        let chunk_end = offset
            .checked_add(chunk_len)
            .ok_or_else(|| "WAV chunk length overflow".to_string())?;
        if chunk_end > bytes.len() {
            return Err("WAV chunk exceeds buffer length".into());
        }
        let body = &bytes[offset..chunk_end];

        if chunk_id == b"fmt " {
            if body.len() < 16 {
                return Err("WAV fmt chunk < 16 bytes".into());
            }
            let mut tag = u16::from_le_bytes([body[0], body[1]]);
            channels = u16::from_le_bytes([body[2], body[3]]);
            sample_rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bits_per_sample = u16::from_le_bytes([body[14], body[15]]);
            // WAVE_FORMAT_EXTENSIBLE (0xFFFE): read subformat GUID's leading u16
            if tag == 0xFFFE && body.len() >= 40 {
                tag = u16::from_le_bytes([body[24], body[25]]);
            }
            format_tag = Some(tag);
        } else if chunk_id == b"data" {
            data_slice = Some(body);
            break;
        }
        // RIFF chunks are word-aligned (padded to even byte count).
        offset = chunk_end + (chunk_len & 1);
    }

    let tag = format_tag.ok_or_else(|| "WAV missing fmt chunk".to_string())?;
    let data = data_slice.ok_or_else(|| "WAV missing data chunk".to_string())?;
    if channels == 0 || channels > 16 {
        return Err(format!("unsupported WAV channel count {channels}"));
    }
    if !(8_000..=192_000).contains(&sample_rate) {
        return Err(format!("unsupported WAV sample rate {sample_rate}"));
    }
    let bytes_per_sample = match (tag, bits_per_sample) {
        (1, 8) => 1usize,
        (1, 16) => 2,
        (1, 24) => 3,
        (1, 32) | (3, 32) => 4,
        _ => {
            return Err(format!(
                "unsupported WAV format tag={tag} bits={bits_per_sample}"
            ));
        }
    };
    let ch = channels as usize;
    let frame_stride = bytes_per_sample * ch;
    let frame_count = data.len() / frame_stride;
    if frame_count > MAX_DECODED_FRAMES {
        return Err("WAV exceeds maximum decoded frame limit".into());
    }

    let mut frames = Vec::with_capacity(frame_count);
    for f_idx in 0..frame_count {
        let base = f_idx * frame_stride;
        let read_chan = |c: usize| -> f32 {
            let p = base + c * bytes_per_sample;
            match (tag, bits_per_sample) {
                (1, 8) => ((data[p] as f32) - 128.0) / 128.0,
                (1, 16) => {
                    let s = i16::from_le_bytes([data[p], data[p + 1]]);
                    (s as f32) / 32_768.0
                }
                (1, 24) => {
                    let raw = (data[p] as i32)
                        | ((data[p + 1] as i32) << 8)
                        | (((data[p + 2] as i8) as i32) << 16);
                    (raw as f32) / 8_388_608.0
                }
                (1, 32) => {
                    let s = i32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]);
                    (s as f32) / 2_147_483_648.0
                }
                (3, 32) => {
                    let s = f32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]);
                    if s.is_finite() { s } else { 0.0 }
                }
                _ => 0.0,
            }
        };
        let l = read_chan(0);
        let r = if ch >= 2 { read_chan(1) } else { l };
        frames.push([l, r]);
    }

    Ok(AudioBuffer {
        sample_rate,
        frames,
    })
}

/// Encodes a stereo `AudioBuffer` as standard 16-bit PCM RIFF/WAVE bytes.
pub fn encode_wav_pcm16(buffer: &AudioBuffer) -> Vec<u8> {
    let channels: u16 = 2;
    let bytes_per_sample: u16 = 2;
    let block_align = channels * bytes_per_sample;
    let byte_rate = buffer.sample_rate * u32::from(block_align);
    let data_bytes = (buffer.frames.len() * usize::from(block_align)) as u32;
    let mut out = Vec::with_capacity(44 + data_bytes as usize);

    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36u32.saturating_add(data_bytes)).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&buffer.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());

    for &[l, r] in &buffer.frames {
        out.extend_from_slice(&float_to_pcm16(l).to_le_bytes());
        out.extend_from_slice(&float_to_pcm16(r).to_le_bytes());
    }
    out
}

#[inline]
fn float_to_pcm16(sample: f32) -> i16 {
    if !sample.is_finite() {
        return 0;
    }
    let clamped = sample.clamp(-1.0, 1.0);
    if clamped < 0.0 {
        (clamped * 32_768.0).round().clamp(-32_768.0, 32_767.0) as i16
    } else {
        (clamped * 32_767.0).round().clamp(-32_768.0, 32_767.0) as i16
    }
}

// ---- QOA v1.0 (Quite OK Audio) Codec --------------------------------------------------------

const QOA_MAGIC: u32 = 0x716f_6166; // "qoaf"
const QOA_SLICE_LEN: usize = 20;
const QOA_SLICES_PER_FRAME: usize = 256;
const QOA_FRAME_LEN: usize = QOA_SLICES_PER_FRAME * QOA_SLICE_LEN; // 5120 samples/channel

const QOA_QUANT_TAB: [usize; 17] = [7, 7, 7, 5, 5, 3, 3, 1, 0, 0, 2, 2, 4, 4, 6, 6, 6];

const QOA_RECIPROCAL_TAB: [i32; 16] = [
    65536, 9363, 3121, 1457, 781, 475, 311, 216, 156, 117, 90, 71, 57, 47, 39, 32,
];

const QOA_DEQUANT_TAB: [[i32; 8]; 16] = [
    [1, -1, 3, -3, 5, -5, 7, -7],
    [5, -5, 18, -18, 32, -32, 49, -49],
    [16, -16, 53, -53, 95, -95, 147, -147],
    [34, -34, 113, -113, 203, -203, 315, -315],
    [63, -63, 210, -210, 378, -378, 588, -588],
    [104, -104, 345, -345, 621, -621, 966, -966],
    [158, -158, 528, -528, 950, -950, 1477, -1477],
    [228, -228, 760, -760, 1368, -1368, 2128, -2128],
    [316, -316, 1053, -1053, 1895, -1895, 2947, -2947],
    [422, -422, 1405, -1405, 2529, -2529, 3934, -3934],
    [548, -548, 1828, -1828, 3290, -3290, 5117, -5117],
    [696, -696, 2320, -2320, 4176, -4176, 6496, -6496],
    [868, -868, 2893, -2893, 5207, -5207, 8099, -8099],
    [1064, -1064, 3548, -3548, 6386, -6386, 9933, -9933],
    [1286, -1286, 4288, -4288, 7718, -7718, 12005, -12005],
    [1536, -1536, 5120, -5120, 9216, -9216, 14336, -14336],
];

#[derive(Clone, Copy, Debug)]
struct QoaLms {
    history: [i32; 4],
    weights: [i32; 4],
}

impl QoaLms {
    fn new() -> Self {
        Self {
            history: [0; 4],
            weights: [0, 0, -(1 << 13), 1 << 14],
        }
    }

    #[inline]
    fn predict(&self) -> i32 {
        let mut p: i64 = 0;
        for i in 0..4 {
            p += (self.history[i] as i64) * (self.weights[i] as i64);
        }
        (p >> 13) as i32
    }

    #[inline]
    fn update(&mut self, sample: i16, residual: i32) {
        let delta = residual >> 4;
        for i in 0..4 {
            self.weights[i] += if self.history[i] < 0 { -delta } else { delta };
        }
        self.history[0] = self.history[1];
        self.history[1] = self.history[2];
        self.history[2] = self.history[3];
        self.history[3] = i32::from(sample);
    }
}

#[inline]
fn qoa_div(v: i32, scalefactor: usize) -> i32 {
    let reciprocal = i64::from(QOA_RECIPROCAL_TAB[scalefactor]);
    let n = (((v as i64) * reciprocal + (1 << 15)) >> 16) as i32;
    let sign_v = (v > 0) as i32 - (v < 0) as i32;
    let sign_n = (n > 0) as i32 - (n < 0) as i32;
    n + sign_v - sign_n
}

/// Encodes a stereo `AudioBuffer` into standard QOA v1.0 `.qoa` binary format (~3.2 bits/sample).
pub fn encode_qoa(buffer: &AudioBuffer) -> Result<Vec<u8>, String> {
    if !(1..=16_777_215).contains(&buffer.sample_rate) {
        return Err(format!(
            "QOA sample rate {} out of 24-bit range",
            buffer.sample_rate
        ));
    }
    if buffer.frames.is_empty() {
        return Err("QOA requires at least 1 audio frame".into());
    }
    let total_samples = u32::try_from(buffer.frames.len())
        .map_err(|_| "QOA sample count exceeds u32".to_string())?;
    let channels = 2usize;

    let mut lms = [QoaLms::new(); 2];
    let mut prev_scalefactor = [0usize; 2];

    let num_frames = buffer.frames.len().div_ceil(QOA_FRAME_LEN);
    let mut out = Vec::with_capacity(8 + num_frames * (8 + 16 * channels + 256 * 8 * channels));
    out.extend_from_slice(&QOA_MAGIC.to_be_bytes());
    out.extend_from_slice(&total_samples.to_be_bytes());

    let mut frame_offset = 0usize;
    while frame_offset < buffer.frames.len() {
        let frame_len = (buffer.frames.len() - frame_offset).min(QOA_FRAME_LEN);
        let slices_per_ch = frame_len.div_ceil(QOA_SLICE_LEN);
        let frame_size = 8 + 16 * channels + slices_per_ch * 8 * channels;

        let sr = buffer.sample_rate;
        let frame_header = ((channels as u64) << 56)
            | ((u64::from(sr) & 0x00ff_ffff) << 32)
            | ((frame_len as u64) << 16)
            | (frame_size as u64);
        out.extend_from_slice(&frame_header.to_be_bytes());

        for ch_lms in &lms {
            for &h in &ch_lms.history {
                out.extend_from_slice(&(h.clamp(-32_768, 32_767) as i16).to_be_bytes());
            }
            for &w in &ch_lms.weights {
                out.extend_from_slice(&(w.clamp(-32_768, 32_767) as i16).to_be_bytes());
            }
        }

        let mut sample_index = 0usize;
        while sample_index < frame_len {
            let slice_len = (frame_len - sample_index).min(QOA_SLICE_LEN);
            for ch in 0..channels {
                let mut best_rank = u64::MAX;
                let mut best_slice = 0u64;
                let mut best_lms = lms[ch];
                let mut best_sf = 0usize;

                for sfi in 0..16 {
                    let scalefactor = (sfi + prev_scalefactor[ch]) & 15;
                    let mut candidate_lms = lms[ch];
                    let mut slice = (scalefactor as u64) << 60;
                    let mut current_rank = 0u64;

                    for si in 0..slice_len {
                        let sample =
                            float_to_pcm16(buffer.frames[frame_offset + sample_index + si][ch]);
                        let predicted = candidate_lms.predict();
                        let residual = i32::from(sample) - predicted;
                        let scaled = qoa_div(residual, scalefactor);
                        let clamped = (scaled + 8).clamp(0, 16) as usize;
                        let quantized = QOA_QUANT_TAB[clamped];
                        let dequantized = QOA_DEQUANT_TAB[scalefactor][quantized];
                        let reconstructed = (predicted + dequantized).clamp(-32_768, 32_767) as i16;

                        let error = i64::from(sample) - i64::from(reconstructed);
                        let error_sq = (error * error) as u64;

                        let weights_penalty = ((candidate_lms.weights[0]
                            * candidate_lms.weights[0]
                            + candidate_lms.weights[1] * candidate_lms.weights[1]
                            + candidate_lms.weights[2] * candidate_lms.weights[2]
                            + candidate_lms.weights[3] * candidate_lms.weights[3])
                            >> 18)
                            .saturating_sub(0x8ff)
                            .max(0) as u64;

                        current_rank = current_rank
                            .saturating_add(error_sq)
                            .saturating_add(weights_penalty * weights_penalty);
                        if current_rank > best_rank {
                            break;
                        }

                        candidate_lms.update(reconstructed, dequantized);
                        let shift = 57 - si * 3;
                        slice |= (quantized as u64) << shift;
                    }

                    if current_rank < best_rank {
                        best_rank = current_rank;
                        best_slice = slice;
                        best_lms = candidate_lms;
                        best_sf = scalefactor;
                    }
                }

                prev_scalefactor[ch] = best_sf;
                lms[ch] = best_lms;
                out.extend_from_slice(&best_slice.to_be_bytes());
            }
            sample_index += QOA_SLICE_LEN;
        }

        frame_offset += frame_len;
    }

    Ok(out)
}

/// Decodes a QOA v1.0 `.qoa` binary slice into a stereo `AudioBuffer` (mono folds to L=R).
pub fn decode_qoa(bytes: &[u8]) -> Result<AudioBuffer, String> {
    if bytes.len() < 16 {
        return Err("QOA buffer too short (< 16 bytes)".into());
    }
    let magic = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != QOA_MAGIC {
        return Err("invalid QOA magic (expected \"qoaf\")".into());
    }
    let total_samples = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if total_samples == 0 || total_samples > MAX_DECODED_FRAMES {
        return Err(format!(
            "invalid or streaming QOA sample count {total_samples}"
        ));
    }

    let mut pos = 8usize;
    let mut sample_rate_opt: Option<u32> = None;
    let mut frames_out: Vec<[f32; 2]> = Vec::with_capacity(total_samples);

    while pos < bytes.len() && frames_out.len() < total_samples {
        if pos + 8 > bytes.len() {
            return Err("truncated QOA frame header".into());
        }
        let fh = u64::from_be_bytes([
            bytes[pos],
            bytes[pos + 1],
            bytes[pos + 2],
            bytes[pos + 3],
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]);
        let channels = ((fh >> 56) & 0xff) as usize;
        let sample_rate = ((fh >> 32) & 0x00ff_ffff) as u32;
        let fsamples = ((fh >> 16) & 0xffff) as usize;
        let fsize = (fh & 0xffff) as usize;

        if !(1..=8).contains(&channels)
            || sample_rate == 0
            || fsamples == 0
            || fsamples > QOA_FRAME_LEN
        {
            return Err("invalid QOA frame descriptor".into());
        }
        if fsize < 8 + 16 * channels || pos + fsize > bytes.len() {
            return Err("QOA frame size out of bounds".into());
        }
        if let Some(sr) = sample_rate_opt {
            if sr != sample_rate {
                return Err("QOA mid-stream sample rate change unsupported".into());
            }
        } else {
            sample_rate_opt = Some(sample_rate);
        }

        let frame_end = pos + fsize;
        pos += 8;

        let mut lms = vec![QoaLms::new(); channels];
        for ch_lms in &mut lms {
            for h in &mut ch_lms.history {
                *h = i32::from(i16::from_be_bytes([bytes[pos], bytes[pos + 1]]));
                pos += 2;
            }
            for w in &mut ch_lms.weights {
                *w = i32::from(i16::from_be_bytes([bytes[pos], bytes[pos + 1]]));
                pos += 2;
            }
        }

        let mut sample_idx = 0usize;
        while sample_idx < fsamples {
            if pos + 8 * channels > frame_end {
                return Err("truncated QOA slice data".into());
            }
            let slice_len = (fsamples - sample_idx).min(QOA_SLICE_LEN);
            let mut decoded_ch = vec![[0i16; QOA_SLICE_LEN]; channels];

            for (ch, ch_samples) in decoded_ch.iter_mut().enumerate().take(channels) {
                let mut slice = u64::from_be_bytes([
                    bytes[pos],
                    bytes[pos + 1],
                    bytes[pos + 2],
                    bytes[pos + 3],
                    bytes[pos + 4],
                    bytes[pos + 5],
                    bytes[pos + 6],
                    bytes[pos + 7],
                ]);
                pos += 8;
                let scalefactor = ((slice >> 60) & 0x0f) as usize;
                slice <<= 4;
                for out_sample in ch_samples.iter_mut().take(slice_len) {
                    let qr = ((slice >> 61) & 0x07) as usize;
                    slice <<= 3;
                    let predicted = lms[ch].predict();
                    let dequantized = QOA_DEQUANT_TAB[scalefactor][qr];
                    let reconstructed = (predicted + dequantized).clamp(-32_768, 32_767) as i16;
                    lms[ch].update(reconstructed, dequantized);
                    *out_sample = reconstructed;
                }
            }

            let ch0 = &decoded_ch[0][..slice_len];
            let ch1 = if channels >= 2 {
                Some(&decoded_ch[1][..slice_len])
            } else {
                None
            };
            for (idx, &s0) in ch0.iter().enumerate() {
                if frames_out.len() >= total_samples {
                    break;
                }
                let l = f32::from(s0) / 32_768.0;
                let r = ch1.map_or(l, |c1| f32::from(c1[idx]) / 32_768.0);
                frames_out.push([l, r]);
            }
            sample_idx += QOA_SLICE_LEN;
        }
        pos = frame_end;
    }

    Ok(AudioBuffer {
        sample_rate: sample_rate_opt.unwrap_or(48_000),
        frames: frames_out,
    })
}

/// Auto-detects `.qoa` (`qoaf`) vs `.wav` (`RIFF....WAVE`) magic bytes and decodes into stereo `f32`.
pub fn decode_audio(bytes: &[u8]) -> Result<AudioBuffer, String> {
    if bytes.len() >= 4 && &bytes[0..4] == b"qoaf" {
        decode_qoa(bytes)
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        decode_wav(bytes)
    } else {
        Err("unrecognized audio container (expected WAV or QOA)".into())
    }
}
