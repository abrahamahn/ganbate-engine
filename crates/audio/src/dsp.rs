//! Deterministic zero-dependency Stereo DSP Insert Chain:
//! - Input Gain, Stereo Balance/Equal-Power Pan, Stereo Routing + 2x2 Mid/Side Width (`0%..200%`)
//! - Digital Biquad High-Pass, Low-Pass, and 3-Band Parametric EQ + exact `20 Hz..20 kHz` response
//! - Feed-forward Stereo-Linked Dynamics Compressor with live Gain Reduction (`GR`) readback
//! - Oversampled (`1x`/`2x`/`4x`) Soft-Clipping Overdrive with post-drive tone filter
//! - Damped Feedback Stereo Delay
//! - Schroeder / Freeverb Stereo Room Reverb (pre-delay + 8 parallel damped combs + 4 allpasses)

use std::f64::consts::PI;

use crate::meter::{db_to_linear, linear_to_db};
use crate::schema::{CompressorSpec, DelaySpec, DspStrip, OverdriveSpec, ReverbSpec, StereoMode};

/// Returns `[[ll, rl], [lr, rr]]` such that `out_l = ll * l + rl * r` and `out_r = lr * l + rr * r`.
pub fn compute_mid_side_matrix(mode: StereoMode, width_pct: f32) -> [[f32; 2]; 2] {
    match mode {
        StereoMode::Mono => [[0.5, 0.5], [0.5, 0.5]],
        StereoMode::Left => [[1.0, 0.0], [1.0, 0.0]],
        StereoMode::Right => [[0.0, 1.0], [0.0, 1.0]],
        StereoMode::Stereo => {
            let w = (width_pct / 100.0).clamp(0.0, 2.0);
            let diag = 0.5 * (1.0 + w);
            let cross = 0.5 * (1.0 - w);
            [[diag, cross], [cross, diag]]
        }
    }
}

/// Applies stereo pan in `-1.0..=1.0` using balance panning (unity at center `pan = 0.0`).
#[inline]
pub fn apply_stereo_pan(frame: [f32; 2], pan: f32) -> [f32; 2] {
    let p = pan.clamp(-1.0, 1.0);
    if p < 0.0 {
        // Pan left: attenuate R and bleed a portion into L
        let k = -p;
        let r_keep = 1.0 - k;
        [frame[0] + frame[1] * k * 0.5, frame[1] * r_keep]
    } else if p > 0.0 {
        let k = p;
        let l_keep = 1.0 - k;
        [frame[0] * l_keep, frame[1] + frame[0] * k * 0.5]
    } else {
        frame
    }
}

// ---- Digital Biquad Filter & Frequency Response ---------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl BiquadCoeffs {
    pub fn bypass() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    fn normalize(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        if a0.abs() < 1e-15 {
            return Self::bypass();
        }
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    pub fn highpass(sample_rate: f64, freq_hz: f64, q: f64) -> Self {
        let f0 = freq_hz.clamp(10.0, sample_rate * 0.495);
        let qv = q.clamp(0.1, 24.0);
        let w0 = 2.0 * PI * f0 / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * qv);
        let b0 = (1.0 + cos_w0) * 0.5;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) * 0.5;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    pub fn lowpass(sample_rate: f64, freq_hz: f64, q: f64) -> Self {
        let f0 = freq_hz.clamp(10.0, sample_rate * 0.495);
        let qv = q.clamp(0.1, 24.0);
        let w0 = 2.0 * PI * f0 / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * qv);
        let b0 = (1.0 - cos_w0) * 0.5;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) * 0.5;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    pub fn peaking(sample_rate: f64, freq_hz: f64, q: f64, gain_db: f64) -> Self {
        if gain_db.abs() < 1e-6 {
            return Self::bypass();
        }
        let f0 = freq_hz.clamp(15.0, sample_rate * 0.495);
        let qv = q.clamp(0.1, 24.0);
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * f0 / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * qv);
        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    pub fn lowshelf(sample_rate: f64, freq_hz: f64, gain_db: f64) -> Self {
        if gain_db.abs() < 1e-6 {
            return Self::bypass();
        }
        let f0 = freq_hz.clamp(15.0, sample_rate * 0.495);
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * f0 / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() * 0.5 * std::f64::consts::SQRT_2;
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    pub fn highshelf(sample_rate: f64, freq_hz: f64, gain_db: f64) -> Self {
        if gain_db.abs() < 1e-6 {
            return Self::bypass();
        }
        let f0 = freq_hz.clamp(15.0, sample_rate * 0.495);
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * f0 / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() * 0.5 * std::f64::consts::SQRT_2;
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Magnitude in dB at `freq_hz` evaluated on the unit circle `z = e^{j w}`.
    pub fn magnitude_db(&self, sample_rate: f64, freq_hz: f64) -> f32 {
        let w = 2.0 * PI * freq_hz.clamp(1.0, sample_rate * 0.499) / sample_rate;
        let c1 = w.cos();
        let s1 = w.sin();
        let c2 = (2.0 * w).cos();
        let s2 = (2.0 * w).sin();

        let num_re = self.b0 + self.b1 * c1 + self.b2 * c2;
        let num_im = -(self.b1 * s1 + self.b2 * s2);
        let den_re = 1.0 + self.a1 * c1 + self.a2 * c2;
        let den_im = -(self.a1 * s1 + self.a2 * s2);

        let num_mag_sq = num_re * num_re + num_im * num_im;
        let den_mag_sq = (den_re * den_re + den_im * den_im).max(1e-24);
        (10.0 * (num_mag_sq / den_mag_sq).max(1e-24).log10()) as f32
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct StereoBiquad {
    coeffs: Option<BiquadCoeffs>,
    z1: [f64; 2],
    z2: [f64; 2],
}

impl StereoBiquad {
    fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            coeffs: Some(coeffs),
            z1: [0.0; 2],
            z2: [0.0; 2],
        }
    }

    #[inline]
    fn process(&mut self, frame: [f32; 2]) -> [f32; 2] {
        let Some(c) = self.coeffs else {
            return frame;
        };
        let mut out = [0.0_f32; 2];
        for ch in 0..2 {
            let x = f64::from(frame[ch]);
            let y = c.b0 * x + self.z1[ch];
            let z1_next = c.b1 * x - c.a1 * y + self.z2[ch];
            let z2_next = c.b2 * x - c.a2 * y;
            self.z1[ch] = if z1_next.abs() < 1e-20 { 0.0 } else { z1_next };
            self.z2[ch] = if z2_next.abs() < 1e-20 { 0.0 } else { z2_next };
            out[ch] = y as f32;
        }
        out
    }
}

/// Evaluates the combined HPF + LPF + 3-Band EQ magnitude response (in dB) of `strip` at each `freq_hz`.
pub fn evaluate_strip_response_db(
    strip: &DspStrip,
    sample_rate: u32,
    freqs_hz: &[f32],
) -> Vec<f32> {
    if strip.bypass {
        return vec![0.0; freqs_hz.len()];
    }
    let fs = f64::from(sample_rate.max(8_000));
    let mut stages = Vec::with_capacity(5);
    if strip.high_pass.enabled {
        stages.push(BiquadCoeffs::highpass(
            fs,
            f64::from(strip.high_pass.frequency_hz),
            f64::from(strip.high_pass.q),
        ));
    }
    if strip.low_pass.enabled {
        stages.push(BiquadCoeffs::lowpass(
            fs,
            f64::from(strip.low_pass.frequency_hz),
            f64::from(strip.low_pass.q),
        ));
    }
    if strip.eq.enabled {
        stages.push(BiquadCoeffs::lowshelf(
            fs,
            f64::from(strip.eq.low_freq_hz),
            f64::from(strip.eq.low_gain_db),
        ));
        stages.push(BiquadCoeffs::peaking(
            fs,
            f64::from(strip.eq.mid_freq_hz),
            f64::from(strip.eq.mid_q),
            f64::from(strip.eq.mid_gain_db),
        ));
        stages.push(BiquadCoeffs::highshelf(
            fs,
            f64::from(strip.eq.high_freq_hz),
            f64::from(strip.eq.high_gain_db),
        ));
    }

    freqs_hz
        .iter()
        .map(|&f_hz| {
            stages
                .iter()
                .map(|st| st.magnitude_db(fs, f64::from(f_hz)))
                .sum::<f32>()
        })
        .collect()
}

// ---- Feed-Forward Stereo-Linked Compressor --------------------------------------------------

struct CompressorProcessor {
    spec: CompressorSpec,
    attack_coeff: f32,
    release_coeff: f32,
    makeup_lin: f32,
    env_gr_db: f32,
    peak_gr_db: f32,
}

impl CompressorProcessor {
    fn new(spec: &CompressorSpec, sample_rate: u32) -> Self {
        let fs = sample_rate.max(8_000) as f32;
        let atk = (-1.0 / (spec.attack_sec.max(0.0001) * fs)).exp();
        let rel = (-1.0 / (spec.release_sec.max(0.001) * fs)).exp();
        Self {
            spec: spec.clone(),
            attack_coeff: atk,
            release_coeff: rel,
            makeup_lin: db_to_linear(spec.makeup_gain_db),
            env_gr_db: 0.0,
            peak_gr_db: 0.0,
        }
    }

    #[inline]
    fn compute_target_gr_db(&self, input_db: f32) -> f32 {
        let thr = self.spec.threshold_db;
        let ratio = self.spec.ratio.max(1.0);
        let knee = self.spec.knee_db.max(0.0);
        let slope = 1.0 - 1.0 / ratio;
        let over = input_db - thr;

        if knee > 0.0 && over > -0.5 * knee && over < 0.5 * knee {
            let x = over + 0.5 * knee;
            slope * (x * x) / (2.0 * knee)
        } else if over >= 0.5 * knee {
            slope * over
        } else {
            0.0
        }
    }

    #[inline]
    fn process(&mut self, frame: [f32; 2]) -> [f32; 2] {
        if !self.spec.enabled {
            return frame;
        }
        let peak = frame[0].abs().max(frame[1].abs());
        let in_db = linear_to_db(peak);
        let target_gr = self.compute_target_gr_db(in_db);
        let coeff = if target_gr > self.env_gr_db {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.env_gr_db = coeff * self.env_gr_db + (1.0 - coeff) * target_gr;
        if self.env_gr_db < 1e-4 {
            self.env_gr_db = 0.0;
        }
        self.peak_gr_db = self.peak_gr_db.max(self.env_gr_db);

        let gain = db_to_linear(-self.env_gr_db) * self.makeup_lin;
        let mix = self.spec.mix.clamp(0.0, 1.0);
        let wet = [frame[0] * gain, frame[1] * gain];
        [
            frame[0] * (1.0 - mix) + wet[0] * mix,
            frame[1] * (1.0 - mix) + wet[1] * mix,
        ]
    }
}

// ---- Oversampled Soft-Clipping Overdrive ----------------------------------------------------

/// Evaluates the normalized soft-clipping transfer curve used by the Overdrive module (`x in -1..=1`).
#[inline]
pub fn overdrive_transfer_sample(x: f32, drive: f32) -> f32 {
    let d = (drive / 100.0).clamp(0.0, 1.0);
    let k = 1.0 + d * 19.0;
    let norm = k.tanh().max(1e-6);
    // Subtle asymmetric tube-like second-harmonic bias scaled by drive
    let asym = 0.08 * d * x * x;
    ((k * x + asym).tanh() - asym.tanh()) / norm
}

struct OverdriveProcessor {
    spec: OverdriveSpec,
    factor: usize,
    out_lin: f32,
    prev_in: [f32; 2],
    tone_lpf: StereoBiquad,
}

impl OverdriveProcessor {
    fn new(spec: &OverdriveSpec, sample_rate: u32) -> Self {
        let factor = spec.oversample.factor();
        let os_sr = f64::from(sample_rate.max(8_000)) * (factor as f64);
        let cutoff = f64::from(spec.tone_hz).min(f64::from(sample_rate) * 0.46);
        let tone_lpf = StereoBiquad::new(BiquadCoeffs::lowpass(
            os_sr,
            cutoff,
            std::f64::consts::FRAC_1_SQRT_2,
        ));
        Self {
            spec: spec.clone(),
            factor,
            out_lin: db_to_linear(spec.output_gain_db),
            prev_in: [0.0; 2],
            tone_lpf,
        }
    }

    #[inline]
    fn process(&mut self, frame: [f32; 2]) -> [f32; 2] {
        if !self.spec.enabled || self.spec.mix <= 0.0 {
            self.prev_in = frame;
            return frame;
        }
        let mut acc = [0.0_f32; 2];
        let inv_f = 1.0 / (self.factor as f32);
        for sub in 0..self.factor {
            let alpha = ((sub + 1) as f32) * inv_f;
            let up = [
                self.prev_in[0] + (frame[0] - self.prev_in[0]) * alpha,
                self.prev_in[1] + (frame[1] - self.prev_in[1]) * alpha,
            ];
            let shaped = [
                overdrive_transfer_sample(up[0], self.spec.drive),
                overdrive_transfer_sample(up[1], self.spec.drive),
            ];
            let filtered = self.tone_lpf.process(shaped);
            acc[0] += filtered[0];
            acc[1] += filtered[1];
        }
        self.prev_in = frame;
        let wet = [acc[0] * inv_f * self.out_lin, acc[1] * inv_f * self.out_lin];
        let mix = self.spec.mix.clamp(0.0, 1.0);
        [
            frame[0] * (1.0 - mix) + wet[0] * mix,
            frame[1] * (1.0 - mix) + wet[1] * mix,
        ]
    }
}

// ---- Stereo Feedback Delay ------------------------------------------------------------------

struct StereoDelayProcessor {
    enabled: bool,
    mix: f32,
    feedback: f32,
    delay_samples_l: usize,
    delay_samples_r: usize,
    damp_coeff: f32,
    damp_state: [f32; 2],
    buf_l: Vec<f32>,
    buf_r: Vec<f32>,
    write_pos: usize,
}

impl StereoDelayProcessor {
    fn new(spec: &DelaySpec, sample_rate: u32) -> Self {
        let fs = sample_rate.max(8_000) as f32;
        let max_len = ((2.1 * fs) as usize).max(64);
        let base_samples = ((spec.time_ms.clamp(1.0, 2_000.0) * 0.001 * fs).round() as usize)
            .clamp(1, max_len - 2);
        let spread_r = if spec.time_ms >= 15.0 {
            (((spec.time_ms * 1.03) * 0.001 * fs).round() as usize).clamp(1, max_len - 2)
        } else {
            base_samples
        };
        let rc = 1.0 / (2.0 * std::f32::consts::PI * spec.lowpass_hz.clamp(300.0, 20_000.0));
        let dt = 1.0 / fs;
        let damp_coeff = (dt / (rc + dt)).clamp(0.01, 1.0);

        Self {
            enabled: spec.enabled && spec.mix > 0.0,
            mix: spec.mix.clamp(0.0, 1.0),
            feedback: spec.feedback.clamp(0.0, 0.95),
            delay_samples_l: base_samples,
            delay_samples_r: spread_r,
            damp_coeff,
            damp_state: [0.0; 2],
            buf_l: vec![0.0; max_len],
            buf_r: vec![0.0; max_len],
            write_pos: 0,
        }
    }

    #[inline]
    fn process(&mut self, frame: [f32; 2]) -> [f32; 2] {
        if !self.enabled {
            return frame;
        }
        let cap = self.buf_l.len();
        let read_l = (self.write_pos + cap - self.delay_samples_l) % cap;
        let read_r = (self.write_pos + cap - self.delay_samples_r) % cap;
        let dl = self.buf_l[read_l];
        let dr = self.buf_r[read_r];

        self.damp_state[0] += self.damp_coeff * (dl - self.damp_state[0]);
        self.damp_state[1] += self.damp_coeff * (dr - self.damp_state[1]);
        for s in &mut self.damp_state {
            if s.abs() < 1e-20 {
                *s = 0.0;
            }
        }

        // Subtle 15% cross-feed for natural stereo ping-pong width
        let fb_l = self.damp_state[0] * 0.85 + self.damp_state[1] * 0.15;
        let fb_r = self.damp_state[1] * 0.85 + self.damp_state[0] * 0.15;
        self.buf_l[self.write_pos] = frame[0] + fb_l * self.feedback;
        self.buf_r[self.write_pos] = frame[1] + fb_r * self.feedback;
        self.write_pos = (self.write_pos + 1) % cap;

        let mix = self.mix;
        [
            frame[0] * (1.0 - mix) + dl * mix,
            frame[1] * (1.0 - mix) + dr * mix,
        ]
    }
}

// ---- Freeverb / Schroeder Stereo Room Reverb ------------------------------------------------

const FREEVERB_COMBS_44K: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
const FREEVERB_ALLPASS_44K: [usize; 4] = [556, 441, 341, 225];
const FREEVERB_STEREO_SPREAD: usize = 23;

struct DampedComb {
    buf: Vec<f32>,
    pos: usize,
    filter_store: f32,
    feedback: f32,
    damp1: f32,
    damp2: f32,
}

impl DampedComb {
    fn new(len: usize, feedback: f32, damp: f32) -> Self {
        Self {
            buf: vec![0.0; len.max(16)],
            pos: 0,
            filter_store: 0.0,
            feedback,
            damp1: damp,
            damp2: 1.0 - damp,
        }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let out = self.buf[self.pos];
        self.filter_store = out * self.damp2 + self.filter_store * self.damp1;
        if self.filter_store.abs() < 1e-20 {
            self.filter_store = 0.0;
        }
        let next = input + self.filter_store * self.feedback;
        self.buf[self.pos] = if next.abs() < 1e-20 { 0.0 } else { next };
        self.pos = (self.pos + 1) % self.buf.len();
        out
    }
}

struct SchroederAllpass {
    buf: Vec<f32>,
    pos: usize,
}

impl SchroederAllpass {
    fn new(len: usize) -> Self {
        Self {
            buf: vec![0.0; len.max(8)],
            pos: 0,
        }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let buf_out = self.buf[self.pos];
        let out = -input + buf_out;
        let next = input + buf_out * 0.5;
        self.buf[self.pos] = if next.abs() < 1e-20 { 0.0 } else { next };
        self.pos = (self.pos + 1) % self.buf.len();
        out
    }
}

struct FreeverbProcessor {
    enabled: bool,
    mix: f32,
    pre_delay_samples: usize,
    pre_buf: Vec<f32>,
    pre_pos: usize,
    combs_l: Vec<DampedComb>,
    combs_r: Vec<DampedComb>,
    allpasses_l: Vec<SchroederAllpass>,
    allpasses_r: Vec<SchroederAllpass>,
}

impl FreeverbProcessor {
    fn new(spec: &ReverbSpec, sample_rate: u32) -> Self {
        let sr = sample_rate.max(8_000) as f32;
        let scale = sr / 44_100.0;
        let feedback = (0.70 + spec.room_size.clamp(0.0, 1.0) * 0.27).clamp(0.70, 0.97);
        let damp = (spec.damping.clamp(0.0, 1.0) * 0.65).clamp(0.0, 0.95);
        let spread = ((FREEVERB_STEREO_SPREAD as f32) * scale).round() as usize;

        let combs_l = FREEVERB_COMBS_44K
            .iter()
            .map(|&n| DampedComb::new(((n as f32) * scale).round() as usize, feedback, damp))
            .collect();
        let combs_r = FREEVERB_COMBS_44K
            .iter()
            .map(|&n| {
                DampedComb::new(
                    ((n as f32) * scale).round() as usize + spread,
                    feedback,
                    damp,
                )
            })
            .collect();
        let allpasses_l = FREEVERB_ALLPASS_44K
            .iter()
            .map(|&n| SchroederAllpass::new(((n as f32) * scale).round() as usize))
            .collect();
        let allpasses_r = FREEVERB_ALLPASS_44K
            .iter()
            .map(|&n| SchroederAllpass::new(((n as f32) * scale).round() as usize + spread))
            .collect();

        let max_pre = ((0.26 * sr) as usize).max(16);
        let pre_samples =
            ((spec.pre_delay_ms.clamp(0.0, 250.0) * 0.001 * sr).round() as usize).min(max_pre - 1);

        Self {
            enabled: spec.enabled && spec.mix > 0.0,
            mix: spec.mix.clamp(0.0, 1.0),
            pre_delay_samples: pre_samples,
            pre_buf: vec![0.0; max_pre],
            pre_pos: 0,
            combs_l,
            combs_r,
            allpasses_l,
            allpasses_r,
        }
    }

    #[inline]
    fn process(&mut self, frame: [f32; 2]) -> [f32; 2] {
        if !self.enabled {
            return frame;
        }
        let mono_in = (frame[0] + frame[1]) * 0.015;
        let delayed_in = if self.pre_delay_samples == 0 {
            mono_in
        } else {
            let cap = self.pre_buf.len();
            let read_idx = (self.pre_pos + cap - self.pre_delay_samples) % cap;
            let out = self.pre_buf[read_idx];
            self.pre_buf[self.pre_pos] = mono_in;
            self.pre_pos = (self.pre_pos + 1) % cap;
            out
        };

        let mut wet_l = 0.0_f32;
        let mut wet_r = 0.0_f32;
        for c in &mut self.combs_l {
            wet_l += c.process(delayed_in);
        }
        for c in &mut self.combs_r {
            wet_r += c.process(delayed_in);
        }
        for ap in &mut self.allpasses_l {
            wet_l = ap.process(wet_l);
        }
        for ap in &mut self.allpasses_r {
            wet_r = ap.process(wet_r);
        }

        let mix = self.mix;
        [
            frame[0] * (1.0 - mix) + wet_l * 2.4 * mix,
            frame[1] * (1.0 - mix) + wet_r * 2.4 * mix,
        ]
    }
}

// ---- Complete Stereo Insert Strip Processor -------------------------------------------------

pub struct DspStripProcessor {
    bypass: bool,
    input_gain_lin: f32,
    pan: f32,
    ms_matrix: [[f32; 2]; 2],
    hpf: StereoBiquad,
    lpf: StereoBiquad,
    eq_low: StereoBiquad,
    eq_mid: StereoBiquad,
    eq_high: StereoBiquad,
    compressor: CompressorProcessor,
    overdrive: OverdriveProcessor,
    delay: StereoDelayProcessor,
    reverb: FreeverbProcessor,
}

impl DspStripProcessor {
    pub fn new(strip: &DspStrip, sample_rate: u32) -> Self {
        let fs = f64::from(sample_rate.max(8_000));
        let hpf = if strip.high_pass.enabled {
            StereoBiquad::new(BiquadCoeffs::highpass(
                fs,
                f64::from(strip.high_pass.frequency_hz),
                f64::from(strip.high_pass.q),
            ))
        } else {
            StereoBiquad::default()
        };
        let lpf = if strip.low_pass.enabled {
            StereoBiquad::new(BiquadCoeffs::lowpass(
                fs,
                f64::from(strip.low_pass.frequency_hz),
                f64::from(strip.low_pass.q),
            ))
        } else {
            StereoBiquad::default()
        };
        let (eq_low, eq_mid, eq_high) = if strip.eq.enabled {
            (
                StereoBiquad::new(BiquadCoeffs::lowshelf(
                    fs,
                    f64::from(strip.eq.low_freq_hz),
                    f64::from(strip.eq.low_gain_db),
                )),
                StereoBiquad::new(BiquadCoeffs::peaking(
                    fs,
                    f64::from(strip.eq.mid_freq_hz),
                    f64::from(strip.eq.mid_q),
                    f64::from(strip.eq.mid_gain_db),
                )),
                StereoBiquad::new(BiquadCoeffs::highshelf(
                    fs,
                    f64::from(strip.eq.high_freq_hz),
                    f64::from(strip.eq.high_gain_db),
                )),
            )
        } else {
            (
                StereoBiquad::default(),
                StereoBiquad::default(),
                StereoBiquad::default(),
            )
        };

        Self {
            bypass: strip.bypass,
            input_gain_lin: db_to_linear(strip.input_gain_db),
            pan: strip.pan,
            ms_matrix: compute_mid_side_matrix(strip.stereo_mode, strip.stereo_width_pct),
            hpf,
            lpf,
            eq_low,
            eq_mid,
            eq_high,
            compressor: CompressorProcessor::new(&strip.compressor, sample_rate),
            overdrive: OverdriveProcessor::new(&strip.overdrive, sample_rate),
            delay: StereoDelayProcessor::new(&strip.delay, sample_rate),
            reverb: FreeverbProcessor::new(&strip.reverb, sample_rate),
        }
    }

    #[inline]
    pub fn process_frame(&mut self, input: [f32; 2]) -> [f32; 2] {
        if self.bypass {
            return input;
        }
        let mut x = [
            input[0] * self.input_gain_lin,
            input[1] * self.input_gain_lin,
        ];
        x = [
            self.ms_matrix[0][0] * x[0] + self.ms_matrix[0][1] * x[1],
            self.ms_matrix[1][0] * x[0] + self.ms_matrix[1][1] * x[1],
        ];
        x = apply_stereo_pan(x, self.pan);
        x = self.hpf.process(x);
        x = self.lpf.process(x);
        x = self.eq_low.process(x);
        x = self.eq_mid.process(x);
        x = self.eq_high.process(x);
        x = self.compressor.process(x);
        x = self.overdrive.process(x);
        x = self.delay.process(x);
        x = self.reverb.process(x);
        [
            if x[0].is_finite() { x[0] } else { 0.0 },
            if x[1].is_finite() { x[1] } else { 0.0 },
        ]
    }

    pub fn process_buffer_in_place(&mut self, frames: &mut [[f32; 2]]) {
        for frame in frames {
            *frame = self.process_frame(*frame);
        }
    }

    pub fn current_gain_reduction_db(&self) -> f32 {
        self.compressor.env_gr_db
    }

    pub fn peak_gain_reduction_db(&self) -> f32 {
        self.compressor.peak_gr_db
    }
}
