//! Deterministic Event-Driven Software Mixer, Round-Robin Multi-Take Selector,
//! Sample-Accurate Curved Fades, Sidechain Ducking, and Offline 48 kHz 16-bit WAV Bounce.

use std::collections::BTreeMap;

use crate::codec::{AudioBuffer, encode_wav_pcm16};
use crate::dsp::{BiquadCoeffs, DspStripProcessor, apply_stereo_pan};
use crate::meter::{ChannelMeterSnapshot, analyze_stereo_frames, db_to_linear};
use crate::schema::{
    BusRole, EventLayer, FadeCurve, MASTER_BUS_ID, RoundRobinMode, SoundBank, TakeContainer,
};
use crate::spatial::{Listener3D, SpatialEmitter3D, evaluate_spatial_3d};

/// Deterministic SplitMix64 PRNG for reproducible round-robin shuffle and humanization jitter.
#[derive(Clone, Debug)]
pub struct AudioRng {
    state: u64,
}

impl AudioRng {
    pub fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0.0, 1.0)`.
    pub fn next_unit_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Bipolar float in `[-1.0, 1.0)`.
    pub fn next_bipolar_f32(&mut self) -> f32 {
        self.next_unit_f32() * 2.0 - 1.0
    }
}

/// Stateful Round-Robin selector per Multi-Take Container (`shuffle-no-repeat` & `sequential`).
#[derive(Clone, Debug)]
pub struct ContainerSelector {
    rng: AudioRng,
    seq_cursor: BTreeMap<String, usize>,
    shuffle_deck: BTreeMap<String, Vec<usize>>,
    last_picked: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedTake {
    pub asset_id: String,
    pub pitch_offset_cents: f32,
    pub gain_offset_db: f32,
}

impl ContainerSelector {
    pub fn seeded(seed: u64) -> Self {
        Self {
            rng: AudioRng::seeded(seed),
            seq_cursor: BTreeMap::new(),
            shuffle_deck: BTreeMap::new(),
            last_picked: BTreeMap::new(),
        }
    }

    pub fn pick(&mut self, pool_id: &str, container: &TakeContainer) -> Option<SelectedTake> {
        let n = container.asset_ids.len();
        if n == 0 {
            return None;
        }
        let idx = if n == 1 {
            0
        } else {
            match container.round_robin_mode {
                RoundRobinMode::Sequential => {
                    let cur = self.seq_cursor.entry(pool_id.to_string()).or_insert(0);
                    let picked = *cur % n;
                    *cur = (*cur + 1) % n;
                    picked
                }
                RoundRobinMode::ShuffleNoRepeat => {
                    let need_refill = self
                        .shuffle_deck
                        .get(pool_id)
                        .is_none_or(|deck| deck.is_empty());
                    if need_refill {
                        let mut deck: Vec<usize> = (0..n).collect();
                        for i in (1..n).rev() {
                            let j = (self.rng.next_u64() as usize) % (i + 1);
                            deck.swap(i, j);
                        }
                        // Deck is popped from the back; prevent boundary repeat against `last_picked`
                        if let Some(&last) = self.last_picked.get(pool_id) {
                            if deck.last() == Some(&last) {
                                deck.swap(0, n - 1);
                            }
                        }
                        self.shuffle_deck.insert(pool_id.to_string(), deck);
                    }
                    let picked = self
                        .shuffle_deck
                        .get_mut(pool_id)
                        .and_then(Vec::pop)
                        .unwrap_or(0);
                    self.last_picked.insert(pool_id.to_string(), picked);
                    picked
                }
            }
        };

        let pitch_offset_cents = if container.pitch_jitter_cents > 0.0 {
            self.rng.next_bipolar_f32() * container.pitch_jitter_cents
        } else {
            0.0
        };
        let gain_offset_db = if container.gain_jitter_db > 0.0 {
            self.rng.next_bipolar_f32() * container.gain_jitter_db
        } else {
            0.0
        };

        Some(SelectedTake {
            asset_id: container.asset_ids[idx].clone(),
            pitch_offset_cents,
            gain_offset_db,
        })
    }
}

/// Evaluates normalized fade curve gain `t in 0.0..=1.0` (`Linear`, `Cosine`, `SCurve` smoothstep).
#[inline]
pub fn evaluate_fade_curve(t: f32, curve: FadeCurve) -> f32 {
    let x = t.clamp(0.0, 1.0);
    match curve {
        FadeCurve::Linear => x,
        FadeCurve::Cosine => 0.5 * (1.0 - (std::f32::consts::PI * x).cos()),
        FadeCurve::SCurve => x * x * (3.0 - 2.0 * x),
    }
}

/// Evaluates the sample-accurate fade-in / fade-out envelope gain at `local_frame` inside `total_frames`.
#[inline]
pub fn clip_fade_envelope(
    local_frame: usize,
    total_frames: usize,
    fade_in_frames: usize,
    fade_out_frames: usize,
    curve: FadeCurve,
) -> f32 {
    if total_frames == 0 || local_frame >= total_frames {
        return 0.0;
    }
    let mut g = 1.0_f32;
    if fade_in_frames > 0 && local_frame < fade_in_frames {
        let t = (local_frame as f32) / (fade_in_frames as f32);
        g *= evaluate_fade_curve(t, curve);
    }
    if fade_out_frames > 0 && local_frame + fade_out_frames > total_frames {
        let rem = total_frames.saturating_sub(local_frame + 1);
        let t = (rem as f32) / (fade_out_frames as f32);
        g *= evaluate_fade_curve(t, curve);
    }
    g
}

/// Synthesizes a deterministic genre-neutral procedural transient/foley buffer when an external
/// audio file has not been loaded into the offline asset map yet.
pub fn synthesize_fallback_take(asset_id: &str, sample_rate: u32, duration_ms: f32) -> AudioBuffer {
    let sr = sample_rate.max(8_000);
    let len =
        (((duration_ms.clamp(20.0, 10_000.0) * 0.001) * (sr as f32)).round() as usize).max(16);
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for b in asset_id.bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    let mut rng = AudioRng::seeded(hash);
    let base_freq = 160.0 + ((hash & 0xff) as f32) * 3.2;
    let mut frames = Vec::with_capacity(len);
    for i in 0..len {
        let t = (i as f32) / (sr as f32);
        let env = (-t * 14.0).exp();
        let tone = (2.0 * std::f32::consts::PI * base_freq * t).sin() * 0.55;
        let click = (-t * 95.0).exp() * rng.next_bipolar_f32() * 0.35;
        let sample = (tone + click) * env * 0.7;
        frames.push([sample, sample]);
    }
    AudioBuffer {
        sample_rate: sr,
        frames,
    }
}

#[derive(Clone, Debug)]
pub struct MixRenderReport {
    pub master: AudioBuffer,
    pub bus_meters: BTreeMap<String, ChannelMeterSnapshot>,
}

/// Renders an Event Action Sheet offline through its per-layer curved fades, per-layer DSP strips,
/// multi-take selector, project buses, sidechain ducking rules, and master DSP strip.
pub fn render_event_offline(
    bank: &SoundBank,
    event_id: &str,
    loaded_assets: &BTreeMap<String, AudioBuffer>,
    seed: u64,
    target_sample_rate: u32,
) -> Result<MixRenderReport, String> {
    let event = bank
        .events
        .get(event_id)
        .ok_or_else(|| format!("unknown event {event_id:?}"))?;
    render_scheduled_events(
        bank,
        &[(event_id, 0.0)],
        event.duration_ms,
        loaded_assets,
        seed,
        target_sample_rate,
    )
}

/// Renders a Choreography Sequence offline across all scheduled steps.
pub fn render_sequence_offline(
    bank: &SoundBank,
    sequence_id: &str,
    loaded_assets: &BTreeMap<String, AudioBuffer>,
    seed: u64,
    target_sample_rate: u32,
) -> Result<MixRenderReport, String> {
    let seq = bank
        .sequences
        .get(sequence_id)
        .ok_or_else(|| format!("unknown sequence {sequence_id:?}"))?;
    let mut triggers: Vec<(&str, f32)> = Vec::with_capacity(seq.steps.len());
    let mut total_ms = 250.0_f32;
    for st in &seq.steps {
        let ev = bank
            .events
            .get(&st.event_id)
            .ok_or_else(|| format!("sequence references unknown event {:?}", st.event_id))?;
        triggers.push((st.event_id.as_str(), st.at_ms.max(0.0)));
        total_ms = total_ms.max(st.at_ms.max(0.0) + ev.duration_ms);
    }
    render_scheduled_events(
        bank,
        &triggers,
        total_ms,
        loaded_assets,
        seed,
        target_sample_rate,
    )
}

fn render_scheduled_events(
    bank: &SoundBank,
    triggers: &[(&str, f32)],
    base_duration_ms: f32,
    loaded_assets: &BTreeMap<String, AudioBuffer>,
    seed: u64,
    target_sample_rate: u32,
) -> Result<MixRenderReport, String> {
    let sr = target_sample_rate.clamp(8_000, 192_000);
    // Include reverb/delay tail headroom (300 ms) so spatial tails decay naturally
    let render_ms = (base_duration_ms.max(50.0)) + 300.0;
    let total_frames = (((render_ms * 0.001) * (sr as f32)).ceil() as usize).max(1);

    let mut bus_buffers: BTreeMap<String, Vec<[f32; 2]>> = bank
        .buses
        .keys()
        .filter(|id| id.as_str() != MASTER_BUS_ID)
        .map(|id| (id.clone(), vec![[0.0, 0.0]; total_frames]))
        .collect();

    let mut selector = ContainerSelector::seeded(seed);
    let mut layer_rng = AudioRng::seeded(seed ^ 0x55aa_55aa_d1ce_cafe);

    for &(event_id, event_offset_ms) in triggers {
        let event = bank
            .events
            .get(event_id)
            .ok_or_else(|| format!("unknown event {event_id:?}"))?;
        let any_layer_solo = event.layers.values().any(|l| l.enabled && l.solo);

        for (_layer_id, layer) in SoundBank::ordered_layers(event) {
            if !layer.enabled || (any_layer_solo && !layer.solo) {
                continue;
            }
            let Some(bus_frames) = bus_buffers.get_mut(&layer.bus) else {
                continue;
            };
            render_layer_into_bus(
                bank,
                layer,
                event_offset_ms,
                loaded_assets,
                &mut selector,
                &mut layer_rng,
                sr,
                bus_frames,
            );
        }
    }

    // Determine bus-level solo state across channel buses
    let any_bus_solo = bank
        .buses
        .iter()
        .any(|(id, b)| id != MASTER_BUS_ID && b.role == BusRole::Channel && b.solo);

    let mut bus_meters = BTreeMap::new();
    let mut processed_buses: BTreeMap<String, Vec<[f32; 2]>> = BTreeMap::new();

    for (bus_id, bus_spec) in bank.ordered_buses() {
        if bus_id == MASTER_BUS_ID {
            continue;
        }
        let mut frames = bus_buffers
            .remove(bus_id)
            .unwrap_or_else(|| vec![[0.0, 0.0]; total_frames]);
        let audible = !bus_spec.muted && (!any_bus_solo || bus_spec.solo);
        if !audible {
            frames.fill([0.0, 0.0]);
            bus_meters.insert(bus_id.to_string(), ChannelMeterSnapshot::default());
            processed_buses.insert(bus_id.to_string(), frames);
            continue;
        }

        let mut proc = DspStripProcessor::new(&bus_spec.strip, sr);
        proc.process_buffer_in_place(&mut frames);
        let vol = bus_spec.volume.clamp(0.0, 2.0);
        for f in &mut frames {
            f[0] *= vol;
            f[1] *= vol;
        }
        let gr_db = proc.peak_gain_reduction_db();
        bus_meters.insert(bus_id.to_string(), analyze_stereo_frames(&frames, gr_db));
        processed_buses.insert(bus_id.to_string(), frames);
    }

    // Apply declarative sidechain ducking rules (`dynamics.ducking`)
    if bank.dynamics.enabled {
        for rule in bank.dynamics.ducking.values() {
            if !rule.enabled || rule.amount <= 0.0 {
                continue;
            }
            let Some(src_frames) = processed_buses.get(&rule.source_bus).cloned() else {
                continue;
            };
            let env = compute_ducking_gain_envelope(
                &src_frames,
                sr,
                rule.amount.clamp(0.0, 1.0),
                rule.attack_ms,
                rule.release_ms,
            );
            for target_id in &rule.target_buses {
                if let Some(dst) = processed_buses.get_mut(target_id) {
                    for (frame, &g) in dst.iter_mut().zip(env.iter()) {
                        frame[0] *= g;
                        frame[1] *= g;
                    }
                    let prev_gr = bus_meters
                        .get(target_id)
                        .map_or(0.0, |m| m.gain_reduction_db);
                    bus_meters.insert(target_id.clone(), analyze_stereo_frames(dst, prev_gr));
                }
            }
        }
    }

    // Sum channel buses into master bus
    let mut master_frames = vec![[0.0_f32, 0.0_f32]; total_frames];
    for frames in processed_buses.values() {
        for (dst, src) in master_frames.iter_mut().zip(frames.iter()) {
            dst[0] += src[0];
            dst[1] += src[1];
        }
    }

    if let Some(master_spec) = bank.buses.get(MASTER_BUS_ID) {
        if master_spec.muted {
            master_frames.fill([0.0, 0.0]);
            bus_meters.insert(MASTER_BUS_ID.into(), ChannelMeterSnapshot::default());
        } else {
            let mut master_proc = DspStripProcessor::new(&master_spec.strip, sr);
            master_proc.process_buffer_in_place(&mut master_frames);
            let vol = master_spec.volume.clamp(0.0, 2.0);
            for f in &mut master_frames {
                f[0] *= vol;
                f[1] *= vol;
            }
            let master_gr = master_proc.peak_gain_reduction_db();
            bus_meters.insert(
                MASTER_BUS_ID.into(),
                analyze_stereo_frames(&master_frames, master_gr),
            );
        }
    }

    Ok(MixRenderReport {
        master: AudioBuffer {
            sample_rate: sr,
            frames: master_frames,
        },
        bus_meters,
    })
}

#[allow(clippy::too_many_arguments)]
fn render_layer_into_bus(
    bank: &SoundBank,
    layer: &EventLayer,
    event_offset_ms: f32,
    loaded_assets: &BTreeMap<String, AudioBuffer>,
    selector: &mut ContainerSelector,
    layer_rng: &mut AudioRng,
    sr: u32,
    bus_frames: &mut [[f32; 2]],
) {
    let mut chosen_asset_id = layer.asset_id.clone();
    let mut pool_pitch_cents = 0.0_f32;
    let mut pool_gain_db = 0.0_f32;

    if let Some(pool_id) = &layer.pool_id {
        if let Some(container) = bank.containers.get(pool_id) {
            if let Some(sel) = selector.pick(pool_id, container) {
                if chosen_asset_id.is_none() {
                    chosen_asset_id = Some(sel.asset_id);
                }
                pool_pitch_cents = sel.pitch_offset_cents;
                pool_gain_db = sel.gain_offset_db;
            }
        }
    }

    let Some(asset_id) = chosen_asset_id else {
        return;
    };

    let layer_pitch_cents = if layer.pitch_jitter_cents > 0.0 {
        layer_rng.next_bipolar_f32() * layer.pitch_jitter_cents
    } else {
        0.0
    };
    let layer_gain_jitter = if layer.gain_jitter_db > 0.0 {
        layer_rng.next_bipolar_f32() * layer.gain_jitter_db
    } else {
        0.0
    };

    let total_semitones = layer.pitch_semitones + (pool_pitch_cents + layer_pitch_cents) / 100.0;
    let rate_scale = 2.0_f32.powf(total_semitones / 12.0);
    let clip_gain = db_to_linear(layer.gain_db + pool_gain_db + layer_gain_jitter);

    let default_ms = bank
        .assets
        .get(&asset_id)
        .map_or(layer.duration_ms, |a| a.default_duration_ms);
    let source = match loaded_assets.get(&asset_id) {
        Some(buf) => buf.resample_hermite(sr, rate_scale),
        None => synthesize_fallback_take(&asset_id, sr, default_ms.max(layer.duration_ms))
            .resample_hermite(sr, rate_scale),
    };

    let trim_frames = ((layer.trim_start_ms.max(0.0) * 0.001 * (sr as f32)).round() as usize)
        .min(source.frames.len());
    let req_clip_frames =
        ((layer.duration_ms.max(1.0) * 0.001 * (sr as f32)).round() as usize).max(1);
    let available = source.frames.len().saturating_sub(trim_frames);
    let clip_len = req_clip_frames.min(available);
    if clip_len == 0 {
        return;
    }

    let fade_in_frames =
        ((layer.fade_in_ms.max(0.0) * 0.001 * (sr as f32)).round() as usize).min(clip_len);
    let fade_out_frames =
        ((layer.fade_out_ms.max(0.0) * 0.001 * (sr as f32)).round() as usize).min(clip_len);

    // Extra tail frames for clip-level delay/reverb inserts
    let has_spatial_fx = !layer.strip.bypass
        && ((layer.strip.delay.enabled && layer.strip.delay.mix > 0.0)
            || (layer.strip.reverb.enabled && layer.strip.reverb.mix > 0.0));
    let fx_tail = if has_spatial_fx {
        ((0.25 * (sr as f32)).round() as usize).min(bus_frames.len())
    } else {
        0
    };

    let mut clip_buf = vec![[0.0_f32, 0.0_f32]; clip_len + fx_tail];
    for (i, dst_frame) in clip_buf.iter_mut().enumerate().take(clip_len) {
        let env = clip_fade_envelope(
            i,
            clip_len,
            fade_in_frames,
            fade_out_frames,
            layer.fade_curve,
        );
        let raw = source.frames[trim_frames + i];
        *dst_frame = apply_stereo_pan(
            [raw[0] * env * clip_gain, raw[1] * env * clip_gain],
            layer.pan,
        );
    }

    let mut strip_proc = DspStripProcessor::new(&layer.strip, sr);
    strip_proc.process_buffer_in_place(&mut clip_buf);

    let start_frame = (((event_offset_ms + layer.start_ms).max(0.0) * 0.001 * (sr as f32)).round()
        as usize)
        .min(bus_frames.len());
    let copy_len = clip_buf
        .len()
        .min(bus_frames.len().saturating_sub(start_frame));
    for i in 0..copy_len {
        bus_frames[start_frame + i][0] += clip_buf[i][0];
        bus_frames[start_frame + i][1] += clip_buf[i][1];
    }
}

fn compute_ducking_gain_envelope(
    source_frames: &[[f32; 2]],
    sample_rate: u32,
    amount: f32,
    attack_ms: f32,
    release_ms: f32,
) -> Vec<f32> {
    let fs = sample_rate.max(8_000) as f32;
    let atk = (-1.0 / ((attack_ms.max(0.5) * 0.001) * fs)).exp();
    let rel = (-1.0 / ((release_ms.max(1.0) * 0.001) * fs)).exp();
    let mut env = 0.0_f32;
    let mut out = Vec::with_capacity(source_frames.len());

    for &[l, r] in source_frames {
        let peak = l.abs().max(r.abs());
        let active = if peak > 0.02 {
            ((peak - 0.02) * 4.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let coeff = if active > env { atk } else { rel };
        env = coeff * env + (1.0 - coeff) * active;
        let gain = (1.0 - env * amount).clamp(0.05, 1.0);
        out.push(gain);
    }
    out
}

/// Bounces an Event Action Sheet to standard `48 kHz` 16-bit PCM Stereo RIFF/WAVE bytes.
pub fn bounce_event_to_wav16(
    bank: &SoundBank,
    event_id: &str,
    loaded_assets: &BTreeMap<String, AudioBuffer>,
    seed: u64,
) -> Result<Vec<u8>, String> {
    let rendered = render_event_offline(bank, event_id, loaded_assets, seed, 48_000)?;
    Ok(encode_wav_pcm16(&rendered.master))
}

/// Bounces a Choreography Sequence to standard `48 kHz` 16-bit PCM Stereo RIFF/WAVE bytes.
pub fn bounce_sequence_to_wav16(
    bank: &SoundBank,
    sequence_id: &str,
    loaded_assets: &BTreeMap<String, AudioBuffer>,
    seed: u64,
) -> Result<Vec<u8>, String> {
    let rendered = render_sequence_offline(bank, sequence_id, loaded_assets, seed, 48_000)?;
    Ok(encode_wav_pcm16(&rendered.master))
}

// ---- Real-Time Bounded Voice Pool & Streaming Block Mixer -----------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoiceHandle(pub u64);

#[derive(Clone, Debug)]
pub struct VoiceTriggerSpec {
    pub bus: String,
    pub priority: u8,
    pub gain_db: f32,
    pub pan: f32,
    pub pitch_semitones: f32,
    pub fade_in_ms: f32,
    pub fade_out_ms: f32,
    pub fade_curve: FadeCurve,
    pub spatial: Option<SpatialEmitter3D>,
}

impl Default for VoiceTriggerSpec {
    fn default() -> Self {
        Self {
            bus: "sfx".into(),
            priority: 128,
            gain_db: 0.0,
            pan: 0.0,
            pitch_semitones: 0.0,
            fade_in_ms: 2.0,
            fade_out_ms: 16.0,
            fade_curve: FadeCurve::Cosine,
            spatial: None,
        }
    }
}

struct ActiveVoice {
    handle: VoiceHandle,
    bus: String,
    priority: u8,
    age_frames: u64,
    cursor: usize,
    frames: Vec<[f32; 2]>,
    fade_in_frames: usize,
    fade_out_frames: usize,
    fade_curve: FadeCurve,
    base_gain: f32,
    base_pan: f32,
    spatial: Option<SpatialEmitter3D>,
    air_lpf: BiquadCoeffs,
    air_z1: [f64; 2],
    air_z2: [f64; 2],
}

/// Real-time bounded polyphonic voice allocator, 3D spatialiser, and bus/master block mixer.
pub struct VoicePool {
    sample_rate: u32,
    max_voices: usize,
    next_id: u64,
    listener: Listener3D,
    voices: Vec<ActiveVoice>,
    bus_procs: BTreeMap<String, DspStripProcessor>,
    bus_volumes: BTreeMap<String, f32>,
    bus_muted: BTreeMap<String, bool>,
    bus_solo: BTreeMap<String, bool>,
    master_proc: DspStripProcessor,
    master_volume: f32,
    master_muted: bool,
}

impl VoicePool {
    pub fn from_soundbank(bank: &SoundBank, max_voices: usize) -> Self {
        let sr = bank.sample_rate.clamp(8_000, 192_000);
        let mut bus_procs = BTreeMap::new();
        let mut bus_volumes = BTreeMap::new();
        let mut bus_muted = BTreeMap::new();
        let mut bus_solo = BTreeMap::new();
        let mut master_proc = DspStripProcessor::new(&Default::default(), sr);
        let mut master_volume = 1.0_f32;
        let mut master_muted = false;

        for (id, spec) in &bank.buses {
            if id == MASTER_BUS_ID || spec.role == BusRole::Master {
                master_proc = DspStripProcessor::new(&spec.strip, sr);
                master_volume = spec.volume.clamp(0.0, 2.0);
                master_muted = spec.muted;
            } else {
                bus_procs.insert(id.clone(), DspStripProcessor::new(&spec.strip, sr));
                bus_volumes.insert(id.clone(), spec.volume.clamp(0.0, 2.0));
                bus_muted.insert(id.clone(), spec.muted);
                bus_solo.insert(id.clone(), spec.solo);
            }
        }

        Self {
            sample_rate: sr,
            max_voices: max_voices.clamp(1, 512),
            next_id: 1,
            listener: Listener3D::default(),
            voices: Vec::with_capacity(max_voices.clamp(1, 512)),
            bus_procs,
            bus_volumes,
            bus_muted,
            bus_solo,
            master_proc,
            master_volume,
            master_muted,
        }
    }

    pub fn active_voice_count(&self) -> usize {
        self.voices.len()
    }

    pub fn set_listener(&mut self, listener: Listener3D) {
        self.listener = listener;
    }

    /// Spawns a voice in the pool, stealing the lowest-priority (or oldest) voice if full.
    /// Returns `None` only if all active voices strictly outrank `spec.priority`.
    pub fn spawn_clip(
        &mut self,
        source: &AudioBuffer,
        spec: VoiceTriggerSpec,
    ) -> Option<VoiceHandle> {
        let rate = 2.0_f32.powf(spec.pitch_semitones.clamp(-24.0, 24.0) / 12.0);
        let resampled = source.resample_hermite(self.sample_rate, rate);
        if resampled.frames.is_empty() {
            return None;
        }

        if self.voices.len() >= self.max_voices {
            // Find victim with lowest priority; break ties by oldest `age_frames`
            let victim = self
                .voices
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.priority
                        .cmp(&b.priority)
                        .then_with(|| b.age_frames.cmp(&a.age_frames))
                })
                .map(|(idx, v)| (idx, v.priority));

            match victim {
                Some((idx, victim_prio)) if victim_prio <= spec.priority => {
                    self.voices.swap_remove(idx);
                }
                _ => return None,
            }
        }

        let handle = VoiceHandle(self.next_id);
        self.next_id += 1;

        let total = resampled.frames.len();
        let fade_in_frames =
            ((spec.fade_in_ms.max(0.0) * 0.001 * (self.sample_rate as f32)).round() as usize)
                .min(total);
        let fade_out_frames = ((spec.fade_out_ms.max(0.0) * 0.001 * (self.sample_rate as f32))
            .round() as usize)
            .min(total);

        let air_lpf = match spec.spatial {
            Some(em) => {
                let sp = evaluate_spatial_3d(&self.listener, &em);
                BiquadCoeffs::lowpass(self.sample_rate as f64, sp.lowpass_cutoff_hz as f64, 0.707)
            }
            None => BiquadCoeffs::bypass(),
        };

        self.voices.push(ActiveVoice {
            handle,
            bus: spec.bus,
            priority: spec.priority,
            age_frames: 0,
            cursor: 0,
            frames: resampled.frames,
            fade_in_frames,
            fade_out_frames,
            fade_curve: spec.fade_curve,
            base_gain: db_to_linear(spec.gain_db),
            base_pan: spec.pan.clamp(-1.0, 1.0),
            spatial: spec.spatial,
            air_lpf,
            air_z1: [0.0; 2],
            air_z2: [0.0; 2],
        });

        Some(handle)
    }

    pub fn stop_voice(&mut self, handle: VoiceHandle) -> bool {
        if let Some(pos) = self.voices.iter().position(|v| v.handle == handle) {
            self.voices.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Mixes `block_frames` stereo samples across all active voices, channel bus DSP strips,
    /// and the master DSP strip, pruning finished voices automatically.
    pub fn render_block(&mut self, block_frames: usize) -> AudioBuffer {
        let n = block_frames.max(1);
        let mut bus_blocks: BTreeMap<String, Vec<[f32; 2]>> = self
            .bus_procs
            .keys()
            .map(|id| (id.clone(), vec![[0.0, 0.0]; n]))
            .collect();

        for voice in &mut self.voices {
            let (spatial_gain, spatial_pan) = match voice.spatial {
                Some(em) => {
                    let sp = evaluate_spatial_3d(&self.listener, &em);
                    voice.air_lpf = BiquadCoeffs::lowpass(
                        self.sample_rate as f64,
                        sp.lowpass_cutoff_hz as f64,
                        0.707,
                    );
                    (sp.distance_gain, sp.stereo_pan)
                }
                None => (1.0, 0.0),
            };

            let gain = voice.base_gain * spatial_gain;
            let pan = (voice.base_pan + spatial_pan).clamp(-1.0, 1.0);
            let total = voice.frames.len();
            let avail = total.saturating_sub(voice.cursor).min(n);

            if let Some(bus_buf) = bus_blocks.get_mut(&voice.bus) {
                for (i, dst) in bus_buf.iter_mut().enumerate().take(avail) {
                    let frame_idx = voice.cursor + i;
                    let env = clip_fade_envelope(
                        frame_idx,
                        total,
                        voice.fade_in_frames,
                        voice.fade_out_frames,
                        voice.fade_curve,
                    );
                    let mut s = voice.frames[frame_idx];
                    if voice.spatial.is_some() {
                        for (ch, sample) in s.iter_mut().enumerate() {
                            let x = *sample as f64;
                            let y = voice.air_lpf.b0 * x + voice.air_z1[ch];
                            voice.air_z1[ch] =
                                voice.air_lpf.b1 * x - voice.air_lpf.a1 * y + voice.air_z2[ch];
                            voice.air_z2[ch] = voice.air_lpf.b2 * x - voice.air_lpf.a2 * y;
                            *sample = y as f32;
                        }
                    }
                    let panned = apply_stereo_pan([s[0] * env * gain, s[1] * env * gain], pan);
                    dst[0] += panned[0];
                    dst[1] += panned[1];
                }
            }
            voice.cursor += avail;
            voice.age_frames += avail as u64;
        }

        self.voices.retain(|v| v.cursor < v.frames.len());

        let any_solo = self.bus_solo.values().any(|&s| s);
        let mut master = vec![[0.0_f32, 0.0_f32]; n];

        for (bus_id, mut block) in bus_blocks {
            let muted = self.bus_muted.get(&bus_id).copied().unwrap_or(false);
            let solo = self.bus_solo.get(&bus_id).copied().unwrap_or(false);
            if muted || (any_solo && !solo) {
                continue;
            }
            if let Some(proc) = self.bus_procs.get_mut(&bus_id) {
                proc.process_buffer_in_place(&mut block);
            }
            let vol = self.bus_volumes.get(&bus_id).copied().unwrap_or(1.0);
            for (dst, src) in master.iter_mut().zip(block.iter()) {
                dst[0] += src[0] * vol;
                dst[1] += src[1] * vol;
            }
        }

        if self.master_muted {
            master.fill([0.0, 0.0]);
        } else {
            self.master_proc.process_buffer_in_place(&mut master);
            for f in &mut master {
                f[0] *= self.master_volume;
                f[1] *= self.master_volume;
            }
        }

        AudioBuffer {
            sample_rate: self.sample_rate,
            frames: master,
        }
    }
}
