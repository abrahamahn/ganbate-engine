//! Canonical `.soundbank.json` document model, validator, and deterministic serializer.
//!
//! All buses (`master` plus project-defined channel buses), sound assets, multi-take round-robin
//! containers, contact-synced event action sheets, choreography sequences, studio mix presets,
//! and sidechain ducking rules are 100% data-driven with zero genre or game hardcoding.
//!
//! Collections serialize as keyed JSON objects (`BTreeMap<String, T>`) with explicit `order`
//! fields so Ganbate Studio's `Op::Set` and `Op::Remove` document protocol can address any bus
//! fader, DSP parameter, or timeline clip atomically without array index conflicts. On input,
//! both keyed objects and ordered arrays (`[{"id": ...}]`) are accepted.

use std::collections::BTreeMap;

pub const SOUNDBANK_KIND: &str = "soundbank";
pub const SOUNDBANK_VERSION: u64 = 1;
pub const DEFAULT_SAMPLE_RATE: u32 = 48_000;
pub const MASTER_BUS_ID: &str = "master";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BusRole {
    Channel,
    Master,
}

impl BusRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Channel => "channel",
            Self::Master => "master",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "channel" => Ok(Self::Channel),
            "master" => Ok(Self::Master),
            _ => Err(format!(
                "unknown bus role {s:?} (expected \"channel\" or \"master\")"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoMode {
    Stereo,
    Mono,
    Left,
    Right,
}

impl StereoMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stereo => "stereo",
            Self::Mono => "mono",
            Self::Left => "left",
            Self::Right => "right",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "stereo" => Ok(Self::Stereo),
            "mono" => Ok(Self::Mono),
            "left" => Ok(Self::Left),
            "right" => Ok(Self::Right),
            _ => Err(format!("unknown stereoMode {s:?}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OversampleFactor {
    None,
    X2,
    X4,
}

impl OversampleFactor {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::X2 => "2x",
            Self::X4 => "4x",
        }
    }

    pub fn factor(self) -> usize {
        match self {
            Self::None => 1,
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(Self::None),
            "2x" => Ok(Self::X2),
            "4x" => Ok(Self::X4),
            _ => Err(format!("unknown oversample mode {s:?}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FadeCurve {
    Linear,
    Cosine,
    SCurve,
}

impl FadeCurve {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Cosine => "cosine",
            Self::SCurve => "s-curve",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "linear" => Ok(Self::Linear),
            "cosine" => Ok(Self::Cosine),
            "s-curve" => Ok(Self::SCurve),
            _ => Err(format!("unknown fadeCurve {s:?}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundRobinMode {
    ShuffleNoRepeat,
    Sequential,
}

impl RoundRobinMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ShuffleNoRepeat => "shuffle-no-repeat",
            Self::Sequential => "sequential",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "shuffle-no-repeat" => Ok(Self::ShuffleNoRepeat),
            "sequential" => Ok(Self::Sequential),
            _ => Err(format!("unknown roundRobinMode {s:?}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FilterBand {
    pub enabled: bool,
    pub frequency_hz: f32,
    pub q: f32,
}

impl FilterBand {
    pub fn default_high_pass() -> Self {
        Self {
            enabled: false,
            frequency_hz: 20.0,
            q: std::f32::consts::FRAC_1_SQRT_2,
        }
    }

    pub fn default_low_pass() -> Self {
        Self {
            enabled: false,
            frequency_hz: 20_000.0,
            q: std::f32::consts::FRAC_1_SQRT_2,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParametricEq {
    pub enabled: bool,
    pub low_gain_db: f32,
    pub low_freq_hz: f32,
    pub mid_gain_db: f32,
    pub mid_freq_hz: f32,
    pub mid_q: f32,
    pub high_gain_db: f32,
    pub high_freq_hz: f32,
}

impl Default for ParametricEq {
    fn default() -> Self {
        Self {
            enabled: false,
            low_gain_db: 0.0,
            low_freq_hz: 120.0,
            mid_gain_db: 0.0,
            mid_freq_hz: 1_200.0,
            mid_q: 1.0,
            high_gain_db: 0.0,
            high_freq_hz: 6_500.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompressorSpec {
    pub enabled: bool,
    pub threshold_db: f32,
    pub knee_db: f32,
    pub ratio: f32,
    pub attack_sec: f32,
    pub release_sec: f32,
    pub makeup_gain_db: f32,
    pub mix: f32,
}

impl Default for CompressorSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold_db: -18.0,
            knee_db: 6.0,
            ratio: 3.0,
            attack_sec: 0.008,
            release_sec: 0.14,
            makeup_gain_db: 0.0,
            mix: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OverdriveSpec {
    pub enabled: bool,
    pub drive: f32,
    pub tone_hz: f32,
    pub oversample: OversampleFactor,
    pub output_gain_db: f32,
    pub mix: f32,
}

impl Default for OverdriveSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            drive: 0.0,
            tone_hz: 6_500.0,
            oversample: OversampleFactor::X2,
            output_gain_db: 0.0,
            mix: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaySpec {
    pub enabled: bool,
    pub time_ms: f32,
    pub feedback: f32,
    pub lowpass_hz: f32,
    pub mix: f32,
}

impl Default for DelaySpec {
    fn default() -> Self {
        Self {
            enabled: false,
            time_ms: 140.0,
            feedback: 0.22,
            lowpass_hz: 4_200.0,
            mix: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReverbSpec {
    pub enabled: bool,
    pub room_size: f32,
    pub damping: f32,
    pub pre_delay_ms: f32,
    pub mix: f32,
}

impl Default for ReverbSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            room_size: 0.35,
            damping: 0.45,
            pre_delay_ms: 12.0,
            mix: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DspStrip {
    pub bypass: bool,
    pub input_gain_db: f32,
    pub pan: f32,
    pub stereo_mode: StereoMode,
    pub stereo_width_pct: f32,
    pub high_pass: FilterBand,
    pub low_pass: FilterBand,
    pub eq: ParametricEq,
    pub compressor: CompressorSpec,
    pub overdrive: OverdriveSpec,
    pub delay: DelaySpec,
    pub reverb: ReverbSpec,
}

impl Default for DspStrip {
    fn default() -> Self {
        Self {
            bypass: false,
            input_gain_db: 0.0,
            pan: 0.0,
            stereo_mode: StereoMode::Stereo,
            stereo_width_pct: 100.0,
            high_pass: FilterBand::default_high_pass(),
            low_pass: FilterBand::default_low_pass(),
            eq: ParametricEq::default(),
            compressor: CompressorSpec::default(),
            overdrive: OverdriveSpec::default(),
            delay: DelaySpec::default(),
            reverb: ReverbSpec::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BusSpec {
    pub order: u32,
    pub label: String,
    pub role: BusRole,
    pub color: String,
    pub volume: f32,
    pub muted: bool,
    pub solo: bool,
    pub strip: DspStrip,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SoundAsset {
    pub order: u32,
    pub pool_id: String,
    pub label: String,
    pub bus: String,
    pub path: String,
    pub take_index: u32,
    pub take_count: u32,
    pub default_duration_ms: f32,
    pub transient_offset_ms: f32,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TakeContainer {
    pub order: u32,
    pub label: String,
    pub bus: String,
    pub asset_ids: Vec<String>,
    pub round_robin_mode: RoundRobinMode,
    pub pitch_jitter_cents: f32,
    pub gain_jitter_db: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EventLayer {
    pub order: u32,
    pub label: String,
    pub asset_id: Option<String>,
    pub pool_id: Option<String>,
    pub bus: String,
    pub enabled: bool,
    pub solo: bool,
    pub start_ms: f32,
    pub duration_ms: f32,
    pub trim_start_ms: f32,
    pub fade_in_ms: f32,
    pub fade_out_ms: f32,
    pub fade_curve: FadeCurve,
    pub gain_db: f32,
    pub pan: f32,
    pub pitch_semitones: f32,
    pub pitch_jitter_cents: f32,
    pub gain_jitter_db: f32,
    pub color: String,
    pub strip: DspStrip,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EventSheet {
    pub order: u32,
    pub cue_id: Option<String>,
    pub label: String,
    pub bus: String,
    pub duration_ms: f32,
    pub contact_marker_ms: f32,
    pub contact_label: String,
    pub description: String,
    pub layers: BTreeMap<String, EventLayer>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SequenceStep {
    pub event_id: String,
    pub at_ms: f32,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChoreographySequence {
    pub order: u32,
    pub label: String,
    pub steps: Vec<SequenceStep>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PresetBusPatch {
    pub volume: Option<f32>,
    pub strip: DspStrip,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MixPreset {
    pub order: u32,
    pub label: String,
    pub description: String,
    pub buses: BTreeMap<String, PresetBusPatch>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DuckingRule {
    pub source_bus: String,
    pub target_buses: Vec<String>,
    pub enabled: bool,
    pub amount: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SoundBankDynamics {
    pub enabled: bool,
    pub tension: f32,
    pub graceful_exit_ms: f32,
    pub ducking: BTreeMap<String, DuckingRule>,
}

impl Default for SoundBankDynamics {
    fn default() -> Self {
        Self {
            enabled: false,
            tension: 0.0,
            graceful_exit_ms: 180.0,
            ducking: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SoundBank {
    pub id: String,
    pub name: String,
    pub sample_rate: u32,
    pub buses: BTreeMap<String, BusSpec>,
    pub assets: BTreeMap<String, SoundAsset>,
    pub containers: BTreeMap<String, TakeContainer>,
    pub events: BTreeMap<String, EventSheet>,
    pub sequences: BTreeMap<String, ChoreographySequence>,
    pub presets: BTreeMap<String, MixPreset>,
    pub dynamics: SoundBankDynamics,
}

impl SoundBank {
    /// Creates a minimal valid genre-neutral SoundBank with `sfx` + `master` buses.
    pub fn starter(id: &str, name: &str) -> Self {
        let mut buses = BTreeMap::new();
        buses.insert(
            "sfx".into(),
            BusSpec {
                order: 0,
                label: "SFX".into(),
                role: BusRole::Channel,
                color: "#60a5fa".into(),
                volume: 1.0,
                muted: false,
                solo: false,
                strip: DspStrip::default(),
            },
        );
        buses.insert(
            MASTER_BUS_ID.into(),
            BusSpec {
                order: 1,
                label: "Master".into(),
                role: BusRole::Master,
                color: "#e8b44a".into(),
                volume: 1.0,
                muted: false,
                solo: false,
                strip: DspStrip::default(),
            },
        );
        Self {
            id: id.into(),
            name: name.into(),
            sample_rate: DEFAULT_SAMPLE_RATE,
            buses,
            assets: BTreeMap::new(),
            containers: BTreeMap::new(),
            events: BTreeMap::new(),
            sequences: BTreeMap::new(),
            presets: BTreeMap::new(),
            dynamics: SoundBankDynamics::default(),
        }
    }

    /// Returns bus entries ordered with channel buses by `(order, id)` followed by `master`.
    pub fn ordered_buses(&self) -> Vec<(&str, &BusSpec)> {
        let mut out: Vec<(&str, &BusSpec)> = self
            .buses
            .iter()
            .map(|(id, spec)| (id.as_str(), spec))
            .collect();
        out.sort_by_key(|(id, spec)| (spec.role == BusRole::Master, spec.order, *id));
        out
    }

    /// Returns event entries ordered by `(order, id)`.
    pub fn ordered_events(&self) -> Vec<(&str, &EventSheet)> {
        let mut out: Vec<(&str, &EventSheet)> = self
            .events
            .iter()
            .map(|(id, ev)| (id.as_str(), ev))
            .collect();
        out.sort_by_key(|(id, ev)| (ev.order, *id));
        out
    }

    /// Returns layer entries of an event ordered by `(order, id)`.
    pub fn ordered_layers(event: &EventSheet) -> Vec<(&str, &EventLayer)> {
        let mut out: Vec<(&str, &EventLayer)> = event
            .layers
            .iter()
            .map(|(id, layer)| (id.as_str(), layer))
            .collect();
        out.sort_by_key(|(id, layer)| (layer.order, *id));
        out
    }
}

mod codec;
mod validate;

pub use validate::validate_strip;
