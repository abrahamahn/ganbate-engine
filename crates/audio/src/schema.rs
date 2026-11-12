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
use std::path::Path;

use ganbate_core::id::is_asset_id;
use ganbate_core::json::{self, Value};

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

    /// Parses and validates a `.soundbank.json` byte slice.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let root = json::parse(bytes)?;
        let bank = Self::from_value(&root)?;
        bank.validate()?;
        Ok(bank)
    }

    /// Parses a `.soundbank.json` JSON `Value` without running cross-reference checks yet.
    pub fn from_value(v: &Value) -> Result<Self, String> {
        let Value::Object(map) = v else {
            return Err("a soundbank document must be a JSON object".into());
        };
        for key in map.keys() {
            if !matches!(
                key.as_str(),
                "ganbate"
                    | "version"
                    | "id"
                    | "name"
                    | "sampleRate"
                    | "buses"
                    | "assets"
                    | "containers"
                    | "events"
                    | "sequences"
                    | "presets"
                    | "dynamics"
            ) {
                return Err(format!("unknown soundbank key {key:?}"));
            }
        }
        let kind = req_str(v, "ganbate", "root")?;
        if kind != SOUNDBANK_KIND {
            return Err(format!("expected \"ganbate\": \"soundbank\", got {kind:?}"));
        }
        let version = v
            .get("version")
            .ok_or_else(|| "missing \"version\"".to_string())?
            .unsigned()
            .map_err(|e| format!("version: {e}"))?;
        if version != SOUNDBANK_VERSION {
            return Err(format!("unsupported soundbank version {version}"));
        }
        let id = req_str(v, "id", "root")?;
        let name = req_str(v, "name", "root")?;
        let sample_rate = match v.get("sampleRate") {
            None => DEFAULT_SAMPLE_RATE,
            Some(sr) => {
                let u = sr.unsigned().map_err(|e| format!("sampleRate: {e}"))?;
                u32::try_from(u).map_err(|_| "sampleRate exceeds u32".to_string())?
            }
        };

        let buses = parse_collection(v.get("buses"), "buses", parse_bus)?;
        let assets = parse_collection(v.get("assets"), "assets", parse_asset)?;
        let containers = parse_collection(v.get("containers"), "containers", parse_container)?;
        let events = parse_collection(v.get("events"), "events", parse_event)?;
        let sequences = parse_collection(v.get("sequences"), "sequences", parse_sequence)?;
        let presets = parse_collection(v.get("presets"), "presets", parse_preset)?;
        let dynamics = match v.get("dynamics") {
            None => SoundBankDynamics::default(),
            Some(dv) => parse_dynamics(dv)?,
        };

        Ok(Self {
            id,
            name,
            sample_rate,
            buses,
            assets,
            containers,
            events,
            sequences,
            presets,
            dynamics,
        })
    }

    /// Validates structural, referential, and DSP parameter invariants.
    pub fn validate(&self) -> Result<(), String> {
        if !is_asset_id(&self.id) && !ganbate_core::id::is_entity_id(&self.id) {
            return Err(format!(
                "soundbank id {:?} must be a 16- or 32-digit lowercase hex id",
                self.id
            ));
        }
        if self.name.trim().is_empty() {
            return Err("soundbank name must not be empty".into());
        }
        if !(8_000..=192_000).contains(&self.sample_rate) {
            return Err(format!(
                "sampleRate {} must be between 8000 and 192000 Hz",
                self.sample_rate
            ));
        }
        let Some(master) = self.buses.get(MASTER_BUS_ID) else {
            return Err("buses must include a \"master\" bus".into());
        };
        if master.role != BusRole::Master {
            return Err("\"master\" bus must have role \"master\"".into());
        }
        for (bus_id, bus) in &self.buses {
            validate_key(bus_id, "bus id")?;
            if bus_id != MASTER_BUS_ID && bus.role == BusRole::Master {
                return Err(format!(
                    "only \"master\" may use role \"master\", got {bus_id:?}"
                ));
            }
            if bus.label.trim().is_empty() {
                return Err(format!("bus {bus_id:?} label must not be empty"));
            }
            check_range(bus.volume, 0.0, 2.0, &format!("buses.{bus_id}.volume"))?;
            validate_strip(&bus.strip, &format!("buses.{bus_id}.strip"))?;
        }

        for (asset_id, asset) in &self.assets {
            validate_key(asset_id, "asset id")?;
            if !self.buses.contains_key(&asset.bus) {
                return Err(format!(
                    "asset {asset_id:?} references unknown bus {:?}",
                    asset.bus
                ));
            }
            validate_rel_path(&asset.path, &format!("assets.{asset_id}.path"))?;
            if asset.take_count == 0 || asset.take_index == 0 || asset.take_index > asset.take_count
            {
                return Err(format!(
                    "asset {asset_id:?} has invalid take {}/{}",
                    asset.take_index, asset.take_count
                ));
            }
            check_range(
                asset.default_duration_ms,
                1.0,
                600_000.0,
                &format!("assets.{asset_id}.defaultDurationMs"),
            )?;
            check_range(
                asset.transient_offset_ms,
                0.0,
                asset.default_duration_ms,
                &format!("assets.{asset_id}.transientOffsetMs"),
            )?;
        }

        for (pool_id, pool) in &self.containers {
            validate_key(pool_id, "container id")?;
            if !self.buses.contains_key(&pool.bus) {
                return Err(format!(
                    "container {pool_id:?} references unknown bus {:?}",
                    pool.bus
                ));
            }
            if pool.asset_ids.is_empty() {
                return Err(format!(
                    "container {pool_id:?} must list at least one assetId"
                ));
            }
            for aid in &pool.asset_ids {
                if !self.assets.contains_key(aid) {
                    return Err(format!(
                        "container {pool_id:?} references unknown asset {aid:?}"
                    ));
                }
            }
            check_range(
                pool.pitch_jitter_cents,
                0.0,
                1_200.0,
                &format!("containers.{pool_id}.pitchJitterCents"),
            )?;
            check_range(
                pool.gain_jitter_db,
                0.0,
                24.0,
                &format!("containers.{pool_id}.gainJitterDb"),
            )?;
        }

        for (event_id, event) in &self.events {
            validate_key(event_id, "event id")?;
            if !self.buses.contains_key(&event.bus) {
                return Err(format!(
                    "event {event_id:?} references unknown bus {:?}",
                    event.bus
                ));
            }
            check_range(
                event.duration_ms,
                20.0,
                600_000.0,
                &format!("events.{event_id}.durationMs"),
            )?;
            check_range(
                event.contact_marker_ms,
                0.0,
                event.duration_ms,
                &format!("events.{event_id}.contactMarkerMs"),
            )?;
            for (layer_id, layer) in &event.layers {
                validate_key(layer_id, "layer id")?;
                if !self.buses.contains_key(&layer.bus) {
                    return Err(format!(
                        "layer {event_id:?}/{layer_id:?} references unknown bus {:?}",
                        layer.bus
                    ));
                }
                if let Some(aid) = &layer.asset_id {
                    if !self.assets.contains_key(aid) {
                        return Err(format!(
                            "layer {event_id:?}/{layer_id:?} references unknown asset {aid:?}"
                        ));
                    }
                }
                if let Some(pid) = &layer.pool_id {
                    if !self.containers.contains_key(pid) {
                        return Err(format!(
                            "layer {event_id:?}/{layer_id:?} references unknown container {pid:?}"
                        ));
                    }
                }
                if layer.asset_id.is_none() && layer.pool_id.is_none() {
                    return Err(format!(
                        "layer {event_id:?}/{layer_id:?} must specify assetId or poolId"
                    ));
                }
                check_range(
                    layer.start_ms,
                    0.0,
                    event.duration_ms,
                    &format!("events.{event_id}.layers.{layer_id}.startMs"),
                )?;
                check_range(
                    layer.duration_ms,
                    1.0,
                    600_000.0,
                    &format!("events.{event_id}.layers.{layer_id}.durationMs"),
                )?;
                check_range(
                    layer.trim_start_ms,
                    0.0,
                    600_000.0,
                    &format!("events.{event_id}.layers.{layer_id}.trimStartMs"),
                )?;
                check_range(
                    layer.fade_in_ms,
                    0.0,
                    layer.duration_ms,
                    &format!("events.{event_id}.layers.{layer_id}.fadeInMs"),
                )?;
                check_range(
                    layer.fade_out_ms,
                    0.0,
                    layer.duration_ms,
                    &format!("events.{event_id}.layers.{layer_id}.fadeOutMs"),
                )?;
                check_range(
                    layer.gain_db,
                    -60.0,
                    24.0,
                    &format!("events.{event_id}.layers.{layer_id}.gainDb"),
                )?;
                check_range(
                    layer.pan,
                    -1.0,
                    1.0,
                    &format!("events.{event_id}.layers.{layer_id}.pan"),
                )?;
                check_range(
                    layer.pitch_semitones,
                    -24.0,
                    24.0,
                    &format!("events.{event_id}.layers.{layer_id}.pitchSemitones"),
                )?;
                check_range(
                    layer.pitch_jitter_cents,
                    0.0,
                    1_200.0,
                    &format!("events.{event_id}.layers.{layer_id}.pitchJitterCents"),
                )?;
                check_range(
                    layer.gain_jitter_db,
                    0.0,
                    24.0,
                    &format!("events.{event_id}.layers.{layer_id}.gainJitterDb"),
                )?;
                validate_strip(
                    &layer.strip,
                    &format!("events.{event_id}.layers.{layer_id}.strip"),
                )?;
            }
        }

        for (seq_id, seq) in &self.sequences {
            validate_key(seq_id, "sequence id")?;
            for (idx, step) in seq.steps.iter().enumerate() {
                if !self.events.contains_key(&step.event_id) {
                    return Err(format!(
                        "sequence {seq_id:?} step {idx} references unknown event {:?}",
                        step.event_id
                    ));
                }
                check_range(
                    step.at_ms,
                    0.0,
                    3_600_000.0,
                    &format!("sequences.{seq_id}.steps[{idx}].atMs"),
                )?;
            }
        }

        for (preset_id, preset) in &self.presets {
            validate_key(preset_id, "preset id")?;
            for (bus_id, patch) in &preset.buses {
                if !self.buses.contains_key(bus_id) {
                    return Err(format!(
                        "preset {preset_id:?} references unknown bus {bus_id:?}"
                    ));
                }
                if let Some(vol) = patch.volume {
                    check_range(
                        vol,
                        0.0,
                        2.0,
                        &format!("presets.{preset_id}.buses.{bus_id}.volume"),
                    )?;
                }
                validate_strip(
                    &patch.strip,
                    &format!("presets.{preset_id}.buses.{bus_id}.strip"),
                )?;
            }
        }

        check_range(self.dynamics.tension, 0.0, 1.0, "dynamics.tension")?;
        check_range(
            self.dynamics.graceful_exit_ms,
            0.0,
            30_000.0,
            "dynamics.gracefulExitMs",
        )?;
        for (rule_id, rule) in &self.dynamics.ducking {
            validate_key(rule_id, "ducking rule id")?;
            if !self.buses.contains_key(&rule.source_bus) {
                return Err(format!(
                    "ducking rule {rule_id:?} references unknown sourceBus {:?}",
                    rule.source_bus
                ));
            }
            for target in &rule.target_buses {
                if !self.buses.contains_key(target) {
                    return Err(format!(
                        "ducking rule {rule_id:?} references unknown target bus {target:?}"
                    ));
                }
                if target == &rule.source_bus {
                    return Err(format!(
                        "ducking rule {rule_id:?} cannot duck its own sourceBus {target:?}"
                    ));
                }
            }
            check_range(
                rule.amount,
                0.0,
                1.0,
                &format!("dynamics.ducking.{rule_id}.amount"),
            )?;
            check_range(
                rule.attack_ms,
                0.5,
                5_000.0,
                &format!("dynamics.ducking.{rule_id}.attackMs"),
            )?;
            check_range(
                rule.release_ms,
                1.0,
                10_000.0,
                &format!("dynamics.ducking.{rule_id}.releaseMs"),
            )?;
        }

        Ok(())
    }

    /// Converts to a JSON `Value` ready for `ganbate_core::json::to_canonical`.
    pub fn to_value(&self) -> Value {
        let buses = Value::Object(
            self.buses
                .iter()
                .map(|(k, v)| (k.clone(), bus_to_value(v)))
                .collect(),
        );
        let assets = Value::Object(
            self.assets
                .iter()
                .map(|(k, v)| (k.clone(), asset_to_value(v)))
                .collect(),
        );
        let containers = Value::Object(
            self.containers
                .iter()
                .map(|(k, v)| (k.clone(), container_to_value(v)))
                .collect(),
        );
        let events = Value::Object(
            self.events
                .iter()
                .map(|(k, v)| (k.clone(), event_to_value(v)))
                .collect(),
        );
        let sequences = Value::Object(
            self.sequences
                .iter()
                .map(|(k, v)| (k.clone(), sequence_to_value(v)))
                .collect(),
        );
        let presets = Value::Object(
            self.presets
                .iter()
                .map(|(k, v)| (k.clone(), preset_to_value(v)))
                .collect(),
        );

        obj(vec![
            ("ganbate", s(SOUNDBANK_KIND)),
            ("version", u(SOUNDBANK_VERSION)),
            ("id", s(&self.id)),
            ("name", s(&self.name)),
            ("sampleRate", u(u64::from(self.sample_rate))),
            ("buses", buses),
            ("assets", assets),
            ("containers", containers),
            ("events", events),
            ("sequences", sequences),
            ("presets", presets),
            ("dynamics", dynamics_to_value(&self.dynamics)),
        ])
    }

    /// Validates and writes canonical JSON text (`ganbate_core::json::to_canonical`).
    pub fn to_canonical_json(&self) -> Result<String, String> {
        self.validate()?;
        json::to_canonical(&self.to_value())
    }
}

fn validate_key(key: &str, kind: &str) -> Result<(), String> {
    if key.is_empty() || key.len() > 96 {
        return Err(format!("{kind} {key:?} must be 1..=96 characters"));
    }
    if key == "__proto__" || key == "constructor" || key == "prototype" {
        return Err(format!("{kind} {key:?} is reserved"));
    }
    Ok(())
}

fn validate_rel_path(path: &str, ctx: &str) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(format!("{ctx}: path must not be empty"));
    }
    let normalized = trimmed.trim_start_matches('/');
    let p = Path::new(normalized);
    if normalized.is_empty()
        || p.components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(format!("{ctx}: invalid relative path {path:?}"));
    }
    Ok(())
}

fn check_range(v: f32, min: f32, max: f32, ctx: &str) -> Result<(), String> {
    if !v.is_finite() || v < min || v > max {
        return Err(format!("{ctx}: {v} out of range [{min}, {max}]"));
    }
    Ok(())
}

pub fn validate_strip(strip: &DspStrip, ctx: &str) -> Result<(), String> {
    check_range(
        strip.input_gain_db,
        -60.0,
        24.0,
        &format!("{ctx}.inputGainDb"),
    )?;
    check_range(strip.pan, -1.0, 1.0, &format!("{ctx}.pan"))?;
    check_range(
        strip.stereo_width_pct,
        0.0,
        200.0,
        &format!("{ctx}.stereoWidthPct"),
    )?;
    check_range(
        strip.high_pass.frequency_hz,
        15.0,
        22_000.0,
        &format!("{ctx}.highPass.frequencyHz"),
    )?;
    check_range(strip.high_pass.q, 0.1, 18.0, &format!("{ctx}.highPass.q"))?;
    check_range(
        strip.low_pass.frequency_hz,
        20.0,
        24_000.0,
        &format!("{ctx}.lowPass.frequencyHz"),
    )?;
    check_range(strip.low_pass.q, 0.1, 18.0, &format!("{ctx}.lowPass.q"))?;
    check_range(
        strip.eq.low_gain_db,
        -24.0,
        24.0,
        &format!("{ctx}.eq.lowGainDb"),
    )?;
    check_range(
        strip.eq.low_freq_hz,
        20.0,
        1_000.0,
        &format!("{ctx}.eq.lowFreqHz"),
    )?;
    check_range(
        strip.eq.mid_gain_db,
        -24.0,
        24.0,
        &format!("{ctx}.eq.midGainDb"),
    )?;
    check_range(
        strip.eq.mid_freq_hz,
        120.0,
        12_000.0,
        &format!("{ctx}.eq.midFreqHz"),
    )?;
    check_range(strip.eq.mid_q, 0.1, 18.0, &format!("{ctx}.eq.midQ"))?;
    check_range(
        strip.eq.high_gain_db,
        -24.0,
        24.0,
        &format!("{ctx}.eq.highGainDb"),
    )?;
    check_range(
        strip.eq.high_freq_hz,
        1_000.0,
        20_000.0,
        &format!("{ctx}.eq.highFreqHz"),
    )?;
    check_range(
        strip.compressor.threshold_db,
        -60.0,
        0.0,
        &format!("{ctx}.compressor.thresholdDb"),
    )?;
    check_range(
        strip.compressor.knee_db,
        0.0,
        40.0,
        &format!("{ctx}.compressor.kneeDb"),
    )?;
    check_range(
        strip.compressor.ratio,
        1.0,
        20.0,
        &format!("{ctx}.compressor.ratio"),
    )?;
    check_range(
        strip.compressor.attack_sec,
        0.0001,
        1.0,
        &format!("{ctx}.compressor.attackSec"),
    )?;
    check_range(
        strip.compressor.release_sec,
        0.005,
        2.5,
        &format!("{ctx}.compressor.releaseSec"),
    )?;
    check_range(
        strip.compressor.makeup_gain_db,
        -12.0,
        24.0,
        &format!("{ctx}.compressor.makeupGainDb"),
    )?;
    check_range(
        strip.compressor.mix,
        0.0,
        1.0,
        &format!("{ctx}.compressor.mix"),
    )?;
    check_range(
        strip.overdrive.drive,
        0.0,
        100.0,
        &format!("{ctx}.overdrive.drive"),
    )?;
    check_range(
        strip.overdrive.tone_hz,
        400.0,
        20_000.0,
        &format!("{ctx}.overdrive.toneHz"),
    )?;
    check_range(
        strip.overdrive.output_gain_db,
        -24.0,
        12.0,
        &format!("{ctx}.overdrive.outputGainDb"),
    )?;
    check_range(
        strip.overdrive.mix,
        0.0,
        1.0,
        &format!("{ctx}.overdrive.mix"),
    )?;
    check_range(
        strip.delay.time_ms,
        1.0,
        2_000.0,
        &format!("{ctx}.delay.timeMs"),
    )?;
    check_range(
        strip.delay.feedback,
        0.0,
        0.95,
        &format!("{ctx}.delay.feedback"),
    )?;
    check_range(
        strip.delay.lowpass_hz,
        300.0,
        20_000.0,
        &format!("{ctx}.delay.lowpassHz"),
    )?;
    check_range(strip.delay.mix, 0.0, 1.0, &format!("{ctx}.delay.mix"))?;
    check_range(
        strip.reverb.room_size,
        0.0,
        1.0,
        &format!("{ctx}.reverb.roomSize"),
    )?;
    check_range(
        strip.reverb.damping,
        0.0,
        1.0,
        &format!("{ctx}.reverb.damping"),
    )?;
    check_range(
        strip.reverb.pre_delay_ms,
        0.0,
        250.0,
        &format!("{ctx}.reverb.preDelayMs"),
    )?;
    check_range(strip.reverb.mix, 0.0, 1.0, &format!("{ctx}.reverb.mix"))?;
    Ok(())
}

// ---- JSON Parse & Emit Helpers --------------------------------------------------------------

fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn s(v: &str) -> Value {
    Value::String(v.into())
}

fn u(v: u64) -> Value {
    Value::Number(v as f64)
}

fn f(v: f32) -> Value {
    Value::Number(f64::from(v))
}

fn req_str(v: &Value, key: &str, ctx: &str) -> Result<String, String> {
    v.get(key)
        .ok_or_else(|| format!("{ctx}: missing {key:?}"))?
        .string()
        .map(String::from)
        .map_err(|e| format!("{ctx}.{key}: {e}"))
}

fn opt_str(v: &Value, key: &str, ctx: &str) -> Result<Option<String>, String> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(val) => val
            .string()
            .map(|s| {
                if s.is_empty() {
                    None
                } else {
                    Some(s.to_string())
                }
            })
            .map_err(|e| format!("{ctx}.{key}: {e}")),
    }
}

fn get_bool(v: &Value, key: &str, default: bool, ctx: &str) -> Result<bool, String> {
    match v.get(key) {
        None => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(format!("{ctx}.{key}: expected boolean")),
    }
}

fn get_f32(v: &Value, key: &str, default: f32, ctx: &str) -> Result<f32, String> {
    match v.get(key) {
        None => Ok(default),
        Some(val) => val.float().map_err(|e| format!("{ctx}.{key}: {e}")),
    }
}

fn get_u32(v: &Value, key: &str, default: u32, ctx: &str) -> Result<u32, String> {
    match v.get(key) {
        None => Ok(default),
        Some(val) => {
            let n = val.unsigned().map_err(|e| format!("{ctx}.{key}: {e}"))?;
            u32::try_from(n).map_err(|_| format!("{ctx}.{key}: integer exceeds u32"))
        }
    }
}

fn get_str_list(v: &Value, key: &str, ctx: &str) -> Result<Vec<String>, String> {
    match v.get(key) {
        None => Ok(vec![]),
        Some(val) => val
            .array()
            .map_err(|e| format!("{ctx}.{key}: {e}"))?
            .iter()
            .enumerate()
            .map(|(i, item)| {
                item.string()
                    .map(String::from)
                    .map_err(|e| format!("{ctx}.{key}[{i}]: {e}"))
            })
            .collect(),
    }
}

/// Parses either `{ "<id>": { ... } }` (canonical) or `[ { "id": "<id>", ... } ]`.
fn parse_collection<T>(
    raw: Option<&Value>,
    ctx: &str,
    parse_item: fn(&str, u32, &Value, &str) -> Result<T, String>,
) -> Result<BTreeMap<String, T>, String> {
    let mut out = BTreeMap::new();
    let Some(val) = raw else {
        return Ok(out);
    };
    match val {
        Value::Object(map) => {
            for (idx, (id, item)) in map.iter().enumerate() {
                let item_ctx = format!("{ctx}.{id}");
                let parsed = parse_item(id, idx as u32, item, &item_ctx)?;
                out.insert(id.clone(), parsed);
            }
        }
        Value::Array(items) => {
            for (idx, item) in items.iter().enumerate() {
                let item_ctx = format!("{ctx}[{idx}]");
                let id = req_str(item, "id", &item_ctx)?;
                if out.contains_key(&id) {
                    return Err(format!("{item_ctx}: duplicate id {id:?}"));
                }
                let parsed = parse_item(&id, idx as u32, item, &item_ctx)?;
                out.insert(id, parsed);
            }
        }
        _ => return Err(format!("{ctx}: expected object or array")),
    }
    Ok(out)
}

fn parse_strip(raw: Option<&Value>, ctx: &str) -> Result<DspStrip, String> {
    let defaults = DspStrip::default();
    let Some(v) = raw else {
        return Ok(defaults);
    };
    if !matches!(v, Value::Object(_)) {
        return Err(format!("{ctx}: expected object"));
    }
    let stereo_mode = match opt_str(v, "stereoMode", ctx)? {
        None => defaults.stereo_mode,
        Some(m) => StereoMode::parse(&m).map_err(|e| format!("{ctx}: {e}"))?,
    };
    let high_pass = match v.get("highPass") {
        None => defaults.high_pass,
        Some(hp) => FilterBand {
            enabled: get_bool(hp, "enabled", false, &format!("{ctx}.highPass"))?,
            frequency_hz: get_f32(hp, "frequencyHz", 20.0, &format!("{ctx}.highPass"))?,
            q: get_f32(
                hp,
                "q",
                std::f32::consts::FRAC_1_SQRT_2,
                &format!("{ctx}.highPass"),
            )?,
        },
    };
    let low_pass = match v.get("lowPass") {
        None => defaults.low_pass,
        Some(lp) => FilterBand {
            enabled: get_bool(lp, "enabled", false, &format!("{ctx}.lowPass"))?,
            frequency_hz: get_f32(lp, "frequencyHz", 20_000.0, &format!("{ctx}.lowPass"))?,
            q: get_f32(
                lp,
                "q",
                std::f32::consts::FRAC_1_SQRT_2,
                &format!("{ctx}.lowPass"),
            )?,
        },
    };
    let eq = match v.get("eq") {
        None => defaults.eq,
        Some(eqv) => {
            let d = ParametricEq::default();
            let sub = format!("{ctx}.eq");
            ParametricEq {
                enabled: get_bool(eqv, "enabled", d.enabled, &sub)?,
                low_gain_db: get_f32(eqv, "lowGainDb", d.low_gain_db, &sub)?,
                low_freq_hz: get_f32(eqv, "lowFreqHz", d.low_freq_hz, &sub)?,
                mid_gain_db: get_f32(eqv, "midGainDb", d.mid_gain_db, &sub)?,
                mid_freq_hz: get_f32(eqv, "midFreqHz", d.mid_freq_hz, &sub)?,
                mid_q: get_f32(eqv, "midQ", d.mid_q, &sub)?,
                high_gain_db: get_f32(eqv, "highGainDb", d.high_gain_db, &sub)?,
                high_freq_hz: get_f32(eqv, "highFreqHz", d.high_freq_hz, &sub)?,
            }
        }
    };
    let compressor = match v.get("compressor") {
        None => defaults.compressor,
        Some(cv) => {
            let d = CompressorSpec::default();
            let sub = format!("{ctx}.compressor");
            CompressorSpec {
                enabled: get_bool(cv, "enabled", d.enabled, &sub)?,
                threshold_db: get_f32(cv, "thresholdDb", d.threshold_db, &sub)?,
                knee_db: get_f32(cv, "kneeDb", d.knee_db, &sub)?,
                ratio: get_f32(cv, "ratio", d.ratio, &sub)?,
                attack_sec: get_f32(cv, "attackSec", d.attack_sec, &sub)?,
                release_sec: get_f32(cv, "releaseSec", d.release_sec, &sub)?,
                makeup_gain_db: get_f32(cv, "makeupGainDb", d.makeup_gain_db, &sub)?,
                mix: get_f32(cv, "mix", d.mix, &sub)?,
            }
        }
    };
    let overdrive = match v.get("overdrive") {
        None => defaults.overdrive,
        Some(ov) => {
            let d = OverdriveSpec::default();
            let sub = format!("{ctx}.overdrive");
            let oversample = match opt_str(ov, "oversample", &sub)? {
                None => d.oversample,
                Some(mode) => OversampleFactor::parse(&mode).map_err(|e| format!("{sub}: {e}"))?,
            };
            OverdriveSpec {
                enabled: get_bool(ov, "enabled", d.enabled, &sub)?,
                drive: get_f32(ov, "drive", d.drive, &sub)?,
                tone_hz: get_f32(ov, "toneHz", d.tone_hz, &sub)?,
                oversample,
                output_gain_db: get_f32(ov, "outputGainDb", d.output_gain_db, &sub)?,
                mix: get_f32(ov, "mix", d.mix, &sub)?,
            }
        }
    };
    let delay = match v.get("delay") {
        None => defaults.delay,
        Some(dv) => {
            let d = DelaySpec::default();
            let sub = format!("{ctx}.delay");
            DelaySpec {
                enabled: get_bool(dv, "enabled", d.enabled, &sub)?,
                time_ms: get_f32(dv, "timeMs", d.time_ms, &sub)?,
                feedback: get_f32(dv, "feedback", d.feedback, &sub)?,
                lowpass_hz: get_f32(dv, "lowpassHz", d.lowpass_hz, &sub)?,
                mix: get_f32(dv, "mix", d.mix, &sub)?,
            }
        }
    };
    let reverb = match v.get("reverb") {
        None => defaults.reverb,
        Some(rv) => {
            let d = ReverbSpec::default();
            let sub = format!("{ctx}.reverb");
            ReverbSpec {
                enabled: get_bool(rv, "enabled", d.enabled, &sub)?,
                room_size: get_f32(rv, "roomSize", d.room_size, &sub)?,
                damping: get_f32(rv, "damping", d.damping, &sub)?,
                pre_delay_ms: get_f32(rv, "preDelayMs", d.pre_delay_ms, &sub)?,
                mix: get_f32(rv, "mix", d.mix, &sub)?,
            }
        }
    };

    Ok(DspStrip {
        bypass: get_bool(v, "bypass", defaults.bypass, ctx)?,
        input_gain_db: get_f32(v, "inputGainDb", defaults.input_gain_db, ctx)?,
        pan: get_f32(v, "pan", defaults.pan, ctx)?,
        stereo_mode,
        stereo_width_pct: get_f32(v, "stereoWidthPct", defaults.stereo_width_pct, ctx)?,
        high_pass,
        low_pass,
        eq,
        compressor,
        overdrive,
        delay,
        reverb,
    })
}

fn parse_bus(id: &str, fallback_order: u32, v: &Value, ctx: &str) -> Result<BusSpec, String> {
    if !matches!(v, Value::Object(_)) {
        return Err(format!("{ctx}: expected object"));
    }
    let default_role = if id == MASTER_BUS_ID {
        BusRole::Master
    } else {
        BusRole::Channel
    };
    let role = match opt_str(v, "role", ctx)? {
        None => default_role,
        Some(r) => BusRole::parse(&r).map_err(|e| format!("{ctx}: {e}"))?,
    };
    Ok(BusSpec {
        order: get_u32(v, "order", fallback_order, ctx)?,
        label: opt_str(v, "label", ctx)?.unwrap_or_else(|| id.to_string()),
        role,
        color: opt_str(v, "color", ctx)?.unwrap_or_else(|| "#7aa2f7".into()),
        volume: get_f32(v, "volume", 1.0, ctx)?,
        muted: get_bool(v, "muted", false, ctx)?,
        solo: get_bool(v, "solo", false, ctx)?,
        strip: parse_strip(v.get("strip"), &format!("{ctx}.strip"))?,
    })
}

fn parse_asset(id: &str, fallback_order: u32, v: &Value, ctx: &str) -> Result<SoundAsset, String> {
    if !matches!(v, Value::Object(_)) {
        return Err(format!("{ctx}: expected object"));
    }
    Ok(SoundAsset {
        order: get_u32(v, "order", fallback_order, ctx)?,
        pool_id: opt_str(v, "poolId", ctx)?.unwrap_or_else(|| id.to_string()),
        label: opt_str(v, "label", ctx)?.unwrap_or_else(|| id.to_string()),
        bus: req_str(v, "bus", ctx)?,
        path: req_str(v, "path", ctx)?,
        take_index: get_u32(v, "takeIndex", 1, ctx)?,
        take_count: get_u32(v, "takeCount", 1, ctx)?,
        default_duration_ms: get_f32(v, "defaultDurationMs", 250.0, ctx)?,
        transient_offset_ms: get_f32(v, "transientOffsetMs", 0.0, ctx)?,
        tags: get_str_list(v, "tags", ctx)?,
    })
}

fn parse_container(
    id: &str,
    fallback_order: u32,
    v: &Value,
    ctx: &str,
) -> Result<TakeContainer, String> {
    if !matches!(v, Value::Object(_)) {
        return Err(format!("{ctx}: expected object"));
    }
    let round_robin_mode = match opt_str(v, "roundRobinMode", ctx)? {
        None => RoundRobinMode::ShuffleNoRepeat,
        Some(m) => RoundRobinMode::parse(&m).map_err(|e| format!("{ctx}: {e}"))?,
    };
    Ok(TakeContainer {
        order: get_u32(v, "order", fallback_order, ctx)?,
        label: opt_str(v, "label", ctx)?.unwrap_or_else(|| id.to_string()),
        bus: req_str(v, "bus", ctx)?,
        asset_ids: get_str_list(v, "assetIds", ctx)?,
        round_robin_mode,
        pitch_jitter_cents: get_f32(v, "pitchJitterCents", 0.0, ctx)?,
        gain_jitter_db: get_f32(v, "gainJitterDb", 0.0, ctx)?,
    })
}

fn parse_layer(id: &str, fallback_order: u32, v: &Value, ctx: &str) -> Result<EventLayer, String> {
    if !matches!(v, Value::Object(_)) {
        return Err(format!("{ctx}: expected object"));
    }
    let fade_curve = match opt_str(v, "fadeCurve", ctx)? {
        None => FadeCurve::Cosine,
        Some(c) => FadeCurve::parse(&c).map_err(|e| format!("{ctx}: {e}"))?,
    };
    Ok(EventLayer {
        order: get_u32(v, "order", fallback_order, ctx)?,
        label: opt_str(v, "label", ctx)?.unwrap_or_else(|| id.to_string()),
        asset_id: opt_str(v, "assetId", ctx)?,
        pool_id: opt_str(v, "poolId", ctx)?,
        bus: req_str(v, "bus", ctx)?,
        enabled: get_bool(v, "enabled", true, ctx)?,
        solo: get_bool(v, "solo", false, ctx)?,
        start_ms: get_f32(v, "startMs", 0.0, ctx)?,
        duration_ms: get_f32(v, "durationMs", 250.0, ctx)?,
        trim_start_ms: get_f32(v, "trimStartMs", 0.0, ctx)?,
        fade_in_ms: get_f32(v, "fadeInMs", 0.0, ctx)?,
        fade_out_ms: get_f32(v, "fadeOutMs", 20.0, ctx)?,
        fade_curve,
        gain_db: get_f32(v, "gainDb", 0.0, ctx)?,
        pan: get_f32(v, "pan", 0.0, ctx)?,
        pitch_semitones: get_f32(v, "pitchSemitones", 0.0, ctx)?,
        pitch_jitter_cents: get_f32(v, "pitchJitterCents", 0.0, ctx)?,
        gain_jitter_db: get_f32(v, "gainJitterDb", 0.0, ctx)?,
        color: opt_str(v, "color", ctx)?.unwrap_or_else(|| "#60a5fa".into()),
        strip: parse_strip(v.get("strip"), &format!("{ctx}.strip"))?,
    })
}

fn parse_event(id: &str, fallback_order: u32, v: &Value, ctx: &str) -> Result<EventSheet, String> {
    if !matches!(v, Value::Object(_)) {
        return Err(format!("{ctx}: expected object"));
    }
    let layers = parse_collection(v.get("layers"), &format!("{ctx}.layers"), parse_layer)?;
    Ok(EventSheet {
        order: get_u32(v, "order", fallback_order, ctx)?,
        cue_id: opt_str(v, "cueId", ctx)?,
        label: opt_str(v, "label", ctx)?.unwrap_or_else(|| id.to_string()),
        bus: req_str(v, "bus", ctx)?,
        duration_ms: get_f32(v, "durationMs", 600.0, ctx)?,
        contact_marker_ms: get_f32(v, "contactMarkerMs", 120.0, ctx)?,
        contact_label: opt_str(v, "contactLabel", ctx)?.unwrap_or_else(|| "Contact".into()),
        description: opt_str(v, "description", ctx)?.unwrap_or_default(),
        layers,
    })
}

fn parse_sequence(
    id: &str,
    fallback_order: u32,
    v: &Value,
    ctx: &str,
) -> Result<ChoreographySequence, String> {
    if !matches!(v, Value::Object(_)) {
        return Err(format!("{ctx}: expected object"));
    }
    let mut steps = Vec::new();
    if let Some(sv) = v.get("steps") {
        for (i, item) in sv
            .array()
            .map_err(|e| format!("{ctx}.steps: {e}"))?
            .iter()
            .enumerate()
        {
            let sctx = format!("{ctx}.steps[{i}]");
            let event_id = req_str(item, "eventId", &sctx)?;
            steps.push(SequenceStep {
                at_ms: get_f32(item, "atMs", 0.0, &sctx)?,
                label: opt_str(item, "label", &sctx)?.unwrap_or_else(|| event_id.clone()),
                event_id,
            });
        }
    }
    Ok(ChoreographySequence {
        order: get_u32(v, "order", fallback_order, ctx)?,
        label: opt_str(v, "label", ctx)?.unwrap_or_else(|| id.to_string()),
        steps,
    })
}

fn parse_preset(id: &str, fallback_order: u32, v: &Value, ctx: &str) -> Result<MixPreset, String> {
    if !matches!(v, Value::Object(_)) {
        return Err(format!("{ctx}: expected object"));
    }
    let mut buses = BTreeMap::new();
    if let Some(bv) = v.get("buses") {
        let Value::Object(map) = bv else {
            return Err(format!("{ctx}.buses: expected object"));
        };
        for (bus_id, patch_val) in map {
            let pctx = format!("{ctx}.buses.{bus_id}");
            let volume = match patch_val.get("volume") {
                None | Some(Value::Null) => None,
                Some(vol) => Some(vol.float().map_err(|e| format!("{pctx}.volume: {e}"))?),
            };
            let strip = parse_strip(patch_val.get("strip"), &format!("{pctx}.strip"))?;
            buses.insert(bus_id.clone(), PresetBusPatch { volume, strip });
        }
    }
    Ok(MixPreset {
        order: get_u32(v, "order", fallback_order, ctx)?,
        label: opt_str(v, "label", ctx)?.unwrap_or_else(|| id.to_string()),
        description: opt_str(v, "description", ctx)?.unwrap_or_default(),
        buses,
    })
}

fn parse_ducking_rule(
    _id: &str,
    _fallback_order: u32,
    v: &Value,
    ctx: &str,
) -> Result<DuckingRule, String> {
    if !matches!(v, Value::Object(_)) {
        return Err(format!("{ctx}: expected object"));
    }
    Ok(DuckingRule {
        source_bus: req_str(v, "sourceBus", ctx)?,
        target_buses: get_str_list(v, "targetBuses", ctx)?,
        enabled: get_bool(v, "enabled", true, ctx)?,
        amount: get_f32(v, "amount", 0.35, ctx)?,
        attack_ms: get_f32(v, "attackMs", 25.0, ctx)?,
        release_ms: get_f32(v, "releaseMs", 220.0, ctx)?,
    })
}

fn parse_dynamics(v: &Value) -> Result<SoundBankDynamics, String> {
    if !matches!(v, Value::Object(_)) {
        return Err("dynamics: expected object".into());
    }
    let d = SoundBankDynamics::default();
    let ducking = parse_collection(v.get("ducking"), "dynamics.ducking", parse_ducking_rule)?;
    Ok(SoundBankDynamics {
        enabled: get_bool(v, "enabled", d.enabled, "dynamics")?,
        tension: get_f32(v, "tension", d.tension, "dynamics")?,
        graceful_exit_ms: get_f32(v, "gracefulExitMs", d.graceful_exit_ms, "dynamics")?,
        ducking,
    })
}

fn strip_to_value(strip: &DspStrip) -> Value {
    obj(vec![
        ("bypass", Value::Bool(strip.bypass)),
        ("inputGainDb", f(strip.input_gain_db)),
        ("pan", f(strip.pan)),
        ("stereoMode", s(strip.stereo_mode.as_str())),
        ("stereoWidthPct", f(strip.stereo_width_pct)),
        (
            "highPass",
            obj(vec![
                ("enabled", Value::Bool(strip.high_pass.enabled)),
                ("frequencyHz", f(strip.high_pass.frequency_hz)),
                ("q", f(strip.high_pass.q)),
            ]),
        ),
        (
            "lowPass",
            obj(vec![
                ("enabled", Value::Bool(strip.low_pass.enabled)),
                ("frequencyHz", f(strip.low_pass.frequency_hz)),
                ("q", f(strip.low_pass.q)),
            ]),
        ),
        (
            "eq",
            obj(vec![
                ("enabled", Value::Bool(strip.eq.enabled)),
                ("lowGainDb", f(strip.eq.low_gain_db)),
                ("lowFreqHz", f(strip.eq.low_freq_hz)),
                ("midGainDb", f(strip.eq.mid_gain_db)),
                ("midFreqHz", f(strip.eq.mid_freq_hz)),
                ("midQ", f(strip.eq.mid_q)),
                ("highGainDb", f(strip.eq.high_gain_db)),
                ("highFreqHz", f(strip.eq.high_freq_hz)),
            ]),
        ),
        (
            "compressor",
            obj(vec![
                ("enabled", Value::Bool(strip.compressor.enabled)),
                ("thresholdDb", f(strip.compressor.threshold_db)),
                ("kneeDb", f(strip.compressor.knee_db)),
                ("ratio", f(strip.compressor.ratio)),
                ("attackSec", f(strip.compressor.attack_sec)),
                ("releaseSec", f(strip.compressor.release_sec)),
                ("makeupGainDb", f(strip.compressor.makeup_gain_db)),
                ("mix", f(strip.compressor.mix)),
            ]),
        ),
        (
            "overdrive",
            obj(vec![
                ("enabled", Value::Bool(strip.overdrive.enabled)),
                ("drive", f(strip.overdrive.drive)),
                ("toneHz", f(strip.overdrive.tone_hz)),
                ("oversample", s(strip.overdrive.oversample.as_str())),
                ("outputGainDb", f(strip.overdrive.output_gain_db)),
                ("mix", f(strip.overdrive.mix)),
            ]),
        ),
        (
            "delay",
            obj(vec![
                ("enabled", Value::Bool(strip.delay.enabled)),
                ("timeMs", f(strip.delay.time_ms)),
                ("feedback", f(strip.delay.feedback)),
                ("lowpassHz", f(strip.delay.lowpass_hz)),
                ("mix", f(strip.delay.mix)),
            ]),
        ),
        (
            "reverb",
            obj(vec![
                ("enabled", Value::Bool(strip.reverb.enabled)),
                ("roomSize", f(strip.reverb.room_size)),
                ("damping", f(strip.reverb.damping)),
                ("preDelayMs", f(strip.reverb.pre_delay_ms)),
                ("mix", f(strip.reverb.mix)),
            ]),
        ),
    ])
}

fn bus_to_value(bus: &BusSpec) -> Value {
    obj(vec![
        ("order", u(u64::from(bus.order))),
        ("label", s(&bus.label)),
        ("role", s(bus.role.as_str())),
        ("color", s(&bus.color)),
        ("volume", f(bus.volume)),
        ("muted", Value::Bool(bus.muted)),
        ("solo", Value::Bool(bus.solo)),
        ("strip", strip_to_value(&bus.strip)),
    ])
}

fn asset_to_value(asset: &SoundAsset) -> Value {
    obj(vec![
        ("order", u(u64::from(asset.order))),
        ("poolId", s(&asset.pool_id)),
        ("label", s(&asset.label)),
        ("bus", s(&asset.bus)),
        ("path", s(&asset.path)),
        ("takeIndex", u(u64::from(asset.take_index))),
        ("takeCount", u(u64::from(asset.take_count))),
        ("defaultDurationMs", f(asset.default_duration_ms)),
        ("transientOffsetMs", f(asset.transient_offset_ms)),
        (
            "tags",
            Value::Array(asset.tags.iter().map(|t| s(t)).collect()),
        ),
    ])
}

fn container_to_value(c: &TakeContainer) -> Value {
    obj(vec![
        ("order", u(u64::from(c.order))),
        ("label", s(&c.label)),
        ("bus", s(&c.bus)),
        (
            "assetIds",
            Value::Array(c.asset_ids.iter().map(|a| s(a)).collect()),
        ),
        ("roundRobinMode", s(c.round_robin_mode.as_str())),
        ("pitchJitterCents", f(c.pitch_jitter_cents)),
        ("gainJitterDb", f(c.gain_jitter_db)),
    ])
}

fn layer_to_value(layer: &EventLayer) -> Value {
    let mut fields = vec![
        ("order", u(u64::from(layer.order))),
        ("label", s(&layer.label)),
        ("bus", s(&layer.bus)),
        ("enabled", Value::Bool(layer.enabled)),
        ("solo", Value::Bool(layer.solo)),
        ("startMs", f(layer.start_ms)),
        ("durationMs", f(layer.duration_ms)),
        ("trimStartMs", f(layer.trim_start_ms)),
        ("fadeInMs", f(layer.fade_in_ms)),
        ("fadeOutMs", f(layer.fade_out_ms)),
        ("fadeCurve", s(layer.fade_curve.as_str())),
        ("gainDb", f(layer.gain_db)),
        ("pan", f(layer.pan)),
        ("pitchSemitones", f(layer.pitch_semitones)),
        ("pitchJitterCents", f(layer.pitch_jitter_cents)),
        ("gainJitterDb", f(layer.gain_jitter_db)),
        ("color", s(&layer.color)),
        ("strip", strip_to_value(&layer.strip)),
    ];
    if let Some(aid) = &layer.asset_id {
        fields.push(("assetId", s(aid)));
    }
    if let Some(pid) = &layer.pool_id {
        fields.push(("poolId", s(pid)));
    }
    obj(fields)
}

fn event_to_value(ev: &EventSheet) -> Value {
    let layers = Value::Object(
        ev.layers
            .iter()
            .map(|(k, v)| (k.clone(), layer_to_value(v)))
            .collect(),
    );
    let mut fields = vec![
        ("order", u(u64::from(ev.order))),
        ("label", s(&ev.label)),
        ("bus", s(&ev.bus)),
        ("durationMs", f(ev.duration_ms)),
        ("contactMarkerMs", f(ev.contact_marker_ms)),
        ("contactLabel", s(&ev.contact_label)),
        ("description", s(&ev.description)),
        ("layers", layers),
    ];
    if let Some(cue) = &ev.cue_id {
        fields.push(("cueId", s(cue)));
    }
    obj(fields)
}

fn sequence_to_value(seq: &ChoreographySequence) -> Value {
    let steps = Value::Array(
        seq.steps
            .iter()
            .map(|st| {
                obj(vec![
                    ("eventId", s(&st.event_id)),
                    ("atMs", f(st.at_ms)),
                    ("label", s(&st.label)),
                ])
            })
            .collect(),
    );
    obj(vec![
        ("order", u(u64::from(seq.order))),
        ("label", s(&seq.label)),
        ("steps", steps),
    ])
}

fn preset_to_value(preset: &MixPreset) -> Value {
    let buses = Value::Object(
        preset
            .buses
            .iter()
            .map(|(bus_id, patch)| {
                let mut fields = vec![("strip", strip_to_value(&patch.strip))];
                if let Some(vol) = patch.volume {
                    fields.push(("volume", f(vol)));
                }
                (bus_id.clone(), obj(fields))
            })
            .collect(),
    );
    obj(vec![
        ("order", u(u64::from(preset.order))),
        ("label", s(&preset.label)),
        ("description", s(&preset.description)),
        ("buses", buses),
    ])
}

fn dynamics_to_value(dyn_spec: &SoundBankDynamics) -> Value {
    let ducking = Value::Object(
        dyn_spec
            .ducking
            .iter()
            .map(|(k, r)| {
                (
                    k.clone(),
                    obj(vec![
                        ("sourceBus", s(&r.source_bus)),
                        (
                            "targetBuses",
                            Value::Array(r.target_buses.iter().map(|t| s(t)).collect()),
                        ),
                        ("enabled", Value::Bool(r.enabled)),
                        ("amount", f(r.amount)),
                        ("attackMs", f(r.attack_ms)),
                        ("releaseMs", f(r.release_ms)),
                    ]),
                )
            })
            .collect(),
    );
    obj(vec![
        ("enabled", Value::Bool(dyn_spec.enabled)),
        ("tension", f(dyn_spec.tension)),
        ("gracefulExitMs", f(dyn_spec.graceful_exit_ms)),
        ("ducking", ducking),
    ])
}
