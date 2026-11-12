use std::collections::BTreeMap;

use ganbate_core::json::{self, Value};

use super::*;

impl SoundBank {
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
