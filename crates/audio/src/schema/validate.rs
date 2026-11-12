use std::path::Path;

use ganbate_core::id::is_asset_id;

use super::*;

impl SoundBank {
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
