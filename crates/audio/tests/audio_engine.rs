use std::collections::BTreeMap;
use std::f32::consts::PI;

use ganbate_audio::{
    AudioBuffer, BusRole, BusSpec, ChoreographySequence, CompressorSpec, ContainerSelector,
    DelaySpec, DspStrip, DspStripProcessor, DuckingRule, EventLayer, EventSheet, FadeCurve,
    FilterBand, MASTER_BUS_ID, OverdriveSpec, OversampleFactor, ParametricEq, ReverbSpec,
    RoundRobinMode, SequenceStep, SoundAsset, SoundBank, StereoMode, TakeContainer,
    analyze_stereo_frames, bounce_event_to_wav16, bounce_sequence_to_wav16, clip_fade_envelope,
    compute_mid_side_matrix, decode_audio, decode_qoa, decode_wav, encode_qoa, encode_wav_pcm16,
    evaluate_fade_curve, evaluate_strip_response_db, render_event_offline,
};

fn build_sine_buffer(
    sample_rate: u32,
    frames: usize,
    freq_l: f32,
    freq_r: f32,
    amp: f32,
) -> AudioBuffer {
    let mut out = Vec::with_capacity(frames);
    for i in 0..frames {
        let t = (i as f32) / (sample_rate as f32);
        let l = (2.0 * PI * freq_l * t).sin() * amp;
        let r = (2.0 * PI * freq_r * t).sin() * amp;
        out.push([l, r]);
    }
    AudioBuffer {
        sample_rate,
        frames: out,
    }
}

fn sample_action_soundbank() -> SoundBank {
    let mut bank = SoundBank::starter("8b4f1d6209c3471aa520e374b8c9d012", "Action Foley Workbench");
    bank.buses.insert(
        "foley".into(),
        BusSpec {
            order: 1,
            label: "Foley".into(),
            role: BusRole::Channel,
            color: "#7fd17f".into(),
            volume: 1.0,
            muted: false,
            solo: false,
            strip: DspStrip::default(),
        },
    );
    bank.buses.insert(
        "dialogue".into(),
        BusSpec {
            order: 2,
            label: "Dialogue".into(),
            role: BusRole::Channel,
            color: "#c792ea".into(),
            volume: 1.0,
            muted: false,
            solo: false,
            strip: DspStrip::default(),
        },
    );

    for take in 1..=4 {
        let id = format!("step_stone_0{take}");
        bank.assets.insert(
            id,
            SoundAsset {
                order: take - 1,
                pool_id: "step_stone".into(),
                label: format!("Stone Step Take {take}"),
                bus: "foley".into(),
                path: format!("audio/foley/step_stone_0{take}.wav"),
                take_index: take,
                take_count: 4,
                default_duration_ms: 220.0,
                transient_offset_ms: 15.0,
                tags: vec!["footstep".into(), "stone".into()],
            },
        );
    }
    bank.assets.insert(
        "voice_callout_01".into(),
        SoundAsset {
            order: 4,
            pool_id: "voice_callout".into(),
            label: "Callout Ready".into(),
            bus: "dialogue".into(),
            path: "audio/dialogue/callout_01.wav".into(),
            take_index: 1,
            take_count: 1,
            default_duration_ms: 320.0,
            transient_offset_ms: 10.0,
            tags: vec!["voice".into()],
        },
    );

    bank.containers.insert(
        "step_stone".into(),
        TakeContainer {
            order: 0,
            label: "Stone Footstep Pool".into(),
            bus: "foley".into(),
            asset_ids: vec![
                "step_stone_01".into(),
                "step_stone_02".into(),
                "step_stone_03".into(),
                "step_stone_04".into(),
            ],
            round_robin_mode: RoundRobinMode::ShuffleNoRepeat,
            pitch_jitter_cents: 35.0,
            gain_jitter_db: 0.8,
        },
    );

    let mut layers = BTreeMap::new();
    layers.insert(
        "lyr_heel".into(),
        EventLayer {
            order: 0,
            label: "Stone Heel Impact".into(),
            asset_id: None,
            pool_id: Some("step_stone".into()),
            bus: "foley".into(),
            enabled: true,
            solo: false,
            start_ms: 65.0,
            duration_ms: 180.0,
            trim_start_ms: 5.0,
            fade_in_ms: 12.0,
            fade_out_ms: 45.0,
            fade_curve: FadeCurve::SCurve,
            gain_db: -1.5,
            pan: -0.15,
            pitch_semitones: 0.0,
            pitch_jitter_cents: 15.0,
            gain_jitter_db: 0.4,
            color: "#7fd17f".into(),
            strip: DspStrip {
                high_pass: FilterBand {
                    enabled: true,
                    frequency_hz: 75.0,
                    q: 0.707,
                },
                ..DspStrip::default()
            },
        },
    );

    bank.events.insert(
        "evt_footstep".into(),
        EventSheet {
            order: 0,
            cue_id: Some("footstep_stone".into()),
            label: "Stone Footstep".into(),
            bus: "foley".into(),
            duration_ms: 420.0,
            contact_marker_ms: 80.0,
            contact_label: "Heel Contact (Frame 5)".into(),
            description: "Contact-synced stone footstep impact".into(),
            layers,
        },
    );

    let mut voice_layers = BTreeMap::new();
    voice_layers.insert(
        "lyr_voice".into(),
        EventLayer {
            order: 0,
            label: "Dialogue Line".into(),
            asset_id: Some("voice_callout_01".into()),
            pool_id: None,
            bus: "dialogue".into(),
            enabled: true,
            solo: false,
            start_ms: 0.0,
            duration_ms: 300.0,
            trim_start_ms: 0.0,
            fade_in_ms: 10.0,
            fade_out_ms: 30.0,
            fade_curve: FadeCurve::Cosine,
            gain_db: 0.0,
            pan: 0.0,
            pitch_semitones: 0.0,
            pitch_jitter_cents: 0.0,
            gain_jitter_db: 0.0,
            color: "#c792ea".into(),
            strip: DspStrip::default(),
        },
    );
    bank.events.insert(
        "evt_callout".into(),
        EventSheet {
            order: 1,
            cue_id: Some("callout_ready".into()),
            label: "Companion Callout".into(),
            bus: "dialogue".into(),
            duration_ms: 400.0,
            contact_marker_ms: 20.0,
            contact_label: "Syllable Onset".into(),
            description: "Dialogue line that ducks foley".into(),
            layers: voice_layers,
        },
    );

    bank.sequences.insert(
        "seq_patrol".into(),
        ChoreographySequence {
            order: 0,
            label: "Patrol Stride + Callout".into(),
            steps: vec![
                SequenceStep {
                    event_id: "evt_footstep".into(),
                    at_ms: 0.0,
                    label: "Left Step".into(),
                },
                SequenceStep {
                    event_id: "evt_callout".into(),
                    at_ms: 90.0,
                    label: "Callout".into(),
                },
                SequenceStep {
                    event_id: "evt_footstep".into(),
                    at_ms: 220.0,
                    label: "Right Step".into(),
                },
            ],
        },
    );

    bank.dynamics.enabled = true;
    bank.dynamics.ducking.insert(
        "dialogue_ducks_foley".into(),
        DuckingRule {
            source_bus: "dialogue".into(),
            target_buses: vec!["foley".into(), "sfx".into()],
            enabled: true,
            amount: 0.55,
            attack_ms: 12.0,
            release_ms: 160.0,
        },
    );

    bank
}

#[test]
fn canonical_soundbank_json_is_fixed_point_and_validates_invariants() {
    let bank = sample_action_soundbank();
    let json_text = bank.to_canonical_json().expect("valid canonical soundbank");
    let parsed = SoundBank::from_json_bytes(json_text.as_bytes()).expect("roundtrip parse");
    assert_eq!(parsed, bank);
    let second_text = parsed.to_canonical_json().expect("second canonical write");
    assert_eq!(json_text, second_text);

    // Missing master bus is rejected
    let mut missing_master = bank.clone();
    missing_master.buses.remove(MASTER_BUS_ID);
    assert!(missing_master.validate().is_err());

    // Traversal asset path is rejected
    let mut traversal = bank.clone();
    traversal.assets.get_mut("step_stone_01").unwrap().path = "../secret.wav".into();
    assert!(traversal.validate().is_err());
}

#[test]
fn wav_and_qoa_codecs_roundtrip_multi_frame_stereo_signals() {
    // 6400 frames (> 1 QOA frame of 5120 samples) to exercise multi-frame QOA encoding/decoding
    let original = build_sine_buffer(48_000, 6_400, 440.0, 880.0, 0.72);

    let wav_bytes = encode_wav_pcm16(&original);
    let wav_decoded = decode_wav(&wav_bytes).expect("WAV decode");
    assert_eq!(wav_decoded.sample_rate, 48_000);
    assert_eq!(wav_decoded.frames.len(), 6_400);
    for (a, b) in original.frames.iter().zip(wav_decoded.frames.iter()) {
        assert!((a[0] - b[0]).abs() < 1e-4);
        assert!((a[1] - b[1]).abs() < 1e-4);
    }

    let qoa_bytes = encode_qoa(&original).expect("QOA encode");
    assert_eq!(&qoa_bytes[0..4], b"qoaf");
    // QOA compresses 16-bit stereo ~5:1 (6400 * 4 = 25600 bytes -> ~5200 bytes)
    assert!(qoa_bytes.len() < wav_bytes.len() / 3);

    let qoa_decoded = decode_qoa(&qoa_bytes).expect("QOA decode");
    assert_eq!(qoa_decoded.sample_rate, 48_000);
    assert_eq!(qoa_decoded.frames.len(), 6_400);

    // Compute signal-to-noise ratio across both channels (skipping 64-sample LMS warmup)
    let mut sig_pow = 0.0_f64;
    let mut err_pow = 0.0_f64;
    for i in 64..6_400 {
        for ch in 0..2 {
            let s = f64::from(original.frames[i][ch]);
            let e = s - f64::from(qoa_decoded.frames[i][ch]);
            sig_pow += s * s;
            err_pow += e * e;
        }
    }
    let snr_db = 10.0 * (sig_pow / err_pow.max(1e-12)).log10();
    assert!(
        snr_db > 25.0,
        "expected QOA SNR > 25 dB, got {snr_db:.2} dB"
    );

    // Auto-detection handles both formats
    assert_eq!(decode_audio(&wav_bytes).unwrap().frames.len(), 6_400);
    assert_eq!(decode_audio(&qoa_bytes).unwrap().frames.len(), 6_400);
}

#[test]
fn mid_side_width_and_phase_correlation_behave_accurately() {
    let mono_m = compute_mid_side_matrix(StereoMode::Stereo, 0.0);
    assert_eq!(mono_m, [[0.5, 0.5], [0.5, 0.5]]);

    let natural_m = compute_mid_side_matrix(StereoMode::Stereo, 100.0);
    assert_eq!(natural_m, [[1.0, 0.0], [0.0, 1.0]]);

    let wide_m = compute_mid_side_matrix(StereoMode::Stereo, 200.0);
    assert_eq!(wide_m, [[1.5, -0.5], [-0.5, 1.5]]);

    // Antiphase signal has -1.0 phase correlation, and collapses to silence at 0% width
    let mut antiphase: Vec<[f32; 2]> = (0..1_024)
        .map(|i| {
            let v = (2.0 * PI * 250.0 * (i as f32) / 48_000.0).sin() * 0.5;
            [v, -v]
        })
        .collect();
    let before = analyze_stereo_frames(&antiphase, 0.0);
    assert!(before.correlation < -0.99);

    let mut mono_proc = DspStripProcessor::new(
        &DspStrip {
            stereo_width_pct: 0.0,
            ..DspStrip::default()
        },
        48_000,
    );
    mono_proc.process_buffer_in_place(&mut antiphase);
    let after = analyze_stereo_frames(&antiphase, 0.0);
    assert!(after.peak_l_db <= -70.0 && after.peak_r_db <= -70.0);
}

#[test]
fn biquad_eq_compressor_overdrive_delay_and_reverb_process_deterministically() {
    let strip = DspStrip {
        high_pass: FilterBand {
            enabled: true,
            frequency_hz: 180.0,
            q: 0.707,
        },
        low_pass: FilterBand {
            enabled: true,
            frequency_hz: 8_000.0,
            q: 0.707,
        },
        eq: ParametricEq {
            enabled: true,
            low_gain_db: -3.0,
            low_freq_hz: 120.0,
            mid_gain_db: 6.0,
            mid_freq_hz: 1_500.0,
            mid_q: 1.4,
            high_gain_db: 2.0,
            high_freq_hz: 6_000.0,
        },
        compressor: CompressorSpec {
            enabled: true,
            threshold_db: -18.0,
            knee_db: 4.0,
            ratio: 6.0,
            attack_sec: 0.002,
            release_sec: 0.08,
            makeup_gain_db: 0.0,
            mix: 1.0,
        },
        overdrive: OverdriveSpec {
            enabled: true,
            drive: 45.0,
            tone_hz: 5_500.0,
            oversample: OversampleFactor::X4,
            output_gain_db: -1.5,
            mix: 0.5,
        },
        delay: DelaySpec {
            enabled: true,
            time_ms: 40.0,
            feedback: 0.35,
            lowpass_hz: 4_000.0,
            mix: 0.3,
        },
        reverb: ReverbSpec {
            enabled: true,
            room_size: 0.6,
            damping: 0.4,
            pre_delay_ms: 10.0,
            mix: 0.25,
        },
        ..DspStrip::default()
    };

    // Analytical frequency response reflects HPF roll-off at 40 Hz, mid boost at 1500 Hz, LPF cut at 16 kHz
    let resp = evaluate_strip_response_db(&strip, 48_000, &[40.0, 1_500.0, 16_000.0]);
    assert!(
        resp[0] < -15.0,
        "40 Hz should be attenuated by HPF + low-shelf, got {}",
        resp[0]
    );
    assert!(
        resp[1] > 4.5,
        "1500 Hz should be boosted by peaking EQ, got {}",
        resp[1]
    );
    assert!(
        resp[2] < -6.0,
        "16 kHz should roll off above 8 kHz LPF, got {}",
        resp[2]
    );

    // Burst signal exercises compressor GR + delay/reverb tail in trailing silence
    let mut frames = vec![[0.0_f32; 2]; 9_600]; // 200 ms @ 48 kHz
    for (i, f) in frames.iter_mut().enumerate().take(2_400) {
        let s = (2.0 * PI * 1_000.0 * (i as f32) / 48_000.0).sin() * 0.85;
        *f = [s, s];
    }
    let mut proc = DspStripProcessor::new(&strip, 48_000);
    proc.process_buffer_in_place(&mut frames);

    assert!(
        proc.peak_gain_reduction_db() > 4.0,
        "compressor should report significant GR on -1.4 dBFS burst, got {}",
        proc.peak_gain_reduction_db()
    );

    // Tail region (after sample 3000, once dry burst has stopped) must carry non-zero delay + reverb energy
    let tail_meter = analyze_stereo_frames(&frames[3_000..], 0.0);
    assert!(
        tail_meter.rms_l_db > -55.0 && tail_meter.rms_r_db > -55.0,
        "delay + Freeverb tail should remain audible after dry burst ends: {tail_meter:?}"
    );
}

#[test]
fn round_robin_curves_ducking_and_offline_48khz_bounce_work_end_to_end() {
    let bank = sample_action_soundbank();
    let pool = bank.containers.get("step_stone").unwrap();
    let mut selector = ContainerSelector::seeded(42);
    let mut prev: Option<String> = None;
    for _ in 0..80 {
        let pick = selector.pick("step_stone", pool).expect("take picked");
        assert_ne!(
            Some(pick.asset_id.as_str()),
            prev.as_deref(),
            "shuffle-no-repeat must never repeat adjacent takes when pool size >= 2"
        );
        prev = Some(pick.asset_id);
    }

    // Fade curves hit 0.0 at t=0, 0.5 at t=0.5, and 1.0 at t=1.0
    for curve in [FadeCurve::Linear, FadeCurve::Cosine, FadeCurve::SCurve] {
        assert!((evaluate_fade_curve(0.0, curve) - 0.0).abs() < 1e-6);
        assert!((evaluate_fade_curve(0.5, curve) - 0.5).abs() < 1e-5);
        assert!((evaluate_fade_curve(1.0, curve) - 1.0).abs() < 1e-6);
    }
    assert_eq!(clip_fade_envelope(0, 100, 20, 20, FadeCurve::SCurve), 0.0);
    assert_eq!(clip_fade_envelope(50, 100, 20, 20, FadeCurve::SCurve), 1.0);

    // Render single event & sequence with ducking and bounce to 48 kHz 16-bit WAV
    let assets = BTreeMap::new();
    let event_report =
        render_event_offline(&bank, "evt_footstep", &assets, 1337, 48_000).expect("render event");
    assert_eq!(event_report.master.sample_rate, 48_000);
    assert!(event_report.bus_meters["foley"].peak_l_db > -45.0);
    assert!(event_report.bus_meters["master"].peak_l_db > -45.0);

    let wav = bounce_event_to_wav16(&bank, "evt_footstep", &assets, 1337).expect("bounce event");
    let bounced_buf = decode_wav(&wav).expect("valid 48kHz 16-bit stereo WAV");
    assert_eq!(bounced_buf.sample_rate, 48_000);
    assert!(!bounced_buf.frames.is_empty());

    // Compare sequence foley energy with ducking ON vs OFF
    let seq_wav_ducked =
        bounce_sequence_to_wav16(&bank, "seq_patrol", &assets, 99).expect("bounce sequence");
    assert!(decode_wav(&seq_wav_ducked).unwrap().frames.len() > 24_000);

    let mut bank_unducked = bank.clone();
    bank_unducked.dynamics.enabled = false;
    let rep_ducked =
        ganbate_audio::render_sequence_offline(&bank, "seq_patrol", &assets, 99, 48_000).unwrap();
    let rep_unducked =
        ganbate_audio::render_sequence_offline(&bank_unducked, "seq_patrol", &assets, 99, 48_000)
            .unwrap();
    assert!(
        rep_ducked.bus_meters["foley"].rms_l_db < rep_unducked.bus_meters["foley"].rms_l_db,
        "dialogue sidechain ducking should attenuate foley RMS ({} vs {})",
        rep_ducked.bus_meters["foley"].rms_l_db,
        rep_unducked.bus_meters["foley"].rms_l_db
    );
}

#[test]
fn gbea_soundbank_cooker_roundtrips_deterministically_and_rejects_tampering() {
    let bank = sample_action_soundbank();
    let mut clips = BTreeMap::new();
    clips.insert(
        "step_stone_01".to_string(),
        build_sine_buffer(48_000, 1_200, 320.0, 480.0, 0.65),
    );
    clips.insert(
        "step_stone_02".to_string(),
        build_sine_buffer(48_000, 1_400, 360.0, 520.0, 0.60),
    );

    let gbea_a = ganbate_audio::cook_soundbank_gbea(&bank, &clips).expect("cook .gbea");
    let gbea_b = ganbate_audio::cook_soundbank_gbea(&bank, &clips).expect("deterministic .gbea");
    assert_eq!(gbea_a, gbea_b);
    assert_eq!(ganbate_audio::gbea_version(&gbea_a), Some(1));

    let unpacked = ganbate_audio::decode_soundbank_gbea(&gbea_a).expect("decode .gbea");
    assert_eq!(unpacked.bank, bank);
    assert_eq!(unpacked.clips.len(), 2);
    assert_eq!(unpacked.clips["step_stone_01"].frames.len(), 1_200);
    assert_eq!(unpacked.clips["step_stone_02"].frames.len(), 1_400);

    // Trailing byte must be rejected
    let mut with_trailing = gbea_a.clone();
    with_trailing.push(0);
    assert!(ganbate_audio::decode_soundbank_gbea(&with_trailing).is_err());

    // Undeclared clip id must be rejected at cook time
    let mut bad_clips = clips.clone();
    bad_clips.insert(
        "undeclared_clip".to_string(),
        build_sine_buffer(48_000, 200, 440.0, 440.0, 0.5),
    );
    assert!(ganbate_audio::cook_soundbank_gbea(&bank, &bad_clips).is_err());
}

#[test]
fn spatial_3d_attenuation_and_voice_pool_priority_stealing_work() {
    use ganbate_audio::{
        DistanceModel, Listener3D, SpatialEmitter3D, VoicePool, VoiceTriggerSpec,
        evaluate_spatial_3d,
    };
    use ganbate_core::Vec3;

    let listener = Listener3D::default(); // eye=(0,0,0), forward=(0,0,-1), right=(+1,0,0)
    let near_right = SpatialEmitter3D {
        position: Vec3(2.0, 0.0, -0.5),
        ref_distance: 1.0,
        max_distance: 30.0,
        rolloff_factor: 1.0,
        model: DistanceModel::InverseClamped,
        air_absorption: 0.5,
    };
    let far_left = SpatialEmitter3D {
        position: Vec3(-18.0, 0.0, -2.0),
        ..near_right
    };

    let m_near = evaluate_spatial_3d(&listener, &near_right);
    let m_far = evaluate_spatial_3d(&listener, &far_left);
    assert!(m_near.stereo_pan > 0.7, "right emitter should pan +R");
    assert!(m_far.stereo_pan < -0.7, "left emitter should pan -L");
    assert!(m_far.distance_gain < m_near.distance_gain);
    assert!(m_far.lowpass_cutoff_hz < m_near.lowpass_cutoff_hz);

    // VoicePool bounded capacity & priority stealing
    let bank = sample_action_soundbank();
    let mut pool = VoicePool::from_soundbank(&bank, 2);
    pool.set_listener(listener);
    let tone = build_sine_buffer(48_000, 2_048, 440.0, 440.0, 0.6);

    let v1 = pool
        .spawn_clip(
            &tone,
            VoiceTriggerSpec {
                bus: "foley".into(),
                priority: 50,
                spatial: Some(near_right),
                ..VoiceTriggerSpec::default()
            },
        )
        .expect("spawn v1");
    let _v2 = pool
        .spawn_clip(
            &tone,
            VoiceTriggerSpec {
                bus: "foley".into(),
                priority: 120,
                ..VoiceTriggerSpec::default()
            },
        )
        .expect("spawn v2");
    assert_eq!(pool.active_voice_count(), 2);

    // Lower-priority voice (priority 10) is rejected when pool is full of >=50 priority voices
    assert!(
        pool.spawn_clip(
            &tone,
            VoiceTriggerSpec {
                bus: "foley".into(),
                priority: 10,
                ..VoiceTriggerSpec::default()
            }
        )
        .is_none()
    );

    // Higher-priority voice (priority 200) steals lowest-priority v1 (priority 50)
    let v3 = pool
        .spawn_clip(
            &tone,
            VoiceTriggerSpec {
                bus: "foley".into(),
                priority: 200,
                spatial: Some(far_left),
                ..VoiceTriggerSpec::default()
            },
        )
        .expect("steal v1");
    assert!(!pool.stop_voice(v1), "v1 should have been stolen");
    assert_eq!(pool.active_voice_count(), 2);

    let block = pool.render_block(512);
    assert_eq!(block.frames.len(), 512);
    let meter = analyze_stereo_frames(&block.frames, 0.0);
    assert!(meter.peak_l_db > -50.0 || meter.peak_r_db > -50.0);
    assert!(pool.stop_voice(v3));
}
