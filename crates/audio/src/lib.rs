//! Ganbate Engine's deterministic, genre-neutral Event-Driven Audio Engine (`ganbate-audio`).
//!
//! Zero third-party Cargo dependencies:
//! - `schema`: Canonical `.soundbank.json` document parser, validator, and `ganbate_core::json` writer
//! - `codec`: RIFF/WAVE (8/16/24/32-bit PCM + 32-bit float) and QOA v1.0 encoder/decoder + Hermite resampler
//! - `cooked`: Deterministic `GBEA` (`.gbea`) binary SoundBank archive cooker and bounded decoder
//! - `spatial`: 3D listener/emitter distance rolloff, azimuth stereo pan, and air-absorption LPF
//! - `dsp`: Gain, Pan, 2x2 Mid/Side Width (`0%..200%`), Biquad HPF/LPF/3-Band EQ, Compressor (with GR),
//!   Oversampled Soft-Clip Overdrive, Stereo Delay, and Freeverb/Schroeder Room Reverb
//! - `meter`: Sample-accurate Stereo Peak dBFS, RMS dBFS, Clipping, and L/R Phase Correlation (`-1..+1`)
//! - `mixer`: Real-time bounded `VoicePool`, Multi-take Round-Robin (`shuffle-no-repeat` / `sequential`),
//!   curved fades (`linear`, `cosine`, `s-curve`), sidechain ducking, and offline `48 kHz` WAV bounce

pub mod codec;
pub mod cooked;
pub mod dsp;
pub mod meter;
pub mod mixer;
pub mod schema;
pub mod spatial;

pub use codec::{AudioBuffer, decode_audio, decode_qoa, decode_wav, encode_qoa, encode_wav_pcm16};
pub use cooked::{
    CookedSoundBank, GBEA_MAGIC, GBEA_VERSION, GbeaCodec, MAX_GBEA_BYTES, MAX_GBEA_CLIPS,
    cook_soundbank_gbea, cook_soundbank_gbea_with_codec, decode_soundbank_gbea, gbea_version,
};
pub use dsp::{
    BiquadCoeffs, DspStripProcessor, apply_stereo_pan, compute_mid_side_matrix,
    evaluate_strip_response_db, overdrive_transfer_sample,
};
pub use meter::{ChannelMeterSnapshot, analyze_stereo_frames, db_to_linear, linear_to_db};
pub use mixer::{
    AudioRng, ContainerSelector, MixRenderReport, SelectedTake, VoiceHandle, VoicePool,
    VoiceTriggerSpec, bounce_event_to_wav16, bounce_sequence_to_wav16, clip_fade_envelope,
    evaluate_fade_curve, render_event_offline, render_sequence_offline,
};
pub use schema::{
    BusRole, BusSpec, ChoreographySequence, CompressorSpec, DelaySpec, DspStrip, DuckingRule,
    EventLayer, EventSheet, FadeCurve, FilterBand, MASTER_BUS_ID, MixPreset, OverdriveSpec,
    OversampleFactor, ParametricEq, PresetBusPatch, ReverbSpec, RoundRobinMode, SOUNDBANK_KIND,
    SOUNDBANK_VERSION, SequenceStep, SoundAsset, SoundBank, SoundBankDynamics, StereoMode,
    TakeContainer,
};
pub use spatial::{
    DistanceModel, Listener3D, SpatialEmitter3D, SpatialMixParams, evaluate_spatial_3d,
};
