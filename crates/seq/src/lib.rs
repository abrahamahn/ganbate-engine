//! Deterministic multi-track timeline (`ganbate.sequence/1`) synchronizing camera cuts/rails,
//! skeletal animation clips, parametric/spring 3D transform curves, particle VFX bursts, and
//! event-driven audio cues on one shared millisecond + frame clock.
//!
//! Portable and zero-dependency: uses `ganbate-core` for canonical JSON read/write.

pub mod curve;

pub use curve::CurveSpec;
use ganbate_core::json::{self, Value};
use std::collections::BTreeMap;

pub const SEQUENCE_SCHEMA: &str = "ganbate.sequence/1";

pub type V3 = [f32; 3];

#[derive(Clone, Debug, PartialEq)]
pub struct Marker {
    pub id: String,
    pub label: String,
    pub at_ms: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CameraSection {
    pub id: String,
    pub start_ms: u32,
    pub end_ms: u32,
    pub view: String,
    pub from_eye: V3,
    pub to_eye: V3,
    pub from_target: V3,
    pub to_target: V3,
    pub fov_from: f32,
    pub fov_to: f32,
    pub curve: CurveSpec,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SkeletalSection {
    pub id: String,
    pub clip: String,
    pub secondary_clip: Option<String>,
    pub start_ms: u32,
    pub end_ms: u32,
    pub clip_offset_s: f32,
    pub speed: f32,
    pub weight: f32,
    pub blend_in_ms: u32,
    pub blend_out_ms: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransformSection {
    pub id: String,
    pub start_ms: u32,
    pub end_ms: u32,
    pub from_pos: V3,
    pub to_pos: V3,
    pub arc_lift: f32,
    pub from_euler_deg: V3,
    pub to_euler_deg: V3,
    pub from_scale: V3,
    pub to_scale: V3,
    pub curve: CurveSpec,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VfxTrigger {
    pub id: String,
    pub effect: String,
    pub at_ms: u32,
    pub duration_ms: u32,
    pub scale: f32,
    pub burst_multiplier: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AudioTrigger {
    pub id: String,
    pub cue: String,
    pub at_ms: u32,
    pub gain_db: f32,
    pub pitch_semitones: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Track {
    Camera {
        id: String,
        label: String,
        muted: bool,
        sections: Vec<CameraSection>,
    },
    SkeletalAnim {
        id: String,
        label: String,
        target: String,
        muted: bool,
        sections: Vec<SkeletalSection>,
    },
    TransformCurve {
        id: String,
        label: String,
        target: String,
        muted: bool,
        sections: Vec<TransformSection>,
    },
    Vfx {
        id: String,
        label: String,
        target: String,
        muted: bool,
        triggers: Vec<VfxTrigger>,
    },
    Audio {
        id: String,
        label: String,
        bus: String,
        muted: bool,
        triggers: Vec<AudioTrigger>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sequence {
    pub name: String,
    pub duration_ms: u32,
    pub fps: u32,
    pub looping: bool,
    pub markers: Vec<Marker>,
    pub tracks: Vec<Track>,
}

impl Default for Sequence {
    fn default() -> Self {
        Self {
            name: "Untitled Sequence".into(),
            duration_ms: 3000,
            fps: 60,
            looping: false,
            markers: Vec::new(),
            tracks: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedCamera {
    pub section_id: String,
    pub view: String,
    pub eye: V3,
    pub target: V3,
    pub fov: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedSkeletalLayer {
    pub target: String,
    pub section_id: String,
    pub clip: String,
    pub secondary_clip: Option<String>,
    pub clip_time_s: f32,
    pub weight: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedTransform {
    pub target: String,
    pub section_id: String,
    pub position: V3,
    pub euler_deg: V3,
    pub scale: V3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActiveVfxSpan {
    pub target: String,
    pub trigger_id: String,
    pub effect: String,
    pub elapsed_s: f32,
    pub progress: f32,
    pub scale: f32,
    pub burst_multiplier: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SequenceFrame {
    pub time_ms: f32,
    pub frame_index: u32,
    pub camera: Option<EvaluatedCamera>,
    pub skeletal: Vec<EvaluatedSkeletalLayer>,
    pub transforms: Vec<EvaluatedTransform>,
    pub active_vfx: Vec<ActiveVfxSpan>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FiredAudioCue {
    pub bus: String,
    pub trigger: AudioTrigger,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FiredVfxBurst {
    pub target: String,
    pub trigger: VfxTrigger,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TriggerSlice {
    pub audio: Vec<FiredAudioCue>,
    pub vfx: Vec<FiredVfxBurst>,
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp3(a: V3, b: V3, t: f32) -> V3 {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

fn hermite01(t: f32) -> f32 {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

impl Sequence {
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("sequence name must not be empty".into());
        }
        if self.duration_ms == 0 {
            return Err("sequence duration_ms must be > 0".into());
        }
        if self.fps == 0 || self.fps > 240 {
            return Err("sequence fps must lie in 1..=240".into());
        }
        for m in &self.markers {
            if m.at_ms > self.duration_ms {
                return Err(format!("marker {:?} exceeds duration_ms", m.id));
            }
        }
        for track in &self.tracks {
            match track {
                Track::Camera { sections, .. } => {
                    for s in sections {
                        if s.end_ms <= s.start_ms || s.end_ms > self.duration_ms {
                            return Err(format!("camera section {:?} has invalid bounds", s.id));
                        }
                        if !(1.0..=175.0).contains(&s.fov_from)
                            || !(1.0..=175.0).contains(&s.fov_to)
                        {
                            return Err(format!("camera section {:?} fov out of range", s.id));
                        }
                        s.curve.validate()?;
                    }
                }
                Track::SkeletalAnim {
                    target, sections, ..
                } => {
                    if target.trim().is_empty() {
                        return Err("skeletal track target must not be empty".into());
                    }
                    for s in sections {
                        if s.end_ms <= s.start_ms || s.end_ms > self.duration_ms {
                            return Err(format!("skeletal section {:?} has invalid bounds", s.id));
                        }
                        if s.clip.trim().is_empty() || !s.speed.is_finite() || s.speed < 0.0 {
                            return Err(format!(
                                "skeletal section {:?} has invalid clip/speed",
                                s.id
                            ));
                        }
                    }
                }
                Track::TransformCurve {
                    target, sections, ..
                } => {
                    if target.trim().is_empty() {
                        return Err("transform track target must not be empty".into());
                    }
                    for s in sections {
                        if s.end_ms <= s.start_ms || s.end_ms > self.duration_ms {
                            return Err(format!("transform section {:?} has invalid bounds", s.id));
                        }
                        s.curve.validate()?;
                    }
                }
                Track::Vfx {
                    target, triggers, ..
                } => {
                    if target.trim().is_empty() {
                        return Err("vfx track target must not be empty".into());
                    }
                    for tr in triggers {
                        if tr.effect.trim().is_empty() || tr.at_ms > self.duration_ms {
                            return Err(format!("vfx trigger {:?} is invalid", tr.id));
                        }
                    }
                }
                Track::Audio { bus, triggers, .. } => {
                    if bus.trim().is_empty() {
                        return Err("audio track bus must not be empty".into());
                    }
                    for tr in triggers {
                        if tr.cue.trim().is_empty() || tr.at_ms > self.duration_ms {
                            return Err(format!("audio trigger {:?} is invalid", tr.id));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn normalize_time_ms(&self, raw_ms: f32) -> f32 {
        if !raw_ms.is_finite() || self.duration_ms == 0 {
            return 0.0;
        }
        let dur = self.duration_ms as f32;
        if self.looping {
            raw_ms.rem_euclid(dur)
        } else {
            raw_ms.clamp(0.0, dur)
        }
    }

    /// Snaps a millisecond timestamp to the nearest frame boundary at `self.fps`.
    pub fn snap_to_frame_ms(&self, raw_ms: f32) -> f32 {
        let clamped = self.normalize_time_ms(raw_ms);
        let step = 1000.0 / (self.fps.max(1) as f32);
        ((clamped / step).round() * step).clamp(0.0, self.duration_ms as f32)
    }

    /// Deterministic evaluation of all non-muted tracks at `raw_ms`.
    pub fn evaluate(&self, raw_ms: f32) -> SequenceFrame {
        let time_ms = self.normalize_time_ms(raw_ms);
        let frame_index = ((time_ms / 1000.0) * (self.fps.max(1) as f32)).floor() as u32;
        let mut camera = None;
        let mut skeletal = Vec::new();
        let mut transforms = Vec::new();
        let mut active_vfx = Vec::new();

        for track in &self.tracks {
            match track {
                Track::Camera {
                    muted: false,
                    sections,
                    ..
                } => {
                    for s in sections {
                        let (a, b) = (s.start_ms as f32, s.end_ms as f32);
                        if time_ms >= a && time_ms <= b {
                            let span_ms = (b - a).max(1e-3);
                            let raw_t = (time_ms - a) / span_ms;
                            let eased = s.curve.sample(raw_t, span_ms * 0.001);
                            camera = Some(EvaluatedCamera {
                                section_id: s.id.clone(),
                                view: s.view.clone(),
                                eye: lerp3(s.from_eye, s.to_eye, eased),
                                target: lerp3(s.from_target, s.to_target, eased),
                                fov: lerp(s.fov_from, s.fov_to, eased),
                            });
                        }
                    }
                }
                Track::SkeletalAnim {
                    target,
                    muted: false,
                    sections,
                    ..
                } => {
                    for s in sections {
                        let (a, b) = (s.start_ms as f32, s.end_ms as f32);
                        if time_ms >= a && time_ms <= b {
                            let elapsed_ms = time_ms - a;
                            let remain_ms = b - time_ms;
                            let fade_in = if s.blend_in_ms > 0 {
                                hermite01(elapsed_ms / (s.blend_in_ms as f32))
                            } else {
                                1.0
                            };
                            let fade_out = if s.blend_out_ms > 0 {
                                hermite01(remain_ms / (s.blend_out_ms as f32))
                            } else {
                                1.0
                            };
                            let w = (s.weight * fade_in * fade_out).clamp(0.0, 1.0);
                            if w > 0.0 {
                                skeletal.push(EvaluatedSkeletalLayer {
                                    target: target.clone(),
                                    section_id: s.id.clone(),
                                    clip: s.clip.clone(),
                                    secondary_clip: s.secondary_clip.clone(),
                                    clip_time_s: s.clip_offset_s + (elapsed_ms * 0.001) * s.speed,
                                    weight: w,
                                });
                            }
                        }
                    }
                }
                Track::TransformCurve {
                    target,
                    muted: false,
                    sections,
                    ..
                } => {
                    // Hold last finished section on this target or interpolate active section.
                    let mut chosen: Option<EvaluatedTransform> = None;
                    for s in sections {
                        let (a, b) = (s.start_ms as f32, s.end_ms as f32);
                        if time_ms >= a && time_ms <= b {
                            let span_ms = (b - a).max(1e-3);
                            let raw_t = ((time_ms - a) / span_ms).clamp(0.0, 1.0);
                            let eased = s.curve.sample(raw_t, span_ms * 0.001);
                            let mut pos = lerp3(s.from_pos, s.to_pos, eased);
                            // Parabolic flight arc peaking at raw_t = 0.5 and settling to 0 at endpoints.
                            pos[1] += 4.0 * raw_t * (1.0 - raw_t) * s.arc_lift;
                            chosen = Some(EvaluatedTransform {
                                target: target.clone(),
                                section_id: s.id.clone(),
                                position: pos,
                                euler_deg: lerp3(s.from_euler_deg, s.to_euler_deg, eased),
                                scale: lerp3(s.from_scale, s.to_scale, eased),
                            });
                        } else if time_ms > b {
                            chosen = Some(EvaluatedTransform {
                                target: target.clone(),
                                section_id: s.id.clone(),
                                position: s.to_pos,
                                euler_deg: s.to_euler_deg,
                                scale: s.to_scale,
                            });
                        }
                    }
                    if let Some(eval) = chosen {
                        transforms.push(eval);
                    }
                }
                Track::Vfx {
                    target,
                    muted: false,
                    triggers,
                    ..
                } => {
                    for tr in triggers {
                        let start = tr.at_ms as f32;
                        let dur = tr.duration_ms.max(1) as f32;
                        let end = start + dur;
                        if time_ms >= start && time_ms <= end {
                            let elapsed_ms = time_ms - start;
                            active_vfx.push(ActiveVfxSpan {
                                target: target.clone(),
                                trigger_id: tr.id.clone(),
                                effect: tr.effect.clone(),
                                elapsed_s: elapsed_ms * 0.001,
                                progress: (elapsed_ms / dur).clamp(0.0, 1.0),
                                scale: tr.scale,
                                burst_multiplier: tr.burst_multiplier,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        SequenceFrame {
            time_ms,
            frame_index,
            camera,
            skeletal,
            transforms,
            active_vfx,
        }
    }

    /// Collects discrete `Audio` and `VFX` triggers firing in `(prev_ms, curr_ms]` (inclusive of
    /// `0` when starting from `prev_ms < 0.0`, and handling loop wrap-around when `self.looping`).
    pub fn triggers_between(&self, prev_ms: f32, curr_ms: f32) -> TriggerSlice {
        let mut out = TriggerSlice::default();
        if !curr_ms.is_finite() || self.duration_ms == 0 {
            return out;
        }
        let dur = self.duration_ms as f32;
        let in_window = |at_ms: u32| -> bool {
            let t = at_ms as f32;
            if !prev_ms.is_finite() || prev_ms < 0.0 {
                return t >= 0.0 && t <= curr_ms;
            }
            if !self.looping {
                return t > prev_ms && t <= curr_ms.min(dur);
            }
            if curr_ms < prev_ms {
                return false;
            }
            if curr_ms - prev_ms >= dur {
                return true;
            }
            let a = prev_ms.rem_euclid(dur);
            let b = curr_ms.rem_euclid(dur);
            if a <= b {
                t > a && t <= b
            } else {
                t > a || t <= b
            }
        };

        for track in &self.tracks {
            match track {
                Track::Audio {
                    bus,
                    muted: false,
                    triggers,
                    ..
                } => {
                    for tr in triggers {
                        if in_window(tr.at_ms) {
                            out.audio.push(FiredAudioCue {
                                bus: bus.clone(),
                                trigger: tr.clone(),
                            });
                        }
                    }
                }
                Track::Vfx {
                    target,
                    muted: false,
                    triggers,
                    ..
                } => {
                    for tr in triggers {
                        if in_window(tr.at_ms) {
                            out.vfx.push(FiredVfxBurst {
                                target: target.clone(),
                                trigger: tr.clone(),
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let root = json::parse(bytes)?;
        Self::from_json_value(&root)
    }

    pub fn from_json_str(raw: &str) -> Result<Self, String> {
        Self::from_json_bytes(raw.as_bytes())
    }

    pub fn from_json_value(root: &Value) -> Result<Self, String> {
        let schema = root
            .get("schema")
            .ok_or_else(|| "missing schema field".to_string())?
            .string()?;
        if schema != SEQUENCE_SCHEMA {
            return Err(format!(
                "expected schema {SEQUENCE_SCHEMA:?}, got {schema:?}"
            ));
        }
        let name = str_or(root, "name", "Untitled Sequence")?;
        let duration_ms = u32_or(root, "durationMs", 3000)?;
        let fps = u32_or(root, "fps", 60)?;
        let looping = bool_or(root, "looping", false)?;

        let mut markers = Vec::new();
        if let Some(arr) = root.get("markers") {
            for item in arr.array()? {
                markers.push(Marker {
                    id: str_req(item, "id")?,
                    label: str_or(item, "label", "")?,
                    at_ms: u32_req(item, "atMs")?,
                });
            }
        }

        let mut tracks = Vec::new();
        if let Some(arr) = root.get("tracks") {
            for item in arr.array()? {
                let kind = str_req(item, "kind")?;
                let id = str_req(item, "id")?;
                let label = str_or(item, "label", &id)?;
                let muted = bool_or(item, "muted", false)?;
                match kind.as_str() {
                    "camera" => {
                        let mut sections = Vec::new();
                        if let Some(sec_val) = item.get("sections") {
                            for s in sec_val.array()? {
                                sections.push(CameraSection {
                                    id: str_req(s, "id")?,
                                    start_ms: u32_req(s, "startMs")?,
                                    end_ms: u32_req(s, "endMs")?,
                                    view: str_or(s, "view", "main")?,
                                    from_eye: v3_or(s, "fromEye", [0.0, 1.6, 3.0])?,
                                    to_eye: v3_or(s, "toEye", [0.0, 1.6, 3.0])?,
                                    from_target: v3_or(s, "fromTarget", [0.0, 1.0, 0.0])?,
                                    to_target: v3_or(s, "toTarget", [0.0, 1.0, 0.0])?,
                                    fov_from: f32_or(s, "fovFrom", 45.0)?,
                                    fov_to: f32_or(s, "fovTo", 45.0)?,
                                    curve: parse_curve(s.get("curve"))?,
                                });
                            }
                        }
                        tracks.push(Track::Camera {
                            id,
                            label,
                            muted,
                            sections,
                        });
                    }
                    "skeletal" => {
                        let target = str_req(item, "target")?;
                        let mut sections = Vec::new();
                        if let Some(sec_val) = item.get("sections") {
                            for s in sec_val.array()? {
                                let secondary_clip = match s.get("secondaryClip") {
                                    Some(Value::String(sc)) if !sc.is_empty() => Some(sc.clone()),
                                    _ => None,
                                };
                                sections.push(SkeletalSection {
                                    id: str_req(s, "id")?,
                                    clip: str_req(s, "clip")?,
                                    secondary_clip,
                                    start_ms: u32_req(s, "startMs")?,
                                    end_ms: u32_req(s, "endMs")?,
                                    clip_offset_s: f32_or(s, "clipOffsetS", 0.0)?,
                                    speed: f32_or(s, "speed", 1.0)?,
                                    weight: f32_or(s, "weight", 1.0)?,
                                    blend_in_ms: u32_or(s, "blendInMs", 0)?,
                                    blend_out_ms: u32_or(s, "blendOutMs", 0)?,
                                });
                            }
                        }
                        tracks.push(Track::SkeletalAnim {
                            id,
                            label,
                            target,
                            muted,
                            sections,
                        });
                    }
                    "transform" => {
                        let target = str_req(item, "target")?;
                        let mut sections = Vec::new();
                        if let Some(sec_val) = item.get("sections") {
                            for s in sec_val.array()? {
                                sections.push(TransformSection {
                                    id: str_req(s, "id")?,
                                    start_ms: u32_req(s, "startMs")?,
                                    end_ms: u32_req(s, "endMs")?,
                                    from_pos: v3_or(s, "fromPos", [0.0; 3])?,
                                    to_pos: v3_or(s, "toPos", [0.0; 3])?,
                                    arc_lift: f32_or(s, "arcLift", 0.0)?,
                                    from_euler_deg: v3_or(s, "fromEulerDeg", [0.0; 3])?,
                                    to_euler_deg: v3_or(s, "toEulerDeg", [0.0; 3])?,
                                    from_scale: v3_or(s, "fromScale", [1.0; 3])?,
                                    to_scale: v3_or(s, "toScale", [1.0; 3])?,
                                    curve: parse_curve(s.get("curve"))?,
                                });
                            }
                        }
                        tracks.push(Track::TransformCurve {
                            id,
                            label,
                            target,
                            muted,
                            sections,
                        });
                    }
                    "vfx" => {
                        let target = str_req(item, "target")?;
                        let mut triggers = Vec::new();
                        if let Some(tr_val) = item.get("triggers") {
                            for tr in tr_val.array()? {
                                triggers.push(VfxTrigger {
                                    id: str_req(tr, "id")?,
                                    effect: str_req(tr, "effect")?,
                                    at_ms: u32_req(tr, "atMs")?,
                                    duration_ms: u32_or(tr, "durationMs", 1000)?,
                                    scale: f32_or(tr, "scale", 1.0)?,
                                    burst_multiplier: f32_or(tr, "burstMultiplier", 1.0)?,
                                });
                            }
                        }
                        tracks.push(Track::Vfx {
                            id,
                            label,
                            target,
                            muted,
                            triggers,
                        });
                    }
                    "audio" => {
                        let bus = str_or(item, "bus", "sfx")?;
                        let mut triggers = Vec::new();
                        if let Some(tr_val) = item.get("triggers") {
                            for tr in tr_val.array()? {
                                triggers.push(AudioTrigger {
                                    id: str_req(tr, "id")?,
                                    cue: str_req(tr, "cue")?,
                                    at_ms: u32_req(tr, "atMs")?,
                                    gain_db: f32_or(tr, "gainDb", 0.0)?,
                                    pitch_semitones: f32_or(tr, "pitchSemitones", 0.0)?,
                                });
                            }
                        }
                        tracks.push(Track::Audio {
                            id,
                            label,
                            bus,
                            muted,
                            triggers,
                        });
                    }
                    other => return Err(format!("unknown sequence track kind {other:?}")),
                }
            }
        }

        let seq = Self {
            name,
            duration_ms,
            fps,
            looping,
            markers,
            tracks,
        };
        seq.validate()?;
        Ok(seq)
    }

    pub fn to_json_value(&self) -> Value {
        let markers = self
            .markers
            .iter()
            .map(|m| {
                Value::Object(BTreeMap::from([
                    ("atMs".into(), num_u32(m.at_ms)),
                    ("id".into(), Value::String(m.id.clone())),
                    ("label".into(), Value::String(m.label.clone())),
                ]))
            })
            .collect();

        let tracks = self
            .tracks
            .iter()
            .map(|t| match t {
                Track::Camera {
                    id,
                    label,
                    muted,
                    sections,
                } => Value::Object(BTreeMap::from([
                    ("id".into(), Value::String(id.clone())),
                    ("kind".into(), Value::String("camera".into())),
                    ("label".into(), Value::String(label.clone())),
                    ("muted".into(), Value::Bool(*muted)),
                    (
                        "sections".into(),
                        Value::Array(
                            sections
                                .iter()
                                .map(|s| {
                                    Value::Object(BTreeMap::from([
                                        ("curve".into(), curve_to_json(&s.curve)),
                                        ("endMs".into(), num_u32(s.end_ms)),
                                        ("fovFrom".into(), num_f32(s.fov_from)),
                                        ("fovTo".into(), num_f32(s.fov_to)),
                                        ("fromEye".into(), v3_json(s.from_eye)),
                                        ("fromTarget".into(), v3_json(s.from_target)),
                                        ("id".into(), Value::String(s.id.clone())),
                                        ("startMs".into(), num_u32(s.start_ms)),
                                        ("toEye".into(), v3_json(s.to_eye)),
                                        ("toTarget".into(), v3_json(s.to_target)),
                                        ("view".into(), Value::String(s.view.clone())),
                                    ]))
                                })
                                .collect(),
                        ),
                    ),
                ])),
                Track::SkeletalAnim {
                    id,
                    label,
                    target,
                    muted,
                    sections,
                } => Value::Object(BTreeMap::from([
                    ("id".into(), Value::String(id.clone())),
                    ("kind".into(), Value::String("skeletal".into())),
                    ("label".into(), Value::String(label.clone())),
                    ("muted".into(), Value::Bool(*muted)),
                    (
                        "sections".into(),
                        Value::Array(
                            sections
                                .iter()
                                .map(|s| {
                                    let mut map = BTreeMap::from([
                                        ("blendInMs".into(), num_u32(s.blend_in_ms)),
                                        ("blendOutMs".into(), num_u32(s.blend_out_ms)),
                                        ("clip".into(), Value::String(s.clip.clone())),
                                        ("clipOffsetS".into(), num_f32(s.clip_offset_s)),
                                        ("endMs".into(), num_u32(s.end_ms)),
                                        ("id".into(), Value::String(s.id.clone())),
                                        ("speed".into(), num_f32(s.speed)),
                                        ("startMs".into(), num_u32(s.start_ms)),
                                        ("weight".into(), num_f32(s.weight)),
                                    ]);
                                    if let Some(sec) = &s.secondary_clip {
                                        map.insert(
                                            "secondaryClip".into(),
                                            Value::String(sec.clone()),
                                        );
                                    }
                                    Value::Object(map)
                                })
                                .collect(),
                        ),
                    ),
                    ("target".into(), Value::String(target.clone())),
                ])),
                Track::TransformCurve {
                    id,
                    label,
                    target,
                    muted,
                    sections,
                } => Value::Object(BTreeMap::from([
                    ("id".into(), Value::String(id.clone())),
                    ("kind".into(), Value::String("transform".into())),
                    ("label".into(), Value::String(label.clone())),
                    ("muted".into(), Value::Bool(*muted)),
                    (
                        "sections".into(),
                        Value::Array(
                            sections
                                .iter()
                                .map(|s| {
                                    Value::Object(BTreeMap::from([
                                        ("arcLift".into(), num_f32(s.arc_lift)),
                                        ("curve".into(), curve_to_json(&s.curve)),
                                        ("endMs".into(), num_u32(s.end_ms)),
                                        ("fromEulerDeg".into(), v3_json(s.from_euler_deg)),
                                        ("fromPos".into(), v3_json(s.from_pos)),
                                        ("fromScale".into(), v3_json(s.from_scale)),
                                        ("id".into(), Value::String(s.id.clone())),
                                        ("startMs".into(), num_u32(s.start_ms)),
                                        ("toEulerDeg".into(), v3_json(s.to_euler_deg)),
                                        ("toPos".into(), v3_json(s.to_pos)),
                                        ("toScale".into(), v3_json(s.to_scale)),
                                    ]))
                                })
                                .collect(),
                        ),
                    ),
                    ("target".into(), Value::String(target.clone())),
                ])),
                Track::Vfx {
                    id,
                    label,
                    target,
                    muted,
                    triggers,
                } => Value::Object(BTreeMap::from([
                    ("id".into(), Value::String(id.clone())),
                    ("kind".into(), Value::String("vfx".into())),
                    ("label".into(), Value::String(label.clone())),
                    ("muted".into(), Value::Bool(*muted)),
                    ("target".into(), Value::String(target.clone())),
                    (
                        "triggers".into(),
                        Value::Array(
                            triggers
                                .iter()
                                .map(|tr| {
                                    Value::Object(BTreeMap::from([
                                        ("atMs".into(), num_u32(tr.at_ms)),
                                        ("burstMultiplier".into(), num_f32(tr.burst_multiplier)),
                                        ("durationMs".into(), num_u32(tr.duration_ms)),
                                        ("effect".into(), Value::String(tr.effect.clone())),
                                        ("id".into(), Value::String(tr.id.clone())),
                                        ("scale".into(), num_f32(tr.scale)),
                                    ]))
                                })
                                .collect(),
                        ),
                    ),
                ])),
                Track::Audio {
                    id,
                    label,
                    bus,
                    muted,
                    triggers,
                } => Value::Object(BTreeMap::from([
                    ("bus".into(), Value::String(bus.clone())),
                    ("id".into(), Value::String(id.clone())),
                    ("kind".into(), Value::String("audio".into())),
                    ("label".into(), Value::String(label.clone())),
                    ("muted".into(), Value::Bool(*muted)),
                    (
                        "triggers".into(),
                        Value::Array(
                            triggers
                                .iter()
                                .map(|tr| {
                                    Value::Object(BTreeMap::from([
                                        ("atMs".into(), num_u32(tr.at_ms)),
                                        ("cue".into(), Value::String(tr.cue.clone())),
                                        ("gainDb".into(), num_f32(tr.gain_db)),
                                        ("id".into(), Value::String(tr.id.clone())),
                                        ("pitchSemitones".into(), num_f32(tr.pitch_semitones)),
                                    ]))
                                })
                                .collect(),
                        ),
                    ),
                ])),
            })
            .collect();

        Value::Object(BTreeMap::from([
            ("durationMs".into(), num_u32(self.duration_ms)),
            ("fps".into(), num_u32(self.fps)),
            ("looping".into(), Value::Bool(self.looping)),
            ("markers".into(), Value::Array(markers)),
            ("name".into(), Value::String(self.name.clone())),
            ("schema".into(), Value::String(SEQUENCE_SCHEMA.into())),
            ("tracks".into(), Value::Array(tracks)),
        ]))
    }

    pub fn to_canonical_json(&self) -> Result<String, String> {
        self.validate()?;
        json::to_canonical(&self.to_json_value())
    }
}

fn num_u32(v: u32) -> Value {
    Value::Number(f64::from(v))
}

fn num_f32(v: f32) -> Value {
    Value::Number(f64::from(v))
}

fn v3_json(v: V3) -> Value {
    Value::Array(vec![num_f32(v[0]), num_f32(v[1]), num_f32(v[2])])
}

fn str_req(obj: &Value, key: &str) -> Result<String, String> {
    Ok(obj
        .get(key)
        .ok_or_else(|| format!("missing string {key:?}"))?
        .string()?
        .to_string())
}

fn str_or(obj: &Value, key: &str, default: &str) -> Result<String, String> {
    match obj.get(key) {
        Some(v) => Ok(v.string()?.to_string()),
        None => Ok(default.to_string()),
    }
}

fn u32_req(obj: &Value, key: &str) -> Result<u32, String> {
    let n = obj
        .get(key)
        .ok_or_else(|| format!("missing integer {key:?}"))?
        .unsigned()?;
    u32::try_from(n).map_err(|_| format!("{key:?} exceeds u32"))
}

fn u32_or(obj: &Value, key: &str, default: u32) -> Result<u32, String> {
    match obj.get(key) {
        Some(v) => u32::try_from(v.unsigned()?).map_err(|_| format!("{key:?} exceeds u32")),
        None => Ok(default),
    }
}

fn f32_or(obj: &Value, key: &str, default: f32) -> Result<f32, String> {
    match obj.get(key) {
        Some(v) => v.float(),
        None => Ok(default),
    }
}

fn bool_or(obj: &Value, key: &str, default: bool) -> Result<bool, String> {
    match obj.get(key) {
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(format!("expected bool for {key:?}")),
        None => Ok(default),
    }
}

fn v3_or(obj: &Value, key: &str, default: V3) -> Result<V3, String> {
    let Some(v) = obj.get(key) else {
        return Ok(default);
    };
    let arr = v.array()?;
    if arr.len() != 3 {
        return Err(format!("{key:?} needs 3 floats"));
    }
    Ok([arr[0].float()?, arr[1].float()?, arr[2].float()?])
}

fn parse_curve(val: Option<&Value>) -> Result<CurveSpec, String> {
    let Some(val) = val else {
        return Ok(CurveSpec::SmoothStep);
    };
    if let Value::String(kind) = val {
        return match kind.as_str() {
            "linear" => Ok(CurveSpec::Linear),
            "smoothstep" => Ok(CurveSpec::SmoothStep),
            "easeOutQuart" => Ok(CurveSpec::EaseOutQuart),
            other => Err(format!("unknown curve {other:?}")),
        };
    }
    let kind = str_or(val, "type", "smoothstep")?;
    let spec = match kind.as_str() {
        "linear" => CurveSpec::Linear,
        "smoothstep" => CurveSpec::SmoothStep,
        "easeOutQuart" => CurveSpec::EaseOutQuart,
        "cubicBezier" => CurveSpec::CubicBezier {
            x1: f32_or(val, "x1", 0.25)?,
            y1: f32_or(val, "y1", 0.1)?,
            x2: f32_or(val, "x2", 0.25)?,
            y2: f32_or(val, "y2", 1.0)?,
        },
        "spring" => CurveSpec::Spring {
            stiffness: f32_or(val, "stiffness", 240.0)?,
            damping: f32_or(val, "damping", 22.0)?,
            mass: f32_or(val, "mass", 1.0)?,
        },
        other => return Err(format!("unknown curve type {other:?}")),
    };
    spec.validate()?;
    Ok(spec)
}

fn curve_to_json(curve: &CurveSpec) -> Value {
    match *curve {
        CurveSpec::Linear => Value::Object(BTreeMap::from([(
            "type".into(),
            Value::String("linear".into()),
        )])),
        CurveSpec::SmoothStep => Value::Object(BTreeMap::from([(
            "type".into(),
            Value::String("smoothstep".into()),
        )])),
        CurveSpec::EaseOutQuart => Value::Object(BTreeMap::from([(
            "type".into(),
            Value::String("easeOutQuart".into()),
        )])),
        CurveSpec::CubicBezier { x1, y1, x2, y2 } => Value::Object(BTreeMap::from([
            ("type".into(), Value::String("cubicBezier".into())),
            ("x1".into(), num_f32(x1)),
            ("x2".into(), num_f32(x2)),
            ("y1".into(), num_f32(y1)),
            ("y2".into(), num_f32(y2)),
        ])),
        CurveSpec::Spring {
            stiffness,
            damping,
            mass,
        } => Value::Object(BTreeMap::from([
            ("damping".into(), num_f32(damping)),
            ("mass".into(), num_f32(mass)),
            ("stiffness".into(), num_f32(stiffness)),
            ("type".into(), Value::String("spring".into())),
        ])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sequence() -> Sequence {
        Sequence {
            name: "Showcase Cutscene".into(),
            duration_ms: 2000,
            fps: 60,
            looping: true,
            markers: vec![Marker {
                id: "m_impact".into(),
                label: "Impact".into(),
                at_ms: 500,
            }],
            tracks: vec![
                Track::Camera {
                    id: "trk_cam".into(),
                    label: "Main Camera".into(),
                    muted: false,
                    sections: vec![CameraSection {
                        id: "cam_push".into(),
                        start_ms: 0,
                        end_ms: 1000,
                        view: "hero_shot".into(),
                        from_eye: [0.0, 2.0, 4.0],
                        to_eye: [0.0, 1.5, 2.0],
                        from_target: [0.0, 1.0, 0.0],
                        to_target: [0.0, 1.2, 0.0],
                        fov_from: 50.0,
                        fov_to: 38.0,
                        curve: CurveSpec::SmoothStep,
                    }],
                },
                Track::SkeletalAnim {
                    id: "trk_actor".into(),
                    label: "Actor Rig".into(),
                    target: "actor_primary".into(),
                    muted: false,
                    sections: vec![
                        SkeletalSection {
                            id: "sec_idle".into(),
                            clip: "actor_idle".into(),
                            secondary_clip: None,
                            start_ms: 0,
                            end_ms: 600,
                            clip_offset_s: 0.0,
                            speed: 1.0,
                            weight: 1.0,
                            blend_in_ms: 0,
                            blend_out_ms: 200,
                        },
                        SkeletalSection {
                            id: "sec_gesture".into(),
                            clip: "actor_gesture".into(),
                            secondary_clip: Some("actor_gesture_alt".into()),
                            start_ms: 400,
                            end_ms: 1600,
                            clip_offset_s: 0.1,
                            speed: 1.0,
                            weight: 1.0,
                            blend_in_ms: 200,
                            blend_out_ms: 100,
                        },
                    ],
                },
                Track::TransformCurve {
                    id: "trk_prop".into(),
                    label: "Prop Flight".into(),
                    target: "prop_token".into(),
                    muted: false,
                    sections: vec![TransformSection {
                        id: "prop_arc".into(),
                        start_ms: 200,
                        end_ms: 700,
                        from_pos: [0.0, 0.5, 0.0],
                        to_pos: [1.0, 0.5, 1.0],
                        arc_lift: 0.3,
                        from_euler_deg: [0.0, 0.0, 0.0],
                        to_euler_deg: [0.0, 180.0, 0.0],
                        from_scale: [1.0, 1.0, 1.0],
                        to_scale: [1.0, 1.0, 1.0],
                        curve: CurveSpec::CubicBezier {
                            x1: 0.22,
                            y1: 1.0,
                            x2: 0.36,
                            y2: 1.0,
                        },
                    }],
                },
                Track::Vfx {
                    id: "trk_vfx".into(),
                    label: "Impact Sparks".into(),
                    target: "socket_center".into(),
                    muted: false,
                    triggers: vec![VfxTrigger {
                        id: "vfx_1".into(),
                        effect: "content/vfx/radial-sparks.vfx.json".into(),
                        at_ms: 700,
                        duration_ms: 500,
                        scale: 1.2,
                        burst_multiplier: 1.5,
                    }],
                },
                Track::Audio {
                    id: "trk_sfx".into(),
                    label: "Foley".into(),
                    bus: "foley".into(),
                    muted: false,
                    triggers: vec![AudioTrigger {
                        id: "aud_1".into(),
                        cue: "prop.land".into(),
                        at_ms: 700,
                        gain_db: -1.5,
                        pitch_semitones: 0.0,
                    }],
                },
            ],
        }
    }

    #[test]
    fn canonical_json_roundtrip_and_multi_track_evaluation() {
        let seq = sample_sequence();
        let text = seq.to_canonical_json().expect("serialize");
        let parsed = Sequence::from_json_bytes(text.as_bytes()).expect("parse");
        assert_eq!(parsed, seq);
        assert_eq!(parsed.to_canonical_json().expect("re-serialize"), text);

        // At 500ms: camera halfway, both skeletal clips cross-fading at 0.5 weight, prop in flight.
        let frame = parsed.evaluate(500.0);
        assert_eq!(frame.frame_index, 30);
        let cam = frame.camera.expect("camera active");
        assert!((cam.eye[2] - 3.0).abs() < 1e-4);
        assert!((cam.fov - 44.0).abs() < 1e-4);
        assert_eq!(frame.skeletal.len(), 2);
        assert!((frame.skeletal[0].weight - 0.5).abs() < 1e-4);
        assert!((frame.skeletal[1].weight - 0.5).abs() < 1e-4);
        assert_eq!(frame.transforms.len(), 1);
        assert!(frame.transforms[0].position[1] > 0.75);

        // Triggers fire cleanly across [650, 750] and across loop wrap [1900, 2750].
        let fired = parsed.triggers_between(650.0, 750.0);
        assert_eq!(fired.audio.len(), 1);
        assert_eq!(fired.vfx.len(), 1);
        assert_eq!(parsed.triggers_between(700.0, 750.0).audio.len(), 0);
        let wrapped = parsed.triggers_between(1900.0, 2750.0);
        assert_eq!(wrapped.audio.len(), 1);
    }

    #[test]
    fn spring_and_bezier_curves_settle_cleanly() {
        let spring = CurveSpec::Spring {
            stiffness: 320.0,
            damping: 18.0,
            mass: 1.0,
        };
        assert_eq!(spring.sample(0.0, 0.5), 0.0);
        assert_eq!(spring.sample(1.0, 0.5), 1.0);
        // Underdamped spring overshoots 1.0 mid-flight.
        let mid = (1..90)
            .map(|i| spring.sample(i as f32 / 100.0, 0.6))
            .fold(0.0f32, f32::max);
        assert!(mid > 1.01, "expected underdamped overshoot, got {mid}");
    }
}
