//! A game's art direction as data: shading style, post-processing, sky palette, place and time,
//! named camera views. Engine defaults are a neutral soft-cel look; games override any field from
//! a JSON Look file so one renderer serves many styles.

use super::math::{V3, hex};
use super::sky::{Palette, SkyKey, TimeOfDay, Weather};
use ganbate_assets::json::Value;

/// Shading and post-processing parameters. Colours are linear RGB.
#[derive(Clone, Debug, PartialEq)]
pub struct Look {
    /// Half-Lambert position of the light/shadow boundary (0..1) and the width of the soft band.
    pub ramp_center: f32,
    pub ramp_softness: f32,
    /// Colour multiplying ambient light on the shadowed side, and how much darker shade gets.
    pub shadow_tint: V3,
    pub shadow_strength: f32,
    pub rim_strength: f32,
    pub rim_power: f32,
    pub specular: f32,
    pub specular_size: f32,
    /// Foliage: how far light wraps around the canopy, and back-lit glow.
    pub foliage_wrap: f32,
    pub foliage_translucency: f32,
    /// Positive values soften textures (sampling coarser mips) for a painted look.
    pub texture_softness: f32,
    pub normal_map_strength: f32,
    pub ambient_occlusion: f32,
    pub ao_radius: f32,
    pub ao_tint: V3,
    pub outline: f32,
    pub outline_color: V3,
    /// Painterly (Kuwahara) filter radius in pixels at 1080p; 0 disables it.
    pub painterly: f32,
    pub painterly_far: f32,
    pub bloom: f32,
    pub bloom_threshold: f32,
    pub exposure: f32,
    pub saturation: f32,
    pub contrast: f32,
    /// Grading tints added to shadows and multiplied into highlights.
    pub lift: V3,
    pub gain: V3,
    pub vignette: f32,
    /// Scale of point-light (lamp) intensity and of lamp-emission glow.
    pub lamp_light: f32,
    pub lamp_glow: f32,
    /// Scale of lamp light on foliage and grass (on top of `lamp_light`): blossom and leaves
    /// crowding a lantern otherwise burn white.
    pub lamp_foliage: f32,
    pub fog: f32,
    pub fog_height: f32,
    pub fog_falloff: f32,
    pub water_color: V3,
    pub water_reflection: f32,
    pub water_ripples: f32,
    pub cloud_scale: f32,
    pub star_density: f32,
    pub moon_size: f32,
    pub palette: Palette,
    /// Painted brushwork: how much world-space brush strokes break up the light/shade edge and
    /// flat colour (0 = none), and the stroke size in metres.
    pub brush: f32,
    pub brush_scale: f32,
    /// Final sharpening (0 = none).
    pub sharpen: f32,
    /// Light bounced up from sunlit (or moonlit) ground onto eaves, ceilings and the shaded
    /// undersides of things: the fraction of the key light added to the ground ambient.
    pub bounce: f32,
    /// Light shafts from the sun or moon through gaps in the scenery (0 = none).
    pub sun_shafts: f32,
    /// Tide marks: surfaces standing in the water are wet and stained up to `tide` metres above
    /// it, their colour multiplied by `tide_color` (0 = no marks).
    pub tide: f32,
    pub tide_color: V3,
    /// Shallow-water colour, shoreline foam strength and sun/moon glitter on water.
    pub water_shallow: V3,
    pub foam: f32,
    pub sparkle: f32,
    /// Procedural surface detail by material name; the first matching rule wins.
    pub details: Vec<DetailRule>,
    /// Per-material surface response overrides by material-name pattern; the first matching rule wins.
    pub surface_rules: Vec<SurfaceRule>,
    pub grass: Grass,
    /// Beaches, rock and meadow patches on materials with the [`DetailPreset::Terrain`] detail.
    pub terrain: Terrain,
    /// Default surface response model for opaque and alpha-blended scene materials.
    pub surface_model: SurfaceModel,
    /// Secondary clear-coat lobe weight (0..1) and roughness (0..1) when using `ClearCoat` or
    /// Look-level clear-coat styling.
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    /// Fabric sheen weight (0..1) when using `CharlieCloth` or Look-level sheen styling.
    pub sheen: f32,
    /// Subsurface wrap / scatter weight (0..1) and RGB tint (linear).
    pub subsurface: f32,
    pub subsurface_color: V3,
    /// Display-referred tone-mapping curve.
    pub tonemap: ToneMap,
    /// Optional scene-linear EV100 camera target (`13.0` is neutral daylight `1.0x` scale).
    pub ev100: Option<f32>,
    /// Optional physical camera settings (`aperture`, `shutter_speed`, `iso`, `compensation`).
    pub camera: Option<CameraExposure>,
}

/// Reference EV100 (`f/8`, `1/125 s`, `ISO 100`) mapped to unity exposure in Ganbate's sky units.
pub const REFERENCE_EV100: f32 = 13.0;

/// Converts an EV100 value into a linear exposure multiplier relative to [`REFERENCE_EV100`].
pub fn ev100_exposure_scale(ev100: f32) -> f32 {
    2.0f32.powf((REFERENCE_EV100 - ev100).clamp(-16.0, 16.0))
}

/// Physical camera settings mapped to EV100 exposure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraExposure {
    /// Relative aperture f-number (`N`, e.g. `8.0` for `f/8`).
    pub aperture: f32,
    /// Shutter duration in seconds (`t`, e.g. `0.008` for `1/125 s`).
    pub shutter_speed: f32,
    /// Sensor ISO sensitivity (`100.0` baseline).
    pub iso: f32,
    /// Exposure compensation in EV stops (`+1.0` doubles brightness).
    pub compensation: f32,
}

impl Default for CameraExposure {
    fn default() -> Self {
        Self {
            aperture: 8.0,
            shutter_speed: 1.0 / 125.0,
            iso: 100.0,
            compensation: 0.0,
        }
    }
}

impl CameraExposure {
    /// Computes effective EV100 (`log2(N^2 / t) - log2(ISO / 100) - compensation`).
    pub fn ev100(&self) -> f32 {
        let n = self.aperture.max(0.5);
        let t = self.shutter_speed.max(1e-6);
        let iso = self.iso.max(1.0);
        (n * n / t).log2() - (iso / 100.0).log2() - self.compensation
    }
}

/// Surface response model decoupling material parameters from art-directed or photoreal BRDFs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum SurfaceModel {
    /// Soft two-tone cel ramp with tinted shadows, toon specular, and fresnel rim.
    #[default]
    StylizedRamp = 0,
    /// Crisp cel band with warm terminator scatter, tangent highlight, and light-gated rim.
    StylizedCharacter = 1,
    /// Energy-conserving microfacet GGX specular with height-correlated Smith visibility and Burley diffuse.
    StandardGGX = 2,
    /// `StandardGGX` base coated with a dielectric (`F0 = 0.04`) secondary specular lobe.
    ClearCoat = 3,
    /// Estevez & Kulla Charlie sheen microfacet lobe for woven cloth, velvet, and felt-like fabrics.
    CharlieCloth = 4,
    /// Dual-lobe specular with wrapped RGB subsurface scattering and back-lit translucency.
    SubsurfaceSkin = 5,
}

impl SurfaceModel {
    pub const ALL: [Self; 6] = [
        Self::StylizedRamp,
        Self::StylizedCharacter,
        Self::StandardGGX,
        Self::ClearCoat,
        Self::CharlieCloth,
        Self::SubsurfaceSkin,
    ];

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "stylized" | "stylized_ramp" => Self::StylizedRamp,
            "stylized_character" | "character" => Self::StylizedCharacter,
            "standard_ggx" | "photoreal" | "ggx" => Self::StandardGGX,
            "clearcoat" | "clear_coat" => Self::ClearCoat,
            "charlie_cloth" | "cloth" => Self::CharlieCloth,
            "subsurface_skin" | "subsurface" | "skin" => Self::SubsurfaceSkin,
            _ => return None,
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StylizedRamp => "stylized_ramp",
            Self::StylizedCharacter => "stylized_character",
            Self::StandardGGX => "standard_ggx",
            Self::ClearCoat => "clearcoat",
            Self::CharlieCloth => "charlie_cloth",
            Self::SubsurfaceSkin => "subsurface_skin",
        }
    }
}

/// Display-referred tone-mapping operator applied before grading and output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum ToneMap {
    /// Warm Narkowicz ACES curve tuned for Ganbate's stylized palette (engine default).
    #[default]
    Stylized = 0,
    /// Troy Sobotka's AgX transform: smooth chroma-preserving highlight desaturation without hue skew.
    AgX = 1,
    /// Stephen Hill's sRGB-to-AP1 fitted ACES RRT+ODT transform for high-contrast photoreal scenes.
    ACESFitted = 2,
}

impl ToneMap {
    pub const ALL: [Self; 3] = [Self::Stylized, Self::AgX, Self::ACESFitted];

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "stylized" | "aces_stylized" => Self::Stylized,
            "agx" => Self::AgX,
            "aces" | "aces_fitted" => Self::ACESFitted,
            _ => return None,
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stylized => "stylized",
            Self::AgX => "agx",
            Self::ACESFitted => "aces_fitted",
        }
    }
}

/// A procedural surface pattern a Look file can give materials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetailPreset {
    None = 0,
    /// Rows of rounded roof tiles aligned to the slope, with lapped edges.
    RoofTiles = 1,
    /// Planks and grain.
    Wood = 2,
    /// Soft mottling and streaks.
    Plaster = 3,
    /// Irregular stones with dark joints.
    Stone = 4,
    /// Polished: glossier, with a sheen.
    Lacquer = 5,
    /// Grassy ground: grass grows on it (see [`Grass`]) and it takes the grass colour.
    Grass = 6,
    /// Open land: grassy ground in broad meadow patches, with beach sand near the water and bare
    /// rock on steep slopes, where no grass grows (see [`Terrain`]).
    Terrain = 7,
    /// A lattice of thin bars (a lantern's wooden frame, paper ribs): `scale` is the cell's
    /// width and height (0 leaves that direction's bars out: `[0, 0.05]` draws only ribs),
    /// `strength` the bar width as a fraction of a cell, `color` the bars' (dark wood by
    /// default). Bars also shade the material's glow.
    Lattice = 8,
}

impl DetailPreset {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "none" => Self::None,
            "roof_tiles" => Self::RoofTiles,
            "wood" => Self::Wood,
            "plaster" => Self::Plaster,
            "stone" => Self::Stone,
            "lacquer" => Self::Lacquer,
            "grass" => Self::Grass,
            "terrain" => Self::Terrain,
            "lattice" => Self::Lattice,
            _ => return None,
        })
    }
}

/// Which materials get which [`DetailPreset`], and how.
#[derive(Clone, Debug, PartialEq)]
pub struct DetailRule {
    /// Material-name patterns; `*` matches any run of characters.
    pub patterns: Vec<String>,
    pub preset: DetailPreset,
    /// Pattern size in metres: tile width and row length, plank width and length, stone size.
    pub scale: [f32; 2],
    /// How strongly the pattern shows (relief, contrast).
    pub strength: f32,
    /// Colour variation between tiles, planks or stones.
    pub variation: f32,
    /// Replaces the material's colour (linear RGB) when set. With the `none` preset the rule
    /// only recolours, and works on any material, foliage and water included.
    pub color: Option<V3>,
    /// Extra gloss (0..1), for lacquer and polished wood.
    pub gloss: f32,
}

impl DetailRule {
    /// True when one of the patterns matches `name`.
    pub fn matches(&self, name: &str) -> bool {
        self.patterns.iter().any(|p| glob(p, name))
    }
}

/// Per-material surface response override matched by material-name glob pattern.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceRule {
    /// Material-name patterns; `*` matches any run of characters.
    pub patterns: Vec<String>,
    pub surface: Option<SurfaceModel>,
    pub clearcoat: Option<f32>,
    pub clearcoat_roughness: Option<f32>,
    pub sheen: Option<f32>,
    pub sheen_tint: Option<f32>,
    pub subsurface: Option<f32>,
}

impl SurfaceRule {
    /// True when one of the patterns matches `name`.
    pub fn matches(&self, name: &str) -> bool {
        self.patterns.iter().any(|p| glob(p, name))
    }
}

/// `*` matches any run of characters; everything else matches itself.
pub fn glob(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == name;
    }
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !name.starts_with(first) || name.len() < first.len() + last.len() || !name.ends_with(last) {
        return false;
    }
    let mut rest = &name[first.len()..name.len() - last.len()];
    for part in &parts[1..parts.len() - 1] {
        match rest.find(part) {
            Some(i) => rest = &rest[i + part.len()..],
            None => return false,
        }
    }
    true
}

/// Grass and flowers grown on materials with the [`DetailPreset::Grass`] detail.
#[derive(Clone, Debug, PartialEq)]
pub struct Grass {
    /// Blade density (1 = the quality's default count, 0 = no grass).
    pub density: f32,
    /// Blade height and width in metres.
    pub height: f32,
    pub width: f32,
    /// Colours at the root and the tip (linear RGB).
    pub base: V3,
    pub tip: V3,
    /// Fraction of blades that carry a flower, and the flower colours.
    pub flowers: f32,
    pub flower_colors: [V3; 4],
    /// How far blades bend in the wind.
    pub sway: f32,
}

impl Default for Grass {
    fn default() -> Self {
        Self {
            density: 0.0,
            height: 0.38,
            width: 0.045,
            base: [0.06, 0.14, 0.03],
            tip: [0.34, 0.52, 0.12],
            flowers: 0.03,
            flower_colors: [
                [0.95, 0.95, 0.9],
                [1.0, 0.82, 0.25],
                [0.95, 0.5, 0.65],
                [0.65, 0.55, 0.95],
            ],
            sway: 1.0,
        }
    }
}

/// How [`DetailPreset::Terrain`] ground turns to sand, rock and meadow. Colours are linear RGB.
#[derive(Clone, Debug, PartialEq)]
pub struct Terrain {
    /// Sand lies within this many metres above the water line (0 = no beaches).
    pub beach: f32,
    /// Bare rock where the slope (1 - normal.y) passes from the first value to the second.
    pub rock_slope: [f32; 2],
    pub sand: V3,
    /// Sand at the water line, drying towards `sand` above it.
    pub wet_sand: V3,
    pub rock: V3,
    /// A second grass tone laid in broad patches about `patch_size` metres across, `patches`
    /// strong (0 = none).
    pub meadow: V3,
    pub patches: f32,
    pub patch_size: f32,
}

impl Default for Terrain {
    fn default() -> Self {
        Self {
            beach: 1.2,
            rock_slope: [0.4, 0.6],
            sand: [0.7, 0.6, 0.4],
            wet_sand: [0.33, 0.28, 0.19],
            rock: [0.3, 0.29, 0.27],
            meadow: [0.24, 0.36, 0.07],
            patches: 0.0,
            patch_size: 45.0,
        }
    }
}

impl Default for Look {
    fn default() -> Self {
        Self {
            ramp_center: 0.5,
            ramp_softness: 0.08,
            shadow_tint: [0.62, 0.68, 0.95],
            shadow_strength: 0.55,
            rim_strength: 0.35,
            rim_power: 3.5,
            specular: 0.35,
            specular_size: 0.25,
            foliage_wrap: 0.6,
            foliage_translucency: 0.5,
            texture_softness: 0.8,
            normal_map_strength: 0.5,
            ambient_occlusion: 0.65,
            ao_radius: 0.6,
            ao_tint: [0.28, 0.3, 0.5],
            outline: 0.25,
            outline_color: [0.05, 0.05, 0.12],
            painterly: 2.5,
            painterly_far: 2.0,
            bloom: 0.6,
            bloom_threshold: 1.0,
            exposure: 1.0,
            saturation: 1.1,
            contrast: 1.05,
            lift: [0.0, 0.004, 0.012],
            gain: [1.0, 1.0, 1.0],
            vignette: 0.25,
            lamp_light: 1.0,
            lamp_glow: 1.0,
            lamp_foliage: 1.0,
            fog: 1.0,
            fog_height: 0.0,
            fog_falloff: 0.08,
            water_color: [0.01, 0.06, 0.1],
            water_reflection: 0.85,
            water_ripples: 1.0,
            cloud_scale: 1.0,
            star_density: 1.0,
            moon_size: 1.0,
            palette: Palette::default(),
            brush: 0.3,
            brush_scale: 0.35,
            sharpen: 0.2,
            bounce: 0.1,
            sun_shafts: 0.5,
            tide: 0.0,
            tide_color: [0.35, 0.4, 0.3],
            water_shallow: [0.05, 0.2, 0.2],
            foam: 0.6,
            sparkle: 0.5,
            details: vec![],
            surface_rules: vec![],
            grass: Grass::default(),
            terrain: Terrain::default(),
            surface_model: SurfaceModel::StylizedRamp,
            clearcoat: 0.0,
            clearcoat_roughness: 0.15,
            sheen: 0.0,
            subsurface: 0.0,
            subsurface_color: [0.85, 0.28, 0.18],
            tonemap: ToneMap::Stylized,
            ev100: None,
            camera: None,
        }
    }
}

impl Look {
    /// The grade changes nothing: saturation and contrast 1, lift 0 and gain 1.
    pub fn grade_is_identity(&self) -> bool {
        self.saturation == 1.0
            && self.contrast == 1.0
            && self.lift == [0.0; 3]
            && self.gain == [1.0; 3]
    }

    /// Scene-linear exposure multiplier combining `exposure` with physical camera (`camera` or `ev100`).
    pub fn effective_exposure(&self) -> f32 {
        if let Some(cam) = &self.camera {
            self.exposure * ev100_exposure_scale(cam.ev100())
        } else if let Some(ev) = self.ev100 {
            self.exposure * ev100_exposure_scale(ev)
        } else {
            self.exposure
        }
    }
}

pub use ganbate_core::xr::{Focus, View};

/// A parsed Look file: the look plus where/when the scene is, its default weather and views.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct LookFile {
    pub look: Look,
    pub time: TimeOfDay,
    pub weather: Weather,
    pub views: Vec<View>,
}
fn num(v: &Value, key: &str, out: &mut f32) -> Result<(), String> {
    if let Some(x) = v.get(key) {
        *out = x.float().map_err(|e| format!("{key}: {e}"))?;
    }
    Ok(())
}
fn vec3(v: &Value) -> Result<V3, String> {
    match v {
        Value::String(s) => hex(s).ok_or_else(|| format!("bad colour {s:?} (want \"#rrggbb\")")),
        Value::Array(a) if a.len() == 3 => Ok([a[0].float()?, a[1].float()?, a[2].float()?]),
        _ => Err("expected [x, y, z] or \"#rrggbb\"".into()),
    }
}
fn col(v: &Value, key: &str, out: &mut V3) -> Result<(), String> {
    if let Some(x) = v.get(key) {
        *out = vec3(x).map_err(|e| format!("{key}: {e}"))?;
    }
    Ok(())
}

impl LookFile {
    /// Parse a Look file. Every field is optional; unknown keys are ignored so files can carry
    /// notes. Colours are `"#rrggbb"` (sRGB) or `[r, g, b]` (linear).
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        Self::from_value(&ganbate_assets::json::parse(bytes)?)
    }

    /// Parses an already decoded Look, for editors that validate a transaction before saving it.
    pub fn from_value(root: &Value) -> Result<Self, String> {
        let mut file = Self::default();
        if let Some(s) = root.get("shading") {
            let l = &mut file.look;
            num(s, "ramp_center", &mut l.ramp_center)?;
            num(s, "ramp_softness", &mut l.ramp_softness)?;
            col(s, "shadow_tint", &mut l.shadow_tint)?;
            num(s, "shadow_strength", &mut l.shadow_strength)?;
            num(s, "rim_strength", &mut l.rim_strength)?;
            num(s, "rim_power", &mut l.rim_power)?;
            num(s, "specular", &mut l.specular)?;
            num(s, "specular_size", &mut l.specular_size)?;
            num(s, "foliage_wrap", &mut l.foliage_wrap)?;
            num(s, "foliage_translucency", &mut l.foliage_translucency)?;
            num(s, "texture_softness", &mut l.texture_softness)?;
            num(s, "normal_map_strength", &mut l.normal_map_strength)?;
            num(s, "lamp_light", &mut l.lamp_light)?;
            num(s, "lamp_glow", &mut l.lamp_glow)?;
            num(s, "lamp_foliage", &mut l.lamp_foliage)?;
            col(s, "water_color", &mut l.water_color)?;
            num(s, "water_reflection", &mut l.water_reflection)?;
            num(s, "water_ripples", &mut l.water_ripples)?;
            col(s, "water_shallow", &mut l.water_shallow)?;
            num(s, "tide", &mut l.tide)?;
            col(s, "tide_color", &mut l.tide_color)?;
            num(s, "foam", &mut l.foam)?;
            num(s, "sparkle", &mut l.sparkle)?;
            num(s, "brush", &mut l.brush)?;
            num(s, "bounce", &mut l.bounce)?;
            num(s, "brush_scale", &mut l.brush_scale)?;
            if let Some(m) = s.get("model") {
                let name = m.string().map_err(|e| format!("shading.model: {e}"))?;
                l.surface_model = SurfaceModel::parse(name)
                    .ok_or_else(|| format!("unknown surface model {name:?}"))?;
            }
            num(s, "clearcoat", &mut l.clearcoat)?;
            num(s, "clearcoat_roughness", &mut l.clearcoat_roughness)?;
            num(s, "sheen", &mut l.sheen)?;
            num(s, "subsurface", &mut l.subsurface)?;
            col(s, "subsurface_color", &mut l.subsurface_color)?;
            l.clearcoat = l.clearcoat.clamp(0.0, 1.0);
            l.clearcoat_roughness = l.clearcoat_roughness.clamp(0.02, 1.0);
            l.sheen = l.sheen.clamp(0.0, 1.0);
            l.subsurface = l.subsurface.clamp(0.0, 1.0);
        }
        if let Some(p) = root.get("post") {
            let l = &mut file.look;
            num(p, "ambient_occlusion", &mut l.ambient_occlusion)?;
            num(p, "ao_radius", &mut l.ao_radius)?;
            col(p, "ao_tint", &mut l.ao_tint)?;
            num(p, "outline", &mut l.outline)?;
            col(p, "outline_color", &mut l.outline_color)?;
            num(p, "painterly", &mut l.painterly)?;
            num(p, "painterly_far", &mut l.painterly_far)?;
            num(p, "bloom", &mut l.bloom)?;
            num(p, "bloom_threshold", &mut l.bloom_threshold)?;
            num(p, "exposure", &mut l.exposure)?;
            num(p, "saturation", &mut l.saturation)?;
            num(p, "contrast", &mut l.contrast)?;
            col(p, "lift", &mut l.lift)?;
            col(p, "gain", &mut l.gain)?;
            num(p, "vignette", &mut l.vignette)?;
            num(p, "fog", &mut l.fog)?;
            num(p, "fog_height", &mut l.fog_height)?;
            num(p, "fog_falloff", &mut l.fog_falloff)?;
            num(p, "sharpen", &mut l.sharpen)?;
            num(p, "sun_shafts", &mut l.sun_shafts)?;
            if let Some(tm) = p.get("tonemap") {
                let name = tm.string().map_err(|e| format!("post.tonemap: {e}"))?;
                l.tonemap =
                    ToneMap::parse(name).ok_or_else(|| format!("unknown tonemap {name:?}"))?;
            }
            if let Some(ev) = p.get("ev100") {
                l.ev100 = Some(ev.float().map_err(|e| format!("post.ev100: {e}"))?);
            }
            if let Some(c) = p.get("camera") {
                let mut cam = CameraExposure::default();
                num(c, "aperture", &mut cam.aperture)?;
                num(c, "shutter_speed", &mut cam.shutter_speed)?;
                num(c, "iso", &mut cam.iso)?;
                num(c, "compensation", &mut cam.compensation)?;
                if !(cam.aperture > 0.0 && cam.shutter_speed > 0.0 && cam.iso > 0.0) {
                    return Err(
                        "post.camera: aperture, shutter_speed and iso must be positive".into(),
                    );
                }
                l.camera = Some(cam);
            }
        }
        if let Some(rules) = root.get("details") {
            for r in rules.array()? {
                file.look.details.push(parse_detail(r)?);
                if let Some(sr) = parse_surface_rule(r)? {
                    file.look.surface_rules.push(sr);
                }
            }
        }
        if let Some(rules) = root.get("surfaces") {
            for r in rules.array()? {
                if let Some(sr) = parse_surface_rule(r)? {
                    file.look.surface_rules.push(sr);
                }
            }
        }
        if let Some(g) = root.get("grass") {
            let grass = &mut file.look.grass;
            num(g, "density", &mut grass.density)?;
            num(g, "height", &mut grass.height)?;
            num(g, "width", &mut grass.width)?;
            col(g, "base", &mut grass.base)?;
            col(g, "tip", &mut grass.tip)?;
            num(g, "flowers", &mut grass.flowers)?;
            num(g, "sway", &mut grass.sway)?;
            if let Some(c) = g.get("flower_colors") {
                let c = c.array()?;
                if c.is_empty() {
                    return Err("grass.flower_colors is empty".into());
                }
                for (i, slot) in grass.flower_colors.iter_mut().enumerate() {
                    *slot = vec3(&c[i % c.len()]).map_err(|e| format!("flower_colors: {e}"))?;
                }
            }
        }
        if let Some(t) = root.get("terrain") {
            let terrain = &mut file.look.terrain;
            num(t, "beach", &mut terrain.beach)?;
            match t.get("rock_slope") {
                Some(Value::Array(a)) if a.len() == 2 => {
                    terrain.rock_slope = [a[0].float()?, a[1].float()?];
                }
                Some(_) => return Err("terrain.rock_slope: expected [start, full]".into()),
                None => {}
            }
            col(t, "sand", &mut terrain.sand)?;
            col(t, "wet_sand", &mut terrain.wet_sand)?;
            col(t, "rock", &mut terrain.rock)?;
            col(t, "meadow", &mut terrain.meadow)?;
            num(t, "patches", &mut terrain.patches)?;
            num(t, "patch_size", &mut terrain.patch_size)?;
        }
        if let Some(s) = root.get("sky") {
            let l = &mut file.look;
            num(s, "cloud_scale", &mut l.cloud_scale)?;
            num(s, "star_density", &mut l.star_density)?;
            num(s, "moon_size", &mut l.moon_size)?;
            col(s, "moon_color", &mut l.palette.moon)?;
            num(s, "moon_intensity", &mut l.palette.moon_intensity)?;
            num(s, "dusk_warmth", &mut l.palette.dusk_warmth)?;
            num(s, "dawn_mist", &mut l.palette.dawn_mist)?;
            if let Some(keys) = s.get("keys") {
                file.look.palette.keys = parse_keys(keys)?;
            }
        }
        if let Some(t) = root.get("time") {
            let time = &mut file.time;
            num(t, "hours", &mut time.hours)?;
            num(t, "day_of_year", &mut time.day_of_year)?;
            num(t, "latitude", &mut time.latitude_deg)?;
            num(t, "moon_lag_hours", &mut time.moon_lag_hours)?;
            if let Some(d) = t.get("moon_declination") {
                time.moon_declination = Some(d.float()?);
            }
            if let Some(n) = t.get("north") {
                time.north = vec3(n).map_err(|e| format!("time.north: {e}"))?;
            }
        }
        if let Some(w) = root.get("weather") {
            if let Some(name) = w.get("preset") {
                let name = name.string()?;
                file.weather =
                    Weather::preset(name).ok_or_else(|| format!("unknown weather {name:?}"))?;
            }
            let wt = &mut file.weather;
            num(w, "clouds", &mut wt.clouds)?;
            num(w, "rain", &mut wt.rain)?;
            num(w, "mist", &mut wt.mist)?;
            num(w, "wind", &mut wt.wind)?;
            num(w, "petals", &mut wt.petals)?;
        }
        if let Some(views) = root.get("views") {
            for v in views.array()? {
                let name = v
                    .get("name")
                    .ok_or("view without name")?
                    .string()?
                    .to_owned();
                let eye = vec3(v.get("eye").ok_or("view without eye")?)?;
                let target = vec3(v.get("target").ok_or("view without target")?)?;
                let mut view = View {
                    name,
                    eye,
                    target,
                    ..View::default()
                };
                num(v, "fov", &mut view.fov)?;
                num(v, "near", &mut view.near)?;
                if let Some(up) = v.get("up") {
                    view.up = vec3(up)?;
                }
                if let Some(shift) = v.get("shift") {
                    let s = shift.array()?;
                    if s.len() != 2 {
                        return Err("view shift: expected [x, y]".into());
                    }
                    view.shift = [s[0].float()?, s[1].float()?];
                }
                if let Some(f) = v.get("focus") {
                    let mut focus = Focus {
                        distance: f32::NAN,
                        depth: 4.0,
                        blur: 0.006,
                    };
                    // The distance has no default: a missing one is refused below.
                    num(f, "distance", &mut focus.distance)?;
                    num(f, "depth", &mut focus.depth)?;
                    num(f, "blur", &mut focus.blur)?;
                    if !focus.blurs() || focus.distance < 0.0 {
                        return Err(format!("view {:?}: focus out of range", view.name));
                    }
                    view.focus = Some(focus);
                }
                if !(view.near > 0.0 && view.fov > 0.0 && view.fov < 180.0) {
                    return Err(format!("view {:?}: fov or near out of range", view.name));
                }
                file.views.push(view);
            }
        }
        Ok(file)
    }
}

fn parse_patterns(r: &Value) -> Result<Vec<String>, String> {
    match r.get("match").ok_or("detail rule without \"match\"")? {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(a) => a
            .iter()
            .map(|v| v.string().map(str::to_owned))
            .collect::<Result<_, _>>(),
        _ => Err("detail \"match\" must be a string or an array of strings".into()),
    }
}

fn parse_detail(r: &Value) -> Result<DetailRule, String> {
    let patterns = parse_patterns(r)?;
    let name = r
        .get("preset")
        .ok_or("detail rule without \"preset\"")?
        .string()?;
    let preset =
        DetailPreset::parse(name).ok_or_else(|| format!("unknown detail preset {name:?}"))?;
    let mut rule = DetailRule {
        patterns,
        preset,
        scale: [0.3, 0.3],
        strength: 1.0,
        variation: 0.1,
        color: None,
        gloss: 0.0,
    };
    match r.get("scale") {
        Some(Value::Array(a)) if a.len() == 2 => rule.scale = [a[0].float()?, a[1].float()?],
        Some(v) => {
            let x = v.float().map_err(|e| format!("detail scale: {e}"))?;
            rule.scale = [x, x];
        }
        None => {}
    }
    num(r, "strength", &mut rule.strength)?;
    num(r, "variation", &mut rule.variation)?;
    num(r, "gloss", &mut rule.gloss)?;
    if let Some(c) = r.get("color") {
        rule.color = Some(vec3(c).map_err(|e| format!("detail color: {e}"))?);
    }
    Ok(rule)
}

fn parse_surface_rule(r: &Value) -> Result<Option<SurfaceRule>, String> {
    let surface = match r.get("surface") {
        Some(s) => {
            let name = s.string().map_err(|e| format!("detail surface: {e}"))?;
            Some(SurfaceModel::parse(name).ok_or_else(|| format!("unknown surface {name:?}"))?)
        }
        None => None,
    };
    let opt_clamped = |key: &str, lo: f32, hi: f32| -> Result<Option<f32>, String> {
        r.get(key)
            .map(|v| {
                v.float()
                    .map(|x| x.clamp(lo, hi))
                    .map_err(|e| format!("detail {key}: {e}"))
            })
            .transpose()
    };
    let clearcoat = opt_clamped("clearcoat", 0.0, 1.0)?;
    let clearcoat_roughness = opt_clamped("clearcoat_roughness", 0.02, 1.0)?;
    let sheen = opt_clamped("sheen", 0.0, 1.0)?;
    let sheen_tint = opt_clamped("sheen_tint", 0.0, 1.0)?;
    let subsurface = opt_clamped("subsurface", 0.0, 1.0)?;
    if surface.is_none()
        && clearcoat.is_none()
        && clearcoat_roughness.is_none()
        && sheen.is_none()
        && sheen_tint.is_none()
        && subsurface.is_none()
    {
        return Ok(None);
    }
    Ok(Some(SurfaceRule {
        patterns: parse_patterns(r)?,
        surface,
        clearcoat,
        clearcoat_roughness,
        sheen,
        sheen_tint,
        subsurface,
    }))
}

fn parse_keys(v: &Value) -> Result<Vec<SkyKey>, String> {
    let mut keys = vec![];
    let base = Palette::default();
    for k in v.array()? {
        let mut altitude = 0.0;
        num(k, "altitude", &mut altitude)?;
        let mut key = base.keys[0];
        // Start from the default key nearest in altitude so partial keys stay sensible.
        if let Some(near) = base.keys.iter().min_by(|a, b| {
            (a.altitude - altitude)
                .abs()
                .total_cmp(&(b.altitude - altitude).abs())
        }) {
            key = *near;
        }
        key.altitude = altitude;
        col(k, "zenith", &mut key.zenith)?;
        col(k, "horizon", &mut key.horizon)?;
        col(k, "glow", &mut key.glow)?;
        col(k, "ambient_sky", &mut key.ambient_sky)?;
        col(k, "ambient_ground", &mut key.ambient_ground)?;
        col(k, "fog", &mut key.fog)?;
        col(k, "cloud_lit", &mut key.cloud_lit)?;
        col(k, "cloud_shadow", &mut key.cloud_shadow)?;
        col(k, "sun", &mut key.sun)?;
        num(k, "sun_intensity", &mut key.sun_intensity)?;
        num(k, "ambient", &mut key.ambient)?;
        num(k, "stars", &mut key.stars)?;
        num(k, "exposure", &mut key.exposure)?;
        keys.push(key);
    }
    if keys.is_empty() {
        return Err("sky.keys is empty".into());
    }
    keys.sort_by(|a, b| a.altitude.total_cmp(&b.altitude));
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_overrides_and_views() {
        let f = LookFile::parse(
            br##"{"shading": {"rim_strength": 0.7, "shadow_tint": "#808080", "lamp_foliage": 0.5,
                             "tide": 0.6},
                 "time": {"hours": 6.5, "north": [0, 0, 1]},
                 "weather": {"preset": "rain", "wind": 2},
                 "views": [{"name": "a", "eye": [0, 1, 2], "target": [0, 0, 0], "fov": 40}]}"##,
        )
        .unwrap();
        assert_eq!(f.look.rim_strength, 0.7);
        assert_eq!((f.look.lamp_foliage, f.look.tide), (0.5, 0.6));
        assert!((f.look.shadow_tint[0] - 0.2158).abs() < 1e-3);
        assert_eq!(f.time.hours, 6.5);
        assert_eq!(f.weather.rain, Weather::preset("rain").unwrap().rain);
        assert_eq!(f.weather.wind, 2.0);
        assert_eq!(f.views[0].fov, 40.0);
        assert_eq!(f.views[0].up, [0.0, 1.0, 0.0]);
        assert_eq!((f.views[0].shift, f.views[0].near), ([0.0; 2], 0.08));
        assert_eq!(f.views[0].focus, None);
    }

    #[test]
    fn view_focus() {
        let f = LookFile::parse(
            br##"{"views": [{"name": "stage", "eye": [0, 2, 3], "target": [0, 0, 0],
                             "focus": {"distance": 3.5}},
                            {"name": "deep", "eye": [0, 2, 3], "target": [0, 0, 0],
                             "focus": {"distance": 2, "depth": 6, "blur": 0.01}}]}"##,
        )
        .unwrap();
        let focus = |i: usize| f.views[i].focus.map(|f| (f.distance, f.depth, f.blur));
        assert_eq!(focus(0), Some((3.5, 4.0, 0.006)));
        assert_eq!(focus(1), Some((2.0, 6.0, 0.01)));
        for bad in [
            &br##"{"views": [{"name": "a", "eye": [0, 1, 2], "target": [0, 0, 0],
                             "focus": {"distance": -1}}]}"##[..],
            br##"{"views": [{"name": "a", "eye": [0, 1, 2], "target": [0, 0, 0],
                            "focus": {"distance": 2, "depth": 0}}]}"##,
            br##"{"views": [{"name": "a", "eye": [0, 1, 2], "target": [0, 0, 0],
                            "focus": {"distance": 2, "blur": 0}}]}"##,
            br##"{"views": [{"name": "a", "eye": [0, 1, 2], "target": [0, 0, 0],
                            "focus": {"depth": 3}}]}"##,
        ] {
            assert!(LookFile::parse(bad).is_err());
        }
    }

    #[test]
    fn view_up_shift_and_near() {
        let f = LookFile::parse(
            br##"{"views": [{"name": "top", "eye": [0, 9, 0], "target": [0, 0, 0],
                             "up": [0, 0, -1], "shift": [0, 0.12], "near": 0.05}]}"##,
        )
        .unwrap();
        let v = &f.views[0];
        assert_eq!(
            (v.up, v.shift, v.near),
            ([0.0, 0.0, -1.0], [0.0, 0.12], 0.05)
        );
        for bad in [
            &br##"{"views": [{"name": "a", "eye": [0, 1, 2], "target": [0, 0, 0], "shift": [1]}]}"##[..],
            br##"{"views": [{"name": "a", "eye": [0, 1, 2], "target": [0, 0, 0], "near": 0}]}"##,
            br##"{"views": [{"name": "a", "eye": [0, 1, 2], "target": [0, 0, 0], "fov": 180}]}"##,
        ] {
            assert!(LookFile::parse(bad).is_err());
        }
    }

    #[test]
    fn detail_rules_and_grass() {
        let f = LookFile::parse(
            br##"{"details": [
                    {"match": ["roof-*", "*-tile"], "preset": "roof_tiles", "scale": [0.25, 0.3]},
                    {"match": "lawn", "preset": "grass", "color": "#4a7a30"},
                    {"match": "lamp", "preset": "lattice", "scale": [0.1, 0.05]}],
                  "grass": {"density": 0.8, "flower_colors": ["#ffffff"]}}"##,
        )
        .unwrap();
        let d = &f.look.details;
        assert_eq!(d[0].preset, DetailPreset::RoofTiles);
        assert!(d[0].matches("roof-main") && d[0].matches("blue-tile") && !d[0].matches("tiles"));
        assert_eq!(d[0].scale, [0.25, 0.3]);
        assert!(d[1].matches("lawn") && !d[1].matches("lawns") && d[1].color.is_some());
        assert_eq!(
            (d[2].preset, d[2].scale),
            (DetailPreset::Lattice, [0.1, 0.05])
        );
        assert_eq!(f.look.grass.density, 0.8);
        assert_eq!(f.look.grass.flower_colors[3], [1.0, 1.0, 1.0]);
        assert!(LookFile::parse(br#"{"details": [{"match": "a", "preset": "fur"}]}"#).is_err());
    }

    #[test]
    fn terrain_block() {
        let f = LookFile::parse(
            br##"{"details": [{"match": "land", "preset": "terrain"}],
                  "terrain": {"beach": 0.8, "rock_slope": [0.3, 0.5], "rock": "#808080",
                              "patches": 0.4}}"##,
        )
        .unwrap();
        assert_eq!(f.look.details[0].preset, DetailPreset::Terrain);
        let t = &f.look.terrain;
        assert_eq!((t.beach, t.rock_slope, t.patches), (0.8, [0.3, 0.5], 0.4));
        assert!((t.rock[0] - 0.2158).abs() < 1e-3);
        assert_eq!(t.sand, Terrain::default().sand);
        assert!(LookFile::parse(br#"{"terrain": {"rock_slope": 0.5}}"#).is_err());
    }

    #[test]
    fn glob_patterns() {
        assert!(glob("a*", "abc") && glob("*c", "abc") && glob("a*c", "abc") && glob("*", ""));
        assert!(glob("a*b*c", "a-b-c") && !glob("a*b*c", "a-c") && !glob("ab*ba", "aba"));
        assert!(glob("exact", "exact") && !glob("exact", "exactly"));
    }

    #[test]
    fn surface_models_tonemap_and_ev100_camera() {
        for m in SurfaceModel::ALL {
            assert_eq!(SurfaceModel::parse(m.as_str()), Some(m));
        }
        assert_eq!(
            SurfaceModel::parse("photoreal"),
            Some(SurfaceModel::StandardGGX)
        );
        for tm in ToneMap::ALL {
            assert_eq!(ToneMap::parse(tm.as_str()), Some(tm));
        }
        assert_eq!(ToneMap::parse("aces"), Some(ToneMap::ACESFitted));

        let sunny = CameraExposure {
            aperture: 8.0,
            shutter_speed: 1.0 / 128.0,
            iso: 100.0,
            compensation: 0.0,
        };
        assert!((sunny.ev100() - 13.0).abs() < 1e-4);
        assert!((ev100_exposure_scale(13.0) - 1.0).abs() < 1e-5);
        assert!((ev100_exposure_scale(12.0) - 2.0).abs() < 1e-5);
        assert!((ev100_exposure_scale(14.0) - 0.5).abs() < 1e-5);

        let f = LookFile::parse(
            br##"{"shading": {"model": "clearcoat", "clearcoat": 0.9, "clearcoat_roughness": 0.08,
                             "sheen": 0.4, "subsurface": 0.6, "subsurface_color": "#ff6644"},
                 "post": {"tonemap": "agx",
                          "camera": {"aperture": 5.656854, "shutter_speed": 0.0078125, "iso": 100, "compensation": 0.0}},
                 "details": [{"match": "cloth-*", "preset": "none", "surface": "charlie_cloth",
                              "sheen": 0.85, "sheen_tint": 0.5}]}"##,
        )
        .unwrap();
        assert_eq!(f.look.surface_model, SurfaceModel::ClearCoat);
        assert_eq!((f.look.clearcoat, f.look.clearcoat_roughness), (0.9, 0.08));
        assert_eq!(f.look.tonemap, ToneMap::AgX);
        // f/5.656854 at 1/128s ISO 100 => EV100 = 12.0 => 2.0x exposure scale.
        assert!((f.look.effective_exposure() - 2.0).abs() < 1e-3);
        assert_eq!(
            f.look.surface_rules[0].surface,
            Some(SurfaceModel::CharlieCloth)
        );
        assert_eq!(
            (
                f.look.surface_rules[0].sheen,
                f.look.surface_rules[0].sheen_tint
            ),
            (Some(0.85), Some(0.5))
        );

        assert!(LookFile::parse(br#"{"shading": {"model": "unknown"}}"#).is_err());
        assert!(LookFile::parse(br#"{"post": {"tonemap": "reinhard"}}"#).is_err());
        assert!(LookFile::parse(br#"{"post": {"camera": {"aperture": 0}}}"#).is_err());
    }
}
