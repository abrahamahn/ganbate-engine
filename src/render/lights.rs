//! Lights a game places while it runs (a stage spotlight over foreground props, a lantern it hangs):
//! point lights and spot lights, shaded like the cooked world's lamps in every pass, binned into
//! the same screen tiles, and moved, recoloured or put out at any time. A game light shines at
//! every hour unless it asks to follow the lamps (lit only as the evening lights the lamps).
//!
//! A spot can cast shadows: the opaque props the game places (their materials casting shadows)
//! block its light, drawn into a shadow map of its own seen from the light (a layer of the
//! renderer's light shadow map, [`Quality::light_shadow_size`] texels a side) whenever a caster
//! moved or the light's view changed, and kept from frame to frame otherwise. The cooked
//! world is lit by it but does not block it, as it blocks none of its lamps.
//!
//! The light buffer holds the world's lamps, then room for [`MAX_GAME_LIGHTS`] game lights; a
//! change rewrites the game lights' part before the next frame.
//!
//! [`Quality::light_shadow_size`]: super::Quality::light_shadow_size
use super::math::{M4, V3, add, look_at, mul, perspective_depth};
use super::renderer::Renderer;
use crate::gpu::Result;
use crate::gpu::pool::Pool;

handle!(
    /// A light a game placed.
    GameLight
);

/// The most game lights alive at once.
pub const MAX_GAME_LIGHTS: usize = 64;

/// The most game lights casting shadows at once.
pub const MAX_SHADOWED_LIGHTS: usize = 2;

/// The widest spot that casts shadows: its outer edge, radians from its direction (80 degrees).
pub const MAX_SHADOW_CONE: f32 = 1.4;

/// A spot light's cone: it shines along `direction`, fully inside `inner` radians of it, fading
/// to nothing at `outer`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpotCone {
    pub direction: V3,
    pub inner: f32,
    pub outer: f32,
}

/// A game light: where it hangs, how far it reaches (metres), its linear colour and intensity
/// (as a cooked lamp's), a cone for a spot light, whether it follows the lamps, and how much of
/// its light the props' shadows take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightDesc {
    pub position: V3,
    pub range: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub spot: Option<SpotCone>,
    /// Lit only as much as the time of day lights the world's lamps (none by day).
    pub follows_lamps: bool,
    /// How much of its light falls in the props' shadows is taken: 0 casts none, 1 leaves the
    /// shadows without it. Spots only, at most [`MAX_SHADOWED_LIGHTS`] at once, none wider than
    /// [`MAX_SHADOW_CONE`].
    pub shadow: f32,
}

/// A light as the shaders read it (`Light` in scene.wgsl): position and range, colour and
/// intensity, the spot's direction and the cosine of its outer edge (below -1.5: all round), the
/// cosine of its inner edge, whether it shines whatever the hour, its shadow map's layer plus one
/// (0: none) and how much its shadows take.
pub(super) type LightRecord = [f32; 16];

/// A cooked lamp's record: all round, following the time of day.
pub(super) fn lamp_record(
    position: V3,
    range: f32,
    color: [f32; 3],
    intensity: f32,
) -> LightRecord {
    let [x, y, z] = position;
    let [r, g, b] = color;
    [
        x, y, z, range, r, g, b, intensity, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0,
    ]
}

/// A game light as kept: its record (its shadow layer not yet given) and, when it casts shadows,
/// the view from it onto its shadow map.
#[derive(Clone, Copy, PartialEq)]
struct Entry {
    record: LightRecord,
    shadow_view: Option<M4>,
}

fn entry(desc: &LightDesc) -> Result<Entry> {
    let mut values = vec![desc.range, desc.intensity, desc.shadow];
    values.extend(desc.position);
    values.extend(desc.color);
    if let Some(s) = desc.spot {
        values.extend(s.direction);
        values.extend([s.inner, s.outer]);
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err("light: every value must be finite".into());
    }
    if desc.range <= 0.0 || desc.intensity < 0.0 || desc.color.iter().any(|c| *c < 0.0) {
        return Err("light: a positive range, and no negative colour or intensity".into());
    }
    if !(0.0..=1.0).contains(&desc.shadow) {
        return Err("light: shadow is between 0 (none) and 1".into());
    }
    let mut r = lamp_record(desc.position, desc.range, desc.color, desc.intensity);
    let mut shadow_view = None;
    if let Some(s) = desc.spot {
        let [dx, dy, dz] = s.direction;
        let length = (dx * dx + dy * dy + dz * dz).sqrt();
        if length < 1e-6 {
            return Err("light: a spot needs a direction".into());
        }
        if !(0.0..std::f32::consts::PI).contains(&s.outer) || s.inner < 0.0 || s.inner > s.outer {
            return Err("light: a spot's edges must be 0 <= inner <= outer < pi".into());
        }
        let direction = [dx / length, dy / length, dz / length];
        r[8..11].copy_from_slice(&direction);
        r[11] = s.outer.cos();
        // An inner edge on the outer one would leave no fade: keep a sliver.
        r[12] = s.inner.cos().max(r[11] + 1e-4);
        if desc.shadow > 0.0 {
            if s.outer > MAX_SHADOW_CONE {
                return Err(format!(
                    "light: a spot casting shadows reaches at most {MAX_SHADOW_CONE} radians out"
                ));
            }
            r[15] = desc.shadow;
            shadow_view = Some(shadow_view_of(
                desc.position,
                direction,
                s.outer,
                desc.range,
            ));
        }
    } else if desc.shadow > 0.0 {
        return Err("light: only a spot casts shadows".into());
    }
    r[13] = if desc.follows_lamps { 0.0 } else { 1.0 };
    Ok(Entry {
        record: r,
        shadow_view,
    })
}

/// World space to a spot's shadow map: from the light along its unit `direction`, the map just
/// spanning the cone, depth 0 near the light to 1 at its range.
fn shadow_view_of(position: V3, direction: V3, outer: f32, range: f32) -> M4 {
    let up = if direction[1].abs() > 0.999 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let view = look_at(position, add(position, direction), up);
    let near = (range * 0.005).clamp(0.01, 0.25);
    mul(&perspective_depth(2.0 * outer, 1.0, near, range), &view)
}

#[derive(Default)]
pub(super) struct GameLights {
    pool: Pool<Entry>,
    /// The live records in the order the light buffer holds them, rebuilt on any change.
    list: Vec<LightRecord>,
    /// The shadowed lights' views, by shadow map layer.
    shadows: Vec<M4>,
    /// The light buffer's game part is stale.
    pub(super) dirty: bool,
}

impl GameLights {
    /// The live game lights in buffer order.
    pub(super) fn records(&self) -> &[LightRecord] {
        &self.list
    }

    /// World space to each shadowed light's shadow map, by layer.
    pub(super) fn shadow_views(&self) -> &[M4] {
        &self.shadows
    }

    fn rebuild(&mut self) {
        self.list.clear();
        self.shadows.clear();
        for (_, e) in self.pool.iter() {
            let mut r = e.record;
            if let Some(view) = e.shadow_view {
                self.shadows.push(view);
                r[14] = self.shadows.len() as f32;
            }
            self.list.push(r);
        }
        self.dirty = true;
    }

    /// Lights casting shadows, besides `except`.
    fn shadowed(&self, except: Option<GameLight>) -> usize {
        self.pool
            .iter()
            .filter(|(id, e)| e.shadow_view.is_some() && except.is_none_or(|l| l.0 != *id))
            .count()
    }
}

impl Renderer {
    /// Hangs a light; it shines from the next frame.
    pub fn create_light(&mut self, desc: &LightDesc) -> Result<GameLight> {
        let e = entry(desc)?;
        if self.game_lights.list.len() >= MAX_GAME_LIGHTS {
            return Err(format!(
                "create_light: at most {MAX_GAME_LIGHTS} game lights"
            ));
        }
        if e.shadow_view.is_some() && self.game_lights.shadowed(None) >= MAX_SHADOWED_LIGHTS {
            return Err(format!(
                "create_light: at most {MAX_SHADOWED_LIGHTS} game lights cast shadows"
            ));
        }
        let id = self.game_lights.pool.insert(e)?;
        self.game_lights.rebuild();
        Ok(GameLight(id))
    }

    /// Moves, recolours or reshapes a light.
    pub fn set_light(&mut self, light: GameLight, desc: &LightDesc) -> Result<()> {
        let e = entry(desc)?;
        if self.game_lights.pool.get_mut(light.0).is_none() {
            return Err(format!("{light:?} was put out"));
        }
        if e.shadow_view.is_some() && self.game_lights.shadowed(Some(light)) >= MAX_SHADOWED_LIGHTS
        {
            return Err(format!(
                "set_light: at most {MAX_SHADOWED_LIGHTS} game lights cast shadows"
            ));
        }
        let slot = self
            .game_lights
            .pool
            .get_mut(light.0)
            .ok_or_else(|| format!("{light:?} was put out"))?;
        if *slot != e {
            *slot = e;
            self.game_lights.rebuild();
        }
        Ok(())
    }

    /// Puts a light out.
    pub fn destroy_light(&mut self, light: GameLight) {
        if self.game_lights.pool.remove(light.0).is_some() {
            self.game_lights.rebuild();
        }
    }

    /// Live game lights.
    pub fn game_lights(&self) -> usize {
        self.game_lights.list.len()
    }
}

#[cfg(test)]
mod tests {
    use super::super::math::transform;
    use super::*;

    fn lamp() -> LightDesc {
        LightDesc {
            position: [0.0, 3.0, 0.0],
            range: 6.0,
            color: [1.0, 0.8, 0.6],
            intensity: 2.0,
            spot: None,
            follows_lamps: false,
            shadow: 0.0,
        }
    }

    fn spot() -> LightDesc {
        LightDesc {
            spot: Some(SpotCone {
                direction: [0.0, -2.0, 0.0],
                inner: 0.2,
                outer: 0.5,
            }),
            ..lamp()
        }
    }

    #[test]
    fn records_say_how_a_light_shines() {
        let e = entry(&lamp()).expect("lamp");
        let r = e.record;
        assert_eq!(&r[..8], &[0.0, 3.0, 0.0, 6.0, 1.0, 0.8, 0.6, 2.0]);
        assert!(r[11] < -1.5, "all round");
        assert_eq!(r[13], 1.0, "whatever the hour");
        assert!(e.shadow_view.is_none());
        let following = LightDesc {
            follows_lamps: true,
            ..spot()
        };
        let r = entry(&following).expect("spot").record;
        assert_eq!(&r[8..11], &[0.0, -1.0, 0.0]);
        assert!((r[11] - 0.5f32.cos()).abs() < 1e-6 && (r[12] - 0.2f32.cos()).abs() < 1e-6);
        assert_eq!(r[13], 0.0, "with the lamps");
        assert_eq!(&r[14..], &[0.0, 0.0], "no shadows");
    }

    #[test]
    fn a_shadowed_spot_sees_its_cone_on_its_map() {
        let shadowed = LightDesc {
            shadow: 0.7,
            ..spot()
        };
        let e = entry(&shadowed).expect("spot");
        assert_eq!(e.record[15], 0.7);
        let view = e.shadow_view.expect("a shadow map");
        let at = |p: V3| {
            let c = transform(&view, [p[0], p[1], p[2], 1.0]);
            [c[0] / c[3], c[1] / c[3], c[2] / c[3]]
        };
        // Straight below the light: the map's middle, deeper further down.
        let below = at([0.0, 0.0, 0.0]);
        assert!(below[0].abs() < 1e-5 && below[1].abs() < 1e-5);
        assert!(below[2] > 0.0 && below[2] < at([0.0, -2.0, 0.0])[2]);
        // The cone's edge three metres down lands on the map's edge.
        let edge = at([3.0 * 0.5f32.tan(), 0.0, 0.0]);
        assert!((edge[0].abs() - 1.0).abs() < 1e-4, "{edge:?}");
        // Layers go to the shadowed lights in the light buffer's order.
        let mut lights = GameLights::default();
        lights
            .pool
            .insert(entry(&lamp()).expect("lamp"))
            .expect("in");
        lights.pool.insert(e).expect("in");
        lights.rebuild();
        assert_eq!(lights.records()[0][14], 0.0);
        assert_eq!(lights.records()[1][14], 1.0, "layer 0");
        assert_eq!(lights.shadow_views(), &[view]);
    }

    #[test]
    fn refuses_lights_that_cannot_shine() {
        for bad in [
            LightDesc {
                range: 0.0,
                ..lamp()
            },
            LightDesc {
                intensity: f32::NAN,
                ..lamp()
            },
            LightDesc {
                spot: Some(SpotCone {
                    direction: [0.0; 3],
                    inner: 0.1,
                    outer: 0.2,
                }),
                ..lamp()
            },
            LightDesc {
                spot: Some(SpotCone {
                    direction: [0.0, -1.0, 0.0],
                    inner: 0.4,
                    outer: 0.2,
                }),
                ..lamp()
            },
            // Shadows: a spot's, no wider than the limit, at most all of its light.
            LightDesc {
                shadow: 0.5,
                ..lamp()
            },
            LightDesc {
                shadow: 1.5,
                ..spot()
            },
            LightDesc {
                shadow: 0.5,
                spot: Some(SpotCone {
                    direction: [0.0, -1.0, 0.0],
                    inner: 0.4,
                    outer: 1.5,
                }),
                ..lamp()
            },
        ] {
            assert!(entry(&bad).is_err(), "{bad:?}");
        }
    }
}
