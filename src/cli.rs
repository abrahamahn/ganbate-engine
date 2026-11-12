//! Shared zero-dependency CLI argument parsing, cooked-world / Look-file loading, and free-fly
//! camera helpers used across the standalone engine binaries (`src/bin/*`).

use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::render::{LookFile, View, math};
use crate::world::World;

/// Minimal zero-dependency command-line flag cursor over `String` arguments.
pub struct Args<I = std::iter::Skip<std::env::Args>> {
    iter: I,
}

impl Args<std::iter::Skip<std::env::Args>> {
    /// Creates a cursor over `std::env::args().skip(1)`.
    pub fn from_env() -> Self {
        Self {
            iter: std::env::args().skip(1),
        }
    }
}

impl<I: Iterator<Item = String>> Args<I> {
    pub fn new(iter: I) -> Self {
        Self { iter }
    }

    pub fn next_arg(&mut self) -> Option<String> {
        self.iter.next()
    }

    pub fn value(&mut self, flag: &str) -> Result<String, String> {
        self.iter
            .next()
            .ok_or_else(|| format!("{flag} needs a value"))
    }

    pub fn path(&mut self, flag: &str) -> Result<PathBuf, String> {
        Ok(PathBuf::from(self.value(flag)?))
    }

    pub fn parsed<T>(&mut self, flag: &str) -> Result<T, String>
    where
        T: FromStr,
        T::Err: std::fmt::Display,
    {
        self.value(flag)?
            .parse::<T>()
            .map_err(|e| format!("{flag}: {e}"))
    }

    pub fn size_wxh(&mut self, flag: &str) -> Result<(u32, u32), String> {
        let raw = self.value(flag)?;
        let (w, h) = raw
            .split_once('x')
            .ok_or_else(|| format!("{flag} must be WxH"))?;
        let width = w.parse::<u32>().map_err(|e| format!("{flag}: {e}"))?;
        let height = h.parse::<u32>().map_err(|e| format!("{flag}: {e}"))?;
        Ok((width, height))
    }
}

/// Decodes a cooked `.gbew` world from `path`.
pub fn load_world(path: &Path) -> Result<World, String> {
    crate::assets::cooked::decode(&crate::assets::read_asset(path)?)
}

/// Parses `look_path` when given, or returns [`LookFile::default`].
pub fn load_look_or_default(look_path: Option<&Path>) -> Result<LookFile, String> {
    match look_path {
        Some(p) => LookFile::parse(&crate::assets::read_asset(p)?),
        None => Ok(LookFile::default()),
    }
}

/// Synthesizes a framing `"overview"` [`View`] from renderer world `bounds` (`(lo, hi)`).
pub fn overview_view(bounds: (math::V3, math::V3), fov: f32) -> View {
    let (lo, hi) = bounds;
    let c = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    let span = ((hi[0] - lo[0]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let r = if span > 1e-3 { span * 0.4 } else { 40.0 };
    View {
        name: "overview".into(),
        eye: [c[0] + r, c[1] + r * 0.6, c[2] + r],
        target: c,
        fov,
        ..View::default()
    }
}

/// Returns the Look file's authored camera views, or a single [`overview_view`] when empty.
pub fn views_or_overview(file: &LookFile, bounds: (math::V3, math::V3), fov: f32) -> Vec<View> {
    if file.views.is_empty() {
        vec![overview_view(bounds, fov)]
    } else {
        file.views.clone()
    }
}

/// Free-fly yaw/pitch camera state shared by interactive players (`ganbate-player`, `ganbate-ios`).
#[derive(Clone, Copy, Debug)]
pub struct FlyCamera {
    pub pos: math::V3,
    pub yaw: f32,
    pub pitch: f32,
}

impl FlyCamera {
    pub fn from_view(v: &View) -> Self {
        let d = math::normalize(math::sub(v.target, v.eye));
        Self {
            pos: v.eye,
            yaw: d[0].atan2(-d[2]),
            pitch: d[1].clamp(-1.0, 1.0).asin(),
        }
    }

    pub fn forward(&self) -> math::V3 {
        let c = self.pitch.cos();
        [c * self.yaw.sin(), self.pitch.sin(), -c * self.yaw.cos()]
    }

    /// Builds a [`View`] looking 30 m ahead (used by the renderer to size shadow cascades).
    pub fn view(&self, fov: f32) -> View {
        View {
            name: "free".into(),
            eye: self.pos,
            target: math::add(self.pos, math::scale(self.forward(), 30.0)),
            fov,
            ..View::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_parses_typed_flags_and_sizes() {
        let raw = vec![
            "--width".to_string(),
            "1280".to_string(),
            "--size".to_string(),
            "1600x900".to_string(),
        ];
        let mut args = Args::new(raw.into_iter());
        assert_eq!(args.next_arg().as_deref(), Some("--width"));
        assert_eq!(args.parsed::<u32>("--width").unwrap(), 1280);
        assert_eq!(args.next_arg().as_deref(), Some("--size"));
        assert_eq!(args.size_wxh("--size").unwrap(), (1600, 900));
        assert!(args.next_arg().is_none());
    }

    #[test]
    fn fly_camera_preserves_view_heading() {
        let v = View {
            name: "test".into(),
            eye: [0.0, 2.0, 10.0],
            target: [0.0, 2.0, 0.0],
            fov: 50.0,
            ..View::default()
        };
        let fly = FlyCamera::from_view(&v);
        let out = fly.view(50.0);
        assert!((out.target[0] - 0.0).abs() < 1e-4);
        assert!((out.target[1] - 2.0).abs() < 1e-4);
        assert!((out.target[2] - -20.0).abs() < 1e-4);
    }
}
