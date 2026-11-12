//! Parametric timing curves and analytic damped springs for deterministic timeline evaluation.

/// Timing curve mapping normalized progress `t` (`0.0..=1.0`) and span duration in seconds to an
/// eased progress value (which may overshoot `1.0` for Bezier or underdamped [`CurveSpec::Spring`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum CurveSpec {
    Linear,
    #[default]
    SmoothStep,
    EaseOutQuart,
    CubicBezier {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    },
    Spring {
        stiffness: f32,
        damping: f32,
        mass: f32,
    },
}

fn cubic_bezier_axis(u: f32, p1: f32, p2: f32) -> f32 {
    let inv = 1.0 - u;
    3.0 * inv * inv * u * p1 + 3.0 * inv * u * u * p2 + u * u * u
}

impl CurveSpec {
    pub fn validate(&self) -> Result<(), String> {
        match *self {
            Self::Linear | Self::SmoothStep | Self::EaseOutQuart => Ok(()),
            Self::CubicBezier { x1, y1, x2, y2 } => {
                if ![x1, y1, x2, y2].into_iter().all(f32::is_finite) {
                    return Err("cubic bezier control points must be finite".into());
                }
                if !(0.0..=1.0).contains(&x1) || !(0.0..=1.0).contains(&x2) {
                    return Err("cubic bezier x1 and x2 must lie in 0..=1".into());
                }
                Ok(())
            }
            Self::Spring {
                stiffness,
                damping,
                mass,
            } => {
                if ![stiffness, damping, mass].into_iter().all(f32::is_finite) {
                    return Err("spring parameters must be finite".into());
                }
                if stiffness <= 0.0 || damping < 0.0 || mass <= 0.0 {
                    return Err("spring requires stiffness > 0, damping >= 0, mass > 0".into());
                }
                Ok(())
            }
        }
    }

    /// Evaluates the curve at normalized `progress` (`0.0..=1.0`) over a span of `duration_s`
    /// seconds. Endpoints `<= 0.0` and `>= 1.0` evaluate to `0.0` and `1.0` respectively so
    /// sections settle cleanly on their authored targets.
    pub fn sample(&self, progress: f32, duration_s: f32) -> f32 {
        if !progress.is_finite() || progress <= 0.0 {
            return 0.0;
        }
        if progress >= 1.0 {
            return 1.0;
        }
        let t = progress.clamp(0.0, 1.0);
        match *self {
            Self::Linear => t,
            Self::SmoothStep => t * t * (3.0 - 2.0 * t),
            Self::EaseOutQuart => {
                let inv = 1.0 - t;
                1.0 - inv * inv * inv * inv
            }
            Self::CubicBezier { x1, y1, x2, y2 } => {
                let mut low = 0.0f32;
                let mut high = 1.0f32;
                for _ in 0..20 {
                    let mid = 0.5 * (low + high);
                    if cubic_bezier_axis(mid, x1, x2) < t {
                        low = mid;
                    } else {
                        high = mid;
                    }
                }
                cubic_bezier_axis(0.5 * (low + high), y1, y2)
            }
            Self::Spring {
                stiffness,
                damping,
                mass,
            } => {
                let dur = if duration_s.is_finite() && duration_s > 1e-4 {
                    duration_s
                } else {
                    1.0
                };
                let sec = t * dur;
                let m = mass.max(1e-4);
                let k = stiffness.max(1e-4);
                let c = damping.max(0.0);
                let omega0 = (k / m).sqrt();
                let zeta = c / (2.0 * (k * m).sqrt());
                let raw = if zeta < 1.0 - 1e-4 {
                    let omegad = omega0 * (1.0 - zeta * zeta).sqrt();
                    let env = (-zeta * omega0 * sec).exp();
                    let ratio = (zeta * omega0) / omegad;
                    1.0 - env * ((omegad * sec).cos() + ratio * (omegad * sec).sin())
                } else if zeta <= 1.0 + 1e-4 {
                    let env = (-omega0 * sec).exp();
                    1.0 - env * (1.0 + omega0 * sec)
                } else {
                    let s = (zeta * zeta - 1.0).sqrt();
                    let r1 = -omega0 * (zeta - s);
                    let r2 = -omega0 * (zeta + s);
                    let denom = (r1 - r2).max(1e-6);
                    1.0 - (r1 * (r2 * sec).exp() - r2 * (r1 * sec).exp()) / denom
                };
                // Smoothly taper any residual tail over the final 10% so t=1 hits 1.0 cleanly.
                if t > 0.9 {
                    let tail = (t - 0.9) / 0.1;
                    let blend = tail * tail * (3.0 - 2.0 * tail);
                    raw + (1.0 - raw) * blend
                } else {
                    raw
                }
            }
        }
    }
}
