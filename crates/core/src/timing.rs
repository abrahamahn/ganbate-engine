//! Frame timing summaries. Missing samples never count as passing a budget.
#[derive(Debug, Clone, Copy)]
pub struct Distribution {
    pub count: usize,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}
impl Distribution {
    /// Nearest rank percentiles in milliseconds. Keeps the input sample order intact.
    pub fn from_ms(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() || samples.iter().any(|v| !v.is_finite() || *v < 0.) {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        let rank = |percent: usize| sorted[(sorted.len() * percent).div_ceil(100) - 1];
        Some(Self {
            count: sorted.len(),
            p50: rank(50),
            p95: rank(95),
            p99: rank(99),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hitch_is_visible_in_tail() {
        let mut times = vec![4.; 98];
        times.extend([8., 24.]);
        let d = Distribution::from_ms(&times).unwrap();
        assert_eq!((d.p50, d.p95, d.p99), (4., 4., 8.));
        assert!(Distribution::from_ms(&[]).is_none());
    }
}

/// Intervals between distinct displayed frames. The caller supplies monotonic
/// presentation timestamps from a single surface and one uninterrupted session.
#[derive(Debug)]
pub struct Pacing {
    pub intervals: Distribution,
    pub duration_s: f64,
    pub mean_fps: f64,
    pub slow_intervals: usize,
    pub worst_minute_fps: Option<f64>,
}
impl Pacing {
    pub fn analyze(times_ms: &[f64], target_hz: u32) -> Result<Self, &'static str> {
        if target_hz == 0 || times_ms.len() < 2 {
            return Err("need a target and two or more presentation times");
        }
        let budget = 1000.0 / f64::from(target_hz);
        let gaps: Vec<_> = times_ms.windows(2).map(|t| t[1] - t[0]).collect();
        if gaps.iter().any(|g| !g.is_finite() || *g <= 0.) {
            return Err("presentation times must be finite and strictly increasing");
        }
        let intervals = Distribution::from_ms(&gaps).ok_or("invalid presentation intervals")?;
        let duration_s = (times_ms[times_ms.len() - 1] - times_ms[0]) * 0.001;
        // Test a sliding minute at every displayed frame. A long stall followed
        // by a burst still leaves a window with few or no new displayed frames.
        let mut start = 0;
        let mut worst: Option<f64> = None;
        for (end, &now) in times_ms.iter().enumerate() {
            if now - times_ms[0] < 60_000.0 {
                continue;
            }
            let lower = now - 60_000.0;
            while start < end && times_ms[start] <= lower {
                start += 1;
            }
            // Sample immediately before each presentation. This conservative lower
            // bound differs by at most one frame per minute at exact endpoints.
            let fps = (end - start) as f64 / 60.0;
            worst = Some(worst.map_or(fps, |prev| prev.min(fps)));
        }
        Ok(Self {
            intervals,
            duration_s,
            mean_fps: gaps.len() as f64 / duration_s,
            slow_intervals: gaps.iter().filter(|g| **g > 1.5 * budget).count(),
            worst_minute_fps: worst,
        })
    }
    pub fn meets(&self, target_hz: u32, min_seconds: u32) -> bool {
        let hz = f64::from(target_hz);
        self.duration_s >= f64::from(min_seconds)
            && self.mean_fps >= hz * 0.98
            && self.worst_minute_fps.is_some_and(|rate| rate >= hz * 0.98)
            && self.intervals.p99 <= 1100.0 / hz
    }
}
#[cfg(test)]
mod pacing_tests {
    use super::*;
    #[test]
    fn handles_cadence_hitches_and_invalid_timestamps() {
        let mut t: Vec<f64> = (0..14_402).map(|n| n as f64 * 1000. / 120.).collect();
        let p = Pacing::analyze(&t, 120).unwrap();
        assert!(p.meets(120, 120));
        assert!(!p.meets(240, 120));
        assert!(!p.meets(120, 121));
        for v in &mut t[7_201..] {
            *v += 3000.;
        }
        let hitch = Pacing::analyze(&t, 120).unwrap();
        assert_eq!(hitch.slow_intervals, 1);
        assert!(!hitch.meets(120, 120));
        assert!(Pacing::analyze(&[2., 2.], 120).is_err());
    }
}
