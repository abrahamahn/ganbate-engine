//! Sample-accurate Stereo Peak dBFS, RMS dBFS, Clipping, and L/R Phase Correlation metering.

pub const MIN_METER_DB: f32 = -72.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelMeterSnapshot {
    pub peak_l_db: f32,
    pub peak_r_db: f32,
    pub rms_l_db: f32,
    pub rms_r_db: f32,
    /// Stereo phase correlation in `-1.0..=1.0` (`+1.0` = mono coherent, `-1.0` = 180° out of phase).
    pub correlation: f32,
    pub clipped: bool,
    pub gain_reduction_db: f32,
}

impl Default for ChannelMeterSnapshot {
    fn default() -> Self {
        Self {
            peak_l_db: MIN_METER_DB,
            peak_r_db: MIN_METER_DB,
            rms_l_db: MIN_METER_DB,
            rms_r_db: MIN_METER_DB,
            correlation: 1.0,
            clipped: false,
            gain_reduction_db: 0.0,
        }
    }
}

#[inline]
pub fn linear_to_db(gain: f32) -> f32 {
    if !gain.is_finite() || gain <= 1e-6 {
        return MIN_METER_DB;
    }
    (20.0 * gain.log10()).max(MIN_METER_DB)
}

#[inline]
pub fn db_to_linear(db: f32) -> f32 {
    if !db.is_finite() || db <= MIN_METER_DB {
        return 0.0;
    }
    10.0_f32.powf(db / 20.0)
}

/// Computes Peak dBFS, RMS dBFS, `>= 1.0` digital clip detection, and Pearson L/R phase correlation.
pub fn analyze_stereo_frames(frames: &[[f32; 2]], gain_reduction_db: f32) -> ChannelMeterSnapshot {
    if frames.is_empty() {
        return ChannelMeterSnapshot {
            gain_reduction_db,
            ..ChannelMeterSnapshot::default()
        };
    }

    let mut peak_l = 0.0_f32;
    let mut peak_r = 0.0_f32;
    let mut sum_sq_l = 0.0_f64;
    let mut sum_sq_r = 0.0_f64;
    let mut sum_lr = 0.0_f64;

    for &[l, r] in frames {
        let ls = if l.is_finite() { l } else { 0.0 };
        let rs = if r.is_finite() { r } else { 0.0 };
        peak_l = peak_l.max(ls.abs());
        peak_r = peak_r.max(rs.abs());
        sum_sq_l += f64::from(ls) * f64::from(ls);
        sum_sq_r += f64::from(rs) * f64::from(rs);
        sum_lr += f64::from(ls) * f64::from(rs);
    }

    let n = frames.len() as f64;
    let rms_l = (sum_sq_l / n).sqrt() as f32;
    let rms_r = (sum_sq_r / n).sqrt() as f32;
    let denom = (sum_sq_l * sum_sq_r).sqrt();
    let correlation = if denom > 1e-12 {
        (sum_lr / denom).clamp(-1.0, 1.0) as f32
    } else {
        1.0
    };

    ChannelMeterSnapshot {
        peak_l_db: linear_to_db(peak_l),
        peak_r_db: linear_to_db(peak_r),
        rms_l_db: linear_to_db(rms_l),
        rms_r_db: linear_to_db(rms_r),
        correlation,
        clipped: peak_l >= 1.0 || peak_r >= 1.0,
        gain_reduction_db,
    }
}
