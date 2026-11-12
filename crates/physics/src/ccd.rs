//! Response to translating surfaces during a continuous sweep.
use crate::math::Vec3;

/// A permitted motion satisfies `normal.dot(motion) >= minimum`. Normals point
/// away from the obstacle and are restricted to the body's unlocked axes.
#[derive(Clone, Copy)]
pub(crate) struct Plane {
    normal: Vec3,
    minimum: f32,
}

impl Plane {
    pub(crate) fn new(normal: Vec3, target: Vec3, mask: Vec3) -> Option<Self> {
        let (free, length) = normal.mul_elem(mask).normalize_and_length();
        (length >= 1e-6).then(|| Self {
            normal: free,
            minimum: normal.dot(target) / length,
        })
    }
}

/// Project onto the moving contact planes while respecting the speed (or distance) budget.
/// A compatible corner needs both planes at once: repeatedly pushing off one and then the
/// other can consume all the sweep iterations at the same instant.
pub(crate) fn permitted(preferred: Vec3, planes: &[Plane], maximum: f32) -> Option<Vec3> {
    let result = project(preferred, planes)?;
    let length = result.normalize_and_length().1;
    if length <= maximum {
        return Some(result);
    }
    // Scaling towards zero would fall through a moving surface. Scale only the motion beyond
    // the shortest feasible response. Every interpolated point still meets all the planes.
    let base = project(Vec3::ZERO, planes)?;
    if base.normalize_and_length().1 > maximum {
        return None;
    }
    // Scale first so that squaring the speeds in this line/sphere intersection cannot overflow.
    let scale = length;
    let b = base / scale;
    let d = result / scale - b;
    let a = d.length_squared();
    if a == 0.0 {
        return Some(base);
    }
    let k = b.dot(d);
    let radius = maximum / scale;
    let c = radius * radius - b.length_squared();
    let t = ((-k + (k * k + a * c.max(0.0)).sqrt()) / a).clamp(0.0, 1.0);
    Some(base + (result - base) * t)
}

fn project(preferred: Vec3, planes: &[Plane]) -> Option<Vec3> {
    let valid = |v: Vec3| {
        let length = v.normalize_and_length().1;
        v.is_finite()
            && planes.iter().all(|p| {
                let epsilon = 4.0 * f32::EPSILON * length.max(p.minimum.abs()).max(1e-4);
                p.normal.dot(v) >= p.minimum - epsilon
            })
    };
    if valid(preferred) {
        return Some(preferred);
    }
    // In three dimensions the closest feasible point lies on one face, an edge, or a vertex.
    // There are at most eight hits in a sweep. Usually the single-face projection suffices.
    let mut closest = None;
    let mut distance = f32::INFINITY;
    let mut consider = |candidate: Vec3| {
        if valid(candidate) {
            let d = (candidate - preferred).normalize_and_length().1;
            if d < distance {
                closest = Some(candidate);
                distance = d;
            }
        }
    };
    for (i, a) in planes.iter().enumerate() {
        // Keep the tangential part and set the normal part directly. Subtracting two
        // large normal speeds can lose a small, nonzero target speed to cancellation.
        let tangent = a.normal.cross(preferred).cross(a.normal);
        consider(tangent + a.normal * a.minimum);
        for (j, b) in planes[..i].iter().enumerate() {
            let line = a.normal.cross(b.normal);
            let area = line.length_squared();
            if area < 1e-12 {
                continue;
            }
            consider(
                (line * preferred.dot(line)
                    + b.normal.cross(line) * a.minimum
                    + line.cross(a.normal) * b.minimum)
                    / area,
            );
            for c in &planes[..j] {
                let det = line.dot(c.normal);
                if det.abs() > 1e-6 {
                    consider(
                        (b.normal.cross(c.normal) * a.minimum
                            + c.normal.cross(a.normal) * b.minimum
                            + line * c.minimum)
                            / det,
                    );
                }
            }
        }
    }
    closest
}
