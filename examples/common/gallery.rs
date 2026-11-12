use ganbate_engine::{Scene, Vec3};

/// A gallery of solids makes a neutral example for any engine client.
pub fn gallery(angle: f32) -> Scene {
    let mut scene = Scene::default();
    let y = -0.86;
    scene.quad(
        [
            Vec3(-18.0, y, -18.0),
            Vec3(18.0, y, -18.0),
            Vec3(18.0, y, 18.0),
            Vec3(-18.0, y, 18.0),
        ],
        [0.10, 0.13, 0.18],
    );
    for (x, z, scale, base, phase) in [
        (0.0, 0.0, 1.15, [0.18, 0.70, 0.65], angle),
        (-1.65, 0.15, 0.72, [0.94, 0.55, 0.21], angle + 0.6),
        (1.55, -0.5, 0.82, [0.43, 0.54, 0.90], angle - 0.8),
    ] {
        let mid = y + scale;
        let top = Vec3(x, mid + 1.28 * scale, z);
        let bottom = Vec3(x, y, z);
        for k in 0..6 {
            let at = |k: usize| {
                let a = std::f32::consts::TAU * k as f32 / 6.0 + phase;
                Vec3(x + 0.69 * scale * a.cos(), mid, z + 0.69 * scale * a.sin())
            };
            let a = at(k);
            let b = at(k + 1);
            let light =
                0.72 + 0.22 * (std::f32::consts::TAU * (k as f32 + 0.5) / 6.0 + phase).cos();
            scene.triangle(top, a, b, base.map(|v| v * light));
            scene.triangle(bottom, b, a, base.map(|v| v * light * 0.62));
        }
    }
    scene
}
