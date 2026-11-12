//! Driver tests for the stylized renderer on a synthetic scene (run with `--ignored`; set
//! `GANBATE_VULKAN_VALIDATION=1` to also check the Vulkan validation layers).

mod environment;
mod models;
mod props;
mod world;

use ganbate_engine::Vec3;
use ganbate_engine::gpu::Device;
use ganbate_engine::render::{Environment, Renderer, TimeOfDay, View, Weather};
use ganbate_engine::world::{
    AlphaMode, Emission, Instance, Material, Mesh, PointLight, Shading, Texture, Transform, World,
};

fn quad(y: f32, half: f32, material: u32) -> Mesh {
    let p = |x: f32, z: f32| Vec3(x, y, z);
    Mesh {
        positions: vec![
            p(-half, -half),
            p(half, -half),
            p(half, half),
            p(-half, half),
        ],
        normals: vec![Vec3(0.0, 1.0, 0.0); 4],
        uvs: vec![[0.0, 0.0], [half, 0.0], [half, half], [0.0, half]],
        indices: vec![0, 2, 1, 0, 3, 2],
        color: [0.5; 3],
        material: Some(material),
    }
}

fn cube(c: [f32; 3], s: f32, material: u32) -> Mesh {
    let mut m = Mesh {
        material: Some(material),
        ..Mesh::default()
    };
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
    ];
    for (n, u, v) in faces {
        let base = m.positions.len() as u32;
        for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            m.positions.push(Vec3(
                c[0] + s * (n[0] + a * u[0] + b * v[0]),
                c[1] + s * (n[1] + a * u[1] + b * v[1]),
                c[2] + s * (n[2] + a * u[2] + b * v[2]),
            ));
            m.normals.push(Vec3(n[0], n[1], n[2]));
            m.uvs.push([a * 0.5 + 0.5, b * 0.5 + 0.5]);
        }
        m.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    m
}

fn material(name: &str, color: [f32; 4], shading: Shading) -> Material {
    Material {
        name: name.into(),
        base_color: color,
        shading,
        double_sided: true,
        alpha: AlphaMode::Opaque,
        ..Material::default()
    }
}

fn world() -> World {
    let mut ground = material("ground", [0.35, 0.55, 0.25, 1.0], Shading::Standard);
    ground.base_color_texture = Some(0);
    let mut lamp = material("lamp", [1.0, 0.6, 0.3, 1.0], Shading::Standard);
    lamp.emissive = [3.0, 1.8, 0.7];
    lamp.emission = Emission::Lamp;
    let checker = vec![
        255, 255, 255, 255, 200, 200, 200, 255, 200, 200, 200, 255, 255, 255, 255, 255,
    ];
    let meshes = vec![
        quad(0.0, 12.0, 0),
        cube([0.0, 1.0, 0.0], 1.0, 1),
        quad(-0.3, 40.0, 2),
        cube([-2.5, 1.0, -1.5], 0.9, 3),
        cube([3.0, 1.6, 1.0], 0.12, 4),
        // An alpha-masked card: its shadow must use the texture's cut-out.
        Mesh {
            positions: vec![
                Vec3(-4.0, 0.0, 2.0),
                Vec3(-2.0, 0.0, 2.0),
                Vec3(-2.0, 2.0, 2.0),
                Vec3(-4.0, 2.0, 2.0),
            ],
            normals: vec![Vec3(0.0, 0.0, 1.0); 4],
            uvs: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            indices: vec![0, 1, 2, 0, 2, 3],
            color: [0.5; 3],
            material: Some(5),
        },
    ];
    // The ground sits above the water and covers the middle; water surrounds it.
    let instances = (0..meshes.len() as u32)
        .map(|mesh| Instance {
            mesh,
            transform: Transform::IDENTITY,
        })
        .collect();
    World {
        meshes,
        instances,
        materials: vec![
            ground,
            material("box", [0.8, 0.15, 0.1, 1.0], Shading::Standard),
            material("water", [0.02, 0.1, 0.15, 1.0], Shading::Water),
            material("canopy", [0.9, 0.5, 0.6, 1.0], Shading::Foliage),
            lamp,
            {
                let mut card = material("card", [0.3, 0.6, 0.3, 1.0], Shading::Foliage);
                card.alpha = AlphaMode::Mask(0.5);
                card.base_color_texture = Some(1);
                card
            },
        ],
        textures: vec![
            Texture {
                width: 2,
                height: 2,
                srgb: true,
                levels: vec![checker, vec![227, 227, 227, 255]],
                encoded: None,
            },
            Texture {
                width: 2,
                height: 2,
                srgb: true,
                levels: vec![
                    vec![
                        255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255,
                    ],
                    vec![255, 255, 255, 128],
                ],
                encoded: None,
            },
        ],
        lights: vec![PointLight {
            position: Vec3(3.0, 1.6, 1.0),
            color: [1.0, 0.6, 0.3],
            intensity: 30.0,
            range: 7.0,
        }],
    }
}

const W: u32 = 128;
const H: u32 = 72;

fn view() -> View {
    View {
        name: "test".into(),
        eye: [0.0, 4.0, 10.0],
        target: [0.0, 0.5, 0.0],
        fov: 50.0,
        ..View::default()
    }
}

fn render(device: &mut Device<'_>, r: &mut Renderer, hours: f32, weather: &str) -> Vec<u8> {
    let time = TimeOfDay {
        hours,
        ..TimeOfDay::default()
    };
    let weather = Weather::preset(weather).expect("preset");
    let env = Environment::resolve(&time, &weather, &r.look().palette, 3.0);
    r.render_still(device, &view(), &env, 1).expect("render")
}

fn luma(px: &[u8], x: u32, y: u32) -> f32 {
    let i = ((y * W + x) * 4) as usize;
    0.2126 * f32::from(px[i]) + 0.7152 * f32::from(px[i + 1]) + 0.0722 * f32::from(px[i + 2])
}

fn region(px: &[u8], x0: u32, y0: u32, x1: u32, y1: u32) -> f32 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            sum += luma(px, x, y);
            n += 1.0;
        }
    }
    sum / n
}

/// Mean absolute difference per channel between two RGBA8 images.
fn difference(a: &[u8], b: &[u8]) -> f32 {
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| u64::from(x.abs_diff(*y)))
        .sum();
    sum as f32 / a.len() as f32
}

/// Land sloping from 3 m down to a seabed 4 m under a sea at 0 m: flat to z = -60, the water
/// line at z = -10, 1.4 m deep at z = 13.
fn shore() -> World {
    let mut land = Mesh {
        material: Some(0),
        ..Mesh::default()
    };
    let height = |z: f32| (3.0 - (z + 60.0) * 0.06).clamp(-4.0, 3.0);
    let (cols, rows) = (31u32, 81u32);
    for j in 0..rows {
        for i in 0..cols {
            let (x, z) = (-60.0 + 4.0 * i as f32, -100.0 + 2.0 * j as f32);
            land.positions.push(Vec3(x, height(z), z));
            // d(height)/dz = -0.06 on the slope.
            let n = if (-60.0..56.0).contains(&z) {
                0.06
            } else {
                0.0
            };
            let l = (1.0f32 + n * n).sqrt();
            land.normals.push(Vec3(0.0, 1.0 / l, n / l));
            land.uvs.push([0.0, 0.0]);
        }
    }
    for j in 0..rows - 1 {
        for i in 0..cols - 1 {
            let k = j * cols + i;
            land.indices
                .extend([k, k + cols, k + 1, k + 1, k + cols, k + cols + 1]);
        }
    }
    let meshes = vec![land, quad(0.0, 400.0, 1)];
    World {
        instances: (0..meshes.len() as u32)
            .map(|mesh| Instance {
                mesh,
                transform: Transform::IDENTITY,
            })
            .collect(),
        meshes,
        materials: vec![
            material("land", [0.2, 0.3, 0.1, 1.0], Shading::Standard),
            material("sea", [0.1, 0.2, 0.3, 1.0], Shading::Water),
        ],
        textures: vec![],
        lights: vec![],
    }
}
