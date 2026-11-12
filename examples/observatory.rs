#[cfg(any(target_os = "linux", target_os = "windows", target_os = "android"))]
#[path = "common/gallery.rs"]
mod common;
#[cfg(any(target_os = "linux", target_os = "windows", target_os = "android"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ganbate_engine::{Camera, Vec3, vulkan::VulkanRenderer};
    use std::io::{BufWriter, Write};
    let output = std::env::args_os()
        .nth(1)
        .unwrap_or_else(|| "observatory.ppm".into());
    let (w, h) = (800, 480);
    let camera = Camera {
        eye: Vec3(2.3, 2.0, 5.3),
        target: Vec3(0.0, 0.15, 0.0),
        vertical_fov_degrees: 43.0,
        near: 0.1,
        far: 60.0,
    };
    let vertices = camera.vertices(&common::gallery(0.45), w, h)?;
    let mut gpu = VulkanRenderer::new()?;
    let frame = gpu.render(w, h, [0.03, 0.045, 0.07, 1.0], &vertices)?;
    let mut out = BufWriter::new(std::fs::File::create(output)?);
    writeln!(out, "P6\n{w} {h}\n255")?;
    for px in frame.chunks_exact(4) {
        out.write_all(&px[..3])?;
    }
    out.flush()?;
    println!("{} | {w}x{h}", gpu.device_name());
    Ok(())
}
#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "android")))]
fn main() {
    eprintln!("This example requires a Vulkan host.");
    std::process::exit(1);
}
