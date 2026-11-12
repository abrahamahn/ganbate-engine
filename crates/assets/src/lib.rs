//! First-party asset import and binary cooking tools.
pub mod bc;
pub mod cooked;
pub mod glb;
pub mod image;
/// The JSON reader and writer now live in `ganbate-core`; this keeps `ganbate_assets::json` working.
pub use ganbate_core::json;
pub mod mesh_opt;
pub mod mip;
pub mod model;
pub mod png;
pub mod zlib;
pub use mesh_opt::optimize_mesh;
use std::io::Read;
pub fn read_asset(path: &std::path::Path) -> Result<Vec<u8>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let cap = f
        .metadata()
        .map(|m| (m.len().min(cooked::MAX_BYTES as u64) as usize).saturating_add(32))
        .unwrap_or(0);
    let mut bytes = Vec::with_capacity(cap);
    f.take(cooked::MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > cooked::MAX_BYTES {
        return Err("asset exceeds file size budget".into());
    }
    Ok(bytes)
}

/// Replace an output only after its complete bytes are on disk. A failed cook leaves the
/// prior asset usable, and concurrent invocations never share a partial file.
pub fn write_asset(path: &std::path::Path, data: &[u8]) -> std::io::Result<()> {
    write_atomically(path, data, |tmp| std::fs::rename(tmp, path))
}

/// Installs a complete new asset without replacing a file created by another writer.
pub fn create_asset(path: &std::path::Path, data: &[u8]) -> std::io::Result<()> {
    write_atomically(path, data, |tmp| {
        std::fs::hard_link(tmp, path)?;
        let _ = std::fs::remove_file(tmp);
        Ok(())
    })
}

fn write_atomically(
    path: &std::path::Path,
    data: &[u8],
    install: impl FnOnce(&std::path::Path) -> std::io::Result<()>,
) -> std::io::Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut retries = 0;
    loop {
        let tmp = dir.join(format!(
            ".ganbate-cook-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let f = match std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp)
        {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && retries < 16 => {
                retries += 1;
                continue;
            }
            Err(e) => return Err(e),
        };
        // The file closes at the end of this block, before the rename.
        let res = {
            let mut f = f;
            f.write_all(data).and_then(|()| f.sync_all())
        };
        let res = res.and_then(|()| install(&tmp));
        if res.is_err() {
            let _ = std::fs::remove_file(tmp);
        }
        return res;
    }
}
