//! Shared by the MSL tests: compile generated Metal Shading Language with a stock clang++
//! against the stand-in `tests/msl/metal_stdlib`.
#![allow(dead_code)]

use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tests run on parallel threads of one process and several compile an entry point of the same
/// name, so each check writes its own file instead of sharing `<tag>.cpp` in the process's folder.
static CHECK_FILE: AtomicUsize = AtomicUsize::new(0);

/// The C++ compiler used for MSL syntax checks, if any.
pub fn clang() -> Option<String> {
    let name = std::env::var("GANBATE_MSL_CLANG").unwrap_or_else(|_| "clang++".into());
    Command::new(&name)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| name)
}

/// Spells `metal::float3(...)`-style vector constructors as the stand-in header's
/// `metal::make_float3(...)` (a stock C++ compiler cannot construct clang vectors that way).
pub fn rewrite_constructors(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + 1024);
    let mut rest = src;
    while let Some(i) = rest.find("metal::") {
        out.push_str(&rest[..i + 7]);
        rest = &rest[i + 7..];
        let ident_len = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let ident = &rest[..ident_len];
        let vector = ["float", "int", "uint", "bool", "half"].iter().any(|t| {
            ident
                .strip_prefix(t)
                .is_some_and(|n| matches!(n, "2" | "3" | "4"))
        }) || ident.starts_with("packed_");
        if vector && rest[ident_len..].starts_with('(') {
            out.push_str("make_");
        }
    }
    out.push_str(rest);
    out
}

/// Compiles `source` with clang++ against the stand-in Metal standard library.
pub fn clang_check(clang: &str, tag: &str, source: &str) -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("ganbate-shader-msl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!(
        "{}-{}.cpp",
        tag.replace(['/', ':', ' '], "_"),
        CHECK_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, rewrite_constructors(source)).map_err(|e| e.to_string())?;
    // This file is shared by ganbate-shader's tests and the engine's; find the header from either.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let include = [root.join("tests/msl"), root.join("crates/shader/tests/msl")]
        .into_iter()
        .find(|d| d.join("metal_stdlib").exists())
        .ok_or("tests/msl/metal_stdlib not found")?;
    let out = Command::new(clang)
        .args([
            "-std=c++14",
            "-fsyntax-only",
            "-Wall",
            "-Wno-unknown-attributes",
            "-Wno-unused-variable",
            "-Wno-unused-function",
            "-Wno-unused-but-set-variable",
            "-Wno-self-assign",
            "-Wno-self-assign-overloaded",
            "-Wno-tautological-compare",
            "-Werror=return-type",
            "-Werror=uninitialized",
            "-Werror=sometimes-uninitialized",
            "-Werror=conversion",
            "-Werror=shadow",
            "-DGANBATE_MSL_LAYOUT_CHECKS",
            "-x",
            "c++",
        ])
        .arg("-I")
        .arg(&include)
        .arg(&path)
        .output()
        .map_err(|e| e.to_string())?;
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    if out.status.success() {
        let _ = std::fs::remove_file(&path);
        Ok(())
    } else {
        Err(format!("{}\n(source kept at {})", stderr, path.display()))
    }
}
