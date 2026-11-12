//! The step must compute the same bits on every platform, wasm included. Only `+`, `−`, `×`, `÷`
//! and `sqrt` are correctly rounded everywhere; the platform's transcendental functions, fused
//! multiply-add and `powi` are not, and iterating a hash map follows its random seed. This test
//! reads the crate's sources and refuses any of them outside comments.

use std::fs;
use std::path::Path;

const FORBIDDEN: &[&str] = &[
    ".sin(",
    ".cos(",
    ".tan(",
    ".sin_cos(",
    ".asin(",
    ".acos(",
    ".atan(",
    ".atan2(",
    ".sinh(",
    ".cosh(",
    ".tanh(",
    ".asinh(",
    ".acosh(",
    ".atanh(",
    ".exp(",
    ".exp2(",
    ".exp_m1(",
    ".ln(",
    ".log(",
    ".log2(",
    ".log10(",
    ".ln_1p(",
    ".powf(",
    ".powi(",
    ".cbrt(",
    ".hypot(",
    ".mul_add(",
    "HashMap",
    "HashSet",
    "Instant::",
    "SystemTime",
];

/// The file's code before its unit tests (which compare against the platform on purpose),
/// without comments.
fn code_without_comments(text: &str) -> String {
    let mut out = String::new();
    for line in text.lines() {
        if line.trim_start().starts_with("#[cfg(test)]") {
            break;
        }
        let line = match line.find("//") {
            Some(i) => &line[..i],
            None => line,
        };
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[test]
fn the_step_uses_only_correctly_rounded_arithmetic() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found = Vec::new();
    let mut files = 0;
    for entry in fs::read_dir(&src).expect("src") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        files += 1;
        let text = fs::read_to_string(&path).expect("read");
        for (n, line) in code_without_comments(&text).lines().enumerate() {
            for f in FORBIDDEN {
                if line.contains(f) {
                    found.push(format!(
                        "{}:{}: {f}",
                        path.file_name().unwrap().to_string_lossy(),
                        n + 1
                    ));
                }
            }
        }
    }
    assert!(files >= 10, "expected the crate's modules, read {files}");
    assert!(
        found.is_empty(),
        "platform-dependent math or nondeterministic iteration in the physics step:\n{}",
        found.join("\n")
    );
}
