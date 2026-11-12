//! `#import` module composition for WGSL shader libraries.
//!
//! Expands `#import module_name` or `#import "module_name"` directives recursively in
//! dependency order with diamond-import deduplication and cycle detection. Directive lines
//! inside `/* ... */` block comments or `//` line comments are ignored.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use crate::{CompileOptions, Compiled, Error, SpecValue, Stage};

/// Named collection of reusable WGSL modules (`common`, `brdf`, `noise`, etc.) that shaders
/// can pull in via `#import <name>`.
#[derive(Clone, Debug, Default)]
pub struct ShaderLibrary {
    modules: Vec<(String, String)>,
}

impl ShaderLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_module(mut self, name: impl Into<String>, source: impl Into<String>) -> Self {
        self.add(name, source);
        self
    }

    /// Registers or replaces a named shader module.
    pub fn add(&mut self, name: impl Into<String>, source: impl Into<String>) {
        let name = name.into();
        let source = source.into();
        if let Some((_, existing)) = self.modules.iter_mut().find(|(k, _)| *k == name) {
            *existing = source;
        } else {
            self.modules.push((name, source));
        }
    }

    pub fn pairs(&self) -> Vec<(&str, &str)> {
        self.modules
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    }

    /// Expands all `#import` directives in `source` using the registered modules.
    pub fn compose(&self, source: &str) -> Result<String, Error> {
        let pairs = self.pairs();
        compose(source, &pairs)
    }

    pub fn compile(
        &self,
        source: &str,
        stage: Stage,
        entry: &str,
        specialize: &[(&str, SpecValue)],
    ) -> Result<Compiled, Error> {
        let pairs = self.pairs();
        crate::compile_with_options(
            source,
            stage,
            entry,
            &CompileOptions {
                imports: &pairs,
                specialize,
            },
        )
    }

    pub fn compile_msl(
        &self,
        source: &str,
        stage: Stage,
        entry: &str,
        specialize: &[(&str, SpecValue)],
    ) -> Result<Compiled, Error> {
        let pairs = self.pairs();
        crate::compile_msl_with_options(
            source,
            stage,
            entry,
            &CompileOptions {
                imports: &pairs,
                specialize,
            },
        )
    }

    pub fn compile_wgsl(
        &self,
        source: &str,
        stage: Stage,
        entry: &str,
        specialize: &[(&str, SpecValue)],
    ) -> Result<Compiled, Error> {
        let pairs = self.pairs();
        crate::compile_wgsl_with_options(
            source,
            stage,
            entry,
            &CompileOptions {
                imports: &pairs,
                specialize,
            },
        )
    }

    pub fn compile_glsl(
        &self,
        source: &str,
        stage: Stage,
        entry: &str,
        specialize: &[(&str, SpecValue)],
    ) -> Result<Compiled, Error> {
        let pairs = self.pairs();
        crate::compile_glsl_with_options(
            source,
            stage,
            entry,
            &CompileOptions {
                imports: &pairs,
                specialize,
            },
        )
    }
}

/// Expands `#import` directives in `source` using `imports` (`(module_name, wgsl_source)`).
pub fn compose(source: &str, imports: &[(&str, &str)]) -> Result<String, Error> {
    compose_cow(source, imports).map(Cow::into_owned)
}

pub(crate) fn compose_cow<'a>(
    source: &'a str,
    imports: &[(&str, &str)],
) -> Result<Cow<'a, str>, Error> {
    if !source.contains("#import") && !source.contains("#define_import_path") {
        return Ok(Cow::Borrowed(source));
    }
    let mut map: HashMap<&str, &str> = HashMap::with_capacity(imports.len());
    for &(name, body) in imports {
        map.insert(name, body);
    }
    let mut included: HashSet<String> = HashSet::new();
    let mut stack: Vec<String> = Vec::new();
    let mut out = String::with_capacity(source.len() + 1024);
    expand_unit(source, &map, &mut included, &mut stack, &mut out)?;
    Ok(Cow::Owned(out))
}

const MAX_IMPORT_DEPTH: usize = 64;

fn expand_unit(
    source: &str,
    map: &HashMap<&str, &str>,
    included: &mut HashSet<String>,
    stack: &mut Vec<String>,
    out: &mut String,
) -> Result<(), Error> {
    let mut comment_depth = 0usize;
    let lines: Vec<&str> = source.split('\n').collect();
    for (idx, raw_line) in lines.iter().enumerate() {
        let line_no = (idx + 1) as u32;
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if comment_depth == 0
            && let Some((_, defined)) = try_parse_directive(line, line_no, "#define_import_path")?
        {
            included.insert(defined.to_owned());
            if idx + 1 < lines.len() {
                out.push('\n');
            }
            continue;
        }
        if comment_depth == 0
            && let Some((col, parsed)) = try_parse_directive(line, line_no, "#import")?
        {
            if included.contains(parsed) {
                if idx + 1 < lines.len() {
                    out.push('\n');
                }
                continue;
            }
            let Some((&canon_name, &target_src)) = map.get_key_value(parsed) else {
                return Err(Error {
                    line: line_no,
                    column: col,
                    message: format!("unknown #import module '{parsed}'"),
                });
            };
            if stack.iter().any(|s| s == canon_name) {
                return Err(Error {
                    line: line_no,
                    column: col,
                    message: format!("cyclic #import of module '{canon_name}'"),
                });
            }
            if stack.len() >= MAX_IMPORT_DEPTH {
                return Err(Error {
                    line: line_no,
                    column: col,
                    message: format!("#import nesting exceeds {MAX_IMPORT_DEPTH} levels"),
                });
            }
            stack.push(canon_name.to_owned());
            expand_unit(target_src, map, included, stack, out)?;
            stack.pop();
            included.insert(canon_name.to_owned());
            if !out.ends_with('\n') {
                out.push('\n');
            }
            continue;
        }

        advance_block_comments(line, &mut comment_depth);
        out.push_str(raw_line);
        if idx + 1 < lines.len() {
            out.push('\n');
        }
    }
    Ok(())
}

/// Returns `Some((1_based_name_column, module_name))` if `line` starts with `directive`
/// (`#import` or `#define_import_path`).
fn try_parse_directive<'a>(
    line: &'a str,
    line_no: u32,
    directive: &str,
) -> Result<Option<(u32, &'a str)>, Error> {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t') {
        i += 1;
    }
    let rest = &line[i..];
    let Some(after_kw) = rest.strip_prefix(directive) else {
        return Ok(None);
    };
    if let Some(first) = after_kw.chars().next()
        && (first.is_ascii_alphanumeric() || first == '_')
    {
        return Ok(None);
    }

    let kw_col = (i + 1) as u32;
    let mut j = i + directive.len();
    while j < bytes.len() && matches!(bytes[j], b' ' | b'\t') {
        j += 1;
    }
    if j >= bytes.len() || line[j..].starts_with("//") {
        return Err(Error {
            line: line_no,
            column: kw_col,
            message: format!("expected module name after '{directive}'"),
        });
    }

    let name_col = (j + 1) as u32;
    let (name, mut end) = if bytes[j] == b'"' {
        let start = j + 1;
        let Some(rel_close) = line[start..].find('"') else {
            return Err(Error {
                line: line_no,
                column: name_col,
                message: format!("unterminated string in '{directive}'"),
            });
        };
        let close = start + rel_close;
        (&line[start..close], close + 1)
    } else {
        let start = j;
        let mut k = start;
        while k < bytes.len() && is_module_char(bytes[k]) {
            k += 1;
        }
        (&line[start..k], k)
    };

    if name.is_empty() || !name.bytes().all(is_module_char) {
        return Err(Error {
            line: line_no,
            column: name_col,
            message: format!("invalid module name in '{directive}'"),
        });
    }

    while end < bytes.len() && matches!(bytes[end], b' ' | b'\t') {
        end += 1;
    }
    if end < bytes.len() && bytes[end] == b';' {
        end += 1;
        while end < bytes.len() && matches!(bytes[end], b' ' | b'\t') {
            end += 1;
        }
    }
    if end < bytes.len() && !line[end..].starts_with("//") {
        return Err(Error {
            line: line_no,
            column: (end + 1) as u32,
            message: format!("unexpected tokens after '{directive}' directive"),
        });
    }
    Ok(Some((name_col, name)))
}

fn is_module_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'/' | b'.')
}

fn advance_block_comments(line: &str, depth: &mut usize) {
    let b = line.as_bytes();
    let mut i = 0usize;
    while i + 1 < b.len() {
        if *depth == 0 && b[i] == b'/' && b[i + 1] == b'/' {
            break;
        }
        if b[i] == b'/' && b[i + 1] == b'*' {
            *depth += 1;
            i += 2;
        } else if *depth > 0 && b[i] == b'*' && b[i + 1] == b'/' {
            *depth -= 1;
            i += 2;
        } else {
            i += 1;
        }
    }
}
