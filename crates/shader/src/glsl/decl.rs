use super::*;

impl Emitter<'_> {
    // ------------------------------------------------------------------ types

    pub(super) fn ty(&self, t: &Ty, span: &crate::lexer::Span) -> R<String> {
        Ok(match t {
            Ty::Scalar(s) => scalar_name(*s).to_string(),
            Ty::Vector(n, s) => vector_name(*n, *s),
            Ty::Matrix(c, r) => matrix_name(*c, *r),
            Ty::Array(e, Some(n)) => {
                if matches!(**e, Ty::Array(..)) {
                    return Err(unsupported(span, "an array of arrays"));
                }
                format!("{}[{n}]", self.ty(e, span)?)
            }
            Ty::Array(_, None) => {
                return Err(unsupported(
                    span,
                    "a runtime-sized array outside a storage buffer",
                ));
            }
            Ty::Struct(id) => {
                if !self.plain[*id] {
                    return Err(unsupported(
                        span,
                        &format!(
                            "struct '{}' as a value (it holds a runtime-sized array or an array of arrays)",
                            self.prog.structs[*id].name
                        ),
                    ));
                }
                self.structs[*id].clone()
            }
            Ty::Void => "void".to_string(),
            other => {
                return Err(span.error(format!(
                    "internal compiler error (GLSL): no value type for {}",
                    type_name(other, &self.prog.structs)
                )));
            }
        })
    }

    pub(super) fn zero(&self, t: &Ty, span: &crate::lexer::Span) -> R<String> {
        Ok(match t {
            Ty::Scalar(s) => match s.concretize() {
                Scalar::Bool => "false".into(),
                Scalar::I32 => "0".into(),
                Scalar::U32 => "0u".into(),
                _ => "0.0".into(),
            },
            Ty::Vector(n, s) => {
                let z = self.zero(&Ty::Scalar(*s), span)?;
                format!("{}({z})", vector_name(*n, *s))
            }
            Ty::Matrix(c, r) => format!("{}(0.0)", matrix_name(*c, *r)),
            Ty::Array(e, Some(n)) => {
                let z = self.zero(e, span)?;
                let items = vec![z; *n as usize].join(", ");
                format!("{}({items})", self.ty(t, span)?)
            }
            Ty::Struct(id) => {
                let mut items = Vec::new();
                for m in &self.prog.structs[*id].members {
                    items.push(self.zero(&m.ty, span)?);
                }
                format!("{}({})", self.ty(t, span)?, items.join(", "))
            }
            other => {
                return Err(span.error(format!(
                    "internal compiler error (GLSL): no zero value for {}",
                    type_name(other, &self.prog.structs)
                )));
            }
        })
    }

    /// A constant of type `ty`.
    pub(super) fn konst(&self, ty: &Ty, v: &ConstValue, span: &crate::lexer::Span) -> R<String> {
        if let Some(s) = ty.scalar()
            && ty.is_scalar()
        {
            return Ok(scalar_lit(v, s));
        }
        if is_zero(v) {
            return self.zero(ty, span);
        }
        let parts = v.components();
        let mut items = Vec::with_capacity(parts.len());
        for (i, p) in parts.iter().enumerate() {
            let pt = match ty {
                Ty::Vector(_, s) => Ty::Scalar(*s),
                Ty::Matrix(_, r) => Ty::Vector(*r, Scalar::F32),
                Ty::Array(e, _) => (**e).clone(),
                Ty::Struct(sid) => self.prog.structs[*sid].members[i].ty.clone(),
                other => other.clone(),
            };
            items.push(self.konst(&pt, p, span)?);
        }
        Ok(format!("{}({})", self.ty(ty, span)?, items.join(", ")))
    }

    pub(super) fn struct_decl(&self, sid: usize, out: &mut String) -> R<()> {
        let s = &self.prog.structs[sid];
        let _ = writeln!(out, "struct {} {{", self.structs[sid]);
        for (i, m) in s.members.iter().enumerate() {
            let _ = writeln!(
                out,
                "  {} {};",
                self.ty(&m.ty, &m.span)?,
                self.members[sid][i]
            );
        }
        out.push_str("};\n\n");
        Ok(())
    }

    // ------------------------------------------------------------------ helpers

    /// Generates helper `key` once with `code` (which may generate other helpers first).
    pub(super) fn helper(&mut self, key: &str, code: impl FnOnce(&mut Self) -> R<String>) -> R<()> {
        if self.helper_keys.contains(key) {
            return Ok(());
        }
        self.helper_keys.insert(key.to_string());
        let text = code(self)?;
        self.helpers.push(text);
        Ok(())
    }

    /// The hidden uniform for `value`, declared once.
    pub(super) fn hidden(&mut self, name: String, value: HiddenValue) -> String {
        if !self.hidden.iter().any(|h| h.name == name) {
            self.hidden.push(Hidden {
                name: name.clone(),
                value,
            });
        }
        name
    }

    /// The combined sampler uniform for `texture` read through `sampler` (or unfiltered).
    pub(super) fn unit(&mut self, texture: GlobalId, sampler: Option<GlobalId>) -> R<String> {
        if let Some((_, _, u)) = self
            .units
            .iter()
            .find(|(t, s, _)| *t == texture && *s == sampler)
        {
            return Ok(u.name.clone());
        }
        let tg = &self.prog.globals[texture];
        let tb = tg.binding.ok_or_else(|| internal("unbound texture"))?;
        let Ty::Texture(view, sample) = tg.ty else {
            return Err(unsupported(&tg.span, "a storage texture"));
        };
        let (sb, compare) = match sampler {
            Some(s) => {
                let sg = &self.prog.globals[s];
                (
                    Some(sg.binding.ok_or_else(|| internal("unbound sampler"))?),
                    matches!(sg.ty, Ty::Sampler(true)),
                )
            }
            None => (None, false),
        };
        let name = match sb {
            Some((g, b)) => format!("ganbate_tex_{}_{}_{g}_{b}", tb.0, tb.1),
            None => format!("ganbate_tex_{}_{}", tb.0, tb.1),
        };
        self.units.push((
            texture,
            sampler,
            Unit {
                name: name.clone(),
                texture: tb,
                sampler: sb,
                view,
                sample,
                compare,
            },
        ));
        Ok(name)
    }

    /// The texture uniform of storage buffer `g`.
    pub(super) fn storage_buffer(&mut self, g: GlobalId) -> R<(String, (u32, u32))> {
        let gl = &self.prog.globals[g];
        let binding = gl.binding.ok_or_else(|| internal("unbound storage"))?;
        let name = format!("ganbate_buf_{}_{}", binding.0, binding.1);
        if !self.storage.iter().any(|(id, _)| *id == g) {
            self.storage.push((
                g,
                Storage {
                    name: name.clone(),
                    binding,
                },
            ));
        }
        Ok((name, binding))
    }

    /// A function `T ganbate_ld_KEY(usampler2D s, uint off)` that loads a `ty` laid out by WGSL's
    /// storage rules at byte `off`.
    pub(super) fn loader(&mut self, ty: &Ty, span: &crate::lexer::Span) -> R<String> {
        let key = self.type_key(ty, span)?;
        let name = format!("ganbate_ld_{key}");
        let words = |e: &mut Self| -> R<()> {
            let mask = STORAGE_ROW_TEXELS - 1;
            let shift = STORAGE_ROW_TEXELS.trailing_zeros();
            e.helper("texel", |_| {
                Ok(format!(
                    "uvec4 ganbate_texel(usampler2D s, uint off) {{\n  uint t = off >> 4u;\n  \
                     return texelFetch(s, ivec2(int(t & {mask}u), int(t >> {shift}u)), 0);\n}}\n"
                ))
            })?;
            e.helper("word", |_| {
                Ok("uint ganbate_word(usampler2D s, uint off) {\n  \
                    return ganbate_texel(s, off)[(off >> 2u) & 3u];\n}\n"
                    .into())
            })?;
            e.helper("pair", |_| {
                Ok("uvec2 ganbate_pair(usampler2D s, uint off) {\n  uvec4 t = ganbate_texel(s, off);\n  \
                    return ((off >> 3u) & 1u) == 0u ? t.xy : t.zw;\n}\n"
                    .into())
            })
        };
        let span = *span;
        let ty_owned = ty.clone();
        self.helper(&format!("ld_{key}"), move |e| {
            words(e)?;
            let t = e.ty(&ty_owned, &span)?;
            let body = match &ty_owned {
                Ty::Scalar(s) => {
                    let w = "ganbate_word(s, off)";
                    match s.concretize() {
                        Scalar::U32 => w.to_string(),
                        Scalar::I32 => format!("int({w})"),
                        Scalar::F32 => format!("uintBitsToFloat({w})"),
                        _ => return Err(internal("bool in a storage buffer")),
                    }
                }
                Ty::Vector(n, s) => {
                    let raw = match n {
                        2 => "ganbate_pair(s, off)".to_string(),
                        3 => "ganbate_texel(s, off).xyz".to_string(),
                        _ => "ganbate_texel(s, off)".to_string(),
                    };
                    match s.concretize() {
                        Scalar::U32 => raw,
                        Scalar::I32 => format!("{t}({raw})"),
                        Scalar::F32 => format!("uintBitsToFloat({raw})"),
                        _ => return Err(internal("bool in a storage buffer")),
                    }
                }
                Ty::Matrix(c, r) => {
                    let col = e.loader(&Ty::Vector(*r, Scalar::F32), &span)?;
                    let stride = matrix_stride(*r);
                    let cols: Vec<String> = (0..u32::from(*c))
                        .map(|i| format!("{col}(s, off + {}u)", i * stride))
                        .collect();
                    format!("{t}({})", cols.join(", "))
                }
                Ty::Array(elem, Some(n)) => {
                    let inner = e.loader(elem, &span)?;
                    let stride = array_stride(elem, &e.prog.structs);
                    return Ok(format!(
                        "{t} ganbate_ld_{key}(usampler2D s, uint off) {{\n  {t} r;\n  \
                         for (uint i = 0u; i < {n}u; i++) {{\n    r[i] = {inner}(s, off + i * {stride}u);\n  }}\n  \
                         return r;\n}}\n",
                        key = e.type_key(&ty_owned, &span)?
                    ));
                }
                Ty::Struct(sid) => {
                    let info = &e.prog.structs[*sid];
                    let layout = info
                        .layout
                        .clone()
                        .ok_or_else(|| internal("storage struct without a layout"))?;
                    let mut parts = Vec::new();
                    for (i, m) in info.members.clone().iter().enumerate() {
                        let f = e.loader(&m.ty, &span)?;
                        parts.push(format!("{f}(s, off + {}u)", layout.offsets[i]));
                    }
                    format!("{t}({})", parts.join(", "))
                }
                _ => return Err(internal("no storage loader for this type")),
            };
            Ok(format!(
                "{t} ganbate_ld_{key}(usampler2D s, uint off) {{\n  return {body};\n}}\n",
                key = e.type_key(&ty_owned, &span)?
            ))
        })?;
        Ok(name)
    }

    /// A short identifier fragment for a value type.
    fn type_key(&self, ty: &Ty, span: &crate::lexer::Span) -> R<String> {
        Ok(match ty {
            Ty::Scalar(s) => match s.concretize() {
                Scalar::U32 => "u".into(),
                Scalar::I32 => "i".into(),
                Scalar::Bool => "b".into(),
                _ => "f".into(),
            },
            Ty::Vector(n, s) => format!("v{n}{}", self.type_key(&Ty::Scalar(*s), span)?),
            Ty::Matrix(c, r) => format!("m{c}x{r}"),
            Ty::Array(e, Some(n)) => format!("a{n}_{}", self.type_key(e, span)?),
            Ty::Struct(id) => format!("s{id}"),
            other => {
                return Err(span.error(format!(
                    "internal compiler error (GLSL): no key for {}",
                    type_name(other, &self.prog.structs)
                )));
            }
        })
    }

    /// Generates a helper with one overload per type in `tys`, named `ganbate_NAME`, from a body
    /// template that sees the type as `T` (and `B` for the matching bool vector).
    pub(super) fn overload(
        &mut self,
        name: &str,
        ty: &Ty,
        code: &str,
        span: &crate::lexer::Span,
    ) -> R<String> {
        let t = self.ty(ty, span)?;
        let b = match ty {
            Ty::Vector(n, _) => vector_name(*n, Scalar::Bool),
            _ => "bool".into(),
        };
        let key = format!("{name}_{t}");
        let text = code.replace("$T", &t).replace("$B", &b);
        self.helper(&key, |_| Ok(text))?;
        Ok(format!("ganbate_{name}"))
    }

    // ------------------------------------------------------------------ functions

    pub(super) fn line(&mut self, text: &str) {
        for _ in 0..self.f.indent {
            self.out.push_str("  ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    pub(super) fn signature(&self, fid: FnId) -> R<String> {
        let f = &self.prog.functions[fid];
        let mut names = self.module.clone();
        let mut params = Vec::new();
        for p in &f.params {
            let name = names.claim(&p.name);
            params.push(format!("{} {name}", self.ty(&p.ty, &p.span)?));
        }
        let ret = match &f.ret {
            Some(t) => self.ty(t, &f.span)?,
            None => "void".into(),
        };
        Ok(format!("{ret} {}({})", self.fns[&fid], params.join(", ")))
    }

    pub(super) fn function(&mut self, fid: FnId) -> R<()> {
        let prog = self.prog;
        let f = &prog.functions[fid];
        self.current_fn = Some(fid);
        self.f = FnCx {
            names: self.module.clone(),
            indent: 1,
            ..FnCx::default()
        };
        // The same parameter names as the prototype.
        for p in &f.params {
            let name = self.f.names.claim(&p.name);
            self.f.values.insert(p.value, name);
        }
        let sig = self.signature(fid)?;
        let _ = writeln!(self.out, "{sig} {{");
        for l in &f.locals {
            let name = self.f.names.claim(&l.name);
            let ty = self.ty(&l.ty, &f.span)?;
            let z = self.zero(&l.ty, &f.span)?;
            self.line(&format!("{ty} {name} = {z};"));
            self.f.locals.push(name);
        }
        self.block(&f.body)?;
        self.out.push_str("}\n\n");
        Ok(())
    }

    // ------------------------------------------------------------------ statements

    /// Emits `stmts`; returns true when control cannot fall off the end. Statements after a
    /// terminator are unreachable and left out.
    fn block(&mut self, stmts: &[Stmt]) -> R<bool> {
        for s in stmts {
            if self.stmt(s)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn nested(&mut self, header: &str, stmts: &[Stmt]) -> R<bool> {
        self.line(header);
        self.f.indent += 1;
        let t = self.block(stmts)?;
        self.f.indent -= 1;
        Ok(t)
    }

    fn stmt(&mut self, s: &Stmt) -> R<bool> {
        match s {
            Stmt::Let(id, e) => {
                let v = self.expr(e)?;
                if self.f.hoisted.contains(id) {
                    let name = self.f.values[id].clone();
                    self.line(&format!("{name} = {v};"));
                } else {
                    let name = self.value_name(*id);
                    let ty = self.ty(&e.ty, &e.span)?;
                    self.line(&format!("{ty} {name} = {v};"));
                    self.f.values.insert(*id, name);
                }
                Ok(false)
            }
            Stmt::Assign {
                place,
                swizzle,
                value,
            } => {
                self.assign(place, swizzle.as_deref(), value)?;
                Ok(false)
            }
            Stmt::Eval(e) => {
                let v = self.expr(e)?;
                let statement = match &e.kind {
                    EK::Call(..) => true,
                    EK::Builtin(..) | EK::Texture(_) => e.ty == Ty::Void,
                    _ => false,
                };
                if statement || e.ty != Ty::Void {
                    self.line(&format!("{v};"));
                }
                Ok(false)
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                let c = self.expr(cond)?;
                let t = self.nested(&format!("if ({c}) {{"), then)?;
                let o = if otherwise.is_empty() {
                    false
                } else {
                    self.nested("} else {", otherwise)?
                };
                self.line("}");
                Ok(t && o)
            }
            Stmt::Loop {
                cond,
                body,
                continuing,
                break_if,
            } => {
                let has_continuing = !continuing.is_empty() || break_if.is_some();
                let mut first = None;
                if has_continuing {
                    // `let`s of the body that the continuing block reads.
                    let mut read = HashSet::new();
                    values_read(continuing, &mut read);
                    if let Some(b) = break_if {
                        walk(b, &mut |x| {
                            if let EK::Value(v) = x.kind {
                                read.insert(v);
                            }
                        });
                    }
                    for st in body {
                        if let Stmt::Let(id, e) = st
                            && read.contains(id)
                        {
                            let name = self.value_name(*id);
                            let ty = self.ty(&e.ty, &e.span)?;
                            let z = self.zero(&e.ty, &e.span)?;
                            self.line(&format!("{ty} {name} = {z};"));
                            self.f.values.insert(*id, name);
                            self.f.hoisted.insert(*id);
                        }
                    }
                    let flag = self.f.names.generated("ganbate_first");
                    self.line(&format!("bool {flag} = true;"));
                    first = Some(flag);
                }
                self.line("while (true) {");
                self.f.indent += 1;
                if let Some(flag) = &first {
                    self.line(&format!("if (!{flag}) {{"));
                    self.f.indent += 1;
                    self.block(continuing)?;
                    if let Some(b) = break_if {
                        let b = self.expr(b)?;
                        self.line(&format!("if ({b}) {{ break; }}"));
                    }
                    self.f.indent -= 1;
                    self.line("}");
                    self.line(&format!("{flag} = false;"));
                }
                if let Some(c) = cond {
                    let c = self.expr(c)?;
                    self.line(&format!("if (!({c})) {{ break; }}"));
                }
                self.block(body)?;
                self.f.indent -= 1;
                self.line("}");
                Ok(cond.is_none() && break_if.is_none() && !breaks(body))
            }
            Stmt::Break => {
                self.line("break;");
                Ok(true)
            }
            Stmt::Continue => {
                self.line("continue;");
                Ok(true)
            }
            Stmt::Return(e) => {
                match e {
                    Some(e) => {
                        let v = self.expr(e)?;
                        self.line(&format!("return {v};"));
                    }
                    None => self.line("return;"),
                }
                Ok(true)
            }
            Stmt::Discard => {
                self.line("discard;");
                Ok(true)
            }
            Stmt::Block(b) => {
                let t = self.nested("{", b)?;
                self.line("}");
                Ok(t)
            }
        }
    }

    /// A fresh name for `let` value `id` of the current function, based on its source name.
    fn value_name(&mut self, id: ValueId) -> String {
        let prog = self.prog;
        let base = self
            .current_fn
            .and_then(|f| prog.functions[f].value_names.get(id as usize))
            .map_or("v", String::as_str);
        self.f.names.claim(base)
    }
}
