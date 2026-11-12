use super::*;

impl<'a> Checker<'a> {
    // ------------------------------------------------------------------ expressions

    pub(crate) fn value(&mut self, e: &ast::Expr) -> R<Expr> {
        let c = self.expr(e)?;
        self.load(c, e.span)
    }

    pub(crate) fn load(&mut self, c: Checked, span: Span) -> R<Expr> {
        match c {
            Checked::Value(v) => {
                if v.ty == Ty::Void {
                    return Err(span.error("this call does not return a value"));
                }
                Ok(v)
            }
            Checked::Ref(place) => {
                if !is_constructible(&place.ty, &self.structs) {
                    return Err(span.error(format!(
                        "cannot load a value of type {}; index it instead",
                        self.ty_name(&place.ty)
                    )));
                }
                Ok(Expr {
                    ty: place.ty.clone(),
                    kind: EK::Load(Box::new(place)),
                    span,
                })
            }
            Checked::Swizzle(place, comps) => {
                let s = place.ty.scalar().expect("vector");
                let base = Expr {
                    ty: place.ty.clone(),
                    kind: EK::Load(Box::new(place)),
                    span,
                };
                Ok(Expr {
                    ty: Ty::Vector(comps.len() as u8, s),
                    kind: EK::Swizzle(Box::new(base), comps),
                    span,
                })
            }
            Checked::Handle(_, ty) => Err(span.error(format!(
                "a {} can only be used as an argument of a texture built-in function",
                self.ty_name(&ty)
            ))),
        }
    }

    pub(crate) fn expr(&mut self, e: &ast::Expr) -> R<Checked> {
        let span = e.span;
        Ok(match &e.kind {
            AK::Int(lit) => Checked::Value(match *lit {
                IntLit::Abstract(v) => konst(
                    Ty::Scalar(Scalar::AbstractInt),
                    ConstValue::AbstractInt(v),
                    span,
                ),
                IntLit::I32(v) => konst(I32, ConstValue::I32(v), span),
                IntLit::U32(v) => konst(U32, ConstValue::U32(v), span),
            }),
            AK::Float(lit) => Checked::Value(match *lit {
                FloatLit::Abstract(v) => konst(
                    Ty::Scalar(Scalar::AbstractFloat),
                    ConstValue::AbstractFloat(v),
                    span,
                ),
                FloatLit::F32(v) => konst(F32, ConstValue::F32(v), span),
            }),
            AK::Bool(b) => Checked::Value(konst(BOOL, ConstValue::Bool(*b), span)),
            AK::Ident { name, templates } => self.ident(name, templates.is_some(), span)?,
            AK::Unary(op, x) => Checked::Value(self.unary(*op, x, span)?),
            AK::Binary(op, l, r) => {
                let l = self.value(l)?;
                let r = self.value(r)?;
                Checked::Value(self.binary(*op, l, r, span)?)
            }
            AK::Call {
                name,
                templates,
                args,
            } => Checked::Value(self.call(name, templates.as_deref(), args, span, false)?),
            AK::Member(base, m) => self.member(base, m)?,
            AK::Index(base, idx) => self.index(base, idx, span)?,
        })
    }

    pub(super) fn ident(&mut self, name: &str, templated: bool, span: Span) -> R<Checked> {
        if templated {
            return Err(span.error(format!(
                "'{name}<...>' is a type; call it to construct a value"
            )));
        }
        if let Some(b) = self.lookup_local(name, span)? {
            return Ok(match b {
                Binding::Value { id, ty, .. } => Checked::Value(Expr {
                    ty,
                    kind: EK::Value(id),
                    span,
                }),
                Binding::Var { id, ty } => Checked::Ref(Place {
                    root: PlaceRoot::Local(id),
                    steps: Vec::new(),
                    ty,
                }),
                Binding::Const { ty, value } => Checked::Value(konst(ty, value, span)),
            });
        }
        match self.module_item(name)? {
            Some(Resolved::Const(ty, v)) => Ok(Checked::Value(konst(ty, v, span))),
            Some(Resolved::Global(gid)) => {
                let Some(fcx) = self.fcx.as_mut() else {
                    return Err(span.error(format!(
                        "'{name}' is a variable and cannot be used in a constant expression"
                    )));
                };
                fcx.globals.insert(gid);
                let g = &self.globals[gid];
                Ok(if g.kind == GlobalKind::Handle {
                    Checked::Handle(gid, g.ty.clone())
                } else {
                    Checked::Ref(Place {
                        root: PlaceRoot::Global(gid),
                        steps: Vec::new(),
                        ty: g.ty.clone(),
                    })
                })
            }
            Some(Resolved::Fn(_)) => Err(span.error(format!("function '{name}' must be called"))),
            Some(Resolved::Struct(_) | Resolved::Alias(_)) => Err(span.error(format!(
                "type '{name}' cannot be used as a value; call it to construct one"
            ))),
            Some(_) => Err(span.error(format!("'{name}' cannot be used as a value"))),
            None => {
                if matches!(self.predeclared_type(name, None, span), Ok(Some(_)))
                    || matches!(name, "vec2" | "vec3" | "vec4" | "array")
                {
                    Err(span.error(format!(
                        "type '{name}' cannot be used as a value; call it to construct one"
                    )))
                } else {
                    Err(span.error(format!("unknown identifier '{name}'")))
                }
            }
        }
    }

    pub(super) fn extract(v: Expr, i: u32, ty: Ty, span: Span) -> Expr {
        if let EK::Const(c) = &v.kind {
            return konst(ty, c.components()[i as usize].clone(), span);
        }
        Expr {
            ty,
            kind: EK::Extract(Box::new(v), i),
            span,
        }
    }

    pub(super) fn member(&mut self, base: &ast::Expr, m: &ast::Ident) -> R<Checked> {
        let b = match self.expr(base)? {
            c @ (Checked::Swizzle(..) | Checked::Handle(..)) => {
                Checked::Value(self.load(c, base.span)?)
            }
            other => other,
        };
        let ty = match &b {
            Checked::Ref(p) => p.ty.clone(),
            Checked::Value(v) => v.ty.clone(),
            _ => unreachable!(),
        };
        match &ty {
            Ty::Struct(sid) => {
                let s = &self.structs[*sid];
                let Some(i) = s.members.iter().position(|x| x.name == m.name) else {
                    return Err(m
                        .span
                        .error(format!("struct '{}' has no member '{}'", s.name, m.name)));
                };
                let mty = s.members[i].ty.clone();
                Ok(match b {
                    Checked::Ref(mut p) => {
                        p.steps.push(Step::Member(i as u32));
                        p.ty = mty;
                        Checked::Ref(p)
                    }
                    Checked::Value(v) => Checked::Value(Self::extract(v, i as u32, mty, m.span)),
                    _ => unreachable!(),
                })
            }
            Ty::Vector(n, s) => {
                let comps = parse_swizzle(&m.name, *n).ok_or_else(|| {
                    m.span.error(format!(
                        "invalid swizzle '.{}' for {}",
                        m.name,
                        self.ty_name(&ty)
                    ))
                })?;
                if comps.len() == 1 {
                    Ok(match b {
                        Checked::Ref(mut p) => {
                            p.steps.push(Step::Index(konst(
                                U32,
                                ConstValue::U32(comps[0]),
                                m.span,
                            )));
                            p.ty = Ty::Scalar(*s);
                            Checked::Ref(p)
                        }
                        Checked::Value(v) => {
                            Checked::Value(Self::extract(v, comps[0], Ty::Scalar(*s), m.span))
                        }
                        _ => unreachable!(),
                    })
                } else {
                    Ok(match b {
                        Checked::Ref(p) => Checked::Swizzle(p, comps),
                        Checked::Value(v) => {
                            let rty = Ty::Vector(comps.len() as u8, *s);
                            if let EK::Const(c) = &v.kind {
                                let cs = c.components();
                                let picked =
                                    comps.iter().map(|&i| cs[i as usize].clone()).collect();
                                Checked::Value(konst(rty, ConstValue::Composite(picked), m.span))
                            } else {
                                Checked::Value(Expr {
                                    ty: rty,
                                    kind: EK::Swizzle(Box::new(v), comps),
                                    span: m.span,
                                })
                            }
                        }
                        _ => unreachable!(),
                    })
                }
            }
            _ => Err(m.span.error(format!(
                "cannot access member '{}' of a value of type {}",
                m.name,
                self.ty_name(&ty)
            ))),
        }
    }

    pub(super) fn index(&mut self, base: &ast::Expr, idx: &ast::Expr, span: Span) -> R<Checked> {
        let b = match self.expr(base)? {
            c @ (Checked::Swizzle(..) | Checked::Handle(..)) => {
                Checked::Value(self.load(c, base.span)?)
            }
            other => other,
        };
        let i = self.value(idx)?;
        let i = match i.ty {
            Ty::Scalar(Scalar::I32 | Scalar::U32) => i,
            Ty::Scalar(Scalar::AbstractInt) => {
                let v = i.as_const().and_then(|c| c.as_i64()).unwrap_or(-1);
                if !(0..=u32::MAX as i64).contains(&v) {
                    return Err(idx.span.error(format!("index {v} is out of bounds")));
                }
                konst(U32, ConstValue::U32(v as u32), idx.span)
            }
            _ => {
                return Err(idx.span.error(format!(
                    "an index must be i32 or u32, found {}",
                    self.ty_name(&i.ty)
                )));
            }
        };
        let const_idx = i.as_const().and_then(|c| c.as_i64());
        let ty = match &b {
            Checked::Ref(p) => p.ty.clone(),
            Checked::Value(v) => v.ty.clone(),
            _ => unreachable!(),
        };
        let (elem, len) = match &ty {
            Ty::Array(e, n) => ((**e).clone(), *n),
            Ty::Vector(n, s) => (Ty::Scalar(*s), Some(*n as u32)),
            Ty::Matrix(c, r) => (Ty::Vector(*r, Scalar::F32), Some(*c as u32)),
            _ => {
                return Err(span.error(format!(
                    "cannot index a value of type {}",
                    self.ty_name(&ty)
                )));
            }
        };
        if let Some(ci) = const_idx
            && (ci < 0 || len.is_some_and(|n| ci >= n as i64))
        {
            return Err(idx.span.error(format!(
                "index {ci} is out of bounds for {}",
                self.ty_name(&ty)
            )));
        }
        Ok(match b {
            Checked::Ref(mut p) => {
                p.steps.push(Step::Index(i));
                p.ty = elem;
                Checked::Ref(p)
            }
            Checked::Value(v) => {
                if let Some(ci) = const_idx {
                    Checked::Value(Self::extract(v, ci as u32, elem, span))
                } else if v.as_const().is_some() && matches!(v.ty, Ty::Array(..) | Ty::Matrix(..)) {
                    let v = self.concretize(v)?;
                    let EK::Const(c) = v.kind else { unreachable!() };
                    let name = match &base.kind {
                        AK::Ident { name, .. } => name.clone(),
                        _ => "const_data".to_string(),
                    };
                    let gid = self.const_data_global(v.ty.clone(), c, name, base.span);
                    let fcx = self.fcx.as_mut().ok_or_else(|| {
                        span.error("a constant expression cannot index with a non-constant index")
                    })?;
                    fcx.globals.insert(gid);
                    Checked::Ref(Place {
                        root: PlaceRoot::Global(gid),
                        steps: vec![Step::Index(i)],
                        ty: elem.concretize(),
                    })
                } else {
                    let v = self.concretize(v)?;
                    let elem = elem.concretize();
                    Checked::Value(Expr {
                        ty: elem,
                        kind: EK::DynIndex(Box::new(v), Box::new(i)),
                        span,
                    })
                }
            }
            _ => unreachable!(),
        })
    }

    fn const_data_global(
        &mut self,
        ty: Ty,
        value: ConstValue,
        name: String,
        span: Span,
    ) -> GlobalId {
        if let Some((_, _, g)) = self
            .const_data
            .iter()
            .find(|(t, v, _)| *t == ty && *v == value)
        {
            return *g;
        }
        self.globals.push(ir::Global {
            name,
            span,
            kind: GlobalKind::ConstData,
            ty: ty.clone(),
            binding: None,
            init: Some(value.clone()),
        });
        let g = self.globals.len() - 1;
        self.const_data.push((ty, value, g));
        g
    }

    pub(super) fn unary(&mut self, op: ast::UnaryOp, x: &ast::Expr, span: Span) -> R<Expr> {
        let iop = match op {
            ast::UnaryOp::AddrOf => {
                return Err(span.error(
                    "pointers are not supported by Ganbate ('&' is only allowed as the argument of arrayLength)",
                ));
            }
            ast::UnaryOp::Deref => return Err(span.error("pointers are not supported by Ganbate")),
            ast::UnaryOp::Neg => ir::UnaryOp::Neg,
            ast::UnaryOp::Not => ir::UnaryOp::Not,
            ast::UnaryOp::BitNot => ir::UnaryOp::BitNot,
        };
        let v = self.value(x)?;
        let ok = v.ty.is_scalar_or_vector()
            && match (iop, v.ty.scalar()) {
                (ir::UnaryOp::Neg, Some(s)) => s.is_numeric() && s != Scalar::U32,
                (ir::UnaryOp::Not, Some(s)) => s == Scalar::Bool,
                (ir::UnaryOp::BitNot, Some(s)) => s.is_int(),
                _ => false,
            };
        if !ok {
            let sym = match iop {
                ir::UnaryOp::Neg => "-",
                ir::UnaryOp::Not => "!",
                ir::UnaryOp::BitNot => "~",
            };
            return Err(span.error(format!(
                "operator '{sym}' cannot be applied to {}",
                self.ty_name(&v.ty)
            )));
        }
        if let Some(c) = v.as_const() {
            let r = consteval::unary(iop, c).map_err(|m| span.error(m))?;
            return Ok(konst(v.ty, r, span));
        }
        Ok(Expr {
            ty: v.ty.clone(),
            kind: EK::Unary(iop, Box::new(v)),
            span,
        })
    }

    /// Convert abstract operands of a binary operator towards the other operand's type.
    pub(super) fn unify(&self, l: Expr, r: Expr, span: Span) -> R<(Expr, Expr)> {
        let la = l.ty.is_abstract();
        let ra = r.ty.is_abstract();
        let towards = |a: Expr, other: &Ty| -> R<Expr> {
            let (Some(from), Some(to)) = (a.ty.leaf_scalar(), other.scalar()) else {
                return Ok(a);
            };
            if from.converts_to(to) {
                let t = a.ty.with_scalar(to);
                self.convert(a, &t, span)
            } else {
                Ok(a)
            }
        };
        Ok(match (la, ra) {
            (true, false) => {
                let rt = r.ty.clone();
                (towards(l, &rt)?, r)
            }
            (false, true) => {
                let lt = l.ty.clone();
                let r = towards(r, &lt)?;
                (l, r)
            }
            (true, true) => {
                let (ls, rs) = (l.ty.leaf_scalar(), r.ty.leaf_scalar());
                if ls == Some(Scalar::AbstractInt) && rs == Some(Scalar::AbstractFloat) {
                    let t = l.ty.with_scalar(Scalar::AbstractFloat);
                    (self.convert(l, &t, span)?, r)
                } else if ls == Some(Scalar::AbstractFloat) && rs == Some(Scalar::AbstractInt) {
                    let t = r.ty.with_scalar(Scalar::AbstractFloat);
                    let r = self.convert(r, &t, span)?;
                    (l, r)
                } else {
                    (l, r)
                }
            }
            (false, false) => (l, r),
        })
    }

    fn arith_type(op: BinaryOp, lt: &Ty, rt: &Ty) -> Option<Ty> {
        use BinaryOp::*;
        match (lt, rt) {
            (Ty::Scalar(a), Ty::Scalar(b)) if a == b && a.is_numeric() => Some(lt.clone()),
            (Ty::Vector(n, a), Ty::Vector(m, b)) if n == m && a == b && a.is_numeric() => {
                Some(lt.clone())
            }
            (Ty::Vector(n, a), Ty::Scalar(b)) | (Ty::Scalar(b), Ty::Vector(n, a))
                if a == b && a.is_numeric() =>
            {
                Some(Ty::Vector(*n, *a))
            }
            (Ty::Matrix(c, r), Ty::Matrix(c2, r2))
                if matches!(op, Add | Sub) && c == c2 && r == r2 =>
            {
                Some(lt.clone())
            }
            (Ty::Matrix(c1, r1), Ty::Matrix(c2, r2)) if op == Mul && c1 == r2 => {
                Some(Ty::Matrix(*c2, *r1))
            }
            (Ty::Matrix(..), Ty::Scalar(Scalar::F32)) if op == Mul => Some(lt.clone()),
            (Ty::Scalar(Scalar::F32), Ty::Matrix(..)) if op == Mul => Some(rt.clone()),
            (Ty::Matrix(c, r), Ty::Vector(n, Scalar::F32)) if op == Mul && n == c => {
                Some(Ty::Vector(*r, Scalar::F32))
            }
            (Ty::Vector(n, Scalar::F32), Ty::Matrix(c, r)) if op == Mul && n == r => {
                Some(Ty::Vector(*c, Scalar::F32))
            }
            _ => None,
        }
    }

    pub(crate) fn binary(&mut self, op: BinaryOp, l: Expr, r: Expr, span: Span) -> R<Expr> {
        use BinaryOp::*;
        let (l, r) = if matches!(op, Shl | Shr) {
            let r = if r.ty.is_abstract() {
                let t = r.ty.with_scalar(Scalar::U32);
                self.convert(r, &t, span)?
            } else {
                r
            };
            let l = if l.ty.is_abstract() && r.as_const().is_none() {
                self.concretize(l)?
            } else {
                l
            };
            (l, r)
        } else {
            self.unify(l, r, span)?
        };
        let (lt, rt) = (l.ty.clone(), r.ty.clone());
        let mismatch = || {
            span.error(format!(
                "operator '{}' cannot be applied to {} and {}",
                op.symbol(),
                type_name(&lt, &self.structs),
                type_name(&rt, &self.structs)
            ))
        };
        let result = match op {
            Add | Sub | Mul | Div | Rem => Self::arith_type(op, &lt, &rt).ok_or_else(mismatch)?,
            Eq | Ne | Lt | Le | Gt | Ge => {
                if lt != rt || !lt.is_scalar_or_vector() {
                    return Err(mismatch());
                }
                if !matches!(op, Eq | Ne) && lt.scalar() == Some(Scalar::Bool) {
                    return Err(mismatch());
                }
                lt.with_scalar(Scalar::Bool)
            }
            And | Or => {
                let s = lt.scalar();
                if lt != rt
                    || !lt.is_scalar_or_vector()
                    || !s.is_some_and(|s| s.is_int() || s == Scalar::Bool)
                {
                    return Err(mismatch());
                }
                lt.clone()
            }
            Xor => {
                if lt != rt || !lt.is_scalar_or_vector() || !lt.scalar().is_some_and(|s| s.is_int())
                {
                    return Err(mismatch());
                }
                lt.clone()
            }
            LogicalAnd | LogicalOr => {
                if lt != BOOL || rt != BOOL {
                    return Err(span.error(format!(
                        "operator '{}' requires bool operands, found {} and {} (use '{}' for vectors of bool)",
                        op.symbol(),
                        self.ty_name(&lt),
                        self.ty_name(&rt),
                        if op == LogicalAnd { "&" } else { "|" }
                    )));
                }
                BOOL
            }
            Shl | Shr => {
                let ok = lt.is_scalar_or_vector()
                    && lt.scalar().is_some_and(|s| s.is_int())
                    && rt == lt.with_scalar(Scalar::U32);
                if !ok {
                    return Err(span.error(format!(
                        "operator '{}' needs an integer left operand and a u32 shift amount of the same shape, found {} and {}",
                        op.symbol(),
                        self.ty_name(&lt),
                        self.ty_name(&rt)
                    )));
                }
                lt.clone()
            }
        };
        if let (Some(a), Some(b)) = (l.as_const(), r.as_const()) {
            match consteval::binary(op, a, &lt, b, &rt) {
                Ok(Some(v)) => return Ok(konst(result, v, span)),
                Ok(None) => {}
                Err(m) => return Err(span.error(m)),
            }
        }
        if result.is_abstract() {
            // Only reachable for unfoldable abstract operations; concretize and retry.
            let l = self.concretize(l)?;
            let r = self.concretize(r)?;
            return self.binary(op, l, r, span);
        }
        Ok(Expr {
            ty: result,
            kind: EK::Binary(op, Box::new(l), Box::new(r)),
            span,
        })
    }

    /// Automatic conversion of `e` to `to` (identity, or abstract → concrete).
    pub(crate) fn convert(&self, e: Expr, to: &Ty, span: Span) -> R<Expr> {
        if e.ty == *to {
            return Ok(e);
        }
        if e.ty.is_abstract()
            && same_shape(&e.ty, to)
            && let (Some(from), Some(ts)) = (e.ty.leaf_scalar(), to.leaf_scalar())
            && from.converts_to(ts)
            && let Some(c) = e.as_const()
        {
            let v = consteval::auto_convert(c, ts).map_err(|m| e.span.error(m))?;
            return Ok(konst(to.clone(), v, e.span));
        }
        let at = if e.span == Span::default() {
            span
        } else {
            e.span
        };
        Err(at.error(format!(
            "type mismatch: expected {}, found {}",
            self.ty_name(to),
            self.ty_name(&e.ty)
        )))
    }

    pub(crate) fn concretize(&self, e: Expr) -> R<Expr> {
        if !e.ty.is_abstract() {
            return Ok(e);
        }
        let to = e.ty.concretize();
        let span = e.span;
        self.convert(e, &to, span)
    }

    pub(super) fn call(
        &mut self,
        name: &str,
        templates: Option<&[ast::Expr]>,
        args: &[ast::Expr],
        span: Span,
        stmt: bool,
    ) -> R<Expr> {
        if self.lookup_local(name, span)?.is_some() {
            return Err(span.error(format!("'{name}' is a local declaration, not a function")));
        }
        if self.names.contains_key(name) {
            let no_tmpl = || -> R<()> {
                if templates.is_some() {
                    Err(span.error(format!("'{name}' does not take template arguments")))
                } else {
                    Ok(())
                }
            };
            return match self.module_item(name)? {
                Some(Resolved::Fn(fid)) => {
                    no_tmpl()?;
                    self.user_call(fid, args, span)
                }
                Some(Resolved::Struct(sid)) => {
                    no_tmpl()?;
                    self.construct(Ctor::Full(Ty::Struct(sid)), args, span)
                }
                Some(Resolved::Alias(t)) => {
                    no_tmpl()?;
                    self.construct(Ctor::Full(t), args, span)
                }
                _ => Err(span.error(format!("'{name}' is not a function or a type"))),
            };
        }
        if name == "bitcast" {
            return self.bitcast(templates, args, span);
        }
        let ctor = match (name, templates) {
            ("vec2" | "vec3" | "vec4", None) => Some(Ctor::Vec(name.as_bytes()[3] - b'0')),
            ("array", None) => Some(Ctor::Array),
            (n, None) if mat_name(n).is_some_and(|m| m.2.is_empty()) => {
                let (c, r, _) = mat_name(n).expect("matrix");
                Some(Ctor::Mat(c, r))
            }
            _ => self
                .predeclared_type(name, templates, span)?
                .map(Ctor::Full),
        };
        if let Some(c) = ctor {
            return self.construct(c, args, span);
        }
        if templates.is_some() {
            return Err(span.error(format!("'{name}' does not take template arguments")));
        }
        self.builtin_call(name, args, span, stmt)
    }

    fn user_call(&mut self, fid: FnId, args: &[ast::Expr], span: Span) -> R<Expr> {
        if self.fcx.is_none() {
            return Err(span.error("function calls are not allowed in constant expressions"));
        }
        let f = &self.functions[fid];
        let fname = f.name.clone();
        if f.stage.is_some() {
            return Err(span.error(format!("entry point '{fname}' cannot be called")));
        }
        if args.len() != f.params.len() {
            return Err(span.error(format!(
                "function '{fname}' expects {} argument(s), got {}",
                f.params.len(),
                args.len()
            )));
        }
        let params: Vec<(String, Ty)> = f
            .params
            .iter()
            .map(|p| (p.name.clone(), p.ty.clone()))
            .collect();
        let ret = f.ret.clone();
        let mut out = Vec::new();
        for (a, (pname, pty)) in args.iter().zip(params) {
            let v = self.value(a)?;
            let v = self.convert(v, &pty, a.span).map_err(|e| Error {
                message: format!("argument '{pname}' of '{fname}': {}", e.message),
                ..e
            })?;
            out.push(v);
        }
        self.fcx_mut().calls.insert(fid);
        Ok(Expr {
            ty: ret.unwrap_or(Ty::Void),
            kind: EK::Call(fid, out),
            span,
        })
    }

    fn bitcast(
        &mut self,
        templates: Option<&[ast::Expr]>,
        args: &[ast::Expr],
        span: Span,
    ) -> R<Expr> {
        let Some([t]) = templates else {
            return Err(span.error("bitcast needs one template argument: bitcast<T>(e)"));
        };
        let to = self.resolve_type(t)?;
        if args.len() != 1 {
            return Err(span.error("bitcast takes exactly one argument"));
        }
        let v = self.value(&args[0])?;
        let v = self.concretize(v)?;
        let ok32 = |t: &Ty| {
            t.is_scalar_or_vector()
                && matches!(t.scalar(), Some(Scalar::I32 | Scalar::U32 | Scalar::F32))
        };
        if !ok32(&to) || !ok32(&v.ty) || v.ty.vec_size() != to.vec_size() {
            return Err(span.error(format!(
                "cannot bitcast {} to {}",
                self.ty_name(&v.ty),
                self.ty_name(&to)
            )));
        }
        if v.ty == to {
            return Ok(v);
        }
        if let Some(c) = v.as_const() {
            let s = to.scalar().expect("scalar");
            let r = consteval::map_leaves(c, &mut |x| consteval::bitcast_scalar(x, s))
                .map_err(|m| span.error(m))?;
            return Ok(konst(to, r, span));
        }
        Ok(Expr {
            ty: to,
            kind: EK::Bitcast(Box::new(v)),
            span,
        })
    }

    /// The component type of an inferred `vecN(...)`.
    fn common_scalar(&self, vals: &[Expr], span: Span) -> R<Scalar> {
        let mut concrete: Option<Scalar> = None;
        let mut any_float = false;
        for v in vals {
            let s = match &v.ty {
                Ty::Scalar(s) | Ty::Vector(_, s) => *s,
                other => {
                    return Err(span.error(format!(
                        "vector components must be scalars or vectors, found {}",
                        self.ty_name(other)
                    )));
                }
            };
            if s.is_abstract() {
                any_float |= s == Scalar::AbstractFloat;
            } else if let Some(c) = concrete {
                if c != s {
                    return Err(span.error(format!(
                        "mixed component types {} and {}",
                        c.name(),
                        s.name()
                    )));
                }
            } else {
                concrete = Some(s);
            }
        }
        Ok(concrete.unwrap_or(if any_float {
            Scalar::AbstractFloat
        } else {
            Scalar::AbstractInt
        }))
    }

    /// The element type of an inferred `array(...)`.
    fn common_type(&self, vals: &[Expr], span: Span) -> R<Ty> {
        let mut t = vals[0].ty.clone();
        for v in &vals[1..] {
            if v.ty == t {
                continue;
            }
            let (a, b) = (t.leaf_scalar(), v.ty.leaf_scalar());
            if same_shape(&t, &v.ty)
                && let (Some(a), Some(b)) = (a, b)
            {
                if a.is_abstract() && a.converts_to(b) {
                    t = t.with_scalar(b);
                    continue;
                }
                if b.is_abstract() && b.converts_to(a) {
                    continue;
                }
            }
            return Err(span.error(format!(
                "array elements have different types: {} and {}",
                self.ty_name(&t),
                self.ty_name(&v.ty)
            )));
        }
        Ok(t)
    }

    pub(super) fn construct(&mut self, ctor: Ctor, args: &[ast::Expr], span: Span) -> R<Expr> {
        let vals: Vec<Expr> = args.iter().map(|a| self.value(a)).collect::<R<_>>()?;
        match ctor {
            Ctor::Full(ty) => self.construct_full(ty, vals, span),
            Ctor::Vec(n) => {
                if vals.is_empty() {
                    return Err(span.error(format!(
                        "cannot infer the component type of vec{n}(); write vec{n}<f32>()"
                    )));
                }
                if let [v] = vals.as_slice()
                    && let Ty::Vector(m, s) = v.ty
                    && m == n
                {
                    return self.construct_full(Ty::Vector(n, s), vals, span);
                }
                let s = self.common_scalar(&vals, span)?;
                self.construct_full(Ty::Vector(n, s), vals, span)
            }
            Ctor::Mat(c, r) => self.construct_full(Ty::Matrix(c, r), vals, span),
            Ctor::Array => {
                if vals.is_empty() {
                    return Err(span.error("cannot infer the type of array(); write array<T, N>()"));
                }
                let e = self.common_type(&vals, span)?;
                if !is_constructible(&e, &self.structs) {
                    return Err(span.error("array elements must be constructible"));
                }
                let n = vals.len() as u32;
                self.construct_full(Ty::Array(Box::new(e), Some(n)), vals, span)
            }
        }
    }

    fn fold_construct(ty: Ty, comps: Vec<Expr>, span: Span) -> Expr {
        if comps.iter().all(|c| c.as_const().is_some()) {
            let flatten = matches!(ty, Ty::Vector(..));
            let mut out = Vec::new();
            for c in &comps {
                let v = c.as_const().expect("const").clone();
                match (flatten, v) {
                    (true, ConstValue::Composite(parts)) => out.extend(parts),
                    (_, v) => out.push(v),
                }
            }
            return konst(ty, ConstValue::Composite(out), span);
        }
        Expr {
            ty,
            kind: EK::Construct(comps),
            span,
        }
    }

    fn construct_full(&mut self, ty: Ty, vals: Vec<Expr>, span: Span) -> R<Expr> {
        let tn = self.ty_name(&ty);
        if vals.is_empty() {
            if !is_constructible(&ty, &self.structs) {
                return Err(span.error(format!("cannot construct a value of type {tn}")));
            }
            if leaf_count(&ty, &self.structs) > 4096 {
                return Ok(self.zero(&ty, span));
            }
            return Ok(konst(
                ty.clone(),
                ConstValue::zero(&ty, &self.structs),
                span,
            ));
        }
        match &ty {
            Ty::Scalar(to) => {
                let [v] = <[Expr; 1]>::try_from(vals)
                    .map_err(|_| span.error(format!("{tn}(...) takes exactly one argument")))?;
                if !v.ty.is_scalar() {
                    return Err(
                        span.error(format!("cannot convert {} to {tn}", self.ty_name(&v.ty)))
                    );
                }
                self.numeric_conversion(v, Ty::Scalar(*to), span)
            }
            Ty::Vector(n, s) => {
                if vals.len() == 1 {
                    match vals[0].ty {
                        Ty::Scalar(_) => {
                            let v = vals.into_iter().next().expect("one");
                            let v = self.convert(v, &Ty::Scalar(*s), span)?;
                            if let Some(c) = v.as_const() {
                                return Ok(konst(
                                    ty.clone(),
                                    ConstValue::Composite(vec![c.clone(); *n as usize]),
                                    span,
                                ));
                            }
                            return Ok(Expr {
                                ty: ty.clone(),
                                kind: EK::Splat(Box::new(v)),
                                span,
                            });
                        }
                        Ty::Vector(m, _) if m == *n => {
                            let v = vals.into_iter().next().expect("one");
                            return self.numeric_conversion(v, ty.clone(), span);
                        }
                        _ => {}
                    }
                }
                let mut comps = Vec::new();
                let mut count = 0u32;
                for v in vals {
                    let vt = v.ty.clone();
                    match vt {
                        Ty::Scalar(_) => {
                            comps.push(self.convert(v, &Ty::Scalar(*s), span)?);
                            count += 1;
                        }
                        Ty::Vector(m, _) => {
                            comps.push(self.convert(v, &Ty::Vector(m, *s), span)?);
                            count += m as u32;
                        }
                        other => {
                            return Err(span.error(format!(
                                "{tn} components must be scalars or vectors, found {}",
                                self.ty_name(&other)
                            )));
                        }
                    }
                }
                if count != *n as u32 {
                    return Err(span.error(format!("{tn} needs {n} components, got {count}")));
                }
                Ok(Self::fold_construct(ty.clone(), comps, span))
            }
            Ty::Matrix(c, r) => {
                if vals.len() == 1 && vals[0].ty == ty {
                    return Ok(vals.into_iter().next().expect("one"));
                }
                let col = Ty::Vector(*r, Scalar::F32);
                if vals.len() == *c as usize {
                    let cols = vals
                        .into_iter()
                        .map(|v| self.convert(v, &col, span))
                        .collect::<R<Vec<_>>>()?;
                    return Ok(Self::fold_construct(ty.clone(), cols, span));
                }
                if vals.len() == (*c as usize) * (*r as usize) {
                    let scalars = vals
                        .into_iter()
                        .map(|v| self.convert(v, &F32, span))
                        .collect::<R<Vec<_>>>()?;
                    let cols = scalars
                        .chunks(*r as usize)
                        .map(|ch| Self::fold_construct(col.clone(), ch.to_vec(), span))
                        .collect();
                    return Ok(Self::fold_construct(ty.clone(), cols, span));
                }
                Err(span.error(format!(
                    "{tn} needs {c} column vectors or {} scalars, got {} argument(s)",
                    c * r,
                    vals.len()
                )))
            }
            Ty::Array(e, Some(n)) => {
                if vals.len() != *n as usize {
                    return Err(span.error(format!("{tn} needs {n} elements, got {}", vals.len())));
                }
                let elems = vals
                    .into_iter()
                    .map(|v| self.convert(v, e, span))
                    .collect::<R<Vec<_>>>()?;
                Ok(Self::fold_construct(ty.clone(), elems, span))
            }
            Ty::Struct(sid) => {
                let mtys: Vec<(String, Ty)> = self.structs[*sid]
                    .members
                    .iter()
                    .map(|m| (m.name.clone(), m.ty.clone()))
                    .collect();
                if vals.len() != mtys.len() {
                    return Err(span.error(format!(
                        "{tn}(...) needs {} member values, got {}",
                        mtys.len(),
                        vals.len()
                    )));
                }
                let mut members = Vec::new();
                for (v, (mname, mty)) in vals.into_iter().zip(mtys) {
                    let v = self.convert(v, &mty, span).map_err(|e| Error {
                        message: format!("member '{mname}' of {tn}: {}", e.message),
                        ..e
                    })?;
                    members.push(v);
                }
                Ok(Self::fold_construct(ty.clone(), members, span))
            }
            _ => Err(span.error(format!("cannot construct a value of type {tn}"))),
        }
    }

    /// Value-constructor conversion between scalar types of the same shape.
    fn numeric_conversion(&self, v: Expr, to: Ty, span: Span) -> R<Expr> {
        if v.ty == to {
            return Ok(v);
        }
        let ts = to.scalar().expect("scalar");
        if let Some(c) = v.as_const() {
            let r = consteval::map_leaves(c, &mut |x| consteval::cast_scalar(x, ts))
                .map_err(|m| span.error(m))?;
            return Ok(konst(to, r, span));
        }
        Ok(Expr {
            ty: to,
            kind: EK::Convert(Box::new(v)),
            span,
        })
    }
}
