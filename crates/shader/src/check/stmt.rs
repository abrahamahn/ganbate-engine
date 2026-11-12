use super::*;

impl<'a> Checker<'a> {
    // ------------------------------------------------------------------ statements

    pub(super) fn push_scope(&mut self) {
        self.fcx_mut().scopes.push(Vec::new());
    }

    pub(super) fn pop_scope(&mut self) {
        self.fcx_mut().scopes.pop();
    }

    pub(super) fn declare(&mut self, name: &ast::Ident, b: Binding) -> R<()> {
        let scope = self.fcx_mut().scopes.last_mut().expect("scope");
        if scope.iter().any(|(n, _)| *n == name.name) {
            return Err(name
                .span
                .error(format!("'{}' is already declared in this scope", name.name)));
        }
        scope.push((name.name.clone(), b));
        Ok(())
    }

    pub(super) fn lookup_local(&self, name: &str, span: Span) -> R<Option<Binding>> {
        let Some(fcx) = &self.fcx else {
            return Ok(None);
        };
        for scope in fcx.scopes.iter().rev() {
            for (n, b) in scope.iter().rev() {
                if n == name {
                    if let Binding::Value {
                        after_continue_of: Some(l),
                        ..
                    } = b
                        && fcx.continuing_of.contains(l)
                    {
                        return Err(span.error(format!(
                            "'{name}' cannot be used in this 'continuing' block: a 'continue' before its declaration can skip it"
                        )));
                    }
                    return Ok(Some(b.clone()));
                }
            }
        }
        Ok(None)
    }

    pub(super) fn check_stmts(&mut self, stmts: &[ast::Stmt], out: &mut Vec<Stmt>) -> R<bool> {
        let mut falls = true;
        for s in stmts {
            let f = self.stmt(s, out)?;
            falls &= f;
        }
        Ok(falls)
    }

    pub(super) fn scoped_block(&mut self, b: &ast::Block) -> R<(Vec<Stmt>, bool)> {
        self.push_scope();
        let mut out = Vec::new();
        let r = self.check_stmts(&b.stmts, &mut out);
        self.pop_scope();
        Ok((out, r?))
    }

    pub(super) fn condition(&mut self, e: &ast::Expr, what: &str) -> R<Expr> {
        let v = self.value(e)?;
        if v.ty != BOOL {
            return Err(e.span.error(format!(
                "{what} condition must be bool, found {}",
                self.ty_name(&v.ty)
            )));
        }
        Ok(v)
    }

    pub(crate) fn zero(&self, ty: &Ty, span: Span) -> Expr {
        Expr {
            ty: ty.clone(),
            kind: EK::Zero,
            span,
        }
    }

    fn stmt(&mut self, s: &ast::Stmt, out: &mut Vec<Stmt>) -> R<bool> {
        let span = s.span;
        match &s.kind {
            SK::Empty => Ok(true),
            SK::Let { name, ty, init } => {
                let v = self.value(init)?;
                let v = match ty {
                    Some(t) => {
                        let t = self.resolve_type(t)?;
                        self.convert(v, &t, init.span)?
                    }
                    None => self.concretize(v)?,
                };
                if !is_constructible(&v.ty, &self.structs) {
                    return Err(init.span.error(format!(
                        "a 'let' cannot hold a value of type {}",
                        self.ty_name(&v.ty)
                    )));
                }
                let ty = v.ty.clone();
                let fcx = self.fcx_mut();
                let id = fcx.value_names.len() as ValueId;
                fcx.value_names.push(name.name.clone());
                let cur = fcx.scopes.len() - 1;
                let after_continue_of = fcx
                    .loops
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, l)| l.body_scope == cur)
                    .filter(|(_, l)| l.continue_seen)
                    .map(|(i, _)| i);
                out.push(Stmt::Let(id, v));
                self.declare(
                    name,
                    Binding::Value {
                        id,
                        ty,
                        after_continue_of,
                    },
                )?;
                Ok(true)
            }
            SK::Var {
                name,
                space,
                ty,
                init,
            } => {
                if let Some(sp) = space {
                    let ok = matches!(sp.as_slice(), [e] if matches!(&e.kind, AK::Ident { name, templates: None } if name == "function"));
                    if !ok {
                        return Err(span.error("only var<function> is allowed inside a function"));
                    }
                }
                let declared = ty.as_ref().map(|t| self.resolve_type(t)).transpose()?;
                let iv = init.as_ref().map(|e| self.value(e)).transpose()?;
                let (ty, value) = match (declared, iv) {
                    (Some(t), Some(v)) => {
                        let v = self.convert(v, &t, init.as_ref().map_or(span, |e| e.span))?;
                        (t, v)
                    }
                    (Some(t), None) => {
                        let z = self.zero(&t, span);
                        (t, z)
                    }
                    (None, Some(v)) => {
                        let v = self.concretize(v)?;
                        (v.ty.clone(), v)
                    }
                    (None, None) => {
                        return Err(name.span.error("a 'var' needs a type or an initializer"));
                    }
                };
                if !is_constructible(&ty, &self.structs) {
                    return Err(name
                        .span
                        .error(format!("a variable cannot have type {}", self.ty_name(&ty))));
                }
                let fcx = self.fcx_mut();
                let id = fcx.locals.len() as LocalId;
                fcx.locals.push(ir::Local {
                    name: name.name.clone(),
                    ty: ty.clone(),
                });
                out.push(Stmt::Assign {
                    place: Place {
                        root: PlaceRoot::Local(id),
                        steps: Vec::new(),
                        ty: ty.clone(),
                    },
                    swizzle: None,
                    value,
                });
                self.declare(name, Binding::Var { id, ty })?;
                Ok(true)
            }
            SK::Const { name, ty, init } => {
                let (ty, value) = self.const_decl(ty.as_ref(), init)?;
                self.declare(name, Binding::Const { ty, value })?;
                Ok(true)
            }
            SK::Assign { lhs, op, rhs } => {
                let t = self.assign_target(lhs)?;
                let r = self.value(rhs)?;
                let value = match op {
                    None => self.convert(r, &t.ty, rhs.span)?,
                    Some(op) => {
                        let cur = Expr {
                            ty: t.ty.clone(),
                            kind: EK::Current,
                            span: lhs.span,
                        };
                        let v = self.binary(*op, cur, r, span)?;
                        if v.ty != t.ty {
                            return Err(span.error(format!(
                                "'{}=' produces {}, which cannot be assigned to {}",
                                op.symbol(),
                                self.ty_name(&v.ty),
                                self.ty_name(&t.ty)
                            )));
                        }
                        v
                    }
                };
                out.push(Stmt::Assign {
                    place: t.place,
                    swizzle: t.swizzle,
                    value,
                });
                Ok(true)
            }
            SK::Increment(e) | SK::Decrement(e) => {
                let inc = matches!(s.kind, SK::Increment(_));
                let t = self.assign_target(e)?;
                let one = match t.ty {
                    Ty::Scalar(Scalar::I32) => ConstValue::I32(1),
                    Ty::Scalar(Scalar::U32) => ConstValue::U32(1),
                    _ => {
                        return Err(span.error(format!(
                            "'{}' requires an i32 or u32 variable, found {}",
                            if inc { "++" } else { "--" },
                            self.ty_name(&t.ty)
                        )));
                    }
                };
                let value = Expr {
                    ty: t.ty.clone(),
                    kind: EK::Binary(
                        if inc { BinaryOp::Add } else { BinaryOp::Sub },
                        Box::new(Expr {
                            ty: t.ty.clone(),
                            kind: EK::Current,
                            span,
                        }),
                        Box::new(konst(t.ty.clone(), one, span)),
                    ),
                    span,
                };
                out.push(Stmt::Assign {
                    place: t.place,
                    swizzle: t.swizzle,
                    value,
                });
                Ok(true)
            }
            SK::Phony(e) => {
                match self.expr(e)? {
                    Checked::Handle(..) => {}
                    Checked::Ref(p) if !is_constructible(&p.ty, &self.structs) => {}
                    other => {
                        let v = self.load(other, e.span)?;
                        out.push(Stmt::Eval(v));
                    }
                }
                Ok(true)
            }
            SK::Call(e) => {
                let AK::Call {
                    name,
                    templates,
                    args,
                } = &e.kind
                else {
                    return Err(e.span.error("expected a function call"));
                };
                let v = self.call(name, templates.as_deref(), args, e.span, true)?;
                out.push(Stmt::Eval(v));
                Ok(true)
            }
            SK::If {
                cond,
                then,
                otherwise,
            } => {
                let c = self.condition(cond, "'if'")?;
                if let Some(&ConstValue::Bool(taken)) = c.as_const() {
                    if taken {
                        let (t, tf) = self.scoped_block(then)?;
                        if let Some(s) = otherwise {
                            self.check_dead_branch(|me| match &s.kind {
                                SK::Block(b) => me.scoped_block(b).map(|_| ()),
                                _ => {
                                    let mut sink = Vec::new();
                                    me.stmt(s, &mut sink).map(|_| ())
                                }
                            })?;
                        }
                        if !t.is_empty() {
                            out.push(Stmt::Block(t));
                        }
                        return Ok(tf);
                    }
                    self.check_dead_branch(|me| me.scoped_block(then).map(|_| ()))?;
                    let (o, of) = match otherwise {
                        None => (Vec::new(), true),
                        Some(s) => match &s.kind {
                            SK::Block(b) => self.scoped_block(b)?,
                            _ => {
                                let mut o = Vec::new();
                                let f = self.stmt(s, &mut o)?;
                                (o, f)
                            }
                        },
                    };
                    if !o.is_empty() {
                        out.push(Stmt::Block(o));
                    }
                    return Ok(of);
                }
                let (t, tf) = self.scoped_block(then)?;
                let (o, of) = match otherwise {
                    None => (Vec::new(), true),
                    Some(s) => match &s.kind {
                        SK::Block(b) => self.scoped_block(b)?,
                        _ => {
                            let mut o = Vec::new();
                            let f = self.stmt(s, &mut o)?;
                            (o, f)
                        }
                    },
                };
                out.push(Stmt::If {
                    cond: c,
                    then: t,
                    otherwise: o,
                });
                Ok(tf || of)
            }
            SK::Loop { body, continuing } => {
                self.loop_stmt(None, body, LoopTail::Continuing(continuing.as_ref()), out)
            }
            SK::While { cond, body } => {
                let c = self.condition(cond, "'while'")?;
                self.loop_stmt(Some(c), body, LoopTail::None, out)
            }
            SK::For {
                init,
                cond,
                update,
                body,
            } => {
                self.push_scope();
                let mut inner = Vec::new();
                let r = (|| {
                    if let Some(i) = init {
                        self.stmt(i, &mut inner)?;
                    }
                    let c = cond
                        .as_ref()
                        .map(|c| self.condition(c, "'for'"))
                        .transpose()?;
                    self.loop_stmt(c, body, LoopTail::Update(update.as_deref()), &mut inner)
                })();
                self.pop_scope();
                let falls = r?;
                out.push(Stmt::Block(inner));
                Ok(falls)
            }
            SK::Break => {
                let fcx = self.fcx_mut();
                match fcx.loops.last_mut() {
                    None => return Err(span.error("'break' outside of a loop")),
                    Some(l) if l.in_continuing => {
                        return Err(span.error(
                            "'break' is not allowed in a continuing block; use 'break if'",
                        ));
                    }
                    Some(l) => l.has_break = true,
                }
                out.push(Stmt::Break);
                Ok(false)
            }
            SK::Continue => {
                let fcx = self.fcx_mut();
                match fcx.loops.last_mut() {
                    None => return Err(span.error("'continue' outside of a loop")),
                    Some(l) if l.in_continuing => {
                        return Err(span.error("'continue' is not allowed in a continuing block"));
                    }
                    Some(l) => l.continue_seen = true,
                }
                out.push(Stmt::Continue);
                Ok(false)
            }
            SK::Discard => {
                self.fcx_mut()
                    .stage_reqs
                    .push((StageReq::Fragment, span, "discard".into()));
                out.push(Stmt::Discard);
                Ok(false)
            }
            SK::Return(v) => {
                if !self.fcx_mut().continuing_of.is_empty() {
                    return Err(span.error("'return' is not allowed in a continuing block"));
                }
                let ret = self.fcx_mut().ret.clone();
                match (v, ret) {
                    (None, None) => out.push(Stmt::Return(None)),
                    (Some(e), Some(t)) => {
                        let x = self.value(e)?;
                        let x = self.convert(x, &t, e.span)?;
                        out.push(Stmt::Return(Some(x)));
                    }
                    (None, Some(t)) => {
                        return Err(span
                            .error(format!("missing return value of type {}", self.ty_name(&t))));
                    }
                    (Some(e), None) => {
                        return Err(e
                            .span
                            .error("this function has no return type and cannot return a value"));
                    }
                }
                Ok(false)
            }
            SK::Block(b) => {
                let (st, f) = self.scoped_block(b)?;
                out.push(Stmt::Block(st));
                Ok(f)
            }
            SK::ConstAssert(e) => {
                if self.dead_depth == 0 {
                    self.const_assert(e)?;
                } else {
                    let v = self.value(e)?;
                    if !matches!(v.as_const(), Some(ConstValue::Bool(_))) {
                        return Err(e
                            .span
                            .error("const_assert requires a constant bool expression"));
                    }
                }
                Ok(true)
            }
        }
    }

    /// Type-checks an unreachable constant `if`/`else` branch so type and syntax errors are
    /// still reported, then rolls back side effects on function calls, global bindings, locals,
    /// and stage requirements so dead code is completely pruned from IR and reflection.
    fn check_dead_branch(&mut self, f: impl FnOnce(&mut Self) -> R<()>) -> R<()> {
        let const_data_len = self.const_data.len();
        let globals_len = self.globals.len();
        let snap = {
            let fcx = self.fcx_mut();
            (
                fcx.locals.len(),
                fcx.value_names.len(),
                fcx.calls.clone(),
                fcx.globals.clone(),
                fcx.stage_reqs.len(),
                fcx.loops.clone(),
            )
        };
        self.dead_depth += 1;
        let res = f(self);
        self.dead_depth -= 1;
        self.const_data.truncate(const_data_len);
        self.globals.truncate(globals_len);
        let fcx = self.fcx_mut();
        fcx.locals.truncate(snap.0);
        fcx.value_names.truncate(snap.1);
        fcx.calls = snap.2;
        fcx.globals = snap.3;
        fcx.stage_reqs.truncate(snap.4);
        fcx.loops = snap.5;
        res
    }

    fn loop_stmt(
        &mut self,
        cond: Option<Expr>,
        body: &ast::Block,
        tail: LoopTail,
        out: &mut Vec<Stmt>,
    ) -> R<bool> {
        let has_cond = cond.is_some();
        self.push_scope();
        let body_scope = self.fcx_mut().scopes.len() - 1;
        self.fcx_mut().loops.push(LoopCx {
            body_scope,
            continue_seen: false,
            has_break: false,
            in_continuing: false,
        });
        let r = (|| -> R<(Vec<Stmt>, Vec<Stmt>, Option<Expr>)> {
            let mut b = Vec::new();
            self.check_stmts(&body.stmts, &mut b)?;
            let fcx = self.fcx_mut();
            let li = fcx.loops.len() - 1;
            fcx.loops[li].in_continuing = true;
            fcx.continuing_of.push(li);
            let mut cont = Vec::new();
            let mut break_if = None;
            let r = (|| -> R<()> {
                match tail {
                    LoopTail::None => {}
                    LoopTail::Update(u) => {
                        if let Some(u) = u {
                            // The update cannot see declarations of the body.
                            let saved = self.fcx_mut().scopes.pop();
                            let r = self.stmt(u, &mut cont);
                            self.fcx_mut().scopes.extend(saved);
                            r?;
                        }
                    }
                    LoopTail::Continuing(c) => {
                        if let Some(c) = c {
                            self.push_scope();
                            let r = (|| -> R<()> {
                                self.check_stmts(&c.body.stmts, &mut cont)?;
                                if let Some(bi) = &c.break_if {
                                    break_if = Some(self.condition(bi, "'break if'")?);
                                }
                                Ok(())
                            })();
                            self.pop_scope();
                            r?;
                        }
                    }
                }
                Ok(())
            })();
            self.fcx_mut().continuing_of.pop();
            r?;
            Ok((b, cont, break_if))
        })();
        let l = self.fcx_mut().loops.pop().expect("loop");
        self.pop_scope();
        let (b, cont, break_if) = r?;
        let falls = has_cond || l.has_break || break_if.is_some();
        out.push(Stmt::Loop {
            cond,
            body: b,
            continuing: cont,
            break_if,
        });
        Ok(falls)
    }

    fn assign_target(&mut self, lhs: &ast::Expr) -> R<Target> {
        let (place, swizzle) = match self.expr(lhs)? {
            Checked::Ref(p) => (p, None),
            Checked::Swizzle(p, comps) => {
                let mut seen = [false; 4];
                for &c in &comps {
                    if std::mem::replace(&mut seen[c as usize], true) {
                        return Err(lhs
                            .span
                            .error("a swizzle assignment target must not repeat components"));
                    }
                }
                (p, Some(comps))
            }
            Checked::Value(v) => {
                let mut root = lhs;
                while let AK::Member(b, _) | AK::Index(b, _) = &root.kind {
                    root = b;
                }
                let msg = match (&root.kind, &v.kind) {
                    (AK::Ident { name, .. }, _) => match self.lookup_local(name, root.span)? {
                        Some(Binding::Value { .. }) => format!(
                            "cannot assign to '{name}': 'let' declarations and parameters are immutable (use 'var')"
                        ),
                        Some(Binding::Const { .. }) => {
                            format!("cannot assign to constant '{name}'")
                        }
                        _ if matches!(v.kind, EK::Const(_)) => {
                            format!("cannot assign to constant '{name}'")
                        }
                        _ => "cannot assign to this expression".to_string(),
                    },
                    (_, EK::Const(_)) => "cannot assign to a constant".to_string(),
                    _ => "cannot assign to this expression".to_string(),
                };
                return Err(lhs.span.error(msg));
            }
            Checked::Handle(..) => {
                return Err(lhs.span.error("cannot assign to a texture or sampler"));
            }
        };
        if let PlaceRoot::Global(g) = place.root {
            let g = &self.globals[g];
            let what = match g.kind {
                GlobalKind::Uniform => Some("uniform buffer"),
                GlobalKind::Storage { read_only: true } => {
                    Some("read-only storage buffer (declare it var<storage, read_write>)")
                }
                GlobalKind::PushConstant => Some("push constant block"),
                GlobalKind::ConstData => Some("constant"),
                _ => None,
            };
            if let Some(what) = what {
                return Err(lhs
                    .span
                    .error(format!("cannot assign to '{}': it is a {what}", g.name)));
            }
        }
        if !is_constructible(&place.ty, &self.structs) {
            return Err(lhs.span.error(format!(
                "cannot assign a value of type {}",
                self.ty_name(&place.ty)
            )));
        }
        let ty = match &swizzle {
            Some(c) => Ty::Vector(c.len() as u8, place.ty.scalar().expect("vector")),
            None => place.ty.clone(),
        };
        Ok(Target { place, swizzle, ty })
    }
}

enum LoopTail<'b> {
    None,
    Update(Option<&'b ast::Stmt>),
    Continuing(Option<&'b ast::Continuing>),
}
