use super::*;

/// What a place's last steps are, for choosing how to store to it.
struct PlaceShape {
    /// The place is a component of a packed vec3.
    split: bool,
    /// The place is a packed vec3 member.
    packed: bool,
    /// The last step indexes a vector.
    vector_tail: bool,
    /// Type of the place before its last step.
    parent: Ty,
}

/// A place as MSL text.
struct PlaceText {
    /// The lvalue (valid unless `split` is set).
    text: String,
    space: Space,
    /// The place is a component of a packed `vec3`: (the packed lvalue, its unpacked type,
    /// the component access such as `.x` or `[i]`).
    split: Option<(String, String, String)>,
    /// The place itself is a packed `vec3` member: its unpacked type.
    packed: Option<String>,
}

/// Is `s` one parenthesized expression, `( ... )`?
fn wrapped(s: &str) -> bool {
    if !s.starts_with('(') || !s.ends_with(')') {
        return false;
    }
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 && i + 1 < s.len() {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
}

/// `s` as an operand of a postfix operator (`.x`, `[i]`).
pub(super) fn postfix(s: &str) -> String {
    if wrapped(s)
        || s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == ':')
    {
        s.to_string()
    } else {
        format!("({s})")
    }
}

/// `s` without one pair of enclosing parentheses, for `if (...)`.
pub(super) fn bare(s: &str) -> &str {
    if wrapped(s) { &s[1..s.len() - 1] } else { s }
}

pub(super) fn is_trivial(e: &Expr) -> bool {
    matches!(e.kind, EK::Const(_) | EK::Zero | EK::Value(_))
}

impl<'p> Writer<'p> {
    // ------------------------------------------------------------------ places

    pub(super) fn temp(&mut self, ty: &Ty, text: String) -> R<String> {
        let t = self.ty_name(ty)?;
        let n = self.fs.names.generated("ganbate_t");
        self.fs.pre.push(format!("const {t} {n} = {text};"));
        Ok(n)
    }

    fn place(&mut self, p: &Place) -> R<PlaceText> {
        let prog = self.prog;
        let (mut text, mut ty, space, mut pointer) = match p.root {
            PlaceRoot::Local(l) => {
                let local = prog.functions[self.cur_fn]
                    .locals
                    .get(l as usize)
                    .ok_or_else(|| internal("unknown local"))?;
                let n = self
                    .fs
                    .locals
                    .get(l as usize)
                    .cloned()
                    .ok_or_else(|| internal("local not declared"))?;
                (n, local.ty.clone(), Space::Thread, false)
            }
            PlaceRoot::Global(g) => {
                let gl = &prog.globals[g];
                let n = self
                    .global_names
                    .get(&g)
                    .cloned()
                    .ok_or_else(|| internal("global not named"))?;
                let space = match gl.kind {
                    GlobalKind::Private => Space::Thread,
                    GlobalKind::Storage { .. } => Space::Device,
                    _ => Space::Constant,
                };
                let pointer = matches!(gl.ty, Ty::Array(_, None));
                (n, gl.ty.clone(), space, pointer)
            }
        };
        let indices: Vec<&Expr> = p
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Index(i) => Some(i),
                _ => None,
            })
            .collect();
        let hoist = indices.len() > 1 && self.needs_order(&indices);
        let mut packed: Option<String> = None;
        let mut split = None;
        for s in &p.steps {
            match s {
                Step::Member(i) => {
                    let Ty::Struct(sid) = ty else {
                        return Err(internal("member of a non-struct"));
                    };
                    let plan = self.struct_plan(sid)?.clone();
                    let i = *i as usize;
                    let _ = write!(text, ".{}", plan.member_names[i]);
                    let mty = prog.structs[sid].members[i].ty.clone();
                    packed = if plan.packed[i] {
                        Some(self.ty_name(&mty)?)
                    } else {
                        None
                    };
                    pointer = matches!(mty, Ty::Array(_, None));
                    ty = mty;
                }
                Step::Index(e) => {
                    let mut idx = self.expr(e)?;
                    if hoist && !is_trivial(e) {
                        idx = self.temp(&e.ty, idx)?;
                    }
                    let constant = e.as_const().and_then(|c| c.as_i64());
                    match ty.clone() {
                        Ty::Vector(_, s) => {
                            let access = match constant {
                                Some(c) if (0..4).contains(&c) => {
                                    format!(".{}", ['x', 'y', 'z', 'w'][c as usize])
                                }
                                _ => format!("[{idx}]"),
                            };
                            if let Some(un) = packed.take() {
                                split = Some((text.clone(), un, access));
                            } else {
                                text.push_str(&access);
                            }
                            ty = Ty::Scalar(s);
                        }
                        Ty::Matrix(_, r) => {
                            let _ = write!(text, "[{idx}]");
                            ty = Ty::Vector(r, Scalar::F32);
                        }
                        Ty::Array(e, n) => {
                            if pointer || n.is_none() {
                                let _ = write!(text, "[{idx}]");
                            } else {
                                let _ = write!(text, ".inner[{idx}]");
                            }
                            pointer = false;
                            ty = *e;
                        }
                        _ => return Err(internal("index into a non-indexable place")),
                    }
                }
            }
        }
        Ok(PlaceText {
            text,
            space,
            split,
            packed,
        })
    }

    /// Where a place lands, without emitting anything.
    fn place_shape(&mut self, p: &Place) -> R<PlaceShape> {
        let prog = self.prog;
        let mut ty = match p.root {
            PlaceRoot::Local(l) => prog.functions[self.cur_fn]
                .locals
                .get(l as usize)
                .ok_or_else(|| internal("unknown local"))?
                .ty
                .clone(),
            PlaceRoot::Global(g) => prog.globals[g].ty.clone(),
        };
        let mut shape = PlaceShape {
            split: false,
            packed: false,
            vector_tail: false,
            parent: ty.clone(),
        };
        for s in &p.steps {
            shape.parent = ty.clone();
            match s {
                Step::Member(i) => {
                    let Ty::Struct(sid) = ty else {
                        return Err(internal("member of a non-struct"));
                    };
                    shape.packed = self.struct_plan(sid)?.packed[*i as usize];
                    ty = prog.structs[sid].members[*i as usize].ty.clone();
                }
                Step::Index(_) => {
                    ty = match ty {
                        Ty::Vector(_, sc) => {
                            shape.split |= shape.packed;
                            shape.vector_tail = true;
                            Ty::Scalar(sc)
                        }
                        Ty::Matrix(_, r) => Ty::Vector(r, Scalar::F32),
                        Ty::Array(e, _) => *e,
                        _ => return Err(internal("index into a non-indexable place")),
                    };
                    shape.packed = false;
                }
            }
        }
        Ok(shape)
    }

    pub(super) fn load(&mut self, p: &Place) -> R<String> {
        let pt = self.place(p)?;
        Ok(match (pt.split, pt.packed) {
            (Some((base, un, access)), _) => format!("{un}({base}){access}"),
            (None, Some(un)) => format!("{un}({})", pt.text),
            (None, None) => pt.text,
        })
    }

    // ------------------------------------------------------------------ assignment

    pub(super) fn assign(&mut self, place: &Place, swizzle: Option<&[u32]>, value: &Expr) -> R<()> {
        let mut indices: Vec<&Expr> = place
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Index(i) => Some(i),
                _ => None,
            })
            .collect();
        let root_writable = match place.root {
            PlaceRoot::Global(g) => self.writable_global(g),
            PlaceRoot::Local(_) => false,
        };
        let uses_current = {
            let mut c = false;
            walk_expr(value, &mut |x| c |= matches!(x.kind, EK::Current));
            c
        };
        let effectful_index = indices
            .iter()
            .any(|i| self.effects(i) || self.reads(i, false));
        indices.push(value);
        // Bind a reference when the target must be evaluated once, before the value.
        let mut by_ref =
            effectful_index && (self.effects(value) || uses_current || self.needs_order(&indices));
        let shape = self.place_shape(place)?;
        let packed_rmw = shape.split || (shape.packed && swizzle.is_some());
        by_ref |= packed_rmw;
        if by_ref && !packed_rmw && shape.vector_tail {
            // Vector components are not addressable: reference the vector, then index it.
            let n = place.steps.len();
            let Some(Step::Index(last)) = place.steps.last() else {
                return Err(internal("vector tail without an index"));
            };
            let parent = Place {
                root: place.root,
                steps: place.steps[..n - 1].to_vec(),
                ty: shape.parent.clone(),
            };
            let ppt = self.place(&parent)?;
            let mut idx = self.expr(last)?;
            if !is_trivial(last) {
                idx = self.temp(&last.ty, idx)?;
            }
            let access = match last.as_const().and_then(|c| c.as_i64()) {
                Some(c) if (0..4).contains(&c) => format!(".{}", ['x', 'y', 'z', 'w'][c as usize]),
                _ => format!("[{idx}]"),
            };
            let space = ppt.space.qualifier();
            let ty = self.ty_name(&shape.parent)?;
            let r = self.fs.names.generated("ganbate_ref");
            self.fs
                .pre
                .push(format!("{space} {ty}& {r} = {};", ppt.text));
            let target = format!("{r}{access}");
            let val = self.with_current(target.clone(), root_writable, value)?;
            self.flush_pre();
            self.line(&format!("{target} = {val};"));
            return Ok(());
        }
        let pt = self.place(place)?;
        let current_is_global = root_writable;
        if !by_ref {
            let target = pt.text.clone();
            let current = match swizzle {
                Some(c) => format!("{}.{}", postfix(&target), swizzle_letters(c)),
                None => match &pt.packed {
                    Some(un) => format!("{un}({target})"),
                    None => target.clone(),
                },
            };
            let v = self.with_current(current, current_is_global, value)?;
            self.flush_pre();
            let lhs = match swizzle {
                Some(c) => format!("{}.{}", postfix(&target), swizzle_letters(c)),
                None => target,
            };
            self.line(&format!("{lhs} = {v};"));
            return Ok(());
        }
        // Reference form.
        let space = pt.space.qualifier();
        let r = self.fs.names.generated("ganbate_ref");
        if let Some((base, un, access)) = pt.split {
            // A component of a packed vec3: read, modify and write the whole vector.
            let packed_ty = un.replacen("metal::", "metal::packed_", 1);
            let v = self.fs.names.generated("ganbate_vec");
            self.fs
                .pre
                .push(format!("{space} {packed_ty}& {r} = {base};"));
            self.fs.pre.push(format!("{un} {v} = {un}({r});"));
            let val = self.with_current(format!("{v}{access}"), current_is_global, value)?;
            self.flush_pre();
            self.line(&format!("{v}{access} = {val};"));
            self.line(&format!("{r} = {v};"));
            return Ok(());
        }
        if let Some(un) = &pt.packed {
            let packed_ty = un.replacen("metal::", "metal::packed_", 1);
            let v = self.fs.names.generated("ganbate_vec");
            self.fs
                .pre
                .push(format!("{space} {packed_ty}& {r} = {};", pt.text));
            self.fs.pre.push(format!("{un} {v} = {un}({r});"));
            let sw = swizzle.map(swizzle_letters);
            let cur = match &sw {
                Some(s) => format!("{v}.{s}"),
                None => v.clone(),
            };
            let val = self.with_current(cur.clone(), current_is_global, value)?;
            self.flush_pre();
            self.line(&format!("{cur} = {val};"));
            self.line(&format!("{r} = {v};"));
            return Ok(());
        }
        let ty = self.ty_name(&place.ty)?;
        self.fs
            .pre
            .push(format!("{space} {ty}& {r} = {};", pt.text));
        let target = match swizzle {
            Some(c) => format!("{r}.{}", swizzle_letters(c)),
            None => r.clone(),
        };
        let val = self.with_current(target.clone(), current_is_global, value)?;
        self.flush_pre();
        self.line(&format!("{target} = {val};"));
        Ok(())
    }

    pub(super) fn with_current(
        &mut self,
        current: String,
        global: bool,
        value: &Expr,
    ) -> R<String> {
        let saved = self.fs.current.replace(current);
        let saved_global = std::mem::replace(&mut self.current_global, global);
        let v = self.expr(value);
        self.fs.current = saved;
        self.current_global = saved_global;
        v
    }

    // ------------------------------------------------------------------ expressions

    pub(super) fn ordered(&mut self, es: &[&Expr]) -> R<Vec<String>> {
        let hoist = es.len() > 1 && self.needs_order(es);
        let mut out = Vec::with_capacity(es.len());
        for e in es {
            let s = self.expr(e)?;
            out.push(if hoist && !is_trivial(e) {
                self.temp(&e.ty, s)?
            } else {
                s
            });
        }
        Ok(out)
    }

    pub(super) fn expr(&mut self, e: &Expr) -> R<String> {
        let prog = self.prog;
        Ok(match &e.kind {
            EK::Const(c) => self.const_value(&e.ty, c)?,
            EK::Zero => self.zero(&e.ty)?,
            EK::Value(v) => self
                .fs
                .values
                .get(v)
                .cloned()
                .ok_or_else(|| internal("value used before its definition"))?,
            EK::Load(p) => self.load(p)?,
            EK::Extract(b, i) => {
                let bs = self.expr(b)?;
                self.extract(&b.ty, &bs, *i)?
            }
            EK::DynIndex(b, i) => {
                let v = self.ordered(&[b, i])?;
                match b.ty {
                    Ty::Array(..) => format!("{}.inner[{}]", postfix(&v[0]), v[1]),
                    _ => format!("{}[{}]", postfix(&v[0]), v[1]),
                }
            }
            EK::Swizzle(b, comps) => {
                let bs = self.expr(b)?;
                format!("{}.{}", postfix(&bs), swizzle_letters(comps))
            }
            EK::Unary(op, x) => {
                let v = self.expr(x)?;
                match op {
                    UnaryOp::Neg if e.ty.scalar() == Some(Scalar::I32) => {
                        let u = unsigned_of(&e.ty);
                        let t = self.ty_name(&e.ty)?;
                        format!("metal::as_type<{t}>(-metal::as_type<{u}>({v}))")
                    }
                    UnaryOp::Neg => format!("(-{v})"),
                    UnaryOp::Not => format!("(!{v})"),
                    UnaryOp::BitNot => format!("(~{v})"),
                }
            }
            EK::Binary(op, l, r) => self.binary(*op, l, r, &e.ty)?,
            EK::Construct(parts) => {
                let refs: Vec<&Expr> = parts.iter().collect();
                let v = self.ordered(&refs)?;
                match &e.ty {
                    Ty::Vector(..) | Ty::Matrix(..) => {
                        format!("{}({})", self.ty_name(&e.ty)?, v.join(", "))
                    }
                    Ty::Array(..) => format!("{}{{{{{}}}}}", self.ty_name(&e.ty)?, v.join(", ")),
                    Ty::Struct(sid) => self.construct_struct(*sid, v)?,
                    _ => return Err(internal("construct of a non-composite")),
                }
            }
            EK::Splat(x) => {
                let v = self.expr(x)?;
                format!("{}({v})", self.ty_name(&e.ty)?)
            }
            EK::Convert(x) => self.convert(x, &e.ty)?,
            EK::Bitcast(x) => {
                let v = self.expr(x)?;
                format!("metal::as_type<{}>({v})", self.ty_name(&e.ty)?)
            }
            EK::Call(f, args) => {
                let refs: Vec<&Expr> = args.iter().collect();
                let mut v = self.ordered(&refs)?;
                let facts = self
                    .facts
                    .get(f)
                    .cloned()
                    .ok_or_else(|| internal("callee not analysed"))?;
                for g in &facts.globals {
                    v.push(self.global_names[g].clone());
                }
                if facts.sizes {
                    v.push(self.sizes_name.clone());
                }
                format!("{}({})", self.fn_names[f], v.join(", "))
            }
            EK::Builtin(b, args) => self.builtin(*b, args, &e.ty)?,
            EK::Texture(tc) => self.texture(tc)?,
            EK::ArrayLength { global, member } => {
                let k = self
                    .sized
                    .iter()
                    .position(|g| g == global)
                    .ok_or_else(|| internal("arrayLength of an unsized buffer"))?;
                let gl = &prog.globals[*global];
                let (offset, elem) = match (member, &gl.ty) {
                    (Some(m), Ty::Struct(sid)) => {
                        let info = &prog.structs[*sid];
                        let off = info.layout.as_ref().map_or(0, |l| l.offsets[*m as usize]);
                        let Ty::Array(e, None) = &info.members[*m as usize].ty else {
                            return Err(internal("arrayLength of a sized member"));
                        };
                        (off, (**e).clone())
                    }
                    (None, Ty::Array(e, None)) => (0, (**e).clone()),
                    _ => return Err(internal("arrayLength operand")),
                };
                let stride = array_stride(&elem, &prog.structs);
                let s = &self.sizes_name;
                if offset == 0 {
                    format!("({s}[{k}] / {stride}u)")
                } else {
                    format!("((metal::max({s}[{k}], {offset}u) - {offset}u) / {stride}u)")
                }
            }
            EK::Current => self
                .fs
                .current
                .clone()
                .ok_or_else(|| internal("compound assignment value missing"))?,
        })
    }

    /// Member `i` of a struct value, component `i` of a vector, column of a matrix, element of
    /// an array.
    pub(super) fn extract(&mut self, ty: &Ty, base: &str, i: u32) -> R<String> {
        let prog = self.prog;
        let b = postfix(base);
        Ok(match ty {
            Ty::Vector(..) => format!("{b}.{}", swizzle_letters(&[i])),
            Ty::Matrix(..) => format!("{b}[{i}]"),
            Ty::Array(..) => format!("{b}.inner[{i}]"),
            Ty::Struct(sid) => {
                let plan = self.struct_plan(*sid)?.clone();
                let m = format!("{b}.{}", plan.member_names[i as usize]);
                if plan.packed[i as usize] {
                    let mt = prog.structs[*sid].members[i as usize].ty.clone();
                    format!("{}({m})", self.ty_name(&mt)?)
                } else {
                    m
                }
            }
            _ => return Err(internal("extract from a non-composite")),
        })
    }

    pub(super) fn binary(&mut self, op: BinaryOp, l: &Expr, r: &Expr, ty: &Ty) -> R<String> {
        use BinaryOp::*;
        if matches!(op, LogicalAnd | LogicalOr) {
            let a = self.expr(l)?;
            if self.needs_hoist(r) {
                // Keep the short circuit: evaluate the right operand (and whatever it hoists)
                // only when needed.
                let t = self.fs.names.generated("ganbate_cond");
                self.fs.pre.push(format!("bool {t} = {a};"));
                let saved = std::mem::take(&mut self.fs.pre);
                let b = self.expr(r)?;
                let inner = std::mem::replace(&mut self.fs.pre, saved);
                let neg = if op == LogicalAnd { "" } else { "!" };
                self.fs.pre.push(format!("if ({neg}{t}) {{"));
                for line in inner {
                    self.fs.pre.push(format!("    {line}"));
                }
                self.fs.pre.push(format!("    {t} = {b};"));
                self.fs.pre.push("}".into());
                return Ok(t);
            }
            let b = self.expr(r)?;
            let o = if op == LogicalAnd { "&&" } else { "||" };
            return Ok(format!("({a} {o} {b})"));
        }
        let v = self.ordered(&[l, r])?;
        let (a, b) = (&v[0], &v[1]);
        let ls = l.ty.scalar().unwrap_or(Scalar::F32).concretize();
        let matrixy = matches!(l.ty, Ty::Matrix(..)) || matches!(r.ty, Ty::Matrix(..));
        Ok(match op {
            Add | Sub | Mul if !matrixy && ls == Scalar::I32 => {
                let t = self.ty_name(ty)?;
                let (ua, ub) = (unsigned_of(&l.ty), unsigned_of(&r.ty));
                format!(
                    "metal::as_type<{t}>(metal::as_type<{ua}>({a}) {} metal::as_type<{ub}>({b}))",
                    op.symbol()
                )
            }
            Rem if ls == Scalar::F32 => {
                let a = self.splat_to(a, &l.ty, ty)?;
                let b = self.splat_to(b, &r.ty, ty)?;
                format!("metal::fmod({a}, {b})")
            }
            Add | Sub | Mul | Div | Rem | Eq | Ne | Lt | Le | Gt | Ge => {
                format!("({a} {} {b})", op.symbol())
            }
            And | Or if ls == Scalar::Bool => {
                if ty.is_scalar() {
                    format!("bool({a} {} {b})", op.symbol())
                } else {
                    format!("({a} {} {b})", op.symbol())
                }
            }
            Xor if ls == Scalar::Bool => format!("({a} != {b})"),
            And | Or | Xor => format!("({a} {} {b})", op.symbol()),
            Shl | Shr => {
                let amount = if r.as_const().is_some() {
                    b.clone()
                } else {
                    format!("({b} & 31u)")
                };
                match (op, ls) {
                    (Shl, Scalar::I32) => {
                        let t = self.ty_name(ty)?;
                        let u = unsigned_of(&l.ty);
                        format!("metal::as_type<{t}>(metal::as_type<{u}>({a}) << {amount})")
                    }
                    (Shr, Scalar::I32) => {
                        let st = self.ty_name(&r.ty.with_scalar(Scalar::I32))?;
                        format!("({a} >> {st}({amount}))")
                    }
                    (Shl, _) => format!("({a} << {amount})"),
                    _ => format!("({a} >> {amount})"),
                }
            }
            LogicalAnd | LogicalOr => unreachable!(),
        })
    }

    /// `text` (of type `from`) broadcast to the vector type `to` when it is a scalar.
    pub(super) fn splat_to(&mut self, text: &str, from: &Ty, to: &Ty) -> R<String> {
        if from.is_scalar() && !to.is_scalar() {
            Ok(format!("{}({text})", self.ty_name(to)?))
        } else {
            Ok(text.to_string())
        }
    }

    pub(super) fn convert(&mut self, x: &Expr, to: &Ty) -> R<String> {
        let v = self.expr(x)?;
        let from = x.ty.scalar().unwrap_or(Scalar::F32).concretize();
        let ts = to.scalar().unwrap_or(Scalar::F32);
        let tn = self.ty_name(to)?;
        Ok(match (from, ts) {
            (a, b) if a == b => v,
            (Scalar::Bool, s) => {
                let (one, zero) = match s {
                    Scalar::F32 => ("1.0f", "0.0f"),
                    Scalar::I32 => ("1", "0"),
                    _ => ("1u", "0u"),
                };
                if to.is_scalar() {
                    format!("({v} ? {one} : {zero})")
                } else {
                    format!("metal::select({tn}({zero}), {tn}({one}), {v})")
                }
            }
            (s, Scalar::Bool) => {
                let zero = match s {
                    Scalar::F32 => "0.0f",
                    Scalar::I32 => "0",
                    _ => "0u",
                };
                if to.is_scalar() {
                    format!("({v} != {zero})")
                } else {
                    let src = self.ty_name(&x.ty.concretize())?;
                    format!("({v} != {src}({zero}))")
                }
            }
            _ => format!("{tn}({v})"),
        })
    }

    pub(super) fn builtin(&mut self, b: B, args: &[Expr], ty: &Ty) -> R<String> {
        let refs: Vec<&Expr> = args.iter().collect();
        let v = self.ordered(&refs)?;
        let s = args
            .first()
            .and_then(|a| a.ty.scalar())
            .unwrap_or(Scalar::F32)
            .concretize();
        let call = |name: &str, v: &[String]| format!("metal::{name}({})", v.join(", "));
        let scalar_arg = args.first().is_some_and(|a| a.ty.is_scalar());
        Ok(match b {
            B::Abs if s == Scalar::U32 => v[0].clone(),
            B::Abs => call("abs", &v),
            B::Sign if s == Scalar::I32 => {
                let h = self.helper("ganbate_sign");
                format!("{h}({})", v[0])
            }
            B::Sign => call("sign", &v),
            B::Min => call("min", &v),
            B::Max => call("max", &v),
            B::Clamp => call("clamp", &v),
            B::Acos => call("acos", &v),
            B::Acosh => call("acosh", &v),
            B::Asin => call("asin", &v),
            B::Asinh => call("asinh", &v),
            B::Atan => call("atan", &v),
            B::Atanh => call("atanh", &v),
            B::Atan2 => call("atan2", &v),
            B::Ceil => call("ceil", &v),
            B::Cos => call("cos", &v),
            B::Cosh => call("cosh", &v),
            B::Degrees => format!("({} * 57.2957795131f)", v[0]),
            B::Radians => format!("({} * 0.0174532925199f)", v[0]),
            B::Exp => call("exp", &v),
            B::Exp2 => call("exp2", &v),
            B::Floor => call("floor", &v),
            B::Fract => call("fract", &v),
            B::InverseSqrt => call("rsqrt", &v),
            B::Log => call("log", &v),
            B::Log2 => call("log2", &v),
            B::Round => call("rint", &v),
            B::Sin => call("sin", &v),
            B::Sinh => call("sinh", &v),
            B::Sqrt => call("sqrt", &v),
            B::Tan => call("tan", &v),
            B::Tanh => call("tanh", &v),
            B::Trunc => call("trunc", &v),
            B::Pow => call("pow", &v),
            B::Step => call("step", &v),
            B::Smoothstep => call("smoothstep", &v),
            B::Fma => call("fma", &v),
            B::Saturate => call("saturate", &v),
            B::Mix => {
                let t = self.splat_to(&v[2], &args[2].ty, ty)?;
                call("mix", &[v[0].clone(), v[1].clone(), t])
            }
            B::Dot if s.is_float() => call("dot", &v),
            B::Dot => {
                let h = self.helper("ganbate_dot");
                format!("{h}({}, {})", v[0], v[1])
            }
            B::Cross => call("cross", &v),
            B::Length if scalar_arg => call("abs", &v),
            B::Length => call("length", &v),
            B::Distance if scalar_arg => format!("metal::abs({} - {})", v[0], v[1]),
            B::Distance => call("distance", &v),
            B::Normalize => call("normalize", &v),
            B::FaceForward => call("faceforward", &v),
            B::Reflect => call("reflect", &v),
            B::Refract => call("refract", &v),
            B::Determinant => call("determinant", &v),
            B::Transpose => call("transpose", &v),
            B::Ldexp => call("ldexp", &v),
            B::Select => {
                let c = match (ty, args[2].ty.is_scalar()) {
                    (Ty::Vector(n, _), true) => {
                        format!("{}({})", vector_name(*n, Scalar::Bool), v[2])
                    }
                    _ => v[2].clone(),
                };
                call("select", &[v[0].clone(), v[1].clone(), c])
            }
            B::All | B::Any if scalar_arg => v[0].clone(),
            B::All => call("all", &v),
            B::Any => call("any", &v),
            B::Dpdx => call("dfdx", &v),
            B::Dpdy => call("dfdy", &v),
            B::Fwidth => call("fwidth", &v),
            B::CountOneBits => call("popcount", &v),
            B::ReverseBits => call("reverse_bits", &v),
            B::Pack4x8Snorm => call("pack_float_to_snorm4x8", &v),
            B::Pack4x8Unorm => call("pack_float_to_unorm4x8", &v),
            B::Pack2x16Snorm => call("pack_float_to_snorm2x16", &v),
            B::Pack2x16Unorm => call("pack_float_to_unorm2x16", &v),
            B::Pack2x16Float => format!("metal::as_type<uint>(metal::half2({}))", v[0]),
            B::Unpack4x8Snorm => call("unpack_snorm4x8_to_float", &v),
            B::Unpack4x8Unorm => call("unpack_unorm4x8_to_float", &v),
            B::Unpack2x16Snorm => call("unpack_snorm2x16_to_float", &v),
            B::Unpack2x16Unorm => call("unpack_unorm2x16_to_float", &v),
            B::Unpack2x16Float => format!("metal::float2(metal::as_type<metal::half2>({}))", v[0]),
            B::WorkgroupBarrier | B::StorageBarrier => {
                return Err(internal("barrier used as a value"));
            }
            _ => return Err(internal("unexpected built-in")),
        })
    }

    pub(super) fn texture(&mut self, tc: &TexCall) -> R<String> {
        let prog = self.prog;
        let ops = tex_operands(tc);
        let hoist = ops.len() > 1 && self.needs_order(&ops);
        let ev = |w: &mut Self, e: &Option<Expr>| -> R<Option<String>> {
            Ok(match e {
                Some(e) => {
                    let s = w.expr(e)?;
                    Some(if hoist && !is_trivial(e) {
                        w.temp(&e.ty, s)?
                    } else {
                        s
                    })
                }
                None => None,
            })
        };
        let coords = ev(self, &tc.coords)?;
        let array_index = ev(self, &tc.array_index)?;
        let level = ev(self, &tc.level)?;
        let bias = ev(self, &tc.bias)?;
        let ddx = ev(self, &tc.ddx)?;
        let ddy = ev(self, &tc.ddy)?;
        let depth_ref = ev(self, &tc.depth_ref)?;
        let offset = ev(self, &tc.offset)?;
        let value = ev(self, &tc.value)?;
        let missing = |what: &str| internal(&format!("texture call without {what}"));
        let tg = &prog.globals[tc.texture];
        let t = self.global_names[&tc.texture].clone();
        let view = match tg.ty {
            Ty::Texture(v, _) => v,
            _ => TextureView::D2,
        };
        let uint = |s: Option<String>, e: &Option<Expr>| -> Option<String> {
            match (s, e) {
                (Some(s), Some(e)) if e.ty.scalar() != Some(Scalar::U32) => {
                    Some(format!("uint({s})"))
                }
                (s, _) => s,
            }
        };
        let array_index = uint(array_index, &tc.array_index);
        match tc.op {
            B::TextureSample
            | B::TextureSampleBias
            | B::TextureSampleLevel
            | B::TextureSampleGrad
            | B::TextureSampleCompare
            | B::TextureSampleCompareLevel => {
                let sg = tc.sampler.ok_or_else(|| missing("a sampler"))?;
                let smp = self.global_names[&sg].clone();
                let mut a = vec![smp, coords.ok_or_else(|| missing("coordinates"))?];
                a.extend(array_index);
                let compare = matches!(
                    tc.op,
                    B::TextureSampleCompare | B::TextureSampleCompareLevel
                );
                if compare {
                    a.push(depth_ref.ok_or_else(|| missing("a depth reference"))?);
                }
                match tc.op {
                    B::TextureSampleBias => a.push(format!(
                        "metal::bias({})",
                        bias.ok_or_else(|| missing("bias"))?
                    )),
                    B::TextureSampleLevel => {
                        let l = level.ok_or_else(|| missing("level"))?;
                        let l = if tc.level.as_ref().is_some_and(|e| e.ty != F32) {
                            format!("float({l})")
                        } else {
                            l
                        };
                        a.push(format!("metal::level({l})"));
                    }
                    B::TextureSampleGrad => {
                        let g = if view == TextureView::Cube {
                            "gradientcube"
                        } else {
                            "gradient2d"
                        };
                        a.push(format!(
                            "metal::{g}({}, {})",
                            ddx.ok_or_else(|| missing("ddx"))?,
                            ddy.ok_or_else(|| missing("ddy"))?
                        ));
                    }
                    B::TextureSampleCompareLevel => a.push("metal::level(0)".into()),
                    _ => {}
                }
                a.extend(offset);
                let f = if compare { "sample_compare" } else { "sample" };
                Ok(format!("{t}.{f}({})", a.join(", ")))
            }
            B::TextureLoad => {
                let c = coords.ok_or_else(|| missing("coordinates"))?;
                let mut a = vec![format!("metal::uint2({c})")];
                a.extend(array_index);
                let l = uint(level, &tc.level).ok_or_else(|| missing("level"))?;
                a.push(l);
                Ok(format!("{t}.read({})", a.join(", ")))
            }
            B::TextureStore => {
                let c = coords.ok_or_else(|| missing("coordinates"))?;
                let v = value.ok_or_else(|| missing("a value"))?;
                Ok(format!("{t}.write({v}, metal::uint2({c}))"))
            }
            B::TextureDimensions => {
                if matches!(tg.ty, Ty::StorageTexture(_)) {
                    let h = self.helper("ganbate_dims_storage");
                    return Ok(format!("{h}({t})"));
                }
                let h = self.helper("ganbate_dims");
                let l = uint(level, &tc.level).unwrap_or_else(|| "0u".into());
                Ok(format!("{h}({t}, {l})"))
            }
            B::TextureNumLevels => Ok(format!("{t}.get_num_mip_levels()")),
            B::TextureNumLayers => Ok(format!("{t}.get_array_size()")),
            _ => Err(internal("unexpected texture built-in")),
        }
    }

    // ------------------------------------------------------------------ entry point

    /// The MSL entry function and its IO structs: (name, source, uses `[[invariant]]`).
    pub(super) fn entry(
        &mut self,
        fid: FnId,
        resources: &[(GlobalId, String)],
        uses_sizes: bool,
    ) -> R<(String, String, bool)> {
        let prog = self.prog;
        let f = &prog.functions[fid];
        let stage = self.stage;
        let ename = self
            .names
            .generated(&format!("ganbate_entry_{}", ascii_ident(&f.name)));
        self.cur_fn = fid;
        self.fs = FnState {
            names: self.names.clone(),
            ..FnState::default()
        };
        let mut io = EntryIo::default();
        let mut args = Vec::new();
        for p in &f.params {
            if !p.io.is_empty() {
                let a = self.entry_input(&p.ty, &p.io, &mut io)?;
                args.push(a);
            } else if let Ty::Struct(sid) = p.ty {
                let mut parts = Vec::new();
                for m in &prog.structs[sid].members {
                    parts.push(self.entry_input(&m.ty, &m.io, &mut io)?);
                }
                args.push(self.construct_struct(sid, parts)?);
            } else {
                return Err(internal("entry parameter without IO attributes"));
            }
        }
        for g in &self.facts[&fid].globals {
            args.push(self.global_names[g].clone());
        }
        if uses_sizes {
            args.push(self.sizes_name.clone());
        }
        let mut src = String::new();
        let in_name = self.names.generated(&format!("{ename}_in"));
        if !io.inputs.is_empty() {
            io.inputs.sort_by_key(|i| i.0);
            let _ = writeln!(src, "struct {in_name} {{");
            for (_, decl) in &io.inputs {
                let _ = writeln!(src, "    {decl};");
            }
            src.push_str("};\n\n");
        }
        // Outputs.
        let mut outputs: Vec<(String, String)> = Vec::new(); // (declaration, source expression)
        let mut invariant = false;
        let result = self.fs.names.generated("ganbate_result");
        if let Some(ret) = &f.ret {
            let mut add = |w: &mut Self, ty: &Ty, attrs: &IoAttrs, from: String| -> R<()> {
                let tn = w.ty_name(ty)?;
                if let Some(bv) = attrs.builtin {
                    match bv {
                        BuiltinValue::Position => {
                            let inv = if attrs.invariant {
                                invariant = true;
                                ", invariant"
                            } else {
                                ""
                            };
                            outputs.push((format!("{tn} position [[position{inv}]]"), from));
                        }
                        BuiltinValue::FragDepth => {
                            outputs.push((format!("{tn} depth [[depth(any)]]"), from))
                        }
                        _ => return Err(internal("unexpected output built-in")),
                    }
                } else if let Some(loc) = attrs.location {
                    let attr = if stage == Stage::Vertex {
                        format!("[[user(locn{loc})]]")
                    } else {
                        format!("[[color({loc})]]")
                    };
                    outputs.push((format!("{tn} loc{loc} {attr}"), from));
                }
                Ok(())
            };
            if !f.ret_io.is_empty() {
                add(self, ret, &f.ret_io, result.clone())?;
            } else if let Ty::Struct(sid) = ret {
                for (i, m) in prog.structs[*sid].members.iter().enumerate() {
                    let from = self.extract(ret, &result, i as u32)?;
                    add(self, &m.ty, &m.io, from)?;
                }
            }
        }
        let out_name = self.names.generated(&format!("{ename}_out"));
        if !outputs.is_empty() {
            let _ = writeln!(src, "struct {out_name} {{");
            for (decl, _) in &outputs {
                let _ = writeln!(src, "    {decl};");
            }
            src.push_str("};\n\n");
        }
        // Parameters.
        let mut params = Vec::new();
        if !io.inputs.is_empty() {
            params.push(format!("{in_name} ganbate_in [[stage_in]]"));
        }
        for (decl, _) in &io.builtins {
            params.push(decl.clone());
        }
        let mut privates = Vec::new();
        for &g in &self.facts[&fid].globals.clone() {
            let gl = &prog.globals[g];
            if gl.kind == GlobalKind::Private {
                let ty = self.ty_name(&gl.ty)?;
                let init = match &gl.init {
                    Some(v) => self.const_value(&gl.ty, v)?,
                    None => self.zero(&gl.ty)?,
                };
                privates.push(format!("{ty} {} = {init};", self.global_names[&g]));
            }
        }
        for (g, attr) in resources {
            let decl = self.global_param(*g)?;
            params.push(format!("{decl} {attr}"));
        }
        if uses_sizes {
            params.push(format!(
                "constant uint* {} [[buffer({SIZES_BUFFER})]]",
                self.sizes_name
            ));
        }
        let qualifier = match stage {
            Stage::Vertex => "vertex",
            Stage::Fragment => "fragment",
            Stage::Compute => "kernel",
        };
        let ret_ty = if outputs.is_empty() {
            "void".to_string()
        } else {
            out_name.clone()
        };
        let _ = writeln!(
            src,
            "{qualifier} {ret_ty} {ename}({}) {{",
            params.join(",\n        ")
        );
        for p in privates {
            let _ = writeln!(src, "    {p}");
        }
        let call = format!("{}({})", self.fn_names[&fid], args.join(", "));
        if outputs.is_empty() {
            let _ = writeln!(src, "    {call};");
        } else {
            let rt = self.ty_name(f.ret.as_ref().expect("outputs need a return value"))?;
            let _ = writeln!(src, "    const {rt} {result} = {call};");
            let o = self.fs.names.generated("ganbate_out");
            let _ = writeln!(src, "    {out_name} {o};");
            for (decl, from) in &outputs {
                let member = decl
                    .split_whitespace()
                    .nth(1)
                    .ok_or_else(|| internal("output declaration"))?;
                let _ = writeln!(src, "    {o}.{member} = {from};");
                if stage == Stage::Vertex && member == "position" {
                    // Vulkan clip space (+Y down) to Metal (+Y up): same framebuffer pixels.
                    let _ = writeln!(src, "    {o}.position.y = -{o}.position.y;");
                }
            }
            let _ = writeln!(src, "    return {o};");
        }
        src.push_str("}\n");
        Ok((ename, src, invariant))
    }

    /// One entry-point input: a `[[stage_in]]` member or a built-in parameter.
    fn entry_input(&mut self, ty: &Ty, attrs: &IoAttrs, io: &mut EntryIo) -> R<String> {
        let tn = self.ty_name(ty)?;
        if let Some(b) = attrs.builtin {
            if let Some((_, n)) = io.builtin_names.iter().find(|(x, _)| *x == b) {
                return Ok(n.clone());
            }
            let (base, attr) = match b {
                BuiltinValue::VertexIndex => ("ganbate_vertex_index", "vertex_id"),
                BuiltinValue::InstanceIndex => ("ganbate_instance_index", "instance_id"),
                BuiltinValue::Position => ("ganbate_position", "position"),
                BuiltinValue::FrontFacing => ("ganbate_front_facing", "front_facing"),
                BuiltinValue::SampleIndex => ("ganbate_sample_index", "sample_id"),
                BuiltinValue::LocalInvocationId => (
                    "ganbate_local_invocation_id",
                    "thread_position_in_threadgroup",
                ),
                BuiltinValue::LocalInvocationIndex => (
                    "ganbate_local_invocation_index",
                    "thread_index_in_threadgroup",
                ),
                BuiltinValue::GlobalInvocationId => {
                    ("ganbate_global_invocation_id", "thread_position_in_grid")
                }
                BuiltinValue::WorkgroupId => {
                    ("ganbate_workgroup_id", "threadgroup_position_in_grid")
                }
                BuiltinValue::NumWorkgroups => ("ganbate_num_workgroups", "threadgroups_per_grid"),
                BuiltinValue::FragDepth => return Err(internal("frag_depth input")),
            };
            let n = self.names.generated(base);
            io.builtins
                .push((format!("{tn} {n} [[{attr}]]"), n.clone()));
            io.builtin_names.push((b, n.clone()));
            return Ok(n);
        }
        let loc = attrs
            .location
            .ok_or_else(|| internal("input without location"))?;
        let attr = if self.stage == Stage::Vertex {
            format!("[[attribute({loc})]]")
        } else {
            let mut a = format!("[[user(locn{loc})");
            let int = ty.scalar().is_some_and(|s| s.is_int());
            if int || attrs.interpolation == Some(Interpolation::Flat) {
                a.push_str(", flat");
            } else {
                let linear = attrs.interpolation == Some(Interpolation::Linear);
                let q = match (attrs.sampling, linear) {
                    (Some(Sampling::Centroid), false) => ", centroid_perspective",
                    (Some(Sampling::Sample), false) => ", sample_perspective",
                    (Some(Sampling::Centroid), true) => ", centroid_no_perspective",
                    (Some(Sampling::Sample), true) => ", sample_no_perspective",
                    (_, true) => ", center_no_perspective",
                    _ => "",
                };
                a.push_str(q);
            }
            a.push_str("]]");
            a
        };
        if !io.inputs.iter().any(|(l, _)| *l == loc) {
            io.inputs.push((loc, format!("{tn} loc{loc} {attr}")));
        }
        Ok(format!("ganbate_in.loc{loc}"))
    }
}

#[derive(Default)]
struct EntryIo {
    /// (location, member declaration) of the `[[stage_in]]` struct.
    inputs: Vec<(u32, String)>,
    /// (parameter declaration, name) of built-in inputs.
    builtins: Vec<(String, String)>,
    builtin_names: Vec<(BuiltinValue, String)>,
}
