use super::*;

impl Emitter<'_> {
    /// Stores `value` into `place` (or into components `swizzle` of the vector at `place`).
    /// Dynamic indices of the target are evaluated once, before the value.
    pub(super) fn assign(&mut self, place: &Place, swizzle: Option<&[u32]>, value: &Expr) -> R<()> {
        if let PlaceRoot::Global(g) = place.root
            && matches!(self.prog.globals[g].kind, GlobalKind::Storage { .. })
        {
            return Err(unsupported(&value.span, "writing a storage buffer"));
        }
        let twice = contains_current(value);
        let target = self.place(place, twice)?;
        let target = match swizzle {
            Some(c) => format!(
                "{target}.{}",
                c.iter()
                    .map(|&i| COMPONENTS[i as usize])
                    .collect::<String>()
            ),
            None => target,
        };
        let saved = self.f.current.replace(target.clone());
        let v = self.expr(value);
        self.f.current = saved;
        let v = v?;
        self.line(&format!("{target} = {v};"));
        Ok(())
    }

    // ------------------------------------------------------------------ expressions

    fn root_ty(&self, root: PlaceRoot) -> R<Ty> {
        match root {
            PlaceRoot::Local(l) => self
                .current_fn
                .and_then(|f| self.prog.functions[f].locals.get(l as usize))
                .map(|l| l.ty.clone())
                .ok_or_else(|| internal("unknown local")),
            PlaceRoot::Global(g) => Ok(self.prog.globals[g].ty.clone()),
        }
    }

    /// The reference expression for `p` (not a storage buffer). With `hoist`, non-constant
    /// indices are first bound to locals so the reference can be spelled twice.
    fn place(&mut self, p: &Place, hoist: bool) -> R<String> {
        let mut s = match p.root {
            PlaceRoot::Local(l) => self
                .f
                .locals
                .get(l as usize)
                .cloned()
                .ok_or_else(|| internal("unknown local"))?,
            PlaceRoot::Global(g) => self
                .globals
                .get(&g)
                .cloned()
                .ok_or_else(|| internal("global variable was not declared"))?,
        };
        let mut ty = self.root_ty(p.root)?;
        for step in &p.steps {
            match step {
                Step::Member(i) => {
                    let Ty::Struct(sid) = ty else {
                        return Err(internal("member of a non-struct"));
                    };
                    let _ = write!(s, ".{}", self.members[sid][*i as usize]);
                    ty = self.prog.structs[sid].members[*i as usize].ty.clone();
                }
                Step::Index(e) => {
                    let constant = e.as_const().and_then(|c| c.as_i64());
                    let (elem, vector) = match &ty {
                        Ty::Vector(_, sc) => (Ty::Scalar(*sc), true),
                        Ty::Matrix(_, r) => (Ty::Vector(*r, Scalar::F32), false),
                        Ty::Array(e, _) => ((**e).clone(), false),
                        _ => return Err(internal("index of a non-composite")),
                    };
                    match constant {
                        Some(c) if vector && (0..4).contains(&c) => {
                            let _ = write!(s, ".{}", COMPONENTS[c as usize]);
                        }
                        Some(c) => {
                            let _ = write!(s, "[{c}]");
                        }
                        None => {
                            let i = self.expr(e)?;
                            if hoist && !matches!(e.kind, EK::Value(_)) {
                                let tmp = self.f.names.generated("ganbate_index");
                                let it = self.ty(&e.ty, &e.span)?;
                                self.line(&format!("{it} {tmp} = {i};"));
                                let _ = write!(s, "[{tmp}]");
                            } else {
                                let _ = write!(s, "[{i}]");
                            }
                        }
                    }
                    ty = elem;
                }
            }
        }
        Ok(s)
    }

    /// A load from storage buffer `g`: the byte offset of the place, then a loader call.
    fn storage_load(&mut self, g: GlobalId, steps: &[Step], span: crate::lexer::Span) -> R<String> {
        let (buf, (group, binding)) = self.storage_buffer(g)?;
        let base = self.hidden(
            format!("{buf}_offset"),
            HiddenValue::StorageOffset(group, binding),
        );
        let mut ty = self.prog.globals[g].ty.clone();
        let mut constant: u64 = 0;
        let mut terms = vec![base];
        for step in steps {
            match step {
                Step::Member(i) => {
                    let Ty::Struct(sid) = ty else {
                        return Err(internal("member of a non-struct"));
                    };
                    let info = &self.prog.structs[sid];
                    let layout = info
                        .layout
                        .as_ref()
                        .ok_or_else(|| internal("storage struct without a layout"))?;
                    constant += u64::from(layout.offsets[*i as usize]);
                    ty = info.members[*i as usize].ty.clone();
                }
                Step::Index(e) => {
                    let (elem, stride) = match &ty {
                        Ty::Array(el, _) => ((**el).clone(), array_stride(el, &self.prog.structs)),
                        Ty::Matrix(_, r) => (Ty::Vector(*r, Scalar::F32), matrix_stride(*r)),
                        Ty::Vector(_, sc) => (Ty::Scalar(*sc), 4),
                        _ => return Err(internal("index of a non-composite")),
                    };
                    match e.as_const().and_then(|c| c.as_i64()) {
                        Some(c) => constant += c.max(0) as u64 * u64::from(stride),
                        None => {
                            let i = self.expr(e)?;
                            terms.push(format!("uint({i}) * {stride}u"));
                        }
                    }
                    ty = elem;
                }
            }
        }
        if constant > 0 {
            terms.push(format!("{}u", constant as u32));
        }
        let f = self.loader(&ty, &span)?;
        Ok(format!("{f}({buf}, {})", terms.join(" + ")))
    }

    fn args(&mut self, args: &[Expr]) -> R<Vec<String>> {
        let mut out = Vec::with_capacity(args.len());
        for a in args {
            out.push(self.expr(a)?);
        }
        Ok(out)
    }

    pub(super) fn expr(&mut self, e: &Expr) -> R<String> {
        Ok(match &e.kind {
            EK::Const(c) => self.konst(&e.ty, c, &e.span)?,
            EK::Zero => self.zero(&e.ty, &e.span)?,
            EK::Value(v) => self
                .f
                .values
                .get(v)
                .cloned()
                .ok_or_else(|| internal("value used before its definition"))?,
            EK::Load(p) => match p.root {
                PlaceRoot::Global(g)
                    if matches!(self.prog.globals[g].kind, GlobalKind::Storage { .. }) =>
                {
                    self.storage_load(g, &p.steps, e.span)?
                }
                _ => self.place(p, false)?,
            },
            EK::Extract(b, i) => {
                let base = self.expr(b)?;
                match &b.ty {
                    Ty::Vector(..) => format!("{base}.{}", COMPONENTS[*i as usize]),
                    Ty::Struct(sid) => format!("{base}.{}", self.members[*sid][*i as usize]),
                    _ => format!("{base}[{i}]"),
                }
            }
            EK::DynIndex(b, i) => {
                let base = self.expr(b)?;
                let i = self.expr(i)?;
                format!("{base}[{i}]")
            }
            EK::Swizzle(b, comps) => {
                let base = self.expr(b)?;
                let c: String = comps.iter().map(|&i| COMPONENTS[i as usize]).collect();
                format!("{base}.{c}")
            }
            EK::Unary(op, x) => {
                let vector = matches!(x.ty, Ty::Vector(..));
                let x = self.expr(x)?;
                match op {
                    UnaryOp::Neg => format!("(-{x})"),
                    UnaryOp::Not if vector => format!("not({x})"),
                    UnaryOp::Not => format!("(!{x})"),
                    UnaryOp::BitNot => format!("(~{x})"),
                }
            }
            EK::Binary(op, l, r) => self.binary(*op, l, r, &e.ty, &e.span)?,
            EK::Construct(parts) => {
                let t = self.ty(&e.ty, &e.span)?;
                format!("{t}({})", self.args(parts)?.join(", "))
            }
            EK::Splat(x) | EK::Convert(x) => {
                let t = self.ty(&e.ty, &e.span)?;
                let x = self.expr(x)?;
                format!("{t}({x})")
            }
            EK::Bitcast(x) => self.bitcast(x, &e.ty, &e.span)?,
            EK::Call(f, args) => {
                let name = self
                    .fns
                    .get(f)
                    .cloned()
                    .ok_or_else(|| internal("callee not emitted"))?;
                format!("{name}({})", self.args(args)?.join(", "))
            }
            EK::Builtin(b, args) => self.builtin(*b, args, &e.ty, &e.span)?,
            EK::Texture(tc) => self.texture(tc, &e.span)?,
            EK::ArrayLength { global, member } => {
                let (buf, (group, binding)) = self.storage_buffer(*global)?;
                let size = self.hidden(
                    format!("{buf}_size"),
                    HiddenValue::StorageSize(group, binding),
                );
                let structs = &self.prog.structs;
                let (offset, elem) = match (member, &self.prog.globals[*global].ty) {
                    (Some(m), Ty::Struct(sid)) => {
                        let info = &structs[*sid];
                        let off = info.layout.as_ref().map_or(0, |l| l.offsets[*m as usize]);
                        (off, info.members[*m as usize].ty.clone())
                    }
                    (_, t) => (0, t.clone()),
                };
                let Ty::Array(el, None) = elem else {
                    return Err(internal("arrayLength of a non-runtime array"));
                };
                let stride = array_stride(&el, structs);
                format!("(({size} - {offset}u) / {stride}u)")
            }
            EK::Current => self
                .f
                .current
                .clone()
                .ok_or_else(|| internal("compound assignment value missing"))?,
        })
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        l: &Expr,
        r: &Expr,
        ty: &Ty,
        span: &crate::lexer::Span,
    ) -> R<String> {
        use BinaryOp::*;
        let a = self.expr(l)?;
        let b = self.expr(r)?;
        let ls = l.ty.scalar().unwrap_or(Scalar::F32).concretize();
        let vector_operands = matches!(l.ty, Ty::Vector(..)) || matches!(r.ty, Ty::Vector(..));
        let splat = |e: &Self, text: &str, from: &Ty| -> R<String> {
            if from.is_scalar() && !ty.is_scalar() {
                Ok(format!("{}({text})", e.ty(ty, span)?))
            } else {
                Ok(text.to_string())
            }
        };
        Ok(match op {
            Eq | Ne | Lt | Le | Gt | Ge if vector_operands => {
                let f = match op {
                    Eq => "equal",
                    Ne => "notEqual",
                    Lt => "lessThan",
                    Le => "lessThanEqual",
                    Gt => "greaterThan",
                    _ => "greaterThanEqual",
                };
                format!("{f}({a}, {b})")
            }
            Eq | Ne | Lt | Le | Gt | Ge => format!("({a} {} {b})", op.symbol()),
            Rem if ls == Scalar::F32 => {
                let (a, b) = (splat(self, &a, &l.ty)?, splat(self, &b, &r.ty)?);
                let f = self.overload(
                    "frem",
                    ty,
                    "$T ganbate_frem($T a, $T b) {\n  return a - b * trunc(a / b);\n}\n",
                    span,
                )?;
                format!("{f}({a}, {b})")
            }
            Rem if ls == Scalar::I32 => {
                let (a, b) = (splat(self, &a, &l.ty)?, splat(self, &b, &r.ty)?);
                let f = self.overload(
                    "irem",
                    ty,
                    "$T ganbate_irem($T a, $T b) {\n  return a - b * (a / b);\n}\n",
                    span,
                )?;
                format!("{f}({a}, {b})")
            }
            Add | Sub | Mul | Div | Rem => format!("({a} {} {b})", op.symbol()),
            And | Or if ls == Scalar::Bool => {
                // Both operands are evaluated, as in WGSL.
                let (name, sym) = if op == And {
                    ("and", "&&")
                } else {
                    ("or", "||")
                };
                let code = match ty {
                    Ty::Vector(n, _) => {
                        let parts: Vec<String> = (0..*n as usize)
                            .map(|i| format!("a.{c} {sym} b.{c}", c = COMPONENTS[i]))
                            .collect();
                        format!(
                            "$T ganbate_{name}($T a, $T b) {{\n  return $T({});\n}}\n",
                            parts.join(", ")
                        )
                    }
                    _ => {
                        format!("bool ganbate_{name}(bool a, bool b) {{\n  return a {sym} b;\n}}\n")
                    }
                };
                let f = self.overload(name, ty, &code, span)?;
                format!("{f}({a}, {b})")
            }
            Xor if ls == Scalar::Bool => {
                if ty.is_scalar() {
                    format!("({a} != {b})")
                } else {
                    format!("notEqual({a}, {b})")
                }
            }
            And | Or | Xor => format!("({a} {} {b})", op.symbol()),
            Shl | Shr => {
                let amount = if r.as_const().is_some() {
                    b
                } else {
                    format!("({b} & 31u)")
                };
                format!("({a} {} {amount})", op.symbol())
            }
            LogicalAnd | LogicalOr => format!("({a} {} {b})", op.symbol()),
        })
    }

    fn bitcast(&mut self, x: &Expr, to: &Ty, span: &crate::lexer::Span) -> R<String> {
        let v = self.expr(x)?;
        let from = x.ty.scalar().unwrap_or(Scalar::F32).concretize();
        let ts = to.scalar().unwrap_or(Scalar::F32).concretize();
        let t = self.ty(to, span)?;
        Ok(match (from, ts) {
            (a, b) if a == b => v,
            (Scalar::F32, Scalar::U32) => format!("floatBitsToUint({v})"),
            (Scalar::F32, Scalar::I32) => format!("floatBitsToInt({v})"),
            (Scalar::U32, Scalar::F32) => format!("uintBitsToFloat({v})"),
            (Scalar::I32, Scalar::F32) => format!("intBitsToFloat({v})"),
            _ => format!("{t}({v})"),
        })
    }

    fn builtin(&mut self, b: B, args: &[Expr], ty: &Ty, span: &crate::lexer::Span) -> R<String> {
        let v = self.args(args)?;
        let s = args
            .first()
            .and_then(|a| a.ty.scalar())
            .unwrap_or(Scalar::F32)
            .concretize();
        let arg_ty = args.first().map(|a| a.ty.clone()).unwrap_or(Ty::Void);
        let call = |name: &str, v: &[String]| format!("{name}({})", v.join(", "));
        Ok(match b {
            B::Abs if s == Scalar::U32 => v[0].clone(),
            B::Abs => call("abs", &v),
            B::Acos => call("acos", &v),
            B::Acosh => call("acosh", &v),
            B::Asin => call("asin", &v),
            B::Asinh => call("asinh", &v),
            B::Atan => call("atan", &v),
            B::Atanh => call("atanh", &v),
            B::Atan2 => call("atan", &v),
            B::Ceil => call("ceil", &v),
            B::Clamp => call("clamp", &v),
            B::Cos => call("cos", &v),
            B::Cosh => call("cosh", &v),
            B::CountOneBits => {
                self.overload(
                    "count",
                    &Ty::Scalar(Scalar::U32),
                    "uint ganbate_count(uint v) {\n  v = v - ((v >> 1u) & 0x55555555u);\n  \
                     v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);\n  \
                     return (((v + (v >> 4u)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24u;\n}\n",
                    span,
                )?;
                if s == Scalar::I32 {
                    self.overload(
                        "count",
                        &Ty::Scalar(Scalar::I32),
                        "int ganbate_count(int v) {\n  return int(ganbate_count(uint(v)));\n}\n",
                        span,
                    )?;
                }
                if let Ty::Vector(n, _) = arg_ty {
                    let parts: Vec<String> = (0..n as usize)
                        .map(|i| format!("ganbate_count(v.{})", COMPONENTS[i]))
                        .collect();
                    let code = format!(
                        "$T ganbate_count($T v) {{\n  return $T({});\n}}\n",
                        parts.join(", ")
                    );
                    self.overload("count", &arg_ty, &code, span)?;
                }
                call("ganbate_count", &v)
            }
            B::Cross => call("cross", &v),
            B::Degrees => call("degrees", &v),
            B::Determinant => call("determinant", &v),
            B::Distance => call("distance", &v),
            B::Dot if s == Scalar::F32 => call("dot", &v),
            B::Dot => {
                let Ty::Vector(n, _) = arg_ty else {
                    return Err(internal("dot of a non-vector"));
                };
                let terms: Vec<String> = (0..n as usize)
                    .map(|i| format!("a.{c} * b.{c}", c = COMPONENTS[i]))
                    .collect();
                let r = scalar_name(s);
                let code = format!(
                    "{r} ganbate_dot($T a, $T b) {{\n  return {};\n}}\n",
                    terms.join(" + ")
                );
                self.overload("dot", &arg_ty, &code, span)?;
                call("ganbate_dot", &v)
            }
            B::Dpdx => call("dFdx", &v),
            B::Dpdy => call("dFdy", &v),
            B::Exp => call("exp", &v),
            B::Exp2 => call("exp2", &v),
            B::FaceForward => call("faceforward", &v),
            B::Floor => call("floor", &v),
            B::Fma => format!("({} * {} + {})", v[0], v[1], v[2]),
            B::Fract => call("fract", &v),
            B::Fwidth => call("fwidth", &v),
            B::InverseSqrt => call("inversesqrt", &v),
            B::Ldexp => {
                let t = self.ty(&arg_ty, span)?;
                format!("({} * exp2({t}({})))", v[0], v[1])
            }
            B::Length => call("length", &v),
            B::Log => call("log", &v),
            B::Log2 => call("log2", &v),
            B::Max => call("max", &v),
            B::Min => call("min", &v),
            B::Mix => call("mix", &v),
            B::Normalize => call("normalize", &v),
            B::Pow => call("pow", &v),
            B::Radians => call("radians", &v),
            B::Reflect => call("reflect", &v),
            B::Refract => call("refract", &v),
            B::ReverseBits => {
                self.overload(
                    "reverse",
                    &Ty::Scalar(Scalar::U32),
                    "uint ganbate_reverse(uint v) {\n  \
                     v = ((v >> 1u) & 0x55555555u) | ((v & 0x55555555u) << 1u);\n  \
                     v = ((v >> 2u) & 0x33333333u) | ((v & 0x33333333u) << 2u);\n  \
                     v = ((v >> 4u) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4u);\n  \
                     v = ((v >> 8u) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8u);\n  \
                     return (v >> 16u) | (v << 16u);\n}\n",
                    span,
                )?;
                if s == Scalar::I32 {
                    self.overload(
                        "reverse",
                        &Ty::Scalar(Scalar::I32),
                        "int ganbate_reverse(int v) {\n  return int(ganbate_reverse(uint(v)));\n}\n",
                        span,
                    )?;
                }
                if let Ty::Vector(n, _) = arg_ty {
                    let parts: Vec<String> = (0..n as usize)
                        .map(|i| format!("ganbate_reverse(v.{})", COMPONENTS[i]))
                        .collect();
                    let code = format!(
                        "$T ganbate_reverse($T v) {{\n  return $T({});\n}}\n",
                        parts.join(", ")
                    );
                    self.overload("reverse", &arg_ty, &code, span)?;
                }
                call("ganbate_reverse", &v)
            }
            B::Round => call("roundEven", &v),
            B::Saturate => format!("clamp({}, 0.0, 1.0)", v[0]),
            B::Sign => call("sign", &v),
            B::Sin => call("sin", &v),
            B::Sinh => call("sinh", &v),
            B::Smoothstep => call("smoothstep", &v),
            B::Sqrt => call("sqrt", &v),
            B::Step => call("step", &v),
            B::Tan => call("tan", &v),
            B::Tanh => call("tanh", &v),
            B::Transpose => call("transpose", &v),
            B::Trunc => call("trunc", &v),
            B::Select => {
                // select(f, t, cond)
                let cond_vector = matches!(args[2].ty, Ty::Vector(..));
                if !cond_vector {
                    format!("({} ? {} : {})", v[2], v[1], v[0])
                } else if s == Scalar::F32 {
                    format!("mix({}, {}, {})", v[0], v[1], v[2])
                } else {
                    let Ty::Vector(n, _) = ty else {
                        return Err(internal("select of a non-vector"));
                    };
                    let parts: Vec<String> = (0..*n as usize)
                        .map(|i| format!("c.{x} ? t.{x} : f.{x}", x = COMPONENTS[i]))
                        .collect();
                    let code = format!(
                        "$T ganbate_select($T f, $T t, $B c) {{\n  return $T({});\n}}\n",
                        parts.join(", ")
                    );
                    self.overload("select", ty, &code, span)?;
                    call("ganbate_select", &v)
                }
            }
            B::All if arg_ty.is_scalar() => v[0].clone(),
            B::Any if arg_ty.is_scalar() => v[0].clone(),
            B::All => call("all", &v),
            B::Any => call("any", &v),
            B::Pack2x16Snorm => call("packSnorm2x16", &v),
            B::Pack2x16Unorm => call("packUnorm2x16", &v),
            B::Pack2x16Float => call("packHalf2x16", &v),
            B::Unpack2x16Snorm => call("unpackSnorm2x16", &v),
            B::Unpack2x16Unorm => call("unpackUnorm2x16", &v),
            B::Unpack2x16Float => call("unpackHalf2x16", &v),
            B::Pack4x8Unorm => {
                self.helper("pack4x8unorm", |_| {
                    Ok("uint ganbate_pack4x8unorm(vec4 v) {\n  \
                        uvec4 u = uvec4(round(clamp(v, 0.0, 1.0) * 255.0));\n  \
                        return u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u);\n}\n"
                        .into())
                })?;
                call("ganbate_pack4x8unorm", &v)
            }
            B::Pack4x8Snorm => {
                self.helper("pack4x8snorm", |_| {
                    Ok("uint ganbate_pack4x8snorm(vec4 v) {\n  \
                        uvec4 u = uvec4(ivec4(round(clamp(v, -1.0, 1.0) * 127.0))) & 255u;\n  \
                        return u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u);\n}\n"
                        .into())
                })?;
                call("ganbate_pack4x8snorm", &v)
            }
            B::Unpack4x8Unorm => {
                self.helper("unpack4x8unorm", |_| {
                    Ok("vec4 ganbate_unpack4x8unorm(uint v) {\n  \
                        return vec4(uvec4(v, v >> 8u, v >> 16u, v >> 24u) & 255u) / 255.0;\n}\n"
                        .into())
                })?;
                call("ganbate_unpack4x8unorm", &v)
            }
            B::Unpack4x8Snorm => {
                self.helper("unpack4x8snorm", |_| {
                    Ok("vec4 ganbate_unpack4x8snorm(uint v) {\n  \
                        ivec4 i = ivec4(uvec4(v << 24u, v << 16u, v << 8u, v)) >> 24;\n  \
                        return clamp(vec4(i) / 127.0, -1.0, 1.0);\n}\n"
                        .into())
                })?;
                call("ganbate_unpack4x8snorm", &v)
            }
            B::WorkgroupBarrier | B::StorageBarrier => {
                return Err(unsupported(span, "a workgroup barrier"));
            }
            B::TextureSample
            | B::TextureSampleBias
            | B::TextureSampleLevel
            | B::TextureSampleGrad
            | B::TextureSampleCompare
            | B::TextureSampleCompareLevel
            | B::TextureLoad
            | B::TextureStore
            | B::TextureDimensions
            | B::TextureNumLevels
            | B::TextureNumLayers => return Err(internal("texture built-in outside a TexCall")),
        })
    }

    fn texture(&mut self, tc: &TexCall, span: &crate::lexer::Span) -> R<String> {
        let tex = &self.prog.globals[tc.texture];
        let Ty::Texture(view, sample) = tex.ty else {
            return Err(unsupported(span, "a storage texture"));
        };
        let depth = sample == SampleType::Depth;
        let opt = |e: &mut Self, x: &Option<Expr>| -> R<Option<String>> {
            x.as_ref().map(|x| e.expr(x)).transpose()
        };
        let coords = opt(self, &tc.coords)?;
        let layer = opt(self, &tc.array_index)?;
        let level = opt(self, &tc.level)?;
        let bias = opt(self, &tc.bias)?;
        let ddx = opt(self, &tc.ddx)?;
        let ddy = opt(self, &tc.ddy)?;
        let reference = opt(self, &tc.depth_ref)?;
        let offset = opt(self, &tc.offset)?;
        let need = |x: &Option<String>| x.clone().ok_or_else(|| internal("missing operand"));
        let lod_int = |l: &Option<String>| match l {
            Some(l) => format!("int({l})"),
            None => "0".into(),
        };
        Ok(match tc.op {
            B::TextureStore => return Err(unsupported(span, "textureStore")),
            B::TextureNumLevels => {
                let (g, b) = tex.binding.ok_or_else(|| internal("unbound texture"))?;
                self.hidden(
                    format!("ganbate_levels_{g}_{b}"),
                    HiddenValue::TextureLevels(g, b),
                )
            }
            B::TextureDimensions => {
                let u = self.unit(tc.texture, None)?;
                let lod = lod_int(&level);
                match view {
                    TextureView::D2Array => format!("uvec2(textureSize({u}, {lod}).xy)"),
                    _ => format!("uvec2(textureSize({u}, {lod}))"),
                }
            }
            B::TextureNumLayers => {
                let u = self.unit(tc.texture, None)?;
                format!("uint(textureSize({u}, 0).z)")
            }
            B::TextureLoad => {
                let u = self.unit(tc.texture, None)?;
                let c = need(&coords)?;
                let p = match view {
                    TextureView::D2 => format!("ivec2({c})"),
                    TextureView::D2Array => {
                        format!("ivec3(ivec2({c}), int({}))", need(&layer)?)
                    }
                    TextureView::Cube => return Err(unsupported(span, "textureLoad on a cube")),
                };
                let r = format!("texelFetch({u}, {p}, {})", lod_int(&level));
                if depth { format!("{r}.r") } else { r }
            }
            B::TextureSample
            | B::TextureSampleBias
            | B::TextureSampleLevel
            | B::TextureSampleGrad => {
                let u = self.unit(tc.texture, tc.sampler)?;
                let c = need(&coords)?;
                let p = match view {
                    TextureView::D2Array => format!("vec3({c}, float({}))", need(&layer)?),
                    _ => c,
                };
                let r = match (tc.op, &offset) {
                    (B::TextureSample, None) => format!("texture({u}, {p})"),
                    (B::TextureSample, Some(o)) => format!("textureOffset({u}, {p}, {o})"),
                    (B::TextureSampleBias, None) => {
                        format!("texture({u}, {p}, {})", need(&bias)?)
                    }
                    (B::TextureSampleBias, Some(o)) => {
                        format!("textureOffset({u}, {p}, {o}, {})", need(&bias)?)
                    }
                    (B::TextureSampleLevel, o) => {
                        let l = need(&level)?;
                        let l = if depth { format!("float({l})") } else { l };
                        match o {
                            None => format!("textureLod({u}, {p}, {l})"),
                            Some(o) => format!("textureLodOffset({u}, {p}, {l}, {o})"),
                        }
                    }
                    (_, None) => {
                        format!("textureGrad({u}, {p}, {}, {})", need(&ddx)?, need(&ddy)?)
                    }
                    (_, Some(o)) => format!(
                        "textureGradOffset({u}, {p}, {}, {}, {o})",
                        need(&ddx)?,
                        need(&ddy)?
                    ),
                };
                if depth { format!("{r}.r") } else { r }
            }
            B::TextureSampleCompare | B::TextureSampleCompareLevel => {
                let u = self.unit(tc.texture, tc.sampler)?;
                let c = need(&coords)?;
                let r = need(&reference)?;
                let p = match view {
                    TextureView::D2 => format!("vec3({c}, {r})"),
                    TextureView::D2Array => format!("vec4({c}, float({}), {r})", need(&layer)?),
                    TextureView::Cube => format!("vec4({c}, {r})"),
                };
                let zero = if view == TextureView::Cube {
                    "vec3(0.0)"
                } else {
                    "vec2(0.0)"
                };
                match (tc.op, view, &offset) {
                    (B::TextureSampleCompare, _, None) => format!("texture({u}, {p})"),
                    (B::TextureSampleCompare, _, Some(o)) => {
                        format!("textureOffset({u}, {p}, {o})")
                    }
                    (_, TextureView::D2, None) => format!("textureLod({u}, {p}, 0.0)"),
                    (_, TextureView::D2, Some(o)) => {
                        format!("textureLodOffset({u}, {p}, 0.0, {o})")
                    }
                    (_, _, None) => format!("textureGrad({u}, {p}, {zero}, {zero})"),
                    (_, _, Some(o)) => {
                        format!("textureGradOffset({u}, {p}, {zero}, {zero}, {o})")
                    }
                }
            }
            _ => return Err(internal("not a texture built-in")),
        })
    }

    // ------------------------------------------------------------------ entry point

    /// `main`: reads the stage inputs, calls the user's entry function and writes its outputs.
    pub(super) fn wrapper(&mut self, fid: FnId) -> R<String> {
        let prog = self.prog;
        let f = &prog.functions[fid];
        self.current_fn = Some(fid);
        self.f = FnCx {
            names: self.module.clone(),
            ..FnCx::default()
        };
        let mut decls: Vec<String> = Vec::new();
        let mut declared: HashSet<u32> = HashSet::new();
        let mut args = Vec::new();
        for p in &f.params {
            if !p.io.is_empty() {
                args.push(self.input(&p.io, &p.ty, &mut decls, &mut declared)?);
            } else if let Ty::Struct(sid) = p.ty {
                let mut parts = Vec::new();
                for m in &prog.structs[sid].members {
                    parts.push(self.input(&m.io, &m.ty, &mut decls, &mut declared)?);
                }
                args.push(format!("{}({})", self.structs[sid], parts.join(", ")));
            } else {
                return Err(internal("entry-point parameter without IO attributes"));
            }
        }
        let call = format!("{}({})", self.fns[&fid], args.join(", "));
        let mut body = Vec::new();
        let mut writes_position = false;
        let mut outputs: Vec<(IoAttrs, Ty, String)> = Vec::new();
        match &f.ret {
            None => body.push(format!("{call};")),
            Some(ty) if !f.ret_io.is_empty() => {
                let t = self.ty(ty, &f.span)?;
                body.push(format!("{t} ganbate_result = {call};"));
                outputs.push((f.ret_io, ty.clone(), "ganbate_result".into()));
            }
            Some(Ty::Struct(sid)) => {
                let t = self.structs[*sid].clone();
                body.push(format!("{t} ganbate_result = {call};"));
                for (i, m) in prog.structs[*sid].members.iter().enumerate() {
                    outputs.push((
                        m.io,
                        m.ty.clone(),
                        format!("ganbate_result.{}", self.members[*sid][i]),
                    ));
                }
            }
            Some(_) => return Err(internal("entry-point output without IO attributes")),
        }
        let position = outputs
            .iter()
            .find(|(io, _, _)| io.builtin == Some(BuiltinValue::Position))
            .map(|(_, _, v)| v.clone());
        let mut linear_w = false;
        for (io, ty, value) in outputs {
            if let Some(b) = io.builtin {
                match (self.stage, b) {
                    (Stage::Vertex, BuiltinValue::Position) => {
                        if io.invariant {
                            decls.push("invariant gl_Position;".into());
                        }
                        body.push(format!("gl_Position = {value};"));
                        writes_position = true;
                    }
                    (Stage::Fragment, BuiltinValue::FragDepth) => {
                        body.push(format!("gl_FragDepth = {value};"));
                    }
                    _ => return Err(unsupported(&io.span, &format!("output {}", b.name()))),
                }
                continue;
            }
            let loc = io.location.unwrap_or(0);
            let t = self.ty(&ty, &io.span)?;
            match self.stage {
                Stage::Vertex => {
                    let q = self.interpolation(&io, &ty)?;
                    decls.push(format!("{q}out {t} ganbate_v{loc};"));
                    if io.interpolation == Some(Interpolation::Linear) {
                        // Screen-linear: interpolate value * w perspective-correctly and divide
                        // by the interpolated w in the fragment shader.
                        let pos = position
                            .clone()
                            .ok_or_else(|| internal("vertex output without a position"))?;
                        if !linear_w {
                            decls.push(format!("{q}out float ganbate_vw;"));
                            body.push(format!("ganbate_vw = {pos}.w;"));
                            linear_w = true;
                        }
                        body.push(format!("ganbate_v{loc} = {value} * {pos}.w;"));
                    } else {
                        body.push(format!("ganbate_v{loc} = {value};"));
                    }
                }
                _ => {
                    decls.push(format!("layout(location = {loc}) out {t} ganbate_o{loc};"));
                    body.push(format!("ganbate_o{loc} = {value};"));
                }
            }
        }
        if writes_position {
            // Vulkan's depth range 0..w becomes GL's -w..w; the depth buffer gets the same values.
            body.push("gl_Position.z = 2.0 * gl_Position.z - gl_Position.w;".into());
        }
        let mut out = String::new();
        for d in decls {
            out.push_str(&d);
            out.push('\n');
        }
        out.push_str("\nvoid main() {\n");
        for b in body {
            let _ = writeln!(out, "  {b}");
        }
        out.push_str("}\n");
        Ok(out)
    }

    fn interpolation(&self, io: &IoAttrs, ty: &Ty) -> R<String> {
        let int = ty.scalar().is_some_and(|s| s.is_int());
        if io.sampling == Some(Sampling::Sample) {
            return Err(unsupported(&io.span, "per-sample interpolation"));
        }
        let centroid = if io.sampling == Some(Sampling::Centroid) {
            "centroid "
        } else {
            ""
        };
        Ok(match io.interpolation {
            _ if int => format!("flat {centroid}"),
            Some(Interpolation::Flat) => format!("flat {centroid}"),
            _ => centroid.to_string(),
        })
    }

    /// The expression that reads one entry-point input, declaring it on first use.
    fn input(
        &mut self,
        io: &IoAttrs,
        ty: &Ty,
        decls: &mut Vec<String>,
        declared: &mut HashSet<u32>,
    ) -> R<String> {
        if let Some(b) = io.builtin {
            return Ok(match (self.stage, b) {
                (Stage::Vertex, BuiltinValue::VertexIndex) => {
                    let base = self.hidden("ganbate_base_vertex".into(), HiddenValue::BaseVertex);
                    format!("uint(gl_VertexID + {base})")
                }
                (Stage::Vertex, BuiltinValue::InstanceIndex) => {
                    let base =
                        self.hidden("ganbate_base_instance".into(), HiddenValue::BaseInstance);
                    format!("uint(gl_InstanceID + {base})")
                }
                (Stage::Fragment, BuiltinValue::Position) => "gl_FragCoord".into(),
                (Stage::Fragment, BuiltinValue::FrontFacing) => "gl_FrontFacing".into(),
                _ => return Err(unsupported(&io.span, &format!("input {}", b.name()))),
            });
        }
        let loc = io.location.unwrap_or(0);
        let t = self.ty(ty, &io.span)?;
        let name = match self.stage {
            Stage::Vertex => format!("ganbate_a{loc}"),
            _ => format!("ganbate_v{loc}"),
        };
        let linear =
            self.stage == Stage::Fragment && io.interpolation == Some(Interpolation::Linear);
        if declared.insert(loc) {
            let decl = match self.stage {
                Stage::Vertex => format!("layout(location = {loc}) in {t} {name};"),
                _ => {
                    let q = self.interpolation(io, ty)?;
                    if linear && !decls.iter().any(|d| d.ends_with(" ganbate_vw;")) {
                        decls.push(format!("{q}in float ganbate_vw;"));
                    }
                    format!("{q}in {t} {name};")
                }
            };
            decls.push(decl);
        }
        Ok(if linear {
            format!("({name} / ganbate_vw)")
        } else {
            name
        })
    }
}

fn contains_current(e: &Expr) -> bool {
    let mut found = false;
    walk(e, &mut |x| {
        if matches!(x.kind, EK::Current) {
            found = true;
        }
    });
    found
}
