use super::*;

impl<'p> Emitter<'p> {
    pub(super) fn assign(&mut self, place: &Place, swizzle: Option<&[u32]>, value: &Expr) -> R<()> {
        let (ptr, fl) = self.place_ptr(place)?;
        let uses_current = contains_current(value);
        match swizzle {
            Some(comps) => {
                let Ty::Vector(n, s) = place.ty else {
                    return Err(internal(value.span, "swizzle store to a non-vector"));
                };
                let vt = self.ty(&place.ty, Flavor::Plain);
                let old = self.op(op::LOAD, vt, &[ptr]);
                if uses_current {
                    let st = self.ty(&Ty::Vector(comps.len() as u8, s), Flavor::Plain);
                    let mut w = vec![old, old];
                    w.extend(comps);
                    let cur = self.op(op::VECTOR_SHUFFLE, st, &w);
                    self.fs.current = Some(Val::plain(cur));
                }
                let new = self.expr(value)?.id;
                self.fs.current = None;
                let mut sel: Vec<u32> = (0..n as u32).collect();
                for (p, &c) in comps.iter().enumerate() {
                    sel[c as usize] = n as u32 + p as u32;
                }
                let mut w = vec![old, new];
                w.extend(sel);
                let res = self.op(op::VECTOR_SHUFFLE, vt, &w);
                self.emit(op::STORE, &[ptr, res]);
            }
            None => {
                if uses_current {
                    let t = self.ty(&place.ty, fl);
                    let cur = self.op(op::LOAD, t, &[ptr]);
                    self.fs.current = Some(Val { id: cur, fl });
                }
                let v = self.expr(value)?;
                self.fs.current = None;
                let v = self.convert_flavor(v, &place.ty, fl)?;
                self.emit(op::STORE, &[ptr, v.id]);
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------ memory

    /// Pointer to a place, and the flavor of the pointee.
    fn place_ptr(&mut self, p: &Place) -> R<(u32, Flavor)> {
        let (base, class, root_fl, wrapped) = match p.root {
            PlaceRoot::Local(l) => {
                let id = *self
                    .fs
                    .locals
                    .get(l as usize)
                    .ok_or_else(|| internal(self.span, "unknown local"))?;
                (id, sc::FUNCTION, Flavor::Plain, false)
            }
            PlaceRoot::Global(g) => {
                let info = self.global(g)?;
                (info.id, info.sc, info.fl, info.wrapped)
            }
        };
        if p.steps.is_empty() && !wrapped {
            return Ok((base, root_fl));
        }
        let mut indices = Vec::new();
        if wrapped {
            indices.push(self.const_u32(0));
        }
        for s in &p.steps {
            match s {
                Step::Member(i) => indices.push(self.const_u32(*i)),
                Step::Index(e) => {
                    let v = self.expr(e)?;
                    indices.push(v.id);
                }
            }
        }
        let fl = if p.steps.is_empty() {
            root_fl
        } else {
            root_fl.member()
        };
        let pointee = self.ty(&p.ty, fl);
        let pt = self.ptr(class, pointee);
        let mut w = vec![base];
        w.extend(indices);
        Ok((self.op(op::ACCESS_CHAIN, pt, &w), fl))
    }

    fn load_place(&mut self, p: &Place) -> R<Val> {
        let (ptr, fl) = self.place_ptr(p)?;
        let t = self.ty(&p.ty, fl);
        Ok(Val {
            id: self.op(op::LOAD, t, &[ptr]),
            fl,
        })
    }

    /// Re-express a composite value in another layout flavor, member by member.
    fn convert_flavor(&mut self, v: Val, ty: &Ty, to: Flavor) -> R<Val> {
        if v.fl == to || !ty.has_layout_variants() {
            return Ok(Val { id: v.id, fl: to });
        }
        if self.ty(ty, v.fl) == self.ty(ty, to) {
            return Ok(Val { id: v.id, fl: to });
        }
        let prog = self.prog;
        let parts: Vec<Ty> = match ty {
            Ty::Struct(sid) => prog.structs[*sid]
                .members
                .iter()
                .map(|m| m.ty.clone())
                .collect(),
            Ty::Array(e, Some(n)) => vec![(**e).clone(); *n as usize],
            _ => {
                return Err(internal(
                    self.span,
                    "cannot convert the layout of this value",
                ));
            }
        };
        if parts.len() > 65_000 {
            return Err(self
                .span
                .error("array is too large to copy between buffer and function memory"));
        }
        let mut ids = Vec::with_capacity(parts.len());
        for (i, pt) in parts.iter().enumerate() {
            let from_t = self.ty(pt, v.fl.member());
            let x = self.op(op::COMPOSITE_EXTRACT, from_t, &[v.id, i as u32]);
            let x = self.convert_flavor(
                Val {
                    id: x,
                    fl: v.fl.member(),
                },
                pt,
                to.member(),
            )?;
            ids.push(x.id);
        }
        let t = self.ty(ty, to);
        Ok(Val {
            id: self.op(op::COMPOSITE_CONSTRUCT, t, &ids),
            fl: to,
        })
    }

    pub(super) fn plain_id(&mut self, v: Val, ty: &Ty) -> R<u32> {
        Ok(self.convert_flavor(v, ty, Flavor::Plain)?.id)
    }

    // ------------------------------------------------------------------ expressions

    fn splat(&mut self, scalar: u32, vec_ty: &Ty) -> u32 {
        let n = vec_ty.vec_size().unwrap_or(1);
        let t = self.ty(vec_ty, Flavor::Plain);
        self.op(op::COMPOSITE_CONSTRUCT, t, &vec![scalar; n as usize])
    }

    pub(super) fn expr(&mut self, e: &Expr) -> R<Val> {
        Ok(match &e.kind {
            EK::Const(c) => Val::plain(self.const_value(&e.ty, c, Flavor::Plain)?),
            EK::Zero => Val::plain(self.null(&e.ty, Flavor::Plain)),
            EK::Value(v) => *self
                .fs
                .values
                .get(v)
                .ok_or_else(|| internal(e.span, "value used before its definition"))?,
            EK::Load(p) => self.load_place(p)?,
            EK::Extract(b, i) => {
                let bv = self.expr(b)?;
                let fl = bv.fl.member();
                let t = self.ty(&e.ty, fl);
                Val {
                    id: self.op(op::COMPOSITE_EXTRACT, t, &[bv.id, *i]),
                    fl,
                }
            }
            EK::DynIndex(b, i) => {
                let bv = self.expr(b)?;
                let iv = self.expr(i)?.id;
                let et = self.ty(&e.ty, Flavor::Plain);
                if let Ty::Vector(..) = b.ty {
                    Val::plain(self.op(op::VECTOR_EXTRACT_DYNAMIC, et, &[bv.id, iv]))
                } else {
                    let pv = self.plain_id(bv, &b.ty)?;
                    let var = self.temp_var(&b.ty);
                    self.emit(op::STORE, &[var, pv]);
                    let pt = self.ptr(sc::FUNCTION, et);
                    let ptr = self.op(op::ACCESS_CHAIN, pt, &[var, iv]);
                    Val::plain(self.op(op::LOAD, et, &[ptr]))
                }
            }
            EK::Swizzle(b, comps) => {
                let bv = self.expr(b)?.id;
                let t = self.ty(&e.ty, Flavor::Plain);
                let mut w = vec![bv, bv];
                w.extend(comps);
                Val::plain(self.op(op::VECTOR_SHUFFLE, t, &w))
            }
            EK::Unary(o, x) => {
                let v = self.expr(x)?.id;
                let t = self.ty(&e.ty, Flavor::Plain);
                let opc = match o {
                    UnaryOp::Neg if e.ty.scalar().is_some_and(|s| s.is_float()) => op::F_NEGATE,
                    UnaryOp::Neg => op::S_NEGATE,
                    UnaryOp::Not => op::LOGICAL_NOT,
                    UnaryOp::BitNot => op::NOT,
                };
                Val::plain(self.op(opc, t, &[v]))
            }
            EK::Binary(o, l, r) => Val::plain(self.binary(*o, l, r, &e.ty)?),
            EK::Construct(parts) => {
                let mut ids = Vec::with_capacity(parts.len());
                for p in parts {
                    let v = self.expr(p)?;
                    ids.push(self.plain_id(v, &p.ty)?);
                }
                if ids.len() > 65_000 {
                    return Err(e.span.error("constructor has too many components"));
                }
                let t = self.ty(&e.ty, Flavor::Plain);
                Val::plain(self.op(op::COMPOSITE_CONSTRUCT, t, &ids))
            }
            EK::Splat(x) => {
                let v = self.expr(x)?.id;
                Val::plain(self.splat(v, &e.ty))
            }
            EK::Convert(x) => Val::plain(self.convert(x, &e.ty)?),
            EK::Bitcast(x) => {
                let v = self.expr(x)?.id;
                let t = self.ty(&e.ty, Flavor::Plain);
                Val::plain(self.op(op::BITCAST, t, &[v]))
            }
            EK::Call(f, args) => {
                let mut ids = Vec::with_capacity(args.len() + 1);
                let fid = *self
                    .fn_ids
                    .get(f)
                    .ok_or_else(|| internal(e.span, "callee not emitted"))?;
                ids.push(fid);
                for a in args {
                    let v = self.expr(a)?;
                    ids.push(self.plain_id(v, &a.ty)?);
                }
                let t = self.ty(&e.ty, Flavor::Plain);
                Val::plain(self.op(op::FUNCTION_CALL, t, &ids))
            }
            EK::Builtin(b, args) => Val::plain(self.builtin(*b, args, &e.ty)?),
            EK::Texture(tc) => Val::plain(self.texture(tc, &e.ty)?),
            EK::ArrayLength { global, member } => {
                let info = self.global(*global)?;
                let t = self.scalar_ty(Scalar::U32);
                Val::plain(self.op(op::ARRAY_LENGTH, t, &[info.id, member.unwrap_or(0)]))
            }
            EK::Current => self
                .fs
                .current
                .ok_or_else(|| internal(e.span, "compound assignment value missing"))?,
        })
    }

    fn arith(&mut self, o: BinaryOp, s: Scalar, t: u32, a: u32, b: u32) -> u32 {
        let f = s.is_float();
        let opc = match o {
            BinaryOp::Add if f => op::F_ADD,
            BinaryOp::Add => op::I_ADD,
            BinaryOp::Sub if f => op::F_SUB,
            BinaryOp::Sub => op::I_SUB,
            BinaryOp::Mul if f => op::F_MUL,
            BinaryOp::Mul => op::I_MUL,
            BinaryOp::Div if f => op::F_DIV,
            BinaryOp::Div if s == Scalar::U32 => op::U_DIV,
            BinaryOp::Div => op::S_DIV,
            BinaryOp::Rem if f => op::F_REM,
            BinaryOp::Rem if s == Scalar::U32 => op::U_MOD,
            _ => op::S_REM,
        };
        self.op(opc, t, &[a, b])
    }

    fn binary(&mut self, o: BinaryOp, l: &Expr, r: &Expr, ty: &Ty) -> R<u32> {
        use BinaryOp::*;
        if matches!(o, LogicalAnd | LogicalOr) {
            return self.logical(o, l, r);
        }
        let a = self.expr(l)?.id;
        let mut b = self.expr(r)?.id;
        let t = self.ty(ty, Flavor::Plain);
        let (lt, rt) = (&l.ty, &r.ty);
        let ls = lt.scalar().unwrap_or(Scalar::F32);
        Ok(match o {
            Add | Sub | Mul | Div | Rem => match (lt, rt) {
                (Ty::Matrix(..), Ty::Matrix(..)) if o == Mul => {
                    self.op(op::MATRIX_TIMES_MATRIX, t, &[a, b])
                }
                (Ty::Matrix(c, rows), Ty::Matrix(..)) => {
                    let col_t = self.ty(&Ty::Vector(*rows, Scalar::F32), Flavor::Plain);
                    let mut cols = Vec::new();
                    for i in 0..*c as u32 {
                        let x = self.op(op::COMPOSITE_EXTRACT, col_t, &[a, i]);
                        let y = self.op(op::COMPOSITE_EXTRACT, col_t, &[b, i]);
                        cols.push(self.arith(o, Scalar::F32, col_t, x, y));
                    }
                    self.op(op::COMPOSITE_CONSTRUCT, t, &cols)
                }
                (Ty::Matrix(..), Ty::Vector(..)) => self.op(op::MATRIX_TIMES_VECTOR, t, &[a, b]),
                (Ty::Vector(..), Ty::Matrix(..)) => self.op(op::VECTOR_TIMES_MATRIX, t, &[a, b]),
                (Ty::Matrix(..), Ty::Scalar(_)) => self.op(op::MATRIX_TIMES_SCALAR, t, &[a, b]),
                (Ty::Scalar(_), Ty::Matrix(..)) => self.op(op::MATRIX_TIMES_SCALAR, t, &[b, a]),
                (Ty::Vector(_, Scalar::F32), Ty::Scalar(_)) if o == Mul => {
                    self.op(op::VECTOR_TIMES_SCALAR, t, &[a, b])
                }
                (Ty::Scalar(_), Ty::Vector(_, Scalar::F32)) if o == Mul => {
                    self.op(op::VECTOR_TIMES_SCALAR, t, &[b, a])
                }
                (Ty::Vector(..), Ty::Scalar(_)) => {
                    let bs = self.splat(b, ty);
                    self.arith(o, ls, t, a, bs)
                }
                (Ty::Scalar(_), Ty::Vector(_, s)) => {
                    let s = *s;
                    let as_ = self.splat(a, ty);
                    self.arith(o, s, t, as_, b)
                }
                _ => self.arith(o, ls, t, a, b),
            },
            Eq | Ne | Lt | Le | Gt | Ge => {
                let opc = match (o, ls) {
                    (Eq, Scalar::Bool) => op::LOGICAL_EQUAL,
                    (Ne, Scalar::Bool) => op::LOGICAL_NOT_EQUAL,
                    (Eq, Scalar::F32) => op::F_ORD_EQUAL,
                    (Ne, Scalar::F32) => op::F_UNORD_NOT_EQUAL,
                    (Lt, Scalar::F32) => op::F_ORD_LESS_THAN,
                    (Le, Scalar::F32) => op::F_ORD_LESS_THAN_EQUAL,
                    (Gt, Scalar::F32) => op::F_ORD_GREATER_THAN,
                    (Ge, Scalar::F32) => op::F_ORD_GREATER_THAN_EQUAL,
                    (Eq, _) => op::I_EQUAL,
                    (Ne, _) => op::I_NOT_EQUAL,
                    (Lt, Scalar::U32) => op::U_LESS_THAN,
                    (Le, Scalar::U32) => op::U_LESS_THAN_EQUAL,
                    (Gt, Scalar::U32) => op::U_GREATER_THAN,
                    (Ge, Scalar::U32) => op::U_GREATER_THAN_EQUAL,
                    (Lt, _) => op::S_LESS_THAN,
                    (Le, _) => op::S_LESS_THAN_EQUAL,
                    (Gt, _) => op::S_GREATER_THAN,
                    _ => op::S_GREATER_THAN_EQUAL,
                };
                self.op(opc, t, &[a, b])
            }
            And | Or | Xor => {
                let opc = match (o, ls == Scalar::Bool) {
                    (And, true) => op::LOGICAL_AND,
                    (Or, true) => op::LOGICAL_OR,
                    (And, false) => op::BITWISE_AND,
                    (Or, false) => op::BITWISE_OR,
                    _ => op::BITWISE_XOR,
                };
                self.op(opc, t, &[a, b])
            }
            Shl | Shr => {
                if r.as_const().is_none() {
                    // WGSL masks the shift amount to the bit width; SPIR-V leaves it undefined.
                    let mask = self.const_splat(rt, ConstValue::U32(31))?;
                    let rtid = self.ty(rt, Flavor::Plain);
                    b = self.op(op::BITWISE_AND, rtid, &[b, mask]);
                }
                let opc = match (o, ls) {
                    (Shl, _) => op::SHIFT_LEFT_LOGICAL,
                    (_, Scalar::U32) => op::SHIFT_RIGHT_LOGICAL,
                    _ => op::SHIFT_RIGHT_ARITHMETIC,
                };
                self.op(opc, t, &[a, b])
            }
            LogicalAnd | LogicalOr => unreachable!(),
        })
    }

    /// `&&` / `||`: `OpLogicalAnd/Or` when the right operand is side-effect free and safe to
    /// evaluate unconditionally; otherwise real short-circuit control flow with an `OpPhi`.
    fn logical(&mut self, o: BinaryOp, l: &Expr, r: &Expr) -> R<u32> {
        let bool_t = self.scalar_ty(Scalar::Bool);
        let a = self.expr(l)?.id;
        if is_simple(self.prog, r) {
            let b = self.expr(r)?.id;
            let opc = if o == BinaryOp::LogicalAnd {
                op::LOGICAL_AND
            } else {
                op::LOGICAL_OR
            };
            return Ok(self.op(opc, bool_t, &[a, b]));
        }
        let start = self.fs.label;
        let rhs_l = self.id();
        let merge = self.id();
        self.emit(op::SELECTION_MERGE, &[merge, 0]);
        if o == BinaryOp::LogicalAnd {
            self.emit(op::BRANCH_CONDITIONAL, &[a, rhs_l, merge]);
        } else {
            self.emit(op::BRANCH_CONDITIONAL, &[a, merge, rhs_l]);
        }
        self.start_block(rhs_l);
        let b = self.expr(r)?.id;
        let rhs_end = self.fs.label;
        self.branch(merge);
        self.start_block(merge);
        let short = self.const_bool(o == BinaryOp::LogicalOr);
        Ok(self.op(op::PHI, bool_t, &[short, start, b, rhs_end]))
    }

    fn convert(&mut self, x: &Expr, to: &Ty) -> R<u32> {
        let v = self.expr(x)?.id;
        let from = x.ty.scalar().unwrap_or(Scalar::F32).concretize();
        let ts = to.scalar().unwrap_or(Scalar::F32);
        let t = self.ty(to, Flavor::Plain);
        Ok(match (from, ts) {
            (a, b) if a == b => v,
            (Scalar::F32, Scalar::I32) => self.op(op::CONVERT_F_TO_S, t, &[v]),
            (Scalar::F32, Scalar::U32) => self.op(op::CONVERT_F_TO_U, t, &[v]),
            (Scalar::I32, Scalar::F32) => self.op(op::CONVERT_S_TO_F, t, &[v]),
            (Scalar::U32, Scalar::F32) => self.op(op::CONVERT_U_TO_F, t, &[v]),
            (Scalar::I32, Scalar::U32) | (Scalar::U32, Scalar::I32) => {
                self.op(op::BITCAST, t, &[v])
            }
            (Scalar::Bool, s) => {
                let (one, zero) = match s {
                    Scalar::F32 => (ConstValue::F32(1.0), ConstValue::F32(0.0)),
                    Scalar::I32 => (ConstValue::I32(1), ConstValue::I32(0)),
                    _ => (ConstValue::U32(1), ConstValue::U32(0)),
                };
                let one = self.const_splat(to, one)?;
                let zero = self.const_splat(to, zero)?;
                self.op(op::SELECT, t, &[v, one, zero])
            }
            (s, Scalar::Bool) => {
                let zero = self.const_splat(&x.ty.concretize(), ConstValue::zero_scalar(s))?;
                let opc = if s == Scalar::F32 {
                    op::F_UNORD_NOT_EQUAL
                } else {
                    op::I_NOT_EQUAL
                };
                self.op(opc, t, &[v, zero])
            }
            _ => return Err(internal(x.span, "unsupported conversion")),
        })
    }

    fn builtin(&mut self, b: B, args: &[Expr], ty: &Ty) -> R<u32> {
        if matches!(b, B::WorkgroupBarrier | B::StorageBarrier) {
            let exec = self.const_u32(spirv::scope::WORKGROUP);
            let (mem, sem) = if b == B::WorkgroupBarrier {
                (
                    spirv::scope::WORKGROUP,
                    spirv::semantics::ACQUIRE_RELEASE | spirv::semantics::WORKGROUP_MEMORY,
                )
            } else {
                (
                    spirv::scope::DEVICE,
                    spirv::semantics::ACQUIRE_RELEASE | spirv::semantics::UNIFORM_MEMORY,
                )
            };
            let mem = self.const_u32(mem);
            let sem = self.const_u32(sem);
            self.emit(op::CONTROL_BARRIER, &[exec, mem, sem]);
            return Ok(0);
        }
        let mut v = Vec::with_capacity(args.len());
        for a in args {
            v.push(self.expr(a)?.id);
        }
        let t = self.ty(ty, Flavor::Plain);
        let s = args
            .first()
            .and_then(|a| a.ty.scalar())
            .unwrap_or(Scalar::F32);
        let by_kind = |f: u32, si: u32, u: u32| match s {
            Scalar::U32 => u,
            Scalar::I32 | Scalar::AbstractInt => si,
            _ => f,
        };
        Ok(match b {
            B::Abs if s == Scalar::U32 => v[0],
            B::Abs => self.ext(t, by_kind(glsl::F_ABS, glsl::S_ABS, 0), &v),
            B::Sign => self.ext(t, by_kind(glsl::F_SIGN, glsl::S_SIGN, 0), &v),
            B::Min => self.ext(t, by_kind(glsl::F_MIN, glsl::S_MIN, glsl::U_MIN), &v),
            B::Max => self.ext(t, by_kind(glsl::F_MAX, glsl::S_MAX, glsl::U_MAX), &v),
            B::Clamp => self.ext(t, by_kind(glsl::F_CLAMP, glsl::S_CLAMP, glsl::U_CLAMP), &v),
            B::Acos => self.ext(t, glsl::ACOS, &v),
            B::Acosh => self.ext(t, glsl::ACOSH, &v),
            B::Asin => self.ext(t, glsl::ASIN, &v),
            B::Asinh => self.ext(t, glsl::ASINH, &v),
            B::Atan => self.ext(t, glsl::ATAN, &v),
            B::Atanh => self.ext(t, glsl::ATANH, &v),
            B::Atan2 => self.ext(t, glsl::ATAN2, &v),
            B::Ceil => self.ext(t, glsl::CEIL, &v),
            B::Cos => self.ext(t, glsl::COS, &v),
            B::Cosh => self.ext(t, glsl::COSH, &v),
            B::Degrees => self.ext(t, glsl::DEGREES, &v),
            B::Exp => self.ext(t, glsl::EXP, &v),
            B::Exp2 => self.ext(t, glsl::EXP2, &v),
            B::Floor => self.ext(t, glsl::FLOOR, &v),
            B::Fract => self.ext(t, glsl::FRACT, &v),
            B::InverseSqrt => self.ext(t, glsl::INVERSE_SQRT, &v),
            B::Log => self.ext(t, glsl::LOG, &v),
            B::Log2 => self.ext(t, glsl::LOG2, &v),
            B::Radians => self.ext(t, glsl::RADIANS, &v),
            B::Round => self.ext(t, glsl::ROUND_EVEN, &v),
            B::Sin => self.ext(t, glsl::SIN, &v),
            B::Sinh => self.ext(t, glsl::SINH, &v),
            B::Sqrt => self.ext(t, glsl::SQRT, &v),
            B::Tan => self.ext(t, glsl::TAN, &v),
            B::Tanh => self.ext(t, glsl::TANH, &v),
            B::Trunc => self.ext(t, glsl::TRUNC, &v),
            B::Pow => self.ext(t, glsl::POW, &v),
            B::Step => self.ext(t, glsl::STEP, &v),
            B::Smoothstep => self.ext(t, glsl::SMOOTH_STEP, &v),
            B::Fma => self.ext(t, glsl::FMA, &v),
            B::Saturate => {
                let zero = self.const_splat(ty, ConstValue::F32(0.0))?;
                let one = self.const_splat(ty, ConstValue::F32(1.0))?;
                self.ext(t, glsl::F_CLAMP, &[v[0], zero, one])
            }
            B::Mix => {
                let mut w = v.clone();
                if ty.vec_size().is_some() && args[2].ty.is_scalar() {
                    w[2] = self.splat(v[2], ty);
                }
                self.ext(t, glsl::F_MIX, &w)
            }
            B::Dot => {
                if s.is_float() {
                    self.op(op::DOT, t, &v)
                } else {
                    let vt = self.ty(&args[0].ty, Flavor::Plain);
                    let prod = self.op(op::I_MUL, vt, &v);
                    let n = args[0].ty.vec_size().unwrap_or(1) as u32;
                    let mut acc = self.op(op::COMPOSITE_EXTRACT, t, &[prod, 0]);
                    for i in 1..n {
                        let x = self.op(op::COMPOSITE_EXTRACT, t, &[prod, i]);
                        acc = self.op(op::I_ADD, t, &[acc, x]);
                    }
                    acc
                }
            }
            B::Cross => self.ext(t, glsl::CROSS, &v),
            B::Length => self.ext(t, glsl::LENGTH, &v),
            B::Distance => self.ext(t, glsl::DISTANCE, &v),
            B::Normalize => self.ext(t, glsl::NORMALIZE, &v),
            B::FaceForward => self.ext(t, glsl::FACE_FORWARD, &v),
            B::Reflect => self.ext(t, glsl::REFLECT, &v),
            B::Refract => self.ext(t, glsl::REFRACT, &v),
            B::Determinant => self.ext(t, glsl::DETERMINANT, &v),
            B::Transpose => self.op(op::TRANSPOSE, t, &v),
            B::Ldexp => self.ext(t, glsl::LDEXP, &v),
            B::Select => {
                let mut c = v[2];
                if let (Some(n), true) = (ty.vec_size(), args[2].ty.is_scalar()) {
                    c = self.splat(c, &Ty::Vector(n, Scalar::Bool));
                }
                self.op(op::SELECT, t, &[c, v[1], v[0]])
            }
            B::All | B::Any if args[0].ty.is_scalar() => v[0],
            B::All => self.op(op::ALL, t, &v),
            B::Any => self.op(op::ANY, t, &v),
            B::Dpdx => self.op(op::DPDX, t, &v),
            B::Dpdy => self.op(op::DPDY, t, &v),
            B::Fwidth => self.op(op::FWIDTH, t, &v),
            B::CountOneBits => self.op(op::BIT_COUNT, t, &v),
            B::ReverseBits => self.op(op::BIT_REVERSE, t, &v),
            B::Pack4x8Snorm => self.ext(t, glsl::PACK_SNORM_4X8, &v),
            B::Pack4x8Unorm => self.ext(t, glsl::PACK_UNORM_4X8, &v),
            B::Pack2x16Snorm => self.ext(t, glsl::PACK_SNORM_2X16, &v),
            B::Pack2x16Unorm => self.ext(t, glsl::PACK_UNORM_2X16, &v),
            B::Pack2x16Float => self.ext(t, glsl::PACK_HALF_2X16, &v),
            B::Unpack4x8Snorm => self.ext(t, glsl::UNPACK_SNORM_4X8, &v),
            B::Unpack4x8Unorm => self.ext(t, glsl::UNPACK_UNORM_4X8, &v),
            B::Unpack2x16Snorm => self.ext(t, glsl::UNPACK_SNORM_2X16, &v),
            B::Unpack2x16Unorm => self.ext(t, glsl::UNPACK_UNORM_2X16, &v),
            B::Unpack2x16Float => self.ext(t, glsl::UNPACK_HALF_2X16, &v),
            _ => return Err(internal(self.span, "unexpected built-in")),
        })
    }

    fn texture(&mut self, tc: &TexCall, ty: &Ty) -> R<u32> {
        use spirv::image_operands as io;
        // Evaluate every operand first, in WGSL argument order: operands may contain control
        // flow (short-circuit `&&`/`||`), while `OpSampledImage` must sit in the same block as
        // the instruction that consumes it.
        let mut eval = |e: &Option<Expr>| -> R<Option<u32>> {
            Ok(match e {
                Some(e) => Some(self.expr(e)?.id),
                None => None,
            })
        };
        let coords = eval(&tc.coords)?;
        let array_index = eval(&tc.array_index)?;
        let level = eval(&tc.level)?;
        let bias = eval(&tc.bias)?;
        let ddx = eval(&tc.ddx)?;
        let ddy = eval(&tc.ddy)?;
        let depth_ref = eval(&tc.depth_ref)?;
        let offset = eval(&tc.offset)?;
        let value = eval(&tc.value)?;
        let span = self.span;
        let missing = move |what: &str| internal(span, &format!("texture call without {what}"));

        let prog = self.prog;
        let tg = &prog.globals[tc.texture];
        let tinfo = self.global(tc.texture)?;
        let img_t = self.ty(&tg.ty, Flavor::Plain);
        let f32_t = self.scalar_ty(Scalar::F32);
        let u32_t = self.scalar_ty(Scalar::U32);
        let ai_unsigned = tc.array_index.as_ref().is_some_and(|a| a.ty == U32);
        match tc.op {
            B::TextureSample
            | B::TextureSampleBias
            | B::TextureSampleLevel
            | B::TextureSampleGrad
            | B::TextureSampleCompare
            | B::TextureSampleCompareLevel => {
                let Ty::Texture(_, sample) = tg.ty else {
                    return Err(internal(self.span, "sampling a non-texture"));
                };
                let depth = sample == SampleType::Depth;
                let mut coords = coords.ok_or_else(|| missing("coords"))?;
                if let Some(a) = array_index {
                    let conv = if ai_unsigned {
                        op::CONVERT_U_TO_F
                    } else {
                        op::CONVERT_S_TO_F
                    };
                    let af = self.op(conv, f32_t, &[a]);
                    let v3 = self.ty(&Ty::Vector(3, Scalar::F32), Flavor::Plain);
                    coords = self.op(op::COMPOSITE_CONSTRUCT, v3, &[coords, af]);
                }
                let compare = matches!(
                    tc.op,
                    B::TextureSampleCompare | B::TextureSampleCompareLevel
                );
                let mut mask = 0;
                let mut extra = Vec::new();
                let opc = match tc.op {
                    B::TextureSample => op::IMAGE_SAMPLE_IMPLICIT_LOD,
                    B::TextureSampleBias => {
                        mask |= io::BIAS;
                        extra.push(bias.ok_or_else(|| missing("bias"))?);
                        op::IMAGE_SAMPLE_IMPLICIT_LOD
                    }
                    B::TextureSampleLevel => {
                        mask |= io::LOD;
                        let mut lod = level.ok_or_else(|| missing("level"))?;
                        let lt = tc.level.as_ref().map(|l| l.ty.clone()).unwrap_or(F32);
                        if lt != F32 {
                            let conv = if lt == U32 {
                                op::CONVERT_U_TO_F
                            } else {
                                op::CONVERT_S_TO_F
                            };
                            lod = self.op(conv, f32_t, &[lod]);
                        }
                        extra.push(lod);
                        op::IMAGE_SAMPLE_EXPLICIT_LOD
                    }
                    B::TextureSampleGrad => {
                        mask |= io::GRAD;
                        extra.push(ddx.ok_or_else(|| missing("ddx"))?);
                        extra.push(ddy.ok_or_else(|| missing("ddy"))?);
                        op::IMAGE_SAMPLE_EXPLICIT_LOD
                    }
                    B::TextureSampleCompare => op::IMAGE_SAMPLE_DREF_IMPLICIT_LOD,
                    _ => {
                        mask |= io::LOD;
                        extra.push(self.const_f32(0.0));
                        op::IMAGE_SAMPLE_DREF_EXPLICIT_LOD
                    }
                };
                if let Some(o) = offset {
                    mask |= io::CONST_OFFSET;
                    extra.push(o);
                }
                let sg = tc.sampler.ok_or_else(|| missing("sampler"))?;
                let sinfo = self.global(sg)?;
                let smp_t = self.ty(&prog.globals[sg].ty, Flavor::Plain);
                let si_t = self.sampled_image_ty(img_t);
                let rt = if compare {
                    f32_t
                } else {
                    self.ty(&Ty::Vector(4, Scalar::F32), Flavor::Plain)
                };
                let img = self.op(op::LOAD, img_t, &[tinfo.id]);
                let smp = self.op(op::LOAD, smp_t, &[sinfo.id]);
                let si = self.op(op::SAMPLED_IMAGE, si_t, &[img, smp]);
                let mut w = vec![si, coords];
                if compare {
                    w.push(depth_ref.ok_or_else(|| missing("depth_ref"))?);
                }
                if mask != 0 {
                    w.push(mask);
                    w.extend(extra);
                }
                let r = self.op(opc, rt, &w);
                if depth && !compare {
                    Ok(self.op(op::COMPOSITE_EXTRACT, f32_t, &[r, 0]))
                } else {
                    Ok(r)
                }
            }
            B::TextureLoad => {
                let Ty::Texture(_, sample) = tg.ty else {
                    return Err(internal(self.span, "loading a non-texture"));
                };
                let ce = tc.coords.as_ref().ok_or_else(|| missing("coords"))?;
                let mut coords = coords.ok_or_else(|| missing("coords"))?;
                if let (Some(mut a), Some(ai)) = (array_index, &tc.array_index) {
                    let cs = ce.ty.scalar().unwrap_or(Scalar::I32);
                    if ai.ty.scalar() != Some(cs) {
                        let st = self.scalar_ty(cs);
                        a = self.op(op::BITCAST, st, &[a]);
                    }
                    let v3 = self.ty(&Ty::Vector(3, cs), Flavor::Plain);
                    coords = self.op(op::COMPOSITE_CONSTRUCT, v3, &[coords, a]);
                }
                let lod = level.ok_or_else(|| missing("level"))?;
                let comp = match sample {
                    SampleType::Sint => Scalar::I32,
                    SampleType::Uint => Scalar::U32,
                    _ => Scalar::F32,
                };
                let rt = self.ty(&Ty::Vector(4, comp), Flavor::Plain);
                let img = self.op(op::LOAD, img_t, &[tinfo.id]);
                let r = self.op(op::IMAGE_FETCH, rt, &[img, coords, io::LOD, lod]);
                if sample == SampleType::Depth {
                    Ok(self.op(op::COMPOSITE_EXTRACT, f32_t, &[r, 0]))
                } else {
                    Ok(r)
                }
            }
            B::TextureStore => {
                let coords = coords.ok_or_else(|| missing("coords"))?;
                let value = value.ok_or_else(|| missing("value"))?;
                let img = self.op(op::LOAD, img_t, &[tinfo.id]);
                self.emit(op::IMAGE_WRITE, &[img, coords, value]);
                Ok(0)
            }
            B::TextureDimensions => {
                self.caps.insert(spirv::capability::IMAGE_QUERY);
                let v2 = self.ty(ty, Flavor::Plain);
                match tg.ty {
                    Ty::Texture(view, _) => {
                        let lod = match level {
                            Some(l) => l,
                            None => self.const_u32(0),
                        };
                        let img = self.op(op::LOAD, img_t, &[tinfo.id]);
                        if view == TextureView::D2Array {
                            let v3 = self.ty(&Ty::Vector(3, Scalar::U32), Flavor::Plain);
                            let q = self.op(op::IMAGE_QUERY_SIZE_LOD, v3, &[img, lod]);
                            Ok(self.op(op::VECTOR_SHUFFLE, v2, &[q, q, 0, 1]))
                        } else {
                            Ok(self.op(op::IMAGE_QUERY_SIZE_LOD, v2, &[img, lod]))
                        }
                    }
                    _ => {
                        let img = self.op(op::LOAD, img_t, &[tinfo.id]);
                        Ok(self.op(op::IMAGE_QUERY_SIZE, v2, &[img]))
                    }
                }
            }
            B::TextureNumLevels => {
                self.caps.insert(spirv::capability::IMAGE_QUERY);
                let img = self.op(op::LOAD, img_t, &[tinfo.id]);
                Ok(self.op(op::IMAGE_QUERY_LEVELS, u32_t, &[img]))
            }
            B::TextureNumLayers => {
                self.caps.insert(spirv::capability::IMAGE_QUERY);
                let zero = self.const_u32(0);
                let v3 = self.ty(&Ty::Vector(3, Scalar::U32), Flavor::Plain);
                let img = self.op(op::LOAD, img_t, &[tinfo.id]);
                let q = self.op(op::IMAGE_QUERY_SIZE_LOD, v3, &[img, zero]);
                Ok(self.op(op::COMPOSITE_EXTRACT, u32_t, &[q, 2]))
            }
            _ => Err(internal(self.span, "unexpected texture built-in")),
        }
    }

    // ------------------------------------------------------------------ entry point

    #[allow(clippy::too_many_arguments)]
    fn io_var(
        &mut self,
        stage: Stage,
        input: bool,
        ty: &Ty,
        attrs: &IoAttrs,
        name: &str,
        io: &mut EntryIo,
        builtin_inputs: &mut HashMap<BuiltinValue, u32>,
    ) -> u32 {
        if input
            && let Some(b) = attrs.builtin
            && let Some(&v) = builtin_inputs.get(&b)
        {
            return v;
        }
        let class = if input { sc::INPUT } else { sc::OUTPUT };
        let t = self.ty(ty, Flavor::Plain);
        let p = self.ptr(class, t);
        let id = self.id();
        inst(&mut self.s.globals, op::VARIABLE, &[p, id, class]);
        self.name(id, name);
        io.interface.push(id);
        if let Some(b) = attrs.builtin {
            use spirv::builtin as sb;
            let code = match b {
                BuiltinValue::VertexIndex => sb::VERTEX_INDEX,
                BuiltinValue::InstanceIndex => sb::INSTANCE_INDEX,
                BuiltinValue::Position if stage == Stage::Fragment => sb::FRAG_COORD,
                BuiltinValue::Position => sb::POSITION,
                BuiltinValue::FrontFacing => sb::FRONT_FACING,
                BuiltinValue::FragDepth => sb::FRAG_DEPTH,
                BuiltinValue::SampleIndex => sb::SAMPLE_ID,
                BuiltinValue::LocalInvocationId => sb::LOCAL_INVOCATION_ID,
                BuiltinValue::LocalInvocationIndex => sb::LOCAL_INVOCATION_INDEX,
                BuiltinValue::GlobalInvocationId => sb::GLOBAL_INVOCATION_ID,
                BuiltinValue::WorkgroupId => sb::WORKGROUP_ID,
                BuiltinValue::NumWorkgroups => sb::NUM_WORKGROUPS,
            };
            self.decorate(id, &[deco::BUILT_IN, code]);
            if b == BuiltinValue::FragDepth {
                io.depth_replacing = true;
            }
            if b == BuiltinValue::SampleIndex {
                self.caps.insert(spirv::capability::SAMPLE_RATE_SHADING);
                self.decorate(id, &[deco::FLAT]);
            }
            if attrs.invariant {
                self.decorate(id, &[deco::INVARIANT]);
            }
            if input {
                builtin_inputs.insert(b, id);
            }
        }
        if let Some(loc) = attrs.location {
            self.decorate(id, &[deco::LOCATION, loc]);
            let varying = (stage == Stage::Vertex && !input) || (stage == Stage::Fragment && input);
            if varying {
                let int = ty.scalar().is_some_and(|s| s.is_int());
                if int || attrs.interpolation == Some(Interpolation::Flat) {
                    self.decorate(id, &[deco::FLAT]);
                } else {
                    if attrs.interpolation == Some(Interpolation::Linear) {
                        self.decorate(id, &[deco::NO_PERSPECTIVE]);
                    }
                    match attrs.sampling {
                        Some(Sampling::Centroid) => self.decorate(id, &[deco::CENTROID]),
                        Some(Sampling::Sample) => {
                            self.caps.insert(spirv::capability::SAMPLE_RATE_SHADING);
                            self.decorate(id, &[deco::SAMPLE]);
                        }
                        _ => {}
                    }
                }
            }
            if let Some(it) = io_type(ty) {
                let entry = Interface {
                    location: loc,
                    ty: it,
                };
                if input {
                    io.inputs.push(entry);
                } else {
                    io.outputs.push(entry);
                }
            }
        }
        id
    }

    pub(super) fn entry_wrapper(&mut self, fid: FnId, stage: Stage) -> R<EntryIo> {
        let prog = self.prog;
        let f = &prog.functions[fid];
        self.fs = FnState::default();
        let void = self.void_ty();
        let fnty = self.fn_ty(void, &[]);
        let main = self.id();
        self.name(main, "main");
        let entry_label = self.id();
        self.fs.label = entry_label;
        let mut io = EntryIo {
            main,
            interface: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            depth_replacing: false,
        };
        let mut builtin_inputs = HashMap::new();
        let mut args = vec![self.fn_ids[&fid]];
        for p in &f.params {
            if !p.io.is_empty() {
                let var = self.io_var(
                    stage,
                    true,
                    &p.ty,
                    &p.io,
                    &p.name,
                    &mut io,
                    &mut builtin_inputs,
                );
                let t = self.ty(&p.ty, Flavor::Plain);
                args.push(self.op(op::LOAD, t, &[var]));
            } else if let Ty::Struct(sid) = p.ty {
                let mut parts = Vec::new();
                for m in &prog.structs[sid].members {
                    let name = format!("{}.{}", p.name, m.name);
                    let var = self.io_var(
                        stage,
                        true,
                        &m.ty,
                        &m.io,
                        &name,
                        &mut io,
                        &mut builtin_inputs,
                    );
                    let t = self.ty(&m.ty, Flavor::Plain);
                    parts.push(self.op(op::LOAD, t, &[var]));
                }
                let st = self.ty(&p.ty, Flavor::Plain);
                args.push(self.op(op::COMPOSITE_CONSTRUCT, st, &parts));
            }
        }
        let rt = match &f.ret {
            Some(t) => self.ty(t, Flavor::Plain),
            None => void,
        };
        let r = self.op(op::FUNCTION_CALL, rt, &args);
        if let Some(ty) = &f.ret {
            if !f.ret_io.is_empty() {
                let name = format!("{}_out", f.name);
                let var = self.io_var(
                    stage,
                    false,
                    ty,
                    &f.ret_io,
                    &name,
                    &mut io,
                    &mut builtin_inputs,
                );
                self.emit(op::STORE, &[var, r]);
            } else if let Ty::Struct(sid) = ty {
                for (i, m) in prog.structs[*sid].members.iter().enumerate() {
                    let t = self.ty(&m.ty, Flavor::Plain);
                    let x = self.op(op::COMPOSITE_EXTRACT, t, &[r, i as u32]);
                    let name = format!("out.{}", m.name);
                    let var = self.io_var(
                        stage,
                        false,
                        &m.ty,
                        &m.io,
                        &name,
                        &mut io,
                        &mut builtin_inputs,
                    );
                    self.emit(op::STORE, &[var, x]);
                }
            }
        }
        self.emit(op::RETURN, &[]);
        let fs = std::mem::take(&mut self.fs);
        inst(&mut self.s.functions, op::FUNCTION, &[void, main, 0, fnty]);
        inst(&mut self.s.functions, op::LABEL, &[entry_label]);
        self.s.functions.extend(fs.body);
        inst(&mut self.s.functions, op::FUNCTION_END, &[]);
        io.inputs.sort_by_key(|i| i.location);
        io.outputs.sort_by_key(|i| i.location);
        Ok(io)
    }
}
