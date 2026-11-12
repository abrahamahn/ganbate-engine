//! Type checking of WGSL built-in function calls (math, packing, barriers and textures).

use crate::ast::{self, ExprKind as AK};
use crate::check::{Checked, Checker, R, konst};
use crate::consteval;
use crate::ir::{
    Builtin as B, Expr, ExprKind as EK, GlobalKind, PlaceRoot, StageReq, Step, TexCall,
};
use crate::lexer::Span;
use crate::types::*;
use crate::{SampleType, TextureView};

fn lookup(name: &str) -> Option<B> {
    Some(match name {
        "abs" => B::Abs,
        "acos" => B::Acos,
        "acosh" => B::Acosh,
        "asin" => B::Asin,
        "asinh" => B::Asinh,
        "atan" => B::Atan,
        "atanh" => B::Atanh,
        "atan2" => B::Atan2,
        "ceil" => B::Ceil,
        "clamp" => B::Clamp,
        "cos" => B::Cos,
        "cosh" => B::Cosh,
        "countOneBits" => B::CountOneBits,
        "cross" => B::Cross,
        "degrees" => B::Degrees,
        "determinant" => B::Determinant,
        "distance" => B::Distance,
        "dot" => B::Dot,
        "dpdx" => B::Dpdx,
        "dpdy" => B::Dpdy,
        "exp" => B::Exp,
        "exp2" => B::Exp2,
        "faceForward" => B::FaceForward,
        "floor" => B::Floor,
        "fma" => B::Fma,
        "fract" => B::Fract,
        "fwidth" => B::Fwidth,
        "inverseSqrt" => B::InverseSqrt,
        "ldexp" => B::Ldexp,
        "length" => B::Length,
        "log" => B::Log,
        "log2" => B::Log2,
        "max" => B::Max,
        "min" => B::Min,
        "mix" => B::Mix,
        "normalize" => B::Normalize,
        "pow" => B::Pow,
        "radians" => B::Radians,
        "reflect" => B::Reflect,
        "refract" => B::Refract,
        "reverseBits" => B::ReverseBits,
        "round" => B::Round,
        "saturate" => B::Saturate,
        "sign" => B::Sign,
        "sin" => B::Sin,
        "sinh" => B::Sinh,
        "smoothstep" => B::Smoothstep,
        "sqrt" => B::Sqrt,
        "step" => B::Step,
        "tan" => B::Tan,
        "tanh" => B::Tanh,
        "transpose" => B::Transpose,
        "trunc" => B::Trunc,
        "select" => B::Select,
        "all" => B::All,
        "any" => B::Any,
        "pack4x8snorm" => B::Pack4x8Snorm,
        "pack4x8unorm" => B::Pack4x8Unorm,
        "pack2x16snorm" => B::Pack2x16Snorm,
        "pack2x16unorm" => B::Pack2x16Unorm,
        "pack2x16float" => B::Pack2x16Float,
        "unpack4x8snorm" => B::Unpack4x8Snorm,
        "unpack4x8unorm" => B::Unpack4x8Unorm,
        "unpack2x16snorm" => B::Unpack2x16Snorm,
        "unpack2x16unorm" => B::Unpack2x16Unorm,
        "unpack2x16float" => B::Unpack2x16Float,
        "workgroupBarrier" => B::WorkgroupBarrier,
        "storageBarrier" => B::StorageBarrier,
        "textureSample" => B::TextureSample,
        "textureSampleBias" => B::TextureSampleBias,
        "textureSampleLevel" => B::TextureSampleLevel,
        "textureSampleGrad" => B::TextureSampleGrad,
        "textureSampleCompare" => B::TextureSampleCompare,
        "textureSampleCompareLevel" => B::TextureSampleCompareLevel,
        "textureLoad" => B::TextureLoad,
        "textureStore" => B::TextureStore,
        "textureDimensions" => B::TextureDimensions,
        "textureNumLevels" => B::TextureNumLevels,
        "textureNumLayers" => B::TextureNumLayers,
        _ => return None,
    })
}

const UNSUPPORTED: &[&str] = &[
    "modf",
    "frexp",
    "quantizeToF16",
    "extractBits",
    "insertBits",
    "firstLeadingBit",
    "firstTrailingBit",
    "countLeadingZeros",
    "countTrailingZeros",
    "dot4I8Packed",
    "dot4U8Packed",
    "pack4xI8",
    "pack4xU8",
    "unpack4xI8",
    "unpack4xU8",
    "dpdxCoarse",
    "dpdxFine",
    "dpdyCoarse",
    "dpdyFine",
    "fwidthCoarse",
    "fwidthFine",
    "textureGather",
    "textureGatherCompare",
    "textureSampleBaseClampToEdge",
    "textureNumSamples",
    "atomicLoad",
    "atomicStore",
    "atomicAdd",
    "atomicSub",
    "atomicMax",
    "atomicMin",
    "atomicAnd",
    "atomicOr",
    "atomicXor",
    "atomicExchange",
    "atomicCompareExchangeWeak",
    "workgroupUniformLoad",
    "textureBarrier",
];

/// Built-ins whose constant arguments are folded at compile time (see `consteval::builtin`).
fn foldable(b: B) -> bool {
    matches!(
        b,
        B::Abs
            | B::Sign
            | B::Acos
            | B::Acosh
            | B::Asin
            | B::Asinh
            | B::Atan
            | B::Atanh
            | B::Ceil
            | B::Cos
            | B::Cosh
            | B::Degrees
            | B::Exp
            | B::Exp2
            | B::Floor
            | B::Fract
            | B::InverseSqrt
            | B::Log
            | B::Log2
            | B::Radians
            | B::Round
            | B::Saturate
            | B::Sin
            | B::Sinh
            | B::Sqrt
            | B::Tan
            | B::Tanh
            | B::Trunc
            | B::Atan2
            | B::Pow
            | B::Step
            | B::Max
            | B::Min
            | B::Clamp
            | B::Fma
            | B::Mix
            | B::Smoothstep
            | B::Dot
            | B::Length
            | B::Distance
            | B::Normalize
            | B::Cross
            | B::Select
            | B::All
            | B::Any
    )
}

fn is_float_ty(t: &Ty) -> bool {
    t.is_scalar_or_vector() && t.scalar().is_some_and(|s| s.is_float())
}

fn is_float_vec(t: &Ty) -> bool {
    matches!(t, Ty::Vector(_, s) if s.is_float())
}

impl Checker<'_> {
    fn bad_args(&self, name: &str, args: &[Expr], span: Span) -> crate::Error {
        let tys: Vec<String> = args.iter().map(|a| self.ty_name(&a.ty)).collect();
        span.error(format!(
            "no overload of '{name}' accepts ({})",
            tys.join(", ")
        ))
    }

    /// Convert all arguments to one common type: the first concrete argument type, else the
    /// abstract type (AbstractFloat when `float` or when any argument is a float).
    fn unify_args(&self, name: &str, args: Vec<Expr>, float: bool, span: Span) -> R<Vec<Expr>> {
        let target = if let Some(c) = args.iter().find(|a| !a.ty.is_abstract()) {
            c.ty.clone()
        } else {
            let any_float = float
                || args
                    .iter()
                    .any(|a| a.ty.leaf_scalar() == Some(Scalar::AbstractFloat));
            args[0].ty.with_scalar(if any_float {
                Scalar::AbstractFloat
            } else {
                Scalar::AbstractInt
            })
        };
        let mut out = Vec::new();
        for a in &args {
            let ok = a.ty == target
                || (a.ty.is_abstract()
                    && a.ty.vec_size() == target.vec_size()
                    && a.ty.is_scalar_or_vector()
                    && target.is_scalar_or_vector()
                    && a.ty
                        .scalar()
                        .zip(target.scalar())
                        .is_some_and(|(f, t)| f.converts_to(t)));
            if !ok {
                return Err(self.bad_args(name, &args, span));
            }
        }
        for a in args {
            out.push(self.convert(a, &target, span)?);
        }
        Ok(out)
    }

    pub(crate) fn builtin_call(
        &mut self,
        name: &str,
        args: &[ast::Expr],
        span: Span,
        stmt: bool,
    ) -> R<Expr> {
        if name == "arrayLength" {
            return self.array_length(args, span);
        }
        let Some(b) = lookup(name) else {
            if UNSUPPORTED.contains(&name) {
                return Err(span.error(format!(
                    "built-in function '{name}' is not supported by Ganbate"
                )));
            }
            return Err(span.error(format!("unknown function '{name}'")));
        };
        if self.fcx.is_none() && !foldable(b) {
            return Err(span.error(format!("'{name}' cannot be used in a constant expression")));
        }
        if matches!(
            b,
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
                | B::TextureNumLayers
        ) {
            let e = self.texture_call(b, name, args, span)?;
            if stmt && e.ty != Ty::Void {
                return Err(span.error(format!("the result of '{name}' must be used")));
            }
            return Ok(e);
        }
        // Abstract arguments stay abstract here so that unification can convert them towards
        // the concrete arguments; `math_call` concretizes whatever cannot be folded.
        let vals: Vec<Expr> = args.iter().map(|a| self.value(a)).collect::<R<_>>()?;
        let e = self.math_call(b, name, vals, span)?;
        if stmt && e.ty != Ty::Void {
            return Err(span.error(format!(
                "the result of '{name}' must be used (assign it to '_' to discard it)"
            )));
        }
        Ok(e)
    }

    fn math_call(&mut self, b: B, name: &str, args: Vec<Expr>, span: Span) -> R<Expr> {
        let n = args.len();
        let want = |k: usize| -> R<()> {
            if n == k {
                Ok(())
            } else {
                Err(span.error(format!("'{name}' expects {k} argument(s), got {n}")))
            }
        };
        let stage = match b {
            B::Dpdx | B::Dpdy | B::Fwidth => Some(StageReq::Fragment),
            B::WorkgroupBarrier | B::StorageBarrier => Some(StageReq::Compute),
            _ => None,
        };
        if let Some(st) = stage
            && let Some(fcx) = self.fcx.as_mut()
        {
            fcx.stage_reqs.push((st, span, name.to_string()));
        }
        let (args, result): (Vec<Expr>, Ty) = match b {
            B::Acos
            | B::Acosh
            | B::Asin
            | B::Asinh
            | B::Atan
            | B::Atanh
            | B::Ceil
            | B::Cos
            | B::Cosh
            | B::Degrees
            | B::Exp
            | B::Exp2
            | B::Floor
            | B::Fract
            | B::InverseSqrt
            | B::Log
            | B::Log2
            | B::Radians
            | B::Round
            | B::Saturate
            | B::Sin
            | B::Sinh
            | B::Sqrt
            | B::Tan
            | B::Tanh
            | B::Trunc
            | B::Dpdx
            | B::Dpdy
            | B::Fwidth => {
                want(1)?;
                let a = self.unify_args(name, args, true, span)?;
                if !is_float_ty(&a[0].ty) {
                    return Err(self.bad_args(name, &a, span));
                }
                let t = a[0].ty.clone();
                (a, t)
            }
            B::Abs | B::Sign => {
                want(1)?;
                let a = self.unify_args(name, args, false, span)?;
                let t = a[0].ty.clone();
                let ok = t.is_scalar_or_vector()
                    && t.scalar()
                        .is_some_and(|s| s.is_numeric() && !(b == B::Sign && s == Scalar::U32));
                if !ok {
                    return Err(self.bad_args(name, &a, span));
                }
                (a, t)
            }
            B::Atan2 | B::Pow | B::Step => {
                want(2)?;
                let a = self.unify_args(name, args, true, span)?;
                if !is_float_ty(&a[0].ty) {
                    return Err(self.bad_args(name, &a, span));
                }
                let t = a[0].ty.clone();
                (a, t)
            }
            B::Min | B::Max => {
                want(2)?;
                let a = self.unify_args(name, args, false, span)?;
                let t = a[0].ty.clone();
                if !t.is_scalar_or_vector() || t.scalar() == Some(Scalar::Bool) {
                    return Err(self.bad_args(name, &a, span));
                }
                (a, t)
            }
            B::Clamp => {
                want(3)?;
                let a = self.unify_args(name, args, false, span)?;
                let t = a[0].ty.clone();
                if !t.is_scalar_or_vector() || t.scalar() == Some(Scalar::Bool) {
                    return Err(self.bad_args(name, &a, span));
                }
                (a, t)
            }
            B::Smoothstep | B::Fma => {
                want(3)?;
                let a = self.unify_args(name, args, true, span)?;
                if !is_float_ty(&a[0].ty) {
                    return Err(self.bad_args(name, &a, span));
                }
                let t = a[0].ty.clone();
                (a, t)
            }
            B::FaceForward => {
                want(3)?;
                let a = self.unify_args(name, args, true, span)?;
                if !is_float_vec(&a[0].ty) {
                    return Err(self.bad_args(name, &a, span));
                }
                let t = a[0].ty.clone();
                (a, t)
            }
            B::Mix => {
                want(3)?;
                let mut args = args;
                if args[0].ty.vec_size().is_some() && args[2].ty.is_scalar() {
                    let t = args.pop().expect("t");
                    let mut ab = self.unify_args(name, args, true, span)?;
                    if !is_float_vec(&ab[0].ty) {
                        ab.push(t);
                        return Err(self.bad_args(name, &ab, span));
                    }
                    let s = ab[0].ty.scalar().expect("scalar");
                    let t = self.convert(t, &Ty::Scalar(s), span)?;
                    let ty = ab[0].ty.clone();
                    ab.push(t);
                    (ab, ty)
                } else {
                    let a = self.unify_args(name, args, true, span)?;
                    if !is_float_ty(&a[0].ty) {
                        return Err(self.bad_args(name, &a, span));
                    }
                    let t = a[0].ty.clone();
                    (a, t)
                }
            }
            B::Dot => {
                want(2)?;
                let a = self.unify_args(name, args, false, span)?;
                match a[0].ty {
                    Ty::Vector(_, s) if s.is_numeric() => (a, Ty::Scalar(s)),
                    _ => return Err(self.bad_args(name, &a, span)),
                }
            }
            B::Cross => {
                want(2)?;
                let a = self.unify_args(name, args, true, span)?;
                match a[0].ty {
                    Ty::Vector(3, s) if s.is_float() => {
                        let t = a[0].ty.clone();
                        (a, t)
                    }
                    _ => return Err(self.bad_args(name, &a, span)),
                }
            }
            B::Length | B::Distance => {
                want(if b == B::Length { 1 } else { 2 })?;
                let a = self.unify_args(name, args, true, span)?;
                if !is_float_ty(&a[0].ty) {
                    return Err(self.bad_args(name, &a, span));
                }
                let s = a[0].ty.scalar().expect("scalar");
                (a, Ty::Scalar(s))
            }
            B::Normalize => {
                want(1)?;
                let a = self.unify_args(name, args, true, span)?;
                if !is_float_vec(&a[0].ty) {
                    return Err(self.bad_args(name, &a, span));
                }
                let t = a[0].ty.clone();
                (a, t)
            }
            B::Reflect => {
                want(2)?;
                let a = self.unify_args(name, args, true, span)?;
                if !is_float_vec(&a[0].ty) {
                    return Err(self.bad_args(name, &a, span));
                }
                let t = a[0].ty.clone();
                (a, t)
            }
            B::Refract => {
                want(3)?;
                let mut args = args;
                let eta = args.pop().expect("eta");
                let mut ab = self.unify_args(name, args, true, span)?;
                if !is_float_vec(&ab[0].ty) {
                    ab.push(eta);
                    return Err(self.bad_args(name, &ab, span));
                }
                let eta = self.convert(eta, &F32, span)?;
                let t = ab[0].ty.clone();
                ab.push(eta);
                (ab, t)
            }
            B::Determinant => {
                want(1)?;
                match args[0].ty {
                    Ty::Matrix(c, r) if c == r => (args, F32),
                    _ => return Err(self.bad_args(name, &args, span)),
                }
            }
            B::Transpose => {
                want(1)?;
                match args[0].ty {
                    Ty::Matrix(c, r) => (args, Ty::Matrix(r, c)),
                    _ => return Err(self.bad_args(name, &args, span)),
                }
            }
            B::Select => {
                want(3)?;
                let mut args = args;
                let cond = args.pop().expect("cond");
                let ft = self.unify_args(name, args, false, span)?;
                let t = ft[0].ty.clone();
                let ok = t.is_scalar_or_vector()
                    && match &cond.ty {
                        Ty::Scalar(Scalar::Bool) => true,
                        Ty::Vector(n, Scalar::Bool) => t.vec_size() == Some(*n),
                        _ => false,
                    };
                let mut all = ft;
                all.push(cond);
                if !ok {
                    return Err(self.bad_args(name, &all, span));
                }
                (all, t)
            }
            B::All | B::Any => {
                want(1)?;
                if !matches!(
                    args[0].ty,
                    Ty::Scalar(Scalar::Bool) | Ty::Vector(_, Scalar::Bool)
                ) {
                    return Err(self.bad_args(name, &args, span));
                }
                (args, BOOL)
            }
            B::CountOneBits | B::ReverseBits => {
                want(1)?;
                let t = args[0].ty.clone();
                if !t.is_scalar_or_vector()
                    || !matches!(t.scalar(), Some(Scalar::I32 | Scalar::U32))
                {
                    return Err(self.bad_args(name, &args, span));
                }
                (args, t)
            }
            B::Ldexp => {
                want(2)?;
                let mut args = args;
                let e = args.pop().expect("exp");
                let f = args.pop().expect("frac");
                if !is_float_ty(&f.ty) {
                    return Err(self.bad_args(name, &[f, e], span));
                }
                let et = f.ty.with_scalar(Scalar::I32);
                let e = self.convert(e, &et, span)?;
                let t = f.ty.clone();
                (vec![f, e], t)
            }
            B::Pack4x8Snorm
            | B::Pack4x8Unorm
            | B::Pack2x16Snorm
            | B::Pack2x16Unorm
            | B::Pack2x16Float => {
                want(1)?;
                let n = if matches!(b, B::Pack4x8Snorm | B::Pack4x8Unorm) {
                    4
                } else {
                    2
                };
                let a = self.convert(
                    args.into_iter().next().expect("arg"),
                    &Ty::Vector(n, Scalar::F32),
                    span,
                )?;
                (vec![a], U32)
            }
            B::Unpack4x8Snorm
            | B::Unpack4x8Unorm
            | B::Unpack2x16Snorm
            | B::Unpack2x16Unorm
            | B::Unpack2x16Float => {
                want(1)?;
                let n = if matches!(b, B::Unpack4x8Snorm | B::Unpack4x8Unorm) {
                    4
                } else {
                    2
                };
                let a = self.convert(args.into_iter().next().expect("arg"), &U32, span)?;
                (vec![a], Ty::Vector(n, Scalar::F32))
            }
            B::WorkgroupBarrier | B::StorageBarrier => {
                want(0)?;
                (args, Ty::Void)
            }
            _ => return Err(span.error(format!("'{name}' is not a math built-in"))),
        };
        if foldable(b) && args.iter().all(|a| a.as_const().is_some()) {
            let consts: Vec<&ConstValue> =
                args.iter().map(|a| a.as_const().expect("const")).collect();
            if let Some(v) =
                consteval::builtin(b, &consts).map_err(|m| span.error(format!("{name}: {m}")))?
            {
                return Ok(konst(result, v, span));
            }
        }
        if result.is_abstract() || args.iter().any(|a| a.ty.is_abstract()) {
            let args = args
                .into_iter()
                .map(|a| self.concretize(a))
                .collect::<R<Vec<_>>>()?;
            return self.math_call(b, name, args, span);
        }
        Ok(Expr {
            ty: result,
            kind: EK::Builtin(b, args),
            span,
        })
    }

    fn array_length(&mut self, args: &[ast::Expr], span: Span) -> R<Expr> {
        if self.fcx.is_none() {
            return Err(span.error("'arrayLength' cannot be used in a constant expression"));
        }
        let [arg] = args else {
            return Err(span.error("'arrayLength' expects one argument"));
        };
        let AK::Unary(ast::UnaryOp::AddrOf, inner) = &arg.kind else {
            return Err(arg
                .span
                .error("'arrayLength' expects a pointer to a runtime-sized array: arrayLength(&buffer.array)"));
        };
        let Checked::Ref(p) = self.expr(inner)? else {
            return Err(inner
                .span
                .error("'arrayLength' requires a storage buffer runtime-sized array"));
        };
        let PlaceRoot::Global(g) = p.root else {
            return Err(inner
                .span
                .error("'arrayLength' requires a storage buffer runtime-sized array"));
        };
        if !matches!(self.globals[g].kind, GlobalKind::Storage { .. })
            || !matches!(p.ty, Ty::Array(_, None))
        {
            return Err(inner
                .span
                .error("'arrayLength' requires a storage buffer runtime-sized array"));
        }
        let member = match p.steps.as_slice() {
            [] => None,
            [Step::Member(i)] => Some(*i),
            _ => {
                return Err(inner
                    .span
                    .error("'arrayLength' requires a storage buffer runtime-sized array"));
            }
        };
        Ok(Expr {
            ty: U32,
            kind: EK::ArrayLength { global: g, member },
            span,
        })
    }

    fn handle_arg(
        &mut self,
        args: &[ast::Expr],
        i: usize,
        name: &str,
        what: &str,
        span: Span,
    ) -> R<(usize, Ty)> {
        let Some(a) = args.get(i) else {
            return Err(span.error(format!("'{name}' is missing its {what} argument")));
        };
        match self.expr(a)? {
            Checked::Handle(g, t) => Ok((g, t)),
            _ => Err(a.span.error(format!(
                "argument {} of '{name}' must be a {what} variable",
                i + 1
            ))),
        }
    }

    fn typed_arg(
        &mut self,
        args: &[ast::Expr],
        i: usize,
        ty: &Ty,
        name: &str,
        what: &str,
        span: Span,
    ) -> R<Expr> {
        let Some(a) = args.get(i) else {
            return Err(span.error(format!("'{name}' is missing its {what} argument")));
        };
        let v = self.value(a)?;
        self.convert(v, ty, a.span).map_err(|e| crate::Error {
            message: format!("{what} of '{name}': {}", e.message),
            ..e
        })
    }

    /// An i32 or u32 scalar (abstract integers become i32), or a vector of them of size `n`.
    fn int_arg(
        &mut self,
        args: &[ast::Expr],
        i: usize,
        n: Option<u8>,
        name: &str,
        what: &str,
        span: Span,
    ) -> R<Expr> {
        let Some(a) = args.get(i) else {
            return Err(span.error(format!("'{name}' is missing its {what} argument")));
        };
        let v = self.value(a)?;
        let v = self.concretize(v)?;
        let ok = match (&v.ty, n) {
            (Ty::Scalar(Scalar::I32 | Scalar::U32), None) => true,
            (Ty::Vector(m, Scalar::I32 | Scalar::U32), Some(k)) => *m == k,
            _ => false,
        };
        if !ok {
            let want = match n {
                None => "i32 or u32".to_string(),
                Some(k) => format!("vec{k}<i32> or vec{k}<u32>"),
            };
            return Err(a.span.error(format!(
                "{what} of '{name}' must be {want}, found {}",
                self.ty_name(&v.ty)
            )));
        }
        Ok(v)
    }

    fn texture_call(&mut self, op: B, name: &str, args: &[ast::Expr], span: Span) -> R<Expr> {
        let (tex, tty) = self.handle_arg(args, 0, name, "texture", span)?;
        let mut call = TexCall {
            op,
            texture: tex,
            sampler: None,
            coords: None,
            array_index: None,
            level: None,
            bias: None,
            ddx: None,
            ddy: None,
            depth_ref: None,
            offset: None,
            value: None,
        };
        let vec2f = Ty::Vector(2, Scalar::F32);
        let vec3f = Ty::Vector(3, Scalar::F32);
        let vec4f = Ty::Vector(4, Scalar::F32);
        let mut i;
        let result = match op {
            B::TextureSample
            | B::TextureSampleBias
            | B::TextureSampleLevel
            | B::TextureSampleGrad
            | B::TextureSampleCompare
            | B::TextureSampleCompareLevel => {
                let Ty::Texture(view, sample) = tty else {
                    return Err(args[0]
                        .span
                        .error(format!("'{name}' requires a sampled texture")));
                };
                let depth = sample == SampleType::Depth;
                let compare = matches!(op, B::TextureSampleCompare | B::TextureSampleCompareLevel);
                if compare && !depth {
                    return Err(args[0]
                        .span
                        .error(format!("'{name}' requires a depth texture")));
                }
                if matches!(op, B::TextureSampleBias | B::TextureSampleGrad) && depth {
                    return Err(args[0]
                        .span
                        .error(format!("'{name}' is not available for depth textures")));
                }
                if matches!(sample, SampleType::Sint | SampleType::Uint) {
                    return Err(args[0]
                        .span
                        .error("integer textures cannot be sampled; use textureLoad"));
                }
                let (smp, sty) = self.handle_arg(args, 1, name, "sampler", span)?;
                match sty {
                    Ty::Sampler(c) if c == compare => {}
                    Ty::Sampler(_) => {
                        return Err(args[1].span.error(format!(
                            "'{name}' requires a {}",
                            if compare {
                                "sampler_comparison"
                            } else {
                                "sampler"
                            }
                        )));
                    }
                    _ => {
                        return Err(args[1]
                            .span
                            .error(format!("argument 2 of '{name}' must be a sampler")));
                    }
                }
                call.sampler = Some(smp);
                let coord_ty = if view == TextureView::Cube {
                    vec3f.clone()
                } else {
                    vec2f.clone()
                };
                call.coords = Some(self.typed_arg(args, 2, &coord_ty, name, "coords", span)?);
                i = 3;
                if view == TextureView::D2Array {
                    call.array_index =
                        Some(self.int_arg(args, i, None, name, "array_index", span)?);
                    i += 1;
                }
                match op {
                    B::TextureSampleBias => {
                        call.bias = Some(self.typed_arg(args, i, &F32, name, "bias", span)?);
                        i += 1;
                    }
                    B::TextureSampleLevel => {
                        call.level = Some(if depth {
                            self.int_arg(args, i, None, name, "level", span)?
                        } else {
                            self.typed_arg(args, i, &F32, name, "level", span)?
                        });
                        i += 1;
                    }
                    B::TextureSampleGrad => {
                        call.ddx = Some(self.typed_arg(args, i, &coord_ty, name, "ddx", span)?);
                        call.ddy =
                            Some(self.typed_arg(args, i + 1, &coord_ty, name, "ddy", span)?);
                        i += 2;
                    }
                    B::TextureSampleCompare | B::TextureSampleCompareLevel => {
                        call.depth_ref =
                            Some(self.typed_arg(args, i, &F32, name, "depth_ref", span)?);
                        i += 1;
                    }
                    _ => {}
                }
                if args.len() > i {
                    if view == TextureView::Cube {
                        return Err(args[i]
                            .span
                            .error(format!("'{name}' takes no offset for cube textures")));
                    }
                    let off =
                        self.typed_arg(args, i, &Ty::Vector(2, Scalar::I32), name, "offset", span)?;
                    match off.as_const() {
                        Some(c)
                            if c.components()
                                .iter()
                                .all(|x| x.as_i64().is_some_and(|v| (-8..=7).contains(&v))) => {}
                        Some(_) => {
                            return Err(args[i]
                                .span
                                .error("texture offset components must be in [-8, 7]"));
                        }
                        None => {
                            return Err(args[i]
                                .span
                                .error("texture offset must be a constant expression"));
                        }
                    }
                    call.offset = Some(off);
                    i += 1;
                }
                if matches!(
                    op,
                    B::TextureSample | B::TextureSampleBias | B::TextureSampleCompare
                ) {
                    self.fcx_mut()
                        .stage_reqs
                        .push((StageReq::Fragment, span, name.to_string()));
                }
                if depth { F32 } else { vec4f }
            }
            B::TextureLoad => {
                let Ty::Texture(view, sample) = tty else {
                    return Err(args[0]
                        .span
                        .error("textureLoad on a write-only storage texture is not allowed"));
                };
                if view == TextureView::Cube {
                    return Err(args[0]
                        .span
                        .error("textureLoad is not available for cube textures"));
                }
                call.coords = Some(self.int_arg(args, 1, Some(2), name, "coords", span)?);
                i = 2;
                if view == TextureView::D2Array {
                    call.array_index =
                        Some(self.int_arg(args, i, None, name, "array_index", span)?);
                    i += 1;
                }
                call.level = Some(self.int_arg(args, i, None, name, "level", span)?);
                i += 1;
                match sample {
                    SampleType::Float => vec4f,
                    SampleType::Sint => Ty::Vector(4, Scalar::I32),
                    SampleType::Uint => Ty::Vector(4, Scalar::U32),
                    SampleType::Depth => F32,
                }
            }
            B::TextureStore => {
                let Ty::StorageTexture(_) = tty else {
                    return Err(args[0]
                        .span
                        .error("textureStore requires a texture_storage_2d"));
                };
                call.coords = Some(self.int_arg(args, 1, Some(2), name, "coords", span)?);
                call.value = Some(self.typed_arg(args, 2, &vec4f, name, "value", span)?);
                i = 3;
                Ty::Void
            }
            B::TextureDimensions => {
                i = 1;
                match tty {
                    Ty::Texture(..) => {
                        if args.len() > 1 {
                            call.level = Some(self.int_arg(args, 1, None, name, "level", span)?);
                            i = 2;
                        }
                    }
                    Ty::StorageTexture(_) => {}
                    _ => return Err(args[0].span.error("textureDimensions requires a texture")),
                }
                Ty::Vector(2, Scalar::U32)
            }
            B::TextureNumLevels => {
                if !matches!(tty, Ty::Texture(..)) {
                    return Err(args[0]
                        .span
                        .error("textureNumLevels requires a sampled or depth texture"));
                }
                i = 1;
                U32
            }
            B::TextureNumLayers => {
                if !matches!(tty, Ty::Texture(TextureView::D2Array, _)) {
                    return Err(args[0]
                        .span
                        .error("textureNumLayers requires an array texture"));
                }
                i = 1;
                U32
            }
            _ => unreachable!("texture built-in"),
        };
        if args.len() != i {
            return Err(span.error(format!(
                "'{name}' got {} argument(s); this overload takes {i}",
                args.len()
            )));
        }
        Ok(Expr {
            ty: result,
            kind: EK::Texture(Box::new(call)),
            span,
        })
    }
}
