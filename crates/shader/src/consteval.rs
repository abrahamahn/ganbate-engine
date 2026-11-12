//! Constant folding of WGSL const-expressions: abstract numerics (i64/f64) and concrete
//! i32/u32/f32/bool, with WGSL's shader-creation errors for overflow, division by zero and
//! non-finite floats.

use crate::ast::BinaryOp;
use crate::ir::{Builtin, UnaryOp};
use crate::types::{ConstValue as V, Scalar, Ty};

pub type R<T> = Result<T, String>;

pub fn map_leaves(v: &V, f: &mut dyn FnMut(&V) -> R<V>) -> R<V> {
    match v {
        V::Composite(c) => Ok(V::Composite(
            c.iter().map(|x| map_leaves(x, f)).collect::<R<_>>()?,
        )),
        leaf => f(leaf),
    }
}

fn finite32(x: f32) -> R<V> {
    if x.is_finite() {
        Ok(V::F32(x))
    } else {
        Err("constant expression evaluates to infinity or NaN".into())
    }
}

fn finite64(x: f64) -> R<V> {
    if x.is_finite() {
        Ok(V::AbstractFloat(x))
    } else {
        Err("constant expression evaluates to infinity or NaN".into())
    }
}

/// WGSL automatic conversion of an abstract leaf scalar to `to`.
pub fn auto_scalar(v: &V, to: Scalar) -> R<V> {
    Ok(match (v, to) {
        (V::AbstractInt(i), Scalar::I32) => V::I32(
            i32::try_from(*i).map_err(|_| format!("value {i} cannot be represented as i32"))?,
        ),
        (V::AbstractInt(i), Scalar::U32) => V::U32(
            u32::try_from(*i).map_err(|_| format!("value {i} cannot be represented as u32"))?,
        ),
        (V::AbstractInt(i), Scalar::F32) => finite32(*i as f32)?,
        (V::AbstractInt(i), Scalar::AbstractFloat) => V::AbstractFloat(*i as f64),
        (V::AbstractInt(i), Scalar::AbstractInt) => V::AbstractInt(*i),
        (V::AbstractFloat(f), Scalar::F32) => {
            let x = *f as f32;
            if !x.is_finite() {
                return Err(format!("value {f} cannot be represented as f32"));
            }
            V::F32(x)
        }
        (V::AbstractFloat(f), Scalar::AbstractFloat) => V::AbstractFloat(*f),
        (V::Bool(b), Scalar::Bool) => V::Bool(*b),
        (V::I32(i), Scalar::I32) => V::I32(*i),
        (V::U32(u), Scalar::U32) => V::U32(*u),
        (V::F32(f), Scalar::F32) => V::F32(*f),
        _ => return Err(format!("cannot convert constant to {}", to.name())),
    })
}

/// Convert every leaf of `v` automatically to `to`.
pub fn auto_convert(v: &V, to: Scalar) -> R<V> {
    map_leaves(v, &mut |x| auto_scalar(x, to))
}

/// Value-constructor conversion (`f32(e)`, `u32(e)`, `bool(e)` ...) of a leaf scalar.
pub fn cast_scalar(v: &V, to: Scalar) -> R<V> {
    let as_f64 = |v: &V| -> f64 {
        match *v {
            V::Bool(b) => b as u8 as f64,
            V::I32(i) => i as f64,
            V::U32(u) => u as f64,
            V::F32(f) => f as f64,
            V::AbstractInt(i) => i as f64,
            V::AbstractFloat(f) => f,
            V::Composite(_) => 0.0,
        }
    };
    Ok(match to {
        Scalar::Bool => V::Bool(match *v {
            V::Bool(b) => b,
            V::I32(i) => i != 0,
            V::U32(u) => u != 0,
            V::F32(f) => f != 0.0,
            V::AbstractInt(i) => i != 0,
            V::AbstractFloat(f) => f != 0.0,
            V::Composite(_) => return Err("cannot convert a composite".into()),
        }),
        Scalar::F32 => match *v {
            V::F32(f) => V::F32(f),
            V::I32(i) => V::F32(i as f32),
            V::U32(u) => V::F32(u as f32),
            V::Bool(b) => V::F32(if b { 1.0 } else { 0.0 }),
            ref other => finite32(as_f64(other) as f32)?,
        },
        Scalar::I32 => match *v {
            V::I32(i) => V::I32(i),
            V::U32(u) => V::I32(u as i32),
            V::Bool(b) => V::I32(b as i32),
            V::AbstractInt(i) => V::I32(
                i32::try_from(i).map_err(|_| format!("value {i} cannot be represented as i32"))?,
            ),
            // f32 → i32 rounds toward zero and saturates, like WGSL's runtime conversion.
            ref other => V::I32(as_f64(other).trunc() as i32),
        },
        Scalar::U32 => match *v {
            V::U32(u) => V::U32(u),
            V::I32(i) => V::U32(i as u32),
            V::Bool(b) => V::U32(b as u32),
            V::AbstractInt(i) => V::U32(
                u32::try_from(i).map_err(|_| format!("value {i} cannot be represented as u32"))?,
            ),
            ref other => V::U32(as_f64(other).trunc() as u32),
        },
        Scalar::AbstractFloat => match *v {
            V::AbstractInt(i) => V::AbstractFloat(i as f64),
            V::AbstractFloat(f) => V::AbstractFloat(f),
            _ => return Err("cannot convert to abstract-float".into()),
        },
        Scalar::AbstractInt => match *v {
            V::AbstractInt(i) => V::AbstractInt(i),
            _ => return Err("cannot convert to abstract-int".into()),
        },
    })
}

pub fn bitcast_scalar(v: &V, to: Scalar) -> R<V> {
    let bits = match *v {
        V::I32(i) => i as u32,
        V::U32(u) => u,
        V::F32(f) => f.to_bits(),
        _ => return Err("bitcast requires a 32-bit concrete value".into()),
    };
    Ok(match to {
        Scalar::I32 => V::I32(bits as i32),
        Scalar::U32 => V::U32(bits),
        Scalar::F32 => finite32(f32::from_bits(bits))?,
        _ => return Err("bitcast target must be i32, u32 or f32".into()),
    })
}

pub fn unary(op: UnaryOp, v: &V) -> R<V> {
    map_leaves(v, &mut |x| {
        Ok(match (op, x) {
            (UnaryOp::Neg, V::AbstractInt(i)) => V::AbstractInt(
                i.checked_neg()
                    .ok_or("integer overflow in constant expression")?,
            ),
            (UnaryOp::Neg, V::I32(i)) => V::I32(
                i.checked_neg()
                    .ok_or("i32 overflow in constant expression")?,
            ),
            (UnaryOp::Neg, V::F32(f)) => V::F32(-f),
            (UnaryOp::Neg, V::AbstractFloat(f)) => V::AbstractFloat(-f),
            (UnaryOp::Not, V::Bool(b)) => V::Bool(!b),
            (UnaryOp::BitNot, V::I32(i)) => V::I32(!i),
            (UnaryOp::BitNot, V::U32(u)) => V::U32(!u),
            (UnaryOp::BitNot, V::AbstractInt(i)) => V::AbstractInt(!i),
            _ => return Err("invalid operand for unary operator".into()),
        })
    })
}

/// Fold a binary operator. Returns `Ok(None)` for operations that are not folded
/// (matrix arithmetic), which the emitter then evaluates at run time.
pub fn binary(op: BinaryOp, l: &V, lt: &Ty, r: &V, rt: &Ty) -> R<Option<V>> {
    let zip = |a: &[V], b: &[V]| -> R<V> {
        Ok(V::Composite(
            a.iter()
                .zip(b)
                .map(|(x, y)| scalar_binary(op, x, y))
                .collect::<R<_>>()?,
        ))
    };
    Ok(Some(match (lt, rt) {
        (Ty::Scalar(_), Ty::Scalar(_)) => scalar_binary(op, l, r)?,
        (Ty::Vector(..), Ty::Vector(..)) => zip(l.components(), r.components())?,
        (Ty::Vector(..), Ty::Scalar(_)) => V::Composite(
            l.components()
                .iter()
                .map(|x| scalar_binary(op, x, r))
                .collect::<R<_>>()?,
        ),
        (Ty::Scalar(_), Ty::Vector(..)) => V::Composite(
            r.components()
                .iter()
                .map(|y| scalar_binary(op, l, y))
                .collect::<R<_>>()?,
        ),
        _ => return Ok(None),
    }))
}

fn scalar_binary(op: BinaryOp, a: &V, b: &V) -> R<V> {
    use BinaryOp::*;
    const OVERFLOW: &str = "integer overflow in constant expression";
    const DIV0: &str = "division by zero in constant expression";
    Ok(match (a, b) {
        (V::AbstractInt(x), V::AbstractInt(y)) => {
            let (x, y) = (*x, *y);
            match op {
                Add => V::AbstractInt(x.checked_add(y).ok_or(OVERFLOW)?),
                Sub => V::AbstractInt(x.checked_sub(y).ok_or(OVERFLOW)?),
                Mul => V::AbstractInt(x.checked_mul(y).ok_or(OVERFLOW)?),
                Div => {
                    if y == 0 {
                        return Err(DIV0.into());
                    }
                    V::AbstractInt(x.checked_div(y).ok_or(OVERFLOW)?)
                }
                Rem => {
                    if y == 0 {
                        return Err(DIV0.into());
                    }
                    V::AbstractInt(x.checked_rem(y).ok_or(OVERFLOW)?)
                }
                And => V::AbstractInt(x & y),
                Or => V::AbstractInt(x | y),
                Xor => V::AbstractInt(x ^ y),
                _ => compare(op, x.cmp(&y))?,
            }
        }
        (V::AbstractInt(x), V::U32(s)) => shift_i64(op, *x, *s)?,
        (V::I32(x), V::U32(s)) if matches!(op, Shl | Shr) => {
            if *s >= 32 {
                return Err(format!("shift by {s} is not less than the bit width (32)"));
            }
            match op {
                Shl => {
                    let wide = (*x as i64) << s;
                    V::I32(i32::try_from(wide).map_err(|_| OVERFLOW)?)
                }
                _ => V::I32(x >> s),
            }
        }
        (V::U32(x), V::U32(y)) if matches!(op, Shl | Shr) => {
            if *y >= 32 {
                return Err(format!("shift by {y} is not less than the bit width (32)"));
            }
            match op {
                Shl => {
                    let wide = (*x as u64) << y;
                    V::U32(u32::try_from(wide).map_err(|_| OVERFLOW)?)
                }
                _ => V::U32(x >> y),
            }
        }
        (V::I32(x), V::I32(y)) => {
            let (x, y) = (*x, *y);
            match op {
                Add => V::I32(x.checked_add(y).ok_or(OVERFLOW)?),
                Sub => V::I32(x.checked_sub(y).ok_or(OVERFLOW)?),
                Mul => V::I32(x.checked_mul(y).ok_or(OVERFLOW)?),
                Div => {
                    if y == 0 {
                        return Err(DIV0.into());
                    }
                    V::I32(x.checked_div(y).ok_or(OVERFLOW)?)
                }
                Rem => {
                    if y == 0 {
                        return Err(DIV0.into());
                    }
                    V::I32(x.checked_rem(y).ok_or(OVERFLOW)?)
                }
                And => V::I32(x & y),
                Or => V::I32(x | y),
                Xor => V::I32(x ^ y),
                _ => compare(op, x.cmp(&y))?,
            }
        }
        (V::U32(x), V::U32(y)) => {
            let (x, y) = (*x, *y);
            match op {
                Add => V::U32(x.checked_add(y).ok_or(OVERFLOW)?),
                Sub => V::U32(x.checked_sub(y).ok_or(OVERFLOW)?),
                Mul => V::U32(x.checked_mul(y).ok_or(OVERFLOW)?),
                Div => V::U32(x.checked_div(y).ok_or(DIV0)?),
                Rem => V::U32(x.checked_rem(y).ok_or(DIV0)?),
                And => V::U32(x & y),
                Or => V::U32(x | y),
                Xor => V::U32(x ^ y),
                _ => compare(op, x.cmp(&y))?,
            }
        }
        (V::F32(x), V::F32(y)) => {
            let (x, y) = (*x, *y);
            match op {
                Add => finite32(x + y)?,
                Sub => finite32(x - y)?,
                Mul => finite32(x * y)?,
                Div => finite32(x / y)?,
                Rem => finite32(x % y)?,
                _ => fcompare(op, x as f64, y as f64)?,
            }
        }
        (V::AbstractFloat(x), V::AbstractFloat(y)) => {
            let (x, y) = (*x, *y);
            match op {
                Add => finite64(x + y)?,
                Sub => finite64(x - y)?,
                Mul => finite64(x * y)?,
                Div => finite64(x / y)?,
                Rem => finite64(x % y)?,
                _ => fcompare(op, x, y)?,
            }
        }
        (V::Bool(x), V::Bool(y)) => V::Bool(match op {
            Eq => x == y,
            Ne => x != y,
            And | LogicalAnd => *x && *y,
            Or | LogicalOr => *x || *y,
            _ => return Err("invalid operator for bool".into()),
        }),
        _ => return Err("mismatched operand types in constant expression".into()),
    })
}

fn shift_i64(op: BinaryOp, x: i64, s: u32) -> R<V> {
    if s >= 64 {
        return Err(format!("shift by {s} is not less than the bit width (64)"));
    }
    Ok(match op {
        BinaryOp::Shl => {
            let wide = (x as i128) << s;
            V::AbstractInt(
                i64::try_from(wide).map_err(|_| "integer overflow in constant expression")?,
            )
        }
        BinaryOp::Shr => V::AbstractInt(x >> s),
        _ => return Err("mismatched operand types in constant expression".into()),
    })
}

fn compare(op: BinaryOp, o: std::cmp::Ordering) -> R<V> {
    use std::cmp::Ordering::*;
    Ok(V::Bool(match op {
        BinaryOp::Eq => o == Equal,
        BinaryOp::Ne => o != Equal,
        BinaryOp::Lt => o == Less,
        BinaryOp::Le => o != Greater,
        BinaryOp::Gt => o == Greater,
        BinaryOp::Ge => o != Less,
        _ => return Err("invalid operator in constant expression".into()),
    }))
}

fn fcompare(op: BinaryOp, x: f64, y: f64) -> R<V> {
    Ok(V::Bool(match op {
        BinaryOp::Eq => x == y,
        BinaryOp::Ne => x != y,
        BinaryOp::Lt => x < y,
        BinaryOp::Le => x <= y,
        BinaryOp::Gt => x > y,
        BinaryOp::Ge => x >= y,
        _ => return Err("invalid operator for floating-point operands".into()),
    }))
}

// ------------------------------------------------------------------- built-in functions

fn leaf_f64(v: &V) -> Option<f64> {
    match *v {
        V::F32(f) => Some(f as f64),
        V::AbstractFloat(f) => Some(f),
        _ => None,
    }
}

/// Rebuild a float leaf of the same kind as `like`.
fn float_like(like: &V, x: f64) -> R<V> {
    match like {
        V::F32(_) => finite32(x as f32),
        _ => finite64(x),
    }
}

fn float1(v: &V, f: impl Fn(f64) -> f64) -> R<V> {
    map_leaves(v, &mut |x| match leaf_f64(x) {
        Some(a) => float_like(x, f(a)),
        None => Err("expected a floating-point constant".into()),
    })
}

fn zip2(a: &V, b: &V, f: &dyn Fn(&V, &V) -> R<V>) -> R<V> {
    match (a, b) {
        (V::Composite(x), V::Composite(y)) => Ok(V::Composite(
            x.iter()
                .zip(y)
                .map(|(p, q)| zip2(p, q, f))
                .collect::<R<_>>()?,
        )),
        (V::Composite(x), q) => Ok(V::Composite(
            x.iter().map(|p| zip2(p, q, f)).collect::<R<_>>()?,
        )),
        _ => f(a, b),
    }
}

fn zip3(a: &V, b: &V, c: &V, f: &dyn Fn(&V, &V, &V) -> R<V>) -> R<V> {
    match (a, b, c) {
        (V::Composite(x), V::Composite(y), V::Composite(z)) => Ok(V::Composite(
            x.iter()
                .zip(y)
                .zip(z)
                .map(|((p, q), r)| zip3(p, q, r, f))
                .collect::<R<_>>()?,
        )),
        (V::Composite(x), V::Composite(y), r) => Ok(V::Composite(
            x.iter()
                .zip(y)
                .map(|(p, q)| zip3(p, q, r, f))
                .collect::<R<_>>()?,
        )),
        _ => f(a, b, c),
    }
}

fn float2(a: &V, b: &V, f: impl Fn(f64, f64) -> f64) -> R<V> {
    zip2(a, b, &|x, y| match (leaf_f64(x), leaf_f64(y)) {
        (Some(p), Some(q)) => float_like(x, f(p, q)),
        _ => Err("expected floating-point constants".into()),
    })
}

fn float3(a: &V, b: &V, c: &V, f: impl Fn(f64, f64, f64) -> f64) -> R<V> {
    zip3(
        a,
        b,
        c,
        &|x, y, z| match (leaf_f64(x), leaf_f64(y), leaf_f64(z)) {
            (Some(p), Some(q), Some(r)) => float_like(x, f(p, q, r)),
            _ => Err("expected floating-point constants".into()),
        },
    )
}

fn int_minmax(a: &V, b: &V, max: bool) -> R<V> {
    Ok(match (a, b) {
        (V::I32(x), V::I32(y)) => V::I32(if max { *x.max(y) } else { *x.min(y) }),
        (V::U32(x), V::U32(y)) => V::U32(if max { *x.max(y) } else { *x.min(y) }),
        (V::AbstractInt(x), V::AbstractInt(y)) => {
            V::AbstractInt(if max { *x.max(y) } else { *x.min(y) })
        }
        _ => match (leaf_f64(a), leaf_f64(b)) {
            (Some(x), Some(y)) => float_like(a, if max { x.max(y) } else { x.min(y) })?,
            _ => return Err("mismatched constant operands".into()),
        },
    })
}

fn sum_products(a: &V, b: &V) -> R<V> {
    let mut acc: Option<V> = None;
    for (x, y) in a.components().iter().zip(b.components()) {
        let p = scalar_binary(BinaryOp::Mul, x, y)?;
        acc = Some(match acc {
            None => p,
            Some(s) => scalar_binary(BinaryOp::Add, &s, &p)?,
        });
    }
    acc.ok_or_else(|| "empty vector".into())
}

/// Fold a built-in call with constant arguments; `Ok(None)` if this built-in is not folded.
pub fn builtin(b: Builtin, args: &[&V]) -> R<Option<V>> {
    use Builtin as B;
    use std::f64::consts::PI;
    let a0 = args.first().copied();
    let v = match (b, args.len()) {
        (B::Abs, 1) => map_leaves(args[0], &mut |x| {
            Ok(match *x {
                V::I32(i) => V::I32(i.wrapping_abs()),
                V::U32(u) => V::U32(u),
                V::AbstractInt(i) => V::AbstractInt(
                    i.checked_abs()
                        .ok_or("integer overflow in constant expression")?,
                ),
                ref f => float_like(f, leaf_f64(f).unwrap_or(0.0).abs())?,
            })
        })?,
        (B::Sign, 1) => map_leaves(args[0], &mut |x| {
            Ok(match *x {
                V::I32(i) => V::I32(i.signum()),
                V::AbstractInt(i) => V::AbstractInt(i.signum()),
                ref f => {
                    let a = leaf_f64(f).ok_or("invalid operand")?;
                    float_like(
                        f,
                        if a > 0.0 {
                            1.0
                        } else if a < 0.0 {
                            -1.0
                        } else {
                            0.0
                        },
                    )?
                }
            })
        })?,
        (B::Acos, 1) => float1(args[0], f64::acos)?,
        (B::Acosh, 1) => float1(args[0], f64::acosh)?,
        (B::Asin, 1) => float1(args[0], f64::asin)?,
        (B::Asinh, 1) => float1(args[0], f64::asinh)?,
        (B::Atan, 1) => float1(args[0], f64::atan)?,
        (B::Atanh, 1) => float1(args[0], f64::atanh)?,
        (B::Ceil, 1) => float1(args[0], f64::ceil)?,
        (B::Cos, 1) => float1(args[0], f64::cos)?,
        (B::Cosh, 1) => float1(args[0], f64::cosh)?,
        (B::Degrees, 1) => float1(args[0], |x| x * 180.0 / PI)?,
        (B::Exp, 1) => float1(args[0], f64::exp)?,
        (B::Exp2, 1) => float1(args[0], f64::exp2)?,
        (B::Floor, 1) => float1(args[0], f64::floor)?,
        (B::Fract, 1) => float1(args[0], |x| x - x.floor())?,
        (B::InverseSqrt, 1) => float1(args[0], |x| 1.0 / x.sqrt())?,
        (B::Log, 1) => float1(args[0], f64::ln)?,
        (B::Log2, 1) => float1(args[0], f64::log2)?,
        (B::Radians, 1) => float1(args[0], |x| x * PI / 180.0)?,
        (B::Round, 1) => float1(args[0], f64::round_ties_even)?,
        (B::Saturate, 1) => float1(args[0], |x| x.clamp(0.0, 1.0))?,
        (B::Sin, 1) => float1(args[0], f64::sin)?,
        (B::Sinh, 1) => float1(args[0], f64::sinh)?,
        (B::Sqrt, 1) => float1(args[0], f64::sqrt)?,
        (B::Tan, 1) => float1(args[0], f64::tan)?,
        (B::Tanh, 1) => float1(args[0], f64::tanh)?,
        (B::Trunc, 1) => float1(args[0], f64::trunc)?,
        (B::Atan2, 2) => float2(args[0], args[1], f64::atan2)?,
        (B::Pow, 2) => float2(args[0], args[1], f64::powf)?,
        (B::Step, 2) => float2(args[0], args[1], |e, x| if e <= x { 1.0 } else { 0.0 })?,
        (B::Max, 2) => zip2(args[0], args[1], &|x, y| int_minmax(x, y, true))?,
        (B::Min, 2) => zip2(args[0], args[1], &|x, y| int_minmax(x, y, false))?,
        (B::Clamp, 3) => zip3(args[0], args[1], args[2], &|e, lo, hi| {
            let m = int_minmax(e, lo, true)?;
            int_minmax(&m, hi, false)
        })?,
        (B::Fma, 3) => float3(args[0], args[1], args[2], |a, b, c| a * b + c)?,
        (B::Mix, 3) => float3(args[0], args[1], args[2], |a, b, t| a * (1.0 - t) + b * t)?,
        (B::Smoothstep, 3) => float3(args[0], args[1], args[2], |lo, hi, x| {
            let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        })?,
        (B::Dot, 2) => sum_products(args[0], args[1])?,
        (B::Length, 1) => {
            let a = args[0];
            let sq: f64 = a
                .components()
                .iter()
                .filter_map(leaf_f64)
                .map(|x| x * x)
                .sum();
            float_like(&a.components()[0], sq.sqrt())?
        }
        (B::Distance, 2) => {
            let d = zip2(args[0], args[1], &|x, y| scalar_binary(BinaryOp::Sub, x, y))?;
            let sq: f64 = d
                .components()
                .iter()
                .filter_map(leaf_f64)
                .map(|x| x * x)
                .sum();
            float_like(&d.components()[0], sq.sqrt())?
        }
        (B::Normalize, 1) => {
            let a = args[0];
            let len: f64 = a
                .components()
                .iter()
                .filter_map(leaf_f64)
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            float1(a, |x| x / len)?
        }
        (B::Cross, 2) => {
            let (p, q) = (args[0].components(), args[1].components());
            let f = |v: &V| leaf_f64(v).unwrap_or(0.0);
            let (a1, a2, a3, b1, b2, b3) =
                (f(&p[0]), f(&p[1]), f(&p[2]), f(&q[0]), f(&q[1]), f(&q[2]));
            V::Composite(vec![
                float_like(&p[0], a2 * b3 - a3 * b2)?,
                float_like(&p[0], a3 * b1 - a1 * b3)?,
                float_like(&p[0], a1 * b2 - a2 * b1)?,
            ])
        }
        (B::Select, 3) => match args[2] {
            V::Bool(c) => {
                if *c {
                    args[1].clone()
                } else {
                    args[0].clone()
                }
            }
            V::Composite(c) => V::Composite(
                c.iter()
                    .enumerate()
                    .map(|(i, ci)| {
                        if ci.as_bool() == Some(true) {
                            args[1].components()[i].clone()
                        } else {
                            args[0].components()[i].clone()
                        }
                    })
                    .collect(),
            ),
            _ => return Ok(None),
        },
        (B::All, 1) => {
            V::Bool(a0.is_some_and(|a| a.components().iter().all(|c| c.as_bool() == Some(true))))
        }
        (B::Any, 1) => {
            V::Bool(a0.is_some_and(|a| a.components().iter().any(|c| c.as_bool() == Some(true))))
        }
        _ => return Ok(None),
    };
    Ok(Some(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abstract_arithmetic_and_overflow() {
        let a = V::AbstractInt(i64::MAX);
        assert!(scalar_binary(BinaryOp::Add, &a, &V::AbstractInt(1)).is_err());
        assert_eq!(
            scalar_binary(BinaryOp::Mul, &V::AbstractInt(6), &V::AbstractInt(7)).unwrap(),
            V::AbstractInt(42)
        );
        assert!(scalar_binary(BinaryOp::Div, &V::I32(1), &V::I32(0)).is_err());
        assert!(scalar_binary(BinaryOp::Add, &V::I32(i32::MAX), &V::I32(1)).is_err());
        assert_eq!(
            scalar_binary(BinaryOp::Rem, &V::I32(-7), &V::I32(3)).unwrap(),
            V::I32(-1)
        );
        assert!(scalar_binary(BinaryOp::Shl, &V::U32(1), &V::U32(32)).is_err());
        assert_eq!(
            scalar_binary(BinaryOp::Shr, &V::U32(0x8000_0000), &V::U32(31)).unwrap(),
            V::U32(1)
        );
    }

    #[test]
    fn conversions() {
        assert_eq!(
            auto_scalar(&V::AbstractInt(3), Scalar::F32).unwrap(),
            V::F32(3.0)
        );
        assert!(auto_scalar(&V::AbstractInt(-1), Scalar::U32).is_err());
        assert!(auto_scalar(&V::AbstractFloat(1.5), Scalar::I32).is_err());
        assert_eq!(
            cast_scalar(&V::F32(-1.75), Scalar::I32).unwrap(),
            V::I32(-1)
        );
        assert_eq!(
            cast_scalar(&V::I32(-1), Scalar::U32).unwrap(),
            V::U32(u32::MAX)
        );
        assert_eq!(
            bitcast_scalar(&V::U32(0x3f80_0000), Scalar::F32).unwrap(),
            V::F32(1.0)
        );
    }

    #[test]
    fn builtins_fold() {
        let v = V::Composite(vec![V::F32(3.0), V::F32(4.0)]);
        assert_eq!(builtin(Builtin::Length, &[&v]).unwrap(), Some(V::F32(5.0)));
        assert_eq!(
            builtin(Builtin::Round, &[&V::AbstractFloat(2.5)]).unwrap(),
            Some(V::AbstractFloat(2.0))
        );
        assert_eq!(
            builtin(Builtin::Sign, &[&V::F32(0.0)]).unwrap(),
            Some(V::F32(0.0))
        );
        assert!(builtin(Builtin::Sqrt, &[&V::F32(-1.0)]).is_err());
    }
}
