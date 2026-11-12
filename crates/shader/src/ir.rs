//! Typed intermediate representation: the checker's output and the emitter's input.
//!
//! Every expression carries its concrete type. Abstract numeric types only ever appear on
//! `Const` expressions; the checker converts them before they reach a non-constant context.
//! Memory is accessed through [`Place`]s (WGSL references), values are SSA.

use std::collections::BTreeSet;

use crate::Stage;
use crate::ast::BinaryOp;
use crate::lexer::Span;
use crate::types::{ConstValue, IoAttrs, StructInfo, Ty};

pub type ValueId = u32;
pub type LocalId = u32;
pub type GlobalId = usize;
pub type FnId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalKind {
    Private,
    Uniform,
    Storage {
        read_only: bool,
    },
    PushConstant,
    /// Textures and samplers.
    Handle,
    /// A constant composite that is indexed dynamically; lowered to a `Private` variable with a
    /// constant initializer.
    ConstData,
}

#[derive(Clone, Debug)]
pub struct Global {
    pub name: String,
    pub span: Span,
    pub kind: GlobalKind,
    pub ty: Ty,
    pub binding: Option<(u32, u32)>,
    pub init: Option<ConstValue>,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub span: Span,
    pub ty: Ty,
    pub io: IoAttrs,
    pub value: ValueId,
}

#[derive(Clone, Debug)]
pub struct Local {
    pub name: String,
    pub ty: Ty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageReq {
    Fragment,
    Compute,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub span: Span,
    pub params: Vec<Param>,
    pub ret: Option<Ty>,
    pub ret_io: IoAttrs,
    pub stage: Option<Stage>,
    pub stage_span: Span,
    pub workgroup_size: [u32; 3],
    pub locals: Vec<Local>,
    /// Debug name of each SSA value (`let`s and parameters), indexed by `ValueId`.
    pub value_names: Vec<String>,
    pub body: Vec<Stmt>,
    pub calls: BTreeSet<FnId>,
    pub globals: BTreeSet<GlobalId>,
    /// Built-ins used directly by this function that restrict the pipeline stage.
    pub stage_reqs: Vec<(StageReq, Span, String)>,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub structs: Vec<StructInfo>,
    pub globals: Vec<Global>,
    pub functions: Vec<Function>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub ty: Ty,
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Const(ConstValue),
    /// The zero value of `ty` (lowered to `OpConstantNull`).
    Zero,
    /// A `let` or parameter.
    Value(ValueId),
    Load(Box<Place>),
    /// Member, constant index or single component of a value.
    Extract(Box<Expr>, u32),
    /// Dynamic index into a vector, array or matrix value.
    DynIndex(Box<Expr>, Box<Expr>),
    /// Multi-component swizzle of a vector value.
    Swizzle(Box<Expr>, Vec<u32>),
    Unary(UnaryOp, Box<Expr>),
    /// Operands are already converted to the operator's overload (mixed vector/scalar and
    /// matrix forms are kept as written).
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// `OpCompositeConstruct` operands: scalars/vectors for vectors, columns for matrices,
    /// elements for arrays, members for structs.
    Construct(Vec<Expr>),
    /// Scalar broadcast to the vector type `ty`.
    Splat(Box<Expr>),
    /// Numeric/boolean conversion to `ty` (same shape).
    Convert(Box<Expr>),
    Bitcast(Box<Expr>),
    Call(FnId, Vec<Expr>),
    Builtin(Builtin, Vec<Expr>),
    Texture(Box<TexCall>),
    ArrayLength {
        global: GlobalId,
        member: Option<u32>,
    },
    /// The current value of the target of a compound assignment / increment.
    Current,
}

impl Expr {
    pub fn as_const(&self) -> Option<&ConstValue> {
        match &self.kind {
            ExprKind::Const(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Place {
    pub root: PlaceRoot,
    pub steps: Vec<Step>,
    pub ty: Ty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaceRoot {
    Local(LocalId),
    Global(GlobalId),
}

#[derive(Clone, Debug)]
pub enum Step {
    Member(u32),
    Index(Expr),
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let(ValueId, Expr),
    /// Store `value` into `place` (or into the given components of the vector at `place`).
    /// `value` may contain `ExprKind::Current`, the target's value before the store.
    Assign {
        place: Place,
        swizzle: Option<Vec<u32>>,
        value: Expr,
    },
    Eval(Expr),
    If {
        cond: Expr,
        then: Vec<Stmt>,
        otherwise: Vec<Stmt>,
    },
    /// All loops: `cond` is tested before each iteration (false exits), `continuing` runs
    /// at the continue target, then `break_if` is tested.
    Loop {
        cond: Option<Expr>,
        body: Vec<Stmt>,
        continuing: Vec<Stmt>,
        break_if: Option<Expr>,
    },
    Break,
    Continue,
    Return(Option<Expr>),
    Discard,
    Block(Vec<Stmt>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Builtin {
    Abs,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Atan2,
    Ceil,
    Clamp,
    Cos,
    Cosh,
    CountOneBits,
    Cross,
    Degrees,
    Determinant,
    Distance,
    Dot,
    Dpdx,
    Dpdy,
    Exp,
    Exp2,
    FaceForward,
    Floor,
    Fma,
    Fract,
    Fwidth,
    InverseSqrt,
    Ldexp,
    Length,
    Log,
    Log2,
    Max,
    Min,
    Mix,
    Normalize,
    Pow,
    Radians,
    Reflect,
    Refract,
    ReverseBits,
    Round,
    Saturate,
    Sign,
    Sin,
    Sinh,
    Smoothstep,
    Sqrt,
    Step,
    Tan,
    Tanh,
    Transpose,
    Trunc,
    Select,
    All,
    Any,
    Pack4x8Snorm,
    Pack4x8Unorm,
    Pack2x16Snorm,
    Pack2x16Unorm,
    Pack2x16Float,
    Unpack4x8Snorm,
    Unpack4x8Unorm,
    Unpack2x16Snorm,
    Unpack2x16Unorm,
    Unpack2x16Float,
    WorkgroupBarrier,
    StorageBarrier,
    TextureSample,
    TextureSampleBias,
    TextureSampleLevel,
    TextureSampleGrad,
    TextureSampleCompare,
    TextureSampleCompareLevel,
    TextureLoad,
    TextureStore,
    TextureDimensions,
    TextureNumLevels,
    TextureNumLayers,
}

/// A texture built-in call with its operands sorted out by the checker.
#[derive(Clone, Debug)]
pub struct TexCall {
    pub op: Builtin,
    pub texture: GlobalId,
    pub sampler: Option<GlobalId>,
    pub coords: Option<Expr>,
    pub array_index: Option<Expr>,
    /// Explicit LOD (f32 for sampled colour textures, i32/u32 for loads, depth and queries).
    pub level: Option<Expr>,
    pub bias: Option<Expr>,
    pub ddx: Option<Expr>,
    pub ddy: Option<Expr>,
    pub depth_ref: Option<Expr>,
    /// Constant `vec2<i32>` texel offset.
    pub offset: Option<Expr>,
    /// Texel for `textureStore`.
    pub value: Option<Expr>,
}
