//! WGSL for WebGPU, generated from the same checked IR as the SPIR-V of an entry point.
//!
//! The output is a self-contained WGSL module holding only what the entry point reaches, with
//! Ganbate's extensions and Vulkan conventions lowered to standard WGSL:
//!
//! * `var<push_constant>` becomes `@group(PUSH_CONSTANT_GROUP) @binding(PUSH_CONSTANT_BINDING)
//!   var<uniform>`. WebGPU has no push constants; the backend binds this uniform with a dynamic
//!   offset into a per-frame ring buffer (256-byte aligned). [`Wgsl::push_constants`] reports
//!   the binding, and the WGSL uniform layout rules are checked for the block. A shader that uses
//!   push constants cannot also use that group.
//! * The entry point gets a generated wrapper, [`Wgsl::entry`]: it declares each built-in input
//!   once (Ganbate allows the same built-in in a struct and as a parameter), marks integer varyings
//!   `@interpolate(flat)`, calls the user's entry function and returns its outputs. A vertex
//!   wrapper negates the y of `@builtin(position)`: Ganbate's clip space is Vulkan's (+y down),
//!   WebGPU's is +y up. The rasterized image, `@builtin(position)` in fragment shaders, winding
//!   (`front_facing`) and derivatives therefore match the Vulkan backend exactly.
//! * Multi-component swizzle assignments become per-component stores; compound assignments
//!   evaluate their target's indices once; dynamically indexed constants are `var<private>`
//!   with an initializer, as in SPIR-V.
//! * `discard` is followed by a `return`, keeping Ganbate's terminate semantics (WGSL's `discard`
//!   demotes the invocation to a helper and continues).
//! * Module-scope names are kept unless they are WGSL reserved words or predeclared names the
//!   output relies on; parameters, `let`s and variables get unique names.
//!
//! No uniformity analysis is done by Ganbate: WebGPU compilers reject derivatives and
//! implicit-LOD sampling in non-uniform control flow, which the Vulkan path accepts.
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::ast::BinaryOp;
use crate::ir::{
    Builtin as B, Expr, ExprKind as EK, FnId, GlobalId, GlobalKind, Place, PlaceRoot, Program,
    Step, Stmt, TexCall, UnaryOp, ValueId,
};
use crate::types::*;
use crate::{Error, Stage};

/// The bind group that push constants move to on WebGPU (the last of the four groups every
/// WebGPU device supports).
pub const PUSH_CONSTANT_GROUP: u32 = 3;
/// The binding of the push-constant uniform inside [`PUSH_CONSTANT_GROUP`].
pub const PUSH_CONSTANT_BINDING: u32 = 0;

/// WebGPU output for one entry point, produced by [`crate::compile_wgsl`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wgsl {
    /// A complete WGSL module.
    pub source: String,
    /// Name of the generated entry point in `source` (usually `main`).
    pub entry: String,
    /// `(group, binding)` of the uniform that replaces the `var<push_constant>` block, if the
    /// entry point uses one. Its size is [`crate::Compiled::push_constant_size`].
    pub push_constants: Option<(u32, u32)>,
}

type R<T> = Result<T, Error>;

/// WGSL reserved words and keywords (WGSL §15.2 and §15.3).
#[rustfmt::skip]
const RESERVED: &[&str] = &[
    "NULL", "Self", "abstract", "active", "alignas", "alignof", "as", "asm", "asm_fragment",
    "async", "attribute", "auto", "await", "become", "binding_array", "cast", "catch", "class",
    "co_await", "co_return", "co_yield", "coherent", "column_major", "common", "compile",
    "compile_fragment", "concept", "const_cast", "consteval", "constexpr", "constinit", "crate",
    "debugger", "decltype", "delete", "demote", "demote_to_helper", "do", "dynamic_cast",
    "enum", "explicit", "export", "extends", "extern", "external", "fallthrough", "filter",
    "final", "finally", "friend", "from", "fxgroup", "get", "goto", "groupshared", "highp",
    "impl", "implements", "import", "inline", "instanceof", "interface", "layout", "lowp",
    "macro", "macro_rules", "match", "mediump", "meta", "mod", "module", "move", "mut",
    "mutable", "namespace", "new", "nil", "noexcept", "noinline", "nointerpolation",
    "non_coherent", "noncoherent", "noperspective", "null", "nullptr", "of", "operator",
    "package", "packoffset", "partition", "pass", "patch", "pixelfragment", "precise",
    "precision", "premerge", "priv", "protected", "pub", "public", "readonly", "ref",
    "regardless", "register", "reinterpret_cast", "require", "resource", "restrict", "self",
    "set", "shared", "sizeof", "smooth", "snorm", "static", "static_assert", "static_cast",
    "std", "subroutine", "super", "target", "template", "this", "thread_local", "throw",
    "trait", "try", "type", "typedef", "typeid", "typename", "typeof", "union", "unless",
    "unorm", "unsafe", "unsized", "use", "using", "varying", "virtual", "volatile", "wgsl",
    "where", "with", "writeonly", "yield",
    // keywords
    "alias", "break", "case", "const", "const_assert", "continue", "continuing", "default",
    "diagnostic", "discard", "else", "enable", "false", "fn", "for", "if", "let", "loop",
    "override", "requires", "return", "struct", "switch", "true", "var", "while",
];

/// Predeclared names the generated code refers to; a user declaration with one of these names
/// would shadow it, so such declarations are renamed.
#[rustfmt::skip]
const PREDECLARED: &[&str] = &[
    "bool", "f16", "f32", "i32", "u32", "vec2", "vec3", "vec4", "mat2x2", "mat2x3", "mat2x4",
    "mat3x2", "mat3x3", "mat3x4", "mat4x2", "mat4x3", "mat4x4", "array", "atomic", "ptr",
    "sampler", "sampler_comparison", "texture_2d", "texture_2d_array", "texture_cube",
    "texture_depth_2d", "texture_depth_2d_array", "texture_depth_cube", "texture_storage_2d",
    "function", "private", "workgroup", "uniform", "storage", "read", "write", "read_write",
    "rgba8unorm", "rgba16float", "rgba32float", "r32float", "rg16float", "r16float", "bitcast",
    "arrayLength", "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "atan2", "ceil",
    "clamp", "cos", "cosh", "countOneBits", "cross", "degrees", "determinant", "distance",
    "dot", "dpdx", "dpdy", "exp", "exp2", "faceForward", "floor", "fma", "fract", "fwidth",
    "inverseSqrt", "ldexp", "length", "log", "log2", "max", "min", "mix", "normalize", "pow",
    "radians", "reflect", "refract", "reverseBits", "round", "saturate", "sign", "sin", "sinh",
    "smoothstep", "sqrt", "step", "tan", "tanh", "transpose", "trunc", "select", "all", "any",
    "pack4x8snorm", "pack4x8unorm", "pack2x16snorm", "pack2x16unorm", "pack2x16float",
    "unpack4x8snorm", "unpack4x8unorm", "unpack2x16snorm", "unpack2x16unorm", "unpack2x16float",
    "workgroupBarrier", "storageBarrier", "textureSample", "textureSampleBias",
    "textureSampleLevel", "textureSampleGrad", "textureSampleCompare",
    "textureSampleCompareLevel", "textureLoad", "textureStore", "textureDimensions",
    "textureNumLevels", "textureNumLayers",
];

const COMPONENTS: [char; 4] = ['x', 'y', 'z', 'w'];

fn usable(name: &str) -> bool {
    !(name.is_empty()
        || name == "_"
        || name.starts_with("__")
        || RESERVED.contains(&name)
        || PREDECLARED.contains(&name))
}

/// A namespace that hands out unique, valid identifiers.
#[derive(Clone, Default)]
struct Names {
    taken: HashSet<String>,
}
impl Names {
    fn claim(&mut self, base: &str) -> String {
        let base = if usable(base) {
            base.to_string()
        } else if base.is_empty() || base == "_" || base.starts_with("__") {
            format!("v{}", base.trim_start_matches('_'))
        } else {
            format!("{base}_")
        };
        if self.taken.insert(base.clone()) {
            return base;
        }
        (1..)
            .map(|i| format!("{base}_{i}"))
            .find(|c| usable(c) && self.taken.insert(c.clone()))
            .expect("an unused name")
    }
}

fn internal(what: &str) -> Error {
    Error {
        line: 1,
        column: 1,
        message: format!("internal compiler error (WGSL): {what}"),
    }
}

fn scalar_name(s: Scalar) -> &'static str {
    match s.concretize() {
        Scalar::Bool => "bool",
        Scalar::I32 => "i32",
        Scalar::U32 => "u32",
        _ => "f32",
    }
}

fn builtin_name(b: B) -> &'static str {
    match b {
        B::Abs => "abs",
        B::Acos => "acos",
        B::Acosh => "acosh",
        B::Asin => "asin",
        B::Asinh => "asinh",
        B::Atan => "atan",
        B::Atanh => "atanh",
        B::Atan2 => "atan2",
        B::Ceil => "ceil",
        B::Clamp => "clamp",
        B::Cos => "cos",
        B::Cosh => "cosh",
        B::CountOneBits => "countOneBits",
        B::Cross => "cross",
        B::Degrees => "degrees",
        B::Determinant => "determinant",
        B::Distance => "distance",
        B::Dot => "dot",
        B::Dpdx => "dpdx",
        B::Dpdy => "dpdy",
        B::Exp => "exp",
        B::Exp2 => "exp2",
        B::FaceForward => "faceForward",
        B::Floor => "floor",
        B::Fma => "fma",
        B::Fract => "fract",
        B::Fwidth => "fwidth",
        B::InverseSqrt => "inverseSqrt",
        B::Ldexp => "ldexp",
        B::Length => "length",
        B::Log => "log",
        B::Log2 => "log2",
        B::Max => "max",
        B::Min => "min",
        B::Mix => "mix",
        B::Normalize => "normalize",
        B::Pow => "pow",
        B::Radians => "radians",
        B::Reflect => "reflect",
        B::Refract => "refract",
        B::ReverseBits => "reverseBits",
        B::Round => "round",
        B::Saturate => "saturate",
        B::Sign => "sign",
        B::Sin => "sin",
        B::Sinh => "sinh",
        B::Smoothstep => "smoothstep",
        B::Sqrt => "sqrt",
        B::Step => "step",
        B::Tan => "tan",
        B::Tanh => "tanh",
        B::Transpose => "transpose",
        B::Trunc => "trunc",
        B::Select => "select",
        B::All => "all",
        B::Any => "any",
        B::Pack4x8Snorm => "pack4x8snorm",
        B::Pack4x8Unorm => "pack4x8unorm",
        B::Pack2x16Snorm => "pack2x16snorm",
        B::Pack2x16Unorm => "pack2x16unorm",
        B::Pack2x16Float => "pack2x16float",
        B::Unpack4x8Snorm => "unpack4x8snorm",
        B::Unpack4x8Unorm => "unpack4x8unorm",
        B::Unpack2x16Snorm => "unpack2x16snorm",
        B::Unpack2x16Unorm => "unpack2x16unorm",
        B::Unpack2x16Float => "unpack2x16float",
        B::WorkgroupBarrier => "workgroupBarrier",
        B::StorageBarrier => "storageBarrier",
        B::TextureSample => "textureSample",
        B::TextureSampleBias => "textureSampleBias",
        B::TextureSampleLevel => "textureSampleLevel",
        B::TextureSampleGrad => "textureSampleGrad",
        B::TextureSampleCompare => "textureSampleCompare",
        B::TextureSampleCompareLevel => "textureSampleCompareLevel",
        B::TextureLoad => "textureLoad",
        B::TextureStore => "textureStore",
        B::TextureDimensions => "textureDimensions",
        B::TextureNumLevels => "textureNumLevels",
        B::TextureNumLayers => "textureNumLayers",
    }
}

/// A float literal that reads back as exactly `f`.
fn float_lit(f: f32) -> String {
    if !f.is_finite() {
        return format!("bitcast<f32>({}u)", f.to_bits());
    }
    // `{:?}` is the shortest representation that round-trips (`1.0`, `0.1`, `1e-20`).
    let s = format!("{:?}f", f.abs());
    if f.is_sign_negative() {
        format!("(-{s})")
    } else {
        s
    }
}

fn int_lit(v: i64, ty: Scalar) -> String {
    match ty {
        Scalar::U32 => format!("{}u", v as u32),
        _ => {
            let v = v as i32;
            if v == i32::MIN {
                "(-2147483647i - 1i)".to_string()
            } else if v < 0 {
                format!("(-{}i)", -(v as i64))
            } else {
                format!("{v}i")
            }
        }
    }
}

/// A scalar constant spelled as a literal of the concrete scalar type `s`.
fn scalar_lit(v: &ConstValue, s: Scalar) -> String {
    let s = s.concretize();
    match (v, s) {
        (ConstValue::Bool(b), _) => b.to_string(),
        (ConstValue::F32(f), _) => float_lit(*f),
        (ConstValue::AbstractFloat(f), _) => float_lit(*f as f32),
        (ConstValue::I32(i), Scalar::F32) => float_lit(*i as f32),
        (ConstValue::U32(u), Scalar::F32) => float_lit(*u as f32),
        (ConstValue::AbstractInt(i), Scalar::F32) => float_lit(*i as f32),
        (ConstValue::I32(i), s) => int_lit(i64::from(*i), s),
        (ConstValue::U32(u), s) => int_lit(i64::from(*u), s),
        (ConstValue::AbstractInt(i), s) => int_lit(*i, s),
        (ConstValue::Composite(_), _) => "0i".to_string(),
    }
}

fn is_zero(v: &ConstValue) -> bool {
    match v {
        ConstValue::Bool(b) => !b,
        ConstValue::I32(i) => *i == 0,
        ConstValue::U32(u) => *u == 0,
        ConstValue::F32(f) => f.to_bits() == 0,
        ConstValue::AbstractInt(i) => *i == 0,
        ConstValue::AbstractFloat(f) => f.to_bits() == 0,
        ConstValue::Composite(c) => c.iter().all(is_zero),
    }
}

/// Functions reachable from `entry`, the entry first.
fn reachable(prog: &Program, entry: FnId) -> Vec<FnId> {
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    let mut stack = vec![entry];
    while let Some(f) = stack.pop() {
        if seen.insert(f) {
            order.push(f);
            for &c in prog.functions[f].calls.iter().rev() {
                stack.push(c);
            }
        }
    }
    order
}

/// Whether `stmts` contain a `break` of the innermost enclosing loop.
fn breaks(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Break => true,
        Stmt::If {
            then, otherwise, ..
        } => breaks(then) || breaks(otherwise),
        Stmt::Block(b) => breaks(b),
        _ => false,
    })
}

fn contains_current(e: &Expr) -> bool {
    match &e.kind {
        EK::Current => true,
        EK::Load(p) => p
            .steps
            .iter()
            .any(|s| matches!(s, Step::Index(i) if contains_current(i))),
        EK::Extract(b, _)
        | EK::Swizzle(b, _)
        | EK::Unary(_, b)
        | EK::Splat(b)
        | EK::Convert(b)
        | EK::Bitcast(b) => contains_current(b),
        EK::DynIndex(a, b) | EK::Binary(_, a, b) => contains_current(a) || contains_current(b),
        EK::Construct(v) | EK::Call(_, v) | EK::Builtin(_, v) => v.iter().any(contains_current),
        EK::Texture(t) => [
            &t.coords,
            &t.array_index,
            &t.level,
            &t.bias,
            &t.ddx,
            &t.ddy,
            &t.depth_ref,
            &t.offset,
            &t.value,
        ]
        .into_iter()
        .flatten()
        .any(contains_current),
        _ => false,
    }
}

#[derive(Default)]
struct FnCx {
    names: Names,
    values: HashMap<ValueId, String>,
    locals: Vec<String>,
    /// Spelling of `ExprKind::Current` while emitting a compound assignment.
    current: Option<String>,
    ret: Option<Ty>,
    indent: usize,
    in_continuing: bool,
}

struct Emitter<'p> {
    prog: &'p Program,
    out: String,
    module: Names,
    structs: Vec<String>,
    members: Vec<Vec<String>>,
    globals: HashMap<GlobalId, String>,
    fns: HashMap<FnId, String>,
    f: FnCx,
    current_fn: Option<FnId>,
}

/// WGSL for `entry`. The caller has already compiled the entry point to SPIR-V, which checked
/// the stage, the reachable built-ins and the resource interface.
pub(crate) fn emit(prog: &Program, stage: Stage, entry: &str) -> R<Wgsl> {
    let fid = prog
        .functions
        .iter()
        .position(|f| f.name == entry && f.stage == Some(stage))
        .ok_or_else(|| internal("entry point not found"))?;
    let reach = reachable(prog, fid);
    let mut used: Vec<GlobalId> = reach
        .iter()
        .flat_map(|&r| prog.functions[r].globals.iter().copied())
        .collect();
    used.sort_unstable();
    used.dedup();

    let mut push = None;
    for &g in &used {
        let gl = &prog.globals[g];
        if gl.kind == GlobalKind::PushConstant {
            push = Some(g);
            uniform_layout(&gl.ty, &prog.structs).map_err(|m| {
                gl.span.error(format!(
                    "push constant block '{}' cannot become a WebGPU uniform buffer: {m}",
                    gl.name
                ))
            })?;
        }
    }
    if push.is_some() {
        for &g in &used {
            let gl = &prog.globals[g];
            if let Some((group, binding)) = gl.binding
                && group == PUSH_CONSTANT_GROUP
            {
                return Err(gl.span.error(format!(
                    "'{}' uses @group({group}) @binding({binding}), but on WebGPU @group({PUSH_CONSTANT_GROUP}) holds the push constants of '{entry}'",
                    gl.name
                )));
            }
        }
    }

    let mut e = Emitter {
        prog,
        out: String::new(),
        module: Names::default(),
        structs: Vec::new(),
        members: Vec::new(),
        globals: HashMap::new(),
        fns: HashMap::new(),
        f: FnCx::default(),
        current_fn: None,
    };
    for s in &prog.structs {
        let name = e.module.claim(&s.name);
        e.structs.push(name);
        let mut member_names = Names::default();
        e.members.push(
            s.members
                .iter()
                .map(|m| member_names.claim(&m.name))
                .collect(),
        );
    }
    for &g in &used {
        let name = e.module.claim(&prog.globals[g].name);
        e.globals.insert(g, name);
    }
    for &r in &reach {
        let name = e.module.claim(&prog.functions[r].name);
        e.fns.insert(r, name);
    }

    let _ = writeln!(
        e.out,
        "// Generated by ganbate-shader from '{entry}' ({stage:?}) for WebGPU.\n"
    );
    for sid in 0..prog.structs.len() {
        e.struct_decl(sid);
    }
    for &g in &used {
        e.global_decl(g);
    }
    // Callees before callers reads naturally; WGSL accepts any order.
    for &r in reach.iter().rev() {
        e.function(r)?;
    }
    let entry_name = e.wrapper(fid, stage)?;
    Ok(Wgsl {
        source: e.out,
        entry: entry_name,
        push_constants: push.map(|_| (PUSH_CONSTANT_GROUP, PUSH_CONSTANT_BINDING)),
    })
}

/// WGSL's layout constraints for the uniform address space.
fn uniform_layout(ty: &Ty, structs: &[StructInfo]) -> Result<(), String> {
    match ty {
        Ty::Array(e, _) => {
            let stride = array_stride(e, structs);
            if stride % 16 != 0 {
                return Err(format!(
                    "uniform arrays need a 16-byte element stride, but {} has stride {stride}",
                    type_name(ty, structs)
                ));
            }
            uniform_layout(e, structs)
        }
        Ty::Struct(id) => {
            let s = &structs[*id];
            let layout = s.layout.as_ref().ok_or("the type is not host-shareable")?;
            for (i, m) in s.members.iter().enumerate() {
                let off = layout.offsets[i];
                if matches!(m.ty, Ty::Struct(_) | Ty::Array(..)) && off % 16 != 0 {
                    return Err(format!(
                        "member '{}' of '{}' must be 16-byte aligned (it is at offset {off})",
                        m.name, s.name
                    ));
                }
                if let Ty::Struct(_) = m.ty
                    && let Some(&next) = layout.offsets.get(i + 1)
                    && next - off < round_up(16, size_of(&m.ty, structs))
                {
                    return Err(format!(
                        "member '{}' of '{}' must start on a 16-byte boundary after struct member '{}'",
                        s.members[i + 1].name,
                        s.name,
                        m.name
                    ));
                }
                uniform_layout(&m.ty, structs)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

impl Emitter<'_> {
    fn ty(&self, t: &Ty) -> String {
        match t {
            Ty::Scalar(s) => scalar_name(*s).to_string(),
            Ty::Vector(n, s) => format!("vec{n}<{}>", scalar_name(*s)),
            Ty::Matrix(c, r) => format!("mat{c}x{r}<f32>"),
            Ty::Array(e, Some(n)) => format!("array<{}, {n}>", self.ty(e)),
            Ty::Array(e, None) => format!("array<{}>", self.ty(e)),
            Ty::Struct(id) => self.structs[*id].clone(),
            Ty::Void => "void".to_string(),
            other => type_name(other, &self.prog.structs),
        }
    }

    /// A constant of type `ty` as a WGSL const-expression.
    fn konst(&self, ty: &Ty, v: &ConstValue) -> String {
        if let Some(s) = ty.scalar()
            && ty.is_scalar()
        {
            return scalar_lit(v, s);
        }
        if is_zero(v) && !ty.is_scalar() {
            return format!("{}()", self.ty(ty));
        }
        let parts = v.components();
        let items: Vec<String> = parts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let pt = match ty {
                    Ty::Vector(_, s) => Ty::Scalar(*s),
                    Ty::Matrix(_, r) => Ty::Vector(*r, Scalar::F32),
                    Ty::Array(e, _) => (**e).clone(),
                    Ty::Struct(sid) => self.prog.structs[*sid].members[i].ty.clone(),
                    other => other.clone(),
                };
                self.konst(&pt, p)
            })
            .collect();
        format!("{}({})", self.ty(ty), items.join(", "))
    }

    fn line(&mut self, text: &str) {
        for _ in 0..self.f.indent {
            self.out.push_str("  ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn struct_decl(&mut self, sid: usize) {
        let s = &self.prog.structs[sid];
        let _ = writeln!(self.out, "struct {} {{", self.structs[sid]);
        for (i, m) in s.members.iter().enumerate() {
            let mut attrs = String::new();
            if let Some(a) = m.align_attr {
                let _ = write!(attrs, "@align({a}) ");
            }
            if let Some(z) = m.size_attr {
                let _ = write!(attrs, "@size({z}) ");
            }
            let _ = writeln!(
                self.out,
                "  {attrs}{}: {},",
                self.members[sid][i],
                self.ty(&m.ty)
            );
        }
        self.out.push_str("};\n\n");
    }

    fn global_decl(&mut self, g: GlobalId) {
        let gl = &self.prog.globals[g];
        let name = &self.globals[&g];
        let ty = self.ty(&gl.ty);
        let (group, binding) = gl.binding.unwrap_or((0, 0));
        let decl = match gl.kind {
            GlobalKind::Private | GlobalKind::ConstData => match &gl.init {
                Some(v) => format!("var<private> {name}: {ty} = {};", self.konst(&gl.ty, v)),
                None => format!("var<private> {name}: {ty};"),
            },
            GlobalKind::Uniform => {
                format!("@group({group}) @binding({binding}) var<uniform> {name}: {ty};")
            }
            GlobalKind::Storage { read_only } => format!(
                "@group({group}) @binding({binding}) var<storage, {}> {name}: {ty};",
                if read_only { "read" } else { "read_write" }
            ),
            GlobalKind::PushConstant => format!(
                "// Ganbate push constants, bound with a dynamic offset.\n@group({PUSH_CONSTANT_GROUP}) @binding({PUSH_CONSTANT_BINDING}) var<uniform> {name}: {ty};"
            ),
            GlobalKind::Handle => format!("@group({group}) @binding({binding}) var {name}: {ty};"),
        };
        self.out.push_str(&decl);
        self.out.push_str("\n\n");
    }

    fn function(&mut self, fid: FnId) -> R<()> {
        let prog = self.prog;
        let f = &prog.functions[fid];
        self.current_fn = Some(fid);
        self.f = FnCx {
            names: self.module.clone(),
            ret: f.ret.clone(),
            indent: 1,
            ..FnCx::default()
        };
        let mut params = Vec::new();
        for p in &f.params {
            let name = self.f.names.claim(&p.name);
            params.push(format!("{name}: {}", self.ty(&p.ty)));
            self.f.values.insert(p.value, name);
        }
        let ret = match &f.ret {
            Some(t) => format!(" -> {}", self.ty(t)),
            None => String::new(),
        };
        let _ = writeln!(
            self.out,
            "fn {}({}){ret} {{",
            self.fns[&fid],
            params.join(", ")
        );
        for l in &f.locals {
            let name = self.f.names.claim(&l.name);
            let ty = self.ty(&l.ty);
            self.line(&format!("var {name}: {ty};"));
            self.f.locals.push(name);
        }
        self.block(&f.body)?;
        self.out.push_str("}\n\n");
        Ok(())
    }

    // ------------------------------------------------------------------ statements

    /// Emits `stmts`; returns true when control cannot fall off the end. Statements after a
    /// terminator are unreachable and left out (the SPIR-V emitter does the same).
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
                let name = self.value_name(*id);
                self.line(&format!("let {name} = {v};"));
                self.f.values.insert(*id, name);
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
                if statement {
                    self.line(&format!("{v};"));
                } else if e.ty != Ty::Void {
                    self.line(&format!("_ = {v};"));
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
                self.line("loop {");
                self.f.indent += 1;
                if let Some(c) = cond {
                    let c = self.expr(c)?;
                    self.line(&format!("if (!({c})) {{ break; }}"));
                }
                self.block(body)?;
                if !continuing.is_empty() || break_if.is_some() {
                    self.line("continuing {");
                    self.f.indent += 1;
                    let saved = std::mem::replace(&mut self.f.in_continuing, true);
                    self.block(continuing)?;
                    if let Some(b) = break_if {
                        let b = self.expr(b)?;
                        self.line(&format!("break if {b};"));
                    }
                    self.f.in_continuing = saved;
                    self.f.indent -= 1;
                    self.line("}");
                }
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
                if self.f.in_continuing {
                    return Ok(false);
                }
                // Keep Ganbate's terminate-invocation semantics.
                match self.f.ret.clone() {
                    Some(t) => {
                        let z = format!("{}()", self.ty(&t));
                        self.line(&format!("return {z};"));
                    }
                    None => self.line("return;"),
                }
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

    /// Stores `value` into `place` (or into components `swizzle` of the vector at `place`).
    /// Dynamic indices of the target are evaluated once, before the value, as in SPIR-V.
    fn assign(&mut self, place: &Place, swizzle: Option<&[u32]>, value: &Expr) -> R<()> {
        let twice = contains_current(value) || swizzle.is_some_and(|c| c.len() > 1);
        let target = self.place(place, twice)?;
        let comps: Option<String> = swizzle.map(|c| {
            c.iter()
                .map(|&i| COMPONENTS[i as usize])
                .collect::<String>()
        });
        let current = match &comps {
            Some(c) => format!("{target}.{c}"),
            None => target.clone(),
        };
        let saved = self.f.current.replace(current.clone());
        let v = self.expr(value);
        self.f.current = saved;
        let v = v?;
        match swizzle {
            None => self.line(&format!("{target} = {v};")),
            Some([c]) => self.line(&format!("{target}.{} = {v};", COMPONENTS[*c as usize])),
            Some(cs) => {
                let tmp = self.f.names.claim("swizzle");
                self.line(&format!("let {tmp} = {v};"));
                for (i, &c) in cs.iter().enumerate() {
                    self.line(&format!(
                        "{target}.{} = {tmp}.{};",
                        COMPONENTS[c as usize], COMPONENTS[i]
                    ));
                }
            }
        }
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

    /// The reference expression for `p`. With `hoist`, non-constant indices are first bound
    /// to `let`s so the reference can be spelled twice without evaluating them twice.
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
                        Some(_) => {
                            let i = self.expr(e)?;
                            let _ = write!(s, "[{i}]");
                        }
                        None => {
                            let i = self.expr(e)?;
                            if hoist && !matches!(e.kind, EK::Value(_)) {
                                let tmp = self.f.names.claim("index");
                                self.line(&format!("let {tmp} = {i};"));
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

    fn args(&mut self, args: &[Expr]) -> R<String> {
        let mut out = Vec::with_capacity(args.len());
        for a in args {
            out.push(self.expr(a)?);
        }
        Ok(out.join(", "))
    }

    fn expr(&mut self, e: &Expr) -> R<String> {
        Ok(match &e.kind {
            EK::Const(c) => self.konst(&e.ty, c),
            EK::Zero => format!("{}()", self.ty(&e.ty)),
            EK::Value(v) => self
                .f
                .values
                .get(v)
                .cloned()
                .ok_or_else(|| internal("value used before its definition"))?,
            EK::Load(p) => self.place(p, false)?,
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
                let x = self.expr(x)?;
                match op {
                    UnaryOp::Neg => format!("(-{x})"),
                    UnaryOp::Not => format!("(!{x})"),
                    UnaryOp::BitNot => format!("(~{x})"),
                }
            }
            EK::Binary(op, l, r) => {
                let l = self.expr(l)?;
                let r = self.expr(r)?;
                format!("({l} {} {r})", binary_symbol(*op))
            }
            EK::Construct(parts) => format!("{}({})", self.ty(&e.ty), self.args(parts)?),
            EK::Splat(x) | EK::Convert(x) => {
                let x = self.expr(x)?;
                format!("{}({x})", self.ty(&e.ty))
            }
            EK::Bitcast(x) => {
                let x = self.expr(x)?;
                format!("bitcast<{}>({x})", self.ty(&e.ty))
            }
            EK::Call(f, args) => {
                let name = self
                    .fns
                    .get(f)
                    .cloned()
                    .ok_or_else(|| internal("callee not emitted"))?;
                format!("{name}({})", self.args(args)?)
            }
            EK::Builtin(b, args) => format!("{}({})", builtin_name(*b), self.args(args)?),
            EK::Texture(tc) => self.texture(tc)?,
            EK::ArrayLength { global, member } => {
                let g = self
                    .globals
                    .get(global)
                    .cloned()
                    .ok_or_else(|| internal("arrayLength of an undeclared global"))?;
                match (member, &self.prog.globals[*global].ty) {
                    (Some(m), Ty::Struct(sid)) => {
                        format!("arrayLength(&{g}.{})", self.members[*sid][*m as usize])
                    }
                    _ => format!("arrayLength(&{g})"),
                }
            }
            EK::Current => self
                .f
                .current
                .clone()
                .ok_or_else(|| internal("compound assignment value missing"))?,
        })
    }

    fn texture(&mut self, tc: &TexCall) -> R<String> {
        let mut args = vec![
            self.globals
                .get(&tc.texture)
                .cloned()
                .ok_or_else(|| internal("texture was not declared"))?,
        ];
        if let Some(s) = tc.sampler {
            args.push(
                self.globals
                    .get(&s)
                    .cloned()
                    .ok_or_else(|| internal("sampler was not declared"))?,
            );
        }
        // WGSL argument order for every texture built-in Ganbate supports.
        let order = [
            &tc.coords,
            &tc.array_index,
            &tc.bias,
            &tc.level,
            &tc.ddx,
            &tc.ddy,
            &tc.depth_ref,
            &tc.offset,
            &tc.value,
        ];
        for a in order.into_iter().flatten() {
            args.push(self.expr(a)?);
        }
        Ok(format!("{}({})", builtin_name(tc.op), args.join(", ")))
    }
}

fn binary_symbol(op: BinaryOp) -> &'static str {
    op.symbol()
}

/// The IO attributes of one wrapper input or output.
fn io_attr(stage: Stage, input: bool, ty: &Ty, io: &IoAttrs) -> String {
    if let Some(b) = io.builtin {
        let invariant = if io.invariant { "@invariant " } else { "" };
        return format!("{invariant}@builtin({})", b.name());
    }
    let mut s = format!("@location({})", io.location.unwrap_or(0));
    let varying = (stage == Stage::Vertex && !input) || (stage == Stage::Fragment && input);
    if varying {
        let int = ty.scalar().is_some_and(|x| x.is_int());
        if int || io.interpolation == Some(Interpolation::Flat) {
            s.push_str(" @interpolate(flat)");
        } else {
            let linear = io.interpolation == Some(Interpolation::Linear);
            let sampling = match io.sampling {
                Some(Sampling::Centroid) => ", centroid",
                Some(Sampling::Sample) => ", sample",
                _ => "",
            };
            if linear || !sampling.is_empty() {
                let kind = if linear { "linear" } else { "perspective" };
                let _ = write!(s, " @interpolate({kind}{sampling})");
            }
        }
    }
    s
}

/// Inputs of the generated entry point: one parameter per location and per built-in.
struct WrapperInputs {
    names: Names,
    params: Vec<String>,
    builtins: HashMap<BuiltinValue, String>,
}

impl Emitter<'_> {
    fn wrapper_input(
        &self,
        stage: Stage,
        w: &mut WrapperInputs,
        io: &IoAttrs,
        ty: &Ty,
        hint: &str,
    ) -> String {
        if let Some(b) = io.builtin
            && let Some(n) = w.builtins.get(&b)
        {
            return n.clone();
        }
        let n = w.names.claim(hint);
        w.params.push(format!(
            "{} {n}: {}",
            io_attr(stage, true, ty, io),
            self.ty(ty)
        ));
        if let Some(b) = io.builtin {
            w.builtins.insert(b, n.clone());
        }
        n
    }

    /// The generated entry point: gathers the inputs, calls the user's entry function and
    /// returns its outputs (with the vertex position's y negated for WebGPU's clip space).
    fn wrapper(&mut self, fid: FnId, stage: Stage) -> R<String> {
        let prog = self.prog;
        let f = &prog.functions[fid];
        let entry = self.module.claim("main");
        let out_struct = match (&f.ret, f.ret_io.is_empty()) {
            (Some(Ty::Struct(_)), true) => Some(self.module.claim(&format!("{}_outputs", f.name))),
            _ => None,
        };
        let mut w = WrapperInputs {
            names: self.module.clone(),
            params: Vec::new(),
            builtins: HashMap::new(),
        };
        let mut args = Vec::new();
        for p in &f.params {
            if !p.io.is_empty() {
                args.push(self.wrapper_input(stage, &mut w, &p.io, &p.ty, &p.name));
            } else if let Ty::Struct(sid) = p.ty {
                let mut parts = Vec::new();
                for m in &prog.structs[sid].members {
                    parts.push(self.wrapper_input(stage, &mut w, &m.io, &m.ty, &m.name));
                }
                args.push(format!("{}({})", self.structs[sid], parts.join(", ")));
            } else {
                return Err(internal("entry-point parameter without IO attributes"));
            }
        }
        let call = format!("{}({})", self.fns[&fid], args.join(", "));
        let flip = |v: String, io: &IoAttrs| {
            if stage == Stage::Vertex && io.builtin == Some(BuiltinValue::Position) {
                format!("({v} * vec4<f32>(1.0f, -1.0f, 1.0f, 1.0f))")
            } else {
                v
            }
        };
        let mut ret = String::new();
        let mut body = Vec::new();
        match &f.ret {
            None => body.push(format!("{call};")),
            Some(ty) if !f.ret_io.is_empty() => {
                ret = format!(
                    " -> {} {}",
                    io_attr(stage, false, ty, &f.ret_io),
                    self.ty(ty)
                );
                body.push(format!("return {};", flip(call, &f.ret_io)));
            }
            Some(Ty::Struct(sid)) => {
                let out = out_struct.ok_or_else(|| internal("output struct name"))?;
                let mut member_names = Names::default();
                let r = w.names.claim("outputs");
                let mut decl = format!("struct {out} {{\n");
                let mut fields = Vec::new();
                for (i, m) in prog.structs[*sid].members.iter().enumerate() {
                    let _ = writeln!(
                        decl,
                        "  {} {}: {},",
                        io_attr(stage, false, &m.ty, &m.io),
                        member_names.claim(&m.name),
                        self.ty(&m.ty)
                    );
                    fields.push(flip(format!("{r}.{}", self.members[*sid][i]), &m.io));
                }
                decl.push_str("};\n\n");
                self.out.push_str(&decl);
                ret = format!(" -> {out}");
                body.push(format!("let {r} = {call};"));
                body.push(format!("return {out}({});", fields.join(", ")));
            }
            Some(_) => return Err(internal("entry-point output without IO attributes")),
        }
        let stage_attr = match stage {
            Stage::Vertex => "@vertex".to_string(),
            Stage::Fragment => "@fragment".to_string(),
            Stage::Compute => {
                let [x, y, z] = f.workgroup_size;
                format!("@compute @workgroup_size({x}, {y}, {z})")
            }
        };
        let _ = writeln!(
            self.out,
            "{stage_attr}\nfn {entry}({}){ret} {{",
            w.params.join(", ")
        );
        for b in body {
            let _ = writeln!(self.out, "  {b}");
        }
        self.out.push_str("}\n");
        Ok(entry)
    }
}
