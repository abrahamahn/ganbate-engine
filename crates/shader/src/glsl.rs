//! GLSL ES 3.00 for WebGL 2, generated from the same checked IR as the SPIR-V of an entry point.
//!
//! WebGL 2 has no bind groups, separate samplers, storage buffers, push constants or compute
//! shaders, and its clip space differs from Ganbate's (Vulkan's). The output keeps the engine's
//! conventions through these lowerings, which the WebGL backend of `ganbate_engine::gpu` follows:
//!
//! * **Uniform buffers** become `layout(std140) uniform ganbate_ubo_G_B { T name; };`. The WGSL
//!   uniform layout of `T` must equal its std140 layout, which WGSL's uniform rules already
//!   guarantee except for explicit `@size`/`@align` gaps (checked here).
//! * **Push constants** become the block `ganbate_push`; the backend fills it from a ring buffer.
//! * **Textures and samplers** pair up. Every (texture, sampler) combination the entry point
//!   samples with is one combined sampler uniform `ganbate_tex_G_B_G_B` (texture binding first);
//!   a texture read without a sampler (`textureLoad`, `textureDimensions`) is `ganbate_tex_G_B`.
//!   [`Glsl::units`] lists them; the backend binds the texture and a sampler object to one
//!   texture unit each. A depth texture sampled through a comparison sampler is a `*Shadow`
//!   sampler.
//! * **Storage buffers** must be read-only. Each is an RGBA32UI texture `ganbate_buf_G_B`,
//!   [`STORAGE_ROW_TEXELS`] texels wide, holding the buffer's bytes in order; loads become
//!   texel fetches at the byte offset WGSL's storage layout gives, plus the hidden uniform
//!   `ganbate_buf_G_B_offset` (the binding's offset in bytes).
//! * **Clip space:** the vertex wrapper remaps depth (`z = 2z - w`), so the depth buffer holds
//!   the values Vulkan would write. Y is not flipped: images lie in memory as on Vulkan (row 0
//!   at the top), `gl_FragCoord` equals Vulkan's `FragCoord`, and triangles that are front-facing
//!   on Vulkan wind clockwise in GL's terms (the backend sets `frontFace` accordingly). The
//!   backend flips the final image when it presents it to the canvas.
//! * **Built-ins** WebGL 2 lacks become hidden uniforms the backend sets per draw:
//!   `ganbate_base_vertex` and `ganbate_base_instance` (added to `vertex_index` / `instance_index`),
//!   `ganbate_levels_G_B` (`textureNumLevels`) and `ganbate_buf_G_B_size` (`arrayLength`). Functions
//!   GLSL ES 3.00 lacks (`fma`, `ldexp`, `countOneBits`, `reverseBits`, 4x8 packing, integer
//!   `dot`, WGSL's truncating `%`, bool `&`/`|` that evaluate both operands) are generated.
//! * **Interface:** vertex inputs are `ganbate_aL` and fragment outputs `ganbate_oL`, both with
//!   `layout(location = L)`; varyings are matched by name between the stages, `ganbate_vL` for
//!   `@location(L)`, flat for integers and `@interpolate(flat)`.
//! * Loops become `while (true)` with the continuing block at the top, guarded by a first-pass
//!   flag; `let`s of the loop body that the continuing block reads are declared before the loop.
//! * Module-scope names are kept unless GLSL reserves them (keywords, built-in functions, the
//!   `gl_`, `webgl_` and `ganbate_` prefixes, double underscores); locals get unique names.
//!
//! * `@interpolate(linear)` varyings carry `value * w`, divided in the fragment shader by the
//!   interpolated clip w (`ganbate_vw`): GLSL ES 3.00 has no `noperspective`.
//!
//! Not available on WebGL 2 (compile errors): compute entry points, storage writes, storage
//! textures, workgroup barriers, `sample_index`, per-sample interpolation and arrays of arrays.
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::ast::BinaryOp;
use crate::ir::{
    Builtin as B, Expr, ExprKind as EK, FnId, GlobalId, GlobalKind, Place, PlaceRoot, Program,
    Step, Stmt, TexCall, UnaryOp, ValueId,
};
use crate::types::*;
use crate::{Error, SampleType, Stage, TextureView};

/// Texels per row of the RGBA32UI textures that carry storage buffers (16 bytes each).
pub const STORAGE_ROW_TEXELS: u32 = 1024;

/// GLSL ES output for one entry point, produced by [`crate::compile_glsl`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glsl {
    /// A complete `#version 300 es` shader.
    pub source: String,
    /// The uniform blocks the entry point reads.
    pub blocks: Vec<Block>,
    /// Combined texture/sampler uniforms, one texture unit each.
    pub units: Vec<Unit>,
    /// Storage buffers, read through RGBA32UI textures (one texture unit each).
    pub storage: Vec<Storage>,
    /// Plain uniforms the backend sets per draw.
    pub hidden: Vec<Hidden>,
}

/// A `layout(std140) uniform` block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub name: String,
    /// `(group, binding)` of the WGSL uniform, or `None` for the push constants.
    pub binding: Option<(u32, u32)>,
    /// WGSL size of the block's type in bytes.
    pub size: u32,
}

/// A combined sampler uniform: one texture unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    pub name: String,
    /// `(group, binding)` of the texture.
    pub texture: (u32, u32),
    /// `(group, binding)` of the sampler, or `None` for unfiltered reads (`texelFetch`,
    /// `textureSize`), which the backend pairs with a nearest, non-comparison sampler.
    pub sampler: Option<(u32, u32)>,
    pub view: TextureView,
    pub sample: SampleType,
    /// A `*Shadow` sampler: the bound sampler compares depth.
    pub compare: bool,
}

/// A read-only storage buffer and the RGBA32UI texture uniform that carries it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Storage {
    pub name: String,
    pub binding: (u32, u32),
}

/// A value the backend supplies through a plain uniform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HiddenValue {
    /// `int`: the draw's base vertex (`vertex_index` adds it to `gl_VertexID`).
    BaseVertex,
    /// `int`: the draw's first instance (`instance_index` adds it to `gl_InstanceID`).
    BaseInstance,
    /// `uint`: the byte offset of the storage binding `(group, binding)`.
    StorageOffset(u32, u32),
    /// `uint`: the byte size of the storage binding `(group, binding)`.
    StorageSize(u32, u32),
    /// `uint`: the mip levels the texture binding `(group, binding)` views.
    TextureLevels(u32, u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hidden {
    pub name: String,
    pub value: HiddenValue,
}

impl HiddenValue {
    /// GLSL type of the uniform.
    pub fn glsl_type(self) -> &'static str {
        match self {
            HiddenValue::BaseVertex | HiddenValue::BaseInstance => "int",
            _ => "uint",
        }
    }
}

type R<T> = Result<T, Error>;

/// GLSL ES 3.00 keywords, reserved words and built-in names, plus desktop GLSL built-ins some
/// compilers also reserve, and names the generated code relies on.
#[rustfmt::skip]
const RESERVED: &[&str] = &[
    // keywords
    "const", "uniform", "layout", "centroid", "flat", "smooth", "break", "continue", "do", "for",
    "while", "switch", "case", "default", "if", "else", "in", "out", "inout", "float", "int",
    "void", "bool", "true", "false", "invariant", "discard", "return", "mat2", "mat3", "mat4",
    "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3", "mat3x4", "mat4x2", "mat4x3", "mat4x4",
    "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "bvec2", "bvec3", "bvec4", "uint",
    "uvec2", "uvec3", "uvec4", "lowp", "mediump", "highp", "precision", "sampler2D",
    "sampler3D", "samplerCube", "sampler2DShadow", "samplerCubeShadow", "sampler2DArray",
    "sampler2DArrayShadow", "isampler2D", "isampler3D", "isamplerCube", "isampler2DArray",
    "usampler2D", "usampler3D", "usamplerCube", "usampler2DArray", "struct",
    // reserved for future use
    "attribute", "varying", "coherent", "volatile", "restrict", "readonly", "writeonly",
    "resource", "atomic_uint", "noperspective", "patch", "sample", "subroutine", "common",
    "partition", "active", "asm", "class", "union", "enum", "typedef", "template", "this",
    "goto", "inline", "noinline", "public", "static", "extern", "external", "interface",
    "long", "short", "double", "half", "fixed", "unsigned", "superp", "input", "output",
    "hvec2", "hvec3", "hvec4", "dvec2", "dvec3", "dvec4", "fvec2", "fvec3", "fvec4",
    "sampler3DRect", "filter", "image1D", "image2D", "image3D", "imageCube", "iimage1D",
    "iimage2D", "iimage3D", "iimageCube", "uimage1D", "uimage2D", "uimage3D", "uimageCube",
    "image1DArray", "image2DArray", "iimage1DArray", "iimage2DArray", "uimage1DArray",
    "uimage2DArray", "imageBuffer", "iimageBuffer", "uimageBuffer", "sampler1D",
    "sampler1DShadow", "sampler1DArray", "sampler1DArrayShadow", "isampler1D",
    "isampler1DArray", "usampler1D", "usampler1DArray", "sampler2DRect", "sampler2DRectShadow",
    "isampler2DRect", "usampler2DRect", "samplerBuffer", "isamplerBuffer", "usamplerBuffer",
    "sampler2DMS", "isampler2DMS", "usampler2DMS", "sampler2DMSArray", "isampler2DMSArray",
    "usampler2DMSArray", "sizeof", "cast", "namespace", "using", "buffer", "shared",
    "precise", "subroutine", "dmat2", "dmat3", "dmat4", "samplerExternalOES",
    // built-in functions
    "radians", "degrees", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh", "pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
    "abs", "sign", "floor", "trunc", "round", "roundEven", "ceil", "fract", "mod", "modf",
    "min", "max", "clamp", "mix", "step", "smoothstep", "isnan", "isinf", "floatBitsToInt",
    "floatBitsToUint", "intBitsToFloat", "uintBitsToFloat", "packSnorm2x16",
    "unpackSnorm2x16", "packUnorm2x16", "unpackUnorm2x16", "packHalf2x16", "unpackHalf2x16",
    "length", "distance", "dot", "cross", "normalize", "faceforward", "reflect", "refract",
    "matrixCompMult", "outerProduct", "transpose", "determinant", "inverse", "lessThan",
    "lessThanEqual", "greaterThan", "greaterThanEqual", "equal", "notEqual", "any", "all",
    "not", "textureSize", "texture", "textureProj", "textureLod", "textureOffset", "texelFetch",
    "texelFetchOffset", "textureProjOffset", "textureLodOffset", "textureProjLod",
    "textureProjLodOffset", "textureGrad", "textureGradOffset", "textureProjGrad",
    "textureProjGradOffset", "dFdx", "dFdy", "fwidth", "main",
    // later or desktop built-ins
    "bitCount", "bitfieldReverse", "bitfieldExtract", "bitfieldInsert", "findLSB", "findMSB",
    "ldexp", "frexp", "fma", "packUnorm4x8", "packSnorm4x8", "unpackUnorm4x8",
    "unpackSnorm4x8", "uaddCarry", "usubBorrow", "umulExtended", "imulExtended", "noise1",
    "noise2", "noise3", "noise4", "texture2D", "texture2DLod", "texture2DProj", "textureCube",
    "textureCubeLod", "shadow2D", "texture3D", "barrier", "memoryBarrier", "imageLoad",
    "imageStore", "textureGather", "textureGatherOffset", "textureQueryLevels",
    "textureQueryLod", "interpolateAtCentroid", "interpolateAtSample", "interpolateAtOffset",
    "dFdxFine", "dFdyFine", "dFdxCoarse", "dFdyCoarse", "fwidthFine", "fwidthCoarse",
    "EmitVertex", "EndPrimitive", "atomicAdd", "atomicCounter",
];

const COMPONENTS: [char; 4] = ['x', 'y', 'z', 'w'];

fn usable(name: &str) -> bool {
    !(name.is_empty()
        || name == "_"
        || name.contains("__")
        || name.starts_with("gl_")
        || name.starts_with("webgl_")
        || name.starts_with("_webgl_")
        || name.starts_with("ganbate_")
        || RESERVED.contains(&name))
}

/// A namespace that hands out unique, valid identifiers.
#[derive(Clone, Default)]
struct Names {
    taken: HashSet<String>,
}
impl Names {
    fn claim(&mut self, base: &str) -> String {
        let mut base = base.to_string();
        while base.contains("__") {
            base = base.replace("__", "_");
        }
        if base.is_empty() || base == "_" {
            base = "v".into();
        }
        if base.starts_with("gl_")
            || base.starts_with("webgl_")
            || base.starts_with("_webgl_")
            || base.starts_with("ganbate_")
        {
            base = format!("u{}{base}", if base.starts_with('_') { "" } else { "_" });
        }
        if RESERVED.contains(&base.as_str()) {
            base.push('_');
        }
        if usable(&base) && self.taken.insert(base.clone()) {
            return base;
        }
        let sep = if base.ends_with('_') { "" } else { "_" };
        (1..)
            .map(|i| format!("{base}{sep}{i}"))
            .find(|c| usable(c) && self.taken.insert(c.clone()))
            .expect("an unused name")
    }
    /// A unique generated name (generated names may use the `ganbate_` prefix).
    fn generated(&mut self, base: &str) -> String {
        if self.taken.insert(base.to_string()) {
            return base.to_string();
        }
        (1..)
            .map(|i| format!("{base}{i}"))
            .find(|c| self.taken.insert(c.clone()))
            .expect("an unused name")
    }
}

fn internal(what: &str) -> Error {
    Error {
        line: 1,
        column: 1,
        message: format!("internal compiler error (GLSL): {what}"),
    }
}

fn unsupported(span: &crate::lexer::Span, what: &str) -> Error {
    span.error(format!("{what} is not available on WebGL 2 (GLSL ES 3.00)"))
}

/// A float literal that reads back as exactly `f`.
fn float_lit(f: f32) -> String {
    if f.is_nan() {
        return "uintBitsToFloat(0x7fc00000u)".into();
    }
    if f.is_infinite() {
        let bits = if f > 0.0 {
            "0x7f800000u"
        } else {
            "0xff800000u"
        };
        return format!("uintBitsToFloat({bits})");
    }
    // `{:?}` is the shortest representation that round-trips (`1.0`, `0.1`, `1e-20`).
    let s = format!("{:?}", f.abs());
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
                "(-2147483647 - 1)".to_string()
            } else if v < 0 {
                format!("(-{})", -(v as i64))
            } else {
                format!("{v}")
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
        (ConstValue::Composite(_), _) => "0".to_string(),
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

fn scalar_name(s: Scalar) -> &'static str {
    match s.concretize() {
        Scalar::Bool => "bool",
        Scalar::I32 => "int",
        Scalar::U32 => "uint",
        _ => "float",
    }
}

fn vector_name(n: u8, s: Scalar) -> String {
    let p = match s.concretize() {
        Scalar::Bool => "b",
        Scalar::I32 => "i",
        Scalar::U32 => "u",
        _ => "",
    };
    format!("{p}vec{n}")
}

fn matrix_name(c: u8, r: u8) -> String {
    if c == r {
        format!("mat{c}")
    } else {
        format!("mat{c}x{r}")
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

fn tex_operands(t: &TexCall) -> impl Iterator<Item = &Expr> {
    [
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
}

/// Calls `f` on every expression in `e`, including those in place indices.
fn walk(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    match &e.kind {
        EK::Load(p) => {
            for s in &p.steps {
                if let Step::Index(i) = s {
                    walk(i, f);
                }
            }
        }
        EK::Extract(b, _)
        | EK::Swizzle(b, _)
        | EK::Unary(_, b)
        | EK::Splat(b)
        | EK::Convert(b)
        | EK::Bitcast(b) => walk(b, f),
        EK::DynIndex(a, b) | EK::Binary(_, a, b) => {
            walk(a, f);
            walk(b, f);
        }
        EK::Construct(v) | EK::Call(_, v) | EK::Builtin(_, v) => v.iter().for_each(|x| walk(x, f)),
        EK::Texture(t) => tex_operands(t).for_each(|x| walk(x, f)),
        _ => {}
    }
}

/// Every `let` value that statements read.
fn values_read(stmts: &[Stmt], out: &mut HashSet<ValueId>) {
    let expr = |e: &Expr, out: &mut HashSet<ValueId>| {
        walk(e, &mut |x| {
            if let EK::Value(v) = x.kind {
                out.insert(v);
            }
        })
    };
    for s in stmts {
        match s {
            Stmt::Let(_, e) | Stmt::Eval(e) => expr(e, out),
            Stmt::Return(e) => {
                if let Some(e) = e {
                    expr(e, out);
                }
            }
            Stmt::Assign { place, value, .. } => {
                for st in &place.steps {
                    if let Step::Index(i) = st {
                        expr(i, out);
                    }
                }
                expr(value, out);
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                expr(cond, out);
                values_read(then, out);
                values_read(otherwise, out);
            }
            Stmt::Loop {
                cond,
                body,
                continuing,
                break_if,
            } => {
                if let Some(c) = cond {
                    expr(c, out);
                }
                if let Some(b) = break_if {
                    expr(b, out);
                }
                values_read(body, out);
                values_read(continuing, out);
            }
            Stmt::Block(b) => values_read(b, out),
            Stmt::Break | Stmt::Continue | Stmt::Discard => {}
        }
    }
}

#[derive(Default)]
struct FnCx {
    names: Names,
    values: HashMap<ValueId, String>,
    locals: Vec<String>,
    /// `let`s declared before their loop because its continuing block reads them.
    hoisted: HashSet<ValueId>,
    /// Spelling of `ExprKind::Current` while emitting a compound assignment.
    current: Option<String>,
    indent: usize,
}

struct Emitter<'p> {
    prog: &'p Program,
    stage: Stage,
    out: String,
    module: Names,
    structs: Vec<String>,
    members: Vec<Vec<String>>,
    /// Structs GLSL can declare (no runtime-sized arrays, no arrays of arrays).
    plain: Vec<bool>,
    globals: HashMap<GlobalId, String>,
    fns: HashMap<FnId, String>,
    f: FnCx,
    current_fn: Option<FnId>,
    units: Vec<(GlobalId, Option<GlobalId>, Unit)>,
    storage: Vec<(GlobalId, Storage)>,
    hidden: Vec<Hidden>,
    /// Generated helper functions in dependency order, and the keys already generated.
    helpers: Vec<String>,
    helper_keys: HashSet<String>,
}

/// GLSL ES for `entry`. The caller has already compiled the entry point to SPIR-V, which checked
/// the stage, the reachable built-ins and the resource interface.
pub(crate) fn emit(prog: &Program, stage: Stage, entry: &str) -> R<Glsl> {
    let fid = prog
        .functions
        .iter()
        .position(|f| f.name == entry && f.stage == Some(stage))
        .ok_or_else(|| internal("entry point not found"))?;
    if stage == Stage::Compute {
        return Err(unsupported(
            &prog.functions[fid].stage_span,
            "a compute entry point",
        ));
    }
    let reach = reachable(prog, fid);
    let mut used: Vec<GlobalId> = reach
        .iter()
        .flat_map(|&r| prog.functions[r].globals.iter().copied())
        .collect();
    used.sort_unstable();
    used.dedup();

    let plain: Vec<bool> = (0..prog.structs.len())
        .map(|sid| struct_is_plain(sid, &prog.structs))
        .collect();
    let mut e = Emitter {
        prog,
        stage,
        out: String::new(),
        module: Names::default(),
        structs: Vec::new(),
        members: Vec::new(),
        plain,
        globals: HashMap::new(),
        fns: HashMap::new(),
        f: FnCx::default(),
        current_fn: None,
        units: Vec::new(),
        storage: Vec::new(),
        hidden: Vec::new(),
        helpers: Vec::new(),
        helper_keys: HashSet::new(),
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
        let gl = &prog.globals[g];
        let name = match gl.kind {
            GlobalKind::Storage { .. } => {
                let (group, binding) = gl.binding.ok_or_else(|| internal("unbound storage"))?;
                format!("ganbate_buf_{group}_{binding}")
            }
            GlobalKind::Handle => String::new(),
            _ => e.module.claim(&gl.name),
        };
        e.globals.insert(g, name);
    }
    for &r in &reach {
        let name = e.module.claim(&prog.functions[r].name);
        e.fns.insert(r, name);
    }

    // Function bodies first: they decide which units, helpers and hidden uniforms exist.
    for &r in reach.iter().rev() {
        e.function(r)?;
    }
    let main = e.wrapper(fid)?;
    let functions = std::mem::take(&mut e.out);

    let mut src = String::new();
    let _ = writeln!(
        src,
        "#version 300 es\n// Generated by ganbate-shader from '{entry}' ({stage:?}) for WebGL 2."
    );
    src.push_str(
        "precision highp float;\nprecision highp int;\nprecision highp sampler2D;\n\
         precision highp sampler2DArray;\nprecision highp samplerCube;\n\
         precision highp sampler2DShadow;\nprecision highp sampler2DArrayShadow;\n\
         precision highp samplerCubeShadow;\nprecision highp isampler2D;\n\
         precision highp isampler2DArray;\nprecision highp isamplerCube;\n\
         precision highp usampler2D;\nprecision highp usampler2DArray;\n\
         precision highp usamplerCube;\n\n",
    );
    for sid in struct_order(prog) {
        if e.plain[sid] {
            e.struct_decl(sid, &mut src)?;
        }
    }
    let mut blocks = Vec::new();
    for &g in &used {
        let gl = &prog.globals[g];
        match gl.kind {
            GlobalKind::Uniform | GlobalKind::PushConstant => {
                std140_check(&gl.ty, &prog.structs).map_err(|m| {
                    gl.span.error(format!(
                        "'{}' cannot become a GLSL std140 uniform block: {m}",
                        gl.name
                    ))
                })?;
                let ty = e.ty(&gl.ty, &gl.span)?;
                let (block, binding) = match gl.kind {
                    GlobalKind::PushConstant => ("ganbate_push".to_string(), None),
                    _ => {
                        let (group, binding) = gl.binding.ok_or_else(|| internal("unbound"))?;
                        (
                            format!("ganbate_ubo_{group}_{binding}"),
                            Some((group, binding)),
                        )
                    }
                };
                let _ = writeln!(
                    src,
                    "layout(std140) uniform {block} {{\n  {ty} {};\n}};",
                    e.globals[&g]
                );
                blocks.push(Block {
                    name: block,
                    binding,
                    size: size_of(&gl.ty, &prog.structs),
                });
            }
            _ => {}
        }
    }
    for (_, _, u) in &e.units {
        let _ = writeln!(
            src,
            "uniform highp {} {};",
            sampler_type(u.view, u.sample, u.compare),
            u.name
        );
    }
    for (_, s) in &e.storage {
        let _ = writeln!(src, "uniform highp usampler2D {};", s.name);
    }
    for h in &e.hidden {
        let _ = writeln!(src, "uniform {} {};", h.value.glsl_type(), h.name);
    }
    src.push('\n');
    for &g in &used {
        let gl = &prog.globals[g];
        match gl.kind {
            GlobalKind::Private => {
                let ty = e.ty(&gl.ty, &gl.span)?;
                let init = match &gl.init {
                    Some(v) => e.konst(&gl.ty, v, &gl.span)?,
                    None => e.zero(&gl.ty, &gl.span)?,
                };
                let _ = writeln!(src, "{ty} {} = {init};", e.globals[&g]);
            }
            GlobalKind::ConstData => {
                let ty = e.ty(&gl.ty, &gl.span)?;
                let init = match &gl.init {
                    Some(v) => e.konst(&gl.ty, v, &gl.span)?,
                    None => e.zero(&gl.ty, &gl.span)?,
                };
                let _ = writeln!(src, "const {ty} {} = {init};", e.globals[&g]);
            }
            _ => {}
        }
    }
    src.push('\n');
    for h in &e.helpers {
        src.push_str(h);
        src.push('\n');
    }
    // Prototypes, so definitions may come in any order.
    for &r in reach.iter().rev() {
        let sig = e.signature(r)?;
        let _ = writeln!(src, "{sig};");
    }
    src.push('\n');
    src.push_str(&functions);
    src.push_str(&main);

    Ok(Glsl {
        source: src,
        blocks,
        units: e.units.into_iter().map(|(_, _, u)| u).collect(),
        storage: e.storage.into_iter().map(|(_, s)| s).collect(),
        hidden: e.hidden,
    })
}

/// Whether GLSL ES 3.00 can declare struct `sid` (and the structs it contains).
fn struct_is_plain(sid: usize, structs: &[StructInfo]) -> bool {
    fn plain(ty: &Ty, structs: &[StructInfo], depth: u32) -> bool {
        match ty {
            Ty::Array(_, None) => false,
            Ty::Array(e, Some(_)) => !matches!(**e, Ty::Array(..)) && plain(e, structs, depth),
            Ty::Struct(id) => {
                depth < MAX_TYPE_DEPTH
                    && structs[*id]
                        .members
                        .iter()
                        .all(|m| plain(&m.ty, structs, depth + 1))
            }
            _ => true,
        }
    }
    plain(&Ty::Struct(sid), structs, 0)
}

/// Structs in an order where every struct comes after the structs it contains.
fn struct_order(prog: &Program) -> Vec<usize> {
    fn visit(sid: usize, prog: &Program, seen: &mut Vec<bool>, out: &mut Vec<usize>) {
        if seen[sid] {
            return;
        }
        seen[sid] = true;
        for m in &prog.structs[sid].members {
            if let Ty::Struct(id) = m.ty.innermost() {
                visit(*id, prog, seen, out);
            }
        }
        out.push(sid);
    }
    let mut seen = vec![false; prog.structs.len()];
    let mut out = Vec::new();
    for sid in 0..prog.structs.len() {
        visit(sid, prog, &mut seen, &mut out);
    }
    out
}

/// std140 alignment and size of `ty`, or why it differs from WGSL's layout.
fn std140(ty: &Ty, structs: &[StructInfo]) -> Result<(u32, u32), String> {
    Ok(match ty {
        Ty::Scalar(_) => (4, 4),
        Ty::Vector(2, _) => (8, 8),
        Ty::Vector(3, _) => (16, 12),
        Ty::Vector(..) => (16, 16),
        Ty::Matrix(_, 2) => {
            return Err("matCx2 columns are 8 bytes apart in WGSL but 16 in std140".into());
        }
        Ty::Matrix(c, _) => (16, 16 * u32::from(*c)),
        Ty::Array(e, Some(n)) => {
            let (a, s) = std140(e, structs)?;
            let stride = round_up(16, round_up(a, s));
            (round_up(16, a), stride * n)
        }
        Ty::Struct(id) => {
            let s = &structs[*id];
            let layout = s.layout.as_ref().ok_or("the type is not host-shareable")?;
            let mut offset = 0;
            let mut align = 16;
            for (i, m) in s.members.iter().enumerate() {
                let (a, size) = std140(&m.ty, structs)?;
                align = align.max(a);
                offset = round_up(a, offset);
                if offset != layout.offsets[i] {
                    return Err(format!(
                        "member '{}' of '{}' is at byte {} in WGSL but std140 puts it at {offset}",
                        m.name, s.name, layout.offsets[i]
                    ));
                }
                offset += size;
                if matches!(m.ty, Ty::Struct(_) | Ty::Array(..)) {
                    offset = round_up(16, offset);
                }
            }
            (align, round_up(align, offset))
        }
        _ => return Err("the type is not host-shareable".into()),
    })
}

fn std140_check(ty: &Ty, structs: &[StructInfo]) -> Result<(), String> {
    std140(ty, structs).map(|_| ())
}

fn sampler_type(view: TextureView, sample: SampleType, compare: bool) -> &'static str {
    match (compare, sample, view) {
        (true, _, TextureView::D2) => "sampler2DShadow",
        (true, _, TextureView::D2Array) => "sampler2DArrayShadow",
        (true, _, TextureView::Cube) => "samplerCubeShadow",
        (false, SampleType::Sint, TextureView::D2) => "isampler2D",
        (false, SampleType::Sint, TextureView::D2Array) => "isampler2DArray",
        (false, SampleType::Sint, TextureView::Cube) => "isamplerCube",
        (false, SampleType::Uint, TextureView::D2) => "usampler2D",
        (false, SampleType::Uint, TextureView::D2Array) => "usampler2DArray",
        (false, SampleType::Uint, TextureView::Cube) => "usamplerCube",
        (false, _, TextureView::D2) => "sampler2D",
        (false, _, TextureView::D2Array) => "sampler2DArray",
        (false, _, TextureView::Cube) => "samplerCube",
    }
}

impl Emitter<'_> {
    // ------------------------------------------------------------------ types

    fn ty(&self, t: &Ty, span: &crate::lexer::Span) -> R<String> {
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

    fn zero(&self, t: &Ty, span: &crate::lexer::Span) -> R<String> {
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
    fn konst(&self, ty: &Ty, v: &ConstValue, span: &crate::lexer::Span) -> R<String> {
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

    fn struct_decl(&self, sid: usize, out: &mut String) -> R<()> {
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
    fn helper(&mut self, key: &str, code: impl FnOnce(&mut Self) -> R<String>) -> R<()> {
        if self.helper_keys.contains(key) {
            return Ok(());
        }
        self.helper_keys.insert(key.to_string());
        let text = code(self)?;
        self.helpers.push(text);
        Ok(())
    }

    /// The hidden uniform for `value`, declared once.
    fn hidden(&mut self, name: String, value: HiddenValue) -> String {
        if !self.hidden.iter().any(|h| h.name == name) {
            self.hidden.push(Hidden {
                name: name.clone(),
                value,
            });
        }
        name
    }

    /// The combined sampler uniform for `texture` read through `sampler` (or unfiltered).
    fn unit(&mut self, texture: GlobalId, sampler: Option<GlobalId>) -> R<String> {
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
    fn storage_buffer(&mut self, g: GlobalId) -> R<(String, (u32, u32))> {
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
    fn loader(&mut self, ty: &Ty, span: &crate::lexer::Span) -> R<String> {
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
    fn overload(
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

    fn line(&mut self, text: &str) {
        for _ in 0..self.f.indent {
            self.out.push_str("  ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn signature(&self, fid: FnId) -> R<String> {
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

    fn function(&mut self, fid: FnId) -> R<()> {
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

    /// Stores `value` into `place` (or into components `swizzle` of the vector at `place`).
    /// Dynamic indices of the target are evaluated once, before the value.
    fn assign(&mut self, place: &Place, swizzle: Option<&[u32]>, value: &Expr) -> R<()> {
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

    fn expr(&mut self, e: &Expr) -> R<String> {
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
    fn wrapper(&mut self, fid: FnId) -> R<String> {
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
