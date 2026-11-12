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
use std::fmt::Write;

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

mod decl;
mod expr;
