//! Metal Shading Language (MSL 2.1+) code generation from the typed IR for one entry point.
//!
//! The same checked program that becomes SPIR-V is printed as MSL source text with one entry
//! point. The output follows these conventions:
//!
//! * **Names.** Every standard-library call and type is qualified with `metal::` (only `uint`
//!   is imported), so user identifiers only have to avoid C++ and MSL keywords, built-in type
//!   names and macros; colliding names get a suffix. Locals, `let`s and parameters get
//!   function-unique names. Generated names start with `ganbate_`.
//! * **Resources.** MSL has no mutable program-scope variables, so every function receives the
//!   resources and `var<private>` globals it uses (transitively) as extra parameters: uniforms
//!   and push constants as `constant T&`, storage buffers as `device T&` (or `const device
//!   T&`; a bare runtime-sized array as `device E*`), private globals as `thread T&`, textures
//!   and samplers by value. Dynamically indexed constants become program-scope `constant`
//!   data. The entry function declares the private globals as locals.
//! * **Bindings.** Per stage, resources get `[[buffer(n)]]`, `[[texture(n)]]` and
//!   `[[sampler(n)]]` indices in `(group, binding)` order, starting at 0 for each kind and
//!   reported in [`Msl::slots`]. `var<push_constant>` is a `constant T&` at
//!   [`PUSH_CONSTANT_BUFFER`] (fill it with `set*Bytes`). `arrayLength` reads the byte sizes of
//!   the bound runtime-sized buffers from a `constant uint*` at [`SIZES_BUFFER`], in
//!   [`Msl::sized_bindings`] order. Vertex `@location` inputs are a `[[stage_in]]` struct with
//!   `[[attribute(location)]]` members, fetched through an `MTLVertexDescriptor` from
//!   [`VERTEX_BUFFER`]. Varyings use `[[user(locnN)]]`; interpolation qualifiers sit on the
//!   fragment inputs.
//! * **Layout.** Structs that live in uniform, storage or push-constant memory reproduce the
//!   WGSL host-shareable layout byte for byte: a `vec3` whose 16-byte MSL slot would overlap
//!   the next member is stored as `packed_T3`, and explicit `char` padding fills every gap and
//!   the tail, so `sizeof` equals the WGSL size. Packed members are unpacked on every read and
//!   written whole. Fixed-size arrays are wrapped in a struct (`inner`) so they can be copied,
//!   passed and returned; strides already agree with WGSL. Other structs use the natural MSL
//!   layout.
//! * **Clip space.** The vertex entry negates `position.y`, so the engine's Vulkan clip space
//!   (+Y down, depth 0..1) lands on the same framebuffer pixels in Metal (+Y up). Pipelines
//!   keep the Vulkan front-face winding; fragment `[[position]]` is already top-left based.
//! * **Evaluation order.** C++ leaves the order of operands and arguments unspecified, so when
//!   an expression contains a call to a function with side effects (writes to private or
//!   storage memory, texture stores, `discard`) next to another such call or a read of
//!   writable memory, the operands are evaluated into temporaries in WGSL order. `&&` and `||`
//!   are left to right and short-circuit in both languages.
//! * **Arithmetic.** Signed `+`, `-`, `*` and negation wrap (computed on `uint` bits), shift
//!   amounts are masked to 0..31 like WGSL, `round` is `rint` (ties to even) and float `%` is
//!   `fmod`. As with SPIR-V, integer division by zero and out-of-range float-to-int
//!   conversions are undefined rather than WGSL's defined results, and `discard` ends the
//!   invocation (`discard_fragment()` and a return).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write;

use crate::ast::BinaryOp;
use crate::ir::{
    Builtin as B, Expr, ExprKind as EK, FnId, GlobalId, GlobalKind, Place, PlaceRoot, Program,
    Step, Stmt, TexCall, UnaryOp, ValueId,
};
use crate::types::*;
use crate::{Error, SampleType, Stage, TextureView};

/// Buffer index of the vertex buffer that feeds `[[stage_in]]` (vertex stage).
pub const VERTEX_BUFFER: u32 = 30;
/// Buffer index of the `var<push_constant>` block (`setVertexBytes` and friends).
pub const PUSH_CONSTANT_BUFFER: u32 = 29;
/// Buffer index of the `uint` byte sizes used by `arrayLength`.
pub const SIZES_BUFFER: u32 = 28;
/// Resource buffers use indices `0..MAX_RESOURCE_BUFFERS`.
pub const MAX_RESOURCE_BUFFERS: u32 = 28;

/// One entry point as Metal Shading Language source, with its binding map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Msl {
    /// A complete MSL translation unit (`#include <metal_stdlib>` and one entry function).
    pub source: String,
    /// Name of the `vertex`, `fragment` or `kernel` function in `source`.
    pub entry_point: String,
    /// Argument-table index of every resource the entry point uses, sorted by (group, binding).
    pub slots: Vec<MslSlot>,
    /// `Some(PUSH_CONSTANT_BUFFER)` when the entry point reads a push-constant block.
    pub push_constant_buffer: Option<u32>,
    /// `(group, binding)` of the runtime-sized storage buffers whose bound byte sizes the
    /// shader reads (as `uint`s, in this order) from [`SIZES_BUFFER`]; empty when
    /// `arrayLength` is not used.
    pub sized_bindings: Vec<(u32, u32)>,
    /// `Some(VERTEX_BUFFER)` when the vertex entry point reads `@location` inputs.
    pub vertex_buffer: Option<u32>,
    /// The vertex output uses `[[invariant]]` (compile with `preserveInvariance`).
    pub invariant: bool,
}

/// Where one `@group(g) @binding(b)` resource is bound for this stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MslSlot {
    pub group: u32,
    pub binding: u32,
    pub index: MslIndex,
}

/// An index in one of a stage's argument tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MslIndex {
    Buffer(u32),
    Texture(u32),
    Sampler(u32),
}

type R<T> = Result<T, Error>;

fn internal(what: &str) -> Error {
    Error {
        line: 1,
        column: 1,
        message: format!("internal compiler error (MSL): {what}"),
    }
}

// ---------------------------------------------------------------------------------- names

const KEYWORDS: &[&str] = &[
    // C++14
    "alignas",
    "alignof",
    "and",
    "and_eq",
    "asm",
    "auto",
    "bitand",
    "bitor",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "char16_t",
    "char32_t",
    "class",
    "compl",
    "const",
    "constexpr",
    "const_cast",
    "continue",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "dynamic_cast",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "not",
    "not_eq",
    "nullptr",
    "operator",
    "or",
    "or_eq",
    "private",
    "protected",
    "public",
    "register",
    "reinterpret_cast",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "static_assert",
    "static_cast",
    "struct",
    "switch",
    "template",
    "this",
    "thread_local",
    "throw",
    "true",
    "try",
    "typedef",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "wchar_t",
    "while",
    "xor",
    "xor_eq",
    // MSL qualifiers, attributes-as-keywords and global names
    "kernel",
    "vertex",
    "fragment",
    "compute",
    "device",
    "constant",
    "thread",
    "threadgroup",
    "threadgroup_imageblock",
    "ray_data",
    "object_data",
    "visible",
    "intersection",
    "mesh",
    "object",
    "stage_in",
    "half",
    "bfloat",
    "uint",
    "uchar",
    "ushort",
    "ulong",
    "size_t",
    "ptrdiff_t",
    "metal",
    "main",
    "std",
    "simd",
    // Macros from metal_stdlib
    "INFINITY",
    "NAN",
    "MAXFLOAT",
    "MAXHALF",
    "NULL",
    "assert",
    "offsetof",
];

/// Prefixes of reserved names (vector/matrix types and limit macros).
fn reserved(name: &str) -> bool {
    if KEYWORDS.contains(&name) {
        return true;
    }
    for t in [
        "bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "half",
        "float", "double", "bfloat",
    ] {
        if let Some(rest) = name.strip_prefix(t)
            && rest.starts_with(|c: char| c.is_ascii_digit())
        {
            return true;
        }
    }
    const MACRO_PREFIXES: &[&str] = &[
        "M_", "FLT_", "DBL_", "HALF_", "HUGE_", "INT_", "UINT_", "SHRT_", "USHRT_", "CHAR_",
        "UCHAR_", "SCHAR_", "LONG_", "ULONG_", "FP_", "__",
    ];
    name.starts_with("packed_")
        || name.starts_with("ganbate_")
        || MACRO_PREFIXES.iter().any(|p| name.starts_with(p))
        || (name.starts_with('_') && name[1..].starts_with(|c: char| c.is_ascii_uppercase()))
}

/// An ASCII C identifier for a WGSL identifier (WGSL allows Unicode letters).
fn ascii_ident(name: &str) -> String {
    let mut s = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            s.push(c);
        } else {
            let _ = write!(s, "_u{:x}_", c as u32);
        }
    }
    if s.is_empty() || s.starts_with(|c: char| c.is_ascii_digit()) {
        s.insert(0, 'v');
    }
    s
}

/// A set of taken names; `alloc` returns a fresh variant of `base`.
#[derive(Clone, Default)]
struct Names(HashSet<String>);

impl Names {
    fn alloc(&mut self, base: &str) -> String {
        let mut s = ascii_ident(base);
        if reserved(&s) || s.ends_with('_') && s.trim_end_matches('_').is_empty() {
            s.push('_');
        }
        if self.0.insert(s.clone()) {
            return s;
        }
        for i in 1.. {
            let c = format!("{s}_{i}");
            if self.0.insert(c.clone()) {
                return c;
            }
        }
        unreachable!()
    }
    /// Claims a generated name (callers pass names starting with `ganbate_`).
    fn generated(&mut self, base: &str) -> String {
        if self.0.insert(base.to_string()) {
            return base.to_string();
        }
        for i in 1.. {
            let c = format!("{base}_{i}");
            if self.0.insert(c.clone()) {
                return c;
            }
        }
        unreachable!()
    }
}

// ---------------------------------------------------------------------------------- layout

/// The MSL definition of a struct: members in order with padding.
#[derive(Clone, Debug)]
struct StructPlan {
    name: String,
    member_names: Vec<String>,
    /// Members stored as `packed_T3`.
    packed: Vec<bool>,
    /// `Some(bytes)` = padding, `None` = the next member.
    fields: Vec<Option<u32>>,
}

/// Address space of a reference to a place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Space {
    Thread,
    Device,
    Constant,
}

impl Space {
    fn qualifier(self) -> &'static str {
        match self {
            Space::Thread => "thread",
            Space::Device => "device",
            Space::Constant => "constant",
        }
    }
}

fn scalar_name(s: Scalar) -> &'static str {
    match s {
        Scalar::Bool => "bool",
        Scalar::I32 | Scalar::AbstractInt => "int",
        Scalar::U32 => "uint",
        Scalar::F32 | Scalar::AbstractFloat => "float",
    }
}

fn vector_name(n: u8, s: Scalar) -> String {
    format!("metal::{}{n}", scalar_name(s))
}

fn unsigned_of(t: &Ty) -> String {
    match t {
        Ty::Vector(n, _) => vector_name(*n, Scalar::U32),
        _ => "uint".into(),
    }
}

/// `f` as an MSL float literal that rounds back to the same bits.
fn float_lit(f: f32) -> String {
    if !f.is_finite() {
        return format!("metal::as_type<float>({:#x}u)", f.to_bits());
    }
    let s = format!("{f:?}");
    if s.contains('.') || s.contains('e') || s.contains("inf") {
        format!("{s}f")
    } else {
        format!("{s}.0f")
    }
}

fn int_lit(i: i32) -> String {
    if i == i32::MIN {
        "(-2147483647 - 1)".into()
    } else if i < 0 {
        format!("({i})")
    } else {
        i.to_string()
    }
}

fn swizzle_letters(comps: &[u32]) -> String {
    comps
        .iter()
        .map(|&c| ['x', 'y', 'z', 'w'][c as usize])
        .collect()
}

fn has_matrix(t: &Ty, structs: &[StructInfo]) -> bool {
    match t {
        Ty::Matrix(..) => true,
        Ty::Array(e, _) => has_matrix(e, structs),
        Ty::Struct(id) => structs[*id]
            .members
            .iter()
            .any(|m| has_matrix(&m.ty, structs)),
        _ => false,
    }
}

// ---------------------------------------------------------------------------------- analysis

/// Per-function facts the emitter needs.
#[derive(Clone, Default)]
struct FnFacts {
    /// Writes private or storage memory, stores to a texture or discards (transitively).
    impure: bool,
    /// Reads `arrayLength` sizes (transitively).
    sizes: bool,
    /// Globals passed as parameters (transitively used), sorted.
    globals: Vec<GlobalId>,
}

fn walk_expr<'e>(e: &'e Expr, f: &mut dyn FnMut(&'e Expr)) {
    f(e);
    match &e.kind {
        EK::Load(p) => walk_place(p, f),
        EK::Extract(b, _)
        | EK::Swizzle(b, _)
        | EK::Unary(_, b)
        | EK::Splat(b)
        | EK::Convert(b)
        | EK::Bitcast(b) => walk_expr(b, f),
        EK::DynIndex(a, b) | EK::Binary(_, a, b) => {
            walk_expr(a, f);
            walk_expr(b, f);
        }
        EK::Construct(v) | EK::Call(_, v) | EK::Builtin(_, v) => {
            for x in v {
                walk_expr(x, f);
            }
        }
        EK::Texture(t) => {
            for x in tex_operands(t) {
                walk_expr(x, f);
            }
        }
        EK::Const(_) | EK::Zero | EK::Value(_) | EK::ArrayLength { .. } | EK::Current => {}
    }
}

fn walk_place<'e>(p: &'e Place, f: &mut dyn FnMut(&'e Expr)) {
    for s in &p.steps {
        if let Step::Index(i) = s {
            walk_expr(i, f);
        }
    }
}

/// Operands of a texture call in WGSL argument order.
fn tex_operands(t: &TexCall) -> Vec<&Expr> {
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
    .collect()
}

fn walk_stmts<'s>(stmts: &'s [Stmt], f: &mut dyn FnMut(&'s Stmt)) {
    for s in stmts {
        f(s);
        match s {
            Stmt::If {
                then, otherwise, ..
            } => {
                walk_stmts(then, f);
                walk_stmts(otherwise, f);
            }
            Stmt::Loop {
                body, continuing, ..
            } => {
                walk_stmts(body, f);
                walk_stmts(continuing, f);
            }
            Stmt::Block(b) => walk_stmts(b, f),
            _ => {}
        }
    }
}

/// Every expression directly owned by a statement (not by nested statements).
fn stmt_exprs(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Let(_, e) | Stmt::Eval(e) | Stmt::Return(Some(e)) => vec![e],
        Stmt::Assign { place, value, .. } => {
            let mut v: Vec<&Expr> = place
                .steps
                .iter()
                .filter_map(|s| match s {
                    Step::Index(i) => Some(i),
                    _ => None,
                })
                .collect();
            v.push(value);
            v
        }
        Stmt::If { cond, .. } => vec![cond],
        Stmt::Loop { cond, break_if, .. } => cond.iter().chain(break_if.iter()).collect(),
        _ => Vec::new(),
    }
}

/// Does the loop body contain a `continue` for this loop (not for a nested one)?
fn has_own_continue(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Continue => true,
        Stmt::If {
            then, otherwise, ..
        } => has_own_continue(then) || has_own_continue(otherwise),
        Stmt::Block(b) => has_own_continue(b),
        _ => false,
    })
}

/// Values declared by `let`s at any depth of `stmts` (excluding nested loop bodies' own
/// continuing references, which are handled by those loops).
fn declared_values(stmts: &[Stmt], out: &mut BTreeSet<ValueId>) {
    walk_stmts(stmts, &mut |s| {
        if let Stmt::Let(id, _) = s {
            out.insert(*id);
        }
    });
}

fn used_values(stmts: &[Stmt], extra: Option<&Expr>, out: &mut BTreeSet<ValueId>) {
    let mut visit = |e: &Expr| {
        walk_expr(e, &mut |x| {
            if let EK::Value(v) = x.kind {
                out.insert(v);
            }
        })
    };
    walk_stmts(stmts, &mut |s| {
        for e in stmt_exprs(s) {
            visit(e);
        }
    });
    if let Some(e) = extra {
        visit(e);
    }
}

// ---------------------------------------------------------------------------------- emitter

#[derive(Default)]
struct FnState {
    names: Names,
    values: HashMap<ValueId, String>,
    locals: Vec<String>,
    /// `let`s declared at function scope because a loop's continuing block reads them.
    hoisted: HashSet<ValueId>,
    /// Statements to place before the one being emitted.
    pre: Vec<String>,
    /// Text of the compound-assignment target's current value.
    current: Option<String>,
    /// The function returns this type (for the return after `discard_fragment()`).
    ret: Option<Ty>,
}

struct Writer<'p> {
    prog: &'p Program,
    stage: Stage,
    names: Names,
    struct_plans: HashMap<usize, StructPlan>,
    /// Structs whose layout must match WGSL's host-shareable layout.
    host_structs: HashSet<usize>,
    array_names: HashMap<Ty, String>,
    /// Type definitions in dependency order.
    types: String,
    defined: HashSet<Ty>,
    fn_names: HashMap<FnId, String>,
    global_names: HashMap<GlobalId, String>,
    facts: HashMap<FnId, FnFacts>,
    /// Index of each sized storage buffer in `Msl::sized_bindings`.
    sized: Vec<GlobalId>,
    sizes_name: String,
    helpers: BTreeSet<&'static str>,
    helper_names: HashMap<&'static str, String>,
    fs: FnState,
    /// The compound-assignment target lives in writable global memory.
    current_global: bool,
    cur_fn: FnId,
    indent: usize,
    out: String,
}

/// Emits MSL for `entry` (which must be a `stage` entry point of `prog`; the SPIR-V emitter
/// has already validated that and the stage requirements).
pub(crate) fn emit(prog: &Program, stage: Stage, entry: &str) -> R<Msl> {
    let fid = prog
        .functions
        .iter()
        .position(|f| f.name == entry && f.stage == Some(stage))
        .ok_or_else(|| internal("entry point not found"))?;
    // Callees before callers (C++ needs declarations first); recursion is rejected earlier.
    let mut order = Vec::new();
    let mut seen = BTreeSet::new();
    fn visit(prog: &Program, f: FnId, seen: &mut BTreeSet<FnId>, order: &mut Vec<FnId>) {
        if seen.insert(f) {
            for &c in &prog.functions[f].calls {
                visit(prog, c, seen, order);
            }
            order.push(f);
        }
    }
    visit(prog, fid, &mut seen, &mut order);

    let mut w = Writer {
        prog,
        stage,
        names: Names::default(),
        struct_plans: HashMap::new(),
        host_structs: HashSet::new(),
        array_names: HashMap::new(),
        types: String::new(),
        defined: HashSet::new(),
        fn_names: HashMap::new(),
        global_names: HashMap::new(),
        facts: HashMap::new(),
        sized: Vec::new(),
        sizes_name: String::new(),
        helpers: BTreeSet::new(),
        helper_names: HashMap::new(),
        fs: FnState::default(),
        current_global: false,
        cur_fn: fid,
        indent: 0,
        out: String::new(),
    };
    w.analyze(&order)?;
    let used: BTreeSet<GlobalId> = order
        .iter()
        .flat_map(|&f| prog.functions[f].globals.iter().copied())
        .collect();

    // User names first, then generated ones, so generated names never shadow them.
    for &g in &used {
        let gl = &prog.globals[g];
        let name = w.names.alloc(&gl.name);
        w.global_names.insert(g, name);
    }
    for &f in &order {
        let name = w.names.alloc(&prog.functions[f].name);
        w.fn_names.insert(f, name);
    }
    w.sizes_name = w.names.generated("ganbate_sizes");
    for &g in &used {
        w.mark_host(g);
    }

    // Resource reflection and argument-table indices.
    let mut slots = Vec::new();
    let mut entry_params: Vec<(GlobalId, String)> = Vec::new();
    let mut by_binding: Vec<(u32, u32, GlobalId)> = used
        .iter()
        .filter_map(|&g| prog.globals[g].binding.map(|(gr, b)| (gr, b, g)))
        .collect();
    by_binding.sort();
    let (mut nb, mut nt, mut ns) = (0u32, 0u32, 0u32);
    for &(group, binding, g) in &by_binding {
        let gl = &prog.globals[g];
        let index = match gl.kind {
            GlobalKind::Uniform | GlobalKind::Storage { .. } => {
                if nb >= MAX_RESOURCE_BUFFERS {
                    return Err(Error {
                        line: gl.span.line,
                        column: gl.span.col,
                        message: format!(
                            "entry point '{entry}' uses more than {MAX_RESOURCE_BUFFERS} buffers, \
                             the most Metal can bind next to vertex data and push constants"
                        ),
                    });
                }
                nb += 1;
                MslIndex::Buffer(nb - 1)
            }
            _ => match gl.ty {
                Ty::Sampler(_) => {
                    ns += 1;
                    MslIndex::Sampler(ns - 1)
                }
                _ => {
                    nt += 1;
                    MslIndex::Texture(nt - 1)
                }
            },
        };
        slots.push(MslSlot {
            group,
            binding,
            index,
        });
        let attr = match index {
            MslIndex::Buffer(i) => format!("[[buffer({i})]]"),
            MslIndex::Texture(i) => format!("[[texture({i})]]"),
            MslIndex::Sampler(i) => format!("[[sampler({i})]]"),
        };
        entry_params.push((g, attr));
    }
    let mut push_constant_buffer = None;
    for &g in &used {
        if prog.globals[g].kind == GlobalKind::PushConstant {
            if push_constant_buffer.is_some() {
                return Err(internal("two push constant blocks"));
            }
            push_constant_buffer = Some(PUSH_CONSTANT_BUFFER);
            entry_params.push((g, format!("[[buffer({PUSH_CONSTANT_BUFFER})]]")));
        }
    }
    // Runtime-sized buffers read by arrayLength, in binding order.
    let uses_sizes = w.facts[&fid].sizes;
    let mut sized_bindings = Vec::new();
    if uses_sizes {
        let mut sized: Vec<(u32, u32, GlobalId)> = Vec::new();
        for &f in &order {
            let mut found = Vec::new();
            walk_stmts(&prog.functions[f].body, &mut |s| {
                for e in stmt_exprs(s) {
                    walk_expr(e, &mut |x| {
                        if let EK::ArrayLength { global, .. } = x.kind {
                            found.push(global);
                        }
                    });
                }
            });
            for g in found {
                let (gr, b) = prog.globals[g].binding.unwrap_or((u32::MAX, u32::MAX));
                if !sized.iter().any(|s| s.2 == g) {
                    sized.push((gr, b, g));
                }
            }
        }
        sized.sort();
        w.sized = sized.iter().map(|s| s.2).collect();
        sized_bindings = sized.iter().map(|s| (s.0, s.1)).collect();
    }

    // Program-scope constant data.
    let mut constants = String::new();
    for &g in &used {
        let gl = &prog.globals[g];
        if gl.kind == GlobalKind::ConstData {
            let ty = w.ty_name(&gl.ty)?;
            let init = match &gl.init {
                Some(v) => w.const_value(&gl.ty, v)?,
                None => w.zero(&gl.ty)?,
            };
            let _ = writeln!(constants, "constant {ty} {} = {init};", w.global_names[&g]);
        }
    }

    for &f in &order {
        w.function(f)?;
    }
    let (entry_point, io_structs, invariant) = w.entry(fid, &entry_params, uses_sizes)?;

    let mut src = String::new();
    let _ = writeln!(
        src,
        "// Generated by ganbate-shader from the WGSL {} entry point '{entry}'.",
        format!("{stage:?}").to_lowercase()
    );
    src.push_str("#include <metal_stdlib>\n\nusing metal::uint;\n\n");
    src.push_str(&w.types);
    if !w.types.is_empty() {
        src.push('\n');
    }
    if !constants.is_empty() {
        src.push_str(&constants);
        src.push('\n');
    }
    src.push_str(&w.helper_source());
    src.push_str(&w.out);
    src.push_str(&io_structs);
    Ok(Msl {
        source: src,
        entry_point,
        slots,
        push_constant_buffer,
        sized_bindings,
        vertex_buffer: None,
        invariant,
    })
    .map(|mut m| {
        if stage == Stage::Vertex && w.has_vertex_inputs(fid) {
            m.vertex_buffer = Some(VERTEX_BUFFER);
        }
        m
    })
}

impl<'p> Writer<'p> {
    // ------------------------------------------------------------------ analysis

    fn analyze(&mut self, order: &[FnId]) -> R<()> {
        let prog = self.prog;
        for &f in order {
            let func = &prog.functions[f];
            let mut facts = FnFacts::default();
            let mut globals: BTreeSet<GlobalId> = func
                .globals
                .iter()
                .copied()
                .filter(|&g| prog.globals[g].kind != GlobalKind::ConstData)
                .collect();
            walk_stmts(&func.body, &mut |s| {
                match s {
                    Stmt::Discard => facts.impure = true,
                    Stmt::Assign { place, .. } => {
                        if let PlaceRoot::Global(g) = place.root
                            && matches!(
                                prog.globals[g].kind,
                                GlobalKind::Private | GlobalKind::Storage { .. }
                            )
                        {
                            facts.impure = true;
                        }
                    }
                    _ => {}
                }
                for e in stmt_exprs(s) {
                    walk_expr(e, &mut |x| match &x.kind {
                        EK::Texture(t) if t.op == B::TextureStore => facts.impure = true,
                        EK::ArrayLength { .. } => facts.sizes = true,
                        _ => {}
                    });
                }
            });
            for c in &func.calls {
                let cf = self
                    .facts
                    .get(c)
                    .ok_or_else(|| internal("callee analysed after caller"))?;
                facts.impure |= cf.impure;
                facts.sizes |= cf.sizes;
                globals.extend(cf.globals.iter().copied());
            }
            facts.globals = globals.into_iter().collect();
            self.facts.insert(f, facts);
        }
        Ok(())
    }

    /// Structs reachable from buffer memory get the exact WGSL layout.
    fn mark_host(&mut self, g: GlobalId) {
        let gl = &self.prog.globals[g];
        if matches!(
            gl.kind,
            GlobalKind::Uniform | GlobalKind::Storage { .. } | GlobalKind::PushConstant
        ) {
            let ty = gl.ty.clone();
            self.mark_host_ty(&ty);
        }
    }

    fn mark_host_ty(&mut self, t: &Ty) {
        match t {
            Ty::Array(e, _) => self.mark_host_ty(e),
            Ty::Struct(id) if self.host_structs.insert(*id) => {
                let members: Vec<Ty> = self.prog.structs[*id]
                    .members
                    .iter()
                    .map(|m| m.ty.clone())
                    .collect();
                for m in members {
                    self.mark_host_ty(&m);
                }
            }
            _ => {}
        }
    }

    fn effects(&self, e: &Expr) -> bool {
        let mut found = false;
        walk_expr(e, &mut |x| {
            if let EK::Call(f, _) = x.kind
                && self.facts.get(&f).is_some_and(|ff| ff.impure)
            {
                found = true;
            }
        });
        found
    }

    fn writable_global(&self, g: GlobalId) -> bool {
        matches!(
            self.prog.globals[g].kind,
            GlobalKind::Private | GlobalKind::Storage { read_only: false }
        )
    }

    /// Reads memory that a call with side effects could change.
    fn reads(&self, e: &Expr, current_global: bool) -> bool {
        let mut found = false;
        walk_expr(e, &mut |x| match &x.kind {
            EK::Load(p) => {
                if let PlaceRoot::Global(g) = p.root
                    && self.writable_global(g)
                {
                    found = true;
                }
            }
            EK::Current if current_global => found = true,
            _ => {}
        });
        found
    }

    /// Must these sibling operands be evaluated into temporaries to keep WGSL's order?
    fn needs_order(&self, es: &[&Expr]) -> bool {
        let cg = self.current_global;
        let eff: Vec<bool> = es.iter().map(|e| self.effects(e)).collect();
        let n_eff = eff.iter().filter(|&&b| b).count();
        if n_eff >= 2 {
            return true;
        }
        if n_eff == 1 {
            return es
                .iter()
                .zip(&eff)
                .any(|(e, &is_eff)| !is_eff && self.reads(e, cg));
        }
        false
    }

    /// Would emitting `e` hoist anything into `pre`?
    fn needs_hoist(&self, e: &Expr) -> bool {
        let mut found = false;
        walk_expr(e, &mut |x| {
            if found {
                return;
            }
            let kids: Vec<&Expr> = match &x.kind {
                EK::Binary(BinaryOp::LogicalAnd | BinaryOp::LogicalOr, ..) => Vec::new(),
                EK::Binary(_, a, b) | EK::DynIndex(a, b) => vec![a, b],
                EK::Construct(v) | EK::Call(_, v) | EK::Builtin(_, v) => v.iter().collect(),
                EK::Texture(t) => tex_operands(t),
                EK::Load(p) => p
                    .steps
                    .iter()
                    .filter_map(|s| match s {
                        Step::Index(i) => Some(i),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            };
            if kids.len() > 1 && self.needs_order(&kids) {
                found = true;
            }
            if let EK::Binary(BinaryOp::LogicalAnd | BinaryOp::LogicalOr, _, r) = &x.kind
                && self.needs_hoist(r)
            {
                found = true;
            }
        });
        found
    }

    fn has_vertex_inputs(&self, fid: FnId) -> bool {
        let f = &self.prog.functions[fid];
        f.params.iter().any(|p| {
            p.io.location.is_some()
                || matches!(p.ty, Ty::Struct(sid) if p.io.is_empty()
                    && self.prog.structs[sid].members.iter().any(|m| m.io.location.is_some()))
        })
    }

    // ------------------------------------------------------------------ types

    fn ty_name(&mut self, t: &Ty) -> R<String> {
        Ok(match t {
            Ty::Scalar(s) => scalar_name(*s).to_string(),
            Ty::Vector(n, s) => vector_name(*n, *s),
            Ty::Matrix(c, r) => format!("metal::float{c}x{r}"),
            Ty::Array(_, Some(_)) => self.array_name(t)?,
            Ty::Array(e, None) => self.ty_name(e)?,
            Ty::Struct(id) => self.struct_plan(*id)?.name.clone(),
            Ty::Texture(view, sample) => {
                let st = match sample {
                    SampleType::Float | SampleType::Depth => "float",
                    SampleType::Sint => "int",
                    SampleType::Uint => "uint",
                };
                let kind = match (sample, view) {
                    (SampleType::Depth, TextureView::D2) => "depth2d",
                    (SampleType::Depth, TextureView::D2Array) => "depth2d_array",
                    (SampleType::Depth, TextureView::Cube) => "depthcube",
                    (_, TextureView::D2) => "texture2d",
                    (_, TextureView::D2Array) => "texture2d_array",
                    (_, TextureView::Cube) => "texturecube",
                };
                format!("metal::{kind}<{st}, metal::access::sample>")
            }
            Ty::StorageTexture(_) => "metal::texture2d<float, metal::access::write>".into(),
            Ty::Sampler(_) => "metal::sampler".into(),
            Ty::Void => "void".into(),
        })
    }

    fn array_name(&mut self, t: &Ty) -> R<String> {
        if let Some(n) = self.array_names.get(t) {
            return Ok(n.clone());
        }
        let Ty::Array(e, Some(n)) = t else {
            return Err(internal("array_name of a non-array"));
        };
        let elem = self.ty_name(e)?;
        let short = elem
            .trim_start_matches("metal::")
            .trim_start_matches("ganbate_array_");
        let name = self.names.generated(&format!("ganbate_array_{short}_{n}"));
        self.array_names.insert(t.clone(), name.clone());
        let _ = writeln!(self.types, "struct {name} {{\n    {elem} inner[{n}];\n}};");
        self.defined.insert(t.clone());
        Ok(name)
    }

    fn struct_plan(&mut self, sid: usize) -> R<&StructPlan> {
        if !self.struct_plans.contains_key(&sid) {
            self.define_struct(sid)?;
        }
        Ok(&self.struct_plans[&sid])
    }

    fn define_struct(&mut self, sid: usize) -> R<()> {
        let prog = self.prog;
        let info = &prog.structs[sid];
        let host = self.host_structs.contains(&sid);
        // Define member types first.
        let mut member_types = Vec::with_capacity(info.members.len());
        for m in &info.members {
            member_types.push(self.ty_name(&m.ty)?);
        }
        let name = self.names.alloc(&info.name);
        let mut member_names = Names::default();
        let mnames: Vec<String> = info
            .members
            .iter()
            .map(|m| member_names.alloc(&m.name))
            .collect();
        let mut packed = vec![false; info.members.len()];
        let mut fields = Vec::new();
        let mut body = String::new();
        let mut pad = 0;
        if host {
            let layout = info
                .layout
                .as_ref()
                .ok_or_else(|| internal("buffer struct without a layout"))?;
            let structs = &prog.structs;
            let mut off = 0u32;
            for (i, m) in info.members.iter().enumerate() {
                let want = layout.offsets[i];
                let next = if i + 1 < info.members.len() {
                    layout.offsets[i + 1]
                } else {
                    layout.size
                };
                let runtime = matches!(m.ty, Ty::Array(_, None));
                let (size, align) = match &m.ty {
                    Ty::Vector(3, _) if next < want + 16 && !runtime => {
                        packed[i] = true;
                        (12, 4)
                    }
                    Ty::Vector(3, _) => (16, 16),
                    t => (size_of(t, structs), align_of(t, structs)),
                };
                if off > want || want % align != 0 {
                    return Err(internal("member offset not reproducible in MSL"));
                }
                if want > off {
                    let _ = writeln!(body, "    char ganbate_pad{pad}[{}];", want - off);
                    pad += 1;
                    fields.push(Some(want - off));
                }
                let ty = if packed[i] {
                    let Ty::Vector(_, s) = m.ty else {
                        unreachable!()
                    };
                    format!("metal::packed_{}3", scalar_name(s))
                } else {
                    member_types[i].clone()
                };
                if runtime {
                    let _ = writeln!(body, "    {ty} {}[1];", mnames[i]);
                } else {
                    let _ = writeln!(body, "    {ty} {};", mnames[i]);
                }
                fields.push(None);
                off = want + size;
            }
            let last_runtime = info
                .members
                .last()
                .is_some_and(|m| matches!(m.ty, Ty::Array(_, None)));
            if !last_runtime && off < layout.size {
                let _ = writeln!(body, "    char ganbate_pad{pad}[{}];", layout.size - off);
                fields.push(Some(layout.size - off));
            }
        } else {
            for (i, m) in info.members.iter().enumerate() {
                let _ = writeln!(body, "    {} {};", member_types[i], mnames[i]);
                let _ = m;
                fields.push(None);
            }
        }
        let _ = writeln!(self.types, "struct {name} {{\n{body}}};");
        if host {
            // The layout must match WGSL byte for byte; let the Metal compiler check it.
            let layout = info.layout.as_ref().expect("checked above");
            let runtime = info
                .members
                .last()
                .is_some_and(|m| matches!(m.ty, Ty::Array(_, None)));
            if !runtime {
                let _ = writeln!(
                    self.types,
                    "static_assert(sizeof({name}) == {}, \"{name}: WGSL layout\");",
                    layout.size
                );
            }
            self.types.push_str("#ifdef GANBATE_MSL_LAYOUT_CHECKS\n");
            for (i, m) in mnames.iter().enumerate() {
                let _ = writeln!(
                    self.types,
                    "static_assert(__builtin_offsetof({name}, {m}) == {}, \"{name}.{m}: WGSL offset\");",
                    layout.offsets[i]
                );
            }
            self.types.push_str("#endif\n");
        }
        self.struct_plans.insert(
            sid,
            StructPlan {
                name,
                member_names: mnames,
                packed,
                fields,
            },
        );
        Ok(())
    }

    /// Aggregate initialisation of struct `sid` from member texts.
    fn construct_struct(&mut self, sid: usize, parts: Vec<String>) -> R<String> {
        let prog = self.prog;
        let plan = self.struct_plan(sid)?.clone();
        let mut out = Vec::new();
        let mut it = parts.into_iter().enumerate();
        let last_member = plan.fields.iter().rposition(|f| f.is_none());
        for (fi, f) in plan.fields.iter().enumerate() {
            match f {
                Some(_) => {
                    if last_member.is_some_and(|l| fi < l) {
                        out.push("{}".to_string());
                    }
                }
                None => {
                    let (i, text) = it.next().ok_or_else(|| internal("struct arity"))?;
                    if plan.packed[i] {
                        let Ty::Vector(_, s) = prog.structs[sid].members[i].ty else {
                            return Err(internal("packed non-vector"));
                        };
                        out.push(format!("metal::packed_{}3({text})", scalar_name(s)));
                    } else {
                        out.push(text);
                    }
                }
            }
        }
        Ok(format!("{}{{{}}}", plan.name, out.join(", ")))
    }

    // ------------------------------------------------------------------ constants

    fn scalar_const(&self, v: &ConstValue, s: Scalar) -> String {
        match (*v).clone() {
            ConstValue::Bool(b) => b.to_string(),
            ConstValue::I32(i) => match s {
                Scalar::U32 => format!("{}u", i as u32),
                Scalar::F32 => float_lit(i as f32),
                _ => int_lit(i),
            },
            ConstValue::U32(u) => match s {
                Scalar::I32 => int_lit(u as i32),
                Scalar::F32 => float_lit(u as f32),
                _ => format!("{u}u"),
            },
            ConstValue::F32(f) => float_lit(f),
            ConstValue::AbstractInt(i) => match s {
                Scalar::U32 => format!("{}u", i as u32),
                Scalar::F32 | Scalar::AbstractFloat => float_lit(i as f32),
                _ => int_lit(i as i32),
            },
            ConstValue::AbstractFloat(f) => float_lit(f as f32),
            ConstValue::Composite(_) => "0".into(),
        }
    }

    fn const_value(&mut self, ty: &Ty, v: &ConstValue) -> R<String> {
        let prog = self.prog;
        Ok(match ty {
            Ty::Scalar(s) => self.scalar_const(v, *s),
            Ty::Vector(n, s) => {
                let parts: Vec<String> = v
                    .components()
                    .iter()
                    .map(|c| self.scalar_const(c, *s))
                    .collect();
                if parts.len() == *n as usize && parts.iter().all(|p| *p == parts[0]) {
                    format!("{}({})", vector_name(*n, *s), parts[0])
                } else {
                    format!("{}({})", vector_name(*n, *s), parts.join(", "))
                }
            }
            Ty::Matrix(c, r) => {
                let col = Ty::Vector(*r, Scalar::F32);
                let mut cols = Vec::new();
                for cv in v.components() {
                    cols.push(self.const_value(&col, cv)?);
                }
                if cols.len() != *c as usize {
                    return Err(internal("matrix constant arity"));
                }
                format!("metal::float{c}x{r}({})", cols.join(", "))
            }
            Ty::Array(e, Some(_)) => {
                let name = self.ty_name(ty)?;
                let mut parts = Vec::new();
                for ev in v.components() {
                    parts.push(self.const_value(e, ev)?);
                }
                format!("{name}{{{{{}}}}}", parts.join(", "))
            }
            Ty::Struct(sid) => {
                let mut parts = Vec::new();
                for (i, mv) in v.components().iter().enumerate() {
                    let mt = prog.structs[*sid].members[i].ty.clone();
                    parts.push(self.const_value(&mt, mv)?);
                }
                self.construct_struct(*sid, parts)?
            }
            _ => return Err(internal("constant of a non-constructible type")),
        })
    }

    fn zero(&mut self, ty: &Ty) -> R<String> {
        let prog = self.prog;
        match ty {
            Ty::Array(..) | Ty::Struct(_) if !has_matrix(ty, &prog.structs) => {
                Ok(format!("{}{{}}", self.ty_name(ty)?))
            }
            _ => {
                let z = ConstValue::zero(ty, &prog.structs);
                self.const_value(ty, &z)
            }
        }
    }

    // ------------------------------------------------------------------ helpers

    fn helper(&mut self, h: &'static str) -> String {
        self.helpers.insert(h);
        if let Some(n) = self.helper_names.get(h) {
            return n.clone();
        }
        let n = self.names.generated(h);
        self.helper_names.insert(h, n.clone());
        n
    }

    fn helper_source(&self) -> String {
        let mut s = String::new();
        let name = |h: &str| self.helper_names[h].clone();
        for &h in &self.helpers {
            match h {
                "ganbate_dims" => {
                    let n = name(h);
                    let _ = writeln!(
                        s,
                        "template <typename T>\nmetal::uint2 {n}(T t, uint lod) {{\n    \
                         return metal::uint2(t.get_width(lod), t.get_height(lod));\n}}\n"
                    );
                }
                "ganbate_dims_storage" => {
                    let n = name(h);
                    let _ = writeln!(
                        s,
                        "metal::uint2 {n}(metal::texture2d<float, metal::access::write> t) {{\n    \
                         return metal::uint2(t.get_width(), t.get_height());\n}}\n"
                    );
                }
                "ganbate_dot" => {
                    let n = name(h);
                    for (st, ut) in [("int", "uint"), ("uint", "uint")] {
                        for k in 2..=4 {
                            let comps = ["x", "y", "z", "w"][..k]
                                .iter()
                                .map(|c| format!("p.{c}"))
                                .collect::<Vec<_>>()
                                .join(" + ");
                            let _ = writeln!(
                                s,
                                "{st} {n}(metal::{st}{k} a, metal::{st}{k} b) {{\n    \
                                 metal::{ut}{k} p = metal::as_type<metal::{ut}{k}>(a) * \
                                 metal::as_type<metal::{ut}{k}>(b);\n    \
                                 return metal::as_type<{st}>({comps});\n}}"
                            );
                        }
                    }
                    s.push('\n');
                }
                "ganbate_sign" => {
                    let n = name(h);
                    let _ = writeln!(
                        s,
                        "int {n}(int x) {{\n    return metal::select(metal::select(0, -1, x < 0), 1, x > 0);\n}}"
                    );
                    for k in 2..=4 {
                        let _ = writeln!(
                            s,
                            "metal::int{k} {n}(metal::int{k} x) {{\n    return metal::select(\
                             metal::select(metal::int{k}(0), metal::int{k}(-1), x < 0), \
                             metal::int{k}(1), x > 0);\n}}"
                        );
                    }
                    s.push('\n');
                }
                _ => {}
            }
        }
        s
    }

    // ------------------------------------------------------------------ functions

    fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn flush_pre(&mut self) {
        let pre = std::mem::take(&mut self.fs.pre);
        for l in pre {
            self.line(&l);
        }
    }

    /// The parameter declaration of global `g` inside ordinary functions.
    fn global_param(&mut self, g: GlobalId) -> R<String> {
        let gl = &self.prog.globals[g];
        let name = self.global_names[&g].clone();
        let ty = self.ty_name(&gl.ty)?;
        Ok(match gl.kind {
            GlobalKind::Private => format!("thread {ty}& {name}"),
            GlobalKind::Uniform | GlobalKind::PushConstant => format!("constant {ty}& {name}"),
            GlobalKind::Storage { read_only } => {
                let c = if read_only { "const " } else { "" };
                if matches!(gl.ty, Ty::Array(_, None)) {
                    format!("{c}device {ty}* {name}")
                } else {
                    format!("{c}device {ty}& {name}")
                }
            }
            GlobalKind::Handle => format!("{ty} {name}"),
            GlobalKind::ConstData => return Err(internal("constant data as a parameter")),
        })
    }

    fn function(&mut self, fid: FnId) -> R<()> {
        let prog = self.prog;
        let f = &prog.functions[fid];
        self.cur_fn = fid;
        self.fs = FnState {
            names: self.names.clone(),
            ret: f.ret.clone(),
            ..FnState::default()
        };
        let ret = match &f.ret {
            Some(t) => self.ty_name(t)?,
            None => "void".into(),
        };
        let mut params = Vec::new();
        for p in &f.params {
            let ty = self.ty_name(&p.ty)?;
            let name = self.fs.names.alloc(&p.name);
            self.fs.values.insert(p.value, name.clone());
            params.push(format!("{ty} {name}"));
        }
        let facts = self.facts[&fid].clone();
        for &g in &facts.globals {
            params.push(self.global_param(g)?);
        }
        if facts.sizes {
            params.push(format!("constant uint* {}", self.sizes_name));
        }
        let name = self.fn_names[&fid].clone();
        let _ = writeln!(self.out, "{ret} {name}({}) {{", params.join(", "));
        self.indent = 1;
        for l in &f.locals {
            let ty = self.ty_name(&l.ty)?;
            let n = self.fs.names.alloc(&l.name);
            self.fs.locals.push(n.clone());
            self.line(&format!("{ty} {n};"));
        }
        // `let`s that a loop's continuing block reads are declared up front.
        let mut hoist = BTreeSet::new();
        walk_stmts(&f.body, &mut |s| {
            if let Stmt::Loop {
                body,
                continuing,
                break_if,
                ..
            } = s
                && has_own_continue(body)
                && (!continuing.is_empty() || break_if.is_some())
            {
                let mut declared = BTreeSet::new();
                declared_values(body, &mut declared);
                let mut used = BTreeSet::new();
                used_values(continuing, break_if.as_ref(), &mut used);
                hoist.extend(declared.intersection(&used).copied());
            }
        });
        for id in hoist {
            let ty = self.let_type(&f.body, id)?;
            let tn = self.ty_name(&ty)?;
            let base = f
                .value_names
                .get(id as usize)
                .cloned()
                .unwrap_or_else(|| "v".into());
            let n = self.fs.names.alloc(&base);
            self.line(&format!("{tn} {n};"));
            self.fs.values.insert(id, n);
            self.fs.hoisted.insert(id);
        }
        let falls = self.block(&f.body)?;
        if let (true, Some(ret)) = (falls, &f.ret) {
            // The checker proves every path returns; keep C++ from seeing a fall-through.
            let z = self.zero(ret)?;
            self.line(&format!("return {z};"));
        }
        self.indent = 0;
        self.out.push_str("}\n\n");
        Ok(())
    }

    fn let_type(&self, body: &[Stmt], id: ValueId) -> R<Ty> {
        let mut ty = None;
        walk_stmts(body, &mut |s| {
            if let Stmt::Let(v, e) = s
                && *v == id
            {
                ty = Some(e.ty.clone());
            }
        });
        ty.ok_or_else(|| internal("hoisted let without a declaration"))
    }

    /// Emits statements; returns whether control can fall off the end.
    fn block(&mut self, stmts: &[Stmt]) -> R<bool> {
        for s in stmts {
            if !self.stmt(s)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn nested(&mut self, header: &str, stmts: &[Stmt]) -> R<bool> {
        self.line(&format!("{header} {{"));
        self.indent += 1;
        let falls = self.block(stmts)?;
        self.indent -= 1;
        self.line("}");
        Ok(falls)
    }

    fn stmt(&mut self, s: &Stmt) -> R<bool> {
        match s {
            Stmt::Let(id, e) => {
                let v = self.expr(e)?;
                self.flush_pre();
                if self.fs.hoisted.contains(id) {
                    let n = self.fs.values[id].clone();
                    self.line(&format!("{n} = {v};"));
                } else {
                    let base = self.prog_fn_value_name(*id);
                    let n = self.fs.names.alloc(&base);
                    let ty = self.ty_name(&e.ty)?;
                    self.line(&format!("const {ty} {n} = {v};"));
                    self.fs.values.insert(*id, n);
                }
                Ok(true)
            }
            Stmt::Assign {
                place,
                swizzle,
                value,
            } => {
                self.assign(place, swizzle.as_deref(), value)?;
                Ok(true)
            }
            Stmt::Eval(e) => {
                if let EK::Builtin(b @ (B::WorkgroupBarrier | B::StorageBarrier), _) = e.kind {
                    let flags = if b == B::WorkgroupBarrier {
                        "mem_threadgroup"
                    } else {
                        "mem_device"
                    };
                    self.line(&format!(
                        "metal::threadgroup_barrier(metal::mem_flags::{flags});"
                    ));
                    return Ok(true);
                }
                let v = self.expr(e)?;
                self.flush_pre();
                if matches!(e.kind, EK::Call(..))
                    || matches!(e.kind, EK::Texture(ref t) if t.op == B::TextureStore)
                {
                    self.line(&format!("{v};"));
                } else {
                    self.line(&format!("(void)({v});"));
                }
                Ok(true)
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                let c = self.expr(cond)?;
                self.flush_pre();
                let tf = self.nested(&format!("if ({})", bare(&c)), then)?;
                let of = if otherwise.is_empty() {
                    true
                } else {
                    self.indent_back_else();
                    self.indent += 1;
                    let f = self.block(otherwise)?;
                    self.indent -= 1;
                    self.line("}");
                    f
                };
                Ok(tf || of)
            }
            Stmt::Loop {
                cond,
                body,
                continuing,
                break_if,
            } => {
                self.loop_stmt(cond.as_ref(), body, continuing, break_if.as_ref())?;
                // Whether the loop can exit is the checker's business; code after a loop
                // that never exits is unreachable but harmless in C++.
                Ok(true)
            }
            Stmt::Break => {
                self.line("break;");
                Ok(false)
            }
            Stmt::Continue => {
                self.line("continue;");
                Ok(false)
            }
            Stmt::Return(e) => {
                match e {
                    Some(e) => {
                        let v = self.expr(e)?;
                        self.flush_pre();
                        self.line(&format!("return {v};"));
                    }
                    None => self.line("return;"),
                }
                Ok(false)
            }
            Stmt::Discard => {
                self.line("metal::discard_fragment();");
                match self.fs.ret.clone() {
                    Some(t) => {
                        let z = self.zero(&t)?;
                        self.line(&format!("return {z};"));
                    }
                    None => self.line("return;"),
                }
                Ok(false)
            }
            Stmt::Block(b) => {
                self.line("{");
                self.indent += 1;
                let f = self.block(b)?;
                self.indent -= 1;
                self.line("}");
                Ok(f)
            }
        }
    }

    /// Turns the `}` that closed an `if` body into `} else {`.
    fn indent_back_else(&mut self) {
        let trimmed = self.out.trim_end_matches('\n').len();
        self.out.truncate(trimmed);
        self.out.push_str(" else {\n");
    }

    fn prog_fn_value_name(&self, id: ValueId) -> String {
        // Values are numbered per function.
        self.prog.functions[self.cur_fn]
            .value_names
            .get(id as usize)
            .cloned()
            .unwrap_or_else(|| "v".into())
    }

    fn loop_stmt(
        &mut self,
        cond: Option<&Expr>,
        body: &[Stmt],
        continuing: &[Stmt],
        break_if: Option<&Expr>,
    ) -> R<()> {
        let flag_form = has_own_continue(body) && (!continuing.is_empty() || break_if.is_some());
        let simple_cond = cond.filter(|c| !self.needs_hoist(c));
        if !flag_form && continuing.is_empty() && break_if.is_none() {
            if let Some(c) = simple_cond {
                let c = self.expr(c)?;
                self.line(&format!("while ({}) {{", bare(&c)));
                self.indent += 1;
                self.block(body)?;
                self.indent -= 1;
                self.line("}");
                return Ok(());
            }
        }
        if !flag_form {
            self.line("while (true) {");
            self.indent += 1;
            self.loop_cond(cond)?;
            if self.block(body)? {
                self.block(continuing)?;
                self.loop_break_if(break_if)?;
            }
            self.indent -= 1;
            self.line("}");
            return Ok(());
        }
        let flag = self.fs.names.generated("ganbate_continuing");
        self.line("{");
        self.indent += 1;
        self.line(&format!("bool {flag} = false;"));
        self.line("while (true) {");
        self.indent += 1;
        self.line(&format!("if ({flag}) {{"));
        self.indent += 1;
        if self.block(continuing)? {
            self.loop_break_if(break_if)?;
        }
        self.indent -= 1;
        self.line("}");
        self.line(&format!("{flag} = true;"));
        self.loop_cond(cond)?;
        self.block(body)?;
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    fn loop_cond(&mut self, cond: Option<&Expr>) -> R<()> {
        if let Some(c) = cond {
            let c = self.expr(c)?;
            self.flush_pre();
            self.line(&format!("if (!{}) {{", postfix(&c)));
            self.line("    break;");
            self.line("}");
        }
        Ok(())
    }

    fn loop_break_if(&mut self, break_if: Option<&Expr>) -> R<()> {
        if let Some(b) = break_if {
            let b = self.expr(b)?;
            self.flush_pre();
            self.line(&format!("if ({}) {{", bare(&b)));
            self.line("    break;");
            self.line("}");
        }
        Ok(())
    }
}

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
fn postfix(s: &str) -> String {
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
fn bare(s: &str) -> &str {
    if wrapped(s) { &s[1..s.len() - 1] } else { s }
}

fn is_trivial(e: &Expr) -> bool {
    matches!(e.kind, EK::Const(_) | EK::Zero | EK::Value(_))
}

impl<'p> Writer<'p> {
    // ------------------------------------------------------------------ places

    fn temp(&mut self, ty: &Ty, text: String) -> R<String> {
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

    fn load(&mut self, p: &Place) -> R<String> {
        let pt = self.place(p)?;
        Ok(match (pt.split, pt.packed) {
            (Some((base, un, access)), _) => format!("{un}({base}){access}"),
            (None, Some(un)) => format!("{un}({})", pt.text),
            (None, None) => pt.text,
        })
    }

    // ------------------------------------------------------------------ assignment

    fn assign(&mut self, place: &Place, swizzle: Option<&[u32]>, value: &Expr) -> R<()> {
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

    fn with_current(&mut self, current: String, global: bool, value: &Expr) -> R<String> {
        let saved = self.fs.current.replace(current);
        let saved_global = std::mem::replace(&mut self.current_global, global);
        let v = self.expr(value);
        self.fs.current = saved;
        self.current_global = saved_global;
        v
    }

    // ------------------------------------------------------------------ expressions

    fn ordered(&mut self, es: &[&Expr]) -> R<Vec<String>> {
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

    fn expr(&mut self, e: &Expr) -> R<String> {
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
    fn extract(&mut self, ty: &Ty, base: &str, i: u32) -> R<String> {
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

    fn binary(&mut self, op: BinaryOp, l: &Expr, r: &Expr, ty: &Ty) -> R<String> {
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
    fn splat_to(&mut self, text: &str, from: &Ty, to: &Ty) -> R<String> {
        if from.is_scalar() && !to.is_scalar() {
            Ok(format!("{}({text})", self.ty_name(to)?))
        } else {
            Ok(text.to_string())
        }
    }

    fn convert(&mut self, x: &Expr, to: &Ty) -> R<String> {
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

    fn builtin(&mut self, b: B, args: &[Expr], ty: &Ty) -> R<String> {
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

    fn texture(&mut self, tc: &TexCall) -> R<String> {
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
    fn entry(
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
