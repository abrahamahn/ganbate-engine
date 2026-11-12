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

mod decl;
mod expr;
