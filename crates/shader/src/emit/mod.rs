//! SPIR-V 1.0 code generation from the typed IR for one entry point.
//!
//! Composite types that can live in buffers get up to four SPIR-V type ids ("flavors"):
//! plain (Function/Private/Input/Output storage, no layout decorations), explicit
//! (`Offset`/`ArrayStride`/`MatrixStride` for nesting inside buffers), and the top-level
//! `Block` / `BufferBlock` variants. Values remember their flavor and are copied member-wise
//! whenever they cross into storage of another flavor.

use std::collections::{BTreeSet, HashMap};

use crate::ast::BinaryOp;
use crate::ir::{
    Builtin as B, Expr, ExprKind as EK, FnId, GlobalId, GlobalKind, Place, PlaceRoot, Program,
    StageReq, Step, Stmt, TexCall, UnaryOp, ValueId,
};
use crate::lexer::Span;
use crate::spirv::{self, decoration as deco, glsl, inst, op, storage_class as sc};
use crate::types::*;
use crate::{
    Binding, BindingKind, Compiled, Error, Interface, IoType, SampleType, Stage, TexelFormat,
    TextureView,
};

type R<T> = Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Flavor {
    Plain,
    Explicit,
    Block,
    BufferBlock { read_only: bool },
}

impl Flavor {
    /// Flavor of a member/element of a composite of this flavor.
    fn member(self) -> Flavor {
        if self == Flavor::Plain {
            Flavor::Plain
        } else {
            Flavor::Explicit
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Val {
    id: u32,
    fl: Flavor,
}

impl Val {
    fn plain(id: u32) -> Val {
        Val {
            id,
            fl: Flavor::Plain,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum TypeKey {
    Void,
    Bool,
    Int(bool),
    Float,
    Vector(u32, u8),
    Matrix(u32, u8),
    Array(u32, u32, Option<u32>),
    RuntimeArray(u32, u32),
    Struct(usize, Flavor),
    Wrapper(GlobalId),
    Pointer(u32, u32),
    Function(u32, Vec<u32>),
    Image(u32, u32, u32, u32, u32, u32),
    Sampler,
    SampledImage(u32),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum ConstKey {
    Scalar(u32, u32),
    Bool(bool),
    Composite(u32, Vec<u32>),
    Null(u32),
}

#[derive(Clone, Copy)]
struct GlobalInfo {
    id: u32,
    sc: u32,
    /// Flavor of the variable's WGSL-visible type.
    fl: Flavor,
    /// The WGSL type is member 0 of a wrapper struct.
    wrapped: bool,
}

struct LoopTargets {
    merge: u32,
    cont: u32,
    has_break: bool,
    has_continue: bool,
}

#[derive(Default)]
struct FnState {
    vars: Vec<u32>,
    body: Vec<u32>,
    label: u32,
    terminated: bool,
    values: HashMap<ValueId, Val>,
    locals: Vec<u32>,
    loops: Vec<LoopTargets>,
    current: Option<Val>,
}

struct Emitter<'p> {
    prog: &'p Program,
    s: spirv::Sections,
    next: u32,
    types: HashMap<TypeKey, u32>,
    consts: HashMap<ConstKey, u32>,
    glsl: u32,
    caps: BTreeSet<u32>,
    globals: HashMap<GlobalId, GlobalInfo>,
    fn_ids: HashMap<FnId, u32>,
    fs: FnState,
    span: Span,
    cur_fn: FnId,
    /// Ids that already carry an `OpName` (identity operations can return an operand's id).
    named: BTreeSet<u32>,
}

fn internal(span: Span, what: &str) -> Error {
    span.error(format!("internal compiler error: {what}"))
}

/// Can `e` be evaluated unconditionally (no calls, side effects, dynamic indexing, reads of
/// runtime-sized storage arrays, or integer division)? Decides whether `&&` / `||` need real
/// short-circuit control flow.
fn is_simple(prog: &Program, e: &Expr) -> bool {
    let simple = |x: &Expr| is_simple(prog, x);
    match &e.kind {
        EK::Const(_) | EK::Zero | EK::Value(_) | EK::Current | EK::ArrayLength { .. } => true,
        EK::Load(p) => {
            // Constant indices are bounds-checked at compile time, except into runtime-sized
            // arrays, which only exist in storage buffers.
            let storage = matches!(p.root, PlaceRoot::Global(g) if matches!(prog.globals[g].kind, GlobalKind::Storage { .. }));
            p.steps.iter().all(|s| match s {
                Step::Member(_) => true,
                Step::Index(i) => i.as_const().is_some() && !storage,
            })
        }
        EK::Extract(b, _)
        | EK::Swizzle(b, _)
        | EK::Unary(_, b)
        | EK::Splat(b)
        | EK::Convert(b)
        | EK::Bitcast(b) => simple(b),
        EK::Binary(op, l, r) => {
            let int_div = matches!(op, BinaryOp::Div | BinaryOp::Rem)
                && e.ty.scalar().is_some_and(|s| s.is_int());
            !int_div && simple(l) && simple(r)
        }
        EK::Construct(parts) => parts.iter().all(simple),
        EK::Builtin(b, args) => {
            !matches!(
                b,
                B::Dpdx | B::Dpdy | B::Fwidth | B::WorkgroupBarrier | B::StorageBarrier
            ) && args.iter().all(simple)
        }
        EK::DynIndex(..) | EK::Call(..) | EK::Texture(_) => false,
    }
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
        _ => false,
    }
}

fn io_type(ty: &Ty) -> Option<IoType> {
    Some(match ty {
        Ty::Scalar(Scalar::F32) => IoType::F32,
        Ty::Scalar(Scalar::I32) => IoType::I32,
        Ty::Scalar(Scalar::U32) => IoType::U32,
        Ty::Vector(2, Scalar::F32) => IoType::Vec2F,
        Ty::Vector(3, Scalar::F32) => IoType::Vec3F,
        Ty::Vector(4, Scalar::F32) => IoType::Vec4F,
        Ty::Vector(2, Scalar::I32) => IoType::Vec2I,
        Ty::Vector(3, Scalar::I32) => IoType::Vec3I,
        Ty::Vector(4, Scalar::I32) => IoType::Vec4I,
        Ty::Vector(2, Scalar::U32) => IoType::Vec2U,
        Ty::Vector(3, Scalar::U32) => IoType::Vec3U,
        Ty::Vector(4, Scalar::U32) => IoType::Vec4U,
        _ => return None,
    })
}

/// Functions reachable from `entry`, in a deterministic order.
fn reachable(prog: &Program, entry: FnId) -> Vec<FnId> {
    let mut seen = BTreeSet::new();
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

pub fn emit(prog: &Program, stage: Stage, entry: &str) -> R<Compiled> {
    let Some(fid) = prog.functions.iter().position(|f| f.name == entry) else {
        let mut eps: Vec<String> = prog
            .functions
            .iter()
            .filter(|f| f.stage == Some(stage))
            .map(|f| format!("'{}'", f.name))
            .collect();
        eps.sort();
        let hint = if eps.is_empty() {
            String::new()
        } else {
            format!(" (available {stage:?} entry points: {})", eps.join(", ")).to_lowercase()
        };
        return Err(Error {
            line: 1,
            column: 1,
            message: format!("no entry point named '{entry}'{hint}"),
        });
    };
    let f = &prog.functions[fid];
    match f.stage {
        None => {
            return Err(f.span.error(format!(
                "'{entry}' is not an entry point (it has no @vertex, @fragment or @compute attribute)"
            )));
        }
        Some(s) if s != stage => {
            return Err(f
                .stage_span
                .error(format!("'{entry}' is a {s:?} entry point, not {stage:?}").to_lowercase()));
        }
        _ => {}
    }
    let reach = reachable(prog, fid);
    for &r in &reach {
        for (req, span, name) in &prog.functions[r].stage_reqs {
            let ok = matches!(
                (req, stage),
                (StageReq::Fragment, Stage::Fragment) | (StageReq::Compute, Stage::Compute)
            );
            if !ok {
                let need = if *req == StageReq::Fragment {
                    "fragment"
                } else {
                    "compute"
                };
                return Err(span.error(format!(
                    "'{name}' can only be used in {need} shaders, but is reachable from the {} entry point '{entry}'",
                    format!("{stage:?}").to_lowercase()
                )));
            }
        }
    }
    let used: BTreeSet<GlobalId> = reach
        .iter()
        .flat_map(|&r| prog.functions[r].globals.iter().copied())
        .collect();

    // Reflection of resources.
    let mut bindings: Vec<(Binding, GlobalId)> = Vec::new();
    let mut push_constant: Option<GlobalId> = None;
    for &g in &used {
        let gl = &prog.globals[g];
        if gl.kind == GlobalKind::PushConstant {
            if let Some(prev) = push_constant {
                return Err(gl.span.error(format!(
                    "entry point '{entry}' uses two push constant blocks ('{}' and '{}'); at most one is allowed",
                    prog.globals[prev].name, gl.name
                )));
            }
            push_constant = Some(g);
        }
        let Some((group, binding)) = gl.binding else {
            continue;
        };
        let kind = match (&gl.kind, &gl.ty) {
            (GlobalKind::Uniform, ty) => BindingKind::UniformBuffer {
                size: size_of(ty, &prog.structs),
            },
            (GlobalKind::Storage { read_only }, _) => BindingKind::StorageBuffer {
                read_only: *read_only,
            },
            (_, Ty::Texture(view, sample)) => BindingKind::Texture {
                view: *view,
                sample: *sample,
            },
            (_, Ty::StorageTexture(format)) => BindingKind::StorageTexture {
                view: TextureView::D2,
                format: *format,
            },
            (_, Ty::Sampler(c)) => BindingKind::Sampler { comparison: *c },
            _ => return Err(internal(gl.span, "unexpected resource kind")),
        };
        if let Some((_, other)) = bindings
            .iter()
            .find(|(b, _)| b.group == group && b.binding == binding)
        {
            return Err(gl.span.error(format!(
                "'{}' and '{}' both use @group({group}) @binding({binding}) in entry point '{entry}'",
                prog.globals[*other].name, gl.name
            )));
        }
        bindings.push((
            Binding {
                group,
                binding,
                kind,
            },
            g,
        ));
    }
    bindings.sort_by_key(|(b, _)| (b.group, b.binding));

    let mut e = Emitter {
        prog,
        s: spirv::Sections::default(),
        next: 1,
        types: HashMap::new(),
        consts: HashMap::new(),
        glsl: 0,
        caps: BTreeSet::new(),
        globals: HashMap::new(),
        fn_ids: HashMap::new(),
        fs: FnState::default(),
        span: f.span,
        cur_fn: fid,
        named: BTreeSet::new(),
    };
    e.caps.insert(spirv::capability::SHADER);
    e.glsl = e.id();
    let mut imp = vec![e.glsl];
    imp.extend(spirv::string("GLSL.std.450"));
    inst(&mut e.s.ext_imports, op::EXT_INST_IMPORT, &imp);
    inst(
        &mut e.s.memory_model,
        op::MEMORY_MODEL,
        &[spirv::ADDRESSING_LOGICAL, spirv::MEMORY_MODEL_GLSL450],
    );
    for &g in &used {
        e.global_var(g)?;
    }
    for &r in &reach {
        let id = e.id();
        e.fn_ids.insert(r, id);
    }
    for &r in &reach {
        e.function(r)?;
    }
    let io = e.entry_wrapper(fid, stage)?;

    let model = match stage {
        Stage::Vertex => spirv::execution_model::VERTEX,
        Stage::Fragment => spirv::execution_model::FRAGMENT,
        Stage::Compute => spirv::execution_model::GL_COMPUTE,
    };
    let mut ep = vec![model, io.main];
    ep.extend(spirv::string("main"));
    ep.extend(&io.interface);
    inst(&mut e.s.entry_points, op::ENTRY_POINT, &ep);
    match stage {
        Stage::Fragment => {
            inst(
                &mut e.s.execution_modes,
                op::EXECUTION_MODE,
                &[io.main, spirv::execution_mode::ORIGIN_UPPER_LEFT],
            );
            if io.depth_replacing {
                inst(
                    &mut e.s.execution_modes,
                    op::EXECUTION_MODE,
                    &[io.main, spirv::execution_mode::DEPTH_REPLACING],
                );
            }
        }
        Stage::Compute => {
            let [x, y, z] = f.workgroup_size;
            inst(
                &mut e.s.execution_modes,
                op::EXECUTION_MODE,
                &[io.main, spirv::execution_mode::LOCAL_SIZE, x, y, z],
            );
        }
        Stage::Vertex => {}
    }
    for &c in &e.caps {
        inst(&mut e.s.capabilities, op::CAPABILITY, &[c]);
    }
    let bound = e.next;
    let words = std::mem::take(&mut e.s).assemble(bound);
    Ok(Compiled {
        stage,
        words,
        bindings: bindings.into_iter().map(|(b, _)| b).collect(),
        push_constant_size: push_constant
            .map_or(0, |g| size_of(&prog.globals[g].ty, &prog.structs)),
        inputs: io.inputs,
        outputs: io.outputs,
        workgroup_size: if stage == Stage::Compute {
            f.workgroup_size
        } else {
            [1, 1, 1]
        },
        msl: None,
        wgsl: None,
        glsl: None,
    })
}

struct EntryIo {
    main: u32,
    interface: Vec<u32>,
    inputs: Vec<Interface>,
    outputs: Vec<Interface>,
    depth_replacing: bool,
}

mod decl;
mod expr;
