//! Name resolution and type checking: AST → typed IR.
//!
//! Implements WGSL's abstract numeric typing: integer and float literals are AbstractInt /
//! AbstractFloat constants that are folded at compile time and converted to the concrete type
//! the context demands (other operand, declared type, parameter, constructor component,
//! return type), or concretized to i32 / f32 when nothing constrains them.

use std::collections::{BTreeSet, HashMap};

use crate::ast::{self, BinaryOp, ExprKind as AK, StmtKind as SK};
use crate::consteval;
use crate::ir::{
    self, Expr, ExprKind as EK, FnId, GlobalId, GlobalKind, LocalId, Place, PlaceRoot, StageReq,
    Step, Stmt, ValueId,
};
use crate::lexer::{FloatLit, IntLit, Span};
use crate::types::*;
use crate::{Error, SampleType, SpecValue, Stage, TexelFormat, TextureView};

pub(crate) type R<T> = Result<T, Error>;

/// Maximum depth of lazily resolved module-scope declarations that refer to later ones.
const MAX_RESOLVE_DEPTH: u32 = 64;

#[cfg(test)]
pub fn check(module: &ast::Module) -> R<ir::Program> {
    check_with(module, &[])
}

pub fn check_with(module: &ast::Module, specialize: &[(&str, SpecValue)]) -> R<ir::Program> {
    let mut c = Checker::new(module, specialize)?;
    for i in 0..module.decls.len() {
        c.resolve_decl(i)?;
    }
    for fid in 0..c.functions.len() {
        c.check_body(fid)?;
    }
    c.check_recursion()?;
    for fid in 0..c.functions.len() {
        if c.functions[fid].stage.is_some() {
            c.validate_entry(fid)?;
        }
    }
    Ok(ir::Program {
        structs: c.structs,
        globals: c.globals,
        functions: c.functions,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Todo,
    Busy,
    Done,
}

#[derive(Clone)]
enum Resolved {
    Pending,
    Struct(usize),
    Alias(Ty),
    Const(Ty, ConstValue),
    Global(GlobalId),
    Fn(FnId),
    Assert,
}

#[derive(Clone)]
pub(crate) enum Binding {
    Value {
        id: ValueId,
        ty: Ty,
        /// Index of the loop whose `continuing` block must not see this `let` because a
        /// `continue` precedes its declaration.
        after_continue_of: Option<usize>,
    },
    Var {
        id: LocalId,
        ty: Ty,
    },
    Const {
        ty: Ty,
        value: ConstValue,
    },
}

#[derive(Clone)]
struct LoopCx {
    body_scope: usize,
    continue_seen: bool,
    has_break: bool,
    in_continuing: bool,
}

pub(crate) struct FnCx {
    scopes: Vec<Vec<(String, Binding)>>,
    locals: Vec<ir::Local>,
    value_names: Vec<String>,
    ret: Option<Ty>,
    loops: Vec<LoopCx>,
    continuing_of: Vec<usize>,
    calls: BTreeSet<FnId>,
    pub(crate) globals: BTreeSet<GlobalId>,
    pub(crate) stage_reqs: Vec<(StageReq, Span, String)>,
}

/// Result of checking an expression: a value, or a reference (WGSL's memory view) that the
/// load rule turns into a value when needed.
pub(crate) enum Checked {
    Value(Expr),
    Ref(Place),
    /// Multi-component swizzle of a vector reference (readable, and assignable in Ganbate).
    Swizzle(Place, Vec<u32>),
    Handle(GlobalId, Ty),
}

enum Ctor {
    Full(Ty),
    Vec(u8),
    Mat(u8, u8),
    Array,
}

struct Target {
    place: Place,
    swizzle: Option<Vec<u32>>,
    ty: Ty,
}

pub(crate) struct Checker<'a> {
    module: &'a ast::Module,
    names: HashMap<&'a str, usize>,
    specialize: HashMap<&'a str, SpecValue>,
    states: Vec<State>,
    /// Current depth of nested lazy declaration resolution.
    resolving: u32,
    dead_depth: u32,
    resolved: Vec<Resolved>,
    pub(crate) structs: Vec<StructInfo>,
    pub(crate) globals: Vec<ir::Global>,
    pub(crate) functions: Vec<ir::Function>,
    fn_decls: Vec<&'a ast::FnDecl>,
    const_data: Vec<(Ty, ConstValue, GlobalId)>,
    pub(crate) fcx: Option<FnCx>,
}

pub(crate) fn konst(ty: Ty, v: ConstValue, span: Span) -> Expr {
    Expr {
        ty,
        kind: EK::Const(v),
        span,
    }
}

fn vec_alias(name: &str) -> Option<(u8, Scalar)> {
    let b = name.as_bytes();
    if b.len() == 5 && name.starts_with("vec") && (b'2'..=b'4').contains(&b[3]) {
        let s = match b[4] {
            b'f' => Scalar::F32,
            b'i' => Scalar::I32,
            b'u' => Scalar::U32,
            _ => return None,
        };
        return Some((b[3] - b'0', s));
    }
    None
}

/// `matCxR` → (C, R, suffix).
fn mat_name(name: &str) -> Option<(u8, u8, &str)> {
    let b = name.as_bytes();
    if b.len() >= 6
        && name.starts_with("mat")
        && b[4] == b'x'
        && (b'2'..=b'4').contains(&b[3])
        && (b'2'..=b'4').contains(&b[5])
    {
        return Some((b[3] - b'0', b[5] - b'0', &name[6..]));
    }
    None
}

fn parse_swizzle(s: &str, n: u8) -> Option<Vec<u32>> {
    if s.is_empty() || s.len() > 4 {
        return None;
    }
    let xyzw = s.chars().all(|c| "xyzw".contains(c));
    let rgba = s.chars().all(|c| "rgba".contains(c));
    if !xyzw && !rgba {
        return None;
    }
    let comps: Vec<u32> = s
        .chars()
        .map(|c| match c {
            'x' | 'r' => 0,
            'y' | 'g' => 1,
            'z' | 'b' => 2,
            _ => 3,
        })
        .collect();
    if comps.iter().any(|&c| c >= n as u32) {
        return None;
    }
    Some(comps)
}

fn same_shape(a: &Ty, b: &Ty) -> bool {
    match (a, b) {
        (Ty::Scalar(_), Ty::Scalar(_)) => true,
        (Ty::Vector(n, _), Ty::Vector(m, _)) => n == m,
        (Ty::Array(x, n), Ty::Array(y, m)) => n == m && same_shape(x, y),
        _ => false,
    }
}

fn contains_runtime_array(ty: &Ty, structs: &[StructInfo]) -> bool {
    match ty {
        Ty::Array(_, None) => true,
        Ty::Array(e, _) => contains_runtime_array(e, structs),
        Ty::Struct(id) => structs[*id]
            .members
            .iter()
            .any(|m| contains_runtime_array(&m.ty, structs)),
        _ => false,
    }
}

fn texel_format(name: &str) -> Option<TexelFormat> {
    Some(match name {
        "rgba8unorm" => TexelFormat::Rgba8Unorm,
        "rgba16float" => TexelFormat::Rgba16Float,
        "rgba32float" => TexelFormat::Rgba32Float,
        "r32float" => TexelFormat::R32Float,
        "rg16float" => TexelFormat::Rg16Float,
        "r16float" => TexelFormat::R16Float,
        _ => return None,
    })
}

impl<'a> Checker<'a> {
    fn new(module: &'a ast::Module, specialize_list: &'a [(&'a str, SpecValue)]) -> R<Self> {
        let mut names: HashMap<&'a str, usize> = HashMap::new();
        for (i, d) in module.decls.iter().enumerate() {
            if let Some(n) = d.name() {
                if let Some(&prev) = names.get(n.name.as_str()) {
                    let p = module.decls[prev]
                        .name()
                        .map(|x| x.span)
                        .unwrap_or_default();
                    return Err(n.span.error(format!(
                        "'{}' is already declared at line {}:{}",
                        n.name, p.line, p.col
                    )));
                }
                names.insert(&n.name, i);
            }
        }
        let mut specialize = HashMap::with_capacity(specialize_list.len());
        for &(spec_name, spec_val) in specialize_list {
            let Some(&decl_idx) = names.get(spec_name) else {
                return Err(Error {
                    line: 1,
                    column: 1,
                    message: format!("unknown specialization constant '{spec_name}'"),
                });
            };
            match &module.decls[decl_idx] {
                ast::Decl::Const { .. } => {
                    specialize.insert(spec_name, spec_val);
                }
                other => {
                    let span = other.name().map(|n| n.span).unwrap_or_default();
                    return Err(span.error(format!(
                        "'{spec_name}' cannot be specialized because it is not a module-scope 'const'"
                    )));
                }
            }
        }
        Ok(Checker {
            module,
            names,
            specialize,
            states: vec![State::Todo; module.decls.len()],
            resolving: 0,
            dead_depth: 0,
            resolved: vec![Resolved::Pending; module.decls.len()],
            structs: Vec::new(),
            globals: Vec::new(),
            functions: Vec::new(),
            fn_decls: Vec::new(),
            const_data: Vec::new(),
            fcx: None,
        })
    }

    pub(crate) fn ty_name(&self, ty: &Ty) -> String {
        type_name(ty, &self.structs)
    }

    pub(crate) fn fcx_mut(&mut self) -> &mut FnCx {
        self.fcx.as_mut().expect("function context")
    }

    // ------------------------------------------------------------------ module scope

    fn resolve_decl(&mut self, i: usize) -> R<()> {
        if self.states[i] == State::Todo && self.resolving >= MAX_RESOLVE_DEPTH {
            let n = self.module.decls[i]
                .name()
                .map(|n| n.span)
                .unwrap_or_default();
            return Err(n.error(format!(
                "declarations depend on each other more than {MAX_RESOLVE_DEPTH} levels deep; declare them in dependency order"
            )));
        }
        match self.states[i] {
            State::Done => return Ok(()),
            State::Busy => {
                let d = &self.module.decls[i];
                let n = d.name().expect("named declaration");
                return Err(n.span.error(format!(
                    "'{}' is defined in terms of itself (cyclic declaration)",
                    n.name
                )));
            }
            State::Todo => {}
        }
        self.states[i] = State::Busy;
        self.resolving += 1;
        // Module-scope declarations never see a function's local scope.
        let saved = self.fcx.take();
        let module = self.module;
        let r = match &module.decls[i] {
            ast::Decl::Struct(s) => self.resolve_struct(s).map(Resolved::Struct),
            ast::Decl::Alias { ty, .. } => self.resolve_type(ty).map(Resolved::Alias),
            ast::Decl::Const { name, ty, init } => {
                let (decl_ty, default_val) = self.const_decl(ty.as_ref(), init)?;
                if let Some(&spec_val) = self.specialize.get(name.name.as_str()) {
                    self.apply_spec_const(name, decl_ty, spec_val)
                        .map(|(t, v)| Resolved::Const(t, v))
                } else {
                    Ok(Resolved::Const(decl_ty, default_val))
                }
            }
            ast::Decl::Var(v) => self.global_var(v).map(Resolved::Global),
            ast::Decl::Fn(f) => self.fn_signature(f).map(Resolved::Fn),
            ast::Decl::ConstAssert(e) => self.const_assert(e).map(|_| Resolved::Assert),
        };
        self.fcx = saved;
        self.resolving -= 1;
        self.resolved[i] = r?;
        self.states[i] = State::Done;
        Ok(())
    }

    fn module_item(&mut self, name: &str) -> R<Option<Resolved>> {
        match self.names.get(name) {
            Some(&i) => {
                self.resolve_decl(i)?;
                Ok(Some(self.resolved[i].clone()))
            }
            None => Ok(None),
        }
    }

    fn const_decl(&mut self, ty: Option<&ast::Expr>, init: &ast::Expr) -> R<(Ty, ConstValue)> {
        let v = self.value(init)?;
        let v = match ty {
            Some(t) => {
                let t = self.resolve_type(t)?;
                self.convert(v, &t, init.span)?
            }
            None => v,
        };
        match v.kind {
            EK::Const(c) => Ok((v.ty, c)),
            _ => Err(init
                .span
                .error("the initializer of a 'const' must be a constant expression")),
        }
    }

    fn apply_spec_const(
        &self,
        name: &ast::Ident,
        decl_ty: Ty,
        spec: SpecValue,
    ) -> R<(Ty, ConstValue)> {
        match (&decl_ty, spec) {
            (Ty::Scalar(Scalar::Bool), SpecValue::Bool(b)) => Ok((BOOL, ConstValue::Bool(b))),
            (Ty::Scalar(Scalar::I32), SpecValue::I32(v)) => Ok((I32, ConstValue::I32(v))),
            (Ty::Scalar(Scalar::AbstractInt), SpecValue::I32(v)) => Ok((
                Ty::Scalar(Scalar::AbstractInt),
                ConstValue::AbstractInt(v as i64),
            )),
            (Ty::Scalar(Scalar::U32), SpecValue::U32(v)) => Ok((U32, ConstValue::U32(v))),
            (Ty::Scalar(Scalar::AbstractInt), SpecValue::U32(v)) => Ok((
                Ty::Scalar(Scalar::AbstractInt),
                ConstValue::AbstractInt(v as i64),
            )),
            (Ty::Scalar(Scalar::F32), SpecValue::F32(v)) => {
                if !v.is_finite() {
                    return Err(name.span.error("specialization float value must be finite"));
                }
                Ok((F32, ConstValue::F32(v)))
            }
            (Ty::Scalar(Scalar::AbstractFloat), SpecValue::F32(v))
            | (Ty::Scalar(Scalar::AbstractInt), SpecValue::F32(v)) => {
                if !v.is_finite() {
                    return Err(name.span.error("specialization float value must be finite"));
                }
                Ok((
                    Ty::Scalar(Scalar::AbstractFloat),
                    ConstValue::AbstractFloat(v as f64),
                ))
            }
            _ => Err(name.span.error(format!(
                "cannot specialize '{}' of type {} with {}",
                name.name,
                self.ty_name(&decl_ty),
                spec.type_name()
            ))),
        }
    }

    fn const_assert(&mut self, e: &ast::Expr) -> R<()> {
        let v = self.value(e)?;
        match v.as_const() {
            Some(ConstValue::Bool(true)) => Ok(()),
            Some(ConstValue::Bool(false)) => Err(e.span.error("const_assert failed")),
            _ => Err(e
                .span
                .error("const_assert requires a constant bool expression")),
        }
    }

    fn attr_int(&mut self, a: &ast::Attr) -> R<u32> {
        if a.args.len() != 1 {
            return Err(a
                .span
                .error(format!("@{} expects exactly one argument", a.name)));
        }
        let v = self.value(&a.args[0])?;
        match v.as_const().and_then(|c| c.as_i64()) {
            Some(n) if (0..=u32::MAX as i64).contains(&n) => Ok(n as u32),
            _ => Err(a.args[0].span.error(format!(
                "@{} requires a non-negative integer constant",
                a.name
            ))),
        }
    }

    fn attr_ident(a: &ast::Attr, i: usize) -> R<&str> {
        match a.args.get(i).map(|e| &e.kind) {
            Some(AK::Ident {
                name,
                templates: None,
            }) => Ok(name),
            _ => Err(a
                .span
                .error(format!("@{} expects an identifier argument", a.name))),
        }
    }

    /// IO attributes plus `@align` / `@size` (the latter only where `layout_attrs` is set).
    fn io_attrs(
        &mut self,
        attrs: &[ast::Attr],
        layout_attrs: bool,
    ) -> R<(IoAttrs, Option<u32>, Option<u32>)> {
        let mut io = IoAttrs::default();
        let (mut align, mut size) = (None, None);
        for a in attrs {
            if io.span == Span::default() {
                io.span = a.span;
            }
            match a.name.as_str() {
                "location" => io.location = Some(self.attr_int(a)?),
                "builtin" => {
                    if a.args.len() != 1 {
                        return Err(a.span.error("@builtin expects one argument"));
                    }
                    let n = Self::attr_ident(a, 0)?;
                    io.builtin = Some(BuiltinValue::from_name(n).ok_or_else(|| {
                        a.args[0].span.error(match n {
                            "sample_mask"
                            | "clip_distances"
                            | "primitive_index"
                            | "subgroup_size"
                            | "subgroup_invocation_id" => {
                                format!("@builtin({n}) is not supported by Ganbate")
                            }
                            _ => format!("unknown built-in value '{n}'"),
                        })
                    })?);
                }
                "interpolate" => {
                    if a.args.is_empty() || a.args.len() > 2 {
                        return Err(a.span.error("@interpolate expects one or two arguments"));
                    }
                    io.interpolation = Some(match Self::attr_ident(a, 0)? {
                        "perspective" => Interpolation::Perspective,
                        "linear" => Interpolation::Linear,
                        "flat" => Interpolation::Flat,
                        other => {
                            return Err(a.args[0]
                                .span
                                .error(format!("unknown interpolation type '{other}'")));
                        }
                    });
                    if a.args.len() == 2 {
                        let s = Self::attr_ident(a, 1)?;
                        io.sampling = Some(match s {
                            "center" => Sampling::Center,
                            "centroid" => Sampling::Centroid,
                            "sample" => Sampling::Sample,
                            "first" | "either" if io.interpolation == Some(Interpolation::Flat) => {
                                Sampling::Center
                            }
                            other => {
                                return Err(a.args[1]
                                    .span
                                    .error(format!("unknown interpolation sampling '{other}'")));
                            }
                        });
                        if io.interpolation == Some(Interpolation::Flat)
                            && !matches!(s, "first" | "either")
                        {
                            return Err(a.args[1]
                                .span
                                .error("flat interpolation cannot have a sampling mode"));
                        }
                    }
                }
                "invariant" => io.invariant = true,
                "align" if layout_attrs => {
                    let n = self.attr_int(a)?;
                    if n == 0 || !n.is_power_of_two() {
                        return Err(a.span.error("@align requires a positive power of two"));
                    }
                    align = Some(n);
                }
                "size" if layout_attrs => {
                    let n = self.attr_int(a)?;
                    if n == 0 {
                        return Err(a.span.error("@size must be positive"));
                    }
                    size = Some(n);
                }
                "blend_src" => {
                    return Err(a
                        .span
                        .error("@blend_src (dual-source blending) is not supported by Ganbate"));
                }
                other => {
                    return Err(a
                        .span
                        .error(format!("attribute @{other} is not valid here")));
                }
            }
        }
        if io.location.is_some() && io.builtin.is_some() {
            return Err(io.span.error("@location and @builtin cannot be combined"));
        }
        if io.location.is_none() && (io.interpolation.is_some() || io.sampling.is_some()) {
            return Err(io.span.error("@interpolate requires @location"));
        }
        if io.invariant && io.builtin != Some(BuiltinValue::Position) {
            return Err(io
                .span
                .error("@invariant is only allowed on @builtin(position)"));
        }
        Ok((io, align, size))
    }

    fn resolve_struct(&mut self, s: &ast::StructDecl) -> R<usize> {
        if s.members.is_empty() {
            return Err(s.name.span.error("a struct must have at least one member"));
        }
        let mut members: Vec<MemberInfo> = Vec::new();
        for (i, m) in s.members.iter().enumerate() {
            if members.iter().any(|x| x.name == m.name.name) {
                return Err(m.name.span.error(format!(
                    "duplicate member '{}' in struct '{}'",
                    m.name.name, s.name.name
                )));
            }
            let ty = self.resolve_type(&m.ty)?;
            if ty.is_handle() {
                return Err(m
                    .ty
                    .span
                    .error("struct members cannot be textures or samplers"));
            }
            if matches!(ty, Ty::Array(_, None)) {
                if i + 1 != s.members.len() {
                    return Err(m
                        .ty
                        .span
                        .error("a runtime-sized array must be the last member of a struct"));
                }
            } else if contains_runtime_array(&ty, &self.structs) {
                return Err(m.ty.span.error(
                    "a struct containing a runtime-sized array cannot be nested in another struct",
                ));
            }
            let (io, align_attr, size_attr) = self.io_attrs(&m.attrs, true)?;
            members.push(MemberInfo {
                name: m.name.name.clone(),
                span: m.name.span,
                ty,
                io,
                align_attr,
                size_attr,
            });
        }
        let layout = if members
            .iter()
            .all(|m| is_host_shareable(&m.ty, &self.structs))
        {
            let mut offset = 0u32;
            let mut align = 1u32;
            let mut offsets = Vec::new();
            for m in &members {
                let a = m
                    .align_attr
                    .unwrap_or_else(|| align_of(&m.ty, &self.structs));
                let natural = size_of(&m.ty, &self.structs);
                let sz = match m.size_attr {
                    Some(n) if n < natural => {
                        return Err(m.span.error(format!(
                            "@size({n}) is smaller than the member's size ({natural})"
                        )));
                    }
                    Some(n) => n,
                    None => natural,
                };
                offset = round_up(a, offset);
                offsets.push(offset);
                offset = offset
                    .checked_add(sz)
                    .ok_or_else(|| m.span.error("struct is too large"))?;
                align = align.max(a);
            }
            Some(StructLayout {
                offsets,
                align,
                size: round_up(align, offset),
            })
        } else {
            None
        };
        let depth = 1 + members
            .iter()
            .map(|m| type_depth(&m.ty, &self.structs))
            .max()
            .unwrap_or(0);
        if depth > MAX_TYPE_DEPTH {
            return Err(s.name.span.error(format!(
                "struct '{}' nests arrays and structs more than {MAX_TYPE_DEPTH} levels deep",
                s.name.name
            )));
        }
        if let Some(l) = &layout
            && l.size as u64 >= MAX_TYPE_SIZE
        {
            return Err(s
                .name
                .span
                .error(format!("struct '{}' is too large", s.name.name)));
        }
        self.structs.push(StructInfo {
            name: s.name.name.clone(),
            members,
            layout,
            depth,
        });
        Ok(self.structs.len() - 1)
    }

    pub(crate) fn resolve_type(&mut self, e: &ast::Expr) -> R<Ty> {
        let AK::Ident { name, templates } = &e.kind else {
            return Err(e.span.error("expected a type"));
        };
        let span = e.span;
        if self.names.contains_key(name.as_str()) {
            if templates.is_some() {
                return Err(span.error(format!("'{name}' does not take template arguments")));
            }
            return match self.module_item(name)? {
                Some(Resolved::Struct(id)) => Ok(Ty::Struct(id)),
                Some(Resolved::Alias(t)) => Ok(t),
                _ => Err(span.error(format!("'{name}' is not a type"))),
            };
        }
        match self.predeclared_type(name, templates.as_deref(), span)? {
            Some(t) => Ok(t),
            None => Err(span.error(format!("unknown type '{name}'"))),
        }
    }

    fn template_scalar(&mut self, t: &ast::Expr, what: &str) -> R<Scalar> {
        match self.resolve_type(t)? {
            Ty::Scalar(s) if !s.is_abstract() => Ok(s),
            other => Err(t.span.error(format!(
                "{what} must be a scalar type, found {}",
                self.ty_name(&other)
            ))),
        }
    }

    fn predeclared_type(
        &mut self,
        name: &str,
        templates: Option<&[ast::Expr]>,
        span: Span,
    ) -> R<Option<Ty>> {
        let plain = |t: Ty| -> R<Option<Ty>> {
            if templates.is_some() {
                Err(span.error(format!("type '{name}' does not take template arguments")))
            } else {
                Ok(Some(t))
            }
        };
        if name == "f16"
            || name.ends_with('h')
                && (vec_alias(&name.replace('h', "f")).is_some()
                    || mat_name(name).is_some_and(|m| m.2 == "h"))
        {
            return Err(span.error("f16 is not supported by Ganbate"));
        }
        if let Some((n, s)) = vec_alias(name) {
            return plain(Ty::Vector(n, s));
        }
        if let Some((c, r, suffix)) = mat_name(name) {
            return match suffix {
                "f" => plain(Ty::Matrix(c, r)),
                "" => match templates {
                    Some([t]) => match self.template_scalar(t, "the matrix component type")? {
                        Scalar::F32 => Ok(Some(Ty::Matrix(c, r))),
                        _ => Err(t.span.error("matrix components must be f32")),
                    },
                    _ => Err(span.error(format!(
                        "'{name}' requires a component type, e.g. {name}<f32>"
                    ))),
                },
                _ => Ok(None),
            };
        }
        let tmpl = templates.unwrap_or(&[]);
        Ok(Some(match name {
            "bool" => return plain(BOOL),
            "i32" => return plain(I32),
            "u32" => return plain(U32),
            "f32" => return plain(F32),
            "vec2" | "vec3" | "vec4" => {
                let n = name.as_bytes()[3] - b'0';
                match tmpl {
                    [t] => Ty::Vector(n, self.template_scalar(t, "the vector component type")?),
                    _ => {
                        return Err(span.error(format!(
                            "'{name}' requires a component type, e.g. {name}<f32>"
                        )));
                    }
                }
            }
            "array" => match tmpl {
                [t] => {
                    let e = self.array_elem(t)?;
                    let t = Ty::Array(Box::new(e), None);
                    self.check_type_limits(&t, span)?;
                    t
                }
                [t, n] => {
                    let e = self.array_elem(t)?;
                    let v = self.value(n)?;
                    let len = match v.as_const().and_then(|c| c.as_i64()) {
                        Some(len) if len > 0 && len <= u32::MAX as i64 => len as u32,
                        Some(_) => {
                            return Err(n
                                .span
                                .error("array element count must be greater than zero"));
                        }
                        None => {
                            return Err(n.span.error(
                                "array element count must be a constant integer expression ('override' counts are not supported)",
                            ));
                        }
                    };
                    let t = Ty::Array(Box::new(e), Some(len));
                    self.check_type_limits(&t, span)?;
                    t
                }
                _ => return Err(span.error("'array' expects array<T> or array<T, N>")),
            },
            "texture_2d" | "texture_2d_array" | "texture_cube" => {
                let view = match name {
                    "texture_2d" => TextureView::D2,
                    "texture_2d_array" => TextureView::D2Array,
                    _ => TextureView::Cube,
                };
                let sample = match tmpl {
                    [t] => match self.template_scalar(t, "the texture sample type")? {
                        Scalar::F32 => SampleType::Float,
                        Scalar::I32 => SampleType::Sint,
                        Scalar::U32 => SampleType::Uint,
                        _ => {
                            return Err(t
                                .span
                                .error("texture sample type must be f32, i32 or u32"));
                        }
                    },
                    _ => {
                        return Err(span
                            .error(format!("'{name}' requires a sample type, e.g. {name}<f32>")));
                    }
                };
                Ty::Texture(view, sample)
            }
            "texture_depth_2d" => return plain(Ty::Texture(TextureView::D2, SampleType::Depth)),
            "texture_depth_2d_array" => {
                return plain(Ty::Texture(TextureView::D2Array, SampleType::Depth));
            }
            "texture_depth_cube" => {
                return plain(Ty::Texture(TextureView::Cube, SampleType::Depth));
            }
            "texture_storage_2d" => match tmpl {
                [f, a] => {
                    let fname = match &f.kind {
                        AK::Ident {
                            name,
                            templates: None,
                        } => name.as_str(),
                        _ => "",
                    };
                    let format = texel_format(fname).ok_or_else(|| {
                        f.span.error(format!(
                            "unsupported storage texture format '{fname}' (Ganbate supports rgba8unorm, rgba16float, rgba32float, r32float, rg16float, r16float)"
                        ))
                    })?;
                    match &a.kind {
                        AK::Ident {
                            name,
                            templates: None,
                        } if name == "write" => {}
                        AK::Ident {
                            name,
                            templates: None,
                        } if name == "read" || name == "read_write" => {
                            return Err(a.span.error(
                                "only write-only storage textures are supported by Ganbate",
                            ));
                        }
                        _ => {
                            return Err(a
                                .span
                                .error("expected a storage texture access mode ('write')"));
                        }
                    }
                    Ty::StorageTexture(format)
                }
                _ => {
                    return Err(span
                        .error("'texture_storage_2d' expects texture_storage_2d<format, write>"));
                }
            },
            "sampler" => return plain(Ty::Sampler(false)),
            "sampler_comparison" => return plain(Ty::Sampler(true)),
            "ptr" => return Err(span.error("pointer types are not supported by Ganbate")),
            "atomic" => return Err(span.error("atomic types are not supported by Ganbate")),
            "texture_1d"
            | "texture_3d"
            | "texture_cube_array"
            | "texture_multisampled_2d"
            | "texture_external"
            | "texture_depth_multisampled_2d"
            | "texture_depth_cube_array"
            | "texture_storage_1d"
            | "texture_storage_2d_array"
            | "texture_storage_3d"
            | "binding_array" => {
                return Err(span.error(format!("'{name}' is not supported by Ganbate")));
            }
            _ => return Ok(None),
        }))
    }

    fn check_type_limits(&self, t: &Ty, span: Span) -> R<()> {
        if type_depth(t, &self.structs) > MAX_TYPE_DEPTH {
            return Err(span.error(format!(
                "type nests arrays and structs more than {MAX_TYPE_DEPTH} levels deep"
            )));
        }
        if let Ty::Array(e, Some(n)) = t
            && is_host_shareable(e, &self.structs)
            && (*n as u64) * (array_stride(e, &self.structs) as u64) >= MAX_TYPE_SIZE
        {
            return Err(span.error(format!("{} is too large", self.ty_name(t))));
        }
        Ok(())
    }

    fn array_elem(&mut self, t: &ast::Expr) -> R<Ty> {
        let e = self.resolve_type(t)?;
        if e.is_handle() {
            return Err(t
                .span
                .error("arrays of textures or samplers are not supported by Ganbate"));
        }
        if contains_runtime_array(&e, &self.structs) {
            return Err(t
                .span
                .error("array elements cannot contain runtime-sized arrays"));
        }
        Ok(e)
    }

    fn global_var(&mut self, v: &ast::GlobalVarDecl) -> R<GlobalId> {
        let (mut group, mut binding) = (None, None);
        for a in &v.attrs {
            match a.name.as_str() {
                "group" => group = Some(self.attr_int(a)?),
                "binding" => binding = Some(self.attr_int(a)?),
                other => {
                    return Err(a
                        .span
                        .error(format!("attribute @{other} is not valid on a variable")));
                }
            }
        }
        let (space, access) = match &v.space {
            None => (None, None),
            Some(args) => {
                let id = |e: &ast::Expr| match &e.kind {
                    AK::Ident {
                        name,
                        templates: None,
                    } => Ok(name.clone()),
                    _ => Err(e.span.error("expected an address space name")),
                };
                if args.is_empty() || args.len() > 2 {
                    return Err(v.span.error("expected var<space> or var<storage, access>"));
                }
                (Some(id(&args[0])?), args.get(1).map(id).transpose()?)
            }
        };
        let declared = v.ty.as_ref().map(|t| self.resolve_type(t)).transpose()?;
        let span = v.name.span;
        let mut init = None;
        let (kind, ty) = match space.as_deref() {
            None => {
                let Some(t) = declared else {
                    return Err(span
                        .error("a module-scope 'var' needs an address space, e.g. var<private>"));
                };
                if !t.is_handle() {
                    return Err(span.error(format!(
                        "module-scope variable '{}' needs an address space (var<private>, var<uniform>, var<storage> or var<push_constant>)",
                        v.name.name
                    )));
                }
                (GlobalKind::Handle, t)
            }
            Some("private") => {
                let iv = match &v.init {
                    Some(e) => {
                        let x = self.value(e)?;
                        let x = match &declared {
                            Some(t) => self.convert(x, t, e.span)?,
                            None => self.concretize(x)?,
                        };
                        let EK::Const(c) = x.kind else {
                            return Err(e.span.error(
                                "a var<private> initializer must be a constant expression",
                            ));
                        };
                        Some((x.ty, c))
                    }
                    None => None,
                };
                let ty = match (declared, &iv) {
                    (Some(t), _) => t,
                    (None, Some((t, _))) => t.clone(),
                    (None, None) => {
                        return Err(span.error("a 'var' needs a type or an initializer"));
                    }
                };
                init = iv.map(|x| x.1);
                if !is_constructible(&ty, &self.structs) {
                    return Err(span.error(format!(
                        "var<private> cannot have type {}",
                        self.ty_name(&ty)
                    )));
                }
                (GlobalKind::Private, ty)
            }
            Some(s @ ("uniform" | "storage" | "push_constant")) => {
                let Some(ty) = declared else {
                    return Err(span.error(format!("a var<{s}> needs an explicit type")));
                };
                let kind = match s {
                    "uniform" => GlobalKind::Uniform,
                    "push_constant" => GlobalKind::PushConstant,
                    _ => {
                        GlobalKind::Storage {
                            read_only: match access.as_deref() {
                                None | Some("read") => true,
                                Some("read_write") => false,
                                Some(other) => {
                                    return Err(span.error(format!("invalid storage access mode '{other}' (use read or read_write)")));
                                }
                            },
                        }
                    }
                };
                if access.is_some() && s != "storage" {
                    return Err(span.error(format!("var<{s}> does not take an access mode")));
                }
                self.check_buffer_type(&ty, kind, span)?;
                (kind, ty)
            }
            Some("workgroup") => {
                return Err(span.error("var<workgroup> is not supported by Ganbate"));
            }
            Some("function") => {
                return Err(span.error("var<function> is only allowed inside functions"));
            }
            Some(other) => return Err(span.error(format!("unknown address space '{other}'"))),
        };
        if v.init.is_some() && kind != GlobalKind::Private {
            return Err(
                span.error("only var<private> variables can have an initializer at module scope")
            );
        }
        let needs_binding = matches!(
            kind,
            GlobalKind::Handle | GlobalKind::Uniform | GlobalKind::Storage { .. }
        );
        let binding = match (group, binding) {
            (Some(g), Some(b)) if needs_binding => Some((g, b)),
            (None, None) if !needs_binding => None,
            _ if needs_binding => {
                return Err(span.error(format!(
                    "resource '{}' needs both @group and @binding",
                    v.name.name
                )));
            }
            _ => {
                return Err(span.error(format!(
                    "'{}' is not a resource and cannot have @group/@binding",
                    v.name.name
                )));
            }
        };
        self.globals.push(ir::Global {
            name: v.name.name.clone(),
            span,
            kind,
            ty,
            binding,
            init,
        });
        Ok(self.globals.len() - 1)
    }

    fn check_buffer_type(&self, ty: &Ty, kind: GlobalKind, span: Span) -> R<()> {
        let name = self.ty_name(ty);
        if !is_host_shareable(ty, &self.structs) {
            return Err(span.error(format!(
                "type {name} cannot be used in a buffer: it is not host-shareable (bool and textures are not allowed)"
            )));
        }
        match kind {
            GlobalKind::Storage { .. } => Ok(()),
            _ => {
                if contains_runtime_array(ty, &self.structs) {
                    return Err(
                        span.error("runtime-sized arrays are only allowed in storage buffers")
                    );
                }
                if kind == GlobalKind::Uniform {
                    self.check_uniform_layout(ty, span)?;
                }
                Ok(())
            }
        }
    }

    /// WGSL's extra layout constraints for the uniform address space (which also make the
    /// layout valid std140 for Vulkan).
    fn check_uniform_layout(&self, ty: &Ty, span: Span) -> R<()> {
        match ty {
            Ty::Array(e, _) => {
                let stride = array_stride(e, &self.structs);
                if stride % 16 != 0 {
                    return Err(span.error(format!(
                        "uniform buffer arrays need a 16-byte element stride, but {} has stride {stride} (use vec4 elements, or wrap the element in a struct with @size(16))",
                        self.ty_name(ty)
                    )));
                }
                self.check_uniform_layout(e, span)
            }
            Ty::Matrix(c, 2) => Err(span.error(format!(
                "mat{c}x2<f32> is not supported in uniform buffers: its 8-byte column stride cannot be expressed in Vulkan 1.0 std140 layout (use vec4 columns or a storage buffer)"
            ))),
            Ty::Struct(id) => {
                let s = &self.structs[*id];
                let layout = s.layout.as_ref().expect("host-shareable struct");
                for (i, m) in s.members.iter().enumerate() {
                    let off = layout.offsets[i];
                    if matches!(m.ty, Ty::Struct(_) | Ty::Array(..)) && off % 16 != 0 {
                        return Err(m.span.error(format!(
                            "in the uniform address space, member '{}' of '{}' must be 16-byte aligned (it is at offset {off}); add @align(16)",
                            m.name, s.name
                        )));
                    }
                    if let Ty::Struct(_) = m.ty
                        && let Some(&next) = layout.offsets.get(i + 1)
                    {
                        let need = round_up(16, size_of(&m.ty, &self.structs));
                        if next - off < need {
                            let nm = &s.members[i + 1];
                            return Err(nm.span.error(format!(
                                "in the uniform address space, member '{}' of '{}' must start at least {need} bytes after struct member '{}' (add @align(16))",
                                nm.name, s.name, m.name
                            )));
                        }
                    }
                    self.check_uniform_layout(&m.ty, m.span)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn fn_signature(&mut self, f: &'a ast::FnDecl) -> R<FnId> {
        let mut stage = None;
        let mut stage_span = f.name.span;
        let mut wg = None;
        for a in &f.attrs {
            match a.name.as_str() {
                s @ ("vertex" | "fragment" | "compute") => {
                    if !a.args.is_empty() {
                        return Err(a.span.error(format!("@{s} takes no arguments")));
                    }
                    if stage.is_some() {
                        return Err(a.span.error("a function can have only one stage attribute"));
                    }
                    stage = Some(match s {
                        "vertex" => Stage::Vertex,
                        "fragment" => Stage::Fragment,
                        _ => Stage::Compute,
                    });
                    stage_span = a.span;
                }
                "workgroup_size" => {
                    if a.args.is_empty() || a.args.len() > 3 {
                        return Err(a.span.error("@workgroup_size expects 1 to 3 arguments"));
                    }
                    let mut size = [1u32; 3];
                    for (i, e) in a.args.iter().enumerate() {
                        let v = self.value(e)?;
                        size[i] = match v.as_const().and_then(|c| c.as_i64()) {
                            Some(n) if n > 0 && n <= u32::MAX as i64 => n as u32,
                            _ => {
                                return Err(e
                                    .span
                                    .error("workgroup size must be a positive integer constant"));
                            }
                        };
                    }
                    wg = Some(size);
                }
                "must_use" | "diagnostic" => {}
                other => {
                    return Err(a
                        .span
                        .error(format!("attribute @{other} is not valid on a function")));
                }
            }
        }
        let mut params: Vec<ir::Param> = Vec::new();
        for (i, p) in f.params.iter().enumerate() {
            if params.iter().any(|q| q.name == p.name.name) {
                return Err(p
                    .name
                    .span
                    .error(format!("duplicate parameter '{}'", p.name.name)));
            }
            let ty = self.resolve_type(&p.ty)?;
            if ty.is_handle() {
                return Err(p.ty.span.error(
                    "textures and samplers cannot be function parameters in Ganbate; use the global variable directly",
                ));
            }
            if !is_constructible(&ty, &self.structs) {
                return Err(p.ty.span.error(format!(
                    "parameter type {} is not constructible",
                    self.ty_name(&ty)
                )));
            }
            let (io, _, _) = self.io_attrs(&p.attrs, false)?;
            if stage.is_none() && !p.attrs.is_empty() {
                return Err(p.attrs[0]
                    .span
                    .error("IO attributes are only allowed on entry-point parameters"));
            }
            params.push(ir::Param {
                name: p.name.name.clone(),
                span: p.name.span,
                ty,
                io,
                value: i as ValueId,
            });
        }
        let (ret, ret_io) = match &f.ret {
            None => (None, IoAttrs::default()),
            Some((attrs, t)) => {
                let ty = self.resolve_type(t)?;
                if !is_constructible(&ty, &self.structs) {
                    return Err(t.span.error(format!(
                        "return type {} is not constructible",
                        self.ty_name(&ty)
                    )));
                }
                let (io, _, _) = self.io_attrs(attrs, false)?;
                if stage.is_none() && !attrs.is_empty() {
                    return Err(attrs[0]
                        .span
                        .error("IO attributes are only allowed on entry-point return values"));
                }
                (Some(ty), io)
            }
        };
        match (stage, wg) {
            (Some(Stage::Compute), None) => {
                return Err(stage_span.error("a @compute entry point needs @workgroup_size"));
            }
            (Some(Stage::Vertex | Stage::Fragment) | None, Some(_)) => {
                return Err(f
                    .name
                    .span
                    .error("@workgroup_size is only allowed on @compute entry points"));
            }
            _ => {}
        }
        let value_names = params.iter().map(|p| p.name.clone()).collect();
        self.functions.push(ir::Function {
            name: f.name.name.clone(),
            span: f.name.span,
            params,
            ret,
            ret_io,
            stage,
            stage_span,
            workgroup_size: wg.unwrap_or([1, 1, 1]),
            locals: Vec::new(),
            value_names,
            body: Vec::new(),
            calls: BTreeSet::new(),
            globals: BTreeSet::new(),
            stage_reqs: Vec::new(),
        });
        self.fn_decls.push(f);
        Ok(self.functions.len() - 1)
    }

    fn check_body(&mut self, fid: FnId) -> R<()> {
        let decl = self.fn_decls[fid];
        let f = &self.functions[fid];
        let mut scope = Vec::new();
        for p in &f.params {
            scope.push((
                p.name.clone(),
                Binding::Value {
                    id: p.value,
                    ty: p.ty.clone(),
                    after_continue_of: None,
                },
            ));
        }
        self.fcx = Some(FnCx {
            scopes: vec![scope],
            locals: Vec::new(),
            value_names: f.value_names.clone(),
            ret: f.ret.clone(),
            loops: Vec::new(),
            continuing_of: Vec::new(),
            calls: BTreeSet::new(),
            globals: BTreeSet::new(),
            stage_reqs: Vec::new(),
        });
        let mut body = Vec::new();
        let falls = self.check_stmts(&decl.body.stmts, &mut body);
        let fcx = self.fcx.take().expect("function context");
        let falls = falls?;
        let f = &mut self.functions[fid];
        if falls && f.ret.is_some() {
            return Err(decl.body.end.error(format!(
                "function '{}' must return a value on every path",
                f.name
            )));
        }
        f.body = body;
        f.locals = fcx.locals;
        f.value_names = fcx.value_names;
        f.calls = fcx.calls;
        f.globals = fcx.globals;
        f.stage_reqs = fcx.stage_reqs;
        Ok(())
    }

    fn check_recursion(&self) -> R<()> {
        // 0 = unvisited, 1 = on stack, 2 = done
        fn visit(fns: &[ir::Function], f: FnId, state: &mut [u8]) -> Result<(), FnId> {
            match state[f] {
                1 => return Err(f),
                2 => return Ok(()),
                _ => {}
            }
            state[f] = 1;
            for &c in &fns[f].calls {
                visit(fns, c, state)?;
            }
            state[f] = 2;
            Ok(())
        }
        let mut state = vec![0u8; self.functions.len()];
        for f in 0..self.functions.len() {
            if let Err(cyc) = visit(&self.functions, f, &mut state) {
                let func = &self.functions[cyc];
                return Err(func.span.error(format!(
                    "function '{}' is recursive; recursion is not allowed",
                    func.name
                )));
            }
        }
        Ok(())
    }

    fn validate_entry(&self, fid: FnId) -> R<()> {
        let f = &self.functions[fid];
        let stage = f.stage.expect("entry point");
        let mut in_locs: HashMap<u32, Span> = HashMap::new();
        let mut out_locs: HashMap<u32, Span> = HashMap::new();
        let mut out_builtins: Vec<BuiltinValue> = Vec::new();
        for p in &f.params {
            if !p.io.is_empty() {
                self.check_io(
                    stage,
                    false,
                    &p.ty,
                    &p.io,
                    p.span,
                    &mut in_locs,
                    &mut out_builtins,
                )?;
            } else if let Ty::Struct(sid) = p.ty {
                for m in &self.structs[sid].members {
                    if m.io.is_empty() {
                        return Err(m.span.error(format!(
                            "member '{}' of entry-point input struct '{}' needs @location or @builtin",
                            m.name, self.structs[sid].name
                        )));
                    }
                    self.check_io(
                        stage,
                        false,
                        &m.ty,
                        &m.io,
                        m.span,
                        &mut in_locs,
                        &mut out_builtins,
                    )?;
                }
            } else {
                return Err(p.span.error(format!(
                    "entry-point parameter '{}' needs @location or @builtin",
                    p.name
                )));
            }
        }
        out_builtins.clear();
        if let Some(ty) = &f.ret {
            if stage == Stage::Compute {
                return Err(f.span.error("a compute entry point cannot return a value"));
            }
            if !f.ret_io.is_empty() {
                self.check_io(
                    stage,
                    true,
                    ty,
                    &f.ret_io,
                    f.span,
                    &mut out_locs,
                    &mut out_builtins,
                )?;
            } else if let Ty::Struct(sid) = ty {
                for m in &self.structs[*sid].members {
                    if m.io.is_empty() {
                        return Err(m.span.error(format!(
                            "member '{}' of entry-point output struct '{}' needs @location or @builtin",
                            m.name, self.structs[*sid].name
                        )));
                    }
                    self.check_io(
                        stage,
                        true,
                        &m.ty,
                        &m.io,
                        m.span,
                        &mut out_locs,
                        &mut out_builtins,
                    )?;
                }
            } else {
                return Err(f
                    .span
                    .error("an entry-point return value needs @location or @builtin"));
            }
        }
        if stage == Stage::Vertex && !out_builtins.contains(&BuiltinValue::Position) {
            return Err(f.span.error(format!(
                "vertex entry point '{}' must return @builtin(position)",
                f.name
            )));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn check_io(
        &self,
        stage: Stage,
        output: bool,
        ty: &Ty,
        io: &IoAttrs,
        span: Span,
        locs: &mut HashMap<u32, Span>,
        builtins: &mut Vec<BuiltinValue>,
    ) -> R<()> {
        use BuiltinValue as B;
        let dir = if output { "output" } else { "input" };
        let span = if io.span == Span::default() {
            span
        } else {
            io.span
        };
        if let Some(b) = io.builtin {
            let ok = match (stage, output) {
                (Stage::Vertex, false) => matches!(b, B::VertexIndex | B::InstanceIndex),
                (Stage::Vertex, true) => b == B::Position,
                (Stage::Fragment, false) => {
                    matches!(b, B::Position | B::FrontFacing | B::SampleIndex)
                }
                (Stage::Fragment, true) => b == B::FragDepth,
                (Stage::Compute, false) => matches!(
                    b,
                    B::LocalInvocationId
                        | B::LocalInvocationIndex
                        | B::GlobalInvocationId
                        | B::WorkgroupId
                        | B::NumWorkgroups
                ),
                (Stage::Compute, true) => false,
            };
            if !ok {
                return Err(span.error(
                    format!("@builtin({}) is not a valid {stage:?} {dir}", b.name()).to_lowercase(),
                ));
            }
            if *ty != b.ty() {
                return Err(span.error(format!(
                    "@builtin({}) must have type {}, found {}",
                    b.name(),
                    self.ty_name(&b.ty()),
                    self.ty_name(ty)
                )));
            }
            if output && builtins.contains(&b) {
                return Err(span.error(format!(
                    "@builtin({}) appears more than once in the outputs",
                    b.name()
                )));
            }
            builtins.push(b);
            return Ok(());
        }
        let loc = io.location.expect("location");
        if stage == Stage::Compute {
            return Err(span.error("compute entry points cannot have @location inputs or outputs"));
        }
        if !matches!(
            ty,
            Ty::Scalar(Scalar::F32 | Scalar::I32 | Scalar::U32)
                | Ty::Vector(_, Scalar::F32 | Scalar::I32 | Scalar::U32)
        ) {
            return Err(span.error(format!(
                "@location({loc}) must be a scalar or vector of f32, i32 or u32, found {}",
                self.ty_name(ty)
            )));
        }
        if let Some(prev) = locs.insert(loc, span) {
            return Err(span.error(format!(
                "@location({loc}) is used twice as an {dir} (first at line {})",
                prev.line
            )));
        }
        Ok(())
    }

    // ------------------------------------------------------------------ statements

    fn push_scope(&mut self) {
        self.fcx_mut().scopes.push(Vec::new());
    }

    fn pop_scope(&mut self) {
        self.fcx_mut().scopes.pop();
    }

    fn declare(&mut self, name: &ast::Ident, b: Binding) -> R<()> {
        let scope = self.fcx_mut().scopes.last_mut().expect("scope");
        if scope.iter().any(|(n, _)| *n == name.name) {
            return Err(name
                .span
                .error(format!("'{}' is already declared in this scope", name.name)));
        }
        scope.push((name.name.clone(), b));
        Ok(())
    }

    fn lookup_local(&self, name: &str, span: Span) -> R<Option<Binding>> {
        let Some(fcx) = &self.fcx else {
            return Ok(None);
        };
        for scope in fcx.scopes.iter().rev() {
            for (n, b) in scope.iter().rev() {
                if n == name {
                    if let Binding::Value {
                        after_continue_of: Some(l),
                        ..
                    } = b
                        && fcx.continuing_of.contains(l)
                    {
                        return Err(span.error(format!(
                            "'{name}' cannot be used in this 'continuing' block: a 'continue' before its declaration can skip it"
                        )));
                    }
                    return Ok(Some(b.clone()));
                }
            }
        }
        Ok(None)
    }

    fn check_stmts(&mut self, stmts: &[ast::Stmt], out: &mut Vec<Stmt>) -> R<bool> {
        let mut falls = true;
        for s in stmts {
            let f = self.stmt(s, out)?;
            falls &= f;
        }
        Ok(falls)
    }

    fn scoped_block(&mut self, b: &ast::Block) -> R<(Vec<Stmt>, bool)> {
        self.push_scope();
        let mut out = Vec::new();
        let r = self.check_stmts(&b.stmts, &mut out);
        self.pop_scope();
        Ok((out, r?))
    }

    fn condition(&mut self, e: &ast::Expr, what: &str) -> R<Expr> {
        let v = self.value(e)?;
        if v.ty != BOOL {
            return Err(e.span.error(format!(
                "{what} condition must be bool, found {}",
                self.ty_name(&v.ty)
            )));
        }
        Ok(v)
    }

    pub(crate) fn zero(&self, ty: &Ty, span: Span) -> Expr {
        Expr {
            ty: ty.clone(),
            kind: EK::Zero,
            span,
        }
    }

    fn stmt(&mut self, s: &ast::Stmt, out: &mut Vec<Stmt>) -> R<bool> {
        let span = s.span;
        match &s.kind {
            SK::Empty => Ok(true),
            SK::Let { name, ty, init } => {
                let v = self.value(init)?;
                let v = match ty {
                    Some(t) => {
                        let t = self.resolve_type(t)?;
                        self.convert(v, &t, init.span)?
                    }
                    None => self.concretize(v)?,
                };
                if !is_constructible(&v.ty, &self.structs) {
                    return Err(init.span.error(format!(
                        "a 'let' cannot hold a value of type {}",
                        self.ty_name(&v.ty)
                    )));
                }
                let ty = v.ty.clone();
                let fcx = self.fcx_mut();
                let id = fcx.value_names.len() as ValueId;
                fcx.value_names.push(name.name.clone());
                let cur = fcx.scopes.len() - 1;
                let after_continue_of = fcx
                    .loops
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, l)| l.body_scope == cur)
                    .filter(|(_, l)| l.continue_seen)
                    .map(|(i, _)| i);
                out.push(Stmt::Let(id, v));
                self.declare(
                    name,
                    Binding::Value {
                        id,
                        ty,
                        after_continue_of,
                    },
                )?;
                Ok(true)
            }
            SK::Var {
                name,
                space,
                ty,
                init,
            } => {
                if let Some(sp) = space {
                    let ok = matches!(sp.as_slice(), [e] if matches!(&e.kind, AK::Ident { name, templates: None } if name == "function"));
                    if !ok {
                        return Err(span.error("only var<function> is allowed inside a function"));
                    }
                }
                let declared = ty.as_ref().map(|t| self.resolve_type(t)).transpose()?;
                let iv = init.as_ref().map(|e| self.value(e)).transpose()?;
                let (ty, value) = match (declared, iv) {
                    (Some(t), Some(v)) => {
                        let v = self.convert(v, &t, init.as_ref().map_or(span, |e| e.span))?;
                        (t, v)
                    }
                    (Some(t), None) => {
                        let z = self.zero(&t, span);
                        (t, z)
                    }
                    (None, Some(v)) => {
                        let v = self.concretize(v)?;
                        (v.ty.clone(), v)
                    }
                    (None, None) => {
                        return Err(name.span.error("a 'var' needs a type or an initializer"));
                    }
                };
                if !is_constructible(&ty, &self.structs) {
                    return Err(name
                        .span
                        .error(format!("a variable cannot have type {}", self.ty_name(&ty))));
                }
                let fcx = self.fcx_mut();
                let id = fcx.locals.len() as LocalId;
                fcx.locals.push(ir::Local {
                    name: name.name.clone(),
                    ty: ty.clone(),
                });
                out.push(Stmt::Assign {
                    place: Place {
                        root: PlaceRoot::Local(id),
                        steps: Vec::new(),
                        ty: ty.clone(),
                    },
                    swizzle: None,
                    value,
                });
                self.declare(name, Binding::Var { id, ty })?;
                Ok(true)
            }
            SK::Const { name, ty, init } => {
                let (ty, value) = self.const_decl(ty.as_ref(), init)?;
                self.declare(name, Binding::Const { ty, value })?;
                Ok(true)
            }
            SK::Assign { lhs, op, rhs } => {
                let t = self.assign_target(lhs)?;
                let r = self.value(rhs)?;
                let value = match op {
                    None => self.convert(r, &t.ty, rhs.span)?,
                    Some(op) => {
                        let cur = Expr {
                            ty: t.ty.clone(),
                            kind: EK::Current,
                            span: lhs.span,
                        };
                        let v = self.binary(*op, cur, r, span)?;
                        if v.ty != t.ty {
                            return Err(span.error(format!(
                                "'{}=' produces {}, which cannot be assigned to {}",
                                op.symbol(),
                                self.ty_name(&v.ty),
                                self.ty_name(&t.ty)
                            )));
                        }
                        v
                    }
                };
                out.push(Stmt::Assign {
                    place: t.place,
                    swizzle: t.swizzle,
                    value,
                });
                Ok(true)
            }
            SK::Increment(e) | SK::Decrement(e) => {
                let inc = matches!(s.kind, SK::Increment(_));
                let t = self.assign_target(e)?;
                let one = match t.ty {
                    Ty::Scalar(Scalar::I32) => ConstValue::I32(1),
                    Ty::Scalar(Scalar::U32) => ConstValue::U32(1),
                    _ => {
                        return Err(span.error(format!(
                            "'{}' requires an i32 or u32 variable, found {}",
                            if inc { "++" } else { "--" },
                            self.ty_name(&t.ty)
                        )));
                    }
                };
                let value = Expr {
                    ty: t.ty.clone(),
                    kind: EK::Binary(
                        if inc { BinaryOp::Add } else { BinaryOp::Sub },
                        Box::new(Expr {
                            ty: t.ty.clone(),
                            kind: EK::Current,
                            span,
                        }),
                        Box::new(konst(t.ty.clone(), one, span)),
                    ),
                    span,
                };
                out.push(Stmt::Assign {
                    place: t.place,
                    swizzle: t.swizzle,
                    value,
                });
                Ok(true)
            }
            SK::Phony(e) => {
                match self.expr(e)? {
                    Checked::Handle(..) => {}
                    Checked::Ref(p) if !is_constructible(&p.ty, &self.structs) => {}
                    other => {
                        let v = self.load(other, e.span)?;
                        out.push(Stmt::Eval(v));
                    }
                }
                Ok(true)
            }
            SK::Call(e) => {
                let AK::Call {
                    name,
                    templates,
                    args,
                } = &e.kind
                else {
                    return Err(e.span.error("expected a function call"));
                };
                let v = self.call(name, templates.as_deref(), args, e.span, true)?;
                out.push(Stmt::Eval(v));
                Ok(true)
            }
            SK::If {
                cond,
                then,
                otherwise,
            } => {
                let c = self.condition(cond, "'if'")?;
                if let Some(&ConstValue::Bool(taken)) = c.as_const() {
                    if taken {
                        let (t, tf) = self.scoped_block(then)?;
                        if let Some(s) = otherwise {
                            self.check_dead_branch(|me| match &s.kind {
                                SK::Block(b) => me.scoped_block(b).map(|_| ()),
                                _ => {
                                    let mut sink = Vec::new();
                                    me.stmt(s, &mut sink).map(|_| ())
                                }
                            })?;
                        }
                        if !t.is_empty() {
                            out.push(Stmt::Block(t));
                        }
                        return Ok(tf);
                    }
                    self.check_dead_branch(|me| me.scoped_block(then).map(|_| ()))?;
                    let (o, of) = match otherwise {
                        None => (Vec::new(), true),
                        Some(s) => match &s.kind {
                            SK::Block(b) => self.scoped_block(b)?,
                            _ => {
                                let mut o = Vec::new();
                                let f = self.stmt(s, &mut o)?;
                                (o, f)
                            }
                        },
                    };
                    if !o.is_empty() {
                        out.push(Stmt::Block(o));
                    }
                    return Ok(of);
                }
                let (t, tf) = self.scoped_block(then)?;
                let (o, of) = match otherwise {
                    None => (Vec::new(), true),
                    Some(s) => match &s.kind {
                        SK::Block(b) => self.scoped_block(b)?,
                        _ => {
                            let mut o = Vec::new();
                            let f = self.stmt(s, &mut o)?;
                            (o, f)
                        }
                    },
                };
                out.push(Stmt::If {
                    cond: c,
                    then: t,
                    otherwise: o,
                });
                Ok(tf || of)
            }
            SK::Loop { body, continuing } => {
                self.loop_stmt(None, body, LoopTail::Continuing(continuing.as_ref()), out)
            }
            SK::While { cond, body } => {
                let c = self.condition(cond, "'while'")?;
                self.loop_stmt(Some(c), body, LoopTail::None, out)
            }
            SK::For {
                init,
                cond,
                update,
                body,
            } => {
                self.push_scope();
                let mut inner = Vec::new();
                let r = (|| {
                    if let Some(i) = init {
                        self.stmt(i, &mut inner)?;
                    }
                    let c = cond
                        .as_ref()
                        .map(|c| self.condition(c, "'for'"))
                        .transpose()?;
                    self.loop_stmt(c, body, LoopTail::Update(update.as_deref()), &mut inner)
                })();
                self.pop_scope();
                let falls = r?;
                out.push(Stmt::Block(inner));
                Ok(falls)
            }
            SK::Break => {
                let fcx = self.fcx_mut();
                match fcx.loops.last_mut() {
                    None => return Err(span.error("'break' outside of a loop")),
                    Some(l) if l.in_continuing => {
                        return Err(span.error(
                            "'break' is not allowed in a continuing block; use 'break if'",
                        ));
                    }
                    Some(l) => l.has_break = true,
                }
                out.push(Stmt::Break);
                Ok(false)
            }
            SK::Continue => {
                let fcx = self.fcx_mut();
                match fcx.loops.last_mut() {
                    None => return Err(span.error("'continue' outside of a loop")),
                    Some(l) if l.in_continuing => {
                        return Err(span.error("'continue' is not allowed in a continuing block"));
                    }
                    Some(l) => l.continue_seen = true,
                }
                out.push(Stmt::Continue);
                Ok(false)
            }
            SK::Discard => {
                self.fcx_mut()
                    .stage_reqs
                    .push((StageReq::Fragment, span, "discard".into()));
                out.push(Stmt::Discard);
                Ok(false)
            }
            SK::Return(v) => {
                if !self.fcx_mut().continuing_of.is_empty() {
                    return Err(span.error("'return' is not allowed in a continuing block"));
                }
                let ret = self.fcx_mut().ret.clone();
                match (v, ret) {
                    (None, None) => out.push(Stmt::Return(None)),
                    (Some(e), Some(t)) => {
                        let x = self.value(e)?;
                        let x = self.convert(x, &t, e.span)?;
                        out.push(Stmt::Return(Some(x)));
                    }
                    (None, Some(t)) => {
                        return Err(span
                            .error(format!("missing return value of type {}", self.ty_name(&t))));
                    }
                    (Some(e), None) => {
                        return Err(e
                            .span
                            .error("this function has no return type and cannot return a value"));
                    }
                }
                Ok(false)
            }
            SK::Block(b) => {
                let (st, f) = self.scoped_block(b)?;
                out.push(Stmt::Block(st));
                Ok(f)
            }
            SK::ConstAssert(e) => {
                if self.dead_depth == 0 {
                    self.const_assert(e)?;
                } else {
                    let v = self.value(e)?;
                    if !matches!(v.as_const(), Some(ConstValue::Bool(_))) {
                        return Err(e
                            .span
                            .error("const_assert requires a constant bool expression"));
                    }
                }
                Ok(true)
            }
        }
    }

    /// Type-checks an unreachable constant `if`/`else` branch so type and syntax errors are
    /// still reported, then rolls back side effects on function calls, global bindings, locals,
    /// and stage requirements so dead code is completely pruned from IR and reflection.
    fn check_dead_branch(&mut self, f: impl FnOnce(&mut Self) -> R<()>) -> R<()> {
        let const_data_len = self.const_data.len();
        let globals_len = self.globals.len();
        let snap = {
            let fcx = self.fcx_mut();
            (
                fcx.locals.len(),
                fcx.value_names.len(),
                fcx.calls.clone(),
                fcx.globals.clone(),
                fcx.stage_reqs.len(),
                fcx.loops.clone(),
            )
        };
        self.dead_depth += 1;
        let res = f(self);
        self.dead_depth -= 1;
        self.const_data.truncate(const_data_len);
        self.globals.truncate(globals_len);
        let fcx = self.fcx_mut();
        fcx.locals.truncate(snap.0);
        fcx.value_names.truncate(snap.1);
        fcx.calls = snap.2;
        fcx.globals = snap.3;
        fcx.stage_reqs.truncate(snap.4);
        fcx.loops = snap.5;
        res
    }

    fn loop_stmt(
        &mut self,
        cond: Option<Expr>,
        body: &ast::Block,
        tail: LoopTail,
        out: &mut Vec<Stmt>,
    ) -> R<bool> {
        let has_cond = cond.is_some();
        self.push_scope();
        let body_scope = self.fcx_mut().scopes.len() - 1;
        self.fcx_mut().loops.push(LoopCx {
            body_scope,
            continue_seen: false,
            has_break: false,
            in_continuing: false,
        });
        let r = (|| -> R<(Vec<Stmt>, Vec<Stmt>, Option<Expr>)> {
            let mut b = Vec::new();
            self.check_stmts(&body.stmts, &mut b)?;
            let fcx = self.fcx_mut();
            let li = fcx.loops.len() - 1;
            fcx.loops[li].in_continuing = true;
            fcx.continuing_of.push(li);
            let mut cont = Vec::new();
            let mut break_if = None;
            let r = (|| -> R<()> {
                match tail {
                    LoopTail::None => {}
                    LoopTail::Update(u) => {
                        if let Some(u) = u {
                            // The update cannot see declarations of the body.
                            let saved = self.fcx_mut().scopes.pop();
                            let r = self.stmt(u, &mut cont);
                            self.fcx_mut().scopes.extend(saved);
                            r?;
                        }
                    }
                    LoopTail::Continuing(c) => {
                        if let Some(c) = c {
                            self.push_scope();
                            let r = (|| -> R<()> {
                                self.check_stmts(&c.body.stmts, &mut cont)?;
                                if let Some(bi) = &c.break_if {
                                    break_if = Some(self.condition(bi, "'break if'")?);
                                }
                                Ok(())
                            })();
                            self.pop_scope();
                            r?;
                        }
                    }
                }
                Ok(())
            })();
            self.fcx_mut().continuing_of.pop();
            r?;
            Ok((b, cont, break_if))
        })();
        let l = self.fcx_mut().loops.pop().expect("loop");
        self.pop_scope();
        let (b, cont, break_if) = r?;
        let falls = has_cond || l.has_break || break_if.is_some();
        out.push(Stmt::Loop {
            cond,
            body: b,
            continuing: cont,
            break_if,
        });
        Ok(falls)
    }

    fn assign_target(&mut self, lhs: &ast::Expr) -> R<Target> {
        let (place, swizzle) = match self.expr(lhs)? {
            Checked::Ref(p) => (p, None),
            Checked::Swizzle(p, comps) => {
                let mut seen = [false; 4];
                for &c in &comps {
                    if std::mem::replace(&mut seen[c as usize], true) {
                        return Err(lhs
                            .span
                            .error("a swizzle assignment target must not repeat components"));
                    }
                }
                (p, Some(comps))
            }
            Checked::Value(v) => {
                let mut root = lhs;
                while let AK::Member(b, _) | AK::Index(b, _) = &root.kind {
                    root = b;
                }
                let msg = match (&root.kind, &v.kind) {
                    (AK::Ident { name, .. }, _) => match self.lookup_local(name, root.span)? {
                        Some(Binding::Value { .. }) => format!(
                            "cannot assign to '{name}': 'let' declarations and parameters are immutable (use 'var')"
                        ),
                        Some(Binding::Const { .. }) => {
                            format!("cannot assign to constant '{name}'")
                        }
                        _ if matches!(v.kind, EK::Const(_)) => {
                            format!("cannot assign to constant '{name}'")
                        }
                        _ => "cannot assign to this expression".to_string(),
                    },
                    (_, EK::Const(_)) => "cannot assign to a constant".to_string(),
                    _ => "cannot assign to this expression".to_string(),
                };
                return Err(lhs.span.error(msg));
            }
            Checked::Handle(..) => {
                return Err(lhs.span.error("cannot assign to a texture or sampler"));
            }
        };
        if let PlaceRoot::Global(g) = place.root {
            let g = &self.globals[g];
            let what = match g.kind {
                GlobalKind::Uniform => Some("uniform buffer"),
                GlobalKind::Storage { read_only: true } => {
                    Some("read-only storage buffer (declare it var<storage, read_write>)")
                }
                GlobalKind::PushConstant => Some("push constant block"),
                GlobalKind::ConstData => Some("constant"),
                _ => None,
            };
            if let Some(what) = what {
                return Err(lhs
                    .span
                    .error(format!("cannot assign to '{}': it is a {what}", g.name)));
            }
        }
        if !is_constructible(&place.ty, &self.structs) {
            return Err(lhs.span.error(format!(
                "cannot assign a value of type {}",
                self.ty_name(&place.ty)
            )));
        }
        let ty = match &swizzle {
            Some(c) => Ty::Vector(c.len() as u8, place.ty.scalar().expect("vector")),
            None => place.ty.clone(),
        };
        Ok(Target { place, swizzle, ty })
    }

    // ------------------------------------------------------------------ expressions

    pub(crate) fn value(&mut self, e: &ast::Expr) -> R<Expr> {
        let c = self.expr(e)?;
        self.load(c, e.span)
    }

    pub(crate) fn load(&mut self, c: Checked, span: Span) -> R<Expr> {
        match c {
            Checked::Value(v) => {
                if v.ty == Ty::Void {
                    return Err(span.error("this call does not return a value"));
                }
                Ok(v)
            }
            Checked::Ref(place) => {
                if !is_constructible(&place.ty, &self.structs) {
                    return Err(span.error(format!(
                        "cannot load a value of type {}; index it instead",
                        self.ty_name(&place.ty)
                    )));
                }
                Ok(Expr {
                    ty: place.ty.clone(),
                    kind: EK::Load(Box::new(place)),
                    span,
                })
            }
            Checked::Swizzle(place, comps) => {
                let s = place.ty.scalar().expect("vector");
                let base = Expr {
                    ty: place.ty.clone(),
                    kind: EK::Load(Box::new(place)),
                    span,
                };
                Ok(Expr {
                    ty: Ty::Vector(comps.len() as u8, s),
                    kind: EK::Swizzle(Box::new(base), comps),
                    span,
                })
            }
            Checked::Handle(_, ty) => Err(span.error(format!(
                "a {} can only be used as an argument of a texture built-in function",
                self.ty_name(&ty)
            ))),
        }
    }

    pub(crate) fn expr(&mut self, e: &ast::Expr) -> R<Checked> {
        let span = e.span;
        Ok(match &e.kind {
            AK::Int(lit) => Checked::Value(match *lit {
                IntLit::Abstract(v) => konst(
                    Ty::Scalar(Scalar::AbstractInt),
                    ConstValue::AbstractInt(v),
                    span,
                ),
                IntLit::I32(v) => konst(I32, ConstValue::I32(v), span),
                IntLit::U32(v) => konst(U32, ConstValue::U32(v), span),
            }),
            AK::Float(lit) => Checked::Value(match *lit {
                FloatLit::Abstract(v) => konst(
                    Ty::Scalar(Scalar::AbstractFloat),
                    ConstValue::AbstractFloat(v),
                    span,
                ),
                FloatLit::F32(v) => konst(F32, ConstValue::F32(v), span),
            }),
            AK::Bool(b) => Checked::Value(konst(BOOL, ConstValue::Bool(*b), span)),
            AK::Ident { name, templates } => self.ident(name, templates.is_some(), span)?,
            AK::Unary(op, x) => Checked::Value(self.unary(*op, x, span)?),
            AK::Binary(op, l, r) => {
                let l = self.value(l)?;
                let r = self.value(r)?;
                Checked::Value(self.binary(*op, l, r, span)?)
            }
            AK::Call {
                name,
                templates,
                args,
            } => Checked::Value(self.call(name, templates.as_deref(), args, span, false)?),
            AK::Member(base, m) => self.member(base, m)?,
            AK::Index(base, idx) => self.index(base, idx, span)?,
        })
    }

    fn ident(&mut self, name: &str, templated: bool, span: Span) -> R<Checked> {
        if templated {
            return Err(span.error(format!(
                "'{name}<...>' is a type; call it to construct a value"
            )));
        }
        if let Some(b) = self.lookup_local(name, span)? {
            return Ok(match b {
                Binding::Value { id, ty, .. } => Checked::Value(Expr {
                    ty,
                    kind: EK::Value(id),
                    span,
                }),
                Binding::Var { id, ty } => Checked::Ref(Place {
                    root: PlaceRoot::Local(id),
                    steps: Vec::new(),
                    ty,
                }),
                Binding::Const { ty, value } => Checked::Value(konst(ty, value, span)),
            });
        }
        match self.module_item(name)? {
            Some(Resolved::Const(ty, v)) => Ok(Checked::Value(konst(ty, v, span))),
            Some(Resolved::Global(gid)) => {
                let Some(fcx) = self.fcx.as_mut() else {
                    return Err(span.error(format!(
                        "'{name}' is a variable and cannot be used in a constant expression"
                    )));
                };
                fcx.globals.insert(gid);
                let g = &self.globals[gid];
                Ok(if g.kind == GlobalKind::Handle {
                    Checked::Handle(gid, g.ty.clone())
                } else {
                    Checked::Ref(Place {
                        root: PlaceRoot::Global(gid),
                        steps: Vec::new(),
                        ty: g.ty.clone(),
                    })
                })
            }
            Some(Resolved::Fn(_)) => Err(span.error(format!("function '{name}' must be called"))),
            Some(Resolved::Struct(_) | Resolved::Alias(_)) => Err(span.error(format!(
                "type '{name}' cannot be used as a value; call it to construct one"
            ))),
            Some(_) => Err(span.error(format!("'{name}' cannot be used as a value"))),
            None => {
                if matches!(self.predeclared_type(name, None, span), Ok(Some(_)))
                    || matches!(name, "vec2" | "vec3" | "vec4" | "array")
                {
                    Err(span.error(format!(
                        "type '{name}' cannot be used as a value; call it to construct one"
                    )))
                } else {
                    Err(span.error(format!("unknown identifier '{name}'")))
                }
            }
        }
    }

    fn extract(v: Expr, i: u32, ty: Ty, span: Span) -> Expr {
        if let EK::Const(c) = &v.kind {
            return konst(ty, c.components()[i as usize].clone(), span);
        }
        Expr {
            ty,
            kind: EK::Extract(Box::new(v), i),
            span,
        }
    }

    fn member(&mut self, base: &ast::Expr, m: &ast::Ident) -> R<Checked> {
        let b = match self.expr(base)? {
            c @ (Checked::Swizzle(..) | Checked::Handle(..)) => {
                Checked::Value(self.load(c, base.span)?)
            }
            other => other,
        };
        let ty = match &b {
            Checked::Ref(p) => p.ty.clone(),
            Checked::Value(v) => v.ty.clone(),
            _ => unreachable!(),
        };
        match &ty {
            Ty::Struct(sid) => {
                let s = &self.structs[*sid];
                let Some(i) = s.members.iter().position(|x| x.name == m.name) else {
                    return Err(m
                        .span
                        .error(format!("struct '{}' has no member '{}'", s.name, m.name)));
                };
                let mty = s.members[i].ty.clone();
                Ok(match b {
                    Checked::Ref(mut p) => {
                        p.steps.push(Step::Member(i as u32));
                        p.ty = mty;
                        Checked::Ref(p)
                    }
                    Checked::Value(v) => Checked::Value(Self::extract(v, i as u32, mty, m.span)),
                    _ => unreachable!(),
                })
            }
            Ty::Vector(n, s) => {
                let comps = parse_swizzle(&m.name, *n).ok_or_else(|| {
                    m.span.error(format!(
                        "invalid swizzle '.{}' for {}",
                        m.name,
                        self.ty_name(&ty)
                    ))
                })?;
                if comps.len() == 1 {
                    Ok(match b {
                        Checked::Ref(mut p) => {
                            p.steps.push(Step::Index(konst(
                                U32,
                                ConstValue::U32(comps[0]),
                                m.span,
                            )));
                            p.ty = Ty::Scalar(*s);
                            Checked::Ref(p)
                        }
                        Checked::Value(v) => {
                            Checked::Value(Self::extract(v, comps[0], Ty::Scalar(*s), m.span))
                        }
                        _ => unreachable!(),
                    })
                } else {
                    Ok(match b {
                        Checked::Ref(p) => Checked::Swizzle(p, comps),
                        Checked::Value(v) => {
                            let rty = Ty::Vector(comps.len() as u8, *s);
                            if let EK::Const(c) = &v.kind {
                                let cs = c.components();
                                let picked =
                                    comps.iter().map(|&i| cs[i as usize].clone()).collect();
                                Checked::Value(konst(rty, ConstValue::Composite(picked), m.span))
                            } else {
                                Checked::Value(Expr {
                                    ty: rty,
                                    kind: EK::Swizzle(Box::new(v), comps),
                                    span: m.span,
                                })
                            }
                        }
                        _ => unreachable!(),
                    })
                }
            }
            _ => Err(m.span.error(format!(
                "cannot access member '{}' of a value of type {}",
                m.name,
                self.ty_name(&ty)
            ))),
        }
    }

    fn index(&mut self, base: &ast::Expr, idx: &ast::Expr, span: Span) -> R<Checked> {
        let b = match self.expr(base)? {
            c @ (Checked::Swizzle(..) | Checked::Handle(..)) => {
                Checked::Value(self.load(c, base.span)?)
            }
            other => other,
        };
        let i = self.value(idx)?;
        let i = match i.ty {
            Ty::Scalar(Scalar::I32 | Scalar::U32) => i,
            Ty::Scalar(Scalar::AbstractInt) => {
                let v = i.as_const().and_then(|c| c.as_i64()).unwrap_or(-1);
                if !(0..=u32::MAX as i64).contains(&v) {
                    return Err(idx.span.error(format!("index {v} is out of bounds")));
                }
                konst(U32, ConstValue::U32(v as u32), idx.span)
            }
            _ => {
                return Err(idx.span.error(format!(
                    "an index must be i32 or u32, found {}",
                    self.ty_name(&i.ty)
                )));
            }
        };
        let const_idx = i.as_const().and_then(|c| c.as_i64());
        let ty = match &b {
            Checked::Ref(p) => p.ty.clone(),
            Checked::Value(v) => v.ty.clone(),
            _ => unreachable!(),
        };
        let (elem, len) = match &ty {
            Ty::Array(e, n) => ((**e).clone(), *n),
            Ty::Vector(n, s) => (Ty::Scalar(*s), Some(*n as u32)),
            Ty::Matrix(c, r) => (Ty::Vector(*r, Scalar::F32), Some(*c as u32)),
            _ => {
                return Err(span.error(format!(
                    "cannot index a value of type {}",
                    self.ty_name(&ty)
                )));
            }
        };
        if let Some(ci) = const_idx
            && (ci < 0 || len.is_some_and(|n| ci >= n as i64))
        {
            return Err(idx.span.error(format!(
                "index {ci} is out of bounds for {}",
                self.ty_name(&ty)
            )));
        }
        Ok(match b {
            Checked::Ref(mut p) => {
                p.steps.push(Step::Index(i));
                p.ty = elem;
                Checked::Ref(p)
            }
            Checked::Value(v) => {
                if let Some(ci) = const_idx {
                    Checked::Value(Self::extract(v, ci as u32, elem, span))
                } else if v.as_const().is_some() && matches!(v.ty, Ty::Array(..) | Ty::Matrix(..)) {
                    let v = self.concretize(v)?;
                    let EK::Const(c) = v.kind else { unreachable!() };
                    let name = match &base.kind {
                        AK::Ident { name, .. } => name.clone(),
                        _ => "const_data".to_string(),
                    };
                    let gid = self.const_data_global(v.ty.clone(), c, name, base.span);
                    let fcx = self.fcx.as_mut().ok_or_else(|| {
                        span.error("a constant expression cannot index with a non-constant index")
                    })?;
                    fcx.globals.insert(gid);
                    Checked::Ref(Place {
                        root: PlaceRoot::Global(gid),
                        steps: vec![Step::Index(i)],
                        ty: elem.concretize(),
                    })
                } else {
                    let v = self.concretize(v)?;
                    let elem = elem.concretize();
                    Checked::Value(Expr {
                        ty: elem,
                        kind: EK::DynIndex(Box::new(v), Box::new(i)),
                        span,
                    })
                }
            }
            _ => unreachable!(),
        })
    }

    fn const_data_global(
        &mut self,
        ty: Ty,
        value: ConstValue,
        name: String,
        span: Span,
    ) -> GlobalId {
        if let Some((_, _, g)) = self
            .const_data
            .iter()
            .find(|(t, v, _)| *t == ty && *v == value)
        {
            return *g;
        }
        self.globals.push(ir::Global {
            name,
            span,
            kind: GlobalKind::ConstData,
            ty: ty.clone(),
            binding: None,
            init: Some(value.clone()),
        });
        let g = self.globals.len() - 1;
        self.const_data.push((ty, value, g));
        g
    }

    fn unary(&mut self, op: ast::UnaryOp, x: &ast::Expr, span: Span) -> R<Expr> {
        let iop = match op {
            ast::UnaryOp::AddrOf => {
                return Err(span.error(
                    "pointers are not supported by Ganbate ('&' is only allowed as the argument of arrayLength)",
                ));
            }
            ast::UnaryOp::Deref => return Err(span.error("pointers are not supported by Ganbate")),
            ast::UnaryOp::Neg => ir::UnaryOp::Neg,
            ast::UnaryOp::Not => ir::UnaryOp::Not,
            ast::UnaryOp::BitNot => ir::UnaryOp::BitNot,
        };
        let v = self.value(x)?;
        let ok = v.ty.is_scalar_or_vector()
            && match (iop, v.ty.scalar()) {
                (ir::UnaryOp::Neg, Some(s)) => s.is_numeric() && s != Scalar::U32,
                (ir::UnaryOp::Not, Some(s)) => s == Scalar::Bool,
                (ir::UnaryOp::BitNot, Some(s)) => s.is_int(),
                _ => false,
            };
        if !ok {
            let sym = match iop {
                ir::UnaryOp::Neg => "-",
                ir::UnaryOp::Not => "!",
                ir::UnaryOp::BitNot => "~",
            };
            return Err(span.error(format!(
                "operator '{sym}' cannot be applied to {}",
                self.ty_name(&v.ty)
            )));
        }
        if let Some(c) = v.as_const() {
            let r = consteval::unary(iop, c).map_err(|m| span.error(m))?;
            return Ok(konst(v.ty, r, span));
        }
        Ok(Expr {
            ty: v.ty.clone(),
            kind: EK::Unary(iop, Box::new(v)),
            span,
        })
    }

    /// Convert abstract operands of a binary operator towards the other operand's type.
    fn unify(&self, l: Expr, r: Expr, span: Span) -> R<(Expr, Expr)> {
        let la = l.ty.is_abstract();
        let ra = r.ty.is_abstract();
        let towards = |a: Expr, other: &Ty| -> R<Expr> {
            let (Some(from), Some(to)) = (a.ty.leaf_scalar(), other.scalar()) else {
                return Ok(a);
            };
            if from.converts_to(to) {
                let t = a.ty.with_scalar(to);
                self.convert(a, &t, span)
            } else {
                Ok(a)
            }
        };
        Ok(match (la, ra) {
            (true, false) => {
                let rt = r.ty.clone();
                (towards(l, &rt)?, r)
            }
            (false, true) => {
                let lt = l.ty.clone();
                let r = towards(r, &lt)?;
                (l, r)
            }
            (true, true) => {
                let (ls, rs) = (l.ty.leaf_scalar(), r.ty.leaf_scalar());
                if ls == Some(Scalar::AbstractInt) && rs == Some(Scalar::AbstractFloat) {
                    let t = l.ty.with_scalar(Scalar::AbstractFloat);
                    (self.convert(l, &t, span)?, r)
                } else if ls == Some(Scalar::AbstractFloat) && rs == Some(Scalar::AbstractInt) {
                    let t = r.ty.with_scalar(Scalar::AbstractFloat);
                    let r = self.convert(r, &t, span)?;
                    (l, r)
                } else {
                    (l, r)
                }
            }
            (false, false) => (l, r),
        })
    }

    fn arith_type(op: BinaryOp, lt: &Ty, rt: &Ty) -> Option<Ty> {
        use BinaryOp::*;
        match (lt, rt) {
            (Ty::Scalar(a), Ty::Scalar(b)) if a == b && a.is_numeric() => Some(lt.clone()),
            (Ty::Vector(n, a), Ty::Vector(m, b)) if n == m && a == b && a.is_numeric() => {
                Some(lt.clone())
            }
            (Ty::Vector(n, a), Ty::Scalar(b)) | (Ty::Scalar(b), Ty::Vector(n, a))
                if a == b && a.is_numeric() =>
            {
                Some(Ty::Vector(*n, *a))
            }
            (Ty::Matrix(c, r), Ty::Matrix(c2, r2))
                if matches!(op, Add | Sub) && c == c2 && r == r2 =>
            {
                Some(lt.clone())
            }
            (Ty::Matrix(c1, r1), Ty::Matrix(c2, r2)) if op == Mul && c1 == r2 => {
                Some(Ty::Matrix(*c2, *r1))
            }
            (Ty::Matrix(..), Ty::Scalar(Scalar::F32)) if op == Mul => Some(lt.clone()),
            (Ty::Scalar(Scalar::F32), Ty::Matrix(..)) if op == Mul => Some(rt.clone()),
            (Ty::Matrix(c, r), Ty::Vector(n, Scalar::F32)) if op == Mul && n == c => {
                Some(Ty::Vector(*r, Scalar::F32))
            }
            (Ty::Vector(n, Scalar::F32), Ty::Matrix(c, r)) if op == Mul && n == r => {
                Some(Ty::Vector(*c, Scalar::F32))
            }
            _ => None,
        }
    }

    pub(crate) fn binary(&mut self, op: BinaryOp, l: Expr, r: Expr, span: Span) -> R<Expr> {
        use BinaryOp::*;
        let (l, r) = if matches!(op, Shl | Shr) {
            let r = if r.ty.is_abstract() {
                let t = r.ty.with_scalar(Scalar::U32);
                self.convert(r, &t, span)?
            } else {
                r
            };
            let l = if l.ty.is_abstract() && r.as_const().is_none() {
                self.concretize(l)?
            } else {
                l
            };
            (l, r)
        } else {
            self.unify(l, r, span)?
        };
        let (lt, rt) = (l.ty.clone(), r.ty.clone());
        let mismatch = || {
            span.error(format!(
                "operator '{}' cannot be applied to {} and {}",
                op.symbol(),
                type_name(&lt, &self.structs),
                type_name(&rt, &self.structs)
            ))
        };
        let result = match op {
            Add | Sub | Mul | Div | Rem => Self::arith_type(op, &lt, &rt).ok_or_else(mismatch)?,
            Eq | Ne | Lt | Le | Gt | Ge => {
                if lt != rt || !lt.is_scalar_or_vector() {
                    return Err(mismatch());
                }
                if !matches!(op, Eq | Ne) && lt.scalar() == Some(Scalar::Bool) {
                    return Err(mismatch());
                }
                lt.with_scalar(Scalar::Bool)
            }
            And | Or => {
                let s = lt.scalar();
                if lt != rt
                    || !lt.is_scalar_or_vector()
                    || !s.is_some_and(|s| s.is_int() || s == Scalar::Bool)
                {
                    return Err(mismatch());
                }
                lt.clone()
            }
            Xor => {
                if lt != rt || !lt.is_scalar_or_vector() || !lt.scalar().is_some_and(|s| s.is_int())
                {
                    return Err(mismatch());
                }
                lt.clone()
            }
            LogicalAnd | LogicalOr => {
                if lt != BOOL || rt != BOOL {
                    return Err(span.error(format!(
                        "operator '{}' requires bool operands, found {} and {} (use '{}' for vectors of bool)",
                        op.symbol(),
                        self.ty_name(&lt),
                        self.ty_name(&rt),
                        if op == LogicalAnd { "&" } else { "|" }
                    )));
                }
                BOOL
            }
            Shl | Shr => {
                let ok = lt.is_scalar_or_vector()
                    && lt.scalar().is_some_and(|s| s.is_int())
                    && rt == lt.with_scalar(Scalar::U32);
                if !ok {
                    return Err(span.error(format!(
                        "operator '{}' needs an integer left operand and a u32 shift amount of the same shape, found {} and {}",
                        op.symbol(),
                        self.ty_name(&lt),
                        self.ty_name(&rt)
                    )));
                }
                lt.clone()
            }
        };
        if let (Some(a), Some(b)) = (l.as_const(), r.as_const()) {
            match consteval::binary(op, a, &lt, b, &rt) {
                Ok(Some(v)) => return Ok(konst(result, v, span)),
                Ok(None) => {}
                Err(m) => return Err(span.error(m)),
            }
        }
        if result.is_abstract() {
            // Only reachable for unfoldable abstract operations; concretize and retry.
            let l = self.concretize(l)?;
            let r = self.concretize(r)?;
            return self.binary(op, l, r, span);
        }
        Ok(Expr {
            ty: result,
            kind: EK::Binary(op, Box::new(l), Box::new(r)),
            span,
        })
    }

    /// Automatic conversion of `e` to `to` (identity, or abstract → concrete).
    pub(crate) fn convert(&self, e: Expr, to: &Ty, span: Span) -> R<Expr> {
        if e.ty == *to {
            return Ok(e);
        }
        if e.ty.is_abstract()
            && same_shape(&e.ty, to)
            && let (Some(from), Some(ts)) = (e.ty.leaf_scalar(), to.leaf_scalar())
            && from.converts_to(ts)
            && let Some(c) = e.as_const()
        {
            let v = consteval::auto_convert(c, ts).map_err(|m| e.span.error(m))?;
            return Ok(konst(to.clone(), v, e.span));
        }
        let at = if e.span == Span::default() {
            span
        } else {
            e.span
        };
        Err(at.error(format!(
            "type mismatch: expected {}, found {}",
            self.ty_name(to),
            self.ty_name(&e.ty)
        )))
    }

    pub(crate) fn concretize(&self, e: Expr) -> R<Expr> {
        if !e.ty.is_abstract() {
            return Ok(e);
        }
        let to = e.ty.concretize();
        let span = e.span;
        self.convert(e, &to, span)
    }

    fn call(
        &mut self,
        name: &str,
        templates: Option<&[ast::Expr]>,
        args: &[ast::Expr],
        span: Span,
        stmt: bool,
    ) -> R<Expr> {
        if self.lookup_local(name, span)?.is_some() {
            return Err(span.error(format!("'{name}' is a local declaration, not a function")));
        }
        if self.names.contains_key(name) {
            let no_tmpl = || -> R<()> {
                if templates.is_some() {
                    Err(span.error(format!("'{name}' does not take template arguments")))
                } else {
                    Ok(())
                }
            };
            return match self.module_item(name)? {
                Some(Resolved::Fn(fid)) => {
                    no_tmpl()?;
                    self.user_call(fid, args, span)
                }
                Some(Resolved::Struct(sid)) => {
                    no_tmpl()?;
                    self.construct(Ctor::Full(Ty::Struct(sid)), args, span)
                }
                Some(Resolved::Alias(t)) => {
                    no_tmpl()?;
                    self.construct(Ctor::Full(t), args, span)
                }
                _ => Err(span.error(format!("'{name}' is not a function or a type"))),
            };
        }
        if name == "bitcast" {
            return self.bitcast(templates, args, span);
        }
        let ctor = match (name, templates) {
            ("vec2" | "vec3" | "vec4", None) => Some(Ctor::Vec(name.as_bytes()[3] - b'0')),
            ("array", None) => Some(Ctor::Array),
            (n, None) if mat_name(n).is_some_and(|m| m.2.is_empty()) => {
                let (c, r, _) = mat_name(n).expect("matrix");
                Some(Ctor::Mat(c, r))
            }
            _ => self
                .predeclared_type(name, templates, span)?
                .map(Ctor::Full),
        };
        if let Some(c) = ctor {
            return self.construct(c, args, span);
        }
        if templates.is_some() {
            return Err(span.error(format!("'{name}' does not take template arguments")));
        }
        self.builtin_call(name, args, span, stmt)
    }

    fn user_call(&mut self, fid: FnId, args: &[ast::Expr], span: Span) -> R<Expr> {
        if self.fcx.is_none() {
            return Err(span.error("function calls are not allowed in constant expressions"));
        }
        let f = &self.functions[fid];
        let fname = f.name.clone();
        if f.stage.is_some() {
            return Err(span.error(format!("entry point '{fname}' cannot be called")));
        }
        if args.len() != f.params.len() {
            return Err(span.error(format!(
                "function '{fname}' expects {} argument(s), got {}",
                f.params.len(),
                args.len()
            )));
        }
        let params: Vec<(String, Ty)> = f
            .params
            .iter()
            .map(|p| (p.name.clone(), p.ty.clone()))
            .collect();
        let ret = f.ret.clone();
        let mut out = Vec::new();
        for (a, (pname, pty)) in args.iter().zip(params) {
            let v = self.value(a)?;
            let v = self.convert(v, &pty, a.span).map_err(|e| Error {
                message: format!("argument '{pname}' of '{fname}': {}", e.message),
                ..e
            })?;
            out.push(v);
        }
        self.fcx_mut().calls.insert(fid);
        Ok(Expr {
            ty: ret.unwrap_or(Ty::Void),
            kind: EK::Call(fid, out),
            span,
        })
    }

    fn bitcast(
        &mut self,
        templates: Option<&[ast::Expr]>,
        args: &[ast::Expr],
        span: Span,
    ) -> R<Expr> {
        let Some([t]) = templates else {
            return Err(span.error("bitcast needs one template argument: bitcast<T>(e)"));
        };
        let to = self.resolve_type(t)?;
        if args.len() != 1 {
            return Err(span.error("bitcast takes exactly one argument"));
        }
        let v = self.value(&args[0])?;
        let v = self.concretize(v)?;
        let ok32 = |t: &Ty| {
            t.is_scalar_or_vector()
                && matches!(t.scalar(), Some(Scalar::I32 | Scalar::U32 | Scalar::F32))
        };
        if !ok32(&to) || !ok32(&v.ty) || v.ty.vec_size() != to.vec_size() {
            return Err(span.error(format!(
                "cannot bitcast {} to {}",
                self.ty_name(&v.ty),
                self.ty_name(&to)
            )));
        }
        if v.ty == to {
            return Ok(v);
        }
        if let Some(c) = v.as_const() {
            let s = to.scalar().expect("scalar");
            let r = consteval::map_leaves(c, &mut |x| consteval::bitcast_scalar(x, s))
                .map_err(|m| span.error(m))?;
            return Ok(konst(to, r, span));
        }
        Ok(Expr {
            ty: to,
            kind: EK::Bitcast(Box::new(v)),
            span,
        })
    }

    /// The component type of an inferred `vecN(...)`.
    fn common_scalar(&self, vals: &[Expr], span: Span) -> R<Scalar> {
        let mut concrete: Option<Scalar> = None;
        let mut any_float = false;
        for v in vals {
            let s = match &v.ty {
                Ty::Scalar(s) | Ty::Vector(_, s) => *s,
                other => {
                    return Err(span.error(format!(
                        "vector components must be scalars or vectors, found {}",
                        self.ty_name(other)
                    )));
                }
            };
            if s.is_abstract() {
                any_float |= s == Scalar::AbstractFloat;
            } else if let Some(c) = concrete {
                if c != s {
                    return Err(span.error(format!(
                        "mixed component types {} and {}",
                        c.name(),
                        s.name()
                    )));
                }
            } else {
                concrete = Some(s);
            }
        }
        Ok(concrete.unwrap_or(if any_float {
            Scalar::AbstractFloat
        } else {
            Scalar::AbstractInt
        }))
    }

    /// The element type of an inferred `array(...)`.
    fn common_type(&self, vals: &[Expr], span: Span) -> R<Ty> {
        let mut t = vals[0].ty.clone();
        for v in &vals[1..] {
            if v.ty == t {
                continue;
            }
            let (a, b) = (t.leaf_scalar(), v.ty.leaf_scalar());
            if same_shape(&t, &v.ty)
                && let (Some(a), Some(b)) = (a, b)
            {
                if a.is_abstract() && a.converts_to(b) {
                    t = t.with_scalar(b);
                    continue;
                }
                if b.is_abstract() && b.converts_to(a) {
                    continue;
                }
            }
            return Err(span.error(format!(
                "array elements have different types: {} and {}",
                self.ty_name(&t),
                self.ty_name(&v.ty)
            )));
        }
        Ok(t)
    }

    fn construct(&mut self, ctor: Ctor, args: &[ast::Expr], span: Span) -> R<Expr> {
        let vals: Vec<Expr> = args.iter().map(|a| self.value(a)).collect::<R<_>>()?;
        match ctor {
            Ctor::Full(ty) => self.construct_full(ty, vals, span),
            Ctor::Vec(n) => {
                if vals.is_empty() {
                    return Err(span.error(format!(
                        "cannot infer the component type of vec{n}(); write vec{n}<f32>()"
                    )));
                }
                if let [v] = vals.as_slice()
                    && let Ty::Vector(m, s) = v.ty
                    && m == n
                {
                    return self.construct_full(Ty::Vector(n, s), vals, span);
                }
                let s = self.common_scalar(&vals, span)?;
                self.construct_full(Ty::Vector(n, s), vals, span)
            }
            Ctor::Mat(c, r) => self.construct_full(Ty::Matrix(c, r), vals, span),
            Ctor::Array => {
                if vals.is_empty() {
                    return Err(span.error("cannot infer the type of array(); write array<T, N>()"));
                }
                let e = self.common_type(&vals, span)?;
                if !is_constructible(&e, &self.structs) {
                    return Err(span.error("array elements must be constructible"));
                }
                let n = vals.len() as u32;
                self.construct_full(Ty::Array(Box::new(e), Some(n)), vals, span)
            }
        }
    }

    fn fold_construct(ty: Ty, comps: Vec<Expr>, span: Span) -> Expr {
        if comps.iter().all(|c| c.as_const().is_some()) {
            let flatten = matches!(ty, Ty::Vector(..));
            let mut out = Vec::new();
            for c in &comps {
                let v = c.as_const().expect("const").clone();
                match (flatten, v) {
                    (true, ConstValue::Composite(parts)) => out.extend(parts),
                    (_, v) => out.push(v),
                }
            }
            return konst(ty, ConstValue::Composite(out), span);
        }
        Expr {
            ty,
            kind: EK::Construct(comps),
            span,
        }
    }

    fn construct_full(&mut self, ty: Ty, vals: Vec<Expr>, span: Span) -> R<Expr> {
        let tn = self.ty_name(&ty);
        if vals.is_empty() {
            if !is_constructible(&ty, &self.structs) {
                return Err(span.error(format!("cannot construct a value of type {tn}")));
            }
            if leaf_count(&ty, &self.structs) > 4096 {
                return Ok(self.zero(&ty, span));
            }
            return Ok(konst(
                ty.clone(),
                ConstValue::zero(&ty, &self.structs),
                span,
            ));
        }
        match &ty {
            Ty::Scalar(to) => {
                let [v] = <[Expr; 1]>::try_from(vals)
                    .map_err(|_| span.error(format!("{tn}(...) takes exactly one argument")))?;
                if !v.ty.is_scalar() {
                    return Err(
                        span.error(format!("cannot convert {} to {tn}", self.ty_name(&v.ty)))
                    );
                }
                self.numeric_conversion(v, Ty::Scalar(*to), span)
            }
            Ty::Vector(n, s) => {
                if vals.len() == 1 {
                    match vals[0].ty {
                        Ty::Scalar(_) => {
                            let v = vals.into_iter().next().expect("one");
                            let v = self.convert(v, &Ty::Scalar(*s), span)?;
                            if let Some(c) = v.as_const() {
                                return Ok(konst(
                                    ty.clone(),
                                    ConstValue::Composite(vec![c.clone(); *n as usize]),
                                    span,
                                ));
                            }
                            return Ok(Expr {
                                ty: ty.clone(),
                                kind: EK::Splat(Box::new(v)),
                                span,
                            });
                        }
                        Ty::Vector(m, _) if m == *n => {
                            let v = vals.into_iter().next().expect("one");
                            return self.numeric_conversion(v, ty.clone(), span);
                        }
                        _ => {}
                    }
                }
                let mut comps = Vec::new();
                let mut count = 0u32;
                for v in vals {
                    let vt = v.ty.clone();
                    match vt {
                        Ty::Scalar(_) => {
                            comps.push(self.convert(v, &Ty::Scalar(*s), span)?);
                            count += 1;
                        }
                        Ty::Vector(m, _) => {
                            comps.push(self.convert(v, &Ty::Vector(m, *s), span)?);
                            count += m as u32;
                        }
                        other => {
                            return Err(span.error(format!(
                                "{tn} components must be scalars or vectors, found {}",
                                self.ty_name(&other)
                            )));
                        }
                    }
                }
                if count != *n as u32 {
                    return Err(span.error(format!("{tn} needs {n} components, got {count}")));
                }
                Ok(Self::fold_construct(ty.clone(), comps, span))
            }
            Ty::Matrix(c, r) => {
                if vals.len() == 1 && vals[0].ty == ty {
                    return Ok(vals.into_iter().next().expect("one"));
                }
                let col = Ty::Vector(*r, Scalar::F32);
                if vals.len() == *c as usize {
                    let cols = vals
                        .into_iter()
                        .map(|v| self.convert(v, &col, span))
                        .collect::<R<Vec<_>>>()?;
                    return Ok(Self::fold_construct(ty.clone(), cols, span));
                }
                if vals.len() == (*c as usize) * (*r as usize) {
                    let scalars = vals
                        .into_iter()
                        .map(|v| self.convert(v, &F32, span))
                        .collect::<R<Vec<_>>>()?;
                    let cols = scalars
                        .chunks(*r as usize)
                        .map(|ch| Self::fold_construct(col.clone(), ch.to_vec(), span))
                        .collect();
                    return Ok(Self::fold_construct(ty.clone(), cols, span));
                }
                Err(span.error(format!(
                    "{tn} needs {c} column vectors or {} scalars, got {} argument(s)",
                    c * r,
                    vals.len()
                )))
            }
            Ty::Array(e, Some(n)) => {
                if vals.len() != *n as usize {
                    return Err(span.error(format!("{tn} needs {n} elements, got {}", vals.len())));
                }
                let elems = vals
                    .into_iter()
                    .map(|v| self.convert(v, e, span))
                    .collect::<R<Vec<_>>>()?;
                Ok(Self::fold_construct(ty.clone(), elems, span))
            }
            Ty::Struct(sid) => {
                let mtys: Vec<(String, Ty)> = self.structs[*sid]
                    .members
                    .iter()
                    .map(|m| (m.name.clone(), m.ty.clone()))
                    .collect();
                if vals.len() != mtys.len() {
                    return Err(span.error(format!(
                        "{tn}(...) needs {} member values, got {}",
                        mtys.len(),
                        vals.len()
                    )));
                }
                let mut members = Vec::new();
                for (v, (mname, mty)) in vals.into_iter().zip(mtys) {
                    let v = self.convert(v, &mty, span).map_err(|e| Error {
                        message: format!("member '{mname}' of {tn}: {}", e.message),
                        ..e
                    })?;
                    members.push(v);
                }
                Ok(Self::fold_construct(ty.clone(), members, span))
            }
            _ => Err(span.error(format!("cannot construct a value of type {tn}"))),
        }
    }

    /// Value-constructor conversion between scalar types of the same shape.
    fn numeric_conversion(&self, v: Expr, to: Ty, span: Span) -> R<Expr> {
        if v.ty == to {
            return Ok(v);
        }
        let ts = to.scalar().expect("scalar");
        if let Some(c) = v.as_const() {
            let r = consteval::map_leaves(c, &mut |x| consteval::cast_scalar(x, ts))
                .map_err(|m| span.error(m))?;
            return Ok(konst(to, r, span));
        }
        Ok(Expr {
            ty: to,
            kind: EK::Convert(Box::new(v)),
            span,
        })
    }
}

enum LoopTail<'b> {
    None,
    Update(Option<&'b ast::Stmt>),
    Continuing(Option<&'b ast::Continuing>),
}
