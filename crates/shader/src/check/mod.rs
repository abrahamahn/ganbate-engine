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
}

mod expr;
mod stmt;
