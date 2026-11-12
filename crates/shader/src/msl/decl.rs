use super::expr::{bare, postfix};
use super::*;

impl<'p> Writer<'p> {
    // ------------------------------------------------------------------ analysis

    pub(super) fn analyze(&mut self, order: &[FnId]) -> R<()> {
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
    pub(super) fn mark_host(&mut self, g: GlobalId) {
        let gl = &self.prog.globals[g];
        if matches!(
            gl.kind,
            GlobalKind::Uniform | GlobalKind::Storage { .. } | GlobalKind::PushConstant
        ) {
            let ty = gl.ty.clone();
            self.mark_host_ty(&ty);
        }
    }

    pub(super) fn mark_host_ty(&mut self, t: &Ty) {
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

    pub(super) fn effects(&self, e: &Expr) -> bool {
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

    pub(super) fn writable_global(&self, g: GlobalId) -> bool {
        matches!(
            self.prog.globals[g].kind,
            GlobalKind::Private | GlobalKind::Storage { read_only: false }
        )
    }

    /// Reads memory that a call with side effects could change.
    pub(super) fn reads(&self, e: &Expr, current_global: bool) -> bool {
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
    pub(super) fn needs_order(&self, es: &[&Expr]) -> bool {
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
    pub(super) fn needs_hoist(&self, e: &Expr) -> bool {
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

    pub(super) fn has_vertex_inputs(&self, fid: FnId) -> bool {
        let f = &self.prog.functions[fid];
        f.params.iter().any(|p| {
            p.io.location.is_some()
                || matches!(p.ty, Ty::Struct(sid) if p.io.is_empty()
                    && self.prog.structs[sid].members.iter().any(|m| m.io.location.is_some()))
        })
    }

    // ------------------------------------------------------------------ types

    pub(super) fn ty_name(&mut self, t: &Ty) -> R<String> {
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

    pub(super) fn array_name(&mut self, t: &Ty) -> R<String> {
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

    pub(super) fn struct_plan(&mut self, sid: usize) -> R<&StructPlan> {
        if !self.struct_plans.contains_key(&sid) {
            self.define_struct(sid)?;
        }
        Ok(&self.struct_plans[&sid])
    }

    pub(super) fn define_struct(&mut self, sid: usize) -> R<()> {
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
    pub(super) fn construct_struct(&mut self, sid: usize, parts: Vec<String>) -> R<String> {
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

    pub(super) fn scalar_const(&self, v: &ConstValue, s: Scalar) -> String {
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

    pub(super) fn const_value(&mut self, ty: &Ty, v: &ConstValue) -> R<String> {
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

    pub(super) fn zero(&mut self, ty: &Ty) -> R<String> {
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

    pub(super) fn helper(&mut self, h: &'static str) -> String {
        self.helpers.insert(h);
        if let Some(n) = self.helper_names.get(h) {
            return n.clone();
        }
        let n = self.names.generated(h);
        self.helper_names.insert(h, n.clone());
        n
    }

    pub(super) fn helper_source(&self) -> String {
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

    pub(super) fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    pub(super) fn flush_pre(&mut self) {
        let pre = std::mem::take(&mut self.fs.pre);
        for l in pre {
            self.line(&l);
        }
    }

    /// The parameter declaration of global `g` inside ordinary functions.
    pub(super) fn global_param(&mut self, g: GlobalId) -> R<String> {
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

    pub(super) fn function(&mut self, fid: FnId) -> R<()> {
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

    pub(super) fn let_type(&self, body: &[Stmt], id: ValueId) -> R<Ty> {
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
    pub(super) fn block(&mut self, stmts: &[Stmt]) -> R<bool> {
        for s in stmts {
            if !self.stmt(s)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn nested(&mut self, header: &str, stmts: &[Stmt]) -> R<bool> {
        self.line(&format!("{header} {{"));
        self.indent += 1;
        let falls = self.block(stmts)?;
        self.indent -= 1;
        self.line("}");
        Ok(falls)
    }

    pub(super) fn stmt(&mut self, s: &Stmt) -> R<bool> {
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
    pub(super) fn indent_back_else(&mut self) {
        let trimmed = self.out.trim_end_matches('\n').len();
        self.out.truncate(trimmed);
        self.out.push_str(" else {\n");
    }

    pub(super) fn prog_fn_value_name(&self, id: ValueId) -> String {
        // Values are numbered per function.
        self.prog.functions[self.cur_fn]
            .value_names
            .get(id as usize)
            .cloned()
            .unwrap_or_else(|| "v".into())
    }

    pub(super) fn loop_stmt(
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

    pub(super) fn loop_cond(&mut self, cond: Option<&Expr>) -> R<()> {
        if let Some(c) = cond {
            let c = self.expr(c)?;
            self.flush_pre();
            self.line(&format!("if (!{}) {{", postfix(&c)));
            self.line("    break;");
            self.line("}");
        }
        Ok(())
    }

    pub(super) fn loop_break_if(&mut self, break_if: Option<&Expr>) -> R<()> {
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
