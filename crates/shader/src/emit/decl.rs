use super::*;

impl<'p> Emitter<'p> {
    pub(super) fn id(&mut self) -> u32 {
        let i = self.next;
        self.next += 1;
        i
    }

    pub(super) fn name(&mut self, id: u32, name: &str) {
        if !self.named.insert(id) {
            return;
        }
        let mut w = vec![id];
        w.extend(spirv::string(name));
        inst(&mut self.s.debug, op::NAME, &w);
    }

    fn member_name(&mut self, id: u32, i: u32, name: &str) {
        let mut w = vec![id, i];
        w.extend(spirv::string(name));
        inst(&mut self.s.debug, op::MEMBER_NAME, &w);
    }

    pub(super) fn decorate(&mut self, id: u32, args: &[u32]) {
        let mut w = vec![id];
        w.extend_from_slice(args);
        inst(&mut self.s.annotations, op::DECORATE, &w);
    }

    fn member_decorate(&mut self, id: u32, member: u32, args: &[u32]) {
        let mut w = vec![id, member];
        w.extend_from_slice(args);
        inst(&mut self.s.annotations, op::MEMBER_DECORATE, &w);
    }

    // ------------------------------------------------------------------ types

    fn cached(&mut self, key: TypeKey, make: impl FnOnce(&mut Self, u32)) -> u32 {
        if let Some(&id) = self.types.get(&key) {
            return id;
        }
        let id = self.id();
        make(self, id);
        self.types.insert(key, id);
        id
    }

    pub(super) fn void_ty(&mut self) -> u32 {
        self.cached(TypeKey::Void, |e, id| {
            inst(&mut e.s.globals, op::TYPE_VOID, &[id])
        })
    }

    pub(super) fn scalar_ty(&mut self, s: Scalar) -> u32 {
        match s {
            Scalar::Bool => self.cached(TypeKey::Bool, |e, id| {
                inst(&mut e.s.globals, op::TYPE_BOOL, &[id])
            }),
            Scalar::I32 | Scalar::AbstractInt => self.cached(TypeKey::Int(true), |e, id| {
                inst(&mut e.s.globals, op::TYPE_INT, &[id, 32, 1])
            }),
            Scalar::U32 => self.cached(TypeKey::Int(false), |e, id| {
                inst(&mut e.s.globals, op::TYPE_INT, &[id, 32, 0])
            }),
            Scalar::F32 | Scalar::AbstractFloat => self.cached(TypeKey::Float, |e, id| {
                inst(&mut e.s.globals, op::TYPE_FLOAT, &[id, 32])
            }),
        }
    }

    pub(super) fn ty(&mut self, t: &Ty, fl: Flavor) -> u32 {
        match t {
            Ty::Scalar(s) => self.scalar_ty(*s),
            Ty::Vector(n, s) => {
                let c = self.scalar_ty(*s);
                let n = *n;
                self.cached(TypeKey::Vector(c, n), |e, id| {
                    inst(&mut e.s.globals, op::TYPE_VECTOR, &[id, c, n as u32])
                })
            }
            Ty::Matrix(c, r) => {
                let col = self.ty(&Ty::Vector(*r, Scalar::F32), Flavor::Plain);
                let c = *c;
                self.cached(TypeKey::Matrix(col, c), |e, id| {
                    inst(&mut e.s.globals, op::TYPE_MATRIX, &[id, col, c as u32])
                })
            }
            Ty::Array(elem, Some(n)) => {
                let efl = fl.member();
                let eid = self.ty(elem, efl);
                let stride = (efl != Flavor::Plain).then(|| array_stride(elem, &self.prog.structs));
                let len = self.const_u32(*n);
                let n = *n;
                self.cached(TypeKey::Array(eid, n, stride), |e, id| {
                    inst(&mut e.s.globals, op::TYPE_ARRAY, &[id, eid, len]);
                    if let Some(s) = stride {
                        e.decorate(id, &[deco::ARRAY_STRIDE, s]);
                    }
                })
            }
            Ty::Array(elem, None) => {
                let eid = self.ty(elem, Flavor::Explicit);
                let stride = array_stride(elem, &self.prog.structs);
                self.cached(TypeKey::RuntimeArray(eid, stride), |e, id| {
                    inst(&mut e.s.globals, op::TYPE_RUNTIME_ARRAY, &[id, eid]);
                    e.decorate(id, &[deco::ARRAY_STRIDE, stride]);
                })
            }
            Ty::Struct(sid) => self.struct_ty(*sid, fl),
            Ty::Texture(..) | Ty::StorageTexture(_) => self.image_ty(t),
            Ty::Sampler(_) => self.cached(TypeKey::Sampler, |e, id| {
                inst(&mut e.s.globals, op::TYPE_SAMPLER, &[id])
            }),
            Ty::Void => self.void_ty(),
        }
    }

    fn layout_member_decorations(&mut self, id: u32, i: u32, ty: &Ty, offset: u32) {
        self.member_decorate(id, i, &[deco::OFFSET, offset]);
        if let Ty::Matrix(_, rows) = ty.innermost() {
            self.member_decorate(id, i, &[deco::COL_MAJOR]);
            self.member_decorate(id, i, &[deco::MATRIX_STRIDE, matrix_stride(*rows)]);
        }
    }

    fn struct_ty(&mut self, sid: usize, fl: Flavor) -> u32 {
        if let Some(&id) = self.types.get(&TypeKey::Struct(sid, fl)) {
            return id;
        }
        let prog = self.prog;
        let info = &prog.structs[sid];
        let mfl = fl.member();
        let members: Vec<u32> = info.members.iter().map(|m| self.ty(&m.ty, mfl)).collect();
        let id = self.id();
        let mut w = vec![id];
        w.extend(&members);
        inst(&mut self.s.globals, op::TYPE_STRUCT, &w);
        self.types.insert(TypeKey::Struct(sid, fl), id);
        let suffix = match fl {
            Flavor::Plain => "",
            Flavor::Explicit => "_std",
            Flavor::Block => "_block",
            Flavor::BufferBlock { read_only: true } => "_ro_buffer",
            Flavor::BufferBlock { read_only: false } => "_buffer",
        };
        self.name(id, &format!("{}{suffix}", info.name));
        for (i, m) in info.members.iter().enumerate() {
            self.member_name(id, i as u32, &m.name);
        }
        if fl != Flavor::Plain {
            let layout = info.layout.as_ref().expect("host-shareable struct");
            for (i, m) in info.members.iter().enumerate() {
                self.layout_member_decorations(id, i as u32, &m.ty, layout.offsets[i]);
                if fl == (Flavor::BufferBlock { read_only: true }) {
                    self.member_decorate(id, i as u32, &[deco::NON_WRITABLE]);
                }
            }
            match fl {
                Flavor::Block => self.decorate(id, &[deco::BLOCK]),
                Flavor::BufferBlock { .. } => self.decorate(id, &[deco::BUFFER_BLOCK]),
                _ => {}
            }
        }
        id
    }

    /// `struct { T }` around a non-struct buffer variable type.
    fn wrapper_ty(&mut self, g: GlobalId, fl: Flavor) -> u32 {
        let prog = self.prog;
        let gl = &prog.globals[g];
        let inner = self.ty(&gl.ty, Flavor::Explicit);
        self.cached(TypeKey::Wrapper(g), |e, id| {
            inst(&mut e.s.globals, op::TYPE_STRUCT, &[id, inner]);
            e.name(id, &format!("{}_block", gl.name));
            e.member_name(id, 0, &gl.name);
            e.layout_member_decorations(id, 0, &gl.ty, 0);
            if fl == (Flavor::BufferBlock { read_only: true }) {
                e.member_decorate(id, 0, &[deco::NON_WRITABLE]);
            }
            let d = if fl == Flavor::Block {
                deco::BLOCK
            } else {
                deco::BUFFER_BLOCK
            };
            e.decorate(id, &[d]);
        })
    }

    fn image_ty(&mut self, t: &Ty) -> u32 {
        let (st, dim, depth, arrayed, sampled, format) = match t {
            Ty::Texture(view, sample) => {
                let st = match sample {
                    SampleType::Float | SampleType::Depth => self.scalar_ty(Scalar::F32),
                    SampleType::Sint => self.scalar_ty(Scalar::I32),
                    SampleType::Uint => self.scalar_ty(Scalar::U32),
                };
                let dim = if *view == TextureView::Cube {
                    spirv::dim::CUBE
                } else {
                    spirv::dim::D2
                };
                (
                    st,
                    dim,
                    (*sample == SampleType::Depth) as u32,
                    (*view == TextureView::D2Array) as u32,
                    1,
                    spirv::image_format::UNKNOWN,
                )
            }
            Ty::StorageTexture(f) => {
                use spirv::image_format as imf;
                let format = match f {
                    TexelFormat::Rgba8Unorm => imf::RGBA8,
                    TexelFormat::Rgba16Float => imf::RGBA16F,
                    TexelFormat::Rgba32Float => imf::RGBA32F,
                    TexelFormat::R32Float => imf::R32F,
                    TexelFormat::Rg16Float => imf::RG16F,
                    TexelFormat::R16Float => imf::R16F,
                };
                if matches!(f, TexelFormat::Rg16Float | TexelFormat::R16Float) {
                    self.caps
                        .insert(spirv::capability::STORAGE_IMAGE_EXTENDED_FORMATS);
                }
                (self.scalar_ty(Scalar::F32), spirv::dim::D2, 0, 0, 2, format)
            }
            _ => unreachable!("image type"),
        };
        self.cached(
            TypeKey::Image(st, dim, depth, arrayed, sampled, format),
            |e, id| {
                inst(
                    &mut e.s.globals,
                    op::TYPE_IMAGE,
                    &[id, st, dim, depth, arrayed, 0, sampled, format],
                )
            },
        )
    }

    pub(super) fn sampled_image_ty(&mut self, image: u32) -> u32 {
        self.cached(TypeKey::SampledImage(image), |e, id| {
            inst(&mut e.s.globals, op::TYPE_SAMPLED_IMAGE, &[id, image])
        })
    }

    pub(super) fn ptr(&mut self, class: u32, pointee: u32) -> u32 {
        self.cached(TypeKey::Pointer(class, pointee), |e, id| {
            inst(&mut e.s.globals, op::TYPE_POINTER, &[id, class, pointee])
        })
    }

    pub(super) fn fn_ty(&mut self, ret: u32, params: &[u32]) -> u32 {
        self.cached(TypeKey::Function(ret, params.to_vec()), |e, id| {
            let mut w = vec![id, ret];
            w.extend(params);
            inst(&mut e.s.globals, op::TYPE_FUNCTION, &w);
        })
    }

    // ------------------------------------------------------------------ constants

    fn const_cached(&mut self, key: ConstKey, make: impl FnOnce(&mut Self, u32)) -> u32 {
        if let Some(&id) = self.consts.get(&key) {
            return id;
        }
        let id = self.id();
        make(self, id);
        self.consts.insert(key, id);
        id
    }

    fn const_bits(&mut self, s: Scalar, bits: u32) -> u32 {
        let t = self.scalar_ty(s);
        self.const_cached(ConstKey::Scalar(t, bits), |e, id| {
            inst(&mut e.s.globals, op::CONSTANT, &[t, id, bits])
        })
    }

    pub(super) fn const_u32(&mut self, v: u32) -> u32 {
        self.const_bits(Scalar::U32, v)
    }

    fn const_i32(&mut self, v: i32) -> u32 {
        self.const_bits(Scalar::I32, v as u32)
    }

    pub(super) fn const_f32(&mut self, v: f32) -> u32 {
        self.const_bits(Scalar::F32, v.to_bits())
    }

    pub(super) fn const_bool(&mut self, b: bool) -> u32 {
        let t = self.scalar_ty(Scalar::Bool);
        self.const_cached(ConstKey::Bool(b), |e, id| {
            inst(
                &mut e.s.globals,
                if b {
                    op::CONSTANT_TRUE
                } else {
                    op::CONSTANT_FALSE
                },
                &[t, id],
            )
        })
    }

    fn const_scalar(&mut self, v: &ConstValue) -> u32 {
        match *v {
            ConstValue::Bool(b) => self.const_bool(b),
            ConstValue::I32(i) => self.const_i32(i),
            ConstValue::U32(u) => self.const_u32(u),
            ConstValue::F32(f) => self.const_f32(f),
            ConstValue::AbstractInt(i) => self.const_i32(i as i32),
            ConstValue::AbstractFloat(f) => self.const_f32(f as f32),
            ConstValue::Composite(_) => self.const_u32(0),
        }
    }

    pub(super) fn null(&mut self, ty: &Ty, fl: Flavor) -> u32 {
        let t = self.ty(ty, fl);
        self.const_cached(ConstKey::Null(t), |e, id| {
            inst(&mut e.s.globals, op::CONSTANT_NULL, &[t, id])
        })
    }

    fn is_zero(v: &ConstValue) -> bool {
        match v {
            ConstValue::Bool(b) => !b,
            ConstValue::I32(i) => *i == 0,
            ConstValue::U32(u) => *u == 0,
            ConstValue::F32(f) => f.to_bits() == 0,
            ConstValue::AbstractInt(i) => *i == 0,
            ConstValue::AbstractFloat(f) => f.to_bits() == 0,
            ConstValue::Composite(c) => c.iter().all(Self::is_zero),
        }
    }

    pub(super) fn const_value(&mut self, ty: &Ty, v: &ConstValue, fl: Flavor) -> R<u32> {
        if ty.is_scalar() {
            return Ok(self.const_scalar(v));
        }
        if Self::is_zero(v) {
            return Ok(self.null(ty, fl));
        }
        let parts = v.components();
        if parts.len() > 65_000 {
            return Err(self
                .span
                .error("constant is too large for a SPIR-V instruction"));
        }
        let mut ids = Vec::with_capacity(parts.len());
        let prog = self.prog;
        for (i, p) in parts.iter().enumerate() {
            let pt = match ty {
                Ty::Vector(_, s) => Ty::Scalar(*s),
                Ty::Matrix(_, r) => Ty::Vector(*r, Scalar::F32),
                Ty::Array(e, _) => (**e).clone(),
                Ty::Struct(sid) => prog.structs[*sid].members[i].ty.clone(),
                _ => return Err(internal(self.span, "constant of non-composite type")),
            };
            ids.push(self.const_value(&pt, p, fl.member())?);
        }
        let t = self.ty(ty, fl);
        Ok(
            self.const_cached(ConstKey::Composite(t, ids.clone()), |e, id| {
                let mut w = vec![t, id];
                w.extend(&ids);
                inst(&mut e.s.globals, op::CONSTANT_COMPOSITE, &w);
            }),
        )
    }

    /// `n` copies of a scalar constant as a vector constant (or the scalar itself).
    pub(super) fn const_splat(&mut self, ty: &Ty, v: ConstValue) -> R<u32> {
        match ty {
            Ty::Vector(n, _) => self.const_value(
                ty,
                &ConstValue::Composite(vec![v; *n as usize]),
                Flavor::Plain,
            ),
            _ => Ok(self.const_scalar(&v)),
        }
    }

    // ------------------------------------------------------------------ globals

    pub(super) fn global_var(&mut self, g: GlobalId) -> R<()> {
        let prog = self.prog;
        let gl = &prog.globals[g];
        let (class, pointee, fl, wrapped) = match gl.kind {
            GlobalKind::Private | GlobalKind::ConstData => (
                sc::PRIVATE,
                self.ty(&gl.ty, Flavor::Plain),
                Flavor::Plain,
                false,
            ),
            GlobalKind::Uniform | GlobalKind::PushConstant | GlobalKind::Storage { .. } => {
                let class = if gl.kind == GlobalKind::PushConstant {
                    sc::PUSH_CONSTANT
                } else {
                    sc::UNIFORM
                };
                let top = match gl.kind {
                    GlobalKind::Storage { read_only } => Flavor::BufferBlock { read_only },
                    _ => Flavor::Block,
                };
                match gl.ty {
                    Ty::Struct(sid) => (class, self.struct_ty(sid, top), top, false),
                    _ => (class, self.wrapper_ty(g, top), Flavor::Explicit, true),
                }
            }
            GlobalKind::Handle => (
                sc::UNIFORM_CONSTANT,
                self.ty(&gl.ty, Flavor::Plain),
                Flavor::Plain,
                false,
            ),
        };
        let p = self.ptr(class, pointee);
        let init = match gl.kind {
            GlobalKind::Private | GlobalKind::ConstData => Some(match &gl.init {
                Some(v) => self.const_value(&gl.ty, v, Flavor::Plain)?,
                None => self.null(&gl.ty, Flavor::Plain),
            }),
            _ => None,
        };
        let id = self.id();
        let mut w = vec![p, id, class];
        w.extend(init);
        inst(&mut self.s.globals, op::VARIABLE, &w);
        self.name(id, &gl.name);
        if let Some((group, binding)) = gl.binding {
            self.decorate(id, &[deco::DESCRIPTOR_SET, group]);
            self.decorate(id, &[deco::BINDING, binding]);
        }
        if matches!(gl.ty, Ty::StorageTexture(_)) {
            self.decorate(id, &[deco::NON_READABLE]);
        }
        self.globals.insert(
            g,
            GlobalInfo {
                id,
                sc: class,
                fl,
                wrapped,
            },
        );
        Ok(())
    }

    pub(super) fn global(&self, g: GlobalId) -> R<GlobalInfo> {
        self.globals
            .get(&g)
            .copied()
            .ok_or_else(|| internal(self.span, "global variable was not emitted"))
    }

    // ------------------------------------------------------------------ functions

    pub(super) fn function(&mut self, fid: FnId) -> R<()> {
        let prog = self.prog;
        let f = &prog.functions[fid];
        self.span = f.span;
        self.cur_fn = fid;
        let ret = match &f.ret {
            Some(t) => self.ty(t, Flavor::Plain),
            None => self.void_ty(),
        };
        let params: Vec<u32> = f
            .params
            .iter()
            .map(|p| self.ty(&p.ty, Flavor::Plain))
            .collect();
        let fty = self.fn_ty(ret, &params);
        let id = self.fn_ids[&fid];
        self.name(id, &f.name);
        let mut header = Vec::new();
        inst(&mut header, op::FUNCTION, &[ret, id, 0, fty]);
        self.fs = FnState::default();
        for (p, &pt) in f.params.iter().zip(&params) {
            let pid = self.id();
            inst(&mut header, op::FUNCTION_PARAMETER, &[pt, pid]);
            self.fs.values.insert(p.value, Val::plain(pid));
            self.name(pid, &p.name);
        }
        let entry = self.id();
        self.fs.label = entry;
        for l in &f.locals {
            let t = self.ty(&l.ty, Flavor::Plain);
            let p = self.ptr(sc::FUNCTION, t);
            let v = self.id();
            inst(&mut self.fs.vars, op::VARIABLE, &[p, v, sc::FUNCTION]);
            self.name(v, &l.name);
            self.fs.locals.push(v);
        }
        self.block(&f.body)?;
        if !self.fs.terminated {
            let o = if f.ret.is_none() {
                op::RETURN
            } else {
                op::UNREACHABLE
            };
            inst(&mut self.fs.body, o, &[]);
        }
        let fs = std::mem::take(&mut self.fs);
        self.s.functions.extend(header);
        inst(&mut self.s.functions, op::LABEL, &[entry]);
        self.s.functions.extend(fs.vars);
        self.s.functions.extend(fs.body);
        inst(&mut self.s.functions, op::FUNCTION_END, &[]);
        Ok(())
    }

    pub(super) fn emit(&mut self, opcode: u16, operands: &[u32]) {
        inst(&mut self.fs.body, opcode, operands);
    }

    /// Emit an instruction with a result type and a fresh result id.
    pub(super) fn op(&mut self, opcode: u16, ty: u32, operands: &[u32]) -> u32 {
        let id = self.id();
        let mut w = Vec::with_capacity(operands.len() + 2);
        w.push(ty);
        w.push(id);
        w.extend_from_slice(operands);
        inst(&mut self.fs.body, opcode, &w);
        id
    }

    pub(super) fn ext(&mut self, ty: u32, instruction: u32, args: &[u32]) -> u32 {
        let mut w = vec![self.glsl, instruction];
        w.extend_from_slice(args);
        self.op(op::EXT_INST, ty, &w)
    }

    pub(super) fn start_block(&mut self, label: u32) {
        self.emit(op::LABEL, &[label]);
        self.fs.label = label;
        self.fs.terminated = false;
    }

    pub(super) fn branch(&mut self, target: u32) {
        self.emit(op::BRANCH, &[target]);
        self.fs.terminated = true;
    }

    pub(super) fn temp_var(&mut self, ty: &Ty) -> u32 {
        let t = self.ty(ty, Flavor::Plain);
        let p = self.ptr(sc::FUNCTION, t);
        let v = self.id();
        inst(&mut self.fs.vars, op::VARIABLE, &[p, v, sc::FUNCTION]);
        v
    }

    // ------------------------------------------------------------------ statements

    fn block(&mut self, stmts: &[Stmt]) -> R<()> {
        for s in stmts {
            if self.fs.terminated {
                break;
            }
            self.stmt(s)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> R<()> {
        match s {
            Stmt::Let(id, e) => {
                let v = self.expr(e)?;
                if !matches!(e.kind, EK::Const(_) | EK::Zero | EK::Value(_) | EK::Current) {
                    let prog = self.prog;
                    if let Some(n) = prog.functions[self.cur_fn].value_names.get(*id as usize) {
                        self.name(v.id, n);
                    }
                }
                self.fs.values.insert(*id, v);
            }
            Stmt::Assign {
                place,
                swizzle,
                value,
            } => self.assign(place, swizzle.as_deref(), value)?,
            Stmt::Eval(e) => {
                self.expr(e)?;
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => self.if_stmt(cond, then, otherwise)?,
            Stmt::Loop {
                cond,
                body,
                continuing,
                break_if,
            } => self.loop_stmt(cond.as_ref(), body, continuing, break_if.as_ref())?,
            Stmt::Break => {
                let l = self
                    .fs
                    .loops
                    .last_mut()
                    .ok_or_else(|| internal(self.span, "break outside loop"))?;
                l.has_break = true;
                let t = l.merge;
                self.branch(t);
            }
            Stmt::Continue => {
                let l = self
                    .fs
                    .loops
                    .last_mut()
                    .ok_or_else(|| internal(self.span, "continue outside loop"))?;
                l.has_continue = true;
                let t = l.cont;
                self.branch(t);
            }
            Stmt::Return(e) => {
                match e {
                    Some(e) => {
                        let v = self.expr(e)?;
                        let v = self.plain_id(v, &e.ty)?;
                        self.emit(op::RETURN_VALUE, &[v]);
                    }
                    None => self.emit(op::RETURN, &[]),
                }
                self.fs.terminated = true;
            }
            Stmt::Discard => {
                self.emit(op::KILL, &[]);
                self.fs.terminated = true;
            }
            Stmt::Block(b) => self.block(b)?,
        }
        Ok(())
    }

    fn if_stmt(&mut self, cond: &Expr, then: &[Stmt], otherwise: &[Stmt]) -> R<()> {
        let c = self.expr(cond)?.id;
        let then_l = self.id();
        let merge = self.id();
        let else_l = if otherwise.is_empty() {
            merge
        } else {
            self.id()
        };
        self.emit(op::SELECTION_MERGE, &[merge, 0]);
        self.emit(op::BRANCH_CONDITIONAL, &[c, then_l, else_l]);
        self.start_block(then_l);
        self.block(then)?;
        let then_falls = !self.fs.terminated;
        if then_falls {
            self.branch(merge);
        }
        let mut else_falls = true;
        if !otherwise.is_empty() {
            self.start_block(else_l);
            self.block(otherwise)?;
            else_falls = !self.fs.terminated;
            if else_falls {
                self.branch(merge);
            }
        }
        self.start_block(merge);
        if !then_falls && !else_falls {
            self.emit(op::UNREACHABLE, &[]);
            self.fs.terminated = true;
        }
        Ok(())
    }

    fn loop_stmt(
        &mut self,
        cond: Option<&Expr>,
        body: &[Stmt],
        continuing: &[Stmt],
        break_if: Option<&Expr>,
    ) -> R<()> {
        let header = self.id();
        let merge = self.id();
        let cont = self.id();
        let body_l = self.id();
        self.branch(header);
        self.start_block(header);
        self.emit(op::LOOP_MERGE, &[merge, cont, 0]);
        if let Some(c) = cond {
            let cond_l = self.id();
            self.branch(cond_l);
            self.start_block(cond_l);
            let v = self.expr(c)?.id;
            self.emit(op::BRANCH_CONDITIONAL, &[v, body_l, merge]);
        } else {
            self.branch(body_l);
        }
        self.start_block(body_l);
        self.fs.loops.push(LoopTargets {
            merge,
            cont,
            has_break: cond.is_some(),
            has_continue: false,
        });
        self.block(body)?;
        let falls = !self.fs.terminated;
        if falls {
            self.branch(cont);
        }
        let cont_reachable = falls || self.fs.loops.last().is_some_and(|l| l.has_continue);
        self.start_block(cont);
        if cont_reachable {
            self.block(continuing)?;
            if !self.fs.terminated {
                match break_if {
                    Some(b) => {
                        let v = self.expr(b)?.id;
                        self.emit(op::BRANCH_CONDITIONAL, &[v, merge, header]);
                        self.fs.terminated = true;
                        if let Some(l) = self.fs.loops.last_mut() {
                            l.has_break = true;
                        }
                    }
                    None => self.branch(header),
                }
            }
        } else {
            self.branch(header);
        }
        let l = self
            .fs
            .loops
            .pop()
            .ok_or_else(|| internal(self.span, "loop stack"))?;
        self.start_block(merge);
        if !l.has_break {
            self.emit(op::UNREACHABLE, &[]);
            self.fs.terminated = true;
        }
        Ok(())
    }
}
