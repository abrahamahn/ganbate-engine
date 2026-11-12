//! Replays a backend-neutral [`Encoder`] into the command stream `web/ganbate.js` executes on
//! WebGL 2: `u32` words, one opcode followed by its operands. Recording checks what GL would
//! otherwise report as errors (formats, usages, attachment/binding feedback loops, missing
//! bindings), resolves each draw's uniform blocks, texture units, storage textures and hidden
//! uniforms against the bound groups, and emits only the GL bindings that changed since the
//! start of the submission. Push constants go to a ring of uniform blocks, shared when equal.
//!
//! Opcodes and operands (floats are their bits):
//! * `BEGIN_PASS` colours, has_depth, then per colour (texture, mip, layer, load, store, r, g,
//!   b, a) and for depth (texture, mip, layer, load, store, clear). load: 0 load, 1 clear,
//!   2 don't care; store: 0 discard, 1 store.
//! * `END_PASS`
//! * `SET_PIPELINE` pipeline: its program and fixed-function state.
//! * `VIEWPORT` x, y, w, h · `SCISSOR` x, y, w, h
//! * `BIND_UBO` point, buffer, offset, size · `PUSH_RANGE` offset, size (at [`PUSH_POINT`])
//! * `BIND_UNIT` unit, texture, sampler (0: the host's nearest sampler), base mip, mip count
//! * `BIND_STORAGE` unit, buffer (its data texture)
//! * `SET_HIDDEN` index, value (a hidden uniform of the current pipeline's program)
//! * `BIND_VAO` vertex array
//! * `DRAW` vertices, instances, first vertex · `DRAW_INDEXED` indices, instances, first index
//! * `COPY_BUFFER` src, src offset, dst, dst offset, size
//! * `COPY_TEXTURE` texture, mip, layer, width, height, buffer, offset
//! * `BLIT` src texture, mip, layer, dst texture, mip, layer, filter (0 nearest, 1 linear)
//! * `COPY_DEPTH` src texture, mip, layer, dst texture, mip, layer, width, height (a depth blit)
use super::ffi::SURFACE;
use super::*;
use crate::gpu::{Command, Encoder, Load, PassView, Span, TransferCmd};

const BEGIN_PASS: u32 = 1;
const END_PASS: u32 = 2;
const SET_PIPELINE: u32 = 3;
const VIEWPORT: u32 = 4;
const SCISSOR: u32 = 5;
const BIND_UBO: u32 = 6;
const PUSH_RANGE: u32 = 7;
const BIND_UNIT: u32 = 8;
const BIND_STORAGE: u32 = 9;
const SET_HIDDEN: u32 = 10;
const BIND_VAO: u32 = 11;
const DRAW: u32 = 12;
const DRAW_INDEXED: u32 = 13;
const COPY_BUFFER: u32 = 14;
const COPY_TEXTURE: u32 = 15;
const BLIT: u32 = 16;
const COPY_DEPTH: u32 = 17;

/// Marks a texture unit holding a storage buffer's data texture in [`Host::units`].
const STORAGE_UNIT: u32 = 1 << 31;

#[inline]
fn hash_bytes(data: &[u8]) -> usize {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h as usize
}

/// Push-constant blocks of one submission, deduplicated without per-block `Vec<u8>` allocations.
pub(super) struct Ring {
    pub bytes: Vec<u8>,
    table: Vec<(u64, u32, u32)>,
    used: usize,
    stamp: u64,
    slot: u32,
}
impl Default for Ring {
    fn default() -> Self {
        Self {
            bytes: Vec::new(),
            table: Vec::new(),
            used: 0,
            stamp: 1,
            slot: PUSH_SLOT,
        }
    }
}
impl Ring {
    fn reset(&mut self, align: u32) {
        self.bytes.clear();
        self.used = 0;
        self.slot = PUSH_SLOT.next_multiple_of(align.max(1));
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.table.fill((0, 0, 0));
            self.stamp = 1;
        }
    }

    #[cold]
    fn grow(&mut self) {
        let new_cap = (self.table.len() * 2).max(64);
        let old = core::mem::replace(&mut self.table, vec![(0, 0, 0); new_cap]);
        let mask = new_cap - 1;
        let stamp = self.stamp;
        for (st, off, len) in old {
            if st != stamp {
                continue;
            }
            let slice = &self.bytes[off as usize..(off + len) as usize];
            let mut pos = hash_bytes(slice) & mask;
            while self.table[pos].0 == stamp {
                pos = (pos + 1) & mask;
            }
            self.table[pos] = (stamp, off, len);
        }
    }

    /// The offset of a block holding `data`, zero-padded to the slot size.
    fn place(&mut self, data: &[u8]) -> u32 {
        if (self.used + 1) * 2 > self.table.len() {
            self.grow();
        }
        let mask = self.table.len() - 1;
        let mut pos = hash_bytes(data) & mask;
        loop {
            let (st, off, len) = self.table[pos];
            if st != self.stamp {
                let offset = self.bytes.len() as u32;
                self.bytes.extend_from_slice(data);
                self.bytes.resize((offset + self.slot) as usize, 0);
                self.table[pos] = (self.stamp, offset, data.len() as u32);
                self.used += 1;
                return offset;
            }
            if len as usize == data.len() && &self.bytes[off as usize..(off + len) as usize] == data
            {
                return off;
            }
            pos = (pos + 1) & mask;
        }
    }
}

/// The GL bindings this submission's stream has made so far, indexed directly by slot.
#[derive(Default)]
struct Host {
    pipeline: Option<u32>,
    /// Uniform block binding point → (buffer, offset, size); the ring is buffer 0.
    points: Vec<Option<(u32, u32, u32)>>,
    /// Texture unit → (texture or `STORAGE_UNIT | buffer`, sampler).
    units: Vec<Option<(u32, u32)>>,
    /// Texture handle → (base mip, mip count) last applied.
    levels: Vec<Option<(u32, u32)>>,
    vao: Option<u32>,
    /// (program, hidden index) → value.
    hidden: Vec<(u32, u32, u32)>,
}
impl Host {
    fn reset(&mut self) {
        self.pipeline = None;
        self.points.fill(None);
        self.units.fill(None);
        self.levels.fill(None);
        self.vao = None;
        self.hidden.clear();
    }
    #[inline]
    fn set_point(&mut self, point: u32, val: (u32, u32, u32)) -> bool {
        let p = point as usize;
        if p >= self.points.len() {
            self.points.resize(p + 1, None);
        }
        if self.points[p] == Some(val) {
            return false;
        }
        self.points[p] = Some(val);
        true
    }
    #[inline]
    fn set_unit(&mut self, unit: u32, val: (u32, u32)) -> bool {
        let u = unit as usize;
        if u >= self.units.len() {
            self.units.resize(u + 1, None);
        }
        if self.units[u] == Some(val) {
            return false;
        }
        self.units[u] = Some(val);
        true
    }
    #[inline]
    fn set_levels(&mut self, tex: u32, val: (u32, u32)) -> bool {
        let t = tex as usize;
        if t >= self.levels.len() {
            self.levels.resize(t + 1, None);
        }
        if self.levels[t] == Some(val) {
            return false;
        }
        self.levels[t] = Some(val);
        true
    }
    #[inline]
    fn set_hidden(&mut self, prog: u32, idx: u32, val: u32) -> bool {
        if let Some(entry) = self
            .hidden
            .iter_mut()
            .find(|(p, i, _)| *p == prog && *i == idx)
        {
            if entry.2 == val {
                return false;
            }
            entry.2 = val;
            return true;
        }
        self.hidden.push((prog, idx, val));
        true
    }
}

/// Bind state of one render pass.
#[derive(Default)]
struct DrawState {
    pipeline: Option<PipelineId>,
    groups: [Option<BindGroupId>; MAX_GROUPS],
    checked: [Option<BindGroupId>; MAX_GROUPS],
    vertex: Option<(BufferId, u64)>,
    index: Option<(BufferId, u64)>,
    push: Option<Span>,
}

/// One draw's parameters.
struct DrawCall {
    indexed: bool,
    base_vertex: i32,
    first_instance: u32,
}

/// Whether two (first, end) ranges overlap.
fn overlap(a: (u32, u32), b: (u32, u32)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

impl Gpu<'_> {
    /// A texture the current submission may use: the canvas texture only in `present`.
    fn usable(&self, id: TextureId, acquired: Option<TextureId>) -> Result<&Texture> {
        let t = self.texture(id)?;
        if t.surface && acquired != Some(id) {
            return Err(format!(
                "{id:?} is the canvas texture; only present() may use it, after acquire()"
            ));
        }
        Ok(t)
    }

    /// Checks that every resource a bind group references is alive.
    fn check_bound(&self, group: &BindGroup) -> Result<()> {
        for (_, b) in &group.entries {
            match *b {
                Bound::Buffer { id, .. } => {
                    self.buffer(id)?;
                }
                Bound::Texture { id, .. } => {
                    self.texture(id)?;
                }
                Bound::Sampler(id) => {
                    self.sampler(id)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn record(
        &self,
        e: &Encoder,
        acquired: Option<TextureId>,
        out: &mut Vec<u32>,
        ring: &mut Ring,
    ) -> Result<()> {
        e.finish()?;
        out.clear();
        out.reserve(e.commands.len() * 6);
        ring.reset(self.info.uniform_align);
        let mut host = Host::default();
        let commands = &e.commands;
        let mut i = 0;
        while i < commands.len() {
            match commands[i] {
                Command::BeginPass {
                    colors,
                    depth,
                    end_cmd,
                } => {
                    let end = if end_cmd != u32::MAX {
                        end_cmd as usize
                    } else {
                        return Err("begin_pass without end_pass".into());
                    };
                    let desc = e.pass(colors, depth);
                    self.record_pass(
                        e,
                        desc,
                        &commands[i + 1..end],
                        out,
                        ring,
                        &mut host,
                        acquired,
                    )?;
                    i = end;
                }
                Command::Transfer(idx) => match &e.transfers[idx as usize] {
                    TransferCmd::Dispatch { .. } => {
                        return Err("dispatch: WebGL 2 has no compute shaders".into());
                    }
                    &TransferCmd::CopyTextureToBuffer {
                        src,
                        mip,
                        layer,
                        dst,
                        offset,
                    } => self.record_texture_copy((src, mip, layer), dst, offset, out, acquired)?,
                    &TransferCmd::CopyBufferToBuffer {
                        src,
                        src_offset,
                        dst,
                        dst_offset,
                        size,
                    } => {
                        self.record_buffer_copy((src, src_offset), (dst, dst_offset), size, out)?
                    }
                    &TransferCmd::Blit { src, dst, filter } => {
                        self.record_blit(src, dst, filter, out, acquired)?;
                        // The host's blit program changes the program, units and vertex array.
                        host.reset();
                    }
                },
                other => return Err(format!("{other:?} recorded outside a pass")),
            }
            i += 1;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_pass(
        &self,
        e: &Encoder,
        desc: PassView<'_>,
        body: &[Command],
        out: &mut Vec<u32>,
        ring: &mut Ring,
        host: &mut Host,
        acquired: Option<TextureId>,
    ) -> Result<()> {
        if desc.colors.is_empty() && desc.depth.is_none() {
            return Err("begin_pass: a pass needs at least one attachment".into());
        }
        if desc.colors.len() > (self.info.max_draw_buffers as usize).min(8) {
            return Err(format!(
                "begin_pass: {} colour attachments exceed the device limit",
                desc.colors.len()
            ));
        }
        let mut size = None;
        let mut attached = [(TextureId::from_raw(0), 0u32, 0u32); 9];
        let mut attached_len = 0usize;
        let mut colors = [Format::Rgba8Unorm; 8];
        let mut colors_len = 0usize;
        let mut depth_format = None;
        let mut words = [0u32; 96];
        words[0] = BEGIN_PASS;
        words[1] = desc.colors.len() as u32;
        words[2] = u32::from(desc.depth.is_some());
        let mut word_len = 3usize;
        let attachments = desc
            .colors
            .iter()
            .map(|a| (a, false))
            .chain(desc.depth.into_iter().map(|a| (a, true)));
        for (a, depth) in attachments {
            let t = self.usable(a.texture, acquired)?;
            let need = if depth {
                TextureUsage::DEPTH_TARGET
            } else {
                TextureUsage::COLOR_TARGET
            };
            if depth != t.desc.format.is_depth() || !t.desc.usage.contains(need) {
                return Err(format!(
                    "begin_pass: {:?} ({:?}, {:?}) cannot be a {} attachment",
                    a.texture,
                    t.desc.format,
                    t.desc.usage,
                    if depth { "depth" } else { "colour" }
                ));
            }
            if a.mip >= t.desc.mips || a.layer >= t.desc.layers {
                return Err(format!(
                    "begin_pass: mip {} layer {} is outside {:?}",
                    a.mip, a.layer, a.texture
                ));
            }
            let extent = t.desc.mip_size(a.mip);
            if *size.get_or_insert(extent) != extent {
                return Err("begin_pass: attachments differ in size".into());
            }
            let sub = (a.texture, a.mip, a.layer);
            if attached[..attached_len].contains(&sub) {
                return Err("begin_pass: the same subresource is attached twice".into());
            }
            attached[attached_len] = sub;
            attached_len += 1;
            let (load, clear) = match a.load {
                Load::Load => (0, [0.0; 4]),
                Load::Clear(v) => (1, v),
                Load::DontCare => (2, [0.0; 4]),
            };
            words[word_len..word_len + 5].copy_from_slice(&[
                t.js,
                a.mip,
                a.layer,
                load,
                u32::from(a.store),
            ]);
            word_len += 5;
            if depth {
                words[word_len] = clear[0].clamp(0.0, 1.0).to_bits();
                word_len += 1;
                depth_format = Some(t.desc.format);
            } else {
                words[word_len..word_len + 4].copy_from_slice(&clear.map(f32::to_bits));
                word_len += 4;
                colors[colors_len] = t.desc.format;
                colors_len += 1;
            }
        }
        out.extend_from_slice(&words[..word_len]);
        let pass_colors = &colors[..colors_len];
        let attached_slice = &attached[..attached_len];
        let (w, h) = size.expect("at least one attachment");
        let mut st = DrawState::default();
        for c in body {
            match *c {
                Command::SetPipeline(id) => {
                    let p = self.pipeline(id)?;
                    if p.colors.as_slice() != pass_colors || p.depth != depth_format {
                        return Err(format!(
                            "set_pipeline: {id:?} renders {:?} + depth {:?}, which does not match \
                             this pass",
                            p.colors, p.depth
                        ));
                    }
                    st.pipeline = Some(id);
                }
                Command::SetBindGroup(g, id) => {
                    let slot = st
                        .groups
                        .get_mut(g as usize)
                        .ok_or_else(|| format!("set_bind_group: group {g} is out of range"))?;
                    *slot = Some(id);
                }
                Command::SetVertexBuffer(id, offset) => {
                    let b = self.buffer(id)?;
                    if !b.desc.usage.contains(BufferUsage::VERTEX) {
                        return Err(format!("set_vertex_buffer: {id:?} lacks VERTEX usage"));
                    }
                    if offset >= b.desc.size || offset % 4 != 0 {
                        return Err(format!(
                            "set_vertex_buffer: offset {offset} must be a multiple of 4 inside the buffer"
                        ));
                    }
                    st.vertex = Some((id, offset));
                }
                Command::SetIndexBuffer(id, offset) => {
                    let b = self.buffer(id)?;
                    if !b.desc.usage.contains(BufferUsage::INDEX) {
                        return Err(format!("set_index_buffer: {id:?} lacks INDEX usage"));
                    }
                    if offset >= b.desc.size || offset % 4 != 0 {
                        return Err(format!(
                            "set_index_buffer: offset {offset} must be a multiple of 4 inside the buffer"
                        ));
                    }
                    st.index = Some((id, offset));
                }
                Command::SetViewport(v) => {
                    out.push(VIEWPORT);
                    out.extend(v.map(f32::to_bits));
                }
                Command::SetScissor([x, y, sw, sh]) => {
                    let (x, y) = (x.min(w), y.min(h));
                    out.extend([SCISSOR, x, y, sw.min(w - x), sh.min(h - y)]);
                }
                Command::PushConstants(span) => st.push = Some(span),
                Command::Draw {
                    vertices,
                    instances,
                    first_vertex,
                    first_instance,
                } => {
                    let call = DrawCall {
                        indexed: false,
                        base_vertex: 0,
                        first_instance,
                    };
                    self.bind_for_draw(e, &mut st, &call, attached_slice, out, ring, host)?;
                    out.extend([DRAW, vertices, instances, first_vertex]);
                }
                Command::DrawIndexed {
                    indices,
                    instances,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    let call = DrawCall {
                        indexed: true,
                        base_vertex,
                        first_instance,
                    };
                    self.bind_for_draw(e, &mut st, &call, attached_slice, out, ring, host)?;
                    let (ib, offset) = st.index.expect("checked by bind_for_draw");
                    let first = offset / 4 + u64::from(first_index);
                    let count = self.buffer(ib)?.desc.size / 4;
                    if first + u64::from(indices) > count {
                        return Err(format!(
                            "draw_indexed: indices {first}..{} are outside {ib:?}",
                            first + u64::from(indices)
                        ));
                    }
                    out.extend([DRAW_INDEXED, indices, instances, first as u32]);
                }
                other => return Err(format!("{other:?} is not allowed inside a pass")),
            }
        }
        out.push(END_PASS);
        Ok(())
    }

    /// The resource bound at `(group, binding)`.
    fn entry(&self, st: &DrawState, (group, binding): (u32, u32)) -> Result<Bound> {
        let gid = st
            .groups
            .get(group as usize)
            .copied()
            .flatten()
            .ok_or_else(|| format!("the pipeline uses @group({group}) but no bind group is set"))?;
        self.group(gid)?
            .entries
            .iter()
            .find(|(b, _)| *b == binding)
            .map(|(_, r)| *r)
            .ok_or_else(|| format!("{gid:?} has no @binding({binding})"))
    }

    /// Emits what the current pipeline needs for one draw: the pipeline, uniform blocks, push
    /// constants, texture units, storage textures, hidden uniforms and vertex array.
    #[allow(clippy::too_many_arguments)]
    fn bind_for_draw(
        &self,
        e: &Encoder,
        st: &mut DrawState,
        call: &DrawCall,
        attached: &[(TextureId, u32, u32)],
        out: &mut Vec<u32>,
        ring: &mut Ring,
        host: &mut Host,
    ) -> Result<()> {
        let id = st.pipeline.ok_or("draw without set_pipeline")?;
        let p = self.pipeline(id)?;
        if p.vertex_input && st.vertex.is_none() {
            return Err(format!("{id:?} reads vertices but no vertex buffer is set"));
        }
        if call.indexed && st.index.is_none() {
            return Err("draw_indexed without set_index_buffer".into());
        }
        for (slot, &expected_id) in p.group_ids.iter().enumerate() {
            if expected_id == 0 {
                continue;
            }
            let gid = st.groups[slot]
                .ok_or_else(|| format!("{id:?} uses @group({slot}) but no bind group is set"))?;
            let g = self.group(gid)?;
            if g.layout_id != expected_id {
                return Err(format!(
                    "{gid:?} was created for a different layout than @group({slot}) of {id:?}"
                ));
            }
            if st.checked[slot] != Some(gid) {
                self.check_bound(g)?;
                st.checked[slot] = Some(gid);
            }
        }
        if host.pipeline != Some(p.js) {
            out.extend([SET_PIPELINE, p.js]);
            host.pipeline = Some(p.js);
        }
        let program = &self.programs[p.program];
        for b in &program.blocks {
            let (buffer, offset, size) = match b.binding {
                None => {
                    let span = st.push.ok_or_else(|| {
                        format!(
                            "{id:?} uses {} bytes of push constants but none are set",
                            p.push_size
                        )
                    })?;
                    if span.len() < p.push_size as usize {
                        return Err(format!(
                            "{id:?} uses {} bytes of push constants but only {} are set",
                            p.push_size,
                            span.len()
                        ));
                    }
                    let start = span.start as usize;
                    let data = &e.bytes[start..start + p.push_size as usize];
                    (0, ring.place(data), b.size.next_multiple_of(16))
                }
                Some(at) => match self.entry(st, at)? {
                    Bound::Buffer { id, offset, size } => {
                        (self.buffer(id)?.js, offset as u32, size as u32)
                    }
                    other => return Err(format!("{at:?} holds {other:?}, not a uniform buffer")),
                },
            };
            if host.set_point(b.point, (buffer, offset, size)) {
                if b.binding.is_none() {
                    out.extend([PUSH_RANGE, offset, size]);
                } else {
                    out.extend([BIND_UBO, b.point, buffer, offset, size]);
                }
            }
        }
        let mut viewed = [(0u32, (0u32, 0u32)); 64];
        let mut viewed_len = 0usize;
        for u in &program.units {
            let Bound::Texture {
                id: tid,
                mips,
                layers,
            } = self.entry(st, u.texture)?
            else {
                return Err(format!("{:?} does not hold a texture", u.texture));
            };
            if attached.iter().any(|&(t, m, l)| {
                t == tid && overlap(mips, (m, m + 1)) && overlap(layers, (l, l + 1))
            }) {
                return Err(format!(
                    "{tid:?} is used as an attachment and as a bound texture in one pass"
                ));
            }
            let t = self.texture(tid)?;
            if u.view == TextureView::D2Array && !t.array {
                return Err(format!("{tid:?} is not an array texture"));
            }
            if let Some(&(_, m)) = viewed[..viewed_len].iter().find(|(js, _)| *js == t.js)
                && m != mips
            {
                return Err(format!(
                    "{tid:?} is viewed with two mip ranges in one draw, which WebGL 2 cannot do"
                ));
            }
            if viewed_len < viewed.len() {
                viewed[viewed_len] = (t.js, mips);
                viewed_len += 1;
            }
            let sampler = match u.sampler {
                Some(at) => match self.entry(st, at)? {
                    Bound::Sampler(s) => self.sampler(s)?.js,
                    other => return Err(format!("{at:?} holds {other:?}, not a sampler")),
                },
                None => 0,
            };
            let levels = (mips.0, mips.1 - mips.0);
            let unit_changed = host.set_unit(u.unit, (t.js, sampler));
            let levels_changed = host.set_levels(t.js, levels);
            if unit_changed || levels_changed {
                out.extend([BIND_UNIT, u.unit, t.js, sampler, levels.0, levels.1]);
            }
        }
        for s in &program.storage {
            let Bound::Buffer { id: bid, .. } = self.entry(st, s.binding)? else {
                return Err(format!("{:?} does not hold a storage buffer", s.binding));
            };
            let js = self.buffer(bid)?.js;
            if host.set_unit(s.unit, (STORAGE_UNIT | js, 0)) {
                out.extend([BIND_STORAGE, s.unit, js]);
            }
        }
        for (index, h) in program.hidden.iter().enumerate() {
            let value = match *h {
                HiddenValue::BaseVertex => call.base_vertex as u32,
                HiddenValue::BaseInstance => call.first_instance,
                HiddenValue::StorageOffset(g, b) => match self.entry(st, (g, b))? {
                    Bound::Buffer { offset, .. } => offset as u32,
                    _ => return Err(format!("({g}, {b}) does not hold a buffer")),
                },
                HiddenValue::StorageSize(g, b) => match self.entry(st, (g, b))? {
                    Bound::Buffer { size, .. } => size as u32,
                    _ => return Err(format!("({g}, {b}) does not hold a buffer")),
                },
                HiddenValue::TextureLevels(g, b) => match self.entry(st, (g, b))? {
                    Bound::Texture { mips, .. } => mips.1 - mips.0,
                    _ => return Err(format!("({g}, {b}) does not hold a texture")),
                },
            };
            if host.set_hidden(program.js, index as u32, value) {
                out.extend([SET_HIDDEN, index as u32, value]);
            }
        }
        // The vertex array: the pipeline's layout over the vertex buffer, shifted by the base
        // vertex (WebGL 2 has no base-vertex draws), with the index buffer.
        let index = match (call.indexed, st.index) {
            (true, Some((ib, _))) => self.buffer(ib)?.js,
            _ => 0,
        };
        let (buffer, offset) = match (p.vertex_input, st.vertex) {
            (true, Some((vb, offset))) => {
                let shifted = offset as i64 + i64::from(call.base_vertex) * i64::from(p.stride);
                if shifted < 0 || shifted >= self.buffer(vb)?.desc.size as i64 {
                    return Err(format!(
                        "base vertex {} moves the vertices outside {vb:?}",
                        call.base_vertex
                    ));
                }
                (self.buffer(vb)?.js, shifted as u32)
            }
            _ => (0, 0),
        };
        let key = (p.layout, buffer, offset, index);
        let cached = self.vaos.borrow().get(&key).copied();
        let vao = match cached {
            Some(v) => v,
            None => {
                let v = ffi::create_vao(p.js, buffer, offset, index)?;
                self.vaos.borrow_mut().insert(key, v);
                v
            }
        };
        if host.vao != Some(vao) {
            out.extend([BIND_VAO, vao]);
            host.vao = Some(vao);
        }
        Ok(())
    }

    fn record_texture_copy(
        &self,
        (src, mip, layer): (TextureId, u32, u32),
        dst: BufferId,
        offset: u64,
        out: &mut Vec<u32>,
        acquired: Option<TextureId>,
    ) -> Result<()> {
        let t = self.usable(src, acquired)?;
        let b = self.buffer(dst)?;
        if !t.desc.usage.contains(TextureUsage::COPY_SRC) {
            return Err(format!(
                "copy_texture_to_buffer: {src:?} lacks COPY_SRC usage"
            ));
        }
        if !b.desc.usage.contains(BufferUsage::COPY_DST) {
            return Err(format!(
                "copy_texture_to_buffer: {dst:?} lacks COPY_DST usage"
            ));
        }
        if b.desc.memory != Memory::Readback {
            return Err(
                "copy_texture_to_buffer: WebGL 2 copies textures into readback buffers".into(),
            );
        }
        if t.desc.format.is_depth() {
            return Err("copy_texture_to_buffer: WebGL 2 cannot read depth textures back".into());
        }
        if mip >= t.desc.mips || layer >= t.desc.layers {
            return Err(format!(
                "copy_texture_to_buffer: mip {mip} layer {layer} is outside {src:?}"
            ));
        }
        let (w, h) = t.desc.mip_size(mip);
        let bytes = u64::from(w * t.desc.format.texel_size()) * u64::from(h);
        if offset % 4 != 0
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > b.desc.size)
        {
            return Err(format!(
                "copy_texture_to_buffer: {bytes} bytes at offset {offset} do not fit {dst:?}"
            ));
        }
        out.extend([COPY_TEXTURE, t.js, mip, layer, w, h, b.js, offset as u32]);
        Ok(())
    }

    fn record_buffer_copy(
        &self,
        (src, src_offset): (BufferId, u64),
        (dst, dst_offset): (BufferId, u64),
        size: u64,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        let s = self.buffer(src)?;
        let d = self.buffer(dst)?;
        if !s.desc.usage.contains(BufferUsage::COPY_SRC) || s.desc.memory == Memory::Readback {
            return Err(format!(
                "copy_buffer_to_buffer: {src:?} lacks COPY_SRC usage"
            ));
        }
        if !d.desc.usage.contains(BufferUsage::COPY_DST) {
            return Err(format!(
                "copy_buffer_to_buffer: {dst:?} lacks COPY_DST usage"
            ));
        }
        if s.index != d.index && d.desc.memory != Memory::Readback {
            return Err(
                "copy_buffer_to_buffer: WebGL 2 copies index buffers only into index buffers"
                    .into(),
            );
        }
        let fits = |offset: u64, len: u64| offset.checked_add(size).is_some_and(|e| e <= len);
        if !fits(src_offset, s.desc.size) || !fits(dst_offset, d.desc.size) {
            return Err("copy_buffer_to_buffer: range out of bounds".into());
        }
        if src == dst && src_offset < dst_offset + size && dst_offset < src_offset + size {
            return Err("copy_buffer_to_buffer: overlapping ranges in one buffer".into());
        }
        if size == 0 {
            return Ok(());
        }
        out.extend([
            COPY_BUFFER,
            s.js,
            src_offset as u32,
            d.js,
            dst_offset as u32,
            size as u32,
        ]);
        Ok(())
    }

    fn record_blit(
        &self,
        (src, src_mip, src_layer): (TextureId, u32, u32),
        (dst, dst_mip, dst_layer): (TextureId, u32, u32),
        filter: Filter,
        out: &mut Vec<u32>,
        acquired: Option<TextureId>,
    ) -> Result<()> {
        let s = self.usable(src, acquired)?;
        let d = self.usable(dst, acquired)?;
        if !s.desc.usage.contains(TextureUsage::COPY_SRC)
            || !d.desc.usage.contains(TextureUsage::COPY_DST) && !d.surface
        {
            return Err("blit: the source needs COPY_SRC and the destination COPY_DST".into());
        }
        if src_mip >= s.desc.mips
            || src_layer >= s.desc.layers
            || dst_mip >= d.desc.mips
            || dst_layer >= d.desc.layers
        {
            return Err("blit: subresource out of range".into());
        }
        if (src, src_mip, src_layer) == (dst, dst_mip, dst_layer) {
            return Err("blit: source and destination are the same subresource".into());
        }
        if src == dst && src_mip == dst_mip {
            return Err("blit: WebGL 2 cannot blit between layers of one mip".into());
        }
        if s.desc.format.is_depth() || d.desc.format.is_depth() {
            // A depth blit is a whole-subresource copy (blitFramebuffer's depth needs a
            // same-format, same-size, Nearest copy).
            let size = s.desc.mip_size(src_mip);
            if s.desc.format != d.desc.format
                || size != d.desc.mip_size(dst_mip)
                || filter != Filter::Nearest
            {
                return Err(
                    "blit: depth blits need identical formats and sizes and Nearest filtering"
                        .into(),
                );
            }
            out.extend([
                COPY_DEPTH, s.js, src_mip, src_layer, d.js, dst_mip, dst_layer, size.0, size.1,
            ]);
            return Ok(());
        }
        if filter == Filter::Linear && is_float32(s.desc.format) && !self.info.float_linear {
            return Err(format!(
                "blit: {:?} cannot be filtered linearly on this device",
                s.desc.format
            ));
        }
        let dst_js = if d.surface { SURFACE } else { d.js };
        let src_js = if s.surface { SURFACE } else { s.js };
        out.extend([
            BLIT,
            src_js,
            src_mip,
            src_layer,
            dst_js,
            dst_mip,
            dst_layer,
            u32::from(filter == Filter::Linear),
        ]);
        Ok(())
    }
}
