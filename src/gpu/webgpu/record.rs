//! Replays a backend-neutral [`Encoder`] into the command stream `web/ganbate.js` executes: `u32`
//! words, one opcode followed by its operands. Recording checks what the browser would
//! otherwise report asynchronously (formats, usages, layouts, attachment and binding
//! conflicts), resolves bind group state at each draw (as the Vulkan backend does) and places
//! push constants in the ring of [`PUSH_SLOT`]-byte uniform blocks, sharing identical blocks.
//!
//! Opcodes and operands (u64 values are two words, low first; floats are their bits):
//! * `BEGIN_RENDER` colours, has_depth, then per colour (texture, mip, layer, load, store, r, g,
//!   b, a) and for depth (texture, mip, layer, load, store, clear). load: 0 load, 1 clear;
//!   store: 0 discard, 1 store.
//! * `END_RENDER`
//! * `SET_PIPELINE` pipeline
//! * `SET_BIND_GROUP` index, group, offsets, offsets...
//! * `SET_VERTEX` slot, buffer, offset(u64) · `SET_INDEX` buffer, offset(u64) (uint32 indices)
//! * `VIEWPORT` x, y, w, h · `SCISSOR` x, y, w, h
//! * `DRAW` vertices, instances, first vertex, first instance
//! * `DRAW_INDEXED` indices, instances, first index, base vertex (i32), first instance
//! * `DISPATCH` pipeline, groups, per group (index, group, offsets, offsets...), x, y, z
//! * `COPY_BUFFER` src, src offset(u64), dst, dst offset(u64), size(u64)
//! * `COPY_TEXTURE` texture, mip, layer, width, height, buffer, offset(u64), bytes per row
//! * `BLIT` src texture, mip, layer, dst texture, mip, layer, filter (0 nearest, 1 linear)
//! * `COPY_DEPTH` src texture, mip, layer, dst texture, mip, layer, width, height (a depth blit)
use super::ffi::{EMPTY_GROUP, PUSH_COMPUTE, PUSH_RENDER, SURFACE};
use super::*;
use crate::gpu::{Command, Encoder, Load, PassView, Span, TransferCmd};

const BEGIN_RENDER: u32 = 1;
const END_RENDER: u32 = 2;
const SET_PIPELINE: u32 = 3;
const SET_BIND_GROUP: u32 = 4;
const SET_VERTEX: u32 = 5;
const SET_INDEX: u32 = 6;
const VIEWPORT: u32 = 7;
const SCISSOR: u32 = 8;
const DRAW: u32 = 9;
const DRAW_INDEXED: u32 = 10;
const DISPATCH: u32 = 11;
const COPY_BUFFER: u32 = 12;
const COPY_TEXTURE: u32 = 13;
const BLIT: u32 = 14;
const COPY_DEPTH: u32 = 15;

fn u64_words(v: u64) -> [u32; 2] {
    [v as u32, (v >> 32) as u32]
}

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

/// Bind state of one render pass.
#[derive(Default)]
struct DrawState {
    pipeline: Option<PipelineId>,
    /// What the encoder set, per group.
    groups: [Option<BindGroupId>; MAX_GROUPS],
    /// What the host has bound, per group: (handle, dynamic offset).
    bound: [Option<(u32, Option<u32>)>; MAX_GROUPS],
    checked: [Option<BindGroupId>; MAX_GROUPS],
    vertex: Option<(u32, u64)>,
    index: Option<(u32, u64)>,
    push: Option<Span>,
}

fn bind(out: &mut Vec<u32>, st: &mut DrawState, slot: usize, js: u32, offset: Option<u32>) {
    if st.bound[slot] != Some((js, offset)) {
        out.extend([SET_BIND_GROUP, slot as u32, js]);
        match offset {
            Some(o) => out.extend([1, o]),
            None => out.push(0),
        }
        st.bound[slot] = Some((js, offset));
    }
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
        for b in &group.bound {
            match *b {
                Bound::Buffer(id) => {
                    self.buffer(id)?;
                }
                Bound::Texture { id, .. } => {
                    self.texture(id)?;
                }
                Bound::Sampler(id) => {
                    self.samplers
                        .get(id.raw())
                        .ok_or_else(|| format!("{id:?} does not belong to this device"))?;
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
                    self.record_pass(e, desc, &commands[i + 1..end], out, ring, acquired)?;
                    i = end;
                }
                Command::Transfer(idx) => match e.transfers[idx as usize] {
                    TransferCmd::Dispatch {
                        pipeline,
                        groups,
                        push,
                        size,
                    } => self.record_dispatch(
                        pipeline,
                        &e.groups[groups.range()],
                        &e.bytes[push.range()],
                        size,
                        out,
                        ring,
                    )?,
                    TransferCmd::CopyTextureToBuffer {
                        src,
                        mip,
                        layer,
                        dst,
                        offset,
                    } => self.record_texture_copy((src, mip, layer), dst, offset, out, acquired)?,
                    TransferCmd::CopyBufferToBuffer {
                        src,
                        src_offset,
                        dst,
                        dst_offset,
                        size,
                    } => {
                        self.record_buffer_copy((src, src_offset), (dst, dst_offset), size, out)?
                    }
                    TransferCmd::Blit { src, dst, filter } => {
                        self.record_blit(src, dst, filter, out, acquired)?
                    }
                },
                other => return Err(format!("{other:?} recorded outside a pass")),
            }
            i += 1;
        }
        Ok(())
    }

    fn record_pass(
        &self,
        e: &Encoder,
        desc: PassView<'_>,
        body: &[Command],
        out: &mut Vec<u32>,
        ring: &mut Ring,
        acquired: Option<TextureId>,
    ) -> Result<()> {
        if desc.colors.is_empty() && desc.depth.is_none() {
            return Err("begin_pass: a pass needs at least one attachment".into());
        }
        if desc.colors.len() > (self.info.max_color_attachments as usize).min(8) {
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
        words[0] = BEGIN_RENDER;
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
                Load::DontCare => (1, [0.0; 4]),
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
        // A subresource cannot be an attachment and bound to a shader in the same pass.
        let mut last_group = None;
        for c in body {
            if let Command::SetBindGroup(_, g) = *c {
                let grp = self.group(g)?;
                if last_group == Some(g) {
                    continue;
                }
                last_group = Some(g);
                for b in &grp.bound {
                    if let Bound::Texture { id, mips, layers } = *b
                        && attached[..attached_len].iter().any(|&(t, m, l)| {
                            t == id && overlap(mips, (m, m + 1)) && overlap(layers, (l, l + 1))
                        })
                    {
                        return Err(format!(
                            "{id:?} is used as an attachment and as a bound texture in one pass"
                        ));
                    }
                }
            }
        }
        out.extend_from_slice(&words[..word_len]);
        let pass_colors = &colors[..colors_len];
        let (w, h) = size.expect("at least one attachment");
        let mut st = DrawState::default();
        for c in body {
            match *c {
                Command::SetPipeline(id) => {
                    let p = self.pipeline(id)?;
                    if p.compute || p.colors.as_slice() != pass_colors || p.depth != depth_format {
                        return Err(format!(
                            "set_pipeline: {id:?} renders {:?} + depth {:?}, which does not match \
                             this pass",
                            p.colors, p.depth
                        ));
                    }
                    if st.pipeline != Some(id) {
                        out.extend([SET_PIPELINE, p.js]);
                        st.pipeline = Some(id);
                    }
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
                    if st.vertex != Some((b.js, offset)) {
                        out.extend([SET_VERTEX, 0, b.js]);
                        out.extend(u64_words(offset));
                        st.vertex = Some((b.js, offset));
                    }
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
                    if st.index != Some((b.js, offset)) {
                        out.extend([SET_INDEX, b.js]);
                        out.extend(u64_words(offset));
                        st.index = Some((b.js, offset));
                    }
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
                    self.bind_for_draw(e, &mut st, false, out, ring)?;
                    out.extend([DRAW, vertices, instances, first_vertex, first_instance]);
                }
                Command::DrawIndexed {
                    indices,
                    instances,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    self.bind_for_draw(e, &mut st, true, out, ring)?;
                    out.extend([
                        DRAW_INDEXED,
                        indices,
                        instances,
                        first_index,
                        base_vertex as u32,
                        first_instance,
                    ]);
                }
                other => return Err(format!("{other:?} is not allowed inside a pass")),
            }
        }
        out.push(END_RENDER);
        Ok(())
    }

    /// Binds what the current pipeline needs at every group: the encoder's bind groups, the
    /// host's empty group at unused slots, and the push constants' ring block.
    fn bind_for_draw(
        &self,
        e: &Encoder,
        st: &mut DrawState,
        indexed: bool,
        out: &mut Vec<u32>,
        ring: &mut Ring,
    ) -> Result<()> {
        let id = st.pipeline.ok_or("draw without set_pipeline")?;
        let p = self.pipeline(id)?;
        if p.vertex_input && st.vertex.is_none() {
            return Err(format!("{id:?} reads vertices but no vertex buffer is set"));
        }
        if indexed && st.index.is_none() {
            return Err("draw_indexed without set_index_buffer".into());
        }
        for (slot, s) in p.slots.iter().enumerate() {
            match s {
                Slot::Empty => bind(out, st, slot, EMPTY_GROUP, None),
                Slot::Group(layout) => {
                    let gid = st.groups[slot].ok_or_else(|| {
                        format!("{id:?} uses @group({slot}) but no bind group is set")
                    })?;
                    let g = self.group(gid)?;
                    if g.layout_js != Some(layout.js) {
                        return Err(format!(
                            "{gid:?} was created for a different layout than @group({slot}) of {id:?}"
                        ));
                    }
                    if st.checked[slot] != Some(gid) {
                        self.check_bound(g)?;
                        st.checked[slot] = Some(gid);
                    }
                    bind(out, st, slot, g.js, None);
                }
                Slot::Push => {
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
                    let offset = ring.place(data);
                    bind(out, st, slot, PUSH_RENDER, Some(offset));
                }
            }
        }
        Ok(())
    }

    fn record_dispatch(
        &self,
        id: PipelineId,
        groups: &[(u32, BindGroupId)],
        push: &[u8],
        size: [u32; 3],
        out: &mut Vec<u32>,
        ring: &mut Ring,
    ) -> Result<()> {
        let p = self.pipeline(id)?;
        if !p.compute {
            return Err(format!("dispatch: {id:?} is not a compute pipeline"));
        }
        let max = self.info.max_workgroups;
        if size.iter().any(|n| *n > max) {
            return Err(format!("dispatch: {size:?} exceeds the device limit {max}"));
        }
        if push.len() < p.push_size as usize {
            return Err(format!(
                "dispatch: {id:?} uses {} bytes of push constants but {} were given",
                p.push_size,
                push.len()
            ));
        }
        let mut chosen = [None; MAX_GROUPS];
        for (i, &(g, group)) in groups.iter().enumerate() {
            if groups[..i].iter().any(|(other, _)| *other == g) {
                return Err(format!("dispatch: @group({g}) is given twice"));
            }
            if let Some(slot) = chosen.get_mut(g as usize) {
                *slot = Some(group);
            }
        }
        let mut words = [0u32; 32];
        words[0] = DISPATCH;
        words[1] = p.js;
        words[2] = p.slots.len() as u32;
        let mut wlen = 3usize;
        for (slot, s) in p.slots.iter().enumerate() {
            match s {
                Slot::Empty => {
                    words[wlen..wlen + 3].copy_from_slice(&[slot as u32, EMPTY_GROUP, 0]);
                    wlen += 3;
                }
                Slot::Group(layout) => {
                    let gid = chosen[slot]
                        .ok_or_else(|| format!("dispatch: {id:?} needs @group({slot})"))?;
                    let g = self.group(gid)?;
                    if g.layout_js != Some(layout.js) {
                        return Err(format!(
                            "{gid:?} was created for a different layout than @group({slot}) of {id:?}"
                        ));
                    }
                    self.check_bound(g)?;
                    words[wlen..wlen + 3].copy_from_slice(&[slot as u32, g.js, 0]);
                    wlen += 3;
                }
                Slot::Push => {
                    let offset = ring.place(&push[..p.push_size as usize]);
                    words[wlen..wlen + 4].copy_from_slice(&[slot as u32, PUSH_COMPUTE, 1, offset]);
                    wlen += 4;
                }
            }
        }
        words[wlen..wlen + 3].copy_from_slice(&size);
        wlen += 3;
        out.extend_from_slice(&words[..wlen]);
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
        if mip >= t.desc.mips || layer >= t.desc.layers {
            return Err(format!(
                "copy_texture_to_buffer: mip {mip} layer {layer} is outside {src:?}"
            ));
        }
        let (w, h) = t.desc.mip_size(mip);
        let texel = t.desc.format.texel_size();
        let row = w * texel;
        if row % 256 != 0 {
            return Err(format!(
                "copy_texture_to_buffer: WebGPU copies rows of a multiple of 256 bytes, and \
                 {w} texels of {:?} are {row} bytes",
                t.desc.format
            ));
        }
        let bytes = u64::from(row) * u64::from(h);
        if offset % u64::from(texel.max(4)) != 0
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > b.desc.size)
        {
            return Err(format!(
                "copy_texture_to_buffer: {bytes} bytes at offset {offset} do not fit {dst:?}"
            ));
        }
        out.extend([COPY_TEXTURE, t.js, mip, layer, w, h, b.js]);
        out.extend(u64_words(offset));
        out.push(row);
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
        let fits = |offset: u64, len: u64| offset.checked_add(size).is_some_and(|e| e <= len);
        if !fits(src_offset, s.desc.size) || !fits(dst_offset, d.desc.size) {
            return Err("copy_buffer_to_buffer: range out of bounds".into());
        }
        if src == dst && src_offset < dst_offset + size && dst_offset < src_offset + size {
            return Err("copy_buffer_to_buffer: overlapping ranges in one buffer".into());
        }
        if src_offset % 4 != 0 || dst_offset % 4 != 0 || size % 4 != 0 {
            return Err("copy_buffer_to_buffer: WebGPU copies whole 4-byte words".into());
        }
        if size == 0 {
            return Ok(());
        }
        out.extend([COPY_BUFFER, s.js]);
        out.extend(u64_words(src_offset));
        out.push(d.js);
        out.extend(u64_words(dst_offset));
        out.extend(u64_words(size));
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
        if s.surface {
            return Err("blit: the canvas texture cannot be a blit source on WebGPU".into());
        }
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
        if s.desc.format.is_depth() || d.desc.format.is_depth() {
            // WebGPU neither scales nor filters depth: a depth blit is a whole-subresource copy.
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
        if filter == Filter::Linear && is_float32(s.desc.format) && !self.info.float32_filterable {
            return Err(format!(
                "blit: {:?} cannot be filtered linearly on this device",
                s.desc.format
            ));
        }
        let dst_js = if d.surface { SURFACE } else { d.js };
        out.extend([
            BLIT,
            s.js,
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
