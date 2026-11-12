//! Vulkan 1.0 implementation of the backend-neutral GPU layer in [`crate::gpu`].
//!
//! Ownership: [`Gpu`] owns the instance, device, queue and every object created through it.
//! Objects live in generational pools addressed by the public `*Id` handles. Destroying an
//! object moves its Vulkan handles to a garbage list tagged with the newest submission serial;
//! they are freed once every submission up to that serial has signalled its fence (or when the
//! device drops, after `vkDeviceWaitIdle`).
//!
//! Submissions: one graphics+compute queue, one command pool, and three command buffers with
//! fences: one for synchronous work (`submit`, uploads, mip generation) and one per frame slot
//! for presentation (two frames in flight). A fence is only reused after it has been waited on,
//! so every submission that is not pending on a fence has completed.
//!
//! Threading: the device is single-threaded (`!Send`, `!Sync`). Every Vulkan object is
//! externally synchronized by `&mut self`; fence waits in `&self` methods only read fences.
//! The validation callback may run on driver threads and only touches the mutex-protected log.
mod pipelines;
mod record;
mod resources;
mod setup;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod swapchain;
mod track;

use super::ffi::*;
use super::{Handle, check, create};
use crate::gpu::{
    BindGroupId, BufferDesc, BufferId, Format, Limits, PipelineId, SamplerId, TextureDesc,
    TextureId, ValidationLog,
};
use core::cell::Cell;
use core::ptr::{null, null_mut};
use ganbate_shader::BindingKind;
use std::collections::HashMap;
use track::{State, Use};

type Result<T> = std::result::Result<T, String>;

/// Frames in flight for presentation.
const FRAMES: usize = 2;
/// Double-buffered asynchronous transfer command buffers for staged uploads & mip generation.
const XFER_SLOTS: usize = 2;
/// Index of the synchronous submission in `Gpu::subs`; frame slot `f` uses `1 + f`.
const IMMEDIATE: usize = 0;
const XFER_BASE: usize = 1 + FRAMES;
const SUB_COUNT: usize = 1 + FRAMES + XFER_SLOTS;
/// Maximum colour attachments per pass.
const MAX_PASS_COLORS: usize = 8;
/// Longest wait for one GPU submission before reporting a hang.
/// How long a submission may take before it is reported as lost (process-wide).
static FENCE_TIMEOUT_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(20_000_000_000);
pub(crate) fn set_fence_timeout(timeout: std::time::Duration) {
    let ns = u64::try_from(timeout.as_nanos())
        .unwrap_or(u64::MAX)
        .max(1_000_000);
    FENCE_TIMEOUT_NS.store(ns, std::sync::atomic::Ordering::Relaxed);
}

const INDEX_BITS: u32 = 20;
const INDEX_MASK: u32 = (1 << INDEX_BITS) - 1;
const GENERATIONS: u32 = 1 << (32 - INDEX_BITS);

/// Generational slots: ids are `generation << 20 | index`.
struct Pool<T> {
    slots: Vec<(u32, Option<T>)>,
    free: Vec<u32>,
}
impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }
}
impl<T> Pool<T> {
    fn insert(&mut self, value: T) -> Result<u32> {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.1 = Some(value);
            return Ok(slot.0 << INDEX_BITS | index);
        }
        let index = self.slots.len() as u32;
        if index > INDEX_MASK {
            return Err("too many live GPU objects of one kind".into());
        }
        self.slots.push((0, Some(value)));
        Ok(index)
    }
    fn get(&self, raw: u32) -> Option<&T> {
        let slot = self.slots.get((raw & INDEX_MASK) as usize)?;
        (slot.0 == raw >> INDEX_BITS).then_some(slot.1.as_ref()?)
    }
    fn get_mut(&mut self, raw: u32) -> Option<&mut T> {
        let slot = self.slots.get_mut((raw & INDEX_MASK) as usize)?;
        (slot.0 == raw >> INDEX_BITS).then_some(slot.1.as_mut()?)
    }
    fn remove(&mut self, raw: u32) -> Option<T> {
        let index = raw & INDEX_MASK;
        let slot = self.slots.get_mut(index as usize)?;
        if slot.0 != raw >> INDEX_BITS {
            return None;
        }
        let value = slot.1.take()?;
        slot.0 = (slot.0 + 1) % GENERATIONS;
        self.free.push(index);
        Some(value)
    }
    fn drain(&mut self) -> Vec<T> {
        self.free.clear();
        self.slots.drain(..).filter_map(|s| s.1).collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Alloc {
    pub memory: u64,
    pub offset: u64,
    pub size: u64,
    /// `u32::MAX` for dedicated `VkDeviceMemory` blocks; otherwise index in `Gpu::slabs`.
    pub slab: u32,
}
impl Alloc {
    pub const NONE: Self = Self {
        memory: 0,
        offset: 0,
        size: 0,
        slab: u32::MAX,
    };
}

struct Slab {
    memory: u64,
    memory_type: u32,
    optimal_image: bool,
    mapped: *mut u8,
    free: Vec<(u64, u64)>,
}

struct Buffer {
    handle: u64,
    alloc: Alloc,
    desc: BufferDesc,
    /// Persistent mapping of host-visible memory; null for `Memory::Device`.
    mapped: *mut u8,
    state: State,
    /// Serial of the newest submission that referenced the buffer.
    last_use: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ViewKey {
    ty: i32,
    base_mip: u32,
    mips: u32,
    base_layer: u32,
    layers: u32,
}

struct Texture {
    image: u64,
    /// `Alloc::NONE` for swapchain images, which the swapchain owns.
    alloc: Alloc,
    desc: TextureDesc,
    format: i32,
    aspect: u32,
    /// Software block-decompression fallback when hardware lacks BC/ASTC support.
    sw_decompress: Option<ganbate_core::world::TextureCompression>,
    /// Per subresource, indexed `layer * mips + mip`.
    states: Vec<State>,
    views: Vec<(ViewKey, u64)>,
    swapchain: bool,
    last_use: u64,
}

struct Sampler {
    handle: u64,
    comparison: bool,
}

/// One binding of a descriptor set layout, as reflected from the shaders.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Entry {
    binding: u32,
    kind: BindingKind,
    /// VkShaderStageFlags of the stages that use the binding.
    stages: u32,
}
#[derive(Clone, Debug, PartialEq)]
struct GroupLayout {
    id: u32,
    set_layout: u64,
    entries: Vec<Entry>,
}

struct Pipeline {
    handle: u64,
    layout: u64,
    compute: bool,
    /// Index = group; empty layouts fill gaps below the highest used group.
    groups: Vec<GroupLayout>,
    push_size: u32,
    push_stages: u32,
    colors: Vec<Format>,
    depth: Option<Format>,
    vertex_input: bool,
}

/// A resource referenced by a bind group, with the use it implies.
#[derive(Clone, Copy, Debug)]
enum Bound {
    Buffer {
        id: BufferId,
        usage: Use,
    },
    Texture {
        id: TextureId,
        mips: (u32, u32),
        layers: (u32, u32),
        usage: Use,
    },
    Sampler(SamplerId),
}

/// Bind groups per pipeline (`@group(0)` to `@group(7)`).
const MAX_BIND_GROUPS: usize = 8;
/// VkDescriptorType values used by bind groups, in pool accounting order.
const DESCRIPTOR_TYPES: [i32; 5] = [0, 2, 3, 6, 7];

struct BindGroup {
    set: u64,
    pool: usize,
    layout_id: u32,
    bound: Vec<Bound>,
    counts: [u32; 5],
}

struct DescriptorPool {
    handle: u64,
    sets: u32,
    counts: [u32; 5],
}

enum Garbage {
    Buffer(u64, Alloc),
    Image(u64, Alloc, Vec<u64>),
    Framebuffer(u64),
    Set(usize, u64, [u32; 5]),
}

struct Submission {
    cmd: Handle,
    fence: u64,
    /// Serial of the submission the fence will signal, until a wait observes it.
    pending: Cell<Option<u64>>,
    /// Acquire semaphore of a frame slot (unused for the synchronous submission).
    acquire: u64,
}
impl Default for Submission {
    fn default() -> Self {
        Self {
            cmd: null_mut(),
            fence: 0,
            pending: Cell::new(None),
            acquire: 0,
        }
    }
}

/// Host-visible upload buffer chunk reused across batched transfer submissions.
struct StagingChunk {
    buffer: u64,
    alloc: Alloc,
    mapped: *mut u8,
    capacity: u64,
    cursor: u64,
    serial: u64,
    active: bool,
}

/// Optional device features, enabled whenever the adapter supports them.
#[derive(Clone, Copy, Debug, Default)]
struct Features {
    anisotropy: bool,
    depth_clamp: bool,
    /// Different blend states per colour target.
    independent_blend: bool,
    /// Writable storage buffers/textures in vertex or fragment shaders.
    vertex_stores: bool,
    fragment_stores: bool,
    /// `Rg16Float`/`R16Float` storage textures.
    storage_formats: bool,
    /// Hardware BC1..BC7 block compression.
    bc_compression: bool,
    /// Hardware ASTC LDR block compression.
    astc_compression: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PassKey {
    /// (VkFormat, loadOp, storeOp)
    colors: [(i32, i32, i32); MAX_PASS_COLORS],
    color_len: u8,
    depth: Option<(i32, i32, i32)>,
}
impl PassKey {
    const fn empty() -> Self {
        Self {
            colors: [(0, 0, 0); MAX_PASS_COLORS],
            color_len: 0,
            depth: None,
        }
    }
    #[inline]
    fn colors(&self) -> &[(i32, i32, i32)] {
        &self.colors[..self.color_len as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FramebufferKey {
    pass: u64,
    views: [u64; MAX_PASS_COLORS + 1],
    len: u8,
    w: u32,
    h: u32,
}

pub(crate) struct Gpu<'w> {
    instance: Handle,
    physical: Handle,
    device: Handle,
    queue: Handle,
    family: u32,
    memory: VkPhysicalDeviceMemoryProperties,
    limits: VkPhysicalDeviceLimits,
    features: Features,
    name: String,
    debug: Option<setup::Debug>,
    slabs: Vec<Slab>,
    buffers: Pool<Buffer>,
    textures: Pool<Texture>,
    samplers: Pool<Sampler>,
    pipelines: Pool<Pipeline>,
    groups: Pool<BindGroup>,
    sampler_cache: HashMap<[u32; 8], SamplerId>,
    set_layouts: HashMap<Vec<(u32, i32, u32)>, u64>,
    layout_ids: HashMap<Vec<Entry>, u32>,
    pipeline_layouts: HashMap<(Vec<u64>, u32, u32), u64>,
    pipeline_cache: u64,
    render_passes: HashMap<PassKey, u64>,
    framebuffers: HashMap<FramebufferKey, u64>,
    descriptor_pools: Vec<DescriptorPool>,
    command_pool: u64,
    subs: [Submission; SUB_COUNT],
    frame: usize,
    serial: u64,
    garbage: Vec<(u64, Garbage)>,
    staging: Vec<StagingChunk>,
    xfer_slot: usize,
    xfer_recording: bool,
    xfer_plan: record::Plan,
    plan: record::Plan,
    scope: record::Scope,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    present: Option<swapchain::Presentation<'w>>,
    _window: core::marker::PhantomData<&'w ()>,
}

impl Gpu<'static> {
    pub fn headless() -> Result<Self> {
        let mut gpu = Self::with_instance(&[])?;
        gpu.open_device(0)?;
        Ok(gpu)
    }
}

/// Platforms with the offscreen backend only (Android today).
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
impl Gpu<'_> {
    pub fn acquire(&mut self) -> Result<Option<TextureId>> {
        Err("presentation is not available on this platform".into())
    }
    pub fn present(&mut self, _: &crate::gpu::Encoder) -> Result<()> {
        Err("presentation is not available on this platform".into())
    }
    pub fn surface_format(&self) -> Format {
        Format::Bgra8Srgb
    }
    pub fn surface_size(&self) -> (u32, u32) {
        (0, 0)
    }
}

impl<'w> Gpu<'w> {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn limits(&self) -> Limits {
        let l = &self.limits;
        Limits {
            max_texture_2d: l.maxImageDimension2D,
            max_push_constants: l.maxPushConstantsSize,
            max_anisotropy: if self.features.anisotropy {
                l.maxSamplerAnisotropy
            } else {
                1.0
            },
            depth_clamp: self.features.depth_clamp,
            uniform_offset_alignment: l.minUniformBufferOffsetAlignment,
            storage_offset_alignment: l.minStorageBufferOffsetAlignment,
            depth_filterable: self.format_features(126) & resources::FEATURE_LINEAR != 0,
            float32_readable: true,
            texture_compression_bc: self.features.bc_compression,
            texture_compression_astc: self.features.astc_compression,
        }
    }
    pub fn validation_log(&self) -> Option<ValidationLog> {
        self.debug.as_ref().map(|d| d.log.clone())
    }
    pub fn frame_slot(&self) -> usize {
        self.frame
    }

    fn buffer(&self, id: BufferId) -> Result<&Buffer> {
        self.buffers
            .get(id.raw())
            .ok_or_else(|| format!("{id:?} was destroyed or does not belong to this device"))
    }
    fn texture(&self, id: TextureId) -> Result<&Texture> {
        self.textures
            .get(id.raw())
            .ok_or_else(|| format!("{id:?} was destroyed or does not belong to this device"))
    }
    fn pipeline(&self, id: PipelineId) -> Result<&Pipeline> {
        self.pipelines
            .get(id.raw())
            .ok_or_else(|| format!("{id:?} does not belong to this device"))
    }
    fn group(&self, id: BindGroupId) -> Result<&BindGroup> {
        self.groups
            .get(id.raw())
            .ok_or_else(|| format!("{id:?} was destroyed or does not belong to this device"))
    }

    /// Waits for submission `which` if it is still pending.
    fn wait(&self, which: usize) -> Result<()> {
        let sub = &self.subs[which];
        if sub.pending.get().is_some() {
            // SAFETY: the fence belongs to this device; waiting does not require external
            // synchronization of the fence.
            let timeout = FENCE_TIMEOUT_NS.load(std::sync::atomic::Ordering::Relaxed);
            let rc = unsafe { vkWaitForFences(self.device, 1, &sub.fence, 1, timeout) };
            if rc == 2 {
                return Err(format!(
                    "GPU submission did not complete within {:.0} s",
                    timeout as f64 / 1e9
                ));
            }
            check(rc, "vkWaitForFences")?;
            sub.pending.set(None);
        }
        Ok(())
    }
    /// Waits for every pending submission with a serial up to `serial`.
    fn wait_serial(&self, serial: u64) -> Result<()> {
        for which in 0..self.subs.len() {
            if self.subs[which].pending.get().is_some_and(|s| s <= serial) {
                self.wait(which)?;
            }
        }
        Ok(())
    }
    /// Every submission with a serial up to the returned one has completed.
    fn completed(&self) -> u64 {
        for sub in &self.subs {
            // SAFETY: querying a fence status needs no external synchronization.
            if sub.pending.get().is_some()
                && unsafe { vkGetFenceStatus(self.device, sub.fence) } == 0
            {
                sub.pending.set(None);
            }
        }
        self.subs
            .iter()
            .filter_map(|s| s.pending.get())
            .min()
            .map_or(self.serial, |s| s - 1)
    }
    /// Frees garbage that no pending submission can reference.
    fn collect(&mut self) {
        let done = self.completed();
        let mut i = 0;
        while i < self.garbage.len() {
            if self.garbage[i].0 <= done {
                let (_, g) = self.garbage.swap_remove(i);
                self.free(g);
            } else {
                i += 1;
            }
        }
    }
    fn discard(&mut self, g: Garbage) {
        self.garbage.push((self.serial, g));
        self.collect();
    }
    fn free(&mut self, g: Garbage) {
        let d = self.device;
        // SAFETY: garbage is only freed after every submission that could reference it has
        // completed (or after vkDeviceWaitIdle), and nothing else holds these handles.
        unsafe {
            match g {
                Garbage::Buffer(buffer, alloc) => {
                    vkDestroyBuffer(d, buffer, null());
                    self.free_alloc(alloc);
                }
                Garbage::Image(image, alloc, views) => {
                    for v in views {
                        vkDestroyImageView(d, v, null());
                    }
                    if alloc.memory != 0 {
                        vkDestroyImage(d, image, null());
                        self.free_alloc(alloc);
                    }
                }
                Garbage::Framebuffer(f) => vkDestroyFramebuffer(d, f, null()),
                Garbage::Set(_, 0, _) => {} // an empty group owns no descriptor set
                Garbage::Set(pool, set, counts) => {
                    let p = &mut self.descriptor_pools[pool];
                    vkFreeDescriptorSets(d, p.handle, 1, &set);
                    p.sets += 1;
                    for (free, used) in p.counts.iter_mut().zip(counts) {
                        *free += used;
                    }
                }
            }
        }
    }
    /// Moves every framebuffer that references one of `views` to the garbage list.
    fn forget_views(&mut self, views: &[u64]) {
        if views.is_empty() || self.framebuffers.is_empty() {
            return;
        }
        let serial = self.serial;
        let garbage = &mut self.garbage;
        self.framebuffers.retain(|k, &mut f| {
            let hit = k.views[..k.len as usize].iter().any(|v| views.contains(v));
            if hit {
                garbage.push((serial, Garbage::Framebuffer(f)));
            }
            !hit
        });
    }
}

impl Drop for Gpu<'_> {
    fn drop(&mut self) {
        // SAFETY: the device is idle before any object is destroyed; children are destroyed
        // before their parents (device objects, swapchain, device, surface, messenger,
        // instance). The validation log outlives the instance because the Arc held by
        // `self.debug` is dropped only after this function returns.
        unsafe {
            let d = self.device;
            if !d.is_null() {
                if self.xfer_recording {
                    let _ = vkEndCommandBuffer(self.subs[XFER_BASE + self.xfer_slot].cmd);
                    self.xfer_recording = false;
                }
                vkDeviceWaitIdle(d);
                for sub in &self.subs {
                    sub.pending.set(None);
                }
                for (_, g) in core::mem::take(&mut self.garbage) {
                    self.free(g);
                }
                for p in self.descriptor_pools.drain(..) {
                    vkDestroyDescriptorPool(d, p.handle, null());
                }
                for (_, f) in self.framebuffers.drain() {
                    vkDestroyFramebuffer(d, f, null());
                }
                for (_, p) in self.render_passes.drain() {
                    vkDestroyRenderPass(d, p, null());
                }
                for p in self.pipelines.drain() {
                    vkDestroyPipeline(d, p.handle, null());
                }
                if self.pipeline_cache != 0 {
                    vkDestroyPipelineCache(d, self.pipeline_cache, null());
                }
                for (_, l) in self.pipeline_layouts.drain() {
                    vkDestroyPipelineLayout(d, l, null());
                }
                for (_, l) in self.set_layouts.drain() {
                    vkDestroyDescriptorSetLayout(d, l, null());
                }
                for s in self.samplers.drain() {
                    vkDestroySampler(d, s.handle, null());
                }
                for t in self.textures.drain() {
                    for (_, v) in t.views {
                        vkDestroyImageView(d, v, null());
                    }
                    if t.alloc.memory != 0 {
                        vkDestroyImage(d, t.image, null());
                        self.free_alloc(t.alloc);
                    }
                }
                for b in self.buffers.drain() {
                    vkDestroyBuffer(d, b.handle, null());
                    self.free_alloc(b.alloc);
                }
                for s in core::mem::take(&mut self.staging) {
                    vkDestroyBuffer(d, s.buffer, null());
                    self.free_alloc(s.alloc);
                }
                for slab in self.slabs.drain(..) {
                    vkFreeMemory(d, slab.memory, null());
                }
                for sub in &self.subs {
                    vkDestroyFence(d, sub.fence, null());
                    if sub.acquire != 0 {
                        vkDestroySemaphore(d, sub.acquire, null());
                    }
                }
                vkDestroyCommandPool(d, self.command_pool, null());
                // Swapchain image views went with the textures above.
                #[cfg(any(target_os = "linux", target_os = "windows"))]
                if let Some(p) = self.present.as_mut() {
                    p.destroy_chain(d);
                }
                vkDestroyDevice(d, null());
            }
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            if let Some(p) = self.present.take() {
                vkDestroySurfaceKHR(self.instance, p.surface, null());
            }
            if let Some(debug) = &self.debug {
                debug.destroy_messenger(self.instance);
            }
            if !self.instance.is_null() {
                vkDestroyInstance(self.instance, null());
            }
        }
    }
}
