//! Metal implementation of the backend-neutral GPU layer in [`crate::gpu`], for macOS and iOS.
//!
//! Ownership: [`Gpu`] owns the `MTLDevice`, one `MTLCommandQueue` and every object created
//! through it, in generational pools addressed by the public `*Id` handles. Destroying an
//! object releases the engine's reference at once: command buffers retain the resources they
//! use until they complete, so objects in flight stay alive.
//!
//! Shaders: pipelines are built from the MSL that `ganbate-shader` emits next to the SPIR-V
//! ([`ganbate_shader::Compiled::msl`]), compiled at pipeline creation with
//! `newLibraryWithSource:`. Each stage's binding map turns `@group/@binding` into
//! `[[buffer]]`, `[[texture]]` and `[[sampler]]` indices; a bind group is resolved against the
//! current pipeline's maps when a draw or dispatch is recorded. Push constants are sent with
//! `set*Bytes` at `ganbate_shader::PUSH_CONSTANT_BUFFER`, the vertex buffer is bound at
//! `ganbate_shader::VERTEX_BUFFER` and described by an `MTLVertexDescriptor`.
//!
//! Synchronization: Metal tracks hazards between passes for the default (tracked) resources
//! of one queue, so recorded scopes need no barriers. Every submission is one command buffer;
//! the backend keeps the uncompleted ones with their serial numbers. `write_buffer` and
//! `read_buffer` wait for the submissions that used a buffer; `submit` waits for its own;
//! `present` keeps at most two frames in flight by waiting, before it returns, for the command
//! buffer that last used the next frame slot. Waiting uses `waitUntilCompleted` (no Objective-C
//! blocks are needed), so the submit timeout of the Vulkan backend does not apply.
//!
//! Threading and pools: the device is single-threaded (`!Send`, `!Sync`). Every public entry
//! point runs inside an autorelease pool; objects that outlive it are retained.
//!
//! This backend has been compiled for macOS and iOS but not yet run on hardware.
mod ffi;
mod pipelines;
mod present;
mod record;
mod resources;

use crate::apple::objc::{Id, NSUInteger, Obj, autoreleasepool, class, send};
use crate::gpu::{
    BufferDesc, BufferId, Format, Limits, PipelineId, SamplerId, TextureDesc, TextureId,
    ValidationLog,
};
use ganbate_shader::{BindingKind, MslSlot};
use std::collections::HashMap;

pub(crate) type Result<T> = std::result::Result<T, String>;

/// Frames in flight for presentation.
const FRAMES: usize = 2;

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

struct Buffer {
    obj: Obj,
    desc: BufferDesc,
    /// `contents` of shared (`Upload`/`Readback`) buffers; null for private ones.
    contents: *mut u8,
    /// Serial of the newest submission that referenced the buffer.
    last_use: u64,
}

/// A texture view: (texture type, first mip, mips, first layer, layers).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ViewKey {
    ty: u64,
    base_mip: u32,
    mips: u32,
    base_layer: u32,
    layers: u32,
}

struct Texture {
    obj: Obj,
    desc: TextureDesc,
    pixel_format: u64,
    /// The texture type it was created with.
    ty: u64,
    sw_decompress: Option<ganbate_core::world::TextureCompression>,
    views: Vec<(ViewKey, Obj)>,
    swapchain: bool,
}

struct Sampler {
    obj: Obj,
    comparison: bool,
}

/// One binding of a bind group layout, merged over the pipeline's stages.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Entry {
    binding: u32,
    kind: BindingKind,
    /// Bit 0 vertex, bit 1 fragment, bit 2 compute.
    stages: u32,
}

/// Which argument table a stage binds into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageKind {
    Vertex,
    Fragment,
    Compute,
}

/// What one shader stage of a pipeline needs bound.
#[derive(Clone, Debug)]
struct StageMap {
    kind: StageKind,
    slots: Vec<MslSlot>,
    push: bool,
    sized: Vec<(u32, u32)>,
}

/// Fixed-function state a render pipeline sets on the encoder.
#[derive(Clone, Debug)]
struct Raster {
    primitive: u64,
    cull: u64,
    winding: u64,
    depth_bias: (f32, f32),
    clip: u64,
    depth_state: Obj,
}

struct Pipeline {
    state: Obj,
    compute: bool,
    /// Index = group; empty lists fill gaps below the highest used group.
    groups: Vec<Vec<Entry>>,
    stages: Vec<StageMap>,
    push_size: u32,
    colors: Vec<Format>,
    depth: Option<Format>,
    vertex_input: bool,
    raster: Option<Raster>,
    threads: [u32; 3],
}

/// A resource referenced by a bind group.
#[derive(Clone, Debug)]
enum Bound {
    Buffer {
        id: BufferId,
        offset: u64,
        size: u64,
    },
    Texture {
        id: TextureId,
        /// The texture or a cached view of it; alive while `id` is.
        obj: Id,
        mips: (u32, u32),
        layers: (u32, u32),
    },
    Sampler(SamplerId),
}

struct BindGroup {
    layout: Vec<Entry>,
    bound: Vec<(u32, Bound)>,
}

pub(crate) struct Gpu<'w> {
    device: Obj,
    queue: Obj,
    name: String,
    /// iOS GPUs before Apple4 (A11) cannot clamp depth.
    depth_clamp: bool,
    bc_compression: bool,
    astc_compression: bool,
    buffers: Pool<Buffer>,
    textures: Pool<Texture>,
    samplers: Pool<Sampler>,
    pipelines: Pool<Pipeline>,
    groups: Pool<BindGroup>,
    sampler_cache: HashMap<[u64; 8], SamplerId>,
    depth_states: HashMap<(u64, bool), Obj>,
    /// Committed command buffers that may not have completed, oldest first.
    in_flight: Vec<(u64, Obj)>,
    serial: u64,
    /// Serial of the presentation that last used each frame slot.
    frame_serials: [u64; FRAMES],
    frame: usize,
    staging: Option<(Obj, u64)>,
    blitter: Option<pipelines::Blitter>,
    present: Option<present::Surface<'w>>,
}

/// GPU errors keep the message Metal gives.
fn ns_error(prefix: &str, error: Id) -> String {
    // SAFETY: error is nil or an autoreleased NSError from the call that just failed.
    format!("{prefix}: {}", unsafe {
        crate::apple::objc::error_text(error)
    })
}

impl Gpu<'static> {
    /// The default GPU (on a Mac, `GANBATE_METAL_DEVICE` picks one by name substring).
    pub fn headless() -> Result<Self> {
        autoreleasepool(|| Self::with_device(Self::pick_device()?))
    }
}

impl<'w> Gpu<'w> {
    fn pick_device() -> Result<Obj> {
        #[cfg(target_os = "macos")]
        if let Ok(want) = std::env::var("GANBATE_METAL_DEVICE") {
            // SAFETY: MTLCopyAllDevices returns an owned NSArray of MTLDevices.
            unsafe {
                let all = Obj::owned(ffi::MTLCopyAllDevices()).ok_or("no Metal devices")?;
                let n = send!(all.id(), "count"; NSUInteger);
                let mut names = Vec::new();
                for i in 0..n {
                    let d = send!(all.id(), "objectAtIndex:", i => NSUInteger; Id);
                    let name = crate::apple::objc::string(send!(d, "name"; Id));
                    if name.contains(&want) {
                        return Obj::retained(d).ok_or_else(|| "nil device".into());
                    }
                    names.push(name);
                }
                return Err(format!(
                    "GANBATE_METAL_DEVICE={want:?} matches none of {names:?}"
                ));
            }
        }
        // SAFETY: returns an owned device or nil.
        unsafe { Obj::owned(ffi::MTLCreateSystemDefaultDevice()) }
            .ok_or_else(|| "Metal is not available on this machine".into())
    }

    fn with_device(device: Obj) -> Result<Self> {
        // SAFETY: plain queries and an owned command queue on a live device.
        unsafe {
            let d = device.id();
            let name = crate::apple::objc::string(send!(d, "name"; Id));
            let queue = Obj::owned(send!(d, "newCommandQueue"; Id))
                .ok_or("newCommandQueue returned nil")?;
            let supports = |family: i64| -> bool {
                send!(d, "respondsToSelector:", crate::apple::objc::sel!("supportsFamily:") => crate::apple::objc::Sel; i8)
                    != 0
                    && send!(d, "supportsFamily:", family => i64; i8) != 0
            };
            let depth_clamp = cfg!(target_os = "macos") || supports(ffi::family::APPLE4);
            let bc_compression = cfg!(target_os = "macos") || supports(ffi::family::MAC2);
            let astc_compression = supports(ffi::family::APPLE2) || supports(ffi::family::APPLE4);
            Ok(Self {
                device,
                queue,
                name,
                depth_clamp,
                bc_compression,
                astc_compression,
                buffers: Pool::default(),
                textures: Pool::default(),
                samplers: Pool::default(),
                pipelines: Pool::default(),
                groups: Pool::default(),
                sampler_cache: HashMap::new(),
                depth_states: HashMap::new(),
                in_flight: Vec::new(),
                serial: 0,
                frame_serials: [0; FRAMES],
                frame: 0,
                staging: None,
                blitter: None,
                present: None,
            })
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn limits(&self) -> Limits {
        Limits {
            // Every Apple GPU Metal supports here (A9 and later, all Macs) reaches 16384.
            max_texture_2d: 16384,
            // `set*Bytes` accepts up to 4 KiB.
            max_push_constants: 4096,
            max_anisotropy: 16.0,
            depth_clamp: self.depth_clamp,
            // Offsets of `constant` buffers must be 256-byte aligned on Intel and AMD Macs;
            // use that everywhere so code tested on one Mac runs on all.
            uniform_offset_alignment: 256,
            storage_offset_alignment: 256,
            depth_filterable: false,
            float32_readable: true,
            texture_compression_bc: self.bc_compression,
            texture_compression_astc: self.astc_compression,
        }
    }

    /// Metal reports API misuse through its own validation layer (`MTL_DEBUG_LAYER=1`) on
    /// the console, not through this log.
    pub fn validation_log(&self) -> Option<ValidationLog> {
        None
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

    /// A new command buffer (retained, so it can outlive the current autorelease pool).
    fn command_buffer(&self) -> Result<Obj> {
        // SAFETY: commandBuffer returns an autoreleased command buffer of a live queue.
        unsafe { Obj::retained(send!(self.queue.id(), "commandBuffer"; Id)) }
            .ok_or_else(|| "commandBuffer returned nil".into())
    }

    /// Commits `cb` and returns its serial.
    fn commit(&mut self, cb: Obj) -> u64 {
        // SAFETY: the command buffer is fully encoded (every encoder ended).
        unsafe { send!(cb.id(), "commit"; ()) };
        self.serial += 1;
        self.in_flight.push((self.serial, cb));
        self.serial
    }

    /// Waits for every committed command buffer with a serial up to `serial`; reports the
    /// first that failed.
    fn wait_serial(&mut self, serial: u64) -> Result<()> {
        if self.in_flight.first().is_none_or(|&(s, _)| s > serial) {
            return Ok(());
        }
        let split = self.in_flight.partition_point(|&(s, _)| s <= serial);
        let mut result = Ok(());
        for (_, cb) in self.in_flight.drain(..split) {
            // SAFETY: waiting on and querying a committed command buffer.
            unsafe {
                send!(cb.id(), "waitUntilCompleted"; ());
                if send!(cb.id(), "status"; NSUInteger) as u64 == ffi::status::ERROR
                    && result.is_ok()
                {
                    result = Err(ns_error(
                        "GPU command buffer failed",
                        send!(cb.id(), "error"; Id),
                    ));
                }
            }
        }
        result
    }

    /// Forgets command buffers that already completed (without blocking).
    fn collect(&mut self) -> Result<()> {
        let mut result = Ok(());
        self.in_flight.retain(|(_, cb)| {
            // SAFETY: querying a committed command buffer.
            let status = unsafe { send!(cb.id(), "status"; NSUInteger) } as u64;
            if status == ffi::status::ERROR && result.is_ok() {
                // SAFETY: as above.
                result = Err(ns_error("GPU command buffer failed", unsafe {
                    send!(cb.id(), "error"; Id)
                }));
            }
            status < ffi::status::COMPLETED
        });
        result
    }

    fn wait_idle(&mut self) -> Result<()> {
        self.wait_serial(u64::MAX)
    }

    /// A descriptor object from `[[Class alloc] init]` (+1).
    fn new_object(class: Id, what: &str) -> Result<Obj> {
        // SAFETY: alloc/init on a framework class returns an owned instance or nil.
        unsafe {
            let a = send!(class, "alloc"; Id);
            Obj::owned(send!(a, "init"; Id)).ok_or_else(|| format!("{what}: alloc/init failed"))
        }
    }
}

impl Drop for Gpu<'_> {
    fn drop(&mut self) {
        autoreleasepool(|| {
            // Command buffers keep what they use alive, but wait so nothing outlives the
            // window a presentation device draws into.
            let _ = self.wait_idle();
            self.groups.drain();
            self.pipelines.drain();
            self.samplers.drain();
            self.textures.drain();
            self.buffers.drain();
            self.present = None;
        });
    }
}
