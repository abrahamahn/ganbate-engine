//! Ganbate shading language: WGSL syntax compiled to SPIR-V 1.0 for Vulkan 1.0.
//!
//! The engine owns its shader toolchain end to end. Shaders are authored once in a documented
//! WGSL subset, compiled here with zero third-party code, and handed to each graphics backend:
//! SPIR-V for Vulkan, Metal Shading Language for Metal ([`compile_msl`]), WGSL for WebGPU
//! ([`compile_wgsl`]) and GLSL ES 3.00 for WebGL 2 ([`compile_glsl`]).
//!
//! The public types below are the contract the GPU layer builds pipelines from. Reflection only
//! lists resources and interface variables that the selected entry point statically uses.
//!
//! # Pipeline
//!
//! `lexer` (tokens, nested comments, WGSL template-list discovery) → `parser` (AST with
//! spans, WGSL operator rules) → `check` (name resolution, types, WGSL abstract-literal typing,
//! constant folding, layout) → `emit` (SPIR-V 1.0 words). Each [`compile`] call checks the
//! whole module but emits only the functions and resources reachable from the selected entry.
//!
//! # Supported language (a WGSL subset)
//!
//! * **Module scope:** `struct` (members with `@location`, `@builtin`, `@interpolate`,
//!   `@invariant`, `@align`, `@size`), `const` (abstract or typed; usable in array sizes,
//!   attributes and other consts; dynamically indexed constant arrays/matrices become a
//!   `Private` variable with a constant initializer), `alias`, `const_assert`, `var<private>`,
//!   `var<uniform>`, `var<storage, read|read_write>` (struct, optionally ending in a runtime
//!   array, or a bare `array<E>`), textures (`texture_2d<f32|i32|u32>`, `texture_2d_array<T>`,
//!   `texture_cube<T>`, `texture_depth_2d`, `texture_depth_2d_array`, `texture_depth_cube`,
//!   `texture_storage_2d<rgba8unorm|rgba16float|rgba32float|r32float|rg16float|r16float, write>`),
//!   `sampler`, `sampler_comparison`, functions and `@vertex` / `@fragment` /
//!   `@compute @workgroup_size(x, y, z)` entry points. Declarations may appear in any order.
//!   `diagnostic(...)` directives are accepted and ignored.
//! * **Types:** `bool`, `i32`, `u32`, `f32`, `vecN<T>` and the `vecNf/i/u` aliases,
//!   `matCxR<f32>` / `matCxRf`, `array<T, N>`, `array<T>` (storage only), structs.
//! * **Statements:** `let`, `var`, `const`, `=`, all compound assignments, `++`, `--`,
//!   `_ = e`, `if`/`else if`/`else`, `for`, `while`, `loop` with `continuing` and `break if`,
//!   `break`, `continue`, `return`, `discard`, blocks, call statements, `const_assert`.
//!   Assignments go through member, index, component and swizzle targets
//!   (`a[i].pos.z = ...`, `v.xy = ...`).
//! * **Expressions:** all WGSL operators with WGSL precedence (including its rule that shifts,
//!   comparisons and `& | ^` do not mix without parentheses), literals of every WGSL form,
//!   swizzles, indexing of vectors, matrices, arrays and runtime arrays (constant or dynamic,
//!   on references or values), value constructors (typed and inferred: `vec3f(..)`,
//!   `vec3(..)`, `array(..)`, `mat3x3f(..)`, structs, zero values), conversions and `bitcast`.
//! * **Built-ins:** abs acos acosh asin asinh atan atanh atan2 ceil clamp cos cosh
//!   countOneBits cross degrees determinant distance dot dpdx dpdy exp exp2 faceForward floor
//!   fma fract fwidth inverseSqrt ldexp length log log2 max min mix normalize pow radians
//!   reflect refract reverseBits round saturate sign sin sinh smoothstep sqrt step tan tanh
//!   transpose trunc select all any arrayLength pack/unpack{4x8snorm, 4x8unorm, 2x16snorm,
//!   2x16unorm, 2x16float} workgroupBarrier storageBarrier, and textureSample,
//!   textureSampleBias, textureSampleLevel, textureSampleGrad, textureSampleCompare,
//!   textureSampleCompareLevel, textureLoad, textureStore, textureDimensions,
//!   textureNumLevels, textureNumLayers (with constant `offset` arguments where WGSL has them).
//!
//! # Ganbate extensions
//!
//! * `var<push_constant> name: T;` (no `@group`/`@binding`): lowered to the `PushConstant`
//!   storage class with a `Block` decoration; at most one may be used per entry point and its
//!   WGSL size is reported in [`Compiled::push_constant_size`].
//! * Multi-component swizzles are assignable (`v.xy = p;`, `v.zx += d;`), which WGSL forbids.
//! * Integer user varyings get `Flat` automatically (WGSL requires `@interpolate(flat)`).
//! * A built-in input may appear more than once among an entry point's inputs (e.g.
//!   `@builtin(position)` both in a struct and as a parameter); all copies read one variable.
//!
//! # Known deviations from WGSL
//!
//! * `&&` and `||` lower to non-short-circuit `OpLogicalAnd` / `OpLogicalOr` when the right
//!   operand is free of side effects and cannot fault (unobservable); they short-circuit with
//!   real control flow (`OpPhi`) when it contains a call, a dynamic index, a storage-buffer
//!   array read, an integer division, or a texture or derivative built-in.
//! * Dynamic indices are not clamped: an out-of-range index is undefined behaviour as in
//!   GLSL (constant out-of-range indices are compile errors).
//! * Integer division or remainder by zero (and `i32::MIN / -1`) at run time follow SPIR-V
//!   (undefined) instead of WGSL's defined results; float→int conversion of out-of-range
//!   values is likewise undefined. Constant expressions follow WGSL and report errors.
//!   Non-constant shift amounts are masked to 0..31 as WGSL requires.
//! * `discard` lowers to `OpKill` (terminates the invocation rather than demoting it to a
//!   helper invocation).
//! * No uniformity analysis is performed (derivatives and implicit-LOD sampling in
//!   non-uniform control flow are not diagnosed).
//! * Clip space is passed through unchanged: `@builtin(position)` is Vulkan's (+Y down,
//!   depth 0..1); no Y flip is inserted.
//! * Constant folding of built-ins covers the common math functions; other built-ins
//!   (e.g. `determinant`, `refract`, pack/unpack) are rejected in module-scope `const`
//!   initializers and evaluated at run time inside functions.
//!
//! # Not supported (clear compile errors)
//!
//! `switch`, pointers (`ptr`, `&`, `*` — except `arrayLength(&buf.arr)`), `override`, `f16`,
//! atomics, `var<workgroup>`, texture/sampler function parameters, other texture types,
//! storage textures with read access, `matCx2<f32>` inside uniform buffers (its 8-byte column
//! stride is not expressible in Vulkan 1.0 std140; use a storage buffer or vec4 columns),
//! `modf`/`frexp`, the bit-extraction built-ins, `textureGather*` and fine/coarse derivatives.
//!
//! # Limits
//!
//! Inputs beyond these limits are rejected with an error (never a crash): 128 levels of
//! syntactic nesting (blocks, `else if` chains, parentheses), expression trees 128 levels deep
//! (including left-deep chains such as `a + b + ...`), 32 levels of array/struct nesting in a
//! type, buffer types under 1 GiB, and 64 levels of module-scope declarations that refer to
//! declarations appearing later in the file. The deepest accepted input compiles on a 2 MiB
//! thread stack in debug builds.
//!
//! Besides [`compile`], [`entry_points`] lists a module's entry points and
//! [`Compiled::to_bytes`] gives the module as little-endian bytes. [`compile_wgsl`] also
//! produces WebGPU WGSL for the entry point from the same checked program ([`wgsl`] describes
//! the lowering: push constants become a dynamically offset uniform in
//! [`wgsl::PUSH_CONSTANT_GROUP`], and clip-space y is flipped in a generated entry wrapper).
//!
//! # SPIR-V conventions
//!
//! SPIR-V 1.0, `Logical GLSL450`, one `OpEntryPoint` named `main` (a generated wrapper that
//! loads the interface, calls the user entry function and stores its results). Uniform buffers
//! are `Uniform` + `Block`, storage buffers `Uniform` + `BufferBlock` (`NonWritable` members when
//! read-only), push constants `PushConstant` + `Block`; non-struct buffer types are wrapped in a
//! one-member struct. Layout follows WGSL's host-shareable rules (vec3 align 16, matrix column
//! stride roundUp(AlignOf(vecR), SizeOf(vecR)), array stride roundUp(AlignOf(E), SizeOf(E))),
//! with WGSL's extra uniform-address-space checks, so uniform layouts are valid std140 and the
//! rest valid std430. Laid-out and plain variants of structs/arrays use distinct type ids;
//! composite values are copied member-wise when they move between buffer and function memory.

use core::fmt;

mod ast;
mod builtins;
mod check;
pub mod compose;
mod consteval;
mod emit;
pub mod glsl;
mod ir;
mod lexer;
mod msl;
mod parser;
mod spirv;
mod types;
pub mod wgsl;

#[cfg(test)]
mod tests;

pub use compose::{ShaderLibrary, compose};
pub use msl::{
    MAX_RESOURCE_BUFFERS, Msl, MslIndex, MslSlot, PUSH_CONSTANT_BUFFER, SIZES_BUFFER, VERTEX_BUFFER,
};

/// Compile-time specialization value for a module-scope `const` declaration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpecValue {
    Bool(bool),
    I32(i32),
    U32(u32),
    F32(f32),
}

impl SpecValue {
    pub(crate) fn type_name(self) -> &'static str {
        match self {
            SpecValue::Bool(_) => "bool",
            SpecValue::I32(_) => "i32",
            SpecValue::U32(_) => "u32",
            SpecValue::F32(_) => "f32",
        }
    }
}

/// Options for `#import` composition and compile-time `const` specialization permutations.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompileOptions<'a> {
    pub imports: &'a [(&'a str, &'a str)],
    pub specialize: &'a [(&'a str, SpecValue)],
}

impl<'a> CompileOptions<'a> {
    pub const fn new() -> Self {
        Self {
            imports: &[],
            specialize: &[],
        }
    }

    pub const fn with_imports(mut self, imports: &'a [(&'a str, &'a str)]) -> Self {
        self.imports = imports;
        self
    }

    pub const fn with_specialize(mut self, specialize: &'a [(&'a str, SpecValue)]) -> Self {
        self.specialize = specialize;
        self
    }
}

/// Pipeline stage of the selected entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

/// One compiled entry point: a SPIR-V module whose entry point is named `main`, plus reflection.
#[derive(Clone, Debug, PartialEq)]
pub struct Compiled {
    pub stage: Stage,
    /// SPIR-V 1.0 words, little-endian when written to disk.
    pub words: Vec<u32>,
    /// Resources the entry point uses, sorted by (group, binding).
    pub bindings: Vec<Binding>,
    /// Size in bytes of the `var<push_constant>` block the entry point uses, or 0.
    pub push_constant_size: u32,
    /// `@location` inputs of the entry point (vertex attributes for the vertex stage).
    pub inputs: Vec<Interface>,
    /// `@location` outputs of the entry point (colour targets for the fragment stage).
    pub outputs: Vec<Interface>,
    /// `@workgroup_size` for compute entry points; `[1, 1, 1]` otherwise.
    pub workgroup_size: [u32; 3],
    /// The same entry point as Metal Shading Language: filled by [`compile`] on Apple
    /// targets and by [`compile_msl`] everywhere; `None` otherwise.
    pub msl: Option<Msl>,
    /// WebGPU WGSL for the same entry point, filled by [`compile_wgsl`] (`None` from
    /// [`compile`]).
    pub wgsl: Option<wgsl::Wgsl>,
    /// GLSL ES 3.00 for WebGL 2, filled by [`compile_glsl`] (`None` otherwise).
    pub glsl: Option<glsl::Glsl>,
}

/// A resource declared with `@group(g) @binding(b)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Binding {
    pub group: u32,
    pub binding: u32,
    pub kind: BindingKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindingKind {
    /// `var<uniform>`; `size` is the WGSL host-shareable size of the declared type in bytes.
    UniformBuffer { size: u32 },
    /// `var<storage, read>` or `var<storage, read_write>`.
    StorageBuffer { read_only: bool },
    /// `texture_2d<T>`, `texture_2d_array<T>`, `texture_cube<T>` and the `texture_depth_*` forms.
    Texture {
        view: TextureView,
        sample: SampleType,
    },
    /// `texture_storage_2d<format, write>` (write-only).
    StorageTexture {
        view: TextureView,
        format: TexelFormat,
    },
    /// `sampler` or `sampler_comparison`.
    Sampler { comparison: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureView {
    D2,
    D2Array,
    Cube,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleType {
    Float,
    Depth,
    Sint,
    Uint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelFormat {
    Rgba8Unorm,
    Rgba16Float,
    Rgba32Float,
    R32Float,
    Rg16Float,
    R16Float,
}

/// A user-defined `@location` interface variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Interface {
    pub location: u32,
    pub ty: IoType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IoType {
    F32,
    Vec2F,
    Vec3F,
    Vec4F,
    I32,
    Vec2I,
    Vec3I,
    Vec4I,
    U32,
    Vec2U,
    Vec3U,
    Vec4U,
}

/// A compile error with the 1-based source position it refers to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub line: u32,
    pub column: u32,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for Error {}

/// Compile `entry` from `source` for `stage`.
///
/// The whole module is parsed and type-checked; only the entry point's call tree and the
/// resources it uses are emitted and reflected.
pub fn compile(source: &str, stage: Stage, entry: &str) -> Result<Compiled, Error> {
    compile_with_options(source, stage, entry, &CompileOptions::new())
}

/// Compile `entry` from `source` with `#import` modules and `const` specialization overrides.
pub fn compile_with_options(
    source: &str,
    stage: Stage,
    entry: &str,
    options: &CompileOptions<'_>,
) -> Result<Compiled, Error> {
    compile_targets(source, stage, entry, options, cfg!(target_vendor = "apple"))
}

/// Like [`compile`], and also emits the entry point as MSL into [`Compiled::msl`] on every
/// platform (see the `msl` module documentation for the conventions).
pub fn compile_msl(source: &str, stage: Stage, entry: &str) -> Result<Compiled, Error> {
    compile_msl_with_options(source, stage, entry, &CompileOptions::new())
}

/// [`compile_msl`] with `#import` composition and `const` specialization overrides.
pub fn compile_msl_with_options(
    source: &str,
    stage: Stage,
    entry: &str,
    options: &CompileOptions<'_>,
) -> Result<Compiled, Error> {
    compile_targets(source, stage, entry, options, true)
}

fn check_program(source: &str, options: &CompileOptions<'_>) -> Result<ir::Program, Error> {
    let composed = compose::compose_cow(source, options.imports)?;
    let tokens = lexer::tokenize(&composed)?;
    let module = parser::parse(tokens)?;
    check::check_with(&module, options.specialize)
}

fn compile_targets(
    source: &str,
    stage: Stage,
    entry: &str,
    options: &CompileOptions<'_>,
    msl: bool,
) -> Result<Compiled, Error> {
    let program = check_program(source, options)?;
    let mut compiled = emit::emit(&program, stage, entry)?;
    if msl {
        compiled.msl = Some(msl::emit(&program, stage, entry)?);
    }
    Ok(compiled)
}

/// [`compile`], plus WebGPU WGSL for the entry point in [`Compiled::wgsl`]. The reflection is
/// the SPIR-V module's; the push-constant block, if used, is the uniform at
/// [`wgsl::Wgsl::push_constants`].
pub fn compile_wgsl(source: &str, stage: Stage, entry: &str) -> Result<Compiled, Error> {
    compile_wgsl_with_options(source, stage, entry, &CompileOptions::new())
}

/// [`compile_wgsl`] with `#import` composition and `const` specialization overrides.
pub fn compile_wgsl_with_options(
    source: &str,
    stage: Stage,
    entry: &str,
    options: &CompileOptions<'_>,
) -> Result<Compiled, Error> {
    let program = check_program(source, options)?;
    let mut compiled = emit::emit(&program, stage, entry)?;
    compiled.wgsl = Some(wgsl::emit(&program, stage, entry)?);
    Ok(compiled)
}

/// [`compile`], plus GLSL ES 3.00 for WebGL 2 in [`Compiled::glsl`] ([`glsl`] describes the
/// lowering: uniform blocks, combined samplers, storage buffers read through textures, push
/// constants as the `ganbate_push` block, and the Vulkan clip-space conventions kept). The
/// reflection is the SPIR-V module's.
pub fn compile_glsl(source: &str, stage: Stage, entry: &str) -> Result<Compiled, Error> {
    compile_glsl_with_options(source, stage, entry, &CompileOptions::new())
}

/// [`compile_glsl`] with `#import` composition and `const` specialization overrides.
pub fn compile_glsl_with_options(
    source: &str,
    stage: Stage,
    entry: &str,
    options: &CompileOptions<'_>,
) -> Result<Compiled, Error> {
    let program = check_program(source, options)?;
    let mut compiled = emit::emit(&program, stage, entry)?;
    compiled.glsl = Some(glsl::emit(&program, stage, entry)?);
    Ok(compiled)
}

/// List the entry points declared in `source` (name and stage) in declaration order, after
/// checking the whole module.
pub fn entry_points(source: &str) -> Result<Vec<(String, Stage)>, Error> {
    entry_points_with_options(source, &CompileOptions::new())
}

/// [`entry_points`] with `#import` composition and `const` specialization overrides.
pub fn entry_points_with_options(
    source: &str,
    options: &CompileOptions<'_>,
) -> Result<Vec<(String, Stage)>, Error> {
    let program = check_program(source, options)?;
    Ok(program
        .functions
        .iter()
        .filter_map(|f| f.stage.map(|s| (f.name.clone(), s)))
        .collect())
}

impl Compiled {
    /// The SPIR-V module as little-endian bytes, ready for `vkCreateShaderModule` or a file.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}
