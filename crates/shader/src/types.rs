//! Types, constant values and the WGSL host-shareable memory layout (AlignOf/SizeOf).

use crate::lexer::Span;
use crate::{SampleType, TexelFormat, TextureView};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    Bool,
    I32,
    U32,
    F32,
    AbstractInt,
    AbstractFloat,
}

impl Scalar {
    pub fn is_abstract(self) -> bool {
        matches!(self, Scalar::AbstractInt | Scalar::AbstractFloat)
    }
    pub fn is_float(self) -> bool {
        matches!(self, Scalar::F32 | Scalar::AbstractFloat)
    }
    pub fn is_int(self) -> bool {
        matches!(self, Scalar::I32 | Scalar::U32 | Scalar::AbstractInt)
    }
    pub fn is_numeric(self) -> bool {
        self != Scalar::Bool
    }
    /// WGSL's default concretization: AbstractInt → i32, AbstractFloat → f32.
    pub fn concretize(self) -> Scalar {
        match self {
            Scalar::AbstractInt => Scalar::I32,
            Scalar::AbstractFloat => Scalar::F32,
            s => s,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Scalar::Bool => "bool",
            Scalar::I32 => "i32",
            Scalar::U32 => "u32",
            Scalar::F32 => "f32",
            Scalar::AbstractInt => "abstract-int",
            Scalar::AbstractFloat => "abstract-float",
        }
    }
    /// Can a value of abstract type `self` be automatically converted to `to`?
    pub fn converts_to(self, to: Scalar) -> bool {
        self == to
            || match self {
                Scalar::AbstractInt => matches!(
                    to,
                    Scalar::I32 | Scalar::U32 | Scalar::F32 | Scalar::AbstractFloat
                ),
                Scalar::AbstractFloat => to == Scalar::F32,
                _ => false,
            }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Scalar(Scalar),
    Vector(u8, Scalar),
    /// `matCxR<f32>`: (columns, rows).
    Matrix(u8, u8),
    /// Element type and fixed length, or `None` for a runtime-sized array.
    Array(Box<Ty>, Option<u32>),
    Struct(usize),
    Texture(TextureView, SampleType),
    StorageTexture(TexelFormat),
    Sampler(bool),
    /// Result "type" of a call to a function without a return value.
    Void,
}

pub const F32: Ty = Ty::Scalar(Scalar::F32);
pub const I32: Ty = Ty::Scalar(Scalar::I32);
pub const U32: Ty = Ty::Scalar(Scalar::U32);
pub const BOOL: Ty = Ty::Scalar(Scalar::Bool);

impl Ty {
    /// Scalar component type of a scalar, vector or matrix.
    pub fn scalar(&self) -> Option<Scalar> {
        match self {
            Ty::Scalar(s) | Ty::Vector(_, s) => Some(*s),
            Ty::Matrix(..) => Some(Scalar::F32),
            _ => None,
        }
    }
    pub fn vec_size(&self) -> Option<u8> {
        match self {
            Ty::Vector(n, _) => Some(*n),
            _ => None,
        }
    }
    pub fn is_scalar(&self) -> bool {
        matches!(self, Ty::Scalar(_))
    }
    pub fn is_scalar_or_vector(&self) -> bool {
        matches!(self, Ty::Scalar(_) | Ty::Vector(..))
    }
    pub fn is_abstract(&self) -> bool {
        match self {
            Ty::Scalar(s) | Ty::Vector(_, s) => s.is_abstract(),
            Ty::Array(e, _) => e.is_abstract(),
            _ => false,
        }
    }
    pub fn concretize(&self) -> Ty {
        match self {
            Ty::Scalar(s) => Ty::Scalar(s.concretize()),
            Ty::Vector(n, s) => Ty::Vector(*n, s.concretize()),
            Ty::Array(e, n) => Ty::Array(Box::new(e.concretize()), *n),
            t => t.clone(),
        }
    }
    /// Same shape (scalar/vector/array of those) with a different leaf scalar.
    pub fn with_scalar(&self, s: Scalar) -> Ty {
        match self {
            Ty::Scalar(_) => Ty::Scalar(s),
            Ty::Vector(n, _) => Ty::Vector(*n, s),
            Ty::Array(e, n) => Ty::Array(Box::new(e.with_scalar(s)), *n),
            t => t.clone(),
        }
    }
    /// The innermost non-array type.
    pub fn innermost(&self) -> &Ty {
        match self {
            Ty::Array(e, _) => e.innermost(),
            t => t,
        }
    }
    pub fn is_handle(&self) -> bool {
        matches!(
            self,
            Ty::Texture(..) | Ty::StorageTexture(_) | Ty::Sampler(_)
        )
    }
    /// Whether this type (transitively) contains structs or arrays, i.e. whether its SPIR-V
    /// type differs between plain and explicitly laid-out variants.
    pub fn has_layout_variants(&self) -> bool {
        matches!(self, Ty::Array(..) | Ty::Struct(_))
    }
    /// Leaf scalar of an (array of) scalar/vector type.
    pub fn leaf_scalar(&self) -> Option<Scalar> {
        match self {
            Ty::Array(e, _) => e.leaf_scalar(),
            t => t.scalar(),
        }
    }
}

/// A compile-time constant. Composites hold vector components, matrix columns, array
/// elements or struct members in order; the type travels alongside.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstValue {
    Bool(bool),
    I32(i32),
    U32(u32),
    F32(f32),
    AbstractInt(i64),
    AbstractFloat(f64),
    Composite(Vec<ConstValue>),
}

impl ConstValue {
    pub fn zero(ty: &Ty, structs: &[StructInfo]) -> ConstValue {
        match ty {
            Ty::Scalar(s) => ConstValue::zero_scalar(*s),
            Ty::Vector(n, s) => {
                ConstValue::Composite(vec![ConstValue::zero_scalar(*s); *n as usize])
            }
            Ty::Matrix(c, r) => ConstValue::Composite(vec![
                ConstValue::Composite(
                    vec![ConstValue::F32(0.0); *r as usize]
                );
                *c as usize
            ]),
            Ty::Array(e, n) => {
                ConstValue::Composite(vec![ConstValue::zero(e, structs); n.unwrap_or(0) as usize])
            }
            Ty::Struct(id) => ConstValue::Composite(
                structs[*id]
                    .members
                    .iter()
                    .map(|m| ConstValue::zero(&m.ty, structs))
                    .collect(),
            ),
            _ => ConstValue::Bool(false),
        }
    }

    pub fn zero_scalar(s: Scalar) -> ConstValue {
        match s {
            Scalar::Bool => ConstValue::Bool(false),
            Scalar::I32 => ConstValue::I32(0),
            Scalar::U32 => ConstValue::U32(0),
            Scalar::F32 => ConstValue::F32(0.0),
            Scalar::AbstractInt => ConstValue::AbstractInt(0),
            Scalar::AbstractFloat => ConstValue::AbstractFloat(0.0),
        }
    }

    pub fn components(&self) -> &[ConstValue] {
        match self {
            ConstValue::Composite(v) => v,
            other => std::slice::from_ref(other),
        }
    }

    /// Integer value of an integer scalar constant.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            ConstValue::I32(v) => Some(v as i64),
            ConstValue::U32(v) => Some(v as i64),
            ConstValue::AbstractInt(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            ConstValue::Bool(b) => Some(b),
            _ => None,
        }
    }
}

/// IO attributes of a parameter, return value or struct member.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IoAttrs {
    pub location: Option<u32>,
    pub builtin: Option<BuiltinValue>,
    pub interpolation: Option<Interpolation>,
    pub sampling: Option<Sampling>,
    pub invariant: bool,
    pub span: Span,
}

impl IoAttrs {
    pub fn is_empty(&self) -> bool {
        self.location.is_none() && self.builtin.is_none()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinValue {
    VertexIndex,
    InstanceIndex,
    Position,
    FrontFacing,
    FragDepth,
    SampleIndex,
    LocalInvocationId,
    LocalInvocationIndex,
    GlobalInvocationId,
    WorkgroupId,
    NumWorkgroups,
}

impl BuiltinValue {
    pub fn from_name(name: &str) -> Option<BuiltinValue> {
        Some(match name {
            "vertex_index" => BuiltinValue::VertexIndex,
            "instance_index" => BuiltinValue::InstanceIndex,
            "position" => BuiltinValue::Position,
            "front_facing" => BuiltinValue::FrontFacing,
            "frag_depth" => BuiltinValue::FragDepth,
            "sample_index" => BuiltinValue::SampleIndex,
            "local_invocation_id" => BuiltinValue::LocalInvocationId,
            "local_invocation_index" => BuiltinValue::LocalInvocationIndex,
            "global_invocation_id" => BuiltinValue::GlobalInvocationId,
            "workgroup_id" => BuiltinValue::WorkgroupId,
            "num_workgroups" => BuiltinValue::NumWorkgroups,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            BuiltinValue::VertexIndex => "vertex_index",
            BuiltinValue::InstanceIndex => "instance_index",
            BuiltinValue::Position => "position",
            BuiltinValue::FrontFacing => "front_facing",
            BuiltinValue::FragDepth => "frag_depth",
            BuiltinValue::SampleIndex => "sample_index",
            BuiltinValue::LocalInvocationId => "local_invocation_id",
            BuiltinValue::LocalInvocationIndex => "local_invocation_index",
            BuiltinValue::GlobalInvocationId => "global_invocation_id",
            BuiltinValue::WorkgroupId => "workgroup_id",
            BuiltinValue::NumWorkgroups => "num_workgroups",
        }
    }

    pub fn ty(self) -> Ty {
        match self {
            BuiltinValue::VertexIndex
            | BuiltinValue::InstanceIndex
            | BuiltinValue::SampleIndex
            | BuiltinValue::LocalInvocationIndex => U32,
            BuiltinValue::Position => Ty::Vector(4, Scalar::F32),
            BuiltinValue::FrontFacing => BOOL,
            BuiltinValue::FragDepth => F32,
            BuiltinValue::LocalInvocationId
            | BuiltinValue::GlobalInvocationId
            | BuiltinValue::WorkgroupId
            | BuiltinValue::NumWorkgroups => Ty::Vector(3, Scalar::U32),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Interpolation {
    Perspective,
    Linear,
    Flat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sampling {
    Center,
    Centroid,
    Sample,
}

#[derive(Clone, Debug)]
pub struct MemberInfo {
    pub name: String,
    pub span: Span,
    pub ty: Ty,
    pub io: IoAttrs,
    pub align_attr: Option<u32>,
    pub size_attr: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct StructLayout {
    pub offsets: Vec<u32>,
    pub align: u32,
    pub size: u32,
}

#[derive(Clone, Debug)]
pub struct StructInfo {
    pub name: String,
    pub members: Vec<MemberInfo>,
    /// Host-shareable layout, if every member is host-shareable.
    pub layout: Option<StructLayout>,
    /// Nesting depth of the struct type (see [`type_depth`]).
    pub depth: u32,
}

/// Maximum nesting of arrays and structs inside one type.
pub const MAX_TYPE_DEPTH: u32 = 32;

/// Largest buffer-layout size accepted for a single type, in bytes.
pub const MAX_TYPE_SIZE: u64 = 1 << 30;

/// Nesting depth of a type: 1 for scalars, vectors, matrices and handles.
pub fn type_depth(ty: &Ty, structs: &[StructInfo]) -> u32 {
    match ty {
        Ty::Array(e, _) => 1 + type_depth(e, structs),
        Ty::Struct(id) => structs[*id].depth,
        _ => 1,
    }
}

/// Number of scalar leaves in a value of this type (saturating), used to avoid
/// materializing huge constants.
pub fn leaf_count(ty: &Ty, structs: &[StructInfo]) -> u64 {
    match ty {
        Ty::Vector(n, _) => *n as u64,
        Ty::Matrix(c, r) => (*c as u64) * (*r as u64),
        Ty::Array(e, n) => leaf_count(e, structs).saturating_mul(n.unwrap_or(0) as u64),
        Ty::Struct(id) => structs[*id]
            .members
            .iter()
            .fold(0u64, |a, m| a.saturating_add(leaf_count(&m.ty, structs))),
        _ => 1,
    }
}

pub fn round_up(align: u32, n: u32) -> u32 {
    n.div_ceil(align).saturating_mul(align)
}

/// WGSL `AlignOf` for host-shareable types.
pub fn align_of(ty: &Ty, structs: &[StructInfo]) -> u32 {
    match ty {
        Ty::Scalar(_) => 4,
        Ty::Vector(2, _) => 8,
        Ty::Vector(..) => 16,
        Ty::Matrix(_, rows) => align_of(&Ty::Vector(*rows, Scalar::F32), structs),
        Ty::Array(e, _) => align_of(e, structs),
        Ty::Struct(id) => structs[*id].layout.as_ref().map_or(16, |l| l.align),
        _ => 4,
    }
}

/// WGSL `SizeOf` for host-shareable types (runtime arrays count as zero elements).
pub fn size_of(ty: &Ty, structs: &[StructInfo]) -> u32 {
    match ty {
        Ty::Scalar(_) => 4,
        Ty::Vector(n, _) => 4 * *n as u32,
        Ty::Matrix(cols, rows) => *cols as u32 * matrix_stride(*rows),
        Ty::Array(e, n) => n.unwrap_or(0).saturating_mul(array_stride(e, structs)),
        Ty::Struct(id) => structs[*id].layout.as_ref().map_or(0, |l| l.size),
        _ => 0,
    }
}

pub fn array_stride(elem: &Ty, structs: &[StructInfo]) -> u32 {
    round_up(align_of(elem, structs), size_of(elem, structs))
}

/// Column stride of `matCxR<f32>`: roundUp(AlignOf(vecR), SizeOf(vecR)).
pub fn matrix_stride(rows: u8) -> u32 {
    let v = Ty::Vector(rows, Scalar::F32);
    round_up(align_of(&v, &[]), size_of(&v, &[]))
}

pub fn is_host_shareable(ty: &Ty, structs: &[StructInfo]) -> bool {
    match ty {
        Ty::Scalar(s) | Ty::Vector(_, s) => matches!(s, Scalar::I32 | Scalar::U32 | Scalar::F32),
        Ty::Matrix(..) => true,
        Ty::Array(e, _) => is_host_shareable(e, structs),
        Ty::Struct(id) => structs[*id].layout.is_some(),
        _ => false,
    }
}

/// Constructible types: scalars, vectors, matrices, fixed-size arrays and structs of those.
pub fn is_constructible(ty: &Ty, structs: &[StructInfo]) -> bool {
    match ty {
        Ty::Scalar(_) | Ty::Vector(..) | Ty::Matrix(..) => true,
        Ty::Array(e, Some(_)) => is_constructible(e, structs),
        Ty::Array(_, None) => false,
        Ty::Struct(id) => structs[*id]
            .members
            .iter()
            .all(|m| is_constructible(&m.ty, structs)),
        _ => false,
    }
}

/// Human-readable WGSL spelling of a type, for diagnostics.
pub fn type_name(ty: &Ty, structs: &[StructInfo]) -> String {
    match ty {
        Ty::Scalar(s) => s.name().to_string(),
        Ty::Vector(n, s) => format!("vec{n}<{}>", s.name()),
        Ty::Matrix(c, r) => format!("mat{c}x{r}<f32>"),
        Ty::Array(e, Some(n)) => format!("array<{}, {n}>", type_name(e, structs)),
        Ty::Array(e, None) => format!("array<{}>", type_name(e, structs)),
        Ty::Struct(id) => structs
            .get(*id)
            .map_or_else(|| "struct".into(), |s| s.name.clone()),
        Ty::Texture(view, sample) => {
            let v = match view {
                TextureView::D2 => "2d",
                TextureView::D2Array => "2d_array",
                TextureView::Cube => "cube",
            };
            match sample {
                SampleType::Depth => format!("texture_depth_{v}"),
                SampleType::Float => format!("texture_{v}<f32>"),
                SampleType::Sint => format!("texture_{v}<i32>"),
                SampleType::Uint => format!("texture_{v}<u32>"),
            }
        }
        Ty::StorageTexture(f) => format!("texture_storage_2d<{}, write>", texel_format_name(*f)),
        Ty::Sampler(false) => "sampler".into(),
        Ty::Sampler(true) => "sampler_comparison".into(),
        Ty::Void => "void".into(),
    }
}

pub fn texel_format_name(f: TexelFormat) -> &'static str {
    match f {
        TexelFormat::Rgba8Unorm => "rgba8unorm",
        TexelFormat::Rgba16Float => "rgba16float",
        TexelFormat::Rgba32Float => "rgba32float",
        TexelFormat::R32Float => "r32float",
        TexelFormat::Rg16Float => "rg16float",
        TexelFormat::R16Float => "r16float",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_vector_matrix_layout() {
        let s: &[StructInfo] = &[];
        assert_eq!((align_of(&F32, s), size_of(&F32, s)), (4, 4));
        let v2 = Ty::Vector(2, Scalar::F32);
        let v3 = Ty::Vector(3, Scalar::F32);
        let v4 = Ty::Vector(4, Scalar::U32);
        assert_eq!((align_of(&v2, s), size_of(&v2, s)), (8, 8));
        assert_eq!((align_of(&v3, s), size_of(&v3, s)), (16, 12));
        assert_eq!((align_of(&v4, s), size_of(&v4, s)), (16, 16));
        assert_eq!(matrix_stride(2), 8);
        assert_eq!(matrix_stride(3), 16);
        assert_eq!(size_of(&Ty::Matrix(3, 3), s), 48);
        assert_eq!(size_of(&Ty::Matrix(4, 4), s), 64);
        assert_eq!(size_of(&Ty::Matrix(2, 2), s), 16);
        assert_eq!(size_of(&Ty::Matrix(3, 2), s), 24);
        assert_eq!(align_of(&Ty::Matrix(2, 3), s), 16);
        assert_eq!(array_stride(&v3, s), 16);
        assert_eq!(array_stride(&F32, s), 4);
        assert_eq!(size_of(&Ty::Array(Box::new(v3), Some(4)), s), 64);
    }
}
