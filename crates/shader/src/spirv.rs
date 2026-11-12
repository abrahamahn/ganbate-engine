//! SPIR-V 1.0 encoding: opcode and enumerant numbers (taken from the Khronos
//! `spirv.core.grammar.json` and `extinst.glsl.std.450.grammar.json`) and a module builder
//! that keeps the logical-layout sections separate until assembly.

pub const MAGIC: u32 = 0x0723_0203;
pub const VERSION_1_0: u32 = 0x0001_0000;

pub mod op {
    pub const NAME: u16 = 5;
    pub const MEMBER_NAME: u16 = 6;
    pub const EXT_INST_IMPORT: u16 = 11;
    pub const EXT_INST: u16 = 12;
    pub const MEMORY_MODEL: u16 = 14;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
    pub const CAPABILITY: u16 = 17;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_MATRIX: u16 = 24;
    pub const TYPE_IMAGE: u16 = 25;
    pub const TYPE_SAMPLER: u16 = 26;
    pub const TYPE_SAMPLED_IMAGE: u16 = 27;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT_TRUE: u16 = 41;
    pub const CONSTANT_FALSE: u16 = 42;
    pub const CONSTANT: u16 = 43;
    pub const CONSTANT_COMPOSITE: u16 = 44;
    pub const CONSTANT_NULL: u16 = 46;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_PARAMETER: u16 = 55;
    pub const FUNCTION_END: u16 = 56;
    pub const FUNCTION_CALL: u16 = 57;
    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const ACCESS_CHAIN: u16 = 65;
    pub const ARRAY_LENGTH: u16 = 68;
    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;
    pub const VECTOR_EXTRACT_DYNAMIC: u16 = 77;
    pub const VECTOR_SHUFFLE: u16 = 79;
    pub const COMPOSITE_CONSTRUCT: u16 = 80;
    pub const COMPOSITE_EXTRACT: u16 = 81;
    pub const TRANSPOSE: u16 = 84;
    pub const SAMPLED_IMAGE: u16 = 86;
    pub const IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
    pub const IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
    pub const IMAGE_SAMPLE_DREF_IMPLICIT_LOD: u16 = 89;
    pub const IMAGE_SAMPLE_DREF_EXPLICIT_LOD: u16 = 90;
    pub const IMAGE_FETCH: u16 = 95;
    pub const IMAGE_WRITE: u16 = 99;
    pub const IMAGE_QUERY_SIZE_LOD: u16 = 103;
    pub const IMAGE_QUERY_SIZE: u16 = 104;
    pub const IMAGE_QUERY_LEVELS: u16 = 106;
    pub const CONVERT_F_TO_U: u16 = 109;
    pub const CONVERT_F_TO_S: u16 = 110;
    pub const CONVERT_S_TO_F: u16 = 111;
    pub const CONVERT_U_TO_F: u16 = 112;
    pub const BITCAST: u16 = 124;
    pub const S_NEGATE: u16 = 126;
    pub const F_NEGATE: u16 = 127;
    pub const I_ADD: u16 = 128;
    pub const F_ADD: u16 = 129;
    pub const I_SUB: u16 = 130;
    pub const F_SUB: u16 = 131;
    pub const I_MUL: u16 = 132;
    pub const F_MUL: u16 = 133;
    pub const U_DIV: u16 = 134;
    pub const S_DIV: u16 = 135;
    pub const F_DIV: u16 = 136;
    pub const U_MOD: u16 = 137;
    pub const S_REM: u16 = 138;
    pub const F_REM: u16 = 140;
    pub const VECTOR_TIMES_SCALAR: u16 = 142;
    pub const MATRIX_TIMES_SCALAR: u16 = 143;
    pub const VECTOR_TIMES_MATRIX: u16 = 144;
    pub const MATRIX_TIMES_VECTOR: u16 = 145;
    pub const MATRIX_TIMES_MATRIX: u16 = 146;
    pub const DOT: u16 = 148;
    pub const ANY: u16 = 154;
    pub const ALL: u16 = 155;
    pub const LOGICAL_EQUAL: u16 = 164;
    pub const LOGICAL_NOT_EQUAL: u16 = 165;
    pub const LOGICAL_OR: u16 = 166;
    pub const LOGICAL_AND: u16 = 167;
    pub const LOGICAL_NOT: u16 = 168;
    pub const SELECT: u16 = 169;
    pub const I_EQUAL: u16 = 170;
    pub const I_NOT_EQUAL: u16 = 171;
    pub const U_GREATER_THAN: u16 = 172;
    pub const S_GREATER_THAN: u16 = 173;
    pub const U_GREATER_THAN_EQUAL: u16 = 174;
    pub const S_GREATER_THAN_EQUAL: u16 = 175;
    pub const U_LESS_THAN: u16 = 176;
    pub const S_LESS_THAN: u16 = 177;
    pub const U_LESS_THAN_EQUAL: u16 = 178;
    pub const S_LESS_THAN_EQUAL: u16 = 179;
    pub const F_ORD_EQUAL: u16 = 180;
    pub const F_UNORD_NOT_EQUAL: u16 = 183;
    pub const F_ORD_LESS_THAN: u16 = 184;
    pub const F_ORD_GREATER_THAN: u16 = 186;
    pub const F_ORD_LESS_THAN_EQUAL: u16 = 188;
    pub const F_ORD_GREATER_THAN_EQUAL: u16 = 190;
    pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const BITWISE_OR: u16 = 197;
    pub const BITWISE_XOR: u16 = 198;
    pub const BITWISE_AND: u16 = 199;
    pub const NOT: u16 = 200;
    pub const BIT_REVERSE: u16 = 204;
    pub const BIT_COUNT: u16 = 205;
    pub const DPDX: u16 = 207;
    pub const DPDY: u16 = 208;
    pub const FWIDTH: u16 = 209;
    pub const CONTROL_BARRIER: u16 = 224;
    pub const PHI: u16 = 245;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const KILL: u16 = 252;
    pub const RETURN: u16 = 253;
    pub const RETURN_VALUE: u16 = 254;
    pub const UNREACHABLE: u16 = 255;
}

pub mod capability {
    pub const SHADER: u32 = 1;
    pub const SAMPLE_RATE_SHADING: u32 = 35;
    pub const STORAGE_IMAGE_EXTENDED_FORMATS: u32 = 49;
    pub const IMAGE_QUERY: u32 = 50;
}

pub mod execution_model {
    pub const VERTEX: u32 = 0;
    pub const FRAGMENT: u32 = 4;
    pub const GL_COMPUTE: u32 = 5;
}

pub mod execution_mode {
    pub const ORIGIN_UPPER_LEFT: u32 = 7;
    pub const DEPTH_REPLACING: u32 = 12;
    pub const LOCAL_SIZE: u32 = 17;
}

pub const ADDRESSING_LOGICAL: u32 = 0;
pub const MEMORY_MODEL_GLSL450: u32 = 1;

pub mod storage_class {
    pub const UNIFORM_CONSTANT: u32 = 0;
    pub const INPUT: u32 = 1;
    pub const UNIFORM: u32 = 2;
    pub const OUTPUT: u32 = 3;
    pub const PRIVATE: u32 = 6;
    pub const FUNCTION: u32 = 7;
    pub const PUSH_CONSTANT: u32 = 9;
}

pub mod dim {
    pub const D2: u32 = 1;
    pub const CUBE: u32 = 3;
}

pub mod image_format {
    pub const UNKNOWN: u32 = 0;
    pub const RGBA32F: u32 = 1;
    pub const RGBA16F: u32 = 2;
    pub const R32F: u32 = 3;
    pub const RGBA8: u32 = 4;
    pub const RG16F: u32 = 7;
    pub const R16F: u32 = 9;
}

pub mod decoration {
    pub const BLOCK: u32 = 2;
    pub const BUFFER_BLOCK: u32 = 3;
    pub const COL_MAJOR: u32 = 5;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const MATRIX_STRIDE: u32 = 7;
    pub const BUILT_IN: u32 = 11;
    pub const NO_PERSPECTIVE: u32 = 13;
    pub const FLAT: u32 = 14;
    pub const CENTROID: u32 = 16;
    pub const SAMPLE: u32 = 17;
    pub const INVARIANT: u32 = 18;
    pub const NON_WRITABLE: u32 = 24;
    pub const NON_READABLE: u32 = 25;
    pub const LOCATION: u32 = 30;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
}

pub mod builtin {
    pub const POSITION: u32 = 0;
    pub const FRAG_COORD: u32 = 15;
    pub const FRONT_FACING: u32 = 17;
    pub const SAMPLE_ID: u32 = 18;
    pub const FRAG_DEPTH: u32 = 22;
    pub const NUM_WORKGROUPS: u32 = 24;
    pub const WORKGROUP_ID: u32 = 26;
    pub const LOCAL_INVOCATION_ID: u32 = 27;
    pub const GLOBAL_INVOCATION_ID: u32 = 28;
    pub const LOCAL_INVOCATION_INDEX: u32 = 29;
    pub const VERTEX_INDEX: u32 = 42;
    pub const INSTANCE_INDEX: u32 = 43;
}

pub mod image_operands {
    pub const BIAS: u32 = 0x1;
    pub const LOD: u32 = 0x2;
    pub const GRAD: u32 = 0x4;
    pub const CONST_OFFSET: u32 = 0x8;
}

pub mod scope {
    pub const DEVICE: u32 = 1;
    pub const WORKGROUP: u32 = 2;
}

pub mod semantics {
    pub const ACQUIRE_RELEASE: u32 = 0x8;
    pub const UNIFORM_MEMORY: u32 = 0x40;
    pub const WORKGROUP_MEMORY: u32 = 0x100;
}

/// GLSL.std.450 extended instruction numbers.
pub mod glsl {
    pub const ROUND_EVEN: u32 = 2;
    pub const TRUNC: u32 = 3;
    pub const F_ABS: u32 = 4;
    pub const S_ABS: u32 = 5;
    pub const F_SIGN: u32 = 6;
    pub const S_SIGN: u32 = 7;
    pub const FLOOR: u32 = 8;
    pub const CEIL: u32 = 9;
    pub const FRACT: u32 = 10;
    pub const RADIANS: u32 = 11;
    pub const DEGREES: u32 = 12;
    pub const SIN: u32 = 13;
    pub const COS: u32 = 14;
    pub const TAN: u32 = 15;
    pub const ASIN: u32 = 16;
    pub const ACOS: u32 = 17;
    pub const ATAN: u32 = 18;
    pub const SINH: u32 = 19;
    pub const COSH: u32 = 20;
    pub const TANH: u32 = 21;
    pub const ASINH: u32 = 22;
    pub const ACOSH: u32 = 23;
    pub const ATANH: u32 = 24;
    pub const ATAN2: u32 = 25;
    pub const POW: u32 = 26;
    pub const EXP: u32 = 27;
    pub const LOG: u32 = 28;
    pub const EXP2: u32 = 29;
    pub const LOG2: u32 = 30;
    pub const SQRT: u32 = 31;
    pub const INVERSE_SQRT: u32 = 32;
    pub const DETERMINANT: u32 = 33;
    pub const F_MIN: u32 = 37;
    pub const U_MIN: u32 = 38;
    pub const S_MIN: u32 = 39;
    pub const F_MAX: u32 = 40;
    pub const U_MAX: u32 = 41;
    pub const S_MAX: u32 = 42;
    pub const F_CLAMP: u32 = 43;
    pub const U_CLAMP: u32 = 44;
    pub const S_CLAMP: u32 = 45;
    pub const F_MIX: u32 = 46;
    pub const STEP: u32 = 48;
    pub const SMOOTH_STEP: u32 = 49;
    pub const FMA: u32 = 50;
    pub const LDEXP: u32 = 53;
    pub const PACK_SNORM_4X8: u32 = 54;
    pub const PACK_UNORM_4X8: u32 = 55;
    pub const PACK_SNORM_2X16: u32 = 56;
    pub const PACK_UNORM_2X16: u32 = 57;
    pub const PACK_HALF_2X16: u32 = 58;
    pub const UNPACK_SNORM_2X16: u32 = 60;
    pub const UNPACK_UNORM_2X16: u32 = 61;
    pub const UNPACK_HALF_2X16: u32 = 62;
    pub const UNPACK_SNORM_4X8: u32 = 63;
    pub const UNPACK_UNORM_4X8: u32 = 64;
    pub const LENGTH: u32 = 66;
    pub const DISTANCE: u32 = 67;
    pub const CROSS: u32 = 68;
    pub const NORMALIZE: u32 = 69;
    pub const FACE_FORWARD: u32 = 70;
    pub const REFLECT: u32 = 71;
    pub const REFRACT: u32 = 72;
}

/// Append one instruction.
pub fn inst(out: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    let count = operands.len() as u32 + 1;
    debug_assert!(count <= 0xFFFF, "instruction too long");
    out.push((count << 16) | u32::from(opcode));
    out.extend_from_slice(operands);
}

/// A literal string: UTF-8, nul-terminated, padded to a word boundary.
pub fn string(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Module sections in SPIR-V logical-layout order.
#[derive(Default)]
pub struct Sections {
    pub capabilities: Vec<u32>,
    pub ext_imports: Vec<u32>,
    pub memory_model: Vec<u32>,
    pub entry_points: Vec<u32>,
    pub execution_modes: Vec<u32>,
    pub debug: Vec<u32>,
    pub annotations: Vec<u32>,
    pub globals: Vec<u32>,
    pub functions: Vec<u32>,
}

impl Sections {
    pub fn assemble(self, bound: u32) -> Vec<u32> {
        let mut words = vec![MAGIC, VERSION_1_0, 0, bound, 0];
        for s in [
            self.capabilities,
            self.ext_imports,
            self.memory_model,
            self.entry_points,
            self.execution_modes,
            self.debug,
            self.annotations,
            self.globals,
            self.functions,
        ] {
            words.extend(s);
        }
        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_are_padded() {
        assert_eq!(string("main"), vec![u32::from_le_bytes(*b"main"), 0]);
        assert_eq!(string("abc"), vec![u32::from_le_bytes(*b"abc\0")]);
        let mut v = Vec::new();
        inst(&mut v, op::CAPABILITY, &[capability::SHADER]);
        assert_eq!(v, vec![(2 << 16) | 17, 1]);
    }
}
