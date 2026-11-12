//! Zero-dependency work-stealing job pool, SIMD SoA frustum/distance/subpixel culling,
//! and linear-time `O(N)` Early-Z front-to-back depth radix sort.

pub mod cull;
pub mod pool;

pub use cull::{
    Aabb, AabbSoa8, CullConfig, CullStats, Frustum, Plane, cull_frustum_soa,
    cull_frustum_soa_scalar, pack_aabbs_soa, par_cull_frustum, par_cull_frustum_into,
    sort_front_to_back_radix, transform_aabb,
};
pub use pool::{JobPool, global_pool};
