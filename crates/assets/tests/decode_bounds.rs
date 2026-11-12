//! The GBEW decoder reserves memory only for what its input can hold: hostile counts fail
//! before anything is allocated for them. A test binary of its own, since it watches every
//! allocation through the global allocator.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use ganbate_assets::{cooked, zlib};

/// The system allocator, recording the largest single request.
struct Largest;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is passed unchanged to the system allocator, which upholds the contract;
// recording a size touches only an atomic.
unsafe impl GlobalAlloc for Largest {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        // SAFETY: the caller's layout is passed through as the contract requires.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator (that is, from `System`) with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        // SAFETY: as for `dealloc`, with the caller's new size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Largest = Largest;

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// An GBEW file of `version` with one mesh (`n` vertices, `k` indices, `flags`) whose body
/// follows, and nothing else in the world.
fn one_mesh(version: u32, n: u32, k: u32, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut f = b"GBEW".to_vec();
    f.extend(words(&[version, 1, 0, 0, 0, 0, n, k, flags]));
    f.extend(words(&[1f32.to_bits(); 3]));
    f.extend(words(&[u32::MAX]));
    f.extend_from_slice(body);
    f
}

/// A coded mesh's stream: its length and the zlib stream of `raw`, zero-padded.
fn stream(raw: &[u8]) -> Vec<u8> {
    let z = zlib::compress(raw);
    let mut s = words(&[z.len() as u32]);
    s.extend(&z);
    s.resize(s.len().next_multiple_of(4), 0);
    s
}

#[test]
fn hostile_counts_fail_before_memory_is_reserved_for_them() {
    let coded = 32;
    let mut texture = b"GBEW".to_vec();
    texture.extend(words(&[3, 0, 0, 0, 1, 0, 1, 1, 0, 1, 9]));
    texture.extend(b"image/png\0\0\0");
    texture.extend(words(&[u32::MAX]));
    let mut entities = b"GBEW".to_vec();
    entities.extend(words(&[4, 0, 0, 0, 0, 0, 250_000]));
    let files: [(&str, Vec<u8>); 7] = [
        // 2^28 - 1 indices claimed by a coded mesh whose stream holds three bytes.
        (
            "coded indices",
            one_mesh(4, 3, 268_435_455, coded, &stream(&[0, 0, 0])),
        ),
        // 30 million vertices and indices: more than a file decodes to.
        (
            "coded vertices",
            one_mesh(3, 30_000_000, 30_000_000, coded, &stream(&[0; 16])),
        ),
        (
            "coded, short stream",
            one_mesh(3, 3_000_000, 3_000_000, coded, &stream(&[0; 64])),
        ),
        // A small stream that inflates to a mebibyte for a mesh of three vertices.
        (
            "coded, inflating",
            one_mesh(3, 3, 3, coded, &stream(&[0; 1 << 20])),
        ),
        (
            "plain mesh",
            one_mesh(3, 0x0fff_ffff, 0x0fff_ffff, 0, &[0; 16]),
        ),
        ("encoded image", texture),
        ("entity table", entities),
    ];
    let mut problems = vec![];
    for (what, file) in files {
        LARGEST.store(0, Ordering::Relaxed);
        let result = cooked::decode_scene(&file);
        let largest = LARGEST.load(Ordering::Relaxed);
        if result.is_ok() || largest > 1 << 20 {
            problems.push(format!(
                "{what}: a {}-byte file made the decoder ask for {largest} bytes ({:?})",
                file.len(),
                result.map(|_| ())
            ));
        }
    }
    assert!(problems.is_empty(), "{problems:#?}");
}
