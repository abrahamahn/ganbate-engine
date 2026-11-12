//! Identifiers for authored things: entities (64 bits) and assets (128 bits), written as
//! lowercase hexadecimal. They are random, so two people adding entities on separate branches
//! never pick the same one, and they never change when a thing is renamed or moved.

use crate::digest::hex;

/// Hex digits in an entity id.
pub const ENTITY_ID_LEN: usize = 16;
/// Hex digits in an asset id.
pub const ASSET_ID_LEN: usize = 32;

#[inline]
fn lower_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}
#[inline]
pub fn is_entity_id(s: &str) -> bool {
    lower_hex(s, ENTITY_ID_LEN)
}
#[inline]
pub fn is_asset_id(s: &str) -> bool {
    lower_hex(s, ASSET_ID_LEN)
}

/// The id of an entity that a prefab instance shows: the instance's id and the prefab
/// entity's id always give the same id, so overrides of it and references to it survive
/// reloads and edits of the prefab.
pub fn derived(instance: &str, inner: &str) -> String {
    let mut h = crate::digest::Sha256::new();
    h.update(instance.as_bytes());
    h.update(b"/");
    h.update(inner.as_bytes());
    hex(&h.finish()[..ENTITY_ID_LEN / 2])
}

/// A source of identifiers (SplitMix64). `from_os` seeds it from the operating system;
/// `seeded` repeats a sequence, for tests and for hosts that supply their own entropy (a
/// browser's `crypto.getRandomValues`).
#[derive(Clone, Debug)]
pub struct Ids {
    state: u64,
}

impl Ids {
    pub fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn from_os() -> Self {
        use std::hash::{BuildHasher, Hasher};
        use std::io::Read;
        let mut bytes = [0u8; 8];
        let urandom = std::fs::File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut bytes))
            .is_ok();
        // Elsewhere (Windows) std's hash keys come from the system's generator.
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write(&bytes);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        h.write_u128(now);
        h.write_u64(u64::from(std::process::id()));
        let seed = if urandom {
            u64::from_le_bytes(bytes) ^ h.finish().rotate_left(17)
        } else {
            h.finish()
        };
        Self::seeded(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn entity(&mut self) -> String {
        hex(&self.next_u64().to_be_bytes())
    }
    pub fn asset(&mut self) -> String {
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&self.next_u64().to_be_bytes());
        b[8..].copy_from_slice(&self.next_u64().to_be_bytes());
        hex(&b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_have_their_shape_and_do_not_repeat() {
        let mut ids = Ids::seeded(7);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..10_000 {
            let e = ids.entity();
            assert!(is_entity_id(&e), "{e}");
            assert!(seen.insert(e));
        }
        let a = ids.asset();
        assert!(is_asset_id(&a) && !is_entity_id(&a));
        assert!(!is_entity_id("0123456789ABCDEF"));
        assert_eq!(Ids::seeded(7).entity(), Ids::seeded(7).entity());
    }

    #[test]
    fn derived_ids_are_stable_and_distinct() {
        let a = derived("00000000000000aa", "00000000000000bb");
        assert!(is_entity_id(&a));
        assert_eq!(a, derived("00000000000000aa", "00000000000000bb"));
        assert_ne!(a, derived("00000000000000bb", "00000000000000aa"));
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn os_seeded_sources_differ() {
        assert_ne!(Ids::from_os().entity(), Ids::from_os().entity());
    }
}
