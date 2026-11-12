//! Stack-inline scratch buffer that spills to a `Vec` only past `N` items.

/// Holds up to `N` `Copy` elements on the stack without heap allocation, spilling to `Vec` when
/// more than `N` elements are pushed.
pub(crate) struct SmallBuf<T: Copy, const N: usize> {
    inline: [T; N],
    len: usize,
    spill: Vec<T>,
}

impl<T: Copy, const N: usize> SmallBuf<T, N> {
    #[inline]
    pub fn new(zero: T) -> Self {
        Self {
            inline: [zero; N],
            len: 0,
            spill: Vec::new(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0 && self.spill.is_empty()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
        self.spill.clear();
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        if !self.spill.is_empty() {
            self.spill.push(value);
        } else if self.len < N {
            self.inline[self.len] = value;
            self.len += 1;
        } else {
            self.spill.reserve(N * 2);
            self.spill.extend_from_slice(&self.inline[..self.len]);
            self.spill.push(value);
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.spill.is_empty() {
            &self.inline[..self.len]
        } else {
            &self.spill
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.spill.is_empty() {
            &mut self.inline[..self.len]
        } else {
            &mut self.spill
        }
    }
}

impl<T: Copy + Ord, const N: usize> SmallBuf<T, N> {
    #[inline]
    pub fn sort_dedup(&mut self) {
        if self.spill.is_empty() {
            let s = &mut self.inline[..self.len];
            s.sort_unstable();
            let mut w = 0usize;
            for r in 0..s.len() {
                if w == 0 || s[r] != s[w - 1] {
                    s[w] = s[r];
                    w += 1;
                }
            }
            self.len = w;
        } else {
            self.spill.sort_unstable();
            self.spill.dedup();
        }
    }
}
