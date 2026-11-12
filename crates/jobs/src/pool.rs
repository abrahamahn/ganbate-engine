//! Zero-dependency work-stealing job pool with scoped zero-allocation parallel chunk dispatch.
//!
//! Each worker owns a double-ended queue (`VecDeque`): the owning worker pushes and pops from the
//! back (LIFO for warm cache lines), while idle workers steal from the front (FIFO to take the
//! oldest, largest remaining chunks). On `wasm32` or single-core hosts, jobs execute inline on the
//! caller thread with zero synchronization overhead.

use std::any::Any;
use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;

#[derive(Clone, Copy)]
struct JobEntry {
    run: unsafe fn(*const (), &PoolInner, usize, usize),
    ctx: *const (),
    job_idx: usize,
}

// SAFETY: `JobEntry` only wraps a `*const ()` pointing to a `Sync` `BatchCtx<F>` whose stack
// lifetime is strictly guarded by `BatchGuard` until `pending` reaches zero with `Acquire`
// synchronization.
unsafe impl Send for JobEntry {}

struct WorkerDeque {
    queue: Mutex<VecDeque<JobEntry>>,
}

impl WorkerDeque {
    fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::with_capacity(64)),
        }
    }

    #[inline]
    fn pop_local(&self) -> Option<JobEntry> {
        self.queue.lock().unwrap().pop_back()
    }

    #[inline]
    fn steal(&self) -> Option<JobEntry> {
        self.queue.lock().unwrap().pop_front()
    }
}

struct PoolInner {
    workers: Vec<WorkerDeque>,
    queued_jobs: AtomicUsize,
    shutdown: AtomicBool,
    park_lock: Mutex<()>,
    park_cv: Condvar,
    complete_lock: Mutex<()>,
    complete_cv: Condvar,
}

impl PoolInner {
    fn try_pop_or_steal(&self, worker_id: usize) -> Option<JobEntry> {
        if self.queued_jobs.load(Ordering::Acquire) == 0 {
            return None;
        }
        if let Some(job) = self.workers[worker_id].pop_local() {
            self.queued_jobs.fetch_sub(1, Ordering::AcqRel);
            return Some(job);
        }
        let n = self.workers.len();
        for offset in 1..n {
            let victim = (worker_id + offset) % n;
            if let Some(job) = self.workers[victim].steal() {
                self.queued_jobs.fetch_sub(1, Ordering::AcqRel);
                return Some(job);
            }
        }
        None
    }

    #[inline]
    fn try_run_one(&self, worker_id: usize) -> bool {
        if let Some(job) = self.try_pop_or_steal(worker_id) {
            // SAFETY: `job.ctx` remains valid until `BatchCtx::pending` reaches 0.
            unsafe { (job.run)(job.ctx, self, worker_id, job.job_idx) };
            true
        } else {
            false
        }
    }
}

/// Persistent work-stealing thread pool for data-parallel engine workloads (culling, skinning,
/// asset baking, scene transforms).
pub struct JobPool {
    inner: Arc<PoolInner>,
    handles: Vec<JoinHandle<()>>,
}

impl Default for JobPool {
    fn default() -> Self {
        Self::new(Self::default_threads())
    }
}

impl JobPool {
    /// Hardware concurrency clamped to `1..=32` (`1` on `wasm32`).
    pub fn default_threads() -> usize {
        #[cfg(target_arch = "wasm32")]
        {
            1
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::available_parallelism()
                .map(|n| n.get().clamp(1, 32))
                .unwrap_or(1)
        }
    }

    /// Creates a job pool with `threads` total participating workers (`threads - 1` background OS
    /// threads plus the calling thread as worker `0`).
    pub fn new(threads: usize) -> Self {
        #[cfg(target_arch = "wasm32")]
        let total = {
            let _ = threads;
            1usize
        };
        #[cfg(not(target_arch = "wasm32"))]
        let total = threads.clamp(1, 64);

        let workers = (0..total).map(|_| WorkerDeque::new()).collect();
        let inner = Arc::new(PoolInner {
            workers,
            queued_jobs: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            park_lock: Mutex::new(()),
            park_cv: Condvar::new(),
            complete_lock: Mutex::new(()),
            complete_cv: Condvar::new(),
        });

        #[cfg(target_arch = "wasm32")]
        let handles = Vec::new();
        #[cfg(not(target_arch = "wasm32"))]
        let handles = {
            let mut h = Vec::with_capacity(total.saturating_sub(1));
            for worker_id in 1..total {
                let pool = Arc::clone(&inner);
                let handle = std::thread::Builder::new()
                    .name(format!("ganbate-job-{worker_id}"))
                    .spawn(move || worker_loop(pool, worker_id))
                    .expect("failed to spawn ganbate-jobs worker thread");
                h.push(handle);
            }
            h
        };

        Self { inner, handles }
    }

    /// Total worker count (including caller participation slot `0`).
    #[inline]
    pub fn worker_count(&self) -> usize {
        self.inner.workers.len()
    }

    /// Splits `0..len` into chunks of at least `min_chunk_size` items and executes `body(worker_id, range)`
    /// across all workers using work stealing. Zero heap allocations per call in steady state.
    pub fn par_for_chunks<F>(&self, len: usize, min_chunk_size: usize, body: F)
    where
        F: Fn(usize, Range<usize>) + Sync,
    {
        if len == 0 {
            return;
        }
        let workers = self.inner.workers.len();
        let min_chunk = min_chunk_size.max(1);
        if workers <= 1 || len <= min_chunk {
            body(0, 0..len);
            return;
        }

        // Aim for ~4 chunks per worker so work-stealing balances uneven chunk costs smoothly.
        let target_chunks = (workers * 4).max(1);
        let chunk_size = len.div_ceil(target_chunks).max(min_chunk);
        let num_chunks = len.div_ceil(chunk_size);
        if num_chunks <= 1 {
            body(0, 0..len);
            return;
        }

        let batch = BatchCtx {
            body: &body,
            len,
            chunk_size,
            pending: AtomicUsize::new(num_chunks),
            panicked: AtomicBool::new(false),
            panic_payload: Mutex::new(None),
        };

        let ctx_ptr = (&batch as *const BatchCtx<'_, F>).cast::<()>();

        // Guard ensures the stack frame cannot unwind or return until all `num_chunks` jobs
        // have decremented `batch.pending` to 0.
        let guard = BatchGuard {
            pool: &self.inner,
            pending: &batch.pending,
        };

        // Distribute chunks across worker deques so each worker starts with local work, while
        // any worker that finishes early steals from slower siblings.
        {
            let mut pushed_per_worker = 0usize;
            for job_idx in 0..num_chunks {
                let target_worker = job_idx % workers;
                let mut q = self.inner.workers[target_worker].queue.lock().unwrap();
                q.push_back(JobEntry {
                    run: run_chunk_job::<F>,
                    ctx: ctx_ptr,
                    job_idx,
                });
                pushed_per_worker += 1;
            }
            self.inner
                .queued_jobs
                .fetch_add(pushed_per_worker, Ordering::Release);
        }

        // Wake only as many parked background workers as needed for `num_chunks - 1` (since
        // caller participates as worker 0), avoiding waking idle SMT threads unnecessarily.
        {
            let _g = self.inner.park_lock.lock().unwrap();
            let wake_needed = num_chunks.saturating_sub(1);
            if wake_needed >= workers.saturating_sub(1) {
                self.inner.park_cv.notify_all();
            } else {
                for _ in 0..wake_needed {
                    self.inner.park_cv.notify_one();
                }
            }
        }

        // Caller participates as worker 0 until all chunks complete.
        guard.wait_until_complete();
        drop(guard);

        if batch.panicked.load(Ordering::Acquire)
            && let Some(payload) = batch.panic_payload.lock().unwrap().take()
        {
            resume_unwind(payload);
        }
    }

    /// Maps `items` in parallel into a freshly allocated `Vec<R>` with a single heap allocation.
    pub fn par_map<T, R, F>(&self, items: &[T], min_chunk_size: usize, map: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Sync,
    {
        let len = items.len();
        if len == 0 {
            return Vec::new();
        }
        let mut out: Vec<MaybeUninit<R>> = Vec::with_capacity(len);
        // SAFETY: `MaybeUninit<R>` requires no initialization; each index `0..len` is written
        // exactly once by disjoint `range` chunks before `assume_init` below.
        unsafe {
            out.set_len(len);
        }
        let out_ptr = SyncMutPtr(out.as_mut_ptr());
        self.par_for_chunks(len, min_chunk_size, |_, range| {
            let base = out_ptr.ptr();
            for i in range {
                let val = map(&items[i]);
                // SAFETY: `range` chunks partition `0..len` into pairwise-disjoint index sets.
                unsafe {
                    base.add(i).write(MaybeUninit::new(val));
                }
            }
        });
        // SAFETY: `par_for_chunks` completed without panicking, so every slot in `0..len` has
        // been initialized with a valid `R`.
        unsafe {
            let mut manual = std::mem::ManuallyDrop::new(out);
            Vec::from_raw_parts(manual.as_mut_ptr().cast::<R>(), len, manual.capacity())
        }
    }
}

impl Drop for JobPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        {
            let _g = self.inner.park_lock.lock().unwrap();
            self.inner.park_cv.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[derive(Clone, Copy)]
struct SyncMutPtr<T>(*mut T);

impl<T> SyncMutPtr<T> {
    #[inline]
    fn ptr(&self) -> *mut T {
        self.0
    }
}
// SAFETY: Disjoint index ownership is enforced by `par_map` chunk partitioning.
unsafe impl<T: Send> Sync for SyncMutPtr<T> {}
unsafe impl<T: Send> Send for SyncMutPtr<T> {}

struct BatchCtx<'a, F> {
    body: &'a F,
    len: usize,
    chunk_size: usize,
    pending: AtomicUsize,
    panicked: AtomicBool,
    panic_payload: Mutex<Option<Box<dyn Any + Send + 'static>>>,
}

struct BatchGuard<'a> {
    pool: &'a PoolInner,
    pending: &'a AtomicUsize,
}

impl BatchGuard<'_> {
    fn wait_until_complete(&self) {
        while self.pending.load(Ordering::Acquire) != 0 {
            if !self.pool.try_run_one(0) {
                let guard = self.pool.complete_lock.lock().unwrap();
                if self.pending.load(Ordering::Acquire) != 0
                    && self.pool.queued_jobs.load(Ordering::Acquire) == 0
                {
                    drop(self.pool.complete_cv.wait(guard).unwrap());
                }
            }
        }
    }
}

impl Drop for BatchGuard<'_> {
    fn drop(&mut self) {
        self.wait_until_complete();
    }
}

unsafe fn run_chunk_job<F>(ctx_ptr: *const (), pool: &PoolInner, worker_id: usize, job_idx: usize)
where
    F: Fn(usize, Range<usize>) + Sync,
{
    // SAFETY: `ctx_ptr` points to a valid `BatchCtx<F>` until `pending.fetch_sub(1, AcqRel)` below.
    let ctx = unsafe { &*ctx_ptr.cast::<BatchCtx<'_, F>>() };
    if !ctx.panicked.load(Ordering::Relaxed) {
        let start = job_idx * ctx.chunk_size;
        let end = (start + ctx.chunk_size).min(ctx.len);
        if start < end {
            let res = catch_unwind(AssertUnwindSafe(|| (ctx.body)(worker_id, start..end)));
            if let Err(payload) = res {
                ctx.panicked.store(true, Ordering::Release);
                let mut slot = ctx.panic_payload.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(payload);
                }
            }
        }
    }
    // CRITICAL: Never dereference `ctx` after this `fetch_sub`!
    if ctx.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
        let _g = pool.complete_lock.lock().unwrap();
        pool.complete_cv.notify_all();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn worker_loop(pool: Arc<PoolInner>, worker_id: usize) {
    let mut spin = 0u32;
    loop {
        if pool.try_run_one(worker_id) {
            spin = 0;
            continue;
        }
        if pool.shutdown.load(Ordering::Acquire) {
            break;
        }
        if spin < 32 {
            spin += 1;
            std::hint::spin_loop();
            continue;
        }
        spin = 0;
        let guard = pool.park_lock.lock().unwrap();
        if pool.shutdown.load(Ordering::Acquire) {
            break;
        }
        if pool.queued_jobs.load(Ordering::Acquire) == 0 {
            drop(pool.park_cv.wait(guard).unwrap());
        }
    }
}

/// Returns the shared process-wide work-stealing [`JobPool`].
pub fn global_pool() -> &'static JobPool {
    static GLOBAL: OnceLock<JobPool> = OnceLock::new();
    GLOBAL.get_or_init(JobPool::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_and_multi_thread_par_map_match_sequential() {
        let input: Vec<u32> = (0..4097).collect();
        let expected: Vec<u64> = input
            .iter()
            .map(|&x| u64::from(x) * u64::from(x) + 7)
            .collect();

        let single = JobPool::new(1);
        assert_eq!(single.worker_count(), 1);
        assert_eq!(
            single.par_map(&input, 64, |&x| u64::from(x) * u64::from(x) + 7),
            expected
        );

        let multi = JobPool::new(4);
        assert_eq!(multi.worker_count(), 4);
        assert_eq!(
            multi.par_map(&input, 64, |&x| u64::from(x) * u64::from(x) + 7),
            expected
        );
    }

    #[test]
    fn work_stealing_completes_skewed_and_nested_jobs() {
        let pool = JobPool::new(4);
        let sum = AtomicUsize::new(0);
        pool.par_for_chunks(256, 4, |_, range| {
            // Nested parallel dispatch on the same pool must never deadlock.
            let inner_items: Vec<usize> = range.collect();
            let partial = pool.par_map(&inner_items, 2, |&v| {
                if v == 0 {
                    // Skewed slow item so other workers steal the remaining queues.
                    for _ in 0..1_000 {
                        std::hint::spin_loop();
                    }
                }
                v * 2
            });
            let local_sum: usize = partial.into_iter().sum();
            sum.fetch_add(local_sum, Ordering::Relaxed);
        });
        let expected: usize = (0..256).map(|v| v * 2).sum();
        assert_eq!(sum.load(Ordering::Relaxed), expected);
    }

    #[test]
    fn worker_panic_propagates_cleanly_to_caller() {
        let pool = JobPool::new(4);
        let res = catch_unwind(AssertUnwindSafe(|| {
            pool.par_for_chunks(128, 4, |_, range| {
                if range.contains(&77) {
                    panic!("expected test panic in chunk");
                }
            });
        }));
        assert!(res.is_err());

        // Pool remains healthy after recovering from a panicked batch.
        let out = pool.par_map(&[1, 2, 3, 4], 1, |&x| x * 10);
        assert_eq!(out, vec![10, 20, 30, 40]);
    }
}
