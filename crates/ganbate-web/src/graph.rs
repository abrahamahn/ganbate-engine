//! The semantic scene graph (`ganbate-graph`) for the host page, so an editor or a running game
//! in the browser commits patches through the same solver as the native tools and agents.
use std::cell::RefCell;
use std::collections::BTreeMap;

use ganbate_engine::json::{self, Value};
use ganbate_graph::Graph;
use ganbate_graph::patch::Patch;
use ganbate_graph::spatial::{Commit, Mode, check, commit};

thread_local! {
    /// The outcome of the last [`ganbate_graph_commit`], as JSON.
    static OUTCOME: RefCell<String> = const { RefCell::new(String::new()) };
}

fn outcome(done: Commit) -> Result<Value, String> {
    Ok(Value::Object(BTreeMap::from([
        ("digest".to_string(), Value::String(done.graph.digest()?)),
        ("graph".into(), done.graph.to_json()),
        (
            "repairs".into(),
            Value::Array(done.repairs.into_iter().map(Value::String).collect()),
        ),
    ])))
}

fn graph(bytes: &[u8]) -> Result<Graph, String> {
    Graph::from_json(&json::parse(bytes)?)
}

/// Keeps `result` (or its error) for [`ganbate_graph_result`]; returns its length.
fn keep(result: Result<Value, String>) -> u32 {
    let value = result.unwrap_or_else(|e| {
        Value::Object(BTreeMap::from([("error".to_string(), Value::String(e))]))
    });
    let text = json::to_compact(&value).unwrap_or_default();
    OUTCOME.with(|cell| {
        *cell.borrow_mut() = text;
        cell.borrow().len() as u32
    })
}

fn mode(reject: u32) -> Mode {
    if reject == 0 {
        Mode::Project
    } else {
        Mode::Reject
    }
}

/// # Safety
/// `ptr` must address `len` readable bytes (any pointer when `len` is 0).
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        // SAFETY: the caller passes `len` readable bytes at `ptr`.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// Commits the patch at `patch` (JSON, `patch_len` bytes) to the graph at `graph` (JSON,
/// `graph_len` bytes), repairing what breaks the solver's rules (`reject` 0) or refusing the patch
/// (`reject` 1). Keeps the outcome — `{"graph", "digest", "repairs"}`, or `{"error"}` — for
/// [`ganbate_graph_result`] and returns its length.
///
/// # Safety
/// `graph` and `patch` must address `graph_len` and `patch_len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_graph_commit(
    graph: *const u8,
    graph_len: usize,
    patch: *const u8,
    patch_len: usize,
    reject: u32,
) -> u32 {
    // SAFETY: forwarded from the caller.
    let (graph_text, patch_text) = unsafe { (bytes(graph, graph_len), bytes(patch, patch_len)) };
    keep((|| {
        let patch = Patch::from_json(&json::parse(patch_text)?)?;
        outcome(commit(&self::graph(graph_text)?, &patch, mode(reject))?)
    })())
}

/// Checks the whole graph at `next` (JSON, `next_len` bytes), as written by hand or received from
/// another participant, against `previous` (JSON; `previous_len` 0 when opening a graph): see
/// `ganbate_graph::spatial::check`. Keeps the outcome like [`ganbate_graph_commit`].
///
/// # Safety
/// `previous` and `next` must address `previous_len` and `next_len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_graph_check(
    previous: *const u8,
    previous_len: usize,
    next: *const u8,
    next_len: usize,
    reject: u32,
) -> u32 {
    // SAFETY: forwarded from the caller.
    let (previous_text, next_text) =
        unsafe { (bytes(previous, previous_len), bytes(next, next_len)) };
    keep((|| {
        let previous = (!previous_text.is_empty())
            .then(|| graph(previous_text))
            .transpose()?;
        outcome(check(previous.as_ref(), &graph(next_text)?, mode(reject))?)
    })())
}

/// Copies the outcome of the last [`ganbate_graph_commit`] to `out` when it fits in `cap` bytes;
/// returns its length.
///
/// # Safety
/// `out` must address `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_graph_result(out: *mut u8, cap: usize) -> u32 {
    OUTCOME.with(|cell| {
        // SAFETY: forwarded from the caller.
        unsafe { crate::give(cell.borrow().as_bytes(), out, cap) }
    })
}
