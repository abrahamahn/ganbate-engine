//! The studio's network side: one thread per connection, up to a limit. Plain requests serve
//! the editor interface, the engine's web runtime, project files and cooked previews;
//! `/api/ws` upgrades to the edit protocol, whose messages go through `Studio::handle` under one
//! lock and out to each client through its own bounded queue and writer thread.
//!
//! Who is answered. Without a pairing token the host answers only requests addressed to this
//! machine by name (`Host: localhost`, `127.0.0.1` or `[::1]` with the host's port), so a web
//! page whose own name has been rebound to 127.0.0.1 is refused; and it opens the edit protocol
//! only for its own pages (`Origin: http://localhost:PORT` or `http://127.0.0.1:PORT`) or for
//! clients that are not browsers (no `Origin`), so no other site can drive it from the user's
//! browser. With a token (`--lan`), everything but the editor's own files needs it, as
//! `?token=` or as the cookie the host sets when a page is opened with the right `?token=`
//! (which is how the editor's fetches of `/engine/`, `/cooked/` and `/project/` carry it).
//! Wrong tokens are counted per peer address; a peer with too many waits before its next one
//! is even compared.
use std::collections::BTreeMap;
use std::io::{BufReader, ErrorKind, Read, Write};
use std::net::{IpAddr, Shutdown, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ganbate_core::id::Ids;
use ganbate_core::json::{self, Value};

use crate::http::{self, Request, Response};
use crate::session::{ClientId, Outbox, Studio};
use crate::ws;

pub struct Config {
    /// Folder of the editor interface (`index.html` and its modules).
    pub editor: PathBuf,
    /// Folder of the engine's web runtime (`ganbate.js`, `ganbate_web.wasm`), as `scripts/build-web.sh`
    /// builds it.
    pub engine: PathBuf,
    /// The pairing token every client must present when the host listens beyond this machine
    /// (`--lan`). Without one, the host answers only requests addressed to this machine.
    pub token: Option<String>,
    pub limits: Limits,
}

/// Bounds that keep a slow, idle or hostile client from tying up the host.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Connections served at once; more are closed as they arrive.
    pub connections: usize,
    /// Time a client has to send a whole request (request line, headers and body).
    pub request_time: Duration,
    /// How often the host pings an edit-protocol client ...
    pub ping_every: Duration,
    /// ... and how long it waits to hear anything from one (a pong at least) before giving up.
    pub idle_time: Duration,
    /// How long one write may wait for a client to take data.
    pub write_time: Duration,
    /// Messages, and bytes, that may wait for one client; a client further behind is dropped.
    pub queue_messages: usize,
    pub queue_bytes: usize,
    /// Wrong pairing tokens one peer address may send within `failure_window`; after that its
    /// tokens are not compared (the answer is 429) until the window has passed.
    pub failures: u32,
    pub failure_window: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            connections: 128,
            request_time: Duration::from_secs(10),
            ping_every: Duration::from_secs(30),
            idle_time: Duration::from_secs(90),
            write_time: Duration::from_secs(30),
            queue_messages: 4096,
            queue_bytes: 64 << 20,
            failures: 10,
            failure_window: Duration::from_secs(60),
        }
    }
}

/// Pause after a failed accept (most likely out of file descriptors) before trying again.
const ACCEPT_BACKOFF: Duration = Duration::from_millis(100);
/// Peer addresses whose wrong tokens are remembered at once.
const MAX_PEERS: usize = 4096;

/// A fresh pairing token: 128 bits from the operating system's random source (two 64-bit
/// draws, each from its own seed), as 32 hex digits.
pub fn new_token() -> String {
    format!(
        "{:016x}{:016x}",
        Ids::from_os().next_u64(),
        Ids::from_os().next_u64()
    )
}

/// Whether `given` is `token`. Every byte is compared whatever the first difference, so the
/// time taken says nothing about how much of a guess was right.
pub fn same_token(given: &str, token: &str) -> bool {
    let (g, t) = (given.as_bytes(), token.as_bytes());
    let mut diff = u8::from(g.len() != t.len());
    for (i, b) in t.iter().enumerate() {
        diff = std::hint::black_box(diff | (b ^ g.get(i).copied().unwrap_or(0)));
    }
    diff == 0
}

/// What a client's writer thread sends: a text message, or a whole frame (a pong) that the
/// reader produced. One writer per socket keeps frames from interleaving.
enum Out {
    Text(String),
    Frame(Vec<u8>),
}

impl Out {
    fn len(&self) -> usize {
        match self {
            Out::Text(t) => t.len(),
            Out::Frame(f) => f.len(),
        }
    }
    fn into_frame(self) -> Vec<u8> {
        match self {
            Out::Text(t) => ws::frame(0x1, t.as_bytes()),
            Out::Frame(f) => f,
        }
    }
}

/// Where one client's messages go: a queue bounded in messages and in bytes, read by the
/// client's writer thread. A client that falls too far behind is cut off: its socket is shut
/// down, which ends its reader and writer.
#[derive(Clone)]
struct Outlet {
    queue: SyncSender<Out>,
    /// Bytes waiting in `queue`.
    waiting: Arc<AtomicUsize>,
    socket: Arc<TcpStream>,
    max_bytes: usize,
}

impl Outlet {
    /// Queues `out` for the writer; false, with the client cut off, when it is too far behind
    /// (or already gone). A message larger than the byte budget still goes to an idle client.
    fn offer(&self, out: Out) -> bool {
        let size = out.len();
        let waiting = self.waiting.fetch_add(size, Ordering::SeqCst);
        let fits = waiting == 0 || waiting.saturating_add(size) <= self.max_bytes;
        if fits && self.queue.try_send(out).is_ok() {
            return true;
        }
        self.waiting.fetch_sub(size, Ordering::SeqCst);
        let _ = self.socket.shutdown(Shutdown::Both);
        false
    }
}

struct Shared {
    studio: Studio,
    clients: BTreeMap<ClientId, Outlet>,
}

/// What every connection thread reads: the configuration and the project's folders.
struct Host {
    config: Config,
    port: u16,
    /// The project folder as given, and with every symlink resolved.
    root: PathBuf,
    real_root: PathBuf,
    /// The cooked previews' folder (inside the project).
    cooked: PathBuf,
    /// The pairing cookie's name. Cookies are not kept apart by port, so the port is in it.
    cookie: String,
    /// Connections being served.
    live: AtomicUsize,
    failures: Mutex<Failures>,
}

/// One of the connections served at once, given back when dropped.
struct Slot(Arc<Host>);

impl Slot {
    fn take(host: &Arc<Host>) -> Option<Slot> {
        let max = host.config.limits.connections;
        host.live
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                (n < max).then_some(n + 1)
            })
            .ok()
            .map(|_| Slot(host.clone()))
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.live.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Serves until the process ends.
pub fn serve(studio: Studio, listener: TcpListener, config: Config) -> Result<(), String> {
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let host = Arc::new(Host {
        port,
        root: studio.project.root.clone(),
        real_root: studio.real_root().to_path_buf(),
        cooked: studio.cache_dir(),
        cookie: format!("ganbate-studio-{port}"),
        live: AtomicUsize::new(0),
        failures: Mutex::new(Failures::default()),
        config,
    });
    let shared = Arc::new(Mutex::new(Shared {
        studio,
        clients: BTreeMap::new(),
    }));
    loop {
        let (stream, peer) = match listener.accept() {
            Ok(c) => c,
            Err(e) => {
                // Most likely out of file descriptors: wait for some to close rather than
                // spin on the error.
                eprintln!("ganbate-studio: accept: {e}");
                std::thread::sleep(ACCEPT_BACKOFF);
                continue;
            }
        };
        // Over the limit, the connection is closed at once (the stream is dropped).
        let Some(slot) = Slot::take(&host) else {
            continue;
        };
        let (shared, host) = (shared.clone(), host.clone());
        let spawned = std::thread::Builder::new()
            .name("ganbate-studio connection".into())
            .spawn(move || {
                let _slot = slot;
                if let Err(e) = connection(stream, peer.ip(), &shared, &host) {
                    if e != "connection closed" {
                        eprintln!("ganbate-studio: {e}");
                    }
                }
            });
        // When no thread can start, the closure is dropped with its stream and slot: that
        // connection closes and the host goes on.
        if let Err(e) = spawned {
            eprintln!("ganbate-studio: cannot start a thread for a connection: {e}");
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic while holding the lock leaves the state as the last complete message left it.
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// The socket as the request reader sees it: while a request is read every read gives up at
/// `deadline`, so a client that trickles bytes cannot hold the connection; after that the
/// socket's own read timeout applies.
struct Timed {
    stream: TcpStream,
    deadline: Option<Instant>,
}

impl Read for Timed {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if let Some(d) = self.deadline {
            let left = d.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(std::io::Error::new(
                    ErrorKind::TimedOut,
                    "the request took too long",
                ));
            }
            self.stream.set_read_timeout(Some(left))?;
        }
        self.stream.read(buf)
    }
}

fn respond(w: &mut TcpStream, resp: Response) -> Result<(), String> {
    resp.header("Connection", "close")
        .write_to(w)
        .map_err(|e| e.to_string())
}

fn connection(
    stream: TcpStream,
    peer: IpAddr,
    shared: &Mutex<Shared>,
    host: &Host,
) -> Result<(), String> {
    let limits = &host.config.limits;
    // Small messages go out at once (each is one write, so nothing waits to be joined).
    let _ = stream.set_nodelay(true);
    stream
        .set_write_timeout(Some(limits.write_time))
        .map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(Timed {
        stream: stream.try_clone().map_err(|e| e.to_string())?,
        deadline: Some(Instant::now() + limits.request_time),
    });
    let mut writer = stream;
    let Some(req) = http::read_request(&mut reader)? else {
        return Ok(());
    };
    let admitted = match host.admit(&req, peer) {
        Ok(a) => a,
        Err(refusal) => return respond(&mut writer, refusal),
    };
    if req.path == "/api/ws" {
        let resp = match ws::accept(&req) {
            Ok(r) => r,
            Err(e) => return respond(&mut writer, Response::text(400, &e)),
        };
        resp.write_to(&mut writer).map_err(|e| e.to_string())?;
        // From here the socket carries the edit protocol: no deadline, but a client that stays
        // silent through several pings is gone.
        reader.get_mut().deadline = None;
        writer
            .set_read_timeout(Some(limits.idle_time))
            .map_err(|e| e.to_string())?;
        return socket(reader, writer, shared, limits);
    }
    if req.path == "/api/mcp" {
        if req.method != "POST" {
            return respond(
                &mut writer,
                Response::text(405, "POST JSON-RPC 2.0 to /api/mcp"),
            );
        }
        let resp = {
            let mut g = lock(shared);
            let (rpc_reply, outbox) = crate::ai::mcp::handle_rpc_bytes(&mut g.studio, &req.body);
            deliver(&mut g, outbox);
            match rpc_reply {
                Some(val) => {
                    let text = json::to_compact(&val).unwrap_or_else(|_| "{}".into());
                    Response::json(200, text)
                }
                None => Response::new(204, "text/plain; charset=utf-8", Vec::new()),
            }
        };
        return respond(&mut writer, resp);
    }
    let mut resp = if req.method != "GET" {
        Response::text(405, "only GET outside the edit protocol")
    } else {
        host.route(&req)
    };
    if let (true, Some(token)) = (admitted.set_cookie, &host.config.token) {
        resp = resp.header(
            "Set-Cookie",
            &format!("{}={token}; Path=/; HttpOnly; SameSite=Strict", host.cookie),
        );
    }
    respond(&mut writer, resp)
}

/// What admitting a request decided beyond yes.
#[derive(Default)]
struct Admitted {
    /// The request showed the right `?token=`: give the browser the pairing cookie, so the
    /// page's own fetches carry it.
    set_cookie: bool,
}

fn forbidden(why: &str) -> Response {
    Response::text(403, why)
}

/// `s` without `prefix`, compared without regard to ASCII case.
fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let head = s.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| &s[prefix.len()..])
}

/// A `Host` or `Origin` authority split into its name and its port, if it has one.
fn split_port(authority: &str) -> (&str, Option<&str>) {
    match authority.rsplit_once(':') {
        Some((name, port)) if !port.contains(']') => (name, Some(port)),
        _ => (authority, None),
    }
}

/// Whether `authority` (from a `Host` or `Origin` header) is `name` at `port`. A browser leaves
/// out the port when it is 80.
fn names(authority: &str, name: &str, port: u16) -> bool {
    let (n, p) = split_port(authority);
    n.eq_ignore_ascii_case(name)
        && match p {
            Some(p) => p == port.to_string(),
            None => port == 80,
        }
}

/// Whether `origin` (an `Origin` header) is the page origin `http://` or `https://` + `host`.
fn same_origin(origin: &str, host: Option<&str>) -> bool {
    host.is_some_and(|host| {
        ["http://", "https://"].iter().any(|scheme| {
            strip_prefix_ci(origin, scheme).is_some_and(|rest| rest.eq_ignore_ascii_case(host))
        })
    })
}

/// Whether a request's `Host` headers name this machine: exactly one, which is localhost,
/// 127.0.0.1 or [::1] at `port`.
fn addressed_here(req: &Request, port: u16) -> bool {
    let mut hosts = req.headers_named("host");
    match (hosts.next(), hosts.next()) {
        (Some(h), None) => ["localhost", "127.0.0.1", "[::1]"]
            .iter()
            .any(|name| names(h, name, port)),
        _ => false,
    }
}

/// Whether an `Origin` is one of the studio's own pages when it listens on `port`.
fn own_page(origin: &str, port: u16) -> bool {
    strip_prefix_ci(origin, "http://").is_some_and(|rest| {
        ["localhost", "127.0.0.1"]
            .iter()
            .any(|name| names(rest, name, port))
    })
}

impl Host {
    /// Decides whether to answer a request, or how to refuse it.
    fn admit(&self, req: &Request, peer: IpAddr) -> Result<Admitted, Response> {
        let protocol = req.path == "/api/ws" || req.path == "/api/mcp";
        let Some(token) = &self.config.token else {
            if !addressed_here(req, self.port) {
                return Err(forbidden(&format!(
                    "ganbate-studio answers only at http://localhost:{}/ (start it with --lan to \
                     reach it from other machines)",
                    self.port
                )));
            }
            if protocol && req.headers_named("origin").any(|o| !own_page(o, self.port)) {
                return Err(forbidden(
                    "the edit protocol is open only to the studio's own pages",
                ));
            }
            return Ok(Admitted::default());
        };
        // The editor's own files are free to load, so a phone can show where the code goes;
        // they never look at the cookie, so one left from an earlier run is no wrong guess.
        let public = req.path == "/" || req.path.starts_with("/editor/");
        let from_query = req.param("token");
        let from_cookie = if public {
            None
        } else {
            req.cookie(&self.cookie)
        };
        let refuse = |r: Response| {
            if public {
                Ok(Admitted::default())
            } else {
                Err(r)
            }
        };
        if from_query.is_none() && from_cookie.is_none() {
            return refuse(forbidden("pairing token required"));
        }
        let now = Instant::now();
        let limits = &self.config.limits;
        {
            let mut failures = lock(&self.failures);
            if failures.barred(peer, now, limits) {
                return refuse(Response::text(
                    429,
                    "too many wrong pairing tokens; wait a minute",
                ));
            }
            let by_query = from_query.as_deref().is_some_and(|t| same_token(t, token));
            let by_cookie = from_cookie.is_some_and(|t| same_token(t, token));
            if !by_query && !by_cookie {
                failures.count(peer, now, limits);
                return refuse(forbidden("wrong pairing token"));
            }
            if by_query {
                return Ok(Admitted {
                    set_cookie: !protocol,
                });
            }
        }
        // A browser sends the cookie with requests that pages on other sites start, too: the
        // edit protocol takes it only from this host's own pages (or from a non-browser).
        if protocol
            && req
                .headers_named("origin")
                .any(|o| !same_origin(o, req.header("host")))
        {
            return Err(forbidden(
                "the edit protocol is open only to the studio's own pages",
            ));
        }
        Ok(Admitted::default())
    }

    fn route(&self, req: &Request) -> Response {
        let p = req.path.as_str();
        if p == "/" {
            return http::serve_file(&self.config.editor, "index.html");
        }
        if p == "/api/project/xray-catalog" {
            let resp = http::serve_inside(
                &self.root.join("public/ganbate-studio"),
                "code-surface-catalog.json",
                &self.real_root,
            );
            if resp.status == 404 {
                return Response::json(
                    404,
                    r#"{"error":"not_found","message":"code-surface-catalog.json not generated yet"}"#
                        .into(),
                );
            }
            return resp;
        }
        if p == "/api/project/file" {
            return self.serve_workspace_file(req);
        }
        if let Some(rest) = p.strip_prefix("/editor/") {
            return http::serve_file(&self.config.editor, rest);
        }
        if let Some(rest) = p.strip_prefix("/engine/") {
            return http::serve_file(&self.config.engine, rest);
        }
        if let Some(rest) = p.strip_prefix("/web/") {
            return http::serve_file(&self.config.engine, rest);
        }
        if let Some(rest) = p.strip_prefix("/vendor/") {
            return http::serve_inside(&self.root.join("public/vendor"), rest, &self.real_root);
        }
        if let Some(rest) = p.strip_prefix("/cooked/") {
            return http::serve_inside(&self.cooked, rest, &self.real_root);
        }
        if let Some(rest) = p.strip_prefix("/project/") {
            return http::serve_inside(&self.root, rest, &self.real_root);
        }
        if let Some(rest) = p.strip_prefix("/games/") {
            return http::serve_inside(&self.root.join("public/games"), rest, &self.real_root);
        }
        if let Some(rest) = p.strip_prefix("/dev-assets/") {
            return http::serve_inside(&self.root.join("public/dev-assets"), rest, &self.real_root);
        }
        if let Some(rest) = p.strip_prefix("/ganbate-studio/") {
            let resp = http::serve_inside(
                &self.root.join("public/ganbate-studio"),
                rest,
                &self.real_root,
            );
            if resp.status != 404 {
                return resp;
            }
            return http::serve_file(&self.config.editor, rest);
        }
        if let Some(rest) = p.strip_prefix("/content/") {
            return http::serve_inside(&self.root.join("content"), rest, &self.real_root);
        }
        Response::text(404, "not found")
    }

    /// Serves a read-only workspace source or binary asset file (`GET /api/project/file?repo=...&path=...`)
    /// strictly confined to either the engine repository root or the enclosing project repository root.
    fn serve_workspace_file(&self, req: &Request) -> Response {
        let Some(rel_path) = req.param("path") else {
            return Response::text(400, "missing path query parameter");
        };
        if rel_path.is_empty() || is_forbidden_workspace_path(&rel_path) {
            return Response::text(403, "forbidden workspace path");
        }
        let repo = req.param("repo").unwrap_or_default();
        let base_root = if repo == "ganbate-engine" || repo == "engine" {
            self.config
                .editor
                .parent()
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(|| self.config.editor.clone())
        } else {
            find_workspace_repo_root(&self.real_root)
        };
        let Ok(real_repo_root) = std::fs::canonicalize(&base_root) else {
            return Response::text(404, "workspace root not found");
        };
        let Some(candidate) = http::safe_join(&real_repo_root, &rel_path) else {
            return Response::text(403, "forbidden");
        };
        let Ok(real_file) = std::fs::canonicalize(&candidate) else {
            return Response::text(404, "not found");
        };
        let Ok(stripped) = real_file.strip_prefix(&real_repo_root) else {
            return Response::text(403, "forbidden: outside workspace root");
        };
        if http::hidden(stripped) || !real_file.is_file() {
            return Response::text(403, "forbidden");
        }
        const MAX_WORKSPACE_FILE_BYTES: u64 = 64 * 1024 * 1024;
        if std::fs::metadata(&real_file)
            .map(|m| m.len() > MAX_WORKSPACE_FILE_BYTES)
            .unwrap_or(true)
        {
            return Response::text(413, "file exceeds 64 MiB preview cap");
        }
        match std::fs::read(&real_file) {
            Ok(bytes) => Response::new(200, http::content_type(&real_file), bytes),
            Err(_) => Response::text(404, "not found"),
        }
    }
}

/// Walks up at most 4 ancestor levels from `project_root` to find the enclosing git/monorepo root.
fn find_workspace_repo_root(project_root: &std::path::Path) -> PathBuf {
    let mut best = project_root.to_path_buf();
    let mut cur = project_root;
    for _ in 0..4 {
        let Some(parent) = cur.parent() else {
            break;
        };
        if parent.join(".git").exists()
            || parent.join("pnpm-workspace.yaml").exists()
            || parent.join("Cargo.toml").exists()
        {
            best = parent.to_path_buf();
        }
        if parent.join(".git").exists() {
            break;
        }
        cur = parent;
    }
    best
}

fn is_forbidden_workspace_path(rel_path: &str) -> bool {
    let lower = rel_path.to_ascii_lowercase();
    for ext in [".pem", ".key", ".p12", ".pfx", ".env", "credentials.json"] {
        if lower.ends_with(ext) {
            return true;
        }
    }
    rel_path
        .trim_start_matches('/')
        .split('/')
        .enumerate()
        .any(|(idx, seg)| {
            let s = seg.to_ascii_lowercase();
            s.starts_with('.')
                || matches!(s.as_str(), "node_modules" | "target")
                || (idx == 0 && matches!(s.as_str(), "private" | "secrets" | "credentials"))
        })
}

/// Wrong pairing tokens by peer address: how many, since when (the window's start).
#[derive(Default)]
struct Failures(BTreeMap<IpAddr, (u32, Instant)>);

impl Failures {
    /// Whether `peer` has used up its wrong tokens for the current window.
    fn barred(&mut self, peer: IpAddr, now: Instant, limits: &Limits) -> bool {
        match self.0.get(&peer) {
            Some(&(n, since)) if now.duration_since(since) < limits.failure_window => {
                n >= limits.failures
            }
            Some(_) => {
                self.0.remove(&peer);
                false
            }
            None => false,
        }
    }

    fn count(&mut self, peer: IpAddr, now: Instant, limits: &Limits) {
        if self.0.len() >= MAX_PEERS && !self.0.contains_key(&peer) {
            // Forget the windows that have passed, or else the oldest one.
            self.0
                .retain(|_, (_, since)| now.duration_since(*since) < limits.failure_window);
            if self.0.len() >= MAX_PEERS {
                if let Some(oldest) = self.0.iter().min_by_key(|(_, (_, s))| *s).map(|(k, _)| *k) {
                    self.0.remove(&oldest);
                }
            }
        }
        let entry = self.0.entry(peer).or_insert((0, now));
        if now.duration_since(entry.1) >= limits.failure_window {
            *entry = (0, now);
        }
        entry.0 = entry.0.saturating_add(1);
    }
}

/// Queues messages for their clients. Called under the lock that handled the message, so
/// every client receives messages in the order the documents changed.
fn deliver(g: &mut Shared, out: Outbox) {
    for (client, msg) in out {
        let Some(outlet) = g.clients.get(&client) else {
            continue;
        };
        let Ok(text) = json::to_compact(&msg) else {
            continue;
        };
        if !outlet.offer(Out::Text(text)) {
            // Too far behind: its socket is shut; its reader ends the session.
            g.clients.remove(&client);
        }
    }
}

/// A client's writer: sends what is queued, and a ping every `ping_every` whatever else goes
/// out, so a live client always has something to answer before its idle time runs out.
fn write_loop(
    rx: Receiver<Out>,
    mut w: TcpStream,
    waiting: Arc<AtomicUsize>,
    ping_every: Duration,
) {
    let mut next_ping = Instant::now() + ping_every;
    loop {
        let now = Instant::now();
        let bytes = if now >= next_ping {
            next_ping = now + ping_every;
            ws::frame(0x9, b"")
        } else {
            match rx.recv_timeout(next_ping - now) {
                Ok(out) => {
                    waiting.fetch_sub(out.len(), Ordering::SeqCst);
                    out.into_frame()
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        };
        if w.write_all(&bytes).and_then(|()| w.flush()).is_err() {
            break;
        }
    }
    let _ = w.write_all(&ws::frame(0x8, &[]));
    // Ends the reader too, if it is still waiting.
    let _ = w.shutdown(Shutdown::Both);
}

fn socket(
    mut reader: BufReader<Timed>,
    writer: TcpStream,
    shared: &Mutex<Shared>,
    limits: &Limits,
) -> Result<(), String> {
    let (tx, rx) = sync_channel::<Out>(limits.queue_messages.max(1));
    let waiting = Arc::new(AtomicUsize::new(0));
    let outlet = Outlet {
        queue: tx,
        waiting: waiting.clone(),
        socket: Arc::new(writer.try_clone().map_err(|e| e.to_string())?),
        max_bytes: limits.queue_bytes,
    };
    let ping_every = limits.ping_every;
    let out = std::thread::Builder::new()
        .name("ganbate-studio writer".into())
        .spawn(move || write_loop(rx, writer, waiting, ping_every))
        .map_err(|e| format!("cannot start a writer thread: {e}"))?;
    let client = {
        let mut g = lock(shared);
        let id = g.studio.connect("client");
        g.clients.insert(id, outlet.clone());
        id
    };
    let pongs = outlet.clone();
    let mut pong = move |frame: Vec<u8>| {
        pongs.offer(Out::Frame(frame));
    };
    let result = loop {
        match ws::read_message(&mut reader, &mut pong) {
            Ok(ws::Message::Text(text)) => {
                let mut g = lock(shared);
                let reply = match json::parse(text.as_bytes()) {
                    Ok(msg) => g.studio.handle(client, &msg),
                    Err(e) => vec![(
                        client,
                        Value::Object(BTreeMap::from([
                            ("type".to_string(), Value::String("error".into())),
                            ("message".into(), Value::String(format!("bad JSON: {e}"))),
                        ])),
                    )],
                };
                deliver(&mut g, reply);
            }
            Ok(ws::Message::Binary(_)) => {}
            Ok(ws::Message::Close) => break Ok(()),
            Err(e) => break Err(e),
        }
    };
    {
        let mut g = lock(shared);
        g.clients.remove(&client);
        g.studio.disconnect(client);
    }
    // The writer ends once every sender is gone, the pong callback's included.
    drop(pong);
    drop(outlet);
    let _ = out.join();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(head: &str) -> Request {
        http::read_request(&mut head.as_bytes()).unwrap().unwrap()
    }

    #[test]
    fn tokens_are_128_bits_and_compared_in_full() {
        let (a, b) = (new_token(), new_token());
        for t in [&a, &b] {
            assert_eq!(t.len(), 32, "{t}");
            assert!(
                t.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')),
                "{t}"
            );
        }
        assert_ne!(a, b);
        assert!(same_token(&a, &a.clone()));
        for wrong in ["", &a[..31], &format!("{a}0"), &b, &a.to_uppercase()] {
            assert!(!same_token(wrong, &a), "{wrong}");
        }
    }

    #[test]
    fn hosts_and_origins_are_read_strictly() {
        assert!(names("localhost:7480", "localhost", 7480));
        assert!(names("LocalHost:7480", "localhost", 7480));
        assert!(names("[::1]:7480", "[::1]", 7480));
        assert!(names("localhost", "localhost", 80));
        assert!(names("[::1]", "[::1]", 80));
        for (authority, port) in [
            ("localhost", 7480),
            ("localhost:7481", 7480),
            ("localhost:07480", 7480),
            ("localhost.:7480", 7480),
            ("localhost:7480:7480", 7480),
            ("[::1]", 7480),
            ("\u{e9}vil:7480", 7480),
        ] {
            assert!(!names(authority, "localhost", port), "{authority}");
            assert!(!names(authority, "[::1]", port), "{authority}");
        }
        let here = |host: &str| request(&format!("GET / HTTP/1.1\r\nHost: {host}\r\n\r\n"));
        assert!(addressed_here(&here("127.0.0.1:7480"), 7480));
        assert!(!addressed_here(&here("rebind.example:7480"), 7480));
        assert!(!addressed_here(&request("GET / HTTP/1.0\r\n\r\n"), 7480));
        let twice = request("GET / HTTP/1.1\r\nHost: localhost:7480\r\nHost: evil\r\n\r\n");
        assert!(!addressed_here(&twice, 7480), "two Host headers");
        assert!(own_page("http://localhost:7480", 7480));
        assert!(own_page("HTTP://127.0.0.1:7480", 7480));
        for origin in [
            "https://evil.example",
            "http://localhost:3000",
            "http://localhost:7480/",
            "http://[::1]:7480",
            "null",
            "",
            "http://",
            "ht\u{e9}",
        ] {
            assert!(!own_page(origin, 7480), "{origin}");
        }
        assert!(same_origin(
            "http://192.168.1.5:7480",
            Some("192.168.1.5:7480")
        ));
        assert!(!same_origin(
            "http://evil.example",
            Some("192.168.1.5:7480")
        ));
        assert!(!same_origin("http://192.168.1.5:7480", None));
    }

    #[test]
    fn wrong_tokens_are_counted_per_peer_in_windows() {
        let limits = Limits {
            failures: 3,
            failure_window: Duration::from_secs(60),
            ..Limits::default()
        };
        let (a, b): (IpAddr, IpAddr) = ("192.168.1.7".parse().unwrap(), "::1".parse().unwrap());
        let t0 = Instant::now();
        let mut f = Failures::default();
        for i in 0..3 {
            assert!(!f.barred(a, t0, &limits), "after {i}");
            f.count(a, t0, &limits);
        }
        assert!(f.barred(a, t0 + Duration::from_secs(59), &limits));
        assert!(
            !f.barred(b, t0, &limits),
            "another peer is not held to a's count"
        );
        assert!(
            !f.barred(a, t0 + Duration::from_secs(60), &limits),
            "the window passed"
        );
        // The table of peers stays bounded.
        for i in 0..(MAX_PEERS as u32 + 10) {
            f.count(IpAddr::from(i.to_be_bytes()), t0, &limits);
        }
        assert!(f.0.len() <= MAX_PEERS);
    }

    #[test]
    fn workspace_file_guard_blocks_secrets_and_dot_segments() {
        for blocked in [
            ".env",
            "config/env/.env.local",
            "private/key.txt",
            "secrets/token.json",
            "node_modules/pkg/index.js",
            "target/debug/app",
            "certs/server.pem",
            "certs/id.key",
        ] {
            assert!(is_forbidden_workspace_path(blocked), "{blocked}");
        }
        for allowed in [
            "main/backend/crates/server/src/main.rs",
            "main/backend/crates/identity/credentials/src/lib.rs",
            "main/tools/scripts/art/sources/hero.glb",
            "src/render/world.rs",
        ] {
            assert!(!is_forbidden_workspace_path(allowed), "{allowed}");
        }
    }
}
