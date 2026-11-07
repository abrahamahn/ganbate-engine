//! The studio host over a real socket: the interface's files, the edit protocol's handshake,
//! a ping answered between messages, paths that try to leave their folders, who is answered
//! (the Host and Origin a request carries, the pairing token and its cookie), the order of
//! broadcasts, and the limits that keep slow, idle or hostile clients from tying the host up.
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use ganbate_core::id::Ids;
use ganbate_core::json::{self, Value};
use ganbate_pipeline::project::{Kind, Project};
use ganbate_studio::server::{Config, Limits, serve};
use ganbate_studio::session::Studio;

fn temp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("ganbate-server-{tag}-{}", Ids::from_os().entity()));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// A project with a model file and one scene, "Main"; returns its folder and the scene's id.
fn project() -> (PathBuf, String) {
    let root = temp("project");
    let mut ids = Ids::seeded(21);
    let mut p = Project::create(&root, "Server test", &mut ids).unwrap();
    std::fs::write(root.join("content/note.glb"), b"glTF").unwrap();
    let id = ids.asset();
    let doc = ganbate_scene::scene::new_document(ganbate_scene::scene::Kind::Scene, &id, "Main");
    p.write_document("content/main.scene.json", Kind::Scene, &doc)
        .unwrap();
    (root, id)
}

/// Folders holding a stand-in editor page and engine runtime.
fn interface() -> (PathBuf, PathBuf) {
    let editor = temp("editor");
    std::fs::write(
        editor.join("index.html"),
        b"<!doctype html><title>studio</title>",
    )
    .unwrap();
    std::fs::write(editor.join("studio.js"), b"// the editor").unwrap();
    let engine = temp("engine");
    std::fs::write(engine.join("ganbate.js"), b"export {};").unwrap();
    std::fs::write(engine.join("character-customization.mjs"), b"export {};").unwrap();
    (editor, engine)
}

struct Running {
    port: u16,
    root: PathBuf,
    scene: String,
}

fn start_with(token: Option<&str>, limits: Limits) -> Running {
    let (root, scene) = project();
    let (editor, engine) = interface();
    let (studio, _) = Studio::open(&root, Ids::seeded(22)).unwrap();
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    let config = Config {
        editor,
        engine,
        token: token.map(String::from),
        limits,
    };
    std::thread::spawn(move || serve(studio, listener, config));
    Running { port, root, scene }
}

fn start(token: Option<&str>) -> Running {
    start_with(token, Limits::default())
}

/// Sends a request head (`headers` end with CRLF) and reads the answer to the end: status,
/// head and body; status 0 when the host closed the connection without answering.
fn request(port: u16, path: &str, headers: &str) -> (u16, String, Vec<u8>) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    // A host that closes at once may refuse the request itself.
    let _ = write!(s, "GET {path} HTTP/1.1\r\n{headers}\r\n");
    let mut all = vec![];
    let _ = s.read_to_end(&mut all);
    let split = all
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map_or(all.len(), |i| i + 4);
    let head = String::from_utf8_lossy(&all[..split]).to_string();
    let status = head.get(9..12).and_then(|s| s.parse().ok()).unwrap_or(0);
    (status, head, all[split..].to_vec())
}

/// A GET addressed to the host as a local browser addresses it.
fn get(port: u16, path: &str) -> (u16, Vec<u8>) {
    let (status, _, body) = request(port, path, &format!("Host: 127.0.0.1:{port}\r\n"));
    (status, body)
}

fn masked(opcode: u8, payload: &[u8]) -> Vec<u8> {
    let mask = [1u8, 2, 3, 4];
    let mut f = vec![0x80 | opcode];
    match payload.len() {
        n if n < 126 => f.push(0x80 | n as u8),
        n if n <= 0xffff => {
            f.push(0x80 | 126);
            f.extend_from_slice(&(n as u16).to_be_bytes());
        }
        n => {
            f.push(0x80 | 127);
            f.extend_from_slice(&(n as u64).to_be_bytes());
        }
    }
    f.extend_from_slice(&mask);
    f.extend(payload.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
    f
}

/// Reads one unmasked server frame: (opcode, payload).
fn frame(r: &mut impl Read) -> std::io::Result<(u8, Vec<u8>)> {
    let mut h = [0u8; 2];
    r.read_exact(&mut h)?;
    let mut len = u64::from(h[1] & 0x7f);
    if len == 126 {
        let mut b = [0u8; 2];
        r.read_exact(&mut b)?;
        len = u64::from(u16::from_be_bytes(b));
    } else if len == 127 {
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        len = u64::from_be_bytes(b);
    }
    let mut p = vec![0; len as usize];
    r.read_exact(&mut p)?;
    Ok((h[0] & 0x0f, p))
}

/// Asks for the edit protocol with `headers` (which name the host); the answer's status when
/// it is not an upgrade.
fn upgrade(
    port: u16,
    query: &str,
    headers: &str,
) -> Result<(TcpStream, BufReader<TcpStream>), u16> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        s,
        "GET /api/ws{query} HTTP/1.1\r\n{headers}Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )
    .unwrap();
    let mut r = BufReader::new(s.try_clone().unwrap());
    let mut status = String::new();
    r.read_line(&mut status).unwrap();
    let code: u16 = status.get(9..12).and_then(|s| s.parse().ok()).unwrap_or(0);
    loop {
        let mut line = String::new();
        if r.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
            break;
        }
    }
    if code != 101 {
        return Err(code);
    }
    Ok((s, r))
}

/// The edit protocol as a local client without a browser (no Origin) asks for it.
fn connect_ws(port: u16, query: &str) -> Result<(TcpStream, BufReader<TcpStream>), u16> {
    upgrade(port, query, &format!("Host: 127.0.0.1:{port}\r\n"))
}

fn send(s: &mut TcpStream, text: &str) {
    s.write_all(&masked(0x1, text.as_bytes())).unwrap();
}

/// The next message of type `ty`, answering pings and skipping other messages on the way.
fn until(s: &mut TcpStream, r: &mut impl Read, ty: &str) -> Value {
    loop {
        let (op, p) = frame(r).unwrap();
        match op {
            0x1 => {
                let v = json::parse(&p).unwrap();
                if v.get("type").and_then(|t| t.string().ok()) == Some(ty) {
                    return v;
                }
            }
            0x9 => s.write_all(&masked(0xA, &p)).unwrap(),
            _ => {}
        }
    }
}

/// Reads until the host closes the connection; how many bytes came, or None if it was still
/// open after `wait` without sending anything.
fn read_until_closed(r: &mut impl Read, s: &TcpStream, wait: Duration) -> Option<usize> {
    s.set_read_timeout(Some(wait)).unwrap();
    let mut total = 0;
    let mut buf = vec![0u8; 1 << 16];
    loop {
        match r.read(&mut buf) {
            Ok(0) => return Some(total),
            Ok(n) => total += n,
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                return None;
            }
            Err(_) => return Some(total),
        }
    }
}

/// Whether the host has closed the connection (without waiting).
fn closed_now(s: &TcpStream) -> bool {
    s.set_nonblocking(true).unwrap();
    let closed = match s.peek(&mut [0u8; 1]) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => e.kind() != ErrorKind::WouldBlock,
    };
    s.set_nonblocking(false).unwrap();
    closed
}

#[test]
fn files_and_the_edit_protocol_over_a_socket() {
    let host = start(None);
    let port = host.port;
    let (status, body) = get(port, "/");
    assert_eq!(status, 200);
    assert!(String::from_utf8_lossy(&body).contains("studio"));
    assert_eq!(
        get(port, "/project/content/note.glb"),
        (200, b"glTF".to_vec())
    );
    // Hidden folders and climbing out are refused; unknown routes are not found.
    assert_eq!(get(port, "/project/.ganbate/anything").0, 403);
    assert_eq!(get(port, "/project/../etc/passwd").0, 403);
    assert_eq!(get(port, "/nowhere").0, 404);

    let (mut s, mut r) = connect_ws(port, "").expect("upgrade");
    s.write_all(&masked(0x1, br#"{"type":"hello","client":"test"}"#))
        .unwrap();
    // A ping between messages is answered with a pong carrying its payload.
    s.write_all(&masked(0x9, b"hi")).unwrap();
    let mut got_welcome = false;
    let mut got_pong = false;
    while !(got_welcome && got_pong) {
        let (op, payload) = frame(&mut r).unwrap();
        match op {
            0x1 => {
                let v = json::parse(&payload).unwrap();
                assert_eq!(v.get("type").unwrap().string().unwrap(), "welcome");
                let assets = v.get("assets").unwrap().array().unwrap();
                assert!(
                    assets
                        .iter()
                        .any(|a| a.get("path").unwrap().string().unwrap() == "content/note.glb")
                );
                got_welcome = true;
            }
            0xA => {
                assert_eq!(payload, b"hi");
                got_pong = true;
            }
            // The host's own keep-alive ping.
            0x9 => s.write_all(&masked(0xA, &payload)).unwrap(),
            other => panic!("unexpected opcode {other}"),
        }
    }
    s.write_all(&masked(0x1, b"not json")).unwrap();
    let error = until(&mut s, &mut r, "error");
    assert!(json::to_compact(&error).unwrap().contains("bad JSON"));
    s.write_all(&masked(0x8, b"")).unwrap();
    loop {
        let (op, _) = frame(&mut r).unwrap();
        if op == 0x8 {
            break;
        }
    }
}

#[test]
fn a_lan_host_wants_its_pairing_token() {
    let host = start(Some("pair1234"));
    let port = host.port;
    assert_eq!(connect_ws(port, "").err(), Some(403));
    assert_eq!(connect_ws(port, "?token=wrong").err(), Some(403));
    assert_eq!(get(port, "/project/content/note.glb").0, 403);
    // The interface itself loads, so a phone can show where to type the code.
    assert_eq!(get(port, "/").0, 200);
    assert_eq!(get(port, "/editor/studio.js").0, 200);
    assert!(connect_ws(port, "?token=pair1234").is_ok());
    assert_eq!(get(port, "/project/content/note.glb?token=pair1234").0, 200);
}

/// DNS rebinding: a page on another site whose name now leads to 127.0.0.1 sends its own name
/// as the Host. Only this machine's names, at this host's port, are answered.
#[test]
fn only_requests_addressed_to_this_machine_are_answered() {
    let host = start(None);
    let port = host.port;
    for name in [
        format!("localhost:{port}"),
        format!("127.0.0.1:{port}"),
        format!("[::1]:{port}"),
        format!("LOCALHOST:{port}"),
    ] {
        let (status, _, body) = request(
            port,
            "/project/ganbate.project.json",
            &format!("Host: {name}\r\n"),
        );
        assert_eq!(status, 200, "{name}");
        assert!(String::from_utf8_lossy(&body).contains("Server test"));
    }
    for headers in [
        format!("Host: rebind.attacker.example:{port}\r\n"),
        "Host: rebind.attacker.example\r\n".to_string(),
        format!("Host: 127.0.0.1.attacker.example:{port}\r\n"),
        format!("Host: localhost:{}\r\n", port.wrapping_add(1)),
        "Host: localhost\r\n".to_string(),
        String::new(),
        format!("Host: localhost:{port}\r\nHost: rebind.attacker.example\r\n"),
    ] {
        for path in [
            "/project/ganbate.project.json",
            "/project/content/main.scene.json",
            "/",
        ] {
            assert_eq!(request(port, path, &headers).0, 403, "{headers:?} {path}");
        }
    }
    // Nor does the edit protocol answer the rebound page (whose Origin matches its Host).
    let rebound = format!(
        "Host: rebind.attacker.example:{port}\r\nOrigin: http://rebind.attacker.example:{port}\r\n"
    );
    assert_eq!(upgrade(port, "", &rebound).err(), Some(403));
}

/// Browsers let any page open a WebSocket anywhere: the edit protocol answers only the
/// studio's own pages, and clients that are not browsers (which send no Origin).
#[test]
fn other_sites_cannot_open_the_edit_protocol() {
    let host = start(None);
    let port = host.port;
    let here = format!("Host: localhost:{port}\r\n");
    for origin in [
        "https://evil.example".to_string(),
        "http://evil.example".to_string(),
        format!("http://localhost:{}", port.wrapping_add(1)),
        format!("https://localhost:{port}"),
        "null".to_string(),
    ] {
        let headers = format!("{here}Origin: {origin}\r\n");
        assert_eq!(upgrade(port, "", &headers).err(), Some(403), "{origin}");
    }
    for origin in [
        format!("http://localhost:{port}"),
        format!("http://127.0.0.1:{port}"),
    ] {
        let headers = format!("{here}Origin: {origin}\r\n");
        assert!(upgrade(port, "", &headers).is_ok(), "{origin}");
    }
    assert!(upgrade(port, "", &here).is_ok(), "no Origin: not a browser");
}

/// With `--lan` the editor's page is opened with `?token=`; its own fetches of the engine,
/// cooked previews and project files carry the token as the cookie the page was given.
#[test]
fn a_paired_page_carries_its_token_in_a_cookie() {
    let host = start(Some("pair1234"));
    let port = host.port;
    let here = format!("Host: 127.0.0.1:{port}\r\n");
    // A cooked preview to fetch.
    let (mut s, mut r) = connect_ws(port, "?token=pair1234").unwrap();
    send(
        &mut s,
        &format!(r#"{{"type":"cook","doc":"{}"}}"#, host.scene),
    );
    let cooked = until(&mut s, &mut r, "cooked");
    let file = cooked.get("file").unwrap().string().unwrap().to_string();
    let preview = format!("/cooked/{}", file.rsplit('/').next().unwrap());
    let fetched = [
        "/engine/ganbate.js",
        preview.as_str(),
        "/project/content/main.scene.json",
    ];
    for path in fetched {
        assert_eq!(
            request(port, path, &here).0,
            403,
            "{path} without the token"
        );
    }
    // Opening the page with the token sets the cookie ...
    let (status, head, _) = request(port, "/?token=pair1234", &here);
    assert_eq!(status, 200);
    let set = head
        .lines()
        .find_map(|l| l.strip_prefix("Set-Cookie: "))
        .expect("the page gets the pairing cookie");
    assert_eq!(
        set,
        format!("ganbate-studio-{port}=pair1234; Path=/; HttpOnly; SameSite=Strict")
    );
    let cookie = format!("Cookie: other=1; ganbate-studio-{port}=pair1234\r\n");
    // ... and the cookie opens what the page fetches.
    for path in fetched {
        assert_eq!(
            request(port, path, &format!("{here}{cookie}")).0,
            200,
            "{path}"
        );
    }
    // The page's own socket may come with the cookie alone (a reload without ?token=) ...
    let own = format!("{here}{cookie}Origin: http://127.0.0.1:{port}\r\n");
    assert!(upgrade(port, "", &own).is_ok());
    assert!(upgrade(port, "", &format!("{here}{cookie}")).is_ok());
    // ... but not a page from another origin: the browser gives the cookie to pages of the same
    // site too (another server on this machine, at another port).
    for origin in [
        format!("http://127.0.0.1:{}", port.wrapping_add(1)),
        "https://evil.example".to_string(),
    ] {
        let foreign = format!("{here}{cookie}Origin: {origin}\r\n");
        assert_eq!(upgrade(port, "", &foreign).err(), Some(403), "{origin}");
    }
    // A wrong cookie, or one for a studio on another port, opens nothing.
    for bad in [
        format!("Cookie: ganbate-studio-{port}=pair12345\r\n"),
        format!(
            "Cookie: ganbate-studio-{}=pair1234\r\n",
            port.wrapping_add(1)
        ),
    ] {
        assert_eq!(
            request(port, "/engine/ganbate.js", &format!("{here}{bad}")).0,
            403
        );
    }
    // The editor's own files never needed it.
    assert_eq!(request(port, "/editor/studio.js", &here).0, 200);
}

/// `ganbate-studio --lan` prints addresses that carry the token, the local one included, and the
/// token is 128 bits.
#[test]
fn the_addresses_a_lan_host_prints_work() {
    let (_child, port, lines) = studio_process(&["--lan"], &[], None);
    let local = lines
        .iter()
        .find_map(|l| l.split(" at ").nth(1))
        .expect("the local address")
        .to_string();
    let query = local.split_once('?').map(|(_, q)| q.to_string()).unwrap();
    let token = query.strip_prefix("token=").unwrap().to_string();
    assert_eq!(local, format!("http://localhost:{port}/?token={token}"));
    assert!(
        lines
            .iter()
            .any(|l| l.ends_with(&format!(":{port}/?token={token}"))
                && l.contains("on the local network"))
    );
    assert_eq!(token.len(), 32);
    assert!(token.bytes().all(|b| b.is_ascii_hexdigit()));
    let here = format!("Host: localhost:{port}\r\n");
    assert_eq!(request(port, "/engine/ganbate.js", &here).0, 403);
    assert_eq!(
        request(port, &format!("/engine/ganbate.js?{query}"), &here).0,
        200
    );
    assert!(upgrade(port, &format!("?{query}"), &here).is_ok());
}

/// Undo, redo and save carry no revision, so two clients can send them at once; every watcher
/// must still hear of the changes in the order they happened, or its copy of the document
/// drifts from the host's (and a `saved` that overtakes newer ops shows unsaved work as saved).
#[test]
fn broadcasts_reach_every_watcher_in_revision_order() {
    let host = start(None);
    let (port, id) = (host.port, host.scene.clone());
    let (mut w, mut wr) = connect_ws(port, "").unwrap();
    send(&mut w, &format!(r#"{{"type":"open","doc":"{id}"}}"#));
    until(&mut w, &mut wr, "doc");
    send(
        &mut w,
        &format!(
            r#"{{"type":"edit","doc":"{id}","rev":0,"ops":[{{"set":["name"],"value":"V"}}]}}"#
        ),
    );
    until(&mut w, &mut wr, "ops");
    // Two clients that do not watch the document undo, redo and save it as fast as they can.
    let senders: Vec<_> = (0..2)
        .map(|k| {
            let (mut s, r) = connect_ws(port, "").unwrap();
            let id = id.clone();
            std::thread::spawn(move || {
                for i in 0..3000 {
                    let ty = if (i + k) % 2 == 0 { "undo" } else { "redo" };
                    send(&mut s, &format!(r#"{{"type":"{ty}","doc":"{id}"}}"#));
                    if i % 7 == 0 {
                        send(&mut s, &format!(r#"{{"type":"save","doc":"{id}"}}"#));
                    }
                }
                (s, r)
            })
        })
        .collect();
    w.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
    let (mut last, mut seen, mut out_of_order) = (1u64, 0, vec![]);
    while let Ok((op, p)) = frame(&mut wr) {
        if op == 0x9 {
            w.write_all(&masked(0xA, &p)).unwrap();
            continue;
        }
        let v = json::parse(&p).unwrap();
        let rev = v.get("rev").and_then(|r| r.unsigned().ok()).unwrap_or(0);
        match v.get("type").and_then(|t| t.string().ok()) {
            Some("ops") => {
                seen += 1;
                if rev != last + 1 {
                    out_of_order.push(format!("ops {rev} after {last}"));
                }
                last = rev;
            }
            Some("saved") if rev != last => {
                out_of_order.push(format!("saved {rev} after ops {last}"))
            }
            _ => {}
        }
    }
    for s in senders {
        let _ = s.join();
    }
    assert!(seen > 100, "only {seen} broadcasts");
    assert_eq!(out_of_order, Vec::<String>::new(), "of {seen}");
}

/// A request must arrive whole within the request time, however it trickles in, and only so
/// many connections are served at once.
#[test]
fn slow_requests_end_and_connections_are_capped() {
    let limits = Limits {
        connections: 4,
        request_time: Duration::from_millis(600),
        ..Limits::default()
    };
    let host = start_with(None, limits);
    let port = host.port;
    // A byte every 50 ms keeps a connection busy but does not keep it open.
    let mut trickle = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let t = Instant::now();
    let mut open_for = None;
    for b in b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        .iter()
        .cycle()
        .take(200)
    {
        if closed_now(&trickle) {
            open_for = Some(t.elapsed());
            break;
        }
        let _ = trickle.write_all(&[*b]);
        std::thread::sleep(Duration::from_millis(50));
    }
    let open_for = open_for.expect("a trickled request kept its connection for 10 s");
    assert!(open_for < Duration::from_secs(3), "open for {open_for:?}");
    // Four idle connections fill the host; a fifth is closed without an answer ...
    let idle: Vec<TcpStream> = (0..4)
        .map(|_| {
            let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
            s.write_all(b"GET / HTTP/1.1\r\n").unwrap();
            s
        })
        .collect();
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(
        get(port, "/").0,
        0,
        "a connection over the limit is closed at once"
    );
    // ... until the idle ones run out of time, which they do on their own.
    for s in &idle {
        let mut r = s.try_clone().unwrap();
        assert!(read_until_closed(&mut r, s, Duration::from_secs(5)).is_some());
    }
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(get(port, "/").0, 200);
}

/// The host pings edit-protocol clients and drops one that stays silent past its idle time; a
/// live client (one that answers pings) stays connected however long it says nothing else.
#[test]
fn silent_edit_clients_are_dropped_and_live_ones_kept() {
    let limits = Limits {
        ping_every: Duration::from_millis(100),
        idle_time: Duration::from_millis(500),
        ..Limits::default()
    };
    let host = start_with(None, limits);
    let port = host.port;
    let (silent, mut silent_r) = connect_ws(port, "").unwrap();
    let (mut live, mut live_r) = connect_ws(port, "").unwrap();
    live.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    let t = Instant::now();
    let mut pings = 0;
    while t.elapsed() < Duration::from_millis(1500) {
        let (op, p) = frame(&mut live_r).expect("a ping within 2 s");
        assert_eq!(op, 0x9, "only pings while nothing happens");
        pings += 1;
        live.write_all(&masked(0xA, &p)).unwrap();
    }
    assert!(pings >= 5, "{pings} pings in 1.5 s");
    send(&mut live, r#"{"type":"ids","count":1}"#);
    until(&mut live, &mut live_r, "ids");
    // The silent client got pings too, then the close.
    assert!(read_until_closed(&mut silent_r, &silent, Duration::from_secs(1)).is_some());
}

/// A client that stops reading (a phone gone to sleep, or a hostile one) does not make the
/// host keep its replies without end: past its queue's bounds it is dropped. Others go on.
#[test]
fn a_client_that_stops_reading_is_dropped() {
    let limits = Limits {
        queue_messages: 64,
        queue_bytes: 1 << 20,
        ..Limits::default()
    };
    let host = start_with(None, limits);
    let port = host.port;
    let (mut s, mut r) = connect_ws(port, "").unwrap();
    // Each answer is about 80 KB: 2,000 of them would be 160 MB.
    let ask = masked(0x1, br#"{"type":"ids","count":4096}"#);
    let all: Vec<u8> = ask.iter().copied().cycle().take(ask.len() * 2000).collect();
    let _ = s.write_all(&all);
    std::thread::sleep(Duration::from_millis(500));
    let got = read_until_closed(&mut r, &s, Duration::from_secs(10));
    let got = got.expect("the host kept a client that does not read");
    assert!(got < 40 << 20, "{got} bytes came before it was dropped");
    let (mut other, mut other_r) = connect_ws(port, "").unwrap();
    send(&mut other, r#"{"type":"ids","count":1}"#);
    until(&mut other, &mut other_r, "ids");
}

/// One write per frame and no Nagle delay: an answer does not wait for the client's delayed
/// acknowledgement of the frame's header (about 40 ms each on Linux).
#[test]
fn answers_do_not_wait_for_delayed_acknowledgements() {
    let host = start(None);
    let (mut s, mut r) = connect_ws(host.port, "").unwrap();
    send(&mut s, r#"{"type":"hello"}"#);
    until(&mut s, &mut r, "welcome");
    let n = 40;
    let t = Instant::now();
    for _ in 0..n {
        send(&mut s, r#"{"type":"ids","count":1}"#);
        until(&mut s, &mut r, "ids");
    }
    let each = t.elapsed() / n;
    assert!(each < Duration::from_millis(20), "{each:?} per round trip");
}

#[test]
fn a_plus_in_a_path_is_a_plus() {
    let host = start(None);
    std::fs::write(host.root.join("content/day+night.look.json"), b"{\"a\":1}").unwrap();
    std::fs::write(host.root.join("content/day night.look.json"), b"{\"b\":2}").unwrap();
    let port = host.port;
    assert_eq!(
        get(port, "/project/content/day+night.look.json"),
        (200, b"{\"a\":1}".to_vec())
    );
    assert_eq!(get(port, "/project/content/day%2Bnight.look.json").0, 200);
    assert_eq!(
        get(port, "/project/content/day%20night.look.json"),
        (200, b"{\"b\":2}".to_vec())
    );
}

#[cfg(unix)]
#[test]
fn project_files_never_resolve_outside_the_project() {
    use std::os::unix::fs::symlink;
    let host = start(None);
    let (port, root) = (host.port, host.root.clone());
    let outside = temp("outside");
    std::fs::write(outside.join("secret.txt"), b"outside the project").unwrap();
    symlink(&outside, root.join("content/shared")).unwrap();
    symlink(outside.join("secret.txt"), root.join("content/secret.txt")).unwrap();
    for path in [
        "/project/content/shared/secret.txt",
        "/project/content/secret.txt",
    ] {
        let (status, body) = get(port, path);
        assert_eq!(status, 403, "{path}: {}", String::from_utf8_lossy(&body));
    }
    // A link into a hidden folder of the project is no way around the hidden rule.
    std::fs::create_dir_all(root.join(".private")).unwrap();
    std::fs::write(root.join(".private/key.txt"), b"key").unwrap();
    symlink(root.join(".private"), root.join("content/private")).unwrap();
    assert_eq!(get(port, "/project/content/private/key.txt").0, 403);
    // Links that stay inside the project still work.
    symlink(
        root.join("content/note.glb"),
        root.join("content/alias.glb"),
    )
    .unwrap();
    assert_eq!(
        get(port, "/project/content/alias.glb"),
        (200, b"glTF".to_vec())
    );
    // Cooked previews are read only from the project's own cache folder.
    std::fs::create_dir_all(outside.join("cooked")).unwrap();
    std::fs::write(outside.join("cooked/leak.gbew"), b"outside").unwrap();
    symlink(&outside, root.join(".ganbate")).unwrap();
    assert_eq!(get(port, "/cooked/leak.gbew").0, 403);
}

/// Guessing the token: a peer that sends too many wrong ones is not answered (429), even with
/// the right one, until its window has passed.
#[test]
fn wrong_tokens_are_limited_per_peer() {
    let limits = Limits {
        failures: 3,
        failure_window: Duration::from_millis(800),
        ..Limits::default()
    };
    let host = start_with(Some("pair1234"), limits);
    let port = host.port;
    for guess in ["a", "b", "c"] {
        assert_eq!(
            get(port, &format!("/engine/ganbate.js?token={guess}")).0,
            403
        );
    }
    assert_eq!(get(port, "/engine/ganbate.js?token=pair1234").0, 429);
    assert_eq!(connect_ws(port, "?token=pair1234").err(), Some(429));
    let cookie = format!("Host: 127.0.0.1:{port}\r\nCookie: ganbate-studio-{port}=pair1234\r\n");
    assert_eq!(request(port, "/engine/ganbate.js", &cookie).0, 429);
    // The page still loads, but a token shown to it is not looked at (no cookie comes back).
    let (status, head, _) = request(
        port,
        "/?token=pair1234",
        &format!("Host: 127.0.0.1:{port}\r\n"),
    );
    assert_eq!(status, 200);
    assert!(!head.contains("Set-Cookie"), "{head}");
    std::thread::sleep(Duration::from_millis(900));
    assert_eq!(get(port, "/engine/ganbate.js?token=pair1234").0, 200);
}

/// A child process, killed when the test ends (whether or not it passed).
struct Proc(Child);

impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A port nothing listens on just now.
fn free_port() -> u16 {
    let l = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    l.local_addr().unwrap().port()
}

/// Starts the `ganbate-studio` binary on a new project, at most `files` descriptors open if given,
/// and waits for its address line; returns it, its port and the lines it printed.
fn studio_process(
    extra: &[&str],
    env: &[(&str, &str)],
    files: Option<u32>,
) -> (Proc, u16, Vec<String>) {
    let (root, _) = project();
    let (editor, engine) = interface();
    let port = free_port();
    let mut cmd = match files {
        Some(n) => {
            let mut c = Command::new("sh");
            // The soft limit only, so the test can raise it again.
            c.arg("-c")
                .arg(format!("ulimit -S -n {n} && exec \"$0\" \"$@\""))
                .arg(env!("CARGO_BIN_EXE_ganbate-studio"));
            c
        }
        None => Command::new(env!("CARGO_BIN_EXE_ganbate-studio")),
    };
    cmd.arg(&root)
        .args(["--port", &port.to_string(), "--editor"])
        .arg(&editor)
        .arg("--engine")
        .arg(&engine)
        .args(extra)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = Proc(cmd.spawn().unwrap());
    let mut out = BufReader::new(child.0.stdout.take().unwrap());
    let mut lines = vec![];
    loop {
        let mut line = String::new();
        assert!(
            out.read_line(&mut line).unwrap() > 0,
            "no address in {lines:?}"
        );
        lines.push(line.trim_end().to_string());
        if line.contains(" at http://localhost:") {
            break;
        }
    }
    // The network line, when there is one, comes right after.
    if extra.contains(&"--lan") {
        let mut line = String::new();
        out.read_line(&mut line).unwrap();
        lines.push(line.trim_end().to_string());
    }
    std::thread::spawn(move || std::io::copy(&mut out, &mut std::io::sink()));
    (child, port, lines)
}

/// When the system will not start another thread, the connection that needed it is closed
/// and the host goes on serving; it used to panic and exit, losing every unsaved edit.
#[cfg(target_os = "linux")]
#[test]
fn a_thread_that_cannot_start_does_not_stop_the_host() {
    // A stack no address space can hold makes every new thread fail to start.
    let (mut child, port, _) = studio_process(&[], &[("RUST_MIN_STACK", "281474976710656")], None);
    let mut err = child.0.stderr.take().unwrap();
    for _ in 0..5 {
        match TcpStream::connect(("127.0.0.1", port)) {
            Ok(mut s) => {
                let _ = write!(s, "GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n");
                let mut answer = vec![];
                let _ = s.read_to_end(&mut answer);
                assert!(answer.is_empty(), "no thread, no answer");
            }
            Err(e) => panic!("the host stopped listening: {e}"),
        }
    }
    std::thread::sleep(Duration::from_millis(200));
    let alive = child.0.try_wait().unwrap().is_none();
    drop(child);
    let mut log = String::new();
    let _ = err.read_to_string(&mut log);
    assert!(alive, "the host exited: {log}");
    assert!(log.contains("cannot start a thread"), "{log}");
}

/// Out of file descriptors, every accept fails until one is free: the host waits between tries
/// instead of spinning on the error (it used to write tens of megabytes of log a second), and
/// answers once descriptors are free again.
#[cfg(target_os = "linux")]
#[test]
fn running_out_of_descriptors_neither_spins_nor_stops_the_host() {
    // Four descriptors: the standard three and the listener. No connection can be accepted.
    let (mut child, port, _) = studio_process(&[], &[], Some(4));
    let mut err = child.0.stderr.take().unwrap();
    let logged = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = logged.clone();
    std::thread::spawn(move || {
        let mut buf = [0u8; 1 << 16];
        while let Ok(k) = err.read(&mut buf) {
            if k == 0 {
                break;
            }
            counter.fetch_add(k, std::sync::atomic::Ordering::SeqCst);
        }
    });
    // A request that waits to be accepted (the system completes the connection meanwhile).
    let mut waiting = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(waiting, "GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n").unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let before = logged.load(std::sync::atomic::Ordering::SeqCst);
    std::thread::sleep(Duration::from_secs(1));
    let during = logged.load(std::sync::atomic::Ordering::SeqCst) - before;
    assert!(child.0.try_wait().unwrap().is_none(), "the host exited");
    assert!(during < 4096, "{during} bytes of log in 1 s");
    // With descriptors to spare again, the waiting request is answered.
    let raised = Command::new("prlimit")
        .args(["--pid", &child.0.id().to_string(), "--nofile=256:"])
        .status();
    if raised.is_ok_and(|s| s.success()) {
        waiting
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut answer = vec![];
        let _ = waiting.read_to_end(&mut answer);
        assert!(
            answer.starts_with(b"HTTP/1.1 200"),
            "{}",
            String::from_utf8_lossy(&answer)
        );
    } else {
        eprintln!("prlimit is not available: the recovery half was not checked");
    }
}

#[test]
fn xray_catalog_and_project_public_assets_are_served_inside_project_root() {
    let host = start(None);
    let port = host.port;

    // Before code-surface-catalog.json exists, /api/project/xray-catalog returns a clean 404 JSON payload.
    let (status_missing, head_missing, body_missing) = request(
        port,
        "/api/project/xray-catalog",
        &format!("Host: 127.0.0.1:{port}\r\n"),
    );
    assert_eq!(status_missing, 404);
    assert!(head_missing.contains("Content-Type: application/json; charset=utf-8"));
    let parsed_missing = json::parse(&body_missing).expect("404 JSON body");
    assert_eq!(
        parsed_missing.get("error").and_then(|v| v.string().ok()),
        Some("not_found")
    );

    // Populate public/ganbate-studio/code-surface-catalog.json and public static asset folders.
    let studio_pub = host.root.join("public/ganbate-studio");
    let games_pub = host.root.join("public/games/rooms");
    let dev_assets_pub = host.root.join("public/dev-assets");
    std::fs::create_dir_all(&studio_pub).unwrap();
    std::fs::create_dir_all(&games_pub).unwrap();
    std::fs::create_dir_all(&dev_assets_pub).unwrap();
    let catalog_bytes = br#"{"schema":"ganbate.code-xray/1","surfaces":[{"id":"world"}]}"#;
    std::fs::write(studio_pub.join("code-surface-catalog.json"), catalog_bytes).unwrap();
    std::fs::write(games_pub.join("stage.glb"), b"glTF-stage").unwrap();
    std::fs::write(dev_assets_pub.join("prop.glb"), b"glTF-prop").unwrap();

    let (status_ok, head_ok, body_ok) = request(
        port,
        "/api/project/xray-catalog",
        &format!("Host: 127.0.0.1:{port}\r\n"),
    );
    assert_eq!(status_ok, 200);
    assert!(head_ok.contains("Content-Type: application/json; charset=utf-8"));
    assert_eq!(body_ok, catalog_bytes);

    assert_eq!(
        get(port, "/ganbate-studio/code-surface-catalog.json"),
        (200, catalog_bytes.to_vec())
    );
    assert_eq!(
        get(port, "/games/rooms/stage.glb"),
        (200, b"glTF-stage".to_vec())
    );
    assert_eq!(
        get(port, "/dev-assets/prop.glb"),
        (200, b"glTF-prop".to_vec())
    );
    assert_eq!(get(port, "/content/note.glb"), (200, b"glTF".to_vec()));
    assert_eq!(get(port, "/web/character-customization.mjs").0, 200);

    // Path traversal and hidden files remain strictly refused.
    assert_eq!(get(port, "/games/../../etc/passwd").0, 403);
    assert_eq!(get(port, "/dev-assets/.hidden/secret.glb").0, 403);
}
