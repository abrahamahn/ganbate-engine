//! `ganbate-studio [PROJECT | --project PATH] [--port N] [--lan] [--open] [--new NAME] [--editor DIR] [--engine DIR]`
//!
//! Serves Ganbate Studio for the project folder PROJECT (creating it with `--new`). Open
//! `http://localhost:PORT/` in a browser. With `--lan` the host also listens on the local
//! network and makes a pairing token that every client must present, phones and running games
//! as well as this machine's browser: the addresses it prints carry it (`?token=`).
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use ganbate_core::id::Ids;
use ganbate_pipeline::project::Project;
use ganbate_studio::server::{Config, Limits, new_token, serve};
use ganbate_studio::session::Studio;

fn main() {
    if let Err(e) = run() {
        eprintln!("ganbate-studio: {e}");
        std::process::exit(1);
    }
}

fn resolve_project_root(input: PathBuf) -> PathBuf {
    if input
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n == "ganbate.project.json")
        || input.is_file()
    {
        input
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    } else {
        input
    }
}

fn try_open_browser(url: &str) {
    let spawn_quiet = |prog: &str, args: &[&str]| -> bool {
        Command::new(prog)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .is_ok()
    };

    if cfg!(target_os = "macos") {
        let _ = spawn_quiet("open", &[url]);
        return;
    }
    if cfg!(target_os = "windows") {
        let _ = spawn_quiet("cmd", &["/C", "start", "", url]);
        return;
    }
    let is_wsl = std::env::var_os("WSL_DISTRO_NAME").is_some()
        || Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists();
    if is_wsl
        && (spawn_quiet("wslview", &[url]) || spawn_quiet("cmd.exe", &["/C", "start", "", url]))
    {
        return;
    }
    if std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some() {
        let _ = spawn_quiet("xdg-open", &[url]);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut project: Option<PathBuf> = None;
    let mut port = 7480u16;
    let mut lan = false;
    let mut open_browser = std::env::var("GANBATE_EDITOR_OPEN")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let mut new_name: Option<String> = None;
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut editor = here.join("../../editor");
    let mut engine = here.join("../../web/dist");
    while let Some(a) = args.next() {
        let mut value = |what: &str| args.next().ok_or(format!("{a} needs {what}"));
        match a.as_str() {
            "--project" => project = Some(PathBuf::from(value("a project folder or manifest")?)),
            "--port" => {
                port = value("a port")?
                    .parse()
                    .map_err(|_| "--port needs a number")?
            }
            "--lan" => lan = true,
            "--open" => open_browser = true,
            "--new" => new_name = Some(value("a project name")?),
            "--editor" => editor = PathBuf::from(value("a folder")?),
            "--engine" => engine = PathBuf::from(value("a folder")?),
            "-h" | "--help" => {
                println!(
                    "usage: ganbate-studio [PROJECT | --project PATH] [--port N] [--lan] [--open] [--new NAME] [--editor DIR] [--engine DIR]"
                );
                return Ok(());
            }
            p if !p.starts_with('-') && project.is_none() => project = Some(PathBuf::from(p)),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let root = resolve_project_root(project.ok_or("name a project folder")?);
    let mut ids = Ids::from_os();
    if let Some(name) = new_name {
        std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
        Project::create(&root, &name, &mut ids)?;
        println!(
            "ganbate-studio: created project {name:?} in {}",
            root.display()
        );
    }
    if !engine.join("ganbate.js").exists() {
        eprintln!(
            "ganbate-studio: no web runtime in {} (run scripts/build-web.sh); the viewport will not start",
            engine.display()
        );
    }
    let token = lan.then(new_token);
    let (studio, notes) = Studio::open(&root, ids)?;
    for n in notes {
        println!("ganbate-studio: {n}");
    }
    let host = if lan { "0.0.0.0" } else { "127.0.0.1" };
    let listener = TcpListener::bind((host, port))
        .map_err(|e| format!("cannot listen on {host}:{port}: {e}"))?;
    // The port asked for may be 0 (any free one).
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let query = token
        .as_ref()
        .map_or(String::new(), |t| format!("?token={t}"));
    let loopback_url = format!("http://127.0.0.1:{port}/{query}");
    println!(
        "ganbate-studio: {} at http://localhost:{port}/{query}",
        studio.project.manifest.name
    );
    if token.is_some() {
        println!("ganbate-studio: on the local network, open http://<this machine>:{port}/{query}");
    }
    println!("Ganbate Studio ready -> {loopback_url}");
    if open_browser {
        try_open_browser(&loopback_url);
    }
    serve(
        studio,
        listener,
        Config {
            editor,
            engine,
            token,
            limits: Limits::default(),
        },
    )
}
