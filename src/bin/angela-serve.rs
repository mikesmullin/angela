use std::sync::Arc;

use angela::http;
use angela::serve::{AppState, Server};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut port: u16 = std::env::var("ANGELA_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8763);
    let mut root: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" if i + 1 < args.len() => {
                port = args[i + 1].parse().unwrap_or(port);
                i += 2;
            }
            "-c" | "--config" if i + 1 < args.len() => {
                root = Some(args[i + 1].clone());
                i += 2;
            }
            _ => i += 1,
        }
    }

    let project_root = angela::project::resolve_project_root(root.as_deref());
    let state = Arc::new(AppState::new(&project_root, port)?);
    let addr = format!("127.0.0.1:{port}");
    eprintln!(
        "angela-rs listening on http://{addr}  projectRoot={}",
        state.project_root
    );
    http::serve(&addr, Arc::new(Server::new(state)))
}
