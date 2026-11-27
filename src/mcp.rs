use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use crate::cancel::Cancel;

const POLL: Duration = Duration::from_millis(50);
const CANCEL_GRACE: Duration = Duration::from_millis(500);

use serde_json::{json, Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    Newline,
    Lsp,
}

#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone)]
pub struct ServerSpec {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub env: Vec<(String, String)>,
    pub framing: Framing,
}

type Pending = Arc<Mutex<HashMap<u64, Sender<Result<Value, String>>>>>;

enum Transport {
    Stdio {
        stdin: Mutex<ChildStdin>,
        child: Mutex<Child>,
        pending: Pending,
        next_id: AtomicU64,
        framing: Framing,
    },
    Http { url: String },
}

pub struct McpServer {
    pub name: String,
    pub tools: Vec<ToolDef>,
    transport: Transport,
    forced_closed: AtomicBool,
}

impl McpServer {
    pub fn connect(spec: &ServerSpec, timeout: Duration) -> Result<Arc<Self>, String> {
        Self::connect_cancellable(spec, timeout, &Cancel::never())
    }

    pub fn connect_cancellable(spec: &ServerSpec, timeout: Duration, stop: &Cancel) -> Result<Arc<Self>, String> {
        if stop.stopped() { return Err("stopped".into()); }
        let mut cmd = Command::new(&spec.command);
        cmd.args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        if let Some(d) = &spec.cwd {
            cmd.current_dir(d);
        }
        for (k, v) in &spec.env {
            cmd.env(k, v);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("spawning MCP server {}: {e}", spec.name))?;

        let stdout = child.stdout.take().ok_or("no stdout")?;
        let stderr = child.stderr.take().ok_or("no stderr")?;
        let stdin = child.stdin.take().ok_or("no stdin")?;

        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));

        {
            let pending = Arc::clone(&pending);
            let framing = spec.framing;
            let name = spec.name.clone();
            std::thread::spawn(move || {
                read_loop(stdout, framing, &pending);
                let mut p = pending.lock().unwrap();
                for (_, tx) in p.drain() {
                    let _ = tx.send(Err(format!("MCP server {name} exited")));
                }
            });
        }
        {
            let name = spec.name.clone();
            std::thread::spawn(move || {
                let r = BufReader::new(stderr);
                for line in r.lines().map_while(Result::ok) {
                    eprintln!("[mcp:{name}] {line}");
                }
            });
        }

        let server = Arc::new(McpServer {
            name: spec.name.clone(),
            tools: Vec::new(),
            forced_closed: AtomicBool::new(false),
            transport: Transport::Stdio {
                stdin: Mutex::new(stdin),
                child: Mutex::new(child),
                pending,
                next_id: AtomicU64::new(1),
                framing: spec.framing,
            },
        });

        server.request_cancellable(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "angela", "version": "0.1.0" },
            }),
            timeout, stop,
        )?;
        server.notify("notifications/initialized", json!({}));

        let listed = server.request_cancellable("tools/list", json!({}), timeout, stop)?;
        let tools = parse_tools(&listed);

        let mut inner = Arc::try_unwrap(server).map_err(|_| "server aliased".to_string())?;
        inner.tools = tools;
        Ok(Arc::new(inner))
    }

    fn write_frame(&self, text: &str) -> Result<(), String> {
        let Transport::Stdio { stdin, framing, .. } = &self.transport else {
            return Err("not a stdio server".into());
        };
        let mut out = stdin.lock().unwrap();
        match framing {
            Framing::Newline => {
                writeln!(out, "{text}").map_err(|e| e.to_string())?;
            }
            Framing::Lsp => {
                write!(out, "Content-Length: {}\r\n\r\n{text}", text.len())
                    .map_err(|e| e.to_string())?;
            }
        }
        out.flush().map_err(|e| e.to_string())
    }

    pub fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value, String> {
        self.request_cancellable(method, params, timeout, &Cancel::never())
    }

    pub fn request_cancellable(&self, method: &str, params: Value, timeout: Duration, stop: &Cancel) -> Result<Value, String> {
        if stop.stopped() { return Err("stopped".into()); }
        match &self.transport {
            Transport::Http { url } => http_rpc(url, method, params, timeout, stop),
            Transport::Stdio {
                pending, next_id, ..
            } => {
                let id = next_id.fetch_add(1, Ordering::Relaxed);
                let (tx, rx): (Sender<_>, Receiver<_>) = channel();
                pending.lock().unwrap().insert(id, tx);

                let body =
                    json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
                if let Err(e) = self.write_frame(&body.to_string()) {
                    pending.lock().unwrap().remove(&id);
                    return Err(e);
                }
                let deadline = Instant::now() + timeout;
                loop {
                    if stop.stopped() {
                        self.notify("notifications/cancelled", json!({ "requestId": id, "reason": "Stopped" }));
                        if matches!(rx.recv_timeout(CANCEL_GRACE), Err(std::sync::mpsc::RecvTimeoutError::Timeout)) {
                            self.force_close();
                        }
                        pending.lock().unwrap().remove(&id);
                        return Err("stopped".into());
                    }
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        pending.lock().unwrap().remove(&id);
                        return Err(format!("MCP {} {method} timed out", self.name));
                    }
                    match rx.recv_timeout(remaining.min(POLL)) {
                        Ok(r) => return if stop.stopped() { Err("stopped".into()) } else { r },
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {},
                        Err(_) => { pending.lock().unwrap().remove(&id); return Err("MCP server exited".into()); }
                    }
                }
            }
        }
    }

    pub fn notify(&self, method: &str, params: Value) {
        if matches!(self.transport, Transport::Http { .. }) {
            return;
        }
        let body = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        let _ = self.write_frame(&body.to_string());
    }

    pub fn call_tool(&self, name: &str, args: &Value, timeout: Duration) -> Result<String, String> {
        self.call_tool_cancellable(name, args, timeout, &Cancel::never())
    }

    pub fn call_tool_cancellable(&self, name: &str, args: &Value, timeout: Duration, stop: &Cancel) -> Result<String, String> {
        let result = self.request_cancellable(
            "tools/call",
            json!({ "name": name, "arguments": args }),
            timeout, stop,
        )?;
        Ok(flatten_content(&result))
    }

    pub fn is_running(&self) -> bool {
        if self.forced_closed.load(Ordering::SeqCst) { return false; }
        match &self.transport {
            Transport::Http { .. } => true,
            Transport::Stdio { child, .. } => child.lock().map(|mut c| matches!(c.try_wait(), Ok(None))).unwrap_or(false),
        }
    }

    fn force_close(&self) {
        if self.forced_closed.swap(true, Ordering::SeqCst) { return; }
        if let Transport::Stdio { child, .. } = &self.transport {
            if let Ok(mut c) = child.lock() {
                if matches!(c.try_wait(), Ok(None)) {
                    #[cfg(unix)]
                    unsafe { libc::kill(-(c.id() as i32), libc::SIGKILL); }
                    #[cfg(not(unix))]
                    let _ = c.kill();
                    let _ = c.wait();
                }
            }
        }
    }

    pub fn close(&self) { self.force_close(); }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        self.close();
    }
}

pub fn connect_http(name: &str, url: &str, timeout: Duration) -> Result<Arc<McpServer>, String> {
    connect_http_cancellable(name, url, timeout, &Cancel::never())
}

pub fn connect_http_cancellable(name: &str, url: &str, timeout: Duration, stop: &Cancel) -> Result<Arc<McpServer>, String> {
    let listed = http_rpc(url, "tools/list", json!({}), timeout, stop)?;
    Ok(Arc::new(McpServer {
        name: name.to_string(),
        tools: parse_tools(&listed),
        forced_closed: AtomicBool::new(false),
        transport: Transport::Http {
            url: url.to_string(),
        },
    }))
}

fn http_rpc(url: &str, method: &str, params: Value, timeout: Duration, stop: &Cancel) -> Result<Value, String> {
    use std::io::{Read as _, Write as _};
    use std::net::TcpStream;

    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("only http:// MCP urls are supported: {url}"))?;
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let host = hostport.split(':').next().unwrap_or(hostport).to_string();
    let addr: std::net::SocketAddr = hostport
        .parse()
        .map_err(|_| format!("bad MCP address: {hostport}"))?;

    let body = json!({ "id": 1, "method": method, "params": params }).to_string();
    let deadline = Instant::now() + timeout;
    let mut sock = loop {
        if stop.stopped() { return Err("stopped".into()); }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() { return Err(format!("MCP over HTTP: cannot reach {hostport}: timed out")); }
        match TcpStream::connect_timeout(&addr, remaining.min(POLL)) {
            Ok(sock) => break sock,
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => continue,
            Err(e) => return Err(format!("MCP over HTTP: cannot reach {hostport}: {e}")),
        }
    };
    sock.set_read_timeout(Some(POLL)).ok();
    sock.set_write_timeout(Some(POLL)).ok();
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    for mut bytes in [head.as_bytes(), body.as_bytes()] {
        while !bytes.is_empty() {
            if stop.stopped() { return Err("stopped".into()); }
            if Instant::now() >= deadline { return Err("MCP over HTTP timed out".into()); }
            match sock.write(bytes) {
                Ok(0) => return Err("MCP over HTTP: connection closed".into()),
                Ok(n) => bytes = &bytes[n..],
                Err(e) if matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted) => continue,
                Err(e) => return Err(e.to_string()),
            }
        }
    }

    let mut raw = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        if stop.stopped() { return Err("stopped".into()); }
        if Instant::now() >= deadline { return Err("MCP over HTTP timed out".into()); }
        match sock.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => raw.extend_from_slice(&buf[..n]),
            Err(e) if matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted) => continue,
            Err(e) => return Err(e.to_string()),
        }
    }
    let text = String::from_utf8_lossy(&raw).to_string();
    let split = text
        .find("\r\n\r\n")
        .ok_or_else(|| "MCP over HTTP: malformed response".to_string())?;
    let (headers, rest) = text.split_at(split);
    let status: u16 = headers
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    let payload: Value = serde_json::from_str(rest.trim_start_matches("\r\n\r\n").trim())
        .unwrap_or(Value::Null);

    if !(200..300).contains(&status) {
        let msg = payload
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("HTTP {status}"));
        return Err(msg);
    }
    Ok(payload
        .get("result")
        .cloned()
        .unwrap_or_else(|| payload.clone()))
}

fn read_loop(stdout: std::process::ChildStdout, framing: Framing, pending: &Pending) {
    let mut reader = BufReader::new(stdout);
    match framing {
        Framing::Newline => {
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {}
                }
                let t = line.trim();
                if t.is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<Value>(t) {
                    dispatch(&v, pending);
                }
            }
        }
        Framing::Lsp => loop {
            let mut header = String::new();
            let mut len: Option<usize> = None;
            loop {
                let mut l = String::new();
                match reader.read_line(&mut l) {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {}
                }
                if l.trim().is_empty() {
                    break;
                }
                header.push_str(&l);
                if let Some(rest) = l.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = rest.trim().parse().ok();
                }
            }
            let Some(n) = len else { continue };
            let mut body = vec![0u8; n];
            if reader.read_exact(&mut body).is_err() {
                return;
            }
            if let Ok(v) = serde_json::from_slice::<Value>(&body) {
                dispatch(&v, pending);
            }
        },
    }
}

fn dispatch(msg: &Value, pending: &Pending) {
    let Some(id) = msg.get("id").and_then(|v| v.as_u64()) else {
        return;
    };
    let Some(tx) = pending.lock().unwrap().remove(&id) else {
        return;
    };
    if let Some(err) = msg.get("error") {
        let m = err
            .get("message")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| err.to_string());
        let _ = tx.send(Err(m));
    } else {
        let _ = tx.send(Ok(msg.get("result").cloned().unwrap_or(Value::Null)));
    }
}

fn parse_tools(listed: &Value) -> Vec<ToolDef> {
    listed
        .get("tools")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|t| {
                    let name = t.get("name")?.as_str()?.to_string();
                    Some(ToolDef {
                        name,
                        description: t
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        input_schema: t
                            .get("inputSchema")
                            .cloned()
                            .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn media_ref_for_image(item: &Value) -> Option<String> {
    use base64::Engine as _;
    let data = item.get("data").and_then(|v| v.as_str()).unwrap_or("");
    let data: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    if data.is_empty() {
        return None;
    }
    let mime = item.get("mimeType").and_then(|v| v.as_str()).unwrap_or("");
    let bytes = base64::engine::general_purpose::STANDARD.decode(&data).ok()?;
    crate::media::save_media_bytes(&bytes, ext_from_mime(mime))
}

fn ext_from_mime(mime: &str) -> &'static str {
    match mime.trim().to_ascii_lowercase().as_str() {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        _ => "bin",
    }
}

pub fn flatten_content(result: &Value) -> String {
    let mut out = String::new();
    if let Some(items) = result.get("content").and_then(|v| v.as_array()) {
        for item in items {
            match item.get("type").and_then(|v| v.as_str()) {
                Some("text") => {
                    if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(t);
                    }
                }
                Some("image") => match media_ref_for_image(item) {
                    Some(name) => {
                        out.push_str(&format!("\n![image](/media/{name})\n"));
                    }
                    None => {
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str("[image content omitted]");
                    }
                },
                Some(other) => {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&format!("[{other} content omitted]"));
                }
                None => {}
            }
        }
    }
    if out.is_empty() {
        if result.is_null() {
            return String::new();
        }
        return result.to_string();
    }
    if result.get("isError").and_then(|v| v.as_bool()).unwrap_or(false) {
        return format!("error: {out}");
    }
    out
}

pub fn schema_parts(schema: &Value) -> (Value, Vec<String>) {
    let props = schema
        .get("properties")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let required = schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    (props, required)
}
