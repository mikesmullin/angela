//! A minimal MCP stdio client — port of `angela/src/mcp-client.mjs`.
//!
//! Speaks JSON-RPC 2.0 over a child process's stdin/stdout. Two framings
//! exist in the wild and both are supported: newline-delimited (the file-io
//! kit's servers) and LSP `Content-Length` (the official SDK). Default is
//! newline, matching the JS.
//!
//! Synchronous by design: the harness drives tools from a blocking turn loop,
//! so a reader thread plus channels fits better than an async client that
//! would need its own runtime inside one that already exists.
//!
//! The servers themselves are **not** ported — they are run as-is from
//! `/workspace/mcp/*`.

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

/// How a server frames messages on stdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// One JSON object per line — the file-io / term-local kit.
    Newline,
    /// `Content-Length: n\r\n\r\n<body>` — the official MCP SDK.
    Lsp,
}

/// A tool as the server advertises it.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// The raw `inputSchema`, passed to the model unchanged.
    pub input_schema: Value,
}

/// How to start one server.
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

/// How a connected server is reached.
///
/// Stdio is the original and still the normal case: a child process, framed
/// JSON-RPC over its pipes. `Http` exists because a **sandbox's** tools are
/// not a process on this machine — they run in a container, and the only way
/// in is sbox's HTTP facade. Teaching the existing client a second transport
/// keeps one MCP client, one tool-registration path and one allowlist gate;
/// the alternative was a second client, or a local process whose whole job
/// was to relay to that URL.
enum Transport {
    Stdio {
        stdin: Mutex<ChildStdin>,
        child: Mutex<Child>,
        pending: Pending,
        next_id: AtomicU64,
        framing: Framing,
    },
    /// One POST per request. sbox's facade is request/response and
    /// stateless per call — there is no stream to hold open, so there is no
    /// pending map and no reader thread.
    Http { url: String },
}

/// One connected MCP server.
pub struct McpServer {
    pub name: String,
    pub tools: Vec<ToolDef>,
    transport: Transport,
    forced_closed: AtomicBool,
}

impl McpServer {
    /// Spawn, `initialize`, and read the tool list.
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
        // Each owned stdio server gets its own group, so escalation reaches
        // subprocesses without signalling the chat server or other MCPs.
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

        // Reader thread: parse frames and hand each reply to its waiter.
        {
            let pending = Arc::clone(&pending);
            let framing = spec.framing;
            let name = spec.name.clone();
            std::thread::spawn(move || {
                read_loop(stdout, framing, &pending);
                // The server died: fail every waiter instead of letting a
                // turn hang forever on a reply that cannot arrive.
                let mut p = pending.lock().unwrap();
                for (_, tx) in p.drain() {
                    let _ = tx.send(Err(format!("MCP server {name} exited")));
                }
            });
        }
        // Server logs are useful but must not interleave into stdout.
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

        // `tools` is set after connect, so rebuild the Arc with it in place.
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

    /// A request that expects a reply.
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
                        // Keep the waiter registered during the grace period:
                        // a reply means this request has actually finished.
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

    /// A notification, which has no reply.
    pub fn notify(&self, method: &str, params: Value) {
        // Nothing to tell an HTTP facade: it holds no session between calls,
        // so there is no "initialized" state for a notification to advance.
        if matches!(self.transport, Transport::Http { .. }) {
            return;
        }
        let body = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        let _ = self.write_frame(&body.to_string());
    }

    /// `tools/call`, flattened to the text the model should see.
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

    /// Whether a cached server can still accept requests.
    pub fn is_running(&self) -> bool {
        if self.forced_closed.load(Ordering::SeqCst) { return false; }
        match &self.transport {
            Transport::Http { .. } => true,
            Transport::Stdio { child, .. } => child.lock().map(|mut c| matches!(c.try_wait(), Ok(None))).unwrap_or(false),
        }
    }

    /// The server is shared, so force-closing it also interrupts any other
    /// calls using it. The pool discards it and starts a fresh one next time.
    fn force_close(&self) {
        if self.forced_closed.swap(true, Ordering::SeqCst) { return; }
        if let Transport::Stdio { child, .. } = &self.transport {
            if let Ok(mut c) = child.lock() {
                // A leader already reaped by try_wait could have its PID
                // reused; never send a group signal to that old number.
                if matches!(c.try_wait(), Ok(None)) {
                    #[cfg(unix)]
                    // The live, unreaped leader reserves this process group.
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

/// Connect to an MCP server reached over HTTP, and read its tool list.
///
/// Used for a sandbox: the tools run in a container, and sbox's facade is
/// the door. There is no handshake to perform — the facade does its own
/// against the guest daemon — so this goes straight to `tools/list`.
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

/// One JSON-RPC call over HTTP.
///
/// Hand-rolled for the same reason the rest of this file is: one loopback
/// JSON POST, no TLS and no redirects, against a service on this machine.
///
/// sbox's facade takes `{id, method, params}` and **adds** the `jsonrpc`
/// field itself — sending it is an `unknown_field` error — and answers
/// `{ok, result}` rather than a bare JSON-RPC envelope. Both quirks are
/// absorbed here so everything above this line sees ordinary MCP.
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
    // `{ok, result}` is the facade's envelope; a bare `result` is accepted
    // too so this is not tied to one server's shape.
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
                // A junk line is skipped, not fatal: servers sometimes print.
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
        return; // a notification from the server; nothing waits on it
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

/// Save an MCP image block's bytes to the media root and return the file
/// name. `None` when the block is not decodable, the root is unregistered,
/// the data is oversized or the write fails.
fn media_ref_for_image(item: &Value) -> Option<String> {
    use base64::Engine as _;
    let data = item.get("data").and_then(|v| v.as_str()).unwrap_or("");
    // Servers wrap long blocks in newlines; the alphabet does not care.
    let data: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    if data.is_empty() {
        return None;
    }
    let mime = item.get("mimeType").and_then(|v| v.as_str()).unwrap_or("");
    let bytes = base64::engine::general_purpose::STANDARD.decode(&data).ok()?;
    crate::media::save_media_bytes(&bytes, ext_from_mime(mime))
}

/// File extension for an MCP image block's `mimeType`. Unknown types become
/// `bin` — the model never reads the bytes, but an extension that is honest
/// about the shape is still worth having.
fn ext_from_mime(mime: &str) -> &'static str {
    match mime.trim().to_ascii_lowercase().as_str() {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        _ => "bin",
    }
}

/// MCP returns `content: [{type:'text', text}, …]`. The model wants text.
///
/// An `isError` result is still returned as text rather than raised: the
/// model can read a tool error and try something else, which is more useful
/// than failing the turn.
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
                // An image block is not a thing the model can read here, but
                // it can be *saved*: the bytes join the media folder and the
                // tool result carries a `/media/…` ref, which reaches the
                // model as an image part at the provider boundary like any
                // other attachment. If the save fails, the marker tells the
                // model the block was there rather than silently dropping it.
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
                // Resources (and anything else) have no text form the model
                // can read here; name them so a turn does not look silently
                // empty.
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
        // Some servers answer with a bare structured result.
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

/// Split an MCP `inputSchema` into the `(properties, required)` pair agl's
/// tool registration wants.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_server(script: &str) -> ServerSpec {
        ServerSpec { name: "test".into(), command: "python3".into(),
            args: vec!["-u".into(), "-c".into(), script.into()], cwd: None,
            env: vec![], framing: Framing::Newline }
    }

    #[test]
    fn stop_interrupts_stdio_connect() {
        let stop = Cancel::never();
        let worker_stop = stop.clone();
        let spec = fake_server("import sys,time; sys.stdin.readline(); time.sleep(30)");
        let start = Instant::now();
        let worker = std::thread::spawn(move || McpServer::connect_cancellable(&spec, Duration::from_secs(20), &worker_stop));
        std::thread::sleep(Duration::from_millis(100));
        stop.stop();
        assert!(worker.join().unwrap().is_err());
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn stop_interrupts_stdio_tool_and_sends_cancel_notification() {
        let script = r#"import sys,json
for line in sys.stdin:
 m=json.loads(line)
 if m.get('method') == 'initialize': print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{}}), flush=True)
 elif m.get('method') == 'tools/list': print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{'tools':[]}}), flush=True)
 elif m.get('method') == 'tools/call': print(json.dumps({'jsonrpc':'2.0','method':'test/called'}), flush=True)
 elif m.get('method') == 'notifications/cancelled': print(json.dumps({'jsonrpc':'2.0','id':999,'result':m['params']}), flush=True)
"#;
        let srv = McpServer::connect(&fake_server(script), Duration::from_secs(2)).unwrap();
        let (tx, rx) = channel();
        if let Transport::Stdio { pending, .. } = &srv.transport { pending.lock().unwrap().insert(999, tx); }
        let stop = Cancel::never();
        let worker_stop = stop.clone();
        let worker_srv = srv.clone();
        let start = Instant::now();
        let worker = std::thread::spawn(move || worker_srv.call_tool_cancellable("slow", &json!({}), Duration::from_secs(20), &worker_stop));
        std::thread::sleep(Duration::from_millis(100));
        stop.stop();
        assert_eq!(worker.join().unwrap().unwrap_err(), "stopped");
        let notification = rx.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
        assert_eq!(notification["requestId"], 3);
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    #[cfg(unix)]
    #[test]
    fn uncooperative_tool_and_its_child_are_killed_after_grace() {
        let script = r#"import sys,json,subprocess
child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
for line in sys.stdin:
 m=json.loads(line)
 if m.get('method') == 'initialize': print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{}}), flush=True)
 elif m.get('method') == 'tools/list': print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{'tools':[{'name':str(child.pid)}]}}), flush=True)
"#;
        let srv = McpServer::connect(&fake_server(script), Duration::from_secs(2)).unwrap();
        let child_pid: i32 = srv.tools[0].name.parse().unwrap();
        let stop = Cancel::never();
        let worker_stop = stop.clone();
        let worker_srv = srv.clone();
        let worker = std::thread::spawn(move || worker_srv.call_tool_cancellable("slow", &json!({}), Duration::from_secs(20), &worker_stop));
        std::thread::sleep(Duration::from_millis(100));
        stop.stop();
        assert_eq!(worker.join().unwrap().unwrap_err(), "stopped");
        assert!(!srv.is_running());
        // An adopted zombie has terminated even if its new parent has not
        // reaped it yet; it cannot continue doing work.
        let ended = (0..100).any(|_| {
            let state = std::fs::read_to_string(format!("/proc/{child_pid}/stat"));
            if state.as_deref().map(|s| s.split_whitespace().nth(2) == Some("Z")).unwrap_or(true) { true }
            else { std::thread::sleep(Duration::from_millis(10)); false }
        });
        assert!(ended, "tool subprocess is still running");
    }

    #[test]
    fn cooperative_tool_finishes_without_killing_the_server() {
        let script = r#"import sys,json
for line in sys.stdin:
 m=json.loads(line)
 method=m.get('method')
 if method == 'initialize': print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{}}), flush=True)
 elif method == 'tools/list': print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{'tools':[]}}), flush=True)
 elif method == 'notifications/cancelled': print(json.dumps({'jsonrpc':'2.0','id':m['params']['requestId'],'error':{'code':-32800,'message':'cancelled'}}), flush=True)
"#;
        let srv = McpServer::connect(&fake_server(script), Duration::from_secs(2)).unwrap();
        let stop = Cancel::never();
        let worker_stop = stop.clone();
        let worker_srv = srv.clone();
        let worker = std::thread::spawn(move || worker_srv.call_tool_cancellable("slow", &json!({}), Duration::from_secs(20), &worker_stop));
        std::thread::sleep(Duration::from_millis(100));
        stop.stop();
        assert_eq!(worker.join().unwrap().unwrap_err(), "stopped");
        assert!(srv.is_running());
    }

    #[test]
    fn stop_interrupts_http_wait() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/rpc", listener.local_addr().unwrap());
        let (accepted, ready) = std::sync::mpsc::channel();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut buf = [0; 4096];
            let _ = socket.read(&mut buf);
            accepted.send(()).unwrap();
            let _ = socket.read(&mut buf);
        });
        let stop = Cancel::never();
        let worker_stop = stop.clone();
        let worker = std::thread::spawn(move || connect_http_cancellable("test", &url, Duration::from_secs(20), &worker_stop));
        ready.recv_timeout(Duration::from_secs(2)).unwrap();
        let start = Instant::now();
        stop.stop();
        assert!(worker.join().unwrap().is_err());
        assert!(start.elapsed() < Duration::from_secs(2));
        server.join().unwrap();
    }

    #[test]
    fn text_content_is_joined() {
        let r = json!({ "content": [{"type":"text","text":"a"},{"type":"text","text":"b"}] });
        assert_eq!(flatten_content(&r), "a\nb");
    }

    #[test]
    fn an_error_result_is_text_not_a_failure() {
        // The model should get to read it and recover.
        let r = json!({ "content": [{"type":"text","text":"no such file"}], "isError": true });
        assert_eq!(flatten_content(&r), "error: no such file");
    }

    #[test]
    fn non_text_content_is_named_rather_than_dropped() {
        let r = json!({ "content": [{"type":"image","data":"…"}] });
        assert_eq!(flatten_content(&r), "[image content omitted]");
    }

    #[test]
    fn an_image_block_becomes_a_media_ref_with_the_bytes_on_disk() {
        let root = crate::media::test_media_root();
        use base64::Engine as _;
        let data = base64::engine::general_purpose::STANDARD.encode(b"PNGDATA");
        let r = json!({ "content": [{"type":"image","data":data,"mimeType":"image/png"}] });
        let out = flatten_content(&r);
        assert!(out.contains("![image](/media/"), "{out}");
        assert!(out.contains(".png)"), "{out}");
        let name = out
            .split_once("/media/")
            .and_then(|(_, rest)| rest.split(')').next())
            .unwrap()
            .to_string();
        assert!(name.starts_with("mcp-"), "{name}");
        let bytes = std::fs::read(root.join(&name)).expect("file saved");
        assert_eq!(bytes, b"PNGDATA");
    }

    #[test]
    fn an_oversized_image_block_falls_back_to_the_marker() {
        use base64::Engine as _;
        let big = vec![7u8; 20 * 1024 * 1024 + 1];
        let data = base64::engine::general_purpose::STANDARD.encode(&big);
        let r = json!({ "content": [{"type":"image","data":data,"mimeType":"image/png"}] });
        assert_eq!(flatten_content(&r), "[image content omitted]");
    }

    #[test]
    fn an_unreadable_image_block_falls_back_to_the_marker() {
        // Not valid base64 — never reaches the save path, and never panics.
        let r = json!({ "content": [{"type":"image","data":"!!!","mimeType":"image/png"}] });
        assert_eq!(flatten_content(&r), "[image content omitted]");
    }

    #[test]
    fn a_structured_result_falls_back_to_json() {
        let r = json!({ "ok": true });
        assert_eq!(flatten_content(&r), r#"{"ok":true}"#);
    }

    #[test]
    fn schema_parts_split_properties_and_required() {
        let s = json!({
            "type": "object",
            "properties": { "path": { "type": "string" }, "limit": { "type": "number" } },
            "required": ["path"]
        });
        let (props, req) = schema_parts(&s);
        assert!(props.get("path").is_some());
        assert_eq!(req, vec!["path".to_string()]);
    }

    #[test]
    fn a_schema_without_required_yields_an_empty_list() {
        let (_, req) = schema_parts(&json!({ "properties": {} }));
        assert!(req.is_empty());
    }

    #[test]
    fn tools_parse_from_a_list_reply() {
        let listed = json!({ "tools": [
            { "name": "read_file", "description": "Read", "inputSchema": { "properties": { "path": {} } } },
            { "description": "nameless, skipped" }
        ]});
        let tools = parse_tools(&listed);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "read_file");
    }
}

#[cfg(test)]
mod http_transport_tests {
    use super::*;

    /// Against a **live** sandbox, so it is not part of the normal run:
    /// `cargo test -- --ignored sandbox_tools` with sbox-b up and a ready
    /// box named in `SBOX_TEST_ID`.
    ///
    /// It exists because the two things most likely to be wrong here cannot
    /// be unit-tested — sbox's facade rejects a `jsonrpc` field and wraps
    /// its reply in `{ok, result}` — and both are quirks of a real server.
    #[test]
    #[ignore]
    fn sandbox_tools_list_over_http() {
        let Ok(id) = std::env::var("SBOX_TEST_ID") else {
            panic!("set SBOX_TEST_ID to a ready sandbox");
        };
        let url = format!("http://127.0.0.1:3000/api/orchestration/sandboxes/{id}/exec/rpc");
        let srv = connect_http("sandbox", &url, Duration::from_secs(30)).expect("connect");
        let names: Vec<&str> = srv.tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"bash"), "tools were {names:?}");
        assert!(names.contains(&"read"), "tools were {names:?}");

        let out = srv
            .call_tool(
                "bash",
                &json!({ "command": "echo angela-http-transport-ok" }),
                Duration::from_secs(30),
            )
            .expect("bash");
        assert!(out.contains("angela-http-transport-ok"), "got {out:?}");
    }
}
