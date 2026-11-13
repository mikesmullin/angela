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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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

/// One connected MCP server.
pub struct McpServer {
    pub name: String,
    pub tools: Vec<ToolDef>,
    stdin: Mutex<ChildStdin>,
    child: Mutex<Child>,
    pending: Pending,
    next_id: AtomicU64,
    framing: Framing,
}

impl McpServer {
    /// Spawn, `initialize`, and read the tool list.
    pub fn connect(spec: &ServerSpec, timeout: Duration) -> Result<Arc<Self>, String> {
        let mut cmd = Command::new(&spec.command);
        cmd.args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
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
            stdin: Mutex::new(stdin),
            child: Mutex::new(child),
            pending,
            next_id: AtomicU64::new(1),
            framing: spec.framing,
        });

        server.request(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "angela", "version": "0.1.0" },
            }),
            timeout,
        )?;
        server.notify("notifications/initialized", json!({}));

        let listed = server.request("tools/list", json!({}), timeout)?;
        let tools = parse_tools(&listed);

        // `tools` is set after connect, so rebuild the Arc with it in place.
        let mut inner = Arc::try_unwrap(server).map_err(|_| "server aliased".to_string())?;
        inner.tools = tools;
        Ok(Arc::new(inner))
    }

    fn write_frame(&self, text: &str) -> Result<(), String> {
        let mut out = self.stdin.lock().unwrap();
        match self.framing {
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
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx): (Sender<_>, Receiver<_>) = channel();
        self.pending.lock().unwrap().insert(id, tx);

        let body = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        if let Err(e) = self.write_frame(&body.to_string()) {
            self.pending.lock().unwrap().remove(&id);
            return Err(e);
        }
        match rx.recv_timeout(timeout) {
            Ok(r) => r,
            Err(_) => {
                self.pending.lock().unwrap().remove(&id);
                Err(format!("MCP {} {method} timed out", self.name))
            }
        }
    }

    /// A notification, which has no reply.
    pub fn notify(&self, method: &str, params: Value) {
        let body = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        let _ = self.write_frame(&body.to_string());
    }

    /// `tools/call`, flattened to the text the model should see.
    pub fn call_tool(&self, name: &str, args: &Value, timeout: Duration) -> Result<String, String> {
        let result = self.request(
            "tools/call",
            json!({ "name": name, "arguments": args }),
            timeout,
        )?;
        Ok(flatten_content(&result))
    }

    pub fn close(&self) {
        // Closing stdin is the polite stop; the kill is the backstop.
        if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
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
                // Images and resources have no text form the model can read
                // here; name them so a turn does not look silently empty.
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
