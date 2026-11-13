//! A deliberately small HTTP/1.1 server: `std::net` + a thread per
//! connection, no framework.
//!
//! angela's surface is a handful of JSON routes plus one streaming endpoint,
//! so an async web stack would be more dependency than the job needs. What it
//! does need, and what this provides, is honest **chunked** streaming: the
//! chat endpoint emits cards as the model produces them, and a response that
//! buffered to completion would defeat the whole point.
//!
//! Not implemented, because angela never uses it: keep-alive (every response
//! closes), HTTP/2, TLS, multipart, compression.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use serde_json::Value;

/// Cap on a request body. A chat prompt is text; anything near this is abuse.
const MAX_BODY: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    /// Path with the query string removed.
    pub path: String,
    pub query: String,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

impl Request {
    /// The body as JSON, or `null` when absent/unparseable — the JS does
    /// `await req.json().catch(() => ({}))` at every call site, so a
    /// malformed body is a default, never a 500.
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }

    /// A query parameter, percent-decoded.
    pub fn query_param(&self, key: &str) -> Option<String> {
        for pair in self.query.split('&') {
            let (k, v) = match pair.split_once('=') {
                Some((k, v)) => (k, v),
                None => (pair, ""),
            };
            if percent_decode(k) == key {
                return Some(percent_decode(v));
            }
        }
        None
    }
}

/// Minimal percent-decoding, `+` meaning space as in a query string.
pub fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                let hex = std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(n) => {
                        out.push(n);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(b[i]);
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Writes one response. Consumed by whichever method you call, so a handler
/// cannot accidentally send two.
pub struct Responder<'a> {
    stream: &'a mut TcpStream,
}

impl<'a> Responder<'a> {
    fn head(&mut self, status: u16, content_type: &str, extra: &[(&str, &str)]) -> std::io::Result<()> {
        let reason = match status {
            200 => "OK",
            201 => "Created",
            204 => "No Content",
            400 => "Bad Request",
            404 => "Not Found",
            405 => "Method Not Allowed",
            413 => "Payload Too Large",
            500 => "Internal Server Error",
            501 => "Not Implemented",
            _ => "OK",
        };
        write!(self.stream, "HTTP/1.1 {status} {reason}\r\n")?;
        write!(self.stream, "Content-Type: {content_type}\r\n")?;
        // The SPA is served from a different origin in dev.
        write!(self.stream, "Access-Control-Allow-Origin: *\r\n")?;
        write!(
            self.stream,
            "Access-Control-Allow-Headers: content-type\r\n"
        )?;
        write!(
            self.stream,
            "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
        )?;
        for (k, v) in extra {
            write!(self.stream, "{k}: {v}\r\n")?;
        }
        Ok(())
    }

    /// A buffered JSON response.
    pub fn json(mut self, status: u16, body: &Value) -> std::io::Result<()> {
        let bytes = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
        self.head(status, "application/json", &[])?;
        write!(self.stream, "Content-Length: {}\r\n", bytes.len())?;
        write!(self.stream, "Connection: close\r\n\r\n")?;
        self.stream.write_all(&bytes)?;
        self.stream.flush()
    }

    pub fn empty(mut self, status: u16) -> std::io::Result<()> {
        self.head(status, "text/plain", &[])?;
        write!(self.stream, "Content-Length: 0\r\n")?;
        write!(self.stream, "Connection: close\r\n\r\n")?;
        self.stream.flush()
    }

    /// Begin a chunked NDJSON stream.
    ///
    /// Chunked rather than a buffered body because each card must reach the
    /// client as it happens; `Content-Length` would force us to know the whole
    /// response up front, which is exactly what streaming rules out.
    pub fn ndjson(mut self) -> std::io::Result<NdjsonWriter<'a>> {
        self.head(200, "application/x-ndjson", &[("Cache-Control", "no-cache")])?;
        write!(self.stream, "Transfer-Encoding: chunked\r\n")?;
        write!(self.stream, "Connection: close\r\n\r\n")?;
        self.stream.flush()?;
        Ok(NdjsonWriter {
            stream: self.stream,
            done: false,
        })
    }
}

/// Emits newline-delimited JSON as chunked HTTP.
pub struct NdjsonWriter<'a> {
    stream: &'a mut TcpStream,
    done: bool,
}

impl<'a> NdjsonWriter<'a> {
    /// Send one row. An error here normally means the client hung up, which
    /// the caller should treat as "stop working", not as a failure to report.
    pub fn send(&mut self, value: &Value) -> std::io::Result<()> {
        let mut line = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
        line.push(b'\n');
        write!(self.stream, "{:x}\r\n", line.len())?;
        self.stream.write_all(&line)?;
        write!(self.stream, "\r\n")?;
        self.stream.flush()
    }

    pub fn finish(mut self) -> std::io::Result<()> {
        self.done = true;
        write!(self.stream, "0\r\n\r\n")?;
        self.stream.flush()
    }
}

impl<'a> Drop for NdjsonWriter<'a> {
    fn drop(&mut self) {
        if !self.done {
            // Terminate the stream even on an early return, or the client
            // waits forever for a chunk that is never coming.
            let _ = write!(self.stream, "0\r\n\r\n");
            let _ = self.stream.flush();
        }
    }
}

/// Where a route writes its answer.
///
/// The router must not know whether it is writing to a socket it owns or to
/// a host application's response stream, so it writes through this instead of
/// touching bytes. `angela-serve` provides [`SocketSink`]; an embedding
/// server (chat-server in psicom/chat) provides its own.
pub trait ResponseSink {
    /// A complete JSON answer.
    fn json(&mut self, status: u16, body: &Value);
    /// Begin an NDJSON stream. Followed by `row`s, then `end`.
    fn begin_ndjson(&mut self);
    /// One row. Returns `false` when the client has gone — the caller should
    /// stop working rather than finish a turn nobody will see.
    fn row(&mut self, value: &Value) -> bool;
    /// Finish the response.
    fn end(&mut self);
}

/// What a server implementation must provide.
pub trait Handler: Send + Sync + 'static {
    fn handle(&self, req: &Request, out: Responder<'_>);
}

/// [`ResponseSink`] over a socket this crate owns.
pub struct SocketSink<'a> {
    responder: Option<Responder<'a>>,
    writer: Option<NdjsonWriter<'a>>,
}

impl<'a> SocketSink<'a> {
    pub fn new(responder: Responder<'a>) -> Self {
        SocketSink {
            responder: Some(responder),
            writer: None,
        }
    }
}

impl<'a> ResponseSink for SocketSink<'a> {
    fn json(&mut self, status: u16, body: &Value) {
        if let Some(r) = self.responder.take() {
            let _ = r.json(status, body);
        }
    }
    fn begin_ndjson(&mut self) {
        if let Some(r) = self.responder.take() {
            self.writer = r.ndjson().ok();
        }
    }
    fn row(&mut self, value: &Value) -> bool {
        match self.writer.as_mut() {
            Some(w) => w.send(value).is_ok(),
            None => false,
        }
    }
    fn end(&mut self) {
        if let Some(w) = self.writer.take() {
            let _ = w.finish();
        }
    }
}

/// Read one request. `None` on a closed or malformed connection.
fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);

    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 {
        return None;
    }
    let mut parts = line.trim_end().split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?.to_string();

    let mut headers = BTreeMap::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).ok()? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }

    let len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if len > MAX_BODY {
        return None;
    }
    let mut body = vec![0u8; len];
    if len > 0 && reader.read_exact(&mut body).is_err() {
        return None;
    }

    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target, String::new()),
    };
    Some(Request {
        method,
        path,
        query,
        headers,
        body,
    })
}

/// Bind and serve forever, one thread per connection.
pub fn serve(addr: &str, handler: Arc<dyn Handler>) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    for conn in listener.incoming() {
        let Ok(mut stream) = conn else { continue };
        let h = Arc::clone(&handler);
        std::thread::spawn(move || {
            if let Some(req) = read_request(&mut stream) {
                // CORS preflight never reaches a route.
                if req.method == "OPTIONS" {
                    let _ = Responder { stream: &mut stream }.empty(204);
                    return;
                }
                h.handle(&req, Responder { stream: &mut stream });
            }
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_params_are_split_and_decoded() {
        let r = Request {
            method: "GET".into(),
            path: "/api/pending".into(),
            query: "tabId=1&name=a%20b&flag&plus=a+b".into(),
            headers: BTreeMap::new(),
            body: Vec::new(),
        };
        assert_eq!(r.query_param("tabId").as_deref(), Some("1"));
        assert_eq!(r.query_param("name").as_deref(), Some("a b"));
        assert_eq!(r.query_param("flag").as_deref(), Some(""));
        assert_eq!(r.query_param("plus").as_deref(), Some("a b"));
        assert_eq!(r.query_param("missing"), None);
    }

    #[test]
    fn a_malformed_body_is_null_not_an_error() {
        let r = Request {
            method: "POST".into(),
            path: "/api/chat".into(),
            query: String::new(),
            headers: BTreeMap::new(),
            body: b"{not json".to_vec(),
        };
        assert!(r.json().is_null());
    }

    #[test]
    fn percent_decoding_handles_truncated_escapes() {
        assert_eq!(percent_decode("a%2"), "a%2");
        assert_eq!(percent_decode("a%zzb"), "a%zzb");
        assert_eq!(percent_decode("%2F"), "/");
    }
}
