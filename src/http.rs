use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use serde_json::Value;

const MAX_BODY: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }

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

pub struct NdjsonWriter<'a> {
    stream: &'a mut TcpStream,
    done: bool,
}

impl<'a> NdjsonWriter<'a> {
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
            let _ = write!(self.stream, "0\r\n\r\n");
            let _ = self.stream.flush();
        }
    }
}

pub trait ResponseSink {
    fn json(&mut self, status: u16, body: &Value);
    fn begin_ndjson(&mut self);
    fn row(&mut self, value: &Value) -> bool;
    fn end(&mut self);
}

pub trait Handler: Send + Sync + 'static {
    fn handle(&self, req: &Request, out: Responder<'_>);
}

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

pub fn serve(addr: &str, handler: Arc<dyn Handler>) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    for conn in listener.incoming() {
        let Ok(mut stream) = conn else { continue };
        let h = Arc::clone(&handler);
        std::thread::spawn(move || {
            if let Some(req) = read_request(&mut stream) {
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
