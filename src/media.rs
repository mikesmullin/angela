use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use agl::provider::Message;
use serde_json::{json, Value};

static MEDIA_ROOT: OnceLock<PathBuf> = OnceLock::new();

pub fn set_media_root(dir: PathBuf) {
    let _ = MEDIA_ROOT.set(dir);
}

pub fn media_root() -> Option<&'static Path> {
    MEDIA_ROOT.get().map(|p| p.as_path())
}

const MAX_MEDIA_BYTES: usize = 20 * 1024 * 1024;

pub fn save_media_bytes(data: &[u8], ext: &str) -> Option<String> {
    let root = media_root()?;
    write_media_file(root, data, ext)
}

fn write_media_file(root: &Path, data: &[u8], ext: &str) -> Option<String> {
    if data.len() > MAX_MEDIA_BYTES {
        return None;
    }
    let ext = sanitize_ext(ext);
    std::fs::create_dir_all(root).ok()?;
    let name = format!("mcp-{}.{}", rand_hex(4), ext);
    let tmp = root.join(format!(".{name}.tmp-{}", std::process::id()));
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp).ok()?;
        if f.write_all(data).and_then(|_| f.sync_all()).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return None;
        }
    }
    std::fs::rename(&tmp, root.join(&name)).ok()?;
    Some(name)
}

fn sanitize_ext(ext: &str) -> String {
    let mut out = String::new();
    for c in ext.chars() {
        if !c.is_ascii_alphanumeric() {
            return "bin".into();
        }
        if out.len() >= 10 {
            return "bin".into();
        }
        out.push(c.to_ascii_lowercase());
    }
    if out.is_empty() { "bin".into() } else { out }
}

fn rand_hex(bytes: usize) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    let mut x = t
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(n.wrapping_mul(1_442_695_040_888_963_407));
    let mut out = String::new();
    for _ in 0..bytes {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.push_str(&format!("{:02x}", (x >> 24) as u8));
    }
    out
}

pub fn with_images(project_root: &Path, msg: Message) -> Message {
    if msg.role != "user" && msg.role != "tool" {
        return msg;
    }
    let Some(Value::String(text)) = msg.content.as_ref() else {
        return msg;
    };
    if !text.contains("![") {
        return msg;
    }
    let media_root = project_root.join(".angela").join("media");
    let mut parts: Vec<Value> = Vec::new();
    let mut clean = String::with_capacity(text.len());
    let mut rest = text.as_str();
    while let Some(start) = rest.find("![") {
        let (head, tail) = rest.split_at(start);
        let Some(close) = tail.find(')') else {
            clean.push_str(rest);
            break;
        };
        let ref_text = &tail[..=close];
        let url = ref_text
            .split_once("](")
            .map(|(_, u)| u.trim_end_matches(')'))
            .unwrap_or("");
        if url.starts_with("/media/") {
            match image_part(&media_root, url.trim_start_matches("/media/")) {
                Some(part) => {
                    parts.push(part);
                    clean.push_str(head);
                    clean.push_str("[attached image]");
                }
                None => {
                    clean.push_str(head);
                    clean.push_str(ref_text);
                }
            }
        } else {
            clean.push_str(head);
            clean.push_str(ref_text);
        }
        rest = &tail[close + 1..];
    }
    clean.push_str(rest);

    if parts.is_empty() {
        return msg;
    }
    let mut out = vec![json!({ "type": "text", "text": clean })];
    out.extend(parts);
    Message {
        role: msg.role,
        content: Some(Value::Array(out)),
        tool_calls: msg.tool_calls,
        tool_call_id: msg.tool_call_id,
        reasoning_content: msg.reasoning_content,
    }
}

fn image_part(media_root: &Path, file: &str) -> Option<Value> {
    if file.is_empty() || file.contains('/') || file.contains('\\') || file.contains("..") {
        return None;
    }
    let bytes = std::fs::read(media_root.join(file)).ok()?;
    let mime = match file.rsplit('.').next().unwrap_or("").to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "application/octet-stream",
    };
    use base64::Engine as _;
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    Some(json!({
        "type": "image_url",
        "image_url": { "url": format!("data:{mime};base64,{encoded}") },
    }))
}
