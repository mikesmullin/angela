//! Images on the wire — an attachment as the model sees it.
//!
//! The transcript stores a pasted image as markdown (`![image](/media/x.jpg)`),
//! which is exactly right for reading and exactly useless for a vision model:
//! the inference endpoint needs pixels, not a path. [`with_images`] is the one
//! conversion — the message's text stays, and every `/media/…` ref becomes a
//! real `image_url` content part (a base64 data URL, the shape every
//! OpenAI-compatible endpoint — llama-server included — accepts).
//!
//! Where the bytes live is chat's media folder (`<project_root>/.angela/media`,
//! the `POST /api/upload` destination); the ref is just its file name. MCP
//! tool results that carry image content blocks are saved there the same way
//! ([`save_media_bytes`]) — the embedder registers the folder at startup via
//! [`set_media_root`].

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use agl::provider::Message;
use serde_json::{json, Value};

/// Where media bytes live. Set once at startup by the embedder; the first
/// call wins so a file already written through this root is never orphaned
/// by a later registration (tests re-inject the same process).
static MEDIA_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// The media root, registered once at startup by the embedder.
pub fn set_media_root(dir: PathBuf) {
    let _ = MEDIA_ROOT.set(dir);
}

/// The registered media root, if any.
pub fn media_root() -> Option<&'static Path> {
    MEDIA_ROOT.get().map(|p| p.as_path())
}

/// Refuse files past this — a tool result carrying an image bigger than this
/// is almost certainly not an image the model should see.
const MAX_MEDIA_BYTES: usize = 20 * 1024 * 1024;

/// Write `data` into the media root as `mcp-<hex>.<ext>` and return the file
/// name. `None` when no root is registered, the data is oversized, or the
/// write fails — the caller decides what a failed save means for the model.
pub fn save_media_bytes(data: &[u8], ext: &str) -> Option<String> {
    let root = media_root()?;
    write_media_file(root, data, ext)
}

/// The write itself, against an explicit root — the part a test can drive
/// without touching the process-wide registration.
fn write_media_file(root: &Path, data: &[u8], ext: &str) -> Option<String> {
    if data.len() > MAX_MEDIA_BYTES {
        return None;
    }
    let ext = sanitize_ext(ext);
    std::fs::create_dir_all(root).ok()?;
    let name = format!("mcp-{}.{}", rand_hex(4), ext);
    // Write to a sibling temp file and rename: a crash halfway through must
    // not leave a truncated image that a later read would serve.
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

/// An extension is safe only if it is a short run of alphanumerics — it is
/// interpolated into a file name, and anything else (a path, a multi-part
/// suffix, punctuation) is garbage, not a shape.
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

/// Small hex token, same shape as the JS's `randomBytes` — ids must not
/// collide within a process, and the guarantee is uniqueness, not secrecy.
fn rand_hex(bytes: usize) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    // Mix a monotonic counter with the clock: ids must not collide within a
    // process, and the JS guarantee is uniqueness, not cryptographic secrecy.
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

/// The provider's view of one **user** or **tool** message: its text with
/// every attached image as an image part. Anything else — an assistant reply,
/// a message with no `/media/…` ref — passes through untouched, so the wire
/// for ordinary turns is byte-identical to what it always was.
///
/// A ref whose file is gone is left as words: the transcript's text is the
/// truth, and a missing image must not kill the turn.
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
                    // The file is gone; the words stay exactly as written.
                    clean.push_str(head);
                    clean.push_str(ref_text);
                }
            }
        } else {
            // Not an attachment (a real link); leave it alone.
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

/// The `image_url` content part for one media file, if the bytes are there.
fn image_part(media_root: &Path, file: &str) -> Option<Value> {
    // The ref names a file; a `..`-style name has no business here and must
    // not reach the filesystem walk.
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

/// Test-only access to the seam in tests: sets the root exactly once per
/// process and returns the one it landed on. The registration is first-call
/// wins, so every test in the process agrees on one root — the dir this
/// helper created, or a foreign one set by an earlier test (AppState::new,
/// say) — and writes land where the assertions look for them.
#[cfg(test)]
pub(crate) fn test_media_root() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "angela-media-test-{}-{}",
            std::process::id(),
            n
        ));
        let _ = std::fs::create_dir_all(&dir);
        set_media_root(dir.clone());
        let landed = media_root().unwrap_or(dir.as_path()).to_path_buf();
        landed
    })
    .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(text: &str) -> Message {
        Message::user(text)
    }

    fn tool_msg(text: &str) -> Message {
        Message::tool_result("call-1", text)
    }

    fn temp_project_with_image(name: &str, bytes: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "media-test-{}-{}",
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let media = dir.join(".angela").join("media");
        std::fs::create_dir_all(&media).unwrap();
        std::fs::write(media.join(name), bytes).unwrap();
        dir
    }

    #[test]
    fn a_message_without_an_attachment_is_untouched() {
        let m = with_images(Path::new("/nope"), msg("just words"));
        assert_eq!(m.content, Some(Value::String("just words".into())));
    }

    #[test]
    fn a_missing_file_leaves_the_words_alone() {
        let text = "look ![image](/media/gone.jpg) at this";
        let m = with_images(Path::new("/nope"), msg(text));
        assert_eq!(m.content, Some(Value::String(text.into())), "the ref must survive");
    }

    #[test]
    fn a_present_file_becomes_an_image_part() {
        let dir = temp_project_with_image("pic.png", b"PNGDATA");

        let m = with_images(&dir, msg("look ![image](/media/pic.png) at this"));
        let parts = m.content.unwrap();
        let arr = parts.as_array().expect("parts array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0], json!({ "type": "text", "text": "look [attached image] at this" }));
        assert_eq!(arr[1]["type"], json!("image_url"));
        let url = arr[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "{url}");
        use base64::Engine as _;
        let b64 = url.strip_prefix("data:image/png;base64,").unwrap();
        assert_eq!(
            base64::engine::general_purpose::STANDARD.decode(b64).unwrap(),
            b"PNGDATA"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_tool_result_with_a_ref_becomes_an_image_part() {
        let dir = temp_project_with_image("shot.png", b"SHOT");

        let m = with_images(&dir, tool_msg("here: ![image](/media/shot.png) done"));
        assert_eq!(m.role, "tool");
        assert_eq!(m.tool_call_id.as_deref(), Some("call-1"));
        let parts = m.content.unwrap();
        let arr = parts.as_array().expect("parts array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0], json!({ "type": "text", "text": "here: [attached image] done" }));
        assert_eq!(arr[1]["type"], json!("image_url"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_assistant_message_is_left_alone_even_with_a_ref() {
        let text = "here: ![image](/media/pic.png)";
        let m = with_images(Path::new("/nope"), Message::assistant(text));
        assert_eq!(m.content, Some(Value::String(text.into())));
    }

    #[test]
    fn a_ref_free_tool_message_passes_untouched() {
        let text = "nothing attached here";
        let m = with_images(Path::new("/nope"), tool_msg(text));
        assert_eq!(m.content, Some(Value::String(text.into())));
        assert_eq!(m.tool_call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn save_media_bytes_round_trips_through_the_seam() {
        let root = test_media_root();
        let name = save_media_bytes(b"hello media", "png").expect("saved");
        assert!(name.starts_with("mcp-"), "{name}");
        assert!(name.ends_with(".png"), "{name}");
        let bytes = std::fs::read(root.join(&name)).unwrap();
        assert_eq!(bytes, b"hello media");
    }

    #[test]
    fn write_media_file_refuses_oversize_and_garbage_extensions() {
        let dir = temp_project_with_image("placeholder.png", b"x");
        let oversized = vec![0u8; MAX_MEDIA_BYTES + 1];
        assert!(write_media_file(&dir, &oversized, "png").is_none());
        // A path-y name must not reach the file system.
        let name = write_media_file(&dir, b"ok", "../evil").unwrap();
        assert!(name.ends_with(".bin"), "{name}");
        assert!(dir.join(&name).exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
