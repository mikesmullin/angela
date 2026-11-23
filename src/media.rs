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
//! the `POST /api/upload` destination); the ref is just its file name.

use std::path::Path;

use agl::provider::Message;
use serde_json::{json, Value};

/// The provider's view of one **user** message: its text with every attached
/// image as an image part. Anything else — an assistant reply, a tool result,
/// a message with no `/media/…` ref — passes through untouched, so the wire
/// for ordinary turns is byte-identical to what it always was.
///
/// A ref whose file is gone is left as words: the transcript's text is the
/// truth, and a missing image must not kill the turn.
pub fn with_images(project_root: &Path, msg: Message) -> Message {
    if msg.role != "user" {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(text: &str) -> Message {
        Message::user(text)
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
        let dir = std::env::temp_dir().join(format!(
            "media-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let media = dir.join(".angela").join("media");
        std::fs::create_dir_all(&media).unwrap();
        std::fs::write(media.join("pic.png"), b"PNGDATA").unwrap();

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
}
