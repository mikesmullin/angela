//! MCP server launch definitions from the chat user's configuration.

use std::path::{Path, PathBuf};

use crate::mcp::{Framing, ServerSpec};

/// The chat MCP configuration is re-read whenever servers are selected.
pub fn config_path() -> PathBuf {
    std::env::var_os("CHAT_MCP_CONFIG").map(PathBuf::from).unwrap_or_else(|| {
        std::env::var_os("ANGELA_ROOT").map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
            .join(".angela/mcp.json")
    })
}

fn servers() -> serde_json::Map<String, serde_json::Value> {
    let path = config_path();
    match std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()) {
        Some(v) => v.get("mcpServers").and_then(|v| v.as_object()).cloned().unwrap_or_default(),
        None => { eprintln!("[mcp] could not read MCP configuration: {}", path.display()); Default::default() }
    }
}

pub fn names() -> Vec<String> {
    servers().into_iter().filter(|(_, v)| v.get("disabled").and_then(|v| v.as_bool()) != Some(true))
        .map(|(name, _)| name).collect()
}

fn expand(value: &str, project_root: &Path) -> String {
    let mut s = value.replace("${projectRoot}", &project_root.to_string_lossy());
    while let Some(start) = s.find("${env:") {
        let Some(end_offset) = s[start..].find('}') else { break };
        let end = start + end_offset;
        let val = std::env::var(&s[start + 6..end]).unwrap_or_default();
        s.replace_range(start..=end, &val);
    }
    s
}

/// Translate a configured stdio server into the MCP client's launch spec.
pub fn preset(name: &str, project_root: &Path) -> Option<ServerSpec> {
    let entries = servers();
    let v = entries.get(name)?;
    if v.get("disabled").and_then(|v| v.as_bool()) == Some(true) { return None; }
    let command = expand(v.get("command")?.as_str()?, project_root);
    let args = v.get("args").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_str()).map(|v| expand(v, project_root)).collect()).unwrap_or_default();
    let env = v.get("env").and_then(|v| v.as_object()).map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), expand(v, project_root)))).collect()).unwrap_or_default();
    let cwd = v.get("cwd").and_then(|v| v.as_str()).map(|s| expand(s, project_root)).or_else(|| Some(project_root.to_string_lossy().into_owned()));
    let framing = match v.get("framing").and_then(|v| v.as_str()) { Some("lsp") | Some("content-length") => Framing::Lsp, _ => Framing::Newline };
    Some(ServerSpec { name: name.to_string(), command, args, cwd, env, framing })
}

/// `file-io` → `file_io`, the prefix the JS gives a server's tools
/// (`file_io__read_file`).
pub fn tool_prefix(server: &str) -> String {
    server.replace('-', "_")
}

/// The prefixed name a model sees for one server's tool.
pub fn prefixed_tool_name(server: &str, tool: &str) -> String {
    format!("{}__{}", tool_prefix(server), tool)
}

/// Split a prefixed name back into `(server, tool)`.
pub fn split_tool_name(prefixed: &str) -> Option<(String, String)> {
    let (p, tool) = prefixed.split_once("__")?;
    Some((p.replace('_', "-"), tool.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_round_trip() {
        let n = prefixed_tool_name("file-io", "read_file");
        assert_eq!(n, "file_io__read_file");
        let (s, t) = split_tool_name(&n).unwrap();
        assert_eq!(s, "file-io");
        assert_eq!(t, "read_file");
    }

    #[test]
    fn a_tool_name_with_underscores_survives_the_split() {
        // Only the FIRST `__` separates; the tool keeps its own underscores.
        let (s, t) = split_tool_name("term_local__bash").unwrap();
        assert_eq!((s.as_str(), t.as_str()), ("term-local", "bash"));
        let (s, t) = split_tool_name("file_io__write_file").unwrap();
        assert_eq!((s.as_str(), t.as_str()), ("file-io", "write_file"));
    }

    #[test]
    fn an_unprefixed_name_has_no_split() {
        assert!(split_tool_name("bash").is_none());
    }

    #[test]
    fn an_unknown_preset_is_none() {
        assert!(preset("not-a-real-server", Path::new("/tmp")).is_none());
    }
}
