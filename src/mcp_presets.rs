use std::path::{Path, PathBuf};

use crate::mcp::{Framing, ServerSpec};

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

fn expand(value: &str, root: &Path) -> String {
    let mut s = value.replace("${projectRoot}", &root.to_string_lossy());
    while let Some(start) = s.find("${env:") {
        let Some(end_offset) = s[start..].find('}') else { break };
        let end = start + end_offset;
        let val = std::env::var(&s[start + 6..end]).unwrap_or_default();
        s.replace_range(start..=end, &val);
    }
    s
}

pub fn preset(name: &str, file_root: &Path) -> Option<ServerSpec> {
    spec_of(servers().get(name)?, name, file_root)
}

pub fn preset_sandbox(name: &str, file_root: &Path) -> Option<ServerSpec> {
    let entries = servers();
    let v = entries.get(name)?;
    if v.get("disabled").and_then(|v| v.as_bool()) == Some(true) { return None; }
    spec_of(v.get("sandbox")?, name, file_root)
}

fn spec_of(v: &serde_json::Value, name: &str, file_root: &Path) -> Option<ServerSpec> {
    let command = expand(v.get("command")?.as_str()?, file_root);
    let args = v.get("args").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_str()).map(|v| expand(v, file_root)).collect()).unwrap_or_default();
    let env = v.get("env").and_then(|v| v.as_object()).map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), expand(v, file_root)))).collect()).unwrap_or_default();
    let cwd = v.get("cwd").and_then(|v| v.as_str()).map(|s| expand(s, file_root)).or_else(|| Some(file_root.to_string_lossy().into_owned()));
    let framing = match v.get("framing").and_then(|v| v.as_str()) { Some("lsp") | Some("content-length") => Framing::Lsp, _ => Framing::Newline };
    Some(ServerSpec { name: name.to_string(), command, args, cwd, env, framing })
}

pub fn tool_prefix(server: &str) -> String {
    server.replace('-', "_")
}

pub fn prefixed_tool_name(server: &str, tool: &str) -> String {
    format!("{}__{}", tool_prefix(server), tool)
}

pub fn split_tool_name(prefixed: &str) -> Option<(String, String)> {
    let (p, tool) = prefixed.split_once("__")?;
    Some((p.replace('_', "-"), tool.to_string()))
}
