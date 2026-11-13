//! Port of `angela/src/mcp-presets.mjs`, narrowed to the two local presets.
//!
//! The MCP **servers themselves are not ported** — they run as-is from
//! `/workspace/mcp/*` under bun. This only works out where they live and with
//! what environment to launch them.
//!
//! Search order, matching the JS:
//!   1. `$ANGELA_MCP_ROOT/{file-io,term-local}/server.mjs`
//!   2. sibling monorepo `<angela>/../mcp/…`
//!   3. standalone packages `<angela>/../mcp-file-io/server.mjs`
//!   4. `/workspace/mcp` (devcontainer convenience)

use std::path::{Path, PathBuf};

use crate::mcp::{Framing, ServerSpec};

/// Directories that may hold `file-io/` and `term-local/` subdirectories.
pub fn mcp_root_candidates() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut add = |p: PathBuf| {
        if !out.contains(&p) {
            out.push(p);
        }
    };
    if let Ok(v) = std::env::var("ANGELA_MCP_ROOT") {
        if !v.trim().is_empty() {
            add(PathBuf::from(v.trim()));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(parent) = cwd.parent() {
            add(parent.join("mcp"));
        }
        add(cwd.join("mcp"));
    }
    add(PathBuf::from("/workspace/mcp"));
    out
}

/// Absolute path to one preset's `server.mjs`, if it exists.
pub fn resolve_server_script(preset: &str) -> Option<PathBuf> {
    for root in mcp_root_candidates() {
        let direct = root.join(preset).join("server.mjs");
        if direct.is_file() {
            return Some(direct);
        }
        // standalone package layout: ../mcp-file-io/server.mjs
        if let Some(parent) = root.parent() {
            let standalone = parent.join(format!("mcp-{preset}")).join("server.mjs");
            if standalone.is_file() {
                return Some(standalone);
            }
        }
    }
    None
}

/// The runtime that executes an `.mjs` MCP server.
fn bun() -> String {
    std::env::var("ANGELA_BUN").unwrap_or_else(|_| "bun".into())
}

/// Build the spec for a preset name, jailed to `project_root`.
///
/// Both servers are sandboxed to the project: `MCP_FILE_ROOT` for file-io and
/// `MCP_TERM_JB_ROOT` for term-local. That jail is the whole safety story for
/// these tools, so it is always set, never left to the server's default.
pub fn preset(name: &str, project_root: &Path) -> Option<ServerSpec> {
    let jail = std::env::var("MCP_FILE_ROOT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| project_root.to_string_lossy().into_owned());
    let script = resolve_server_script(name)?;

    let env = match name {
        "file-io" => vec![("MCP_FILE_ROOT".to_string(), jail)],
        "term-local" => vec![
            ("MCP_TERM_CWD".to_string(), jail.clone()),
            ("MCP_TERM_JB_ROOT".to_string(), jail),
            (
                // Without a PATH the sandbox cannot find even the commands
                // the allow-list permits: `uname` came back "command not
                // found" until this was passed through.
                "MCP_TERM_PATH".to_string(),
                std::env::var("MCP_TERM_PATH")
                    .ok()
                    .filter(|p| !p.trim().is_empty())
                    .or_else(|| std::env::var("PATH").ok())
                    .unwrap_or_else(|| "/usr/local/bin:/usr/bin:/bin".into()),
            ),
            (
                "MCP_TERM_ALLOW".to_string(),
                std::env::var("MCP_TERM_ALLOW").unwrap_or_else(|_| {
                    // The JS default allow-list. Deliberately conservative:
                    // anything beyond this needs an explicit opt-in.
                    "echo,pwd,ls,true,uname,date,cat,head,wc,which,bun,node".into()
                }),
            ),
        ],
        _ => Vec::new(),
    };

    Some(ServerSpec {
        name: name.to_string(),
        command: bun(),
        args: vec![script.to_string_lossy().into_owned()],
        cwd: Some(project_root.to_string_lossy().into_owned()),
        env,
        framing: Framing::Newline,
    })
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
    fn the_two_local_presets_resolve_when_installed() {
        // Skips rather than fails where /workspace/mcp is absent.
        if resolve_server_script("file-io").is_none() {
            eprintln!("no local mcp checkout; skipping");
            return;
        }
        let root = Path::new("/tmp");
        let f = preset("file-io", root).unwrap();
        assert!(f.args[0].ends_with("file-io/server.mjs"), "{:?}", f.args);
        assert_eq!(f.env[0].0, "MCP_FILE_ROOT");
        assert_eq!(f.env[0].1, "/tmp", "jailed to the project root");

        let t = preset("term-local", root).unwrap();
        let keys: Vec<&str> = t.env.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"MCP_TERM_JB_ROOT"));
        assert!(keys.contains(&"MCP_TERM_ALLOW"));
        // Without a PATH even an allow-listed command is "not found".
        assert!(keys.contains(&"MCP_TERM_PATH"));
        let path = t.env.iter().find(|(k, _)| k == "MCP_TERM_PATH").unwrap();
        assert!(!path.1.is_empty());
    }

    #[test]
    fn an_unknown_preset_is_none() {
        assert!(preset("not-a-real-server", Path::new("/tmp")).is_none());
    }
}
