//! Agent definitions — port of `angela/src/agent-loader.mjs`, in a
//! **declarative** format.
//!
//! The JS loads `.coffee` / `.mjs` modules and calls them. Rust cannot
//! execute those, so definitions become JSON at
//! `.angela/agents/<name>.json`.
//!
//! An audit of the real definitions (`gl1`, `blz`, `ada`) showed the
//! CoffeeScript logic is almost entirely **environment-variable fallbacks** —
//! `process.env.LUCY_MODEL or process.env.COWORK_MODEL or ''` — plus a little
//! array de-duplication. So string values here support shell-style
//! interpolation:
//!
//! ```text
//! "${LUCY_MODEL}"                  → the variable, or "" when unset
//! "${LUCY_MODEL:-llama-server:default}"  → with a fallback
//! "${LUCY_MODEL:-${COWORK_MODEL}}" → chained, covering `or` chains
//! ```
//!
//! What is deliberately **not** carried over: the `prepare(def, ctx)` hook and
//! any other arbitrary code. That is the cost of the format decision, and a
//! definition needing it has to move the logic into the harness.
//!
//! JSON rather than YAML because angela already depends on `serde_json`, and
//! agl's vendored YAML subset cannot express lists at all — it skips every
//! line without a colon, so `- file-io` would vanish silently.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};



/// One agent definition, after defaults and interpolation.
#[derive(Debug, Clone)]
pub struct AgentDef {
    pub name: String,
    pub description: String,
    /// Empty means "let agl resolve `default_model` from its config", which
    /// is re-read per inference — so an empty model is a feature, not a gap.
    pub model: String,
    pub models: Vec<String>,
    pub context_window: Option<u64>,
    pub cwd: Option<String>,
    pub env: BTreeMap<String, String>,
    /// MCP preset names, e.g. `["file-io", "term-local"]`.
    pub mcp: Vec<String>,
    /// Auto-approve patterns, one per line.
    pub allowlist: String,
    /// Which tools are registered for the model. `None` = all of them.
    /// Distinct from `allowlist`: a tool can be enabled and still need
    /// approval.
    pub tools_enabled: Option<Vec<String>>,
    pub policy_mode: String,
    pub parallel_tools: bool,
    /// Who the agent **is**. Written once, by whoever defined it.
    /// (Called `system` in the file format; the name changed because it is
    /// now one of three things that make up the system prompt.)
    pub soul: String,
    /// What it is working **toward**. The user's steering, appended to the
    /// system prompt.
    pub goals: String,
    /// What it has **learned**: a blackboard that survives between sessions,
    /// which the agent may rewrite and the user can read and edit.
    pub memory: String,
    /// When it speaks: `invoked` (only when @mentioned), `relevant` (a
    /// System One call decides per message), `engaged` (every message).
    pub engagement: String,
    pub starters: Vec<Value>,
    /// Row id once it lives in the database.
    pub id: Option<i64>,
    pub avatar: Option<String>,
    /// The colour this agent is drawn in — `#rrggbb`, or `None` for the
    /// app's default accent.
    pub accent_color: Option<String>,
    /// Where it was imported from, if it came from a file. Historical.
    pub file: Option<String>,
}

/// The three engagement modes, as a type rather than a loose string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engagement {
    /// Speaks only when `@mentioned`.
    Invoked,
    /// A System One call decides, per message, whether this one has anything
    /// to add.
    Relevant,
    /// Speaks on every message.
    Engaged,
}

impl Engagement {
    pub fn parse(s: &str) -> Engagement {
        match s.trim().to_ascii_lowercase().as_str() {
            "relevant" => Engagement::Relevant,
            "engaged" | "always" => Engagement::Engaged,
            // Anything unrecognised is the quiet option. An agent that
            // speaks when it should not is far more disruptive than one that
            // waits to be asked.
            _ => Engagement::Invoked,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Engagement::Invoked => "invoked",
            Engagement::Relevant => "relevant",
            Engagement::Engaged => "engaged",
        }
    }
}

pub const ENGAGEMENTS: &[&str] = &["invoked", "relevant", "engaged"];

impl Default for AgentDef {
    fn default() -> Self {
        AgentDef {
            name: String::new(),
            description: String::new(),
            model: String::new(),
            models: Vec::new(),
            context_window: None,
            cwd: None,
            env: BTreeMap::new(),
            mcp: Vec::new(),
            allowlist: String::new(),
            tools_enabled: None,
            // The JS default: every tool call asks unless allow-listed.
            policy_mode: "ask".into(),
            parallel_tools: true,
            soul: String::new(),
            goals: String::new(),
            memory: String::new(),
            // Answers every message. This is what every agent did before
            // engagement existed, so it is what an imported definition and a
            // freshly created one both get; `invoked` and `relevant` are
            // opt-ins.
            engagement: "engaged".into(),
            starters: Vec::new(),
            id: None,
            avatar: None,
            accent_color: None,
            file: None,
        }
    }
}

impl AgentDef {
    pub fn engagement(&self) -> Engagement {
        Engagement::parse(&self.engagement)
    }

    /// The system prompt the model actually sees: soul, then goals, then
    /// memory.
    ///
    /// Concatenated in that order and nowhere else, so every caller gets the
    /// same prompt. The headings are there because without them an agent
    /// cannot tell its own standing instructions from a note it left itself
    /// last week — and it is allowed to rewrite only the latter.
    pub fn system_prompt(&self) -> String {
        let mut out = self.soul.trim().to_string();
        let goals = self.goals.trim();
        if !goals.is_empty() {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str("## Goals\n\n");
            out.push_str(goals);
        }
        let memory = self.memory.trim();
        if !memory.is_empty() {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str("## Memory\n\nWhat you have learned so far:\n\n");
            out.push_str(memory);
        }
        out
    }

    /// The full definition, as the editor reads and writes it.
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "models": self.models,
            "contextWindow": self.context_window,
            "cwd": self.cwd,
            "env": self.env,
            "mcp": self.mcp,
            "allowlist": self.allowlist,
            "toolsEnabled": self.tools_enabled,
            "policyMode": self.policy_mode,
            "parallelTools": self.parallel_tools,
            "soul": self.soul,
            "goals": self.goals,
            "memory": self.memory,
            "engagement": self.engagement,
            "starters": self.starters,
            "avatar": self.avatar,
            "accentColor": self.accent_color,
            // Only ever set on a definition that came in from a file.
            "agentFile": self.file,
        })
    }
}

/// The app's accent when an agent has not chosen one.
pub const DEFAULT_ACCENT: &str = "#a78bfa";

/// Accept `#rgb` / `#rrggbb` (with or without the `#`), reject anything else.
///
/// Rejected rather than coerced: this value is interpolated straight into a
/// `style` attribute, so "whatever the user typed" is not a thing to pass
/// through, and a silently-wrong colour is worse than the default.
pub fn normalize_hex(raw: &str) -> Option<String> {
    let t = raw.trim().trim_start_matches('#');
    if t.is_empty() {
        return None;
    }
    if !t.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    match t.len() {
        3 => {
            // #abc -> #aabbcc, so everything downstream sees one form.
            let mut out = String::from("#");
            for c in t.chars() {
                out.push(c.to_ascii_lowercase());
                out.push(c.to_ascii_lowercase());
            }
            Some(out)
        }
        6 => Some(format!("#{}", t.to_ascii_lowercase())),
        _ => None,
    }
}

/// Expand `${VAR}` and `${VAR:-fallback}`, recursively, in one string.
///
/// Unknown variables become empty, matching `process.env.X` being undefined
/// in the `or` chains these definitions came from.
pub fn expand_env(input: &str) -> String {
    expand_depth(input, 0)
}

fn expand_depth(input: &str, depth: usize) -> String {
    if depth > 8 || !input.contains("${") {
        // A cycle would otherwise recurse forever; 8 is far past real use.
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len());
    let b: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < b.len() {
        if b[i] == '$' && i + 1 < b.len() && b[i + 1] == '{' {
            // Find the matching brace, allowing nesting in the fallback.
            let mut depth_br = 1;
            let mut j = i + 2;
            while j < b.len() && depth_br > 0 {
                if b[j] == '{' {
                    depth_br += 1;
                } else if b[j] == '}' {
                    depth_br -= 1;
                    if depth_br == 0 {
                        break;
                    }
                }
                j += 1;
            }
            if depth_br != 0 {
                // Unbalanced: leave the text exactly as written.
                out.push(b[i]);
                i += 1;
                continue;
            }
            let inner: String = b[i + 2..j].iter().collect();
            let (name, fallback) = match inner.split_once(":-") {
                Some((n, f)) => (n.trim().to_string(), Some(f.to_string())),
                None => (inner.trim().to_string(), None),
            };
            let value = std::env::var(&name).ok().filter(|v| !v.is_empty());
            match value {
                Some(v) => out.push_str(&v),
                None => {
                    if let Some(f) = fallback {
                        out.push_str(&expand_depth(&f, depth + 1));
                    }
                }
            }
            i = j + 1;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    out
}

fn s(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(x)) => expand_env(x),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        _ => String::new(),
    }
}

/// A list of strings from either an array or a newline-separated string.
///
/// `toolsEnabled` and `allowlist` are written both ways in the real
/// definitions, so both are accepted.
fn string_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(|t| expand_env(t)))
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect(),
        Some(Value::String(text)) => expand_env(text)
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect(),
        _ => Vec::new(),
    }
}

/// Parse one definition object over the defaults.
pub fn from_json(name_hint: &str, v: &Value, file: Option<String>) -> AgentDef {
    let o = v.as_object().cloned().unwrap_or_default();
    let get = |k: &str| o.get(k);
    let mut def = AgentDef {
        name: {
            let n = s(get("name"));
            if n.is_empty() {
                name_hint.to_string()
            } else {
                n
            }
        },
        description: s(get("description")),
        model: s(get("model")),
        models: string_list(get("models")),
        context_window: get("contextWindow")
            .or_else(|| o.get("context_window"))
            .and_then(|x| x.as_u64()),
        cwd: {
            let c = s(get("cwd"));
            if c.is_empty() {
                None
            } else {
                Some(c)
            }
        },
        env: get("env")
            .and_then(|e| e.as_object())
            .map(|m| {
                m.iter()
                    .map(|(k, val)| (k.clone(), s(Some(val))))
                    .collect()
            })
            .unwrap_or_default(),
        mcp: string_list(get("mcp")),
        allowlist: s(get("allowlist")),
        tools_enabled: match get("toolsEnabled").or_else(|| o.get("tools_enabled")) {
            // Explicit null, or absent, means "all tools".
            None | Some(Value::Null) => None,
            other => {
                let list = string_list(other);
                if list.is_empty() {
                    None
                } else {
                    Some(list)
                }
            }
        },
        policy_mode: {
            let p = s(get("policyMode").or_else(|| o.get("policy_mode")));
            if p.is_empty() {
                "ask".into()
            } else {
                p
            }
        },
        parallel_tools: get("parallel_tools")
            .or_else(|| o.get("parallelTools"))
            .and_then(|x| x.as_bool())
            .unwrap_or(true),
        // `system` is the old file-format name for the same field; both are
        // accepted so an imported definition keeps its prompt.
        soul: {
            let v = s(get("soul"));
            if v.is_empty() { s(get("system")) } else { v }
        },
        goals: s(get("goals")),
        memory: s(get("memory")),
        // Only an *explicit* value is parsed. A definition with no
        // `engagement` key predates the field, and its behaviour was to
        // answer everything — so it keeps the default rather than being
        // silently muted by the import.
        engagement: match get("engagement") {
            Some(v) if !s(Some(v)).trim().is_empty() => {
                Engagement::parse(&s(Some(v))).as_str().to_string()
            }
            _ => AgentDef::default().engagement,
        },
        starters: get("starters")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default(),
        id: get("id").and_then(|x| x.as_i64()),
        avatar: {
            let a = s(get("avatar"));
            if a.is_empty() { None } else { Some(a) }
        },
        accent_color: normalize_hex(&s(get("accentColor").or_else(|| o.get("accent_color")))),
        file,
    };
    // De-duplicate `models` while keeping order — the JS does this with
    // `.filter((m, i, a) => m && a.indexOf(m) === i)`.
    let mut seen = Vec::new();
    def.models.retain(|m| {
        if m.is_empty() || seen.contains(m) {
            false
        } else {
            seen.push(m.clone());
            true
        }
    });
    def
}

/// Directories that may hold agent definitions.
///
/// Deliberately **excludes the project root**, unlike the JS search path.
/// The JS looks for `.coffee`/`.mjs`, which no ordinary project has lying
/// around; this port uses `.json`, which every project has — scanning the
/// root made `rustex1.json` show up as an agent named "chat". A definition
/// must live in a dedicated agents directory.
fn agent_dirs(project_root: &Path) -> Vec<PathBuf> {
    vec![
        crate::project::agents_dir(project_root), // .angela/agents
        project_root.join("agents"),              // legacy/local convenience
    ]
}

/// Load `<name>` from the first search directory that has it — **import
/// only**; see [`list_files`].
pub fn load_file(project_root: &Path, name: &str) -> Option<AgentDef> {
    if name.is_empty() || name.contains('/') || name.contains("..") {
        return None;
    }
    for dir in agent_dirs(project_root) {
        let p = dir.join(format!("{name}.json"));
        if let Ok(text) = std::fs::read_to_string(&p) {
            if let Ok(v) = serde_json::from_str::<Value>(&text) {
                return Some(from_json(name, &v, Some(p.to_string_lossy().into_owned())));
            }
        }
    }
    None
}

/// Every definition on disk, sorted by name — **import only**.
///
/// Agents live in the `agents` table now. This reads the pre-database files
/// so [`crate::agent_store::AgentStore::import_files`] can bring them in
/// once; nothing in the running server reads a definition from disk.
pub fn list_files(project_root: &Path) -> Vec<AgentDef> {
    let mut out: Vec<AgentDef> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for dir in agent_dirs(project_root) {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let fname = e.file_name().to_string_lossy().to_string();
            let Some(stem) = fname.strip_suffix(".json") else {
                continue;
            };
            // First directory wins, matching the JS search order.
            if seen.contains(&stem.to_string()) {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(e.path()) {
                if let Ok(v) = serde_json::from_str::<Value>(&text) {
                    seen.push(stem.to_string());
                    out.push(from_json(
                        stem,
                        &v,
                        Some(e.path().to_string_lossy().into_owned()),
                    ));
                }
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The definition used when a project defines no agents.
///
/// Local file I/O and shell, asking before each call — safe to hand someone
/// who has not configured anything.
pub fn default_agent() -> AgentDef {
    AgentDef {
        name: "default".into(),
        description: "Local coding assistant (file-io + term-local)".into(),
        mcp: vec!["file-io".into(), "term-local".into()],
        soul: "You are a coding assistant with access to the local project.\n\
                 Use the tools to list, read, search and edit files, and to run \
                 shell commands, rather than guessing. Be concise and concrete, \
                 and use Markdown."
            .into(),
        policy_mode: "ask".into(),
        ..Default::default()
    }
}

/// The system prompt the built-in default produces, for callers that have
/// no definition at all.
pub fn default_system_prompt() -> String {
    default_agent().system_prompt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_interpolation_covers_the_coffee_or_chains() {
        std::env::set_var("ANGELA_TEST_A", "alpha");
        std::env::remove_var("ANGELA_TEST_B");
        assert_eq!(expand_env("${ANGELA_TEST_A}"), "alpha");
        assert_eq!(expand_env("${ANGELA_TEST_B}"), "");
        assert_eq!(expand_env("${ANGELA_TEST_B:-fallback}"), "fallback");
        // `process.env.B or process.env.A or 'x'`
        assert_eq!(
            expand_env("${ANGELA_TEST_B:-${ANGELA_TEST_A:-x}}"),
            "alpha"
        );
        assert_eq!(expand_env("pre-${ANGELA_TEST_A}-post"), "pre-alpha-post");
        std::env::remove_var("ANGELA_TEST_A");
    }

    #[test]
    fn an_empty_variable_is_treated_as_unset() {
        // `process.env.X or Y` skips an empty string too.
        std::env::set_var("ANGELA_TEST_EMPTY", "");
        assert_eq!(expand_env("${ANGELA_TEST_EMPTY:-used}"), "used");
        std::env::remove_var("ANGELA_TEST_EMPTY");
    }

    #[test]
    fn unbalanced_braces_are_left_alone() {
        assert_eq!(expand_env("${UNCLOSED"), "${UNCLOSED");
        assert_eq!(expand_env("plain text"), "plain text");
    }

    #[test]
    fn lists_accept_arrays_and_multiline_strings() {
        let as_array = json!({ "mcp": ["file-io", "term-local"] });
        assert_eq!(
            from_json("x", &as_array, None).mcp,
            vec!["file-io", "term-local"]
        );
        let as_text = json!({ "mcp": "file-io\nterm-local\n" });
        assert_eq!(
            from_json("x", &as_text, None).mcp,
            vec!["file-io", "term-local"]
        );
    }

    #[test]
    fn comments_and_blanks_are_stripped_from_list_strings() {
        // The real allowlists are commented multiline strings.
        let v = json!({ "toolsEnabled": "# files\nfile_io__read_file\n\n# shell\nterm_local__bash\n" });
        let d = from_json("x", &v, None);
        assert_eq!(
            d.tools_enabled.unwrap(),
            vec!["file_io__read_file", "term_local__bash"]
        );
    }

    #[test]
    fn tools_enabled_absent_or_null_means_all() {
        assert!(from_json("x", &json!({}), None).tools_enabled.is_none());
        assert!(from_json("x", &json!({ "toolsEnabled": null }), None)
            .tools_enabled
            .is_none());
    }

    #[test]
    fn defaults_match_the_js() {
        let d = from_json("x", &json!({}), None);
        assert_eq!(d.name, "x", "the filename names the agent");
        assert_eq!(d.policy_mode, "ask");
        assert!(d.parallel_tools);
        assert_eq!(d.model, "", "empty model → agl resolves default_model");
        assert!(d.tools_enabled.is_none());
    }

    #[test]
    fn models_deduplicate_and_keep_order() {
        let v = json!({ "models": ["a", "b", "a", "", "c"] });
        assert_eq!(from_json("x", &v, None).models, vec!["a", "b", "c"]);
    }

    #[test]
    fn snake_and_camel_keys_both_work() {
        let v = json!({ "policy_mode": "auto", "context_window": 32768 });
        let d = from_json("x", &v, None);
        assert_eq!(d.policy_mode, "auto");
        assert_eq!(d.context_window, Some(32768));
    }

    #[test]
    fn load_and_list_read_from_the_agents_dir() {
        let tmp = tempfile::tempdir().unwrap();
        crate::project::ensure_angela_layout(tmp.path()).unwrap();
        let dir = crate::project::agents_dir(tmp.path());
        std::fs::write(
            dir.join("coder.json"),
            r#"{"description":"c","mcp":["file-io"],"system":"hi"}"#,
        )
        .unwrap();
        std::fs::write(dir.join("notjson.txt"), "ignored").unwrap();

        let d = load_file(tmp.path(), "coder").expect("loaded");
        assert_eq!(d.name, "coder");
        assert_eq!(d.mcp, vec!["file-io"]);
        assert!(d.file.unwrap().ends_with("coder.json"));

        let all = list_files(tmp.path());
        assert_eq!(all.len(), 1, "only .json files are definitions");
        assert!(load_file(tmp.path(), "missing").is_none());
    }

    #[test]
    fn project_root_json_is_not_an_agent() {
        // Regression: `.json` collides with ordinary project files, so the
        // project root must NOT be scanned. `rustex1.json` once listed as an
        // agent called "chat".
        let tmp = tempfile::tempdir().unwrap();
        crate::project::ensure_angela_layout(tmp.path()).unwrap();
        std::fs::write(tmp.path().join("rustex1.json"), r#"{"name":"chat"}"#).unwrap();
        std::fs::write(tmp.path().join("package.json"), r#"{"name":"pkg"}"#).unwrap();
        assert!(list_files(tmp.path()).is_empty(), "project json is not an agent");
        assert!(load_file(tmp.path(), "rustex1").is_none());
        assert!(load_file(tmp.path(), "package").is_none());
    }

    #[test]
    fn load_refuses_traversal_names() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(load_file(tmp.path(), "../escape").is_none());
        assert!(load_file(tmp.path(), "a/b").is_none());
    }

}
