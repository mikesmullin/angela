use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};



#[derive(Debug, Clone)]
pub struct AgentDef {
    pub name: String,
    pub description: String,
    pub model: String,
    pub reasoning_effort: String,
    pub models: Vec<String>,
    pub context_window: Option<u64>,
    pub env: BTreeMap<String, String>,
    pub mcp: Vec<String>,
    pub allowlist: String,
    pub tools_enabled: Option<Vec<String>>,
    pub policy_mode: String,
    pub parallel_tools: bool,
    pub soul: String,
    pub goals: String,
    pub memory: String,
    pub engagement: String,
    pub starters: Vec<Value>,
    pub id: Option<i64>,
    pub avatar: Option<String>,
    pub accent_color: Option<String>,
    pub icon: Option<String>,
    pub max_turns: Option<u32>,
    pub file: Option<String>,
    pub containerfile: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engagement {
    Invoked,
    Relevant,
    Engaged,
}

impl Engagement {
    pub fn parse(s: &str) -> Engagement {
        match s.trim().to_ascii_lowercase().as_str() {
            "relevant" => Engagement::Relevant,
            "engaged" | "always" => Engagement::Engaged,
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
            reasoning_effort: String::new(),
            models: Vec::new(),
            context_window: None,
            env: BTreeMap::new(),
            mcp: Vec::new(),
            allowlist: String::new(),
            tools_enabled: None,
            policy_mode: "ask".into(),
            parallel_tools: true,
            soul: String::new(),
            goals: String::new(),
            memory: String::new(),
            engagement: "engaged".into(),
            starters: Vec::new(),
            id: None,
            avatar: None,
            accent_color: None,
            icon: None,
            max_turns: None,
            containerfile: String::new(),
            file: None,
        }
    }
}

impl AgentDef {
    pub fn engagement(&self) -> Engagement {
        Engagement::parse(&self.engagement)
    }

    pub fn system_prompt(&self) -> String {
        self.system_prompt_with_appendage("")
    }

    pub fn system_prompt_with_appendage(&self, appendage: &str) -> String {
        let mut out = self.soul.trim().to_string();
        let extra = appendage.trim();
        if !extra.is_empty() {
            if !out.is_empty() { out.push_str("\n\n"); }
            out.push_str(extra);
        }
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

    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "reasoningEffort": self.reasoning_effort,
            "models": self.models,
            "contextWindow": self.context_window,
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
            "icon": self.icon,
            "maxTurns": self.max_turns,
            "containerFile": if self.containerfile.is_empty() {
                None
            } else {
                Some(self.containerfile.clone())
            },
            "agentFile": self.file,
        })
    }
}

pub const DEFAULT_ACCENT: &str = "#a78bfa";

pub const DEFAULT_ICON: &str = "robot";

pub fn normalize_icon(raw: &str) -> Option<String> {
    let lower = raw.trim().to_ascii_lowercase();
    let mut t = lower.trim();
    t = t.strip_prefix("ph ").unwrap_or(t).trim();
    t = t.strip_prefix("ph-").unwrap_or(t);
    t = t.trim_matches('-');
    if t.is_empty() || t.len() > 64 {
        return None;
    }
    if !t
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return None;
    }
    Some(t.to_string())
}

pub const MAX_TURNS_CEILING: u32 = 1000;

pub fn normalize_max_turns(raw: Option<&Value>) -> Option<u32> {
    let n = match raw? {
        Value::Number(n) => n.as_f64()?,
        Value::String(t) => t.trim().parse::<f64>().ok()?,
        _ => return None,
    };
    if !n.is_finite() || n < 0.0 {
        return None;
    }
    if n == 0.0 {
        return Some(0);
    }
    Some((n as u32).min(MAX_TURNS_CEILING))
}

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

pub fn expand_env(input: &str) -> String {
    expand_depth(input, 0)
}

fn expand_depth(input: &str, depth: usize) -> String {
    if depth > 8 || !input.contains("${") {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len());
    let b: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < b.len() {
        if b[i] == '$' && i + 1 < b.len() && b[i + 1] == '{' {
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
        reasoning_effort: s(get("reasoningEffort").or_else(|| get("reasoning_effort"))),
        models: string_list(get("models")),
        context_window: get("contextWindow")
            .or_else(|| o.get("context_window"))
            .and_then(|x| x.as_u64()),
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
        soul: {
            let v = s(get("soul"));
            if v.is_empty() { s(get("system")) } else { v }
        },
        goals: s(get("goals")),
        memory: s(get("memory")),
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
        icon: normalize_icon(&s(get("icon"))),
        max_turns: normalize_max_turns(get("maxTurns").or_else(|| o.get("max_turns"))),
        containerfile: s(get("containerFile").or_else(|| o.get("containerfile"))),
        file,
    };
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

fn agent_dirs(project_root: &Path) -> Vec<PathBuf> {
    vec![
        crate::project::agents_dir(project_root),
        project_root.join("agents"),
    ]
}

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

pub fn default_system_prompt() -> String {
    default_agent().system_prompt()
}
