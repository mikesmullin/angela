use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::project::{ensure_angela_layout, sessions_dir};

pub fn now_iso() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let millis = now.subsec_millis();
    let (y, mo, d, h, mi, s) = civil_from_unix(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{millis:03}Z")
}

fn civil_from_unix(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (h, mi, s) = (
        (rem / 3600) as u32,
        ((rem % 3600) / 60) as u32,
        (rem % 60) as u32,
    );
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d, h, mi, s)
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

pub fn new_session_id() -> String {
    let iso = now_iso();
    let stem: String = iso
        .chars()
        .take(19)
        .map(|c| if c == ':' || c == '.' { '-' } else { c })
        .collect();
    format!("{stem}_{}", rand_hex(3))
}

pub fn new_event_id_pub() -> String {
    new_event_id()
}

fn new_event_id() -> String {
    format!("e{}", rand_hex(4))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventWrapper {
    pub event_id: String,
    pub ts: String,
    pub event_type: String,
    pub duration_ms: Option<f64>,
    pub payload: Value,
    #[serde(default)]
    pub meta: Map<String, Value>,
}

impl EventWrapper {
    pub fn visible(&self) -> bool {
        self.meta
            .get("visible")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }
}

#[derive(Debug, Clone, Default)]
pub struct NewEvent {
    pub event_type: String,
    pub payload: Value,
    pub duration_ms: Option<f64>,
    pub ts: Option<String>,
    pub meta: Option<Map<String, Value>>,
    pub event_id: Option<String>,
    pub speaker: Option<i64>,
}

impl NewEvent {
    pub fn new(event_type: impl Into<String>, payload: Value) -> Self {
        NewEvent {
            event_type: event_type.into(),
            payload,
            ..Default::default()
        }
    }
}

pub fn fold_event_metas(events: Vec<EventWrapper>) -> Vec<EventWrapper> {
    if events.is_empty() {
        return events;
    }
    let mut extra: std::collections::HashMap<String, Map<String, Value>> =
        std::collections::HashMap::new();

    for wrap in &events {
        if wrap.event_type != "context_meta" {
            continue;
        }
        let p = match wrap.payload.as_object() {
            Some(o) => o,
            None => continue,
        };
        let ids: Vec<String> = match p.get("event_ids").and_then(|v| v.as_array()) {
            Some(a) => a
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            None => p
                .get("event_id")
                .and_then(|v| v.as_str())
                .map(|s| vec![s.to_string()])
                .unwrap_or_default(),
        };
        let mut rest = p.clone();
        rest.remove("event_id");
        rest.remove("event_ids");
        if rest.is_empty() {
            continue;
        }
        for eid in ids {
            if eid.is_empty() {
                continue;
            }
            let slot = extra.entry(eid).or_default();
            for (k, v) in &rest {
                slot.insert(k.clone(), v.clone());
            }
        }
    }
    if extra.is_empty() {
        return events;
    }
    events
        .into_iter()
        .map(|mut wrap| {
            if let Some(patch) = extra.get(&wrap.event_id) {
                for (k, v) in patch {
                    wrap.meta.insert(k.clone(), v.clone());
                }
                if let Some(v) = patch.get("visible") {
                    let truthy = match v {
                        Value::Bool(b) => *b,
                        Value::Null => false,
                        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
                        Value::String(s) => !s.is_empty(),
                        _ => true,
                    };
                    wrap.meta.insert("visible".into(), Value::Bool(truthy));
                }
            }
            wrap
        })
        .collect()
}

#[derive(Debug, Clone, Default)]
pub struct NewSession {
    pub agent: Option<String>,
    pub agent_file: Option<String>,
    pub model: Option<String>,
    pub title: Option<String>,
    pub mcp: Vec<String>,
    pub policy_mode: Option<String>,
    pub approve_mode: Option<Value>,
    pub allowlist_source: Option<String>,
    pub allowlist: Option<String>,
    pub allowlist_mode: Option<String>,
    pub context_window: Option<u64>,
    pub config: Option<Value>,
    pub overrides: Option<Value>,
}

pub struct SessionStore {
    pub project_root: PathBuf,
    pub dir: PathBuf,
}

impl SessionStore {
    pub fn new(project_root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = project_root.as_ref().to_path_buf();
        ensure_angela_layout(&root)?;
        Ok(SessionStore {
            dir: sessions_dir(&root),
            project_root: root,
        })
    }

    pub fn state_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    pub fn log_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.jsonl"))
    }

    pub fn create(&self, meta: NewSession) -> std::io::Result<Value> {
        let id = new_session_id();
        let now = now_iso();
        let ordinal = self
            .list_meta()
            .iter()
            .filter_map(|s| s.get("ordinal").and_then(|v| v.as_u64()))
            .max()
            .unwrap_or(0)
            + 1;
        let state = json!({
            "id": id,
            "ordinal": ordinal,
            "apiVersion": "angela/v1",
            "logEncoding": "delta/v1",
            "kind": "AgentSession",
            "createdAt": now,
            "updatedAt": now,
            "status": "ready",
            "agent": meta.agent,
            "agentFile": meta.agent_file,
            "model": meta.model,
            "title": meta.title,
            "mcp": meta.mcp,
            "policyMode": meta.policy_mode.unwrap_or_else(|| "ask".into()),
            "approveMode": meta.approve_mode.unwrap_or(Value::Null),
            "allowlistSource": meta.allowlist_source,
            "allowlist": meta.allowlist,
            "contextWindow": meta.context_window,
            "projectRoot": self.project_root.to_string_lossy(),
            "config": meta.config.unwrap_or(Value::Null),
            "overrides": meta.overrides.unwrap_or(Value::Null),
            "turns": 0,
            "eventCount": 0,
            "lastPromptTokens": Value::Null,
            "lastTotalTokens": Value::Null,
            "lastCompletionTokens": Value::Null,
            "logFile": format!("{id}.jsonl"),
        });
        self.write_state(&id, state.clone())?;
        self.append_event(
            &id,
            NewEvent::new(
                "session",
                json!({
                    "kind": "created",
                    "agent": state["agent"],
                    "model": state["model"],
                    "mcp": state["mcp"],
                    "overrides": state["overrides"],
                    "config": state["config"],
                }),
            ),
        )?;
        Ok(self.load(&id).unwrap_or(state))
    }

    pub fn load(&self, id: &str) -> Option<Value> {
        let p = self.state_path(id);
        let text = std::fs::read_to_string(p).ok()?;
        serde_json::from_str(&text).ok()
    }

    fn write_state(&self, id: &str, mut state: Value) -> std::io::Result<()> {
        if !self.dir.exists() {
            std::fs::create_dir_all(&self.dir)?;
        }
        if let Some(o) = state.as_object_mut() {
            o.insert("updatedAt".into(), json!(now_iso()));
        }
        let body = serde_json::to_string_pretty(&state).unwrap_or_default();
        std::fs::write(self.state_path(id), format!("{body}\n"))
    }

    pub fn save(&self, state: &Value) -> std::io::Result<()> {
        let id = state.get("id").and_then(|v| v.as_str()).unwrap_or_default();
        self.write_state(id, state.clone())
    }

    pub fn update_meta(&self, id: &str, patch: Map<String, Value>) -> Option<Value> {
        let mut state = self.load(id)?;
        {
            let o = state.as_object_mut()?;
            for (k, v) in patch {
                o.insert(k, v);
            }
        }
        self.write_state(id, state.clone()).ok()?;
        self.load(id)
    }

    pub fn append_event(&self, id: &str, ev: NewEvent) -> std::io::Result<EventWrapper> {
        let wrapper = EventWrapper {
            event_id: ev.event_id.unwrap_or_else(new_event_id),
            ts: ev.ts.unwrap_or_else(now_iso),
            event_type: ev.event_type,
            duration_ms: ev.duration_ms.filter(|d| d.is_finite()),
            payload: ev.payload,
            meta: ev.meta.unwrap_or_default(),
        };
        let line = serde_json::to_string(&wrapper).unwrap_or_default();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.log_path(id))?;
        writeln!(f, "{line}")?;

        if let Some(mut state) = self.load(id) {
            let n = state
                .get("eventCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            if let Some(o) = state.as_object_mut() {
                o.insert("eventCount".into(), json!(n + 1));
            }
            let _ = self.write_state(id, state);
        }
        Ok(wrapper)
    }

    pub fn list_events(&self, id: &str) -> Vec<EventWrapper> {
        let text = match std::fs::read_to_string(self.log_path(id)) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<EventWrapper>(l).ok())
            .collect()
    }

    pub fn list_events_folded(&self, id: &str) -> Vec<EventWrapper> {
        fold_event_metas(self.list_events(id))
    }

    pub fn record_turn_complete(&self, id: &str, duration_ms: Option<f64>) {
        let Some(mut state) = self.load(id) else {
            return;
        };
        let turns = state.get("turns").and_then(|v| v.as_u64()).unwrap_or(0) + 1;
        if let Some(o) = state.as_object_mut() {
            o.insert("turns".into(), json!(turns));
            o.insert("status".into(), json!("ready"));
        }
        let _ = self.write_state(id, state);
        let _ = self.append_event(
            id,
            NewEvent {
                event_type: "session".into(),
                payload: json!({ "kind": "turn_complete", "turn": turns }),
                duration_ms,
                ..Default::default()
            },
        );
    }

    pub fn set_status(&self, id: &str, status: &str) {
        let Some(mut state) = self.load(id) else {
            return;
        };
        if let Some(o) = state.as_object_mut() {
            o.insert("status".into(), json!(status));
        }
        let _ = self.write_state(id, state);
    }

    pub fn backfill_ordinals(&self) -> usize {
        let metas = self.list_meta();
        let mut next = metas
            .iter()
            .filter_map(|s| s.get("ordinal").and_then(|v| v.as_u64()))
            .max()
            .unwrap_or(0);
        let mut missing: Vec<&Value> = metas
            .iter()
            .filter(|s| s.get("ordinal").and_then(|v| v.as_u64()).is_none())
            .collect();
        missing.sort_by_key(|s| {
            s.get("createdAt")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        });
        let n = missing.len();
        for s in missing {
            let Some(id) = s.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            next += 1;
            let mut patch = Map::new();
            patch.insert("ordinal".into(), json!(next));
            self.update_meta(id, patch);
        }
        n
    }

    pub fn list(&self) -> Vec<String> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut ids: Vec<String> = rd
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                name.strip_suffix(".json").map(|s| s.to_string())
            })
            .collect();
        ids.sort();
        ids.reverse();
        ids
    }

    pub fn list_meta(&self) -> Vec<Value> {
        self.list().into_iter().filter_map(|id| self.load(&id)).collect()
    }

    pub fn delete(&self, id: &str) -> bool {
        if id.is_empty() || id.contains("..") || id.contains('/') || id.contains('\\') {
            return false;
        }
        let mut n = 0;
        for p in [self.state_path(id), self.log_path(id)] {
            if p.exists() && std::fs::remove_file(&p).is_ok() {
                n += 1;
            }
        }
        n > 0
    }

    pub fn clean(&self) -> usize {
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return 0;
        };
        let mut n = 0;
        for e in rd.filter_map(|e| e.ok()) {
            if std::fs::remove_file(e.path()).is_ok() {
                n += 1;
            }
        }
        n
    }
}




#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Goto {
    pub enabled: bool,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextItem {
    pub role: String,
    pub content: String,
    pub id: Option<String>,
    pub visible: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub goto: Option<Goto>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ok: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denied: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<Option<String>>,
}

impl ContextItem {
    fn base(role: &str, content: String, id: Option<String>, visible: bool) -> Self {
        ContextItem {
            role: role.to_string(),
            content,
            id,
            visible,
            tool_calls: None,
            tool_call_id: None,
            goto: None,
            name: None,
            args: None,
            ok: None,
            denied: None,
            explanation: None,
        }
    }
}

fn js_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "null".into(),
        other => other.to_string(),
    }
}

fn goto_of(meta: &Map<String, Value>) -> Option<Goto> {
    let g = meta.get("goto")?.as_object()?;
    Some(Goto {
        enabled: g.get("enabled").map(truthy).unwrap_or(false),
        label: g
            .get("label")
            .map(|v| if v.is_null() { String::new() } else { js_string(v) })
            .unwrap_or_default(),
    })
}

pub fn js_truthy(v: &Value) -> bool {
    truthy(v)
}

pub fn js_string_of(v: &Value) -> String {
    js_string(v)
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Null => false,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        _ => true,
    }
}

impl SessionStore {
    pub fn list_visible_context_messages(&self, id: &str) -> Vec<Value> {
        let mut out = Vec::new();
        for wrap in self.list_events_folded(id) {
            if wrap.event_type != "context_window" {
                continue;
            }
            let Some(p) = wrap.payload.as_object() else {
                continue;
            };
            let role = p.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if role.is_empty() {
                continue;
            }
            match p.get("content") {
                None | Some(Value::Null) => continue,
                _ => {}
            }
            if p.get("kind").and_then(|v| v.as_str()) == Some("provider_messages_snapshot") {
                continue;
            }
            if wrap.meta.get("visible") == Some(&Value::Bool(false)) {
                continue;
            }
            if role != "user" && role != "assistant" {
                continue;
            }
            out.push(json!({
                "event_id": wrap.event_id,
                "role": role,
                "content": js_string(p.get("content").unwrap()),
                "meta": wrap.meta,
            }));
        }
        out
    }

    pub fn build_context_window_items(&self, id: &str) -> Vec<ContextItem> {
        context_items_from_events(self.list_events_folded(id))
    }
}

pub fn context_items_from_events(events: Vec<EventWrapper>) -> Vec<ContextItem> {
    {
        let mut items: Vec<ContextItem> = Vec::new();

        for wrap in events {
            let Some(p) = wrap.payload.as_object() else {
                continue;
            };
            let visible = wrap.meta.get("visible") != Some(&Value::Bool(false));
            let eid = Some(wrap.event_id.clone());
            let goto = goto_of(&wrap.meta);

            let kind = p.get("kind").and_then(|v| v.as_str());
            if wrap.event_type == "context_window" && kind == Some("provider_messages_snapshot") {
                continue;
            }

            if wrap.event_type == "context_window" {
                let role = p.get("role").and_then(|v| v.as_str()).unwrap_or("");
                let content = p.get("content");
                if !role.is_empty() && !matches!(content, None | Some(Value::Null)) {
                    let mut it =
                        ContextItem::base(role, js_string(content.unwrap()), eid, visible);
                    it.goto = goto;
                    items.push(it);
                    continue;
                }
            }

            let is_reasoning = wrap.event_type == "reasoning"
                || p.get("type").and_then(|v| v.as_str()) == Some("reasoning");
            if is_reasoning {
                let text = p
                    .get("text")
                    .filter(|v| truthy(v))
                    .or_else(|| p.get("content").filter(|v| truthy(v)));
                if let Some(t) = text {
                    let mut it = ContextItem::base("reasoning", js_string(t), eid, visible);
                    it.goto = goto;
                    items.push(it);
                    continue;
                }
            }

            if wrap.event_type == "tool_call" {
                if let Some(name) = p.get("name").filter(|v| truthy(v)).and_then(|v| v.as_str()) {
                    let call_id = first_str(&[p.get("id"), p.get("tool_call_id")])
                        .or_else(|| eid.clone())
                        .unwrap_or_else(|| format!("call_{}", items.len()));
                    let args_str = match p.get("args") {
                        Some(Value::String(s)) => s.clone(),
                        Some(v) if !v.is_null() => v.to_string(),
                        _ => "{}".to_string(),
                    };
                    let mut it = ContextItem::base("assistant", String::new(), eid, visible);
                    it.tool_calls = Some(json!([{
                        "id": call_id,
                        "type": "function",
                        "function": { "name": name, "arguments": args_str },
                    }]));
                    it.name = Some(name.to_string());
                    it.args = p.get("args").cloned();
                    it.explanation =
                        Some(p.get("explanation").filter(|v| truthy(v)).map(js_string));
                    it.goto = goto;
                    items.push(it);
                    continue;
                }
            }

            if wrap.event_type == "tool_result" {
                if let Some(name) = p.get("name").filter(|v| truthy(v)).and_then(|v| v.as_str()) {
                    let call_id =
                        first_str(&[p.get("tool_call_id"), p.get("id"), p.get("call_id")])
                            .or_else(|| eid.clone())
                            .unwrap_or_else(|| format!("call_{}", items.len()));
                    let content = match p.get("text") {
                        Some(t) if !t.is_null() => js_string(t),
                        _ => match p.get("content") {
                            Some(Value::String(s)) => s.clone(),
                            Some(v) if !v.is_null() => v.to_string(),
                            _ => Value::Object(p.clone()).to_string(),
                        },
                    };
                    let mut it = ContextItem::base("tool", content, eid, visible);
                    it.tool_call_id = Some(call_id);
                    it.name = Some(name.to_string());
                    it.ok = p.get("ok").cloned();
                    it.denied = p.get("denied").cloned();
                    it.explanation =
                        Some(p.get("explanation").filter(|v| truthy(v)).map(js_string));
                    it.goto = goto;
                    items.push(it);
                }
            }
        }
        items
    }
}

fn first_str(cands: &[Option<&Value>]) -> Option<String> {
    for c in cands {
        if let Some(v) = c {
            if truthy(v) {
                return Some(js_string(v));
            }
        }
    }
    None
}
