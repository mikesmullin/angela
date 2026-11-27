use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::agent_loader;
use crate::conv_store::ConvStore;
use crate::harness::{self, LiveTabs, McpPool, TurnRequest};
use crate::http::{Handler, Request, Responder, ResponseSink, SocketSink};
use crate::session_store::{EventWrapper, NewSession, SessionStore};

pub struct AppState {
    pub store: ConvStore,
    pub project_root: String,
    pub port: u16,
    pub auto_approve: bool,
    pub mcp_root: String,
    pub tabs: LiveTabs,
    pub cancels: crate::cancel::Cancels,
    pub approvals: std::sync::Arc<crate::approvals::Approvals>,
    pub mcp: McpPool,
    pub rt: tokio::runtime::Runtime,
    pub prompt_sender: Option<PromptSender>,
    pub queues: crate::queue::Queues,
    pub queue_sender: Option<QueueSender>,
    pub seat_sender: Option<SeatSender>,
}

pub type PromptSender = std::sync::Arc<dyn Fn(&str, String, String, String) + Send + Sync>;

pub type SeatSender = std::sync::Arc<dyn Fn(&str) + Send + Sync>;

pub type QueueSender = std::sync::Arc<dyn Fn() + Send + Sync>;

fn seat_changed(st: &AppState, session: &str) {
    if let Some(send) = &st.seat_sender {
        send(session);
    }
}

impl AppState {
    pub fn new(project_root: impl AsRef<std::path::Path>, port: u16) -> std::io::Result<Self> {
        let root = project_root.as_ref().to_path_buf();
        crate::media::set_media_root(root.join(".angela").join("media"));
        let store = ConvStore::new(&root)
            .map_err(|e| std::io::Error::other(format!("conversations.db: {e}")))?;
        let report = crate::migrate::run(&store, &root);
        if report.imported > 0 {
            eprintln!(
                "angela: imported {} session(s) ({} nodes) into conversations.db",
                report.imported, report.nodes
            );
            match crate::migrate::verify(&store, &root) {
                Ok(n) => eprintln!("angela: verified {n} imported transcript(s) against the files"),
                Err(e) => eprintln!("angela: MIGRATION VERIFY FAILED — {e}"),
            }
        }
        for f in &report.failed {
            eprintln!("angela: import failed — {f}");
        }
        let imported = store.agents().import_files(&root);
        if imported > 0 {
            eprintln!("angela: imported {imported} agent definition(s) into conversations.db");
        }
        let backfilled = store.backfill_ordinals();
        if backfilled > 0 {
            eprintln!("angela: numbered {backfilled} pre-existing session(s)");
        }
        Ok(AppState {
            store,
            project_root: root.to_string_lossy().into_owned(),
            port,
            auto_approve: false,
            mcp_root: std::env::var("ANGELA_MCP_ROOT").unwrap_or_else(|_| "/workspace/mcp".into()),
            tabs: LiveTabs::default(),
            cancels: crate::cancel::Cancels::default(),
            approvals: std::sync::Arc::new(crate::approvals::Approvals::new()),
            mcp: McpPool::default(),
            rt: tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?,
            prompt_sender: None,
            queues: crate::queue::Queues::new(),
            queue_sender: None,
            seat_sender: None,
        })
    }
}

fn clone_telemetry(v: Option<&Value>) -> Value {
    match v {
        Some(x) if x.is_object() => x.clone(),
        _ => Value::Null,
    }
}

fn goto_from_meta(meta: &Map<String, Value>) -> Value {
    match meta.get("goto") {
        Some(g) if g.is_object() => json!({
            "enabled": g.get("enabled").map(crate::session_store::js_truthy).unwrap_or(false),
            "label": g.get("label").and_then(|v| v.as_str()).unwrap_or(""),
        }),
        _ => Value::Null,
    }
}

fn invite_mentions(store: &ConvStore, session: &str, text: &str) -> usize {
    let names = crate::group::mentions(text);
    if names.is_empty() {
        return 0;
    }
    let definitions = store.agents().list();
    let identity = store.identity();
    let mut seated = 0;
    for name in names {
        if let Some(def) = definitions.iter().find(|d| d.name.eq_ignore_ascii_case(&name)) {
            if store.invite_mentioned_agent(session, def).is_some() {
                seated += 1;
            }
        } else if let Some(user) = identity
            .user_by_name(&name)
            .filter(|u| !u.disabled)
        {
            let display = if user.display_name.trim().is_empty() {
                user.name.clone()
            } else {
                user.display_name.clone()
            };
            if store.invite_user(session, &user.name, &display).is_some() {
                seated += 1;
            }
        }
    }
    seated
}

pub fn reconstruct_messages_from_log(store: &SessionStore, id: &str) -> Vec<Value> {
    reconstruct_from_events(store.list_events_folded(id))
}

pub fn reconstruct_messages(store: &ConvStore, id: &str) -> Vec<Value> {
    let mut rows = reconstruct_from_events(store.list_events_folded(id));
    let roster: std::collections::HashMap<i64, String> = store
        .participants(id)
        .into_iter()
        .map(|p| (p.id, p.handle))
        .collect();
    let authors: std::collections::HashMap<String, String> = store
        .path(id)
        .into_iter()
        .filter_map(|n| {
            let pid = n.participant_id?;
            Some((n.id, roster.get(&pid)?.clone()))
        })
        .collect();
    for row in &mut rows {
        let Some(eid) = row.get("eventId").and_then(|v| v.as_str()).map(String::from) else {
            continue;
        };
        if let Some(handle) = authors.get(&eid) {
            if let Some(o) = row.as_object_mut() {
                o.insert("handle".into(), json!(handle));
            }
        }
    }
    rows
}

fn prompt_parent(store: &ConvStore, id: &str, target: &str) -> Option<Option<String>> {
    store.path(id).into_iter()
        .find(|n| n.id == target && n.kind == "context_window" && n.role.as_deref() == Some("user"))
        .map(|n| n.parent_id)
}

pub fn reconstruct_from_events(events: Vec<EventWrapper>) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::new();
    let marks: std::collections::HashMap<String, (String, Value)> = events
        .iter()
        .filter_map(|w| {
            let t = w.meta.get("turn")?.as_str()?.to_string();
            Some((
                w.event_id.clone(),
                (t, w.meta.get("round").cloned().unwrap_or(Value::Null)),
            ))
        })
        .collect();
    let times: std::collections::HashMap<String, String> = events
        .iter()
        .map(|w| (w.event_id.clone(), w.ts.clone()))
        .collect();
    let mut last_gen: Value = Value::Null;

    for wrap in events {
        let p = &wrap.payload;
        if wrap.event_type == "gen_info" && !p.is_null() {
            last_gen = match wrap.meta.get("telemetry") {
                Some(t) if t.is_object() => t.clone(),
                _ => p.clone(),
            };
            continue;
        }
        let Some(po) = p.as_object() else { continue };
        let frozen = wrap.meta.get("telemetry").filter(|t| t.is_object());
        let visible = wrap.meta.get("visible") != Some(&Value::Bool(false));
        let goto = goto_from_meta(&wrap.meta);
        let eid = wrap.event_id.clone();
        let role = po.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let has_content = !matches!(po.get("content"), None | Some(Value::Null));
        let is_snapshot =
            po.get("kind").and_then(|v| v.as_str()) == Some("provider_messages_snapshot");

        if wrap.event_type == "error" {
            if let Some(text) = po.get("message").or_else(|| po.get("text")).and_then(Value::as_str) {
                messages.push(json!({
                    "id": eid, "eventId": eid, "kind": "error", "text": text,
                    "handle": po.get("handle").cloned().unwrap_or(Value::Null),
                    "visible": visible, "goto": goto,
                }));
                continue;
            }
        }

        if wrap.event_type == "context_window" && !role.is_empty() && has_content && !is_snapshot {
            let telemetry = match frozen {
                Some(f) => f.clone(),
                None if role == "assistant" => last_gen.clone(),
                None => Value::Null,
            };
            messages.push(json!({
                "id": eid,
                "eventId": eid,
                "kind": if role == "user" { "user" } else { "assistant" },
                "text": crate::session_store::js_string_of(po.get("content").unwrap()),
                "from": po.get("from").cloned().unwrap_or(Value::Null),
                "visible": visible,
                "telemetry": clone_telemetry(Some(&telemetry)),
                "goto": goto,
            }));
            continue;
        }

        let is_reasoning = wrap.event_type == "reasoning"
            || po.get("type").and_then(|v| v.as_str()) == Some("reasoning");
        let rtext = po
            .get("text")
            .filter(|v| crate::session_store::js_truthy(v))
            .or_else(|| po.get("content").filter(|v| crate::session_store::js_truthy(v)));
        if is_reasoning {
            if let Some(t) = rtext {
                let tel = match wrap.meta.get("telemetry") {
                    Some(x) if x.is_object() => x.clone(),
                    _ => match frozen {
                        Some(f) => f.clone(),
                        None => last_gen.clone(),
                    },
                };
                messages.push(json!({
                    "id": eid,
                    "eventId": eid,
                    "kind": "reasoning",
                    "text": crate::session_store::js_string_of(t),
                    "visible": visible,
                    "telemetry": clone_telemetry(Some(&tel)),
                    "goto": goto,
                }));
                continue;
            }
        }

        if wrap.event_type == "system" {
            if let Some(text) = po.get("content").and_then(|v| v.as_str()) {
                messages.push(json!({
                    "id": eid, "eventId": eid, "kind": "system_prompt",
                    "text": text, "bytes": text.len(), "visible": visible, "goto": goto,
                }));
                continue;
            }
        }
        if wrap.event_type == "agent_prompt" {
            if let Some(text) = po.get("content").and_then(|v| v.as_str()) {
                messages.push(json!({
                    "id": eid, "eventId": eid, "kind": "agent_prompt",
                    "text": text, "bytes": text.len(),
                    "model": po.get("model").cloned().unwrap_or(Value::Null),
                    "historyTurns": po.get("historyTurns").cloned().unwrap_or(Value::Null),
                    "visible": visible, "goto": goto,
                }));
                continue;
            }
        }
        if wrap.event_type == "stopped" {
            messages.push(json!({
                "id": eid, "eventId": eid, "kind": "system",
                "text": crate::harness::STOPPED_NOTICE,
                "visible": visible, "goto": goto,
            }));
            continue;
        }
        if wrap.event_type == "approval" {
            messages.push(json!({
                "id": eid, "eventId": eid, "kind": "approval",
                "name": po.get("name").cloned().unwrap_or(Value::Null),
                "command": po.get("command").cloned().unwrap_or(Value::Null),
                "args": po.get("args").cloned().unwrap_or(Value::Null),
                "reason": po.get("reason").cloned().unwrap_or(Value::Null),
                "status": po.get("status").cloned().unwrap_or(json!("deny")),
                "visible": visible, "goto": goto,
            }));
            continue;
        }
        if wrap.event_type == "decision" {
            messages.push(json!({
                "id": eid, "eventId": eid, "kind": "decision",
                "verdict": p.clone(),
                "speak": po.get("speak").cloned().unwrap_or(Value::Null),
                "confidence": po.get("confidence").cloned().unwrap_or(Value::Null),
                "reason": po.get("reason").cloned().unwrap_or(Value::Null),
                "error": po.get("error").cloned().unwrap_or(Value::Null),
                "visible": visible, "goto": goto,
            }));
            continue;
        }

        let name = po.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let tel = match frozen {
            Some(f) => f.clone(),
            None => last_gen.clone(),
        };
        if wrap.event_type == "tool_call" && !name.is_empty() {
            messages.push(json!({
                "id": eid,
                "eventId": eid,
                "kind": "tool_call",
                "name": name,
                "args": po.get("args").cloned().unwrap_or(Value::Null),
                "explanation": po.get("explanation").cloned().unwrap_or(Value::Null),
                "visible": visible,
                "telemetry": clone_telemetry(Some(&tel)),
                "full_chars": po.get("full_chars").cloned().unwrap_or(Value::Null),
                "truncated": po.get("truncated").cloned().unwrap_or(Value::Null),
                "goto": goto,
            }));
        } else if wrap.event_type == "tool_result" && !name.is_empty() {
            messages.push(json!({
                "id": eid,
                "eventId": eid,
                "kind": "tool_result",
                "name": name,
                "ok": po.get("ok").cloned().unwrap_or(Value::Null),
                "denied": po.get("denied").cloned().unwrap_or(Value::Null),
                "text": po.get("text").cloned().unwrap_or(Value::Null),
                "explanation": po.get("explanation").cloned().unwrap_or(Value::Null),
                "visible": visible,
                "telemetry": clone_telemetry(Some(&tel)),
                "full_chars": po.get("full_chars").cloned().unwrap_or(Value::Null),
                "truncated": po.get("truncated").cloned().unwrap_or(Value::Null),
                "goto": goto,
            }));
        }
    }
    let turn_at = |row: &Value| -> Option<String> {
        let id = row.get("eventId")?.as_str()?;
        marks.get(id).map(|(t, _)| t.clone())
    };
    let seq: Vec<Option<String>> = messages.iter().map(turn_at).collect();
    for row in messages.iter_mut() {
        let ts = row
            .get("eventId")
            .and_then(|v| v.as_str())
            .and_then(|id| times.get(id))
            .cloned();
        if let (Some(ts), Some(o)) = (ts, row.as_object_mut()) {
            o.insert("ts".into(), json!(ts));
        }
    }
    for (i, row) in messages.iter_mut().enumerate() {
        let Some(turn) = seq[i].clone() else { continue };
        let round = row
            .get("eventId")
            .and_then(|v| v.as_str())
            .and_then(|id| marks.get(id))
            .map(|(_, r)| r.clone())
            .unwrap_or(Value::Null);
        let Some(o) = row.as_object_mut() else { continue };
        let first = i == 0 || seq[i - 1].as_deref() != Some(turn.as_str());
        let last = i + 1 >= seq.len() || seq[i + 1].as_deref() != Some(turn.as_str());
        o.insert("turn".into(), json!(turn));
        o.insert("round".into(), round);
        o.insert("turnFirst".into(), json!(first));
        o.insert("turnLast".into(), json!(last));
    }
    messages
}

fn agent_from_body(body: &Value, base: Option<crate::agent_loader::AgentDef>) -> crate::agent_loader::AgentDef {
    let mut def = base.unwrap_or_default();
    let Some(o) = body.as_object() else { return def };
    let st = |k: &str| o.get(k).and_then(|v| v.as_str()).map(String::from);
    let list = |k: &str| -> Option<Vec<String>> {
        match o.get(k) {
            Some(Value::Array(a)) => Some(
                a.iter()
                    .filter_map(|x| x.as_str())
                    .map(|x| x.trim().to_string())
                    .filter(|x| !x.is_empty())
                    .collect(),
            ),
            Some(Value::String(text)) => Some(
                text.lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .collect(),
            ),
            _ => None,
        }
    };
    if let Some(v) = st("name") { def.name = v.trim().to_string(); }
    if let Some(v) = st("description") { def.description = v; }
    if let Some(v) = st("model") { def.model = v; }
    if let Some(v) = st("reasoningEffort") { def.reasoning_effort = v; }
    if let Some(v) = list("models") { def.models = v; }
    if let Some(v) = list("mcp") { def.mcp = v; }
    if let Some(v) = st("allowlist") { def.allowlist = v; }
    if let Some(v) = st("soul").or_else(|| st("system")) { def.soul = v; }
    if let Some(v) = st("goals") { def.goals = v; }
    if let Some(v) = st("memory") { def.memory = v; }
    if let Some(v) = st("engagement") { def.engagement = v; }
    if let Some(v) = st("policyMode").or_else(|| st("policy_mode")) {
        if !v.trim().is_empty() { def.policy_mode = v; }
    }
    if let Some(v) = st("avatar") { def.avatar = Some(v).filter(|a| !a.trim().is_empty()); }
    if let Some(v) = st("accentColor").or_else(|| st("accent_color")) {
        def.accent_color = crate::agent_loader::normalize_hex(&v);
    }
    if let Some(v) = st("icon") {
        def.icon = crate::agent_loader::normalize_icon(&v);
    }
    match o.get("maxTurns").or_else(|| o.get("max_turns")) {
        Some(Value::Null) => def.max_turns = None,
        Some(v) => def.max_turns = crate::agent_loader::normalize_max_turns(Some(v)),
        None => {}
    }
    match o.get("containerFile").or_else(|| o.get("containerfile")) {
        Some(Value::Null) => def.containerfile = String::new(),
        Some(Value::String(s)) => def.containerfile = s.trim().to_string(),
        _ => {}
    }
    match o.get("contextWindow").or_else(|| o.get("context_window")) {
        Some(Value::Null) => def.context_window = None,
        Some(v) => {
            if let Some(n) = v.as_u64() { def.context_window = Some(n); }
            if v.as_str().map(|t| t.trim().is_empty()).unwrap_or(false) {
                def.context_window = None;
            }
        }
        None => {}
    }
    if let Some(v) = o.get("parallelTools").or_else(|| o.get("parallel_tools")) {
        def.parallel_tools = crate::session_store::js_truthy(v);
    }
    match o.get("toolsEnabled").or_else(|| o.get("tools_enabled")) {
        Some(Value::Null) => def.tools_enabled = None,
        Some(other) => {
            let l = list("toolsEnabled").or_else(|| list("tools_enabled")).unwrap_or_default();
            let _ = other;
            def.tools_enabled = if l.is_empty() { None } else { Some(l) };
        }
        None => {}
    }
    if let Some(Value::Object(env)) = o.get("env") {
        def.env = env
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
            .collect();
    }
    if let Some(Value::Array(a)) = o.get("starters") {
        def.starters = a.clone();
    }
    def
}

fn session_route(path: &str) -> Option<(&str, &str)> {
    let rest = path.strip_prefix("/api/sessions/")?;
    match rest.split_once('/') {
        Some((id, tail)) => Some((id, tail)),
        None => Some((rest, "")),
    }
}

pub fn roster_json(store: &ConvStore, conv: &str) -> Vec<Value> {
    let agents = store.agents();
    store
        .participants(conv)
        .iter()
        .map(|p| {
            let def = p.agent_name.as_deref().and_then(|n| agents.get(n));
            p.to_json_with(def.as_ref())
        })
        .collect()
}

pub fn leaf_messages(store: &ConvStore, id: &str) -> Vec<Value> {
    let msgs = reconstruct_messages(store, id);
    let by_id: std::collections::HashMap<&str, &Value> = msgs
        .iter()
        .filter_map(|m| m.get("eventId").and_then(|v| v.as_str()).map(|k| (k, m)))
        .collect();
    store
        .memory_leaves(id)
        .into_iter()
        .map(|node_id| {
            by_id
                .get(node_id.as_str())
                .map(|m| (*m).clone())
                .unwrap_or(Value::Null)
        })
        .collect()
}

fn memory_view_json(store: &ConvStore, id: &str) -> Value {
    let msgs = leaf_messages(store, id);
    let parts = crate::memory::view(store, id);
    let count = crate::memory::count(store, id);
    json!({
        "sessionId": id,
        "count": count,
        "compacted": parts.iter().any(|p| p.coord.width() > 1),
        "parts": parts.iter().map(|p| p.to_json(Some(&msgs))).collect::<Vec<_>>(),
        "roots": crate::memory::roots(count)
            .iter()
            .map(|c| c.reference())
            .collect::<Vec<_>>(),
    })
}

pub struct Server {
    pub state: Arc<AppState>,
}

impl Server {
    pub fn new(state: Arc<AppState>) -> Self {
        Server { state }
    }

    pub fn sessions_list_for_test(&self) -> Value {
        self.sessions_list()
    }

    fn sessions_list(&self) -> Value {
        let sessions: Vec<Value> = self
            .state
            .store
            .list_meta()
            .into_iter()
            .map(|s| {
                let id = s.get("id").cloned().unwrap_or(Value::Null);
                let title = [s.get("title"), s.get("agent")]
                    .into_iter()
                    .flatten()
                    .find(|v| crate::session_store::js_truthy(v))
                    .cloned()
                    .unwrap_or_else(|| id.clone());
                json!({
                    "id": id,
                    "ordinal": s.get("ordinal").cloned().unwrap_or(Value::Null),
                    "title": title,
                    "agent": s.get("agent").cloned().unwrap_or(Value::Null),
                    "model": s.get("model").cloned().unwrap_or(Value::Null),
                    "createdAt": s.get("createdAt").cloned().unwrap_or(Value::Null),
                    "updatedAt": s.get("updatedAt").cloned().unwrap_or(Value::Null),
                    "status": s.get("status").cloned().unwrap_or(Value::Null),
                    "turns": s.get("turns").cloned().unwrap_or(json!(0)),
                    "contextWindow": s.get("contextWindow").cloned().unwrap_or(Value::Null),
                    "project": s
                        .get("project")
                        .cloned()
                        .unwrap_or_else(|| json!("")),
                })
            })
            .collect();
        json!({ "projectRoot": self.state.project_root, "sessions": sessions })
    }
}

impl Handler for Server {
    fn handle(&self, req: &Request, out: Responder<'_>) {
        let mut sink = SocketSink::new(out);
        self.dispatch(req, &mut sink);
    }
}

impl Server {
    pub fn dispatch(&self, req: &Request, sink: &mut dyn ResponseSink) {
        let m = req.method.as_str();
        let p = req.path.as_str();
        let st = &self.state;
        match (m, p) {
            ("GET", "/api/health") => sink.json(
                200,
                &json!({
                    "ok": true,
                    "port": st.port,
                    "projectRoot": st.project_root,
                    "autoApprove": st.auto_approve,
                    "agents": st.store.agents().list()
                        .iter()
                        .map(|a| json!(a.name))
                        .collect::<Vec<_>>(),
                    "mcpRoot": st.mcp_root,
                }),
            ),

            ("GET", "/api/agents") => {
                let agents: Vec<Value> =
                    st.store.agents().list().iter().map(|a| a.to_json()).collect();
                sink.json(
                    200,
                    &json!({
                        "projectRoot": st.project_root,
                        "agents": agents,
                        "engagements": crate::agent_loader::ENGAGEMENTS,
                        "mcpPresets": crate::mcp_presets::names(),
                    }),
                )
            }

            ("GET", "/api/settings") => sink.json(
                200,
                &json!({
                    "settings": st.store.settings(),
                    "keys": crate::settings::KEYS
                        .iter()
                        .map(|(k, about)| json!({ "key": k, "about": about }))
                        .collect::<Vec<_>>(),
                }),
            ),

            ("POST", "/api/settings") => {
                let body = req.json();
                let patch = body.as_object().cloned().unwrap_or_default();
                let n = st.store.set_settings(&patch);
                sink.json(
                    200,
                    &json!({ "ok": true, "saved": n, "settings": st.store.settings() }),
                )
            }

            ("POST", "/api/agents") => {
                let body = req.json();
                let def = agent_from_body(&body, None);
                match st.store.agents().save(&def, None) {
                    Ok(a) => sink.json(200, &json!({ "ok": true, "agent": a.to_json() })),
                    Err(e) => sink.json(400, &json!({ "error": e.to_string() })),
                }
            }

            ("GET", path) if path.starts_with("/api/agents/") => {
                let name = &path["/api/agents/".len()..];
                match st.store.agents().get(name) {
                    Some(a) => sink.json(200, &json!({ "agent": a.to_json() })),
                    None => sink.json(404, &json!({ "error": format!("no agent named {name:?}") })),
                }
            }

            ("POST", path) if path.starts_with("/api/agents/") => {
                let name = &path["/api/agents/".len()..];
                let agents = st.store.agents();
                let Some(existing) = agents.get(name) else {
                    sink.json(404, &json!({ "error": format!("no agent named {name:?}") }));
                    return;
                };
                let body = req.json();
                let def = agent_from_body(&body, Some(existing));
                match agents.save(&def, Some(name)) {
                    Ok(a) => sink.json(200, &json!({ "ok": true, "agent": a.to_json() })),
                    Err(e) => sink.json(400, &json!({ "error": e.to_string() })),
                }
            }

            ("DELETE", path) if path.starts_with("/api/agents/") => {
                let name = &path["/api/agents/".len()..];
                match st.store.agents().delete(name) {
                    Ok(seated) => sink.json(200, &json!({ "ok": true, "name": name, "seated": seated })),
                    Err(e) => sink.json(404, &json!({ "error": e.to_string() })),
                }
            }

            ("GET", "/api/sessions") => sink.json(200, &self.sessions_list()),

            ("DELETE", "/api/sessions") => {
                let n = st.store.clean();
                sink.json(200, &json!({ "ok": true, "removed": n }))
            }

            ("POST", "/api/session/new") => {
                let body = req.json();
                let agent = body.get("agent").and_then(|v| v.as_str()).map(String::from);
                let model = body.get("model").and_then(|v| v.as_str()).map(String::from);
                let project = body.get("project").and_then(Value::as_str).unwrap_or("").trim();
                if !project.is_empty() {
                    if let Err(error) = crate::projects::normalize(project) {
                        sink.json(400, &json!({"ok": false, "error": error}));
                        return;
                    }
                }
                let mut default_agents = Vec::new();
                match st.store.project_default_agents(project) {
                    Ok(names) => for name in names {
                        match st.store.agents().get(&name) {
                            Some(def) => default_agents.push(def),
                            None => { sink.json(400, &json!({"ok": false, "error": format!("default agent no longer exists: {name}")})); return; }
                        }
                    },
                    Err(error) => { sink.json(400, &json!({"ok": false, "error": error})); return; }
                }
                match st.store.create(NewSession {
                    agent,
                    model,
                    title: body
                        .get("title")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    ..Default::default()
                }) {
                    Ok(s) => {
                        if !project.is_empty() {
                            let id = s["id"].as_str().unwrap_or("");
                            if let Err(error) = st.store.assign_project(id, project) {
                                sink.json(500, &json!({"ok": false, "error": error}));
                                return;
                            }
                            let mut seated = 0;
                            for def in &default_agents {
                                if st.store.invite_mentioned_agent(id, def).is_none() {
                                    sink.json(500, &json!({"ok": false, "error": format!("could not invite default agent: {}", def.name)}));
                                    return;
                                }
                                seated += 1;
                            }
                            if seated > 0 { seat_changed(st, id); }
                            sink.json(200, &json!({ "ok": true, "session": st.store.load(id) }))
                        } else {
                            sink.json(200, &json!({ "ok": true, "session": s }))
                        }
                    }
                    Err(e) => sink.json(500, &json!({ "error": e.to_string() })),
                }
            }

            ("POST", path) if matches!(session_route(path), Some((_, "cursor"))) => {
                let (id, _) = session_route(path).unwrap();
                let body = req.json();
                let target = body
                    .get("eventId")
                    .or_else(|| body.get("event_id"))
                    .and_then(|v| v.as_str());
                if st.store.load(id).is_none() {
                    sink.json(404, &json!({ "error": "not found" }));
                    return;
                }
                let before = body.get("before").and_then(Value::as_bool).unwrap_or(false);
                let replacement_parent = if before {
                    match target.and_then(|target| prompt_parent(&st.store, id, target)) {
                        Some(parent) => Some(parent),
                        _ => {
                            sink.json(400, &json!({"ok": false, "error": "user message not found on current branch"}));
                            return;
                        }
                    }
                } else { None };
                let destination = if before { replacement_parent.as_ref().and_then(|p| p.as_deref()) } else { target };
                let ok = st.store.set_cursor(id, destination);
                sink.json(
                    if ok { 200 } else { 400 },
                    &json!({ "ok": ok, "sessionId": id, "cursor": st.store.cursor(id),
                             "messages": if ok { reconstruct_messages(&st.store, id) } else { vec![] } }),
                )
            }

            ("GET", path) if matches!(session_route(path), Some((_, "tree"))) => {
                let (id, _) = session_route(path).unwrap();
                if st.store.load(id).is_none() {
                    sink.json(404, &json!({ "error": "not found" }));
                    return;
                }
                let nodes = st.store.all_nodes(id);
                let on_path: Vec<String> =
                    st.store.path(id).into_iter().map(|n| n.id).collect();
                sink.json(
                    200,
                    &json!({
                        "sessionId": id,
                        "cursor": st.store.cursor(id),
                        "onPath": on_path,
                        "nodes": nodes.iter().map(|n| n.to_json()).collect::<Vec<_>>(),
                    }),
                )
            }

            ("GET", path) if matches!(session_route(path), Some((_, "participants"))) => {
                let (id, _) = session_route(path).unwrap();
                sink.json(
                    200,
                    &json!({ "sessionId": id, "participants": roster_json(&st.store, id) }),
                )
            }

            ("POST", path) if matches!(session_route(path), Some((_, "participants"))) => {
                let (id, _) = session_route(path).unwrap();
                if st.store.load(id).is_none() {
                    sink.json(404, &json!({ "error": "not found" }));
                    return;
                }
                let body = req.json();
                let gs = |k: &str| body.get(k).and_then(|v| v.as_str());
                if let Some(name) = gs("name") {
                    let agent = st
                        .store
                        .agents()
                        .list()
                        .into_iter()
                        .find(|d| d.name.eq_ignore_ascii_case(name));
                    let invited = match &agent {
                        Some(def) => {
                            let handle = def.name.to_ascii_lowercase();
                            st.store.invite_mentioned_agent(id, def).or_else(|| {
                                st.store
                                    .participants(id)
                                    .into_iter()
                                    .find(|q| q.handle == handle && q.active)
                            })
                        }
                        None => {
                            let Some(user) = st
                                .store
                                .identity()
                                .user_by_name(name)
                                .filter(|u| !u.disabled)
                            else {
                                sink.json(
                                    404,
                                    &json!({
                                        "error": format!(
                                                    "no agent or user named {name:?} \
                                                     — define an agent with + in the Agents rail"
                                                )
                                    }),
                                );
                                return;
                            };
                            let display = if user.display_name.trim().is_empty() {
                                user.name.clone()
                            } else {
                                user.display_name.clone()
                            };
                            let handle = crate::conv_store::handle_from(&user.name).to_ascii_lowercase();
                            st.store.invite_user(id, &user.name, &display).or_else(|| {
                                st.store
                                    .participants(id)
                                    .into_iter()
                                    .find(|q| q.handle == handle && q.active)
                            })
                        }
                    };
                    match invited {
                        Some(p) => {
                            seat_changed(st, id);
                            sink.json(200, &json!({ "ok": true, "participant": p.to_json() }))
                        }
                        None => sink.json(400, &json!({ "error": "could not seat participant" })),
                    }
                    return;
                }
                let kind = gs("kind").unwrap_or("agent");
                let agent_name = gs("agentName").or_else(|| gs("agent"));
                let def = agent_name.and_then(|n| st.store.agents().get(n));
                let system = gs("systemPrompt")
                    .map(String::from)
                    .or_else(|| def.as_ref().map(|d| d.system_prompt()))
                    .unwrap_or_default();
                let model = gs("model")
                    .map(String::from)
                    .or_else(|| def.as_ref().map(|d| d.model.clone()).filter(|m| !m.is_empty()));
                match st.store.add_participant(
                    id,
                    kind,
                    gs("handle").unwrap_or(""),
                    gs("displayName"),
                    agent_name,
                    model.as_deref(),
                    &system,
                    gs("avatar"),
                    gs("role"),
                ) {
                    Some(p) => {
                        seat_changed(st, id);
                        sink.json(200, &json!({ "ok": true, "participant": p.to_json() }))
                    }
                    None => sink.json(400, &json!({ "error": "could not seat participant" })),
                }
            }

            ("POST", path) if matches!(session_route(path), Some((_, "compact"))) => {
                let (id, _) = session_route(path).unwrap();
                let Some(meta) = st.store.effective_session(id).and_then(|v| v.get("effective").cloned()) else {
                    sink.json(404, &json!({ "error": "not found" }));
                    return;
                };
                let body = req.json();
                let high = body
                    .get("high")
                    .and_then(|v| v.as_u64())
                    .or_else(|| meta.get("memoryHigh").and_then(|v| v.as_u64()))
                    .unwrap_or(crate::memory::HIGH as u64) as usize;
                let model = meta
                    .get("model")
                    .and_then(|v| v.as_str())
                    .filter(|m| !m.trim().is_empty())
                    .unwrap_or(crate::harness::DEFAULT_MODEL);
                let compactor = crate::memory::AglCompactor::new(&st.rt, model);
                let built = crate::memory::build_all(&st.store, id, &compactor);
                crate::memory::fit(&st.store, id, high);
                sink.json(
                    200,
                    &json!({
                        "ok": true,
                        "built": built.to_json(),
                        "high": high,
                        "view": memory_view_json(&st.store, id),
                    }),
                )
            }

            ("GET", path) if matches!(session_route(path), Some((_, "view"))) => {
                let (id, _) = session_route(path).unwrap();
                if st.store.load(id).is_none() {
                    sink.json(404, &json!({ "error": "not found" }));
                    return;
                }
                sink.json(200, &memory_view_json(&st.store, id))
            }

            ("GET", path) if matches!(session_route(path), Some((_, "zoom"))) => {
                let (id, _) = session_route(path).unwrap();
                if st.store.load(id).is_none() {
                    sink.json(404, &json!({ "error": "not found" }));
                    return;
                }
                let num = |k: &str| req.query_param(k).and_then(|v| v.parse::<u64>().ok());
                let (Some(from), Some(n)) = (num("id"), num("n")) else {
                    sink.json(400, &json!({ "error": "id and n are required" }));
                    return;
                };
                match crate::memory::zoom(&st.store, id, from, n) {
                    Ok(crate::memory::Zoom::Halves(parts)) => {
                        let msgs = leaf_messages(&st.store, id);
                        sink.json(
                            200,
                            &json!({
                                "kind": "halves",
                                "ref": format!("{from}+{n}"),
                                "parts": parts.iter()
                                    .map(|p| p.to_json(Some(&msgs))).collect::<Vec<_>>(),
                            }),
                        )
                    }
                    Ok(crate::memory::Zoom::Message { index, node_id }) => {
                        let msgs = leaf_messages(&st.store, id);
                        let row = msgs
                            .iter()
                            .find(|m| m.get("eventId").and_then(|v| v.as_str()) == Some(&node_id))
                            .cloned()
                            .or_else(|| msgs.get(index as usize).cloned())
                            .unwrap_or(Value::Null);
                        sink.json(
                            200,
                            &json!({ "kind": "message", "ref": format!("{from}+{n}"),
                                     "index": index, "nodeId": node_id, "message": row }),
                        )
                    }
                    Err(e) => sink.json(400, &json!({ "error": e })),
                }
            }

            ("POST", path) if matches!(session_route(path), Some((_, "settings"))) => {
                let (id, _) = session_route(path).unwrap();
                if st.store.load(id).is_none() {
                    sink.json(404, &json!({ "error": "not found" }));
                    return;
                }
                let body = req.json();
                let mut patch = Map::new();
                for key in ["purpose", "notes", "routing", "memoryHigh", "title", "workspaceRoot", "worktree", "soulAppendage"] {
                    if let Some(v) = body.get(key) {
                        patch.insert(key.to_string(), v.clone());
                    }
                }
                sink.json(
                    200,
                    &json!({ "ok": true, "session": st.store.update_meta(id, patch) }),
                )
            }

            ("GET", path) if path.starts_with("/api/sessions/") => {
                let id = &path["/api/sessions/".len()..];
                match st.store.load(id) {
                    Some(meta) => {
                        let override_text =
                            meta.get("allowlist").and_then(|v| v.as_str()).map(String::from);
                        let roster: Vec<Value> = roster_json(&st.store, id);
                        sink.json(
                            200,
                            &json!({
                                "session": meta,
                                "messages": reconstruct_messages(&st.store, id),
                                "queue": crate::queue::snapshot(&st.store, id),
                                "participants": roster,
                                "cursor": st.store.cursor(id),
                                "allowlist": override_text.clone().unwrap_or_default(),
                                "allowlistOverridden": override_text.is_some(),
                                "allowlistMode": meta.get("allowlistMode")
                                    .cloned().unwrap_or(json!("allowlist")),
                            }),
                        )
                    }
                    None => sink.json(404, &json!({ "error": "not found" })),
                }
            }

            ("DELETE", path) if path.starts_with("/api/sessions/") => {
                let id = &path["/api/sessions/".len()..];
                let ok = st.store.delete(id);
                st.tabs.forget_session(id);
                st.cancels.forget(id);
                st.approvals.forget(id);
                sink.json(200, &json!({ "ok": ok, "id": id }))
            }

            ("POST", "/api/chat") => {
                let body = req.json();
                let sv = |k: &str| body.get(k).and_then(|v| v.as_str()).map(String::from);
                let content = sv("content").unwrap_or_default();
                if content.trim().is_empty() {
                    sink.json(400, &json!({ "error": "empty content" }));
                    return;
                }
                let mut turn = TurnRequest {
                    tab_id: sv("tabId").unwrap_or_else(|| "1".into()),
                    content,
                    agent: sv("agent"),
                    model: sv("model"),
                    new_session: body
                        .get("newSession")
                        .map(crate::session_store::js_truthy)
                        .unwrap_or(false),
                    session_id: sv("sessionId"),
                    from: sv("from"),
                };
                if !crate::group::mentions(&turn.content).is_empty() {
                    match harness::resolve_session(&st.store, &st.tabs, &turn) {
                        Ok(id) => {
                            if invite_mentions(&st.store, &id, &turn.content) > 0 {
                                seat_changed(st, &id);
                            }
                            turn.session_id = Some(id);
                            turn.new_session = false;
                        }
                        Err(e) => {
                            sink.json(500, &json!({ "error": e.to_string() }));
                            return;
                        }
                    }
                }
                let seated_def = turn
                    .session_id
                    .clone()
                    .or_else(|| st.tabs.get(&turn.tab_id).map(|e| e.session_id))
                    .filter(|_| !turn.new_session)
                    .and_then(|sid| {
                        let agents = st.store.agent_participants(&sid);
                        agents
                            .first()
                            .filter(|_| agents.len() == 1)
                            .and_then(|p| p.agent_name.clone())
                    })
                    .and_then(|n| st.store.agents().get(&n));
                let def = turn
                    .agent
                    .as_deref()
                    .and_then(|n| st.store.agents().get(n))
                    .or(seated_def)
                    .or_else(|| st.store.agents().list().into_iter().next())
                    .unwrap_or_else(agent_loader::default_agent);

                let seated = turn
                    .session_id
                    .clone()
                    .or_else(|| st.tabs.get(&turn.tab_id).map(|e| e.session_id))
                    .filter(|_| !turn.new_session)
                    .map(|sid| st.store.agent_participants(&sid).len())
                    .unwrap_or(0);

                sink.begin_ndjson();
                {
                    let mut emit = |v: &Value| sink.row(v);
                    let _ = if seated != 1 {
                        harness::run_round(
                            st, &st.store, &st.tabs, &st.rt, &st.project_root,
                            &st.mcp, &st.cancels, &st.approvals, &def, turn, &mut emit,
                        )
                    } else {
                        harness::run_turn(
                            st, &st.store, &st.tabs, &st.rt, &st.project_root,
                            &st.mcp, &st.cancels, &st.approvals, &def, turn, &mut emit,
                        )
                    };
                }
                sink.end();
                return;
            }

            ("POST", "/api/allowlist") => {
                let body = req.json();
                let session_id = body
                    .get("sessionId")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .or_else(|| {
                        body.get("tabId")
                            .and_then(|v| v.as_str())
                            .and_then(|t| st.tabs.get(t).map(|e| e.session_id))
                    });
                let Some(sid) = session_id.filter(|s| st.store.load(s).is_some()) else {
                    sink.json(404, &json!({ "error": "session not found" }));
                    return;
                };
                let overridden = body
                    .get("overridden")
                    .map(crate::session_store::js_truthy)
                    .unwrap_or(true);
                let text = body
                    .get("allowlist")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut patch = Map::new();
                if let Some(mode) = body.get("mode").and_then(|v| v.as_str()) {
                    let mode = if mode == "unrestricted" {
                        "unrestricted"
                    } else {
                        "allowlist"
                    };
                    patch.insert("allowlistMode".into(), json!(mode));
                }
                if body.get("allowlist").is_some() && overridden {
                    patch.insert("allowlist".into(), json!(text));
                    patch.insert("allowlistSource".into(), json!("ui"));
                } else if body.get("allowlist").is_some() {
                    patch.insert("allowlist".into(), Value::Null);
                    patch.insert("allowlistSource".into(), json!("agent"));
                }
                st.store.update_meta(&sid, patch);
                sink.json(
                    200,
                    &json!({
                        "ok": true,
                        "sessionId": sid,
                        "allowlistSource": if overridden { "ui" } else { "agent" },
                        "overridden": overridden,
                        "allowlistMode": st.store.load(&sid)
                            .and_then(|s| s.get("allowlistMode").cloned())
                            .unwrap_or(json!("allowlist")),
                        "bytes": text.len(),
                    }),
                )
            }

            ("POST", path) if path.starts_with("/api/participants/") => {
                let Some(pid) = path["/api/participants/".len()..].parse::<i64>().ok() else {
                    sink.json(400, &json!({ "error": "bad participant id" }));
                    return;
                };
                let body = req.json();
                let patch = body.as_object().cloned().unwrap_or_default();
                match st.store.update_participant(pid, &patch) {
                    Some(p) => {
                        seat_changed(st, &p.conversation_id);
                        sink.json(200, &json!({ "ok": true, "participant": p.to_json() }))
                    }
                    None => sink.json(404, &json!({ "error": "not found" })),
                }
            }

            ("DELETE", path) if path.starts_with("/api/participants/") => {
                let Some(pid) = path["/api/participants/".len()..].parse::<i64>().ok() else {
                    sink.json(400, &json!({ "error": "bad participant id" }));
                    return;
                };
                let conv = st.store.participant(pid).map(|p| p.conversation_id);
                let ok = st.store.remove_participant(pid);
                if let Some(conv) = conv {
                    seat_changed(st, &conv);
                }
                sink.json(200, &json!({ "ok": ok, "id": pid }))
            }

            ("POST", path) if path.starts_with("/api/nodes/") && path.ends_with("/visibility") => {
                let node = &path["/api/nodes/".len()..path.len() - "/visibility".len()];
                let body = req.json();
                let visible = body
                    .get("visible")
                    .map(crate::session_store::js_truthy)
                    .unwrap_or(true);
                let ok = st.store.set_visibility(node, visible);
                sink.json(if ok { 200 } else { 404 }, &json!({ "ok": ok, "id": node, "visible": visible }))
            }

            ("POST", path) if path.starts_with("/api/nodes/") && path.ends_with("/goto") => {
                let node = &path["/api/nodes/".len()..path.len() - "/goto".len()];
                let body = req.json();
                let enabled = body
                    .get("enabled")
                    .map(crate::session_store::js_truthy)
                    .unwrap_or(true);
                let label = body.get("label").and_then(|v| v.as_str());
                let ok = st.store.set_goto(node, enabled, label);
                sink.json(if ok { 200 } else { 404 }, &json!({ "ok": ok, "id": node }))
            }

            ("POST", "/api/event/meta") => {
                let body = req.json();
                let ids: Vec<String> = match body.get("eventIds").and_then(|v| v.as_array()) {
                    Some(a) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
                    None => body
                        .get("eventId")
                        .and_then(|v| v.as_str())
                        .map(|s| vec![s.to_string()])
                        .unwrap_or_default(),
                };
                if ids.is_empty() {
                    sink.json(400, &json!({ "error": "eventId or eventIds is required" }));
                    return;
                }
                let mut n = 0;
                for target in &ids {
                    if let Some(v) = body.get("visible") {
                        if st.store.set_visibility(target, crate::session_store::js_truthy(v)) {
                            n += 1;
                        }
                    }
                    if let Some(g) = body.get("goto").and_then(|g| g.as_object()) {
                        st.store.set_goto(
                            target,
                            g.get("enabled").map(crate::session_store::js_truthy).unwrap_or(false),
                            g.get("label").and_then(|v| v.as_str()),
                        );
                    }
                }
                sink.json(200, &json!({ "ok": true, "patched": n, "eventIds": ids }))
            }

            ("POST", "/api/abort") => {
                let body = req.json();
                let session = body
                    .get("sessionId")
                    .or_else(|| body.get("session_id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if session.trim().is_empty() {
                    sink.json(400, &json!({ "error": "sessionId required" }));
                    return;
                }
                let stopped = st.cancels.stop(&session);
                sink.json(200, &json!({ "ok": true, "stopped": stopped, "sessionId": session }))
            }

            ("POST", path) if path.starts_with("/api/sessions/") && path.ends_with("/dequeue") => {
                let rest = &path["/api/sessions/".len()..];
                let session = rest.trim_end_matches('/').trim_end_matches("dequeue").trim_end_matches('/');
                let body = req.json();
                let item = body
                    .get("itemId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if session.is_empty() || item.is_empty() {
                    sink.json(400, &json!({ "ok": false, "error": "sessionId and itemId required" }));
                    return;
                }
                match st.store.queue_remove(session, &item) {
                    Some(row) => {
                        if let Some(send) = &st.queue_sender {
                            send();
                        }
                        sink.json(200, &json!({
                            "ok": true,
                            "sessionId": session,
                            "itemId": item,
                            "text": row.get("text").cloned().unwrap_or(Value::Null),
                        }))
                    }
                    None => sink.json(
                        404,
                        &json!({ "ok": false, "error": "that message is no longer queued" }),
                    ),
                }
            }

            ("POST", "/api/approve") => {
                let body = req.json();
                let id = body
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let decision = crate::approvals::Decision::parse(
                    body.get("decision").and_then(|v| v.as_str()).unwrap_or(""),
                );
                if id.trim().is_empty() {
                    sink.json(400, &json!({ "error": "id required" }));
                    return;
                }
                let ok = st.approvals.resolve(&id, decision);
                sink.json(
                    200,
                    &json!({ "ok": ok, "id": id, "decision": decision.as_str() }),
                )
            }

            ("GET", "/api/pending") => {
                let session = req.query_param("sessionId").unwrap_or_default();
                sink.json(200, &json!({ "pending": st.approvals.pending(&session) }))
            }

            ("GET", "/api/context-window") => sink.json(
                501,
                &json!({ "error": "not implemented in angela-rs yet", "route": p }),
            ),

            _ => sink.json(404, &json!({ "error": "not found", "route": p })),
        }
        sink.end();
    }
}
