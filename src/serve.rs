//! Port of `angela/src/serve.mjs` — the HTTP surface the chat SPA talks to.
//!
//! Routes (method + path), from the JS:
//!
//! ```text
//! POST   /api/chat           NDJSON stream of {type:'card'|'event'|'done'}
//! POST   /api/approve        resolve one approval request
//! POST   /api/abort          stop the live turn on a tab
//! POST   /api/session/new    close the tab's live session
//! POST   /api/allowlist      persist the UI allowlist
//! POST   /api/event/meta     patch event visibility / goto
//! GET    /api/pending        pending approvals
//! GET    /api/health
//! GET    /api/agents
//! GET    /api/context-window
//! GET    /api/sessions
//! DELETE /api/sessions       clean all
//! GET    /api/sessions/:id
//! DELETE /api/sessions/:id
//! ```
//!
//! `chat-server` in `psicom/chat` already proxies `/api/*` here, so this
//! drops in behind the existing proxy with no front-end change.

use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::agent_loader;
use crate::conv_store::ConvStore;
use crate::harness::{self, LiveTabs, McpPool, TurnRequest};
use crate::http::{Handler, Request, Responder, ResponseSink, SocketSink};
use crate::session_store::{EventWrapper, NewSession, SessionStore};

/// Server-wide state.
pub struct AppState {
    pub store: ConvStore,
    pub project_root: String,
    pub port: u16,
    pub auto_approve: bool,
    pub mcp_root: String,
    pub tabs: LiveTabs,
    /// MCP servers, started on demand and shared across turns.
    pub mcp: McpPool,
    /// One runtime for the whole server: the HTTP layer is a thread per
    /// connection, but agl is async, so each turn blocks on this.
    pub rt: tokio::runtime::Runtime,
}

impl AppState {
    pub fn new(project_root: impl AsRef<std::path::Path>, port: u16) -> std::io::Result<Self> {
        let root = project_root.as_ref().to_path_buf();
        let store = ConvStore::new(&root)
            .map_err(|e| std::io::Error::other(format!("conversations.db: {e}")))?;
        // Bring any pre-SQLite sessions in. Idempotent and non-destructive:
        // the `.jsonl` files are read and left alone, so this is also the
        // rollback plan — delete the db and they import again.
        let report = crate::migrate::run(&store, &root);
        if report.imported > 0 {
            eprintln!(
                "angela: imported {} session(s) ({} nodes) into conversations.db",
                report.imported, report.nodes
            );
            // The gate: the tree must render what the files rendered. A
            // mismatch here means the parent chain is wrong, and shipping it
            // would quietly reorder someone's transcript.
            match crate::migrate::verify(&store, &root) {
                Ok(n) => eprintln!("angela: verified {n} imported transcript(s) against the files"),
                Err(e) => eprintln!("angela: MIGRATION VERIFY FAILED — {e}"),
            }
        }
        for f in &report.failed {
            eprintln!("angela: import failed — {f}");
        }
        // Agents moved from `.angela/agents/*.json` into the database. The
        // files are imported once and then neither read nor written — they
        // stay on disk as the rollback, exactly like the session files.
        let imported = store.agents().import_files(&root);
        if imported > 0 {
            eprintln!("angela: imported {imported} agent definition(s) into conversations.db");
        }
        // Number any conversation that predates the `ordinal` field, so the
        // tab strip has a stable 1-based label for every one of them.
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
            mcp: McpPool::default(),
            rt: tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?,
        })
    }
}

/// `cloneTelemetry` — a shallow copy so UI bubbles never share one object.
/// In Rust the clone is implicit; the `null` normalisation is what matters.
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

/// Rebuild the UI transcript from the log. Port of
/// `reconstructMessagesFromLog`.
///
/// Emits `kind: user | assistant | reasoning | tool_call | tool_result` —
/// the five card kinds the chat UI renders.
pub fn reconstruct_messages_from_log(store: &SessionStore, id: &str) -> Vec<Value> {
    reconstruct_from_events(store.list_events_folded(id))
}

/// The same transcript, from the conversation tree's rendered path.
///
/// The two functions share [`reconstruct_from_events`] rather than each
/// having their own copy of the card-shaping rules, which is what lets the
/// migration verifier compare them meaningfully: a difference in the output
/// can only come from a difference in the events, which is exactly what it
/// is checking.
pub fn reconstruct_messages(store: &ConvStore, id: &str) -> Vec<Value> {
    let mut rows = reconstruct_from_events(store.list_events_folded(id));
    // Attribute each row to the seat that produced it.
    //
    // The event wrapper has no author — it never needed one when `assistant`
    // could only mean one agent. The node does, so the handle is joined back
    // on here rather than threaded through the event shape, which the JS-era
    // logs also have to parse. Without this a reloaded group conversation
    // renders three agents as one anonymous voice.
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

/// Shape a folded event list into the five card kinds the UI renders.
pub fn reconstruct_from_events(events: Vec<EventWrapper>) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::new();
    // Which turn each event belongs to, by event id.
    //
    // Collected up front and applied afterwards by `eventId` rather than
    // tracked inside the loop: the loop `continue`s out of most branches, so
    // anything bookkept at the bottom of it is reached by only some rows.
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
    // When each row happened, by event id. Same reason as the turn marks:
    // the loop `continue`s out of most branches, so anything set at the
    // bottom of it reaches only some rows.
    let times: std::collections::HashMap<String, String> = events
        .iter()
        .map(|w| (w.event_id.clone(), w.ts.clone()))
        .collect();
    // Older logs lack `meta.telemetry`, so a trailing gen_info is the
    // fallback for the round it belongs to.
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
        // Prefer telemetry frozen when the event was appended.
        let frozen = wrap.meta.get("telemetry").filter(|t| t.is_object());
        let visible = wrap.meta.get("visible") != Some(&Value::Bool(false));
        let goto = goto_from_meta(&wrap.meta);
        let eid = wrap.event_id.clone();
        let role = po.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let has_content = !matches!(po.get("content"), None | Some(Value::Null));
        let is_snapshot =
            po.get("kind").and_then(|v| v.as_str()) == Some("provider_messages_snapshot");

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

        // --- what the model was given, and what decided it ran -----------
        //
        // These three carry no conversation: they are the turn's own
        // paperwork, kept so an odd reply can be traced to the prompt that
        // produced it. They are never fed back to a model.
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
    // A turn is one inference request. Marking the first and last row of each
    // run is what lets the transcript draw them under a single rail rather
    // than as unrelated rows.
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

/// Build a definition from a request body, over an existing one.
///
/// Absent keys keep what the stored definition had, rather than resetting to
/// the type default: the editor sends whole forms today, but a caller
/// patching one field must not silently blank the other fifteen.
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
        // Normalised, not stored verbatim: it ends up in a `style`
        // attribute, so "whatever was typed" is not a thing to pass through.
        // An unparseable colour clears it back to the app default rather
        // than being kept as a value nothing can render.
        def.accent_color = crate::agent_loader::normalize_hex(&v);
    }
    if let Some(v) = st("cwd") { def.cwd = Some(v).filter(|c| !c.trim().is_empty()); }
    match o.get("contextWindow").or_else(|| o.get("context_window")) {
        Some(Value::Null) => def.context_window = None,
        Some(v) => {
            if let Some(n) = v.as_u64() { def.context_window = Some(n); }
            // A blank field in a form arrives as "", which means "unset".
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
        // Explicit null, or an empty list, means every tool — which is NOT
        // the same as "no tools".
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

/// Split `/api/sessions/<id>/<rest>` into its id and sub-path.
///
/// The bare `/api/sessions/<id>` form returns an empty `rest`, so one
/// matcher handles both and the sub-routes cannot be shadowed by the
/// catch-all the way a `starts_with` chain would shadow them.
fn session_route(path: &str) -> Option<(&str, &str)> {
    let rest = path.strip_prefix("/api/sessions/")?;
    match rest.split_once('/') {
        Some((id, tail)) => Some((id, tail)),
        None => Some((rest, "")),
    }
}

/// The roster, each seat carrying its agent's accent colour.
pub fn roster_json(store: &ConvStore, conv: &str) -> Vec<Value> {
    let agents = store.agents();
    store
        .participants(conv)
        .iter()
        .map(|p| {
            let accent = p
                .agent_name
                .as_deref()
                .and_then(|n| agents.get(n))
                .and_then(|d| d.accent_color);
            p.to_json_with(accent.as_deref())
        })
        .collect()
}

/// The transcript rows that are the memory tree's leaves, in leaf order.
///
/// Not the same list as [`reconstruct_messages`]: the tree's leaves are the
/// user/assistant messages, while a transcript also carries reasoning and
/// tool rows. Looking a leaf up by its position in the raw transcript puts
/// every leaf after the first reasoning row against the wrong message — so
/// the lookup is by **node id**, and the result is aligned to leaf index.
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

/// The memory tree's view as the UI reads it: one entry per thread or
/// message, oldest first.
fn memory_view_json(store: &ConvStore, id: &str) -> Value {
    let msgs = leaf_messages(store, id);
    let parts = crate::memory::view(store, id);
    let count = crate::memory::count(store, id);
    json!({
        "sessionId": id,
        "count": count,
        // True once anything has been merged: the UI shows the "compact"
        // affordance differently for a conversation that is still flat.
        "compacted": parts.iter().any(|p| p.coord.width() > 1),
        "parts": parts.iter().map(|p| p.to_json(Some(&msgs))).collect::<Vec<_>>(),
        "roots": crate::memory::roots(count)
            .iter()
            .map(|c| c.reference())
            .collect::<Vec<_>>(),
    })
}

/// The route table.
pub struct Server {
    pub state: Arc<AppState>,
}

impl Server {
    pub fn new(state: Arc<AppState>) -> Self {
        Server { state }
    }

    /// Test hook: the route body without going through HTTP.
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
                // `title || agent || id` — the JS fallback chain.
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
                })
            })
            .collect();
        json!({ "projectRoot": self.state.project_root, "sessions": sessions })
    }
}

/// The standalone socket server delegates to the same dispatch the embedded
/// one uses, so a route can never behave differently depending on who hosts it.
impl Handler for Server {
    fn handle(&self, req: &Request, out: Responder<'_>) {
        let mut sink = SocketSink::new(out);
        self.dispatch(req, &mut sink);
    }
}

impl Server {
    /// Route one request, writing through `sink`.
    ///
    /// Transport-agnostic on purpose: `angela-serve` owns its socket, while
    /// chat-server in psicom/chat drives this from its own response stream.
    /// One router, two hosts.
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
                    // Agent loading is not ported yet; the JS also returns []
                    // when no agents are defined for the project.
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

            // ---- agents ----------------------------------------------------
            //
            // Definitions live in the database, so these are the only way to
            // read or change one; there is no file to edit any more.
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
                // Over the stored definition, so a partial body patches
                // rather than blanks.
                let def = agent_from_body(&body, Some(existing));
                match agents.save(&def, Some(name)) {
                    Ok(a) => sink.json(200, &json!({ "ok": true, "agent": a.to_json() })),
                    Err(e) => sink.json(400, &json!({ "error": e.to_string() })),
                }
            }

            ("DELETE", path) if path.starts_with("/api/agents/") => {
                let name = &path["/api/agents/".len()..];
                match st.store.agents().delete(name) {
                    // `seated` is how many conversations had it: they keep
                    // their transcripts, so the UI can say so plainly.
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
                // The JS closes the tab's live session; with no live-tab map
                // yet, creating the session record is the honest equivalent
                // and is what the UI's "+" needs.
                let body = req.json();
                let agent = body.get("agent").and_then(|v| v.as_str()).map(String::from);
                let model = body.get("model").and_then(|v| v.as_str()).map(String::from);
                match st.store.create(NewSession {
                    agent,
                    model,
                    title: body
                        .get("title")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    ..Default::default()
                }) {
                    Ok(s) => sink.json(200, &json!({ "ok": true, "session": s })),
                    Err(e) => sink.json(500, &json!({ "error": e.to_string() })),
                }
            }

            // ---- the conversation tree ------------------------------------
            //
            // Moving the cursor IS the fork primitive: point it at any node,
            // then POST /api/chat, and the new message becomes that node's
            // child. Nothing is rewritten, so the branch you came from is
            // still there and still reachable by pointing back at it.
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
                let ok = st.store.set_cursor(id, target);
                sink.json(
                    if ok { 200 } else { 400 },
                    &json!({ "ok": ok, "sessionId": id, "cursor": st.store.cursor(id),
                             "messages": if ok { reconstruct_messages(&st.store, id) } else { vec![] } }),
                )
            }

            // The whole forest, not just the rendered path — this is what a
            // branch picker draws.
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

            // Seat another agent. Two or more active agents is what switches
            // /api/chat onto the group path.
            ("POST", path) if matches!(session_route(path), Some((_, "participants"))) => {
                let (id, _) = session_route(path).unwrap();
                if st.store.load(id).is_none() {
                    sink.json(404, &json!({ "error": "not found" }));
                    return;
                }
                let body = req.json();
                let gs = |k: &str| body.get(k).and_then(|v| v.as_str());
                let kind = gs("kind").unwrap_or("agent");
                let agent_name = gs("agentName").or_else(|| gs("agent"));
                // An agent definition supplies the system prompt and model
                // unless the request overrides them, so seating a configured
                // agent takes one field.
                let def = agent_name.and_then(|n| st.store.agents().get(n));
                // Soul + goals + memory, composed once, so a seated agent
                // carries the same prompt the editor shows.
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
                    Some(p) => sink.json(200, &json!({ "ok": true, "participant": p.to_json() })),
                    None => sink.json(400, &json!({ "error": "could not seat participant" })),
                }
            }

            // ---- the memory tree ------------------------------------------
            //
            // Run compaction: build every summary the tree is missing, then
            // fit the view to the budget. `high` lets the caller compact a
            // short conversation on purpose, which is the only way to see
            // threading before a chat is genuinely large.
            ("POST", path) if matches!(session_route(path), Some((_, "compact"))) => {
                let (id, _) = session_route(path).unwrap();
                let Some(meta) = st.store.load(id) else {
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

            // Open one thread, one level. Lazy by design: a client walks down
            // only the branch someone actually asked about.
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
                        // The bottom of the tree: the original message, whole.
                        // Aligned to leaf index, so the fallback below lands
                        // on the same row the node id names.
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

            // Room framing: purpose and notes are the shared header every
            // participant's prompt carries; routing picks everyone-vs-lead.
            ("POST", path) if matches!(session_route(path), Some((_, "settings"))) => {
                let (id, _) = session_route(path).unwrap();
                if st.store.load(id).is_none() {
                    sink.json(404, &json!({ "error": "not found" }));
                    return;
                }
                let body = req.json();
                let mut patch = Map::new();
                for key in ["purpose", "notes", "routing", "memoryHigh", "title"] {
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
                        // The UI seeds its textarea from here: the session's
                        // override when present, else the agent default.
                        let agent_default = meta
                            .get("agent")
                            .and_then(|v| v.as_str())
                            .and_then(|n| st.store.agents().get(n))
                            .or_else(|| st.store.agents().list().into_iter().next())
                            .map(|d| d.allowlist)
                            .unwrap_or_default();
                        let override_text =
                            meta.get("allowlist").and_then(|v| v.as_str()).map(String::from);
                        let roster: Vec<Value> = roster_json(&st.store, id);
                        sink.json(
                            200,
                            &json!({
                                "session": meta,
                                "messages": reconstruct_messages(&st.store, id),
                                // The roster and the cursor are what the
                                // multi-participant UI draws from; a
                                // two-seat chat simply has two entries.
                                "participants": roster,
                                "cursor": st.store.cursor(id),
                                "allowlist": override_text.clone().unwrap_or(agent_default),
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
                // The JS drops any live tab pointing at the deleted session.
                st.tabs.forget_session(id);
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
                let turn = TurnRequest {
                    tab_id: sv("tabId").unwrap_or_else(|| "1".into()),
                    content,
                    agent: sv("agent"),
                    model: sv("model"),
                    new_session: body
                        .get("newSession")
                        .map(crate::session_store::js_truthy)
                        .unwrap_or(false),
                    session_id: sv("sessionId"),
                };
                // Resolve the agent: the request's name, else the project's
                // single agent, else the built-in default.
                // Which definition runs the turn. The **seated** agent comes
                // before the project's first-by-name one: since a
                // conversation starts empty and agents join by invitation,
                // the roster is the answer to "who is in this chat", and
                // falling back to list order would run `@scout` with
                // whichever definition happens to sort first.
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

                // How many agents are seated decides the path. Exactly one:
                // the provider-side history IS the transcript, which is
                // cheaper and is what every existing session already holds.
                // Two or more: no shared history can exist (see `group.rs`),
                // so each runs in its own context with the room flattened
                // into text. None: nothing answers, and `run_round` says so.
                let seated = turn
                    .session_id
                    .clone()
                    .or_else(|| st.tabs.get(&turn.tab_id).map(|e| e.session_id))
                    .filter(|_| !turn.new_session)
                    .map(|sid| st.store.agent_participants(&sid).len())
                    // A conversation that does not exist yet has nobody in
                    // it. `run_round` is the path that handles an empty
                    // roster: it stores the message and says so.
                    .unwrap_or(0);

                sink.begin_ndjson();
                {
                    let mut emit = |v: &Value| sink.row(v);
                    let _ = if seated != 1 {
                        harness::run_round(
                            &st.store, &st.tabs, &st.rt, &st.project_root,
                            &st.mcp, &def, turn, &mut emit,
                        )
                    } else {
                        harness::run_turn(
                            &st.store, &st.tabs, &st.rt, &st.project_root,
                            &st.mcp, &def, turn, &mut emit,
                        )
                    };
                }
                sink.end();
                return;
            }

            // Persist the UI's allowlist onto the session (textarea blur).
            //
            // `overridden: false` clears the override so the agent default
            // takes over again — the UI's "revert" — rather than storing an
            // empty string, which would mean "deny everything".
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
                // `mode` is independent of the rules: setting unrestricted
                // must not touch `allowlist`, or the rules are lost.
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
                    Some(p) => sink.json(200, &json!({ "ok": true, "participant": p.to_json() })),
                    None => sink.json(404, &json!({ "error": "not found" })),
                }
            }

            ("DELETE", path) if path.starts_with("/api/participants/") => {
                let Some(pid) = path["/api/participants/".len()..].parse::<i64>().ok() else {
                    sink.json(400, &json!({ "error": "bad participant id" }));
                    return;
                };
                // An agent that has spoken is deactivated, not deleted: its
                // nodes are in the tree and must keep their author.
                let ok = st.store.remove_participant(pid);
                sink.json(200, &json!({ "ok": ok, "id": pid }))
            }

            // Visibility and go-to: a column write now, where the JSONL
            // format needed a whole appended `context_meta` line that every
            // reader then had to fold forward.
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

            // The old append-a-patch route, now backed by the columns. Kept
            // at its original path so an older client keeps working.
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

            // Not ported yet — answered explicitly so the UI sees a clear
            // signal rather than a 404 it would read as a routing bug.
            ("POST", "/api/approve")
            | ("POST", "/api/abort")
            | ("GET", "/api/pending")
            | ("GET", "/api/context-window") => sink.json(
                501,
                &json!({ "error": "not implemented in angela-rs yet", "route": p }),
            ),

            _ => sink.json(404, &json!({ "error": "not found", "route": p })),
        }
        sink.end();
    }
}

#[cfg(test)]
mod transcript_tests {
    use super::*;
    use crate::conv_store::ConvStore;
    use crate::harness::TurnMark;
    use crate::session_store::{NewEvent, NewSession};

    fn conv() -> (ConvStore, String) {
        let s = ConvStore::memory("/tmp/p").unwrap();
        let id = s.create(NewSession::default()).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        (s, id)
    }

    #[test]
    fn a_turn_is_marked_from_its_first_row_to_its_last() {
        let (s, id) = conv();
        s.append_event(
            &id,
            NewEvent::new("context_window", json!({"role":"user","content":"q"})),
        )
        .unwrap();

        let a = TurnMark::new("alice", None, 0);
        for ev in [
            a.ev("system", json!({ "content": "You are Alice." })),
            a.ev("agent_prompt", json!({ "content": "q, flattened" })),
            a.ev("tool_call", json!({ "name": "read", "id": "c1" })),
            a.ev("tool_result", json!({ "name": "read", "text": "ok", "ok": true })),
            a.ev("context_window", json!({"role":"assistant","content":"A"})),
        ] {
            s.append_event(&id, ev).unwrap();
        }
        let b = TurnMark::new("bob", None, 0);
        s.append_event(&id, b.ev("context_window", json!({"role":"assistant","content":"B"})))
            .unwrap();

        let rows = reconstruct_messages(&s, &id);
        let kinds: Vec<&str> = rows.iter().map(|r| r["kind"].as_str().unwrap()).collect();
        assert_eq!(
            kinds,
            vec![
                "user",
                "system_prompt",
                "agent_prompt",
                "tool_call",
                "tool_result",
                "assistant",
                "assistant"
            ]
        );

        // The user's own message belongs to no turn: it is what they typed,
        // not part of anyone's inference request.
        assert!(rows[0].get("turn").is_none());

        // Alice's five rows are one run, marked at both ends.
        let alice: Vec<&Value> = rows[1..6].iter().collect();
        assert!(alice.iter().all(|r| r["turn"] == json!(a.id)));
        assert_eq!(alice[0]["turnFirst"], json!(true));
        assert_eq!(alice[0]["turnLast"], json!(false));
        assert_eq!(alice[4]["turnFirst"], json!(false));
        assert_eq!(alice[4]["turnLast"], json!(true));

        // Bob's single row is its own run: first AND last.
        assert_eq!(rows[6]["turn"], json!(b.id));
        assert_eq!(rows[6]["turnFirst"], json!(true));
        assert_eq!(rows[6]["turnLast"], json!(true));
    }

    #[test]
    fn the_prompt_the_agent_got_is_kept_apart_from_what_the_user_typed() {
        let (s, id) = conv();
        s.append_event(
            &id,
            NewEvent::new("context_window", json!({"role":"user","content":"hi"})),
        )
        .unwrap();
        let m = TurnMark::new("alice", None, 2);
        s.append_event(
            &id,
            m.ev(
                "agent_prompt",
                json!({ "content": "[Conversation] …hi…", "model": "m", "historyTurns": 3 }),
            ),
        )
        .unwrap();

        let rows = reconstruct_messages(&s, &id);
        assert_eq!(rows[0]["text"], "hi");
        // Two different things. Conflating them is exactly what makes "why
        // did it answer that?" unanswerable.
        assert_eq!(rows[1]["kind"], "agent_prompt");
        assert_eq!(rows[1]["text"], "[Conversation] …hi…");
        assert_eq!(rows[1]["historyTurns"], 3);
        assert_eq!(rows[1]["round"], 2);
    }

    #[test]
    fn a_decision_survives_as_a_row_with_what_it_was_asked() {
        let (s, id) = conv();
        let m = TurnMark::new("fish", None, -1);
        s.append_event(
            &id,
            m.ev(
                "decision",
                json!({
                    "speak": false, "confidence": 0.8, "error": Value::Null,
                    "state": { "you_are": "@fish — a cook" },
                    "answers": { "should_reply": { "choice": "no" } },
                }),
            ),
        )
        .unwrap();
        let rows = reconstruct_messages(&s, &id);
        assert_eq!(rows[0]["kind"], "decision");
        assert_eq!(rows[0]["speak"], false);
        assert_eq!(rows[0]["confidence"], 0.8);
        // The state is the only place the description it judged on appears.
        assert_eq!(rows[0]["verdict"]["state"]["you_are"], "@fish — a cook");
    }

    #[test]
    fn rows_from_before_turn_marking_simply_have_no_turn() {
        let (s, id) = conv();
        for ev in [
            NewEvent::new("context_window", json!({"role":"user","content":"q"})),
            NewEvent::new("reasoning", json!({"text":"hmm"})),
            NewEvent::new("context_window", json!({"role":"assistant","content":"a"})),
        ] {
            s.append_event(&id, ev).unwrap();
        }
        let rows = reconstruct_messages(&s, &id);
        assert_eq!(rows.len(), 3);
        // An older transcript renders as it always did, with no rail.
        assert!(rows.iter().all(|r| r.get("turn").is_none()));
    }
}
