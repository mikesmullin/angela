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
use crate::harness::{self, LiveTabs, McpPool, TurnRequest};
use crate::http::{Handler, Request, Responder, ResponseSink, SocketSink};
use crate::session_store::{NewSession, SessionStore};

/// Server-wide state.
pub struct AppState {
    pub store: SessionStore,
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
        let store = SessionStore::new(&root)?;
        // Number any sessions that predate the `ordinal` field, so the tab
        // strip has a stable 1-based label for every one of them.
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
    let events = store.list_events_folded(id);
    let mut messages: Vec<Value> = Vec::new();
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
    messages
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
                    "agents": agent_loader::list(std::path::Path::new(&st.project_root))
                        .iter()
                        .map(|a| json!(a.name))
                        .collect::<Vec<_>>(),
                    "mcpRoot": st.mcp_root,
                }),
            ),

            ("GET", "/api/agents") => {
                let root = std::path::Path::new(&st.project_root);
                let agents: Vec<Value> =
                    agent_loader::list(root).iter().map(|a| a.to_json()).collect();
                sink.json(200, &json!({ "projectRoot": st.project_root, "agents": agents }))
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

            ("GET", path) if path.starts_with("/api/sessions/") => {
                let id = &path["/api/sessions/".len()..];
                match st.store.load(id) {
                    Some(meta) => {
                        // The UI seeds its textarea from here: the session's
                        // override when present, else the agent default.
                        let root = std::path::Path::new(&st.project_root);
                        let agent_default = meta
                            .get("agent")
                            .and_then(|v| v.as_str())
                            .and_then(|n| agent_loader::load(root, n))
                            .or_else(|| agent_loader::list(root).into_iter().next())
                            .map(|d| d.allowlist)
                            .unwrap_or_default();
                        let override_text =
                            meta.get("allowlist").and_then(|v| v.as_str()).map(String::from);
                        sink.json(
                            200,
                            &json!({
                                "session": meta,
                                "messages": reconstruct_messages_from_log(&st.store, id),
                                "allowlist": override_text.clone().unwrap_or(agent_default),
                                "allowlistOverridden": override_text.is_some(),
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
                let root = std::path::Path::new(&st.project_root);
                let def = turn
                    .agent
                    .as_deref()
                    .and_then(|n| agent_loader::load(root, n))
                    .or_else(|| {
                        let all = agent_loader::list(root);
                        all.into_iter().next()
                    })
                    .unwrap_or_else(agent_loader::default_agent);

                sink.begin_ndjson();
                {
                    let mut emit = |v: &Value| sink.row(v);
                    let _ = harness::run_turn(
                        &st.store,
                        &st.tabs,
                        &st.rt,
                        &st.project_root,
                        &st.mcp,
                        &def,
                        turn,
                        &mut emit,
                    );
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
                if overridden {
                    patch.insert("allowlist".into(), json!(text));
                    patch.insert("allowlistSource".into(), json!("ui"));
                } else {
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
                        "bytes": text.len(),
                    }),
                )
            }

            // Not ported yet — answered explicitly so the UI sees a clear
            // signal rather than a 404 it would read as a routing bug.
            ("POST", "/api/approve")
            | ("POST", "/api/abort")
            | ("POST", "/api/event/meta")
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
