use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agl::agent::{Agent, Factory};
use agl::agent::TurnEvent;
use agl::provider::{Channel, Message};
use serde_json::{json, Value};

use crate::agent_loader::AgentDef;
use crate::allowlist;
use crate::mcp::McpServer;
use crate::mcp_presets;
use crate::approvals::Approvals;
use crate::cancel::{self, Cancel, Cancels, Ran};
use crate::conv_store::ConvStore;
use crate::session_store::{ContextItem, NewEvent, NewSession};

const TOOL_TIMEOUT: Duration = Duration::from_secs(120);
const VIDEO_TRANSCRIPT_TIMEOUT: Duration = Duration::from_secs(2 * 60 * 60 + 60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Debug, Clone, Default)]
pub struct TabEntry {
    pub session_id: String,
    pub agent: Option<String>,
    pub model: String,
    pub last_prompt_tokens: Option<u64>,
    pub last_total_tokens: Option<u64>,
}

#[derive(Default)]
pub struct LiveTabs {
    map: Mutex<HashMap<String, TabEntry>>,
}

impl LiveTabs {
    pub fn get(&self, tab_id: &str) -> Option<TabEntry> {
        self.map.lock().unwrap().get(tab_id).cloned()
    }

    pub fn set(&self, tab_id: &str, entry: TabEntry) {
        self.map.lock().unwrap().insert(tab_id.to_string(), entry);
    }

    pub fn clear(&self, tab_id: &str) -> Option<TabEntry> {
        self.map.lock().unwrap().remove(tab_id)
    }

    pub fn forget_session(&self, session_id: &str) {
        self.map
            .lock()
            .unwrap()
            .retain(|_, e| e.session_id != session_id);
    }
}

pub const STOPPED_NOTICE: &str = "Stopped.";

struct Armed<'a> {
    cancels: &'a Cancels,
    session: String,
    switch: Arc<Cancel>,
}

impl<'a> Armed<'a> {
    fn new(cancels: &'a Cancels, session: &str) -> Self {
        Armed {
            cancels,
            session: session.to_string(),
            switch: cancels.begin(session),
        }
    }

    fn stopped(&self) -> bool {
        self.switch.stopped()
    }
}

impl Drop for Armed<'_> {
    fn drop(&mut self) {
        self.cancels.finish(&self.session, &self.switch);
    }
}

#[derive(Default)]
pub struct McpPool {
    servers: Mutex<HashMap<String, Result<Arc<McpServer>, String>>>,
}

impl McpPool {
    fn get_or_connect<F>(&self, name: &str, stop: &Cancel, connect: F) -> Result<Arc<McpServer>, String>
    where F: FnOnce() -> Result<Arc<McpServer>, String> {
        let mut servers = loop {
            if stop.stopped() { return Err("stopped".into()); }
            match self.servers.try_lock() {
                Ok(guard) => break guard,
                Err(std::sync::TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(50)),
                Err(std::sync::TryLockError::Poisoned(e)) => break e.into_inner(),
            }
        };
        if let Some(result) = servers.get(name) {
            if result.as_ref().map(|server| server.is_running()).unwrap_or(true) {
                return result.clone();
            }
            servers.remove(name);
        }
        let result = connect();
        if !stop.stopped() { servers.insert(name.to_string(), result.clone()); }
        result
    }
    pub fn ensure(
        &self,
        names: &[String],
        file_root: &std::path::Path,
    ) -> (Vec<Arc<McpServer>>, Vec<String>) {
        self.ensure_cancellable(names, file_root, &Cancel::never())
    }

    fn ensure_cancellable(&self, names: &[String], file_root: &std::path::Path, stop: &Cancel) -> (Vec<Arc<McpServer>>, Vec<String>) {
        let mut out = Vec::new();
        let mut errors = Vec::new();
        for name in names {
            if stop.stopped() { break; }
            let result = self.get_or_connect(
                &format!("{name}::{}", file_root.display()),
                stop, || {
                    let spec = mcp_presets::preset(name, file_root)
                        .ok_or_else(|| format!("unknown or uninstalled MCP preset `{name}`"))?;
                    McpServer::connect_cancellable(&spec, CONNECT_TIMEOUT, stop)
                        .map_err(|e| format!("MCP `{name}`: {e}"))
                },
            );
            match result {
                Ok(srv) => out.push(srv),
                Err(e) => errors.push(e),
            }
        }
        (out, errors)
    }

    pub fn ensure_in_sandbox(
        &self,
        names: &[String],
        file_root: &std::path::Path,
        sandbox: &str,
    ) -> (Vec<Arc<McpServer>>, Vec<String>) {
        self.ensure_in_sandbox_cancellable(names, file_root, sandbox, &Cancel::never())
    }

    fn ensure_in_sandbox_cancellable(&self, names: &[String], file_root: &std::path::Path, sandbox: &str, stop: &Cancel) -> (Vec<Arc<McpServer>>, Vec<String>) {
        if sandbox.trim().is_empty() {
            return self.ensure_cancellable(names, file_root, stop);
        }
        let kept: Vec<String> = names
            .iter()
            .filter(|n| !DISPLACED_BY_SANDBOX.contains(&n.as_str()))
            .cloned()
            .collect();
        let (mut out, mut errors) = (Vec::new(), Vec::new());
        for name in &kept {
            if stop.stopped() { break; }
            let result = self.get_or_connect(
                &format!("{name}:sandbox:{sandbox}"),
                stop,
                || {
                    let guest = crate::mcp_presets::preset_sandbox(name, file_root)
                        .or_else(|| crate::mcp_presets::preset(name, file_root))
                        .ok_or_else(|| format!("unknown or uninstalled MCP preset `{name}`"))?;
                    let spec = crate::sandbox::exec_spec(sandbox, &guest)?;
                    McpServer::connect_cancellable(&spec, CONNECT_TIMEOUT, stop)
                        .map_err(|e| format!("MCP `{name}` (in sandbox {sandbox}): {e}"))
                },
            );
            match result {
                Ok(srv) => out.push(srv),
                Err(e) => errors.push(e),
            }
        }
        if stop.stopped() { return (out, errors); }

        let url = sandbox_rpc_url(sandbox);
        match self.get_or_connect(&format!("sandbox:{sandbox}"), stop, || {
            crate::mcp::connect_http_cancellable(SANDBOX_SERVER, &url, CONNECT_TIMEOUT, stop)
                .map_err(|e| format!("sandbox `{sandbox}`: {e}"))
        }) {
            Ok(srv) => out.push(srv),
            Err(e) => errors.push(format!(
                "{e} — this conversation is assigned to that sandbox, so its file \
                 and shell tools are unavailable. They have NOT fallen back to this \
                 machine."
            )),
        }
        (out, errors)
    }

    pub fn shutdown(&self) {
        for (_, s) in self.servers.lock().unwrap().drain() {
            if let Ok(s) = s { s.close(); }
        }
    }
}

fn file_root_for(store: &ConvStore, session_id: &str, project_root: &std::path::Path) -> std::path::PathBuf {
    let root = store
        .effective_session(session_id)
        .and_then(|m| m.get("effective").cloned())
        .and_then(|m| m.get("workspaceRoot").and_then(Value::as_str).map(String::from));
    match root {
        Some(r) if !r.trim().is_empty() => {
            let p = std::path::PathBuf::from(&r);
            if p.is_absolute() && p.is_dir() {
                return p;
            }
            project_root.to_path_buf()
        }
        _ => project_root.to_path_buf(),
    }
}

pub const DISPLACED_BY_SANDBOX: &[&str] = &["file-io", "term-local"];

pub const SANDBOX_SERVER: &str = "sandbox";

pub const SANDBOX_MOUNT: &str = "/work";

fn sandbox_rpc_url(sandbox: &str) -> String {
    let host = std::env::var("SBOX_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port = std::env::var("SBOX_PORT").unwrap_or_else(|_| "3000".into());
    format!("http://{host}:{port}/api/orchestration/sandboxes/{sandbox}/exec/rpc")
}

pub const DEFAULT_MODEL: &str = "llama-server:default";

pub struct TurnRequest {
    pub tab_id: String,
    pub content: String,
    pub agent: Option<String>,
    pub model: Option<String>,
    pub new_session: bool,
    pub session_id: Option<String>,
    pub from: Option<String>,
}

#[derive(Debug)]
pub enum TurnError {
    EmptyContent,
    Store(String),
    Agent(String),
    Stopped,
}

impl std::fmt::Display for TurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TurnError::EmptyContent => write!(f, "empty content"),
            TurnError::Store(m) => write!(f, "session store: {m}"),
            TurnError::Agent(m) => write!(f, "{m}"),
            TurnError::Stopped => write!(f, "stopped"),
        }
    }
}

fn item_to_message(item: &ContextItem) -> Option<Message> {
    match item.role.as_str() {
        "user" | "assistant" if item.tool_calls.is_none() => Some(Message::text(
            &item.role,
            item.content.clone(),
        )),
        "assistant" => {
            let calls = serde_json::from_value(item.tool_calls.clone()?).ok()?;
            Some(Message {
                role: "assistant".into(),
                content: Some(Value::String(item.content.clone())),
                tool_calls: Some(calls),
                tool_call_id: None,
                reasoning_content: None,
            })
        }
        "tool" => Some(Message::tool_result(
            item.tool_call_id.clone()?,
            item.content.clone(),
        )),
        _ => None,
    }
}

pub fn resolve_session(
    store: &ConvStore,
    tabs: &LiveTabs,
    req: &TurnRequest,
) -> Result<String, TurnError> {
    if !req.new_session {
        if let Some(sid) = req.session_id.as_deref().filter(|s| !s.is_empty()) {
            if store.load(sid).is_some() {
                return Ok(sid.to_string());
            }
        }
        if let Some(entry) = tabs.get(&req.tab_id) {
            if store.load(&entry.session_id).is_some() {
                return Ok(entry.session_id);
            }
        }
    }
    let state = store
        .create(NewSession {
            agent: req.agent.clone(),
            model: Some(req.model.clone().unwrap_or_else(|| DEFAULT_MODEL.into())),
            ..Default::default()
        })
        .map_err(|e| TurnError::Store(e.to_string()))?;
    Ok(state["id"].as_str().unwrap_or_default().to_string())
}

pub fn title_from_first_message(store: &ConvStore, session_id: &str, content: &str) {
    let Some(state) = store.load(session_id) else {
        return;
    };
    let has_title = state
        .get("title")
        .map(crate::session_store::js_truthy)
        .unwrap_or(false);
    if has_title {
        return;
    }
    let chars: Vec<char> = content.chars().collect();
    let title = if chars.len() > 64 {
        format!("{}…", chars[..64].iter().collect::<String>())
    } else {
        content.to_string()
    };
    let mut patch = serde_json::Map::new();
    patch.insert("title".into(), json!(title));
    store.update_meta(session_id, patch);
}

pub fn register_tools(
    agent: &mut Agent,
    servers: &[Arc<McpServer>],
    enabled: Option<&Vec<String>>,
    allow_text: &str,
    policy_mode: &str,
    session: &str,
    approvals: Arc<Approvals>,
    stop: Arc<Cancel>,
) -> usize {
    let enforce = !policy_mode.eq_ignore_ascii_case("auto");
    let allow_text = allow_text.to_string();
    let mut n = 0;
    for srv in servers {
        for tool in &srv.tools {
            let name = mcp_presets::prefixed_tool_name(&srv.name, &tool.name);
            if let Some(list) = enabled {
                if !list.iter().any(|e| e == &name || e == &tool.name) {
                    continue;
                }
            }
            let (props, required) = crate::mcp::schema_parts(&tool.input_schema);
            let srv = Arc::clone(srv);
            let remote = tool.name.clone();
            let gate_name = name.clone();
            let allow_text = allow_text.clone();
            let approvals = Arc::clone(&approvals);
            let stop = Arc::clone(&stop);
            let session = session.to_string();
            let ok = agent.tool(
                &name,
                &tool.description,
                props,
                required,
                move |args: &Value| {
                    if stop.stopped() { return Ok("stopped".into()); }
                    if enforce {
                        let cmd = allowlist::command_line(&gate_name, args);
                        if !allowlist::is_allowed(&allow_text, &gate_name, &cmd)
                            && !approvals.pinned(&session, &gate_name, &cmd)
                        {
                            let decision = approvals.ask_cancellable(&session, &gate_name, &cmd, args, &stop);
                            if !decision.allowed() {
                                return Ok(format!(
                                    "denied: `{gate_name}`{} was not approved by the user. \
                                     Do not try it again — use a permitted tool, or explain \
                                     what you needed it for.",
                                    if cmd.is_empty() {
                                        String::new()
                                    } else {
                                        format!(" with `{cmd}`")
                                    }
                                ));
                            }
                        }
                    }
                    if stop.stopped() { return Ok("stopped".into()); }
                    let timeout = if gate_name == "video_transcribe__transcript" {
                        VIDEO_TRANSCRIPT_TIMEOUT
                    } else {
                        TOOL_TIMEOUT
                    };
                    match srv.call_tool_cancellable(&remote, args, timeout, &stop) {
                        Ok(text) => Ok(text),
                        Err(e) => Ok(format!("error: {e}")),
                    }
                },
            );
            if ok.is_ok() {
                n += 1;
            }
        }
    }
    n
}

fn turn_reasoning(def: &AgentDef, session: Option<&Value>) -> Option<String> {
    Some(def.reasoning_effort.clone()).filter(|s| !s.trim().is_empty())
        .or_else(|| session.and_then(|s| s.get("config")).and_then(|c| c.get("reasoningEffort"))
            .and_then(Value::as_str).filter(|s| !s.trim().is_empty()).map(String::from))
}

fn combined_allowlist(agent: &str, session: Option<&str>) -> String {
    [agent, session.unwrap_or("")].into_iter()
        .filter(|s| !s.trim().is_empty())
        .collect::<Vec<_>>().join("\n")
}

pub fn build_agent(
    store: &ConvStore,
    session_id: &str,
    model: &str,
    system_prompt: Option<String>,
) -> Result<Agent, TurnError> {
    let agent = Agent::factory(Factory {
        model: Some(model.to_string()),
        system_prompt,
        retain_history: Some(true),
        parallel_tools: Some(true),
        ..Default::default()
    })
    .map_err(|e| TurnError::Agent(e.to_string()))?;

    for item in store.build_context_window_items(session_id) {
        if !item.visible {
            continue;
        }
        if let Some(msg) = item_to_message(&item) {
            agent.push_context(crate::media::with_images(&store.project_root, msg));
        }
    }
    Ok(agent)
}

#[derive(Debug, Clone)]
pub struct TurnMark {
    pub id: String,
    pub handle: String,
    pub participant: Option<i64>,
    pub round: i64,
}

#[derive(Default)]
struct Said {
    reasoning: String,
    content: String,
    participant: Option<i64>,
}

impl Said {
    fn new(participant: Option<i64>) -> Self {
        Said { participant, ..Default::default() }
    }

    fn card(&self, kind: &str, text: &str, event_id: &str, mark: &TurnMark) -> Value {
        let mut card = json!({
            "kind": kind, "text": text, "eventId": event_id,
            "turn": mark.id, "handle": mark.handle, "visible": true,
        });
        if let Some(pid) = self.participant {
            card["participantId"] = json!(pid);
        }
        json!({ "type": "card", "card": card })
    }

    fn flush_reasoning(&mut self, store: &ConvStore, conv: &str, mark: &TurnMark, emit: &mut dyn FnMut(&Value) -> bool) {
        let text = std::mem::take(&mut self.reasoning);
        if text.trim().is_empty() {
            return;
        }
        if let Ok(w) = store.append_event(conv, mark.ev("reasoning", json!({ "text": text }))) {
            let card = self.card("reasoning", &text, &w.event_id, mark);
            emit(&card);
        }
    }

    fn flush_content(&mut self, store: &ConvStore, conv: &str, mark: &TurnMark, emit: &mut dyn FnMut(&Value) -> bool) {
        let text = std::mem::take(&mut self.content);
        if text.trim().is_empty() {
            return;
        }
        if let Ok(w) = store.append_event(
            conv,
            mark.ev("context_window", json!({ "role": "assistant", "content": text })),
        ) {
            let card = self.card("assistant", &text, &w.event_id, mark);
            emit(&card);
        }
    }

    fn flush(&mut self, store: &ConvStore, conv: &str, mark: &TurnMark, emit: &mut dyn FnMut(&Value) -> bool) {
        self.flush_reasoning(store, conv, mark, emit);
        self.flush_content(store, conv, mark, emit);
    }

    fn finish(&mut self, store: &ConvStore, conv: &str, mark: &TurnMark, emit: &mut dyn FnMut(&Value) -> bool) {
        self.flush_reasoning(store, conv, mark, emit);
        self.content.clear();
    }
}

impl TurnMark {
    pub fn new(handle: &str, participant: Option<i64>, round: i64) -> Self {
        TurnMark {
            id: format!("t{}", &crate::session_store::new_event_id_pub()[1..]),
            handle: handle.to_string(),
            participant,
            round,
        }
    }

    fn meta(&self) -> serde_json::Map<String, Value> {
        let mut m = serde_json::Map::new();
        m.insert("turn".into(), json!(self.id));
        if !self.handle.is_empty() {
            m.insert("handle".into(), json!(self.handle));
        }
        m.insert("round".into(), json!(self.round));
        m
    }

    pub fn ev(&self, kind: &str, payload: Value) -> NewEvent {
        NewEvent {
            event_type: kind.into(),
            payload,
            meta: Some(self.meta()),
            speaker: self.participant,
            ..Default::default()
        }
    }

    pub fn ev_with(&self, kind: &str, payload: Value, extra: &[(&str, Value)]) -> NewEvent {
        let mut ev = self.ev(kind, payload);
        let mut meta = ev.meta.take().unwrap_or_default();
        for (k, v) in extra {
            meta.insert((*k).to_string(), v.clone());
        }
        ev.meta = Some(meta);
        ev
    }
}

fn record_prompt(
    store: &ConvStore,
    conv: &str,
    mark: &TurnMark,
    system: &str,
    wire: &str,
    model: &str,
    history_turns: usize,
    emit: &mut dyn FnMut(&Value) -> bool,
) {
    if !system.trim().is_empty() {
        let w = store.append_event(
            conv,
            mark.ev(
                "system",
                json!({ "content": system, "bytes": system.len(), "model": model }),
            ),
        );
        emit(&json!({
            "type": "card",
            "card": { "kind": "system_prompt", "text": system, "bytes": system.len(),
                      "handle": mark.handle, "turn": mark.id, "round": mark.round,
                      "eventId": w.map(|x| x.event_id).unwrap_or_default(),
                      "participantId": mark.participant, "visible": true },
        }));
    }
    let w = store.append_event(
        conv,
        mark.ev(
            "agent_prompt",
            json!({
                "content": wire,
                "bytes": wire.len(),
                "model": model,
                "historyTurns": history_turns,
            }),
        ),
    );
    emit(&json!({
        "type": "card",
        "card": { "kind": "agent_prompt", "text": wire, "bytes": wire.len(),
                  "model": model, "historyTurns": history_turns,
                  "handle": mark.handle, "turn": mark.id, "round": mark.round,
                  "eventId": w.map(|x| x.event_id).unwrap_or_default(),
                  "participantId": mark.participant, "visible": true },
    }));
}

fn flush_approvals(
    store: &ConvStore,
    conv: &str,
    mark: &TurnMark,
    approvals: &Approvals,
    emit: &mut dyn FnMut(&Value) -> bool,
) {
    for row in approvals.take_answered(conv) {
        let w = store.append_event(conv, mark.ev("approval", row.clone()));
        let mut card = row;
        card["eventId"] = json!(w.map(|x| x.event_id).unwrap_or_default());
        card["turn"] = json!(mark.id);
        card["handle"] = json!(mark.handle);
        card["visible"] = json!(true);
        let _ = emit(&json!({ "type": "card", "card": card }));
    }
}

enum QueueOutcome {
    Started(Vec<std::sync::Arc<agl::queue::Permit>>),
    Dequeued,
    Deleted,
    Refused(String),
}

#[allow(clippy::too_many_arguments)]
fn queue_wait(
    queues: &crate::queue::Queues,
    store: &ConvStore,
    session_id: &str,
    item_id: &str,
    is_automation: bool,
    check: &dyn Fn(&Value) -> Result<(), String>,
    bump: &dyn Fn(),
    slots: &[String],
    sleep: impl Fn(Duration),
    tick: Duration,
) -> QueueOutcome {
    let bump = bump;
    let mut last_reason: Option<String> = None;
    loop {
        let meta = store.effective_session(session_id).and_then(|v| v.get("effective").cloned()).unwrap_or(Value::Null);
        let gate_ok = match check(&meta) {
            Ok(()) => true,
            Err(reason) => {
                if is_automation {
                    store.queue_remove(session_id, item_id);
                    bump();
                    return QueueOutcome::Refused(reason);
                }
                false
            }
        };
        let reason = crate::queue::reason(store, queues, session_id, item_id, gate_ok, slots);
        if last_reason.as_deref() != Some(reason.as_str()) {
            store.queue_set_reason(session_id, item_id, &reason);
            bump();
            last_reason = Some(reason);
        }
        if gate_ok {
            match queues.try_start(store, session_id, item_id, slots) {
                crate::queue::Start::Won(permits) => {
                    bump();
                    return QueueOutcome::Started(permits);
                }
                crate::queue::Start::Gone => {
                    return if store.load(session_id).is_none() {
                        QueueOutcome::Deleted
                    } else {
                        QueueOutcome::Dequeued
                    };
                }
                crate::queue::Start::Later => {}
            }
        } else if store.queue_row(session_id, item_id).is_none() {
            return if store.load(session_id).is_none() {
                QueueOutcome::Deleted
            } else {
                QueueOutcome::Dequeued
            };
        }
        sleep(tick);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_turn(
    app: &Arc<crate::serve::AppState>,
    store: &ConvStore,
    tabs: &LiveTabs,
    rt: &tokio::runtime::Runtime,
    project_root: &str,
    pool: &McpPool,
    cancels: &Cancels,
    approvals: &Arc<Approvals>,
    def: &AgentDef,
    req: TurnRequest,
    emit: &mut dyn FnMut(&Value) -> bool,
) -> Result<(), TurnError> {
    let content = req.content.trim().to_string();
    if content.is_empty() {
        return Err(TurnError::EmptyContent);
    }
    let model = Some(def.model.clone()).filter(|m| !m.trim().is_empty())
        .or_else(|| req.model.clone().filter(|m| !m.trim().is_empty()))
        .or_else(|| req.session_id.as_deref().and_then(|id| store.effective_session(id).and_then(|v| v.get("effective").cloned()))
            .and_then(|v| v.get("model").and_then(Value::as_str).map(String::from))
            .filter(|m| !m.trim().is_empty()))
        .unwrap_or_else(|| DEFAULT_MODEL.into());
    let session_id = resolve_session(store, tabs, &req)?;
    emit(&json!({ "type": "session", "tabId": req.tab_id, "sessionId": session_id }));

    tabs.set(
        &req.tab_id,
        TabEntry {
            session_id: session_id.clone(),
            agent: req.agent.clone(),
            model: model.clone(),
            ..Default::default()
        },
    );

    let item_id = store.queue_push(
        &session_id,
        &content,
        req.from.as_deref().unwrap_or_default(),
        &def.name,
        &model,
        &req.tab_id,
    );
    let slot_specs = vec![agl::agent::queue_spec(&model)];
    let mut held_slots: std::collections::HashMap<String, std::sync::Arc<agl::queue::Permit>> =
        std::collections::HashMap::new();
    match queue_wait(
        &app.queues,
        store,
        &session_id,
        &item_id,
        req.tab_id == "automation",
        &|m: &Value| crate::session_props::check(m, def),
        &|| {
            if let Some(send) = &app.queue_sender {
                send();
            }
        },
        &slot_specs,
        |_| std::thread::sleep(Duration::from_millis(250)),
        Duration::from_millis(250),
    ) {
        QueueOutcome::Started(permits) => {
            for (spec, p) in slot_specs.iter().cloned().zip(permits) {
                held_slots.insert(spec, p);
            }
        }
        QueueOutcome::Dequeued => {
            emit(&json!({ "type": "done", "ok": true, "dequeued": true, "sessionId": session_id }));
            return Ok(());
        }
        QueueOutcome::Deleted => {
            let text = "session was deleted while waiting";
            emit(&json!({ "type": "card", "card": { "kind": "error", "text": text } }));
            emit(&json!({ "type": "done", "ok": false, "error": text, "sessionId": session_id }));
            return Ok(());
        }
        QueueOutcome::Refused(reason) => {
            let text = format!("cannot run: the session is not configured — {reason}");
            store.set_status(&session_id, "error");
            emit(&json!({ "type": "card", "card": { "kind": "error", "text": text } }));
            emit(&json!({ "type": "done", "ok": false, "error": text, "sessionId": session_id }));
            return Ok(());
        }
    }
    let _running = crate::queue::Running::new(&app.queues, &session_id, &item_id);
    let stop = Armed::new(cancels, &session_id);

    title_from_first_message(store, &session_id, &content);
    store.set_status(&session_id, "running");
    let mut user_payload = serde_json::json!({ "role": "user", "content": content });
    if let Some(from) = req.from.as_ref().filter(|f| !f.trim().is_empty()) {
        user_payload["from"] = json!(from.trim());
    }
    let (event_id, ts) = match store.append_event(&session_id, NewEvent::new("context_window", user_payload)) {
        Ok(w) => (w.event_id, w.ts),
        Err(_) => (String::new(), String::new()),
    };
    emit(&json!({
        "type": "card",
        "card": {
            "kind": "user",
            "text": content,
            "from": req.from.as_deref().unwrap_or_default(),
            "eventId": event_id,
            "ts": ts,
            "visible": true,
        },
    }));

    let me = store.agent_participants(&session_id).into_iter().next();
    let mark = TurnMark::new(
        me.as_ref().map(|p| p.handle.as_str()).unwrap_or(&def.name),
        me.as_ref().map(|p| p.id),
        0,
    );

    if let Some(p) = me.clone() {
        let roster = vec![p];
        let quiet = |store: &ConvStore, emit: &mut dyn FnMut(&Value) -> bool| {
            store.set_status(&session_id, "ready");
            emit(&json!({
                "type": "done", "ok": true, "sessionId": session_id,
                "roundStatus": "settled", "reason": "not_engaged",
            }));
        };
        match classify(store, &roster, &content, emit).first().map(|(_, i)| i.clone()) {
            None => {
                quiet(store, emit);
                return Ok(());
            }
            Some(crate::group::Invite::Ask) => {
                if !decide_now(
                    store, rt, &stop.switch, &session_id, &roster, &roster[0], &content,
                    &mark, emit,
                ) {
                    quiet(store, emit);
                    return Ok(());
                }
            }
            _ => {}
        }
    }

    let context_window = store
        .load(&session_id)
        .and_then(|s| s.get("contextWindow").and_then(|v| v.as_u64()));

    let root = std::path::Path::new(project_root);
    let file_root = file_root_for(store, &session_id, root);
    let sandbox = match crate::sandbox::ensure(store, &session_id, def, &stop.switch) {
        Ok(id) => id.unwrap_or_default(),
        Err(msg) => {
            store.set_status(&session_id, "error");
            let _ = store.append_event(
                &session_id,
                NewEvent::new("error", json!({ "message": msg })),
            );
            emit(&json!({ "type": "card", "card": { "kind": "error", "text": msg } }));
            emit(&json!({ "type": "done", "ok": false, "error": msg, "sessionId": session_id }));
            return Err(TurnError::Agent(msg));
        }
    };
    let (servers, mcp_errors) = pool.ensure_in_sandbox_cancellable(&def.mcp, &file_root, &sandbox, &stop.switch);
    for e in &mcp_errors {
        if stop.stopped() { break; }
        let _ = store.append_event(&session_id, NewEvent::new("error", json!({ "message": e })));
        emit(&json!({ "type": "card", "card": { "kind": "error", "text": e } }));
    }
    if !sandbox.is_empty()
        && !stop.stopped()
        && !servers.iter().any(|s| s.name == SANDBOX_SERVER)
    {
        let msg = format!(
            "the sandbox for @{} did not come up — refusing to run outside it",
            def.name
        );
        store.set_status(&session_id, "error");
        let _ = store.append_event(&session_id, NewEvent::new("error", json!({ "message": msg })));
        emit(&json!({ "type": "card", "card": { "kind": "error", "text": msg } }));
        emit(&json!({ "type": "done", "ok": false, "error": msg, "sessionId": session_id }));
        return Err(TurnError::Agent(msg));
    }

    let started = std::time::Instant::now();
    let task = me
        .as_ref()
        .and_then(|p| crate::group::plan(store, &session_id, p, 0, true, "The user addressed you."));
    let system = task
        .as_ref()
        .map(|t| t.system_prompt.clone())
        .unwrap_or_else(|| {
            let view = store.effective_session(&session_id).unwrap_or(Value::Null);
            def.system_prompt_with_appendage(view["effective"]["soulAppendage"].as_str().unwrap_or(""))
        });
    let prompt = task
        .as_ref()
        .map(|t| t.prompt.clone())
        .unwrap_or_else(|| content.clone());
    let mut agent = Agent::factory(Factory {
        model: Some(model.clone()),
        slot: held_slots.get(&agl::agent::queue_spec(&model)).cloned(),
        reasoning_effort: turn_reasoning(def, store.effective_session(&session_id).and_then(|v| v.get("effective").cloned()).as_ref()),
        system_prompt: Some(system.clone()).filter(|p| !p.trim().is_empty()),
        retain_history: Some(true),
        parallel_tools: Some(true),
        max_turns: def.max_turns.map(|v| v as usize),
        message_filter: Some(std::sync::Arc::new({
            let root = store.project_root.clone();
            move |msg| crate::media::with_images(&root, msg)
        })),
        ..Default::default()
    })
    .map_err(|e| TurnError::Agent(e.to_string()))?;
    match &task {
        Some(t) => {
            for msg in &t.context {
                agent.push_context(crate::media::with_images(&store.project_root, msg.clone()));
            }
        }
        None => {
            for item in store.build_context_window_items(&session_id) {
                if !item.visible {
                    continue;
                }
                if let Some(msg) = item_to_message(&item) {
                    agent.push_context(crate::media::with_images(&store.project_root, msg));
                }
            }
        }
    }
    let session_state = store.effective_session(&session_id).and_then(|v| v.get("effective").cloned());
    let overridden = session_state
        .as_ref()
        .and_then(|s| s.get("allowlist"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let rules = combined_allowlist(&def.allowlist, overridden.as_deref());

    let mode = session_state
        .as_ref()
        .and_then(|s| s.get("allowlistMode"))
        .and_then(|v| v.as_str())
        .unwrap_or("allowlist")
        .to_string();
    let unrestricted = mode == "unrestricted";
    let allow_text = if unrestricted {
        "*".to_string()
    } else {
        rules.clone()
    };
    let tool_count = register_tools(
        &mut agent,
        &servers,
        def.tools_enabled.as_ref(),
        &allow_text,
        &def.policy_mode,
        &session_id,
        Arc::clone(approvals),
        Arc::clone(&stop.switch),
    );
    crate::chat_tools::register(
        &mut agent,
        app,
        def.tools_enabled.as_ref(),
        &allow_text,
        &def.policy_mode,
        &session_id,
        Arc::clone(approvals),
        Arc::clone(&stop.switch),
    );
    let tool_names: Vec<String> = agent.tools.keys().cloned().collect();

    emit(&json!({
        "type": "session",
        "tabId": req.tab_id,
        "sessionId": session_id,
        "agent": def.name,
        "model": model,
        "contextWindow": context_window,
        "tokensUsed": 0,
        "tools": tool_names,
        "mcp": def.mcp,
        "allowlist": rules,
        "effectiveAllowlist": allow_text,
        "allowlistMode": mode,
        "allowlistSource": if overridden.is_some() { "agent + session" } else { "agent" },
        "allowlistOverridden": overridden.is_some(),
        "policyMode": def.policy_mode,
        "projectRoot": project_root,
    }));
    let _ = tool_count;

    let history_turns = task
        .as_ref()
        .map(|t| t.context.len())
        .unwrap_or_else(|| store.build_context_window_items(&session_id).len());
    let wire = match &task {
        Some(t) => crate::group::render_wire(&t.messages(), &mark.handle),
        None => format!("{prompt}\n"),
    };
    record_prompt(
        store,
        &session_id,
        &mark,
        &system,
        &wire,
        &model,
        history_turns,
        emit,
    );

    let mut said = Said::new(mark.participant);
    let mut first_token_at: Option<std::time::Duration> = None;
    let answer = {
        let sid = session_id.clone();
        let who = mark.handle.clone();
        let mut on_event = |e: TurnEvent<'_>| match e {
            TurnEvent::Queued { .. } => {}
            TurnEvent::Delta { text, channel } => {
                if first_token_at.is_none() {
                    first_token_at = Some(started.elapsed());
                }
                match channel {
                Channel::Reasoning => {
                    said.reasoning.push_str(text);
                    emit(&json!({ "type": "reasoning_delta", "text": text,
                                  "sessionId": sid, "handle": who }));
                }
                Channel::Content => {
                    said.content.push_str(text);
                    emit(&json!({ "type": "assistant_delta", "text": text,
                                  "sessionId": sid, "handle": who }));
                }
                }
            }
            TurnEvent::ToolCall { id, name, args } => {
                said.flush(store, &sid, &mark, emit);
                let w = store.append_event(
                    &sid,
                    mark.ev("tool_call", json!({ "id": id, "name": name, "args": args })),
                );
                let event_id = w.map(|x| x.event_id).unwrap_or_default();
                emit(&json!({
                    "type": "card",
                    "card": { "kind": "tool_call", "name": name, "args": args,
                              "eventId": event_id, "turn": mark.id, "handle": mark.handle,
                              "visible": true },
                }));
            }
            TurnEvent::ToolResult { id, name, text } => {
                flush_approvals(store, &sid, &mark, approvals, emit);
                let ok = !text.starts_with("error:") && !text.starts_with("denied:");
                let w = store.append_event(
                    &sid,
                    mark.ev(
                        "tool_result",
                        json!({ "tool_call_id": id, "name": name, "text": text, "ok": ok,
                                "denied": text.starts_with("denied:") }),
                    ),
                );
                let event_id = w.map(|x| x.event_id).unwrap_or_default();
                emit(&json!({
                    "type": "card",
                    "card": { "kind": "tool_result", "name": name, "text": text,
                              "ok": ok, "denied": text.starts_with("denied:"),
                              "eventId": event_id, "turn": mark.id, "handle": mark.handle,
                              "visible": true },
                }));
            }
        };
        cancel::block_on(
            rt,
            &stop.switch,
            agent.run_message(
                crate::media::with_images(&store.project_root, Message::user(prompt.clone())),
                true,
                &mut on_event,
            ),
        )
    };

    let answer = match answer {
        Ran::Stopped => {
            said.flush(store, &session_id, &mark, emit);
            let _ = store.append_event(
                &session_id,
                mark.ev("stopped", json!({ "reason": "user" })),
            );
            store.record_turn_complete(&session_id, None);
            emit(&json!({
                "type": "card",
                "card": { "kind": "system", "text": STOPPED_NOTICE, "visible": true },
            }));
            emit(&json!({
                "type": "done", "ok": true, "stopped": true, "sessionId": session_id,
            }));
            return Ok(());
        }
        Ran::Done(Ok(a)) => a,
        Ran::Done(Err(e)) => {
            store.set_status(&session_id, "error");
            let msg = e.to_string();
            let _ = store.append_event(
                &session_id,
                NewEvent::new("error", json!({ "message": msg })),
            );
            emit(&json!({ "type": "card", "card": { "kind": "error", "text": msg } }));
            emit(&json!({ "type": "done", "ok": false, "error": msg, "sessionId": session_id }));
            return Err(TurnError::Agent(msg));
        }
    };

    said.finish(store, &session_id, &mark, emit);

    let usage = agent.last_usage.lock().unwrap().clone();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let tok_per_sec = usage
        .as_ref()
        .and_then(|u| u.completion_tokens)
        .filter(|_| elapsed_ms > 0.0)
        .map(|ct| ct as f64 / (elapsed_ms / 1000.0));
    let telemetry = match &usage {
        Some(u) => json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
            "tok_per_sec": tok_per_sec,
            "duration_ms": elapsed_ms,
            "ttft_ms": first_token_at.map(|d| d.as_secs_f64() * 1000.0),
            "model": model,
        }),
        None => Value::Null,
    };
    if let Some(u) = &usage {
        if let Some(p) = u.prompt_tokens {
            emit(&json!({
                "type": "usage",
                "prompt_tokens": p,
                "total_tokens": u.total_tokens,
                "sessionId": session_id,
            }));
            let mut patch = serde_json::Map::new();
            patch.insert("lastPromptTokens".into(), json!(p));
            patch.insert("lastTotalTokens".into(), json!(u.total_tokens));
            patch.insert("lastCompletionTokens".into(), json!(u.completion_tokens));
            store.update_meta(&session_id, patch);
        }
    }

    let _ = store.append_event(&session_id, {
        let mut ev = mark.ev_with(
            "gen_info",
            telemetry.clone(),
            &[("telemetry", telemetry.clone())],
        );
        ev.duration_ms = Some(elapsed_ms);
        ev
    });

    let text = answer.trim().to_string();
    let mut sealed_at = String::new();
    let mut sealed_id = String::new();
    if !text.is_empty() {
        if let Ok(w) = store.append_event(
            &session_id,
            mark.ev("context_window", json!({ "role": "assistant", "content": text })),
        ) {
            sealed_at = w.ts;
            sealed_id = w.event_id;
        }
    }
    let handle = store
        .agent_participants(&session_id)
        .first()
        .map(|p| p.handle.clone());
    emit(&json!({
        "type": "card",
        "card": { "kind": "assistant", "text": text, "telemetry": telemetry,
                  "handle": handle, "turn": mark.id, "ts": sealed_at,
                  "eventId": sealed_id, "streaming": false },
    }));

    store.record_turn_complete(&session_id, Some(elapsed_ms));
    emit(&json!({ "type": "done", "ok": true, "sessionId": session_id }));
    Ok(())
}




#[allow(clippy::too_many_arguments)]
pub fn run_round(
    app: &Arc<crate::serve::AppState>,
    store: &ConvStore,
    tabs: &LiveTabs,
    rt: &tokio::runtime::Runtime,
    project_root: &str,
    pool: &McpPool,
    cancels: &Cancels,
    approvals: &Arc<Approvals>,
    def: &AgentDef,
    req: TurnRequest,
    emit: &mut dyn FnMut(&Value) -> bool,
) -> Result<(), TurnError> {
    let content = req.content.trim().to_string();
    if content.is_empty() {
        return Err(TurnError::EmptyContent);
    }
    let conv = resolve_session(store, tabs, &req)?;
    emit(&json!({ "type": "session", "tabId": req.tab_id, "sessionId": conv }));
    tabs.set(
        &req.tab_id,
        TabEntry {
            session_id: conv.clone(),
            agent: req.agent.clone(),
            model: req.model.clone().unwrap_or_default(),
            ..Default::default()
        },
    );

    let item_id = store.queue_push(
        &conv,
        &content,
        req.from.as_deref().unwrap_or_default(),
        &def.name,
        req.model.as_deref().unwrap_or_default(),
        &req.tab_id,
    );
    let slot_specs = round_slot_specs(store, &conv, def);
    let mut held_slots: std::collections::HashMap<String, std::sync::Arc<agl::queue::Permit>> =
        std::collections::HashMap::new();
    match queue_wait(
        &app.queues,
        store,
        &conv,
        &item_id,
        req.tab_id == "automation",
        &|m: &Value| crate::session_props::check(m, def),
        &|| {
            if let Some(send) = &app.queue_sender {
                send();
            }
        },
        &slot_specs,
        |_| std::thread::sleep(Duration::from_millis(250)),
        Duration::from_millis(250),
    ) {
        QueueOutcome::Started(permits) => {
            for (spec, p) in slot_specs.iter().cloned().zip(permits) {
                held_slots.insert(spec, p);
            }
        }
        QueueOutcome::Dequeued => {
            emit(&json!({ "type": "done", "ok": true, "dequeued": true, "sessionId": conv }));
            return Ok(());
        }
        QueueOutcome::Deleted => {
            let text = "session was deleted while waiting";
            emit(&json!({ "type": "card", "card": { "kind": "error", "text": text } }));
            emit(&json!({ "type": "done", "ok": false, "error": text, "sessionId": conv }));
            return Ok(());
        }
        QueueOutcome::Refused(reason) => {
            let text = format!("cannot run: the session is not configured — {reason}");
            store.set_status(&conv, "error");
            emit(&json!({ "type": "card", "card": { "kind": "error", "text": text } }));
            emit(&json!({ "type": "done", "ok": false, "error": text, "sessionId": conv }));
            return Ok(());
        }
    }
    let _running = crate::queue::Running::new(&app.queues, &conv, &item_id);
    let stop = Armed::new(cancels, &conv);

    title_from_first_message(store, &conv, &content);
    store.set_status(&conv, "running");
    let mut user_payload = serde_json::json!({ "role": "user", "content": content });
    if let Some(from) = req.from.as_ref().filter(|f| !f.trim().is_empty()) {
        user_payload["from"] = json!(from.trim());
    }
    let user_node = store
        .append_event(&conv, NewEvent::new("context_window", user_payload))
        .map_err(|e| TurnError::Store(e.to_string()))?;
    emit(&json!({
        "type": "card",
        "card": {
            "kind": "user",
            "text": content,
            "from": req.from.as_deref().unwrap_or_default(),
            "eventId": user_node.event_id,
            "ts": user_node.ts,
            "visible": true,
        },
    }));

    let roster = store.participants(&conv);

    let header = store.effective_session(&conv).and_then(|v| v.get("effective").cloned()).unwrap_or(Value::Null);

    if !roster.iter().any(|p| p.is_agent() && p.active) {
        let humans = roster.iter().filter(|p| !p.is_agent() && p.active).count();
        if humans < 2 {
            let notice = "No agents in this conversation yet. \
                          Invite one with /invite @name — see the Agents rail.";
            emit(&json!({
                "type": "card",
                "card": { "kind": "system", "text": notice, "visible": true },
            }));
        }
        store.set_status(&conv, "ready");
        emit(&json!({
            "type": "done", "ok": true, "sessionId": conv,
            "rounds": 0, "messages": 0,
            "roundStatus": "settled", "reason": "no_agents",
            "userEventId": user_node.event_id,
        }));
        return Ok(());
    }
    let routing = header
        .get("routing")
        .and_then(|v| v.as_str())
        .filter(|v| !v.is_empty())
        .unwrap_or("everyone")
        .to_string();

    emit(&json!({
        "type": "session",
        "tabId": req.tab_id,
        "sessionId": conv,
        "participants": roster.iter().map(|p| p.to_json()).collect::<Vec<_>>(),
        "routing": routing,
        "projectRoot": project_root,
    }));

    let invites: Vec<(i64, crate::group::Invite)> = if routing == "lead" {
        lead_assignment(store, rt, &stop.switch, &conv, &roster, &content, emit)
            .into_iter()
            .map(|pid| (pid, crate::group::Invite::Mentioned))
            .collect()
    } else {
        classify(store, &roster, &content, emit)
    };
    let mut responders: Vec<i64> = invites.iter().map(|(pid, _)| *pid).collect();
    let mut round_index: i64 = 0;
    let mut published: usize = 0;

    let settled = 'rounds: loop {
        if stop.stopped() {
            break crate::group::Settled::Stopped;
        }
        if let Some(s) = crate::group::check_bounds(round_index, published, responders.len()) {
            break s;
        }
        let mut round_texts: Vec<(i64, String)> = Vec::new();
        let mut replied: Vec<i64> = Vec::new();

        for pid in responders.clone() {
            if stop.stopped() {
                break 'rounds crate::group::Settled::Stopped;
            }
            let Some(me) = roster.iter().find(|p| p.id == pid).cloned() else { continue };
            let reply_required = round_index == 0;
            let how = if round_index == 0 {
                if crate::group::mentions(&content).contains(&me.handle) {
                    "The user directly addressed you."
                } else {
                    "The conversation picked you to answer the user's message."
                }
            } else {
                ""
            };
            let Some(task) = crate::group::plan(store, &conv, &me, round_index, reply_required, how)
            else {
                continue;
            };

            emit(&json!({
                "type": "speaking",
                "sessionId": conv,
                "participantId": me.id,
                "handle": me.handle,
                "round": round_index,
                "taskId": task.task_id,
            }));

            let participant_def = me.agent_name.as_deref()
                .and_then(|name| store.agents().get(name))
                .unwrap_or_else(|| def.clone());
            let model = Some(participant_def.model.clone()).filter(|m| !m.trim().is_empty())
                .or_else(|| header.get("model").and_then(|v| v.as_str()).map(String::from).filter(|m| !m.trim().is_empty()))
                .unwrap_or_else(|| DEFAULT_MODEL.into());
            let mark = TurnMark::new(&me.handle, Some(me.id), round_index);

            if round_index == 0
                && invites
                    .iter()
                    .any(|(id, inv)| *id == pid && *inv == crate::group::Invite::Ask)
                && !decide_now(
                    store, rt, &stop.switch, &conv, &roster, &me, &content, &mark, emit,
                )
            {
                continue;
            }
            let root = std::path::Path::new(project_root);
            let file_root = file_root_for(store, &conv, root);
            let sandbox = match crate::sandbox::ensure(store, &conv, &participant_def, &stop.switch) {
                Ok(id) => id.unwrap_or_default(),
                Err(msg) => {
                    let _ = store.append_event(
                        &conv,
                        mark.ev("error", json!({ "message": msg, "handle": me.handle })),
                    );
                    emit(&json!({ "type": "card", "card": { "kind": "error", "text": msg } }));
                    continue;
                }
            };
            let (servers, mcp_errors) =
                pool.ensure_in_sandbox_cancellable(&participant_def.mcp, &file_root, &sandbox, &stop.switch);
            for e in &mcp_errors {
                if stop.stopped() { break; }
                let _ = store.append_event(&conv, mark.ev("error", json!({ "message": e })));
                emit(&json!({ "type": "card", "card": { "kind": "error", "text": e } }));
            }
            if !sandbox.is_empty()
                && !stop.stopped()
                && !servers.iter().any(|s| s.name == SANDBOX_SERVER)
            {
                let msg = format!(
                    "the sandbox for @{} did not come up — refusing to run outside it",
                    participant_def.name
                );
                let _ = store.append_event(
                    &conv,
                    mark.ev("error", json!({ "message": msg, "handle": me.handle })),
                );
                emit(&json!({ "type": "card", "card": { "kind": "error", "text": msg } }));
                continue;
            }
            let reply = match run_participant(
                app, store, rt, &stop.switch, approvals, &servers, &participant_def, &conv, &me,
                &task, &model, &held_slots, &mark, emit,
            ) {
                Ok(r) => r,
                Err(TurnError::Stopped) => break 'rounds crate::group::Settled::Stopped,
                Err(e) => {
                    let msg = e.to_string();
                    let _ = store.append_event(
                        &conv,
                        mark.ev("error", json!({ "message": msg, "handle": me.handle })),
                    );
                    emit(&json!({ "type": "card", "card": { "kind": "error", "text": msg } }));
                    continue;
                }
            };

            let reply = crate::group::strip_echoed_prefix(&reply, &me.handle);

            if crate::group::is_pass(&reply) {
                store.append_own_turn(&conv, me.id, &task.prompt, &reply, Some(&task.task_id));
                advance(store, &conv, &me.id, task.seen_through_seq, round_index, &task.task_id);
                emit(&json!({
                    "type": "passed",
                    "sessionId": conv,
                    "participantId": me.id,
                    "handle": me.handle,
                    "round": round_index,
                }));
                replied.push(me.id);
                continue;
            }

            let text = crate::group::clamp_message(&reply);
            let node = store
                .append_event(
                    &conv,
                    mark.ev("context_window", json!({ "role": "assistant", "content": text })),
                )
                .map_err(|e| TurnError::Store(e.to_string()))?;
            store.append_own_turn(&conv, me.id, &task.prompt, &text, Some(&task.task_id));
            let seq = store.node(&node.event_id).map(|n| n.seq).unwrap_or(task.seen_through_seq);
            advance(store, &conv, &me.id, seq, round_index, &task.task_id);

            emit(&json!({
                "type": "card",
                "card": {
                    "kind": "assistant", "text": text, "eventId": node.event_id,
                    "participantId": me.id, "handle": me.handle, "turn": mark.id,
                    "ts": node.ts,
                    "round": round_index, "streaming": false, "visible": true,
                },
            }));
            round_texts.push((me.id, text));
            replied.push(me.id);
            published += 1;
            if published >= crate::group::MAX_MESSAGES {
                break;
            }
        }

        round_index += 1;
        responders = crate::group::responders(&roster, &content, round_index, &round_texts, &replied);
        if responders.is_empty() {
            break crate::group::Settled::SilentRound;
        }
    };

    let s = settled;
    let mut patch = serde_json::Map::new();
    patch.insert("roundIndex".into(), json!(round_index));
    patch.insert("roundStatus".into(), json!(s.status()));
    store.update_meta(&conv, patch);
    for p in roster.iter().filter(|p| p.is_agent()) {
        let (seq, _, _, _) = store.participant_view(&conv, p.id);
        store.set_participant_view(&conv, p.id, seq, round_index, false, None, Some(s.status()));
    }
    store.record_turn_complete(&conv, None);
    let was_stopped = s == crate::group::Settled::Stopped;
    if was_stopped {
        let _ = store.append_event(
            &conv,
            NewEvent::new("stopped", json!({ "reason": "user" })),
        );
        emit(&json!({
            "type": "card",
            "card": { "kind": "system", "text": STOPPED_NOTICE, "visible": true },
        }));
    }
    emit(&json!({
        "type": "done",
        "ok": true,
        "stopped": was_stopped,
        "sessionId": conv,
        "rounds": round_index,
        "messages": published,
        "roundStatus": s.status(),
        "reason": s.reason(),
        "userEventId": user_node.event_id,
    }));
    Ok(())
}

fn round_slot_specs(store: &ConvStore, conv: &str, def: &AgentDef) -> Vec<String> {
    let header = store.effective_session(conv).and_then(|v| v.get("effective").cloned()).unwrap_or(Value::Null);
    let agents = store.agents();
    let mut out: Vec<String> = Vec::new();
    for me in store.agent_participants(conv) {
        let pdef = me
            .agent_name
            .as_deref()
            .and_then(|n| agents.get(n))
            .unwrap_or_else(|| def.clone());
        let model = Some(pdef.model.clone())
            .filter(|m| !m.trim().is_empty())
            .or_else(|| {
                header
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| DEFAULT_MODEL.into());
        let spec = agl::agent::queue_spec(&model);
        if !out.contains(&spec) {
            out.push(spec);
        }
    }
    out.sort();
    out
}

fn advance(store: &ConvStore, conv: &str, pid: &i64, seq: i64, round: i64, task_id: &str) {
    store.set_participant_view(conv, *pid, seq, round, false, Some(task_id), Some("running"));
}

#[allow(clippy::too_many_arguments)]
fn run_participant(
    app: &Arc<crate::serve::AppState>,
    store: &ConvStore,
    rt: &tokio::runtime::Runtime,
    stop: &Arc<Cancel>,
    approvals: &Arc<Approvals>,
    servers: &[Arc<McpServer>],
    def: &AgentDef,
    conv: &str,
    me: &crate::conv_store::Participant,
    task: &crate::group::Task,
    model: &str,
    held_slots: &std::collections::HashMap<String, std::sync::Arc<agl::queue::Permit>>,
    mark: &TurnMark,
    emit: &mut dyn FnMut(&Value) -> bool,
) -> Result<String, TurnError> {
    let system = if task.system_prompt.trim().is_empty() {
        let view = store.effective_session(conv).unwrap_or(Value::Null);
        let base = def.system_prompt_with_appendage(view["effective"]["soulAppendage"].as_str().unwrap_or(""));
        format!(
            "{base}\n\nYou are @{} in a conversation with other participants. \
             Reply as yourself only.",
            me.handle
        )
        .trim()
        .to_string()
    } else {
        task.system_prompt.clone()
    };
    let mut agent = Agent::factory(Factory {
        model: Some(model.to_string()),
        slot: held_slots.get(&agl::agent::queue_spec(model)).cloned(),
        reasoning_effort: turn_reasoning(def, store.effective_session(conv).and_then(|v| v.get("effective").cloned()).as_ref()),
        system_prompt: Some(system.clone()),
        retain_history: Some(true),
        parallel_tools: Some(true),
        max_turns: def.max_turns.map(|v| v as usize),
        message_filter: Some(std::sync::Arc::new({
            let root = store.project_root.clone();
            move |msg| crate::media::with_images(&root, msg)
        })),
        ..Default::default()
    })
    .map_err(|e| TurnError::Agent(e.to_string()))?;
    for msg in &task.context {
        agent.push_context(crate::media::with_images(&store.project_root, msg.clone()));
    }

    let session_state = store.effective_session(conv).and_then(|v| v.get("effective").cloned());
    let overridden = session_state
        .as_ref()
        .and_then(|s| s.get("allowlist"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let rules = combined_allowlist(&def.allowlist, overridden.as_deref());
    let unrestricted = session_state
        .as_ref()
        .and_then(|s| s.get("allowlistMode"))
        .and_then(|v| v.as_str())
        == Some("unrestricted");
    let allow_text = if unrestricted { "*".to_string() } else { rules };
    register_tools(
        &mut agent,
        servers,
        def.tools_enabled.as_ref(),
        &allow_text,
        &def.policy_mode,
        conv,
        Arc::clone(approvals),
        Arc::clone(stop),
    );
    crate::chat_tools::register(
        &mut agent,
        app,
        def.tools_enabled.as_ref(),
        &allow_text,
        &def.policy_mode,
        conv,
        Arc::clone(approvals),
        Arc::clone(stop),
    );

    record_prompt(
        store,
        conv,
        mark,
        &system,
        &crate::group::render_wire(&task.messages(), &task.handle),
        model,
        task.context.len(),
        emit,
    );

    let mut said = Said::new(Some(me.id));
    let handle = me.handle.clone();
    let pid = me.id;
    let answer = {
        let mut on_event = |e: TurnEvent<'_>| match e {
            TurnEvent::Queued { .. } => {}
            TurnEvent::Delta { text, channel } => match channel {
                Channel::Reasoning => {
                    said.reasoning.push_str(text);
                    emit(&json!({ "type": "reasoning_delta", "text": text,
                                  "sessionId": conv, "handle": handle, "participantId": pid }));
                }
                Channel::Content => {
                    said.content.push_str(text);
                    emit(&json!({ "type": "assistant_delta", "text": text,
                                  "sessionId": conv, "handle": handle, "participantId": pid }));
                }
            },
            TurnEvent::ToolCall { id, name, args } => {
                said.flush(store, conv, mark, emit);
                let w = store.append_event(
                    conv,
                    mark.ev("tool_call", json!({ "id": id, "name": name, "args": args })),
                );
                let event_id = w.map(|x| x.event_id).unwrap_or_default();
                emit(&json!({
                    "type": "card",
                    "card": { "kind": "tool_call", "name": name, "args": args,
                              "eventId": event_id, "participantId": pid,
                              "turn": mark.id, "handle": handle, "visible": true },
                }));
            }
            TurnEvent::ToolResult { id, name, text } => {
                flush_approvals(store, conv, mark, approvals, emit);
                let ok = !text.starts_with("error:") && !text.starts_with("denied:");
                let w = store.append_event(
                    conv,
                    mark.ev(
                        "tool_result",
                        json!({ "tool_call_id": id, "name": name, "text": text,
                                "ok": ok, "denied": text.starts_with("denied:") }),
                    ),
                );
                let event_id = w.map(|x| x.event_id).unwrap_or_default();
                emit(&json!({
                    "type": "card",
                    "card": { "kind": "tool_result", "name": name, "text": text, "ok": ok,
                              "denied": text.starts_with("denied:"), "eventId": event_id,
                              "participantId": pid, "turn": mark.id, "handle": handle,
                              "visible": true },
                }));
            }
        };
        cancel::block_on(
            rt,
            stop,
            agent.run_message(
                crate::media::with_images(&store.project_root, Message::user(task.prompt.clone())),
                true,
                &mut on_event,
            ),
        )
    };

    let answer = match answer {
        Ran::Stopped => {
            said.flush(store, conv, mark, emit);
            return Err(TurnError::Stopped);
        }
        Ran::Done(r) => r.map_err(|e| TurnError::Agent(e.to_string()))?,
    };

    said.finish(store, conv, mark, emit);
    Ok(answer)
}

fn lead_assignment(
    store: &ConvStore,
    rt: &tokio::runtime::Runtime,
    stop: &Cancel,
    conv: &str,
    roster: &[crate::conv_store::Participant],
    content: &str,
    emit: &mut dyn FnMut(&Value) -> bool,
) -> Vec<i64> {
    let agents: Vec<&crate::conv_store::Participant> =
        roster.iter().filter(|p| p.is_agent() && p.active).collect();
    let Some(lead) = agents
        .iter()
        .find(|p| p.role.as_deref() == Some("lead"))
        .or_else(|| agents.first())
    else {
        return Vec::new();
    };
    let header = store.effective_session(conv).and_then(|v| v.get("effective").cloned()).unwrap_or(Value::Null);
    let prompt = crate::group::routing_prompt(
        header.get("title").and_then(|v| v.as_str()).unwrap_or("untitled"),
        header.get("purpose").and_then(|v| v.as_str()),
        lead,
        roster,
        content,
    );
    let model = lead
        .model
        .clone()
        .filter(|m| !m.trim().is_empty())
        .or_else(|| header.get("model").and_then(|v| v.as_str()).map(String::from))
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_MODEL.into());
    let routed = Agent::factory(Factory {
        model: Some(model),
        system_prompt: Some("You route messages. Reply with JSON only.".into()),
        retain_history: Some(false),
        ..Default::default()
    })
    .ok()
    .and_then(|a| match cancel::block_on(rt, stop, a.run(&prompt)) {
        Ran::Done(r) => r.ok(),
        Ran::Stopped => None,
    })
    .and_then(|r| crate::group::parse_assignment(&r));

    match routed {
        Some(handles) if !handles.is_empty() => {
            let ids: Vec<i64> = agents
                .iter()
                .filter(|p| handles.contains(&p.handle))
                .map(|p| p.id)
                .collect();
            emit(&json!({ "type": "routed", "sessionId": conv, "by": lead.handle,
                          "assigned": handles }));
            if ids.is_empty() {
                vec![lead.id]
            } else {
                ids
            }
        }
        _ => {
            emit(&json!({ "type": "routed", "sessionId": conv, "by": lead.handle,
                          "assigned": [lead.handle.clone()], "fallback": true }));
            vec![lead.id]
        }
    }
}


fn classify(
    store: &ConvStore,
    roster: &[crate::conv_store::Participant],
    content: &str,
    emit: &mut dyn FnMut(&Value) -> bool,
) -> Vec<(i64, crate::group::Invite)> {
    let agents = store.agents();
    let engagement_of = |p: &crate::conv_store::Participant| {
        if let Some(over) = p.engagement.as_deref().filter(|s| !s.trim().is_empty()) {
            return crate::agent_loader::Engagement::parse(over);
        }
        p.agent_name
            .as_deref()
            .and_then(|n| agents.get(n))
            .map(|d| d.engagement())
            .unwrap_or(crate::agent_loader::Engagement::Engaged)
    };

    let mut out = Vec::new();
    for (pid, invite) in crate::group::invitations(roster, &engagement_of, content) {
        let Some(me) = roster.iter().find(|p| p.id == pid) else { continue };
        if invite == crate::group::Invite::Skip {
            emit(&json!({
                "type": "not_speaking", "participantId": pid, "handle": me.handle,
                "reason": "invoked", "detail": "waiting to be @mentioned",
            }));
            continue;
        }
        out.push((pid, invite));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn decide_now(
    store: &ConvStore,
    rt: &tokio::runtime::Runtime,
    stop: &Cancel,
    conv: &str,
    roster: &[crate::conv_store::Participant],
    me: &crate::conv_store::Participant,
    content: &str,
    mark: &TurnMark,
    emit: &mut dyn FnMut(&Value) -> bool,
) -> bool {
    let user_label = store.user_label();
    let mut history: Vec<String> = Vec::new();
    let mut headed: Option<(i32, i32)> = None;
    for n in store.path(conv).iter() {
        if n.kind != "context_window" || !n.visible {
            continue;
        }
        let Some(text) = n.content.clone() else { continue };
        let sender = match n.role.as_deref() {
            Some("user") => user_label.clone(),
            _ => n
                .participant_id
                .and_then(|id| roster.iter().find(|p| p.id == id))
                .map(|p| p.handle.clone())
                .unwrap_or_else(|| "assistant".into()),
        };
        if let Some(e) = crate::environment::epoch_from_iso(&n.ts) {
            let key = crate::environment::day_key(e);
            if headed != Some(key) {
                if let Some(h) = crate::relevance::day_header(&n.ts) {
                    history.push(h);
                }
                headed = Some(key);
            }
        }
        history.push(crate::relevance::history_line(&n.ts, &sender, &text));
    }

    let identity = me
        .agent_name
        .as_deref()
        .and_then(|n| store.agents().get(n))
        .map(|d| {
            let view = store.effective_session(conv).unwrap_or(Value::Null);
            d.system_prompt_with_appendage(view["effective"]["soulAppendage"].as_str().unwrap_or(""))
        })
        .unwrap_or_else(|| me.system_prompt.clone());
    let verdict = crate::relevance::should_speak(rt, stop, me, &identity, &history);

    let payload = verdict.to_json();
    let w = store.append_event(conv, mark.ev("decision", payload.clone()));
    emit(&json!({
        "type": "card",
        "card": { "kind": "decision", "handle": me.handle, "turn": mark.id,
                  "participantId": me.id, "verdict": payload,
                  "speak": verdict.speak, "confidence": verdict.confidence,
                  "reason": verdict.reason, "error": verdict.error,
                  "eventId": w.map(|x| x.event_id).unwrap_or_default(),
                  "visible": true },
    }));
    emit(&json!({
        "type": if verdict.speak { "speaking_because" } else { "not_speaking" },
        "sessionId": conv, "participantId": me.id, "handle": me.handle,
        "reason": "relevant", "verdict": verdict.to_json(),
    }));
    let _ = content;
    verdict.speak
}
