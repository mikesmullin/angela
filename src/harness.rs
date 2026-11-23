//! The turn loop — the part of `angela/src/index.mjs` that drives agl and
//! turns a model turn into session events plus a card stream.
//!
//! ## How a turn runs
//!
//! 1. Resolve the tab's session (resume, or create one).
//! 2. Title a fresh session from its first user message.
//! 3. Persist the user message as a `context_window` event.
//! 4. Hydrate the agent's context window **from the log**, then run agl with
//!    streaming deltas.
//! 5. Persist reasoning and the assistant reply as events.
//! 6. Record the completed turn.
//!
//! Hydrating from the log rather than from a live in-process agent is what
//! `build_context_window_items` was written for ("session open / resume"), and
//! it means a server restart does not amnesia the conversation — the JS keeps
//! a live harness per tab *and* can rehydrate; this port starts from the
//! durable half.

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

/// How long one tool call may take before the turn gives up on it.
const TOOL_TIMEOUT: Duration = Duration::from_secs(120);
/// Whisper may transcribe an hour-long recording locally; the MCP tool has its
/// own two-hour deadline, and Stop remains available throughout.
const VIDEO_TRANSCRIPT_TIMEOUT: Duration = Duration::from_secs(2 * 60 * 60 + 60);
/// Starting a server should be fast; a hang here would stall the first turn.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

/// Which session a UI tab is currently pointed at.
#[derive(Debug, Clone, Default)]
pub struct TabEntry {
    pub session_id: String,
    pub agent: Option<String>,
    pub model: String,
    pub last_prompt_tokens: Option<u64>,
    pub last_total_tokens: Option<u64>,
}

/// `liveTabs` — tab id → the session it is driving.
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

    /// Forget every tab pointing at a session — used when it is deleted.
    pub fn forget_session(&self, session_id: &str) {
        self.map
            .lock()
            .unwrap()
            .retain(|_, e| e.session_id != session_id);
    }
}

/// What the transcript says where a turn was cut off.
///
/// Said rather than left blank: a reply that stops halfway and a reply that
/// finished are the same shape on screen, and the difference is one the
/// reader needs — especially on a reload, where there is no memory of having
/// pressed the button.
pub const STOPPED_NOTICE: &str = "Stopped.";

/// A turn's registration in [`Cancels`], removed however the turn ends.
///
/// A guard rather than a call at the end: `run_turn` leaves through a dozen
/// `?`s and early returns, and a session left in the map would make the
/// *next* stop press look like it worked while hitting nothing.
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

/// Process-wide MCP connections, started on demand and shared across turns.
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
    /// Connect (or reuse) the named presets, returning those that came up.
    ///
    /// A server that fails to start is reported and skipped rather than
    /// failing the turn: losing shell access should not cost you the chat.
    ///
    /// `file_root` is what each server's `${projectRoot}` expands to — the
    /// agent's workspace root, passed by the caller. The cache key includes
    /// it because two agents with different roots need different `file-io`
    /// processes: the first root would be a lie for the second. One `file-io`,
    /// same root, one process, no matter how many turns or sessions reuse it.
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
            // Keep lookup and initialization together: different sessions can
            // request the same server concurrently on different HTTP threads.
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

    /// [`ensure`](Self::ensure), as seen from inside the session's sandbox.
    ///
    /// A sandbox is a property of the **session**, not of an agent: an
    /// isolation layer that constrains every participant in it. So this is
    /// not a per-agent substitution of one server for another — it is the
    /// session's filesystem and shell, and it applies the same way to
    /// whoever is speaking.
    ///
    /// Two consequences, and they are separate:
    ///
    /// * The two servers whose whole meaning is "where the files and the
    ///   shell are" are **removed** from every participant, whether or not
    ///   this particular one was configured with them. A conversation
    ///   working in a sandbox must not also be able to read and write this
    ///   machine — through any of its participants.
    /// * The sandbox's own `read` / `write` / `bash` are **present** for
    ///   every participant, likewise whether or not this one had host file
    ///   tools. Inside the boundary that is the point of the boundary: the
    ///   capability is already constrained by the container, and an agent
    ///   that cannot see the session's working tree cannot take part in
    ///   work happening there.
    ///
    /// Every other server — the browser, the researcher, the camera — is
    /// untouched, because none of them is about *this* host's filesystem.
    ///
    /// An unreachable sandbox is an **error, not a fallback**. Silently
    /// leaving the host tools in place would mean a conversation the user
    /// believes is sandboxed quietly writing to their machine, which is the
    /// worst outcome available here.
    ///
    /// `file_root` is the agent's own workspace root: this is what `ensure`
    /// uses to expand `${projectRoot}` for this agent's servers. The file and
    /// shell servers get the sandbox's, instead — that discards `file_root`
    /// for them, which is exactly the isolation a sandbox is for.
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
        let (mut out, mut errors) = self.ensure_cancellable(&kept, file_root, stop);
        if stop.stopped() { return (out, errors); }

        // Unconditional: the sandbox is the session's, so every participant
        // addresses it, including one whose own `mcp` list named no file or
        // shell server. The earlier version attached it only to agents that
        // had host tools to displace, which made the isolation layer look
        // per-agent and left the other participants unable to see the very
        // tree the session is working in.
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

/// The file-system root an agent's MCP servers should believe the disk begins
/// at: the agent's own workspace root (relative values resolving under the
/// project root), or the project root when it named none.
///
/// This is what `${projectRoot}` in the MCP config expands to *for this
/// agent* — before the per-agent field it was always the project root.
fn file_root_for(def: &AgentDef, project_root: &std::path::Path) -> std::path::PathBuf {
    let cwd = def.cwd.as_deref().map(|c| c.trim());
    match cwd {
        Some(c) if !c.is_empty() => {
            let cwd_root: std::path::PathBuf = c.into();
            if cwd_root.is_absolute() {
                cwd_root
            } else {
                project_root.join(c)
            }
        }
        _ => project_root.to_path_buf(),
    }
}

/// The servers a sandbox assignment takes over from.
///
/// Exactly the two that mean "the filesystem" and "the shell". Listed rather
/// than inferred: a server is displaced because of what it *is*, and there
/// is no property of an MCP server that says so.
pub const DISPLACED_BY_SANDBOX: &[&str] = &["file-io", "term-local"];

/// What the sandbox's server is called, and therefore the prefix its tools
/// wear: `sandbox__bash`, `sandbox__read`, `sandbox__write`.
///
/// Deliberately **not** `file-io` / `term-local`. Borrowing those names
/// would keep existing allowlists working, at the cost of a tool called
/// `file_io__read_file` that reads a different machine's files — and a
/// schema that is not quite the one that name has always had. A tool whose
/// name lies is worse than one an allowlist has to learn.
pub const SANDBOX_SERVER: &str = "sandbox";

/// Where a sandbox has the session's checkout mounted.
///
/// One fixed guest path, so an agent's instructions never have to vary with
/// which repo happens to be attached. It lives here, beside the server
/// name, because both halves need it and they are in different crates: the
/// chat server builds the bind mount from it, and
/// [`environment::section`](crate::environment::section) tells the agent
/// about it. Two spellings of this would be a sandbox whose files the agent
/// cannot find.
pub const SANDBOX_MOUNT: &str = "/work";

/// Where sbox serves one sandbox's tools.
///
/// The `exec` facade, not `mcp`: `exec` is the guest's file-and-shell
/// daemon, which is what is being substituted here. `mcp` is the desktop
/// surface — screenshots and clicks — and is a different capability that a
/// headless box does not even have.
fn sandbox_rpc_url(sandbox: &str) -> String {
    let host = std::env::var("SBOX_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port = std::env::var("SBOX_PORT").unwrap_or_else(|_| "3000".into());
    format!("http://{host}:{port}/api/orchestration/sandboxes/{sandbox}/exec/rpc")
}

/// The default when the client names no model. Matches the JS fallback chain
/// ending at the user's `~/.config/agl/config.yaml` `default_model`.
pub const DEFAULT_MODEL: &str = "llama-server:default";

/// What a turn needs.
pub struct TurnRequest {
    pub tab_id: String,
    pub content: String,
    pub agent: Option<String>,
    pub model: Option<String>,
    /// Force a brand-new session rather than resuming the tab's.
    pub new_session: bool,
    /// Resume a specific session (the history rail clicking an old one).
    pub session_id: Option<String>,
    /// Who said it — the seat's name in a multi-human room (PLAN44 A6).
    /// `None` is "the user" and renders as "You".
    pub from: Option<String>,
}

#[derive(Debug)]
pub enum TurnError {
    EmptyContent,
    Store(String),
    Agent(String),
    /// The user pressed stop. Not a failure: the turn did what it was told.
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

/// Map one hydrated context row back to a provider message.
///
/// Tool rows carry their call id; a tool-call row carries the calls. Rows the
/// provider has no use for (`reasoning`) are dropped here, mirroring
/// `messages_for_provider` in agl.
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
        _ => None, // reasoning is UI-only
    }
}

/// Resolve which session this turn belongs to, creating one if needed.
pub fn resolve_session(
    store: &ConvStore,
    tabs: &LiveTabs,
    req: &TurnRequest,
) -> Result<String, TurnError> {
    if !req.new_session {
        // An explicit resume wins over whatever the tab was on.
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

/// Give a fresh session a title from its first user message.
///
/// 64 characters plus an ellipsis, as the JS does — the history rail shows
/// this, so the truncation point is user-visible.
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

/// Register every tool the connected servers advertise.
///
/// Names are prefixed `file_io__read_file` style, matching the JS, so a
/// definition's `toolsEnabled` / `allowlist` entries carry over unchanged and
/// two servers cannot collide on a common name like `read_file`.
///
/// Every call is gated by the allowlist **before** it reaches the server,
/// and deny is the default: an empty allowlist permits nothing.
///
/// A call the list does not cover is not simply refused — it is **asked
/// about**. The handler blocks in [`approvals`](crate::approvals) while the
/// person decides, and only a denial (or a timeout, or nobody there to ask)
/// returns text the model can read, so it can choose another approach rather
/// than failing the turn.
///
/// `session` is what an "always" is pinned to: approvals do not outlive the
/// conversation they were given in.
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
    // `auto` is the documented escape hatch for a trusted project; anything
    // else (including the default `ask`) enforces the list.
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
                            // The turn stops here until somebody answers.
                            let decision = approvals.ask_cancellable(&session, &gate_name, &cmd, args, &stop);
                            if !decision.allowed() {
                                // Denied calls never reach the server. The
                                // model is told plainly, so it can say so or
                                // try something permitted instead of
                                // retrying the same call forever.
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

/// Session rules add to each agent's own rules, one rule per line.
fn combined_allowlist(agent: &str, session: Option<&str>) -> String {
    [agent, session.unwrap_or("")].into_iter()
        .filter(|s| !s.trim().is_empty())
        .collect::<Vec<_>>().join("\n")
}

/// Build an agl agent seeded with the session's prior conversation.
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

    // Hydrate from the durable log. Hidden rows stay hidden from the
    // provider but are not deleted — the eye toggle in the UI.
    for item in store.build_context_window_items(session_id) {
        if !item.visible {
            continue;
        }
        if let Some(msg) = item_to_message(&item) {
            agent.push_context(msg);
        }
    }
    Ok(agent)
}

/// Everything one agent turn emits, stamped so the UI can draw it as a run.
///
/// A "turn" here is **one inference request** — from the moment the prompt is
/// assembled to the moment the model stops. Every node it produces (the
/// system prompt, the prompt as the agent actually received it, each tool
/// call and result, the reasoning, the answer) carries the same `meta.turn`,
/// which is what lets the transcript group them under one rail instead of
/// showing a flat list where you cannot tell whose turn a tool call belonged
/// to.
///
/// Stamping the handle too is redundant with `participant_id` and worth it:
/// the reader is a template, and resolving a seat per row to draw a label is
/// work the writer can do once.
#[derive(Debug, Clone)]
pub struct TurnMark {
    pub id: String,
    pub handle: String,
    pub participant: Option<i64>,
    pub round: i64,
}

/// What the model has said since the last time anything was written down.
///
/// agl runs one turn as several rounds — think, call a tool, read the result,
/// think again — and returns only the final round's answer. Buffering all of
/// it and writing a single row when the turn settles put every thought
/// *below* the tool calls it actually preceded, and fused separate thoughts
/// into one card. Flushing at each tool-call boundary puts each one on the
/// rail where it happened.
///
/// Content said during a tool round is dropped by agl (only the last round's
/// answer comes back), so it is written here or nowhere.
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

    /// Seal the thinking so far, if there is any.
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

    /// Seal anything said aloud before a tool call — narration the model
    /// offered on its way to using a tool. A real assistant row, because the
    /// agent did say it, and the room's transcript should show it.
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

    /// Both, in the order they were produced: the model thinks, then speaks,
    /// then reaches for the tool.
    fn flush(&mut self, store: &ConvStore, conv: &str, mark: &TurnMark, emit: &mut dyn FnMut(&Value) -> bool) {
        self.flush_reasoning(store, conv, mark, emit);
        self.flush_content(store, conv, mark, emit);
    }

    /// At the end of a turn only the thinking is flushed: the final round's
    /// content is what `run_events` returns, and the caller writes that as
    /// the turn's answer. Flushing it here too would publish it twice.
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

    /// An event belonging to this turn.
    pub fn ev(&self, kind: &str, payload: Value) -> NewEvent {
        NewEvent {
            event_type: kind.into(),
            payload,
            meta: Some(self.meta()),
            speaker: self.participant,
            ..Default::default()
        }
    }

    /// An event belonging to this turn, carrying extra meta.
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

/// Record what the model was actually given, as two cards.
///
/// This is the whole point of the feature: when an agent behaves oddly, the
/// question is always "what did it actually receive?", and before this the
/// answer was not in the transcript at all. The system prompt is stored per
/// turn rather than once per conversation because it is *composed* — soul
/// plus goals plus memory, any of which may have been edited since the last
/// turn — so "the system prompt" is only meaningful as of a given turn.
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

/// Move any answered approvals into the transcript.
///
/// Called between a tool call and its result, which is exactly where the
/// question was asked: the card then sits above the result it decided,
/// rather than at the end of the turn where it would read as a footnote to
/// something that already happened.
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

/// Run one turn, emitting protocol rows through `emit`.
///
/// `emit` returns `false` when the client has gone away, which stops the
/// turn rather than finishing work nobody will see.
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
    // The agent's setting overrides the session's selected model.
    let model = Some(def.model.clone()).filter(|m| !m.trim().is_empty())
        .or_else(|| req.model.clone().filter(|m| !m.trim().is_empty()))
        .or_else(|| req.session_id.as_deref().and_then(|id| store.load(id))
            .and_then(|v| v.get("model").and_then(Value::as_str).map(String::from))
            .filter(|m| !m.trim().is_empty()))
        .unwrap_or_else(|| DEFAULT_MODEL.into());
    let session_id = resolve_session(store, tabs, &req)?;
    // Stoppable from here on. The session id is the key the stop button
    // uses, so this is the earliest point the turn *can* be registered —
    // and `Cancels::begin` hands back an already-thrown switch if the user
    // got there first, which is the race on a brand-new session.
    let stop = Armed::new(cancels, &session_id);
    // Name the turn before setup so Stop can address even a stalled connection.
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
    title_from_first_message(store, &session_id, &content);
    store.set_status(&session_id, "running");

    // The user's message is durable before any inference runs: if the model
    // call fails, the prompt is still in the transcript. `from` names the
    // human who said it — a multi-human room is unreadable without it.
    let mut user_payload = serde_json::json!({ "role": "user", "content": content });
    if let Some(from) = req.from.as_ref().filter(|f| !f.trim().is_empty()) {
        user_payload["from"] = json!(from.trim());
    }
    let _ = store.append_event(
        &session_id,
        NewEvent::new("context_window", user_payload),
    );

    // One mark for this whole turn, made before anything it covers — the
    // engagement decision included, so the decision card shares the turn's
    // id and the rail runs through it rather than leaving it orphaned.
    let me = store.agent_participants(&session_id).into_iter().next();
    let mark = TurnMark::new(
        me.as_ref().map(|p| p.handle.as_str()).unwrap_or(&def.name),
        me.as_ref().map(|p| p.id),
        0,
    );

    // A lone invoked agent is implicitly addressed by the user's message;
    // relevant agents still pass through their relevance decision.
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

    // Tools come up before the model is told they exist.
    let root = std::path::Path::new(project_root);
    // Where this agent's files and shell begin: its own workspace root if it
    // named one, else the project root.
    let file_root = file_root_for(def, root);
    // Which machine "the filesystem" means for this conversation.
    let sandbox = store
        .load(&session_id)
        .and_then(|s| s.get("sandboxId").and_then(|v| v.as_str().map(str::to_string)))
        .unwrap_or_default();
    let (servers, mcp_errors) = pool.ensure_in_sandbox_cancellable(&def.mcp, &file_root, &sandbox, &stop.switch);
    for e in &mcp_errors {
        if stop.stopped() { break; }
        let _ = store.append_event(&session_id, NewEvent::new("error", json!({ "message": e })));
        emit(&json!({ "type": "card", "card": { "kind": "error", "text": e } }));
    }

    // Timed from before the agent is built so the number covers the whole
    // turn the user waited through, not just the final inference.
    let started = std::time::Instant::now();
    // One agent or several, the prompt is built the same way — by
    // `group::plan`. Two separate builders is how the single-agent path came
    // to be sending only the latest message while the group path sent the
    // whole flow: the same bug cannot hide in one of two implementations if
    // there is only one.
    let task = me
        .as_ref()
        .and_then(|p| crate::group::plan(store, &session_id, p, 0, true, "The user addressed you."));
    let system = task
        .as_ref()
        .map(|t| t.system_prompt.clone())
        .unwrap_or_else(|| def.system_prompt());
    let prompt = task
        .as_ref()
        .map(|t| t.prompt.clone())
        .unwrap_or_else(|| content.clone());
    let mut agent = Agent::factory(Factory {
        model: Some(model.clone()),
        reasoning_effort: turn_reasoning(def, store.load(&session_id).as_ref()),
        system_prompt: Some(system.clone()).filter(|p| !p.trim().is_empty()),
        retain_history: Some(true),
        parallel_tools: Some(true),
        // How many model round-trips this turn may take. `None` leaves
        // agl's default of 5, which is three tool calls and an answer —
        // enough for a chat reply, not enough for an agent that has to go
        // and look something up first.
        max_turns: def.max_turns.map(|v| v as usize),
        ..Default::default()
    })
    .map_err(|e| TurnError::Agent(e.to_string()))?;
    match &task {
        Some(t) => {
            for msg in &t.context {
                agent.push_context(msg.clone());
            }
        }
        // No agent seated — only reachable by calling this directly. Fall
        // back to hydrating the raw transcript, as it always did.
        None => {
            for item in store.build_context_window_items(&session_id) {
                if !item.visible {
                    continue;
                }
                if let Some(msg) = item_to_message(&item) {
                    agent.push_context(msg);
                }
            }
        }
    }
    // Session rules supplement this agent's own rules.
    let session_state = store.load(&session_id);
    let overridden = session_state
        .as_ref()
        .and_then(|s| s.get("allowlist"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let rules = combined_allowlist(&def.allowlist, overridden.as_deref());

    // `unrestricted` is a temporary override, NOT an edit: it runs the turn
    // with `*` while leaving the rules exactly as written, so switching back
    // restores them. Storing `*` into `allowlist` would destroy them.
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
    // The coordinator family (chat__sessions_* and friends): same gates as
    // every other tool — registered only if the agent's tool list allows,
    // each call checked against the allowlist or the approval queue.
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
        // `allowlist` is the rules as written; `effectiveAllowlist` is what
        // actually gated this turn. They differ only in unrestricted mode.
        "allowlist": rules,
        "effectiveAllowlist": allow_text,
        "allowlistMode": mode,
        "allowlistSource": if overridden.is_some() { "agent + session" } else { "agent" },
        "allowlistOverridden": overridden.is_some(),
        "policyMode": def.policy_mode,
        "projectRoot": project_root,
    }));
    let _ = tool_count;

    // What the model is about to be given, on the record.
    let history_turns = task
        .as_ref()
        .map(|t| t.context.len())
        .unwrap_or_else(|| store.build_context_window_items(&session_id).len());
    // Everything the model gets, not just the final turn: the card is the
    // answer to "what did it actually receive?".
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

    // ---- the inference ----------------------------------------------------
    let mut said = Said::new(mark.participant);
    // Time to first token: the latency the user actually feels before
    // anything appears, which total duration hides entirely.
    let mut first_token_at: Option<std::time::Duration> = None;
    let answer = {
        let sid = session_id.clone();
        // Named on every delta, not only on the sealed card: the live
        // reasoning card picks its colour from this, and without it the
        // turn streams in the default accent and only snaps to the agent's
        // once it finishes.
        let who = mark.handle.clone();
        let mut on_event = |e: TurnEvent<'_>| match e {
            // The turn queue: this turn is waiting for a free slot of its
            // `provider:model`. Durable and on screen immediately — the
            // user must see why nothing is streaming yet.
            TurnEvent::Queued { spec, position } => {
                let w = store.append_event(
                    &sid,
                    mark.ev("queued", json!({ "spec": spec, "position": position })),
                );
                let event_id = w.map(|x| x.event_id).unwrap_or_default();
                emit(&json!({
                    "type": "card",
                    "card": { "kind": "queued", "text": format!("waiting for {spec} slot…"),
                              "spec": spec, "position": position,
                              "eventId": event_id, "turn": mark.id, "handle": who,
                              "visible": true },
                }));
            }
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
            // A tool call is durable and on screen the moment it is made,
            // not after the turn settles — a long shell command would
            // otherwise look like a hang.
            TurnEvent::ToolCall { id, name, args } => {
                // Before the call, so the thought that led to it sits above
                // it on the rail instead of below every tool in the turn.
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
                // Whatever was asked while this call was gated belongs
                // above its result, on the record.
                flush_approvals(store, &sid, &mark, approvals, emit);
                // A denial is a failure from the UI's point of view, even
                // though it is a well-formed result the model can read.
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
            agent.run_events(&prompt, true, &mut on_event),
        )
    };

    let answer = match answer {
        // Stopped mid-request. What it had already said stays: the user
        // asked for the rest not to be generated, not for the half they
        // had already read to be taken off the record.
        Ran::Stopped => {
            said.flush(store, &session_id, &mark, emit);
            let _ = store.append_event(
                &session_id,
                mark.ev("stopped", json!({ "reason": "user" })),
            );
            // `record_turn_complete` sets the status ready itself, and it
            // counts the turn — which this was: the user spent one, and the
            // agent spent part of one answering it.
            store.record_turn_complete(&session_id, None);
            emit(&json!({
                "type": "card",
                "card": { "kind": "system", "text": STOPPED_NOTICE, "visible": true },
            }));
            // `ok` because nothing went wrong — the composer needs to come
            // back out of `waiting` the same way it does on a good turn.
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

    // ---- persist + finalize ----------------------------------------------
    said.finish(store, &session_id, &mark, emit);

    let usage = agent.last_usage.lock().unwrap().clone();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    // Generation rate over the whole turn. Tool execution is included, so a
    // tool-heavy turn reads slower than the model actually generated — the
    // honest number for "how long did this take me", which is what the
    // metric is for.
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

    // `gen_info` MUST be appended before the assistant row it describes:
    // `reconstruct_messages_from_log` carries `lastGen` FORWARD onto later
    // rows, so a gen_info written afterwards would never reach the message
    // and a reloaded transcript would show no metrics.
    let _ = store.append_event(&session_id, {
        let mut ev = mark.ev_with(
            "gen_info",
            telemetry.clone(),
            &[("telemetry", telemetry.clone())],
        );
        // The wrapper's own duration, which the metrics row reads back.
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
    // Always finalize the bubble, even when empty, so a client streaming into
    // a live bubble knows the turn is sealed.
    //
    // The handle rides along so the live card says the same thing the
    // reloaded transcript will: `reconstruct_messages` reads it from the
    // node's author, and a card without one would briefly be anonymous.
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

#[cfg(test)]
mod tests {
    /// End to end against a **live** sandbox:
    /// `SBOX_TEST_ID=<ready id> cargo test -- --ignored swapped_for_sandbox`
    ///
    /// The thing worth proving is not that a list was filtered — it is that
    /// what the model ends up holding are the *guest's* tools, under names
    /// that say so, with the host's file and shell tools gone.
    #[test]
    #[ignore]
    fn a_sandbox_is_the_whole_session_s_files_and_shell() {
        let Ok(id) = std::env::var("SBOX_TEST_ID") else {
            panic!("set SBOX_TEST_ID to a ready sandbox");
        };
        let names: Vec<String> = ["file-io", "term-local"].iter().map(|s| s.to_string()).collect();
        let root = std::env::var("ANGELA_ROOT").unwrap_or_else(|_| "/workspace/psicom/chat".into());
        let pool = McpPool::default();

        let (plain, _) = pool.ensure(&names, std::path::Path::new(&root));
        let host: Vec<String> = plain.iter().map(|s| s.name.clone()).collect();
        assert!(host.contains(&"file-io".to_string()), "host servers were {host:?}");

        let (swapped, errors) =
            pool.ensure_in_sandbox(&names, std::path::Path::new(&root), &id);
        assert!(errors.is_empty(), "{errors:?}");
        let got: Vec<String> = swapped.iter().map(|s| s.name.clone()).collect();
        assert_eq!(got, vec!["sandbox".to_string()], "servers were {got:?}");

        let tools: Vec<String> = swapped[0]
            .tools
            .iter()
            .map(|t| crate::mcp_presets::prefixed_tool_name("sandbox", &t.name))
            .collect();
        assert!(tools.contains(&"sandbox__bash".to_string()), "{tools:?}");
        assert!(tools.contains(&"sandbox__read".to_string()), "{tools:?}");
        // The host's are gone, not merely shadowed.
        assert!(!tools.iter().any(|t| t.starts_with("file_io__")), "{tools:?}");
        assert!(!tools.iter().any(|t| t.starts_with("term_local__")), "{tools:?}");

        // Session-wide, not per-agent: a participant whose own `mcp` list
        // named no file or shell server still works in the session's
        // sandbox, because the sandbox is the session's filesystem and the
        // participant is in the session.
        let others: Vec<String> = ["web-researcher"].iter().map(|s| s.to_string()).collect();
        let (also, errors) =
            pool.ensure_in_sandbox(&others, std::path::Path::new(&root), &id);
        assert!(errors.is_empty(), "{errors:?}");
        let got: Vec<String> = also.iter().map(|s| s.name.clone()).collect();
        assert!(got.contains(&"sandbox".to_string()), "servers were {got:?}");
        assert!(got.contains(&"web-researcher".to_string()), "servers were {got:?}");
    }

    /// The swap is about *which machine*, so it must take exactly the two
    /// servers that mean "this machine's files and shell" and leave
    /// everything else alone. A browser or a researcher is not a filesystem.
    #[test]
    fn a_sandbox_displaces_the_host_file_and_shell_servers_and_nothing_else() {
        let names: Vec<String> = ["file-io", "term-local", "web-researcher", "camera"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let kept: Vec<&String> = names
            .iter()
            .filter(|n| !super::DISPLACED_BY_SANDBOX.contains(&n.as_str()))
            .collect();
        assert_eq!(kept, vec!["web-researcher", "camera"]);
    }

    /// The guest's file-and-shell daemon, not the desktop surface: a
    /// headless sandbox has no desktop, and `read`/`write`/`bash` are what
    /// is being substituted.
    #[test]
    fn the_sandbox_url_points_at_the_exec_facade() {
        let u = super::sandbox_rpc_url("sandbox2-abc");
        assert!(u.ends_with("/api/orchestration/sandboxes/sandbox2-abc/exec/rpc"), "{u}");
        assert!(!u.contains("/mcp/"), "{u}");
    }

    /// With no sandbox assigned nothing is displaced — the overwhelming
    /// majority of conversations, and the path that must not change.
    #[test]
    fn no_sandbox_means_the_ordinary_server_list() {
        let pool = McpPool::default();
        let (servers, errors) =
            pool.ensure_in_sandbox(&[], std::path::Path::new("/tmp"), "   ");
        assert!(servers.is_empty());
        assert!(errors.is_empty());
    }

    use super::*;

    #[test]
    fn stop_interrupts_wait_for_pool_and_cancelled_setup_is_not_cached() {
        let pool = Arc::new(McpPool::default());
        let held = pool.servers.lock().unwrap();
        let stop = Cancel::never();
        let worker_stop = stop.clone();
        let worker_pool = pool.clone();
        let worker = std::thread::spawn(move || worker_pool.get_or_connect("slow", &worker_stop, || Err("should not connect".into())).err().unwrap());
        std::thread::sleep(Duration::from_millis(75));
        stop.stop();
        assert_eq!(worker.join().unwrap(), "stopped");
        drop(held);

        let stop = Cancel::never();
        let result = pool.get_or_connect("slow", &stop, || { stop.stop(); Err("stopped".into()) });
        assert!(result.is_err());
        let retry = pool.get_or_connect("slow", &Cancel::never(), || Err("retried".into()));
        assert_eq!(retry.err().unwrap(), "retried");
    }

    #[test]
    fn a_closed_stdio_server_is_reconnected_on_next_use() {
        let pool = McpPool::default();
        let spec = crate::mcp::ServerSpec {
            name: "reconnect-test".into(), command: "python3".into(),
            args: vec!["-u".into(), "-c".into(), r#"import sys,json
for line in sys.stdin:
 m=json.loads(line)
 if 'id' in m: print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{'tools':[]}}), flush=True)
"#.into()], cwd: None, env: vec![], framing: crate::mcp::Framing::Newline,
        };
        let connect = || McpServer::connect(&spec, Duration::from_secs(2));
        let first = pool.get_or_connect("server", &Cancel::never(), connect).unwrap();
        first.close();
        let second = pool.get_or_connect("server", &Cancel::never(), connect).unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(second.is_running());
    }

    #[test]
    fn concurrent_turns_initialize_each_server_only_once() {
        let pool = Arc::new(McpPool::default());
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut threads = Vec::new();
        for _ in 0..8 {
            let pool = Arc::clone(&pool);
            let attempts = Arc::clone(&attempts);
            threads.push(std::thread::spawn(move || {
                pool.get_or_connect("shared", &Cancel::never(), || {
                    attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Err("startup failed".into())
                }).err().unwrap()
            }));
        }
        for thread in threads { assert_eq!(thread.join().unwrap(), "startup failed"); }
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn empty_mcp_selection_connects_no_servers() {
        let (servers, errors) = McpPool::default().ensure(&[], std::path::Path::new("/tmp"));
        assert!(servers.is_empty());
        assert!(errors.is_empty());
    }

    #[test]
    fn agent_reasoning_overrides_session_reasoning() {
        let mut def = AgentDef::default();
        let session = json!({"config": {"reasoningEffort": "low"}});
        assert_eq!(turn_reasoning(&def, Some(&session)).as_deref(), Some("low"));
        def.reasoning_effort = "xhigh".into();
        assert_eq!(turn_reasoning(&def, Some(&session)).as_deref(), Some("xhigh"));
        def.reasoning_effort.clear();
        assert_eq!(turn_reasoning(&def, None), None);
    }

    #[test]
    fn session_rules_add_to_each_agents_own_rules() {
        let researcher = combined_allowlist("web_researcher__search", Some("file_io__read_file"));
        let joker = combined_allowlist("", Some("file_io__read_file"));
        assert!(allowlist::is_allowed(&researcher, "web_researcher__search", ""));
        assert!(allowlist::is_allowed(&researcher, "file_io__read_file", ""));
        assert!(allowlist::is_allowed(&joker, "file_io__read_file", ""));
        assert!(!allowlist::is_allowed(&joker, "web_researcher__search", ""));
        assert!(allowlist::is_allowed(&combined_allowlist("web_researcher__search", None), "web_researcher__search", ""));
    }

    fn store() -> (tempfile::TempDir, ConvStore) {
        let tmp = tempfile::tempdir().unwrap();
        let s = ConvStore::memory(tmp.path()).unwrap();
        (tmp, s)
    }

    #[test]
    fn a_fresh_tab_gets_a_new_session_and_reuses_it() {
        let (_t, s) = store();
        let tabs = LiveTabs::default();
        let req = TurnRequest {
            tab_id: "1".into(),
            content: "hi".into(),
            agent: None,
            model: None,
            new_session: false,
            session_id: None,
            from: None,
        };
        let a = resolve_session(&s, &tabs, &req).unwrap();
        tabs.set(
            "1",
            TabEntry {
                session_id: a.clone(),
                ..Default::default()
            },
        );
        let b = resolve_session(&s, &tabs, &req).unwrap();
        assert_eq!(a, b, "the tab stays on its session");
    }

    #[test]
    fn new_session_forces_a_fresh_one() {
        let (_t, s) = store();
        let tabs = LiveTabs::default();
        let first = resolve_session(
            &s,
            &tabs,
            &TurnRequest {
                tab_id: "1".into(),
                content: "hi".into(),
                agent: None,
                model: None,
                new_session: false,
                session_id: None,
                from: None,
            },
        )
        .unwrap();
        tabs.set(
            "1",
            TabEntry {
                session_id: first.clone(),
                ..Default::default()
            },
        );
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let second = resolve_session(
            &s,
            &tabs,
            &TurnRequest {
                tab_id: "1".into(),
                content: "hi".into(),
                agent: None,
                model: None,
                new_session: true,
                session_id: None,
                from: None,
            },
        )
        .unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn an_explicit_session_id_resumes_it() {
        let (_t, s) = store();
        let tabs = LiveTabs::default();
        let target = s.create(NewSession::default()).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        let got = resolve_session(
            &s,
            &tabs,
            &TurnRequest {
                tab_id: "9".into(),
                content: "hi".into(),
                agent: None,
                model: None,
                new_session: false,
                session_id: Some(target.clone()),
                from: None,
            },
        )
        .unwrap();
        assert_eq!(got, target);
    }

    #[test]
    fn an_unknown_session_id_falls_through_to_a_new_one() {
        let (_t, s) = store();
        let tabs = LiveTabs::default();
        let got = resolve_session(
            &s,
            &tabs,
            &TurnRequest {
                tab_id: "9".into(),
                content: "hi".into(),
                agent: None,
                model: None,
                new_session: false,
                session_id: Some("does-not-exist".into()),
                from: None,
            },
        )
        .unwrap();
        assert!(s.load(&got).is_some());
    }

    #[test]
    fn the_title_comes_from_the_first_message_and_sticks() {
        let (_t, s) = store();
        let id = s.create(NewSession::default()).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        title_from_first_message(&s, &id, "Migrate the auth service");
        assert_eq!(s.load(&id).unwrap()["title"], "Migrate the auth service");
        // a second message must not rename the session
        title_from_first_message(&s, &id, "something else");
        assert_eq!(s.load(&id).unwrap()["title"], "Migrate the auth service");
    }

    #[test]
    fn a_long_title_is_truncated_with_an_ellipsis() {
        let (_t, s) = store();
        let id = s.create(NewSession::default()).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        let long = "x".repeat(100);
        title_from_first_message(&s, &id, &long);
        let t = s.load(&id).unwrap()["title"].as_str().unwrap().to_string();
        assert_eq!(t.chars().count(), 65, "64 chars + the ellipsis");
        assert!(t.ends_with('…'));
    }

    #[test]
    fn titles_truncate_by_character_not_byte() {
        let (_t, s) = store();
        let id = s.create(NewSession::default()).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        // multi-byte input must not be cut mid-character
        title_from_first_message(&s, &id, &"é".repeat(100));
        let t = s.load(&id).unwrap()["title"].as_str().unwrap().to_string();
        assert_eq!(t.chars().count(), 65);
    }

    #[test]
    fn hydration_skips_hidden_rows_and_reasoning() {
        let (_t, s) = store();
        let id = s.create(NewSession::default()).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        s.append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"one"}))).unwrap();
        let hidden = s
            .append_event(&id, NewEvent::new("context_window", json!({"role":"assistant","content":"hidden"})))
            .unwrap();
        s.append_event(&id, NewEvent::new("reasoning", json!({"text":"ui only"}))).unwrap();
        s.append_event(&id, NewEvent::new("context_meta", json!({"event_id": hidden.event_id, "visible": false}))).unwrap();

        let msgs: Vec<_> = s
            .build_context_window_items(&id)
            .iter()
            .filter(|i| i.visible)
            .filter_map(item_to_message)
            .collect();
        assert_eq!(msgs.len(), 1, "hidden and reasoning rows are not sent");
        assert_eq!(msgs[0].role, "user");
    }
}

// ===========================================================================
// Group conversations: more than one agent seated in the same room
// ===========================================================================

/// Run a whole discussion: one user message, then every addressed
/// participant speaks, then anyone they pulled in, until the round settles.
///
/// This is the N-participant path. It exists beside [`run_turn`] rather than
/// replacing it because the two have genuinely different shapes: with one
/// agent the provider-side history *is* the transcript, and that is both
/// cheaper and what every existing session already contains. With two or
/// more, no such shared history can exist (see [`crate::group`]) — each
/// agent gets its own stable system prompt, its own private history, and the
/// room as flattened text. `/api/chat` picks by roster size, so a two-seat
/// conversation behaves exactly as it did.
///
/// Participants speak **one at a time, in seat order**, and each one's
/// prompt is assembled after the previous reply is durable. That sequencing
/// is the only reason a later speaker can see an earlier one: the room log
/// is the channel between them.
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
    // One switch for the whole round, not one per speaker: "stop" means stop
    // the conversation, not hand the floor to the next agent.
    let stop = Armed::new(cancels, &conv);
    tabs.set(
        &req.tab_id,
        TabEntry {
            session_id: conv.clone(),
            agent: req.agent.clone(),
            model: req.model.clone().unwrap_or_default(),
            ..Default::default()
        },
    );
    title_from_first_message(store, &conv, &content);
    store.set_status(&conv, "running");

    // The user's message is durable before any model runs. `from` names the
    // human who said it (PLAN44 A6) — same rule as the single-agent path.
    let mut user_payload = serde_json::json!({ "role": "user", "content": content });
    if let Some(from) = req.from.as_ref().filter(|f| !f.trim().is_empty()) {
        user_payload["from"] = json!(from.trim());
    }
    let user_node = store
        .append_event(&conv, NewEvent::new("context_window", user_payload))
        .map_err(|e| TurnError::Store(e.to_string()))?;

    let roster = store.participants(&conv);
    let header = store.load(&conv).unwrap_or(Value::Null);

    // Nobody to answer. The message is already durable, so this is a state
    // the user can fix and then re-send from — but silence with no
    // explanation would read as a broken app, so it is stated plainly.
    // Unless the room has other humans in it: a DM is not a broken agent
    // queue (PLAN44 A4), and a "no agents" card in a chat is noise.
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

    // --- round 0 responders -------------------------------------------------
    //
    // Who speaks is each agent's own `engagement`, not a property of the
    // message: `engaged` always, `invoked` only when named, `relevant` when
    // a System One call says this one is theirs. A mention beats all three.
    //
    // Classification only here; a `relevant` agent's System One call happens
    // immediately before its own turn, further down. Two reasons: the
    // decision then sees what the agents ahead of it have just said, and
    // the decision card lands next to the turn it decided — same turn id,
    // contiguous rows, one rail — instead of orphaned several turns above.
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

    // The loop yields *why* it stopped, so there is no window in which
    // "settled" holds a value nobody chose.
    let settled = 'rounds: loop {
        // Checked at the top of every round as well as inside each speaker's
        // request: a stop that lands between two agents must not buy the
        // room another round.
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
            // Directly addressed in round 0 must answer; a later-round
            // mention is an invitation, and "(pass)" is a legitimate reply.
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
            // One mark for the whole of this participant's turn, made here
            // so the reply it publishes lands on the same rail as the
            // decision, prompt, reasoning and tool calls around it.
            let mark = TurnMark::new(&me.handle, Some(me.id), round_index);

            // A `relevant` agent is asked now, not up front: it then sees
            // what the agents ahead of it just said, and its decision card
            // shares this turn's id so the rail runs through it.
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
            // A participant's own workspace root, or the project root when it
            // named none. The sandbox, when assigned, supersedes it for file
            // and shell access — this is just the root the participant's
            // own servers expand.
            let file_root = file_root_for(&participant_def, root);
            // The sandbox belongs to the conversation, not to the agent, so
            // every participant in a group works in the same box.
            let sandbox = store
                .load(&conv)
                .and_then(|s| s.get("sandboxId").and_then(|v| v.as_str().map(str::to_string)))
                .unwrap_or_default();
            let (servers, mcp_errors) =
                pool.ensure_in_sandbox_cancellable(&participant_def.mcp, &file_root, &sandbox, &stop.switch);
            for e in &mcp_errors {
                if stop.stopped() { break; }
                let _ = store.append_event(&conv, mark.ev("error", json!({ "message": e })));
                emit(&json!({ "type": "card", "card": { "kind": "error", "text": e } }));
            }
            let reply = match run_participant(
                app, store, rt, &stop.switch, approvals, &servers, &participant_def, &conv, &me,
                &task, &model, &mark, emit,
            ) {
                Ok(r) => r,
                // Not an error card: the user did this on purpose. Leaving
                // by the label rather than by `break` keeps the recorded
                // reason "stopped" — a plain break falls through to the
                // next round's own bookkeeping, which would call a stopped
                // discussion a silent one.
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

            // Its own history shows it stamped, labelled lines; a model that
            // copies that form gets the copy removed here, before anything
            // else reads the reply — `is_pass` included, or "17:43 [fish]
            // (pass)" would be published as a message instead of silence.
            let reply = crate::group::strip_echoed_prefix(&reply, &me.handle);

            // A pass is silence, not an empty message: publishing it would
            // clutter the room and show every other agent a blank turn.
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
            // The prompt is stored with the reply so this participant's own
            // history is byte-stable on the next turn, whatever the room does.
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
    // Settle every participant too, not just the header: a view left
    // `running` would make a restarted server think a round is still in
    // flight and wait for a reply nobody is generating.
    for p in roster.iter().filter(|p| p.is_agent()) {
        let (seq, _, _, _) = store.participant_view(&conv, p.id);
        store.set_participant_view(&conv, p.id, seq, round_index, false, None, Some(s.status()));
    }
    store.record_turn_complete(&conv, None);
    // Said once for the whole discussion, not once per speaker: the user
    // pressed one button, and a row per agent would read like several
    // separate failures.
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

/// Advance a participant's watermark past what it has now seen.
fn advance(store: &ConvStore, conv: &str, pid: &i64, seq: i64, round: i64, task_id: &str) {
    store.set_participant_view(conv, *pid, seq, round, false, Some(task_id), Some("running"));
}

/// One participant's inference, in its own context.
///
/// The agent is built with **that participant's** system prompt, seeded with
/// **that participant's** own `(prompt, reply)` pairs, and asked the
/// flattened room prompt. Nothing from the shared transcript is pushed as a
/// role, which is the invariant the whole group design rests on.
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
    mark: &TurnMark,
    emit: &mut dyn FnMut(&Value) -> bool,
) -> Result<String, TurnError> {
    let system = if task.system_prompt.trim().is_empty() {
        // Fall back to the agent definition, with the participant's identity
        // appended: an agent that does not know its own handle cannot be
        // told apart from its peers in the flattened room.
        let base = def.system_prompt();
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
        reasoning_effort: turn_reasoning(def, store.load(conv).as_ref()),
        system_prompt: Some(system.clone()),
        // The history is pushed explicitly below, and must not also
        // accumulate across participants.
        retain_history: Some(true),
        parallel_tools: Some(true),
        // How many model round-trips this turn may take. `None` leaves
        // agl's default of 5, which is three tool calls and an answer —
        // enough for a chat reply, not enough for an agent that has to go
        // and look something up first.
        max_turns: def.max_turns.map(|v| v as usize),
        ..Default::default()
    })
    .map_err(|e| TurnError::Agent(e.to_string()))?;
    // The conversation as this participant sees it: its own words as its
    // own turns, everyone else's as `[sender]`-prefixed user turns.
    for msg in &task.context {
        agent.push_context(msg.clone());
    }

    let session_state = store.load(conv);
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
    // Coordinators take part in group rooms too — the same tools, gates,
    // and (later) the same owner-ACL check as the single-agent path.
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

    // What this participant was actually given, on the record — the
    // flattened room prompt, not the user's typed message. They are
    // different things, and only one of them explains the reply.
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
            // Same turn queue as the single-agent path: waiting for a slot.
            TurnEvent::Queued { spec, position } => {
                let w = store.append_event(
                    conv,
                    mark.ev("queued", json!({ "spec": spec, "position": position })),
                );
                let event_id = w.map(|x| x.event_id).unwrap_or_default();
                emit(&json!({
                    "type": "card",
                    "card": { "kind": "queued", "text": format!("waiting for {spec} slot…"),
                              "spec": spec, "position": position,
                              "eventId": event_id, "participantId": pid,
                              "turn": mark.id, "handle": handle, "visible": true },
                }));
            }
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
                // Before the call, so the thought that led to it sits above
                // it on the rail instead of below every tool in the turn.
                // A second call in the same round finds the buffers empty.
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
                // Same as the single-agent path: the question that gated
                // this call goes above its result.
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
            agent.run_events(&task.prompt, true, &mut on_event),
        )
    };

    let answer = match answer {
        // Whatever this participant had already said is kept, then the stop
        // is handed up: the room must not move on to the next speaker.
        Ran::Stopped => {
            said.flush(store, conv, mark, emit);
            return Err(TurnError::Stopped);
        }
        Ran::Done(r) => r.map_err(|e| TurnError::Agent(e.to_string()))?,
    };

    said.finish(store, conv, mark, emit);
    Ok(answer)
}

/// Lead routing: ask the designated lead who should answer, parse its JSON.
///
/// On an unparseable reply the lead answers alone rather than the message
/// going nowhere — a routing failure must not swallow the user's turn.
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
    let header = store.load(conv).unwrap_or(Value::Null);
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
        // A stopped routing call assigns nobody, and the round-level check
        // ends the discussion before anyone is asked to speak.
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


/// Round-0 invitations: who is a candidate, and on what grounds.
///
/// Pure classification — no model calls. The `relevant` agents' System One
/// call happens later, immediately before each one's own turn, so that the
/// decision sees what the agents ahead of it said and its card lands on the
/// same rail as the turn it decided.
fn classify(
    store: &ConvStore,
    roster: &[crate::conv_store::Participant],
    content: &str,
    emit: &mut dyn FnMut(&Value) -> bool,
) -> Vec<(i64, crate::group::Invite)> {
    let agents = store.agents();
    // A participant's own `engagement` is a per-conversation override; with
    // none, the agent definition's current value wins, so editing an agent
    // changes how it behaves in conversations it is already in.
    let engagement_of = |p: &crate::conv_store::Participant| {
        if let Some(over) = p.engagement.as_deref().filter(|s| !s.trim().is_empty()) {
            return crate::agent_loader::Engagement::parse(over);
        }
        p.agent_name
            .as_deref()
            .and_then(|n| agents.get(n))
            .map(|d| d.engagement())
            // No definition at all (a seat whose agent was deleted, or one
            // added by hand): answer, rather than go mute for no stated
            // reason.
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

/// Ask System One whether this agent should take the turn it is about to
/// take. `true` means go ahead.
///
/// The verdict gets a card on **this turn's** mark, so it reads as the first
/// step of the turn rather than as an unrelated row — and when the answer is
/// no, as a turn that stopped there.
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
    // The conversation as the decider reads it: `[sender] text`, oldest
    // first, ending with the message being judged. Attribution is the point
    // — "has this already been said, and was it me who said it" is most of
    // what the decision turns on.
    let user_label = store.user_label();
    let mut history: Vec<String> = Vec::new();
    let mut headed: Option<(i32, i32)> = None;
    for n in store.path(conv).iter() {
        if n.kind != "context_window" || !n.visible {
            continue;
        }
        let Some(text) = n.content.clone() else { continue };
        let sender = match n.role.as_deref() {
            // The same label the turn's own prompt uses: an agent told the
            // speaker is "Bob" in one and "user" in the other is being
            // asked about a different conversation.
            Some("user") => user_label.clone(),
            _ => n
                .participant_id
                .and_then(|id| roster.iter().find(|p| p.id == id))
                .map(|p| p.handle.clone())
                .unwrap_or_else(|| "assistant".into()),
        };
        // Same day headers and clocks as the turn's own prompt: the two have
        // to describe one conversation.
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

    // The same identity the agent itself is given — soul, goals, memory —
    // so the decision is made on the same picture the reply would be.
    let identity = me
        .agent_name
        .as_deref()
        .and_then(|n| store.agents().get(n))
        .map(|d| d.system_prompt())
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
