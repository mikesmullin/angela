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
use crate::conv_store::ConvStore;
use crate::session_store::{ContextItem, NewEvent, NewSession};

/// How long one tool call may take before the turn gives up on it.
const TOOL_TIMEOUT: Duration = Duration::from_secs(120);
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

/// Connected MCP servers, started on demand and reused across turns.
///
/// Spawning a server per turn would pay process startup on every prompt and
/// throw away whatever state the server holds; the JS keeps them on the live
/// tab for the same reason.
#[derive(Default)]
pub struct McpPool {
    servers: Mutex<HashMap<String, Arc<McpServer>>>,
}

impl McpPool {
    /// Connect (or reuse) the named presets, returning those that came up.
    ///
    /// A server that fails to start is reported and skipped rather than
    /// failing the turn: losing shell access should not cost you the chat.
    pub fn ensure(
        &self,
        names: &[String],
        project_root: &std::path::Path,
    ) -> (Vec<Arc<McpServer>>, Vec<String>) {
        let mut out = Vec::new();
        let mut errors = Vec::new();
        for name in names {
            if let Some(existing) = self.servers.lock().unwrap().get(name) {
                out.push(Arc::clone(existing));
                continue;
            }
            let Some(spec) = mcp_presets::preset(name, project_root) else {
                errors.push(format!("unknown or uninstalled MCP preset `{name}`"));
                continue;
            };
            match McpServer::connect(&spec, CONNECT_TIMEOUT) {
                Ok(srv) => {
                    self.servers
                        .lock()
                        .unwrap()
                        .insert(name.clone(), Arc::clone(&srv));
                    out.push(srv);
                }
                Err(e) => errors.push(format!("MCP `{name}`: {e}")),
            }
        }
        (out, errors)
    }

    pub fn shutdown(&self) {
        for (_, s) in self.servers.lock().unwrap().drain() {
            s.close();
        }
    }
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
}

#[derive(Debug)]
pub enum TurnError {
    EmptyContent,
    Store(String),
    Agent(String),
}

impl std::fmt::Display for TurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TurnError::EmptyContent => write!(f, "empty content"),
            TurnError::Store(m) => write!(f, "session store: {m}"),
            TurnError::Agent(m) => write!(f, "{m}"),
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
/// Every call is gated by the allowlist **before** it reaches the server. A
/// denied call returns text the model can read, so it can choose a different
/// approach rather than failing the turn — and, critically, deny is the
/// default: an empty allowlist permits nothing.
pub fn register_tools(
    agent: &mut Agent,
    servers: &[Arc<McpServer>],
    enabled: Option<&Vec<String>>,
    allow_text: &str,
    policy_mode: &str,
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
            let ok = agent.tool(
                &name,
                &tool.description,
                props,
                required,
                move |args: &Value| {
                    if enforce {
                        let cmd = allowlist::command_line(&gate_name, args);
                        if !allowlist::is_allowed(&allow_text, &gate_name, &cmd) {
                            // Denied calls never reach the server.
                            return Ok(format!(
                                "denied: `{gate_name}`{} is not on this session's allowlist. \
                                 Ask the user to allow it, or use a permitted tool.",
                                if cmd.is_empty() {
                                    String::new()
                                } else {
                                    format!(" with `{cmd}`")
                                }
                            ));
                        }
                    }
                    match srv.call_tool(&remote, args, TOOL_TIMEOUT) {
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

/// Run one turn, emitting protocol rows through `emit`.
///
/// `emit` returns `false` when the client has gone away, which stops the
/// turn rather than finishing work nobody will see.
#[allow(clippy::too_many_arguments)]
pub fn run_turn(
    store: &ConvStore,
    tabs: &LiveTabs,
    rt: &tokio::runtime::Runtime,
    project_root: &str,
    pool: &McpPool,
    def: &AgentDef,
    req: TurnRequest,
    emit: &mut dyn FnMut(&Value) -> bool,
) -> Result<(), TurnError> {
    let content = req.content.trim().to_string();
    if content.is_empty() {
        return Err(TurnError::EmptyContent);
    }
    // Precedence: the request, then the agent definition, then the default.
    // An empty model is meaningful — it lets agl resolve `default_model`
    // from its config on every inference.
    let model = req
        .model
        .clone()
        .filter(|m| !m.trim().is_empty())
        .or_else(|| Some(def.model.clone()).filter(|m| !m.trim().is_empty()))
        .unwrap_or_else(|| DEFAULT_MODEL.into());
    let session_id = resolve_session(store, tabs, &req)?;

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
    // call fails, the prompt is still in the transcript.
    let _ = store.append_event(
        &session_id,
        NewEvent::new(
            "context_window",
            json!({ "role": "user", "content": content }),
        ),
    );

    let context_window = store
        .load(&session_id)
        .and_then(|s| s.get("contextWindow").and_then(|v| v.as_u64()));

    // Tools come up before the model is told they exist.
    let root = std::path::Path::new(project_root);
    let (servers, mcp_errors) = pool.ensure(&def.mcp, root);
    for e in &mcp_errors {
        let _ = store.append_event(&session_id, NewEvent::new("error", json!({ "message": e })));
        emit(&json!({ "type": "card", "card": { "kind": "error", "text": e } }));
    }

    // Timed from before the agent is built so the number covers the whole
    // turn the user waited through, not just the final inference.
    let started = std::time::Instant::now();
    let mut agent = build_agent(
        store,
        &session_id,
        &model,
        Some(def.system.clone()).filter(|p| !p.trim().is_empty()),
    )?;
    // The session's own allowlist (set from the UI) overrides the agent
    // definition's; `allowlistSource` records which one is in force.
    let session_state = store.load(&session_id);
    let overridden = session_state
        .as_ref()
        .and_then(|s| s.get("allowlist"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let rules = overridden.clone().unwrap_or_else(|| def.allowlist.clone());

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
        "allowlistSource": if overridden.is_some() { "ui" } else { "agent" },
        "allowlistOverridden": overridden.is_some(),
        "policyMode": def.policy_mode,
        "projectRoot": project_root,
    }));
    let _ = tool_count;

    // ---- the inference ----------------------------------------------------
    let mut reasoning = String::new();
    // Time to first token: the latency the user actually feels before
    // anything appears, which total duration hides entirely.
    let mut first_token_at: Option<std::time::Duration> = None;
    let answer = {
        let sid = session_id.clone();
        let mut on_event = |e: TurnEvent<'_>| match e {
            TurnEvent::Delta { text, channel } => {
                if first_token_at.is_none() {
                    first_token_at = Some(started.elapsed());
                }
                match channel {
                Channel::Reasoning => {
                    reasoning.push_str(text);
                    emit(&json!({ "type": "reasoning_delta", "text": text, "sessionId": sid }));
                }
                Channel::Content => {
                    emit(&json!({ "type": "assistant_delta", "text": text, "sessionId": sid }));
                }
                }
            }
            // A tool call is durable and on screen the moment it is made,
            // not after the turn settles — a long shell command would
            // otherwise look like a hang.
            TurnEvent::ToolCall { id, name, args } => {
                let w = store.append_event(
                    &sid,
                    NewEvent::new(
                        "tool_call",
                        json!({ "id": id, "name": name, "args": args }),
                    ),
                );
                let event_id = w.map(|x| x.event_id).unwrap_or_default();
                emit(&json!({
                    "type": "card",
                    "card": { "kind": "tool_call", "name": name, "args": args,
                              "eventId": event_id, "visible": true },
                }));
            }
            TurnEvent::ToolResult { id, name, text } => {
                // A denial is a failure from the UI's point of view, even
                // though it is a well-formed result the model can read.
                let ok = !text.starts_with("error:") && !text.starts_with("denied:");
                let w = store.append_event(
                    &sid,
                    NewEvent::new(
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
                              "eventId": event_id, "visible": true },
                }));
            }
        };
        rt.block_on(agent.run_events(&content, true, &mut on_event))
    };

    let answer = match answer {
        Ok(a) => a,
        Err(e) => {
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
    if !reasoning.trim().is_empty() {
        let w = store.append_event(
            &session_id,
            NewEvent::new("reasoning", json!({ "text": reasoning })),
        );
        if let Ok(w) = w {
            emit(&json!({
                "type": "card",
                "card": { "kind": "reasoning", "text": reasoning, "eventId": w.event_id, "visible": true },
            }));
        }
    }

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
    let _ = store.append_event(
        &session_id,
        NewEvent {
            event_type: "gen_info".into(),
            payload: telemetry.clone(),
            duration_ms: Some(elapsed_ms),
            meta: Some({
                let mut m = serde_json::Map::new();
                m.insert("telemetry".into(), telemetry.clone());
                m
            }),
            ..Default::default()
        },
    );

    let text = answer.trim().to_string();
    if !text.is_empty() {
        let _ = store.append_event(
            &session_id,
            NewEvent::new(
                "context_window",
                json!({ "role": "assistant", "content": text }),
            ),
        );
    }
    // Always finalize the bubble, even when empty, so a client streaming into
    // a live bubble knows the turn is sealed.
    emit(&json!({
        "type": "card",
        "card": { "kind": "assistant", "text": text, "telemetry": telemetry, "streaming": false },
    }));

    store.record_turn_complete(&session_id, Some(elapsed_ms));
    emit(&json!({ "type": "done", "ok": true, "sessionId": session_id }));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    store: &ConvStore,
    tabs: &LiveTabs,
    rt: &tokio::runtime::Runtime,
    project_root: &str,
    pool: &McpPool,
    def: &AgentDef,
    req: TurnRequest,
    emit: &mut dyn FnMut(&Value) -> bool,
) -> Result<(), TurnError> {
    let content = req.content.trim().to_string();
    if content.is_empty() {
        return Err(TurnError::EmptyContent);
    }
    let conv = resolve_session(store, tabs, &req)?;
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

    // The user's message is durable before any model runs.
    let user_node = store
        .append_event(
            &conv,
            NewEvent::new("context_window", json!({ "role": "user", "content": content })),
        )
        .map_err(|e| TurnError::Store(e.to_string()))?;

    let roster = store.participants(&conv);
    let header = store.load(&conv).unwrap_or(Value::Null);
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

    let root = std::path::Path::new(project_root);
    let (servers, mcp_errors) = pool.ensure(&def.mcp, root);
    for e in &mcp_errors {
        let _ = store.append_event(&conv, NewEvent::new("error", json!({ "message": e })));
        emit(&json!({ "type": "card", "card": { "kind": "error", "text": e } }));
    }

    // --- round 0 responders -------------------------------------------------
    let mut responders: Vec<i64> = if routing == "lead" {
        lead_assignment(store, rt, &conv, &roster, &content, emit)
    } else {
        crate::group::responders(&roster, &content, 0, &[], &[])
    };
    let mut round_index: i64 = 0;
    let mut published: usize = 0;

    // The loop yields *why* it stopped, so there is no window in which
    // "settled" holds a value nobody chose.
    let settled = loop {
        if let Some(s) = crate::group::check_bounds(round_index, published, responders.len()) {
            break s;
        }
        let mut round_texts: Vec<(i64, String)> = Vec::new();
        let mut replied: Vec<i64> = Vec::new();

        for pid in responders.clone() {
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

            let model = me
                .model
                .clone()
                .filter(|m| !m.trim().is_empty())
                .or_else(|| header.get("model").and_then(|v| v.as_str()).map(String::from))
                .filter(|m| !m.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_MODEL.into());
            let reply = match run_participant(
                store, rt, &servers, def, &conv, &me, &task, &model, emit,
            ) {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    let _ = store.append_event(
                        &conv,
                        NewEvent {
                            event_type: "error".into(),
                            payload: json!({ "message": msg, "handle": me.handle }),
                            speaker: Some(me.id),
                            ..Default::default()
                        },
                    );
                    emit(&json!({ "type": "card", "card": { "kind": "error", "text": msg } }));
                    continue;
                }
            };

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
                    NewEvent {
                        event_type: "context_window".into(),
                        payload: json!({ "role": "assistant", "content": text }),
                        speaker: Some(me.id),
                        ..Default::default()
                    },
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
                    "participantId": me.id, "handle": me.handle,
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
    emit(&json!({
        "type": "done",
        "ok": true,
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
    store: &ConvStore,
    rt: &tokio::runtime::Runtime,
    servers: &[Arc<McpServer>],
    def: &AgentDef,
    conv: &str,
    me: &crate::conv_store::Participant,
    task: &crate::group::Task,
    model: &str,
    emit: &mut dyn FnMut(&Value) -> bool,
) -> Result<String, TurnError> {
    let system = if task.system_prompt.trim().is_empty() {
        // Fall back to the agent definition, with the participant's identity
        // appended: an agent that does not know its own handle cannot be
        // told apart from its peers in the flattened room.
        let base = def.system.clone();
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
        system_prompt: Some(system),
        // The history is pushed explicitly below, and must not also
        // accumulate across participants.
        retain_history: Some(true),
        parallel_tools: Some(true),
        ..Default::default()
    })
    .map_err(|e| TurnError::Agent(e.to_string()))?;
    for (prompt, reply) in &task.history {
        agent.push_context(Message::user(prompt.clone()));
        agent.push_context(Message::assistant(reply.clone()));
    }

    let session_state = store.load(conv);
    let overridden = session_state
        .as_ref()
        .and_then(|s| s.get("allowlist"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let rules = overridden.unwrap_or_else(|| def.allowlist.clone());
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
    );

    let mut reasoning = String::new();
    let handle = me.handle.clone();
    let pid = me.id;
    let answer = {
        let mut on_event = |e: TurnEvent<'_>| match e {
            TurnEvent::Delta { text, channel } => match channel {
                Channel::Reasoning => {
                    reasoning.push_str(text);
                    emit(&json!({ "type": "reasoning_delta", "text": text,
                                  "sessionId": conv, "handle": handle, "participantId": pid }));
                }
                Channel::Content => {
                    emit(&json!({ "type": "assistant_delta", "text": text,
                                  "sessionId": conv, "handle": handle, "participantId": pid }));
                }
            },
            TurnEvent::ToolCall { id, name, args } => {
                let w = store.append_event(
                    conv,
                    NewEvent {
                        event_type: "tool_call".into(),
                        payload: json!({ "id": id, "name": name, "args": args }),
                        speaker: Some(pid),
                        ..Default::default()
                    },
                );
                let event_id = w.map(|x| x.event_id).unwrap_or_default();
                emit(&json!({
                    "type": "card",
                    "card": { "kind": "tool_call", "name": name, "args": args,
                              "eventId": event_id, "participantId": pid,
                              "handle": handle, "visible": true },
                }));
            }
            TurnEvent::ToolResult { id, name, text } => {
                let ok = !text.starts_with("error:") && !text.starts_with("denied:");
                let w = store.append_event(
                    conv,
                    NewEvent {
                        event_type: "tool_result".into(),
                        payload: json!({ "tool_call_id": id, "name": name, "text": text,
                                         "ok": ok, "denied": text.starts_with("denied:") }),
                        speaker: Some(pid),
                        ..Default::default()
                    },
                );
                let event_id = w.map(|x| x.event_id).unwrap_or_default();
                emit(&json!({
                    "type": "card",
                    "card": { "kind": "tool_result", "name": name, "text": text, "ok": ok,
                              "denied": text.starts_with("denied:"), "eventId": event_id,
                              "participantId": pid, "handle": handle, "visible": true },
                }));
            }
        };
        rt.block_on(agent.run_events(&task.prompt, true, &mut on_event))
            .map_err(|e| TurnError::Agent(e.to_string()))?
    };

    if !reasoning.trim().is_empty() {
        let w = store.append_event(
            conv,
            NewEvent {
                event_type: "reasoning".into(),
                payload: json!({ "text": reasoning }),
                speaker: Some(pid),
                ..Default::default()
            },
        );
        if let Ok(w) = w {
            emit(&json!({
                "type": "card",
                "card": { "kind": "reasoning", "text": reasoning, "eventId": w.event_id,
                          "participantId": pid, "handle": me.handle, "visible": true },
            }));
        }
    }
    Ok(answer)
}

/// Lead routing: ask the designated lead who should answer, parse its JSON.
///
/// On an unparseable reply the lead answers alone rather than the message
/// going nowhere — a routing failure must not swallow the user's turn.
fn lead_assignment(
    store: &ConvStore,
    rt: &tokio::runtime::Runtime,
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
    .and_then(|a| rt.block_on(a.run(&prompt)).ok())
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
