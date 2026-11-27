//! `chat` tools — the coordinator's hands on the app itself.
//!
//! The agent-team pattern this serves: one **coordinator** agent that
//! manages the other agents on the human's behalf — listing sessions,
//! reading what happened in them, opening new ones, seating agents, and
//! submitting prompts. Ordinary agent tools cannot reach conversations
//! other than their own; these five can, which is exactly why they are
//! gated like every other tool (the agent's tool allowlist, the approval
//! queue) and why a user must be permitted to grant them before an agent
//! may hold them (the user-level `tools:` allowlist — see PLAN44).
//!
//! Naming follows the runtime convention every other tool obeys —
//! `server__tool`, e.g. `file_io__read_file` — so the allowlist machinery
//! treats them unchanged. The plan's `sessions.list` etc. are these:
//!
//! | plan                | tool fn                   |
//! |---------------------|---------------------------|
//! | `sessions.list`     | `chat__sessions_list`     |
//! | `sessions.read`     | `chat__sessions_read`     |
//! | `sessions.create`   | `chat__sessions_create`   |
//! | `sessions.prompt`   | `chat__sessions_prompt`   |
//! | `participants.list` | `chat__participants_list` |
//!
//! Authority (PLAN44 B5): a coordinator acts under its **owner's** rights —
//! it can see and drive exactly what the user who created it could. The
//! user-ACL layer is not built yet (workstream C); until it is, these tools
//! see what any turn in this host sees. The seams are in place: every
//! access goes through `app.store`, one choke point to filter by owner.
//!
//! `chat__sessions_prompt` is **enqueue**, never run-to-completion: the
//! submitted turn starts on its own thread and waits for a free model slot
//! like any other turn (the c=N turn queue). Blocking here would deadlock a
//! single-slot local model — the calling turn holds the slot its nested
//! turn is queued behind.

use std::sync::Arc;

use agl::agent::Agent;
use serde_json::{json, Value};

use crate::approvals::Approvals;
use crate::cancel::Cancel;
use crate::serve::AppState;

/// The prefix every tool of this family carries. One server name, five
/// functions — the model sees `chat__sessions_list`, the allowlist matches
/// the same string.
const PREFIX: &str = "chat";

/// Cap on one `chat__sessions_read` answer. A transcript can be arbitrarily
/// long and the coordinator's own context is not; a truncated answer that
/// says so is more useful than a request that dies mid-body.
const READ_CAP: usize = 48 * 1024;

fn tool_name(tool: &str) -> String {
    format!("{PREFIX}__{tool}")
}

/// Register one tool with the same runtime gates every MCP tool gets:
/// `enabled` decides whether it exists for this agent at all, and the
/// allowlist (or a human at the approval queue) decides each call.
#[allow(clippy::too_many_arguments)]
fn one(
    agent: &mut Agent,
    tool: &str,
    description: &str,
    // The tool's FULL input schema (the same shape MCP servers hand out);
    // `Agent::tool` wants the properties map and required list separately,
    // so it is split here — one split, every tool. Passing the full schema
    // through turned `type`/`required` into property names and the wire
    // schema into something any validating endpoint rejects.
    schema: Value,
    enabled: Option<&Vec<String>>,
    allow_text: &str,
    enforce: bool,
    session: &str,
    approvals: &Arc<Approvals>,
    stop: &Arc<Cancel>,
    body: Box<dyn Fn(&Value) -> Result<String, String> + Send + Sync>,
) -> bool {
    let gate_name = tool_name(tool);
    if let Some(list) = enabled {
        if !list.iter().any(|e| e == &gate_name) {
            return false;
        }
    }
    let gate_for_body = gate_name.clone();
    let allow_text = allow_text.to_string();
    let session = session.to_string();
    let approvals = Arc::clone(approvals);
    let stop = Arc::clone(stop);
    let (properties, required) = crate::mcp::schema_parts(&schema);
    agent
        .tool(&gate_name, description, properties, required, move |args: &Value| {
            let gate_name = &gate_for_body;
            if stop.stopped() {
                return Ok("stopped".into());
            }
            if enforce {
                let cmd = crate::allowlist::command_line(&gate_name, args);
                if !crate::allowlist::is_allowed(&allow_text, &gate_name, &cmd)
                    && !approvals.pinned(&session, &gate_name, &cmd)
                {
                    let decision =
                        approvals.ask_cancellable(&session, &gate_name, &cmd, args, &stop);
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
            if stop.stopped() {
                return Ok("stopped".into());
            }
            body(args)
        })
        .is_ok()
}

/// Register the five chat-control tools. Returns how many landed.
pub fn register(
    agent: &mut Agent,
    app: &Arc<AppState>,
    enabled: Option<&Vec<String>>,
    allow_text: &str,
    policy_mode: &str,
    session: &str,
    approvals: Arc<Approvals>,
    stop: Arc<Cancel>,
) -> usize {
    let enforce = !policy_mode.eq_ignore_ascii_case("auto");
    let mut n = 0;

    // ---- sessions.list ---------------------------------------------------
    let a = Arc::clone(app);
    if one(
        agent,
        "sessions_list",
        "List the conversations in this workspace, with their metadata: \
         id, title, agent, model, status, turn count, context window cap, \
         created/modified times, and how long ago they were modified. \
         Optional `query` filters by title or id.",
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "optional substring to filter titles/ids by" }
            }
        }),
        enabled,
        allow_text,
        enforce,
        session,
        &approvals,
        &stop,
        Box::new(move |args: &Value| {
            let query = args.get("query").and_then(Value::as_str).unwrap_or("").trim().to_lowercase();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64;
            let mut rows: Vec<Value> = Vec::new();
            for s in a.store.list_meta() {
                let id = s.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                let title = s
                    .get("title")
                    .filter(|v| crate::session_store::js_truthy(v))
                    .cloned()
                    .unwrap_or_else(|| json!(id.clone()));
                if !query.is_empty() {
                    let hay = format!("{id} {}", title.as_str().unwrap_or_default()).to_lowercase();
                    if !hay.contains(&query) {
                        continue;
                    }
                }
                let created = s.get("createdAt").and_then(Value::as_i64).unwrap_or(0);
                let updated = s.get("updatedAt").and_then(Value::as_i64).unwrap_or(0);
                rows.push(json!({
                    "id": id,
                    "title": title,
                    "agent": s.get("agent").cloned().unwrap_or(Value::Null),
                    "model": s.get("model").cloned().unwrap_or(Value::Null),
                    "status": s.get("status").cloned().unwrap_or(Value::Null),
                    "turns": s.get("turns").cloned().unwrap_or(json!(0)),
                    "contextWindow": s.get("contextWindow").cloned().unwrap_or(Value::Null),
                    "createdAt": created,
                    "modifiedAt": updated,
                    "elapsedMs": now.saturating_sub(created),
                    "modifiedMsAgo": now.saturating_sub(updated),
                }));
            }
            Ok(json!({ "sessions": rows }).to_string())
        }),
    ) {
        n += 1;
    }

    // ---- sessions.read ---------------------------------------------------
    let a = Arc::clone(app);
    if one(
        agent,
        "sessions_read",
        "Read one conversation's transcript. `mode`: \"summary\" (only the \
         user prompts and assistant replies) or \"detail\" (everything, \
         including reasoning and tool calls). Long transcripts are truncated \
         with a notice.",
        json!({
            "type": "object",
            "properties": {
                "sessionId": { "type": "string", "description": "which conversation to read" },
                "mode": { "type": "string", "enum": ["summary", "detail"], "description": "summary (default) or detail" }
            },
            "required": ["sessionId"]
        }),
        enabled,
        allow_text,
        enforce,
        session,
        &approvals,
        &stop,
        Box::new(move |args: &Value| {
            let id = args.get("sessionId").and_then(Value::as_str).unwrap_or("").trim();
            if id.is_empty() {
                return Err("sessionId is required".into());
            }
            let detail = args.get("mode").and_then(Value::as_str) == Some("detail");
            let rows: Vec<Value> = crate::serve::reconstruct_messages(&a.store, id)
                .into_iter()
                .filter(|r| {
                    detail
                        || matches!(
                            r.get("kind").and_then(Value::as_str),
                            Some("user") | Some("assistant")
                        )
                })
                .collect();
            let mut text = json!({ "sessionId": id, "mode": if detail { "detail" } else { "summary" }, "messages": rows }).to_string();
            if text.len() > READ_CAP {
                text.truncate(READ_CAP);
                text.push_str(&format!(
                    "\n… truncated at {READ_CAP} bytes — ask again with a narrower question."
                ));
            }
            Ok(text)
        }),
    ) {
        n += 1;
    }

    // ---- sessions.create -------------------------------------------------
    let a = Arc::clone(app);
    if one(
        agent,
        "sessions_create",
        "Create a new conversation and return its id. When `agent` names a \
         defined agent it is seated in the room (an empty room answers no \
         prompts), and `model` picks the provider:model it starts with.",
        json!({
            "type": "object",
            "properties": {
                "title": { "type": "string" },
                "agent": { "type": "string", "description": "agent definition name" },
                "model": { "type": "string", "description": "provider:model, blank for the agent default" }
            }
        }),
        enabled,
        allow_text,
        enforce,
        session,
        &approvals,
        &stop,
        Box::new(move |args: &Value| {
            let s = |k: &str| args.get(k).and_then(Value::as_str).map(|v| v.trim().to_string());
            let created = a
                .store
                .create(crate::session_store::NewSession {
                    agent: s("agent"),
                    model: s("model"),
                    title: s("title"),
                    ..Default::default()
                })
                .map_err(|e| e.to_string())?;
            let id = created
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            // Seat the agent, or the room is a room nobody answers in —
            // `run_round`'s no-agents path says so in as many words. A name
            // with no definition is reported, not silently dropped.
            let seated = s("agent")
                .filter(|name| !name.is_empty())
                .and_then(|name| a.store.agents().get(&name))
                .and_then(|def| a.store.invite_mentioned_agent(&id, &def))
                .is_some();
            Ok(json!({
                "id": id,
                "seated": seated,
                "created": created,
            })
            .to_string())
        }),
    ) {
        n += 1;
    }

    // ---- sessions.prompt -------------------------------------------------
    let a = Arc::clone(app);
    if one(
        agent,
        "sessions_prompt",
        "Enqueue a user prompt for a conversation. The turn runs on its own \
         thread as soon as a model slot is free (it does not run inside \
         this call). Returns immediately with the session id.",
        json!({
            "type": "object",
            "properties": {
                "sessionId": { "type": "string", "description": "which conversation to prompt" },
                "content": { "type": "string", "description": "the user message to submit" },
                "agent": { "type": "string", "description": "agent to address (optional)" },
                "model": { "type": "string", "description": "provider:model (optional)" }
            },
            "required": ["sessionId", "content"]
        }),
        enabled,
        allow_text,
        enforce,
        session,
        &approvals,
        &stop,
        Box::new(move |args: &Value| {
            let s = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or("").trim().to_string();
            let sid = s("sessionId");
            let content = s("content");
            if sid.is_empty() || content.is_empty() {
                return Err("sessionId and content are required".into());
            }
            let Some(sender) = a.prompt_sender.clone() else {
                return Ok(
                    "error: this host does not support submitting prompts to sessions.".into(),
                );
            };
            sender(&sid, content, s("agent"), s("model"));
            Ok(format!("enqueued: the turn for {sid} runs when a model slot is free."))
        }),
    ) {
        n += 1;
    }

    // ---- participants.list ----------------------------------------------
    let a = Arc::clone(app);
    if one(
        agent,
        "participants_list",
        "List a conversation's participants — who is in the room. Optional \
         `kind` narrows to \"agent\" or \"human\".",
        json!({
            "type": "object",
            "properties": {
                "sessionId": { "type": "string" },
                "kind": { "type": "string", "enum": ["agent", "human"], "description": "optional filter (human/user are the same seat kind)" }
            },
            "required": ["sessionId"]
        }),
        enabled,
        allow_text,
        enforce,
        session,
        &approvals,
        &stop,
        Box::new(move |args: &Value| {
            let id = args.get("sessionId").and_then(Value::as_str).unwrap_or("").trim();
            if id.is_empty() {
                return Err("sessionId is required".into());
            }
            let kind = args.get("kind").and_then(Value::as_str).unwrap_or("");
            // The store names the human seat "user"; accept "human" as the
            // filter's spelling of the same thing.
            let want = match kind {
                "human" => "user",
                other => other,
            };
            let rows: Vec<Value> = crate::serve::roster_json(&a.store, id)
                .into_iter()
                .filter(|p| want.is_empty() || p.get("kind").and_then(Value::as_str) == Some(want))
                .collect();
            Ok(json!({ "participants": rows }).to_string())
        }),
    ) {
        n += 1;
    }

    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_loader;
    use crate::session_store::NewEvent;

    /// An app in a temp dir with a capturing prompt sender.
    fn app_with_capture() -> (Arc<AppState>, Arc<std::sync::Mutex<Vec<(String, String, String, String)>>>) {
        let dir = std::env::temp_dir().join(format!("chat-tools-test-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let mut state = AppState::new(&dir, 0).unwrap();
        let seen: Arc<std::sync::Mutex<Vec<(String, String, String, String)>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let cap = Arc::clone(&seen);
        state.prompt_sender = Some(Arc::new(move |s: &str, c: String, a: String, m: String| {
            cap.lock().unwrap().push((s.to_string(), c, a, m));
        }));
        (Arc::new(state), seen)
    }

    fn agent() -> Agent {
        Agent::factory(agl::agent::Factory {
            model: Some("llama-server:default".into()),
            ..Default::default()
        })
        .unwrap()
    }

    fn call(agent: &Agent, tool: &str, args: Value) -> String {
        (agent.tools[tool].handler)(&args).unwrap()
    }

    #[test]
    fn registration_counts_and_respects_the_tool_list() {
        let (app, _cap) = app_with_capture();
        let approvals = Arc::new(Approvals::new());
        let stop = Cancel::never();
        let mut a = agent();
        assert_eq!(
            register(&mut a, &app, None, "*", "auto", "s1", Arc::clone(&approvals), Arc::clone(&stop)),
            5
        );
        let names: Vec<_> = a.tools.keys().cloned().collect();
        for t in ["chat__sessions_list", "chat__sessions_read", "chat__sessions_create",
                  "chat__sessions_prompt", "chat__participants_list"] {
            assert!(names.contains(&t.to_string()), "{t} missing");
        }
        // Only the allow-listed tool exists when the agent names one.
        let mut b = agent();
        let only = vec!["chat__sessions_list".to_string()];
        assert_eq!(
            register(&mut b, &app, Some(&only), "*", "auto", "s1", approvals, stop),
            1
        );
    }

    #[test]
    fn create_list_read_and_participants_work() {
        let (app, _cap) = app_with_capture();
        let approvals = Arc::new(Approvals::new());
        let stop = Cancel::never();
        let mut a = agent();
        register(&mut a, &app, None, "*", "auto", "s1", approvals, stop);

        let made = call(&a, "chat__sessions_create", json!({ "title": "from the coordinator" }));
        let id = serde_json::from_str::<Value>(&made).unwrap()["id"].as_str().unwrap().to_string();

        // The list sees it, and the query filter finds it.
        let listed = call(&a, "chat__sessions_list", json!({}));
        assert!(listed.contains("from the coordinator"));
        let filtered = call(&a, "chat__sessions_list", json!({ "query": "coordinator" }));
        assert!(filtered.contains(&id));
        let miss = call(&a, "chat__sessions_list", json!({ "query": "no-such-thing" }));
        assert!(!miss.contains(&id));

        // Summary vs detail: a turn's paperwork is detail-only.
        app.store.append_event(&id, NewEvent::new("context_window", json!({ "role": "user", "content": "hi" }))).unwrap();
        app.store.append_event(&id, NewEvent::new("tool_call", json!({ "name": "x" }))).unwrap();
        app.store.append_event(&id, NewEvent::new("context_window", json!({ "role": "assistant", "content": "hello" }))).unwrap();
        let summary = call(&a, "chat__sessions_read", json!({ "sessionId": id }));
        assert!(summary.contains("hello") && summary.contains("hi"));
        assert!(!summary.contains("tool_call"));
        let detail = call(&a, "chat__sessions_read", json!({ "sessionId": id, "mode": "detail" }));
        assert!(detail.contains("tool_call"));

        // Participants: a fresh room already has the human seat — agents
        // are invited, never assumed.
        let parts = call(&a, "chat__participants_list", json!({ "sessionId": id }));
        assert!(parts.contains("\"kind\":\"user\""), "{parts}");
        let humans_only =
            call(&a, "chat__participants_list", json!({ "sessionId": id, "kind": "human" }));
        assert!(humans_only.contains("\"kind\":\"user\""), "{humans_only}");
        let agents_only =
            call(&a, "chat__participants_list", json!({ "sessionId": id, "kind": "agent" }));
        assert!(agents_only.contains("\"participants\":[]"), "{agents_only}");

        // Naming an agent at create time seats it — an empty room answers
        // no prompts.
        app.store
            .agents()
            .save(&agent_loader::AgentDef { name: "tester".into(), ..Default::default() }, None)
            .unwrap();
        let made = call(&a, "chat__sessions_create", json!({ "title": "seated", "agent": "tester" }));
        let made = serde_json::from_str::<Value>(&made).unwrap();
        assert_eq!(made["seated"], json!(true), "{made}");
        let sid2 = made["id"].as_str().unwrap();
        let parts2 = call(&a, "chat__participants_list", json!({ "sessionId": sid2, "kind": "agent" }));
        assert!(parts2.contains("tester"), "{parts2}");
        // An unknown name is reported, not silently dropped.
        let made = call(&a, "chat__sessions_create", json!({ "title": "unseated", "agent": "ghost" }));
        let made = serde_json::from_str::<Value>(&made).unwrap();
        assert_eq!(made["seated"], json!(false), "{made}");
    }

    #[test]
    fn prompt_enqueues_and_never_blocks() {
        let (app, seen) = app_with_capture();
        let approvals = Arc::new(Approvals::new());
        let stop = Cancel::never();
        let mut a = agent();
        register(&mut a, &app, None, "*", "auto", "s1", approvals, stop);

        let made = call(&a, "chat__sessions_create", json!({ "title": "target" }));
        let id = serde_json::from_str::<Value>(&made).unwrap()["id"].as_str().unwrap().to_string();
        let out = call(&a, "chat__sessions_prompt", json!({
            "sessionId": id, "content": "say hi", "agent": "ping", "model": ""
        }));
        assert!(out.starts_with("enqueued"), "{out}");
        let got = seen.lock().unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, id);
        assert_eq!(got[0].1, "say hi");
        assert_eq!(got[0].2, "ping");
    }

    #[test]
    fn a_host_without_a_sender_says_so() {
        let dir = std::env::temp_dir().join(format!("chat-tools-test-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let app = Arc::new(AppState::new(&dir, 0).unwrap());
        let approvals = Arc::new(Approvals::new());
        let stop = Cancel::never();
        let mut a = agent();
        register(&mut a, &app, None, "*", "auto", "s1", approvals, stop);
        let out = call(&a, "chat__sessions_prompt", json!({ "sessionId": "x", "content": "hi" }));
        assert!(out.starts_with("error:"), "{out}");
    }
}
