use std::sync::Arc;

use agl::agent::Agent;
use serde_json::{json, Value};

use crate::approvals::Approvals;
use crate::cancel::Cancel;
use crate::serve::AppState;

const PREFIX: &str = "chat";

pub type ExtraTools = Arc<dyn Fn(&mut Agent, &str) + Send + Sync>;

static EXTRA: std::sync::OnceLock<ExtraTools> = std::sync::OnceLock::new();

pub fn set_extra_tools(f: ExtraTools) {
    let _ = EXTRA.set(f);
}

const READ_CAP: usize = 48 * 1024;

fn tool_name(tool: &str) -> String {
    format!("{PREFIX}__{tool}")
}

#[allow(clippy::too_many_arguments)]
fn one(
    agent: &mut Agent,
    tool: &str,
    description: &str,
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

    if let Some(extra) = EXTRA.get() {
        extra(agent, session);
    }

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
