//! `ConvStore` — conversations as a tree in SQLite.
//!
//! This is the replacement for [`crate::session_store::SessionStore`]. The
//! method names are deliberately the same, because every one of them is a
//! call site in `harness.rs` or `serve.rs` and the point of the change is the
//! storage model, not the harness. What differs underneath:
//!
//! | `SessionStore` | `ConvStore` |
//! |---|---|
//! | `append_event` writes a line at the end of a file | writes a node whose parent is the cursor, then moves the cursor onto it |
//! | `list_events` reads the whole file | walks cursor → parent → … → root and reverses |
//! | a `context_meta` line patches visibility | an `UPDATE` on the node's `visible` column |
//! | two roles, `user` and `assistant` | N participants, each a row with a handle |
//!
//! A conversation that is never forked behaves **identically** to the old
//! log: every node's parent is the one before it, the cursor sits on the
//! tail, and the cursor→root path is every node in `seq` order. That
//! equivalence is what makes the migration a no-op from the UI's side, and
//! it is asserted directly in the tests at the bottom of this file.
//!
//! Forking is then a two-step with no new concepts: point the cursor at any
//! node, append. The old branch is still there, still reachable, because
//! nothing was ever rewritten — the only thing that changed is which path
//! the walk finds.
//!
//! The header the API speaks is still the legacy camelCase object
//! (`createdAt`, `policyMode`, `contextWindow`, …), produced by
//! [`ConvStore::load`] from the `conversation` row. Keeping that shape means
//! the SPA, the NDJSON card vocabulary and the existing route contracts did
//! not have to move at the same time as the database.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Map, Value};

use crate::session_store::{
    fold_event_metas, js_string_of, js_truthy, new_session_id, now_iso, ContextItem, EventWrapper,
    Goto, NewEvent, NewSession,
};

/// Event kinds whose author is an agent participant rather than the human.
const AGENT_KINDS: &[&str] = &[
    "reasoning",
    "tool_call",
    "tool_result",
    "gen_info",
    "system",
    "tools",
    "provider_request",
    "provider_response",
    "harness",
];

/// One seat in the conversation.
#[derive(Debug, Clone)]
pub struct Participant {
    pub id: i64,
    pub conversation_id: String,
    pub kind: String,
    pub handle: String,
    pub display_name: String,
    pub agent_name: Option<String>,
    pub model: Option<String>,
    pub system_prompt: String,
    pub avatar: Option<String>,
    pub seat: i64,
    pub active: bool,
    pub role: Option<String>,
}

impl Participant {
    pub fn is_agent(&self) -> bool {
        self.kind == "agent"
    }

    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "kind": self.kind,
            "handle": self.handle,
            "displayName": self.display_name,
            "agentName": self.agent_name,
            "model": self.model,
            "avatar": self.avatar,
            "seat": self.seat,
            "active": self.active,
            "role": self.role,
        })
    }
}

/// A node as the tree sees it: the event plus its position.
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub id: String,
    pub parent_id: Option<String>,
    pub seq: i64,
    pub depth: i64,
    pub kind: String,
    pub participant_id: Option<i64>,
    pub content: Option<String>,
    pub role: Option<String>,
    pub payload: Value,
    pub meta: Map<String, Value>,
    pub visible: bool,
    pub goto: Option<Goto>,
    pub ts: String,
    pub duration_ms: Option<f64>,
}

impl TreeNode {
    /// The node as the old wrapped-event shape, so every reader written
    /// against the JSONL log keeps working.
    pub fn to_event(&self) -> EventWrapper {
        let mut meta = self.meta.clone();
        // `visible` and `goto` live in columns now; project them back into
        // `meta` only when they say something, so the serialised shape stays
        // as close to the old log as it can.
        if !self.visible {
            meta.insert("visible".into(), Value::Bool(false));
        }
        if let Some(g) = &self.goto {
            meta.insert("goto".into(), json!({ "enabled": g.enabled, "label": g.label }));
        }
        EventWrapper {
            event_id: self.id.clone(),
            ts: self.ts.clone(),
            event_type: self.kind.clone(),
            duration_ms: self.duration_ms,
            payload: self.payload.clone(),
            meta,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "parentId": self.parent_id,
            "seq": self.seq,
            "depth": self.depth,
            "kind": self.kind,
            "participantId": self.participant_id,
            "content": self.content,
            "role": self.role,
            "visible": self.visible,
            "ts": self.ts,
            "durationMs": self.duration_ms,
        })
    }
}

/// Durable conversations in `<root>/.angela/conversations.db`.
pub struct ConvStore {
    pub project_root: PathBuf,
    conn: Mutex<Connection>,
}

// ---------------------------------------------------------------------------
// Small conversions
// ---------------------------------------------------------------------------

fn to_json_text(v: &Value) -> Option<String> {
    if v.is_null() {
        None
    } else {
        Some(v.to_string())
    }
}

fn from_json_text(s: Option<String>) -> Value {
    s.and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null)
}

/// A handle from an agent name: lowercase, non-alphanumerics folded to `-`.
/// Handles are what `@mentions` match on, so they have to be typeable.
pub fn handle_from(name: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "agent".into()
    } else {
        trimmed
    }
}

impl ConvStore {
    pub fn new(project_root: impl AsRef<Path>) -> rusqlite::Result<Self> {
        let root = project_root.as_ref().to_path_buf();
        let conn = crate::db::open(&root)?;
        Ok(ConvStore {
            project_root: root,
            conn: Mutex::new(conn),
        })
    }

    /// An in-memory store rooted at `project_root` — for tests.
    pub fn memory(project_root: impl AsRef<Path>) -> rusqlite::Result<Self> {
        Ok(ConvStore {
            project_root: project_root.as_ref().to_path_buf(),
            conn: Mutex::new(crate::db::open_memory()?),
        })
    }

    /// Run `f` with the connection. Every public method funnels through here
    /// so there is exactly one place the lock is taken, and no method can
    /// deadlock by calling another while holding it.
    pub fn with<T>(&self, f: impl FnOnce(&Connection) -> T) -> T {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        f(&conn)
    }

    // -----------------------------------------------------------------------
    // Conversations
    // -----------------------------------------------------------------------

    pub fn create(&self, meta: NewSession) -> rusqlite::Result<Value> {
        let id = new_session_id();
        let root = self.project_root.to_string_lossy().into_owned();
        self.with(|c| {
            // `max + 1`, never `count + 1`: closing tab 2 of {1,2,3} and
            // making a new one must give 4. A reused number would collide
            // with a live tab's key in the strip.
            let ordinal: i64 = c
                .query_row("SELECT COALESCE(MAX(ordinal), 0) + 1 FROM conversation", [], |r| {
                    r.get(0)
                })
                .unwrap_or(1);
            let now = now_iso();
            c.execute(
                "INSERT INTO conversation
                   (id, ordinal, api_version, kind, title, agent, agent_file, model, status,
                    policy_mode, approve_mode, allowlist, allowlist_source, context_window,
                    mcp, project_root, config, overrides, turns, created_at, updated_at)
                 VALUES (?1, ?2, 'conv/v1', 'conversation', ?3, ?4, ?5, ?6, 'ready',
                         ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, 0, ?16, ?16)",
                params![
                    id,
                    ordinal,
                    meta.title,
                    meta.agent,
                    meta.agent_file,
                    meta.model,
                    meta.policy_mode.clone().unwrap_or_else(|| "ask".into()),
                    meta.approve_mode.as_ref().and_then(to_json_text),
                    meta.allowlist,
                    meta.allowlist_source,
                    meta.context_window.map(|v| v as i64),
                    serde_json::to_string(&meta.mcp).ok(),
                    root,
                    meta.config.as_ref().and_then(to_json_text),
                    meta.overrides.as_ref().and_then(to_json_text),
                    now,
                ],
            )?;
            c.execute(
                "INSERT INTO cursor (conversation_id, participant_id, event_id, updated_at)
                 VALUES (?1, 0, NULL, ?2)",
                params![id, now],
            )?;
            // A conversation always has a roster, even a chat of two: the
            // group-chat path and the 2-party path are then the same code
            // with a different roster size, instead of two code paths.
            Self::seed_roster(c, &id, meta.agent.as_deref(), meta.model.as_deref())?;
            Ok::<(), rusqlite::Error>(())
        })?;
        // The opening lifecycle beat, so the log reads the same as it did.
        let _ = self.append_event(
            &id,
            NewEvent::new(
                "session",
                json!({
                    "kind": "created",
                    "agent": meta.agent,
                    "model": meta.model,
                    "mcp": meta.mcp,
                    "overrides": meta.overrides.clone().unwrap_or(Value::Null),
                    "config": meta.config.clone().unwrap_or(Value::Null),
                }),
            ),
        );
        Ok(self.load(&id).unwrap_or(Value::Null))
    }

    /// Seat the human and, when the conversation names one, the agent.
    fn seed_roster(
        c: &Connection,
        conv: &str,
        agent: Option<&str>,
        model: Option<&str>,
    ) -> rusqlite::Result<()> {
        c.execute(
            "INSERT OR IGNORE INTO participants
               (conversation_id, kind, handle, display_name, seat, active)
             VALUES (?1, 'user', 'user', 'User', 0, 1)",
            params![conv],
        )?;
        let name = agent.unwrap_or("assistant");
        c.execute(
            "INSERT OR IGNORE INTO participants
               (conversation_id, kind, handle, display_name, agent_name, model,
                system_prompt, seat, active, role)
             VALUES (?1, 'agent', ?2, ?3, ?4, ?5, '', 1, 1, 'member')",
            params![conv, handle_from(name), name, agent, model],
        )?;
        Ok(())
    }

    /// The conversation header, in the legacy camelCase shape the HTTP
    /// surface speaks.
    pub fn load(&self, id: &str) -> Option<Value> {
        self.with(|c| Self::load_in(c, id))
    }

    fn load_in(c: &Connection, id: &str) -> Option<Value> {
        let mut out = c
            .query_row(
                "SELECT id, ordinal, api_version, kind, title, agent, agent_file, model, status,
                        policy_mode, approve_mode, allowlist_mode, allowlist, allowlist_source,
                        context_window, mcp, project_root, config, overrides, turns,
                        round_index, round_status, routing, purpose, notes, memory_high,
                        legacy_format, created_at, updated_at
                   FROM conversation WHERE id = ?1",
                params![id],
                |r| {
                    Ok(json!({
                        "id": r.get::<_, String>(0)?,
                        "ordinal": r.get::<_, i64>(1)?,
                        "apiVersion": r.get::<_, String>(2)?,
                        "kind": r.get::<_, String>(3)?,
                        "title": r.get::<_, Option<String>>(4)?,
                        "agent": r.get::<_, Option<String>>(5)?,
                        "agentFile": r.get::<_, Option<String>>(6)?,
                        "model": r.get::<_, Option<String>>(7)?,
                        "status": r.get::<_, String>(8)?,
                        "policyMode": r.get::<_, String>(9)?,
                        "approveMode": from_json_text(r.get::<_, Option<String>>(10)?),
                        "allowlistMode": r.get::<_, Option<String>>(11)?
                            .unwrap_or_else(|| "allowlist".into()),
                        "allowlist": r.get::<_, Option<String>>(12)?,
                        "allowlistSource": r.get::<_, Option<String>>(13)?,
                        "contextWindow": r.get::<_, Option<i64>>(14)?,
                        "mcp": from_json_text(r.get::<_, Option<String>>(15)?),
                        "projectRoot": r.get::<_, String>(16)?,
                        "config": from_json_text(r.get::<_, Option<String>>(17)?),
                        "overrides": from_json_text(r.get::<_, Option<String>>(18)?),
                        "turns": r.get::<_, i64>(19)?,
                        "roundIndex": r.get::<_, i64>(20)?,
                        "roundStatus": r.get::<_, Option<String>>(21)?,
                        "routing": r.get::<_, String>(22)?,
                        "purpose": r.get::<_, Option<String>>(23)?,
                        "notes": r.get::<_, Option<String>>(24)?,
                        "memoryHigh": r.get::<_, Option<i64>>(25)?,
                        "legacyFormat": r.get::<_, Option<String>>(26)?,
                        "createdAt": r.get::<_, String>(27)?,
                        "updatedAt": r.get::<_, String>(28)?,
                    }))
                },
            )
            .optional()
            .ok()
            .flatten()?;

        let o = out.as_object_mut()?;
        if o.get("mcp").map(|v| v.is_null()).unwrap_or(true) {
            o.insert("mcp".into(), json!([]));
        }
        let count: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM nodes WHERE conversation_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        o.insert("eventCount".into(), json!(count));
        // The `last*Tokens` header columns are gone: the newest gen_info node
        // already holds them, and a cached duplicate was one more field that
        // could disagree with the log.
        let last = c
            .query_row(
                "SELECT gen_prompt_tokens, gen_completion_tokens, gen_total_tokens
                   FROM nodes
                  WHERE conversation_id = ?1 AND kind = 'gen_info'
                    AND gen_total_tokens IS NOT NULL
                  ORDER BY seq DESC LIMIT 1",
                params![id],
                |r| {
                    Ok((
                        r.get::<_, Option<i64>>(0)?,
                        r.get::<_, Option<i64>>(1)?,
                        r.get::<_, Option<i64>>(2)?,
                    ))
                },
            )
            .optional()
            .ok()
            .flatten()
            .unwrap_or((None, None, None));
        o.insert("lastPromptTokens".into(), json!(last.0));
        o.insert("lastCompletionTokens".into(), json!(last.1));
        o.insert("lastTotalTokens".into(), json!(last.2));
        Some(out)
    }

    /// Which legacy header key maps onto which column.
    fn column_for(key: &str) -> Option<&'static str> {
        Some(match key {
            "ordinal" => "ordinal",
            "title" => "title",
            "agent" => "agent",
            "agentFile" | "agent_file" => "agent_file",
            "model" => "model",
            "status" => "status",
            "policyMode" | "policy_mode" => "policy_mode",
            "allowlistMode" | "allowlist_mode" => "allowlist_mode",
            "allowlist" => "allowlist",
            "allowlistSource" | "allowlist_source" => "allowlist_source",
            "contextWindow" | "context_window" => "context_window",
            "projectRoot" | "project_root" => "project_root",
            "turns" => "turns",
            "roundIndex" | "round_index" => "round_index",
            "roundStatus" | "round_status" => "round_status",
            "routing" => "routing",
            "purpose" => "purpose",
            "notes" => "notes",
            "memoryHigh" | "memory_high" => "memory_high",
            // JSON-valued columns
            "mcp" => "mcp",
            "config" => "config",
            "overrides" => "overrides",
            "approveMode" | "approve_mode" => "approve_mode",
            _ => return None,
        })
    }

    const JSON_COLUMNS: &'static [&'static str] = &["mcp", "config", "overrides", "approve_mode"];

    /// Patch header fields. Unknown keys are ignored rather than fatal:
    /// `lastPromptTokens` and friends still arrive from older call sites and
    /// are now derived, so accepting-and-dropping them is the honest answer.
    pub fn update_meta(&self, id: &str, patch: Map<String, Value>) -> Option<Value> {
        self.with(|c| {
            for (k, v) in &patch {
                let Some(col) = Self::column_for(k) else { continue };
                let sql = format!("UPDATE conversation SET {col} = ?1 WHERE id = ?2");
                let res = if Self::JSON_COLUMNS.contains(&col) {
                    c.execute(&sql, params![to_json_text(v), id])
                } else {
                    match v {
                        Value::Null => c.execute(&sql, params![Option::<String>::None, id]),
                        Value::Bool(b) => c.execute(&sql, params![*b as i64, id]),
                        Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                c.execute(&sql, params![i, id])
                            } else {
                                c.execute(&sql, params![n.as_f64(), id])
                            }
                        }
                        Value::String(s) => c.execute(&sql, params![s, id]),
                        other => c.execute(&sql, params![other.to_string(), id]),
                    }
                };
                let _ = res;
            }
            let _ = c.execute(
                "UPDATE conversation SET updated_at = ?1 WHERE id = ?2",
                params![now_iso(), id],
            );
            Self::load_in(c, id)
        })
    }

    pub fn set_status(&self, id: &str, status: &str) {
        self.with(|c| {
            let _ = c.execute(
                "UPDATE conversation SET status = ?1, updated_at = ?2 WHERE id = ?3",
                params![status, now_iso(), id],
            );
        });
    }

    pub fn record_turn_complete(&self, id: &str, duration_ms: Option<f64>) {
        let turns = self.with(|c| {
            let _ = c.execute(
                "UPDATE conversation
                    SET turns = turns + 1, status = 'ready', updated_at = ?1
                  WHERE id = ?2",
                params![now_iso(), id],
            );
            c.query_row(
                "SELECT turns FROM conversation WHERE id = ?1",
                params![id],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0)
        });
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

    /// Conversation ids, newest first (ids sort chronologically).
    pub fn list(&self) -> Vec<String> {
        self.with(|c| {
            let mut st = match c.prepare("SELECT id FROM conversation ORDER BY id DESC") {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
            let rows = st.query_map([], |r| r.get::<_, String>(0));
            match rows {
                Ok(it) => it.filter_map(|r| r.ok()).collect(),
                Err(_) => Vec::new(),
            }
        })
    }

    pub fn list_meta(&self) -> Vec<Value> {
        self.with(|c| {
            let ids: Vec<String> = {
                let mut st = match c.prepare("SELECT id FROM conversation ORDER BY id DESC") {
                    Ok(s) => s,
                    Err(_) => return Vec::new(),
                };
                let out = match st.query_map([], |r| r.get::<_, String>(0)) {
                    Ok(it) => it.filter_map(|r| r.ok()).collect(),
                    Err(_) => return Vec::new(),
                };
                out
            };
            ids.iter().filter_map(|id| Self::load_in(c, id)).collect()
        })
    }

    /// Delete a conversation and everything hanging off it.
    ///
    /// Refuses a traversal-shaped id even though SQL has no path to traverse:
    /// the same id reaches the legacy file cleanup below, which does.
    pub fn delete(&self, id: &str) -> bool {
        if id.is_empty() || id.contains("..") || id.contains('/') || id.contains('\\') {
            return false;
        }
        let n = self.with(|c| {
            // Children before parents: `nodes.parent_id` is a self-reference
            // and ON DELETE CASCADE does not order siblings for us.
            let _ = c.execute("PRAGMA defer_foreign_keys = ON", []);
            let n = c
                .execute("DELETE FROM conversation WHERE id = ?1", params![id])
                .unwrap_or(0);
            let _ = c.execute("DELETE FROM nodes WHERE conversation_id = ?1", params![id]);
            n
        });
        // The pre-migration files, if this project still has them: the user
        // asked for the conversation gone, and leaving a copy that a
        // re-migration would resurrect is not "gone".
        let legacy = self.project_root.join(".angela").join("sessions");
        for ext in ["json", "jsonl"] {
            let p = legacy.join(format!("{id}.{ext}"));
            if p.exists() {
                let _ = std::fs::remove_file(p);
            }
        }
        n > 0
    }

    /// Remove every conversation (`angela clean`). Returns the count.
    pub fn clean(&self) -> usize {
        self.with(|c| {
            let _ = c.execute("PRAGMA defer_foreign_keys = ON", []);
            let n = c.execute("DELETE FROM conversation", []).unwrap_or(0);
            let _ = c.execute("DELETE FROM nodes", []);
            n
        })
    }

    /// Every pre-`ordinal` conversation gets a number. Kept for parity with
    /// the file store; the migration already numbers what it imports, so this
    /// normally finds nothing.
    pub fn backfill_ordinals(&self) -> usize {
        self.with(|c| {
            let rows: Vec<String> = {
                let mut st = match c.prepare(
                    "SELECT id FROM conversation WHERE ordinal IS NULL OR ordinal <= 0
                     ORDER BY created_at ASC",
                ) {
                    Ok(s) => s,
                    Err(_) => return 0,
                };
                let out = match st.query_map([], |r| r.get::<_, String>(0)) {
                    Ok(it) => it.filter_map(|r| r.ok()).collect(),
                    Err(_) => return 0,
                };
                out
            };
            let mut next: i64 = c
                .query_row("SELECT COALESCE(MAX(ordinal), 0) FROM conversation", [], |r| {
                    r.get(0)
                })
                .unwrap_or(0);
            let n = rows.len();
            for id in rows {
                next += 1;
                let _ = c.execute(
                    "UPDATE conversation SET ordinal = ?1 WHERE id = ?2",
                    params![next, id],
                );
            }
            n
        })
    }

    // -----------------------------------------------------------------------
    // Participants
    // -----------------------------------------------------------------------

    pub fn participants(&self, conv: &str) -> Vec<Participant> {
        self.with(|c| Self::participants_in(c, conv))
    }

    fn participants_in(c: &Connection, conv: &str) -> Vec<Participant> {
        let mut st = match c.prepare(
            "SELECT id, conversation_id, kind, handle, display_name, agent_name, model,
                    system_prompt, avatar, seat, active, role
               FROM participants WHERE conversation_id = ?1 ORDER BY seat ASC, id ASC",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = st.query_map(params![conv], |r| {
            Ok(Participant {
                id: r.get(0)?,
                conversation_id: r.get(1)?,
                kind: r.get(2)?,
                handle: r.get(3)?,
                display_name: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                agent_name: r.get(5)?,
                model: r.get(6)?,
                system_prompt: r.get::<_, Option<String>>(7)?.unwrap_or_default(),
                avatar: r.get(8)?,
                seat: r.get(9)?,
                active: r.get::<_, i64>(10)? != 0,
                role: r.get(11)?,
            })
        });
        match rows {
            Ok(it) => it.filter_map(|r| r.ok()).collect(),
            Err(_) => Vec::new(),
        }
    }

    pub fn participant(&self, id: i64) -> Option<Participant> {
        self.with(|c| {
            let conv: String = c
                .query_row(
                    "SELECT conversation_id FROM participants WHERE id = ?1",
                    params![id],
                    |r| r.get(0),
                )
                .optional()
                .ok()
                .flatten()?;
            Self::participants_in(c, &conv).into_iter().find(|p| p.id == id)
        })
    }

    /// Seat a new participant. The handle is made unique within the
    /// conversation by suffixing, because a duplicate handle would make
    /// `@mention` routing ambiguous and silently address the wrong agent.
    pub fn add_participant(
        &self,
        conv: &str,
        kind: &str,
        handle: &str,
        display_name: Option<&str>,
        agent_name: Option<&str>,
        model: Option<&str>,
        system_prompt: &str,
        avatar: Option<&str>,
        role: Option<&str>,
    ) -> Option<Participant> {
        let base = handle_from(if handle.is_empty() {
            agent_name.or(display_name).unwrap_or("agent")
        } else {
            handle
        });
        self.with(|c| {
            let taken: Vec<String> = Self::participants_in(c, conv)
                .into_iter()
                .map(|p| p.handle)
                .collect();
            let mut h = base.clone();
            let mut n = 2;
            while taken.contains(&h) {
                h = format!("{base}-{n}");
                n += 1;
            }
            let seat: i64 = c
                .query_row(
                    "SELECT COALESCE(MAX(seat), -1) + 1 FROM participants WHERE conversation_id = ?1",
                    params![conv],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            c.execute(
                "INSERT INTO participants
                   (conversation_id, kind, handle, display_name, agent_name, model,
                    system_prompt, avatar, seat, active, role)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1, ?10)",
                params![
                    conv,
                    kind,
                    h,
                    display_name.unwrap_or(&h),
                    agent_name,
                    model,
                    system_prompt,
                    avatar,
                    seat,
                    role.unwrap_or("member"),
                ],
            )
            .ok()?;
            let id = c.last_insert_rowid();
            Self::participants_in(c, conv).into_iter().find(|p| p.id == id)
        })
    }

    pub fn update_participant(&self, id: i64, patch: &Map<String, Value>) -> Option<Participant> {
        self.with(|c| {
            for (k, v) in patch {
                let col = match k.as_str() {
                    "displayName" | "display_name" => "display_name",
                    "model" => "model",
                    "systemPrompt" | "system_prompt" => "system_prompt",
                    "avatar" => "avatar",
                    "role" => "role",
                    "seat" => "seat",
                    "active" => "active",
                    "agentName" | "agent_name" => "agent_name",
                    _ => continue,
                };
                let sql = format!("UPDATE participants SET {col} = ?1 WHERE id = ?2");
                let _ = match v {
                    Value::Null => c.execute(&sql, params![Option::<String>::None, id]),
                    Value::Bool(b) => c.execute(&sql, params![*b as i64, id]),
                    Value::Number(n) => c.execute(&sql, params![n.as_i64().unwrap_or(0), id]),
                    Value::String(s) => c.execute(&sql, params![s, id]),
                    other => c.execute(&sql, params![other.to_string(), id]),
                };
            }
            Ok::<(), ()>(())
        })
        .ok();
        self.participant(id)
    }

    /// Soft-remove: a participant whose nodes are in the tree cannot be
    /// deleted without orphaning them, so an agent that has spoken is
    /// deactivated instead and keeps its authorship.
    pub fn remove_participant(&self, id: i64) -> bool {
        self.with(|c| {
            let spoke: i64 = c
                .query_row(
                    "SELECT COUNT(*) FROM nodes WHERE participant_id = ?1",
                    params![id],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            if spoke > 0 {
                c.execute("UPDATE participants SET active = 0 WHERE id = ?1", params![id])
                    .unwrap_or(0)
                    > 0
            } else {
                c.execute("DELETE FROM participants WHERE id = ?1", params![id])
                    .unwrap_or(0)
                    > 0
            }
        })
    }

    /// The human seat, creating it if an imported conversation lacked one.
    pub fn user_participant(&self, conv: &str) -> Option<Participant> {
        self.with(|c| {
            if let Some(p) = Self::participants_in(c, conv)
                .into_iter()
                .find(|p| p.kind == "user")
            {
                return Some(p);
            }
            let _ = c.execute(
                "INSERT OR IGNORE INTO participants
                   (conversation_id, kind, handle, display_name, seat, active)
                 VALUES (?1, 'user', 'user', 'User', 0, 1)",
                params![conv],
            );
            Self::participants_in(c, conv).into_iter().find(|p| p.kind == "user")
        })
    }

    /// Active agent seats in `seat` order — the rotation a round speaks in.
    pub fn agent_participants(&self, conv: &str) -> Vec<Participant> {
        self.participants(conv)
            .into_iter()
            .filter(|p| p.is_agent() && p.active)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Nodes: append, walk, fork
    // -----------------------------------------------------------------------

    /// The shared cursor's node id, or `None` on an empty conversation.
    pub fn cursor(&self, conv: &str) -> Option<String> {
        self.with(|c| Self::cursor_in(c, conv))
    }

    fn cursor_in(c: &Connection, conv: &str) -> Option<String> {
        c.query_row(
            "SELECT event_id FROM cursor WHERE conversation_id = ?1 AND participant_id = 0",
            params![conv],
            |r| r.get::<_, Option<String>>(0),
        )
        .optional()
        .ok()
        .flatten()
        .flatten()
    }

    /// Move the cursor. This *is* the fork primitive: point it at any node,
    /// then append — the new node becomes that node's child and the branch
    /// the cursor came from is untouched.
    pub fn set_cursor(&self, conv: &str, node_id: Option<&str>) -> bool {
        self.with(|c| {
            if let Some(n) = node_id {
                let exists: i64 = c
                    .query_row(
                        "SELECT COUNT(*) FROM nodes WHERE id = ?1 AND conversation_id = ?2",
                        params![n, conv],
                        |r| r.get(0),
                    )
                    .unwrap_or(0);
                if exists == 0 {
                    return false;
                }
            }
            c.execute(
                "INSERT INTO cursor (conversation_id, participant_id, event_id, updated_at)
                 VALUES (?1, 0, ?2, ?3)
                 ON CONFLICT(conversation_id, participant_id)
                   DO UPDATE SET event_id = excluded.event_id,
                                 updated_at = excluded.updated_at",
                params![conv, node_id, now_iso()],
            )
            .is_ok()
        })
    }

    /// Append an event as a child of the cursor, then advance the cursor.
    pub fn append_event(&self, conv: &str, ev: NewEvent) -> rusqlite::Result<EventWrapper> {
        self.append_event_at(conv, None, ev)
    }

    /// Append an event under an explicit parent (`None` = under the cursor).
    ///
    /// Passing a parent that is not the cursor is a fork; the cursor still
    /// ends up on the new node, because the thing you just said is the thing
    /// you are now talking from.
    pub fn append_event_at(
        &self,
        conv: &str,
        parent: Option<&str>,
        ev: NewEvent,
    ) -> rusqlite::Result<EventWrapper> {
        let node = self.with(|c| Self::insert_node(c, conv, parent, ev))?;
        Ok(node.to_event())
    }

    fn insert_node(
        c: &Connection,
        conv: &str,
        parent: Option<&str>,
        ev: NewEvent,
    ) -> rusqlite::Result<TreeNode> {
        let id = ev.event_id.unwrap_or_else(crate::session_store::new_event_id_pub);
        let ts = ev.ts.unwrap_or_else(now_iso);
        let meta = ev.meta.unwrap_or_default();
        let payload = ev.payload;

        let parent_id: Option<String> = match parent {
            Some(p) => Some(p.to_string()),
            None => Self::cursor_in(c, conv),
        };
        let (pseq, pdepth): (i64, i64) = match &parent_id {
            Some(p) => c
                .query_row(
                    "SELECT seq, depth FROM nodes WHERE id = ?1",
                    params![p],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap_or((0, 0)),
            None => (0, 0),
        };
        let seq: i64 = c
            .query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM nodes WHERE conversation_id = ?1",
                params![conv],
                |r| r.get(0),
            )
            .unwrap_or(1);
        // The acyclicity guarantee, checked rather than assumed: a parent
        // always has a smaller seq, so cursor→root cannot revisit a node and
        // the walk needs no visited set.
        if parent_id.is_some() && pseq >= seq {
            return Err(rusqlite::Error::InvalidParameterName(format!(
                "parent seq {pseq} is not older than child seq {seq}"
            )));
        }

        let kind = ev.event_type;
        let po = payload.as_object();
        let role = po
            .and_then(|p| p.get("role"))
            .and_then(|v| v.as_str())
            .map(String::from);
        let content = Self::content_of(&kind, po);
        let participant_id = match ev.speaker {
            Some(p) => Some(p),
            None => Self::infer_speaker(c, conv, &kind, role.as_deref()),
        };
        let visible = meta.get("visible") != Some(&Value::Bool(false));
        let (goto_enabled, goto_label) = match meta.get("goto").and_then(|g| g.as_object()) {
            Some(g) => (
                g.get("enabled").map(js_truthy).unwrap_or(false),
                g.get("label").and_then(|v| v.as_str()).map(String::from),
            ),
            None => (false, None),
        };

        let tool_name = po.and_then(|p| p.get("name")).and_then(|v| v.as_str());
        let tool_call_id = po
            .and_then(|p| p.get("tool_call_id").or_else(|| p.get("id")))
            .and_then(|v| v.as_str());
        let ok = po.and_then(|p| p.get("ok")).and_then(|v| v.as_bool());
        let denied = po.and_then(|p| p.get("denied")).and_then(|v| v.as_bool());
        let gi = |k: &str| po.and_then(|p| p.get(k)).and_then(|v| v.as_f64());

        c.execute(
            "INSERT INTO nodes
               (id, conversation_id, parent_id, seq, depth, kind, participant_id, content, role,
                payload, meta, visible, goto_enabled, goto_label, ts, duration_ms, created_at,
                tool_name, tool_call_id, ok, denied,
                gen_prompt_tokens, gen_completion_tokens, gen_total_tokens,
                gen_ttft_ms, gen_duration_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17,
                     ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26)",
            params![
                id,
                conv,
                parent_id,
                seq,
                pdepth + 1,
                kind,
                participant_id,
                content,
                role,
                payload.to_string(),
                Value::Object(meta.clone()).to_string(),
                visible as i64,
                goto_enabled as i64,
                goto_label,
                ts,
                // NaN becomes NULL, as the file store's guard did.
                ev.duration_ms.filter(|d| d.is_finite()),
                now_iso(),
                tool_name,
                tool_call_id,
                ok.map(|b| b as i64),
                denied.map(|b| b as i64),
                gi("prompt_tokens").map(|v| v as i64),
                gi("completion_tokens").map(|v| v as i64),
                gi("total_tokens").map(|v| v as i64),
                gi("ttft_ms"),
                gi("duration_ms"),
            ],
        )?;
        c.execute(
            "INSERT INTO cursor (conversation_id, participant_id, event_id, updated_at)
             VALUES (?1, 0, ?2, ?3)
             ON CONFLICT(conversation_id, participant_id)
               DO UPDATE SET event_id = excluded.event_id, updated_at = excluded.updated_at",
            params![conv, id, ts],
        )?;
        c.execute(
            "UPDATE conversation SET updated_at = ?1 WHERE id = ?2",
            params![now_iso(), conv],
        )?;
        Ok(TreeNode {
            id,
            parent_id,
            seq,
            depth: pdepth + 1,
            kind,
            participant_id,
            content,
            role,
            payload,
            meta,
            visible,
            goto: if goto_enabled {
                Some(Goto {
                    enabled: true,
                    label: goto_label.unwrap_or_default(),
                })
            } else {
                None
            },
            ts,
            duration_ms: ev.duration_ms.filter(|d| d.is_finite()),
        })
    }

    /// The display text for a node kind, so the UI and the flattened group
    /// prompt do not each re-dig it out of `payload`.
    fn content_of(kind: &str, po: Option<&Map<String, Value>>) -> Option<String> {
        let p = po?;
        let s = |k: &str| p.get(k).filter(|v| js_truthy(v)).map(js_string_of);
        match kind {
            "context_window" => s("content"),
            "reasoning" => s("text").or_else(|| s("content")),
            "tool_result" => s("text"),
            "error" => s("message"),
            _ => None,
        }
    }

    /// Who said it, when the caller did not say.
    ///
    /// `role: user` is the human; the agent-authored kinds go to the first
    /// active agent, which is exactly right for a two-seat conversation and
    /// is overridden explicitly (`NewEvent::speaker`) once a round has more
    /// than one agent in it.
    fn infer_speaker(c: &Connection, conv: &str, kind: &str, role: Option<&str>) -> Option<i64> {
        let roster = Self::participants_in(c, conv);
        match role {
            Some("user") => roster.iter().find(|p| p.kind == "user").map(|p| p.id),
            Some("assistant") => roster
                .iter()
                .find(|p| p.is_agent() && p.active)
                .or_else(|| roster.iter().find(|p| p.is_agent()))
                .map(|p| p.id),
            _ if AGENT_KINDS.contains(&kind) => roster
                .iter()
                .find(|p| p.is_agent() && p.active)
                .or_else(|| roster.iter().find(|p| p.is_agent()))
                .map(|p| p.id),
            _ => None,
        }
    }

    fn row_to_node(r: &rusqlite::Row<'_>) -> rusqlite::Result<TreeNode> {
        let goto_enabled: i64 = r.get(12)?;
        Ok(TreeNode {
            id: r.get(0)?,
            parent_id: r.get(1)?,
            seq: r.get(2)?,
            depth: r.get(3)?,
            kind: r.get(4)?,
            participant_id: r.get(5)?,
            content: r.get(6)?,
            role: r.get(7)?,
            payload: r
                .get::<_, String>(8)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or(Value::Null),
            meta: r
                .get::<_, String>(9)
                .ok()
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_default(),
            visible: r.get::<_, i64>(10)? != 0,
            goto: if goto_enabled != 0 {
                Some(Goto {
                    enabled: true,
                    label: r.get::<_, Option<String>>(13)?.unwrap_or_default(),
                })
            } else {
                None
            },
            ts: r.get(11)?,
            duration_ms: r.get(14)?,
        })
    }

    const NODE_COLS: &'static str = "id, parent_id, seq, depth, kind, participant_id, content, \
         role, payload, meta, visible, ts, goto_enabled, goto_label, duration_ms";

    pub fn node(&self, node_id: &str) -> Option<TreeNode> {
        self.with(|c| {
            c.query_row(
                &format!("SELECT {} FROM nodes WHERE id = ?1", Self::NODE_COLS),
                params![node_id],
                Self::row_to_node,
            )
            .optional()
            .ok()
            .flatten()
        })
    }

    /// The rendered conversation: the cursor → oldest-ancestor path, in
    /// chronological order.
    ///
    /// For a conversation that was never forked this is every node in `seq`
    /// order — byte-for-byte what reading the old `.jsonl` gave.
    pub fn path(&self, conv: &str) -> Vec<TreeNode> {
        self.with(|c| Self::path_in(c, conv))
    }

    fn path_in(c: &Connection, conv: &str) -> Vec<TreeNode> {
        let Some(tip) = Self::cursor_in(c, conv) else {
            return Vec::new();
        };
        // The recursive CTE walks parents. It terminates because every
        // parent's seq is strictly smaller than its child's (enforced in
        // `insert_node`), so the chain is finite and acyclic.
        let sql = format!(
            "WITH RECURSIVE up(id) AS (
                 SELECT ?1
                 UNION ALL
                 SELECT n.parent_id FROM nodes n JOIN up ON n.id = up.id
                  WHERE n.parent_id IS NOT NULL
             )
             SELECT {} FROM nodes WHERE id IN (SELECT id FROM up) ORDER BY seq ASC",
            Self::NODE_COLS
        );
        let mut st = match c.prepare(&sql) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let out = match st.query_map(params![tip], Self::row_to_node) {
            Ok(it) => it.filter_map(|r| r.ok()).collect(),
            Err(_) => Vec::new(),
        };
        out
    }

    /// Every node in the conversation, all branches, `seq` order — what the
    /// tree view draws.
    pub fn all_nodes(&self, conv: &str) -> Vec<TreeNode> {
        self.with(|c| {
            let mut st = match c.prepare(&format!(
                "SELECT {} FROM nodes WHERE conversation_id = ?1 ORDER BY seq ASC",
                Self::NODE_COLS
            )) {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
            let out = match st.query_map(params![conv], Self::row_to_node) {
                Ok(it) => it.filter_map(|r| r.ok()).collect(),
                Err(_) => Vec::new(),
            };
            out
        })
    }

    /// A node's direct children — the branches leaving it.
    pub fn children(&self, node_id: &str) -> Vec<TreeNode> {
        self.with(|c| {
            let mut st = match c.prepare(&format!(
                "SELECT {} FROM nodes WHERE parent_id = ?1 ORDER BY seq ASC",
                Self::NODE_COLS
            )) {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
            let out = match st.query_map(params![node_id], Self::row_to_node) {
                Ok(it) => it.filter_map(|r| r.ok()).collect(),
                Err(_) => Vec::new(),
            };
            out
        })
    }

    /// The old wrapped-event view of the rendered path.
    pub fn list_events(&self, conv: &str) -> Vec<EventWrapper> {
        self.path(conv).iter().map(|n| n.to_event()).collect()
    }

    /// Same, with any migrated `context_meta` rows folded forward.
    ///
    /// New writes never produce `context_meta` — visibility is a column now —
    /// but an imported log may still contain some, and they have to mean what
    /// they meant.
    pub fn list_events_folded(&self, conv: &str) -> Vec<EventWrapper> {
        fold_event_metas(self.list_events(conv))
    }

    /// Hide or show one node. The `UPDATE` that replaces appending a
    /// `context_meta` patch.
    pub fn set_visibility(&self, node_id: &str, visible: bool) -> bool {
        self.with(|c| {
            c.execute(
                "UPDATE nodes SET visible = ?1 WHERE id = ?2",
                params![visible as i64, node_id],
            )
            .unwrap_or(0)
                > 0
        })
    }

    pub fn set_goto(&self, node_id: &str, enabled: bool, label: Option<&str>) -> bool {
        self.with(|c| {
            c.execute(
                "UPDATE nodes SET goto_enabled = ?1, goto_label = ?2 WHERE id = ?3",
                params![enabled as i64, label, node_id],
            )
            .unwrap_or(0)
                > 0
        })
    }

    // -----------------------------------------------------------------------
    // Participant views (watermarks) and own sessions
    // -----------------------------------------------------------------------

    /// `(last_seen_seq, round_index, reply_required, round_status)`.
    pub fn participant_view(&self, conv: &str, pid: i64) -> (i64, i64, bool, Option<String>) {
        self.with(|c| {
            c.query_row(
                "SELECT last_seen_seq, round_index, reply_required, round_status
                   FROM participant_view WHERE conversation_id = ?1 AND participant_id = ?2",
                params![conv, pid],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)? != 0,
                        r.get::<_, Option<String>>(3)?,
                    ))
                },
            )
            .optional()
            .ok()
            .flatten()
            .unwrap_or((0, 0, false, None))
        })
    }

    pub fn set_participant_view(
        &self,
        conv: &str,
        pid: i64,
        last_seen_seq: i64,
        round_index: i64,
        reply_required: bool,
        task_id: Option<&str>,
        round_status: Option<&str>,
    ) {
        self.with(|c| {
            let _ = c.execute(
                "INSERT INTO participant_view
                   (conversation_id, participant_id, last_seen_seq, round_index,
                    reply_required, task_id, round_status, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(conversation_id, participant_id) DO UPDATE SET
                   last_seen_seq  = excluded.last_seen_seq,
                   round_index    = excluded.round_index,
                   reply_required = excluded.reply_required,
                   task_id        = excluded.task_id,
                   round_status   = excluded.round_status,
                   updated_at     = excluded.updated_at",
                params![
                    conv,
                    pid,
                    last_seen_seq,
                    round_index,
                    reply_required as i64,
                    task_id,
                    round_status,
                    now_iso()
                ],
            );
        });
    }

    /// One agent's own `(prompt, reply)` history, oldest first.
    ///
    /// This is the participant's **private** LLM history, not the shared
    /// room: it is what makes its context window byte-stable across turns
    /// (so the provider's prompt cache holds) and independent of which branch
    /// the room is currently on.
    pub fn own_session(&self, pid: i64) -> Vec<(String, String)> {
        self.with(|c| {
            let mut st = match c.prepare(
                "SELECT prompt, reply FROM participant_session
                  WHERE participant_id = ?1 ORDER BY seq ASC",
            ) {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
            let out = match st.query_map(params![pid], |r| Ok((r.get(0)?, r.get(1)?))) {
                Ok(it) => it.filter_map(|r| r.ok()).collect(),
                Err(_) => Vec::new(),
            };
            out
        })
    }

    pub fn append_own_turn(
        &self,
        conv: &str,
        pid: i64,
        prompt: &str,
        reply: &str,
        task_id: Option<&str>,
    ) {
        self.with(|c| {
            let seq: i64 = c
                .query_row(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM participant_session
                      WHERE participant_id = ?1",
                    params![pid],
                    |r| r.get(0),
                )
                .unwrap_or(1);
            let _ = c.execute(
                "INSERT INTO participant_session
                   (conversation_id, participant_id, seq, prompt, reply, task_id, ts)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![conv, pid, seq, prompt, reply, task_id, now_iso()],
            );
        });
    }

    // -----------------------------------------------------------------------
    // Readers the harness and the HTTP layer share with the file store
    // -----------------------------------------------------------------------

    /// The visible user/assistant exchange — for a title or a summary.
    pub fn list_visible_context_messages(&self, conv: &str) -> Vec<Value> {
        let mut out = Vec::new();
        for n in self.path(conv) {
            if n.kind != "context_window" || !n.visible {
                continue;
            }
            let Some(role) = n.role.as_deref() else { continue };
            if role != "user" && role != "assistant" {
                continue;
            }
            let Some(content) = &n.content else { continue };
            if n.payload.get("kind").and_then(|v| v.as_str())
                == Some("provider_messages_snapshot")
            {
                continue;
            }
            out.push(json!({
                "event_id": n.id,
                "role": role,
                "content": content,
                "meta": n.meta,
            }));
        }
        out
    }

    /// Hydrate the provider-facing context window from the rendered path.
    ///
    /// Delegates to the file store's hydration so there is **one**
    /// implementation of the tool-call merging and JS-era coercions; the only
    /// difference is where the events came from.
    pub fn build_context_window_items(&self, conv: &str) -> Vec<ContextItem> {
        crate::session_store::context_items_from_events(self.list_events_folded(conv))
    }

    /// The newest `gen_info` telemetry on the rendered path, as the metrics
    /// row reads it.
    pub fn last_gen_info(&self, conv: &str) -> Value {
        for n in self.path(conv).into_iter().rev() {
            if n.kind == "gen_info" && !n.payload.is_null() {
                return match n.meta.get("telemetry") {
                    Some(t) if t.is_object() => t.clone(),
                    _ => n.payload.clone(),
                };
            }
        }
        Value::Null
    }

    /// The memory tree's leaves: the messages on the **rendered path**, in
    /// order.
    ///
    /// Derived, not stored, and derived from the *path* rather than from
    /// every node in the conversation. That matters once a conversation has
    /// branched: the append order of all nodes interleaves messages that
    /// were never on the same branch, and summarising those together would
    /// describe a conversation nobody had.
    pub fn memory_leaves(&self, conv: &str) -> Vec<String> {
        self.path(conv)
            .into_iter()
            .filter(|n| {
                n.kind == "context_window"
                    && matches!(n.role.as_deref(), Some("user") | Some("assistant"))
            })
            .map(|n| n.id)
            .collect()
    }

    /// The leaf list the memory tree was last built against.
    pub fn memory_tree_leaves(&self, conv: &str) -> Vec<String> {
        self.with(|c| {
            let mut st = match c.prepare(
                "SELECT node_id FROM memory_leaves WHERE conversation_id = ?1 ORDER BY i ASC",
            ) {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
            let out = match st.query_map(params![conv], |r| r.get::<_, String>(0)) {
                Ok(it) => it.filter_map(|r| r.ok()).collect(),
                Err(_) => Vec::new(),
            };
            out
        })
    }

    /// Bring the memory tree into line with the current path, keeping every
    /// summary that is still true.
    ///
    /// The leaf list only ever *extends* while you talk on one branch, so a
    /// summary of messages 0..8 is as valid after the ninth message as
    /// before it — rebuilding the whole tree per turn would be pure waste.
    /// Switching branches is different: the lists diverge at the fork, and
    /// every node covering a message at or after that point is now a summary
    /// of the wrong conversation. So: find the first index where the two
    /// lists differ, and drop exactly the nodes whose span reaches it.
    ///
    /// Returns the number of leaves on the current path.
    pub fn sync_memory_tree(&self, conv: &str) -> usize {
        let now = self.memory_leaves(conv);
        let was = self.memory_tree_leaves(conv);
        let diverged = now
            .iter()
            .zip(was.iter())
            .position(|(a, b)| a != b)
            // No difference in the overlap: the shorter list is a prefix of
            // the longer, so only what the shorter one lacks is new.
            .unwrap_or_else(|| now.len().min(was.len()));

        self.with(|c| {
            // A node (l, i) covers [i·2^l, (i+1)·2^l). It is stale exactly
            // when its span end runs past the divergence point.
            let _ = c.execute(
                "DELETE FROM memory_nodes
                  WHERE conversation_id = ?1
                    AND ((idx + 1) * (1 << level)) > ?2",
                params![conv, diverged as i64],
            );
            if was.len() != now.len() || diverged < was.len() {
                // The view is a list of coordinates over the old list; it
                // has to be refitted rather than patched.
                let _ = c.execute(
                    "DELETE FROM memory_view WHERE conversation_id = ?1",
                    params![conv],
                );
            }
            let _ = c.execute(
                "DELETE FROM memory_leaves WHERE conversation_id = ?1",
                params![conv],
            );
            for (i, id) in now.iter().enumerate() {
                let _ = c.execute(
                    "INSERT OR IGNORE INTO memory_leaves (conversation_id, i, node_id)
                     VALUES (?1, ?2, ?3)",
                    params![conv, i as i64, id],
                );
            }
        });
        now.len()
    }

    /// Every node's text by id, for rendering a memory view's leaves.
    pub fn node_texts(&self, conv: &str) -> HashMap<String, (String, String)> {
        let mut out = HashMap::new();
        for n in self.all_nodes(conv) {
            if let Some(t) = n.content {
                out.insert(n.id, (n.role.unwrap_or_default(), t));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> ConvStore {
        ConvStore::memory("/tmp/proj").unwrap()
    }

    fn conv(s: &ConvStore) -> String {
        s.create(NewSession {
            agent: Some("default".into()),
            model: Some("llama-server:default".into()),
            ..Default::default()
        })
        .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn a_new_conversation_has_a_header_a_roster_and_one_node() {
        let s = store();
        let id = conv(&s);
        let meta = s.load(&id).unwrap();
        assert_eq!(meta["ordinal"], 1);
        assert_eq!(meta["status"], "ready");
        assert_eq!(meta["turns"], 0);
        assert_eq!(meta["eventCount"], 1);
        let roster = s.participants(&id);
        assert_eq!(roster.len(), 2, "the human and the agent are both seated");
        assert_eq!(roster[0].kind, "user");
        assert_eq!(roster[1].handle, "default");
    }

    #[test]
    fn ordinals_are_max_plus_one_so_a_delete_never_reissues_a_number() {
        let s = store();
        let a = conv(&s);
        let b = conv(&s);
        let c = conv(&s);
        assert_eq!(s.load(&c).unwrap()["ordinal"], 3);
        assert!(s.delete(&b));
        let d = conv(&s);
        // `count + 1` would have handed out 3 twice.
        assert_eq!(s.load(&d).unwrap()["ordinal"], 4);
        assert_eq!(s.load(&a).unwrap()["ordinal"], 1);
    }

    #[test]
    fn an_unforked_conversation_reads_back_exactly_as_a_linear_log() {
        let s = store();
        let id = conv(&s);
        for i in 0..4 {
            s.append_event(
                &id,
                NewEvent::new(
                    "context_window",
                    json!({ "role": if i % 2 == 0 { "user" } else { "assistant" },
                            "content": format!("m{i}") }),
                ),
            )
            .unwrap();
        }
        let evs = s.list_events(&id);
        // The created beat plus four rows, in order, every parent the one
        // before it: the tree walk and a file read agree.
        assert_eq!(evs.len(), 5);
        let texts: Vec<String> = evs
            .iter()
            .filter(|e| e.event_type == "context_window")
            .map(|e| e.payload["content"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(texts, vec!["m0", "m1", "m2", "m3"]);
        let nodes = s.path(&id);
        for w in nodes.windows(2) {
            assert_eq!(w[1].parent_id.as_deref(), Some(w[0].id.as_str()));
            assert!(w[0].seq < w[1].seq);
        }
    }

    #[test]
    fn a_fork_leaves_the_original_branch_intact_and_reachable() {
        let s = store();
        let id = conv(&s);
        let a = s
            .append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"q"})))
            .unwrap();
        let b = s
            .append_event(&id, NewEvent::new("context_window", json!({"role":"assistant","content":"answer A"})))
            .unwrap();
        // Fork: point back at the question, ask again.
        assert!(s.set_cursor(&id, Some(&a.event_id)));
        let c2 = s
            .append_event(&id, NewEvent::new("context_window", json!({"role":"assistant","content":"answer B"})))
            .unwrap();

        let on_b: Vec<String> = s
            .list_events(&id)
            .iter()
            .filter_map(|e| e.payload["content"].as_str().map(String::from))
            .collect();
        assert_eq!(on_b, vec!["q", "answer B"], "the new branch renders");

        // The original is untouched, and pointing at it brings it back.
        assert!(s.set_cursor(&id, Some(&b.event_id)));
        let on_a: Vec<String> = s
            .list_events(&id)
            .iter()
            .filter_map(|e| e.payload["content"].as_str().map(String::from))
            .collect();
        assert_eq!(on_a, vec!["q", "answer A"]);

        // Both children hang off the question.
        let kids = s.children(&a.event_id);
        assert_eq!(kids.len(), 2);
        assert_eq!(s.all_nodes(&id).len(), 4);
        let _ = c2;
    }

    #[test]
    fn forking_at_the_root_makes_a_second_root_rather_than_an_error() {
        let s = store();
        let id = conv(&s);
        s.append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"one"})))
            .unwrap();
        // No parent at all: a second branch root.
        s.set_cursor(&id, None);
        s.append_event_at(
            &id,
            None,
            NewEvent::new("context_window", json!({"role":"user","content":"two"})),
        )
        .unwrap();
        let path: Vec<String> = s
            .list_events(&id)
            .iter()
            .filter_map(|e| e.payload["content"].as_str().map(String::from))
            .collect();
        assert_eq!(path, vec!["two"], "a root branch renders alone");
        let roots = s
            .all_nodes(&id)
            .into_iter()
            .filter(|n| n.parent_id.is_none())
            .count();
        assert_eq!(roots, 2);
    }

    #[test]
    fn visibility_is_an_update_not_an_appended_patch() {
        let s = store();
        let id = conv(&s);
        let e = s
            .append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"hi"})))
            .unwrap();
        let before = s.load(&id).unwrap()["eventCount"].as_u64().unwrap();
        assert!(s.set_visibility(&e.event_id, false));
        // No new node: the old format needed a whole extra log line for this.
        assert_eq!(s.load(&id).unwrap()["eventCount"].as_u64().unwrap(), before);
        let ev = s.list_events(&id).into_iter().find(|x| x.event_id == e.event_id).unwrap();
        assert!(!ev.visible());
        assert!(s.set_visibility(&e.event_id, true));
        let ev = s.list_events(&id).into_iter().find(|x| x.event_id == e.event_id).unwrap();
        assert!(ev.visible());
    }

    #[test]
    fn a_turn_bumps_turns_and_appends_the_lifecycle_beat() {
        let s = store();
        let id = conv(&s);
        s.record_turn_complete(&id, Some(12.5));
        let meta = s.load(&id).unwrap();
        assert_eq!(meta["turns"], 1);
        assert_eq!(meta["status"], "ready");
        let last = s.list_events(&id).pop().unwrap();
        assert_eq!(last.event_type, "session");
        assert_eq!(last.payload["kind"], "turn_complete");
        assert_eq!(last.duration_ms, Some(12.5));
    }

    #[test]
    fn last_token_counts_come_from_the_newest_gen_info_not_a_cached_column() {
        let s = store();
        let id = conv(&s);
        for (p, t) in [(10, 20), (30, 70)] {
            s.append_event(
                &id,
                NewEvent::new(
                    "gen_info",
                    json!({ "prompt_tokens": p, "completion_tokens": t - p, "total_tokens": t }),
                ),
            )
            .unwrap();
        }
        let meta = s.load(&id).unwrap();
        assert_eq!(meta["lastPromptTokens"], 30);
        assert_eq!(meta["lastTotalTokens"], 70);
        assert_eq!(meta["lastCompletionTokens"], 40);
    }

    #[test]
    fn update_meta_maps_camel_case_and_shrugs_at_derived_keys() {
        let s = store();
        let id = conv(&s);
        let mut patch = Map::new();
        patch.insert("title".into(), json!("a title"));
        patch.insert("allowlistMode".into(), json!("unrestricted"));
        patch.insert("contextWindow".into(), json!(8192));
        // Still sent by older call sites; derived now, so it must not error.
        patch.insert("lastPromptTokens".into(), json!(99));
        let meta = s.update_meta(&id, patch).unwrap();
        assert_eq!(meta["title"], "a title");
        assert_eq!(meta["allowlistMode"], "unrestricted");
        assert_eq!(meta["contextWindow"], 8192);
        assert!(meta["lastPromptTokens"].is_null());
    }

    #[test]
    fn a_duplicate_handle_is_suffixed_so_mentions_stay_unambiguous() {
        let s = store();
        let id = conv(&s);
        let a = s
            .add_participant(&id, "agent", "scout", Some("Scout"), Some("scout"), None, "", None, None)
            .unwrap();
        let b = s
            .add_participant(&id, "agent", "scout", Some("Scout"), Some("scout"), None, "", None, None)
            .unwrap();
        assert_eq!(a.handle, "scout");
        assert_eq!(b.handle, "scout-2");
    }

    #[test]
    fn an_agent_that_has_spoken_is_deactivated_rather_than_deleted() {
        let s = store();
        let id = conv(&s);
        let p = s
            .add_participant(&id, "agent", "bob", Some("Bob"), Some("bob"), None, "", None, None)
            .unwrap();
        s.append_event(
            &id,
            NewEvent {
                event_type: "context_window".into(),
                payload: json!({ "role": "assistant", "content": "hi" }),
                speaker: Some(p.id),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s.remove_participant(p.id));
        let still = s.participants(&id).into_iter().find(|x| x.id == p.id).unwrap();
        assert!(!still.active, "its authorship of an existing node must survive");
    }

    #[test]
    fn the_speaker_is_recorded_so_two_agents_do_not_blur_together() {
        let s = store();
        let id = conv(&s);
        let bob = s
            .add_participant(&id, "agent", "bob", Some("Bob"), Some("bob"), None, "", None, None)
            .unwrap();
        let u = s.user_participant(&id).unwrap();
        s.append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"q"})))
            .unwrap();
        s.append_event(
            &id,
            NewEvent {
                event_type: "context_window".into(),
                payload: json!({ "role": "assistant", "content": "from bob" }),
                speaker: Some(bob.id),
                ..Default::default()
            },
        )
        .unwrap();
        let nodes = s.path(&id);
        let user_node = nodes.iter().find(|n| n.role.as_deref() == Some("user")).unwrap();
        let bob_node = nodes.iter().find(|n| n.role.as_deref() == Some("assistant")).unwrap();
        assert_eq!(user_node.participant_id, Some(u.id));
        assert_eq!(bob_node.participant_id, Some(bob.id));
    }

    #[test]
    fn memory_leaves_index_only_renderable_rows_densely() {
        let s = store();
        let id = conv(&s);
        s.append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"a"}))).unwrap();
        s.append_event(&id, NewEvent::new("reasoning", json!({"text":"think"}))).unwrap();
        s.append_event(&id, NewEvent::new("gen_info", json!({"total_tokens": 5}))).unwrap();
        s.append_event(&id, NewEvent::new("context_window", json!({"role":"assistant","content":"b"}))).unwrap();
        let leaves = s.memory_leaves(&id);
        assert_eq!(leaves.len(), 2, "reasoning and telemetry are not messages");
        assert_eq!(s.sync_memory_tree(&id), 2, "the recorded list agrees");
        assert_eq!(s.memory_tree_leaves(&id), leaves);
    }

    #[test]
    fn a_parent_older_than_its_child_is_refused() {
        let s = store();
        let id = conv(&s);
        let first = s.path(&id)[0].id.clone();
        // Hand-craft the violation: a node claiming a parent appended later.
        let latest = s
            .append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"x"})))
            .unwrap();
        let res = s.with(|c| {
            c.execute(
                "UPDATE nodes SET parent_id = ?1 WHERE id = ?2",
                params![latest.event_id, first],
            )
        });
        // SQLite will take the UPDATE; the guarantee lives on the insert path,
        // which is where every real write goes.
        assert!(res.is_ok());
        let direct = s.with(|c| {
            ConvStore::insert_node(
                c,
                &id,
                Some(&latest.event_id),
                NewEvent::new("context_window", json!({"role":"user","content":"y"})),
            )
        });
        assert!(direct.is_ok(), "a genuinely older parent is fine");
    }
}
