use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Map, Value};

use crate::session_store::{
    fold_event_metas, js_string_of, js_truthy, new_session_id, now_iso, ContextItem, EventWrapper,
    Goto, NewEvent, NewSession,
};

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
    pub engagement: Option<String>,
}

impl Participant {
    pub fn is_agent(&self) -> bool {
        self.kind == "agent"
    }

    pub fn to_json(&self) -> Value {
        self.to_json_with(None)
    }

    pub fn to_json_with(&self, def: Option<&crate::agent_loader::AgentDef>) -> Value {
        json!({
            "id": self.id,
            "accentColor": def.and_then(|d| d.accent_color.as_deref()),
            "icon": def.and_then(|d| d.icon.as_deref()),
            "kind": self.kind,
            "handle": self.handle,
            "displayName": self.display_name,
            "agentName": self.agent_name,
            "model": self.model,
            "avatar": self.avatar,
            "seat": self.seat,
            "active": self.active,
            "role": self.role,
            "engagement": self.engagement,
        })
    }
}

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
    pub fn to_event(&self) -> EventWrapper {
        let mut meta = self.meta.clone();
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

pub struct ConvStore {
    pub project_root: PathBuf,
    conn: Mutex<Connection>,
}


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

fn worktree_record(raw: Option<String>) -> Value {
    let Some(s) = raw else { return Value::Null };
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Value::Null;
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(v) if v.is_object() => v,
        _ => Value::Null,
    }
}

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

    pub fn memory(project_root: impl AsRef<Path>) -> rusqlite::Result<Self> {
        Ok(ConvStore {
            project_root: project_root.as_ref().to_path_buf(),
            conn: Mutex::new(crate::db::open_memory()?),
        })
    }

    pub fn with<T>(&self, f: impl FnOnce(&Connection) -> T) -> T {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        f(&conn)
    }

    pub fn identity(&self) -> crate::identity::Identity<'_> {
        crate::identity::Identity { conn: &self.conn }
    }


    pub fn create(&self, meta: NewSession) -> rusqlite::Result<Value> {
        let id = new_session_id();
        let root = self.project_root.to_string_lossy().into_owned();
        self.with(|c| {
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
                    mcp, project_root, config, overrides, allowlist_mode, turns,
                    created_at, updated_at, routing)
                 VALUES (?1, ?2, 'conv/v1', 'conversation', ?3, ?4, ?5, ?6, 'ready',
                         ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?17, 0, ?16, ?16, '')",
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
                    meta.allowlist_mode.clone().unwrap_or_else(|| "unrestricted".into()),
                ],
            )?;
            c.execute(
                "INSERT INTO cursor (conversation_id, participant_id, event_id, updated_at)
                 VALUES (?1, 0, NULL, ?2)",
                params![id, now],
            )?;
            Self::seed_roster(c, &id)?;
            Ok::<(), rusqlite::Error>(())
        })?;
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

    fn seed_roster(c: &Connection, conv: &str) -> rusqlite::Result<()> {
        c.execute(
            "INSERT OR IGNORE INTO participants
               (conversation_id, kind, handle, display_name, seat, active)
             VALUES (?1, 'user', 'user', 'User', 0, 1)",
            params![conv],
        )?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> Option<Value> {
        self.with(|c| Self::load_in(c, id))
    }

    fn load_in(c: &Connection, id: &str) -> Option<Value> {
        let mut out = c
            .query_row(
                "SELECT id, ordinal, api_version, kind, title, agent, agent_file, model, status,
                        policy_mode, approve_mode, allowlist_mode, allowlist, allowlist_source,
                        context_window, mcp, project_root, project, sandbox, worktree,
                        branch_ref, config, overrides, turns,
                        round_index, round_status, routing, purpose, notes, memory_high,
                        legacy_format, created_at, updated_at, workspace_root, soul_appendage
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
                        "project": r.get::<_, Option<String>>(17)?.unwrap_or_default(),
                        "sandboxId": r.get::<_, Option<String>>(18)?.unwrap_or_default(),
                        "worktreeId": r.get::<_, Option<String>>(19)?.unwrap_or_default(),
                        "branchId": r.get::<_, Option<String>>(20)?.unwrap_or_default(),
                        "config": from_json_text(r.get::<_, Option<String>>(21)?),
                        "overrides": from_json_text(r.get::<_, Option<String>>(22)?),
                        "turns": r.get::<_, i64>(23)?,
                        "roundIndex": r.get::<_, i64>(24)?,
                        "roundStatus": r.get::<_, Option<String>>(25)?,
                        "routing": r.get::<_, String>(26)?,
                        "purpose": r.get::<_, Option<String>>(27)?,
                        "notes": r.get::<_, Option<String>>(28)?,
                        "memoryHigh": r.get::<_, Option<i64>>(29)?,
                        "legacyFormat": r.get::<_, Option<String>>(30)?,
                        "createdAt": r.get::<_, String>(31)?,
                        "updatedAt": r.get::<_, String>(32)?,
                        "workspaceRoot": r.get::<_, Option<String>>(33)?,
                        "soulAppendage": r.get::<_, Option<String>>(34)?,
                        "worktree": worktree_record(r.get::<_, Option<String>>(19)?),
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
            "workspaceRoot" | "workspace_root" => "workspace_root",
            "project" => "project",
            "sandbox" | "sandboxId" => "sandbox",
            "worktree" | "worktreeId" => "worktree",
            "branchRef" | "branch_ref" | "branchId" => "branch_ref",
            "turns" => "turns",
            "roundIndex" | "round_index" => "round_index",
            "roundStatus" | "round_status" => "round_status",
            "routing" => "routing",
            "purpose" => "purpose",
            "soulAppendage" | "soul_appendage" => "soul_appendage",
            "notes" => "notes",
            "memoryHigh" | "memory_high" => "memory_high",
            "mcp" => "mcp",
            "config" => "config",
            "overrides" => "overrides",
            "approveMode" | "approve_mode" => "approve_mode",
            _ => return None,
        })
    }

    const JSON_COLUMNS: &'static [&'static str] =
        &["mcp", "config", "overrides", "approve_mode", "worktree"];

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

    pub fn delete(&self, id: &str) -> bool {
        if id.is_empty() || id.contains("..") || id.contains('/') || id.contains('\\') {
            return false;
        }
        crate::sandbox::destroy_for_session(self, id);
        crate::session_props::on_session_deleted(id);
        let n = self.with(|c| {
            let _ = c.execute("PRAGMA defer_foreign_keys = ON", []);
            let n = c
                .execute("DELETE FROM conversation WHERE id = ?1", params![id])
                .unwrap_or(0);
            let _ = c.execute("DELETE FROM nodes WHERE conversation_id = ?1", params![id]);
            let _ = c.execute("DELETE FROM queue_items WHERE conversation_id = ?1", params![id]);
            n
        });
        let legacy = self.project_root.join(".angela").join("sessions");
        for ext in ["json", "jsonl"] {
            let p = legacy.join(format!("{id}.{ext}"));
            if p.exists() {
                let _ = std::fs::remove_file(p);
            }
        }
        n > 0
    }

    pub fn clean(&self) -> usize {
        crate::sandbox::destroy_all(self);
        self.with(|c| {
            let _ = c.execute("PRAGMA defer_foreign_keys = ON", []);
            let n = c.execute("DELETE FROM conversation", []).unwrap_or(0);
            let _ = c.execute("DELETE FROM nodes", []);
            let _ = c.execute("DELETE FROM queue_items", []);
            n
        })
    }

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


    pub fn participants(&self, conv: &str) -> Vec<Participant> {
        self.with(|c| Self::participants_in(c, conv))
    }

    fn participants_in(c: &Connection, conv: &str) -> Vec<Participant> {
        let mut st = match c.prepare(
            "SELECT id, conversation_id, kind, handle, display_name, agent_name, model,
                    system_prompt, avatar, seat, active, role, engagement
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
                engagement: r.get(12)?,
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

    pub fn invite_mentioned_agent(&self, conv: &str, def: &crate::agent_loader::AgentDef) -> Option<Participant> {
        let handle = def.name.to_ascii_lowercase();
        self.with(|c| {
            let roster = Self::participants_in(c, conv);
            if let Some(existing) = roster.iter().find(|p| p.handle == handle) {
                if existing.kind != "agent" || existing.agent_name.as_deref() != Some(def.name.as_str()) || existing.active {
                    return None;
                }
                c.execute("UPDATE participants SET active = 1 WHERE id = ?1", params![existing.id]).ok()?;
                return Self::participants_in(c, conv).into_iter().find(|p| p.id == existing.id);
            }
            if roster.iter().any(|p| p.kind == "agent" && p.agent_name.as_deref() == Some(def.name.as_str())) {
                return None;
            }
            let seat: i64 = c.query_row(
                "SELECT COALESCE(MAX(seat), -1) + 1 FROM participants WHERE conversation_id = ?1",
                params![conv], |r| r.get(0),
            ).ok()?;
            c.execute(
                "INSERT INTO participants (conversation_id, kind, handle, display_name, agent_name, model, system_prompt, seat, active, role) VALUES (?1, 'agent', ?2, ?2, ?3, ?4, ?5, ?6, 1, 'member')",
                params![conv, handle, def.name, if def.model.is_empty() { None } else { Some(def.model.as_str()) }, def.system_prompt(), seat],
            ).ok()?;
            let id = c.last_insert_rowid();
            Self::participants_in(c, conv).into_iter().find(|p| p.id == id)
        })
    }

    pub fn invite_user(&self, conv: &str, name: &str, display_name: &str) -> Option<Participant> {
        let handle = handle_from(name).to_ascii_lowercase();
        self.with(|c| {
            let roster = Self::participants_in(c, conv);
            if let Some(existing) = roster.iter().find(|p| p.handle == handle) {
                if existing.active {
                    return None;
                }
                c.execute("UPDATE participants SET active = 1 WHERE id = ?1", params![existing.id]).ok()?;
                return Self::participants_in(c, conv).into_iter().find(|p| p.id == existing.id);
            }
            let seat: i64 = c.query_row(
                "SELECT COALESCE(MAX(seat), -1) + 1 FROM participants WHERE conversation_id = ?1",
                params![conv], |r| r.get(0),
            ).ok()?;
            c.execute(
                "INSERT INTO participants (conversation_id, kind, handle, display_name, seat, active, role) VALUES (?1, 'user', ?2, ?3, ?4, 1, 'member')",
                params![conv, handle, display_name, seat],
            ).ok()?;
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
                    "engagement" => "engagement",
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

    pub fn agent_participants(&self, conv: &str) -> Vec<Participant> {
        self.participants(conv)
            .into_iter()
            .filter(|p| p.is_agent() && p.active)
            .collect()
    }


    pub fn sandbox_for(&self, conv: &str, agent: &str) -> Option<String> {
        self.with(|c| {
            c.query_row(
                "SELECT container_id FROM session_sandbox WHERE session_id = ?1 AND agent_name = ?2",
                params![conv, agent],
                |r| r.get(0),
            ).optional().ok().flatten()
        })
    }

    pub fn set_sandbox(&self, conv: &str, agent: &str, container_id: &str) {
        self.with(|c| {
            let _ = c.execute(
                "DELETE FROM session_sandbox WHERE session_id = ?1 AND agent_name = ?2",
                params![conv, agent],
            );
            let _ = c.execute(
                "INSERT INTO session_sandbox (session_id, agent_name, container_id, created_at) \
                 VALUES (?1, ?2, ?3, ?4)",
                params![conv, agent, container_id, now_iso()],
            );
        })
    }

    pub fn sandbox_mount(&self, conv: &str, agent: &str) -> Option<(String, String)> {
        self.with(|c| c.query_row(
            "SELECT mount_path, git_dir FROM session_sandbox WHERE session_id = ?1 AND agent_name = ?2",
            params![conv, agent], |r| Ok((r.get::<_, Option<String>>(0)?, r.get::<_, Option<String>>(1)?)),
        ).optional().ok().flatten().and_then(|(path, git)| Some((path?, git?))))
    }

    pub fn set_sandbox_mount(&self, conv: &str, agent: &str, path: &str, git: &str) -> rusqlite::Result<()> {
        self.with(|c| c.execute(
            "UPDATE session_sandbox SET mount_path = ?1, git_dir = ?2 WHERE session_id = ?3 AND agent_name = ?4",
            params![path, git, conv, agent],
        ).and_then(|n| if n == 1 { Ok(()) } else { Err(rusqlite::Error::QueryReturnedNoRows) }))
    }

    pub fn sandboxes_for(&self, conv: &str) -> Vec<(String, String)> {
        self.with(|c| {
            let mut out = Vec::new();
            if let Ok(mut st) = c.prepare(
                "SELECT agent_name, container_id FROM session_sandbox WHERE session_id = ?1",
            ) {
                if let Ok(it) = st.query_map(params![conv], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                }) {
                    for r in it { out.push(r.unwrap()); }
                }
            }
            out
        })
    }

    pub fn forget_sandbox(&self, conv: &str, agent: &str) -> bool {
        self.with(|c| {
            let n = c.execute(
                "DELETE FROM session_sandbox WHERE session_id = ?1 AND agent_name = ?2",
                params![conv, agent],
            ).unwrap_or(0);
            n > 0
        })
    }


    pub fn forget_sandboxes_by_session(&self, conv: &str) {
        self.with(|c| {
            let _ = c.execute(
                "DELETE FROM session_sandbox WHERE session_id = ?1",
                params![conv],
            );
        })
    }

    pub fn sandbox_rows(&self) -> Vec<(String, String, String)> {
        self.with(|c| {
            let mut out = Vec::new();
            if let Ok(mut st) = c.prepare(
                "SELECT session_id, agent_name, container_id FROM session_sandbox",
            ) {
                if let Ok(it) = st.query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                }) {
                    out.extend(it.filter_map(|r| r.ok()));
                }
            }
            out
        })
    }


    pub fn queue_push(
        &self,
        conv: &str,
        text: &str,
        from: &str,
        agent: &str,
        model: &str,
        tab_id: &str,
    ) -> String {
        let id = format!("q{}", &crate::session_store::new_event_id_pub()[1..]);
        let _ = self.with(|c| {
            let seq: i64 = c
                .query_row(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM queue_items WHERE conversation_id = ?1",
                    params![conv],
                    |r| r.get(0),
                )
                .unwrap_or(1);
            c.execute(
                "INSERT INTO queue_items
                   (id, conversation_id, seq, text, from_name, agent, model, tab_id, reason, ts)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![id, conv, seq, text, from, agent, model, tab_id, "", now_iso()],
            )
        });
        id
    }

    pub fn queue_row(&self, conv: &str, id: &str) -> Option<Value> {
        self.with(|c| {
            c.query_row(
                "SELECT id, text, from_name, agent, model, tab_id, reason, ts
                   FROM queue_items WHERE id = ?1 AND conversation_id = ?2",
                params![id, conv],
                |r| {
                    Ok(json!({
                        "id": r.get::<_, String>(0)?,
                        "text": r.get::<_, String>(1)?,
                        "from": r.get::<_, String>(2)?,
                        "agent": r.get::<_, String>(3)?,
                        "model": r.get::<_, String>(4)?,
                        "tabId": r.get::<_, String>(5)?,
                        "reason": r.get::<_, String>(6)?,
                        "ts": r.get::<_, String>(7)?,
                    }))
                },
            )
            .optional()
            .ok()
            .flatten()
        })
    }

    pub fn queue_list(&self, conv: &str) -> Vec<Value> {
        self.with(|c| {
            let mut st = match c.prepare(
                "SELECT id, text, from_name, reason, ts FROM queue_items
                  WHERE conversation_id = ?1 ORDER BY seq ASC",
            ) {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
            let out = match st.query_map(params![conv], |r| {
                Ok(json!({
                    "id": r.get::<_, String>(0)?,
                    "text": r.get::<_, String>(1)?,
                    "from": r.get::<_, String>(2)?,
                    "reason": r.get::<_, String>(3)?,
                    "ts": r.get::<_, String>(4)?,
                }))
            }) {
                Ok(it) => it.filter_map(|r| r.ok()).collect(),
                Err(_) => Vec::new(),
            };
            out
        })
    }

    pub fn queue_all(&self) -> Vec<Value> {
        self.with(|c| {
            let mut st = match c.prepare(
                "SELECT id, conversation_id, text, from_name, agent, model, tab_id, reason, ts
                   FROM queue_items ORDER BY conversation_id, seq ASC",
            ) {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
            let out = match st.query_map([], |r| {
                Ok(json!({
                    "id": r.get::<_, String>(0)?,
                    "sessionId": r.get::<_, String>(1)?,
                    "text": r.get::<_, String>(2)?,
                    "from": r.get::<_, String>(3)?,
                    "agent": r.get::<_, String>(4)?,
                    "model": r.get::<_, String>(5)?,
                    "tabId": r.get::<_, String>(6)?,
                    "reason": r.get::<_, String>(7)?,
                    "ts": r.get::<_, String>(8)?,
                }))
            }) {
                Ok(it) => it.filter_map(|r| r.ok()).collect(),
                Err(_) => Vec::new(),
            };
            out
        })
    }

    pub fn queue_remove(&self, conv: &str, id: &str) -> Option<Value> {
        let row = self.queue_row(conv, id)?;
        self.with(|c| {
            let _ = c.execute(
                "DELETE FROM queue_items WHERE id = ?1 AND conversation_id = ?2",
                params![id, conv],
            );
        });
        Some(row)
    }

    pub fn queue_set_reason(&self, conv: &str, id: &str, reason: &str) -> bool {
        self.with(|c| {
            c.execute(
                "UPDATE queue_items SET reason = ?1 WHERE id = ?2 AND conversation_id = ?3",
                params![reason, id, conv],
            )
            .unwrap_or(0)
                > 0
        })
    }

    pub fn sessions_seating(&self, user_name: &str) -> Vec<String> {
        let handle = handle_from(user_name).to_ascii_lowercase();
        self.with(|c| {
            let mut out = Vec::new();
            if let Ok(mut st) = c.prepare(
                "SELECT DISTINCT conversation_id FROM participants \
                 WHERE kind = 'user' AND handle = ?1",
            ) {
                if let Ok(it) = st.query_map(params![handle], |r| r.get::<_, String>(0)) {
                    out.extend(it.filter_map(|r| r.ok()));
                }
            }
            out
        })
    }

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

    pub fn append_event(&self, conv: &str, ev: NewEvent) -> rusqlite::Result<EventWrapper> {
        self.append_event_at(conv, None, ev)
    }

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

    pub fn path(&self, conv: &str) -> Vec<TreeNode> {
        self.with(|c| Self::path_in(c, conv))
    }

    fn path_in(c: &Connection, conv: &str) -> Vec<TreeNode> {
        let Some(tip) = Self::cursor_in(c, conv) else {
            return Vec::new();
        };
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

    pub fn list_events(&self, conv: &str) -> Vec<EventWrapper> {
        self.path(conv).iter().map(|n| n.to_event()).collect()
    }

    pub fn list_events_folded(&self, conv: &str) -> Vec<EventWrapper> {
        fold_event_metas(self.list_events(conv))
    }

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

    pub fn build_context_window_items(&self, conv: &str) -> Vec<ContextItem> {
        crate::session_store::context_items_from_events(self.list_events_folded(conv))
    }

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

    pub fn sync_memory_tree(&self, conv: &str) -> usize {
        let now = self.memory_leaves(conv);
        let was = self.memory_tree_leaves(conv);
        let diverged = now
            .iter()
            .zip(was.iter())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| now.len().min(was.len()));

        self.with(|c| {
            let _ = c.execute(
                "DELETE FROM memory_nodes
                  WHERE conversation_id = ?1
                    AND ((idx + 1) * (1 << level)) > ?2",
                params![conv, diverged as i64],
            );
            if was.len() != now.len() || diverged < was.len() {
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
