use serde_json::{json, Map, Value};

use crate::conv_store::ConvStore;
use crate::session_store::{js_truthy, NewEvent, SessionStore};

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Report {
    pub imported: usize,
    pub skipped: usize,
    pub nodes: usize,
    pub failed: Vec<String>,
}

impl Report {
    pub fn to_json(&self) -> Value {
        json!({
            "imported": self.imported,
            "skipped": self.skipped,
            "nodes": self.nodes,
            "failed": self.failed,
        })
    }
}

pub fn run(store: &ConvStore, root: &std::path::Path) -> Report {
    let mut report = Report::default();
    let dir = root.join(".angela").join("sessions");
    if !dir.exists() {
        return report;
    }
    let Ok(legacy) = SessionStore::new(root) else {
        return report;
    };
    let mut ids = legacy.list();
    ids.reverse();

    for id in ids {
        let already = store.with(|c| {
            c.query_row(
                "SELECT COUNT(*) FROM conversation WHERE id = ?1",
                rusqlite::params![id],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0)
        });
        if already > 0 {
            report.skipped += 1;
            continue;
        }
        match import_one(store, &legacy, &id) {
            Ok(n) => {
                report.imported += 1;
                report.nodes += n;
            }
            Err(e) => report.failed.push(format!("{id}: {e}")),
        }
    }
    report
}

fn import_one(store: &ConvStore, legacy: &SessionStore, id: &str) -> Result<usize, String> {
    let header = legacy.load(id).ok_or_else(|| "no header".to_string())?;
    let g = |k: &str| header.get(k).cloned().unwrap_or(Value::Null);
    let gs = |k: &str| header.get(k).and_then(|v| v.as_str()).map(String::from);

    let ordinal = header
        .get("ordinal")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let created = gs("createdAt").unwrap_or_else(crate::session_store::now_iso);
    let updated = gs("updatedAt").unwrap_or_else(|| created.clone());

    store
        .with(|c| {
            let ord = if ordinal > 0 {
                ordinal
            } else {
                c.query_row(
                    "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM conversation",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .unwrap_or(1)
            };
            c.execute(
                "INSERT INTO conversation
                   (id, ordinal, api_version, kind, title, agent, agent_file, model, status,
                    policy_mode, approve_mode, allowlist_mode, allowlist, allowlist_source,
                    context_window, mcp, project_root, config, overrides, turns,
                    legacy_format, created_at, updated_at)
                 VALUES (?1, ?2, 'conv/v1', 'conversation', ?3, ?4, ?5, ?6, ?7,
                         ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18,
                         'jsonl', ?19, ?20)",
                rusqlite::params![
                    id,
                    ord,
                    gs("title"),
                    gs("agent"),
                    gs("agentFile"),
                    gs("model"),
                    match gs("status").as_deref() {
                        Some("running") | Some("busy") => "ready",
                        Some(s) => s,
                        None => "ready",
                    },
                    gs("policyMode").unwrap_or_else(|| "ask".into()),
                    opt_json(&g("approveMode")),
                    gs("allowlistMode"),
                    gs("allowlist"),
                    gs("allowlistSource"),
                    header.get("contextWindow").and_then(|v| v.as_i64()),
                    opt_json(&g("mcp")),
                    gs("projectRoot")
                        .unwrap_or_else(|| store.project_root.to_string_lossy().into_owned()),
                    opt_json(&g("config")),
                    opt_json(&g("overrides")),
                    header.get("turns").and_then(|v| v.as_i64()).unwrap_or(0),
                    created,
                    updated,
                ],
            )?;
            c.execute(
                "INSERT INTO cursor (conversation_id, participant_id, event_id, updated_at)
                 VALUES (?1, 0, NULL, ?2)",
                rusqlite::params![id, updated],
            )?;
            c.execute(
                "INSERT OR IGNORE INTO participants
                   (conversation_id, kind, handle, display_name, seat, active)
                 VALUES (?1, 'user', 'user', 'User', 0, 1)",
                rusqlite::params![id],
            )?;
            let agent = gs("agent").unwrap_or_else(|| "assistant".into());
            c.execute(
                "INSERT OR IGNORE INTO participants
                   (conversation_id, kind, handle, display_name, agent_name, model,
                    system_prompt, seat, active, role)
                 VALUES (?1, 'agent', ?2, ?3, ?4, ?5, '', 1, 1, 'member')",
                rusqlite::params![
                    id,
                    crate::conv_store::handle_from(&agent),
                    agent,
                    gs("agent"),
                    gs("model")
                ],
            )?;
            Ok::<(), rusqlite::Error>(())
        })
        .map_err(|e| e.to_string())?;

    let events = legacy.list_events(id);
    let mut count = 0usize;
    for wrap in &events {
        let ev = NewEvent {
            event_type: wrap.event_type.clone(),
            payload: wrap.payload.clone(),
            duration_ms: wrap.duration_ms,
            ts: Some(wrap.ts.clone()),
            meta: Some(wrap.meta.clone()),
            event_id: Some(wrap.event_id.clone()),
            speaker: None,
        };
        if store.append_event(id, ev).is_ok() {
            count += 1;
        }
    }

    apply_context_meta(store, &events);
    store.sync_memory_tree(id);
    Ok(count)
}

fn opt_json(v: &Value) -> Option<String> {
    if v.is_null() {
        None
    } else {
        Some(v.to_string())
    }
}

fn apply_context_meta(store: &ConvStore, events: &[crate::session_store::EventWrapper]) {
    for wrap in events {
        if wrap.event_type != "context_meta" {
            continue;
        }
        let Some(p) = wrap.payload.as_object() else { continue };
        let ids: Vec<String> = match p.get("event_ids").and_then(|v| v.as_array()) {
            Some(a) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
            None => p
                .get("event_id")
                .and_then(|v| v.as_str())
                .map(|s| vec![s.to_string()])
                .unwrap_or_default(),
        };
        let mut rest: Map<String, Value> = p.clone();
        rest.remove("event_id");
        rest.remove("event_ids");
        if rest.is_empty() {
            continue;
        }
        for target in ids {
            if let Some(v) = rest.get("visible") {
                store.set_visibility(&target, js_truthy(v));
            }
            if let Some(g) = rest.get("goto").and_then(|g| g.as_object()) {
                store.set_goto(
                    &target,
                    g.get("enabled").map(js_truthy).unwrap_or(false),
                    g.get("label").and_then(|v| v.as_str()),
                );
            }
        }
    }
}

pub fn verify(store: &ConvStore, root: &std::path::Path) -> Result<usize, String> {
    let legacy = SessionStore::new(root).map_err(|e| e.to_string())?;
    let mut checked = 0;
    for id in legacy.list() {
        if store.load(&id).is_none() {
            continue;
        }
        let want = crate::serve::reconstruct_messages_from_log(&legacy, &id);
        let got = crate::serve::reconstruct_messages(store, &id);
        if want.len() != got.len() {
            return Err(format!(
                "{id}: file renders {} rows, tree renders {}",
                want.len(),
                got.len()
            ));
        }
        for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            for k in ["kind", "text", "visible"] {
                if w.get(k) != g.get(k) {
                    return Err(format!(
                        "{id}: row {i} field {k}: file {:?} != tree {:?}",
                        w.get(k),
                        g.get(k)
                    ));
                }
            }
        }
        checked += 1;
    }
    Ok(checked)
}
