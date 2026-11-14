//! One-shot import of `.angela/sessions/<id>.json{,l}` into
//! `.angela/conversations.db`.
//!
//! Runs on every server start and is **idempotent**: a session whose id is
//! already a conversation is skipped, so the cost after the first start is
//! one `SELECT` per file. It is also **non-destructive** — the legacy files
//! are read and left exactly where they are. That is the rollback plan: if
//! the SQLite path turns out to be wrong about something, the old files are
//! still the unmodified truth, and deleting `conversations.db` re-imports
//! them from scratch.
//!
//! The parent chain it builds is the one that makes the switch invisible:
//! each node's parent is the line before it, the first line is the root, and
//! the cursor ends on the tail. A linear log becomes a linear branch, so
//! `ConvStore::path` returns the same events in the same order that reading
//! the file did — which is what [`verify`] checks, per session, by comparing
//! the rendered transcripts field by field.

use serde_json::{json, Map, Value};

use crate::conv_store::ConvStore;
use crate::session_store::{js_truthy, NewEvent, SessionStore};

/// What one import run did.
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

/// Import every legacy session under `<root>/.angela/sessions` that the
/// database does not already have.
pub fn run(store: &ConvStore, root: &std::path::Path) -> Report {
    let mut report = Report::default();
    let dir = root.join(".angela").join("sessions");
    if !dir.exists() {
        return report;
    }
    let Ok(legacy) = SessionStore::new(root) else {
        return report;
    };
    // Oldest first, so `ordinal`-less sessions are numbered in the order they
    // were actually created rather than in directory order.
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
            // `ordinal` is copied, not reassigned: the tab strip's numbers are
            // what the user has been looking at, and renumbering on import
            // would silently relabel every open tab.
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
                    // The JS wrote `busy` where the Rust path writes `running`;
                    // normalise so the UI has one vocabulary. Anything still
                    // mid-turn when the process died is not running now.
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
            // Two participants: the human, and the agent the header names.
            // Nothing is rewritten — `role: user|assistant` keeps working,
            // it just now also points at a seat.
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

    // The log, line by line, each node's parent the one before it.
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
        // `append_event` parents on the cursor, which the previous insert
        // advanced — so the chain forms without tracking the id here.
        if store.append_event(id, ev).is_ok() {
            count += 1;
        }
    }

    // Fold the `context_meta` patches into their targets' columns. The rows
    // themselves are kept as nodes so the imported log stays replayable, but
    // nothing writes new ones: visibility is an UPDATE now.
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

/// Replay the append-only visibility/goto patches as column writes.
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
        // A patch carrying nothing but pointers was skipped by the old
        // reader, not applied as an empty change.
        if rest.is_empty() {
            continue;
        }
        for target in ids {
            if let Some(v) = rest.get("visible") {
                // `Boolean(patch.visible)` — JS truthiness, not a cast.
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

/// The no-behaviour-change gate: for every imported conversation, the
/// transcript the tree renders must equal the one the file rendered.
///
/// Compares the fields the UI actually draws (`kind`, `text`, `visible`) in
/// order, which is a stronger check than counting rows: an off-by-one in the
/// parent chain would keep the count and scramble the order.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_store::NewSession;

    /// Build a legacy session on disk, then import it.
    fn legacy_project() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let legacy = SessionStore::new(dir.path()).unwrap();
        let s = legacy
            .create(NewSession {
                agent: Some("default".into()),
                model: Some("llama-server:default".into()),
                title: Some("ported".into()),
                ..Default::default()
            })
            .unwrap();
        let id = s["id"].as_str().unwrap().to_string();
        legacy
            .append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"hi"})))
            .unwrap();
        legacy
            .append_event(&id, NewEvent::new("reasoning", json!({"text":"thinking"})))
            .unwrap();
        legacy
            .append_event(
                &id,
                NewEvent::new(
                    "gen_info",
                    json!({"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}),
                ),
            )
            .unwrap();
        legacy
            .append_event(
                &id,
                NewEvent::new("context_window", json!({"role":"assistant","content":"hello"})),
            )
            .unwrap();
        legacy.record_turn_complete(&id, Some(99.0));
        (dir, id)
    }

    #[test]
    fn a_legacy_session_imports_as_one_linear_branch() {
        let (dir, id) = legacy_project();
        let store = ConvStore::memory(dir.path()).unwrap();
        let rep = run(&store, dir.path());
        assert_eq!(rep.imported, 1);
        assert!(rep.failed.is_empty(), "{:?}", rep.failed);

        let header = store.load(&id).unwrap();
        assert_eq!(header["title"], "ported");
        assert_eq!(header["turns"], 1);
        assert_eq!(header["legacyFormat"], "jsonl");
        assert_eq!(header["eventCount"].as_u64().unwrap(), rep.nodes as u64);

        // One branch: every node's parent is the one before it, no roots but
        // the first.
        let nodes = store.path(&id);
        assert_eq!(nodes.len(), rep.nodes);
        assert!(nodes[0].parent_id.is_none());
        for w in nodes.windows(2) {
            assert_eq!(w[1].parent_id.as_deref(), Some(w[0].id.as_str()));
        }
        // Node ids are the legacy event ids, so any id a client cached still
        // resolves.
        let legacy = SessionStore::new(dir.path()).unwrap();
        let want: Vec<String> = legacy.list_events(&id).iter().map(|e| e.event_id.clone()).collect();
        let got: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
        assert_eq!(want, got);
    }

    #[test]
    fn the_rendered_transcript_is_identical_to_the_files() {
        let (dir, _) = legacy_project();
        let store = ConvStore::memory(dir.path()).unwrap();
        run(&store, dir.path());
        // The no-behaviour-change gate.
        assert_eq!(verify(&store, dir.path()).unwrap(), 1);
    }

    #[test]
    fn importing_twice_changes_nothing() {
        let (dir, id) = legacy_project();
        let store = ConvStore::memory(dir.path()).unwrap();
        let first = run(&store, dir.path());
        let count = store.load(&id).unwrap()["eventCount"].clone();
        let second = run(&store, dir.path());
        assert_eq!(second.imported, 0);
        assert_eq!(second.skipped, 1);
        assert_eq!(second.nodes, 0);
        // No duplicated nodes on the second pass.
        assert_eq!(store.load(&id).unwrap()["eventCount"], count);
        assert_eq!(first.imported, 1);
    }

    #[test]
    fn the_legacy_files_are_left_alone() {
        let (dir, id) = legacy_project();
        let legacy = SessionStore::new(dir.path()).unwrap();
        let before = std::fs::read_to_string(legacy.log_path(&id)).unwrap();
        let store = ConvStore::memory(dir.path()).unwrap();
        run(&store, dir.path());
        let after = std::fs::read_to_string(legacy.log_path(&id)).unwrap();
        // The rollback plan depends on this byte-for-byte.
        assert_eq!(before, after);
        assert!(legacy.state_path(&id).exists());
    }

    #[test]
    fn a_context_meta_patch_becomes_a_column_write() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = SessionStore::new(dir.path()).unwrap();
        let s = legacy.create(NewSession::default()).unwrap();
        let id = s["id"].as_str().unwrap().to_string();
        let target = legacy
            .append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"secret"})))
            .unwrap();
        legacy
            .append_event(
                &id,
                NewEvent::new("context_meta", json!({ "event_id": target.event_id, "visible": false })),
            )
            .unwrap();

        let store = ConvStore::memory(dir.path()).unwrap();
        run(&store, dir.path());
        let node = store.node(&target.event_id).unwrap();
        assert!(!node.visible, "the patch must have been folded into the column");
        // The patch row is kept, so the imported log is still replayable.
        assert!(store
            .all_nodes(&id)
            .iter()
            .any(|n| n.kind == "context_meta"));
    }

    #[test]
    fn the_ordinal_a_user_sees_survives_the_import() {
        let (dir, id) = legacy_project();
        let legacy = SessionStore::new(dir.path()).unwrap();
        let want = legacy.load(&id).unwrap()["ordinal"].as_i64().unwrap();
        let store = ConvStore::memory(dir.path()).unwrap();
        run(&store, dir.path());
        assert_eq!(store.load(&id).unwrap()["ordinal"].as_i64().unwrap(), want);
    }

    #[test]
    fn a_session_killed_mid_turn_comes_back_ready_not_running() {
        let (dir, id) = legacy_project();
        let legacy = SessionStore::new(dir.path()).unwrap();
        legacy.set_status(&id, "running");
        let store = ConvStore::memory(dir.path()).unwrap();
        run(&store, dir.path());
        // Nothing is running after a restart; leaving it `running` would show
        // a spinner no process is behind.
        assert_eq!(store.load(&id).unwrap()["status"], "ready");
    }
}
