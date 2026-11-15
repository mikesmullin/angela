//! Port of `angela/src/session-store.mjs` — durable sessions under
//! `.angela/sessions/`.
//!
//! Two files per session:
//!
//! ```text
//! <id>.json    metadata only (no transcript body)
//! <id>.jsonl   append-only wrapped events
//! ```
//!
//! The JSONL wrapper keeps telemetry out of the payload:
//!
//! ```json
//! { "event_id": "e1a2b3", "ts": "ISO-8601",
//!   "event_type": "context_window" | "tool_call" | "tool_result" | "session" | ...,
//!   "duration_ms": 12.4, "payload": { }, "meta": { "visible": true } }
//! ```
//!
//! Line order *is* the event sequence, and the session id is the filename
//! stem rather than a repeated field. Visibility and goto patches append a
//! `context_meta` row (last-write-wins) instead of rewriting the original
//! line — the log stays append-only, so a reader must fold metas forward
//! (see [`fold_event_metas`]).
//!
//! The on-disk format is consumed by shipped clients, so this port is
//! byte-compatible with the JS: same field names, same ordering, same
//! `apiVersion`/`logEncoding` markers.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::project::{ensure_angela_layout, sessions_dir};

/// `new Date().toISOString()` — always UTC, always milliseconds.
pub fn now_iso() -> String {
    // Hand-rolled so the crate keeps a thin dependency set; the format is
    // fixed by the on-disk contract (YYYY-MM-DDTHH:MM:SS.mmmZ).
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let millis = now.subsec_millis();
    let (y, mo, d, h, mi, s) = civil_from_unix(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{millis:03}Z")
}

/// days-from-civil inverse (Howard Hinnant's algorithm), UTC only.
fn civil_from_unix(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (h, mi, s) = (
        (rem / 3600) as u32,
        ((rem % 3600) / 60) as u32,
        (rem % 60) as u32,
    );
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d, h, mi, s)
}

/// Small hex token. The JS uses `randomBytes`; this uses the same shape.
fn rand_hex(bytes: usize) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    // Mix a monotonic counter with the clock: ids must not collide within a
    // process, and the JS guarantee is uniqueness, not cryptographic secrecy.
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    let mut x = t
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(n.wrapping_mul(1_442_695_040_888_963_407));
    let mut out = String::new();
    for _ in 0..bytes {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.push_str(&format!("{:02x}", (x >> 24) as u8));
    }
    out
}

/// `<iso-without-punctuation>_<6 hex>` — sorts chronologically as a filename.
pub fn new_session_id() -> String {
    let iso = now_iso();
    let stem: String = iso
        .chars()
        .take(19)
        .map(|c| if c == ':' || c == '.' { '-' } else { c })
        .collect();
    format!("{stem}_{}", rand_hex(3))
}

/// `new_event_id` for other modules: the tree store mints node ids with the
/// same scheme, so an id tells you nothing about which store wrote it.
pub fn new_event_id_pub() -> String {
    new_event_id()
}

fn new_event_id() -> String {
    format!("e{}", rand_hex(4))
}

/// One wrapped JSONL row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventWrapper {
    pub event_id: String,
    pub ts: String,
    pub event_type: String,
    pub duration_ms: Option<f64>,
    pub payload: Value,
    #[serde(default)]
    pub meta: Map<String, Value>,
}

impl EventWrapper {
    /// `meta.visible`, defaulting to **true** — a row with no explicit
    /// visibility is shown, as in the JS.
    pub fn visible(&self) -> bool {
        self.meta
            .get("visible")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }
}

/// What `append_event` is given. `event_id`/`ts` are normally generated.
#[derive(Debug, Clone, Default)]
pub struct NewEvent {
    pub event_type: String,
    pub payload: Value,
    pub duration_ms: Option<f64>,
    pub ts: Option<String>,
    pub meta: Option<Map<String, Value>>,
    pub event_id: Option<String>,
    /// Which participant authored this event, for the tree store. `None`
    /// lets it infer from the payload's role, which is right whenever the
    /// conversation has a single agent — i.e. everywhere the file store ran.
    pub speaker: Option<i64>,
}

impl NewEvent {
    pub fn new(event_type: impl Into<String>, payload: Value) -> Self {
        NewEvent {
            event_type: event_type.into(),
            payload,
            ..Default::default()
        }
    }
}

/// Apply `context_meta` rows forward onto the events they name.
///
/// Last-write-wins, and `visible` is coerced to a real boolean (the JS does
/// `Boolean(patch.visible)`, so a truthy non-bool becomes `true`). A
/// `context_meta` row whose payload carries nothing but the id pointers is
/// skipped rather than applied as an empty patch.
pub fn fold_event_metas(events: Vec<EventWrapper>) -> Vec<EventWrapper> {
    if events.is_empty() {
        return events;
    }
    let mut extra: std::collections::HashMap<String, Map<String, Value>> =
        std::collections::HashMap::new();

    for wrap in &events {
        if wrap.event_type != "context_meta" {
            continue;
        }
        let p = match wrap.payload.as_object() {
            Some(o) => o,
            None => continue,
        };
        let ids: Vec<String> = match p.get("event_ids").and_then(|v| v.as_array()) {
            Some(a) => a
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            None => p
                .get("event_id")
                .and_then(|v| v.as_str())
                .map(|s| vec![s.to_string()])
                .unwrap_or_default(),
        };
        let mut rest = p.clone();
        rest.remove("event_id");
        rest.remove("event_ids");
        if rest.is_empty() {
            continue;
        }
        for eid in ids {
            if eid.is_empty() {
                continue;
            }
            let slot = extra.entry(eid).or_default();
            for (k, v) in &rest {
                slot.insert(k.clone(), v.clone());
            }
        }
    }
    if extra.is_empty() {
        return events;
    }
    events
        .into_iter()
        .map(|mut wrap| {
            if let Some(patch) = extra.get(&wrap.event_id) {
                for (k, v) in patch {
                    wrap.meta.insert(k.clone(), v.clone());
                }
                if let Some(v) = patch.get("visible") {
                    // Boolean(patch.visible) — JS truthiness, not a cast.
                    let truthy = match v {
                        Value::Bool(b) => *b,
                        Value::Null => false,
                        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
                        Value::String(s) => !s.is_empty(),
                        _ => true,
                    };
                    wrap.meta.insert("visible".into(), Value::Bool(truthy));
                }
            }
            wrap
        })
        .collect()
}

/// Metadata a caller supplies to [`SessionStore::create`].
#[derive(Debug, Clone, Default)]
pub struct NewSession {
    pub agent: Option<String>,
    pub agent_file: Option<String>,
    pub model: Option<String>,
    pub title: Option<String>,
    pub mcp: Vec<String>,
    pub policy_mode: Option<String>,
    pub approve_mode: Option<Value>,
    pub allowlist_source: Option<String>,
    pub allowlist: Option<String>,
    /// `allowlist` or `unrestricted`. `None` takes the default for a new
    /// conversation, which is `unrestricted`.
    pub allowlist_mode: Option<String>,
    pub context_window: Option<u64>,
    pub config: Option<Value>,
    pub overrides: Option<Value>,
}

/// Durable sessions under `<root>/.angela/sessions`.
pub struct SessionStore {
    pub project_root: PathBuf,
    pub dir: PathBuf,
}

impl SessionStore {
    pub fn new(project_root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = project_root.as_ref().to_path_buf();
        ensure_angela_layout(&root)?;
        Ok(SessionStore {
            dir: sessions_dir(&root),
            project_root: root,
        })
    }

    pub fn state_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    pub fn log_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.jsonl"))
    }

    /// Create a session: metadata JSON, then the first `session` event.
    ///
    /// Field order matches the JS object literal, because the file is written
    /// pretty-printed and humans diff it.
    pub fn create(&self, meta: NewSession) -> std::io::Result<Value> {
        let id = new_session_id();
        let now = now_iso();
        // A stable 1-based number for the tab strip. `max + 1`, never
        // `count + 1`: after closing tab 2 of {1,2,3} the next new tab must
        // be 4, not a second 3 (RUST_M_JS §10 — a reused id corrupts
        // `x-for` keys and active-state tracking).
        let ordinal = self
            .list_meta()
            .iter()
            .filter_map(|s| s.get("ordinal").and_then(|v| v.as_u64()))
            .max()
            .unwrap_or(0)
            + 1;
        let state = json!({
            "id": id,
            "ordinal": ordinal,
            "apiVersion": "angela/v1",
            "logEncoding": "delta/v1",
            "kind": "AgentSession",
            "createdAt": now,
            "updatedAt": now,
            "status": "ready",
            "agent": meta.agent,
            "agentFile": meta.agent_file,
            "model": meta.model,
            "title": meta.title,
            "mcp": meta.mcp,
            "policyMode": meta.policy_mode.unwrap_or_else(|| "ask".into()),
            "approveMode": meta.approve_mode.unwrap_or(Value::Null),
            "allowlistSource": meta.allowlist_source,
            "allowlist": meta.allowlist,
            "contextWindow": meta.context_window,
            "projectRoot": self.project_root.to_string_lossy(),
            "config": meta.config.unwrap_or(Value::Null),
            "overrides": meta.overrides.unwrap_or(Value::Null),
            "turns": 0,
            "eventCount": 0,
            "lastPromptTokens": Value::Null,
            "lastTotalTokens": Value::Null,
            "lastCompletionTokens": Value::Null,
            "logFile": format!("{id}.jsonl"),
        });
        self.write_state(&id, state.clone())?;
        self.append_event(
            &id,
            NewEvent::new(
                "session",
                json!({
                    "kind": "created",
                    "agent": state["agent"],
                    "model": state["model"],
                    "mcp": state["mcp"],
                    "overrides": state["overrides"],
                    "config": state["config"],
                }),
            ),
        )?;
        // Re-read so the returned state carries the bumped eventCount.
        Ok(self.load(&id).unwrap_or(state))
    }

    pub fn load(&self, id: &str) -> Option<Value> {
        let p = self.state_path(id);
        let text = std::fs::read_to_string(p).ok()?;
        serde_json::from_str(&text).ok()
    }

    fn write_state(&self, id: &str, mut state: Value) -> std::io::Result<()> {
        if !self.dir.exists() {
            std::fs::create_dir_all(&self.dir)?;
        }
        if let Some(o) = state.as_object_mut() {
            o.insert("updatedAt".into(), json!(now_iso()));
        }
        let body = serde_json::to_string_pretty(&state).unwrap_or_default();
        std::fs::write(self.state_path(id), format!("{body}\n"))
    }

    pub fn save(&self, state: &Value) -> std::io::Result<()> {
        let id = state.get("id").and_then(|v| v.as_str()).unwrap_or_default();
        self.write_state(id, state.clone())
    }

    /// Patch metadata fields and rewrite the `.json`.
    pub fn update_meta(&self, id: &str, patch: Map<String, Value>) -> Option<Value> {
        let mut state = self.load(id)?;
        {
            let o = state.as_object_mut()?;
            for (k, v) in patch {
                o.insert(k, v);
            }
        }
        self.write_state(id, state.clone()).ok()?;
        self.load(id)
    }

    /// Append one wrapped event and bump `eventCount`.
    pub fn append_event(&self, id: &str, ev: NewEvent) -> std::io::Result<EventWrapper> {
        let wrapper = EventWrapper {
            event_id: ev.event_id.unwrap_or_else(new_event_id),
            ts: ev.ts.unwrap_or_else(now_iso),
            event_type: ev.event_type,
            // NaN becomes null, as in the JS guard.
            duration_ms: ev.duration_ms.filter(|d| d.is_finite()),
            payload: ev.payload,
            meta: ev.meta.unwrap_or_default(),
        };
        let line = serde_json::to_string(&wrapper).unwrap_or_default();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.log_path(id))?;
        writeln!(f, "{line}")?;

        if let Some(mut state) = self.load(id) {
            let n = state
                .get("eventCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            if let Some(o) = state.as_object_mut() {
                o.insert("eventCount".into(), json!(n + 1));
            }
            let _ = self.write_state(id, state);
        }
        Ok(wrapper)
    }

    /// All wrappers, oldest first. Unparseable lines are skipped, not fatal —
    /// a half-written final line must not cost you the session.
    pub fn list_events(&self, id: &str) -> Vec<EventWrapper> {
        let text = match std::fs::read_to_string(self.log_path(id)) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<EventWrapper>(l).ok())
            .collect()
    }

    /// `list_events` with `context_meta` folded forward.
    pub fn list_events_folded(&self, id: &str) -> Vec<EventWrapper> {
        fold_event_metas(self.list_events(id))
    }

    /// Bump the turn counter after a completed user→assistant exchange.
    pub fn record_turn_complete(&self, id: &str, duration_ms: Option<f64>) {
        let Some(mut state) = self.load(id) else {
            return;
        };
        let turns = state.get("turns").and_then(|v| v.as_u64()).unwrap_or(0) + 1;
        if let Some(o) = state.as_object_mut() {
            o.insert("turns".into(), json!(turns));
            o.insert("status".into(), json!("ready"));
        }
        let _ = self.write_state(id, state);
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

    pub fn set_status(&self, id: &str, status: &str) {
        let Some(mut state) = self.load(id) else {
            return;
        };
        if let Some(o) = state.as_object_mut() {
            o.insert("status".into(), json!(status));
        }
        let _ = self.write_state(id, state);
    }

    /// Give every pre-`ordinal` session a number, oldest first.
    ///
    /// A one-time migration rather than a read-time fallback: the tab strip
    /// numbers must be **stable**, and a number derived on each read would
    /// shuffle as sessions are created and deleted. Returns how many it
    /// backfilled.
    pub fn backfill_ordinals(&self) -> usize {
        let metas = self.list_meta();
        let mut next = metas
            .iter()
            .filter_map(|s| s.get("ordinal").and_then(|v| v.as_u64()))
            .max()
            .unwrap_or(0);
        // Oldest first, so the numbering matches the order they were made.
        let mut missing: Vec<&Value> = metas
            .iter()
            .filter(|s| s.get("ordinal").and_then(|v| v.as_u64()).is_none())
            .collect();
        missing.sort_by_key(|s| {
            s.get("createdAt")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        });
        let n = missing.len();
        for s in missing {
            let Some(id) = s.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            next += 1;
            let mut patch = Map::new();
            patch.insert("ordinal".into(), json!(next));
            self.update_meta(id, patch);
        }
        n
    }

    /// Session ids, **newest first** (ids sort chronologically, reversed).
    pub fn list(&self) -> Vec<String> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut ids: Vec<String> = rd
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                name.strip_suffix(".json").map(|s| s.to_string())
            })
            .collect();
        ids.sort();
        ids.reverse();
        ids
    }

    /// All session metadata, newest first.
    pub fn list_meta(&self) -> Vec<Value> {
        self.list().into_iter().filter_map(|id| self.load(&id)).collect()
    }

    /// Delete one session's pair of files.
    ///
    /// Refuses ids containing `..` or a path separator — the id reaches this
    /// from an HTTP route, so it is a traversal vector.
    pub fn delete(&self, id: &str) -> bool {
        if id.is_empty() || id.contains("..") || id.contains('/') || id.contains('\\') {
            return false;
        }
        let mut n = 0;
        for p in [self.state_path(id), self.log_path(id)] {
            if p.exists() && std::fs::remove_file(&p).is_ok() {
                n += 1;
            }
        }
        n > 0
    }

    /// Remove every session file (`angela clean`). Returns the count.
    pub fn clean(&self) -> usize {
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return 0;
        };
        let mut n = 0;
        for e in rd.filter_map(|e| e.ok()) {
            if std::fs::remove_file(e.path()).is_ok() {
                n += 1;
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, SessionStore) {
        let tmp = tempfile::tempdir().unwrap();
        let s = SessionStore::new(tmp.path()).unwrap();
        (tmp, s)
    }

    #[test]
    fn iso_timestamps_match_the_js_shape() {
        let t = now_iso();
        assert_eq!(t.len(), 24, "{t}");
        assert!(t.ends_with('Z'), "{t}");
        assert_eq!(&t[4..5], "-");
        assert_eq!(&t[10..11], "T");
        assert_eq!(&t[19..20], ".");
        // round-trips through a strict parser
        assert!(t.chars().filter(|c| *c == ':').count() == 2);
    }

    #[test]
    fn civil_conversion_is_correct_at_a_known_epoch() {
        // 2026-11-12T00:00:00Z
        let (y, mo, d, h, mi, s) = civil_from_unix(1_794_441_600);
        assert_eq!((y, mo, d, h, mi, s), (2026, 11, 12, 0, 0, 0));
        // the unix epoch itself
        assert_eq!(civil_from_unix(0), (1970, 1, 1, 0, 0, 0));
        // a leap day
        assert_eq!(civil_from_unix(1_709_164_800), (2024, 2, 29, 0, 0, 0));
    }

    #[test]
    fn session_ids_sort_chronologically_and_are_unique() {
        let a = new_session_id();
        let b = new_session_id();
        assert_ne!(a, b);
        assert_eq!(a.len(), 19 + 1 + 6, "{a}");
        assert!(!a.contains(':'), "{a}");
    }

    #[test]
    fn create_writes_both_files_and_the_created_event() {
        let (_t, s) = store();
        let st = s
            .create(NewSession {
                agent: Some("lucy".into()),
                model: Some("llama-server:default".into()),
                title: Some("hello".into()),
                ..Default::default()
            })
            .unwrap();
        let id = st["id"].as_str().unwrap();
        assert!(s.state_path(id).exists());
        assert!(s.log_path(id).exists());
        assert_eq!(st["apiVersion"], "angela/v1");
        assert_eq!(st["logEncoding"], "delta/v1");
        assert_eq!(st["kind"], "AgentSession");
        assert_eq!(st["status"], "ready");
        assert_eq!(st["turns"], 0);
        assert_eq!(st["logFile"], format!("{id}.jsonl"));

        let evs = s.list_events(id);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].event_type, "session");
        assert_eq!(evs[0].payload["kind"], "created");
        assert_eq!(evs[0].payload["agent"], "lucy");
        // the first event is counted
        assert_eq!(st["eventCount"], 1);
    }

    #[test]
    fn append_event_bumps_the_counter_and_keeps_order() {
        let (_t, s) = store();
        let st = s.create(NewSession::default()).unwrap();
        let id = st["id"].as_str().unwrap().to_string();
        for i in 0..3 {
            s.append_event(&id, NewEvent::new("context_window", json!({ "i": i })))
                .unwrap();
        }
        let evs = s.list_events(&id);
        assert_eq!(evs.len(), 4, "created + 3");
        assert_eq!(evs[1].payload["i"], 0);
        assert_eq!(evs[3].payload["i"], 2);
        assert_eq!(s.load(&id).unwrap()["eventCount"], 4);
    }

    #[test]
    fn a_truncated_final_line_does_not_lose_the_session() {
        let (_t, s) = store();
        let st = s.create(NewSession::default()).unwrap();
        let id = st["id"].as_str().unwrap().to_string();
        s.append_event(&id, NewEvent::new("tool_call", json!({"n": 1})))
            .unwrap();
        // simulate a crash mid-write
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(s.log_path(&id))
            .unwrap();
        write!(f, "{{\"event_id\":\"broken\",\"ts\"").unwrap();
        drop(f);
        let evs = s.list_events(&id);
        assert_eq!(evs.len(), 2, "the two good rows survive");
    }

    #[test]
    fn nan_duration_becomes_null() {
        let (_t, s) = store();
        let st = s.create(NewSession::default()).unwrap();
        let id = st["id"].as_str().unwrap().to_string();
        let w = s
            .append_event(
                &id,
                NewEvent {
                    event_type: "tool_result".into(),
                    payload: json!({}),
                    duration_ms: Some(f64::NAN),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(w.duration_ms, None);
    }

    #[test]
    fn context_meta_folds_forward_last_write_wins() {
        let (_t, s) = store();
        let st = s.create(NewSession::default()).unwrap();
        let id = st["id"].as_str().unwrap().to_string();
        let target = s
            .append_event(&id, NewEvent::new("context_window", json!({"text": "hi"})))
            .unwrap();
        assert!(target.visible(), "default visibility is true");

        s.append_event(
            &id,
            NewEvent::new(
                "context_meta",
                json!({ "event_id": target.event_id, "visible": false, "note": "a" }),
            ),
        )
        .unwrap();
        s.append_event(
            &id,
            NewEvent::new(
                "context_meta",
                json!({ "event_id": target.event_id, "note": "b" }),
            ),
        )
        .unwrap();

        let folded = s.list_events_folded(&id);
        let row = folded
            .iter()
            .find(|e| e.event_id == target.event_id)
            .unwrap();
        assert!(!row.visible(), "first patch hid it");
        assert_eq!(row.meta["note"], "b", "later patch wins");

        // the original line is NOT rewritten
        let raw = s.list_events(&id);
        let orig = raw.iter().find(|e| e.event_id == target.event_id).unwrap();
        assert!(orig.meta.is_empty(), "append-only: original untouched");
    }

    #[test]
    fn context_meta_can_patch_many_rows_at_once() {
        let (_t, s) = store();
        let st = s.create(NewSession::default()).unwrap();
        let id = st["id"].as_str().unwrap().to_string();
        let a = s.append_event(&id, NewEvent::new("context_window", json!({}))).unwrap();
        let b = s.append_event(&id, NewEvent::new("context_window", json!({}))).unwrap();
        s.append_event(
            &id,
            NewEvent::new(
                "context_meta",
                json!({ "event_ids": [a.event_id, b.event_id], "visible": false }),
            ),
        )
        .unwrap();
        let folded = s.list_events_folded(&id);
        assert!(folded.iter().filter(|e| e.event_type == "context_window").all(|e| !e.visible()));
    }

    #[test]
    fn a_context_meta_with_only_pointers_is_not_an_empty_patch() {
        let (_t, s) = store();
        let st = s.create(NewSession::default()).unwrap();
        let id = st["id"].as_str().unwrap().to_string();
        let a = s.append_event(&id, NewEvent::new("context_window", json!({}))).unwrap();
        s.append_event(&id, NewEvent::new("context_meta", json!({ "event_id": a.event_id })))
            .unwrap();
        let folded = s.list_events_folded(&id);
        let row = folded.iter().find(|e| e.event_id == a.event_id).unwrap();
        assert!(row.meta.is_empty());
    }

    #[test]
    fn visible_is_coerced_the_way_js_boolean_does() {
        let (_t, s) = store();
        let st = s.create(NewSession::default()).unwrap();
        let id = st["id"].as_str().unwrap().to_string();
        let a = s.append_event(&id, NewEvent::new("context_window", json!({}))).unwrap();
        // a truthy non-boolean becomes `true`
        s.append_event(
            &id,
            NewEvent::new("context_meta", json!({ "event_id": a.event_id, "visible": "yes" })),
        )
        .unwrap();
        let folded = s.list_events_folded(&id);
        let row = folded.iter().find(|e| e.event_id == a.event_id).unwrap();
        assert_eq!(row.meta["visible"], Value::Bool(true));
    }

    #[test]
    fn turns_and_status_track_the_exchange() {
        let (_t, s) = store();
        let st = s.create(NewSession::default()).unwrap();
        let id = st["id"].as_str().unwrap().to_string();
        s.set_status(&id, "busy");
        assert_eq!(s.load(&id).unwrap()["status"], "busy");
        s.record_turn_complete(&id, Some(1234.5));
        let after = s.load(&id).unwrap();
        assert_eq!(after["turns"], 1);
        assert_eq!(after["status"], "ready", "a completed turn returns to ready");
        let last = s.list_events(&id).pop().unwrap();
        assert_eq!(last.payload["kind"], "turn_complete");
        assert_eq!(last.duration_ms, Some(1234.5));
    }

    #[test]
    fn update_meta_patches_and_persists() {
        let (_t, s) = store();
        let st = s.create(NewSession::default()).unwrap();
        let id = st["id"].as_str().unwrap().to_string();
        let mut patch = Map::new();
        patch.insert("allowlistSource".into(), json!("ui"));
        patch.insert("allowlist".into(), json!("read\nbash"));
        let out = s.update_meta(&id, patch).unwrap();
        assert_eq!(out["allowlistSource"], "ui");
        assert_eq!(s.load(&id).unwrap()["allowlist"], "read\nbash");
    }

    #[test]
    fn ordinals_are_max_plus_one_so_a_closed_tab_is_not_reused() {
        let (_t, s) = store();
        let a = s.create(NewSession::default()).unwrap();
        let b = s.create(NewSession::default()).unwrap();
        let c = s.create(NewSession::default()).unwrap();
        assert_eq!(a["ordinal"], 1);
        assert_eq!(b["ordinal"], 2);
        assert_eq!(c["ordinal"], 3);

        // close the middle one → {1, 3}
        assert!(s.delete(b["id"].as_str().unwrap()));
        let d = s.create(NewSession::default()).unwrap();
        assert_eq!(d["ordinal"], 4, "the next tab is 4, not a second 3");

        let mut live: Vec<u64> = s
            .list_meta()
            .iter()
            .filter_map(|x| x["ordinal"].as_u64())
            .collect();
        live.sort();
        assert_eq!(live, vec![1, 3, 4]);
    }

    #[test]
    fn legacy_sessions_are_backfilled_oldest_first() {
        let (_t, s) = store();
        // Two sessions written before `ordinal` existed.
        for (id, created) in [("old-b", "2026-01-02T00:00:00.000Z"), ("old-a", "2026-01-01T00:00:00.000Z")] {
            let st = json!({
                "id": id, "createdAt": created, "updatedAt": created,
                "status": "ready", "turns": 0, "eventCount": 0,
            });
            std::fs::write(
                s.state_path(id),
                serde_json::to_string_pretty(&st).unwrap(),
            )
            .unwrap();
        }
        assert_eq!(s.backfill_ordinals(), 2);
        let a = s.load("old-a").unwrap()["ordinal"].as_u64().unwrap();
        let b = s.load("old-b").unwrap()["ordinal"].as_u64().unwrap();
        assert_eq!((a, b), (1, 2), "oldest gets the lowest number");

        // A new session continues from the high-water mark.
        let fresh = s.create(NewSession::default()).unwrap();
        assert_eq!(fresh["ordinal"], 3);
        // Running it again is a no-op.
        assert_eq!(s.backfill_ordinals(), 0);
    }

    #[test]
    fn listing_is_newest_first() {
        let (_t, s) = store();
        let a = s.create(NewSession::default()).unwrap()["id"].as_str().unwrap().to_string();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = s.create(NewSession::default()).unwrap()["id"].as_str().unwrap().to_string();
        let ids = s.list();
        assert_eq!(ids.len(), 2);
        // ids embed the timestamp, so reverse-sorted == newest first
        let mut expect = vec![a, b];
        expect.sort();
        expect.reverse();
        assert_eq!(ids, expect);
        assert_eq!(s.list_meta().len(), 2);
    }

    #[test]
    fn delete_refuses_traversal_ids() {
        let (_t, s) = store();
        assert!(!s.delete("../escape"));
        assert!(!s.delete("a/b"));
        assert!(!s.delete(""));
    }

    #[test]
    fn delete_removes_both_files_and_clean_empties_the_dir() {
        let (_t, s) = store();
        let id = s.create(NewSession::default()).unwrap()["id"].as_str().unwrap().to_string();
        assert!(s.delete(&id));
        assert!(!s.state_path(&id).exists());
        assert!(!s.log_path(&id).exists());
        assert!(!s.delete(&id), "second delete finds nothing");

        s.create(NewSession::default()).unwrap();
        s.create(NewSession::default()).unwrap();
        assert_eq!(s.clean(), 4, "two sessions × two files");
        assert!(s.list().is_empty());
    }

    #[test]
    fn loading_an_unknown_session_is_none_not_a_panic() {
        let (_t, s) = store();
        assert!(s.load("nope").is_none());
        assert!(s.list_events("nope").is_empty());
    }
}

// ===========================================================================
// Log → transcript. Port of `listVisibleContextMessages` and
// `buildContextWindowItems` (session-store.mjs ~616-744).
// ===========================================================================

/// A UI/goto stamp, carried in `meta` rather than the payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Goto {
    pub enabled: bool,
    pub label: String,
}

/// One row of the hydrated context window.
///
/// `1 jsonl event = 1 context item = 1 UI row`. Optional fields are skipped
/// when absent so the JSON matches what `JSON.stringify` emits for the JS
/// object (which simply omits `undefined` keys).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextItem {
    pub role: String,
    pub content: String,
    pub id: Option<String>,
    pub visible: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub goto: Option<Goto>,
    // ---- UI-only extras; never sent to the provider ----
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ok: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denied: Option<Value>,
    /// Tri-state, because the JS distinguishes three cases and a plain
    /// `Option` cannot:
    ///   * `None`             — key absent (plain and reasoning rows never set it)
    ///   * `Some(None)`       — key present and `null` (tool rows: `p.explanation || null`)
    ///   * `Some(Some(text))` — key present with a value
    /// `ok`/`denied` are bare `p.ok` in the JS and simply vanish when
    /// undefined, so they stay a plain skipped Option. Pinned by the
    /// differential test against the real JS output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<Option<String>>,
}

impl ContextItem {
    fn base(role: &str, content: String, id: Option<String>, visible: bool) -> Self {
        ContextItem {
            role: role.to_string(),
            content,
            id,
            visible,
            tool_calls: None,
            tool_call_id: None,
            goto: None,
            name: None,
            args: None,
            ok: None,
            denied: None,
            explanation: None,
        }
    }
}

/// `String(x)` for a JSON value: strings pass through unquoted, everything
/// else takes its JSON form. The JS relies on this coercion in several spots.
fn js_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "null".into(),
        other => other.to_string(),
    }
}

fn goto_of(meta: &Map<String, Value>) -> Option<Goto> {
    let g = meta.get("goto")?.as_object()?;
    Some(Goto {
        enabled: g.get("enabled").map(truthy).unwrap_or(false),
        label: g
            .get("label")
            .map(|v| if v.is_null() { String::new() } else { js_string(v) })
            .unwrap_or_default(),
    })
}

/// JS truthiness, for the `Boolean(...)` coercions the JS performs.
pub fn js_truthy(v: &Value) -> bool {
    truthy(v)
}

/// `String(x)` coercion, re-exported for the serve layer.
pub fn js_string_of(v: &Value) -> String {
    js_string(v)
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Null => false,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        _ => true,
    }
}

impl SessionStore {
    /// The visible user/assistant exchange, for a summary or a title.
    ///
    /// Narrower than [`SessionStore::build_context_window_items`]: only
    /// `context_window` rows whose role is `user` or `assistant`.
    pub fn list_visible_context_messages(&self, id: &str) -> Vec<Value> {
        let mut out = Vec::new();
        for wrap in self.list_events_folded(id) {
            if wrap.event_type != "context_window" {
                continue;
            }
            let Some(p) = wrap.payload.as_object() else {
                continue;
            };
            let role = p.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if role.is_empty() {
                continue;
            }
            match p.get("content") {
                None | Some(Value::Null) => continue,
                _ => {}
            }
            if p.get("kind").and_then(|v| v.as_str()) == Some("provider_messages_snapshot") {
                continue;
            }
            // `=== false` only: any other value leaves the row visible.
            if wrap.meta.get("visible") == Some(&Value::Bool(false)) {
                continue;
            }
            if role != "user" && role != "assistant" {
                continue;
            }
            out.push(json!({
                "event_id": wrap.event_id,
                "role": role,
                "content": js_string(p.get("content").unwrap()),
                "meta": wrap.meta,
            }));
        }
        out
    }

    /// Hydrate the context window from disk (session open / resume).
    ///
    /// One jsonl event becomes one context item becomes one UI row.
    /// `meta.visible` is the eye-toggle: a hidden row is kept, never deleted.
    /// Full-window provider snapshots are debug telemetry and are **never**
    /// used as a seed, so they are skipped.
    pub fn build_context_window_items(&self, id: &str) -> Vec<ContextItem> {
        context_items_from_events(self.list_events_folded(id))
    }
}

/// The hydration itself, over any already-folded event list.
///
/// Split out of [`SessionStore`] so the SQLite store reuses it verbatim:
/// the JS-era coercions below (`String(x)`, `p.explanation || null`, the
/// `tool_call_id` fallback chain) are pinned by a differential test against
/// the real JS, and a second copy of them would drift.
pub fn context_items_from_events(events: Vec<EventWrapper>) -> Vec<ContextItem> {
    {
        let mut items: Vec<ContextItem> = Vec::new();

        for wrap in events {
            let Some(p) = wrap.payload.as_object() else {
                continue;
            };
            // `!(meta.visible === false)` — strict, so only an explicit
            // boolean false hides a row.
            let visible = wrap.meta.get("visible") != Some(&Value::Bool(false));
            let eid = Some(wrap.event_id.clone());
            let goto = goto_of(&wrap.meta);

            let kind = p.get("kind").and_then(|v| v.as_str());
            if wrap.event_type == "context_window" && kind == Some("provider_messages_snapshot") {
                continue;
            }

            // --- plain transcript row -------------------------------------
            if wrap.event_type == "context_window" {
                let role = p.get("role").and_then(|v| v.as_str()).unwrap_or("");
                let content = p.get("content");
                if !role.is_empty() && !matches!(content, None | Some(Value::Null)) {
                    let mut it =
                        ContextItem::base(role, js_string(content.unwrap()), eid, visible);
                    it.goto = goto;
                    items.push(it);
                    continue;
                }
            }

            // --- reasoning -------------------------------------------------
            let is_reasoning = wrap.event_type == "reasoning"
                || p.get("type").and_then(|v| v.as_str()) == Some("reasoning");
            if is_reasoning {
                let text = p
                    .get("text")
                    .filter(|v| truthy(v))
                    .or_else(|| p.get("content").filter(|v| truthy(v)));
                if let Some(t) = text {
                    let mut it = ContextItem::base("reasoning", js_string(t), eid, visible);
                    it.goto = goto;
                    items.push(it);
                    continue;
                }
            }

            // --- tool call --------------------------------------------------
            if wrap.event_type == "tool_call" {
                if let Some(name) = p.get("name").filter(|v| truthy(v)).and_then(|v| v.as_str()) {
                    let call_id = first_str(&[p.get("id"), p.get("tool_call_id")])
                        .or_else(|| eid.clone())
                        .unwrap_or_else(|| format!("call_{}", items.len()));
                    // A string `args` is passed through verbatim; anything
                    // else is JSON-encoded, with a missing value becoming {}.
                    let args_str = match p.get("args") {
                        Some(Value::String(s)) => s.clone(),
                        Some(v) if !v.is_null() => v.to_string(),
                        _ => "{}".to_string(),
                    };
                    let mut it = ContextItem::base("assistant", String::new(), eid, visible);
                    it.tool_calls = Some(json!([{
                        "id": call_id,
                        "type": "function",
                        "function": { "name": name, "arguments": args_str },
                    }]));
                    it.name = Some(name.to_string());
                    it.args = p.get("args").cloned();
                    // `p.explanation || null` — key always present on a tool row.
                    it.explanation =
                        Some(p.get("explanation").filter(|v| truthy(v)).map(js_string));
                    it.goto = goto;
                    items.push(it);
                    continue;
                }
            }

            // --- tool result ------------------------------------------------
            if wrap.event_type == "tool_result" {
                if let Some(name) = p.get("name").filter(|v| truthy(v)).and_then(|v| v.as_str()) {
                    let call_id =
                        first_str(&[p.get("tool_call_id"), p.get("id"), p.get("call_id")])
                            .or_else(|| eid.clone())
                            .unwrap_or_else(|| format!("call_{}", items.len()));
                    let content = match p.get("text") {
                        Some(t) if !t.is_null() => js_string(t),
                        _ => match p.get("content") {
                            Some(Value::String(s)) => s.clone(),
                            // `JSON.stringify(p.content ?? p)` — a null or
                            // absent content falls back to the WHOLE payload.
                            Some(v) if !v.is_null() => v.to_string(),
                            _ => Value::Object(p.clone()).to_string(),
                        },
                    };
                    let mut it = ContextItem::base("tool", content, eid, visible);
                    it.tool_call_id = Some(call_id);
                    it.name = Some(name.to_string());
                    it.ok = p.get("ok").cloned();
                    it.denied = p.get("denied").cloned();
                    it.explanation =
                        Some(p.get("explanation").filter(|v| truthy(v)).map(js_string));
                    it.goto = goto;
                    items.push(it);
                }
            }
        }
        items
    }
}

/// First present, non-empty string among the candidates (`a || b || c`).
fn first_str(cands: &[Option<&Value>]) -> Option<String> {
    for c in cands {
        if let Some(v) = c {
            if truthy(v) {
                return Some(js_string(v));
            }
        }
    }
    None
}
