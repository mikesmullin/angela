use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::{json, Value};

use crate::conv_store::ConvStore;

pub const REASON_GATE: &str = "waiting for the session's workspace root and worktree";
pub const REASON_TURN: &str = "waiting for the current turn";
pub const REASON_AHEAD_ONE: &str = "waiting for the message ahead";
pub const REASON_STARTING: &str = "starting";

pub enum Start {
    Won(Vec<std::sync::Arc<agl::queue::Permit>>),
    Later,
    Gone,
}

#[derive(Default)]
pub struct Queues {
    running: Mutex<HashMap<String, String>>,
}

impl Queues {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn try_start(&self, store: &ConvStore, session: &str, id: &str, slots: &[String]) -> Start {
        let mut running = self.running.lock().unwrap();
        if running.contains_key(session) {
            return Start::Later;
        }
        if store.queue_row(session, id).is_none() {
            return Start::Gone;
        }
        let head = store
            .queue_list(session)
            .first()
            .and_then(|r| r.get("id"))
            .and_then(Value::as_str)
            .map(str::to_string);
        if head.as_deref() != Some(id) {
            return Start::Later;
        }
        let mut permits = Vec::new();
        for spec in slots {
            match agl::queue::try_acquire(spec) {
                Some(p) => permits.push(std::sync::Arc::new(p)),
                None => {
                    drop(permits);
                    return Start::Later;
                }
            }
        }
        store.queue_remove(session, id);
        running.insert(session.to_string(), id.to_string());
        Start::Won(permits)
    }

    pub fn finish(&self, session: &str, id: &str) {
        let mut running = self.running.lock().unwrap();
        if running.get(session).map(String::as_str) == Some(id) {
            running.remove(session);
        }
    }

    pub fn busy(&self, session: &str) -> bool {
        self.running.lock().unwrap().contains_key(session)
    }
}

pub fn reason(
    store: &ConvStore,
    queues: &Queues,
    session: &str,
    id: &str,
    gate_ok: bool,
    slots: &[String],
) -> String {
    if !gate_ok {
        return REASON_GATE.to_string();
    }
    let items = store.queue_list(session);
    let position = items
        .iter()
        .position(|r| r.get("id").and_then(Value::as_str) == Some(id))
        .unwrap_or(0);
    if position > 0 {
        return match position {
            1 => REASON_AHEAD_ONE.to_string(),
            n => format!("waiting for the {n} messages ahead"),
        };
    }
    if queues.busy(session) {
        return REASON_TURN.to_string();
    }
    for spec in slots {
        let st = agl::queue::stats(spec);
        if st.held >= st.limit || st.waiting > 0 {
            return format!("waiting for {spec} slot");
        }
    }
    REASON_STARTING.to_string()
}

pub fn snapshot(store: &ConvStore, session: &str) -> Value {
    let items = store.queue_list(session);
    let reason = items
        .first()
        .map(|r| {
            r.get("reason")
                .and_then(Value::as_str)
                .unwrap_or(REASON_STARTING)
                .to_string()
        })
        .unwrap_or_default();
    json!({ "items": items, "reason": reason })
}

pub fn snapshot_all(store: &ConvStore) -> Value {
    let mut by_session: HashMap<String, Vec<Value>> = HashMap::new();
    for row in store.queue_all() {
        let Some(sid) = row.get("sessionId").and_then(Value::as_str).map(str::to_string) else {
            continue;
        };
        let mut item = row.clone();
        if let Some(map) = item.as_object_mut() {
            map.remove("sessionId");
            map.remove("agent");
            map.remove("model");
            map.remove("tabId");
        }
        by_session.entry(sid).or_default().push(item);
    }
    let mut out = serde_json::Map::new();
    for (sid, items) in by_session {
        let reason = items
            .first()
            .map(|r| {
                r.get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or(REASON_STARTING)
                    .to_string()
            })
            .unwrap_or_default();
        out.insert(sid, json!({ "items": items, "reason": reason }));
    }
    Value::Object(out)
}

pub struct Running<'a> {
    queues: &'a Queues,
    session: String,
    id: String,
}

impl<'a> Running<'a> {
    pub fn new(queues: &'a Queues, session: &str, id: &str) -> Self {
        Running {
            queues,
            session: session.to_string(),
            id: id.to_string(),
        }
    }
}

impl Drop for Running<'_> {
    fn drop(&mut self) {
        self.queues.finish(&self.session, &self.id);
    }
}
