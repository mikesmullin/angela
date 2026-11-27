//! The message queue — prompts waiting to become turns.
//!
//! A typed message does not become history the moment it is sent. It enters
//! **the queue**, is drawn inside the "waiting for" card (with why it is
//! waiting and a way to take it back), and only when its turn starts does it
//! leave the queue and become a user prompt card in the transcript. Until
//! then it is pending work, not a message that was said — which is exactly
//! what lets it be dequeued: nothing of it has been recorded yet.
//!
//! The waiting messages are rows (`queue_items`, see [`ConvStore`]); this
//! module holds the run-time half — **which message a session is running**.
//! A session runs one turn at a time, so "the running message" is what
//! turns the queue into an order rather than a pile: a message starts only
//! when it is the head of the line and nothing else is running.
//!
//! When a message starts, its row is deleted (the `context_window` node it
//! appends takes over as the durable record) and the session is marked busy
//! — in one step, so no two messages of one session can start together.

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::{json, Value};

use crate::conv_store::ConvStore;

/// Why the head of a line is waiting. The card explains the line by it.
pub const REASON_GATE: &str = "waiting for the session's workspace root and worktree";
pub const REASON_TURN: &str = "waiting for the current turn";
pub const REASON_AHEAD_ONE: &str = "waiting for the message ahead";
pub const REASON_STARTING: &str = "starting";

/// What came of asking to start one queued message.
pub enum Start {
    /// It won the line: the row is gone, the session is marked busy, and
    /// these are the model slots its turn now holds — taken in the same
    /// breath, so nobody can slip in between.
    Won(Vec<std::sync::Arc<agl::queue::Permit>>),
    /// Something else is running, an earlier message is still ahead, or the
    /// turn's model slot is taken.
    Later,
    /// It is no longer in the queue — dequeued, or the session went away
    /// while it waited.
    Gone,
}

/// The run-time half: which message each session is running.
#[derive(Default)]
pub struct Queues {
    running: Mutex<HashMap<String, String>>,
}

impl Queues {
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to take `id` out of the waiting line and onto the floor — and
    /// take its turn's model slots (`slots`) in the same breath.
    ///
    /// Only the head of the line may start, and only when the session is
    /// idle. Winning deletes the row, marks the session busy, and holds
    /// every slot the turn will model on — **one critical section**, which
    /// is the whole point: two waiting messages (in two sessions, say) must
    /// never both conclude that the floor and the slot are theirs. A taken
    /// slot means nothing starts: all-or-nothing, so a turn that begins
    /// never has to wait for its own model again.
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
                    // All-or-nothing: give back what was taken.
                    drop(permits);
                    return Start::Later;
                }
            }
        }
        store.queue_remove(session, id);
        running.insert(session.to_string(), id.to_string());
        Start::Won(permits)
    }

    /// The running message's turn ended: the floor is free again.
    pub fn finish(&self, session: &str, id: &str) {
        let mut running = self.running.lock().unwrap();
        if running.get(session).map(String::as_str) == Some(id) {
            running.remove(session);
        }
    }

    /// Whether a message has started here and not yet finished.
    pub fn busy(&self, session: &str) -> bool {
        self.running.lock().unwrap().contains_key(session)
    }
}

/// Why one message is waiting, for its row in the card.
///
/// `slots` are the model specs its turn will hold — when the line's head
/// waits on one of those, the wait is the model's, and the card names it.
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

/// One session's queue as the "waiting for" card draws it: the messages,
/// and the line's explanation (the head message's — it is what holds the
/// line up).
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

/// Every session's queue, keyed by session id — what the event feed pushes.
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

/// Marks a running message done when its turn ends — on every return path
/// of `run_turn`/`run_round`, without a line per path.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> ConvStore {
        ConvStore::memory("/tmp/queue-test").unwrap()
    }

    fn conv(s: &ConvStore) -> String {
        s.create(crate::session_store::NewSession::default())
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn one_message_per_session_takes_the_floor() {
        let s = store();
        let id = conv(&s);
        let q = Queues::new();
        let a = s.queue_push(&id, "first", "", "coder", "", "1");
        let b = s.queue_push(&id, "second", "", "coder", "", "1");
        // The head wins; the second cannot start behind it.
        assert!(matches!(q.try_start(&s, &id, &b, &[]), Start::Later));
        assert!(matches!(q.try_start(&s, &id, &a, &[]), Start::Won(_)));
        assert!(q.busy(&id));
        // A started message has no queue row left: its prompt node is the
        // record now, so dequeue can never take it back mid-run.
        assert!(s.queue_row(&id, &a).is_none());
        assert!(matches!(q.try_start(&s, &id, &b, &[]), Start::Later), "the floor is taken");
        q.finish(&id, &a);
        assert!(!q.busy(&id));
        assert!(matches!(q.try_start(&s, &id, &b, &[]), Start::Won(_)));
    }

    #[test]
    fn a_dequeued_message_just_is_gone() {
        let s = store();
        let id = conv(&s);
        let q = Queues::new();
        let a = s.queue_push(&id, "take me back", "", "coder", "", "1");
        assert!(s.queue_remove(&id, &a).is_some());
        assert!(matches!(q.try_start(&s, &id, &a, &[]), Start::Gone));
        // Dequeueing twice finds nothing, which is the honest answer.
        assert!(s.queue_remove(&id, &a).is_none());
    }

    #[test]
    fn the_snapshot_explains_the_line() {
        let s = store();
        let id = conv(&s);
        let q = Queues::new();
        let a = s.queue_push(&id, "first", "", "coder", "", "1");
        let b = s.queue_push(&id, "second", "", "coder", "", "1");
        assert_eq!(reason(&s, &q, &id, &a, false, &[]), REASON_GATE);
        assert_eq!(reason(&s, &q, &id, &a, true, &[]), REASON_STARTING);
        assert_eq!(reason(&s, &q, &id, &b, true, &[]), REASON_AHEAD_ONE);
        s.queue_set_reason(&id, &a, REASON_GATE);
        let snap = snapshot(&s, &id);
        assert_eq!(snap["reason"], json!(REASON_GATE));
        assert_eq!(snap["items"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn a_message_waits_on_the_model_slot_and_says_so() {
        // One message per session, one slot per model: a message whose
        // model is busy stays queued — with the model named on the card —
        // and the take that frees it is the start itself.
        let s = store();
        let id = conv(&s);
        let q = Queues::new();
        let item = s.queue_push(&id, "hello", "", "coder", "slot-test:m", "1");
        let slots = vec!["slot-test:m".to_string()];

        // The model is busy with another session's turn.
        let busy = agl::queue::try_acquire("slot-test:m").expect("free slot");
        assert!(matches!(q.try_start(&s, &id, &item, &slots), Start::Later));
        let why = reason(&s, &q, &id, &item, true, &slots);
        assert!(why.contains("slot-test:m"), "the card names the model: {why}");
        assert!(s.queue_row(&id, &item).is_some(), "still enqueued");

        // The other turn ends: the start takes the slot atomically.
        drop(busy);
        match q.try_start(&s, &id, &item, &slots) {
            Start::Won(permits) => assert_eq!(permits.len(), 1, "the slot came with the start"),
            _ => panic!("the slot is free — the message must start"),
        }
    }

    #[test]
    fn two_sessions_never_both_start_on_one_slot() {
        // The race the atomic take closes: both lines are free and both
        // messages are heads — and exactly one may take the slot.
        let s = store();
        let a = conv(&s);
        let b = conv(&s);
        let q = Queues::new();
        let ma = s.queue_push(&a, "from a", "", "coder", "race-test:m", "1");
        let mb = s.queue_push(&b, "from b", "", "coder", "race-test:m", "1");
        let slots = vec!["race-test:m".to_string()];
        let wa = q.try_start(&s, &a, &ma, &slots);
        let wb = q.try_start(&s, &b, &mb, &slots);
        let started = [matches!(wa, Start::Won(_)), matches!(wb, Start::Won(_))];
        assert_eq!(started, [true, false], "exactly one takes the slot");
    }

    #[test]
    fn deleting_a_session_takes_its_queue_with_it() {
        let s = store();
        let id = conv(&s);
        s.queue_push(&id, "orphan", "", "coder", "", "1");
        assert!(s.delete(&id));
        assert!(s.queue_all().is_empty());
    }
}
