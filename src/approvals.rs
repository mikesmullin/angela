//! Human approval for a tool the allowlist does not permit.
//!
//! The allowlist is deny-by-default, which is the right default and a
//! terrible ultimatum: without a way to say "yes, this once", every new tool
//! means editing a rule file mid-conversation. So a call that fails the list
//! does not fail the turn — it **stops** it, asks, and waits.
//!
//! ```text
//!   tool call ─► allowlist ─► allowed ──────────────────────────► run it
//!                    │
//!                    └─ not allowed ─► pinned this session? ────► run it
//!                                           │
//!                                           └─ ask ──► Allow   ► run it
//!                                                  ├─► Always  ► pin, run it
//!                                                  └─► Deny    ► tell the model
//! ```
//!
//! **Why the waiting is a block and not a message.** The turn is driven by
//! `rt.block_on` on the connection's own thread, and a tool handler is called
//! from inside that poll — so while it waits, that thread cannot write to the
//! NDJSON stream. The request therefore goes into this registry, where it is
//! read by `GET /api/pending` on *another* connection, and answered by
//! `POST /api/approve` on a third. The browser polls while a turn is in
//! flight; the card it draws is the same card the transcript keeps
//! afterwards.
//!
//! The timeout is not a nicety: a tab closed on an unanswered question would
//! otherwise hold an MCP server, a provider connection and a thread open for
//! as long as the process lives.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use crate::cancel::Cancel;

use serde_json::{json, Value};

/// How long an unanswered question waits before it answers itself with
/// "no". Long enough to read a command and think about it; short enough
/// that a forgotten tab does not pin resources indefinitely.
pub const ASK_TIMEOUT: Duration = Duration::from_secs(300);

/// What the person said.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Run it, this once.
    Allow,
    /// Run it, and stop asking about this exact call for this session.
    Always,
    /// Do not run it; the model is told so and can try something else.
    Deny,
}

impl Decision {
    /// Parses the wire form. Anything unrecognised is a **deny**: a
    /// malformed answer to "may I run this?" must not read as yes.
    pub fn parse(s: &str) -> Decision {
        match s.trim().to_ascii_lowercase().as_str() {
            "allow" | "yes" | "y" => Decision::Allow,
            "always" | "a" => Decision::Always,
            _ => Decision::Deny,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Decision::Allow => "allow",
            Decision::Always => "always",
            Decision::Deny => "deny",
        }
    }

    pub fn allowed(self) -> bool {
        !matches!(self, Decision::Deny)
    }
}

/// One question, as the UI sees it.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: String,
    pub session: String,
    pub tool: String,
    /// The string the allowlist was matched against — a terminal tool's
    /// command line, or the arguments. The card shows it, because "why was
    /// this not permitted?" is unanswerable without it.
    pub command: String,
    pub args: Value,
    pub reason: String,
}

impl Request {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "sessionId": self.session,
            "kind": "approval",
            "name": self.tool,
            "command": self.command,
            "args": self.args,
            "reason": self.reason,
            "status": "pending",
        })
    }
}

#[derive(Default)]
struct State {
    /// Questions waiting on an answer, by request id.
    waiting: HashMap<String, (Request, mpsc::Sender<Decision>)>,
    /// `session \0 tool \0 command` for every "always" given this session.
    pins: HashSet<String>,
    /// Answered questions, per session, until the turn that raised them
    /// picks them up and writes them to the transcript.
    answered: HashMap<String, Vec<Value>>,
    next_id: u64,
}

#[derive(Default)]
pub struct Approvals {
    state: Mutex<State>,
}

fn pin_key(session: &str, tool: &str, command: &str) -> String {
    format!("{session}\0{tool}\0{command}")
}

impl Approvals {
    pub fn new() -> Approvals {
        Approvals::default()
    }

    /// Has this exact call already been waved through for this session?
    ///
    /// Both forms are checked: "always" on a bare tool permits the tool,
    /// "always" on a tool **with** a command permits only that command —
    /// approving `bash: ls -la` must not approve `bash: rm -rf /`.
    pub fn pinned(&self, session: &str, tool: &str, command: &str) -> bool {
        let s = self.state.lock().unwrap();
        s.pins.contains(&pin_key(session, tool, command))
            || s.pins.contains(&pin_key(session, tool, ""))
    }

    /// Ask, and **block** until someone answers or [`ASK_TIMEOUT`] passes.
    ///
    /// Called from a tool handler, which is to say from inside the turn:
    /// everything that turn would do next is waiting on this.
    pub fn ask(&self, session: &str, tool: &str, command: &str, args: &Value) -> Decision {
        self.ask_cancellable(session, tool, command, args, &Cancel::never())
    }

    pub fn ask_cancellable(&self, session: &str, tool: &str, command: &str, args: &Value, stop: &Cancel) -> Decision {
        if stop.stopped() { return Decision::Deny; }
        let (tx, rx) = mpsc::channel();
        let req = {
            let mut s = self.state.lock().unwrap();
            s.next_id += 1;
            let req = Request {
                id: format!("apr{}", s.next_id),
                session: session.to_string(),
                tool: tool.to_string(),
                command: command.to_string(),
                args: args.clone(),
                reason: "not on this session's allowlist".to_string(),
            };
            s.waiting.insert(req.id.clone(), (req.clone(), tx));
            req
        };

        let deadline = Instant::now() + ASK_TIMEOUT;
        let decision = loop {
            if stop.stopped() { break Decision::Deny; }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() { break Decision::Deny; }
            match rx.recv_timeout(remaining.min(Duration::from_millis(50))) {
                Ok(d) => break if stop.stopped() { Decision::Deny } else { d },
                Err(RecvTimeoutError::Timeout) => {},
                Err(RecvTimeoutError::Disconnected) => break Decision::Deny,
            }
        };

        let mut s = self.state.lock().unwrap();
        s.waiting.remove(&req.id);
        if decision == Decision::Always {
            s.pins.insert(pin_key(session, tool, command));
        }
        let mut row = req.to_json();
        row["status"] = json!(decision.as_str());
        s.answered.entry(session.to_string()).or_default().push(row);
        decision
    }

    /// Answer one. `false` when there is no such question — it was already
    /// answered, or it timed out while the browser was deciding.
    pub fn resolve(&self, id: &str, decision: Decision) -> bool {
        let sender = {
            let s = self.state.lock().unwrap();
            s.waiting.get(id).map(|(_, tx)| tx.clone())
        };
        match sender {
            // The waiter removes its own entry, so a send that fails means
            // it has already given up.
            Some(tx) => tx.send(decision).is_ok(),
            None => false,
        }
    }

    /// What this session is waiting on, oldest first — what the browser
    /// polls for while a turn is in flight.
    pub fn pending(&self, session: &str) -> Vec<Value> {
        let s = self.state.lock().unwrap();
        let mut out: Vec<(&String, Value)> = s
            .waiting
            .iter()
            .filter(|(_, (r, _))| r.session == session)
            .map(|(id, (r, _))| (id, r.to_json()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(b.0));
        out.into_iter().map(|(_, v)| v).collect()
    }

    /// Take the answered questions so the turn can write them to the
    /// transcript. Draining is deliberate: they belong to the conversation
    /// now, not to this registry.
    pub fn take_answered(&self, session: &str) -> Vec<Value> {
        let mut s = self.state.lock().unwrap();
        s.answered.remove(session).unwrap_or_default()
    }

    /// A session is gone: deny whatever it was waiting on (dropping the
    /// sender unblocks the waiter), and drop its pins.
    pub fn forget(&self, session: &str) {
        let mut s = self.state.lock().unwrap();
        s.waiting.retain(|_, (r, _)| r.session != session);
        s.pins.retain(|k| !k.starts_with(&format!("{session}\0")));
        s.answered.remove(session);
    }

    #[cfg(test)]
    fn waiting_count(&self) -> usize {
        self.state.lock().unwrap().waiting.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn ask_in_background(a: &Arc<Approvals>, cmd: &'static str) -> thread::JoinHandle<Decision> {
        let a = Arc::clone(a);
        thread::spawn(move || a.ask("s1", "term_local__bash", cmd, &json!({})))
    }

    /// Wait for the question to show up rather than sleeping a fixed amount:
    /// the asking thread has to get as far as registering before the answer
    /// can land, and a sleep that is long enough on this machine is a flake
    /// on a slower one.
    fn wait_for_question(a: &Approvals) -> String {
        for _ in 0..200 {
            let p = a.pending("s1");
            if let Some(first) = p.first() {
                return first["id"].as_str().unwrap().to_string();
            }
            thread::sleep(Duration::from_millis(5));
        }
        panic!("no question was ever asked");
    }

    #[test]
    fn stop_cancels_and_removes_a_pending_approval() {
        let a = Arc::new(Approvals::new());
        let stop = Cancel::never();
        let worker_stop = stop.clone();
        let worker = a.clone();
        let h = thread::spawn(move || worker.ask_cancellable("s1", "tool", "", &json!({}), &worker_stop));
        let id = wait_for_question(&a);
        stop.stop();
        assert_eq!(h.join().unwrap(), Decision::Deny);
        assert!(a.pending("s1").is_empty());
        assert!(!a.resolve(&id, Decision::Allow));
    }

    #[test]
    fn an_answer_releases_the_waiting_turn() {
        let a = Arc::new(Approvals::new());
        let h = ask_in_background(&a, "ls -la");
        let id = wait_for_question(&a);
        assert!(a.resolve(&id, Decision::Allow));
        assert_eq!(h.join().unwrap(), Decision::Allow);
        assert_eq!(a.waiting_count(), 0);
    }

    #[test]
    fn always_pins_only_the_call_that_was_approved() {
        let a = Arc::new(Approvals::new());
        let h = ask_in_background(&a, "ls -la");
        let id = wait_for_question(&a);
        a.resolve(&id, Decision::Always);
        assert_eq!(h.join().unwrap(), Decision::Always);

        assert!(a.pinned("s1", "term_local__bash", "ls -la"));
        // The dangerous part: approving one command must not approve the
        // next one that happens to use the same tool.
        assert!(!a.pinned("s1", "term_local__bash", "rm -rf /"));
        // Nor the same command in a different conversation.
        assert!(!a.pinned("s2", "term_local__bash", "ls -la"));
    }

    #[test]
    fn an_unknown_id_is_not_an_answer() {
        let a = Approvals::new();
        assert!(!a.resolve("apr404", Decision::Allow));
    }

    #[test]
    fn a_forgotten_session_stops_waiting() {
        let a = Arc::new(Approvals::new());
        let h = ask_in_background(&a, "sleep 9999");
        wait_for_question(&a);
        a.forget("s1");
        // Dropping the sender is what releases it; the answer is no.
        assert_eq!(h.join().unwrap(), Decision::Deny);
    }

    #[test]
    fn the_answer_reaches_the_transcript_once() {
        let a = Arc::new(Approvals::new());
        let h = ask_in_background(&a, "ls -la");
        let id = wait_for_question(&a);
        a.resolve(&id, Decision::Deny);
        h.join().unwrap();

        let rows = a.take_answered("s1");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["status"], "deny");
        assert_eq!(rows[0]["name"], "term_local__bash");
        assert_eq!(rows[0]["command"], "ls -la");
        // Drained, not copied: a second turn must not replay it.
        assert!(a.take_answered("s1").is_empty());
    }

    #[test]
    fn anything_that_is_not_a_yes_is_a_no() {
        assert_eq!(Decision::parse("allow"), Decision::Allow);
        assert_eq!(Decision::parse("ALWAYS"), Decision::Always);
        assert_eq!(Decision::parse("deny"), Decision::Deny);
        assert_eq!(Decision::parse(""), Decision::Deny);
        assert_eq!(Decision::parse("sure, why not"), Decision::Deny);
    }
}
