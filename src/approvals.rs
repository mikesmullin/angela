use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use crate::cancel::Cancel;

use serde_json::{json, Value};

pub const ASK_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Always,
    Deny,
}

impl Decision {
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

#[derive(Debug, Clone)]
pub struct Request {
    pub id: String,
    pub session: String,
    pub tool: String,
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
    waiting: HashMap<String, (Request, mpsc::Sender<Decision>)>,
    pins: HashSet<String>,
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

    pub fn pinned(&self, session: &str, tool: &str, command: &str) -> bool {
        let s = self.state.lock().unwrap();
        s.pins.contains(&pin_key(session, tool, command))
            || s.pins.contains(&pin_key(session, tool, ""))
    }

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

    pub fn resolve(&self, id: &str, decision: Decision) -> bool {
        let sender = {
            let s = self.state.lock().unwrap();
            s.waiting.get(id).map(|(_, tx)| tx.clone())
        };
        match sender {
            Some(tx) => tx.send(decision).is_ok(),
            None => false,
        }
    }

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

    pub fn take_answered(&self, session: &str) -> Vec<Value> {
        let mut s = self.state.lock().unwrap();
        s.answered.remove(session).unwrap_or_default()
    }

    pub fn forget(&self, session: &str) {
        let mut s = self.state.lock().unwrap();
        s.waiting.retain(|_, (r, _)| r.session != session);
        s.pins.retain(|k| !k.starts_with(&format!("{session}\0")));
        s.answered.remove(session);
    }


}
