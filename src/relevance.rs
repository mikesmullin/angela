use serde_json::{json, Value};

use crate::conv_store::Participant;

pub fn endpoint_url() -> String {
    std::env::var("ANGELA_DECISIONS_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:1237/v1".to_string())
}

fn decision_model() -> String {
    std::env::var("ANGELA_DECISIONS_MODEL").unwrap_or_default()
}

fn question() -> Value {
    json!({
        "should_reply": {
            "type": "choice",
            "instructions": "Should you respond?",
            "criteria": {
                "yes": "You will add something to the conversation.",
                "no": "You remain silent this turn."
            }
        }
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub speak: bool,
    pub confidence: Option<f64>,
    pub error: Option<String>,
    pub reason: String,
    pub state: Value,
    pub question: Value,
    pub answers: Value,
}

impl Verdict {
    fn quiet(error: impl Into<String>) -> Self {
        let error = error.into();
        Verdict {
            speak: false,
            confidence: None,
            error: if error.is_empty() { None } else { Some(error.clone()) },
            reason: error,
            state: Value::Null,
            question: Value::Null,
            answers: Value::Null,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "speak": self.speak,
            "confidence": self.confidence,
            "reason": self.reason,
            "error": self.error,
            "state": self.state,
            "question": self.question,
            "answers": self.answers,
        })
    }
}

const MAX_IDENTITY_CHARS: usize = 1200;
const MAX_HISTORY_LINES: usize = 12;
const MAX_HISTORY_LINE_CHARS: usize = 600;

pub fn build_state(name: &str, identity: &str, history: &[String]) -> String {
    let clipped: String = identity.trim().chars().take(MAX_IDENTITY_CHARS).collect();
    let mut out = crate::group::preamble(name, &clipped);
    out.push('\n');
    let start = history.len().saturating_sub(MAX_HISTORY_LINES);
    for line in &history[start..] {
        let clipped: String = line.chars().take(MAX_HISTORY_LINE_CHARS).collect();
        out.push_str(&clipped);
        out.push('\n');
    }
    out
}

pub fn history_line(ts: &str, sender: &str, text: &str) -> String {
    match crate::environment::epoch_from_iso(ts) {
        Some(e) => format!("{} [{sender}] {text}", crate::environment::clock(e)),
        None => format!("[{sender}] {text}"),
    }
}

pub fn day_header(ts: &str) -> Option<String> {
    let e = crate::environment::epoch_from_iso(ts)?;
    Some(format!("--- {} ---", crate::environment::day_label(e)))
}

pub fn should_speak(
    rt: &tokio::runtime::Runtime,
    stop: &crate::cancel::Cancel,
    me: &Participant,
    identity: &str,
    history: &[String],
) -> Verdict {
    let quiet = Verdict::quiet;
    let client = match agl::provider::Generic::new(agl::provider::Endpoint {
        base_url: endpoint_url(),
        api_key: std::env::var("ANGELA_DECISIONS_API_KEY").ok(),
        ca_file: None,
    }) {
        Ok(c) => c,
        Err(e) => return quiet(e.to_string()),
    };

    let name = if me.display_name.trim().is_empty() {
        me.handle.clone()
    } else {
        me.display_name.clone()
    };
    let state = json!(build_state(&name, identity, history));

    let question = question();
    let got = crate::cancel::block_on(rt, stop, async {
        client
            .decisions(&decision_model(), &state, &question, None)
            .await
    });
    let answers = match got {
        crate::cancel::Ran::Stopped => {
            let mut v = quiet("stopped".to_string());
            v.state = state;
            v.question = question;
            return v;
        }
        crate::cancel::Ran::Done(Ok(d)) => d.answers,
        crate::cancel::Ran::Done(Err(e)) => {
            let mut v = quiet(e.to_string());
            v.state = state;
            v.question = question;
            return v;
        }
    };
    judge(answers, state, question)
}

fn judge(answers: Value, state: Value, question: Value) -> Verdict {
    let a = answers["should_reply"].clone();
    let choice = a["choice"].as_str().unwrap_or("").to_string();
    let confidence = a["confidence"].as_f64();
    let pct = |c: Option<f64>| match c {
        Some(c) => format!("{:.0}%", c * 100.0),
        None => "no confidence reported".to_string(),
    };
    let mut v = match choice.as_str() {
        "yes" => Verdict {
            speak: true,
            confidence,
            reason: format!("relevant, {} sure", pct(confidence)),
            ..Verdict::quiet("")
        },
        "no" => Verdict {
            speak: false,
            confidence,
            reason: format!("not relevant, {} sure", pct(confidence)),
            ..Verdict::quiet("")
        },
        other => Verdict::quiet(format!("unexpected decision {other:?}")),
    };
    v.state = state;
    v.question = question;
    v.answers = answers;
    v
}
