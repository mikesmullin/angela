//! "Should this agent speak?" — the `relevant` engagement mode.
//!
//! An agent set to `relevant` does not answer everything and does not wait to
//! be named. Before each round a **System One** model is asked, once per
//! agent, whether this particular message is that agent's business. It is a
//! finite-choice decision, not a generation: the endpoint returns a choice
//! with probabilities and no tokens, so it is fast and cheap enough to run
//! for every seated agent on every message.
//!
//! The question is deliberately framed around *who the agent is* rather than
//! around the conversation's topic. "Is this about deploys?" is a question
//! the model can only answer well if it already knows the roster; "you are
//! an ops engineer who reads deploy logs — is this yours?" is answerable from
//! the two things in front of it.
//!
//! When the decision endpoint is unreachable, misconfigured, or answers
//! something unexpected, the agent **stays quiet**. That default is the
//! conservative one: an agent that speaks when it should not is disruptive
//! and costs a generation, while one that stays quiet can still be
//! @mentioned.

use serde_json::{json, Value};

use crate::conv_store::Participant;

/// Where the System One endpoint lives.
///
/// `ANGELA_DECISIONS_URL` overrides. The default is the local llama-server
/// decision build on :1237 — a *different* server from the one that
/// generates, which is the point: deciding and generating want different
/// models.
pub fn endpoint_url() -> String {
    std::env::var("ANGELA_DECISIONS_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:1237/v1".to_string())
}

fn decision_model() -> String {
    std::env::var("ANGELA_DECISIONS_MODEL").unwrap_or_default()
}

/// The question put to the decision model.
///
/// `choice` rather than a boolean because the endpoint's three question kinds
/// are `choice`, `score` and `noul`, and a two-way choice with written
/// criteria gives the model something to anchor on — a bare boolean does not
/// say what "yes" would mean.
///
/// Short, and in the second person. The state already says who the agent is
/// and what was said; restating that here as instructions ("judge only from
/// who they are…") spends tokens telling the model what it can already see,
/// and the third person makes it reason about a participant rather than as
/// one.
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

/// What the decision model said.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub speak: bool,
    /// `0.0..=1.0`, or `None` when the endpoint reported none.
    pub confidence: Option<f64>,
    /// Set when the endpoint could not be used; `speak` is then false.
    /// Distinct from being talked out of it: "nobody was asked" and "the
    /// model was not sure enough" are different facts about a quiet agent.
    pub error: Option<String>,
    /// Why it came out this way, in words, for the decision card.
    pub reason: String,
    /// Exactly what was sent, and what came back.
    ///
    /// Kept so the decision can be shown as a card: "why did @fish not
    /// answer that" is unanswerable from a boolean, and the state is the
    /// only place the description the model judged on is visible.
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

/// How much of the agent's identity the decision prompt carries.
///
/// A decision prompt is not a generation prompt: a 4 KB persona would
/// dominate the conversation it is being compared against.
const MAX_IDENTITY_CHARS: usize = 1200;
/// How many recent messages it carries, newest last.
const MAX_HISTORY_LINES: usize = 12;
/// …and how much of each, so one pasted file cannot crowd out the rest.
const MAX_HISTORY_LINE_CHARS: usize = 600;

/// The state put to the decision model, as prose addressed to the agent.
///
/// A structured object made the model reason *about* a participant; written
/// this way it reasons *as* one, which is the question being asked. It is
/// also the same identity the agent itself is given — soul, goals, memory —
/// so the decision is made on the same picture the reply would be.
///
/// The newest message is simply the last line of the history. It used to be
/// a separate field as well, which meant every prompt carried it twice.
pub fn build_state(name: &str, identity: &str, history: &[String]) -> String {
    let clipped: String = identity.trim().chars().take(MAX_IDENTITY_CHARS).collect();
    // The same preamble the agent's own turn is built from: it should be
    // deciding whether to speak from the picture it would speak from.
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

/// One history line: `14:00 [sender] text`.
///
/// The clock is how an agent gets a sense of time passing — whether the last
/// message was a minute ago or yesterday afternoon changes what a reply
/// should say, and nothing else in the prompt carries that.
pub fn history_line(ts: &str, sender: &str, text: &str) -> String {
    match crate::environment::epoch_from_iso(ts) {
        Some(e) => format!("{} [{sender}] {text}", crate::environment::clock(e)),
        // A timestamp we cannot read loses its clock, not its line.
        None => format!("[{sender}] {text}"),
    }
}

/// `--- Fri, Nov 13 ---`, the header that opens a day's run.
pub fn day_header(ts: &str) -> Option<String> {
    let e = crate::environment::epoch_from_iso(ts)?;
    Some(format!("--- {} ---", crate::environment::day_label(e)))
}

/// Ask whether `me` should take a turn on `message`.
///
/// `recent` is the last few lines of the conversation, oldest first, already
/// formatted; it gives the model the "has this been covered already" context
/// that the newest message alone cannot.
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
    // Stoppable like any other model call. It is a short one, but "stop"
    // that waits for a decision before taking effect is still a wait the
    // user did not ask for.
    let got = crate::cancel::block_on(rt, stop, async {
        client
            .decisions(&decision_model(), &state, &question, None)
            .await
    });
    let answers = match got {
        // Silence is the right answer to a stopped decision: it means this
        // agent does not take a turn, which is what stopping asked for.
        crate::cancel::Ran::Stopped => {
            let mut v = quiet("stopped".to_string());
            v.state = state;
            v.question = question;
            return v;
        }
        crate::cancel::Ran::Done(Ok(d)) => d.answers,
        crate::cancel::Ran::Done(Err(e)) => {
            let mut v = quiet(e.to_string());
            // Even a failed call is worth showing: it is the difference
            // between "the model said no" and "nobody was asked".
            v.state = state;
            v.question = question;
            return v;
        }
    };
    judge(answers, state, question)
}

/// Turn the endpoint's answer into a verdict.
///
/// Split out so the rule is testable without a decision server, which is
/// the part worth testing.
fn judge(answers: Value, state: Value, question: Value) -> Verdict {
    let a = answers["should_reply"].clone();
    let choice = a["choice"].as_str().unwrap_or("").to_string();
    let confidence = a["confidence"].as_f64();
    let pct = |c: Option<f64>| match c {
        Some(c) => format!("{:.0}%", c * 100.0),
        None => "no confidence reported".to_string(),
    };
    // The decider's answer is the answer. Confidence is reported, and shown
    // on the card, but it does not gate: a model that says yes is taken at
    // its word.
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
        // A shape we do not recognise is not a yes.
        other => Verdict::quiet(format!("unexpected decision {other:?}")),
    };
    v.state = state;
    v.question = question;
    v.answers = answers;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seat(handle: &str, display: &str) -> Participant {
        Participant {
            id: 1,
            conversation_id: "c".into(),
            kind: "agent".into(),
            handle: handle.into(),
            display_name: display.into(),
            agent_name: Some(handle.into()),
            model: None,
            system_prompt: String::new(),
            avatar: None,
            seat: 1,
            active: true,
            role: None,
            engagement: None,
        }
    }

    #[test]
    fn the_question_matches_the_endpoints_schema_and_stays_short() {
        let q = question();
        let s = &q["should_reply"];
        // The endpoint accepts only choice/score/noul, and a choice must
        // carry a non-empty `criteria` object — both 400 otherwise.
        assert_eq!(s["type"], "choice");
        assert!(s["criteria"].is_object());
        assert_eq!(s["criteria"].as_object().unwrap().len(), 2);
        // Short, and in the second person: the state already says who the
        // agent is and what was said, so instructions restating that spend
        // tokens on what the model can already see.
        let i = s["instructions"].as_str().unwrap();
        assert_eq!(i, "Should you respond?");
        for text in [i, s["criteria"]["yes"].as_str().unwrap(), s["criteria"]["no"].as_str().unwrap()] {
            assert!(text.len() < 60, "too wordy: {text}");
            assert!(!text.contains("participant"), "third person: {text}");
        }
    }

    #[test]
    fn the_state_reads_as_prose_addressed_to_the_agent() {
        let state = build_state(
            "Fish",
            "Whenever someone mentions fish You like to talk about Cajun creole cuisine\n\n             ## Goals\n\nSpread the values of the Cajun creole people",
            &[
                history_line("", "user", "hi"),
                history_line("", "fish", "Hey there!"),
                history_line("", "user", "i'm at the wet market"),
            ],
        );
        assert!(state.starts_with("Your name is Fish.\n"), "{state}");
        assert!(state.contains("Cajun creole cuisine"));
        // The agent's own goals, verbatim from the identity it is given.
        assert!(state.contains("## Goals"));
        assert!(state.contains("## Conversation History"));
        assert!(state.contains("**Session type:** Multi-user thread"));
        // Attributed, oldest first, newest last — the message being judged
        // is the final line, not a separate field repeating it.
        let lines: Vec<&str> = state.lines().filter(|l| l.starts_with('[')).collect();
        assert_eq!(
            lines,
            vec!["[user] hi", "[fish] Hey there!", "[user] i'm at the wet market"]
        );
    }

    #[test]
    fn an_agent_with_no_identity_still_gets_a_usable_state() {
        let state = build_state("scout", "", &[history_line("", "user", "hello")]);
        assert!(state.starts_with("Your name is scout.\n"));
        // No stray blank section where the identity would have been.
        assert!(!state.contains("\n\n\n"), "{state:?}");
        assert!(state.contains("[user] hello"));
    }

    #[test]
    fn a_long_persona_cannot_drown_out_the_conversation() {
        let huge = "word ".repeat(4000);
        let state = build_state("x", &huge, &[history_line("", "user", "hi")]);
        assert!(state.chars().count() < MAX_IDENTITY_CHARS + 400, "{}", state.len());
        // …and the conversation still makes it in.
        assert!(state.contains("[user] hi"));
    }

    #[test]
    fn only_the_recent_history_is_carried_and_each_line_is_bounded() {
        let many: Vec<String> = (0..40)
            .map(|i| history_line("", "user", &format!("message {i}")))
            .collect();
        let state = build_state("x", "", &many);
        let kept: Vec<&str> = state.lines().filter(|l| l.starts_with('[')).collect();
        assert_eq!(kept.len(), MAX_HISTORY_LINES);
        // The newest survive; the oldest are what gets dropped.
        assert!(kept.last().unwrap().contains("message 39"));
        assert!(!state.contains("message 0]"));

        // One pasted file cannot crowd out the rest.
        let long = history_line("", "user", &"x".repeat(5000));
        let state = build_state("x", "", &[long, history_line("", "user", "after")]);
        assert!(state.contains("[user] after"));
        assert!(state.lines().all(|l| l.chars().count() <= MAX_HISTORY_LINE_CHARS));
    }

    fn answer(choice: &str, confidence: f64) -> Value {
        json!({ "should_reply": { "choice": choice, "confidence": confidence } })
    }

    #[test]
    fn a_yes_is_taken_at_its_word() {
        // The bar this used to carry is gone: a model that says yes speaks,
        // whatever its margin. Confidence is still reported — it is on the
        // decision card — it just does not gate.
        for sure in [0.05, 0.33, 0.49, 0.85] {
            let v = judge(answer("yes", sure), Value::Null, Value::Null);
            assert!(v.speak, "{sure} should still speak");
            assert_eq!(v.confidence, Some(sure));
            assert!(v.error.is_none());
            assert!(v.reason.starts_with("relevant,"), "{}", v.reason);
        }
    }

    #[test]
    fn a_yes_with_no_confidence_reported_still_speaks() {
        let v = judge(
            json!({ "should_reply": { "choice": "yes" } }),
            Value::Null,
            Value::Null,
        );
        assert!(v.speak);
        assert!(v.reason.contains("no confidence reported"), "{}", v.reason);
    }

    #[test]
    fn a_no_stays_a_no_however_sure_it_is() {
        for c in [0.1, 0.9] {
            let v = judge(answer("no", c), Value::Null, Value::Null);
            assert!(!v.speak);
            assert!(v.reason.contains("not relevant"), "{}", v.reason);
        }
    }

    #[test]
    fn an_unrecognised_answer_is_not_a_yes() {
        let v = judge(
            json!({ "should_reply": { "choice": "maybe", "confidence": 0.99 } }),
            Value::Null,
            Value::Null,
        );
        assert!(!v.speak);
        assert!(v.error.is_some(), "a shape we do not understand is reported");
    }

    #[test]
    fn an_unreachable_endpoint_keeps_the_agent_quiet() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        std::env::set_var("ANGELA_DECISIONS_URL", "http://127.0.0.1:9/v1");
        let never = crate::cancel::Cancel::never();
        let v = should_speak(&rt, &never, &seat("x", "x"), "", &[history_line("", "user", "hello")]);
        // Quiet, and the reason is reported rather than swallowed: an agent
        // that speaks when it should not is the more disruptive failure.
        assert!(!v.speak);
        assert!(v.error.is_some(), "the reason must survive");
        std::env::remove_var("ANGELA_DECISIONS_URL");
    }

    #[test]
    fn the_endpoint_is_configurable_and_defaults_to_the_decision_server() {
        std::env::remove_var("ANGELA_DECISIONS_URL");
        assert!(endpoint_url().contains("1237"));
        std::env::set_var("ANGELA_DECISIONS_URL", "http://example.invalid/v1");
        assert_eq!(endpoint_url(), "http://example.invalid/v1");
        std::env::remove_var("ANGELA_DECISIONS_URL");
    }
}
