//! Group conversations: how N participants share one room without ever
//! sharing one LLM conversation.
//!
//! The architectural fact this module is built on (from the hermes agent, see
//! `tmp/GROUP_CHAT.md`): **a conversation with more than two participants is
//! not a multi-role LLM conversation.** You cannot simply add a third voice
//! as another `assistant` row — the provider has no way to express "this
//! assistant turn was someone else", and the model's own identity blurs into
//! the other agents'.
//!
//! So each agent participant runs in its own context, and sees the room as
//! *text*:
//!
//! ```text
//! [system]     this participant's own stable system prompt
//! [user]       its own earlier prompt   ┐ its private history, verbatim,
//! [assistant]  its own earlier reply    ┘ from `participant_session`
//! …
//! [user]       the flattened group state since its watermark  ← the room
//! ```
//!
//! Three consequences worth stating, because they are easy to undo by
//! accident:
//!
//! * **Nobody else's message is ever a role.** Other participants arrive as
//!   `@handle: …` lines inside that one current user message. The model can
//!   only tell "I said this" from "they said this" by those labels and its
//!   own system prompt, which is why the labels are not cosmetic.
//! * **The system prompt and the own history are byte-stable.** They are
//!   replayed exactly as stored, never re-rendered, so the provider's prompt
//!   cache keeps hitting. Only the final user message changes per turn.
//! * **The watermark is per `(conversation, participant)`.** It is what stops
//!   an agent being shown the same message twice, and what lets a participant
//!   who has been quiet for six turns catch up in one.
//!
//! Budgets and caps are hermes' own numbers, so a prompt this builds is the
//! size hermes would have built.

use sha2::{Digest, Sha256};

use crate::conv_store::{ConvStore, Participant, TreeNode};

/// Hard cap on the flattened prompt. Beyond this the prompt is a liability,
/// not context.
pub const MAX_PROMPT_BYTES: usize = 128 * 1024;
/// Lines of room history in one prompt, newest kept.
pub const MAX_DELTA_LINES: usize = 24;
/// Rounds before a discussion is declared bounded.
pub const MAX_ROUNDS: i64 = 3;
/// Messages in one discussion before it is declared bounded.
pub const MAX_MESSAGES: usize = 10;
/// A single participant's message, truncated at this size when published.
pub const MAX_MEMBER_TEXT_BYTES: usize = 64 * 1024;
/// How much of a tool result is worth showing a teammate.
const TOOL_SUMMARY_CHARS: usize = 80;

/// The marker a participant returns to stay silent.
pub const PASS: &str = "(pass)";

/// Everything one participant's turn needs, derived purely from durable
/// state — so a restart mid-round rebuilds exactly this.
#[derive(Debug, Clone)]
pub struct Task {
    pub participant_id: i64,
    pub handle: String,
    pub system_prompt: String,
    /// The participant's own `(prompt, reply)` history, oldest first.
    pub history: Vec<(String, String)>,
    /// The single current user message: the flattened room.
    pub prompt: String,
    pub round_index: i64,
    pub reply_required: bool,
    /// The highest `seq` this prompt includes.
    pub seen_through_seq: i64,
    /// `sha256` of the coordinates, including the prompt — replay-verifiable.
    pub task_id: String,
}

impl Task {
    /// The exact message list to send the provider.
    pub fn messages(&self) -> Vec<agl::provider::Message> {
        let mut out = vec![agl::provider::Message::system(self.system_prompt.clone())];
        for (prompt, reply) in &self.history {
            out.push(agl::provider::Message::user(prompt.clone()));
            out.push(agl::provider::Message::assistant(reply.clone()));
        }
        out.push(agl::provider::Message::user(self.prompt.clone()));
        out
    }
}

/// `@handle` mentions in a message, in order, de-duplicated.
///
/// Deliberately strict about the trailing boundary: `@bob's` mentions `bob`,
/// but `bob@example.com` is an address and mentions nobody, so a `@` must
/// start the token.
pub fn mentions(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'@' {
            i += 1;
            continue;
        }
        // A `@` glued to the end of a word is part of that word.
        if i > 0 {
            let prev = bytes[i - 1];
            if prev.is_ascii_alphanumeric() || prev == b'@' {
                i += 1;
                continue;
            }
        }
        let start = i + 1;
        let mut end = start;
        while end < bytes.len()
            && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'-' || bytes[end] == b'_')
        {
            end += 1;
        }
        if end > start {
            let h = text[start..end].to_ascii_lowercase();
            let h = h.trim_matches('-').to_string();
            if !h.is_empty() && !out.contains(&h) {
                out.push(h);
            }
        }
        i = end.max(i + 1);
    }
    out
}

/// Who speaks in this round.
///
/// Round 0: whoever the triggering message @-mentioned, or — when it
/// mentioned nobody — every active agent. Later rounds: only agents a *reply
/// in this round* pulled in, and only if they have not spoken in it yet.
/// That second rule is what makes a discussion terminate on its own: nobody
/// new mentioned means nobody new speaks.
pub fn responders(
    roster: &[Participant],
    trigger_text: &str,
    round_index: i64,
    round_texts: &[(i64, String)],
    already_replied: &[i64],
) -> Vec<i64> {
    let agents: Vec<&Participant> = roster.iter().filter(|p| p.is_agent() && p.active).collect();
    if round_index == 0 {
        let named = mentions(trigger_text);
        if named.is_empty() {
            return agents.iter().map(|p| p.id).collect();
        }
        return agents
            .iter()
            .filter(|p| named.contains(&p.handle))
            .map(|p| p.id)
            .collect();
    }
    let mut named: Vec<String> = Vec::new();
    for (_, text) in round_texts {
        for h in mentions(text) {
            if !named.contains(&h) {
                named.push(h);
            }
        }
    }
    agents
        .iter()
        .filter(|p| named.contains(&p.handle) && !already_replied.contains(&p.id))
        .map(|p| p.id)
        .collect()
}

/// One room line, as a teammate reads it.
///
/// `tool_call`, `reasoning` and `gen_info` are deliberately absent: they are
/// an agent's internal workings, and pasting them into a peer's prompt both
/// wastes the budget and invites the peer to answer the tool instead of the
/// person.
fn format_line(node: &TreeNode, roster: &[Participant]) -> Option<String> {
    let who = node
        .participant_id
        .and_then(|id| roster.iter().find(|p| p.id == id));
    match node.kind.as_str() {
        "context_window" => {
            let text = node.content.as_deref()?;
            match node.role.as_deref() {
                Some("user") => Some(format!("User (user): {text}")),
                Some("assistant") => {
                    let h = who.map(|p| p.handle.as_str()).unwrap_or("assistant");
                    Some(format!("@{h}: {text}"))
                }
                _ => None,
            }
        }
        "tool_result" => {
            let name = node
                .payload
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("tool");
            let raw = node.content.clone().unwrap_or_default();
            let flat: String = raw.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
            let short = truncate_chars(&flat, TOOL_SUMMARY_CHARS);
            Some(format!("(tool: {name} → \"{short}\")"))
        }
        "error" => node.content.as_deref().map(|t| format!("(error: {t})")),
        _ => None,
    }
}

fn truncate_chars(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= n {
        s.to_string()
    } else {
        format!("{}…", chars[..n].iter().collect::<String>())
    }
}

/// Truncate to at most `max` **bytes**, on a character boundary.
fn truncate_bytes(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    if max <= 3 {
        return String::new();
    }
    let budget = max - 3; // room for the ellipsis
    let mut end = 0;
    for (i, _) in s.char_indices() {
        if i > budget {
            break;
        }
        end = i;
    }
    format!("{}…", &s[..end])
}

/// Build the flattened group prompt for `me`.
///
/// Returns the prompt and the highest `seq` it covers. Packing is
/// **newest-first** on purpose: when the budget runs out it is the oldest
/// line that gets cut, because the newest line is the one being answered.
/// The sentinel is not decoration — without it the model cannot tell a short
/// history from a truncated one, and will happily refer to messages it never
/// saw.
pub fn flatten(
    title: &str,
    purpose: Option<&str>,
    notes: Option<&str>,
    me: &Participant,
    roster: &[Participant],
    delta: &[&TreeNode],
    reply_required: bool,
    how_you_got_here: &str,
) -> (String, i64) {
    let peers: Vec<String> = roster
        .iter()
        .filter(|p| p.id != me.id && p.active && p.is_agent())
        .map(|p| format!("@{}", p.handle))
        .collect();
    let others = if peers.is_empty() {
        "the user".to_string()
    } else {
        format!("{} and the user", peers.join(", "))
    };

    let mut opening = vec![
        format!(
            "[Conversation: \"{title}\"] You are @{}, one participant with {others}.",
            me.handle
        ),
        String::new(),
    ];
    if let Some(p) = purpose.filter(|s| !s.trim().is_empty()) {
        opening.push(format!("Conversation purpose (set by the user): {p}"));
    }
    if let Some(n) = notes.filter(|s| !s.trim().is_empty()) {
        opening.push(
            "Conversation notes (shared with every participant):".to_string(),
        );
        for line in n.lines() {
            opening.push(format!("  {line}"));
        }
    }
    if opening.len() > 2 {
        opening.push(String::new());
    }
    opening.push("New messages on this branch since your last turn (oldest first):".to_string());

    let mut rules = vec![String::new(), "Rules for this conversation:".to_string()];
    if reply_required {
        rules.push(format!(
            "  - {how_you_got_here} Reply with your answer, result, or concrete blocker."
        ));
        rules.push(format!(
            "  - Do not return \"{PASS}\". If you cannot proceed, explain why and what you need from the user."
        ));
    } else {
        rules.push(
            "  - Reply with one conversational message only when you have something new worth adding."
                .to_string(),
        );
        rules.push(format!(
            "  - If you have nothing new to add, reply with exactly \"{PASS}\"."
        ));
    }
    rules.push(
        "  - Mention a teammate by handle to pull them into the next round; do not repeat points already made."
            .to_string(),
    );
    rules.push("  - Your reply is published to every participant verbatim.".to_string());

    // The framing is counted once; whatever is left is the history budget.
    let fixed = opening.join("\n").len() + rules.join("\n").len();
    let mut available = MAX_PROMPT_BYTES.saturating_sub(fixed + 1);

    // Only the newest `MAX_DELTA_LINES` are candidates at all.
    let window: Vec<&&TreeNode> = delta.iter().rev().take(MAX_DELTA_LINES).collect();
    let mut seen_through = 0i64;
    for n in &window {
        seen_through = seen_through.max(n.seq);
    }

    let mut selected: Vec<String> = Vec::new();
    let mut truncated_any = false;
    for node in &window {
        let Some(body) = format_line(node, roster) else { continue };
        let line = format!("  {body}");
        if line.len() > available {
            // The oldest line that would not fit: keep a truncated head of it
            // when there is room worth keeping, then stop.
            if selected.is_empty() && available > 32 {
                selected.push(truncate_bytes(&line, available));
            }
            truncated_any = true;
            break;
        }
        available -= line.len() + 1;
        selected.push(line);
    }
    if truncated_any {
        selected.push("  [Earlier messages on this branch were omitted to fit this turn.]".into());
    }
    selected.reverse();
    if selected.is_empty() {
        selected.push("  (nothing new)".to_string());
    }

    let mut parts = opening;
    parts.extend(selected);
    parts.extend(rules);
    let prompt = parts.join("\n");
    // The cap is a hard one; a prompt over it is a bug upstream, and sending
    // it anyway would fail at the provider with a far less useful message.
    let prompt = if prompt.len() > MAX_PROMPT_BYTES {
        truncate_bytes(&prompt, MAX_PROMPT_BYTES)
    } else {
        prompt
    };
    (prompt, seen_through)
}

/// `sha256` of the coordinates a prompt is a pure function of.
///
/// Stored with the reply, so a replay can prove the prompt it is about to
/// re-send is the same one the durable state still implies.
pub fn task_id(
    conv: &str,
    participant_id: i64,
    cursor: &str,
    watermark: i64,
    round_index: i64,
    prompt: &str,
) -> String {
    let mut h = Sha256::new();
    h.update(conv.as_bytes());
    h.update([0]);
    h.update(participant_id.to_string().as_bytes());
    h.update([0]);
    h.update(cursor.as_bytes());
    h.update([0]);
    h.update(watermark.to_string().as_bytes());
    h.update([0]);
    h.update(round_index.to_string().as_bytes());
    h.update([0]);
    h.update(Sha256::digest(prompt.as_bytes()));
    format!("ctask:{}", hex(&h.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Assemble the turn for one participant from durable state alone.
pub fn plan(
    store: &ConvStore,
    conv: &str,
    me: &Participant,
    round_index: i64,
    reply_required: bool,
    how_you_got_here: &str,
) -> Option<Task> {
    let header = store.load(conv)?;
    let title = header
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("untitled")
        .to_string();
    let purpose = header.get("purpose").and_then(|v| v.as_str()).map(String::from);
    let notes = header.get("notes").and_then(|v| v.as_str()).map(String::from);
    let roster = store.participants(conv);
    let path = store.path(conv);
    let cursor = store.cursor(conv).unwrap_or_default();
    let (watermark, _, _, _) = store.participant_view(conv, me.id);

    // The delta: what is new on this branch, minus this participant's own
    // nodes — those are in its private history already, and showing an agent
    // its own words twice is how you teach it to repeat itself.
    let delta: Vec<&TreeNode> = path
        .iter()
        .filter(|n| n.seq > watermark && n.visible && n.participant_id != Some(me.id))
        .collect();

    let (prompt, seen_through) = flatten(
        &title,
        purpose.as_deref(),
        notes.as_deref(),
        me,
        &roster,
        &delta,
        reply_required,
        how_you_got_here,
    );
    let seen_through = if seen_through == 0 { watermark } else { seen_through };
    let task_id = task_id(conv, me.id, &cursor, watermark, round_index, &prompt);
    Some(Task {
        participant_id: me.id,
        handle: me.handle.clone(),
        system_prompt: me.system_prompt.clone(),
        history: store.own_session(me.id),
        prompt,
        round_index,
        reply_required,
        seen_through_seq: seen_through,
        task_id,
    })
}

/// Did the participant decline to speak?
pub fn is_pass(reply: &str) -> bool {
    let t = reply.trim();
    t.eq_ignore_ascii_case(PASS) || t.eq_ignore_ascii_case("pass")
}

/// Cap one published message, as hermes does.
pub fn clamp_message(text: &str) -> String {
    truncate_bytes(text, MAX_MEMBER_TEXT_BYTES)
}

/// Why a discussion stopped.
#[derive(Debug, Clone, PartialEq)]
pub enum Settled {
    /// Nobody new was addressed: the clean ending.
    SilentRound,
    /// `MAX_ROUNDS` reached.
    MaxRounds,
    /// `MAX_MESSAGES` reached.
    MaxMessages,
}

impl Settled {
    pub fn reason(&self) -> &'static str {
        match self {
            Settled::SilentRound => "silent_round",
            Settled::MaxRounds => "max_rounds",
            Settled::MaxMessages => "max_messages",
        }
    }
    pub fn status(&self) -> &'static str {
        match self {
            Settled::SilentRound => "settled",
            _ => "bounded",
        }
    }
}

/// Should the discussion continue? `None` means yes.
pub fn check_bounds(round_index: i64, messages: usize, responders: usize) -> Option<Settled> {
    if messages >= MAX_MESSAGES {
        return Some(Settled::MaxMessages);
    }
    if round_index >= MAX_ROUNDS {
        return Some(Settled::MaxRounds);
    }
    if responders == 0 {
        return Some(Settled::SilentRound);
    }
    None
}

/// Parse a lead's routing reply: the first balanced JSON object with an
/// `assign` key, tolerant of prose around it.
///
/// Tolerant because a lead is a chat model answering in prose by default;
/// insisting on a bare JSON body would make lead routing fail most of the
/// time, and the fallback (the lead answers alone) hides that failure.
pub fn parse_assignment(reply: &str) -> Option<Vec<String>> {
    let bytes = reply.as_bytes();
    for start in 0..bytes.len() {
        if bytes[start] != b'{' {
            continue;
        }
        let mut depth = 0i32;
        let mut in_str = false;
        let mut esc = false;
        for i in start..bytes.len() {
            let ch = bytes[i];
            if in_str {
                if esc {
                    esc = false;
                } else if ch == b'\\' {
                    esc = true;
                } else if ch == b'"' {
                    in_str = false;
                }
                continue;
            }
            match ch {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        let slice = &reply[start..=i];
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(slice) {
                            if let Some(a) = v.get("assign").and_then(|x| x.as_array()) {
                                let out: Vec<String> = a
                                    .iter()
                                    .filter_map(|x| x.as_str())
                                    .map(|s| s.trim_start_matches('@').to_ascii_lowercase())
                                    .filter(|s| !s.is_empty())
                                    .collect();
                                return Some(out);
                            }
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// The lead's routing prompt: who is here, what came in, answer with JSON.
pub fn routing_prompt(
    title: &str,
    purpose: Option<&str>,
    lead: &Participant,
    roster: &[Participant],
    message: &str,
) -> String {
    let mut out = vec![format!(
        "[Conversation: \"{title}\"] You are @{}, the lead of this conversation.",
        lead.handle
    )];
    if let Some(p) = purpose.filter(|s| !s.trim().is_empty()) {
        out.push(String::new());
        out.push(format!("Conversation purpose (set by the user): {p}"));
    }
    out.push(String::new());
    out.push("Participants:".to_string());
    for p in roster.iter().filter(|p| p.is_agent() && p.active) {
        if p.id == lead.id {
            out.push(format!("- @{} (you)", p.handle));
        } else {
            out.push(format!("- @{}", p.handle));
        }
    }
    out.push(String::new());
    out.push("The message (from the user):".to_string());
    out.push(format!("  {}", message.replace('\n', "\n  ")));
    out.push(String::new());
    out.push(
        "Reply with JSON naming who should answer, e.g. {\"assign\": [\"@alice\", \"@bob\"]}. \
         Assign yourself if you are the right one to answer."
            .to_string(),
    );
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_store::{NewEvent, NewSession};
    use serde_json::json;

    fn seat(kind: &str, id: i64, handle: &str) -> Participant {
        Participant {
            id,
            conversation_id: "c".into(),
            kind: kind.into(),
            handle: handle.into(),
            display_name: handle.into(),
            agent_name: Some(handle.into()),
            model: None,
            system_prompt: format!("You are {handle}."),
            avatar: None,
            seat: id,
            active: true,
            role: Some("member".into()),
        }
    }

    #[test]
    fn mentions_are_handles_not_email_addresses() {
        assert_eq!(mentions("hey @alice and @bob!"), vec!["alice", "bob"]);
        assert_eq!(mentions("@Carol, look"), vec!["carol"]);
        assert_eq!(mentions("write to bob@example.com"), Vec::<String>::new());
        assert_eq!(mentions("@alice @alice"), vec!["alice"], "de-duplicated");
        assert_eq!(mentions("ask @multi-word-handle ok"), vec!["multi-word-handle"]);
        assert_eq!(mentions("no one here"), Vec::<String>::new());
    }

    #[test]
    fn round_zero_addresses_everyone_unless_the_user_named_names() {
        let roster = vec![seat("user", 1, "user"), seat("agent", 2, "alice"), seat("agent", 3, "bob")];
        assert_eq!(responders(&roster, "deploy is failing", 0, &[], &[]), vec![2, 3]);
        assert_eq!(responders(&roster, "@bob can you look?", 0, &[], &[]), vec![3]);
    }

    #[test]
    fn later_rounds_only_speak_when_pulled_in_and_not_twice() {
        let roster = vec![seat("user", 1, "user"), seat("agent", 2, "alice"), seat("agent", 3, "bob")];
        let said = vec![(2i64, "checked it, @bob please restart the pod".to_string())];
        assert_eq!(responders(&roster, "", 1, &said, &[]), vec![3]);
        // bob already spoke this round: no second turn from one mention.
        assert_eq!(responders(&roster, "", 1, &said, &[3]), Vec::<i64>::new());
        // nobody mentioned → the round is silent, which is how it terminates
        assert_eq!(
            responders(&roster, "", 1, &[(2, "all done".into())], &[]),
            Vec::<i64>::new()
        );
    }

    #[test]
    fn a_bounded_discussion_stops_on_each_cap() {
        assert_eq!(check_bounds(0, 0, 2), None);
        assert_eq!(check_bounds(0, 0, 0), Some(Settled::SilentRound));
        assert_eq!(check_bounds(MAX_ROUNDS, 0, 2), Some(Settled::MaxRounds));
        assert_eq!(check_bounds(0, MAX_MESSAGES, 2), Some(Settled::MaxMessages));
        assert_eq!(Settled::SilentRound.status(), "settled");
        assert_eq!(Settled::MaxRounds.status(), "bounded");
    }

    #[test]
    fn a_pass_is_recognised_however_it_is_typed() {
        assert!(is_pass("(pass)"));
        assert!(is_pass("  (PASS) \n"));
        assert!(is_pass("pass"));
        assert!(!is_pass("I pass on that, but here is why"));
    }

    #[test]
    fn an_assignment_survives_prose_around_it() {
        assert_eq!(
            parse_assignment("Sure thing. {\"assign\": [\"@alice\", \"bob\"]} — those two."),
            Some(vec!["alice".to_string(), "bob".to_string()])
        );
        assert_eq!(parse_assignment("I think alice should do it"), None);
        // A nested object must not end the scan early.
        assert_eq!(
            parse_assignment("{\"why\": {\"a\": 1}, \"assign\": [\"@carol\"]}"),
            Some(vec!["carol".to_string()])
        );
    }

    // --- the prompt itself --------------------------------------------------

    fn room() -> (ConvStore, String, Vec<Participant>) {
        let store = ConvStore::memory("/tmp/p").unwrap();
        let id = store
            .create(NewSession {
                agent: Some("alice".into()),
                title: Some("Ops War Room".into()),
                ..Default::default()
            })
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        store.add_participant(&id, "agent", "bob", Some("Bob"), Some("bob"), None, "You are Bob.", None, None);
        let roster = store.participants(&id);
        (store, id, roster)
    }

    #[test]
    fn a_participant_sees_peers_as_text_lines_never_as_roles() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        let bob = roster.iter().find(|p| p.handle == "bob").unwrap().clone();

        store
            .append_event(
                &id,
                NewEvent::new("context_window", json!({"role":"user","content":"deploy is failing"})),
            )
            .unwrap();
        store
            .append_event(
                &id,
                NewEvent {
                    event_type: "context_window".into(),
                    payload: json!({"role":"assistant","content":"rollback failed at 14:02"}),
                    speaker: Some(bob.id),
                    ..Default::default()
                },
            )
            .unwrap();

        let task = plan(&store, &id, &alice, 0, true, "The user directly addressed you.").unwrap();
        assert!(task.prompt.contains("You are @alice, one participant with @bob and the user."));
        assert!(task.prompt.contains("  User (user): deploy is failing"));
        assert!(task.prompt.contains("  @bob: rollback failed at 14:02"));
        assert!(task.prompt.contains("Do not return \"(pass)\""));

        // The wire format: exactly one system message, then alice's own
        // history (empty here), then ONE user message carrying the room.
        let msgs = task.messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
        // Bob's line is inside that user message, not a role of its own.
        assert!(msgs.iter().all(|m| m.role != "assistant"));
    }

    #[test]
    fn a_participant_does_not_see_its_own_messages_in_the_group_delta() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        store
            .append_event(
                &id,
                NewEvent {
                    event_type: "context_window".into(),
                    payload: json!({"role":"assistant","content":"ALICE SAID THIS"}),
                    speaker: Some(alice.id),
                    ..Default::default()
                },
            )
            .unwrap();
        let task = plan(&store, &id, &alice, 1, false, "").unwrap();
        // It belongs in her private history, not twice in the room summary.
        assert!(!task.prompt.contains("ALICE SAID THIS"));
    }

    #[test]
    fn the_watermark_is_what_stops_a_message_arriving_twice() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        let first = store
            .append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"one"})))
            .unwrap();
        let t1 = plan(&store, &id, &alice, 0, true, "x").unwrap();
        assert!(t1.prompt.contains("one"));

        // Alice has now seen up to that node.
        let seq = store.node(&first.event_id).unwrap().seq;
        store.set_participant_view(&id, alice.id, seq, 0, false, None, None);
        store
            .append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"two"})))
            .unwrap();
        let t2 = plan(&store, &id, &alice, 0, true, "x").unwrap();
        assert!(t2.prompt.contains("two"));
        assert!(!t2.prompt.contains("User (user): one"), "already seen");
    }

    #[test]
    fn own_history_is_replayed_verbatim_so_the_prompt_cache_holds() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        store.append_own_turn(&id, alice.id, "PROMPT 1", "REPLY 1", None);
        store.append_own_turn(&id, alice.id, "PROMPT 2", "REPLY 2", None);
        let task = plan(&store, &id, &alice, 1, false, "").unwrap();
        let msgs = task.messages();
        assert_eq!(msgs.len(), 6, "system + 2 pairs + the current prompt");
        let txt = |m: &agl::provider::Message| {
            m.content.as_ref().and_then(|v| v.as_str()).unwrap_or("").to_string()
        };
        assert_eq!(txt(&msgs[1]), "PROMPT 1");
        assert_eq!(txt(&msgs[2]), "REPLY 1");
        assert_eq!(txt(&msgs[3]), "PROMPT 2");
        assert_eq!(txt(&msgs[4]), "REPLY 2");
        // Stable prefix: only the last message differs between turns.
        let again = plan(&store, &id, &alice, 1, false, "").unwrap().messages();
        assert_eq!(
            msgs[..5].iter().map(|m| m.content.clone()).collect::<Vec<_>>(),
            again[..5].iter().map(|m| m.content.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_tool_result_reaches_a_teammate_as_a_short_summary() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        let bob = roster.iter().find(|p| p.handle == "bob").unwrap().clone();
        store
            .append_event(
                &id,
                NewEvent {
                    event_type: "tool_result".into(),
                    payload: json!({"name":"list_dir","text":"a\nb\nc","ok":true}),
                    speaker: Some(bob.id),
                    ..Default::default()
                },
            )
            .unwrap();
        // A tool CALL is internal and must not leak into a peer's prompt.
        store
            .append_event(
                &id,
                NewEvent {
                    event_type: "tool_call".into(),
                    payload: json!({"name":"list_dir","args":{"path":"/secret"},"id":"call_1"}),
                    speaker: Some(bob.id),
                    ..Default::default()
                },
            )
            .unwrap();
        let task = plan(&store, &id, &alice, 0, true, "x").unwrap();
        assert!(task.prompt.contains("(tool: list_dir → \"a b c\")"));
        assert!(!task.prompt.contains("/secret"));
    }

    #[test]
    fn the_delta_keeps_the_newest_lines_and_says_so_when_it_cuts() {
        let roster = vec![seat("user", 1, "user"), seat("agent", 2, "alice")];
        let me = roster[1].clone();
        // More lines than the cap, numbered so we can see which survived.
        let nodes: Vec<TreeNode> = (0..MAX_DELTA_LINES + 6)
            .map(|i| TreeNode {
                id: format!("n{i}"),
                parent_id: None,
                seq: i as i64 + 1,
                depth: 1,
                kind: "context_window".into(),
                participant_id: Some(1),
                content: Some(format!("line {i}")),
                role: Some("user".into()),
                payload: json!({"role":"user","content":format!("line {i}")}),
                meta: Default::default(),
                visible: true,
                goto: None,
                ts: "t".into(),
                duration_ms: None,
            })
            .collect();
        let refs: Vec<&TreeNode> = nodes.iter().collect();
        let (prompt, seen) = flatten("T", None, None, &me, &roster, &refs, false, "");
        assert_eq!(seen, nodes.last().unwrap().seq);
        let kept = prompt.matches("line ").count();
        assert_eq!(kept, MAX_DELTA_LINES, "capped at the newest 24");
        assert!(prompt.contains(&format!("line {}", MAX_DELTA_LINES + 5)), "newest kept");
        assert!(!prompt.contains("line 0"), "oldest dropped");
        assert!(prompt.contains("If you have nothing new to add"), "pass-allowed variant");
    }

    #[test]
    fn an_oversized_line_is_truncated_with_the_omission_sentinel() {
        let roster = vec![seat("user", 1, "user"), seat("agent", 2, "alice")];
        let me = roster[1].clone();
        let huge = "x".repeat(MAX_PROMPT_BYTES * 2);
        let node = TreeNode {
            id: "n".into(),
            parent_id: None,
            seq: 1,
            depth: 1,
            kind: "context_window".into(),
            participant_id: Some(1),
            content: Some(huge.clone()),
            role: Some("user".into()),
            payload: json!({"role":"user","content":huge}),
            meta: Default::default(),
            visible: true,
            goto: None,
            ts: "t".into(),
            duration_ms: None,
        };
        let refs = vec![&node];
        let (prompt, _) = flatten("T", None, None, &me, &roster, &refs, true, "x");
        assert!(prompt.len() <= MAX_PROMPT_BYTES, "the hard cap holds");
        // Without this the model would cite messages it never saw.
        assert!(prompt.contains("omitted to fit this turn"));
    }

    #[test]
    fn purpose_and_notes_appear_once_as_framing() {
        let roster = vec![seat("user", 1, "user"), seat("agent", 2, "alice")];
        let me = roster[1].clone();
        let (prompt, _) = flatten(
            "Room",
            Some("Coordinate incident response."),
            Some("Runbooks live in /ops.\nNever paste secrets."),
            &me,
            &roster,
            &[],
            true,
            "The user directly addressed you.",
        );
        assert!(prompt.contains("Conversation purpose (set by the user): Coordinate incident response."));
        assert!(prompt.contains("  Runbooks live in /ops."));
        assert!(prompt.contains("  Never paste secrets."));
        assert_eq!(prompt.matches("Coordinate incident response").count(), 1);
        assert!(prompt.contains("(nothing new)"));
    }

    #[test]
    fn a_task_id_is_a_pure_function_of_durable_state() {
        let a = task_id("c1", 2, "n9", 4, 0, "the prompt");
        let b = task_id("c1", 2, "n9", 4, 0, "the prompt");
        assert_eq!(a, b, "same coordinates, same id — replay is possible");
        assert_ne!(a, task_id("c1", 2, "n9", 4, 1, "the prompt"), "round matters");
        assert_ne!(a, task_id("c1", 3, "n9", 4, 0, "the prompt"), "who matters");
        assert_ne!(a, task_id("c1", 2, "n9", 4, 0, "other"), "the prompt is hashed in");
        assert!(a.starts_with("ctask:"));
    }

    #[test]
    fn a_published_message_is_capped() {
        let huge = "y".repeat(MAX_MEMBER_TEXT_BYTES * 2);
        assert!(clamp_message(&huge).len() <= MAX_MEMBER_TEXT_BYTES);
        assert_eq!(clamp_message("short"), "short");
    }

    #[test]
    fn the_routing_prompt_lists_the_roster_and_asks_for_json() {
        let roster = vec![
            seat("user", 1, "user"),
            seat("agent", 2, "alice"),
            seat("agent", 3, "bob"),
        ];
        let lead = roster[1].clone();
        let p = routing_prompt("Room", Some("ops"), &lead, &roster, "deploy is failing");
        assert!(p.contains("You are @alice, the lead"));
        assert!(p.contains("- @alice (you)"));
        assert!(p.contains("- @bob"));
        assert!(p.contains("deploy is failing"));
        assert!(p.contains("\"assign\""));
    }

    #[test]
    fn a_hidden_node_is_not_in_anyone_s_prompt() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        let e = store
            .append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"REDACTED"})))
            .unwrap();
        store.set_visibility(&e.event_id, false);
        let task = plan(&store, &id, &alice, 0, true, "x").unwrap();
        assert!(!task.prompt.contains("REDACTED"));
    }
}
