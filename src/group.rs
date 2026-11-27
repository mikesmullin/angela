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

use crate::agent_loader::Engagement;
use crate::conv_store::{ConvStore, Participant, TreeNode};

/// Hard cap on the flattened prompt. Beyond this the prompt is a liability,
/// not context.
pub const MAX_PROMPT_BYTES: usize = 128 * 1024;
/// Rounds before a discussion is declared bounded.
pub const MAX_ROUNDS: i64 = 3;
/// Messages in one discussion before it is declared bounded.
pub const MAX_MESSAGES: usize = 10;
/// A single participant's message, truncated at this size when published.
pub const MAX_MEMBER_TEXT_BYTES: usize = 64 * 1024;
/// The marker a participant returns to stay silent.
pub const PASS: &str = "(pass)";

/// Everything one participant's turn needs, derived purely from durable
/// state — so a restart mid-round rebuilds exactly this.
#[derive(Debug, Clone)]
pub struct Task {
    pub participant_id: i64,
    pub handle: String,
    pub system_prompt: String,
    /// The conversation as this participant sees it, up to but excluding the
    /// turn it is being asked to take: its own messages as `assistant`,
    /// everyone else's coalesced into `user` turns of `[sender] text` lines.
    pub context: Vec<agl::provider::Message>,
    /// The final user turn — what it is replying to.
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
        out.extend(self.context.iter().cloned());
        out.push(agl::provider::Message::user(self.prompt.clone()));
        out
    }
}

/// The standing part of a participant's system prompt.
///
/// Addressed to the agent, and ending with the line that explains the
/// `[sender]` convention the conversation turns use — so the format is
/// introduced once, where standing instructions belong, rather than
/// re-explained in every message.
///
/// Shared with the engagement decider (`relevance`), deliberately: an agent
/// should be deciding whether to speak from the same picture it would speak
/// from.
pub fn preamble(name: &str, identity: &str) -> String {
    let mut out = format!("Your name is {name}.\n");
    let identity = identity.trim();
    if !identity.is_empty() {
        out.push('\n');
        out.push_str(identity);
        out.push('\n');
    }
    out.push_str("\n## Conversation History\n\n");
    out.push_str(
        "**Session type:** Multi-user thread — messages are prefixed with \
         [sender name]. Multiple users may participate.\n",
    );
    out
}

/// How many conversation turns one prompt carries, newest kept.
pub const MAX_CONTEXT_TURNS: usize = 40;

/// Project the room onto one participant's view of it.
///
/// Its own messages become `assistant` turns; everything else — the human
/// and the other agents alike — is coalesced into `user` turns of
/// `[sender] text` lines. That is what gives the agent the *flow* of the
/// conversation rather than only the message that triggered it, and it is
/// the shape a chat model is trained on: its own words as its own turns.
///
/// Returns the turns before the one being taken, and the final user turn.
pub fn project(
    path: &[TreeNode],
    me_id: i64,
    roster: &[Participant],
    user_label: &str,
) -> (Vec<agl::provider::Message>, String) {
    let me_handle = roster
        .iter()
        .find(|p| p.id == me_id)
        .map(|p| p.handle.clone())
        .unwrap_or_else(|| "assistant".into());
    // Build alternating blocks first; pairing into messages is then trivial
    // and the coalescing rule lives in one place.
    enum Block {
        Said(Vec<String>),
        Mine(String),
    }
    let mut blocks: Vec<Block> = Vec::new();
    // The day of the last header written, not of the last message seen: an
    // agent's own turn can cross midnight, and it carries no timestamp (it
    // is the model's own words — stamping them teaches it to emit stamps).
    // Tracking headers instead means the next line that *does* carry a
    // clock still announces the new day.
    let mut headed: Option<(i32, i32)> = None;
    for n in path {
        if n.kind != "context_window" || !n.visible {
            continue;
        }
        if !matches!(n.role.as_deref(), Some("user") | Some("assistant")) {
            continue;
        }
        let Some(text) = n.content.clone() else { continue };
        if n.participant_id == Some(me_id) {
            // Its own words carry the same clock and label as everyone
            // else's. The alternative — leaving them bare — makes the
            // agent's own lines the only ones in the transcript with no
            // time on them, which is exactly the thing the clocks were
            // added to show. `strip_echoed_prefix` below handles the cost:
            // a model that mimics the form has it removed before publishing.
            blocks.push(Block::Mine(crate::relevance::history_line(
                &n.ts,
                &me_handle,
                &text,
            )));
            continue;
        }
        let sender = match n.role.as_deref() {
            Some("user") => user_label.to_string(),
            _ => n
                .participant_id
                .and_then(|id| roster.iter().find(|p| p.id == id))
                .map(|p| p.handle.clone())
                .unwrap_or_else(|| "assistant".into()),
        };
        let line = crate::relevance::history_line(&n.ts, &sender, &text);
        // A header whenever the calendar day changes, so a run of messages
        // reads as a day rather than as an undated list.
        let mut pending: Vec<String> = Vec::new();
        if let Some(e) = crate::environment::epoch_from_iso(&n.ts) {
            let key = crate::environment::day_key(e);
            if headed != Some(key) {
                if let Some(h) = crate::relevance::day_header(&n.ts) {
                    pending.push(h);
                }
                headed = Some(key);
            }
        }
        pending.push(line);
        match blocks.last_mut() {
            Some(Block::Said(lines)) => lines.extend(pending),
            _ => blocks.push(Block::Said(pending)),
        }
    }

    // Keep the newest turns. An unbounded history is a prompt that grows
    // without limit, and the recent flow is what the agent is replying to.
    let start = blocks.len().saturating_sub(MAX_CONTEXT_TURNS);
    let blocks = &blocks[start..];

    let mut out: Vec<agl::provider::Message> = Vec::new();
    for b in blocks {
        match b {
            Block::Said(lines) => out.push(agl::provider::Message::user(lines.join("\n"))),
            Block::Mine(text) => out.push(agl::provider::Message::assistant(text.clone())),
        }
    }
    // The last user turn is what this turn answers; anything after it would
    // be this agent's own last word, which is not something to reply to.
    let prompt = match out.last() {
        Some(m) if m.role == "user" => {
            let text = m
                .content
                .as_ref()
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            out.pop();
            text
        }
        // Nothing new since this agent last spoke. It should not have been
        // asked, but a turn with no prompt is worse than a nudge.
        _ => "(continue)".to_string(),
    };
    (out, prompt)
}

/// The conversation turns as one readable block, for the transcript's
/// "Prompt sent" card.
///
/// The card exists to answer "what did the model actually receive?", and
/// after the projection that is no longer one string — it is a list of
/// turns. Showing only the last one would under-report by exactly the
/// context the projection was added to supply.
pub fn render_wire(msgs: &[agl::provider::Message], me: &str) -> String {
    let mut out = String::new();
    for m in msgs {
        if m.role == "system" {
            continue;
        }
        let text = m
            .content
            .as_ref()
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        // Verbatim, both roles. Every turn already carries its own clock
        // and `[sender]` label, so the card is a literal rendering of what
        // was sent rather than a prettier version of it — which is the only
        // way it can answer "what did the model actually receive?".
        let _ = me;
        out.push_str(&format!("{text}\n"));
    }
    out
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

/// What a seated agent's engagement decides, before anyone is asked.
#[derive(Debug, Clone, PartialEq)]
pub enum Invite {
    /// Named outright — it speaks whatever its engagement says.
    Mentioned,
    /// `engaged`: it speaks on every message.
    Always,
    /// `relevant`: a System One call decides. The caller makes it, because
    /// this module stays free of I/O.
    Ask,
    /// `invoked` and not mentioned: it stays out.
    Skip,
}

/// Round-0 invitations, one per active agent, from its engagement.
///
/// A mention always wins. Naming an agent is an unambiguous request for it,
/// and making a `relevant` agent pass a relevance check before answering a
/// direct question would be a bug the user cannot work around. A sole active
/// invoked agent is implicitly addressed by the user's message.
pub fn invitations(
    roster: &[Participant],
    engagement_of: &dyn Fn(&Participant) -> Engagement,
    trigger_text: &str,
) -> Vec<(i64, Invite)> {
    let named = mentions(trigger_text);
    let lone_agent = roster.iter().filter(|p| p.is_agent() && p.active).count() == 1;
    roster
        .iter()
        .filter(|p| p.is_agent() && p.active)
        .map(|p| {
            let invite = if named.contains(&p.handle) {
                Invite::Mentioned
            } else {
                match engagement_of(p) {
                    Engagement::Engaged => Invite::Always,
                    Engagement::Relevant => Invite::Ask,
                    Engagement::Invoked if lone_agent => Invite::Mentioned,
                    Engagement::Invoked => Invite::Skip,
                }
            };
            (p.id, invite)
        })
        .collect()
}

/// Who speaks in this round.
///
/// Round 0: whoever the triggering message @-mentioned, or — when it
/// mentioned nobody — every active agent. Later rounds: only agents a *reply
/// in this round* pulled in, and only if they have not spoken in it yet.
/// That second rule is what makes a discussion terminate on its own: nobody
/// new mentioned means nobody new speaks.
///
/// Round 0 here ignores engagement, and the caller uses [`invitations`]
/// instead when the roster has definitions to consult. This remains the
/// behaviour for a conversation whose agents have no definition at all.
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
    let header = store.effective_session(conv)?.get("effective")?.clone();
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

    // The definition's current identity, not the copy taken when the agent
    // was seated. Editing an agent has to change how it behaves in the
    // conversations it is already in — the same rule `engagement` follows —
    // and a stale snapshot would make the editor appear to do nothing. The
    // stored prompt is the fallback for a seat with no definition behind it
    // (one added by hand, or whose agent was deleted).
    let identity = me
        .agent_name
        .as_deref()
        .and_then(|n| store.agents().get(n))
        .map(|d| d.system_prompt())
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| me.system_prompt.clone());
    let name = if me.display_name.trim().is_empty() {
        me.handle.clone()
    } else {
        me.display_name.clone()
    };

    // The conversation as this participant sees it. Not a watermark delta:
    // an agent given only what is new since it last looked has no sense of
    // how the conversation got here, which is most of what tells it whether
    // its answer still fits.
    let (context, prompt) = project(&path, me.id, &roster, &store.user_label());

    let mut system = preamble(&name, &identity);
    if let Some(p) = purpose.filter(|s| !s.trim().is_empty()) {
        system.push_str(&format!("\n**Purpose:** {p}\n"));
    }
    if let Some(n) = notes.filter(|s| !s.trim().is_empty()) {
        system.push_str("\n**Notes:**\n");
        for line in n.lines() {
            system.push_str(&format!("{line}\n"));
        }
    }
    system.push_str(&rules_section(reply_required, how_you_got_here, title.as_str()));
    // Last, because it is the one part that changes between turns: where
    // the agent is and what time it is, neither of which a model can know
    // and both of which it will otherwise guess at confidently.
    system.push_str(&crate::environment::section(
        crate::environment::now_epoch(),
        &store.setting("location"),
        // The raw setting, not `user_label()`: an unset name should leave
        // the line out, not announce that the user is called "user".
        &store.setting("user__full_name"),
        // A sandbox is the session's, so it is part of "where the agent
        // is" for every participant — and `header` is already loaded, so
        // this costs nothing.
        header.get("sandboxId").and_then(|v| v.as_str()).unwrap_or_default(),
    ));

    let seen_through = path.last().map(|n| n.seq).unwrap_or(watermark);
    let task_id = task_id(conv, me.id, &cursor, watermark, round_index, &prompt);
    Some(Task {
        participant_id: me.id,
        handle: me.handle.clone(),
        system_prompt: system,
        context,
        prompt,
        round_index,
        reply_required,
        seen_through_seq: seen_through,
        task_id,
    })
}

/// The standing instructions, as the tail of the system prompt.
///
/// In the system prompt rather than appended to the user turn, so the
/// conversation turns stay pure conversation — which is the point of
/// projecting them in the first place. They are kept, and not folded into
/// the identity, because the `(pass)` protocol is a *protocol*: an agent
/// never told it may pass will never pass, and the round machinery reads
/// that reply.
fn rules_section(reply_required: bool, how_you_got_here: &str, title: &str) -> String {
    let mut out = String::from("\n## How to reply\n\n");
    if !title.trim().is_empty() && title != "untitled" {
        out.push_str(&format!("This conversation is called \"{title}\".\n\n"));
    }
    if reply_required {
        let how = how_you_got_here.trim();
        if !how.is_empty() {
            out.push_str(&format!("- {how}\n"));
        }
        out.push_str("- Reply with your answer, result, or concrete blocker.\n");
        out.push_str(&format!(
            "- Do not return \"{PASS}\". If you cannot proceed, say why and what you need.\n"
        ));
    } else {
        out.push_str("- Reply only when you have something new worth adding.\n");
        out.push_str(&format!(
            "- If you have nothing to add, reply with exactly \"{PASS}\".\n"
        ));
    }
    out.push_str("- Do not repeat points already made.\n");
    out.push_str(
        "- Reply as yourself, in your own voice. Do not prefix your reply with a \
         timestamp or a [sender] label — those are added for you.\n",
    );
    out
}

/// Remove a `14:05 [handle] ` prefix a model copied from its own history.
///
/// The history shows an agent its own past turns in the same stamped,
/// labelled form as everyone else's, which is what gives it a sense of time
/// passing — and also an example to imitate. The system prompt says not to;
/// this is the belt to that braces, applied before anything reads the reply
/// (`(pass)` detection included, or a passed turn arrives as a message).
pub fn strip_echoed_prefix(reply: &str, handle: &str) -> String {
    let t = reply.trim_start();
    // `HH:MM `
    let after_clock = match t.as_bytes() {
        [a, b, b':', c, d, b' ', ..]
            if a.is_ascii_digit()
                && b.is_ascii_digit()
                && c.is_ascii_digit()
                && d.is_ascii_digit() =>
        {
            &t[6..]
        }
        _ => t,
    };
    // `[handle] `, only this agent's own — another participant's label in a
    // reply is the model quoting someone, which is its business.
    let tag = format!("[{handle}] ");
    let out = after_clock.strip_prefix(tag.as_str()).unwrap_or(after_clock);
    if std::ptr::eq(out, reply) {
        reply.to_string()
    } else {
        out.to_string()
    }
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
    /// The user pressed stop.
    Stopped,
}

impl Settled {
    pub fn reason(&self) -> &'static str {
        match self {
            Settled::SilentRound => "silent_round",
            Settled::MaxRounds => "max_rounds",
            Settled::MaxMessages => "max_messages",
            Settled::Stopped => "stopped",
        }
    }
    pub fn status(&self) -> &'static str {
        match self {
            Settled::SilentRound => "settled",
            // Its own status, not "bounded": the discussion did not run out
            // of rope, somebody took it away. A restarted server reads this
            // to decide whether a round is still in flight, and "stopped"
            // answers that as clearly as the other two.
            Settled::Stopped => "stopped",
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
            engagement: None,
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
    fn engagement_decides_who_is_invited_in_round_zero() {
        let roster = vec![
            seat("user", 1, "user"),
            seat("agent", 2, "quiet"),   // invoked
            seat("agent", 3, "maybe"),   // relevant
            seat("agent", 4, "eager"),   // engaged
        ];
        let eng = |p: &Participant| match p.handle.as_str() {
            "maybe" => Engagement::Relevant,
            "eager" => Engagement::Engaged,
            _ => Engagement::Invoked,
        };
        let got = invitations(&roster, &eng, "anyone around?");
        assert_eq!(
            got,
            vec![
                (2, Invite::Skip),   // invoked, not named: stays out
                (3, Invite::Ask),    // relevant: the decider is asked
                (4, Invite::Always), // engaged: always speaks
            ]
        );
    }

    #[test]
    fn a_mention_overrides_every_engagement() {
        let roster = vec![
            seat("user", 1, "user"),
            seat("agent", 2, "quiet"),
            seat("agent", 3, "maybe"),
        ];
        let eng = |p: &Participant| match p.handle.as_str() {
            "maybe" => Engagement::Relevant,
            _ => Engagement::Invoked,
        };
        let got = invitations(&roster, &eng, "@quiet @maybe thoughts?");
        // Naming an agent is an unambiguous request for it; making a
        // `relevant` one pass a check first would be unfixable from the UI.
        assert_eq!(got, vec![(2, Invite::Mentioned), (3, Invite::Mentioned)]);
    }

    #[test]
    fn a_single_invoked_agent_is_addressed_without_a_mention() {
        let mut inactive = seat("agent", 3, "former");
        inactive.active = false;
        let roster = vec![seat("user", 1, "user"), seat("agent", 2, "alice"), inactive];
        let invoked = |_: &Participant| Engagement::Invoked;
        assert_eq!(invitations(&roster, &invoked, "hello"), vec![(2, Invite::Mentioned)]);
        assert_eq!(invitations(&roster, &invoked, "@alice hello"), vec![(2, Invite::Mentioned)]);
        assert_eq!(invitations(&roster, &invoked, "@someone hello"), vec![(2, Invite::Mentioned)]);
        assert_eq!(invitations(&roster, &|_| Engagement::Relevant, "hello"), vec![(2, Invite::Ask)]);
        assert_eq!(invitations(&roster, &|_| Engagement::Engaged, "hello"), vec![(2, Invite::Always)]);
    }

    #[test]
    fn an_inactive_agent_is_never_invited() {
        let mut roster = vec![seat("user", 1, "user"), seat("agent", 2, "gone")];
        roster[1].active = false;
        let eng = |_: &Participant| Engagement::Engaged;
        assert!(invitations(&roster, &eng, "@gone hello").is_empty());
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
        // Agents are invited, never assumed — a new conversation is the
        // user alone, so the room's two agents are seated here.
        store.add_participant(&id, "agent", "alice", Some("Alice"), Some("alice"), None, "You are Alice.", None, None);
        store.add_participant(&id, "agent", "bob", Some("Bob"), Some("bob"), None, "You are Bob.", None, None);
        let roster = store.participants(&id);
        (store, id, roster)
    }

    /// Everything `plan` produced, as `(role, text)`.
    fn wire(t: &Task) -> Vec<(String, String)> {
        t.messages()
            .iter()
            .map(|m| {
                (
                    m.role.clone(),
                    m.content
                        .as_ref()
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                )
            })
            .collect()
    }

    /// The transcript with its clocks and day headers removed, so an
    /// assertion about *who said what* is not also an assertion about when
    /// the test ran.
    fn undated(text: &str) -> String {
        text.lines()
            .filter(|l| !l.starts_with("--- "))
            .map(|l| {
                // `14:05 [who] …` -> `[who] …`
                match l.split_once(' ') {
                    Some((head, rest))
                        if head.len() == 5 && head.as_bytes()[2] == b':' => rest.to_string(),
                    _ => l.to_string(),
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn say(store: &ConvStore, id: &str, role: &str, who: Option<i64>, text: &str) {
        store
            .append_event(
                id,
                NewEvent {
                    event_type: "context_window".into(),
                    payload: json!({ "role": role, "content": text }),
                    speaker: who,
                    ..Default::default()
                },
            )
            .unwrap();
    }

    #[test]
    fn the_agent_gets_the_flow_of_the_conversation_not_just_the_last_line() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        let bob = roster.iter().find(|p| p.handle == "bob").unwrap().clone();

        say(&store, &id, "user", None, "what time is it");
        say(&store, &id, "assistant", Some(alice.id), "about three");
        say(&store, &id, "user", None, "i might go to the market");
        say(&store, &id, "assistant", Some(bob.id), "bring a bag");
        say(&store, &id, "user", None, "maybe some fish");

        let task = plan(&store, &id, &alice, 0, true, "x").unwrap();
        let w = wire(&task);

        // Its own words are its own turns; everyone else's — the human and
        // the other agent alike — are `[sender]`-prefixed user turns,
        // consecutive ones coalesced into one.
        assert_eq!(
            w.iter().map(|(r, _)| r.as_str()).collect::<Vec<_>>(),
            vec!["system", "user", "assistant", "user"]
        );
        assert_eq!(undated(&w[1].1), "[user] what time is it");
        assert_eq!(undated(&w[2].1), "[alice] about three");
        assert_eq!(
            undated(&w[3].1),
            "[user] i might go to the market\n[bob] bring a bag\n[user] maybe some fish"
        );
        // The turn it is answering is the last user block, not only the
        // final message — the context around it is the point.
        assert_eq!(task.prompt, w[3].1);
    }

    #[test]
    fn the_wire_renders_as_one_attributed_transcript() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        let bob = roster.iter().find(|p| p.handle == "bob").unwrap().clone();
        say(&store, &id, "user", None, "q1");
        say(&store, &id, "assistant", Some(alice.id), "mine");
        say(&store, &id, "assistant", Some(bob.id), "bob's");
        say(&store, &id, "user", None, "q2");

        let task = plan(&store, &id, &alice, 0, true, "x").unwrap();
        let wire = render_wire(&task.messages(), &task.handle);

        // Every line attributed, and no `--- role ---` dividers: those name
        // the wire's shape rather than the speaker, and leave the agent's
        // own lines as the only anonymous ones.
        assert_eq!(undated(&wire), "[user] q1\n[alice] mine\n[bob] bob's\n[user] q2");
        assert!(!wire.contains("--- user ---") && !wire.contains("--- assistant ---"));
        // The system prompt is not part of the transcript.
        assert!(!wire.contains("Your name is"));
    }

    #[test]
    fn the_system_prompt_ends_with_the_environment() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        say(&store, &id, "user", None, "hi");
        let sys = plan(&store, &id, &alice, 0, true, "x").unwrap().system_prompt;
        assert!(sys.contains("## Environment"), "{sys}");
        assert!(sys.contains("You are on "));
        assert!(sys.contains("The current time is "));
        // Last: it is the one part that changes between turns, so anything
        // stable that followed it would be re-sent for no reason.
        let env = sys.find("## Environment").unwrap();
        let rules = sys.find("## How to reply").unwrap();
        assert!(rules < env, "the environment is the suffix");
    }

    /// Append a message with an explicit timestamp.
    fn say_at(store: &ConvStore, id: &str, iso: &str, role: &str, who: Option<i64>, text: &str) {
        store
            .append_event(
                id,
                NewEvent {
                    event_type: "context_window".into(),
                    payload: json!({ "role": role, "content": text }),
                    speaker: who,
                    ts: Some(iso.to_string()),
                    ..Default::default()
                },
            )
            .unwrap();
    }

    #[test]
    fn the_transcript_is_grouped_by_day_and_stamped_by_minute() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        let bob = roster.iter().find(|p| p.handle == "bob").unwrap().clone();
        // Two local days, chosen well inside each so no zone shifts them.
        say_at(&store, &id, "2026-11-13T19:00:00.000Z", "user", None, "hello everyone");
        say_at(&store, &id, "2026-11-13T19:01:00.000Z", "assistant", Some(bob.id), "hi, Bob!");
        say_at(&store, &id, "2026-11-14T19:02:00.000Z", "user", None, "morning");

        let prompt = plan(&store, &id, &alice, 0, true, "x").unwrap().prompt;
        let lines: Vec<&str> = prompt.lines().collect();

        // A header opens each day; every message carries a 24-hour clock.
        assert!(lines[0].starts_with("--- ") && lines[0].ends_with(" ---"), "{lines:?}");
        assert!(lines[1].contains("[user] hello everyone"));
        assert!(lines[2].contains("[bob] hi, Bob!"));
        let headers: Vec<&&str> = lines.iter().filter(|l| l.starts_with("--- ")).collect();
        assert_eq!(headers.len(), 2, "one per day: {lines:?}");
        assert_ne!(headers[0], headers[1]);
        // Same day, same header — not one per message.
        assert_eq!(
            lines.iter().filter(|l| l.contains("[")).count(),
            3,
            "{lines:?}"
        );
        for l in lines.iter().filter(|l| !l.starts_with("--- ")) {
            let clock = &l[..5];
            assert_eq!(&clock[2..3], ":", "expected HH:MM, got {l:?}");
            assert!(clock[..2].parse::<u32>().unwrap() < 24, "{l:?}");
        }
    }

    #[test]
    fn an_unreadable_timestamp_loses_its_clock_not_its_line() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        say_at(&store, &id, "whenever", "user", None, "still here");
        let prompt = plan(&store, &id, &alice, 0, true, "x").unwrap().prompt;
        assert_eq!(prompt, "[user] still here");
    }

    #[test]
    fn the_human_is_named_by_the_configured_setting() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        say(&store, &id, "user", None, "hello");

        // Unset: a bracketed empty label would read as a participant with
        // no name, so it falls back rather than going blank.
        assert_eq!(undated(&plan(&store, &id, &alice, 0, true, "x").unwrap().prompt), "[user] hello");

        let mut patch = serde_json::Map::new();
        patch.insert("user__full_name".into(), json!("Bob"));
        store.set_settings(&patch);
        assert_eq!(undated(&plan(&store, &id, &alice, 0, true, "x").unwrap().prompt), "[Bob] hello");

        // Only the human is renamed; an agent is still its handle.
        let bob = roster.iter().find(|p| p.handle == "bob").unwrap().clone();
        say(&store, &id, "assistant", Some(bob.id), "hi back");
        let p = plan(&store, &id, &alice, 0, true, "x").unwrap().prompt;
        assert_eq!(undated(&p), "[Bob] hello\n[bob] hi back");
    }

    #[test]
    fn the_system_prompt_names_the_agent_and_introduces_the_format() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        say(&store, &id, "user", None, "hello");
        let task = plan(&store, &id, &alice, 0, true, "The user addressed you.").unwrap();
        let sys = &task.system_prompt;

        assert!(sys.starts_with("Your name is Alice.\n"), "{sys}");
        assert!(sys.contains("You are Alice."), "the identity: {sys}");
        assert!(sys.contains("## Conversation History"));
        assert!(sys.contains("**Session type:** Multi-user thread"));
        // The `(pass)` protocol is a protocol: an agent never told it may
        // pass will never pass, and the round machinery reads that reply.
        assert!(sys.contains("Do not return \"(pass)\""), "{sys}");
        assert!(sys.contains("The user addressed you."));
        // …and the other way round in a later round.
        let later = plan(&store, &id, &alice, 1, false, "").unwrap().system_prompt;
        assert!(later.contains("reply with exactly \"(pass)\""), "{later}");
        assert!(!later.contains("Do not return"));
    }

    #[test]
    fn an_agents_own_messages_come_back_as_its_own_turns() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        say(&store, &id, "user", None, "q1");
        say(&store, &id, "assistant", Some(alice.id), "ALICE SAID THIS");
        say(&store, &id, "user", None, "q2");

        let w = wire(&plan(&store, &id, &alice, 0, true, "x").unwrap());
        // Its own words are its own `assistant` turns — the shape a chat
        // model is trained on — rather than `[alice] …` lines inside
        // somebody else's user block.
        let own: Vec<&(String, String)> = w.iter().filter(|(r, _)| r == "assistant").collect();
        assert_eq!(own.len(), 1, "{w:?}");
        assert!(own[0].1.ends_with("ALICE SAID THIS"), "{:?}", own[0].1);
        // Stamped and labelled like every other line, so its own turns are
        // not the only ones in the transcript with no time on them.
        assert_eq!(undated(&own[0].1), "[alice] ALICE SAID THIS");
        assert!(own[0].1.contains("[alice] "));
        // …and the user blocks still do not carry it.
        assert!(!w.iter().any(|(r, t)| r == "user" && t.contains("[alice]")));
    }

    #[test]
    fn a_copied_prefix_is_stripped_from_the_reply() {
        // Showing an agent its own turns stamped and labelled also shows it
        // a form to imitate. The system prompt says not to; this is the
        // belt to that braces.
        assert_eq!(strip_echoed_prefix("17:43 [fish] hello", "fish"), "hello");
        assert_eq!(strip_echoed_prefix("17:43 hello", "fish"), "hello");
        assert_eq!(strip_echoed_prefix("[fish] hello", "fish"), "hello");
        assert_eq!(strip_echoed_prefix("  17:43 [fish] hello", "fish"), "hello");
        // A pass has to survive the strip, or it is published as a message
        // instead of being silence.
        assert!(is_pass(&strip_echoed_prefix("17:43 [fish] (pass)", "fish")));
        // Left alone: an ordinary reply, a time that is part of the
        // sentence, and another participant's label (the model quoting
        // someone, which is its business).
        assert_eq!(strip_echoed_prefix("hello", "fish"), "hello");
        assert_eq!(strip_echoed_prefix("17:43 is when we meet", "fish"), "is when we meet");
        assert_eq!(strip_echoed_prefix("[joker] said that", "fish"), "[joker] said that");
        assert_eq!(strip_echoed_prefix("1:4 [fish] x", "fish"), "1:4 [fish] x");
    }

    #[test]
    fn purpose_and_notes_ride_in_the_system_prompt() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        let mut patch = serde_json::Map::new();
        patch.insert("purpose".into(), json!("Coordinate incident response."));
        patch.insert("notes".into(), json!("Runbooks live in /ops.\nNever paste secrets."));
        store.update_meta(&id, patch);
        say(&store, &id, "user", None, "hi");

        let sys = plan(&store, &id, &alice, 0, true, "x").unwrap().system_prompt;
        assert!(sys.contains("Coordinate incident response."));
        assert!(sys.contains("Runbooks live in /ops."));
        assert!(sys.contains("Never paste secrets."));
        // Standing framing belongs in the system prompt, said once.
        assert_eq!(sys.matches("Coordinate incident response").count(), 1);
    }

    #[test]
    fn only_the_recent_turns_are_carried() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        // Alternating, so each message is its own block.
        for i in 0..(MAX_CONTEXT_TURNS + 10) {
            say(&store, &id, "user", None, &format!("u{i}"));
            say(&store, &id, "assistant", Some(alice.id), &format!("a{i}"));
        }
        say(&store, &id, "user", None, "newest");

        let task = plan(&store, &id, &alice, 0, true, "x").unwrap();
        // An unbounded history is a prompt that grows without limit.
        assert_eq!(task.context.len(), MAX_CONTEXT_TURNS - 1);
        assert_eq!(undated(&task.prompt), "[user] newest");
        let w = wire(&task);
        assert!(!w.iter().any(|(_, t)| t.contains("u0")), "oldest dropped");
    }

    #[test]
    fn a_seat_follows_its_agents_current_definition() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        say(&store, &id, "user", None, "hi");
        assert!(plan(&store, &id, &alice, 0, true, "x")
            .unwrap()
            .system_prompt
            .contains("You are Alice."));

        // Editing an agent must change how it behaves in conversations it
        // is already in.
        let mut def = crate::agent_loader::AgentDef {
            name: "alice".into(),
            soul: "You are Alice, rewritten.".into(),
            ..Default::default()
        };
        store.agents().save(&def, None).unwrap();
        def.goals = "Ship it.".into();
        store.agents().save(&def, Some("alice")).unwrap();

        let sys = plan(&store, &id, &alice, 0, true, "x").unwrap().system_prompt;
        assert!(sys.contains("You are Alice, rewritten."), "{sys}");
        assert!(sys.contains("Ship it."), "goals reach the prompt: {sys}");
    }

    #[test]
    fn a_seat_with_no_definition_keeps_the_prompt_it_was_given() {
        let (store, id, _) = room();
        let p = store
            .add_participant(&id, "agent", "adhoc", Some("Adhoc"), None, None, "You are Adhoc.", None, None)
            .unwrap();
        say(&store, &id, "user", None, "hi");
        let sys = plan(&store, &id, &p, 0, true, "x").unwrap().system_prompt;
        assert!(sys.starts_with("Your name is Adhoc.\n"));
        assert!(sys.contains("You are Adhoc."));
    }

    #[test]
    fn a_hidden_node_is_not_in_anyone_s_prompt() {
        let (store, id, roster) = room();
        let alice = roster.iter().find(|p| p.handle == "alice").unwrap().clone();
        let e = store
            .append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"REDACTED"})))
            .unwrap();
        store.set_visibility(&e.event_id, false);
        say(&store, &id, "user", None, "visible one");
        let task = plan(&store, &id, &alice, 0, true, "x").unwrap();
        let all = wire(&task)
            .iter()
            .map(|(_, t)| t.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!all.contains("REDACTED"));
        assert!(all.contains("visible one"));
    }

    #[test]
    fn the_decider_and_the_turn_share_one_preamble() {
        let p = preamble("Fish", "You like Cajun food.");
        assert!(p.starts_with("Your name is Fish.\n"));
        assert!(p.contains("You like Cajun food."));
        assert!(p.contains("## Conversation History"));
        assert!(p.contains("**Session type:** Multi-user thread"));
        // An agent should decide whether to speak from the picture it would
        // speak from, so this is the one place the framing is written.
        assert!(crate::relevance::build_state("Fish", "You like Cajun food.", &[])
            .starts_with(&p));
        // No stray blank section when there is no identity.
        assert!(!preamble("x", "").contains("\n\n\n"));
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


}
