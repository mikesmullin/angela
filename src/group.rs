use sha2::{Digest, Sha256};

use crate::agent_loader::Engagement;
use crate::conv_store::{ConvStore, Participant, TreeNode};

pub const MAX_PROMPT_BYTES: usize = 128 * 1024;
pub const MAX_ROUNDS: i64 = 3;
pub const MAX_MESSAGES: usize = 10;
pub const MAX_MEMBER_TEXT_BYTES: usize = 64 * 1024;
pub const PASS: &str = "(pass)";

#[derive(Debug, Clone)]
pub struct Task {
    pub participant_id: i64,
    pub handle: String,
    pub system_prompt: String,
    pub context: Vec<agl::provider::Message>,
    pub prompt: String,
    pub round_index: i64,
    pub reply_required: bool,
    pub seen_through_seq: i64,
    pub task_id: String,
}

impl Task {
    pub fn messages(&self) -> Vec<agl::provider::Message> {
        let mut out = vec![agl::provider::Message::system(self.system_prompt.clone())];
        out.extend(self.context.iter().cloned());
        out.push(agl::provider::Message::user(self.prompt.clone()));
        out
    }
}

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

pub const MAX_CONTEXT_TURNS: usize = 40;

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
    enum Block {
        Said(Vec<String>),
        Mine(String),
    }
    let mut blocks: Vec<Block> = Vec::new();
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

    let start = blocks.len().saturating_sub(MAX_CONTEXT_TURNS);
    let blocks = &blocks[start..];

    let mut out: Vec<agl::provider::Message> = Vec::new();
    for b in blocks {
        match b {
            Block::Said(lines) => out.push(agl::provider::Message::user(lines.join("\n"))),
            Block::Mine(text) => out.push(agl::provider::Message::assistant(text.clone())),
        }
    }
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
        _ => "(continue)".to_string(),
    };
    (out, prompt)
}

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
        let _ = me;
        out.push_str(&format!("{text}\n"));
    }
    out
}

pub fn mentions(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'@' {
            i += 1;
            continue;
        }
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

#[derive(Debug, Clone, PartialEq)]
pub enum Invite {
    Mentioned,
    Always,
    Ask,
    Skip,
}

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

fn truncate_bytes(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    if max <= 3 {
        return String::new();
    }
    let budget = max - 3;
    let mut end = 0;
    for (i, _) in s.char_indices() {
        if i > budget {
            break;
        }
        end = i;
    }
    format!("{}…", &s[..end])
}

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

    let identity = me
        .agent_name
        .as_deref()
        .and_then(|n| store.agents().get(n))
        .map(|d| d.system_prompt_with_appendage(header.get("soulAppendage").and_then(|v| v.as_str()).unwrap_or("")))
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| {
            let extra = header.get("soulAppendage").and_then(|v| v.as_str()).unwrap_or("").trim();
            if extra.is_empty() { me.system_prompt.clone() } else { format!("{}\n\n{}", me.system_prompt.trim(), extra) }
        });
    let name = if me.display_name.trim().is_empty() {
        me.handle.clone()
    } else {
        me.display_name.clone()
    };

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
    system.push_str(&crate::environment::section(
        crate::environment::now_epoch(),
        &store.setting("location"),
        &store.setting("user__full_name"),
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

pub fn strip_echoed_prefix(reply: &str, handle: &str) -> String {
    let t = reply.trim_start();
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
    let tag = format!("[{handle}] ");
    let out = after_clock.strip_prefix(tag.as_str()).unwrap_or(after_clock);
    if std::ptr::eq(out, reply) {
        reply.to_string()
    } else {
        out.to_string()
    }
}

pub fn is_pass(reply: &str) -> bool {
    let t = reply.trim();
    t.eq_ignore_ascii_case(PASS) || t.eq_ignore_ascii_case("pass")
}

pub fn clamp_message(text: &str) -> String {
    truncate_bytes(text, MAX_MEMBER_TEXT_BYTES)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Settled {
    SilentRound,
    MaxRounds,
    MaxMessages,
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
            Settled::Stopped => "stopped",
            _ => "bounded",
        }
    }
}

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
