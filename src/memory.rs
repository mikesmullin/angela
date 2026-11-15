//! The memory tree: how a conversation keeps growing without ever
//! outgrowing the context window, and without throwing anything away.
//!
//! Over the messages grows a **binary tree of one-line summaries**. Node
//! `(0, i)` is message `i` in at most [`NODE`] bytes; node `(l, i)` is the
//! merge of `(l-1, 2i)` and `(l-1, 2i+1)`. A node whose source already fits
//! is its own summary and costs nothing; anything larger is compacted by a
//! model call.
//!
//! What you read at any moment is a **view**: the topmost nodes that cover
//! the whole conversation, oldest first. The newest messages sit in it as
//! leaves — full detail — and as they age they are replaced by their parent,
//! so the same budget buys an unbounded history at falling resolution. This
//! is downsampling, and the lossiness is the point: fidelity is traded for
//! retention.
//!
//! The part that makes the trade acceptable to a human *and* to a model is
//! [`zoom`]: any summary line can be opened into the two lines that were
//! merged to make it, and those into theirs, until you reach the original
//! message. A collapsed thread is not a lossy archive of the past — it is an
//! index into it. So "compaction" here never means deletion; it means the
//! conversation gains levels.
//!
//! ```text
//! view over 11 messages, budget exceeded:
//!
//!   (3,0) ─ "early: set up the repo, picked SQLite, agreed on the schema"   ← 8 messages
//!     └ zoom → (2,0) "set up the repo, picked SQLite" + (2,1) "agreed on the schema"
//!                └ zoom → (1,0) … + (1,1) …
//!                           └ zoom → (0,0) the original message
//!   (1,4) ─ "wrote the migration and its verifier"                          ← 2 messages
//!   (0,10) ─ the newest message, verbatim                                   ← 1 message
//! ```
//!
//! Coordinates follow an implicit heap: `slot(l, i) = (2i+1)·2^l - 1`, so a
//! node's parent is always `(l+1, i >> 1)` and no parent pointer is stored.

use serde_json::{json, Value};

use crate::conv_store::ConvStore;

/// Max bytes in one summary line (~70 words).
pub const NODE: usize = 512;
/// Attempts to get a model to fit [`NODE`] before the source is clipped.
pub const TRIES: usize = 5;
/// Default byte budget for a conversation's view.
pub const HIGH: usize = 128_000;
/// Byte budget for the view a compactor itself is shown.
pub const SEEN: usize = 32_000;
/// Summary lines per prompt-cache block.
pub const BLOCK: usize = 4;
/// What a view renders for a node whose summary has not been built yet.
pub const WAIT: &str = "(not summarized yet: zoom it)";

/// A node's position in the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coord {
    pub l: u32,
    pub i: u64,
}

impl Coord {
    pub fn new(l: u32, i: u64) -> Self {
        Coord { l, i }
    }
    /// Messages this node covers: `[i·2^l, (i+1)·2^l)`.
    pub fn span(&self) -> (u64, u64) {
        let w = 1u64 << self.l;
        (self.i * w, (self.i + 1) * w)
    }
    /// How many messages it covers.
    pub fn width(&self) -> u64 {
        1u64 << self.l
    }
    /// The exclusive end of its span — "how much of the chat it knows about".
    pub fn ctx(&self) -> u64 {
        self.span().1
    }
    /// The heap position, `(2i+1)·2^l - 1`.
    pub fn slot(&self) -> u64 {
        (2 * self.i + 1) * (1u64 << self.l) - 1
    }
    pub fn parent(&self) -> Coord {
        Coord::new(self.l + 1, self.i >> 1)
    }
    pub fn children(&self) -> Option<(Coord, Coord)> {
        if self.l == 0 {
            return None;
        }
        Some((
            Coord::new(self.l - 1, self.i * 2),
            Coord::new(self.l - 1, self.i * 2 + 1),
        ))
    }
    /// The `id+n` reference a model uses to name this line.
    pub fn reference(&self) -> String {
        let (start, _) = self.span();
        format!("{}+{}", start, self.width())
    }
}

/// Resolve an `id+n` reference back to a node.
///
/// `n` must be a power of two and `id` a multiple of `n`, because those are
/// the only spans the tree actually has. Anything else is a model
/// mis-citing a line, and is better refused than silently rounded.
pub fn part(id: u64, n: u64, count: u64) -> Option<Coord> {
    if n == 0 || !n.is_power_of_two() {
        return None;
    }
    if id % n != 0 || id + n > count {
        return None;
    }
    Some(Coord::new(n.trailing_zeros(), id / n))
}

/// The top-level nodes that exactly partition `count` messages, largest
/// first.
///
/// Each one is the biggest *aligned* power-of-two block that fits at the
/// current offset — alignment matters because an unaligned block is not a
/// node of this tree, it is two.
pub fn roots(count: u64) -> Vec<Coord> {
    let mut out = Vec::new();
    let mut at = 0u64;
    let mut rem = count;
    while rem > 0 {
        let mut l = 63 - rem.leading_zeros().min(63);
        // Shrink until the block starts on its own boundary.
        while l > 0 && at % (1u64 << l) != 0 {
            l -= 1;
        }
        let c = Coord::new(l, at >> l);
        out.push(c);
        at += c.width();
        rem -= c.width();
    }
    out
}

/// One line of a view.
#[derive(Debug, Clone)]
pub struct Part {
    pub coord: Coord,
    /// The summary, or `None` when the node is not built yet.
    pub text: Option<String>,
    /// True when the model never fit the limit and the source was clipped.
    pub clipped: bool,
}

impl Part {
    pub fn bytes(&self) -> usize {
        // An unbuilt node contributes nothing to the budget: it is not
        // rendered, so charging for it would shrink the view for no reason.
        self.text.as_ref().map(|t| t.len()).unwrap_or(0)
    }

    /// `messages` must be indexed by **leaf index**, not by transcript row.
    ///
    /// The two differ: the tree's leaves are the user/assistant messages
    /// only, while a transcript also carries reasoning and tool rows. Index
    /// one with the other and every leaf after the first reasoning row shows
    /// somebody else's message — see
    /// [`crate::serve::leaf_messages`], which builds the aligned list.
    pub fn to_json(&self, messages: Option<&Vec<Value>>) -> Value {
        let (start, _) = self.coord.span();
        let mut out = json!({
            "ref": self.coord.reference(),
            "id": start,
            "n": self.coord.width(),
            "level": self.coord.l,
            "index": self.coord.i,
            "slot": self.coord.slot(),
            "built": self.text.is_some(),
            "clipped": self.clipped,
            "text": self.text.clone().unwrap_or_else(|| WAIT.to_string()),
            // A width-1 part is a real message, which the UI renders as the
            // card it always was rather than as a collapsed thread.
            "leaf": self.coord.width() == 1,
        });
        if let (Some(msgs), true) = (messages, self.coord.width() == 1) {
            if let Some(m) = msgs.get(start as usize) {
                out.as_object_mut()
                    .unwrap()
                    .insert("message".into(), m.clone());
            }
        }
        out
    }
}

/// Where a summary comes from: a model call, or a stub in tests.
///
/// Behind a trait because the tree's shape, its budget arithmetic and its
/// retry/clip behaviour are all testable without a running model — and would
/// otherwise be testable only against one.
pub trait Compactor {
    /// Return one line of at most [`NODE`] bytes, or an error.
    fn compact(&self, prompt: &str) -> Result<String, String>;
}

/// The 512-dash ruler the prompt shows the model, so "512 bytes" is a length
/// it can see rather than a number it has to imagine.
fn ruler() -> String {
    "-".repeat(NODE)
}

/// The prompt for a single message.
pub fn leaf_prompt(id: u64, source: &str) -> String {
    format!(
        "Compaction: compress message {id} into one line of at most {NODE} bytes \
         (about 70 words), the length of this ruler:\n{}\n<input>\n{source}\n</input>",
        ruler()
    )
}

/// The prompt for merging two adjacent lines.
pub fn merge_prompt(c: Coord, source: &str) -> String {
    let (a, b) = c.children().expect("merge of a leaf");
    let (start, end) = c.span();
    format!(
        "Compaction: merge lines {} and {}, adjacent, into one line of at most {NODE} bytes \
         (about 70 words), the length of this ruler:\n{}\n\
         The conversation may hold their messages, {start} to {}, in more detail: \
         take details of them from there too.\n<input>\n{source}\n</input>",
        a.reference(),
        b.reference(),
        ruler(),
        end.saturating_sub(1),
    )
}

/// The retry note: the whole line again, shortened, with the cut point shown.
fn over_note(line: &str) -> String {
    format!(
        "\n\nYour last line for this <input> was {} bytes, over the {NODE}-byte limit. \
         Write the whole line again, cutting just enough of the least valuable items \
         to fit before this cut:\n{}| <- LIMIT",
        line.len(),
        truncate_bytes(line, NODE)
    )
}

fn truncate_bytes(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = 0;
    for (i, _) in s.char_indices() {
        if i > max {
            break;
        }
        end = i;
    }
    s[..end].to_string()
}

/// Flatten a line for display: a summary is one line by definition.
fn flat(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// How many messages the tree covers.
pub fn count(store: &ConvStore, conv: &str) -> u64 {
    store.memory_leaves(conv).len() as u64
}

/// Bring the tree in line with the branch now being rendered, discarding any
/// summary that no longer describes it. Cheap when nothing moved.
pub fn sync(store: &ConvStore, conv: &str) -> u64 {
    store.sync_memory_tree(conv) as u64
}

/// A built summary, if there is one.
pub fn get(store: &ConvStore, conv: &str, c: Coord) -> Option<(String, bool)> {
    store.with(|conn| {
        conn.query_row(
            "SELECT text, clipped FROM memory_nodes
              WHERE conversation_id = ?1 AND level = ?2 AND idx = ?3",
            rusqlite::params![conv, c.l, c.i as i64],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? != 0)),
        )
        .ok()
    })
}

pub fn built(store: &ConvStore, conv: &str, c: Coord) -> bool {
    get(store, conv, c).is_some()
}

fn put(store: &ConvStore, conv: &str, c: Coord, text: &str, clipped: bool) {
    let text = flat(text);
    store.with(|conn| {
        let _ = conn.execute(
            "INSERT INTO memory_nodes (conversation_id, level, idx, slot, text, bytes, clipped, ts)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(conversation_id, level, idx) DO UPDATE SET
               text = excluded.text, bytes = excluded.bytes,
               clipped = excluded.clipped, ts = excluded.ts",
            rusqlite::params![
                conv,
                c.l,
                c.i as i64,
                c.slot() as i64,
                text,
                text.len() as i64,
                clipped as i64,
                crate::session_store::now_iso(),
            ],
        );
    });
}

/// The raw text of message `i`, as the tree's level 0 sees it.
pub fn message_line(store: &ConvStore, conv: &str, i: u64) -> Option<String> {
    let leaves = store.memory_leaves(conv);
    let node_id = leaves.get(i as usize)?;
    let n = store.node(node_id)?;
    let who = match n.role.as_deref() {
        Some("user") => "user".to_string(),
        Some("assistant") => n
            .participant_id
            .and_then(|p| store.participant(p))
            .map(|p| p.handle)
            .unwrap_or_else(|| "assistant".into()),
        other => other.unwrap_or("note").to_string(),
    };
    Some(format!("{who}: {}", n.content.unwrap_or_default()))
}

/// What a node is summarising: one message, or its two children's lines.
pub fn source(store: &ConvStore, conv: &str, c: Coord) -> Option<String> {
    if c.l == 0 {
        return message_line(store, conv, c.i);
    }
    let (a, b) = c.children()?;
    let ta = get(store, conv, a)?.0;
    let tb = get(store, conv, b)?.0;
    Some(format!("{ta}\n{tb}"))
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

/// What one build pass did.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Built {
    /// Nodes whose source already fit, stored without a model call.
    pub free: usize,
    /// Nodes a model compacted.
    pub compacted: usize,
    /// Nodes whose source was clipped because no attempt fit.
    pub clipped: usize,
    pub errors: Vec<String>,
}

impl Built {
    pub fn total(&self) -> usize {
        self.free + self.compacted + self.clipped
    }
    pub fn to_json(&self) -> Value {
        json!({
            "free": self.free,
            "compacted": self.compacted,
            "clipped": self.clipped,
            "total": self.total(),
            "errors": self.errors,
        })
    }
}

/// Every node that can be built now and is not built yet, in build order.
///
/// Bottom-up and left-to-right: a merge needs both children, so ordering by
/// level then by index means one pass builds the whole tree. A node whose
/// span runs past the end of the conversation is skipped — it is not wrong,
/// it is *not finished*, and merging a half-filled pair would have to be
/// redone when the other half arrives.
pub fn pending(store: &ConvStore, conv: &str) -> Vec<Coord> {
    let n = count(store, conv);
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let max_l = 64 - n.leading_zeros();
    for l in 0..max_l {
        let width = 1u64 << l;
        let mut i = 0u64;
        while (i + 1) * width <= n {
            let c = Coord::new(l, i);
            if !built(store, conv, c) {
                out.push(c);
            }
            i += 1;
        }
    }
    out
}

/// Build every pending node, cheapest first.
///
/// A node whose source already fits [`NODE`] is stored as-is — most short
/// messages never cost a model call at all, which is what keeps the scheme
/// affordable. Everything else goes to the compactor, retried up to
/// [`TRIES`] times with the over-limit note, keeping the **shortest** line
/// any attempt produced. If every attempt fails or comes back empty, the
/// source is clipped to [`NODE`] bytes and stored anyway: a tree that stalls
/// on one refusal would stop summarising the whole conversation.
pub fn build_all(store: &ConvStore, conv: &str, compactor: &dyn Compactor) -> Built {
    sync(store, conv);
    let mut report = Built::default();
    for c in pending(store, conv) {
        let Some(src) = source(store, conv, c) else {
            // A gap at this level: its children are not built, so neither is
            // it. A later pass picks it up.
            continue;
        };
        let one_line = flat(&src);
        if one_line.len() <= NODE {
            put(store, conv, c, &one_line, false);
            report.free += 1;
            continue;
        }
        let prompt = if c.l == 0 {
            leaf_prompt(c.i, &src)
        } else {
            merge_prompt(c, &src)
        };
        let mut best: Option<String> = None;
        let mut attempt = prompt.clone();
        let mut last_err: Option<String> = None;
        for _ in 0..TRIES {
            match compactor.compact(&attempt) {
                Ok(text) => {
                    let line = flat(text.trim());
                    if line.is_empty() {
                        break;
                    }
                    // Keep the shortest of the attempts, not the last: a
                    // retry can come back longer.
                    if best.as_ref().map(|b| line.len() < b.len()).unwrap_or(true) {
                        best = Some(line.clone());
                    }
                    if line.len() <= NODE {
                        break;
                    }
                    attempt = format!("{prompt}{}", over_note(&line));
                }
                Err(e) => {
                    last_err = Some(e);
                    break;
                }
            }
        }
        match best.filter(|b| b.len() <= NODE) {
            Some(line) => {
                put(store, conv, c, &line, false);
                report.compacted += 1;
            }
            None => {
                // The tree never stalls: clip and move on.
                put(store, conv, c, &truncate_bytes(&one_line, NODE), true);
                report.clipped += 1;
                if let Some(e) = last_err {
                    report.errors.push(format!("{}: {e}", c.reference()));
                }
            }
        }
    }
    report
}

// ---------------------------------------------------------------------------
// The view
// ---------------------------------------------------------------------------

fn load_view(store: &ConvStore, conv: &str) -> Option<(Vec<Coord>, u64)> {
    store.with(|conn| {
        conn.query_row(
            "SELECT parts, end_count FROM memory_view WHERE conversation_id = ?1",
            rusqlite::params![conv],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
        )
        .ok()
    })
    .and_then(|(text, end)| {
        let v: Vec<(u32, u64)> = serde_json::from_str(&text).ok()?;
        Some((v.into_iter().map(|(l, i)| Coord::new(l, i)).collect(), end as u64))
    })
}

fn save_view(store: &ConvStore, conv: &str, parts: &[Coord], end: u64) {
    let payload: Vec<(u32, u64)> = parts.iter().map(|c| (c.l, c.i)).collect();
    let text = serde_json::to_string(&payload).unwrap_or_else(|_| "[]".into());
    store.with(|conn| {
        let _ = conn.execute(
            "INSERT INTO memory_view (conversation_id, parts, end_count, updated_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(conversation_id) DO UPDATE SET
               parts = excluded.parts, end_count = excluded.end_count,
               updated_at = excluded.updated_at",
            rusqlite::params![conv, text, end as i64, crate::session_store::now_iso()],
        );
    });
}

/// How "due for merging" a sibling pair is: the oldest pairs score highest.
///
/// Age is measured from the pair's own end, in units of the pair's own size,
/// so a big old node and a small old node are compared fairly. This is the
/// whole "detail fades with age" rule in one expression.
fn due(a: Coord, end: u64) -> f64 {
    (end as f64 + 1.0) / (a.width() as f64) - a.i as f64
}

/// Recompute the view: extend it to the end of the conversation, then merge
/// the oldest pairs until it fits the budget.
///
/// Called after messages are appended and after a build pass. Merging
/// targets `high / 2` rather than `high` so the next few messages can be
/// appended at full detail without immediately forcing another merge —
/// hysteresis, not sloppiness.
pub fn fit(store: &ConvStore, conv: &str, high: usize) -> Vec<Part> {
    let n = sync(store, conv);
    let mut parts: Vec<Coord> = match load_view(store, conv) {
        Some((p, end)) if end <= n && !p.is_empty() => p,
        // No view yet (or the conversation was rewound): start from the
        // coarsest honest covering and let the merge loop refine it.
        _ => Vec::new(),
    };

    // --- reach: drop anything past the end, then append the new leaves ----
    parts.retain(|c| c.ctx() <= n);
    let covered = parts.iter().map(|c| c.ctx()).max().unwrap_or(0);
    for i in covered..n {
        parts.push(Coord::new(0, i));
    }
    parts.sort_by_key(|c| c.span().0);

    // --- merge the oldest due pair until the view fits --------------------
    let size = |ps: &Vec<Coord>| -> usize {
        ps.iter()
            .map(|c| get(store, conv, *c).map(|(t, _)| t.len()).unwrap_or(0))
            .sum()
    };
    let target = high / 2;
    let mut guard = 0;
    while size(&parts) > high && guard < 10_000 {
        guard += 1;
        // Candidate pairs: adjacent siblings whose parent is already built.
        // An unbuilt parent is not a merge we can make yet — the view would
        // lose the text and show nothing.
        let mut best: Option<(usize, f64)> = None;
        for k in 0..parts.len().saturating_sub(1) {
            let (a, b) = (parts[k], parts[k + 1]);
            if a.l != b.l || a.i % 2 != 0 || b.i != a.i + 1 {
                continue;
            }
            if !built(store, conv, a.parent()) {
                continue;
            }
            let d = due(a, n.saturating_sub(1));
            // Ties go to the lower index — the genuinely older pair.
            if best.map(|(_, bd)| d > bd).unwrap_or(true) {
                best = Some((k, d));
            }
        }
        let Some((k, _)) = best else { break };
        let parent = parts[k].parent();
        parts.splice(k..k + 2, [parent]);
        if size(&parts) <= target {
            break;
        }
    }

    save_view(store, conv, &parts, n);
    parts
        .into_iter()
        .map(|c| {
            let got = get(store, conv, c);
            Part {
                coord: c,
                text: got.as_ref().map(|(t, _)| t.clone()),
                clipped: got.map(|(_, cl)| cl).unwrap_or(false),
            }
        })
        .collect()
}

/// The current view without recomputing it.
pub fn view(store: &ConvStore, conv: &str) -> Vec<Part> {
    let n = sync(store, conv);
    let mut parts = match load_view(store, conv) {
        Some((p, _)) if !p.is_empty() => p,
        // Never fitted: the conversation is still flat, which is the honest
        // view of a chat nobody has compacted.
        _ => Vec::new(),
    };
    // Reach to the end without merging. A persisted view is a snapshot of a
    // shorter conversation; showing it as-is would silently hide every
    // message sent since, which is the one thing a history view must not do.
    parts.retain(|c| c.ctx() <= n);
    let covered = parts.iter().map(|c| c.ctx()).max().unwrap_or(0);
    for i in covered..n {
        parts.push(Coord::new(0, i));
    }
    parts.sort_by_key(|c| c.span().0);
    parts
        .into_iter()
        .map(|c| {
            let got = get(store, conv, c);
            Part {
                coord: c,
                text: got.as_ref().map(|(t, _)| t.clone()),
                clipped: got.map(|(_, cl)| cl).unwrap_or(false),
            }
        })
        .collect()
}

/// Render a view as the text a model request carries.
///
/// Lines are packed [`BLOCK`] to a block, and each is prefixed with its
/// `id+n` reference — which is how the model names a line it wants to
/// [`zoom`] into. Unbuilt nodes are omitted rather than shown as a
/// placeholder, because a placeholder is a sentence the model will try to
/// interpret.
pub fn render(parts: &[Part]) -> String {
    let mut out = String::from("<chat>\n");
    let mut n = 0;
    for p in parts {
        let Some(text) = &p.text else { continue };
        out.push_str(&format!("  {}|{}\n", p.coord.reference(), flat(text)));
        n += 1;
        if n % BLOCK == 0 {
            out.push('\n');
        }
    }
    out.push_str("</chat>");
    out
}

/// What opening a thread one level returns.
#[derive(Debug, Clone)]
pub enum Zoom {
    /// The two lines that were merged into this one.
    Halves(Vec<Part>),
    /// The bottom: the original message, in full.
    Message { index: u64, node_id: String },
}

/// Open one summary line into what it was made from.
///
/// One level at a time, by design: this is what lets a client lazy-load a
/// deep history, and what lets a model spend its budget only on the branch
/// it actually cares about.
pub fn zoom(store: &ConvStore, conv: &str, id: u64, n: u64) -> Result<Zoom, String> {
    let total = sync(store, conv);
    let c = part(id, n, total).ok_or_else(|| {
        // Naming the real roots back is what lets a model correct itself
        // instead of guessing again.
        let refs: Vec<String> = roots(total).iter().map(|r| r.reference()).collect();
        format!(
            "no line {id}+{n}; this conversation's top-level lines are: {}",
            refs.join(", ")
        )
    })?;
    if c.width() == 1 {
        let leaves = store.memory_leaves(conv);
        let node_id = leaves
            .get(c.i as usize)
            .cloned()
            .ok_or_else(|| format!("message {} is not in this conversation", c.i))?;
        return Ok(Zoom::Message {
            index: c.i,
            node_id,
        });
    }
    let (a, b) = c.children().expect("non-leaf has children");
    let mk = |x: Coord| {
        let got = get(store, conv, x);
        Part {
            coord: x,
            text: got.as_ref().map(|(t, _)| t.clone()),
            clipped: got.map(|(_, cl)| cl).unwrap_or(false),
        }
    };
    Ok(Zoom::Halves(vec![mk(a), mk(b)]))
}

// ---------------------------------------------------------------------------
// The real compactor
// ---------------------------------------------------------------------------

/// Compaction through `agl`, on whatever model the conversation uses.
///
/// `ANGELA_COMPACTOR` overrides it: summarising is a cheaper job than
/// answering, and pointing it at a small fast model is the usual reason to
/// care.
pub struct AglCompactor<'a> {
    pub rt: &'a tokio::runtime::Runtime,
    pub model: String,
}

impl<'a> AglCompactor<'a> {
    pub fn new(rt: &'a tokio::runtime::Runtime, fallback_model: &str) -> Self {
        let model = std::env::var("ANGELA_COMPACTOR")
            .ok()
            .filter(|m| !m.trim().is_empty())
            .unwrap_or_else(|| fallback_model.to_string());
        AglCompactor { rt, model }
    }
}

impl Compactor for AglCompactor<'_> {
    fn compact(&self, prompt: &str) -> Result<String, String> {
        // No tools and no retained history: a compaction is a pure function
        // of its <input>, and history would both cost tokens and let one
        // summary leak into the next.
        let agent = agl::agent::Agent::factory(agl::agent::Factory {
            model: Some(self.model.clone()),
            system_prompt: Some(
                "You write one-line summaries. Reply with the line only: no preamble, \
                 no quotes, no markdown, no trailing newline."
                    .into(),
            ),
            retain_history: Some(false),
            ..Default::default()
        })
        .map_err(|e| e.to_string())?;
        self.rt
            .block_on(agent.run(prompt))
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_store::{NewEvent, NewSession};
    use std::cell::RefCell;

    // --- coordinates --------------------------------------------------------

    #[test]
    fn slots_follow_the_heap_layout() {
        // slot(l, i) = (2i+1)·2^l - 1
        assert_eq!(Coord::new(0, 0).slot(), 0);
        assert_eq!(Coord::new(0, 1).slot(), 2);
        assert_eq!(Coord::new(0, 2).slot(), 4);
        assert_eq!(Coord::new(1, 0).slot(), 1);
        assert_eq!(Coord::new(1, 1).slot(), 5);
        assert_eq!(Coord::new(2, 0).slot(), 3);
        assert_eq!(Coord::new(3, 0).slot(), 7);
        // A node's parent is (l+1, i>>1), so no parent pointer is needed.
        assert_eq!(Coord::new(0, 3).parent(), Coord::new(1, 1));
        assert_eq!(Coord::new(1, 5).parent(), Coord::new(2, 2));
        assert_eq!(
            Coord::new(2, 1).children(),
            Some((Coord::new(1, 2), Coord::new(1, 3)))
        );
        assert_eq!(Coord::new(0, 7).children(), None);
    }

    #[test]
    fn a_node_covers_a_half_open_span() {
        assert_eq!(Coord::new(0, 5).span(), (5, 6));
        assert_eq!(Coord::new(3, 0).span(), (0, 8));
        assert_eq!(Coord::new(2, 3).span(), (12, 16));
        assert_eq!(Coord::new(2, 3).ctx(), 16);
        assert_eq!(Coord::new(3, 1).reference(), "8+8");
    }

    #[test]
    fn roots_partition_the_chat_into_aligned_power_of_two_blocks() {
        assert_eq!(roots(0), vec![]);
        assert_eq!(roots(1), vec![Coord::new(0, 0)]);
        assert_eq!(roots(8), vec![Coord::new(3, 0)]);
        // 11 = 8 + 2 + 1, and every block starts on its own boundary.
        let r = roots(11);
        assert_eq!(r, vec![Coord::new(3, 0), Coord::new(1, 4), Coord::new(0, 10)]);
        // They tile the range exactly, with no gaps and no overlap.
        let mut at = 0;
        for c in &r {
            assert_eq!(c.span().0, at);
            at = c.span().1;
        }
        assert_eq!(at, 11);
        for n in 1..200u64 {
            let mut at = 0;
            for c in roots(n) {
                assert_eq!(c.span().0, at, "gap at n={n}");
                assert_eq!(c.span().0 % c.width(), 0, "misaligned at n={n}");
                at = c.span().1;
            }
            assert_eq!(at, n, "roots must cover exactly n={n}");
        }
    }

    #[test]
    fn a_reference_only_resolves_to_a_span_the_tree_actually_has() {
        assert_eq!(part(0, 8, 11), Some(Coord::new(3, 0)));
        assert_eq!(part(8, 2, 11), Some(Coord::new(1, 4)));
        // Not a power of two, misaligned, or past the end: all refused
        // rather than rounded into a different line.
        assert_eq!(part(0, 3, 11), None);
        assert_eq!(part(1, 2, 11), None);
        assert_eq!(part(8, 8, 11), None);
        assert_eq!(part(0, 0, 11), None);
    }

    #[test]
    fn the_oldest_pair_is_the_most_due() {
        // Same level: the lower index (older) must score higher.
        let end = 15;
        assert!(due(Coord::new(1, 0), end) > due(Coord::new(1, 6), end));
        // Same age in message terms, different sizes: measured in units of
        // its own width, so they compare fairly rather than by raw distance.
        assert!(due(Coord::new(0, 0), end) > due(Coord::new(3, 0), end));
    }

    // --- building -----------------------------------------------------------

    /// A compactor that returns the first `n` words, and counts its calls.
    struct Fake {
        calls: RefCell<usize>,
        words: usize,
        fail: bool,
        /// Ignore the limit on the first `stubborn` attempts.
        stubborn: RefCell<usize>,
    }

    impl Fake {
        fn new(words: usize) -> Self {
            Fake {
                calls: RefCell::new(0),
                words,
                fail: false,
                stubborn: RefCell::new(0),
            }
        }
    }

    impl Compactor for Fake {
        fn compact(&self, prompt: &str) -> Result<String, String> {
            *self.calls.borrow_mut() += 1;
            if self.fail {
                return Err("no model".into());
            }
            let body = prompt.split("<input>").nth(1).unwrap_or(prompt);
            let mut s = *self.stubborn.borrow_mut();
            if s > 0 {
                *self.stubborn.borrow_mut() = s - 1;
                // Deliberately over the limit, to exercise the retry note.
                return Ok("z".repeat(NODE + 50));
            }
            s = 0;
            let _ = s;
            Ok(body
                .split_whitespace()
                .take(self.words)
                .collect::<Vec<_>>()
                .join(" "))
        }
    }

    fn chat(msgs: &[(&str, &str)]) -> (ConvStore, String) {
        let store = ConvStore::memory("/tmp/p").unwrap();
        let id = store
            .create(NewSession {
                agent: Some("default".into()),
                title: Some("t".into()),
                ..Default::default()
            })
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        for (role, text) in msgs {
            store
                .append_event(
                    &id,
                    NewEvent::new(
                        "context_window",
                        json!({ "role": role, "content": text }),
                    ),
                )
                .unwrap();
        }
        (store, id)
    }

    #[test]
    fn short_messages_cost_no_model_call_at_all() {
        let (store, id) = chat(&[("user", "hello"), ("assistant", "hi there")]);
        assert_eq!(count(&store, &id), 2);
        let fake = Fake::new(5);
        let rep = build_all(&store, &id, &fake);
        // 2 leaves + 1 merge, all under 512 bytes.
        assert_eq!(rep.total(), 3);
        assert_eq!(rep.free, 3);
        assert_eq!(rep.compacted, 0);
        assert_eq!(*fake.calls.borrow(), 0, "nothing needed compacting");
        assert_eq!(
            get(&store, &id, Coord::new(0, 0)).unwrap().0,
            "user: hello"
        );
        // The merge is its two children joined.
        let merged = get(&store, &id, Coord::new(1, 0)).unwrap().0;
        // Labelled by the speaker's handle where there is one. This chat has
        // no agent seated (a new conversation is the user alone), so the
        // generic role is the honest label.
        assert_eq!(merged, "user: hello assistant: hi there");
    }

    #[test]
    fn a_long_message_is_compacted_and_a_pair_merges_above_it() {
        let long = "word ".repeat(400); // well over NODE
        let (store, id) = chat(&[("user", long.trim()), ("assistant", "ok")]);
        let fake = Fake::new(4);
        let rep = build_all(&store, &id, &fake);
        assert_eq!(rep.compacted, 1, "only the long leaf needed a call");
        assert!(rep.free >= 2);
        let leaf = get(&store, &id, Coord::new(0, 0)).unwrap().0;
        assert!(leaf.len() <= NODE);
        // The merge above it exists and covers both messages.
        assert!(built(&store, &id, Coord::new(1, 0)));
        assert_eq!(Coord::new(1, 0).span(), (0, 2));
    }

    #[test]
    fn a_stubborn_model_is_retried_then_the_source_is_clipped() {
        let long = "alpha ".repeat(300);
        let (store, id) = chat(&[("user", long.trim())]);
        let fake = Fake::new(4);
        // Over the limit on every attempt.
        *fake.stubborn.borrow_mut() = TRIES + 2;
        let rep = build_all(&store, &id, &fake);
        assert_eq!(*fake.calls.borrow(), TRIES, "retried exactly TRIES times");
        assert_eq!(rep.clipped, 1);
        let (text, clipped) = get(&store, &id, Coord::new(0, 0)).unwrap();
        assert!(clipped, "recorded as clipped, not passed off as a summary");
        assert!(text.len() <= NODE);
        // The tree did not stall: the node is built, so its parent can be.
        assert!(built(&store, &id, Coord::new(0, 0)));
    }

    #[test]
    fn a_model_that_fits_on_retry_has_its_shortest_line_kept() {
        let long = "beta ".repeat(300);
        let (store, id) = chat(&[("user", long.trim())]);
        let fake = Fake::new(4);
        *fake.stubborn.borrow_mut() = 2; // two over-limit answers, then a good one
        let rep = build_all(&store, &id, &fake);
        assert_eq!(rep.compacted, 1);
        assert_eq!(rep.clipped, 0);
        assert_eq!(*fake.calls.borrow(), 3);
        assert!(get(&store, &id, Coord::new(0, 0)).unwrap().0.len() <= NODE);
    }

    #[test]
    fn a_failing_compactor_clips_rather_than_losing_the_message() {
        let long = "gamma ".repeat(300);
        let (store, id) = chat(&[("user", long.trim())]);
        let fake = Fake {
            calls: RefCell::new(0),
            words: 4,
            fail: true,
            stubborn: RefCell::new(0),
        };
        let rep = build_all(&store, &id, &fake);
        assert_eq!(rep.clipped, 1);
        assert!(!rep.errors.is_empty(), "the failure is reported, not hidden");
        // Nothing was lost: the original message is still a node in the tree.
        assert_eq!(count(&store, &id), 1);
    }

    #[test]
    fn a_half_filled_pair_waits_for_its_other_half() {
        let (store, id) = chat(&[("user", "a"), ("assistant", "b"), ("user", "c")]);
        let fake = Fake::new(5);
        build_all(&store, &id, &fake);
        // 3 messages: leaves 0,1,2 and the merge (1,0). (1,1) covers [2,4),
        // which runs past the end, so it is not built yet.
        assert!(built(&store, &id, Coord::new(1, 0)));
        assert!(!built(&store, &id, Coord::new(1, 1)));
        assert!(!built(&store, &id, Coord::new(2, 0)));
        // The fourth message completes both.
        store
            .append_event(&id, NewEvent::new("context_window", json!({"role":"assistant","content":"d"})))
            .unwrap();
        build_all(&store, &id, &fake);
        assert!(built(&store, &id, Coord::new(1, 1)));
        assert!(built(&store, &id, Coord::new(2, 0)));
        assert_eq!(
            get(&store, &id, Coord::new(2, 0)).unwrap().0,
            "user: a assistant: b user: c assistant: d"
        );
    }

    #[test]
    fn building_twice_builds_nothing_the_second_time() {
        let (store, id) = chat(&[("user", "a"), ("assistant", "b")]);
        let fake = Fake::new(5);
        assert_eq!(build_all(&store, &id, &fake).total(), 3);
        assert_eq!(build_all(&store, &id, &fake).total(), 0);
        assert!(pending(&store, &id).is_empty());
    }

    // --- the view -----------------------------------------------------------

    fn long_chat(n: usize) -> (ConvStore, String) {
        let msgs: Vec<(String, String)> = (0..n)
            .map(|i| {
                (
                    if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                    format!("message number {i} with enough text to be worth a few bytes"),
                )
            })
            .collect();
        let refs: Vec<(&str, &str)> = msgs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        chat(&refs)
    }

    #[test]
    fn an_uncompacted_conversation_is_a_flat_view_of_leaves() {
        let (store, id) = long_chat(5);
        let v = view(&store, &id);
        assert_eq!(v.len(), 5);
        assert!(v.iter().all(|p| p.coord.width() == 1));
        // Nothing built yet, so nothing claims to be a summary.
        assert!(v.iter().all(|p| p.text.is_none()));
    }

    #[test]
    fn a_generous_budget_leaves_every_message_at_full_detail() {
        let (store, id) = long_chat(16);
        build_all(&store, &id, &Fake::new(6));
        let v = fit(&store, &id, HIGH);
        assert_eq!(v.len(), 16, "nothing to merge: it already fits");
        assert!(v.iter().all(|p| p.coord.width() == 1));
    }

    #[test]
    fn a_tight_budget_folds_the_oldest_messages_into_threads() {
        let (store, id) = long_chat(16);
        build_all(&store, &id, &Fake::new(6));
        let v = fit(&store, &id, 200);
        assert!(v.len() < 16, "the view was reduced");
        // Oldest first, contiguous, covering the whole chat exactly once.
        let mut at = 0u64;
        for p in &v {
            assert_eq!(p.coord.span().0, at);
            at = p.coord.span().1;
        }
        assert_eq!(at, 16);
        // The coarse parts are at the front and the fine ones at the back:
        // detail fades with age.
        let widths: Vec<u64> = v.iter().map(|p| p.coord.width()).collect();
        assert!(
            widths.first().unwrap() >= widths.last().unwrap(),
            "expected old-coarse/new-fine, got {widths:?}"
        );
        assert!(widths.iter().any(|w| *w > 1), "something became a thread");
    }

    #[test]
    fn new_messages_join_the_view_at_full_detail() {
        let (store, id) = long_chat(16);
        build_all(&store, &id, &Fake::new(6));
        fit(&store, &id, 200);
        for i in 0..2 {
            store
                .append_event(
                    &id,
                    NewEvent::new("context_window", json!({"role":"user","content":format!("new {i}")})),
                )
                .unwrap();
        }
        build_all(&store, &id, &Fake::new(6));
        let v = fit(&store, &id, 200);
        let last = v.last().unwrap();
        // The newest message is a leaf: recent data stays at fine resolution.
        assert_eq!(last.coord.width(), 1);
        assert_eq!(last.coord.span(), (17, 18));
        let mut at = 0u64;
        for p in &v {
            assert_eq!(p.coord.span().0, at);
            at = p.coord.span().1;
        }
        assert_eq!(at, 18, "still covers every message");
    }

    #[test]
    fn the_view_survives_a_restart() {
        let (store, id) = long_chat(16);
        build_all(&store, &id, &Fake::new(6));
        let fitted: Vec<String> = fit(&store, &id, 200)
            .iter()
            .map(|p| p.coord.reference())
            .collect();
        // `view` reads the persisted one rather than recomputing.
        let reloaded: Vec<String> = view(&store, &id)
            .iter()
            .map(|p| p.coord.reference())
            .collect();
        assert_eq!(fitted, reloaded);
    }

    #[test]
    fn a_rendered_view_names_every_line_so_a_model_can_zoom_it() {
        let (store, id) = long_chat(8);
        build_all(&store, &id, &Fake::new(6));
        let text = render(&fit(&store, &id, 150));
        assert!(text.starts_with("<chat>"));
        assert!(text.ends_with("</chat>"));
        // Each line is `id+n|text`, which is the reference `zoom` takes.
        for line in text.lines().filter(|l| l.contains('|')) {
            let r = line.trim().split('|').next().unwrap();
            let (id_s, n_s) = r.split_once('+').expect("a reference");
            assert!(id_s.parse::<u64>().is_ok());
            assert!(n_s.parse::<u64>().is_ok());
        }
    }

    // --- zoom ---------------------------------------------------------------

    #[test]
    fn zoom_opens_one_level_at_a_time_down_to_the_original_message() {
        let (store, id) = long_chat(8);
        build_all(&store, &id, &Fake::new(6));
        // The whole chat as one line.
        let Zoom::Halves(h) = zoom(&store, &id, 0, 8).unwrap() else {
            panic!("expected two halves")
        };
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].coord.reference(), "0+4");
        assert_eq!(h[1].coord.reference(), "4+4");
        assert!(h.iter().all(|p| p.text.is_some()));

        // Descend the left branch to the bottom.
        let Zoom::Halves(h) = zoom(&store, &id, 0, 4).unwrap() else { panic!() };
        assert_eq!(h[0].coord.reference(), "0+2");
        let Zoom::Halves(h) = zoom(&store, &id, 0, 2).unwrap() else { panic!() };
        assert_eq!(h[0].coord.reference(), "0+1");

        // Width 1 is the original message, not another summary.
        let Zoom::Message { index, node_id } = zoom(&store, &id, 0, 1).unwrap() else {
            panic!("expected the message itself")
        };
        assert_eq!(index, 0);
        let node = store.node(&node_id).unwrap();
        assert!(node.content.unwrap().contains("message number 0"));
    }

    #[test]
    fn a_bad_reference_is_answered_with_the_real_roots() {
        let (store, id) = long_chat(11);
        build_all(&store, &id, &Fake::new(6));
        let err = zoom(&store, &id, 0, 3).unwrap_err();
        // Naming the actual lines is what lets the caller correct itself.
        assert!(err.contains("0+8"), "{err}");
        assert!(err.contains("8+2"), "{err}");
        assert!(err.contains("10+1"), "{err}");
        assert!(zoom(&store, &id, 1, 2).is_err(), "misaligned");
        assert!(zoom(&store, &id, 8, 8).is_err(), "past the end");
    }

    #[test]
    fn nothing_is_lost_every_message_is_still_reachable_after_compaction() {
        let (store, id) = long_chat(16);
        build_all(&store, &id, &Fake::new(6));
        let v = fit(&store, &id, 120);
        assert!(v.len() < 16, "it really did compact");

        // Walk every view part down to its leaves and collect the messages.
        let mut found: Vec<u64> = Vec::new();
        let mut stack: Vec<Coord> = v.iter().map(|p| p.coord).collect();
        while let Some(c) = stack.pop() {
            let (start, _) = c.span();
            match zoom(&store, &id, start, c.width()).unwrap() {
                Zoom::Message { index, .. } => found.push(index),
                Zoom::Halves(h) => stack.extend(h.iter().map(|p| p.coord)),
            }
        }
        found.sort();
        // The tree never shrinks, only coarsens.
        assert_eq!(found, (0..16).collect::<Vec<u64>>());
    }

    #[test]
    fn a_leaf_part_carries_its_original_message_for_the_ui() {
        let (store, id) = long_chat(3);
        build_all(&store, &id, &Fake::new(6));
        let msgs = crate::serve::leaf_messages(&store, &id);
        let v = view(&store, &id);
        let j = v[0].to_json(Some(&msgs));
        assert_eq!(j["leaf"], true);
        assert_eq!(j["n"], 1);
        // The UI renders a width-1 part as the real card, so it needs the row.
        assert!(j["message"]["text"].as_str().unwrap().contains("message number 0"));
        // A thread carries no message: it is an index, not a row.
        let thread = Part {
            coord: Coord::new(1, 0),
            text: Some("summary".into()),
            clipped: false,
        };
        assert!(thread.to_json(Some(&msgs)).get("message").is_none());
    }

    #[test]
    fn leaf_parts_line_up_with_messages_even_past_a_reasoning_row() {
        let (store, id) = chat(&[("user", "first")]);
        // A reasoning row sits in the transcript but is NOT a tree leaf, so
        // indexing the transcript by leaf number drifts from here on.
        store
            .append_event(&id, NewEvent::new("reasoning", json!({ "text": "pondering" })))
            .unwrap();
        store
            .append_event(
                &id,
                NewEvent::new("context_window", json!({"role":"assistant","content":"second"})),
            )
            .unwrap();
        store
            .append_event(
                &id,
                NewEvent::new("context_window", json!({"role":"user","content":"third"})),
            )
            .unwrap();
        build_all(&store, &id, &Fake::new(6));

        let aligned = crate::serve::leaf_messages(&store, &id);
        let v = view(&store, &id);
        let texts: Vec<String> = v
            .iter()
            .map(|p| {
                p.to_json(Some(&aligned))["message"]["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        assert_eq!(texts, vec!["first", "second", "third"]);

        // The naive version — indexing the raw transcript — gets it wrong
        // from the reasoning row onward. Pinned so the fix cannot regress.
        let raw = crate::serve::reconstruct_messages(&store, &id);
        assert_eq!(raw.len(), 4, "the transcript has one more row than the tree");
        assert_ne!(raw[1]["text"], aligned[1]["text"]);
    }

    #[test]
    fn only_messages_are_leaves_not_reasoning_or_telemetry() {
        let (store, id) = chat(&[("user", "a")]);
        store.append_event(&id, NewEvent::new("reasoning", json!({"text":"hmm"}))).unwrap();
        store.append_event(&id, NewEvent::new("gen_info", json!({"total_tokens":3}))).unwrap();
        store.append_event(&id, NewEvent::new("context_window", json!({"role":"assistant","content":"b"}))).unwrap();
        // A summary of the chat is a summary of what was said, not of the
        // machinery around it.
        assert_eq!(count(&store, &id), 2);
        build_all(&store, &id, &Fake::new(5));
        assert_eq!(get(&store, &id, Coord::new(1, 0)).unwrap().0, "user: a assistant: b");
    }

    #[test]
    fn the_tree_follows_the_branch_you_are_looking_at() {
        let (store, id) = chat(&[("user", "a"), ("assistant", "b"), ("user", "c"), ("assistant", "d")]);
        build_all(&store, &id, &Fake::new(6));
        assert_eq!(count(&store, &id), 4);
        assert_eq!(
            get(&store, &id, Coord::new(2, 0)).unwrap().0,
            "user: a assistant: b user: c assistant: d"
        );

        // Fork at message 1 ("b") and say something else.
        let leaves = store.memory_leaves(&id);
        assert!(store.set_cursor(&id, Some(&leaves[1])));
        store
            .append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"X"})))
            .unwrap();

        // The branch now has 3 messages: a, b, X. The old (1,1) summarised
        // "c, d", which are not on this branch at all, so it must be gone —
        // while (1,0) summarised "a, b", which still holds.
        assert_eq!(sync(&store, &id), 3);
        assert!(built(&store, &id, Coord::new(1, 0)), "an untouched prefix survives");
        assert!(!built(&store, &id, Coord::new(1, 1)), "a summary of the other branch is dropped");
        assert!(!built(&store, &id, Coord::new(2, 0)), "and so is anything spanning the fork");

        let texts: Vec<String> = view(&store, &id)
            .iter()
            .map(|p| p.text.clone().unwrap_or_default())
            .collect();
        assert!(texts.iter().all(|t| !t.contains("user: c")), "{texts:?}");

        // Back on the original branch, its own summaries are rebuilt.
        let all = store.all_nodes(&id);
        let d_node = all.iter().find(|n| n.content.as_deref() == Some("d")).unwrap();
        store.set_cursor(&id, Some(&d_node.id));
        build_all(&store, &id, &Fake::new(6));
        assert_eq!(count(&store, &id), 4);
        assert_eq!(
            get(&store, &id, Coord::new(2, 0)).unwrap().0,
            "user: a assistant: b user: c assistant: d"
        );
    }

    #[test]
    fn a_persisted_view_still_shows_messages_sent_since_it_was_fitted() {
        let (store, id) = long_chat(16);
        build_all(&store, &id, &Fake::new(6));
        fit(&store, &id, 200);
        store
            .append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"newest"})))
            .unwrap();
        // No build, no fit: just reading the view back. It must still cover
        // every message — a stale snapshot that hides the newest message is
        // the one failure a history view cannot have.
        let v = view(&store, &id);
        let mut at = 0u64;
        for p in &v {
            assert_eq!(p.coord.span().0, at);
            at = p.coord.span().1;
        }
        assert_eq!(at, 17);
        assert_eq!(v.last().unwrap().coord, Coord::new(0, 16));
    }

    #[test]
    fn an_empty_conversation_has_no_tree_and_does_not_panic() {
        let (store, id) = chat(&[]);
        assert_eq!(count(&store, &id), 0);
        assert!(pending(&store, &id).is_empty());
        assert!(view(&store, &id).is_empty());
        assert!(fit(&store, &id, HIGH).is_empty());
        assert!(zoom(&store, &id, 0, 1).is_err());
        assert_eq!(render(&[]), "<chat>\n</chat>");
    }

    #[test]
    fn a_participant_s_handle_labels_its_messages_in_the_tree() {
        let store = ConvStore::memory("/tmp/p").unwrap();
        let id = store
            .create(NewSession { agent: Some("alice".into()), ..Default::default() })
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        let bob = store
            .add_participant(&id, "agent", "bob", Some("Bob"), Some("bob"), None, "", None, None)
            .unwrap();
        store.append_event(&id, NewEvent::new("context_window", json!({"role":"user","content":"q"}))).unwrap();
        store
            .append_event(
                &id,
                NewEvent {
                    event_type: "context_window".into(),
                    payload: json!({"role":"assistant","content":"from bob"}),
                    speaker: Some(bob.id),
                    ..Default::default()
                },
            )
            .unwrap();
        build_all(&store, &id, &Fake::new(5));
        // Who said it survives into the summary, so a merged thread does not
        // lose attribution in a group chat.
        assert_eq!(get(&store, &id, Coord::new(0, 1)).unwrap().0, "bob: from bob");
    }
}
