use serde_json::{json, Value};

use crate::conv_store::ConvStore;

pub const NODE: usize = 512;
pub const TRIES: usize = 5;
pub const HIGH: usize = 128_000;
pub const SEEN: usize = 32_000;
pub const BLOCK: usize = 4;
pub const WAIT: &str = "(not summarized yet: zoom it)";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coord {
    pub l: u32,
    pub i: u64,
}

impl Coord {
    pub fn new(l: u32, i: u64) -> Self {
        Coord { l, i }
    }
    pub fn span(&self) -> (u64, u64) {
        let w = 1u64 << self.l;
        (self.i * w, (self.i + 1) * w)
    }
    pub fn width(&self) -> u64 {
        1u64 << self.l
    }
    pub fn ctx(&self) -> u64 {
        self.span().1
    }
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
    pub fn reference(&self) -> String {
        let (start, _) = self.span();
        format!("{}+{}", start, self.width())
    }
}

pub fn part(id: u64, n: u64, count: u64) -> Option<Coord> {
    if n == 0 || !n.is_power_of_two() {
        return None;
    }
    if id % n != 0 || id + n > count {
        return None;
    }
    Some(Coord::new(n.trailing_zeros(), id / n))
}

pub fn roots(count: u64) -> Vec<Coord> {
    let mut out = Vec::new();
    let mut at = 0u64;
    let mut rem = count;
    while rem > 0 {
        let mut l = 63 - rem.leading_zeros().min(63);
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

#[derive(Debug, Clone)]
pub struct Part {
    pub coord: Coord,
    pub text: Option<String>,
    pub clipped: bool,
}

impl Part {
    pub fn bytes(&self) -> usize {
        self.text.as_ref().map(|t| t.len()).unwrap_or(0)
    }

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

pub trait Compactor {
    fn compact(&self, prompt: &str) -> Result<String, String>;
}

fn ruler() -> String {
    "-".repeat(NODE)
}

pub fn leaf_prompt(id: u64, source: &str) -> String {
    format!(
        "Compaction: compress message {id} into one line of at most {NODE} bytes \
         (about 70 words), the length of this ruler:\n{}\n<input>\n{source}\n</input>",
        ruler()
    )
}

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

fn flat(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}


pub fn count(store: &ConvStore, conv: &str) -> u64 {
    store.memory_leaves(conv).len() as u64
}

pub fn sync(store: &ConvStore, conv: &str) -> u64 {
    store.sync_memory_tree(conv) as u64
}

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

pub fn source(store: &ConvStore, conv: &str, c: Coord) -> Option<String> {
    if c.l == 0 {
        return message_line(store, conv, c.i);
    }
    let (a, b) = c.children()?;
    let ta = get(store, conv, a)?.0;
    let tb = get(store, conv, b)?.0;
    Some(format!("{ta}\n{tb}"))
}


#[derive(Debug, Default, Clone, PartialEq)]
pub struct Built {
    pub free: usize,
    pub compacted: usize,
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

pub fn build_all(store: &ConvStore, conv: &str, compactor: &dyn Compactor) -> Built {
    sync(store, conv);
    let mut report = Built::default();
    for c in pending(store, conv) {
        let Some(src) = source(store, conv, c) else {
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

fn due(a: Coord, end: u64) -> f64 {
    (end as f64 + 1.0) / (a.width() as f64) - a.i as f64
}

pub fn fit(store: &ConvStore, conv: &str, high: usize) -> Vec<Part> {
    let n = sync(store, conv);
    let mut parts: Vec<Coord> = match load_view(store, conv) {
        Some((p, end)) if end <= n && !p.is_empty() => p,
        _ => Vec::new(),
    };

    parts.retain(|c| c.ctx() <= n);
    let covered = parts.iter().map(|c| c.ctx()).max().unwrap_or(0);
    for i in covered..n {
        parts.push(Coord::new(0, i));
    }
    parts.sort_by_key(|c| c.span().0);

    let size = |ps: &Vec<Coord>| -> usize {
        ps.iter()
            .map(|c| get(store, conv, *c).map(|(t, _)| t.len()).unwrap_or(0))
            .sum()
    };
    let target = high / 2;
    let mut guard = 0;
    while size(&parts) > high && guard < 10_000 {
        guard += 1;
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

pub fn view(store: &ConvStore, conv: &str) -> Vec<Part> {
    let n = sync(store, conv);
    let mut parts = match load_view(store, conv) {
        Some((p, _)) if !p.is_empty() => p,
        _ => Vec::new(),
    };
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

#[derive(Debug, Clone)]
pub enum Zoom {
    Halves(Vec<Part>),
    Message { index: u64, node_id: String },
}

pub fn zoom(store: &ConvStore, conv: &str, id: u64, n: u64) -> Result<Zoom, String> {
    let total = sync(store, conv);
    let c = part(id, n, total).ok_or_else(|| {
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
