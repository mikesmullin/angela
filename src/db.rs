//! `.angela/conversations.db` — the SQLite schema and how it gets opened.
//!
//! One file per project root replaces the per-session `<id>.json` header and
//! `<id>.jsonl` log. The reason is not tidiness: the conversation stops being
//! a *line* and becomes a *tree*. A node names its parent, a cursor names the
//! active leaf, and "the conversation" is the walk from the cursor to its
//! oldest ancestor. An append-only text log cannot answer "what are this
//! node's children" without reading all of it, and cannot be edited in place
//! to materialise a visibility change — both of which this schema does in one
//! statement.
//!
//! Three things are deliberate and load-bearing:
//!
//! * **`seq` is monotonic per conversation and a parent's `seq` is always
//!   smaller than its child's.** That single rule is what makes cycles
//!   impossible, so the cursor→root walk always terminates without a visited
//!   set. It is enforced on insert, not by a trigger, so the error surfaces
//!   at the call site that broke it.
//! * **`payload` keeps the original event payload verbatim.** The JS-era
//!   telemetry types (`provider_request`, `harness`, …) carry fields this
//!   code has no opinion about; storing the JSON whole means a migration can
//!   never silently drop one. The denormalised columns beside it
//!   (`tool_call_id`, `gen_*`) are a cache for queries, never the truth.
//! * **`visible` / `goto_*` are columns, not log entries.** The JSONL format
//!   expressed "hide this row" by appending a `context_meta` patch that every
//!   reader had to fold forward. Here it is an `UPDATE`. Migrated
//!   `context_meta` rows are still *kept* as nodes so an old log stays fully
//!   replayable, but nothing writes new ones.
//!
//! `cursor.participant_id` and `participant_view` use `0` for
//! "conversation-wide" rather than NULL, because SQLite treats NULLs in a
//! primary key as distinct from each other — a NULL row would not be unique,
//! which is the one property the shared cursor needs.

use rusqlite::Connection;
use std::path::{Path, PathBuf};

/// Bumped when the DDL below changes in a way an existing file must be
/// brought forward to. Recorded in `schema_meta`.
pub const SCHEMA_VERSION: i64 = 8;

pub const DDL: &str = r#"
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per conversation: what the `<id>.json` header used to hold.
-- `lastPromptTokens` and friends are gone; they are derivable from the
-- newest gen_info node, and a cached copy was one more thing to keep honest.
CREATE TABLE IF NOT EXISTS conversation (
    id               TEXT PRIMARY KEY,
    ordinal          INTEGER NOT NULL,
    api_version      TEXT NOT NULL DEFAULT 'conv/v1',
    kind             TEXT NOT NULL DEFAULT 'conversation',
    title            TEXT,
    agent            TEXT,
    agent_file       TEXT,
    model            TEXT,
    status           TEXT NOT NULL DEFAULT 'ready',
    policy_mode      TEXT NOT NULL DEFAULT 'ask',
    approve_mode     TEXT,
    allowlist_mode   TEXT,
    allowlist        TEXT,
    allowlist_source TEXT,
    context_window   INTEGER,
    mcp              TEXT,
    project_root     TEXT NOT NULL,
    -- Which folder this conversation is filed under, as a path-ish label
    -- ("psicom/chat"). A plain string, not a foreign key: the hierarchy is
    -- derived from the '/' in the name, so moving a folder is one UPDATE
    -- and there is no tree to keep consistent with the labels.
    project          TEXT,
    -- Where this conversation's work happens. `sandbox` names a box in the
    -- sbox service; `worktree` and `branch` name the checkout bound into
    -- it. They are opaque ids owned by other services, so they are plain
    -- text with no constraint — a sandbox destroyed elsewhere leaves a
    -- dangling id here, and the UI falls back rather than failing.
    sandbox          TEXT,
    worktree         TEXT,
    branch_ref       TEXT,
    config           TEXT,
    overrides        TEXT,
    turns            INTEGER NOT NULL DEFAULT 0,
    -- group-chat round machinery (see group.rs)
    round_index      INTEGER NOT NULL DEFAULT 0,
    round_status     TEXT,
    routing          TEXT NOT NULL DEFAULT 'everyone',
    purpose          TEXT,
    notes            TEXT,
    -- memory tree budget; NULL = the module default
    memory_high      INTEGER,
    -- 'jsonl' on a conversation that came in from the legacy files
    legacy_format    TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

-- The folders that exist, independent of whether anything is in them.
--
-- The assignment itself lives on `conversation.project`; this table only
-- remembers NAMES. Without it a folder would exist exactly as long as a
-- conversation pointed at it, so creating one before filing anything into
-- it, or dragging the last session out of one, would make it vanish — both
-- of which the UI offers as ordinary gestures.
CREATE TABLE IF NOT EXISTS projects (
    name       TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

-- App settings: one row per key.
--
-- Key/value rather than a column per setting, so adding one is a write
-- rather than a migration. The cost is that nothing is typed — which is why
-- the writer keeps a list of the keys it knows and drops the rest, instead
-- of letting any caller invent a setting.
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Agent definitions. Formerly one `<name>.json` per agent under
-- `.angela/agents/`; now rows, so the UI can create and edit them without a
-- file-write round trip and without two readers disagreeing about what an
-- agent is.
--
-- The three prompt fields are separate on purpose rather than one blob:
-- `soul` is who the agent is (written once), `goals` is what it is working
-- toward (the user's steering), and `memory` is what it has learned and may
-- rewrite itself. They concatenate into the system prompt in that order, and
-- keeping them apart is what lets an agent edit its own memory without
-- being able to rewrite its own character.
CREATE TABLE IF NOT EXISTS agents (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL UNIQUE,
    description    TEXT NOT NULL DEFAULT '',
    model          TEXT NOT NULL DEFAULT '',
    models         TEXT NOT NULL DEFAULT '[]',
    context_window INTEGER,
    cwd            TEXT,
    env            TEXT NOT NULL DEFAULT '{}',
    mcp            TEXT NOT NULL DEFAULT '[]',
    allowlist      TEXT NOT NULL DEFAULT '',
    tools_enabled  TEXT,                              -- NULL = every tool
    policy_mode    TEXT NOT NULL DEFAULT 'ask',
    parallel_tools INTEGER NOT NULL DEFAULT 1,
    soul           TEXT NOT NULL DEFAULT '',          -- was `system`
    goals          TEXT NOT NULL DEFAULT '',
    memory         TEXT NOT NULL DEFAULT '',
    -- invoked  : speaks only when @mentioned
    -- relevant : a System One call decides, per message
    -- engaged  : speaks on every message
    engagement     TEXT NOT NULL DEFAULT 'invoked',
    starters       TEXT NOT NULL DEFAULT '[]',
    avatar         TEXT,
    -- The colour this agent is drawn in: its handle, its reasoning, its rail.
    -- NULL means the app's default accent, so an agent nobody has coloured
    -- still looks like the rest of the UI rather than like a missing value.
    accent_color   TEXT,
    -- The Phosphor icon name it is drawn with. NULL means the default
    -- robot, so an agent nobody has given an icon still looks like an
    -- agent rather than like a missing glyph.
    icon           TEXT,
    reasoning_effort TEXT NOT NULL DEFAULT '',
    max_turns      INTEGER,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

-- The roster. A 2-party chat is a roster of two: the human and one agent.
CREATE TABLE IF NOT EXISTS participants (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,              -- 'user' | 'agent'
    handle          TEXT NOT NULL,              -- the @handle in flattened prompts
    display_name    TEXT,
    agent_name      TEXT,                       -- .angela/agents/<name>.json
    model           TEXT,
    system_prompt   TEXT NOT NULL DEFAULT '',
    avatar          TEXT,
    seat            INTEGER NOT NULL DEFAULT 0,
    active          INTEGER NOT NULL DEFAULT 1,
    role            TEXT,                       -- 'lead' | 'member' | …
    -- NULL means "whatever the agent definition currently says", so editing
    -- an agent changes how it behaves in the conversations it is already in.
    -- A value here is a per-conversation override.
    engagement      TEXT,
    UNIQUE (conversation_id, handle)
);

-- The conversation graph. One row per event, exactly the granularity the
-- JSONL had, so every event type can be a fork point.
CREATE TABLE IF NOT EXISTS nodes (
    id                    TEXT PRIMARY KEY,
    conversation_id       TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    parent_id             TEXT REFERENCES nodes(id),   -- NULL = a branch root
    seq                   INTEGER NOT NULL,
    depth                 INTEGER NOT NULL,
    kind                  TEXT NOT NULL,               -- the legacy event_type
    participant_id        INTEGER REFERENCES participants(id),
    content               TEXT,
    role                  TEXT,
    payload               TEXT NOT NULL,               -- original payload JSON
    meta                  TEXT NOT NULL DEFAULT '{}',
    visible               INTEGER NOT NULL DEFAULT 1,
    goto_enabled          INTEGER NOT NULL DEFAULT 0,
    goto_label            TEXT,
    ts                    TEXT NOT NULL,
    duration_ms           REAL,
    created_at            TEXT NOT NULL,
    -- denormalised hot fields; a cache over `payload`, never the truth
    tool_name             TEXT,
    tool_call_id          TEXT,
    ok                    INTEGER,
    denied                INTEGER,
    gen_prompt_tokens     INTEGER,
    gen_completion_tokens INTEGER,
    gen_total_tokens      INTEGER,
    gen_ttft_ms           REAL,
    gen_duration_ms       REAL,
    UNIQUE (conversation_id, seq)
);

-- The active leaf. `participant_id = 0` is the shared, conversation-wide
-- cursor, which is the only one v1 writes; a non-zero row is a participant
-- exploring their own branch.
CREATE TABLE IF NOT EXISTS cursor (
    conversation_id TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    participant_id  INTEGER NOT NULL DEFAULT 0,
    event_id        TEXT REFERENCES nodes(id),
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (conversation_id, participant_id)
);

-- Where each agent participant has read up to, and where it sits in the
-- current round. `last_seen_seq` is the hermes watermark.
CREATE TABLE IF NOT EXISTS participant_view (
    conversation_id TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    participant_id  INTEGER NOT NULL REFERENCES participants(id) ON DELETE CASCADE,
    last_seen_seq   INTEGER NOT NULL DEFAULT 0,
    round_index     INTEGER NOT NULL DEFAULT 0,
    reply_required  INTEGER NOT NULL DEFAULT 0,
    task_id         TEXT,
    round_status    TEXT,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (conversation_id, participant_id)
);

-- Each agent's own (prompt, reply) history: its private LLM conversation,
-- separate from the shared room. Storing the prompt verbatim is what keeps
-- the agent's context window byte-stable across turns and across forks,
-- which is the whole point of the hermes shape.
CREATE TABLE IF NOT EXISTS participant_session (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    participant_id  INTEGER NOT NULL REFERENCES participants(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    prompt          TEXT NOT NULL,
    reply           TEXT NOT NULL,
    task_id         TEXT,
    ts              TEXT NOT NULL,
    UNIQUE (participant_id, seq)
);

-- ---------------------------------------------------------------------------
-- The memory tree (memory.rs): a binary tree of one-line summaries over the
-- conversation's messages. Node (l, i) covers messages [i*2^l, (i+1)*2^l).
-- ---------------------------------------------------------------------------

-- Which message index each renderable node is. Dense, append-only, assigned
-- in `seq` order, so the tree's coordinates stay stable as the chat grows.
CREATE TABLE IF NOT EXISTS memory_leaves (
    conversation_id TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    i               INTEGER NOT NULL,
    node_id         TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    PRIMARY KEY (conversation_id, i),
    UNIQUE (conversation_id, node_id)
);

-- One built summary line per node. `slot` is the heap position
-- (2i+1)*2^l - 1, stored so a view can be ordered without recomputing it.
CREATE TABLE IF NOT EXISTS memory_nodes (
    conversation_id TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    level           INTEGER NOT NULL,
    idx             INTEGER NOT NULL,
    slot            INTEGER NOT NULL,
    text            TEXT NOT NULL,
    bytes           INTEGER NOT NULL,
    -- 1 when the model never produced a short enough line and the source was
    -- truncated instead, so the tree never stalls on a refusal
    clipped         INTEGER NOT NULL DEFAULT 0,
    ts              TEXT NOT NULL,
    PRIMARY KEY (conversation_id, level, idx)
);

-- The current view: the topmost nodes covering the chat, oldest first.
-- Persisted so a restart renders the same threads it rendered before.
CREATE TABLE IF NOT EXISTS memory_view (
    conversation_id TEXT PRIMARY KEY REFERENCES conversation(id) ON DELETE CASCADE,
    parts           TEXT NOT NULL,   -- JSON [[l, i], …]
    end_count       INTEGER NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_conv_seq       ON nodes(conversation_id, seq);
CREATE INDEX IF NOT EXISTS idx_nodes_parent         ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_conv_part_kind ON nodes(conversation_id, participant_id, kind, seq);
CREATE INDEX IF NOT EXISTS idx_nodes_conv_vis       ON nodes(conversation_id, visible, seq);
CREATE INDEX IF NOT EXISTS idx_nodes_tool_call_id   ON nodes(tool_call_id);
CREATE INDEX IF NOT EXISTS idx_nodes_gen_conv       ON nodes(conversation_id, seq) WHERE kind = 'gen_info';
CREATE INDEX IF NOT EXISTS idx_psession_part        ON participant_session(participant_id, seq);
CREATE INDEX IF NOT EXISTS idx_view_conv_part       ON participant_view(conversation_id, participant_id);
CREATE INDEX IF NOT EXISTS idx_memory_slot          ON memory_nodes(conversation_id, slot);
CREATE INDEX IF NOT EXISTS idx_participants_conv    ON participants(conversation_id, seat);
CREATE INDEX IF NOT EXISTS idx_agents_name          ON agents(name);
"#;

/// `<root>/.angela/conversations.db`.
pub fn db_path(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join(".angela").join("conversations.db")
}

/// Open (creating if absent) and bring the schema up to date.
///
/// WAL so a reader never blocks the writer mid-turn; `foreign_keys` on
/// because the roster/cursor references are the thing keeping a conversation
/// internally consistent, and `NORMAL` synchronous because WAL already
/// survives a process crash and the extra fsync per append is felt on a
/// token-by-token write path.
pub fn open(root: impl AsRef<Path>) -> rusqlite::Result<Connection> {
    let path = db_path(&root);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let conn = Connection::open(&path)?;
    init(&conn)?;
    Ok(conn)
}

/// An in-memory database with the same schema — for tests.
pub fn open_memory() -> rusqlite::Result<Connection> {
    let conn = Connection::open_in_memory()?;
    init(&conn)?;
    Ok(conn)
}

fn init(conn: &Connection) -> rusqlite::Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    // A turn and a list request can land at once; fail slowly rather than
    // immediately with SQLITE_BUSY.
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    conn.execute_batch(DDL)?;
    // Columns added after a database may already exist. `ALTER TABLE ADD
    // COLUMN` is a no-op-shaped error when the column is there, so the
    // result is discarded rather than branched on — cheaper and less
    // fragile than reading `pragma_table_info` for each one.
    for stmt in [
        "ALTER TABLE participants ADD COLUMN engagement TEXT",
        "ALTER TABLE agents ADD COLUMN accent_color TEXT",
        "ALTER TABLE agents ADD COLUMN reasoning_effort TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE agents ADD COLUMN max_turns INTEGER",
        "ALTER TABLE agents ADD COLUMN icon TEXT",
        "ALTER TABLE conversation ADD COLUMN project TEXT",
        "ALTER TABLE conversation ADD COLUMN sandbox TEXT",
        "ALTER TABLE conversation ADD COLUMN worktree TEXT",
        "ALTER TABLE conversation ADD COLUMN branch_ref TEXT",
    ] {
        let _ = conn.execute(stmt, []);
    }
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('version', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [SCHEMA_VERSION.to_string()],
    )?;
    Ok(())
}

/// The recorded schema version, or 0 for a database that has none.
pub fn version(conn: &Connection) -> i64 {
    conn.query_row(
        "SELECT value FROM schema_meta WHERE key = 'version'",
        [],
        |r| r.get::<_, String>(0),
    )
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schema_applies_and_is_idempotent() {
        let conn = open_memory().unwrap();
        assert_eq!(version(&conn), SCHEMA_VERSION);
        // Re-running the DDL must not fail: `open` does it on every start.
        conn.execute_batch(DDL).unwrap();
    }

    #[test]
    fn foreign_keys_are_enforced() {
        let conn = open_memory().unwrap();
        let err = conn.execute(
            "INSERT INTO participants (conversation_id, kind, handle, seat)
             VALUES ('nope', 'agent', 'a', 0)",
            [],
        );
        assert!(err.is_err(), "a participant of no conversation must be refused");
    }

    #[test]
    fn the_shared_cursor_row_is_unique() {
        let conn = open_memory().unwrap();
        conn.execute(
            "INSERT INTO conversation (id, ordinal, project_root, created_at, updated_at)
             VALUES ('c1', 1, '/tmp', 'now', 'now')",
            [],
        )
        .unwrap();
        for _ in 0..2 {
            conn.execute(
                "INSERT INTO cursor (conversation_id, participant_id, event_id, updated_at)
                 VALUES ('c1', 0, NULL, 'now')
                 ON CONFLICT(conversation_id, participant_id)
                   DO UPDATE SET updated_at = excluded.updated_at",
                [],
            )
            .unwrap();
        }
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM cursor", [], |r| r.get(0))
            .unwrap();
        // `participant_id = 0` rather than NULL is what makes this 1: SQLite
        // treats NULLs in a primary key as distinct, so a NULL row would
        // have inserted twice.
        assert_eq!(n, 1);
    }
}
