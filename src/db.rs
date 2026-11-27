use rusqlite::Connection;
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: i64 = 16;

pub const DDL: &str = r#"CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
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
    project          TEXT,
    workspace_root   TEXT,
    sandbox          TEXT,
    worktree         TEXT,
    branch_ref       TEXT,
    config           TEXT,
    overrides        TEXT,
    turns            INTEGER NOT NULL DEFAULT 0,
    round_index      INTEGER NOT NULL DEFAULT 0,
    round_status     TEXT,
    routing          TEXT NOT NULL DEFAULT 'everyone',
    purpose          TEXT,
    notes            TEXT,
    soul_appendage   TEXT,
    memory_high      INTEGER,
    legacy_format    TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_sandbox (
    session_id   TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    agent_name   TEXT NOT NULL,
    container_id TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    mount_path   TEXT,
    git_dir      TEXT,
    PRIMARY KEY (session_id, agent_name)
);
CREATE TABLE IF NOT EXISTS projects (
    name       TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    settings TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS inherited_worktree (
    session_id TEXT PRIMARY KEY REFERENCES conversation(id) ON DELETE CASCADE,
    record TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
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
    tools_enabled  TEXT,
    policy_mode    TEXT NOT NULL DEFAULT 'ask',
    parallel_tools INTEGER NOT NULL DEFAULT 1,
    soul           TEXT NOT NULL DEFAULT '',
    goals          TEXT NOT NULL DEFAULT '',
    memory         TEXT NOT NULL DEFAULT '',
    engagement     TEXT NOT NULL DEFAULT 'invoked',
    starters       TEXT NOT NULL DEFAULT '[]',
    avatar         TEXT,
    accent_color   TEXT,
    icon           TEXT,
    reasoning_effort TEXT NOT NULL DEFAULT '',
    max_turns      INTEGER,
    containerfile  TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS participants (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,
    handle          TEXT NOT NULL,
    display_name    TEXT,
    agent_name      TEXT,
    model           TEXT,
    system_prompt   TEXT NOT NULL DEFAULT '',
    avatar          TEXT,
    seat            INTEGER NOT NULL DEFAULT 0,
    active          INTEGER NOT NULL DEFAULT 1,
    role            TEXT,
    engagement      TEXT,
    UNIQUE (conversation_id, handle)
);
CREATE TABLE IF NOT EXISTS nodes (
    id                    TEXT PRIMARY KEY,
    conversation_id       TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    parent_id             TEXT REFERENCES nodes(id),
    seq                   INTEGER NOT NULL,
    depth                 INTEGER NOT NULL,
    kind                  TEXT NOT NULL,
    participant_id        INTEGER REFERENCES participants(id),
    content               TEXT,
    role                  TEXT,
    payload               TEXT NOT NULL,
    meta                  TEXT NOT NULL DEFAULT '{}',
    visible               INTEGER NOT NULL DEFAULT 1,
    goto_enabled          INTEGER NOT NULL DEFAULT 0,
    goto_label            TEXT,
    ts                    TEXT NOT NULL,
    duration_ms           REAL,
    created_at            TEXT NOT NULL,
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
CREATE TABLE IF NOT EXISTS cursor (
    conversation_id TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    participant_id  INTEGER NOT NULL DEFAULT 0,
    event_id        TEXT REFERENCES nodes(id),
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (conversation_id, participant_id)
);
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
CREATE TABLE IF NOT EXISTS queue_items (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    text            TEXT NOT NULL,
    from_name       TEXT NOT NULL DEFAULT '',
    agent           TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    tab_id          TEXT NOT NULL DEFAULT '',
    reason          TEXT NOT NULL DEFAULT '',
    ts              TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_leaves (
    conversation_id TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    i               INTEGER NOT NULL,
    node_id         TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    PRIMARY KEY (conversation_id, i),
    UNIQUE (conversation_id, node_id)
);
CREATE TABLE IF NOT EXISTS memory_nodes (
    conversation_id TEXT NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    level           INTEGER NOT NULL,
    idx             INTEGER NOT NULL,
    slot            INTEGER NOT NULL,
    text            TEXT NOT NULL,
    bytes           INTEGER NOT NULL,
    clipped         INTEGER NOT NULL DEFAULT 0,
    ts              TEXT NOT NULL,
    PRIMARY KEY (conversation_id, level, idx)
);
CREATE TABLE IF NOT EXISTS memory_view (
    conversation_id TEXT PRIMARY KEY REFERENCES conversation(id) ON DELETE CASCADE,
    parts           TEXT NOT NULL,
    end_count       INTEGER NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS principal (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL COLLATE NOCASE UNIQUE,
    display_name TEXT NOT NULL DEFAULT '',
    kind         TEXT NOT NULL DEFAULT 'user',
    created_at   INTEGER NOT NULL,
    disabled     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS device (
    id           INTEGER PRIMARY KEY,
    user_id      INTEGER NOT NULL REFERENCES principal(id),
    token_hash   TEXT NOT NULL UNIQUE,
    token_hint   TEXT NOT NULL DEFAULT '',
    fingerprint  TEXT NOT NULL DEFAULT '',
    user_agent   TEXT NOT NULL DEFAULT '',
    ip           TEXT NOT NULL DEFAULT '',
    created_at   INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    revoked_at   INTEGER
);
CREATE TABLE IF NOT EXISTS user_capability (
    user_id    INTEGER NOT NULL REFERENCES principal(id),
    capability TEXT NOT NULL,
    PRIMARY KEY (user_id, capability)
);
CREATE TABLE IF NOT EXISTS user_allowlist (
    user_id    INTEGER NOT NULL REFERENCES principal(id),
    kind       TEXT NOT NULL,
    pattern    TEXT NOT NULL,
    PRIMARY KEY (user_id, kind, pattern)
);
CREATE TABLE IF NOT EXISTS resource (
    id       INTEGER PRIMARY KEY,
    schema   TEXT NOT NULL DEFAULT 'chat',
    name     TEXT NOT NULL,
    kind     TEXT NOT NULL,
    owner_id INTEGER NOT NULL REFERENCES principal(id),
    UNIQUE (schema, name)
);
CREATE TABLE IF NOT EXISTS grant_ (
    grantee    INTEGER NOT NULL REFERENCES principal(id),
    object_id  INTEGER NOT NULL REFERENCES resource(id),
    privilege  TEXT NOT NULL,
    grantor    INTEGER NOT NULL REFERENCES principal(id),
    with_grant INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (grantee, object_id, privilege)
);
CREATE INDEX IF NOT EXISTS idx_device_user      ON device(user_id);
CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY,
    at        INTEGER NOT NULL,
    actor     TEXT NOT NULL,
    acting_as TEXT NOT NULL DEFAULT '',
    action    TEXT NOT NULL,
    detail    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_audit_at ON audit_log(at);
CREATE INDEX IF NOT EXISTS idx_resource_owner   ON resource(owner_id);
CREATE INDEX IF NOT EXISTS idx_grant_grantee    ON grant_(grantee);
CREATE INDEX IF NOT EXISTS idx_grant_object     ON grant_(object_id);
CREATE INDEX IF NOT EXISTS idx_nodes_conv_seq       ON nodes(conversation_id, seq);
CREATE INDEX IF NOT EXISTS idx_nodes_parent         ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_conv_part_kind ON nodes(conversation_id, participant_id, kind, seq);
CREATE INDEX IF NOT EXISTS idx_nodes_conv_vis       ON nodes(conversation_id, visible, seq);
CREATE INDEX IF NOT EXISTS idx_nodes_tool_call_id   ON nodes(tool_call_id);
CREATE INDEX IF NOT EXISTS idx_nodes_gen_conv       ON nodes(conversation_id, seq) WHERE kind = 'gen_info';
CREATE INDEX IF NOT EXISTS idx_psession_part        ON participant_session(participant_id, seq);
CREATE INDEX IF NOT EXISTS idx_view_conv_part       ON participant_view(conversation_id, participant_id);
CREATE INDEX IF NOT EXISTS idx_memory_slot          ON memory_nodes(conversation_id, slot);
CREATE INDEX IF NOT EXISTS idx_queue_conv_seq       ON queue_items(conversation_id, seq);
CREATE INDEX IF NOT EXISTS idx_participants_conv    ON participants(conversation_id, seat);
CREATE INDEX IF NOT EXISTS idx_agents_name          ON agents(name);"#;

pub fn db_path(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join(".angela").join("conversations.db")
}

pub fn open(root: impl AsRef<Path>) -> rusqlite::Result<Connection> {
    let path = db_path(&root);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let conn = Connection::open(&path)?;
    init(&conn)?;
    Ok(conn)
}

pub fn open_memory() -> rusqlite::Result<Connection> {
    let conn = Connection::open_in_memory()?;
    init(&conn)?;
    Ok(conn)
}

fn init(conn: &Connection) -> rusqlite::Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    conn.execute_batch(DDL)?;
    for stmt in [
        "ALTER TABLE projects ADD COLUMN settings TEXT NOT NULL DEFAULT '{}'",
        "ALTER TABLE session_sandbox ADD COLUMN mount_path TEXT",
        "ALTER TABLE session_sandbox ADD COLUMN git_dir TEXT",
        "ALTER TABLE participants ADD COLUMN engagement TEXT",
        "ALTER TABLE agents ADD COLUMN accent_color TEXT",
        "ALTER TABLE agents ADD COLUMN reasoning_effort TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE agents ADD COLUMN max_turns INTEGER",
        "ALTER TABLE agents ADD COLUMN icon TEXT",
        "ALTER TABLE agents ADD COLUMN containerfile TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE conversation ADD COLUMN project TEXT",
        "ALTER TABLE conversation ADD COLUMN sandbox TEXT",
        "ALTER TABLE conversation ADD COLUMN worktree TEXT",
        "ALTER TABLE conversation ADD COLUMN branch_ref TEXT",
        "ALTER TABLE conversation ADD COLUMN workspace_root TEXT",
        "ALTER TABLE conversation ADD COLUMN soul_appendage TEXT",
        "ALTER TABLE principal ADD COLUMN location TEXT NOT NULL DEFAULT ''",
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
