//! **angela** — an agentic chat harness: durable sessions, an approval
//! queue, tool policy, and an HTTP/NDJSON surface for a chat UI.
//!
//! Rust port of the Bun/JS angela at `/workspace/angela`, targeting feature
//! parity. The JS is the reference implementation: where its behaviour is
//! surprising, this port reproduces the surprise and records why, rather than
//! quietly "fixing" it — the on-disk session format and the HTTP contract are
//! both consumed by shipped clients.
//!
//! Port status:
//!
//! | JS source | Rust module | state |
//! |---|---|---|
//! | `src/project.mjs` | [`project`] | ported |
//! | `src/session-store.mjs` | [`session_store`] | ported — now the **legacy reader** only (see below) |
//! | `src/serve.mjs` | [`serve`] | ported |
//! | `src/index.mjs` | [`harness`] | ported |
//! | `src/agent-loader.mjs` | [`agent_loader`] | ported (`.coffee` → JSON), then superseded by [`agent_store`]: definitions are rows now, and the files are import-only |
//! | `src/allowlist.mjs`, `policy.mjs` | [`allowlist`] | ported |
//! | `src/mcp-*.mjs` | [`mcp`] | ported |
//!
//! Beyond the port, storage moved from the per-session `.json`/`.jsonl` pair
//! to a single `.angela/conversations.db`, which is what makes three things
//! possible that a flat log cannot express:
//!
//! | module | what it adds |
//! |---|---|
//! | [`db`], [`conv_store`] | the conversation as a **tree**: nodes name a parent, a cursor names the active leaf, and forking is "move the cursor, then append" |
//! | [`group`] | **N participants** — each agent in its own context, reading the room as flattened text (the hermes shape), with watermarks, rounds and `@mentions` |
//! | [`memory`] | an **unbounded history in a bounded window**: a binary tree of summaries, coarse with age, where any line opens into the two it was merged from, down to the original message |
//! | [`agent_store`], [`relevance`] | **agents are rows**, created and edited in the UI, each carrying a soul, goals, a memory it may rewrite, and an `engagement` that decides when it speaks |
//! | [`migrate`] | the one-shot, idempotent, non-destructive import of the old files — which is also the rollback plan |
//!
//! [`session_store`] is deliberately kept: it is the importer's reader and
//! the reference the differential tests measure the port against.

pub mod agent_loader;
pub mod agent_store;
pub mod approvals;
pub mod cancel;
pub mod chat_tools;
pub mod conv_store;
pub mod db;
pub mod environment;
pub mod allowlist;
pub mod group;
pub mod harness;
pub mod http;
pub mod identity;
pub mod mcp;
pub mod memory;
pub mod migrate;
pub mod mcp_presets;
pub mod media;
pub mod project;
pub mod queue;
pub mod relevance;
pub mod sandbox;
pub mod serve;
pub mod projects;
pub mod settings;
pub mod session_props;
pub mod session_store;
