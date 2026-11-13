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
//! | `src/session-store.mjs` | [`session_store`] | in progress |
//! | `src/serve.mjs` | `serve` | todo |
//! | `src/index.mjs` | `harness` | todo |
//! | `src/agent-loader.mjs` | `agent_loader` | todo (`.coffee` → declarative) |
//! | `src/allowlist.mjs`, `policy.mjs` | `policy` | todo |
//! | `src/mcp-*.mjs` | `mcp` | todo |

pub mod agent_loader;
pub mod allowlist;
pub mod harness;
pub mod http;
pub mod mcp;
pub mod mcp_presets;
pub mod project;
pub mod serve;
pub mod session_store;
