# angela-rs — port status

Rust port of the Bun/JS angela at `/workspace/angela`, built on the `agl`
crate (`../agl-rs`).

```
/workspace/angela      JavaScript (Bun) — the reference implementation
/workspace/angela-rs   this crate
```

`cargo test` — **81 passing, 0 failing** (70 unit, 9 cross-compat, 2 live end-to-end).

## Why the JS stays the reference

The on-disk session format and the HTTP contract are both consumed by shipped
clients. Where the JS behaviour is surprising, this port reproduces the
surprise and records why, rather than quietly "fixing" it.

## What is ported

| Module | From | Notes |
|---|---|---|
| `src/project.rs` | `project.mjs` | Project root, `.angela/{agents,sessions}` layout, agent search order. |
| `src/http.rs` | (new) | A deliberately small HTTP/1.1 server: `std::net` + a thread per connection, no framework. Chunked streaming, because the chat endpoint must emit cards as the model produces them. No keep-alive, HTTP/2, TLS or compression — angela uses none of them. |
| `src/mcp.rs` | `mcp-client.mjs` | JSON-RPC over a child process's stdio, both framings (newline for the file-io kit, LSP `Content-Length` for the official SDK). Reader thread + channels, since the turn loop is blocking. The MCP **servers are not ported** — they run as-is from `/workspace/mcp/*`. |
| `src/mcp_presets.rs` | `mcp-presets.mjs` | The two local presets, `file-io` and `term-local`, jailed to the project root. Tool names are prefixed `file_io__read_file` so two servers cannot collide and existing `allowlist` entries carry over. |
| `src/agent_loader.rs` | `agent-loader.mjs` | Declarative **JSON** definitions in `.angela/agents/<name>.json`, with defaults, `toolsEnabled` / `allowlist` as array-or-multiline-string, and `${VAR}` / `${VAR:-fallback}` interpolation. |
| `src/allowlist.rs` | `allowlist.mjs` | Deny-by-default tool gate, with the small case-insensitive anchored regex engine (`.` `*` `+` `?` `[]` `{m,n}` `\b`) ported atom for atom. Checked against the JS by a differential test. |
| `src/harness.rs` | `index.mjs` (turn loop) | Resolve/create the tab's session, title it from the first user message, persist the prompt **before** inference, hydrate the agent from the log, stream deltas, persist reasoning + reply, record the turn. |
| `src/serve.rs` | `serve.mjs` (partial) | `/api/health`, `/api/agents`, `/api/sessions` (list + clean), `/api/sessions/:id` (get + delete), `/api/session/new`, and `reconstruct_messages_from_log` — the UI transcript shape. Unported routes answer **501 with the route name**, so the UI sees a clear signal instead of a 404 it would read as a routing bug. |
| `src/bin/angela-serve.rs` | `serve.mjs` entry | Binds the surface. Defaults match the JS (port 8763, cwd as project root), so it drops in behind `chat-server`'s existing `/api/*` proxy with no front-end change. |
| `src/session_store.rs` | `session-store.mjs` | Session create/load/save/`update_meta`, append-only JSONL events, `fold_event_metas`, turn + status tracking, listing newest-first, delete/clean with traversal refusal, plus `build_context_window_items` and `list_visible_context_messages` — the log→transcript bridge the chat UI renders from. |

### Cross-compatibility is tested against real JS output

`tests/xcompat.rs` runs against a fixture produced by executing the **actual**
`angela/src/session-store.mjs` under Bun — not against an assumption about its
format. It asserts the Rust store reads a JS-written session (metadata,
event sequence, `duration_ms` numbers, the `context_meta` fold, and the
append-only guarantee that the original line is never rewritten), and that
Rust can append to a JS session without breaking it. Regenerate the fixture
with `tests/fixtures/REGENERATE.md`.

### The differential test earned its keep

`build_context_window_items` is checked by running the **real JS** over a log
that exercises every item shape and every skip rule, then comparing the JSON
row by row. It caught two divergences that unit tests written from my reading
of the source would have missed:

1. `explanation` is `p.explanation || null` on tool rows, so the key is
   **always present** there (null when absent) — but plain and reasoning rows
   never set it at all. Three states, so the field is `Option<Option<String>>`:
   absent / present-null / present-valued.
2. `ok` and `denied` are bare `p.ok` / `p.denied`, which simply vanish when
   undefined — the opposite rule to `explanation`, on the same row.

### Verified cross-stack, not just cross-format

Beyond the fixture tests, the running Rust server was checked against the
running JS one:

- `/api/health`, `/api/sessions` and `/api/agents` return the **same keys in
  the same order** as the live JS angela on `:8763`.
- A session written by the **real JS `SessionStore`** into the Rust server's
  project root is then listed and fully reconstructed by the **Rust HTTP
  server** — all five card kinds (`user`, `reasoning`, `tool_call`,
  `tool_result`, `assistant`) in log order, the hidden row still hidden, and
  the `provider_messages_snapshot` correctly absent.

### A turn runs end to end, in Rust

`POST /api/chat` drives agl and streams the protocol the SPA expects:

```
{"type":"session","tabId":"1","sessionId":"…","model":"llama-server:default",…}
{"type":"reasoning_delta","text":"The user wants me to…"}
{"type":"assistant_delta","text":"hello from rust"}
{"type":"card","card":{"kind":"reasoning","text":"…","eventId":"e79024b43"}}
{"type":"usage","prompt_tokens":29,"total_tokens":65}
{"type":"card","card":{"kind":"assistant","text":"hello from rust",
                       "telemetry":{…},"streaming":false}}
{"type":"done","ok":true,"sessionId":"…"}
```

Verified against the live llama-server: deltas arrive incrementally, the
session is titled from the first user message, both turns land on **one**
session, and a second turn on the same tab recalls the first. Pinned by
`tests/turn_live.rs`, which skips rather than fails when no endpoint is up.

Two ordering choices worth stating:

- The user's message is persisted **before** inference runs, so a model
  failure still leaves the prompt in the transcript.
- The agent is hydrated from the **log** rather than from a live in-process
  agent, which is what `build_context_window_items` was written for
  ("session open / resume") — so a server restart does not amnesia the
  conversation. The JS keeps a live harness per tab *and* can rehydrate; this
  port starts from the durable half.

### Agent definitions: `.coffee` → JSON

Rust cannot execute the `.coffee` modules, so definitions became JSON. The
format was chosen **after auditing the real ones** (`gl1/lucy`, `gl1/coder`,
`blz`, `ada`): their CoffeeScript is almost entirely environment-variable
fallback chains — `process.env.LUCY_MODEL or process.env.COWORK_MODEL or ''`
— plus a little array de-duplication. So string values support
`${VAR}` and `${VAR:-fallback}`, chainable, which covers that idiom directly.

JSON rather than YAML because `serde_json` is already a dependency and agl's
vendored YAML subset **cannot express lists at all** — it skips every line
without a colon, so `- file-io` would vanish silently.

Not carried over: the `prepare(def, ctx)` hook and any other arbitrary code.
That is the real cost of the decision; a definition needing it must move the
logic into the harness.

One trap this introduced and the tests now pin: `.json` collides with
ordinary project files in a way `.coffee` never did. The JS search path
includes the project root, so the first run listed **`rustex1.json` as an
agent named "chat"**. Definitions must now live in a dedicated agents
directory.

### Tools, end to end

A turn registers every tool the connected MCP servers advertise, and agl
gained a `TurnEvent` observer so a tool call reaches the transcript **when it
runs** rather than after the turn settles — a long shell command would
otherwise look like a hang. Each call and result is persisted as a
`tool_call` / `tool_result` event before it is emitted.

Verified against the live stack: 9 tools registered (7 `file-io`, 2
`term-local`); `file_io__list_dir` and `file_io__read_file` return real
listings and file contents; `term_local__exec` runs real binaries. A failed
call comes back as **text the model can read**, not an aborted turn — which
is how, when `term_local__bash` (an emulated shell with a builtin set) lacked
`uname`, the model recovered by itself via `term_local__exec`.

Two sharp edges found and fixed while wiring it:
- the term sandbox needs `MCP_TERM_PATH`; without it even an allow-listed
  command is "command not found";
- a server that fails to start is reported as an error card and skipped, not
  fatal — losing shell access should not cost you the chat.

### The allowlist is enforced

Every tool call is gated **before** it reaches the MCP server. The properties
that matter:

- **Deny by default.** Empty or whitespace-only text permits nothing. A
  missing allowlist must never mean "allow everything".
- **A malformed pattern denies**, rather than collapsing into a wildcard.
- Terminal tools (`term_*`, `*__bash`, `*__exec`) match on the **command
  line**; everything else matches on the tool name alone.
- A denied call returns text the model can read ("denied: … is not on this
  session's allowlist"), so it can take a different route instead of the turn
  failing. It is recorded with `ok: false, denied: true`, so the UI renders it
  as blocked rather than successful.
- `policyMode: "auto"` is the documented escape hatch that disables
  enforcement for a trusted project; the default `ask` enforces.

Because this is a security boundary, agreement with the JS is checked by a
**differential test** (`tests/allowlist_xcompat.rs`) over 30 cases spanning
exact names, wildcards, anchoring, character classes, counts, malformed
patterns and terminal command matching — all 30 agree. A second test pins the
deny-by-default properties directly so they cannot regress quietly.

Verified live: `file_io__read_file` ran and returned real contents, while
`file_io__write_file` was blocked and **no file was created** on disk.

### Allowlist editing from the UI

`POST /api/allowlist` persists the session's override, and
`GET /api/sessions/:id` returns the effective text so the editor can seed
itself. `overridden: false` **clears** the override so the agent default
applies again, rather than storing an empty string — which would mean "deny
everything" and silently disable every tool.

### Behaviours reproduced deliberately

- A row with no explicit `meta.visible` is **visible**.
- `context_meta` folds forward, last-write-wins, and `visible` is coerced with
  JS truthiness (`Boolean(patch.visible)`), so `"yes"` becomes `true`.
- A `context_meta` payload carrying only id pointers is **not** applied as an
  empty patch.
- A non-finite `duration_ms` is stored as `null`.
- `visible` in `build_context_window_items` is `!(meta.visible === false)` —
  strict, so only an explicit boolean false hides a row. (The fold writes a
  real boolean, so the two rules agree after folding.)
- A `tool_result` with null/absent `content` falls back to JSON-encoding the
  **whole payload**, not an empty string.
- A string `args` on a tool call passes through verbatim; anything else is
  JSON-encoded, and a missing value becomes `{}`.
- Full-window `provider_messages_snapshot` rows are debug telemetry and are
  never used as a transcript seed.
- **Session ordering resolves only to the second.** Ids embed the timestamp as
  `YYYY-MM-DDTHH-MM-SS` plus three random bytes, and listing sorts by
  filename — so sessions created within one second tie-break on randomness,
  not time. The JS `newId()` has identical precision, so this is faithful
  rather than a defect. Pinned by `ids_only_order_to_the_second`.
- A session list title is `title || agent || id` — the JS fallback chain.
- An unparseable JSONL line is skipped, not fatal — a half-written final line
  must not cost you the session.
- `delete` refuses ids containing `..` or a path separator: the id arrives
  from an HTTP route, so it is a traversal vector.

## What is NOT ported yet

- **`session-store.mjs` remainder**: `reconstructProviderRequest`,
  `toolCallResultPairs`, `visibilityGroupEventIds`, `updateEventsMeta`,
  `extractUsage`/`extractFinishReason`/`extractAssistantText`/
  `extractReasoningContent`, `instrumentProviderLogging`.
- **`serve.mjs` remainder** — the routes that need the harness or the
  approval queue: `POST /api/chat` (the NDJSON card stream — the big one),
  `/api/approve`, `/api/abort`, `/api/allowlist`, `/api/event/meta`,
  `GET /api/pending`, `/api/context-window`. Each currently answers 501.
  `/api/agents` returns `[]` until the agent loader lands.
- **Interactive approvals.** The allowlist is enforced, but a call that is
  *not* allowed is refused outright rather than prompting. `ApprovalQueue` and
  `policy.mjs`'s ask-the-user path are unported, so `/api/approve`,
  `/api/pending` and `/api/abort` remain 501 and `approval_needed` is never
  emitted. Today the allowlist is the whole policy: on the list it runs, off
  the list it is denied.
- **MCP beyond the two local presets** (`context-loop`, `todo`, the `rsh`
  remote pair, HTTP transport).
- **Interrupting a running turn.** The JS aborts a prior in-flight turn when a
  new prompt arrives on the same tab; this port has no abort handle yet.
- **`index.mjs`** (1924 lines) — the harness: the turn loop that drives agl
  and emits cards.
- **`agent-loader.mjs`** — agent definitions. Decision taken: convert the
  `.coffee` definitions to a declarative format Rust reads natively. Needs an
  audit of what the existing definitions actually do before the format is
  fixed.
- `allowlist.mjs` / `policy.mjs`, the MCP client stack, `approval-queue.mjs`,
  `model-spec.mjs`, `plugins.mjs`, `tui.mjs`.

## Card kinds the chat UI consumes

From `serve.mjs`, the NDJSON stream sends `{type: 'card'|'event'|'done'}`,
where card kinds are `assistant`, `reasoning`, `tool_call`, `tool_result`,
`approval_needed`, `allowlist_updated`. These map 1:1 onto the storybook card
components already built in `psicom/chat`.
