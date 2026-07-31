# angela

Thin harness around [agl-ai](../agl): **MCP-only tools**, sessions, allowlist + approval, CLI + library.

Always uses a **real** AGL model. The harness directs the model (system prompt, tools, events, abort, policy) — it does not simulate one.

---

## Install CLI

```bash
cd /workspace/angela && bun link   # → ~/.bun/bin/angela
angela --version
```

---

## Project layout

Default **project root** is the directory you run `angela` from (`cwd`).  
Override with **`-c` / `--config <path>`**.

```
<project>/
  .angela/
    agents/
      my-agent.coffee           # angela -a my-agent
    sessions/
      <id>.json                 # session state
      <id>.jsonl                # inference / tool event log
```

| Path | Role |
|------|------|
| `.angela/agents/` | Agent definition files (`-a`) |
| `.angela/sessions/<id>.json` | Session **metadata only** (agent, model, turns, status — no transcript) |
| `.angela/sessions/<id>.jsonl` | Append-only **wrapped** events (context window + provider I/O + tools) |
| project root | Default jail for local `file-io` / just-bash (`MCP_FILE_ROOT`) |

#### Session JSONL event shape

Each line is a **wrapper** around a verbatim payload (telemetry stays outside `payload`):

```json
{
  "ts": "2026-07-30T01:00:00.000Z",
  "event_type": "context_window | provider_request | provider_response | tool_call | tool_result | …",
  "duration_ms": 412.5,
  "payload": { }
}
```

Line order is the event sequence; session id is the filename (`<id>.jsonl`), not repeated per line.

| `event_type` | `payload` (verbatim / unpolluted) |
|--------------|-------------------------------------|
| `context_window` | Chat message or messages snapshot for the model |
| `provider_request` | Exact body passed to the AI provider `inference()` |
| `provider_response` | Exact provider response (or error object); `duration_ms` on wrapper |
| `tool_call` / `tool_result` / `policy` | Harness event objects |
| `session` | Lifecycle (`created`, `turn_complete`, `aborted`) |

```bash
cd /path/to/project
angela -a my-agent -t

angela clean                  # delete .angela/sessions/*
```

---

## Defining an agent for angela

Agent modules live under **`<project>/.angela/agents/`**, not inside the angela package.

### CoffeeScript (recommended)

```coffee
# <project>/.angela/agents/my-agent.coffee

module.exports = (ctx) ->
  # ctx: { presets, env, agentPath, projectRoot, …plugin extras }
  name: 'my-agent'
  description: 'optional blurb'
  model: 'lm-studio:google/gemma-4-12b-qat'   # optional; CLI -m overrides
  mcp: [                          # preset names and/or raw MCP objects
    'file-io'
    'term-local'
  ]
  system: '''
    You are …
  '''
  allowlist: '''
    file_io__list_dir
    term_local__bash:ls.*
  '''
  policyMode: 'ask'               # ask | strict | open
  agentsMd:                       # optional — append AGENTS.md to system prompt
    path: './AGENTS.md'
    via: 'local'
  # prepare: (def, ctx) -> def    # optional async hook
```

### JavaScript / ESM

```js
// <project>/.angela/agents/my-agent.mjs
export default function createAgent(ctx) {
  return {
    name: 'my-agent',
    model: 'lm-studio:google/gemma-4-12b-qat',
    mcp: ['file-io', 'term-local'],
    system: 'You are …',
  };
}
```

### Load paths for `-a <name>`

1. `<root>/.angela/agents/<name>.coffee` (also `.mjs` / `.js`)
2. `<root>/agents/<name>…` (legacy convenience)
3. Absolute path or `./relative` under project root

### MCP presets (core)

| Name | Purpose |
|------|---------|
| `file-io` | Local files (jailed to project root) |
| `term-local` | Local just-bash + host `exec` |

Optional **private plugins** under `angela/plugins/*` (gitignored) may register additional presets, model hosts, and AGENTS.md backends. See sibling `angela-plugins` if you maintain private extensions.

CLI flags (`-m`, `--mcp`, `--approve*`, `--allowlist`, `-s`, …) **always override** the agent file when set. Effective config and which keys were overridden are stored on the session metadata JSON (`config` + `overrides`).

---

## CLI quick reference

```bash
angela [options] [prompt...]
angela clean [-c path]

  -c, --config <path>      project root (default: cwd)
  -a, --agent <name|file>  load .angela/agents/…
  -m, --model <spec>
  -s, --system <text>
  --mcp <name>             repeatable preset
  -t, --tui                OpenTUI (welcome if no prompt; live stream if prompt)
  --approve                y/n/a (implied with -t)
  --approve=auto|deny
  -j, --jsonl
```

### OpenTUI

```bash
angela -a my-agent -t
angela -t --mcp file-io --mcp term-local -m lm-studio:google/gemma-4-12b-qat "hello"
```

Approvals: **y** allow · **n** deny · **a** always.

---

## Library API

```js
import { Angela, scriptedApprover, DEFAULT_LUCY_ALLOWLIST } from 'angela';

const harness = await Angela.create({
  projectRoot: process.cwd(),
  model: 'lm-studio:google/gemma-4-12b-qat',
  allowlist: DEFAULT_LUCY_ALLOWLIST,
  onApproval: scriptedApprover(['allow']),
  policyMode: 'ask',
  mcp: [/* resolved MCP entries */],
  onEvent: (ev) => console.log(ev),
});

const session = await harness.session.create({ title: 'cli' });
const result = await session.run({ prompt: 'list files' });
await harness.close();
// sessions written under <projectRoot>/.angela/sessions/
```

---

## Tests

```bash
cd /workspace/angela && bun install && bun test
```

Live model tests need LM Studio (or override `FAV_LOCAL_LLM`).
