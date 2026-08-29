#!/usr/bin/env bun
/**
 * angela CLI
 *
 *   cd /path/to/project && angela -a my-agent -t
 *   angela -c /path/to/project -a my-agent -t "…"
 *   angela clean
 */
import {
  Angela,
  autoApprover,
  DEFAULT_LUCY_ALLOWLIST,
  loadAgent,
  buildMcpPresets,
  resolveMcpList,
  resolveProjectRoot,
  ensureAngelaLayout,
  SessionStore,
} from '../src/index.mjs';
import { defaultMcpRoot } from '../src/mcp-presets.mjs';

function usage(code = 0) {
  const text = `angela — AGL harness CLI

Usage:
  angela [options] [prompt...]
  angela serve [-c path] [--port 8763]   HTTP chat API for gl1 / clients
  angela clean [-c path]                 remove .angela/sessions/*

Project root (-c / cwd):
  Controls where .angela/ lives (agents + sessions) and the default
  local sandbox jail for file-io / just-bash (MCP_FILE_ROOT).

  Default: current working directory
  Override: -c / --config <path>

Agents (-a):
  Loaded from <root>/.angela/agents/<name>.coffee (etc.)

Options:
  -c, --config <path>      project root (default: cwd)
  -a, --agent <name|file>  load agent from .angela/agents/
  -m, --model <spec>       override model (provider:model or host:provider:model)
                           e.g. lm-studio:gemma-4 | local:copilot:gpt-5.6-luna
  -s, --system <text>      override system prompt
  --mcp <name>             MCP preset (repeatable); overrides agent mcp if set
  -t, --tui                OpenTUI chat
  --approve                interactive approval
  --approve=auto|deny
  --allowlist <file>
  --port <n>               serve port (default 8763 / ANGELA_PORT)
  -j, --jsonl
  -h, --help
  -V, --version

Examples:
  cd /path/to/project && angela -a my-agent -t
  angela serve -c /path/to/project
  angela -c /path/to/project -a my-agent -t "list files in the project"
  angela clean -c /path/to/project
`;
  if (code === 0) console.log(text);
  else console.error(text);
  process.exit(code);
}

function parseArgs(argv) {
  const opts = {
    configPath: null,
    model: null,
    system: null,
    mcp: [],
    jsonl: false,
    approve: null,
    tui: false,
    allowlistFile: null,
    agent: null,
    promptParts: [],
    setModel: false,
    setSystem: false,
    setMcp: false,
    clean: false,
    serve: false,
    port: null,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '-h' || a === '--help') usage(0);
    if (a === '-V' || a === '--version') {
      console.log('angela 0.0.5');
      process.exit(0);
    }
    if (a === 'clean') {
      opts.clean = true;
      continue;
    }
    if (a === 'serve') {
      opts.serve = true;
      continue;
    }
    if (a === '--port') {
      opts.port = Number(argv[++i]);
      continue;
    }
    if (a === '-c' || a === '--config') {
      opts.configPath = argv[++i];
      continue;
    }
    if (a === '-a' || a === '--agent') {
      opts.agent = argv[++i];
      continue;
    }
    if (a === '-m' || a === '--model') {
      opts.model = argv[++i];
      opts.setModel = true;
      continue;
    }
    if (a === '-s' || a === '--system') {
      opts.system = argv[++i];
      opts.setSystem = true;
      continue;
    }
    if (a === '--mcp') {
      opts.mcp.push(argv[++i]);
      opts.setMcp = true;
      continue;
    }
    if (a === '-t' || a === '--tui') {
      opts.tui = true;
      continue;
    }
    if (a === '--approve') {
      opts.approve = true;
      continue;
    }
    if (a.startsWith('--approve=')) {
      opts.approve = a.slice('--approve='.length);
      continue;
    }
    if (a === '--allowlist') {
      opts.allowlistFile = argv[++i];
      continue;
    }
    if (a === '-j' || a === '--jsonl') {
      opts.jsonl = true;
      continue;
    }
    if (a.startsWith('-')) {
      console.error(`Unknown option: ${a}`);
      usage(1);
    }
    opts.promptParts.push(a);
  }
  opts.prompt = opts.promptParts.join(' ').trim();
  return opts;
}

function ansiApprover() {
  return async (req) => {
    const box = [
      '',
      '┌─ angela approval ─────────────────────────────────────',
      `│ id:      ${req.id}`,
      `│ tool:    ${req.tool}`,
      `│ command: ${(req.command || '').slice(0, 60)}`,
      '│  [y]allow  [n]deny  [a]always',
      '└──────────────────────────────────────────────────────',
      '> ',
    ].join('\n');
    process.stderr.write(box);
    const readline = await import('node:readline');
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stderr,
    });
    try {
      const answer = await new Promise((resolve) => rl.question('', resolve));
      const a = String(answer || '').trim().toLowerCase();
      if (a === 'y' || a === 'yes') return 'allow';
      if (a === 'a' || a === 'always') return 'always';
      return 'deny';
    } finally {
      rl.close();
    }
  };
}

/**
 * CLI flags always win over agent file when explicitly set.
 * Records which keys were overridden for session metadata.
 */
async function buildRunConfig(opts) {
  const projectRoot = resolveProjectRoot(opts.configPath);
  ensureAngelaLayout(projectRoot);
  console.error(`angela: project ${projectRoot}`);

  let agent = null;
  if (opts.agent) {
    agent = await loadAgent(opts.agent, { projectRoot });
    console.error(
      `angela: agent ${agent.name} from ${agent.path}` +
        (agent.meta?.agentsMd?.bytes
          ? ` (+AGENTS.md ${agent.meta.agentsMd.bytes}B)`
          : agent.meta?.agentsMd?.error
            ? ` (AGENTS.md: ${agent.meta.agentsMd.error})`
            : ''),
    );
  }

  const presets = buildMcpPresets(defaultMcpRoot(), { projectRoot });
  /** @type {Record<string, { from: string, to: any }>} */
  const overrides = {};

  // --- mcp ---
  let mcp = [];
  let mcpNames = [];
  if (opts.setMcp) {
    mcp = resolveMcpList(opts.mcp, presets);
    mcpNames = opts.mcp.slice();
    overrides.mcp = {
      from: agent?.mcp?.map((m) => m.name || m) || null,
      to: mcpNames,
    };
  } else if (agent?.mcp?.length) {
    mcp = agent.mcp;
    mcpNames = agent.mcp.map((m) => m.name);
  }

  // --- model ---
  let model;
  if (opts.setModel && opts.model) {
    model = opts.model;
    overrides.model = { from: agent?.model || null, to: model };
  } else {
    model =
      agent?.model ||
      process.env.FAV_LOCAL_LLM ||
      process.env.COWORK_MODEL ||
      'lm-studio:google/gemma-4-12b-qat';
  }

  // --- system ---
  let system;
  if (opts.setSystem && opts.system) {
    system = opts.system;
    overrides.system = {
      from: agent?.system ? `(${agent.system.length} chars)` : null,
      to: `(${system.length} chars)`,
    };
  } else {
    system =
      agent?.system ||
      'You are a helpful assistant. Use tools when needed. Prefer short answers.';
  }

  // --- allowlist ---
  let allowlist = agent?.allowlist ?? DEFAULT_LUCY_ALLOWLIST;
  let allowlistSource = agent?.allowlist
    ? 'agent'
    : 'default';
  if (opts.allowlistFile) {
    allowlist = await Bun.file(opts.allowlistFile).text();
    allowlistSource = `file:${opts.allowlistFile}`;
    overrides.allowlist = {
      from: agent?.allowlist ? 'agent' : 'default',
      to: allowlistSource,
    };
  }

  // --- approve / policy ---
  // CLI --approve* takes precedence over agent.policyMode for interactive mode.
  // agent.policyMode still applies as base unless we map approve modes.
  let policyMode = agent?.policyMode || 'ask';
  let approveMode = opts.approve; // null | true | 'auto' | 'deny' | 'stdin'
  if (opts.approve !== null && opts.approve !== undefined) {
    overrides.approve = {
      from: agent?.policyMode || 'ask',
      to: opts.approve === true ? true : opts.approve,
    };
    if (opts.approve === 'auto') {
      // auto-allow is still policyMode ask with auto approver; record as open-ish
      policyMode = 'ask';
    } else if (opts.approve === 'deny') {
      policyMode = 'strict';
    } else {
      policyMode = 'ask';
    }
  }

  if (Object.keys(overrides).length) {
    console.error(
      `angela: CLI overrides: ${Object.keys(overrides).join(', ')}`,
    );
  }

  return {
    projectRoot,
    agentName: agent?.name || null,
    agentFile: agent?.path || null,
    model,
    system,
    mcp,
    mcpNames,
    allowlist,
    allowlistSource,
    policyMode,
    approveMode,
    parallel_tools: agent?.parallel_tools ?? true,
    overrides: Object.keys(overrides).length ? overrides : null,
  };
}

async function runPlain(opts, cfg) {
  if (!opts.prompt) {
    console.error(
      'Missing prompt. Pass a prompt, or use -t/--tui for interactive chat.',
    );
    usage(1);
  }

  let onApproval = null;
  if (opts.approve === true || opts.approve === 'stdin') {
    onApproval = ansiApprover();
  } else if (opts.approve === 'auto') {
    onApproval = autoApprover();
  } else if (opts.approve === 'deny') {
    onApproval = async () => 'deny';
  }

  const onEvent = opts.jsonl
    ? (ev) => process.stderr.write(JSON.stringify(ev) + '\n')
    : (ev) => {
        if (ev.type === 'tool_call') {
          console.error(`→ ${ev.name}(${JSON.stringify(ev.args || {})})`);
        } else if (ev.type === 'tool_result') {
          const preview = String(ev.text || '')
            .slice(0, 200)
            .replace(/\n/g, ' ');
          console.error(
            `← ${ev.name} ${ev.ok === false ? 'ERR' : 'ok'}${ev.denied ? ' DENIED' : ''}: ${preview}`,
          );
        } else if (ev.type === 'approval_needed') {
          console.error(`? approval ${ev.tool}`);
        }
      };

  const harness = await Angela.create({
    projectRoot: cfg.projectRoot,
    agentName: cfg.agentName,
    agentFile: cfg.agentFile,
    model: cfg.model,
    system: cfg.system,
    mcp: cfg.mcp,
    allowlist: cfg.allowlist,
    allowlistSource: cfg.allowlistSource,
    onApproval,
    policyMode: cfg.policyMode,
    approveMode: cfg.approveMode,
    overrides: cfg.overrides,
    parallel_tools: cfg.parallel_tools,
    onEvent,
  });

  const session = await harness.session.create({
    title: cfg.agentName || 'cli',
    agent: cfg.agentName,
  });
  console.error(`angela: session ${session.id}`);

  const onSignal = async () => {
    console.error('\nangela: aborting…');
    try {
      await session.abort('SIGINT');
    } catch {
      /* ignore */
    }
    try {
      await harness.close();
    } catch {
      /* ignore */
    }
    process.exit(130);
  };
  process.on('SIGINT', onSignal);

  try {
    const result = await session.run({ prompt: opts.prompt });
    if (result?.command === 'new') {
      process.stdout.write(`new session ${result.sessionId}\n`);
      return;
    }
    const text =
      typeof result === 'string'
        ? result
        : result?.summary ||
          result?.choices?.[0]?.message?.content ||
          (result != null ? JSON.stringify(result) : '');
    process.stdout.write(String(text).trimEnd() + '\n');
  } finally {
    process.off('SIGINT', onSignal);
    await harness.close();
  }
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));

  if (opts.clean) {
    const root = resolveProjectRoot(opts.configPath);
    const store = new SessionStore(root);
    const n = store.clean();
    console.log(`angela clean: removed ${n} file(s) under ${store.dir}`);
    return;
  }

  if (opts.serve) {
    const { startServe } = await import('../src/serve.mjs');
    const root = resolveProjectRoot(opts.configPath);
    startServe({
      projectRoot: root,
      port: opts.port || Number(process.env.ANGELA_PORT || 8763),
      host: process.env.ANGELA_HOST || '127.0.0.1',
    });
    // Keep process alive (Bun.serve holds the event loop).
    return;
  }

  const cfg = await buildRunConfig(opts);

  let approveMode = opts.approve;
  if (opts.tui && approveMode == null) approveMode = true;

  if (opts.tui) {
    const { runTui } = await import('../src/tui.mjs');
    await runTui({
      projectRoot: cfg.projectRoot,
      model: cfg.model,
      system: cfg.system,
      mcp: cfg.mcp,
      allowlist: cfg.allowlist,
      allowlistSource: cfg.allowlistSource,
      policyMode: cfg.policyMode,
      initialPrompt: opts.prompt || '',
      approveMode,
      agentName: cfg.agentName,
      agentFile: cfg.agentFile,
      overrides: cfg.overrides,
    });
    return;
  }

  await runPlain(opts, cfg);
}

main().catch((err) => {
  console.error(err?.stack || err?.message || err);
  process.exit(1);
});
