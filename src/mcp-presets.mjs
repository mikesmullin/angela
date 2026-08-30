/**
 * Named MCP server presets for CLI + agent definition files.
 * Core package only ships local servers. Remote presets come from plugins.
 *
 * MCP script discovery (first hit wins) — no hard-coded machine paths required:
 *   1. $ANGELA_MCP_ROOT/{file-io,term-local,context-loop,todo}/server.mjs
 *   2. Sibling monorepo: <angela>/../mcp/{file-io,term-local,todo}/server.mjs
 *   3. Standalone packages: <angela>/../mcp-file-io/server.mjs,
 *      <angela>/../mcp-context-loop/server.mjs, <angela>/../mcp-todo/server.mjs
 *   4. node_modules: @mikesmullin/mcp-file-io or mcp-file-io
 *   5. Legacy: /workspace/mcp (devcontainer convenience only)
 */
import { resolve, join, dirname } from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { mergePluginMcpPresets, getPlugins } from './plugins.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
/** Package root of the angela install (…/angela), even when bun-linked. */
const ANGELA_PKG_ROOT = resolve(__dirname, '..');
const require = createRequire(import.meta.url);

/**
 * Resolve a script under several candidate locations.
 * @param {string[]} relPaths paths relative to a "mcp root" or absolute candidates
 * @returns {string|null}
 */
function firstExisting(candidates) {
  for (const c of candidates) {
    if (c && existsSync(c)) return c;
  }
  return null;
}

/**
 * Candidate directories that may contain file-io/ and term-local/ subdirs.
 * @returns {string[]}
 */
export function mcpRootCandidates() {
  const out = [];
  const add = (p) => {
    if (p && !out.includes(p)) out.push(p);
  };

  if (process.env.ANGELA_MCP_ROOT) add(resolve(process.env.ANGELA_MCP_ROOT));

  // Sibling of the angela package: ~/src/angela + ~/src/mcp
  add(resolve(ANGELA_PKG_ROOT, '../mcp'));
  // Nested under angela (optional vendoring)
  add(resolve(ANGELA_PKG_ROOT, 'mcp'));
  // Cwd-relative monorepo (running from sheets/ or workspace root)
  add(resolve(process.cwd(), 'mcp'));
  add(resolve(process.cwd(), '../mcp'));
  add(resolve(process.cwd(), '../../mcp'));
  // Devcontainer / this workspace layout
  add('/workspace/mcp');

  return out;
}

/**
 * Default monorepo-style root (directory that contains file-io/, term-local/, …).
 * Prefer {@link resolveMcpServerScript} for actual server paths.
 */
export function defaultMcpRoot() {
  for (const root of mcpRootCandidates()) {
    if (existsSync(join(root, 'file-io/server.mjs')) || existsSync(join(root, 'term-local/server.mjs'))) {
      return root;
    }
  }
  // Best-effort default for error messages / plugins even if not installed yet.
  return process.env.ANGELA_MCP_ROOT
    ? resolve(process.env.ANGELA_MCP_ROOT)
    : resolve(ANGELA_PKG_ROOT, '../mcp');
}

/**
 * Resolve the absolute path to an MCP server entry script.
 * @param {'file-io' | 'term-local' | string} preset
 * @returns {string|null}
 */
export function resolveMcpServerScript(preset) {
  const candidates = [];

  if (preset === 'file-io') {
    for (const root of mcpRootCandidates()) {
      candidates.push(join(root, 'file-io/server.mjs'));
    }
    // Standalone package layouts (server.mjs at package root)
    candidates.push(resolve(ANGELA_PKG_ROOT, '../mcp-file-io/server.mjs'));
    candidates.push(resolve(process.cwd(), '../mcp-file-io/server.mjs'));
    candidates.push(resolve(process.cwd(), 'node_modules/@mikesmullin/mcp-file-io/server.mjs'));
    candidates.push(resolve(process.cwd(), 'node_modules/mcp-file-io/server.mjs'));
    candidates.push(resolve(ANGELA_PKG_ROOT, 'node_modules/@mikesmullin/mcp-file-io/server.mjs'));
    candidates.push(resolve(ANGELA_PKG_ROOT, 'node_modules/mcp-file-io/server.mjs'));
    // require.resolve when package is installed
    for (const name of ['@mikesmullin/mcp-file-io/server.mjs', 'mcp-file-io/server.mjs', '@mikesmullin/mcp-file-io', 'mcp-file-io']) {
      try {
        const resolved = require.resolve(name, { paths: [process.cwd(), ANGELA_PKG_ROOT] });
        if (resolved.endsWith('.mjs') || resolved.endsWith('.js')) candidates.push(resolved);
        else candidates.push(join(dirname(resolved), 'server.mjs'));
      } catch {
        /* not installed */
      }
    }
  } else if (preset === 'term-local') {
    for (const root of mcpRootCandidates()) {
      candidates.push(join(root, 'term-local/server.mjs'));
    }
    candidates.push(resolve(ANGELA_PKG_ROOT, '../mcp-term-local/server.mjs'));
  } else if (preset === 'context-loop') {
    for (const root of mcpRootCandidates()) {
      candidates.push(join(root, 'context-loop/server.mjs'));
    }
    candidates.push(resolve(ANGELA_PKG_ROOT, '../mcp-context-loop/server.mjs'));
    candidates.push(resolve(process.cwd(), '../mcp-context-loop/server.mjs'));
    candidates.push(
      resolve(process.cwd(), 'node_modules/@mikesmullin/mcp-context-loop/server.mjs'),
    );
    candidates.push(
      resolve(ANGELA_PKG_ROOT, 'node_modules/@mikesmullin/mcp-context-loop/server.mjs'),
    );
  } else if (preset === 'todo') {
    for (const root of mcpRootCandidates()) {
      candidates.push(join(root, 'todo/server.mjs'));
    }
    candidates.push(resolve(ANGELA_PKG_ROOT, '../mcp-todo/server.mjs'));
    candidates.push(resolve(process.cwd(), '../mcp-todo/server.mjs'));
    candidates.push(
      resolve(process.cwd(), 'node_modules/@mikesmullin/mcp-todo/server.mjs'),
    );
    candidates.push(
      resolve(ANGELA_PKG_ROOT, 'node_modules/@mikesmullin/mcp-todo/server.mjs'),
    );
  } else {
    // Generic: $root/<preset>/server.mjs
    for (const root of mcpRootCandidates()) {
      candidates.push(join(root, preset, 'server.mjs'));
    }
  }

  return firstExisting(candidates);
}

/**
 * @param {string} [mcpRoot] unused for script path when discovery succeeds; kept for plugins API
 * @param {{
 *   projectRoot?: string,
 *   path?: string | null,
 *   env?: Record<string, string> | null,
 *   cwd?: string | null,
 * }} [opts]
 *   projectRoot → jail for file-io / just-bash
 *   path / env / cwd → agent coffee (PATH, extras, default cwd for term tools)
 */
export function buildMcpPresets(mcpRoot = defaultMcpRoot(), opts = {}) {
  const bun = process.execPath;
  // Local sandbox root: env override, else project root (-c / cwd)
  const jail =
    process.env.MCP_FILE_ROOT ||
    process.env.MCP_TERM_JB_ROOT ||
    opts.projectRoot ||
    process.cwd();

  const agentPath =
    opts.path || process.env.MCP_TERM_PATH || '';
  const agentEnv = opts.env && typeof opts.env === 'object' ? opts.env : {};
  const envPairs = Object.entries(agentEnv)
    .filter(([k, v]) => k && v != null && k !== 'PATH')
    .map(([k, v]) => `${k}=${v}`)
    .join('\n');

  const termLocalEnv = {
    MCP_TERM_CWD: process.env.MCP_TERM_CWD || jail,
    MCP_TERM_JB_ROOT: process.env.MCP_TERM_JB_ROOT || jail,
    MCP_TERM_ALLOW:
      process.env.MCP_TERM_ALLOW ||
      'echo,pwd,ls,true,uname,date,cat,head,wc,which,bun,node',
  };
  if (agentPath) termLocalEnv.MCP_TERM_PATH = agentPath;
  if (envPairs) termLocalEnv.MCP_TERM_ENV = envPairs;

  const fileIoScript = resolveMcpServerScript('file-io');
  const termLocalScript = resolveMcpServerScript('term-local');
  const contextLoopScript = resolveMcpServerScript('context-loop');
  const todoScript = resolveMcpServerScript('todo');

  const base = {
    'file-io': {
      name: 'file-io',
      command: bun,
      // May be null until installed — resolveMcpList reports a helpful error.
      args: [fileIoScript || join(mcpRoot, 'file-io/server.mjs')],
      env: {
        MCP_FILE_ROOT: jail,
      },
    },
    'term-local': {
      name: 'term-local',
      command: bun,
      args: [termLocalScript || join(mcpRoot, 'term-local/server.mjs')],
      env: termLocalEnv,
    },
    // Opt-in CONTEXT LOOP tools (goto / loop_start_marker). Not in default
    // agent mcp lists or DEFAULT_LUCY_ALLOWLIST. prefix:false so the model
    // still sees tool names `goto` and `loop_start_marker`.
    'context-loop': {
      name: 'context-loop',
      command: bun,
      args: [
        contextLoopScript || join(mcpRoot, 'context-loop/server.mjs'),
      ],
      env: {},
      prefix: false,
    },
    // Opt-in session checklist (todo_read / todo_write). prefix:false so the
    // model sees the Copilot/Claude-style names. In-memory unless MCP_TODO_PATH.
    todo: {
      name: 'todo',
      command: bun,
      args: [todoScript || join(mcpRoot, 'todo/server.mjs')],
      env: {},
      prefix: false,
    },
  };

  // Plugins may add remote MCP presets when installed under plugins/
  if (getPlugins().length) {
    return mergePluginMcpPresets(base, mcpRoot, opts);
  }
  return base;
}

function missingServerHelp(preset, attemptedPath) {
  const roots = mcpRootCandidates().join('\n    - ');
  return [
    `MCP server script missing for preset "${preset}": ${attemptedPath}`,
    '',
    'Install options (pick one):',
    '',
    '  A) Monorepo-style sibling (recommended with bun link):',
    '       git clone git@github.com:mikesmullin/mcp.git ../mcp',
    '       # expects: ../mcp/file-io/server.mjs  (and term-local/ if needed)',
    '',
    '  B) file-io only:',
    '       git clone git@github.com:mikesmullin/mcp-file-io.git ../mcp-file-io',
    '',
    '  C) Point explicitly:',
    '       export ANGELA_MCP_ROOT=/absolute/path/to/mcp',
    '',
    'Searched MCP roots:',
    `    - ${roots}`,
  ].join('\n');
}

/**
 * Resolve preset names and/or raw MCP entry objects into connect configs.
 * @param {Array<string|object>} list
 * @param {Record<string, object>} [presets]
 */
export function resolveMcpList(list = [], presets = buildMcpPresets()) {
  const out = [];
  for (const item of list) {
    if (!item) continue;
    if (typeof item === 'string') {
      const p = presets[item];
      if (!p) {
        throw new Error(
          `Unknown MCP preset "${item}". Known: ${Object.keys(presets).join(', ') || '(none)'}` +
            (getPlugins().length
              ? ''
              : ' — install optional plugins for remote presets'),
        );
      }
      // Re-resolve script path in case presets were built before install.
      const script =
        (item === 'file-io' ||
        item === 'term-local' ||
        item === 'context-loop' ||
        item === 'todo'
          ? resolveMcpServerScript(item)
          : null) || p.args?.[0];
      if (!script || !existsSync(script)) {
        throw new Error(missingServerHelp(item, script || p.args?.[0] || '(none)'));
      }
      out.push({ ...p, args: [script, ...(p.args || []).slice(1)], env: { ...p.env } });
      continue;
    }
    if (typeof item === 'object' && item.name && item.command) {
      out.push({
        name: item.name,
        command: item.command,
        args: item.args || [],
        env: { ...(item.env || {}) },
        cwd: item.cwd,
      });
      continue;
    }
    throw new Error(`Invalid MCP entry: ${JSON.stringify(item)}`);
  }
  return out;
}

export default { buildMcpPresets, resolveMcpList, defaultMcpRoot, resolveMcpServerScript, mcpRootCandidates };
