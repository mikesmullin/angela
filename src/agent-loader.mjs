/**
 * Load angela agent definition files (.coffee / .mjs / .js).
 *
 * An agent module may export:
 *   - default function (ctx) => def | Promise<def>
 *   - default object def
 *   - named export createAgent / agent
 *
 * Agent def shape (all optional except name recommended):
 *   {
 *     name: 'my-agent',
 *     // model: optional — if omitted, AGL reads ~/.config/agl/config.yaml
 *     // default_model on every inference (see defaultAgentFields).
 *     model: 'lm-studio:google/gemma-4-12b-qat',
 *     system: '...',                 // base system prompt
 *     mcp: ['file-io', 'term-local'], // preset names and/or raw objects
 *     allowlist: '...',
 *     // toolsEnabled: which tools are registered for the LLM at inference time.
 *     // null/omit = all tools. string (multiline) or string[] = only those names.
 *     // UI overrides are authoritative over this default.
 *     // Distinct from allowlist (auto-approve); a tool can be enabled but still
 *     // require approval when not on the allowlist.
 *     toolsEnabled: 'file_io__read_file\\nterm_local__bash',
 *     policyMode: 'ask',
 *     parallel_tools: true,
 *     // Default cwd for term tools when the tool omits cwd
 *     cwd: process.cwd(),
 *     agentsMd: { path: './AGENTS.md', via: 'local' }, // plugins may add remote vias
 *     // Welcome-screen one-click prompts (cowork UI). Strings or { label, prompt }.
 *     starters: [
 *       'What tool call functions are available to you? List them by name and description',
 *       { label: 'Project overview', prompt: 'Summarize this project structure' },
 *     ],
 *     // optional hooks:
 *     prepare: async (def, ctx) => def,  // mutate after load (e.g. inject AGENTS.md)
 *   }
 *
 * materializeAgentDef merges the agent return value over defaultAgentFields()
 * so coffee files only need to declare overrides.
 */

/**
 * Reasonable defaults for agent defs. Agent coffee return values are merged
 * on top of this (agent keys win; omitted keys keep these defaults).
 *
 * Model omitted → empty string so AGL resolves ~/.config/agl/config.yaml
 * `default_model` on every inference (no process restart needed).
 *
 * @returns {{ model: string, policyMode: string, parallel_tools: boolean }}
 */
export function defaultAgentFields() {
  return {
    model: '',
    policyMode: 'ask',
    parallel_tools: true,
  };
}

/**
 * App-level default agent factory (same shape as .angela/agents/*.coffee exports).
 * Host apps (e.g. brain viz) register this once at startup via setDefaultAgent().
 * Used when:
 *   - no agent files exist under .angela/agents/ (listAgents → synthetic "default")
 *   - loadAgent('default') or loadAgent of a missing name
 *   - a file agent omits keys (system, allowlist, mcp, …) — file wins per-key
 *
 * @type {null|((ctx: object) => object|Promise<object>)|object}
 */
let _defaultAgentFactory = null;

/**
 * Register (or clear) the host app's default agent factory.
 * @param {((ctx: object) => object|Promise<object>)|object|null|undefined} factory
 */
export function setDefaultAgent(factory) {
  if (factory == null) {
    _defaultAgentFactory = null;
    return;
  }
  if (typeof factory !== 'function' && (typeof factory !== 'object' || Array.isArray(factory))) {
    throw new Error(
      'setDefaultAgent expects a function (ctx) => def, a def object, or null',
    );
  }
  _defaultAgentFactory = factory;
}

/** @returns {typeof _defaultAgentFactory} */
export function getDefaultAgentFactory() {
  return _defaultAgentFactory;
}

/**
 * Invoke the registered default agent factory (if any).
 * @param {object} [ctx]
 * @returns {Promise<object|null>} raw def (not materialized), or null
 */
export async function invokeDefaultAgent(ctx = {}) {
  const factory = _defaultAgentFactory;
  if (!factory) return null;
  const def = typeof factory === 'function' ? await factory(ctx) : factory;
  if (!def || typeof def !== 'object' || Array.isArray(def)) {
    throw new Error('default agent factory must return a definition object');
  }
  return { ...def };
}

/**
 * Build agent-loader ctx for factory invocation.
 */
function agentFactoryCtx(projectRoot, opts = {}, agentPath = null) {
  const root = resolveProjectRoot(projectRoot);
  return {
    presets: buildMcpPresets(opts.mcpRoot || defaultMcpRoot(), {
      projectRoot: root,
    }),
    env: process.env,
    agentPath,
    projectRoot: root,
    ...pluginAgentContext(),
  };
}

/**
 * Merge layer underlays: later objects win for defined keys only.
 * @param {...object|null|undefined} layers
 */
export function mergeAgentDefs(...layers) {
  const out = {};
  for (const layer of layers) {
    if (!layer || typeof layer !== 'object' || Array.isArray(layer)) continue;
    for (const [k, v] of Object.entries(layer)) {
      if (v !== undefined) out[k] = v;
    }
  }
  return out;
}

/**
 * Normalize agent.starters / starterPrompts for UI.
 * @param {unknown} raw
 * @returns {Array<{ label: string, prompt: string }>|null} null = use UI defaults
 */
export function normalizeStarters(raw) {
  if (raw == null) return null;
  const list = Array.isArray(raw) ? raw : [raw];
  const out = [];
  for (const item of list) {
    if (item == null) continue;
    if (typeof item === 'string') {
      const prompt = item.trim();
      if (!prompt) continue;
      out.push({ label: prompt, prompt });
      continue;
    }
    if (typeof item === 'object') {
      const prompt = String(
        item.prompt ?? item.text ?? item.message ?? item.value ?? '',
      ).trim();
      if (!prompt) continue;
      const label = String(item.label ?? item.title ?? item.name ?? prompt).trim();
      out.push({ label: label || prompt, prompt });
    }
  }
  return out.length ? out : null;
}

/**
 * Normalize agent.toolsEnabled for harness + UI.
 *
 * - `null` / `undefined` → `null` (all registered tools enabled at inference)
 * - string → tool names (one per line; `#` comments; commas ok); may be `[]`
 * - string[] → trimmed names (empty entries dropped); may be `[]`
 *
 * An explicit empty list means no tools are enabled (unlike null = all).
 *
 * @param {unknown} raw
 * @returns {string[]|null}
 */
export function normalizeToolsEnabled(raw) {
  if (raw == null) return null;
  if (typeof raw === 'string') {
    const names = [];
    for (const line of raw.split(/\r?\n/)) {
      const stripped = line.replace(/#.*$/, '').trim();
      if (!stripped) continue;
      for (const part of stripped.split(',')) {
        const n = part.trim();
        if (n) names.push(n);
      }
    }
    return names;
  }
  if (Array.isArray(raw)) {
    return raw.map((x) => String(x).trim()).filter(Boolean);
  }
  return null;
}

/**
 * Filter a full tool-name catalog by toolsEnabled selection.
 * @param {string[]} allNames
 * @param {string[]|null|undefined} enabled null/undefined = all
 * @returns {string[]}
 */
export function filterToolNames(allNames, enabled) {
  const all = Array.isArray(allNames) ? allNames.map(String) : [];
  if (enabled == null) return [...all];
  const set = new Set(enabled.map(String));
  return all.filter((n) => set.has(n));
}
import { resolve, join, dirname, isAbsolute, extname } from 'node:path';
import { existsSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { buildMcpPresets, resolveMcpList, defaultMcpRoot } from './mcp-presets.mjs';
import {
  applyPluginAgentsMd,
  pluginAgentContext,
  loadPlugins,
} from './plugins.mjs';
import { DEFAULT_LUCY_ALLOWLIST } from './allowlist.mjs';
import { agentSearchDirs, resolveProjectRoot } from './project.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

/**
 * Resolve -a my-agent / my-agent.coffee / ./path to an absolute path.
 * Primary: <projectRoot>/.angela/agents/
 *
 * @param {string} spec
 * @param {{ projectRoot?: string }} [opts]
 */
export function resolveAgentPath(spec, opts = {}) {
  if (!spec) throw new Error('agent path required');
  const raw = String(spec).trim();
  const projectRoot = resolveProjectRoot(opts.projectRoot);

  if (isAbsolute(raw) && existsSync(raw)) return raw;
  if (raw.startsWith('./') || raw.startsWith('../')) {
    const p = resolve(projectRoot, raw);
    if (existsSync(p)) return p;
  }

  const candidates = [];
  const bases = agentSearchDirs(projectRoot);

  const names = [];
  if (extname(raw)) {
    names.push(raw);
  } else {
    names.push(
      `${raw}.coffee`,
      `${raw}.mjs`,
      `${raw}.js`,
      join(raw, 'index.coffee'),
      join(raw, 'index.mjs'),
    );
  }

  for (const dir of bases) {
    for (const n of names) {
      candidates.push(join(dir, n));
    }
  }
  candidates.push(resolve(projectRoot, raw));

  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  throw new Error(
    `Agent not found: ${spec}\n` +
      `project root: ${projectRoot}\n` +
      `Searched:\n${candidates.map((c) => `  ${c}`).join('\n')}`,
  );
}

async function importAgentModule(absPath) {
  const ext = extname(absPath).toLowerCase();
  if (ext === '.coffee') {
    // Compile Coffee → CJS and evaluate (Bun's coffee import is not reliable here).
    const coffee = (await import('coffeescript')).default;
    const { readFileSync } = await import('node:fs');
    const src = readFileSync(absPath, 'utf8');
    const js = coffee.compile(src, { bare: true, header: false, filename: absPath });
    const module = { exports: {} };
    const exports = module.exports;
    // eslint-disable-next-line no-new-func
    const run = new Function('module', 'exports', 'require', `${js}\n;return module.exports;`);
    const exp = run(module, exports, require);
    if (typeof exp === 'function') return { default: exp };
    if (exp && typeof exp === 'object') {
      return {
        default: exp.default || exp.createAgent || exp.agent || exp.create || exp,
        ...exp,
      };
    }
    return { default: exp };
  }
  return import(pathToFileURL(absPath).href);
}

const AGENTS_CTX_HEADER =
  '\n\n# Project Context — The following project context files have been loaded and should be followed\n\n# AGENTS.md\n\n';

/**
 * Apply agentsMd config: local file, or plugin backends (optional private plugins).
 */
export async function applyAgentsMd(def) {
  const cfg = def.agentsMd;
  if (!cfg || !cfg.path) return def;
  const via = String(cfg.via || 'local').toLowerCase();

  // Built-in: local filesystem
  if (via === 'local' || via === 'fs' || via === 'file') {
    try {
      const { readFileSync } = await import('node:fs');
      const { resolve } = await import('node:path');
      const abs = resolve(String(cfg.path));
      const body = readFileSync(abs, 'utf8');
      const base = def.system || '';
      def.system = base + AGENTS_CTX_HEADER + body + '\n';
      def._agentsMdLoaded = { path: abs, bytes: body.length, via: 'local' };
    } catch (err) {
      console.error(
        `angela: AGENTS.md local load failed (continuing): ${err.message}`,
      );
      def._agentsMdLoaded = { path: cfg.path, error: err.message, via: 'local' };
    }
    return def;
  }

  // Optional plugins (e.g. remote backends)
  await loadPlugins();
  const before = def._agentsMdLoaded;
  def = await applyPluginAgentsMd(def);
  if (def._agentsMdLoaded && def._agentsMdLoaded !== before) return def;

  console.error(
    `angela: agentsMd.via=${via} not supported (no plugin handled it); skipping`,
  );
  return def;
}

/**
 * Normalize + resolve an agent definition into harness-ready config fields.
 */
export async function materializeAgentDef(
  def,
  { mcpRoot, projectRoot, skipAgentsMd = false } = {},
) {
  const root = resolveProjectRoot(projectRoot);
  // Agent return value overlays harness defaults (omit model → AGL default_model).
  const defaults = defaultAgentFields();
  let d = { ...defaults, ...(def && typeof def === 'object' ? def : {}) };
  // Empty string model is treated as "use default"
  if (d.model == null || String(d.model).trim() === '') {
    d.model = defaults.model;
  } else {
    d.model = String(d.model).trim();
  }

  // path / env / cwd from agent def (legacy lucy_* aliases still accepted)
  const agentPath =
    d.path || d.PATH || d.lucy_path || d.lucyPath || null;
  const agentEnv =
    d.env && typeof d.env === 'object' && !Array.isArray(d.env)
      ? { ...d.env }
      : {};
  const agentCwd =
    d.cwd ||
    d.default_cwd ||
    d.defaultCwd ||
    d.lucy_cwd ||
    d.lucyCwd ||
    null;

  await loadPlugins();
  const presets = buildMcpPresets(mcpRoot || defaultMcpRoot(), {
    projectRoot: root,
    path: agentPath,
    env: agentEnv,
    cwd: agentCwd,
  });

  if (typeof d.prepare === 'function') {
    d = {
      ...defaults,
      ...(await d.prepare(d, {
        presets,
        projectRoot: root,
        ...pluginAgentContext(),
      })),
    };
  }

  // Built-in AGENTS.md hook (skip for listAgents UI catalog)
  if (!skipAgentsMd) {
    d = await applyAgentsMd(d);
  }

  // Re-apply model default if prepare / agentsMd left it empty
  if (d.model == null || String(d.model).trim() === '') {
    d.model = defaults.model;
  } else {
    d.model = String(d.model).trim();
  }

  // Re-resolve path/env/cwd after prepare (may mutate)
  const pathFinal =
    d.path || d.PATH || d.lucy_path || d.lucyPath || agentPath || null;
  const envFinal =
    d.env && typeof d.env === 'object' && !Array.isArray(d.env)
      ? { ...d.env }
      : agentEnv;
  const cwdFinal =
    d.cwd ||
    d.default_cwd ||
    d.defaultCwd ||
    d.lucy_cwd ||
    d.lucyCwd ||
    agentCwd ||
    null;
  const presetsFinal = buildMcpPresets(mcpRoot || defaultMcpRoot(), {
    projectRoot: root,
    path: pathFinal,
    env: envFinal,
    cwd: cwdFinal,
  });

  const mcpList = resolveMcpList(d.mcp || [], presetsFinal);

  // models catalog optional (for UI dropdowns). Always include resolved model.
  let models = Array.isArray(d.models)
    ? d.models.map(String).map((s) => s.trim()).filter(Boolean)
    : null;
  if (models && d.model && !models.includes(d.model)) {
    models = [d.model, ...models];
  } else if (!models && d.model) {
    models = [d.model];
  }
  const contextWindow =
    d.context_window != null
      ? Number(d.context_window)
      : d.contextWindow != null
        ? Number(d.contextWindow)
        : null;

  // Welcome starters (cowork); null → client uses built-in defaults
  const starters = normalizeStarters(
    d.starters ?? d.starterPrompts ?? d.starter_prompts ?? d.welcomeStarters,
  );

  // toolsEnabled: null = all tools; string[] = only these at inference time
  const toolsEnabled = normalizeToolsEnabled(
    d.toolsEnabled ?? d.tools_enabled ?? d.enabledTools ?? d.enabled_tools,
  );

  return {
    name: d.name || 'agent',
    model: d.model,
    models,
    contextWindow:
      Number.isFinite(contextWindow) && contextWindow > 0 ? contextWindow : null,
    system: d.system,
    mcp: mcpList,
    path: pathFinal,
    env: envFinal,
    cwd: cwdFinal,
    allowlist: d.allowlist ?? DEFAULT_LUCY_ALLOWLIST,
    toolsEnabled,
    policyMode: d.policyMode || defaults.policyMode || 'ask',
    parallel_tools: d.parallel_tools ?? defaults.parallel_tools ?? true,
    starters,
    projectRoot: root,
    meta: {
      agentsMd: d._agentsMdLoaded || null,
      description: d.description || '',
      pathDirs: pathFinal ? String(pathFinal).split(':').filter(Boolean).length : 0,
    },
    raw: d,
  };
}

/**
 * Load agent file and materialize.
 * Falls back to the host setDefaultAgent() factory when:
 *   - spec is "default" / empty
 *   - the agent file cannot be resolved and a default factory is registered
 *
 * File agents are merged over the app default def (file keys win; omitted keys
 * inherit system/allowlist/mcp/… from the default factory when present).
 *
 * @param {string} spec - name or path
 */
export async function loadAgent(spec, opts = {}) {
  const projectRoot = resolveProjectRoot(opts.projectRoot);
  await loadPlugins();

  const rawSpec = spec == null ? '' : String(spec).trim();
  const wantDefault =
    !rawSpec || rawSpec === 'default' || rawSpec === 'DEFAULT';

  let abs = null;
  if (!wantDefault) {
    try {
      abs = resolveAgentPath(rawSpec, { projectRoot });
    } catch (err) {
      if (!_defaultAgentFactory) throw err;
      // Missing file agent → app default (named "default")
      abs = null;
    }
  }

  const ctx = agentFactoryCtx(projectRoot, opts, abs);

  /** @type {object|null} */
  let appDef = null;
  if (_defaultAgentFactory) {
    try {
      appDef = await invokeDefaultAgent(ctx);
    } catch (err) {
      console.error(
        'angela: default agent factory failed:',
        err?.message || err,
      );
      if (!abs) throw err;
    }
  }

  let def;
  if (abs) {
    const mod = await importAgentModule(abs);
    const factory =
      mod.default ||
      mod.createAgent ||
      mod.agent ||
      mod.create ||
      null;
    if (!factory) {
      throw new Error(`Agent module has no default export: ${abs}`);
    }
    const fileDef =
      typeof factory === 'function' ? await factory(ctx) : factory;
    if (!fileDef || typeof fileDef !== 'object') {
      throw new Error(`Agent factory must return a definition object: ${abs}`);
    }
    // App default underlay, then file (file wins). Strip app name so a coffee
    // agent is not renamed "default" when it omits name.
    if (appDef) {
      const { name: _ignoreName, ...appRest } = appDef;
      def = mergeAgentDefs(appRest, fileDef);
    } else {
      def = { ...fileDef };
    }
    if (!def.name) {
      def.name = abs.split('/').pop().replace(/\.(coffee|mjs|js)$/i, '');
    }
  } else {
    if (!appDef) {
      throw new Error(
        rawSpec
          ? `Agent not found: ${rawSpec} (no setDefaultAgent() fallback)`
          : 'No agent spec and no setDefaultAgent() factory registered',
      );
    }
    def = mergeAgentDefs(appDef, { name: 'default' });
  }

  const materialized = await materializeAgentDef(def, {
    ...opts,
    projectRoot,
    skipAgentsMd: Boolean(opts.skipAgentsMd),
  });
  materialized.path = abs;
  materialized.virtual = !abs;
  return materialized;
}

/**
 * List agent definition files under `<projectRoot>/.angela/agents/`.
 * Returns lightweight entries for UI dropdowns (does not fully materialize MCP).
 *
 * @param {string} [projectRoot]
 * @returns {Promise<Array<{
 *   name: string,
 *   file: string,
 *   path: string,
 *   model: string | null,
 *   models: string[] | null,
 *   contextWindow: number | null,
 *   description: string,
 *   starters: Array<{ label: string, prompt: string }> | null,
 * }>>}
 */
export async function listAgents(projectRoot) {
  const { readdirSync, readFileSync } = await import('node:fs');
  const root = resolveProjectRoot(projectRoot);
  const dir = agentsDirFromRoot(root);
  const out = [];

  if (existsSync(dir)) {
    const files = readdirSync(dir).filter((f) =>
      /\.(coffee|mjs|js)$/i.test(f),
    );

    for (const file of files) {
      const abs = join(dir, file);
      const name = file.replace(/\.(coffee|mjs|js)$/i, '');
      // Disk agent named "default" is fine; virtual default is separate.
      let model = null;
      let models = null;
      let contextWindow = null;
      let description = '';
      let allowlist = '';
      /** @type {string[]|null} */
      let toolsEnabled = null;
      /** @type {Array<{ label: string, prompt: string }>|null} */
      let starters = null;

      try {
        // UI catalog: resolve coffee defaults but skip AGENTS.md (may be remote/slow)
        const loaded = await loadAgent(name, {
          projectRoot: root,
          skipAgentsMd: true,
        });
        model = loaded.model || null;
        models = loaded.models;
        contextWindow = loaded.contextWindow;
        description = loaded.meta?.description || '';
        allowlist = loaded.allowlist || '';
        toolsEnabled = loaded.toolsEnabled ?? null;
        starters = loaded.starters || null;
      } catch (err) {
        try {
          const src = readFileSync(abs, 'utf8');
          const m = src.match(/model\s*:\s*['"]([^'"]+)['"]/);
          if (m) model = m[1];
          const cw = src.match(/context_window\s*:\s*(\d+)/);
          if (cw) contextWindow = Number(cw[1]);
        } catch {
          /* ignore */
        }
        console.error(
          `angela listAgents: failed to load ${name}:`,
          err?.message || err,
        );
      }

      out.push({
        name,
        file,
        path: abs,
        model,
        models,
        contextWindow,
        description,
        allowlist,
        toolsEnabled,
        starters,
        virtual: false,
      });
    }
  } else {
    console.error(`angela listAgents: no agents dir at ${dir}`);
  }

  // Virtual "default" only when no disk agents — otherwise hide it from the UI.
  if (out.length === 0 && _defaultAgentFactory) {
    try {
      const loaded = await loadAgent('default', {
        projectRoot: root,
        skipAgentsMd: true,
      });
      out.push({
        name: 'default',
        file: null,
        path: null,
        model: loaded.model || null,
        models: loaded.models,
        contextWindow: loaded.contextWindow,
        description: loaded.meta?.description || '',
        allowlist: loaded.allowlist || '',
        toolsEnabled: loaded.toolsEnabled ?? null,
        starters: loaded.starters || null,
        virtual: true,
      });
    } catch (err) {
      console.error(
        'angela listAgents: default agent failed:',
        err?.message || err,
      );
    }
  }

  out.sort((a, b) => a.name.localeCompare(b.name));
  return out;
}

function agentsDirFromRoot(projectRoot) {
  return join(resolveProjectRoot(projectRoot), '.angela', 'agents');
}

export default {
  loadAgent,
  listAgents,
  resolveAgentPath,
  materializeAgentDef,
  applyAgentsMd,
  normalizeStarters,
  normalizeToolsEnabled,
  filterToolNames,
  defaultAgentFields,
  setDefaultAgent,
  getDefaultAgentFactory,
  invokeDefaultAgent,
  mergeAgentDefs,
};
