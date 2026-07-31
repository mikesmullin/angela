/**
 * Optional private plugins under angela/plugins/* (gitignored).
 *
 * Each plugin directory exports default (or named `plugin`) from index.mjs / index.js:
 *   {
 *     name: string,
 *     register?(api): void | Promise<void>,   // once after import
 *     mcpPresets?(mcpRoot, opts): object,     // extra MCP presets
 *     applyAgentsMd?(def): Promise<object>,   // remote AGENTS.md etc.
 *     attachModelClient?(agent, parsed): Promise<boolean>,
 *     abort?(reason: string): void | Promise<void>,
 *     agentContext?(): object,                // agent factory ctx extras
 *   }
 *
 * Missing plugins/ is fine — core angela stays public without remote backends.
 */
import { readdirSync, existsSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ANGELA_ROOT = resolveAngelaRoot();

/** @type {Array<object>} */
let plugins = [];
let loaded = false;
/** @type {Promise<Array<object>>|null} */
let loadPromise = null;

function resolveAngelaRoot() {
  // src/ → package root
  return join(__dirname, '..');
}

/**
 * Directories that may contain plugin packages.
 * @returns {string[]}
 */
export function pluginSearchRoots() {
  const roots = [];
  const env = process.env.ANGELA_PLUGINS_DIR || process.env.ANGELA_PLUGINS;
  if (env) {
    for (const part of String(env).split(':')) {
      const p = part.trim();
      if (p) roots.push(p);
    }
  }
  roots.push(join(ANGELA_ROOT, 'plugins'));
  // Convenience: sibling private repo when not symlinked yet
  if (existsSync('/workspace/angela-plugins')) {
    roots.push('/workspace/angela-plugins');
  }
  if (existsSync('/workspace/mstack/angela-plugins')) {
    roots.push('/workspace/mstack/angela-plugins');
  }
  return roots;
}

/**
 * @param {string} root
 * @returns {string[]} absolute plugin package dirs
 */
function discoverPluginDirs(root) {
  if (!existsSync(root)) return [];
  try {
    const st = statSync(root);
    // Root itself is a single plugin package
    if (st.isDirectory() && existsSync(join(root, 'index.mjs'))) {
      return [root];
    }
    if (st.isDirectory() && existsSync(join(root, 'package.json'))) {
      // multi-plugin monorepo: each subdir
      return readdirSync(root)
        .map((n) => join(root, n))
        .filter((p) => {
          try {
            return (
              statSync(p).isDirectory() &&
              (existsSync(join(p, 'index.mjs')) ||
                existsSync(join(p, 'index.js')))
            );
          } catch {
            return false;
          }
        });
    }
    if (!st.isDirectory()) return [];
    return readdirSync(root)
      .map((n) => join(root, n))
      .filter((p) => {
        try {
          return (
            statSync(p).isDirectory() &&
            (existsSync(join(p, 'index.mjs')) || existsSync(join(p, 'index.js')))
          );
        } catch {
          return false;
        }
      });
  } catch {
    return [];
  }
}

/**
 * Load all plugins once. Safe if plugins/ is missing.
 * @returns {Promise<object[]>}
 */
export async function loadPlugins() {
  if (loaded) return plugins;
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    /** @type {Map<string, object>} */
    const byName = new Map();
    const seenDir = new Set();

    for (const root of pluginSearchRoots()) {
      for (const dir of discoverPluginDirs(root)) {
        const key = dir;
        if (seenDir.has(key)) continue;
        seenDir.add(key);
        const entry = existsSync(join(dir, 'index.mjs'))
          ? join(dir, 'index.mjs')
          : join(dir, 'index.js');
        try {
          const mod = await import(pathToFileURL(entry).href + `?t=${Date.now()}`);
          const plugin = mod.default || mod.plugin || null;
          if (!plugin || typeof plugin !== 'object') {
            console.error(`angela: plugin at ${dir} has no default export object`);
            continue;
          }
          const name = String(plugin.name || dir.split('/').pop());
          if (byName.has(name)) continue; // first wins
          if (typeof plugin.register === 'function') {
            await plugin.register({
              angelaRoot: ANGELA_ROOT,
              pluginDir: dir,
            });
          }
          byName.set(name, { ...plugin, name, _dir: dir });
          if (process.env.ANGELA_DEBUG_PLUGINS) {
            console.error(`angela: loaded plugin "${name}" from ${dir}`);
          }
        } catch (err) {
          console.error(
            `angela: failed to load plugin ${dir}:`,
            err?.message || err,
          );
        }
      }
    }

    plugins = [...byName.values()];
    loaded = true;
    return plugins;
  })();
  return loadPromise;
}

/** @returns {object[]} */
export function getPlugins() {
  return plugins;
}

/**
 * Merge plugin MCP presets into a base map.
 * @param {Record<string, object>} base
 * @param {string} mcpRoot
 * @param {object} opts
 */
export function mergePluginMcpPresets(base, mcpRoot, opts = {}) {
  const out = { ...base };
  for (const p of plugins) {
    if (typeof p.mcpPresets !== 'function') continue;
    try {
      const extra = p.mcpPresets(mcpRoot, opts) || {};
      Object.assign(out, extra);
    } catch (err) {
      console.error(
        `angela: plugin ${p.name} mcpPresets failed:`,
        err?.message || err,
      );
    }
  }
  return out;
}

/**
 * @param {object} def
 */
export async function applyPluginAgentsMd(def) {
  for (const p of plugins) {
    if (typeof p.applyAgentsMd !== 'function') continue;
    try {
      const next = await p.applyAgentsMd(def);
      if (next) return next;
    } catch (err) {
      console.error(
        `angela: plugin ${p.name} applyAgentsMd failed:`,
        err?.message || err,
      );
    }
  }
  return def;
}

/**
 * @param {object} agent
 * @param {object} parsed model spec
 * @returns {Promise<boolean>} true if a plugin handled attachment
 */
export async function attachPluginModelClient(agent, parsed) {
  for (const p of plugins) {
    if (typeof p.attachModelClient !== 'function') continue;
    try {
      const handled = await p.attachModelClient(agent, parsed);
      if (handled) return true;
    } catch (err) {
      console.error(
        `angela: plugin ${p.name} attachModelClient failed:`,
        err?.message || err,
      );
      throw err;
    }
  }
  return false;
}

/**
 * @param {string} reason
 */
export async function abortPlugins(reason) {
  for (const p of plugins) {
    if (typeof p.abort !== 'function') continue;
    try {
      await p.abort(reason);
    } catch {
      /* optional */
    }
  }
}

/**
 * Extra fields for agent factory ctx.
 */
export function pluginAgentContext() {
  const ctx = {};
  for (const p of plugins) {
    if (typeof p.agentContext !== 'function') continue;
    try {
      Object.assign(ctx, p.agentContext() || {});
    } catch {
      /* ignore */
    }
  }
  return ctx;
}

export default {
  loadPlugins,
  getPlugins,
  pluginSearchRoots,
  mergePluginMcpPresets,
  applyPluginAgentsMd,
  attachPluginModelClient,
  abortPlugins,
  pluginAgentContext,
};
