/**
 * Named MCP server presets for CLI + agent definition files.
 * Core package only ships local servers. Remote presets come from plugins.
 */
import { resolve, join, dirname } from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { mergePluginMcpPresets, getPlugins } from './plugins.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));

export function defaultMcpRoot() {
  if (process.env.ANGELA_MCP_ROOT) return process.env.ANGELA_MCP_ROOT;
  if (existsSync('/workspace/mcp')) return '/workspace/mcp';
  // Public layout: sibling mcp-* packages or ANGELA_MCP_ROOT
  return resolve(__dirname, '../../mcp');
}

/**
 * @param {string} [mcpRoot]
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

  const base = {
    'file-io': {
      name: 'file-io',
      command: bun,
      args: [join(mcpRoot, 'file-io/server.mjs')],
      env: {
        MCP_FILE_ROOT: jail,
      },
    },
    'term-local': {
      name: 'term-local',
      command: bun,
      args: [join(mcpRoot, 'term-local/server.mjs')],
      env: termLocalEnv,
    },
  };

  // Plugins may add remote MCP presets when installed under plugins/
  if (getPlugins().length) {
    return mergePluginMcpPresets(base, mcpRoot, opts);
  }
  return base;
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
      if (!existsSync(p.args[0])) {
        throw new Error(`MCP server script missing: ${p.args[0]}`);
      }
      out.push({ ...p, env: { ...p.env } });
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

export default { buildMcpPresets, resolveMcpList, defaultMcpRoot };
