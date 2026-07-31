/**
 * Project / workspace root for angela.
 *
 * Default: process.cwd() when you run `angela`.
 * Override: `-c/--config <path>` (project root that contains `.angela/`).
 *
 * Layout:
 *   <root>/
 *     .angela/
 *       agents/          # -a my-agent → .angela/agents/my-agent.coffee
 *       sessions/        # durable session state + inference logs
 */
import { resolve, join } from 'node:path';
import { existsSync, mkdirSync } from 'node:fs';

/**
 * @param {string} [configPath] - from -c/--config; defaults to cwd
 */
export function resolveProjectRoot(configPath) {
  const root = configPath
    ? resolve(String(configPath))
    : resolve(process.cwd());
  return root;
}

/**
 * @param {string} projectRoot
 */
export function angelaDir(projectRoot) {
  return join(projectRoot, '.angela');
}

export function agentsDir(projectRoot) {
  return join(angelaDir(projectRoot), 'agents');
}

export function sessionsDir(projectRoot) {
  return join(angelaDir(projectRoot), 'sessions');
}

/**
 * Ensure .angela/{agents,sessions} exist.
 */
export function ensureAngelaLayout(projectRoot) {
  const dirs = [angelaDir(projectRoot), agentsDir(projectRoot), sessionsDir(projectRoot)];
  for (const d of dirs) {
    if (!existsSync(d)) mkdirSync(d, { recursive: true });
  }
  return {
    root: projectRoot,
    angela: angelaDir(projectRoot),
    agents: agentsDir(projectRoot),
    sessions: sessionsDir(projectRoot),
  };
}

/**
 * Paths searched for `-a <name>` (first hit wins).
 */
export function agentSearchDirs(projectRoot) {
  return [
    agentsDir(projectRoot), // primary: <root>/.angela/agents/
    join(projectRoot, 'agents'), // legacy/local convenience
    projectRoot,
  ];
}

export default {
  resolveProjectRoot,
  angelaDir,
  agentsDir,
  sessionsDir,
  ensureAngelaLayout,
  agentSearchDirs,
};
