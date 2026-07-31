/**
 * Model routing specs for angela / cowork.
 *
 * Display & agent coffee form:
 *   host:provider:model   e.g. local:copilot:gpt-5.6-luna
 *   provider:model        classic (host defaults to local)
 *
 * Optional plugins may register additional first-segment hosts that route
 * inference through a private backend (see plugins/).
 */

import {
  resolveContextWindow as aglResolveContextWindow,
  resolveContextWindowAsync as aglResolveContextWindowAsync,
} from 'agl-ai';

/** Explicit local (optional prefix). */
export const LOCAL_HOSTS = new Set(['local', 'pc', 'host']);

/** @type {Set<string>} plugin-registered remote host aliases */
const remoteHosts = new Set();

/**
 * @type {null | ((host: string) => string)}
 */
let resolveRemoteTargetFn = null;

/**
 * Plugins call this to teach parseModelSpec about remote hosts.
 * @param {{
 *   hosts: Iterable<string>,
 *   resolveTarget?: (host: string) => string,
 * }} opts
 */
export function registerRemoteModelHosts(opts = {}) {
  for (const h of opts.hosts || []) {
    const s = String(h || '').toLowerCase().trim();
    if (s) remoteHosts.add(s);
  }
  if (typeof opts.resolveTarget === 'function') {
    resolveRemoteTargetFn = opts.resolveTarget;
  }
}

/** @returns {Set<string>} */
export function getRemoteHostAliases() {
  return remoteHosts;
}

/**
 * Map logical remote host → backend target string (plugin-defined).
 * @param {string} host
 */
export function resolveRemoteTarget(host) {
  if (typeof resolveRemoteTargetFn === 'function') {
    return resolveRemoteTargetFn(host);
  }
  return null;
}

/** @deprecated use resolveRemoteTarget — kept for older callers */
export function resolveRshTarget(host) {
  return resolveRemoteTarget(host);
}

/** @deprecated empty unless plugins register hosts */
export const RSH_HOST_ALIASES = remoteHosts;

/**
 * @typedef {{
 *   host: 'local'|string,
 *   provider: string,
 *   model: string,
 *   agl: string,
 *   display: string,
 *   viaRsh: boolean,
 *   viaRemote: boolean,
 * }} ModelSpec
 */

/**
 * Parse UI / coffee / -m model string into routing + AGL pieces.
 * @param {string} spec
 * @returns {ModelSpec}
 */
export function parseModelSpec(spec) {
  const display = String(spec ?? '').trim();
  if (!display) {
    return {
      host: 'local',
      provider: '',
      model: '',
      agl: '',
      display: '',
      viaRsh: false,
      viaRemote: false,
    };
  }

  const i1 = display.indexOf(':');
  if (i1 <= 0) {
    return {
      host: 'local',
      provider: '',
      model: display,
      agl: display,
      display,
      viaRsh: false,
      viaRemote: false,
    };
  }

  const first = display.slice(0, i1);
  const rest = display.slice(i1 + 1);
  const firstLower = first.toLowerCase();

  const isRemote = remoteHosts.has(firstLower);
  const isLocal = LOCAL_HOSTS.has(firstLower);

  if (isRemote || isLocal) {
    const i2 = rest.indexOf(':');
    if (i2 <= 0 || i2 >= rest.length - 1) {
      throw new Error(
        `Invalid host-prefixed model "${display}". Expected host:provider:model ` +
          `(e.g. local:copilot:gpt-5.6-luna).`,
      );
    }
    const provider = rest.slice(0, i2);
    const model = rest.slice(i2 + 1);
    const host = isRemote ? firstLower : 'local';
    const viaRemote = isRemote;
    return {
      host,
      provider,
      model,
      agl: `${provider}:${model}`,
      display,
      viaRsh: viaRemote, // back-compat name
      viaRemote,
    };
  }

  // Classic provider:model
  return {
    host: 'local',
    provider: first,
    model: rest,
    agl: display,
    display,
    viaRsh: false,
    viaRemote: false,
  };
}

/**
 * @param {string} spec
 * @param {{ default?: number }} [opts]
 */
export function resolveContextWindow(spec, opts = {}) {
  try {
    const p = parseModelSpec(spec);
    return aglResolveContextWindow(p.agl || spec, opts);
  } catch {
    return aglResolveContextWindow(spec, opts);
  }
}

/**
 * @param {string} spec
 * @param {{ default?: number, force?: boolean }} [opts]
 * @returns {Promise<number>}
 */
export async function resolveContextWindowAsync(spec, opts = {}) {
  try {
    const p = parseModelSpec(spec);
    return await aglResolveContextWindowAsync(p.agl || spec, opts);
  } catch {
    return await aglResolveContextWindowAsync(spec, opts);
  }
}

/**
 * @param {string[]} list
 */
export function normalizeModelList(list) {
  const out = [];
  const seen = new Set();
  for (const raw of list || []) {
    const s = String(raw || '').trim();
    if (!s || seen.has(s)) continue;
    seen.add(s);
    out.push(s);
  }
  return out;
}
