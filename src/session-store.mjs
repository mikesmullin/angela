/**
 * Durable sessions under `.angela/sessions/`.
 *
 * Pair of files per session:
 *
 *   <id>.json   — **metadata only** (no context-window / transcript body)
 *   <id>.jsonl  — append-only **wrapped** events
 *
 * JSONL line shape (wrapper keeps telemetry out of the payload):
 *
 *   {
 *     "event_id": "e1a2b3",
 *     "ts": "ISO-8601",
 *     "event_type": "context_window" | "provider_request" | "provider_response"
 *                 | "tool_call" | "tool_result" | "policy" | "session" | "error"
 *                 | "usage" | "gen_info" | "system" | "tools" | "context_meta" | …,
 *     "duration_ms": 12.4 | null,
 *     "payload": { … },
 *     "meta": {
 *       "visible": true,           // context_window: include in next provider request
 *       "telemetry": { … }         // gen stats (gen_info + the round's assistant/reasoning)
 *     }
 *   }
 *
 * Encoding (`logEncoding: "delta/v1"` on the session JSON):
 *   Each transcript item is stored once (context_window / tool_call / tool_result).
 *   system prompt + tool schemas are written once (re-emitted only on change).
 *   provider_request stores headers + event_id pointers, not messages[].
 *   Full-window snapshots are not written (ANGELA_LOG_SNAPSHOTS=1 to dump them).
 *   Visibility / goto patches append context_meta (last-write-wins); the original
 *   line is not rewritten.
 *
 * Line order is the event sequence; session id is the filename stem (not repeated).
 *
 * `angela clean` removes `.angela/sessions/*`
 */
import { join } from 'node:path';
import {
  existsSync,
  mkdirSync,
  readdirSync,
  rmSync,
  appendFileSync,
  writeFileSync,
  readFileSync,
} from 'node:fs';
import { randomBytes, createHash } from 'node:crypto';
import { sessionsDir, ensureAngelaLayout } from './project.mjs';

function nowIso() {
  return new Date().toISOString();
}

function newId() {
  const t = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const r = randomBytes(3).toString('hex');
  return `${t}_${r}`;
}

function safeClone(value) {
  if (value === undefined) return null;
  try {
    return structuredClone(value);
  } catch {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch {
      return { _unserializable: String(value) };
    }
  }
}

function stableHash(value) {
  try {
    return createHash('sha256')
      .update(typeof value === 'string' ? value : JSON.stringify(value ?? null))
      .digest('hex')
      .slice(0, 16);
  } catch {
    return null;
  }
}

/** Debug dumps of the full provider messages[] window. Default off. */
export function logSnapshotsEnabled() {
  const v = String(process.env.ANGELA_LOG_SNAPSHOTS || '')
    .trim()
    .toLowerCase();
  return v === '1' || v === 'true' || v === 'yes';
}

/**
 * Fold append-only `context_meta` patches onto earlier events (last-write-wins).
 * Raw jsonl lines are unchanged; this is the read view for visible/goto.
 * @param {object[]} events
 * @returns {object[]}
 */
export function foldEventMetas(events) {
  if (!Array.isArray(events) || !events.length) return events || [];
  /** @type {Map<string, object>} */
  const extra = new Map();
  for (const wrap of events) {
    if (wrap?.event_type !== 'context_meta') continue;
    const p = wrap.payload && typeof wrap.payload === 'object' ? wrap.payload : {};
    const ids = Array.isArray(p.event_ids)
      ? p.event_ids
      : p.event_id
        ? [p.event_id]
        : [];
    const { event_id: _eid, event_ids: _eids, ...rest } = p;
    if (!Object.keys(rest).length) continue;
    for (const eid of ids) {
      if (!eid) continue;
      extra.set(String(eid), { ...(extra.get(String(eid)) || {}), ...rest });
    }
  }
  if (!extra.size) return events;
  return events.map((wrap) => {
    const eid = wrap?.event_id;
    const patch = eid ? extra.get(String(eid)) : null;
    if (!patch) return wrap;
    const meta = { ...(wrap.meta || {}), ...patch };
    if (Object.prototype.hasOwnProperty.call(patch, 'visible')) {
      meta.visible = Boolean(patch.visible);
    }
    return { ...wrap, meta };
  });
}

/**
 * Map one jsonl event to an OpenAI-shaped chat message (for request reconstruct).
 * @param {object} wrap
 * @returns {object|null}
 */
export function eventToProviderMessage(wrap) {
  if (!wrap) return null;
  const p = wrap.payload;
  if (!p || typeof p !== 'object') return null;
  if (wrap.event_type === 'system' && p.content != null) {
    return { role: 'system', content: String(p.content) };
  }
  if (
    wrap.event_type === 'context_window' &&
    p.role &&
    p.content != null &&
    p.kind !== 'provider_messages_snapshot'
  ) {
    const msg = { role: p.role, content: String(p.content) };
    if (p.role === 'assistant' && Array.isArray(p.tool_calls) && p.tool_calls.length) {
      msg.tool_calls = p.tool_calls;
      if (!msg.content) msg.content = '';
    }
    if (p.role === 'tool' && (p.tool_call_id || p.id)) {
      msg.tool_call_id = p.tool_call_id || p.id;
    }
    return msg;
  }
  if (wrap.event_type === 'tool_call' && p.name) {
    const callId = p.id || p.tool_call_id || wrap.event_id;
    const args =
      typeof p.args === 'string' ? p.args : JSON.stringify(p.args ?? {});
    return {
      role: 'assistant',
      content: '',
      tool_calls: [
        {
          id: callId,
          type: 'function',
          function: { name: p.name, arguments: args },
        },
      ],
    };
  }
  if (
    wrap.event_type === 'tool_result' &&
    (p.name || p.text != null || p.content != null)
  ) {
    const callId =
      p.tool_call_id || p.id || p.call_id || wrap.event_id;
    const content =
      p.text != null
        ? String(p.text)
        : typeof p.content === 'string'
          ? p.content
          : JSON.stringify(p.content ?? p);
    return { role: 'tool', tool_call_id: callId, content };
  }
  return null;
}

export class SessionStore {
  /**
   * @param {string} projectRoot
   */
  constructor(projectRoot) {
    this.projectRoot = projectRoot;
    ensureAngelaLayout(projectRoot);
    this.dir = sessionsDir(projectRoot);
  }

  /**
   * Create a new session — metadata JSON only + first session event in jsonl.
   *
   * @param {{
   *   agent?: string,
   *   model?: string,
   *   title?: string,
   *   mcp?: string[],
   *   policyMode?: string,
   *   approveMode?: string | boolean | null,
   *   allowlistSource?: string | null,
   *   config?: object,          // effective resolved config (non-secret)
   *   overrides?: object,       // which CLI flags overrode the agent file
   *   agentFile?: string | null,
   * }} meta
   */
  create(meta = {}) {
    const id = newId();
    const state = {
      id,
      apiVersion: 'angela/v1',
      logEncoding: 'delta/v1',
      kind: 'AgentSession',
      createdAt: nowIso(),
      updatedAt: nowIso(),
      status: 'ready',
      agent: meta.agent || null,
      agentFile: meta.agentFile || null,
      model: meta.model || null,
      title: meta.title || null,
      mcp: meta.mcp || [],
      policyMode: meta.policyMode || 'ask',
      approveMode: meta.approveMode ?? null,
      allowlistSource: meta.allowlistSource || null,
      /** Effective allowlist text when overridden (ui/cli file); null = use agent default at load time */
      allowlist: meta.allowlist ?? null,
      contextWindow: meta.contextWindow ?? null,
      projectRoot: this.projectRoot,
      // Effective run config + which CLI flags won over the agent file
      config: meta.config || null,
      overrides: meta.overrides || null,
      // counters only — not transcript content
      turns: 0,
      eventCount: 0,
      /** Last completion `usage.prompt_tokens` from the provider (null until first round). */
      lastPromptTokens: null,
      lastTotalTokens: null,
      lastCompletionTokens: null,
      logFile: `${id}.jsonl`,
    };
    this.#writeState(id, state);
    this.appendEvent(id, {
      event_type: 'session',
      payload: {
        kind: 'created',
        agent: state.agent,
        model: state.model,
        mcp: state.mcp,
        overrides: state.overrides,
        config: state.config,
      },
    });
    return state;
  }

  load(id) {
    const p = this.statePath(id);
    if (!existsSync(p)) return null;
    return JSON.parse(readFileSync(p, 'utf8'));
  }

  statePath(id) {
    return join(this.dir, `${id}.json`);
  }

  logPath(id) {
    return join(this.dir, `${id}.jsonl`);
  }

  #writeState(id, state) {
    if (!existsSync(this.dir)) mkdirSync(this.dir, { recursive: true });
    state.updatedAt = nowIso();
    writeFileSync(this.statePath(id), JSON.stringify(state, null, 2) + '\n', 'utf8');
  }

  save(state) {
    this.#writeState(state.id, state);
  }

  /**
   * Patch session metadata fields and rewrite the .json file.
   * Used for mid-session UI updates (e.g. allowlist blur), same as CLI
   * `--allowlist` being reflected on the session record.
   *
   * @param {string} id
   * @param {Record<string, any>} patch
   * @returns {object|null} updated state
   */
  updateMeta(id, patch = {}) {
    const state = this.load(id);
    if (!state) return null;
    const allowed = new Set([
      'title',
      'status',
      'model',
      'allowlistSource',
      'allowlist',
      'approveMode',
      'policyMode',
      'contextWindow',
      'overrides',
      'config',
      'lastPromptTokens',
      'lastTotalTokens',
      'lastCompletionTokens',
    ]);
    for (const [k, v] of Object.entries(patch)) {
      if (!allowed.has(k)) continue;
      state[k] = v;
    }
    // Keep config mirror in sync when present
    if (state.config && typeof state.config === 'object') {
      if (patch.allowlistSource != null) {
        state.config.allowlistSource = patch.allowlistSource;
      }
      if (patch.allowlist != null) {
        state.config.allowlistBytes = String(patch.allowlist).length;
      }
      if (patch.model != null) state.config.model = patch.model;
      if (patch.contextWindow != null) {
        state.config.contextWindow = patch.contextWindow;
      }
      if (patch.overrides != null) state.config.overrides = patch.overrides;
    }
    this.#writeState(id, state);
    return state;
  }

  /**
   * Append a wrapped event to the jsonl log.
   *
   * @param {string} id
   * @param {{
   *   event_type: string,
   *   payload?: any,
   *   duration_ms?: number | null,
   *   ts?: string,
   *   meta?: object | null,
   *   event_id?: string | null,
   * }} opts
   *   `payload` is stored verbatim (cloned); do not put telemetry inside it.
   *   `meta` holds UI/persistence flags (visible) and gen telemetry.
   */
  appendEvent(
    id,
    {
      event_type,
      payload = null,
      duration_ms = null,
      ts = null,
      meta = null,
      event_id = null,
    },
  ) {
    const state = this.load(id);
    const eid =
      event_id ||
      `e${randomBytes(4).toString('hex')}`;
    const wrapper = {
      event_id: eid,
      ts: ts || nowIso(),
      event_type,
      duration_ms:
        duration_ms == null || Number.isNaN(Number(duration_ms))
          ? null
          : Number(duration_ms),
      payload: safeClone(payload),
      meta: meta == null ? {} : safeClone(meta) || {},
    };
    appendFileSync(this.logPath(id), JSON.stringify(wrapper) + '\n', 'utf8');
    if (state) {
      state.eventCount = (state.eventCount || 0) + 1;
      this.save(state);
    }
    return wrapper;
  }

  /** Read all jsonl wrappers for a session (newest last). Raw lines; no meta fold. */
  listEvents(id) {
    const p = this.logPath(id);
    if (!existsSync(p)) return [];
    const out = [];
    for (const line of readFileSync(p, 'utf8').split('\n')) {
      if (!line.trim()) continue;
      try {
        out.push(JSON.parse(line));
      } catch {
        /* skip */
      }
    }
    return out;
  }

  /** listEvents with context_meta last-write-wins applied to earlier rows. */
  listEventsFolded(id) {
    return foldEventMetas(this.listEvents(id));
  }

  /**
   * Rebuild the inference request body from a slim (or legacy verbatim) log line.
   * @param {string} id
   * @param {string} requestEventId
   * @returns {object|null}
   */
  reconstructProviderRequest(id, requestEventId) {
    if (!requestEventId) return null;
    const events = this.listEventsFolded(id);
    const idx = events.findIndex(
      (w) => w.event_id === requestEventId && w.event_type === 'provider_request',
    );
    if (idx < 0) return null;
    const wrap = events[idx];
    const p = wrap.payload && typeof wrap.payload === 'object' ? wrap.payload : {};

    // Legacy snapshot logs stored the full messages[] on the request.
    if (Array.isArray(p.messages)) {
      return {
        model: p.model ?? null,
        stream: p.stream,
        context_window: p.context_window ?? null,
        messages: p.messages,
        tools: p.tools ?? null,
      };
    }

    /** @type {Map<string, object>} */
    const byId = new Map();
    for (const ev of events) {
      if (ev?.event_id) byId.set(ev.event_id, ev);
    }

    let messages;
    if (Array.isArray(p.message_event_ids) && p.message_event_ids.length) {
      messages = [];
      for (const eid of p.message_event_ids) {
        const src = byId.get(eid);
        const msg = eventToProviderMessage(src);
        if (msg) messages.push(msg);
      }
    } else {
      // Fall back: fold visible transcript items that occurred before this request.
      messages = [];
      let system = null;
      for (let i = 0; i < idx; i++) {
        const ev = events[i];
        if (ev.event_type === 'system') {
          system = eventToProviderMessage(ev);
          continue;
        }
        if (ev.event_type === 'tools' || ev.event_type === 'context_meta') continue;
        if (ev.event_type === 'provider_request' || ev.event_type === 'provider_response') {
          continue;
        }
        if (ev.event_type === 'gen_info' || ev.event_type === 'session') continue;
        if (ev.meta && ev.meta.visible === false) continue;
        const msg = eventToProviderMessage(ev);
        if (msg) messages.push(msg);
      }
      if (system) messages.unshift(system);
    }

    const toolsWrap = p.tools_event_id ? byId.get(p.tools_event_id) : null;
    const tools = toolsWrap?.payload?.tools ?? null;
    return {
      model: p.model ?? null,
      stream: p.stream,
      context_window: p.context_window ?? null,
      messages,
      tools,
    };
  }

  /**
   * Snapshot of the most recent provider gen_info telemetry for a session.
   * Assistant/reasoning events freeze a copy; tool events store gen_info_event_id.
   * @returns {object|null}
   */
  lastGenTelemetry(id) {
    const events = this.listEvents(id);
    for (let i = events.length - 1; i >= 0; i--) {
      const wrap = events[i];
      if (wrap.event_type !== 'gen_info') continue;
      const tel = wrap.meta?.telemetry || wrap.payload;
      if (tel && typeof tel === 'object') {
        return safeClone({
          ...tel,
          gen_info_event_id: wrap.event_id || null,
        });
      }
    }
    return null;
  }

  /**
   * Map tool_call ↔ tool_result event_ids (FIFO per tool name).
   * Providers require call+result pairs; eye-hide must toggle both.
   *
   * @param {string} id session id
   * @returns {Map<string, string>} eventId → partner eventId
   */
  toolCallResultPairs(id) {
    /** @type {Map<string, string[]>} */
    const pending = new Map();
    /** @type {Map<string, string>} */
    const pairOf = new Map();
    for (const wrap of this.listEvents(id)) {
      const et = wrap.event_type;
      const name = wrap.payload?.name;
      if (!name || !wrap.event_id) continue;
      if (et === 'tool_call') {
        if (!pending.has(name)) pending.set(name, []);
        pending.get(name).push(wrap.event_id);
      } else if (et === 'tool_result') {
        const q = pending.get(name);
        if (q?.length) {
          const callId = q.shift();
          pairOf.set(callId, wrap.event_id);
          pairOf.set(wrap.event_id, callId);
        }
      }
    }
    return pairOf;
  }

  /**
   * Event ids that must share visibility with `eventId` (self + tool pair).
   * @param {string} id
   * @param {string} eventId
   * @returns {string[]}
   */
  visibilityGroupEventIds(id, eventId) {
    if (!eventId) return [];
    const pairOf = this.toolCallResultPairs(id);
    const partner = pairOf.get(eventId);
    return partner ? [eventId, partner] : [eventId];
  }

  /**
   * Patch `meta` on one or more events by appending a `context_meta` delta.
   * Original jsonl lines are left unchanged; readers fold last-write-wins.
   *
   * @param {string} id
   * @param {string|string[]} eventIds
   * @param {object} metaPatch
   * @returns {object[]|null} folded wrappers for the matched ids (null if none)
   */
  updateEventsMeta(id, eventIds, metaPatch = {}) {
    const ids = [
      ...(Array.isArray(eventIds) ? eventIds : [eventIds]).filter(Boolean),
    ];
    if (!ids.length) return null;
    const want = new Set(ids.map(String));
    const events = this.listEvents(id);
    const resolved = [];
    for (const wrap of events) {
      if (wrap?.event_type === 'context_meta') continue;
      const match =
        (wrap.event_id && want.has(String(wrap.event_id))) ||
        (!wrap.event_id && wrap.ts && want.has(String(wrap.ts)));
      if (match && wrap.event_id && !resolved.includes(wrap.event_id)) {
        resolved.push(wrap.event_id);
      }
    }
    if (!resolved.length) return null;
    const patch = { ...(metaPatch || {}) };
    if (Object.prototype.hasOwnProperty.call(patch, 'visible')) {
      patch.visible = Boolean(patch.visible);
    }
    this.appendEvent(id, {
      event_type: 'context_meta',
      payload: {
        event_ids: resolved,
        ...patch,
      },
    });
    const folded = this.listEventsFolded(id);
    const resolvedSet = new Set(resolved);
    return folded.filter((w) => w.event_id && resolvedSet.has(w.event_id));
  }

  /**
   * Patch `meta` on one jsonl event by event_id (append context_meta).
   *
   * @returns {object|null} folded wrapper
   */
  updateEventMeta(id, eventId, metaPatch = {}) {
    const list = this.updateEventsMeta(id, eventId, metaPatch);
    return list?.[0] || null;
  }

  /**
   * Set visible on an event and its tool_call/tool_result partner (if any).
   * @returns {{ events: object[], eventIds: string[] }|null}
   */
  setEventVisible(id, eventId, visible) {
    const group = this.visibilityGroupEventIds(id, eventId);
    const events = this.updateEventsMeta(id, group, { visible: Boolean(visible) });
    if (!events) return null;
    return { events, eventIds: group };
  }

  /**
   * Visible context messages for the next provider turn (user/assistant only).
   * Skips snapshots and events with meta.visible === false.
   */
  listVisibleContextMessages(id) {
    const msgs = [];
    for (const wrap of this.listEventsFolded(id)) {
      if (wrap.event_type !== 'context_window') continue;
      const p = wrap.payload;
      if (!p || !p.role || p.content == null) continue;
      if (p.kind === 'provider_messages_snapshot') continue;
      if (wrap.meta && wrap.meta.visible === false) continue;
      if (p.role !== 'user' && p.role !== 'assistant') continue;
      msgs.push({
        event_id: wrap.event_id || null,
        role: p.role,
        content: String(p.content),
        meta: wrap.meta || {},
      });
    }
    return msgs;
  }

  /**
   * Hydrate agent.context_window from disk (session open / resume only).
   *
   * 1 jsonl event = 1 context item = 1 UI row.
   * `meta.visible` is the eye-toggle flag (never deleted, only hidden).
   * Snapshots are debug/telemetry only — never used as seed.
   */
  buildContextWindowItems(id) {
    const items = [];
    for (const wrap of this.listEventsFolded(id)) {
      const p = wrap.payload;
      if (!p) continue;
      const visible = !(wrap.meta && wrap.meta.visible === false);
      const eid = wrap.event_id || null;
      // UI/goto stamps live in meta (not payload)
      const goto =
        wrap.meta?.goto && typeof wrap.meta.goto === 'object'
          ? {
              enabled: Boolean(wrap.meta.goto.enabled),
              label: String(wrap.meta.goto.label || ''),
            }
          : undefined;

      // Skip frozen provider dumps — not part of the curated list
      if (
        wrap.event_type === 'context_window' &&
        p.kind === 'provider_messages_snapshot'
      ) {
        continue;
      }

      if (
        wrap.event_type === 'context_window' &&
        p.role &&
        p.content != null
      ) {
        items.push({
          role: p.role,
          content: String(p.content),
          id: eid,
          visible,
          ...(goto ? { goto } : {}),
        });
        continue;
      }

      if (
        (wrap.event_type === 'reasoning' || p?.type === 'reasoning') &&
        (p?.text || p?.content)
      ) {
        items.push({
          role: 'reasoning',
          content: String(p.text || p.content || ''),
          id: eid,
          visible,
          ...(goto ? { goto } : {}),
        });
        continue;
      }

      if (wrap.event_type === 'tool_call' && p.name) {
        const callId = p.id || p.tool_call_id || eid || `call_${items.length}`;
        const args =
          typeof p.args === 'string'
            ? p.args
            : JSON.stringify(p.args ?? {});
        items.push({
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: callId,
              type: 'function',
              function: { name: p.name, arguments: args },
            },
          ],
          id: eid,
          visible,
          // UI-only extras (not sent to provider)
          name: p.name,
          args: p.args,
          explanation: p.explanation || null,
          ...(goto ? { goto } : {}),
        });
        continue;
      }

      if (wrap.event_type === 'tool_result' && p.name) {
        const callId =
          p.tool_call_id || p.id || p.call_id || eid || `call_${items.length}`;
        items.push({
          role: 'tool',
          tool_call_id: callId,
          content:
            p.text != null
              ? String(p.text)
              : typeof p.content === 'string'
                ? p.content
                : JSON.stringify(p.content ?? p),
          id: eid,
          visible,
          name: p.name,
          ok: p.ok,
          denied: p.denied,
          explanation: p.explanation || null,
          ...(goto ? { goto } : {}),
        });
      }
    }
    return items;
  }

  /** @deprecated use appendEvent — kept for short call sites */
  append(id, event) {
    // Legacy: treat whole object as payload under event_type from event.type
    const { type, event_type, duration_ms, ...rest } = event || {};
    return this.appendEvent(id, {
      event_type: event_type || type || 'event',
      payload: Object.keys(rest).length ? rest : null,
      duration_ms: duration_ms ?? null,
    });
  }

  /**
   * Bump turn counter after a completed user→assistant exchange.
   * Transcript lives only in the jsonl log.
   */
  recordTurnComplete(id, { duration_ms = null } = {}) {
    const state = this.load(id);
    if (!state) return;
    state.turns = (state.turns || 0) + 1;
    state.status = 'ready';
    this.save(state);
    this.appendEvent(id, {
      event_type: 'session',
      duration_ms,
      payload: { kind: 'turn_complete', turn: state.turns },
    });
  }

  setStatus(id, status) {
    const state = this.load(id);
    if (!state) return;
    state.status = status;
    this.save(state);
  }

  list() {
    if (!existsSync(this.dir)) return [];
    return readdirSync(this.dir)
      .filter((f) => f.endsWith('.json'))
      .map((f) => f.replace(/\.json$/, ''))
      .sort()
      .reverse();
  }

  /** Load all session metadata objects (newest first). */
  listMeta() {
    return this.list()
      .map((id) => this.load(id))
      .filter(Boolean);
  }

  /**
   * Delete one session (both .json and .jsonl).
   * @returns {boolean}
   */
  delete(id) {
    if (!id || String(id).includes('..') || String(id).includes('/')) return false;
    let n = 0;
    for (const p of [this.statePath(id), this.logPath(id)]) {
      if (existsSync(p)) {
        rmSync(p, { force: true });
        n++;
      }
    }
    return n > 0;
  }

  /**
   * Remove all session files (`angela clean`).
   */
  clean() {
    if (!existsSync(this.dir)) return 0;
    const files = readdirSync(this.dir);
    let n = 0;
    for (const f of files) {
      rmSync(join(this.dir, f), { force: true });
      n++;
    }
    return n;
  }
}

/**
 * Wrap AGL provider client so every API round-trip is logged as
 * provider_request / provider_response (payload = verbatim body).
 * Uses Proxy — provider modules often freeze `inference`.
 *
 * @param {object} agent - AGL Agent instance
 * @param {SessionStore} store
 * @param {string} sessionId
 */
/**
 * Extract OpenAI-style usage from a provider response body (gl1 extractUsage).
 * @returns {{ prompt_tokens: number, completion_tokens: number, total_tokens: number } | null}
 */
export function extractUsage(res) {
  if (!res || typeof res !== 'object') return null;
  const u = res.usage || res.Usage || null;
  if (!u || typeof u !== 'object') return null;
  const prompt = Number(
    u.prompt_tokens ?? u.promptTokens ?? u.input_tokens ?? u.inputTokens,
  );
  if (!Number.isFinite(prompt) || prompt < 0) return null;
  const completion = Number(
    u.completion_tokens ??
      u.completionTokens ??
      u.output_tokens ??
      u.outputTokens ??
      0,
  );
  const total = Number(
    u.total_tokens ??
      u.totalTokens ??
      (Number.isFinite(completion) ? prompt + completion : prompt),
  );
  return {
    prompt_tokens: prompt,
    completion_tokens: Number.isFinite(completion) ? completion : 0,
    total_tokens: Number.isFinite(total) ? total : prompt,
  };
}

/** OpenAI-style finish_reason + optional LM Studio stop_reason. */
export function extractFinishReason(res) {
  if (!res || typeof res !== 'object') return { finish_reason: null, stop_reason: null };
  const choice = res.choices?.[0] || res.choice || null;
  const finish =
    choice?.finish_reason ??
    choice?.finishReason ??
    res.finish_reason ??
    res.finishReason ??
    null;
  const stop =
    res.stop_reason ??
    res.stopReason ??
    choice?.stop_reason ??
    res.stats?.stop_reason ??
    res.stats?.stopReason ??
    null;
  return {
    finish_reason: finish != null ? String(finish) : null,
    stop_reason: stop != null ? String(stop) : null,
  };
}

/**
 * Model "thinking" / reasoning from a completion (Gemma 4 via LM Studio, etc.).
 * @returns {string|null}
 */
export function extractAssistantText(res) {
  // The round's final assistant TEXT (content only — tool_call-only rounds
  // carry no prose and yield null). String content passes through; a
  // content-parts array joins its text parts.
  if (!res || typeof res !== 'object') return null;
  const msg = res.choices?.[0]?.message || res.message || null;
  if (!msg || typeof msg !== 'object' || msg.tool_calls?.length) return null;
  const c = msg.content;
  if (typeof c === 'string') return c.trim() ? c : null;
  if (Array.isArray(c)) {
    const s = c
      .filter((part) => part?.type === 'text' && typeof part.text === 'string')
      .map((part) => part.text)
      .join('\n')
      .trim();
    return s ? s : null;
  }
  return null;
}

export function extractReasoningContent(res) {
  if (!res || typeof res !== 'object') return null;
  const msg = res.choices?.[0]?.message || res.message || null;
  if (!msg || typeof msg !== 'object') return null;
  const raw =
    msg.reasoning_content ??
    msg.reasoningContent ??
    msg.reasoning ??
    msg.thinking ??
    null;
  if (raw == null) return null;
  const s = String(raw).trim();
  return s.length ? s : null;
}

/**
 * @param {object} agent
 * @param {SessionStore} store
 * @param {string} sessionId
 * @param {{ onUsage?: (u: object) => void }} [hooks]
 */
export function instrumentProviderLogging(agent, store, sessionId, hooks = {}) {
  if (!agent?.client?.inference || !store || !sessionId) return;

  /** @type {string|null} */
  let systemEventId = null;
  /** @type {string|null} */
  let systemHash = null;
  /** @type {string|null} */
  let toolsEventId = null;
  /** @type {string|null} */
  let toolsHash = null;

  function persistStableBlobs(req) {
    const msgs = Array.isArray(req?.messages) ? req.messages : [];
    const sys = msgs.find((m) => m && m.role === 'system');
    const sysContent = sys?.content != null ? String(sys.content) : '';
    if (sysContent) {
      const h = stableHash(sysContent);
      if (h && h !== systemHash) {
        const w = store.appendEvent(sessionId, {
          event_type: 'system',
          payload: { content: sysContent, bytes: sysContent.length },
        });
        systemEventId = w.event_id;
        systemHash = h;
      }
    }
    if (Array.isArray(req?.tools) && req.tools.length) {
      const cloned = safeClone(req.tools);
      const h = stableHash(cloned);
      if (h && h !== toolsHash) {
        const w = store.appendEvent(sessionId, {
          event_type: 'tools',
          payload: { tools: cloned, hash: h },
        });
        toolsEventId = w.event_id;
        toolsHash = h;
      }
    }
  }

  function messageEventIdsFromAgent() {
    const ids = [];
    if (systemEventId) ids.push(systemEventId);
    const cw = agent?.context_window;
    if (Array.isArray(cw)) {
      for (const row of cw) {
        if (!row || row.visible === false) continue;
        if (row.id) ids.push(row.id);
      }
    }
    return ids;
  }

  function slimRequestPayload(req, wide = false) {
    persistStableBlobs(req);
    const c = {};
    if (req && typeof req === 'object') {
      for (const [k, v] of Object.entries(req)) {
        if (
          k === 'messages' ||
          k === 'tools' ||
          k === 'signal' ||
          k === 'on_delta'
        ) {
          continue;
        }
        if (typeof v === 'function') continue;
        c[k] = safeClone(v);
      }
    }
    const messages = Array.isArray(req?.messages) ? req.messages : [];
    const tools = Array.isArray(req?.tools) ? req.tools : null;
    c.messages_count = messages.length;
    c.tools_count = tools ? tools.length : 0;
    c.system_event_id = systemEventId;
    c.tools_event_id = toolsEventId;
    c.message_event_ids = messageEventIdsFromAgent();
    c.roles = messages.map((m) =>
      m && typeof m === 'object' ? m.role || null : null,
    );
    if (wide) c._wide = true;
    return c;
  }

  function noteUsageAndTelemetry(res, model, durationMs, ttftMs, errorMsg = null) {
    // Authoritative counts only from the completion response `usage` object.
    // Never invent tokens (no chars/4); omit prompt fields if the provider
    // did not report usage (common on some stream final chunks).
    const usage = extractUsage(res);
    const { finish_reason, stop_reason } = extractFinishReason(res);
    const dur = Number(durationMs) || 0;
    const comp = usage ? Number(usage.completion_tokens) || 0 : 0;
    const tokPerSec =
      dur > 0 && comp > 0 ? Math.round((comp / (dur / 1000)) * 100) / 100 : null;

    const st = store.load(sessionId);
    // Prefer fresh provider usage; fall back to last stored for pie continuity
    const promptTokens = usage
      ? usage.prompt_tokens
      : st?.lastPromptTokens ?? null;
    const totalTokens = usage
      ? usage.total_tokens
      : st?.lastTotalTokens ?? promptTokens;
    const completionTokens = usage
      ? usage.completion_tokens
      : st?.lastCompletionTokens ?? null;

    const telemetry = {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: totalTokens,
      tok_per_sec: tokPerSec,
      ttft_ms: ttftMs != null && Number.isFinite(ttftMs) ? Math.round(ttftMs) : null,
      duration_ms: dur > 0 ? Math.round(dur * 100) / 100 : null,
      finish_reason: errorMsg ? 'error' : finish_reason,
      stop_reason: errorMsg ? null : stop_reason,
      error: errorMsg,
      model: model || null,
      usage_source: usage ? 'provider' : 'cached_or_none',
    };

    if (st && !errorMsg && usage) {
      st.lastPromptTokens = usage.prompt_tokens;
      st.lastTotalTokens = usage.total_tokens;
      st.lastCompletionTokens = usage.completion_tokens;
      store.save(st);
    }

    // Mirror onto AGL agent for callers that read last_prompt_tokens
    if (usage && agent) {
      agent.last_prompt_tokens = usage.prompt_tokens;
      agent.last_completion_tokens = usage.completion_tokens;
      agent.last_total_tokens = usage.total_tokens;
    }

    const wrap = store.appendEvent(sessionId, {
      event_type: 'gen_info',
      duration_ms: dur || null,
      payload: telemetry,
      meta: { telemetry },
    });

    // Gemma 4 / LM Studio: reasoning_content (thinking) for UI brain bubble
    const reasoningText = !errorMsg ? extractReasoningContent(res) : null;
    let reasoningWrap = null;
    if (reasoningText) {
      reasoningWrap = store.appendEvent(sessionId, {
        event_type: 'reasoning',
        duration_ms: dur || null,
        payload: {
          type: 'reasoning',
          text: reasoningText,
          model: model || null,
          reasoning_tokens:
            res?.usage?.completion_tokens_details?.reasoning_tokens ?? null,
        },
        meta: {
          visible: true,
          telemetry: {
            ...telemetry,
            reasoning_chars: reasoningText.length,
          },
        },
      });
    }

    try {
      // Only emit usage when the provider actually reported tokens this round
      if (usage) {
        hooks.onUsage?.({
          type: 'usage',
          ...usage,
          ...telemetry,
          event_id: wrap.event_id,
          sessionId,
        });
      }
      hooks.onGenInfo?.({
        type: 'gen_info',
        ...telemetry,
        event_id: wrap.event_id,
        sessionId,
      });
      const assistantText = extractAssistantText(res);
      if (assistantText) {
        hooks.onAssistant?.({
          type: 'assistant',
          text: assistantText,
          event_id: wrap.event_id,
          sessionId,
          telemetry,
        });
      }
      if (reasoningText) {
        hooks.onReasoning?.({
          type: 'reasoning',
          text: reasoningText,
          model: model || null,
          event_id: reasoningWrap?.event_id || null,
          sessionId,
          reasoning_tokens:
            res?.usage?.completion_tokens_details?.reasoning_tokens ?? null,
          telemetry,
        });
      }
    } catch {
      /* ignore */
    }
    return telemetry;
  }

  function wrapClient(client, wide = false) {
    if (!client?.inference || client.__angelaInstrumented) return client;
    const orig = client.inference.bind(client);
    const proxy = new Proxy(client, {
      get(target, prop, receiver) {
        if (prop === '__angelaInstrumented') return true;
        if (prop === 'inference') {
          return async (req) => {
            const t0 = performance.now();
            let ttftMs = null;
            // Wrap on_delta to capture time-to-first-token when streaming
            const origDelta = req?.on_delta;
            const req2 =
              req && typeof req === 'object'
                ? {
                    ...req,
                    on_delta: (tok, meta) => {
                      if (ttftMs == null) ttftMs = performance.now() - t0;
                      try {
                        origDelta?.(tok, meta);
                      } catch {
                        /* ignore */
                      }
                    },
                  }
                : req;

            store.appendEvent(sessionId, {
              event_type: 'provider_request',
              payload: slimRequestPayload(req, wide),
            });
            if (logSnapshotsEnabled() && req?.messages) {
              store.appendEvent(sessionId, {
                event_type: 'context_window',
                payload: {
                  kind: 'provider_messages_snapshot',
                  messages: req.messages,
                  model: req.model,
                  tools: req.tools ?? null,
                },
              });
            }
            try {
              const res = await orig(req2 && req2.on_delta ? req2 : req);
              const durationMs = performance.now() - t0;
              // Non-streaming: no delta — TTFT ≈ full latency is wrong; leave null
              // unless stream produced a first token.
              store.appendEvent(sessionId, {
                event_type: 'provider_response',
                duration_ms: durationMs,
                payload: res,
              });
              noteUsageAndTelemetry(res, req?.model, durationMs, ttftMs, null);
              return res;
            } catch (err) {
              const durationMs = performance.now() - t0;
              const msg = err?.message || String(err);
              store.appendEvent(sessionId, {
                event_type: 'provider_response',
                duration_ms: durationMs,
                payload: {
                  error: true,
                  message: msg,
                  name: err?.name,
                  status: err?.status,
                  ...(wide ? { _wide: true } : {}),
                },
              });
              noteUsageAndTelemetry(
                null,
                req?.model,
                durationMs,
                ttftMs,
                msg,
              );
              throw err;
            }
          };
        }
        const v = Reflect.get(target, prop, receiver);
        return typeof v === 'function' ? v.bind(target) : v;
      },
    });
    return proxy;
  }

  agent.client = wrapClient(agent.client, false);
  if (agent._wideClient && agent._wideClient !== agent.client) {
    agent._wideClient = wrapClient(agent._wideClient, true);
  }
}

export default SessionStore;
