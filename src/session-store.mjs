/**
 * Durable sessions under `.angela/sessions/`.
 *
 * Pair of files per session:
 *
 *   <id>.json   — **metadata only** (no context-window / transcript body)
 *   <id>.jsonl  — append-only **wrapped** events
 *
 * JSONL line shape (wrapper keeps telemetry out of the verbatim payload):
 *
 *   {
 *     "event_id": "e1a2b3",
 *     "ts": "ISO-8601",
 *     "event_type": "context_window" | "provider_request" | "provider_response"
 *                 | "tool_call" | "tool_result" | "policy" | "session" | "error"
 *                 | "usage" | "gen_info" | …,
 *     "duration_ms": 12.4 | null,
 *     "payload": { … verbatim original … },
 *     "meta": {
 *       "visible": true,           // context_window: include in next provider request
 *       "telemetry": {             // gen stats for assistant turns / provider rounds
 *         "completion_tokens": 947,
 *         "tok_per_sec": 59.94,
 *         "ttft_ms": 130,
 *         "duration_ms": 15800,
 *         "finish_reason": "stop",
 *         "stop_reason": null,
 *         "error": null
 *       }
 *     }
 *   }
 *
 * Line order is the event sequence; session id is the filename stem (not repeated).
 *
 * - context_window  → payload is a chat message (or messages[]) as used for the model
 * - provider_*      → payload is the exact object passed to / returned from the AI provider
 * - other harness events keep their natural payload under `payload`
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
import { randomBytes } from 'node:crypto';
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

  /** Read all jsonl wrappers for a session (newest last). */
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

  /**
   * Snapshot of the most recent provider gen_info telemetry for a session.
   * Used when writing tool/context events so each jsonl line carries its own
   * frozen metrics (UI must not re-point older bubbles at later rounds).
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
   * Patch `meta` on one or more jsonl events by event_id (one rewrite).
   *
   * @param {string} id
   * @param {string|string[]} eventIds
   * @param {object} metaPatch
   * @returns {object[]|null} updated wrappers (null if none matched)
   */
  updateEventsMeta(id, eventIds, metaPatch = {}) {
    const ids = new Set(
      (Array.isArray(eventIds) ? eventIds : [eventIds]).filter(Boolean),
    );
    if (!ids.size) return null;
    const p = this.logPath(id);
    if (!existsSync(p)) return null;
    const lines = readFileSync(p, 'utf8').split('\n');
    /** @type {object[]} */
    const updated = [];
    const out = [];
    for (const line of lines) {
      if (!line.trim()) {
        out.push(line);
        continue;
      }
      let wrap;
      try {
        wrap = JSON.parse(line);
      } catch {
        out.push(line);
        continue;
      }
      const match =
        (wrap.event_id && ids.has(wrap.event_id)) ||
        (!wrap.event_id && wrap.ts && ids.has(wrap.ts));
      if (match) {
        wrap.meta = { ...(wrap.meta || {}), ...metaPatch };
        if (Object.prototype.hasOwnProperty.call(metaPatch, 'visible')) {
          wrap.meta.visible = Boolean(metaPatch.visible);
        }
        updated.push(wrap);
        out.push(JSON.stringify(wrap));
      } else {
        out.push(line);
      }
    }
    if (!updated.length) return null;
    writeFileSync(
      p,
      out.filter((l) => String(l).trim()).join('\n') + '\n',
      'utf8',
    );
    return updated;
  }

  /**
   * Patch `meta` on one jsonl event by event_id.
   * Rewrites the whole log file (sessions are small enough).
   *
   * @returns {object|null} updated wrapper
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
    for (const wrap of this.listEvents(id)) {
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
    for (const wrap of this.listEvents(id)) {
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
              payload: wide
                ? { ...(safeClone(req) || {}), _wide: true }
                : // Don't persist huge message bodies twice with stream flag noise
                  (() => {
                    const c = safeClone(req) || {};
                    if (c.on_delta) delete c.on_delta;
                    return c;
                  })(),
            });
            if (req?.messages) {
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
