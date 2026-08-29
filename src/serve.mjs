/**
 * angela serve — multi-client HTTP chat API over the Angela harness.
 *
 * Bind: 127.0.0.1:8763 (override ANGELA_PORT / --port)
 * Project root: -c / ANGELA_PROJECT_ROOT / cwd
 *
 * Compatible with brain/cowork NDJSON clients:
 *   POST /api/chat          → NDJSON stream
 *   POST /api/approve
 *   POST /api/abort
 *   GET  /api/health
 *   GET  /api/agents
 *   GET  /api/sessions
 *   GET|DELETE /api/sessions/:id
 *   DELETE /api/sessions
 *   POST /api/session/new
 *   POST /api/allowlist
 *   GET  /api/context-window
 *   POST /api/event/meta
 *
 * Adapted from cowork/server.mjs (static SPA + HMR + sheet routes removed).
 */
import { resolve, dirname, join, extname, relative } from 'node:path';
import { existsSync, readFileSync, watch as fsWatch } from 'node:fs';
import {
  Angela,
  ApprovalQueue,
  DEFAULT_LUCY_ALLOWLIST,
  loadAgent,
  listAgents,
  SessionStore,
  resolveProjectRoot,
  ensureAngelaLayout,
  defaultMcpRoot,
  resolveContextWindow,
  resolveContextWindowAsync,
} from './index.mjs';

const PORT = Number(process.env.ANGELA_PORT || process.env.COWORK_PORT || 8763);

/** Project root: env / -c, else cwd. */
let PROJECT_ROOT = resolveProjectRoot(
  process.env.ANGELA_PROJECT_ROOT || process.env.COWORK_ROOT || process.cwd(),
);
ensureAngelaLayout(PROJECT_ROOT);

export function setProjectRoot(path) {
  PROJECT_ROOT = resolveProjectRoot(path || process.cwd());
  ensureAngelaLayout(PROJECT_ROOT);
  return PROJECT_ROOT;
}

const MCP_ROOT = process.env.ANGELA_MCP_ROOT || defaultMcpRoot();
const AUTO_APPROVE = process.env.ANGELA_AUTO_APPROVE === '1' || process.env.COWORK_AUTO_APPROVE === '1';


/** tabId → { harness, session, queue, agentName, model, allowlist, contextWindow, tokensUsed } */
const liveTabs = new Map();
/** tabId → ApprovalQueue while a run is in flight */
const tabQueues = new Map();
/** tabId → { text, overridden } stash before first chat creates a session */
const pendingAllowlists = new Map();

function store() {
  return new SessionStore(PROJECT_ROOT);
}

async function loadAgentSafe(name) {
  try {
    // Agent coffee always from serve -c root; MCP jail may be overridden per chat.
    return await loadAgent(name, { projectRoot: PROJECT_ROOT, mcpRoot: MCP_ROOT });
  } catch (err) {
    console.error('[cowork] loadAgent failed:', name, err?.message || err);
    return null;
  }
}

/**
 * Rewrite local MCP jail envs so file-io / term-local sandboxes match the
 * client's workspace root (gl1 --path), without reloading agent coffee.
 * @param {object[]} mcpList
 * @param {string} sandboxRoot
 */
function applySandboxToMcp(mcpList, sandboxRoot) {
  const root = String(sandboxRoot || '').trim();
  if (!root || !Array.isArray(mcpList)) return mcpList || [];
  return mcpList.map((entry) => {
    if (!entry || typeof entry !== 'object') return entry;
    const env = { ...(entry.env || {}) };
    // file-io
    if (env.MCP_FILE_ROOT != null || entry.name === 'file-io') {
      env.MCP_FILE_ROOT = root;
    }
    // term-local just-bash jail + default cwd
    if (env.MCP_TERM_JB_ROOT != null || entry.name === 'term-local') {
      env.MCP_TERM_JB_ROOT = root;
      env.MCP_TERM_CWD = env.MCP_TERM_CWD || root;
    }
    return { ...entry, env };
  });
}

async function handleAgents() {
  const agents = await listAgents(PROJECT_ROOT);
  // Collect unique model specs and refresh LM Studio context once for all.
  const allModels = new Set();
  for (const a of agents) {
    for (const m of a.models || (a.model ? [a.model] : [])) allModels.add(m);
    if (a.model) allModels.add(a.model);
  }
  /** @type {Map<string, number>} */
  const resolved = new Map();
  await Promise.all(
    [...allModels].map(async (m) => {
      const n = await resolveContextWindowAsync(m, {
        default: 32_768,
        force: true,
      });
      resolved.set(m, n);
    }),
  );

  return Response.json({
    projectRoot: PROJECT_ROOT,
    agents: agents.map((a) => {
      const models = a.models || (a.model ? [a.model] : []);
      /** @type {Record<string, number>} */
      const contextWindows = {};
      for (const m of models) {
        contextWindows[m] =
          resolved.get(m) ??
          resolveContextWindow(m, { default: a.contextWindow || 32_768 });
      }
      // Default pie max for agent default model
      const defaultModel = a.model || models[0] || null;
      const contextWindow = defaultModel
        ? (resolved.get(defaultModel) ??
          resolveContextWindow(defaultModel, {
            default: a.contextWindow || 32_768,
          }))
        : a.contextWindow || 32_768;
      return {
        name: a.name,
        file: a.file,
        model: a.model,
        models,
        contextWindow,
        contextWindows,
        description: a.description || '',
        // full allowlist text for the browser "I" panel seed
        allowlist: a.allowlist || '',
        // Welcome one-clicks from agent coffee (null → client defaults)
        starters: Array.isArray(a.starters) ? a.starters : null,
      };
    }),
  });
}

async function handleSessionsList() {
  const meta = store().listMeta();
  return Response.json({
    projectRoot: PROJECT_ROOT,
    sessions: meta.map((s) => ({
      id: s.id,
      title: s.title || s.agent || s.id,
      agent: s.agent,
      model: s.model,
      createdAt: s.createdAt,
      updatedAt: s.updatedAt,
      status: s.status,
      turns: s.turns || 0,
      contextWindow: s.contextWindow ?? null,
    })),
  });
}

/** Shallow-clone telemetry so UI bubbles never share one mutable object. */
function cloneTelemetry(t) {
  if (!t || typeof t !== 'object') return null;
  try {
    return structuredClone(t);
  } catch {
    return { ...t };
  }
}

function gotoFromMeta(wrap) {
  const g = wrap?.meta?.goto;
  if (!g || typeof g !== 'object') {
    return { enabled: false, label: '' };
  }
  return {
    enabled: Boolean(g.enabled),
    label: String(g.label || ''),
  };
}

function reconstructMessagesFromLog(id) {
  const events = store().listEventsFolded(id);
  const messages = [];
  /** @type {object|null} last gen_info payload (for older logs missing meta.telemetry) */
  let lastGen = null;
  for (const wrap of events) {
    const p = wrap.payload;
    if (wrap.event_type === 'gen_info' && p) {
      lastGen = wrap.meta?.telemetry || p;
      continue;
    }
    if (!p) continue;
    // Prefer frozen meta.telemetry written when the event was appended
    const frozen = wrap.meta?.telemetry || null;
    if (
      wrap.event_type === 'context_window' &&
      p.role &&
      p.content != null &&
      p.kind !== 'provider_messages_snapshot'
    ) {
      const telemetry =
        frozen ||
        (p.role === 'assistant' ? lastGen : null);
      messages.push({
        id: wrap.event_id || `${id}-${messages.length}`,
        eventId: wrap.event_id || null,
        kind: p.role === 'user' ? 'user' : 'assistant',
        text: String(p.content),
        visible: wrap.meta?.visible !== false,
        telemetry: cloneTelemetry(telemetry),
        goto: gotoFromMeta(wrap),
      });
    } else if (
      (wrap.event_type === 'reasoning' || p?.type === 'reasoning') &&
      (p?.text || p?.content)
    ) {
      messages.push({
        id: wrap.event_id || `${id}-rs-${messages.length}`,
        eventId: wrap.event_id || null,
        kind: 'reasoning',
        text: String(p.text || p.content || ''),
        visible: wrap.meta?.visible !== false,
        telemetry: cloneTelemetry(wrap.meta?.telemetry || frozen || lastGen),
        goto: gotoFromMeta(wrap),
      });
    } else if (wrap.event_type === 'tool_call' && p.name) {
      messages.push({
        id: wrap.event_id || `${id}-tc-${messages.length}`,
        eventId: wrap.event_id || null,
        kind: 'tool_call',
        name: p.name,
        args: p.args,
        explanation: p.explanation || null,
        visible: wrap.meta?.visible !== false,
        telemetry: cloneTelemetry(frozen || lastGen),
        full_chars: p.full_chars,
        truncated: p.truncated,
        goto: gotoFromMeta(wrap),
      });
    } else if (wrap.event_type === 'tool_result' && p.name) {
      messages.push({
        id: wrap.event_id || `${id}-tr-${messages.length}`,
        eventId: wrap.event_id || null,
        kind: 'tool_result',
        name: p.name,
        ok: p.ok,
        denied: p.denied,
        text: p.text,
        explanation: p.explanation || null,
        visible: wrap.meta?.visible !== false,
        telemetry: cloneTelemetry(frozen || lastGen),
        full_chars: p.full_chars,
        truncated: p.truncated,
        goto: gotoFromMeta(wrap),
      });
    }
  }
  return messages;
}

async function handleEventMeta(req) {
  const body = await req.json().catch(() => ({}));
  const sessionId = String(body.sessionId || '');
  const eventId = String(body.eventId || '');
  if (!sessionId || !eventId) {
    return Response.json({ error: 'sessionId and eventId required' }, { status: 400 });
  }
  // Visibility: also toggle tool_call ↔ tool_result pair (providers need pairs).
  if (Object.prototype.hasOwnProperty.call(body, 'visible')) {
    const result = store().setEventVisible(
      sessionId,
      eventId,
      Boolean(body.visible),
    );
    if (!result) {
      return Response.json({ error: 'event not found' }, { status: 404 });
    }
    // Live agent: mutate agent.context_window in place (no re-seed / no snapshot filter)
    const vis = Boolean(body.visible);
    for (const entry of liveTabs.values()) {
      if (entry?.session?.id === sessionId && entry.session.setContextVisible) {
        entry.session.setContextVisible(result.eventIds, vis);
      }
    }
    return Response.json({
      ok: true,
      event: result.events[0] || null,
      events: result.events,
      eventIds: result.eventIds,
      paired: result.eventIds.length > 1,
    });
  }

  // Goto bookmark: { enabled, label } stored in meta.goto
  if (Object.prototype.hasOwnProperty.call(body, 'goto')) {
    const gIn = body.goto && typeof body.goto === 'object' ? body.goto : {};
    const goto = {
      enabled: Boolean(gIn.enabled),
      label: String(gIn.label ?? '').trim(),
    };
    // Uniqueness among enabled labels (case-insensitive)
    if (goto.enabled && goto.label) {
      const want = goto.label.toLowerCase();
      for (const wrap of store().listEventsFolded(sessionId)) {
        const og = wrap.meta?.goto;
        if (!og?.enabled) continue;
        const lab = String(og.label || '')
          .trim()
          .toLowerCase();
        if (!lab || lab !== want) continue;
        if ((wrap.event_id || null) === eventId) continue;
        return Response.json(
          {
            error: 'duplicate_goto_label',
            message: `Label "${goto.label}" is already used by another enabled bookmark`,
          },
          { status: 409 },
        );
      }
    }
    const updated = store().updateEventMeta(sessionId, eventId, { goto });
    if (!updated) {
      return Response.json({ error: 'event not found' }, { status: 404 });
    }
    for (const entry of liveTabs.values()) {
      if (entry?.session?.id === sessionId && entry.session.setGotoMeta) {
        entry.session.setGotoMeta(eventId, goto);
      }
    }
    return Response.json({
      ok: true,
      event: updated,
      events: [updated],
      eventIds: [eventId],
      goto,
    });
  }

  const patch = {};
  if (body.telemetry && typeof body.telemetry === 'object') {
    patch.telemetry = body.telemetry;
  }
  const updated = store().updateEventMeta(sessionId, eventId, patch);
  if (!updated) {
    return Response.json({ error: 'event not found' }, { status: 404 });
  }
  return Response.json({ ok: true, event: updated, events: [updated], eventIds: [eventId] });
}

async function handleSessionGet(id) {
  const meta = store().load(id);
  if (!meta) return Response.json({ error: 'not found' }, { status: 404 });
  return Response.json({
    session: meta,
    messages: reconstructMessagesFromLog(id),
  });
}

async function handleSessionDelete(id) {
  const ok = store().delete(id);
  // Drop live tab if it pointed at this session
  for (const [tabId, entry] of liveTabs) {
    if (entry?.session?.id === id) {
      try {
        await entry.harness?.close();
      } catch {
        /* ignore */
      }
      liveTabs.delete(tabId);
    }
  }
  return Response.json({ ok, id });
}

async function handleSessionsClean() {
  for (const [, entry] of liveTabs) {
    try {
      await entry.harness?.close();
    } catch {
      /* ignore */
    }
  }
  liveTabs.clear();
  tabQueues.clear();
  const n = store().clean();
  return Response.json({ ok: true, removed: n });
}

async function closeLiveTab(tabId) {
  const entry = liveTabs.get(tabId);
  if (!entry) return;
  try {
    await entry.harness?.close();
  } catch {
    /* ignore */
  }
  liveTabs.delete(tabId);
}

function applyAllowlistToEntry(entry, text, { persist = false, source = 'ui' } = {}) {
  const al = text == null ? '' : String(text);
  if (entry.harness?.policy) {
    if (typeof entry.harness.policy.setAllowlist === 'function') {
      entry.harness.policy.setAllowlist(al);
    } else {
      entry.harness.policy.allowlist = al;
    }
  }
  entry.allowlist = al;
  entry.allowlistSource = source;
  entry.allowlistOverridden = source === 'ui' || source.startsWith('file:');

  if (persist && entry.session?.id && entry.harness?.sessionStore) {
    const st = entry.harness.sessionStore.load(entry.session.id);
    const prevSource = st?.allowlistSource || 'agent';
    const overrides = { ...(st?.overrides || {}) };
    if (source === 'ui' || source.startsWith('file:')) {
      overrides.allowlist = { from: prevSource === 'ui' ? 'agent' : prevSource, to: source };
    } else {
      delete overrides.allowlist;
    }
    entry.harness.sessionStore.updateMeta(entry.session.id, {
      allowlistSource: source,
      allowlist: source === 'ui' || source.startsWith('file:') ? al : null,
      overrides: Object.keys(overrides).length ? overrides : null,
    });
    entry.harness.sessionStore.appendEvent(entry.session.id, {
      event_type: 'session',
      payload: {
        kind: 'allowlist_updated',
        source,
        bytes: al.length,
      },
    });
  }
}

/**
 * Create or reuse a live angela harness for this browser tab.
 *
 * @param {{ allowlistOverridden?: boolean, resumeSessionId?: string|null }} opts
 *   When allowlistOverridden, `allowlist` is treated like CLI --allowlist (wins over agent).
 *   Otherwise agent coffee allowlist is used and UI text is display-only.
 *   resumeSessionId: open existing disk session (jsonl seed for retain_history).
 */
async function ensureLiveTab({
  tabId,
  agentName,
  model,
  allowlist,
  allowlistOverridden = false,
  forceNew,
  resumeSessionId = null,
  /** Client workspace root → MCP file/term sandbox (gl1 --path). */
  projectRoot: clientProjectRoot = null,
}) {
  const resumeId =
    !forceNew && resumeSessionId ? String(resumeSessionId).trim() : '';
  const sandboxRoot = (() => {
    const raw = clientProjectRoot != null ? String(clientProjectRoot).trim() : '';
    if (raw && !raw.startsWith('mac:')) return raw;
    return PROJECT_ROOT;
  })();

  const existing = liveTabs.get(tabId);
  if (
    existing &&
    !forceNew &&
    existing.agentName === agentName &&
    existing.model === model &&
    existing.sandboxRoot === sandboxRoot &&
    (!resumeId || existing.session?.id === resumeId)
  ) {
    if (allowlistOverridden && allowlist != null) {
      applyAllowlistToEntry(existing, allowlist, {
        persist: true,
        source: 'ui',
      });
    }
    return existing;
  }

  await closeLiveTab(tabId);

  // Prefer agent/model from the disk session when resuming
  let resumeState = null;
  if (resumeId) {
    resumeState = store().load(resumeId);
    if (!resumeState) {
      console.warn('[cowork] resume session missing on disk:', resumeId);
    }
  }

  const def = await loadAgentSafe(
    (resumeState?.agent && String(resumeState.agent)) || agentName,
  );
  if (!def) {
    throw new Error(
      `Agent not found: ${agentName} (project ${PROJECT_ROOT}/.angela/agents/)`,
    );
  }

  // Same precedence as angela CLI: agent coffee is base; UI (like -m / --allowlist)
  // only wins when the user explicitly overrode.
  const overrides = {};
  let effectiveModel = def.model;
  const modelHint =
    (model && String(model).trim()) ||
    (resumeState?.model && String(resumeState.model).trim()) ||
    '';
  if (modelHint) {
    effectiveModel = modelHint;
    if (effectiveModel !== def.model) {
      overrides.model = { from: def.model || null, to: effectiveModel };
    }
  }

  let effectiveAllowlist = def.allowlist || DEFAULT_LUCY_ALLOWLIST;
  let allowlistSource = def.allowlist ? 'agent' : 'default';
  if (allowlistOverridden && allowlist != null) {
    effectiveAllowlist = String(allowlist);
    allowlistSource = 'ui';
    overrides.allowlist = {
      from: def.allowlist ? 'agent' : 'default',
      to: 'ui',
    };
  } else if (resumeState?.allowlistSource === 'ui' && resumeState.allowlist) {
    effectiveAllowlist = String(resumeState.allowlist);
    allowlistSource = 'ui';
  }

  // Pie / Agent.factory max context: live LM Studio loaded_context_length when available
  const contextWindow = await resolveContextWindowAsync(effectiveModel, {
    default: resumeState?.contextWindow || def.contextWindow || 32_768,
    force: true,
  });

  // MCP from agent coffee (loaded under serve -c), then jail remapped to client workspace.
  const mcpRaw = (def.mcp || []).filter((entry) => {
    const script = entry?.args?.[0];
    if (script && !existsSync(script)) {
      console.warn('[cowork] skip MCP (missing script):', entry.name, script);
      return false;
    }
    return true;
  });
  const mcp = applySandboxToMcp(mcpRaw, sandboxRoot);

  const harness = await Angela.create({
    model: effectiveModel,
    mcp,
    allowlist: effectiveAllowlist,
    allowlistSource,
    policyMode: def.policyMode || 'ask',
    system: def.system,
    contextWindow,
    // Agent coffee + session disk stay under serve -c; only MCP jail uses client sandbox.
    projectRoot: PROJECT_ROOT,
    agentName: def.name,
    agentFile: def.path,
    parallel_tools: def.parallel_tools ?? true,
    overrides: Object.keys(overrides).length ? overrides : null,
    onApproval: null, // set per-run
  });

  let session;
  if (resumeState) {
    session = await harness.session.open(resumeId);
    console.log(
      '[cowork] resumed session',
      resumeId,
      'visible context msgs:',
      store().listVisibleContextMessages(resumeId).length,
    );
  } else {
    session = await harness.session.create({
      title: null,
      agent: def.name,
    });
  }

  const entry = {
    harness,
    session,
    agentName: def.name,
    model: effectiveModel,
    sandboxRoot,
    allowlist: effectiveAllowlist,
    allowlistSource,
    allowlistOverridden: allowlistSource === 'ui',
    agentAllowlist: def.allowlist || DEFAULT_LUCY_ALLOWLIST,
    contextWindow,
    tokensUsed: Number(resumeState?.lastPromptTokens) || 0,
    lastPromptTokens:
      resumeState?.lastPromptTokens != null
        ? Number(resumeState.lastPromptTokens)
        : null,
    lastTotalTokens:
      resumeState?.lastTotalTokens != null
        ? Number(resumeState.lastTotalTokens)
        : null,
    models: def.models || (def.model ? [def.model] : []),
    mcpNames: mcp.map((m) => m.name),
  };
  // Consume any pending UI allowlist from pre-session blur
  const pending = pendingAllowlists.get(tabId);
  if (pending?.overridden && !allowlistOverridden) {
    applyAllowlistToEntry(entry, pending.text, {
      persist: true,
      source: 'ui',
    });
  }
  pendingAllowlists.delete(tabId);

  liveTabs.set(tabId, entry);
  return entry;
}

async function handleChatStream(req) {
  const body = await req.json();
  const tabId = String(body.tabId || '1');
  const content = String(body.content || '').trim();
  const agentName = String(body.agent || 'lucy');
  const model = body.model ? String(body.model) : null;
  let allowlist =
    body.allowlist != null ? String(body.allowlist) : undefined;
  // Only treat as CLI --allowlist equivalent when client marks override
  let allowlistOverridden = Boolean(body.allowlistOverridden);
  // Apply pending blur-save from before first message
  if (!allowlistOverridden && pendingAllowlists.has(tabId)) {
    const p = pendingAllowlists.get(tabId);
    if (p?.overridden) {
      allowlist = p.text;
      allowlistOverridden = true;
    }
  }
  const forceNew = Boolean(body.newSession);
  const resumeSessionId = body.sessionId ? String(body.sessionId) : null;
  // gl1 (and other clients) pass the active workspace root so MCP file-io /
  // term-local jails match the editor sandbox, not only serve -c.
  const clientProjectRoot =
    body.projectRoot != null
      ? String(body.projectRoot)
      : body.sandboxRoot != null
        ? String(body.sandboxRoot)
        : null;

  if (!content) {
    return Response.json({ error: 'empty content' }, { status: 400 });
  }

  const queue = new ApprovalQueue();
  tabQueues.set(tabId, queue);

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (obj) => {
        try {
          controller.enqueue(encoder.encode(JSON.stringify(obj) + '\n'));
        } catch {
          /* closed */
        }
      };

      const cards = [];
      const pushCard = (card) => {
        cards.push(card);
        send({ type: 'card', card });
      };

      queue.subscribe((reqItem) => {
        send({
          type: 'approval_needed',
          id: reqItem.id,
          tool: reqItem.tool,
          command: reqItem.command,
          args: reqItem.args,
          reason: reqItem.reason,
          tabId,
        });
        pushCard({
          kind: 'approval_needed',
          id: reqItem.id,
          name: reqItem.tool,
          command: reqItem.command,
          args: reqItem.args,
          status: 'pending',
        });
      });

      const onApproval = AUTO_APPROVE
        ? async () => 'allow'
        : queue.createApprover();

      let entry;
      try {
        // New prompt while previous turn still running → interrupt stream first
        const prior = liveTabs.get(tabId);
        if (prior?.session?.running && !forceNew) {
          try {
            await prior.session.abort('interrupted by new prompt');
          } catch {
            /* ignore */
          }
        }

        entry = await ensureLiveTab({
          tabId,
          agentName,
          model,
          allowlist,
          allowlistOverridden,
          forceNew,
          resumeSessionId: forceNew ? null : resumeSessionId,
          projectRoot: clientProjectRoot,
        });

        // Wire approval for this run
        entry.harness.policy.onApproval = onApproval;
        if (allowlistOverridden && allowlist != null) {
          applyAllowlistToEntry(entry, allowlist, {
            persist: Boolean(entry.session?.id),
            source: 'ui',
          });
        }

        // Title from first user message
        if (entry.harness.sessionStore) {
          const st = entry.harness.sessionStore.load(entry.session.id);
          if (st && !st.title) {
            st.title =
              content.slice(0, 64) + (content.length > 64 ? '…' : '');
            entry.harness.sessionStore.save(st);
          }
        }

        /** @type {object|null} */
        let lastGenInfo = null;
        const onEvent = (ev) => {
          send({ type: 'event', event: ev });
          // gl1: update pie from provider usage after each inference round
          if (ev.type === 'usage' || ev.type === 'gen_info') {
            // Only update pie from real provider counts (not null / missing).
            // Number(null) === 0 would falsely clear the pie.
            if (ev.prompt_tokens != null && ev.prompt_tokens !== '') {
              const prompt = Number(ev.prompt_tokens);
              if (Number.isFinite(prompt) && prompt >= 0) {
                entry.tokensUsed = prompt;
                entry.lastPromptTokens = prompt;
                const total = Number(ev.total_tokens);
                entry.lastTotalTokens = Number.isFinite(total)
                  ? total
                  : prompt;
              }
            }
            lastGenInfo = {
              completion_tokens: ev.completion_tokens ?? null,
              prompt_tokens:
                ev.prompt_tokens != null
                  ? Number(ev.prompt_tokens)
                  : entry.lastPromptTokens ?? null,
              total_tokens:
                ev.total_tokens != null
                  ? Number(ev.total_tokens)
                  : entry.lastTotalTokens ?? null,
              tok_per_sec: ev.tok_per_sec ?? null,
              ttft_ms: ev.ttft_ms ?? null,
              duration_ms: ev.duration_ms ?? null,
              finish_reason: ev.finish_reason ?? null,
              stop_reason: ev.stop_reason ?? null,
              error: ev.error ?? null,
              model: ev.model || entry.model,
              event_id: ev.event_id || null,
              usage_source: ev.usage_source ?? null,
            };
            // Client pie: only push type:usage when we have provider prompt_tokens
            if (
              entry.lastPromptTokens != null &&
              (ev.type === 'usage' || ev.usage_source === 'provider')
            ) {
              send({
                type: 'usage',
                prompt_tokens: entry.lastPromptTokens,
                completion_tokens: lastGenInfo.completion_tokens,
                total_tokens: entry.lastTotalTokens,
                contextWindow: entry.contextWindow,
                sessionId: entry.session?.id,
                model: lastGenInfo.model,
                ...lastGenInfo,
              });
            }
            send({
              type: 'gen_info',
              ...lastGenInfo,
              sessionId: entry.session?.id,
              contextWindow: entry.contextWindow,
            });
            return;
          }
          if (ev.type === 'goto_applied') {
            send({
              type: 'goto_applied',
              sessionId: ev.sessionId || entry.session?.id,
              label: ev.label,
              hideIds: Array.isArray(ev.hideIds) ? ev.hideIds : [],
              keepIds: Array.isArray(ev.keepIds) ? ev.keepIds : [],
              anchorId: ev.anchorId || null,
              last_item_attempted: ev.last_item_attempted,
              last_item_success: ev.last_item_success,
              items_processed: ev.items_processed,
              items_remaining: ev.items_remaining,
              next_item: ev.next_item,
              learnings: ev.learnings,
            });
            return;
          }
          if (
            ev.type === 'loop_start_marker_set' ||
            ev.type === 'waypoint_set'
          ) {
            send({
              type: 'loop_start_marker_set',
              sessionId: ev.sessionId || entry.session?.id,
              eventId: ev.eventId || null,
              label: ev.label || '',
              goto: ev.goto || { enabled: true, label: ev.label || '' },
              clearedIds: Array.isArray(ev.clearedIds) ? ev.clearedIds : [],
              moved: Boolean(ev.moved),
            });
            return;
          }
          if (ev.type === 'context_event') {
            send({
              type: 'context_event',
              event_id: ev.event_id,
              role: ev.role,
              content: ev.content,
              visible: ev.visible !== false,
              // Prefer snapshot frozen on the jsonl event
              telemetry: cloneTelemetry(ev.telemetry || lastGenInfo),
            });
            return;
          }
          if (ev.type === 'assistant_delta') {
            // Live typewriter tokens (LM Studio / streaming providers)
            send({
              type: 'assistant_delta',
              text: ev.text || '',
              sessionId: entry.session?.id,
            });
            return;
          }
          if (ev.type === 'reasoning_delta') {
            send({
              type: 'reasoning_delta',
              text: ev.text || '',
              sessionId: entry.session?.id,
            });
            return;
          }
          if (ev.type === 'reasoning') {
            pushCard({
              kind: 'reasoning',
              text: ev.text || '',
              eventId: ev.event_id || null,
              telemetry: cloneTelemetry(ev.telemetry || lastGenInfo),
              visible: true,
            });
            return;
          }
          if (ev.type === 'tool_call') {
            pushCard({
              kind: 'tool_call',
              name: ev.name,
              args: ev.args,
              explanation: ev.explanation || null,
              eventId: ev.event_id || null,
              // Snapshot at insert time (from event or last gen_info)
              telemetry: cloneTelemetry(ev.telemetry || lastGenInfo),
              visible: true,
            });
          } else if (ev.type === 'tool_result') {
            pushCard({
              kind: 'tool_result',
              name: ev.name,
              ok: ev.ok,
              denied: ev.denied,
              text: ev.text,
              explanation: ev.explanation || null,
              eventId: ev.event_id || null,
              telemetry: cloneTelemetry(ev.telemetry || lastGenInfo),
              visible: true,
              full_chars: ev.full_chars,
              truncated: ev.truncated,
            });
          }
        };
        entry.harness.onEvent = onEvent;

        const tools = (await entry.harness.listMcpTools()).map((t) => t.name);
        // Pie fill = last provider usage.prompt_tokens only (no chars/4)
        const usedBefore = entry.lastPromptTokens ?? 0;

        send({
          type: 'session',
          tabId,
          sessionId: entry.session.id,
          agent: entry.agentName,
          model: entry.model,
          models: entry.models,
          contextWindow: entry.contextWindow,
          tokensUsed: usedBefore,
          tools,
          mcp: entry.mcpNames || [],
          projectRoot: PROJECT_ROOT,
        });

        const result = await entry.session.run({ prompt: content });
        if (result?.command === 'new' && result.session) {
          entry.session = result.session;
          send({
            type: 'session',
            command: 'new',
            tabId,
            sessionId: result.sessionId,
            previousId: result.previousId,
          });
          send({ type: 'done' });
          return;
        }
        const text =
          typeof result === 'string'
            ? result
            : result?.summary ||
              result?.choices?.[0]?.message?.content ||
              (result != null ? JSON.stringify(result) : '');

        // Session holds #agent privately — pie from provider usage events only
        const tokensUsed =
          entry.lastPromptTokens != null ? entry.lastPromptTokens : usedBefore;
        entry.tokensUsed = tokensUsed;

        // Prefer telemetry already frozen on the context_window assistant event
        let asstTelemetry = cloneTelemetry(lastGenInfo);
        try {
          const snap = entry.harness?.sessionStore?.lastGenTelemetry?.(
            entry.session.id,
          );
          if (snap) asstTelemetry = cloneTelemetry(snap);
        } catch {
          /* ignore */
        }
        // Finalize streamed assistant bubble (or create if provider was non-stream).
        // Skip empty / raw-provider dumps (tool-only turns have no user-visible text).
        const asstText = String(text || '').trim();
        const looksLikeProviderDump =
          asstText.startsWith('{') &&
          /"choices"\s*:/.test(asstText) &&
          /"usage"\s*:/.test(asstText);
        if (asstText && !looksLikeProviderDump) {
          pushCard({
            kind: 'assistant',
            text: asstText,
            telemetry: asstTelemetry,
            streaming: false,
          });
        } else {
          // Still seal any live stream bubble on the client via empty finalize
          send({
            type: 'card',
            card: {
              kind: 'assistant',
              text: '',
              telemetry: asstTelemetry,
              streaming: false,
            },
          });
        }
        send({
          type: 'done',
          ok: true,
          cards,
          model: entry.model,
          sessionId: entry.session.id,
          contextWindow: entry.contextWindow,
          tokensUsed,
          prompt_tokens: entry.lastPromptTokens ?? null,
          total_tokens: entry.lastTotalTokens ?? null,
          // Pie / session-level only — do not use this to rewrite bubble metrics
          gen_info: cloneTelemetry(lastGenInfo),
        });
      } catch (err) {
        const msg = err?.message || String(err);
        // AGL throws Error(reason) on cooperative cancel — reason is often
        // "user stop" (no "abort" substring). Treat like gl1 stopPane.
        const aborted =
          entry?.session?._aborted ||
          err?.aborted === true ||
          /abort|user stop|interrupted/i.test(msg);
        if (aborted) {
          send({
            type: 'done',
            ok: true,
            aborted: true,
            cards,
            model: entry?.model || model,
            sessionId: entry?.session?.id,
          });
        } else {
          console.error('[cowork] chat error:', err);
          send({
            type: 'done',
            ok: false,
            aborted: false,
            error: msg,
            cards,
            model: entry?.model || model,
            sessionId: entry?.session?.id,
          });
        }
      } finally {
        queue.clear('deny');
        tabQueues.delete(tabId);
        try {
          controller.close();
        } catch {
          /* ignore */
        }
      }
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Cache-Control': 'no-cache',
      'X-Accel-Buffering': 'no',
    },
  });
}

async function handleApprove(req) {
  const body = await req.json();
  const id = String(body.id || '');
  const decision = body.decision || 'deny';
  const tabId = body.tabId != null ? String(body.tabId) : null;

  let ok = false;
  if (tabId && tabQueues.has(tabId)) {
    ok = tabQueues.get(tabId).resolve(id, decision);
  } else {
    for (const q of tabQueues.values()) {
      if (q.resolve(id, decision)) {
        ok = true;
        break;
      }
    }
  }
  return Response.json({ ok, id, decision });
}

async function handlePending(req) {
  const url = new URL(req.url);
  const tabId = url.searchParams.get('tabId');
  if (tabId && tabQueues.has(tabId)) {
    return Response.json({ pending: tabQueues.get(tabId).list() });
  }
  const all = [];
  for (const [tid, q] of tabQueues) {
    for (const r of q.list()) all.push({ ...r, tabId: tid });
  }
  return Response.json({ pending: all });
}

async function handleAbort(req) {
  const body = await req.json().catch(() => ({}));
  const tabId = String(body.tabId || '1');
  const q = tabQueues.get(tabId);
  if (q) q.clear('deny');
  const entry = liveTabs.get(tabId);
  if (entry?.session) await entry.session.abort('user stop');
  return Response.json({ ok: true, sessionId: entry?.session?.id });
}

async function handleNewSession(req) {
  const body = await req.json().catch(() => ({}));
  const tabId = String(body.tabId || '1');
  await closeLiveTab(tabId);
  return Response.json({ ok: true });
}

/**
 * Persist UI allowlist (on textarea blur) — supersedes agent coffee like
 * `angela --allowlist <file>`. Updates live policy + session .json.
 *
 * POST /api/allowlist { tabId, sessionId?, allowlist, overridden?: true }
 */
async function handleAllowlist(req) {
  const body = await req.json().catch(() => ({}));
  const tabId = String(body.tabId || '1');
  const sessionId = body.sessionId ? String(body.sessionId) : null;
  const text = body.allowlist != null ? String(body.allowlist) : '';
  // If client says overridden=false, restore agent default
  const overridden = body.overridden !== false;

  let entry = liveTabs.get(tabId);

  // Prefer live tab; else patch disk session by id (history view)
  if (entry) {
    const agentBase = entry.agentAllowlist ?? entry.allowlist;
    if (!overridden) {
      applyAllowlistToEntry(entry, agentBase, {
        persist: true,
        source: entry.agentAllowlist ? 'agent' : 'default',
      });
      return Response.json({
        ok: true,
        sessionId: entry.session?.id,
        allowlistSource: entry.allowlistSource,
        overridden: false,
        bytes: String(entry.allowlist || '').length,
      });
    }
    applyAllowlistToEntry(entry, text, { persist: true, source: 'ui' });
    return Response.json({
      ok: true,
      sessionId: entry.session?.id,
      allowlistSource: 'ui',
      overridden: true,
      bytes: text.length,
    });
  }

  if (sessionId) {
    const s = store();
    const st = s.load(sessionId);
    if (!st) return Response.json({ error: 'session not found' }, { status: 404 });
    if (!overridden) {
      s.updateMeta(sessionId, {
        allowlistSource: st.agent ? 'agent' : 'default',
        allowlist: null,
        overrides: (() => {
          const o = { ...(st.overrides || {}) };
          delete o.allowlist;
          return Object.keys(o).length ? o : null;
        })(),
      });
      return Response.json({
        ok: true,
        sessionId,
        allowlistSource: 'agent',
        overridden: false,
      });
    }
    const overrides = { ...(st.overrides || {}) };
    overrides.allowlist = {
      from: st.allowlistSource || 'agent',
      to: 'ui',
    };
    s.updateMeta(sessionId, {
      allowlistSource: 'ui',
      allowlist: text,
      overrides,
    });
    s.appendEvent(sessionId, {
      event_type: 'session',
      payload: { kind: 'allowlist_updated', source: 'ui', bytes: text.length },
    });
    return Response.json({
      ok: true,
      sessionId,
      allowlistSource: 'ui',
      overridden: true,
      bytes: text.length,
    });
  }

  // No live session yet — stash until first chat
  if (overridden) {
    pendingAllowlists.set(tabId, { text, overridden: true });
  } else {
    pendingAllowlists.delete(tabId);
  }

  return Response.json({
    ok: true,
    sessionId: null,
    pending: true,
    allowlistSource: overridden ? 'ui' : null,
    overridden,
    bytes: text.length,
  });
}

/**
 * POST /api/sheet/entities — list *.yaml names (cheap), read one page only.
 * Body: { cwd, offset?: number, limit?: number }
 */
async function handleHealth() {
  let agents = [];
  try {
    agents = await listAgents(PROJECT_ROOT);
  } catch {
    /* ignore */
  }
  return Response.json({
    ok: true,
    port: PORT,
    projectRoot: PROJECT_ROOT,
    autoApprove: AUTO_APPROVE,
    agents: agents.map((a) => a.name),
    mcpRoot: MCP_ROOT,
  });
}


/** @type {{ projectRoot?: string, port?: number, host?: string }} */
export function startServe(opts = {}) {
  const host = opts.host || process.env.ANGELA_HOST || '127.0.0.1';
  const port = Number(opts.port || PORT);
  if (opts.projectRoot) {
    process.env.ANGELA_PROJECT_ROOT = opts.projectRoot;
    setProjectRoot(opts.projectRoot);
  }

  const server = Bun.serve({
    hostname: host,
    port,
    idleTimeout: 255,
    async fetch(req) {
      const url = new URL(req.url);
      const path = url.pathname;
      try {
        if (req.method === 'POST' && path === '/api/chat') return handleChatStream(req);
        if (req.method === 'POST' && path === '/api/approve') return handleApprove(req);
        if (req.method === 'GET' && path === '/api/pending') return handlePending(req);
        if (req.method === 'POST' && path === '/api/abort') return handleAbort(req);
        if (req.method === 'POST' && path === '/api/session/new') return handleNewSession(req);
        if (req.method === 'POST' && path === '/api/allowlist') return handleAllowlist(req);
        if (req.method === 'GET' && path === '/api/health') return handleHealth();
        if (req.method === 'GET' && path === '/api/agents') return handleAgents();
        if (req.method === 'GET' && path === '/api/context-window') {
          const model = url.searchParams.get('model') || '';
          const n = await resolveContextWindowAsync(model, { force: true });
          return Response.json({ model, contextWindow: n });
        }
        if (req.method === 'POST' && path === '/api/event/meta') return handleEventMeta(req);
        if (req.method === 'GET' && path === '/api/sessions') return handleSessionsList();
        if (req.method === 'DELETE' && path === '/api/sessions') return handleSessionsClean();
        if (req.method === 'GET' && path.startsWith('/api/sessions/')) {
          const id = decodeURIComponent(path.slice('/api/sessions/'.length));
          return handleSessionGet(id);
        }
        if (req.method === 'DELETE' && path.startsWith('/api/sessions/')) {
          const id = decodeURIComponent(path.slice('/api/sessions/'.length));
          return handleSessionDelete(id);
        }
        if (req.method === 'GET' && path === '/') {
          return Response.json({
            service: 'angela-serve',
            port,
            projectRoot: PROJECT_ROOT,
            docs: 'POST /api/chat (NDJSON), GET /api/health',
          });
        }
        return new Response('Not found', { status: 404 });
      } catch (err) {
        console.error('[angela serve] request error:', err);
        return Response.json({ error: err?.message || String(err) }, { status: 500 });
      }
    },
  });

  console.error(
    `angela serve http://${host}:${server.port} root=${PROJECT_ROOT} autoApprove=${AUTO_APPROVE}`,
  );
  return server;
}

// CLI entry when run directly: `bun src/serve.mjs` or via `angela serve`
if (import.meta.main) {
  startServe();
}
