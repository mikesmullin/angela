/**
 * angela — thin harness around AGL + MCP-only tools (POC).
 *
 *   import { Angela } from 'angela';
 *   const harness = await Angela.create({ model, mcp, onEvent });
 *   const session = await harness.session.create();
 *   const result = await session.run({ prompt: '...' });
 *   await session.abort();
 *   await harness.close();
 *
 * Always uses a real AGL Agent (no mock agent path). The harness tames the
 * model via system prompt, tool schemas (MCP), max turns, and events.
 */
import Agent from 'agl-ai';
import {
  parseModelSpec,
  resolveContextWindow,
  resolveContextWindowAsync,
} from './model-spec.mjs';
import {
  McpStdioClient,
  mcpResultToText,
  mcpResultToProviderContent,
  mcpResultImageMeta,
  modelSupportsImages,
  toolName,
} from './mcp-client.mjs';
import { PolicyEngine, DEFAULT_LUCY_ALLOWLIST } from './policy.mjs';
import { SessionStore, instrumentProviderLogging } from './session-store.mjs';
import { resolveProjectRoot, ensureAngelaLayout } from './project.mjs';
import {
  loadPlugins,
  attachPluginModelClient,
  abortPlugins,
} from './plugins.mjs';
import {
  startContextHost,
  isContextLoopNavToolName,
} from './context-host.mjs';
import {
  normalizeToolsEnabled,
  filterToolNames,
} from './agent-loader.mjs';
import {
  augmentSystemPrompt,
  normalizeReasoningEffort,
  isGemma4Model,
  REASONING_LEVELS,
} from './think-augment.mjs';

export {
  augmentSystemPrompt,
  normalizeReasoningEffort,
  isGemma4Model,
  REASONING_LEVELS,
  GEMMA_REASONING_PHRASES,
  gemma4ThinkPrefix,
} from './think-augment.mjs';

/**
 * Slash commands the harness handles itself (never sent to the model).
 * `/new` is the whole trimmed prompt (case-insensitive). Trailing words are not a command.
 * @param {unknown} prompt
 * @returns {{ command: 'new' } | null}
 */
export function parseSessionCommand(prompt) {
  const t = String(prompt ?? '').trim();
  if (/^\/new$/i.test(t)) return { command: 'new' };
  return null;
}

function emit(onEvent, type, data = {}) {
  const ev = { type, ts: Date.now(), ...data };
  try {
    onEvent?.(ev);
  } catch (err) {
    console.error('angela onEvent error:', err);
  }
  return ev;
}

/**
 * Tools that can run terminal/shell commands (angela/cowork path only).
 * Used for optional `explanation` (human log) — not for pure file tools.
 */
export function isTerminalCommandTool(serverName, mcpToolName, fullName = '') {
  const s = String(serverName || '').toLowerCase().replace(/_/g, '-');
  const t = String(mcpToolName || '').toLowerCase();
  const n = String(fullName || '').toLowerCase();
  // MCP term-* servers: bash / exec (not list_dir)
  if (
    (s === 'term-local' ||
      s.startsWith('term-') ||
      s.includes('term-local') ||
      s.includes('term_local')) &&
    (t === 'bash' || t === 'exec')
  ) {
    return true;
  }
  // gl1-style names if ever registered as-is
  if (/^terminal__exec$/i.test(n) || /terminal__exec/i.test(n)) return true;
  if (/(?:^|__)(?:bash|exec)$/i.test(n) && /term/i.test(n)) return true;
  return false;
}

/**
 * Register MCP tools onto an AGL agent (Tool method).
 * Every call goes through PolicyEngine (allowlist + onApproval).
 */
export async function registerMcpTools(
  agent,
  clients,
  { onEvent, prefix = true, policy = null } = {},
) {
  const registered = [];
  const supportsImages = modelSupportsImages(agent?.model);
  let progressSeq = 0;
  const inFlight = new Map();
  for (const client of clients) {
    client.onNotification = (msg) => {
      if (msg?.method !== 'notifications/progress') return;
      const p = msg.params || {};
      const meta = inFlight.get(p.progressToken);
      emit(onEvent, 'tool_delta', {
        server: client.name,
        name: meta?.name,
        tool: meta?.tool,
        progressToken: p.progressToken,
        output: p.output,
        text: p.output,
      });
    };
    const tools = await client.listTools();
    for (const t of tools) {
      const usePrefix = client.prefix !== false && prefix !== false;
      const name = usePrefix ? toolName(client.name, t.name) : t.name;
      const description = `[${client.name}] ${t.description || t.name}`;
      const properties = t.inputSchema?.properties || {};
      const required = t.inputSchema?.required || [];

      // Q: optional explanation only on terminal runners (bash/exec), for
      // cowork + angela TUI human log — not file_rsh / list_dir / etc.
      const wantsExplanation = isTerminalCommandTool(
        client.name,
        t.name,
        name,
      );
      const toolProps = wantsExplanation
        ? {
            ...properties,
            explanation: {
              type: 'string',
              description:
                'One sentence on why you are running this command (for the human log / audit). Optional but preferred.',
            },
          }
        : properties;

      agent.Tool(
        name,
        description,
        toolProps,
        required,
        async (_ctx, args) => {
          const raw = args && typeof args === 'object' ? args : {};
          const explanation =
            wantsExplanation &&
            raw.explanation != null &&
            String(raw.explanation).trim()
              ? String(raw.explanation).trim()
              : null;
          // Strip harness-only explanation unless the MCP tool already declares it
          // (e.g. term_rsh__bash keeps it for the result envelope).
          const toolArgs = { ...raw };
          if (
            !Object.prototype.hasOwnProperty.call(properties, 'explanation')
          ) {
            delete toolArgs.explanation;
          }

          emit(onEvent, 'tool_call', {
            server: client.name,
            tool: t.name,
            name,
            args: toolArgs,
            ...(explanation ? { explanation } : {}),
          });

          if (policy) {
            const auth = await policy.authorize({
              tool: name,
              server: client.name,
              mcpTool: t.name,
              args: toolArgs,
            });
            if (!auth.allowed) {
              const msg =
                `DENIED (${auth.via}): tool ${name} not permitted` +
                (auth.request?.command
                  ? ` for command: ${auth.request.command}`
                  : '');
              emit(onEvent, 'tool_result', {
                server: client.name,
                tool: t.name,
                name,
                ok: false,
                denied: true,
                via: auth.via,
                text: msg,
                ...(explanation ? { explanation } : {}),
              });
              return msg;
            }
          }

          try {
            const progressToken = `${name}:${++progressSeq}:${Date.now()}`;
            inFlight.set(progressToken, { name, server: client.name, tool: t.name });
            let result;
            try {
              result = await client.callTool(t.name, toolArgs, { progressToken });
            } finally {
              inFlight.delete(progressToken);
            }
            // Full tool body to model + UI — no ANGELA_TOOL_RESULT_CAP.
            const text = mcpResultToText(result);
            const image = mcpResultImageMeta(result);
            const content = mcpResultToProviderContent(result, { supportsImages });
            emit(onEvent, 'tool_result', {
              server: client.name,
              tool: t.name,
              name,
              ok: !result?.isError,
              text,
              full_chars: text.length,
              truncated: false,
              ...(image ? { image } : {}),
              ...(explanation ? { explanation } : {}),
            });
            return content;
          } catch (err) {
            const msg = err?.message || String(err);
            emit(onEvent, 'tool_result', {
              server: client.name,
              tool: t.name,
              name,
              ok: false,
              text: msg,
              ...(explanation ? { explanation } : {}),
            });
            return `ERROR: ${msg}`;
          }
        },
      );
      registered.push({ server: client.name, mcp: t.name, name, description });
    }
  }
  return registered;
}

class Session {
  #harness;
  #id;
  #agent = null;
  #running = false;
  #turnEvents = [];
  history = [];

  constructor(harness, id) {
    this.#harness = harness;
    this.#id = id;
    /** Set by abort() — cooperative cancel (gl1 stopPane). */
    this._aborted = false;
  }

  get id() {
    return this.#id;
  }

  async #ensureAgent() {
    if (this.#agent) return this.#agent;
    const h = this.#harness;
    const agent = await h._createAgent();
    // Log every provider HTTP round-trip; emit usage for context pie (gl1)
    if (h.sessionStore) {
      instrumentProviderLogging(agent, h.sessionStore, this.#id, {
        onUsage: (u) => {
          emit(h.onEvent, 'usage', u);
        },
        onGenInfo: (g) => {
          emit(h.onEvent, 'gen_info', g);
        },
        onReasoning: (r) => {
          // Instrumenter already wrote jsonl; mirror into agent.context_window
          if (r?.text && agent.context_window) {
            agent.pushContext({
              role: 'reasoning',
              content: String(r.text),
              id: r.event_id || undefined,
              visible: true,
              _persisted: true,
            });
          }
          emit(h.onEvent, 'reasoning', r);
        },
      });
    }
    await registerMcpTools(agent, h.clients, {
      onEvent: (ev) => {
        this.#turnEvents.push(ev);
        this.#onHarnessToolEvent(agent, h, ev);
        h.onEvent?.(ev);
      },
      policy: h.policy,
    });

    // Snapshot full tool registry, then apply toolsEnabled (null = all tools).
    // Disabled tools are removed from agent.tools so they are never sent to the
    // LLM and cannot be invoked (as if agent.Tool() was never called).
    snapshotToolCatalog(agent);
    applyToolsEnabledToAgent(agent, h.toolsEnabled);

    // Hydrate agent.context_window from jsonl only on open/resume (once).
    if (h.sessionStore && agent.retain_history && !agent._angelaCtxSeeded) {
      this.#seedAgentContext(agent, h.sessionStore);
      agent._angelaCtxSeeded = true;
    }
    this.#installContextBridge(agent);

    this.#agent = agent;
    return agent;
  }

  /**
   * Full tool catalog (all registered tools) with enabled flags for the UI.
   * Ensures the AGL agent is created so MCP tools are registered.
   * @returns {Promise<Array<{ name: string, description: string, enabled: boolean }>>}
   */
  async listToolCatalog() {
    const agent = await this.#ensureAgent();
    const catalog = agent._angelaToolCatalog || agent.tools || {};
    const enabledSet = new Set(Object.keys(agent.tools || {}));
    return Object.keys(catalog)
      .sort((a, b) => a.localeCompare(b))
      .map((name) => {
        const fn = catalog[name];
        return {
          name,
          description: fn?._description || '',
          enabled: enabledSet.has(name),
        };
      });
  }

  /**
   * Current toolsEnabled selection: null = all tools; string[] = subset.
   * @returns {string[]|null}
   */
  getToolsEnabled() {
    return this.#harness.toolsEnabled;
  }

  /**
   * Restrict which tools exist on the live AGL agent for inference.
   * null = enable all registered tools; array = only those names.
   * @param {string[]|null|undefined} enabled
   */
  setToolsEnabled(enabled) {
    this.#harness.setToolsEnabled(enabled);
  }

  /**
   * Apply harness.toolsEnabled to this session's agent if already created.
   * @param {string[]|null|undefined} enabled
   */
  applyToolsEnabled(enabled) {
    if (!this.#agent) return;
    applyToolsEnabledToAgent(this.#agent, enabled);
  }

  /**
   * CONTEXT LOOP nav row? (goto / loop_start_marker tool calls + results)
   * @param {object} m
   */
  #isContextLoopNavRow(m) {
    if (!m || typeof m !== 'object') return false;
    if (m.role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length) {
      return m.tool_calls.every((tc) =>
        isContextLoopNavToolName(tc?.function?.name),
      );
    }
    if (m.role === 'tool') {
      return isContextLoopNavToolName(m.name);
    }
    return false;
  }

  /**
   * Hide all context items after the bookmark, except the trailing CONTEXT LOOP
   * nav suffix (latest goto tool_call — and later its tool result stays visible).
   * @param {object|string} args
   * @returns {string}
   */
  applyGoto(args) {
    const agent = this.#agent;
    const store = this.#harness.sessionStore;
    if (!agent?.context_window) {
      return 'goto: error — no agent context window';
    }

    const a =
      args && typeof args === 'object' && !Array.isArray(args)
        ? args
        : { label: args };

    const lab = String(a.label ?? '').trim();
    const lastItem = String(a.last_item_attempted ?? '').trim();
    const successRaw = a.last_item_success;
    const lastOk =
      successRaw === true ||
      successRaw === 'true' ||
      successRaw === 1 ||
      successRaw === '1';
    const lastFail =
      successRaw === false ||
      successRaw === 'false' ||
      successRaw === 0 ||
      successRaw === '0';
    const processed = Number(a.items_processed);
    const remaining = Number(a.items_remaining);
    const nextItem = String(a.next_item ?? '').trim();
    const learnings = String(a.learnings ?? '').trim();

    const missing = [];
    if (!lab) missing.push('label');
    if (!lastItem) missing.push('last_item_attempted');
    if (!lastOk && !lastFail) missing.push('last_item_success');
    if (!Number.isFinite(processed) || processed < 0) missing.push('items_processed');
    if (!Number.isFinite(remaining) || remaining < 0) missing.push('items_remaining');
    if (!nextItem) missing.push('next_item');
    // learnings may be empty string
    if (a.learnings == null) missing.push('learnings');
    if (missing.length) {
      return (
        `goto: error — missing or invalid field(s): ${missing.join(', ')}. ` +
        'Fill every parameter so the next-iteration brief can be generated.'
      );
    }

    const want = lab.toLowerCase();
    let idx = -1;
    let matchedLabel = '';
    for (let i = 0; i < agent.context_window.length; i++) {
      const g = agent.context_window[i]?.goto;
      if (!g?.enabled) continue;
      const gLab = String(g.label || '').trim();
      if (gLab && gLab.toLowerCase() === want) {
        idx = i;
        matchedLabel = gLab;
        break;
      }
    }
    if (idx < 0) {
      return (
        `goto: no enabled bookmark labeled "${lab}". ` +
        'Call loop_start_marker first (or ask the human to set a bookmark).'
      );
    }

    // Keep trailing CONTEXT LOOP nav rows (this goto tool_call, etc.) visible.
    let keepFrom = agent.context_window.length;
    while (
      keepFrom > idx + 1 &&
      this.#isContextLoopNavRow(agent.context_window[keepFrom - 1])
    ) {
      keepFrom--;
    }

    const hideIds = [];
    const keepIds = [];
    for (let i = idx + 1; i < agent.context_window.length; i++) {
      const row = agent.context_window[i];
      if (!row) continue;
      if (i >= keepFrom) {
        // Immune: latest nav suffix (this goto invocation)
        row.visible = true;
        if (row.id) keepIds.push(row.id);
        continue;
      }
      row.visible = false;
      if (row.id) hideIds.push(row.id);
    }

    if (store) {
      if (hideIds.length) {
        store.updateEventsMeta(this.#id, hideIds, { visible: false });
      }
      if (keepIds.length) {
        store.updateEventsMeta(this.#id, keepIds, { visible: true });
      }
    }

    emit(this.#harness.onEvent, 'goto_applied', {
      sessionId: this.#id,
      label: matchedLabel,
      hideIds,
      keepIds,
      anchorId: agent.context_window[idx]?.id || null,
      last_item_attempted: lastItem,
      last_item_success: lastOk,
      items_processed: processed,
      items_remaining: remaining,
      next_item: nextItem,
      learnings,
    });

    const done = remaining === 0 || /^none$/i.test(nextItem);
    const statusLine = lastOk
      ? `Last unit "${lastItem}" succeeded.`
      : `Last unit "${lastItem}" did NOT fully succeed — fix before advancing if needed.`;

    if (done) {
      return [
        `goto: CONTEXT LOOP complete for "${matchedLabel}".`,
        statusLine,
        `Progress: ${processed} processed, 0 remaining.`,
        learnings ? `Learnings: ${learnings}` : 'Learnings: (none)',
        '',
        'STOP the loop. Do not call loop_start_marker or goto again unless the human',
        'asks for more work. Optionally summarize what was done.',
      ].join('\n');
    }

    return [
      `goto: CONTEXT LOOP iteration closed for "${matchedLabel}".`,
      `Hid ${hideIds.length} context item(s) after the marker; kept this goto`,
      `call + this result visible as your handoff.`,
      '',
      '=== WHAT YOU CAN AND CANNOT SEE NOW ===',
      'VISIBLE: system + setup/instructions above the marker, the bookmark itself,',
      '  and this goto brief (and its tool_call).',
      'HIDDEN (intentionally): tool calls/results from the unit you just finished',
      '  (bash, sed, file reads, prior loop_start_marker playbooks, etc.).',
      '  You will NOT find the previous file contents or command transcripts in',
      '  context — that is expected after goto. Do not ask the human to paste them.',
      '  Do not invent tracker files to replace what was hidden.',
      '',
      '=== ITERATION BRIEF (prime for next completion) ===',
      statusLine,
      lastOk
        ? `Do NOT re-process "${lastItem}" — it is done. Advance to the next remaining unit.`
        : `Last unit "${lastItem}" failed/incomplete — on the next iteration, either retry it or pick the true remaining unit after a fresh read.`,
      `Progress snapshot (your report): ${processed} done, ${remaining} remaining.`,
      `Suggested next unit: ${nextItem}`,
      '  (Authoritative: after you re-read the source, the file/list wins if it',
      '  disagrees with this suggestion.)',
      learnings
        ? `Carry-over learnings (keep using these): ${learnings}`
        : 'Carry-over learnings: (none recorded)',
      `Exact label for the next goto: ${matchedLabel}`,
      `  You will use \`goto ${matchedLabel}\` again after the next single unit.`,
      '',
      'DO NEXT (immediately, in order):',
      `1) Call loop_start_marker with label "${matchedLabel}" and the SAME job fields`,
      '   (source_of_truth, unit_of_work, item_action, done_when).',
      '   Required: the previous playbook text was hidden by this goto; this re-injects it.',
      '2) RE-READ source_of_truth with tools (fresh). Never use a remembered list.',
      '3) If done_when is already true on that fresh read → STOP (no more work).',
      `4) Else process EXACTLY ONE remaining unit (prefer "${nextItem}" if still pending).`,
      `5) Call goto again with full status fields (label "${matchedLabel}",`,
      '   last_item_*, items_*, next_item, learnings).',
      '=== end brief ===',
    ].join('\n');
  }

  /**
   * Enable (or move) a goto bookmark on the previous context item.
   * Required job fields produce per-iteration boilerplate in the tool result.
   * @param {object|string} args
   * @returns {string}
   */
  applyLoopStartMarker(args) {
    const agent = this.#agent;
    if (!agent?.context_window?.length) {
      return 'loop_start_marker: error — no agent context window';
    }

    // Back-compat: string-only label
    const a =
      args && typeof args === 'object' && !Array.isArray(args)
        ? args
        : { label: args };
    const lab = String(a.label ?? '').trim();
    const source = String(a.source_of_truth ?? '').trim();
    const unit = String(a.unit_of_work ?? '').trim();
    const action = String(a.item_action ?? '').trim();
    const doneWhen = String(a.done_when ?? '').trim();

    const missing = [];
    if (!lab) missing.push('label');
    if (!source) missing.push('source_of_truth');
    if (!unit) missing.push('unit_of_work');
    if (!action) missing.push('item_action');
    if (!doneWhen) missing.push('done_when');
    if (missing.length) {
      return (
        `loop_start_marker: error — missing required field(s): ${missing.join(', ')}. ` +
        'Fill every parameter so the playbook can be generated.'
      );
    }

    const want = lab.toLowerCase();

    // Skip trailing assistant tool_call rows for nav tools (current tool turn)
    let i = agent.context_window.length - 1;
    while (i >= 0) {
      const m = agent.context_window[i];
      const names = (m?.tool_calls || [])
        .map((tc) => tc?.function?.name)
        .filter(Boolean);
      const isNavTool =
        m?.role === 'assistant' &&
        names.length > 0 &&
        names.every((n) => isContextLoopNavToolName(n));
      if (isNavTool) {
        i--;
        continue;
      }
      break;
    }
    if (i < 0) {
      return (
        'loop_start_marker: error — there is no previous context item to mark. ' +
        'Ensure task instructions exist above, then call loop_start_marker.'
      );
    }

    const row = agent.context_window[i];
    const eventId = row?.id || null;
    if (!eventId) {
      return (
        'loop_start_marker: error — previous context item has no event id yet; ' +
        'try again after it is persisted.'
      );
    }

    // Move: clear any other enabled bookmarks with this label (case-insensitive)
    const clearedIds = [];
    const clearId = (id, keepLabel) => {
      if (!id || id === eventId || clearedIds.includes(id)) return;
      this.setGotoMeta(id, { enabled: false, label: keepLabel });
      clearedIds.push(id);
    };
    for (const m of agent.context_window) {
      if (!m?.goto?.enabled) continue;
      const existingLab = String(m.goto.label || '').trim();
      if (existingLab.toLowerCase() !== want) continue;
      if (m.id === eventId) continue;
      if (m.id) clearId(m.id, existingLab);
      else m.goto = { enabled: false, label: existingLab };
    }
    const store = this.#harness.sessionStore;
    if (store) {
      for (const wrap of store.listEventsFolded(this.#id)) {
        const g = wrap.meta?.goto;
        if (!g?.enabled) continue;
        const existingLab = String(g.label || '').trim();
        if (existingLab.toLowerCase() !== want) continue;
        const eid = wrap.event_id || null;
        if (!eid || eid === eventId) continue;
        clearId(eid, existingLab);
      }
    }

    this.setGotoMeta(eventId, { enabled: true, label: lab });

    const moved = clearedIds.length > 0;
    emit(this.#harness.onEvent, 'loop_start_marker_set', {
      sessionId: this.#id,
      eventId,
      label: lab,
      goto: { enabled: true, label: lab },
      clearedIds,
      moved,
      source_of_truth: source,
      unit_of_work: unit,
      item_action: action,
      done_when: doneWhen,
    });
    emit(this.#harness.onEvent, 'waypoint_set', {
      sessionId: this.#id,
      eventId,
      label: lab,
      goto: { enabled: true, label: lab },
      clearedIds,
      moved,
    });

    // Per-iteration playbook — survives until goto hides it; call this tool
    // again after each goto to re-inject the same instructions.
    const place = moved
      ? `moved bookmark "${lab}" to event ${eventId} (cleared ${clearedIds.length} older placement(s))`
      : `enabled bookmark "${lab}" on previous item (event ${eventId})`;

    return [
      `loop_start_marker: CONTEXT LOOP iteration opened (${place}).`,
      '',
      '=== CONTEXT LOOP PLAYBOOK (follow exactly this iteration) ===',
      `label: ${lab}`,
      `You will use \`goto ${lab}\` after finishing the single unit of work below.`,
      '',
      `source_of_truth: ${source}`,
      `unit_of_work: ${unit}`,
      `item_action: ${action}`,
      `done_when: ${doneWhen}`,
      '',
      'STEPS NOW:',
      `1) RE-READ source_of_truth (${source}) with tools — do not use a remembered list.`,
      `2) If done_when is already true on that fresh read → STOP (do not process; do not goto).`,
      `3) Otherwise pick EXACTLY ONE unit matching: ${unit}`,
      `4) Perform item_action on that unit only: ${action}`,
      `5) Immediately call goto({ label: "${lab}" }) — do not start a second unit first.`,
      '',
      'After goto: call loop_start_marker again with the SAME five fields (re-entrant),',
      'then repeat STEPS NOW until done_when is satisfied on a fresh read.',
      '=== end playbook ===',
    ].join('\n');
  }

  /** @deprecated use applyLoopStartMarker */
  applyWaypoint(label) {
    return this.applyLoopStartMarker(
      typeof label === 'string' ? { label } : label,
    );
  }

  /**
   * Persist / update goto bookmark on a context item (live agent + jsonl meta).
   * @param {string} eventId
   * @param {{ enabled?: boolean, label?: string }} goto
   */
  setGotoMeta(eventId, goto) {
    const agent = this.#agent;
    const store = this.#harness.sessionStore;
    if (!eventId) return false;
    const g = {
      enabled: Boolean(goto?.enabled),
      label: String(goto?.label ?? '').trim(),
    };
    if (agent?.context_window) {
      for (const row of agent.context_window) {
        if (row.id === eventId) {
          row.goto = g;
          break;
        }
      }
    }
    if (store) {
      store.updateEventMeta(this.#id, eventId, { goto: g });
    }
    return true;
  }

  /**
   * tool_call / tool_result / reasoning: 1 jsonl event = 1 context_window item.
   * AGL already pushContext'd the LLM-shaped row; we persist UI event + stamp id.
   */
  #onHarnessToolEvent(agent, h, ev) {
    if (!h.sessionStore || !ev?.type) return;
    const et =
      ev.type === 'tool_call' ||
      ev.type === 'tool_result' ||
      ev.type === 'approval_needed' ||
      ev.type === 'policy' ||
      ev.type === 'reasoning'
        ? ev.type
        : 'harness';
    const snap =
      et === 'tool_call' || et === 'tool_result' || et === 'approval_needed'
        ? h.sessionStore.lastGenTelemetry(this.#id)
        : null;

    // Reasoning is written by instrumenter + mirrored in onReasoning — skip here
    if (et === 'reasoning') return;

    if (et !== 'tool_call' && et !== 'tool_result' && et !== 'approval_needed' && et !== 'policy') {
      if (et === 'harness') {
        h.sessionStore.appendEvent(this.#id, {
          event_type: 'harness',
          payload: ev,
          meta: { visible: true },
        });
      }
      return;
    }

    const wrap = h.sessionStore.appendEvent(this.#id, {
      event_type: et,
      payload: ev,
      meta: {
        visible: true,
        ...(snap?.gen_info_event_id
          ? { gen_info_event_id: snap.gen_info_event_id }
          : {}),
      },
    });
    if (wrap?.event_id) ev.event_id = wrap.event_id;
    if (snap) ev.telemetry = snap;

    if (!wrap?.event_id || !agent.context_window?.length) return;

    if (et === 'tool_call' && ev.name) {
      for (let i = agent.context_window.length - 1; i >= 0; i--) {
        const m = agent.context_window[i];
        if (
          m.role === 'assistant' &&
          m.tool_calls?.[0]?.function?.name === ev.name
        ) {
          m.id = wrap.event_id;
          m.name = ev.name;
          m.args = ev.args;
          m.explanation = ev.explanation || null;
          m._persisted = true;
          if (m.tool_calls[0]) {
            m.tool_calls[0].id = m.tool_calls[0].id || wrap.event_id;
          }
          break;
        }
      }
    } else if (et === 'tool_result') {
      // AGL pushes tool row after the MCP callback returns — mark pending id
      agent._pendingToolResultEventId = wrap.event_id;
      agent._pendingToolResultMeta = {
        name: ev.name,
        ok: ev.ok,
        denied: ev.denied,
        explanation: ev.explanation || null,
      };
    }
  }

  /**
   * Load jsonl → agent.context_window once (open/resume). Memory is primary after.
   */
  #seedAgentContext(agent, store) {
    const items = store.buildContextWindowItems(this.#id);
    agent.context_window = [];
    for (const m of items || []) {
      if (!m?.role) continue;
      agent.context_window.push({
        visible: m.visible !== false,
        ...m,
        id: m.id || m.event_id,
        _persisted: true,
      });
    }
  }

  /**
   * Bridge: when AGL appends to context_window, persist freeform user/assistant
   * to jsonl (tools/reasoning go through #onHarnessToolEvent).
   */
  #installContextBridge(agent) {
    if (agent._angelaCtxBridge) return;
    agent._angelaCtxBridge = true;
    const session = this;
    const store = this.#harness.sessionStore;
    const orig = agent.pushContext.bind(agent);
    agent.pushContext = (item) => {
      // Stamp tool_result id from pending MCP event
      if (
        item?.role === 'tool' &&
        agent._pendingToolResultEventId &&
        !item.id
      ) {
        item = {
          ...item,
          id: agent._pendingToolResultEventId,
          ...agent._pendingToolResultMeta,
          _persisted: true,
        };
        agent._pendingToolResultEventId = null;
        agent._pendingToolResultMeta = null;
      }

      const row = orig(item);

      // Persist freeform user/assistant (not already on disk)
      if (
        store &&
        !row._persisted &&
        (row.role === 'user' ||
          (row.role === 'assistant' && !row.tool_calls) ||
          row.role === 'reasoning')
      ) {
        session.#persistContextItem(row);
      }
      return row;
    };
  }

  #persistContextItem(row) {
    const store = this.#harness.sessionStore;
    if (!store || row._persisted) return;
    const visible = row.visible !== false;
    let wrap = null;
    if (row.role === 'user' || (row.role === 'assistant' && !row.tool_calls)) {
      wrap = store.appendEvent(this.#id, {
        event_type: 'context_window',
        payload: { role: row.role, content: row.content ?? '' },
        meta: { visible },
      });
    } else if (row.role === 'reasoning') {
      wrap = store.appendEvent(this.#id, {
        event_type: 'reasoning',
        payload: { type: 'reasoning', text: row.content ?? '' },
        meta: { visible },
      });
    }
    if (wrap?.event_id) {
      row.id = wrap.event_id;
      row._persisted = true;
    }
  }

  /**
   * End-of-turn: ensure every in-memory context item is on disk (crash recovery).
   */
  #flushContextWindowToDisk(agent) {
    const store = this.#harness.sessionStore;
    if (!store || !agent?.context_window) return;
    for (const row of agent.context_window) {
      if (row._persisted) continue;
      if (row.role === 'assistant' && row.tool_calls) {
        // tool_call events come from MCP path; if missing, write now
        const tc = row.tool_calls[0];
        const name = tc?.function?.name;
        if (!name) continue;
        let args = {};
        try {
          args = JSON.parse(tc.function.arguments || '{}');
        } catch {
          args = {};
        }
        const wrap = store.appendEvent(this.#id, {
          event_type: 'tool_call',
          payload: {
            type: 'tool_call',
            name,
            args,
            id: tc.id,
          },
          meta: { visible: row.visible !== false },
        });
        if (wrap?.event_id) {
          row.id = wrap.event_id;
          row._persisted = true;
        }
      } else if (row.role === 'tool') {
        const wrap = store.appendEvent(this.#id, {
          event_type: 'tool_result',
          payload: {
            type: 'tool_result',
            name: row.name || 'tool',
            text: row.content,
            tool_call_id: row.tool_call_id,
            ok: row.ok !== false,
          },
          meta: { visible: row.visible !== false },
        });
        if (wrap?.event_id) {
          row.id = wrap.event_id;
          row._persisted = true;
        }
      } else {
        this.#persistContextItem(row);
      }
    }
    // Persist visibility for all items (in case eye toggled mid-turn without other writes)
    // Eye toggle already writes immediately — nothing else needed.
  }

  /**
   * Eye toggle: update agent.context_window item(s) by event id.
   * Jsonl meta is updated by the caller (session-store).
   */
  setContextVisible(eventIds, visible) {
    const agent = this.#agent;
    if (!agent?.setContextVisible) return false;
    const ids = Array.isArray(eventIds) ? eventIds : [eventIds];
    let ok = false;
    for (const id of ids) {
      if (id != null && agent.setContextVisible(id, visible)) ok = true;
    }
    return ok;
  }

  /**
   * Whether `label` is free for an enabled goto (case-insensitive).
   * @param {string} label
   * @param {string|null} [exceptEventId] allow this event to keep its own label
   */
  isGotoLabelUnique(label, exceptEventId = null) {
    const want = String(label || '')
      .trim()
      .toLowerCase();
    if (!want) return true;
    const store = this.#harness.sessionStore;
    if (!store) return true;
    for (const wrap of store.listEventsFolded(this.#id)) {
      const g = wrap.meta?.goto;
      if (!g?.enabled) continue;
      const lab = String(g.label || '')
        .trim()
        .toLowerCase();
      if (!lab || lab !== want) continue;
      const eid = wrap.event_id || null;
      if (exceptEventId && eid === exceptEventId) continue;
      return false;
    }
    return true;
  }

  get agent() {
    return this.#agent;
  }

  /**
   * `/new` (whole prompt): close this session and return a fresh empty one.
   * Callers must switch to `result.session` — this object is the previous session.
   */
  async #replaceWithEmptySession() {
    if (this.#running || this.#agent) {
      try {
        await this.abort('new session');
      } catch {
        /* ignore */
      }
    }
    const store = this.#harness.sessionStore;
    store?.appendEvent(this.#id, {
      event_type: 'session',
      payload: { kind: 'replaced', command: '/new' },
    });
    store?.setStatus(this.#id, 'closed');
    let title = null;
    try {
      title = store?.load(this.#id)?.title || null;
    } catch {
      /* ignore */
    }
    const next = await this.#harness.session.create({ title });
    emit(this.#harness.onEvent, 'session_new', {
      command: 'new',
      sessionId: next.id,
      previousId: this.#id,
    });
    emit(this.#harness.onEvent, 'status', {
      sessionId: next.id,
      phase: 'ready',
      previousId: this.#id,
    });
    return {
      command: 'new',
      sessionId: next.id,
      previousId: this.#id,
      session: next,
    };
  }

  async run({ prompt, maxTurns, ...ctx } = {}) {
    if (!prompt) throw new Error('session.run requires prompt');
    const cmd = parseSessionCommand(prompt);
    if (cmd?.command === 'new') {
      return this.#replaceWithEmptySession();
    }
    const agent = await this.#ensureAgent();
    this.#running = true;
    this._aborted = false;
    this.#turnEvents = [];
    const store = this.#harness.sessionStore;
    const turnT0 = performance.now();

    // 1 jsonl event = 1 context item (user). Memory first, then disk.
    const userEv = store?.appendEvent(this.#id, {
      event_type: 'context_window',
      payload: { role: 'user', content: prompt },
      meta: { visible: true },
    });
    const userCtxId = userEv?.event_id || null;
    if (agent.retain_history) {
      agent.pushContext({
        role: 'user',
        content: prompt,
        id: userCtxId,
        visible: true,
        _persisted: true,
      });
    }

    emit(this.#harness.onEvent, 'context_event', {
      event_id: userEv?.event_id,
      role: 'user',
      content: prompt,
      visible: true,
    });
    store?.setStatus(this.#id, 'running');
    emit(this.#harness.onEvent, 'status', {
      sessionId: this.#id,
      phase: 'running',
    });

    try {
      void maxTurns;
      const result = await agent.run({
        prompt,
        skip_user_append: Boolean(agent.retain_history),
        user_id: userCtxId,
        ...ctx,
      });
      const summary = summarizeResult(result);
      this.history.push({ role: 'user', content: prompt });
      this.history.push({ role: 'assistant', content: summary });

      const telemetry = store?.lastGenTelemetry(this.#id) || null;

      // Freeform assistant may already be on context_window from AGL; ensure disk
      let asstEv = null;
      const lastAsst = [...(agent.context_window || [])]
        .reverse()
        .find((m) => m.role === 'assistant' && !m.tool_calls);
      if (lastAsst && !lastAsst._persisted) {
        asstEv = store?.appendEvent(this.#id, {
          event_type: 'context_window',
          payload: {
            role: 'assistant',
            content: lastAsst.content ?? summary,
          },
          meta: {
            visible: lastAsst.visible !== false,
            ...(telemetry ? { telemetry } : {}),
          },
          duration_ms: performance.now() - turnT0,
        });
        if (asstEv?.event_id) {
          lastAsst.id = asstEv.event_id;
          lastAsst._persisted = true;
        }
      } else if (summary && !lastAsst) {
        asstEv = store?.appendEvent(this.#id, {
          event_type: 'context_window',
          payload: { role: 'assistant', content: summary },
          meta: {
            visible: true,
            ...(telemetry ? { telemetry } : {}),
          },
          duration_ms: performance.now() - turnT0,
        });
        if (agent.retain_history && asstEv) {
          agent.pushContext({
            role: 'assistant',
            content: summary,
            id: asstEv.event_id,
            visible: true,
            _persisted: true,
          });
        }
      } else if (lastAsst) {
        asstEv = { event_id: lastAsst.id };
      }

      this.#flushContextWindowToDisk(agent);

      emit(this.#harness.onEvent, 'context_event', {
        event_id: asstEv?.event_id || lastAsst?.id,
        role: 'assistant',
        content: summary || lastAsst?.content || '',
        visible: true,
        telemetry,
      });
      store?.appendEvent(this.#id, {
        event_type: 'harness',
        duration_ms: performance.now() - turnT0,
        payload: { kind: 'run_result', result },
      });
      store?.recordTurnComplete(this.#id, {
        duration_ms: performance.now() - turnT0,
      });

      emit(this.#harness.onEvent, 'final', {
        sessionId: this.#id,
        result: summary,
        raw: result,
        event_id: asstEv?.event_id || lastAsst?.id,
        telemetry,
      });
      emit(this.#harness.onEvent, 'status', {
        sessionId: this.#id,
        phase: 'ready',
      });
      return result;
    } catch (err) {
      // Persist partial context_window so Stop/interrupt is recoverable
      try {
        this.#flushContextWindowToDisk(agent);
      } catch {
        /* ignore */
      }
      const msg = err?.message || String(err);
      const wasAbort =
        agent.aborted ||
        this._aborted ||
        /abort|user stop|interrupted/i.test(msg);
      if (wasAbort) {
        store?.appendEvent(this.#id, {
          event_type: 'session',
          duration_ms: performance.now() - turnT0,
          payload: { kind: 'aborted', reason: msg },
        });
        store?.setStatus(this.#id, 'ready');
        emit(this.#harness.onEvent, 'aborted', {
          sessionId: this.#id,
          reason: msg,
        });
        const e = new Error(msg);
        e.aborted = true;
        emit(this.#harness.onEvent, 'status', {
          sessionId: this.#id,
          phase: 'ready',
        });
        throw e;
      }
      store?.appendEvent(this.#id, {
        event_type: 'error',
        duration_ms: performance.now() - turnT0,
        payload: {
          message: msg,
          name: err?.name,
        },
      });
      store?.setStatus(this.#id, 'error');
      emit(this.#harness.onEvent, 'error', {
        sessionId: this.#id,
        message: msg,
      });
      emit(this.#harness.onEvent, 'status', {
        sessionId: this.#id,
        phase: 'ready',
      });
      throw err;
    } finally {
      this.#running = false;
    }
  }

  async abort(reason = 'user stop') {
    this._aborted = true;
    // AGL: abort in-flight provider stream (AbortController on inference)
    if (this.#agent?.abort) {
      this.#agent.abort(reason);
    }
    // Kill in-flight remote / plugin tool work (optional plugins)
    await abortPlugins(reason);
    this.#harness.sessionStore?.appendEvent(this.#id, {
      event_type: 'session',
      payload: { kind: 'aborted', reason },
    });
    emit(this.#harness.onEvent, 'aborted', { sessionId: this.#id, reason });
  }

  get running() {
    return this.#running;
  }
}

/**
 * Human-facing text for the final assistant bubble / context_window.
 * Never dump a raw provider payload (choices/usage/tool_calls) into the UI.
 */
function summarizeResult(result) {
  if (result == null) return '';
  if (typeof result === 'object' && result.command) return '';
  if (typeof result === 'string') {
    // Guard: if someone already stringified a provider response, don't show it.
    const t = result.trim();
    if (t.startsWith('{') && /"choices"\s*:/.test(t) && /"usage"\s*:/.test(t)) {
      try {
        return summarizeResult(JSON.parse(t));
      } catch {
        return '';
      }
    }
    return result;
  }
  if (typeof result === 'object') {
    if (result.summary != null && result.summary !== '') {
      return String(result.summary);
    }
    if (result.output !== undefined && result.output !== null) {
      return typeof result.output === 'string'
        ? result.output
        : JSON.stringify(result.output);
    }
    const msg = result.choices?.[0]?.message;
    if (msg) {
      const content = msg.content;
      if (content != null && String(content).trim() !== '') {
        return String(content);
      }
      const calls = msg.tool_calls;
      if (Array.isArray(calls) && calls.length) {
        // Tool loop usually continues; if we still surface this, keep it short.
        const names = calls
          .map((c) => c?.function?.name)
          .filter(Boolean);
        return names.length
          ? `(calling ${names.join(', ')})`
          : '';
      }
    }
    // Prefer AGL helper when present
    try {
      const last = Agent.lastAssistantResponse?.(result);
      if (last) return String(last);
    } catch {
      /* ignore */
    }
    return '';
  }
  return String(result);
}

/**
 * Snapshot agent.tools as the full catalog (before toolsEnabled filtering).
 * @param {object} agent
 */
export function snapshotToolCatalog(agent) {
  if (!agent || agent._angelaToolCatalog) return;
  agent._angelaToolCatalog = { ...(agent.tools || {}) };
}

/**
 * Rebuild agent.tools from the full catalog using toolsEnabled.
 * @param {object} agent
 * @param {string[]|null|undefined} enabled null/undefined = all tools
 */
export function applyToolsEnabledToAgent(agent, enabled) {
  if (!agent) return;
  snapshotToolCatalog(agent);
  const catalog = agent._angelaToolCatalog || {};
  const names = filterToolNames(Object.keys(catalog), enabled);
  const next = {};
  for (const name of names) {
    if (catalog[name]) next[name] = catalog[name];
  }
  agent.tools = next;
  agent._angelaToolsEnabled = enabled == null ? null : [...names];
}

export class Angela {
  clients = [];
  onEvent = null;
  /** @type {PolicyEngine} */
  policy = null;
  /** @type {SessionStore|null} */
  sessionStore = null;
  projectRoot = null;
  /**
   * toolsEnabled gate for inference: null = all tools; string[] = subset.
   * UI overrides should set this via setToolsEnabled (authoritative).
   * @type {string[]|null}
   */
  toolsEnabled = null;
  #config;
  #sessions = new Map();
  /** Un-augmented system prompt (think prefix applied via #applySystemPrompt). */
  #systemBase = null;
  /** @type {{ path: string, close: () => void }|null} */
  #contextHost = null;

  constructor(config) {
    this.#config = config;
    // think + reasoning_effort (Angela harness; see think-augment.mjs)
    this.#config.think = Boolean(config.think);
    this.#config.reasoning_effort = normalizeReasoningEffort(
      config.reasoning_effort ?? config.reasoningEffort ?? null,
    );
    this.onEvent = config.onEvent || null;
    this.projectRoot = resolveProjectRoot(config.projectRoot);
    ensureAngelaLayout(this.projectRoot);
    this.sessionStore =
      config.persistSessions === false
        ? null
        : new SessionStore(this.projectRoot);
    this.policy = new PolicyEngine({
      allowlist: config.allowlist ?? DEFAULT_LUCY_ALLOWLIST,
      onApproval: config.onApproval ?? null,
      mode: config.policyMode || 'ask',
      onEvent: (ev) => emit(this.onEvent, ev.type, ev),
    });
    // null = all tools; string[] from agent def or UI
    this.toolsEnabled = normalizeToolsEnabled(config.toolsEnabled);
    if (config.system != null) this.#systemBase = String(config.system);
  }

  /**
   * Update toolsEnabled and re-filter any live session agents.
   * null/undefined = all tools; array/string = only those names (may be empty).
   * @param {string[]|string|null|undefined} enabled
   */
  setToolsEnabled(enabled) {
    if (enabled == null) {
      this.toolsEnabled = null;
    } else {
      // Explicit selection (including []) — never collapse back to "all"
      this.toolsEnabled = normalizeToolsEnabled(enabled) ?? [];
    }
    for (const session of this.#sessions.values()) {
      try {
        session.applyToolsEnabled?.(this.toolsEnabled);
      } catch {
        /* ignore */
      }
    }
  }

  /**
   * Set (or replace) the base system prompt used on next Agent.factory / live agent.
   * Think-token / Gemma effort phrases are applied on top when think is enabled.
   * Call lazily right before session.run — e.g. to inject fresh system_time.
   * @param {string|null|undefined} text
   */
  setSystemPrompt(text) {
    this.#systemBase = text == null ? null : String(text);
    this.#applySystemPrompt();
  }

  /**
   * Enable/disable Gemma-4 `<|think|>` system-prompt injection (no-op for other models).
   * @param {boolean} on
   */
  setThink(on) {
    this.#config.think = Boolean(on);
    this.#applySystemPrompt();
  }

  /** @returns {boolean} */
  getThink() {
    return Boolean(this.#config.think);
  }

  /**
   * Set reasoning effort (low|medium|high|xhigh|max). Passed to AGL providers;
   * for Gemma 4 + think, also selects the depth phrase after `<|think|>`.
   * @param {string|null|undefined} level
   */
  setReasoningEffort(level) {
    this.#config.reasoning_effort = normalizeReasoningEffort(level);
    this.#applySystemPrompt();
    this.#applyReasoningEffortToAgents();
  }

  /** @returns {string|null} */
  getReasoningEffort() {
    return this.#config.reasoning_effort ?? null;
  }

  /** @returns {string|null|undefined} augmented system prompt currently in effect */
  getSystemPrompt() {
    return this.#config.system;
  }

  /** @returns {string|null} base system prompt before think augmentation */
  getSystemPromptBase() {
    return this.#systemBase;
  }

  #applySystemPrompt() {
    const base =
      this.#systemBase != null
        ? this.#systemBase
        : this.#config.system != null
          ? String(this.#config.system)
          : '';
    const augmented = augmentSystemPrompt(base, {
      model: this.#config.model,
      think: this.#config.think,
      reasoning_effort: this.#config.reasoning_effort,
    });
    this.#config.system = augmented;
    for (const session of this.#sessions.values()) {
      try {
        const agent = session.agent;
        if (agent) {
          agent.system_prompt = augmented || '';
          agent.reasoning_effort = this.#config.reasoning_effort ?? null;
        }
      } catch {
        /* ignore */
      }
    }
  }

  #applyReasoningEffortToAgents() {
    const effort = this.#config.reasoning_effort ?? null;
    for (const session of this.#sessions.values()) {
      try {
        if (session.agent) session.agent.reasoning_effort = effort;
      } catch {
        /* ignore */
      }
    }
  }

  static async create(config = {}) {
    // Optional private plugins (angela/plugins/*) — no-op if absent
    await loadPlugins();
    const inst = new Angela(config);
    inst.#contextHost = await startContextHost(inst);
    await inst.#connectMcp();
    emit(inst.onEvent, 'session_init', {
      mcpServers: inst.clients.map((c) => c.name),
      model: config.model || process.env.FAV_LOCAL_LLM || Agent.default.model,
      policyMode: inst.policy.mode,
      projectRoot: inst.projectRoot,
    });
    return inst;
  }

  /**
   * Session whose agent loop is running, else the most recently created one.
   * Used by the context-loop MCP host to route goto / loop_start_marker.
   */
  activeSession() {
    for (const s of this.#sessions.values()) {
      if (s.running) return s;
    }
    const ids = [...this.#sessions.keys()];
    if (!ids.length) return null;
    return this.#sessions.get(ids[ids.length - 1]) || null;
  }

  /** Unix socket path for optional MCP context-window calls (`ANGELA_HOST_SOCK`). */
  get contextHostPath() {
    return this.#contextHost?.path || null;
  }

  async #connectMcp() {
    const list = this.#config.mcp || [];
    const hostSock = this.#contextHost?.path || '';
    for (const entry of list) {
      try {
        const client = new McpStdioClient({
          name: entry.name,
          command: entry.command,
          args: entry.args || [],
          env: {
            ...(entry.env || {}),
            ...(hostSock ? { ANGELA_HOST_SOCK: hostSock } : {}),
          },
          cwd: entry.cwd,
          onStderr: entry.onStderr,
          prefix: entry.prefix,
          framing: entry.framing,
        });
        await client.connect();
        this.clients.push(client);
      } catch (err) {
        console.error(
          `angela: MCP ${entry.name || '?'} failed: ${err?.message || err}`,
        );
      }
    }
  }

  async _createAgent() {
    const modelDisplay =
      this.#config.model ||
      process.env.FAV_LOCAL_LLM ||
      process.env.COWORK_MODEL ||
      'lm-studio:google/gemma-4-12b-qat';

    // host:provider:model (e.g. laptop:copilot:gpt-5.6-luna) — strip host for AGL
    const parsed = parseModelSpec(modelDisplay);
    const modelAgl = parsed.agl || modelDisplay;

    // Prefer base + augment so think/effort stay consistent with setSystemPrompt.
    const systemBase =
      this.#systemBase != null
        ? this.#systemBase
        : this.#config.system ||
          'You are a helpful coding assistant. Use tools when needed. Prefer concise answers.';
    if (this.#systemBase == null && this.#config.system == null) {
      this.#systemBase = systemBase;
    }
    const system = augmentSystemPrompt(systemBase, {
      model: modelDisplay,
      think: this.#config.think,
      reasoning_effort: this.#config.reasoning_effort,
    });
    this.#config.system = system;

    // Conversational sessions (cowork / TUI) need multi-turn context: each
    // session.run() must resend prior user/assistant/tool turns. AGL default
    // is one-shot; enable retain_history unless the caller opts out.
    const retain_history = this.#config.retain_history !== false;

    // Token budget (number) — AGL context_window_size.
    // Prefer live LM Studio loaded_context_length over static 32k tables.
    const context_window_size =
      this.#config.contextWindow ??
      (await resolveContextWindowAsync(modelDisplay)) ??
      null;

    // Streaming on by default (LM Studio SSE + copilot when tool-free).
    // Opt out: Angela.create({ stream: false }) or ANGELA_STREAM=0.
    const streamDefault = process.env.ANGELA_STREAM !== '0';
    const stream =
      this.#config.stream != null ? Boolean(this.#config.stream) : streamDefault;

    const reasoning_effort = this.#config.reasoning_effort ?? null;

    const harness = this;
    const wrappedOnDelta = (tok, meta) => {
      const text = tok == null ? '' : String(tok);
      if (!text) return;
      const channel =
        meta && typeof meta === 'object' && meta.channel
          ? String(meta.channel)
          : 'content';
      try {
        if (channel === 'reasoning') {
          emit(harness.onEvent, 'reasoning_delta', { text });
        } else {
          emit(harness.onEvent, 'assistant_delta', { text });
        }
      } catch {
        /* ignore */
      }
      try {
        harness.#config.on_delta?.(tok, meta);
      } catch {
        /* ignore */
      }
    };

    const agent = await Agent.factory({
      ...this.#config,
      model: modelAgl,
      system_prompt: system,
      parallel_tools: this.#config.parallel_tools ?? true,
      stream,
      reasoning_effort,
      on_delta: wrappedOnDelta,
      retain_history,
      ...(context_window_size != null ? { context_window_size } : {}),
      ...(this.#config.output_tool ? { output_tool: this.#config.output_tool } : {}),
    });

    // Optional plugins may attach a remote/custom provider client (host:provider:model).
    if (parsed.viaRemote || parsed.viaRsh) {
      const handled = await attachPluginModelClient(agent, parsed);
      if (!handled) {
        throw new Error(
          `angela: model host "${parsed.host}" requires an optional plugin ` +
            `(none handled attachModelClient). Spec: ${modelDisplay}`,
        );
      }
      agent.model_display = parsed.display;
      agent.model_host = parsed.host;
    } else {
      agent.model_display = parsed.display || modelAgl;
      agent.model_host = 'local';
    }

    return agent;
  }

  get session() {
    const self = this;
    return {
      create: async ({ title, agent } = {}) => {
        let id;
        if (self.sessionStore) {
          const cfg = self.#config;
          // Persist full allowlist text only when overridden (cli file / ui),
          // same idea as recording --allowlist path on the CLI.
          const allowlistSource = cfg.allowlistSource || null;
          const storeAllowlistText =
            allowlistSource &&
            allowlistSource !== 'agent' &&
            allowlistSource !== 'default'
              ? cfg.allowlist ?? null
              : null;
          const state = self.sessionStore.create({
            agent: agent || cfg.agentName || null,
            agentFile: cfg.agentFile || null,
            model: cfg.model,
            title: title || null,
            mcp: (self.clients || []).map((c) => c.name),
            policyMode: cfg.policyMode || 'ask',
            approveMode: cfg.approveMode ?? null,
            allowlistSource,
            allowlist: storeAllowlistText,
            contextWindow: cfg.contextWindow ?? null,
            overrides: cfg.overrides || null,
            config: {
              model: cfg.model,
              mcp: (self.clients || []).map((c) => c.name),
              policyMode: cfg.policyMode || 'ask',
              approveMode: cfg.approveMode ?? null,
              allowlistSource,
              contextWindow: cfg.contextWindow ?? null,
              agent: agent || cfg.agentName || null,
              agentFile: cfg.agentFile || null,
              projectRoot: self.projectRoot,
              allowlistBytes: cfg.allowlist
                ? String(cfg.allowlist).length
                : 0,
              systemPromptBytes: cfg.system
                ? String(cfg.system).length
                : 0,
            },
          });
          id = state.id;
        } else {
          id = `s${Date.now()}`;
        }
        const session = new Session(self, id);
        self.#sessions.set(id, session);
        emit(self.onEvent, 'status', {
          sessionId: id,
          phase: 'ready',
          title,
          projectRoot: self.projectRoot,
        });
        return session;
      },
      /**
       * Resume a persisted session (json + jsonl). Does not rewrite history —
       * Session.run() reloads visible context_window events into AGL retain_history.
       * @param {string} id
       */
      open: async (id) => {
        if (!id) throw new Error('session.open requires id');
        const existing = self.#sessions.get(id);
        if (existing) return existing;
        if (!self.sessionStore) {
          throw new Error('session.open requires sessionStore');
        }
        const state = self.sessionStore.load(id);
        if (!state) {
          throw new Error(`session not found: ${id}`);
        }
        self.sessionStore.setStatus(id, 'ready');
        self.sessionStore.appendEvent(id, {
          event_type: 'session',
          payload: {
            kind: 'resumed',
            agent: state.agent,
            model: state.model,
          },
        });
        const session = new Session(self, id);
        self.#sessions.set(id, session);
        emit(self.onEvent, 'status', {
          sessionId: id,
          phase: 'ready',
          resumed: true,
          title: state.title,
          projectRoot: self.projectRoot,
        });
        return session;
      },
      get: (id) => self.#sessions.get(id),
      list: () => [...self.#sessions.keys()],
      /** Disk session ids under .angela/sessions/ */
      listPersisted: () => self.sessionStore?.list() || [],
    };
  }

  /** List tools currently available from connected MCP servers. */
  async listMcpTools() {
    const out = [];
    for (const client of this.clients) {
      const tools = await client.listTools();
      for (const t of tools) {
        out.push({
          server: client.name,
          name: toolName(client.name, t.name),
          mcp: t.name,
          description: t.description,
        });
      }
    }
    return out;
  }

  async close() {
    await Promise.all(this.clients.map((c) => c.close()));
    this.clients = [];
    try {
      this.#contextHost?.close();
    } catch {
      /* ignore */
    }
    this.#contextHost = null;
  }
}

export {
  mcpResultToText,
  mcpResultParts,
  mcpResultToProviderContent,
  mcpResultImageMeta,
  modelSupportsImages,
  toolName,
} from './mcp-client.mjs';

export {
  PolicyEngine,
  scriptedApprover,
  autoApprover,
  denyApprover,
  stdinApprover,
  DEFAULT_LUCY_ALLOWLIST,
  isAllowed,
} from './policy.mjs';

export { ApprovalQueue } from './approval-queue.mjs';
export {
  loadAgent,
  listAgents,
  resolveAgentPath,
  normalizeToolsEnabled,
  filterToolNames,
  normalizeStarters,
  defaultAgentFields,
  materializeAgentDef,
  setDefaultAgent,
  getDefaultAgentFactory,
  invokeDefaultAgent,
  mergeAgentDefs,
} from './agent-loader.mjs';
export {
  buildMcpPresets,
  resolveMcpList,
  defaultMcpRoot,
  resolveMcpServerScript,
} from './mcp-presets.mjs';
export {
  resolveProjectRoot,
  ensureAngelaLayout,
  agentsDir,
  sessionsDir,
} from './project.mjs';
export {
  SessionStore,
  extractUsage,
  extractFinishReason,
  foldEventMetas,
  eventToProviderMessage,
  logSnapshotsEnabled,
} from './session-store.mjs';
export {
  resolveContextWindow,
  resolveContextWindowAsync,
  parseModelSpec,
  resolveRemoteTarget,
  resolveRshTarget,
  registerRemoteModelHosts,
} from './model-spec.mjs';
export { loadPlugins, getPlugins } from './plugins.mjs';
export {
  startContextHost,
  isContextLoopNavToolName,
} from './context-host.mjs';

export default Angela;
