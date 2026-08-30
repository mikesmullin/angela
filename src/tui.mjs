/**
 * angela OpenTUI chat shell.
 *
 *   -t / --tui
 *   · with prompt: show live transcript + start agentic loop immediately
 *   · without: welcome / first-message screen (browser-like)
 *   Approvals use ApprovalQueue; press y / n / a while a request is pending.
 */
import {
  createCliRenderer,
  BoxRenderable,
  TextRenderable,
  ScrollBoxRenderable,
  InputRenderable,
  InputRenderableEvents,
} from '@opentui/core';
import {
  Angela,
  ApprovalQueue,
  DEFAULT_LUCY_ALLOWLIST,
  autoApprover,
} from './index.mjs';

function summarizeResult(result) {
  if (result == null) return '';
  if (typeof result === 'object' && result.command) return '';
  if (typeof result === 'string') return result;
  if (typeof result === 'object') {
    if (result.summary) return String(result.summary);
    if (result.output !== undefined) return String(result.output);
    const content = result.choices?.[0]?.message?.content;
    if (content) return String(content);
    try {
      return JSON.stringify(result);
    } catch {
      return String(result);
    }
  }
  return String(result);
}

function preview(s, n = 220) {
  const t = String(s || '').replace(/\n/g, ' ');
  return t.length > n ? `${t.slice(0, n)}…` : t;
}

/**
 * @param {{
 *   model: string,
 *   system: string,
 *   mcp: object[],
 *   allowlist?: string,
 *   initialPrompt?: string,
 *   approveMode?: null | true | 'auto' | 'deny' | 'tui' | 'stdin',
 *   agentName?: string | null,
 *   projectRoot?: string,
 * }} config
 */
export async function runTui(config) {
  const {
    model,
    system,
    mcp = [],
    allowlist = DEFAULT_LUCY_ALLOWLIST,
    initialPrompt = '',
    approveMode = true,
    agentName = null,
    projectRoot = process.cwd(),
  } = config;

  const queue = new ApprovalQueue();
  let onApproval = null;
  if (approveMode === 'auto') onApproval = autoApprover();
  else if (approveMode === 'deny') onApproval = async () => 'deny';
  else if (
    approveMode === true ||
    approveMode === 'tui' ||
    approveMode === 'stdin'
  ) {
    onApproval = queue.createApprover();
  }

  let destroyed = false;
  let running = false;
  let harness = null;
  let session = null;
  let lineId = 0;
  /** @type {Map<string, TextRenderable>} */
  const approvalLines = new Map();

  const renderer = await createCliRenderer({
    exitOnCtrlC: true,
    clearOnShutdown: true,
    backgroundColor: '#0f1419',
    useMouse: true,
    autoFocus: true,
    onDestroy: () => {
      destroyed = true;
    },
  });

  const root = new BoxRenderable(renderer, {
    id: 'root',
    flexGrow: 1,
    flexDirection: 'column',
    backgroundColor: '#0f1419',
  });
  renderer.root.add(root);

  const headerTitle = agentName ? ` angela · ${agentName} ` : ' angela ';
  const header = new BoxRenderable(renderer, {
    id: 'header',
    height: 3,
    border: true,
    borderColor: '#2d3a4f',
    title: headerTitle,
    titleColor: '#93c5fd',
    backgroundColor: '#1a2332',
    flexDirection: 'column',
    paddingLeft: 1,
  });
  root.add(header);

  const statusText = new TextRenderable(renderer, {
    id: 'status',
    content: `${agentName ? agentName + ' · ' : ''}model ${model}  ·  starting…`,
    fg: '#8b9bb4',
  });
  header.add(statusText);

  const scroll = new ScrollBoxRenderable(renderer, {
    id: 'scroll',
    flexGrow: 1,
    border: true,
    borderColor: '#2d3a4f',
    stickyScroll: true,
    stickyStart: 'bottom',
    scrollX: false,
    scrollY: true,
    rootOptions: { backgroundColor: '#0f1419' },
    contentOptions: {
      backgroundColor: '#0f1419',
      flexDirection: 'column',
      gap: 0,
      paddingLeft: 1,
      paddingRight: 1,
      paddingTop: 1,
    },
  });
  root.add(scroll);

  const inputBox = new BoxRenderable(renderer, {
    id: 'inputBox',
    height: 3,
    border: true,
    borderColor: '#3b82f6',
    title: ' message · Enter send · Ctrl+C quit ',
    titleColor: '#93c5fd',
    backgroundColor: '#1a2332',
    paddingLeft: 1,
  });
  root.add(inputBox);

  const input = new InputRenderable(renderer, {
    id: 'input',
    width: '100%',
    placeholder: 'Type your first message…',
    backgroundColor: '#1a2332',
    textColor: '#e7ecf3',
    cursorColor: '#3b82f6',
    focusedBackgroundColor: '#1a2332',
    focusedTextColor: '#e7ecf3',
  });
  inputBox.add(input);
  input.focus();

  function setStatus(s) {
    statusText.content = s;
  }

  function addLine(text, fg = '#e7ecf3') {
    const id = `ln${lineId++}`;
    const t = new TextRenderable(renderer, {
      id,
      content: text,
      fg,
      wrapMode: 'word',
      width: '100%',
    });
    scroll.content.add(t);
    try {
      // sticky bottom
      scroll.scrollTop = Math.max(0, scroll.scrollHeight);
    } catch {
      /* ignore */
    }
    return t;
  }

  function showWelcome() {
    addLine(agentName ? `angela · ${agentName}` : 'Cowork / angela', '#e7ecf3');
    addLine('', '#64748b');
    addLine('Modular agent shell: TUI → angela harness → MCP tools.', '#8b9bb4');
    addLine(
      mcp.length
        ? `MCP: ${mcp.map((m) => m.name).join(', ')}`
        : 'No MCP servers (chat-only).',
      '#8b9bb4',
    );
    addLine(`Model: ${model}`, '#8b9bb4');
    addLine('', '#64748b');
    addLine('What would you like to work on?', '#93c5fd');
    addLine('Type a message below and press Enter to start a session.', '#8b9bb4');
    addLine('', '#64748b');
    addLine('Tips', '#94a3b8');
    addLine('  · tool approval → y allow · n deny · a always', '#64748b');
    if (agentName === 'lucy') {
      addLine('  · Agents from .angela/agents/ · MCP presets (local + optional plugins)', '#64748b');
      addLine('  · AGENTS.md loaded from work laptop over RSH', '#64748b');
    }
  }

  async function ensureSession() {
    if (harness) return;
    setStatus(`model ${model}  ·  connecting…`);
    harness = await Angela.create({
      projectRoot,
      agentName,
      agentFile: config.agentFile || null,
      model,
      system,
      mcp,
      allowlist,
      allowlistSource: config.allowlistSource || null,
      onApproval,
      policyMode: config.policyMode || 'ask',
      approveMode,
      overrides: config.overrides || null,
      onEvent: (ev) => {
        if (destroyed) return;
        handleEvent(ev);
      },
    });
    session = await harness.session.create({
      title: agentName || 'tui',
      agent: agentName,
    });
    addLine(`session ${session.id}`, '#64748b');
    const tools = await harness.listMcpTools();
    setStatus(
      `model ${model}  ·  ready  ·  ${tools.length} tools  ·  ${
        mcp.map((m) => m.name).join(',') || 'no-mcp'
      }`,
    );
  }

  let liveToolLine = null;

  function handleEvent(ev) {
    if (ev.type === 'tool_call') {
      liveToolLine = null;
      addLine(
        `→ ${ev.name} ${preview(JSON.stringify(ev.args || {}))}`,
        '#64748b',
      );
    } else if (ev.type === 'tool_delta') {
      const tail = preview(ev.output || ev.text || '', 80);
      const line = `… ${ev.name || ev.progressToken || 'tool'} ${tail}`;
      if (liveToolLine) {
        liveToolLine.content = line;
      } else {
        liveToolLine = addLine(line, '#64748b');
      }
    } else if (ev.type === 'tool_result') {
      liveToolLine = null;
      const tag = ev.denied ? 'DENIED' : ev.ok === false ? 'ERR' : 'ok';
      const color =
        ev.denied || ev.ok === false ? '#fca5a5' : '#86efac';
      addLine(`← ${ev.name} [${tag}] ${preview(ev.text)}`, color);
    } else if (ev.type === 'approval_needed') {
      const line = addLine(
        `⚠ APPROVAL ${ev.id}  ${ev.tool}  ${preview(ev.command || JSON.stringify(ev.args), 70)}   [y]allow [n]deny [a]always`,
        '#fbbf24',
      );
      approvalLines.set(ev.id, line);
      setStatus(`model ${model}  ·  waiting approval · ${ev.id}`);
    } else if (ev.type === 'error') {
      addLine(`error: ${ev.message}`, '#f87171');
    } else if (ev.type === 'aborted') {
      addLine('(aborted)', '#f87171');
    }
  }

  // Global keys for approval (only when queue has items)
  renderer.keyInput.on('keypress', (key) => {
    const pending = queue.list();
    if (!pending.length) return;
    const ch = String(key?.name || key?.sequence || '').toLowerCase();
    if (ch !== 'y' && ch !== 'n' && ch !== 'a') return;
    // Don't steal keys if user is mid-type with non-empty input and not y/n/a alone
    // Always allow y/n/a to resolve oldest pending approval.
    const req = pending[0];
    if (ch === 'y') {
      queue.resolve(req.id, 'allow');
      const ln = approvalLines.get(req.id);
      if (ln) {
        ln.content = `✓ allowed  ${req.tool}`;
        ln.fg = '#86efac';
      }
      setStatus(`model ${model}  ·  running…`);
    } else if (ch === 'n') {
      queue.resolve(req.id, 'deny');
      const ln = approvalLines.get(req.id);
      if (ln) {
        ln.content = `✗ denied  ${req.tool}`;
        ln.fg = '#fca5a5';
      }
    } else if (ch === 'a') {
      queue.resolve(req.id, 'always');
      const ln = approvalLines.get(req.id);
      if (ln) {
        ln.content = `✓ always  ${req.tool}`;
        ln.fg = '#86efac';
      }
      setStatus(`model ${model}  ·  running…`);
    }
  });

  async function runPrompt(prompt) {
    const p = String(prompt || '').trim();
    if (!p || running || destroyed) return;
    running = true;
    input.placeholder = 'Agent working…  approvals: y / n / a';
    try {
      await ensureSession();
      addLine(`You: ${p}`, '#93c5fd');
      setStatus(`model ${model}  ·  running…`);
      const result = await session.run({ prompt: p });
      if (result?.command === 'new' && result.session) {
        session = result.session;
        addLine('(new session)', '#86efac');
        setStatus(`model ${model}  ·  ready  ·  new session`);
        return;
      }
      const text = summarizeResult(result);
      addLine(`Assistant: ${text || '(empty)'}`, '#e7ecf3');
      setStatus(
        `model ${model}  ·  ready  ·  ${
          mcp.map((m) => m.name).join(',') || 'no-mcp'
        }`,
      );
    } catch (err) {
      addLine(`Error: ${err?.message || err}`, '#f87171');
      setStatus(`model ${model}  ·  error`);
    } finally {
      running = false;
      input.placeholder = 'Message…';
      try {
        input.focus();
      } catch {
        /* ignore */
      }
    }
  }

  input.on(InputRenderableEvents.ENTER, () => {
    const v = String(input.value || '').trim();
    if (!v) return;
    input.value = '';
    void runPrompt(v);
  });

  // Initial view
  if (initialPrompt && String(initialPrompt).trim()) {
    addLine(`session · ${model}`, '#64748b');
    void runPrompt(String(initialPrompt).trim());
  } else {
    showWelcome();
    setStatus(
      `model ${model}  ·  ready  ·  ${
        mcp.map((m) => m.name).join(',') || 'no-mcp'
      }`,
    );
  }

  // Stay alive until renderer is destroyed (Ctrl+C with exitOnCtrlC)
  await new Promise((resolve) => {
    const finish = async () => {
      if (destroyed) {
        resolve();
        return;
      }
      destroyed = true;
      queue.clear('deny');
      try {
        await harness?.close();
      } catch {
        /* ignore */
      }
      resolve();
    };
    renderer.on('destroy', () => {
      void finish();
    });
  });
}

export default { runTui };
