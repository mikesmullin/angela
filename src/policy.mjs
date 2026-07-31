/**
 * Central tool policy for angela — allowlist + human approval.
 *
 * Used by: integration tests (headless), browser, CLI/TUI.
 * UIs only implement presentation; decisions flow through PolicyEngine.
 */

import { isAllowed, DEFAULT_LUCY_ALLOWLIST } from './allowlist.mjs';

/**
 * @typedef {'allow'|'deny'|'always'} ApprovalDecision
 * @typedef {{
 *   id: string,
 *   tool: string,
 *   server?: string,
 *   mcpTool?: string,
 *   args: object,
 *   command?: string,
 *   reason: string,
 * }} ApprovalRequest
 */

export class PolicyEngine {
  /**
   * @param {{
   *   allowlist?: string,
   *   onApproval?: (req: ApprovalRequest) => Promise<ApprovalDecision>|ApprovalDecision,
   *   /** 'ask' = require approval when not allowlisted; 'strict' = deny if no onApproval; 'open' = allow all (dev only) *\/
   *   mode?: 'ask'|'strict'|'open',
   *   onEvent?: (ev: object) => void,
   * }} opts
   */
  constructor(opts = {}) {
    this.allowlist = opts.allowlist ?? DEFAULT_LUCY_ALLOWLIST;
    this.onApproval = opts.onApproval ?? null;
    this.mode = opts.mode || 'ask';
    this.onEvent = opts.onEvent || null;
    /** @type {Set<string>} session pins from "always" */
    this.sessionAlways = new Set();
    this._id = 1;
  }

  setAllowlist(text) {
    this.allowlist = text || '';
  }

  /**
   * Build a stable command line string for terminal tools (allowlist params).
   */
  static commandFromArgs(toolName, args = {}) {
    if (args.command != null && Array.isArray(args.args)) {
      return [String(args.command), ...args.args.map(String)].join(' ');
    }
    if (typeof args.script === 'string') return args.script.trim();
    if (typeof args.command === 'string' && !args.args) return args.command.trim();
    try {
      return JSON.stringify(args);
    } catch {
      return String(args);
    }
  }

  /**
   * @returns {Promise<{ allowed: boolean, via: string, request?: ApprovalRequest }>}
   */
  async authorize({ tool, server, mcpTool, args }) {
    const command = PolicyEngine.commandFromArgs(tool, args || {});
    const pinKey = `${tool}\0${command}`;

    if (this.mode === 'open') {
      this._emit('policy', { tool, decision: 'allow', via: 'open_mode' });
      return { allowed: true, via: 'open_mode' };
    }

    if (this.sessionAlways.has(pinKey) || this.sessionAlways.has(tool)) {
      this._emit('policy', { tool, decision: 'allow', via: 'session_always' });
      return { allowed: true, via: 'session_always' };
    }

    if (isAllowed(this.allowlist, tool, command)) {
      this._emit('policy', { tool, decision: 'allow', via: 'allowlist', command });
      return { allowed: true, via: 'allowlist' };
    }

    const request = {
      id: `apr${this._id++}`,
      tool,
      server,
      mcpTool,
      args: args || {},
      command,
      reason: 'not on allowlist',
    };

    this._emit('approval_needed', { ...request });

    if (!this.onApproval) {
      if (this.mode === 'strict' || this.mode === 'ask') {
        this._emit('policy', { tool, decision: 'deny', via: 'no_approver' });
        return { allowed: false, via: 'no_approver', request };
      }
    }

    let decision;
    try {
      decision = await this.onApproval(request);
    } catch (err) {
      this._emit('policy', {
        tool,
        decision: 'deny',
        via: 'approver_error',
        error: err?.message,
      });
      return { allowed: false, via: 'approver_error', request };
    }

    if (decision === 'always') {
      this.sessionAlways.add(pinKey);
      this._emit('policy', { tool, decision: 'allow', via: 'approved_always' });
      return { allowed: true, via: 'approved_always', request };
    }
    if (decision === 'allow') {
      this._emit('policy', { tool, decision: 'allow', via: 'approved' });
      return { allowed: true, via: 'approved', request };
    }

    this._emit('policy', { tool, decision: 'deny', via: 'denied' });
    return { allowed: false, via: 'denied', request };
  }

  _emit(type, data) {
    try {
      this.onEvent?.({ type, ts: Date.now(), ...data });
    } catch {
      /* ignore */
    }
  }
}

/** Headless approver: script of decisions by tool name or id order. */
export function scriptedApprover(script = []) {
  let i = 0;
  return async (req) => {
    if (i >= script.length) return 'deny';
    const step = script[i++];
    if (typeof step === 'string') return step;
    if (step.tool && step.tool !== req.tool) {
      // optional filter — skip until match or use decision anyway
    }
    return step.decision || step.d || 'deny';
  };
}

/** Always allow (tests that still exercise event emission when forced). */
export function autoApprover() {
  return async () => 'allow';
}

/** Always deny. */
export function denyApprover() {
  return async () => 'deny';
}

/**
 * Interactive CLI approver (stdin). Used until opentui TUI lands.
 * Prints a compact prompt; y/n/a = allow / deny / always.
 */
export function stdinApprover({ input = process.stdin, output = process.stderr } = {}) {
  return async (req) => {
    const readline = await import('node:readline');
    const rl = readline.createInterface({ input, output });
    const q = () =>
      new Promise((resolve) => {
        output.write(
          `\n[angela approval #${req.id}] tool=${req.tool}\n` +
            `  command: ${req.command || '(args)'}\n` +
            `  args: ${JSON.stringify(req.args)}\n` +
            `  [y]allow  [n]deny  [a]always this session > `,
        );
        rl.question('', (answer) => {
          const a = String(answer || '').trim().toLowerCase();
          if (a === 'y' || a === 'yes') resolve('allow');
          else if (a === 'a' || a === 'always') resolve('always');
          else resolve('deny');
        });
      });
    try {
      return await q();
    } finally {
      rl.close();
    }
  };
}

export { DEFAULT_LUCY_ALLOWLIST, isAllowed };
