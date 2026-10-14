/**
 * Minimal MCP HTTP (streamable) client: JSON-RPC over POST, initialize,
 * list/call tools. Mirrors McpStdioClient's interface so the harness treats
 * both transports identically (psicom: cua-driver spacesd /mcp + execd exec
 * facade are HTTP MCP servers).
 *
 * Transport notes (MCP streamable HTTP):
 *   - POST JSON-RPC with `Accept: application/json, text/event-stream`.
 *   - The server replies `application/json` (single response), an SSE stream
 *     carrying the response, or `202 Accepted` (notifications).
 *   - The server MAY assign `Mcp-Session-Id` on initialize; echo it after.
 *   - abort() aborts the in-flight fetches (real cancellation) AND rejects
 *     the pending promises, so the agent turn never wedges on a stuck server.
 */
import { mcpAbortError } from './mcp-client.mjs';

const DEFAULT_TIMEOUT_MS = 120_000;

/** Expand `${VAR}` references in header values from process.env. */
function interpolateHeaders(headers = {}) {
  const out = {};
  for (const [k, v] of Object.entries(headers || {})) {
    out[k] = String(v ?? '').replace(/\$\{([A-Z0-9_]+)\}/g, (m, name) =>
      process.env[name] != null ? process.env[name] : m,
    );
  }
  return out;
}

export class McpHttpClient {
  #name;
  #url;
  #headers;
  #timeoutMs;
  #sessionId = null;
  #nextId = 1;
  /** id → { resolve, reject, method, ac } */
  #pending = new Map();

  constructor({ name, url, headers = {}, prefix, timeoutMs, onNotification } = {}) {
    this.#name = name || 'mcp';
    this.#url = String(url || '');
    this.#headers = headers || {};
    this.onNotification = onNotification || null;
    /** When false, register tools under the MCP name as-is (no server__ prefix). */
    this.prefix = prefix !== false;
    this.#timeoutMs =
      Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : DEFAULT_TIMEOUT_MS;
  }

  get name() {
    return this.#name;
  }

  #baseHeaders() {
    const h = { ...interpolateHeaders(this.#headers) };
    if (this.#sessionId) h['Mcp-Session-Id'] = this.#sessionId;
    return h;
  }

  async #post(body, { signal } = {}) {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(new Error(`MCP ${this.#name} request timed out`)), this.#timeoutMs);
    const onOuterAbort = () => ac.abort(signal?.reason ?? new Error('aborted'));
    if (signal) {
      if (signal.aborted) onOuterAbort();
      else signal.addEventListener('abort', onOuterAbort, { once: true });
    }
    try {
      return await fetch(this.#url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json, text/event-stream',
          ...this.#baseHeaders(),
        },
        body: JSON.stringify(body),
        signal: ac.signal,
      });
    } finally {
      clearTimeout(timer);
      if (signal) signal.removeEventListener('abort', onOuterAbort);
    }
  }

  /** Parse a POST response body: JSON object, SSE stream (data: lines), or empty. */
  async #parseResponse(response, awaitId = null) {
    const ct = String(response.headers.get('content-type') || '');
    const text = await response.text();
    if (!text.trim()) return null;
    if (ct.includes('text/event-stream')) {
      let result = null;
      for (const line of text.split('\n')) {
        const s = line.trim();
        if (!s.startsWith('data:')) continue;
        const data = s.slice(5).trim();
        if (!data || data === '[DONE]') continue;
        let msg;
        try {
          msg = JSON.parse(data);
        } catch {
          continue;
        }
        if (msg && typeof msg === 'object') {
          if (msg.id === undefined && msg.method && this.onNotification) {
            try {
              this.onNotification(msg);
            } catch {
              /* ignore */
            }
          }
          if (awaitId == null || msg.id === awaitId) result = msg;
        }
      }
      return result;
    }
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  }

  async connect() {
    const id = this.#nextId++;
    const response = await this.#post({
      jsonrpc: '2.0',
      id,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: { name: 'angela', version: '0.0.1' },
      },
    });
    if (!response.ok) {
      throw new Error(
        `MCP ${this.#name} initialize failed: ${response.status} ${response.statusText}`,
      );
    }
    const sid = response.headers.get('mcp-session-id');
    if (sid) this.#sessionId = sid;
    const msg = await this.#parseResponse(response, id);
    if (msg?.error) {
      throw new Error(msg.error.message || JSON.stringify(msg.error));
    }
    // notifications/initialized — 202/empty expected; never fatal.
    try {
      await this.#post({ jsonrpc: '2.0', method: 'notifications/initialized', params: {} });
    } catch {
      /* ignore */
    }
  }

  async listTools() {
    const id = this.#nextId++;
    const response = await this.#post({ jsonrpc: '2.0', id, method: 'tools/list', params: {} });
    if (!response.ok) {
      throw new Error(`MCP ${this.#name} tools/list failed: ${response.status} ${response.statusText}`);
    }
    const msg = await this.#parseResponse(response, id);
    if (msg?.error) throw new Error(msg.error.message || JSON.stringify(msg.error));
    return msg?.result?.tools || [];
  }

  async callTool(name, args = {}, extra = {}) {
    const id = this.#nextId++;
    const params = { name, arguments: args };
    if (extra.progressToken != null && extra.progressToken !== '') {
      params._meta = { ...(extra._meta || {}), progressToken: extra.progressToken };
    } else if (extra._meta) {
      params._meta = extra._meta;
    }
    const pending = {
      method: 'tools/call',
      resolve: null,
      reject: null,
      ac: new AbortController(),
    };
    const promise = new Promise((resolve, reject) => {
      pending.resolve = resolve;
      pending.reject = reject;
    });
    this.#pending.set(id, pending);
    try {
      const response = await this.#post(
        { jsonrpc: '2.0', id, method: 'tools/call', params },
        { signal: pending.ac.signal },
      );
      if (!response.ok) {
        throw new Error(`MCP ${this.#name} tools/call failed: ${response.status} ${response.statusText}`);
      }
      const msg = await this.#parseResponse(response, id);
      if (msg?.error) throw new Error(msg.error.message || JSON.stringify(msg.error));
      return msg?.result ?? null;
    } finally {
      this.#pending.delete(id);
    }
  }

  /**
   * Cancel in-flight `tools/call` requests (Stop / new prompt). Aborts the
   * real HTTP fetches AND rejects the local pending promises; best-effort
   * `notifications/cancelled` POST so the server can kill children too.
   */
  abort(reason = 'aborted') {
    const err = mcpAbortError(reason);
    for (const [id, p] of [...this.#pending]) {
      if (p.method && p.method !== 'tools/call') continue;
      this.#pending.delete(id);
      try {
        p.ac.abort(err);
      } catch {
        /* ignore */
      }
      try {
        p.reject(err);
      } catch {
        /* ignore */
      }
      try {
        this.#post({
          jsonrpc: '2.0',
          method: 'notifications/cancelled',
          params: { requestId: id, reason: reason || 'aborted' },
        }).catch(() => {});
      } catch {
        /* ignore */
      }
    }
  }

  /** Stateless transport — nothing to tear down. */
  async close() {
    this.abort('closed');
  }
}
