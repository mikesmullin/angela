/**
 * Minimal MCP stdio client: spawn server, initialize, list/call tools.
 */
import { spawn } from 'node:child_process';

export class McpStdioClient {
  #proc = null;
  #buffer = '';
  #nextId = 1;
  #pending = new Map();
  #name;
  #onStderr;

  constructor({ name, command, args = [], env = {}, cwd, onStderr, onNotification, prefix, framing } = {}) {
    this.#name = name || 'mcp';
    this.#onStderr = onStderr;
    this.onNotification = onNotification || null;
    this.command = command;
    this.args = args;
    this.env = env;
    this.cwd = cwd;
    /** When false, register tools under the MCP name as-is (no server__ prefix). */
    this.prefix = prefix !== false;
    /** `newline` = ada/file-io kit; `lsp` = official MCP SDK Content-Length. */
    this.framing = framing === 'lsp' ? 'lsp' : 'newline';
  }

  get name() {
    return this.#name;
  }

  async connect() {
    if (this.#proc) return;

    this.#proc = spawn(this.command, this.args, {
      cwd: this.cwd,
      env: { ...process.env, ...this.env },
      stdio: ['pipe', 'pipe', 'pipe'],
      shell: false,
    });

    this.#proc.stdout.setEncoding('utf8');
    this.#proc.stdout.on('data', (chunk) => this.#onData(chunk));
    this.#proc.stderr.setEncoding('utf8');
    this.#proc.stderr.on('data', (chunk) => {
      if (this.#onStderr) this.#onStderr(chunk);
      else process.stderr.write(`[mcp:${this.#name}] ${chunk}`);
    });
    this.#proc.on('exit', (code) => {
      for (const [, p] of this.#pending) {
        p.reject(new Error(`MCP server ${this.#name} exited (${code})`));
      }
      this.#pending.clear();
      this.#proc = null;
    });

    const init = this.#request('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'angela', version: '0.0.1' },
    });
    const timeout = new Promise((_, reject) => {
      setTimeout(
        () => reject(new Error(`MCP ${this.#name} initialize timed out`)),
        15000,
      );
    });
    await Promise.race([init, timeout]);
    this.#notify('notifications/initialized', {});
  }

  async listTools() {
    const result = await this.#request('tools/list', {});
    return result?.tools || [];
  }

  async callTool(name, args = {}, extra = {}) {
    const params = { name, arguments: args };
    if (extra.progressToken != null && extra.progressToken !== '') {
      params._meta = { ...(extra._meta || {}), progressToken: extra.progressToken };
    } else if (extra._meta) {
      params._meta = extra._meta;
    }
    return this.#request('tools/call', params);
  }

  async close() {
    if (!this.#proc) return;
    try {
      this.#proc.stdin.end();
    } catch { /* ignore */ }
    const proc = this.#proc;
    await new Promise((resolve) => {
      const t = setTimeout(() => {
        try { proc.kill('SIGKILL'); } catch { /* ignore */ }
        resolve();
      }, 2000);
      proc.on('exit', () => {
        clearTimeout(t);
        resolve();
      });
    });
    this.#proc = null;
  }

  #onData(chunk) {
    this.#buffer += chunk;
    if (this.framing === 'lsp') this.#drainLsp();
    else this.#drainNewlines();
  }

  #handleMsg(msg) {
    if (!msg || typeof msg !== 'object') return;
    if (msg.id !== undefined && this.#pending.has(msg.id)) {
      const p = this.#pending.get(msg.id);
      this.#pending.delete(msg.id);
      if (msg.error) {
        p.reject(new Error(msg.error.message || JSON.stringify(msg.error)));
      } else {
        p.resolve(msg.result);
      }
      return;
    }
    if (msg.method && msg.id === undefined) {
      try {
        this.onNotification?.(msg);
      } catch {
        /* ignore */
      }
    }
  }

  #drainNewlines() {
    let idx;
    while ((idx = this.#buffer.indexOf('\n')) !== -1) {
      const line = this.#buffer.slice(0, idx).replace(/\r$/, '');
      this.#buffer = this.#buffer.slice(idx + 1);
      if (!line.trim()) continue;
      try {
        this.#handleMsg(JSON.parse(line));
      } catch {
        /* ignore partial/junk */
      }
    }
  }

  #drainLsp() {
    while (true) {
      const sep = this.#buffer.indexOf('\r\n\r\n');
      if (sep < 0) return;
      const header = this.#buffer.slice(0, sep);
      const m = /Content-Length:\s*(\d+)/i.exec(header);
      if (!m) {
        this.#buffer = this.#buffer.slice(sep + 4);
        continue;
      }
      const len = Number(m[1]);
      const start = sep + 4;
      if (this.#buffer.length < start + len) return;
      const body = this.#buffer.slice(start, start + len);
      this.#buffer = this.#buffer.slice(start + len);
      try {
        this.#handleMsg(JSON.parse(body));
      } catch {
        /* ignore */
      }
    }
  }

  #writeMsg(obj) {
    if (!this.#proc) return;
    const json = JSON.stringify(obj);
    if (this.framing === 'lsp') {
      const body = Buffer.from(json, 'utf8');
      this.#proc.stdin.write(`Content-Length: ${body.length}\r\n\r\n`);
      this.#proc.stdin.write(body);
    } else {
      this.#proc.stdin.write(json + '\n');
    }
  }

  #request(method, params) {
    if (!this.#proc) return Promise.reject(new Error('MCP not connected'));
    const id = this.#nextId++;
    const payload = { jsonrpc: '2.0', id, method, params };
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#writeMsg(payload);
    });
  }

  #notify(method, params) {
    this.#writeMsg({ jsonrpc: '2.0', method, params });
  }
}

/** Raw MCP content parts (text / image / …). */
export function mcpResultParts(result) {
  if (!result) return [];
  if (typeof result === 'string') return [{ type: 'text', text: result }];
  if (Array.isArray(result)) return result;
  if (Array.isArray(result.content)) return result.content;
  return [];
}

function imageCaption(part) {
  const mime = part?.mimeType || 'image';
  return `Read image file [${mime}]`;
}

/** Map MCP tool result content to a plain string for the model / UI. */
export function mcpResultToText(result) {
  if (!result) return '';
  if (typeof result === 'string') return result;
  const parts = mcpResultParts(result);
  return parts
    .map((c) => {
      if (c?.type === 'text') return c.text;
      if (c?.type === 'image') return '';
      return JSON.stringify({ type: c?.type, mimeType: c?.mimeType });
    })
    .filter((s) => s != null && s !== '')
    .join('\n');
}

/**
 * Convert MCP content parts to AGL/OpenAI tool-message content.
 * Image parts become `{ type: 'image_url', image_url: { url: data:... } }`.
 * Returns a string when there are no images (existing tool-result shape).
 *
 * @param {object} result
 * @param {{ supportsImages?: boolean }} [opts]
 * @returns {string | Array<{ type: string, text?: string, image_url?: { url: string } }>}
 */
export function mcpResultToProviderContent(result, opts = {}) {
  const supportsImages = opts.supportsImages !== false;
  const parts = mcpResultParts(result);
  const hasImage = parts.some((p) => p?.type === 'image');
  if (!hasImage) return mcpResultToText(result);

  const out = [];
  let sawImage = false;
  for (const p of parts) {
    if (p?.type === 'text') {
      out.push({ type: 'text', text: String(p.text ?? '') });
      continue;
    }
    if (p?.type === 'image') {
      sawImage = true;
      if (!supportsImages) continue;
      const mime = p.mimeType || 'image/png';
      const data = p.data || '';
      out.push({
        type: 'image_url',
        image_url: { url: `data:${mime};base64,${data}` },
      });
    }
  }
  if (sawImage && !supportsImages) {
    out.push({
      type: 'text',
      text: '[Current model does not support images. The image will be omitted from this request.]',
    });
  }
  if (out.length === 1 && out[0].type === 'text') return out[0].text || '';
  if (!out.length) return mcpResultToText(result) || imageCaption(parts.find((p) => p?.type === 'image'));
  return out;
}

/** Compact image metadata for jsonl / UI (no base64 payload). */
export function mcpResultImageMeta(result) {
  const img = mcpResultParts(result).find((p) => p?.type === 'image');
  if (!img) return null;
  const data = img.data || '';
  let bytes = 0;
  try {
    bytes = Buffer.byteLength(data, 'base64');
  } catch {
    bytes = data.length;
  }
  return { mimeType: img.mimeType || 'image', bytes };
}

/**
 * Whether this model id is expected to accept image_url / image parts.
 * Unknown ids default true (attach; the provider may drop).
 *
 * Gemma 4 is natively multimodal (text + image in, all sizes including
 * 12B QAT and 26B-A4B). Do not require a "vision" / "vl" suffix — that
 * was a Gemma 2 vs PaliGemma split and wrongly omitted Gemma 4 images.
 */
export function modelSupportsImages(model) {
  const s = String(model || '').toLowerCase();
  if (!s) return true;

  if (
    /\b(vision|vl-|vl\b|gpt-4o|gpt-4\.1|gpt-5|claude|gemini|omni|paligemma)\b/.test(s)
  ) {
    return true;
  }

  // Gemma 4 family: E2B, E4B, 12B, 26B-A4B, 31B — image input on all.
  if (/gemma[-_]?4/.test(s)) return true;
  // Gemma 3n + Gemma 3 4B/12B/27B
  if (/gemma[-_]?3n/.test(s)) return true;
  if (/gemma[-_]?3-?(4|12|27)b/.test(s)) return true;

  // Known text-only
  if (/gemma[-_]?2/.test(s)) return false;
  if (/gemma[-_]?3-?(270m|1b)\b/.test(s)) return false;
  if (/gemma[-_]?1\b/.test(s)) return false;
  if (
    /\b(llama-?3|qwen2(?:\.5)?|mistral|deepseek|coder)\b/.test(s) &&
    !/vision|vl/.test(s)
  ) {
    return false;
  }
  return true;
}

/**
 * Sanitize tool name for multi-provider AGL/OpenAI rules.
 * Hyphens → underscores (some providers reject hyphens; see plan §4.6).
 */
export function toolName(serverName, mcpToolName) {
  const raw = `${serverName}__${mcpToolName}`;
  return raw
    .replace(/-/g, '_')
    .replace(/[^a-zA-Z0-9_]/g, '_')
    .replace(/^[^a-zA-Z]+/, 't_')
    .slice(0, 64);
}
