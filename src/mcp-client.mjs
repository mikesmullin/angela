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

  constructor({ name, command, args = [], env = {}, cwd, onStderr, prefix, framing } = {}) {
    this.#name = name || 'mcp';
    this.#onStderr = onStderr;
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

  async callTool(name, args = {}) {
    return this.#request('tools/call', { name, arguments: args });
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

/** Map MCP tool result content to a plain string for the model. */
export function mcpResultToText(result) {
  if (!result) return '';
  if (typeof result === 'string') return result;
  const parts = result.content || [];
  return parts
    .map((c) => (c?.type === 'text' ? c.text : JSON.stringify(c)))
    .join('\n');
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
