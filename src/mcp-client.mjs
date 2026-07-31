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

  constructor({ name, command, args = [], env = {}, cwd, onStderr } = {}) {
    this.#name = name || 'mcp';
    this.#onStderr = onStderr;
    this.command = command;
    this.args = args;
    this.env = env;
    this.cwd = cwd;
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

    await this.#request('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'angela', version: '0.0.1' },
    });
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
    let idx;
    while ((idx = this.#buffer.indexOf('\n')) !== -1) {
      const line = this.#buffer.slice(0, idx).replace(/\r$/, '');
      this.#buffer = this.#buffer.slice(idx + 1);
      if (!line.trim()) continue;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        continue;
      }
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
  }

  #request(method, params) {
    if (!this.#proc) return Promise.reject(new Error('MCP not connected'));
    const id = this.#nextId++;
    const payload = { jsonrpc: '2.0', id, method, params };
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#proc.stdin.write(JSON.stringify(payload) + '\n');
    });
  }

  #notify(method, params) {
    if (!this.#proc) return;
    this.#proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n');
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
