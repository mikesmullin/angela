/**
 * Unix-socket HTTP host so optional MCP servers (mcp-context-loop) can call
 * Session context-window methods without those tools being built into angela.
 *
 * Bun.serve({ unix }) + fetch({ unix }). Path is passed to MCP children as
 * ANGELA_HOST_SOCK.
 */
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { existsSync, unlinkSync } from 'node:fs';
import { randomBytes } from 'node:crypto';

/** Bare or prefixed MCP tool names that are CONTEXT LOOP navigation. */
export function isContextLoopNavToolName(name) {
  const base = String(name || '').split('__').pop();
  return (
    base === 'goto' || base === 'loop_start_marker' || base === 'waypoint'
  );
}

/**
 * @param {object} harness Angela instance (needs activeSession())
 * @param {{ socketPath?: string }} [opts]
 * @returns {Promise<{ path: string, close: () => void }>}
 */
export async function startContextHost(harness, opts = {}) {
  const path =
    opts.socketPath ||
    join(
      tmpdir(),
      `angela-host-${process.pid}-${randomBytes(4).toString('hex')}.sock`,
    );
  if (existsSync(path)) {
    try {
      unlinkSync(path);
    } catch {
      /* ignore */
    }
  }

  function session() {
    if (typeof harness?.activeSession === 'function') {
      return harness.activeSession();
    }
    return null;
  }

  const server = Bun.serve({
    unix: path,
    async fetch(req) {
      const url = new URL(req.url);
      const method = url.pathname.replace(/^\//, '') || 'health';
      if (req.method === 'GET' && (method === 'health' || method === '')) {
        return new Response('ok');
      }
      if (req.method !== 'POST') {
        return new Response('method not allowed', { status: 405 });
      }
      let args = {};
      const ct = req.headers.get('content-type') || '';
      if (ct.includes('json')) {
        try {
          args = await req.json();
        } catch {
          args = {};
        }
      } else {
        const raw = await req.text();
        if (raw) {
          try {
            args = JSON.parse(raw);
          } catch {
            args = {};
          }
        }
      }
      const sess = session();
      if (!sess) {
        return new Response('angela: no active session', { status: 409 });
      }
      try {
        if (method === 'goto') {
          if (typeof sess.applyGoto !== 'function') {
            return new Response('angela: applyGoto not available', { status: 501 });
          }
          return new Response(String(sess.applyGoto(args) ?? ''), { status: 200 });
        }
        if (method === 'loop_start_marker' || method === 'waypoint') {
          if (typeof sess.applyLoopStartMarker !== 'function') {
            return new Response('angela: applyLoopStartMarker not available', {
              status: 501,
            });
          }
          return new Response(String(sess.applyLoopStartMarker(args) ?? ''), {
            status: 200,
          });
        }
        return new Response(`unknown method: ${method}`, { status: 404 });
      } catch (err) {
        return new Response(err?.message || String(err), { status: 500 });
      }
    },
  });

  return {
    path,
    close() {
      try {
        server.stop(true);
      } catch {
        /* ignore */
      }
      try {
        if (existsSync(path)) unlinkSync(path);
      } catch {
        /* ignore */
      }
    },
  };
}

export default { startContextHost, isContextLoopNavToolName };
