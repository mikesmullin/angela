# js-session fixture

Produced by the **real** JavaScript `angela/src/session-store.mjs`, so the
Rust port is tested against bytes the JS actually writes rather than against
an assumption about its format.

Regenerate (needs Bun and `/workspace/angela`):

```sh
rm -rf tests/fixtures/js-session && mkdir -p /tmp/xc
cat > /tmp/xc/write_js.mjs <<'JS'
import SessionStore from '/workspace/angela/src/session-store.mjs';
const store = new SessionStore(process.argv[2]);
const st = store.create({ agent: 'lucy', model: 'llama-server:default', title: 'xcompat' });
const a = store.appendEvent(st.id, { event_type: 'context_window', payload: { role: 'user', content: 'hello' } });
store.appendEvent(st.id, { event_type: 'tool_call', payload: { name: 'read', args: { f: 'x.py' } }, duration_ms: 12.5 });
store.appendEvent(st.id, { event_type: 'context_meta', payload: { event_id: a.event_id, visible: false, note: 'hidden' } });
store.recordTurnComplete(st.id, { duration_ms: 99 });
JS
bun run /tmp/xc/write_js.mjs "$PWD/tests/fixtures/js-session"
```
