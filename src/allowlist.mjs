/**
 * Deny-by-default tool allowlist — port of gl1 `src/agents/allowlist.zig`.
 *
 * Rule format (one per line): `toolPattern` or `toolPattern:paramPattern`
 * - Case-insensitive full-match basic regex (literals, `.`, `*+?`, `[]`, `{m,n}`, `\b`)
 * - `*` alone = match anything
 * - Terminal tools: param pattern matches the command line (program + args)
 * - Non-terminal: tool match alone is enough if no param pattern, or if param present still only tool is required (gl1 behavior for non-terminal)
 *
 * Empty / whitespace-only text ⇒ nothing allowed.
 */

function isWordChar(c) {
  return /[A-Za-z0-9_]/.test(c);
}

function isWordBoundary(text, ti) {
  const prev = ti > 0 ? text[ti - 1] : '';
  const cur = ti < text.length ? text[ti] : '';
  const prevW = prev !== '' && isWordChar(prev);
  const curW = cur !== '' && isWordChar(cur);
  return prevW !== curW;
}

function parseAtom(pat, pi) {
  if (pi >= pat.length) return null;
  if (pat[pi] === '\\' && pi + 1 < pat.length) {
    const n = pat[pi + 1];
    if (n === 'b') return null; // handled at matchHere
    return { atom: { type: 'lit', ch: n }, next: pi + 2 };
  }
  if (pat[pi] === '.') return { atom: { type: 'any' }, next: pi + 1 };
  if (pat[pi] === '[') {
    let i = pi + 1;
    let neg = false;
    if (i < pat.length && pat[i] === '^') {
      neg = true;
      i++;
    }
    const chars = [];
    const ranges = [];
    while (i < pat.length && pat[i] !== ']') {
      if (i + 2 < pat.length && pat[i + 1] === '-' && pat[i + 2] !== ']') {
        ranges.push([pat[i].toLowerCase(), pat[i + 2].toLowerCase()]);
        i += 3;
      } else {
        chars.push(pat[i].toLowerCase());
        i++;
      }
    }
    if (i >= pat.length) return null;
    return { atom: { type: 'class', neg, chars, ranges }, next: i + 1 };
  }
  return { atom: { type: 'lit', ch: pat[pi] }, next: pi + 1 };
}

function parseCount(pat, pi) {
  // pi points at '{'
  let i = pi + 1;
  let num = '';
  while (i < pat.length && /[0-9]/.test(pat[i])) num += pat[i++];
  if (num === '') return null;
  const min = parseInt(num, 10);
  let max = min;
  if (i < pat.length && pat[i] === ',') {
    i++;
    if (i < pat.length && pat[i] === '}') {
      max = Number.MAX_SAFE_INTEGER;
      i++;
      return { min, max, next: i };
    }
    num = '';
    while (i < pat.length && /[0-9]/.test(pat[i])) num += pat[i++];
    if (num === '') return null;
    max = parseInt(num, 10);
  }
  if (i >= pat.length || pat[i] !== '}') return null;
  return { min, max, next: i + 1 };
}

function atomMatches(atom, ch) {
  const c = ch.toLowerCase();
  if (atom.type === 'any') return true;
  if (atom.type === 'lit') return c === atom.ch.toLowerCase();
  if (atom.type === 'class') {
    let ok = atom.chars.includes(c);
    for (const [a, b] of atom.ranges) {
      if (c >= a && c <= b) ok = true;
    }
    return atom.neg ? !ok : ok;
  }
  return false;
}

function matchHere(pat, piIn, text, ti) {
  let pi = piIn;
  if (pi >= pat.length) return ti === text.length;

  if (pi + 1 < pat.length && pat[pi] === '\\' && pat[pi + 1] === 'b') {
    if (!isWordBoundary(text, ti)) return false;
    return matchHere(pat, pi + 2, text, ti);
  }

  const ap = parseAtom(pat, pi);
  if (!ap) return null;
  let atom = ap.atom;
  pi = ap.next;

  let min = 1;
  let max = 1;
  if (pi < pat.length) {
    const q = pat[pi];
    if (q === '*') {
      min = 0;
      max = Number.MAX_SAFE_INTEGER;
      pi++;
    } else if (q === '+') {
      min = 1;
      max = Number.MAX_SAFE_INTEGER;
      pi++;
    } else if (q === '?') {
      min = 0;
      max = 1;
      pi++;
    } else if (q === '{') {
      const cq = parseCount(pat, pi);
      if (!cq) return null;
      min = cq.min;
      max = cq.max;
      pi = cq.next;
    }
  }

  let count = 0;
  let t = ti;
  while (count < max && t < text.length && atomMatches(atom, text[t])) {
    t++;
    count++;
  }
  if (count < min) return false;

  let k = count;
  while (true) {
    const sub = matchHere(pat, pi, text, ti + k);
    if (sub === true) return true;
    if (sub === null) return null;
    if (k <= min) break;
    k--;
  }
  return false;
}

/** Whole text must match pattern (anchored). false on bad pattern. */
export function regexFull(pattern, text) {
  const r = matchHere(pattern, 0, text, 0);
  return r === true;
}

function matchToken(pattern, text) {
  if (pattern === '*') return true;
  return regexFull(pattern, text);
}

function startsWithCI(s, prefix) {
  return s.toLowerCase().startsWith(prefix.toLowerCase());
}

/**
 * @param {string} allowlistText
 * @param {string} toolName  provider-facing tool name (e.g. term_local__bash)
 * @param {string} [command] command line for terminal tools
 */
export function isAllowed(allowlistText, toolName, command = '') {
  const text = String(allowlistText || '');
  if (!text.trim()) return false;

  const isTerminal =
    startsWithCI(toolName, 'terminal') ||
    startsWithCI(toolName, 'term_') ||
    /__(bash|exec)$/i.test(toolName);

  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    const colon = line.indexOf(':');
    const toolPat = (colon >= 0 ? line.slice(0, colon) : line).trim();
    const paramPat = colon >= 0 ? line.slice(colon + 1).trim() : null;

    if (!matchToken(toolPat, toolName)) continue;
    if (paramPat == null || paramPat === '') return true;
    if (!isTerminal) return true;
    if (matchToken(paramPat, command || '')) return true;
  }
  return false;
}

/** Lucy-ish default for angela MCP tools (local + remote term names). */
export const DEFAULT_LUCY_ALLOWLIST = `
# file (read-only)
file_io__list_dir
file_io__read_file
file_io__stat
file_io__list_dir:.*
file_io__read_file:.*
file_io__stat:.*

# local term (just-bash + host exec)
term_local__bash:pwd
term_local__bash:ls.*
term_local__bash:echo.*
term_local__bash:cat .+
term_local__bash:head .+
term_local__bash:uname.*
term_local__bash:true
term_local__exec:echo.*
term_local__exec:pwd
term_local__exec:true
term_local__exec:ls.*
`.trim();

export default { isAllowed, regexFull, DEFAULT_LUCY_ALLOWLIST };
