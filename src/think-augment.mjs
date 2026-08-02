/**
 * Think / reasoning-effort augmentation for Angela + AGL.
 *
 * - reasoning_effort: low|medium|high|xhigh|max → passed to AGL (Copilot/Anthropic
 *   effort, etc.) on every inference.
 * - think + model matching *:gemma4: inject `<|think|>` and a depth phrase into
 *   the system prompt (LM Studio Gemma 4 workaround — no native effort API).
 */

/** @type {readonly string[]} */
export const REASONING_LEVELS = Object.freeze([
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
]);

/**
 * Anthropic-style effort levels → system-prompt guidance for Gemma 4.
 * Applied after `<|think|>` when think is enabled.
 */
export const GEMMA_REASONING_PHRASES = Object.freeze({
  low: 'Keep your thinking minimal and brief. Prefer the simplest correct path and avoid overthinking. Jump quickly to a direct answer.',
  medium:
    'Reason carefully with moderate depth. Cover the important points step by step, but stay focused and avoid unnecessary length.',
  high: 'Think thoroughly and carefully step by step. Examine the problem from the key angles and verify your reasoning before answering.',
  xhigh:
    'Engage in extended deep reasoning. Explore multiple approaches, check intermediate steps carefully, and consider edge cases and alternatives thoroughly.',
  max: 'Reason at maximum depth. Analyze exhaustively from first principles, explore many alternatives, self-critique rigorously, and take as much thinking as needed for the highest-quality answer.',
});

/**
 * True when the model id looks like Gemma 4 (pattern `*:gemma4` / gemma-4 / gemma_4).
 * @param {string|null|undefined} model
 */
export function isGemma4Model(model) {
  return /gemma[-_]?4/i.test(String(model || ''));
}

/**
 * Normalize effort to a known level or null.
 * @param {string|null|undefined} level
 * @returns {string|null}
 */
export function normalizeReasoningEffort(level) {
  if (level == null || level === '' || level === false) return null;
  const s = String(level).toLowerCase().trim();
  if (s === 'off' || s === 'none' || s === 'false') return null;
  if (REASONING_LEVELS.includes(s)) return s;
  // common aliases
  if (s === 'x-high' || s === 'extra-high') return 'xhigh';
  if (s === 'minimal' || s === 'min') return 'low';
  if (s === 'maximum') return 'max';
  return null;
}

/**
 * Build the Gemma-4 system-prompt prefix for think + effort.
 * @param {{ think?: boolean, reasoning_effort?: string|null }} opts
 * @returns {string} empty, or `<|think|>\n` + optional phrase + blank lines
 */
export function gemma4ThinkPrefix(opts = {}) {
  if (!opts.think) return '';
  const level = normalizeReasoningEffort(opts.reasoning_effort) || 'medium';
  const phrase = GEMMA_REASONING_PHRASES[level] || GEMMA_REASONING_PHRASES.medium;
  return `<|think|>\n${phrase}\n\n`;
}

/**
 * Augment a system prompt with think/effort for the active model.
 * Non-Gemma models: returns system unchanged (effort is passed on the API via AGL).
 *
 * @param {string|null|undefined} system
 * @param {{
 *   model?: string|null,
 *   think?: boolean,
 *   reasoning_effort?: string|null,
 * }} [opts]
 * @returns {string}
 */
export function augmentSystemPrompt(system, opts = {}) {
  const base = system == null ? '' : String(system);
  if (!opts.think) return base;
  if (!isGemma4Model(opts.model)) return base;
  const prefix = gemma4ThinkPrefix(opts);
  if (!prefix) return base;
  // Avoid double-prefix if caller already injected
  if (base.startsWith('<|think|>')) return base;
  return prefix + base;
}
