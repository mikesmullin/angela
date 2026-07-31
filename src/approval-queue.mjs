/**
 * Shared approval queue — same object for browser, CLI, and tests.
 *
 *   const q = new ApprovalQueue();
 *   const harness = await Angela.create({ onApproval: q.createApprover() });
 *   // later from UI:
 *   q.resolve(id, 'allow' | 'deny' | 'always');
 */

export class ApprovalQueue {
  constructor() {
    /** @type {Map<string, { resolve: Function, req: object, at: number }>} */
    this.pending = new Map();
    this.listeners = new Set();
  }

  /** Subscribe to new approval requests (UI). */
  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  _notify(req) {
    for (const fn of this.listeners) {
      try {
        fn(req);
      } catch {
        /* ignore */
      }
    }
  }

  /**
   * @returns {(req: object) => Promise<'allow'|'deny'|'always'>}
   */
  createApprover() {
    return (req) =>
      new Promise((resolve) => {
        this.pending.set(req.id, { resolve, req, at: Date.now() });
        this._notify(req);
      });
  }

  /**
   * @param {string} id
   * @param {'allow'|'deny'|'always'|string} decision
   */
  resolve(id, decision) {
    const entry = this.pending.get(id);
    if (!entry) return false;
    this.pending.delete(id);
    const d = String(decision || 'deny').toLowerCase();
    const norm =
      d === 'y' || d === 'yes' || d === 'allow'
        ? 'allow'
        : d === 'a' || d === 'always'
          ? 'always'
          : 'deny';
    entry.resolve(norm);
    return true;
  }

  list() {
    return [...this.pending.values()].map((p) => ({ ...p.req, at: p.at }));
  }

  clear(decision = 'deny') {
    for (const [id] of this.pending) {
      this.resolve(id, decision);
    }
  }
}

export default ApprovalQueue;
