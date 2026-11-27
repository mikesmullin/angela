//! The session-properties gate — the seam the embedder uses to say whether a
//! session is fully configured before a turn starts.
//!
//! A session carries a `workspaceRoot` (the host file-io jail its tools
//! believe the disk begins at) and a `worktree` record (its git checkout,
//! provisioned and mounted read-write into its sandbox). Until the embedder
//! answers "both are set", every turn parks. That answer is a hook: the
//! embedder may repair the meta first (e.g. re-provision a worktree that
//! vanished) and only then answer — so [`set_check`] never reflects anything
//! other than what the embedder just did.
//!
//! The two hooks are process-wide and first-call-wins, like every other
//! seam in this crate: [`set_check`] answers "is this session ready?", and
//! [`set_on_deleted`] gets told the session went away, so the embedder can
//! reap the worktree the session owned.

use serde_json::Value;
use std::sync::OnceLock;

/// The embedder answers: is this session's meta fully configured (workspace
/// root + provisioned worktree)? `Err(reason)` = not yet (the reason is a
/// short human phrase for logs; the card text is fixed). The hook may repair
/// meta (e.g. re-provision a vanished worktree) before answering.
pub type Check = std::sync::Arc<dyn Fn(&Value) -> Result<(), String> + Send + Sync>;

/// The embedder is told a session was deleted. Best-effort: the hook owns
/// the session's worktree and may reap it.
pub type Deleted = std::sync::Arc<dyn Fn(&str) + Send + Sync>;

static CHECK: OnceLock<Check> = OnceLock::new();
static DELETED: OnceLock<Deleted> = OnceLock::new();

/// Register the readiness hook. First call wins (a second registration is
/// ignored, the same rule as every other process-wide seam).
pub fn set_check(f: Check) {
    let _ = CHECK.set(f);
}

/// Register the deletion hook. First call wins.
pub fn set_on_deleted(f: Deleted) {
    let _ = DELETED.set(f);
}

/// Answer the readiness question for `meta`. Ok(()) when no hook is
/// registered — a standalone binary has no embedder to ask, so it must not
/// park every turn.
pub fn check(meta: &Value) -> Result<(), String> {
    match CHECK.get() {
        Some(f) => f(meta),
        None => Ok(()),
    }
}

/// Tell the deletion hook (if any) that a session went away. A no-op when no
/// hook is registered.
pub fn on_session_deleted(id: &str) {
    if let Some(f) = DELETED.get() {
        f(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_no_hook_registered_the_check_passes() {
        assert!(check(&serde_json::json!({ "id": "s1" })).is_ok());
        assert!(check(&Value::Null).is_ok());
    }
}
