use crate::agent_loader::AgentDef;
use serde_json::Value;
use std::sync::OnceLock;

pub type Check = std::sync::Arc<dyn Fn(&Value, &AgentDef) -> Result<(), String> + Send + Sync>;

pub type Deleted = std::sync::Arc<dyn Fn(&str) + Send + Sync>;

static CHECK: OnceLock<Check> = OnceLock::new();
static DELETED: OnceLock<Deleted> = OnceLock::new();

pub fn set_check(f: Check) {
    let _ = CHECK.set(f);
}

pub fn set_on_deleted(f: Deleted) {
    let _ = DELETED.set(f);
}

pub fn check(meta: &Value, agent: &AgentDef) -> Result<(), String> {
    match CHECK.get() {
        Some(f) => f(meta, agent),
        None => Ok(()),
    }
}

pub fn on_session_deleted(id: &str) {
    if let Some(f) = DELETED.get() {
        f(id);
    }
}
