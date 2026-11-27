use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{json, Value};

use crate::agent_loader::AgentDef;
use crate::cancel::Cancel;
use crate::conv_store::ConvStore;

const BOOT_TIMEOUT: Duration = Duration::from_secs(180);
const BOOT_POLL: Duration = Duration::from_millis(500);

pub struct SandboxRow {
    pub id: String,
    pub image: String,
    pub state: String,
    pub created_at: String,
}

pub trait Backend: Send + Sync {
    fn create(
        &self,
        image: &str,
        worktree_path: &str,
        repo_git_dir: &str,
    ) -> Result<String, String>;
    fn prepare_worktree(&self, _id: &str) -> Result<(), String> { Ok(()) }
    fn state(&self, id: &str) -> Option<String>;
    fn destroy(&self, id: &str) -> Result<(), String>;
    fn list(&self) -> Result<Vec<SandboxRow>, String>;
    fn exec_argv(&self, container: &str, guest: &[String], env: &[(String, String)]) -> Vec<String>;
}

static BACKEND: OnceLock<Box<dyn Backend>> = OnceLock::new();

pub fn set_backend(backend: Box<dyn Backend>) {
    let _ = BACKEND.set(backend);
}

pub fn backend_available() -> bool {
    BACKEND.get().is_some()
}

fn backend() -> Result<&'static Box<dyn Backend>, String> {
    BACKEND
        .get()
        .ok_or_else(|| "no sandbox backend is configured (agent sandboxes need the embedding app's sbox client)".to_string())
}

pub fn exec_spec(
    container: &str,
    guest: &crate::mcp::ServerSpec,
) -> Result<crate::mcp::ServerSpec, String> {
    let be = backend()?;
    let mut argv: Vec<String> = vec![guest.command.clone()];
    argv.extend(guest.args.iter().cloned());
    let mut host = be.exec_argv(container, &argv, &guest.env);
    let command = host.remove(0);
    Ok(crate::mcp::ServerSpec {
        name: guest.name.clone(),
        command,
        args: host,
        cwd: None,
        env: Vec::new(),
        framing: guest.framing.clone(),
    })
}

fn image_for(containerfile: &str) -> String {
    let name = match Path::new(containerfile).parent().and_then(|p| p.file_name()) {
        Some(n) => n.to_str().unwrap_or("containerfile").to_string(),
        None => "containerfile".to_string(),
    };
    if name == "headed" || name == "headless" {
        return name;
    }
    format!("sbox-{name}:latest")
}

fn live(state: Option<&str>) -> bool {
    matches!(state, Some("ready" | "starting" | "cloning"))
}

fn create_and_wait(
    cancel: &Cancel,
    image: &str,
    worktree_path: &str,
    repo_git_dir: &str,
) -> Result<String, String> {
    let be = backend()?;
    let id = be.create(image, worktree_path, repo_git_dir)?;
    let iterations = (BOOT_TIMEOUT.as_millis() as usize) / (BOOT_POLL.as_millis() as usize).max(1);
    for _ in 0..iterations {
        if cancel.stopped() {
            return Err("stopped while the sandbox was booting".into());
        }
        match be.state(&id).as_deref() {
            Some("ready") => return Ok(id),
            Some("error" | "stopped") => {
                return Err(format!("sandbox {id} failed to boot (image {image})"))
            }
            _ => {}
        }
        std::thread::sleep(BOOT_POLL);
    }
    Ok(id)
}

fn session_mount(store: &ConvStore, session_id: &str) -> Result<(String, String), String> {
    let view = store.effective_session(session_id).ok_or("session not found")?;
    let session = &view["session"];
    let effective = &view["effective"];
    let authored = session.get("worktree").filter(|v| v.is_object());
    let inherited_configured = effective.get("worktree").is_some_and(Value::is_object);
    let record = authored.cloned().or_else(|| {
        if !inherited_configured { return None; }
        let config = &effective["worktree"];
        store.inherited_worktree(session_id).filter(|record| {
            ["repoId", "branchPattern", "pathPattern"].iter().all(|key| {
                record.get(*key).and_then(Value::as_str).unwrap_or("")
                    == config.get(*key).and_then(Value::as_str).unwrap_or("")
            })
        })
    });
    if let Some(record) = record {
        if !record.get("provisioned").and_then(Value::as_bool).unwrap_or(false) {
            return Err("worktree is not provisioned".into());
        }
        let path = record.get("path").and_then(Value::as_str).unwrap_or("");
        if !Path::new(path).is_absolute() || !Path::new(path).is_dir() {
            return Err("worktree path is not an existing absolute directory".into());
        }
        let output = std::process::Command::new("git")
            .args(["-C", path, "rev-parse", "--git-common-dir"])
            .output().map_err(|e| format!("cannot inspect worktree git directory: {e}"))?;
        if !output.status.success() { return Err("cannot inspect worktree git directory".into()); }
        let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let common = Path::new(&raw);
        let common = if common.is_absolute() { common.to_path_buf() } else { Path::new(path).join(common) };
        let common = common.canonicalize().map_err(|e| format!("cannot resolve worktree git directory: {e}"))?;
        return Ok((path.to_string(), common.to_string_lossy().into_owned()));
    }
    if inherited_configured { return Err("inherited worktree is not provisioned".into()); }
    let workspace = effective.get("workspaceRoot").and_then(Value::as_str).unwrap_or("").trim();
    if !workspace.is_empty() && (!Path::new(workspace).is_absolute() || !Path::new(workspace).is_dir()) {
        return Err("workspace root is not an existing absolute directory".into());
    }
    Ok((workspace.to_string(), String::new()))
}

pub fn ensure(
    store: &ConvStore,
    session_id: &str,
    def: &AgentDef,
    cancel: &Cancel,
) -> Result<Option<String>, String> {
    let containerfile = def.containerfile.trim();
    if containerfile.is_empty() {
        return Ok(None);
    }
    let effective = store.effective_session(session_id).ok_or("session not found")?["effective"].clone();
    crate::session_props::check(&effective, def)?;
    let (worktree, git_dir) = session_mount(store, session_id)?;
    if let Some(cid) = store.sandbox_for(session_id, &def.name) {
        if live(backend().ok().and_then(|be| be.state(&cid)).as_deref()) {
            let prior = store.sandbox_mount(session_id, &def.name);
            let matches = prior.as_ref().is_some_and(|(p, g)| p == &worktree && g == &git_dir)
                || (prior.is_none() && worktree.is_empty() && git_dir.is_empty());
            if !matches {
                return Err(format!("the existing sandbox for @{} has different or unknown mounts; remove it before starting a turn with the new checkout", def.name));
            }
            if !git_dir.is_empty() { backend()?.prepare_worktree(&cid)?; }
            return Ok(Some(cid));
        }
        store.forget_sandbox(session_id, &def.name);
    }
    let cid = create_and_wait(cancel, &image_for(containerfile), &worktree, &git_dir)
        .map_err(|e| format!("could not start the sandbox for @{}: {e}", def.name))?;
    store.set_sandbox(session_id, &def.name, &cid);
    store.set_sandbox_mount(session_id, &def.name, &worktree, &git_dir)
        .map_err(|e| format!("could not record sandbox mounts: {e}"))?;
    if !git_dir.is_empty() { backend()?.prepare_worktree(&cid)?; }
    Ok(Some(cid))
}

pub fn destroy_for_session(store: &ConvStore, session_id: &str) -> usize {
    let mut n = 0;
    if let Ok(be) = backend() {
        for (_agent, cid) in store.sandboxes_for(session_id) {
            if be.destroy(&cid).is_ok() {
                n += 1;
            }
        }
    }
    store.forget_sandboxes_by_session(session_id);
    n
}

pub fn destroy_all(store: &ConvStore) {
    if let Ok(be) = backend() {
        for (_sid, _agent, cid) in store.sandbox_rows() {
            let _ = be.destroy(&cid);
        }
    }
    for (sid, _agent, _cid) in store.sandbox_rows() {
        store.forget_sandboxes_by_session(&sid);
    }
}

pub fn list_for_menu(store: &ConvStore) -> Result<Value, String> {
    let be = backend()?;
    let rows = be.list()?;
    let by_container: HashMap<String, (String, String)> = store
        .sandbox_rows()
        .into_iter()
        .map(|(sid, aid, cid)| (cid, (sid, aid)))
        .collect();
    let alive: HashSet<String> = store.list().into_iter().collect();

    let mut out_rows: Vec<Value> = Vec::new();
    for row in &rows {
        let (session, agent) = by_container
            .get(&row.id)
            .filter(|(sid, _)| alive.contains(sid))
            .cloned()
            .unwrap_or_default();
        out_rows.push(json!({
            "id": row.id,
            "image": row.image,
            "state": row.state,
            "session": session,
            "agent": agent,
            "createdAt": row.created_at,
        }));
    }
    Ok(Value::Array(out_rows))
}
