//! Agent-owned sandboxes (PLAN55 D2): one container per (session → agent)
//! pair, lazily launched at that agent's first turn in that session.
//!
//! ## Who owns what
//!
//! This module owns the **policy**: which (session, agent) runs in which
//! container, when one is launched (first turn), and when one dies (session
//! deletion). It does **not** own the transport to sbox-b: that stays in
//! chat's `sbox.rs` — the seam where the image and the bind mounts are
//! decided (PLAN55, "the seam") — behind the [`Backend`] trait registered at
//! startup. There is exactly one sbox-b client, and the worktree it mounts is
//! the session's own: destroying a container never reaps the checkout (the
//! session owns it).
//!
//! Without a registered backend (standalone `angela-serve`) sandboxed turns
//! answer an error rather than silently running on the host: an unreachable
//! sandbox is an error, not a fallback.
//!
//! ## Image convention (PLAN55 D7)
//! `sbox-<the Containerfile's parent directory>:latest` — unless the parent
//! directory is a bare sandbox-backend KIND name (`headed` / `headless`), in
//! which case the bare kind is passed through and the backend maps it to its
//! image matrix.
//! `/workspace/models/sbox/video/Containerfile` → `sbox-video:latest`;
//! `/workspace/models/sbox/headed/Containerfile` → `headed`.
//!
//! ## Lifecycle
//! - [`ensure`] is idempotent per (session, agent): a living container is
//!   reused; a dead or absent one is (re)created and the turn **blocks** until
//!   it reports `ready` (PLAN55: "the turn blocks until the instance has
//!   booted its image").
//! - [`destroy_for_session`] tears down every container a session owns
//!   (the deletion cascade, PLAN55 D2).
//! - [`list_for_menu`] feeds the sandboxes menu: each container with the
//!   (session, agent) that serves it, plus a per-image live rollup (D6, D13 —
//!   there is deliberately no per-container session count: it is always 1).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{json, Value};

use crate::agent_loader::AgentDef;
use crate::cancel::Cancel;
use crate::conv_store::ConvStore;

/// How long a turn will wait for its image to boot before giving up. The
/// container is left alone (it may still come up); the next turn reuses it.
const BOOT_TIMEOUT: Duration = Duration::from_secs(180);
const BOOT_POLL: Duration = Duration::from_millis(500);

/// One tracked container, as the menu shows it.
pub struct SandboxRow {
    pub id: String,
    pub image: String,
    pub state: String,
    /// When sbox-b created it (ISO-8601), as the form shows it.
    pub created_at: String,
}

/// The transport to sbox-b — implemented by the embedding app (chat's
/// `sbox.rs`). Every method is the seam's concern: `create` decides the
/// image's volumes — the session's worktree directory, read-write, at
/// [`crate::harness::SANDBOX_MOUNT`] (an empty path mounts nothing) — and
/// the embedder owns reaping: destroying a sandbox never reaps a worktree.
pub trait Backend: Send + Sync {
    /// Launch `image` as a new sandbox and return its id (sbox-b's, which is
    /// the container's identity — there are no friendly names, PLAN55 D3/D10).
    /// `worktree_path` is the session's provisioned worktree, mounted read-write
    /// at [`crate::harness::SANDBOX_MOUNT`] (empty = mount nothing).
    /// Create a sandbox whose `worktree_path` (when non-empty) is mounted
    /// read-write at [`crate::harness::SANDBOX_MOUNT`], and whose
    /// `repo_git_dir` (when non-empty) is mounted at its own absolute path
    /// so the checkout's gitdir pointer resolves inside the guest. Both are
    /// empty for "mount nothing".
    fn create(
        &self,
        image: &str,
        worktree_path: &str,
        repo_git_dir: &str,
    ) -> Result<String, String>;
    /// The container's state (`ready`, `starting`, …), `None` when sbox-b no
    /// longer tracks it.
    fn state(&self, id: &str) -> Option<String>;
    /// Stop and delete a container. Destroying one sandbox never reaps the
    /// checkout — the session owns it.
    fn destroy(&self, id: &str) -> Result<(), String>;
    /// Every container sbox-b tracks (the menu's raw rows).
    fn list(&self) -> Result<Vec<SandboxRow>, String>;
    /// The host argv that runs `guest` (command + args) **inside** `container`
    /// with its stdin/stdout piped back — e.g.
    /// `podman exec -i <id> /usr/local/bin/bun /opt/video/server.mjs`. `env`
    /// travels as the runtime's own `-e` flags (the guest sees its image's
    /// environment regardless).
    fn exec_argv(&self, container: &str, guest: &[String], env: &[(String, String)]) -> Vec<String>;
}

static BACKEND: OnceLock<Box<dyn Backend>> = OnceLock::new();

/// Register the transport. Called once at startup by the embedding app; a
/// second call is ignored (the first registration wins, like every other
/// process-wide seam).
pub fn set_backend(backend: Box<dyn Backend>) {
    let _ = BACKEND.set(backend);
}

/// Whether a transport exists. Without one, sandboxed turns error out.
pub fn backend_available() -> bool {
    BACKEND.get().is_some()
}

fn backend() -> Result<&'static Box<dyn Backend>, String> {
    BACKEND
        .get()
        .ok_or_else(|| "no sandbox backend is configured (agent sandboxes need the embedding app's sbox client)".to_string())
}

/// Wrap a guest launch spec in the runtime's `exec`: the host argv that runs
/// the command inside `container` (PLAN55 — an image's own MCP servers run
/// where their side effects belong; the video pipeline's transcript lands in
/// the container that will die with its session). The returned spec is an
/// ordinary stdio spec to the MCP client.
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
        // The `exec` client's own cwd is irrelevant; the guest process takes
        // the image's default.
        cwd: None,
        env: Vec::new(),
        framing: guest.framing.clone(),
    })
}

/// The image (or kind) for an agent whose Containerfile lives at
/// `containerfile`.
///
/// PLAN55 D7: `sbox-<parent dir of the Containerfile>:latest` — unless the
/// parent directory is exactly a sandbox-backend KIND name (`headed` /
/// `headless`), in which case the bare kind name is returned: the backend maps
/// kinds to its own image matrix, and `sbox-headed` would invent a name no
/// matrix entry can produce.
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

/// Still usable for inference: ready, or still coming up. A turn that reuses a
/// `starting` box simply waits for it to finish (it already booted once).
fn live(state: Option<&str>) -> bool {
    matches!(state, Some("ready" | "starting" | "cloning"))
}

/// Create a container and **block** until it is `ready` (or the deadline hits
/// / the switch is thrown). Returns the new id, or `Err` when creation itself
/// failed — the caller's turn waits on this (PLAN55: the turn blocks until the
/// image boots).
fn create_and_wait(
    cancel: &Cancel,
    image: &str,
    worktree_path: &str,
    repo_git_dir: &str,
) -> Result<String, String> {
    let be = backend()?;
    let id = be.create(image, worktree_path, repo_git_dir)?;
    // Both in ms — dividing seconds by milliseconds here once made the loop
    // run zero times and the turn race the boot.
    let iterations = (BOOT_TIMEOUT.as_millis() as usize) / (BOOT_POLL.as_millis() as usize).max(1);
    for _ in 0..iterations {
        if cancel.stopped() {
            return Err("stopped while the sandbox was booting".into());
        }
        match be.state(&id).as_deref() {
            Some("ready") => return Ok(id),
            // Boot failed: do not spin until the deadline on a dead box.
            Some("error" | "stopped") => {
                return Err(format!("sandbox {id} failed to boot (image {image})"))
            }
            _ => {}
        }
        // Sleep every iteration — including when the state probe answered
        // nothing at all, which is exactly when a tight loop would spin.
        std::thread::sleep(BOOT_POLL);
    }
    // Timed out while still starting: leave it — the next turn reuses it once
    // it does come up.
    Ok(id)
}

/// The whole resolve: (session, agent) → a live container id, launching on
/// first use.
///
/// * `Ok(None)` — the agent has no Containerfile: host mode, exactly today's
///   behaviour.
/// * `Ok(Some(id))` — the agent's turns run in this container.
/// * `Err(msg)` — the agent has a Containerfile but no sandbox could be
///   provided. **Not** a fallback to the host: an agent that is defined to run
///   in a sandbox must never quietly write to the host instead.
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
    // Reuse a living one if this (session, agent) already has one.
    if let Some(cid) = store.sandbox_for(session_id, &def.name) {
        if live(backend().ok().and_then(|be| be.state(&cid)).as_deref()) {
            return Ok(Some(cid));
        }
        // Stale: the container is gone. Drop the mapping; recreate below.
        store.forget_sandbox(session_id, &def.name);
    }
    // The session's provisioned checkout — the mount the seam binds in
    // (empty is fine: a transcription sandbox needs no worktree).
    let (worktree, git_dir) = session_checkout(store, session_id);
    let cid = create_and_wait(cancel, &image_for(containerfile), &worktree, &git_dir)
        .map_err(|e| format!("could not start the sandbox for @{}: {e}", def.name))?;
    store.set_sandbox(session_id, &def.name, &cid);
    Ok(Some(cid))
}

/// The session's provisioned worktree path — empty when the session has no
/// provisioned worktree. A legacy plain-string `worktree` column (or anything
/// that is not an object with `provisioned == true` and a non-empty `path`)
/// answers empty, never a guess.
/// The session's checkout as the mounts a sandbox needs: the worktree
/// directory (mounted at `SANDBOX_MOUNT`) and the repo's `.git` directory
/// (mounted at its own absolute path — the worktree's gitdir pointer is an
/// absolute host path, and git inside the guest only works while it
/// resolves). Both empty when the session has no provisioned worktree.
fn session_checkout(store: &ConvStore, session_id: &str) -> (String, String) {
    let meta = store.load(session_id).unwrap_or(Value::Null);
    let Some(record) = meta.get("worktree") else {
        return (String::new(), String::new());
    };
    let Some(obj) = record.as_object() else {
        return (String::new(), String::new());
    };
    let provisioned = obj.get("provisioned").and_then(Value::as_bool).unwrap_or(false);
    let path = obj.get("path").and_then(Value::as_str).unwrap_or_default();
    if !provisioned || path.trim().is_empty() {
        return (String::new(), String::new());
    }
    let git_dir = obj
        .get("repoPath")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| format!("{p}/.git"))
        .unwrap_or_default();
    (path.to_string(), git_dir)
}

/// Destroy every container the session owns (PLAN55 D2 cascade). Best-effort
/// per container: one dead box must not stop the others from being reaped.
/// Destroying one sandbox never reaps the checkout — the session owns it —
/// and the mapping rows go even when the destroy fails, so a session that is
/// gone leaves nothing pointing at anything.
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

/// `angela clean`: every mapped container goes with every session.
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

/// The menu data: every container the seam tracks, annotated with the
/// (session, agent) that owns it — only when that session still exists.
///
/// Shape: a plain array of rows —
/// ```jsonc
/// [ { "id": "…", "image": "sbox-video:latest", "state": "ready",
///     "session": "<id>", "agent": "video" } ]
/// ```
/// The per-image rollup (D6) is derived from these rows at render time by
/// `sandboxes_menu::group_rows`; callers that show a listing per viewer
/// filter *these rows* (the visibility gate) and the rollup follows.
pub fn list_for_menu(store: &ConvStore) -> Result<Value, String> {
    let be = backend()?;
    let rows = be.list()?;
    // (container_id) -> (session_id, agent_name)
    let by_container: HashMap<String, (String, String)> = store
        .sandbox_rows()
        .into_iter()
        .map(|(sid, aid, cid)| (cid, (sid, aid)))
        .collect();
    // The sessions that still exist (a deleted session's row must not
    // annotate a container — the FK cascade removes it, but be explicit).
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
    // Just the rows: the per-image rollup is derived from them at render
    // time (one grouping, `sandboxes_menu::group_rows`), and a second copy
    // served alongside would be a second answer to keep honest — and a
    // side channel around the viewer's listing gate.
    Ok(Value::Array(out_rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    #[test]
    fn the_image_name_follows_the_containerfile_convention() {
        assert_eq!(
            image_for("/workspace/models/sbox/video/Containerfile"),
            "sbox-video:latest"
        );
        assert_eq!(image_for("/srv/sbox/chewy/Containerfile"), "sbox-chewy:latest");
        assert_eq!(image_for("no-parent"), "sbox-containerfile:latest");
    }

    #[test]
    fn a_kind_name_parent_passes_through_as_a_kind() {
        // headed/headless are sandbox-backend KIND names, not images — the
        // backend maps them onto its own image matrix.
        assert_eq!(image_for("/workspace/models/sbox/headed/Containerfile"), "headed");
        assert_eq!(image_for("/workspace/models/sbox/headless/Containerfile"), "headless");
    }

    #[test]
    fn live_states_are_the_ones_a_turn_may_reuse() {
        assert!(live(Some("ready")));
        assert!(live(Some("starting")));
        assert!(live(Some("cloning")));
        assert!(!live(Some("error")));
        assert!(!live(Some("stopped")));
        assert!(!live(None));
    }

    /// One backend, dumb on purpose: boots after `boot_polls` state probes.
    /// The OnceLock seam allows exactly one registration per process, so the
    /// whole lifecycle is one test — and anything else in the test binary that
    /// deletes a session may call `destroy` here with ids this mock ignores.
    struct MockBackend {
        boot_polls: Arc<AtomicU32>,
        destroyed: Arc<Mutex<Vec<String>>>,
    }

    impl Backend for MockBackend {
        fn create(
            &self,
            _image: &str,
            _worktree_path: &str,
            _repo_git_dir: &str,
        ) -> Result<String, String> {
            Ok("mock-container".into())
        }
        fn state(&self, _id: &str) -> Option<String> {
            // "creating" for the first two probes, then "ready" — a turn must
            // WAIT through those (the seconds/millis loop-bound bug returned
            // before the first probe and raced the boot).
            let n = self.boot_polls.fetch_add(1, Ordering::SeqCst);
            Some(if n < 2 { "creating".into() } else { "ready".into() })
        }
        fn destroy(&self, id: &str) -> Result<(), String> {
            self.destroyed.lock().unwrap().push(id.to_string());
            Ok(())
        }
        fn list(&self) -> Result<Vec<SandboxRow>, String> {
            Ok(vec![SandboxRow {
                id: "mock-container".into(),
                image: "sbox-video:latest".into(),
                state: "ready".into(),
                created_at: "2026-11-23T00:00:00Z".into(),
            }])
        }
        fn exec_argv(&self, container: &str, guest: &[String], _env: &[(String, String)]) -> Vec<String> {
            let mut v = vec!["mock-runtime".into(), "exec".into(), "-i".into(), container.to_string()];
            v.extend(guest.iter().cloned());
            v
        }
    }

    #[test]
    fn a_first_turn_boots_the_container_and_the_session_owns_it() {
        let probes = Arc::new(AtomicU32::new(0));
        let destroyed = Arc::new(Mutex::new(Vec::new()));
        set_backend(Box::new(MockBackend {
            boot_polls: Arc::clone(&probes),
            destroyed: Arc::clone(&destroyed),
        }));

        let store = ConvStore::memory("/tmp/sandbox-policy-test").unwrap();
        let id = store
            .create(crate::session_store::NewSession::default())
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        let mut def = AgentDef::default();
        def.name = "video".into();
        def.containerfile = "/workspace/models/sbox/video/Containerfile".into();

        // An agent with no Containerfile is host mode: no container at all.
        let host_def = AgentDef::default();
        assert_eq!(ensure(&store, &id, &host_def, &Cancel::never()).unwrap(), None);

        // The sandboxed agent: the turn must block until the box is ready —
        // several state probes, not zero.
        let got = ensure(&store, &id, &def, &Cancel::never()).unwrap();
        assert_eq!(got.as_deref(), Some("mock-container"));
        let polled = probes.load(Ordering::SeqCst);
        assert!(polled >= 3, "the turn polled the state {polled} times — it must wait for ready");
        assert_eq!(store.sandbox_for(&id, "video").as_deref(), Some("mock-container"));

        // A second turn reuses the mapping — no new container.
        assert_eq!(ensure(&store, &id, &def, &Cancel::never()).unwrap().as_deref(), Some("mock-container"));

        // The menu annotates the container with its (session, agent).
        let menu = list_for_menu(&store).unwrap();
        assert_eq!(menu[0]["session"], json!(id));
        assert_eq!(menu[0]["agent"], json!("video"));

        // Deletion cascade: the container is destroyed with its session.
        assert!(store.delete(&id));
        assert!(store.sandbox_for(&id, "video").is_none());
        assert!(destroyed.lock().unwrap().contains(&"mock-container".to_string()));
    }
}
