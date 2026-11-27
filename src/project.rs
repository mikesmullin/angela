//! Port of `angela/src/project.mjs` — the project root and `.angela/` layout.
//!
//! ```text
//! <root>/
//!   .angela/
//!     agents/     # -a my-agent  → .angela/agents/my-agent.<ext>
//!     sessions/   # durable session state + inference logs
//! ```

use std::io;
use std::path::{Path, PathBuf};

/// Project root: an explicit `-c/--config` path, else the process working directory.
pub fn resolve_project_root(config_path: Option<&str>) -> PathBuf {
    match config_path {
        Some(p) if !p.trim().is_empty() => {
            let p = Path::new(p.trim());
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join(p)
            }
        }
        _ => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    }
}

pub fn angela_dir(root: &Path) -> PathBuf {
    root.join(".angela")
}

pub fn agents_dir(root: &Path) -> PathBuf {
    angela_dir(root).join("agents")
}

pub fn sessions_dir(root: &Path) -> PathBuf {
    angela_dir(root).join("sessions")
}

/// The three paths `ensure_angela_layout` guarantees.
#[derive(Debug, Clone)]
pub struct Layout {
    pub root: PathBuf,
    pub angela: PathBuf,
    pub agents: PathBuf,
    pub sessions: PathBuf,
}

/// Ensure `.angela/{agents,sessions}` exist.
pub fn ensure_angela_layout(root: &Path) -> io::Result<Layout> {
    let layout = Layout {
        root: root.to_path_buf(),
        angela: angela_dir(root),
        agents: agents_dir(root),
        sessions: sessions_dir(root),
    };
    for d in [&layout.angela, &layout.agents, &layout.sessions] {
        if !d.exists() {
            std::fs::create_dir_all(d)?;
        }
    }
    Ok(layout)
}

/// Directories searched for `-a <name>`, first hit wins.
pub fn agent_search_dirs(root: &Path) -> Vec<PathBuf> {
    vec![
        agents_dir(root),          // primary
        root.join("agents"),       // legacy/local convenience
        root.to_path_buf(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_created_and_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let l = ensure_angela_layout(tmp.path()).unwrap();
        assert!(l.agents.is_dir());
        assert!(l.sessions.is_dir());
        // second call must not fail
        ensure_angela_layout(tmp.path()).unwrap();
        assert!(l.sessions.is_dir());
    }

    #[test]
    fn search_order_puts_the_angela_dir_first() {
        let dirs = agent_search_dirs(Path::new("/p"));
        assert_eq!(dirs[0], PathBuf::from("/p/.angela/agents"));
        assert_eq!(dirs[1], PathBuf::from("/p/agents"));
        assert_eq!(dirs[2], PathBuf::from("/p"));
    }
}
