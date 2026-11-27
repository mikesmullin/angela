use std::io;
use std::path::{Path, PathBuf};

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

#[derive(Debug, Clone)]
pub struct Layout {
    pub root: PathBuf,
    pub angela: PathBuf,
    pub agents: PathBuf,
    pub sessions: PathBuf,
}

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

pub fn agent_search_dirs(root: &Path) -> Vec<PathBuf> {
    vec![
        agents_dir(root),
        root.join("agents"),
        root.to_path_buf(),
    ]
}
