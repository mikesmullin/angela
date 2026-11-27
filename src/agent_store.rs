use rusqlite::{params, OptionalExtension};
use serde_json::{json, Map, Value};

use crate::agent_loader::{AgentDef, Engagement};
use crate::conv_store::ConvStore;
use crate::session_store::now_iso;

#[derive(Debug, Clone, PartialEq)]
pub enum AgentError {
    EmptyName,
    BadName(String),
    Duplicate(String),
    NotFound(String),
    BadContainerfile(String),
    Db(String),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::EmptyName => write!(f, "an agent needs a name"),
            AgentError::BadName(n) => write!(
                f,
                "\"{n}\" is not a usable agent name — use letters, digits, '-' or '_'"
            ),
            AgentError::Duplicate(n) => write!(f, "an agent named \"{n}\" already exists"),
            AgentError::NotFound(n) => write!(f, "no agent named \"{n}\""),
            AgentError::BadContainerfile(p) => write!(
                f,
                "\"{p}\" is not an absolute host path — a Containerfile must be one (PLAN55 D1)"
            ),
            AgentError::Db(e) => write!(f, "{e}"),
        }
    }
}

pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

pub struct AgentStore<'a> {
    pub store: &'a ConvStore,
}

fn json_text(v: &Value) -> String {
    v.to_string()
}

fn parse_list(text: Option<String>) -> Vec<String> {
    text.and_then(|t| serde_json::from_str::<Vec<String>>(&t).ok())
        .unwrap_or_default()
}

impl<'a> AgentStore<'a> {
    pub fn new(store: &'a ConvStore) -> Self {
        AgentStore { store }
    }

    const COLS: &'static str = "id, name, description, model, models, context_window, env, \
         mcp, allowlist, tools_enabled, policy_mode, parallel_tools, soul, goals, memory, \
         engagement, starters, avatar, accent_color, reasoning_effort, max_turns, icon, \
         containerfile";

    fn row(r: &rusqlite::Row<'_>) -> rusqlite::Result<AgentDef> {
        Ok(AgentDef {
            id: Some(r.get(0)?),
            name: r.get(1)?,
            description: r.get(2)?,
            model: r.get(3)?,
            models: parse_list(r.get(4)?),
            context_window: r.get::<_, Option<i64>>(5)?.map(|v| v.max(0) as u64),
            env: r
                .get::<_, Option<String>>(6)?
                .and_then(|t| serde_json::from_str(&t).ok())
                .unwrap_or_default(),
            mcp: parse_list(r.get(7)?),
            allowlist: r.get(8)?,
            tools_enabled: r.get::<_, Option<String>>(9)?.and_then(|t| {
                let v: Vec<String> = serde_json::from_str(&t).ok()?;
                if v.is_empty() {
                    None
                } else {
                    Some(v)
                }
            }),
            policy_mode: r.get(10)?,
            parallel_tools: r.get::<_, i64>(11)? != 0,
            soul: r.get(12)?,
            goals: r.get(13)?,
            memory: r.get(14)?,
            engagement: r.get(15)?,
            starters: r
                .get::<_, Option<String>>(16)?
                .and_then(|t| serde_json::from_str(&t).ok())
                .unwrap_or_default(),
            avatar: r.get(17)?,
            accent_color: r.get(18)?,
            reasoning_effort: r.get(19)?,
            max_turns: r.get::<_, Option<i64>>(20)?.map(|v| v as u32),
            icon: r.get(21)?,
            containerfile: r.get::<_, String>(22).unwrap_or_default(),
            file: None,
        })
    }

    pub fn list(&self) -> Vec<AgentDef> {
        self.store.with(|c| {
            let mut st = match c.prepare(&format!(
                "SELECT {} FROM agents ORDER BY name ASC",
                Self::COLS
            )) {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
            let out = match st.query_map([], Self::row) {
                Ok(it) => it.filter_map(|r| r.ok()).collect(),
                Err(_) => Vec::new(),
            };
            out
        })
    }

    pub fn get(&self, name: &str) -> Option<AgentDef> {
        if name.is_empty() {
            return None;
        }
        self.store.with(|c| {
            c.query_row(
                &format!("SELECT {} FROM agents WHERE name = ?1", Self::COLS),
                params![name],
                Self::row,
            )
            .optional()
            .ok()
            .flatten()
        })
    }

    pub fn count(&self) -> usize {
        self.store.with(|c| {
            c.query_row("SELECT COUNT(*) FROM agents", [], |r| r.get::<_, i64>(0))
                .unwrap_or(0) as usize
        })
    }

    pub fn save(&self, def: &AgentDef, previous_name: Option<&str>) -> Result<AgentDef, AgentError> {
        let name = def.name.trim().to_string();
        if name.is_empty() {
            return Err(AgentError::EmptyName);
        }
        if !valid_name(&name) {
            return Err(AgentError::BadName(name));
        }
        let containerfile = def.containerfile.trim().to_string();
        if !containerfile.is_empty() && !std::path::Path::new(&containerfile).is_absolute() {
            return Err(AgentError::BadContainerfile(containerfile));
        }
        let engagement = Engagement::parse(&def.engagement).as_str().to_string();
        let accent = def
            .accent_color
            .as_deref()
            .and_then(crate::agent_loader::normalize_hex);
        let icon = def
            .icon
            .as_deref()
            .and_then(crate::agent_loader::normalize_icon);
        let now = now_iso();

        self.store.with(|c| {
            let existing: Option<i64> = c
                .query_row("SELECT id FROM agents WHERE name = ?1", params![name], |r| {
                    r.get(0)
                })
                .optional()
                .map_err(|e| AgentError::Db(e.to_string()))?;
            let target: Option<i64> = match previous_name {
                None => {
                    if existing.is_some() {
                        return Err(AgentError::Duplicate(name.clone()));
                    }
                    None
                }
                Some(prev) => {
                    let found: Option<i64> = c
                        .query_row("SELECT id FROM agents WHERE name = ?1", params![prev], |r| {
                            r.get(0)
                        })
                        .optional()
                        .map_err(|e| AgentError::Db(e.to_string()))?;
                    let Some(t) = found else {
                        return Err(AgentError::NotFound(prev.to_string()));
                    };
                    if let Some(other) = existing {
                        if other != t {
                            return Err(AgentError::Duplicate(name.clone()));
                        }
                    }
                    Some(t)
                }
            };

            let tools = def
                .tools_enabled
                .as_ref()
                .map(|v| json_text(&json!(v)));
            let env: Map<String, Value> = def
                .env
                .iter()
                .map(|(k, v)| (k.clone(), json!(v)))
                .collect();

            let id = match target {
                Some(id) => {
                    c.execute(
                        "UPDATE agents SET
                           name = ?1, description = ?2, model = ?3, models = ?4,
                           context_window = ?5, env = ?6, mcp = ?7, allowlist = ?8,
                           tools_enabled = ?9, policy_mode = ?10, parallel_tools = ?11,
                           soul = ?12, goals = ?13, memory = ?14, engagement = ?15,
                           starters = ?16, avatar = ?17, accent_color = ?18,
                           updated_at = ?19
                         WHERE id = ?20",
                        params![
                            name,
                            def.description,
                            def.model,
                            json_text(&json!(def.models)),
                            def.context_window.map(|v| v as i64),
                            json_text(&Value::Object(env)),
                            json_text(&json!(def.mcp)),
                            def.allowlist,
                            tools,
                            def.policy_mode,
                            def.parallel_tools as i64,
                            def.soul,
                            def.goals,
                            def.memory,
                            engagement,
                            json_text(&json!(def.starters)),
                            def.avatar,
                            accent,
                            now,
                            id,
                        ],
                    )
                    .map_err(|e| AgentError::Db(e.to_string()))?;
                    id
                }
                None => {
                    c.execute(
                        "INSERT INTO agents
                           (name, description, model, models, context_window, env, mcp,
                            allowlist, tools_enabled, policy_mode, parallel_tools, soul, goals,
                            memory, engagement, starters, avatar, accent_color,
                            created_at, updated_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                                 ?15, ?16, ?17, ?18, ?19, ?19)",
                        params![
                            name,
                            def.description,
                            def.model,
                            json_text(&json!(def.models)),
                            def.context_window.map(|v| v as i64),
                            json_text(&Value::Object(env)),
                            json_text(&json!(def.mcp)),
                            def.allowlist,
                            tools,
                            def.policy_mode,
                            def.parallel_tools as i64,
                            def.soul,
                            def.goals,
                            def.memory,
                            engagement,
                            json_text(&json!(def.starters)),
                            def.avatar,
                            accent,
                            now,
                        ],
                    )
                    .map_err(|e| AgentError::Db(e.to_string()))?;
                    c.last_insert_rowid()
                }
            };
            c.execute(
                "UPDATE agents SET reasoning_effort = ?1, max_turns = ?2, icon = ?3, \
                 containerfile = ?4 WHERE id = ?5",
                params![
                    def.reasoning_effort,
                    def.max_turns.map(|v| v as i64),
                    icon,
                    containerfile,
                    id,
                ],
            )
            .map_err(|e| AgentError::Db(e.to_string()))?;
            if let Some(prev) = previous_name {
                if prev != name {
                    let _ = c.execute(
                        "UPDATE participants SET agent_name = ?1 WHERE agent_name = ?2",
                        params![name, prev],
                    );
                    let _ = c.execute(
                        "UPDATE conversation SET agent = ?1 WHERE agent = ?2",
                        params![name, prev],
                    );
                }
            }
            Ok(id)
        })?;
        self.get(&name).ok_or(AgentError::NotFound(name))
    }

    pub fn delete(&self, name: &str) -> Result<usize, AgentError> {
        let seated = self.store.with(|c| {
            let n = c
                .execute("DELETE FROM agents WHERE name = ?1", params![name])
                .unwrap_or(0);
            if n == 0 {
                return Err(AgentError::NotFound(name.to_string()));
            }
            let seated: i64 = c
                .query_row(
                    "SELECT COUNT(*) FROM participants WHERE agent_name = ?1",
                    params![name],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            Ok(seated as usize)
        })?;
        Ok(seated)
    }

    fn import_done(&self) -> bool {
        self.store.with(|c| {
            c.query_row(
                "SELECT value FROM schema_meta WHERE key = 'agents_imported'",
                [],
                |r| r.get::<_, String>(0),
            )
            .is_ok()
        })
    }

    fn mark_import_done(&self) {
        self.store.with(|c| {
            let _ = c.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('agents_imported', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                rusqlite::params![now_iso()],
            );
        });
    }

    pub fn import_files(&self, project_root: &std::path::Path) -> usize {
        if self.import_done() {
            return 0;
        }
        if self.count() > 0 {
            self.mark_import_done();
            return 0;
        }
        let mut n = 0;
        for def in crate::agent_loader::list_files(project_root) {
            if self.get(&def.name).is_some() {
                continue;
            }
            if self.save(&def, None).is_ok() {
                n += 1;
            }
        }
        self.mark_import_done();
        n
    }

    pub fn set_memory(&self, name: &str, memory: &str) -> bool {
        self.store.with(|c| {
            c.execute(
                "UPDATE agents SET memory = ?1, updated_at = ?2 WHERE name = ?3",
                params![memory, now_iso(), name],
            )
            .unwrap_or(0)
                > 0
        })
    }
}

impl ConvStore {
    pub fn agents(&self) -> AgentStore<'_> {
        AgentStore::new(self)
    }
}
