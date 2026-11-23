//! Agents as rows.
//!
//! An agent used to be a `<name>.json` under `.angela/agents/`. That was the
//! right shape when agents were written by hand in an editor and read once at
//! startup; it is the wrong one now that the UI creates and edits them, where
//! a file store means a read-modify-write race, a second reader that can
//! disagree with the first, and no place to put anything the agent writes
//! about itself.
//!
//! So definitions live in the `agents` table and **nowhere else**. The files
//! are imported once ([`AgentStore::import_files`]) and then neither read nor
//! written.
//! They are left on disk deliberately, as the rollback: nothing in this
//! module can lose a definition that a file still holds.
//!
//! Two things are worth knowing before changing anything here:
//!
//! * **The system prompt is composed, not stored.** `soul` + `goals` +
//!   `memory` concatenate in [`AgentDef::system_prompt`]. Keeping them apart
//!   is what lets an agent rewrite its own memory without being able to
//!   rewrite its own character.
//! * **`name` is the identity.** Sessions, participants and the API all refer
//!   to an agent by name, so a rename has to carry the roster with it (see
//!   [`AgentStore::save`]).

use rusqlite::{params, OptionalExtension};
use serde_json::{json, Map, Value};

use crate::agent_loader::{AgentDef, Engagement};
use crate::conv_store::ConvStore;
use crate::session_store::now_iso;

/// What went wrong saving a definition — all of them things the UI should
/// say out loud rather than swallow.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentError {
    EmptyName,
    /// A name that is not usable as an `@handle` or a lookup key.
    BadName(String),
    Duplicate(String),
    NotFound(String),
    /// A Containerfile that is not an absolute host path (PLAN55 D1). Refused
    /// rather than cleared: silently dropping it would leave the agent running
    /// on the host when its author asked for a sandbox.
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

/// Names reach `@mention` parsing, participant handles and URL paths, so the
/// character set is the intersection of what all three accept.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Agent definitions in the conversation database.
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

    const COLS: &'static str = "id, name, description, model, models, context_window, cwd, env, \
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
            cwd: r.get(6)?,
            env: r
                .get::<_, Option<String>>(7)?
                .and_then(|t| serde_json::from_str(&t).ok())
                .unwrap_or_default(),
            mcp: parse_list(r.get(8)?),
            allowlist: r.get(9)?,
            // NULL is "every tool", which is not the same as an empty list.
            tools_enabled: r.get::<_, Option<String>>(10)?.and_then(|t| {
                let v: Vec<String> = serde_json::from_str(&t).ok()?;
                if v.is_empty() {
                    None
                } else {
                    Some(v)
                }
            }),
            policy_mode: r.get(11)?,
            parallel_tools: r.get::<_, i64>(12)? != 0,
            soul: r.get(13)?,
            goals: r.get(14)?,
            memory: r.get(15)?,
            engagement: r.get(16)?,
            starters: r
                .get::<_, Option<String>>(17)?
                .and_then(|t| serde_json::from_str(&t).ok())
                .unwrap_or_default(),
            avatar: r.get(18)?,
            accent_color: r.get(19)?,
            reasoning_effort: r.get(20)?,
            max_turns: r.get::<_, Option<i64>>(21)?.map(|v| v as u32),
            icon: r.get(22)?,
            containerfile: r.get::<_, String>(23).unwrap_or_default(),
            file: None,
        })
    }

    /// Every definition, by name.
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

    /// Create or update one definition.
    ///
    /// `previous_name` is the row to replace; `None` creates. A rename
    /// carries the roster with it, because participants refer to an agent by
    /// name and leaving them pointing at a name that no longer exists would
    /// silently drop every seated copy back to the built-in default.
    pub fn save(&self, def: &AgentDef, previous_name: Option<&str>) -> Result<AgentDef, AgentError> {
        let name = def.name.trim().to_string();
        if name.is_empty() {
            return Err(AgentError::EmptyName);
        }
        if !valid_name(&name) {
            return Err(AgentError::BadName(name));
        }
        // PLAN55 D1: a Containerfile is an absolute host path or it is nothing.
        // A relative path would silently name the wrong image (its parent
        // directory is the image name), so it is refused here rather than
        // mis-resolved at launch time.
        let containerfile = def.containerfile.trim().to_string();
        if !containerfile.is_empty() && !std::path::Path::new(&containerfile).is_absolute() {
            return Err(AgentError::BadContainerfile(containerfile));
        }
        let engagement = Engagement::parse(&def.engagement).as_str().to_string();
        // Normalised on the way in, so nothing downstream has to wonder
        // whether a stored colour is safe to put in a style attribute.
        let accent = def
            .accent_color
            .as_deref()
            .and_then(crate::agent_loader::normalize_hex);
        // Same reasoning as the accent: it is interpolated into a `class`
        // attribute, so it is normalised on the way in rather than trusted
        // on the way out.
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
            // `previous_name: None` is a *create*. A taken name is then a
            // conflict, not an invitation to overwrite somebody's agent —
            // which is exactly what a bare upsert here would do.
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
                    // Renaming onto a name someone else already holds.
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
                           context_window = ?5, cwd = ?6, env = ?7, mcp = ?8, allowlist = ?9,
                           tools_enabled = ?10, policy_mode = ?11, parallel_tools = ?12,
                           soul = ?13, goals = ?14, memory = ?15, engagement = ?16,
                           starters = ?17, avatar = ?18, accent_color = ?19,
                           updated_at = ?20
                         WHERE id = ?21",
                        params![
                            name,
                            def.description,
                            def.model,
                            json_text(&json!(def.models)),
                            def.context_window.map(|v| v as i64),
                            def.cwd,
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
                           (name, description, model, models, context_window, cwd, env, mcp,
                            allowlist, tools_enabled, policy_mode, parallel_tools, soul, goals,
                            memory, engagement, starters, avatar, accent_color,
                            created_at, updated_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                                 ?15, ?16, ?17, ?18, ?19, ?20, ?20)",
                        params![
                            name,
                            def.description,
                            def.model,
                            json_text(&json!(def.models)),
                            def.context_window.map(|v| v as i64),
                            def.cwd,
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
            // These columns were all added after the original INSERT was
            // written; setting them here beats renumbering twenty-one
            // positional parameters for each one.
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
            // Carry the roster and any session headers across a rename.
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

    /// Remove a definition.
    ///
    /// Participants that were running it keep their seat and their messages —
    /// the transcript is a record of what happened, and deleting a definition
    /// does not un-say what it said. They fall back to the built-in default
    /// if the conversation runs again.
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

    /// Has the one-shot file import already run?
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

    /// Bring `.angela/agents/*.json` in — **once per project, ever**.
    ///
    /// "Once" is recorded in `schema_meta`, not inferred from whether each
    /// name is present. Skipping per name looks the same on a fresh start
    /// and is not: delete an agent in the UI and the next restart imports it
    /// straight back from the file it came from, which is exactly what
    /// happened. A deletion has to stick.
    ///
    /// The files are never modified or removed — they remain the rollback,
    /// and are re-imported if `conversations.db` is deleted.
    pub fn import_files(&self, project_root: &std::path::Path) -> usize {
        if self.import_done() {
            return 0;
        }
        // An install that already has agents has already been through this,
        // from before the flag existed. Record that rather than importing
        // over whatever the user has since changed or deleted.
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

    /// Replace an agent's memory — the one field it may write about itself.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> ConvStore {
        ConvStore::memory("/tmp/p").unwrap()
    }

    fn def(name: &str) -> AgentDef {
        AgentDef {
            name: name.into(),
            description: "a test agent".into(),
            soul: "You are a test.".into(),
            mcp: vec!["file-io".into()],
            ..Default::default()
        }
    }

    #[test]
    fn a_definition_round_trips_through_the_database() {
        let s = store();
        let a = s.agents();
        let mut d = def("scout");
        d.goals = "Find things.".into();
        d.memory = "The repo is Rust.".into();
        d.engagement = "relevant".into();
        d.reasoning_effort = "xhigh".into();
        d.max_turns = Some(12);
        // Typed with a prefix, as it is written on the Phosphor site:
        // stored normalised, so nothing downstream has to strip it.
        d.icon = Some("ph-dog".into());
        d.models = vec!["m1".into(), "m2".into()];
        d.context_window = Some(8192);
        d.tools_enabled = Some(vec!["file_io__read_file".into()]);
        d.env.insert("K".into(), "V".into());
        let saved = a.save(&d, None).unwrap();

        let got = a.get("scout").unwrap();
        assert_eq!(got.name, "scout");
        assert_eq!(got.goals, "Find things.");
        assert_eq!(got.memory, "The repo is Rust.");
        assert_eq!(got.engagement, "relevant");
        assert_eq!(got.reasoning_effort, "xhigh");
        assert_eq!(got.max_turns, Some(12));
        assert_eq!(got.icon.as_deref(), Some("dog"));
        assert_eq!(got.models, vec!["m1", "m2"]);
        assert_eq!(got.context_window, Some(8192));
        assert_eq!(got.mcp, vec!["file-io"]);
        assert_eq!(got.tools_enabled, Some(vec!["file_io__read_file".into()]));
        assert_eq!(got.env.get("K").map(String::as_str), Some("V"));
        assert_eq!(got.id, saved.id);
        assert!(got.file.is_none(), "a row has no file");
    }

    #[test]
    fn the_system_prompt_is_soul_then_goals_then_memory() {
        let mut d = def("x");
        d.soul = "You are Scout.".into();
        d.goals = "Keep the build green.".into();
        d.memory = "CI runs on push.".into();
        let p = d.system_prompt();
        let soul = p.find("You are Scout.").unwrap();
        let goals = p.find("Keep the build green.").unwrap();
        let memory = p.find("CI runs on push.").unwrap();
        assert!(soul < goals && goals < memory, "{p}");
        // Headed, so the agent can tell its standing instructions from a note
        // it left itself — it may rewrite only the latter.
        assert!(p.contains("## Goals"));
        assert!(p.contains("## Memory"));
    }

    #[test]
    fn empty_sections_leave_no_empty_headings() {
        let mut d = def("x");
        d.soul = "Just a soul.".into();
        assert_eq!(d.system_prompt(), "Just a soul.");
        d.soul = String::new();
        d.goals = "only goals".into();
        assert_eq!(d.system_prompt(), "## Goals\n\nonly goals");
    }

    #[test]
    fn a_name_has_to_be_usable_as_a_handle() {
        let s = store();
        let a = s.agents();
        assert!(matches!(a.save(&def(""), None), Err(AgentError::EmptyName)));
        assert!(matches!(
            a.save(&def("has space"), None),
            Err(AgentError::BadName(_))
        ));
        assert!(matches!(
            a.save(&def("../etc"), None),
            Err(AgentError::BadName(_))
        ));
        assert!(a.save(&def("ok-name_2"), None).is_ok());
    }

    #[test]
    fn a_duplicate_name_is_refused_rather_than_silently_merged() {
        let s = store();
        let a = s.agents();
        a.save(&def("scout"), None).unwrap();
        a.save(&def("ranger"), None).unwrap();
        // Creating a second "scout".
        assert!(matches!(
            a.save(&def("scout"), None),
            Err(AgentError::Duplicate(_))
        ));
        // Updating in place is fine; creating a second is not.
        assert!(a.save(&def("scout"), Some("scout")).is_ok());
        // Renaming "ranger" onto the taken name.
        let mut renamed = def("scout");
        assert!(matches!(
            a.save(&renamed, Some("ranger")),
            Err(AgentError::Duplicate(_))
        ));
        renamed.name = "pathfinder".into();
        assert!(a.save(&renamed, Some("ranger")).is_ok());
        assert_eq!(a.list().len(), 2);
        // Editing an agent that is not there is a miss, not a quiet create.
        assert!(matches!(
            a.save(&def("ghost"), Some("ghost")),
            Err(AgentError::NotFound(_))
        ));
    }

    #[test]
    fn saving_over_the_same_name_updates_rather_than_duplicates() {
        let s = store();
        let a = s.agents();
        let first = a.save(&def("scout"), None).unwrap();
        let mut d = def("scout");
        d.description = "changed".into();
        let second = a.save(&d, Some("scout")).unwrap();
        assert_eq!(first.id, second.id);
        assert_eq!(second.description, "changed");
        assert_eq!(a.count(), 1);
    }

    #[test]
    fn a_rename_carries_the_roster_with_it() {
        let s = store();
        let a = s.agents();
        a.save(&def("scout"), None).unwrap();
        let conv = s
            .create(crate::session_store::NewSession {
                agent: Some("scout".into()),
                ..Default::default()
            })
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        s.add_participant(&conv, "agent", "scout", Some("Scout"), Some("scout"), None, "", None, None);

        let mut d = def("pathfinder");
        a.save(&d, Some("scout")).unwrap();
        d.name = "pathfinder".into();

        // A seat left pointing at a name nobody holds would quietly fall back
        // to the built-in default — a different agent, same conversation.
        let seated = s.participants(&conv);
        assert!(seated
            .iter()
            .any(|p| p.agent_name.as_deref() == Some("pathfinder")));
        assert!(!seated
            .iter()
            .any(|p| p.agent_name.as_deref() == Some("scout")));
        assert_eq!(s.load(&conv).unwrap()["agent"], "pathfinder");
    }

    #[test]
    fn deleting_a_definition_leaves_what_it_already_said_alone() {
        let s = store();
        let a = s.agents();
        a.save(&def("scout"), None).unwrap();
        let conv = s
            .create(crate::session_store::NewSession::default())
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        let p = s
            .add_participant(&conv, "agent", "scout", Some("Scout"), Some("scout"), None, "", None, None)
            .unwrap();
        s.append_event(
            &conv,
            crate::session_store::NewEvent {
                event_type: "context_window".into(),
                payload: json!({ "role": "assistant", "content": "I said this" }),
                speaker: Some(p.id),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(a.delete("scout").unwrap(), 1, "it was seated once");
        assert!(a.get("scout").is_none());
        // The transcript is a record of what happened.
        assert!(s
            .path(&conv)
            .iter()
            .any(|n| n.content.as_deref() == Some("I said this")));
        assert!(matches!(a.delete("scout"), Err(AgentError::NotFound(_))));
    }

    #[test]
    fn a_deleted_agent_stays_deleted_across_restarts() {
        let dir = tempfile::tempdir().unwrap();
        let agents = dir.path().join(".angela").join("agents");
        std::fs::create_dir_all(&agents).unwrap();
        std::fs::write(agents.join("default.json"), r#"{"description":"the shipped one"}"#)
            .unwrap();

        let s = ConvStore::memory(dir.path()).unwrap();
        let a = s.agents();
        assert_eq!(a.import_files(dir.path()), 1);
        assert!(a.get("default").is_some());

        // The user deletes it. Every later start must leave it deleted —
        // skipping per name would import it straight back from the file.
        a.delete("default").unwrap();
        for _ in 0..3 {
            assert_eq!(a.import_files(dir.path()), 0);
        }
        assert!(a.get("default").is_none(), "a deletion has to stick");
        // The file is untouched: it is still the rollback.
        assert!(agents.join("default.json").exists());
    }

    #[test]
    fn an_install_that_already_has_agents_is_not_imported_over() {
        let dir = tempfile::tempdir().unwrap();
        let agents = dir.path().join(".angela").join("agents");
        std::fs::create_dir_all(&agents).unwrap();
        std::fs::write(agents.join("default.json"), r#"{"description":"the shipped one"}"#)
            .unwrap();

        // A database from before the one-shot flag existed, with agents in
        // it and the shipped one already deleted.
        let s = ConvStore::memory(dir.path()).unwrap();
        let a = s.agents();
        a.save(&def("mine"), None).unwrap();
        assert_eq!(a.import_files(dir.path()), 0);
        assert!(a.get("default").is_none());
        assert_eq!(a.count(), 1);
    }

    #[test]
    fn an_unknown_engagement_falls_back_to_the_quiet_one() {
        let s = store();
        let a = s.agents();
        let mut d = def("x");
        d.engagement = "whenever-it-feels-like-it".into();
        // A garbage *value* is the quiet option — speaking when it should
        // not is more disruptive than waiting. An *absent* field is
        // different: that is a definition that predates the field, and it
        // keeps the behaviour it had (see the import test).
        assert_eq!(a.save(&d, None).unwrap().engagement, "invoked");
        assert_eq!(Engagement::parse("RELEVANT"), Engagement::Relevant);
        assert_eq!(Engagement::parse("always"), Engagement::Engaged);
    }

    #[test]
    fn memory_is_writable_on_its_own() {
        let s = store();
        let a = s.agents();
        let mut d = def("scout");
        d.soul = "unchanged".into();
        a.save(&d, None).unwrap();
        assert!(a.set_memory("scout", "learned something"));
        let got = a.get("scout").unwrap();
        assert_eq!(got.memory, "learned something");
        // Writing memory must not disturb who the agent is.
        assert_eq!(got.soul, "unchanged");
    }

    #[test]
    fn files_are_imported_once_and_read_in_full() {
        let dir = tempfile::tempdir().unwrap();
        let agents = dir.path().join(".angela").join("agents");
        std::fs::create_dir_all(&agents).unwrap();
        std::fs::write(
            agents.join("filey.json"),
            r#"{"description":"from a file","system":"old field name","mcp":["file-io"]}"#,
        )
        .unwrap();
        std::fs::write(agents.join("second.json"), r#"{"description":"also a file"}"#).unwrap();

        let s = ConvStore::memory(dir.path()).unwrap();
        let a = s.agents();
        assert_eq!(a.import_files(dir.path()), 2);

        // `system` is the file-format name for `soul`.
        assert_eq!(a.get("filey").unwrap().soul, "old field name");
        assert_eq!(a.get("filey").unwrap().mcp, vec!["file-io"]);
        // The file predates `engagement`; muting it on import would change
        // how an existing project behaves.
        assert_eq!(a.get("filey").unwrap().engagement, "engaged");

        // Once, ever. An edit made afterwards is never overwritten by the
        // file it came from, and a deletion is never undone.
        let mut edited = a.get("filey").unwrap();
        edited.description = "the edited version".into();
        a.save(&edited, Some("filey")).unwrap();
        a.delete("second").unwrap();

        assert_eq!(a.import_files(dir.path()), 0);
        assert_eq!(a.get("filey").unwrap().description, "the edited version");
        assert!(a.get("second").is_none());
        // And the files are left exactly where they were.
        assert!(agents.join("filey.json").exists());
        assert!(agents.join("second.json").exists());
    }

}
