use rusqlite::params;
use serde_json::{json, Value};

use crate::conv_store::ConvStore;
use crate::session_store::now_iso;

pub const MAX_NAME: usize = 200;
pub const MAX_DEPTH: usize = 8;

pub fn normalize(raw: &str) -> Result<String, String> {
    let parts: Vec<&str> = raw
        .split('/')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        return Err("a project needs a name".into());
    }
    if parts.len() > MAX_DEPTH {
        return Err(format!("a project may be at most {MAX_DEPTH} levels deep"));
    }
    if parts.iter().any(|p| *p == "." || *p == "..") {
        return Err("a project name cannot contain . or .. segments".into());
    }
    if parts.iter().any(|p| p.chars().any(char::is_control)) {
        return Err("a project name cannot contain control characters".into());
    }
    let name = parts.join("/");
    if name.len() > MAX_NAME {
        return Err(format!("a project name may be at most {MAX_NAME} characters"));
    }
    Ok(name)
}

fn is_descendant(to: &str, from: &str) -> bool {
    to == from || to.starts_with(&format!("{from}/"))
}

pub const INHERITABLE: &[&str] = &["purpose", "notes", "routing", "memoryHigh", "workspaceRoot", "worktree", "soulAppendage", "defaultAgents"];

pub fn ancestors(raw: &str) -> Vec<String> {
    let Ok(name) = normalize(raw) else { return Vec::new() };
    let mut out = Vec::new();
    let mut path = String::new();
    for part in name.split('/') {
        if !path.is_empty() { path.push('/'); }
        path.push_str(part);
        out.push(path.clone());
    }
    out
}

fn present(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::String(s) => !s.trim().is_empty(),
        Value::Object(m) => !m.is_empty(),
        _ => true,
    }
}

impl ConvStore {
    pub fn project_props(&self, name: &str) -> Value {
        self.with(|c| c.query_row("SELECT settings FROM projects WHERE name = ?1", params![name], |r| r.get::<_, String>(0))
            .ok().and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .filter(Value::is_object).unwrap_or_else(|| json!({})))
    }

    pub fn set_project_props(&self, raw: &str, patch: &Value) -> Result<Value, String> {
        let name = normalize(raw)?;
        let patch = patch.as_object().ok_or("settings must be an object")?;
        if patch.keys().any(|k| !INHERITABLE.contains(&k.as_str())) {
            return Err("unknown project setting".into());
        }
        let mut props = self.project_props(&name);
        let obj = props.as_object_mut().unwrap();
        for (key, value) in patch {
            if present(value) { obj.insert(key.clone(), value.clone()); } else { obj.remove(key); }
        }
        let encoded = serde_json::to_string(&props).map_err(|e| e.to_string())?;
        self.with(|c| c.execute("INSERT INTO projects (name, created_at, settings) VALUES (?1, ?2, ?3) ON CONFLICT(name) DO UPDATE SET settings=excluded.settings", params![name, now_iso(), encoded]).map_err(|e| e.to_string()))?;
        Ok(props)
    }

    pub fn effective_session(&self, id: &str) -> Option<Value> {
        let session = self.load(id)?;
        let mut inherited = serde_json::Map::new();
        let mut chain = Vec::new();
        for name in ancestors(session.get("project").and_then(Value::as_str).unwrap_or("")) {
            let props = self.project_props(&name);
            if let Some(obj) = props.as_object() {
                for (key, value) in obj {
                    if present(value) { inherited.insert(key.clone(), value.clone()); }
                }
                if !obj.is_empty() { chain.push(json!({"name": name, "props": props})); }
            }
        }
        let mut effective = session.clone();
        let obj = effective.as_object_mut()?;
        for (key, value) in &inherited {
            if !obj.get(key).is_some_and(present) { obj.insert(key.clone(), value.clone()); }
        }
        Some(json!({"session": session, "effective": effective, "inherited": inherited, "chain": chain}))
    }

    pub fn project_default_agents(&self, project: &str) -> Result<Vec<String>, String> {
        let mut names = Vec::new();
        for ancestor in ancestors(project) {
            let props = self.project_props(&ancestor);
            if let Some(value) = props.get("defaultAgents") {
                let list = value.as_array().ok_or("invalid default agents")?;
                names = list.iter().map(|v| v.as_str().filter(|s| !s.is_empty()).map(str::to_string).ok_or_else(|| "invalid default agent name".to_string())).collect::<Result<Vec<_>, _>>()?;
            }
        }
        Ok(names)
    }

    pub fn inherited_worktree(&self, id: &str) -> Option<Value> {
        self.with(|c| c.query_row("SELECT record FROM inherited_worktree WHERE session_id = ?1", params![id], |r| r.get::<_, String>(0)).ok().and_then(|s| serde_json::from_str(&s).ok()))
    }

    pub fn set_inherited_worktree(&self, id: &str, record: &Value) -> Result<(), String> {
        let encoded = serde_json::to_string(record).map_err(|e| e.to_string())?;
        self.with(|c| c.execute("INSERT INTO inherited_worktree(session_id, record) VALUES (?1, ?2) ON CONFLICT(session_id) DO UPDATE SET record=excluded.record", params![id, encoded]).map(|_| ()).map_err(|e| e.to_string()))
    }

    pub fn projects(&self) -> Vec<String> {
        self.with(|c| {
            let mut names: Vec<String> = Vec::new();
            let mut push = |n: String| {
                let n = n.trim().to_string();
                if !n.is_empty() && !names.contains(&n) {
                    names.push(n);
                }
            };
            if let Ok(mut st) = c.prepare("SELECT name FROM projects") {
                if let Ok(rows) = st.query_map([], |r| r.get::<_, String>(0)) {
                    for n in rows.flatten() {
                        push(n);
                    }
                }
            }
            if let Ok(mut st) = c.prepare(
                "SELECT DISTINCT project FROM conversation WHERE project IS NOT NULL AND project <> ''",
            ) {
                if let Ok(rows) = st.query_map([], |r| r.get::<_, String>(0)) {
                    for n in rows.flatten() {
                        push(n);
                    }
                }
            }
            names.sort();
            names
        })
    }

    pub fn projects_detail(&self) -> Value {
        let names = self.projects();
        let out: Vec<Value> = names
            .iter()
            .map(|name| {
                let n: i64 = self
                    .with(|c| {
                        c.query_row(
                            "SELECT COUNT(*) FROM conversation WHERE project = ?1",
                            params![name],
                            |r| r.get::<_, i64>(0),
                        )
                        .unwrap_or(0)
                    });
                json!({ "id": name, "name": name, "icon": "brackets-angle", "count": n, "props": self.project_props(name) })
            })
            .collect();
        json!({ "projects": out })
    }

    pub fn create_project(&self, raw: &str) -> Result<String, String> {
        let name = normalize(raw)?;
        let now = now_iso();
        self.with(|c| {
            let _ = c.execute(
                "INSERT INTO projects (name, created_at) VALUES (?1, ?2)
                 ON CONFLICT(name) DO NOTHING",
                params![name, now],
            );
        });
        Ok(name)
    }

    pub fn rename_project(&self, from: &str, to: &str) -> Result<String, String> {
        let from = normalize(from)?;
        let to = normalize(to)?;
        if from == to {
            return Ok(to);
        }
        if is_descendant(&to, &from) {
            return Err(format!("cannot move {from} inside itself"));
        }
        let now = now_iso();
        let like = format!("{from}/%");
        let cut = from.len() as i64 + 1;
        self.with(|c| {
            let _ = c.execute(
                "INSERT INTO projects (name, created_at, settings) SELECT ?1, ?2, settings FROM projects WHERE name = ?3 ON CONFLICT(name) DO UPDATE SET settings=excluded.settings",
                params![to, now, from],
            );
            let _ = c.execute(
                "UPDATE OR REPLACE projects SET name = ?1 || substr(name, ?2) WHERE name LIKE ?3",
                params![to, cut, like],
            );
            let _ = c.execute("DELETE FROM projects WHERE name = ?1", params![from]);
            let _ = c.execute(
                "UPDATE conversation SET project = ?1 WHERE project = ?2",
                params![to, from],
            );
            let _ = c.execute(
                "UPDATE conversation SET project = ?1 || substr(project, ?2) WHERE project LIKE ?3",
                params![to, cut, like],
            );
        });
        Ok(to)
    }

    pub fn delete_project(&self, raw: &str) -> Result<usize, String> {
        let name = normalize(raw)?;
        let like = format!("{name}/%");
        Ok(self.with(|c| {
            let _ = c.execute(
                "DELETE FROM projects WHERE name = ?1 OR name LIKE ?2",
                params![name, like],
            );
            c.execute(
                "UPDATE conversation SET project = NULL WHERE project = ?1 OR project LIKE ?2",
                params![name, like],
            )
            .unwrap_or(0)
        }))
    }

    pub fn assign_project(&self, id: &str, raw: &str) -> Result<String, String> {
        let name = if raw.trim().is_empty() {
            String::new()
        } else {
            self.create_project(raw)?
        };
        let changed = self.with(|c| {
            c.execute(
                "UPDATE conversation SET project = ?1, updated_at = ?2 WHERE id = ?3",
                params![
                    if name.is_empty() { None } else { Some(name.as_str()) },
                    now_iso(),
                    id
                ],
            )
            .unwrap_or(0)
        });
        if changed == 0 {
            return Err(format!("no such conversation: {id}"));
        }
        Ok(name)
    }
}
