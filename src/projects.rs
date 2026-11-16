//! Projects — the folders conversations are filed under.
//!
//! A project is a **name**, and a conversation's membership is the same name
//! written on `conversation.project`. There is no id, no parent pointer and
//! no tree: the hierarchy is read out of the `/` in the name, exactly as a
//! filesystem path implies its directories.
//!
//! ```text
//!   psicom/chat     ─┐
//!   psicom/sbox      ├─ two projects, three folders: psicom, chat, sbox
//!   scratch         ─┘
//! ```
//!
//! **Why names and not rows with ids.** Every operation the UI offers is
//! then a string operation. Renaming a folder — including one with children
//! — is one `UPDATE ... WHERE project = ? OR project LIKE ?`. Moving a
//! conversation between folders is one assignment. With ids and a parent
//! column, each of those is a tree walk that has to stay consistent with the
//! labels, and the labels are what the user actually sees and types.
//!
//! The cost is that a name is the identity, so a rename rewrites every
//! conversation in the subtree. That is a single statement over a table with
//! one row per conversation, which is the right trade at this size.
//!
//! **What the table is for.** [`ConvStore::projects`] merges two sources: the
//! `projects` table and the distinct labels actually in use. The table alone
//! would miss a conversation whose label was set directly; the labels alone
//! would lose any folder that is currently empty — and creating a folder
//! before putting anything in it, or dragging the last conversation out of
//! one, are both ordinary gestures. Neither source is sufficient; the union
//! is.

use rusqlite::params;
use serde_json::{json, Value};

use crate::conv_store::ConvStore;
use crate::session_store::now_iso;

/// Longest name we will store. Generous for a path-ish label, short enough
/// that it cannot be used to stuff the column.
pub const MAX_NAME: usize = 200;
/// How deep the implied hierarchy may go.
pub const MAX_DEPTH: usize = 8;

/// Tidy a name into its canonical form, or reject it.
///
/// Collapses repeated and stray separators so `  psicom / / chat/ ` and
/// `psicom/chat` are the same folder rather than two that look identical in
/// the tree. A name that normalises to nothing is not a name.
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
    // `.` and `..` read as path traversal to anyone looking at the tree, and
    // the name is shown as a path. Nothing here touches the filesystem, so
    // this is about legibility rather than safety — but a folder called `..`
    // is a folder nobody can reason about.
    if parts.iter().any(|p| *p == "." || *p == "..") {
        return Err("a project name cannot contain . or .. segments".into());
    }
    // Control characters would render as nothing and make two different
    // names look like one.
    if parts.iter().any(|p| p.chars().any(char::is_control)) {
        return Err("a project name cannot contain control characters".into());
    }
    let name = parts.join("/");
    if name.len() > MAX_NAME {
        return Err(format!("a project name may be at most {MAX_NAME} characters"));
    }
    Ok(name)
}

/// Would renaming `from` to `to` put a folder inside itself?
///
/// `a` → `a/b` is the move that destroys the subtree: every child matches the
/// `LIKE` and gets rewritten, then matches again under its new prefix. The
/// UI can offer the drag; this is where it has to be refused.
fn is_descendant(to: &str, from: &str) -> bool {
    to == from || to.starts_with(&format!("{from}/"))
}

impl ConvStore {
    /// Every folder that exists: registered, in use, or both.
    ///
    /// Sorted, so the tree the UI builds is stable between reads — an
    /// unsorted listing would reshuffle siblings on every poll.
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

    /// The folders, each with how many conversations are filed directly in
    /// it — what the sidebar's counters show.
    ///
    /// Directly: a conversation in `psicom/chat` counts for `psicom/chat`
    /// and not for `psicom`, because the tree shows `psicom`'s own children
    /// beneath it and counting them twice would double every total.
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
                json!({ "id": name, "name": name, "icon": "brackets-angle", "count": n })
            })
            .collect();
        json!({ "projects": out })
    }

    /// Register a folder. Idempotent: creating one that already exists is
    /// success, because the caller's intent — "there should be a folder
    /// called this" — is already satisfied.
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

    /// Rename a folder and everything beneath it.
    ///
    /// Both halves move: the registry rows and the conversation labels. A
    /// rename that moved only one would split the folder in two — the old
    /// name still listed because conversations point at it, the new one
    /// listed because it is registered.
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
        // `substr(name, len+1)` keeps each descendant's tail and swaps only
        // the prefix, so `a/b/c` under `a` → `x` becomes `x/b/c`.
        let cut = from.len() as i64 + 1;
        self.with(|c| {
            let _ = c.execute(
                "INSERT INTO projects (name, created_at) VALUES (?1, ?2)
                 ON CONFLICT(name) DO NOTHING",
                params![to, now],
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

    /// Forget a folder. Its conversations are **unfiled**, not deleted —
    /// deleting a folder in a file manager does delete what is in it, but a
    /// conversation is not a file in this folder, it merely carries its
    /// name, and losing work to a tidy-up is not a trade worth offering.
    ///
    /// Descendants go too: a folder cannot survive its parent, or the tree
    /// would show it rooted at a name that no longer exists.
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

    /// File a conversation under a folder — the drag-and-drop in the
    /// sidebar, and the composer's project picker.
    ///
    /// An empty target unfiles it. The folder is registered on the way
    /// through, so dropping onto a name that was only ever a label makes it
    /// a real folder rather than leaving it to vanish later.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_store::NewSession;

    fn store() -> ConvStore {
        ConvStore::memory("/tmp/proj").unwrap()
    }

    fn conv(s: &ConvStore, title: &str) -> String {
        s.create(NewSession {
            title: Some(title.into()),
            ..Default::default()
        })
        .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn a_name_is_tidied_before_it_becomes_an_identity() {
        assert_eq!(normalize("  psicom / chat/ ").unwrap(), "psicom/chat");
        assert_eq!(normalize("//a//b//").unwrap(), "a/b");
        assert!(normalize("   ").is_err());
        assert!(normalize("a/../b").is_err());
    }

    #[test]
    fn an_empty_folder_survives_having_nothing_in_it() {
        let s = store();
        s.create_project("scratch").unwrap();
        // The whole reason the table exists: no conversation points here.
        assert!(s.projects().contains(&"scratch".to_string()));
    }

    #[test]
    fn a_label_nobody_registered_is_still_a_folder() {
        let s = store();
        let id = conv(&s, "one");
        s.assign_project(&id, "ad-hoc").unwrap();
        assert!(s.projects().contains(&"ad-hoc".to_string()));
    }

    #[test]
    fn renaming_a_folder_takes_its_children_and_their_conversations() {
        let s = store();
        let a = conv(&s, "a");
        let b = conv(&s, "b");
        s.assign_project(&a, "psicom").unwrap();
        s.assign_project(&b, "psicom/chat").unwrap();

        s.rename_project("psicom", "work").unwrap();

        assert_eq!(s.load(&a).unwrap()["project"], "work");
        assert_eq!(s.load(&b).unwrap()["project"], "work/chat");
        let names = s.projects();
        assert!(names.contains(&"work".to_string()));
        assert!(names.contains(&"work/chat".to_string()));
        assert!(!names.iter().any(|n| n.starts_with("psicom")));
    }

    #[test]
    fn a_folder_cannot_be_moved_inside_itself() {
        let s = store();
        s.create_project("a").unwrap();
        // Left to run, this rewrites every child under its own new prefix
        // and the subtree is gone.
        assert!(s.rename_project("a", "a/b").is_err());
        assert!(s.rename_project("a", "a").is_ok()); // a no-op, not an error
    }

    #[test]
    fn deleting_a_folder_unfiles_its_conversations_rather_than_deleting_them() {
        let s = store();
        let a = conv(&s, "a");
        let b = conv(&s, "b");
        s.assign_project(&a, "doomed").unwrap();
        s.assign_project(&b, "doomed/inner").unwrap();

        assert_eq!(s.delete_project("doomed").unwrap(), 2);

        // Still there, just unfiled.
        assert_eq!(s.load(&a).unwrap()["project"], "");
        assert_eq!(s.load(&b).unwrap()["project"], "");
        assert!(!s.projects().iter().any(|n| n.starts_with("doomed")));
    }

    #[test]
    fn an_empty_assignment_unfiles() {
        let s = store();
        let a = conv(&s, "a");
        s.assign_project(&a, "somewhere").unwrap();
        s.assign_project(&a, "").unwrap();
        assert_eq!(s.load(&a).unwrap()["project"], "");
        // The folder itself outlives the conversation leaving it.
        assert!(s.projects().contains(&"somewhere".to_string()));
    }

    #[test]
    fn assigning_to_a_conversation_that_is_not_there_is_an_error() {
        let s = store();
        assert!(s.assign_project("nope", "x").is_err());
    }

    #[test]
    fn counts_are_direct_members_only() {
        let s = store();
        let a = conv(&s, "a");
        let b = conv(&s, "b");
        s.assign_project(&a, "p").unwrap();
        s.assign_project(&b, "p/child").unwrap();
        let d = s.projects_detail();
        let rows = d["projects"].as_array().unwrap();
        let find = |n: &str| {
            rows.iter()
                .find(|r| r["name"] == n)
                .map(|r| r["count"].as_i64().unwrap())
        };
        // `p` holds one conversation of its own; `b` belongs to `p/child`,
        // which the tree draws beneath it.
        assert_eq!(find("p"), Some(1));
        assert_eq!(find("p/child"), Some(1));
    }
}
