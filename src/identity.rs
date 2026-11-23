//! Identity, devices, and the ACL — the WHO, the WHAT and the grants.
//!
//! The shape is PLAN44's three-table ACL, with one deliberate merge: the
//! `principal` table **is** the user table (a user is a principal of kind
//! `'user'`, a role a principal of kind `'role'`), because two tables for
//! one concept are two tables that drift apart.
//!
//! Three rules everything here enforces, from the plan's decisions:
//!
//! * **The admin is the first user row (id 1)** — exactly one, by id. Its
//!   powers are implicit: no capability rows, no grant rows, no checks.
//! * **Default deny.** A user with no capability rows and no grants can
//!   only take part in sessions others invite it to (a `post` grant).
//! * **Delegation is `with_grant`.** A user may re-share only privileges
//!   its own grant carries `with_grant` on — there is no separate
//!   "delegate" capability to hold.
//!
//! Session cookies live here as `device` rows: the cookie's value is never
//! stored, only its sha256 and a masked hint, so the profile can list what
//! it cannot reveal. One login on a new device = one row.

use std::sync::Mutex;

use rusqlite::Connection;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The account capabilities (Layer 1). Keep in step with PLAN44's catalog.
pub const CAPABILITIES: &[&str] = &[
    "session.create",
    "session.invite",
    "agent.create",
    "automation.create",
    "project.create",
    "sandbox.create",
];

/// The per-resource privileges (Layer 2).
pub const PRIVILEGES: &[&str] = &[
    "read", "post", "steer", "seat", "edit_meta", "edit", "grant", "delete",
];

/// The user-level allowlist kinds (Layer 1b).
pub const ALLOWLIST_KINDS: &[&str] = &["models", "mcp", "tools", "sandboxes"];

/// One person (or role) row.
#[derive(Debug, Clone, PartialEq)]
pub struct User {
    pub id: i64,
    pub name: String,
    pub display_name: String,
    /// The user's own profile note (per-user, never global).
    pub location: String,
    pub kind: String,
    pub disabled: bool,
}

impl User {
    /// The admin is user id 1 — exactly one, by id (PLAN44 D-catalog).
    pub fn is_admin(&self) -> bool {
        self.id == 1
    }
}

/// One recognized device — which is also one issued session cookie.
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    pub id: i64,
    pub user_id: i64,
    pub token_hint: String,
    pub fingerprint: String,
    pub user_agent: String,
    pub ip: String,
    pub created_at: i64,
    pub last_seen_at: i64,
    pub revoked: bool,
}

/// The store's identity half, sharing the conversation database and its
/// lock. Every method takes the lock for one statement — never held across
/// another, same as [`crate::conv_store::ConvStore`].
pub struct Identity<'a> {
    pub(crate) conn: &'a Mutex<Connection>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// 32 random bytes as hex, from the OS. Session tokens must be
/// unguessable; there is no seeding mistake to make here.
fn random_token() -> String {
    let mut buf = [0u8; 32];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        use std::io::Read;
        let _ = f.read_exact(&mut buf);
    }
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

fn hash_token(token: &str) -> String {
    let mut h = Sha256::new();
    h.update(token.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// `ab12…9f` — enough to recognise a row in the profile list, never enough
/// to reconstruct the cookie (its value is not stored at all).
fn token_hint(token: &str) -> String {
    if token.len() <= 6 {
        return "…".into();
    }
    format!("{}…{}", &token[..4], &token[token.len() - 2..])
}

impl Identity<'_> {
    // ---- the WHO: users ---------------------------------------------------

    /// Create a user. `name` is the case-insensitive login. The first user
    /// ever created becomes the admin purely by being id 1 — nothing else
    /// about it is special, and nothing else can be.
    pub fn create_user(&self, name: &str, display_name: &str) -> Result<User, String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("a user needs a name".into());
        }
        let conn = self.conn.lock().unwrap();
        let now = now_ms();
        conn.execute(
            "INSERT INTO principal (name, display_name, kind, created_at) VALUES (?1, ?2, 'user', ?3)",
            rusqlite::params![name, display_name.trim(), now],
        )
        .map_err(|e| match e {
            rusqlite::Error::SqliteFailure(_, Some(msg)) if msg.contains("UNIQUE") => {
                "that name is taken".into()
            }
            e => e.to_string(),
        })?;
        let id = conn.last_insert_rowid();
        Ok(User {
            id,
            name: name.to_string(),
            display_name: display_name.trim().to_string(),
            location: String::new(),
            kind: "user".into(),
            disabled: false,
        })
    }

    fn user_row(conn: &Connection, where_sql: &str, arg: &dyn rusqlite::ToSql) -> Option<User> {
        conn.query_row(
            &format!(
                "SELECT id, name, display_name, location, kind, disabled FROM principal WHERE {where_sql}"
            ),
            [arg],
            |r| {
                Ok(User {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    display_name: r.get(2)?,
                    location: r.get(3)?,
                    kind: r.get(4)?,
                    disabled: r.get::<_, i64>(5)? != 0,
                })
            },
        )
        .ok()
    }

    /// By login name. Case-insensitive: the `name` column is COLLATE NOCASE.
    pub fn user_by_name(&self, name: &str) -> Option<User> {
        let conn = self.conn.lock().unwrap();
        Self::user_row(&conn, "name = ?1 AND kind = 'user'", &name.trim())
    }

    pub fn user_by_id(&self, id: i64) -> Option<User> {
        let conn = self.conn.lock().unwrap();
        Self::user_row(&conn, "id = ?1", &id)
    }

    pub fn list_users(&self) -> Vec<User> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, name, display_name, location, kind, disabled FROM principal WHERE kind = 'user' ORDER BY id")
            .unwrap();
        let rows = stmt
            .query_map([], |r| {
                Ok(User {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    display_name: r.get(2)?,
                    location: r.get(3)?,
                    kind: r.get(4)?,
                    disabled: r.get::<_, i64>(5)? != 0,
                })
            })
            .unwrap();
        rows.filter_map(Result::ok).collect()
    }

    /// Save a user's own profile fields. Only the location is writable —
    /// the name is fixed at login and changes only by re-creating the
    /// account (PLAN44).
    pub fn set_location(&self, id: i64, location: &str) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE principal SET location = ?2 WHERE id = ?1",
            rusqlite::params![id, location.trim()],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn set_disabled(&self, id: i64, disabled: bool) -> Result<(), String> {
        if id == 1 {
            return Err("the admin account cannot be disabled".into());
        }
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE principal SET disabled = ?2 WHERE id = ?1",
            rusqlite::params![id, disabled as i64],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn set_display_name(&self, id: i64, display_name: &str) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE principal SET display_name = ?2 WHERE id = ?1",
            rusqlite::params![id, display_name.trim()],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    // ---- devices (one per issued session cookie) --------------------------

    /// Mint a session cookie for a user. The plaintext token is returned
    /// **once** (it goes in the `Set-Cookie`); only its hash and hint are
    /// kept.
    pub fn issue_device(
        &self,
        user_id: i64,
        fingerprint: &str,
        user_agent: &str,
        ip: &str,
    ) -> Result<(String, Device), String> {
        let token = random_token();
        let conn = self.conn.lock().unwrap();
        let now = now_ms();
        conn.execute(
            "INSERT INTO device (user_id, token_hash, token_hint, fingerprint, user_agent, ip, created_at, last_seen_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
            rusqlite::params![
                user_id,
                hash_token(&token),
                token_hint(&token),
                fingerprint.trim(),
                user_agent.trim(),
                ip.trim(),
                now
            ],
        )
        .map_err(|e| e.to_string())?;
        let id = conn.last_insert_rowid();
        let hint = token_hint(&token);
        Ok((
            token,
            Device {
                id,
                user_id,
                token_hint: hint,
                fingerprint: fingerprint.trim().to_string(),
                user_agent: user_agent.trim().to_string(),
                ip: ip.trim().to_string(),
                created_at: now,
                last_seen_at: now,
                revoked: false,
            },
        ))
    }

    /// Resolve a session cookie to its user. `None` for unknown, revoked or
    /// disabled — the edge treats all three the same: not logged in. The
    /// row's `last_seen_at` moves forward on every successful look-up.
    pub fn device_by_token(&self, token: &str) -> Option<(User, Device)> {
        if token.trim().is_empty() {
            return None;
        }
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT d.id, d.user_id, d.token_hint, d.fingerprint, d.user_agent, d.ip,
                        d.created_at, d.last_seen_at, d.revoked_at
                 FROM device d JOIN principal p ON p.id = d.user_id
                 WHERE d.token_hash = ?1 AND d.revoked_at IS NULL AND p.disabled = 0 AND p.kind = 'user'",
                [hash_token(token.trim())],
                |r| {
                    Ok((
                        Device {
                            id: r.get(0)?,
                            user_id: r.get(1)?,
                            token_hint: r.get(2)?,
                            fingerprint: r.get(3)?,
                            user_agent: r.get(4)?,
                            ip: r.get(5)?,
                            created_at: r.get(6)?,
                            last_seen_at: r.get(7)?,
                            revoked: false,
                        },
                        r.get::<_, i64>(1)?,
                    ))
                },
            )
            .ok()?;
        let _ = conn.execute(
            "UPDATE device SET last_seen_at = ?2 WHERE id = ?1",
            rusqlite::params![row.0.id, now_ms()],
        );
        let user = Self::user_row(&conn, "id = ?1", &row.1)?;
        Some((user, row.0))
    }

    /// The devices still recognized — live cookies only, most recently
    /// seen first. A revoked row is history, not a device the profile
    /// offers to manage (PLAN44 A2).
    pub fn list_devices(&self, user_id: i64) -> Vec<Device> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, user_id, token_hint, fingerprint, user_agent, ip, created_at, last_seen_at, revoked_at
                 FROM device WHERE user_id = ?1 AND revoked_at IS NULL ORDER BY last_seen_at DESC",
            )
            .unwrap();
        let rows = stmt
            .query_map([user_id], |r| {
                Ok(Device {
                    id: r.get(0)?,
                    user_id: r.get(1)?,
                    token_hint: r.get(2)?,
                    fingerprint: r.get(3)?,
                    user_agent: r.get(4)?,
                    ip: r.get(5)?,
                    created_at: r.get(6)?,
                    last_seen_at: r.get(7)?,
                    revoked: r.get::<_, Option<i64>>(8)?.is_some(),
                })
            })
            .unwrap();
        rows.filter_map(Result::ok).collect()
    }

    /// Force-expire one device's cookie. Scoping by `user_id` means a user
    /// can only revoke their own — the admin passes 0 for "any".
    pub fn revoke_device(&self, user_id: i64, device_id: i64) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        let n = conn
            .execute(
                "UPDATE device SET revoked_at = ?3 WHERE id = ?2 AND revoked_at IS NULL
                 AND (?1 = 0 OR user_id = ?1)",
                rusqlite::params![user_id, device_id, now_ms()],
            )
            .map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("no such device".into());
        }
        Ok(())
    }

    // ---- Layer 1: account capabilities ------------------------------------

    pub fn grant_capability(&self, user_id: i64, capability: &str) -> Result<(), String> {
        if !CAPABILITIES.contains(&capability) {
            return Err(format!("unknown capability {capability:?}"));
        }
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO user_capability (user_id, capability) VALUES (?1, ?2)",
            rusqlite::params![user_id, capability],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn revoke_capability(&self, user_id: i64, capability: &str) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "DELETE FROM user_capability WHERE user_id = ?1 AND capability = ?2",
            rusqlite::params![user_id, capability],
        );
    }

    /// The admin holds every capability implicitly. Everybody else needs
    /// the row (default deny).
    pub fn has_capability(&self, user_id: i64, capability: &str) -> bool {
        if user_id == 1 {
            return true;
        }
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT 1 FROM user_capability WHERE user_id = ?1 AND capability = ?2",
            rusqlite::params![user_id, capability],
            |_| Ok(()),
        )
        .is_ok()
    }

    pub fn capabilities(&self, user_id: i64) -> Vec<String> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT capability FROM user_capability WHERE user_id = ?1 ORDER BY capability")
            .unwrap();
        let rows = stmt.query_map([user_id], |r| r.get::<_, String>(0)).unwrap();
        rows.filter_map(Result::ok).collect()
    }

    // ---- Layer 1b: user-level allowlists ----------------------------------

    /// Replace one allowlist. Empty = "nothing may be assigned" (distinct
    /// from `*` = everything); both are answers, so a missing row and an
    /// empty one mean the same: deny.
    pub fn set_allowlist(&self, user_id: i64, kind: &str, patterns: &[String]) -> Result<(), String> {
        if !ALLOWLIST_KINDS.contains(&kind) {
            return Err(format!("unknown allowlist kind {kind:?}"));
        }
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM user_allowlist WHERE user_id = ?1 AND kind = ?2",
            rusqlite::params![user_id, kind],
        )
        .map_err(|e| e.to_string())?;
        for p in patterns {
            let p = p.trim();
            if p.is_empty() {
                continue;
            }
            conn.execute(
                "INSERT OR IGNORE INTO user_allowlist (user_id, kind, pattern) VALUES (?1, ?2, ?3)",
                rusqlite::params![user_id, kind, p],
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn allowlist(&self, user_id: i64, kind: &str) -> Vec<String> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT pattern FROM user_allowlist WHERE user_id = ?1 AND kind = ?2 ORDER BY pattern")
            .unwrap();
        let rows = stmt
            .query_map(rusqlite::params![user_id, kind], |r| r.get::<_, String>(0))
            .unwrap();
        rows.filter_map(Result::ok).collect()
    }

    /// May this user put `name` (of `kind`) on an agent it creates or
    /// edits? The admin may; everybody else needs `*` or the exact name on
    /// their user-level allowlist. Restricts **editing**, never using: an
    /// agent's own tools keep working for anyone its owner invites in.
    pub fn may_assign(&self, user_id: i64, kind: &str, name: &str) -> bool {
        if user_id == 1 {
            return true;
        }
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT 1 FROM user_allowlist WHERE user_id = ?1 AND kind = ?2 AND pattern IN ('*', ?3) LIMIT 1",
            rusqlite::params![user_id, kind, name],
            |_| Ok(()),
        )
        .is_ok()
    }

    // ---- the audit log: who did what as whom ----------------------------

    /// Record an identity transition (PLAN44 D4). `acting_as` is the
    /// masquerade target when the actor took another account's face.
    pub fn audit(&self, actor: &str, acting_as: &str, action: &str, detail: &str) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "INSERT INTO audit_log (at, actor, acting_as, action, detail) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![now_ms(), actor, acting_as, action, detail],
        );
    }

    /// The newest entries first — the admin's audit view.
    pub fn audit_tail(&self, n: usize) -> Vec<Value> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT at, actor, acting_as, action, detail FROM audit_log
                 ORDER BY at DESC, id DESC LIMIT ?1",
            )
            .unwrap();
        let rows = stmt
            .query_map([n as i64], |r| {
                Ok(serde_json::json!({
                    "at": r.get::<_, i64>(0)?,
                    "actor": r.get::<_, String>(1)?,
                    "actingAs": r.get::<_, String>(2)?,
                    "action": r.get::<_, String>(3)?,
                    "detail": r.get::<_, String>(4)?,
                }))
            })
            .unwrap();
        rows.filter_map(Result::ok).collect()
    }

    // ---- Layer 2: resources and grants ------------------------------------

    /// The resource row for `(kind, name)`, created on first sight with
    /// `owner_id` as its owner. Existing entities get their row lazily —
    /// `ensure_resource` is idempotent and never moves ownership.
    pub fn ensure_resource(&self, kind: &str, name: &str, owner_id: i64) -> Result<i64, String> {
        if name.trim().is_empty() {
            return Err("a resource needs a name".into());
        }
        let conn = self.conn.lock().unwrap();
        if let Some(id) = Self::resource_id(&conn, kind, name) {
            return Ok(id);
        }
        conn.execute(
            "INSERT INTO resource (schema, name, kind, owner_id) VALUES ('chat', ?1, ?2, ?3)",
            rusqlite::params![name, kind, owner_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(conn.last_insert_rowid())
    }

    fn resource_id(conn: &Connection, kind: &str, name: &str) -> Option<i64> {
        conn.query_row(
            "SELECT id FROM resource WHERE schema = 'chat' AND kind = ?1 AND name = ?2",
            rusqlite::params![kind, name],
            |r| r.get(0),
        )
        .ok()
    }

    /// `(id, owner_id)` for a resource, if its row exists.
    pub fn resource(&self, kind: &str, name: &str) -> Option<(i64, i64)> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, owner_id FROM resource WHERE schema = 'chat' AND kind = ?1 AND name = ?2",
            rusqlite::params![kind, name],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok()
    }

    /// Grant one privilege on a resource. The grantor must be the owner, or
    /// hold `grant` **with** `with_grant` on that resource — delegation is
    /// the flag on your own grant, never a separate permission.
    #[allow(clippy::too_many_arguments)]
    pub fn grant(
        &self,
        grantee: i64,
        kind: &str,
        name: &str,
        privilege: &str,
        grantor: i64,
        with_grant: bool,
    ) -> Result<(), String> {
        if !PRIVILEGES.contains(&privilege) {
            return Err(format!("unknown privilege {privilege:?}"));
        }
        let conn = self.conn.lock().unwrap();
        let Some(id) = Self::resource_id(&conn, kind, name) else {
            return Err("no such resource".into());
        };
        let owner: i64 = conn
            .query_row("SELECT owner_id FROM resource WHERE id = ?1", [id], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let may = grantor == 1
            || grantor == owner
            || conn
                .query_row(
                    "SELECT 1 FROM grant_ WHERE grantee = ?1 AND object_id = ?2 AND privilege = 'grant' AND with_grant = 1",
                    rusqlite::params![grantor, id],
                    |_| Ok(()),
                )
                .is_ok();
        if !may {
            return Err("only the owner (or a grant carrying with_grant) may share this".into());
        }
        conn.execute(
            "INSERT INTO grant_ (grantee, object_id, privilege, grantor, with_grant)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(grantee, object_id, privilege) DO UPDATE SET grantor = ?4, with_grant = ?5",
            rusqlite::params![grantee, id, privilege, grantor, with_grant as i64],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Take one privilege back. The owner and the admin may always; a
    /// delegator may take back what it granted.
    pub fn revoke_grant(
        &self,
        actor: i64,
        kind: &str,
        name: &str,
        grantee: i64,
        privilege: &str,
    ) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        let Some(id) = Self::resource_id(&conn, kind, name) else {
            return Err("no such resource".into());
        };
        let owner: i64 = conn
            .query_row("SELECT owner_id FROM resource WHERE id = ?1", [id], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let may = actor == 1 || actor == owner
            || conn
                .query_row(
                    "SELECT 1 FROM grant_ WHERE grantee = ?1 AND object_id = ?2 AND grantor = ?1",
                    rusqlite::params![actor, id],
                    |_| Ok(()),
                )
                .is_ok();
        if !may {
            return Err("only the owner, the admin, or the grantor may revoke this".into());
        }
        conn.execute(
            "DELETE FROM grant_ WHERE grantee = ?1 AND object_id = ?2 AND privilege = ?3",
            rusqlite::params![grantee, id, privilege],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Does `user_id` hold `privilege` on `(kind, name)`? The admin always
    /// does; the owner always does on their own resource; everyone else
    /// needs the grant row. A resource with no row is invisible — deny.
    pub fn can(&self, user_id: i64, kind: &str, name: &str, privilege: &str) -> bool {
        if user_id == 1 {
            return true;
        }
        let conn = self.conn.lock().unwrap();
        let Some(id) = Self::resource_id(&conn, kind, name) else {
            return false;
        };
        let owner: i64 = conn
            .query_row("SELECT owner_id FROM resource WHERE id = ?1", [id], |r| r.get(0))
            .unwrap_or(-1);
        if owner == user_id {
            return true;
        }
        conn.query_row(
            "SELECT 1 FROM grant_ WHERE grantee = ?1 AND object_id = ?2 AND privilege = ?3",
            rusqlite::params![user_id, id, privilege],
            |_| Ok(()),
        )
        .is_ok()
    }

    /// Resources of one kind a user may `read` — the list query behind
    /// "unshared resources are invisible".
    pub fn readable(&self, user_id: i64, kind: &str, names: &[String]) -> Vec<String> {
        if user_id == 1 {
            return names.to_vec();
        }
        let conn = self.conn.lock().unwrap();
        names
            .iter()
            .filter(|n| {
                let Some(id) = Self::resource_id(&conn, kind, n) else {
                    return false;
                };
                let owner: i64 = conn
                    .query_row("SELECT owner_id FROM resource WHERE id = ?1", [id], |r| r.get(0))
                    .unwrap_or(-1);
                owner == user_id
                    || conn
                        .query_row(
                            "SELECT 1 FROM grant_ WHERE grantee = ?1 AND object_id = ?2 AND privilege = 'read'",
                            rusqlite::params![user_id, id],
                            |_| Ok(()),
                        )
                        .is_ok()
            })
            .cloned()
            .collect()
    }

    /// Every grant on one resource — the share dialog's data. Returns
    /// `(owner_id, rows)`; `owner_id` is `None` when the row does not exist.
    pub fn grants_on(&self, kind: &str, name: &str) -> (Option<i64>, Vec<Value>) {
        let conn = self.conn.lock().unwrap();
        let Some(id) = Self::resource_id(&conn, kind, name) else {
            return (None, Vec::new());
        };
        let owner: i64 = conn
            .query_row("SELECT owner_id FROM resource WHERE id = ?1", [id], |r| r.get(0))
            .unwrap_or(-1);
        let mut stmt = conn
            .prepare(
                "SELECT g.grantee, g.privilege, g.grantor, g.with_grant, p.name
                 FROM grant_ g JOIN principal p ON p.id = g.grantee
                 WHERE g.object_id = ?1 ORDER BY g.grantee, g.privilege",
            )
            .unwrap();
        let rows = stmt
            .query_map([id], |r| {
                Ok(serde_json::json!({
                    "grantee": r.get::<_, i64>(0)?,
                    "privilege": r.get::<_, String>(1)?,
                    "grantor": r.get::<_, i64>(2)?,
                    "withGrant": r.get::<_, i64>(3)? != 0,
                    "granteeName": r.get::<_, String>(4)?,
                }))
            })
            .unwrap();
        (Some(owner), rows.filter_map(Result::ok).collect())
    }

    /// Transfer ownership (the owner or the admin only).
    pub fn set_owner(&self, actor: i64, kind: &str, name: &str, new_owner: i64) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        let Some(id) = Self::resource_id(&conn, kind, name) else {
            return Err("no such resource".into());
        };
        let owner: i64 = conn
            .query_row("SELECT owner_id FROM resource WHERE id = ?1", [id], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        if actor != 1 && actor != owner {
            return Err("only the owner or the admin may transfer ownership".into());
        }
        conn.execute(
            "UPDATE resource SET owner_id = ?2 WHERE id = ?1",
            rusqlite::params![id, new_owner],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv_store::ConvStore;

    fn id_store() -> (ConvStore, std::path::PathBuf) {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "identity-test-{}-{}-{}",
            std::process::id(),
            now_ms() as u64,
            n
        ));
        (ConvStore::new(&dir).unwrap(), dir)
    }

    #[test]
    fn the_first_user_is_the_admin_and_names_are_case_insensitive() {
        let (store, _d) = id_store();
        let id = store.identity();
        let bob = id.create_user("Bob", "Bob").unwrap();
        assert!(bob.is_admin(), "the first user is the admin");
        assert_eq!(id.create_user("Bob", "dup").unwrap_err(), "that name is taken");
        assert!(id.user_by_name("BOB").is_some(), "login is case-insensitive");
        let alice = id.create_user("alice", "Alice").unwrap();
        assert!(!alice.is_admin());
        // The admin cannot be disabled, by design.
        assert!(id.set_disabled(1, true).is_err());
    }

    #[test]
    fn devices_mint_validate_and_revoke_without_storing_the_cookie() {
        let (store, _d) = id_store();
        let id = store.identity();
        let u = id.create_user("Bob", "").unwrap();
        let (token, dev) = id.issue_device(u.id, "fp-abc", "ua", "10.0.0.2").unwrap();
        assert_eq!(token.len(), 64, "32 bytes hex");
        assert_ne!(dev.token_hint, token, "the hint is never the token");
        let (user, d) = id.device_by_token(&token).unwrap();
        assert_eq!(user.id, u.id);
        assert_eq!(d.id, dev.id);
        assert!(id.device_by_token("wrong").is_none());
        id.revoke_device(u.id, dev.id).unwrap();
        assert!(id.device_by_token(&token).is_none(), "revoked stops validating");
        // A second device cannot be revoked through the first user's scope
        // by a different user.
        let (t2, d2) = id.issue_device(u.id, "fp", "ua", "ip").unwrap();
        let other = id.create_user("alice", "").unwrap();
        assert!(id.revoke_device(other.id, d2.id).is_err());
        assert!(id.device_by_token(&t2).is_some());
    }

    #[test]
    fn capabilities_default_deny_but_the_admin_needs_none() {
        let (store, _d) = id_store();
        let id = store.identity();
        let admin = id.create_user("Bob", "").unwrap();
        let u = id.create_user("alice", "").unwrap();
        assert!(!id.has_capability(u.id, "session.create"), "default deny");
        assert!(id.has_capability(admin.id, "session.create"), "implicit");
        id.grant_capability(u.id, "session.create").unwrap();
        assert!(id.has_capability(u.id, "session.create"));
        id.revoke_capability(u.id, "session.create");
        assert!(!id.has_capability(u.id, "session.create"));
        assert!(id.grant_capability(u.id, "root.everything").is_err(), "unknown caps are refused");
    }

    #[test]
    fn allowlists_are_ceilings_on_editing_never_on_using() {
        let (store, _d) = id_store();
        let id = store.identity();
        let admin = id.create_user("Bob", "").unwrap();
        let u = id.create_user("alice", "").unwrap();
        assert!(!id.may_assign(u.id, "tools", "chat__sessions_list"), "empty = deny");
        id.set_allowlist(u.id, "tools", &["chat__sessions_list".into()]).unwrap();
        assert!(id.may_assign(u.id, "tools", "chat__sessions_list"));
        assert!(!id.may_assign(u.id, "tools", "term_local__bash"), "not on the list");
        id.set_allowlist(u.id, "tools", &["*".into()]).unwrap();
        assert!(id.may_assign(u.id, "tools", "term_local__bash"));
        // The admin's ceiling does not exist.
        assert!(id.may_assign(admin.id, "tools", "anything"));
    }

    #[test]
    fn grants_are_invisible_until_given_and_delegation_needs_with_grant() {
        let (store, _d) = id_store();
        let id = store.identity();
        let owner = id.create_user("Bob", "").unwrap();
        let fam = id.create_user("alice", "").unwrap();
        let stranger = id.create_user("kid", "").unwrap();
        id.ensure_resource("agent", "ping", owner.id).unwrap();

        assert!(!id.can(fam.id, "agent", "ping", "read"), "no grant = invisible");
        assert!(id.can(owner.id, "agent", "ping", "delete"), "the owner holds everything");
        id.grant(fam.id, "agent", "ping", "post", owner.id, false).unwrap();
        assert!(id.can(fam.id, "agent", "ping", "post"));
        assert!(!id.can(fam.id, "agent", "ping", "read"), "post does not imply read");

        // A grant without with_grant cannot be re-shared.
        assert!(id.grant(stranger.id, "agent", "ping", "read", fam.id, false).is_err());
        // With it, it can.
        id.grant(fam.id, "agent", "ping", "grant", owner.id, true).unwrap();
        id.grant(stranger.id, "agent", "ping", "read", fam.id, false).unwrap();
        assert!(id.can(stranger.id, "agent", "ping", "read"));

        // The admin sees and does everything.
        assert!(id.can(1, "agent", "ping", "delete"));
        id.set_owner(1, "agent", "ping", fam.id).unwrap();
        assert_eq!(id.resource("agent", "ping").unwrap().1, fam.id);
    }

    #[test]
    fn readable_filters_the_list_to_what_is_shared() {
        let (store, _d) = id_store();
        let id = store.identity();
        let owner = id.create_user("Bob", "").unwrap();
        let fam = id.create_user("alice", "").unwrap();
        id.ensure_resource("session", "s1", owner.id).unwrap();
        id.ensure_resource("session", "s2", owner.id).unwrap();
        id.ensure_resource("session", "s3", owner.id).unwrap();
        id.grant(fam.id, "session", "s2", "read", owner.id, false).unwrap();
        let all = vec!["s1".to_string(), "s2".to_string(), "s3".to_string()];
        assert_eq!(id.readable(fam.id, "session", &all), vec!["s2".to_string()]);
        assert_eq!(id.readable(owner.id, "session", &all), all);
    }
}
