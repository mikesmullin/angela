//! App settings — a handful of key/value rows in the conversation database.
//!
//! Key/value rather than a column per setting, so adding one is a write
//! rather than a migration. The cost is that nothing is typed, which is why
//! [`KEYS`] exists: the writer takes the keys it knows and drops the rest,
//! instead of letting any caller invent a setting that then has to be
//! supported forever.
//!
//! Settings live beside the conversations on purpose. They describe *this
//! project* — who is using it, how it should behave here — not the machine,
//! so they belong with the thing they are about and travel with it.

use serde_json::{json, Map, Value};

use crate::conv_store::ConvStore;
use crate::session_store::now_iso;

/// The settings this build understands, and what each is for.
///
/// Anything else sent to `POST /api/settings` is ignored rather than stored:
/// an unrecognised key is a typo or an older client, and keeping it would
/// mean a row nothing reads and nobody can find.
pub const KEYS: &[(&str, &str)] = &[("user__full_name", "The user's full name")];

pub fn is_known(key: &str) -> bool {
    KEYS.iter().any(|(k, _)| *k == key)
}

impl ConvStore {
    /// Every setting, as a flat object. Absent keys are present and empty,
    /// so a client can bind a form to the result without null checks.
    pub fn settings(&self) -> Value {
        let mut out = Map::new();
        for (k, _) in KEYS {
            out.insert((*k).to_string(), json!(""));
        }
        self.with(|c| {
            let mut st = match c.prepare("SELECT key, value FROM settings") {
                Ok(s) => s,
                Err(_) => return,
            };
            let rows = st.query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            });
            if let Ok(it) = rows {
                for (k, v) in it.flatten() {
                    if is_known(&k) {
                        out.insert(k, json!(v));
                    }
                }
            }
        });
        Value::Object(out)
    }

    pub fn setting(&self, key: &str) -> String {
        self.settings()
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()
    }

    /// What to call the human in a prompt: `[Bob]` rather than `[user]`.
    ///
    /// One place, because it appears in two prompts that must agree — the
    /// turn the agent runs on and the System One call deciding whether it
    /// runs at all. An agent told the speaker is "Bob" in one and "user"
    /// in the other is being asked about a different conversation.
    ///
    /// Falls back to `user` when unset: a bracketed empty label would read
    /// as a participant with no name rather than as a default.
    pub fn user_label(&self) -> String {
        let name = self.setting("user__full_name");
        if name.trim().is_empty() {
            "user".to_string()
        } else {
            name
        }
    }

    /// Write the keys this build knows about. Returns how many it stored.
    pub fn set_settings(&self, patch: &Map<String, Value>) -> usize {
        let now = now_iso();
        let mut n = 0;
        self.with(|c| {
            for (k, v) in patch {
                if !is_known(k) {
                    continue;
                }
                // Everything is text: a setting that arrives as a number or
                // a bool is still a setting, and coercing it here beats
                // refusing it over its JSON type.
                let text = match v {
                    Value::String(s) => s.clone(),
                    Value::Null => String::new(),
                    other => other.to_string(),
                };
                let ok = c.execute(
                    "INSERT INTO settings (key, value, updated_at) VALUES (?1, ?2, ?3)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                    updated_at = excluded.updated_at",
                    rusqlite::params![k, text.trim(), now],
                );
                if ok.is_ok() {
                    n += 1;
                }
            }
        });
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> ConvStore {
        ConvStore::memory("/tmp/p").unwrap()
    }

    fn patch(pairs: &[(&str, &str)]) -> Map<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), json!(v)))
            .collect()
    }

    #[test]
    fn an_unset_setting_reads_as_empty_not_missing() {
        let s = store();
        let got = s.settings();
        // Present and empty, so a form can bind to it without null checks.
        assert_eq!(got["user__full_name"], "");
        assert_eq!(s.setting("user__full_name"), "");
    }

    #[test]
    fn a_setting_round_trips_and_overwrites() {
        let s = store();
        assert_eq!(s.set_settings(&patch(&[("user__full_name", "Ada Lovelace")])), 1);
        assert_eq!(s.setting("user__full_name"), "Ada Lovelace");
        s.set_settings(&patch(&[("user__full_name", "Ada King")]));
        assert_eq!(s.setting("user__full_name"), "Ada King");
    }

    #[test]
    fn an_unknown_key_is_dropped_rather_than_stored() {
        let s = store();
        // A typo or an older client. Storing it would mean a row nothing
        // reads and nobody can find.
        assert_eq!(s.set_settings(&patch(&[("usr_full_nme", "oops")])), 0);
        assert!(s.settings().get("usr_full_nme").is_none());
        let n: i64 = s.with(|c| {
            c.query_row("SELECT COUNT(*) FROM settings", [], |r| r.get(0))
                .unwrap_or(0)
        });
        assert_eq!(n, 0);
    }

    #[test]
    fn the_user_label_falls_back_rather_than_going_blank() {
        let s = store();
        // A bracketed empty label would read as a participant with no name.
        assert_eq!(s.user_label(), "user");
        s.set_settings(&patch(&[("user__full_name", "Bob")]));
        assert_eq!(s.user_label(), "Bob");
        s.set_settings(&patch(&[("user__full_name", "   ")]));
        assert_eq!(s.user_label(), "user");
    }

    #[test]
    fn values_are_trimmed_and_clearable() {
        let s = store();
        s.set_settings(&patch(&[("user__full_name", "  Ada  ")]));
        assert_eq!(s.setting("user__full_name"), "Ada");
        s.set_settings(&patch(&[("user__full_name", "")]));
        assert_eq!(s.setting("user__full_name"), "");
    }

    #[test]
    fn a_non_string_value_is_coerced_rather_than_refused() {
        let s = store();
        let mut p = Map::new();
        p.insert("user__full_name".into(), json!(42));
        s.set_settings(&p);
        assert_eq!(s.setting("user__full_name"), "42");
    }
}
