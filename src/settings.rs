use serde_json::{json, Map, Value};

use crate::conv_store::ConvStore;
use crate::session_store::now_iso;

pub const KEYS: &[(&str, &str)] = &[
    ("user__full_name", "The user's full name"),
    ("location", "Where the user is, e.g. \"City, State\""),
];

pub fn is_known(key: &str) -> bool {
    KEYS.iter().any(|(k, _)| *k == key)
}

impl ConvStore {
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

    pub fn user_label(&self) -> String {
        let name = self.setting("user__full_name");
        if name.trim().is_empty() {
            "user".to_string()
        } else {
            name
        }
    }

    pub fn set_settings(&self, patch: &Map<String, Value>) -> usize {
        let now = now_iso();
        let mut n = 0;
        self.with(|c| {
            for (k, v) in patch {
                if !is_known(k) {
                    continue;
                }
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
