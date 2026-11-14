//! Cross-compatibility with the JavaScript angela.
//!
//! The on-disk session format is consumed by shipped clients, so "parity"
//! here means the Rust store can read what the JS store wrote — folded metas,
//! counters and all. The fixture in `tests/fixtures/` was produced by running
//! the real `angela/src/session-store.mjs` under Bun; regenerate it with
//! `tests/fixtures/REGENERATE.md` if the JS format changes.

use angela::session_store::{NewEvent, SessionStore};
use serde_json::json;

fn fixture_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/js-session")
}

#[test]
fn the_rust_store_reads_a_js_written_session() {
    let root = fixture_root();
    if !root.join(".angela/sessions").is_dir() {
        eprintln!("fixture missing; skipping (see tests/fixtures/REGENERATE.md)");
        return;
    }
    let store = SessionStore::new(&root).unwrap();

    let ids = store.list();
    assert_eq!(ids.len(), 1, "fixture holds exactly one session");
    let id = &ids[0];

    // --- metadata the JS wrote -------------------------------------------
    let st = store.load(id).expect("load the JS-written state");
    assert_eq!(st["apiVersion"], "angela/v1");
    assert_eq!(st["logEncoding"], "delta/v1");
    assert_eq!(st["kind"], "AgentSession");
    assert_eq!(st["agent"], "lucy");
    assert_eq!(st["model"], "llama-server:default");
    assert_eq!(st["title"], "xcompat");
    assert_eq!(st["turns"], 1, "recordTurnComplete bumped it");
    assert_eq!(st["status"], "ready");
    assert_eq!(st["logFile"], format!("{id}.jsonl"));

    // --- the event log ----------------------------------------------------
    let raw = store.list_events(id);
    let types: Vec<&str> = raw.iter().map(|e| e.event_type.as_str()).collect();
    assert_eq!(
        types,
        vec![
            "session",       // created
            "context_window",
            "tool_call",
            "context_meta",
            "session",       // turn_complete
        ]
    );
    assert_eq!(raw[0].payload["kind"], "created");
    assert_eq!(raw[2].duration_ms, Some(12.5), "JS numbers survive");
    assert_eq!(raw[4].payload["kind"], "turn_complete");
    assert_eq!(raw[4].duration_ms, Some(99.0));

    // --- the fold the JS expects a reader to perform ----------------------
    let folded = store.list_events_folded(id);
    let hidden = folded
        .iter()
        .find(|e| e.event_type == "context_window")
        .expect("the user row");
    assert!(
        !hidden.visible(),
        "the context_meta row must hide the earlier event"
    );
    assert_eq!(hidden.meta["note"], "hidden");

    // and the append-only guarantee: the original line is untouched on disk
    let orig = raw
        .iter()
        .find(|e| e.event_type == "context_window")
        .unwrap();
    assert!(orig.meta.is_empty(), "JS never rewrote the original line");
}

#[test]
fn rust_appends_to_a_js_session_without_breaking_it() {
    // Copy the fixture so the test never mutates it.
    let src = fixture_root();
    if !src.join(".angela/sessions").is_dir() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let dst = tmp.path().join(".angela/sessions");
    std::fs::create_dir_all(&dst).unwrap();
    for e in std::fs::read_dir(src.join(".angela/sessions")).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), dst.join(e.file_name())).unwrap();
    }

    let store = SessionStore::new(tmp.path()).unwrap();
    let id = store.list()[0].clone();
    let before = store.load(&id).unwrap()["eventCount"].as_u64().unwrap();

    store
        .append_event(&id, NewEvent::new("context_window", json!({"role":"assistant","content":"hi"})))
        .unwrap();

    let after = store.load(&id).unwrap();
    assert_eq!(after["eventCount"].as_u64().unwrap(), before + 1);
    // every line still parses, including the ones JS wrote
    let evs = store.list_events(&id);
    assert_eq!(evs.len() as u64, before + 1);
    assert_eq!(evs.last().unwrap().payload["content"], "hi");
}

// ===========================================================================
// Differential test: the SAME event log through both implementations.
// ===========================================================================

/// The JS output was produced by running the real `buildContextWindowItems`
/// over a log that exercises every item shape and every skip rule. Comparing
/// against it is stronger than asserting my reading of the JS: if the port
/// diverges on any field, this fails and names the row.
#[test]
fn build_context_window_items_matches_the_js_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/js-items");
    let expected_path = root.join("expected.json");
    if !expected_path.exists() {
        eprintln!("fixture missing; skipping (see tests/fixtures/REGENERATE.md)");
        return;
    }
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&expected_path).unwrap()).unwrap();
    let id = expected["id"].as_str().unwrap();

    let store = SessionStore::new(&root).unwrap();
    let items = store.build_context_window_items(id);
    let got = serde_json::to_value(&items).unwrap();
    let want = &expected["items"];

    let want_arr = want.as_array().unwrap();
    assert_eq!(
        items.len(),
        want_arr.len(),
        "item count differs\n rust: {}\n js:   {}",
        serde_json::to_string_pretty(&got).unwrap(),
        serde_json::to_string_pretty(want).unwrap()
    );

    for (i, (g, w)) in got.as_array().unwrap().iter().zip(want_arr).enumerate() {
        assert_eq!(
            g, w,
            "row {i} differs\n rust: {}\n js:   {}",
            serde_json::to_string_pretty(g).unwrap(),
            serde_json::to_string_pretty(w).unwrap()
        );
    }
}

#[test]
fn list_visible_context_messages_matches_the_js_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/js-items");
    let expected_path = root.join("expected.json");
    if !expected_path.exists() {
        return;
    }
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&expected_path).unwrap()).unwrap();
    let id = expected["id"].as_str().unwrap();
    let store = SessionStore::new(&root).unwrap();
    let got = serde_json::to_value(store.list_visible_context_messages(id)).unwrap();
    assert_eq!(
        got, expected["msgs"],
        "rust: {}",
        serde_json::to_string_pretty(&got).unwrap()
    );
}

// ===========================================================================
// Route-level checks against the same JS-written fixture.
// ===========================================================================

/// The HTTP layer's transcript must come out in the five card kinds the chat
/// UI renders, in log order, from a session the JS wrote.
#[test]
fn the_serve_layer_reconstructs_js_written_sessions() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/js-items");
    if !root.join(".angela/sessions").is_dir() {
        return;
    }
    let store = SessionStore::new(&root).unwrap();
    let id = store.list()[0].clone();
    let msgs = angela::serve::reconstruct_messages_from_log(&store, &id);

    let kinds: Vec<&str> = msgs
        .iter()
        .map(|m| m["kind"].as_str().unwrap_or(""))
        .collect();
    // From the fixture: user, hidden assistant, assistant, 2 reasoning,
    // 3 tool_calls, 4 tool_results.
    assert_eq!(kinds.iter().filter(|k| **k == "user").count(), 1);
    assert_eq!(kinds.iter().filter(|k| **k == "assistant").count(), 2);
    assert_eq!(kinds.iter().filter(|k| **k == "reasoning").count(), 2);
    assert_eq!(kinds.iter().filter(|k| **k == "tool_call").count(), 3);
    assert_eq!(kinds.iter().filter(|k| **k == "tool_result").count(), 4);

    // Every row carries the fields the UI binds to.
    for m in &msgs {
        assert!(m.get("id").is_some(), "{m}");
        assert!(m.get("eventId").is_some(), "{m}");
        assert!(m.get("visible").unwrap().is_boolean(), "{m}");
        assert!(m.get("telemetry").is_some(), "{m}");
        assert!(m.get("goto").is_some(), "{m}");
    }

    // The context_meta hide survives into the UI shape.
    let hidden = msgs.iter().filter(|m| !m["visible"].as_bool().unwrap()).count();
    assert_eq!(hidden, 1, "exactly the row the JS hid");

    // A snapshot row is telemetry and must never reach the transcript.
    assert!(
        !msgs.iter().any(|m| m["text"].as_str() == Some("DUMP")),
        "provider_messages_snapshot leaked into the transcript"
    );
}

/// The sessions list is newest-first and carries the nine fields the JS sends.
#[test]
fn the_sessions_list_shape_matches_the_js() {
    use std::sync::Arc;
    let tmp = tempfile::tempdir().unwrap();
    let state = Arc::new(angela::serve::AppState::new(tmp.path(), 8763).unwrap());

    let a = state
        .store
        .create(angela::session_store::NewSession {
            title: Some("older".into()),
            agent: Some("lucy".into()),
            ..Default::default()
        })
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let _b = state
        .store
        .create(angela::session_store::NewSession {
            // no title: the JS falls back to `agent`, then `id`
            agent: Some("cowork".into()),
            ..Default::default()
        })
        .unwrap();

    let listed = angela::serve::Server::new(Arc::clone(&state)).sessions_list_for_test();
    let rows = listed["sessions"].as_array().unwrap();
    assert_eq!(rows.len(), 2);

    // NOTE: ordering is NOT asserted between these two. Session ids carry the
    // timestamp only to the second (`2026-11-13T05-25-30` + random hex), so
    // two sessions created inside the same second tie-break on random bytes.
    // The JS `newId()` has exactly the same precision, so this is faithful
    // rather than a port defect — see `ids_only_order_to_the_second`.
    let by_agent = |name: &str| {
        rows.iter()
            .find(|r| r["agent"] == name)
            .unwrap_or_else(|| panic!("no row for {name}"))
    };
    assert_eq!(
        by_agent("cowork")["title"],
        "cowork",
        "a title-less session falls back to the agent name"
    );
    assert_eq!(by_agent("lucy")["title"], "older");
    assert_eq!(by_agent("lucy")["id"], a["id"]);

    let keys: Vec<&str> = rows[0].as_object().unwrap().keys().map(|s| s.as_str()).collect();
    assert_eq!(
        keys,
        vec![
            // `ordinal` is ours, not the JS's: the tab strip needs a stable
            // 1-based label, and deriving one at read time would reshuffle
            // it as sessions come and go.
            "id", "ordinal", "title", "agent", "model", "createdAt", "updatedAt", "status",
            "turns", "contextWindow"
        ]
    );
}

/// Ordering is by filename, and the filename's time component resolves only
/// to the second — so sessions a second or more apart sort reliably, and
/// sessions inside one second do not. Pinned so the limit is documented
/// rather than discovered.
#[test]
fn ids_only_order_to_the_second() {
    let tmp = tempfile::tempdir().unwrap();
    let store = SessionStore::new(tmp.path()).unwrap();
    let first = store
        .create(angela::session_store::NewSession::default())
        .unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let second = store
        .create(angela::session_store::NewSession::default())
        .unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    // distinct seconds => the stems differ and sort correctly
    assert_ne!(first[..19], second[..19], "a second apart");
    assert_eq!(store.list(), vec![second, first], "newest first");
}
