//! End-to-end turn against a live OpenAI-compatible endpoint.
//!
//! Skips (rather than fails) when no endpoint is reachable, so the suite
//! stays green on a machine with no model running. Point it elsewhere with
//! `AGL_BASE_URL` / `ANGELA_TEST_MODEL`.

use angela::agent_loader;
use angela::harness::{run_turn, LiveTabs, McpPool, TurnRequest};
use angela::session_store::SessionStore;
use serde_json::Value;

fn endpoint_up() -> bool {
    let base = std::env::var("AGL_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234".into());
    let addr = base
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    let addr = addr.split('/').next().unwrap_or(addr);
    std::net::TcpStream::connect_timeout(
        &match addr.parse() {
            Ok(a) => a,
            Err(_) => return false,
        },
        std::time::Duration::from_millis(400),
    )
    .is_ok()
}

#[test]
fn a_turn_streams_persists_and_is_remembered() {
    if !endpoint_up() {
        eprintln!("no inference endpoint; skipping live turn test");
        return;
    }
    let model = std::env::var("ANGELA_TEST_MODEL")
        .unwrap_or_else(|_| "llama-server:default".into());

    let tmp = tempfile::tempdir().unwrap();
    let store = SessionStore::new(tmp.path()).unwrap();
    let tabs = LiveTabs::default();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    // No MCP presets: this test is about the turn loop, not tools.
    let pool = McpPool::default();
    let def = agent_loader::AgentDef::default();

    // ---- turn one --------------------------------------------------------
    let mut rows: Vec<Value> = Vec::new();
    run_turn(
        &store,
        &tabs,
        &rt,
        "/tmp",
        &pool,
        &def,
        TurnRequest {
            tab_id: "1".into(),
            content: "Reply with exactly: pineapple".into(),
            agent: None,
            model: Some(model.clone()),
            new_session: false,
            session_id: None,
        },
        &mut |v: &Value| {
            rows.push(v.clone());
            true
        },
    )
    .expect("turn one");

    let kinds: Vec<&str> = rows.iter().filter_map(|r| r["type"].as_str()).collect();
    assert_eq!(kinds.first(), Some(&"session"), "the session row leads");
    assert_eq!(kinds.last(), Some(&"done"), "and `done` closes the stream");
    assert!(
        rows.iter().any(|r| r["type"] == "assistant_delta"),
        "the answer must stream, not arrive in one lump"
    );
    let final_card = rows
        .iter()
        .rev()
        .find(|r| r["type"] == "card" && r["card"]["kind"] == "assistant")
        .expect("a sealed assistant card");
    assert_eq!(final_card["card"]["streaming"], false);

    // ---- it persisted ----------------------------------------------------
    let sid = rows[0]["sessionId"].as_str().unwrap().to_string();
    let st = store.load(&sid).expect("session on disk");
    assert_eq!(st["turns"], 1);
    assert_eq!(st["status"], "ready");
    assert_eq!(
        st["title"], "Reply with exactly: pineapple",
        "titled from the first user message"
    );

    let msgs = angela::serve::reconstruct_messages_from_log(&store, &sid);
    assert_eq!(msgs.first().unwrap()["kind"], "user");
    assert!(
        msgs.iter().any(|m| m["kind"] == "assistant"),
        "the reply is in the transcript"
    );

    // ---- turn two on the same tab remembers turn one ---------------------
    let mut answer = String::new();
    run_turn(
        &store,
        &tabs,
        &rt,
        "/tmp",
        &pool,
        &def,
        TurnRequest {
            tab_id: "1".into(),
            content: "What word did I just ask for? One word.".into(),
            agent: None,
            model: Some(model),
            new_session: false,
            session_id: None,
        },
        &mut |v: &Value| {
            if v["type"] == "assistant_delta" {
                answer.push_str(v["text"].as_str().unwrap_or(""));
            }
            true
        },
    )
    .expect("turn two");

    assert!(
        answer.to_lowercase().contains("pineapple"),
        "the second turn lost the first: {answer:?}"
    );
    let st = store.load(&sid).unwrap();
    assert_eq!(st["turns"], 2, "both turns recorded on ONE session");
    assert_eq!(store.list().len(), 1, "the tab reused its session");
}

#[test]
fn an_empty_prompt_is_refused_before_any_session_is_made() {
    let tmp = tempfile::tempdir().unwrap();
    let store = SessionStore::new(tmp.path()).unwrap();
    let tabs = LiveTabs::default();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let pool = McpPool::default();
    let def = agent_loader::AgentDef::default();
    let err = run_turn(
        &store,
        &tabs,
        &rt,
        "/tmp",
        &pool,
        &def,
        TurnRequest {
            tab_id: "1".into(),
            content: "   ".into(),
            agent: None,
            model: None,
            new_session: false,
            session_id: None,
        },
        &mut |_: &Value| true,
    );
    assert!(err.is_err());
    assert!(store.list().is_empty(), "no junk session was created");
}
