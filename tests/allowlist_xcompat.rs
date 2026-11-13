//! Differential test: the Rust allowlist matcher vs the JavaScript one.
//!
//! This is a **security boundary**, so agreement is checked against the real
//! `angela/src/allowlist.mjs` rather than against my reading of it. The
//! expected results in `tests/fixtures/allow_expected.json` were produced by
//! running the JS over `allow_cases.json`.
//!
//! Regenerate:
//! ```sh
//! cat > /tmp/js_allow.mjs <<'JS'
//! import { isAllowed } from '/workspace/angela/src/allowlist.mjs';
//! const cases = JSON.parse(await Bun.file(process.argv[2]).text());
//! console.log(JSON.stringify(cases.map(([a,t,c]) => isAllowed(a,t,c))));
//! JS
//! bun run /tmp/js_allow.mjs tests/fixtures/allow_cases.json \
//!   > tests/fixtures/allow_expected.json
//! ```

use angela::allowlist::is_allowed;

#[test]
fn the_matcher_agrees_with_the_javascript() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let cases: Vec<Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("allow_cases.json")).unwrap())
            .unwrap();
    let expected: Vec<bool> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("allow_expected.json")).unwrap())
            .unwrap();
    assert_eq!(cases.len(), expected.len(), "fixture mismatch");

    let mut disagreements = Vec::new();
    for (case, want) in cases.iter().zip(&expected) {
        let got = is_allowed(&case[0], &case[1], &case[2]);
        if got != *want {
            disagreements.push(format!(
                "allowlist={:?} tool={:?} cmd={:?} → rust={got} js={want}",
                case[0], case[1], case[2]
            ));
        }
    }
    assert!(
        disagreements.is_empty(),
        "the Rust matcher disagrees with the JS on {} case(s):\n{}",
        disagreements.len(),
        disagreements.join("\n")
    );
}

/// Whatever else changes, these must never become permissive.
#[test]
fn the_deny_by_default_properties_hold() {
    for empty in ["", "   ", "\n\n", "# comment only\n"] {
        assert!(
            !is_allowed(empty, "term_local__bash", "rm -rf /"),
            "empty allowlist permitted a call: {empty:?}"
        );
    }
    // A malformed pattern must deny, not wildcard.
    assert!(!is_allowed("[unclosed", "term_local__bash", "rm -rf /"));
    // A rule for one tool must not leak to another.
    assert!(!is_allowed("file_io__read_file", "term_local__bash", "ls"));
}
