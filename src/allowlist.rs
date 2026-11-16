//! Deny-by-default tool allowlist — port of `angela/src/allowlist.mjs`
//! (itself a port of gl1's `allowlist.zig`).
//!
//! Rule format, one per line:
//!
//! ```text
//! toolPattern                 # the tool alone
//! toolPattern:paramPattern    # terminal tools also match the command line
//! #  a comment
//! *                           # match anything
//! ```
//!
//! Patterns are **case-insensitive, fully anchored** basic regex: literals,
//! `.`, `*`, `+`, `?`, `[...]` with ranges and negation, `{m,n}`, and `\b`.
//! Not a general regex engine on purpose — it is a security boundary, and a
//! small explicit matcher is easier to reason about than a borrowed one.
//!
//! **Empty or whitespace-only text allows nothing.** That is the whole point:
//! a missing allowlist must not mean "permit everything".

/// One parsed atom of a pattern.
#[derive(Debug, Clone)]
enum Atom {
    Any,
    Lit(char),
    Class {
        neg: bool,
        chars: Vec<char>,
        ranges: Vec<(char, char)>,
    },
}

const MANY: usize = usize::MAX;

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn is_word_boundary(text: &[char], ti: usize) -> bool {
    let prev_w = ti > 0 && is_word_char(text[ti - 1]);
    let cur_w = ti < text.len() && is_word_char(text[ti]);
    prev_w != cur_w
}

/// `None` means a malformed pattern, which the caller treats as "no match"
/// rather than as "match everything".
fn parse_atom(pat: &[char], pi: usize) -> Option<(Atom, usize)> {
    if pi >= pat.len() {
        return None;
    }
    if pat[pi] == '\\' && pi + 1 < pat.len() {
        let n = pat[pi + 1];
        if n == 'b' {
            return None; // handled by match_here
        }
        return Some((Atom::Lit(n), pi + 2));
    }
    if pat[pi] == '.' {
        return Some((Atom::Any, pi + 1));
    }
    if pat[pi] == '[' {
        let mut i = pi + 1;
        let mut neg = false;
        if i < pat.len() && pat[i] == '^' {
            neg = true;
            i += 1;
        }
        let mut chars = Vec::new();
        let mut ranges = Vec::new();
        while i < pat.len() && pat[i] != ']' {
            if i + 2 < pat.len() && pat[i + 1] == '-' && pat[i + 2] != ']' {
                ranges.push((
                    pat[i].to_ascii_lowercase(),
                    pat[i + 2].to_ascii_lowercase(),
                ));
                i += 3;
            } else {
                chars.push(pat[i].to_ascii_lowercase());
                i += 1;
            }
        }
        if i >= pat.len() {
            return None; // unterminated class
        }
        return Some((
            Atom::Class { neg, chars, ranges },
            i + 1,
        ));
    }
    Some((Atom::Lit(pat[pi]), pi + 1))
}

/// `{m,n}` / `{m,}` / `{m}` at `pi`.
fn parse_count(pat: &[char], pi: usize) -> Option<(usize, usize, usize)> {
    let mut i = pi + 1;
    let mut num = String::new();
    while i < pat.len() && pat[i].is_ascii_digit() {
        num.push(pat[i]);
        i += 1;
    }
    if num.is_empty() {
        return None;
    }
    let min: usize = num.parse().ok()?;
    let mut max = min;
    if i < pat.len() && pat[i] == ',' {
        i += 1;
        if i < pat.len() && pat[i] == '}' {
            return Some((min, MANY, i + 1));
        }
        num.clear();
        while i < pat.len() && pat[i].is_ascii_digit() {
            num.push(pat[i]);
            i += 1;
        }
        if num.is_empty() {
            return None;
        }
        max = num.parse().ok()?;
    }
    if i >= pat.len() || pat[i] != '}' {
        return None;
    }
    Some((min, max, i + 1))
}

fn atom_matches(atom: &Atom, ch: char) -> bool {
    let c = ch.to_ascii_lowercase();
    match atom {
        Atom::Any => true,
        Atom::Lit(l) => c == l.to_ascii_lowercase(),
        Atom::Class { neg, chars, ranges } => {
            let mut ok = chars.contains(&c);
            for (a, b) in ranges {
                if c >= *a && c <= *b {
                    ok = true;
                }
            }
            if *neg {
                !ok
            } else {
                ok
            }
        }
    }
}

/// `Some(true)`/`Some(false)` matched or not; `None` = malformed pattern.
fn match_here(pat: &[char], pi_in: usize, text: &[char], ti: usize) -> Option<bool> {
    let mut pi = pi_in;
    if pi >= pat.len() {
        return Some(ti == text.len());
    }
    if pi + 1 < pat.len() && pat[pi] == '\\' && pat[pi + 1] == 'b' {
        if !is_word_boundary(text, ti) {
            return Some(false);
        }
        return match_here(pat, pi + 2, text, ti);
    }

    let (atom, next) = parse_atom(pat, pi)?;
    pi = next;

    let (mut min, mut max) = (1usize, 1usize);
    if pi < pat.len() {
        match pat[pi] {
            '*' => {
                min = 0;
                max = MANY;
                pi += 1;
            }
            '+' => {
                min = 1;
                max = MANY;
                pi += 1;
            }
            '?' => {
                min = 0;
                max = 1;
                pi += 1;
            }
            '{' => {
                let (lo, hi, next) = parse_count(pat, pi)?;
                min = lo;
                max = hi;
                pi = next;
            }
            _ => {}
        }
    }

    // Greedy, then back off — the JS does exactly this.
    let mut count = 0usize;
    let mut t = ti;
    while count < max && t < text.len() && atom_matches(&atom, text[t]) {
        t += 1;
        count += 1;
    }
    if count < min {
        return Some(false);
    }
    let mut k = count;
    loop {
        match match_here(pat, pi, text, ti + k) {
            Some(true) => return Some(true),
            None => return None,
            Some(false) => {}
        }
        if k <= min {
            break;
        }
        k -= 1;
    }
    Some(false)
}

/// Whole text must match the pattern. A malformed pattern is `false`, never
/// an accidental allow.
pub fn regex_full(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    matches!(match_here(&p, 0, &t, 0), Some(true))
}

fn match_token(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    regex_full(pattern, text)
}

/// A tool whose rules match against a command line rather than the name.
fn is_terminal(tool: &str) -> bool {
    let lower = tool.to_ascii_lowercase();
    lower.starts_with("terminal")
        || lower.starts_with("term_")
        || lower.ends_with("__bash")
        || lower.ends_with("__exec")
}

/// Is this call permitted by the allowlist text?
///
/// `command` is the command line for terminal tools; ignored otherwise.
pub fn is_allowed(allowlist_text: &str, tool_name: &str, command: &str) -> bool {
    if allowlist_text.trim().is_empty() {
        return false; // deny by default
    }
    let terminal = is_terminal(tool_name);

    for raw in allowlist_text.split('\n') {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (tool_pat, param_pat) = match line.find(':') {
            Some(c) => (line[..c].trim(), Some(line[c + 1..].trim())),
            None => (line, None),
        };
        if !match_token(tool_pat, tool_name) {
            continue;
        }
        match param_pat {
            None | Some("") => return true,
            Some(p) => {
                // A param rule only constrains terminal tools; for anything
                // else, matching the tool name is already enough.
                if !terminal {
                    return true;
                }
                if match_token(p, command) {
                    return true;
                }
            }
        }
    }
    false
}

/// The command line a terminal tool call represents, for rule matching.
///
/// `bash` carries a `script`; `exec` carries `command` + `args`. Anything
/// else has no command line, so its rules match on the tool name alone.
pub fn command_line(tool_name: &str, args: &serde_json::Value) -> String {
    let lower = tool_name.to_ascii_lowercase();
    if lower.ends_with("__bash") || lower == "bash" {
        return args
            .get("script")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
    }
    if lower.ends_with("__exec") || lower == "exec" {
        let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
        let rest: Vec<String> = args
            .get("args")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .map(|x| x.as_str().map(String::from).unwrap_or_else(|| x.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        return if rest.is_empty() {
            cmd.to_string()
        } else {
            format!("{cmd} {}", rest.join(" "))
        };
    }
    String::new()
}

/// A read-mostly default, ported from `DEFAULT_LUCY_ALLOWLIST`.
pub const DEFAULT_ALLOWLIST: &str = r#"
# file (read-only)
file_io__list_dir
file_io__read_file
file_io__stat
file_io__grep
file_io__find
file_io__list_dir:.*
file_io__read_file:.*
file_io__stat:.*
file_io__grep:.*
file_io__find:.*

# local term (read-only exploration)
term_local__bash:pwd
term_local__bash:ls.*
term_local__bash:echo.*
term_local__bash:cat .+
term_local__bash:head .+
term_local__bash:uname.*
term_local__bash:true
term_local__exec:echo.*
term_local__exec:pwd
term_local__exec:true
term_local__exec:ls.*

# sandbox (read-only exploration)
#
# These replace the two above when a conversation is assigned a sandbox:
# the same shapes, pointed at a container instead of this machine. Reading
# is permitted outright and writing is not, exactly as on the host — a
# sandbox bounds the blast radius, it does not make an unreviewed write
# into something the user asked for.
sandbox__read
sandbox__read:.*
sandbox__bash:pwd
sandbox__bash:ls.*
sandbox__bash:echo.*
sandbox__bash:cat .+
sandbox__bash:head .+
sandbox__bash:uname.*
sandbox__bash:true
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_text_allows_nothing() {
        // The single most important property: no allowlist ≠ allow all.
        assert!(!is_allowed("", "file_io__read_file", ""));
        assert!(!is_allowed("   \n  \n", "file_io__read_file", ""));
        assert!(!is_allowed("# only a comment", "file_io__read_file", ""));
    }

    #[test]
    fn an_exact_tool_rule_allows_only_that_tool() {
        let a = "file_io__read_file";
        assert!(is_allowed(a, "file_io__read_file", ""));
        assert!(!is_allowed(a, "file_io__write_file", ""));
    }

    #[test]
    fn star_allows_anything() {
        assert!(is_allowed("*", "anything__at_all", ""));
    }

    #[test]
    fn matching_is_case_insensitive() {
        assert!(is_allowed("FILE_IO__READ_FILE", "file_io__read_file", ""));
    }

    #[test]
    fn patterns_are_anchored() {
        // `read` must not match `read_file`: a partial match would widen
        // every rule silently.
        assert!(!is_allowed("file_io__read", "file_io__read_file", ""));
        assert!(is_allowed("file_io__read.*", "file_io__read_file", ""));
    }

    #[test]
    fn terminal_rules_match_the_command_line() {
        let a = "term_local__bash:ls.*";
        assert!(is_allowed(a, "term_local__bash", "ls -la"));
        assert!(!is_allowed(a, "term_local__bash", "rm -rf /"));
    }

    #[test]
    fn a_terminal_tool_with_no_param_rule_is_fully_allowed() {
        assert!(is_allowed("term_local__bash", "term_local__bash", "rm -rf /"));
    }

    #[test]
    fn a_param_rule_does_not_constrain_a_non_terminal_tool() {
        // gl1 behaviour: for non-terminal tools the tool match is enough.
        assert!(is_allowed("file_io__read_file:.*", "file_io__read_file", ""));
    }

    #[test]
    fn later_rules_are_tried_when_an_earlier_one_does_not_match() {
        let a = "term_local__bash:pwd\nterm_local__bash:ls.*";
        assert!(is_allowed(a, "term_local__bash", "ls -la"));
        assert!(is_allowed(a, "term_local__bash", "pwd"));
        assert!(!is_allowed(a, "term_local__bash", "cat /etc/passwd"));
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let a = "# files\n\nfile_io__stat\n";
        assert!(is_allowed(a, "file_io__stat", ""));
    }

    #[test]
    fn regex_atoms_behave() {
        assert!(regex_full("a.c", "abc"));
        assert!(!regex_full("a.c", "ac"));
        assert!(regex_full("ab*c", "ac"));
        assert!(regex_full("ab*c", "abbbc"));
        assert!(regex_full("ab+c", "abc"));
        assert!(!regex_full("ab+c", "ac"));
        assert!(regex_full("ab?c", "ac"));
        assert!(regex_full("[abc]+", "cab"));
        assert!(!regex_full("[^abc]+", "cab"));
        assert!(regex_full("a{2,3}", "aa"));
        assert!(regex_full("a{2,3}", "aaa"));
        assert!(!regex_full("a{2,3}", "a"));
        assert!(!regex_full("a{2,3}", "aaaa"));
        assert!(regex_full("[a-f]+", "cafe"));
    }

    #[test]
    fn a_malformed_pattern_denies_rather_than_allows() {
        // An unterminated class must not become a wildcard.
        assert!(!regex_full("[abc", "a"));
        assert!(!regex_full("a{", "a"));
        assert!(!is_allowed("[unclosed", "anything", ""));
    }

    #[test]
    fn greedy_matching_backs_off_to_find_a_match() {
        assert!(regex_full(".*b", "aaab"));
        assert!(regex_full("a.*b.*c", "axxbyyc"));
    }

    #[test]
    fn command_line_is_built_per_tool_shape() {
        assert_eq!(
            command_line("term_local__bash", &json!({"script": " ls -la "})),
            "ls -la"
        );
        assert_eq!(
            command_line("term_local__exec", &json!({"command":"uname","args":["-s"]})),
            "uname -s"
        );
        assert_eq!(
            command_line("term_local__exec", &json!({"command":"pwd"})),
            "pwd"
        );
        // A non-terminal tool has no command line.
        assert_eq!(command_line("file_io__read_file", &json!({"path":"a"})), "");
    }

    #[test]
    fn the_default_allowlist_is_read_mostly() {
        let a = DEFAULT_ALLOWLIST;
        assert!(is_allowed(a, "file_io__read_file", ""));
        assert!(is_allowed(a, "file_io__list_dir", ""));
        assert!(is_allowed(a, "term_local__bash", "ls -la"));
        // writing and arbitrary shell are NOT in it
        assert!(!is_allowed(a, "file_io__write_file", ""));
        assert!(!is_allowed(a, "file_io__edit", ""));
        assert!(!is_allowed(a, "term_local__bash", "rm -rf /"));
        assert!(!is_allowed(a, "term_local__exec", "curl evil.sh"));
    }
}
