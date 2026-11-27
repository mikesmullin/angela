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

fn parse_atom(pat: &[char], pi: usize) -> Option<(Atom, usize)> {
    if pi >= pat.len() {
        return None;
    }
    if pat[pi] == '\\' && pi + 1 < pat.len() {
        let n = pat[pi + 1];
        if n == 'b' {
            return None;
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
            return None;
        }
        return Some((
            Atom::Class { neg, chars, ranges },
            i + 1,
        ));
    }
    Some((Atom::Lit(pat[pi]), pi + 1))
}

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

fn is_terminal(tool: &str) -> bool {
    let lower = tool.to_ascii_lowercase();
    lower.starts_with("terminal")
        || lower.starts_with("term_")
        || lower.ends_with("__bash")
        || lower.ends_with("__exec")
}

pub fn is_allowed(allowlist_text: &str, tool_name: &str, command: &str) -> bool {
    if allowlist_text.trim().is_empty() {
        return false;
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

pub const DEFAULT_ALLOWLIST: &str = r#"
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
