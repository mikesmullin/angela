//! The `## Environment` section of a system prompt: where the agent is and
//! what time it is.
//!
//! Both are things a model cannot know and will otherwise guess at — and
//! guess confidently. An agent asked "what time is it" answers from its
//! training data or refuses; one with a shell tool runs `apt` on a machine
//! that has `pacman`.
//!
//! The time is rendered to the **minute**, deliberately. It is appended to
//! the system prompt, so every change to it invalidates the provider's
//! prompt cache for the whole request; at minute resolution a round of
//! several agents usually shares one value, where seconds would guarantee a
//! miss on every single turn.

use std::sync::OnceLock;

/// What to call this machine's operating system.
///
/// On Linux the distribution, not just "linux": an agent with a shell tool
/// needs to know whether that is `apt` or `pacman`, and the kernel name does
/// not say. Read once — a machine does not change distribution mid-session.
pub fn os_name() -> &'static str {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        if let Ok(text) = std::fs::read_to_string("/etc/os-release") {
            for line in text.lines() {
                if let Some(v) = line.strip_prefix("PRETTY_NAME=") {
                    let v = v.trim().trim_matches('"').trim();
                    if !v.is_empty() {
                        return v.to_string();
                    }
                }
            }
        }
        match std::env::consts::OS {
            "macos" => "macOS".to_string(),
            "windows" => "Windows".to_string(),
            other => other.to_string(),
        }
    })
}

const DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// `1st`, `2nd`, `3rd`, `4th` … and `11th`/`12th`/`13th`, which are the
/// three the naive rule gets wrong.
pub fn ordinal(day: u32) -> &'static str {
    match (day % 10, day % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    }
}

/// The local wall clock, from the platform itself.
///
/// `localtime_r` rather than a date library because of one field: `tm_zone`,
/// the zone *abbreviation* ("MST"). chrono's `Local` formats `%Z` as a
/// numeric offset, which is not what a person reads on a clock. The call is
/// per-use, so a session running across a DST change says the right thing
/// afterwards.
fn local_parts(epoch: i64) -> Option<(libc::tm, String)> {
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: `localtime_r` fills the caller's `tm` and is the thread-safe
    // form; `tm_zone` points into static storage owned by libc.
    let ok = unsafe { !libc::localtime_r(&epoch as *const i64, &mut tm).is_null() };
    if !ok {
        return None;
    }
    let zone = if tm.tm_zone.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(tm.tm_zone) }
            .to_string_lossy()
            .into_owned()
    };
    Some((tm, zone))
}

/// `Sat, Nov 14th, 2026 @ 5:37PM MST`
pub fn long_time(epoch: i64) -> String {
    let Some((tm, zone)) = local_parts(epoch) else {
        return String::new();
    };
    let day = tm.tm_mday.max(0) as u32;
    let hour24 = tm.tm_hour.clamp(0, 23);
    // 0 and 12 both read as 12 on a clock face.
    let hour12 = match hour24 % 12 {
        0 => 12,
        h => h,
    };
    let meridiem = if hour24 < 12 { "AM" } else { "PM" };
    let wday = DAYS[tm.tm_wday.clamp(0, 6) as usize];
    let month = MONTHS[tm.tm_mon.clamp(0, 11) as usize];
    let year = tm.tm_year + 1900;
    let mut out = format!(
        "{wday}, {month} {day}{}, {year} @ {hour12}:{:02}{meridiem}",
        ordinal(day),
        tm.tm_min.clamp(0, 59),
    );
    if !zone.is_empty() {
        out.push(' ');
        out.push_str(&zone);
    }
    out
}

pub fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Seconds since the epoch for an ISO-8601 UTC timestamp, as the store
/// writes them (`2026-11-14T23:50:35.333Z`).
///
/// Hand-parsed rather than pulling in a date library: the input is one fixed
/// shape this codebase produces itself, and `timegm` does the calendar
/// arithmetic. Returns `None` on anything that does not match, so a
/// malformed timestamp loses its clock rather than its line.
pub fn epoch_from_iso(iso: &str) -> Option<i64> {
    let b = iso.as_bytes();
    if b.len() < 19 || b[4] != b'-' || b[7] != b'-' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let num = |from: usize, to: usize| iso.get(from..to)?.parse::<i32>().ok();
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    tm.tm_year = num(0, 4)? - 1900;
    tm.tm_mon = num(5, 7)? - 1;
    tm.tm_mday = num(8, 10)?;
    tm.tm_hour = num(11, 13)?;
    tm.tm_min = num(14, 16)?;
    tm.tm_sec = num(17, 19)?;
    // SAFETY: `timegm` reads the struct we just filled. It is UTC-based,
    // which is what the stored timestamp is.
    Some(unsafe { libc::timegm(&mut tm) })
}

/// `14:00` — the local wall clock, 24-hour.
pub fn clock(epoch: i64) -> String {
    match local_parts(epoch) {
        Some((tm, _)) => format!("{:02}:{:02}", tm.tm_hour.clamp(0, 23), tm.tm_min.clamp(0, 59)),
        None => String::new(),
    }
}

/// `Fri, Nov 13` — the local calendar day, for a run's header.
pub fn day_label(epoch: i64) -> String {
    match local_parts(epoch) {
        Some((tm, _)) => format!(
            "{}, {} {}",
            DAYS[tm.tm_wday.clamp(0, 6) as usize],
            MONTHS[tm.tm_mon.clamp(0, 11) as usize],
            tm.tm_mday.max(0),
        ),
        None => String::new(),
    }
}

/// Which local day an instant falls on, for spotting the boundary.
///
/// Year and day-of-year rather than the formatted label: two days a year
/// apart render identically as "Fri, Nov 13", and comparing the strings
/// would miss the change.
pub fn day_key(epoch: i64) -> (i32, i32) {
    match local_parts(epoch) {
        Some((tm, _)) => (tm.tm_year, tm.tm_yday),
        None => (0, 0),
    }
}

/// `a` or `an`, by the following word's first letter.
///
/// A vowel letter is the wrong rule in general ("a unicorn"), but right for
/// every OS name that actually turns up — Arch, Ubuntu, openSUSE, Alpine —
/// and "a Arch Linux system" is wrong in a way a reader trips over.
fn article(word: &str) -> &'static str {
    match word.chars().next().map(|c| c.to_ascii_lowercase()) {
        Some('a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    }
}

/// The section appended to a system prompt.
///
/// `location` and `user_name` come from the app's settings and are each
/// **omitted when unset** rather than written empty: "Your current
/// geographic location: ." is a worse answer than no answer, and a model
/// will try to make sense of it.
pub fn section(epoch: i64, location: &str, user_name: &str, sandbox: &str) -> String {
    let mut out = String::from("\n## Environment\n\n");
    let sandbox = sandbox.trim();
    if sandbox.is_empty() {
        let os = os_name();
        out.push_str(&format!("You are on {} {os} system.\n", article(os)));
    } else {
        // The host's distribution is the wrong answer in a sandboxed
        // session, and wrong in the exact way this module exists to
        // prevent: an agent told it is on Arch runs `pacman` inside a
        // container that has `apk`. Saying which box, and where the
        // session's checkout is mounted, is also the difference between a
        // working `bash` call and one that starts by hunting for the files.
        out.push_str(&format!(
            "You are working inside the sandbox `{sandbox}` — an isolated \
             container, not this user's machine. Your file and shell tools \
             act in there and nowhere else; the host is not reachable from \
             this conversation. The session's git checkout is mounted \
             read-write at `{}`, which is where the work is. The image is a \
             minimal Linux (Alpine, so `apk` and BusyBox, not `apt` or \
             `pacman`); don't assume a tool is installed because it is \
             common.\n",
            crate::harness::SANDBOX_MOUNT,
        ));
    }
    out.push_str(&format!(
        "The current time is {} ({}).\n",
        long_time(epoch),
        epoch,
    ));
    let location = location.trim();
    if !location.is_empty() {
        out.push_str(&format!("Your current geographic location: {location}.\n"));
    }
    let user_name = user_name.trim();
    if !user_name.is_empty() {
        out.push_str(&format!(
            "The user (human operator)'s name is {user_name}.\n"
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an epoch for a given local wall-clock time, so the assertions
    /// below are about formatting rather than about this machine's zone.
    fn epoch_for(y: i32, mon: i32, day: i32, hour: i32, min: i32) -> i64 {
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        tm.tm_year = y - 1900;
        tm.tm_mon = mon - 1;
        tm.tm_mday = day;
        tm.tm_hour = hour;
        tm.tm_min = min;
        tm.tm_isdst = -1;
        unsafe { libc::mktime(&mut tm) }
    }

    #[test]
    fn the_long_form_reads_like_a_clock() {
        let t = long_time(epoch_for(2026, 11, 14, 17, 37));
        // Sat, Nov 14th, 2026 @ 5:37PM <zone>
        assert!(t.starts_with("Sat, Nov 14th, 2026 @ 5:37PM "), "{t}");
        // The zone abbreviation, not a numeric offset — that is the whole
        // reason this uses `localtime_r` rather than a date library.
        let zone = t.rsplit(' ').next().unwrap();
        assert!(!zone.is_empty());
        assert!(
            zone.chars().all(|c| c.is_ascii_alphabetic()),
            "expected an abbreviation like MST, got {zone:?}"
        );
    }

    #[test]
    fn noon_and_midnight_both_read_as_twelve() {
        assert!(long_time(epoch_for(2026, 11, 14, 0, 5)).contains("@ 12:05AM"));
        assert!(long_time(epoch_for(2026, 11, 14, 12, 5)).contains("@ 12:05PM"));
        assert!(long_time(epoch_for(2026, 11, 14, 13, 0)).contains("@ 1:00PM"));
        // Minutes keep their leading zero; the hour does not.
        assert!(long_time(epoch_for(2026, 11, 14, 9, 0)).contains("@ 9:00PM")
            || long_time(epoch_for(2026, 11, 14, 21, 0)).contains("@ 9:00PM"));
    }

    #[test]
    fn ordinals_handle_the_teens() {
        for (d, want) in [
            (1, "st"), (2, "nd"), (3, "rd"), (4, "th"),
            // The three the naive `% 10` rule gets wrong.
            (11, "th"), (12, "th"), (13, "th"),
            (21, "st"), (22, "nd"), (23, "rd"), (31, "st"),
        ] {
            assert_eq!(ordinal(d), want, "day {d}");
        }
    }

    #[test]
    fn the_section_says_where_and_when() {
        let s = section(epoch_for(2026, 11, 14, 17, 37), "Boise, Idaho", "Bob", "");
        assert!(s.contains("## Environment"));
        assert!(s.contains(" system."));
        // "a Arch Linux system" is wrong in a way a reader trips over.
        assert!(
            s.contains("You are on a ") || s.contains("You are on an "),
            "{s}"
        );
        for (word, want) in [("Arch Linux", "an"), ("Ubuntu", "an"), ("macOS", "a"),
                             ("Windows", "a"), ("openSUSE", "an"), ("Debian", "a")] {
            assert_eq!(article(word), want, "{word}");
        }
        assert!(s.contains("The current time is Sat, Nov 14th, 2026 @ 5:37PM"));
        // The epoch too: a model that cannot parse the prose can still do
        // arithmetic on the number.
        assert!(s.contains(&format!("({})", epoch_for(2026, 11, 14, 17, 37))));
        assert!(s.contains("Your current geographic location: Boise, Idaho."));
        assert!(s.contains("The user (human operator)'s name is Bob."));
    }

    #[test]
    fn an_unset_location_or_name_is_left_out_entirely() {
        // "Your current geographic location: ." is a worse answer than no
        // answer, and a model will try to make sense of it.
        let s = section(epoch_for(2026, 11, 14, 17, 37), "", "   ", "");
        assert!(!s.contains("geographic location"), "{s}");
        assert!(!s.contains("human operator"), "{s}");
        // What is known is still said.
        assert!(s.contains("The current time is "));

        let only_where = section(epoch_for(2026, 11, 14, 17, 37), "Boise, Idaho", "", "");
        assert!(only_where.contains("geographic location: Boise, Idaho."));
        assert!(!only_where.contains("human operator"));
    }


    /// In a sandboxed session the host's distribution is the wrong answer,
    /// and wrong in the way this module exists to prevent: an agent told it
    /// is on Arch runs `pacman` in a container that has `apk`. It must also
    /// learn where the checkout is, or a correct `bash` tool spends its
    /// first calls hunting for the files.
    #[test]
    fn a_sandboxed_session_describes_the_box_not_the_host() {
        let s = section(epoch_for(2026, 11, 14, 17, 37), "Boise, Idaho", "Bob", "sandbox2-abc");
        assert!(s.contains("sandbox2-abc"), "{s}");
        assert!(s.contains(crate::harness::SANDBOX_MOUNT), "{s}");
        assert!(!s.contains("You are on a "), "{s}");
        assert!(!s.contains(os_name()), "{s}");
        // Everything else about "where and when" still holds.
        assert!(s.contains("The current time is Sat, Nov 14th, 2026 @ 5:37PM"), "{s}");
        assert!(s.contains("Your current geographic location: Boise, Idaho."), "{s}");
        assert!(s.contains("The user (human operator)'s name is Bob."), "{s}");
    }

    #[test]
    fn an_iso_timestamp_round_trips_through_the_parser() {
        let epoch = epoch_for(2026, 11, 14, 17, 37);
        // The shape the store writes.
        let iso = crate::session_store::now_iso();
        assert!(epoch_from_iso(&iso).is_some(), "{iso}");
        // Round-trips against the platform's own UTC conversion, rather
        // than against a magic number nobody can check by eye.
        let a = epoch_from_iso("2026-11-13T21:00:00.000Z").unwrap();
        let b = epoch_from_iso("2026-11-13T21:01:00.000Z").unwrap();
        assert_eq!(b - a, 60);
        let day = epoch_from_iso("2026-11-14T21:00:00.000Z").unwrap();
        assert_eq!(day - a, 86_400);
        // It is UTC in: the same wall-clock string is the same instant
        // wherever this runs.
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        unsafe { libc::gmtime_r(&a as *const i64, &mut tm) };
        assert_eq!((tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour), (2026, 11, 13, 21));
        // Malformed input loses its clock, not its line.
        for bad in ["", "yesterday", "2026-11-13", "2026/11/13T21:00:00Z"] {
            assert!(epoch_from_iso(bad).is_none(), "{bad}");
        }
        let _ = epoch;
    }

    #[test]
    fn the_clock_and_the_day_label_read_as_a_transcript_would() {
        let e = epoch_for(2026, 11, 13, 14, 5);
        assert_eq!(clock(e), "14:05");
        assert_eq!(clock(epoch_for(2026, 11, 13, 9, 0)), "09:00");
        assert_eq!(clock(epoch_for(2026, 11, 13, 0, 0)), "00:00");
        assert_eq!(day_label(e), "Fri, Nov 13");
    }

    #[test]
    fn the_day_key_changes_at_midnight_and_not_before() {
        let late = epoch_for(2026, 11, 13, 23, 59);
        let early = epoch_for(2026, 11, 14, 0, 1);
        assert_ne!(day_key(late), day_key(early));
        assert_eq!(day_key(late), day_key(epoch_for(2026, 11, 13, 0, 1)));
        // A year apart renders the same label, so the key cannot be the
        // label.
        assert_eq!(day_label(epoch_for(2025, 11, 13, 12, 0))[5..], day_label(late)[5..]);
        assert_ne!(day_key(epoch_for(2025, 11, 13, 12, 0)), day_key(late));
    }

    #[test]
    fn the_os_is_named_once_and_is_not_empty() {
        let a = os_name();
        assert!(!a.trim().is_empty());
        // Cached: a machine does not change distribution mid-session.
        assert!(std::ptr::eq(a, os_name()));
    }

    #[test]
    fn the_time_is_rendered_to_the_minute_not_the_second() {
        // Appended to the system prompt, so every change invalidates the
        // provider's prompt cache for the whole request. At minute
        // resolution a round of several agents usually shares one value.
        let base = epoch_for(2026, 11, 14, 17, 37);
        assert_eq!(long_time(base), long_time(base + 1));
        assert_eq!(long_time(base), long_time(base + 59));
        assert_ne!(long_time(base), long_time(base + 60));
    }
}
