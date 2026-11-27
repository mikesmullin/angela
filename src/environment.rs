use std::sync::OnceLock;

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

pub fn ordinal(day: u32) -> &'static str {
    match (day % 10, day % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    }
}

fn local_parts(epoch: i64) -> Option<(libc::tm, String)> {
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
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

pub fn long_time(epoch: i64) -> String {
    let Some((tm, zone)) = local_parts(epoch) else {
        return String::new();
    };
    let day = tm.tm_mday.max(0) as u32;
    let hour24 = tm.tm_hour.clamp(0, 23);
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
    Some(unsafe { libc::timegm(&mut tm) })
}

pub fn clock(epoch: i64) -> String {
    match local_parts(epoch) {
        Some((tm, _)) => format!("{:02}:{:02}", tm.tm_hour.clamp(0, 23), tm.tm_min.clamp(0, 59)),
        None => String::new(),
    }
}

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

pub fn day_key(epoch: i64) -> (i32, i32) {
    match local_parts(epoch) {
        Some((tm, _)) => (tm.tm_year, tm.tm_yday),
        None => (0, 0),
    }
}

fn article(word: &str) -> &'static str {
    match word.chars().next().map(|c| c.to_ascii_lowercase()) {
        Some('a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    }
}

pub fn section(epoch: i64, location: &str, user_name: &str, sandbox: &str) -> String {
    let mut out = String::from("\n## Environment\n\n");
    let sandbox = sandbox.trim();
    if sandbox.is_empty() {
        let os = os_name();
        out.push_str(&format!("You are on {} {os} system.\n", article(os)));
    } else {
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
