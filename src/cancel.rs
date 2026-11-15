//! Stopping a turn that is already talking to a model.
//!
//! A flag the turn loop polls would not be enough. Almost all of a turn's
//! time is spent inside **one HTTP request** to the provider: the model is
//! thinking, nothing is being emitted, and there is no loop iteration to
//! check a flag in. Stopping has to reach that request, or "stop" means
//! "stop after the thing you are waiting for finishes anyway" — which is
//! what a user pressing the button is trying not to pay for.
//!
//! What reaches it is a **drop**. `tokio::select!` drops the branch that
//! loses the race; dropping a `reqwest` future closes its connection; and a
//! closed connection is a cancelled request. So the switch here is a thing
//! to *await*, not only a thing to read — [`Cancel::stopped_fut`] is what a
//! turn races its own work against.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a stop that found nothing to stop stays armed.
///
/// A stop can beat the turn it is meant for. The browser aborts its fetch
/// and posts the stop in the same breath, while the turn registers itself a
/// moment later — the session id it registers under is not known until the
/// session is resolved. Arming the switch for that window makes the race
/// come out the way the user meant it.
///
/// It expires because the other way to lose this race is for a stop to
/// arrive when nothing is running at all, and a switch left armed forever
/// would cancel the next turn instead — the user's *next message*, which
/// they never asked to stop.
const ARMED_FOR: Duration = Duration::from_secs(5);

/// One turn's stop switch.
pub struct Cancel {
    stopped: AtomicBool,
    notify: tokio::sync::Notify,
}

impl Cancel {
    fn new(stopped: bool) -> Self {
        Cancel {
            stopped: AtomicBool::new(stopped),
            notify: tokio::sync::Notify::new(),
        }
    }

    /// A switch nobody holds: work that cannot be stopped.
    ///
    /// For call paths with no turn behind them — a test, or a model call
    /// made outside a conversation — so they can use the same cancellable
    /// helpers without inventing a registry to hold one entry.
    pub fn never() -> Arc<Self> {
        Arc::new(Cancel::new(false))
    }

    /// Throw the switch. Idempotent: two presses are one stop.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        // `notify_waiters`, not `notify_one`: a group turn can have several
        // waits over the life of one switch, and every one of them is
        // waiting for the same single event.
        self.notify.notify_waiters();
    }

    /// Whether this turn has been asked to stop.
    pub fn stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// Resolves once the switch is thrown — including if it already was.
    ///
    /// The waiter is registered *before* the flag is read. `notified()` does
    /// not subscribe until it is polled, so in the other order a stop landing
    /// between the read and the first poll is missed by both, and the wait
    /// never ends — the turn would run to completion with the user looking at
    /// a stopped composer.
    pub async fn stopped_fut(&self) {
        let fut = self.notify.notified();
        tokio::pin!(fut);
        fut.as_mut().enable();
        if self.stopped() {
            return;
        }
        fut.await;
    }
}

/// The turns that can still be stopped, by session id.
///
/// Keyed by session rather than by tab because that is what the user means
/// by "stop": the conversation in front of them. (The tab id the client
/// sends is the constant `"1"`, so it could not tell two of them apart
/// anyway.)
#[derive(Default)]
pub struct Cancels {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// Turns running right now.
    live: HashMap<String, Arc<Cancel>>,
    /// Stops that arrived before the turn they were meant for.
    pending: HashMap<String, Instant>,
}

impl Cancels {
    /// Register the turn now running for `session` and return its switch.
    ///
    /// If a stop is already pending for that session the switch comes back
    /// **already thrown**, so the turn ends at its first cancellation point
    /// rather than opening a request to a model the user has just said they
    /// do not want.
    pub fn begin(&self, session: &str) -> Arc<Cancel> {
        let mut st = self.state.lock().unwrap();
        let armed = match st.pending.remove(session) {
            Some(at) => at.elapsed() < ARMED_FOR,
            None => false,
        };
        let c = Arc::new(Cancel::new(armed));
        // Last writer wins. A second turn on one session should not be made
        // unstoppable by the first turn's entry sitting in the map.
        st.live.insert(session.to_string(), c.clone());
        c
    }

    /// Deregister when the turn ends.
    ///
    /// Identity-checked rather than removed by key: another turn on the same
    /// session may have started in the meantime, and dropping *its* entry
    /// would leave the running turn with no way to be stopped.
    pub fn finish(&self, session: &str, mine: &Arc<Cancel>) {
        let mut st = self.state.lock().unwrap();
        if let Some(cur) = st.live.get(session) {
            if Arc::ptr_eq(cur, mine) {
                st.live.remove(session);
            }
        }
    }

    /// Throw the switch for `session`'s turn.
    ///
    /// Returns whether a turn was actually running. `false` is not a
    /// failure — the turn may have finished a moment earlier, and the stop
    /// stays armed briefly in case it is instead about to start.
    pub fn stop(&self, session: &str) -> bool {
        let mut st = self.state.lock().unwrap();
        // Expire stale arms while we are holding the lock; this is the only
        // path that adds them, so it is the only place they can pile up.
        st.pending.retain(|_, at| at.elapsed() < ARMED_FOR);
        match st.live.get(session) {
            Some(c) => {
                c.stop();
                true
            }
            None => {
                st.pending.insert(session.to_string(), Instant::now());
                false
            }
        }
    }

    /// Stop every live turn for `session` and forget any pending stop.
    ///
    /// For a session being deleted: a turn whose conversation is gone has
    /// nowhere to write what it is paying for.
    pub fn forget(&self, session: &str) {
        let mut st = self.state.lock().unwrap();
        st.pending.remove(session);
        if let Some(c) = st.live.remove(session) {
            c.stop();
        }
    }

    /// How many turns could be stopped right now. Tests and diagnostics.
    pub fn live_count(&self) -> usize {
        self.state.lock().unwrap().live.len()
    }
}

/// What came of a piece of work that could have been stopped.
pub enum Ran<T> {
    /// It finished on its own.
    Done(T),
    /// Somebody pressed stop. The work was dropped where it stood.
    Stopped,
}

/// `rt.block_on(fut)`, abandoning `fut` if `stop` is thrown.
///
/// The abandoning is the whole point: `select!` drops the losing branch, and
/// dropping the future drops the `reqwest` request inside it, which closes
/// the connection to the provider. That is what makes this a cancelled HTTP
/// request rather than one whose answer nobody reads.
pub fn block_on<T>(
    rt: &tokio::runtime::Runtime,
    stop: &Cancel,
    fut: impl std::future::Future<Output = T>,
) -> Ran<T> {
    rt.block_on(async {
        tokio::select! {
            // `biased` so the stop branch is polled first. A turn that has
            // already been told to stop then never opens the request at
            // all, instead of opening one and dropping it a moment later.
            biased;
            _ = stop.stopped_fut() => Ran::Stopped,
            v = fut => Ran::Done(v),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
    }

    #[test]
    fn a_stop_reaches_a_turn_that_is_already_waiting() {
        let cancels = Cancels::default();
        let c = cancels.begin("s1");
        assert!(!c.stopped());
        assert!(cancels.stop("s1"), "a live turn was there to stop");
        assert!(c.stopped());
    }

    #[test]
    fn a_stop_that_beats_its_turn_is_still_honoured() {
        let cancels = Cancels::default();
        // The browser aborted and posted the stop before the turn got as far
        // as registering itself.
        assert!(!cancels.stop("s1"), "nothing was running yet");
        let c = cancels.begin("s1");
        assert!(c.stopped(), "the turn starts already stopped");
        // Consumed, not sticky: the *next* turn on that session is the
        // user's next message and must run.
        let c2 = cancels.begin("s1");
        assert!(!c2.stopped());
    }

    #[test]
    fn finishing_a_turn_does_not_unregister_the_one_that_replaced_it() {
        let cancels = Cancels::default();
        let first = cancels.begin("s1");
        let second = cancels.begin("s1");
        // The first turn ends late, after the second has taken the slot.
        cancels.finish("s1", &first);
        assert_eq!(cancels.live_count(), 1, "the second turn is still listed");
        assert!(cancels.stop("s1"));
        assert!(second.stopped(), "stopping reaches the turn that is running");
        cancels.finish("s1", &second);
        assert_eq!(cancels.live_count(), 0);
    }

    #[test]
    fn a_wait_ends_whether_the_stop_comes_before_or_during_it() {
        let rt = rt();
        // Thrown first: the wait must not hang because it missed the signal.
        let c = Arc::new(Cancel::new(false));
        c.stop();
        rt.block_on(async { c.stopped_fut().await });

        // Thrown while the wait is parked.
        let c = Arc::new(Cancel::new(false));
        let c2 = c.clone();
        rt.block_on(async move {
            let waiter = tokio::spawn(async move { c2.stopped_fut().await });
            // Yield enough for the waiter to park on the notify.
            tokio::time::sleep(Duration::from_millis(20)).await;
            c.stop();
            waiter.await.unwrap();
        });
    }

    #[test]
    fn work_already_stopped_never_starts() {
        let rt = rt();
        let c = Arc::new(Cancel::new(false));
        c.stop();
        let started = Arc::new(AtomicBool::new(false));
        let flag = started.clone();
        let out = block_on(&rt, &c, async move {
            flag.store(true, Ordering::SeqCst);
            7
        });
        assert!(matches!(out, Ran::Stopped));
        assert!(
            !started.load(Ordering::SeqCst),
            "the future was never polled, so no request was opened"
        );
    }

    #[test]
    fn work_that_outlives_a_stop_is_dropped_mid_await() {
        let rt = rt();
        let c = Arc::new(Cancel::new(false));
        let c2 = c.clone();
        // Stand-in for a provider request: a long wait that would finish if
        // nobody stopped it.
        let finished = Arc::new(AtomicBool::new(false));
        let flag = finished.clone();
        let out = block_on(&rt, &c, async move {
            // Thrown from inside, which is the only way to reach the switch
            // on a single-threaded runtime.
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                c2.stop();
            });
            tokio::time::sleep(Duration::from_secs(30)).await;
            flag.store(true, Ordering::SeqCst);
            7
        });
        assert!(matches!(out, Ran::Stopped));
        assert!(!finished.load(Ordering::SeqCst), "the wait was abandoned");
    }

    #[test]
    fn work_that_finishes_first_comes_back_whole() {
        let rt = rt();
        let c = Arc::new(Cancel::new(false));
        let out = block_on(&rt, &c, async { 7 });
        assert!(matches!(out, Ran::Done(7)));
    }

    #[test]
    fn deleting_a_session_stops_its_turn() {
        let cancels = Cancels::default();
        let c = cancels.begin("s1");
        cancels.forget("s1");
        assert!(c.stopped());
        assert_eq!(cancels.live_count(), 0);
    }
}
