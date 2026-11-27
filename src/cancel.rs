use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const ARMED_FOR: Duration = Duration::from_secs(5);

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

    pub fn never() -> Arc<Self> {
        Arc::new(Cancel::new(false))
    }

    pub fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    pub fn stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

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

#[derive(Default)]
pub struct Cancels {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    live: HashMap<String, Arc<Cancel>>,
    pending: HashMap<String, Instant>,
}

impl Cancels {
    pub fn begin(&self, session: &str) -> Arc<Cancel> {
        let mut st = self.state.lock().unwrap();
        let armed = match st.pending.remove(session) {
            Some(at) => at.elapsed() < ARMED_FOR,
            None => false,
        };
        let c = Arc::new(Cancel::new(armed));
        st.live.insert(session.to_string(), c.clone());
        c
    }

    pub fn finish(&self, session: &str, mine: &Arc<Cancel>) {
        let mut st = self.state.lock().unwrap();
        if let Some(cur) = st.live.get(session) {
            if Arc::ptr_eq(cur, mine) {
                st.live.remove(session);
            }
        }
    }

    pub fn stop(&self, session: &str) -> bool {
        let mut st = self.state.lock().unwrap();
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

    pub fn forget(&self, session: &str) {
        let mut st = self.state.lock().unwrap();
        st.pending.remove(session);
        if let Some(c) = st.live.remove(session) {
            c.stop();
        }
    }

    pub fn live_count(&self) -> usize {
        self.state.lock().unwrap().live.len()
    }
}

pub enum Ran<T> {
    Done(T),
    Stopped,
}

pub fn block_on<T>(
    rt: &tokio::runtime::Runtime,
    stop: &Cancel,
    fut: impl std::future::Future<Output = T>,
) -> Ran<T> {
    rt.block_on(async {
        tokio::select! {
            biased;
            _ = stop.stopped_fut() => Ran::Stopped,
            v = fut => Ran::Done(v),
        }
    })
}
