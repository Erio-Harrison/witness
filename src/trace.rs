//! Ground-truth trace and injected waits, for checking tools that attribute latency from outside.
//!
//! The trace has two sources, on one clock (`now_ns`, CLOCK_MONOTONIC): host events recorded here
//! (request parsed, tokenized, admitted to a slot, each unit of GPU work, each token, each socket
//! write) and the core's record of every CUDA call it makes (core/witness_core.h). `drain_jsonl`
//! returns both, oldest first, one JSON object per line.
//!
//! Injected waits make a known amount of a known kind of delay appear at a known place. Everything
//! is off by default; when off, each hook costs one relaxed atomic load.
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CudaRec {
    t0_ns: u64,
    t1_ns: u64,
    cpu_ns: u64,
    site: u32,
    injected_us: u32,
}

extern "C" {
    fn wk_now_ns() -> u64;
    fn wk_trace_enable(on: i32);
    fn wk_trace_drain(out: *mut CudaRec, cap: i32, dropped: *mut u64) -> i32;
    fn wk_trace_site(id: u32, out: *mut std::ffi::c_char, cap: i32) -> i32;
    fn wk_inject_cuda_delay(site: *const std::ffi::c_char, microseconds: u32);
}

struct HostEvent {
    t0: u64,
    t1: u64,
    kind: &'static str,
    req: u64,
    a: i64,
    b: i64,
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static EVENTS: Mutex<Vec<HostEvent>> = Mutex::new(Vec::new());

/// Busy-wait on the GPU thread before each unit of work: CPU the thread is seen to burn.
pub static UNIT_SPIN_US: AtomicU64 = AtomicU64::new(0);
/// Sleep on the GPU thread before each unit of work: time the thread is seen off-CPU.
pub static UNIT_SLEEP_US: AtomicU64 = AtomicU64::new(0);
/// Sleep before each socket write of a response.
pub static SOCKET_DELAY_US: AtomicU64 = AtomicU64::new(0);
/// Streamed responses hold tokens back and write this many per socket write.
pub static COALESCE_TOKENS: AtomicU64 = AtomicU64::new(0);

pub fn now_ns() -> u64 {
    unsafe { wk_now_ns() }
}

pub fn enabled() -> bool {
    ENABLED.load(Relaxed)
}

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Relaxed);
    unsafe { wk_trace_enable(on as i32) };
}

/// Sleep `microseconds` inside every CUDA call of the core whose text contains `site`.
pub fn inject_cuda_delay(site: &str, microseconds: u32) {
    let site = CString::new(site).unwrap_or_default();
    unsafe { wk_inject_cuda_delay(site.as_ptr(), microseconds) };
}

/// A host event spanning `t0..now`. `a` and `b` are kind-specific numbers (see `drain_jsonl`).
pub fn span(kind: &'static str, req: u64, t0: u64, a: i64, b: i64) {
    if enabled() {
        EVENTS.lock().unwrap().push(HostEvent { t0, t1: now_ns(), kind, req, a, b });
    }
}

/// A host event at one instant.
pub fn mark(kind: &'static str, req: u64, a: i64, b: i64) {
    if enabled() {
        let t = now_ns();
        EVENTS.lock().unwrap().push(HostEvent { t0: t, t1: t, kind, req, a, b });
    }
}

/// The injected waits of the GPU thread, applied before a unit of work.
pub fn unit_injection() {
    let spin = UNIT_SPIN_US.load(Relaxed);
    if spin > 0 {
        let t0 = now_ns();
        while now_ns() - t0 < spin * 1000 {
            std::hint::spin_loop();
        }
        mark("injected_spin", 0, spin as i64, 0);
    }
    let sleep = UNIT_SLEEP_US.load(Relaxed);
    if sleep > 0 {
        std::thread::sleep(std::time::Duration::from_micros(sleep));
        mark("injected_sleep", 0, sleep as i64, 0);
    }
}

/// Everything recorded since the last drain, oldest first, as JSON lines.
///
/// Host events: {"src":"host","kind",...,"req","t0","t1","a","b"} with kinds
///   http_request (a = body bytes), tokenize (a = tokens), admit (a = slot, b = cached tokens),
///   prefill (a = slot, b = tokens), decode (a = rows), token (a = token id, b = index),
///   socket_write (a = bytes, b = tokens carried), done (a = generated tokens),
///   injected_spin / injected_sleep / injected_socket_delay (a = microseconds).
/// CUDA calls: {"src":"cuda","site","t0","t1","cpu_ns","injected_us"}.
pub fn drain_jsonl() -> String {
    let mut lines: Vec<(u64, String)> = Vec::new();
    for e in EVENTS.lock().unwrap().drain(..) {
        lines.push((e.t0, format!(r#"{{"src":"host","kind":"{}","req":{},"t0":{},"t1":{},"a":{},"b":{}}}"#, e.kind, e.req, e.t0, e.t1, e.a, e.b)));
    }
    let mut names: Vec<String> = Vec::new();
    let mut buf = vec![CudaRec::default(); 4096];
    let mut dropped_total = 0u64;
    loop {
        let mut dropped = 0u64;
        let n = unsafe { wk_trace_drain(buf.as_mut_ptr(), buf.len() as i32, &mut dropped) } as usize;
        dropped_total += dropped;
        for r in &buf[..n] {
            while names.len() <= r.site as usize {
                let mut name = [0 as std::ffi::c_char; 96];
                let len = unsafe { wk_trace_site(names.len() as u32, name.as_mut_ptr(), name.len() as i32) }.max(0) as usize;
                names.push(name[..len].iter().map(|&c| c as u8 as char).collect());
            }
            lines.push((r.t0_ns, format!(r#"{{"src":"cuda","site":"{}","t0":{},"t1":{},"cpu_ns":{},"injected_us":{}}}"#,
                names[r.site as usize].trim(), r.t0_ns, r.t1_ns, r.cpu_ns, r.injected_us)));
        }
        if n < buf.len() {
            break;
        }
    }
    lines.sort_by_key(|l| l.0);
    let mut out = String::new();
    if dropped_total > 0 {
        out.push_str(&format!("{{\"src\":\"trace\",\"dropped_cuda_records\":{dropped_total}}}\n"));
    }
    for (_, line) in lines {
        out.push_str(&line);
        out.push('\n');
    }
    out
}
