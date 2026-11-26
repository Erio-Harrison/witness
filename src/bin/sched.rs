//! Checks and timings of the scheduler (prefix cache, slot copies, concurrency, cancellation).
//!
//! usage: sched check|bench WEIGHTS.wit DATA_DIR [--no-speculate]
//!
//! DATA_DIR holds code.i32 and special.i32 from tools/tokens.py. Prompts are fill-in-the-middle
//! prompts cut from real code: <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|>.
//!
//! check: every scenario's output is compared with a plain run of the same prompt on a bare engine
//! (no cache, no other requests). Reusing a cached prefix or sharing a step changes the arithmetic
//! slightly (wide vs narrow path, chunk boundaries), so the outputs may part ways, but only at a
//! token where the plain run's own top-2 logit margin is under 0.5.
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc::Receiver;
use witness::sched::{Event, Finish, Request, SchedOptions, Scheduler, Stats};
use witness::{Engine, Options, Row};

const MARGIN: f32 = 0.5;
const CHUNK: usize = 512;

fn read_i32(path: &Path) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect()
}

struct Data {
    code: Vec<i32>,
    fim_prefix: i32,
    fim_suffix: i32,
    fim_middle: i32,
    stop: Vec<i32>,
}

impl Data {
    fn load(dir: &Path) -> Data {
        let sp = read_i32(&dir.join("special.i32"));
        Data { code: read_i32(&dir.join("code.i32")), fim_prefix: sp[0], fim_suffix: sp[1], fim_middle: sp[2], stop: sp[3..].to_vec() }
    }

    /// A FIM prompt whose prefix is `prefix` tokens of code ending at `cursor` and whose suffix is
    /// the `suffix` tokens that follow a gap of 40 tokens (the part to be filled in).
    fn fim(&self, cursor: usize, prefix: usize, suffix: usize) -> Vec<i32> {
        let mut p = vec![self.fim_prefix];
        p.extend_from_slice(&self.code[cursor - prefix..cursor]);
        p.push(self.fim_suffix);
        p.extend_from_slice(&self.code[cursor + 40..cursor + 40 + suffix]);
        p.push(self.fim_middle);
        p
    }
}

fn top_margin(v: &[f32]) -> (i32, f32) {
    let (mut best, mut second) = (0, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > v[best] {
            second = v[best];
            best = i;
        } else if x > second {
            second = x;
        }
    }
    (best as i32, v[best] - second)
}

/// Greedy continuation on a bare engine, with the top-2 margin behind every token.
fn plain(engine: &mut Engine, prompt: &[i32], gen: usize) -> (Vec<i32>, Vec<f32>) {
    let mut logits = Vec::new();
    for (i, piece) in prompt.chunks(CHUNK).enumerate() {
        engine.prefill(0, (i * CHUNK) as i32, piece, 0, Some(&mut logits), None).expect("prefill");
    }
    let (mut tokens, mut margins) = (Vec::new(), Vec::new());
    for t in 0..gen {
        let (token, margin) = top_margin(&logits);
        tokens.push(token);
        margins.push(margin);
        if t + 1 < gen {
            engine.step_fast(&[Row { token, slot: 0, pos: (prompt.len() + t) as i32 }], Some(&mut logits)).expect("step");
        }
    }
    (tokens, margins)
}

fn collect(rx: Receiver<Event>) -> (Vec<i32>, Stats) {
    let mut tokens = Vec::new();
    loop {
        match rx.recv().expect("scheduler dropped the request") {
            Event::Started { .. } => {}
            Event::Token(t) => tokens.push(t),
            Event::Done(stats) => return (tokens, stats),
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Prints one scenario result and returns whether it is acceptable.
fn verdict(name: &str, got: &[i32], stats: &Stats, want: &(Vec<i32>, Vec<f32>), expect_cached: Option<usize>) -> bool {
    let (tokens, margins) = want;
    let n = got.len().min(tokens.len());
    let diverged = (0..n).find(|&i| got[i] != tokens[i]);
    let mut ok = stats.error.is_none();
    let outcome = match diverged {
        None if got.len() == tokens.len() => "same tokens as the plain run".to_string(),
        None => {
            ok = false;
            format!("FAIL: {} tokens, plain run has {}", got.len(), tokens.len())
        }
        Some(i) if margins[i] < MARGIN => format!("parts from the plain run at token {i}, where its margin is {:.3}", margins[i]),
        Some(i) => {
            ok = false;
            format!("FAIL: differs from the plain run at token {i}, where its margin is {:.3}", margins[i])
        }
    };
    let mut cache = format!("{} of {} prompt tokens cached{}", stats.cached_tokens, stats.prompt_tokens, if stats.copied { " (copied from a busy slot)" } else { "" });
    if let Some(want) = expect_cached {
        if stats.cached_tokens != want {
            ok = false;
            cache += &format!(" FAIL: expected {want}");
        }
    }
    if let Some(e) = &stats.error {
        cache += &format!(" FAIL: {e}");
    }
    println!("  {name}: {outcome}; {cache}; first token after {:.0} ms", stats.ttft_ms);
    ok
}

fn check(weights: &Path, data: &Data) -> bool {
    const GEN: usize = 32;
    let request = |tokens: &[i32], max_tokens: usize| Request { id: 0, tokens: tokens.to_vec(), max_tokens, stop: Vec::new() };

    // Prompts of every scenario, then their plain runs on a bare engine.
    let cold = data.fim(3000, 600, 120);
    let typing: Vec<Vec<i32>> = (0..5).map(|k| data.fim(3000 + 3 * (k + 1), 600 + 3 * (k + 1), 120)).collect();
    let parallel: Vec<Vec<i32>> = (0..4).map(|k| data.fim(9000 + 4000 * k, 300 + 150 * k, 60 + 20 * k)).collect();
    let long_a = data.fim(30000, 1200, 200);
    let long_b = data.fim(30004, 1204, 200);
    let after_cancel = data.fim(50000, 400, 80);
    let stopping = data.fim(60000, 300, 60);
    let mut all: Vec<&Vec<i32>> = vec![&cold, &long_a, &long_b, &after_cancel, &stopping];
    all.extend(&typing);
    all.extend(&parallel);
    let plain_runs: Vec<(Vec<i32>, Vec<f32>)> = {
        let mut engine = Engine::new(weights, Options { slots: 1, max_ctx: 4096, max_chunk: CHUNK as u32, kv_bf16: 1 }).expect("engine");
        all.iter().map(|p| plain(&mut engine, p, 64)).collect()
    };
    let want = |p: &Vec<i32>, n: usize| -> (Vec<i32>, Vec<f32>) {
        let (t, m) = &plain_runs[all.iter().position(|q| std::ptr::eq(*q, p)).unwrap()];
        (t[..n].to_vec(), m[..n].to_vec())
    };

    let mut ok = true;

    // Speculative decoding must not change a single token: the same requests through a scheduler
    // without it and one with it, both starting cold.
    println!("== speculative decoding off vs on, 96 tokens each");
    let spec_prompts = [&cold, &long_a, &after_cancel, &stopping, &parallel[3]];
    let run_all = |speculate: bool| -> Vec<(Vec<i32>, Stats)> {
        let sched = Scheduler::new(weights.to_path_buf(), SchedOptions { slots: 8, max_ctx: 4096, chunk: CHUNK as u32, speculate }).expect("scheduler");
        spec_prompts.iter().map(|p| collect(sched.submit(request(p, 96)))).collect()
    };
    let (plain_out, spec_out) = (run_all(false), run_all(true));
    for (k, ((a, sa), (b, sb))) in plain_out.iter().zip(&spec_out).enumerate() {
        let same = a == b;
        ok &= same && sa.guessed == 0;
        println!("  prompt {k}: {}; {} steps for {} tokens ({} guesses fed, {} right) instead of {}",
            if same { "identical tokens" } else { "FAIL: tokens differ" }, sb.steps, sb.generated, sb.guessed, sb.guessed_right, sa.steps);
    }

    let speculate = true;
    let sched = Scheduler::new(weights.to_path_buf(), SchedOptions { slots: 8, max_ctx: 4096, chunk: CHUNK as u32, speculate }).expect("scheduler");

    println!("== one request, nothing cached");
    let (got, stats) = collect(sched.submit(request(&cold, GEN)));
    ok &= verdict("cold", &got, &stats, &want(&cold, GEN), Some(0));

    println!("== typing: the prefix grows by 3 tokens per request, the suffix stays");
    for (k, prompt) in typing.iter().enumerate() {
        // Shared with the previous prompt: <|fim_prefix|> and its whole prefix.
        let (got, stats) = collect(sched.submit(request(prompt, GEN)));
        ok &= verdict(&format!("keystroke {}", k + 1), &got, &stats, &want(prompt, GEN), Some(1 + 600 + 3 * k));
    }

    println!("== four different requests at once");
    let receivers: Vec<_> = parallel.iter().map(|p| sched.submit(request(p, GEN))).collect();
    for (k, rx) in receivers.into_iter().enumerate() {
        let (got, stats) = collect(rx);
        ok &= verdict(&format!("request {}", k + 1), &got, &stats, &want(&parallel[k], GEN), None);
    }

    println!("== a request that overlaps one sharing its prefix (the user kept typing)");
    let rx_a = sched.submit(request(&long_a, 64));
    // Wait for A's first token, so that its prompt is in its slot and it is generating.
    let first = loop {
        match rx_a.recv().expect("first token") {
            Event::Started { .. } => {}
            Event::Token(t) => break t,
            Event::Done(s) => panic!("request ended early: {:?}", s.error),
        }
    };
    let rx_b = sched.submit(request(&long_b, GEN));
    let (got_b, stats_b) = collect(rx_b);
    let (mut got_a, stats_a) = collect(rx_a);
    got_a.insert(0, first);
    ok &= verdict("first request", &got_a, &stats_a, &want(&long_a, 64), None);
    ok &= verdict("second request", &got_b, &stats_b, &want(&long_b, GEN), Some(1 + 1200));
    if !stats_b.copied {
        ok = false;
        println!("  FAIL: the second request did not copy the prefix from the busy slot");
    }

    println!("== cancellation: the client goes away after 3 tokens");
    let rx = sched.submit(request(&cold, 64));
    for _ in 0..4 {
        let _ = rx.recv(); // Started, then 3 tokens
    }
    drop(rx);
    let (got, stats) = collect(sched.submit(request(&after_cancel, GEN)));
    ok &= verdict("next request", &got, &stats, &want(&after_cancel, GEN), None);

    println!("== stop tokens and limits");
    let (full, _) = want(&stopping, 64);
    let stop_at = 10;
    let stop_token = full[stop_at];
    let expected_len = full.iter().position(|&t| t == stop_token).unwrap();
    let (got, stats) = collect(sched.submit(Request { id: 0, tokens: stopping.clone(), max_tokens: 64, stop: vec![stop_token] }));
    let stopped = stats.finish == Finish::Stop && got.len() == expected_len && got[..] == full[..expected_len];
    println!("  stop token: {} after {} tokens ({:?})", if stopped { "ended before it" } else { "FAIL" }, got.len(), stats.finish);
    ok &= stopped;
    let (_, stats) = collect(sched.submit(request(&stopping, 4096)));
    let refused = stats.finish == Finish::Error;
    println!("  request longer than the context: {}", if refused { "refused" } else { "FAIL: accepted" });
    ok &= refused;
    let (got, stats) = collect(sched.submit(request(&cold, GEN)));
    ok &= verdict("request after the refusal", &got, &stats, &want(&cold, GEN), None);
    let _ = data.stop.len();
    ok
}

fn bench(weights: &Path, data: &Data, speculate: bool) {
    println!("speculative decoding {}", if speculate { "on" } else { "off" });
    let sched = Scheduler::new(weights.to_path_buf(), SchedOptions { slots: 8, max_ctx: 8192, chunk: CHUNK as u32, speculate }).expect("scheduler");
    let request = |tokens: Vec<i32>, max_tokens: usize| Request { id: 0, tokens, max_tokens, stop: Vec::new() };
    let run = |tokens: Vec<i32>, max_tokens: usize| collect(sched.submit(request(tokens, max_tokens))).1;
    run(data.fim(2000, 200, 50), 8); // warm-up

    println!("first token, nothing cached (FIM prompt, suffix 1/5 of it), median of 3:");
    for (k, len) in [128usize, 1024, 2048, 4096].into_iter().enumerate() {
        let suffix = len / 5;
        let ttft: Vec<f64> = (0..3).map(|i| run(data.fim(8000 + 9000 * k + 5000 * i, len - suffix - 3, suffix), 4).ttft_ms).collect();
        println!("  {len:5} tokens: {:7.1} ms", median(ttft));
    }

    println!("first token while typing (prefix grows by 3 tokens per request), median of 10:");
    for (prefix, suffix) in [(1500usize, 0usize), (1500, 64), (1500, 300), (3000, 300), (1500, 800)] {
        let cursor = 70000;
        run(data.fim(cursor, prefix, suffix), 4);
        let mut ttft = Vec::new();
        let mut recomputed = 0;
        for k in 1..=10 {
            let stats = run(data.fim(cursor + 3 * k, prefix + 3 * k, suffix), 4);
            recomputed = stats.prompt_tokens - stats.cached_tokens;
            ttft.push(stats.ttft_ms);
        }
        println!("  prefix {prefix:4}, suffix {suffix:3}: {:6.1} ms ({recomputed} tokens recomputed)", median(ttft));
    }

    println!("generation through the scheduler (one step per unit, host in the loop), 128 tokens at 1024 context:");
    for n in [1usize, 2, 4] {
        let receivers: Vec<_> = (0..n).map(|k| sched.submit(request(data.fim(20000 + 7000 * k, 800, 221), 128))).collect();
        let stats: Vec<Stats> = receivers.into_iter().map(|rx| collect(rx).1).collect();
        let per_user = median(stats.iter().map(|s| (s.generated - 1) as f64 / s.decode_ms * 1e3).collect());
        let ttft = stats.iter().map(|s| s.ttft_ms).fold(0.0, f64::max);
        let (steps, guessed, right) = stats.iter().fold((0, 0, 0), |a, s| (a.0 + s.steps, a.1 + s.guessed, a.2 + s.guessed_right));
        println!("  {n} at once: {per_user:5.1} tok/s per request, {:6.1} total; slowest first token {ttft:6.1} ms; {steps} steps, {right} of {guessed} guesses right",
            per_user * n as f64);
    }

    println!("a request arriving while another with the same prefix is generating (prefix 1500, suffix 300):");
    let cursor = 90000;
    let rx_a = sched.submit(request(data.fim(cursor, 1500, 300), 200));
    let _ = (rx_a.recv(), rx_a.recv()); // Started and the first token
    let stats = run(data.fim(cursor + 3, 1503, 300), 4);
    println!("  first token after {:.1} ms, {} of {} prompt tokens cached{}", stats.ttft_ms, stats.cached_tokens, stats.prompt_tokens, if stats.copied { " by copying" } else { "" });
    drop(rx_a);
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 || !["check", "bench"].contains(&args[1].as_str()) {
        eprintln!("usage: sched check|bench WEIGHTS.wit DATA_DIR [--no-speculate]");
        return ExitCode::from(2);
    }
    let (weights, data) = (PathBuf::from(&args[2]), Data::load(Path::new(&args[3])));
    if args[1] == "bench" {
        bench(&weights, &data, !args.iter().any(|a| a == "--no-speculate"));
        return ExitCode::SUCCESS;
    }
    let ok = check(&weights, &data);
    println!("\n{}", if ok { "PASS" } else { "FAIL" });
    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
