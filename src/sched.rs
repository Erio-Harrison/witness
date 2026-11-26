//! The scheduler: one thread owns the engine and serves completion requests from a pool of KV slots.
//!
//! Prefix cache. Every slot remembers the tokens whose K/V it holds. A request takes the slot that
//! shares the longest prefix with its prompt and computes only the rest. If that slot is still
//! serving another request, the shared prefix is copied into a free slot instead of waiting; that
//! is what makes overlapping requests from one editor (the user kept typing) cheap.
//!
//! Scheduling. Work is done in units: one prefill chunk of one request that still has prompt left, or
//! one decode step for all requests that are generating (up to `MAX_JOBS`). Which request's prompt
//! goes first: a request that continues one still being served (it shares that one's prefix: the
//! same editor position, typed further) goes ahead of everything, since the older one is probably
//! stale; otherwise the request with the least prompt left, which minimizes the average wait and
//! never interrupts a prompt for an equally long one that arrived later. When both
//! kinds are waiting, prompts go first and generation gets one step per two chunks, so a long prompt
//! cannot stall generation altogether and generation costs a new request little. Requests beyond `MAX_JOBS` wait in a queue. A request whose receiver is
//! dropped is cancelled at the next unit.
//!
//! Speculative decoding. A decode step costs little more with 24 rows than with one. Rows that no request needs carry guesses: if the last few tokens of a sequence occurred
//! earlier in it, the tokens that followed them then are fed at the next positions in the same step.
//! Each guess that matches what the model actually predicts makes the following row's prediction
//! valid too, so one step yields several tokens. The tokens are exactly those of one-at-a-time
//! decoding: a row is used only if every token fed before it was the model's own choice.
//!
//! Sampling is greedy.
use crate::{trace, Config, Engine, Options, Row};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct SchedOptions {
    /// KV slots. More than `MAX_JOBS` leaves room for cached prefixes of finished requests.
    pub slots: u32,
    /// Positions per slot: the longest prompt plus generation.
    pub max_ctx: u32,
    /// Prompt tokens computed per unit of work: the granularity of cancellation and of interleaving
    /// with decode steps.
    pub chunk: u32,
    /// Use spare rows of a decode step for guessed tokens.
    pub speculate: bool,
}

impl Default for SchedOptions {
    fn default() -> Self {
        SchedOptions { slots: 8, max_ctx: 8192, chunk: 1024, speculate: true }
    }
}

pub struct Request {
    /// The caller's name for the request in the trace.
    pub id: u64,
    pub tokens: Vec<i32>,
    /// Tokens to generate at most. 0 computes the prompt into the cache and generates nothing.
    pub max_tokens: usize,
    /// Generation ends before any of these is emitted.
    pub stop: Vec<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Finish {
    /// A stop token was produced.
    Stop,
    /// `max_tokens` were generated.
    Length,
    /// The request could not be served; see `Stats::error`.
    Error,
}

#[derive(Clone, Debug)]
pub struct Stats {
    pub finish: Finish,
    pub error: Option<String>,
    pub prompt_tokens: usize,
    /// Prompt tokens whose K/V were reused from the cache.
    pub cached_tokens: usize,
    /// Whether the cached prefix had to be copied because its slot was in use.
    pub copied: bool,
    pub generated: usize,
    /// Decode steps run for the request, guessed tokens fed, and guesses that were right. Each
    /// right guess is a token that cost no step.
    pub steps: usize,
    pub guessed: usize,
    pub guessed_right: usize,
    /// From `submit` to the start of work on the request.
    pub queue_ms: f64,
    /// From `submit` to the first generated token.
    pub ttft_ms: f64,
    /// From the first generated token to the last.
    pub decode_ms: f64,
}

pub enum Event {
    /// The request has a slot: `cached` of its prompt tokens were reused.
    Started { cached: usize, copied: bool },
    Token(i32),
    Done(Stats),
}

enum Msg {
    Submit(Request, Sender<Event>, Instant),
    Shutdown,
}

pub struct Scheduler {
    tx: Sender<Msg>,
    thread: Option<JoinHandle<()>>,
    pub config: Config,
}

impl Scheduler {
    /// Loads the model on a new thread, which then owns the GPU.
    pub fn new(weights: PathBuf, options: SchedOptions) -> Result<Scheduler, String> {
        if options.slots as usize <= MAX_JOBS {
            return Err(format!("{} slots: need more than {MAX_JOBS}", options.slots));
        }
        let (tx, rx) = channel();
        let (ready_tx, ready_rx) = channel();
        let thread = std::thread::Builder::new()
            .name("witness-gpu".into())
            .spawn(move || {
                let engine_options = Options { slots: options.slots, max_ctx: options.max_ctx, max_chunk: options.chunk, kv_bf16: 1 };
                match Engine::new(&weights, engine_options) {
                    Ok(engine) => {
                        let _ = ready_tx.send(Ok(engine.config));
                        Worker::new(engine, options).run(rx);
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        let config = ready_rx.recv().map_err(|_| "the GPU thread died while loading".to_string())??;
        Ok(Scheduler { tx, thread: Some(thread), config })
    }

    /// Queues a request. Tokens arrive on the returned channel as they are generated, then one
    /// `Done`. Dropping the receiver cancels the request.
    pub fn submit(&self, request: Request) -> Receiver<Event> {
        let (tx, rx) = channel();
        let _ = self.tx.send(Msg::Submit(request, tx, Instant::now()));
        rx
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

struct Slot {
    /// Tokens whose K/V this slot holds, position by position.
    tokens: Vec<i32>,
    busy: bool,
    last_used: u64,
}

struct Job {
    id: u64,
    prompt: Vec<i32>,
    max_tokens: usize,
    stop: Vec<i32>,
    tx: Sender<Event>,
    slot: usize,
    /// The token to feed next, once the prompt is in the slot.
    next: i32,
    decoding: bool,
    stats: Stats,
    submitted: Instant,
    first_token: Option<Instant>,
    /// Order of arrival.
    seq: u64,
    /// Continues a request that was still being served when this one arrived.
    urgent: bool,
}

struct Worker {
    engine: Engine,
    options: SchedOptions,
    slots: Vec<Slot>,
    jobs: Vec<Job>,
    queue: VecDeque<(Request, Sender<Event>, Instant)>,
    clock: u64,
    /// Prefill chunks done since the last decode step.
    chunks_since_decode: u32,
}

/// Requests served at once; more wait in the queue.
pub const MAX_JOBS: usize = 4;

/// Prefill chunks between two decode steps while prompts are waiting.
const DECODE_EVERY: u32 = 2;

/// Rows of a decode step that are used. Measured: 24 rows cost 1.4 ms more than one (21.2 against
/// 19.7 ms), 32 rows 3.5 ms more.
const STEP_ROWS: usize = 24;

/// A guess needs the sequence's last 1 to 8 tokens to have occurred before.
const GUESS_MATCH_MIN: usize = 1;
const GUESS_MATCH_MAX: usize = 8;

/// Shortest shared prefix worth reusing: below it the request starts from scratch in the least
/// recently used slot instead of taking over (or copying from) the slot that happens to match.
const MIN_REUSE: usize = 32;

fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

impl Worker {
    fn new(engine: Engine, options: SchedOptions) -> Worker {
        let slots = (0..options.slots).map(|_| Slot { tokens: Vec::new(), busy: false, last_used: 0 }).collect();
        Worker { engine, options, slots, jobs: Vec::new(), queue: VecDeque::new(), clock: 0, chunks_since_decode: 0 }
    }

    fn run(mut self, rx: Receiver<Msg>) {
        loop {
            // Block only when there is nothing to do.
            if self.jobs.is_empty() && self.queue.is_empty() {
                match rx.recv() {
                    Ok(Msg::Submit(r, tx, t)) => self.queue.push_back((r, tx, t)),
                    Ok(Msg::Shutdown) | Err(_) => return,
                }
            }
            loop {
                match rx.try_recv() {
                    Ok(Msg::Submit(r, tx, t)) => self.queue.push_back((r, tx, t)),
                    Ok(Msg::Shutdown) | Err(TryRecvError::Disconnected) => return,
                    Err(TryRecvError::Empty) => break,
                }
            }
            while self.jobs.len() < MAX_JOBS {
                match self.queue.pop_front() {
                    Some((r, tx, t)) => self.admit(r, tx, t),
                    None => break,
                }
            }
            self.work();
        }
    }

    fn fail(tx: &Sender<Event>, request: &Request, submitted: Instant, error: String) {
        let stats = Stats {
            finish: Finish::Error,
            error: Some(error),
            prompt_tokens: request.tokens.len(),
            cached_tokens: 0,
            copied: false,
            generated: 0,
            steps: 0,
            guessed: 0,
            guessed_right: 0,
            queue_ms: ms_since(submitted),
            ttft_ms: 0.0,
            decode_ms: 0.0,
        };
        let _ = tx.send(Event::Done(stats));
    }

    /// Gives the request a slot, reusing the longest cached prefix.
    fn admit(&mut self, request: Request, tx: Sender<Event>, submitted: Instant) {
        let n = request.tokens.len();
        if n == 0 {
            return Self::fail(&tx, &request, submitted, "empty prompt".into());
        }
        if n + request.max_tokens + STEP_ROWS > self.options.max_ctx as usize {
            return Self::fail(&tx, &request, submitted, format!("{n} prompt + {} generated tokens exceed the context of {}", request.max_tokens, self.options.max_ctx));
        }
        if request.tokens.iter().any(|&t| t < 0 || t as u32 >= self.engine.config.vocab) {
            return Self::fail(&tx, &request, submitted, "token id out of range".into());
        }
        // The last prompt token is always recomputed: its logits are the first prediction.
        let shared = |slot: &Slot| common_prefix(&slot.tokens, &request.tokens[..n - 1]);
        let best = (0..self.slots.len()).max_by_key(|&i| (shared(&self.slots[i]), !self.slots[i].busy)).unwrap();
        // A few shared tokens (every FIM prompt starts with the same one) are not worth a cache.
        let cached = Some(shared(&self.slots[best])).filter(|&n| n >= MIN_REUSE).unwrap_or(0);
        let lru_idle = (0..self.slots.len()).filter(|&i| !self.slots[i].busy).min_by_key(|&i| self.slots[i].last_used);
        let lru_idle = lru_idle.expect("more slots than rows, so an idle slot exists");

        let (slot, copied) = if cached == 0 {
            (lru_idle, false) // nothing to reuse: give up the least recently used cache
        } else if !self.slots[best].busy {
            (best, false)
        } else {
            if let Err(e) = self.engine.copy_kv(best as i32, lru_idle as i32, cached) {
                return Self::fail(&tx, &request, submitted, e);
            }
            (lru_idle, true)
        };
        self.slots[slot].tokens.clear();
        self.slots[slot].tokens.extend_from_slice(&request.tokens[..cached]);
        self.slots[slot].busy = true;
        self.clock += 1;
        let stats = Stats {
            finish: Finish::Length,
            error: None,
            prompt_tokens: n,
            cached_tokens: cached,
            copied,
            generated: 0,
            steps: 0,
            guessed: 0,
            guessed_right: 0,
            queue_ms: ms_since(submitted),
            ttft_ms: 0.0,
            decode_ms: 0.0,
        };
        trace::mark("admit", request.id, slot as i64, cached as i64);
        let _ = tx.send(Event::Started { cached, copied });
        self.jobs.push(Job {
            id: request.id,
            prompt: request.tokens,
            max_tokens: request.max_tokens,
            stop: request.stop,
            tx,
            slot,
            next: 0,
            decoding: false,
            stats,
            submitted,
            first_token: None,
            seq: self.clock,
            urgent: copied,
        });
    }

    /// One unit of work.
    fn work(&mut self) {
        trace::unit_injection();
        // Urgent requests first, the newest of them; then the least prompt left, the oldest on a tie.
        let prefill = (0..self.jobs.len()).filter(|&i| !self.jobs[i].decoding).min_by_key(|&i| {
            let job = &self.jobs[i];
            let left = job.prompt.len() - self.slots[job.slot].tokens.len();
            if job.urgent { (0, u64::MAX - job.seq, 0) } else { (1, left as u64, job.seq) }
        });
        let decoding = self.jobs.iter().any(|j| j.decoding);
        // With both kinds of work waiting, prompts go first: requests that are generating get one
        // step per DECODE_EVERY chunks. A step would cost each waiting prompt 10% of a chunk, and
        // gives the generating requests little while a chunk takes ten times as long as a step.
        let result = match prefill {
            Some(i) if !decoding || self.chunks_since_decode < DECODE_EVERY => {
                self.chunks_since_decode += 1;
                self.prefill_chunk(i)
            }
            _ if decoding => {
                self.chunks_since_decode = 0;
                self.decode_step()
            }
            _ => Ok(()),
        };
        if let Err(e) = result {
            // An engine error is not attributable to one request: fail them all.
            for mut job in std::mem::take(&mut self.jobs) {
                job.stats.finish = Finish::Error;
                job.stats.error = Some(e.clone());
                self.release(&job);
                let _ = job.tx.send(Event::Done(job.stats));
            }
        }
    }

    fn release(&mut self, job: &Job) {
        self.clock += 1;
        self.slots[job.slot].busy = false;
        self.slots[job.slot].last_used = self.clock;
    }

    /// Hands `token` to job `i` as its next generated token. Returns false when the job is over
    /// (stop token, length, or the client went away); the job is then removed.
    fn emit(&mut self, i: usize, token: i32) -> bool {
        let job = &mut self.jobs[i];
        let mut over = None;
        if job.stop.contains(&token) {
            over = Some(Finish::Stop);
        } else {
            if job.first_token.is_none() {
                job.first_token = Some(Instant::now());
                job.stats.ttft_ms = ms_since(job.submitted);
            }
            job.stats.generated += 1;
            job.next = token;
            trace::mark("token", job.id, token as i64, job.stats.generated as i64 - 1);
            if job.tx.send(Event::Token(token)).is_err() {
                // Cancelled: nobody is listening, so there is no Done to send either.
                let job = self.jobs.swap_remove(i);
                self.release(&job);
                return false;
            }
            if job.stats.generated >= job.max_tokens {
                over = Some(Finish::Length);
            }
        }
        match over {
            Some(finish) => {
                let mut job = self.jobs.swap_remove(i);
                job.stats.finish = finish;
                job.stats.decode_ms = job.first_token.map_or(0.0, ms_since);
                trace::mark("done", job.id, job.stats.generated as i64, 0);
                self.release(&job);
                let _ = job.tx.send(Event::Done(job.stats));
                false
            }
            None => true,
        }
    }

    fn prefill_chunk(&mut self, i: usize) -> Result<(), String> {
        let job = &self.jobs[i];
        let slot = job.slot;
        let done = self.slots[slot].tokens.len();
        let end = (done + self.options.chunk as usize).min(job.prompt.len());
        let t0 = trace::now_ns();
        let first = self.engine.prefill(slot as i32, done as i32, &job.prompt[done..end], 0, None, None)?;
        trace::span("prefill", job.id, t0, slot as i64, (end - done) as i64);
        self.slots[slot].tokens.extend_from_slice(&job.prompt[done..end]);
        if end == self.jobs[i].prompt.len() {
            if self.jobs[i].max_tokens == 0 {
                // Cache warm-up: the prompt is in the slot and nothing is generated.
                let mut job = self.jobs.swap_remove(i);
                job.stats.ttft_ms = ms_since(job.submitted);
                trace::mark("done", job.id, 0, 0);
                self.release(&job);
                let _ = job.tx.send(Event::Done(job.stats));
                return Ok(());
            }
            self.jobs[i].decoding = true;
            self.emit(i, first);
        }
        Ok(())
    }

    /// Guesses for the tokens after `next`, which is about to be fed after `context`: the tokens that
    /// followed the most recent earlier occurrence of the sequence's last tokens. The longest tail
    /// of `GUESS_MATCH_MAX` down to `GUESS_MATCH_MIN` tokens that occurs anywhere wins; for each
    /// length the sequence itself is searched first, then `others` (what other slots hold: other
    /// requests' text, often from the same project).
    ///
    /// Measured on the model's output for 32 benchmark prompts (code continuation), tokens per
    /// step with 7 guesses: tails of 2..8 tokens 1.85-2.59, of 1..8 tokens 2.14-3.00, plus other
    /// requests' text 2.28-2.96. Choosing the most frequent continuation instead of the most recent,
    /// or longer tails, changed nothing.
    fn guesses(context: &[i32], next: i32, others: &[&[i32]], want: usize, out: &mut Vec<i32>) {
        out.clear();
        if want == 0 {
            return;
        }
        let n = context.len() + 1; // the sequence including `next`
        let at = |i: usize| if i + 1 == n { next } else { context[i] };
        for len in (GUESS_MATCH_MIN..=GUESS_MATCH_MAX).rev() {
            if n < len + 1 {
                continue;
            }
            // Candidate ends e (exclusive) < n with [e - len, e) == [n - len, n), most recent first.
            let mut e = n - 1;
            while e >= len {
                if at(e - 1) == next && (1..len).all(|k| at(e - 1 - k) == at(n - 1 - k)) {
                    out.extend((e..n.min(e + want)).map(at));
                    return;
                }
                e -= 1;
            }
            for other in others {
                let m = other.len();
                let mut e = m;
                while e >= len && e > 0 {
                    // [e - len, e) of `other` against the tail; what follows must exist (e < m).
                    if e < m && other[e - 1] == next && (1..len).all(|k| other[e - 1 - k] == at(n - 1 - k)) {
                        out.extend_from_slice(&other[e..m.min(e + want)]);
                        return;
                    }
                    e -= 1;
                }
            }
        }
    }

    fn decode_step(&mut self) -> Result<(), String> {
        let rows_of: Vec<usize> = (0..self.jobs.len()).filter(|&i| self.jobs[i].decoding).collect();
        // Spare rows are shared out evenly among the decoding requests as guesses; the remainder
        // goes to a different request each step.
        let spare = if self.options.speculate { STEP_ROWS - rows_of.len() } else { 0 };
        let (share, extra) = (spare / rows_of.len(), spare % rows_of.len());
        self.clock += 1;
        let turn = self.clock as usize;
        let mut rows: Vec<Row> = Vec::with_capacity(STEP_ROWS);
        let mut first_row = Vec::with_capacity(rows_of.len());
        let mut real_guesses = Vec::with_capacity(rows_of.len());
        let mut guess = Vec::new();
        for (k, &i) in rows_of.iter().enumerate() {
            let job = &self.jobs[i];
            let context = &self.slots[job.slot].tokens;
            let pos = context.len();
            let others: Vec<&[i32]> = (0..self.slots.len()).filter(|&o| o != job.slot).map(|o| self.slots[o].tokens.as_slice()).collect();
            let spare = share + ((k + turn) % rows_of.len() < extra) as usize;
            // The share is used in full even when the request is about to end and few of the
            // guesses could still be emitted: every distinct shape of a step is a CUDA graph to
            // capture, and shrinking shares near the end made the shapes multiply under concurrency.
            // (Admission keeps STEP_ROWS positions free at the end of the context for this.)
            let want = spare;
            Self::guesses(context, job.next, &others, want, &mut guess);
            real_guesses.push(guess.len());
            // A request always fills its share of rows: where it has no guess it repeats its last
            // token. A step's cost barely depends on its rows, and a fixed shape per number of
            // requests keeps the engine to a handful of step graphs.
            let filler = guess.last().copied().unwrap_or(job.next);
            guess.resize(want, filler);
            first_row.push(rows.len());
            rows.push(Row { token: job.next, slot: job.slot as i32, pos: pos as i32 });
            for (k, &token) in guess.iter().enumerate() {
                rows.push(Row { token, slot: job.slot as i32, pos: (pos + 1 + k) as i32 });
            }
        }
        first_row.push(rows.len());
        let t0 = trace::now_ns();
        let out = self.engine.generate(&rows, 1)?;
        trace::span("decode", 0, t0, rows.len() as i64, (rows.len() - rows_of.len()) as i64);

        // Highest index first: `emit` removes finished jobs with swap_remove.
        for (k, &i) in rows_of.iter().enumerate().rev() {
            let (begin, end) = (first_row[k], first_row[k + 1]);
            let slot = self.jobs[i].slot;
            self.jobs[i].stats.steps += 1;
            self.jobs[i].stats.guessed += real_guesses[k];
            // Row r's prediction is valid if every token fed up to row r was the model's choice:
            // row `begin` always, row r + 1 if its guessed token equals row r's prediction.
            for r in begin..end {
                self.slots[slot].tokens.push(rows[r].token);
                let alive = self.emit(i, out[r]);
                if !alive || r + 1 == end || rows[r + 1].token != out[r] {
                    break;
                }
                self.jobs[i].stats.guessed_right += 1;
            }
        }
        Ok(())
    }
}
