//! HTTP front end: llama-server's `/infill` (what the llama.vscode extension calls) and an
//! OpenAI-style `/v1/completions`, over a small HTTP/1.1 server (one thread per connection).
//!
//!   GET  /health                     {"status":"ok"}
//!   GET  /props, /v1/models          model name and context size
//!   POST /tokenize, /detokenize      {"content": text} <-> {"tokens": [ids]}
//!   POST /infill                     see `infill`
//!   POST /v1/completions             prompt (+ suffix for fill-in-the-middle), max_tokens, stream,
//!                                    ignore_eos
//!   POST /trace  {"enabled": bool}   GET /trace returns and clears the trace (JSON lines)
//!   POST /inject                     injected waits, see `inject`
//!
//! Sampling is greedy whatever the request asks for. Every byte of a response goes out through
//! `Conn::send`, which is where socket writes are traced and delayed.
use crate::sched::{Event, Finish, Request, Scheduler, Stats};
use crate::tok::Tok;
use crate::trace;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

/// llama-server's default batch size; `/infill` sizes the prefix and suffix from it.
const N_BATCH: usize = 2048;

pub struct Server {
    pub sched: Scheduler,
    pub tok: Tok,
    pub model: String,
    pub n_ctx: usize,
    next_id: AtomicU64,
}

struct Conn {
    stream: TcpStream,
    req: u64,
}

impl Conn {
    /// The only place response bytes are written. `tokens` is how many generated tokens they carry.
    fn send(&mut self, bytes: &[u8], tokens: usize) -> std::io::Result<()> {
        let delay = trace::SOCKET_DELAY_US.load(Relaxed);
        if delay > 0 {
            std::thread::sleep(std::time::Duration::from_micros(delay));
            trace::mark("injected_socket_delay", self.req, delay as i64, 0);
        }
        let t0 = trace::now_ns();
        let result = self.stream.write_all(bytes);
        trace::span("socket_write", self.req, t0, bytes.len() as i64, tokens as i64);
        result
    }

    fn respond(&mut self, status: &str, content_type: &str, body: &[u8], tokens: usize) -> std::io::Result<()> {
        let mut out = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\n\r\n",
            body.len()
        )
        .into_bytes();
        out.extend_from_slice(body);
        self.send(&out, tokens)
    }

    fn json(&mut self, status: &str, value: &Value, tokens: usize) -> std::io::Result<()> {
        self.respond(status, "application/json", value.to_string().as_bytes(), tokens)
    }

    fn error(&mut self, status: &str, message: &str) -> std::io::Result<()> {
        self.json(status, &json!({"error": {"message": message, "type": "invalid_request_error"}}), 0)
    }

    /// Starts a server-sent-event response; the connection closes when it ends.
    fn sse_start(&mut self) -> std::io::Result<()> {
        self.send(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n", 0)
    }

    fn sse(&mut self, data: &str, tokens: usize) -> std::io::Result<()> {
        self.send(format!("data: {data}\n\n").as_bytes(), tokens)
    }
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
    close: bool,
}

fn read_request(reader: &mut BufReader<TcpStream>) -> Option<HttpRequest> {
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 {
        return None;
    }
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next()?.to_string(), parts.next()?.to_string());
    let path = target.split('?').next().unwrap_or("").to_string();
    let (mut length, mut close) = (0usize, false);
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).ok()? == 0 {
            return None;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            let (name, value) = (name.trim().to_ascii_lowercase(), value.trim());
            if name == "content-length" {
                length = value.parse().ok()?;
            } else if name == "connection" && value.eq_ignore_ascii_case("close") {
                close = true;
            }
        }
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).ok()?;
    Some(HttpRequest { method, path, body, close })
}

/// What a generation produced, as far as the HTTP layer let it run.
struct Outcome {
    text: String,
    generated: usize,
    cached: usize,
    prompt_tokens: usize,
    stop_type: &'static str,
    has_new_line: bool,
    /// Submit to first token, and first token to end, in ms.
    prompt_ms: f64,
    predicted_ms: f64,
}

/// Stops applied on the generated text, as llama-server applies them for `/infill`.
#[derive(Default)]
struct TextLimits {
    /// Once a line break has been generated, stop at the next one after this long. 0 = no limit.
    t_max_predict_ms: f64,
    /// Stop when a new line starts with less indentation than this, and drop that line. 0 = off.
    n_indent: usize,
}

impl Server {
    pub fn new(sched: Scheduler, tok: Tok, model: String, n_ctx: usize) -> Server {
        Server { sched, tok, model, n_ctx, next_id: AtomicU64::new(1) }
    }

    /// Accepts connections forever.
    pub fn serve(self: Arc<Self>, listener: TcpListener) {
        for stream in listener.incoming().flatten() {
            let server = self.clone();
            let _ = std::thread::Builder::new().name("witness-http".into()).spawn(move || server.connection(stream));
        }
    }

    fn connection(&self, stream: TcpStream) {
        let _ = stream.set_nodelay(true);
        let Ok(read_half) = stream.try_clone() else { return };
        let mut reader = BufReader::new(read_half);
        let mut conn = Conn { stream, req: 0 };
        while let Some(request) = read_request(&mut reader) {
            conn.req = self.next_id.fetch_add(1, Relaxed);
            trace::mark("http_request", conn.req, request.body.len() as i64, 0);
            let keep = self.route(&mut conn, &request).unwrap_or(false);
            if !keep || request.close {
                break;
            }
        }
    }

    /// Returns whether the connection can take another request.
    fn route(&self, conn: &mut Conn, request: &HttpRequest) -> std::io::Result<bool> {
        let body = || -> Result<Value, String> { serde_json::from_slice(&request.body).map_err(|e| format!("invalid JSON: {e}")) };
        match (request.method.as_str(), request.path.as_str()) {
            ("OPTIONS", _) => conn.respond("204 No Content", "text/plain", b"", 0)?,
            ("GET", "/health") | ("GET", "/v1/health") => conn.json("200 OK", &json!({"status": "ok"}), 0)?,
            ("GET", "/v1/models") | ("GET", "/models") => {
                conn.json("200 OK", &json!({"object": "list", "data": [{"id": self.model, "object": "model", "owned_by": "witness"}]}), 0)?
            }
            ("GET", "/props") => conn.json(
                "200 OK",
                &json!({"default_generation_settings": {"n_ctx": self.n_ctx}, "total_slots": crate::sched::MAX_JOBS, "model_path": self.model, "build_info": "witness"}),
                0,
            )?,
            ("POST", "/tokenize") => match body() {
                Ok(v) => conn.json("200 OK", &json!({"tokens": self.tok.encode(v["content"].as_str().unwrap_or(""))}), 0)?,
                Err(e) => conn.error("400 Bad Request", &e)?,
            },
            ("POST", "/detokenize") => match body() {
                Ok(v) => {
                    let ids: Vec<i32> = v["tokens"].as_array().map(|a| a.iter().filter_map(|t| t.as_i64()).map(|t| t as i32).collect()).unwrap_or_default();
                    conn.json("200 OK", &json!({"content": self.tok.decode(&ids)}), 0)?
                }
                Err(e) => conn.error("400 Bad Request", &e)?,
            },
            ("GET", "/trace") => conn.respond("200 OK", "application/x-ndjson", trace::drain_jsonl().as_bytes(), 0)?,
            ("POST", "/trace") => match body() {
                Ok(v) => {
                    trace::set_enabled(v["enabled"].as_bool().unwrap_or(false));
                    conn.json("200 OK", &json!({"enabled": trace::enabled()}), 0)?
                }
                Err(e) => conn.error("400 Bad Request", &e)?,
            },
            ("POST", "/inject") => match body() {
                Ok(v) => conn.json("200 OK", &inject(&v), 0)?,
                Err(e) => conn.error("400 Bad Request", &e)?,
            },
            ("POST", "/infill") => return match body() {
                Ok(v) => self.infill(conn, &v),
                Err(e) => conn.error("400 Bad Request", &e).map(|_| true),
            },
            ("POST", "/v1/completions") | ("POST", "/completions") => return match body() {
                Ok(v) => self.completions(conn, &v),
                Err(e) => conn.error("400 Bad Request", &e).map(|_| true),
            },
            _ => conn.error("404 Not Found", "no such endpoint")?,
        }
        Ok(true)
    }

    /// Runs a generation, applying the text limits. `on_text` gets each piece of new text and the
    /// number of tokens it took; returning an error from it (the client went away) ends the run.
    fn generate(&self, req: u64, tokens: Vec<i32>, max_tokens: usize, stop: Vec<i32>, limits: &TextLimits,
                mut on_text: impl FnMut(&str, usize) -> std::io::Result<()>) -> Result<Outcome, String> {
        let submitted = Instant::now();
        let mut out = Outcome { text: String::new(), generated: 0, cached: 0, prompt_tokens: tokens.len(), stop_type: "none", has_new_line: false, prompt_ms: 0.0, predicted_ms: 0.0 };
        let rx = self.sched.submit(Request { id: req, tokens, max_tokens, stop });
        let mut detok = self.tok.stream();
        let mut first_token: Option<Instant> = None;
        let mut last_nl_pos = 0usize;
        let finish = |out: &mut Outcome, first_token: Option<Instant>| {
            out.predicted_ms = first_token.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
        };
        loop {
            match rx.recv() {
                Err(_) => return Err("the scheduler stopped".into()),
                Ok(Event::Started { cached, .. }) => out.cached = cached,
                Ok(Event::Done(Stats { finish: Finish::Error, error, .. })) => return Err(error.unwrap_or_default()),
                Ok(Event::Done(stats)) => {
                    out.stop_type = if stats.finish == Finish::Stop { "eos" } else { "limit" };
                    if first_token.is_none() {
                        out.prompt_ms = stats.ttft_ms;
                    }
                    finish(&mut out, first_token);
                    return Ok(out);
                }
                Ok(Event::Token(token)) => {
                    if first_token.is_none() {
                        first_token = Some(Instant::now());
                        out.prompt_ms = submitted.elapsed().as_secs_f64() * 1e3;
                    }
                    out.generated += 1;
                    let piece = detok(token);
                    out.text.push_str(&piece);
                    // llama-server's /infill stops, checked after every token:
                    let mut stop_now = false;
                    if out.has_new_line && limits.n_indent > 0 {
                        if last_nl_pos > 0 {
                            let rest = &out.text.as_bytes()[last_nl_pos..];
                            let indent = rest.iter().take_while(|&&c| c == b' ' || c == b'\t').count();
                            if indent < rest.len() && indent < limits.n_indent {
                                out.text.truncate(last_nl_pos + indent);
                                stop_now = true;
                            }
                        }
                        if let Some(p) = out.text[last_nl_pos.min(out.text.len())..].find('\n') {
                            last_nl_pos += p + 1;
                        }
                    }
                    if piece.contains('\n') {
                        out.has_new_line = true;
                        let elapsed = first_token.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
                        if limits.t_max_predict_ms > 0.0 && elapsed > limits.t_max_predict_ms {
                            stop_now = true;
                        }
                    }
                    if stop_now {
                        out.stop_type = "limit";
                        finish(&mut out, first_token);
                        return Ok(out); // dropping the receiver cancels the request
                    }
                    if on_text(&piece, 1).is_err() {
                        finish(&mut out, first_token);
                        return Ok(out);
                    }
                }
            }
        }
    }

    /// llama-server's FIM prompt (format_prompt_infill): repository context, then
    /// <|fim_prefix|> prefix prompt <|fim_suffix|> suffix <|fim_middle|>, sized as llama-server does.
    fn infill_prompt(&self, v: &Value, n_predict: usize) -> Vec<i32> {
        let t = &self.tok;
        let text = |key: &str| v[key].as_str().unwrap_or("");
        let mut extra = vec![t.fim_repo];
        extra.extend(t.encode("myproject\n"));
        for chunk in v["input_extra"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
            extra.push(t.fim_sep);
            extra.extend(t.encode(&format!("{}\n", chunk["filename"].as_str().unwrap_or("tmp"))));
            extra.extend(t.encode(chunk["text"].as_str().unwrap_or("")));
        }
        extra.push(t.fim_sep);
        extra.extend(t.encode("filename\n"));

        let (prefix, suffix, prompt) = (t.encode(text("input_prefix")), t.encode(text("input_suffix")), t.encode(text("prompt")));
        let n_prefix = prefix.len().min(3 * (N_BATCH / 4));
        let n_suffix = suffix.len().min((N_BATCH / 4).saturating_sub(2 + prompt.len()));
        let n_extra = extra.len().min(self.n_ctx.saturating_sub(N_BATCH + 2 * n_predict));

        let mut tokens = extra[extra.len() - n_extra..].to_vec();
        tokens.push(t.fim_prefix);
        tokens.extend_from_slice(&prefix[prefix.len() - n_prefix..]);
        tokens.extend(prompt);
        tokens.push(t.fim_suffix);
        tokens.extend_from_slice(&suffix[..n_suffix]);
        tokens.push(t.fim_middle);
        tokens
    }

    /// POST /infill, as llama-server: input_prefix, input_suffix, optional input_extra
    /// [{filename, text}], prompt, n_predict (0 only fills the cache), t_max_predict_ms, n_indent,
    /// stream. The response has llama-server's fields: content, timings, tokens_cached, ...
    fn infill(&self, conn: &mut Conn, v: &Value) -> std::io::Result<bool> {
        if !v["input_prefix"].is_string() || !v["input_suffix"].is_string() {
            return conn.error("400 Bad Request", "\"input_prefix\" and \"input_suffix\" are required").map(|_| true);
        }
        let n_predict = v["n_predict"].as_i64().filter(|&n| n >= 0).map_or(128, |n| n as usize);
        let t0 = trace::now_ns();
        let tokens = self.infill_prompt(v, n_predict);
        trace::span("tokenize", conn.req, t0, tokens.len() as i64, 0);
        let limits = TextLimits { t_max_predict_ms: v["t_max_predict_ms"].as_f64().unwrap_or(0.0), n_indent: v["n_indent"].as_u64().unwrap_or(0) as usize };
        let stream = v["stream"].as_bool().unwrap_or(false);
        let n_predict = n_predict.min(self.n_ctx.saturating_sub(tokens.len()));
        self.run(conn, tokens, n_predict, self.tok.eog.clone(), &limits, stream, Style::Llama)
    }

    /// POST /v1/completions: prompt, optional suffix (then the prompt is a FIM prompt), max_tokens,
    /// stream, ignore_eos.
    fn completions(&self, conn: &mut Conn, v: &Value) -> std::io::Result<bool> {
        let Some(prompt) = v["prompt"].as_str() else {
            return conn.error("400 Bad Request", "\"prompt\" must be a string").map(|_| true);
        };
        let t0 = trace::now_ns();
        let mut tokens = Vec::new();
        match v["suffix"].as_str() {
            Some(suffix) => {
                tokens.push(self.tok.fim_prefix);
                tokens.extend(self.tok.encode(prompt));
                tokens.push(self.tok.fim_suffix);
                tokens.extend(self.tok.encode(suffix));
                tokens.push(self.tok.fim_middle);
            }
            None => tokens = self.tok.encode(prompt),
        }
        trace::span("tokenize", conn.req, t0, tokens.len() as i64, 0);
        let max_tokens = v["max_tokens"].as_u64().unwrap_or(16) as usize;
        let stop = if v["ignore_eos"].as_bool().unwrap_or(false) { Vec::new() } else { self.tok.eog.clone() };
        let stream = v["stream"].as_bool().unwrap_or(false);
        self.run(conn, tokens, max_tokens, stop, &TextLimits::default(), stream, Style::OpenAi)
    }

    fn run(&self, conn: &mut Conn, tokens: Vec<i32>, max_tokens: usize, stop: Vec<i32>, limits: &TextLimits, stream: bool, style: Style) -> std::io::Result<bool> {
        let req = conn.req;
        if !stream {
            return match self.generate(req, tokens, max_tokens, stop, limits, |_, _| Ok(())) {
                Ok(out) => conn.json("200 OK", &style.finished(self, &out, max_tokens, true), out.generated).map(|_| true),
                Err(e) => conn.error("400 Bad Request", &e).map(|_| true),
            };
        }
        conn.sse_start()?;
        // With token coalescing injected, pieces are held back and written several at a time.
        let (mut held, mut held_tokens) = (String::new(), 0usize);
        let result = self.generate(req, tokens, max_tokens, stop, limits, |piece, n| {
            held.push_str(piece);
            held_tokens += n;
            if (held_tokens as u64) < trace::COALESCE_TOKENS.load(Relaxed).max(1) {
                return Ok(());
            }
            let chunk = style.chunk(self, &held);
            let sent = conn.sse(&chunk.to_string(), held_tokens);
            held.clear();
            held_tokens = 0;
            sent
        });
        match result {
            Ok(out) => {
                if held_tokens > 0 {
                    conn.sse(&style.chunk(self, &held).to_string(), held_tokens)?;
                }
                conn.sse(&style.finished(self, &out, max_tokens, false).to_string(), 0)?;
                if matches!(style, Style::OpenAi) {
                    conn.sse("[DONE]", 0)?;
                }
            }
            Err(e) => conn.sse(&json!({"error": {"message": e}}).to_string(), 0)?,
        }
        Ok(false)
    }
}

#[derive(Clone, Copy)]
enum Style {
    Llama,
    OpenAi,
}

impl Style {
    /// A streamed piece of text.
    fn chunk(&self, server: &Server, text: &str) -> Value {
        match self {
            Style::Llama => json!({"index": 0, "content": text, "stop": false}),
            Style::OpenAi => json!({"object": "text_completion", "model": server.model, "choices": [{"index": 0, "text": text, "finish_reason": null}]}),
        }
    }

    /// The whole response (`with_text`), or the last event of a stream.
    fn finished(&self, server: &Server, out: &Outcome, n_predict: usize, with_text: bool) -> Value {
        let text = if with_text { out.text.as_str() } else { "" };
        let prompt_n = out.prompt_tokens - out.cached;
        let per = |ms: f64, n: usize| if n > 0 { ms / n as f64 } else { 0.0 };
        let rate = |ms: f64, n: usize| if ms > 0.0 { n as f64 / ms * 1e3 } else { 0.0 };
        match self {
            Style::Llama => json!({
                "index": 0,
                "content": text,
                "tokens": [],
                "stop": true,
                "model": server.model,
                "tokens_predicted": out.generated,
                "tokens_evaluated": out.prompt_tokens,
                "generation_settings": {"n_ctx": server.n_ctx, "n_predict": n_predict},
                "has_new_line": out.has_new_line,
                "truncated": false,
                "stop_type": out.stop_type,
                "stopping_word": "",
                "tokens_cached": out.prompt_tokens + out.generated,
                "timings": {
                    "cache_n": out.cached,
                    "prompt_n": prompt_n,
                    "prompt_ms": out.prompt_ms,
                    "prompt_per_token_ms": per(out.prompt_ms, prompt_n),
                    "prompt_per_second": rate(out.prompt_ms, prompt_n),
                    "predicted_n": out.generated,
                    "predicted_ms": out.predicted_ms,
                    "predicted_per_token_ms": per(out.predicted_ms, out.generated),
                    "predicted_per_second": rate(out.predicted_ms, out.generated),
                },
            }),
            Style::OpenAi => json!({
                "object": "text_completion",
                "model": server.model,
                "choices": [{"index": 0, "text": text, "finish_reason": if out.stop_type == "eos" { "stop" } else { "length" }}],
                "usage": {"prompt_tokens": out.prompt_tokens, "completion_tokens": out.generated, "total_tokens": out.prompt_tokens + out.generated,
                          "prompt_tokens_details": {"cached_tokens": out.cached}},
            }),
        }
    }
}

/// POST /inject: any of
///   gpu_thread_spin_us, gpu_thread_sleep_us   before every unit of GPU work (prefill chunk, step)
///   socket_delay_us                           before every socket write of a response
///   coalesce_tokens                           tokens per socket write of a streamed response
///   cuda_delay_us + cuda_site                 sleep inside CUDA calls whose text contains cuda_site
/// A field left out keeps its value; 0 turns it off. Returns the current settings.
fn inject(v: &Value) -> Value {
    for (key, knob) in [
        ("gpu_thread_spin_us", &trace::UNIT_SPIN_US),
        ("gpu_thread_sleep_us", &trace::UNIT_SLEEP_US),
        ("socket_delay_us", &trace::SOCKET_DELAY_US),
        ("coalesce_tokens", &trace::COALESCE_TOKENS),
    ] {
        if let Some(n) = v[key].as_u64() {
            knob.store(n, Relaxed);
        }
    }
    if let Some(us) = v["cuda_delay_us"].as_u64() {
        trace::inject_cuda_delay(v["cuda_site"].as_str().unwrap_or(""), us as u32);
    }
    json!({
        "gpu_thread_spin_us": trace::UNIT_SPIN_US.load(Relaxed),
        "gpu_thread_sleep_us": trace::UNIT_SLEEP_US.load(Relaxed),
        "socket_delay_us": trace::SOCKET_DELAY_US.load(Relaxed),
        "coalesce_tokens": trace::COALESCE_TOKENS.load(Relaxed),
    })
}
