//! The witness server.
//!
//! usage: server WEIGHTS.wit TOKENIZER.json [--port 8012] [--host 127.0.0.1] [--ctx 8192] [--slots 8]
//!               [--chunk 1024] [--no-speculate] [--trace]
//!
//! Serves llama-server's /infill (point the llama.vscode extension's endpoint at it) and
//! /v1/completions; see src/http.rs. --trace starts with the ground-truth trace enabled.
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;
use witness::http::Server;
use witness::sched::{SchedOptions, Scheduler};
use witness::tok::Tok;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: server WEIGHTS.wit TOKENIZER.json [--port 8012] [--host 127.0.0.1] [--ctx 8192] [--slots 8] [--chunk 1024] [--no-speculate] [--trace]");
        std::process::exit(2);
    }
    let flag = |name: &str, default: &str| -> String {
        args.iter().position(|a| a == name).map_or(default.to_string(), |i| args[i + 1].clone())
    };
    let number = |name: &str, default: &str| -> u32 { flag(name, default).parse().unwrap_or_else(|_| panic!("{name} must be a number")) };
    let (ctx, slots, chunk) = (number("--ctx", "8192"), number("--slots", "8"), number("--chunk", "1024"));
    let weights = PathBuf::from(&args[1]);
    let model = weights.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();

    let tok = Tok::load(&PathBuf::from(&args[2])).unwrap_or_else(|e| fail(&e));
    let sched = Scheduler::new(weights, SchedOptions { slots, max_ctx: ctx, chunk, speculate: !args.iter().any(|a| a == "--no-speculate") }).unwrap_or_else(|e| fail(&e));
    if args.iter().any(|a| a == "--trace") {
        witness::trace::set_enabled(true);
    }
    let address = format!("{}:{}", flag("--host", "127.0.0.1"), flag("--port", "8012"));
    let listener = TcpListener::bind(&address).unwrap_or_else(|e| fail(&format!("{address}: {e}")));
    eprintln!("witness: {model}, context {ctx}, {slots} slots, listening on http://{address}");
    Arc::new(Server::new(sched, tok, model, ctx as usize)).serve(listener);
}

fn fail(message: &str) -> ! {
    eprintln!("witness: {message}");
    std::process::exit(1);
}
