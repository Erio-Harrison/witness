//! Rough timing of the two paths, from inside the process (no HTTP, no tokenizer).
//!
//! usage: bench WEIGHTS.wit TOKENS.i32 [--chunk N] [--gen N] [--reps N] [--rows N] [--kv-fp32]
//!
//! For each prompt length, prefills that many tokens (the token file repeated as needed) with the
//! wide path in chunks of --chunk, then generates --gen tokens with the step graph. Prints the
//! median over --reps of the prefill time (to the first token's logits) and of the decode speed.
//! With --rows N, N sequences are prefilled one after another and then decoded together.
//! BENCH_LENS=128,2048 picks the prompt lengths. WK_PROFILE=1 also runs a few fused steps one kernel
//! at a time and prints the time per kernel kind (inflated by the wait after each kernel).
use std::path::Path;
use std::time::Instant;
use witness::{Engine, Options, Row};

const LENS: [usize; 4] = [128, 1024, 2048, 4096];

fn lens() -> Vec<usize> {
    match std::env::var("BENCH_LENS") {
        Ok(v) => v.split(',').map(|x| x.parse().expect("BENCH_LENS")).collect(),
        Err(_) => LENS.to_vec(),
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: bench WEIGHTS.wit TOKENS.i32 [--chunk N] [--gen N] [--reps N]");
        std::process::exit(2);
    }
    let flag = |name: &str, default: usize| -> usize {
        args.iter().position(|a| a == name).map_or(default, |i| args[i + 1].parse().expect(name))
    };
    let (chunk, gen, reps, rows) = (flag("--chunk", 1024), flag("--gen", 64), flag("--reps", 5), flag("--rows", 1));
    let kv_bf16 = !args.iter().any(|a| a == "--kv-fp32") as u32;
    let bytes = std::fs::read(&args[2]).expect("TOKENS.i32");
    let source: Vec<i32> = bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let max_len = *LENS.iter().max().unwrap();
    let tokens: Vec<i32> = source.iter().cycle().take(max_len).copied().collect();

    let options = Options { slots: rows as u32, max_ctx: (max_len + gen + 1) as u32, max_chunk: chunk as u32, kv_bf16 };
    let mut engine = Engine::new(Path::new(&args[1]), options).expect("engine");
    engine.prefill(0, 0, &tokens[..chunk.min(256)], 0, None, None).expect("warm-up");

    println!("chunk {chunk}, {gen} decoded tokens, {rows} row(s), {} KV, median of {reps}", if kv_bf16 == 1 { "bf16" } else { "fp32" });
    for len in lens() {
        let (mut prefill_ms, mut decode_tps) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            let mut first = vec![0; rows];
            for (slot, next) in first.iter_mut().enumerate() {
                let start = Instant::now();
                for (i, piece) in tokens[..len].chunks(chunk).enumerate() {
                    *next = engine.prefill(slot as i32, (i * chunk) as i32, piece, 0, None, None).expect("prefill");
                }
                prefill_ms.push(start.elapsed().as_secs_f64() * 1e3);
            }
            let batch: Vec<Row> = (0..rows).map(|slot| Row { token: first[slot], slot: slot as i32, pos: len as i32 }).collect();
            let start = Instant::now();
            engine.generate(&batch, gen).expect("generate");
            decode_tps.push(gen as f64 / start.elapsed().as_secs_f64());
            if std::env::var_os("WK_PROFILE").is_some() {
                let batch: Vec<Row> = (0..rows).map(|slot| Row { token: first[slot], slot: slot as i32, pos: len as i32 }).collect();
                for _ in 0..16 {
                    engine.step_fast(&batch, None).expect("step_fast");
                }
            }
        }
        let ms = median(prefill_ms);
        let tps = median(decode_tps);
        println!("prompt {len:5}: prefill {ms:7.1} ms ({:5.0} tok/s), decode {tps:5.1} tok/s per row, {:5.1} total ({:.2} ms/step)", len as f64 / ms * 1e3, tps * rows as f64, 1e3 / tps);
    }
}
