//! Correctness check of the engine against the HF reference dumps (tools/ref_dump.py).
//!
//! usage: check WEIGHTS.wit REF_DIR [--tap-tol X] [--wide-tol X] [--margin X]
//!
//! The error of a value is max|ours - ref| / max|ref| over one layer's tensor (or over a case's
//! logits), against the fp32 reference. "Flips" are positions whose argmax differs from a reference;
//! they are allowed only where that reference's own top-2 margin is under --margin. The bf16
//! reference is the original model.
//!
//! Part 1, exact configuration (fp32 KV cache). For every case in REF_DIR:
//!  reference     one token per step, unfused kernels, teacher-forced. Every value within --tap-tol.
//!  wide exact    prefill in uneven chunks with fp32 products. Every value within --tap-tol: this
//!                checks the wide path's logic, including attention across chunk boundaries.
//!  fast          one token per step, fused kernels. Logits within --tap-tol.
//!  wide          prefill in one chunk with the real bf16-input products. Only the logits are held
//!                to a limit (--wide-tol): rounding a product's input to bf16 costs ~0.4% of the
//!                largest element, and the few huge residual-stream values these models carry
//!                through the middle layers turn that into several percent of the intermediate
//!                values once they cancel. The values are still printed (CHECK_LAYERS=1 prints
//!                every layer as error/scale).
//!  generation    greedy, compared with HF's: reference kernels; wide prefill of the prompt in
//!                uneven chunks then reference kernels; fused kernels then the step graph.
//! Then two cases run as two rows: reference steps must give bit-identical logits to the single-row
//! run, and graph generation the same tokens.
//!
//! Part 2, normal configuration (bf16 KV cache): fast and wide logits within --wide-tol, and
//! generation with a wide prefill followed by the step graph.
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use witness::{Config, Engine, Options, Row, Taps, ALL_LOGITS, EXACT};

const BATCH_POSITIONS: usize = 24;
const KINDS: [&str; 8] = ["x_in", "norm1", "qkv", "attn", "o", "norm2", "act", "mlp"];
const CHUNK_SIZES: [usize; 3] = [37, 64, 512];

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
}

/// Index of the largest value and its lead over the second largest.
fn top(v: &[f32]) -> (i32, f32) {
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

/// `(start, len)` pieces of `0..n` with uneven sizes.
fn chunks(n: usize) -> Vec<(usize, usize)> {
    let (mut out, mut start) = (Vec::new(), 0);
    while start < n {
        let len = CHUNK_SIZES[out.len() % CHUNK_SIZES.len()].min(n - start);
        out.push((start, len));
        start += len;
    }
    out
}

#[derive(Clone, Copy, Default)]
struct Err {
    diff: f32,
    scale: f32,
}

impl Err {
    fn add(&mut self, ours: &[f32], reference: &[f32]) {
        assert_eq!(ours.len(), reference.len());
        for (a, b) in ours.iter().zip(reference) {
            self.diff = self.diff.max((a - b).abs());
            self.scale = self.scale.max(b.abs());
        }
    }
    fn rel(&self) -> f32 {
        self.diff / self.scale
    }
}

/// One case's reference data.
struct Reference {
    name: String,
    t_len: usize,
    n_prompt: usize,
    tokens: Vec<i32>,
    dims: [usize; 8],
    taps: Vec<Vec<f32>>,
    final_norm: Vec<f32>,
    fp32: Vec<f32>,
    bf16: Vec<f32>,
}

impl Reference {
    fn load(dir: &Path, c: &Config) -> Reference {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let meta = read_u32(&dir.join("meta.bin"));
        assert_eq!(&meta[2..], &[c.layers, c.hidden, c.qkv_dim() as u32, c.mlp, c.vocab], "{name}: reference is for another model");
        let hidden = c.hidden as usize;
        Reference {
            name,
            t_len: meta[0] as usize,
            n_prompt: meta[1] as usize,
            tokens: read_u32(&dir.join("tokens.i32")).into_iter().map(|t| t as i32).collect(),
            dims: [hidden, hidden, c.qkv_dim(), hidden, hidden, hidden, c.mlp as usize, hidden],
            taps: KINDS.iter().map(|k| read_f32(&dir.join(format!("{k}.f32")))).collect(),
            final_norm: read_f32(&dir.join("final_norm.f32")),
            fp32: read_f32(&dir.join("logits_fp32.f32")),
            bf16: read_f32(&dir.join("logits_bf16.f32")),
        }
    }

    fn vocab(&self) -> usize {
        self.bf16.len() / self.t_len
    }
}

/// KL(reference || ours) of the next-token distributions given by two rows of logits, in nats.
fn kl(reference: &[f32], ours: &[f32]) -> f64 {
    let log_softmax = |v: &[f32]| -> Vec<f64> {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let z = v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
        v.iter().map(|&x| x as f64 - z).collect()
    };
    let (p, q) = (log_softmax(reference), log_softmax(ours));
    p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
}

/// Errors of one pass over a case, accumulated chunk by chunk.
struct Pass {
    errs: Vec<[Err; 8]>,
    final_norm: Err,
    fp32: Err,
    bf16: Err,
    /// (position, reference margin) where our argmax differs.
    fp32_flips: Vec<(usize, f32)>,
    bf16_flips: Vec<(usize, f32)>,
    /// KL(HF fp32 || ours) per position.
    kl: Vec<f64>,
}

impl Pass {
    fn new(layers: usize) -> Pass {
        Pass { errs: vec![[Err::default(); 8]; layers], final_norm: Err::default(), fp32: Err::default(), bf16: Err::default(), fp32_flips: Vec::new(), bf16_flips: Vec::new(), kl: Vec::new() }
    }

    /// Adds `n` rows of taps and logits that sit at positions `t0..t0 + n` of the case.
    fn add(&mut self, r: &Reference, t0: usize, n: usize, taps: &Taps, logits: &[f32]) {
        let (layers, t_len) = (self.errs.len(), r.t_len);
        let ours = [&taps.x_in, &taps.norm1, &taps.qkv, &taps.attn, &taps.o, &taps.norm2, &taps.act, &taps.mlp];
        for l in 0..layers {
            for k in 0..8 {
                let d = r.dims[k];
                self.errs[l][k].add(&ours[k][l * n * d..(l + 1) * n * d], &r.taps[k][(l * t_len + t0) * d..(l * t_len + t0 + n) * d]);
            }
        }
        let hidden = r.dims[0];
        self.final_norm.add(&taps.final_norm, &r.final_norm[t0 * hidden..(t0 + n) * hidden]);
        self.add_logits(r, t0, n, logits);
    }

    fn add_logits(&mut self, r: &Reference, t0: usize, n: usize, logits: &[f32]) {
        let vocab = r.vocab();
        for i in 0..n {
            let (t, lg) = (t0 + i, &logits[i * vocab..(i + 1) * vocab]);
            let (r32, r16) = (&r.fp32[t * vocab..(t + 1) * vocab], &r.bf16[t * vocab..(t + 1) * vocab]);
            self.fp32.add(lg, r32);
            self.bf16.add(lg, r16);
            self.kl.push(kl(r32, lg));
            let am = top(lg).0;
            let ((a32, m32), (a16, m16)) = (top(r32), top(r16));
            if a32 != am {
                self.fp32_flips.push((t, m32));
            }
            if a16 != am {
                self.bf16_flips.push((t, m16));
            }
        }
    }

    /// Prints the pass and returns whether it is within the limits: every value under `tol`, or
    /// with `logits_only` just the logits.
    fn report(&self, label: &str, tol: f32, margin: f32, logits_only: bool) -> bool {
        let layers = self.errs.len();
        print!("  {label}:");
        if self.final_norm.scale > 0.0 {
            print!(" worst layer per value:");
            for k in 0..8 {
                let (l, worst) = (0..layers).map(|l| (l, self.errs[l][k].rel())).fold((0, 0.0f32), |a, b| if b.1 > a.1 { b } else { a });
                print!(" {} {:.1e}@{}", KINDS[k], worst, l);
            }
            print!(" final_norm {:.1e}", self.final_norm.rel());
        }
        println!(" logits {:.1e}", self.fp32.rel());
        if std::env::var_os("CHECK_LAYERS").is_some() && self.final_norm.scale > 0.0 {
            for l in 0..layers {
                print!("      layer {l:2}:");
                for k in 0..8 {
                    print!(" {} {:.1e}/{:.1e}", KINDS[k], self.errs[l][k].diff, self.errs[l][k].scale);
                }
                println!();
            }
        }
        let mut first_bad = None;
        if !logits_only {
            'scan: for l in 0..layers {
                for k in 0..8 {
                    if !(self.errs[l][k].rel() <= tol) {
                        first_bad = Some(format!("layer {l} {} ({:.1e})", KINDS[k], self.errs[l][k].rel()));
                        break 'scan;
                    }
                }
            }
            if first_bad.is_none() && !(self.final_norm.rel() <= tol) {
                first_bad = Some(format!("final_norm ({:.1e})", self.final_norm.rel()));
            }
        }
        if first_bad.is_none() && !(self.fp32.rel() <= tol) {
            first_bad = Some(format!("logits ({:.1e})", self.fp32.rel()));
        }
        let mut ok = true;
        if let Some(what) = &first_bad {
            ok = false;
            println!("    FAIL: first value over {tol:e}, in execution order: {what}");
        }
        let worst = |flips: &[(usize, f32)]| flips.iter().map(|f| f.1).fold(0.0f32, f32::max);
        println!("    flips vs fp32: {} (largest reference margin {:.3}); vs bf16: {} (largest margin {:.3}); max |logit - bf16| {:.2}",
            self.fp32_flips.len(), worst(&self.fp32_flips), self.bf16_flips.len(), worst(&self.bf16_flips), self.bf16.diff);
        if !self.kl.is_empty() {
            println!("    KL(HF fp32 || ours): mean {:.2e}, worst position {:.2e}", self.kl.iter().sum::<f64>() / self.kl.len() as f64, self.kl.iter().cloned().fold(0.0, f64::max));
        }
        if worst(&self.bf16_flips) >= margin || worst(&self.fp32_flips) >= margin {
            ok = false;
            println!("    FAIL: argmax differs where a reference's margin is not under {margin}");
        }
        ok
    }
}

/// Compares a greedy continuation (`ours[i]` is the token at position `n_prompt + i`) with HF's.
/// A difference is acceptable where the bf16 reference cannot decide.
fn compare_generation(r: &Reference, ours: &[i32], label: &str, margin: f32) -> bool {
    let vocab = r.vocab();
    for (i, &token) in ours.iter().enumerate() {
        let t = r.n_prompt + i;
        if token != r.tokens[t] {
            let m = top(&r.bf16[(t - 1) * vocab..t * vocab]).1;
            println!("  generation ({label}): first differs from HF at generated token {i} (bf16 reference margin there {m:.3})");
            if m >= margin {
                println!("    FAIL: that margin is not under {margin}");
            }
            return m < margin;
        }
    }
    println!("  generation ({label}): identical to HF for all {} tokens", ours.len());
    true
}

/// Greedy continuation with reference steps, given the first generated token.
fn generate_reference(engine: &mut Engine, r: &Reference, slot: i32, first: i32) -> Vec<i32> {
    let mut out = vec![first];
    for t in r.n_prompt..r.t_len - 1 {
        let next = engine.step(&[Row { token: *out.last().unwrap(), slot, pos: t as i32 }], None, None).expect("step")[0];
        out.push(next);
    }
    out
}

/// Greedy continuation with the step graph, given the first generated token.
fn generate_graph(engine: &mut Engine, r: &Reference, slot: i32, first: i32) -> Vec<i32> {
    let mut out = vec![first];
    let more = r.t_len - r.n_prompt - 1;
    if more > 0 {
        out.extend(engine.generate(&[Row { token: first, slot, pos: r.n_prompt as i32 }], more).expect("generate"));
    }
    out
}

struct Limits {
    tap_tol: f32,
    wide_tol: f32,
    margin: f32,
}

fn exact_configuration(weights: &Path, refs: &[Reference], lim: &Limits) -> bool {
    let mut engine = Engine::new(weights, Options { slots: 4, max_ctx: 1024, max_chunk: 512, kv_bf16: 0 }).expect("engine");
    let (layers, vocab) = (engine.config.layers as usize, engine.config.vocab as usize);
    let Limits { tap_tol, wide_tol, margin } = *lim;
    let mut ok = true;
    let mut heads: Vec<Vec<f32>> = Vec::new();
    let mut graph_tokens: Vec<Vec<i32>> = Vec::new();
    for r in refs {
        let (t_len, n_prompt) = (r.t_len, r.n_prompt);
        println!("== {}: {n_prompt} prompt + {} generated tokens", r.name, t_len - n_prompt);
        let (mut taps, mut logits) = (Taps::default(), Vec::new());

        let mut pass = Pass::new(layers);
        let mut head_logits = Vec::new();
        let mut narrow_at_prompt_end = Vec::new();
        for t in 0..t_len {
            engine.step(&[Row { token: r.tokens[t], slot: 0, pos: t as i32 }], Some(&mut logits), Some(&mut taps)).expect("step");
            pass.add(r, t, 1, &taps, &logits);
            if t < BATCH_POSITIONS {
                head_logits.extend_from_slice(&logits);
            }
            if t + 1 == n_prompt {
                narrow_at_prompt_end = logits.clone();
            }
        }
        ok &= pass.report("reference", tap_tol, margin, false);

        let mut pass = Pass::new(layers);
        for (start, len) in chunks(t_len) {
            engine.prefill(0, start as i32, &r.tokens[start..start + len], ALL_LOGITS | EXACT, Some(&mut logits), Some(&mut taps)).expect("prefill");
            pass.add(r, start, len, &taps, &logits);
        }
        ok &= pass.report(&format!("wide exact, chunks {:?}", chunks(t_len).iter().map(|c| c.1).collect::<Vec<_>>()), tap_tol, margin, false);

        let mut pass = Pass::new(layers);
        for t in 0..t_len {
            engine.step_fast(&[Row { token: r.tokens[t], slot: 0, pos: t as i32 }], Some(&mut logits)).expect("step_fast");
            pass.add_logits(r, t, 1, &logits);
        }
        ok &= pass.report("fast", tap_tol, margin, true);

        let mut pass = Pass::new(layers);
        engine.prefill(0, 0, &r.tokens, ALL_LOGITS, Some(&mut logits), Some(&mut taps)).expect("prefill");
        pass.add(r, 0, t_len, &taps, &logits);
        ok &= pass.report("wide", wide_tol, margin, true);
        let wide_at_prompt_end = logits[(n_prompt - 1) * vocab..n_prompt * vocab].to_vec();

        let mut first = 0;
        for t in 0..n_prompt {
            first = engine.step(&[Row { token: r.tokens[t], slot: 1, pos: t as i32 }], None, None).expect("step")[0];
        }
        let ours = generate_reference(&mut engine, r, 1, first);
        ok &= compare_generation(r, &ours, "reference", margin);

        for (start, len) in chunks(n_prompt) {
            first = engine.prefill(1, start as i32, &r.tokens[start..start + len], 0, Some(&mut logits), None).expect("prefill");
        }
        let (mut chunked_vs_whole, mut wide_vs_narrow) = (Err::default(), Err::default());
        chunked_vs_whole.add(&logits, &wide_at_prompt_end);
        wide_vs_narrow.add(&logits, &narrow_at_prompt_end);
        let ours = generate_reference(&mut engine, r, 1, first);
        ok &= compare_generation(r, &ours, "wide prefill of the prompt, then reference", margin);
        println!("  logits after the prompt: wide chunked vs wide in one chunk max |diff| {:.4}; wide vs narrow {:.4} (largest logit {:.1})",
            chunked_vs_whole.diff, wide_vs_narrow.diff, wide_vs_narrow.scale);

        for t in 0..n_prompt {
            first = engine.step_fast(&[Row { token: r.tokens[t], slot: 1, pos: t as i32 }], None).expect("step_fast")[0];
        }
        let ours = generate_graph(&mut engine, r, 1, first);
        ok &= compare_generation(r, &ours, "fused kernels, then the step graph", margin);
        heads.push(head_logits);
        graph_tokens.push(ours);
    }

    if refs.len() >= 2 {
        let (a, b) = (&refs[0], &refs[1]);
        let n = BATCH_POSITIONS.min(a.t_len).min(b.t_len);
        let mut logits = Vec::new();
        let mut differing = 0usize;
        for t in 0..n {
            let rows = [Row { token: a.tokens[t], slot: 2, pos: t as i32 }, Row { token: b.tokens[t], slot: 3, pos: t as i32 }];
            engine.step(&rows, Some(&mut logits), None).expect("step");
            for (row, head) in heads[..2].iter().enumerate() {
                let (got, want) = (&logits[row * vocab..(row + 1) * vocab], &head[t * vocab..(t + 1) * vocab]);
                differing += got.iter().zip(want).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            }
        }
        println!("== two rows ({} + {})", a.name, b.name);
        println!("  reference steps, {n} positions: {differing} logits differ from the single-row run");
        ok &= differing == 0;

        // Graph generation of both continuations at once, from their own prompts.
        let mut firsts = [0; 2];
        for (row, r) in [a, b].into_iter().enumerate() {
            for t in 0..r.n_prompt {
                firsts[row] = engine.step_fast(&[Row { token: r.tokens[t], slot: 2 + row as i32, pos: t as i32 }], None).expect("step_fast")[0];
            }
        }
        let more = (a.t_len - a.n_prompt).min(b.t_len - b.n_prompt) - 1;
        let rows = [Row { token: firsts[0], slot: 2, pos: a.n_prompt as i32 }, Row { token: firsts[1], slot: 3, pos: b.n_prompt as i32 }];
        let both = engine.generate(&rows, more).expect("generate");
        let same = (0..2).all(|row| both[row * more..(row + 1) * more] == graph_tokens[row][1..1 + more]);
        println!("  step graph, {more} tokens each: {}", if same { "same tokens as the single-row runs" } else { "tokens DIFFER from the single-row runs" });
        ok &= same;
        if !(differing == 0 && same) {
            println!("  FAIL: rows are not independent");
        }
    }
    ok
}

fn normal_configuration(weights: &Path, refs: &[Reference], lim: &Limits) -> bool {
    let mut engine = Engine::new(weights, Options { slots: 2, max_ctx: 1024, max_chunk: 512, kv_bf16: 1 }).expect("engine");
    let layers = engine.config.layers as usize;
    let mut ok = true;
    for r in refs {
        println!("== {}", r.name);
        // The yardstick: how far the original model run in bf16 (HF) is from the same weights in fp32,
        // by the same three measures as ours. --wide-tol (1e-1) is about the largest of its logit errors
        // on short prompts; what says whether we are as close to the model as HF bf16 is, is the
        // comparison of the flips and of the KL divergence with these.
        let vocab = r.vocab();
        let diff = r.bf16.iter().zip(&r.fp32).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let scale = r.fp32.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
        let rows = |v: &'static str, t: usize| if v == "fp32" { &r.fp32[t * vocab..(t + 1) * vocab] } else { &r.bf16[t * vocab..(t + 1) * vocab] };
        let flips = (0..r.t_len).filter(|&t| top(rows("fp32", t)).0 != top(rows("bf16", t)).0).count();
        let kls: Vec<f64> = (0..r.t_len).map(|t| kl(rows("fp32", t), rows("bf16", t))).collect();
        println!("  HF bf16 against HF fp32: logits {:.1e}, flips {flips}, KL mean {:.2e}, worst position {:.2e}", diff / scale,
            kls.iter().sum::<f64>() / kls.len() as f64, kls.iter().cloned().fold(0.0, f64::max));
        // The acceptance rule of this configuration: at least as close to the fp32 model as HF's bf16 run.
        let yard_kl = kls.iter().sum::<f64>() / kls.len() as f64;
        let closer = |pass: &Pass| -> bool {
            let ours = pass.kl.iter().sum::<f64>() / pass.kl.len() as f64;
            let good = ours <= yard_kl && pass.fp32_flips.len() <= flips;
            if !good {
                println!("    FAIL: further from HF fp32 than HF bf16 is (mean KL {ours:.2e} against {yard_kl:.2e}, flips {} against {flips})", pass.fp32_flips.len());
            }
            good
        };
        let mut logits = Vec::new();
        let mut pass = Pass::new(layers);
        for t in 0..r.t_len {
            engine.step_fast(&[Row { token: r.tokens[t], slot: 0, pos: t as i32 }], Some(&mut logits)).expect("step_fast");
            pass.add_logits(r, t, 1, &logits);
        }
        ok &= pass.report("fast", lim.wide_tol, lim.margin, true);
        ok &= closer(&pass);

        let mut pass = Pass::new(layers);
        for (start, len) in chunks(r.t_len) {
            engine.prefill(0, start as i32, &r.tokens[start..start + len], ALL_LOGITS, Some(&mut logits), None).expect("prefill");
            pass.add_logits(r, start, len, &logits);
        }
        ok &= pass.report("wide, chunked", lim.wide_tol, lim.margin, true);
        ok &= closer(&pass);

        let mut first = 0;
        for (start, len) in chunks(r.n_prompt) {
            first = engine.prefill(1, start as i32, &r.tokens[start..start + len], 0, None, None).expect("prefill");
        }
        let ours = generate_graph(&mut engine, r, 1, first);
        ok &= compare_generation(r, &ours, "wide prefill of the prompt, then the step graph", lim.margin);
    }
    ok
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: check WEIGHTS.wit REF_DIR [--tap-tol X] [--wide-tol X] [--margin X]");
        return ExitCode::from(2);
    }
    let flag = |name: &str, default: f32| -> f32 {
        args.iter().position(|a| a == name).map_or(default, |i| args[i + 1].parse().expect(name))
    };
    let lim = Limits { tap_tol: flag("--tap-tol", 1e-3), wide_tol: flag("--wide-tol", 1e-1), margin: flag("--margin", 0.5) };
    let weights = Path::new(&args[1]);

    let c = Engine::new(weights, Options { slots: 1, max_ctx: 8, max_chunk: 0, kv_bf16: 1 }).expect("engine").config;
    println!("model: {} layers, hidden {}, {} q / {} kv heads x {}, mlp {}, vocab {}", c.layers, c.hidden, c.q_heads, c.kv_heads, c.head_dim, c.mlp, c.vocab);
    println!("limits: {:e} (reference, wide exact, fast with fp32 KV), {:e} on logits (wide, bf16 KV); argmax may differ from a reference only where its top-2 margin < {}",
        lim.tap_tol, lim.wide_tol, lim.margin);
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&args[2]).expect("REF_DIR").map(|e| e.unwrap().path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    let refs: Vec<Reference> = dirs.iter().map(|d| Reference::load(d, &c)).collect();

    println!("\n#### exact configuration: fp32 KV cache");
    let exact = exact_configuration(weights, &refs, &lim);
    println!("\n#### normal configuration: bf16 KV cache");
    let normal = normal_configuration(weights, &refs, &lim);

    let ok = exact && normal;
    println!("\n{}", if ok { "PASS" } else { "FAIL" });
    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
