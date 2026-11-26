//! Safe wrapper over the CUDA core (core/witness_core.h), and the scheduler built on it.
pub mod http;
pub mod sched;
pub mod tok;
pub mod trace;

use std::ffi::{c_char, c_int, CStr, CString};
use std::path::Path;

pub const MAX_ROWS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    pub vocab: u32,
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub mlp: u32,
    pub rms_eps: f32,
    pub rope_theta: f32,
}

impl Config {
    pub fn qkv_dim(&self) -> usize {
        (self.hidden + 2 * self.kv_heads * self.head_dim) as usize
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Independent sequences the KV cache holds.
    pub slots: u32,
    /// Positions per slot.
    pub max_ctx: u32,
    /// Most tokens one `prefill` call takes (0: `prefill` unavailable).
    pub max_chunk: u32,
    /// KV cache element type: 1 = bf16 (normal), 0 = fp32 (exact, for checking).
    pub kv_bf16: u32,
}

/// `prefill`: return logits for every position, not just the last.
pub const ALL_LOGITS: u32 = 1;
/// `prefill`: fp32 products instead of bf16-input tensor-core ones. Slow; for checking.
pub const EXACT: u32 = 2;

/// Feed `token` at position `pos` of sequence `slot`; positions `0..pos` of the slot must be filled.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub token: i32,
    pub slot: i32,
    pub pos: i32,
}

#[repr(C)]
struct RawTaps {
    x_in: *mut f32,
    norm1: *mut f32,
    qkv: *mut f32,
    attn: *mut f32,
    o: *mut f32,
    norm2: *mut f32,
    act: *mut f32,
    mlp: *mut f32,
    final_norm: *mut f32,
}

/// Intermediate values of one step, each `[layers][rows][dim]` (`final_norm`: `[rows][hidden]`).
/// The names match the files written by tools/ref_dump.py.
#[derive(Default)]
pub struct Taps {
    pub x_in: Vec<f32>,
    pub norm1: Vec<f32>,
    pub qkv: Vec<f32>,
    pub attn: Vec<f32>,
    pub o: Vec<f32>,
    pub norm2: Vec<f32>,
    pub act: Vec<f32>,
    pub mlp: Vec<f32>,
    pub final_norm: Vec<f32>,
}

enum RawEngine {}

extern "C" {
    fn wk_create(path: *const c_char, opt: *const Options, out: *mut *mut RawEngine) -> c_int;
    fn wk_destroy(e: *mut RawEngine);
    fn wk_get_config(e: *const RawEngine, out: *mut Config);
    fn wk_step(e: *mut RawEngine, rows: *const Row, n: c_int, logits: *mut f32, argmax: *mut i32, taps: *const RawTaps) -> c_int;
    fn wk_step_fast(e: *mut RawEngine, rows: *const Row, n: c_int, logits: *mut f32, argmax: *mut i32) -> c_int;
    fn wk_generate(e: *mut RawEngine, rows: *const Row, n: c_int, n_tokens: c_int, out: *mut i32) -> c_int;
    fn wk_prefill(e: *mut RawEngine, slot: i32, pos0: i32, tokens: *const i32, n: c_int, flags: u32, logits: *mut f32, argmax: *mut i32, taps: *const RawTaps) -> c_int;
    fn wk_copy_kv(e: *mut RawEngine, src_slot: i32, dst_slot: i32, n: i32) -> c_int;
    fn wk_last_error() -> *const c_char;
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(wk_last_error()) }.to_string_lossy().into_owned()
}

pub struct Engine {
    raw: *mut RawEngine,
    pub config: Config,
}

impl Engine {
    pub fn new(weights: &Path, options: Options) -> Result<Engine, String> {
        let path = CString::new(weights.to_str().ok_or("weights path is not UTF-8")?).map_err(|e| e.to_string())?;
        let mut raw = std::ptr::null_mut();
        if unsafe { wk_create(path.as_ptr(), &options, &mut raw) } != 0 {
            return Err(last_error());
        }
        let mut config = Config::default();
        unsafe { wk_get_config(raw, &mut config) };
        Ok(Engine { raw, config })
    }

    /// Runs one step. Returns the argmax token of each row. `logits`, if given, receives
    /// `[rows][vocab]`; `taps`, if given, is resized and filled.
    pub fn step(&mut self, rows: &[Row], logits: Option<&mut Vec<f32>>, taps: Option<&mut Taps>) -> Result<Vec<i32>, String> {
        let c = self.config;
        let n = rows.len();
        if n == 0 || n > MAX_ROWS {
            return Err(format!("{n} rows, expected 1..={MAX_ROWS}"));
        }
        let logits_ptr = match logits {
            Some(v) => {
                v.resize(n * c.vocab as usize, 0.0);
                v.as_mut_ptr()
            }
            None => std::ptr::null_mut(),
        };
        let raw_taps = taps.map(|t| self.size_taps(t, n));
        let taps_ptr = raw_taps.as_ref().map_or(std::ptr::null(), |t| t as *const RawTaps);
        let mut argmax = vec![0i32; n];
        if unsafe { wk_step(self.raw, rows.as_ptr(), n as c_int, logits_ptr, argmax.as_mut_ptr(), taps_ptr) } != 0 {
            return Err(last_error());
        }
        Ok(argmax)
    }

    /// `step` with the fused kernels: no taps.
    pub fn step_fast(&mut self, rows: &[Row], logits: Option<&mut Vec<f32>>) -> Result<Vec<i32>, String> {
        let n = rows.len();
        if n == 0 || n > MAX_ROWS {
            return Err(format!("{n} rows, expected 1..={MAX_ROWS}"));
        }
        let logits_ptr = match logits {
            Some(v) => {
                v.resize(n * self.config.vocab as usize, 0.0);
                v.as_mut_ptr()
            }
            None => std::ptr::null_mut(),
        };
        let mut argmax = vec![0i32; n];
        if unsafe { wk_step_fast(self.raw, rows.as_ptr(), n as c_int, logits_ptr, argmax.as_mut_ptr()) } != 0 {
            return Err(last_error());
        }
        Ok(argmax)
    }

    /// Greedy generation with no host work between steps: each row's token is fed at its position
    /// and every step's argmax at the next one. Returns `[rows][n_tokens]`.
    pub fn generate(&mut self, rows: &[Row], n_tokens: usize) -> Result<Vec<i32>, String> {
        let mut out = vec![0i32; rows.len() * n_tokens];
        if unsafe { wk_generate(self.raw, rows.as_ptr(), rows.len() as c_int, n_tokens as c_int, out.as_mut_ptr()) } != 0 {
            return Err(last_error());
        }
        Ok(out)
    }

    /// Feeds `tokens` at positions `pos0..` of `slot` in one pass and returns the argmax after the
    /// last one. `logits`, if given, receives the last position's `[vocab]`, or `[n][vocab]` with
    /// `ALL_LOGITS`. Taps are `[layers][n][dim]`.
    pub fn prefill(&mut self, slot: i32, pos0: i32, tokens: &[i32], flags: u32, logits: Option<&mut Vec<f32>>, taps: Option<&mut Taps>) -> Result<i32, String> {
        let n = tokens.len();
        let rows = if flags & ALL_LOGITS != 0 { n } else { 1 };
        let logits_ptr = match logits {
            Some(v) => {
                v.resize(rows * self.config.vocab as usize, 0.0);
                v.as_mut_ptr()
            }
            None => std::ptr::null_mut(),
        };
        let raw_taps = taps.map(|t| self.size_taps(t, n));
        let taps_ptr = raw_taps.as_ref().map_or(std::ptr::null(), |t| t as *const RawTaps);
        let mut argmax = 0i32;
        if unsafe { wk_prefill(self.raw, slot, pos0, tokens.as_ptr(), n as c_int, flags, logits_ptr, &mut argmax, taps_ptr) } != 0 {
            return Err(last_error());
        }
        Ok(argmax)
    }

    /// Copies the K/V of positions `0..n` from `src_slot` to `dst_slot`.
    pub fn copy_kv(&mut self, src_slot: i32, dst_slot: i32, n: usize) -> Result<(), String> {
        if unsafe { wk_copy_kv(self.raw, src_slot, dst_slot, n as i32) } != 0 {
            return Err(last_error());
        }
        Ok(())
    }

    fn size_taps(&self, t: &mut Taps, n: usize) -> RawTaps {
        let c = self.config;
        let per_layer = c.layers as usize * n;
        let size = |v: &mut Vec<f32>, len: usize| {
            v.resize(len, 0.0);
            v.as_mut_ptr()
        };
        RawTaps {
            x_in: size(&mut t.x_in, per_layer * c.hidden as usize),
            norm1: size(&mut t.norm1, per_layer * c.hidden as usize),
            qkv: size(&mut t.qkv, per_layer * c.qkv_dim()),
            attn: size(&mut t.attn, per_layer * c.hidden as usize),
            o: size(&mut t.o, per_layer * c.hidden as usize),
            norm2: size(&mut t.norm2, per_layer * c.hidden as usize),
            act: size(&mut t.act, per_layer * c.mlp as usize),
            mlp: size(&mut t.mlp, per_layer * c.hidden as usize),
            final_norm: size(&mut t.final_norm, n * c.hidden as usize),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { wk_destroy(self.raw) }
    }
}
