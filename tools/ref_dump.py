"""Dump reference activations from HF transformers for the witness correctness checks.

usage: python tools/ref_dump.py MODEL_DIR OUT_DIR [--prompt 96] [--gen 32]

For each test case, OUT_DIR/<case>/ gets (all little endian, raw):

  meta.bin          u32 T, n_prompt, layers, hidden, qkv, mlp, vocab
  tokens.i32        [T]  prompt (n_prompt tokens) followed by HF's own bf16 greedy continuation (plain argmax)
  logits_bf16.f32   [T][vocab]  the bf16 model on those T tokens (the quality reference)
  logits_fp32.f32   [T][vocab]  the same weights run in fp32 (the kernel reference)
  fp32 run, one value per layer and position, [layers][T][dim]:
    x_in.f32   layer input             norm1.f32  input_layernorm output
    qkv.f32    q_proj|k_proj|v_proj outputs, before RoPE
    attn.f32   attention output, the input of o_proj      o.f32  o_proj output
    norm2.f32  post_attention_layernorm output
    act.f32    silu(gate) * up, the input of down_proj     mlp.f32  down_proj output
  final_norm.f32    [T][hidden]

witness keeps weights in bf16 and activations in fp32, which is exactly what the fp32 run computes,
so those files must match to fp32 rounding. The bf16 run rounds every activation to bf16 and is what
"same output as the original model" is judged against.
"""
import argparse
import corpus
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("out_dir")
ap.add_argument("--prompt", type=int, default=96)
ap.add_argument("--gen", type=int, default=32)
a = ap.parse_args()

rust = corpus.text()
chinese = (ROOT / "tools" / "zh_sample.md").read_text(encoding="utf-8")
lines = rust.splitlines()
CASES = {
    "rust": rust[4000:],
    "chinese": chinese,
    "fim": "<|fim_prefix|>" + "\n".join(lines[200:230]) + "\n<|fim_suffix|>\n" + "\n".join(lines[232:236]) + "<|fim_middle|>",
}

tok = AutoTokenizer.from_pretrained(a.model_dir)
model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16).cuda().eval()
cfg = model.config
L, H, I, V = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
QKV = H + 2 * cfg.num_key_value_heads * (H // cfg.num_attention_heads)


def save(path, t):
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(path)


# bf16 pass: greedy continuation, then one full forward over prompt + continuation.
seqs = {}
with torch.no_grad():
    for name, text in CASES.items():
        ids = tok(text, return_tensors="pt").input_ids[:, : a.prompt].cuda()
        if name == "fim":  # keep the whole prompt so it ends in <|fim_middle|>
            ids = tok(text, return_tensors="pt").input_ids.cuda()
        # Plain greedy decoding with the KV cache. Not model.generate(): the checkpoint's
        # generation_config.json carries a repetition penalty that would change the argmax.
        out = model(ids, use_cache=True)
        full = ids
        for _ in range(a.gen):
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            full = torch.cat([full, nxt], dim=1)
            out = model(nxt, past_key_values=out.past_key_values, use_cache=True)
        d = Path(a.out_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        full.cpu().numpy().astype("<i4").tofile(d / "tokens.i32")
        save(d / "logits_bf16.f32", model(full).logits[0])
        (d / "meta.bin").write_bytes(struct.pack("<7I", full.shape[1], ids.shape[1], L, H, QKV, I, V))
        seqs[name] = full
        print(f"{name}: prompt {ids.shape[1]} + {full.shape[1] - ids.shape[1]} generated: "
              f"{tok.decode(full[0, ids.shape[1]:])!r}")

# fp32 pass over the same tokens, with a hook on every value witness computes.
model = model.float()
taps = {}


def grab(key, pre=False):
    def hook(_m, args, out=None):
        taps.setdefault(key, []).append((args[0] if pre else out)[0])
    return hook


for layer in model.model.layers:
    at = layer.self_attn
    layer.register_forward_pre_hook(grab("x_in", pre=True))
    layer.input_layernorm.register_forward_hook(grab("norm1"))
    for n in "qkv":
        getattr(at, n + "_proj").register_forward_hook(grab(n))
    at.o_proj.register_forward_pre_hook(grab("attn", pre=True))
    at.o_proj.register_forward_hook(grab("o"))
    layer.post_attention_layernorm.register_forward_hook(grab("norm2"))
    layer.mlp.down_proj.register_forward_pre_hook(grab("act", pre=True))
    layer.mlp.down_proj.register_forward_hook(grab("mlp"))
model.model.norm.register_forward_hook(grab("final_norm"))

with torch.no_grad():
    for name, full in seqs.items():
        taps.clear()
        d = Path(a.out_dir) / name
        save(d / "logits_fp32.f32", model(full).logits[0])
        qkv = [torch.cat([taps["q"][l], taps["k"][l], taps["v"][l]], dim=-1) for l in range(L)]
        assert qkv[0].shape[-1] == QKV
        save(d / "qkv.f32", torch.stack(qkv))
        for key in ("x_in", "norm1", "attn", "o", "norm2", "act", "mlp"):
            assert len(taps[key]) == L, (key, len(taps[key]))
            save(d / f"{key}.f32", torch.stack(taps[key]))
        save(d / "final_norm.f32", taps["final_norm"][0])
print("wrote", a.out_dir)
