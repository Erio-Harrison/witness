"""Convert a Qwen2.5 HF checkpoint (bf16 safetensors) into a witness weight file.

usage: python tools/convert.py MODEL_DIR OUT.wit

The file is uploaded to the GPU as one blob, so the layout is the layout the kernels read:

  header   magic "WITNESS1", u32 version, u32 vocab, hidden, layers, q_heads, kv_heads, head_dim, mlp,
           f32 rms_eps, rope_theta, u32 n_tensors, u32 pad, u64 offsets[n_tensors], u64 file_size
  tensors  each aligned to 64 bytes, in this order (n_tensors = 2 + 7 * layers):
             0            embed       bf16 [vocab][hidden]        (also the lm_head: tied)
             1            final_norm  f32  [hidden]
             2 + 7l + 0   norm1       f32  [hidden]
             2 + 7l + 1   qkv_w       bf16 [q + 2 kv][hidden]     rows: q_proj, k_proj, v_proj
             2 + 7l + 2   qkv_b       f32  [q + 2 kv]
             2 + 7l + 3   o_w         bf16 [hidden][hidden]
             2 + 7l + 4   norm2       f32  [hidden]
             2 + 7l + 5   gate_up_w   bf16 [2 mlp][hidden]        row 2i = gate_proj i, row 2i+1 = up_proj i
             2 + 7l + 6   down_w      bf16 [hidden][mlp]

Matrix weights stay bf16 bit for bit. Norm weights and biases are widened to f32, which is exact.
After writing, the file is read back and every tensor is compared with the checkpoint.
Needs numpy only.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC, VERSION, ALIGN = b"WITNESS1", 1, 64


def load_safetensors(path):
    """name -> array view of the file, bf16 tensors as their raw 16-bit patterns (numpy has no bf16).

    The format: u64 length of a JSON header, the header ({name: {dtype, shape, data_offsets}}), then
    the tensors' bytes. Read by hand so that converting needs nothing but numpy.
    """
    data = np.memmap(path, np.uint8, "r")
    n = int(np.frombuffer(data, "<u8", 1, 0)[0])
    header = json.loads(bytes(data[8:8 + n]))
    kinds = {"BF16": np.uint16, "F32": np.float32}
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        assert info["dtype"] in kinds, f"{name}: {info['dtype']} (this converter expects the bf16 checkpoint)"
        t = np.frombuffer(data, kinds[info["dtype"]], offset=8 + n + begin, count=(end - begin) // kinds[info["dtype"]]().itemsize)
        out[name] = (info["dtype"], t.reshape(info["shape"]))
    return out


def bits(t):
    """bf16 tensor -> its raw 16-bit patterns."""
    assert t[0] == "BF16", t[0]
    return t[1]


def widen(t):
    """bf16 -> f32, exactly: a bf16 value is the upper 16 bits of the f32 with the same value."""
    if t[0] == "F32":
        return np.ascontiguousarray(t[1])
    return (t[1].astype(np.uint32) << 16).view(np.float32)


def tensors(cfg, sd):
    """The file's tensors in order, as numpy arrays (uint16 = bf16 bits, float32)."""
    out = [bits(sd["model.embed_tokens.weight"]), widen(sd["model.norm.weight"])]
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        a = p + "self_attn."
        gate, up = bits(sd[p + "mlp.gate_proj.weight"]), bits(sd[p + "mlp.up_proj.weight"])
        gate_up = np.empty((2 * gate.shape[0], gate.shape[1]), np.uint16)
        gate_up[0::2], gate_up[1::2] = gate, up
        out += [
            widen(sd[p + "input_layernorm.weight"]),
            np.concatenate([bits(sd[a + f"{n}_proj.weight"]) for n in "qkv"]),
            np.concatenate([widen(sd[a + f"{n}_proj.bias"]) for n in "qkv"]),
            bits(sd[a + "o_proj.weight"]),
            widen(sd[p + "post_attention_layernorm.weight"]),
            gate_up,
            bits(sd[p + "mlp.down_proj.weight"]),
        ]
    return out


def main():
    model_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = json.loads((model_dir / "config.json").read_text())
    assert cfg["model_type"] == "qwen2" and cfg["tie_word_embeddings"] and cfg["hidden_act"] == "silu", cfg
    sd = load_safetensors(model_dir / "model.safetensors")
    ts = tensors(cfg, sd)
    hidden, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    assert hidden % heads == 0 and hidden % 8 == 0 and cfg["intermediate_size"] % 8 == 0  # 16-byte weight loads

    head_len = 8 + 4 * 8 + 4 * 2 + 4 * 2 + 8 * len(ts) + 8
    offsets, pos = [], head_len
    for t in ts:
        pos = (pos + ALIGN - 1) // ALIGN * ALIGN
        offsets.append(pos)
        pos += t.nbytes
    header = MAGIC + struct.pack(
        "<8I2f2I", VERSION, cfg["vocab_size"], hidden, cfg["num_hidden_layers"], heads,
        cfg["num_key_value_heads"], hidden // heads, cfg["intermediate_size"],
        cfg["rms_norm_eps"], cfg["rope_theta"], len(ts), 0,
    ) + struct.pack(f"<{len(ts)}Q", *offsets) + struct.pack("<Q", pos)
    assert len(header) == head_len

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        for t, off in zip(ts, offsets):
            f.write(b"\0" * (off - f.tell()))
            f.write(t.tobytes())
        assert f.tell() == pos

    # Round trip: read the file back and compare every tensor with a fresh pass over the checkpoint.
    data = np.memmap(out_path, np.uint8, "r")
    assert len(data) == pos
    for i, (t, off) in enumerate(zip(tensors(cfg, sd), offsets)):
        back = np.frombuffer(data, t.dtype, t.size, off).reshape(t.shape)
        assert np.array_equal(back, t), f"tensor {i} differs after round trip"
    print(f"{out_path}: {len(ts)} tensors, {pos / 1e9:.3f} GB, round trip exact")


if __name__ == "__main__":
    main()
