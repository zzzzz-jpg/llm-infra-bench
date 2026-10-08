#!/usr/bin/env python3
"""
capacity_model.py — First-principles checks on your measured numbers.

A benchmark tells you what happened; this tells you whether what happened is
physically plausible. Two calculations, both cheap and both worth running
before you trust (or present) any result:

1. KV-cache budget. How many concurrent sequences of length L actually fit in
   the VRAM you left for the cache? If your measured knee is far below this
   number, you are not memory-bound and should look at scheduling/compute.

       bytes_per_token = 2 (K and V)
                       x num_hidden_layers
                       x num_key_value_heads        <- GQA: not num_attention_heads
                       x head_dim
                       x dtype_bytes

2. Decode roofline. Single-sequence decode is bandwidth bound: every generated
   token reads every weight once.

       tpot_floor_ms ~= (params x bytes_per_param) / hbm_bandwidth x 1000

   If your measured TPOT at concurrency 1 is within ~1.3x of this floor, the
   engine is healthy and further gains must come from batching, not kernels.
   If it is 3x the floor, you are paying overhead (Python, sampling,
   detokenisation, or a PCIe-attached card with no NVLink).

   Prefill is compute bound:

       prefill_flops ~= 2 x params x prompt_tokens
       ttft_floor_ms ~= prefill_flops / (gpu_flops x efficiency) x 1000

Usage
-----
    python3 capacity_model.py --model Qwen/Qwen2.5-7B-Instruct --kv-gb 40
    python3 capacity_model.py --params-b 7.6 --layers 28 --kv-heads 4 --head-dim 128 \
        --kv-gb 40 --context 8192 --gpu A100-80G
    python3 capacity_model.py --gpu-table          # roofline table for common GPUs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# name -> (HBM GB/s, dense FP16/BF16 TFLOPs, VRAM GB)
GPUS: Dict[str, Dict[str, float]] = {
    "A100-40G":  {"bw": 1555, "tflops": 312,  "vram": 40},
    "A100-80G":  {"bw": 2039, "tflops": 312,  "vram": 80},
    "H100-SXM":  {"bw": 3350, "tflops": 989,  "vram": 80},
    "H200":      {"bw": 4800, "tflops": 989,  "vram": 141},
    "L4":        {"bw": 300,  "tflops": 121,  "vram": 24},
    "L40S":      {"bw": 864,  "tflops": 362,  "vram": 48},
    "RTX4090":   {"bw": 1008, "tflops": 165,  "vram": 24},   # dense, no sparsity
    "RTX3090":   {"bw": 936,  "tflops": 71,   "vram": 24},
    "A10G":      {"bw": 600,  "tflops": 125,  "vram": 24},
    "MI300X":    {"bw": 5300, "tflops": 1307, "vram": 192},
}

# Common 7-8B instruct models: params(B), layers, kv_heads, head_dim, hidden
KNOWN: Dict[str, Dict[str, Any]] = {
    "Qwen2.5-7B":        {"params_b": 7.62, "layers": 28, "kv_heads": 4,  "head_dim": 128, "hidden": 3584},
    "Llama-3.1-8B":      {"params_b": 8.03, "layers": 32, "kv_heads": 8,  "head_dim": 128, "hidden": 4096},
    "Mistral-7B-v0.3":   {"params_b": 7.25, "layers": 32, "kv_heads": 8,  "head_dim": 128, "hidden": 4096},
    "GLM-4-9B":          {"params_b": 9.4,  "layers": 40, "kv_heads": 2,  "head_dim": 128, "hidden": 4096},
    "Yi-1.5-9B":         {"params_b": 9.0,  "layers": 48, "kv_heads": 4,  "head_dim": 128, "hidden": 4096},
}


def read_model_config(path: str) -> Optional[Dict[str, Any]]:
    """Pull the architecture out of a HF checkpoint (local dir or cached repo)."""
    candidates = [Path(path) / "config.json"]
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        candidates.append(Path(hf_hub_download(path, "config.json")))
    except Exception:
        pass
    for c in candidates:
        if c.exists():
            cfg = json.loads(c.read_text(encoding="utf-8"))
            heads = cfg.get("num_attention_heads")
            hidden = cfg.get("hidden_size")
            return {
                "layers": cfg.get("num_hidden_layers"),
                "kv_heads": cfg.get("num_key_value_heads", heads),
                "head_dim": cfg.get("head_dim") or (hidden // heads if heads and hidden else None),
                "hidden": hidden,
                "vocab": cfg.get("vocab_size"),
                "max_pos": cfg.get("max_position_embeddings"),
                "torch_dtype": cfg.get("torch_dtype"),
                "source": str(c),
            }
    return None


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int,
                       dtype_bytes: int) -> int:
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def fmt_bytes(n: float) -> str:
    for unit, div in (("TiB", 1 << 40), ("GiB", 1 << 30), ("MiB", 1 << 20),
                      ("KiB", 1 << 10)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:.0f} B"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="HF id or local dir; reads config.json")
    p.add_argument("--preset", help="one of " + ", ".join(KNOWN))
    p.add_argument("--params-b", type=float)
    p.add_argument("--layers", type=int)
    p.add_argument("--kv-heads", type=int)
    p.add_argument("--head-dim", type=int)
    p.add_argument("--kv-dtype", default="fp16", choices=["fp16", "bf16", "fp8", "int8"])
    p.add_argument("--weight-dtype", default="fp16", choices=["fp16", "bf16", "fp8", "int8", "int4"])
    p.add_argument("--kv-gb", type=float, default=0,
                   help="VRAM left for the KV cache after weights (GiB). "
                        "Estimate: (vram x gpu-memory-utilization) - weight_bytes - ~2GiB overhead")
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--gpu", default="A100-80G", help=", ".join(GPUS))
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    p.add_argument("--mfu", type=float, default=0.5,
                   help="achievable fraction of peak FLOPs for prefill (0.3-0.6 typical)")
    p.add_argument("--bw-eff", type=float, default=0.8,
                   help="achievable fraction of peak HBM bandwidth for decode")
    p.add_argument("--measured-tpot-ms", type=float, default=0,
                   help="your measured TPOT at concurrency 1, to compare to the floor")
    p.add_argument("--measured-ttft-ms", type=float, default=0)
    p.add_argument("--gpu-table", action="store_true")
    a = p.parse_args()

    if a.gpu_table:
        print(f"{'GPU':12s} {'HBM GB/s':>9} {'FP16 TFLOPs':>12} {'VRAM GB':>8} "
              f"{'7B fp16 TPOT floor':>20} {'7B fp16 tok/s @1':>17}")
        for name, g in GPUS.items():
            floor = 7.6e9 * 2 / (g["bw"] * 1e9) * 1000 / a.bw_eff
            print(f"{name:12s} {g['bw']:>9.0f} {g['tflops']:>12.0f} {g['vram']:>8.0f} "
                  f"{floor:>18.2f}ms {1000 / floor:>16.1f}")
        print("\nTPOT floor = weight bytes / achievable bandwidth. It is a *floor*: "
              "real engines add sampling, Python and detokenisation overhead, and "
              "batching is what turns it into throughput.")
        return 0

    arch: Dict[str, Any] = {}
    if a.model:
        got = read_model_config(a.model)
        if got:
            arch.update(got)
            print(f"[model] architecture read from {got['source']}")
        else:
            print(f"[model] WARN could not read config.json for {a.model}; "
                  f"use --preset or explicit flags", file=sys.stderr)
    if a.preset:
        if a.preset not in KNOWN:
            sys.exit(f"unknown preset; choose from {', '.join(KNOWN)}")
        arch.update(KNOWN[a.preset])
        arch.setdefault("params_b", KNOWN[a.preset]["params_b"])
    for k, v in (("layers", a.layers), ("kv_heads", a.kv_heads),
                 ("head_dim", a.head_dim)):
        if v:
            arch[k] = v
    if a.params_b:
        arch["params_b"] = a.params_b
    arch.setdefault("params_b", 7.6)
    if not all(k in arch for k in ("layers", "kv_heads", "head_dim")):
        sys.exit("need --layers --kv-heads --head-dim (or --model / --preset)")

    gpu = GPUS.get(a.gpu)
    if not gpu:
        sys.exit(f"unknown --gpu {a.gpu}; choose from {', '.join(GPUS)}")

    kv_bytes_map = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}
    w_bytes_map = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
    kvb = kv_bytes_per_token(arch["layers"], arch["kv_heads"],
                             arch["head_dim"], kv_bytes_map[a.kv_dtype])
    params = arch["params_b"] * 1e9
    weight_bytes = params * w_bytes_map[a.weight_dtype]

    print("=" * 74)
    print(f"Model architecture (per replica, TP={a.tp})")
    print("=" * 74)
    print(f"  params                {arch['params_b']:.2f} B")
    print(f"  layers / kv_heads / head_dim   {arch['layers']} / {arch['kv_heads']} / {arch['head_dim']}")
    print(f"  weight bytes ({a.weight_dtype:5s})  {fmt_bytes(weight_bytes)}")
    print(f"  KV bytes per token ({a.kv_dtype:4s}) {kvb:,} B  = {kvb / 1024:.1f} KiB")
    print(f"      (GQA with {arch['kv_heads']} KV heads, not "
          f"{arch.get('hidden', 0) // max(1, arch['head_dim']) if arch.get('hidden') else '?'} "
          "attention heads — using the wrong number here overestimates KV by 4-8x)")
    print()

    print("=" * 74)
    print("1 · KV-cache budget")
    print("=" * 74)
    if a.kv_gb <= 0:
        avail = gpu["vram"] * 1e9 * 0.90 - weight_bytes / a.tp - 2e9
        print(f"  --kv-gb not given; estimated as "
              f"{gpu['vram']}GB x 0.90 - weights/{a.tp} - 2GB overhead = {fmt_bytes(max(0, avail))}")
    else:
        avail = a.kv_gb * (1 << 30)
    avail = max(0.0, avail)
    for ctx in sorted({a.context, 2048, 4096, 8192, 16384, 32768}):
        per_seq = kvb * ctx / a.tp
        n = int(avail // per_seq) if per_seq else 0
        marker = "   <-- your --context" if ctx == a.context else ""
        print(f"  context {ctx:>6} tok : {fmt_bytes(per_seq):>10} per sequence "
              f"-> {n:>5} concurrent sequences{marker}")
    per_seq = kvb * a.context / a.tp
    n_max = int(avail // per_seq) if per_seq else 0
    print(f"\n  => KV-cache concurrency ceiling at {a.context} tokens: {n_max}")
    print("     Compare with your measured knee. If knee << ceiling, you are NOT")
    print("     KV-bound: look at max-num-seqs, scheduler overhead or compute.")
    print("     If knee ~= ceiling, adding VRAM (or fp8 KV cache, or shorter")
    print("     max-model-len) buys concurrency directly.")
    if a.kv_dtype in ("fp16", "bf16"):
        fp8_n = int(avail // (kvb / 2 * a.context / a.tp))
        gain = fp8_n / n_max if n_max else 0
        print(f"     Switching --kv-cache-dtype fp8 would fit ~{fp8_n} sequences "
              f"in the same VRAM ({gain:.1f}x) at a small quality cost.")
    print()

    print("=" * 74)
    print("2 · Roofline floors (single sequence, TP=" + str(a.tp) + ")")
    print("=" * 74)
    bw = gpu["bw"] * 1e9 * a.bw_eff / a.tp
    fl = gpu["tflops"] * 1e12 * a.mfu / a.tp
    tpot_floor_ms = weight_bytes / a.tp / bw * 1000
    print(f"  decode  : {fmt_bytes(weight_bytes / a.tp)} weights / "
          f"{gpu['bw'] * a.bw_eff / a.tp:.0f} GB/s effective")
    print(f"            TPOT floor = {tpot_floor_ms:.2f} ms/tok "
          f"-> {1000 / tpot_floor_ms:.1f} tok/s ceiling for ONE sequence")
    if a.measured_tpot_ms:
        ratio = a.measured_tpot_ms / tpot_floor_ms
        verdict = ("healthy — overhead is small, scale via batching"
                   if ratio < 1.5 else
                   "acceptable — visible per-step overhead"
                   if ratio < 2.5 else
                   "INVESTIGATE — sampling/detokenisation/PCIe/thermal overhead, "
                   "or the weights are not resident in HBM")
        print(f"            measured TPOT = {a.measured_tpot_ms:.2f} ms "
              f"-> {ratio:.2f}x the floor: {verdict}")

    prompt = 1024
    flops = 2 * params * prompt
    ttft_floor_ms = flops / fl * 1000
    print(f"  prefill : 2 x {arch['params_b']:.2f}B x {prompt} tok = {flops / 1e12:.2f} TFLOP")
    print(f"            at {gpu['tflops'] * a.mfu / a.tp:.0f} TFLOP/s effective "
          f"-> TTFT floor = {ttft_floor_ms:.1f} ms for a {prompt}-token prompt")
    print(f"            ~{ttft_floor_ms * a.context / prompt:.0f} ms at "
          f"{a.context} tokens (prefill scales linearly with prompt length)")
    if a.measured_ttft_ms:
        print(f"            measured TTFT = {a.measured_ttft_ms:.1f} ms "
              f"-> {a.measured_ttft_ms / max(1e-9, ttft_floor_ms):.2f}x the floor "
              "(includes queueing, tokenisation and scheduling, so >1 is expected "
              "under load; compare at concurrency 1)")
    print()

    print("=" * 74)
    print("3 · What this means for the cluster")
    print("=" * 74)
    batch_tok_s = gpu["tflops"] * 1e12 * a.mfu / (2 * params / a.tp)
    print(f"  compute-bound decode ceiling (all sequences together): "
          f"~{batch_tok_s:.0f} tok/s")
    print(f"  bandwidth-bound single-stream:                       "
          f"~{1000 / tpot_floor_ms:.0f} tok/s")
    print(f"  => batching should buy you up to ~{batch_tok_s / max(1e-9, 1000 / tpot_floor_ms):.0f}x "
          "over a single stream before compute saturates.")
    print("     If your measured peak output tok/s is far below the compute ceiling,")
    print("     the batch is not filling: raise the offered rate, check max-num-seqs,")
    print("     or the KV ceiling from section 1 is the real limit.")
    print()
    print("  Cross-check with your measurement:")
    print("    python3 bench/capacity.py bench/results --target-rps <X>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
