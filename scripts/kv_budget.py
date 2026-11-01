#!/usr/bin/env python3
"""Size the KV cache of a model, for hybrid-attention architectures especially.

Most KV calculators count every layer. Qwen3.5 (and Qwen3.6, and Qwen3-Next) are
hybrids: only some layers are full attention, the rest are linear attention with a
fixed-size state. Counting all 32 layers overestimates by 4x on a 32-layer model
with 8 full-attention layers.

    # 4B on this card, calibrated against vLLM's own reported 76 933 tokens
    python kv_budget.py --layers 32 --full-attn 8 --kv-heads 4 --head-dim 256 \
        --vram-mib 8151 --util 0.85 --weights-mib 3891 --check 76933

    # read the numbers straight out of a config.json
    python kv_budget.py --from-config ~/models/qwen3.5-9b-awq/config.json \
        --vram-mib 8151 --util 0.86 --weights-mib 8660
"""
from __future__ import annotations

import argparse
import json
import os

KIB = 1024
MIB = 1024 * 1024


def config_values(path: str) -> dict:
    """Pull the text-tower hyper-parameters out of a (possibly multimodal) config."""
    cfg = json.load(open(path))
    scope = cfg.get("text_config") or cfg

    out = {
        "layers": scope.get("num_hidden_layers"),
        "kv_heads": scope.get("num_key_value_heads") or scope.get("num_attention_heads"),
        "head_dim": scope.get("head_dim"),
        "hidden": scope.get("hidden_size"),
        "full_attn": None,
    }
    if not out["head_dim"] and out["hidden"]:
        heads = scope.get("num_attention_heads") or 1
        out["head_dim"] = out["hidden"] // heads

    # The layer pattern is spelled out in newer configs as a list of strings, or
    # implied by layer_types.
    layer_types = scope.get("layer_types") or []
    if layer_types:
        out["full_attn"] = sum(1 for t in layer_types if "full" in str(t))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-config")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--full-attn", type=int, default=8, help="layers that carry a growing KV cache")
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--dtype-bytes", type=float, default=2.0, help="2 = bf16/fp16, 1 = fp8")
    ap.add_argument("--vram-mib", type=float, default=8151)
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--weights-mib", type=float, default=3891)
    ap.add_argument("--overhead-mib", type=float, default=0)
    ap.add_argument("--check", type=int, default=0, help="tokens reported by the engine, for validation")
    args = ap.parse_args()

    if args.from_config:
        got = config_values(args.from_config)
        args.layers = got["layers"] or args.layers
        args.kv_heads = got["kv_heads"] or args.kv_heads
        args.head_dim = got["head_dim"] or args.head_dim
        if got["full_attn"]:
            args.full_attn = got["full_attn"]
        print(f"from {os.path.basename(os.path.dirname(args.from_config))}: "
              f"layers={args.layers} full_attn={args.full_attn} "
              f"kv_heads={args.kv_heads} head_dim={args.head_dim}\n")

    per_token = args.full_attn * 2 * args.kv_heads * args.head_dim * args.dtype_bytes
    budget = args.vram_mib * args.util - args.weights_mib - args.overhead_mib
    max_tokens = budget * MIB / per_token

    print(f"kv per token     : {per_token:.0f} bytes ({per_token / KIB:.1f} KiB)")
    print(f"kv layers        : {args.full_attn} of {args.layers} "
          f"(the other {args.layers - args.full_attn} are linear attention: fixed state)")
    print(f"vram budget      : {args.vram_mib:.0f} x {args.util:.2f} = {args.vram_mib * args.util:.0f} MiB")
    print(f"minus weights    : {args.weights_mib:.0f} MiB")
    if args.overhead_mib:
        print(f"minus overhead   : {args.overhead_mib:.0f} MiB")
    print(f"left for kv      : {budget:.0f} MiB")
    print(f"max kv tokens    : {max_tokens:.0f}\n")

    print(f"{'context':>10}{'kv/seq':>12}{'concurrency':>14}")
    for ctx in (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144):
        kv_seq = ctx * per_token
        conc = max_tokens / ctx
        flag = "" if conc >= 1 else "   does not fit"
        print(f"{ctx:>10}{kv_seq / MIB:>10.0f}Mi{conc:>14.2f}{flag}")

    if args.check:
        implied_kv = args.check * per_token / MIB
        implied_other = args.vram_mib * args.util - args.weights_mib - implied_kv
        error = (max_tokens - args.check) / args.check * 100
        print(f"\nvalidation against {args.check} tokens reported by the engine:")
        print(f"  implied kv        : {implied_kv:.0f} MiB")
        print(f"  implied other     : {implied_other:.0f} MiB (activations + cuda graphs)")
        print(f"  formula error     : {error:+.2f} %")
        print(f"  recalibrate with  : --overhead-mib {implied_other:.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
