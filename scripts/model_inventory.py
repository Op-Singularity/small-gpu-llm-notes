#!/usr/bin/env python3
"""Report where a safetensors checkpoint spends its bytes.

Reads only the shard headers, so it is instant even on a 20 GB model and needs no
GPU. Useful for answering "why does this 9B not fit when that 4B does" — the
answer is usually the vocabulary, not the parameter count.

    python model_inventory.py ~/models/qwen3.5-9b-awq
    python model_inventory.py ~/models/qwen3.5-9b-awq --top 20
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import defaultdict


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(length).decode("utf-8"))


def tensors_of(root: str) -> dict[str, dict]:
    """name -> {dtype, shape, bytes} across every shard."""
    out: dict[str, dict] = {}
    index = os.path.join(root, "model.safetensors.index.json")
    if os.path.exists(index):
        files = sorted(set(json.load(open(index))["weight_map"].values()))
    else:
        files = [f for f in os.listdir(root) if f.endswith(".safetensors")]

    for name in files:
        header = read_header(os.path.join(root, name))
        header.pop("__metadata__", None)
        for tensor, meta in header.items():
            offsets = meta.get("data_offsets") or [0, 0]
            out[tensor] = {
                "dtype": meta.get("dtype", "?"),
                "shape": meta.get("shape", []),
                "bytes": offsets[1] - offsets[0],
                "file": name,
            }
    return out


def classify(name: str) -> str:
    low = name.lower()
    if ".visual." in low or low.startswith("visual.") or "vision" in low:
        return "vision"
    if "embed_tokens" in low:
        return "embedding"
    if "lm_head" in low or "output" in low:
        return "lm_head"
    if low.endswith((".weight_packed", ".qweight", ".weight_scale", ".qzeros", ".weight_shape")):
        return "body (quantised)"
    if "layers." in low or "mtp." in low:
        return "body"
    return "other"


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--config", action="store_true", help="also print the quantisation config")
    args = ap.parse_args()

    if not os.path.isdir(args.path):
        print(f"not a directory: {args.path}", file=sys.stderr)
        return 2

    tensors = tensors_of(args.path)
    if not tensors:
        print("no safetensors found", file=sys.stderr)
        return 2

    total = sum(t["bytes"] for t in tensors.values())
    groups: dict[str, list[int]] = defaultdict(list)
    for name, meta in tensors.items():
        groups[classify(name)].append(meta["bytes"])

    print(f"{args.path}\n{len(tensors)} tensors, {human(total)}\n")
    print(f"{'group':<20}{'tensors':>8}{'size':>12}{'share':>8}")
    for group, sizes in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
        subtotal = sum(sizes)
        print(f"{group:<20}{len(sizes):>8}{human(subtotal):>12}{subtotal / total * 100:>7.1f}%")

    print(f"\nlargest tensors")
    for name, meta in sorted(tensors.items(), key=lambda kv: -kv[1]["bytes"])[: args.top]:
        shape = "x".join(str(d) for d in meta["shape"])
        print(f"  {human(meta['bytes']):>10}  {meta['dtype']:<7}{shape:<20}{name}")

    cfg_path = os.path.join(args.path, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        tied = cfg.get("tie_word_embeddings")
        print(f"\ntie_word_embeddings: {tied}")
        quant = cfg.get("quantization_config") or {}
        if quant:
            groups_cfg = quant.get("config_groups") or {}
            for key, value in groups_cfg.items():
                w = value.get("weights") or {}
                print(f"  {key}: {w.get('num_bits')}-bit, group {w.get('group_size')}, "
                      f"targets {value.get('targets')}")
            ignored = quant.get("ignore") or []
            big = [x for x in ignored if "lm_head" in x or "embed" in x]
            print(f"  ignore list: {len(ignored)} entries"
                  + (f", including vocabulary: {big}" if big else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
