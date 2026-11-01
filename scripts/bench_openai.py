#!/usr/bin/env python3
"""Benchmark any OpenAI-compatible endpoint across a ladder of prompt lengths.

Time-to-first-token and decode tokens/s are measured separately because on a
memory-tiered engine (experts in host RAM, KV in VRAM) they scale differently:
prompt processing is bandwidth-bound on the PCIe link, decoding is bound by how
much context is actually live.

    python bench_openai.py --base http://127.0.0.1:8080/v1 --model my-model
    python bench_openai.py --base http://127.0.0.1:8080/v1 --model m --ladder 64,1024,4096
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import requests


def stream_once(url: str, model: str, prompt: str, max_tokens: int, timeout: float = 1800.0) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first = None
    usage = None
    with requests.post(url, json=body, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", "replace")
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                if (choice.get("delta") or {}).get("content") and first is None:
                    first = time.perf_counter()
    t1 = time.perf_counter()
    if first is None:
        first = t1

    usage = usage or {}
    out_tokens = usage.get("completion_tokens") or 0
    in_tokens = usage.get("prompt_tokens") or 0
    ttft = first - t0
    decode = (out_tokens - 1) / (t1 - first) if out_tokens > 1 and t1 > first else 0.0
    return {
        "prompt_tokens": in_tokens,
        "completion_tokens": out_tokens,
        "ttft_ms": ttft * 1000.0,
        "decode_tps": decode,
        "total_s": t1 - t0,
    }


def filler(tokens: int, seed: str) -> str:
    """A prompt of roughly `tokens` tokens (Chinese text tokenises ~1.5 chars/token)."""
    unit = f"{seed}的显存占用与上下文长度线性相关，而线性注意力层只维护固定状态。"
    return unit * max(1, int(tokens / 25)) + "\n\n请用一句话总结上面这段话。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ladder", default="0,1024,4096,8192", help="prompt sizes in tokens")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=3, help="runs per rung, median reported")
    args = ap.parse_args()

    url = args.base.rstrip("/") + "/chat/completions"
    rungs = [int(x) for x in args.ladder.split(",") if x.strip()]

    print(f"endpoint {url}\nmodel    {args.model}\n")
    print(f"{'prompt':>8} {'ttft_ms':>9} {'decode':>8} {'prompt_tps':>11} {'total_s':>8}")

    for rung in rungs:
        prompt = "用一句话解释什么是显存带宽。" if rung == 0 else filler(rung, "KV cache")
        runs = []
        for _ in range(args.repeat):
            try:
                runs.append(stream_once(url, args.model, prompt, args.max_tokens))
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"{rung:>8}  failed: {exc}")
                runs = []
                break
        if not runs:
            continue

        ttft = statistics.median(r["ttft_ms"] for r in runs)
        dec = statistics.median(r["decode_tps"] for r in runs)
        taken = statistics.median(r["total_s"] for r in runs)
        ptok = max(r["prompt_tokens"] for r in runs)
        prompt_tps = ptok / (ttft / 1000.0) if ttft > 0 and ptok else 0.0
        print(f"{ptok:>8} {ttft:>9.0f} {dec:>8.2f} {prompt_tps:>11.0f} {taken:>8.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
