# small-gpu-llm-notes

[![checks](https://github.com/Op-Singularity/small-gpu-llm-notes/actions/workflows/checks.yml/badge.svg)](https://github.com/Op-Singularity/small-gpu-llm-notes/actions/workflows/checks.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![hardware](https://img.shields.io/badge/hardware-RTX%205060%20Laptop%208%20GB-informational)
![models](https://img.shields.io/badge/models-4B%20%7C%209B%20%7C%2035B%20MoE-informational)
![docs](https://img.shields.io/badge/docs-6%20notes-lightgrey)

Field notes from trying to serve models far larger than the card allows — on one
RTX 5060 Laptop (8 GB), 32 GB of RAM, Windows + WSL2.

Everything below is measured on that machine. The point of the repo is not "it
works" but **why it does or does not**, with the numbers that decide it.

## The short version

| Model | Runtime | Weights | Where they live | Decode |
|---|---|---:|---|---:|
| Qwen3.5-4B AWQ | vLLM (WSL) | 3.76 GiB | all in VRAM | **53 tok/s** |
| Qwen3.5-9B AWQ | vLLM | 8.45 GiB | does not fit | — |
| **Qwen3.5-9B GGUF Q4_K_M** | **llama.cpp** | **5.28 GiB** | **all in VRAM** | **60 tok/s** |
| Qwen3.6-35B-A3B (MoE) | Strata | 22.9 GB | 4.4 GiB VRAM + 18.75 GiB RAM | **27 tok/s** |

The 35B is nine times the 4B and still answers at half its speed, because only
3B parameters are active per token and the experts stream from host memory.

## Contents

| # | Note | What it answers |
|---|---|---|
| 01 | [What this GPU actually does](docs/01-hardware.md) | Measured bandwidth and the real VRAM ceiling — 265 GB/s, not 448; 6.87 GiB, not 8 |
| 02 | [KV cache per token](docs/02-kv-budget.md) | Why a hybrid-attention model needs 32 KiB per token, not 128, and how the formula checks out to 0.008 % |
| 03 | [Two limits in vLLM 0.31](docs/03-vllm-limits.md) | CPU offload vs Triton, and the text-only path that dies allocating the output head — six attempts, all reproduced |
| 04 | [GGUF vs AWQ](docs/04-gguf-vs-awq.md) | Why the same 9B is 2.3 GiB smaller as GGUF: the vocabulary stays BF16 in AWQ |
| 05 | [**Making a 35B MoE start on 8 GB**](docs/05-strata-35b-on-8gb.md) | The interesting one: a constant 1159 MiB shortfall, traced to a staging buffer in the engine's source |
| 06 | [All measurements](docs/06-measurements.md) | Every number in one place, with the machine state it was taken in |

## Reproduce the measurement

```bash
pip install requests
python scripts/bench_openai.py --base http://127.0.0.1:8080/v1 --model <name>
```

It reports time-to-first-token and decode tokens/s for a ladder of prompt
lengths, which is the only honest way to describe a memory-tiered engine: the
speed depends on how much context is actually live.

```bash
python scripts/model_inventory.py ~/models/qwen3.5-9b-awq
```

Walks the safetensors headers and prints where the bytes went — the step that
told me a 9B is 22 % embedding and 22 % output head, which is the whole reason it
does not fit.

```bash
python scripts/kv_budget.py --layers 32 --full-attn 8 --kv-heads 4 --head-dim 256 \
    --vram-mib 8151 --util 0.85 --weights-mib 3891 --check 76933
```

Same arithmetic as [kv-cache-lab](https://github.com/Op-Singularity/kv-cache-lab),
in Python, for when you want it in a notebook rather than a binary.

## The one paragraph worth reading

`docs/05` is the interesting one. A 35B MoE refused to start at any context
length with a constant 1159 MiB shortfall. The engine's own error message said
"fewer slots or a shorter context", and neither helped, because the number did
not depend on either. Reading the source showed the shortfall was a *staging
buffer* — `2 * (max_layer + tail)` — sized for a prompt-processing path (MMQ)
that streams a whole layer's experts into VRAM. That path can be switched off
with an environment variable, the staging buffer goes to zero, and the model
starts with every expert in host memory.

The lesson is not the flag. It is that a resource error message states the
symptom; the allocation that produced it may be somewhere else entirely.

## Related

- [kv-cache-lab](https://github.com/Op-Singularity/kv-cache-lab) — the same KV budget arithmetic as a C++17 CLI with tests
- [awq-checkpoint-surgery](https://github.com/Op-Singularity/awq-checkpoint-surgery) — the tooling behind `docs/04`, including byte-exact verification of int4 packing

## License

MIT. Measurements are my own; model weights are not included.
