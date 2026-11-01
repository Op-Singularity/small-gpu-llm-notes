# 06 · All measurements

Machine: RTX 5060 Laptop (8151 MiB, sm_120), 32 GB RAM, Windows 11 + WSL2 Ubuntu
26.04, driver 617.14. Decode numbers are medians of 3 runs unless noted, 64
output tokens, temperature 0, single stream.

## Bandwidth

| Measurement | Value |
|---|---:|
| Theoretical | 448 GB/s |
| Device copy kernel | **265 GB/s** (59 %) |

## Memory ceiling

| Item | MiB |
|---|---:|
| Total | 8151 |
| CUDA context | ~1050 |
| Free at `vllm serve` startup | 6870 |
| `util 0.92` requires | 7470 → fails |
| `util 0.86` requires | 6820 → works |

## Model footprints

| Model | Container | Size | Fits 6.87 GiB |
|---|---|---:|---|
| Qwen3.5-4B | AWQ 4-bit | 3.76 GiB | yes |
| Qwen3.5-9B | AWQ 4-bit | 8.45 GiB | no |
| Qwen3.5-9B | full BF16 | 17.98 GiB | no |
| Qwen3.5-9B | GGUF Q4_K_M | ~5.6 GiB | yes (llama.cpp) |
| Qwen3.6-35B-A3B | GGUF UD-Q4_K_XL | 22.9 GB | yes, tiered |

### Where the 9B's bytes are

| Part | GiB |
|---|---:|
| body (24 GDN + 8 full attn + MLP, 4-bit) | 3.81 |
| `lm_head` BF16 (untied) | 1.90 |
| `embed_tokens` BF16 | 1.90 |
| vision tower | 0.85 |
| **total** | **8.45** |

## vLLM 4B baseline

`--gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 4`

| Metric | Value |
|---|---:|
| Time to first token (20-token prompt) | 118 ms |
| Decode | **53.3 tok/s** |
| Time to first token (1404-token prompt) | 651 ms |
| Prompt processing | 2157 tok/s |
| VRAM in use | 6450 MiB |
| KV cache reported | 76 933 tokens |

## KV cache, Qwen3.5 (24 linear + 8 full attention)

| Quantity | Value |
|---|---:|
| Per token | 32 KiB |
| Per 4 k context | 128 MiB |
| Per 32 k context | 1024 MiB |
| Concurrency at 32 k, 3.0 GiB KV budget | 2.9 |

## Strata 35B (MoE, `STRATA_Q36_NO_MMQ=1`)

| Quantity | Value |
|---|---:|
| Weights on GPU | 4.40 GiB |
| Experts in host memory | 18.75 GiB |
| Experts in VRAM | 0 |
| VRAM in use | 6410–7010 MiB depending on context |
| Load time | 50–55 s |
| Warm-up | 5.5–6.6 s |

### Decode vs context actually used

| Context in use | Decode |
|---|---:|
| < 512 tokens | 26.2–27.5 tok/s |
| ~1.2 k | 9.9 tok/s |
| ~4 k | 3.5 tok/s |
| ~8 k | 2.2 tok/s |

### Prompt processing (Strata)

| Prompt | tok/s |
|---|---:|
| 935 tokens | 154 |
| 3647 tokens | 202 |
| 6227 tokens | 212 |

### Context ceiling with the 35B

| `--max-context` | Free VRAM | Starts |
|---|---:|---|
| 4096 | 0.80 GiB | yes |
| 8192 | 0.60 GiB | yes |
| 16384 | 0.21 GiB | yes |
| 24576 | 0 GiB | no (32 MiB short) |
| 32768 | 0 GiB | no (32 MiB short) |

## Throughput per watt, roughly

Not measured properly, but for scale: the 4B does 53 tok/s with 6.4 GB resident;
the 35B does 27 tok/s while paging 18.75 GiB of experts from host memory every
token. On a machine where RAM and VRAM are the same order of magnitude, "fits in
VRAM" is worth an order of magnitude in speed, and "fits in RAM" is worth an order
of magnitude in model size. Pick one.
