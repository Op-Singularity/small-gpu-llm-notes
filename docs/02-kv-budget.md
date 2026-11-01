# 02 · KV cache per token, for a model that is half linear attention

Qwen3.5 is a hybrid: of 32 layers, **24 are linear attention (GDN) and 8 are full
attention**. Only the full-attention layers carry a KV cache that grows with
context; the linear layers keep a fixed-size state.

Getting that wrong overestimates the cache by 4×.

```
kv_bytes_per_token = n_full_layers * 2 (K,V) * n_kv_heads * head_dim * dtype_bytes
                   = 8 * 2 * 4 * 256 * 2
                   = 32 768 bytes  =  32 KiB
```

## Checking it against the engine

vLLM reports its own KV budget at startup. On the 4B with
`--max-model-len 32768 --gpu-memory-utilization 0.85` it reported **76 933**
tokens of cache.

Working backwards from the formula:

| Step | Value |
|---|---:|
| VRAM × util | 8151 MiB × 0.85 = 6928 MiB |
| weights (4B AWQ) | 3891 MiB |
| left for KV | 3037 MiB |
| ÷ 32 KiB | 97 184 tokens |
| vLLM reported | 76 933 tokens |
| implied non-KV overhead | 661 MiB |
| error after accounting for that | **0.008 %** |

The 661 MiB discrepancy is activation buffers plus CUDA graphs — a real cost, and
roughly constant, so calibrate once and reuse it.

```bash
python ../scripts/kv_budget.py --layers 32 --full-attn 8 --kv-heads 4 \
    --head-dim 256 --vram-mib 8151 --util 0.85 --weights-mib 3891 --check 76933
```

## Concurrency

`max_tokens ÷ context` is how many sequences fit. On this card with 3.0 GiB for
KV:

| Context | KV per sequence | Concurrent sequences |
|---:|---:|---:|
| 4 K | 128 MiB | 23.7 |
| 8 K | 256 MiB | 11.8 |
| 32 K | 1024 MiB | 2.9 |
| 128 K | 4096 MiB | 0.7 |

Note the last row: at 128 K **one** sequence does not fit. Native 262 K context
advertised by a model is a statement about the architecture, not about your card.

## The same math on an 8 GB card for a 9B

The 9B has an **untied** vocabulary: `embed_tokens` and `lm_head` are separate,
1.895 GiB each in BF16.

| Component | Size |
|---|---:|
| body (24 GDN + 8 full attention + MLP, 4-bit AWQ) | 3.81 GiB |
| `lm_head` (BF16, untied) | 1.90 GiB |
| `embed_tokens` (BF16) | 1.90 GiB |
| vision tower (multimodal) | 0.85 GiB |
| **total** | **8.45 GiB** |

Against a 6.87 GiB ceiling. The 4B escapes this because it *ties* the two
matrices (1.27 GiB total instead of 3.79).

That is the whole story of "why does 9B not fit when 4B does": not the parameter
count, the vocabulary.
