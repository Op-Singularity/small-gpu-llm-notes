# 01 · What this GPU actually does

RTX 5060 Laptop, 8 GB, driver 617.14, WSL2 Ubuntu 26.04, PyTorch 2.13 + CUDA 13.0.

## Spec sheet vs measurement

| | Spec | Measured |
|---|---:|---:|
| VRAM | 8 GiB (8151 MiB usable) | 8151 MiB |
| Memory bandwidth | 448 GB/s | **265 GB/s** |

A simple copy kernel reaches 265 GB/s, about **59 % of the theoretical figure**.
That number matters more than the spec sheet for two reasons:

1. It sets the decode-time ceiling. Decoding one token reads every active weight
   once, so `tokens/s ≤ 265 GB/s ÷ bytes_read_per_token`.
2. It tells you how much of the gap to the spec sheet is *static* (a mobile part
   that throttles, a laptop memory configuration) rather than per-run noise.

Measure it yourself before trusting any throughput estimate:

```bash
# a 1 GiB copy, timed; bandwidth = 2 * size / time
python - <<'PY'
import torch, time
a = torch.empty(256 << 20, dtype=torch.float32, device='cuda')  # 1 GiB
b = torch.empty_like(a)
for _ in range(3): b.copy_(a)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(10): b.copy_(a)
torch.cuda.synchronize()
gb = (a.numel() * 4 * 2 * 10) / 1e9
print(f"{gb / (time.perf_counter() - t):.0f} GB/s")
PY
```

## What the card can hold

The engine never gets the full 8151 MiB:

| Item | Cost |
|---|---:|
| CUDA context | ~1.05 GiB |
| vLLM `--gpu-memory-utilization 0.92` requests 7.3 GiB | — |
| Real free memory at startup | **6.87 GiB** |

That is why `util 0.92` fails outright on this card:

```
ValueError: Free memory on device cuda:0 (6.87/7.93 GiB) on startup is less than
desired GPU memory utilization (0.92, 7.3 GiB).
```

The ceiling on this machine is **util ≈ 0.86** (6.82 GiB), and any budget has to
be built from that, not from 8 GiB.
