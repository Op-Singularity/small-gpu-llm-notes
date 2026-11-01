# 05 · Making a 35B MoE start on an 8 GB card

The interesting part of this repo. Model: Qwen3.6-35B-A3B, UD-Q4_K_XL, 22.9 GB of
GGUF. Runtime: [Strata](https://github.com/Niko1221/Strata) 0.2.9 (its own
engine, VRAM + RAM tiering). Card: 8 GB.

## The wall

Every configuration failed with the same message, and the number never moved:

```
strata-q36: context 4096 cells, prompt chunk 512, MTP block loaded, 0.75 GiB of VRAM free
strata-q36: not enough VRAM left for the hybrid experts' buffers and slots
            (1159 MiB needed, 771 MiB free after the weights, the context and the sessions)
```

| Attempt | Free VRAM |
|---|---:|
| 16 384 context | 0.00 GiB |
| 8 192 context | 0.57 GiB |
| 4 096 context | 0.77 GiB |
| 2 048 context | 0.85 GiB |
| `--kv q4_0` (KV to 4-bit) | no change |
| `--kv-grow` (KV on demand) | no change |
| `--expert-vram 0` (no expert slots) | no change |
| `--hybrid-rows 32` | no change |
| `--tune quick` (official auto-tuner) | "failed" |

The message says *"use fewer slots or a shorter context"*. Cutting the context by
8× recovered 0.35 GiB and the requirement stayed at 1159 MiB. So the requirement
was not about context, and not about slots.

## Reading the allocator

`src/qwen35/q36_engine.cu`, in `place_experts()`:

```cpp
const size_t staging = E.use_mmq && E.main_hybrid ? 2 * (max_layer + kMmqTail) : 0;
size_t slot_bytes = budget2 > staging ? budget2 - staging : 0;

const size_t Tn = E.tmax, Nn = g.n_embd, Kn = m.top_k;
size_t need = staging + Tn * Kn * Nn * 4 + Tn * Kn * 4;
for (Layer* L : all)
    need += (S > 0 ? (size_t) S * L->stride + kMmqTail : 0);

if (need > fr) { err = "not enough VRAM left for the hybrid experts' buffers and slots (...)"; }
```

`staging` is **twice the largest layer's expert blob**. It exists for the MMQ
prompt path, which streams an entire layer's experts into VRAM to run an int8
matmul over the prompt chunk. It is a fixed cost: independent of context length
and of how many expert slots are cached. That is why every knob above moved
nothing.

And the switch is an environment variable, a few hundred lines earlier:

```cpp
use_mmq = MMQ::built() && std::getenv("STRATA_Q36_NO_MMQ") == nullptr;
```

## The fix

```bat
set STRATA_Q36_NO_MMQ=1
```

```
strata-q36: prompt experts on the native grouped kernels
strata-q36: context 8192 cells, prompt chunk 512, MTP block loaded, 0.60 GiB of VRAM free
strata-q36: hybrid experts: 0 of 256 per layer in VRAM (0.00 GiB),
            all 18.75 GiB in pageable host memory, the rest of a batch of <= 512 rows
            on 9 CPU threads, bigger prompt chunks streamed (0.00 GiB staging); 50.4 s
strata-q36: warmed up in 6.6 s
READY 8192
```

With the staging buffer gone the accounting is: 4.40 GiB of non-expert weights on
the GPU, 18.75 GiB of routed experts in host memory, **zero** experts in VRAM,
and every token streams the experts it needs over PCIe.

## What that costs

| Actual context | Decode |
|---|---:|
| short conversation (< 512 tokens) | **26–27 tok/s** |
| ~1.2 k tokens | ~10 tok/s |
| ~4 k tokens | ~3.5 tok/s |
| ~8 k tokens | ~2.2 tok/s |

Two things worth knowing:

- **Cold start is 5× slower.** The first request after load reads 18.75 GiB of
  experts from disk; the first measurement was 4.7 tok/s and the second 26.6.
  Warm up before believing a number.
- **`--max-context` does not slow short chats.** 8 k and 16 k settings both give
  26.6–27.1 tok/s on a short prompt. Only the context actually in use costs time.
  So the practical "dynamic context" is: allocate the largest safe ceiling and let
  usage decide the speed.

## The ceiling is context

| Context | Free VRAM | Starts |
|---|---:|---|
| 4 k | 0.80 GiB | yes |
| 8 k | 0.60 GiB | yes |
| 16 k | 0.21 GiB | yes |
| 24 k / 32 k | 0 GiB (32 MiB short) | no |

Note how the gap at 24 k and 32 k is the *same* 32 MiB: once the allocator fills
the card it reports a constant shortfall, which is what made the original message
so misleading in the first place.

## Generalisation

- A resource error tells you the symptom. The allocation behind it can be a
  buffer for a completely different code path.
- If a reported requirement does not change when you change the obvious input,
  you are looking at a constant, not a formula. Find it in the source.
- The right question is rarely "how do I make this smaller", it is "which
  allocation am I actually paying for".
