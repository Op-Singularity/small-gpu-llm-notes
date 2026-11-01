# Running a 35B MoE on an 8 GB laptop GPU

Field notes from trying to serve models far larger than the card allows — on one
RTX 5060 Laptop (8 GB), 32 GB of RAM, Windows + WSL2.

Everything below is measured on that machine. The point of the repo is not "it
works" but **why it does or does not**, with the numbers that decide it.

## The short version

| Model | Runtime | Weights | Where they live | Decode |
|---|---|---:|---|---:|
| Qwen3.5-4B AWQ | vLLM (WSL) | 3.76 GiB | all in VRAM | **53 tok/s** |
| Qwen3.5-9B AWQ | vLLM | 8.45 GiB | does not fit | — |
| Qwen3.6-35B-A3B (MoE) | Strata | 22.9 GB | 4.4 GiB VRAM + 18.75 GiB RAM | **27 tok/s** |

The 35B is nine times the 4B and still answers at half its speed, because only
3B parameters are active per token and the experts stream from host memory.

## What is in here

| Doc | What it answers |
|---|---|
| [`docs/01-hardware.md`](docs/01-hardware.md) | What this GPU actually does (measured, not spec sheet) |
| [`docs/02-kv-budget.md`](docs/02-kv-budget.md) | KV cache per token for a hybrid-attention model |
| [`docs/03-vllm-limits.md`](docs/03-vllm-limits.md) | Two real limits in vLLM 0.31 that stop a 9B on 8 GB |
| [`docs/04-gguf-vs-awq.md`](docs/04-gguf-vs-awq.md) | Why the same model is 2.3 GiB smaller as GGUF |
| [`docs/05-strata-35b-on-8gb.md`](docs/05-strata-35b-on-8gb.md) | Finding the one environment variable that makes a 35B fit |
| [`docs/06-measurements.md`](docs/06-measurements.md) | All numbers in one table |

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
told me a 9B is 22% embedding and 22% output head, which is the whole reason it
does not fit.

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

## License

MIT. Measurements are my own; model weights are not included.
