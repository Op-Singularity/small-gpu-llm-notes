# 03 · Two limits in vLLM 0.31 that stop a 9B on 8 GB

Both were found by trying, and both reproduce. vLLM 0.31.0, torch 2.13.0+cu130,
RTX 5060 Laptop (sm_120), 8151 MiB.

## 1 · CPU offload is incompatible with this model's Triton kernels

The obvious escape for "weights are 1.5 GiB too big" is `--cpu-offload-gb`:
keep most of the model on the GPU and stream the rest from host memory.

```bash
vllm serve ~/models/qwen3.5-9b-awq \
  --gpu-memory-utilization 0.86 --max-model-len 4096 \
  --cpu-offload-gb 2.5 --enforce-eager
```

```
ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)
```

The linear-attention layers (GDN) run through Triton kernels. Offloaded weights
live in CPU memory, and the kernel receives a host pointer it cannot dereference.
This is not a tuning problem: **offload and this architecture do not compose** in
this version.

Note the first thing you hit before that, with a plausible-looking `0.92`:

```
ValueError: Free memory on device cuda:0 (6.87/7.93 GiB) on startup is less than
desired GPU memory utilization (0.92, 7.3 GiB).
```

The CUDA context (~1.05 GiB) is already allocated when the check runs, so 0.92
can never be satisfied on this card.

## 2 · The text-only architecture path dies during weight creation

Removing the vision tower and switching `architectures` to `Qwen3_5ForCausalLM`
produces a checkpoint that is 6.06 GiB — comfortably under the 6.87 GiB ceiling.
It still does not load:

```
File ".../qwen3_5.py", line 349, in __init__
    self.lm_head = ParallelLMHead(
File ".../vocab_parallel_embedding.py", line 55, in create_weights
    torch.empty(
RuntimeError: CUDA driver error: device not ready
```

`load_model` wraps model construction in `with target_device:` — i.e. weights are
created on **real** GPU memory, not on the `meta` device:

```python
# vllm/model_executor/model_loader/base_loader.py
target_device = torch.device(load_device)
with set_default_torch_dtype(model_config.dtype):
    with target_device:
        model = self.create_model(...)          # allocations happen here
```

So the transient footprint is the whole model plus the un-tied `lm_head` again,
and at 8 GB the allocation fails — reported as a driver error rather than OOM.

## What I tried, in order

| Attempt | Result |
|---|---|
| `--cpu-offload-gb 2.5` | Triton cannot read CPU weights |
| `--gpu-memory-utilization 0.92` | startup check fails (context already resident) |
| text-only checkpoint, `ParallelLMHead` on `meta` device | vLLM's global DeviceContext overrides the local `with` |
| `create_weights(..., device="cpu")` | `torch_call_dispatcher("aten::empty") API call failed` — this torch build rejects the keyword |
| zero-length placeholder for `lm_head` (replaced by `tie_weights`) | creation passes, weights load in 1.24 s, then Marlin repack needs more |
| `--linear-backend triton` | accepted, but `process_weights_after_loading` still repacks through Marlin |

Peak VRAM during the last attempt, sampled at 0.5 s:

```
Loading weights took 1.24 seconds
peak: 7642 MiB / 8151 MiB        (93.7 %)
RuntimeError: CUDA driver error: device not ready
```

## The conclusion that generalises

vLLM's load path needs roughly **2× the weight footprint** in transient VRAM
(create → load → repack into the kernel layout). A model that "fits" by the
arithmetic of `weights + KV` can still fail to load on a card this small.

This is why the same model as GGUF starts without complaint: llama.cpp mmaps the
weights and never needs a second copy. See `04-gguf-vs-awq.md`.
