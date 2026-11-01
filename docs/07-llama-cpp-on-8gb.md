# 07 · The 9B that actually works: GGUF + llama.cpp

After the AWQ route failed on vLLM (see `03-vllm-limits.md`), the same 9B runs
comfortably — and faster than the 4B — as a 5.28 GiB Q4_K_M GGUF under llama.cpp.

| Model | Runtime | Weights | Where | Context | Decode |
|---|---|---:|---|---:|---:|
| Qwen3.5-4B AWQ | vLLM | 3.76 GiB | all in VRAM | 32k | 53 tok/s |
| **Qwen3.5-9B Q4_K_M** | **llama.cpp** | **5.28 GiB** | **all in VRAM** | **64k** | **60 tok/s** |

The 9B is 2.4× the parameters and still 13 % faster, because 5.28 GiB fits
entirely in the card — no CPU offload, no tiering, no Marlin repack.

```
$ llama-bench -m Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128

| model                |  size    | params | backend | ngl | type_k | type_v | fa |  test |      t/s |
| qwen35 9B Q4_K - M   | 5.28 GiB | 8.95 B | CUDA    |  99 |   q8_0 |   q8_0 |  1 | pp512 |  2505.19 |
| qwen35 9B Q4_K - M   | 5.28 GiB | 8.95 B | CUDA    |  99 |   q8_0 |   q8_0 |  1 | tg128 |     60.39 |
```

Decode holds up with context in use: **51.7 tok/s at 4k**, **49.3 at 8k**. 64k
context loads with q8_0 KV cache; peak VRAM is 6062 MiB of 8151.

## Why GGUF is smaller than AWQ (again, with numbers)

The AWQ checkpoint is 8.45 GiB because `targets: ["Linear"]` with `lm_head` in
the ignore list leaves both vocabulary matrices in BF16 — 1.90 GiB each.
K-quants quantise the vocabulary too, so the same 9B lands at 5.28 GiB. That
2.3 GiB is the entire difference between "does not fit" and "fits with room for
a 64k cache".

## Building llama.cpp for sm_120: three walls

This machine has CUDA 13.4 available only as pip wheels, plus a system CUDA 12.8.
Neither works out of the box.

### Wall 1 · system CUDA 12.8 vs gcc 15

```
error: -- unsupported GNU version! gcc versions later than 14 are not supported!
```

Pin the host compiler to gcc-14 (`-ccbin g++-14`) and you hit the next one:

```
error: exception specification is incompatible with that of previous function "cospi"
```

CUDA 12.8's `math_functions.h` declares `cospi`/`sinpi` without `noexcept`, while
glibc 2.41 in Ubuntu 26.04 declares them with it. Not fixable without patching
headers — abandon the system toolkit.

### Wall 2 · pip toolkit has no unversioned `.so`

CMake reports:

```
Target "ggml-cuda" links to: CUDA::cublas
but the target was not found.
```

The pip `nvidia/cu13` package ships `libcublas.so.13` but no `libcublas.so`
symlink, so `find_library(NAMES cublas)` fails. One line fixes it:

```bash
cd $CUDA_HOME/lib
for b in libcublas libcublasLt libcudart; do ln -sf ${b}.so.* ${b}.so; done
```

### Wall 3 · CCCL refuses the version skew

```
cccl/cuda/std/__cccl/cuda_toolkit.h:41:8: error:
    #error "CUDA compiler and CUDA toolkit headers are incompatible"
```

The check compares `CUDART_VERSION` against the compiler version. Upstream leaves
an explicit escape hatch for exactly this case:

```
-DCMAKE_CUDA_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
```

### The configuration that works

```bash
export CUDA_HOME=$HOME/vllm-env/lib/python3.12/site-packages/nvidia/cu13
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DCMAKE_CUDA_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
cmake --build build -j16
```

Verification:

```
$ llama-cli --list-devices
Available devices:
  CUDA0: NVIDIA GeForce RTX 5060 Laptop GPU (8123 MiB, 7033 MiB free)
```

## Serving

```bash
llama-server -m Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 \
  --host 127.0.0.1 --port 8081 --jinja -a qwen3.5-9b
```

OpenAI-compatible on `http://127.0.0.1:8081/v1`, model name `qwen3.5-9b`.

## One practical note: thinking mode eats your tokens

Qwen3.5 emits `reasoning_content` before `content`. With a 64-token budget the
reply looks empty because the budget went to reasoning.

| Mode | Time | Tokens out | What you get |
|---|---:|---:|---|
| thinking off | 1.0 s | 31 | the answer directly |
| thinking on | 6.5 s | 300 | reasoning first, `content` empty |

Turn it off per request:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

Measured with it off: **30–40 tok/s** end-to-end including time to first token.
