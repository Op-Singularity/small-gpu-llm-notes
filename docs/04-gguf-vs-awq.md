# 04 · Why the same model is 2.3 GiB smaller as GGUF

Same 9B, same weights, two containers:

| Format | Size | Vocabulary tensors |
|---|---:|---|
| AWQ 4-bit (compressed-tensors) | **8.45 GiB** | `lm_head` + `embed_tokens`, **BF16** (1.90 GiB each) |
| GGUF Q4_K_M | **~5.6 GiB** | quantised along with everything else |

The difference is not "GGUF is better". It is that AWQ as published here
**ignores the vocabulary on purpose**:

```json
"quantization_config": {
  "config_groups": { "group_0": { "weights": { "num_bits": 4, "group_size": 32, "symmetric": true },
                                  "targets": ["Linear"] } },
  "ignore": ["model.visual...", "lm_head", "..."],
  "quant_method": "compressed-tensors",
  "format": "pack-quantized"
}
```

`targets: ["Linear"]`, and both the vocabulary matrices sit in the ignore list, so
22 % of the model (1.90 GiB) plus another 22 % stays in BF16. K-quants in
llama.cpp quantise those too — which is exactly why a 4-bit GGUF of a 9B is
smaller than a 4-bit AWQ of it.

## Where the bytes go in the AWQ checkpoint

`scripts/model_inventory.py` walks the shard headers:

```
tensors 1189   real bytes 8.45 GiB

  other (quantised body)   854  3.807 GiB  (45.1 %)
  lm_head                    1  1.895 GiB  (22.4 %)
  embed                      1  1.895 GiB  (22.4 %)
  vision                   333  0.849 GiB  (10.1 %)
```

Three useful facts fall out:

1. The body is only 45 % of the file. Compression is not the problem; the
   vocabulary and the vision tower are.
2. The vision tower is 0.85 GiB of a model you are using for text.
3. Removing both still leaves 7.60 GiB — over an 8 GB card's 6.87 GiB ceiling.

## Consequences if you want to shrink an AWQ checkpoint

| Change | Saves | Result |
|---|---:|---:|
| drop the vision tower | 0.85 | 7.60 GiB |
| + `lm_head` → FP8 | 0.95 | 6.65 GiB (KV budget 0.15 GiB ≈ 4 k context) |
| + `lm_head` → 4-bit | 1.43 | 6.17 GiB (KV budget 0.6 GiB ≈ 19 k context) |

Doing the arithmetic is the easy part. Making it *load* is not: see
`03-vllm-limits.md`, and `github.com/…/awq-checkpoint-surgery` for the tooling
that performs the surgery (with byte-exact verification of the packing order).

## A note on ties

Whether the vocabulary is tied is a property of the model, not the format:

```json
"tie_word_embeddings": false
```

The 4B ties them (1.27 GiB total), the 9B does not (3.79 GiB). That single flag
is worth 2.5 GiB of VRAM and is the real reason a 4B runs comfortably on this
card while a 9B does not.
