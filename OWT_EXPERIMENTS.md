# OpenWebText Experiments

This file records only OpenWebText experiments. It is meant for comparing OWT runs across model size, context length, optimizer settings, throughput, memory use, and validation loss.

## Dataset And Tokenizer

OpenWebText data on AutoDL:

| File | Size | Notes |
| --- | ---: | --- |
| `/root/autodl-tmp/cs336/data/owt_train.txt` | 12 GB | Raw training text |
| `/root/autodl-tmp/cs336/data/owt_valid.txt` | 277 MB | Raw validation text |
| `/root/autodl-tmp/cs336/data/owt_train_tokens.npy` | 5.1 GB | Tokenized train split |
| `/root/autodl-tmp/cs336/data/owt_valid_tokens.npy` | 127 MB | Tokenized valid split |

Token counts:

```text
train tokens = 2,727,120,452
valid tokens = 66,401,098
```

Tokenizer:

```python
vocab_size = 32_000
special_tokens = ["<|endoftext|>"]
```

Tokenizer training and encoding notes:

- BPE trained on the 12 GB OWT train text.
- Final OWT BPE training time: about 36.5 minutes.
- Parallel cached tokenization:
  - train split: about 167.15 seconds.
  - valid split: about 5.73 seconds.
- Main acceleration came from pretoken LRU caching plus 16-process sharded encoding.

## Experiment Summary

Throughput is approximate because the logged wall-clock time includes evaluation overhead.

| Run | Status | Script / Output Dir | Context | Model | Params | Batch | Steps | Tokens / Step | Tokens Seen | Latest Eval | Train Loss | Valid Loss | Checkpoint Iter | Runtime | Approx Throughput | GPU Mem |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| OWT smoke | done | `owt_lm_smoke` | 256 | 4L, d=512, h=16, d_ff=1344 | 45.2M | 32 | 20 | 8,192 | 0.16M | 15 | 9.1575 | 9.1818 | 20 | 1.5 sec | smoke only | n/a |
| OWT 5k baseline | done | `owt_lm` | 256 | 4L, d=512, h=16, d_ff=1344 | 45.2M | 32 | 5,000 | 8,192 | 40.96M | 4,900 | 5.0261 | 5.0473 | 5,000 | 314.8 sec | about 130k tok/s | about 9 GB observed |
| OWT 20k baseline | done | `owt_lm_pdf_baseline` | 256 | 4L, d=512, h=16, d_ff=1344 | 45.2M | 64 | 20,000 | 16,384 | 327.68M | 19,500 | 4.2312 | 4.2442 | 20,000 | 37.6 min | about 142k tok/s | about 9 GB observed |
| OWT 512ctx 6layer | done | `owt_lm_512ctx_6layer` | 512 | 6L, d=768, h=12, d_ff=2048 | 91.6M | 32 | 50,000 | 16,384 | 819.20M | 49,500 | 3.6937 | 3.6905 | 50,000 | about 201.1 min at step 49,500 | about 67k tok/s | 22.5 GB |

Notes:

- `Tokens / Step = batch_size * context_length`.
- `Tokens Seen = Tokens / Step * planned_steps` for completed runs, and target tokens for the in-progress 512-context run.
- Parameter counts were measured by instantiating the actual `TransformerLM` implementation.
- The 256-context 4-layer OWT model has 45,224,448 parameters.
- The 512-context 6-layer OWT model has 91,629,312 parameters.
- `Checkpoint Iter` can lag behind `Latest Eval` while a run is still in progress because checkpoints are saved every 2,000 steps.

## Run Details

### 1. OWT Smoke

Purpose:

- Confirm CUDA, data paths, model construction, loss, backward pass, and checkpointing before longer OWT training.

Config:

```python
vocab_size = 32_000
context_length = 256
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1_344

batch_size = 32
num_steps = 20
max_lr = 3e-4
min_lr_ratio = 0.1
weight_decay = 0.01
betas = (0.9, 0.95)
max_grad_norm = 1.0
```

Result:

```text
latest eval step = 15
train loss = 9.1575
valid loss = 9.1818
checkpoint iteration = 20
```

Takeaway:

- This was only a smoke test, not a meaningful quality experiment.

### 2. OWT 5k Baseline

Purpose:

- First real OpenWebText training run after tokenization.

Config:

```python
vocab_size = 32_000
context_length = 256
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1_344

batch_size = 32
num_steps = 5_000
tokens_per_step = 8_192
max_lr = 3e-4
min_lr_ratio = 0.1
weight_decay = 0.01
betas = (0.9, 0.95)
max_grad_norm = 1.0
```

Result:

```text
latest eval step = 4,900
train loss = 5.0261
valid loss = 5.0473
checkpoint iteration = 5,000
runtime = 314.8 seconds
approx throughput = 40.96M tokens / 314.8s = about 130k tokens/s
```

Takeaway:

- This run was only slightly better than the naive baseline and showed that more training or a stronger setup was needed.

### 3. OWT 20k Baseline

Purpose:

- Increase training tokens while keeping the same 256-context model family.

Config:

```python
vocab_size = 32_000
context_length = 256
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1_344

batch_size = 64
num_steps = 20_000
tokens_per_step = 16_384
max_lr = 3e-4
min_lr_ratio = 0.1
warmup_iters = 1_000
weight_decay = 0.01
betas = (0.9, 0.95)
max_grad_norm = 1.0
```

Result:

```text
latest eval step = 19,500
train loss = 4.2312
valid loss = 4.2442
checkpoint iteration = 20,000
runtime = about 37 min 35 sec
approx throughput = 327.68M tokens / 2255.3s = about 145k tokens/s
```

Change from 5k baseline:

```text
batch_size: 32 -> 64
steps: 5,000 -> 20,000
tokens_per_step: 8,192 -> 16,384
target tokens seen: 40.96M -> 327.68M
valid loss: 5.0473 -> 4.2442
```

Takeaway:

- The loss improved mainly because the model saw about 8x more training tokens.
- Model architecture and context length did not change.

### 4. OWT 512ctx 6layer

Purpose:

- Test the high-impact leaderboard-aligned changes:
  - context length 256 -> 512.
  - transformer depth 4 -> 6.
  - width 512 -> 768.
  - more total target tokens.

Config:

```python
vocab_size = 32_000
context_length = 512
d_model = 768
num_layers = 6
num_heads = 12
d_ff = 2_048

batch_size = 32
num_steps = 50_000
tokens_per_step = 16_384
max_lr = 3e-4
min_lr_ratio = 0.1
warmup_iters = 1_000
weight_decay = 0.01
betas = (0.9, 0.95)
max_grad_norm = 1.0
```

Measured model size:

```text
total parameters = 91,629,312
about 91.6M parameters
```

Final result as of 2026-06-12 16:54 CST:

```text
status = done
latest eval step = 49,500
train loss = 3.6937
valid loss = 3.6905
checkpoint iteration = 50,000
runtime at latest eval = 12,066.3 seconds = about 201.1 minutes
GPU = RTX 5090 32GB
GPU memory = about 22.5GB / 32GB
GPU utilization = 99%
approx throughput = 49,500 * 16,384 / 12,066.3 = about 67k tokens/s
```

Change from 20k baseline:

```text
context_length: 256 -> 512
num_layers: 4 -> 6
d_model: 512 -> 768
d_ff: 1,344 -> 2,048
batch_size: 64 -> 32
tokens_per_step: unchanged at 16,384
target tokens seen: 327.68M -> 819.20M
valid loss so far: 4.2442 -> 3.7111
final valid loss: 4.2442 -> 3.6905
throughput: about 145k tok/s -> about 67k tok/s
GPU memory: about 9GB -> about 22.5GB
```

Takeaway:

- Quality improved significantly, but training throughput dropped because the model is larger and the sequence length is longer.
- This run is not just "more steps"; it primarily tests a stronger context/model configuration.
- The final validation loss improved by about 0.55 compared with the 20k 256-context baseline.
- After this run finishes, the next efficient improvements to test should be recorded before code changes:
  - bf16 autocast.
  - weight tying.
  - possibly larger batch size after bf16 reduces memory.
  - `torch.compile` if it works with the current training loop.

## Next Experiment Candidates

Do not change all of these at once. Record baseline metrics before each change.

1. `bf16` autocast only.
2. Weight tying only.
3. `bf16 + weight tying` together if we decide they are tightly coupled for memory/speed.
4. Increase batch size after bf16, for example 32 -> 64.
5. Try 8 layers after efficiency improvements are in place.
6. Consider Muon only after the AdamW baseline and efficiency improvements are recorded.
