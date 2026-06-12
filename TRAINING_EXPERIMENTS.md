# CS336 A1 Training Experiments

This file records model-training runs so we can compare future experiments by configuration and final validation loss.

## Experiment Protocol

Before changing any training configuration or efficiency-related implementation, record the current baseline in this file.

Each change should include:

- Run name and git commit.
- Exact model config: vocab size, context length, layer count, `d_model`, heads, `d_ff`, tying/mixed precision/compile settings if used.
- Exact training config: batch size, steps, tokens per step, optimizer, LR schedule, weight decay, gradient clipping.
- Hardware: GPU model, VRAM, CPU/memory if relevant.
- Efficiency metrics before the change:
  - tokens per step.
  - wall-clock seconds per eval interval or per 1k steps.
  - estimated tokens/second.
  - peak/typical GPU memory.
  - GPU utilization.
- Quality metrics before the change:
  - latest train loss.
  - latest valid loss.
  - latest logged step.
  - checkpoint iteration.
- The single intended change, or at most two tightly related changes.
- Result after the change using the same metrics, so we can compare effect instead of relying on memory.

Rule of thumb:

- Do not change steps, model size, optimizer, mixed precision, and batch size all at once.
- Prefer one-factor or two-factor ablations so the result is interpretable.
- If a long run is already in progress, let it finish and record it before starting a new experiment.

## Summary Table

| Run | Dataset | Device | Vocab | Context | Model | Batch | Steps | Tokens Seen | Last Eval Step | Train Loss | Valid Loss | Checkpoint Iter | Runtime |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyStories baseline | TinyStories | Apple Silicon MPS | 10,000 | 256 | 4L, d=512, h=16, d_ff=1344 | 32 | 5,000 | 40.96M | 4,900 | 1.8958 | 1.8751 | 5,000 | about 53.6 min |
| OWT short baseline | OpenWebText | RTX 5090 32GB | 32,000 | 256 | 4L, d=512, h=16, d_ff=1344 | 32 | 5,000 | 40.96M | 4,900 | 5.0261 | 5.0473 | 5,000 | about 5.2 min |
| OWT 20k baseline | OpenWebText | RTX 5090 32GB | 32,000 | 256 | 4L, d=512, h=16, d_ff=1344 | 64 | 20,000 | 327.68M | 19,500 | 4.2312 | 4.2442 | 20,000 | about 37.6 min |
| OWT 512ctx 6layer | OpenWebText | RTX 5090 32GB | 32,000 | 512 | 6L, d=768, h=12, d_ff=2048 | 32 | 50,000 | 819.20M | 49,500 | 3.6937 | 3.6905 | 50,000 | about 201.1 min at step 49,500 |

Notes:

- `Tokens Seen = batch_size * context_length * steps`.
- `Last Eval Step` is the last step written to `train_log.jsonl`.
- `Checkpoint Iter` is the iteration saved inside `checkpoint.pt`.
- The 20k OWT run has checkpoint iteration 20,000 even though the last logged evaluation is step 19,500.

## Run Details

### TinyStories Baseline

Purpose:

- Verify the end-to-end training pipeline on an easier dataset.
- Confirm that tokenizer, data loading, model, optimizer, checkpointing, logging, and generation work together.

Model config:

```python
vocab_size = 10_000
context_length = 256
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1_344
rope_theta = 10_000.0
```

Training config:

```python
batch_size = 32
num_steps = 5_000
max_lr = 3e-4
min_lr_ratio = 0.1
weight_decay = 0.01
betas = (0.9, 0.95)
max_grad_norm = 1.0
device = "mps"
```

Observed result:

```text
last eval step: 4,900
train loss: 1.8958
valid loss: 1.8751
checkpoint iteration: 5,000
```

Takeaway:

- TinyStories is much easier than OpenWebText because it has simpler syntax, smaller topic variety, and more repeated story patterns.

### OpenWebText Short Baseline

Purpose:

- Quick CUDA sanity check on the remote RTX 5090 instance.
- Confirm the OpenWebText tokenized data, model, optimizer, and logs all work before a longer run.

Model config:

```python
vocab_size = 32_000
context_length = 256
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1_344
rope_theta = 10_000.0
```

Training config:

```python
batch_size = 32
num_steps = 5_000
max_lr = 3e-4
min_lr_ratio = 0.1
weight_decay = 0.01
betas = (0.9, 0.95)
max_grad_norm = 1.0
device = "cuda"
```

Observed result:

```text
last eval step: 4,900
train loss: 5.0261
valid loss: 5.0473
checkpoint iteration: 5,000
```

Takeaway:

- OpenWebText is substantially harder than TinyStories.
- The run beat the naive baseline loss of about 5.00 only slightly, so it was mainly a short pipeline validation run.

### OpenWebText 20k Baseline

Purpose:

- Run a more meaningful OpenWebText baseline after preprocessing and remote-training setup were working.
- Compare longer training against the 5k short baseline.

Model config:

```python
vocab_size = 32_000
context_length = 256
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1_344
rope_theta = 10_000.0
```

Training config:

```python
batch_size = 64
num_steps = 20_000
max_lr = 3e-4
min_lr_ratio = 0.1
warmup_iters = 1_000
weight_decay = 0.01
betas = (0.9, 0.95)
max_grad_norm = 1.0
device = "cuda"
```

Observed result:

```text
last eval step: 19,500
train loss: 4.2312
valid loss: 4.2442
checkpoint iteration: 20,000
runtime: about 37 min 35 sec
```

Takeaway:

- Increasing training from 5k to 20k and batch size from 32 to 64 improved OWT valid loss from about 5.05 to about 4.24.
- The training direction is correct, but this is still far from the CS336 leaderboard range around 3.x.

## Current Hypotheses For Lower Loss

Most likely high-impact changes:

1. Increase `context_length` from 256 to 512.
2. Increase model capacity, for example from 4 layers to 6 or 8 layers.
3. Increase `d_model` from 512 to 768 if VRAM allows.
4. Train longer, for example 50k or 100k steps.
5. Tune `betas`, `weight_decay`, and learning-rate schedule after the architecture and context length are closer to the leaderboard setup.

Suggested next OWT experiment:

```python
vocab_size = 32_000
context_length = 512
d_model = 768
num_layers = 6
num_heads = 12
d_ff = 2_048

batch_size = 32
num_steps = 50_000
max_lr = 3e-4
min_lr_ratio = 0.1
warmup_iters = 1_000
weight_decay = 0.01
betas = (0.9, 0.95)
max_grad_norm = 1.0
```

If memory is sufficient on RTX 5090 32GB, try `batch_size = 64`.
