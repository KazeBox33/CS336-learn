# CS336 Learning Handoff

This document records the current learning state, project progress, and preferred teaching style for continuing this CS336 Assignment 1 project on another machine.

本文档用于在其他电脑或新的对话中继续 CS336 A1 项目时快速交接：包括当前项目进度、学习者知识储备、教学方式偏好和下一步任务。

## Latest Status 2026-06-11

Current active phase:

- Assignment 1 experiments pipeline after core implementation.
- TinyStories raw text has been downloaded locally under `data/`.
- TinyStories BPE tokenizer has been trained with vocab size 10,000 and `<|endoftext|>`.
- Tokenizer artifacts are under `outputs/tinystories_bpe/`:
  - `vocab.pkl`
  - `merges.pkl`
  - `metrics.json`
  - `profile.txt`
- `scripts/tokenize_tinystories.py` has been added to stream-tokenize TinyStories text into `.npy` token arrays using `Tokenizer.encode_iterable(...)` and `array("H")` / `np.uint16`.
- `data/` and `outputs/` are ignored by git because they contain large generated/downloaded artifacts.

Immediate next step:

1. Run TinyStories tokenization:

```bash
cd /Users/xiedongjin/Workspace/learning/CS336-learn/assignment1-basics

uv run python scripts/tokenize_tinystories.py \
  --input-path ../data/TinyStoriesV2-GPT4-train.txt \
  --vocab-path ../outputs/tinystories_bpe/vocab.pkl \
  --merges-path ../outputs/tinystories_bpe/merges.pkl \
  --output-path ../data/tinystories_train_tokens.npy

uv run python scripts/tokenize_tinystories.py \
  --input-path ../data/TinyStoriesV2-GPT4-valid.txt \
  --vocab-path ../outputs/tinystories_bpe/vocab.pkl \
  --merges-path ../outputs/tinystories_bpe/merges.pkl \
  --output-path ../data/tinystories_valid_tokens.npy
```

After tokenization:

- Verify `.npy` files with `np.load(..., mmap_mode="r")`.
- Do a single-minibatch overfit sanity check before long training.
- Then run TinyStories low-resource training on Mac:
  - `vocab_size=10000`
  - `context_length=256`
  - `d_model=512`
  - `d_ff=1344`
  - `num_layers=4`
  - `num_heads=16`
  - `rope_theta=10000`
  - target low-resource token budget around `32 * 5000 * 256 = 40,960,000` tokens.

Experiment roadmap to remember:

1. TinyStories tokenizer training report:
   - elapsed time
   - memory
   - longest token
   - profiling bottleneck
2. TinyStories data tokenization into `.npy`.
3. Single minibatch overfit sanity check.
4. TinyStories base training run with experiment logging.
5. Learning-rate sweep, including at least one divergent run.
6. Batch-size experiment.
7. Generate at least 256 tokens from TinyStories checkpoint and comment on fluency.
8. Ablations:
   - remove RMSNorm
   - post-norm Transformer
   - NoPE
   - SwiGLU vs SiLU
9. OpenWebText:
   - download OWT sample
   - train OWT BPE tokenizer with vocab size 32,000
   - tokenize OWT
   - train same architecture/iterations as TinyStories
   - generate OWT text and compare losses/fluency
10. Optional leaderboard:
   - H100 1.5-hour limit
   - OpenWebText training data only
   - target validation loss below naive baseline 5.0.

## Current Project State

Repository:

- Root repo: `CS336-learn`
- Assignment repo: `assignment1-basics/`
- Main code files currently touched:
  - `assignment1-basics/cs336_basics/bpe.py`
  - `assignment1-basics/cs336_basics/tokenizer.py`
  - `assignment1-basics/cs336_basics/modules.py`
  - `assignment1-basics/cs336_basics/optim.py`
  - `assignment1-basics/tests/adapters.py`

Estimated A1 progress:

- Overall A1 including experiments, written analysis, training loop, generation, and ablations: about 60%.
- Core implementation chain `tokenizer -> TransformerLM -> loss`: about 70%.
- Current active section: A1 Section 4, training a Transformer LM.
- Current active task: custom `AdamW` optimizer.

## Completed Work

Tokenizer:

- Implemented byte-level BPE tokenizer training.
- Handles special tokens as hard boundaries during BPE training.
- Uses GPT-2 style regex pre-tokenization.
- Added multiprocessing pre-tokenization and incremental pair-stat updates.
- Trained a TinyStories BPE tokenizer with vocab size 10,000 and `<|endoftext|>`.
- Implemented tokenizer `encode`, `decode`, special-token handling, and `encode_iterable`.

Transformer model:

- Implemented `Linear`.
- Implemented `Embedding`.
- Implemented `RMSNorm`.
- Implemented `silu`.
- Implemented `SwiGLU`.
- Implemented numerically stable `softmax`.
- Implemented scaled dot-product attention.
- Implemented causal multi-head self-attention.
- Implemented RoPE.
- Implemented multi-head self-attention with RoPE.
- Implemented pre-norm `TransformerBlock`.
- Implemented `TransformerLM`.
- Implemented core `cross_entropy` using `torch.logsumexp` and `torch.gather`.

Validation already passed at different points:

```bash
cd /root/workspace/CS336-learn/assignment1-basics
uv run pytest tests/test_train_bpe.py -q
uv run pytest tests/test_tokenizer.py -q
uv run pytest tests/test_model.py::test_rope -q
uv run pytest tests/test_model.py::test_multihead_self_attention_with_rope -q
uv run pytest tests/test_model.py::test_transformer_block -q
uv run pytest tests/test_model.py::test_transformer_lm tests/test_model.py::test_transformer_lm_truncated_input -q
uv run pytest tests/test_nn_utils.py::test_softmax_matches_pytorch -q
uv run pytest tests/test_nn_utils.py::test_cross_entropy -q
```

## Current WIP

File: `assignment1-basics/cs336_basics/optim.py`

Current AdamW status:

- `AdamW` class exists and inherits from `torch.optim.Optimizer`.
- `__init__` stores AdamW hyperparameters in `defaults`.
- `step()` has been started.
- The user has learned:
  - what `closure` is;
  - what `self.param_groups` means;
  - what `p.grad` is;
  - why `self.state[p]` stores per-parameter optimizer history;
  - why AdamW needs `step`, `exp_avg`, and `exp_avg_sq`.
- The user has not yet reviewed the next AdamW state-update lines in depth:

```python
state["step"] += 1
exp_avg = state["exp_avg"]
exp_avg_sq = state["exp_avg_sq"]
exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
```

Continue from this point. Do not skip directly to a full `step()` implementation unless explicitly asked.

## Next Tasks

Immediate next task:

1. Continue teaching AdamW from the state update section.
2. Explain PyTorch in-place methods such as `mul_`, `add_`, and `addcmul_`.
3. Then implement bias correction and parameter update.
4. Connect `get_adamw_cls` in `tests/adapters.py`.
5. Run:

```bash
cd /root/workspace/CS336-learn/assignment1-basics
uv run pytest tests/test_optimizer.py::test_adamw -q
```

Remaining A1 implementation topics:

- AdamW resource accounting.
- Learning rate schedule.
- Gradient clipping.
- Data loading with `get_batch`.
- Checkpoint save/load.
- Training loop.
- Decoding/generation.
- Experiment logging.
- Larger experiments and ablations.

## Learner Knowledge State

Observed strengths:

- Has backend programming experience, especially Java/C++ style engineering concepts.
- Can follow rigorous engineering explanations when concepts are tied to concrete code.
- Understands the high-level LLM pipeline better now:
  - Unicode/UTF-8/bytes;
  - BPE tokenizer training;
  - tokenizer encode/decode;
  - token IDs;
  - Transformer LM architecture;
  - logits;
  - cross entropy loss;
  - optimizer state at a conceptual level.
- Has successfully worked through many PyTorch model components and official tests.

Still actively learning:

- Python syntax and runtime behavior:
  - tuple syntax like `(batch_size,)`;
  - references vs assignment;
  - dict mutation;
  - generators and `yield`;
  - `with ... as ...`;
  - class creation and type annotations;
  - method chaining.
- PyTorch tensor shape reasoning:
  - `unsqueeze`, `squeeze`;
  - `view`, `transpose`, `contiguous`;
  - broadcasting;
  - `torch.gather`;
  - `torch.logsumexp`;
  - in-place tensor methods ending with `_`.
- PyTorch training mechanics:
  - `.grad`;
  - `loss.backward()`;
  - optimizer `param_groups`;
  - optimizer `state`;
  - `zero_grad`;
  - AdamW moment estimates.

Recent concepts understood:

- `targets` in cross entropy are correct token IDs, not probabilities or one-hot vectors.
- During language-model training, every sequence position predicts the next token in parallel.
- `torch.gather(..., dim=-1, index=targets.unsqueeze(-1))` picks the correct token logit for each sample/position.
- `torch.logsumexp` is the stable form of `log(sum(exp(logits)))`, internally equivalent to subtracting the max.
- `from __future__ import annotations` only affects type annotation evaluation and is not central to optimizer behavior.

## Teaching Preferences

Default language:

- Chinese by default.
- Keep English technical terms when useful, with Chinese explanation.

Teaching style:

- Teach one small step at a time.
- First explain the goal.
- Then show the minimal relevant code snippet.
- Then explain every line from Python/PyTorch syntax and runtime behavior.
- Then explain the algorithmic meaning.
- Then provide the exact test command.

Implementation style:

- Prefer core, efficient, clear implementations.
- Do not add excessive defensive boilerplate.
- Add checks only when they materially improve clarity or prevent likely misuse.
- Avoid intentionally naive implementations.
- Do not directly edit or dump a full implementation unless the user explicitly asks.
- If the user asks to implement/apply code, make focused edits and run relevant tests.

Important workflow:

- If the user asks "next step", do not jump ahead.
- If the user says they have not read a section yet, pause implementation and explain.
- Treat unfinished code as learning progress.
- When asked to save progress, commit WIP clearly instead of silently completing missing pieces.

## Useful Commands

Check status:

```bash
git status --short
```

Run common tests:

```bash
cd /root/workspace/CS336-learn/assignment1-basics
uv run pytest tests/test_model.py::test_transformer_lm tests/test_model.py::test_transformer_lm_truncated_input -q
uv run pytest tests/test_nn_utils.py::test_cross_entropy -q
```

Current likely next test after finishing AdamW:

```bash
cd /root/workspace/CS336-learn/assignment1-basics
uv run pytest tests/test_optimizer.py::test_adamw -q
```
