# CS336 A1 Project Optimization Notes

This document records engineering problems, debugging evidence, optimization work, and measurable results from the CS336 Assignment 1 implementation. It is written as source material for future resume bullets, project writeups, and interview discussion.

## Project Scope

- Implemented an end-to-end small language model training stack for CS336 Assignment 1.
- Covered tokenizer training, text-to-token preprocessing, data loading, Transformer modules, AdamW, learning-rate scheduling, gradient clipping, checkpointing, training, generation, and training-curve visualization.
- Ran TinyStories locally/on Apple Silicon and OpenWebText on an AutoDL RTX 5090 instance.

## Hardware And Data

### Local Mac

- Used for development, tests, TinyStories experiments, and initial end-to-end validation.
- TinyStories train text: about 2.1 GB.
- TinyStories valid text: about 21 MB.
- TinyStories tokenized train data: about 540.8M tokens, 1.1 GB.
- TinyStories tokenized valid data: about 5.46M tokens, 11 MB.

### AutoDL GPU Instance

- GPU: NVIDIA GeForce RTX 5090, 32 GB VRAM.
- CPU/memory from instance page: 25 vCPU, 90 GB RAM.
- Data disk: 200 GB.
- PyTorch on remote: `2.8.0+cu128`, CUDA available.

### OpenWebText Data

- OWT train text: 12 GB.
- OWT valid text: 277 MB.
- OWT tokenized train data: 2,727,120,452 tokens, 5.1 GB.
- OWT tokenized valid data: 66,401,098 tokens, 127 MB.

## Major Difficulties And Fixes

### 1. Remote Environment And Network Instability

Problem:

- AutoDL could not reliably clone from GitHub through HTTPS.
- Initial `git clone https://github.com/KazeBox33/CS336-learn.git` failed with:
  `GnuTLS recv error (-110): The TLS connection was non-properly terminated`.
- Remote Python environment also needed CUDA-capable PyTorch.

Actions:

- Configured SSH public-key login from the Mac to the AutoDL instance.
- Avoided unstable remote GitHub clone by syncing the local repo to AutoDL with `rsync`.
- Installed `uv` on remote for Python environment management.
- Created an SSH reverse proxy tunnel from AutoDL to the Mac's Clash/Mihomo proxy:
  - Local proxy port discovered: `127.0.0.1:7897`.
  - Remote access path: `socks5h://127.0.0.1:7897`.
- Verified remote access to GitHub and HuggingFace through the proxy.

Resume angle:

- Built a robust remote training workflow under unstable network conditions using SSH key auth, `rsync`, `uv`, and SSH reverse proxy tunneling.

## BPE Tokenizer Training Optimization

### Problem

The initial BPE tokenizer implementation was correct for small tests but too slow or appeared stuck on 12 GB OpenWebText. It produced no intermediate files until the full `train_bpe()` call returned, making it hard to distinguish slow progress from a dead process.

### Root Causes

1. BPE training is CPU-bound, not GPU-bound. RTX 5090 does not accelerate Python text statistics and pair merging.
2. OWT is much larger than TinyStories:
   - TinyStories train text: about 2.1 GB.
   - OWT train text: about 12 GB.
3. The expensive part was not only pretokenization. Profiling showed the merge/update loop dominated total time.
4. Older heap-based BPE versions pushed many stale heap entries:
   - Pair frequencies change repeatedly during a merge step.
   - Python `heapq` has no efficient in-place priority update.
   - Old frequency entries remained in the heap and had to be lazily discarded.

### Optimizations

1. Added pair frequency routing:
   - Maintained `pair_to_pretokens` so each merge updates only affected pretokens.
   - Avoided rescanning every pretoken on every merge.

2. Added max-heap pair selection:
   - Replaced repeated full scans for the best pair with heap-based selection.
   - Used lazy validation against `pair_counts`.

3. Reduced stale heap entries:
   - Collected changed pairs in a `set` during one merge step.
   - Pushed each changed pair back to the heap once per merge step instead of many times.

4. Parallelized pretokenization:
   - Used `ProcessPoolExecutor`.
   - Split large files on special-token boundaries.
   - OWT BPE was run with `--num-processes 16`.

### Measured Result

OWT BPE training completed successfully:

- Input: 12 GB OWT train text.
- Vocab size: 32,000.
- Merges: 31,743.
- Total time: 2,187.5 seconds, about 36.5 minutes.
- Peak RSS: about 15.3 GB.
- Output files:
  - `vocab.pkl`
  - `merges.pkl`
  - `merges.txt`
  - `metrics.json`
  - `profile.txt`

Profile highlights:

- `pretokenize_file`: about 483.7 seconds, about 8.1 minutes.
- `build_pair_stats`: about 78.0 seconds.
- Total `train_bpe`: about 2,173.9 seconds.
- Merge/update loop was the dominant cost after pretokenization.

Resume angle:

- Optimized a Python byte-level BPE trainer with affected-token indexing, heap-based pair selection, stale-entry reduction, and multiprocessing pretokenization; trained a 32K-vocab tokenizer on 12 GB OpenWebText in about 36.5 minutes.

## Text-To-Token Encoding Optimization

### Problem

After BPE training, encoding OWT text into token IDs was initially too slow. A single-process script spent a long time encoding the 12 GB train split and produced no final `.npy` until completion.

### Root Causes

1. `_encode_pretoken()` repeatedly encoded the same common pretokens.
   - Natural language has many repeated pieces such as `" the"`, `" and"`, suffixes, punctuation, and whitespace patterns.
2. The original implementation reconstructed byte tokens repeatedly:
   - `tuple(bytes([byte]) for byte in pretoken)`.
3. The original full-data script accumulated all tokens in memory before saving.
4. A single Python process could not utilize the available CPU cores.

### Optimizations

1. Added pretoken-level LRU cache:
   - `Tokenizer(..., cache_size=1_000_000)`.
   - Repeated pretokens are encoded once and then reused.

2. Precomputed byte tokens:
   - Built the 256 byte-token objects once in `Tokenizer.__init__`.

3. Precompiled the regex pattern:
   - Reused compiled `PATTERN` instead of repeatedly invoking the pattern string.

4. Added true parallel OWT tokenization:
   - Split the input text by line-safe byte ranges.
   - Used `ProcessPoolExecutor` with 16 workers.
   - Each worker encoded its chunk independently.
   - Each worker wrote a temporary `uint16` binary shard.
   - The parent process merged shards into one standard `.npy` using `np.lib.format.open_memmap`.
   - This avoided holding all 2.7B tokens in memory at once.

5. Added progress logging:
   - Printed per-worker progress and final merge statistics.
   - Made long preprocessing jobs observable.

### Correctness Checks

- Verified single-process no-cache output equals cached single-process output.
- Verified cached single-process output equals cached 16-process output.
- `tests/test_tokenizer.py`: 23 passed, 2 skipped.

### Performance Comparison

Benchmark input:

- OWT 500K-line sample.
- Text size: about 61 MB.
- Output tokens: 14,535,849.

| Version | Time | Throughput | Speedup vs no-cache single process |
| --- | ---: | ---: | ---: |
| No cache, single process | 75.13 s | about 0.193M tokens/s | 1.00x |
| LRU cache, single process | 14.07 s | about 1.03M tokens/s | 5.34x |
| LRU cache, 16 processes | 1.59 s | about 9.14M tokens/s | 47.25x |

Full OWT tokenization result:

| Split | Lines | Tokens | Time | Output size |
| --- | ---: | ---: | ---: | ---: |
| Train | 94,568,885 | 2,727,120,452 | 167.15 s | 5.1 GB |
| Valid | 2,301,019 | 66,401,098 | 5.73 s | 127 MB |

Full train throughput:

- About 16.31M tokens/s.

Resume angle:

- Improved OWT text-to-token preprocessing from an impractical single-process Python pipeline to a 16-process cached encoder, achieving 47x speedup on a 500K-line benchmark and tokenizing 12 GB / 2.7B-token OWT train data in about 167 seconds.

## TinyStories End-To-End Validation

### Single-Batch Overfit Test

Purpose:

- Verify model architecture, forward pass, loss, backward pass, optimizer, and update path.

Result:

- Device: Apple Silicon MPS.
- Steps: 300.
- Loss dropped from 9.2346 to 0.0002.
- Runtime: about 5.99 seconds.

Interpretation:

- The model can memorize a fixed minibatch.
- This strongly suggests that core forward/backward/optimizer wiring is correct.

### TinyStories LM Training

Command characteristics:

- Vocab size: 10,000.
- Context length: 256.
- Model: 4 layers, `d_model=512`, 16 heads, `d_ff=1344`.
- Optimizer: AdamW.
- Schedule: warmup + cosine decay.

Training result:

- `max_iters=5000`, batch size 32, LR 3e-4.
- Train/valid loss dropped from about 9.25 to about 1.87-1.90 range.
- Generated coherent TinyStories-style text from prompt `"Once upon a time"`.

Resume angle:

- Built and validated an end-to-end Transformer LM training loop, including sanity-check overfitting, full TinyStories training, checkpointing, generation, and loss-curve visualization.

## OpenWebText LM Training

Current run:

- Model config:
  - Vocab size: 32,000.
  - Context length: 256.
  - `d_model=512`.
  - 4 Transformer layers.
  - 16 attention heads.
  - `d_ff=1344`.
- Training config:
  - Device: CUDA.
  - GPU: RTX 5090 32 GB.
  - Batch size: 32.
  - `max_iters=5000`.
  - LR: 3e-4.
  - Eval interval: 100.
  - Save interval: 500.

Early observed metrics:

| Step | Train loss | Valid loss | LR |
| ---: | ---: | ---: | ---: |
| 0 | 10.3902 | 10.3910 | 0 |
| 100 | 7.1217 | 7.1701 | 2.999e-4 |
| 500 | 6.0133 | 6.0355 | 2.945e-4 |
| 1000 | 5.5909 | 5.5920 | 2.762e-4 |
| 2000 | 5.2578 | 5.2676 | 2.092e-4 |
| 3000 | 5.1406 | 5.1457 | 1.249e-4 |
| 3700 | 5.1196 | 5.1096 | 7.340e-5 |
| 4900 | 5.0261 | 5.0473 | 3.027e-5 |

GPU status during training:

- VRAM used: about 9 GB out of 32 GB.
- GPU utilization: about 95%.

Resume angle:

- Scaled the assignment implementation from TinyStories to OpenWebText on a remote RTX 5090 instance, preparing 2.7B token training data and running CUDA training with stable loss reduction.

## Training Visualization

Problem:

- Needed a lightweight way to inspect `train_loss`, `valid_loss`, and learning rate without adding heavy plotting dependencies.

Action:

- Added `scripts/plot_train_log.py`.
- Reads JSONL training logs.
- Generates an SVG with two panels:
  - Train/valid loss.
  - Learning rate.
- Handles repeated runs in the same log by defaulting to the last run after step reset.

Resume angle:

- Added lightweight experiment observability by logging training metrics to JSONL and generating dependency-free SVG loss/LR curves.

## Potential Resume Bullets

- Implemented a complete Transformer language model training stack for CS336 A1, including byte-level BPE tokenization, data loading, Transformer modules, AdamW, LR scheduling, gradient clipping, checkpointing, generation, and metric visualization.
- Optimized a Python BPE tokenizer trainer with affected-token indexing, heap-based pair selection, stale-entry reduction, and multiprocessing pretokenization, enabling 32K-vocab tokenizer training on 12 GB OpenWebText in about 36.5 minutes.
- Accelerated text-to-token preprocessing with pretoken LRU caching and a 16-process sharded encoder, achieving 47x speedup on a 500K-line OWT benchmark and tokenizing 2.7B OWT train tokens in about 167 seconds.
- Built a robust remote GPU training workflow on AutoDL RTX 5090, handling GitHub/HuggingFace network instability with SSH key auth, `rsync`, `uv`, and SSH reverse proxy tunneling through a local Clash/Mihomo proxy.
- Validated model correctness via single-batch overfit from loss 9.23 to 0.0002 in 300 steps, then trained TinyStories and OpenWebText language models with stable loss reduction and checkpointed generation.

## Interview Talking Points

1. Why BPE training is CPU-bound:
   - It is mostly string processing, counting, Python data structures, and heap/set operations.
   - GPU acceleration helps Transformer matrix multiplications, not Python tokenizer training.

2. Why stale heap entries happen:
   - `heapq` does not support efficient priority update.
   - When pair frequencies change, old heap entries remain.
   - Lazy deletion works but becomes expensive if too many stale entries accumulate.
   - Batching changed pairs with a `set` reduces duplicate stale entries.

3. Why caching speeds up encoding:
   - BPE encoding is deterministic per pretoken.
   - Natural language repeats many pretokens.
   - Caching turns repeated BPE merge work into dictionary lookup.

4. Why memmap-style merging matters:
   - OWT train produced 2.7B token IDs.
   - Keeping all intermediate token arrays in RAM would be wasteful and risky.
   - Worker shards plus `open_memmap` allow scalable output assembly.

5. What changed when scaling from TinyStories to OpenWebText:
   - Data grew from GB-scale small stories to 12 GB web text.
   - Tokenizer training and tokenization became real bottlenecks.
   - Needed remote GPU workflow, preprocessing acceleration, and better observability.
