# CS336-learn

This repository tracks my self-study of Stanford CS336: Language Modeling from Scratch.

## Goals

- Build a working understanding of modern language model training from raw text to generation.
- Implement the core components by hand: byte-level BPE tokenization, Transformer LM modules, loss, optimizer, training loop, checkpointing, and decoding.
- Learn the engineering workflow around tests, debugging, profiling, and experiment logging.

## Current Progress

- Watched Lecture 1: overview and tokenization.
- Watched Lecture 2: PyTorch and resource accounting.
- Watched Lecture 3: Transformer architectures and hyperparameters.
- Set up the Spring 2025 Assignment 1 workspace in `assignment1-basics/`.
- Started Assignment 1 Section 2, BPE tokenizer training.

## Assignment 1 Status

Current focus: `train_bpe`.

Completed:

- Project environment configured with `uv`.
- VS Code/Pylance/Ruff setup for WSL development.
- `tests/adapters.py` now routes `run_train_bpe` to `cs336_basics.bpe.train_bpe`.
- Initial byte vocabulary setup started in `cs336_basics/bpe.py`.

Next:

- Finish special-token-aware pre-tokenization.
- Build `pretoken_counts`.
- Implement BPE pair counting and merge loop.
- Pass `uv run pytest tests/test_train_bpe.py`.
