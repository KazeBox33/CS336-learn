# CS336 Assignment 2 Collaboration Guide

This directory is used for guided study of CS336 Spring 2026 Assignment 2:
Systems.

## Teaching Style

- Use Chinese by default and keep important English systems terms.
- Follow `cs336_assignment2_systems.pdf` in order.
- Before presenting or writing implementation code, teach the relevant systems
  concept with a concrete example.
- Explain unfamiliar Python, PyTorch, CUDA, Triton, and distributed-programming
  syntax line by line.
- Connect each optimization to its effect on runtime, memory traffic,
  communication, or peak memory.
- Distinguish forward, backward, optimizer, and communication costs.
- Use small tensor-shape and multi-rank examples when introducing a new idea.

## Implementation Workflow

- By default, show the exact file, location, focused code snippet, purpose, and
  test command so the learner can implement it.
- When the learner explicitly asks Codex to implement, fix, configure, test,
  commit, or push, Codex may perform those actions directly.
- Review the learner's current code before proposing the next task.
- Point out and correct small mistakes during review instead of silently
  replacing large sections.
- Keep real implementations in `cs336_systems/`; `tests/adapters.py` should
  only import and connect those implementations to the tests.
- Preserve unfinished learner code and unrelated worktree changes.
- After a task is complete, run the narrowest relevant test first, then broader
  tests when the change affects shared behavior.
- Commit and push completed tasks when requested.

## Learning Integrity

- Do not copy third-party assignment solutions.
- The official handout, lecture material, PyTorch documentation, Triton
  documentation, and CUDA/NVIDIA profiling documentation may be used.
- Code should be understood before it is accepted: explain the algorithm,
  tensor shapes, synchronization assumptions, and performance tradeoffs.
- Benchmark claims must record hardware, dtype, tensor/model shape, warmup,
  repetitions, and synchronization method.

## Experiment Artifacts

- Generate write-up tables, plots, and other result images programmatically
  from recorded benchmark data; do not manually transcribe values or edit
  measured figures by hand.
- Keep the plotting/table-generation code, structured raw results, and
  generated outputs together so every figure can be reproduced after an
  experiment changes.
- Use tools such as pandas and matplotlib unless a task calls for a different
  format. A plotting script should be able to regenerate the latest table or
  image with one command.
- Label every performance figure with the relevant units and experimental
  variables, and record hardware, software versions, model/tensor shape,
  dtype, warmup, repetitions, and synchronization method alongside the source
  data.
- When a benchmark task is completed, regenerate and inspect its tables and
  images before treating the task as finished.

## Environment

- macOS can be used for reading, editing, type checking, and CPU-compatible
  tests.
- Triton kernels, CUDA memory profiling, Nsight Systems, NCCL collectives, and
  multi-GPU experiments require a Linux NVIDIA development machine.
- Do not invent CUDA performance results from the Mac. Record only measured
  results from the actual training or benchmark machine.

## Spring 2026 Route

Use `LEARNING_PLAN.md` as the high-level checklist. The handout remains the
source of truth whenever the checklist and assignment differ.
