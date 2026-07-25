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
