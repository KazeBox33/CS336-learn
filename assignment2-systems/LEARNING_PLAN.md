# CS336 Spring 2026 Assignment 2 Learning Plan

This directory contains the official Spring 2026 Assignment 2 starter imported
from `stanford-cs336/assignment2-systems`.

- Upstream commit: `ca8bc81a59b70516f7ebb2da4808daade877c736`
- Handout: `cs336_assignment2_systems.pdf`

This file records the learning order only. It does not contain assignment
solutions.

## Part 1: Profiling and Benchmarking

1. Understand Transformer parameter and activation memory accounting.
2. Build a synchronized forward/backward benchmark with warmup.
3. Add NVTX ranges and inspect execution with Nsight Systems.
4. Study mixed-precision accumulation and benchmark autocast.
5. Capture CUDA memory snapshots and explain peak-memory components.

## Part 2: Single-GPU Memory

1. Understand which tensors autograd saves for backward.
2. Study operator fusion and how it reduces intermediate allocations.
3. Implement memory-optimal activation/gradient checkpointing.
4. Measure the compute-for-memory tradeoff.

## Part 3: GPU Kernels

1. Benchmark ordinary PyTorch attention.
2. Compare eager execution with `torch.compile`.
3. Learn Triton's program model, block pointers, masks, and memory layout.
4. Implement FlashAttention-2 forward behavior.
5. Implement the required backward behavior.
6. Verify numerical correctness and benchmark forward/backward performance.
7. Optionally move more backward work into custom Triton kernels.

## Part 4: Distributed Data Parallel

1. Benchmark single-node collective communication.
2. Implement and benchmark naive DDP.
3. Measure a flat-gradient all-reduce baseline.
4. Overlap individual-parameter gradient communication with backward compute.
5. Inspect overlap and idle regions with Nsight Systems.

## Part 5: Optimizer State Sharding

1. Shard optimizer ownership across data-parallel ranks.
2. Apply local optimizer updates to owned parameters.
3. Communicate updated parameters so all ranks remain consistent.
4. Measure memory, communication, and iteration-time changes.

## Part 6: Fully Sharded Data Parallel

1. Shard model parameters, gradients, and optimizer state.
2. All-gather parameters before computation.
3. Reduce-scatter gradients after backward computation.
4. Support mixed-precision compute while retaining suitable master state.
5. Verify equivalence with non-sharded training and measure peak memory.

## Part 7: Parallelism Analysis

1. Analyze alternate ring all-reduce.
2. Derive data-parallel memory and communication costs.
3. Derive FSDP memory and communication costs.
4. Derive tensor-parallel forward/backward communication.
5. Analyze two-dimensional FSDP plus tensor parallelism.

## Part 8: Final Benchmark

1. Establish a correct baseline.
2. Change one optimization at a time.
3. Record hardware, software versions, shapes, dtype, and timing method.
4. Compare correctness, peak memory, communication, and wall-clock time.

## Working Rule

For every task, proceed in this order:

1. Concept lesson and concrete example.
2. Relevant syntax and tensor/rank data flow.
3. Exact implementation location and adapter boundary.
4. Learner implementation or explicitly requested Codex implementation.
5. Focused correctness test.
6. Benchmark and result record when required.
7. Commit and push the completed task.
