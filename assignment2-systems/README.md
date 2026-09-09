# CS336 Spring 2026 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.13.13
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 78 packages in 168ms
Python 3.13.13 (main, Apr  7 2026, 20:49:46) [Clang 22.1.1 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cs336_basics
...
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Single-node All-Reduce benchmark

Run the assignment's 12 NCCL configurations on a Linux machine with at least
six NVIDIA GPUs:

```sh
uv run python -m cs336_systems.distributed_benchmark \
  --backend nccl \
  --world-sizes 2 4 6 \
  --sizes-mib 1 10 100 1024 \
  --warmup-steps 5 \
  --measurement-steps 20
```

Generate the latency and bandwidth figures exclusively from the recorded JSON:

```sh
uv run python -m cs336_systems.plot_distributed_benchmark
```

For a small CPU-only correctness smoke test, use Gloo with a deliberately tiny
tensor instead of the assignment's full data sizes:

```sh
uv run python -m cs336_systems.distributed_benchmark \
  --backend gloo \
  --world-sizes 2 \
  --sizes-mib 0.001 \
  --warmup-steps 1 \
  --measurement-steps 2
```

## Naive DDP benchmark

Measure the full XL training step and the individual-gradient communication
section on one node with two NVIDIA GPUs:

```sh
uv run python -m cs336_systems.naive_ddp_benchmark \
  --backend nccl \
  --world-size 2 \
  --model-size xl \
  --global-batch-size 4 \
  --context-length 512 \
  --warmup-steps 5 \
  --measurement-steps 10
```

The script records raw per-rank timings and critical-path summary statistics in
`results/distributed/naive_ddp_benchmark.json`.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.
