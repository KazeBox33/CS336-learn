"""Single-configuration benchmark for the materialized PyTorch attention path."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

AttentionFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one ordinary PyTorch attention configuration."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=100)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute ordinary scaled dot-product attention.

    The inputs have shape ``(batch_size, sequence_length, d_model)``. This
    intentionally materializes the ``(batch_size, sequence_length,
    sequence_length)`` score and probability tensors.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must all have shape (batch, sequence, d_model)")

    d_model = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def _synchronize(device: torch.device) -> None:
    """Wait until work submitted to an asynchronous accelerator is complete."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _make_inputs(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, sequence_length, d_model)
    return tuple(
        torch.randn(shape, device=device, requires_grad=True)
        for _ in range(3)
    )  # type: ignore[return-value]


def benchmark_single_configuration(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: str = "cuda",
    warmup_steps: int = 5,
    measurement_steps: int = 100,
    attention_fn: AttentionFunction = scaled_dot_product_attention,
    implementation: str = "eager",
) -> dict[str, Any]:
    """Benchmark one ``(batch_size, sequence_length, d_model)`` setting.

    Forward timings use ``no_grad`` because this measures the forward operator
    itself. Backward timings build a fresh graph before each timed backward;
    otherwise the first backward would free the graph needed by later steps.
    """
    if warmup_steps < 0 or measurement_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and measurement_steps must be positive")

    torch_device = torch.device(device)
    q, k, v = _make_inputs(batch_size, sequence_length, d_model, torch_device)

    for _ in range(warmup_steps):
        with torch.no_grad():
            attention_fn(q, k, v)
        output = attention_fn(q, k, v)
        output.sum().backward()
        q.grad = k.grad = v.grad = None

    _synchronize(torch_device)

    forward_timings_ms: list[float] = []
    for _ in range(measurement_steps):
        _synchronize(torch_device)
        start = time.perf_counter()
        with torch.no_grad():
            output = attention_fn(q, k, v)
        _synchronize(torch_device)
        forward_timings_ms.append((time.perf_counter() - start) * 1000)
        del output

    backward_timings_ms: list[float] = []
    backward_start_memory_bytes: list[int] = [] # 保存每次backward开始前，CUDA当前已经分配的显存
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device) #重置峰值显存统计

    for _ in range(measurement_steps):
        q.grad = k.grad = v.grad = None
        output = attention_fn(q, k, v) # 每次backward后都需要一张新的计算图
        loss = output.sum()
        _synchronize(torch_device)

        if torch_device.type == "cuda":
            backward_start_memory_bytes.append(torch.cuda.memory_allocated(torch_device))

        start = time.perf_counter()
        loss.backward()
        _synchronize(torch_device)
        backward_timings_ms.append((time.perf_counter() - start) * 1000)

    result: dict[str, Any] = {
        "status": "ok",
        "implementation": implementation,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "d_model": d_model,
        "device": str(torch_device),
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "forward_timings_ms": forward_timings_ms,
        "backward_timings_ms": backward_timings_ms,
        "forward_mean_ms": sum(forward_timings_ms) / len(forward_timings_ms),
        "backward_mean_ms": sum(backward_timings_ms) / len(backward_timings_ms),
        "backward_start_memory_bytes": backward_start_memory_bytes,
    }

    if torch_device.type == "cuda":
        result["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(torch_device)

    return result


def main() -> None:
    args = parse_args()
    implementation = "compiled" if args.compile else "eager"
    attention_fn = (
        torch.compile(scaled_dot_product_attention)  
        if args.compile
        else scaled_dot_product_attention
    )

    try:
        result = benchmark_single_configuration(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            d_model=args.d_model,
            device=args.device,
            warmup_steps=args.warmup_steps,
            measurement_steps=args.measurement_steps,
            attention_fn=attention_fn,
            implementation=implementation,
        )
    except torch.OutOfMemoryError as error: # 当前只捕获OOM
        result = { 
            "status": "oom",
            "implementation": implementation,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "d_model": args.d_model,
            "device": args.device,
            "warmup_steps": args.warmup_steps,
            "measurement_steps": args.measurement_steps,
            "error": str(error),
        }

    result_json = json.dumps(result, indent=2)
    print(result_json)

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(result_json, encoding="utf-8")


if __name__ == "__main__":
    main()
