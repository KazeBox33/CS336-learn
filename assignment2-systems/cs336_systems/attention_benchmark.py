"""Single-configuration benchmark for the materialized PyTorch attention path."""

from __future__ import annotations

import math
import time
from typing import Any

import torch


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
            scaled_dot_product_attention(q, k, v)
        output = scaled_dot_product_attention(q, k, v)
        output.sum().backward()
        q.grad = k.grad = v.grad = None

    _synchronize(torch_device)

    forward_timings_ms: list[float] = []
    for _ in range(measurement_steps):
        _synchronize(torch_device)
        start = time.perf_counter()
        with torch.no_grad():
            output = scaled_dot_product_attention(q, k, v)
        _synchronize(torch_device)
        forward_timings_ms.append((time.perf_counter() - start) * 1000)
        del output

    backward_timings_ms: list[float] = []
    backward_start_memory_bytes: list[int] = []
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)

    for _ in range(measurement_steps):
        q.grad = k.grad = v.grad = None
        output = scaled_dot_product_attention(q, k, v)
        loss = output.sum()
        _synchronize(torch_device)

        if torch_device.type == "cuda":
            backward_start_memory_bytes.append(torch.cuda.memory_allocated(torch_device))

        start = time.perf_counter()
        loss.backward()
        _synchronize(torch_device)
        backward_timings_ms.append((time.perf_counter() - start) * 1000)

    result: dict[str, Any] = {
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
