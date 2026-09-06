"""Benchmark materialized attention against the partially fused FlashAttention-2 path."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from cs336_systems.flash_attention import FlashAttentionTriton

try:
    triton = import_module("triton")
except ModuleNotFoundError:
    triton = None


DEFAULT_SEQUENCE_LENGTHS = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]
DEFAULT_D_MODELS = [16, 32, 64, 128]
DEFAULT_DTYPES = ["bfloat16", "float32"]
DEFAULT_IMPLEMENTATIONS = ["pytorch", "triton"]
PHASES = ["forward", "backward", "end_to_end"]
QUANTILES = [0.2, 0.5, 0.8]

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

AttentionFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton FlashAttention-2 against materialized PyTorch attention."
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQUENCE_LENGTHS,
    )
    parser.add_argument(
        "--d-models",
        type=int,
        nargs="+",
        default=DEFAULT_D_MODELS,
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=DEFAULT_DTYPES,
        default=DEFAULT_DTYPES,
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=DEFAULT_IMPLEMENTATIONS,
        default=DEFAULT_IMPLEMENTATIONS,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=25,
        help="Warmup duration passed to triton.testing.do_bench.",
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=100,
        help="Measurement duration passed to triton.testing.do_bench.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/attention/flash_attention_benchmark.json"),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed configuration instead of recording it.",
    )
    return parser.parse_args()


def materialized_causal_attention( # 普通 attention 
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute causal attention while explicitly materializing the score matrix."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape (batch, sequence, d_model)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")
    if causal_mask.shape != (q.shape[1], k.shape[1]):
        raise ValueError("causal_mask must have shape (n_queries, n_keys)")

    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~causal_mask, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def make_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(sequence_length, device=device)
    return positions[:, None] >= positions[None, :]


def make_inputs(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, sequence_length, d_model)
    q = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    return q, k, v


def do_bench( # do_bench 封装
    function: Callable[[], Any],
    *,
    warmup_ms: int,
    rep_ms: int,
    grad_to_none: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    if triton is None:
        raise RuntimeError("triton.testing.do_bench requires Triton")

    timings = triton.testing.do_bench(
        function,
        warmup=warmup_ms,
        rep=rep_ms,
        grad_to_none=grad_to_none,
        quantiles=QUANTILES,
    )
    p20_ms, median_ms, p80_ms = (float(value) for value in timings)
    return {
        "p20_ms": p20_ms,
        "median_ms": median_ms,
        "p80_ms": p80_ms,
    }


def benchmark_forward(
    attention: AttentionFunction,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, float]:
    return do_bench(
        lambda: attention(q, k, v),
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
    )


def benchmark_backward( # backward 
    attention: AttentionFunction,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, float]:
    output = attention(q, k, v)
    timings = do_bench(
        lambda: output.backward(grad_output, retain_graph=True),
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        grad_to_none=[q, k, v],
    )
    q.grad = k.grad = v.grad = None
    return timings


def benchmark_end_to_end( # end to end
    attention: AttentionFunction,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, float]:
    def forward_backward() -> None:
        output = attention(q, k, v)
        output.backward(grad_output)

    timings = do_bench(
        forward_backward,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        grad_to_none=[q, k, v],
    )
    q.grad = k.grad = v.grad = None
    return timings


def make_attention_function(
    implementation: str,
    sequence_length: int,
    device: torch.device,
) -> tuple[AttentionFunction, torch.Tensor | None]:
    if implementation == "pytorch":  # 如果用pytorch实现就 走这
        causal_mask = make_causal_mask(sequence_length, device)

        def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return materialized_causal_attention(q, k, v, causal_mask)

        return attention, causal_mask

    if implementation == "triton": # triron 就这

        def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return FlashAttentionTriton.apply(q, k, v, True)

        return attention, None

    raise ValueError(f"unknown implementation: {implementation}")


def clear_cuda_cache(tensors: tuple[torch.Tensor, ...] = ()) -> None:
    for tensor in tensors:
        if tensor.is_leaf:
            tensor.grad = None
    gc.collect()
    torch.cuda.empty_cache()


def benchmark_implementation(
    implementation: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    warmup_ms: int,
    rep_ms: int,
    fail_fast: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "implementation": implementation,
        "status": "ok",
    }

    try:
        attention, causal_mask = make_attention_function(
            implementation,
            q.shape[1],
            q.device,
        )
    except torch.OutOfMemoryError as error:
        if fail_fast:
            raise
        row["status"] = "oom"
        row["error"] = str(error)
        for phase in PHASES:
            row[f"{phase}_status"] = "not_run"
        clear_cuda_cache((q, k, v))
        return row

    phase_functions = {
        "forward": lambda: benchmark_forward(
            attention,
            q,
            k,
            v,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        ),
        "backward": lambda: benchmark_backward(
            attention,
            q,
            k,
            v,
            grad_output,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        ),
        "end_to_end": lambda: benchmark_end_to_end(
            attention,
            q,
            k,
            v,
            grad_output,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        ),
    }

    for phase, benchmark_phase in phase_functions.items():
        try:
            statistics = benchmark_phase()
        except torch.OutOfMemoryError as error:
            if fail_fast:
                raise
            row["status"] = "partial" if any(row.get(f"{name}_status") == "ok" for name in PHASES) else "oom"
            row[f"{phase}_status"] = "oom"
            row[f"{phase}_error"] = str(error)
        except Exception as error:
            if fail_fast:
                raise
            row["status"] = "partial" if any(row.get(f"{name}_status") == "ok" for name in PHASES) else "error"
            row[f"{phase}_status"] = "error"
            row[f"{phase}_error"] = f"{type(error).__name__}: {error}"
        else:
            row[f"{phase}_status"] = "ok"
            for statistic, value in statistics.items():
                row[f"{phase}_{statistic}"] = value
        finally:
            clear_cuda_cache((q, k, v))

    del causal_mask
    return row


def add_speedups(rows: list[dict[str, Any]]) -> None:
    rows_by_implementation = {row["implementation"]: row for row in rows}
    pytorch_row = rows_by_implementation.get("pytorch")
    triton_row = rows_by_implementation.get("triton")
    if pytorch_row is None or triton_row is None:
        return

    for phase in PHASES:
        pytorch_latency = pytorch_row.get(f"{phase}_median_ms")
        triton_latency = triton_row.get(f"{phase}_median_ms")
        if pytorch_latency is not None and triton_latency is not None:
            triton_row[f"{phase}_speedup_vs_pytorch"] = pytorch_latency / triton_latency


def write_results(
    output_path: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2),
        encoding="utf-8",
    )

    csv_path = output_path.with_suffix(".csv")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_args(args: argparse.Namespace) -> torch.device:
    if triton is None:
        raise RuntimeError("Triton is not installed; run this benchmark on a CUDA Linux machine")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this benchmark requires an NVIDIA GPU")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.warmup_ms < 0 or args.rep_ms <= 0:
        raise ValueError("warmup_ms must be non-negative and rep_ms must be positive")
    if any(length <= 0 for length in args.sequence_lengths):
        raise ValueError("sequence lengths must be positive")
    if any(d_model < 16 or d_model & (d_model - 1) for d_model in args.d_models):
        raise ValueError("d_model values must be powers of two and at least 16")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("flash benchmarking requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    return device


def main() -> None:
    args = parse_args()
    device = validate_args(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device_properties = torch.cuda.get_device_properties(device)
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": device_properties.total_memory,
        "torch_version": torch.__version__,
        "triton_version": getattr(triton, "__version__", "unknown"),
        "cuda_version": torch.version.cuda,
        "batch_size": args.batch_size,
        "is_causal": True,
        "sequence_lengths": args.sequence_lengths,
        "d_models": args.d_models,
        "dtypes": args.dtypes,
        "implementations": args.implementations,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "seed": args.seed,
    }

    rows: list[dict[str, Any]] = []
    total_configurations = len(args.sequence_lengths) * len(args.d_models) * len(args.dtypes)
    configuration_index = 0

    for sequence_length in args.sequence_lengths:
        for d_model in args.d_models:
            for dtype_name in args.dtypes:
                configuration_index += 1
                print(
                    f"[{configuration_index}/{total_configurations}] sequence_length={sequence_length} "
                    f"d_model={d_model} dtype={dtype_name}",
                    flush=True,
                )

                dtype = DTYPES[dtype_name]
                try:
                    q, k, v = make_inputs(
                        args.batch_size,
                        sequence_length,
                        d_model,
                        dtype,
                        device,
                    )
                    grad_output = torch.randn_like(q)
                except torch.OutOfMemoryError as error:
                    if args.fail_fast:
                        raise
                    for implementation in args.implementations:
                        rows.append(
                            {
                                "implementation": implementation,
                                "status": "oom",
                                "batch_size": args.batch_size,
                                "sequence_length": sequence_length,
                                "d_model": d_model,
                                "dtype": dtype_name,
                                "is_causal": True,
                                "error": str(error),
                            }
                        )
                    clear_cuda_cache()
                    write_results(args.output_path, metadata, rows)
                    continue

                configuration_rows = []
                for implementation in args.implementations:
                    print(f"  benchmarking {implementation}", flush=True)
                    row = benchmark_implementation(
                        implementation,
                        q,
                        k,
                        v,
                        grad_output,
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                        fail_fast=args.fail_fast,
                    )
                    row.update(
                        {
                            "batch_size": args.batch_size,
                            "sequence_length": sequence_length,
                            "d_model": d_model,
                            "dtype": dtype_name,
                            "is_causal": True,
                        }
                    )
                    configuration_rows.append(row)
                    print(f"    status={row['status']}", flush=True)

                add_speedups(configuration_rows)
                rows.extend(configuration_rows)
                write_results(args.output_path, metadata, rows)

                del q, k, v, grad_output
                clear_cuda_cache()

    print(f"Wrote {args.output_path}", flush=True)
    print(f"Wrote {args.output_path.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
