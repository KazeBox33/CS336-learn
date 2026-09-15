"""Profile memory and speed with and without optimizer state sharding."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

from cs336_systems.benchmark import MODEL_CONFIGS
from cs336_systems.ddp import NaiveDDP
from cs336_systems.sharded_optimizer import ShardedOptimizer


OPTIMIZER_IMPLEMENTATIONS = ("baseline", "sharded")
MEMORY_CHECKPOINTS = (
    "after_model_initialization",
    "before_optimizer_step",
    "after_optimizer_step",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark optimizer state sharding.")
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=OPTIMIZER_IMPLEMENTATIONS,
        default=list(OPTIMIZER_IMPLEMENTATIONS),
    )
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/distributed/optimizer_state_sharding"),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.world_size < 2:
        raise ValueError("world_size must be at least 2")
    if args.global_batch_size <= 0 or args.global_batch_size % args.world_size != 0:
        raise ValueError("global_batch_size must be positive and divisible by world_size")
    if args.context_length <= 0 or args.vocab_size <= 0:
        raise ValueError("context_length and vocab_size must be positive")
    if args.warmup_steps < 0 or args.measurement_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and measurement_steps must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay must be non-negative")

    if args.backend == "nccl":
        if not dist.is_nccl_available() or not torch.cuda.is_available():
            raise RuntimeError("NCCL benchmark requires a CUDA-enabled PyTorch installation")
        if torch.cuda.device_count() < args.world_size:
            raise RuntimeError(f"requested {args.world_size} GPUs, but only {torch.cuda.device_count()} are available")


def _build_model(
    model_size: str,
    *,
    vocab_size: int,
    context_length: int,
    device: torch.device,
) -> BasicsTransformerLM:
    config = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
    )
    return model.to(device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        _synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _capture_memory(device: torch.device) -> dict[str, int | None]:
    """Capture allocator counters after all preceding CUDA work has completed."""
    if device.type != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "phase_peak_allocated_bytes": None,
            "phase_peak_reserved_bytes": None,
        }

    _synchronize(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "phase_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "phase_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    state_owner = optimizer
    if isinstance(optimizer, ShardedOptimizer):
        if optimizer._local_optimizer is None:
            raise RuntimeError("local optimizer has not been initialized")
        state_owner = optimizer._local_optimizer

    return sum(value.numel() * value.element_size() for state in state_owner.state.values() for value in state.values() if isinstance(value, torch.Tensor))


def _component_bytes(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    parameters = list(model.parameters())
    return {
        "parameters": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
        "gradients": sum(parameter.grad.numel() * parameter.grad.element_size() for parameter in parameters if parameter.grad is not None),
        "local_optimizer_state": _optimizer_state_bytes(optimizer),
    }


def theoretical_fp32_memory_bytes(
    parameter_count: int,
    *,
    world_size: int,
    sharded: bool,
) -> dict[str, float]:
    """Estimate persistent FP32 parameter, gradient, and Adam state memory per rank."""
    if parameter_count < 0 or world_size <= 0:
        raise ValueError("parameter_count must be non-negative and world_size must be positive")

    parameter_bytes = parameter_count * 4
    gradient_bytes = parameter_count * 4
    optimizer_state_bytes = parameter_count * 8 / (world_size if sharded else 1)
    return {
        "parameters": float(parameter_bytes),
        "gradients": float(gradient_bytes),
        "optimizer_state": float(optimizer_state_bytes),
        "total": float(parameter_bytes + gradient_bytes + optimizer_state_bytes),
    }


def summarize_rank_timings(rank_timings_ms: list[list[float]]) -> dict[str, float]:
    """Summarize the slowest rank for each measured training iteration."""
    if not rank_timings_ms:
        raise ValueError("timings must contain at least one rank")
    step_counts = {len(timings) for timings in rank_timings_ms}
    if len(step_counts) != 1 or 0 in step_counts:
        raise ValueError("every rank must provide the same non-zero number of timings")

    critical_path_ms = [max(per_rank) for per_rank in zip(*rank_timings_ms, strict=True)]
    return {
        "step_mean_ms": statistics.fmean(critical_path_ms),
        "step_std_ms": statistics.stdev(critical_path_ms) if len(critical_path_ms) > 1 else 0.0,
    }


def _run_training_step(
    model: NaiveDDP,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    device = inputs.device
    _synchronize(device)
    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()
    model.finish_gradient_synchronization()
    optimizer.step()

    _synchronize(device)
    return (time.perf_counter() - start) * 1000.0


def _profile_first_step(
    model: NaiveDDP,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[dict[str, dict[str, int | None]], dict[str, int]]:
    device = inputs.device
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()
    model.finish_gradient_synchronization()
    del logits, loss

    memory = {"before_optimizer_step": _capture_memory(device)}
    _reset_peak_memory(device)
    optimizer.step()
    memory["after_optimizer_step"] = _capture_memory(device)
    return memory, _component_bytes(model, optimizer)


def _maximum_memory_across_ranks(
    rank_records: list[dict[str, Any]],
) -> dict[str, dict[str, int | None]]:
    summary: dict[str, dict[str, int | None]] = {}
    for checkpoint in MEMORY_CHECKPOINTS:
        checkpoint_summary: dict[str, int | None] = {}
        for metric in (
            "allocated_bytes",
            "reserved_bytes",
            "phase_peak_allocated_bytes",
            "phase_peak_reserved_bytes",
        ):
            values = [record["memory"][checkpoint][metric] for record in rank_records if record["memory"][checkpoint][metric] is not None]
            checkpoint_summary[metric] = max(values) if values else None
        summary[checkpoint] = checkpoint_summary
    return summary


def _maximum_components_across_ranks(rank_records: list[dict[str, Any]]) -> dict[str, int]:
    return {component: max(record["component_bytes_after_step"][component] for record in rank_records) for component in ("parameters", "gradients", "local_optimizer_state")}


def _worker(
    rank: int,
    world_size: int,
    implementation: str,
    backend: str,
    model_size: str,
    global_batch_size: int,
    context_length: int,
    vocab_size: int,
    warmup_steps: int,
    measurement_steps: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    master_addr: str,
    master_port: int,
    output_path: str,
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=10),
    )
    try:
        _reset_peak_memory(device)
        torch.manual_seed(seed + rank)
        model = NaiveDDP(
            _build_model(
                model_size,
                vocab_size=vocab_size,
                context_length=context_length,
                device=device,
            )
        )
        model.train()
        model_initialization_memory = _capture_memory(device)

        optimizer_kwargs = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
        if implementation == "sharded":
            optimizer: torch.optim.Optimizer = ShardedOptimizer(
                model.parameters(),
                torch.optim.AdamW,
                **optimizer_kwargs,
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

        local_batch_size = global_batch_size // world_size
        torch.manual_seed(seed + 10_000 + rank)
        inputs = torch.randint(
            0,
            vocab_size,
            (local_batch_size, context_length),
            device=device,
        )
        targets = torch.randint(
            0,
            vocab_size,
            (local_batch_size, context_length),
            device=device,
        )

        _reset_peak_memory(device)
        step_memory, component_bytes = _profile_first_step(model, optimizer, inputs, targets)
        memory = {
            "after_model_initialization": model_initialization_memory,
            **step_memory,
        }

        for _ in range(warmup_steps):
            _run_training_step(model, optimizer, inputs, targets)

        _synchronize(device)
        dist.barrier()
        step_timings_ms = [_run_training_step(model, optimizer, inputs, targets) for _ in range(measurement_steps)]

        local_record = {
            "rank": rank,
            "memory": memory,
            "component_bytes_after_step": component_bytes,
            "step_timings_ms": step_timings_ms,
        }
        rank_records: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(rank_records, local_record)

        if rank == 0:
            complete_records = [record for record in rank_records if record is not None]
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            result: dict[str, Any] = {
                "implementation": implementation,
                "backend": backend,
                "model_size": model_size,
                "model_config": MODEL_CONFIGS[model_size],
                "parameter_count": parameter_count,
                "world_size": world_size,
                "global_batch_size": global_batch_size,
                "local_batch_size": local_batch_size,
                "context_length": context_length,
                "vocab_size": vocab_size,
                "warmup_steps": warmup_steps,
                "measurement_steps": measurement_steps,
                "memory_counters_available": device.type == "cuda",
                "rank_records": complete_records,
                "maximum_memory_across_ranks": _maximum_memory_across_ranks(complete_records),
                "maximum_component_bytes_across_ranks": _maximum_components_across_ranks(complete_records),
                "theoretical_fp32_bytes_per_rank": theoretical_fp32_memory_bytes(
                    parameter_count,
                    world_size=world_size,
                    sharded=implementation == "sharded",
                ),
                **summarize_rank_timings([record["step_timings_ms"] for record in complete_records]),
            }
            Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _mib(value: int | float | None) -> str:
    return "n/a" if value is None else f"{value / 1024**2:.1f}"


def render_report(document: dict[str, Any]) -> str:
    results = document["results"]
    lines = [
        "# Optimizer State Sharding Accounting",
        "",
        "## Peak memory",
        "",
        "| Implementation | Model init MiB | Before step peak MiB | After step peak MiB | Adam state MiB/rank (ideal) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for implementation, result in results.items():
        maximum = result["maximum_memory_across_ranks"]
        theoretical = result["theoretical_fp32_bytes_per_rank"]
        lines.append(
            f"| {implementation} | "
            f"{_mib(maximum['after_model_initialization']['phase_peak_allocated_bytes'])} | "
            f"{_mib(maximum['before_optimizer_step']['phase_peak_allocated_bytes'])} | "
            f"{_mib(maximum['after_optimizer_step']['phase_peak_allocated_bytes'])} | "
            f"{_mib(theoretical['optimizer_state'])} |"
        )

    lines.extend(
        [
            "",
            "## Persistent component breakdown after the first step",
            "",
            "| Implementation | Parameters MiB | Gradients MiB | Local optimizer state MiB |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for implementation, result in results.items():
        components = result["maximum_component_bytes_across_ranks"]
        lines.append(f"| {implementation} | {_mib(components['parameters'])} | {_mib(components['gradients'])} | {_mib(components['local_optimizer_state'])} |")

    lines.extend(
        [
            "",
            "## Training speed",
            "",
            "| Implementation | Mean iteration ms | Std ms |",
            "| --- | ---: | ---: |",
        ]
    )
    for implementation, result in results.items():
        lines.append(f"| {implementation} | {result['step_mean_ms']:.3f} | {result['step_std_ms']:.3f} |")

    if {"baseline", "sharded"}.issubset(results):
        baseline = results["baseline"]
        sharded = results["sharded"]
        baseline_peak = baseline["maximum_memory_across_ranks"]["after_optimizer_step"]["phase_peak_allocated_bytes"]
        sharded_peak = sharded["maximum_memory_across_ranks"]["after_optimizer_step"]["phase_peak_allocated_bytes"]
        if baseline_peak is not None and sharded_peak is not None:
            memory_reduction = (baseline_peak - sharded_peak) / baseline_peak * 100.0
            lines.extend(
                [
                    "",
                    f"After the first optimizer step, peak allocated memory was "
                    f"{_mib(baseline_peak)} MiB without sharding and "
                    f"{_mib(sharded_peak)} MiB with sharding, a {memory_reduction:.1f}% reduction.",
                    "Parameters and gradients remain replicated; the reduction comes from partitioning Adam's local optimizer state across ranks.",
                ]
            )

        timing_change = (sharded["step_mean_ms"] - baseline["step_mean_ms"]) / baseline["step_mean_ms"] * 100.0
        lines.extend(
            [
                "",
                f"Mean iteration time changed from {baseline['step_mean_ms']:.3f} ms to "
                f"{sharded['step_mean_ms']:.3f} ms ({timing_change:+.1f}%). The sharded "
                "optimizer performs less local optimizer work per rank but adds parameter broadcasts.",
            ]
        )

    lines.extend(
        [
            "",
            "## ZeRO Stage 1 comparison",
            "",
            "This educational implementation assigns whole parameter tensors to owners and broadcasts each updated tensor separately. ZeRO Stage 1 uses more balanced flattened partitions and bucketed collectives, so it achieves similar optimizer-state sharding with fewer communication launches and better load balance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for implementation in args.implementations:
        output_path = args.output_dir / f"{implementation}.json"
        mp.spawn(
            _worker,
            args=(
                args.world_size,
                implementation,
                args.backend,
                args.model_size,
                args.global_batch_size,
                args.context_length,
                args.vocab_size,
                args.warmup_steps,
                args.measurement_steps,
                args.learning_rate,
                args.weight_decay,
                args.seed,
                args.master_addr,
                _find_free_port(),
                str(output_path),
            ),
            nprocs=args.world_size,
            join=True,
        )
        results[implementation] = json.loads(output_path.read_text(encoding="utf-8"))

    comparison = {
        "benchmark": "optimizer_state_sharding_accounting",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda if args.backend == "nccl" else None,
        "gpu_names": ([torch.cuda.get_device_name(index) for index in range(args.world_size)] if args.backend == "nccl" else []),
        "results": results,
    }
    comparison_path = args.output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    report = render_report(json.loads(comparison_path.read_text(encoding="utf-8")))
    report_path = args.output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    for implementation, result in results.items():
        print(f"{implementation}: {result['step_mean_ms']:.3f} +/- {result['step_std_ms']:.3f} ms")
    print(f"wrote {comparison_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
