"""Benchmark training with the individual-gradient NaiveDDP implementation."""

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
from cs336_basics.optimizer import AdamW

from cs336_systems.benchmark import MODEL_CONFIGS
from cs336_systems.ddp import NaiveDDP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naïve data-parallel training.")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/distributed/naive_ddp_benchmark.json"),
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

    if args.backend == "nccl":
        if not dist.is_nccl_available() or not torch.cuda.is_available():
            raise RuntimeError("NCCL benchmark requires a CUDA-enabled PyTorch installation")
        if torch.cuda.device_count() < args.world_size:
            raise RuntimeError(
                f"requested {args.world_size} GPUs, but only {torch.cuda.device_count()} are available"
            )


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


def _run_training_step(
    model: NaiveDDP,
    optimizer: AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    device = inputs.device
    _synchronize(device)
    step_start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()

    _synchronize(device)
    communication_start = time.perf_counter()
    model.finish_gradient_synchronization()
    _synchronize(device)
    communication_seconds = time.perf_counter() - communication_start

    optimizer.step()
    _synchronize(device)
    step_seconds = time.perf_counter() - step_start
    return step_seconds * 1000.0, communication_seconds * 1000.0


def _gather_timings(
    step_timings_ms: list[float],
    communication_timings_ms: list[float],
    *,
    device: torch.device,
    world_size: int,
) -> tuple[list[list[float]], list[list[float]]]:
    local = torch.tensor(
        [step_timings_ms, communication_timings_ms],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)

    rank_step_timings = [rank_timings[0].cpu().tolist() for rank_timings in gathered]
    rank_communication_timings = [rank_timings[1].cpu().tolist() for rank_timings in gathered]
    return rank_step_timings, rank_communication_timings


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty timing list")
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_training_timings(
    rank_step_timings_ms: list[list[float]],
    rank_communication_timings_ms: list[list[float]],
) -> dict[str, float]:
    if len(rank_step_timings_ms) != len(rank_communication_timings_ms):
        raise ValueError("step and communication timings must contain the same ranks")
    if not rank_step_timings_ms:
        raise ValueError("timings must contain at least one rank")

    step_counts = {len(timings) for timings in rank_step_timings_ms}
    communication_counts = {len(timings) for timings in rank_communication_timings_ms}
    if step_counts != communication_counts or len(step_counts) != 1 or 0 in step_counts:
        raise ValueError("every rank must provide the same non-zero number of step and communication timings")

    critical_step_ms = [max(per_rank) for per_rank in zip(*rank_step_timings_ms, strict=True)]
    critical_communication_ms = [
        max(per_rank) for per_rank in zip(*rank_communication_timings_ms, strict=True)
    ]
    step_mean_ms, step_std_ms = _mean_and_std(critical_step_ms)
    communication_mean_ms, communication_std_ms = _mean_and_std(critical_communication_ms)

    return {
        "step_mean_ms": step_mean_ms,
        "step_std_ms": step_std_ms,
        "communication_mean_ms": communication_mean_ms,
        "communication_std_ms": communication_std_ms,
        "communication_fraction_pct": communication_mean_ms / step_mean_ms * 100.0,
    }


def _worker(
    rank: int,
    world_size: int,
    backend: str,
    model_size: str,
    global_batch_size: int,
    context_length: int,
    vocab_size: int,
    warmup_steps: int,
    measurement_steps: int,
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
        optimizer = AdamW(model.parameters())

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

        for _ in range(warmup_steps):
            _run_training_step(model, optimizer, inputs, targets)

        _synchronize(device)
        dist.barrier()
        step_timings_ms: list[float] = []
        communication_timings_ms: list[float] = []
        for _ in range(measurement_steps):
            step_ms, communication_ms = _run_training_step(model, optimizer, inputs, targets)
            step_timings_ms.append(step_ms)
            communication_timings_ms.append(communication_ms)

        rank_step_timings, rank_communication_timings = _gather_timings(
            step_timings_ms,
            communication_timings_ms,
            device=device,
            world_size=world_size,
        )

        if rank == 0:
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            result: dict[str, Any] = {
                "benchmark": "naive_ddp_training",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "hostname": platform.node(),
                "platform": platform.platform(),
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda if backend == "nccl" else None,
                "nccl": torch.cuda.nccl.version() if backend == "nccl" else None,
                "gpu_names": (
                    [torch.cuda.get_device_name(index) for index in range(world_size)]
                    if backend == "nccl"
                    else []
                ),
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
                "rank_step_timings_ms": rank_step_timings,
                "rank_communication_timings_ms": rank_communication_timings,
                **summarize_training_timings(
                    rank_step_timings,
                    rank_communication_timings,
                ),
            }
            Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    args = parse_args()
    _validate_args(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.unlink(missing_ok=True)

    mp.spawn(
        _worker,
        args=(
            args.world_size,
            args.backend,
            args.model_size,
            args.global_batch_size,
            args.context_length,
            args.vocab_size,
            args.warmup_steps,
            args.measurement_steps,
            args.seed,
            args.master_addr,
            _find_free_port(),
            str(args.output_path),
        ),
        nprocs=args.world_size,
        join=True,
    )

    result = json.loads(args.output_path.read_text(encoding="utf-8"))
    print(f"model: {result['model_size']} ({result['parameter_count']:,} parameters)")
    print(f"world size: {result['world_size']}")
    print(f"global/local batch size: {result['global_batch_size']}/{result['local_batch_size']}")
    print(f"step: {result['step_mean_ms']:.3f} +/- {result['step_std_ms']:.3f} ms")
    print(
        "gradient communication: "
        f"{result['communication_mean_ms']:.3f} +/- {result['communication_std_ms']:.3f} ms "
        f"({result['communication_fraction_pct']:.2f}% of step)"
    )
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
