"""Profile FSDP memory accounting and forward all-gather overlap."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import time
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM, Embedding, Linear
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from cs336_systems.benchmark import MODEL_CONFIGS
from cs336_systems.fsdp import FullyShardedDataParallel


COMPUTE_DTYPES: dict[str, torch.dtype | None] = {
    "fp32": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile FSDP all-gather overlap.")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--compute-dtype", choices=COMPUTE_DTYPES, default="bf16")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/distributed/fsdp_accounting/profile.json"),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.world_size < 2:
        raise ValueError("world_size must be at least 2")
    if args.global_batch_size <= 0 or args.global_batch_size % args.world_size != 0:
        raise ValueError("global_batch_size must be positive and divisible by world_size")
    if args.context_length <= 0 or args.vocab_size <= 0:
        raise ValueError("context_length and vocab_size must be positive")
    if args.warmup_steps < 1 or args.measurement_steps <= 0:
        raise ValueError("warmup_steps must be positive and measurement_steps must be positive")

    if args.backend == "nccl":
        if not dist.is_nccl_available() or not torch.cuda.is_available():
            raise RuntimeError("NCCL profiling requires a CUDA-enabled PyTorch installation")
        if torch.cuda.device_count() < args.world_size:
            raise RuntimeError(f"requested {args.world_size} GPUs, but only {torch.cuda.device_count()} are available")


def theoretical_fsdp_memory_bytes(
    sharded_parameter_count: int,
    replicated_parameter_count: int,
    *,
    world_size: int,
) -> dict[str, float]:
    """Estimate persistent FP32 parameter, gradient, and AdamW state memory."""
    if sharded_parameter_count < 0 or replicated_parameter_count < 0:
        raise ValueError("parameter counts must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    bytes_per_parameter = 16.0
    baseline = bytes_per_parameter * (sharded_parameter_count + replicated_parameter_count)
    fsdp = bytes_per_parameter * (sharded_parameter_count / world_size + replicated_parameter_count)
    saved = baseline - fsdp
    return {
        "baseline_bytes_per_rank": baseline,
        "fsdp_bytes_per_rank": fsdp,
        "saved_bytes_per_rank": saved,
        "saved_fraction": saved / baseline if baseline else 0.0,
    }


def summarize_rank_timings(
    rank_forward_timings_ms: list[list[float]],
    rank_step_timings_ms: list[list[float]],
) -> dict[str, float]:
    """Summarize the slowest rank in every measured iteration."""
    if not rank_forward_timings_ms or len(rank_forward_timings_ms) != len(rank_step_timings_ms):
        raise ValueError("forward and step timings must contain the same non-zero ranks")
    forward_counts = {len(values) for values in rank_forward_timings_ms}
    step_counts = {len(values) for values in rank_step_timings_ms}
    if forward_counts != step_counts or len(forward_counts) != 1 or 0 in forward_counts:
        raise ValueError("every rank must provide the same non-zero number of timings")

    critical_forward = [max(per_rank) for per_rank in zip(*rank_forward_timings_ms, strict=True)]
    critical_step = [max(per_rank) for per_rank in zip(*rank_step_timings_ms, strict=True)]
    return {
        "forward_mean_ms": statistics.fmean(critical_forward),
        "forward_std_ms": (statistics.stdev(critical_forward) if len(critical_forward) > 1 else 0.0),
        "step_mean_ms": statistics.fmean(critical_step),
        "step_std_ms": (statistics.stdev(critical_step) if len(critical_step) > 1 else 0.0),
    }


def _build_model(
    model_size: str,
    *,
    vocab_size: int,
    context_length: int,
    device: torch.device,
) -> BasicsTransformerLM:
    config = MODEL_CONFIGS[model_size]
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
    ).to(device)


def _parameter_partition(model: torch.nn.Module) -> tuple[int, int]:
    sharded_ids = {id(child.weight) for child in model.modules() if isinstance(child, (Linear, Embedding))}
    sharded = 0
    replicated = 0
    for parameter in model.parameters():
        if id(parameter) in sharded_ids:
            sharded += parameter.numel()
        else:
            replicated += parameter.numel()
    return sharded, replicated


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nvtx_range(
    message: str,
    device: torch.device,
) -> AbstractContextManager[None]:
    if device.type == "cuda":
        return torch.cuda.nvtx.range(message)
    return nullcontext()


def _run_training_step(
    model: FullyShardedDataParallel,
    optimizer: AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    device = inputs.device
    _synchronize(device)
    step_start = time.perf_counter()

    with _nvtx_range("fsdp_training_step", device):
        optimizer.zero_grad(set_to_none=True)

        _synchronize(device)
        forward_start = time.perf_counter()
        with _nvtx_range("fsdp_forward", device):
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
        _synchronize(device)
        forward_ms = (time.perf_counter() - forward_start) * 1000.0

        with _nvtx_range("fsdp_backward", device):
            loss.backward()
        with _nvtx_range("fsdp_finish_gradient_synchronization", device):
            model.finish_gradient_synchronization()
        with _nvtx_range("fsdp_optimizer_step", device):
            optimizer.step()

    _synchronize(device)
    step_ms = (time.perf_counter() - step_start) * 1000.0
    return forward_ms, step_ms


def _gather_timings(
    forward_timings_ms: list[float],
    step_timings_ms: list[float],
    *,
    device: torch.device,
    world_size: int,
) -> tuple[list[list[float]], list[list[float]]]:
    local = torch.tensor(
        [forward_timings_ms, step_timings_ms],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return (
        [rank_timings[0].cpu().tolist() for rank_timings in gathered],
        [rank_timings[1].cpu().tolist() for rank_timings in gathered],
    )


def _worker(
    rank: int,
    world_size: int,
    backend: str,
    model_size: str,
    global_batch_size: int,
    context_length: int,
    vocab_size: int,
    compute_dtype_name: str,
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
        torch.manual_seed(seed)
        base_model = _build_model(
            model_size,
            vocab_size=vocab_size,
            context_length=context_length,
            device=device,
        )
        sharded_count, replicated_count = _parameter_partition(base_model)
        model = FullyShardedDataParallel(
            base_model,
            compute_dtype=COMPUTE_DTYPES[compute_dtype_name],
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
        forward_timings_ms = []
        step_timings_ms = []
        with _nvtx_range("fsdp_measurement", device):
            for _ in range(measurement_steps):
                forward_ms, step_ms = _run_training_step(
                    model,
                    optimizer,
                    inputs,
                    targets,
                )
                forward_timings_ms.append(forward_ms)
                step_timings_ms.append(step_ms)

        rank_forward, rank_step = _gather_timings(
            forward_timings_ms,
            step_timings_ms,
            device=device,
            world_size=world_size,
        )
        if rank == 0:
            accounting = theoretical_fsdp_memory_bytes(
                sharded_count,
                replicated_count,
                world_size=world_size,
            )
            result: dict[str, Any] = {
                "benchmark": "fsdp_accounting",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "hostname": platform.node(),
                "platform": platform.platform(),
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda if backend == "nccl" else None,
                "nccl": torch.cuda.nccl.version() if backend == "nccl" else None,
                "gpu_names": ([torch.cuda.get_device_name(index) for index in range(world_size)] if backend == "nccl" else []),
                "backend": backend,
                "model_size": model_size,
                "model_config": MODEL_CONFIGS[model_size],
                "world_size": world_size,
                "global_batch_size": global_batch_size,
                "local_batch_size": local_batch_size,
                "context_length": context_length,
                "vocab_size": vocab_size,
                "compute_dtype": compute_dtype_name,
                "warmup_steps": warmup_steps,
                "measurement_steps": measurement_steps,
                "sharded_parameter_count": sharded_count,
                "replicated_parameter_count": replicated_count,
                "theoretical_persistent_fp32_memory": accounting,
                "rank_forward_timings_ms": rank_forward,
                "rank_step_timings_ms": rank_step,
                **summarize_rank_timings(rank_forward, rank_step),
            }
            Path(output_path).write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )
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
            args.compute_dtype,
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
    memory = result["theoretical_persistent_fp32_memory"]
    print(
        "persistent FP32 model-state memory: "
        f"{memory['baseline_bytes_per_rank'] / 1024**3:.2f} -> "
        f"{memory['fsdp_bytes_per_rank'] / 1024**3:.2f} GiB/rank "
        f"({memory['saved_fraction'] * 100:.1f}% saved)"
    )
    print(f"forward: {result['forward_mean_ms']:.3f} +/- {result['forward_std_ms']:.3f} ms")
    print(f"training step: {result['step_mean_ms']:.3f} +/- {result['step_std_ms']:.3f} ms")
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
