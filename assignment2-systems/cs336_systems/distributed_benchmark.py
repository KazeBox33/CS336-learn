"""Benchmark single-node All-Reduce communication with PyTorch Distributed."""

from __future__ import annotations

import argparse
import csv
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


DEFAULT_SIZES_MIB = [1.0, 10.0, 100.0, 1024.0]  # 实验默认配置
DEFAULT_WORLD_SIZES = [2, 4, 6]
FLOAT32_BYTES = torch.tensor([], dtype=torch.float32).element_size()  # element_size() 返回数据占多少字节


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-node All-Reduce across multiple process counts and tensor sizes."
    )
    parser.add_argument("--sizes-mib", type=float, nargs="+", default=DEFAULT_SIZES_MIB)
    parser.add_argument("--world-sizes", type=int, nargs="+", default=DEFAULT_WORLD_SIZES)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=20)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/distributed/single_node"),
    )
    return parser.parse_args()


def mib_to_numel(size_mib: float, element_size: int = FLOAT32_BYTES) -> int:  # 把 MiB 换成元素数量
    """Convert a binary MiB size to a whole number of tensor elements."""
    if size_mib <= 0:
        raise ValueError("size_mib must be positive")
    if element_size <= 0:
        raise ValueError("element_size must be positive")

    num_bytes = round(size_mib * 1024**2)
    num_elements = num_bytes // element_size
    if num_elements == 0:
        raise ValueError("size_mib is too small to hold one element")
    return num_elements


def _percentile(values: list[float], percentile: float) -> float:
    """Compute a linearly interpolated percentile without NumPy."""
    if not values:
        raise ValueError("cannot summarize an empty timing list")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0 and 1")

    ordered = sorted(values)
    position = percentile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_rank_timings(
    rank_timings_ms: list[list[float]],
    *,
    tensor_bytes: int,
    world_size: int,
) -> dict[str, float]:
    """Summarize each repetition by the slowest participating rank."""
    if len(rank_timings_ms) != world_size:
        raise ValueError("rank_timings_ms must contain one row per rank")
    if world_size <= 1:
        raise ValueError("All-Reduce benchmark requires at least two ranks")
    if tensor_bytes <= 0:
        raise ValueError("tensor_bytes must be positive")

    measurement_counts = {len(timings) for timings in rank_timings_ms}
    if len(measurement_counts) != 1 or not measurement_counts or 0 in measurement_counts:
        raise ValueError("every rank must provide the same non-zero number of timings")

    # 原本每行表示一个 rank 的多轮结果，转置后每行表示同一轮中所有 rank 的结果，再取其中最大值。
    critical_path_ms = [max(per_rank) for per_rank in zip(*rank_timings_ms, strict=True)]
    median_ms = statistics.median(critical_path_ms)
    algorithmic_bandwidth_gbps = tensor_bytes / (median_ms / 1000.0) / 1e9
    ring_traffic_factor = 2.0 * (world_size - 1) / world_size

    return {
        "mean_ms": statistics.fmean(critical_path_ms),
        "median_ms": median_ms,
        "p20_ms": _percentile(critical_path_ms, 0.2),
        "p80_ms": _percentile(critical_path_ms, 0.8),
        "min_ms": min(critical_path_ms),
        "max_ms": max(critical_path_ms),
        "algorithmic_bandwidth_gbps": algorithmic_bandwidth_gbps,
        "bus_bandwidth_gbps": algorithmic_bandwidth_gbps * ring_traffic_factor,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _allocate_tensor(num_elements: int, device: torch.device, rank: int) -> torch.Tensor | None:
    try:
        return torch.full(
            (num_elements,),
            fill_value=float(rank + 1),
            dtype=torch.float32,
            device=device,
        )
    except torch.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return None


def _all_ranks_allocated(tensor: torch.Tensor | None, device: torch.device) -> bool:
    allocation_status = torch.tensor(
        [tensor is not None],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(allocation_status, op=dist.ReduceOp.MIN)  # 取最小值
    return bool(allocation_status.item())


def _verify_all_reduce(tensor: torch.Tensor, world_size: int) -> None:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # 相加起来
    _synchronize(tensor.device)

    expected = world_size * (world_size + 1) / 2
    sample = tensor[: min(16, tensor.numel())]  # 只检查前 16 个元素
    if not torch.allclose(sample, torch.full_like(sample, expected)):  # 比较结果
        raise RuntimeError("All-Reduce correctness check failed")


def _measure_all_reduce(
    tensor: torch.Tensor,
    *,
    warmup_steps: int,
    measurement_steps: int,
) -> list[float]:
    for _ in range(warmup_steps):
        tensor.fill_(1.0)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    _synchronize(tensor.device)

    dist.barrier()
    local_timings_ms: list[float] = []
    for _ in range(measurement_steps):
        tensor.fill_(1.0)  # 原地修改
        _synchronize(tensor.device)

        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        _synchronize(tensor.device)
        local_timings_ms.append((time.perf_counter() - start) * 1000.0)

    return local_timings_ms


def _gather_rank_timings(
    local_timings_ms: list[float],
    device: torch.device,
    world_size: int,
) -> list[list[float]]:
    local = torch.tensor(local_timings_ms, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)  # 收集所有 rank 的结果
    return [timings.cpu().tolist() for timings in gathered]


def _worker(
    rank: int,
    world_size: int,
    backend: str,
    sizes_mib: list[float],
    warmup_steps: int,
    measurement_steps: int,
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

    records: list[dict[str, Any]] = []
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=10),
    )

    try:
        for size_mib in sizes_mib:
            num_elements = mib_to_numel(size_mib)
            tensor_bytes = num_elements * FLOAT32_BYTES
            tensor = _allocate_tensor(num_elements, device, rank)  # 创建包含 num_elements 个元素的本地张量

            if not _all_ranks_allocated(tensor, device):  # 任一 rank 失败时，所有 rank 都会得到 0
                if rank == 0:
                    records.append(
                        {
                            "status": "oom",
                            "backend": backend,
                            "world_size": world_size,
                            "size_mib": size_mib,
                            "num_elements": num_elements,
                            "tensor_bytes": tensor_bytes,
                            "dtype": "float32",
                        }
                    )
                del tensor
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            assert tensor is not None
            _verify_all_reduce(tensor, world_size)
            local_timings_ms = _measure_all_reduce(
                tensor,
                warmup_steps=warmup_steps,
                measurement_steps=measurement_steps,
            )
            rank_timings_ms = _gather_rank_timings(local_timings_ms, device, world_size)

            if rank == 0:
                records.append(
                    {
                        "status": "ok",
                        "backend": backend,
                        "world_size": world_size,
                        "size_mib": size_mib,
                        "num_elements": num_elements,
                        "tensor_bytes": tensor_bytes,
                        "dtype": "float32",
                        "warmup_steps": warmup_steps,
                        "measurement_steps": measurement_steps,
                        "rank_timings_ms": rank_timings_ms,
                        **summarize_rank_timings(
                            rank_timings_ms,
                            tensor_bytes=tensor_bytes,
                            world_size=world_size,
                        ),
                    }
                )

            del tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if rank == 0:
            Path(output_path).write_text(json.dumps(records, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _validate_environment(backend: str, world_sizes: list[int]) -> None:
    if any(world_size < 2 for world_size in world_sizes):
        raise ValueError("every world size must be at least 2")

    if backend == "nccl":
        if not dist.is_nccl_available() or not torch.cuda.is_available():
            raise RuntimeError("NCCL benchmark requires a CUDA-enabled PyTorch installation")
        required_gpus = max(world_sizes)
        available_gpus = torch.cuda.device_count()
        if available_gpus < required_gpus:
            raise RuntimeError(
                f"requested {required_gpus} GPUs, but only {available_gpus} are available"
            )


def _metadata(backend: str, world_sizes: list[int]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "backend": backend,
    }
    if backend == "nccl":
        metadata.update(
            {
                "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(max(world_sizes))
                ],
            }
        )
    else:
        metadata["processor"] = platform.processor()
    return metadata


def _write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "status",
        "backend",
        "world_size",
        "size_mib",
        "tensor_bytes",
        "dtype",
        "mean_ms",
        "median_ms",
        "p20_ms",
        "p80_ms",
        "min_ms",
        "max_ms",
        "algorithmic_bandwidth_gbps",
        "bus_bandwidth_gbps",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if args.measurement_steps <= 0:
        raise ValueError("measurement_steps must be positive")

    _validate_environment(args.backend, args.world_sizes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    for world_size in args.world_sizes:
        world_output_path = args.output_dir / f"all_reduce_world_size_{world_size}.json"
        print(f"Benchmarking world_size={world_size} with backend={args.backend}", flush=True)
        mp.spawn(
            _worker,
            args=(
                world_size,
                args.backend,
                args.sizes_mib,
                args.warmup_steps,
                args.measurement_steps,
                args.master_addr,
                _find_free_port(),
                str(world_output_path),
            ),
            nprocs=world_size,
            join=True,
        )
        all_records.extend(json.loads(world_output_path.read_text(encoding="utf-8")))

        summary = {
            "benchmark": "single_node_all_reduce",
            "metadata": _metadata(args.backend, args.world_sizes),
            "configuration": {
                "sizes_mib": args.sizes_mib,
                "world_sizes": args.world_sizes,
                "dtype": "float32",
                "warmup_steps": args.warmup_steps,
                "measurement_steps": args.measurement_steps,
                "timing_statistic": "slowest rank per repetition",
                "synchronization": "process barrier before measurements and device synchronization around each collective",
            },
            "results": all_records,
        }
        summary_path = args.output_dir / "all_reduce_results.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_csv(all_records, args.output_dir / "all_reduce_results.csv")

    print(f"Wrote benchmark results to {summary_path}")


if __name__ == "__main__":
    main()
