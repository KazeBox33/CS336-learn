from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock

from cs336_systems.checkpointing import (
    checkpoint_layers_in_segments,
    recursive_checkpoint_layers,
    run_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile activation-checkpointing strategies on an XL Transformer block stack."
    )
    parser.add_argument(
        "--strategy",
        choices=("none", "segment", "recursive"),
        default="segment",
    )
    parser.add_argument("--segment-size", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=2560)
    parser.add_argument("--d-ff", type=int, default=10240)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile-block", action="store_true")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def apply_block_stack(
    block: TransformerBlock,
    x: torch.Tensor,
    num_layers: int,
    strategy: str,
    segment_size: int,
) -> torch.Tensor:
    layers = (block,) * num_layers

    if strategy == "none":
        return run_layers(layers, x)
    if strategy == "segment":
        return checkpoint_layers_in_segments(layers, x, segment_size)
    if strategy == "recursive":
        return recursive_checkpoint_layers(layers, x)

    raise ValueError(f"Unknown checkpoint strategy: {strategy}")


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def profile(args: argparse.Namespace) -> dict[str, object]:
    if args.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if args.strategy == "segment" and args.segment_size <= 0:
        raise ValueError("segment_size must be positive")

    positional_encoder = RotaryEmbedding(
        context_length=args.context_length,
        dim=args.d_model // args.num_heads,
    )
    block = TransformerBlock(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        positional_encoder=positional_encoder,
    ).to(args.device)

    if args.compile_block:
        block = torch.compile(block, fullgraph=True)

    x = torch.randn(
        args.batch_size,
        args.context_length,
        args.d_model,
        device=args.device,
        requires_grad=True,
    )

    block.zero_grad(set_to_none=True)
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated()
    else:
        memory_before = None

    synchronize(args.device)
    start = time.perf_counter()

    output = apply_block_stack(
        block=block,
        x=x,
        num_layers=args.num_layers,
        strategy=args.strategy,
        segment_size=args.segment_size,
    )
    output.sum().backward()

    synchronize(args.device)
    elapsed_seconds = time.perf_counter() - start

    peak_memory = (
        torch.cuda.max_memory_allocated()
        if args.device.startswith("cuda")
        else None
    )

    return {
        "strategy": args.strategy,
        "segment_size": args.segment_size if args.strategy == "segment" else None,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_heads": args.num_heads,
        "device": args.device,
        "compile_block": args.compile_block,
        "elapsed_seconds": elapsed_seconds,
        "memory_before_bytes": memory_before,
        "peak_memory_bytes": peak_memory,
        "peak_memory_mib": (
            peak_memory / 1024**2 if peak_memory is not None else None
        ),
    }


def main() -> None:
    args = parse_args()
    result = profile(args)

    print(json.dumps(result, indent=2))

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
