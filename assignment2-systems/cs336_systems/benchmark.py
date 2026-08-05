import argparse
import json
import math
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch

import cs336_basics.model as basics_model
from einops import einsum
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW

MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
    "10B": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a CS336 Transformer training step.")

    parser.add_argument(
        "--model-size",
        choices=MODEL_CONFIGS.keys(),
        default="small",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)

    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward", "full"),
        default="full",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use CUDA BF16 autocast during forward computation",
    )

    parser.add_argument("--output-path", type=Path, default=None)

    parser.add_argument(
        "--annotate-attention",
        action="store_true",
    )

    return parser.parse_args()


def build_model(args: argparse.Namespace) -> BasicsTransformerLM:
    model_config = MODEL_CONFIGS[args.model_size]

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=model_config["d_model"],
        d_ff=model_config["d_ff"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
    )

    return model.to(args.device)


def make_random_batch(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=args.device,
    )
    targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=args.device,
    )
    return inputs, targets


def nvtx_range(message: str, device: str):
    if device.startswith("cuda"):
        return torch.cuda.nvtx.range(message)

    return nullcontext()


def mixed_precision_context(device: str, enabled: bool):
    if not enabled:
        return nullcontext()

    if not device.startswith("cuda"):
        raise ValueError("BF16 mixed precision benchmarking requires a CUDA device")

    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    )


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    device = Q.device.type

    with nvtx_range("scaled dot product attention", device):
        d_k = K.shape[-1]

        with nvtx_range("computing attention scores", device):
            attention_scores = einsum(
                Q,
                K,
                "... query d_k, ... key d_k -> ... query key",
            ) / math.sqrt(d_k)

            if mask is not None:
                attention_scores = torch.where(
                    mask,
                    attention_scores,
                    float("-inf"),
                )

        with nvtx_range("computing softmax", device):
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx_range("final matmul", device):
            output = einsum(
                attention_weights,
                V,
                "... query key, ... key d_v -> ... query d_v",
            )

    return output


def run_step(
    model: BasicsTransformerLM,
    optimizer: AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
) -> None:
    device = inputs.device.type

    if mode != "forward":
        optimizer.zero_grad(set_to_none=True)

    with nvtx_range("forward", device):
        logits = model(inputs)

    if mode == "forward":
        return

    with nvtx_range("loss and backward", device):
        loss = cross_entropy(logits, targets)
        loss.backward()

    if mode == "full":
        with nvtx_range("optimizer", device):
            optimizer.step()


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def measure_steps(
    model: BasicsTransformerLM,
    optimizer: AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    warmup_steps: int,
    measurement_steps: int,
    device: str,
) -> list[float]:
    for _ in range(warmup_steps):
        run_step(model, optimizer, inputs, targets, mode)

    synchronize(device)

    timings = []
    with nvtx_range("benchmark_measurement", device):
        for _ in range(measurement_steps):
            synchronize(device)
            start = time.perf_counter()

            run_step(model, optimizer, inputs, targets, mode)

            synchronize(device)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

    return timings


def summarize_timings(timings: list[float]) -> tuple[float, float]:
    timings_ms = [elapsed * 1000 for elapsed in timings]  # 使用毫秒

    mean_ms = statistics.mean(timings_ms)  # 平均值
    std_ms = statistics.stdev(timings_ms)  # 标准差

    return mean_ms, std_ms


def main() -> None:
    args = parse_args()

    if args.annotate_attention:
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    model = build_model(args)
    model.train()

    optimizer = AdamW(model.parameters())
    inputs, targets = make_random_batch(args)

    timings = measure_steps(
        model=model,
        optimizer=optimizer,
        inputs=inputs,
        targets=targets,
        mode=args.mode,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        device=args.device,
    )

    mean_ms, std_ms = summarize_timings(timings)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())

    result = {
        "model_size": args.model_size,
        "model_config": MODEL_CONFIGS[args.model_size],
        "parameters": num_parameters,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "device": args.device,
        "mode": args.mode,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "timings_ms": [elapsed * 1000 for elapsed in timings],
        "mean_ms": mean_ms,
        "std_ms": std_ms,
    }

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    print(f"model size: {args.model_size}")
    print(f"parameters: {num_parameters:,}")
    print(f"batch size: {args.batch_size}")
    print(f"context length: {args.context_length}")
    print(f"device: {args.device}")
    print(f"mode: {args.mode}")
    print(f"time: {mean_ms:.3f} +/- {std_ms:.3f} ms")


if __name__ == "__main__":
    main()
