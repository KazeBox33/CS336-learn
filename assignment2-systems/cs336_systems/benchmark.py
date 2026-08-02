import argparse
import statistics
import time

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
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


def run_step(
    model: BasicsTransformerLM,
    optimizer: AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
) -> None:
    if mode != "forward":
        optimizer.zero_grad(set_to_none=True)

    logits = model(inputs)

    if mode == "forward":
        return

    loss = cross_entropy(logits, targets)
    loss.backward()

    if mode == "full":
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

    print(f"model size: {args.model_size}")
    print(f"parameters: {num_parameters:,}")
    print(f"batch size: {args.batch_size}")
    print(f"context length: {args.context_length}")
    print(f"device: {args.device}")
    print(f"mode: {args.mode}")
    print(f"time: {mean_ms:.3f} +/- {std_ms:.3f} ms")


if __name__ == "__main__":
    main()
