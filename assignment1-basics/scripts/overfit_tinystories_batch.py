from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.modules import TransformerLM, cross_entropy
from cs336_basics.optim import AdamW, gradient_clipping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a TransformerLM to one fixed TinyStories batch.")
    parser.add_argument("--data-path", type=Path, default=Path("../data/tinystories_train_tokens.npy"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=384)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-l2-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    dataset = np.load(args.data_path, mmap_mode="r")
    x, y = get_batch(
        dataset,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start = perf_counter()
    model.train()

    with torch.no_grad():
        initial_loss = cross_entropy(model(x), y).item()
    print(f"step 0: loss {initial_loss:.4f}")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_l2_norm)
        optimizer.step()

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(f"step {step}: loss {loss.item():.4f}")

    with torch.no_grad():
        final_loss = cross_entropy(model(x), y).item()

    elapsed_seconds = perf_counter() - start
    print(f"final loss after update: {final_loss:.4f}")
    print(f"elapsed seconds: {elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
