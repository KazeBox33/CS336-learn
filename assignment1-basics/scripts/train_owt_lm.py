from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from cs336_basics.train import TrainConfig, train


@dataclass
class OWTTrainingConfig:
    batch_size: int = 64
    max_iters: int = 20_000
    train_data_path: str = "../data/owt_train_tokens.npy"
    valid_data_path: str = "../data/owt_valid_tokens.npy"
    output_dir: str = "../outputs/owt_lm"

    # Model related parameters
    vocab_size: int = 32_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1_344
    rope_theta: float = 10_000.0

    # Optimizer related parameters
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    max_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_iters: int = 1_000
    max_grad_norm: float = 1.0
    eps: float = 1e-8

    # Logging & checkpointing
    eval_interval: int = 500
    eval_iters: int = 50
    save_interval: int = 2_000

    # Others
    device: str = "cuda"


CONFIG = OWTTrainingConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an OpenWebText Transformer language model.")
    parser.add_argument("--train-data-path", type=Path, default=Path(CONFIG.train_data_path))
    parser.add_argument("--valid-data-path", type=Path, default=Path(CONFIG.valid_data_path))
    parser.add_argument("--output-dir", type=Path, default=Path(CONFIG.output_dir))
    parser.add_argument("--device", default=CONFIG.device)

    parser.add_argument("--max-iters", type=int, default=CONFIG.max_iters)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--eval-interval", type=int, default=CONFIG.eval_interval)
    parser.add_argument("--eval-iters", type=int, default=CONFIG.eval_iters)
    parser.add_argument("--save-interval", type=int, default=CONFIG.save_interval)

    parser.add_argument("--lr", type=float, default=CONFIG.max_lr)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=CONFIG.min_lr_ratio)
    parser.add_argument("--warmup-iters", type=int, default=CONFIG.warmup_iters)
    parser.add_argument("--weight-decay", type=float, default=CONFIG.weight_decay)
    parser.add_argument("--beta1", type=float, default=CONFIG.betas[0])
    parser.add_argument("--beta2", type=float, default=CONFIG.betas[1])
    parser.add_argument("--eps", type=float, default=CONFIG.eps)
    parser.add_argument("--max-l2-norm", type=float, default=CONFIG.max_grad_norm)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warmup_iters = args.warmup_iters
    if warmup_iters is None:
        warmup_iters = max(1, args.max_iters // 100)
    min_lr = args.min_lr if args.min_lr is not None else args.lr * args.min_lr_ratio

    output_dir = args.output_dir
    config = TrainConfig(
        train_data_path=str(args.train_data_path),
        valid_data_path=str(args.valid_data_path),
        checkpoint_path=str(output_dir / "checkpoint.pt"),
        vocab_size=CONFIG.vocab_size,
        context_length=CONFIG.context_length,
        d_model=CONFIG.d_model,
        num_layers=CONFIG.num_layers,
        num_heads=CONFIG.num_heads,
        d_ff=CONFIG.d_ff,
        rope_theta=CONFIG.rope_theta,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        save_interval=args.save_interval,
        max_learning_rate=args.lr,
        min_learning_rate=min_lr,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=args.max_iters,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        max_l2_norm=args.max_l2_norm,
        device=args.device,
        log_path=str(output_dir / "train_log.jsonl"),
    )

    train(config)


if __name__ == "__main__":
    main()
