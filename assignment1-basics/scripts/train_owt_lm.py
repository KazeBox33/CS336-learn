from __future__ import annotations

import argparse
from pathlib import Path

from cs336_basics.train import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an OpenWebText Transformer language model.")
    parser.add_argument("--train-data-path", type=Path, default=Path("../data/owt_train_tokens.npy"))
    parser.add_argument("--valid-data-path", type=Path, default=Path("../data/owt_valid_tokens.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("../outputs/owt_lm"))
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--max-iters", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-l2-norm", type=float, default=1.0)
    parser.add_argument("--model-name", default="owt_transformer")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dry-run", action="store_true")
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
        vocab_size=32_000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10_000.0,
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
        model_name=args.model_name,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    train(config)


if __name__ == "__main__":
    main()
