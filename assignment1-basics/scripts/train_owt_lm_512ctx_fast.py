from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from cs336_basics.train import TrainConfig as RuntimeTrainConfig, train


@dataclass
class ModelConfig:
    model_name: str = "owt_transformer_512ctx_6layer_bf16"
    vocab_size: int = 32_000
    context_length: int = 512
    d_model: int = 768
    num_layers: int = 6
    num_heads: int = 12
    d_ff: int = 2_048
    rope_theta: float = 10_000.0


@dataclass
class TrainingConfig:
    batch_size: int = 48
    num_steps: int = 50_000
    train_data_path: str = "/root/autodl-tmp/cs336/data/owt_train_tokens.npy"
    valid_data_path: str = "/root/autodl-tmp/cs336/data/owt_valid_tokens.npy"
    save_checkpoint_dir: str = "/root/autodl-tmp/cs336/outputs/owt_lm_512ctx_fast_bf16_bs48"

    # Optimizer related parameters
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    max_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_iters: int = 1_000
    max_grad_norm: float = 1.0
    eps: float = 1e-8

    # Efficiency related parameters
    use_bf16: bool = True
    tie_embeddings: bool = False
    compile_model: bool = False
    matmul_precision: str = "high"

    # Logging & checkpointing
    log_path: str = "/root/autodl-tmp/cs336/outputs/owt_lm_512ctx_fast_bf16_bs48/train_log.jsonl"
    eval_interval: int = 500
    eval_iters: int = 50
    save_interval: int = 2_000

    # Others
    device: str = "cuda"


MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a faster 512-context OpenWebText Transformer LM.")
    parser.add_argument("--train-data-path", type=Path, default=Path(TRAINING_CONFIG.train_data_path))
    parser.add_argument("--valid-data-path", type=Path, default=Path(TRAINING_CONFIG.valid_data_path))
    parser.add_argument("--output-dir", type=Path, default=Path(TRAINING_CONFIG.save_checkpoint_dir))
    parser.add_argument("--device", default=TRAINING_CONFIG.device)

    parser.add_argument("--max-iters", type=int, default=TRAINING_CONFIG.num_steps)
    parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG.batch_size)
    parser.add_argument("--eval-interval", type=int, default=TRAINING_CONFIG.eval_interval)
    parser.add_argument("--eval-iters", type=int, default=TRAINING_CONFIG.eval_iters)
    parser.add_argument("--save-interval", type=int, default=TRAINING_CONFIG.save_interval)
    parser.add_argument("--log-path", type=Path, default=Path(TRAINING_CONFIG.log_path))

    parser.add_argument("--lr", type=float, default=TRAINING_CONFIG.max_lr)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=TRAINING_CONFIG.min_lr_ratio)
    parser.add_argument("--warmup-iters", type=int, default=TRAINING_CONFIG.warmup_iters)
    parser.add_argument("--weight-decay", type=float, default=TRAINING_CONFIG.weight_decay)
    parser.add_argument("--beta1", type=float, default=TRAINING_CONFIG.betas[0])
    parser.add_argument("--beta2", type=float, default=TRAINING_CONFIG.betas[1])
    parser.add_argument("--eps", type=float, default=TRAINING_CONFIG.eps)
    parser.add_argument("--max-l2-norm", type=float, default=TRAINING_CONFIG.max_grad_norm)

    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--tie-embeddings", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--matmul-precision", default=TRAINING_CONFIG.matmul_precision)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_lr = args.min_lr if args.min_lr is not None else args.lr * args.min_lr_ratio

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RuntimeTrainConfig(
        train_data_path=str(args.train_data_path),
        valid_data_path=str(args.valid_data_path),
        checkpoint_path=str(output_dir / "checkpoint.pt"),
        vocab_size=MODEL_CONFIG.vocab_size,
        context_length=MODEL_CONFIG.context_length,
        d_model=MODEL_CONFIG.d_model,
        num_layers=MODEL_CONFIG.num_layers,
        num_heads=MODEL_CONFIG.num_heads,
        d_ff=MODEL_CONFIG.d_ff,
        rope_theta=MODEL_CONFIG.rope_theta,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        save_interval=args.save_interval,
        max_learning_rate=args.lr,
        min_learning_rate=min_lr,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.max_iters,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        max_l2_norm=args.max_l2_norm,
        device=args.device,
        log_path=str(args.log_path),
        use_bf16=not args.no_bf16,
        tie_embeddings=args.tie_embeddings,
        compile_model=args.compile_model,
        matmul_precision=args.matmul_precision,
    )

    train(config)


if __name__ == "__main__":
    main()
