"""Run on/off-policy GRPO on GSM8K, or check the configuration without GPUs."""

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import torch

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.checkpoint import save_grpo_checkpoint
from cs336_alignment.grpo_experiment import (
    GRPOExperimentConfig,
    initialize_training_components,
    load_experiment_data,
    open_rollout_server,
    run_grpo_training_loop,
    write_grpo_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = GRPOExperimentConfig()
    parser.add_argument("--model-id", default=defaults.model_id)
    parser.add_argument("--train-path", type=Path, default=defaults.train_path)
    parser.add_argument("--val-path", type=Path, default=defaults.val_path)
    parser.add_argument("--prompt-name", choices=["r1_zero"], default=defaults.prompt_name)
    for name in (
        "seed", "n_train_examples", "n_val_examples", "num_rollout_steps",
        "rollout_batch_size", "group_size", "gradient_accumulation_steps",
        "sampling_max_tokens", "eval_interval", "rollout_log_interval",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "sampling_temperature", "max_grad_norm"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--train-batch-size", type=int, default=None,
                        help="Responses per optimizer step; defaults to rollout batch size")
    parser.add_argument("--importance-reweighting-method", choices=["none", "noclip", "grpo", "gspo"],
                        default=defaults.importance_reweighting_method)
    parser.add_argument("--cliprange", type=float, default=defaults.cliprange)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new result directory")
    parser.add_argument("--training-gpu", type=int, default=0)
    parser.add_argument("--inference-gpu", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true", help="Check data and settings without loading a model")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = GRPOExperimentConfig(**{
            field.name: getattr(args, field.name) for field in fields(GRPOExperimentConfig)
        })
    except ValueError as error:
        parser.error(str(error))
    if args.validation_batch_size <= 0:
        parser.error("--validation-batch-size must be positive")
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint-interval must be positive")
    if min(args.training_gpu, args.inference_gpu) < 0 or args.training_gpu == args.inference_gpu:
        parser.error("Training and inference must use distinct non-negative GPU indices")
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    data = load_experiment_data(config) # 加载数据
    settings = asdict(config)
    settings.update(train_path=str(config.train_path.resolve()), val_path=str(config.val_path.resolve()))
    manifest = {
        "config": settings,
        "training_device": f"cuda:{args.training_gpu}",
        "inference_gpu": args.inference_gpu,
        "validation_batch_size": args.validation_batch_size,
        "checkpoint_interval": args.checkpoint_interval,
        "n_loaded_train_examples": len(data.train_examples),
        "n_loaded_val_examples": len(data.val_examples),
        "prompts_per_rollout_batch": config.prompts_per_rollout_batch,
        "optimizer_steps_per_rollout": config.optimizer_steps_per_rollout,
        "completed": False,
    }
    if args.dry_run:
        print(json.dumps({**manifest, "dry_run": True}, indent=2, allow_nan=False))
        return
    if not torch.cuda.is_available() or max(args.training_gpu, args.inference_gpu) >= torch.cuda.device_count():
        parser.error("Training requires two selected CUDA GPUs; use --dry-run to check data without GPUs")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "run.json"
    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
    policy, tokenizer, optimizer = initialize_training_components(config, manifest["training_device"])

    def checkpoint_after_step(step, rng):
        if step % args.checkpoint_interval == 0 or step == config.num_rollout_steps:
            path = save_grpo_checkpoint(
                policy, tokenizer, optimizer, step,
                args.output_dir / "checkpoints" / f"step_{step:04d}", rng=rng,
            )
            manifest["last_checkpoint"] = str(path.relative_to(args.output_dir))
            with manifest_path.open("w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2, allow_nan=False)

    with open_rollout_server(config, policy, inference_gpu=args.inference_gpu) as server:
        steps = run_grpo_training_loop(
            config, data, server, policy, tokenizer, optimizer, r1_zero_reward_fn,
            validation_batch_size=args.validation_batch_size,
            rollout_output_dir=args.output_dir / "rollouts",
            on_step=checkpoint_after_step,
        )
        count = write_grpo_metrics(steps, args.output_dir / "metrics.jsonl")
    manifest.update(completed=True, num_metric_records=count)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "num_metric_records": count}))


if __name__ == "__main__":
    main()
