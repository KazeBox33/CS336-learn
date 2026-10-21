import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.supplement.checkpoint import save_sft_model
from cs336_alignment.supplement.data import PackedSFTDataset, iterate_batches
from cs336_alignment.supplement.sft import create_sft_lr_scheduler, evaluate_sft, run_sft_epoch


@dataclass(frozen=True)
class SFTTrainingConfig:
    epochs: int = 1
    seq_length: int = 512
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    seed: int = 0

    def __post_init__(self):
        for name in ("epochs", "seq_length", "microbatch_size", "gradient_accumulation_steps", "log_interval", "eval_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        for name in ("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm < 0:
            raise ValueError("learning_rate must be positive; weight_decay and max_grad_norm must be non-negative.")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1).")


def run_sft_training(
    config: SFTTrainingConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_path: str | Path,
    val_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Train an already loaded single-device model, validate, log, and export it."""
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"SFT run output already exists: {output_dir}")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_dataset = PackedSFTDataset(tokenizer, train_path, config.seq_length, shuffle=True)
    val_dataset = PackedSFTDataset(tokenizer, val_path, config.seq_length, shuffle=False)
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Training and validation data must each contain at least one packed sequence.")
    train_loader = iterate_batches(train_dataset, config.microbatch_size, shuffle=True)
    val_loader = iterate_batches(val_dataset, config.microbatch_size, shuffle=False)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_updates = config.epochs * updates_per_epoch
    # Keep a decay phase even in tiny smoke tests with only one planned update.
    warmup_updates = min(math.ceil(total_updates * config.warmup_ratio), total_updates - 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = create_sft_lr_scheduler(optimizer, total_updates, warmup_updates)

    parameter = next(model.parameters())
    manifest = {
        "config": asdict(config),
        "train_path": str(Path(train_path).resolve()),
        "val_path": str(Path(val_path).resolve()),
        "model_id": model.config._name_or_path,
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "num_train_sequences": len(train_dataset),
        "num_val_sequences": len(val_dataset),
        "effective_batch_size": config.microbatch_size * config.gradient_accumulation_steps,
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "warmup_updates": warmup_updates,
        "completed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "run.json"
    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)

    global_update = 0
    with (output_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics_file:
        def log(record):
            line = json.dumps(record, allow_nan=False)
            metrics_file.write(line + "\n")
            metrics_file.flush()
            print(line, flush=True)

        val_metrics = evaluate_sft(model, val_loader)
        log({"split": "validation", "update": 0, "epoch": 0, **val_metrics})
        for epoch in range(1, config.epochs + 1):
            for _, metrics in run_sft_epoch(
                model, optimizer, train_loader, config.gradient_accumulation_steps,
                config.max_grad_norm, scheduler=scheduler,
            ):
                global_update += 1
                if global_update % config.log_interval == 0 or global_update == total_updates:
                    log({"split": "train", "update": global_update, "epoch": epoch, **metrics})
                if global_update % config.eval_interval == 0 or global_update == total_updates:
                    val_metrics = evaluate_sft(model, val_loader)
                    log({"split": "validation", "update": global_update, "epoch": epoch, **val_metrics})
    saved_path = save_sft_model(model, tokenizer, output_dir / "final")
    manifest.update(
        completed=True, optimizer_updates=global_update,
        final_val_loss=val_metrics["val_loss"], model_dir=str(saved_path.resolve()),
    )
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
    return manifest
