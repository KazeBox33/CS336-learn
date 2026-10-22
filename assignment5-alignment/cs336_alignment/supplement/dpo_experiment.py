import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.supplement.checkpoint import save_best_dpo_checkpoint
from cs336_alignment.supplement.dpo_training import evaluate_dpo, run_dpo_epoch
from cs336_alignment.supplement.hh import HHPreferenceExample, load_hh_preferences


@dataclass(frozen=True)
class DPOTrainingConfig:
    epochs: int = 1
    gradient_accumulation_steps: int = 64
    validation_size: int = 200
    learning_rate: float = 1e-6
    beta: float = 0.1
    max_grad_norm: float | None = None
    log_interval: int = 1
    eval_interval: int = 20
    seed: int = 0

    def __post_init__(self):
        for name in ("epochs", "gradient_accumulation_steps", "validation_size", "log_interval", "eval_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and non-negative.")
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm < 0):
            raise ValueError("max_grad_norm must be finite and non-negative, or None.")


def split_hh_preferences(
    examples: Sequence[HHPreferenceExample], validation_size: int, seed: int,
) -> tuple[list[HHPreferenceExample], list[HHPreferenceExample]]:
    """Hold out validation_size distinct instructions, keeping their pairs together."""
    instructions = list(dict.fromkeys(example.instruction for example in examples))
    if not isinstance(validation_size, int) or not 0 < validation_size < len(instructions):
        raise ValueError("validation_size must leave at least one training instruction.")
    held_out = set(random.Random(seed).sample(instructions, validation_size))
    train = [example for example in examples if example.instruction not in held_out]
    validation = [example for example in examples if example.instruction in held_out]
    return train, validation


def run_dpo_training(
    config: DPOTrainingConfig,
    lm: PreTrainedModel,
    lm_ref: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    data_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Connect HH splitting, RMSprop updates, validation, logs, and best exports."""
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"DPO run output already exists: {output_dir}")
    policy_parameters = {id(parameter) for parameter in lm.parameters()}
    if any(id(parameter) in policy_parameters for parameter in lm_ref.parameters()):
        raise ValueError("Policy and reference must not share Parameter objects.")
    examples = load_hh_preferences(data_dir)
    train, validation = split_hh_preferences(examples, config.validation_size, config.seed)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    lm_ref.requires_grad_(False)
    lm_ref.eval()
    optimizer = torch.optim.RMSprop(lm.parameters(), lr=config.learning_rate)
    updates_per_epoch = math.ceil(len(train) / config.gradient_accumulation_steps)
    total_updates = config.epochs * updates_per_epoch
    manifest = {
        "config": asdict(config), "data_dir": str(Path(data_dir).resolve()),
        "model_id": lm.config._name_or_path, "reference_model_id": lm_ref.config._name_or_path,
        "policy_device": str(next(lm.parameters()).device),
        "reference_device": str(next(lm_ref.parameters()).device),
        "dtype": str(next(lm.parameters()).dtype),
        "num_train_examples": len(train), "num_val_examples": len(validation),
        "num_train_instructions": len({example.instruction for example in train}),
        "num_val_instructions": len({example.instruction for example in validation}),
        "updates_per_epoch": updates_per_epoch, "total_updates": total_updates,
        "completed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "run.json"

    def persist():
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, allow_nan=False)

    persist()
    global_update = 0
    best = None
    with (output_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics_file:
        def log(record):
            line = json.dumps(record, allow_nan=False)
            metrics_file.write(line + "\n")
            metrics_file.flush()
            print(line, flush=True)

        def validate_and_save(epoch):
            nonlocal best
            metrics = evaluate_dpo(lm, lm_ref, tokenizer, validation, config.beta)
            best = save_best_dpo_checkpoint(
                lm, tokenizer, accuracy=metrics["val_accuracy"], update=global_update,
                output_dir=output_dir / "checkpoints", best=best,
            )
            manifest.update(best_accuracy=best.accuracy, best_update=best.update, best_model_dir=str(best.path.resolve()))
            persist()
            log({"split": "validation", "update": global_update, "epoch": epoch,
                 **metrics, "best_accuracy": best.accuracy, "best_update": best.update})
            return metrics

        val_metrics = validate_and_save(0)
        for epoch in range(1, config.epochs + 1):
            epoch_examples = rng.sample(train, len(train))
            for _, metrics in run_dpo_epoch(
                lm, lm_ref, tokenizer, optimizer, epoch_examples,
                config.gradient_accumulation_steps, config.beta, config.max_grad_norm,
            ):
                global_update += 1
                if global_update % config.log_interval == 0 or global_update == total_updates:
                    log({"split": "train", "update": global_update, "epoch": epoch, **metrics})
                if global_update % config.eval_interval == 0 or global_update == total_updates:
                    val_metrics = validate_and_save(epoch)
    manifest.update(completed=True, optimizer_updates=global_update,
                    final_val_loss=val_metrics["val_loss"], final_val_accuracy=val_metrics["val_accuracy"])
    persist()
    return manifest
