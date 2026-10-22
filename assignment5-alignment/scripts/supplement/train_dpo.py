"""Train DPO from an SFT checkpoint, with a frozen independent reference copy."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.supplement.dpo_experiment import DPOTrainingConfig, run_dpo_training
from cs336_alignment.supplement.hh import HH_FILENAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="SFT checkpoint directory or HF model ID")
    parser.add_argument("--data-dir", type=Path, default=Path("data/hh"))
    parser.add_argument("--output-dir", type=Path, required=True, help="A new run directory")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    parser.add_argument("--local-files-only", action="store_true")
    defaults = DPOTrainingConfig()
    for field in fields(DPOTrainingConfig):
        default = getattr(defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"),
                            type=float if field.name == "max_grad_norm" else type(default), default=default,
                            help="Number of distinct held-out instructions, keeping all their pairs together"
                            if field.name == "validation_size" else None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = DPOTrainingConfig(**{field.name: getattr(args, field.name) for field in fields(DPOTrainingConfig)})
    except ValueError as error:
        parser.error(str(error))
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("--output-dir must not already exist")
    for filename in HH_FILENAMES:
        if not (args.data_dir / filename).is_file():
            parser.error(f"Missing HH data file: {args.data_dir / filename}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    models = []
    for device in (args.policy_device, args.reference_device):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=getattr(torch, args.dtype),
            attn_implementation=args.attn_implementation, local_files_only=args.local_files_only,
        ).to(device)
        model.config.use_cache = False
        models.append(model)
    manifest = run_dpo_training(config, models[0], models[1], tokenizer, args.data_dir, args.output_dir)
    print(json.dumps({key: manifest[key] for key in ("best_model_dir", "best_accuracy", "best_update")}, allow_nan=False))


if __name__ == "__main__":
    main()
