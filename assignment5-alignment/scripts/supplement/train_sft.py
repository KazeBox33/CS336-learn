"""Run single-device instruction SFT without Hugging Face Trainer."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.supplement.sft_training import SFTTrainingConfig, run_sft_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="HF model ID or local model/tokenizer directory")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new run directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    defaults = SFTTrainingConfig()
    for field in fields(SFTTrainingConfig):
        default = getattr(defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(default), default=default)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = SFTTrainingConfig(**{field.name: getattr(args, field.name) for field in fields(SFTTrainingConfig)})
    except ValueError as error:
        parser.error(str(error))
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("--output-dir must not already exist")
    for path in (args.train_path, args.val_path):
        if not path.is_file():
            parser.error(f"Data file does not exist: {path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation, local_files_only=args.local_files_only,
    ).to(args.device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    manifest = run_sft_training(config, model, tokenizer, args.train_path, args.val_path, args.output_dir)
    print(json.dumps({"model_dir": manifest["model_dir"], "final_val_loss": manifest["final_val_loss"]}, allow_nan=False))


if __name__ == "__main__":
    main()
