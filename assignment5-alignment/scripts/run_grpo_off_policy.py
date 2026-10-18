"""Print a reproducible off-policy experiment plan; train only with --execute."""

import argparse
import shlex
import subprocess
from dataclasses import fields
from pathlib import Path

from cs336_alignment.grpo_experiment import GRPOExperimentConfig
from scripts.train_grpo import build_parser


def build_commands(
    output_root: Path, seeds: list[int], training_args: list[str], include_on_policy: bool = False,
) -> list[list[str]]:
    output_root = output_root.resolve()
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct random seeds.")
    reserved = {
        "--output-dir", "--seed", "--rollout-batch-size", "--train-batch-size",
        "--gradient-accumulation-steps", "--importance-reweighting-method", "--cliprange", "--dry-run",
    }
    if any(argument.split("=", 1)[0] in reserved for argument in training_args):
        raise ValueError("Batch sizes, seeds, clipping and output paths are controlled by the experiment plan.")
    methods = [("offpolicy_naive", "none", None), ("offpolicy_noclip", "noclip", None),
               ("offpolicy_clip", "grpo", "0.2"), ("offpolicy_gspo", "gspo", "3e-4")]
    if include_on_policy:
        methods.insert(0, ("onpolicy", "none", None))
    commands = []
    for name, method, clip in methods:
        for seed in seeds:
            args = [
                *training_args, "--rollout-batch-size", "256",
                "--train-batch-size", "256" if name == "onpolicy" else "8",
                "--gradient-accumulation-steps", "32" if name == "onpolicy" else "1",
                "--importance-reweighting-method", method, "--seed", str(seed),
                "--output-dir", str(output_root / name / f"seed_{seed}"),
            ]
            if clip is not None:
                args.extend(["--cliprange", clip])
            train_parser = build_parser()
            train_parser.allow_abbrev = False
            parsed = train_parser.parse_args(args)
            GRPOExperimentConfig(**{field.name: getattr(parsed, field.name) for field in fields(GRPOExperimentConfig)})
            if parsed.output_dir.exists():
                raise FileExistsError(f"Experiment output already exists: {parsed.output_dir}")
            commands.append(["uv", "run", "--frozen", "--extra", "gpu", "python", "-m", "scripts.train_grpo", *args])
    return commands


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--include-on-policy", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually launch sequential GPU training runs")
    parser.add_argument("training_args", nargs=argparse.REMAINDER, help="Shared train_grpo options after --")
    args = parser.parse_args(argv)
    training_args = args.training_args
    if training_args[:1] == ["--"]:
        training_args = training_args[1:]
    try:
        commands = build_commands(args.output_root, args.seeds, training_args, args.include_on_policy)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
