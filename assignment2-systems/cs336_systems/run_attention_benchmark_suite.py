"""Run the ordinary-attention benchmark matrix in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_D_MODELS = [16, 32, 64, 128]
DEFAULT_SEQUENCE_LENGTHS = [256, 1024, 4096, 8192, 16384]
DEFAULT_IMPLEMENTATIONS = ["eager", "compiled"]  # 新增两种实现方式


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PyTorch attention benchmark matrix.")
    parser.add_argument("--d-models", type=int, nargs="+", default=DEFAULT_D_MODELS)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=DEFAULT_SEQUENCE_LENGTHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=100)
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=DEFAULT_IMPLEMENTATIONS,
        default=DEFAULT_IMPLEMENTATIONS,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/attention/pytorch"))
    return parser.parse_args()


def run_configuration(
    args: argparse.Namespace,
    d_model: int,
    sequence_length: int,
    implementation: str, #实现类型
) -> dict[str, Any]:
    output_path = args.output_dir / f"{implementation}_b{args.batch_size}_l{sequence_length}_d{d_model}.json"
    command = [
        sys.executable,
        "-m",
        "cs336_systems.attention_benchmark",
        "--batch-size",
        str(args.batch_size),
        "--sequence-length",
        str(sequence_length),
        "--d-model",
        str(d_model),
        "--device",
        args.device,
        "--warmup-steps",
        str(args.warmup_steps),
        "--measurement-steps",
        str(args.measurement_steps),
        "--output-path",
        str(output_path),
    ]
    if implementation == "compiled":
        command.append("--compile")

    subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for d_model in args.d_models:
        for sequence_length in args.sequence_lengths:
            for implementation in args.implementations:
                print(
                    f"Running implementation={implementation}, d_model={d_model}, sequence_length={sequence_length}",
                    flush=True,
                )
                result = run_configuration(args, d_model, sequence_length, implementation)
                results.append(result)
                print(f"  status={result['status']}", flush=True)

    summary = {
        "batch_size": args.batch_size,
        "device": args.device,
        "d_models": args.d_models,
        "sequence_lengths": args.sequence_lengths,
        "implementations": args.implementations,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
