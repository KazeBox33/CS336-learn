"""Compare per-parameter and flattened-gradient DDP synchronization."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from cs336_systems.benchmark import MODEL_CONFIGS


IMPLEMENTATIONS = ("naive", "flat")
COMPARABLE_FIELDS = (
    "backend",
    "model_size",
    "parameter_count",
    "world_size",
    "global_batch_size",
    "local_batch_size",
    "context_length",
    "vocab_size",
    "warmup_steps",
    "measurement_steps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare individual and flattened-gradient DDP benchmarks."
    )
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/distributed/flat_ddp_comparison"),
    )
    return parser.parse_args()


def _validate_comparable_results(results: dict[str, dict[str, Any]]) -> None:
    if set(results) != set(IMPLEMENTATIONS):
        raise ValueError(f"results must contain exactly {IMPLEMENTATIONS}")

    for implementation, result in results.items():
        if result.get("ddp_implementation") != implementation:
            raise ValueError(
                f"result labeled {implementation!r} contains "
                f"{result.get('ddp_implementation')!r}"
            )

    baseline = results["naive"]
    for field in COMPARABLE_FIELDS:
        if results["flat"].get(field) != baseline.get(field):
            raise ValueError(f"benchmark results differ in {field!r}")


def build_comparison(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build summary metrics from two benchmark result documents."""
    _validate_comparable_results(results)
    naive = results["naive"]
    flat = results["flat"]

    naive_step_ms = float(naive["step_mean_ms"])
    flat_step_ms = float(flat["step_mean_ms"])
    naive_communication_ms = float(naive["communication_mean_ms"])
    flat_communication_ms = float(flat["communication_mean_ms"])
    if min(
        naive_step_ms,
        flat_step_ms,
        naive_communication_ms,
        flat_communication_ms,
    ) <= 0:
        raise ValueError("mean timing values must be positive")

    return {
        "benchmark": "minimal_ddp_flat_comparison",
        "configuration": {field: naive[field] for field in COMPARABLE_FIELDS},
        "results": results,
        "comparison": {
            "step_speedup": naive_step_ms / flat_step_ms,
            "communication_speedup": naive_communication_ms / flat_communication_ms,
            "step_time_reduction_pct": (naive_step_ms - flat_step_ms)
            / naive_step_ms
            * 100.0,
            "communication_time_reduction_pct": (
                naive_communication_ms - flat_communication_ms
            )
            / naive_communication_ms
            * 100.0,
        },
    }


def _run_benchmark(
    args: argparse.Namespace,
    implementation: str,
    output_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "cs336_systems.naive_ddp_benchmark",
        "--ddp-implementation",
        implementation,
        "--model-size",
        args.model_size,
        "--global-batch-size",
        str(args.global_batch_size),
        "--context-length",
        str(args.context_length),
        "--vocab-size",
        str(args.vocab_size),
        "--world-size",
        str(args.world_size),
        "--backend",
        args.backend,
        "--warmup-steps",
        str(args.warmup_steps),
        "--measurement-steps",
        str(args.measurement_steps),
        "--seed",
        str(args.seed),
        "--master-addr",
        args.master_addr,
        "--output-path",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


def _write_csv(comparison: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "implementation",
                "step_mean_ms",
                "step_std_ms",
                "communication_mean_ms",
                "communication_std_ms",
                "communication_fraction_pct",
            ),
        )
        writer.writeheader()
        for implementation in IMPLEMENTATIONS:
            result = comparison["results"][implementation]
            writer.writerow(
                {
                    "implementation": implementation,
                    "step_mean_ms": result["step_mean_ms"],
                    "step_std_ms": result["step_std_ms"],
                    "communication_mean_ms": result["communication_mean_ms"],
                    "communication_std_ms": result["communication_std_ms"],
                    "communication_fraction_pct": result[
                        "communication_fraction_pct"
                    ],
                }
            )


def format_markdown_report(comparison: dict[str, Any]) -> str:
    """Format the assignment table and commentary from structured results."""
    naive = comparison["results"]["naive"]
    flat = comparison["results"]["flat"]
    metrics = comparison["comparison"]

    def describe_change(value: float) -> str:
        direction = "reduction" if value >= 0 else "increase"
        return f"{abs(value):.2f}% {direction}"

    return "\n".join(
        [
            "# Flat-gradient DDP comparison",
            "",
            "| Implementation | Step mean (ms) | Communication mean (ms) | Communication share |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| Per-parameter | {naive['step_mean_ms']:.3f} | "
                f"{naive['communication_mean_ms']:.3f} | "
                f"{naive['communication_fraction_pct']:.2f}% |"
            ),
            (
                f"| Flat-gradient | {flat['step_mean_ms']:.3f} | "
                f"{flat['communication_mean_ms']:.3f} | "
                f"{flat['communication_fraction_pct']:.2f}% |"
            ),
            "",
            (
                "Flattening gradients into one collective changed mean communication "
                f"time from {naive['communication_mean_ms']:.3f} ms to "
                f"{flat['communication_mean_ms']:.3f} ms, a "
                f"{describe_change(metrics['communication_time_reduction_pct'])}."
            ),
            (
                f"Mean training-step time changed from {naive['step_mean_ms']:.3f} ms "
                f"to {flat['step_mean_ms']:.3f} ms "
                f"({metrics['step_speedup']:.3f}x relative speed)."
            ),
            "",
        ]
    )


def _print_summary(comparison: dict[str, Any]) -> None:
    print("implementation | step (ms) | communication (ms) | communication (%)")
    for implementation in IMPLEMENTATIONS:
        result = comparison["results"][implementation]
        print(
            f"{implementation:>14} | "
            f"{result['step_mean_ms']:>9.3f} | "
            f"{result['communication_mean_ms']:>18.3f} | "
            f"{result['communication_fraction_pct']:>17.2f}"
        )

    metrics = comparison["comparison"]
    print(f"step speedup: {metrics['step_speedup']:.3f}x")
    print(f"communication speedup: {metrics['communication_speedup']:.3f}x")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        implementation: _run_benchmark(
            args,
            implementation,
            args.output_dir / f"{implementation}.json",
        )
        for implementation in IMPLEMENTATIONS
    }
    comparison = build_comparison(results)

    comparison_path = args.output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _write_csv(comparison, args.output_dir / "comparison.csv")
    (args.output_dir / "comparison.md").write_text(
        format_markdown_report(comparison),
        encoding="utf-8",
    )
    _print_summary(comparison)
    print(f"wrote {comparison_path}")


if __name__ == "__main__":
    main()
